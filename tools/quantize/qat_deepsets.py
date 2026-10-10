"""Quantization-aware training (QAT) for the DeepSets top-tagging student.

Warm-starts from an existing float DeepSets checkpoint (already trained via
KD from a PET2 teacher), replaces its nn.Linear layers with Brevitas
QuantLinear (weight + input-activation quantization at a chosen bit width),
then fine-tunes for a short schedule -- preserving the SAME distillation
recipe (alpha/beta/T against the same precomputed teacher logits) the base
checkpoint was trained with, so quantization is the only new variable.

Deliberately standalone: does NOT edit network.py/train.py/cli.py/utils.py.
It only *imports* from them (DeepSets, train_model, load_data, etc.) and
does the Brevitas wrapping itself, with the `brevitas` import kept inside
this file so importing the shared `omnilearned` package elsewhere (e.g. the
omnilearned-clean env used by ongoing training jobs, which doesn't have
Brevitas installed) is completely unaffected.

QAT is always full-quant: power-of-2 weight/activation scales, Int16 biases,
QuantDynamicTanh (gamma folded, po2 alpha), and by default a fixed tanh input
range (--tanh-in-max 4), a quantized residual stream and pool (--res-bits 10)
and unsigned inputs after ReLU (--relu-uint). Every op between Quant nodes is
then exact in fixed point, so the exported QONNX graph is bit-exact in hls4ml.

Workflow
--------
1. Float KD training (omnilearned train) of a fixed-N ReLU DeepSets student:
   --arch deep-sets --size <size> --act-layer relu --deepsets-fixed-n <N>
   --distill ... --save-tag <float_tag>. Full-quant needs the fixed-N body.
2. QAT (this script): warm-starts from <float_tag>, writes <qat_tag>.
3. QONNX export: qat_deepsets_export_qonnx.py --tag <qat_tag>
   Test-set eval: qat_deepsets_eval.py --tag <qat_tag>
   Both rebuild the model from the QAT checkpoint's arch_config; no shape flags.

Inputs and outputs
------------------
--tag <float_tag>      float checkpoint to start from: <--init-dir>/best_model_<float_tag>.pt
--save-tag <qat_tag>   QAT checkpoint written to <--output-dir>/best_model_<qat_tag>.pt
                       (plus last_model_* and training_*.json)
--init-dir/--output-dir default to /pscratch/sd/t/twamorka/omnilearned/checkpoints/.

The model shape is read from the float checkpoint's arch_config, which train.py
saves. A float checkpoint saved before arch_config existed (e.g.
distill_top_deepsets_distillnet_fpga_a05_T4) stops with an error unless you pass
--size, --deepsets-fixed-n and, if not relu, --act-layer.

The KD recipe defaults (--distill-alpha 0.5 --distill-beta 0.5 --distill-t 4,
teacher fine_tune_top_l) and the schedule defaults (15 epochs, lr 5e-5, wd 0.5,
1000 iterations/epoch) are the ones every full-quant graph so far used.

Examples
--------
Float checkpoint with arch_config (any train.py run since arch_config was added):
    python tools/quantize/qat_deepsets.py \
        --tag ps_d12p2r1m1_n16_e50 \
        --save-tag qat_ps_d12p2r1m1_n16_e50_8bit_fullQuant

Older float checkpoint without arch_config (the distillnet student, N = 64):
    python tools/quantize/qat_deepsets.py \
        --tag distill_top_deepsets_distillnet_fpga_a05_T4 \
        --size distillnet --deepsets-fixed-n 64 \
        --save-tag qat_top_deepsets_distillnet_fpga_a05_T4_8bit_fullQuant

Multi-GPU (one node, 4 GPUs, ~1 h for d12 N=16) from the repo root, inside an
allocation such as
salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 -A m3246:
    srun bash -c "source scripts/export_ddp.sh; python tools/quantize/qat_deepsets.py --tag ... --save-tag ..."
scripts/qat_train_deepsets_distillnet_fpga_8bit.sh is a ready-made launcher for
the distillnet example.

Use the omnilearned-fpga env's python (it has Brevitas); "python" above means that one.
"""

import argparse
import math
import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers.optimization import get_cosine_schedule_with_warmup
from pytorch_optimizer import Lion

from omnilearned.dataloader import load_data
from omnilearned.layers import DynamicTanh
from omnilearned.network import DeepSets, ACT_LAYERS
from omnilearned.train import train_model
from omnilearned.utils import (
    ddp_setup,
    get_checkpoint_name,
    get_deepsets_parameters,
    get_param_groups,
    is_master_node,
    restore_checkpoint,
)

CHECKPOINT_DIR = "/pscratch/sd/t/twamorka/omnilearned/checkpoints/"
DATA_PATH = "/global/cfs/cdirs/m4567/www/"
TEACHER_DIR_L = "/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l"


def dyt_linear_pairs(model):
    """(DynamicTanh, the Linear it feeds) for every DeepSets norm. Uses module
    paths, so call it on the float model before wrapping."""
    body, head = model.body, model.classifier
    pairs = [(body.embed.norm, body.embed.fc2)]
    pairs += [(n, blk.fc1) for n, blk in zip(body.phi_norms, body.phi_blocks)]
    pairs += [(n, blk.fc1) for n, blk in zip(head.rho_norms, head.rho)]
    return pairs


@torch.no_grad()
def fold_dyt_gamma(model):
    """Fold each DynamicTanh's per-channel gamma into the input columns of the
    Linear it feeds: W @ (gamma * t) == (W * gamma) @ t. Removes the gamma Mul
    from the exported graph (one fewer float op for hls4ml, and fewer
    multipliers). Float warm-start only; the full-quant wrap drops gamma."""
    for norm, lin in dyt_linear_pairs(model):
        lin.weight.mul_(norm.weight[None, :])
        norm.weight.fill_(1.0)
    # The embed norm sits after ReLU(fc1), and ReLU is positively homogeneous, so the
    # po2 rounding of its alpha can be absorbed exactly into fc1. Without this,
    # 0.77 -> 1 alone drops accuracy from 0.93 to 0.74. phi/rho norms see the
    # residual stream, so their rounding (7.8, 9.6 -> 8) is left to QAT.
    embed = model.body.embed
    if isinstance(embed.act, nn.ReLU):
        a = embed.norm.alpha
        a_po2 = torch.pow(2.0, torch.round(torch.log2(a)))
        embed.fc1.weight.mul_(a / a_po2)
        embed.fc1.bias.mul_(a / a_po2)
        a.copy_(a_po2)


def floor_round(quant):
    """The Brevitas quantizer with floor instead of round-half-even (QONNX rounding_mode FLOOR, hls4ml AP_TRN: no
    rounding adder or tie logic in the requant)."""
    from brevitas.inject.enum import FloatToIntImplType

    return quant.let(float_to_int_impl_type=FloatToIntImplType.FLOOR)


class QuantDynamicTanh(nn.Module):
    """hls4ml-exact DynamicTanh: tanh(Quant8(alpha_po2 * x)), no gamma.

    alpha is rounded to a power of 2 (straight-through in log2), so the Mul is a
    bit shift. The tanh input is quantized to a power-of-2 grid, so a tanh LUT
    with one entry per code (TableSize = 8 / scale) is exact; the tanh output
    goes straight into the next QuantLinear's input quantizer."""

    def __init__(self, alpha, act_bits, in_max=0.0, floor=False):
        super().__init__()
        import brevitas.nn as qnn
        from brevitas.inject.enum import ScalingImplType
        from brevitas.quant import Int8ActPerTensorFixedPoint

        quant = floor_round(Int8ActPerTensorFixedPoint) if floor else Int8ActPerTensorFixedPoint
        if in_max:
            # Fixed input range [-in_max, in_max) instead of a learned one. in_max = 4 matches the hls4ml tanh
            # table span: 8-bit tanh(4) already rounds to the top output code, so clipping there costs nothing,
            # and the resolution is 1/32 where a learned range can drift to 1/4.
            quant = quant.let(scaling_impl_type=ScalingImplType.CONST, min_val=-in_max, max_val=in_max)

        self.alpha = nn.Parameter(alpha.detach().clone())
        self.act_quant = qnn.QuantIdentity(act_quant=quant, bit_width=act_bits, return_quant_tensor=False)

    def forward(self, x):
        # log/pow, not log2/exp2: ONNX has no Exp2; FoldConstants folds this to a constant
        log_a = torch.log(self.alpha) / math.log(2.0)
        alpha = torch.pow(2.0, log_a + (torch.round(log_a) - log_a).detach())
        return torch.tanh(self.act_quant(alpha * x))


def quant_residual_stream(model, res_bits, floor=False):
    """Quantize the fixed-N DeepSets residual stream: the embed output, every phi residual sum (the last one is the
    pool input) and the pool output, each with a power-of-2 QuantIdentity. Otherwise hls4ml carries these sums at
    full accumulator width (25-bit per particle, a 37-bit pool accumulator). Adds body.res_quant and replaces the
    body's fixed-N forward (network.py is left untouched)."""
    import types

    import brevitas.nn as qnn
    from brevitas.quant import Int8ActPerTensorFixedPoint

    body = model.body
    assert body.fixed_n and not body.pid and not body.add_info and not body.conditional, "fixed-N plain body only"
    act_quant = floor_round(Int8ActPerTensorFixedPoint) if floor else Int8ActPerTensorFixedPoint
    body.res_quant = nn.ModuleList(  # embed out, each phi sum, pool out
        qnn.QuantIdentity(act_quant=act_quant, bit_width=res_bits, return_quant_tensor=False)
        for _ in range(len(body.phi_blocks) + 2)
    )

    def forward_fixed_n(self, x, cond=None, pid=None, add_info=None):
        q = self.res_quant
        if x.shape[1] > self.fixed_n:
            x = x[:, : self.fixed_n, :]
        h = q[0](self.embed(x))
        for i, (norm, phi) in enumerate(zip(self.phi_norms, self.phi_blocks)):
            h = q[i + 1](h + phi(norm(h)))
        z = nn.functional.adaptive_avg_pool1d(h.transpose(1, 2), 1).flatten(1)
        return q[-1](z)

    body._forward_fixed_n = types.MethodType(forward_fixed_n, body)


def build_deepsets(shape):
    """Float DeepSets classifier from a shape dict (an arch_config, or the same keys from the CLI)."""
    return DeepSets(
        input_dim=4, num_classes=2, mode="classifier",
        num_interaction_layers=shape.get("num_interaction_layers", 0),
        interaction_k=shape.get("interaction_k", 0),
        act_layer=ACT_LAYERS[shape["act_layer"]],
        fixed_n=shape["fixed_n"],
        **get_deepsets_parameters(shape["size"]),
    )


def read_arch_config(checkpoint_dir, tag):
    """The arch_config saved with a checkpoint, or None for checkpoints saved before it existed."""
    path = os.path.join(checkpoint_dir, get_checkpoint_name(tag))
    return torch.load(path, map_location="cpu", weights_only=False).get("arch_config")


def load_qat_model(checkpoint_dir, tag):
    """Rebuild a QAT checkpoint from its arch_config and load its weights. Returns (model, arch_config)."""
    cfg = read_arch_config(checkpoint_dir, tag)
    q = (cfg or {}).get("quant") or {}
    if not q.get("full_quant"):
        raise ValueError(f"{tag} is not a full-quant QAT checkpoint (no arch_config with quant.full_quant)")
    model = build_deepsets(cfg)
    # Keys missing from older full-quant checkpoints (r1-r4) mean the option was off.
    wrap_linears_qat(model, q["weight_bits"], tanh_in_max=q.get("tanh_in_max") or 0.0,
                     res_bits=q.get("res_bits") or 0, relu_uint=bool(q.get("relu_uint")),
                     a_bits=q.get("act_bits"), in_bits=q.get("in_bits"), tanh_bits=q.get("tanh_bits"),
                     floor=q.get("round") == "floor")
    restore_checkpoint(model, checkpoint_dir, get_checkpoint_name(tag), 0, is_main_node=True)
    return model, cfg


def wrap_linears_qat(model, bits, tanh_in_max=4.0, res_bits=10, relu_uint=True,
                     a_bits=None, in_bits=None, tanh_bits=None, floor=False):
    """Replace every nn.Linear in `model` with a Brevitas QuantLinear in
    place, copying over the existing (already-trained) weight/bias so this
    is a warm start, not a random re-init. Submodule names/paths are
    preserved exactly (setattr on the same parent, same child_name), so
    everything downstream that walks named_parameters() by path --
    no_weight_decay() matching, get_param_groups()'s "body.embed"/"norm"
    name checks, save_checkpoint's model.module.body.state_dict() -- keeps
    working unmodified.

    Biases are quantized too (Int16Bias, on the accumulator grid) and every
    DynamicTanh becomes a QuantDynamicTanh (gamma dropped -- call
    fold_dyt_gamma first when warm-starting from a float model). Every op
    between Quant nodes is then exact in fixed point.

    tanh_in_max: fixed tanh-input range [-x, x); 0 = learned.
    res_bits: also quantize the residual stream and pool (quant_residual_stream); 0 = off.
    relu_uint: unsigned input quantizer for a Linear fed directly by a ReLU (MLP.fc2), one more bit of resolution.
    bits sets the weights; a_bits (default bits) the Linear inputs, in_bits (default a_bits) the network input (the
    embed fc1 input), tanh_bits (default a_bits) the tanh inputs. floor: activation quantizers round down (weights
    keep round-half-even; they are constants)."""
    import brevitas.nn as qnn
    # Power-of-2 scales: in firmware each scale is a bit shift, not a multiplier.
    from brevitas.quant import (
        Int8ActPerTensorFixedPoint,
        Int8WeightPerTensorFixedPoint,
        Int16Bias,
        Uint8ActPerTensorFixedPoint,
    )

    a_bits = a_bits or bits
    in_bits, tanh_bits = in_bits or a_bits, tanh_bits or a_bits
    if res_bits:
        quant_residual_stream(model, res_bits, floor)

    for module in list(model.modules()):
        for child_name, child in module.named_children():
            if isinstance(child, DynamicTanh):
                setattr(module, child_name, QuantDynamicTanh(child.alpha, tanh_bits, tanh_in_max, floor))

    targets = []
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                targets.append((module, child_name, child))

    for module, child_name, child in targets:
        # MLP: fc1 -> act -> norm -> fc2; fc2 sees the ReLU output only when norm is Identity (not the embed MLP)
        after_relu = (relu_uint and child_name == "fc2" and isinstance(getattr(module, "act", None), nn.ReLU)
                      and isinstance(getattr(module, "norm", None), nn.Identity))
        input_quant = Uint8ActPerTensorFixedPoint if after_relu else Int8ActPerTensorFixedPoint
        qlin = qnn.QuantLinear(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            weight_quant=Int8WeightPerTensorFixedPoint,
            weight_bit_width=bits,
            input_quant=floor_round(input_quant) if floor else input_quant,
            input_bit_width=in_bits if child is model.body.embed.fc1 else a_bits,
            bias_quant=Int16Bias,
            return_quant_tensor=False,
        )
        qlin.weight.data.copy_(child.weight.data)
        if child.bias is not None:
            qlin.bias.data.copy_(child.bias.data)
        setattr(module, child_name, qlin)

    return model


def prune_hidden_linears(model, optimizer, frac, total_steps):
    """Unstructured magnitude pruning, per layer, of every QuantLinear except embed fc1 (4 inputs) and the output
    layer. After each optimizer step the smallest-|w| fraction frac * (1 - (1 - t)^3) of each weight is set to 0, with
    t = step / (0.6 * total_steps) capped at 1 (Zhu & Gupta cubic ramp, then fine-tuning at frac). A zero weight
    quantizes to an exact 0, checkpoints hold the zeros (train_model saves after validation), and the hls4ml
    io_parallel weights are constants, so Vitis drops those multipliers."""
    skip = (model.body.embed.fc1, model.classifier.out)
    layers = [m for m in model.modules() if type(m).__name__ == "QuantLinear" and m not in skip]
    ramp = max(1, int(0.6 * total_steps))  # ponytail: fixed ramp end, add a flag if the schedule needs tuning
    step = 0

    @torch.no_grad()
    def apply(*_):
        nonlocal step
        step += 1
        s = frac * (1 - (1 - min(step / ramp, 1.0)) ** 3)
        for m in layers:
            k = round(s * m.weight.numel())
            if k:
                w = m.weight
                w.mul_(w.abs() > w.abs().flatten().kthvalue(k).values)

    optimizer.register_step_post_hook(apply)
    if is_master_node():
        print(f"Pruning {len(layers)} QuantLinear layers to {frac:.0%} by step {ramp} of {total_steps}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="float checkpoint to start from: <init-dir>/best_model_<tag>.pt")
    ap.add_argument("--save-tag", required=True, help="tag of the QAT checkpoint written to <output-dir>")
    ap.add_argument("--init-dir", default=CHECKPOINT_DIR, help="dir holding the float --tag checkpoint")
    ap.add_argument("--output-dir", default=CHECKPOINT_DIR, help="dir the QAT checkpoint is written to")
    ap.add_argument("--bits", type=int, default=8, help="all bit widths below default to this")
    ap.add_argument("--w-bits", type=int, default=None, help="weights (default --bits)")
    ap.add_argument("--a-bits", type=int, default=None, help="Linear inputs (default --bits)")
    ap.add_argument("--in-bits", type=int, default=None, help="network input = embed fc1 input (default --a-bits)")
    ap.add_argument("--tanh-bits", type=int, default=None, help="DyT tanh inputs (default --a-bits)")
    ap.add_argument("--round", choices=["round", "floor"], default="round",
                    help="activation quantizer rounding: round-half-even (hls4ml RND_CONV) or floor (TRN)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup-epoch", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-path", default=DATA_PATH, help="dataset root holding top/ (e.g. a $PSCRATCH copy)")
    ap.add_argument("--teacher-dir", default=TEACHER_DIR_L)
    ap.add_argument("--teacher-tag", default="fine_tune_top_l")
    ap.add_argument("--distill-alpha", type=float, default=0.5)
    ap.add_argument("--distill-beta", type=float, default=0.5)
    ap.add_argument("--distill-t", type=float, default=4.0)
    # Only for float checkpoints saved before arch_config existed; otherwise the shape comes from the checkpoint.
    ap.add_argument("--size", help="e.g. distillnet, d12p2r1m1 (old float checkpoints only)")
    ap.add_argument("--act-layer", default="relu", choices=sorted(ACT_LAYERS),
                    help="float checkpoint's activation (old float checkpoints only)")
    ap.add_argument("--deepsets-fixed-n", type=int, default=0,
                    help="float checkpoint's fixed-N slot count (old float checkpoints only)")
    ap.add_argument("--tanh-in-max", type=float, default=4.0,
                    help="fixed tanh-input range [-x, x) (4 = hls4ml table span); 0 = learned")
    ap.add_argument("--res-bits", type=int, default=10,
                    help="quantize the residual stream and pool output to this many bits (0 = off)")
    ap.add_argument("--relu-uint", action=argparse.BooleanOptionalAction, default=True,
                    help="unsigned input quantizer for Linears fed by a ReLU (phi/rho fc2)")
    ap.add_argument("--prune", type=float, default=0.0,
                    help="magnitude-prune this fraction of each hidden Linear's weights (prune_hidden_linears; 0 = off)")
    args = ap.parse_args()

    local_rank, rank, size = ddp_setup()

    shape = read_arch_config(args.init_dir, args.tag)
    if shape is None:
        if args.size is None:
            ap.error(f"{args.tag} has no arch_config: pass --size, --act-layer and --deepsets-fixed-n")
        shape = {"size": args.size, "act_layer": args.act_layer, "fixed_n": args.deepsets_fixed_n}
    model = build_deepsets(shape)
    restore_checkpoint(model, args.init_dir, get_checkpoint_name(args.tag), local_rank, is_main_node=is_master_node())
    n_params = sum(p.numel() for p in model.parameters())
    if is_master_node():
        print(f"Warm-started from {args.tag} (size={shape['size']} act={shape['act_layer']} "
              f"fixed_n={shape['fixed_n']}): {n_params:,} params")

    fold_dyt_gamma(model)
    w_bits, a_bits = args.w_bits or args.bits, args.a_bits or args.bits
    in_bits, tanh_bits = args.in_bits or a_bits, args.tanh_bits or a_bits
    wrap_linears_qat(model, w_bits, tanh_in_max=args.tanh_in_max, res_bits=args.res_bits, relu_uint=args.relu_uint,
                     a_bits=a_bits, in_bits=in_bits, tanh_bits=tanh_bits, floor=args.round == "floor")
    # Saved into the checkpoint so the exact quantized network can be rebuilt for
    # FPGA conversion. Quantizer types and bit widths are read back from the
    # wrapped layers, so the record always matches what was trained.
    qlin = next(m for m in model.modules() if type(m).__name__ == "QuantLinear")
    model.arch_config = {
        "arch": "deep-sets",
        "size": shape["size"],
        "dims": get_deepsets_parameters(shape["size"]),
        "input_dim": 4,
        "num_classes": 2,
        "act_layer": shape["act_layer"],
        "fixed_n": shape["fixed_n"],
        "num_interaction_layers": shape.get("num_interaction_layers", 0),
        "interaction_k": shape.get("interaction_k", 0),
        "energy_weighted_pool": False,
        "pid": False,
        "add_info": False,
        "conditional": False,
        "quant": {
            "scope": "every nn.Linear: weight and input",
            "weight_quant": qlin.weight_quant.quant_injector.__name__,
            "act_quant": qlin.input_quant.quant_injector.__name__,
            "weight_bits": int(qlin.weight_quant.bit_width()),
            "act_bits": a_bits,  # Linear inputs; qlin is embed fc1, whose input has in_bits
            "in_bits": int(qlin.input_quant.bit_width()),
            "tanh_bits": tanh_bits,
            "round": args.round,  # activation quantizers; weights always round-half-even
            "full_quant": True,  # Int16Bias biases + QuantDynamicTanh (gamma folded, po2 alpha)
            "tanh_in_max": args.tanh_in_max,  # 0 = learned tanh-input range
            "res_bits": args.res_bits,  # residual stream + pool QuantIdentity bits (0 = off)
            "relu_uint": args.relu_uint,  # unsigned input quant after ReLU
            "prune": args.prune,  # hidden-Linear weight sparsity (record only; the zeros are in the weights)
        },
    }
    if is_master_node():
        n_qlin = sum(1 for m in model.modules() if type(m).__name__ == "QuantLinear")
        print(f"Wrapped {n_qlin} nn.Linear layers as Brevitas QuantLinear (w{w_bits} a{a_bits} in{in_bits} "
              f"tanh{tanh_bits} res{args.res_bits} {args.round})")

    train_loader = load_data(
        "top", dataset_type="train", use_cond=True, path=args.data_path, batch=args.batch,
        num_workers=args.num_workers, rank=rank, size=size, mode="classifier",
        teacher_labels_dir=args.teacher_dir, teacher_tag=args.teacher_tag,
    )
    val_loader = load_data(
        "top", dataset_type="val", use_cond=True, path=args.data_path, batch=args.batch,
        num_workers=args.num_workers, rank=rank, size=size, mode="classifier",
        teacher_labels_dir=args.teacher_dir, teacher_tag=args.teacher_tag,
    )

    param_groups = get_param_groups(model, args.wd, args.lr, lr_factor=1.0, fine_tune=False)
    optimizer = Lion(param_groups, betas=(0.95, 0.98))

    train_steps = args.iterations if args.iterations > 0 else len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(train_steps * args.warmup_epoch),
        num_training_steps=train_steps * args.epochs,
    )
    if args.prune:
        prune_hidden_linears(model, optimizer, args.prune, train_steps * args.epochs)

    kwarg = {}
    if torch.cuda.is_available():
        device = local_rank
        model.to(local_rank)
        kwarg["device_ids"] = [device]
    else:
        model.cpu()
        device = "cpu"
    model = DDP(model, **kwarg)

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        mode="classifier",
        num_epochs=args.epochs,
        device=device,
        output_dir=args.output_dir,
        save_tag=args.save_tag,
        iterations_per_epoch=train_steps,
        ema_model=None,
        distill=True,
        distill_alpha=args.distill_alpha,
        distill_beta=args.distill_beta,
        distill_T=args.distill_t,
    )


if __name__ == "__main__":
    main()
