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

Must be run with the omnilearned-fpga/env python (has Brevitas/QONNX/hls4ml;
omnilearned-clean/env does not).

Usage:
    /global/homes/t/twamorka/omnilearned-fpga/env/bin/python qat_deepsets.py \
        --tag distill_top_deepsets_distillnet_scratch_a05_T4 \
        --size distillnet --bits 8 \
        --save-tag qat_top_deepsets_distillnet_a05_T4_8bit \
        --epochs 15 --lr 5e-5

For a checkpoint trained with the optional message-passing block, pass the
SAME --num-interaction-layers / --interaction-k the float run used so the
architecture (and its state_dict keys) match before restore, e.g.:
    ... --tag distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64 \
        --num-interaction-layers 1 --interaction-k 64
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


class QuantDynamicTanh(nn.Module):
    """hls4ml-exact DynamicTanh: tanh(Quant8(alpha_po2 * x)), no gamma.

    alpha is rounded to a power of 2 (straight-through in log2), so the Mul is a
    bit shift. The tanh input is quantized to a power-of-2 grid, so a tanh LUT
    with one entry per code (TableSize = 8 / scale) is exact; the tanh output
    goes straight into the next QuantLinear's input quantizer."""

    def __init__(self, alpha, act_bits):
        super().__init__()
        import brevitas.nn as qnn
        from brevitas.quant import Int8ActPerTensorFixedPoint

        self.alpha = nn.Parameter(alpha.detach().clone())
        self.act_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=act_bits, return_quant_tensor=False
        )

    def forward(self, x):
        # log/pow, not log2/exp2: ONNX has no Exp2; FoldConstants folds this to a constant
        log_a = torch.log(self.alpha) / math.log(2.0)
        alpha = torch.pow(2.0, log_a + (torch.round(log_a) - log_a).detach())
        return torch.tanh(self.act_quant(alpha * x))


def wrap_linears_qat(model, weight_bits, act_bits, full_quant=False):
    """Replace every nn.Linear in `model` with a Brevitas QuantLinear in
    place, copying over the existing (already-trained) weight/bias so this
    is a warm start, not a random re-init. Submodule names/paths are
    preserved exactly (setattr on the same parent, same child_name), so
    everything downstream that walks named_parameters() by path --
    no_weight_decay() matching, get_param_groups()'s "body.embed"/"norm"
    name checks, save_checkpoint's model.module.body.state_dict() -- keeps
    working unmodified.

    full_quant: also quantize biases (Int16Bias, on the accumulator grid) and
    swap every DynamicTanh for QuantDynamicTanh (gamma dropped -- call
    fold_dyt_gamma first when warm-starting from a float model). Every op
    between Quant nodes is then exact in fixed point."""
    import brevitas.nn as qnn
    # Power-of-2 scales: in firmware each scale is a bit shift, not a multiplier.
    from brevitas.quant import Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint, Int16Bias

    if full_quant:
        for module in list(model.modules()):
            for child_name, child in module.named_children():
                if isinstance(child, DynamicTanh):
                    setattr(module, child_name, QuantDynamicTanh(child.alpha, act_bits))

    targets = []
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                targets.append((module, child_name, child))

    for module, child_name, child in targets:
        qlin = qnn.QuantLinear(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            weight_quant=Int8WeightPerTensorFixedPoint,
            weight_bit_width=weight_bits,
            input_quant=Int8ActPerTensorFixedPoint,
            input_bit_width=act_bits,
            bias_quant=Int16Bias if full_quant else None,
            return_quant_tensor=False,
        )
        qlin.weight.data.copy_(child.weight.data)
        if child.bias is not None:
            qlin.bias.data.copy_(child.bias.data)
        setattr(module, child_name, qlin)

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="base float checkpoint tag to warm-start from")
    ap.add_argument("--size", required=True, help="e.g. small, distillnet, tiny, micro, nano")
    ap.add_argument("--save-tag", required=True, help="new tag for the QAT checkpoint")
    ap.add_argument("--init-dir", default=CHECKPOINT_DIR, help="dir holding the float --tag checkpoint")
    ap.add_argument("--output-dir", default=CHECKPOINT_DIR, help="dir the QAT checkpoint is written to")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup-epoch", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--teacher-dir", default=TEACHER_DIR_L)
    ap.add_argument("--teacher-tag", default="fine_tune_top_l")
    ap.add_argument("--distill-alpha", type=float, default=0.5)
    ap.add_argument("--distill-beta", type=float, default=0.5)
    ap.add_argument("--distill-t", type=float, default=4.0)
    ap.add_argument("--num-interaction-layers", type=int, default=0,
                    help="must match the float checkpoint's message-passing depth")
    ap.add_argument("--interaction-k", type=int, default=0,
                    help="must match the float checkpoint's leading-pT constituent cap")
    ap.add_argument("--act-layer", default="gelu", choices=sorted(ACT_LAYERS),
                    help="must match the float checkpoint's activation")
    ap.add_argument("--deepsets-fixed-n", type=int, default=0,
                    help="must match the float checkpoint's fixed-N/no-mask body (0 = masked-mean)")
    ap.add_argument("--full-quant", action="store_true",
                    help="also quantize biases and DynamicTanh (gamma folded, po2 alpha) for bit-exact hls4ml")
    args = ap.parse_args()

    local_rank, rank, size = ddp_setup()

    ds_params = get_deepsets_parameters(args.size)
    model = DeepSets(
        input_dim=4, num_classes=2, mode="classifier",
        num_interaction_layers=args.num_interaction_layers,
        interaction_k=args.interaction_k,
        act_layer=ACT_LAYERS[args.act_layer],
        fixed_n=args.deepsets_fixed_n,
        **ds_params,
    )
    restore_checkpoint(
        model, args.init_dir, get_checkpoint_name(args.tag), local_rank, is_main_node=is_master_node()
    )
    n_params = sum(p.numel() for p in model.parameters())
    if is_master_node():
        print(f"Warm-started from {args.tag} ({args.size}): {n_params:,} params")

    if args.full_quant:
        fold_dyt_gamma(model)
    wrap_linears_qat(model, weight_bits=args.bits, act_bits=args.bits, full_quant=args.full_quant)
    # Saved into the checkpoint so the exact quantized network can be rebuilt for
    # FPGA conversion. Quantizer types and bit widths are read back from the
    # wrapped layers, so the record always matches what was trained.
    qlin = next(m for m in model.modules() if type(m).__name__ == "QuantLinear")
    model.arch_config = {
        "arch": "deep-sets",
        "size": args.size,
        "dims": ds_params,
        "input_dim": 4,
        "num_classes": 2,
        "act_layer": args.act_layer,
        "fixed_n": args.deepsets_fixed_n,  # 0 = masked mean over valid particles
        "num_interaction_layers": args.num_interaction_layers,
        "interaction_k": args.interaction_k,
        "energy_weighted_pool": False,
        "pid": False,
        "add_info": False,
        "conditional": False,
        "quant": {
            "scope": "every nn.Linear: weight and input",
            "weight_quant": qlin.weight_quant.quant_injector.__name__,
            "act_quant": qlin.input_quant.quant_injector.__name__,
            "weight_bits": int(qlin.weight_quant.bit_width()),
            "act_bits": int(qlin.input_quant.bit_width()),
            "full_quant": args.full_quant,  # Int16Bias biases + QuantDynamicTanh (gamma folded, po2 alpha)
        },
    }
    if is_master_node():
        n_qlin = sum(1 for m in model.modules() if type(m).__name__ == "QuantLinear")
        print(f"Wrapped {n_qlin} nn.Linear layers as Brevitas QuantLinear ({args.bits}-bit)")

    train_loader = load_data(
        "top", dataset_type="train", use_cond=True, path=DATA_PATH, batch=args.batch,
        num_workers=args.num_workers, rank=rank, size=size, mode="classifier",
        teacher_labels_dir=args.teacher_dir, teacher_tag=args.teacher_tag,
    )
    val_loader = load_data(
        "top", dataset_type="val", use_cond=True, path=DATA_PATH, batch=args.batch,
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
