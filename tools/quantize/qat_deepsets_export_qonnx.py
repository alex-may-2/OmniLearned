"""Export a full-quant QAT DeepSets checkpoint (qat_deepsets.py) to QONNX.

Rebuilds the model from the checkpoint's arch_config (shape and quantizer
settings), restores the weights, then serialises the graph with
``brevitas.export.export_qonnx``. The exported model is run through the QONNX
``cleanup`` + ``InferShapes`` passes and, if a real test batch and
onnxruntime are available, checked for numerical parity against PyTorch.

A thin wrapper is exported instead of ``DeepSets`` directly: DeepSets.forward
returns a dict and takes an unused ``y`` argument, neither of which ONNX
tracing wants. The wrapper's forward is ``x -> classifier(body(x))``, i.e. the
raw (B, num_classes) logits.

Must run with the omnilearned-fpga/env python (has Brevitas / QONNX).

Usage:
    /global/homes/t/twamorka/omnilearned-fpga/env/bin/python \
        tools/quantize/qat_deepsets_export_qonnx.py \
        --tag qat_top_deepsets_distillnet_fpga_a05_T4_8bit_fullQuant
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from qat_deepsets import load_qat_model

CHECKPOINT_DIR = "/pscratch/sd/t/twamorka/omnilearned/checkpoints/"
DATA_PATH = "/global/cfs/cdirs/m4567/www/"
OUT_DIR = "/pscratch/sd/t/twamorka/omnilearned/qonnx/"
N_SLOTS = 150  # production padded constituent count


class LogitWrapper(nn.Module):
    """x (B, N, 4) -> logits (B, num_classes). Drops DeepSets' dict output
    and its unused label argument so the ONNX graph has a single clean
    input and output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.classifier(self.model.body(x))


def real_batch(size, num_workers):
    """One real top-tagging test batch as (X, y) numpy, or None if the
    dataloader is unavailable in this environment."""
    try:
        from omnilearned.dataloader import load_data

        loader = load_data(
            "top", dataset_type="test", use_cond=True, path=DATA_PATH,
            batch=size, num_workers=num_workers, rank=0, size=1,
            mode="classifier", shuffle=False,
        )
        batch = next(iter(loader))
        return batch["X"].float().numpy(), batch["y"].numpy()
    except Exception as exc:  # noqa: BLE001 - purely a best-effort parity aid
        print(f"[warn] could not load a real batch ({exc}); using synthetic input")
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="QAT checkpoint save-tag")
    ap.add_argument("--batch", type=int, default=64,
                    help="batch size for the parity check / export dummy")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR, help="dir holding the QAT --tag checkpoint")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(0)

    model, cfg = load_qat_model(args.checkpoint_dir, args.tag)
    model.cpu().eval()
    fixed_n = cfg["fixed_n"]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.tag} ({cfg['size']}, fixed_n={fixed_n}, quant={cfg['quant']}): {n_params:,} params")

    wrapper = LogitWrapper(model).cpu().eval()

    X_np, y_np = real_batch(args.batch, args.num_workers)
    if X_np is None:
        X_np = np.random.randn(args.batch, N_SLOTS, 4).astype(np.float32)
        X_np[:, N_SLOTS // 3:, 2] = 0.0  # zero log-pT tail -> masked out
    if fixed_n:
        # Feed exactly the fixed-N leading-pT slots so the exported graph
        # carries no in-body Slice (the body's `x.shape[1] > n` guard is then
        # False). Host-side truncation is part of this model's contract.
        X_np = np.ascontiguousarray(X_np[:, :fixed_n, :])
    dummy = torch.from_numpy(X_np).float()
    print(f"Export input shape: {tuple(dummy.shape)}")

    with torch.no_grad():
        torch_logits = wrapper(dummy).numpy()

    raw_path = os.path.join(args.out_dir, f"{args.tag}.onnx")
    from brevitas.export import export_qonnx

    # dynamo=False: the TorchScript exporter keeps the pool as GlobalAveragePool;
    # the dynamo one (torch>=2.9 default) decomposes it back to ReduceMean.
    export_qonnx(wrapper, args=dummy, export_path=raw_path, dynamo=False)
    print(f"Wrote raw QONNX -> {raw_path}")

    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.transformation.fold_constants import FoldConstants
    from qonnx.transformation.infer_shapes import InferShapes
    from qonnx.transformation.general import (
        GiveReadableTensorNames,
        GiveUniqueNodeNames,
        RemoveUnusedTensors,
    )

    m = ModelWrapper(raw_path)
    m = m.transform(InferShapes())
    m = m.transform(FoldConstants())
    m = m.transform(GiveUniqueNodeNames())
    m = m.transform(GiveReadableTensorNames())
    m = m.transform(RemoveUnusedTensors())
    clean_path = os.path.join(args.out_dir, f"{args.tag}_clean.onnx")
    m.save(clean_path)
    print(f"Wrote cleaned QONNX -> {clean_path}")

    op_hist = {}
    for node in m.graph.node:
        op_hist[node.op_type] = op_hist.get(node.op_type, 0) + 1
    print(f"\nGraph: {len(m.graph.node)} nodes")
    for op, cnt in sorted(op_hist.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {cnt:4d}  {op}")

    # Parity via QONNX's own executor -- stock onnxruntime cannot run the
    # QONNX `Quant` custom op.
    try:
        from qonnx.core.onnx_exec import execute_onnx

        in_name = m.graph.input[0].name
        qonnx_logits = execute_onnx(m, {in_name: X_np})[m.graph.output[0].name]
        max_abs = float(np.abs(qonnx_logits - torch_logits).max())
        agree = float((qonnx_logits.argmax(1) == torch_logits.argmax(1)).mean())
        print("\nParity (PyTorch vs QONNX executor on this batch):")
        print(f"  max |Δ logit|      : {max_abs:.3e}")
        print(f"  argmax agreement   : {agree*100:.2f}%")
        if y_np is not None:
            acc_t = float((torch_logits.argmax(1) == y_np).mean())
            acc_q = float((qonnx_logits.argmax(1) == y_np).mean())
            print(f"  batch acc torch/qonnx: {acc_t*100:.2f}% / {acc_q*100:.2f}%")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] QONNX parity check skipped: {exc}")


if __name__ == "__main__":
    main()
