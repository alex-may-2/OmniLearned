"""Convert the QONNX DeepSets graph with hls4ml, check C-sim vs qonnx, optionally synthesize.

Usage (from synthesis/, on rdsrv409):
    source /sdf/group/atlas/sw/conda/etc/profile.d/conda.sh
    conda activate /u1/alexmay/conda/envs/omnilearned-hls
    source /afs/slac/g/reseng/xilinx/2024.1/Vitis_HLS/2024.1/settings64.sh   # only needed for --synth
    python convert.py              # io_stream + Resource: convert + csim smoke test + real-jet check
    python convert.py --synth      # also run Vitis HLS synthesis
    # io_parallel, particles unrolled 16-wide (check csim parity first, then synth with a 1 h cap)
    python convert.py --io-type io_parallel --strategy Latency --pf 16 2>&1 | tee logs/convert_parallel_pf16.txt
    timeout 1h python convert.py --io-type io_parallel --strategy Latency --pf 16 --synth 2>&1 | tee logs/synth_parallel_pf16.txt

Each option set writes its own project, hls_prj/deepsets_distillnet_8bit_<io>_<strategy>_rf<N>[_pf<N>], so runs never
overwrite each other.

The monkeypatches below work around hls4ml bugs hit by this graph (hls4ml fork qibin2020@1d85133).
"""

import argparse
import copy
import importlib
import os

import hls4ml
import numpy as np
from hls4ml.model.attributes import AttributeDict
from hls4ml.model.graph import ModelGraph
from hls4ml.model.layers import ApplyAlpha
from hls4ml.model.optimizer.passes import move_scales
from hls4ml.model.optimizer.passes.bn_fuse import FuseBatchNormalization
from hls4ml.model.optimizer.passes.multi_dense import ReplaceMultidimensionalDenseWithConv
from hls4ml.model.types import FixedPrecisionType, NamedType, TensorVariable, WeightVariable
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.util.cleanup import cleanup_model
from sklearn.metrics import roc_auc_score, roc_curve

PART = "xcvu13p-flga2577-2-e"
BATCH = 64

p = argparse.ArgumentParser()
p.add_argument("--onnx", default="onnx_graphs/qat_top_deepsets_distillnet_fpga_a05_T4_8bit_clean.onnx")
p.add_argument("--synth", action="store_true")
p.add_argument("--io-type", default="io_stream", choices=["io_stream", "io_parallel"])
p.add_argument("--reuse-factor", type=int, default=1)
p.add_argument("--strategy", default="Resource", choices=["Resource", "Latency"])
# io_parallel only: particles processed in parallel by the per-particle layers (divisor of 64; 64 = fully unrolled)
p.add_argument("--pf", type=int, default=16)
args = p.parse_args()

# The original 8-bit graph keeps its old project names; other graphs get a suffix from their file name.
_tag = os.path.basename(args.onnx).removesuffix("_clean.onnx").split("_8bit")[-1]
OUT_DIR = f"hls_prj/deepsets_distillnet_8bit{_tag}_{args.io_type}_{args.strategy.lower()}_rf{args.reuse_factor}"
if args.io_type == "io_parallel":
    OUT_DIR += f"_pf{args.pf}"

# hls4ml bug: Layer._validate_attributes wraps ApplyAlpha's scale/bias_precision in NamedType, and
# ScaleDownAdd rebuilds ApplyAlpha from those attributes, which update_precision rejects. Unwrap it.
_update_precision = WeightVariable.update_precision
WeightVariable.update_precision = lambda self, p: _update_precision(self, p.precision if isinstance(p, NamedType) else p)

# hls4ml bug: after a rank-3 (per-particle) Dense, ApplyAlpha scale/bias are broadcast to [particles, n_out],
# which FuseBatchNormalization multiplies straight into the [n_in, n_out] weight. Collapse to [n_out] when every
# particle row is identical (always true for per-channel constants), otherwise skip the fuse.
_fuse = FuseBatchNormalization.transform


def _fuse_rank3(self, model, node):
    saved = {(w, a): getattr(w, a) for w in (node.weights["scale"], node.weights["bias"]) for a in ("data", "data_unquantized")}
    for (w, attr), d in saved.items():
        d = np.asarray(d)
        if d.ndim > 1:
            rows = d.reshape(-1, d.shape[-1])
            if not (rows == rows[0]).all():
                break
            setattr(w, attr, rows[0])
    else:
        if _fuse(self, model, node):
            return True
    for (w, attr), d in saved.items():
        setattr(w, attr, d)
    return False


FuseBatchNormalization.transform = _fuse_rank3

# hls4ml bug: move_scales passes rebuild ApplyAlpha from the old node's AttributeDict, which still holds the old
# output TensorVariable. It never gets a precision and later fails as "UnspecifiedPrecisionType". Drop it.
_make_node = ModelGraph.make_node


def _make_node_clean(self, kind, name, attributes, inputs, outputs=None, initialize=True):
    if initialize and isinstance(attributes, AttributeDict):
        attributes = {k: v for k, v in attributes.items() if not isinstance(v, TensorVariable)}
    return _make_node(self, kind, name, attributes, inputs, outputs, initialize)


ModelGraph.make_node = _make_node_clean

# hls4ml bug: ReplaceMultidimensionalDenseWithConv (rank-3 Dense -> Conv1D) rebuilds weights from the Dense's original
# weight_data/bias_data attributes, losing any BN/bias fused into it, and drops the weight/bias quantizers.
_dense_to_conv = ReplaceMultidimensionalDenseWithConv.transform


def _dense_to_conv_keep_fused(self, model, node):
    w, b = node.weights["weight"], node.weights["bias"]
    node.set_attr("weight_data", w.data_unquantized)
    node.set_attr("bias_data", b.data_unquantized)
    res = _dense_to_conv(self, model, node)
    conv = model.graph[node.name]
    conv.set_attr("weight_quantizer", w.quantizer)
    conv.set_attr("bias_quantizer", b.quantizer)
    conv.add_weights(quantizer=w.quantizer)
    conv.add_bias(quantizer=b.quantizer)
    return res


ReplaceMultidimensionalDenseWithConv.transform = _dense_to_conv_keep_fused


# hls4ml bug: move_scales passes push an ApplyAlpha below its consumer without checking fan-out. With the DeepSets
# residual (Add_1 feeds both the DynamicTanh branch and the skip Add), the other branch silently loses the bias.
def _single_use_alphas(match):
    def wrapped(self, node):
        for inp in node.inputs:
            src = node.get_input_node(inp)
            if isinstance(src, ApplyAlpha) and sum(len(v) for v in src.get_output_use_map().values()) > 1:
                return False
        return match(self, node)

    return wrapped


for _cls in (move_scales.ScaleDownMatMul, move_scales.ScaleDownAdd, move_scales.BiasDownAdd, move_scales.ScaleDownConv):
    _cls.match = _single_use_alphas(_cls.match)

# hls4ml bug: the ONNX mean pool is Transpose([B,64,32] -> [B,32,64]) + channels-first GlobalAveragePool. hls4ml keeps
# the Transpose as a real data reorder but configures the pool as channels-last (n_in=64 particles, n_filt=32), so it
# averages the wrong elements. The pool config already matches the untransposed tensor: drop the Transpose.
onnx_to_hls = importlib.import_module("hls4ml.converters.onnx_to_hls")  # the package attr is shadowed by a function
_parse_onnx = onnx_to_hls.parse_onnx_model


def _parse_onnx_drop_pool_transpose(onnx_model):
    layers, inputs, outputs = _parse_onnx(onnx_model)
    for t in [layer for layer in layers if layer["class_name"] == "Transpose" and layer["perm"] == [1, 0]]:
        users = [layer for layer in layers if t["outputs"][0] in layer.get("inputs", [])]
        if len(users) == 1 and users[0]["class_name"] == "GlobalAveragePooling1D":
            users[0]["inputs"] = [t["inputs"][0] if i == t["outputs"][0] else i for i in users[0]["inputs"]]
            layers.remove(t)
    return layers, inputs, outputs


onnx_to_hls.parse_onnx_model = _parse_onnx_drop_pool_transpose

model = ModelWrapper(args.onnx)
model = cleanup_model(model).transform(GemmToMatMul())
model = cleanup_model(model)

cfg = hls4ml.utils.config_from_onnx_model(
    model, granularity="name", backend="Vitis", default_precision="fixed<16,6>", default_reuse_factor=args.reuse_factor
)
cfg["Model"]["Strategy"] = args.strategy


def quant_type(node):
    """ap_fixed type equal to a power-of-2-scale, zero-offset QONNX Quant (signed, not narrow)."""
    scale = model.get_initializer(node.input[1]).item()
    bits = int(model.get_initializer(node.input[3]).item())
    return f"fixed<{bits},{bits + int(np.log2(scale))},RND_CONV,SAT>"


def is_quant(node):
    return node is not None and node.op_type == "Quant"


for name, layer_cfg in cfg["LayerName"].items():
    layer_cfg["Strategy"] = args.strategy
for t in model.get_nodes_by_op_type("Tanh"):
    layer_cfg = cfg["LayerName"][t.name]
    pre, post = model.find_producer(t.input[0]), model.find_consumer(t.output[0])
    if is_quant(pre) and is_quant(post):
        # Full-quant graph: the input is on a 2^-k grid and the output is re-quantized, so a LUT with one entry
        # per input code (the table spans [-4, 4)) holding values already rounded to the output type is exact.
        # Inputs beyond +-4 clamp to the end entries, which round to the same 8-bit output as the float tanh.
        layer_cfg["TableSize"] = int(round(8 / model.get_initializer(pre.input[1]).item()))
        layer_cfg["table_t"] = quant_type(post)
        layer_cfg["Precision"]["result"] = quant_type(post)
    else:
        # DynamicTanh: alpha ~8-10 amplifies LUT error of the previous tanh, so use a finer table / output
        # (table type must be set via "table_t"; Activation ignores Precision["table"] and keeps fixed<18,8>)
        layer_cfg["TableSize"] = 4096
        layer_cfg["table_t"] = "fixed<18,2>"
        layer_cfg["Precision"]["result"] = "fixed<16,2>"
# The input feeds a Quant directly: use its type as the input type, so the host->fixed conversion is that Quant
# (RND_CONV = round-half-even, like qonnx) instead of a fixed<16,6> truncation followed by a re-rounding.
in_name = model.graph.input[0].name
if is_quant(model.find_consumer(in_name)):
    cfg["LayerName"][in_name]["Precision"]["result"] = quant_type(model.find_consumer(in_name))
# Sum over 64 particles overflows the default fixed<16,6> accumulator (wraps by 64 -> mean off by exactly 1.0)
cfg["LayerName"]["GlobalAveragePool_0"]["Precision"]["accum"] = "fixed<32,19>"
if args.io_type == "io_parallel":
    # Per-particle MatMul_0..3 become PointwiseConv1D "Dense_MatMul_<i>"; default PF=1 loops over the 64 particles.
    # Set on MatMul_<i>: MatmulConstToDense copies that config over any "Dense_MatMul_<i>" entry.
    for i in range(4):
        cfg["LayerName"][f"MatMul_{i}"]["ParallelizationFactor"] = args.pf


def convert(hls_cfg):
    return hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir=OUT_DIR,
        project_name="deepsets",
        backend="Vitis",
        io_type=args.io_type,
        part=PART,
        hls_config=hls_cfg,
    )


# Exact mean pool: hls4ml does not infer GlobalPooling1D types, and accum / 64 keeps only the accumulator's
# fractional bits. Convert once (no compile) to read the pooled tensor's inferred type, then give the
# accumulator 6 more integer and fractional bits and the result 6 more fractional bits.
pool_in = convert(copy.deepcopy(cfg)).graph["GlobalAveragePool_0"].get_input_variable().type.precision
if isinstance(pool_in, FixedPrecisionType):
    w, i = pool_in.width, pool_in.integer
    cfg["LayerName"]["GlobalAveragePool_0"]["Precision"]["accum"] = f"fixed<{w + 12},{i + 6}>"
    cfg["LayerName"]["GlobalAveragePool_0"]["Precision"]["result"] = f"fixed<{w + 6},{i}>"
hls_model = convert(cfg)
hls_model.compile()


def run_both(x):
    ref = execute_onnx(model, {"global_in": x})[model.graph.output[0].name]
    hls = np.asarray(hls_model.predict(np.ascontiguousarray(x))).reshape(ref.shape)
    return ref, hls


# Smoke test on random input
x = np.random.default_rng(0).standard_normal((BATCH, 64, 4)).astype(np.float32)
ref, hls = run_both(x)
print(f"[random] max|dlogit|={np.abs(ref - hls).max():.4g} argmax agree={np.mean(ref.argmax(1) == hls.argmax(1)):.4f}")

# Real jets, full batches only (graph batch dim fixed at 64)
data = np.load("top_test_10k_n64.npz")
n = len(data["x"]) // BATCH * BATCH
X, y = data["x"][:n].astype(np.float32), data["y"][:n]
refs, hlss = zip(*(run_both(X[i : i + BATCH]) for i in range(0, n, BATCH)))
ref, hls = np.concatenate(refs), np.concatenate(hlss)
print(f"[jets n={n}] max|dlogit|={np.abs(ref - hls).max():.4g} argmax agree={np.mean(ref.argmax(1) == hls.argmax(1)):.4f}")


def metrics(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    score = e[:, 1] / e.sum(1)
    fpr, tpr, _ = roc_curve(y, score)
    return np.mean(logits.argmax(1) == y), roc_auc_score(y, score), 1 / np.interp(0.5, tpr, fpr)


for name, logits in (("qonnx", ref), ("hls", hls)):
    acc, auc, rej = metrics(logits)
    print(f"[{name}] acc={acc:.4f} AUC={auc:.4f} 1/eB@eS=0.5={rej:.1f}")

if args.synth:
    hls_model.build(csim=False, synth=True, export=False)
    hls4ml.report.read_vivado_report(OUT_DIR)
