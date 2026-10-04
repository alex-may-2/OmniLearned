"""Convert the QONNX DeepSets graph with hls4ml, check C-sim vs qonnx, optionally synthesize.

Usage (from synthesis/; C-sim also runs on Perlmutter, --synth only on rdsrv409):
    source /sdf/group/atlas/sw/conda/etc/profile.d/conda.sh
    conda activate /u1/alexmay/conda/envs/omnilearned-hls
    source /afs/slac/g/reseng/xilinx/2024.1/Vitis_HLS/2024.1/settings64.sh   # only needed for --synth
    python convert.py              # full-quant graph, io_stream + Resource: convert + csim smoke test + real-jet check
    python convert.py --onnx onnx_graphs/qat_top_deepsets_distillnet_fpga_a05_T4_8bit_clean.onnx   # first QAT graph
    python convert.py --synth      # also run Vitis HLS synthesis
    # io_parallel, particles unrolled 16-wide (check csim parity first, then synth with a 1 h cap)
    python convert.py --io-type io_parallel --strategy Latency --pf 16 2>&1 | tee logs/convert_parallel_pf16.txt
    timeout 1h python convert.py --io-type io_parallel --strategy Latency --pf 16 --synth 2>&1 | tee logs/synth_parallel_pf16.txt

Each graph + option set writes its own project, hls_prj/deepsets_<stem>_<io>_<strategy>_rf<N>[_pf<N>][_clk..][_mlf]...
<stem> is --name if given (e.g. ps_d12p2r1m1_n16), else distillnet_8bit[<graph suffix>]. Every full-quant graph has
the same suffix, so give --name for any graph other than r7, or projects of different graphs overwrite each other.

The monkeypatches below work around hls4ml bugs hit by this graph (hls4ml fork qibin2020@1d85133).
"""

import argparse
import copy
import importlib
import os
import re

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
p.add_argument("--onnx", default="onnx_graphs/qat_top_deepsets_distillnet_fpga_a05_T4_8bit_fullQuant_clean.onnx")
p.add_argument("--synth", action="store_true")
p.add_argument("--clock", type=float, default=5.0, help="target clock period in ns")
p.add_argument("--mult-limit-fix", action="store_true", help="io_parallel: conv multiplier limit covers all PF pixels")
p.add_argument("--clone-fanout", action="store_true", help="io_parallel: one copy per reader of each residual skip array")
p.add_argument("--dsp-mult", action="store_true", help="bind all multiplies to DSPs (config_op mul -impl dsp)")
p.add_argument("--dsp-add", action="store_true", help="bind adds/subs to DSPs too (config_op add/sub -impl dsp)")
p.add_argument("--reshape-channels", action="store_true",
               help="io_parallel: ARRAY_RESHAPE the inter-layer arrays (one DATAFLOW channel per array, not per element)")
p.add_argument("--auto-rewind", action="store_true",
               help="io_parallel C/RTL fix: no explicit conv rewind + input capture process (Vitis auto-rewinds every conv)")
p.add_argument("--io-type", default="io_stream", choices=["io_stream", "io_parallel"])
p.add_argument("--reuse-factor", type=int, default=1)
p.add_argument("--strategy", default="Resource", choices=["Resource", "Latency"])
# io_parallel only: particles processed in parallel by the per-particle layers (divisor of 64; 64 = fully unrolled)
p.add_argument("--pf", type=int, default=16)
p.add_argument("--name", help="project stem: hls_prj/deepsets_<name>_<io>_... (e.g. ps_d12p2r1m1_n16)")
args = p.parse_args()

# The original 8-bit graph keeps its old project names; other graphs get a suffix from their file name.
_tag = os.path.basename(args.onnx).removesuffix("_clean.onnx").split("_8bit")[-1]
OUT_DIR = f"hls_prj/deepsets_{args.name or 'distillnet_8bit' + _tag}_{args.io_type}_{args.strategy.lower()}_rf{args.reuse_factor}"
if args.io_type == "io_parallel":
    OUT_DIR += f"_pf{args.pf}"
if args.clock != 5:
    OUT_DIR += f"_clk{args.clock:g}"
if args.mult_limit_fix:
    OUT_DIR += "_mlf"
if args.clone_fanout:
    OUT_DIR += "_clone"
if args.dsp_mult:
    OUT_DIR += "_dsp"
if args.dsp_add:
    OUT_DIR += "_dspadd"
if args.reshape_channels:
    OUT_DIR += "_rsh"
if args.auto_rewind:
    OUT_DIR += "_arw"

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
N_PART = model.get_tensor_shape(model.graph.input[0].name)[1]  # particles per jet (64 for r7)
BATCH = model.get_tensor_shape(model.graph.input[0].name)[0]  # graph batch (64 for the QAT exports)

cfg = hls4ml.utils.config_from_onnx_model(
    model, granularity="name", backend="Vitis", default_precision="fixed<16,6>", default_reuse_factor=args.reuse_factor
)
cfg["Model"]["Strategy"] = args.strategy


def quant_type(node):
    """ap_(u)fixed type equal to a power-of-2-scale, zero-offset, not narrow QONNX Quant."""
    scale = model.get_initializer(node.input[1]).item()
    bits = int(model.get_initializer(node.input[3]).item())
    signed = next((a.i for a in node.attribute if a.name == "signed"), 1)
    rnd = "TRN" if next((a.s for a in node.attribute if a.name == "rounding_mode"), b"ROUND") == b"FLOOR" else "RND_CONV"
    return f"{'' if signed else 'u'}fixed<{bits},{bits + int(np.log2(scale))},{rnd},SAT>"


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
    # Per-particle MatMuls (rank-3 input) become PointwiseConv1D "Dense_MatMul_<i>"; default PF=1 loops over the
    # particles. Set on MatMul_<i>: MatmulConstToDense copies that config over any "Dense_MatMul_<i>" entry.
    for node in model.get_nodes_by_op_type("MatMul"):
        if len(model.get_tensor_shape(node.input[0])) == 3:
            cfg["LayerName"][node.name]["ParallelizationFactor"] = args.pf


def convert(hls_cfg):
    return hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir=OUT_DIR,
        project_name="deepsets",
        backend="Vitis",
        io_type=args.io_type,
        part=PART,
        clock_period=args.clock,
        hls_config=hls_cfg,
    )


# Types hls4ml does not infer, read from a first conversion (no compile):
# - Exact mean pool: GlobalPooling1D is not inferred, and accum / 64 keeps only the accumulator's fractional bits.
#   Give the accumulator 6 more integer and fractional bits and the result 6 more fractional bits.
# - A ReLU not fused with a following Quant (e.g. a Mul by alpha != 1 in between) falls back to the default
#   fixed<16,6> and truncates. ReLU is exact in its input's type.
first = convert(copy.deepcopy(cfg)).graph
pool_in = first["GlobalAveragePool_0"].get_input_variable().type.precision
# Other N: see mini_parallel.py (6 more fractional bits keep the truncated sum / N away from rounding ties).
k = int(np.ceil(np.log2(N_PART)))
extra = 0 if N_PART & (N_PART - 1) == 0 else 6
if isinstance(pool_in, FixedPrecisionType):
    w, i = pool_in.width, pool_in.integer
    cfg["LayerName"]["GlobalAveragePool_0"]["Precision"]["accum"] = f"fixed<{w + 2 * k + extra},{i + k}>"
    cfg["LayerName"]["GlobalAveragePool_0"]["Precision"]["result"] = f"fixed<{w + k},{i}>"
# io_parallel: a full-quant pool -> Flatten -> Quant rounds in the pool itself (exact for the types above). As a
# separate zero-latency Quant it chains with the next alpha + Quant in one cycle and misses 5 ns.
gap = model.get_nodes_by_op_type("GlobalAveragePool")[0]
pool_q = model.find_consumer(model.find_consumer(gap.output[0]).output[0])
if args.io_type == "io_parallel" and is_quant(pool_q):
    cfg["LayerName"]["GlobalAveragePool_0"]["Precision"]["result"] = quant_type(pool_q)
for relu in model.get_nodes_by_op_type("Relu"):
    layer = first.get(relu.name)
    fused = is_quant(model.find_consumer(relu.output[0]))  # a FLOOR Quant fused into the ReLU is TRN too
    if layer is not None and not fused and layer.get_output_variable().type.precision.rounding_mode.name == "TRN":
        in_t = layer.get_input_variable().type.precision
        if isinstance(in_t, FixedPrecisionType):
            cfg["LayerName"][relu.name]["Precision"]["result"] = f"fixed<{in_t.width},{in_t.integer}>"
hls_model = convert(cfg)
hls_model.compile()
if args.mult_limit_fix:  # synthesis-only, see mini_parallel.py: II = n_partitions instead of growing with PF
    f = f"{OUT_DIR}/firmware/nnet_utils/nnet_conv1d_latency.h"
    old = "    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::mult_config::multiplier_limit"
    src = open(f).read()
    assert old in src
    open(f, "w").write(src.replace(old, "    const unsigned mult_limit_all = CONFIG_T::mult_config::multiplier_limit * "
                                   "CONFIG_T::n_pixels;\n    #pragma HLS ALLOCATION operation instances=mul limit=mult_limit_all"))


def clone_fanout(cpp, top):
    """Copied from mini_parallel.py. io_parallel residuals: the skip array of each Add is also read by the DyT branch. A DATAFLOW array with two
    readers is not a channel, so Vitis chains the producer into both consumers (pool 3.0 ns + alpha/Quant 2.8 ns in
    one cycle, misses 5 ns). Give each reader its own copy through a clone process, as hls4ml does for io_stream."""
    src = open(cpp).read()
    lines = src.split("\n")
    decl = {m[1]: (m[0], m[2]) for m in re.findall(r"^\s*(\w+) (layer\w+)\[([^\]]+)\];", src, re.M)}
    alias = dict(re.findall(r"auto& (layer\w+) = (layer\w+);", src))
    calls = [i for i, l in enumerate(lines) if l.strip().startswith("nnet::")]
    args = lambda l: [a.strip() for a in l[l.index(">(") + 2 : l.rindex(");")].split(",")]
    n = 0
    for i in [i for i in calls if lines[i].strip().startswith("nnet::add<")]:
        skip = args(lines[i])[0]
        readers = [j for j in calls if j != i and skip in args(lines[j])[: 1]]
        if not readers:
            continue
        t, size = decl[alias.get(skip, skip)]
        a, b = f"{skip}_cpa", f"{skip}_cpb"
        j = readers[0]
        lines[j] = lines[j].replace(f"({skip},", f"({a},", 1)
        lines[i] = lines[i].replace(f"({skip},", f"({b},", 1)
        lines[j] = (
            f"    {t} {a}[{size}];\n    #pragma HLS ARRAY_PARTITION variable={a} complete dim=0\n"
            f"    {t} {b}[{size}];\n    #pragma HLS ARRAY_PARTITION variable={b} complete dim=0\n"
            f"    ps_clone<{t}, {size}>({skip}, {a}, {b});\n" + lines[j]
        )
        n += 1
    clone = (
        "template <class T, int N> void ps_clone(T src[N], T a[N], T b[N]) {\n    #pragma HLS PIPELINE\n"
        "    for (int i = 0; i < N; i++) {\n        #pragma HLS UNROLL\n        a[i] = src[i];\n        b[i] = src[i];\n    }\n}\n\n"
    )
    out = "\n".join(lines)
    k = out.index(f"void {top}(")
    open(cpp, "w").write(out[:k] + clone + out[k:])
    print(f"[clone] {n} fan-out arrays split in {cpp}")


def auto_rewind(cpp, top):
    """Copied from mini_parallel.py. io_parallel C/RTL fix (2026-10-03, run log D2-D8). hls4ml's explicit `rewind` on the pointwise-conv PartitionLoop
    makes the RTL differ from C-sim (cosim: mini BIS-b 333 of 500 jets wrong; H1g, H2-d16s4 argmax 0.965). Without it,
    Vitis auto-rewinds every conv that reads an internal channel (II unchanged, RTL exact), but not the first conv,
    which reads the top-level port (II n_partitions + depth). So drop the rewind and give the input its own capture
    process. Synthesis only: the arithmetic and C-sim are unchanged."""
    h = os.path.join(os.path.dirname(cpp), "nnet_utils", "nnet_conv1d_latency.h")
    s = open(h).read()
    old = "#pragma HLS PIPELINE II=CONFIG_T::reuse_factor rewind"
    assert s.count(old) == 1, h
    open(h, "w").write(s.replace(old, "#pragma HLS PIPELINE II=CONFIG_T::reuse_factor"))
    s = open(cpp).read()
    size = re.search(r"input_t global_in\[([^\]]+)\]", s)[1]
    first = re.search(r"\n(\s*nnet::\w+<[^\n]*>\()global_in,", s)
    s = s.replace(first[0], f"\n{first[1]}global_in_cp,", 1)
    k = s.index("// hls-fpga-machine-learning insert layers") + len("// hls-fpga-machine-learning insert layers")
    s = s[:k] + (f"\n    input_t global_in_cp[{size}];\n    #pragma HLS ARRAY_PARTITION variable=global_in_cp complete dim=0\n"
                 f"    ps_copy<input_t, {size}>(global_in, global_in_cp);\n") + s[k:]
    j = s.index(f"void {top}(")
    s = s[:j] + ("template <class T, int N> void ps_copy(T src[N], T dst[N]) {\n    #pragma HLS PIPELINE\n"
                 "    for (int i = 0; i < N; i++) {\n        #pragma HLS UNROLL\n        dst[i] = src[i];\n    }\n}\n\n") + s[j:]
    open(cpp, "w").write(s)
    print(f"[auto-rewind] conv rewind dropped, input capture before {first[1].strip()}")


def reshape_channels(cpp):
    """Copied from mini_parallel.py. io_parallel DATAFLOW: each element of a completely partitioned inter-layer array is its own channel (a depth-2
    FIFO), ~2,900 FIFOs and 64k LUT of FIFO control in H1g. ARRAY_RESHAPE packs each array into one wide word, so each
    layer boundary is one channel. Pragmas only: the arithmetic (and C-sim) are unchanged.
    Arrays a pointwise conv reads or writes stay partitioned: the conv indexes them per PF partition, and on a
    reshaped word that indexing costs ~400k LUT per conv (K1 probe, 2026-10-03)."""
    s = open(cpp).read()
    k = s.index("// hls-fpga-machine-learning insert layers")
    conv_io = {a.strip() for c in re.findall(r"pointwise_conv_1d_cl<[^>]*>\(([^,]+,[^,]+),", s) for a in c.split(",")}
    body, n = re.subn(r"#pragma HLS ARRAY_PARTITION variable=(layer\w+) complete dim=0",
                      lambda m: m[0] if m[1] in conv_io else f"#pragma HLS ARRAY_RESHAPE variable={m[1]} complete dim=0", s[k:])
    n -= sum(f"variable={a} " in s[k:] for a in conv_io)
    open(cpp, "w").write(s[:k] + body)
    print(f"[reshape] {n} inter-layer arrays reshaped in {cpp}")


if args.clone_fanout:
    clone_fanout(f"{OUT_DIR}/firmware/deepsets.cpp", "deepsets")
if args.reshape_channels:
    reshape_channels(f"{OUT_DIR}/firmware/deepsets.cpp")
if args.auto_rewind:
    auto_rewind(f"{OUT_DIR}/firmware/deepsets.cpp", "deepsets")
if args.dsp_mult:  # synthesis-only, as mini_parallel.py --dsp-mult
    tcl, clk = f"{OUT_DIR}/build_prj.tcl", "create_clock -period $clock_period -name default"
    src = open(tcl).read()
    assert clk in src
    open(tcl, "w").write(src.replace(clk, clk + "\nconfig_op mul -impl dsp"))
if args.dsp_add:  # synthesis-only, as mini_parallel.py --dsp-add
    tcl, clk = f"{OUT_DIR}/build_prj.tcl", "create_clock -period $clock_period -name default"
    src = open(tcl).read()
    assert clk in src
    open(tcl, "w").write(src.replace(clk, clk + "\nconfig_op add -impl dsp\nconfig_op sub -impl dsp"))


def run_both(x):
    ref = execute_onnx(model, {"global_in": x})[model.graph.output[0].name]
    hls = np.asarray(hls_model.predict(np.ascontiguousarray(x))).reshape(ref.shape)
    return ref, hls


# Smoke test on random input
x = np.random.default_rng(0).standard_normal((BATCH, N_PART, 4)).astype(np.float32)
ref, hls = run_both(x)
print(f"[random] max|dlogit|={np.abs(ref - hls).max():.4g} argmax agree={np.mean(ref.argmax(1) == hls.argmax(1)):.4f}")

# Real jets, full batches only (graph batch dim fixed at 64)
data = np.load("top_test_10k_n64.npz")
n = len(data["x"]) // BATCH * BATCH
X, y = data["x"][:n, :N_PART].astype(np.float32), data["y"][:n]  # leading-pT slots, like --deepsets-fixed-n
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

if args.synth and np.abs(ref - hls).max() > 0:
    raise SystemExit("C-sim is not bit-exact on the jets: no synthesis")
if args.synth:
    hls_model.build(csim=False, synth=True, export=False)
    hls4ml.report.read_vivado_report(OUT_DIR)
