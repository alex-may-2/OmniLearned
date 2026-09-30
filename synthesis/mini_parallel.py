"""Mini DeepSets with random weights: how small must the model be for a fully parallel (io_parallel) hls4ml design?

Three graph flavours, all with 8-bit power-of-2 Quant nodes like the Brevitas QAT export (random weights, so only
the C-sim parity and the synthesis numbers mean anything, not the physics):

default       ReLU only, no biases, no tanh (first io_parallel scan):
                  Quant(in) -> [MatMul(Quant W) -> Relu -> Quant] per phi layer (per particle) -> mean pool
                  -> [MatMul -> Relu -> Quant] per rho layer -> MatMul -> 2 logits
--full-quant  same --phi/--rho widths, with the layer elements of the full-quant QAT model (qat_deepsets.py
              --full-quant --tanh-in-max 4): every Linear gets an Int16 bias (16-bit Quant, scale s_in * s_w) and
              every hidden Relu is followed by the quantized DynamicTanh
                  Mul(po2 alpha) -> Quant(8-bit, 1/32, [-4, 4)) -> Tanh -> Quant(8-bit, 1/128)
--distillnet  the real student topology with full-quant elements (--dim 32 --ratio 2 = the trained model):
                  embed  4 -> H Relu DyT -> D                       (H = ratio * D)
                  phi    h + fc2(Quant(Relu(fc1(DyT(h)))))           x --phi-blocks, per particle
                  mean pool, rho blocks (same, post pool) x --rho-blocks, Quant -> Linear -> 2 logits

Usage (from synthesis/, on rdsrv409, same env as convert.py):
    python mini_parallel.py --n 16                  # convert + C-sim parity, PF = n (fully parallel)
    timeout 1h python mini_parallel.py --n 16 --synth 2>&1 | tee logs/mini_n16.txt
    python mini_parallel.py --full-quant --n 16 --phi 32,16 --rho 16 --pf 8
    python mini_parallel.py --distillnet --n 32 --pf 2

Writes hls_prj/mini[_fq|_ds<..>]_n<N>_..._pf<PF>/ (the ONNX graph is saved there as model.onnx). Independent of
convert.py; the hls4ml patches and the full-quant type settings it needs are copied from there.
"""

import argparse
import copy
import importlib
import os

import hls4ml
import numpy as np
import onnx
from hls4ml.model.attributes import AttributeDict
from hls4ml.model.graph import ModelGraph
from hls4ml.model.layers import ApplyAlpha
from hls4ml.model.optimizer.passes import move_scales
from hls4ml.model.optimizer.passes.bn_fuse import FuseBatchNormalization
from hls4ml.model.optimizer.passes.multi_dense import ReplaceMultidimensionalDenseWithConv
from hls4ml.model.types import FixedPrecisionType, NamedType, TensorVariable, WeightVariable
from onnx import TensorProto, helper, numpy_helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.util.cleanup import cleanup_model

PART = "xcvu13p-flga2577-2-e"
BATCH = 8
ACT_SCALE = 2.0**-4  # 8-bit activations cover [-8, 8) signed / [0, 16) unsigned
TANH_IN_SCALE = 2.0**-5  # --tanh-in-max 4: [-4, 4), one tanh table entry per code (256)
TANH_OUT_SCALE = 2.0**-7  # fixed<8,1>: [-1, 1)

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=16, help="particles per jet")
p.add_argument("--phi", default="64,32", help="per-particle layer widths (default and --full-quant)")
p.add_argument("--rho", default="32", help="post-pool hidden widths, then 2 logits (default and --full-quant)")
p.add_argument("--full-quant", action="store_true", help="Int16 biases + quantized DynamicTanh after hidden ReLUs")
p.add_argument("--distillnet", action="store_true", help="real student topology, full-quant elements")
p.add_argument("--dim", type=int, default=32, help="--distillnet: base_dim")
p.add_argument("--ratio", type=int, default=2, help="--distillnet: mlp_ratio")
p.add_argument("--phi-blocks", type=int, default=1, help="--distillnet: residual phi blocks after the embed")
p.add_argument("--rho-blocks", type=int, default=1, help="--distillnet: residual rho blocks")
p.add_argument("--pf", type=int, default=None, help="ParallelizationFactor of the per-particle layers (default: n)")
p.add_argument("--synth", action="store_true")
args = p.parse_args()
fq = args.full_quant or args.distillnet
phi = [int(d) for d in args.phi.split(",")]
rho = [int(d) for d in args.rho.split(",")] if args.rho else []
pf = args.pf or args.n
if args.distillnet:
    OUT_DIR = f"hls_prj/mini_ds_d{args.dim}r{args.ratio}_phi{args.phi_blocks}_rho{args.rho_blocks}_n{args.n}_pf{pf}"
else:
    OUT_DIR = f"hls_prj/mini{'_fq' if fq else ''}_n{args.n}_phi{'-'.join(map(str, phi))}_rho{'-'.join(map(str, rho))}_pf{pf}"

# --- hls4ml patches copied from convert.py (see the comments there) ---
_update_precision = WeightVariable.update_precision
WeightVariable.update_precision = lambda self, p: _update_precision(self, p.precision if isinstance(p, NamedType) else p)

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

_make_node = ModelGraph.make_node


def _make_node_clean(self, kind, name, attributes, inputs, outputs=None, initialize=True):
    if initialize and isinstance(attributes, AttributeDict):
        attributes = {k: v for k, v in attributes.items() if not isinstance(v, TensorVariable)}
    return _make_node(self, kind, name, attributes, inputs, outputs, initialize)


ModelGraph.make_node = _make_node_clean

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

onnx_to_hls = importlib.import_module("hls4ml.converters.onnx_to_hls")
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


# --- build the QONNX graph ---
rng = np.random.default_rng(0)
nodes, inits, macs = [], [], []  # macs: d_in * d_out per MatMul, in graph order


def const(name, value):
    inits.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), name))
    return name


def count(op):
    return sum(n.op_type == op for n in nodes)


def quant(x, scale, signed, narrow=0, bits=8):
    i = count("Quant")
    out = f"Quant_{i}_out"
    nodes.append(
        helper.make_node(
            "Quant",
            [x, const(f"Quant_{i}_s", scale), const(f"Quant_{i}_z", 0.0), const(f"Quant_{i}_b", float(bits))],
            [out],
            name=f"Quant_{i}",
            domain="qonnx.custom_op.general",
            signed=signed,
            narrow=narrow,
            rounding_mode="ROUND",
        )
    )
    return out


def matmul(x, d_in, d_out, out=None, in_scale=None):
    """MatMul with an 8-bit weight Quant; with in_scale (full-quant) also an Int16 bias on the accumulator grid."""
    i = count("MatMul")
    w = rng.normal(0, 1 / np.sqrt(d_in), (d_in, d_out))
    w_scale = 2.0 ** np.ceil(np.log2(np.abs(w).max() / 127))
    macs.append(d_in * d_out)
    mm_out = f"MatMul_{i}_out" if in_scale else out or f"MatMul_{i}_out"
    nodes.append(helper.make_node("MatMul", [x, quant(const(f"W_{i}", w), w_scale, 1, narrow=1)], [mm_out], name=f"MatMul_{i}"))
    if not in_scale:
        return mm_out
    b = quant(const(f"B_{i}", rng.normal(0, 0.1, d_out)), in_scale * w_scale, 1, bits=16)
    out = out or f"Bias_{i}_out"
    nodes.append(helper.make_node("Add", [mm_out, b], [out], name=f"Bias_{i}"))
    return out


def relu(x):
    i = count("Relu")
    nodes.append(helper.make_node("Relu", [x], [f"Relu_{i}_out"], name=f"Relu_{i}"))
    return f"Relu_{i}_out"


def dyt(x, alpha):
    """Full-quant DynamicTanh: tanh(Quant8(alpha_po2 * x)), output re-quantized by the next Linear's input Quant."""
    i = count("Mul")
    nodes.append(helper.make_node("Mul", [x, const(f"alpha_{i}", alpha)], [f"Mul_{i}_out"], name=f"Mul_{i}"))
    t = quant(f"Mul_{i}_out", TANH_IN_SCALE, 1)
    nodes.append(helper.make_node("Tanh", [t], [f"Tanh_{i}_out"], name=f"Tanh_{i}"))
    return quant(f"Tanh_{i}_out", TANH_OUT_SCALE, 1)


def pool(x):
    nodes.extend(
        [
            helper.make_node("Transpose", [x], ["Transpose_0_out"], name="Transpose_0", perm=[0, 2, 1]),
            helper.make_node("GlobalAveragePool", ["Transpose_0_out"], ["Pool_out"], name="GlobalAveragePool_0"),
            helper.make_node("Flatten", ["Pool_out"], ["Flatten_out"], name="Flatten_0", axis=1),
        ]
    )
    return "Flatten_out"


def residual(h, d, hidden, alpha):
    """Pre-norm residual block: h + fc2(Quant(Relu(fc1(DyT(h)))))."""
    t = matmul(dyt(h, alpha), d, hidden, in_scale=TANH_OUT_SCALE)
    t = matmul(quant(relu(t), ACT_SCALE, 1), hidden, d, in_scale=ACT_SCALE)
    i = sum(n.name.startswith("Res_") for n in nodes)
    nodes.append(helper.make_node("Add", [h, t], [f"Res_{i}_out"], name=f"Res_{i}"))
    return f"Res_{i}_out"


x = quant("global_in", ACT_SCALE, 1)
if args.distillnet:
    d, hidden = args.dim, args.ratio * args.dim
    # alpha po2 like the trained model: embed 2 (after ReLU), residual-stream norms 8
    x = matmul(dyt(relu(matmul(x, 4, hidden, in_scale=ACT_SCALE)), 2.0), hidden, d, in_scale=TANH_OUT_SCALE)
    for _ in range(args.phi_blocks):
        x = residual(x, d, hidden, 8.0)
    n_pp = count("MatMul")
    x = pool(x)
    for _ in range(args.rho_blocks):
        x = residual(x, d, hidden, 8.0)
    matmul(quant(x, ACT_SCALE, 1), d, 2, out="logits", in_scale=ACT_SCALE)
else:
    s, d = ACT_SCALE, 4
    for d_out in phi:
        x = matmul(x, d, d_out, in_scale=s if fq else None)
        x, s, d = (dyt(relu(x), 2.0), TANH_OUT_SCALE, d_out) if fq else (quant(relu(x), ACT_SCALE, 0), s, d_out)
    n_pp = count("MatMul")
    x = pool(x)
    x, s = (quant(x, TANH_OUT_SCALE, 1), TANH_OUT_SCALE) if fq else (quant(x, ACT_SCALE, 0), s)
    for d_out in rho:
        x = matmul(x, d, d_out, in_scale=s if fq else None)
        x, s, d = (dyt(relu(x), 2.0), TANH_OUT_SCALE, d_out) if fq else (quant(relu(x), ACT_SCALE, 0), s, d_out)
    matmul(x, d, 2, out="logits", in_scale=s if fq else None)

graph = helper.make_graph(
    nodes,
    "mini_deepsets",
    [helper.make_tensor_value_info("global_in", TensorProto.FLOAT, [BATCH, args.n, 4])],
    [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [BATCH, 2])],
    inits,
)
os.makedirs(OUT_DIR, exist_ok=True)
onnx_path = f"{OUT_DIR}/model.onnx"
onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9), onnx_path)
model = cleanup_model(ModelWrapper(onnx_path))

# --- convert ---
cfg = hls4ml.utils.config_from_onnx_model(model, granularity="name", backend="Vitis", default_precision="fixed<16,6>")
cfg["Model"]["Strategy"] = "Latency"
for layer_cfg in cfg["LayerName"].values():
    layer_cfg["Strategy"] = "Latency"
for i in range(n_pp):  # set on MatMul_<i>: MatmulConstToDense copies it onto the Dense_MatMul_<i> PointwiseConv1D
    cfg["LayerName"][f"MatMul_{i}"]["ParallelizationFactor"] = pf
cfg["LayerName"]["GlobalAveragePool_0"]["Precision"]["accum"] = "fixed<32,16>"  # sum of n 8-bit values


def quant_type(node):
    """ap_fixed type equal to a power-of-2-scale, zero-offset QONNX Quant (signed, not narrow)."""
    scale = model.get_initializer(node.input[1]).item()
    bits = int(model.get_initializer(node.input[3]).item())
    return f"fixed<{bits},{bits + int(np.log2(scale))},RND_CONV,SAT>"


def is_quant(node):
    return node is not None and node.op_type == "Quant"


def convert(hls_cfg):
    return hls4ml.converters.convert_from_onnx_model(
        model, output_dir=OUT_DIR, project_name="mini", backend="Vitis", io_type="io_parallel", part=PART, hls_config=hls_cfg
    )


if fq:  # exact types, as convert.py sets them for the full-quant graph
    for t in model.get_nodes_by_op_type("Tanh"):
        pre, post = model.find_producer(t.input[0]), model.find_consumer(t.output[0])
        cfg["LayerName"][t.name]["TableSize"] = int(round(8 / model.get_initializer(pre.input[1]).item()))
        cfg["LayerName"][t.name]["table_t"] = quant_type(post)
        cfg["LayerName"][t.name]["Precision"]["result"] = quant_type(post)
    cfg["LayerName"]["global_in"]["Precision"]["result"] = quant_type(model.find_consumer("global_in"))
    first = convert(copy.deepcopy(cfg)).graph
    pool_in = first["GlobalAveragePool_0"].get_input_variable().type.precision
    k = int(np.ceil(np.log2(args.n)))  # exact /n for power-of-2 n
    if isinstance(pool_in, FixedPrecisionType):
        w, i = pool_in.width, pool_in.integer
        cfg["LayerName"]["GlobalAveragePool_0"]["Precision"]["accum"] = f"fixed<{w + 2 * k},{i + k}>"
        cfg["LayerName"]["GlobalAveragePool_0"]["Precision"]["result"] = f"fixed<{w + k},{i}>"
    for r in model.get_nodes_by_op_type("Relu"):
        layer = first.get(r.name)
        if layer is not None and layer.get_output_variable().type.precision.rounding_mode.name == "TRN":
            in_t = layer.get_input_variable().type.precision
            if isinstance(in_t, FixedPrecisionType):
                cfg["LayerName"][r.name]["Precision"]["result"] = f"fixed<{in_t.width},{in_t.integer}>"

hls_model = convert(cfg)
hls_model.compile()
for layer in hls_model.get_layers():
    if isinstance(layer, ApplyAlpha):  # item 3 of io_parallel_report.md: per-channel, not per-particle broadcast
        print(f"[alpha] {layer.name}: n_in={layer.get_attr('n_in')} n_filt={layer.get_attr('n_filt')}")

# --- C-sim parity vs qonnx on random inputs (inputs on the 8-bit grid) ---
xin = (np.round(rng.normal(0, 2, (BATCH, args.n, 4)) / ACT_SCALE) * ACT_SCALE).clip(-8, 8 - ACT_SCALE).astype(np.float32)
ref = execute_onnx(model, {"global_in": xin})[model.graph.output[0].name]
hls = np.asarray(hls_model.predict(np.ascontiguousarray(xin))).reshape(ref.shape)
print(f"[random] max|dlogit|={np.abs(ref - hls).max():.4g} argmax agree={np.mean(ref.argmax(1) == hls.argmax(1)):.3f}")
print(f"[model] {OUT_DIR} per-particle MatMuls={n_pp} MACs/jet={args.n * sum(macs[:n_pp]) + sum(macs[n_pp:])}")

if args.synth:
    hls_model.build(csim=False, synth=True, export=False)
    hls4ml.report.read_vivado_report(OUT_DIR)
