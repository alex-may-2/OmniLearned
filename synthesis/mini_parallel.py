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
              Quant scales, alphas and the 10-bit residual/pool Quants match the r7 graph
              (onnx_graphs/..._fullQuant_r7_clean.onnx); --res-bits 0 / --no-relu-uint drop the r7-only Quants.

Usage (from synthesis/, on rdsrv409, same env as convert.py):
    python mini_parallel.py --n 16                  # convert + C-sim parity, PF = n (fully parallel)
    timeout 1h python mini_parallel.py --n 16 --synth 2>&1 | tee logs/mini_n16.txt
    python mini_parallel.py --full-quant --n 16 --phi 32,16 --rho 16 --pf 8
    python mini_parallel.py --distillnet --n 32 --pf 2
    python mini_parallel.py --distillnet --dim 8 --n 16 --pf 4 --mult-limit-fix --clock 2.78   # II-search levers
    python mini_parallel.py --distillnet --n 32 --io-type io_stream --clock 2.78

Writes hls_prj/mini[_fq|_ds<..>]_n<N>_..._pf<PF>[_<levers>]/ (the ONNX graph is saved there as model.onnx).
Independent of convert.py; the hls4ml patches and the full-quant type settings it needs are copied from there.
"""

import argparse
import copy
import importlib
import os
import re

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
from qonnx.transformation.infer_shapes import InferShapes
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
p.add_argument("--rho-ratio", type=int, default=None, help="--distillnet: rho blocks' mlp_ratio (default --ratio)")
p.add_argument("--phi-blocks", type=int, default=1, help="--distillnet: residual phi blocks after the embed")
p.add_argument("--rho-blocks", type=int, default=1, help="--distillnet: residual rho blocks")
p.add_argument("--pf", type=int, default=None, help="ParallelizationFactor of the per-particle layers (default: n)")
p.add_argument("--res-bits", type=int, default=10, help="--distillnet: bits of the r7 residual-stream/pool Quants (0: none)")
p.add_argument("--relu-uint", action=argparse.BooleanOptionalAction, default=True, help="--distillnet: r7 unsigned block-ReLU Quants")
p.add_argument("--no-embed-dyt", action="store_true", help="--distillnet: embed ReLU -> unsigned Quant, no DyT")
p.add_argument("--cut", default=None,
               help="debug: end the graph at this ONNX tensor (e.g. Quant_3_out) for C/RTL cosim bisection")
p.add_argument("--w-bits", type=int, default=8, help="--distillnet: weight bits")
p.add_argument("--a-bits", type=int, default=8, help="--distillnet: activation bits (Linear inputs; 8-bit ranges kept)")
p.add_argument("--in-bits", type=int, default=None, help="--distillnet: input Quant bits, range +-4 (default --a-bits)")
p.add_argument("--tanh-bits", type=int, default=None, help="--distillnet: tanh input bits (default --a-bits)")
p.add_argument("--tanh-in-max", type=float, default=4.0, help="--distillnet: tanh input range [-x, x)")
p.add_argument("--round", choices=["round", "floor"], default="round", help="activation Quant rounding (weights: round)")
p.add_argument("--io-type", default="io_parallel", choices=["io_parallel", "io_stream"])
p.add_argument("--clock", type=float, default=5.0, help="target clock period in ns")
p.add_argument("--rf", type=int, default=1, help="ReuseFactor of the per-particle layers")
p.add_argument("--mult-limit-fix", action="store_true", help="conv multiplier limit covers all n_pixels of a partition")
p.add_argument("--dsp-mult", action="store_true", help="bind all multiplies to DSPs (config_op mul -impl dsp)")
p.add_argument("--dsp-add", action="store_true", help="bind adds/subs to DSPs too (config_op add/sub -impl dsp)")
p.add_argument("--reshape-channels", action="store_true",
               help="io_parallel: ARRAY_RESHAPE the inter-layer arrays (one DATAFLOW channel per array, not per element)")
p.add_argument("--pipeline-style", choices=["pipeline", "dataflow"], default=None, help="hls4ml Model PipelineStyle")
p.add_argument("--pipeline-ii", type=int, default=None, help="with --pipeline-style pipeline: top-level II (hls4ml PipelineInterval)")
p.add_argument("--strategy", choices=["Latency", "Resource"], default=None, help="default: Resource for io_stream, else Latency")
p.add_argument("--synth", action="store_true")
args = p.parse_args()
args.in_bits, args.tanh_bits = args.in_bits or args.a_bits, args.tanh_bits or args.a_bits
A = 2.0 ** (8 - args.a_bits)  # a-bit activations keep the 8-bit ranges
ACT_SCALE, TANH_OUT_SCALE = ACT_SCALE * A, TANH_OUT_SCALE * A
TANH_IN_SCALE = args.tanh_in_max / 2 ** (args.tanh_bits - 1)
ROUNDING = args.round.upper()
fq = args.full_quant or args.distillnet
phi = [int(d) for d in args.phi.split(",")]
rho = [int(d) for d in args.rho.split(",")] if args.rho else []
pf = args.pf or args.n
stream = args.io_type == "io_stream"
if args.distillnet:
    OUT_DIR = f"hls_prj/mini_ds_d{args.dim}r{args.ratio}_phi{args.phi_blocks}_rho{args.rho_blocks}_n{args.n}_pf{pf}"
    OUT_DIR += f"_q{args.res_bits}{'u' if args.relu_uint else ''}{'_noembdyt' if args.no_embed_dyt else ''}"
    OUT_DIR += f"_h{args.rho_ratio}" if args.rho_ratio and args.rho_ratio != args.ratio else ""
    qs = [("w", args.w_bits, 8), ("a", args.a_bits, 8), ("i", args.in_bits, args.a_bits), ("t", args.tanh_bits, args.a_bits)]
    qs = "".join(f"{k}{v:g}" for k, v, d in qs + [("tm", args.tanh_in_max, 4)] if v != d)
    OUT_DIR += (f"_{qs}" if qs else "") + ("_fl" if args.round == "floor" else "")
else:
    OUT_DIR = f"hls_prj/mini{'_fq' if fq else ''}_n{args.n}_phi{'-'.join(map(str, phi))}_rho{'-'.join(map(str, rho))}_pf{pf}"
levers = [(f"rf{args.rf}", args.rf > 1), ("mlf", args.mult_limit_fix), ("dsp", args.dsp_mult)]
levers += [("dspadd", args.dsp_add), ("rsh", args.reshape_channels)]
levers += [(str(args.pipeline_style), args.pipeline_style), ("stream", stream), (f"clk{args.clock:g}", args.clock != 5)]
levers += [(f"ii{args.pipeline_ii}", args.pipeline_ii)]
levers += [(str(args.strategy).lower(), args.strategy and args.strategy != ("Resource" if stream else "Latency"))]
OUT_DIR += "".join(f"_{tag}" for tag, on in levers if on)
OUT_DIR += f"_cut{args.cut}" if args.cut else ""

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


def quant(x, scale, signed, narrow=0, bits=None, rounding=None):
    i = count("Quant")
    out = f"Quant_{i}_out"
    nodes.append(
        helper.make_node(
            "Quant",
            [x, const(f"Quant_{i}_s", scale), const(f"Quant_{i}_z", 0.0), const(f"Quant_{i}_b", float(bits or args.a_bits))],
            [out],
            name=f"Quant_{i}",
            domain="qonnx.custom_op.general",
            signed=signed,
            narrow=narrow,
            rounding_mode=rounding or ROUNDING,
        )
    )
    return out


def matmul(x, d_in, d_out, out=None, in_scale=None):
    """MatMul with an 8-bit weight Quant; with in_scale (full-quant) also an Int16 bias on the accumulator grid."""
    i = count("MatMul")
    w = rng.normal(0, 1 / np.sqrt(d_in), (d_in, d_out))
    w_scale = 2.0 ** np.ceil(np.log2(np.abs(w).max() / (2 ** (args.w_bits - 1) - 1)))
    macs.append(d_in * d_out)
    mm_out = f"MatMul_{i}_out" if in_scale else out or f"MatMul_{i}_out"
    nodes.append(helper.make_node("MatMul", [x, quant(const(f"W_{i}", w), w_scale, 1, narrow=1, bits=args.w_bits, rounding="ROUND")], [mm_out], name=f"MatMul_{i}"))
    if not in_scale:
        return mm_out
    b = quant(const(f"B_{i}", rng.normal(0, 0.1, d_out)), in_scale * w_scale, 1, bits=16, rounding="ROUND")
    out = out or f"Bias_{i}_out"
    nodes.append(helper.make_node("Add", [mm_out, b], [out], name=f"Bias_{i}"))
    return out


def relu(x):
    i = count("Relu")
    nodes.append(helper.make_node("Relu", [x], [f"Relu_{i}_out"], name=f"Relu_{i}"))
    return f"Relu_{i}_out"


def dyt(x, alpha):
    """Full-quant DynamicTanh: tanh(Quant8(alpha_po2 * x)), output re-quantized by the next Linear's input Quant."""
    if alpha != 1:  # a Mul by 1 (r7 embed) makes hls4ml drop the next Quant; without it ReLU + Quant fuse as in r7
        i = count("Mul")
        nodes.append(helper.make_node("Mul", [x, const(f"alpha_{i}", alpha)], [f"Mul_{i}_out"], name=f"Mul_{i}"))
        x = f"Mul_{i}_out"
    t = quant(x, TANH_IN_SCALE, 1, bits=args.tanh_bits)
    j = count("Tanh")
    nodes.append(helper.make_node("Tanh", [t], [f"Tanh_{j}_out"], name=f"Tanh_{j}"))
    return quant(f"Tanh_{j}_out", TANH_OUT_SCALE, 1)


def pool(x):
    nodes.extend(
        [
            helper.make_node("Transpose", [x], ["Transpose_0_out"], name="Transpose_0", perm=[0, 2, 1]),
            helper.make_node("GlobalAveragePool", ["Transpose_0_out"], ["Pool_out"], name="GlobalAveragePool_0"),
            helper.make_node("Flatten", ["Pool_out"], ["Flatten_out"], name="Flatten_0", axis=1),
        ]
    )
    return "Flatten_out"


def relu_quant(x, scale):
    """Block ReLU and the next Linear's input Quant: unsigned 8-bit like r7 (--relu-uint), else signed 1/16."""
    return quant(relu(x), scale, 0) if args.relu_uint else quant(relu(x), ACT_SCALE, 1)


def res_quant(x, scale):
    """r7 residual-stream / pool Quant (--res-bits, signed); no Quant with --res-bits 0."""
    return quant(x, scale, 1, bits=args.res_bits) if args.res_bits else x


def residual(h, d, hidden, alpha, relu_scale=ACT_SCALE):
    """Pre-norm residual block: h + fc2(Quant(Relu(fc1(DyT(h)))))."""
    t = matmul(dyt(h, alpha), d, hidden, in_scale=TANH_OUT_SCALE)
    t = relu_quant(t, relu_scale)
    t = matmul(t, hidden, d, in_scale=relu_scale if args.relu_uint else ACT_SCALE)
    i = sum(n.name.startswith("Res_") for n in nodes)
    nodes.append(helper.make_node("Add", [h, t], [f"Res_{i}_out"], name=f"Res_{i}"))
    return f"Res_{i}_out"


IN_SCALE = 4 / 2 ** (args.in_bits - 1) if args.distillnet else ACT_SCALE  # r7 input Quant: 8-bit, 1/32 (+-4)
x = quant("global_in", IN_SCALE, 1, bits=args.in_bits if args.distillnet else None)
if args.distillnet:
    d, hidden = args.dim, args.ratio * args.dim
    # Scales and po2 alphas from the r7 graph: embed alpha 1, residual-stream norms 8; block ReLU Quants 1/64 (phi)
    # and 1/128 (rho); 10-bit residual stream 1/128 (embed out), 1/64 (after each phi block), pool out 1/512.
    t = relu(matmul(x, 4, hidden, in_scale=IN_SCALE))
    if args.no_embed_dyt:
        x = matmul(quant(t, 2.0**-5, 0), hidden, d, in_scale=2.0**-5)
    else:
        x = matmul(dyt(t, 1.0), hidden, d, in_scale=TANH_OUT_SCALE)
    x = res_quant(x, 2.0**-7)
    for _ in range(args.phi_blocks):
        x = res_quant(residual(x, d, hidden, 8.0, A * 2.0**-6), 2.0**-6)
    n_pp = count("MatMul")
    x = res_quant(pool(x), 2.0**-9)
    for _ in range(args.rho_blocks):
        x = residual(x, d, (args.rho_ratio or args.ratio) * d, 8.0, A * 2.0**-7)
    out_scale = TANH_OUT_SCALE if args.res_bits else ACT_SCALE  # r7: 8-bit 1/128 before the output Linear
    matmul(quant(x, out_scale, 1), d, 2, out="logits", in_scale=out_scale)
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

in_vi = helper.make_tensor_value_info("global_in", TensorProto.FLOAT, [BATCH, args.n, 4])
out_vi = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [BATCH, 2])
if args.cut:  # keep only the nodes the cut tensor needs; its shape from one execution of the full graph
    full = ModelWrapper(helper.make_model(helper.make_graph(nodes, "full", [in_vi], [out_vi], inits),
                                          opset_imports=[helper.make_opsetid("", 13)], ir_version=9)).transform(InferShapes())
    shape = execute_onnx(full, {"global_in": np.zeros((BATCH, args.n, 4), np.float32)}, True)[args.cut].shape
    need, keep = {args.cut}, []
    for nd in reversed(nodes):
        if set(nd.output) & need:
            keep.insert(0, nd)
            need |= set(nd.input)
    nodes[:], inits[:] = keep, [t for t in inits if t.name in need]
    out_vi = helper.make_tensor_value_info(args.cut, TensorProto.FLOAT, list(shape))
graph = helper.make_graph(nodes, "mini_deepsets", [in_vi], [out_vi], inits)
os.makedirs(OUT_DIR, exist_ok=True)
onnx_path = f"{OUT_DIR}/model.onnx"
onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9), onnx_path)
model = cleanup_model(ModelWrapper(onnx_path))

# --- convert ---
cfg = hls4ml.utils.config_from_onnx_model(model, granularity="name", backend="Vitis", default_precision="fixed<16,6>")
strategy = args.strategy or ("Resource" if stream else "Latency")  # io_stream default: as the r7 io_stream build
cfg["Model"]["Strategy"] = strategy
if args.pipeline_style:
    cfg["Model"]["PipelineStyle"] = args.pipeline_style
if args.pipeline_ii:
    cfg["Model"]["PipelineInterval"] = args.pipeline_ii
for layer_cfg in cfg["LayerName"].values():
    layer_cfg["Strategy"] = strategy
for i in range(n_pp):  # set on MatMul_<i>: MatmulConstToDense copies it onto the Dense_MatMul_<i> PointwiseConv1D
    if f"MatMul_{i}" not in cfg["LayerName"]:  # --cut before it
        continue
    if not stream:
        cfg["LayerName"][f"MatMul_{i}"]["ParallelizationFactor"] = pf
    cfg["LayerName"][f"MatMul_{i}"]["ReuseFactor"] = args.rf
if "GlobalAveragePool_0" in cfg["LayerName"]:  # not with --cut before the pool
    cfg["LayerName"]["GlobalAveragePool_0"]["Precision"]["accum"] = "fixed<32,16>"  # sum of n 8-bit values


def quant_type(node):
    """ap_(u)fixed type equal to a power-of-2-scale, zero-offset, not narrow QONNX Quant."""
    scale = model.get_initializer(node.input[1]).item()
    bits = int(model.get_initializer(node.input[3]).item())
    signed = next((a.i for a in node.attribute if a.name == "signed"), 1)
    rnd = "TRN" if next((a.s for a in node.attribute if a.name == "rounding_mode"), b"ROUND") == b"FLOOR" else "RND_CONV"
    return f"{'' if signed else 'u'}fixed<{bits},{bits + int(np.log2(scale))},{rnd},SAT>"


def is_quant(node):
    return node is not None and node.op_type == "Quant"


def convert(hls_cfg):
    return hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir=OUT_DIR,
        project_name="mini",
        backend="Vitis",
        io_type=args.io_type,
        part=PART,
        clock_period=args.clock,
        hls_config=hls_cfg,
    )


if fq:  # exact types, as convert.py sets them for the full-quant graph
    for t in model.get_nodes_by_op_type("Tanh"):
        pre, post = model.find_producer(t.input[0]), model.find_consumer(t.output[0])
        cfg["LayerName"][t.name]["TableSize"] = int(round(8 / model.get_initializer(pre.input[1]).item()))
        cfg["LayerName"][t.name]["table_t"] = quant_type(post)
        cfg["LayerName"][t.name]["Precision"]["result"] = quant_type(post)
    cfg["LayerName"]["global_in"]["Precision"]["result"] = quant_type(model.find_consumer("global_in"))
    first = convert(copy.deepcopy(cfg)).graph
    pool_in = first["GlobalAveragePool_0"].get_input_variable().type.precision if "GlobalAveragePool_0" in first else None
    k = int(np.ceil(np.log2(args.n)))  # exact /n for power-of-2 n
    # Other n: the pool truncates sum / n in accum_t, and the true mean is never closer than 1 / (n * 2^10) to a
    # rounding tie of the 10-bit pool Quant, so >= log2(n) + 11 fractional bits keep it exact (6 more than 2k).
    extra = 0 if args.n & (args.n - 1) == 0 else 6
    if isinstance(pool_in, FixedPrecisionType):
        w, i = pool_in.width, pool_in.integer
        cfg["LayerName"]["GlobalAveragePool_0"]["Precision"]["accum"] = f"fixed<{w + 2 * k + extra},{i + k}>"
        cfg["LayerName"]["GlobalAveragePool_0"]["Precision"]["result"] = f"fixed<{w + k},{i}>"
    # r7 pool -> Flatten -> Quant: round in the pool itself (exact for power-of-2 n). As a separate zero-latency
    # Quant it chains with the next alpha + Quant in one cycle (2.98 + 2.82 ns) and misses 5 ns in io_parallel.
    for gap in model.get_nodes_by_op_type("GlobalAveragePool"):
        q = model.find_consumer(model.find_consumer(gap.output[0]).output[0])
        if is_quant(q):
            cfg["LayerName"]["GlobalAveragePool_0"]["Precision"]["result"] = quant_type(q)
    for r in model.get_nodes_by_op_type("Relu"):
        layer = first.get(r.name)
        fused = is_quant(model.find_consumer(r.output[0]))  # a FLOOR Quant fused into the ReLU is TRN too
        if layer is not None and not fused and layer.get_output_variable().type.precision.rounding_mode.name == "TRN":
            in_t = layer.get_input_variable().type.precision
            if isinstance(in_t, FixedPrecisionType):
                cfg["LayerName"][r.name]["Precision"]["result"] = f"fixed<{in_t.width},{in_t.integer}>"

hls_model = convert(cfg)
hls_model.compile()
for layer in hls_model.get_layers():
    if isinstance(layer, ApplyAlpha):  # per-channel, not per-particle broadcast (io_parallel_report.md)
        print(f"[alpha] {layer.name}: n_in={layer.get_attr('n_in')} n_filt={layer.get_attr('n_filt')}")


def patch(path, old, new):
    s = open(path).read()
    assert old in s, f"{old!r} not in {path}"
    open(path, "w").write(s.replace(old, new))


# Synthesis-only levers, applied to the written project (C-sim is unaffected; build() does not rewrite it)
if args.mult_limit_fix:
    # The partition loop is pipelined at II = RF but ALLOCATION caps the multipliers at one pixel's n_in * n_out / RF,
    # while each iteration does n_pixels = PF pixels: II grew with PF (II 8 at n 16, PF 8; io_parallel_report.md).
    patch(
        f"{OUT_DIR}/firmware/nnet_utils/nnet_conv1d_latency.h",
        "    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::mult_config::multiplier_limit",
        "    const unsigned mult_limit_all = CONFIG_T::mult_config::multiplier_limit * CONFIG_T::n_pixels;\n"
        "    #pragma HLS ALLOCATION operation instances=mul limit=mult_limit_all",
    )


def clone_fanout(cpp):
    """io_parallel residuals: the skip array of each Add is also read by the DyT branch. A DATAFLOW array with two
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
    k = out.index("void mini(")
    open(cpp, "w").write(out[:k] + clone + out[k:])
    print(f"[clone] {n} fan-out arrays split in {cpp}")


def reshape_channels(cpp):
    """io_parallel DATAFLOW: each element of a completely partitioned inter-layer array is its own channel (a depth-2
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


if not stream:
    clone_fanout(f"{OUT_DIR}/firmware/mini.cpp")
if args.reshape_channels:
    reshape_channels(f"{OUT_DIR}/firmware/mini.cpp")
if args.dsp_mult:
    clk = "create_clock -period $clock_period -name default"
    patch(f"{OUT_DIR}/build_prj.tcl", clk, clk + "\nconfig_op mul -impl dsp")
if args.dsp_add:
    clk = "create_clock -period $clock_period -name default"
    patch(f"{OUT_DIR}/build_prj.tcl", clk, clk + "\nconfig_op add -impl dsp\nconfig_op sub -impl dsp")

# --- C-sim parity vs qonnx on random inputs (inputs on the 8-bit grid) ---
xin = (np.round(rng.normal(0, 2, (BATCH, args.n, 4)) / IN_SCALE) * IN_SCALE).clip(-(2 ** (args.in_bits - 1)) * IN_SCALE, (2 ** (args.in_bits - 1) - 1) * IN_SCALE).astype(np.float32)
ref = execute_onnx(model, {"global_in": xin})[model.graph.output[0].name]
hls = np.asarray(hls_model.predict(np.ascontiguousarray(xin))).reshape(ref.shape)
print(f"[random] max|dlogit|={np.abs(ref - hls).max():.4g} argmax agree={np.mean(ref.argmax(1) == hls.argmax(1)):.3f}")
print(f"[model] {OUT_DIR} per-particle MatMuls={n_pp} MACs/jet={args.n * sum(macs[:n_pp]) + sum(macs[n_pp:])}")

if args.synth and np.abs(ref - hls).max() > 0:
    raise SystemExit("C-sim is not bit-exact: no synthesis")
if args.synth:
    hls_model.build(csim=False, synth=True, export=False)
    hls4ml.report.read_vivado_report(OUT_DIR)
