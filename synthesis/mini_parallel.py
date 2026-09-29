"""Mini DeepSets with random weights: how small must the model be for a fully parallel (io_parallel) hls4ml design?

Builds a QONNX graph shaped like the distillnet student but without DynamicTanh and biases:
    Quant(in) -> [MatMul(Quant W) -> Relu -> Quant] per phi layer (per particle) -> mean pool
             -> [MatMul -> Relu -> Quant] per rho layer -> MatMul -> 2 logits
All Quant nodes are 8-bit with power-of-2 scales (like the Brevitas QAT export). Weights are random, so only the
C-sim parity and the synthesis numbers mean anything, not the physics.

Usage (from synthesis/, on rdsrv409, same env as convert.py):
    python mini_parallel.py --n 16                  # convert + C-sim parity, PF = n (fully parallel)
    timeout 1h python mini_parallel.py --n 16 --synth 2>&1 | tee logs/mini_n16.txt

Writes hls_prj/mini_n<N>_phi<..>_rho<..>_pf<PF>/ (the ONNX graph is saved there as model.onnx). Independent of
convert.py; the two hls4ml patches it needs are copied from there.
"""

import argparse
import importlib
import os

import hls4ml
import numpy as np
import onnx
from hls4ml.model.optimizer.passes.multi_dense import ReplaceMultidimensionalDenseWithConv
from onnx import TensorProto, helper, numpy_helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.util.cleanup import cleanup_model

PART = "xcvu13p-flga2577-2-e"
BATCH = 8
ACT_SCALE = 2.0**-4  # 8-bit activations cover [-8, 8) signed / [0, 16) unsigned

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=16, help="particles per jet")
p.add_argument("--phi", default="64,32", help="per-particle layer widths")
p.add_argument("--rho", default="32", help="post-pool hidden widths (then 2 logits)")
p.add_argument("--pf", type=int, default=None, help="ParallelizationFactor of the phi layers (default: n)")
p.add_argument("--synth", action="store_true")
args = p.parse_args()
phi = [int(d) for d in args.phi.split(",")]
rho = [int(d) for d in args.rho.split(",")] if args.rho else []
pf = args.pf or args.n
OUT_DIR = f"hls_prj/mini_n{args.n}_phi{'-'.join(map(str, phi))}_rho{'-'.join(map(str, rho))}_pf{pf}"

# --- hls4ml patches copied from convert.py (see the comments there) ---
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
nodes, inits = [], []


def const(name, value):
    inits.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), name))
    return name


def quant(x, scale, signed, narrow=0):
    i = sum(n.op_type == "Quant" for n in nodes)
    out = f"Quant_{i}_out"
    nodes.append(
        helper.make_node(
            "Quant",
            [x, const(f"Quant_{i}_s", scale), const(f"Quant_{i}_z", 0.0), const(f"Quant_{i}_b", 8.0)],
            [out],
            name=f"Quant_{i}",
            domain="qonnx.custom_op.general",
            signed=signed,
            narrow=narrow,
            rounding_mode="ROUND",
        )
    )
    return out


def matmul(x, d_in, d_out, out=None):
    i = sum(n.op_type == "MatMul" for n in nodes)
    w = rng.normal(0, 1 / np.sqrt(d_in), (d_in, d_out))
    w_scale = 2.0 ** np.ceil(np.log2(np.abs(w).max() / 127))
    out = out or f"MatMul_{i}_out"
    nodes.append(helper.make_node("MatMul", [x, quant(const(f"W_{i}", w), w_scale, 1, narrow=1)], [out], name=f"MatMul_{i}"))
    return out


def relu_q(x):
    i = sum(n.op_type == "Relu" for n in nodes)
    nodes.append(helper.make_node("Relu", [x], [f"Relu_{i}_out"], name=f"Relu_{i}"))
    return quant(f"Relu_{i}_out", ACT_SCALE, 0)


x, d = quant("global_in", ACT_SCALE, 1), 4
for d_out in phi:
    x, d = relu_q(matmul(x, d, d_out)), d_out
n_phi = len(phi)
nodes += [
    helper.make_node("Transpose", [x], ["Transpose_0_out"], name="Transpose_0", perm=[0, 2, 1]),
    helper.make_node("GlobalAveragePool", ["Transpose_0_out"], ["Pool_out"], name="GlobalAveragePool_0"),
    helper.make_node("Flatten", ["Pool_out"], ["Flatten_out"], name="Flatten_0", axis=1),
]
x = quant("Flatten_out", ACT_SCALE, 0)
for d_out in rho:
    x, d = relu_q(matmul(x, d, d_out)), d_out
matmul(x, d, 2, out="logits")

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
for i in range(n_phi):  # set on MatMul_<i>: MatmulConstToDense copies it onto the Dense_MatMul_<i> PointwiseConv1D
    cfg["LayerName"][f"MatMul_{i}"]["ParallelizationFactor"] = pf
cfg["LayerName"]["GlobalAveragePool_0"]["Precision"]["accum"] = "fixed<32,16>"  # sum of n unsigned 8-bit values

hls_model = hls4ml.converters.convert_from_onnx_model(
    model, output_dir=OUT_DIR, project_name="mini", backend="Vitis", io_type="io_parallel", part=PART, hls_config=cfg
)
hls_model.compile()

# --- C-sim parity vs qonnx on random inputs (inputs on the 8-bit grid) ---
xin = (np.round(rng.normal(0, 2, (BATCH, args.n, 4)) / ACT_SCALE) * ACT_SCALE).clip(-8, 8 - ACT_SCALE).astype(np.float32)
ref = execute_onnx(model, {"global_in": xin})[model.graph.output[0].name]
hls = np.asarray(hls_model.predict(np.ascontiguousarray(xin))).reshape(ref.shape)
print(f"[random] max|dlogit|={np.abs(ref - hls).max():.4g} argmax agree={np.mean(ref.argmax(1) == hls.argmax(1)):.3f}")
macs = args.n * sum(a * b for a, b in zip([4] + phi, phi)) + sum(a * b for a, b in zip([phi[-1]] + rho, rho + [2]))
print(f"[model] n={args.n} phi={phi} rho={rho} pf={pf} MACs/jet={macs}")

if args.synth:
    hls_model.build(csim=False, synth=True, export=False)
    hls4ml.report.read_vivado_report(OUT_DIR)
