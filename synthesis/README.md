# hls4ml conversion + synthesis (off-NERSC machine with Vitis/Vivado)

Notes for the agent/person running the FPGA step. Training, QAT and QONNX
export already happened on NERSC; this machine only converts and synthesizes.

## Input

`onnx_graphs/qat_top_deepsets_distillnet_fpga_a05_T4_8bit_clean.onnx` (55 KB), copied from
`/pscratch/sd/a/alexmay/omnilearned/qonnx/fpga/` on Perlmutter.

- Model: DeepSets "distillnet" student (base_dim 32, phi 2 layers, rho 1 layer,
  ReLU, DynamicTanh norm), ~11k params, KD'd from `fine_tune_top_l`, then 8-bit
  Brevitas QAT (power-of-2 scales, per-tensor fixed point).
- Task: top-tagging, 2 logits.
- Input `global_in`: `[64, 64, 4]` = (batch, particles, features). Particles are
  the leading-pT 64 slots of the 150-slot dataset, no mask. Batch dim is fixed
  to 64 from the export dummy input.
- Output: `[64, 2]` logits.
- Ops (46 nodes): Quant x14, Add x6, Mul x6, Transpose x5, MatMul x4, Gemm x3,
  Relu x3, Tanh x3, Flatten, GlobalAveragePool. All are in hls4ml 1.3's
  supported ONNX set. Mean pool is `GlobalAveragePool` (not `ReduceMean`).
- Verified on NERSC: onnx checker OK, IR version 9, every Quant 8-bit,
  qonnx-executor vs PyTorch max |Δ logit| = 0, 100% argmax agreement.

## Environment

Python 3.10-3.12 venv/conda. Versions known to work for the export side:

```
qonnx==1.0.0
onnx==1.20.1          # onnx>=1.21 writes IR 14, onnxruntime 1.30 rejects it
onnxruntime==1.30.0
onnxoptimizer==0.4.2
numpy
hls4ml                # patched fork, see below
```

hls4ml must include the ONNX-frontend fixes from
https://github.com/fastmachinelearning/hls4ml/compare/main...qibin2020:hls4ml:fix/onnx-frontend-shape-bugs
(stock hls4ml misreads the batch dim / shapes of ONNX graphs). Two commits:

- `3949f38` MatmulConstToDense: derive dims from the last axis. Without it a
  rank-3 MatMul like our per-particle `[B, 64, C] x [C, D]` gets the wrong
  output size (e.g. `[1,8,4] x [4,6]` -> 384 values instead of 48).
- `1d85133` Reshape: resolve the ONNX target shape explicitly and keep the
  batch dim, instead of blindly dropping leading entries (e.g. `[1,8,4]` ->
  `[-1,4]` modeled as 4 elements instead of 32).

Install the branch directly (it is based on hls4ml `main`, not the 1.3 release):

```bash
pip install "git+https://github.com/qibin2020/hls4ml.git@fix/onnx-frontend-shape-bugs"
```

or cherry-pick both commits onto your own hls4ml checkout and `pip install -e .`.

PyTorch, Brevitas and the `omnilearned` package are not needed.

Xilinx toolchain: Vitis HLS (hls4ml `Vitis` backend) or Vivado HLS
(`Vivado` backend). Source `settings64.sh` so `vitis_hls`/`vivado` are on PATH.

## Plan

1. **Convert + C-sim** with a custom script written on this machine:

   ```python
   import hls4ml
   from qonnx.core.modelwrapper import ModelWrapper
   from qonnx.util.cleanup import cleanup_model
   from qonnx.transformation.gemm_to_matmul import GemmToMatMul

   model = ModelWrapper("qat_top_deepsets_distillnet_fpga_a05_T4_8bit_clean.onnx")
   model = cleanup_model(model).transform(GemmToMatMul())
   model = cleanup_model(model)

   cfg = hls4ml.utils.config_from_onnx_model(
       model, granularity="name", backend="Vitis",
       default_precision="fixed<16,6>")
   hls_model = hls4ml.converters.convert_from_onnx_model(
       model, output_dir="hls_prj/deepsets_distillnet_8bit",
       project_name="deepsets", backend="Vitis", io_type="io_stream",
       part="<target FPGA part>", hls_config=cfg)
   hls_model.compile()   # g++ C-sim, no Vitis needed
   ```

   Quick smoke test: random float32 `[64, 64, 4]` through `hls_model.predict()`
   vs `qonnx.core.onnx_exec.execute_onnx(model, {"global_in": x})`. This only
   shows the conversion runs; it does not prove physics performance (fixed-point
   overflow/saturation depends on the real input distribution).

2. **Performance check on real jets** with `top_test_10k_n64.npz` (copy into `synthesis/`,
   also at `/pscratch/sd/a/alexmay/omnilearned/qonnx/fpga/` on Perlmutter):

   - `x`: float32 `[10000, 64, 4]`, first 10k jets of the top-tagging test
     split, leading-pT 64 particle slots, same preprocessing as training
     (`omnilearned.dataloader.load_data("top", dataset_type="test", ...)`).
   - `y`: int `[10000]` labels (5080 / 4920).

   Run both models in batches of 64 (graph batch dim is fixed; 9984 jets =
   156 full batches) and compare:
   - max |Δ logit| and argmax agreement, HLS vs qonnx executor
   - accuracy, AUC, background rejection 1/eB at signal eff 0.5, vs `y`
     (scores = softmax of the logits, class 1)

   Reference from the qonnx executor on NERSC (9984 jets):

   | metric        | qonnx (8-bit QAT) |
   |---------------|-------------------|
   | accuracy      | 0.9181            |
   | AUC           | 0.9755            |
   | 1/eB @ eS=0.5 | 169.3             |

   The HLS model should match these closely; a large drop means a
   precision/overflow problem, not a conversion bug.

3. **Likely trouble spots** (unverified until conversion runs):
   - 3-D tensors: per-particle MatMul / Transpose on `[batch, 64, C]`. hls4ml
     may need `io_parallel` or reshaping; try both io types if one fails.
   - DynamicTanh = `Tanh` + `Mul` by a learned scalar; check the Mul becomes a
     constant scale, not an elementwise layer with a second input.
   - Precision: accumulators after the 64-way mean may need wider types than
     `fixed<16,6>`; if csim parity is off, inspect with
     `hls4ml.model.profiling` and raise per-layer precision.
   - Latency vs resources: set `ReuseFactor` / `Strategy` in the config if the
     design is too big (64 particles x 32-wide phi layers).

4. **Synthesize** after csim parity is good:

   ```python
   hls_model.build(csim=False, synth=True, export=False)   # vsynth=True for Vivado logic synth
   hls4ml.report.read_vivado_report("hls_prj/deepsets_distillnet_8bit")
   ```

   make sure to run only on rdsrv409 (has memory limit whihc may be problematic)

5. Report back: which stage passed/failed, csim max |Δ|, argmax agreement and
   real-jet metrics vs the table above,
   synth report numbers, and any graph changes needed (those must be made on
   the PyTorch side in `src/omnilearned/network.py` and re-exported on NERSC
   with `tools/quantize/qat_deepsets_export_qonnx.py`).

6. just start with this first run to see if it runs. If synthesis is taking longer than 1 hour kill it.
