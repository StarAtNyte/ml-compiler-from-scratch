# Appendix D: Complete source code

This appendix contains the whole repository that the book builds: the compiler (`uopc/`), its tests
(`tests/`), and the example programs (`examples/`). Each file is included at build time from the same commit
as the text, so it always matches the listings in the chapters. Click a file name to expand it. In the PDF
every file is printed in full.

The files are in the order the course introduces them. Line counts include comments and blank lines.

## D.1 The compiler: `uopc/`

| Week | Modules |
|---|---|
| 1 | `uop.py`, `dtype.py`, `rewrite.py`, `symbolic.py`, `uir.py` |
| 2–3 | `render_c.py`, `runtime.py`, `linearize.py` |
| 4 | `rangeify.py`, `composite.py` |
| 5 | `kernelize.py`, `autodiff.py`, `tensor.py`, `nn.py` |
| 6 | `opt.py` |
| 7–8 | `gpu.py`, `cuda.py` |

@source uopc/__init__.py uopc/dtype.py uopc/uop.py uopc/rewrite.py uopc/symbolic.py uopc/uir.py uopc/render_c.py uopc/runtime.py uopc/linearize.py uopc/rangeify.py uopc/composite.py uopc/kernelize.py uopc/autodiff.py uopc/tensor.py uopc/nn.py uopc/opt.py uopc/gpu.py uopc/cuda.py

## D.2 Tests: `tests/`

One file per week. `helpers.py` lowers and runs a shaped graph against numpy; `conftest.py` puts it on the path.

@source tests/conftest.py tests/helpers.py tests/test_w01_uops.py tests/test_w02_render.py tests/test_w03_loops.py tests/test_w04_rangeify.py tests/test_w05_tensor.py tests/test_w06_opt.py tests/test_w07_gpu.py tests/test_w09_models.py

## D.3 Examples: `examples/`

@source examples/datasets.py examples/mnist_mlp.py examples/bench_gemm.py examples/flash_attention.py examples/gpt.py

## D.4 Project files

@source pyproject.toml
