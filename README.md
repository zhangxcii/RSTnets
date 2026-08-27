# Reynolds Stress Transport nets
This repository contains the code for reconstructing incompressible Reynolds-averaged Navier-Stokes (RANS) solutions from boundary data, through the incorporation of Reynolds stress transport (RST) formulations via physics-informed neural networks (PINNs) — without introducing empirical models for the unclosed terms. In particular, the RST equations are formulated where the exact forms for the convection and production terms are employed as derived from the Reynolds averaging process while the unclosed terms that need empirical modelling in traditional turbulence models are absorbed into an inadequacy term. This leads to the unclosed equation systems below:

$$\nabla \cdot \boldsymbol{U} = 0$$

$$\boldsymbol{U} \cdot \nabla \boldsymbol{U} = - \nabla P + \frac{1}{Re} \nabla^2 \boldsymbol{U} + \nabla \cdot \boldsymbol{\tau} $$

$$\boldsymbol{U} \cdot \nabla \boldsymbol{\tau} = - (\boldsymbol{\tau} \cdot \nabla \boldsymbol{U} + (\boldsymbol{\tau} \cdot \nabla \boldsymbol{U})^T) + \boldsymbol{\eta}$$

The unclosed equation systems are then solved by PINNs approach to predict the turbulent flow fields based on only the information at domain boundaries, leading to the baseline RSTnet (Reynolds Stress Transport net). Based on it, a mechanism of enforcing the realisability condition for the Reynolds stress is further proposed via customised neural network designs, leading to two RSTnet variants. For more details, please refer to my paper: **Reconstruction of Reynolds-averaged Navier-Stokes solutions from boundary data with partial Reynolds stress transport constraints**. This repository is maintained as a general research framework for RANS+RST PINNs, and it is under active development to cope with more generic flow setups.

- The baseline RSTnet
<img src=docs/RSTnet-a.png>

- RSTnet with non-negative normal Reynolds stresses
<img src=docs/RSTnet-b.png>

- RSTnet with realisable Reynolds stresses
<img src=docs/RSTnet-c.png>

## Installation

Clone the repository, create a fresh conda environment, and install the package into it:

```sh
git clone https://github.com/zhangxcii/RSTnets.git
cd RSTnets
conda create -n rstnets python=3.10 -y
conda activate rstnets
pip install -e .            # base (CPU) install
```

For GPU support or the optional torch backend, replace the last line with one of:

```sh
pip install -e ".[gpu]"     # GPU-enabled (pulls tensorflow[and-cuda])
pip install -e ".[torch]"   # also enable the torch backend
```

Dependencies (numpy, scipy, matplotlib, tensorflow, tensorflow-probability, tf-keras,
deepxde) are declared in `pyproject.toml` and installed automatically. `torch` is an optional
extra, not a base dependency — the base install stays TensorFlow-only/lightweight.

### Backends

Each case selects a backend via `"backend": "tensorflow"` or `"backend": "torch"` in its
`<name>_config.json` (defaults to `"tensorflow"` if omitted), overridable per-run with
`--backend {tensorflow,torch}` on `train.py`/`restore.py`.

## Repository layout

- **`rstnets/`** — the installable package. All code here is independent of any specific
  flow case: the unified `RSTnet` model class, data loading, the Adam+L-BFGS training loop,
  plotting/error utilities, config loading, and the two entry-point scripts `train.py`/
  `restore.py`.
- **`examples/`** — one directory per flow case. Each holds *data, configuration, and any
  case-specific code*: `<name>_config.json` (physics constants,
  network architecture, training defaults — filename must match the folder name) and `data/`
  (training/reference `.npy` files). Training writes everything
  (`Model-adam/`, `Model-lbfgs/`, loss history, plots, errors) into `examples/<name>/results/`.

- **`examples/template/`** — copy this directory to start a new case: `cp -r
  examples/template examples/my_flow`. See `examples/template/README.md`.
- **`deprecated/`** — the pre-refactor script *and* its pretrained checkpoints
  (`Data/`, `Model-adam/`, `Model-lbfgs/`). Preserved for reference, not maintained going forward.

## Usage

`pip install -e .` once; after that, `train.py`/`restore.py` can be run from **any directory**
— the example name always resolves against `<repo_root>/examples/`, regardless of the
caller's cwd or how the script itself was invoked (`python rstnets/train.py --case
phill`, `cd rstnets && python train.py --case phill`, or an absolute script
path from anywhere all behave identically). A bare positional also works (`python
rstnets/train.py phill`) if you prefer.

Train the phill case from scratch (Adam, then L-BFGS, then plots + relative-L2 errors):

```sh
python rstnets/train.py --case phill
```

Restore from an already-trained checkpoint without retraining:

```sh
python rstnets/restore.py --case phill
```

Validate a case/dataset without spending any training time:

```sh
python rstnets/train.py --case phill --dry-run
```

Bring your own dataset:

```sh
cp -r examples/template examples/my_flow
# edit examples/my_flow/template_config.json -> rename to my_flow_config.json, set invRe/description
# add your domain_in.npy / domain_data.npy / bc_in.npy / bc_data.npy under examples/my_flow/data/
python rstnets/train.py --case my_flow --dry-run   # validate first
python rstnets/train.py --case my_flow
```

`--case` also accepts a bare positional argument as a shorthand (`python rstnets/train.py
phill`). A value containing a `/` is treated as an explicit path instead of an
`examples/` name — e.g. `python rstnets/train.py --case ./path/to/a/case` or `../other/case`.

If you need custom plots or geometry-specific postprocessing beyond the
default contour comparison, add that code to your own example directory (not `rstnets/`) and
point `<name>_config.json`'s `postprocess_module` at it — see `examples/template/README.md`
and `examples/phill/postprocess.py` for the pattern.

Run `python rstnets/train.py --help` / `python rstnets/restore.py --help` for the full flag
list (iteration-count overrides, `--precision`, `--no-gpu`, `--seed`, `--output-dir`).

## Citation
Please refer to the reference below for citations. And I am happy to answer any questions (Email: jc.zhang15@gmail.com).
```
@article{zhang2026rstnets,
  author  = {Jincheng Zhang},
  title   = {Reconstruction of Reynolds-averaged Navier-Stokes solutions from boundary data with partial Reynolds stress transport constraints},
  journal = {Draft},
  volume  = {tbd},
  number  = {tbd},
  pages   = {tbd},
  year    = {2026},
  doi     = {tbd}
}
```
