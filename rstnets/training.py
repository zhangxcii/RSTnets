import importlib.util
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import SUPPORTED_BACKENDS, CaseConfig, RunConfig, effective_value
from .data import DataBundle, load_case_data
from .utils import plot_contour, relative_l2

# Pressure is excluded from the default report: it is reconstructed only up to an additive constant unless explicitly supervised.
DEFAULT_REPORT_FIELDS = ["u", "v", "uu", "uv", "vv"]


def _init_backend(run_config: RunConfig, case_config: CaseConfig):
    import os
    if not run_config.gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    backend = effective_value(run_config.backend, case_config.backend)
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"backend must be one of {SUPPORTED_BACKENDS}, got {backend!r}")
    precision = effective_value(run_config.precision, case_config.precision)

    if backend == "tensorflow":
        os.environ.setdefault("DDE_BACKEND", "tensorflow")
        import tensorflow as tf
        tf.keras.backend.set_floatx(precision)
        tf.compat.v1.enable_eager_execution()
    else:
        import torch
        torch.set_default_dtype(torch.float64 if precision == "float64" else torch.float32)

    return backend, precision


def build_model(backend: str, case_config: CaseConfig, data: DataBundle, precision=None):
    if backend == "tensorflow":
        from .model import RSTnet
    else:
        from .model_torch import RSTnet
    return RSTnet(case_config, data.train_dd, data.train_pi,
                  Lxscale=data.Lxscale, Lyscale=data.Lyscale, precision=precision)


def train_case(run_config: RunConfig, case_config: CaseConfig, dry_run: bool = False):
    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend, precision = _init_backend(run_config, case_config)
    data = load_case_data(run_config.data_path, case_config, seed=run_config.seed)
    model = build_model(backend, case_config, data, precision=precision)

    td = case_config.training_defaults
    n_iters_adam = effective_value(run_config.n_iters_adam, td.n_iters_adam)
    n_lbfgs = effective_value(run_config.n_lbfgs, td.n_lbfgs)
    print_interval = effective_value(run_config.print_interval, td.print_interval)
    batch_data = data.train_dd.shape[0] if td.batch_data == "full" else int(td.batch_data)
    batch_pi = data.train_pi.shape[0] // td.batch_pi_divisor

    lbfgs_batch_data_cfg = effective_value(td.lbfgs_batch_data, td.batch_data)
    lbfgs_batch_pi_divisor_cfg = effective_value(td.lbfgs_batch_pi_divisor, td.batch_pi_divisor)
    lbfgs_batch_data = data.train_dd.shape[0] if lbfgs_batch_data_cfg == "full" else int(lbfgs_batch_data_cfg)
    lbfgs_batch_pi = data.train_pi.shape[0] // lbfgs_batch_pi_divisor_cfg

    print(f"case={case_config.name!r} backend={backend} precision={precision} n_iters_adam={n_iters_adam} "
          f"n_lbfgs={n_lbfgs} batch_data={batch_data} batch_pi={batch_pi} lbfgs_batch_mode={td.lbfgs_batch_mode} "
          f"lbfgs_batch_data={lbfgs_batch_data} lbfgs_batch_pi={lbfgs_batch_pi}")
    print(f"train_dd shape={data.train_dd.shape} train_pi shape={data.train_pi.shape}")

    if dry_run:
        print("--dry-run: config and data validated, model built, skipping training.")
        return None

    (output_dir / "Model-adam").mkdir(exist_ok=True)
    (output_dir / "Model-lbfgs").mkdir(exist_ok=True)

    training_start = time.time()

    model.train(n_iters_adam, batch_data, batch_pi,
                history_path=str(output_dir / "train_hist.txt"), print_interval=print_interval)
    model.save(path=str(output_dir / "Model-adam") + "/")

    model.train_lbfgs(n_lbfgs, history_path=str(output_dir / "train_hist_lbfgs.txt"),
                       lbfgs_options=td.lbfgs_options, batch_mode=td.lbfgs_batch_mode,
                       batch_data=lbfgs_batch_data, batch_pi=lbfgs_batch_pi)
    model.save(path=str(output_dir / "Model-lbfgs") + "/")

    total_time = time.time() - training_start

    return _plot_and_report(run_config.data_path, case_config, data, model, output_dir,
                             total_time=total_time)


def restore_case(run_config: RunConfig, case_config: CaseConfig, dry_run: bool = False):
    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend, precision = _init_backend(run_config, case_config)
    data = load_case_data(run_config.data_path, case_config, seed=run_config.seed)
    model = build_model(backend, case_config, data, precision=precision)

    checkpoint_dir = run_config.checkpoint_dir or str(output_dir / "Model-lbfgs")
    if not checkpoint_dir.endswith("/"):
        checkpoint_dir += "/"

    print(f"case={case_config.name!r} backend={backend} precision={precision} checkpoint_dir={checkpoint_dir}")
    if dry_run:
        print("--dry-run: config and data validated, skipping restore.")
        return None

    model.restore(path=checkpoint_dir)

    return _plot_and_report(run_config.data_path, case_config, data, model, output_dir)


def _resolve_x_source(case_config: CaseConfig, data: DataBundle):
    x_source = case_config.plotting.x_source
    if x_source == "x" or x_source not in data.extras:
        return data.domain_in[:, 0].flatten(), "x"
    return data.extras[x_source].flatten(), x_source


def _run_postprocess_hook(case_dir, case_config: CaseConfig, data, model, pred, star, x_plot, y_plot, output_dir):
    """Dynamically load and call a case-owned `run(...)` hook, if the case declares one via
    `postprocess_module` in case_config.json. Keeps case-specific plotting/geometry code
    (e.g. examples/phill/postprocess.py) living in the case's own directory instead of rstnets."""
    module_path = case_config.postprocess_module
    if module_path is None:
        return
    spec = importlib.util.spec_from_file_location(f"rstnets_postprocess_{case_config.name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise AttributeError(f"{module_path} must define a `run(...)` function to be used as postprocess_module")
    module.run(case_dir=case_dir, case_config=case_config, data=data, model=model,
               pred=pred, star=star, x_plot=x_plot, y_plot=y_plot, output_dir=output_dir)


def _plot_and_report(case_dir, case_config: CaseConfig, data: DataBundle, model, output_dir: Path,
                      total_time=None):
    fields = case_config.data.fields
    field_index = {f: i for i, f in enumerate(fields)}
    report_fields = [f for f in DEFAULT_REPORT_FIELDS if f in field_index]

    x_star, y_star = data.domain_in[:, 0:1], data.domain_in[:, 1:2]
    u_pred, v_pred, p_pred, uu_pred, uv_pred, vv_pred = model.predict(x_star, y_star)
    pred = dict(u=u_pred.flatten(), v=v_pred.flatten(), p=p_pred.flatten(),
                uu=uu_pred.flatten(), uv=uv_pred.flatten(), vv=vv_pred.flatten())
    star = {f: data.domain_data[:, field_index[f]] for f in fields}

    x_plot, x_label = _resolve_x_source(case_config, data)
    y_plot = y_star.flatten()
    plotting = case_config.plotting

    fig, axs = plt.subplots(len(report_fields), 2, figsize=(8, 1.6 * len(report_fields)), dpi=300)
    if len(report_fields) == 1:
        axs = axs[None, :]
    for row, field in enumerate(report_fields):
        levels = np.linspace(star[field].min(), star[field].max(), 51)
        for col, (label, values) in enumerate([("RSTnet", pred[field]), ("Reference", star[field])]):
            plot_contour(axs[row, col], x_plot, y_plot, values, levels=levels,
                         xlabel=x_label, axis_limits=plotting.axis_limits,
                         xticks=plotting.xticks, yticks=plotting.yticks, tick_stride=plotting.tick_stride)
            axs[row, col].set_title(f"{label}:{field}")
    plt.savefig(output_dir / "contour_compare.png")
    plt.close(fig)

    _run_postprocess_hook(case_dir, case_config, data, model, pred, star, x_plot, y_plot, output_dir)

    print("#########################################")
    errors = {}
    for field in report_fields:
        err = relative_l2(star[field], pred[field])
        errors[field] = err
        print(f"{field} l2: {err}")
    if total_time is not None:
        print(f"total training time: {total_time:.2f}s ({total_time / 60:.2f} min)")
        errors["training_time_seconds"] = total_time
    print("#########################################")
    return errors
