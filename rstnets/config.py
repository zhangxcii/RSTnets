import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

SUPPORTED_BACKENDS = ("tensorflow", "torch")
SUPPORTED_PRECISIONS = ("float32", "float64")
SUPPORTED_LBFGS_BATCH_MODES = ("full", "batch")


@dataclasses.dataclass
class NetworkConfig:
    layers: List[int]
    learning_rate: float = 0.001


@dataclasses.dataclass
class TrainingDefaults:
    n_iters_adam: int
    n_lbfgs: int
    batch_data: str = "full"
    batch_pi_divisor: int = 1000
    print_interval: int = 100
    lbfgs_options: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"maxcor": 100, "ftol": "eps", "maxiter": 200, "maxls": 50})
    lbfgs_batch_mode: str = "full"
    lbfgs_batch_data: Optional[str] = None
    lbfgs_batch_pi_divisor: Optional[int] = None


@dataclasses.dataclass
class DataConfig:
    domain_data: str = "domain_data.npy"
    domain_in: str = "domain_in.npy"
    bc_data: str = "bc_data.npy"
    bc_in: str = "bc_in.npy"
    fields: List[str] = dataclasses.field(default_factory=lambda: ["u", "v", "p", "uu", "uv", "vv"])
    extras: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PlottingConfig:
    x_source: str = "x"
    axis_limits: Optional[Dict[str, List[float]]] = None
    xticks: Optional[List[float]] = None
    yticks: Optional[List[float]] = None
    tick_stride: int = 25


@dataclasses.dataclass
class CaseConfig:
    name: str
    invRe: float
    network: NetworkConfig
    precision: str
    use_coord_scaling: bool
    supervise_pressure: bool
    if_positivity: list
    if_realisability: bool
    training_defaults: TrainingDefaults
    data: DataConfig
    plotting: PlottingConfig
    backend: str = "tensorflow"
    description: str = ""
    postprocess_module: Optional[Path] = None
    custom: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict, case_dir=None) -> "CaseConfig":
        missing_required = [
            field for field, value in (("physics.invRe", raw.get("physics", {}).get("invRe")),
                                        ("description", raw.get("description")))
            if value == "REQUIRED" or value is None
        ]
        if missing_required:
            raise ValueError(
                f"case_config.json is missing required field(s): {', '.join(missing_required)}")

        network = NetworkConfig(**raw["network"])
        training_defaults = TrainingDefaults(**raw["training_defaults"])
        if training_defaults.lbfgs_batch_mode not in SUPPORTED_LBFGS_BATCH_MODES:
            raise ValueError(f"training_defaults.lbfgs_batch_mode must be one of "
                              f"{SUPPORTED_LBFGS_BATCH_MODES}, got {training_defaults.lbfgs_batch_mode!r}")
        data = DataConfig(**raw.get("data", {}))
        plotting = PlottingConfig(**raw.get("plotting", {}))

        precision = raw["precision"]
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"precision must be one of {SUPPORTED_PRECISIONS}, got {precision!r}")

        backend = raw.get("backend", "tensorflow")
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"backend must be one of {SUPPORTED_BACKENDS}, got {backend!r}")

        postprocess_module = raw.get("postprocess_module")
        if postprocess_module is not None:
            if case_dir is None:
                raise ValueError("postprocess_module is relative to the case directory; "
                                  "case_dir must be provided to resolve it")
            postprocess_module = (Path(case_dir) / postprocess_module).resolve()
            if not postprocess_module.exists():
                raise FileNotFoundError(f"postprocess_module not found: {postprocess_module}")

        return cls(
            name=raw["name"],
            description=raw.get("description", ""),
            invRe=raw["physics"]["invRe"],
            network=network,
            precision=precision,
            use_coord_scaling=raw["coordinate_scaling"]["enabled"],
            supervise_pressure=raw["loss"]["supervise_pressure"],
            if_positivity=raw["constraints"]["if_positivity"],
            if_realisability=raw["constraints"]["if_realisability"],
            training_defaults=training_defaults,
            data=data,
            plotting=plotting,
            backend=backend,
            postprocess_module=postprocess_module,
            custom=raw.get("custom", {}),
        )

    @classmethod
    def load(cls, case_dir) -> "CaseConfig":
        case_dir = Path(case_dir)
        path = case_dir / f"{case_dir.name}_config.json"
        if not path.exists():
            # fall back to a single unambiguous *_config.json, in case the folder was
            # renamed without also renaming its config file
            candidates = list(case_dir.glob("*_config.json"))
            if len(candidates) == 1:
                path = candidates[0]
            else:
                raise FileNotFoundError(
                    f"No {case_dir.name}_config.json found in {case_dir} "
                    f"(and no single unambiguous *_config.json fallback)")
        with open(path) as f:
            return cls.from_dict(json.load(f), case_dir=case_dir)


@dataclasses.dataclass
class RunConfig:
    data_path: str
    output_dir: str
    backend: Optional[str] = None
    precision: Optional[str] = None
    n_iters_adam: Optional[int] = None
    n_lbfgs: Optional[int] = None
    checkpoint_dir: Optional[str] = None
    seed: int = 0
    gpu: bool = True
    print_interval: Optional[int] = None

    def __post_init__(self):
        # backend may be None here (meaning "defer to case_config.backend")
        if self.backend is not None and self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"backend must be one of {SUPPORTED_BACKENDS}, got {self.backend!r}")
        if self.precision is not None and self.precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"precision must be one of {SUPPORTED_PRECISIONS}, got {self.precision!r}")

    @classmethod
    def load(cls, path) -> "RunConfig":
        with open(path) as f:
            raw = json.load(f)
        return cls(**raw)


def effective_value(run_value, case_value):
    """run_config overrides case_config default when not None."""
    return case_value if run_value is None else run_value


# Resolved from this file's own location, not the caller's cwd, so examples/ always resolves
# the same regardless of how or from where an entry-point script is invoked.
REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def _looks_like_a_path(value: str) -> bool:
    """True if `value` should be treated as an explicit path rather than a bare examples/
    name, i.e. it names a location, not just an identifier (contains a separator, or is
    an absolute/./..-relative reference)."""
    p = Path(value)
    return p.is_absolute() or len(p.parts) > 1 or value in (".", "..")


def resolve_example(name_or_path) -> Path:
    """Resolve a `--case`/positional value to an example directory.

    A bare name (no path separators, e.g. "phill") ALWAYS resolves to
    <repo_root>/examples/<name>/, regardless of the caller's cwd - so the command behaves
    identically from any directory. A value that looks like a path (contains a separator, or is
    absolute/./..-relative) is instead resolved as a real filesystem path (relative to cwd if
    not absolute) - the escape hatch for a case outside examples/.
    """
    if _looks_like_a_path(name_or_path):
        example_dir = Path(name_or_path).resolve()
        if example_dir.is_dir():
            return example_dir
        raise FileNotFoundError(f"No such directory: {example_dir}")

    example_dir = EXAMPLES_DIR / name_or_path
    if example_dir.is_dir():
        return example_dir
    raise FileNotFoundError(
        f"Could not resolve {name_or_path!r} as an examples/ name (looked for {example_dir}). "
        f"Pass an explicit path (e.g. './{name_or_path}' or '../{name_or_path}') to point "
        f"outside examples/.")
