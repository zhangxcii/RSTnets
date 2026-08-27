import dataclasses
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import CaseConfig


@dataclasses.dataclass
class DataBundle:
    train_dd: np.ndarray        # [x, y, u, v, p, uu, uv, vv] sparse boundary/supervised set
    train_pi: np.ndarray        # [x, y] full-domain physics collocation set
    domain_data: np.ndarray     # [u, v, p, uu, uv, vv] full-domain reference values (plotting/error)
    domain_in: np.ndarray       # [x, y] full-domain coordinates
    Lxscale: Optional[float]
    Lyscale: Optional[float]
    extras: Dict[str, np.ndarray]


def load_case_data(case_dir, case_config: CaseConfig, seed: Optional[int] = None) -> DataBundle:
    case_dir = Path(case_dir)
    data_dir = case_dir / "data"
    dc = case_config.data

    domain_data = np.load(data_dir / dc.domain_data)
    domain_in = np.load(data_dir / dc.domain_in)
    bc_data = np.load(data_dir / dc.bc_data)
    bc_in = np.load(data_dir / dc.bc_in)

    extras = {}
    for key, filename in dc.extras.items():
        path = data_dir / filename
        if path.exists():
            extras[key] = np.load(path)

    x, y = domain_in[:, 0:1], domain_in[:, 1:2]
    train_pi = np.concatenate([x, y], axis=1)

    if case_config.use_coord_scaling:
        Lxscale = float(train_pi[:, 0].max() - train_pi[:, 0].min())
        Lyscale = float(train_pi[:, 1].max() - train_pi[:, 1].min())
    else:
        Lxscale, Lyscale = None, None

    xbc, ybc = bc_in[:, 0:1], bc_in[:, 1:2]
    train_dd = np.concatenate([xbc, ybc, bc_data], axis=1)

    rng = np.random.default_rng(seed)
    rng.shuffle(train_pi)
    rng.shuffle(train_dd)

    return DataBundle(
        train_dd=train_dd, train_pi=train_pi,
        domain_data=domain_data, domain_in=domain_in,
        Lxscale=Lxscale, Lyscale=Lyscale, extras=extras,
    )
