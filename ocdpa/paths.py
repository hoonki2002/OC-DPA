"""Where things live. Everything can be overridden by an environment variable.

  OCDPA_CACHE   encoded planes, large traces, results, NeuroSim run cache   (default ./cache)
  NEUROSIM_DIR  3D+NeuroSim V1.0 with the files of neurosim_patch/ built in (default third_party/3D_NeuroSim_V1.0)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("OCDPA_CACHE", "cache"))
DATA = REPO / "data"


def data_file(name: str) -> Path:
    """An input of the hardware model: the copy in the cache if there is one (a fresh measurement), else data/."""
    return CACHE / name if (CACHE / name).exists() else DATA / name
NEUROSIM = Path(os.environ.get("NEUROSIM_DIR", REPO / "third_party" / "3D_NeuroSim_V1.0"))
PROBE = NEUROSIM / "Monolithic3D" / "NeuroSIM" / "probe"
PROBE3D = NEUROSIM / "Monolithic3D" / "NeuroSIM" / "probe3d"
ALLOCATOR = NEUROSIM / "Monolithic3D" / "NeuroSIM" / "allocator"
