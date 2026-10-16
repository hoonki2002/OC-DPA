"""Where things live. Everything can be overridden by an environment variable.

  OCDPA_CACHE   encoded planes, routing traces, residual tables, results   (default ./cache)
  NEUROSIM_DIR  3D+NeuroSim V1.0 with the files of neurosim_patch/ built in (default third_party/3D_NeuroSim_V1.0)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("OCDPA_CACHE", "cache"))
NEUROSIM = Path(os.environ.get("NEUROSIM_DIR", REPO / "third_party" / "3D_NeuroSim_V1.0"))
PROBE = NEUROSIM / "Monolithic3D" / "NeuroSIM" / "probe"        # one analog column read, per cell type
PROBE3D = NEUROSIM / "Monolithic3D" / "NeuroSIM" / "probe3d"    # the same, split by memory/logic tier
ALLOCATOR = NEUROSIM / "Monolithic3D" / "NeuroSIM" / "allocator"  # precision allocator PPA
