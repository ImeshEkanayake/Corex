from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    shared = here.parent / "_shared_tabular_experiment.py"
    sys.argv = [str(shared), "--dataset", "meps", "--task", "privacy_utility", "--method", "corex_targeted", "--output-dir", str(here / "corex_targeted")]
    runpy.run_path(shared, run_name="__main__")
