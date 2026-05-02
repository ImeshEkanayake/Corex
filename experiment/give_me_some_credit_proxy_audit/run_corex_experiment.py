from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    script = here / "run_corex_experiments.py"
    sys.argv = [str(script), "--output-dir", str(here / "corex" / "outputs")]
    runpy.run_path(script, run_name="__main__")
