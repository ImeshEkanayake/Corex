from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    script = here / "optimize_corex_adaptive.py"
    sys.argv = [str(script), "--output-dir", str(here / "corex_adaptive" / "outputs")]
    runpy.run_path(script, run_name="__main__")
