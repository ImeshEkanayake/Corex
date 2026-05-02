from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    script = here / "run_table2_privacy_utility.py"
    sys.argv = [str(script), "--output-dir", str(here / "lime_targeted" / "outputs")]
    runpy.run_path(script, run_name="__main__")
