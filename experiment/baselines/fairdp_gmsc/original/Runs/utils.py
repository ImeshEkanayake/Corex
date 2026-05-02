from Runs.run_clean import run as run_clean
from Runs.run_smooth import run as run_smooth
from Runs.run_dpsgd import run as run_dpsgd
from Runs.run_dpsgdf import run as run_dpsgdf
from Runs.run_dpsgds import run as run_dpsgds
from Runs.run_fairdp import run as run_fairdp
from Runs.run_dpissgd import run as run_dpissgd

# from Runs.run_func import run as run_fm
# from Runs.run_mix import run as run_mix
# from Runs.run_mix_v2 import run as run_mix_v2


def init_run(gmode: str):

    run_dict = {
        "clean": run_clean,
        "smooth": run_smooth,
        "dpsgd": run_dpsgd,
        "dpsgdf": run_dpsgdf,
        "dpsgds": run_dpsgds,
        "dpissgd": run_dpissgd,
        "fairdp": run_fairdp,
        # "mixed": run_mix,
        # "mixed2": run_mix_v2,
    }

    return run_dict[gmode]
