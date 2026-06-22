#!/usr/bin/env python3
"""
Local Python driver for VS Code debugging with MACE.

Run THIS file in VS Code, not extra_worker_unified.py.

It uses absolute paths relative to this file, so it does not depend on the
current working directory.
"""

import os
import runpy
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

D0 = Path(os.environ.get("D0", SCRIPT_DIR / "debug_mlneb_mace")).resolve()

os.environ["D0"] = str(D0)
os.environ["EVAL_BACKEND"] = "mace"
os.environ["N_IMAGES"] = os.environ.get("N_IMAGES", "18")
os.environ["FMAX"] = os.environ.get("FMAX", "0.05")
os.environ["MAX_UNC"] = os.environ.get("MAX_UNC", "0.05")
os.environ["ML_STEPS"] = os.environ.get("ML_STEPS", "500")
os.environ["AL_STEPS"] = os.environ.get("AL_STEPS", "100")
os.environ["MACE_DEVICE"] = os.environ.get("MACE_DEVICE", "cpu")
os.environ["MACE_DTYPE"] = os.environ.get("MACE_DTYPE", "float64")
os.environ.pop("CATLEARN_USE_MPI4PY", None)

RESET_D0 = os.environ.get("RESET_D0", "0") == "1"

STATE0 = D0 / "catlearn_state.pkl"
STATE_AFTER = D0 / "catlearn_state.pkl"
PENDING = D0 / "pending_eval.traj"
CANDIDATES = D0 / "candidates.pkl"
META = D0 / "candidate_meta.pkl"


def run_script(script_path, args):
    """Run another Python script in-process so VS Code breakpoints work."""
    import sys

    script_path = str(Path(script_path).resolve())
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        os.chdir(SCRIPT_DIR)
        sys.argv = [script_path] + [str(a) for a in args]
        runpy.run_path(script_path, run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def count_candidates():
    import dill as pickle

    with open(CANDIDATES, "rb") as f:
        return len(pickle.load(f))


def main():
    if RESET_D0 and D0.exists():
        shutil.rmtree(D0)
    D0.mkdir(parents=True, exist_ok=True)

    workflow = SCRIPT_DIR / "mlneb_workflow_unified.py"
    worker = SCRIPT_DIR / "extra_worker_unified.py"

    print("===== prepare_state =====")
    run_script(workflow, ["prepare_state"])

    print("===== initial worker =====")
    run_script(worker, ["initial", STATE0, PENDING, CANDIDATES, META])

    print("===== initial eval input =====")
    os.environ.pop("CANDIDATE_INDEX", None)
    run_script(workflow, ["write_eval_input"])

    print("===== initial MACE eval =====")
    run_script(workflow, ["run_mace_eval"])

    print("===== load initial eval =====")
    run_script(workflow, ["load_eval"])
    run_script(workflow, ["check_convergence"])

    for al_step in range(1, int(os.environ["AL_STEPS"]) + 1):
        if (D0 / "MLNEB_DONE").exists():
            print("MLNEB_DONE found; stopping.")
            break

        print(f"===== AL_STEP {al_step}: next worker =====")
        os.environ["AL_STEP"] = str(al_step)
        run_script(worker, ["next", STATE_AFTER, PENDING, CANDIDATES, META])

        n_cand = count_candidates()
        print(f"NCAND={n_cand}")
        if n_cand == 0:
            print("No candidates; stopping.")
            break

        for candidate_index in range(n_cand):
            print(f"===== candidate {candidate_index} =====")
            os.environ["CANDIDATE_INDEX"] = str(candidate_index)
            run_script(workflow, ["write_eval_input"])
            run_script(workflow, ["run_mace_eval"])
            run_script(workflow, ["load_eval"])

        os.environ.pop("CANDIDATE_INDEX", None)
        run_script(workflow, ["check_convergence"])

    print("===== LOCAL DEBUG RUN FINISHED =====")


if __name__ == "__main__":
    main()
