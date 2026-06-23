#!/usr/bin/env python3
"""
Unified MLNEB phase workflow for both:
  1) local debugging with MACE
  2) cluster/VASP workflow with external `srun vasp_std`

Key design:
- CatLearn proposes evaluations in a worker.
- This workflow writes inputs, runs/loads external evaluations, and updates state.
- Candidate prediction data is carried via candidates.pkl payloads:
    {"atoms": Atoms, "energy_pred": ..., "unc": ...}
"""

import os
import sys
import shutil
from copy import deepcopy
import dill as pickle
import numpy as np
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from catlearn.activelearning.mlneb import MLNEB
import importlib.util

def load_user_module():
    path = os.environ.get("CATLEARN_USER_MODULE")
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("catlearn_user_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

USER_MODULE = load_user_module()

def get_magmom():
    if USER_MODULE and hasattr(USER_MODULE, "get_magmom"):
        return USER_MODULE.get_magmom()
    return None

import inspect

def build_calc(magmom=None, workdir=None):
    if USER_MODULE and hasattr(USER_MODULE, "get_calculator"):
        func = USER_MODULE.get_calculator

        kwargs = {}
        sig = inspect.signature(func)

        if "magmom" in sig.parameters:
            kwargs["magmom"] = magmom

        if "workdir" in sig.parameters:
            kwargs["workdir"] = workdir

        return func(**kwargs)

    return None

def get_endpoints():
    if USER_MODULE and hasattr(USER_MODULE, "get_endpoints"):
        return USER_MODULE.get_endpoints()
    return read("initial.traj"), read("final.traj")

def require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set.")
    return value


def bool_env(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


D0 = os.environ.get("D0")
if D0 is None:
    raise RuntimeError("D0 is not set. Export D0 or pass it to run_mlneb_core.sh.")
EVAL_BACKEND = os.environ.get("EVAL_BACKEND", "vasp").lower()  # "vasp" or "mace"

N_IMAGES = int(os.environ.get("N_IMAGES", "18"))
FMAX = float(os.environ.get("FMAX", "0.05"))
MAX_UNC = float(os.environ.get("MAX_UNC", "0.05"))
ML_STEPS = int(os.environ.get("ML_STEPS", "500"))
NEB_INTERPOLATION = os.environ.get("NEB_INTERPOLATION", "idpp")

STATE_PKL = os.path.join(D0, "catlearn_state.pkl")
STATE_AFTER_EVAL_PKL = os.path.join(D0, "catlearn_state.pkl")
# Alias used by old cluster run.sh versions:
STATE_AFTER_VASP_PKL = os.path.join(D0, "catlearn_state.pkl")

CALC_PKL = os.path.join(D0, "ase_calc.pkl")
PENDING_TRAJ = os.path.join(D0, "pending_eval.traj")
CANDIDATES_PKL = os.path.join(D0, "candidates.pkl")
CANDIDATE_META_PKL = os.path.join(D0, "candidate_meta.pkl")
DONE_FILE = os.path.join(D0, "MLNEB_DONE")
CURRENT_EVAL_DIR_TXT = os.path.join(D0, "current_eval_dir.txt")

def dump_atomic(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_state(mlneb):
    dump_atomic(mlneb, STATE_AFTER_EVAL_PKL)
    # Keep the VASP-named file too, so older run.sh continues to work.
    dump_atomic(mlneb, STATE_AFTER_VASP_PKL)

def load_state():
    state = os.environ.get("STATE_IN")
    if state is None:
        if os.path.exists(STATE_AFTER_EVAL_PKL):
            state = STATE_AFTER_EVAL_PKL
        elif os.path.exists(STATE_AFTER_VASP_PKL):
            state = STATE_AFTER_VASP_PKL
        else:
            state = STATE_PKL
    with open(state, "rb") as f:
        return pickle.load(f)

def build_mlneb_and_calc():
    os.makedirs(D0, exist_ok=True)
    initial,final = get_endpoints()
    magmom        =  get_magmom()
    calc          = build_calc(magmom,D0)
    if calc is None:
        raise RuntimeError("No calculator returned. Define get_calculator() in CATLEARN_USER_MODULE.")
    if USER_MODULE and hasattr(USER_MODULE, "ensure_endpoint_results"): 
        USER_MODULE.ensure_endpoint_results(calc,D0)

    mlneb_calc = deepcopy(calc)
    if EVAL_BACKEND == "vasp":
        mlneb_calc.directory = os.path.join(D0, "mlneb")

    mlneb = MLNEB(
        start=initial,
        end=final,
        ase_calc=mlneb_calc,
        neb_interpolation=NEB_INTERPOLATION,
        n_images=N_IMAGES + 2,
        climb=True,
        start_without_ci=True,
        reuse_ci_path=True,
        unc_convergence=MAX_UNC,
        use_restart=True,
        check_unc=True,
        verbose=True,
        local_opt_kwargs=dict(logfile=os.path.join(D0, "mlneb_opt.log")),
        parallel_run=True,
        parallel_eval=False,
        seed=1,
    )
    return mlneb, calc


def phase_prepare_state():
    mlneb, calc = build_mlneb_and_calc()
    dump_atomic(mlneb, STATE_PKL)
    dump_atomic(calc, CALC_PKL)

def phase_count_candidates():
    payload = load_pickle(CANDIDATES_PKL)
    print(len(payload))

def get_eval_dir(candidate_index):
    if candidate_index is None:
        return os.path.join(D0, "external_eval_initial")
    return os.path.join(D0, f"external_eval_{int(candidate_index):04d}")


def clean_eval_dir(eval_dir):
    # For VASP this prevents stale CHGCAR/WAVECAR/OUTCAR reuse.
    # For local MACE it also keeps candidate directories unambiguous.
    if bool_env("CLEAN_EVAL_DIR", True) and os.path.isdir(eval_dir):
        shutil.rmtree(eval_dir)
    os.makedirs(eval_dir, exist_ok=True)


def select_atoms_for_evaluation():
    candidate_index = os.environ.get("CANDIDATE_INDEX")
    if candidate_index is None:
        atoms = read(PENDING_TRAJ)
    else:
        payload = load_pickle(CANDIDATES_PKL)
        item = payload[int(candidate_index)]
        atoms = item["atoms"] if isinstance(item, dict) else item
        write(PENDING_TRAJ, atoms)
    return atoms, candidate_index


def phase_write_eval_input():
    atoms, candidate_index = select_atoms_for_evaluation()
    eval_dir = get_eval_dir(candidate_index)
    clean_eval_dir(eval_dir)

    if EVAL_BACKEND == "vasp":
        calc = load_pickle(CALC_PKL)
        atoms.calc = deepcopy(calc)
        atoms.calc.directory = eval_dir
        com = atoms.get_center_of_mass(scaled=True)
        atoms.calc.dipol = (com[0], com[1], com[2])
        atoms.calc.write_input(atoms)

    # In both modes keep an explicit input structure.
    write(os.path.join(eval_dir, "input_atoms.traj"), atoms)

    with open(CURRENT_EVAL_DIR_TXT, "w") as f:
        f.write(eval_dir + "\n")

# Backward-compatible phase name.
phase_write_vasp_input = phase_write_eval_input

def phase_run_mace_eval():
    if EVAL_BACKEND != "mace":
        raise RuntimeError("run_mace_eval phase requires EVAL_BACKEND=mace")

    calc = load_pickle(CALC_PKL)
    with open(CURRENT_EVAL_DIR_TXT) as f:
        eval_dir = f.read().strip()

    atoms = read(os.path.join(eval_dir, "input_atoms.traj"))
    atoms.calc = deepcopy(calc)

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    write(os.path.join(eval_dir, "evaluated.traj"), atoms)

def read_evaluated_atoms(eval_dir):
    if EVAL_BACKEND == "mace":
        path = os.path.join(eval_dir, "evaluated.traj")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return read(path)

    vasprun = os.path.join(eval_dir, "vasprun.xml")
    outcar = os.path.join(eval_dir, "OUTCAR")
    if os.path.exists(vasprun):
        return read(vasprun)
    if os.path.exists(outcar):
        return read(outcar)
    raise FileNotFoundError(f"Neither vasprun.xml nor OUTCAR found in {eval_dir}")


def apply_prediction_payload(mlneb):
    """Attach the GP prediction/uncertainty belonging to CANDIDATE_INDEX.

    This replaces the original evaluate_candidates -> broadcast_predictions()
    coupling, but does it with values saved by the worker while still in the
    original candidate-generation context.
    """
    candidate_index = os.environ.get("CANDIDATE_INDEX")
    if candidate_index is None:
        return False

    payload = load_pickle(CANDIDATES_PKL)
    item = payload[int(candidate_index)]

    if isinstance(item, dict):
        mlneb.energy_pred = item.get("energy_pred", np.nan)
        mlneb.unc = item.get("unc", np.nan)
        if "pred_energies" in item:
            mlneb.pred_energies = item["pred_energies"]
        if "uncertainties" in item:
            mlneb.uncertainties = item["uncertainties"]

    return True


def phase_load_eval():
    mlneb = load_state()

    with open(CURRENT_EVAL_DIR_TXT) as f:
        eval_dir = f.read().strip()

    evaluated = read_evaluated_atoms(eval_dir)

    # Candidate evaluations have prediction payload; initial extra_initial_data does not.
    is_predicted = apply_prediction_payload(mlneb)

    mlneb.finalize_external_evaluation(evaluated, is_predicted=is_predicted)

    mlneb.print_statement()
    save_state(mlneb)

# Backward-compatible phase name.
phase_load_vasp_eval = phase_load_eval

def phase_check_convergence():
    mlneb = load_state()

    fmax = float(os.environ.get("FMAX", "0.05"))

    method_converged = False
    if os.path.exists(CANDIDATE_META_PKL):
        meta = load_pickle(CANDIDATE_META_PKL)
        method_converged = bool(meta.get("method_converged", False))

    converged = bool(mlneb.check_convergence(
        fmax,
        method_converged,
    ))

    if converged:
        with open(DONE_FILE, "w") as f:
            f.write("converged\n")

def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: mlneb_workflow_unified.py "
            "{prepare_state|write_eval_input|write_vasp_input|run_mace_eval|"
            "load_eval|load_vasp_eval|count_candidates|check_convergence}"
        )

    phase = sys.argv[1]
    if phase == "prepare_state":
        phase_prepare_state()
    elif phase == "write_eval_input":
        phase_write_eval_input()
    elif phase == "write_vasp_input":
        phase_write_vasp_input()
    elif phase == "run_mace_eval":
        phase_run_mace_eval()
    elif phase == "load_eval":
        phase_load_eval()
    elif phase == "load_vasp_eval":
        phase_load_vasp_eval()
    elif phase == "count_candidates":
        phase_count_candidates()
    elif phase == "check_convergence":
        phase_check_convergence()
    else:
        raise SystemExit(f"Unknown phase: {phase}")


if __name__ == "__main__":
    main()

