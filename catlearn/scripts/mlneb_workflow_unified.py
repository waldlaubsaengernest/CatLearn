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
import shlex
from .calcfunctions import get_calcfunction

def load_user_module(path=None):
    if path is None:
        path = os.environ.get("CATLEARN_USER_MODULE")
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("catlearn_user_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

INPUT_FILE = os.environ.get("INPUT")
USER_MODULE = load_user_module()

def get_magmom():
    if USER_MODULE and hasattr(USER_MODULE, "get_magmom"):
        return USER_MODULE.get_magmom()
    return None

import inspect

from .adjust_constrains import (
    repair_mlneb_internal_constraints,
    repair_mlneb_database_atoms,
    repair_mlneb_database_targets,
    repair_mlneb_database_features,
    repair_mlneb_training_state,
    apply_mlneb_reference_constraints_to_atoms,
    install_mlneb_constraint_guard,
)

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
N_IMAGES = int(os.environ.get("N_IMAGES", "18"))
FMAX = float(os.environ.get("FMAX", "0.05"))
MAX_UNC = float(os.environ.get("MAX_UNC", "0.05"))
ML_STEPS = int(os.environ.get("ML_STEPS", "500"))
NEB_INTERPOLATION = os.environ.get(
    "NEB_INTERPOLATION", "idpp")

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
    
def get_calc_parameter(calc, name, default=None):
    keys = (name, name.lower(), name.upper())

    for attr in ("parameters", "int_params", "float_params", "string_params", "input_params"):
        data = getattr(calc, attr, None)
        if not isinstance(data, dict):
            continue

        for key in keys:
            value = data.get(key)
            if value is not None:
                return value

    return default

def phase_print_calc_env():
    calc = load_pickle(CALC_PKL)
    calcfunc = get_calcfunction(calc)

    for key, value in calcfunc.shell_env().items():
        print(f"export {key}={shlex.quote(str(value))}")

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

def atoms_has_energy_and_forces(atoms):
    try:
        atoms.get_potential_energy()
        atoms.get_forces()
        return True
    except Exception:
        return False

def build_mlneb_and_calc():
    os.makedirs(D0, exist_ok=True)
    if ".traj" in NEB_INTERPOLATION:
        inp_path       = read(NEB_INTERPOLATION)
        initial, final = inp_path[0], inp_path[-1]
    else:
        initial, final = get_endpoints()

    magmom        =  get_magmom()
    calc          = build_calc(magmom,D0)
    if calc is None:
        raise RuntimeError("No calculator returned. Define get_calculator() in CATLEARN_USER_MODULE.")
    if USER_MODULE and hasattr(USER_MODULE, "ensure_endpoint_results"): 
        USER_MODULE.ensure_endpoint_results(calc,D0)
        initial,final = get_endpoints()

    mlneb_calc = deepcopy(calc)
    mlneb_calc = get_calcfunction(calc).prepare_mlneb_calc(mlneb_calc, D0)
    
    restart = False
    if os.environ.get("RESTART", "0") == "1": restart=True
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
        restart=restart,
    )
    return mlneb, calc

def phase_prepare_state():
    if INPUT_FILE:
        inp = load_user_module(INPUT_FILE)
        calc = inp.calc
        mlneb = inp.mlneb
    else:
        mlneb, calc = build_mlneb_and_calc()
    
    if hasattr(mlneb,"restart") and mlneb.restart:
        os.environ["RESTART"] = "1"

    restart_file = os.path.join(D0, "MLNEB_RESTART_REQUESTED")

    if hasattr(mlneb, "restart") and mlneb.restart:
        os.environ["RESTART"] = "1"
        with open(restart_file, "w") as f:
            f.write("1\n")
        repair_mlneb_internal_constraints(mlneb)
    else:
        if os.path.exists(restart_file):
            os.remove(restart_file)

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

    calc = load_pickle(CALC_PKL)
    calcfunc = get_calcfunction(calc)
    calcfunc.write_input(atoms, eval_dir, calc=calc, user_module=USER_MODULE)

    with open(CURRENT_EVAL_DIR_TXT, "w") as f:
        f.write(eval_dir + "\n")

# Backward-compatible phase name.
phase_write_vasp_input = phase_write_eval_input

def get_current_eval_dir():
    with open(CURRENT_EVAL_DIR_TXT) as f:
        return f.read().strip()


def phase_run_singlepoint():
    calc = load_pickle(CALC_PKL)
    calcfunc = get_calcfunction(calc)
    calcfunc.run_singlepoint(get_current_eval_dir(), calc=calc)


def phase_run_mace_eval():
    # Backward-compatible phase name used by older local debug scripts.
    phase_run_singlepoint()


def phase_check_eval():
    calc = load_pickle(CALC_PKL)
    calcfunc = get_calcfunction(calc)
    calcfunc.check_eval(get_current_eval_dir())



def read_evaluated_atoms(eval_dir):
    calc = load_pickle(CALC_PKL)
    calcfunc = get_calcfunction(calc)
    return calcfunc.read_results(eval_dir)


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

    # VASP/vasprun/OUTCAR readers can lose ASE constraints.  This must happen
    # BEFORE finalize_external_evaluation(), otherwise CatLearn writes a target
    # vector with all-atom forces instead of free-atom forces.
    apply_mlneb_reference_constraints_to_atoms(
        mlneb,
        evaluated,
        label=f"evaluated result from {eval_dir}",
    )

    # Normalize the SinglePointCalculator payload after applying constraints.
    # Keep the raw all-atom forces in calc.results; CatLearn will select active
    # components according to constraints when building targets.
    energy = float(evaluated.get_potential_energy())
    forces = np.asarray(evaluated.get_forces(apply_constraint=False), dtype=float)
    if forces.shape != (len(evaluated), 3):
        raise RuntimeError(
            f"evaluated forces have wrong shape {forces.shape}; "
            f"expected {(len(evaluated), 3)} from {eval_dir}"
        )
    evaluated.calc = SinglePointCalculator(
        evaluated,
        energy=energy,
        forces=forces,
    )

    # Candidate evaluations have prediction payload; initial extra_initial_data does not.
    is_predicted = apply_prediction_payload(mlneb)

    mlneb.finalize_external_evaluation(evaluated, is_predicted=is_predicted)
    repair_mlneb_training_state(mlneb)

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
            "{prepare_state|write_eval_input|write_vasp_input|run_singlepoint|"
            "run_mace_eval|check_eval|load_eval|load_vasp_eval|"
            "count_candidates|check_convergence|print_calc_env}"
        )

    phase = sys.argv[1]
    if phase == "prepare_state":
        phase_prepare_state()
    elif phase == "write_eval_input":
        phase_write_eval_input()
    elif phase == "write_vasp_input":
        phase_write_vasp_input()
    elif phase == "run_singlepoint":
        phase_run_singlepoint()
    elif phase == "run_mace_eval":
        phase_run_mace_eval()
    elif phase == "check_eval":
        phase_check_eval()
    elif phase == "load_eval":
        phase_load_eval()
    elif phase == "load_vasp_eval":
        phase_load_vasp_eval()
    elif phase == "count_candidates":
        phase_count_candidates()
    elif phase == "check_convergence":
        phase_check_convergence()
    elif phase == "print_calc_env":
        phase_print_calc_env()
    else:
        raise SystemExit(f"Unknown phase: {phase}")


if __name__ == "__main__":
    main()

