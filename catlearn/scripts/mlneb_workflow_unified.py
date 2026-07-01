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

def _as_atoms(obj):
    return getattr(obj, "atoms", obj)

def _fixed_indices(atoms):
    fixed = set()
    for c in getattr(atoms, "constraints", []):
        if hasattr(c, "index"):
            idx = c.index
            try:
                fixed.update(int(i) for i in idx)
            except TypeError:
                fixed.add(int(idx))
    return tuple(sorted(fixed))

def _fixed_count(atoms):
    return len(_fixed_indices(atoms))

def _same_atom_order(a, b):
    return len(a) == len(b) and np.array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())

def _apply_reference_constraints(atoms, ref, ref_constraints=None, label="atoms"):
    atoms = _as_atoms(atoms)
    ref = _as_atoms(ref)

    if not _same_atom_order(atoms, ref):
        raise RuntimeError(f"{label}: atom count/order mismatch while applying constraints")

    if ref_constraints is None:
        ref_constraints = ref.constraints

    before = _fixed_count(atoms)
    expected = _fixed_count(ref)

    # Always set the exact reference constraints, not only when the count differs.
    # Same count but different FixAtoms object/index mask would still change the
    # GP fingerprint dimension or ordering in CatLearn.
    atoms.set_constraint(deepcopy(ref_constraints))

    after = _fixed_count(atoms)
    if before != after:
        print(f"constraint repair: {label} n_fixed {before} -> {after}", flush=True)

    if after != expected:
        raise RuntimeError(
            f"{label}: constraint repair failed; expected {expected} fixed atoms, got {after}"
        )

def repair_mlneb_internal_constraints(mlneb):
    """Repair constraints on all Atoms already stored inside the MLNEB object."""
    if not hasattr(mlneb, "structures") or not mlneb.structures:
        return

    ref = _as_atoms(mlneb.structures[0])
    ref_constraints = deepcopy(ref.constraints)

    for attr in ["structures", "best_structures", "prev_calculations", "images", "evaluated"]:
        objs = getattr(mlneb, attr, None)
        if objs is None:
            continue
        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        for i, obj in enumerate(objs):
            _apply_reference_constraints(
                _as_atoms(obj),
                ref,
                ref_constraints=ref_constraints,
                label=f"mlneb.{attr}[{i}]",
            )



def install_mlneb_constraint_guard(mlneb):
    """Temporarily guard CatLearn structure-copy calls that may drop constraints.

    The returned uninstall function MUST be called before pickling mlneb.

    Important:
    A plain repair_mlneb_internal_constraints(mlneb) is not sufficient here,
    because LocalNEB.copy_atoms/get_atoms_property may operate on a freshly
    copied local Atoms/Structure object before it is stored back into
    mlneb.best_structures.  Therefore this guard also repairs Atoms-like objects
    found in method args, kwargs and return values.
    """
    originals = []

    def get_ref():
        if not hasattr(mlneb, "structures") or not mlneb.structures:
            return None, None
        ref = _as_atoms(mlneb.structures[0])
        return ref, deepcopy(ref.constraints)

    def repair_any(obj, ref, ref_constraints, label, depth=0):
        if obj is None or ref is None:
            return

        # Avoid accidentally walking huge/nested arbitrary objects forever.
        if depth > 3:
            return

        atoms = _as_atoms(obj)
        if hasattr(atoms, "get_atomic_numbers") and hasattr(atoms, "set_constraint"):
            try:
                _apply_reference_constraints(
                    atoms,
                    ref,
                    ref_constraints=ref_constraints,
                    label=label,
                )
            except RuntimeError:
                # Real atom-order mismatch should still be fatal.
                raise
            except Exception as exc:
                raise RuntimeError(f"{label}: failed to repair constraints: {exc}") from exc
            return

        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                repair_any(item, ref, ref_constraints, f"{label}[{i}]", depth + 1)
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                # Candidate payload dicts usually store the actual Atoms under
                # "atoms"; repairing all values is still safe because non-Atoms
                # values are ignored.
                repair_any(value, ref, ref_constraints, f"{label}[{key!r}]", depth + 1)
            return

    def repair_call_objects(args, kwargs, label):
        ref, ref_constraints = get_ref()
        if ref is None:
            return

        for i, arg in enumerate(args):
            repair_any(arg, ref, ref_constraints, f"{label}.arg{i}")
        for key, value in kwargs.items():
            repair_any(value, ref, ref_constraints, f"{label}.kwarg[{key!r}]")

    def repair_result(result, label):
        ref, ref_constraints = get_ref()
        if ref is None:
            return result
        repair_any(result, ref, ref_constraints, f"{label}.return")
        return result

    def wrap_method(obj, method_name):
        if obj is None or not hasattr(obj, method_name):
            return

        try:
            current = getattr(obj, method_name)
        except Exception:
            return

        if getattr(current, "_mlneb_constraint_guard_wrapped", False):
            return

        def guarded(*args, **kwargs):
            label = f"{type(obj).__name__}.{method_name}"
            repair_mlneb_internal_constraints(mlneb)
            repair_call_objects(args, kwargs, label)

            result = current(*args, **kwargs)

            repair_result(result, label)
            repair_mlneb_internal_constraints(mlneb)
            return result

        try:
            guarded.__name__ = getattr(current, "__name__", method_name)
            guarded.__doc__ = getattr(current, "__doc__", None)
            guarded._mlneb_constraint_guard_wrapped = True
            setattr(obj, method_name, guarded)
        except Exception:
            return

        originals.append((obj, method_name, current))

    # MLNEB-level calls.
    for method_name in [
        "find_next_candidates",
        "find_next_candidate",
        "initiate_structure",
        "copy_best_structures",
        "get_structures",
    ]:
        wrap_method(mlneb, method_name)

    # Optimizer wrappers, commonly MLNEB.method = Sequential(...LocalNEB...).
    objects = []
    method = getattr(mlneb, "method", None)
    if method is not None:
        objects.append(method)
        inner = getattr(method, "method", None)
        if inner is not None:
            objects.append(inner)

    for obj in objects:
        for method_name in [
            "get_structures",
            "get_structures_parallel",
            "copy_atoms",
            "get_atoms_property",
        ]:
            wrap_method(obj, method_name)

    def uninstall_guard():
        for obj, method_name, original in reversed(originals):
            try:
                setattr(obj, method_name, original)
            except Exception:
                pass
        originals.clear()
        repair_mlneb_internal_constraints(mlneb)

    return uninstall_guard


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

