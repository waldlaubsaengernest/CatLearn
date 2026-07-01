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
    repair_mlneb_database_atoms(mlneb)

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





def _repair_atoms_container_constraints(obj, ref, ref_constraints, label, depth=0):
    """Recursively repair Atoms/Structure objects inside common containers."""
    if obj is None:
        return 0
    if depth > 4:
        return 0

    repaired = 0

    atoms = _as_atoms(obj)
    if hasattr(atoms, "get_atomic_numbers") and hasattr(atoms, "set_constraint"):
        before = _fixed_count(atoms)
        _apply_reference_constraints(
            atoms,
            ref,
            ref_constraints=ref_constraints,
            label=label,
        )
        after = _fixed_count(atoms)
        return int(before != after)

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            repaired += _repair_atoms_container_constraints(
                item, ref, ref_constraints, f"{label}[{i}]", depth + 1
            )
        return repaired

    if isinstance(obj, tuple):
        # tuples are immutable, but contained Atoms objects are mutable.
        for i, item in enumerate(obj):
            repaired += _repair_atoms_container_constraints(
                item, ref, ref_constraints, f"{label}[{i}]", depth + 1
            )
        return repaired

    if isinstance(obj, dict):
        for key, value in obj.items():
            repaired += _repair_atoms_container_constraints(
                value, ref, ref_constraints, f"{label}[{key!r}]", depth + 1
            )
        return repaired

    return repaired


def repair_mlneb_database_atoms(mlneb):
    """Repair constraints on Atoms stored inside mlcalc.mlmodel.database.

    This is required for baseline correction: even if database.targets have the
    correct dimension, CatLearn recomputes y_base from atoms_list.  If those
    database atoms lost constraints, y_base becomes mixed-dimensional again.
    """
    if not hasattr(mlneb, "structures") or not mlneb.structures:
        return

    try:
        db = mlneb.mlcalc.mlmodel.database
    except Exception:
        return

    ref = _as_atoms(mlneb.structures[0])
    ref_constraints = deepcopy(ref.constraints)

    repaired = 0

    # First try common explicit attributes.
    for attr in [
        "atoms_list",
        "atoms",
        "structures",
        "data",
        "images",
        "candidates",
    ]:
        if hasattr(db, attr):
            try:
                value = getattr(db, attr)
            except Exception:
                continue
            repaired += _repair_atoms_container_constraints(
                value, ref, ref_constraints, f"database.{attr}"
            )

    # Then scan db.__dict__ conservatively for additional containers containing Atoms.
    for attr, value in getattr(db, "__dict__", {}).items():
        if attr in {"targets", "features"}:
            continue
        if attr.startswith("_"):
            continue
        if attr in ["atoms_list", "atoms", "structures", "data", "images", "candidates"]:
            continue
        if isinstance(value, (list, tuple, dict)):
            repaired += _repair_atoms_container_constraints(
                value, ref, ref_constraints, f"database.{attr}"
            )

    if repaired:
        print(f"database atom constraint repair: repaired {repaired} object(s)", flush=True)

def repair_mlneb_database_targets(mlneb):
    """Repair mixed target-vector dimensions in CatLearn's ML database.

    CatLearn force-training targets are vectors:
        [energy, force components for active/free atoms]

    If a newly added DFT result lost constraints, its target may become
        1 + 3 * natoms
    while the correct constrained target is
        1 + 3 * nfree

    This function converts all-atom target vectors to free-atom target vectors
    using mlneb.structures[0] as the constraint reference. It is a no-op for
    already clean states.
    """
    if not hasattr(mlneb, "structures") or not mlneb.structures:
        return

    ref = _as_atoms(mlneb.structures[0])
    fixed = set(_fixed_indices(ref))
    natoms = len(ref)
    free = [i for i in range(natoms) if i not in fixed]

    expected_dim = 1 + 3 * len(free)
    all_atom_dim = 1 + 3 * natoms

    db = None
    try:
        db = mlneb.mlcalc.mlmodel.database
    except Exception:
        return

    targets = getattr(db, "targets", None)
    if targets is None:
        return

    repaired = 0
    new_targets = []

    for i, target in enumerate(targets):
        arr = np.asarray(target, dtype=float).reshape(-1)

        if arr.size == expected_dim:
            new_targets.append(arr)
            continue

        if arr.size == all_atom_dim:
            energy = float(arr[0])
            forces_all = arr[1:].reshape(natoms, 3)
            forces_free = forces_all[free].reshape(-1)
            fixed_forces = forces_all[list(sorted(fixed))] if fixed else np.empty((0, 3))

            max_fixed_force = float(np.max(np.abs(fixed_forces))) if fixed_forces.size else 0.0
            print(
                f"database target repair: target {i} dim {arr.size} -> {expected_dim} "
                f"(removed fixed atom forces; max_fixed_force={max_fixed_force:.3e})",
                flush=True,
            )

            new_targets.append(np.concatenate(([energy], forces_free)))
            repaired += 1
            continue

        raise RuntimeError(
            f"database target {i} has unexpected dimension {arr.size}; "
            f"expected {expected_dim} or {all_atom_dim}"
        )

    if repaired:
        db.targets = new_targets
        print(f"database target repair: repaired {repaired} target(s)", flush=True)





def _get_feature_vector(fp):
    if fp is None:
        return None

    if hasattr(fp, "get_vector"):
        try:
            return np.asarray(fp.get_vector(), dtype=float).reshape(-1)
        except Exception:
            return None

    if isinstance(fp, np.ndarray):
        return np.asarray(fp, dtype=float).reshape(-1)

    return None


def _get_feature_derivatives(fp):
    if fp is None:
        return None

    if hasattr(fp, "get_derivatives"):
        try:
            return np.asarray(fp.get_derivatives(), dtype=float)
        except Exception:
            return None

    return None


class RepairedFeatureObject:
    """Minimal fallback wrapper for repaired feature vector and derivatives."""
    def __init__(self, vector, derivatives=None):
        self.vector = np.asarray(vector, dtype=float)
        self.derivatives = None if derivatives is None else np.asarray(derivatives, dtype=float)

    def get_vector(self):
        return self.vector

    def get_derivatives(self):
        if self.derivatives is None:
            raise AttributeError("No derivatives stored in RepairedFeatureObject")
        return self.derivatives


def _set_feature_vector(fp, vector, label):
    vector = np.asarray(vector, dtype=float)

    if hasattr(fp, "set_vector"):
        try:
            fp.set_vector(vector)
            return fp
        except Exception:
            pass

    for attr in ["vector", "features", "fingerprint", "fp"]:
        if hasattr(fp, attr):
            try:
                setattr(fp, attr, vector)
                return fp
            except Exception:
                pass

    return RepairedFeatureObject(vector, _get_feature_derivatives(fp))


def _set_feature_derivatives(fp, derivatives, label):
    derivatives = np.asarray(derivatives, dtype=float)

    if hasattr(fp, "set_derivatives"):
        try:
            fp.set_derivatives(derivatives)
            return fp
        except Exception:
            pass

    for attr in ["derivatives", "dfeatures", "fingerprint_derivatives", "fp_deriv", "deriv"]:
        if hasattr(fp, attr):
            try:
                setattr(fp, attr, derivatives)
                return fp
            except Exception:
                pass

    vec = _get_feature_vector(fp)
    if vec is not None:
        print(
            f"feature derivative repair: {label} has no known derivative setter; "
            "using RepairedFeatureObject fallback",
            flush=True,
        )
        return RepairedFeatureObject(vec, derivatives)

    return fp


def _shrink_feature_object_arrays(fp, keep_feature_indices, keep_coord_indices,
                                  all_pair_dim, all_coord_dim):
    """Shrink feature/derivative-like arrays inside a feature object.

    Conservative rule:
    - first axis equal all_pair_dim -> keep non fixed-fixed pair rows
    - second axis equal all_coord_dim -> keep free-atom coordinate columns

    This covers arrays used by get_vector/get_derivatives in common CatLearn
    feature objects without touching unrelated arrays.
    """
    changed = 0

    data = getattr(fp, "__dict__", None)
    if not isinstance(data, dict):
        return changed

    for attr, value in list(data.items()):
        try:
            arr = np.asarray(value)
        except Exception:
            continue

        new = arr
        did = False

        if new.ndim >= 1 and new.shape[0] == all_pair_dim:
            new = new[keep_feature_indices, ...]
            did = True

        if new.ndim >= 2 and new.shape[1] == all_coord_dim:
            new = new[:, keep_coord_indices, ...]
            did = True

        if did:
            try:
                setattr(fp, attr, new)
                changed += 1
            except Exception:
                pass

    return changed


def _repair_feature_derivatives(fp, keep_feature_indices, keep_coord_indices,
                                expected_feature_dim, all_pair_dim,
                                expected_coord_dim, all_coord_dim, label):
    deriv = _get_feature_derivatives(fp)
    if deriv is None:
        return fp, 0

    arr = np.asarray(deriv, dtype=float)

    if arr.ndim != 2:
        raise RuntimeError(
            f"{label}: unexpected derivative array rank {arr.ndim}; "
            f"shape={arr.shape}"
        )

    original_shape = arr.shape
    changed = False

    # CatLearn expects shape (n_features, n_force_components).
    if arr.shape[0] == all_pair_dim:
        arr = arr[keep_feature_indices, :]
        changed = True
    elif arr.shape[0] == expected_feature_dim:
        pass
    else:
        raise RuntimeError(
            f"{label}: unexpected derivative feature dimension {arr.shape[0]}; "
            f"expected {expected_feature_dim} or {all_pair_dim}; "
            f"full shape={original_shape}"
        )

    if arr.shape[1] == all_coord_dim:
        arr = arr[:, keep_coord_indices]
        changed = True
    elif arr.shape[1] == expected_coord_dim:
        pass
    else:
        raise RuntimeError(
            f"{label}: unexpected derivative coordinate dimension {arr.shape[1]}; "
            f"expected {expected_coord_dim} or {all_coord_dim}; "
            f"full shape={original_shape}"
        )

    if changed:
        fp = _set_feature_derivatives(fp, arr, label)
        print(
            f"feature derivative repair: {label} shape {original_shape} -> {arr.shape}",
            flush=True,
        )
        return fp, 1

    return fp, 0


def _repair_feature_item(fp, keep_feature_indices, keep_coord_indices,
                         expected_feature_dim, all_pair_dim,
                         expected_coord_dim, all_coord_dim, label):
    repaired = 0

    vec = _get_feature_vector(fp)
    if vec is not None:
        if vec.size == expected_feature_dim:
            pass
        elif vec.size == all_pair_dim:
            new_vec = vec[keep_feature_indices]
            fp = _set_feature_vector(fp, new_vec, label)
            print(
                f"feature repair: {label} dim {vec.size} -> {expected_feature_dim}",
                flush=True,
            )
            repaired += 1
        else:
            raise RuntimeError(
                f"{label}: unexpected feature dimension {vec.size}; "
                f"expected {expected_feature_dim} or {all_pair_dim}"
            )

    # Mutate any derivative-like object arrays before reading get_derivatives().
    _shrink_feature_object_arrays(
        fp,
        keep_feature_indices,
        keep_coord_indices,
        all_pair_dim,
        all_coord_dim,
    )

    fp, n = _repair_feature_derivatives(
        fp,
        keep_feature_indices,
        keep_coord_indices,
        expected_feature_dim,
        all_pair_dim,
        expected_coord_dim,
        all_coord_dim,
        label,
    )
    repaired += n

    return fp, repaired


def _repair_feature_container(obj, keep_feature_indices, keep_coord_indices,
                              expected_feature_dim, all_pair_dim,
                              expected_coord_dim, all_coord_dim,
                              label, depth=0):
    if obj is None or depth > 4:
        return obj, 0

    if isinstance(obj, list):
        repaired = 0
        for i, item in enumerate(obj):
            obj[i], n = _repair_feature_container(
                item, keep_feature_indices, keep_coord_indices,
                expected_feature_dim, all_pair_dim,
                expected_coord_dim, all_coord_dim,
                f"{label}[{i}]", depth + 1
            )
            repaired += n
        return obj, repaired

    if isinstance(obj, tuple):
        repaired = 0
        new_items = []
        for i, item in enumerate(obj):
            new_item, n = _repair_feature_container(
                item, keep_feature_indices, keep_coord_indices,
                expected_feature_dim, all_pair_dim,
                expected_coord_dim, all_coord_dim,
                f"{label}[{i}]", depth + 1
            )
            new_items.append(new_item)
            repaired += n
        return tuple(new_items), repaired

    if isinstance(obj, dict):
        repaired = 0
        for key, value in list(obj.items()):
            obj[key], n = _repair_feature_container(
                value, keep_feature_indices, keep_coord_indices,
                expected_feature_dim, all_pair_dim,
                expected_coord_dim, all_coord_dim,
                f"{label}[{key!r}]", depth + 1
            )
            repaired += n
        return obj, repaired

    fp, repaired = _repair_feature_item(
        obj,
        keep_feature_indices,
        keep_coord_indices,
        expected_feature_dim,
        all_pair_dim,
        expected_coord_dim,
        all_coord_dim,
        label,
    )
    return fp, repaired


def repair_mlneb_database_features(mlneb):
    """Repair mixed constrained/unconstrained stored fingerprint dimensions.

    For a 79 atom system with 36 fixed atoms:
        all_pair_dim = 79 * 78 / 2 = 3081
        expected_feature_dim = all_pair_dim - 36 * 35 / 2 = 2451
        all_coord_dim = 3 * 79 = 237
        expected_coord_dim = 3 * (79 - 36) = 129

    The repair removes:
    - fixed-fixed pair entries from feature vectors / derivative rows
    - fixed-atom coordinate components from derivative columns
    """
    if not hasattr(mlneb, "structures") or not mlneb.structures:
        return

    ref = _as_atoms(mlneb.structures[0])
    natoms = len(ref)
    fixed = set(_fixed_indices(ref))

    if not fixed:
        return

    all_pair_dim = natoms * (natoms - 1) // 2
    fixed_fixed_dim = len(fixed) * (len(fixed) - 1) // 2
    expected_feature_dim = all_pair_dim - fixed_fixed_dim

    all_coord_dim = 3 * natoms
    free_atoms = [i for i in range(natoms) if i not in fixed]
    expected_coord_dim = 3 * len(free_atoms)

    keep_feature_indices = []
    k = 0
    for i in range(natoms):
        for j in range(i + 1, natoms):
            if not (i in fixed and j in fixed):
                keep_feature_indices.append(k)
            k += 1
    keep_feature_indices = np.asarray(keep_feature_indices, dtype=int)

    keep_coord_indices = []
    for i in free_atoms:
        keep_coord_indices.extend([3 * i, 3 * i + 1, 3 * i + 2])
    keep_coord_indices = np.asarray(keep_coord_indices, dtype=int)

    repaired = 0

    objects = []

    try:
        db = mlneb.mlcalc.mlmodel.database
        objects.append(("database", db))
    except Exception:
        pass

    try:
        objects.append(("mlmodel", mlneb.mlcalc.mlmodel))
    except Exception:
        pass

    for obj_label, obj in objects:
        if hasattr(obj, "features"):
            value = getattr(obj, "features")
            new_value, n = _repair_feature_container(
                value,
                keep_feature_indices,
                keep_coord_indices,
                expected_feature_dim,
                all_pair_dim,
                expected_coord_dim,
                all_coord_dim,
                f"{obj_label}.features"
            )
            if n:
                setattr(obj, "features", new_value)
                repaired += n

        for attr, value in getattr(obj, "__dict__", {}).items():
            if attr == "features" or attr.startswith("_"):
                continue
            if "feature" not in attr.lower() and "finger" not in attr.lower():
                continue
            if not isinstance(value, (list, tuple, dict)):
                continue

            new_value, n = _repair_feature_container(
                value,
                keep_feature_indices,
                keep_coord_indices,
                expected_feature_dim,
                all_pair_dim,
                expected_coord_dim,
                all_coord_dim,
                f"{obj_label}.{attr}"
            )
            if n:
                try:
                    setattr(obj, attr, new_value)
                except Exception:
                    pass
                repaired += n

    if repaired:
        print(f"feature repair: repaired {repaired} feature/derivative object(s)", flush=True)

def repair_mlneb_training_state(mlneb):
    """Run all non-persistent MLNEB consistency repairs needed before training."""
    repair_mlneb_internal_constraints(mlneb)
    repair_mlneb_database_atoms(mlneb)
    repair_mlneb_database_features(mlneb)
    repair_mlneb_database_targets(mlneb)

def apply_mlneb_reference_constraints_to_atoms(mlneb, atoms, label="atoms"):
    """Apply mlneb.structures[0] constraints to one external Atoms object."""
    if not hasattr(mlneb, "structures") or not mlneb.structures:
        return atoms
    ref = _as_atoms(mlneb.structures[0])
    ref_constraints = deepcopy(ref.constraints)
    _apply_reference_constraints(
        _as_atoms(atoms),
        ref,
        ref_constraints=ref_constraints,
        label=label,
    )
    return atoms

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
    repair_mlneb_internal_constraints(mlneb)
    repair_mlneb_database_targets(mlneb)

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

