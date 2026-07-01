#!/usr/bin/env python3
"""
Constraint and CatLearn MLNEB cache repair utilities.

This module centralizes all repairs related to:
  - ASE constraints lost by VASP/ASE readers or CatLearn copies
  - MLNEB internal structures
  - CatLearn database atoms
  - CatLearn database targets
  - CatLearn fingerprint vectors and derivatives
  - temporary find_next_candidates guard

The functions are intentionally data-only except for install_mlneb_constraint_guard(),
which must be uninstalled before pickling MLNEB state.
"""

from copy import deepcopy

import numpy as np

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


def _iter_atoms_like(obj, depth=0):
    """Yield Atoms-like objects from nested CatLearn/ASE containers."""
    if obj is None or depth > 5:
        return

    atoms = _as_atoms(obj)
    if hasattr(atoms, "get_atomic_numbers") and hasattr(atoms, "set_constraint"):
        yield atoms
        return

    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_atoms_like(item, depth + 1)
        return

    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_atoms_like(value, depth + 1)
        return


def get_mlneb_constraint_reference(mlneb, require_fixed=True):
    """Find a constrained Atoms object to use as FixAtoms reference."""
    candidates = []

    for attr in ["structures", "best_structures", "prev_calculations", "images", "evaluated"]:
        for atoms in _iter_atoms_like(getattr(mlneb, attr, None)):
            candidates.append((f"mlneb.{attr}", atoms))

    try:
        db = mlneb.mlcalc.mlmodel.database
    except Exception:
        db = None

    if db is not None:
        for attr in ["atoms_list", "atoms", "structures", "data", "images", "candidates"]:
            for atoms in _iter_atoms_like(getattr(db, attr, None)):
                candidates.append((f"database.{attr}", atoms))

        for attr, value in getattr(db, "__dict__", {}).items():
            if attr in {"targets", "features"} or attr.startswith("_"):
                continue
            if attr in ["atoms_list", "atoms", "structures", "data", "images", "candidates"]:
                continue
            if isinstance(value, (list, tuple, dict)):
                for atoms in _iter_atoms_like(value):
                    candidates.append((f"database.{attr}", atoms))

    best_label = None
    best_atoms = None
    best_nfixed = -1

    for label, atoms in candidates:
        nfix = _fixed_count(atoms)
        if nfix > best_nfixed:
            best_label = label
            best_atoms = atoms
            best_nfixed = nfix

    if best_atoms is None:
        raise RuntimeError("could not find any Atoms object in MLNEB state for constraint reference")

    if require_fixed and best_nfixed <= 0:
        raise RuntimeError(
            "could not find a constrained reference Atoms object in MLNEB state; "
            "all inspected Atoms have n_fixed=0. Do not continue silently with "
            "expected_dim == all_atom_dim."
        )

    if best_label != "mlneb.structures" or best_nfixed <= 0:
        print(f"constraint reference: using {best_label} with n_fixed={best_nfixed}", flush=True)

    return best_atoms


def get_mlneb_reference_constraints(mlneb, require_fixed=True):
    ref = get_mlneb_constraint_reference(mlneb, require_fixed=require_fixed)
    return ref, deepcopy(ref.constraints)

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

    ref, ref_constraints = get_mlneb_reference_constraints(mlneb)

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

    ref, ref_constraints = get_mlneb_reference_constraints(mlneb)

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

    ref, _ref_constraints = get_mlneb_reference_constraints(mlneb)
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

    ref, _ref_constraints = get_mlneb_reference_constraints(mlneb)
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
    ref, ref_constraints = get_mlneb_reference_constraints(mlneb)
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
        try:
            return get_mlneb_reference_constraints(mlneb)
        except Exception:
            return None, None

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
