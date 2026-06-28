#!/usr/bin/env python3
"""
Local Python driver for debugging the MLNEB workflow with a Python/ASE calculator
such as MACE.

This mirrors run_mlneb_core.sh as closely as possible, but runs the Python
workflow/worker scripts in-process so VS Code breakpoints work.

Important differences to the old local script:
  - It does NOT require copying mlneb_workflow_unified.py or extra_worker_unified.py
    next to this file.
  - It accepts the same INPUT argument style as run_mlneb_core.sh:
        python run_local_mace.py input.py
    or:
        INPUT=input.py python run_local_mace.py
  - It also sets CATLEARN_USER_MODULE=INPUT for compatibility with helper code.
  - D0 defaults to the current working directory, like the shell script.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional


START_CWD = Path.cwd().resolve()
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (START_CWD / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def setup_environment(argv: list[str]) -> None:
    input_arg = argv[0] if argv else None

    input_file = (
        resolve_optional_path(input_arg)
        or resolve_optional_path(os.environ.get("INPUT"))
    )

    # Debug convenience: if neither INPUT nor argv is given, but an input.py is
    # present in the current working directory, use it. This avoids the common
    # "No calculator returned" failure when launching directly from VS Code.
    if input_file is None:
        cwd_input = START_CWD / "input.py"
        if cwd_input.exists():
            input_file = str(cwd_input.resolve())

    if input_file is not None:
        os.environ["INPUT"] = input_file
        # The workflow primarily uses INPUT, but some helper code/user modules
        # may still expect CATLEARN_USER_MODULE.
        os.environ.setdefault("CATLEARN_USER_MODULE", input_file)

    if "INPUT" not in os.environ and "CATLEARN_USER_MODULE" not in os.environ:
        raise RuntimeError(
            "No input file configured. Use one of:\n"
            "  python run_local_mace.py input.py\n"
            "  INPUT=input.py python run_local_mace.py\n"
            "  CATLEARN_USER_MODULE=input.py python run_local_mace.py"
        )

    d0 = resolve_optional_path(os.environ.get("D0")) or str(START_CWD)
    os.environ["D0"] = d0

    # Same workflow defaults as run_mlneb_core.sh, plus MACE debug defaults.
    os.environ["N_IMAGES"] = os.environ.get("N_IMAGES", "18")
    os.environ["FMAX"] = os.environ.get("FMAX", "0.05")
    os.environ["MAX_UNC"] = os.environ.get("MAX_UNC", "0.05")
    os.environ["ML_STEPS"] = os.environ.get("ML_STEPS", "500")
    os.environ["AL_STEPS"] = os.environ.get("AL_STEPS", "100")
    os.environ["CLEAN_EVAL_DIR"] = os.environ.get("CLEAN_EVAL_DIR", "1")
    os.environ["RESTART"] = os.environ.get("RESTART", "0")
    os.environ["MACE_DEVICE"] = os.environ.get("MACE_DEVICE", "cpu")
    os.environ["MACE_DTYPE"] = os.environ.get("MACE_DTYPE", "float64")
    os.environ["MACE_DEBUG_FAIL"]="1"
    os.environ["MACE_DEBUG_FAIL_MODE"]="random"
    os.environ["MACE_DEBUG_FAIL_PROB"]="0.3"
    # Compatibility with older mlneb_workflow_unified.py versions, which still
    # branch on EVAL_BACKEND for run_mace_eval/load_eval.
    os.environ["EVAL_BACKEND"] = os.environ.get("EVAL_BACKEND", "mace")

    # New generic name, old VASP name kept as compatibility alias because some
    # worker versions still look for VASP_FALLBACK_CANDIDATES.
    fallback_candidates = os.environ.get(
        "FALLBACK_CANDIDATES",
        os.environ.get("VASP_FALLBACK_CANDIDATES", "8"),
    )
    os.environ["FALLBACK_CANDIDATES"] = fallback_candidates
    os.environ.setdefault("VASP_FALLBACK_CANDIDATES", fallback_candidates)
    os.environ["FALLBACK_ORDER"] = os.environ.get("FALLBACK_ORDER", "uncertainty")

    # Local debug should not accidentally start mpi4py because of a cluster env.
    os.environ.pop("CATLEARN_USE_MPI4PY", None)


def find_script(env_name: str, module_name: str, filename: str) -> Path:
    override = os.environ.get(env_name)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"{env_name} points to missing file: {path}")
        return path

    spec = importlib.util.find_spec(module_name)
    if spec is not None and spec.origin:
        path = Path(spec.origin).resolve()
        if path.exists():
            return path

    candidates = [
        START_CWD / filename,
        START_CWD / "catlearn" / "scripts" / filename,
        SCRIPT_DIR / filename,
        SCRIPT_DIR / "catlearn" / "scripts" / filename,
        SCRIPT_DIR / "scripts" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    raise RuntimeError(
        f"Could not find {filename}. Set {env_name} explicitly, e.g.\n"
        f"  export {env_name}=/path/to/{filename}"
    )


def script_text_has_phase(script_path: Path, phase: str) -> bool:
    try:
        text = script_path.read_text()
    except Exception:
        return False

    patterns = (
        f'phase == "{phase}"',
        f"phase == '{phase}'",
        f"def phase_{phase}(",
    )
    return any(pattern in text for pattern in patterns)


def prepend_sys_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def run_script(script_path: Path, args: list[object], *, capture_stdout: bool = False) -> str:
    """Run another Python script in-process so VS Code breakpoints work."""
    script_path = script_path.resolve()

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    old_sys_path = sys.path[:]

    stdout = io.StringIO()

    try:
        os.chdir(START_CWD)

        # Make imports beside the workflow script work, e.g. calcfunctions/.
        prepend_sys_path(START_CWD)
        prepend_sys_path(script_path.parent)

        sys.argv = [str(script_path)] + [str(a) for a in args]

        if capture_stdout:
            with contextlib.redirect_stdout(stdout):
                runpy.run_path(str(script_path), run_name="__main__")
        else:
            runpy.run_path(str(script_path), run_name="__main__")

    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path

    return stdout.getvalue()


def run_script_ok(script_path: Path, args: list[object], *, capture_stdout: bool = False):
    try:
        output = run_script(script_path, args, capture_stdout=capture_stdout)
        return True, output
    except SystemExit as exc:
        if exc.code in (None, 0):
            return True, ""
        print(
            f"FAILED: {script_path.name} {' '.join(map(str, args))} "
            f"exited with {exc.code}"
        )
        return False, ""
    except Exception:
        print(f"FAILED: {script_path.name} {' '.join(map(str, args))}")
        traceback.print_exc()
        return False, ""


def parse_export_lines(text: str) -> None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("export "):
            continue

        # Handles both:
        #   export A=value
        #   export A='value with spaces'
        parts = shlex.split(line)
        for token in parts[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            os.environ[key] = value


def apply_calc_env(workflow: Path) -> None:
    if script_text_has_phase(workflow, "print_calc_env"):
        ok, output = run_script_ok(workflow, ["print_calc_env"], capture_stdout=True)
        if not ok:
            raise RuntimeError("print_calc_env failed")
        print(output, end="")
        parse_export_lines(output)

    # Fallback for older workflow versions without calcfunctions.
    if "MLNEB_CALC_COMMAND" not in os.environ:
        if script_text_has_phase(workflow, "run_singlepoint"):
            os.environ["MLNEB_CALC_COMMAND"] = "mlneb-workflow run_singlepoint"
        else:
            os.environ["MLNEB_CALC_COMMAND"] = "mlneb-workflow run_mace_eval"

    os.environ.setdefault("MLNEB_CALC_RUN_IN_EVAL_DIR", "0")
    os.environ.setdefault("MLNEB_CALC_NAME", "local")


def workflow_phase_from_command(command: str) -> Optional[str]:
    try:
        tokens = shlex.split(command)
    except Exception:
        return None

    if len(tokens) >= 2 and tokens[0] == "mlneb-workflow":
        return tokens[1]

    return None


def run_external_command(command: str, evaldir: Path) -> bool:
    run_in_eval = os.environ.get("MLNEB_CALC_RUN_IN_EVAL_DIR", "0") == "1"
    cwd = evaldir if run_in_eval else START_CWD

    result = subprocess.run(command, shell=True, cwd=str(cwd))
    if result.returncode != 0:
        print(f"ERROR: calculator command failed with rc={result.returncode}")
        return False

    return True


def generic_check_evaluated_traj(evaldir: Path) -> bool:
    """Fallback result check for old workflow versions without check_eval."""
    try:
        import numpy as np
        from ase.io import read

        atoms = read(evaldir / "evaluated.traj")
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)

        if not np.isfinite(energy):
            raise RuntimeError(f"non-finite energy: {energy}")
        if not np.all(np.isfinite(forces)):
            raise RuntimeError("non-finite force component")

        max_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
        print(
            "generic evaluated.traj check OK: "
            f"energy={energy:.12g}, max_force={max_force:.12g}"
        )
        return True

    except Exception:
        print("FAILED: generic evaluated.traj check")
        traceback.print_exc()
        return False


def run_calculation_checked(workflow: Path, evaldir: Path) -> bool:
    command = os.environ.get("MLNEB_CALC_COMMAND", "").strip()
    if not command:
        raise RuntimeError("MLNEB_CALC_COMMAND is empty")

    phase = workflow_phase_from_command(command)

    if phase is not None:
        ok, _ = run_script_ok(workflow, [phase])
        if not ok:
            return False
    else:
        if not run_external_command(command, evaldir):
            return False

    if script_text_has_phase(workflow, "check_eval"):
        ok, _ = run_script_ok(workflow, ["check_eval"])
        return ok

    return generic_check_evaluated_traj(evaldir)


def count_candidates(candidates_pkl: Path) -> int:
    import dill as pickle

    with open(candidates_pkl, "rb") as f:
        return len(pickle.load(f))


def read_target_success(meta_pkl: Path) -> int:
    import dill as pickle

    try:
        with open(meta_pkl, "rb") as f:
            meta = pickle.load(f)
        value = int(meta.get("target_success", 1))
    except Exception:
        value = 1

    return max(1, value)


def read_current_eval_dir(d0: Path) -> Path:
    return Path((d0 / "current_eval_dir.txt").read_text().strip()).resolve()


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    setup_environment(argv)

    d0 = Path(os.environ["D0"]).resolve()
    reset_d0 = os.environ.get("RESET_D0", "0") == "1"

    if reset_d0 and d0.exists():
        shutil.rmtree(d0)

    d0.mkdir(parents=True, exist_ok=True)

    workflow = find_script(
        "MLNEB_WORKFLOW_SCRIPT",
        "catlearn.scripts.mlneb_workflow_unified",
        "mlneb_workflow_unified.py",
    )
    worker = find_script(
        "MLNEB_EXTRA_WORKER_SCRIPT",
        "catlearn.scripts.extra_worker_unified",
        "extra_worker_unified.py",
    )

    print(f"D0={d0}")
    print(f"INPUT={os.environ.get('INPUT', '')}")
    print(f"CATLEARN_USER_MODULE={os.environ.get('CATLEARN_USER_MODULE', '')}")
    print(f"workflow={workflow}")
    print(f"worker={worker}")

    state0 = d0 / "catlearn_state.pkl"
    state_after = d0 / "catlearn_state.pkl"
    pending = d0 / "pending_eval.traj"
    candidates = d0 / "candidates.pkl"
    meta = d0 / "candidate_meta.pkl"

    print("===== prepare_state =====")
    run_script(workflow, ["prepare_state"])

    print("===== calculator env =====")
    apply_calc_env(workflow)
    print(f"MLNEB_CALC_NAME={os.environ.get('MLNEB_CALC_NAME', '')}")
    print(f"MLNEB_CALC_COMMAND={os.environ.get('MLNEB_CALC_COMMAND', '')}")
    print(f"MLNEB_CALC_RUN_IN_EVAL_DIR={os.environ.get('MLNEB_CALC_RUN_IN_EVAL_DIR', '')}")
    print(f"VASP_NELM={os.environ.get('VASP_NELM', '')}")

    restart = os.environ.get("RESTART", "0") == "1"

    if not restart:
        print("===== initial worker =====")
        run_script(worker, ["initial", state0, pending, candidates, meta])

        print("===== initial eval input =====")
        os.environ.pop("CANDIDATE_INDEX", None)
        run_script(workflow, ["write_eval_input"])
        evaldir = read_current_eval_dir(d0)

        print("===== initial single point/check =====")
        if not run_calculation_checked(workflow, evaldir):
            raise RuntimeError(
                "Initial single point failed or did not pass checks; "
                "no fallback candidate exists."
            )

        print("===== load initial eval =====")
        run_script(workflow, ["load_eval"])
    else:
        print("Skipping initial MLNEB evaluation because RESTART=1 is set.")

    for al_step in range(1, int(os.environ["AL_STEPS"]) + 1):
        if (d0 / "MLNEB_DONE").exists():
            print("MLNEB_DONE found; stopping.")
            break

        print(f"===== AL_STEP {al_step}: next worker =====")
        os.environ["AL_STEP"] = str(al_step)
        run_script(worker, ["next", state_after, pending, candidates, meta])

        n_cand = count_candidates(candidates)
        target_success = read_target_success(meta)

        print(f"AL_STEP={al_step} NCAND={n_cand}")
        print(f"TARGET_SUCCESS={target_success}")

        if n_cand == 0:
            print("No candidates returned; stopping.")
            break

        success_count = 0

        for candidate_index in range(n_cand):
            os.environ["CANDIDATE_INDEX"] = str(candidate_index)

            print(f"===== candidate {candidate_index} of {n_cand - 1} =====")
            run_script(workflow, ["write_eval_input"])
            evaldir = read_current_eval_dir(d0)

            if run_calculation_checked(workflow, evaldir):
                print(f"Candidate {candidate_index} passed checks; loading evaluation into MLNEB.")
                run_script(workflow, ["load_eval"])
                success_count += 1

                if success_count >= target_success:
                    print(f"Reached TARGET_SUCCESS={target_success}; continuing to next AL step.")
                    break
            else:
                print(
                    f"WARNING: candidate {candidate_index} failed single point/check; "
                    "trying next candidate."
                )

        os.environ.pop("CANDIDATE_INDEX", None)

        if success_count == 0:
            raise RuntimeError(
                f"All {n_cand} candidates failed single point/check in AL_STEP={al_step}."
            )

        if success_count < target_success:
            print(
                f"WARNING: only {success_count} of TARGET_SUCCESS={target_success} "
                "candidates passed; continuing with successful points only."
            )

        ok, _ = run_script_ok(workflow, ["check_convergence"])
        if not ok:
            print("WARNING: check_convergence failed; continuing like run_mlneb_core.sh does.")

    print("===== LOCAL DEBUG RUN FINISHED =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
