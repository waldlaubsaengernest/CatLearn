#!/usr/bin/env python3
"""
Create an exact CatLearn/MLNEB file-restart state for a chosen state N.

Convention:
  N is the CatLearn state/step number from ml_summary.txt / ml_time.txt.

Observed CatLearn MLNEB file counts after state N:
  - evaluated.traj contains initial + final + initial-step structure + the
    state-dependent evaluated images.  Therefore it keeps N + 2 frames.
  - predicted_evaluated.traj contains only predicted/evaluated images,
    not initial and final and not the initial extra step.  Therefore it keeps
    N - 1 frames.
  - predicted.traj contains complete predicted paths.  For state N it keeps
    N - 1 complete paths.
  - ml_summary.txt and ml_time.txt keep header + N data rows.
  - last_path.traj is the last complete predicted path kept in predicted.traj,
    i.e. path N - 1.

State N=1 has no predicted path yet and is therefore not an exact CatLearn
file-restart state for restart=True.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional


REQUIRED_FILES = (
    "predicted.traj",
    "evaluated.traj",
    "ml_summary.txt",
    "ml_time.txt",
)

OPTIONAL_COPY_FILES = (
    "initial_struc.traj",
)


def ase_io():
    from ase.io import read, write
    from ase.io.trajectory import Trajectory
    return read, write, Trajectory


def traj_len(path: Path) -> int:
    _, _, Trajectory = ase_io()
    with Trajectory(str(path), "r") as traj:
        return len(traj)


def read_frames(path: Path, index):
    read, _, _ = ase_io()
    frames = read(path, index)
    if not isinstance(frames, list):
        frames = [frames]
    return frames


def write_frames(path: Path, frames) -> None:
    _, write, _ = ase_io()
    if not isinstance(frames, list):
        frames = [frames]
    write(path, frames)


def count_table_steps(path: Path) -> int:
    lines = path.read_text().splitlines()
    return max(0, len(lines) - 1) if lines else 0


def copy_table_until_state(src: Path, dst: Path, state: int) -> None:
    lines = src.read_text().splitlines()
    need = state + 1  # header + state data lines
    if len(lines) < need:
        raise RuntimeError(
            f"{src} has only {len(lines)} lines; need {need} for state={state}"
        )
    dst.write_text("\n".join(lines[:need]) + "\n")


def infer_n_structures(src: Path, explicit: Optional[int]) -> int:
    if explicit is not None:
        if explicit < 1:
            raise RuntimeError("--n-structures must be >= 1")
        return explicit

    for name in ("initial_struc.traj", "converged.traj", "last_path.traj"):
        path = src / name
        if not path.exists():
            continue
        try:
            n = traj_len(path)
        except Exception:
            continue
        if n > 1:
            return n

    n_images = os.environ.get("N_IMAGES")
    if n_images not in (None, ""):
        try:
            return int(n_images) + 2
        except Exception:
            pass

    raise RuntimeError(
        "Could not infer n_structures per predicted MLNEB path. "
        "Pass --n-structures explicitly, e.g. --n-structures 20."
    )


def check_required_files(src: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (src / name).exists()]
    if missing:
        raise RuntimeError(
            f"Missing required CatLearn restart files in {src}: "
            + ", ".join(missing)
        )


def available_info(src: Path, n_structures: Optional[int]) -> dict:
    check_required_files(src)

    summary_states = count_table_steps(src / "ml_summary.txt")
    time_states = count_table_steps(src / "ml_time.txt")
    evaluated_frames = traj_len(src / "evaluated.traj")
    predicted_frames = traj_len(src / "predicted.traj")

    n = infer_n_structures(src, n_structures)
    predicted_paths = predicted_frames // n
    predicted_remainder = predicted_frames % n

    pred_eval_path = src / "predicted_evaluated.traj"
    if pred_eval_path.exists():
        predicted_evaluated_frames = traj_len(pred_eval_path)
        pred_eval_limit = predicted_evaluated_frames + 1  # state N needs N-1
    else:
        predicted_evaluated_frames = None
        pred_eval_limit = 10**18

    info = {
        "summary_states": summary_states,
        "time_states": time_states,
        "evaluated_frames": evaluated_frames,
        "predicted_frames": predicted_frames,
        "predicted_evaluated_frames": predicted_evaluated_frames,
        "n_structures": n,
        "predicted_paths": predicted_paths,
        "predicted_remainder": predicted_remainder,
        "safe_max_state": min(
            summary_states,
            time_states,
            evaluated_frames - 2,     # state N needs N+2 evaluated frames
            predicted_paths + 1,      # state N needs N-1 predicted paths
            pred_eval_limit,          # state N needs N-1 pred_eval frames
        ),
    }

    return info


def cmd_list(args: argparse.Namespace) -> int:
    src = Path(args.source).resolve()
    info = available_info(src, args.n_structures)

    print(f"Source: {src}")
    print(f"ml_summary states:             {info['summary_states']}")
    print(f"ml_time states:                {info['time_states']}")
    print(f"evaluated.traj frames:         {info['evaluated_frames']}")
    print(f"predicted.traj frames:         {info['predicted_frames']}")
    if info["predicted_evaluated_frames"] is None:
        print("predicted_evaluated.traj:      not present")
    else:
        print(
            "predicted_evaluated.traj frames: "
            f"{info['predicted_evaluated_frames']}"
        )
    print(f"n_structures per path:         {info['n_structures']}")
    print(f"predicted complete paths:      {info['predicted_paths']}")

    if info["predicted_remainder"]:
        print(
            "WARNING: predicted.traj frame count is not divisible by "
            f"n_structures; remainder={info['predicted_remainder']}"
        )

    safe = int(info["safe_max_state"])
    if safe < 2:
        print("Available exact file-restart states: none")
        print("Reason: state 1 has no predicted.traj path.")
        return 0

    print("Available exact file-restart states:")
    for state in range(2, safe + 1):
        print(state)

    return 0


def write_traj_slice(src_path: Path, dst_path: Path, n_keep: int) -> None:
    if n_keep < 1:
        raise RuntimeError(f"Refusing to write empty trajectory for {src_path}")

    n_total = traj_len(src_path)
    if n_total < n_keep:
        raise RuntimeError(
            f"{src_path} has only {n_total} frames; need {n_keep}"
        )

    frames = read_frames(src_path, f":{n_keep}")
    write_frames(dst_path, frames)


def write_last_predicted_path(
    src_path: Path,
    dst_path: Path,
    state: int,
    n_structures: int,
) -> None:
    # State N uses predicted path N-1.
    # zero-based slice for path N-1:
    #   start = (N - 2) * n_structures
    #   stop  = (N - 1) * n_structures
    start = (state - 2) * n_structures
    stop = (state - 1) * n_structures

    n_total = traj_len(src_path)
    if n_total < stop:
        raise RuntimeError(
            f"{src_path} has only {n_total} frames; need {stop} "
            f"for state={state}, n_structures={n_structures}"
        )

    frames = read_frames(src_path, f"{start}:{stop}")
    if len(frames) != n_structures:
        raise RuntimeError(
            f"Internal error while writing last_path.traj: got {len(frames)} "
            f"frames, expected {n_structures}"
        )

    write_frames(dst_path, frames)


def write_predicted_evaluated_if_present(src: Path, dst: Path, state: int) -> None:
    src_path = src / "predicted_evaluated.traj"
    if not src_path.exists():
        return

    keep = state - 1
    if keep < 1:
        return

    write_traj_slice(src_path, dst / "predicted_evaluated.traj", keep)


def cmd_number(args: argparse.Namespace) -> int:
    src = Path(args.source).resolve()
    dst = Path(args.out).resolve()
    state = int(args.state)

    if state < 2:
        raise RuntimeError(
            "Exact CatLearn file restart needs state >= 2. "
            "State 1 has no predicted.traj path."
        )

    check_required_files(src)
    n_structures = infer_n_structures(src, args.n_structures)
    info = available_info(src, n_structures)
    safe = int(info["safe_max_state"])

    if state > safe:
        raise RuntimeError(
            f"Requested state={state}, but only states 2..{safe} are safely "
            "available for exact CatLearn file restart from the current files."
        )

    if dst.exists():
        raise RuntimeError(
            f"Output directory already exists: {dst}. "
            "Refusing to overwrite; remove it or choose --out."
        )

    tmp_dst = dst.with_name(dst.name + ".tmp")
    if tmp_dst.exists():
        raise RuntimeError(
            f"Temporary output directory already exists: {tmp_dst}. "
            "Remove it before retrying."
        )

    tmp_dst.mkdir(parents=True)

    try:
        predicted_keep = (state - 1) * n_structures
        evaluated_keep = state + 2
        predicted_evaluated_keep = state - 1

        write_traj_slice(
            src / "predicted.traj",
            tmp_dst / "predicted.traj",
            predicted_keep,
        )

        write_last_predicted_path(
            src / "predicted.traj",
            tmp_dst / "last_path.traj",
            state,
            n_structures,
        )

        write_traj_slice(
            src / "evaluated.traj",
            tmp_dst / "evaluated.traj",
            evaluated_keep,
        )

        write_predicted_evaluated_if_present(src, tmp_dst, state)

        copy_table_until_state(
            src / "ml_summary.txt",
            tmp_dst / "ml_summary.txt",
            state,
        )
        copy_table_until_state(
            src / "ml_time.txt",
            tmp_dst / "ml_time.txt",
            state,
        )

        for name in OPTIONAL_COPY_FILES:
            src_path = src / name
            if src_path.exists():
                shutil.copy2(src_path, tmp_dst / name)

        (tmp_dst / "README_restart_state.txt").write_text(
            "\n".join(
                [
                    "CatLearn/MLNEB exact file-restart state created by mlneb-state.",
                    f"source = {src}",
                    f"state = {state}",
                    f"n_structures = {n_structures}",
                    "",
                    "Kept frame counts:",
                    f"  predicted.traj = {predicted_keep} = (state - 1) * n_structures",
                    f"  last_path.traj = {n_structures}",
                    f"  evaluated.traj = {evaluated_keep} = state + 2",
                    f"  predicted_evaluated.traj = {predicted_evaluated_keep} = state - 1, if present",
                    f"  ml_summary.txt = header + {state} rows",
                    f"  ml_time.txt = header + {state} rows",
                    "",
                    "Use these files for MLNEB(..., restart=True).",
                    "Do not combine restart=True with neb_interpolation=last_path.traj",
                    "if you want exact CatLearn file-restart behavior.",
                    "",
                    "Files intentionally NOT copied:",
                    "  converged.traj",
                    "  MLNEB_DONE",
                    "",
                ]
            )
            + "\n"
        )

        os.replace(tmp_dst, dst)

    except Exception:
        if tmp_dst.exists():
            shutil.rmtree(tmp_dst)
        raise

    print(f"Wrote restart files to: {dst}")
    print(f"state: {state}")
    print(f"n_structures: {n_structures}")
    print(f"predicted.traj frames: {predicted_keep}")
    print(f"last_path.traj frames: {n_structures}")
    print(f"evaluated.traj frames: {evaluated_keep}")
    print(f"predicted_evaluated.traj frames, if present: {predicted_evaluated_keep}")
    print("Inspect the files before copying/using them.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlneb-state",
        description="List or create exact CatLearn/MLNEB file-restart states.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List available exact restart state numbers.")
    p_list.add_argument("source", nargs="?", default=".")
    p_list.add_argument("--n-structures", type=int, default=None)
    p_list.set_defaults(func=cmd_list)

    p_number = sub.add_parser(
        "number",
        help="Create new_state/ restart files for a given state number.",
    )
    p_number.add_argument("state", type=int)
    p_number.add_argument("source", nargs="?", default=".")
    p_number.add_argument("--n-structures", type=int, default=None)
    p_number.add_argument("--out", default="new_state")
    p_number.set_defaults(func=cmd_number)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0].isdigit():
        argv = ["number"] + argv

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
