#!/usr/bin/env python3
"""
Unified extra worker for both local serial debugging and MPI cluster execution.

Usage:
  python extra_worker_unified.py initial STATE_IN PENDING_TRAJ CANDIDATES_PKL META_PKL
  srun python extra_worker_unified.py next STATE_IN PENDING_TRAJ CANDIDATES_PKL META_PKL

The worker writes candidates.pkl as a list of payload dicts:
  {
    "atoms": Atoms,
    "energy_pred": float,
    "unc": float,
    "pred_energies": ...,
    "uncertainties": ...
  }

This preserves the prediction values that the original CatLearn
evaluate_candidates()/broadcast_predictions() path would otherwise associate
with each candidate.
"""

import os
import sys

import dill as pickle
import numpy as np


class SerialComm:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, obj, root=0):
        return obj

    def Barrier(self):
        return None


def should_use_mpi():
    if os.environ.get("CATLEARN_USE_MPI4PY", "0") == "1":
        return True
    try:
        return int(os.environ.get("SLURM_NTASKS", "1")) > 1
    except ValueError:
        return False


def get_comm():
    if should_use_mpi():
        os.environ["CATLEARN_USE_MPI4PY"] = "1"
        from mpi4py import MPI
        return MPI.COMM_WORLD
    os.environ.pop("CATLEARN_USE_MPI4PY", None)
    return SerialComm()


comm = get_comm()
rank = comm.Get_rank()
size = comm.Get_size()


def dump_atomic_rank0(obj, path):
    if rank == 0:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    comm.Barrier()


def load_state_bcast(path):
    if rank == 0:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    else:
        obj = None
    return comm.bcast(obj, root=0)


def finite_or_nan(value):
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value


def make_candidate_payloads(mlneb, candidates):
    """Capture candidate Atoms plus associated prediction values.

    This must run directly after find_next_candidates(), while the MLNEB object
    still has the prediction queues/lists in the same state as the original
    evaluate_candidates() path expects.
    """
    payloads = []

    if candidates is None:
        return payloads

    for candidate in candidates:
        # This mirrors what original evaluate_candidates() would do before
        # evaluate(candidate, is_predicted=True).
        if hasattr(mlneb,"use_database_check") and mlneb.use_database_check:
            candidate = mlneb.ensure_candidate_not_in_database(
                candidate,
                show_message=True,
            )
        try:
            mlneb.broadcast_predictions()
        except Exception as exc:
            if rank == 0:
                print(f"WARNING: broadcast_predictions failed: {exc}", flush=True)

        payloads.append(
            {
                "atoms": candidate,
                "energy_pred": finite_or_nan(getattr(mlneb, "energy_pred", np.nan)),
                "unc": finite_or_nan(getattr(mlneb, "unc", np.nan)),
                "pred_energies": getattr(mlneb, "pred_energies", None),
                "uncertainties": getattr(mlneb, "uncertainties", None),
            }
        )

    return payloads


def main():
    if len(sys.argv) < 5:
        if rank == 0:
            print(
                "Usage: extra_worker_unified.py "
                "{initial|next} STATE_IN PENDING_TRAJ CANDIDATES_PKL [CANDIDATE_META_PKL]",
                flush=True,
            )
        raise SystemExit(1)

    mode = sys.argv[1]
    state_in = sys.argv[2]
    pending_traj = sys.argv[3]
    candidates_pkl = sys.argv[4]
    meta_pkl = sys.argv[5] if len(sys.argv) > 5 else os.path.join(
        os.path.dirname(candidates_pkl), "candidate_meta.pkl"
    )

    if rank == 0:
        print(f"extra_worker mode={mode} rank/size={rank}/{size}", flush=True)

    mlneb = load_state_bcast(state_in)

    if mode == "initial":
        os.environ["CATLEARN_WRITE_EVAL_ONLY"] = "1"
        os.environ["CATLEARN_STATE_PKL"] = state_in
        os.environ["CATLEARN_PENDING_TRAJ"] = pending_traj

        try:
            mlneb.extra_initial_data()
        except SystemExit:
            pass

        comm.Barrier()
        if rank == 0:
            print("Initial pending evaluation written.", flush=True)
        return

    if mode == "next":
        fmax = float(os.environ.get("FMAX", "0.05"))
        max_unc = float(os.environ.get("MAX_UNC", "0.05"))
        ml_steps = int(os.environ.get("ML_STEPS", "500"))
        al_step = int(os.environ.get("AL_STEP", "1"))

        mlneb.train_mlmodel()
        candidates, method_converged = mlneb.find_next_candidates(
            fmax=mlneb.scale_fmax * fmax,
            step=al_step,
            ml_steps=ml_steps,
            max_unc=max_unc,
            dtrust=None,
        )

        payloads = make_candidate_payloads(mlneb, candidates)

        if rank == 0:
            dump_data = {
                "method_converged": method_converged,
                "al_step": al_step,
                "n_candidates": len(payloads),
            }
            with open(state_in, "wb") as f:
                pickle.dump(mlneb, f)
            with open(candidates_pkl, "wb") as f:
                pickle.dump(payloads, f)
            with open(meta_pkl, "wb") as f:
                pickle.dump(dump_data, f)

        comm.Barrier()
        return

    if rank == 0:
        print(f"Unknown mode: {mode}", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
