#!/bin/bash

INPUT="${1:-${INPUT:-}}"
D0="${D0:-$PWD}"

export INPUT D0
echo D0

EVAL_BACKEND="${EVAL_BACKEND:-vasp}"
N_IMAGES="${N_IMAGES:-18}"
FMAX="${FMAX:-0.05}"
MAX_UNC="${MAX_UNC:-0.05}"
NEB_INTERPOLATION="${NEB_INTERPOLATION:-idpp}"
ML_STEPS="${ML_STEPS:-500}"
AL_STEPS="${AL_STEPS:-100}"
CLEAN_EVAL_DIR="${CLEAN_EVAL_DIR:-0}"

VASP_COMMAND="${VASP_COMMAND:-srun vasp_std}"

export EVAL_BACKEND N_IMAGES FMAX MAX_UNC NEB_INTERPOLATION ML_STEPS AL_STEPS CLEAN_EVAL_DIR VASP_COMMAND

set -euo pipefail
set -x

mkdir -p "$D0"

STATE0="$D0/catlearn_state.pkl"
STATE_AFTER="$D0/catlearn_state.pkl"
PENDING="$D0/pending_eval.traj"
CANDIDATES="$D0/candidates.pkl"
META="$D0/candidate_meta.pkl"

run_vasp_checked () {
    local evaldir="$1"
    cd "$evaldir"

    $VASP_COMMAND
    local rc=$?

    if [ "$rc" -ne 0 ]; then
        echo "ERROR: VASP failed in $evaldir with rc=$rc"
        exit "$rc"
    fi

    if [ ! -s OUTCAR ]; then
        echo "ERROR: OUTCAR missing/empty in $evaldir"
        exit 91
    fi

    cd -
}

PREPARE_OUTPUT="$(mlneb-workflow prepare_state)"
echo "$PREPARE_OUTPUT"
eval "$(printf '%s\n' "$PREPARE_OUTPUT" | tail -n 1)"

if [ "${RESTART:-0}" != "1" ]; then
    srun mlneb-extra-worker initial "$STATE0" "$PENDING" "$CANDIDATES" "$META"

    unset CANDIDATE_INDEX
    mlneb-workflow write_vasp_input
    EVALDIR=$(cat "$D0/current_eval_dir.txt")
    run_vasp_checked "$EVALDIR"

    mlneb-workflow load_vasp_eval
else
    echo "RESTART=1; skipping initial evaluation and entering AL loop."
fi

for AL_STEP in $(seq 1 "$AL_STEPS"); do
    if [ -f "$D0/MLNEB_DONE" ]; then
        echo "MLNEB_DONE found; stopping."
        break
    fi

    export AL_STEP

    srun mlneb-extra-worker next "$STATE_AFTER" "$PENDING" "$CANDIDATES" "$META"

    NCAND=$(mlneb-workflow count_candidates | tail -n 1)
    echo "AL_STEP=$AL_STEP NCAND=$NCAND"

    if ! [[ "$NCAND" =~ ^[0-9]+$ ]]; then
        echo "ERROR: count_candidates did not return an integer: '$NCAND'"
        exit 92
    fi

    if [ "$NCAND" -eq 0 ]; then
        echo "No candidates returned; stopping."
        break
    fi

    for CANDIDATE_INDEX in $(seq 0 $((NCAND - 1))); do
        export CANDIDATE_INDEX

        mlneb-workflow write_vasp_input
        EVALDIR=$(cat "$D0/current_eval_dir.txt")
        run_vasp_checked "$EVALDIR"

        mlneb-workflow load_vasp_eval
    done

    unset CANDIDATE_INDEX
    mlneb-workflow check_convergence || true
done

echo "MLNEB workflow finished."
