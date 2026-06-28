#!/bin/bash

INPUT="${1:-${INPUT:-}}"
D0="${D0:-$PWD}"

export INPUT D0
echo D0

N_IMAGES="${N_IMAGES:-18}"
FMAX="${FMAX:-0.05}"
MAX_UNC="${MAX_UNC:-0.05}"
NEB_INTERPOLATION="${NEB_INTERPOLATION:-idpp}"
ML_STEPS="${ML_STEPS:-500}"
AL_STEPS="${AL_STEPS:-100}"
CLEAN_EVAL_DIR="${CLEAN_EVAL_DIR:-1}"
RESTART="${RESTART:-0}"
FALLBACK_CANDIDATES="${FALLBACK_CANDIDATES:-${VASP_FALLBACK_CANDIDATES:-8}}"
VASP_FALLBACK_CANDIDATES="${VASP_FALLBACK_CANDIDATES:-$FALLBACK_CANDIDATES}"
FALLBACK_ORDER="${FALLBACK_ORDER:-uncertainty}"
VASP_FAIL_ON_NELM="${VASP_FAIL_ON_NELM:-1}"

VASP_COMMAND="${VASP_COMMAND:-srun vasp_std}"

export N_IMAGES FMAX MAX_UNC NEB_INTERPOLATION ML_STEPS AL_STEPS CLEAN_EVAL_DIR RESTART VASP_COMMAND
export FALLBACK_CANDIDATES VASP_FALLBACK_CANDIDATES FALLBACK_ORDER VASP_FAIL_ON_NELM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLNEB_SHELL_LIB="${MLNEB_SHELL_LIB:-$SCRIPT_DIR/mlneb_shell_lib.sh}"

if [ ! -f "$MLNEB_SHELL_LIB" ]; then
    echo "ERROR: MLNEB shell library not found: $MLNEB_SHELL_LIB"
    exit 90
fi

source "$MLNEB_SHELL_LIB"

set -euo pipefail
set -x

mkdir -p "$D0"

STATE0="$D0/catlearn_state.pkl"
STATE_AFTER="$D0/catlearn_state.pkl"
PENDING="$D0/pending_eval.traj"
CANDIDATES="$D0/candidates.pkl"
META="$D0/candidate_meta.pkl"

mlneb-workflow prepare_state

eval "$(mlneb-workflow print_calc_env)"
export MLNEB_CALC_NAME MLNEB_CALC_COMMAND MLNEB_CALC_RUN_IN_EVAL_DIR VASP_NELM
echo "MLNEB_CALC_NAME=${MLNEB_CALC_NAME:-}"
echo "MLNEB_CALC_COMMAND=${MLNEB_CALC_COMMAND:-}"
echo "MLNEB_CALC_RUN_IN_EVAL_DIR=${MLNEB_CALC_RUN_IN_EVAL_DIR:-}"
echo "VASP_NELM=${VASP_NELM:-}"

if [ "${RESTART:-0}" != "1" ]; then
    srun mlneb-extra-worker initial "$STATE0" "$PENDING" "$CANDIDATES" "$META"

    unset CANDIDATE_INDEX
    mlneb-workflow write_vasp_input
    EVALDIR=$(cat "$D0/current_eval_dir.txt")

    if ! mlneb_run_calculation_checked "$EVALDIR"; then
        echo "ERROR: initial single point failed or did not pass checks; no fallback candidate exists."
        exit 94
    fi

    mlneb-workflow load_vasp_eval
else
    echo "Skipping initial MLNEB evaluation because RESTART=1 is set."
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

    TARGET_SUCCESS=$(mlneb_read_target_success "$META")
    echo "TARGET_SUCCESS=$TARGET_SUCCESS"

    mlneb_require_positive_int "TARGET_SUCCESS" "$TARGET_SUCCESS"
    mlneb_require_nonnegative_int "NCAND" "$NCAND"

    if [ "$NCAND" -eq 0 ]; then
        echo "No candidates returned; stopping."
        break
    fi

    SUCCESS_COUNT=0

    for CANDIDATE_INDEX in $(seq 0 $((NCAND - 1))); do
        export CANDIDATE_INDEX

        echo "Trying candidate $CANDIDATE_INDEX of $((NCAND - 1))"
        mlneb-workflow write_vasp_input
        EVALDIR=$(cat "$D0/current_eval_dir.txt")

        if mlneb_run_calculation_checked "$EVALDIR"; then
            echo "Candidate $CANDIDATE_INDEX passed checks; loading evaluation into MLNEB."
            mlneb-workflow load_vasp_eval

            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

            if [ "$SUCCESS_COUNT" -ge "$TARGET_SUCCESS" ]; then
                echo "Reached TARGET_SUCCESS=$TARGET_SUCCESS; continuing to next AL step."
                break
            fi
        else
            rc=$?
            echo "WARNING: Candidate $CANDIDATE_INDEX failed single point/check with rc=$rc; trying next candidate."
        fi
    done

    unset CANDIDATE_INDEX

    if [ "$SUCCESS_COUNT" -eq 0 ]; then
        echo "ERROR: all $NCAND candidates failed single point/check in AL_STEP=$AL_STEP."
        exit 95
    fi

    if [ "$SUCCESS_COUNT" -lt "$TARGET_SUCCESS" ]; then
        echo "WARNING: only $SUCCESS_COUNT of TARGET_SUCCESS=$TARGET_SUCCESS candidates converged; continuing with successful points only."
    fi

    mlneb-workflow check_convergence || true
done

echo "MLNEB workflow finished."
