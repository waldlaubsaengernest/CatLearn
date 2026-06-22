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

    srun vasp_std
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

mlneb-workflow prepare_state

srun mlneb-extra-worker initial "$STATE0" "$PENDING" "$CANDIDATES" "$META"

unset CANDIDATE_INDEX
mlneb-workflow write_vasp_input
EVALDIR=$(cat "$D0/current_eval_dir.txt")
run_vasp_checked "$EVALDIR"

mlneb-workflow load_vasp_eval
mlneb-workflow check_convergence || true

for AL_STEP in $(seq 1 "$AL_STEPS"); do
    if [ -f "$D0/MLNEB_DONE" ]; then
        echo "MLNEB_DONE found; stopping."
        break
    fi

    export AL_STEP

    srun mlneb-extra-worker next "$STATE_AFTER" "$PENDING" "$CANDIDATES" "$META"

    NCAND=$(mlneb-workflow count_candidates 2>/dev/null | tail -n 1)
    echo "AL_STEP=$AL_STEP NCAND=$NCAND"

    if [ -z "$NCAND" ] || [ "$NCAND" -eq 0 ]; then
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
