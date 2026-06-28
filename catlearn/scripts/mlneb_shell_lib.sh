#!/bin/bash
# Reusable helper functions for MLNEB-style external evaluation workflows.
#
# The shell script starts the single point calculation explicitly.  Calculator
# details are provided by `mlneb-workflow print_calc_env`, which reads the
# pickled ASE calculator and selects a helper from calcfunctions/.

mlneb_run_calculation_checked () {
    local evaldir="$1"
    local command="${MLNEB_CALC_COMMAND:-}"
    local run_in_eval_dir="${MLNEB_CALC_RUN_IN_EVAL_DIR:-0}"
    local rc

    if [ -z "$command" ]; then
        echo "ERROR: MLNEB_CALC_COMMAND is empty."
        return 96
    fi

    if [ "$run_in_eval_dir" = "1" ]; then
        (
            cd "$evaldir"
            set +e
            eval "$command"
            rc=$?
            set -e
            exit "$rc"
        )
    else
        (
            set +e
            eval "$command"
            rc=$?
            set -e
            exit "$rc"
        )
    fi

    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "ERROR: calculator command failed in $evaldir with rc=$rc"
        return "$rc"
    fi

    mlneb-workflow check_eval
}


mlneb_read_target_success () {
    local meta_pkl="$1"

    python - "$meta_pkl" <<'PY_TARGET_SUCCESS'
import pickle
import sys

try:
    with open(sys.argv[1], "rb") as handle:
        meta = pickle.load(handle)
    value = int(meta.get("target_success", 1))
except Exception:
    value = 1

if value < 1:
    value = 1

print(value)
PY_TARGET_SUCCESS
}


mlneb_require_nonnegative_int () {
    local name="$1"
    local value="$2"

    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $name did not parse as a non-negative integer: '$value'"
        return 1
    fi
}


mlneb_require_positive_int () {
    local name="$1"
    local value="$2"

    if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; then
        echo "ERROR: $name did not parse as a positive integer: '$value'"
        return 1
    fi
}
