#!/bin/bash
# Reusable helper functions for MLNEB/MLGO-style external evaluation workflows.
#
# Source this file from a run script:
#   source "$SCRIPT_DIR/mlneb_shell_lib.sh"
#
# The functions intentionally return non-zero on recoverable evaluation
# failures instead of exiting the whole job. The caller decides whether to
# retry another candidate or abort.

mlneb_vasp_converged_checked () {
    local evaldir="$1"

    python - "$evaldir" <<'PY_VASP_CHECK'
import os
import re
import sys

evaldir = sys.argv[1]

outcar = os.path.join(evaldir, "OUTCAR")
oszicar = os.path.join(evaldir, "OSZICAR")
incar = os.path.join(evaldir, "INCAR")

def fail(code, message):
    print(f"VASP_CHECK_FAIL: {message}", flush=True)
    sys.exit(code)

def warn(message):
    print(f"VASP_CHECK_WARN: {message}", flush=True)

if not os.path.isfile(outcar) or os.path.getsize(outcar) == 0:
    fail(2, f"OUTCAR missing/empty in {evaldir}")

if not os.path.isfile(oszicar) or os.path.getsize(oszicar) == 0:
    fail(3, f"OSZICAR missing/empty in {evaldir}")

severe_patterns = [
    "ZBRENT: fatal error",
    "VERY BAD NEWS",
    "BRMIX: very serious problems",
    "EDDDAV: Call to ZHEGV failed",
    "ZHEGV failed",
    "LAPACK: Routine ZPOTRF failed",
    "internal error in subroutine",
]

for filename in (outcar, oszicar):
    try:
        text = open(filename, errors="ignore").read()
    except Exception:
        text = ""

    upper_text = text.upper()

    for pattern in severe_patterns:
        if pattern.upper() in upper_text:
            fail(4, f"found severe VASP message {pattern!r} in {os.path.basename(filename)}")

nelm_env = os.environ.get("VASP_NELM", "").strip()

if nelm_env:
    nelm = int(nelm_env)
else:
    nelm = 60

if os.path.isfile(incar):
if os.path.isfile(incar):
    for raw_line in open(incar, errors="ignore"):
        line = raw_line.split("#", 1)[0].split("!", 1)[0]
        match = re.search(r"\bNELM\s*=\s*([0-9]+)", line, flags=re.I)
        if match:
            nelm = int(match.group(1))

fail_on_nelm = os.environ.get("VASP_FAIL_ON_NELM", "1").strip().lower()
fail_on_nelm = fail_on_nelm not in ("0", "false", "no", "off")

summary_seen = False
last_finished_e_step = None
current_e_step = None

electronic_re = re.compile(
    r"^\s*(?:DAV|RMM|CG|N|SDA|DMP|MIX)[: ]+\s*([0-9]+)\b",
    flags=re.I,
)

with open(oszicar, errors="ignore") as handle:
    for line in handle:
        match = electronic_re.match(line)
        if match:
            current_e_step = int(match.group(1))
            continue

        if " F=" in line and " E0=" in line:
            summary_seen = True
            last_finished_e_step = current_e_step
            current_e_step = None

if not summary_seen:
    fail(5, "OSZICAR has no completed F=/E0= summary line")

if last_finished_e_step is None:
    warn("could not parse last electronic iteration count; accepting completed OSZICAR summary")
    sys.exit(0)

if fail_on_nelm and last_finished_e_step >= nelm:
    fail(
        6,
        f"last electronic cycle used {last_finished_e_step} steps, "
        f"NELM={nelm}; treating as unconverged"
    )

print(
    f"VASP_CHECK_OK: last electronic cycle used {last_finished_e_step} steps, "
    f"NELM={nelm}",
    flush=True,
)
sys.exit(0)
PY_VASP_CHECK
}


mlneb_run_vasp_checked () {
    local evaldir="$1"
    local vasp_command="${2:-${VASP_COMMAND:-srun vasp_std}}"
    local rc

    (
        cd "$evaldir"
        set +e
        $vasp_command
        rc=$?
        set -e
        exit "$rc"
    )

    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "ERROR: VASP command failed in $evaldir with rc=$rc"
        return "$rc"
    fi

    if [ ! -s "$evaldir/OUTCAR" ]; then
        echo "ERROR: OUTCAR missing/empty in $evaldir"
        return 91
    fi

    mlneb_vasp_converged_checked "$evaldir"
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
