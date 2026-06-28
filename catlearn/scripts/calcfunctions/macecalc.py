import hashlib
import os
from pathlib import Path

from .base import BaseDFTcalc


class MACEcalc(BaseDFTcalc):
    """Calcfunction helper for MACE and MACE-like ASE calculators.

    MACE has no electronic SCF convergence in the VASP sense.  The normal
    checks are therefore inherited from BaseDFTcalc: energy and forces must
    exist and be finite.

    For debugging the fallback workflow, set MACE_DEBUG_FAIL=1.  The debug
    failure is raised in check_singlepoint_convergence(), i.e. after the local
    single point has run and before the point is loaded into the MLNEB training
    set.  This mimics a VASP calculation that finished but failed the
    convergence/result check.

    Useful examples:

      # Fail candidate 0 in every AL step, so candidate 1 should be tried.
      export MACE_DEBUG_FAIL=1
      export MACE_DEBUG_FAIL_MODE=candidates
      export MACE_DEBUG_FAIL_CANDIDATES=0

      # Deterministic pseudo-random failures with probability 30%.
      export MACE_DEBUG_FAIL=1
      export MACE_DEBUG_FAIL_MODE=random
      export MACE_DEBUG_FAIL_PROB=0.3
      export MACE_DEBUG_FAIL_SEED=17

      # Also allow the initial point to fail.  Default is 0 because there is no
      # fallback for the initial MLNEB evaluation.
      export MACE_DEBUG_FAIL_INITIAL=1
    """

    name = "mace"

    @classmethod
    def matches(cls, calc):
        text = (
            calc.__class__.__name__ + " " + calc.__class__.__module__
        ).lower()
        return "mace" in text

    def check_singlepoint_convergence(self, eval_dir):
        self._debug_maybe_fail(eval_dir)

    def _debug_maybe_fail(self, eval_dir):
        if not _env_true("MACE_DEBUG_FAIL", default=False):
            return

        label = os.environ.get("CANDIDATE_INDEX", "initial")
        is_initial = label == "initial"

        if is_initial and not _env_true("MACE_DEBUG_FAIL_INITIAL", default=False):
            return

        mode = os.environ.get("MACE_DEBUG_FAIL_MODE", "random").strip().lower()

        fail = False
        reason = ""

        if mode in ("always", "all"):
            fail = True
            reason = "mode=always"

        elif mode in ("candidate", "candidates", "indices", "index"):
            items = _parse_list(os.environ.get("MACE_DEBUG_FAIL_CANDIDATES", "0"))
            fail = label in items
            reason = f"candidate_index={label} in {sorted(items)}"

        elif mode in ("step", "steps", "al_step", "al_steps"):
            step = os.environ.get("AL_STEP", "")
            items = _parse_list(os.environ.get("MACE_DEBUG_FAIL_STEPS", "1"))
            fail = step in items
            reason = f"al_step={step} in {sorted(items)}"

        elif mode in ("every", "every_n"):
            n = int(os.environ.get("MACE_DEBUG_FAIL_EVERY", "2"))
            if n < 1:
                raise RuntimeError("MACE_DEBUG_FAIL_EVERY must be >= 1")
            if label != "initial":
                fail = ((int(label) + 1) % n) == 0
                reason = f"(candidate_index + 1) % {n} == 0"

        elif mode in ("random", "prob", "probability"):
            prob = float(os.environ.get("MACE_DEBUG_FAIL_PROB", "0.5"))
            if prob < 0.0 or prob > 1.0:
                raise RuntimeError("MACE_DEBUG_FAIL_PROB must be between 0 and 1")

            seed = os.environ.get("MACE_DEBUG_FAIL_SEED", "0")
            token = _debug_token(eval_dir)
            u = _stable_unit_random(f"{seed}|{token}")

            fail = u < prob
            reason = f"u={u:.6f} < prob={prob:.6f}, token={token}"

        else:
            raise RuntimeError(
                "Unknown MACE_DEBUG_FAIL_MODE="
                f"{mode!r}. Use always, candidates, steps, every_n, or random."
            )

        if fail:
            raise RuntimeError(
                "MACE debug failure triggered. "
                f"mode={mode}, {reason}. "
                "This is intentional and tests the MLNEB fallback path."
            )


def _env_true(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_list(text):
    out = set()
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.add(item)
    return out


def _debug_token(eval_dir):
    path = Path(eval_dir)
    return "|".join(
        [
            f"al={os.environ.get('AL_STEP', 'initial')}",
            f"cand={os.environ.get('CANDIDATE_INDEX', 'initial')}",
            f"evaldir={path.name}",
        ]
    )


def _stable_unit_random(text):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    return integer / float(2**64)
