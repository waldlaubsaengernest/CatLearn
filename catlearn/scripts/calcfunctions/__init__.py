import os

from .base import BaseDFTcalc
from .vaspcalc import VASPcalc
from .macecalc import MACEcalc


def get_calcfunction(calc):
    """Return the small helper object belonging to an ASE calculator.

    This intentionally does not replace the ASE calculator. It only knows how
    to write inputs, read results, and check a finished single point.
    """

    requested = os.environ.get("MLNEB_CALCFUNCTION", "").strip().lower()

    if requested:
        mapping = {
            "base": BaseDFTcalc,
            "dft": BaseDFTcalc,
            "vasp": VASPcalc,
            "vaspcalc": VASPcalc,
            "mace": MACEcalc,
            "macecalc": MACEcalc,
        }
        try:
            return mapping[requested](calc)
        except KeyError as exc:
            raise RuntimeError(
                f"Unknown MLNEB_CALCFUNCTION={requested!r}. "
                f"Known: {sorted(mapping)}"
            ) from exc

    for cls in (VASPcalc, MACEcalc):
        if cls.matches(calc):
            return cls(calc)

    return BaseDFTcalc(calc)
