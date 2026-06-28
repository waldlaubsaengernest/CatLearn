import os
import re
from copy import deepcopy

import numpy as np
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

from .base import BaseDFTcalc


class VASPcalc(BaseDFTcalc):
    name = "vasp"

    @classmethod
    def matches(cls, calc):
        text = (
            calc.__class__.__name__ + " " +
            calc.__class__.__module__
        ).lower()
        return "vasp" in text

    def prepare_mlneb_calc(self, calc, d0):
        calc.directory = os.path.join(d0, "mlneb")
        return calc

    def shell_env(self):
        nelm = self.get_parameter("nelm", "")
        command = (
            os.environ.get("MLNEB_CALC_COMMAND")
            or os.environ.get("VASP_COMMAND")
            or "srun vasp_std"
        )

        return {
            "MLNEB_CALC_NAME": self.name,
            "MLNEB_CALC_COMMAND": command,
            "MLNEB_CALC_RUN_IN_EVAL_DIR": "1",
            "VASP_NELM": "" if nelm in (None, "") else str(int(nelm)),
        }

    def _read_ase_sort_file(self, eval_dir):
        path = os.path.join(eval_dir, "ase-sort.dat")

        if not os.path.exists(path):
            return None, None

        sort = []
        resort = []

        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                sort.append(int(parts[0]))
                resort.append(int(parts[1]))

        return np.asarray(sort, dtype=int), np.asarray(resort, dtype=int)

    def _reorder_result_to_input_order(self, eval_dir, atoms):
        input_atoms_path = os.path.join(eval_dir, "input_atoms.traj")

        if not os.path.exists(input_atoms_path):
            print(
                f"[VASP-REORDER] no input_atoms.traj in {eval_dir}; "
                "leaving result unchanged",
                flush=True,
            )
            return atoms

        ref = read(input_atoms_path)

        if len(ref) != len(atoms):
            raise RuntimeError(
                f"[VASP-REORDER] atom count mismatch in {eval_dir}: "
                f"input_atoms={len(ref)} result={len(atoms)}"
            )

        ref_numbers = np.asarray(ref.get_atomic_numbers())
        result_numbers = np.asarray(atoms.get_atomic_numbers())

        sort, resort = self._read_ase_sort_file(eval_dir)

        if np.array_equal(result_numbers, ref_numbers):
            print("[VASP-REORDER] result already in CatLearn atom order", flush=True)
            return atoms

        if sort is None or resort is None:
            raise RuntimeError(
                f"[VASP-REORDER] atom order changed but ase-sort.dat missing "
                f"in {eval_dir}"
            )

        if len(resort) != len(atoms):
            raise RuntimeError(
                f"[VASP-REORDER] ase-sort.dat length mismatch in {eval_dir}: "
                f"{len(resort)} entries for {len(atoms)} atoms"
            )

        if np.array_equal(result_numbers[resort], ref_numbers):
            index = resort
            map_name = "resort"
        elif np.array_equal(result_numbers[sort], ref_numbers):
            index = sort
            map_name = "sort"
        else:
            bad = np.where(result_numbers != ref_numbers)[0][:20]
            msg = "\n".join(
                f"{i}: ref={ref[i].symbol} result={atoms[i].symbol}"
                for i in bad
            )
            raise RuntimeError(
                "[VASP-REORDER] cannot map VASP result back to CatLearn order.\n"
                f"First direct mismatches:\n{msg}"
            )

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        reordered = ref.copy()
        reordered.set_positions(atoms.get_positions()[index])
        reordered.set_cell(atoms.get_cell())
        reordered.set_pbc(atoms.get_pbc())
        reordered.set_atomic_numbers(ref.get_atomic_numbers())
        reordered.set_constraint(ref.constraints)
        reordered.set_tags(ref.get_tags())

        reordered.calc = SinglePointCalculator(
            reordered,
            energy=float(energy),
            forces=np.asarray(forces, dtype=float)[index],
        )

        print(
            f"[VASP-REORDER] transformed VASP result to CatLearn order using {map_name}",
            flush=True,
        )

        write(os.path.join(eval_dir, "evaluated_reordered_debug.traj"), reordered)

        return reordered

    def read_results(self, eval_dir):
        vasprun = os.path.join(eval_dir, "vasprun.xml")
        outcar = os.path.join(eval_dir, "OUTCAR")

        if os.path.exists(vasprun):
            atoms = read(vasprun, index=-1)
        elif os.path.exists(outcar):
            atoms = read(outcar, index=-1)
        else:
            raise FileNotFoundError(
                f"Neither vasprun.xml nor OUTCAR found in {eval_dir}"
            )

        atoms = self._reorder_result_to_input_order(eval_dir, atoms)
        write(os.path.join(eval_dir, "evaluated.traj"), atoms)
        return atoms

    def _get_nelm(self, eval_dir):
        nelm_env = os.environ.get("VASP_NELM", "").strip()

        if nelm_env:
            nelm = int(nelm_env)
        else:
            value = self.get_parameter("nelm", None)
            nelm = int(value) if value not in (None, "") else 60

        incar = os.path.join(eval_dir, "INCAR")
        if os.path.isfile(incar):
            for raw_line in open(incar, errors="ignore"):
                line = raw_line.split("#", 1)[0].split("!", 1)[0]
                match = re.search(r"\bNELM\s*=\s*([0-9]+)", line, flags=re.I)
                if match:
                    nelm = int(match.group(1))

        return nelm

    def check_singlepoint_convergence(self, eval_dir):
        outcar = os.path.join(eval_dir, "OUTCAR")
        oszicar = os.path.join(eval_dir, "OSZICAR")

        if not os.path.isfile(outcar) or os.path.getsize(outcar) == 0:
            raise RuntimeError(f"[VASP] OUTCAR missing/empty in {eval_dir}")

        if not os.path.isfile(oszicar) or os.path.getsize(oszicar) == 0:
            raise RuntimeError(f"[VASP] OSZICAR missing/empty in {eval_dir}")

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
                    raise RuntimeError(
                        f"[VASP] found severe message {pattern!r} "
                        f"in {os.path.basename(filename)}"
                    )

        nelm = self._get_nelm(eval_dir)

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
            raise RuntimeError("[VASP] OSZICAR has no completed F=/E0= summary line")

        if last_finished_e_step is None:
            print(
                "[VASP] WARNING: could not parse electronic iteration count; "
                "accepting completed OSZICAR summary",
                flush=True,
            )
            return

        if fail_on_nelm and last_finished_e_step >= nelm:
            raise RuntimeError(
                f"[VASP] last electronic cycle used {last_finished_e_step} "
                f"steps, NELM={nelm}; treating as unconverged"
            )

        print(
            f"[VASP] convergence check OK: last electronic cycle used "
            f"{last_finished_e_step} steps, NELM={nelm}",
            flush=True,
        )
