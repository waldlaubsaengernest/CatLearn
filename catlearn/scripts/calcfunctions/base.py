import os
from copy import deepcopy

import numpy as np
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator


class BaseDFTcalc:
    """Small helper around an ASE calculator.

    The helper does NOT decide when a calculation is run.  The shell script
    still performs the run step explicitly.  This class only provides the
    default operations which are valid for many ASE calculators:

      - write input if the ASE calculator supports write_input()
      - provide a default shell command for Python/ASE single points
      - read evaluated.traj
      - optional generic finite energy/force checks

    Calculator-specific subclasses can override only the pieces they need.
    """

    name = "base"

    def __init__(self, calc):
        self.calc = calc

    @classmethod
    def matches(cls, calc):
        return False

    def get_parameter(self, name, default=None):
        keys = (name, name.lower(), name.upper())

        for attr in (
            "parameters",
            "int_params",
            "float_params",
            "string_params",
            "input_params",
        ):
            data = getattr(self.calc, attr, None)
            if not isinstance(data, dict):
                continue

            for key in keys:
                value = data.get(key)
                if value is not None:
                    return value

        return default

    def prepare_mlneb_calc(self, calc, d0):
        return calc

    def shell_env(self):
        # The shell still starts the calculation.  For pure Python ASE
        # calculators the explicit shell command is this workflow phase.
        return {
            "MLNEB_CALC_NAME": self.name,
            "MLNEB_CALC_COMMAND": (
                os.environ.get("MLNEB_CALC_COMMAND")
                or os.environ.get("CALC_COMMAND")
                or "mlneb-workflow run_singlepoint"
            ),
            "MLNEB_CALC_RUN_IN_EVAL_DIR": os.environ.get(
            "MLNEB_CALC_RUN_IN_EVAL_DIR",
            "1",
            ),
        }

    def write_input(self, atoms, eval_dir, calc=None, user_module=None):
        calc = deepcopy(calc if calc is not None else self.calc)

        if hasattr(calc, "directory"):
            try:
                calc.directory = eval_dir
            except Exception:
                pass

        atoms.calc = calc

        if user_module is not None and hasattr(user_module, "update_dipol"):
            atoms.calc = user_module.update_dipol(atoms, atoms.calc)

        if hasattr(atoms.calc, "write_input"):
            try:
                atoms.calc.write_input(atoms)
            except NotImplementedError:
                pass

        write(os.path.join(eval_dir, "input_atoms.traj"), atoms)

    def run_singlepoint(self, eval_dir, calc=None):
        """Run a Python/ASE single point.

        This method is only called by the explicit shell command
        `mlneb-workflow run_singlepoint`.  It is not called implicitly by
        check_eval or by the base class.
        """
        calc = deepcopy(calc if calc is not None else self.calc)

        atoms = read(os.path.join(eval_dir, "input_atoms.traj"))
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        atoms.calc = SinglePointCalculator(
            atoms,
            energy=float(energy),
            forces=np.asarray(forces, dtype=float),
        )

        write(os.path.join(eval_dir, "evaluated.traj"), atoms)

    def read_results(self, eval_dir):
        path = os.path.join(eval_dir, "evaluated.traj")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return read(path)

    def check_singlepoint_convergence(self, eval_dir):
        # Default: no calculator-specific convergence test.
        return None

    def check_energy_forces(self, atoms, eval_dir):
        try:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
        except Exception as exc:
            raise RuntimeError(
                f"[{self.name}] failed to read energy/forces in {eval_dir}: {exc}"
            ) from exc

        if not np.isfinite(energy):
            raise RuntimeError(f"[{self.name}] non-finite energy in {eval_dir}: {energy}")

        if forces.ndim != 2 or forces.shape[1] != 3:
            raise RuntimeError(
                f"[{self.name}] invalid forces shape in {eval_dir}: {forces.shape}"
            )

        if not np.all(np.isfinite(forces)):
            raise RuntimeError(f"[{self.name}] non-finite forces in {eval_dir}")

        max_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0

        max_force_limit = os.environ.get("EVAL_MAX_FORCE")
        if max_force_limit not in (None, ""):
            limit = float(max_force_limit)
            if max_force > limit:
                raise RuntimeError(
                    f"[{self.name}] max force {max_force:.6g} eV/Ang exceeds "
                    f"EVAL_MAX_FORCE={limit:.6g} in {eval_dir}"
                )

        max_abs_energy_limit = os.environ.get("EVAL_MAX_ABS_ENERGY")
        if max_abs_energy_limit not in (None, ""):
            limit = float(max_abs_energy_limit)
            if abs(energy) > limit:
                raise RuntimeError(
                    f"[{self.name}] |energy|={abs(energy):.6g} eV exceeds "
                    f"EVAL_MAX_ABS_ENERGY={limit:.6g} in {eval_dir}"
                )

        print(
            f"[{self.name}] result check OK: "
            f"energy={energy:.12g} eV max_force={max_force:.12g} eV/Ang",
            flush=True,
        )

    def check_eval(self, eval_dir):
        self.check_singlepoint_convergence(eval_dir)
        atoms = self.read_results(eval_dir)
        self.check_energy_forces(atoms, eval_dir)
