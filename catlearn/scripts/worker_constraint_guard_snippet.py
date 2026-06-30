# Add this in mlneb-extra-worker, after loading mlneb and before mlneb.find_next_candidates():
try:
    from catlearn.scripts.mlneb_workflow_unified import (
        repair_mlneb_internal_constraints,
        install_mlneb_constraint_guard,
    )
    repair_mlneb_internal_constraints(mlneb)
    install_mlneb_constraint_guard(mlneb)
except Exception as exc:
    raise RuntimeError(f"failed to install MLNEB constraint guard in worker: {exc}") from exc
