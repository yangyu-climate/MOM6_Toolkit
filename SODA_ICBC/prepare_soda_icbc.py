#!/usr/bin/env python3
"""Stage SODA-derived MOM6 IC/BC files for the bering_ocn.001 test case.

The SODA-to-grid interpolation is deliberately kept separate from staging.
Give this script a directory containing the five MOM6-ready products:

  init_tracers_filled.nc, init_eta_filled.nc, init_vel_filled.nc
  forcing_obc_segment_001.nc, forcing_obc_segment_002.nc,
  forcing_obc_segment_003.nc

It validates the required variables and writes them into the case's ``ocn``
input directory.  This prevents a partially generated case from being used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

import xarray as xr


REQUIRED = {
    "init_tracers_filled.nc": ("temp", "salt"),
    "init_eta_filled.nc": ("eta_t",),
    "init_vel_filled.nc": ("u", "v"),
    "forcing_obc_segment_001.nc": ("u_segment_001", "v_segment_001", "eta_segment_001", "temp_segment_001", "salt_segment_001"),
    "forcing_obc_segment_002.nc": ("u_segment_002", "v_segment_002", "eta_segment_002", "temp_segment_002", "salt_segment_002"),
    "forcing_obc_segment_003.nc": ("u_segment_003", "v_segment_003", "eta_segment_003", "temp_segment_003", "salt_segment_003"),
}


def validate(path: Path, variables: tuple[str, ...]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with xr.open_dataset(path, decode_times=False) as ds:
        missing = [name for name in variables if name not in ds.variables]
        if missing:
            raise ValueError(f"{path}: missing variables {missing}; found {list(ds.variables)}")
        for name in variables:
            if ds[name].size == 0:
                raise ValueError(f"{path}: variable {name} is empty")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--products", type=Path, required=True, help="Directory containing MOM6-ready SODA products")
    p.add_argument("--case", type=Path, required=True, help="Case directory, e.g. .../croc_cases/bering_ocn.001")
    p.add_argument("--input-subdir", default="ocn", help="Case input subdirectory (default: ocn)")
    p.add_argument("--force", action="store_true", help="Replace existing files after validation")
    args = p.parse_args()

    target = args.case / args.input_subdir
    target.mkdir(parents=True, exist_ok=True)
    for name, variables in REQUIRED.items():
        validate(args.products / name, variables)

    with tempfile.TemporaryDirectory(prefix="mom6_soda_stage_", dir=target) as tmp:
        tmp_path = Path(tmp)
        for name in REQUIRED:
            shutil.copy2(args.products / name, tmp_path / name)
        for name in REQUIRED:
            destination = target / name
            if destination.exists() and not args.force:
                raise FileExistsError(f"{destination} exists; use --force to replace it")
        for name in REQUIRED:
            shutil.move(str(tmp_path / name), target / name)

    print(f"Staged {len(REQUIRED)} validated SODA products in {target}")
    print("MOM6 user_nl_mom must reference:")
    print("  TEMP_SALT_Z_INIT_FILE = init_tracers_filled.nc")
    print("  SURFACE_HEIGHT_IC_FILE = init_eta_filled.nc")
    print("  VELOCITY_FILE = init_vel_filled.nc")
    print("  forcing_obc_segment_001.nc ... forcing_obc_segment_003.nc")


if __name__ == "__main__":
    main()
