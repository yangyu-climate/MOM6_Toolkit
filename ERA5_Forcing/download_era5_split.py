#!/usr/bin/env python3
# Copyright © 2026 Yang Yu. All rights reserved.
# Author: Yang Yu <yang.yu@whoi.edu>

"""Download ERA5 data in small requests and merge each month."""

from pathlib import Path
import calendar
import argparse

import cdsapi
import xarray as xr


DEFAULT_YEAR = 2022
DEFAULT_GRID = (0.5, 0.5)

# ERA5 variable names and short names used in the downloaded NetCDF files.
VARIABLES = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "total_precipitation": "tp",
    "surface_solar_radiation_downwards": "ssrd",
    "surface_thermal_radiation_downwards": "strd",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and merge monthly ERA5 single-level data."
    )
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument(
        "--grid", type=float, nargs=2, metavar=("LAT_STEP", "LON_STEP"),
        default=DEFAULT_GRID,
        help="Grid spacing in degrees: latitude longitude (default: 0.5 0.5).",
    )
    parser.add_argument("--part-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    year = args.year
    part_dir = args.part_dir or Path(f"ERA5_{year}_parts")
    out_dir = args.out_dir or Path(f"ERA5_{year}")
    part_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    grid_tag = "x".join(str(value).replace(".", "p") for value in args.grid)

    for month in range(1, 13):
        mm = f"{month:02d}"
        ndays = calendar.monthrange(year, month)[1]
        days = [f"{day:02d}" for day in range(1, ndays + 1)]
        times = [f"{hour:02d}:00" for hour in range(24)]
        part_files = []

        for variable, short_name in VARIABLES.items():
            # Include the requested grid in the file name.  Otherwise a
            # previous run at another resolution would be silently reused.
            part_file = part_dir / f"ERA5.{grid_tag}.{year}.{mm}.{short_name}.nc"
            part_files.append(part_file)

            if part_file.exists() and part_file.stat().st_size > 0:
                print(f"Skip existing: {part_file}")
                continue

            print(f"Downloading {year}-{mm}: {variable}")

            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [variable],
                    "year": str(year),
                    "month": mm,
                    "day": days,
                    "time": times,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                    # The requested regular grid can be changed with --grid.
                    "grid": list(args.grid),
                },
                str(part_file),
            )

        combined_file = out_dir / f"ERA5.{grid_tag}.{year}.{mm}.nc"
        if combined_file.exists() and combined_file.stat().st_size > 0:
            print(f"Skip merged file: {combined_file}")
            continue

        datasets = [xr.open_dataset(path) for path in part_files]
        try:
            merged = xr.merge(datasets, compat="override", join="exact")
            merged.to_netcdf(combined_file)
            print(f"Wrote: {combined_file}")
        finally:
            for dataset in datasets:
                dataset.close()


if __name__ == "__main__":
    main()
