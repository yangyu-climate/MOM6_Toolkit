#!/usr/bin/env python3
# Copyright © 2026 Yang Yu. All rights reserved.
# Author: Yang Yu <yang.yu@whoi.edu>

"""Convert monthly ERA5 single-level files to MOM6/CESM-DATM forcing files."""

from pathlib import Path
import shutil
import subprocess
import argparse

import cftime
import numpy as np
import xarray as xr
from netCDF4 import Dataset


DEFAULT_YEAR = 2022


def saturation_vapor_pressure(td):
    """Saturation vapor pressure from dewpoint temperature in K; returns Pa."""
    tc = td - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert monthly ERA5 files to MOM6/CESM-DATM forcing files."
    )
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mesh-exe", default="ESMF_Scrip2Unstruct")
    return parser.parse_args()


def main():
    args = parse_args()
    year = args.year
    input_dir = args.input_dir or Path(f"ERA5_{year}")
    output_dir = args.output_dir or Path(f"ERA5_MOM6_{year}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(f"ERA5.*.{year}.*.nc"))
    if len(files) != 12:
        raise FileNotFoundError(
            f"Expected 12 monthly files in {input_dir}, found {len(files)}"
        )

    ds = xr.open_mfdataset(files, combine="by_coords")

    required = {"u10", "v10", "t2m", "d2m", "msl", "sp", "tp", "ssrd", "strd"}
    missing = sorted(required.difference(ds.data_vars))
    if missing:
        raise KeyError(f"Missing ERA5 variables: {', '.join(missing)}")

    # CDS may call the valid time coordinate "valid_time" instead of "time".
    # Normalize it to "time" for the JRA/DATM-compatible output.
    if "time" not in ds.coords:
        for candidate in ("valid_time", "datetime", "date"):
            if candidate in ds.coords or candidate in ds.variables:
                ds = ds.rename({candidate: "time"})
                break
    if "time" not in ds.coords:
        raise KeyError(
            "No time coordinate found. Expected 'time' or 'valid_time'; "
            f"available coordinates are: {list(ds.coords)}"
        )

    # Make coordinate names match the original JRA files.
    rename_coords = {}
    if "latitude" not in ds.coords and "lat" in ds.coords:
        rename_coords["lat"] = "latitude"
    if "longitude" not in ds.coords and "lon" in ds.coords:
        rename_coords["lon"] = "longitude"
    if rename_coords:
        ds = ds.rename(rename_coords)
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise KeyError("Input files must contain latitude/longitude coordinates")

    # Ensure latitude is monotonic south-to-north.
    if ds.latitude.size > 1 and ds.latitude[1] < ds.latitude[0]:
        ds = ds.sortby("latitude")

    # Match the JRA noleap calendar by removing February 29.
    leap_day = (ds.time.dt.month == 2) & (ds.time.dt.day == 29)
    ds = ds.sel(time=~leap_day)

    # Use an explicit noleap time axis, as in the JRA files.
    noleap_time = []
    for value in ds.time.values:
        if not hasattr(value, "year"):
            value = np.datetime64(value, "s").astype(object)
        noleap_time.append(
            cftime.DatetimeNoLeap(
                value.year,
                value.month,
                value.day,
                value.hour,
                value.minute,
                value.second,
            )
        )
    ds = ds.assign_coords(time=("time", noleap_time))

    # ERA5 raw variables -> JRA-compatible MOM6/DATM forcing variables.
    # tp, ssrd and strd are hourly accumulations from ERA5.
    rho_water = 1000.0
    seconds_per_hour = 3600.0

    td = ds["d2m"]
    ps = ds["sp"]
    vapor_pressure = saturation_vapor_pressure(td)
    shum = 0.622 * vapor_pressure / (ps - 0.378 * vapor_pressure)

    fields = {
        "prec": ds["tp"] * rho_water / seconds_per_hour,
        "lwdn": ds["strd"] / seconds_per_hour,
        "swdn": ds["ssrd"] / seconds_per_hour,
        "q_10": shum,
        "slp": ds["msl"],
        # ERA5 provides 2-m air temperature; it is used as the atmospheric
        # bottom temperature required by the ocean forcing interface.
        "t_10": ds["t2m"],
        "u_10": ds["u10"],
        "v_10": ds["v10"],
    }

    units = {
        "prec": "kg m-2 s-1",
        "lwdn": "W m-2",
        "swdn": "W m-2",
        "q_10": "kg kg-1",
        "slp": "Pa",
        "t_10": "K",
        "u_10": "m s-1",
        "v_10": "m s-1",
    }

    for name, data in fields.items():
        data = data.rename(name).transpose("time", "latitude", "longitude")
        data.attrs = {
            "standard_name": {
                "prec": "precipitation_flux",
                "lwdn": "surface_downwelling_longwave_flux_in_air",
                "swdn": "surface_downwelling_shortwave_flux_in_air",
                "q_10": "specific_humidity",
                "slp": "air_pressure_at_mean_sea_level",
                "t_10": "air_temperature",
                "u_10": "eastward_wind",
                "v_10": "northward_wind",
            }[name],
            "long_name": f"{name} -- ERA5 atmosphere data for MOM6",
            "units": units[name],
            "cell_methods": "area: mean time: point",
            "missing_value": 1.0e20,
        }
        output = data.to_dataset()
        output.attrs = {
            "title": f"{name} -- ERA5 atmosphere data on "
                      f"{grid_spacing(ds.latitude.values):g} x "
                      f"{grid_spacing(ds.longitude.values):g} degree grid",
            "source": "ERA5, converted to JRA-compatible MOM6/CESM-DATM format",
            "Conventions": "CF-1.0",
        }
        output.to_netcdf(
            output_dir / f"ERA5.{year}.{name}.nc",
            encoding={
                "time": {
                    "units": f"hours since {year}-01-01 00:00:00",
                    "calendar": "noleap",
                },
                name: {
                    "dtype": "f4",
                    "_FillValue": np.float32(1.0e20),
                },
            },
            unlimited_dims=["time"],
        )
        print(f"Wrote {output_dir / f'ERA5.{year}.{name}.nc'}")

    # Create a SCRIP grid matching the processed ERA5 forcing grid.
    # This replaces the original JRA TL319 mesh.
    scrip_file = output_dir / "ERA5_SCRIP.nc"
    mesh_file = output_dir / "ERA5_ESMFmesh.nc"
    create_scrip_grid(ds.latitude.values, ds.longitude.values, scrip_file)

    esmf_exe = shutil.which(args.mesh_exe)
    if esmf_exe is None:
        print(f"{args.mesh_exe} was not found in PATH.")
        print("Load ESMF, then run:")
        print(f"{args.mesh_exe} {scrip_file} {mesh_file} 0 ESMF")
    else:
        subprocess.run(
            [esmf_exe, str(scrip_file), str(mesh_file), "0", "ESMF"],
            check=True,
        )
        print(f"Wrote {mesh_file}")

    ds.close()


def cell_edges(values, lower=None, upper=None):
    values = np.asarray(values, dtype=float)
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    if lower is not None:
        edges[0] = lower
    if upper is not None:
        edges[-1] = upper
    return edges


def grid_spacing(values):
    """Return the regular spacing of a coordinate, in degrees."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError("A grid coordinate must contain at least two points")
    spacing = np.diff(values)
    if not np.allclose(spacing, spacing[0], rtol=0.0, atol=1.0e-8):
        raise ValueError("ERA5 coordinates must form a regular grid")
    return abs(float(spacing[0]))


def create_scrip_grid(lat, lon, filename):
    """Write a SCRIP grid using the same lat/lon as the processed forcing."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    if lat[1] < lat[0]:
        lat = lat[::-1]

    lat_edges = cell_edges(lat, lower=-90.0, upper=90.0)
    lon_edges = cell_edges(lon)
    nlat = lat.size
    nlon = lon.size
    ngrid = nlat * nlon

    center_lat = np.repeat(lat, nlon)
    center_lon = np.tile(lon, nlat)
    corner_lat = np.empty((ngrid, 4), dtype="f8")
    corner_lon = np.empty((ngrid, 4), dtype="f8")

    k = 0
    for j in range(nlat):
        for i in range(nlon):
            corner_lat[k, :] = [
                lat_edges[j], lat_edges[j], lat_edges[j + 1], lat_edges[j + 1]
            ]
            corner_lon[k, :] = [
                lon_edges[i], lon_edges[i + 1], lon_edges[i + 1], lon_edges[i]
            ]
            k += 1

    with Dataset(filename, "w", format="NETCDF4_CLASSIC") as dst:
        dst.createDimension("grid_size", ngrid)
        dst.createDimension("grid_corners", 4)
        dst.createDimension("grid_rank", 1)

        dst.createVariable("grid_dims", "i4", ("grid_rank",))[:] = [ngrid]

        v = dst.createVariable("grid_center_lat", "f8", ("grid_size",))
        v.units = "degrees"
        v[:] = center_lat

        v = dst.createVariable("grid_center_lon", "f8", ("grid_size",))
        v.units = "degrees"
        v[:] = center_lon

        v = dst.createVariable(
            "grid_corner_lat", "f8", ("grid_size", "grid_corners")
        )
        v.units = "degrees"
        v[:] = corner_lat

        v = dst.createVariable(
            "grid_corner_lon", "f8", ("grid_size", "grid_corners")
        )
        v.units = "degrees"
        v[:] = corner_lon

        dst.createVariable("grid_imask", "i4", ("grid_size",))[:] = 1
        dst.title = "ERA5 0.5 degree SCRIP grid"

    print(f"Wrote {filename}")


if __name__ == "__main__":
    main()
