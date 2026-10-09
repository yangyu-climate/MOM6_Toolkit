#!/usr/bin/env python3
"""Convert yearly NCEP reanalysis files to MOM6/CESM-DATM forcing files.

The input files are expected to use the NCEP/NCAR naming convention, for
example ``air.2m.gauss.2020.nc`` and ``uwnd.10m.gauss.2020.nc``.  The output
files use the JRA-compatible field names expected by CORE_IAF_JRA DATM:
prec, lwdn, swdn, q_10, slp, t_10, u_10 and v_10.

The NCEP archive has relative humidity at sigma=0.995 rather than a direct
10-m specific-humidity field.  q_10 is therefore an approximation based on
that humidity, 2-m temperature, and 995 hPa pressure.  For production runs,
replace it with a better near-surface humidity product if available.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cftime
import numpy as np
import xarray as xr
from netCDF4 import Dataset


FIELDS = {
    "prec": ("prate.sfc.gauss", "prate"),
    "lwdn": ("dlwrf.sfc.gauss", "dlwrf"),
    "swdn": ("dswrf.sfc.gauss", "dswrf"),
    "slp": ("slp", "slp"),
    "t_10": ("air.2m.gauss", "air"),
    "u_10": ("uwnd.10m.gauss", "uwnd"),
    "v_10": ("vwnd.10m.gauss", "vwnd"),
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("ncep_forcing.json"),
        help="JSON parameter file (default: ncep_forcing.json next to this script)",
    )
    p.add_argument("--input-dir", type=Path, default=Path("/user/yang.yu/Data/NCEP"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--year", type=int, nargs="+")
    p.add_argument("--mesh-exe", default="ESMF_Scrip2Unstruct")
    return p.parse_args()


def load_config(cli: argparse.Namespace) -> dict:
    config = {}
    if not cli.config.exists():
        raise FileNotFoundError(f"Parameter file not found: {cli.config}")
    if cli.config:
        with cli.config.open(encoding="utf-8") as stream:
            config = json.load(stream)
    if cli.input_dir is not None and cli.input_dir != Path("/user/yang.yu/Data/NCEP"):
        config["input_dir"] = str(cli.input_dir)
    if cli.output_dir is not None:
        config["output_dir"] = str(cli.output_dir)
    if cli.year is not None:
        config["years"] = cli.year
    config.setdefault("input_dir", "/user/yang.yu/Data/NCEP")
    config.setdefault("output_dir", "NCEP_MOM6")
    if not config.get("years"):
        start = config.get("year_start")
        end = config.get("year_end")
        if start is None or end is None:
            raise ValueError("Specify years or year_start/year_end in the config")
        if int(end) < int(start):
            raise ValueError("year_end must be greater than or equal to year_start")
        config["years"] = list(range(int(start), int(end) + 1))
    config.setdefault("mesh_exe", cli.mesh_exe)
    config["years"] = [int(year) for year in config["years"]]
    return config


def open_year(directory: Path, stem: str, variable: str, year: int) -> xr.DataArray:
    path = directory / f"{stem}.{year}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path, decode_times=True)
    if variable not in ds:
        raise KeyError(f"{variable!r} not found in {path}")
    data = ds[variable]
    rename = {}
    for old, new in (("lat", "latitude"), ("lon", "longitude")):
        if old in data.dims or old in data.coords:
            rename[old] = new
    if rename:
        data = data.rename(rename)
    if data.latitude.size > 1 and float(data.latitude[1]) < float(data.latitude[0]):
        data = data.sortby("latitude")
    return data


def noleap_time(data: xr.DataArray) -> xr.DataArray:
    time = data["time"]
    keep = ~((time.dt.month == 2) & (time.dt.day == 29))
    data = data.sel(time=keep)
    converted = []
    for value in data.time.values:
        value = np.datetime64(value, "s").astype(object) if not hasattr(value, "year") else value
        converted.append(cftime.DatetimeNoLeap(value.year, value.month, value.day,
                                                value.hour, value.minute, value.second))
    return data.assign_coords(time=("time", converted))


def scrip_grid(lat: np.ndarray, lon: np.ndarray, filename: Path) -> None:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat[1] < lat[0]:
        lat = lat[::-1]

    def edges(values: np.ndarray, lower=None, upper=None) -> np.ndarray:
        result = np.empty(values.size + 1)
        result[1:-1] = 0.5 * (values[:-1] + values[1:])
        result[0] = values[0] - 0.5 * (values[1] - values[0])
        result[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
        if lower is not None:
            result[0] = lower
        if upper is not None:
            result[-1] = upper
        return result

    le, oe = edges(lat, -90, 90), edges(lon)
    ngrid = lat.size * lon.size
    clat, clon = np.repeat(lat, lon.size), np.tile(lon, lat.size)
    corner_lat = np.empty((ngrid, 4))
    corner_lon = np.empty((ngrid, 4))
    k = 0
    for j in range(lat.size):
        for i in range(lon.size):
            corner_lat[k] = [le[j], le[j], le[j + 1], le[j + 1]]
            corner_lon[k] = [oe[i], oe[i + 1], oe[i + 1], oe[i]]
            k += 1

    with Dataset(filename, "w", format="NETCDF4_CLASSIC") as nc:
        nc.createDimension("grid_size", ngrid)
        nc.createDimension("grid_corners", 4)
        nc.createDimension("grid_rank", 1)
        nc.createVariable("grid_dims", "i4", ("grid_rank",))[:] = [ngrid]
        center_lat = nc.createVariable("grid_center_lat", "f8", ("grid_size",))
        center_lat.units = "degrees"
        center_lat[:] = clat
        center_lon = nc.createVariable("grid_center_lon", "f8", ("grid_size",))
        center_lon.units = "degrees"
        center_lon[:] = clon
        corner_lat_var = nc.createVariable("grid_corner_lat", "f8", ("grid_size", "grid_corners"))
        corner_lat_var.units = "degrees"
        corner_lat_var[:] = corner_lat
        corner_lon_var = nc.createVariable("grid_corner_lon", "f8", ("grid_size", "grid_corners"))
        corner_lon_var.units = "degrees"
        corner_lon_var[:] = corner_lon
        nc.createVariable("grid_imask", "i4", ("grid_size",))[:] = 1
        nc.title = "NCEP Gaussian 1.875 degree SCRIP grid"


def write_field(data: xr.DataArray, name: str, year: int, output: Path) -> None:
    units = {"prec": "kg m-2 s-1", "lwdn": "W m-2", "swdn": "W m-2",
             "q_10": "kg kg-1", "slp": "Pa", "t_10": "K",
             "u_10": "m s-1", "v_10": "m s-1"}[name]
    data = noleap_time(data).rename(name).transpose("time", "latitude", "longitude")
    data.attrs.update(units=units, long_name=f"{name} -- NCEP forcing for MOM6")
    ds = data.to_dataset()
    ds.attrs.update(Conventions="CF-1.0", source="NCEP/NCAR Reanalysis converted for MOM6 DATM")
    # Do not carry NCEP's source chunking/compression encoding into the new
    # file.  Those settings can be invalid after latitude sorting/renaming.
    for variable in ds.variables.values():
        variable.encoding = {}
        variable.attrs.pop("_FillValue", None)
        variable.attrs.pop("missing_value", None)
    ds.to_netcdf(
        output / f"NCEP.{year}.{name}.nc",
        format="NETCDF4_CLASSIC",
        unlimited_dims=["time"],
        encoding={
            "time": {
                "dtype": "f8",
                "units": f"hours since {year}-01-01 00:00:00",
                "calendar": "noleap",
            },
            name: {"dtype": "f4", "_FillValue": np.float32(1e20)},
        },
    )


def write_streams_file(config: dict, years: list[int], output: Path) -> Path:
    """Fill the NCEP stream template and write one combined stream file."""
    root = output.resolve()
    year_first, year_last = min(years), max(years)
    template = Path(config.get("streams_template", Path(__file__).with_name("user_nl_datm_streams.ncep")))
    text = template.read_text(encoding="utf-8")
    # Replace longer tokens before ROOT; otherwise MESH_ROOT would become
    # MESH_/path and leave an invalid meshfile entry.
    replacements = {
        "YEAR_FIRST": str(year_first),
        "YEAR_LAST": str(year_last),
        "YEAR_ALIGN": str(year_first),
        "MESH_ROOT": str(output.resolve()),
        "ROOT": str(output.resolve()),
    }
    for field in (*FIELDS.keys(), "q_10"):
        replacements[f"{field.upper()}_FILES"] = ",".join(
            str(root / f"NCEP.{year}.{field}.nc") for year in years
        )
    for old, new in replacements.items():
        text = text.replace(old, new)
    destination = output / "user_nl_datm_streams"
    destination.write_text(text, encoding="utf-8")
    return destination


def main() -> None:
    cli = args()
    config = load_config(cli)
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    years = config["years"]
    for year in years:
        raw = {name: open_year(input_dir, stem, var, year)
               for name, (stem, var) in FIELDS.items()}
        rh = open_year(input_dir, "rhum.sig995", "rhum", year)
        # rhum.sig995 is 2.5-degree (73x144), while the other NCEP fields
        # use the 1.875-degree Gaussian grid (94x192).  Put RH on the same
        # grid before calculating q_10; nearest fills the periodic longitude
        # endpoint that lies just outside the linear interpolation range.
        rh_linear = rh.interp_like(raw["t_10"], method="linear")
        rh_nearest = rh.interp(
            latitude=raw["t_10"].latitude,
            longitude=raw["t_10"].longitude,
            method="nearest",
        )
        rh = rh_linear.fillna(rh_nearest)
        # RH is percent; use 995 hPa and 2-m air temperature to estimate q.
        tk = raw["t_10"]
        tc = tk - 273.15
        es = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
        e = (rh / 100.0) * es
        raw["q_10"] = 0.622 * e / (99500.0 - 0.378 * e)
        for name, data in raw.items():
            write_field(data, name, year, output_dir)
    print(f"Wrote {write_streams_file(config, years, output_dir)}")

    first = next(iter(raw.values()))
    scrip = output_dir / "NCEP_SCRIP.nc"
    mesh = output_dir / "NCEP_ESMFmesh.nc"
    scrip_grid(first.latitude.values, first.longitude.values, scrip)
    exe = shutil.which(config["mesh_exe"])
    if exe:
        subprocess.run([exe, str(scrip), str(mesh), "0", "ESMF"], check=True)
    else:
        print(f"ESMF_Scrip2Unstruct not found; run: {config['mesh_exe']} {scrip} {mesh} 0 ESMF")


if __name__ == "__main__":
    main()
