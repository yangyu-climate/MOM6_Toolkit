#!/usr/bin/env python3
"""Generate MOM6 IC and OBC files from SODA3.15.2 five-day NetCDF files.

This is tailored to the bering_ocn.001 grid.  It reads SODA's native variables
(``temp``, ``salt``, ``u``, ``v``, ``ssh``), interpolates them to the MOM6
supergrid with scipy, and writes the filenames referenced by ``user_nl_mom``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator


FILL = np.float32(1.0e20)


def setup_logging(log_file: Path) -> logging.Logger:
    """Write progress both to the terminal and to a persistent log file."""
    logger = logging.getLogger("soda_icbc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


class OBCWriter:
    """Append one processed SODA time slice directly to an OBC NetCDF file."""

    def __init__(self, path: Path, segment: int, depth: np.ndarray,
                 dz: np.ndarray, boundary_y: np.ndarray, boundary_x: np.ndarray,
                 start: np.datetime64):
        tag = f"{segment:03d}"
        self.start = start
        self.depth = depth.astype("f4")
        self.dz = dz.astype("f4")
        self.ds = Dataset(path, "w", format="NETCDF4_CLASSIC")
        nz = len(depth)
        ny, nx = boundary_y.shape
        names = {
            "nz_temp": f"nz_segment_{tag}_temp",
            "nz_salt": f"nz_segment_{tag}_salt",
            "nz_u": f"nz_segment_{tag}_u",
            "nz_v": f"nz_segment_{tag}_v",
            "ny": f"ny_segment_{tag}",
            "nx": f"nx_segment_{tag}",
        }
        self.names = names
        self.ds.createDimension("time", None)
        for name in (names["nz_temp"], names["nz_salt"], names["nz_u"], names["nz_v"]):
            self.ds.createDimension(name, nz)
        self.ds.createDimension(names["ny"], ny)
        self.ds.createDimension(names["nx"], nx)

        time = self.ds.createVariable("time", "f8", ("time",))
        time.axis = "T"
        time.units = f"days since {start} 00:00:00"
        time.calendar = "gregorian"
        self.ds.createVariable("depth", "f4", (names["nz_temp"],))[:] = self.depth
        for key in ("nz_temp", "nz_salt", "nz_u", "nz_v"):
            coord = self.ds.createVariable(names[key], "f4", (names[key],))
            coord[:] = self.depth
            coord.axis = "Z"
            coord.units = "m"
        for key, size in (("ny", ny), ("nx", nx)):
            coord = self.ds.createVariable(names[key], "i4", (names[key],))
            coord[:] = np.arange(size)
            coord.axis = "Y" if key == "ny" else "X"
        self.ds.createVariable(f"lon_segment_{tag}", "f4", (names["ny"], names["nx"]))[:] = boundary_x
        self.ds.createVariable(f"lat_segment_{tag}", "f4", (names["ny"], names["nx"]))[:] = boundary_y
        dims4 = ("time", names["nz_temp"], names["ny"], names["nx"])
        dims2 = ("time", names["ny"], names["nx"])
        for var, dims in (
            (f"u_segment_{tag}", ("time", names["nz_u"], names["ny"], names["nx"])),
            (f"v_segment_{tag}", ("time", names["nz_v"], names["ny"], names["nx"])),
            (f"temp_segment_{tag}", dims4),
            (f"salt_segment_{tag}", ("time", names["nz_salt"], names["ny"], names["nx"])),
            (f"eta_segment_{tag}", dims2),
            (f"dz_temp_segment_{tag}", dims4),
            (f"dz_salt_segment_{tag}", ("time", names["nz_salt"], names["ny"], names["nx"])),
            (f"dz_u_segment_{tag}", ("time", names["nz_u"], names["ny"], names["nx"])),
            (f"dz_v_segment_{tag}", ("time", names["nz_v"], names["ny"], names["nx"])),
        ):
            self.ds.createVariable(var, "f4", dims, fill_value=FILL)
        self.tag = tag

    def append(self, date: np.datetime64, u: np.ndarray, v: np.ndarray,
               eta: np.ndarray, temp: np.ndarray, salt: np.ndarray) -> None:
        i = len(self.ds.dimensions["time"])
        self.ds.variables["time"][i] = (date - self.start) / np.timedelta64(1, "D")
        self.ds.variables[f"u_segment_{self.tag}"][i] = u.astype("f4")
        self.ds.variables[f"v_segment_{self.tag}"][i] = v.astype("f4")
        self.ds.variables[f"eta_segment_{self.tag}"][i] = eta.astype("f4")
        self.ds.variables[f"temp_segment_{self.tag}"][i] = temp.astype("f4")
        self.ds.variables[f"salt_segment_{self.tag}"][i] = salt.astype("f4")
        for name, field in (("temp", temp), ("salt", salt), ("u", u), ("v", v)):
            self.ds.variables[f"dz_{name}_segment_{self.tag}"][i] = np.broadcast_to(
                self.dz[:, None, None], field.shape).astype("f4")
        self.ds.sync()

    def close(self) -> None:
        self.ds.close()


def fill_nearest(values: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    """Fill NaNs in each horizontal plane from the nearest valid value."""
    result = np.asarray(values, dtype="f4").copy()
    leading = result.shape[:-2]
    for index in np.ndindex(leading):
        plane = result[index]
        valid = np.isfinite(plane)
        if not valid.any():
            result[index] = fallback
            continue
        if not valid.all():
            nearest = distance_transform_edt(~valid, return_distances=False,
                                             return_indices=True)
            result[index][~valid] = plane[tuple(nearest[:, ~valid])]
    return result


def regular_axis(da: xr.DataArray, dim: str, values: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a strictly ascending coordinate and data reordered to match it."""
    coord = np.asarray(da[dim].values, dtype=float)
    order = np.argsort(coord)
    coord = coord[order]
    values = np.take(values, order, axis=axis)
    keep = np.r_[True, np.diff(coord) > 0.0]
    return coord[keep], np.take(values, np.flatnonzero(keep), axis=axis)


def load_config(path: Path) -> SimpleNamespace:
    with path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    required = ("soda_dir", "hgrid", "output_dir", "start", "end")
    missing = [name for name in required if name not in cfg]
    if missing:
        raise KeyError(f"Missing settings in {path}: {', '.join(missing)}")
    cfg["soda_dir"] = Path(cfg["soda_dir"])
    cfg["hgrid"] = Path(cfg["hgrid"])
    cfg["output_dir"] = Path(cfg["output_dir"])
    cfg.setdefault("pattern", "soda3.15.2_5dy_ocean_or_*.nc")
    cfg.setdefault("log_file", "generate_soda_icbc.log")
    return SimpleNamespace(**cfg)


def lon_for_source(lon: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Put target longitudes in the same convention as SODA."""
    out = np.asarray(target, dtype=float).copy()
    if np.nanmax(lon) <= 180 and np.nanmin(lon) < 0:
        out[out > 180] -= 360
    elif np.nanmin(lon) >= 0:
        out[out < 0] += 360
    return out


def interp(da: xr.DataArray, zname: str | None, yname: str, xname: str,
           z: np.ndarray | None, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Interpolate one SODA field to arbitrary 2-D target points."""
    if zname:
        da = da.squeeze("time", drop=True).transpose(zname, yname, xname)
        values = np.asarray(da.values, dtype=float)
        src_z, values = regular_axis(da, zname, values, 0)
        src_y, values = regular_axis(da, yname, values, 1)
        src_x, values = regular_axis(da, xname, values, 2)
        points = np.column_stack((
            np.broadcast_to(z[:, None, None], (len(z),) + y.shape).ravel(),
            np.broadcast_to(y[None, :, :], (len(z),) + y.shape).ravel(),
            np.broadcast_to(x[None, :, :], (len(z),) + x.shape).ravel(),
        ))
        fn = RegularGridInterpolator((src_z, src_y, src_x), values,
                                     bounds_error=False, fill_value=np.nan)
        return fn(points).reshape((len(z),) + y.shape)
    da = da.squeeze("time", drop=True).transpose(yname, xname)
    values = np.asarray(da.values, dtype=float)
    src_y, values = regular_axis(da, yname, values, 0)
    src_x, values = regular_axis(da, xname, values, 1)
    fn = RegularGridInterpolator((src_y, src_x), values,
                                 bounds_error=False, fill_value=np.nan)
    return fn(np.column_stack((y.ravel(), x.ravel()))).reshape(y.shape)


def source_date(path: Path) -> np.datetime64:
    token = path.stem.rsplit("_", 3)[-3:]
    return np.datetime64("-".join(token))


def main() -> None:
    # Keep the parameter-file interface intentionally fixed: users edit this
    # file and run the script without command-line options.
    config_file = Path(__file__).with_name("soda_icbc_config.json")
    a = load_config(config_file)
    log_path = Path(a.log_file)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = setup_logging(log_path)
    log.info("Reading configuration: %s", config_file)
    log.info("Simulation period: %s to %s; initial-condition date: %s", a.start, a.end,
             a.initial_date or a.start)
    start = np.datetime64(a.start)
    end = np.datetime64(a.end)
    initial = np.datetime64(a.initial_date) if a.initial_date else start
    files = []
    for path in sorted(a.soda_dir.glob(a.pattern)):
        # Filename suffix is YYYY_MM_DD.
        date = np.datetime64(path.stem[-10:].replace("_", "-"))
        if start <= date <= end:
            files.append((date, path))
    if not files:
        raise FileNotFoundError(f"No SODA files in {a.soda_dir} between {a.start} and {a.end}")
    files.sort()
    ic_file = min(files, key=lambda item: abs(item[0] - initial))[1]
    log.info("Found %d SODA files; initial condition uses: %s", len(files), ic_file.name)

    log.info("Reading MOM6 hgrid: %s", a.hgrid)
    with xr.open_dataset(a.hgrid, decode_times=False) as grid:
        # MOM6 supergrid: tracer centers are odd/odd points; u and v use
        # odd/all and all/odd points, respectively.
        lon = np.asarray(grid["x"].values, dtype=float)
        lat = np.asarray(grid["y"].values, dtype=float)
        center_y, center_x = lat[1::2, 1::2], lon[1::2, 1::2]
        # MOM6 C-grid velocity points: u is (ny,nxp) and v is (nyp,nx).
        u_y, u_x = lat[1::2, 0::2], lon[1::2, 0::2]
        v_y, v_x = lat[0::2, 1::2], lon[0::2, 1::2]
        south_y, south_x = lat[0:1, :], lon[0:1, :]
        north_y, north_x = lat[-1:, :], lon[-1:, :]
        west_y, west_x = lat[:, 0:1], lon[:, 0:1]
    log.info("Grid sizes: tracer=%s, u=%s, v=%s", center_x.shape, u_x.shape,
             v_x.shape)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", a.output_dir)
    initial_fields = None
    obc_coords = {1: (south_y, south_x), 2: (north_y, north_x), 3: (west_y, west_x)}
    depth = None
    depth_edges = None
    writers = {}
    t0 = np.datetime64(a.start)
    for index, (date, path) in enumerate(files, 1):
        log.info("[%d/%d] Reading and interpolating: %s", index, len(files), path.name)
        with xr.open_dataset(path, decode_times=False) as ds:
            # SODA longitude is 0..360 in some releases and -280..80 in this one.
            source_lon = np.asarray(ds["xt_ocean"].values, dtype=float)
            source_lat = np.asarray(ds["yt_ocean"].values, dtype=float)
            source_xu = np.asarray(ds["xu_ocean"].values, dtype=float)
            source_yu = np.asarray(ds["yu_ocean"].values, dtype=float)
            center_xs = lon_for_source(source_lon, center_x)
            center_y_s = center_y
            u_xs = lon_for_source(source_xu, u_x)
            v_xs = lon_for_source(source_xu, v_x)
            z = np.asarray(ds["st_ocean"].values, dtype=float)
            if depth is None:
                depth = z
                depth_edges = np.asarray(ds["st_edges_ocean"].values, dtype=float)
            temp = interp(ds["temp"], "st_ocean", "yt_ocean", "xt_ocean", z, center_y_s, center_xs)
            salt = interp(ds["salt"], "st_ocean", "yt_ocean", "xt_ocean", z, center_y_s, center_xs)
            eta = interp(ds["ssh"], None, "yt_ocean", "xt_ocean", None, center_y_s, center_xs)
            u_center = interp(ds["u"], "st_ocean", "yu_ocean", "xu_ocean", z, u_y, u_xs)
            v_center = interp(ds["v"], "st_ocean", "yu_ocean", "xu_ocean", z, v_y, v_xs)
            temp = fill_nearest(temp)
            salt = fill_nearest(salt)
            eta = fill_nearest(eta)
            u_center = fill_nearest(u_center)
            v_center = fill_nearest(v_center)
            if date == np.datetime64(ic_file.stem[-10:].replace("_", "-")):
                initial_fields = (temp, salt, eta, u_center, v_center)

            for seg, (yy, xx) in {1: (south_y, south_x), 2: (north_y, north_x), 3: (west_y, west_x)}.items():
                # For OBCs use the same SODA C-grid variables and retain the
                # supergrid boundary lengths expected by MOM6.
                ty = lon_for_source(source_lat, yy)
                tx = lon_for_source(source_lon, xx)
                tu = interp(ds["u"], "st_ocean", "yu_ocean", "xu_ocean", z, ty, lon_for_source(source_xu, xx))
                tv = interp(ds["v"], "st_ocean", "yu_ocean", "xu_ocean", z, lon_for_source(source_yu, yy), tx)
                tt = interp(ds["temp"], "st_ocean", "yt_ocean", "xt_ocean", z, ty, tx)
                ts = interp(ds["salt"], "st_ocean", "yt_ocean", "xt_ocean", z, ty, tx)
                te = interp(ds["ssh"], None, "yt_ocean", "xt_ocean", None, ty, tx)
                tu = fill_nearest(tu)
                tv = fill_nearest(tv)
                tt = fill_nearest(tt)
                ts = fill_nearest(ts)
                te = fill_nearest(te)
                if seg not in writers:
                    writers[seg] = OBCWriter(
                        a.output_dir / f"forcing_obc_segment_{seg:03d}.nc", seg,
                        depth, np.diff(depth_edges), obc_coords[seg][0],
                        obc_coords[seg][1], t0)
                    if date > t0:
                        writers[seg].append(t0, tu, tv, te, tt, ts)
                writers[seg].append(date, tu, tv, te, tt, ts)
                log.info("[%d/%d] Wrote segment %03d time slice", index, len(files), seg)
        log.info("[%d/%d] Interpolation and missing-value filling complete", index, len(files))

    for writer in writers.values():
        writer.close()
    if initial_fields is None:
        raise RuntimeError("Initial-date SODA record was not selected")
    temp_initial, salt_initial, eta_initial, u_initial, v_initial = initial_fields
    encoding = {name: {"dtype": "f4", "_FillValue": FILL} for name in ("temp", "salt", "eta_t", "u", "v")}
    log.info("Writing initial temperature/salinity: init_tracers_filled.nc")
    xr.Dataset({
        "temp": (("zl", "ny", "nx"), temp_initial.astype("f4")),
        "salt": (("zl", "ny", "nx"), salt_initial.astype("f4")),
    }, coords={"zl": depth, "xh": (("ny", "nx"), center_x), "yh": (("ny", "nx"), center_y),
              "ny": np.arange(center_y.shape[0]), "nx": np.arange(center_x.shape[1])}).to_netcdf(a.output_dir / "init_tracers_filled.nc", encoding={k: encoding[k] for k in ("temp", "salt")})
    log.info("Writing initial sea-surface height: init_eta_filled.nc")
    xr.Dataset({"eta_t": (("ny", "nx"), eta_initial.astype("f4"))},
               coords={"xh": (("ny", "nx"), center_x), "yh": (("ny", "nx"), center_y),
                       "ny": np.arange(center_y.shape[0]), "nx": np.arange(center_x.shape[1])}).to_netcdf(a.output_dir / "init_eta_filled.nc", encoding={"eta_t": {"dtype": "f4", "_FillValue": FILL}})
    log.info("Writing initial velocity: init_vel_filled.nc")
    xr.Dataset({
        "u": (("zl", "ny", "nxp"), u_initial.astype("f4")),
        "v": (("zl", "nyp", "nx"), v_initial.astype("f4")),
    }, coords={"zl": depth, "xq": (("ny", "nxp"), u_x), "yh": (("ny", "nxp"), u_y),
              "xh": (("nyp", "nx"), v_x), "yq": (("nyp", "nx"), v_y),
              "ny": np.arange(u_y.shape[0]), "nxp": np.arange(u_x.shape[1]),
              "nyp": np.arange(v_y.shape[0]), "nx": np.arange(v_x.shape[1])}).to_netcdf(a.output_dir / "init_vel_filled.nc", encoding={"u": encoding["u"], "v": encoding["v"]})

    log.info("Complete: IC and 3 OBC files generated")
    log.info("Progress log: %s", log_path)


if __name__ == "__main__":
    main()
