#!/usr/bin/env python3
"""Generate MOM6 IC and OBC files from SODA3.15.2 five-day NetCDF files.

It reads SODA's native variables (``temp``, ``salt``, ``u``, ``v``, ``ssh``),
interpolates them to the MOM6 supergrid, and applies the same important MOM6
regional-forcing rules used by CrocoDash: bathymetry-aware masks, local water
column thicknesses, and land/missing-value cleanup.
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
from scipy.spatial import cKDTree


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
        self.dz = np.asarray(dz, dtype="f4")
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
            if self.dz.ndim == 1:
                layer_dz = np.broadcast_to(self.dz[:, None, None], field.shape)
            else:
                layer_dz = np.broadcast_to(self.dz, field.shape)
            self.ds.variables[f"dz_{name}_segment_{self.tag}"][i] = layer_dz.astype("f4")
        self.ds.sync()

    def close(self) -> None:
        self.ds.close()


def fill_nearest(values: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    """Fill NaNs in each horizontal plane from the nearest valid value."""
    result = np.asarray(values, dtype="f4").copy()
    if result.ndim == 3:
        valid = np.isfinite(result) & (np.abs(result) < 0.5 * float(FILL))
        result[~valid] = np.nan
        if not valid.any():
            return np.full_like(result, fallback)
        if not valid.all():
            nearest = distance_transform_edt(~valid, return_distances=False,
                                             return_indices=True)
            result[~valid] = result[tuple(nearest[:, ~valid])]
        return result
    leading = result.shape[:-2]
    for index in np.ndindex(leading):
        plane = result[index]
        # SODA/NetCDF masked values can arrive as finite values such as 1e20.
        # Treat these sentinels as missing before nearest-neighbor filling.
        valid = np.isfinite(plane) & (np.abs(plane) < 0.5 * float(FILL))
        plane[~valid] = np.nan
        if not valid.any():
            result[index] = fallback
            continue
        if not valid.all():
            nearest = distance_transform_edt(~valid, return_distances=False,
                                             return_indices=True)
            result[index][~valid] = plane[tuple(nearest[:, ~valid])]
    return result


def fill_zero(values: np.ndarray) -> np.ndarray:
    """Replace NaNs and NetCDF missing-value sentinels with zero."""
    result = np.asarray(values, dtype="f4").copy()
    invalid = (~np.isfinite(result)) | (np.abs(result) >= 0.5 * float(FILL))
    result[invalid] = 0.0
    return result


def regular_axis(da: xr.DataArray, dim: str, values: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a strictly ascending coordinate and data reordered to match it."""
    coord = np.asarray(da[dim].values, dtype=float)
    order = np.argsort(coord)
    coord = coord[order]
    values = np.take(values, order, axis=axis)
    keep = np.r_[True, np.diff(coord) > 0.0]
    return coord[keep], np.take(values, np.flatnonzero(keep), axis=axis)


def time_weighted(before_date: np.datetime64, before: tuple[np.ndarray, ...],
                  after_date: np.datetime64, after: tuple[np.ndarray, ...],
                  target: np.datetime64) -> tuple[np.ndarray, ...]:
    """Linearly interpolate a group of fields to a target time."""
    denominator = (after_date - before_date) / np.timedelta64(1, "s")
    weight = ((target - before_date) / np.timedelta64(1, "s")) / denominator
    return tuple((1.0 - weight) * left + weight * right
                 for left, right in zip(before, after))


def load_config(path: Path) -> SimpleNamespace:
    with path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    required = ("soda_dir", "ocn_dir", "output_dir", "start", "end")
    missing = [name for name in required if name not in cfg]
    if missing:
        raise KeyError(f"Missing settings in {path}: {', '.join(missing)}")
    cfg["soda_dir"] = Path(cfg["soda_dir"])
    cfg["ocn_dir"] = Path(cfg["ocn_dir"])
    # The parameter is the MOM6 ``ocn`` directory.  This avoids hard-coding
    # hash-dependent hgrid/topog/vgrid filenames.
    ocn_dir = cfg["ocn_dir"]
    hgrids = sorted(ocn_dir.glob("ocean_hgrid_*.nc"))
    if len(hgrids) != 1:
        raise FileNotFoundError(
            f"Expected exactly one ocean_hgrid_*.nc in {ocn_dir}, found {len(hgrids)}"
        )
    cfg["hgrid"] = hgrids[0]
    cfg["output_dir"] = Path(cfg["output_dir"])
    # These are optional so existing parameter files remain valid.  For a
    # standard MOM6 input directory, infer them from ocean_hgrid automatically.
    hgrid_name = cfg["hgrid"].name
    cfg.setdefault("topog", str(cfg["hgrid"].with_name(hgrid_name.replace("hgrid", "topog"))))
    cfg.setdefault("vgrid", str(cfg["hgrid"].with_name(hgrid_name.replace("hgrid", "vgrid"))))
    cfg["topog"] = Path(cfg["topog"])
    cfg["vgrid"] = Path(cfg["vgrid"])
    cfg.setdefault("pattern", "soda3.15.2_5dy_ocean_or_*.nc")
    cfg.setdefault("log_file", "generate_soda_icbc.log")
    cfg.setdefault("initial_date", None)
    return SimpleNamespace(**cfg)


def load_case_grid(cfg: SimpleNamespace, center_x: np.ndarray,
                   center_y: np.ndarray) -> dict:
    """Load the case bathymetry and derive MOM6 C-grid masks.

    CrocoDash passes the case bathymetry into regional_mom6.  This local
    implementation uses the same information explicitly so SODA output does
    not contain source values over land or a full-depth column below the local
    ocean bottom.
    """
    with xr.open_dataset(cfg.topog, decode_times=False) as topo:
        mask = np.asarray(topo["mask"].values, dtype=bool)
        depth = np.asarray(topo["depth"].values, dtype=float)
        tx = np.asarray(topo["x"].values, dtype=float)
        ty = np.asarray(topo["y"].values, dtype=float)
    # The case topog is on tracer centers.  It is expected to match hgrid;
    # retain a clear error instead of silently applying a shifted mask.
    if mask.shape != center_x.shape:
        raise ValueError(f"topog mask shape {mask.shape} does not match tracer grid {center_x.shape}")
    # C-grid masks: a face is wet only when both neighboring tracer cells are
    # wet.  At the outer edge, use the adjacent tracer cell.
    umask = np.zeros((mask.shape[0], mask.shape[1] + 1), dtype=bool)
    umask[:, 1:-1] = mask[:, :-1] & mask[:, 1:]
    umask[:, 0] = mask[:, 0]
    umask[:, -1] = mask[:, -1]
    vmask = np.zeros((mask.shape[0] + 1, mask.shape[1]), dtype=bool)
    vmask[1:-1, :] = mask[:-1, :] & mask[1:, :]
    vmask[0, :] = mask[0, :]
    vmask[-1, :] = mask[-1, :]
    with xr.open_dataset(cfg.vgrid, decode_times=False) as vgrid:
        dz = np.asarray(vgrid["dz"].values, dtype=float)
    return {"mask": mask, "depth": depth, "umask": umask, "vmask": vmask,
            "topo_x": tx, "topo_y": ty, "dz": dz,
            "tree": cKDTree(np.column_stack((tx.ravel(), ty.ravel())))}


def nearest_depth(mask_grid: dict, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Nearest-neighbor case depth at arbitrary supergrid points."""
    _, indices = mask_grid["tree"].query(
        np.column_stack((x.ravel(), y.ravel())))
    return mask_grid["depth"].ravel()[indices].reshape(y.shape)


def nearest_mask(mask_grid: dict, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Nearest-neighbor wet mask at arbitrary supergrid points."""
    _, indices = mask_grid["tree"].query(
        np.column_stack((x.ravel(), y.ravel())))
    return mask_grid["mask"].ravel()[indices].reshape(y.shape)


def local_layer_thickness(dz: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Clip nominal MOM6 layer thicknesses to local bathymetry."""
    edges = np.concatenate(([0.0], np.cumsum(dz)))
    out = np.maximum(0.0, np.minimum(edges[1:, None, None], depth[None])
                     - edges[:-1, None, None])
    return out.astype("f4")


def apply_mask_and_depth(fields: tuple[np.ndarray, ...], masks: tuple[np.ndarray, ...],
                         dz: np.ndarray) -> tuple[np.ndarray, ...]:
    """Apply wet masks and remove values below each local ocean bottom."""
    u, v, eta, temp, salt = fields
    um, vm, tm = masks
    wet_u = um[None] & (dz > 0)
    wet_v = vm[None] & (dz > 0)
    wet_t = tm[None] & (dz > 0)
    u = np.where(wet_u, u, 0.0).astype("f4")
    v = np.where(wet_v, v, 0.0).astype("f4")
    temp = np.where(wet_t, temp, 0.0).astype("f4")
    salt = np.where(wet_t, salt, 0.0).astype("f4")
    eta = np.where(tm, eta, 0.0).astype("f4")
    return u, v, eta, temp, salt


def lon_for_source(lon: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Put target longitudes in the same convention as SODA."""
    out = np.asarray(target, dtype=float).copy()
    source_min = np.nanmin(lon)
    source_max = np.nanmax(lon)
    # Handle regional conventions such as SODA's -280..80 grid.  A target
    # longitude is shifted by full turns until it falls in the source range.
    for _ in range(3):
        out[out < source_min] += 360.0
        out[out > source_max] -= 360.0
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
    all_files = []
    for path in sorted(a.soda_dir.glob(a.pattern)):
        # Filename suffix is YYYY_MM_DD.
        date = np.datetime64(path.stem[-10:].replace("_", "-"))
        all_files.append((date, path))
    all_files.sort()
    inside = [i for i, (date, _) in enumerate(all_files) if start <= date <= end]
    if not inside:
        raise FileNotFoundError(f"No SODA files in {a.soda_dir} between {a.start} and {a.end}")
    first = max(0, inside[0] - 1)
    last = min(len(all_files), inside[-1] + 2)
    files = all_files[first:last]
    log.info("Found %d SODA files in the requested period plus boundary brackets", len(files))

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
    case_grid = load_case_grid(a, center_x, center_y)
    log.info("Using case bathymetry mask: %s", a.topog)
    log.info("Using case vertical grid metadata: %s", a.vgrid)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", a.output_dir)
    initial_fields = None
    obc_coords = {1: (south_y, south_x), 2: (north_y, north_x), 3: (west_y, west_x)}
    depth = None
    depth_edges = None
    writers = {}
    t0 = np.datetime64(a.start)
    previous_center = None
    previous_obc = {}
    started = set()
    ended = set()
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
            eta = fill_zero(eta)
            u_center = fill_zero(u_center)
            v_center = fill_zero(v_center)
            center_dz = local_layer_thickness(
                np.diff(depth_edges), case_grid["depth"])
            u_dz = local_layer_thickness(
                np.diff(depth_edges), nearest_depth(case_grid, u_y, u_x))
            v_dz = local_layer_thickness(
                np.diff(depth_edges), nearest_depth(case_grid, v_y, v_x))
            u_center = np.where((u_dz > 0.0) & case_grid["umask"][None],
                                u_center, 0.0).astype("f4")
            v_center = np.where((v_dz > 0.0) & case_grid["vmask"][None],
                                v_center, 0.0).astype("f4")
            temp = np.where((center_dz > 0.0) & case_grid["mask"][None],
                            temp, 0.0).astype("f4")
            salt = np.where((center_dz > 0.0) & case_grid["mask"][None],
                            salt, 0.0).astype("f4")
            eta = np.where(case_grid["mask"], eta, 0.0).astype("f4")
            center_fields = (u_center, v_center, eta, temp, salt)
            # Keep the historical tuple order used by time interpolation:
            # temp, salt, eta, u, v.
            center_fields = (center_fields[3], center_fields[4],
                             center_fields[2], center_fields[0], center_fields[1])
            if date == initial:
                initial_fields = center_fields
            elif (initial_fields is None and previous_center is not None
                  and previous_center[0] < initial < date):
                initial_fields = time_weighted(
                    previous_center[0], previous_center[1], date,
                    center_fields, initial)
                log.info("Time-weighted initial condition at %s", initial)
            previous_center = (date, center_fields)

            for seg, (yy, xx) in {1: (south_y, south_x), 2: (north_y, north_x), 3: (west_y, west_x)}.items():
                # For OBCs use the same SODA C-grid variables and retain the
                # supergrid boundary lengths expected by MOM6.
                ty = yy
                tx = lon_for_source(source_lon, xx)
                tu = interp(ds["u"], "st_ocean", "yu_ocean", "xu_ocean", z, ty, lon_for_source(source_xu, xx))
                tv = interp(ds["v"], "st_ocean", "yu_ocean", "xu_ocean", z, yy, tx)
                tt = interp(ds["temp"], "st_ocean", "yt_ocean", "xt_ocean", z, ty, tx)
                ts = interp(ds["salt"], "st_ocean", "yt_ocean", "xt_ocean", z, ty, tx)
                te = interp(ds["ssh"], None, "yt_ocean", "xt_ocean", None, ty, tx)
                tu = fill_zero(tu)
                tv = fill_zero(tv)
                tt = fill_nearest(tt)
                ts = fill_nearest(ts)
                te = fill_zero(te)
                boundary_mask = nearest_mask(case_grid, yy, xx)
                boundary_depth = nearest_depth(case_grid, yy, xx)
                boundary_dz = local_layer_thickness(
                    np.diff(depth_edges), boundary_depth)
                wet = boundary_dz > 0.0
                tu = np.where(wet & boundary_mask[None], tu, 0.0).astype("f4")
                tv = np.where(wet & boundary_mask[None], tv, 0.0).astype("f4")
                tt = np.where(wet & boundary_mask[None], tt, 0.0).astype("f4")
                ts = np.where(wet & boundary_mask[None], ts, 0.0).astype("f4")
                te = np.where(boundary_mask, te, 0.0).astype("f4")
                current = (tu, tv, te, tt, ts)
                if seg not in writers:
                    writers[seg] = OBCWriter(
                        a.output_dir / f"forcing_obc_segment_{seg:03d}.nc", seg,
                        depth, boundary_dz, obc_coords[seg][0],
                        obc_coords[seg][1], t0)
                previous = previous_obc.get(seg)
                if date < start:
                    pass
                elif seg not in started:
                    if previous is not None and previous[0] < start:
                        weighted = time_weighted(previous[0], previous[1], date,
                                                 current, start)
                        writers[seg].append(start, *weighted)
                        log.info("Time-weighted segment %03d start: %s", seg, start)
                    writers[seg].append(date, *current)
                    started.add(seg)
                    log.info("[%d/%d] Wrote segment %03d time slice", index, len(files), seg)
                elif date <= end:
                    writers[seg].append(date, *current)
                    log.info("[%d/%d] Wrote segment %03d time slice", index, len(files), seg)
                elif seg not in ended:
                    if previous is not None and previous[0] < end:
                        weighted = time_weighted(previous[0], previous[1], date,
                                                 current, end)
                        writers[seg].append(end, *weighted)
                        log.info("Time-weighted segment %03d end: %s", seg, end)
                    ended.add(seg)
                previous_obc[seg] = (date, current)
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
