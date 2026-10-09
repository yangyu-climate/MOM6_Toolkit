#!/usr/bin/env python3
"""Download, process, and configure a multi-year ERA5 MOM6 forcing set."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


FIELDS = ("prec", "lwdn", "swdn", "q_10", "slp", "t_10", "u_10", "v_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).with_name("era5_forcing.json"),
        help="JSON parameter file (default: era5_forcing.json next to this script)",
    )
    parser.add_argument("--year", type=int, nargs="+", help="Override years")
    parser.add_argument("--year-start", type=int)
    parser.add_argument("--year-end", type=int)
    parser.add_argument("--grid", type=float, nargs=2, metavar=("LAT", "LON"))
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream)

    if args.year:
        years = args.year
    else:
        start = args.year_start if args.year_start is not None else config.get("year_start")
        end = args.year_end if args.year_end is not None else config.get("year_end")
        if start is None or end is None:
            years = config.get("years")
        else:
            years = list(range(int(start), int(end) + 1))
    if not years:
        raise ValueError("Specify years or year_start/year_end in the parameter file")
    if any(int(year) < 1940 for year in years):
        raise ValueError("ERA5 years must be >= 1940")
    config["years"] = [int(year) for year in years]
    config["_config_dir"] = args.config.parent.resolve()
    config["grid"] = args.grid or config.get("grid", [0.5, 0.5])
    if len(config["grid"]) != 2 or any(float(value) <= 0 for value in config["grid"]):
        raise ValueError("grid must contain two positive values")
    for key in ("part_root", "input_root", "output_dir"):
        if key not in config:
            raise KeyError(f"Missing {key!r} in {args.config}")
    config.setdefault("mesh_exe", "ESMF_Scrip2Unstruct")
    config.setdefault("streams_template", str(Path(__file__).with_name("user_nl_datm_streams")))
    return config


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def write_streams(config: dict, years: list[int], output: Path) -> Path:
    template = Path(config["streams_template"])
    if not template.is_absolute():
        template = config["_config_dir"] / template
    text = template.read_text(encoding="utf-8")
    root = output.resolve().as_posix()
    replacements = {
        "YEAR_FIRST": str(min(years)),
        "YEAR_LAST": str(max(years)),
        "YEAR_ALIGN": str(min(years)),
        "MESH_ROOT": root,
    }
    for field in FIELDS:
        replacements[f"{field.upper()}_FILES"] = ",".join(
            f"{root}/ERA5.{year}.{field}.nc" for year in years
        )
    for old, new in replacements.items():
        text = text.replace(old, new)
    unresolved = [token for token in ("YEAR_FIRST", "YEAR_LAST", "YEAR_ALIGN", "MESH_ROOT") if token in text]
    if unresolved:
        raise ValueError(f"Unresolved stream template tokens: {', '.join(unresolved)}")
    destination = output / "user_nl_datm_streams"
    destination.write_text(text, encoding="utf-8")
    return destination


def main() -> None:
    args = parse_args()
    config = load_config(args)
    years = config["years"]
    script_dir = Path(__file__).parent
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    for year in years:
        part_dir = Path(config["part_root"]) / str(year)
        input_dir = Path(config["input_root"]) / str(year)
        grid = [str(value) for value in config["grid"]]
        run([
            sys.executable, str(script_dir / "download_era5_split.py"),
            "--year", str(year), "--grid", *grid,
            "--part-dir", str(part_dir), "--out-dir", str(input_dir),
        ])
        run([
            sys.executable, str(script_dir / "process_era5_for_mom6.py"),
            "--year", str(year), "--input-dir", str(input_dir),
            "--output-dir", str(output), "--mesh-exe", str(config["mesh_exe"]),
        ])

    streams = write_streams(config, years, output)
    print(f"Wrote {streams}")


if __name__ == "__main__":
    main()
