# ERA5 Forcing for MOM6 in the CESM3 Framework

This repository provides a workflow for downloading ERA5 single-level atmospheric data and converting it into forcing files for MOM6 simulations based on the CESM3 framework. The generated files are configured for use with the CESM Data Atmosphere (DATM) component.

Author: Yang Yu <yang.yu@whoi.edu>  
Copyright © 2026 Yang Yu. All rights reserved.

The workflow accepts any supported year and regular latitude–longitude grid spacing supported by the CDS request. Defaults are **2022** and **0.5° × 0.5°**.

## Workflow

```text
Copernicus Climate Data Store
            │
            ▼
download_era5_split.py
            │  monthly ERA5 NetCDF files
            ▼
process_era5_for_mom6.py
            │  MOM6/DATM variables + SCRIP grid
            ▼
prepare_era5_for_mom6.py + era5_forcing.json
            │
            ▼
user_nl_datm_streams
            │
            ▼
       CESM3 / MOM6 / DATM
```

The main files are:

| File | Purpose |
|---|---|
| [`download_era5_split.py`](download_era5_split.py) | Downloads ERA5 data month by month and merges variables into monthly files. |
| [`process_era5_for_mom6.py`](process_era5_for_mom6.py) | Converts ERA5 variables to CESM3/MOM6/DATM-compatible forcing files and creates a SCRIP grid. |
| [`user_nl_datm_streams`](user_nl_datm_streams) | Overrides CESM3 DATM stream settings and points them to the generated forcing files. |
| [`era5_forcing.json`](era5_forcing.json) | Parameter file containing paths, year range, grid, and mesh executable. |
| [`prepare_era5_for_mom6.py`](prepare_era5_for_mom6.py) | Runs the multi-year download/conversion workflow and generates `user_nl_datm_streams`. |

## Input variables

The downloader requests the following ERA5 single-level variables:

- 10 m u-component of wind
- 10 m v-component of wind
- 2 m temperature
- 2 m dewpoint temperature
- Mean sea-level pressure
- Surface pressure
- Total precipitation
- Surface solar radiation downward
- Surface thermal radiation downward

The requested data are hourly and regridded by CDS to a 0.5° × 0.5° grid.

## Output variables

The processing script writes one NetCDF file per atmospheric field for the CESM3/MOM6 DATM interface:

| Output field | Description | Units |
|---|---|---|
| `prec` | Precipitation flux | `kg m-2 s-1` |
| `lwdn` | Downward longwave radiation | `W m-2` |
| `swdn` | Downward shortwave radiation | `W m-2` |
| `q_10` | Specific humidity at 10 m | `kg kg-1` |
| `slp` | Mean sea-level pressure | `Pa` |
| `t_10` | Near-surface air temperature | `K` |
| `u_10` | Eastward wind at 10 m | `m s-1` |
| `v_10` | Northward wind at 10 m | `m s-1` |

The output time axis uses a **no-leap calendar** to match the JRA-compatible DATM stream configuration. February 29 is removed from leap years.

The script also creates:

- `ERA5_SCRIP.nc` — SCRIP grid description
- `ERA5_ESMFmesh.nc` — ESMF unstructured mesh, when `ESMF_Scrip2Unstruct` is available

## Requirements

### Python

Python 3.9 or newer is recommended. Install the Python dependencies with:

```bash
python -m pip install cdsapi xarray cftime numpy netCDF4 dask
```

### CDS API credentials

The downloader uses `cdsapi`. Configure the Copernicus Climate Data Store credentials according to the CDS documentation before running it. The account must have access to the `reanalysis-era5-single-levels` dataset.

### ESMF, optional

`ESMF_Scrip2Unstruct` is optional for the Python workflow, but it is required if the ESMF mesh should be generated automatically:

```bash
ESMF_Scrip2Unstruct ERA5_MOM6_2022/ERA5_SCRIP.nc \
                    ERA5_MOM6_2022/ERA5_ESMFmesh.nc 0 ESMF
```

## Usage

Run the complete multi-year workflow from the repository directory:

```bash
python prepare_era5_for_mom6.py
```

The script automatically reads `era5_forcing.json` from the same directory.
The default parameter file downloads the range `year_start` through
`year_end`. Command-line options can override the range or grid:

Runtime information is printed to the terminal and written to:

```text
era5_forcing.log
```

The log contains download and conversion commands for each year, subprocess
output, and the path of the generated `user_nl_datm_streams` file.

To use another parameter file, pass `--config` explicitly:

```bash
python prepare_era5_for_mom6.py \
    --config /path/to/custom_era5_forcing.json \
    --year-start 2019 --year-end 2022 \
    --grid 0.5 0.5
```

For each year, monthly source files are stored below `part_root` and merged
files below `input_root`. Processed files for all years are written to the
single `output_dir`, and the final `output_dir/user_nl_datm_streams` contains
all yearly files in each DATM `datafiles` entry.

The individual download and conversion scripts remain available for debugging:

### 1. Download and merge ERA5 data

```bash
python download_era5_split.py --year 2022 --grid 0.5 0.5
```

This creates:

```text
ERA5_2022_parts/ERA5.0p5x0p5.2022.MM.<variable>.nc
ERA5_2022/ERA5.0p5x0p5.2022.MM.nc
```

For example, download 2019 at 1° × 1° resolution:

```bash
python download_era5_split.py --year 2019 --grid 1 1
```

Use `--part-dir` and `--out-dir` to override the automatically generated directories.

Existing non-empty files are skipped, so interrupted downloads can be resumed.

### 2. Convert the monthly files

```bash
python process_era5_for_mom6.py \
    --year 2022 \
    --input-dir ERA5_2022 \
    --output-dir ERA5_MOM6_2022
```

For a different year and resolution:

```bash
python download_era5_split.py --year 2019 --grid 1 1
python process_era5_for_mom6.py \
    --year 2019 \
    --input-dir ERA5_2019 \
    --output-dir ERA5_MOM6_2019
```

This creates:

```text
ERA5_MOM6_2022/ERA5.2022.prec.nc
ERA5_MOM6_2022/ERA5.2022.lwdn.nc
ERA5_MOM6_2022/ERA5.2022.swdn.nc
ERA5_MOM6_2022/ERA5.2022.q_10.nc
ERA5_MOM6_2022/ERA5.2022.slp.nc
ERA5_MOM6_2022/ERA5.2022.t_10.nc
ERA5_MOM6_2022/ERA5.2022.u_10.nc
ERA5_MOM6_2022/ERA5.2022.v_10.nc
ERA5_MOM6_2022/ERA5_SCRIP.nc
ERA5_MOM6_2022/ERA5_ESMFmesh.nc
```

### 3. Configure CESM3 DATM

Copy the generated `output_dir/user_nl_datm_streams` into the CESM3/MOM6 case configuration. The template file is not intended to be copied directly; it contains placeholders that are filled by `prepare_era5_for_mom6.py`.

```text
CORE_IAF_JRA_1p5_2023.GCGCS.*:meshfile
CORE_IAF_JRA_1p5_2023.*:datafiles
```


## Configuration

The scripts no longer require source-code edits for a different year or resolution. Use `--help` to see all options:

```bash
python download_era5_split.py --help
python process_era5_for_mom6.py --help
```

When using the combined workflow, the year settings and multi-year
`datafiles` entries are generated automatically. For manual template editing,
the relevant settings are:

```text
...:year_first = YYYY
...:year_last = YYYY
...:year_align = YYYY
```

## Important scientific and technical notes

- ERA5 precipitation and radiation fields are hourly accumulations. The processing script converts them to fluxes by dividing by 3600 seconds.
- Specific humidity is calculated from 2 m dewpoint temperature and surface pressure.
- ERA5 2 m air temperature is written as `t_10` because the MOM6 atmospheric forcing interface expects the corresponding near-surface temperature field.
- Latitude is normalized to south-to-north order.
- The output uses CF-style metadata and a no-leap time axis.
- Verify the DATM stream interpolation, calendar, and accumulation conventions for the specific MOM6/CESM configuration before production runs.
- Verify that the generated SCRIP/ESMF mesh is accepted by the installed ESMF and DATM versions.

## Reproducibility

For a reproducible production run, record:

- ERA5 dataset version and retrieval date
- CDS request parameters
- Python and package versions
- CESM3/MOM6 commit or release
- ESMF version
- The exact values of `--year`, `--grid`, `--input-dir`, and `--output-dir`

## License and data attribution

Copyright © 2026 Yang Yu. All rights reserved.

This repository contains processing code and configuration. Unless otherwise stated, the code is not licensed for redistribution, modification, or commercial use without the author's permission.

ERA5 data are provided by the Copernicus Climate Change Service / Copernicus Climate Data Store and remain subject to their terms of use and attribution requirements.

If an open-source license is intended, replace this notice with the selected license text before publishing a formal release.
