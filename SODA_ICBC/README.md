# MOM6 SODA forcing

This directory contains a reusable SODA3.15.2-to-MOM6 IC/BC generator and a
staging utility. The generator accepts arbitrary simulation dates; it is not
limited to the current March 2020 test.

`/user/yang.yu/Clark/application/MOM6/crocodile2026/croc_cases/bering_ocn.001`

The current case expects these MOM6 products in its `ocn` input directory:

- `init_tracers_filled.nc` (`temp`, `salt`)
- `init_eta_filled.nc` (`eta_t`)
- `init_vel_filled.nc` (`u`, `v`)
- `forcing_obc_segment_001.nc`
- `forcing_obc_segment_002.nc`
- `forcing_obc_segment_003.nc`

## Generate from raw SODA

The SODA files use `temp`, `salt`, `u`, `v`, and `ssh`, with coordinates
`xt_ocean`, `yt_ocean`, `xu_ocean`, `yu_ocean`, and `st_ocean`. Edit
`soda_icbc_config.json`, then generate the MOM6 files with:

```bash
python generate_soda_icbc.py
```

For another period, change only `start`, `end`, and optionally `initial_date`
in the JSON file. If `initial_date` is omitted, the nearest available SODA
record to `start` is used for the initial condition. The OBC files contain
every SODA record between `start` and `end`; MOM6 performs time interpolation
between those records.

The generator writes `init_tracers_filled.nc`, `init_eta_filled.nc`,
`init_vel_filled.nc`, and the three `forcing_obc_segment_*.nc` files.

## Stage already-generated products

If another regridding workflow has already produced the six MOM6 files, stage
them atomically with:

```bash
python prepare_soda_icbc.py \
  --products /path/to/SODA_MOM6_products \
  --case /user/yang.yu/Clark/application/MOM6/crocodile2026/croc_cases/bering_ocn.001
```

Use `--force` only when intentionally replacing the current case input files.
The existing `user_nl_mom` configuration already points to the expected names
and uses three OBC segments: south (001), north (002), and west (003).
