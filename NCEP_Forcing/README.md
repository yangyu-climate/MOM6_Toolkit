# NCEP forcing generator

Edit `ncep_forcing.json` to choose the input directory, output directory, and
year range. For example, `year_start: 2018` and `year_end: 2021` generates
2018, 2019, 2020, and 2021. The script automatically reads this file from the
same directory as the script. Then run:

```bash
python prepare_ncep_for_mom6.py
```

An alternate parameter file can be selected explicitly:

```bash
python prepare_ncep_for_mom6.py --config ncep_forcing_test.json
```

For all selected years together, the script writes one merged file per DATM
field:

```text
NCEP.prec.nc
NCEP.lwdn.nc
NCEP.swdn.nc
NCEP.q_10.nc
NCEP.slp.nc
NCEP.t_10.nc
NCEP.u_10.nc
NCEP.v_10.nc
user_nl_datm_streams
```

It also writes one shared `NCEP_SCRIP.nc` grid. If `ESMF_Scrip2Unstruct`
is available, it creates `NCEP_ESMFmesh.nc` automatically.

To run a selected year, copy the generated stream file into the case:

```bash
cp /user/yang.yu/Data/NCEP/MOM6_DATM/user_nl_datm_streams \
   /user/yang.yu/Clark/application/MOM6/crocodile2026/croc_cases/bering_ocn.001/user_nl_datm_streams
```

Then regenerate the case namelists and submit it with the normal CIME commands:

```bash
./case.setup --reset
./case.build
./case.submit
```
