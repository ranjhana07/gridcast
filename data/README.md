# Data

Nothing in this folder is committed except this file. `raw`, `cache` and `processed` are created on demand and ignored by git.

## Load

UCI ElectricityLoadDiagrams20112014, 370 Portuguese customers at 15 minute resolution.

```
curl -L -o eld.zip https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
unzip eld.zip -d data/raw
```

`gridcast build-data --raw data/raw/LD2011_2014.txt` resamples to hourly means, keeps meters that are active from 2012 on (at most 2 percent zero hours), sums the first 30 of them into one microgrid series and writes `data/processed/microgrid_hourly.csv`.

The March daylight saving hour is zero for every meter in the source file. Those hours are treated as missing and filled by the gap filler.

PJM hourly load files (`Datetime,PJME_MW`) can be read with `gridcast.data.pjm.load_pjm`; they are not part of the benchmark.

## Weather

Hourly temperature, humidity, cloud cover, wind and shortwave radiation come from the Open-Meteo historical archive for Lisbon (38.72, -9.14). Responses are cached per calendar year under `data/cache`, so a second run makes no requests.

The benchmark treats observed weather at the target hour as the weather forecast. Real forecasts carry error, so results with real forecast inputs would be somewhat worse.

## Solar

The dataset has no PV measurements. `solar_kw` is simulated from Open-Meteo shortwave radiation for a 500 kW array (performance ratio 0.82, temperature derating). It is a synthetic target and only exercises the solar code path.
