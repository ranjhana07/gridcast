# Dashboard

Static page built with plain HTML, CSS and JavaScript. Charts come from ApexCharts loaded from jsDelivr, so the browser needs internet access the first time.

Panels:

- 24 hour forecast with the conformal interval and the last three days of observations
- backtest replay: pick a model and a forecast origin and compare against what happened
- MAE and interval coverage by horizon for every benchmarked model
- dispatch schedule: charge, discharge, state of charge, grid draw and tariff price
- LightGBM feature importance by family
- backtest summary and dispatch cost tables

The page talks to the API on the same origin. To point it somewhere else, open it with `?api=http://host:8000`.

Serving options:

- `gridcast serve` mounts this folder at `/` next to the API
- the compose stack serves it from nginx and proxies `/api` to the API container
