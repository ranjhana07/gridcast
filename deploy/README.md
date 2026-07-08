# Deploy

`nginx.conf` is mounted into the dashboard container by `docker-compose.yml`. It serves the static dashboard and forwards `/api` and `/health` to the `api` service.

The compose stack was not run while building this repository because Docker was not available on the development machine. The services are:

| service | image | purpose |
| --- | --- | --- |
| postgres | timescale/timescaledb | backend store for MLflow |
| mlflow | ghcr.io/mlflow/mlflow | tracking server and model registry |
| api | built from `Dockerfile` | FastAPI service |
| scheduler | built from `Dockerfile` | daily retrain and hourly forecast loop |
| dashboard | nginx | static dashboard and API proxy |
