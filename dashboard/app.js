const params = new URLSearchParams(window.location.search);
const API_BASE = (params.get("api") || "").replace(/\/$/, "");

const palette = {
  actual: "#64748b",
  point: "#0f766e",
  band: "#14b8a6",
  models: ["#0f766e", "#2563eb", "#d97706", "#9333ea", "#dc2626", "#0891b2"],
  charge: "#2563eb",
  discharge: "#d97706",
  soc: "#0f766e",
  price: "#9333ea",
};

const state = {
  target: "load_kw",
  charts: {},
  horizon: null,
  summary: null,
};

function prefersDark() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function baseOptions(height) {
  return {
    chart: {
      height,
      fontFamily: "inherit",
      toolbar: { show: false },
      animations: { enabled: false },
      background: "transparent",
    },
    theme: { mode: prefersDark() ? "dark" : "light" },
    grid: { borderColor: prefersDark() ? "#25303b" : "#e1e6eb" },
    dataLabels: { enabled: false },
    legend: { position: "top", horizontalAlign: "right" },
    tooltip: { shared: true, x: { format: "dd MMM HH:mm" } },
  };
}

async function getJson(path) {
  const response = await fetch(API_BASE + path);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || response.statusText);
  }
  return response.json();
}

async function tryJson(path) {
  try {
    return await getJson(path);
  } catch (error) {
    return null;
  }
}

function drawChart(key, selector, options) {
  if (state.charts[key]) {
    state.charts[key].destroy();
  }
  state.charts[key] = new ApexCharts(document.querySelector(selector), options);
  state.charts[key].render();
}

function showEmpty(selector, message) {
  const node = document.querySelector(selector);
  node.innerHTML = '<p class="hint" style="padding:24px 0">' + message + "</p>";
}

function toTime(value) {
  return new Date(value).getTime();
}

function format(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function renderKpis(forecast, dispatch, summary) {
  const cards = [];
  if (forecast) {
    const peak = Math.max(...forecast.point);
    const peakAt = forecast.timestamps[forecast.point.indexOf(peak)];
    const energy = forecast.point.reduce((a, b) => a + b, 0);
    cards.push(["Forecast peak", format(peak) + " kW", "at " + peakAt.slice(11, 16)]);
    cards.push(["Next 24h energy", format(energy / 1000, 1) + " MWh", "model " + forecast.model]);
    const width =
      forecast.upper.reduce((a, b, i) => a + b - forecast.lower[i], 0) / forecast.upper.length;
    const level = Math.round((forecast.interval.upper_level - forecast.interval.lower_level) * 100);
    cards.push(["Mean " + level + "% interval", format(width) + " kW", "calibrated with conformal"]);
  }
  if (dispatch) {
    const saving = dispatch.cost_without - dispatch.cost_with;
    cards.push([
      "Planned peak cut",
      format(dispatch.peak_without_kw - dispatch.peak_with_kw) + " kW",
      format(dispatch.peak_without_kw) + " to " + format(dispatch.peak_with_kw) + " kW",
    ]);
    cards.push(["Energy saving, next day", "$" + format(saving, 2), "before demand charge"]);
  }
  if (summary && summary.metrics.length) {
    const best = [...summary.metrics].sort((a, b) => a.mae - b.mae)[0];
    cards.push(["Best backtest MAE", format(best.mae, 1) + " kW", best.model]);
  }
  document.getElementById("kpis").innerHTML = cards
    .map(
      ([label, value, note]) =>
        '<div class="kpi"><div class="label">' +
        label +
        '</div><div class="value">' +
        value +
        '</div><div class="note">' +
        note +
        "</div></div>"
    )
    .join("");
}

function renderForecast(forecast) {
  if (!forecast) {
    showEmpty("#chart-forecast", "No forecast has been published yet. Run the forecast flow.");
    return;
  }
  document.getElementById("forecast-meta").textContent =
    "origin " + forecast.origin.replace("T", " ").slice(0, 16) + ", " + forecast.model;
  const history = forecast.history.timestamps
    .slice(-72)
    .map((t, i, all) => [toTime(t), forecast.history.values[forecast.history.values.length - all.length + i]]);
  const band = forecast.timestamps.map((t, i) => ({
    x: toTime(t),
    y: [forecast.lower[i], forecast.upper[i]],
  }));
  const point = forecast.timestamps.map((t, i) => [toTime(t), forecast.point[i]]);
  drawChart("forecast", "#chart-forecast", {
    ...baseOptions(340),
    series: [
      { type: "rangeArea", name: "Interval", data: band },
      { type: "line", name: "Forecast", data: point },
      { type: "line", name: "Observed", data: history },
    ],
    colors: [palette.band, palette.point, palette.actual],
    stroke: { width: [0, 3, 2], curve: "smooth" },
    fill: { opacity: [0.25, 1, 1] },
    xaxis: { type: "datetime", labels: { datetimeUTC: false } },
    yaxis: { labels: { formatter: (v) => format(v) }, title: { text: "kW" } },
  });
}

async function renderReplay() {
  const model = document.getElementById("replay-model").value;
  const origin = document.getElementById("replay-origin").value;
  if (!model || !origin) {
    showEmpty("#chart-replay", "No backtest predictions found. Run the benchmark command.");
    return;
  }
  const series = await tryJson(
    "/api/backtest/series?model=" + encodeURIComponent(model) + "&origin=" + encodeURIComponent(origin)
  );
  if (!series) {
    showEmpty("#chart-replay", "Could not load that origin.");
    return;
  }
  const start = toTime(series.origin);
  const hour = 3600 * 1000;
  const x = series.horizon.map((h) => start + h * hour);
  const data = [
    { type: "line", name: "Actual", data: x.map((t, i) => [t, series.actual[i]]) },
    { type: "line", name: "Forecast", data: x.map((t, i) => [t, series.point[i]]) },
  ];
  const colors = [palette.actual, palette.point];
  const widths = [2, 3];
  const opacity = [1, 1];
  if (series.lower) {
    data.unshift({
      type: "rangeArea",
      name: "Interval",
      data: x.map((t, i) => ({ x: t, y: [series.lower[i], series.upper[i]] })),
    });
    colors.unshift(palette.band);
    widths.unshift(0);
    opacity.unshift(0.25);
  }
  drawChart("replay", "#chart-replay", {
    ...baseOptions(320),
    series: data,
    colors,
    stroke: { width: widths, curve: "smooth" },
    fill: { opacity },
    xaxis: { type: "datetime", labels: { datetimeUTC: false } },
    yaxis: { labels: { formatter: (v) => format(v) }, title: { text: "kW" } },
  });
}

async function loadReplayChoices() {
  const info = await tryJson("/api/backtest/origins");
  const modelSelect = document.getElementById("replay-model");
  const originSelect = document.getElementById("replay-origin");
  if (!info) {
    showEmpty("#chart-replay", "No backtest predictions found. Run the benchmark command.");
    return;
  }
  const preferred = info.models.includes("lightgbm") ? "lightgbm" : info.models[0];
  modelSelect.innerHTML = info.models
    .map((m) => '<option value="' + m + '"' + (m === preferred ? " selected" : "") + ">" + m + "</option>")
    .join("");
  const fill = async () => {
    const origins = await tryJson("/api/backtest/origins?model=" + modelSelect.value);
    originSelect.innerHTML = origins.origins
      .map((o, i, all) => '<option value="' + o + '"' + (i === all.length - 1 ? " selected" : "") + ">" + o.replace("T", " ").slice(0, 16) + "</option>")
      .join("");
    await renderReplay();
  };
  modelSelect.onchange = fill;
  originSelect.onchange = renderReplay;
  await fill();
}

function modelColor(models, name) {
  return palette.models[models.indexOf(name) % palette.models.length];
}

function renderHorizon(data) {
  if (!data || !data.errors.length) {
    showEmpty("#chart-horizon", "No backtest tables found.");
    showEmpty("#chart-coverage", "No backtest tables found.");
    return;
  }
  const models = [...new Set(data.errors.map((r) => r.model))];
  const errorSeries = models.map((m) => ({
    name: m,
    data: data.errors.filter((r) => r.model === m).map((r) => [r.horizon, r.mae]),
  }));
  drawChart("horizon", "#chart-horizon", {
    ...baseOptions(300),
    chart: { ...baseOptions(300).chart, type: "line" },
    series: errorSeries,
    colors: models.map((m) => modelColor(models, m)),
    stroke: { width: 2, curve: "smooth" },
    xaxis: { type: "numeric", title: { text: "horizon (hours)" }, tickAmount: 12, decimalsInFloat: 0 },
    yaxis: { labels: { formatter: (v) => format(v) } },
    tooltip: { shared: true, intersect: false },
  });
  const coverage = data.coverage || [];
  if (!coverage.length) {
    showEmpty("#chart-coverage", "No interval coverage in this backtest.");
    return;
  }
  const covModels = [...new Set(coverage.map((r) => r.model))];
  document.getElementById("coverage-hint").textContent = "nominal 80%";
  drawChart("coverage", "#chart-coverage", {
    ...baseOptions(300),
    chart: { ...baseOptions(300).chart, type: "line" },
    series: covModels.map((m) => ({
      name: m,
      data: coverage.filter((r) => r.model === m).map((r) => [r.horizon, +(r.coverage * 100).toFixed(1)]),
    })),
    colors: covModels.map((m) => modelColor(models, m)),
    stroke: { width: 2, curve: "smooth" },
    xaxis: { type: "numeric", title: { text: "horizon (hours)" }, tickAmount: 12, decimalsInFloat: 0 },
    yaxis: { min: 50, max: 100, labels: { formatter: (v) => v + "%" } },
    annotations: { yaxis: [{ y: 80, borderColor: "#94a3b8", strokeDashArray: 4 }] },
    tooltip: { shared: true, intersect: false },
  });
}
