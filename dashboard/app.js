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

function renderImportance(data) {
  if (!data || !data.families.length) {
    showEmpty("#chart-importance", "Feature importance has not been computed. Run gridcast importance.");
    return;
  }
  const rows = [...data.families].sort((a, b) => b.share - a.share);
  drawChart("importance", "#chart-importance", {
    ...baseOptions(260),
    chart: { ...baseOptions(260).chart, type: "bar" },
    series: [{ name: "Share of gain", data: rows.map((r) => +(r.share * 100).toFixed(2)) }],
    colors: [palette.point],
    plotOptions: { bar: { horizontal: true, barHeight: "60%" } },
    xaxis: { categories: rows.map((r) => r.family), labels: { formatter: (v) => v + "%" } },
    legend: { show: false },
    tooltip: { shared: false, y: { formatter: (v) => v + "%" } },
  });
}

function renderDispatch(plan) {
  if (!plan) {
    showEmpty("#chart-dispatch", "No dispatch schedule has been published yet.");
    return;
  }
  document.getElementById("dispatch-meta").textContent =
    "battery " + format(plan.battery.capacity_kwh) + " kWh / " + format(plan.battery.power_kw) + " kW";
  const x = plan.timestamps.map(toTime);
  drawChart("dispatch", "#chart-dispatch", {
    ...baseOptions(360),
    chart: { ...baseOptions(360).chart, type: "line", stacked: false },
    series: [
      { type: "column", name: "Charge kW", data: x.map((t, i) => [t, plan.charge[i]]) },
      { type: "column", name: "Discharge kW", data: x.map((t, i) => [t, -plan.discharge[i]]) },
      { type: "line", name: "State of charge kWh", data: x.map((t, i) => [t, plan.soc[i]]) },
      { type: "line", name: "Grid draw kW", data: x.map((t, i) => [t, plan.grid[i]]) },
      { type: "line", name: "Price $/kWh", data: x.map((t, i) => [t, plan.price[i]]) },
    ],
    colors: [palette.charge, palette.discharge, palette.soc, palette.actual, palette.price],
    stroke: { width: [0, 0, 3, 2, 2], curve: "stepline" },
    plotOptions: { bar: { columnWidth: "70%" } },
    xaxis: { type: "datetime", labels: { datetimeUTC: false } },
    yaxis: [
      { seriesName: "Charge kW", title: { text: "battery power, kW" }, labels: { formatter: (v) => format(v) } },
      { seriesName: "Charge kW", show: false },
      { opposite: true, title: { text: "state of charge, kWh" }, labels: { formatter: (v) => format(v) } },
      { seriesName: "Grid draw kW", show: false },
      { opposite: true, show: false, min: 0 },
    ],
  });
}

function renderTable(selector, columns, rows, bestKey) {
  const table = document.querySelector(selector);
  if (!rows.length) {
    table.innerHTML = "<tr><td>No data</td></tr>";
    return;
  }
  const best = bestKey ? Math.min(...rows.map((r) => r[bestKey])) : null;
  const head = "<tr>" + columns.map((c) => "<th>" + c.label + "</th>").join("") + "</tr>";
  const body = rows
    .map(
      (row) =>
        '<tr class="' +
        (bestKey && row[bestKey] === best ? "best" : "") +
        '">' +
        columns.map((c) => "<td>" + c.render(row) + "</td>").join("") +
        "</tr>"
    )
    .join("");
  table.innerHTML = head + body;
}

function renderMetricsTable(summary) {
  if (!summary) {
    renderTable("#table-metrics", [], []);
    return;
  }
  const coverage = Object.fromEntries(summary.coverage.map((r) => [r.model, r]));
  const timing = Object.fromEntries(summary.timing.map((r) => [r.model, r]));
  const rows = summary.metrics.map((r) => ({
    ...r,
    coverage: coverage[r.model] ? coverage[r.model].coverage : null,
    width: coverage[r.model] ? coverage[r.model].width : null,
    fit: timing[r.model] ? timing[r.model].fit_seconds : null,
  }));
  renderTable(
    "#table-metrics",
    [
      { label: "Model", render: (r) => r.model },
      { label: "MAE", render: (r) => format(r.mae, 1) },
      { label: "RMSE", render: (r) => format(r.rmse, 1) },
      { label: "sMAPE %", render: (r) => format(r.smape, 2) },
      { label: "Coverage %", render: (r) => (r.coverage === null ? "n/a" : format(r.coverage * 100, 1)) },
      { label: "Width kW", render: (r) => format(r.width, 0) },
      { label: "Fit s", render: (r) => format(r.fit, 1) },
    ],
    rows,
    "mae"
  );
}

function renderSavingsTable(data) {
  const rows = data ? data.strategies : [];
  renderTable(
    "#table-savings",
    [
      { label: "Strategy", render: (r) => r.strategy },
      { label: "Energy $", render: (r) => format(r.energy_cost) },
      { label: "Demand $", render: (r) => format(r.demand_cost) },
      { label: "Total $", render: (r) => format(r.total_cost) },
      { label: "Peak kW", render: (r) => format(r.peak_kw) },
      { label: "Saving %", render: (r) => format(r.saving_pct_vs_no_battery, 2) },
    ],
    rows,
    "total_cost"
  );
}

async function refresh() {
  const status = document.getElementById("status");
  const target = state.target;
  const [forecast, plan, summary, horizon, savings, importance] = await Promise.all([
    tryJson("/api/forecast?target=" + target),
    tryJson("/api/dispatch"),
    tryJson("/api/backtest/summary"),
    tryJson("/api/backtest/horizon"),
    tryJson("/api/dispatch/savings"),
    tryJson("/api/importance"),
  ]);
  const health = await tryJson("/health");
  status.textContent = health ? "online" : "api unreachable";
  status.className = health ? "pill" : "pill bad";
  renderKpis(forecast, target === "load_kw" ? plan : null, summary);
  renderForecast(forecast);
  renderHorizon(horizon);
  renderImportance(importance);
  renderDispatch(target === "load_kw" ? plan : null);
  renderMetricsTable(summary);
  renderSavingsTable(savings);
  if (summary && summary.meta) {
    const meta = summary.meta;
    document.getElementById("footer-meta").textContent =
      meta.rows.toLocaleString() + " hourly rows, " + meta.start.slice(0, 10) + " to " + meta.end.slice(0, 10) +
      ", " + meta.config.n_folds + " folds of " + meta.config.fold_hours / 24 + " days, " + meta.hardware.cpu;
  }
}

document.getElementById("target").addEventListener("change", (event) => {
  state.target = event.target.value;
  refresh();
});

refresh();
loadReplayChoices();
setInterval(refresh, 5 * 60 * 1000);
