async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${url}`);
  }
  return response.json();
}

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function formatMetric(value) {
  return Number(value).toFixed(3);
}

function renderSummary(summary) {
  setText("cityValue", summary.city);
  setText("rangeValue", `${summary.data_start} -> ${summary.data_end}`);
  setText("rowsValue", `${summary.train_rows} 行`);
  setText("latestValue", `${summary.latest_forecast_date} / ${summary.latest_update_time}`);
}

function renderPredictions(payload) {
  const root = document.getElementById("forecastCards");
  root.innerHTML = "";

  payload.days.forEach((day) => {
    const card = document.createElement("article");
    card.className = "forecast-card";
    card.innerHTML = `
      <div class="forecast-top">
        <span>${day.forecast_date}</span>
        <strong>Lead ${day.lead_day}</strong>
      </div>
      <h3>${day.predicted.textDay}</h3>
      <dl class="forecast-list">
        <div><dt>AI 最高温</dt><dd>${day.predicted.tempMax}°C</dd></div>
        <div><dt>原始最高温</dt><dd>${day.raw.tempMax}°C</dd></div>
        <div><dt>AI 最低温</dt><dd>${day.predicted.tempMin}°C</dd></div>
        <div><dt>原始最低温</dt><dd>${day.raw.tempMin}°C</dd></div>
        <div><dt>AI 降水</dt><dd>${day.predicted.precip} mm</dd></div>
        <div><dt>原始降水</dt><dd>${day.raw.precip} mm</dd></div>
        <div><dt>AI 天气</dt><dd>${day.predicted.textDay}</dd></div>
        <div><dt>原始天气</dt><dd>${day.raw.textDay}</dd></div>
      </dl>
    `;
    root.appendChild(card);
  });
}

function renderMetrics(backtest) {
  const metrics = backtest.metrics;
  const root = document.getElementById("metricsGrid");
  root.innerHTML = "";

  const entries = [
    ["tempMax MAE", formatMetric(metrics.tempMax_mae)],
    ["tempMax RMSE", formatMetric(metrics.tempMax_rmse)],
    ["tempMin MAE", formatMetric(metrics.tempMin_mae)],
    ["tempMin RMSE", formatMetric(metrics.tempMin_rmse)],
    ["precip MAE", formatMetric(metrics.precip_mae)],
    ["precip RMSE", formatMetric(metrics.precip_rmse)],
    ["textDay accuracy", formatMetric(metrics.textDay_accuracy)],
  ];

  entries.forEach(([label, value]) => {
    const item = document.createElement("article");
    item.className = "metric-card";
    item.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    root.appendChild(item);
  });
}

function drawLineChart(canvasId, labels, datasets, formatter) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = 36;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fffdf7";
  ctx.fillRect(0, 0, width, height);

  const allValues = datasets.flatMap((dataset) => dataset.values);
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const span = Math.max(maxValue - minValue, 1);

  ctx.strokeStyle = "#d4c9b5";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, padding / 2);
  ctx.lineTo(padding, height - padding);
  ctx.lineTo(width - padding / 2, height - padding);
  ctx.stroke();

  const xStep = (width - padding * 1.5) / Math.max(labels.length - 1, 1);

  for (let i = 0; i < labels.length; i += 1) {
    const x = padding + i * xStep;
    ctx.fillStyle = "#6a6150";
    ctx.font = "12px Georgia";
    ctx.fillText(labels[i].slice(5), x - 18, height - 10);
  }

  datasets.forEach((dataset) => {
    ctx.strokeStyle = dataset.color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    dataset.values.forEach((value, index) => {
      const x = padding + index * xStep;
      const y = height - padding - ((value - minValue) / span) * (height - padding * 1.6);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    dataset.values.forEach((value, index) => {
      const x = padding + index * xStep;
      const y = height - padding - ((value - minValue) / span) * (height - padding * 1.6);
      ctx.fillStyle = dataset.color;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = "11px Georgia";
      ctx.fillText(formatter(value), x - 10, y - 10);
    });
  });
}

function renderCharts(predictions) {
  const labels = predictions.days.map((day) => day.forecast_date);
  drawLineChart(
    "tempChart",
    labels,
    [
      { color: "#d94841", values: predictions.days.map((day) => day.predicted.tempMax) },
      { color: "#2f5fb3", values: predictions.days.map((day) => day.predicted.tempMin) },
    ],
    (value) => `${value.toFixed(1)}`
  );

  drawLineChart(
    "precipChart",
    labels,
    [{ color: "#25787d", values: predictions.days.map((day) => day.predicted.precip) }],
    (value) => `${value.toFixed(1)}`
  );
}

function renderQuality(quality) {
  document.getElementById("qualitySummary").innerHTML = `
    <div class="quality-pill">总文件 ${quality.total_files}</div>
    <div class="quality-pill">有效快照 ${quality.valid_snapshots}</div>
    <div class="quality-pill">异常快照 ${quality.invalid_snapshots}</div>
    <div class="quality-pill">缺失标签 ${quality.unresolved_targets}</div>
  `;

  const body = document.getElementById("qualityTable");
  body.innerHTML = "";
  quality.invalid_examples.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${item.fetch_date}</td>
      <td>${item.reason}</td>
      <td>${item.daily_count}</td>
    `;
    body.appendChild(row);
  });
}

async function loadDashboard() {
  const [summary, predictions, backtest, quality] = await Promise.all([
    fetchJson("/api/summary"),
    fetchJson("/api/predictions"),
    fetchJson("/api/backtest"),
    fetchJson("/api/quality"),
  ]);

  renderSummary(summary);
  renderPredictions(predictions);
  renderMetrics(backtest);
  renderCharts(predictions);
  renderQuality(quality);
}

document.getElementById("refreshButton").addEventListener("click", async () => {
  const button = document.getElementById("refreshButton");
  button.disabled = true;
  button.textContent = "刷新中...";
  try {
    await fetchJson("/api/refresh");
    await loadDashboard();
  } finally {
    button.disabled = false;
    button.textContent = "刷新模型缓存";
  }
});

loadDashboard().catch((error) => {
  document.body.insertAdjacentHTML(
    "beforeend",
    `<div class="error-banner">${error.message}</div>`
  );
});
