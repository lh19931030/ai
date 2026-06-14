const state = {
  datasetId: null,
  columns: [],
  targetColumn: null,
  taskType: "classification",
};

const modelOptions = {
  classification: [
    ["logistic_regression", "Logistic Regression"],
    ["random_forest_classifier", "Random Forest Classifier"],
    ["svm", "SVM"],
  ],
  regression: [
    ["linear_regression", "Linear Regression"],
    ["random_forest_regressor", "Random Forest Regressor"],
    ["svr", "SVR"],
  ],
};

const $ = (selector) => document.querySelector(selector);
const els = {
  csvFile: $("#csvFile"),
  dropZone: $("#dropZone"),
  uploadError: $("#uploadError"),
  datasetMeta: $("#datasetMeta"),
  headerToggle: $("#headerToggle"),
  targetSelect: $("#targetSelect"),
  classificationBtn: $("#classificationBtn"),
  regressionBtn: $("#regressionBtn"),
  modelSelect: $("#modelSelect"),
  statsBtn: $("#statsBtn"),
  trainBtn: $("#trainBtn"),
  statusText: $("#statusText"),
  taskPill: $("#taskPill"),
  shapeText: $("#shapeText"),
  previewTable: $("#previewTable"),
  statsTable: $("#statsTable"),
  statsState: $("#statsState"),
  distributionGrid: $("#distributionGrid"),
  distributionState: $("#distributionState"),
  trainState: $("#trainState"),
  metricsGrid: $("#metricsGrid"),
  chartWrap: $("#chartWrap"),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(5).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(value);
}

function setStatus(message, isError = false) {
  els.statusText.textContent = message;
  els.uploadError.textContent = isError ? message : "";
}

function enableControls(enabled) {
  [
    els.headerToggle,
    els.targetSelect,
    els.classificationBtn,
    els.regressionBtn,
    els.modelSelect,
    els.statsBtn,
    els.trainBtn,
  ].forEach((el) => {
    el.disabled = !enabled;
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed.");
  return payload;
}

function renderTable(container, columns, rows) {
  if (!rows.length) {
    container.className = "table-wrap empty";
    container.textContent = "No rows available";
    return;
  }
  const head = columns.map((column) => `<th title="${escapeHtml(column)}">${escapeHtml(column)}</th>`).join("");
  const body = rows
    .map((row) => {
      const cells = columns
        .map((column) => {
          const value = formatValue(row[column]);
          return `<td title="${escapeHtml(value)}">${escapeHtml(value)}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  container.className = "table-wrap";
  container.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderModels() {
  els.modelSelect.innerHTML = modelOptions[state.taskType]
    .map(([value, label]) => `<option value="${value}">${label}</option>`)
    .join("");
}

function renderTask() {
  els.classificationBtn.classList.toggle("active", state.taskType === "classification");
  els.regressionBtn.classList.toggle("active", state.taskType === "regression");
  els.taskPill.textContent = state.taskType === "classification" ? "Classification" : "Regression";
}

function applySummary(summary) {
  state.datasetId = summary.dataset_id;
  state.columns = summary.columns;
  state.targetColumn = summary.target_column;
  state.taskType = summary.task_type;

  els.datasetMeta.textContent = summary.filename;
  els.headerToggle.checked = summary.has_header;
  els.targetSelect.innerHTML = summary.columns.map((column) => `<option value="${escapeHtml(column)}">${escapeHtml(column)}</option>`).join("");
  els.targetSelect.value = state.targetColumn;
  els.shapeText.textContent = `${summary.rows.toLocaleString()} rows / ${summary.columns.length.toLocaleString()} columns`;

  renderTable(els.previewTable, summary.columns, summary.preview);
  renderTask();
  renderModels();
  enableControls(true);
  setStatus("Dataset loaded.");

  els.statsTable.className = "table-wrap empty";
  els.statsTable.textContent = "Run stats after upload";
  els.distributionGrid.className = "dist-grid empty";
  els.distributionGrid.textContent = "No distributions yet";
  els.metricsGrid.className = "metrics empty";
  els.metricsGrid.textContent = "Train a model to see metrics";
  els.chartWrap.className = "chart empty";
  els.chartWrap.textContent = "No visualization yet";
  els.statsState.textContent = "Idle";
  els.distributionState.textContent = "Idle";
  els.trainState.textContent = "Idle";
}

async function uploadFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".csv")) {
    setStatus("Only CSV files are supported.", true);
    return;
  }
  enableControls(false);
  setStatus("Uploading dataset...");
  const form = new FormData();
  form.append("file", file);
  try {
    applySummary(await api("/api/upload", { method: "POST", body: form }));
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function reparse() {
  if (!state.datasetId) return;
  enableControls(false);
  setStatus("Updating header setting...");
  try {
    applySummary(
      await api("/api/reparse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: state.datasetId, has_header: els.headerToggle.checked }),
      })
    );
  } catch (error) {
    setStatus(error.message, true);
    enableControls(true);
  }
}

async function loadStats() {
  if (!state.datasetId) return;
  els.statsState.textContent = "Running";
  els.distributionState.textContent = "Running";
  setStatus("Calculating statistics...");
  try {
    const payload = await api("/api/stats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: state.datasetId }),
    });
    renderTable(els.statsTable, ["column", "dtype", "missing", "unique", "mean", "std"], payload.summaries);
    els.statsState.textContent = `${payload.summaries.length} columns`;
    els.distributionGrid.className = "dist-grid";
    els.distributionGrid.innerHTML = payload.distributions.map(renderDistribution).join("");
    els.distributionState.textContent = `${payload.distributions.length} charts`;
    setStatus("Statistics ready.");
  } catch (error) {
    els.statsState.textContent = "Error";
    els.distributionState.textContent = "Error";
    setStatus(error.message, true);
  }
}

function renderDistribution(dist) {
  const max = Math.max(...dist.values, 1);
  const rows = dist.labels
    .map((label, index) => {
      const value = dist.values[index];
      const width = Math.max(2, (value / max) * 100);
      return `<div class="bar-row"><span class="bar-label" title="${escapeHtml(label)}">${escapeHtml(label)}</span><span class="bar-track"><span class="bar-fill" style="width:${width}%"></span></span><span class="bar-value">${value}</span></div>`;
    })
    .join("");
  return `<article class="dist-card"><h4>${escapeHtml(dist.column)}</h4>${rows}</article>`;
}

async function trainModel() {
  if (!state.datasetId) return;
  els.trainState.textContent = "Training";
  els.trainBtn.disabled = true;
  setStatus("Training model...");
  try {
    const payload = await api("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: state.datasetId,
        target_column: state.targetColumn,
        task_type: state.taskType,
        model_name: els.modelSelect.value,
      }),
    });
    els.metricsGrid.className = "metrics";
    els.metricsGrid.innerHTML = Object.entries(payload.metrics)
      .map(([label, value]) => `<article class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(formatValue(value))}</strong></article>`)
      .join("");
    els.chartWrap.className = "chart";
    els.chartWrap.innerHTML = `<img src="${payload.visualization}" alt="Model visualization" />`;
    els.trainState.textContent = `${payload.train_rows} train / ${payload.test_rows} test`;
    setStatus("Model training complete.");
  } catch (error) {
    els.trainState.textContent = "Error";
    setStatus(error.message, true);
  } finally {
    els.trainBtn.disabled = false;
  }
}

els.csvFile.addEventListener("change", (event) => uploadFile(event.target.files[0]));
els.headerToggle.addEventListener("change", reparse);
els.targetSelect.addEventListener("change", () => {
  state.targetColumn = els.targetSelect.value;
});
els.classificationBtn.addEventListener("click", () => {
  state.taskType = "classification";
  renderTask();
  renderModels();
});
els.regressionBtn.addEventListener("click", () => {
  state.taskType = "regression";
  renderTask();
  renderModels();
});
els.statsBtn.addEventListener("click", loadStats);
els.trainBtn.addEventListener("click", trainModel);

["dragenter", "dragover"].forEach((eventName) => {
  els.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  els.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropZone.classList.remove("dragging");
  });
});
els.dropZone.addEventListener("drop", (event) => uploadFile(event.dataTransfer.files[0]));
