/**
 * Phenox Model Recovery - Frontend Application
 * Interacts with FastAPI backend to display real-time MLOps metrics,
 * accuracy charts, recovery events, and live predictions.
 */

// State
let chartInstance = null;
let autoRefreshTimer = null;
let isAutoRefreshEnabled = true;
let currentThreshold = 0.90;

// DOM Elements & Initialization
document.addEventListener("DOMContentLoaded", () => {
  initChart();
  initNavigation();
  initActionButtons();
  initPresetButtons();
  initPlaygroundSliders();
  
  // Initial data load
  refreshAllData();

  // Setup auto-refresh (5s interval)
  autoRefreshTimer = setInterval(() => {
    if (isAutoRefreshEnabled) {
      refreshAllData(true);
    }
  }, 5000);
});

// Toast notification helper
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  
  let iconSvg = "";
  if (type === "success") {
    iconSvg = `<svg viewBox="0 0 24 24" width="18" height="18" stroke="var(--accent-emerald)" fill="none" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
  } else if (type === "alert" || type === "danger") {
    iconSvg = `<svg viewBox="0 0 24 24" width="18" height="18" stroke="var(--accent-rose)" fill="none" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;
  } else {
    iconSvg = `<svg viewBox="0 0 24 24" width="18" height="18" stroke="var(--accent-cyan)" fill="none" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;
  }

  toast.innerHTML = `
    ${iconSvg}
    <span>${message}</span>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(50px)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// Master refresh function
async function refreshAllData(silent = false) {
  const refreshBtn = document.getElementById("btn-manual-refresh");
  if (refreshBtn && !silent) {
    refreshBtn.classList.add("spinning");
  }

  try {
    await Promise.all([
      fetchStatus(),
      fetchMonitoring(),
      fetchEvents(),
      fetchModels()
    ]);
  } catch (error) {
    console.error("Error refreshing dashboard data:", error);
  } finally {
    if (refreshBtn && !silent) {
      setTimeout(() => refreshBtn.classList.remove("spinning"), 500);
    }
  }
}

// 1. Fetch System Status & Model Highlights
async function fetchStatus() {
  const res = await fetch("/api/status");
  if (!res.ok) throw new Error("Status endpoint error");
  const data = await res.json();

  currentThreshold = data.threshold;

  // Header status chip
  const headerChip = document.getElementById("header-status-chip");
  if (headerChip) {
    headerChip.className = "status-chip";
    if (data.system_status_code === "DEGRADED") {
      headerChip.classList.add("degraded");
      headerChip.innerHTML = `<span class="pulse-dot degraded"></span> Model Degraded`;
    } else if (data.system_status_code === "RECOVERED") {
      headerChip.classList.add("recovered");
      headerChip.innerHTML = `<span class="pulse-dot"></span> Backup Active (Recovered)`;
    } else {
      headerChip.innerHTML = `<span class="pulse-dot"></span> System Operational`;
    }
  }

  // Sidebar health badge
  const sidebarHealth = document.getElementById("sidebar-health-mode");
  const sidebarDot = document.getElementById("sidebar-pulse-dot");
  if (sidebarHealth && sidebarDot) {
    sidebarHealth.textContent = data.system_status;
    sidebarDot.className = "pulse-dot";
    if (data.system_status_code === "DEGRADED") sidebarDot.classList.add("degraded");
    else if (data.system_status_code === "RECOVERED") sidebarDot.classList.add("recovering");
  }

  // KPI Overview Cards
  const kpiStatus = document.getElementById("kpi-system-status");
  const kpiActiveModel = document.getElementById("kpi-active-model");
  const kpiAccuracy = document.getElementById("kpi-current-accuracy");
  const kpiThreshold = document.getElementById("kpi-threshold");
  const kpiRecoveryCount = document.getElementById("kpi-recovery-count");

  if (kpiStatus) kpiStatus.textContent = data.system_status;
  if (kpiActiveModel) kpiActiveModel.textContent = data.active_model.key.toUpperCase();
  if (kpiAccuracy) {
    kpiAccuracy.textContent = `${data.overview.current_accuracy_pct}%`;
    kpiAccuracy.className = "kpi-value";
    if (data.active_model.is_below_threshold) {
      kpiAccuracy.style.color = "var(--accent-rose)";
    } else {
      kpiAccuracy.style.color = "var(--accent-emerald)";
    }
  }
  if (kpiThreshold) kpiThreshold.textContent = `${data.threshold_pct}%`;
  if (kpiRecoveryCount) kpiRecoveryCount.textContent = `${data.recovery_count}`;

  // Active Model Card
  renderActiveModelCard(data.active_model, data.threshold_pct);

  // Backup Model Card
  renderBackupModelCard(data.backup_model, data.active_model.key);

  // Update Recovery Pipeline Stages
  updatePipelineStages(data.recovery_status.current_stage);
}

function renderActiveModelCard(active, thresholdPct) {
  const nameEl = document.getElementById("active-model-name");
  const versionEl = document.getElementById("active-model-version");
  const statusEl = document.getElementById("active-model-status");
  const accNumEl = document.getElementById("active-model-acc-num");
  const accFillEl = document.getElementById("active-model-acc-fill");
  const algoEl = document.getElementById("active-model-algo");
  const pathEl = document.getElementById("active-model-path");
  const sizeEl = document.getElementById("active-model-size");
  const pipelineEl = document.getElementById("active-model-pipeline");

  if (nameEl) nameEl.textContent = active.name;
  if (versionEl) versionEl.textContent = active.version;
  
  if (statusEl) {
    statusEl.className = "model-status-indicator";
    if (active.status === "Healthy") {
      statusEl.classList.add("healthy");
      statusEl.innerHTML = `<span class="pulse-dot"></span> Healthy`;
    } else if (active.status === "Degraded") {
      statusEl.classList.add("degraded");
      statusEl.innerHTML = `<span class="pulse-dot degraded"></span> Degraded`;
    } else {
      statusEl.classList.add("healthy");
      statusEl.innerHTML = `<span class="pulse-dot"></span> ${active.status}`;
    }
  }

  if (accNumEl) {
    accNumEl.textContent = `${active.accuracy_pct}%`;
    accNumEl.className = "acc-number " + (active.is_below_threshold ? "red" : "green");
  }

  if (accFillEl) {
    accFillEl.style.width = `${Math.min(active.accuracy_pct, 100)}%`;
    accFillEl.className = "accuracy-bar-fill " + (active.is_below_threshold ? "degraded" : "healthy");
  }

  if (algoEl) algoEl.textContent = active.algorithm;
  if (pathEl) pathEl.textContent = active.path;
  if (sizeEl) sizeEl.textContent = active.file_size;
  if (pipelineEl) pipelineEl.textContent = (active.pipeline_steps || []).join(" -> ") || "StandardScaler -> Classifier";
}

function renderBackupModelCard(backup, activeKey) {
  const nameEl = document.getElementById("backup-model-name");
  const versionEl = document.getElementById("backup-model-version");
  const statusEl = document.getElementById("backup-model-status");
  const accNumEl = document.getElementById("backup-model-acc-num");
  const accFillEl = document.getElementById("backup-model-acc-fill");
  const algoEl = document.getElementById("backup-model-algo");
  const pathEl = document.getElementById("backup-model-path");
  const sizeEl = document.getElementById("backup-model-size");
  const btnPromote = document.getElementById("btn-promote-backup");

  if (nameEl) nameEl.textContent = backup.name;
  if (versionEl) versionEl.textContent = backup.version;

  if (statusEl) {
    statusEl.className = "model-status-indicator";
    if (activeKey === "backup") {
      statusEl.classList.add("healthy");
      statusEl.innerHTML = `<span class="pulse-dot"></span> Active in Production`;
    } else {
      statusEl.classList.add("standby");
      statusEl.innerHTML = `<span class="pulse-dot"></span> Hot Standby`;
    }
  }

  if (accNumEl) accNumEl.textContent = `${backup.accuracy_pct}%`;
  if (accFillEl) {
    accFillEl.style.width = `${backup.accuracy_pct}%`;
    accFillEl.className = "accuracy-bar-fill healthy";
  }

  if (algoEl) algoEl.textContent = backup.algorithm;
  if (pathEl) pathEl.textContent = backup.path;
  if (sizeEl) sizeEl.textContent = backup.file_size;

  if (btnPromote) {
    if (activeKey === "backup") {
      btnPromote.textContent = "Currently Active";
      btnPromote.disabled = true;
      btnPromote.className = "btn btn-secondary btn-sm";
    } else {
      btnPromote.textContent = "Promote to Active";
      btnPromote.disabled = false;
      btnPromote.className = "btn btn-primary btn-sm";
    }
  }
}

// 2. Initialize and Update Accuracy Monitoring Chart
function initChart() {
  const ctx = document.getElementById("accuracyChart");
  if (!ctx) return;

  const gradient = ctx.getContext("2d").createLinearGradient(0, 0, 0, 320);
  gradient.addColorStop(0, "rgba(6, 182, 212, 0.35)");
  gradient.addColorStop(1, "rgba(6, 182, 212, 0.0)");

  chartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Evaluated Accuracy (%)",
          data: [],
          borderColor: "#06b6d4",
          backgroundColor: gradient,
          borderWidth: 3,
          pointBackgroundColor: [],
          pointBorderColor: "#ffffff",
          pointBorderWidth: 2,
          pointRadius: 6,
          pointHoverRadius: 8,
          fill: true,
          tension: 0.35
        },
        {
          label: "Threshold (90.00%)",
          data: [],
          borderColor: "rgba(244, 63, 94, 0.85)",
          borderWidth: 2,
          borderDash: [6, 6],
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          backgroundColor: "rgba(15, 23, 42, 0.95)",
          borderColor: "rgba(148, 163, 184, 0.2)",
          borderWidth: 1,
          titleColor: "#f8fafc",
          bodyColor: "#94a3b8",
          padding: 12,
          boxPadding: 6,
          callbacks: {
            label: function(context) {
              const val = context.parsed.y;
              if (context.datasetIndex === 1) {
                return ` Safety Threshold: ${val.toFixed(2)}%`;
              }
              const isBelow = val < (currentThreshold * 100);
              return ` Accuracy: ${val.toFixed(2)}% ${isBelow ? "⚠️ (BELOW THRESHOLD)" : "✓ (Healthy)"}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            color: "rgba(255, 255, 255, 0.05)"
          },
          ticks: {
            color: "#64748b",
            font: { family: "JetBrains Mono", size: 11 }
          }
        },
        y: {
          min: 40,
          max: 105,
          grid: {
            color: "rgba(255, 255, 255, 0.05)"
          },
          ticks: {
            color: "#64748b",
            font: { family: "JetBrains Mono", size: 11 },
            callback: (v) => `${v}%`
          }
        }
      }
    }
  });
}

async function fetchMonitoring() {
  const res = await fetch("/api/monitoring");
  if (!res.ok) throw new Error("Monitoring endpoint error");
  const data = await res.json();

  if (!chartInstance) return;

  const points = data.points || [];
  const labels = points.map(p => {
    // Show only time if date is today
    const parts = p.timestamp.split(" ");
    return parts[1] || p.timestamp;
  });

  const accuracyData = points.map(p => p.accuracy);
  const thresholdLine = points.map(() => data.threshold_pct);

  // Dynamic point colors: red if below threshold, cyan/green if healthy
  const pointColors = points.map(p => {
    return p.accuracy < data.threshold_pct ? "#f43f5e" : "#10b981";
  });

  chartInstance.data.labels = labels;
  chartInstance.data.datasets[0].data = accuracyData;
  chartInstance.data.datasets[0].pointBackgroundColor = pointColors;
  chartInstance.data.datasets[1].data = thresholdLine;
  chartInstance.data.datasets[1].label = `Threshold (${data.threshold_pct}%)`;

  chartInstance.update();

  // Update summary stats under chart
  const minAcc = accuracyData.length ? Math.min(...accuracyData) : 0;
  const maxAcc = accuracyData.length ? Math.max(...accuracyData) : 0;
  const degradedCount = points.filter(p => p.accuracy < data.threshold_pct).length;

  const minEl = document.getElementById("stat-min-acc");
  const maxEl = document.getElementById("stat-max-acc");
  const degEl = document.getElementById("stat-deg-count");
  if (minEl) minEl.textContent = `${minAcc.toFixed(1)}%`;
  if (maxEl) maxEl.textContent = `${maxAcc.toFixed(1)}%`;
  if (degEl) degEl.textContent = `${degradedCount} incidents`;
}

// 3. Fetch & Render Recovery Events Timeline
async function fetchEvents() {
  const res = await fetch("/api/events");
  if (!res.ok) throw new Error("Events endpoint error");
  const data = await res.json();

  const container = document.getElementById("events-timeline-container");
  if (!container) return;

  const events = data.events || [];
  container.innerHTML = "";

  // Render events in reverse chronological order
  [...events].reverse().forEach(evt => {
    const item = document.createElement("div");
    item.className = "timeline-event-item";

    const severityClass = evt.severity || "info";

    item.innerHTML = `
      <div class="event-bullet ${severityClass}"></div>
      <div class="event-card">
        <div class="event-header">
          <div class="event-title">
            <span class="tag-badge" style="background: rgba(255,255,255,0.06); font-size: 0.72rem; padding: 2px 6px;">
              Stage ${evt.stage}: ${evt.stage_name}
            </span>
            <span>${evt.title}</span>
          </div>
          <span class="event-time">${evt.timestamp}</span>
        </div>
        <p class="event-desc">${evt.description}</p>
      </div>
    `;
    container.appendChild(item);
  });
}

function updatePipelineStages(activeStage) {
  for (let i = 1; i <= 5; i++) {
    const stepEl = document.getElementById(`stage-step-${i}`);
    if (!stepEl) continue;

    stepEl.className = "stage-step";
    if (i < activeStage) {
      stepEl.classList.add("completed");
    } else if (i === activeStage) {
      if (activeStage === 3) {
        stepEl.classList.add("alert");
      } else {
        stepEl.classList.add("active");
      }
    }
  }

  // Update track fill percentage
  const trackFill = document.getElementById("pipeline-track-fill");
  if (trackFill) {
    const pct = ((activeStage - 1) / 4) * 100;
    trackFill.style.width = `${pct}%`;
  }
}

// 4. Fetch & Render Model Registry Table
async function fetchModels() {
  const res = await fetch("/api/models");
  if (!res.ok) throw new Error("Models endpoint error");
  const data = await res.json();

  const tbody = document.getElementById("registry-table-body");
  if (!tbody) return;

  tbody.innerHTML = "";

  (data.models || []).forEach(m => {
    const tr = document.createElement("tr");

    let statusPill = "";
    if (m.is_active) {
      statusPill = `<span class="tag-badge" style="background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border: 1px solid rgba(16, 185, 129, 0.35);"><span class="pulse-dot"></span> Active</span>`;
    } else if (m.status === "Standby") {
      statusPill = `<span class="tag-badge" style="background: rgba(99, 102, 241, 0.15); color: var(--accent-indigo); border: 1px solid rgba(99, 102, 241, 0.35);">Standby</span>`;
    } else {
      statusPill = `<span class="tag-badge" style="background: rgba(148, 163, 184, 0.1); color: var(--text-secondary);">Registered</span>`;
    }

    const accColor = m.accuracy_pct < (currentThreshold * 100) ? "var(--accent-rose)" : "var(--accent-emerald)";

    tr.innerHTML = `
      <td>
        <div class="model-name-cell">
          <span class="model-primary-name">${m.name}</span>
          <span class="model-sub-desc">${m.version} &bull; ${m.type}</span>
        </div>
      </td>
      <td><span style="font-family: var(--font-mono); font-size: 0.8rem;">${m.algorithm}</span></td>
      <td>
        <span style="font-family: var(--font-display); font-weight: 700; color: ${accColor};">
          ${m.accuracy_pct}%
        </span>
      </td>
      <td>${statusPill}</td>
      <td><span style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-muted);">${m.file_size}</span></td>
      <td><span style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-muted);">${m.created_time}</span></td>
      <td>
        ${m.is_active ? 
          `<button class="btn btn-secondary btn-sm" disabled style="opacity: 0.5;">Active</button>` : 
          `<button class="btn btn-primary btn-sm" onclick="handleSwitchModel('${m.key}')">Activate</button>`
        }
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// 5. Interactive Actions
function initActionButtons() {
  // Manual refresh
  document.getElementById("btn-manual-refresh")?.addEventListener("click", () => {
    refreshAllData();
    showToast("Dashboard telemetry refreshed", "info");
  });

  // Run normal evaluation
  document.getElementById("btn-run-evaluation")?.addEventListener("click", async () => {
    try {
      showToast("Running active model evaluation on validation set...", "info");
      const res = await fetch("/api/monitor/evaluate", { method: "POST" });
      const data = await res.json();
      
      if (data.failover_triggered) {
        showToast(`Degradation detected! Accuracy: ${data.accuracy_pct}%. Failover to Backup activated!`, "alert");
      } else {
        showToast(`Evaluation completed. Model is healthy (${data.accuracy_pct}%).`, "success");
      }
      refreshAllData();
    } catch (e) {
      showToast("Evaluation request failed: " + e.message, "danger");
    }
  });

  // Simulate drift & failover
  document.getElementById("btn-simulate-drift")?.addEventListener("click", async () => {
    try {
      showToast("Simulating data drift & noise injection...", "warning");
      const res = await fetch("/api/monitor/simulate-drift", { method: "POST" });
      const data = await res.json();
      
      showToast(`Data drift simulated! Primary dropped to ${data.drift_accuracy_pct}%. Failover to Backup model completed!`, "alert");
      refreshAllData();
    } catch (e) {
      showToast("Simulation error: " + e.message, "danger");
    }
  });

  // Reset to primary
  document.getElementById("btn-reset-primary")?.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/recovery/reset", { method: "POST" });
      const data = await res.json();
      showToast("Model registry reset to 'primary'. System ready for demo.", "success");
      refreshAllData();
    } catch (e) {
      showToast("Reset error: " + e.message, "danger");
    }
  });

  // Promote backup button on backup card
  document.getElementById("btn-promote-backup")?.addEventListener("click", () => {
    handleSwitchModel("backup");
  });

  // Auto-refresh toggle
  const toggleAuto = document.getElementById("toggle-auto-refresh");
  if (toggleAuto) {
    toggleAuto.addEventListener("change", (e) => {
      isAutoRefreshEnabled = e.target.checked;
      showToast(`Auto-refresh ${isAutoRefreshEnabled ? "Enabled" : "Paused"}`, "info");
    });
  }

  // Predict button in playground
  document.getElementById("btn-predict")?.addEventListener("click", runPrediction);
}

window.handleSwitchModel = async function(modelKey) {
  try {
    const res = await fetch("/api/recovery/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_key: modelKey })
    });
    const data = await res.json();
    if (res.ok) {
      showToast(`Active model switched to '${modelKey}'`, "success");
      refreshAllData();
    } else {
      showToast(data.detail || "Error switching model", "danger");
    }
  } catch (e) {
    showToast("Switch failed: " + e.message, "danger");
  }
};

// 6. Live Prediction Playground
function initPresetButtons() {
  const presets = {
    setosa: [5.1, 3.5, 1.4, 0.2],
    versicolor: [5.9, 3.0, 4.2, 1.5],
    virginica: [6.5, 3.0, 5.2, 2.0]
  };

  document.querySelectorAll(".preset-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      const flower = pill.getAttribute("data-preset");
      if (presets[flower]) {
        const [sl, sw, pl, pw] = presets[flower];
        setFeatureInputs(sl, sw, pl, pw);
        runPrediction();
      }
    });
  });
}

function initPlaygroundSliders() {
  const features = ["sepal_length", "sepal_width", "petal_length", "petal_width"];
  features.forEach(f => {
    const slider = document.getElementById(`slider-${f}`);
    const num = document.getElementById(`num-${f}`);
    if (slider && num) {
      slider.addEventListener("input", (e) => { num.value = e.target.value; });
      num.addEventListener("input", (e) => { slider.value = e.target.value; });
    }
  });
}

function setFeatureInputs(sl, sw, pl, pw) {
  const vals = {
    sepal_length: sl,
    sepal_width: sw,
    petal_length: pl,
    petal_width: pw
  };
  for (const [k, v] of Object.entries(vals)) {
    const slider = document.getElementById(`slider-${k}`);
    const num = document.getElementById(`num-${k}`);
    if (slider) slider.value = v;
    if (num) num.value = v;
  }
}

async function runPrediction() {
  const sl = parseFloat(document.getElementById("num-sepal_length")?.value || 5.1);
  const sw = parseFloat(document.getElementById("num-sepal_width")?.value || 3.5);
  const pl = parseFloat(document.getElementById("num-petal_length")?.value || 1.4);
  const pw = parseFloat(document.getElementById("num-petal_width")?.value || 0.2);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sepal_length: sl,
        sepal_width: sw,
        petal_length: pl,
        petal_width: pw
      })
    });
    if (!res.ok) throw new Error("Prediction API error");
    const data = await res.json();

    renderPredictionResult(data);
  } catch (e) {
    showToast("Prediction error: " + e.message, "danger");
  }
}

function renderPredictionResult(data) {
  const flowerNameEl = document.getElementById("pred-flower-name");
  const modelPillEl = document.getElementById("pred-model-pill");
  const latencyEl = document.getElementById("pred-latency");
  const flowerEmojiEl = document.getElementById("pred-flower-emoji");

  if (flowerNameEl) flowerNameEl.textContent = `Iris ${data.predicted_class_name}`;
  if (modelPillEl) modelPillEl.textContent = `Served by: ${data.active_model.toUpperCase()}`;
  if (latencyEl) latencyEl.textContent = `${data.latency_ms} ms`;

  const flowerEmojis = {
    setosa: "🌸",
    versicolor: "🌺",
    virginica: "🪻"
  };
  if (flowerEmojiEl) {
    flowerEmojiEl.textContent = flowerEmojis[data.predicted_class_name.toLowerCase()] || "🌼";
  }

  // Render probability bars
  const probs = data.probabilities || {};
  for (const [cls, pct] of Object.entries(probs)) {
    const fillEl = document.getElementById(`prob-fill-${cls.toLowerCase()}`);
    const textEl = document.getElementById(`prob-pct-${cls.toLowerCase()}`);
    if (fillEl) fillEl.style.width = `${pct}%`;
    if (textEl) textEl.textContent = `${pct}%`;
  }
}

// 7. Sidebar Navigation Switching
function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach(item => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      navItems.forEach(i => i.classList.remove("active"));
      item.classList.add("active");

      const targetId = item.getAttribute("data-target");
      if (targetId) {
        const targetSection = document.getElementById(targetId);
        if (targetSection) {
          targetSection.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }
    });
  });
}
