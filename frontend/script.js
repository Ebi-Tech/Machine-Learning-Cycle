// Handwritten Digit Recognition Pipeline - dashboard logic.
// Plain JS, no framework. Talks to the Flask API on the same origin.

let chartInstances = {};
let sessionUploadCount = 0;
let latestRetrainHistory = [];

const DIGIT_COLORS = [
  '#4F8FF7', '#F76E6E', '#4CD37B', '#F7C948', '#B27FF0',
  '#F7924F', '#4FD1F7', '#F74FA3', '#8AC24A', '#7B8CF7',
];

// ---------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------

const TAB_TITLES = {
  dashboard: 'Dashboard',
  predict: 'Predict',
  visualizations: 'Visualizations',
  retrain: 'Retrain',
  history: 'History',
};

const NAV_BTN_ACTIVE =
  'w-full text-left px-4 py-3 rounded-lg transition-all duration-200 border-l-2 flex items-start gap-3 text-white bg-blue-600/15 border-blue-500';
const NAV_BTN_INACTIVE =
  'w-full text-left px-4 py-3 rounded-lg transition-all duration-200 border-l-2 flex items-start gap-3 text-gray-400 hover:text-gray-200 hover:bg-gray-800/50 border-transparent';

function switchTab(tabName) {
  if (!TAB_TITLES[tabName]) return;

  document.querySelectorAll('[data-panel]').forEach((panel) => {
    panel.classList.toggle('hidden', panel.id !== `panel-${tabName}`);
  });

  document.querySelectorAll('[data-tab]').forEach((btn) => {
    btn.className = btn.dataset.tab === tabName ? NAV_BTN_ACTIVE : NAV_BTN_INACTIVE;
  });

  const titleEl = document.getElementById('top-bar-title');
  if (titleEl) titleEl.textContent = TAB_TITLES[tabName];
}

// Mobile sidebar: below the lg breakpoint the sidebar collapses to an
// icons-only rail (handled by CSS). The hamburger button expands it into
// a full-width overlay with a backdrop; selecting a tab or tapping the
// backdrop closes it again.
function openSidebar() {
  document.getElementById('sidebar').classList.add('sidebar-open');
  document.getElementById('sidebar-backdrop').classList.remove('hidden');
}

function closeSidebar() {
  document.getElementById('sidebar').classList.remove('sidebar-open');
  document.getElementById('sidebar-backdrop').classList.add('hidden');
}

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (sidebar.classList.contains('sidebar-open')) {
    closeSidebar();
  } else {
    openSidebar();
  }
}

function initDeploymentBadge() {
  const badge = document.getElementById('deployment-badge');
  if (!badge) return;
  const isLocal = ['localhost', '127.0.0.1', ''].includes(window.location.hostname);
  if (isLocal) {
    badge.textContent = 'Local';
    badge.className = 'text-xs text-gray-400 bg-gray-800 rounded-full px-3 py-1';
  } else {
    badge.textContent = 'Cloud: Live';
    badge.className = 'text-xs text-green-400 bg-green-400/10 rounded-full px-3 py-1';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initDeploymentBadge();
  switchTab('dashboard');

  document.querySelectorAll('[data-tab]').forEach((btn) => {
    btn.addEventListener('click', () => {
      switchTab(btn.dataset.tab);
      closeSidebar();
    });
  });

  document.getElementById('sidebar-toggle').addEventListener('click', toggleSidebar);
  document.getElementById('sidebar-backdrop').addEventListener('click', closeSidebar);

  fetchHealth();
  setInterval(fetchHealth, 10000);

  fetchVisualizations();
  fetchRetrainHistory();

  document.getElementById('file-input').addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) runPrediction(e.target.files[0], 'full');
  });
  wireDropZone('drop-zone', (file) => runPrediction(file, 'full'));

  document.getElementById('quick-file-input').addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) runPrediction(e.target.files[0], 'quick');
  });
  wireDropZone('quick-drop-zone', (file) => runPrediction(file, 'quick'));

  document.getElementById('clear-btn').addEventListener('click', clearPrediction);

  document.getElementById('upload-images-btn').addEventListener('click', handleUpload);
  document.getElementById('upload-zip-btn').addEventListener('click', handleZipUpload);
  document.getElementById('retrain-btn').addEventListener('click', triggerRetrain);
});

function wireDropZone(zoneId, onFile) {
  const zone = document.getElementById(zoneId);
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('border-blue-500');
  });
  zone.addEventListener('dragleave', () => {
    zone.classList.remove('border-blue-500');
  });
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('border-blue-500');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFile(e.dataTransfer.files[0]);
    }
  });
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

function formatUptime(seconds) {
  const s = Math.max(0, Math.floor(seconds || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return `${h}h ${m}m ${sec}s`;
}

function formatAccuracy(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '--';
  return value.toFixed(4);
}

const ALERT_STYLES = {
  success: { border: 'border-green-500', text: 'text-green-400' },
  warning: { border: 'border-amber-500', text: 'text-amber-400' },
  error: { border: 'border-red-500', text: 'text-red-400' },
  info: { border: 'border-blue-500', text: 'text-blue-300' },
};

function showAlert(containerId, type, message) {
  const container = document.getElementById(containerId);
  const style = ALERT_STYLES[type] || ALERT_STYLES.info;
  container.innerHTML = `
    <div class="border-l-4 ${style.border} bg-gray-900/60 rounded-r-md px-3 py-2 text-sm ${style.text} mt-2">${message}</div>
  `;
}

// ---------------------------------------------------------------------
// Health / sidebar status / dashboard stat cards
// ---------------------------------------------------------------------

async function fetchHealth() {
  const statusBadge = document.getElementById('status-badge');
  const statusDot = document.getElementById('status-dot');
  const uptimeEl = document.getElementById('stat-uptime');
  const lastTrainedEl = document.getElementById('stat-last-trained');
  const paramsEl = document.getElementById('stat-params');
  const sidebarDot = document.getElementById('sidebar-status-dot');
  const sidebarText = document.getElementById('sidebar-status-text');
  const sidebarUptime = document.getElementById('sidebar-uptime-compact');

  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error('health check failed');
    const data = await res.json();

    const online = data.status === 'healthy' && data.model_loaded;

    statusBadge.textContent = online ? 'Online' : 'Offline';
    statusBadge.className = online
      ? 'inline-block text-sm font-semibold px-2.5 py-1 rounded-full text-green-400 bg-green-400/10 border border-green-400/20 shadow-[0_0_8px_rgba(46,160,67,0.4)]'
      : 'inline-block text-sm font-semibold px-2.5 py-1 rounded-full text-red-400 bg-red-400/10 border border-red-400/20';
    statusDot.classList.toggle('hidden', !online);

    uptimeEl.textContent = formatUptime(data.uptime_seconds);
    lastTrainedEl.textContent = data.model_last_trained
      ? new Date(data.model_last_trained).toLocaleString()
      : 'Unknown';
    paramsEl.textContent = data.model_parameters != null
      ? data.model_parameters.toLocaleString()
      : 'Unknown';

    sidebarDot.className = `w-2 h-2 rounded-full shrink-0 ${online ? 'bg-green-500' : 'bg-red-500'}`;
    sidebarText.textContent = online ? 'Online' : 'Offline';
    sidebarUptime.textContent = formatUptime(data.uptime_seconds);
  } catch (err) {
    statusBadge.textContent = 'Offline';
    statusBadge.className = 'inline-block text-sm font-semibold px-2.5 py-1 rounded-full text-red-400 bg-red-400/10 border border-red-400/20';
    statusDot.classList.add('hidden');
    uptimeEl.textContent = 'Unknown';
    lastTrainedEl.textContent = 'Unknown';
    paramsEl.textContent = 'Unknown';

    sidebarDot.className = 'w-2 h-2 rounded-full shrink-0 bg-red-500';
    sidebarText.textContent = 'Offline';
    sidebarUptime.textContent = '--';
  }
}

// ---------------------------------------------------------------------
// Prediction: shared logic for the full Predict tab and the Dashboard's
// Quick Predict widget. Both call the same /predict endpoint; only the
// DOM elements they update differ, and the quick version has no
// probability bars.
// ---------------------------------------------------------------------

const PREDICT_TARGETS = {
  full: {
    preview: 'preview-image',
    result: 'predict-result',
    digit: 'predicted-digit',
    confidence: 'confidence-value',
    procTime: 'processing-time-badge',
    bars: true,
  },
  quick: {
    preview: 'quick-preview-image',
    result: 'quick-predict-result',
    digit: 'quick-predicted-digit',
    confidence: 'quick-confidence-value',
    procTime: 'quick-processing-time-badge',
    bars: false,
  },
};

// Shows (or clears) a low-confidence warning banner above the predicted
// digit inside the given result container. Created and removed dynamically
// since there's no static placeholder for it in the HTML.
function showConfidenceWarning(resultContainerId, warningText) {
  const container = document.getElementById(resultContainerId);
  const existing = container.querySelector('.confidence-warning-banner');
  if (existing) existing.remove();

  if (warningText) {
    const banner = document.createElement('div');
    banner.className =
      'confidence-warning-banner text-amber-400 text-sm bg-amber-400/10 rounded-lg px-4 py-2 mb-4';
    banner.textContent = warningText;
    container.insertBefore(banner, container.firstChild);
  }
}

async function runPrediction(file, targetKey) {
  const t = PREDICT_TARGETS[targetKey];
  const previewImage = document.getElementById(t.preview);
  const objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;

  document.getElementById(t.result).classList.remove('hidden');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      document.getElementById(t.digit).textContent = '?';
      return;
    }

    showConfidenceWarning(t.result, data.warning);

    document.getElementById(t.digit).textContent = data.predicted_digit;

    const pct = Math.round(data.confidence * 100);
    document.getElementById(t.confidence).textContent = `${pct}% confidence`;
    document.getElementById(t.procTime).textContent = `${data.processing_time_ms.toFixed(1)} ms`;

    if (t.bars) {
      data.probabilities.forEach((p, digit) => {
        const barPct = Math.round(p * 100);
        const bar = document.getElementById(`prob-bar-${digit}`);
        const text = document.getElementById(`prob-pct-${digit}`);
        bar.style.width = `${barPct}%`;
        text.textContent = `${barPct}%`;
        bar.className = digit === data.predicted_digit
          ? 'h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-500 ease-out'
          : 'h-full rounded-full bg-gray-700 transition-all duration-500 ease-out';
      });
    }
  } catch (err) {
    document.getElementById(t.digit).textContent = '?';
  }
}

function clearPrediction() {
  showConfidenceWarning('predict-result', null);
  document.getElementById('predict-result').classList.add('hidden');
  document.getElementById('preview-image').src = '';
  document.getElementById('predicted-digit').textContent = '-';
  document.getElementById('confidence-value').textContent = '-- confidence';
  document.getElementById('processing-time-badge').textContent = '-- ms';
  for (let digit = 0; digit < 10; digit++) {
    const bar = document.getElementById(`prob-bar-${digit}`);
    const text = document.getElementById(`prob-pct-${digit}`);
    bar.style.width = '0%';
    text.textContent = '0%';
    bar.className = 'h-full rounded-full bg-gray-700 transition-all duration-500 ease-out';
  }
  document.getElementById('file-input').value = '';
}

function buildProbabilityBars() {
  const container = document.getElementById('probability-bars');
  container.innerHTML = '';
  for (let digit = 0; digit < 10; digit++) {
    const row = document.createElement('div');
    row.className = 'flex items-center gap-3';
    row.innerHTML = `
      <span class="w-8 text-right font-mono text-gray-400 text-sm">${digit}</span>
      <div class="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
        <div id="prob-bar-${digit}" class="h-full rounded-full bg-gray-700 transition-all duration-500 ease-out" style="width:0%"></div>
      </div>
      <span id="prob-pct-${digit}" class="w-12 text-right font-mono text-xs text-gray-500">0%</span>
    `;
    container.appendChild(row);
  }
}

// ---------------------------------------------------------------------
// Retrain panel: upload
// ---------------------------------------------------------------------

async function handleUpload() {
  const labelSelect = document.getElementById('label-select');
  const filesInput = document.getElementById('files-input');
  const label = labelSelect.value;

  if (!label) {
    showAlert('upload-alert-container', 'error', 'Select a digit label before uploading.');
    return;
  }
  if (!filesInput.files || filesInput.files.length === 0) {
    showAlert('upload-alert-container', 'error', 'Choose at least one image to upload.');
    return;
  }

  const formData = new FormData();
  for (const file of filesInput.files) formData.append('files', file);
  formData.append('label', label);

  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      showAlert('upload-alert-container', 'error', data.error || 'Upload failed.');
      return;
    }

    showAlert('upload-alert-container', 'success',
      `Uploaded ${data.uploaded} image(s) for label ${data.label}.`);

    sessionUploadCount += data.uploaded;
    document.getElementById('upload-tally-badge').textContent =
      `${sessionUploadCount} images uploaded this session`;

    filesInput.value = '';
    labelSelect.selectedIndex = 0;
  } catch (err) {
    showAlert('upload-alert-container', 'error', 'Upload failed: could not reach the server.');
  }
}

async function handleZipUpload() {
  const zipInput = document.getElementById('zip-input');

  if (!zipInput.files || zipInput.files.length === 0) {
    showAlert('upload-alert-container', 'error', 'Choose a .zip file to upload.');
    return;
  }

  const formData = new FormData();
  formData.append('files', zipInput.files[0]);

  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      showAlert('upload-alert-container', 'error', data.error || 'ZIP upload failed.');
      return;
    }

    let detail = `Uploaded ${data.uploaded} image(s) from ZIP.`;
    if (data.per_class) {
      const breakdown = Object.entries(data.per_class)
        .map(([digit, count]) => `${digit}: ${count}`)
        .join(', ');
      detail += ` (${breakdown})`;
    }
    showAlert('upload-alert-container', 'success', detail);

    sessionUploadCount += data.uploaded;
    document.getElementById('upload-tally-badge').textContent =
      `${sessionUploadCount} images uploaded this session`;

    zipInput.value = '';
  } catch (err) {
    showAlert('upload-alert-container', 'error', 'ZIP upload failed: could not reach the server.');
  }
}

// ---------------------------------------------------------------------
// Retrain panel: trigger retraining
// ---------------------------------------------------------------------

async function triggerRetrain() {
  const btn = document.getElementById('retrain-btn');
  const resultContainer = document.getElementById('retrain-result-container');

  btn.disabled = true;
  const originalText = btn.textContent;
  btn.innerHTML = `<span class="loading loading-spinner"></span> Retraining...`;
  resultContainer.innerHTML = '';

  try {
    const res = await fetch('/retrain', { method: 'POST' });
    const data = await res.json();

    if (!res.ok) {
      resultContainer.innerHTML = `
        <div class="border-l-4 border-red-500 bg-gray-900/60 rounded-r-md px-4 py-3 text-sm text-red-400">
          ${data.error || 'Retraining failed.'}
        </div>
      `;
      return;
    }

    const promoted = data.promoted;
    const borderClass = promoted ? 'border-green-500' : 'border-amber-500';
    const titleClass = promoted ? 'text-green-400' : 'text-amber-400';
    const title = promoted ? 'Model Updated' : 'Retraining Rejected';

    resultContainer.innerHTML = `
      <div class="border-l-4 ${borderClass} bg-gray-900/60 rounded-r-md px-4 py-3 text-sm">
        <div class="font-medium ${titleClass}">${title}</div>
        <div class="text-gray-400 text-xs space-y-0.5 mt-1.5">
          <div>Baseline accuracy: <span class="font-mono text-gray-300">${formatAccuracy(data.baseline_accuracy)}</span></div>
          <div>New accuracy: <span class="font-mono text-gray-300">${formatAccuracy(data.new_accuracy)}</span></div>
          <div>Change: <span class="font-mono text-gray-300">${(data.accuracy_change * 100).toFixed(2)} pp</span></div>
          <div>Samples used: <span class="font-mono text-gray-300">${data.samples_used}</span></div>
          <div class="text-gray-500 pt-1">${data.reason}</div>
        </div>
      </div>
    `;
  } catch (err) {
    resultContainer.innerHTML = `
      <div class="border-l-4 border-red-500 bg-gray-900/60 rounded-r-md px-4 py-3 text-sm text-red-400">
        Retraining failed: could not reach the server.
      </div>
    `;
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
    fetchHealth();
    fetchVisualizations();
    fetchRetrainHistory();
  }
}

// ---------------------------------------------------------------------
// Visualizations panel
// ---------------------------------------------------------------------

async function fetchVisualizations() {
  const infoAlert = document.getElementById('viz-info-alert');
  const vizGrid = document.getElementById('viz-grid');

  try {
    const res = await fetch('/visualizations');

    if (res.status === 404) {
      infoAlert.classList.remove('hidden');
      vizGrid.classList.add('hidden');
      return;
    }
    if (!res.ok) throw new Error('visualizations fetch failed');

    const data = await res.json();
    infoAlert.classList.add('hidden');
    vizGrid.classList.remove('hidden');

    Chart.defaults.color = '#e1e4e8';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.08)';

    if (chartInstances.classDistribution) chartInstances.classDistribution.destroy();
    if (chartInstances.f1) chartInstances.f1.destroy();

    const labels = Array.from({ length: 10 }, (_, i) => String(i));

    chartInstances.classDistribution = new Chart(
      document.getElementById('chart-class-distribution'),
      {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Training samples',
            data: data.class_distribution,
            backgroundColor: DIGIT_COLORS,
          }],
        },
        options: {
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false } },
            y: { beginAtZero: true },
          },
        },
      }
    );

    chartInstances.f1 = new Chart(
      document.getElementById('chart-f1'),
      {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'F1 score',
            data: data.per_class_f1,
            backgroundColor: DIGIT_COLORS,
          }],
        },
        options: {
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false } },
            y: { min: 0.95, max: 1.0 },
          },
        },
      }
    );

    buildConfusionMatrix(data.confusion_matrix);
  } catch (err) {
    infoAlert.classList.remove('hidden');
    vizGrid.classList.add('hidden');
  }
}

function buildConfusionMatrix(matrix) {
  const container = document.getElementById('confusion-matrix-container');
  const maxValue = Math.max(...matrix.map((row) => Math.max(...row)), 1);
  const headerClass = 'text-xs font-semibold text-gray-400 bg-gray-900 p-1.5';

  let html = '<div class="overflow-x-auto rounded-lg border border-gray-800/50">';
  html += `<table class="border-collapse w-full"><thead><tr><th class="${headerClass}"></th>`;
  for (let p = 0; p < 10; p++) {
    html += `<th class="${headerClass} text-center">${p}</th>`;
  }
  html += '</tr></thead><tbody>';

  for (let t = 0; t < 10; t++) {
    html += `<tr><th class="${headerClass}">${t}</th>`;
    for (let p = 0; p < 10; p++) {
      const value = matrix[t][p];
      const opacity = value / maxValue;
      const isDiagonal = t === p;
      const color = isDiagonal
        ? `rgba(46, 160, 67, ${opacity})`
        : `rgba(79, 143, 247, ${opacity})`;
      const textClass = opacity > 0.5 ? 'text-white' : 'text-gray-500';
      html += `<td class="text-[10px] font-mono text-center p-1.5 ${textClass}" style="background-color:${color};">${value}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table></div>';

  container.innerHTML = html;
}

// ---------------------------------------------------------------------
// Retrain history panel + Dashboard "Recent Activity"
// ---------------------------------------------------------------------

function renderRecentActivity(entries) {
  const container = document.getElementById('recent-activity-list');
  if (!container) return;

  if (!entries || entries.length === 0) {
    container.innerHTML = '<p class="text-sm text-gray-600 italic">No activity yet.</p>';
    return;
  }

  container.innerHTML = '';
  entries.forEach((entry) => {
    const row = document.createElement('div');
    row.className = 'flex items-center justify-between text-sm bg-gray-900/50 rounded-lg px-3 py-2 border border-gray-800/50';

    const left = document.createElement('span');
    left.className = 'text-gray-400 text-xs';
    left.textContent = new Date(entry.timestamp).toLocaleString();

    const badge = document.createElement('span');
    badge.className = entry.promoted
      ? 'text-xs px-2 py-0.5 rounded-full font-medium text-green-400 bg-green-400/10'
      : 'text-xs px-2 py-0.5 rounded-full font-medium text-amber-400 bg-amber-400/10';
    badge.textContent = entry.promoted ? 'Promoted' : 'Rejected';

    row.appendChild(left);
    row.appendChild(badge);
    container.appendChild(row);
  });
}

async function fetchRetrainHistory() {
  const tbody = document.getElementById('retrain-history-body');
  tbody.innerHTML = '';

  try {
    const res = await fetch('/retrain-history');
    if (!res.ok) throw new Error('retrain history fetch failed');
    const history = await res.json();
    latestRetrainHistory = history || [];

    if (!history || history.length === 0) {
      tbody.innerHTML = `
        <tr><td colspan="7" class="text-center italic text-gray-600 py-8">No retraining attempts yet.</td></tr>
      `;
      renderRecentActivity([]);
      return;
    }

    const sorted = [...history].sort(
      (a, b) => new Date(b.timestamp) - new Date(a.timestamp)
    );

    renderRecentActivity(sorted.slice(0, 3));

    sorted.forEach((entry) => {
      const tr = document.createElement('tr');
      tr.className = 'text-sm even:bg-gray-800/30';

      const tsTd = document.createElement('td');
      tsTd.className = 'py-2 pr-4 text-gray-300 whitespace-nowrap';
      tsTd.textContent = new Date(entry.timestamp).toLocaleString();
      tr.appendChild(tsTd);

      const statusTd = document.createElement('td');
      statusTd.className = 'py-2 pr-4';
      const badge = document.createElement('span');
      badge.className = entry.promoted
        ? 'text-xs px-2 py-0.5 rounded-full font-medium text-green-400 bg-green-400/10'
        : 'text-xs px-2 py-0.5 rounded-full font-medium text-amber-400 bg-amber-400/10';
      badge.textContent = entry.promoted ? 'Promoted' : 'Rejected';
      statusTd.appendChild(badge);
      tr.appendChild(statusTd);

      const baselineTd = document.createElement('td');
      baselineTd.className = 'py-2 pr-4 font-mono text-gray-300';
      baselineTd.textContent = formatAccuracy(entry.baseline_accuracy);
      tr.appendChild(baselineTd);

      const newAccTd = document.createElement('td');
      newAccTd.className = 'py-2 pr-4 font-mono text-gray-300';
      newAccTd.textContent = formatAccuracy(entry.new_accuracy);
      tr.appendChild(newAccTd);

      const changeTd = document.createElement('td');
      const changePct = entry.accuracy_change * 100;
      changeTd.textContent = `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)} pp`;
      changeTd.className = `py-2 pr-4 font-mono ${changePct >= 0 ? 'text-green-400' : 'text-red-400'}`;
      tr.appendChild(changeTd);

      const samplesTd = document.createElement('td');
      samplesTd.className = 'py-2 pr-4 font-mono text-gray-300';
      samplesTd.textContent = entry.samples_used;
      tr.appendChild(samplesTd);

      const reasonTd = document.createElement('td');
      reasonTd.className = 'py-2 text-xs text-gray-500 leading-relaxed';
      reasonTd.textContent = entry.reason || '';
      tr.appendChild(reasonTd);

      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `
      <tr><td colspan="7" class="text-center italic text-gray-600 py-8">Could not load retraining history.</td></tr>
    `;
    renderRecentActivity([]);
  }
}

// Build the 10 probability rows once, at load time.
buildProbabilityBars();
