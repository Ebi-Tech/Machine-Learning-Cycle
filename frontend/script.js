// Handwritten Digit Recognition Pipeline - dashboard logic.
// Plain JS, no framework. Talks to the Flask API on the same origin.

let chartInstances = {};
let sessionUploadCount = 0;

const DIGIT_COLORS = [
  '#4F8FF7', '#F76E6E', '#4CD37B', '#F7C948', '#B27FF0',
  '#F7924F', '#4FD1F7', '#F74FA3', '#8AC24A', '#7B8CF7',
];

document.addEventListener('DOMContentLoaded', () => {
  fetchHealth();
  setInterval(fetchHealth, 10000);

  fetchVisualizations();
  fetchRetrainHistory();

  document.getElementById('file-input').addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) handlePredict(e.target.files[0]);
  });

  const dropZone = document.getElementById('drop-zone');
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-primary');
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('border-primary');
  });
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-primary');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handlePredict(e.dataTransfer.files[0]);
    }
  });

  document.getElementById('clear-btn').addEventListener('click', clearPrediction);

  document.getElementById('upload-images-btn').addEventListener('click', handleUpload);
  document.getElementById('upload-zip-btn').addEventListener('click', handleZipUpload);
  document.getElementById('retrain-btn').addEventListener('click', triggerRetrain);
});

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
// Health / stats bar
// ---------------------------------------------------------------------

async function fetchHealth() {
  const statusBadge = document.getElementById('status-badge');
  const statusDot = document.getElementById('status-dot');
  const liveBadge = document.getElementById('live-badge');
  const uptimeEl = document.getElementById('stat-uptime');
  const lastTrainedEl = document.getElementById('stat-last-trained');
  const paramsEl = document.getElementById('stat-params');

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
    liveBadge.classList.toggle('hidden', !online);

    uptimeEl.textContent = formatUptime(data.uptime_seconds);
    lastTrainedEl.textContent = data.model_last_trained
      ? new Date(data.model_last_trained).toLocaleString()
      : 'Unknown';
    paramsEl.textContent = data.model_parameters != null
      ? data.model_parameters.toLocaleString()
      : 'Unknown';
  } catch (err) {
    statusBadge.textContent = 'Offline';
    statusBadge.className = 'inline-block text-sm font-semibold px-2.5 py-1 rounded-full text-red-400 bg-red-400/10 border border-red-400/20';
    statusDot.classList.add('hidden');
    liveBadge.classList.add('hidden');
    uptimeEl.textContent = 'Unknown';
    lastTrainedEl.textContent = 'Unknown';
    paramsEl.textContent = 'Unknown';
  }
}

// ---------------------------------------------------------------------
// Prediction panel
// ---------------------------------------------------------------------

async function handlePredict(file) {
  const previewImage = document.getElementById('preview-image');
  const objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;

  document.getElementById('predict-result').classList.remove('hidden');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      document.getElementById('predicted-digit').textContent = '?';
      return;
    }

    document.getElementById('predicted-digit').textContent = data.predicted_digit;

    const pct = Math.round(data.confidence * 100);
    document.getElementById('confidence-value').textContent = `${pct}% confidence`;

    document.getElementById('processing-time-badge').textContent =
      `${data.processing_time_ms.toFixed(1)} ms`;

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
  } catch (err) {
    document.getElementById('predicted-digit').textContent = '?';
  }
}

function clearPrediction() {
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
// Retrain history panel
// ---------------------------------------------------------------------

async function fetchRetrainHistory() {
  const tbody = document.getElementById('retrain-history-body');
  tbody.innerHTML = '';

  try {
    const res = await fetch('/retrain-history');
    if (!res.ok) throw new Error('retrain history fetch failed');
    const history = await res.json();

    if (!history || history.length === 0) {
      tbody.innerHTML = `
        <tr><td colspan="7" class="text-center italic text-gray-600 py-8">No retraining attempts yet.</td></tr>
      `;
      return;
    }

    const sorted = [...history].sort(
      (a, b) => new Date(b.timestamp) - new Date(a.timestamp)
    );

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
  }
}

// Build the 10 probability rows once, at load time.
buildProbabilityBars();
