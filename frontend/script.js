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

function showAlert(containerId, type, message) {
  const container = document.getElementById(containerId);
  container.innerHTML = `<div class="alert alert-${type} mt-2"><span>${message}</span></div>`;
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
    statusBadge.className = online ? 'badge badge-success' : 'badge badge-error';
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
    statusBadge.className = 'badge badge-error';
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
    const radial = document.getElementById('confidence-radial');
    radial.style.setProperty('--value', pct);
    radial.textContent = `${pct}%`;

    document.getElementById('processing-time-badge').textContent =
      `${data.processing_time_ms.toFixed(1)} ms`;

    data.probabilities.forEach((p, digit) => {
      const barPct = Math.round(p * 100);
      const bar = document.getElementById(`prob-bar-${digit}`);
      const text = document.getElementById(`prob-pct-${digit}`);
      bar.value = barPct;
      text.textContent = `${barPct}%`;
      bar.className = digit === data.predicted_digit
        ? 'progress progress-primary w-full'
        : 'progress w-full';
    });
  } catch (err) {
    document.getElementById('predicted-digit').textContent = '?';
  }
}

function clearPrediction() {
  document.getElementById('predict-result').classList.add('hidden');
  document.getElementById('preview-image').src = '';
  document.getElementById('predicted-digit').textContent = '-';
  const radial = document.getElementById('confidence-radial');
  radial.style.setProperty('--value', 0);
  radial.textContent = '0%';
  document.getElementById('processing-time-badge').textContent = '-- ms';
  for (let digit = 0; digit < 10; digit++) {
    const bar = document.getElementById(`prob-bar-${digit}`);
    const text = document.getElementById(`prob-pct-${digit}`);
    bar.value = 0;
    text.textContent = '0%';
    bar.className = 'progress w-full';
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
      <span class="w-4 text-sm text-gray-400">${digit}</span>
      <progress id="prob-bar-${digit}" class="progress w-full" value="0" max="100"></progress>
      <span id="prob-pct-${digit}" class="w-12 text-xs text-right text-gray-400">0%</span>
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
        <div class="alert alert-error">
          <span>${data.error || 'Retraining failed.'}</span>
        </div>
      `;
      return;
    }

    const promoted = data.promoted;
    const alertClass = promoted ? 'alert-success' : 'alert-warning';
    const title = promoted ? 'Model Updated' : 'Retraining Rejected';
    const icon = promoted
      ? `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 shrink-0 stroke-current" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>`
      : `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 shrink-0 stroke-current" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>`;

    resultContainer.innerHTML = `
      <div class="alert ${alertClass} items-start">
        ${icon}
        <div>
          <h3 class="font-bold">${title}</h3>
          <div class="text-sm space-y-0.5 mt-1">
          <div>Baseline accuracy: ${formatAccuracy(data.baseline_accuracy)}</div>
          <div>New accuracy: ${formatAccuracy(data.new_accuracy)}</div>
          <div>Change: ${(data.accuracy_change * 100).toFixed(2)} pp</div>
          <div>Samples used: ${data.samples_used}</div>
          <div class="italic mt-1">${data.reason}</div>
          </div>
        </div>
      </div>
    `;
  } catch (err) {
    resultContainer.innerHTML = `
      <div class="alert alert-error">
        <span>Retraining failed: could not reach the server.</span>
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

  let html = '<table class="text-xs border-collapse"><thead><tr><th class="p-1"></th>';
  for (let p = 0; p < 10; p++) {
    html += `<th class="p-1 text-center text-gray-400">${p}</th>`;
  }
  html += '</tr></thead><tbody>';

  for (let t = 0; t < 10; t++) {
    html += `<tr><th class="p-1 text-gray-400">${t}</th>`;
    for (let p = 0; p < 10; p++) {
      const value = matrix[t][p];
      const opacity = value / maxValue;
      const isDiagonal = t === p;
      const color = isDiagonal
        ? `rgba(46, 160, 67, ${opacity})`
        : `rgba(79, 143, 247, ${opacity})`;
      const textClass = opacity > 0.5 ? 'text-white' : 'text-gray-400';
      html += `<td class="p-1 text-center text-xs ${textClass}" style="background-color:${color};">${value}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';

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
        <tr><td colspan="7" class="text-center italic text-gray-400">No retraining attempts yet.</td></tr>
      `;
      return;
    }

    const sorted = [...history].sort(
      (a, b) => new Date(b.timestamp) - new Date(a.timestamp)
    );

    sorted.forEach((entry) => {
      const tr = document.createElement('tr');

      const tsTd = document.createElement('td');
      tsTd.textContent = new Date(entry.timestamp).toLocaleString();
      tr.appendChild(tsTd);

      const statusTd = document.createElement('td');
      const badge = document.createElement('span');
      badge.className = entry.promoted ? 'badge badge-success' : 'badge badge-warning';
      badge.textContent = entry.promoted ? 'Promoted' : 'Rejected';
      statusTd.appendChild(badge);
      tr.appendChild(statusTd);

      const baselineTd = document.createElement('td');
      baselineTd.textContent = formatAccuracy(entry.baseline_accuracy);
      tr.appendChild(baselineTd);

      const newAccTd = document.createElement('td');
      newAccTd.textContent = formatAccuracy(entry.new_accuracy);
      tr.appendChild(newAccTd);

      const changeTd = document.createElement('td');
      const changePct = entry.accuracy_change * 100;
      changeTd.textContent = `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)} pp`;
      changeTd.className = changePct >= 0 ? 'text-green-400' : 'text-red-400';
      tr.appendChild(changeTd);

      const samplesTd = document.createElement('td');
      samplesTd.textContent = entry.samples_used;
      tr.appendChild(samplesTd);

      const reasonTd = document.createElement('td');
      const reason = entry.reason || '';
      const truncated = reason.length > 60 ? `${reason.slice(0, 60)}...` : reason;
      const tooltip = document.createElement('div');
      tooltip.className = 'tooltip';
      tooltip.setAttribute('data-tip', reason);
      const span = document.createElement('span');
      span.textContent = truncated;
      tooltip.appendChild(span);
      reasonTd.appendChild(tooltip);
      tr.appendChild(reasonTd);

      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `
      <tr><td colspan="7" class="text-center italic text-gray-400">Could not load retraining history.</td></tr>
    `;
  }
}

// Build the 10 probability rows once, at load time.
buildProbabilityBars();
