// State
let currentSelectedSampleId = "stylegan_portrait";
let currentUploadedFile = null;
let currentAnalysisResult = null;
let benchmarkData = null;
let currentViewMode = "orig";

// Initialize
document.addEventListener("DOMContentLoaded", async () => {
  setupDropzone();
  await loadSamples();
  await loadBenchmarks();
  // Auto-run analysis on the first preset to immediately showcase the project
  runAnalysis();
});

// Tab Navigation
function switchTab(tabId) {
  document.querySelectorAll(".nav-tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(tab => tab.classList.remove("active"));
  
  const targetBtn = document.getElementById(`tab-btn-${tabId}`);
  const targetTab = document.getElementById(`tab-${tabId}`);
  if (targetBtn && targetTab) {
    targetBtn.classList.add("active");
    targetTab.classList.add("active");
  }
}

// Load Preset Samples from API
async function loadSamples() {
  try {
    const res = await fetch("/api/samples");
    const samples = await res.json();
    const grid = document.getElementById("sample-grid");
    grid.innerHTML = "";

    samples.forEach(sample => {
      const isSelected = sample.id === currentSelectedSampleId;
      const card = document.createElement("div");
      card.className = `sample-card ${isSelected ? 'active' : ''}`;
      card.id = `sample-card-${sample.id}`;
      card.onclick = () => selectSample(sample);

      const tagClass = sample.category.includes("AI") ? "tag-fake" : "tag-real";

      card.innerHTML = `
        <div class="sample-thumb-wrapper">
          <img src="${sample.url}" alt="${sample.title}">
          <span class="sample-type-tag ${tagClass}">${sample.category}</span>
        </div>
        <div class="sample-info">
          <h4>${sample.title}</h4>
          <p>${sample.generator}</p>
        </div>
      `;
      grid.appendChild(card);
    });
  } catch (err) {
    console.error("Failed to load samples:", err);
  }
}

// Select Preset Sample
function selectSample(sample) {
  currentSelectedSampleId = sample.id;
  currentUploadedFile = null;

  document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
  const card = document.getElementById(`sample-card-${sample.id}`);
  if (card) card.classList.add("active");

  document.getElementById("current-image-name").innerText = `Selected: ${sample.title}`;
  document.getElementById("current-image-tag").innerText = sample.category;
  document.getElementById("active-preview-img").src = sample.url;

  // Run analysis immediately
  runAnalysis();
}

// Setup Drag & Drop File Handling
function setupDropzone() {
  const dropzone = document.getElementById("dropzone");
  
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      processSelectedFile(files[0]);
    }
  });
}

function handleFileSelect(event) {
  const files = event.target.files;
  if (files && files.length > 0) {
    processSelectedFile(files[0]);
  }
}

function processSelectedFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please select an image file (JPG, PNG, WEBP)");
    return;
  }
  currentUploadedFile = file;
  currentSelectedSampleId = null;

  // Uncheck preset cards
  document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));

  // Preview local image
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("active-preview-img").src = e.target.result;
    document.getElementById("current-image-name").innerText = `Custom Upload: ${file.name}`;
    document.getElementById("current-image-tag").innerText = "Custom Upload";
    runAnalysis();
  };
  reader.readAsDataURL(file);
}

// Run Analysis
async function runAnalysis() {
  const btn = document.getElementById("btn-analyze");
  const btnText = document.getElementById("btn-analyze-text");
  btn.disabled = true;
  btnText.innerHTML = `<span class="spinner"></span> Processing CNN, ViT & Hybrid Fusion...`;

  try {
    const formData = new FormData();
    if (currentUploadedFile) {
      formData.append("file", currentUploadedFile);
    } else if (currentSelectedSampleId) {
      formData.append("sample_id", currentSelectedSampleId);
    } else {
      formData.append("sample_id", "midjourney_astronaut");
    }

    const res = await fetch("/api/analyze", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      throw new Error(`Server returned error ${res.status}`);
    }

    currentAnalysisResult = await res.json();
    displayResults(currentAnalysisResult);

  } catch (err) {
    console.error("Analysis failed:", err);
    alert("Analysis failed: " + err.message);
  } finally {
    btn.disabled = false;
    btnText.innerHTML = `⚡ Run Hybrid CNN+ViT Forensic Analysis`;
  }
}

// Display Forensic Results
function displayResults(data) {
  const container = document.getElementById("results-container");
  container.style.display = "block";

  // 1. Verdict Banner
  const banner = document.getElementById("verdict-banner");
  const vIcon = document.getElementById("verdict-icon");
  const vTitle = document.getElementById("verdict-title");
  const vSub = document.getElementById("verdict-subtitle");
  const vConf = document.getElementById("verdict-confidence");
  const vDim = document.getElementById("verdict-dimensions");
  const vSource = document.getElementById("verdict-model-source");
  const vLat = document.getElementById("verdict-latency");
  const domainAlert = document.getElementById("domain-warning-banner");
  const domainText = document.getElementById("domain-warning-text");

  // Handle Domain Warning
  if (data.domain_warning) {
    domainAlert.style.display = "block";
    domainText.innerText = data.domain_warning;
  } else {
    domainAlert.style.display = "none";
  }

  // Model source badge & latency
  if (vSource) {
    if (data.model_source === "trained") {
      vSource.innerText = "✓ Trained Model (" + (data.model_version || "PyTorch GPU") + ")";
      vSource.style.background = "rgba(16, 185, 129, 0.15)";
      vSource.style.color = "#10b981";
      vSource.style.borderColor = "rgba(16, 185, 129, 0.4)";
    } else {
      vSource.innerText = "Heuristic Fallback";
      vSource.style.background = "rgba(245, 158, 11, 0.15)";
      vSource.style.color = "#f59e0b";
      vSource.style.borderColor = "rgba(245, 158, 11, 0.4)";
    }
  }

  if (vLat) {
    vLat.innerText = data.inference_ms ? `${data.inference_ms} ms` : "-- ms";
  }

  if (data.is_fake) {
    banner.className = "verdict-banner is-fake";
    vIcon.innerText = "⚠️";
    vTitle.innerText = "AI-Generated / Synthetic Image";
    vSub.innerText = `Detected Footprint: ${data.generator_type}`;
    vConf.innerText = `${data.confidence}%`;
    vConf.style.color = "var(--accent-rose)";
  } else {
    banner.className = "verdict-banner is-real";
    vIcon.innerText = "🛡️";
    vTitle.innerText = "Authentic Photograph";
    vSub.innerText = `Source Verification: Authentic Camera Sensor Capture`;
    vConf.innerText = `${data.confidence}%`;
    vConf.style.color = "var(--accent-emerald)";
  }
  vDim.innerText = data.dimensions || "1024 × 1024";

  // 2. Model Scores Grid
  // CNN
  const cnn = data.models.cnn;
  document.getElementById("cnn-score-val").innerText = `${cnn.probability_fake}%`;
  document.getElementById("cnn-pred-badge").innerText = cnn.prediction;
  document.getElementById("cnn-score-bar").style.width = `${cnn.probability_fake}%`;
  document.getElementById("cnn-score-bar").style.background = cnn.prediction === "AI-Generated" ? "var(--accent-rose)" : "var(--accent-emerald)";

  // ViT
  const vit = data.models.vit;
  document.getElementById("vit-score-val").innerText = `${vit.probability_fake}%`;
  document.getElementById("vit-pred-badge").innerText = vit.prediction;
  document.getElementById("vit-score-bar").style.width = `${vit.probability_fake}%`;
  document.getElementById("vit-score-bar").style.background = vit.prediction === "AI-Generated" ? "var(--accent-amber)" : "var(--accent-emerald)";

  // Hybrid / Guided
  const hybrid = data.models.hybrid;
  document.getElementById("hybrid-score-val").innerText = `${hybrid.probability_fake}%`;
  document.getElementById("hybrid-pred-badge").innerText = hybrid.prediction;
  document.getElementById("hybrid-score-bar").style.width = `${hybrid.probability_fake}%`;
  document.getElementById("hybrid-score-bar").style.background = hybrid.prediction === "AI-Generated" ? "var(--accent-primary)" : "var(--accent-emerald)";

  // 3. Visualizer
  currentViewMode = "orig";
  updateVisualizerView();

  // 4. Forensics Metrics
  const f = data.forensics;
  document.getElementById("val-noise-uni").innerText = `${f.noise_uniformity}%`;
  document.getElementById("bar-noise-uni").style.width = `${f.noise_uniformity}%`;

  document.getElementById("val-patch-div").innerText = `${f.patch_discrepancy}%`;
  document.getElementById("bar-patch-div").style.width = `${f.patch_discrepancy}%`;

  document.getElementById("val-spectral-energy").innerText = `${f.spectral_frequency_energy}%`;
  document.getElementById("bar-spectral-energy").style.width = `${Math.min(f.spectral_frequency_energy, 100)}%`;

  document.getElementById("val-grad-var").innerText = `${f.laplacian_gradient_variance}%`;
  document.getElementById("bar-grad-var").style.width = `${Math.min(f.laplacian_gradient_variance, 100)}%`;

  // 5. Reasoning Text
  document.getElementById("analysis-reasoning-text").innerText = data.reasoning;
}

// Switch Visualizer View Mode
function setViewMode(mode) {
  currentViewMode = mode;
  document.querySelectorAll(".vis-tab-btn").forEach(btn => btn.classList.remove("active"));
  const btn = document.getElementById(`btn-view-${mode}`);
  if (btn) btn.classList.add("active");
  updateVisualizerView();
}

function updateVisualizerView() {
  if (!currentAnalysisResult) return;
  const viewportImg = document.getElementById("viewport-img");
  const caption = document.getElementById("view-caption");
  const legend = document.getElementById("legend-scale-box");

  if (currentViewMode === "orig") {
    viewportImg.src = document.getElementById("active-preview-img").src;
    caption.innerText = "Viewing: Original photographic input image";
    legend.style.display = "none";
  } else if (currentViewMode === "ela") {
    viewportImg.src = currentAnalysisResult.heatmaps.ela;
    caption.innerText = "Viewing: CNN Error Level Analysis (ELA) — Highlights high-frequency compression & localized pixel-level noise artifacts";
    legend.style.display = "flex";
  } else if (currentViewMode === "vit") {
    viewportImg.src = currentAnalysisResult.heatmaps.vit_attention;
    caption.innerText = "Viewing: ViT Patch Self-Attention Heatmap — Highlights non-adjacent contextual discrepancies & global lighting anomalies";
    legend.style.display = "flex";
  } else if (currentViewMode === "gradcam") {
    viewportImg.src = currentAnalysisResult.heatmaps.gradcam || currentAnalysisResult.heatmaps.ela;
    caption.innerText = "Viewing: CNN Grad-CAM Spatial Attribution — Visualizes feature activations responsible for real/fake classification decision";
    legend.style.display = "flex";
  } else if (currentViewMode === "fft") {
    viewportImg.src = currentAnalysisResult.heatmaps.fft_spectrum || currentAnalysisResult.heatmaps.ela;
    caption.innerText = "Viewing: 2D FFT Radial Frequency Power Spectrum — Compares natural sensor noise decay vs artificial spectral spikes / smoothing";
    legend.style.display = "flex";
  }
}

// Load Benchmark Data for Tab 2
async function loadBenchmarks() {
  try {
    const res = await fetch("/api/benchmark");
    benchmarkData = await res.json();
    renderBenchmarkTables(benchmarkData);
  } catch (err) {
    console.error("Failed to load benchmarks:", err);
  }
}

function renderBenchmarkTables(data) {
  // 1. Comparison Table
  const tbodyComp = document.getElementById("tbody-comparison");
  tbodyComp.innerHTML = "";
  data.model_comparison.forEach(m => {
    const isHybrid = m.id === "hybrid";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-weight: 700; ${isHybrid ? 'color: var(--accent-cyan);' : ''}">
        ${isHybrid ? '⭐ ' : ''}${m.name}
      </td>
      <td style="font-family: var(--font-mono); font-size: 12px; color: var(--text-muted);">${m.backbone}</td>
      <td class="${isHybrid ? 'highlight-cell' : ''}">${m.accuracy}%</td>
      <td>${m.precision}%</td>
      <td>${m.recall}%</td>
      <td>${m.f1_score}%</td>
      <td>${m.auc_roc}</td>
      <td style="font-family: var(--font-mono);">${m.latency_ms} ms</td>
    `;
    tbodyComp.appendChild(tr);
  });

  // 2. Generator Breakdown
  const tbodyGen = document.getElementById("tbody-generator");
  tbodyGen.innerHTML = "";
  data.generator_breakdown.forEach(g => {
    const tr = document.createElement("tr");
    const guidedVal = g.guided_acc !== undefined ? g.guided_acc : g.hybrid_acc;
    tr.innerHTML = `
      <td style="font-weight: 500;">${g.generator}</td>
      <td style="font-family: var(--font-mono);">${g.cnn_acc}%</td>
      <td style="font-family: var(--font-mono);">${g.vit_acc}%</td>
      <td style="font-family: var(--font-mono); font-weight: 600; color: var(--accent-blue-light);">${guidedVal}%</td>
      <td style="color: var(--text-secondary); font-size: 12px;">${g.notes}</td>
    `;
    tbodyGen.appendChild(tr);
  });
}

// Switch Confusion Matrix
function selectConfusionMatrix(modelKey, btnElement) {
  if (!benchmarkData || !benchmarkData.confusion_matrices) return;

  document.querySelectorAll(".cm-btn").forEach(b => b.classList.remove("active"));
  if (btnElement) btnElement.classList.add("active");

  const cm = benchmarkData.confusion_matrices[modelKey];
  if (!cm) return;

  document.getElementById("cm-tn").innerText = cm.true_negative.toLocaleString();
  document.getElementById("cm-fp").innerText = cm.false_positive.toLocaleString();
  document.getElementById("cm-fn").innerText = cm.false_negative.toLocaleString();
  document.getElementById("cm-tp").innerText = cm.true_positive.toLocaleString();
  document.getElementById("cm-active-name").innerText = cm.model_name;
}
