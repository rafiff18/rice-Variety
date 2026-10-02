document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const btnBrowse = document.getElementById('btnBrowse');
  const dropzoneIdle = document.getElementById('dropzoneIdle');
  const dropzonePreview = document.getElementById('dropzonePreview');
  const previewImage = document.getElementById('previewImage');
  const btnChangeImage = document.getElementById('btnChangeImage');
  const btnPredict = document.getElementById('btnPredict');
  const scanLaser = document.getElementById('scanLaser');

  // Camera elements
  const btnCamera = document.getElementById('btnCamera');
  const webcamContainer = document.getElementById('webcamContainer');
  const webcamVideo = document.getElementById('webcamVideo');
  const btnSnap = document.getElementById('btnSnap');
  const btnCloseCamera = document.getElementById('btnCloseCamera');
  let mediaStream = null;

  // Samples
  const samplesGrid = document.getElementById('samplesGrid');

  // Results
  const resultPlaceholder = document.getElementById('resultPlaceholder');
  const resultContent = document.getElementById('resultContent');
  const heroVarietyName = document.getElementById('heroVarietyName');
  const heroTagline = document.getElementById('heroTagline');
  const confidenceVal = document.getElementById('confidenceVal');
  const heroResultCard = document.getElementById('heroResultCard');
  const probList = document.getElementById('probList');
  const latencyTag = document.getElementById('latencyTag');
  const latencyVal = document.getElementById('latencyVal');

  // Specs
  const specOrigin = document.getElementById('specOrigin');
  const specGI = document.getElementById('specGI');
  const specShape = document.getElementById('specShape');
  const specTexture = document.getElementById('specTexture');
  const specUses = document.getElementById('specUses');
  const specRatio = document.getElementById('specRatio');

  let currentBlob = null;

  // --- 1. Load Samples for 1-Click Testing ---
  async function loadSamples() {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      if (data.samples && data.samples.length > 0) {
        samplesGrid.innerHTML = '';
        data.samples.forEach(sample => {
          const chip = document.createElement('div');
          chip.className = 'sample-chip';
          chip.title = `Test with ${sample.class_name}`;
          chip.innerHTML = `
            <img src="${sample.image_b64}" alt="${sample.class_name}">
            <span class="sample-chip-name">${sample.class_name}</span>
          `;
          chip.addEventListener('click', async () => {
            // Convert b64 to blob
            const res = await fetch(sample.image_b64);
            const blob = await res.blob();
            selectImageBlob(blob, sample.image_b64);
            // Run automatic prediction
            runPrediction();
          });
          samplesGrid.appendChild(chip);
        });
      }
    } catch (err) {
      console.warn('Samples fetch warning:', err);
      samplesGrid.innerHTML = '<span style="font-size:0.75rem; color:#64748b;">Samples unavailable</span>';
    }
  }
  loadSamples();

  // --- 2. File Selection & Drag-and-Drop ---
  btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener('click', () => {
    if (!currentBlob) fileInput.click();
  });

  btnChangeImage.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('drag-active');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('drag-active');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (dt.files && dt.files[0]) {
      handleFile(dt.files[0]);
    }
  });

  function handleFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (JPG, PNG, WEBP).');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      selectImageBlob(file, e.target.result);
    };
    reader.readAsDataURL(file);
  }

  function selectImageBlob(blob, dataUrl) {
    currentBlob = blob;
    previewImage.src = dataUrl;
    dropzoneIdle.classList.add('hidden');
    dropzonePreview.classList.remove('hidden');
    btnPredict.disabled = false;
  }

  // --- 3. Camera Capture ---
  btnCamera.addEventListener('click', async () => {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      webcamVideo.srcObject = mediaStream;
      webcamContainer.classList.remove('hidden');
    } catch (err) {
      alert('Could not access camera: ' + err.message);
    }
  });

  btnCloseCamera.addEventListener('click', stopCamera);

  function stopCamera() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
    }
    webcamContainer.classList.add('hidden');
  }

  btnSnap.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = webcamVideo.videoWidth || 640;
    canvas.height = webcamVideo.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob((blob) => {
      selectImageBlob(blob, canvas.toDataURL('image/jpeg'));
      stopCamera();
      runPrediction();
    }, 'image/jpeg', 0.95);
  });

  // --- 4. Prediction Execution ---
  btnPredict.addEventListener('click', runPrediction);

  async function runPrediction() {
    if (!currentBlob) return;

    btnPredict.disabled = true;
    scanLaser.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', currentBlob, 'grain.jpg');

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Prediction failed');
      }

      const data = await res.json();
      renderResults(data);
    } catch (err) {
      alert('Error analyzing grain: ' + err.message);
    } finally {
      btnPredict.disabled = false;
      scanLaser.classList.add('hidden');
    }
  }

  function renderResults(data) {
    resultPlaceholder.classList.add('hidden');
    resultContent.classList.remove('hidden');

    const pred = data.prediction;
    const meta = data.metadata || {};

    // Hero Card
    heroVarietyName.textContent = pred.class_name;
    heroTagline.textContent = meta.tagline || 'Verified Commercial Rice Variety';
    confidenceVal.textContent = pred.confidence_str;

    if (meta.accent_color) {
      heroResultCard.style.borderColor = meta.accent_color;
      heroResultCard.style.boxShadow = `0 0 30px ${meta.accent_color}33`;
    }

    // Latency
    latencyVal.textContent = data.latency_ms;
    latencyTag.classList.remove('hidden');

    // Probabilities
    probList.innerHTML = '';
    data.probabilities.forEach(p => {
      const item = document.createElement('div');
      item.className = 'prob-item';
      item.innerHTML = `
        <div class="prob-item-header">
          <span class="prob-name">${p.class_name}</span>
          <span class="prob-val">${p.percentage_str}</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: 0%; background: ${p.accent_color || '#38bdf8'};"></div>
        </div>
      `;
      probList.appendChild(item);

      // Animate bar width
      setTimeout(() => {
        const fill = item.querySelector('.prob-bar-fill');
        fill.style.width = `${Math.max(p.probability, 1)}%`;
      }, 50);
    });

    // Specifications
    specOrigin.textContent = meta.origin || '-';
    specGI.textContent = meta.glycemic_index || '-';
    specShape.textContent = meta.shape || '-';
    specTexture.textContent = meta.texture || '-';
    specUses.textContent = meta.best_uses || '-';
    specRatio.textContent = meta.cooking_ratio || '-';
  }
});
