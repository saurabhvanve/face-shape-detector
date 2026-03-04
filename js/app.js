/**
 * app.js – Main application controller for the Face Shape Detector.
 *
 * Orchestrates UI, model loading, image upload, webcam capture,
 * face detection, shape classification, and result display.
 */

import { loadModels, detectFace, drawLandmarks, areModelsLoaded } from './faceDetection.js';
import { analyzeFaceShape } from './faceShapeLogic.js';
import {
    compressImage,
    downloadCanvasAsImage,
    saveToLocalStorage,
    loadFromLocalStorage,
    showElement,
    hideElement
} from './utils.js';

/* ────────────────────────── Constants ──────────────────────────────────── */

const STORAGE_KEY = 'faceShapeDetector_lastResult';
const STORAGE_THEME_KEY = 'faceShapeDetector_theme';

/* ────────────────────────── DOM References ─────────────────────────────── */

const $ = (sel) => document.querySelector(sel);

const dom = {
    // Loading
    loadingOverlay: $('#loading-overlay'),
    loadingText: $('#loading-text'),

    // Input area
    uploadZone: $('#upload-zone'),
    fileInput: $('#file-input'),
    cameraBtn: $('#camera-btn'),
    uploadBtn: $('#upload-btn'),

    // Display area
    displaySection: $('#display-section'),
    sourceImage: $('#source-image'),
    sourceVideo: $('#source-video'),
    overlayCanvas: $('#overlay-canvas'),

    // Controls
    captureBtn: $('#capture-btn'),
    stopCameraBtn: $('#stop-camera-btn'),
    newAnalysisBtn: $('#new-analysis-btn'),
    downloadBtn: $('#download-btn'),

    // Results
    resultCard: $('#result-card'),
    resultShape: $('#result-shape'),
    resultConfidence: $('#result-confidence'),
    confidenceBar: $('#confidence-bar'),
    resultDescription: $('#result-description'),
    ratiosGrid: $('#ratios-grid'),

    // Status
    statusMessage: $('#status-message'),

    // Theme
    themeToggle: $('#theme-toggle'),

    // Error
    errorMessage: $('#error-message'),
    errorText: $('#error-text'),
    errorDismiss: $('#error-dismiss')
};

/* ────────────────────────── State ──────────────────────────────────────── */

let cameraStream = null;
let animFrameId = null;
let isProcessing = false;

/* ────────────────────────── Initialization ─────────────────────────────── */

async function init() {
    setupTheme();
    setupEventListeners();
    restoreLastResult();

    try {
        showElement(dom.loadingOverlay);
        await loadModels((msg) => {
            if (dom.loadingText) dom.loadingText.textContent = msg;
        });
        hideElement(dom.loadingOverlay);
        setStatus('Ready — upload an image or start the camera.');
    } catch (err) {
        hideElement(dom.loadingOverlay);
        showError(`Failed to load AI models: ${err.message}. Please refresh.`);
    }
}

/* ────────────────────────── Event Listeners ────────────────────────────── */

function setupEventListeners() {
    // Upload button
    dom.uploadBtn?.addEventListener('click', () => dom.fileInput?.click());

    // File input change
    dom.fileInput?.addEventListener('change', handleFileSelect);

    // Drag and drop
    dom.uploadZone?.addEventListener('dragover', (e) => {
        e.preventDefault();
        dom.uploadZone.classList.add('drag-over');
    });
    dom.uploadZone?.addEventListener('dragleave', () => {
        dom.uploadZone.classList.remove('drag-over');
    });
    dom.uploadZone?.addEventListener('drop', handleDrop);

    // Camera
    dom.cameraBtn?.addEventListener('click', startCamera);
    dom.captureBtn?.addEventListener('click', captureFromCamera);
    dom.stopCameraBtn?.addEventListener('click', stopCamera);

    // Controls
    dom.newAnalysisBtn?.addEventListener('click', resetToUpload);
    dom.downloadBtn?.addEventListener('click', handleDownload);

    // Theme
    dom.themeToggle?.addEventListener('click', toggleTheme);

    // Error dismiss
    dom.errorDismiss?.addEventListener('click', () => hideElement(dom.errorMessage));
}

/* ────────────────────────── File Upload ────────────────────────────────── */

function handleFileSelect(e) {
    const file = e.target.files?.[0];
    if (file) processImageFile(file);
}

function handleDrop(e) {
    e.preventDefault();
    dom.uploadZone.classList.remove('drag-over');

    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith('image/')) {
        processImageFile(file);
    } else {
        showError('Please drop a valid image file.');
    }
}

async function processImageFile(file) {
    if (isProcessing) return;
    if (!areModelsLoaded()) {
        showError('Models are still loading. Please wait.');
        return;
    }

    isProcessing = true;
    setStatus('Compressing image…');
    hideElement(dom.errorMessage);

    try {
        const img = await compressImage(file);
        showImageForAnalysis(img);
        setStatus('Detecting face…');

        // Small delay to let the image render
        await new Promise((r) => setTimeout(r, 100));
        await analyzeImage(dom.sourceImage);
    } catch (err) {
        showError(err.message);
        setStatus('Error occurred.');
    } finally {
        isProcessing = false;
        // Reset file input so the same file can be re-selected
        if (dom.fileInput) dom.fileInput.value = '';
    }
}

function showImageForAnalysis(img) {
    hideElement(dom.uploadZone);
    showElement(dom.displaySection);
    hideElement(dom.sourceVideo);
    showElement(dom.sourceImage);
    hideElement(dom.captureBtn);
    hideElement(dom.stopCameraBtn);

    dom.sourceImage.src = img.src;
    dom.sourceImage.onload = () => {
        // Match canvas to displayed image size
        dom.overlayCanvas.width = dom.sourceImage.naturalWidth;
        dom.overlayCanvas.height = dom.sourceImage.naturalHeight;
    };
}

/* ────────────────────────── Camera ─────────────────────────────────────── */

async function startCamera() {
    if (!areModelsLoaded()) {
        showError('Models are still loading. Please wait.');
        return;
    }

    hideElement(dom.errorMessage);
    setStatus('Requesting camera access…');

    try {
        // Check for getUserMedia support
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error('Camera is not supported in this browser.');
        }

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
        });

        hideElement(dom.uploadZone);
        showElement(dom.displaySection);
        hideElement(dom.sourceImage);
        showElement(dom.sourceVideo);
        showElement(dom.captureBtn);
        showElement(dom.stopCameraBtn);
        hideElement(dom.resultCard);

        dom.sourceVideo.srcObject = cameraStream;
        await dom.sourceVideo.play();

        dom.overlayCanvas.width = dom.sourceVideo.videoWidth;
        dom.overlayCanvas.height = dom.sourceVideo.videoHeight;

        setStatus('Camera active — click Capture to analyze.');
        startLiveDetection();
    } catch (err) {
        if (err.name === 'NotAllowedError') {
            showError('Camera permission denied. Please allow camera access and try again.');
        } else if (err.name === 'NotFoundError') {
            showError('No camera found on this device.');
        } else {
            showError(`Camera error: ${err.message}`);
        }
        setStatus('Camera failed.');
    }
}

function startLiveDetection() {
    let detecting = false;

    const loop = async () => {
        if (!cameraStream) return;

        if (!detecting && dom.sourceVideo.readyState >= 2) {
            detecting = true;
            try {
                const result = await detectFace(dom.sourceVideo);
                if (result) {
                    drawLandmarks(dom.overlayCanvas, result.landmarks, result.box);
                } else {
                    // Clear canvas if no face
                    const ctx = dom.overlayCanvas.getContext('2d');
                    ctx.clearRect(0, 0, dom.overlayCanvas.width, dom.overlayCanvas.height);
                }
            } catch {
                // Silently ignore frame errors
            }
            detecting = false;
        }

        animFrameId = requestAnimationFrame(loop);
    };

    animFrameId = requestAnimationFrame(loop);
}

async function captureFromCamera() {
    if (isProcessing || !cameraStream) return;

    isProcessing = true;
    setStatus('Analyzing captured frame…');

    try {
        // Capture current frame to a temporary canvas
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = dom.sourceVideo.videoWidth;
        tempCanvas.height = dom.sourceVideo.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(dom.sourceVideo, 0, 0);

        // Stop the live feed
        stopCameraStream();

        // Show the captured frame as an image
        const img = new Image();
        img.src = tempCanvas.toDataURL('image/jpeg', 0.9);
        await new Promise((resolve) => { img.onload = resolve; });

        showImageForAnalysis(img);
        await new Promise((r) => setTimeout(r, 100));
        await analyzeImage(dom.sourceImage);
    } catch (err) {
        showError(err.message);
    } finally {
        isProcessing = false;
    }
}

function stopCamera() {
    stopCameraStream();
    resetToUpload();
}

function stopCameraStream() {
    if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
    }
    if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = null;
    }
    if (dom.sourceVideo) {
        dom.sourceVideo.srcObject = null;
    }
}

/* ────────────────────────── Analysis ───────────────────────────────────── */

async function analyzeImage(imgElement) {
    setStatus('Running AI face detection…');

    const result = await detectFace(imgElement);

    if (!result) {
        showError('No face detected. Please try a clearer photo with good lighting.');
        setStatus('No face found.');
        hideElement(dom.resultCard);
        return;
    }

    if (result.allDetections.length > 1) {
        setStatus(`${result.allDetections.length} faces found — analyzing the largest.`);
    }

    // Draw landmarks
    drawLandmarks(dom.overlayCanvas, result.landmarks, result.box);

    // Classify shape
    const analysis = analyzeFaceShape(result.landmarks);

    // Display result
    displayResult(analysis);
    setStatus('Analysis complete!');

    // Save to localStorage
    saveToLocalStorage(STORAGE_KEY, {
        ...analysis,
        timestamp: Date.now()
    });
}

/* ────────────────────────── Result Display ─────────────────────────────── */

function displayResult(analysis) {
    showElement(dom.resultCard);
    showElement(dom.downloadBtn);
    showElement(dom.newAnalysisBtn);

    // Shape icon map
    const shapeIcons = {
        Oval: '🥚', Round: '🔵', Square: '🟦',
        Heart: '💜', Diamond: '💎', Oblong: '📐'
    };

    dom.resultShape.textContent = `${shapeIcons[analysis.shape] || '✨'} ${analysis.shape}`;
    dom.resultConfidence.textContent = `${analysis.confidence}%`;
    dom.confidenceBar.style.width = `${analysis.confidence}%`;
    dom.resultDescription.textContent = analysis.description;

    // Populate ratios grid
    dom.ratiosGrid.innerHTML = '';
    const ratioLabels = {
        widthToLength: 'Width / Length',
        jawToCheekbone: 'Jaw / Cheekbones',
        foreheadToCheekbone: 'Forehead / Cheekbones',
        foreheadToJaw: 'Forehead / Jaw',
        chinToFaceLength: 'Chin / Face Length'
    };

    for (const [key, label] of Object.entries(ratioLabels)) {
        const item = document.createElement('div');
        item.className = 'ratio-item';
        item.innerHTML = `
            <span class="ratio-label">${label}</span>
            <span class="ratio-value">${analysis.ratios[key]}</span>
        `;
        dom.ratiosGrid.appendChild(item);
    }

    // Animate result card entrance
    dom.resultCard.classList.remove('animate-in');
    void dom.resultCard.offsetWidth; // force reflow
    dom.resultCard.classList.add('animate-in');
}

/* ────────────────────────── Restore Last Result ───────────────────────── */

function restoreLastResult() {
    const saved = loadFromLocalStorage(STORAGE_KEY);
    if (saved && saved.shape) {
        displayResult(saved);
        setStatus('Showing your last result. Upload a new image to re-analyze.');
    }
}

/* ────────────────────────── UI Helpers ─────────────────────────────────── */

function resetToUpload() {
    stopCameraStream();

    showElement(dom.uploadZone);
    hideElement(dom.displaySection);
    hideElement(dom.resultCard);
    hideElement(dom.captureBtn);
    hideElement(dom.stopCameraBtn);
    hideElement(dom.downloadBtn);
    hideElement(dom.errorMessage);

    // Clear canvas
    const ctx = dom.overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, dom.overlayCanvas.width, dom.overlayCanvas.height);

    dom.sourceImage.src = '';
    setStatus('Ready — upload an image or start the camera.');
}

function setStatus(msg) {
    if (dom.statusMessage) dom.statusMessage.textContent = msg;
}

function showError(msg) {
    if (dom.errorMessage && dom.errorText) {
        dom.errorText.textContent = msg;
        showElement(dom.errorMessage);
    }
}

function handleDownload() {
    // Composite the image + overlay into a single canvas
    const exportCanvas = document.createElement('canvas');
    const source = dom.sourceImage.naturalWidth
        ? dom.sourceImage
        : dom.sourceVideo;

    exportCanvas.width = source.naturalWidth || source.videoWidth || dom.overlayCanvas.width;
    exportCanvas.height = source.naturalHeight || source.videoHeight || dom.overlayCanvas.height;

    const ctx = exportCanvas.getContext('2d');
    ctx.drawImage(source, 0, 0, exportCanvas.width, exportCanvas.height);
    ctx.drawImage(dom.overlayCanvas, 0, 0, exportCanvas.width, exportCanvas.height);

    downloadCanvasAsImage(exportCanvas);
}

/* ────────────────────────── Theme Toggle ───────────────────────────────── */

function setupTheme() {
    const saved = loadFromLocalStorage(STORAGE_THEME_KEY);
    const theme = saved || 'dark';
    document.documentElement.setAttribute('data-theme', theme);
    updateThemeIcon(theme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    saveToLocalStorage(STORAGE_THEME_KEY, next);
    updateThemeIcon(next);
}

function updateThemeIcon(theme) {
    if (dom.themeToggle) {
        dom.themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
        dom.themeToggle.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
    }
}

/* ────────────────────────── Boot ───────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', init);
