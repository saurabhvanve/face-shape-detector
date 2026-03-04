/**
 * faceDetection.js – Face detection & landmark extraction using @vladmandic/face-api.
 *
 * Loads SSD MobileNet v1 + 68-point face landmark models from CDN.
 * Provides detection for both static images and video frames.
 */

/* global faceapi */

const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1/model';

let modelsLoaded = false;

/* ────────────────────────── Model Loading ──────────────────────────────── */

/**
 * Lazy-load face detection and landmark models.
 * Safe to call multiple times — only loads once.
 * @param {Function} [onProgress] – Optional callback for progress updates.
 * @returns {Promise<void>}
 */
export async function loadModels(onProgress) {
    if (modelsLoaded) return;

    if (typeof faceapi === 'undefined') {
        throw new Error(
            'face-api.js is not loaded. Make sure the CDN script tag is present.'
        );
    }

    try {
        if (onProgress) onProgress('Loading face detection model…');
        await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);

        if (onProgress) onProgress('Loading face landmark model…');
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);

        modelsLoaded = true;
        if (onProgress) onProgress('Models ready!');
    } catch (err) {
        throw new Error(`Failed to load ML models: ${err.message}`);
    }
}

/**
 * Check if models have been loaded.
 * @returns {boolean}
 */
export function areModelsLoaded() {
    return modelsLoaded;
}

/* ────────────────────────── Face Detection ─────────────────────────────── */

/**
 * Detect faces with landmarks in an image or video element.
 *
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} input
 * @returns {Promise<{landmarks: Array<{x,y}>, box: {x,y,width,height}, allDetections: Array}|null>}
 *   Returns the largest face's landmarks and bounding box, or null if no face found.
 */
export async function detectFace(input) {
    if (!modelsLoaded) {
        throw new Error('Models not loaded. Call loadModels() first.');
    }

    const options = new faceapi.SsdMobilenetv1Options({
        minConfidence: 0.5
    });

    const detections = await faceapi
        .detectAllFaces(input, options)
        .withFaceLandmarks();

    if (!detections || detections.length === 0) {
        return null;
    }

    // Pick the largest face (by bounding box area)
    const sorted = [...detections].sort((a, b) => {
        const areaA = a.detection.box.width * a.detection.box.height;
        const areaB = b.detection.box.width * b.detection.box.height;
        return areaB - areaA;
    });

    const best = sorted[0];
    const points = best.landmarks.positions.map((pt) => ({
        x: pt.x,
        y: pt.y
    }));

    return {
        landmarks: points,
        box: {
            x: best.detection.box.x,
            y: best.detection.box.y,
            width: best.detection.box.width,
            height: best.detection.box.height
        },
        allDetections: detections
    };
}

/* ────────────────────────── Landmark Drawing ───────────────────────────── */

/**
 * Draw face landmarks and connections on a canvas.
 *
 * @param {HTMLCanvasElement} canvas – Overlay canvas (must match source dimensions).
 * @param {Array<{x: number, y: number}>} landmarks – 68 landmark points.
 * @param {{x,y,width,height}} box – Face bounding box.
 */
export function drawLandmarks(canvas, landmarks, box) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!landmarks || landmarks.length === 0) return;

    // Draw bounding box
    ctx.strokeStyle = 'rgba(0, 210, 255, 0.7)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    ctx.setLineDash([]);

    // Connection groups for the 68-point model
    const groups = [
        { range: [0, 16], color: 'rgba(0, 210, 255, 0.8)', label: 'Jaw' },
        { range: [17, 21], color: 'rgba(255, 107, 107, 0.8)', label: 'Right Brow' },
        { range: [22, 26], color: 'rgba(255, 107, 107, 0.8)', label: 'Left Brow' },
        { range: [27, 30], color: 'rgba(78, 205, 196, 0.8)', label: 'Nose Bridge' },
        { range: [31, 35], color: 'rgba(78, 205, 196, 0.8)', label: 'Nose Tip' },
        { range: [36, 41], color: 'rgba(255, 217, 61, 0.8)', label: 'Right Eye', closed: true },
        { range: [42, 47], color: 'rgba(255, 217, 61, 0.8)', label: 'Left Eye', closed: true },
        { range: [48, 59], color: 'rgba(255, 107, 182, 0.8)', label: 'Outer Lip', closed: true },
        { range: [60, 67], color: 'rgba(255, 107, 182, 0.8)', label: 'Inner Lip', closed: true }
    ];

    // Draw connections
    for (const group of groups) {
        ctx.strokeStyle = group.color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        for (let i = group.range[0]; i <= group.range[1]; i++) {
            const pt = landmarks[i];
            if (i === group.range[0]) {
                ctx.moveTo(pt.x, pt.y);
            } else {
                ctx.lineTo(pt.x, pt.y);
            }
        }

        if (group.closed) {
            ctx.closePath();
        }
        ctx.stroke();
    }

    // Draw landmark points
    for (let i = 0; i < landmarks.length; i++) {
        const pt = landmarks[i];
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(0, 210, 255, 0.6)';
        ctx.lineWidth = 0.8;
        ctx.stroke();
    }

    // Draw measurement lines (visual feedback of what we're measuring)
    drawMeasurementLine(ctx, landmarks[17], landmarks[26], 'rgba(255,200,0,0.5)'); // forehead
    drawMeasurementLine(ctx, landmarks[1], landmarks[15], 'rgba(0,255,150,0.5)');  // cheekbones
    drawMeasurementLine(ctx, landmarks[4], landmarks[12], 'rgba(255,100,100,0.5)'); // jaw
}

/**
 * Draw a dashed measurement line between two points.
 */
function drawMeasurementLine(ctx, a, b, color) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
}
