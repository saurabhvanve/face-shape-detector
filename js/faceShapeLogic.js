/**
 * faceShapeLogic.js – Face shape classification from 68-point landmarks.
 *
 * Landmark map (0-indexed):
 *   0–16  : Jaw contour (right ear → chin → left ear)
 *   17–21 : Right eyebrow
 *   22–26 : Left eyebrow
 *   27–30 : Nose bridge
 *   31–35 : Lower nose
 *   36–41 : Right eye
 *   42–47 : Left eye
 *   48–59 : Outer lips
 *   60–67 : Inner lips
 */

import { distance, midpoint } from './utils.js';

/* ────────────────────────── Shape Descriptions ────────────────────────── */

const SHAPE_DESCRIPTIONS = {
    Oval: 'Your face is elegantly balanced — longer than it is wide with gently rounded contours. Considered the most versatile face shape for hairstyles and accessories.',
    Round: 'Your face is soft and symmetrical with full cheeks and a rounded jawline. Width and length are nearly equal, giving a youthful appearance.',
    Square: 'Your face has a strong, angular jawline with a broad forehead. Width and length are similar, conveying strength and confidence.',
    Heart: 'Your face features a wider forehead tapering to a narrower chin, creating a romantic heart or inverted-triangle silhouette.',
    Diamond: 'Your face has dramatic cheekbones that are wider than both your forehead and jawline, creating a striking angular look.',
    Oblong: 'Your face is noticeably longer than it is wide with a relatively even width from forehead to jaw. Sometimes called rectangular.'
};

/* ────────────────────────── Measurement Extraction ────────────────────── */

/**
 * Extract key facial measurements from 68 landmarks.
 * @param {Array<{x: number, y: number}>} lm – Array of 68 landmark points.
 * @returns {Object} – Named measurements in pixels.
 */
function extractMeasurements(lm) {
    // Forehead width: distance between outer eyebrow ends (17 → 26)
    const foreheadWidth = distance(lm[17], lm[26]);

    // Cheekbone width: widest part of face (1 → 15, roughly at cheekbone level)
    const cheekboneWidth = distance(lm[1], lm[15]);

    // Jaw width: distance across the jaw (4 → 12)
    const jawWidth = distance(lm[4], lm[12]);

    // Face length: from chin (8) to midpoint of the top of nose bridge (27)
    // We approximate the top of the head by projecting above pt 27
    const noseBridgeTop = lm[27];
    const chin = lm[8];
    const foreheadMid = midpoint(lm[19], lm[24]); // mid-brow as proxy for forehead center

    // Extend forehead midpoint upward by the brow-to-nose-bridge distance
    const browToNose = distance(foreheadMid, noseBridgeTop);
    const foreheadTop = {
        x: foreheadMid.x,
        y: foreheadMid.y - browToNose * 0.9
    };
    const faceLength = distance(foreheadTop, chin);

    // Jaw angle (how angular the jaw is) — angle at points 6, 8, 10
    const jawAngle = calculateAngle(lm[6], lm[8], lm[10]);

    // Chin sharpness — distance from chin (8) to midpoint of jaw corners (4, 12)
    const jawMid = midpoint(lm[4], lm[12]);
    const chinLength = distance(lm[8], jawMid);

    return {
        foreheadWidth,
        cheekboneWidth,
        jawWidth,
        faceLength,
        jawAngle,
        chinLength
    };
}

/**
 * Calculate angle (in degrees) at point B formed by points A–B–C.
 */
function calculateAngle(a, b, c) {
    const ab = { x: a.x - b.x, y: a.y - b.y };
    const cb = { x: c.x - b.x, y: c.y - b.y };
    const dot = ab.x * cb.x + ab.y * cb.y;
    const cross = ab.x * cb.y - ab.y * cb.x;
    const angle = Math.atan2(Math.abs(cross), dot);
    return (angle * 180) / Math.PI;
}

/* ────────────────────────── Ratio Computation ─────────────────────────── */

/**
 * Compute normalized ratios from raw measurements.
 * @param {Object} m – Measurements from extractMeasurements.
 * @returns {Object} – Named ratios.
 */
function computeRatios(m) {
    return {
        widthToLength: m.cheekboneWidth / m.faceLength,
        jawToCheekbone: m.jawWidth / m.cheekboneWidth,
        foreheadToCheekbone: m.foreheadWidth / m.cheekboneWidth,
        foreheadToJaw: m.foreheadWidth / m.jawWidth,
        chinToFaceLength: m.chinLength / m.faceLength
    };
}

/* ────────────────────────── Classification ─────────────────────────────── */

/**
 * Score each face shape based on ratios and measurements.
 * Returns the best match with a confidence score.
 *
 * @param {Object} ratios – From computeRatios.
 * @param {Object} measurements – From extractMeasurements.
 * @returns {{ shape: string, confidence: number, ratios: Object, measurements: Object, description: string }}
 */
function classifyShape(ratios, measurements) {
    const scores = {
        Oval: 0,
        Round: 0,
        Square: 0,
        Heart: 0,
        Diamond: 0,
        Oblong: 0
    };

    const { widthToLength, jawToCheekbone, foreheadToCheekbone, foreheadToJaw, chinToFaceLength } = ratios;
    const { jawAngle } = measurements;

    /* ── Oval ──────────────────────────────────────────────── */
    // Face is longer than wide, cheekbones widest, soft jaw
    if (widthToLength >= 0.65 && widthToLength <= 0.85) scores.Oval += 3;
    else if (widthToLength > 0.85 && widthToLength <= 0.90) scores.Oval += 1;

    if (foreheadToCheekbone >= 0.85 && foreheadToCheekbone <= 1.0) scores.Oval += 2;
    if (jawToCheekbone >= 0.7 && jawToCheekbone <= 0.9) scores.Oval += 2;
    if (jawAngle >= 110 && jawAngle <= 150) scores.Oval += 2;

    /* ── Round ─────────────────────────────────────────────── */
    // Width ≈ length, soft angles
    if (widthToLength >= 0.85 && widthToLength <= 1.05) scores.Round += 3;
    if (jawAngle >= 130) scores.Round += 2;
    if (jawToCheekbone >= 0.8 && jawToCheekbone <= 1.0) scores.Round += 1;
    if (foreheadToCheekbone >= 0.85 && foreheadToCheekbone <= 1.05) scores.Round += 1;

    /* ── Square ────────────────────────────────────────────── */
    // Width ≈ length, strong angular jaw
    if (widthToLength >= 0.82 && widthToLength <= 1.05) scores.Square += 2;
    if (jawAngle < 125) scores.Square += 3;
    if (jawToCheekbone >= 0.88) scores.Square += 3;
    if (foreheadToCheekbone >= 0.88 && foreheadToCheekbone <= 1.05) scores.Square += 1;

    /* ── Heart ─────────────────────────────────────────────── */
    // Wide forehead, narrow chin/jaw
    if (foreheadToJaw >= 1.15) scores.Heart += 3;
    if (foreheadToCheekbone >= 0.95) scores.Heart += 2;
    if (jawToCheekbone < 0.78) scores.Heart += 2;
    if (chinToFaceLength < 0.25) scores.Heart += 2;

    /* ── Diamond ───────────────────────────────────────────── */
    // Cheekbones wider than forehead AND jaw
    if (foreheadToCheekbone < 0.85) scores.Diamond += 3;
    if (jawToCheekbone < 0.82) scores.Diamond += 3;
    if (widthToLength >= 0.65 && widthToLength <= 0.90) scores.Diamond += 1;

    /* ── Oblong ────────────────────────────────────────────── */
    // Face much longer than wide
    if (widthToLength < 0.70) scores.Oblong += 4;
    else if (widthToLength < 0.75) scores.Oblong += 2;
    if (jawToCheekbone >= 0.75 && jawToCheekbone <= 0.95) scores.Oblong += 1;
    if (foreheadToCheekbone >= 0.80 && foreheadToCheekbone <= 1.0) scores.Oblong += 1;

    // Find the winning shape
    const maxScore = Math.max(...Object.values(scores));
    const shape = Object.keys(scores).find((k) => scores[k] === maxScore) || 'Oval';

    // Confidence: ratio of winning score to total possible points
    const totalPoints = Object.values(scores).reduce((a, b) => a + b, 0);
    const confidence = totalPoints > 0
        ? Math.min(Math.round((maxScore / totalPoints) * 100 + 15), 98)
        : 50;

    return {
        shape,
        confidence,
        ratios: {
            widthToLength: Math.round(widthToLength * 100) / 100,
            jawToCheekbone: Math.round(jawToCheekbone * 100) / 100,
            foreheadToCheekbone: Math.round(foreheadToCheekbone * 100) / 100,
            foreheadToJaw: Math.round(foreheadToJaw * 100) / 100,
            chinToFaceLength: Math.round(chinToFaceLength * 100) / 100
        },
        measurements: {
            foreheadWidth: Math.round(measurements.foreheadWidth),
            cheekboneWidth: Math.round(measurements.cheekboneWidth),
            jawWidth: Math.round(measurements.jawWidth),
            faceLength: Math.round(measurements.faceLength),
            jawAngle: Math.round(measurements.jawAngle)
        },
        description: SHAPE_DESCRIPTIONS[shape]
    };
}

/* ────────────────────────── Public API ─────────────────────────────────── */

/**
 * Analyze face shape from 68 landmark points.
 * @param {Array<{x: number, y: number}>} landmarks – 68 landmark positions.
 * @returns {{ shape: string, confidence: number, ratios: Object, measurements: Object, description: string }}
 */
export function analyzeFaceShape(landmarks) {
    if (!landmarks || landmarks.length < 68) {
        throw new Error('Invalid landmarks: expected 68 points.');
    }

    const measurements = extractMeasurements(landmarks);
    const ratios = computeRatios(measurements);
    return classifyShape(ratios, measurements);
}
