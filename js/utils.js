/**
 * utils.js – Utility helpers for the Face Shape Detector
 * Handles image compression, downloads, localStorage, and DOM helpers.
 */

/**
 * Compress and resize an image file using an offscreen canvas.
 * @param {File} file – The image file to compress.
 * @param {number} [maxWidth=800] – Maximum width in pixels.
 * @param {number} [quality=0.85] – JPEG quality (0–1).
 * @returns {Promise<HTMLImageElement>} – A loaded HTMLImageElement.
 */
export async function compressImage(file, maxWidth = 800, quality = 0.85) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onerror = () => reject(new Error('Failed to read image file.'));
        reader.onload = (e) => {
            const img = new Image();
            img.onerror = () => reject(new Error('Failed to decode image.'));
            img.onload = () => {
                // Skip compression if already small enough
                if (img.width <= maxWidth) {
                    resolve(img);
                    return;
                }

                const scale = maxWidth / img.width;
                const canvas = document.createElement('canvas');
                canvas.width = maxWidth;
                canvas.height = Math.round(img.height * scale);

                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                const compressed = new Image();
                compressed.onerror = () => reject(new Error('Failed to compress image.'));
                compressed.onload = () => resolve(compressed);
                compressed.src = canvas.toDataURL('image/jpeg', quality);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
}

/**
 * Download a canvas element as a PNG image.
 * @param {HTMLCanvasElement} canvas – The canvas to export.
 * @param {string} [filename='face-shape-result.png'] – Download filename.
 */
export function downloadCanvasAsImage(canvas, filename = 'face-shape-result.png') {
    const link = document.createElement('a');
    link.download = filename;
    link.href = canvas.toDataURL('image/png');
    link.click();
}

/**
 * Save data to localStorage as JSON.
 * @param {string} key
 * @param {*} data
 */
export function saveToLocalStorage(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
    } catch {
        // Storage full or unavailable – silently ignore
    }
}

/**
 * Load data from localStorage.
 * @param {string} key
 * @returns {*|null}
 */
export function loadFromLocalStorage(key) {
    try {
        const raw = localStorage.getItem(key);
        return raw ? JSON.parse(raw) : null;
    } catch {
        return null;
    }
}

/**
 * Show a DOM element by removing the 'hidden' class.
 * @param {HTMLElement} el
 */
export function showElement(el) {
    if (el) el.classList.remove('hidden');
}

/**
 * Hide a DOM element by adding the 'hidden' class.
 * @param {HTMLElement} el
 */
export function hideElement(el) {
    if (el) el.classList.add('hidden');
}

/**
 * Create a debounced version of a function.
 * @param {Function} fn
 * @param {number} ms – Debounce delay in milliseconds.
 * @returns {Function}
 */
export function debounce(fn, ms) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), ms);
    };
}

/**
 * Calculate Euclidean distance between two {x, y} points.
 * @param {{x: number, y: number}} a
 * @param {{x: number, y: number}} b
 * @returns {number}
 */
export function distance(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

/**
 * Get the midpoint between two {x, y} points.
 * @param {{x: number, y: number}} a
 * @param {{x: number, y: number}} b
 * @returns {{x: number, y: number}}
 */
export function midpoint(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}
