//! Image processing utilities for eye tracking
//!
//! Provides preprocessing functions optimized for eye detection.
//! Works with raw pixel data for ONNX inference.

use serde::{Deserialize, Serialize};

/// Eye region detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyeRegion {
    /// Bounding box x (top-left)
    pub x: u32,
    /// Bounding box y (top-left)
    pub y: u32,
    /// Bounding box width
    pub width: u32,
    /// Bounding box height
    pub height: u32,
    /// Detection confidence 0-1
    pub confidence: f32,
    /// Whether this is the left eye
    pub is_left: bool,
}

impl EyeRegion {
    /// Get bounding box as (x, y, width, height)
    pub fn bounds(&self) -> (u32, u32, u32, u32) {
        (self.x, self.y, self.width, self.height)
    }

    /// Get center point of the region
    pub fn center(&self) -> (u32, u32) {
        (
            self.x + self.width / 2,
            self.y + self.height / 2,
        )
    }

    /// Get the aspect ratio of the bounding box
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

/// Preprocessing options for eye tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessOptions {
    /// Target width after resize
    pub target_width: u32,
    /// Target height after resize
    pub target_height: u32,
    /// Apply histogram equalization
    pub equalize_histogram: bool,
    /// Normalize pixel values to [0, 1]
    pub normalize: bool,
    /// Apply Gaussian blur
    pub blur_kernel_size: Option<u32>,
    /// Convert to grayscale
    pub to_grayscale: bool,
    /// Adjust contrast
    pub contrast_factor: f32,
    /// Adjust brightness
    pub brightness_offset: i32,
}

impl Default for PreprocessOptions {
    fn default() -> Self {
        Self {
            target_width: 224,
            target_height: 224,
            equalize_histogram: true,
            normalize: true,
            blur_kernel_size: None,
            to_grayscale: true,
            contrast_factor: 1.0,
            brightness_offset: 0,
        }
    }
}

impl PreprocessOptions {
    /// Create options optimized for ONNX inference
    pub fn for_onnx() -> Self {
        Self {
            target_width: 224,
            target_height: 224,
            equalize_histogram: false,
            normalize: true,
            blur_kernel_size: None,
            to_grayscale: false, // Keep RGB for ML models
            contrast_factor: 1.0,
            brightness_offset: 0,
        }
    }

    /// Create options for edge-based detection
    pub fn for_edge_detection() -> Self {
        Self {
            target_width: 320,
            target_height: 240,
            equalize_histogram: true,
            normalize: true,
            blur_kernel_size: Some(3),
            to_grayscale: true,
            contrast_factor: 1.5,
            brightness_offset: 10,
        }
    }
}

/// Preprocess raw pixel data for eye tracking inference
pub fn preprocess_frame(pixels: &[u8], width: u32, height: u32, options: &PreprocessOptions) -> Vec<f32> {
    let src_width = width;
    let src_height = height;
    let dst_width = options.target_width;
    let dst_height = options.target_height;

    let mut output: Vec<f32> = Vec::with_capacity(
        (dst_width * dst_height) as usize * if options.to_grayscale { 1 } else { 3 },
    );

    // Convert to grayscale first if needed
    let grayscale: Vec<u8> = if options.to_grayscale && pixels.len() as u32 == width * height * 3 {
        rgb_to_grayscale(pixels, width, height)
    } else if options.to_grayscale {
        pixels.to_vec()
    } else {
        pixels.to_vec()
    };

    let src_data: &[u8] = if options.to_grayscale {
        &grayscale
    } else {
        pixels
    };

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            // Bilinear interpolation
            let src_x = (dst_x as f32) * (src_width as f32) / (dst_width as f32);
            let src_y = (dst_y as f32) * (src_height as f32) / (dst_height as f32);

            let x0 = src_x as u32;
            let y0 = src_y as u32;
            let x1 = (x0 + 1).min(src_width - 1);
            let y1 = (y0 + 1).min(src_height - 1);

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let get_pixel = |x: u32, y: u32| -> f32 {
                let idx = (y * src_width + x) as usize;
                src_data.get(idx).copied().unwrap_or(0) as f32
            };

            // Bilinear interpolation
            let p00 = get_pixel(x0, y0);
            let p10 = get_pixel(x1, y0);
            let p01 = get_pixel(x0, y1);
            let p11 = get_pixel(x1, y1);

            let value = p00 * (1.0 - fx) * (1.0 - fy)
                + p10 * fx * (1.0 - fy)
                + p01 * (1.0 - fx) * fy
                + p11 * fx * fy;

            // Apply brightness/contrast adjustments
            let mut adjusted = value;
            adjusted = (adjusted - 128.0) * options.contrast_factor + 128.0;
            adjusted += options.brightness_offset as f32;
            adjusted = adjusted.clamp(0.0, 255.0);

            // Normalize to [0, 1]
            let normalized = if options.normalize {
                adjusted / 255.0
            } else {
                adjusted
            };

            output.push(normalized);
        }
    }

    // Apply histogram equalization if requested
    if options.equalize_histogram {
        equalize_histogram_inplace(&mut output);
    }

    output
}

/// Convert RGB pixels to grayscale
fn rgb_to_grayscale(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut grayscale = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            if idx + 2 < pixels.len() {
                // ITU-R BT.601 conversion
                let r = pixels[idx] as f32;
                let g = pixels[idx + 1] as f32;
                let b = pixels[idx + 2] as f32;
                let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                grayscale.push(gray as u8);
            }
        }
    }
    grayscale
}

/// Equalize histogram in-place (CLAHE-lite)
fn equalize_histogram_inplace(pixels: &mut [f32]) {
    if pixels.is_empty() {
        return;
    }

    // Build histogram
    let mut hist = [0u32; 256];
    for &p in pixels.iter() {
        let bin = ((p * 255.0).round() as usize).clamp(0, 255);
        hist[bin] += 1;
    }

    // Build CDF
    let mut cdf = [0u32; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    // Normalize CDF
    let cdf_min = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0);
    let total = pixels.len() as u32;
    let max_val = 255.0;

    for pixel in pixels.iter_mut() {
        let bin = ((*pixel * 255.0).round() as usize).clamp(0, 255);
        let cdf_val = cdf[bin];
        let equalized = ((cdf_val - cdf_min) as f32 / (total - cdf_min) as f32) * max_val;
        *pixel = equalized / 255.0;
    }
}

/// Simple face/eye region detection heuristic
/// Note: For production, use ML-based detection (MediaPipe, etc.)
pub fn detect_eye_region(pixels: &[u8], width: u32, height: u32) -> Option<EyeRegion> {
    // Convert to grayscale if RGB
    let gray = if pixels.len() as u32 == width * height * 3 {
        rgb_to_grayscale(pixels, width, height)
    } else {
        pixels.to_vec()
    };

    // Assume face is in center-upper portion of frame
    let face_x = width / 4;
    let face_y = height / 6;
    let face_w = width / 2;
    let face_h = height / 2;

    // Within face region, look for eye-like bright spots
    let mut best_x = face_x + face_w / 4;
    let mut best_y = face_y + face_h / 4;
    let mut best_score = 0.0f32;

    // Scan for bright regions (eyes reflect IR in many setups)
    let step = 10;
    for y in (face_y..(face_y + face_h.saturating_sub(50))).step_by(step as usize) {
        for x in (face_x..(face_x + face_w.saturating_sub(50))).step_by(step as usize) {
            let mut sum = 0.0f32;
            let mut count = 0u32;

            for dy in 0..30 {
                for dx in 0..30 {
                    if (y + dy) < height && (x + dx) < width {
                        let idx = ((y + dy) * width + (x + dx)) as usize;
                        if idx < gray.len() {
                            sum += gray[idx] as f32;
                            count += 1;
                        }
                    }
                }
            }

            if count > 0 {
                let avg = sum / count as f32;
                // Look for moderately bright regions (eyes aren't pure white)
                let score = if avg > 60.0 && avg < 200.0 {
                    avg / 128.0
                } else {
                    0.0
                };

                if score > best_score {
                    best_score = score;
                    best_x = x;
                    best_y = y;
                }
            }
        }
    }

    if best_score > 0.1 {
        Some(EyeRegion {
            x: best_x,
            y: best_y,
            width: 80,
            height: 40,
            confidence: best_score,
            is_left: best_x < width / 2,
        })
    } else {
        // Return default region in upper-center
        Some(EyeRegion {
            x: width / 3,
            y: height / 4,
            width: width / 3,
            height: height / 4,
            confidence: 0.3,
            is_left: true,
        })
    }
}

/// Crop a region from raw pixel data
pub fn crop_region(pixels: &[u8], width: u32, height: u32, region: &EyeRegion) -> Vec<u8> {
    let x = region.x.min(width.saturating_sub(1));
    let y = region.y.min(height.saturating_sub(1));
    let w = region.width.min(width.saturating_sub(x));
    let h = region.height.min(height.saturating_sub(y));

    let is_rgb = pixels.len() as u32 == width * height * 3;
    let bpp = if is_rgb { 3 } else { 1 };

    let src_stride = width as usize * bpp;
    let dst_stride = w as usize * bpp;

    let mut data = vec![0u8; (w * h) as usize * bpp];

    for dy in 0..h {
        let src_row = (y + dy) as usize * src_stride;
        let dst_row = dy as usize * dst_stride;
        let src_start = src_row + x as usize * bpp;
        let src_end = src_start + dst_stride;

        if src_end <= pixels.len() {
            data[dst_row..dst_row + dst_stride]
                .copy_from_slice(&pixels[src_start..src_end]);
        }
    }

    data
}

/// Extract eye landmarks (simplified)
/// Returns normalized (0-1) coordinates
pub fn extract_eye_landmarks(width: u32, height: u32, eye_region: &EyeRegion) -> Vec<(f32, f32)> {
    let mut landmarks = Vec::with_capacity(6);

    // Simplified: estimate landmarks based on region
    let (cx, cy) = eye_region.center();
    let w = eye_region.width as f32;
    let h = eye_region.height as f32;

    // Inner corner
    landmarks.push((
        (eye_region.x as f32 + w * 0.2) / width as f32,
        cy as f32 / height as f32,
    ));

    // Outer corner
    landmarks.push((
        (eye_region.x as f32 + w * 0.8) / width as f32,
        cy as f32 / height as f32,
    ));

    // Center (pupil estimate)
    landmarks.push((cx as f32 / width as f32, cy as f32 / height as f32));

    // Upper lid points
    landmarks.push((
        (cx as f32 - w * 0.15) / width as f32,
        (eye_region.y as f32 + h * 0.2) / height as f32,
    ));
    landmarks.push((
        cx as f32 / width as f32,
        (eye_region.y as f32 + h * 0.1) / height as f32,
    ));
    landmarks.push((
        (cx as f32 + w * 0.15) / width as f32,
        (eye_region.y as f32 + h * 0.2) / height as f32,
    ));

    landmarks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_options_defaults() {
        let opts = PreprocessOptions::default();
        assert_eq!(opts.target_width, 224);
        assert!(opts.normalize);
    }

    #[test]
    fn test_eye_region() {
        let region = EyeRegion {
            x: 100,
            y: 50,
            width: 80,
            height: 40,
            confidence: 0.8,
            is_left: true,
        };

        assert_eq!(region.bounds(), (100, 50, 80, 40));
        assert_eq!(region.center(), (140, 70));
        assert!((region.aspect_ratio() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_crop_region() {
        let pixels = vec![0u8; 640 * 480 * 3];
        let region = EyeRegion {
            x: 100,
            y: 50,
            width: 80,
            height: 40,
            confidence: 1.0,
            is_left: true,
        };

        let cropped = crop_region(&pixels, 640, 480, &region);
        assert_eq!(cropped.len(), 80 * 40 * 3);
    }

    #[test]
    fn test_landmark_extraction() {
        let region = EyeRegion {
            x: 200,
            y: 100,
            width: 100,
            height: 50,
            confidence: 1.0,
            is_left: true,
        };

        let landmarks = extract_eye_landmarks(640, 480, &region);
        assert_eq!(landmarks.len(), 6);

        // All landmarks should be normalized
        for (x, y) in &landmarks {
            assert!(*x >= 0.0 && *x <= 1.0);
            assert!(*y >= 0.0 && *y <= 1.0);
        }
    }
}
