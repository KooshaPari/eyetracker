//! Image processing utilities for eye tracking
//!
//! Provides preprocessing functions optimized for eye detection.

use crate::{Frame, FrameFormat};
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

/// Preprocess a frame for eye tracking inference
pub fn preprocess_frame(frame: &Frame, options: &PreprocessOptions) -> Vec<f32> {
    // Start with grayscale if requested
    let source = if options.to_grayscale || frame.format != FrameFormat::Grayscale8 {
        frame.to_grayscale()
    } else {
        frame.clone()
    };

    let mut pixels: Vec<f32> = Vec::with_capacity(
        (options.target_width * options.target_height) as usize,
    );

    // Resize and convert to normalized floats
    let src_width = source.width;
    let src_height = source.height;
    let dst_width = options.target_width;
    let dst_height = options.target_height;

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
                source.data.get(idx).copied().unwrap_or(0) as f32
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

            pixels.push(normalized);
        }
    }

    // Apply histogram equalization if requested
    if options.equalize_histogram {
        equalize_histogram_inplace(&mut pixels);
    }

    pixels
}

/// Equalize histogram in-place (CLAHE-lite)
fn equalize_histogram_inplace(pixels: &mut [f32]) {
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
pub fn detect_eye_region(frame: &Frame) -> Option<EyeRegion> {
    // Simplified heuristic detection based on brightness patterns
    // This is a fallback - proper detection requires ML model

    let gray = frame.to_grayscale();
    let width = gray.width;
    let height = gray.height;

    // Assume face is in center-upper portion of frame
    let face_x = width / 4;
    let face_y = height / 6;
    let face_w = width / 2;
    let face_h = height / 2;

    // Within face region, look for eye-like bright spots
    let mut best_x = face_x + face_w / 4;
    let mut best_y = face_y + face_h / 4;
    let mut best_score = 0.0;

    // Scan for bright regions (eyes reflect IR in many setups)
    let step = 10;
    for y in (face_y..(face_y + face_h - 50)).step_by(step as usize) {
        for x in (face_x..(face_x + face_w - 50)).step_by(step as usize) {
            let mut sum = 0.0;
            let mut count = 0;

            for dy in 0..30 {
                for dx in 0..30 {
                    if (y + dy) < height && (x + dx) < width {
                        let idx = ((y + dy) * width + (x + dx)) as usize;
                        if idx < gray.data.len() {
                            sum += gray.data[idx] as f32;
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

/// Crop a frame to a specific region
pub fn crop_eye_region(frame: &Frame, region: &EyeRegion) -> Frame {
    let x = region.x.min(frame.width - 1);
    let y = region.y.min(frame.height - 1);
    let w = region.width.min(frame.width - x);
    let h = region.height.min(frame.height - y);

    let bpp = frame.format.bytes_per_pixel();
    let src_stride = frame.width as usize * bpp;
    let dst_stride = w as usize * bpp;

    let mut data = vec![0u8; (w * h) as usize * bpp];

    for dy in 0..h {
        let src_row = (y + dy) as usize * src_stride;
        let dst_row = dy as usize * dst_stride;
        let src_start = src_row + x as usize * bpp;
        let src_end = src_start + dst_stride;

        if src_end <= frame.data.len() {
            data[dst_row..dst_row + dst_stride]
                .copy_from_slice(&frame.data[src_start..src_end]);
        }
    }

    Frame {
        data,
        width: w,
        height: h,
        format: frame.format,
        metadata: frame.metadata.clone(),
    }
}

/// Extract eye landmarks (simplified)
/// Returns normalized (0-1) coordinates
pub fn extract_eye_landmarks(frame: &Frame, eye_region: &EyeRegion) -> Vec<(f32, f32)> {
    let mut landmarks = Vec::with_capacity(6);

    // Simplified: estimate landmarks based on region
    let (cx, cy) = eye_region.center();
    let w = eye_region.width as f32;
    let h = eye_region.height as f32;

    // Inner corner
    landmarks.push((
        (eye_region.x as f32 + w * 0.2) / frame.width as f32,
        (cy as f32) / frame.height as f32,
    ));

    // Outer corner
    landmarks.push((
        (eye_region.x as f32 + w * 0.8) / frame.width as f32,
        (cy as f32) / frame.height as f32,
    ));

    // Center (pupil estimate)
    landmarks.push((cx as f32 / frame.width as f32, cy as f32 / frame.height as f32));

    // Upper lid points
    landmarks.push((
        (cx as f32 - w * 0.15) / frame.width as f32,
        (eye_region.y as f32 + h * 0.2) / frame.height as f32,
    ));
    landmarks.push((
        cx as f32 / frame.width as f32,
        (eye_region.y as f32 + h * 0.1) / frame.height as f32,
    ));
    landmarks.push((
        (cx as f32 + w * 0.15) / frame.width as f32,
        (eye_region.y as f32 + h * 0.2) / frame.height as f32,
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
        let frame = Frame::new(vec![0u8; 640 * 480 * 3], 640, 480, FrameFormat::Rgb8);

        let region = EyeRegion {
            x: 100,
            y: 50,
            width: 80,
            height: 40,
            confidence: 1.0,
            is_left: true,
        };

        let cropped = crop_eye_region(&frame, &region);
        assert_eq!(cropped.width, 80);
        assert_eq!(cropped.height, 40);
    }

    #[test]
    fn test_landmark_extraction() {
        let frame = Frame::new(vec![0u8; 640 * 480], 640, 480, FrameFormat::Grayscale8);

        let region = EyeRegion {
            x: 200,
            y: 100,
            width: 100,
            height: 50,
            confidence: 1.0,
            is_left: true,
        };

        let landmarks = extract_eye_landmarks(&frame, &region);
        assert_eq!(landmarks.len(), 6);

        // All landmarks should be normalized
        for (x, y) in &landmarks {
            assert!(*x >= 0.0 && *x <= 1.0);
            assert!(*y >= 0.0 && *y <= 1.0);
        }
    }
}
