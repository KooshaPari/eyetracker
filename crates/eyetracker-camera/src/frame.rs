//! Frame data structures
//!
//! Represents captured camera frames with metadata.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Supported frame formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameFormat {
    /// 8-bit RGB (24 bits per pixel)
    Rgb8,
    /// 8-bit grayscale (8 bits per pixel)
    Grayscale8,
    /// 8-bit YUV (12 bits per pixel, YUV 4:2:2)
    Yuv422,
    /// 10-bit YUV (20 bits per pixel, YUV 4:2:2-10)
    Yuv422_10,
    /// 8-bit BGRA (32 bits per pixel)
    Bgra8,
    /// 16-bit raw (for specialized cameras)
    Raw16,
}

impl FrameFormat {
    /// Get bytes per pixel for this format
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            FrameFormat::Rgb8 => 3,
            FrameFormat::Grayscale8 => 1,
            FrameFormat::Yuv422 => 2,
            FrameFormat::Yuv422_10 => 3, // Approximate
            FrameFormat::Bgra8 => 4,
            FrameFormat::Raw16 => 2,
        }
    }

    /// Check if this format is grayscale
    pub fn is_grayscale(&self) -> bool {
        matches!(self, FrameFormat::Grayscale8)
    }

    /// Check if this format has color
    pub fn has_color(&self) -> bool {
        !self.is_grayscale()
    }
}

impl std::fmt::Display for FrameFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameFormat::Rgb8 => write!(f, "RGB8"),
            FrameFormat::Grayscale8 => write!(f, "Grayscale8"),
            FrameFormat::Yuv422 => write!(f, "YUV422"),
            FrameFormat::Yuv422_10 => write!(f, "YUV422-10"),
            FrameFormat::Bgra8 => write!(f, "BGRA8"),
            FrameFormat::Raw16 => write!(f, "Raw16"),
        }
    }
}

/// Frame metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMetadata {
    /// Capture timestamp
    pub timestamp: SystemTime,
    /// Frame sequence number
    pub frame_number: u64,
    /// Estimated FPS at time of capture
    pub fps: f64,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
}

impl FrameMetadata {
    /// Get frame dimensions as (width, height)
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Calculate frame size in bytes (approximate)
    pub fn size_bytes(&self, format: FrameFormat) -> usize {
        (self.width as usize) * (self.height as usize) * format.bytes_per_pixel()
    }
}

/// A captured camera frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    /// Raw pixel data
    pub data: Vec<u8>,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Pixel format
    pub format: FrameFormat,
    /// Capture metadata
    pub metadata: FrameMetadata,
}

impl Frame {
    /// Create a new frame with the given parameters
    pub fn new(data: Vec<u8>, width: u32, height: u32, format: FrameFormat) -> Self {
        let timestamp = SystemTime::now();
        Self {
            data,
            width,
            height,
            format,
            metadata: FrameMetadata {
                timestamp,
                frame_number: 0,
                fps: 0.0,
                width,
                height,
            },
        }
    }

    /// Get frame dimensions as (width, height)
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get frame area in pixels
    pub fn area(&self) -> u32 {
        self.width * self.height
    }

    /// Calculate expected data size in bytes
    pub fn expected_size(&self) -> usize {
        self.area() as usize * self.format.bytes_per_pixel()
    }

    /// Check if frame data size matches expected
    pub fn is_valid_size(&self) -> bool {
        self.data.len() >= self.expected_size()
    }

    /// Get raw data slice
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable raw data slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Convert to grayscale if not already
    pub fn to_grayscale(&self) -> Frame {
        if self.format == FrameFormat::Grayscale8 {
            return self.clone();
        }

        let mut gray_data = vec![0u8; (self.width * self.height) as usize];

        match self.format {
            FrameFormat::Rgb8 => {
                for i in 0..(self.width * self.height) as usize {
                    let r = self.data[i * 3] as f32;
                    let g = self.data[i * 3 + 1] as f32;
                    let b = self.data[i * 3 + 2] as f32;
                    // Luminance formula
                    gray_data[i] = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                }
            }
            FrameFormat::Bgra8 => {
                for i in 0..(self.width * self.height) as usize {
                    let b = self.data[i * 4] as f32;
                    let g = self.data[i * 4 + 1] as f32;
                    let r = self.data[i * 4 + 2] as f32;
                    gray_data[i] = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                }
            }
            _ => {
                // Fallback: just use first channel
                let bpp = self.format.bytes_per_pixel();
                for i in 0..(self.width * self.height) as usize {
                    gray_data[i] = self.data[i * bpp];
                }
            }
        }

        Frame {
            data: gray_data,
            width: self.width,
            height: self.height,
            format: FrameFormat::Grayscale8,
            metadata: FrameMetadata {
                timestamp: self.metadata.timestamp,
                frame_number: self.metadata.frame_number,
                fps: self.metadata.fps,
                width: self.width,
                height: self.height,
            },
        }
    }

    /// Get pixel at (x, y) as RGB tuple
    pub fn get_pixel_rgb(&self, x: u32, y: u32) -> Option<(u8, u8, u8)> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let idx = ((y * self.width + x) * self.format.bytes_per_pixel() as u32) as usize;

        match self.format {
            FrameFormat::Rgb8 if idx + 2 < self.data.len() => {
                Some((self.data[idx], self.data[idx + 1], self.data[idx + 2]))
            }
            FrameFormat::Bgra8 if idx + 3 < self.data.len() => {
                Some((self.data[idx + 2], self.data[idx + 1], self.data[idx]))
            }
            FrameFormat::Grayscale8 if idx < self.data.len() => {
                let v = self.data[idx];
                Some((v, v, v))
            }
            _ => None,
        }
    }

    /// Calculate mean brightness
    pub fn mean_brightness(&self) -> f64 {
        let gray = self.to_grayscale();
        let sum: u64 = gray.data.iter().map(|&p| p as u64).sum();
        sum as f64 / gray.data.len() as f64
    }

    /// Calculate brightness histogram
    pub fn brightness_histogram(&self, bins: usize) -> Vec<u32> {
        let gray = self.to_grayscale();
        let mut hist = vec![0u32; bins];
        let bin_size = 256 / bins;

        for &pixel in &gray.data {
            let bin = (pixel as usize / bin_size).min(bins - 1);
            hist[bin] += 1;
        }

        hist
    }
}

impl std::fmt::Display for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Frame {}x{} ({}) fps={:.1} frame={}",
            self.width,
            self.height,
            self.format,
            self.metadata.fps,
            self.metadata.frame_number
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_format_bpp() {
        assert_eq!(FrameFormat::Rgb8.bytes_per_pixel(), 3);
        assert_eq!(FrameFormat::Grayscale8.bytes_per_pixel(), 1);
        assert_eq!(FrameFormat::Bgra8.bytes_per_pixel(), 4);
    }

    #[test]
    fn test_frame_creation() {
        let data = vec![0u8; 640 * 480 * 3];
        let frame = Frame::new(data, 640, 480, FrameFormat::Rgb8);

        assert_eq!(frame.width, 640);
        assert_eq!(frame.height, 480);
        assert!(frame.is_valid_size());
    }

    #[test]
    fn test_frame_to_grayscale() {
        // Create an RGB frame
        let mut data = vec![0u8; 4 * 3];
        data[0] = 255; // R
        data[1] = 128; // G
        data[2] = 64;  // B

        let frame = Frame::new(data, 2, 2, FrameFormat::Rgb8);
        let gray = frame.to_grayscale();

        assert_eq!(gray.format, FrameFormat::Grayscale8);
        assert!(gray.is_valid_size());
    }

    #[test]
    fn test_frame_pixel_access() {
        let mut data = vec![0u8; 4 * 3];
        data[0] = 255; // R at (0,0)
        data[1] = 0;   // G
        data[2] = 0;   // B

        let frame = Frame::new(data, 2, 2, FrameFormat::Rgb8);

        assert_eq!(frame.get_pixel_rgb(0, 0), Some((255, 0, 0)));
        assert_eq!(frame.get_pixel_rgb(1, 0), Some((0, 0, 0))); // Black
        assert_eq!(frame.get_pixel_rgb(10, 10), None); // Out of bounds
    }

    #[test]
    fn test_frame_histogram() {
        let data = vec![128u8; 100];
        let frame = Frame::new(data, 10, 10, FrameFormat::Grayscale8);

        let hist = frame.brightness_histogram(16);
        assert_eq!(hist.len(), 16);
        assert_eq!(hist[8], 100); // All pixels in center bin
    }
}
