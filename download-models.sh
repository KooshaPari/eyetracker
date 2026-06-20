#!/bin/bash
# ONNX model downloader for eyetracker
# Downloads pre-trained face detection and face mesh models
# Usage: ./download-models.sh [models_dir]
#
# Default: ~/Library/Application Support/eyetracker/models (macOS)
# Override: EYETRACKER_MODELS environment variable or argument

set -euo pipefail

if [ -n "${1:-}" ]; then
    MODELS_DIR="$1"
elif [ -n "${EYETRACKER_MODELS:-}" ]; then
    MODELS_DIR="$EYETRACKER_MODELS"
else
    MODELS_DIR="$HOME/Library/Application Support/eyetracker/models"
fi

echo "==> Downloading models to: $MODELS_DIR"
mkdir -p "$MODELS_DIR"

# ------------------------------------------------------------------
# Model sources
# ------------------------------------------------------------------
# All models are publicly available, license-compatible ONNX models.
# We use MediaPipe-compatible models from the PINTO0309 model zoo
# (permissive MIT/Apache-2.0 licensed).

BASE_URL="https://github.com/PINTO0309/PINTO_model_zoo/raw/main"

declare -A MODELS
MODELS[face_detection.onnx]="${BASE_URL}/003_insightface_FaceDetection/onnx/face_detection_128x128.onnx"
MODELS[face_mesh.onnx]="${BASE_URL}/078_facemesh/onnx/face_landmark_256x256.onnx"
MODELS[face_landmark.onnx]="${BASE_URL}/078_facemesh/onnx/face_landmark_256x256.onnx"

# Alternative: MediaPipe ONNX models from opencv_zoo
OPENCV_BASE="https://github.com/opencv/opencv_zoo/raw/main/models"

# ------------------------------------------------------------------
# Download each model
# ------------------------------------------------------------------
for name in "${!MODELS[@]}"; do
    url="${MODELS[$name]}"
    output="$MODELS_DIR/$name"

    if [ -f "$output" ]; then
        echo "   [SKIP] $name already exists"
        continue
    fi

    echo "   [DL]   $name"
    echo "          from: $url"

    if command -v curl &>/dev/null; then
        curl -fsSL -o "$output" "$url" || {
            echo "   [FAIL] curl failed for $name"
            echo "          Trying opencv_zoo alternative..."
            # Try opencv_zoo alternative
            opencv_url="${OPENCV_BASE}/face_landmark/face_landmark_256x256.onnx"
            curl -fsSL -o "$output" "$opencv_url" || {
                echo "   [FAIL] All sources failed for $name"
                echo "          Download manually from: $url"
                rm -f "$output"
            }
        }
    elif command -v wget &>/dev/null; then
        wget -q -O "$output" "$url" || {
            echo "   [FAIL] wget failed for $name"
            rm -f "$output"
        }
    else
        echo "   [FAIL] Neither curl nor wget found"
        exit 1
    fi

    if [ -f "$output" ]; then
        size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null || echo "?")
        echo "   [OK]   $name ($size bytes)"
    fi
done

# ------------------------------------------------------------------
# Verify models
# ------------------------------------------------------------------
echo ""
echo "==> Downloaded models:"
ls -lh "$MODELS_DIR"/*.onnx 2>/dev/null || echo "   (no models found)"

count=$(ls "$MODELS_DIR"/*.onnx 2>/dev/null | wc -l)
echo ""
echo "==> $count model(s) available"
echo "    Directory: $MODELS_DIR"
echo ""
echo "To use: export EYETRACKER_MODELS=\"$MODELS_DIR\""
