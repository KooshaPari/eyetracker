#!/bin/bash
# ONNX model downloader for eyetracker
# Downloads pre-trained face detection and face mesh models
# Usage: ./download-models.sh [--force] [models_dir]
#
# Default: ~/Library/Application Support/eyetracker/models (macOS)
# Override: EYETRACKER_MODELS environment variable or argument
#
# Flags:
#   --force, -f    Re-download models even if they exist

set -euo pipefail

FORCE=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE=true
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -n "${POSITIONAL_ARGS[0]:-}" ]; then
    MODELS_DIR="${POSITIONAL_ARGS[0]}"
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
# All models are publicly available under permissive licenses (Apache 2.0).
#
# Face detection: YuNet from OpenCV Zoo
#   https://github.com/opencv/opencv_zoo/ (Apache 2.0)
#
# Face landmark: MediaPipe-compatible model from sherpa-onnx
#   https://github.com/k2-fsa/sherpa-onnx (Apache 2.0)
#
# Fallback: OpenCV Zoo face landmark model

FACE_DETECTION_URL="https://github.com/opencv/opencv_zoo/raw/refs/heads/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
FACE_LANDMARK_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/face_landmark_mediapipe_256x256.onnx"
FACE_LANDMARK_FALLBACK="https://github.com/opencv/opencv_zoo/raw/refs/heads/master/models/face_landmark/face_landmark_256x256.onnx"

declare -A MODELS
MODELS[face_detection.onnx]="$FACE_DETECTION_URL"
MODELS[face_landmark.onnx]="$FACE_LANDMARK_URL"

download_model() {
    local name="$1"
    local url="$2"
    local fallback_url="${3:-}"
    local output="$MODELS_DIR/$name"

    # Check if already exists and is valid
    if [ -f "$output" ] && [ "$FORCE" = false ]; then
        local size
        size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null || echo "0")
        if [ "$size" -gt 1024 ]; then
            echo "   [SKIP] $name ($size bytes) — use --force to re-download"
            return 0
        else
            echo "   [WARN] $name exists but is too small ($size bytes), re-downloading..."
        fi
    fi

    echo "   [DL]   $name"
    echo "          from: $url"

    # Try primary URL
    if command -v curl &>/dev/null; then
        if curl -fL -o "$output" "$url" 2>/dev/null; then
            : # success
        elif [ -n "$fallback_url" ]; then
            echo "   [WARN] Primary URL failed, trying fallback..."
            echo "          from: $fallback_url"
            rm -f "$output"
            curl -fL -o "$output" "$fallback_url" || {
                echo "   [FAIL] All sources failed for $name"
                rm -f "$output"
                return 1
            }
        else
            echo "   [FAIL] curl failed for $name"
            rm -f "$output"
            return 1
        fi
    elif command -v wget &>/dev/null; then
        if wget -q -O "$output" "$url" 2>/dev/null; then
            : # success
        elif [ -n "$fallback_url" ]; then
            echo "   [WARN] Primary URL failed, trying fallback..."
            echo "          from: $fallback_url"
            rm -f "$output"
            wget -q -O "$output" "$fallback_url" || {
                echo "   [FAIL] All sources failed for $name"
                rm -f "$output"
                return 1
            }
        else
            echo "   [FAIL] wget failed for $name"
            rm -f "$output"
            return 1
        fi
    else
        echo "   [FAIL] Neither curl nor wget found"
        return 1
    fi

    # Verify downloaded file
    if [ -f "$output" ]; then
        local size
        size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null || echo "0")
        if [ "$size" -lt 1024 ]; then
            echo "   [FAIL] $name is too small ($size bytes), download may be corrupt"
            rm -f "$output"
            return 1
        fi
        echo "   [OK]   $name ($size bytes)"
    else
        echo "   [FAIL] $name was not created"
        return 1
    fi
}

SUCCESS=0
FAILURE=0

for name in "${!MODELS[@]}"; do
    case "$name" in
        face_landmark.onnx)
            if download_model "$name" "${MODELS[$name]}" "$FACE_LANDMARK_FALLBACK"; then
                SUCCESS=$((SUCCESS + 1))
            else
                FAILURE=$((FAILURE + 1))
            fi
            ;;
        *)
            if download_model "$name" "${MODELS[$name]}"; then
                SUCCESS=$((SUCCESS + 1))
            else
                FAILURE=$((FAILURE + 1))
            fi
            ;;
    esac
done

# Summary
echo ""
echo "==> Download summary:"
echo "    Success: $SUCCESS"
echo "    Failure: $FAILURE"
echo "    Directory: $MODELS_DIR"
echo ""

if [ "$FAILURE" -gt 0 ]; then
    echo "==> Some models failed to download."
    echo "    You can try manually:"
    echo "    curl -fL -o \"$MODELS_DIR/face_detection.onnx\" \"$FACE_DETECTION_URL\""
    echo "    curl -fL -o \"$MODELS_DIR/face_landmark.onnx\" \"$FACE_LANDMARK_URL\""
    exit 1
fi

echo "==> All models downloaded successfully!"
echo "    To use: export EYETRACKER_MODELS=\"$MODELS_DIR\""
