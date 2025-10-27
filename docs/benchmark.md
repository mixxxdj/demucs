# Demucs Benchmarking Guide

This guide explains how to run performance benchmarks for Demucs music source separation using both PyTorch and C++ ONNX implementations.

## Overview

The benchmarking system provides comprehensive evaluation including:

- **Separation timing** - How long inference takes
- **SI-SDR evaluation** - Audio quality metrics (Scale-Invariant Signal-to-Distortion Ratio)
- **JSON results output** - Structured data for further analysis
- **Resume capability** - Skip already processed tracks

Both benchmark scripts use a shared `benchmark_common.py` module to ensure consistent evaluation methodology.

## Prerequisites

### Dataset Requirements

- **MusDB-HQ dataset** - Download from [https://sigsep.github.io/datasets/musdb.html](https://sigsep.github.io/datasets/musdb.html)
- Expected directory structure:
  ```
  musdb-root/
  ├── test/
  │   ├── track1/
  │   │   ├── mixture.wav
  │   │   ├── drums.wav
  │   │   ├── bass.wav
  │   │   ├── other.wav
  │   │   └── vocals.wav
  │   ├── track2/
  │   └── ...
  └── train/ (optional)
  ```

### Python Dependencies

```bash
pip install torch torchaudio torchmetrics numpy
```

## PyTorch Benchmark

The PyTorch benchmark uses the original Demucs PyTorch implementation for inference.

### Basic Usage

```bash
cd benchmark
python benchmark-pytorch.py --musdb-root /path/to/musdb-hq/
```

### Arguments

- `--musdb-root` (required): Path to MusDB dataset root directory
- `--output-root` (optional): Where to store separated outputs (default: inside musdb-root)
- `--output-dir` (optional): Output directory name (default: `test-separated-pytorch`)
- `--json-out` (optional): JSON file for benchmark results (default: `benchmark_results_pytorch.json`)
- `--force-reseparate` (optional): Force re-separation even if files already exist

## C++ ONNX Benchmark

The C++ ONNX benchmark uses a C++ CLI tool with ONNX runtime for inference.

### Prerequisites

- **C++ CLI tool built** - The C++ CLI build path.
- **ONNX model file** - The exported ONNX model path.

### Basic Usage

```bash
cd benchmark
python benchmark-cpp-onnx.py --musdb-root /path/to/musdb-hq/
```

### Arguments

- `--musdb-root` (required): Path to MusDB dataset root directory
- `--output-root` (optional): Where to store separated outputs (default: inside musdb-root)
- `--output-dir` (optional): Output directory name (default: `test-separated-cpp`)
- `--json-out` (optional): JSON file for benchmark results (default: `benchmark_results_cpp.json`)
- `--cli-path` (optional): Path to C++ ONNX CLI executable (default: `../cppscripts/build/build-cli/demucs`)
- `--model-path` (optional): Path to ONNX model file (default: `../onnx-models/htdemucs.ort`)
- `--force-reseparate` (optional): Force re-separation even if files already exist

## Output Files

### Separated Audio

Both benchmarks create the same output structure:
```
output-directory/
├── track1/
│   ├── target_0_drums.wav
│   ├── target_1_bass.wav
│   ├── target_2_other.wav
│   └── target_3_vocals.wav
├── track2/
└── ...
```

### JSON Results

Benchmark results are saved as JSON files with detailed statistics:

```json
{
  "summary": {
    "model_identifier": "htdemucs",
    "total_tracks": 50,
    "total_proc_sec": 0.0,
    "total_wall_time_sec": 0.011753082275390625,
    "total_audio_sec": 12470.980884353747,
    "avg_time_per_min": 0.0,
    "sisdr_mean_per_stem": {
      "drums": 9.457932338714599,
      "bass": 7.763716798424721,
      "other": 4.687467608451843,
      "vocals": 7.8351172041893005
    },
    "overall_sisdr_mean": 7.436058487445116
  },
  "per_track_stats": [...],
  "per_track_sisdr": [...]
}
```
