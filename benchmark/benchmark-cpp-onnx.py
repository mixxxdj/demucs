"""
C++ ONNX-based Demucs benchmarking script.

This script uses C++ ONNX CLI tool for inference and imports common functionality
from benchmark_common to avoid code duplication.
"""

import argparse
import subprocess
from pathlib import Path

from benchmark_common import run_benchmark

DEFAULT_CLI_PATH = '../cppscripts/build/build-cli/demucs'
DEFAULT_MODEL_PATH = '../onnx-models/htdemucs.ort'


def separate_cpp_onnx(mixture_path, out_dir, cli_path, model_path):
    """Separate audio using C++ ONNX CLI tool."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [str(cli_path), str(model_path), str(mixture_path), str(out_dir)]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='Benchmark Demucs separation using C++ ONNX CLI')
    parser.add_argument('--musdb-root', type=Path, help='Path to MusDB root', required=True)
    parser.add_argument('--output-root', type=Path, default=None, help='Where to store separated outputs (default: inside musdb-root)')
    parser.add_argument('--output-dir', type=str, default='test-separated-cpp', help='Output directory name (default: test-separated-cpp)')
    parser.add_argument('--json-out', type=str, default='benchmark_results_cpp.json', help='Output JSON file for benchmarks')
    parser.add_argument('--cli-path', type=str, default=DEFAULT_CLI_PATH, help='Path to the C++ ONNX CLI executable')
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH, help='Path to the ONNX model file')
    parser.add_argument('--force-reseparate', action='store_true', help='Force re-separation even if files already exist')
    args = parser.parse_args()
    
    # Setup paths
    musdb_root = Path(args.musdb_root)
    output_root = Path(args.output_root) if args.output_root else musdb_root
    out_dir = output_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cli_path = Path(args.cli_path)
    model_path = Path(args.model_path)
    
    print(f'Using CLI: {cli_path}')
    print(f'Using model: {model_path}')

    # Run benchmark using common flow
    run_benchmark(
        musdb_root=musdb_root,
        out_dir=out_dir,
        force_reseparate=args.force_reseparate,
        json_out=args.json_out,
        separate_func=separate_cpp_onnx,
        model_identifier=str(model_path),
        # Arguments passed to separate_cpp_onnx
        cli_path=cli_path,
        model_path=model_path
    )


if __name__ == '__main__':
    main()