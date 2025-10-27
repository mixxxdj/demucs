"""
PyTorch-based Demucs benchmarking script.

This script uses PyTorch for inference and imports common functionality
from benchmark_common to avoid code duplication.
"""

import argparse
from pathlib import Path

import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model

from benchmark_common import run_benchmark, STEM_MAP, STEM_NAMES

DEFAULT_MODEL = 'htdemucs'


def separate_pytorch(mixture_path, out_dir, model_name):
    """Separate audio using PyTorch Demucs model."""
    # Load model
    model = get_model(model_name)
    
    # Load and preprocess audio
    audio, rate = torchaudio.load(str(mixture_path))
    if rate != 44100:
        audio = torchaudio.functional.resample(audio, rate, 44100)
    
    # Normalize
    ref = audio.mean(0)
    audio = (audio - ref.mean()) / ref.std()
    
    # Apply model
    sources = apply_model(model, audio[None])[0]
    
    # Denormalize
    sources = sources * ref.std() + ref.mean()
    
    # Save stems
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for target_idx in range(len(STEM_NAMES)):
        target_name = STEM_MAP[target_idx]
        out_audio = sources[target_idx].detach().cpu()
        out_path = out_dir / f'target_{target_idx}_{target_name}.wav'
        torchaudio.save(str(out_path), out_audio, sample_rate=44100)


def main():
    parser = argparse.ArgumentParser(description='PyTorch Demucs Benchmarking Script')
    parser.add_argument('--musdb-root', type=Path, help='Path to MusDB root', required=True)
    parser.add_argument('--output-root', type=Path, default=None, help='Where to store separated outputs (default: inside musdb-root)')
    parser.add_argument('--output-dir', type=str, default='test-separated-pytorch', help='Output directory name (default: test-separated-pytorch)')
    parser.add_argument('--json-out', type=str, default='benchmark_results_pytorch.json', help='Output JSON file for benchmarks')
    parser.add_argument('--force-reseparate', action='store_true', help='Force re-separation even if files already exist')
    args = parser.parse_args()

    # Setup paths
    musdb_root = Path(args.musdb_root)
    output_root = Path(args.output_root) if args.output_root else musdb_root
    out_dir = output_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = DEFAULT_MODEL
    print(f'Using model: {model_name}')

    # Run benchmark using common flow
    run_benchmark(
        musdb_root=musdb_root,
        out_dir=out_dir,
        force_reseparate=args.force_reseparate,
        json_out=args.json_out,
        separate_func=separate_pytorch,
        model_identifier=model_name,
        # Arguments passed to separate_pytorch
        model_name=model_name
    )


if __name__ == '__main__':
    main()