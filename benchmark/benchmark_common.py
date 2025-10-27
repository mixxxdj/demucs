"""
Common functionality for Demucs benchmarking scripts.

This module contains shared code between benchmark-pytorch.py and benchmark-cpp-onnx.py
to eliminate duplication and ensure consistency across different backends.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Callable

import numpy as np
import torch
import torchaudio
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

STEM_NAMES = ['drums', 'bass', 'other', 'vocals']
STEM_MAP = {i: name for i, name in enumerate(STEM_NAMES)}
GT_STEM_FILES = [f'{name}.wav' for name in STEM_NAMES]
SEP_STEM_FILES = [f'target_{i}_{name}.wav' for i, name in enumerate(STEM_NAMES)]


def find_mixture_wavs(root):
    """Find all mixture.wav files in the directory tree."""
    root = Path(root)
    return list(root.rglob('mixture.wav'))


def check_track_separated(sep_dir):
    """Check if all expected stem files exist for a track."""
    if not sep_dir.exists():
        return False
    
    for sep_stem in SEP_STEM_FILES:
        sep_path = sep_dir / sep_stem
        if not sep_path.exists():
            return False
    
    return True


def compute_sisdr(ref, est, metric):
    """Compute SI-SDR between reference and estimated audio tensors."""
    # Ensure tensors are 2D: [channels, samples]
    if ref.ndim == 1:
        ref = ref.unsqueeze(0)
    if est.ndim == 1:
        est = est.unsqueeze(0)
    
    # Align lengths
    min_len = min(ref.shape[-1], est.shape[-1])
    ref = ref[..., :min_len]
    est = est[..., :min_len]
    
    with torch.no_grad():
        score = metric(est, ref)
    return float(score)


def evaluate_sisdr(musdb_test: Path, out_dir: Path, per_track_stats: List[Dict]) -> tuple:
    """
    Evaluate SI-SDR scores for all separated tracks.
    
    Returns:
        tuple: (tm_scores dict, per_track_sisdr list)
    """
    print("\nEvaluating SI-SDR...")
    sisdr_metric = ScaleInvariantSignalDistortionRatio()
    tm_scores = {stem: [] for stem in STEM_NAMES}
    per_track_sisdr = []
    
    for track_stat in per_track_stats:
        rel_dir = Path(track_stat['track'])
        gt_dir = musdb_test / rel_dir
        sep_dir = out_dir / rel_dir
        track_sisdr = {}
        
        for gt_stem, sep_stem, stem_name in zip(GT_STEM_FILES, SEP_STEM_FILES, STEM_NAMES):
            gt_path = gt_dir / gt_stem
            sep_path = sep_dir / sep_stem
            
            if not gt_path.exists() or not sep_path.exists():
                continue
            
            # Load audio using torchaudio
            gt_audio, sr_gt = torchaudio.load(str(gt_path))
            sep_audio, sr_sep = torchaudio.load(str(sep_path))
            
            if sr_gt != sr_sep:
                print(f'  Sample rate mismatch: {gt_stem} (gt: {sr_gt}, sep: {sr_sep})')
                continue
            
            sisdr = compute_sisdr(gt_audio, sep_audio, sisdr_metric)
            tm_scores[stem_name].append(sisdr)
            track_sisdr[stem_name] = sisdr
        
        per_track_sisdr.append({'track': str(rel_dir), 'sisdr': track_sisdr})
    
    return tm_scores, per_track_sisdr


def print_timing_summary(total_start: float, total_end: float, total_proc_sec: float, 
                        total_audio_sec: float, separated_count: int, skipped_count: int, 
                        total_tracks: int):
    """Print timing summary statistics."""
    print(f"\n{'='*60}")
    print("Separation Summary:")
    print(f"  Total tracks: {total_tracks}")
    print(f"  Separated: {separated_count}")
    print(f"  Skipped (already existed): {skipped_count}")
    print(f"  Wall time: {total_end - total_start:.2f} seconds")
    print(f"  Processing time: {total_proc_sec:.2f} seconds")
    print(f"  Total audio: {total_audio_sec/60:.2f} minutes")
    if total_proc_sec > 0 and total_audio_sec > 0:
        avg_time_per_min = total_proc_sec / (total_audio_sec/60)
        print(f"  Average time per minute (separated tracks): {avg_time_per_min:.2f} seconds")
    print(f"{'='*60}")


def create_summary(model_identifier: str, total_tracks: int, total_proc_sec: float, 
                  total_wall_time_sec: float, total_audio_sec: float, tm_scores: Dict) -> Dict:
    """Create summary statistics dictionary."""
    return {
        'model_identifier': model_identifier,
        'total_tracks': total_tracks,
        'total_proc_sec': total_proc_sec,
        'total_wall_time_sec': total_wall_time_sec,
        'total_audio_sec': total_audio_sec,
        'avg_time_per_min': total_proc_sec / (total_audio_sec/60) if total_audio_sec > 0 else None,
        'sisdr_mean_per_stem': {stem: float(np.mean(scores)) if scores else None for stem, scores in tm_scores.items()},
        'overall_sisdr_mean': float(np.mean([s for scores in tm_scores.values() for s in scores])) if any(tm_scores.values()) else None
    }


def save_results(results: Dict, json_path: Path):
    """Save benchmark results to JSON file and print summary."""
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    summary = results['summary']
    print(f'\nBenchmark results written to {json_path}')
    print(f"Overall SI-SDR: {summary['overall_sisdr_mean']:.2f} dB" if summary['overall_sisdr_mean'] else "Overall SI-SDR: N/A")


def run_benchmark(musdb_root: Path, out_dir: Path, force_reseparate: bool, 
                 json_out: str, separate_func: Callable, model_identifier: str, 
                 **separate_kwargs) -> Dict:
    """
    Main benchmarking flow that works with any separation backend.
    
    Args:
        musdb_root: Path to MusDB dataset root
        out_dir: Output directory for separated tracks
        force_reseparate: Whether to force re-separation
        json_out: JSON output filename
        separate_func: Function to call for separation (backend-specific)
        model_identifier: String identifier for the model/backend used
        **separate_kwargs: Additional arguments to pass to separate_func
    
    Returns:
        Dictionary with benchmark results
    """
    # Setup paths
    musdb_test = musdb_root / 'test'
    mixture_files = find_mixture_wavs(musdb_test)
    print(f'Found {len(mixture_files)} mixture.wav files.')
    print(f'Output directory: {out_dir}')
    
    if not mixture_files:
        print("No mixture.wav files found. Check your --musdb-root path.")
        return {}

    # Separation with detailed timing
    total_start = time.time()
    total_proc_sec = 0.0
    total_audio_sec = 0.0
    per_track_stats = []
    
    skipped_count = 0
    separated_count = 0
    
    for mixture_path in mixture_files:
        rel_dir = mixture_path.parent.relative_to(musdb_test)
        track_out_dir = out_dir / rel_dir
        
        # Check if track is already separated
        if not force_reseparate and check_track_separated(track_out_dir):
            print(f'Skipping (already separated): {mixture_path}')
            skipped_count += 1
            
            # Still get audio duration for stats
            audio_info = torchaudio.info(str(mixture_path))
            duration_sec = audio_info.num_frames / audio_info.sample_rate
            total_audio_sec += duration_sec
            per_track_stats.append({
                'track': str(rel_dir),
                'proc_time_sec': 0.0,  # Already separated
                'audio_duration_sec': duration_sec,
                'skipped': True
            })
            continue
        
        track_out_dir.mkdir(parents=True, exist_ok=True)
        print(f'Separating: {mixture_path}\n -> {track_out_dir}')
        
        start = time.time()
        try:
            separate_func(mixture_path, track_out_dir, **separate_kwargs)
            separated_count += 1
        except Exception as e:
            print(f"Error processing {mixture_path}: {e}")
            continue
        end = time.time()
        
        # Get audio duration from file
        audio_info = torchaudio.info(str(mixture_path))
        duration_sec = audio_info.num_frames / audio_info.sample_rate
        proc_time = end - start
        print(f"  Track processed in {proc_time:.2f} seconds. Length: {duration_sec/60:.2f} min.")
        
        total_proc_sec += proc_time
        total_audio_sec += duration_sec
        per_track_stats.append({
            'track': str(rel_dir),
            'proc_time_sec': proc_time,
            'audio_duration_sec': duration_sec,
            'skipped': False
        })
    
    total_end = time.time()
    
    # Print timing summary
    print_timing_summary(total_start, total_end, total_proc_sec, total_audio_sec, 
                         separated_count, skipped_count, len(mixture_files))
    
    # SI-SDR Evaluation
    tm_scores, per_track_sisdr = evaluate_sisdr(musdb_test, out_dir, per_track_stats)
    
    # Create summary
    summary = create_summary(model_identifier, len(mixture_files), total_proc_sec, 
                           total_end - total_start, total_audio_sec, tm_scores)
    
    # Prepare results
    results = {
        'summary': summary,
        'per_track_stats': per_track_stats,
        'per_track_sisdr': per_track_sisdr
    }
    
    # Save results
    script_dir = Path(__file__).parent if __file__ else Path('.')
    json_path = script_dir / json_out
    save_results(results, json_path)
    
    return results