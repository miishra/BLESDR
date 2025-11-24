#!/usr/bin/env python3
"""
Analyze saved I/Q chunks from BLE packet captures
Verifies alignment with actual packet boundaries
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re

def load_iq_chunk(filepath):
    """Load complex I/Q data from binary file"""
    data = np.fromfile(filepath, dtype=np.float32)
    # Reshape to complex (I,Q pairs)
    iq = data[::2] + 1j * data[1::2]
    return iq

def parse_filename(filename):
    """Extract packet info from filename"""
    # Format: ble_pkt_000000_ch37_541_1005.dat
    match = re.match(r'ble_pkt_(\d+)_ch(\d+)_(\d+)_(\d+)\.dat', filename)
    if match:
        return {
            'pkt_idx': int(match.group(1)),
            'channel': int(match.group(2)),
            'start_idx': int(match.group(3)),
            'end_idx': int(match.group(4)),
            'span': int(match.group(4)) - int(match.group(3))
        }
    return None

def compute_energy(iq):
    """Compute instantaneous energy"""
    return np.abs(iq)**2

def detect_preamble_candidate(iq, sps=2):
    """
    Detect preamble-like patterns (alternating 0/1 bits)
    BLE preamble is 0xAA (10101010) or 0x55 (01010101)
    """
    # Discriminator (phase difference)
    phase_diff = np.angle(iq[1:] * np.conj(iq[:-1]))
    
    # For preamble, we expect alternating phase jumps
    # Look for sections with consistent alternation
    
    # Downsample to symbol rate (assume 2 samples/symbol)
    symbols = []
    for i in range(0, len(phase_diff), sps):
        if i+sps <= len(phase_diff):
            sym_avg = np.mean(phase_diff[i:i+sps])
            symbols.append(1 if sym_avg > 0 else -1)
    
    if len(symbols) < 8:
        return 0
    
    # Count transitions in first 8 symbols (preamble)
    transitions = 0
    for i in range(7):
        if symbols[i] != symbols[i+1]:
            transitions += 1
    
    # Perfect alternation = 7 transitions
    # Allow 5-7 for some noise tolerance
    preamble_score = transitions / 7.0
    return preamble_score

def analyze_chunk(filepath, verbose=True):
    """Analyze a single I/Q chunk"""
    info = parse_filename(filepath.name)
    if info is None:
        print(f"Warning: Could not parse filename: {filepath.name}")
        return None
    
    iq = load_iq_chunk(filepath)
    
    if len(iq) == 0:
        print(f"Warning: Empty file: {filepath.name}")
        return None
    
    # Compute metrics
    energy = compute_energy(iq)
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)
    peak_energy = np.max(energy)
    
    # Energy rise time (10% to 90%)
    sorted_e = np.sort(energy)
    e_10 = sorted_e[int(0.1 * len(sorted_e))]
    e_90 = sorted_e[int(0.9 * len(sorted_e))]
    
    rise_start = np.where(energy >= e_10)[0][0] if np.any(energy >= e_10) else 0
    rise_end = np.where(energy >= e_90)[0][0] if np.any(energy >= e_90) else len(energy)-1
    rise_samples = rise_end - rise_start
    
    # Preamble detection
    preamble_score = detect_preamble_candidate(iq, sps=2)
    
    # Check if energy starts immediately (good alignment)
    # First 10% of samples should have significant energy
    early_energy = np.mean(energy[:len(energy)//10])
    late_energy = np.mean(energy[-len(energy)//10:])
    energy_ratio = early_energy / (late_energy + 1e-12)
    
    results = {
        **info,
        'n_samples': len(iq),
        'mean_energy': mean_energy,
        'std_energy': std_energy,
        'peak_energy': peak_energy,
        'snr_db': 10*np.log10(peak_energy / (mean_energy + 1e-12)),
        'rise_samples': rise_samples,
        'preamble_score': preamble_score,
        'energy_ratio': energy_ratio,
        'good_alignment': (rise_samples < len(energy)//4) and (energy_ratio > 0.5)
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"File: {filepath.name}")
        print(f"  Packet index: {info['pkt_idx']}")
        print(f"  Channel: {info['channel']}")
        print(f"  Sample indices: [{info['start_idx']}, {info['end_idx']})")
        print(f"  Span: {info['span']} samples")
        print(f"  Actual samples: {len(iq)}")
        print(f"  Mean energy: {mean_energy:.6f}")
        print(f"  Peak energy: {peak_energy:.6f}")
        print(f"  SNR: {results['snr_db']:.1f} dB")
        print(f"  Rise time: {rise_samples} samples")
        print(f"  Preamble score: {preamble_score:.2f} (1.0 = perfect alternation)")
        print(f"  Energy ratio (early/late): {energy_ratio:.2f}")
        print(f"  Alignment: {'GOOD' if results['good_alignment'] else 'POOR'}")
    
    return results

def plot_chunk(filepath, save_path=None):
    """Plot I/Q chunk with energy and phase"""
    info = parse_filename(filepath.name)
    if info is None:
        return
    
    iq = load_iq_chunk(filepath)
    if len(iq) == 0:
        return
    
    energy = compute_energy(iq)
    phase = np.angle(iq)
    phase_diff = np.angle(iq[1:] * np.conj(iq[:-1]))
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # I/Q constellation
    axes[0].plot(iq.real, iq.imag, 'b.', alpha=0.3, markersize=2)
    axes[0].set_xlabel('I')
    axes[0].set_ylabel('Q')
    axes[0].set_title(f'Packet {info["pkt_idx"]} - I/Q Constellation')
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Energy over time
    axes[1].plot(energy, 'r-', linewidth=0.5)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Energy')
    axes[1].set_title('Instantaneous Energy')
    axes[1].grid(True, alpha=0.3)
    
    # Phase over time
    axes[2].plot(phase, 'g-', linewidth=0.5)
    axes[2].set_xlabel('Sample')
    axes[2].set_ylabel('Phase (rad)')
    axes[2].set_title('Carrier Phase')
    axes[2].grid(True, alpha=0.3)
    
    # Phase difference (discriminator)
    axes[3].plot(phase_diff, 'b-', linewidth=0.5)
    axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[3].set_xlabel('Sample')
    axes[3].set_ylabel('Phase Diff (rad)')
    axes[3].set_title('Discriminator Output (for GFSK demodulation)')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {save_path}")
    else:
        plt.show()
    
    plt.close()

def analyze_directory(dir_path, plot_first_n=5):
    """Analyze all I/Q chunks in a directory"""
    dir_path = Path(dir_path)
    
    if not dir_path.exists():
        print(f"Error: Directory not found: {dir_path}")
        return
    
    # Find all .dat files
    files = sorted(dir_path.glob('ble_pkt_*.dat'))
    
    if len(files) == 0:
        print(f"No I/Q chunk files found in {dir_path}")
        return
    
    print(f"Found {len(files)} I/Q chunk files")
    
    # Analyze all
    results = []
    for i, filepath in enumerate(files):
        result = analyze_chunk(filepath, verbose=(i < 20))  # Verbose for first 20
        if result:
            results.append(result)
        
        # Plot first few
        if i < plot_first_n:
            plot_path = dir_path / f"plot_{filepath.stem}.png"
            plot_chunk(filepath, save_path=plot_path)
    
    # Summary statistics
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS (all packets)")
        print(f"{'='*60}")
        print(f"Total packets analyzed: {len(results)}")
        
        good_alignment = sum(1 for r in results if r['good_alignment'])
        print(f"Good alignment: {good_alignment}/{len(results)} ({100*good_alignment/len(results):.1f}%)")
        
        avg_preamble = np.mean([r['preamble_score'] for r in results])
        print(f"Average preamble score: {avg_preamble:.3f}")
        
        avg_snr = np.mean([r['snr_db'] for r in results])
        print(f"Average SNR: {avg_snr:.1f} dB")
        
        avg_rise = np.mean([r['rise_samples'] for r in results])
        print(f"Average rise time: {avg_rise:.1f} samples")
        
        # Check consistency
        expected_spans = [r['span'] for r in results]
        actual_samples = [r['n_samples'] for r in results]
        
        mismatches = sum(1 for e, a in zip(expected_spans, actual_samples) if e != a)
        if mismatches > 0:
            print(f"\nWARNING: {mismatches} files have span != actual samples")
        
        print(f"\nSample count range: {min(actual_samples)} to {max(actual_samples)}")
        print(f"Expected span range: {min(expected_spans)} to {max(expected_spans)}")
        
        # List poor alignment cases
        poor = [r for r in results if not r['good_alignment']]
        if poor:
            print(f"\nPackets with poor alignment:")
            for r in poor[:10]:  # Show first 10
                print(f"  Packet {r['pkt_idx']:06d}: rise={r['rise_samples']}, "
                      f"preamble={r['preamble_score']:.2f}, ratio={r['energy_ratio']:.2f}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_iq_chunks.py <directory_with_iq_chunks> [num_plots]")
        print("Example: python analyze_iq_chunks.py ./iq_chunks 10")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    num_plots = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    analyze_directory(dir_path, plot_first_n=num_plots)
