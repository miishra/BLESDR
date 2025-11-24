#!/usr/bin/env python3
"""
Robust peak finder using percentile-based thresholds
Works even with noisy/varying energy distributions
"""

import numpy as np
import matplotlib.pyplot as plt

print("Loading input I/Q file...")
iq = np.fromfile('ble_20pkts_cfo+50k.f32', dtype=np.float32)
iq_complex = iq[::2] + 1j * iq[1::2]

print(f"Total complex samples: {len(iq_complex)}")

# Compute energy
energy = np.abs(iq_complex)**2

print(f"\nEnergy statistics:")
print(f"  Mean: {np.mean(energy):.9f}")
print(f"  Median: {np.median(energy):.9f}")
print(f"  Std: {np.std(energy):.9f}")
print(f"  Max: {np.max(energy):.9f}")
print(f"  Min: {np.min(energy):.9f}")

# Percentile-based thresholding (more robust)
p90 = np.percentile(energy, 90)
p95 = np.percentile(energy, 95)
p99 = np.percentile(energy, 99)

print(f"\nPercentiles:")
print(f"  90th: {p90:.9f}")
print(f"  95th: {p95:.9f}")
print(f"  99th: {p99:.9f}")

# Try multiple threshold strategies
print(f"\n{'='*80}")
print("TRYING MULTIPLE DETECTION STRATEGIES:")
print(f"{'='*80}")

strategies = [
    ("90th percentile", p90),
    ("95th percentile", p95),
    ("99th percentile", p99),
    ("Mean + 1σ", np.mean(energy) + 1*np.std(energy)),
    ("Mean + 2σ", np.mean(energy) + 2*np.std(energy)),
    ("Fixed 0.3", 0.3),
    ("Fixed 0.5", 0.5),
]

best_bursts = None
best_count = 0
best_name = ""

for name, threshold in strategies:
    above = energy > threshold
    bursts = []
    in_burst = False
    start = 0
    
    for i in range(len(above)):
        if above[i] and not in_burst:
            start = i
            in_burst = True
        elif not above[i] and in_burst:
            if i - start > 50:  # Min 50 samples
                bursts.append((start, i))
            in_burst = False
    
    print(f"  {name:20s} (thr={threshold:.3f}): {len(bursts):2d} bursts")
    
    # Prefer strategy that finds ~20 packets
    if 15 <= len(bursts) <= 25 and len(bursts) > best_count:
        best_bursts = bursts
        best_count = len(bursts)
        best_name = name

if best_bursts is None:
    print("\n⚠️  No good strategy found, using 95th percentile...")
    threshold = p95
    above = energy > threshold
    best_bursts = []
    in_burst = False
    start = 0
    for i in range(len(above)):
        if above[i] and not in_burst:
            start = i
            in_burst = True
        elif not above[i] and in_burst:
            if i - start > 50:
                best_bursts.append((start, i))
            in_burst = False
    best_name = "95th percentile (fallback)"
    threshold = p95
else:
    threshold = [t for n, t in strategies if n == best_name][0]

print(f"\n✅ Using: {best_name}, threshold={threshold:.3f}")
print(f"   Found {len(best_bursts)} bursts\n")

# Show first 20 bursts
print(f"{'='*80}")
print("DETECTED ENERGY BURSTS (likely packets):")
print(f"{'='*80}")
print(f"{'Idx':<4} {'Start':>6} {'End':>6} {'Span':>5} {'Peak Energy':>12}")
print("-"*80)

for i, (start, end) in enumerate(best_bursts[:20]):
    span = end - start
    peak_e = np.max(energy[start:end])
    print(f"{i:<4d} {start:6d} {end:6d} {span:5d} {peak_e:12.6f}")

# Decoder indices (your output after timing fix)
decoder_indices = [
    (540, 1004),
    (5012, 5476),
    (9484, 9948),
    (13956, 14420),
    (18428, 18892),
    (22900, 23364),
    (27372, 27836),
    (31844, 32308),
    (36316, 36780),
    (40788, 41252),
    (45260, 45724),
    (49732, 50196),
    (54204, 54668),
    (58676, 59140),
    (63148, 63612),
    (67620, 68084),
    (72092, 72556),
    (76564, 77028),
    (81036, 81500),
    (85508, 85972),
]

# Compare
if len(best_bursts) >= 5:
    print(f"\n{'='*80}")
    print("COMPARISON: Actual Energy vs Decoder Indices")
    print(f"{'='*80}")
    print(f"{'Pkt':<4} {'Energy Burst':^20} {'Decoder':^20} {'Offset':>10}")
    print("-"*80)
    
    offsets = []
    for i in range(min(len(best_bursts), len(decoder_indices))):
        act_start, act_end = best_bursts[i]
        dec_start, dec_end = decoder_indices[i]
        
        offset_start = dec_start - act_start
        offsets.append(offset_start)
        
        print(f"{i:<4d} [{act_start:5d}, {act_end:5d}) "
              f"[{dec_start:5d}, {dec_end:5d}) "
              f"{offset_start:+6d} ({offset_start/464:+.2f}×)")
    
    print("-"*80)
    avg_offset = np.mean(offsets)
    std_offset = np.std(offsets)
    print(f"Average offset: {avg_offset:.1f} ± {std_offset:.1f} samples")
    print(f"This is {avg_offset/464:.2f}× the packet length")
    print(f"{'='*80}")
    
    if abs(avg_offset) < 50:
        print("✅ GOOD: Decoder timing is accurate (< 50 sample offset)")
    else:
        print(f"❌ BAD: Decoder is {avg_offset:.0f} samples off")
        print(f"   Need to adjust abs_cursor calculation by {-avg_offset:.0f} samples")

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Full energy profile
plot_len = min(10000, len(energy))
axes[0].plot(energy[:plot_len], 'b-', linewidth=0.5, alpha=0.7)
axes[0].axhline(y=threshold, color='orange', linestyle='--', linewidth=1.5, 
                label=f'Threshold = {threshold:.3f}')

# Mark detected bursts
for i, (start, end) in enumerate(best_bursts):
    if end <= plot_len:
        axes[0].axvspan(start, end, alpha=0.25, color='green')
        axes[0].axvline(start, color='green', linestyle='-', linewidth=1)
        if i < 5:
            axes[0].text(start, energy[start:end].max(), f'{i}', 
                        fontsize=9, color='green', fontweight='bold')

# Mark decoder indices
for i, (start, end) in enumerate(decoder_indices):
    if end <= plot_len:
        axes[0].axvspan(start, end, alpha=0.15, color='red')
        axes[0].axvline(start, color='red', linestyle=':', linewidth=2)

axes[0].set_ylabel('Energy')
axes[0].set_title(f'Energy Profile - {best_name}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Zoom on first 3 packets
zoom_len = 7000
axes[1].plot(energy[:zoom_len], 'b-', linewidth=1)
axes[1].axhline(y=threshold, color='orange', linestyle='--', alpha=0.5)

for i, (start, end) in enumerate(best_bursts[:3]):
    if end <= zoom_len:
        axes[1].axvspan(start, end, alpha=0.3, color='green', label='Energy' if i==0 else '')
        axes[1].axvline(start, color='green', linestyle='-', linewidth=2)

for i, (start, end) in enumerate(decoder_indices[:3]):
    if end <= zoom_len:
        axes[1].axvspan(start, end, alpha=0.2, color='red', label='Decoder' if i==0 else '')
        axes[1].axvline(start, color='red', linestyle=':', linewidth=2)

axes[1].set_ylabel('Energy')
axes[1].set_xlabel('Sample Index')
axes[1].set_title('First 3 Packets - Energy vs Decoder')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Histogram
axes[2].hist(np.log10(energy[energy > 1e-9]), bins=100, alpha=0.7, color='blue')
axes[2].axvline(np.log10(threshold), color='red', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold:.3f}')
axes[2].set_xlabel('log10(Energy)')
axes[2].set_ylabel('Count')
axes[2].set_title('Energy Distribution (log scale)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('packet_detection_robust.png', dpi=150)
print(f"\n💾 Saved: packet_detection_robust.png")