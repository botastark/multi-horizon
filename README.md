# Multi-Horizon Active Sensing

Multi-agent active sensing system with decentralized planning and information sharing.

## Quick Start

### Running Benchmarks

#### Greedy IG Benchmark

Run the greedy information gain benchmark with various news modes (IG, IG+BS, IG+BM, IGd, IGd+BS, IGd+BM):

```bash
python src/main.py --config configs/benchmark_greedy_ig.json
```

This will run experiments for all configured news modes and save results to `trials/greedy_ig_gaussian_r5_corner_N8_<NEWS_MODE>_commR<MULTIPLIER>/`.

**Communication Range Configuration:**

The communication range is controlled by `radius_multiplier` in the config file:
- `radius_multiplier: 5` → 5 × h_displacement = 5 × 3.125m = 15.625m
- `radius_multiplier: -1` → Unlimited range

where `h_displacement = (field_len/2) / n_h_act` and `n_h_act = 8` for 8 agents, otherwise 5.

### Plotting Results

#### Plot Greedy IG with Limited Range (R=5)

```bash
python plotter.py --compare-news --paths trials/*_commR5/txt/ --radius 5 --comm-range 5
```

This generates `plots/greedy_ig_info_sharing_comparison_r5_commR5_N8.png` showing:
- Left column: IG, IG+BS, IG+BM
- Right column: IGd, IGd+BS, IGd+BM
- Title: "Greedy Ig - Effect of information sharing... (Comm Range = 5×3.125m = 15.6m)"

#### Plot Greedy IG with Unlimited Range

```bash
python plotter.py --compare-news --paths trials/*_commRinf/txt/ --radius 5 --comm-range -1
```

This generates `plots/greedy_ig_info_sharing_comparison_r5_commRinf_N8.png` with unlimited communication range.

#### Compare Both Communication Ranges

Generate both plots to compare limited vs unlimited range:

```bash
# Limited range (15.625m)
python plotter.py --compare-news --paths trials/*_commR5/txt/ --radius 5 --comm-range 5

# Unlimited range
python plotter.py --compare-news --paths trials/*_commRinf/txt/ --radius 5 --comm-range -1
```

## Configuration

- **Config files**: `configs/benchmark_greedy_ig.json`
- **Results directory**: `trials/`
- **Plots directory**: `plots/`

## System Parameters

- Grid: 400×400 cells, 50m×50m field, cell_size=0.125m
- Agents: N=8, corner start positions
- Gaussian field radius: 5m (for map generation)
- Communication range: Calculated from radius_multiplier × h_displacement
