# Sprint 23 Capability Benchmark Report

Sprint 23 introduced three new capability benchmarks demonstrating prxteinmpnn's architectural advantages over PyTorch and ColabDesign baselines.

---

## DedupGather Heterogeneous Batch

K unique structures deduplicated before scoring; N=32 total. prxteinmpnn scores only unique structures and scatters results, while ColabDesign and PyTorch score all N.

### H200

| K unique | dedup_ratio | prxteinmpnn (ms) | ColabDesign (ms) | PyTorch (ms) | Speedup vs CD | Speedup vs PT |
|---|---|---|---|---|---|---|
| 1 | 0.0312 | 1.14 | 6.95 | 92.18 | 6.1× | 80.6× |
| 2 | 0.0625 | 2.24 | 15.87 | 183.10 | 7.1× | 81.8× |
| 4 | 0.1250 | 4.52 | 31.69 | 368.37 | 7.0× | 81.4× |
| 8 | 0.2500 | 8.95 | 63.37 | 734.51 | 7.1× | 82.1× |
| 16 | 0.5000 | 17.84 | 126.61 | 1475.58 | 7.1× | 82.7× |
| 32 | 1.0000 | 36.03 | 254.80 | 2957.74 | 7.1× | 82.1× |

### A100

| K unique | dedup_ratio | prxteinmpnn (ms) | ColabDesign (ms) | PyTorch (ms) | Speedup vs CD | Speedup vs PT |
|---|---|---|---|---|---|---|
| 1 | 0.0312 | 1.18 | 6.73 | 98.94 | 5.7× | 83.7× |
| 2 | 0.0625 | 2.16 | 13.30 | 194.64 | 6.2× | 90.2× |
| 4 | 0.1250 | 4.34 | 26.71 | 391.30 | 6.2× | 90.1× |
| 8 | 0.2500 | 8.68 | 53.75 | 785.61 | 6.2× | 90.5× |
| 16 | 0.5000 | 17.36 | 106.66 | 1582.01 | 6.1× | 91.2× |
| 32 | 1.0000 | 34.98 | 211.28 | 3151.99 | 6.0× | 90.1× |

### L40s

| K unique | dedup_ratio | prxteinmpnn (ms) | ColabDesign (ms) | PyTorch (ms) | Speedup vs CD | Speedup vs PT |
|---|---|---|---|---|---|---|
| 1 | 0.0312 | 1.14 | 6.60 | 88.69 | 5.8× | 78.1× |
| 2 | 0.0625 | 2.11 | 13.15 | 176.04 | 6.2× | 83.2× |
| 4 | 0.1250 | 4.21 | 26.11 | 353.96 | 6.2× | 84.0× |
| 8 | 0.2500 | 8.43 | 52.47 | 711.31 | 6.2× | 84.4× |
| 16 | 0.5000 | 17.07 | 104.95 | 1433.51 | 6.1× | 84.0× |
| 32 | 1.0000 | 34.28 | 210.53 | 2867.54 | 6.1× | 83.7× |

### Blackwell (SM120)

| K unique | dedup_ratio | prxteinmpnn (ms) | ColabDesign (ms) | PyTorch (ms) | Speedup vs CD | Speedup vs PT |
|---|---|---|---|---|---|---|
| 1 | 0.0312 | 0.90 | 5.59 | 70.42 | 6.2× | 78.0× |
| 2 | 0.0625 | 1.72 | 12.38 | 140.12 | 7.2× | 81.5× |
| 4 | 0.1250 | 3.35 | 24.80 | 280.48 | 7.4× | 83.7× |
| 8 | 0.2500 | 6.75 | 46.95 | 562.93 | 7.0× | 83.4× |
| 16 | 0.5000 | 13.50 | 93.06 | 1136.69 | 6.9× | 84.2× |
| 32 | 1.0000 | 26.76 | 186.22 | 2255.70 | 7.0× | 84.3× |

### Cross-Hardware Summary (K=1 and K=32)

| Hardware | K | prxteinmpnn (ms) | PyTorch (ms) | Speedup vs PT |
|---|---|---|---|---|
| H200 | 1 | 1.14 | 92.18 | 80.6× |
| H200 | 32 | 36.03 | 2957.74 | 82.1× |
| A100 | 1 | 1.18 | 98.94 | 83.7× |
| A100 | 32 | 34.98 | 3151.99 | 90.1× |
| L40s | 1 | 1.14 | 88.69 | 78.1× |
| L40s | 32 | 34.28 | 2867.54 | 83.7× |
| Blackwell (SM120) | 1 | 0.90 | 70.42 | 78.0× |
| Blackwell (SM120) | 32 | 26.76 | 2255.70 | 84.3× |

---

## Mixed-Length Heterogeneous Batch

Batch of 4 sequences with lengths [76, 150, 300, 500]. prxteinmpnn packs into a single padded batch; ColabDesign and PyTorch run sequentially.

| Hardware | batch_lengths | prxteinmpnn (ms) | ColabDesign seq (ms) | PyTorch padded (ms) | PyTorch seq (ms) | Speedup vs PT padded | Speedup vs PT seq |
|---|---|---|---|---|---|---|---|
| H200 | [76, 150, 300, 500] | 4.12 | 94.18 | 2279.99 | 1327.51 | 553.5× | 322.3× |
| A100 | [76, 150, 300, 500] | 4.08 | 91.16 | 2282.06 | 1296.13 | 558.9× | 317.4× |
| L40s | [76, 150, 300, 500] | 4.16 | 89.74 | 2629.93 | 1532.34 | 632.6× | 368.6× |
| Blackwell (SM120) | [76, 150, 300, 500] | 3.75 | 77.44 | 1757.84 | 1016.76 | 469.2× | 271.4× |

---

## Temperature Array Sweep

M simultaneous temperatures JIT-compiled into a single forward pass. ColabDesign and PyTorch baselines were not run in this sprint (they require M sequential calls; dedup section covers that comparison).

### Task: Autoregressive Sample

Latency is per-temperature (total / M), seq_len=76, batch=1.

| Hardware | M | prxteinmpnn per-temp (ms) | prxteinmpnn total (ms) |
|---|---|---|---|
| H200 | 1 | 17.11 | 17.11 |
| H200 | 2 | 8.58 | 17.17 |
| H200 | 4 | 4.28 | 17.10 |
| H200 | 8 | 2.15 | 17.18 |
| A100 | 1 | 17.33 | 17.33 |
| A100 | 2 | 8.61 | 17.23 |
| A100 | 4 | 4.32 | 17.26 |
| A100 | 8 | 2.15 | 17.22 |
| L40s | 1 | 17.18 | 17.18 |
| L40s | 2 | 8.62 | 17.23 |
| L40s | 4 | 4.29 | 17.15 |
| L40s | 8 | 2.15 | 17.23 |
| Blackwell (SM120) | 1 | 15.57 | 15.57 |
| Blackwell (SM120) | 2 | 7.36 | 14.72 |
| Blackwell (SM120) | 4 | 3.68 | 14.71 |
| Blackwell (SM120) | 8 | 1.84 | 14.71 |

### Task: Score Conditional

Latency is per-temperature (total / M), seq_len=76, batch=1.

| Hardware | M | prxteinmpnn per-temp (ms) | prxteinmpnn total (ms) |
|---|---|---|---|
| H200 | 1 | 1.51 | 1.51 |
| H200 | 2 | 0.74 | 1.49 |
| H200 | 4 | 0.38 | 1.51 |
| H200 | 8 | 0.18 | 1.47 |
| A100 | 1 | 1.55 | 1.55 |
| A100 | 2 | 0.76 | 1.52 |
| A100 | 4 | 0.38 | 1.51 |
| A100 | 8 | 0.19 | 1.52 |
| L40s | 1 | 1.54 | 1.54 |
| L40s | 2 | 0.76 | 1.51 |
| L40s | 4 | 0.37 | 1.48 |
| L40s | 8 | 0.19 | 1.51 |
| Blackwell (SM120) | 1 | 1.11 | 1.11 |
| Blackwell (SM120) | 2 | 0.54 | 1.08 |
| Blackwell (SM120) | 4 | 0.26 | 1.05 |
| Blackwell (SM120) | 8 | 0.13 | 1.05 |

### M-Scaling Efficiency (ar_sample, seq_len=76, batch=1)

Ideal scaling: per-temp latency is constant as M increases (M temperatures compiled as a single JIT call).

| Hardware | M=1 (ms) | M=2 (ms) | M=4 (ms) | M=8 (ms) | Speedup per-temp at M=8 (ideal: 8.0×) |
|---|---|---|---|---|---|
| H200 | 17.11 | 8.58 | 4.28 | 2.15 | 7.97× |
| A100 | 17.33 | 8.61 | 4.32 | 2.15 | 8.05× |
| L40s | 17.18 | 8.62 | 4.29 | 2.15 | 7.98× |
| Blackwell (SM120) | 15.57 | 7.36 | 3.68 | 1.84 | 8.47× |

