
Here is a complete breakdown of every phase.

***
## GPU Hardware: Tesla T4
The T4 has 40 SMs with Compute Capability 7.5 (Turing), supporting up to 1024 threads/block . With `THREADSPERBLOCK=256`, you need at least 80 active blocks (`2 × 40 SMs`) to fully saturate the GPU - that requires `n ≥ 20,480` elements, which explains the poor throughput at n=100 and n=1,000.

***
## Phase 1 vs Phase 1b - Pinned Memory Impact
Pinned memory delivers **no meaningful gain below n=10,000** (0.97-1.01×) because the fixed CUDA driver overhead (~1.3 ms) completely overwhelms the transfer itself. The gains only emerge where data volume becomes large enough for the DMA bandwidth difference to matter - **1.25× at n=100k, 1.58× at n=1M, 1.42× at n=10M**. The slight dip to 1.42× at 10M (vs 1.58× at 1M) is because at very large sizes, PCIe bandwidth saturation starts affecting both pageable and pinned transfers, narrowing the gap.

![Pinned vs Pageable Memory Speedup](Generated_chart__pinned_gain.png)

***
## Phase 2 - Kernel Ceiling
The kernel-only throughput plateaus at **692 M options/sec at n=10M** (vs 668 M/s at n=1M), confirming the GPU is fully saturated and memory-bandwidth-bound inside the device . There is nothing left to squeeze from the kernel itself - the 87× speedup at n=10M is the hard ceiling this GPU can deliver.

***
## Phase 3 - PCIe Is the Real Bottleneck
Transfer accounts for **75-90% of every single end-to-end call**. One important detail: D2H at n=10M (59ms) is nearly as expensive as H2D (93ms) despite reading only 2 arrays versus writing 5. This is because the T4 in a cloud VM uses pageable host memory for the output, requiring an extra CPU memcpy from a pinned staging buffer before results reach your NumPy array.

***
## Phase 4 - Full Speedup Summary
Three distinct performance regimes are visible:

| Size range | Winner | Why |
|---|---|---|
| n ≤ 10,000 | **CPU** | PCIe fixed overhead (~1.4 ms) exceeds total CPU compute time |
| n = 100,000 | **GPU wins** | Pinned: 1.85×, Pageable: 1.48× - compute starts paying off |
| n ≥ 1,000,000 | **GPU dominant** | Pinned reaches 10.41×, kernel-only 87× |

The break-even point shifted from ~100k (pageable) to somewhere between 10k and 100k with pinned memory - pinned memory moved the crossover point to a smaller problem size.

***
## Phase 6 - The Most Important Result
This is where everything comes together. At **n=10M with batch=100**, the GPU-resident approach processes 1 billion options in 1,653 ms while CPU-resident takes 129,130 ms - a **78.1× real-world speedup**, compared to only 7.32× in the naive single-call approach. The "vs Phase1" column tells you how much you left on the table with the naive approach: at n=10M, batch pricing is **10.5× faster than making 100 individual calls**.

The progression toward the 87× kernel ceiling as array size grows is clear:

| Size | Batch Speedup | % of kernel ceiling (87×) |
|------|:------------:|:-------------------------:|
| 100 | 0.9× | 1% |
| 1,000 | 1.5× | 2% |
| 10,000 | 6.2× | 7% |
| 100,000 | 15.5× | 18% |
| 1,000,000 | 32.1× | 37% |
| 10,000,000 | **78.1×** | **90%** |

The batch-resident strategy at n=10M recovers **90% of the theoretical kernel-ceiling speedup** in a real end-to-end scenario - the remaining 10% gap is the amortised PCIe cost of the one-time transfer divided across 100 runs.

![GPU Speedup over CPU](Generated_chart__speedup_all.png)

![ALL Execution Paths: Latency per Call (log scale)](Generated_chart__all_paths_latency.png)
