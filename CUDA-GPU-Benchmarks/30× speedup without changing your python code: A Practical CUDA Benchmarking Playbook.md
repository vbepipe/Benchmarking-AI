# 30× Speedup Without Changing Your Python Code: <br>A Practical CUDA Benchmarking Playbook

What if you could get **~30× faster Black-Scholes pricing** - not by rewriting the algorithm, but by rethinking *how the system executes it*?

I gave a straightforward NumPy Black-Scholes implementation to an AI agent that specializes in **Numba + CUDA performance optimization**.

It didn’t change the formula.

Instead it applied **systems thinking**.

Result: **~34× speedup** on a full pipeline benchmark.

---

## What changed (high level)

The original code was a clean NumPy implementation.

The optimized version introduced a **GPU execution pipeline**:

• Ported the compute-heavy section to a **Numba CUDA kernel**

• Used **pinned host memory** for faster async transfers

• Split the dataset into **chunks processed by multiple CUDA streams**

• Overlapped **Host→GPU copy, kernel execution, and GPU→Host copy**

• **Pre-allocated GPU memory** to eliminate allocation overhead

• Added warm-up iterations for fair benchmarking

• Verified correctness with `np.allclose`

The math stayed the same.
The **system architecture changed.**

---

## Why systems thinking beats kernel micro-optimization

Many engineers jump straight into optimizing kernel math.

But the biggest wins often come from:

• Eliminating memory allocation overhead

• Overlapping compute and transfers

• Using pinned memory for high-throughput PCIe transfers

• Running multiple streams to keep the GPU busy

In other words:

**Optimize the pipeline, not just the kernel.**

---

## Practical GPU optimization checklist

1. Profile where time is spent (compute vs transfers)
2. Use **pinned host memory** for large transfers
3. Chunk workloads and use **CUDA streams**
4. **Pre-allocate device memory**
5. Warm up kernels before benchmarking
6. Always `cuda.synchronize()` when measuring time
7. Verify numerical correctness after porting

---

## Benchmark result

CPU (NumPy): ~25.8s
GPU pipeline: ~0.75s

**Speedup: ~34×**

---

## Full code (original + optimized)

I’ve shared a notebook with the **complete benchmark, original NumPy code, and the optimized CUDA pipeline implementation** here:

**GitHub Notebook:**
[https://github.com/vbepipe/Benchmarking-AI/blob/main/CUDA-GPU-Benchmarks/cuda-performance-cuda-stream-pinned-memory-version-1.ipynb](https://github.com/vbepipe/Benchmarking-AI/blob/main/CUDA-GPU-Benchmarks/cuda-performance-cuda-stream-pinned-memory-version-1.ipynb)

---

## Takeaway

AI agents can accelerate engineering workflows by quickly exploring **system-level optimization strategies** that would otherwise take hours of manual experimentation.

Sometimes the fastest code is not about rewriting algorithms.

It’s about **designing the right execution pipeline.**

---

#CUDA #Numba #Python #GPUComputing #HighPerformanceComputing #SystemsThinking #AIforEngineering

---
