# Custom18 VictoryGLU: A New Benchmark in Activation Performance

We evaluated five activation functions across multiple model depths, measuring training and validation loss under early stopping. The results demonstrate that **Custom18 VictoryGLU** is the superior choice, securing the best validation loss in **3 out of 4** model configurations, effectively dethroning SwiGLU in shallow networks and CustomV2 in deep networks.

## Activation Function Performance Overview (Early Stop Implemented)

| Activation | 4-Layer Val<br>(2000 iters) | 8-Layer Val<br>(2000 iters) | 6-Layer Val<br>(4000 iters) | 12-Layer Val<br>(8000 iters) | Winner Count |
|------------|-------------------|-------------------|-------------------|---------------------|--------------|
| **Custom18 VictoryGLU** | **1.7922** ✓ (step 2000) | **1.5135** ✓ (step 2000) | 1.5021 (step 3500) | **1.5166** ✓ (step 2500) | **3** |
| **CustomV2** | 1.8437 (step 2000) | 1.5397 (step 2000) | **1.4887** ✓ (step 4000) | 1.5246 (step 2500) | **1** |
| **SwiGLU** | 1.8176 (step 2000) | 1.5169 (step 2000) | 1.5146 (step 4000) | 1.5267 (step 2500) | **0** |
| **CustomV3** | 1.8687 (step 2000) | 1.5449 (step 2000) | 1.4914 (step 4000) | 1.5485 (step 3500) | **0** |
| **GELU** | 1.9061 (step 2000) | 1.5411 (step 2000) | 1.5025 (step 4000) | 1.5281 (step 2500) | **0** |

**Training Configuration:**
- 4-Layer and 8-Layer models: trained for 2000 iterations total
- 6-Layer model: trained for 4000 iterations total
- 12-Layer model: trained for 8000 iterations total

The table shows that **Custom18 VictoryGLU** dominates the field, achieving the lowest validation loss in shallow (4-layer), medium (8-layer), and deep (12-layer) architectures. **CustomV2** remains a strong contender, holding the record for the 6-layer configuration.

---

<br> 

## Activation Function Analysis (With Early Stopping)

### Custom18 VictoryGLU Analysis

**Strengths:**
- **The new champion**: Wins 3 out of 4 categories, demonstrating superior performance across a wide range of model depths.
- **Shallow Model Dominance**: Significantly outperforms SwiGLU in the 4-layer test (1.7922 vs 1.8176).
- **Deep Model Stability**: Successfully scales to 12 layers, beating the previous winner CustomV2 (1.5166 vs 1.5246).
- **Consistent Convergence**: Reaches optimal validation loss efficiently, often peaking around step 2000-2500 in deeper models.

**Weaknesses:**
- **6-Layer Anomaly**: It did not win the 6-layer category, falling behind both CustomV2 and CustomV3, suggesting there may be specific depth/width ratios where V2/V3 dynamics are preferable.

---

### CustomV2 Analysis

**Strengths:**
- **Mid-range Specialist**: Retains the crown for the 6-layer architecture (1.4887), significantly outperforming the new VictoryGLU in this specific configuration.
- **Competitive Deep Performance**: Remains the second-best option for 12-layer models, only marginally behind VictoryGLU.

**Weaknesses:**
- **Shallow Performance**: Struggles to compete with VictoryGLU and SwiGLU in 4-layer and 8-layer configurations.

---

### SwiGLU Analysis

**Strengths:**
- **Consistent Runner-up**: While it secured no wins in this comparison, it remains a very strong baseline, consistently beating GELU and often placing 2nd in shallow architectures.
- **Fast Convergence**: Like VictoryGLU, it tends to reach good loss values quickly.

**Weaknesses:**
- **Outclassed**: It has been effectively superseded by VictoryGLU, which beats it in every single category tested (4L, 6L, 8L, and 12L).

---

### CustomV3 Analysis

**Strengths:**
- **Late Bloomer**: In the 12-layer run, it achieved its best loss at step 3500 (later than others), reinforcing its characteristic gradual learning curve.
- **6-Layer Competence**: Performed very well in the 6-layer test (2nd place), beating VictoryGLU.

**Weaknesses:**
- **No Wins**: Fails to secure a top spot in any category under early stopping conditions.
- **Inefficient**: Generally requires more iterations to reach competitive loss levels compared to VictoryGLU.

---

### GELU Analysis

**Strengths:**
- **Baseline Stability**: Predictable behavior, but offers no performance advantage.

**Weaknesses:**
- **Obsolete**: Consistently the worst or near-worst performer across all depths. There is no statistical reason to prefer GELU over VictoryGLU or SwiGLU in this benchmark.

---

## Recommendation 

**Custom18 VictoryGLU** is the clear recommendation for general-purpose training, providing the best validation loss in the majority of configurations (Shallow, Medium, and Deep). 

However, **CustomV2** remains a valid alternative for specific mid-depth architectures (e.g., 6 layers), where it demonstrated a unique advantage. For all other cases, VictoryGLU provides the most reliable and performant results.

---

Source Log Files:

[https://github.com/vbepipe/Benchmarking-AI/tree/main/nanoGPT](https://github.com/vbepipe/Benchmarking-AI/tree/main/nanoGPT)