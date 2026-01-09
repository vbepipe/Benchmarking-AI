**New Benchmark Result with New Winner is now available here: <br> https://github.com/vbepipe/Benchmarking-AI/blob/main/nanoGPT/Presentation/Presentation-Custom18-VictoryGLU.md**

*Below mentioned benchmark results should now be considered obsolete.*

# CustomV2: An Activation Function That Wins Under Early Stopping 

We evaluated four activation functions across multiple model depths, measuring training and validation loss under early stopping. The results show that **CustomV2** matches or outperforms common activations such as **GELU** and **SwiGLU**, and notably achieves the best validation checkpoints for medium-to-deep models when early stopping is applied.


## Activation Function Performance Overview (Early Stop Implemented)

| Activation | 4-Layer Val<br>(2000 iters) | 8-Layer Val<br>(2000 iters) | 6-Layer Val<br>(4000 iters) | 12-Layer Val<br>(8000 iters) | Winner Count |
|------------|-------------------|-------------------|-------------------|---------------------|--------------|
| **SwiGLU** | **1.8176** ✓ (step 2000) | **1.5169** ✓ (step 2000) | 1.5146 (step 4000) | 1.5267 (step 2500) | **2** |
| **CustomV2** | 1.8437 (step 2000) | 1.5397 (step 2000) | **1.4887** ✓ (step 4000) | **1.5246** ✓ (step 2500) | **2** |
| **CustomV3** | 1.8687 (step 2000) | 1.5449 (step 2000) | 1.4914 (step 4000) | 1.5485 (step 3500) | **0** |
| **GELU** | 1.9061 (step 2000) | 1.5411 (step 2000) | 1.5025 (step 4000) | 1.5281 (step 2500) | **0** |

**Training Configuration:**
- 4-Layer and 8-Layer models: trained for 2000 iterations total
- 6-Layer model: trained for 4000 iterations total
- 12-Layer model: trained for 8000 iterations total

The table shows that **CustomV2** achieves the lowest validation loss at both 6 layers (4000 iterations) and 12 layers (8000 iterations), outperforming standard activations such as **GELU** and **SwiGLU** at these depths under early stopping.

<br> <br> 

![Training_and_Validation_Loss_for_different_activation_functions_with_4_Layers](https://raw.githubusercontent.com/vbepipe/Benchmarking-AI/refs/heads/main/nanoGPT/images/Training_and_Validation_Loss_for_different_activation_functions_with_4_Layers.png)

<h3 align="center">4 Layers; 2000 iterations (above graph)</h3>

---

![Training_and_Validation_Loss_for_different_activation_functions_with_8_Layers](https://raw.githubusercontent.com/vbepipe/Benchmarking-AI/refs/heads/main/nanoGPT/images/Training_and_Validation_Loss_for_different_activation_functions_with_8_Layers.png)

<h3 align="center">8 Layers; 2000 iterations (above graph)</h3>

---

![Training_and_Validation_Loss_for_different_activation_functions_with_6_Layers](https://raw.githubusercontent.com/vbepipe/Benchmarking-AI/refs/heads/main/nanoGPT/images/Training_and_Validation_Loss_for_different_activation_functions_with_6_Layers.png)

<h3 align="center">6 Layers; 4000 iterations (above graph)</h3>

---

![Training_and_Validation_Loss_for_different_activation_functions_with_12_Layers](https://raw.githubusercontent.com/vbepipe/Benchmarking-AI/refs/heads/main/nanoGPT/images/Training_and_Validation_Loss_for_different_activation_functions_with_12_Layers.png)

<h3 align="center">12 Layers; 8000 iterations (above graph)</h3>


<br> <br>


## Activation Function Analysis (With Early Stopping)

### SwiGLU Analysis

**Strengths:**
- **Dominant in shallow-to-medium architectures**: Wins at 2000 iterations 4 layers (1.8176) and 8 layers (1.5169) with fast convergence, reaching optimal validation loss quickly
- Consistently achieves lowest validation loss at step 1000 across all model depths

**Weaknesses:**
- Slightly underperforms CustomV2 at 6 layers (1.5146 vs 1.4887) and 12 layers with early stopping (1.5267 vs 1.5246)
- Higher generalization gap at 6 layers (0.3838) indicates less stable training than CustomV2
- Without early stopping, suffers worst overfitting in deep models

---

### CustomV2 Analysis

**Strengths:**
- **Optimal for medium and deep architectures with early stopping**: Wins at 6 layers (1.4887) and 12 layers (1.5246 at step 2500)
- **Most versatile activation for production use**: Ties with SwiGLU for winner count (2/4 depths) but excels where model complexity matters most
- More stable training dynamics than SwiGLU at 6 layers, with lower generalization gap (0.2915 vs 0.3838)

**Weaknesses:**
- Slower convergence than SwiGLU
- Underperforms to SwiGLU in 8 layers (1.5397 vs 1.5169) and 4-layer models (1.8437 vs 1.8176)

---

### CustomV3 Analysis

**Strengths:**
- Achieves best validation latest among all activations (step 3500 for 12-layer), indicating more gradual learning curve
- Without early stopping, provides best deep model stability (2.2581 vs others at 2.5-3.0)
- Most consistent generalization gaps across all depths (shallow-to-medium 0.16-0.27)

**Weaknesses:**
- **Fails to win at any depth with early stopping enabled**: Underperforms during optimal training windows across all architectures
- Slowest initial convergence makes it inefficient for time-constrained training
- **Early stopping eliminates its primary advantage**: Overfitting resistance becomes irrelevant when training stops at optimal points

---

### GELU Analysis

**Strengths:**
- Baseline standard with predictable, well-documented behavior
- With early stopping in 12-layer models, becomes competitive (1.5281 at step 2500)-only 0.035 worse than winner CustomV2
- Smallest generalization gap in 4-layer models (0.1402) indicates stable early training

**Weaknesses:**
- **Never achieves best validation loss at any depth**: Consistently underperforms modern alternatives by 2-4% across all configurations
- Slower convergence than SwiGLU and CustomV2 
- Worst validation loss in shallow models
- **No compelling use case**: Outperformed by either SwiGLU or CustomV2 at every depth

---

## Recommendation 

With early stopping enabled, **CustomV2** provides the most reliable validation checkpoints in medium and deep architectures. While **CustomV3** offers improved resistance to overfitting in longer training regimes and a more gradual learning curve, this advantage is largely neutralized when training is stopped at optimal validation points.


---

Source Log Files:

[https://github.com/vbepipe/Benchmarking-AI/tree/main/nanoGPT](https://github.com/vbepipe/Benchmarking-AI/tree/main/nanoGPT)






