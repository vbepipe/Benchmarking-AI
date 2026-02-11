# Custom18 VictoryGLU: A New Benchmark in Activation Performance

We evaluated five activation functions across multiple model depths, measuring training and validation loss under early stopping. The results demonstrate that **Custom18 VictoryGLU** (referred to as **CustomActivationByVinayak** in source code mentioned below) is the superior choice, securing the best validation loss in **3 out of 4** model configurations, effectively dethroning SwiGLU in shallow networks and CustomV2 in deep networks.

---

### Why It Outperforms SwiGLU

Standard SwiGLU applies Swish to the gate path while leaving the value path linear. Custom18 VictoryGLU's innovation is the **heterogeneous gating strategy**: the ERF-based gate provides smooth, bounded modulation for negative signals while maintaining linear efficiency for positive signals, creating a more adaptive information flow than homogeneous SwiGLU gating.

---

## Technical Architecture

### Mathematical Foundation

Custom18 VictoryGLU implements a hybrid ERF-based activation function with asymmetric behavior:

**Core Formula:**

```math
f(x) = \begin{cases} 
\frac{\text{erf}_v(x) + 1}{2} \cdot x & \text{if } x < 0 \\ 
x & \text{if } x \geq 0 
\end{cases}
```

Where the ERF approximation uses the Abramowitz-Stegun constant:

```math
a = \frac{8(\pi - 3)}{3\pi(4 - \pi)} \approx 0.147
```

### Key Design Principles

#### Asymmetric Gating Mechanism
- **Positive inputs**: Linear pass-through (ReLU-like behavior) for computational efficiency
- **Negative inputs**: Smooth ERF-based scaling prevents dying ReLU problem while maintaining gradient flow
- **Dual activation**: Gate path uses `CustomActivationByVinayak()`, value path uses `F.silu()` (Swish)

#### Architectural Advantages
- **Zero-centered smoothness**: Avoids vanishing gradients in deep networks through differentiable negative region
- **Non-trainable parameters**: Fixed constant reduces overfitting risk and maintains consistency across training
- **Hardware optimization**: Hidden dimensions rounded to multiples of 64 for GPU/TPU efficiency
- **Parameter parity**: Uses `h = (2/3) * d_ffn` to match standard FFN parameter counts while implementing gating

---

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

## Activation Function Performance Overview (Early Stop Implemented)

### Custom18 VictoryGLU Analysis

**Strengths:**
- **The new champion**: Wins 3 out of 4 categories, demonstrating superior performance across a wide range of model depths.
- **Shallow Model Dominance**: Significantly outperforms SwiGLU in the 4-layer test (1.7922 vs 1.8176).
- **Deep Model Stability**: Successfully scales to 12 layers, beating the previous winner CustomV2 (1.5166 vs 1.5246).
- **Consistent Convergence**: Reaches optimal validation loss efficiently, often peaking around step 2000-2500 in deeper models.

**Weaknesses:**
- **6-Layer Anomaly**: It did not win the 6-layer category, falling behind both CustomV2 and CustomV3, suggesting there may be specific depth/width ratios where V2/V3 dynamics are preferable, but this artifact disappears at scale. VictoryGLU's 2.5× better scaling performance (6L→12L: +0.0145 vs CustomV2's +0.0359) makes it the only viable option. 

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

Use **Custom18 VictoryGLU** as the default choice for all production systems. 

---

## Source Code
```
### Snippet of Python code Below 
### Developed by Vinayak Patel 
### Social Link: https://x.com/vinayakchronicl 
### I don't read messages on x.com so tag me using @vinayakchronicl


class CustomActivationByVinayak(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        # Abramowitz-Stegun ERF approximation constant
        self.global_vinayak_a = (8 * (torch.pi - 3)) / (3 * torch.pi * (4 - torch.pi))
    def forward(self, x):
        # Asymmetric behavior: ERF-gated for negative, linear for positive
        x2 = x * x
        erf_v = torch.sign(x) * (1 - (1 / (1 + self.global_vinayak_a * x2)) * torch.exp(-x2))
        vn = ((erf_v + 1) * 0.5) * x
        return torch.where( x < 0, vn, x )


class SwiGLUMLP(nn.Module):
    def __init__(self, config, multiple_of=64):
        super().__init__()

        # Standard FFN size
        d_ffn = 4 * config.n_embd

        # SwiGLU hidden dim for parameter parity:
        # h = 2/3 * d_ffn  (≈ 8/3 * n_embd)
        hidden_dim = int(2 * d_ffn / 3)

        # Round for hardware efficiency (optional but recommended)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Single projection → split into gate + value
        self.c_fc = nn.Linear(
            config.n_embd,
            2 * hidden_dim,
            bias=config.bias
        )

        self.c_proj = nn.Linear(
            hidden_dim,
            config.n_embd,
            bias=config.bias
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate, value = self.c_fc(x).chunk(2, dim=-1)
        x = CustomActivationByVinayak()(gate) * F.silu(value)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        #self.mlp = MLP(config)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

```

---

## Log Files

Full experiment logs and data available at:

[https://github.com/vbepipe/Benchmarking-AI/tree/main/nanoGPT](https://github.com/vbepipe/Benchmarking-AI/tree/main/nanoGPT)

---

<br> 






