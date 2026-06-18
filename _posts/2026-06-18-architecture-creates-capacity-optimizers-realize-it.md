---
layout: post
title: "Architecture Creates Capacity. Optimizers Realize It."
subtitle: "Why LLM design needs to track realized capacity, not just parameter count or loss"
author: "Nandan Kumar Jha"
date: 2026-06-18
permalink: /blog/optimizer-induced-capacity/
description: "A flagship essay on realized capacity: why architecture creates available degrees of freedom, while optimizers determine which capacity is actually realized."
reading_time: "~20 min read"
---

> **Core thesis.** Architecture defines the representational degrees of freedom a model *could* use. Optimization decides which of those degrees of freedom become active, distributed, transferable, and adaptable.

<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure1_optimizer_capacity_scaling.png' | relative_url }}" alt="Same architecture, different optimizer, different realized capacity">
  <figcaption><strong>Figure 1.</strong> <em>Same architecture, different optimizer, different realized capacity scaling.</em> The architecture and data are fixed, but the optimizer changes the training trajectory and the resulting spectral scaling laws. The right panel summarizes optimizer-level scaling exponents: $\beta_{\mathrm{soft}}$ for Average-realized Capacity Scaling, $\beta_{\mathrm{hard}}$ for Dominant-mode Capacity Scaling, and $\Delta\beta=\beta_{\mathrm{soft}}-\beta_{\mathrm{hard}}$ for Scaling Asymmetry.</figcaption>
</figure>

## TL;DR

- **Same loss is not the same model.** Prior work shows that models with the same pretraining loss can differ substantially in downstream transfer; training dynamics can select different same-loss solutions.
- **Parameter count is nominal capacity.** Width, depth, heads, experts, and FLOPs describe what the architecture makes available, not what training actually uses.
- **Realized capacity is the missing coordinate.** We need internal telemetry for which representation directions become active, which regimes receive capacity, and which structures become strong enough to transfer.
- **Spectral geometry gives one such telemetry layer.** In this post, $R_{\mathrm{soft}}$ is the **Average-realized Capacity** effective-rank measure, $R_{\mathrm{hard}}$ is the **Dominant-mode Capacity** effective-rank measure, and **Capacity Asymmetry** is the log-ratio $\Delta = \log R_{\mathrm{soft}} - \log R_{\mathrm{hard}}$. When comparing how ranks scale with FFN width, we use $\beta_{\mathrm{soft}}$, $\beta_{\mathrm{hard}}$, and $\Delta\beta=\beta_{\mathrm{soft}}-\beta_{\mathrm{hard}}$.
- **The empirical claim, in one line.** Holding architecture, data, and width schedule fixed, optimizers produce sharply different capacity-scaling exponents. On rare-token (TAIL) representations, AdamW exhibits weak hard-rank scaling ($\beta_{\mathrm{hard}} = 0.44$) while Muon and NorMuon achieve near-linear hard-rank scaling ($\beta_{\mathrm{hard}} \approx 1.0$) — a $2.3\times$ larger exponent under identical architecture. Soft-rank scaling separates these optimizers much less, so the optimizer-induced effect is concentrated where it matters most: dominant-mode capacity in the long tail.
- **Architecture–optimizer co-design should be a first-class LLM design axis.** The right question is not only which architecture scales, or which optimizer trains fastest, but which pair converts compute, parameters, and data into useful internal structure.

> **Citation note.** I cite primary papers and original technical sources at the point where an external concept enters the argument. Claims about the optimizer-induced spectral-scaling-law experiments refer to our paper *Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws* and its project/code release linked below. For very recent optimizers, I distinguish peer-reviewed papers from arXiv reports and original public write-ups.

**Paper context.** This essay is the first blog in a series around our paper **[Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws](https://arxiv.org/abs/2605.21803)** by **Nandan Kumar Jha** and **Brandon Reagen**. Project page: [optimizer-scaling-laws.github.io](https://optimizer-scaling-laws.github.io/). Code: [optimizer-scaling-laws/spectral-scaling-laws](https://github.com/optimizer-scaling-laws/spectral-scaling-laws).

---

## 1. Same loss is not the same model

Most LLM training dashboards are organized around a small set of signals: loss, gradient norm, learning rate, tokens/sec, memory, and sometimes downstream evaluations. These signals are necessary. They are not sufficient.

The reason is simple: **the same loss can be reached through different internal organizations of the model**.

A language model can reduce average next-token error by building broad, distributed representations across many directions, or by concentrating prediction-relevant variance into a smaller number of dominant modes. It can allocate capacity uniformly across token regimes, or it can spend most of its structure on frequent patterns while leaving rare regimes under-realized. It can use added FFN width as genuinely new representational structure, or it can leave much of that width only nominally present.

From the outside, two models may look similar: same architecture, similar parameter count, similar pretraining loss. Internally, they may not be the same model at all.

This matters because many things we care about in LLMs are not fully summarized by average loss: downstream transfer, rare-token behavior, domain robustness, multilingual coverage, tool-use reliability, code identifiers, scientific terminology, future adaptation, and the reserve of degrees of freedom left for continued learning.

Prior work gives direct evidence for this concern. Liu et al. show that pretraining loss does not fully explain downstream performance: among models with the same pretraining loss, changing the training algorithm can produce different downstream transfer, revealing an implicit bias of the pretraining procedure itself ([Liu et al., 2022](https://arxiv.org/abs/2210.14199)). More recent optimizer-level work makes the geometric version of the point: same-loss pretraining can converge to solutions with different relationships among task-specific minima, and optimizer design can bias training toward solutions that transfer better ([Chen et al., 2026](https://arxiv.org/abs/2604.09258)). Representation-geometry studies also show why this is plausible: standard metrics like loss can miss non-monotonic changes in learned representation geometry during pretraining and post-training ([Li et al., 2025](https://arxiv.org/abs/2509.23024)).

This means the upstream–downstream gap is not only a question of nominal capacity. It is also a question of **realized capacity**: which internal directions training activates, which token regimes receive capacity, which minima become reachable, and which structures become strong enough to transfer.

So the question is not only:

> How low did the loss go?

It is also:

> What internal capacity did training realize in order to reach that loss?

That second question is the subject of this post.

The same issue appears inside our own experiments as a controlled loss-matching example. Extending AdamW training from 6K to 12K improves validation perplexity and brings it close to the low-rank Dion $(r=1/16)$ control across the FFN-width sweep. But the realized-capacity trajectories do not match: AdamW 12K remains much weaker in hard-rank growth, while Dion preserves steadily increasing hard and soft effective ranks.

<figure class="figure-wide">
  <picture>
    <source srcset="{{ '/assets/blog/architecture-optimizer-codesign/figure2_matched_loss_capacity_comparison.svg' | relative_url }}" type="image/svg+xml">
    <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure2_matched_loss_capacity_comparison.png' | relative_url }}" alt="Realized capacity comparison for matched loss">
  </picture>
  <figcaption><strong>Figure 2.</strong> <em>Realized capacity comparison for matched loss.</em> Extended AdamW training improves validation loss, but it does not recover the same rank trajectory as the low-rank Dion control. This is the concrete version of the broader argument: loss can improve while width-to-capacity scaling remains structurally different. The slope difference between the AdamW-12K hard-rank trace and Dion's is exactly what the optimizer-level scaling-asymmetry coefficient $\Delta\beta$ in Figure 1 quantifies.</figcaption>
</figure>

---

## 2. Architecture creates nominal capacity

A common way to reason about LLM design is to treat architecture as the main object and optimization as the procedure that fits it. In this view, architecture gives us the model, and the optimizer is the training algorithm used to find the weights.

That view is useful, but incomplete.

Architecture certainly matters. It provides things optimization cannot create after the fact:

- **Structural guarantees:** causal masking, equivariance, locality, sparse activation, routing constraints.
- **Compute pattern:** FLOPs per token, KV-cache structure, memory movement, conditional computation.
- **Dimensional ceilings:** width, depth, number of heads, FFN expansion, embedding dimension.
- **Compositional structure:** residual streams, normalization placement, attention blocks, MLP blocks, parameter sharing.

These are hard design choices. No optimizer can make a non-causal model causal, give a dense model sparse inference, or recover capacity that the architecture never admitted.

But architecture has a crucial limitation: **it creates capacity; it does not guarantee that capacity is used**. Adding FFN width raises a ceiling on representable structure but does not commit training to fill those new directions; increasing depth opens more computation paths but does not route signal through them; adding heads, experts, or low-rank structure changes the available function class without selecting which parts of it become load-bearing.

Architecture defines the space of possible computations. Which computation training actually realizes is decided elsewhere — by the optimizer.

---

## 3. Optimizers realize capacity

Optimization is not just the engine that minimizes loss. In overparameterized models, many solutions can fit the training data comparably well. The optimizer helps decide which of those solutions training actually finds.

This is the sense in which optimizers realize capacity: they shape which available directions grow, which spectral modes become load-bearing, which circuits become active, and which parts of the hypothesis space are reachable.

### 3.1 Hard Inductive Bias vs. Soft Inductive Bias

Architecture is the natural instrument for **Hard Inductive Bias**.

It restricts the support of the hypothesis space. Causal masking forbids attention to future tokens. Convolutional sharing enforces translation structure. Sparse MoE routing changes which parameters can activate for each token. These are not preferences; they are constraints. This hard-inductive-bias/soft-inductive-bias distinction is also consistent with the broader no-free-lunch/Kolmogorov-complexity view: learning succeeds when the learner's biases concentrate search on the low-complexity structure that real data tends to exhibit ([Goldblum et al., 2024](https://arxiv.org/abs/2304.05366); [Wilson, 2025](https://arxiv.org/abs/2503.02113)).

Optimization is the natural instrument for **Soft Inductive Bias**.

It does not usually remove hypotheses from the function class. Instead, it changes which hypotheses are easy to reach. AdamW, Muon, Shampoo, Dion, and related optimizers encode different geometric assumptions about parameters: coordinate-wise scaling, matrix-level orthogonalization, Kronecker-factored curvature, low-rank projected updates, and different forms of momentum and preconditioning.

The distinction can be summarized as support versus measure:

<table>
<thead><tr><th>Design object</th><th>Role</th><th>Practical effect</th></tr></thead>
<tbody>
<tr><td>Architecture</td><td>Hard Inductive Bias: restricts support</td><td>Determines what functions and computation patterns are possible</td></tr>
<tr><td>Optimizer</td><td>Soft Inductive Bias: reweights reachable solutions</td><td>Determines which solutions are likely to be found</td></tr>
<tr><td>Data</td><td>Likelihood/evidence</td><td>Determines where the solution is sharply constrained or underdetermined</td></tr>
<tr><td>Trained model</td><td>Joint outcome</td><td>The actual computation implemented by the learned weights</td></tr>
</tbody>
</table>

Equivalently, as a Bayesian-style analogy:

<div class="math-box">
$$
p(h \mid \mathcal{D}, \mathcal{A}, \mathcal{O})
\;\propto\;
p(\mathcal{D} \mid h)\,
p_{\mathcal{O}}(h \mid \mathcal{A}).
$$
</div>

Here, $\mathcal{A}$ is the architecture, $\mathcal{O}$ is the optimizer/training algorithm, $\mathcal{D}$ is the data, and $h$ is the trained hypothesis. Architecture acts as the support of the hypothesis space; the optimizer induces a prior over reachable solutions inside that support; data acts as the likelihood.

AdamW comes from decoupled weight decay for adaptive optimization ([Loshchilov and Hutter, 2019](https://arxiv.org/abs/1711.05101)); Shampoo is a structure-aware tensor preconditioner ([Gupta, Koren, and Singer, 2018](https://arxiv.org/abs/1802.09568)); Muon was introduced as a matrix-orthogonalized optimizer in an original public write-up and has since been studied for LLM scaling ([Jordan et al., 2024](https://kellerjordan.github.io/posts/muon/); [Liu et al., 2025](https://arxiv.org/abs/2502.16982)); NorMuon combines Muon-style orthogonalization with neuron-wise adaptive rates ([Li et al., 2025](https://arxiv.org/abs/2510.05491)); and Dion studies distributed orthonormalized updates with low-rank communication structure ([Ahn et al., 2025](https://arxiv.org/abs/2504.05295)). These choices change the path through parameter space, the effective prior over solutions, and the spectral structure that emerges in representations.

The architecture determines what functions are available. The optimizer determines which available functions become likely.

### 3.2 Static graph vs. effective graph

The Transformer architecture gives us a static operator graph: attention blocks, MLP blocks, residual additions, layer normalizations, and projection matrices ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).

But the trained model is not just this static graph. Inside it is an **effective signal-flow graph**: the set of routes through heads, MLP directions, residual-stream features, and layer compositions that actually carry meaningful signal for a behavior.

The operator graph is the menu. The effective graph is what training orders from the menu. This framing is closely related to the transformer-circuits view of a Transformer as a composition of residual-stream paths and attention/MLP components ([Elhage et al., 2021](https://transformer-circuits.pub/2021/framework/index.html)).

<table>
<thead><tr><th>Graph</th><th>Fixed by</th><th>What it contains</th><th>What it misses</th></tr></thead>
<tbody>
<tr><td>Static operator graph</td><td>Architecture</td><td>Matmuls, attention, MLPs, residual additions, normalization placement</td><td>Which routes actually carry meaningful signal after training</td></tr>
<tr><td>Effective signal-flow graph</td><td>Architecture × optimizer × data</td><td>Load-bearing paths, active representation directions, behavior-specific circuits</td><td>Formal paths that exist but carry negligible signal</td></tr>
</tbody>
</table>

Two models with the same architecture can therefore have different effective graphs. The same paths exist in both models, but different paths become load-bearing, and even the same path can carry signal with different spectral shape. One optimizer may fragment a rare signal across coordinate-wise statistics. Another may preserve it as a coherent matrix-level direction.

This matters for interpretability. A circuit found in an AdamW-trained model is not automatically a circuit of the architecture. It may be a circuit of the architecture–optimizer pair. Change the optimizer, and the same architecture may build a different effective computation for the same behavior.

### 3.3 Reachability: the trainable frontier is pair-dependent

There is a third phenomenon between Hard Inductive Bias and Soft Inductive Bias: **reachability**.

A function may be representable by the architecture but effectively unreachable under a given optimizer. In that case, the optimizer has not changed the formal hypothesis space, but it has changed the subset of that space training can actually access.

This is why architecture search under a fixed optimizer can be misleading. An architecture that appears unstable or underperforming under one optimizer may become viable under another. Conversely, an architecture that looks strong under AdamW may be partly strong because AdamW and that architecture form a historically well-tuned pair.

The trainable frontier is therefore not the set of architectures alone. It is the set of viable architecture–optimizer pairs.

---

## 4. Realized capacity: the missing coordinate

The architecture–optimizer distinction becomes sharper if we separate nominal capacity from realized capacity.

> **Nominal capacity** is the capacity implied by architecture: parameter count, width, depth, heads, experts, and FLOPs.

> **Realized capacity** is the capacity actually used by the trained model: which representation directions become active, how spectral mass is distributed, which modes grow with width, and which data regimes receive usable internal structure.

A wider road does not guarantee higher traffic flow if the routing policy does not use the lanes. A wider FFN does not guarantee more useful representation capacity if training concentrates activity into a small subset of directions.

This leads to a useful decomposition:

<div class="math-box">
$$
C_{\mathrm{realized}}(\mathcal{A}, \mathcal{O}, \mathcal{D})
\;\approx\;
C_{\mathrm{available}}(\mathcal{A})
\;\times\;
\rho_{\mathrm{realized}}(\mathcal{O}, \mathcal{D}; \mathcal{A}).
$$
</div>

In words: architecture sets the available capacity, while optimization and data determine the realized fraction. The first term is controlled mostly by architecture. The second is controlled by the optimization trajectory and its implicit bias, conditioned on the data distribution.

A compact way to keep the distinction straight is:

<table>
<thead><tr><th>Observable</th><th>What it tells us</th><th>Mostly controlled by</th></tr></thead>
<tbody>
<tr><td>Parameter count</td><td>How many trainable degrees of freedom exist</td><td>Architecture</td></tr>
<tr><td>Width, depth, heads, experts</td><td>The nominal capacity ceiling</td><td>Architecture</td></tr>
<tr><td>FLOPs and memory movement</td><td>The cost and shape of computation</td><td>Architecture</td></tr>
<tr><td>Average-realized Capacity</td><td>How broadly representation mass spreads across modes</td><td>Training dynamics</td></tr>
<tr><td>Dominant-mode Capacity</td><td>How many directions become load-bearing</td><td>Architecture–optimizer pair</td></tr>
<tr><td>Capacity Asymmetry</td><td>The gap between broad support and dominant-mode structure</td><td>Architecture–optimizer pair</td></tr>
<tr><td>Frequency-conditioned capacity</td><td>Which token regimes receive usable capacity</td><td>Data distribution × optimizer</td></tr>
<tr><td>Downstream/continued-learning behavior</td><td>Whether internal structure transfers or remains plastic</td><td>Model state × evaluation regime</td></tr>
</tbody>
</table>

This decomposition is not meant to be a literal scalar law. It is a mental model. But it makes a concrete prediction: **the same architectural intervention can have different realized effects under different optimizers**.

If the optimizer-realized fraction is near one, adding architecture translates cleanly into realized capacity. If that fraction is far below one, adding architecture raises the ceiling but does not fill it. In the second regime, optimizer choice becomes first-order.

That is the regime modern LLMs increasingly live in: long-tail data, sparse supervision, deep networks, rare skills, tool-use patterns, code, multilingual data, domain-specific knowledge, and increasingly structured behaviors.

<figure>
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure3_phase_diagram_for_realized_capacity.png' | relative_url }}" alt="Phase diagram for realized capacity">
  <figcaption><strong>Figure 3.</strong> <em>Phase diagram for realized capacity.</em> Average-realized Capacity (normalized $\log R_{\mathrm{soft}}$) on the horizontal axis; Dominant-mode Capacity (normalized $\log R_{\mathrm{hard}}$) on the vertical. The forbidden region encodes the constraint $R_{\mathrm{hard}} \le R_{\mathrm{soft}}$. Four regimes emerge: Collapsed (low average, low dominant-mode), Compact/coherent (moderate support, coherent usable structure), Diffuse/weakly load-bearing (broad support but weaker dominant structure), and Distributed usable (broad support with high dominant-mode capacity). The Asymmetry gap $\Delta = \log R_{\mathrm{soft}} - \log R_{\mathrm{hard}}$ measures how much of the apparently "used" capacity is genuinely load-bearing.</figcaption>
</figure>

---

## 5. Spectral telemetry for realized capacity

If realized capacity is the object we care about, we need measurements that see it.

Parameter count does not see it. FLOPs do not see it. Loss sees output error, but not the internal structure used to reduce that error.

Spectral geometry gives a practical telemetry layer.

Given a representation covariance, its eigenspectrum tells us how variance is distributed across directions. Effective-rank measures summarize this distribution; they sit in the broader information-theoretic tradition of treating capacity as effective dimensionality, with roots in rate-distortion theory ([Shannon, 1959](https://gwern.net/doc/cs/algorithm/information/1959-shannon.pdf)). The entropy-based effective-rank formulation traces back to Roy and Vetterli, and the broader Rényi family gives a principled way to vary the sensitivity to tail versus dominant modes ([Roy and Vetterli, 2007](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf); [Rényi, 1961](https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/On-Measures-of-Entropy-and-Information/chapter/On-Measures-of-Entropy-and-Information/bsmsp/1200512181)). They are not perfect, but they expose structure that scalar loss hides.

Let the eigenvalues of a representation covariance be $(\lambda_1, \ldots, \lambda_d)$, and normalize them into a spectral distribution:

<div class="math-box">
$$
p_i = \frac{\lambda_i}{\sum_j \lambda_j}.
$$
</div>

The Rényi-entropy family gives a continuum of effective ranks:

<div class="math-box">
$$
H_\alpha(p)
= \frac{1}{1-\alpha} \log \sum_i p_i^\alpha,
\qquad
R_\alpha(p) = \exp\!\left(H_\alpha(p)\right).
$$
</div>

Two special cases are especially useful:

<div class="math-box">
$$
R_1(p) = \exp\!\left(-\sum_i p_i \log p_i\right),
\qquad
R_2(p) = \frac{1}{\sum_i p_i^2}.
$$
</div>

In this blog, I use the following naming convention:

- **Average-realized Capacity** refers to the entropy-like rank $R_{\mathrm{soft}} = R_1$. It measures broad spectral support: how much representation mass is spread across many directions.
- **Dominant-mode Capacity** refers to the participation-ratio-like rank $R_{\mathrm{hard}} = R_2$. It measures how many directions carry substantial, load-bearing mass.
- **Capacity Asymmetry** refers to $\Delta = \log R_{\mathrm{soft}} - \log R_{\mathrm{hard}} = \log\!\left(R_{\mathrm{soft}} / R_{\mathrm{hard}}\right)$. It measures the multiplicative gap between broad support and dominant-mode structure. In the phase diagram, these ranks are shown as width-comparable log coordinates, e.g. normalized $\log R_{\mathrm{soft}}$ and normalized $\log R_{\mathrm{hard}}$. The inequality $R_{\mathrm{hard}} \le R_{\mathrm{soft}}$ holds in general, so valid points lie on or below the diagonal in Figure 3.

We report $R_{\mathrm{soft}}$ and $R_{\mathrm{hard}}$ as effective-rank capacity measures, and define Capacity Asymmetry in log space, $\Delta = \log R_{\mathrm{soft}} - \log R_{\mathrm{hard}}$, so that asymmetry measures the multiplicative gap between average-realized and dominant-mode capacity.

---

## 6. Rare-token regimes expose capacity allocation

Natural language is not statistically uniform. Token frequency follows a long-tailed, approximately Zipfian distribution. A small number of HEAD tokens receive enormous gradient mass. A large number of TAIL tokens receive sparse, noisy, and intermittent supervision.

This matters because the role of the optimizer changes with the statistical regime.

For HEAD tokens, the likelihood is sharp. The model sees abundant evidence. Many reasonable optimizers can find similar solutions because the data strongly constrains what should be learned.

For TAIL tokens, the likelihood is weak. The model sees fewer examples. Many solutions remain compatible with sparse evidence. In this regime, the implicit bias of the optimizer matters much more.

In Bayesian language:

> Where data is dense, the likelihood dominates. Where data is sparse, the prior dominates. The optimizer acts like a Soft Inductive Bias over reachable solutions.

This gives a simple regime map:

<table>
<thead><tr><th>Token regime</th><th>Data condition</th><th>Dominant bottleneck</th><th>Expected stronger lever</th></tr></thead>
<tbody>
<tr><td>HEAD</td><td>Dense, frequent, statistically well-conditioned supervision</td><td>Architectural ceiling and compute pattern</td><td>Architecture</td></tr>
<tr><td>MID</td><td>Partially constrained, mixed signal quality</td><td>Capacity allocation across modes</td><td>Architecture–optimizer pair</td></tr>
<tr><td>TAIL</td><td>Sparse, noisy, underdetermined supervision</td><td>Signal preservation and implicit prior</td><td>Optimizer</td></tr>
</tbody>
</table>

This predicts that optimizer-induced differences should become more visible in MID and TAIL regimes than in HEAD regimes. It also explains why architecture-only changes can under-deliver for long-tail behavior. Adding capacity raises the ceiling, but sparse data does not automatically fill the ceiling. The optimizer must preserve and amplify weak joint signals rather than fragment them.

This is practically important. Many frontier LLM behaviors are long-tail behaviors:

- low-resource multilingual modeling,
- rare code libraries and identifiers,
- scientific and technical vocabulary,
- tool-use edge cases,
- long-tail factual recall,
- structured reasoning patterns,
- rare instruction formats,
- expert specialization in sparse MoE settings.

If a model's average loss is dominated by frequent patterns, then loss can hide whether rare regimes are receiving usable representational capacity. Frequency-conditioned spectral telemetry is one way to expose that hidden allocation.

<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure4_capacity_scaling_asymmetry_token_regimes.png' | relative_url }}" alt="Capacity-scaling asymmetry across token regimes">
  <figcaption><strong>Figure 4.</strong> <em>Capacity-scaling asymmetry across token-frequency regimes.</em> Here $\Delta\beta = \beta_{\mathrm{soft}} - \beta_{\mathrm{hard}}$ measures whether broad spectral support scales faster than dominant-mode capacity as FFN width increases. Values near zero indicate low scaling asymmetry. The small negative MID values for Muon and NorMuon should be read as effectively zero asymmetry in this finite-width fit: hard-rank scaling keeps pace with soft-rank scaling.</figcaption>
</figure>

---

## 7. Why co-design is the right abstraction

The conclusion is not that optimizers matter more than architectures. That would be the wrong lesson.

The right lesson is that they do different jobs.

Architecture is the right tool when we need hard structure:

- enforce causal or equivariant constraints,
- reduce inference cost,
- create routing paths,
- add experts or sparsity,
- increase width or depth,
- provide modularity,
- change memory and compute patterns.

Optimization is the right tool when we need to decide how capacity is filled:

- which directions grow,
- which modes become dominant,
- whether capacity spreads or concentrates,
- whether rare signals remain coherent,
- which minima are reachable,
- how added width turns into representation structure,
- how the same architecture behaves across data regimes.

Both are necessary because they operate at different levels. Architecture controls support. Optimization controls measure. Architecture provides the operator graph. Optimization shapes the effective signal-flow graph. Architecture gives nominal capacity. Optimization realizes a fraction of it.

<table>
<thead><tr><th>If the goal is…</th><th>Architecture contributes…</th><th>Optimization contributes…</th></tr></thead>
<tbody>
<tr><td>Structural guarantees</td><td>Hard constraints such as masking, sparsity, routing, equivariance</td><td>Cannot create the guarantee, but can train within it</td></tr>
<tr><td>Efficient inference</td><td>FLOP/memory pattern, activation sparsity, cache structure</td><td>Cannot usually change inference structure after training</td></tr>
<tr><td>Long-tail capability</td><td>Extra representational room and routing paths</td><td>Preserves weak signals and decides which rare directions grow</td></tr>
<tr><td>Stable deep training</td><td>Normalization, residual topology, parameterization</td><td>Conditioning, preconditioning, and reachability</td></tr>
<tr><td>Transfer at same loss</td><td>Available representation structures</td><td>Which same-loss solution and geometry are selected</td></tr>
<tr><td>Future learnability</td><td>Reserve architectural capacity</td><td>Plasticity-preserving dynamics and regularization</td></tr>
<tr><td>Capacity-aware scaling</td><td>More available degrees of freedom</td><td>Higher realized fraction of those degrees of freedom</td></tr>
</tbody>
</table>

This is why the object of design should be the pair.

---

## 8. Beyond this paper: scaling laws, plasticity, and training dynamics

The architecture–optimizer view is not an isolated concern. It is part of a broader shift in how modern AI systems are being understood.

First, recent discussions in vision make the same methodological point from another domain. A CVPR 2026 invited-talk abstract argues that benchmark-level scaling can hide divergent internal strategies: vision models may match or surpass human accuracy on ImageNet while failing simple cognitive probes and relying on visual strategies unlike human perception. The proposed remedy is not merely to push scale further, but to align scaling laws with stronger learning objectives, naturalistic data, and architectural inductive biases inspired by biological vision ([CVPR 2026 invited talk abstract](https://cvpr.thecvf.com/virtual/2026/invited-talk/40399)). The analogy to LLMs is direct: the question is not only whether a model scales, but what internal strategy scaling produces.

Second, continual learning provides a useful analogy for why architecture alone is not the whole story. Wider networks can reduce catastrophic forgetting by giving tasks more room and reducing gradient interference ([Mirzadeh et al., 2022](https://arxiv.org/abs/2110.11526)), but future learnability is not determined by raw width alone. Work on loss of plasticity suggests that trainability can degrade through changes in activation structure, feature-rank collapse, spectral collapse, optimizer dynamics, and regularization ([Abbas et al., 2023](https://arxiv.org/abs/2303.07507); [Dohare et al., 2024](https://www.nature.com/articles/s41586-024-07711-7); [He et al., 2025](https://arxiv.org/abs/2509.22335)). This reinforces the architecture–optimizer view of scaling: width and depth define potential capacity, but learning dynamics determine how much of that capacity remains realized, usable, and adaptable.

Third, recent position work makes the methodological point explicit: AI systems should be studied as time-evolving training processes, not only as static artifacts to analyze or patch after training. Scaling laws have made loss prediction routine; the harder challenge is building predictive accounts of capabilities, biases, robustness, and safety-relevant behavior as they emerge through training ([Biderman et al., 2026](https://arxiv.org/abs/2606.06533)).

These three threads converge on a common point: the vision-scaling, plasticity, and training-dynamics communities all face the same diagnostic gap that scalar-loss scaling laws leave open. Each is, in its own setting, asking what internal structure scaling actually produces rather than only how a benchmark number moves.

Optimizer-induced spectral scaling laws fit naturally into this broader agenda. They track one measurable part of the coupling between architecture, data, and optimization dynamics: how training converts nominal capacity into realized internal structure.

---

## 9. Toward capacity-aware LLM design

The historical default for LLMs has been a strong and productive equilibrium: dense Transformer architecture plus AdamW-like optimization plus carefully tuned data and schedules. That recipe worked extremely well. But it also made it easy to mistake properties of a particular **architecture–optimizer pair** for properties of the architecture alone.

Modern models are deeper, more heterogeneous, more sparse, more tool-oriented, more multilingual, more code-heavy, and more dependent on rare, structured, and long-tail behaviors. In these regimes, the optimizer's Soft Inductive Bias can become first-order.

This is the core claim behind optimizer-induced spectral scaling laws. If two optimizers train the same architecture but produce different spectral-capacity scaling, then the optimizer has changed the model's realized capacity. The matched-loss control in Figure 2 strengthens the point: even improving or matching loss does not guarantee matching internal geometry. If those effects concentrate in MID and TAIL regimes, then the optimizer is not only changing optimization speed. It is changing how the model allocates internal structure across the data distribution.

That is a design axis.

The next generation of LLM design should not only ask how many parameters we train, how many tokens we consume, or how low the loss goes.

It should also ask what capacity becomes real.

Architecture creates degrees of freedom. Optimization decides which of them become active. Data determines where evidence is dense enough to constrain the solution and where the optimizer's prior fills in the gaps. The trained model is the joint outcome of all three.

> **Capacity-aware LLM design means tracking not just what the model could represent, but what training actually made it represent.**

---

## What this does not claim

There are several ways to overstate this argument, so it is worth being explicit.

First, spectral rank is not a complete theory of intelligence, generalization, or downstream ability. It is a telemetry signal. It should be interpreted alongside loss, evaluations, robustness tests, calibration, interpretability, and application-specific metrics.

Second, optimizer-induced spectral capacity is not automatically good. More capacity is not always better. The right question is where capacity appears, whether it is stable, whether it supports useful behaviors, and whether it improves generalization in the regimes that matter.

Third, architecture remains indispensable. Optimizers cannot create structural guarantees, reduce inference cost by construction, or represent functions excluded by the architecture. Co-design is not optimizer maximalism.

Fourth, the strongest claims require more evidence: larger models, more architectures, longer training, downstream probes, continued-learning tests, interpretability comparisons, and direct studies of rare-regime behavior.

But the basic point is already hard to ignore: architecture alone does not determine how a model uses its capacity, and loss alone does not reveal it.

---

## Open questions

This framing leads to several questions that I think are worth treating as first-class pretraining-science problems.

**1. Do optimizer-induced spectra predict downstream behavior?**
If two models have similar loss but different realized capacity, which spectral differences predict transfer, robustness, rare-token reliability, or domain adaptation?

**2. Are mechanistic circuits optimizer-conditional?**
If the optimizer changes the effective signal-flow graph, then circuits discovered in one trained model may not be stable across optimizers, even with the same architecture.

**3. How should we search over pairs?**
Neural architecture search typically fixes the optimizer. Optimizer evaluation typically fixes the architecture. The co-design view suggests this may miss regions where neither component looks optimal alone, but the pair is strong.

**4. Can realized-capacity telemetry improve scaling laws?**
Classical scaling laws predict loss from parameters, data, and compute ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361); [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)). A capacity-aware scaling law would also track how internal representation structure scales, and whether added parameters become usable degrees of freedom.

**5. What is the connection to plasticity and future learnability?**
A model with similar loss but different spectral allocation may differ in its ability to keep learning. If realized capacity measures active degrees of freedom, then capacity telemetry may also help diagnose loss of plasticity, representational collapse, or the exhaustion of useful directions during continued training. Recent work even reframes plasticity itself as a measure of empowerment over future learning trajectories, suggesting that *what training preserves* may be as important as what it minimizes ([Abel et al., 2025](https://arxiv.org/abs/2505.10361)).

---

## Suggested links

- Paper: *[Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws](https://arxiv.org/abs/2605.21803)*
- Code: [`optimizer-scaling-laws/spectral-scaling-laws`](https://github.com/optimizer-scaling-laws/spectral-scaling-laws)
- Project page: [`optimizer-scaling-laws.github.io`](https://optimizer-scaling-laws.github.io/)
- Next post: *Rényi Effective Rank: A Spectral Lens on Realized Capacity*

---

## References

### This project

- Jha, N. K., and Reagen, B. *Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws*. arXiv:2605.21803, 2026. [`arXiv`](https://arxiv.org/abs/2605.21803) / [`project`](https://optimizer-scaling-laws.github.io/) / [`code`](https://github.com/optimizer-scaling-laws/spectral-scaling-laws)

### Scaling laws and Transformer architecture

- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. *Scaling Laws for Neural Language Models*. arXiv:2001.08361, 2020. [`arXiv`](https://arxiv.org/abs/2001.08361)
- Hoffmann, J. et al. *Training Compute-Optimal Large Language Models*. NeurIPS, 2022. [`arXiv`](https://arxiv.org/abs/2203.15556)
- Vaswani, A. et al. *Attention Is All You Need*. NeurIPS, 2017. [`arXiv`](https://arxiv.org/abs/1706.03762)
- CVPR 2026 invited talk abstract. *Scaling laws, neural laws, and human-like visual strategies* (title paraphrased from abstract). [`CVPR virtual page`](https://cvpr.thecvf.com/virtual/2026/invited-talk/40399)

### Upstream–downstream gap and training dynamics

- Liu, H., Xie, S. M., Li, Z., and Ma, T. *Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models*. arXiv:2210.14199, 2022. [`arXiv`](https://arxiv.org/abs/2210.14199)
- Chen, H., Zhang, H., Li, X., Dong, Y., Shen, K., and Zhu, J. *Nexus: Same Pretraining Loss, Better Downstream Generalization via Common Minima*. arXiv:2604.09258, 2026. [`arXiv`](https://arxiv.org/abs/2604.09258)
- Li, M. Z., Agrawal, K. K., Ghosh, A., Teru, K. K., Santoro, A., Lajoie, G., and Richards, B. A. *Tracing the Representation Geometry of Language Models from Pretraining to Post-training*. arXiv:2509.23024, 2025. [`arXiv`](https://arxiv.org/abs/2509.23024)
- Biderman, S., Khan, M. A., Mireshghallah, N., Arnett, C., Barez, F., and Saphra, N. *Position: Don't Just "Fix it in Post": A Science of AI Must Study Training Dynamics*. arXiv:2606.06533, 2026. [`arXiv`](https://arxiv.org/abs/2606.06533)

### Optimization and optimizer-induced bias

- Loshchilov, I., and Hutter, F. *Decoupled Weight Decay Regularization*. ICLR, 2019. [`arXiv`](https://arxiv.org/abs/1711.05101)
- Gupta, V., Koren, T., and Singer, Y. *Shampoo: Preconditioned Stochastic Tensor Optimization*. ICML, 2018. [`arXiv`](https://arxiv.org/abs/1802.09568)
- Jordan, K. et al. *Muon: An Optimizer for Hidden Layers in Neural Networks*. Original public technical write-up, 2024. [`write-up`](https://kellerjordan.github.io/posts/muon/)
- Liu, J. et al. *Muon is Scalable for LLM Training*. arXiv:2502.16982, 2025. [`arXiv`](https://arxiv.org/abs/2502.16982)
- Li, Z., Liu, L., Liang, C., Chen, W., and Zhao, T. *NorMuon: Making Muon More Efficient and Scalable*. arXiv:2510.05491, 2025. [`arXiv`](https://arxiv.org/abs/2510.05491)
- Ahn, K., Xu, B., Abreu, N., and Langford, J. *Dion: Distributed Orthonormalized Updates*. arXiv:2504.05295, 2025. [`arXiv`](https://arxiv.org/abs/2504.05295)

### Spectral capacity and information-theoretic framing

- Shannon, C. E. *Coding Theorems for a Discrete Source with a Fidelity Criterion*. IRE International Convention Record, 1959. [`PDF`](https://gwern.net/doc/cs/algorithm/information/1959-shannon.pdf)
- Rényi, A. *On Measures of Entropy and Information*. Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability, 1961. [`Project Euclid`](https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/On-Measures-of-Entropy-and-Information/chapter/On-Measures-of-Entropy-and-Information/bsmsp/1200512181)
- Roy, O., and Vetterli, M. *The Effective Rank: A Measure of Effective Dimensionality*. EUSIPCO, 2007. [`PDF`](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf)

### Inductive bias, interpretability, and plasticity

- Goldblum, M., Finzi, M., Rowan, K., and Wilson, A. G. *The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning*. ICML, 2024. [`arXiv`](https://arxiv.org/abs/2304.05366)
- Wilson, A. G. *Deep Learning is Not So Mysterious or Different*. arXiv:2503.02113, 2025. [`arXiv`](https://arxiv.org/abs/2503.02113)
- Elhage, N. et al. *A Mathematical Framework for Transformer Circuits*. Transformer Circuits, 2021. [`article`](https://transformer-circuits.pub/2021/framework/index.html)
- Mirzadeh, S. I., Chaudhry, A., Yin, D., Hu, H., Pascanu, R., Gorur, D., and Farajtabar, M. *Wide Neural Networks Forget Less Catastrophically*. ICML, 2022. [`arXiv`](https://arxiv.org/abs/2110.11526)
- Abbas, Z., Zhao, R., Modayil, J., White, A., and Machado, M. C. *Loss of Plasticity in Continual Deep Reinforcement Learning*. arXiv:2303.07507, 2023. [`arXiv`](https://arxiv.org/abs/2303.07507)
- He, N., Guo, K., Prakash, A., Tiwari, S., Tao, R. Y., Serapio, T., Greenwald, A., and Konidaris, G. *Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning*. arXiv:2509.22335, 2025. [`arXiv`](https://arxiv.org/abs/2509.22335)
- Dohare, S., Hernandez-Garcia, J. F., Rahman, P., Mahmood, A. R., and Sutton, R. S. *Maintaining Plasticity in Deep Continual Learning*. Nature, 2024. [`article`](https://www.nature.com/articles/s41586-024-07711-7)
- Abel, D. et al. *Plasticity as the Mirror of Empowerment*. arXiv:2505.10361, 2025. [`arXiv`](https://arxiv.org/abs/2505.10361)
