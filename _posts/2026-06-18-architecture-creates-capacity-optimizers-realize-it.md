---
layout: post
title: "Architecture Creates Capacity, Optimizers Realize It."
subtitle: "Why LLM scaling needs realized-capacity telemetry, not just loss curves and parameter counts"
author: "Nandan Kumar Jha"
date: 2026-06-18
permalink: /blog/optimizer-induced-capacity/
description: "Why architecture creates available degrees of freedom, while optimization determines which internal capacity training actually realizes."
reading_time: "~25 min read"
---

> **Core thesis.** Architecture specifies the degrees of freedom a model *could* use. Optimization determines which of those degrees of freedom become active, load-bearing, transferable, and usable during training.

## TL;DR

- **Loss curves and parameter counts are incomplete telemetry.** They show how a training run looks from the outside, but not whether added architectural capacity becomes usable internal structure.
- **The empirical result is controlled and direct.** Holding the Transformer architecture, data, and FFN-width schedule fixed, different optimizers produce different spectral-capacity scaling laws.
- **The long tail is where the gap becomes most consequential.** In rare-token representations, AdamW and Dion-1/16 realize much weaker dominant-mode capacity scaling than Muon and NorMuon under the same architecture.
- **Optimizer choice is not only a convergence-speed decision.** It shapes which representation directions grow, which weak signals remain coherent, and which same-loss solutions become reachable.
- **The design implication is architecture–optimizer co-design.** The right object is not the architecture alone, or the optimizer alone, but the pair that converts compute, parameters, and data into realized internal structure.

## The one-minute version

Scaling laws have made us good at asking how loss changes with parameters, data, and compute. But they leave a quieter question under-measured: when we add capacity to a model, does training actually use it?

Our experiments suggest that the answer depends strongly on the optimizer. The architecture, data, and width schedule are fixed. What changes is the training algorithm. Yet the resulting models realize different spectral scaling laws inside their FFN representations. The difference is especially visible in rare-token regimes, where data is sparse and optimizer-induced bias has more room to matter.

This is the central point of the blog: **optimization is not a neutral procedure that merely fills a fixed architecture. In overparameterized LLMs, optimization helps decide what capacity becomes real.**

This is the controlled setup behind the empirical claim. The architecture and data are fixed; the optimizer is the intervention. What changes is the training trajectory, and with it the internal spectral-capacity profile reached by the model.

<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure0_same_architecture_same_data_different_optimizer.png' | relative_url }}" alt="Same architecture, same data, different optimizer conceptual setup">
  <figcaption><strong>Figure 0.</strong> <em>Same architecture, same data, different optimizer.</em> The model family, data distribution, and training task are held fixed. Optimizer choice changes the path through parameter space, and those different training dynamics can lead to different internal spectral-capacity profiles, even when final loss is similar.</figcaption>
</figure>

## 1. The blind spot: same loss is not the same model

Most LLM training dashboards are organized around a small set of external signals: loss, gradient norm, learning rate, tokens/sec, memory, and downstream evaluation scores. These signals are necessary. They are not sufficient.

The reason is simple: **the same loss can be reached through different internal organizations of the model**.

A language model can reduce next-token error by distributing representation across many directions, or by concentrating prediction-relevant variance into a smaller number of dominant modes. It can allocate capacity broadly across token regimes, or spend most of its internal structure on frequent patterns while leaving rare regimes under-realized. It can use added FFN width as genuinely new representational structure, or leave much of that width visible only in the parameter count.

From the outside, two models may look similar: same architecture, similar parameter count, similar validation loss. Internally, they may not be the same model.

This matters because many behaviors we care about in frontier systems are not fully summarized by average pretraining loss: downstream transfer, rare-token reliability, domain robustness, multilingual coverage, code identifiers, scientific terminology, tool-use edge cases, future adaptation, and the reserve of degrees of freedom left for continued learning.

Prior work already makes this concern concrete. Liu et al. show that pretraining loss does not fully explain downstream performance: among models with the same pretraining loss, changing the training algorithm can produce different downstream transfer, revealing an implicit bias of the pretraining procedure itself ([Liu et al., 2022](https://arxiv.org/abs/2210.14199)). More recent optimizer-level work makes the geometric version of the point: same-loss pretraining can converge to solutions with different relationships among task-specific minima, and optimizer design can bias training toward solutions that transfer better ([Chen et al., 2026](https://arxiv.org/abs/2604.09258)). Representation-geometry studies also show why this is plausible: scalar metrics can miss non-monotonic changes in learned representation geometry during pretraining and post-training ([Li et al., 2025](https://arxiv.org/abs/2509.23024)).

So the question is not only **how low did the loss go?** It is also:

> **What internal capacity did training realize in order to reach that loss?**

That second question is the subject of this post.

## 2. The empirical surprise: same architecture, different capacity scaling

The controlled result behind this essay is simple:

> **When architecture, data, and width schedule are fixed, changing the optimizer changes how representational capacity scales.**

In the paper, we vary FFN width under the same model family and compare how spectral effective ranks grow. The optimizer changes the slope of that growth. In other words, the model does not merely train faster or slower; it converts the same architectural budget into different internal capacity.

Figure 1 gives the aggregate view across token-frequency regimes. The important quantity is not just the final rank value. It is the **scaling exponent**: how much additional FFN width becomes additional realized spectral capacity. In this aggregate view, Muon and NorMuon exhibit much stronger hard-rank scaling than AdamW, while also showing lower soft–hard scaling asymmetry. That means the added width is not merely broadening weak spectral support; more of it becomes load-bearing dominant-mode structure.

<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure1_optimizer_capacity_scaling.png' | relative_url }}" alt="Aggregated optimizer-level realized-capacity scaling">
  <figcaption><strong>Figure 1.</strong> <em>Aggregated optimizer-level realized-capacity scaling.</em> Architecture, data, and FFN-width schedule are fixed; only the optimizer changes. Panel A shows aggregated scaling exponents for soft effective rank, $\beta_{\mathrm{soft}}$, and hard effective rank, $\beta_{\mathrm{hard}}$. Panel B shows scaling asymmetry, $\Delta\beta = \beta_{\mathrm{soft}} - \beta_{\mathrm{hard}}$. Muon and NorMuon show stronger hard-rank scaling and lower asymmetry than AdamW, indicating that more of the added width becomes load-bearing realized capacity. Figure 4 later resolves where this effect is strongest across HEAD, MID, and TAIL regimes.</figcaption>
</figure>

This is the main empirical reason to treat optimizer choice as a design axis. If two optimizers train the same architecture but produce different capacity-scaling exponents, then optimization is not just finding weights inside a fixed architecture. It is shaping which parts of that architecture become load-bearing.

This does not mean that one optimizer is universally better, or that spectral rank is a sufficient measure of capability. The narrower claim is more precise: **optimizer choice can change the internal scaling law by which width becomes usable representation.** That is already enough to make it a first-class object in pretraining science.

---

## 3. Loss can improve while capacity geometry remains different

The same issue appears inside our experiments as a controlled loss-matching example. Extending AdamW training from 6K to 12K improves validation perplexity and brings it close to the low-rank Dion-1/16 control across the FFN-width sweep. But the realized-capacity trajectories do not match: AdamW-12K remains much weaker in hard-rank growth, while Dion-1/16 preserves steadily increasing hard and soft effective ranks.

This is the sanity check that prevents the argument from collapsing into "one run trained longer." More training can improve loss without recovering the same width-to-capacity geometry.

<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure2_matched_loss_capacity_comparison.png' | relative_url }}" alt="Matched loss but different realized capacity">
  <figcaption><strong>Figure 2.</strong> <em>Matched loss, different realized capacity.</em> Extending AdamW training improves validation perplexity and brings it closer to the Dion-1/16 control, but the realized-capacity trajectories remain different. The hard-rank trajectory is especially diagnostic: closer loss does not imply matched internal geometry. The lesson is not that loss is unimportant; it is that loss alone does not tell us whether two training runs converted width into the same internal structure.</figcaption>
</figure>

This is the practical warning. If we only compare loss curves, we may conclude that two training runs have reached similar solutions. But if their internal capacity trajectories differ, they may not have built the same kind of model.

That difference can matter for downstream transfer, rare-token reliability, future adaptation, and interpretability. A model's loss tells us something about output error. It does not tell us whether the model used the same internal directions, reached the same kind of minimum, or preserved the same capacity for future learning.

For pretraining, this is a familiar problem in a sharper form. We already know that validation loss is an imperfect proxy for downstream behavior. The deeper issue is that loss is also an imperfect proxy for **the internal mechanism by which the model became good**.

## 4. Nominal capacity vs. realized capacity

The architecture–optimizer distinction becomes clearer if we separate two notions of capacity.

> **Nominal capacity** is the capacity implied by the architecture: parameter count, width, depth, heads, experts, routing paths, memory layout, and FLOPs.

> **Realized capacity** is the capacity actually used by the trained model: which representation directions become active, how spectral mass is distributed, which modes grow with width, and which data regimes receive usable internal structure.

Architecture certainly matters. It gives the learning system things optimization cannot create after the fact: causal masking, equivariance, sparse routing, inference-time compute patterns, memory movement, dimensional ceilings, residual topology, normalization placement, and parameter sharing. No optimizer can make a non-causal model causal, give a dense model sparse inference, or recover capacity that the architecture never admitted.

But architecture has a crucial limitation: **it creates available capacity; it does not guarantee realized capacity**. Adding FFN width raises the ceiling on representable structure, but it does not force training to fill those new directions. Increasing depth opens additional computation paths, but it does not guarantee that signal will route through them. Adding heads, experts, or low-rank structure changes the available function class without selecting which parts of it become load-bearing.

A useful mental model is:

<div class="math-box">
$$
C_{\mathrm{realized}}(\mathcal{A}, \mathcal{O}, \mathcal{D})
\;\approx\;
C_{\mathrm{available}}(\mathcal{A})
\;\times\;
\rho_{\mathrm{realized}}(\mathcal{O}, \mathcal{D}; \mathcal{A}).
$$
</div>

Here, $\mathcal{A}$ is the architecture, $\mathcal{O}$ is the optimizer/training algorithm, and $\mathcal{D}$ is the data. The equation is not meant to be a literal scalar law. It expresses a design principle: architecture sets the available capacity; optimization and data determine how much of that capacity becomes active, coherent, and useful.

The distinction is easy to lose if all we log is loss and parameter count:

<table>
<thead><tr><th>Observable</th><th>What it tells us</th><th>Mostly controlled by</th></tr></thead>
<tbody>
<tr><td>Parameter count</td><td>How many trainable degrees of freedom exist</td><td>Architecture</td></tr>
<tr><td>Width, depth, heads, experts</td><td>The nominal capacity ceiling</td><td>Architecture</td></tr>
<tr><td>FLOPs and memory movement</td><td>The cost and shape of computation</td><td>Architecture</td></tr>
<tr><td>Soft effective rank</td><td>How broadly representation mass spreads across modes</td><td>Training dynamics</td></tr>
<tr><td>Hard effective rank</td><td>How many directions become load-bearing</td><td>Architecture–optimizer pair</td></tr>
<tr><td>Capacity asymmetry</td><td>The gap between broad support and dominant-mode structure</td><td>Architecture–optimizer pair</td></tr>
<tr><td>Frequency-conditioned capacity</td><td>Which token regimes receive usable capacity</td><td>Data distribution × optimizer</td></tr>
<tr><td>Downstream/continued-learning behavior</td><td>Whether internal structure transfers or remains plastic</td><td>Model state × evaluation regime</td></tr>
</tbody>
</table>

This decomposition makes a concrete prediction: **the same architectural intervention can have different realized effects under different optimizers**.

If $\rho_{\mathrm{realized}}$ is high, added width translates cleanly into usable representation. If it is low, added width raises the ceiling without filling it. In the second regime, optimizer choice becomes first-order.

Modern LLMs increasingly operate in precisely the regime where this distinction matters: long-tail data, sparse supervision, multilingual and code-heavy distributions, deep residual computation, tool-use edge cases, and behaviors that are valuable but statistically rare.

## 5. Spectral telemetry for realized capacity

If realized capacity is the object we care about, we need measurements that see it.

Parameter count does not see it. FLOPs do not see it. Loss sees output error, but not the internal structure used to reduce that error.

Spectral geometry gives one practical telemetry layer.

Given a representation covariance, its eigenspectrum tells us how variance is distributed across directions. Effective-rank measures summarize this distribution; they sit in the broader information-theoretic tradition of treating capacity as effective dimensionality, with roots in rate-distortion theory ([Shannon, 1959](https://gwern.net/doc/cs/algorithm/information/1959-shannon.pdf)). The entropy-based effective-rank formulation traces back to Roy and Vetterli, and the broader Rényi family gives a principled way to vary the sensitivity to tail versus dominant modes ([Roy and Vetterli, 2007](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf); [Rényi, 1961](https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/On-Measures-of-Entropy-and-Information/chapter/On-Measures-of-Entropy-and-Information/bsmsp/1200512181)). These quantities are not a complete theory of capability, but they expose structure that scalar loss hides.

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

- **Soft effective rank** $R_{\mathrm{soft}} = R_1$ measures broad spectral support: how much representation mass is spread across many directions. I sometimes call this **Average-realized Capacity**.
- **Hard effective rank** $R_{\mathrm{hard}} = R_2$ measures how many directions carry substantial, load-bearing mass. I sometimes call this **Dominant-mode Capacity**.
- **Capacity asymmetry** is $\Delta = \log R_{\mathrm{soft}} - \log R_{\mathrm{hard}} = \log\!\left(R_{\mathrm{soft}} / R_{\mathrm{hard}}\right)$. It measures the multiplicative gap between broad support and dominant-mode structure.

<figure>
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure3_phase_diagram_for_realized_capacity.png' | relative_url }}" alt="Phase diagram for realized capacity">
  <figcaption><strong>Figure 3.</strong> <em>Conceptual phase diagram for realized capacity.</em> The horizontal axis tracks broad spectral support, using normalized $\log R_{\mathrm{soft}}$; the vertical axis tracks load-bearing dominant-mode capacity, using normalized $\log R_{\mathrm{hard}}$. The shaded forbidden region reflects the constraint $R_{\mathrm{hard}} \le R_{\mathrm{soft}}$. The low-asymmetry frontier marks the regime where broad support and load-bearing dominant modes are closely matched. The gap $\Delta = \log R_{\mathrm{soft}} - \log R_{\mathrm{hard}}$ measures capacity asymmetry: how much broad spectral support is not yet matched by load-bearing structure.</figcaption>
</figure>

This phase diagram separates cases that scalar loss can collapse together. A model can be collapsed in both soft and hard rank; compact but coherent; broad but weakly load-bearing; or distributed and usable. The asymmetry gap asks whether apparently broad capacity has become load-bearing structure.

That is why the distinction between soft and hard rank matters. Broad spectral support is useful, but it is not the whole story. In many regimes we also care about whether capacity becomes strong enough to carry signal, transfer, and support rare behaviors. The gap between support and load-bearing structure is precisely what capacity asymmetry tries to expose.

---

## 6. Why optimizers realize capacity

Optimization is not just the engine that minimizes loss. In overparameterized models, many solutions can fit the training data comparably well. The optimizer helps decide which of those solutions training actually finds.

This is the sense in which optimizers realize capacity: they shape which available directions grow, which spectral modes become load-bearing, which circuits become active, and which parts of the hypothesis space are reachable.

Architecture is the natural instrument for **hard inductive bias**. It restricts the support of the hypothesis space. Causal masking forbids attention to future tokens. Sparse MoE routing changes which parameters can activate for each token. Convolutional sharing enforces translation structure. These are not preferences; they are constraints. This hard-inductive-bias/soft-inductive-bias distinction is also consistent with the broader no-free-lunch/Kolmogorov-complexity view: learning succeeds when the learner's biases concentrate search on the low-complexity structure that real data tends to exhibit ([Goldblum et al., 2024](https://arxiv.org/abs/2304.05366); [Wilson, 2025](https://arxiv.org/abs/2503.02113)).

Optimization is the natural instrument for **soft inductive bias**. It does not usually remove hypotheses from the function class. Instead, it changes which hypotheses are easy to reach. AdamW, Muon, Shampoo, Dion, and related optimizers encode different geometric assumptions about parameters: coordinate-wise scaling, matrix-level orthogonalization, Kronecker-factored curvature, low-rank projected updates, and different forms of momentum and preconditioning.

The distinction can be summarized as support versus measure:

<table>
<thead><tr><th>Design object</th><th>Role</th><th>Practical effect</th></tr></thead>
<tbody>
<tr><td>Architecture</td><td>Hard inductive bias: restricts support</td><td>Determines what functions and computation patterns are possible</td></tr>
<tr><td>Optimizer</td><td>Soft inductive bias: reweights reachable solutions</td><td>Determines which solutions are likely to be found</td></tr>
<tr><td>Data</td><td>Likelihood/evidence</td><td>Determines where the solution is sharply constrained or underdetermined</td></tr>
<tr><td>Trained model</td><td>Joint outcome</td><td>The actual computation implemented by the learned weights</td></tr>
</tbody>
</table>

A useful analogy — not a literal Bayesian claim — is:

<div class="math-box">
$$
p(h \mid \mathcal{D}, \mathcal{A}, \mathcal{O})
\;\propto\;
p(\mathcal{D} \mid h)\,
p_{\mathcal{O}}(h \mid \mathcal{A}).
$$
</div>

Here, $\mathcal{A}$ is the architecture, $\mathcal{O}$ is the optimizer/training algorithm, $\mathcal{D}$ is the data, and $h$ is the trained hypothesis. Architecture acts like the support of the hypothesis space; the optimizer induces something like a prior over reachable solutions inside that support; data acts like the likelihood.

AdamW comes from decoupled weight decay for adaptive optimization ([Loshchilov and Hutter, 2019](https://arxiv.org/abs/1711.05101)); Shampoo is a structure-aware tensor preconditioner ([Gupta, Koren, and Singer, 2018](https://arxiv.org/abs/1802.09568)); Muon was introduced as a matrix-orthogonalized optimizer in an original public write-up and has since been studied for LLM scaling ([Jordan et al., 2024](https://kellerjordan.github.io/posts/muon/); [Liu et al., 2025](https://arxiv.org/abs/2502.16982)); NorMuon combines Muon-style orthogonalization with neuron-wise adaptive rates ([Li et al., 2025](https://arxiv.org/abs/2510.05491)); and Dion studies distributed orthonormalized updates with low-rank communication structure ([Ahn et al., 2025](https://arxiv.org/abs/2504.05295)). These choices change the path through parameter space, the effective prior over solutions, and the spectral structure that emerges in representations.

This is also why architecture search under a fixed optimizer can be misleading. An architecture that appears unstable or underperforming under one optimizer may become viable under another. Conversely, an architecture that looks strong under AdamW may be partly strong because AdamW and that architecture form a historically well-tuned pair.

The trainable frontier is therefore not the set of architectures alone. It is the set of viable architecture–optimizer pairs.

---

## 7. Effective graphs, circuits, and reachability

The Transformer architecture gives us a static operator graph: attention blocks, MLP blocks, residual additions, layer normalizations, and projection matrices ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)). But the trained model is not just this static graph.

Inside it is an **effective signal-flow graph**: the set of routes through heads, MLP directions, residual-stream features, and layer compositions that actually carry meaningful signal for a behavior. This framing is closely related to the transformer-circuits view of a Transformer as a composition of residual-stream paths and attention/MLP components ([Elhage et al., 2021](https://transformer-circuits.pub/2021/framework/index.html)).

<table>
<thead><tr><th>Graph</th><th>Fixed by</th><th>What it contains</th><th>What it misses</th></tr></thead>
<tbody>
<tr><td>Static operator graph</td><td>Architecture</td><td>Matmuls, attention, MLPs, residual additions, normalization placement</td><td>Which routes actually carry meaningful signal after training</td></tr>
<tr><td>Effective signal-flow graph</td><td>Architecture × optimizer × data</td><td>Load-bearing paths, active representation directions, behavior-specific circuits</td><td>Formal paths that exist but carry negligible signal</td></tr>
</tbody>
</table>

Two models with the same architecture can therefore have different effective graphs. The same formal paths exist in both models, but different paths become load-bearing, and even the same path can carry signal with different spectral shape. One optimizer may fragment a rare signal across coordinate-wise statistics. Another may preserve it as a coherent matrix-level direction.

This matters for interpretability. A circuit found in an AdamW-trained model is not automatically a circuit of the architecture. It may be a circuit of the architecture–optimizer pair. Change the optimizer, and the same architecture may build a different effective computation for the same behavior.

There is also a reachability issue. A function may be representable by the architecture but effectively unreachable under a given optimizer and schedule. In that case, the optimizer has not changed the formal hypothesis space, but it has changed the subset of that space training can actually access.

This is one of the reasons I prefer the language of **realized capacity** to raw capacity. It points to the model that training actually produced, not only the model class we wrote down.

---

## 8. Rare-token regimes expose optimizer-induced capacity allocation

Natural language is not statistically uniform. Token frequency follows a long-tailed, approximately Zipfian distribution. A small number of HEAD tokens receive enormous gradient mass. A large number of TAIL tokens receive sparse, noisy, and intermittent supervision.

This matters because the role of the optimizer changes with the statistical regime.

For HEAD tokens, the likelihood is sharp: many reasonable optimizers can find similar solutions because the data strongly constrains what should be learned. For TAIL tokens, the likelihood is weak: many solutions remain compatible with sparse evidence, and the implicit bias of the optimizer matters much more.

In the Bayesian analogy:

> Where data is dense, the likelihood dominates. Where data is sparse, the prior dominates. The optimizer acts like a soft inductive bias over reachable solutions.

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
- rare code libraries, identifiers, and scientific terminology,
- long-tail factual recall and tool-use edge cases,
- structured reasoning patterns and rare instruction formats,
- expert specialization in sparse MoE settings.

If a model's average loss is dominated by frequent patterns, then loss can hide whether rare regimes are receiving usable representational capacity. Frequency-conditioned spectral telemetry is one way to expose that hidden allocation.

Figure 4 resolves the aggregate result from Figure 1 into HEAD, MID, and TAIL regimes. The left panel shows dominant-mode capacity scaling directly through $\beta_{\mathrm{hard}}$. This is the most important long-tail signal: in TAIL representations, AdamW and Dion-1/16 have much weaker hard-rank scaling, while Muon and NorMuon approach near-linear dominant-mode capacity growth. The right panel shows the corresponding soft–hard scaling asymmetry.

<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure4_capacity_scaling_asymmetry_token_regimes.png' | relative_url }}" alt="Frequency-conditioned realized-capacity scaling across token regimes">
  <figcaption><strong>Figure 4.</strong> <em>Frequency-conditioned realized-capacity scaling.</em> Panel A shows hard-rank scaling exponents, $\beta_{\mathrm{hard}}$, across HEAD, MID, and TAIL token regimes. Panel B shows soft–hard scaling asymmetry, $\Delta\beta = \beta_{\mathrm{soft}} - \beta_{\mathrm{hard}}$. The regime-conditioned view reveals where the optimizer effect becomes most consequential: in MID and especially TAIL regimes, where sparse supervision gives optimizer-induced bias more room to shape which representation directions become load-bearing.</figcaption>
</figure>

The rare-token result is not a side observation. It is one of the main reasons this matters for frontier LLMs. The behaviors we increasingly care about are not only high-frequency language modeling behaviors. They often live in sparse, structured, high-value parts of the data distribution.

A capacity-aware training run should therefore not only ask whether loss improved on average. It should ask whether the long tail received usable internal structure.

## 9. Why co-design is the right abstraction

The conclusion is not that optimizers matter more than architectures. They do different jobs.

Architecture is the right tool when we need hard structure:

- enforce causal or equivariant constraints,
- increase width or depth,
- add experts, sparsity, or routing structure,
- change memory and compute patterns,
- reduce inference cost.

Optimization is the right tool when we need to decide how capacity is filled:

- which directions grow and which modes become dominant,
- whether capacity spreads or concentrates,
- whether rare signals remain coherent,
- which minima are reachable,
- how the same architecture behaves across data regimes.

Both are necessary because they operate at different levels: architecture sets the support, the optimizer reweights it.

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

The historical default for LLMs has been a strong and productive equilibrium: dense Transformer architecture plus AdamW-like optimization plus carefully tuned data and schedules. That recipe worked extremely well. But it also made it easy to mistake properties of a particular architecture–optimizer pair for properties of the architecture alone.

Co-design is the right abstraction because the object we deploy is not an architecture. It is a trained model. And the trained model is the joint outcome of architecture, data, optimizer, schedule, initialization, regularization, and scale.

If two optimizers convert the same FFN width into different realized capacity, then a scaling decision is incomplete unless it specifies the optimizer. Likewise, an optimizer evaluation is incomplete if it reports speed and loss but not the internal capacity geometry it induces.

---

## 10. What should change in LLM practice?

The practical message is not "replace loss." Loss remains the central external training signal. The point is to add internal telemetry that tells us what kind of solution training is building.

A capacity-aware pretraining run should make several comparisons routine:

- **Compare optimizers at matched loss**, not only at matched steps or wall-clock time.
- **Log soft and hard effective ranks** during width, depth, and optimizer sweeps.
- **Inspect frequency-conditioned capacity**, especially MID and TAIL regimes.
- **Diagnose wasted width**, where parameters increase but hard-rank scaling remains weak.
- **Treat optimizer choice as part of model design**, not as an implementation detail fixed after architecture is chosen.
- **Look for architecture–optimizer interactions**, because the same architecture can have different realized capacity under different training dynamics.

These are not expensive philosophical additions. They are telemetry. The goal is to know whether a training run is converting added architectural budget into internal structure that can carry signal, transfer, and remain adaptable.

## 11. Broader research agenda

The architecture–optimizer view is part of a broader shift in how modern AI systems are being studied: from static artifacts to time-evolving training processes.

### Scaling laws

Classical scaling laws predict loss from parameters, data, and compute ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361); [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)). That has been enormously useful. But loss-based scaling laws do not tell us whether added parameters became usable internal dimensions, whether capacity was allocated to rare regimes, or whether two same-loss models reached the same representation geometry.

A capacity-aware scaling law would not replace loss scaling. It would complement it with internal scaling variables: how soft rank, hard rank, asymmetry, and frequency-conditioned capacity scale with width, depth, optimizer, and data.

### Plasticity and future learnability

Continual learning provides a useful analogy. Wider networks can reduce catastrophic forgetting by giving tasks more room and reducing gradient interference ([Mirzadeh et al., 2022](https://arxiv.org/abs/2110.11526)), but future learnability is not determined by raw width alone. Work on loss of plasticity suggests that trainability can degrade through changes in activation structure, feature-rank collapse, spectral collapse, optimizer dynamics, and regularization ([Abbas et al., 2023](https://arxiv.org/abs/2303.07507); [Dohare et al., 2024](https://www.nature.com/articles/s41586-024-07711-7); [He et al., 2025](https://arxiv.org/abs/2509.22335)).

This reinforces the architecture–optimizer view: width and depth define potential capacity, but learning dynamics determine how much of that capacity remains realized, usable, and adaptable.

### Interpretability and circuits

If optimizers change the effective signal-flow graph, then circuits may be optimizer-conditional. A behavior discovered in one trained model may depend not only on the Transformer architecture, but on the training dynamics that made particular paths load-bearing.

This suggests a sharper interpretability question: which circuits are architecture-stable, and which are training-dynamics-specific?

### Training dynamics as the object of study

Recent position work makes the methodological point explicit: AI systems should be studied as time-evolving training processes, not only as static artifacts to analyze or patch after training. Scaling laws have made loss prediction routine; the harder challenge is building predictive accounts of capabilities, biases, robustness, and safety-relevant behavior as they emerge through training ([Biderman et al., 2026](https://arxiv.org/abs/2606.06533)).

Optimizer-induced spectral scaling laws fit naturally into this agenda. They track one measurable part of the coupling between architecture, data, and optimization dynamics: how training converts nominal capacity into realized internal structure.

## 12. Toward capacity-aware LLM design

This is the core claim behind optimizer-induced spectral scaling laws. If two optimizers train the same architecture but produce different spectral-capacity scaling, then the optimizer has changed the model's realized capacity. The matched-loss control in Figure 2 strengthens the point: even improving or matching loss does not guarantee matching internal geometry. Figure 4 shows where this becomes most consequential: MID and TAIL regimes, where optimizer-induced bias can shape how capacity is allocated across the data distribution.

That is a design axis.

The next generation of LLM design should not only ask how many parameters we train, how many tokens we consume, or how low the loss goes. It should also ask what capacity becomes real — and where in the data distribution that capacity becomes usable.

> **Capacity-aware LLM design means tracking not just what the model could represent, but what training actually made it represent.**

The long-term version of this agenda is not just better plots. It is a more complete science of pretraining: one that connects external scaling laws to internal representation geometry, optimizer-induced bias, long-tail capability, transfer, and future learnability.

Architecture creates the space of possible computation. Optimization decides which parts of that space become real.

## What this does not claim

There are several ways to overstate this argument, so it is worth being explicit.

First, spectral rank is not a complete theory of intelligence, generalization, or downstream ability. It is a telemetry signal. It should be interpreted alongside loss, evaluations, robustness tests, calibration, interpretability, and application-specific metrics. We are not claiming that spectral rank alone predicts capability; we are claiming that it exposes an internal axis that loss and parameter count do not measure.

Second, optimizer-induced spectral capacity is not automatically good. More capacity is not always better. The right question is where capacity appears, whether it is stable, whether it supports useful behaviors, and whether it improves generalization in the regimes that matter.

Third, architecture remains indispensable. Optimizers cannot create structural guarantees, reduce inference cost by construction, or represent functions excluded by the architecture. Co-design is not optimizer maximalism.

Fourth, the strongest claims require more evidence: larger models, more architectures, longer training, downstream probes, continued-learning tests, interpretability comparisons, and direct studies of rare-regime behavior.

## Open questions

This framing leads to several questions that I think are worth treating as first-class pretraining-science problems.

**1. Do optimizer-induced spectra predict downstream behavior?**  
If two models have similar loss but different realized capacity, which spectral differences predict transfer, robustness, rare-token reliability, or domain adaptation?

**2. Are mechanistic circuits optimizer-conditional?**  
If the optimizer changes the effective signal-flow graph, then circuits discovered in one trained model may not be stable across optimizers, even with the same architecture.

**3. How should we search over architecture–optimizer pairs?**  
Neural architecture search typically fixes the optimizer. Optimizer evaluation typically fixes the architecture. The co-design view suggests this may miss regions where neither component looks optimal alone, but the pair is strong.

**4. Can realized-capacity telemetry improve scaling laws?**  
A capacity-aware scaling law would track how internal representation structure scales, and whether added parameters become usable degrees of freedom.

**5. What is the connection to plasticity and future learnability?**  
A model with similar loss but different spectral allocation may differ in its ability to keep learning. If realized capacity measures active degrees of freedom, then capacity telemetry may also help diagnose loss of plasticity, representational collapse, or the exhaustion of useful directions during continued training. Recent work even reframes plasticity itself as a measure of empowerment over future learning trajectories, suggesting that *what training preserves* may be as important as what it minimizes ([Abel et al., 2025](https://arxiv.org/abs/2505.10361)).

## References

### Scaling laws and Transformer architecture

- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. *Scaling Laws for Neural Language Models*. arXiv:2001.08361, 2020. [`arXiv`](https://arxiv.org/abs/2001.08361)
- Hoffmann, J. et al. *Training Compute-Optimal Large Language Models*. NeurIPS, 2022. [`arXiv`](https://arxiv.org/abs/2203.15556)
- Vaswani, A. et al. *Attention Is All You Need*. NeurIPS, 2017. [`arXiv`](https://arxiv.org/abs/1706.03762)

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
