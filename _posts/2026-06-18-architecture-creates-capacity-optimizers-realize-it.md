---
layout: post
title: "Architecture Creates Capacity, Optimizers Realize It."
subtitle: "Why LLM scaling laws should account for realized capacity, not just loss curves and parameter counts"
author: "Nandan Kumar Jha"
date: 2026-06-18
permalink: /blog/optimizer-induced-capacity/
description: "Why LLM scaling should account for realized capacity: the same architecture and training data can produce different internal capacity under different optimizers."
reading_time: "~28 min read"
image: /assets/blog/architecture-optimizer-codesign/figure0_same_architecture_same_data_different_optimizer.png
tags:
  - scaling-laws
  - optimization
  - representation-learning
  - llm-pretraining
  - spectral-geometry
  - continual-learning
---
> **Core thesis.** Architecture defines the degrees of freedom a model *could* use. Optimization helps determine which of those degrees of freedom become realized, *as variance-carrying directions*, during training.


<style>
.tldr-box {
  background: #f8fafc;
  border-left: 4px solid #334155;
  padding: 1.1rem 1.25rem;
  margin: 1.25rem 0 1.75rem 0;
  border-radius: 8px;
}
.tldr-box p {
  margin-top: 0;
  margin-bottom: 0.85rem;
}
.tldr-box ul {
  margin-bottom: 0;
  padding-left: 1.25rem;
}
.tldr-box li {
  margin-bottom: 0.75rem;
}
.tldr-box li:last-child {
  margin-bottom: 0;
}

.toc-box {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 0.85rem 1.05rem;
  margin: 1.4rem 0 1.9rem 0;
}
.toc-box summary {
  cursor: pointer;
  font-weight: 700;
  color: #0f172a;
}
.toc-box ol {
  margin: 0.75rem 0 0 1.25rem;
  padding-left: 0.75rem;
}
.toc-box ol ol {
  margin-top: 0.35rem;
  margin-bottom: 0.35rem;
}
.toc-box li {
  margin: 0.35rem 0;
}
.toc-box a {
  text-decoration: none;
}
.toc-box a:hover {
  text-decoration: underline;
}
</style>


## TL;DR
{: #tldr }

<div class="tldr-box">

<p><strong>Capacity is not just what the architecture makes possible; it is what training actually makes usable.</strong></p>

<ul>
  <li><strong>Finding.</strong> Holding architecture, training data, tokenizer, and FFN-width schedule fixed, optimizer choice changes the internal spectral capacity realized inside FFN representations.</li>

  <li><strong>Matched loss is not matched representation.</strong> Across FFN-width configurations, models trained with the same architecture but different optimizers can reach similar validation loss while learning different internal representations. Longer AdamW training matches the validation loss of Dion-1/16, but does not recover the dominant-mode capacity scaling achieved by Dion-1/16.</li>

  <li><strong>Implication.</strong> Architecture sets the representational degrees of freedom; optimization helps decide which become active, variance-carrying directions, and how they are allocated across token-frequency regimes. This matters most for rare tokens, where training data provides weaker constraints and optimizer-induced bias has more room to shape which directions grow. Capacity is therefore realized by the architecture–optimizer pair, not by architecture alone.</li>
</ul>

</div>

That is the controlled fact behind this post. Scaling laws made us good at asking how loss changes with parameters, data, and compute. They left a quieter question under-measured: when we add architectural capacity to a model, does training actually use it?

Our experiments suggest that the answer depends strongly on the optimizer. The trained models realize different spectral scaling laws inside their FFN representations, especially in rare-token regimes where supervision is sparse and optimizer-induced bias has more room to decide which directions grow.

The point is not that loss stops mattering. The point is that **optimization is not a neutral procedure that merely fills a fixed architecture. In overparameterized LLMs, optimization helps decide what capacity becomes real.**

<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure0_same_architecture_same_data_different_optimizer.png' | relative_url }}" alt="Same architecture, same data, different optimizer conceptual setup">
  <figcaption><strong>Figure 0.</strong> <em>Same architecture, same data, different optimizer.</em> The model family, data distribution, and training task are held fixed. Optimizer choice changes the path through parameter space, and those different training dynamics can lead to different internal spectral-capacity profiles, even when final loss is similar.</figcaption>
</figure>

<details class="toc-box">
  <summary>On this page</summary>
  <ol>
    <li><a href="#same-architecture-different-capacity-scaling">Same architecture, different capacity scaling</a></li>
    <li><a href="#matched-loss-is-not-matched-geometry">Matched loss is not matched representation</a></li>
    <li><a href="#rare-tokens-expose-optimizer-induced-capacity-allocation">Rare-token capacity allocation</a></li>
    <li>
      <a href="#five-views-of-the-same-gap">Five views of the same gap</a>
      <ol>
        <li><a href="#view-i-upstream-downstream-gap">Upstream–downstream gap</a></li>
        <li><a href="#view-ii-rate-distortion">Rate–distortion</a></li>
        <li><a href="#view-iii-realized-capacity">Realized capacity</a></li>
        <li><a href="#view-iv-hard-and-soft-inductive-bias">Hard and soft inductive bias</a></li>
        <li><a href="#view-v-plasticity-and-continual-learning">Plasticity and continual learning</a></li>
      </ol>
    </li>
    <li><a href="#why-optimizers-realize-capacity">Why optimizers realize capacity</a></li>
    <li><a href="#the-design-object-is-the-architecture-optimizer-pair">Architecture–optimizer co-design</a></li>
    <li><a href="#what-changes-in-pretraining-practice">What changes in pretraining practice?</a></li>
    <li><a href="#conclusion-toward-capacity-aware-llm-design">Conclusion</a></li>
    <li><a href="#what-this-does-not-claim">What this does not claim</a></li>
    <li><a href="#open-questions">Open questions</a></li>
  </ol>
</details>

## 1. Same architecture, different capacity scaling
{: #same-architecture-different-capacity-scaling }

The result behind this post is simple:

> **When architecture, training data, tokenizer, and width schedule are fixed, changing the optimizer changes how representational capacity scales.**

For the first pass through the evidence, only three terms are needed.

<div class="metric-box">
  <strong>Metrics used in this post.</strong>
  <ul>
    <li><strong>Average/diffuse capacity</strong>: soft spectral rank; how broadly representation variance spreads across eigenmodes.</li>
    <li><strong>Dominant-mode capacity</strong>: hard spectral rank; how many strong eigenmodes carry substantial variance.</li>
    <li><strong>Capacity asymmetry</strong>: the gap between diffuse and dominant-mode capacity.</li>
    <li><strong>Scaling exponent</strong>: how quickly realized capacity grows as FFN width increases.</li>
  </ul>
</div>

In the paper, we vary FFN width under the same model family and compare how spectral effective ranks grow. The optimizer changes the slope of that growth. The model does not merely train faster or slower; it converts the same architectural budget into a different internal capacity profile.

Figure 1 gives the aggregate view. The important quantity is not just final rank value, but the **scaling exponent**: how much additional FFN width becomes additional realized spectral capacity. In the TAIL regime, AdamW has weak dominant-mode capacity scaling ($\beta_{\mathrm{hard}} \approx 0.44$), while Muon and NorMuon approach near-linear dominant-mode capacity scaling ($\beta_{\mathrm{hard}} \approx 1.0$). That is roughly a $2.3\times$ larger exponent under the same architecture and data.

<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure1_optimizer_capacity_scaling.png' | relative_url }}" alt="Aggregated optimizer-level realized-capacity scaling">
  <figcaption><strong>Figure 1.</strong> <em>Aggregated optimizer-level realized-capacity scaling.</em> Architecture, training data, and FFN-width schedule are fixed; only the optimizer changes. Panel A shows aggregated scaling exponents for average/diffuse capacity, $\beta_{\mathrm{soft}}$, and dominant-mode capacity, $\beta_{\mathrm{hard}}$. Panel B shows capacity asymmetry, $\Delta\beta = \beta_{\mathrm{soft}} - \beta_{\mathrm{hard}}$. Muon and NorMuon show stronger dominant-mode capacity scaling and lower asymmetry than AdamW, indicating that more added width appears in dominant eigenmodes. Figure 3 later resolves where this effect is strongest across HEAD, MID, and TAIL regimes.</figcaption>
</figure>

The full result tables, frequency-conditioned fits, and additional ablations are on the [project page](https://optimizer-scaling-laws.github.io/). The blog's role is interpretive: to explain why these scaling differences matter for how we think about architecture, optimization, and realized capacity.

A scale caveat is important. These are controlled GPT-2-scale studies, including 160M and 350M model families on FineWeb-Edu. That is large enough to test whether optimizer choice can change internal capacity-scaling exponents under matched architecture and training data. It is not large enough to claim that the same optimizer ordering must persist unchanged at multi-billion-parameter scale. Concurrent optimizer-scaling work also suggests that preconditioned optimizers can change apparent scaling behavior, while their advantage may attenuate at larger LLM scale ([Ramani and Jain, 2026](https://arxiv.org/abs/2605.29387)).

The senior question is therefore not, "does this settle frontier scaling?" It is: **what trend survives as we move from 160M to 350M, what attenuates at larger scale, and what experiment would decide it?** The next clean experiment is a calibrated billion-scale sweep that keeps architecture, training data, tokenizer, width schedule, and compute accounting fixed while measuring whether average/diffuse capacity, dominant-mode capacity, and HEAD/MID/TAIL effects persist.

The narrow claim is precise: **optimizer choice can change the internal scaling law by which width becomes usable representation.** That is enough to make it a first-class object in pretraining science.

<p class="takeaway-inline"><strong>Takeaway.</strong> Optimizer choice changes not only training speed or final loss, but the scaling law by which added FFN width becomes realized internal capacity.</p>

## 2. Matched loss is not matched representation
{: #matched-loss-is-not-matched-geometry }

A natural objection is that one optimizer may simply be training faster. The matched-loss control addresses that shortcut.

Extending AdamW training from 6K to 12K improves validation perplexity and brings it close to the low-rank Dion-1/16 control across the FFN-width sweep. But the realized-capacity trajectories do not match. AdamW-12K remains much weaker in dominant-mode capacity growth, while Dion-1/16 preserves steadily increasing dominant-mode and average/diffuse capacity.

The sharper point is counterintuitive: in this matched-loss control, training AdamW longer improves loss while weakening the dominant-mode capacity scaling trend. The dominant-mode capacity exponent drops from about $0.29$ to about $0.03$. Wider AdamW models are not simply "catching up" internally as loss improves. On this diagnostic, they lose usable dominant-mode capacity scaling.

<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure2_matched_loss_capacity_comparison.png' | relative_url }}" alt="Matched loss but different realized capacity">
  <figcaption><strong>Figure 2.</strong> <em>Matched loss, different realized capacity.</em> Extending AdamW training improves validation perplexity and brings it closer to the Dion-1/16 control, but the realized-capacity trajectories remain different. The dominant-mode capacity trajectory is especially diagnostic: closer loss does not imply matched internal geometry.</figcaption>
</figure>

The full matched-loss comparison and accompanying fits are available on the [project page](https://optimizer-scaling-laws.github.io/). The gap is also not closed by learning-rate tuning: it survives a learning-rate sweep reported with the project results.

This is the practical warning. If we only compare loss curves, we may conclude that two training runs reached similar solutions. But if their internal capacity trajectories differ, they may not have built the same kind of model. Loss tells us something about output error. It does not tell us whether the model used the same internal directions, reached the same kind of minimum, or preserved the same capacity for future adaptation.

<p class="takeaway-inline"><strong>Takeaway.</strong> Matched validation loss can still hide different width-to-capacity trajectories. Loss matching is necessary for a fair comparison, but it is not enough to establish matched internal geometry.</p>

## 3. Rare tokens expose optimizer-induced capacity allocation
{: #rare-tokens-expose-optimizer-induced-capacity-allocation }

Natural language is long-tailed. A small number of HEAD tokens receive enormous gradient mass. A large number of TAIL tokens receive sparse, noisy, intermittent supervision.

That changes the role of optimization. For HEAD tokens, the data strongly constrains what should be learned. Many reasonable optimizers can find similar solutions. For TAIL tokens, the evidence is weaker: many internal organizations remain compatible with sparse observations, and the optimizer's implicit bias has more room to shape which representation directions grow.

In the Bayesian analogy: where data is dense, the likelihood dominates; where data is sparse, the prior matters more. The optimizer acts like a soft inductive bias over reachable solutions.

<table>
<thead><tr><th>Token regime</th><th>Data condition</th><th>Dominant bottleneck</th><th>Expected stronger lever</th></tr></thead>
<tbody>
<tr><td>HEAD</td><td>Dense, frequent, well-conditioned supervision</td><td>Architectural ceiling and compute pattern</td><td>Architecture</td></tr>
<tr><td>MID</td><td>Partially constrained, mixed signal quality</td><td>Capacity allocation across modes</td><td>Architecture–optimizer pair</td></tr>
<tr><td>TAIL</td><td>Sparse, noisy, underdetermined supervision</td><td>Signal preservation and implicit prior</td><td>Optimizer</td></tr>
</tbody>
</table>

This predicts that optimizer-induced differences should be more visible in MID and TAIL regimes than in HEAD regimes. It also explains why architecture-only changes can under-deliver for long-tail behavior: adding capacity raises the ceiling, but sparse data does not automatically fill it. The optimizer must preserve and amplify weak joint signals rather than fragment them.

Many frontier-relevant LLM behaviors live in this sparse, high-value part of the distribution: low-resource languages, rare code libraries, identifiers, scientific terminology, long-tail factual recall, tool-use edge cases, unusual instruction formats, and expert specialization in sparse MoE settings.

Figure 3 resolves the aggregate result from Figure 1 into HEAD, MID, and TAIL regimes. The left panel shows dominant-mode capacity scaling directly through $\beta_{\mathrm{hard}}$. In TAIL representations, AdamW and Dion-1/16 have much weaker dominant-mode capacity scaling, while Muon and NorMuon approach near-linear dominant-mode growth. The right panel shows the corresponding average/diffuse–dominant capacity asymmetry.

<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure4_capacity_scaling_asymmetry_token_regimes.png' | relative_url }}" alt="Frequency-conditioned realized-capacity scaling across token regimes">
  <figcaption><strong>Figure 3.</strong> <em>Frequency-conditioned realized-capacity scaling.</em> Panel A shows dominant-mode capacity scaling exponents, $\beta_{\mathrm{hard}}$, across HEAD, MID, and TAIL token regimes. Panel B shows average/diffuse–dominant capacity asymmetry, $\Delta\beta = \beta_{\mathrm{soft}} - \beta_{\mathrm{hard}}$. The regime-conditioned view reveals where the optimizer effect becomes most consequential: MID and especially TAIL regimes, where sparse supervision gives optimizer-induced bias more room to shape which eigenmodes accumulate variance.</figcaption>
</figure>

The rare-token result is not a side observation. It is one of the main reasons this matters for frontier LLMs. If average loss is dominated by frequent patterns, loss can hide whether rare regimes receive usable representational capacity. Frequency-conditioned spectral telemetry exposes that hidden allocation.

A capacity-aware training run should therefore not only ask whether loss improved on average. It should ask whether the long tail received usable internal structure.

<p class="takeaway-inline"><strong>Takeaway.</strong> Rare-token regimes are where optimizer-induced bias has the most room to shape which weak signals become coherent representation directions.</p>

## 4. Five views of the same gap
{: #five-views-of-the-same-gap }

The empirical result is not an isolated optimizer anecdote. It belongs to a broader pattern: scalar training success does not fully describe the internal state of the learned model. The common warning is that loss and architecture alone do not specify the representation that training produced.

Optimizer-induced spectral scaling gives this warning a concrete pretraining-science object: the width-to-capacity conversion law of a fixed architecture can change with the optimizer. The map below shows the role each view plays in the argument.

<style>
.view-map {
  width: 100%;
  border-collapse: collapse;
  margin: 1.1rem 0 1.75rem 0;
  font-size: 0.95rem;
}
.view-map th {
  text-align: left;
  padding: 0.65rem 0.75rem;
  border-bottom: 1px solid #cbd5e1;
  background: #f8fafc;
}
.view-map td {
  padding: 0.65rem 0.75rem;
  border-bottom: 1px solid #e2e8f0;
  vertical-align: top;
}
.view-map td:first-child {
  font-weight: 600;
  white-space: nowrap;
}
</style>

<table class="view-map">
<thead>
<tr>
<th>View</th>
<th>Main lesson</th>
<th>What it adds here</th>
</tr>
</thead>
<tbody>
<tr>
<td>Upstream–downstream gap</td>
<td>Same loss is not the same useful representation</td>
<td>Loss alone does not identify the representation learned for downstream tasks.</td>
</tr>
<tr>
<td>Rate–distortion</td>
<td>One objective can hide different internal structures</td>
<td>Spectral capacity gives a missing internal axis beyond scalar prediction error.</td>
</tr>
<tr>
<td>Realized capacity</td>
<td>Nominal width is not usable representation capacity</td>
<td>Width matters only if training converts it into active representation directions.</td>
</tr>
<tr>
<td>Hard/soft inductive bias</td>
<td>Same architecture does not mean same reachable solution</td>
<td>The architecture–optimizer pair is the real design object.</td>
</tr>
<tr>
<td>Plasticity</td>
<td>Realized capacity must remain adaptable for future tasks</td>
<td>Capacity matters for future learning, not only current loss.</td>
</tr>
</tbody>
</table>

### View I — Upstream–downstream gap: same loss is not the same useful representation
{: #view-i-upstream-downstream-gap }

Loss curves, gradient norms, learning rates, tokens/sec, memory, and downstream evaluations are useful signals. They are not complete descriptions of the trained model.

The reason is simple: **the same loss can be reached through different internal organizations of the model**. A language model can reduce next-token error by spreading variance across many directions, concentrating prediction-relevant variance into fewer dominant modes, or allocating capacity unevenly across token-frequency regimes.

Prior work makes this concern concrete. Liu et al. show that pretraining loss does not fully explain downstream performance: among models with the same pretraining loss, changing the training algorithm can produce different downstream transfer ([Liu et al., 2022](https://arxiv.org/abs/2210.14199)). Optimizer-level work gives a geometric version of the point: same-loss pretraining can converge to solutions with different relationships among task-specific minima, and optimizer design can bias training toward solutions that transfer better ([Chen et al., 2026](https://arxiv.org/abs/2604.09258)). Representation-geometry studies give the same warning from another angle: scalar metrics can miss non-monotonic changes in learned geometry during pretraining and post-training ([Li et al., 2025](https://arxiv.org/abs/2509.23024)).

There is also a theoretical reason to expect this gap. Wu, Lee, and Ge show that low pretraining loss alone does not guarantee that every downstream-relevant feature is recoverable from the representation; downstream transfer depends on properties of the learned representation and the task, including weakly constrained or low-probability structure ([Wu, Lee, and Ge, 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/93712c59f6a81bd92040facf04c8b308-Abstract-Conference.html)).

Our result is a spectral-capacity version of the same warning. If two optimizers can reach comparable loss while realizing different dominant-mode capacity scaling, then pretraining loss is not a sufficient description of the learned representation. The question is not only **how low did the loss go?** It is also:

> **What internal capacity did training realize in order to reach that loss?**

### View II — Rate–distortion: one objective can hide different internal structures
{: #view-ii-rate-distortion }

A useful analogy comes from rate–distortion thinking. Spectral rank is not literally rate: classical rate is a KL-style encoding cost for a latent variable relative to a marginal distribution, while spectral rank measures how variance is distributed across eigendirections.

The useful analogy is narrower: **a scalar objective can hide distinct internal representation states**.

Alemi et al. made this concrete for VAEs: identical ELBO values can correspond to qualitatively different points in the rate–distortion plane ([Alemi et al., 2018](https://proceedings.mlr.press/v80/alemi18a.html)). LLM pretraining has a similar failure mode. Validation loss can tell us that the model predicts well on average without telling us what representation it used to get there. Two models may have similar loss while allocating variance differently across eigenmodes, layers, or token-frequency regimes.

Spectral capacity is one way to expose that hidden coordinate. It does not replace loss, just as rate–distortion plots do not replace the ELBO. It adds a second axis: not only how well the model predicts, but what internal structure training built to make those predictions possible.

### View III — Realized capacity: nominal width is not usable representation capacity
{: #view-iii-realized-capacity }

The architecture–optimizer distinction becomes clearest if we separate two notions of capacity.

> **Nominal capacity** is the capacity implied by the architecture: parameter count, width, depth, heads, experts, routing paths, memory layout, and FLOPs.

> **Realized capacity** is the capacity actually used by the trained model: which representation directions become active, how variance is distributed across eigenmodes, which modes grow with width, and which data regimes receive usable internal structure.

Architecture gives the learning system things optimization cannot create after the fact: causal masking, equivariance, sparse routing, inference-time compute patterns, dimensional ceilings, residual topology, normalization placement, and parameter sharing. But architecture has a crucial limitation: it creates available capacity; it does not guarantee that training will fill those degrees of freedom.

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

Here, $\mathcal{A}$ is the architecture, $\mathcal{O}$ is the optimizer/training algorithm, and $\mathcal{D}$ is the data. This is not a literal scalar law. It is a design principle: architecture sets available capacity; optimization and data determine how much of that capacity becomes active, coherent, and useful.

<table>
<thead><tr><th>Observable</th><th>What it tells us</th><th>Mostly controlled by</th></tr></thead>
<tbody>
<tr><td>Parameter count, width, depth, heads</td><td>The nominal capacity ceiling</td><td>Architecture</td></tr>
<tr><td>FLOPs and memory movement</td><td>The cost and shape of computation</td><td>Architecture</td></tr>
<tr><td>Average/diffuse capacity</td><td>How broadly variance is distributed across eigenmodes</td><td>Training dynamics</td></tr>
<tr><td>Dominant-mode capacity</td><td>How many dominant eigenmodes carry substantial variance</td><td>Architecture–optimizer pair</td></tr>
<tr><td>Capacity asymmetry</td><td>The gap between diffuse and dominant-mode capacity</td><td>Architecture–optimizer pair</td></tr>
<tr><td>Frequency-conditioned capacity</td><td>Which token regimes receive usable capacity</td><td>Data distribution × optimizer</td></tr>
</tbody>
</table>

This decomposition predicts that **the same architectural intervention can have different realized effects under different optimizers**. If $\rho_{\mathrm{realized}}$ is high, added width translates into usable representation. If it is low, added width raises the ceiling without filling it.

Spectral geometry gives one practical way to measure this distinction. Given a representation covariance, its eigenspectrum tells us how variance is distributed across eigenmodes. Effective-rank measures summarize that distribution. The entropy-based formulation traces back to Roy and Vetterli, and the broader Rényi family gives a principled way to vary sensitivity to diffuse versus dominant eigenmodes ([Roy and Vetterli, 2007](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf); [Rényi, 1961](https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/On-Measures-of-Entropy-and-Information/chapter/On-Measures-of-Entropy-and-Information/bsmsp/1200512181)).

The following definitions make this distinction measurable.

Let the eigenvalues of a representation covariance be $(\lambda_1, \ldots, \lambda_d)$, and normalize them into a variance distribution over eigenmodes:

<div class="math-box">
$$
p_i = \frac{\lambda_i}{\sum_j \lambda_j}.
$$
</div>

For this post, two effective ranks are enough:

<div class="math-box">
$$
R_1(p) = \exp\!\left(-\sum_i p_i \log p_i\right),
\qquad
R_2(p) = \frac{1}{\sum_i p_i^2}.
$$
</div>

I use the following convention throughout:

- **Average/diffuse capacity** is the soft spectral rank, $R_{\mathrm{soft}} = R_1$. It measures how broadly representation variance is distributed across eigenmodes.
- **Dominant-mode capacity** is the hard spectral rank, $R_{\mathrm{hard}} = R_2$. It measures how many dominant eigenmodes carry substantial variance.
- **Capacity asymmetry** is $\Delta = \log R_{\mathrm{soft}} - \log R_{\mathrm{hard}} = \log\!\left(R_{\mathrm{soft}} / R_{\mathrm{hard}}\right)$. It measures the multiplicative gap between diffuse and dominant-mode capacity.

<figure>
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure3_phase_diagram_for_realized_capacity.png' | relative_url }}" alt="Phase diagram for realized capacity">
  <figcaption><strong>Figure 4.</strong> <em>Conceptual phase diagram for realized capacity.</em> The horizontal axis tracks average/diffuse capacity, using normalized $\log R_{\mathrm{soft}}$; the vertical axis tracks dominant-mode capacity, using normalized $\log R_{\mathrm{hard}}$. The shaded forbidden region reflects the constraint $R_{\mathrm{hard}} \le R_{\mathrm{soft}}$. The low-asymmetry frontier marks the regime where diffuse variance and dominant eigenmode variance are closely matched.</figcaption>
</figure>

This phase diagram separates cases that scalar loss can collapse together. A model can be compact and concentrated; diffuse but weak in dominant eigenmodes; or broad in diffuse capacity while also high in dominant-mode capacity. Capacity asymmetry asks whether high diffuse capacity is matched by variance in dominant eigenmodes.

### View IV — Hard and soft inductive bias: same architecture does not mean same reachable solution
{: #view-iv-hard-and-soft-inductive-bias }

Inductive bias explains why realized capacity can differ under the same architecture. No-free-lunch and complexity-based views make the basic lesson unavoidable: learning systems need biases that privilege some solutions over others, because the data alone rarely specifies a unique model ([Goldblum et al., 2024](https://arxiv.org/abs/2304.05366)).

Architecture provides a hard inductive bias: it restricts the support of possible computation. Causal masking, residual topology, normalization placement, attention structure, parameter sharing, sparse routing, and FFN width all decide what kinds of computation the model can express efficiently.

Optimization provides a softer inductive bias. It does not usually change the formal hypothesis space, but it changes which solutions inside that space are reachable, stable, and likely under a given training budget. This aligns with the broader view that modern overparameterized learning often works by keeping a flexible hypothesis space while imposing soft preferences over data-consistent solutions ([Wilson, 2025](https://arxiv.org/abs/2503.02113)). In overparameterized models, many parameter settings can reach similar loss. The optimizer helps decide which one training actually finds.

This distinction is especially important in rare-token regimes. Where data is dense, the likelihood strongly constrains the solution. Where data is sparse, many internal organizations remain compatible with the observations, and optimizer-induced bias has more room to decide which representation directions grow.

From this perspective, the effective computational graph is not determined by the architecture alone. The formal Transformer graph defines the available paths, but the optimizer helps decide which heads, MLP directions, residual-stream features, and layer compositions actually carry signal. In the transformer-circuits view, a model is a composition of residual-stream paths and attention/MLP components; our result suggests that which paths become active can be conditional on the architecture–optimizer pair ([Elhage et al., 2021](https://transformer-circuits.pub/2021/framework/index.html)).

### View V — Plasticity and continual learning: realized capacity must remain adaptable for future tasks
{: #view-v-plasticity-and-continual-learning }

Continual learning adds a forward-looking version of the same issue. Forgetting asks whether the model keeps the past; plasticity asks whether it can still learn the future.

This matters for pretraining because many valuable future behaviors look like sparse, delayed, or underrepresented learning problems: domain specialization, rare-task acquisition, fine-tuning on small expert datasets, new tools, new formats, and long-tail facts. A model can have the nominal capacity to represent these behaviors and still fail to keep useful directions available for later adaptation.

Wider networks can reduce catastrophic forgetting by giving tasks more room and reducing gradient interference ([Mirzadeh et al., 2022](https://arxiv.org/abs/2110.11526)), but future learnability is not determined by raw width alone. Under fixed parameter budgets, architectural work also suggests that width and depth can favor different sides of the stability–plasticity trade-off: wider networks tend to improve stability, while deeper networks can improve plasticity ([Lu et al., 2025](https://arxiv.org/abs/2506.03951)).

Recent plasticity work makes the limitation more concrete. Plasticity loss has been linked to dormant units, activation drift, feature-rank collapse, spectral collapse, norm growth, optimizer dynamics, and regularization choices ([Abbas et al., 2023](https://arxiv.org/abs/2303.07507); [Dohare et al., 2024](https://www.nature.com/articles/s41586-024-07711-7); [Lyle et al., 2024](https://arxiv.org/abs/2402.18762); [He et al., 2025](https://arxiv.org/abs/2509.22335)). In language models, Han et al. show that pretraining weight decay can improve downstream fine-tuning plasticity, even when the best pretraining validation loss would select a different model ([Han et al., 2026](https://arxiv.org/abs/2602.11137)).

Spectral ranks do not prove plasticity by themselves. They measure how capacity is currently realized, while future learnability depends on whether realized or weakly used directions remain available for adaptation. But if two same-loss models allocate spectral capacity differently — especially across MID and TAIL token regimes — they may also differ in how much useful structure remains available for later specialization. Rare-task scaling work makes this connection concrete: larger models can learn rare and complex tasks partly by reducing interference and retaining rare-task features long enough for sparse signals to accumulate ([Huang et al., 2026](https://arxiv.org/abs/2605.29548)). Recent work also reframes plasticity as empowerment over future learning trajectories, which makes the same point in a different language: what training preserves can be as important as what it minimizes ([Abel et al., 2025](https://arxiv.org/abs/2505.10361)).

The continual-learning lesson is not that architecture does not matter. It is that capacity must remain not only present, but usable for later learning. That depends on the interaction between architecture, data, optimizer, regularization, and training dynamics.

## 5. Why optimizers realize capacity
{: #why-optimizers-realize-capacity }

Optimization is not just the engine that minimizes loss. In overparameterized models, many solutions can fit the training data comparably well. The optimizer helps decide which of those solutions training actually finds.

Section 4 framed this as soft inductive bias. The operational version is more concrete: an optimizer transforms raw gradients into parameter updates. It chooses the geometry through which gradients move the weights — scaling some coordinates, preconditioning or orthogonalizing matrices, accumulating momentum, applying regularization, or constraining updates to a lower-dimensional subspace. Over many steps, these choices affect which representation directions grow, which eigenmodes accumulate variance, and which computational paths become active.

<table>
<thead><tr><th>Mechanism</th><th>Examples</th><th>How it can affect realized capacity</th></tr></thead>
<tbody>
<tr><td>Adaptive scaling and weight decay</td><td>AdamW-style updates</td><td>Change per-parameter update scale, norm growth, and which same-loss basins are easier to reach</td></tr>
<tr><td>Matrix or tensor preconditioning</td><td>Shampoo-style structure-aware updates</td><td>Change the conditioning of weight updates and how added width is converted into representation modes</td></tr>
<tr><td>Matrix orthogonalization</td><td>Muon and NorMuon-style updates</td><td>Change the geometry of updates across weight matrices, which can alter how variance spreads into dominant eigenmodes</td></tr>
<tr><td>Low-rank or communication-constrained updates</td><td>Dion-style updates</td><td>Restrict the update subspace and can reshape how capacity is allocated across modes and token regimes</td></tr>
</tbody>
</table>

The important point is not that one mechanism is universally better. It is that each optimizer defines a different training geometry. AdamW comes from decoupled weight decay for adaptive optimization ([Loshchilov and Hutter, 2019](https://arxiv.org/abs/1711.05101)); Shampoo is a structure-aware tensor preconditioner ([Gupta, Koren, and Singer, 2018](https://arxiv.org/abs/1802.09568)); Muon was introduced as a matrix-orthogonalized optimizer and has since been studied for LLM scaling ([Jordan et al., 2024](https://kellerjordan.github.io/posts/muon/); [Liu et al., 2025](https://arxiv.org/abs/2502.16982)); NorMuon combines Muon-style orthogonalization with neuron-wise adaptive rates ([Li et al., 2025](https://arxiv.org/abs/2510.05491)); and Dion studies distributed orthonormalized updates with low-rank communication structure ([Ahn et al., 2025](https://arxiv.org/abs/2504.05295)).

These choices do not change the formal architecture. They change the trajectory through parameter space: which basins are reachable, which update directions are favored, which weak signals are preserved long enough to accumulate, and which representation eigendirections become variance-carrying structure. This is the mechanistic reason matched loss need not imply matched representation. Optimizer choice therefore belongs in the capacity-scaling object, not merely in the training recipe.


## 6. The design object is the architecture–optimizer pair
{: #the-design-object-is-the-architecture-optimizer-pair }

The implication is not that optimizers matter more than architectures. They solve different parts of the design problem.

Architecture determines what computation is structurally available and affordable: masking, routing, width, depth, normalization, residual topology, memory layout, and inference-time cost. Optimization determines how training fills those available degrees of freedom: which directions grow, which rare signals remain coherent, which minima are reachable, and which internal geometry is selected at similar loss.

<table>
<thead><tr><th>If the goal is...</th><th>Architecture contributes...</th><th>Optimization contributes...</th></tr></thead>
<tbody>
<tr><td>Structural guarantees</td><td>Masking, sparsity, routing, equivariance, cache and memory structure</td><td>Cannot create the guarantee, but can train effectively within it</td></tr>
<tr><td>Efficient inference</td><td>FLOP pattern, activation sparsity, communication, and memory movement</td><td>Can improve the trained solution, but usually cannot change inference structure after training</td></tr>
<tr><td>Long-tail capability</td><td>Representational room, routing paths, and capacity ceilings</td><td>Preserves weak signals and decides which rare-token directions become variance-carrying</td></tr>
<tr><td>Stable deep training</td><td>Residual topology, normalization, parameterization, and depth/width balance</td><td>Conditioning, preconditioning, momentum, regularization, and reachability</td></tr>
<tr><td>Capacity-aware scaling</td><td>More available degrees of freedom</td><td>A higher realized fraction of those degrees of freedom</td></tr>
</tbody>
</table>

This is why architecture-only conclusions can be misleading. A model family that appears weak under one optimizer may become viable under another; a model family that appears strong under AdamW may partly reflect a historically well-tuned architecture–optimizer pair.

The object we deploy is not an architecture. It is a trained model: the joint outcome of architecture, training data, optimizer, schedule, initialization, regularization, and scale. If two optimizers convert the same FFN width into different realized capacity, then a scaling decision is incomplete unless it specifies the optimizer. Likewise, an optimizer comparison is incomplete if it reports speed and loss but not the internal capacity geometry it induces.

This leads to a practical question for pretraining: what should we log if we want to know whether added architectural capacity became usable internal structure?

## 7. What changes in pretraining practice?
{: #what-changes-in-pretraining-practice }

The practical message is not "replace loss." Loss remains the central external training signal. The point is to pair loss with internal telemetry that tells us what kind of solution training is building.

A capacity-aware pretraining report should answer five questions:

<table>
<thead><tr><th>Question</th><th>Diagnostic</th><th>Failure mode it catches</th></tr></thead>
<tbody>
<tr><td>Did same-loss models build the same representation?</td><td>Matched-loss optimizer comparisons</td><td>Similar perplexity hiding different internal geometry</td></tr>
<tr><td>Does added width become usable capacity?</td><td>Average/diffuse and dominant-mode capacity scaling</td><td>Parameters increasing while dominant modes do not grow</td></tr>
<tr><td>Is capacity broad but weakly concentrated?</td><td>Capacity asymmetry</td><td>Diffuse variance without matching dominant-mode structure</td></tr>
<tr><td>Where in the data distribution does capacity appear?</td><td>Frequency-conditioned capacity across HEAD, MID, and TAIL regimes</td><td>Average loss improving while long-tail regimes remain under-realized</td></tr>
<tr><td>Is the effect tied to a design pair?</td><td>Architecture–optimizer interaction sweeps</td><td>Attributing to architecture alone what depends on the optimizer</td></tr>
</tbody>
</table>

These diagnostics are telemetry signals, not philosophical additions. They tell us whether added architectural budget becomes internal structure that can carry signal, transfer, and remain adaptable. They also make optimizer comparisons more informative: not only which optimizer reaches a loss fastest, but which optimizer converts the same architecture into usable representation.

Classical scaling laws predict loss from parameters, training data, and compute ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361); [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)). That has been enormously useful. But loss-based scaling laws do not tell us whether added parameters became usable internal dimensions, whether capacity was allocated to rare regimes, or whether two same-loss models reached the same representation geometry. A similar warning appears outside language modeling: a CVPR 2026 invited talk argues that scaling can improve benchmark accuracy while widening the gap between neural and human-like visual strategies, motivating stronger learning and architectural inductive biases rather than scaling alone ([CVPR 2026 invited talk abstract](https://cvpr.thecvf.com/virtual/2026/invited-talk/40399)).

A capacity-aware scaling law would complement loss scaling with internal variables: how average/diffuse capacity, dominant-mode capacity, capacity asymmetry, and frequency-conditioned capacity scale with width, depth, optimizer, and training data. More broadly, recent position work argues that AI systems should be studied as time-evolving training processes, not only as static artifacts to analyze or patch after training ([Biderman et al., 2026](https://arxiv.org/abs/2606.06533)). Optimizer-induced spectral scaling laws fit naturally into that agenda: they track one measurable part of how training converts nominal capacity into realized internal structure.

## 8. Conclusion: toward capacity-aware LLM design
{: #conclusion-toward-capacity-aware-llm-design }

This is the core claim behind optimizer-induced spectral scaling laws: if two optimizers train the same architecture but produce different spectral-capacity scaling, then optimizer choice has changed the model's realized capacity. The matched-loss comparison in Figure 2 strengthens the point: similar validation loss does not imply matched internal representation. Figure 3 shows where the difference becomes most consequential: MID and TAIL regimes, where optimizer-induced bias can shape how capacity is allocated across the data distribution.

The five views above are not separate arguments. They are different lenses on the same gap. Upstream–downstream transfer shows that same loss need not mean the same useful representation. Rate–distortion thinking reminds us that one scalar objective can hide different internal coordinates. Realized-capacity telemetry asks whether nominal width became usable representation capacity. The hard/soft inductive-bias view explains why the same architecture can lead to different reachable solutions under different optimizers. Plasticity and continual learning ask whether the realized capacity remains adaptable for future tasks.

This also changes how we should think about interpretability. The formal Transformer graph is fixed by the architecture, but the effective computational graph — which heads, MLP directions, residual-stream features, and layer compositions actually carry signal — may depend on the architecture–optimizer pair.

The next generation of LLM design should therefore ask more than how many parameters we train, how many tokens we consume, or how low the loss goes. It should also ask what capacity becomes real, where in the data distribution it becomes usable, and whether it remains available for transfer and future learning.

> **Capacity-aware LLM design means tracking not just what the model could represent, but what training actually made usable.**

That is the larger agenda: connect external scaling laws to internal representation geometry, optimizer-induced bias, long-tail capability, transfer, plasticity, and future learnability.

Architecture creates the space of possible computation. Optimization helps decide which parts of that space training makes active, stable, and usable.

## What this does not claim
{: #what-this-does-not-claim }

There are several ways to overstate this argument, so it is worth being explicit.

First, spectral rank is not a complete theory of intelligence, generalization, or downstream ability. It is a telemetry signal. It should be interpreted alongside loss, evaluations, robustness tests, calibration, interpretability, and application-specific metrics. The claim is not that spectral rank alone predicts capability; the claim is that it exposes an internal axis that loss and parameter count do not measure.

Second, more realized spectral capacity is not automatically better. The right question is where capacity appears, whether it is stable, whether it supports useful behaviors, and whether it improves generalization in the regimes that matter. A capacity metric is useful only if it is tied back to behavior, robustness, transfer, or future learnability.

Third, co-design is not optimizer maximalism. Architecture remains indispensable: optimizers cannot create structural guarantees, reduce inference cost by construction, or represent functions excluded by the architecture. The point is that architecture sets the available degrees of freedom, while optimization helps determine which of them training makes usable.

Fourth, the scale question remains open. The present evidence establishes a controlled optimizer-induced capacity-scaling effect at GPT-2 160M/350M scale. Stronger frontier-scale claims require larger models, longer training, more architectures, downstream probes, continued-learning tests, interpretability comparisons, and direct studies of rare-regime behavior.

## Open questions
{: #open-questions }

This framing leads to several first-class pretraining-science questions.

**1. Which spectral differences predict downstream behavior?**
If two models have similar loss but different realized capacity, which spectral axes predict transfer, robustness, rare-token reliability, or domain adaptation?

**2. Are effective computational graphs optimizer-conditional?**
If the optimizer changes which routes through the model carry meaningful signal, then circuits discovered in one trained model may not be stable across optimizers, even with the same architecture.

**3. How should we search over architecture–optimizer pairs?**
Neural architecture search typically fixes the optimizer. Optimizer evaluation typically fixes the architecture. The co-design view suggests that this may miss regions where neither component looks optimal alone, but the pair is strong.

**4. Can realized-capacity telemetry improve scaling laws?**
A capacity-aware scaling law would track how internal representation structure scales with width, depth, optimizer, and training data — and whether added parameters become usable degrees of freedom.

**5. Does realized capacity predict plasticity and future learnability?**
A model with similar loss but different spectral allocation may differ in its ability to keep learning. Capacity telemetry may therefore help diagnose loss of plasticity, representational collapse, or exhaustion of useful directions during continued training.


## Citation
{: #citation }

If you find this post useful, please cite the associated paper.

```bibtex
@article{jha2026optimizerinduced,
  title   = {Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws},
  author  = {Jha, Nandan Kumar and Reagen, Brandon},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.21803}
}
```


## References
{: #references }

<div class="references">

<h3>Scaling laws and Transformer architecture</h3>

<ol class="references-list">
  <li>Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. <em>Scaling Laws for Neural Language Models</em>. arXiv:2001.08361, 2020. <a href="https://arxiv.org/abs/2001.08361">arXiv</a></li>
  <li>Hoffmann, J. et al. <em>Training Compute-Optimal Large Language Models</em>. NeurIPS, 2022. <a href="https://arxiv.org/abs/2203.15556">arXiv</a></li>
  <li>Ramani, V., and Jain, S. V. <em>On the Optimizer Dependence of Neural Scaling Laws</em>. ICML HiLD Workshop, 2026. <a href="https://arxiv.org/abs/2605.29387">arXiv</a></li>
  <li>CVPR 2026 invited talk abstract on scaling laws, neural laws, and human-like visual strategies. <a href="https://cvpr.thecvf.com/virtual/2026/invited-talk/40399">CVPR virtual</a></li>
  <li>Vaswani, A. et al. <em>Attention Is All You Need</em>. NeurIPS, 2017. <a href="https://arxiv.org/abs/1706.03762">arXiv</a></li>
</ol>

<h3>Upstream–downstream gap and training dynamics</h3>

<ol class="references-list" start="6">
  <li>Liu, H., Xie, S. M., Li, Z., and Ma, T. <em>Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models</em>. arXiv:2210.14199, 2022. <a href="https://arxiv.org/abs/2210.14199">arXiv</a></li>
  <li>Wu, J., Lee, J. D., and Ge, R. <em>Connecting Pre-trained Language Models and Downstream Tasks via Properties of Representations</em>. NeurIPS, 2023. <a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/93712c59f6a81bd92040facf04c8b308-Abstract-Conference.html">NeurIPS</a></li>
  <li>Chen, H., Zhang, H., Li, X., Dong, Y., Shen, K., and Zhu, J. <em>Nexus: Same Pretraining Loss, Better Downstream Generalization via Common Minima</em>. arXiv:2604.09258, 2026. <a href="https://arxiv.org/abs/2604.09258">arXiv</a></li>
  <li>Li, M. Z., Agrawal, K. K., Ghosh, A., Teru, K. K., Santoro, A., Lajoie, G., and Richards, B. A. <em>Tracing the Representation Geometry of Language Models from Pretraining to Post-training</em>. arXiv:2509.23024, 2025. <a href="https://arxiv.org/abs/2509.23024">arXiv</a></li>
  <li>Biderman, S., Khan, M. A., Mireshghallah, N., Arnett, C., Barez, F., and Saphra, N. <em>Position: Don't Just "Fix it in Post": A Science of AI Must Study Training Dynamics</em>. arXiv:2606.06533, 2026. <a href="https://arxiv.org/abs/2606.06533">arXiv</a></li>
</ol>

<h3>Optimization and optimizer-induced bias</h3>

<ol class="references-list" start="11">
  <li>Loshchilov, I., and Hutter, F. <em>Decoupled Weight Decay Regularization</em>. ICLR, 2019. <a href="https://arxiv.org/abs/1711.05101">arXiv</a></li>
  <li>Gupta, V., Koren, T., and Singer, Y. <em>Shampoo: Preconditioned Stochastic Tensor Optimization</em>. ICML, 2018. <a href="https://arxiv.org/abs/1802.09568">arXiv</a></li>
  <li>Jordan, K. et al. <em>Muon: An Optimizer for Hidden Layers in Neural Networks</em>. Original public technical write-up, 2024. <a href="https://kellerjordan.github.io/posts/muon/">write-up</a></li>
  <li>Liu, J. et al. <em>Muon is Scalable for LLM Training</em>. arXiv:2502.16982, 2025. <a href="https://arxiv.org/abs/2502.16982">arXiv</a></li>
  <li>Li, Z., Liu, L., Liang, C., Chen, W., and Zhao, T. <em>NorMuon: Making Muon More Efficient and Scalable</em>. arXiv:2510.05491, 2025. <a href="https://arxiv.org/abs/2510.05491">arXiv</a></li>
  <li>Ahn, K., Xu, B., Abreu, N., and Langford, J. <em>Dion: Distributed Orthonormalized Updates</em>. arXiv:2504.05295, 2025. <a href="https://arxiv.org/abs/2504.05295">arXiv</a></li>
</ol>

<h3>Spectral capacity and information-theoretic framing</h3>

<ol class="references-list" start="17">
  <li>Alemi, A. A., Poole, B., Fischer, I., Dillon, J. V., Saurous, R. A., and Murphy, K. <em>Fixing a Broken ELBO</em>. ICML, 2018. <a href="https://proceedings.mlr.press/v80/alemi18a.html">PMLR</a></li>
  <li>Rényi, A. <em>On Measures of Entropy and Information</em>. Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability, 1961. <a href="https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/On-Measures-of-Entropy-and-Information/chapter/On-Measures-of-Entropy-and-Information/bsmsp/1200512181">Project Euclid</a></li>
  <li>Roy, O., and Vetterli, M. <em>The Effective Rank: A Measure of Effective Dimensionality</em>. EUSIPCO, 2007. <a href="https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf">PDF</a></li>
</ol>

<h3>Inductive bias, interpretability, and plasticity</h3>

<ol class="references-list" start="20">
  <li>Goldblum, M., Finzi, M., Rowan, K., and Wilson, A. G. <em>The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning</em>. ICML, 2024. <a href="https://arxiv.org/abs/2304.05366">arXiv</a></li>
  <li>Wilson, A. G. <em>Deep Learning is Not So Mysterious or Different</em>. ICML, 2025. <a href="https://arxiv.org/abs/2503.02113">arXiv</a></li>
  <li>Elhage, N. et al. <em>A Mathematical Framework for Transformer Circuits</em>. Transformer Circuits, 2021. <a href="https://transformer-circuits.pub/2021/framework/index.html">article</a></li>
  <li>Mirzadeh, S. I., Chaudhry, A., Yin, D., Hu, H., Pascanu, R., Gorur, D., and Farajtabar, M. <em>Wide Neural Networks Forget Less Catastrophically</em>. ICML, 2022. <a href="https://arxiv.org/abs/2110.11526">arXiv</a></li>
  <li>Lu, A., Yuan, H., Feng, T., and Sun, Y. <em>Rethinking the Stability-Plasticity Trade-off in Continual Learning from an Architectural Perspective</em>. ICML, 2025. <a href="https://arxiv.org/abs/2506.03951">arXiv</a></li>
  <li>Abbas, Z., Zhao, R., Modayil, J., White, A., and Machado, M. C. <em>Loss of Plasticity in Continual Deep Reinforcement Learning</em>. arXiv:2303.07507, 2023. <a href="https://arxiv.org/abs/2303.07507">arXiv</a></li>
  <li>Lyle, C., Zheng, Z., Khetarpal, K., van Hasselt, H., Pascanu, R., Martens, J., and Dabney, W. <em>Disentangling the Causes of Plasticity Loss in Neural Networks</em>. arXiv:2402.18762, 2024. <a href="https://arxiv.org/abs/2402.18762">arXiv</a></li>
  <li>He, N., Guo, K., Prakash, A., Tiwari, S., Tao, R. Y., Serapio, T., Greenwald, A., and Konidaris, G. <em>Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning</em>. arXiv:2509.22335, 2025. <a href="https://arxiv.org/abs/2509.22335">arXiv</a></li>
  <li>Dohare, S., Hernandez-Garcia, J. F., Rahman, P., Mahmood, A. R., and Sutton, R. S. <em>Maintaining Plasticity in Deep Continual Learning</em>. Nature, 2024. <a href="https://www.nature.com/articles/s41586-024-07711-7">article</a></li>
  <li>Han, T., Bordt, S., Zhang, H., and Kakade, S. <em>Weight Decay Improves Language Model Plasticity</em>. arXiv:2602.11137, 2026. <a href="https://arxiv.org/abs/2602.11137">arXiv</a></li>
  <li>Huang, J., Wurgaft, D., Bansal, R., Ruis, L., Saphra, N., Alvarez-Melis, D., Lampinen, A. K., Potts, C., and Lubana, E. S. <em>Why Larger Models Learn More: Effects of Capacity, Interference, and Rare-Task Retention</em>. arXiv:2605.29548, 2026. <a href="https://arxiv.org/abs/2605.29548">arXiv</a></li>
  <li>Abel, D. et al. <em>Plasticity as the Mirror of Empowerment</em>. arXiv:2505.10361, 2025. <a href="https://arxiv.org/abs/2505.10361">arXiv</a></li>
</ol>

</div>
