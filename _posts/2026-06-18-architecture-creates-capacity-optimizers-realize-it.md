---
layout: post
title: "Architecture Creates Capacity, Optimizers Realize It"
subtitle: "Why LLM scaling laws should account for realized capacity, not just loss curves and parameter counts"
author: "Nandan Kumar Jha"
date: 2026-06-18
permalink: /blog/optimizer-induced-capacity/
description: "Why LLM scaling should account for realized capacity: the same architecture and training data can produce different internal spectral capacity under different optimizers."
reading_time: "~24 min read"
image: /assets/blog/architecture-optimizer-codesign/figure0_same_architecture_same_data_different_optimizer.png
tags:
  - scaling-laws
  - optimization
  - representation-learning
  - llm-pretraining
  - spectral-geometry
  - continual-learning
---

<style>

.resource-links {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  align-items: center;
  margin: 0.25rem 0 1.35rem 0;
  font-size: 0.98rem;
}
.resource-links a {
  text-decoration: none;
  font-weight: 600;
}
.resource-links a:hover {
  text-decoration: underline;
}
.resource-links span {
  color: #94a3b8;
}

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

.metric-box {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1rem 1.25rem;
  margin: 1.25rem 0 1.75rem 0;
}
.metric-box strong:first-child {
  display: block;
  margin-bottom: 0.5rem;
  color: #0f172a;
}
.metric-box ul {
  margin: 0;
  padding-left: 1.25rem;
}
.metric-box li {
  margin-bottom: 0.35rem;
}
.metric-box li:last-child {
  margin-bottom: 0;
}

.takeaway-inline {
  background: #fef2f2;
  border-left: 4px solid #c0392b;
  padding: 0.75rem 1rem;
  margin: 1.25rem 0;
  border-radius: 0 6px 6px 0;
  font-size: 0.97rem;
}
.takeaway-inline strong {
  color: #c0392b;
}

.references-list {
  padding-left: 1.5rem;
  font-size: 0.94rem;
  line-height: 1.55;
}
.references-list li {
  margin-bottom: 0.6rem;
}
.references-list li:last-child {
  margin-bottom: 0;
}


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


/* Avoid double separator lines before the first section heading. */
h2#tldr {
  border-top: none;
  padding-top: 0;
}

</style>

## TL;DR
{: #tldr}

<div class="tldr-box">
<p><strong>Capacity is not just what the architecture makes possible; it is what training converts into usable representation.</strong></p>
<ul>
  <li><strong>Finding.</strong> Holding architecture, training data, tokenizer, and FFN-width schedule fixed; optimizer choice changes the spectral capacity realized inside FFN representations.</li>
  <li><strong>Matched loss is not matched representation.</strong> The same architecture can reach similar validation loss under different optimizers while learning different internal representation geometry. Longer AdamW training can match Dion (1/16) in validation loss, but not in dominant-mode capacity scaling.</li>
  <li><strong>Implication.</strong> Architecture sets the available degrees of freedom; training dynamics help determine which ones become active and variance-carrying, and how that capacity is allocated across token-frequency regimes. The effect is clearest for rare tokens, where sparse supervision leaves more room for optimizer-induced bias.</li>
</ul>
</div>

Throughout this post, **realized capacity** means **realized spectral capacity**: variance-carrying internal directions measured through representation eigenspectra. It is not a complete measure of intelligence, transfer, or downstream capability; it is an internal telemetry signal that loss curves and parameter counts do not directly capture.

This post starts from a controlled fact and asks what follows from it. Classical scaling laws taught us to ask how loss changes with parameters, training data, and compute. They leave another question open: when we add architectural capacity, does training convert it into measurable internal structure? In our experiments, that conversion depends strongly on optimizer choice, especially in rare-token regimes where supervision is sparse.


<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure0_same_architecture_same_data_different_optimizer.png' | relative_url }}" alt="Same architecture, same data, different optimizer conceptual setup">
  <figcaption><strong>Figure 0.</strong> <em>Same architecture, same training data, different optimizer.</em> With the model and data held fixed, different optimizers induce different paths through parameter space — and those trajectories can yield different internal spectral-capacity profiles even when final loss is similar.</figcaption>
</figure>

<details class="toc-box">
  <summary>On this page</summary>
  <ol>
    <li><a href="#same-architecture-different-capacity-scaling">Same architecture, different capacity scaling</a></li>
    <li><a href="#matched-loss-is-not-matched-geometry">Matched loss is not matched representation</a></li>
    <li><a href="#rare-tokens-expose-optimizer-induced-capacity-allocation">Rare-token capacity allocation</a></li>
    <li>
      <a href="#three-views-of-the-same-gap">Three views of the same gap</a>
      <ol>
        <li><a href="#view-i-scalar-objectives-under-identify-internal-structure">Scalar objectives under-identify internal structure</a></li>
        <li><a href="#view-ii-nominal-capacity-is-not-realized-capacity">Nominal capacity is not realized capacity</a></li>
        <li><a href="#view-iii-reachable-capacity-is-optimizer-conditional">Reachable capacity is optimizer-conditional</a></li>
      </ol>
    </li>
    <li><a href="#how-optimizers-shape-realized-capacity">How optimizers shape realized capacity</a></li>
    <li><a href="#the-design-object-is-the-architecture-optimizer-pair">Architecture–optimizer co-design</a></li>
    <li><a href="#what-changes-in-pretraining-practice">What changes in pretraining practice?</a></li>
    <li><a href="#what-this-does-not-claim">What this does not claim</a></li>
    <li><a href="#open-questions">Open questions</a></li>
    <li><a href="#conclusion-toward-capacity-aware-llm-design">Conclusion</a></li>
  </ol>
</details>

## 1. Same architecture, different capacity scaling
{: #same-architecture-different-capacity-scaling}

The result behind this post is simple:

> **When architecture, training data, tokenizer, and FFN-width schedule are fixed, changing the optimizer changes how realized spectral capacity scales.**

Four quantities help interpret the result.

<div class="metric-box">
  <strong>Metrics used in this post.</strong>
  <ul>
    <li><strong>Average/diffuse capacity:</strong> soft spectral rank; how broadly representation variance spreads across eigenmodes.</li>
    <li><strong>Dominant-mode capacity:</strong> hard spectral rank; how many strong eigenmodes carry substantial variance.</li>
    <li><strong>Capacity asymmetry:</strong> a diagnostic of imbalance between diffuse capacity and dominant-mode capacity; it is not a quantity to maximize.</li>
    <li><strong>Scaling exponent:</strong> how quickly realized capacity grows as FFN width increases.</li>
  </ul>
</div>

We vary FFN width within the same model family and compare how realized spectral capacity grows. The optimizer changes the slope of that growth. The model does not merely train faster or slower; it converts the same architectural budget into a different internal-capacity profile.

Figure 1 gives the aggregate view. The key quantity is not only the final rank value, but the **scaling exponent**: how much added FFN width becomes realized spectral capacity. In the TAIL regime, AdamW shows weak dominant-mode scaling across the width sweep ($\beta_{\mathrm{hard}} \approx 0.44$), while Muon and NorMuon approach near-linear scaling ($\beta_{\mathrm{hard}} \approx 1.0$). This is roughly a $2.3\times$ larger fitted exponent under the same architecture and data.

<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure1_optimizer_capacity_scaling.png' | relative_url }}" alt="Aggregated optimizer-level realized-capacity scaling">
  <figcaption><strong>Figure 1.</strong> <em>Aggregated optimizer-level realized-capacity scaling.</em> Architecture, training data, and FFN-width schedule are fixed; only the optimizer changes. Panel A reports aggregated scaling exponents for average/diffuse capacity, $\beta_{\mathrm{soft}}$, and dominant-mode capacity, $\beta_{\mathrm{hard}}$. Panel B shows how capacity asymmetry scales with width, $\Delta\beta = \beta_{\mathrm{soft}} - \beta_{\mathrm{hard}}$. Muon and NorMuon show stronger dominant-mode capacity scaling and a smaller $\Delta\beta$ than AdamW, indicating that more added width appears in dominant eigenmodes.</figcaption>
</figure>

The full result tables, frequency-conditioned fits, and ablations are on the <a href="https://optimizer-scaling-laws.github.io/" target="_blank" rel="noopener noreferrer">project page</a>. Here, I focus on what these scaling differences imply for architecture, optimization, and realized capacity.

These are controlled GPT-2-scale studies, including 160M and 350M model families on FineWeb-Edu. They show that optimizer choice can change internal capacity-scaling exponents under matched architecture and training data; they do not show that the same optimizer ordering must persist unchanged at multi-billion-parameter scale. The next experiment is a calibrated billion-scale sweep that keeps architecture, training data, tokenizer, FFN-width schedule, and compute accounting fixed while measuring whether the same average/diffuse, dominant-mode, and HEAD/MID/TAIL effects persist.

<p class="takeaway-inline"><strong>Takeaway.</strong> Optimizer choice can change not only convergence speed or final loss, but the scaling law exponents by which added FFN width becomes realized spectral capacity.</p>


## 2. Matched loss is not matched representation
{: #matched-loss-is-not-matched-geometry}

A natural objection is that one optimizer may simply train faster. The matched-loss control tests that possibility.

Extending AdamW training from 6K to 12K improves validation perplexity and brings it close to Dion-1/16 across the FFN-width sweep. But the realized-capacity profiles do not match; AdamW-12K remains much weaker in dominant-mode capacity scaling.

The scaling trend is counterintuitive: longer AdamW training improves loss but weakens dominant-mode scaling, with the exponent dropping from about **$0.29$** to about **$0.03$**. Wider AdamW models are therefore not simply “catching up” internally as loss improves.


<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure2_matched_loss_capacity_comparison.png' | relative_url }}" alt="Matched loss but different realized capacity">
  <figcaption><strong>Figure 2.</strong> <em>Matched loss, different realized capacity.</em> Extending AdamW training improves perplexity — closer to Dion-1/16. However, the realized-capacity profiles remain substantially different. The hard-rank scaling trend is especially diagnostic: closer loss does not imply matched internal geometry.</figcaption>
</figure>

This gap is also not closed by learning-rate tuning; the paper discusses the learning-rate sweep and its spectral implications. Practically, loss curves can make two runs look equivalent even when their internal capacity trajectories differ. Loss measures average output error; it does not establish that training produced the same representation directions, minima, or adaptation-relevant internal structure.

<p class="takeaway-inline"><strong>Takeaway.</strong> Matched validation loss can still hide different width-to-capacity realization trajectories. Loss matching is necessary for a fair comparison, but it is not enough to establish matched internal geometry.</p>


## 3. Rare tokens expose optimizer-induced capacity allocation
{: #rare-tokens-expose-optimizer-induced-capacity-allocation}

Natural language is long-tailed. A small number of HEAD tokens receive dense, repeated supervision; TAIL tokens receive sparse, noisy, intermittent supervision.

This changes the role of optimization. For HEAD tokens, data strongly constrains what should be learned, so many reasonable optimizers can find similar solutions. For TAIL tokens, the evidence is weaker: many representations remain compatible with sparse observations, leaving more room for optimizer-induced bias to shape which directions grow.

In Bayesian terms, dense regimes are likelihood-dominated; sparse regimes are prior-sensitive. Here, the optimizer acts like a soft prior over which solutions training reaches.

This leads to a prediction: optimizer-induced differences should be most visible where the data underdetermines the representation.

<table>
<thead><tr><th>Token regime</th><th>Data condition</th><th>Dominant bottleneck</th><th>Expected stronger lever</th></tr></thead>
<tbody>
<tr><td>HEAD</td><td>Dense, frequent, well-conditioned supervision</td><td>Compute and architectural ceiling</td><td>Architecture</td></tr>
<tr><td>MID</td><td>Partially constrained, mixed signal quality</td><td>Capacity allocation across modes</td><td>Architecture–optimizer pair</td></tr>
<tr><td>TAIL</td><td>Sparse, noisy, underdetermined supervision</td><td>Signal preservation and implicit bias</td><td>Optimizer / training dynamics</td></tr>
</tbody>
</table>

This is why architecture-only changes can under-deliver for long-tail behavior: adding capacity raises the ceiling, but sparse supervision does not automatically fill it. The optimizer may need to preserve and amplify weak signals rather than fragment them.

This is where the result becomes relevant for capability-oriented pretraining questions. If realized capacity affects how sparse regimes transfer, then low-resource languages, rare scientific and code vocabulary, long-tail factual recall, tool-use edge cases, and sparse expert specialization are natural places to test this diagnostic. Spectral measurement alone does not prove that connection.

Figure 3 resolves the aggregate result from Figure 1 into HEAD, MID, and TAIL regimes. The left panel shows dominant-mode scaling through $\beta_{\mathrm{hard}}$: in TAIL representations, AdamW and Dion-1/16 are much weaker, while Muon and NorMuon approach near-linear dominant-mode growth. Dion-1/16 remains useful as a matched-loss control, but in the TAIL regime it behaves closer to AdamW than to Muon/NorMuon. The right panel shows how asymmetry scales, $\Delta\beta = \beta_{\mathrm{soft}} - \beta_{\mathrm{hard}}$.


<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure4_capacity_scaling_asymmetry_token_regimes.png' | relative_url }}" alt="Frequency-conditioned realized-capacity scaling across token regimes">
  <figcaption><strong>Figure 3.</strong> <em>Frequency-conditioned realized-capacity scaling.</em> Panel A shows dominant-mode capacity scaling exponents, $\beta_{\mathrm{hard}}$, across HEAD, MID, and TAIL token regimes. Panel B shows how capacity asymmetry scales with width, $\Delta\beta = \beta_{\mathrm{soft}} - \beta_{\mathrm{hard}}$. The regime-conditioned view reveals where the optimizer effect is strongest: MID and especially TAIL regimes.</figcaption>
</figure>

This matters for pretraining science because average loss is dominated by frequent patterns and can hide whether sparse regimes receive realized spectral capacity. Frequency-conditioned spectral telemetry exposes that hidden allocation.

A capacity-aware training run should therefore ask not only whether average loss improved, but whether the long tail received measurable internal structure that predicts rare-regime behavior.

<p class="takeaway-inline"><strong>Takeaway.</strong> Rare-token regimes are where optimizer-induced bias has the most room to shape which weak signals become coherent representation directions.</p>

## 4. Three views of the same gap
{: #three-views-of-the-same-gap}

The empirical result is not an isolated optimizer anecdote. It points to a broader gap: scalar objective value, nominal architecture, and learned internal representation are often blurred, but they are not the same object.

Section 2 made the paper-specific evidence concrete: matched validation loss can hide different spectral-capacity trajectories. This section organizes what that result suggests for **pretraining science**.

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
<td>Scalar objectives hide internal structure</td>
<td>Similar loss can hide different learned representations.</td>
<td>Places the matched-loss result inside a broader failure mode of loss-only comparison.</td>
</tr>
<tr>
<td>Nominal capacity is not realized capacity</td>
<td>Width matters only if training converts it into variance-carrying representation directions.</td>
<td>Names the missing internal axis: how much architectural capacity becomes realized spectral capacity.</td>
</tr>
<tr>
<td>Reachable capacity is optimizer-conditional</td>
<td>The same architecture does not imply the same reachable solution.</td>
<td>Explains why optimizer-induced bias can shape capacity allocation, with implications for effective computation graphs and later plasticity that the paper does not yet test.</td>
</tr>
</tbody>
</table>

The sequence moves from measurement to mechanism: scalar objectives can hide the model’s internal state, realized capacity names the missing internal axis, and optimizer-induced bias explains why the reachable solution can change under fixed architecture.


### View I — Scalar objectives under-identify internal structure
{: #view-i-scalar-objectives-under-identify-internal-structure}

Loss, gradient norms, and downstream evaluations are useful signals, but they are not complete descriptions of the trained model. The same scalar value can arise from different internal organizations: variance may spread across many directions, concentrate in a few dominant modes, or appear unevenly across token-frequency regimes. The matched-loss control above is one spectral-capacity instance of this broader under-identification problem.

The upstream–downstream literature gives empirical precedent. The same pretraining loss can still produce different downstream transfer (<a href="https://arxiv.org/abs/2210.14199" target="_blank" rel="noopener noreferrer">Liu et al., 2022</a>), different geometry among task-specific minima (<a href="https://arxiv.org/abs/2604.09258" target="_blank" rel="noopener noreferrer">Chen et al., 2026</a>), and non-monotonic changes in learned representation geometry that scalar metrics miss (<a href="https://arxiv.org/abs/2509.23024" target="_blank" rel="noopener noreferrer">Li et al., 2025</a>). Theoretical work makes the same point: low pretraining loss alone need not guarantee that every downstream-relevant feature is recoverable from the representation (<a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/93712c59f6a81bd92040facf04c8b308-Abstract-Conference.html" target="_blank" rel="noopener noreferrer">Wu, Lee, and Ge, 2023</a>).

Rate–distortion is a useful analogy, not a literal equivalence. Spectral rank is not rate; it measures how variance is distributed across eigendirections. But the warning is similar: one scalar objective can hide different internal codes. Alemi et al. show this for VAEs, where identical ELBO values can correspond to different points in the rate–distortion plane (<a href="https://proceedings.mlr.press/v80/alemi18a.html" target="_blank" rel="noopener noreferrer">Alemi et al., 2018</a>). In LLM pretraining, validation loss can likewise tell us that prediction error is low without telling us which internal coordinates training used.

The question is therefore not only **how low did loss go?** but also:

> **What internal capacity did training realize in order to reach that loss?**

### View II — Nominal capacity is not realized capacity
{: #view-ii-nominal-capacity-is-not-realized-capacity}

The architecture–optimizer distinction becomes clearest if we separate two notions of capacity.

> **Nominal capacity** is the capacity implied by the architecture: parameter count, width, depth, heads, experts, routing paths, memory layout, and FLOPs.

> **Realized spectral capacity** is the capacity measured as active variance-carrying structure inside the trained model: which representation directions become active, how variance is distributed across eigenmodes, which modes grow with width, and how capacity is allocated across token regimes (HEAD/MID/TAIL).

Architecture gives the learning system what no optimizer can add later: causal masking, routing structure, dimensional ceilings, residual topology, and symmetry constraints when present. But these define *available* capacity only; whether training realizes those degrees of freedom is a separate question.


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

Here, $\mathcal{A}$ is the architecture, $\mathcal{O}$ is the optimizer/training algorithm, and $\mathcal{D}$ is the training data. This is not a literal scalar law; it is a design principle. Architecture sets available capacity. Optimization and training data influence how much of that capacity becomes coherent in representation space. Whether that structure is behaviorally useful must be tested separately.

<table>
<thead><tr><th>Observable</th><th>What it tells us</th><th>Strongly influenced by</th></tr></thead>
<tbody>
<tr><td>Parameter count, width, depth, heads</td><td>The nominal capacity ceiling</td><td>Architecture</td></tr>
<tr><td>FLOPs and memory movement</td><td>The cost and shape of computation</td><td>Architecture</td></tr>
<tr><td>Average/diffuse capacity</td><td>How broadly variance is distributed across eigenmodes</td><td>Training dynamics</td></tr>
<tr><td>Dominant-mode capacity</td><td>How many dominant eigenmodes carry substantial variance</td><td>Architecture–optimizer pair</td></tr>
<tr><td>Capacity asymmetry</td><td>Whether diffuse capacity is matched by dominant-mode capacity; not a quantity to maximize</td><td>Architecture–optimizer pair</td></tr>
<tr><td>Frequency-conditioned capacity</td><td>Which token regimes receive realized spectral capacity</td><td>Data distribution × optimizer</td></tr>
</tbody>
</table>

The same architectural intervention can therefore have different realized effects under different optimizers. If $\rho_{\mathrm{realized}}$ is high, added width translates into measured representation capacity. If it is low, added width raises the ceiling without filling it.

Spectral geometry makes the distinction measurable. Given a representation covariance, its eigenspectrum tells us how variance is distributed across eigenmodes. Effective-rank measures summarize that distribution; the entropy-based formulation traces back to Roy and Vetterli, and the Rényi family varies sensitivity to diffuse versus dominant eigenmodes (<a href="https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf" target="_blank" rel="noopener noreferrer">Roy and Vetterli, 2007</a>; <a href="https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/On-Measures-of-Entropy-and-Information/chapter/On-Measures-of-Entropy-and-Information/bsmsp/1200512181" target="_blank" rel="noopener noreferrer">Rényi, 1961</a>).

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

I use the following convention:

- **Average/diffuse capacity** is the soft spectral rank, $R_{\mathrm{soft}} = R_1$.
- **Dominant-mode capacity** is the hard spectral rank, $R_{\mathrm{hard}} = R_2$.
- **Capacity asymmetry** measures the imbalance between these two quantities:
  $$
  \Delta = \log R_{\mathrm{soft}} - \log R_{\mathrm{hard}}
  = \log\left(R_{\mathrm{soft}} / R_{\mathrm{hard}}\right).
  $$
  It is the multiplicative gap between diffuse and dominant-mode capacity in a single model, not a capacity type to maximize. Lower or non-growing asymmetry generally means dominant modes are keeping pace with diffuse spectral support; growing asymmetry means width is increasing diffuse capacity faster than dominant-mode capacity.

Under the power-law fits used here, $\log R_{\mathrm{soft}}$ and $\log R_{\mathrm{hard}}$ grow linearly in $\log$ FFN width with slopes $\beta_{\mathrm{soft}}$ and $\beta_{\mathrm{hard}}$. Therefore, the exponent gap reported in the figures,
$$
\Delta\beta = \beta_{\mathrm{soft}} - \beta_{\mathrm{hard}},
$$
is the slope of $\Delta$ against $\log$ width. It measures how capacity asymmetry scales as the model widens, not the asymmetry value in any one model.

<figure>
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure3_phase_diagram_for_realized_capacity.png' | relative_url }}" alt="Phase diagram for realized capacity">
  <figcaption><strong>Figure 4.</strong> <em>Conceptual phase diagram for realized capacity.</em> The horizontal axis tracks average/diffuse capacity, using normalized $\log R_{\mathrm{soft}}$; the vertical axis tracks dominant-mode capacity, using normalized $\log R_{\mathrm{hard}}$. The shaded forbidden region reflects the constraint $R_{\mathrm{hard}} \le R_{\mathrm{soft}}$. The low-asymmetry frontier marks the regime where diffuse variance and dominant eigenmode variance are closely matched.</figcaption>
</figure>

This phase diagram separates cases that scalar loss can collapse together: compact and concentrated; diffuse but weak in dominant modes; or broad in diffuse capacity while also high in dominant-mode capacity. Capacity asymmetry asks whether diffuse capacity is matched by dominant eigenmode variance.

### View III — Reachable capacity is optimizer-conditional
{: #view-iii-reachable-capacity-is-optimizer-conditional}

Inductive bias explains why realized capacity can differ under the same architecture. Learning systems need biases that privilege some solutions over others because data rarely specifies a unique model (<a href="https://arxiv.org/abs/2304.05366" target="_blank" rel="noopener noreferrer">Goldblum et al., 2024</a>).

Architecture provides a hard inductive bias: it restricts the support of possible computation through causal masking, residual topology, attention structure, parameter sharing, and routing. Optimization provides a softer bias: it usually leaves the formal hypothesis space unchanged, but changes which solutions are reachable, stable, and likely under a given training budget. This aligns with the broader view that modern overparameterized learning often keeps a flexible hypothesis space while imposing soft preferences over data-consistent solutions (<a href="https://arxiv.org/abs/2503.02113" target="_blank" rel="noopener noreferrer">Wilson, 2025</a>).

This distinction matters especially in rare-token regimes, where data weakly constrains the solution and optimizer-induced bias has more room to shape which representation directions grow. It also suggests that the effective computational graph is not determined by architecture alone. The formal Transformer graph defines the available paths, but the optimizer can influence which heads, MLP directions, residual-stream features, and layer compositions carry signal. In the transformer-circuits view, a model is a composition of residual-stream paths and attention/MLP components; our result suggests that active paths are conditional on the architecture–optimizer pair (<a href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener noreferrer">Elhage et al., 2021</a>).

#### Forward-looking implication: realized capacity may matter for plasticity
{: #forward-looking-implication-realized-capacity-may-matter-for-plasticity}

Continual learning adds a forward-looking version of the same issue. Forgetting asks whether the model keeps the past; plasticity asks whether it can still learn the future. This is a conjectural extension of the paper's evidence, not a direct claim from spectral rank alone: the paper measures current realized capacity, while future learnability depends on whether realized or weakly used directions remain available for adaptation.

The connection is worth preserving because many valuable future behaviors are sparse, delayed, or underrepresented learning problems: domain specialization, rare-task acquisition, fine-tuning on small expert datasets, new tools, new formats, and long-tail facts. Width can reduce catastrophic forgetting by giving tasks more room (<a href="https://arxiv.org/abs/2110.11526" target="_blank" rel="noopener noreferrer">Mirzadeh et al., 2022</a>), but future learnability is not determined by raw width alone; width and depth can favor different sides of the stability–plasticity trade-off (<a href="https://arxiv.org/abs/2506.03951" target="_blank" rel="noopener noreferrer">Lu et al., 2025</a>).

Recent plasticity work links plasticity loss to dormant units, feature-rank or spectral collapse, optimizer dynamics, and regularization (<a href="https://arxiv.org/abs/2303.07507" target="_blank" rel="noopener noreferrer">Abbas et al., 2023</a>; <a href="https://www.nature.com/articles/s41586-024-07711-7" target="_blank" rel="noopener noreferrer">Dohare et al., 2024</a>; <a href="https://arxiv.org/abs/2402.18762" target="_blank" rel="noopener noreferrer">Lyle et al., 2024</a>; <a href="https://arxiv.org/abs/2509.22335" target="_blank" rel="noopener noreferrer">He et al., 2025</a>). In language models, pretraining weight decay can improve downstream fine-tuning plasticity even when validation loss would select a different model (<a href="https://arxiv.org/abs/2602.11137" target="_blank" rel="noopener noreferrer">Han et al., 2026</a>). Rare-task scaling work gives an adjacent mechanism: larger models can learn rare and complex tasks partly by reducing interference and retaining rare-task features long enough for sparse signals to accumulate (<a href="https://arxiv.org/abs/2605.29548" target="_blank" rel="noopener noreferrer">Huang et al., 2026</a>); recent work also frames plasticity as empowerment over future learning trajectories (<a href="https://arxiv.org/abs/2505.10361" target="_blank" rel="noopener noreferrer">Abel et al., 2025</a>).

The continual-learning lesson is not that architecture does not matter. It is that capacity may need to be realized during pretraining and preserved in forms that remain available for later learning. That depends on architecture, training data, optimizer, regularization, and training dynamics.

## 5. How optimizers shape realized capacity
{: #how-optimizers-shape-realized-capacity}

How can optimizer choice change realized capacity, not just training speed? An optimizer transforms raw gradients into parameter updates. Its update geometry — coordinate-wise scaling, matrix preconditioning, orthogonalization, momentum, or constrained subspaces — can affect which representation directions grow, which eigenmodes accumulate variance, and which computational paths become active.

<table>
<thead><tr><th>Mechanism</th><th>Examples</th><th>How it can affect realized capacity</th></tr></thead>
<tbody>
<tr><td>Adaptive scaling and weight decay</td><td>AdamW-style updates</td><td>Change per-parameter update scale, norm growth, and which same-loss basins are easier to reach</td></tr>
<tr><td>Matrix or tensor preconditioning</td><td>Shampoo-style structure-aware updates</td><td>Change the conditioning of weight updates and how added width is converted into representation modes</td></tr>
<tr><td>Matrix orthogonalization</td><td>Muon and NorMuon-style updates</td><td>Change the geometry of updates across weight matrices, which can alter how variance spreads into dominant eigenmodes</td></tr>
<tr><td>Low-rank or communication-constrained updates</td><td>Dion-style updates</td><td>Restrict the update subspace and can reshape how capacity is allocated across modes and token regimes</td></tr>
</tbody>
</table>

The point is not that one mechanism is universally better. It is that each optimizer defines a different training geometry. AdamW comes from decoupled weight decay for adaptive optimization (<a href="https://arxiv.org/abs/1711.05101" target="_blank" rel="noopener noreferrer">Loshchilov and Hutter, 2019</a>); Shampoo is a structure-aware tensor preconditioner (<a href="https://arxiv.org/abs/1802.09568" target="_blank" rel="noopener noreferrer">Gupta, Koren, and Singer, 2018</a>); Muon was introduced as a matrix-orthogonalized optimizer and has since been studied for LLM scaling (<a href="https://kellerjordan.github.io/posts/muon/" target="_blank" rel="noopener noreferrer">Jordan et al., 2024</a>; <a href="https://arxiv.org/abs/2502.16982" target="_blank" rel="noopener noreferrer">Liu et al., 2025</a>); NorMuon combines Muon-style orthogonalization with neuron-wise adaptive rates (<a href="https://arxiv.org/abs/2510.05491" target="_blank" rel="noopener noreferrer">Li et al., 2025</a>); and Dion studies distributed orthonormalized updates with low-rank communication structure (<a href="https://arxiv.org/abs/2504.05295" target="_blank" rel="noopener noreferrer">Ahn et al., 2025</a>).

These choices do not change the formal architecture. They change the trajectory through parameter space: which basins are reachable, which update directions are favored, which weak signals persist long enough to accumulate, and which representation eigendirections become variance-carrying structure. This gives one plausible reason matched loss need not imply matched representation. Optimizer choice therefore belongs in the capacity-scaling object, not merely in the training recipe.

## 6. The design object is the architecture–optimizer pair
{: #the-design-object-is-the-architecture-optimizer-pair}

The implication is not that optimizers matter more than architectures. Architecture and optimization solve different parts of the design problem:

<table>
<thead><tr><th>If the goal is...</th><th>Architecture contributes...</th><th>Optimization contributes...</th></tr></thead>
<tbody>
<tr><td>Structural guarantees</td><td>Masking, routing, equivariance, cache and memory constraints</td><td>Trains within these constraints; does not create them</td></tr>
<tr><td>Efficient inference</td><td>FLOP, activation, communication, and memory pattern</td><td>Improves the learned solution; rarely changes the deployment graph</td></tr>
<tr><td>Long-tail capability</td><td>Representational room and routing paths</td><td>Can preserve weak signals and shape rare-token directions</td></tr>
<tr><td>Stable deep training</td><td>Residuals, normalization, and parameterization</td><td>Conditioning, momentum, regularization, and reachability</td></tr>
<tr><td>Capacity-aware scaling</td><td>Available degrees of freedom</td><td>Realized fraction and allocation of those degrees of freedom</td></tr>
</tbody>
</table>

The practical consequence is simple: scaling decisions should specify both architecture and optimizer. Optimizer comparisons should report not only speed and loss, but also the internal capacity geometry they induce. This leads to a concrete pretraining question: what should we log to know whether added architectural capacity became realized internal structure?

## 7. What changes in pretraining practice?
{: #what-changes-in-pretraining-practice}

The practical takeaway is not that loss—or loss-based scaling analysis—should be replaced. Loss remains the central training objective and a low-variance signal for scaling laws. But loss alone does not tell us what kind of solution the model is converging toward, whether it is using its parameter budget effectively, or whether the architecture–optimizer pair provides the right inductive biases. For that, we need internal telemetry.

A capacity-aware pretraining report should answer five practical questions:

<table>
<thead><tr><th>Question</th><th>Diagnostic</th><th>Failure mode it catches</th></tr></thead>
<tbody>
<tr><td>Did same-loss models build the same representation?</td><td>Matched-loss optimizer comparisons</td><td>Similar perplexity hiding different internal geometry</td></tr>
<tr><td>Does added width become realized capacity?</td><td>Average/diffuse and dominant-mode capacity scaling</td><td>Parameters increasing while dominant modes do not grow</td></tr>
<tr><td>Is capacity broad but weakly concentrated?</td><td>Capacity asymmetry</td><td>Diffuse capacity growing without matching dominant-mode structure</td></tr>
<tr><td>Where in the data distribution does capacity appear?</td><td>Frequency-conditioned capacity across HEAD, MID, and TAIL regimes</td><td>Average loss improving while long-tail regimes remain under-realized</td></tr>
<tr><td>Is the effect tied to a design pair?</td><td>Architecture–optimizer interaction sweeps</td><td>Attributing to architecture alone what depends on the optimizer</td></tr>
</tbody>
</table>

These diagnostics are telemetry signals, not extra philosophical commitments. They make optimizer comparisons more informative: not only which optimizer reaches a target loss fastest, but which one converts the same architecture into realized spectral representation.

Classical scaling laws predict loss from parameters, data, and compute (<a href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noopener noreferrer">Kaplan et al., 2020</a>; <a href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noopener noreferrer">Hoffmann et al., 2022</a>). That remains central. But loss-based scaling laws do not tell us whether added parameters became realized internal dimensions, whether capacity was allocated to rare regimes, or whether same-loss models reached the same representation geometry.

A capacity-aware scaling law would complement loss scaling with internal variables: average/diffuse capacity, dominant-mode capacity, capacity asymmetry, and frequency-conditioned capacity as functions of width, depth, optimizer, and data. This fits naturally with recent position work arguing that AI systems should be studied as training processes, not only as static artifacts to analyze or patch after training (<a href="https://arxiv.org/abs/2606.06533" target="_blank" rel="noopener noreferrer">Biderman et al., 2026</a>).


## 8. What this does not claim
{: #what-this-does-not-claim}

It is easy to overstate this argument, so the boundaries matter.

First, spectral rank is not a complete theory of intelligence, generalization, or downstream ability. It is a telemetry signal that should be interpreted alongside loss, evaluations, robustness, calibration, interpretability, and task-specific metrics.

Second, more realized spectral capacity is not automatically better. The useful question is where capacity appears, whether it is stable, and whether it predicts behavior, robustness, transfer, or future learnability.

Third, co-design is not optimizer maximalism. Architecture remains indispensable: optimizers cannot create structural guarantees, reduce inference cost by construction, or represent functions excluded by the architecture. The point is that architecture sets available degrees of freedom, while optimization helps determine which of them training realizes.

Fourth, the scale question remains open. The present evidence establishes a controlled optimizer-induced capacity-scaling effect at GPT-2 160M/350M scale. Stronger frontier-scale claims require larger models, longer training, more architectures, downstream probes, continued-learning tests, interpretability comparisons, and direct studies of rare-regime behavior.

## 9. Open questions
{: #open-questions}

This framing leads to concrete pretraining-science questions.

**1. Which spectral differences predict downstream behavior?**
If two models have similar loss but different realized capacity, which spectral axes predict transfer, robustness, rare-token reliability, or domain adaptation?

**2. Are effective computational graphs optimizer-conditional?**
If the optimizer changes which routes through the model carry meaningful signal, then circuits discovered in one trained model may not be stable across optimizers, even with the same architecture.

**3. How should we search over architecture–optimizer pairs?**
Neural architecture search typically fixes the optimizer. Optimizer evaluation typically fixes the architecture. The co-design view suggests that this may miss regions where neither component looks optimal alone, but the pair is strong.

**4. Can realized-capacity telemetry improve scaling laws?**
A capacity-aware scaling law would track how internal representation structure scales with width, depth, optimizer, and training data — and whether added parameters become realized degrees of freedom.

**5. Does realized capacity predict plasticity and future learnability?**
A model with similar loss but different spectral allocation may differ in its ability to keep learning. Capacity telemetry may therefore help diagnose loss of plasticity, representational collapse, or exhaustion of useful directions during continued training.


## 10. Conclusion: toward capacity-aware LLM design
{: #conclusion-toward-capacity-aware-llm-design}

The core claim is simple: if two optimizers train the same architecture but produce different spectral-capacity scaling, optimizer choice has changed the model's realized capacity. The matched-loss comparison in Figure 2 strengthens the point: similar validation loss does not imply matched internal representation. Figure 3 shows where the measured difference is largest: MID and TAIL regimes, where optimizer-induced bias can shape capacity allocation across the data distribution.

The three views above point to the same gap: scalar objectives under-identify internal structure, nominal capacity is not realized capacity, and reachable capacity is optimizer-conditional.

The next generation of LLM design should therefore ask more than how many parameters we train, how many tokens we consume, or how low the loss goes. It should also ask what capacity becomes realized, where it appears in the data distribution, and whether those realized directions predict transfer or future learning.

> **Capacity-aware LLM design means tracking not just what the model could represent, but what training actually realized — and then testing which realized directions support behavior, transfer, and future learning.**

Architecture creates the space of possible computation. Optimization helps decide which parts of that space training makes active, stable, and measurable as internal structure.

## Citation
{: #citation}

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
{: #references}

<div class="references">

<h3>Scaling laws</h3>

<ol class="references-list">
  <li>Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. <em>Scaling Laws for Neural Language Models</em>. arXiv:2001.08361, 2020. <a href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Hoffmann, J. et al. <em>Training Compute-Optimal Large Language Models</em>. NeurIPS, 2022. <a href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  </ol>

<h3>Upstream–downstream gap and training dynamics</h3>

<ol class="references-list" start="3">
  <li>Liu, H., Xie, S. M., Li, Z., and Ma, T. <em>Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models</em>. arXiv:2210.14199, 2022. <a href="https://arxiv.org/abs/2210.14199" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Wu, J., Lee, J. D., and Ge, R. <em>Connecting Pre-trained Language Models and Downstream Tasks via Properties of Representations</em>. NeurIPS, 2023. <a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/93712c59f6a81bd92040facf04c8b308-Abstract-Conference.html" target="_blank" rel="noopener noreferrer">NeurIPS</a></li>
  <li>Chen, H., Zhang, H., Li, X., Dong, Y., Shen, K., and Zhu, J. <em>Nexus: Same Pretraining Loss, Better Downstream Generalization via Common Minima</em>. arXiv:2604.09258, 2026. <a href="https://arxiv.org/abs/2604.09258" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Li, M. Z., Agrawal, K. K., Ghosh, A., Teru, K. K., Santoro, A., Lajoie, G., and Richards, B. A. <em>Tracing the Representation Geometry of Language Models from Pretraining to Post-training</em>. arXiv:2509.23024, 2025. <a href="https://arxiv.org/abs/2509.23024" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Biderman, S., Khan, M. A., Mireshghallah, N., Arnett, C., Barez, F., and Saphra, N. <em>Position: Don't Just "Fix it in Post": A Science of AI Must Study Training Dynamics</em>. arXiv:2606.06533, 2026. <a href="https://arxiv.org/abs/2606.06533" target="_blank" rel="noopener noreferrer">arXiv</a></li>
</ol>

<h3>Optimization and optimizer-induced bias</h3>

<ol class="references-list" start="8">
  <li>Loshchilov, I., and Hutter, F. <em>Decoupled Weight Decay Regularization</em>. ICLR, 2019. <a href="https://arxiv.org/abs/1711.05101" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Gupta, V., Koren, T., and Singer, Y. <em>Shampoo: Preconditioned Stochastic Tensor Optimization</em>. ICML, 2018. <a href="https://arxiv.org/abs/1802.09568" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Jordan, K. et al. <em>Muon: An Optimizer for Hidden Layers in Neural Networks</em>. Original public technical write-up, 2024. <a href="https://kellerjordan.github.io/posts/muon/" target="_blank" rel="noopener noreferrer">write-up</a></li>
  <li>Liu, J. et al. <em>Muon is Scalable for LLM Training</em>. arXiv:2502.16982, 2025. <a href="https://arxiv.org/abs/2502.16982" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Li, Z., Liu, L., Liang, C., Chen, W., and Zhao, T. <em>NorMuon: Making Muon More Efficient and Scalable</em>. arXiv:2510.05491, 2025. <a href="https://arxiv.org/abs/2510.05491" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Ahn, K., Xu, B., Abreu, N., and Langford, J. <em>Dion: Distributed Orthonormalized Updates</em>. arXiv:2504.05295, 2025. <a href="https://arxiv.org/abs/2504.05295" target="_blank" rel="noopener noreferrer">arXiv</a></li>
</ol>

<h3>Spectral capacity and information-theoretic framing</h3>

<ol class="references-list" start="14">
  <li>Alemi, A. A., Poole, B., Fischer, I., Dillon, J. V., Saurous, R. A., and Murphy, K. <em>Fixing a Broken ELBO</em>. ICML, 2018. <a href="https://proceedings.mlr.press/v80/alemi18a.html" target="_blank" rel="noopener noreferrer">PMLR</a></li>
  <li>Rényi, A. <em>On Measures of Entropy and Information</em>. Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability, 1961. <a href="https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/On-Measures-of-Entropy-and-Information/chapter/On-Measures-of-Entropy-and-Information/bsmsp/1200512181" target="_blank" rel="noopener noreferrer">Project Euclid</a></li>
  <li>Roy, O., and Vetterli, M. <em>The Effective Rank: A Measure of Effective Dimensionality</em>. EUSIPCO, 2007. <a href="https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf" target="_blank" rel="noopener noreferrer">PDF</a></li>
</ol>

<h3>Inductive bias, interpretability, and plasticity</h3>

<ol class="references-list" start="17">
  <li>Goldblum, M., Finzi, M., Rowan, K., and Wilson, A. G. <em>The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning</em>. ICML, 2024. <a href="https://arxiv.org/abs/2304.05366" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Wilson, A. G. <em>Deep Learning is Not So Mysterious or Different</em>. ICML, 2025. <a href="https://arxiv.org/abs/2503.02113" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Elhage, N. et al. <em>A Mathematical Framework for Transformer Circuits</em>. Transformer Circuits, 2021. <a href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener noreferrer">article</a></li>
  <li>Mirzadeh, S. I., Chaudhry, A., Yin, D., Hu, H., Pascanu, R., Gorur, D., and Farajtabar, M. <em>Wide Neural Networks Forget Less Catastrophically</em>. ICML, 2022. <a href="https://arxiv.org/abs/2110.11526" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Lu, A., Yuan, H., Feng, T., and Sun, Y. <em>Rethinking the Stability-Plasticity Trade-off in Continual Learning from an Architectural Perspective</em>. ICML, 2025. <a href="https://arxiv.org/abs/2506.03951" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Abbas, Z., Zhao, R., Modayil, J., White, A., and Machado, M. C. <em>Loss of Plasticity in Continual Deep Reinforcement Learning</em>. arXiv:2303.07507, 2023. <a href="https://arxiv.org/abs/2303.07507" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Dohare, S., Hernandez-Garcia, J. F., Rahman, P., Mahmood, A. R., and Sutton, R. S. <em>Maintaining Plasticity in Deep Continual Learning</em>. Nature, 2024. <a href="https://www.nature.com/articles/s41586-024-07711-7" target="_blank" rel="noopener noreferrer">article</a></li>
  <li>Lyle, C., Zheng, Z., Khetarpal, K., van Hasselt, H., Pascanu, R., Martens, J., and Dabney, W. <em>Disentangling the Causes of Plasticity Loss in Neural Networks</em>. arXiv:2402.18762, 2024. <a href="https://arxiv.org/abs/2402.18762" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>He, N., Guo, K., Prakash, A., Tiwari, S., Tao, R. Y., Serapio, T., Greenwald, A., and Konidaris, G. <em>Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning</em>. arXiv:2509.22335, 2025. <a href="https://arxiv.org/abs/2509.22335" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Han, T., Bordt, S., Zhang, H., and Kakade, S. <em>Weight Decay Improves Language Model Plasticity</em>. arXiv:2602.11137, 2026. <a href="https://arxiv.org/abs/2602.11137" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Huang, J., Wurgaft, D., Bansal, R., Ruis, L., Saphra, N., Alvarez-Melis, D., Lampinen, A. K., Potts, C., and Lubana, E. S. <em>Why Larger Models Learn More: Effects of Capacity, Interference, and Rare-Task Retention</em>. arXiv:2605.29548, 2026. <a href="https://arxiv.org/abs/2605.29548" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Abel, D. et al. <em>Plasticity as the Mirror of Empowerment</em>. arXiv:2505.10361, 2025. <a href="https://arxiv.org/abs/2505.10361" target="_blank" rel="noopener noreferrer">arXiv</a></li>
</ol>

</div>
