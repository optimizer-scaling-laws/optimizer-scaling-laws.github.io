---
layout: post
title: "Architecture Creates Capacity, Optimizers Realize It"
subtitle: "Why LLM scaling laws should account for realized capacity, not just loss curves and parameter counts"
author: "Nandan Kumar Jha"
date: 2026-06-26
permalink: /blog/optimizer-induced-capacity/
description: "Why LLM scaling should account for realized capacity: the same architecture and training data can produce different internal spectral capacity under different optimizers."
reading_time: "~26 min read"
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
  list-style: none;
  margin: 0.75rem 0 0 0.6rem;
  padding-left: 0.6rem;
}
.toc-box > ol {
  counter-reset: sec;
}
.toc-box ol ol {
  counter-reset: sub;
  margin: 0.35rem 0 0.35rem 1.1rem;
  padding-left: 0;
}
.toc-box li {
  margin: 0.35rem 0;
}
.toc-box > ol > li {
  counter-increment: sec;
}
.toc-box ol ol > li {
  counter-increment: sub;
}
.toc-box > ol > li::before {
  content: counter(sec) ". ";
  color: #64748b;
  font-variant-numeric: tabular-nums;
}
.toc-box ol ol > li::before {
  content: counter(sec) "." counter(sub) "\00a0\00a0";
  color: #64748b;
  font-variant-numeric: tabular-nums;
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
<p><strong>Capacity is not only what the architecture makes possible, it is what training converts into variance-carrying representation.</strong></p>
<ul>
  <li><strong>Finding.</strong> Holding architecture, training data, tokenizer, and FFN-width schedule fixed, optimizer choice changes the spectral capacity realized inside FFN representations.</li>
  <li><strong>Matched loss is not matched representation.</strong> The same architecture can reach similar validation loss under different optimizers while learning different representation. Longer AdamW training can match Dion-1/16 in validation loss, but not in realized capacity scaling.</li>
  <li><strong>Implication.</strong> Architecture determines the available degrees of freedom, training dynamics determine which of them become active, variance-carrying representation directions, and how they are allocated across token-frequency regimes. The effect is strongest for rare tokens, where sparse supervision leaves more room for optimizer-induced bias to shape the learned representation.</li>
</ul>
</div>

Throughout this post, **realized capacity** means **realized spectral capacity**: variance-carrying eigenmodes in FFN eigenspectrum representation. Note that, it is not a complete measure of intelligence, transfer, or downstream capability; it is an internal telemetry signal that loss curves and parameter counts do not directly capture.

This post starts from a single finding and asks what follows from it. Classical scaling laws taught us to ask how loss changes with parameters, training data, and compute. They leave another question open: when we add architectural capacity, does training convert it into useful internal structure? In our experiments, that conversion depends strongly on optimizer choice, especially in rare-token regimes where supervision is sparse.


<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure0_same_architecture_same_data_different_optimizer.png' | relative_url }}" alt="Same architecture, same data, different optimizer conceptual setup">
  <figcaption><strong>Figure 0.</strong> <em>Same architecture, same training data, different optimizer.</em> With the model and data held fixed, different optimizers induce different paths through parameter space, which results in different internal spectral-capacity profiles even when final loss is similar.</figcaption>
</figure>

<details class="toc-box">
  <summary>On this page</summary>
  <ol>
    <li><a href="#same-architecture-different-capacity-scaling">Same architecture, different capacity scaling</a></li>
    <li><a href="#matched-loss-is-not-matched-representation">Matched loss is not matched representation</a></li>
    <li><a href="#rare-tokens-expose-optimizer-induced-capacity-allocation">Rare-token capacity allocation</a></li>
    <li>
      <a href="#three-views-of-the-same-gap">Three views of the same gap</a>
      <ol>
        <li><a href="#view-i-scalar-objectives-under-identify-internal-structure">Scalar objectives are not internal structure</a></li>
        <li><a href="#view-ii-nominal-capacity-is-not-realized-capacity">Nominal capacity is not realized capacity</a></li>
        <li><a href="#view-iii-reachable-capacity-is-optimizer-conditional">Reachable capacity is optimizer-conditional</a></li>
      </ol>
    </li>
    <li><a href="#why-optimizer-choice-changes-capacity-not-just-speed">Why optimizer choice changes capacity, not just speed</a></li>
    <li><a href="#the-design-object-is-the-architecture-optimizer-pair">Architecture–optimizer co-design</a></li>
    <li><a href="#what-changes-in-pretraining-practice">What changes in pretraining practice?</a></li>
    <li><a href="#what-this-does-not-claim">What this does not claim</a></li>
    <li><a href="#open-questions">Open questions</a></li>
    <li><a href="#conclusion-toward-capacity-aware-llm-design">Conclusion</a></li>
  </ol>
</details>

## 1. Same architecture, different capacity scaling
{: #same-architecture-different-capacity-scaling}

> **When architecture, training data, tokenizer, and FFN-width schedule are fixed, changing the optimizer changes how realized spectral capacity scales.**

Four quantities help interpret the result.

<div class="metric-box">
  <strong>Metrics used in this post.</strong>
  <ul>
    <li><strong>Diffuse capacity:</strong> soft spectral rank, how broadly representation variance spreads across eigenmodes.</li>
    <li><strong>Dominant-mode capacity:</strong> hard spectral rank, how many eigenmodes carry substantial variance.</li>
    <li><strong>Capacity asymmetry:</strong> the gap between soft and hard rank. It diagnoses whether capacity is broadly distributed across many eigenmodes or concentrated in a few dominant directions. Lower asymmetry means a more even spread.</li>
    <li><strong>Scaling exponent:</strong> how quickly realized capacity grows as FFN width increases.</li>
  </ul>
</div>

We vary FFN width within the same model family and compare how realized spectral capacity grows. The model does not merely train faster or slower; it converts the same architectural budget into a different internal-capacity profile.

Figure 1 gives the global view of realized capacity scaling. AdamW shows the weakest dominant-mode scaling ($\beta_{\mathrm{hard}} \approx 0.29$), while Muon and NorMuon achieve near-linear scaling ($\beta_{\mathrm{hard}} \approx 0.80$), roughly a $2.8\times$ larger exponent under the same architecture and training data. This gap becomes more pronounced in the rare-token regime, which Section 3 examines.

<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure1_optimizer_capacity_scaling.png' | relative_url }}" alt="Aggregated optimizer-level realized-capacity scaling">
  <figcaption><strong>Figure 1.</strong> <em>Aggregated (pooled across the token-regimes) optimizer-induced spectral scaling.</em> Architecture, training data, and FFN-width schedule are fixed; only the optimizer changes. Panel A shows scaling exponents for diffuse capacity, $\beta_{\mathrm{soft}}$, and dominant-mode capacity, $\beta_{\mathrm{hard}}$. Panel B shows how capacity asymmetry scales with width, $\Delta\beta = \beta_{\mathrm{soft}} - \beta_{\mathrm{hard}}$. Muon and NorMuon show stronger dominant-mode capacity scaling and a smaller $\Delta\beta$ than AdamW, indicating that added width converts into dominant-mode capacity.</figcaption>
</figure>

The full result tables, frequency-conditioned fits, and ablations are on the <a href="https://optimizer-scaling-laws.github.io/" target="_blank" rel="noopener noreferrer">project page</a>.

These are GPT-2-scale studies, including 160M and 350M model families on FineWeb-Edu. They show that optimizer choice can change realized spectral capacity-scaling under matched architecture and training data; they do not guarantee that the same optimizer ordering must persist at multi-billion-parameter scale. To ensure this, a calibrated billion-scale sweep that keeps architecture, training data, tokenizer, FFN-width schedule fixed, while measuring whether the same diffuse, dominant-mode, and HEAD/MID/TAIL effects persist.

<p class="takeaway-inline"><strong>Takeaway.</strong> Optimizer choice can change not only convergence speed or final loss, but the scaling law exponents by which added FFN width becomes realized spectral capacity.</p>


## 2. Matched loss is not matched representation
{: #matched-loss-is-not-matched-representation}

A natural objection is that one optimizer may merely train faster. The matched-loss comparison tests that possibility.

Extending AdamW training from 6K to 12K training steps improves validation perplexity and brings it close to Dion-1/16 across the FFN-width sweep. But the realized-capacity scaling does not match; AdamW-12K remains much weaker in dominant-mode capacity scaling.

The scaling trend is counterintuitive: longer AdamW training improves loss, however, weakens dominant-mode scaling---the hard-rank exponent drops from **0.29** to **0.03**. That is, the added FFN width no longer converts into spectral capacity, particularly in wider models, even when loss continues to improve.


<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure2_matched_loss_capacity_comparison.png' | relative_url }}" alt="Matched loss but different realized capacity">
  <figcaption><strong>Figure 2.</strong> <em>Matched loss, different realized capacity.</em> Extending AdamW training improves perplexity — closer to Dion-1/16. However, the realized-capacity profiles remain substantially different, particularly the hard-rank scaling trend.</figcaption>
</figure>

This gap is not an artifact of learning-rate tuning. Our learning-rate sweep shows that the spectral gap persists across learning rates. Practically, two training runs can look similar by loss while realizing very different internal capacity. Loss measures average output error; it does not establish that training produced the same representation directions, solution geometry, or adaptation-relevant internal structure.

<p class="takeaway-inline"><strong>Takeaway.</strong> Matched validation loss can still mask different width-to-capacity realization trajectories. Loss matching is necessary for a fair comparison, but it is not enough to establish matched representation geometry.</p>


## 3. Rare tokens expose optimizer-induced capacity allocation
{: #rare-tokens-expose-optimizer-induced-capacity-allocation}

Natural language is long-tailed. A small number of HEAD tokens receive dense, repeated training signal; TAIL tokens receive sparse, noisy, intermittent signal.

This suggests a straightforward regime-dependent interpretation. For HEAD tokens, dense supervision leaves optimizers less room to produce qualitatively different representations. For TAIL tokens, sparse supervision gives optimizer-induced bias more degrees of freedom in determining which weak signals become variance-carrying representation directions.

In Bayesian terms, dense regimes are likelihood-dominated, while sparse regimes are more prior-sensitive. In this analogy, the optimizer contributes an implicit bias over which solutions training is likely to reach.


<table>
<thead>
<tr>
<th>Token regime</th>
<th>How constrained by data?</th>
<th>Key spectral observations</th>
<th>Design focus</th>
</tr>
</thead>
<tbody>
<tr>
<td>HEAD</td>
<td>Strong</td>
<td>Optimizer gap is smallest; architecture is most competitive in hard-rank scaling</td>
<td>Architecture-sensitive</td>
</tr>
<tr>
<td>MID</td>
<td>Partial</td>
<td>Optimizer gap widens; Muon/NorMuon reduce soft–hard asymmetry relative to AdamW</td>
<td>Architecture–optimizer co-design</td>
</tr>
<tr>
<td>TAIL</td>
<td>Weak</td>
<td>Optimizer choice dominates hard-rank scaling ($\beta_{\mathrm{hard}} \approx 0.44$ vs $\approx 1.0$)</td>
<td>Optimizer and training dynamics</td>
</tr>
</tbody>
</table>


All three regimes are measured directly in Figure 3, so the table summarizes observed spectral behavior rather than extrapolating from a single regime. Token frequency is a proxy for how strongly the data constrains a token’s representation. The optimizer effect is present across the distribution, but grows as that constraint weakens — smallest in HEAD, larger in MID, and strongest in TAIL. The last column should be read narrowly: HEAD is not “architecture-only”; rather, in these hard-rank measurements, architectural changes are most competitive in the HEAD regime.

This is why architecture-only changes can under-deliver for long-tail behavior: adding capacity raises the nominal ceiling, but sparse training signal does not automatically convert that capacity into variance-carrying directions. The optimizer may need to preserve and amplify weak signals rather than let them vanish or fragment.

This makes the result relevant to capability-oriented pretraining questions. If realized capacity affects transfer in sparse regimes, then low-resource languages, rare scientific and code vocabulary, long-tail factual recall, tool-use edge cases, and sparse expert specialization are natural places to test this diagnostic. Spectral measurement alone does not prove that behavioral connection.

Figure 3 resolves the aggregate result from Figure 1 into HEAD, MID, and TAIL regimes. The left panel shows dominant-mode scaling through $\beta_{\mathrm{hard}}$. In TAIL representations, AdamW and Dion-1/16 have substantially weaker hard-rank scaling, with AdamW at $\beta_{\mathrm{hard}} \approx 0.44$, while Muon and NorMuon approach near-linear dominant-mode growth, $\beta_{\mathrm{hard}} \approx 1.0$---roughly a $2.3\times$ larger exponent. For Dion-1/16, spectral scaling behavior remains closer to AdamW than to Muon/NorMuon in the TAIL regime. The right panel shows the scaling asymmetry, $\Delta\beta = \beta_{\mathrm{soft}} - \beta_{\mathrm{hard}}$: Muon and NorMuon keep soft- and hard-rank scaling nearly aligned, while AdamW shows a larger gap.


<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure4_capacity_scaling_asymmetry_token_regimes.png' | relative_url }}" alt="Frequency-conditioned realized-capacity scaling across token regimes">
  <figcaption><strong>Figure 3.</strong> <em>Frequency-conditioned realized-capacity scaling.</em> Panel A shows dominant-mode capacity scaling exponents, $\beta_{\mathrm{hard}}$, across HEAD, MID, and TAIL token regimes. Panel B shows how capacity asymmetry scales with width, $\Delta\beta = \beta_{\mathrm{soft}} - \beta_{\mathrm{hard}}$. The regime-conditioned view reveals where the optimizer effect is strongest: MID and especially TAIL regimes.</figcaption>
</figure>

Since loss is an aggregate metric, it can be dominated by frequent tokens and obscure whether rare tokens receive realized spectral capacity. A capacity-aware training run should therefore ask not only whether average loss improved, but whether the long tail received measurable internal structure that predicts rare-regime behavior.

<p class="takeaway-inline"><strong>Takeaway.</strong> Rare-token regimes are where optimizer-induced bias most strongly shapes which weak signals become variance-carrying representation directions.</p>

## 4. Three views of the same gap
{: #three-views-of-the-same-gap}

The result is not merely an optimizer-specific observation. It points to a broader gap: scalar objective value, nominal architecture, and learned internal representation are often conflated, but they are not the same object.

Section 2 made this concrete: matched validation loss can mask different spectral-capacity trajectories. The three views below generalize it for **pretraining science**.

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
<td>Scalar objectives are not internal structure</td>
<td>Similar loss can mask different learned representations.</td>
<td>Places the matched-loss result inside a broader failure mode of loss-only comparison.</td>
</tr>
<tr>
<td>Nominal capacity is not realized capacity</td>
<td>Width matters only if training converts it into variance-carrying representation directions.</td>
<td>Names the missing internal axis: how much architectural capacity becomes realized spectral capacity.</td>
</tr>
<tr>
<td>Reachable solution is optimizer-conditional</td>
<td>The same architecture does not imply the same reachable solution.</td>
<td>Explains why optimizer-induced bias can shape capacity allocation, even under fixed architecture.</td>
</tr>
</tbody>
</table>

The sequence moves from measurement to mechanism: scalar objectives can obscure the model’s internal state, realized capacity names the missing internal axis, and optimizer-induced bias explains why the reachable solution can change under fixed architecture.


### View I — Scalar objectives are not internal structure
{: #view-i-scalar-objectives-under-identify-internal-structure}

Loss, gradient norms, and downstream evaluations are useful signals, but no single scalar metric fully describes a trained model’s learned representation or solution geometry. The same value can arise from different internal structures: variance may spread across many directions, concentrate in a few dominant modes, or appear unevenly across token-frequency regimes. The matched-loss result above is one spectral-capacity instance of this broader under-identification problem.

The upstream–downstream literature gives empirical precedent. The same pretraining loss can still produce different downstream transfer (<a href="https://arxiv.org/abs/2210.14199" target="_blank" rel="noopener noreferrer">Liu et al., 2022</a>), different geometry among task-specific minima (<a href="https://arxiv.org/abs/2604.09258" target="_blank" rel="noopener noreferrer">Chen et al., 2026</a>), and non-monotonic changes in learned representation geometry that scalar metrics miss (<a href="https://arxiv.org/abs/2509.23024" target="_blank" rel="noopener noreferrer">Li et al., 2025a</a>). Theoretical work makes the same point: low pretraining loss alone need not guarantee that every downstream-relevant feature is recoverable from the representation (<a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/93712c59f6a81bd92040facf04c8b308-Abstract-Conference.html" target="_blank" rel="noopener noreferrer">Wu, Lee, and Ge, 2023</a>). In this post, realized spectral capacity provides one measurable axis of that under-identification.


Rate–distortion theory gives a useful formal analogy. A scalar objective can identify one point on a trade-off surface without specifying the internal allocation that produced it. <a href="https://proceedings.mlr.press/v80/alemi18a.html" target="_blank" rel="noopener noreferrer">Alemi et al., 2018</a> show that fixing the ELBO corresponds to a point on a rate--distortion curve, so models with the same ELBO can occupy different rate--distortion tradeoffs; changing the objective weights moves the solution along that curve. <a href="https://proceedings.mlr.press/v119/gao20a.html" target="_blank" rel="noopener noreferrer">Gao and Chaudhari, 2020</a> generalize this to an equilibrium surface relating rate, distortion, and loss, where different multipliers can reach different internal tradeoffs at the same loss.

Our setting has the same structural shape, but not the same literal variables. Validation loss is the scalar coordinate; realized spectral capacity is one internal allocation it leaves underdetermined; and optimizer choice plays a role analogous to a multiplier, selecting which region of training geometry is reached. Spectral rank is not rate, and we impose no explicit rate–distortion objective. 


The question is therefore not only **how low did loss go?** but also:

> **What internal capacity did training realize in order to reach that loss?**

### View II — Nominal capacity is not realized capacity
{: #view-ii-nominal-capacity-is-not-realized-capacity}

The architecture–optimizer distinction becomes clearest if we separate two notions of capacity.

> **Nominal capacity** is the capacity implied by the architecture: parameter count, width, depth, heads, experts, routing paths, memory layout, and FLOPs.

> **Realized spectral capacity** is the capacity measured as active variance-carrying structure inside the trained model: which representation directions become active, how variance is distributed across eigenmodes, which modes grow with width.

Architecture gives the learning system things optimization cannot create: causal masking, equivariance, sparse routing, dimensional ceilings, and residual topology. But architecture creates nominal capacity — the available degrees of freedom — and does not guarantee that training will use all of them.

A useful mental model is:

<div class="math-box">
$$
C_{\mathrm{realized}}(\mathcal{A}, \mathcal{O}, \mathcal{D})
\;\approx\;
C_{\mathrm{nominal}}(\mathcal{A})
\;\times\;
\rho_{\mathrm{realized}}(\mathcal{O}, \mathcal{D}; \mathcal{A}).
$$
</div>

Here, $\mathcal{A}$ is the architecture, $\mathcal{O}$ is the optimizer/training algorithm, and $\mathcal{D}$ is the training data. This is not a literal scalar law; it is a design principle. Architecture sets nominal capacity. Optimization and training data influence how much of that capacity realized in representation space. Whether that structure is behaviorally useful must be tested separately.

The same architectural intervention can therefore have different realized effects under different optimizers. If $\rho_{\mathrm{realized}}$ is high, added width translates into measured representation capacity. If it is low, added width raises the ceiling without filling it.

To measure this distinction, we look at spectral geometry. Given a representation covariance, its eigenspectrum characterizes how variance is distributed across eigenmodes. Effective-rank measures summarize that distribution; the entropy-based formulation traces back to <a href="https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf" target="_blank" rel="noopener noreferrer">Roy and Vetterli, 2007</a>, and the Rényi family varies sensitivity to diffuse versus dominant eigenmodes (<a href="https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/On-Measures-of-Entropy-and-Information/chapter/On-Measures-of-Entropy-and-Information/bsmsp/1200512181" target="_blank" rel="noopener noreferrer">Rényi, 1961</a>).

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

These are the two capacity measures from Section 1, now made precise: diffuse capacity is the soft spectral rank, $R_{\mathrm{soft}} = R_1$; dominant-mode capacity is the hard spectral rank, $R_{\mathrm{hard}} = R_2$. Capacity asymmetry is their log-ratio, $\log\left(R_{\mathrm{soft}} / R_{\mathrm{hard}}\right)$, the multiplicative gap between diffuse and dominant-mode capacity in a single model. 

Because $R_{\mathrm{soft}}$ and $R_{\mathrm{hard}}$ are Rényi effective ranks of orders 1 and 2, and Rényi rank is non-increasing in the order, the hard rank can never exceed the soft rank. This makes the upper-left region of the phase map in Figure 4 necessarily forbidden, not merely unobserved, and the frontier $R_{\mathrm{hard}} = R_{\mathrm{soft}}$ a true boundary. The exponent gap reported in the figures, $\Delta\beta = \beta_{\mathrm{soft}} - \beta_{\mathrm{hard}}$, measures how capacity asymmetry scales as the FFN widens.  

<figure class="figure-wide">
  <img src="{{ '/assets/blog/architecture-optimizer-codesign/figure3_phase_diagram_for_realized_capacity.png' | relative_url }}" alt="Phase map for realized capacity">
  <figcaption><strong>Figure 4.</strong> <em>Conceptual phase map for realized capacity.</em> The horizontal axis tracks diffuse capacity, using normalized $\log R_{\mathrm{soft}}$; the vertical axis tracks dominant-mode capacity, using normalized $\log R_{\mathrm{hard}}$. The shaded forbidden region reflects the forced ordering $R_{\mathrm{hard}} \le R_{\mathrm{soft}}$. The low-asymmetry frontier marks the regime where diffuse variance and dominant eigenmode variance are closely matched.</figcaption>
</figure>

This phase map separates realized-capacity profiles that a single loss value can conflate: collapsed, with low capacity; compact, with moderate capacity and low asymmetry; broad, with high capacity and low asymmetry; and brittle, with high asymmetry. Capacity asymmetry measures whether broad spectral spread is matched by substantial variance in dominant eigenmodes. This is what separates brittle profiles from compact and broad ones.


### View III — Reachable capacity is optimizer-conditional
{: #view-iii-reachable-capacity-is-optimizer-conditional}

Realized capacity can differ under one architecture because learning is shaped by inductive bias, and architecture is only part of it. Since data rarely specifies a unique model, a learner needs biases that privilege some solutions over others (<a href="https://arxiv.org/abs/2304.05366" target="_blank" rel="noopener noreferrer">Goldblum et al., 2024</a>). Architecture supplies a hard bias: causal masking, residual topology, attention structure, parameter sharing, and routing restrict the support of possible computation. The optimizer supplies a softer one — it leaves the formal hypothesis space largely intact but changes which solutions are reachable, stable, and likely under a given training budget, in line with the view that modern overparameterized learning keeps a flexible hypothesis space and imposes soft preferences over data-consistent solutions (<a href="https://arxiv.org/abs/2503.02113" target="_blank" rel="noopener noreferrer">Wilson, 2025</a>). What this post adds is a way to quantify that soft preference: realized spectral capacity is one axis along which it becomes measurable, and along which optimizers can be distinguished even where their loss curves cannot.

#### Forward-looking implication: realized capacity may matter for plasticity
{: #forward-looking-implication-realized-capacity-may-matter-for-plasticity}

Continual learning raises a forward-looking version of the same question. Forgetting concerns whether a model retains the past; plasticity concerns whether it can still learn the future. This is a conjecture rather than something the spectral measurements establish: the paper measures realized capacity at a fixed point, whereas future learnability depends on whether realized or weakly used directions remain available for later adaptation. The conjecture is worth stating because the insensitivity of loss to current realized capacity, documented above, recurs for future adaptability. Han et al. report that larger pretraining weight decay yields more adaptable base models — greater downstream gains after fine-tuning — even when a lower validation loss would have selected a different checkpoint (<a href="https://arxiv.org/abs/2602.11137" target="_blank" rel="noopener noreferrer">Han et al., 2026</a>), the same decoupling of loss from the property of interest, now one stage later in the training pipeline. The most direct evidence appears at LLM scale: GPT-style models trained on long multilingual streams measurably lose the ability to learn a held-out language, and they do so even under stationary pretraining, where validation loss continues to fall while adaptability declines (<a href="https://arxiv.org/abs/2606.24752" target="_blank" rel="noopener noreferrer">Hernandez-Garcia et al., 2026</a>). A decreasing loss and a declining ability to learn are not mutually exclusive; here they co-occur.

If realized capacity does bound later learnability, the relevant quantity is the number of usable directions — the effective rank this post already tracks. A recurring signature in the plasticity literature is the loss of usable directions: across reinforcement learning, vision, and language, it appears as dormant units, a falling effective feature rank, or a degenerating curvature spectrum (<a href="https://arxiv.org/abs/2303.07507" target="_blank" rel="noopener noreferrer">Abbas et al., 2023</a>; <a href="https://www.nature.com/articles/s41586-024-07711-7" target="_blank" rel="noopener noreferrer">Dohare et al., 2024</a>; <a href="https://arxiv.org/abs/2402.18762" target="_blank" rel="noopener noreferrer">Lyle et al., 2024</a>; <a href="https://arxiv.org/abs/2509.22335" target="_blank" rel="noopener noreferrer">He et al., 2025</a>) — and at LLM scale the same pattern is observed directly, as dormant FFN units and attention heads that collapse (over-concentrate) or become diffuse (near-uniform) as the model's ability to learn from new data declines. What differs across these results is not the signature but the mechanism that produces it. In Wilson's hard/soft terminology, architecture provides the harder structural constraint, while optimization provides the softer dynamical bias that determines which capacity is actually realized during training.

**Two complementary biases on one quantity.** Realized capacity is the number of usable representation directions; architecture and the optimizer act on it from opposite sides.

<table>
<thead><tr><th></th><th>Architecture — structural bias</th><th>Optimizer — dynamical bias</th></tr></thead>
<tbody>
<tr><td>Role</td><td>Creates capacity by constraining which directions <em>can</em> carry signal</td><td>Realizes capacity by shaping which directions <em>actually do</em> carry signal</td></tr>
<tr><td>When it acts</td><td>Chosen at design time; constrains the training trajectory throughout</td><td>Acts throughout training; tunable via optimizer choice, schedule, and regularization</td></tr>
<tr><td>How capacity is lost</td><td>Rank bottlenecks, poor width/depth balance, unstable or collapsing pathways</td><td>Dormant units, gradient singular-value collapse, norm growth, optimizer-induced concentration</td></tr>
<tr><td>How capacity is preserved</td><td>Non-collapsing rank pathways, width/depth balance, architecture-level plasticity mechanisms</td><td>Orthogonalized updates, weight decay, targeted resets, and optimizer choices that keep spectra balanced</td></tr>
</tbody>
</table>

The lower two rows of the table correspond to concrete mechanisms. On the optimizer side, PAPO restores plasticity by adding a fraction of the gradient's polar factor to each update, uniformly raising the near-zero singular values that stability-preserving methods drive toward zero (<a href="https://openreview.net/forum?id=b7P2WegaBY" target="_blank" rel="noopener noreferrer">Zheng et al., 2026</a>); that polar factor is mathematically close to the orthogonalization operation used in Muon-style updates. This makes the connection suggestive: the same class of geometry-shaping updates that changes spectral capacity here may also help preserve plasticity. On the architecture side, width and depth divide the trade-off — wider networks forget less, deeper networks remain more plastic (<a href="https://arxiv.org/abs/2110.11526" target="_blank" rel="noopener noreferrer">Mirzadeh et al., 2022</a>; <a href="https://arxiv.org/abs/2506.03951" target="_blank" rel="noopener noreferrer">Lu et al., 2025</a>) — and interpolation layers maintain a non-collapsing rank contribution with no change to the optimizer (<a href="https://openreview.net/forum?id=pAhGjPOlwy" target="_blank" rel="noopener noreferrer">Koeppe et al., 2026</a>). Scale belongs to neither column and resolves neither: additional parameters postpone the onset of plasticity loss, but only sublinearly — delaying the collapse without preventing it — even as larger models retain rare-task structure longer by reducing interference (<a href="https://arxiv.org/abs/2605.29548" target="_blank" rel="noopener noreferrer">Huang et al., 2026</a>).

The continual-learning reading is not that architecture ceases to matter. It is that capacity must be realized during pretraining and preserved in a form that remains available for later learning — which is also what a formal account of plasticity measures: how much an agent can still be shaped by what it observes (<a href="https://arxiv.org/abs/2505.10361" target="_blank" rel="noopener noreferrer">Abel et al., 2025</a>). Scale postpones the loss of that form but does not guarantee it; which forms survive depends jointly on architecture, optimizer, regularization, and training dynamics.

## 5. Why optimizer choice changes capacity, not just speed
{: #why-optimizer-choice-changes-capacity-not-just-speed}

Views I through III established that optimizer choice changes realized capacity; they did not establish why an optimizer should affect representation geometry rather than only convergence speed. The reason is that an optimizer does more than set how quickly the loss decreases: it transforms raw gradients into parameter updates, and that update geometry — coordinate-wise scaling, weight decay, matrix preconditioning, orthogonalization, momentum, or constrained subspaces — can change which representation directions grow, which eigenmodes accumulate variance, and which weak signals persist long enough to become usable structure.

<table>
<thead><tr><th>Mechanism</th><th>Examples</th><th>How it can affect realized capacity</th></tr></thead>
<tbody>
<tr><td>Adaptive scaling and weight decay</td><td>AdamW-style updates (<a href="https://arxiv.org/abs/1711.05101" target="_blank" rel="noopener noreferrer">Loshchilov and Hutter, 2019</a>)</td><td>Change per-parameter update scale, norm growth, and which same-loss basins are easier to reach</td></tr>
<tr><td>Matrix or tensor preconditioning</td><td>Shampoo-style structure-aware updates (<a href="https://arxiv.org/abs/1802.09568" target="_blank" rel="noopener noreferrer">Gupta, Koren, and Singer, 2018</a>)</td><td>Change the conditioning of weight updates and how added width is converted into representation modes</td></tr>
<tr><td>Matrix orthogonalization</td><td>Muon and NorMuon-style updates (<a href="https://kellerjordan.github.io/posts/muon/" target="_blank" rel="noopener noreferrer">Jordan et al., 2024</a>; <a href="https://arxiv.org/abs/2502.16982" target="_blank" rel="noopener noreferrer">Liu et al., 2025</a>; <a href="https://arxiv.org/abs/2510.05491" target="_blank" rel="noopener noreferrer">Li et al., 2025b</a>)</td><td>Change the geometry of updates across weight matrices, which can alter how variance spreads into dominant eigenmodes</td></tr>
<tr><td>Low-rank or communication-constrained updates</td><td>Dion-style updates (<a href="https://arxiv.org/abs/2504.05295" target="_blank" rel="noopener noreferrer">Ahn et al., 2025</a>)</td><td>Restrict the update subspace and can reshape how capacity is allocated across modes and token regimes</td></tr>
</tbody>
</table>

The claim is not that one optimizer mechanism is universally better. It is that each optimizer defines a different training geometry. These choices do not change the formal architecture, but they change the trajectory through parameter space: which basins are reachable, which update directions are amplified, which weak signals survive, and which representation eigendirections become variance-carrying structure. This makes matched loss an insufficient description of what the model has learned internally. Optimizer choice therefore belongs inside the capacity-scaling object, not merely in the training recipe.

## 6. The design object is the architecture–optimizer pair
{: #the-design-object-is-the-architecture-optimizer-pair}

Architecture and optimization are complementary, not competing: they solve different parts of one design problem. The division this post measures for capacity — architecture sets the available degrees of freedom, the optimizer sets the realized fraction and its allocation — recurs across other design goals as well:

<table>
<thead><tr><th>If the goal is...</th><th>Architecture contributes...</th><th>Optimization contributes...</th></tr></thead>
<tbody>
<tr><td>Guarantees by construction</td><td>Masking, routing, and equivariance</td><td>Trains within these constraints; does not create them</td></tr>
<tr><td>Long-tail capability</td><td>Representational capacity and routing paths</td><td>Can preserve weak signals and shape rare-token directions</td></tr>
<tr><td>Stable deep training</td><td>Residuals, normalization, and parameterization</td><td>Conditioning, momentum, regularization, and reachability</td></tr>
<tr><td>Capacity-aware scaling</td><td>Available degrees of freedom</td><td>Realized fraction and allocation of those degrees of freedom</td></tr>
</tbody>
</table>

In practice, scaling decisions should specify both architecture and optimizer, and optimizer comparisons should report not only speed and loss but also the quality of the representation they induce. This raises a concrete pretraining question: what should we log to know whether added architectural capacity became realized internal structure?

## 7. What changes in pretraining practice?
{: #what-changes-in-pretraining-practice}

Loss remains the central training objective and a low-variance signal for scaling laws; nothing here proposes replacing it. What loss alone does not reveal is the kind of solution the model is converging toward, whether it is using its parameter budget effectively, or whether the architecture–optimizer pair provides the right inductive biases. For that, we need internal telemetry.

A capacity-aware pretraining report should answer five practical questions:

<table>
<thead><tr><th>Question</th><th>Diagnostic</th><th>Failure mode it catches</th></tr></thead>
<tbody>
<tr><td>Did same-loss models build the same representation?</td><td>Matched-loss optimizer comparisons</td><td>Similar perplexity masking different representation geometry</td></tr>
<tr><td>Does added FFN width convert into realized capacity?</td><td>Diffuse and dominant-mode capacity scaling</td><td>Parameters increasing while dominant modes do not grow</td></tr>
<tr><td>Is capacity broad but weakly concentrated?</td><td>Capacity asymmetry</td><td>Diffuse capacity growing without matching dominant-mode structure</td></tr>
<tr><td>Where in the data distribution does capacity appear?</td><td>Frequency-conditioned capacity across HEAD, MID, and TAIL regimes</td><td>Average loss improving while long-tail regimes remain under-realized</td></tr>
<tr><td>Is the effect tied to a design pair?</td><td>Architecture–optimizer interaction sweeps</td><td>Attributing to architecture alone what depends on the optimizer</td></tr>
</tbody>
</table>

These diagnostics are telemetry signals, not extra philosophical commitments. They make optimizer comparisons more informative: not only which optimizer reaches a target loss fastest, but how each one realizes the same architecture's capacity.

The same measures extend beyond a single training run. Classical scaling laws predict loss from parameters, data, and compute (<a href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noopener noreferrer">Kaplan et al., 2020</a>; <a href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noopener noreferrer">Hoffmann et al., 2022</a>) and remain central; a capacity-aware scaling law would complement them with internal variables — diffuse capacity, dominant-mode capacity, capacity asymmetry, and frequency-conditioned capacity as functions of width, depth, optimizer, and data. This aligns with recent position work arguing that AI systems should be studied as training processes, not only as static artifacts to analyze or patch after training (<a href="https://arxiv.org/abs/2606.06533" target="_blank" rel="noopener noreferrer">Biderman et al., 2026</a>).

Logging this telemetry is inexpensive. Soft and hard ranks are computed from the eigenspectrum of a layer's FFN post-activation covariance, requiring only eigenvalues rather than stored eigenvectors. In our prior ICLR 2026 work, NerVE (<a href="https://arxiv.org/abs/2603.06922" target="_blank" rel="noopener noreferrer">Jha and Reagen, 2026</a>), logging them every 1,000 steps on GPT-2-scale runs added roughly 1% wall-clock overhead and tens of megabytes of GPU memory per layer. The telemetry is not uniformly robust, however: pre-activation spectra remain stable under token subsampling, while post-activation spectra are more sensitive — especially hard-rank estimates in the tail — because the nonlinearity and token sparsity make the mid-to-tail eigenspectrum easier to distort. This suggests a two-level strategy: pre-activation soft and hard ranks for low-cost, frequent monitoring, and full-batch post-activation ranks when making claims about realized capacity.


## 8. What this does not claim
{: #what-this-does-not-claim}

The claim here is narrow: holding architecture, data, and tokenizer fixed, optimizer choice changes realized spectral capacity. The boundaries below mark what it does not establish.

First, spectral rank is not a complete theory of intelligence, generalization, or downstream ability. It is a telemetry signal that should be interpreted alongside loss, evaluations, robustness, calibration, interpretability, and task-specific metrics.

Second, more realized spectral capacity is not automatically better. The useful question is where capacity appears, whether it is stable, and whether it predicts behavior, robustness, transfer, or future learnability.

Third, co-design is not optimizer maximalism. Architecture remains indispensable: optimizers cannot create the guarantees architecture provides by construction, reduce inference cost structurally, or represent functions excluded by the architecture.

Fourth, the scale question remains open. The present evidence establishes an optimizer-induced capacity-scaling effect at GPT-2 160M/350M scale. Stronger frontier-scale claims require larger models, longer training, more architectures, downstream probes, continued-learning tests, interpretability comparisons, and direct studies of rare-regime behavior.

## 9. Open questions
{: #open-questions}

If realized capacity is measurable and optimizer-conditional, the open questions become concrete: what it predicts, what it reveals about internal computation, how architecture and optimization interact, and how to act on it in design.

**1. Which spectral differences predict downstream behavior?**
If two models have similar loss but different realized capacity, which spectral axes predict transfer, robustness, rare-token reliability, or domain adaptation?

**2. Does realized capacity predict plasticity and future learnability?**
A model with similar loss but different spectral allocation may differ in its ability to keep learning. Capacity telemetry may therefore help diagnose loss of plasticity, representational collapse, or exhaustion of useful directions during continued training.

**3. Are effective computational graphs optimizer-conditional?**
If the optimizer changes which routes through the model carry meaningful signal, then circuits discovered in one trained model may not be stable across optimizers, even with the same architecture (<a href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener noreferrer">Elhage et al., 2021</a>).

**4. How do architecture and optimizer modulate each other?**
The effect of an architectural change can depend on the optimizer. Removing RoPE, for instance, shifts perplexity by different amounts across optimizers. Likewise, the FFN nonlinearity can reshape spectra in an optimizer-dependent way: the optimizer that enters the nonlinearity with the most capacity is not always the one that leaves with it. Whether these interactions are systematic — predictable from the architecture–optimizer pair rather than either component alone — remains largely unstudied.

**5. How should we search over architecture–optimizer pairs?**
Neural architecture search typically fixes the optimizer. Optimizer evaluation typically fixes the architecture. The co-design view suggests that this may miss regions where neither component looks optimal alone, but the pair is strong.


## 10. Conclusion: toward capacity-aware LLM design
{: #conclusion-toward-capacity-aware-llm-design}

The core claim is direct: if two optimizers train the same architecture but produce different spectral-capacity scaling, optimizer choice has changed the model's realized capacity. The matched-loss comparison in Figure 2 strengthens the point: similar validation loss does not imply matched representation geometry. Figure 3 shows where the measured difference is largest: MID and TAIL regimes, where optimizer-induced bias can shape capacity allocation across the data distribution.

The three views above point to the same gap: scalar objectives are not internal structure, nominal capacity is not realized capacity, and reachable capacity is optimizer-conditional.

The next generation of LLM design should therefore ask more than how many parameters we train, how many tokens we consume, or how low the loss goes. It should also ask what capacity becomes realized and where it appears in the data distribution.

> **Capacity-aware LLM design means tracking not only what the model could represent, but what training actually realized — and then testing which realized directions support behavior, transfer, and future learning.**

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

<h3>Upstream–downstream behavior</h3>

<ol class="references-list" start="3">
  <li>Liu, H., Xie, S. M., Li, Z., and Ma, T. <em>Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models</em>. arXiv:2210.14199, 2022. <a href="https://arxiv.org/abs/2210.14199" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Wu, J., Lee, J. D., and Ge, R. <em>Connecting Pre-trained Language Models and Downstream Tasks via Properties of Representations</em>. NeurIPS, 2023. <a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/93712c59f6a81bd92040facf04c8b308-Abstract-Conference.html" target="_blank" rel="noopener noreferrer">NeurIPS</a></li>
  <li>Chen, H., Zhang, H., Li, X., Dong, Y., Shen, K., and Zhu, J. <em>Nexus: Same Pretraining Loss, Better Downstream Generalization via Common Minima</em>. arXiv:2604.09258, 2026. <a href="https://arxiv.org/abs/2604.09258" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Li, M. Z., Agrawal, K. K., Ghosh, A., Teru, K. K., Santoro, A., Lajoie, G., and Richards, B. A. <em>Tracing the Representation Geometry of Language Models from Pretraining to Post-training</em>. arXiv:2509.23024, 2025a. <a href="https://arxiv.org/abs/2509.23024" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Biderman, S., Khan, M. A., Mireshghallah, N., Arnett, C., Barez, F., and Saphra, N. <em>Position: Don't Just "Fix it in Post": A Science of AI Must Study Training Dynamics</em>. arXiv:2606.06533, 2026. <a href="https://arxiv.org/abs/2606.06533" target="_blank" rel="noopener noreferrer">arXiv</a></li>
</ol>

<h3>Optimization and optimizer-induced bias</h3>

<ol class="references-list" start="8">
  <li>Loshchilov, I., and Hutter, F. <em>Decoupled Weight Decay Regularization</em>. ICLR, 2019. <a href="https://arxiv.org/abs/1711.05101" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Gupta, V., Koren, T., and Singer, Y. <em>Shampoo: Preconditioned Stochastic Tensor Optimization</em>. ICML, 2018. <a href="https://arxiv.org/abs/1802.09568" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Jordan, K. et al. <em>Muon: An Optimizer for Hidden Layers in Neural Networks</em>. Original public technical write-up, 2024. <a href="https://kellerjordan.github.io/posts/muon/" target="_blank" rel="noopener noreferrer">write-up</a></li>
  <li>Liu, J. et al. <em>Muon is Scalable for LLM Training</em>. arXiv:2502.16982, 2025. <a href="https://arxiv.org/abs/2502.16982" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Li, Z., Liu, L., Liang, C., Chen, W., and Zhao, T. <em>NorMuon: Making Muon More Efficient and Scalable</em>. arXiv:2510.05491, 2025b. <a href="https://arxiv.org/abs/2510.05491" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Ahn, K., Xu, B., Abreu, N., and Langford, J. <em>Dion: Distributed Orthonormalized Updates</em>. arXiv:2504.05295, 2025. <a href="https://arxiv.org/abs/2504.05295" target="_blank" rel="noopener noreferrer">arXiv</a></li>
</ol>

<h3>Spectral capacity and information-theoretic framing</h3>

<ol class="references-list" start="14">
  <li>Alemi, A. A., Poole, B., Fischer, I., Dillon, J. V., Saurous, R. A., and Murphy, K. <em>Fixing a Broken ELBO</em>. ICML, 2018. <a href="https://proceedings.mlr.press/v80/alemi18a.html" target="_blank" rel="noopener noreferrer">PMLR</a></li>
  <li>Gao, Y., and Chaudhari, P. <em>A Free-Energy Principle for Representation Learning</em>. ICML, 2020. <a href="https://proceedings.mlr.press/v119/gao20a.html" target="_blank" rel="noopener noreferrer">PMLR</a></li>
  <li>Rényi, A. <em>On Measures of Entropy and Information</em>. Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability, 1961. <a href="https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/On-Measures-of-Entropy-and-Information/chapter/On-Measures-of-Entropy-and-Information/bsmsp/1200512181" target="_blank" rel="noopener noreferrer">Project Euclid</a></li>
  <li>Roy, O., and Vetterli, M. <em>The Effective Rank: A Measure of Effective Dimensionality</em>. EUSIPCO, 2007. <a href="https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf" target="_blank" rel="noopener noreferrer">PDF</a></li>
  <li>Jha, N. K., and Reagen, B. <em>NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks</em>. ICLR, 2026. <a href="https://arxiv.org/abs/2603.06922" target="_blank" rel="noopener noreferrer">arXiv</a></li>
</ol>

<h3>Inductive bias and interpretability</h3>

<ol class="references-list" start="19">
  <li>Goldblum, M., Finzi, M., Rowan, K., and Wilson, A. G. <em>The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning</em>. ICML, 2024. <a href="https://arxiv.org/abs/2304.05366" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Wilson, A. G. <em>Deep Learning is Not So Mysterious or Different</em>. ICML, 2025. <a href="https://arxiv.org/abs/2503.02113" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Elhage, N. et al. <em>A Mathematical Framework for Transformer Circuits</em>. Transformer Circuits, 2021. <a href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noopener noreferrer">article</a></li>
</ol>

<h3>Plasticity and continual learning</h3>

<ol class="references-list" start="22">
  <li>Mirzadeh, S. I., Chaudhry, A., Yin, D., Hu, H., Pascanu, R., Gorur, D., and Farajtabar, M. <em>Wide Neural Networks Forget Less Catastrophically</em>. ICML, 2022. <a href="https://arxiv.org/abs/2110.11526" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Lu, A., Yuan, H., Feng, T., and Sun, Y. <em>Rethinking the Stability-Plasticity Trade-off in Continual Learning from an Architectural Perspective</em>. ICML, 2025. <a href="https://arxiv.org/abs/2506.03951" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Abbas, Z., Zhao, R., Modayil, J., White, A., and Machado, M. C. <em>Loss of Plasticity in Continual Deep Reinforcement Learning</em>. arXiv:2303.07507, 2023. <a href="https://arxiv.org/abs/2303.07507" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood, A. R., and Sutton, R. S. <em>Loss of Plasticity in Deep Continual Learning</em>. Nature, 2024. <a href="https://www.nature.com/articles/s41586-024-07711-7" target="_blank" rel="noopener noreferrer">article</a></li>
  <li>Lyle, C., Zheng, Z., Khetarpal, K., van Hasselt, H., Pascanu, R., Martens, J., and Dabney, W. <em>Disentangling the Causes of Plasticity Loss in Neural Networks</em>. arXiv:2402.18762, 2024. <a href="https://arxiv.org/abs/2402.18762" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>He, N., Guo, K., Prakash, A., Tiwari, S., Tao, R. Y., Serapio, T., Greenwald, A., and Konidaris, G. <em>Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning</em>. arXiv:2509.22335, 2025. <a href="https://arxiv.org/abs/2509.22335" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Han, T., Bordt, S., Zhang, H., and Kakade, S. <em>Weight Decay Improves Language Model Plasticity</em>. arXiv:2602.11137, 2026. <a href="https://arxiv.org/abs/2602.11137" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Huang, J., Wurgaft, D., Bansal, R., Ruis, L., Saphra, N., Alvarez-Melis, D., Lampinen, A. K., Potts, C., and Lubana, E. S. <em>Why Larger Models Learn More: Effects of Capacity, Interference, and Rare-Task Retention</em>. arXiv:2605.29548, 2026. <a href="https://arxiv.org/abs/2605.29548" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Abel, D. et al. <em>Plasticity as the Mirror of Empowerment</em>. arXiv:2505.10361, 2025. <a href="https://arxiv.org/abs/2505.10361" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Hernandez-Garcia, J. F., Figliolia, T., and Millidge, B. <em>Can Scale Save Us From Plasticity Loss in Large Language Models?</em> arXiv:2606.24752, 2026. <a href="https://arxiv.org/abs/2606.24752" target="_blank" rel="noopener noreferrer">arXiv</a></li>
  <li>Zheng, G., Yang, E., Wang, X., Chen, Y., He, F., Zheng, Q., Wang, P., and Shen, L. <em>Plasticity Activation via Polar Operator: A Plug-in Method for Balancing Stability and Plasticity</em>. ICML, 2026. <a href="https://openreview.net/forum?id=b7P2WegaBY" target="_blank" rel="noopener noreferrer">OpenReview</a></li>
  <li>Koeppe, N., Vecchietti, L. F., Han, D., Li, D., and Lee, S. W. <em>Mitigating Plasticity Loss through Architectural Design in Continual Learning</em>. ICML, 2026. <a href="https://openreview.net/forum?id=pAhGjPOlwy" target="_blank" rel="noopener noreferrer">OpenReview</a></li>
</ol>

</div>
