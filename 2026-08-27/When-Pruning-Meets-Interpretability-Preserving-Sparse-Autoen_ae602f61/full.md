# When Pruning Meets Interpretability: Preserving Sparse Autoencoder Robustness in LLMs

Suchit Gupte, Xueru Zhang & Mohammad Mahdi Khalili

Department of Computer Science

The Ohio State University

Columbus, OH 43210, USA

{gupte.31, zhang.12807, khalili.17}@osu.edu

## Abstract

Sparse autoencoders (SAEs) are widely used to interpret the internal representations of large language models (LLMs), yet their reliability under post-hoc model compression remains poorly understood. We present a systematic study of how pruning affects SAE behavior and theoretically show that, for a fixed SAE, its impact is governed by perturbation energy, a covariance-weighted norm. This perspective exposes a key limitation of MAGNITUDE pruning: by ignoring activation geometry, it distorts the learned representation space and degrades SAE functionality. Activationaware methods such as WANDA and SPARSEGPT, in contrast, implicitly control perturbation energy and are therefore substantially more robust at preserving SAE behavior. We further reveal a consistent structural vulnerability across all pruning methods: middle layers are significantly more sensitive to pruning than early or late layers. Guided by this insight, we propose a Layer-wise sparsity allocation strategy, achieving lower perplexity under the same average pruning sparsity. Experiments across four model architectures validate our theoretical findings. The code is publicly available at sae-robustness-under-pruning.

## 1 Introduction

The superposition hypothesis (Elhage et al., 2022) posits that neural networks represent more features than they have neurons, giving rise to polysemanticity where individual neurons respond to multiple unrelated concepts. Sparse autoencoders (SAEs) (Cunningham et al., 2023) have emerged as the leading method for resolving this: by training an overcomplete sparse dictionary on a model’s activations, they recover monosemantic, interpretable features (Bricken et al., 2023; Zhu et al., 2026), an approach that scales to frontier models (Templeton et al., 2024; Lieberum et al., 2024). SAE-based analysis has consequently become a core workflow in mechanistic interpretability (Shu et al., 2025; Sharkey et al., 2025). SAEs are used to identify circuits (Conmy et al., 2023; Wang et al., 2023; Chanin et al., 2025), erase spurious correlations (Karvonen et al., 2024), and test causal hypotheses about model behavior. All of these applications rely on the implicit assumption that any SAE provides a faithful and reliable decomposition of the underlying activations to which it is applied.

Independently, weight pruning has become a standard post-hoc technique for reducing the computational cost of deploying large language models. Methods ranging from classical MAGNITUDE pruning (Han et al., 2016) to modern one-shot approaches such as SPARSEGPT (Frantar & Alistarh, 2023) and WANDA (Sun et al., 2024) can remove a substantial fraction of model weights with only modest degradation in downstream task performance. Given the prohibitive cost of deploying large dense models, compression techniques including pruning, quantization, and distillation are standard tools in the openweight deployment pipeline (Muralidharan et al., 2024; Ma et al., 2023; Guo et al., 2025; Cai et al., 2024; Zhu et al., 2025), often applied after interpretability analysis has been conducted on the dense checkpoint and any associated SAEs have been trained. Crucially, the criteria by which these methods select weights to prune: whether by magnitude alone, by activation statistics, or by second-order Hessian, differ substantially in how they perturb the model’s internal activation distribution, a distinction that has received little attention outside of predictive performance evaluations.

These two workflows are inherently in tension. An SAE is trained on the activation distribution of a specific dense model, whereas pruning alters that model’s weight matrices, perturbing all downstream activations. When these perturbations remain small relative to the SAE’s training distribution, the SAE may continue to function approximately as intended. Otherwise, its latents can fail silently (Chanin et al., 2025; Karvonen et al., 2024; Gupte et al., 2025), and importantly, such failures may not be detected by standard language modeling benchmarks (Egashira et al., 2026; Sharkey et al., 2025). Despite this, there is currently no theoretical framework that characterizes when pruning preserves SAE validity, nor an explanation for why different pruning methods lead to varying degrees of degradation.

A natural response to a changed model is to retrain the interpretability tooling itself. Crosscoders (Lindsey, 2024), for instance, train a joint dictionary across two model states (e.g., base and fine-tuned) to discover what changed between them. This is effective, but it requires training a new dictionary for each pair of model variants, precisely the retraining cost we aim to avoid. Our work instead asks: given an SAE already trained on a dense model, under what conditions does it remain valid after pruning, without retraining? This is the more common deployment scenario, since organizations that have invested resources in SAE training need to know whether existing analysis transfers to a compressed model rather than needing to characterize what changed. The two approaches are thus complementary: crosscoders are the right tool for understanding a specific change, whereas answering the transfer question requires a theoretical account of when pruning leaves an existing SAE intact, which is exactly what is missing. This paper provides that account.

We begin by theoretically examining the impact of MAGNITUDE, SPARSEGPT, and WANDA pruning, and show that SAE degradation is governed by perturbation energy $( \varepsilon ^ { 2 } )$ , a covariance-weighted norm that weights perturbations more heavily along high-variance input directions. This perspective establishes a formal separation among the three pruning methods: each method can be seen as minimizing a mathematically distinct approximation of $\varepsilon ^ { 2 }$ . MAGNITUDE pruning, which ignores activation geometry, induces the largest perturbation energy and the most severe SAE degradation, whereas activation-aware methods such as WANDA and SPARSEGPT implicitly control this quantity and are substantially more robust at preserving SAE behavior.

We validate our theory using four SAEBench (Karvonen et al., 2025) metric categories: Core (Karvonen et al., 2025), Feature Absorption (Chanin et al., 2025), SCR, and TPP (Karvonen et al., 2024), which together capture the full range of qualitative SAE failure modes. Beyond comparing methods, our experiments reveal a consistent structural vulnerability: middle layers are significantly more sensitive to pruning than early or late layers, across all methods and sparsity levels. Building on this insight, we explore a layer-wise sparsity allocation strategy that preserves SAE quality and achieves lower perplexity while maintaining the same average sparsity level.

The remainder of the paper is organized as follows. Sec. 2 presents background on sparse autoencoders and formulates pruning as activation perturbation. Sec. 3 develops the perturbation-theoretic framework and derives the formal separation among pruning methods. Sec. 4 argues for the sufficiency of the SAEBench evaluation suite in capturing all relevant SAE failure modes. Sec. 5 describes the experimental setup, and Sec. 6 presents the results. Sec 7 concludes the paper and discusses limitations and future directions.

## 2 Preliminaries

## 2.1 Sparse autoencoders for mechanistic interpretability

Definition 2.1 (Sparse Autoencoder). A sparse autoencoder (Cunningham et al., 2023) is a map $f _ { \theta } : \mathbb { R } ^ { d }  \bar { \mathbb { R } ^ { d } }$ with encoder $E _ { \theta } : \mathbb { R } ^ { d } \overset { \mathbf { \bar { \Delta } } } {  } \mathbb { R } ^ { m }$ and decoder $D _ { \theta } : \mathbb { R } ^ { m }  \mathbb { R } ^ { d } .$ , where m $\gg d \colon$

$$
f _ { \theta } ( x ) = D _ { \theta } \big ( \sigma \big ( E _ { \theta } ( x ) \big ) \big ) , \quad E _ { \theta } ( x ) = W _ { \mathrm { e n c } } x + b _ { \mathrm { e n c } } , \quad D _ { \theta } ( z ) = W _ { \mathrm { d e c } } z + b _ { \mathrm { d e c } } ,\tag{1}
$$

where σ is an arbitrary activation function, $W _ { \mathrm { e n c } } \in \mathbb { R } ^ { m \times d } , W _ { \mathrm { d e c } } \in \mathbb { R } ^ { d \times m }$

The SAE is trained on activations $x \sim \mathcal { D }$ from the model’s forward pass to minimize:

$$
\mathcal { L } _ { \mathrm { S A E } } = \mathbb { E } _ { x \sim \mathcal { D } } \Big [ \| x - f _ { \theta } ( x ) \| ^ { 2 } + \lambda \| \sigma \big ( E _ { \theta } ( x ) \big ) \| _ { 1 } \Big ] .\tag{2}
$$

A key property we assume is that trained SAEs are Lipschitz continuous.

Assumption 2.2 (Lipschitz continuity of the SAE). The trained SAE $f _ { \theta }$ is L-Lipschitz:

$$
\| f _ { \theta } ( x ^ { \prime } ) - f _ { \theta } ( x ) \| \leq L \| x ^ { \prime } - x \| \forall x , x ^ { \prime } \in \mathbb { R } ^ { d } .\tag{3}
$$

This assumption holds for standard SAE architectures whenever coordinate-wise activation function $\sigma$ is Lipschitz, as satisfied by common choices such as ReLU and GELU.

## 2.2 Weight pruning as activation perturbation

Consider a linear layer with weight matrix $W \in \mathbb { R } ^ { d \times p }$ producing activation $x = W x _ { \mathrm { i n } } ,$ where $x _ { \mathrm { i n } } \in \mathbb { R } ^ { p }$ is the layer input. Pruning modifies the weight matrix by zeroing a subset of entries $S \subset [ d ] \times [ p ]$ , resulting in $W ^ { \prime } = \breve { W } + \Delta W$ . The corresponding pruned activation is

$$
x ^ { \prime } = W ^ { \prime } x _ { \mathrm { i n } } = x + \delta , \qquad \delta : = \Delta W x _ { \mathrm { i n } } .\tag{4}
$$

We next summarize the objectives of common pruning methods.

Definition 2.3 (Pruning objectives). Let S be the pruning mask at target sparsity s.

1. MAGNITUDE pruning (Han et al., 2016) selects S to minimize $\begin{array} { r } { \| \Delta W \| _ { F } ^ { 2 } = \sum _ { ( i , j ) \in S } w _ { i j } ^ { 2 } . } \end{array}$

2. WANDA (Sun et al., 2024) selects S to minimize $\begin{array} { r } { \sum _ { ( i , j ) \in S } w _ { i j } ^ { 2 } \cdot \mathbb { E } \left\lceil ( x _ { \mathrm { i n } } ) _ { j } ^ { 2 } \right\rceil } \end{array}$

3. SPARSEGPT (Frantar & Alistarh, 2023) solves layer-wise regression to approximately minimize E $: [ \Vert \Delta W x _ { \mathrm { i n } }  ^ { 2 } ]$ via second-order optimization.

## 3 Perturbation Theory: SAE Degradation under Pruning

Next, we theoretically examine how pruning affects SAE behavior. Specifically, we develop a unified perturbation-theoretic analysis that bounds SAE degradation in terms of a datadependent quantity, which we call the perturbation energy, together with the SAE’s Lipschitz constant. This quantity provides a theoretical foundation for understanding the varying impact of different pruning techniques and explains why SAEs can be robust to some methods while vulnerable to others. Proof details are provided in Appendix A.1.

Impact of pruning on SAE. Recall from Sec. 2.2 that pruning replaces W with $W ^ { \prime } = W + \Delta W _ { . }$ shifting the activation by $\delta = \Delta W x _ { \mathrm { i n } } .$ Because the $\mathsf { \bar { S } A E } f _ { \theta }$ is L-Lipschitz (Assumption 2.2), the triangle inequality immediately gives a per-sample reconstruction bound.

Lemma 3.1. Let $f _ { \theta }$ be L-Lipschitz and let $x ^ { \prime } = x + \delta .$ . Then:

$$
\begin{array} { r } { \| { \boldsymbol x } ^ { \prime } - f _ { \boldsymbol \theta } ( { \boldsymbol x } ^ { \prime } ) \| \ \le \ \| { \boldsymbol x } - f _ { \boldsymbol \theta } ( { \boldsymbol x } ) \| + ( 1 + L ) \| \delta \| . } \end{array}\tag{5}
$$

Squaring, taking expectations, and applying Cauchy–Schwarz to the cross term yields Thm. 3.2, which bounds expected degradation in SAE reconstruction quality after pruning.

Theorem 3.2 (Expected reconstruction degradation). Under Assumption 2.2,

$$
\begin{array} { r l } { \mathbb { E } \big [ \| x ^ { \prime } - f _ { \theta } ( x ^ { \prime } ) \| ^ { 2 } \big ] } & { \le \underbrace { \mathbb { E } \big [ \| x - f _ { \theta } ( x ) \| ^ { 2 } \big ] } _ { i n t r i n s i c S A E e r r o r } + ( 1 + L ) ^ { 2 } \varepsilon ^ { 2 } + 2 ( 1 + L ) \sqrt { \mathbb { E } [ \| x - f _ { \theta } ( x ) \| ^ { 2 } ] \cdot \varepsilon ^ { 2 } } } \end{array}\tag{6}
$$

where the perturbation energy $\varepsilon ^ { 2 }$ admits the covariance decomposition

$$
\varepsilon ^ { 2 } : = \mathbb { E } \left[ \left\| \delta \right\| ^ { 2 } \right] = \mathbb { E } \left[ \left\| \Delta W x _ { \mathrm { i n } } \right\| ^ { 2 } \right] = \mathrm { t r } \big ( \Delta W \Sigma _ { x _ { \mathrm { i n } } } \Delta W ^ { \top } \big ) ,\tag{7}
$$

with $\Sigma _ { x _ { \mathrm { i n } } } = \mathbb { E } [ x _ { \mathrm { i n } } x _ { \mathrm { i n } } ^ { \top } ]$ the second-moment matrix of layer inputs.

Although Thm. 3.2 is an upper bound, it provides insight into what drives SAE reconstruction degradation after pruning. The bound decomposes into two components: (1) intrinsic SAE error which reflects how well the SAE was trained on the original model and is independent of pruning; (2) perturbation energy $\varepsilon ^ { 2 }$ which captures the direct damage caused by pruning. The third term captures their interaction, showing that a poorly trained SAE amplifies the effect of pruning: even small $\varepsilon ^ { 2 }$ can cause disproportionately large degradation when the intrinsic error is large.

The bound also depends on the Lipschitz constant L, a property of the SAE architecture and training that is independent of the pruning method: both the $( \dot { 1 } + L ) ^ { 2 }$ and cross terms grow with $L ,$ so a less smooth SAE amplifies any given perturbation energy. Since we hold SAE weights fixed throughout all experiments, L is constant across pruning methods, making $\varepsilon ^ { 2 }$ the only operative variable when comparing methods; all method-level statements in this paper should be read under this fixed-SAE assumption.

If we write the eigen decomposition $\begin{array} { r } { \Sigma _ { x _ { \mathrm { i n } } } = \sum _ { i } \lambda _ { i } v _ { i } v _ { i } ^ { \top } } \end{array}$ with $\lambda _ { 1 } \geq \cdot \cdot \cdot \geq \lambda _ { p } > 0 .$ , then

$$
\varepsilon ^ { 2 } = \sum _ { i } \lambda _ { i } \| \Delta W v _ { i } \| ^ { 2 } .\tag{8}
$$

This is a covariance-weighted squared norm of ∆W: each term $\lambda _ { i } \parallel \Delta W v _ { i } \parallel ^ { 2 }$ quantifies how strongly ∆W perturbs the i-th principal direction of the input distribution, scaled by the variance along that direction. Perturbations aligned with high-variance directions contribute most to $\varepsilon ^ { 2 } .$ , while those along low-variance directions have little effect. The impact of pruning on SAEs therefore depends not only on how much the weights change, but on how those changes align with the input covariance structure, and pruning methods differ precisely in how well they account for this geometry.

Separation among pruning methods. The covariance weighting in (8) induces a fundamental distinction between MAGNITUDE and activation-aware pruning methods.

Corollary 3.3 (Comparison of pruning methods). At any target sparsity s:

• MAGNITUDE pruning minimizes $\| \Delta W \| _ { F } ^ { 2 } = \mathrm { t r } ( \Delta W I \Delta W ^ { \top } )$ , which coincides with minimizing $\varepsilon ^ { 2 }$ only when $\Sigma _ { x _ { \mathrm { i n } } } \propto I .$

• WANDA minimizes tr(∆W dia $\mathsf { y } ( \mathbb { E } [ x _ { \mathrm { i n } } x _ { \mathrm { i n } } ^ { \top } ] ) \Delta W ^ { \top } ,$ ), a diagonal approximation to $\varepsilon ^ { 2 }$

• SPARSEGPT approximately minimizes $\varepsilon ^ { 2 }$ directly via second-order information.

Corollary 3.3 highlights that the three pruning methods differ fundamentally in how they approximate $\varepsilon ^ { 2 } \colon$ MAGNITUDE pruning, by ignoring the covariance structure of the activations, approximates $\varepsilon ^ { 2 }$ poorly and consequently induces larger perturbation energy and more severe SAE degradation. In contrast, WANDA and SPARSEGPT implicitly account for this structure, better controlling $\varepsilon ^ { 2 }$ and thus more effectively preserving SAE behavior.

This separation can be made precise: the three methods form a natural hierarchy in how faithfully they approximate the true objective $\varepsilon ^ { 2 }$

$$
\underbrace { \mathrm { t r } ( \Delta W I \Delta W ^ { \top } ) } _ { \mathrm { M A G N I T U D E } } \prec \underbrace { \mathrm { t r } ( \Delta W \mathrm { d i a g } ( \Sigma _ { x _ { \mathrm { i n } } } ) \Delta W ^ { \top } ) } _ { \mathrm { W A N D A } } \prec \underbrace { \mathrm { t r } ( \Delta W \Sigma _ { x _ { \mathrm { i n } } } \Delta W ^ { \top } ) } _ { \mathrm { S p a r s e G P T } } ,\tag{9}
$$

where ≺ denotes “is a coarser approximation of the true perturbation energy than ${ \bf \chi } ^ { \prime \prime } .$ . MAGNI-TUDE pruning ignores $\Sigma _ { x _ { \mathrm { i n } } }$ entirely; WANDA uses only its diagonal; SparseGPT uses the full matrix, mirroring exactly the ordering of SAE degradation we observe empirically in Sec 6.

## 4 From Theory to Evaluation: SAEBench as a Sufficient Validation Suite

Sec. 3 shows that, for a fixed SAE, $\varepsilon ^ { 2 }$ governs SAE degradation under pruning. Validating this empirically requires metrics that collectively cover every way an SAE can fail when its input distribution shifts. We argue that the four SAEBench categories, Core, Feature Absorption, SCR, and TPP, are jointly sufficient as they cover three orthogonal failure modes.

Omitting any one would leave a blind spot in the validation. We describe each failure mode and its corresponding metric in turn, then argue their joint sufficiency (see Appendix A.2 for metric definitions).

Failure Mode 1: Reconstruction Degrades. When pruning shifts the activation distribution, the SAE may no longer accurately reconstruct its inputs. Since the SAE’s reconstruction is inserted back into the model’s forward pass, degraded reconstruction directly corrupts the model’s predictions. The Core metrics (Karvonen et al., 2025) detect this failure by measuring how well the SAE’s reconstruction preserves model behavior, using the KL divergence score, CE loss score, explained variance, MSE, and cosine similarity (see Appendix A.2.1). Preserved Core metrics indicate that the SAE remains within its effective reconstruction region after pruning. However, Core metrics are aggregated over all latents and tokens, making them insensitive to localized failures: a score of 0.99 is consistent with 5% of latents behaving incorrectly, provided the rest compensate. The SAE can silently reorganize its internal representations while keeping aggregate scores high, leaving the failure modes below entirely undetected.

Failure Mode 2: Features Break. Even when aggregate reconstruction appears intact, individual latents may fail. A latent that should activate on a given input may go silent (false negative), or one that should be silent may fire spuriously (false positive). When this happens, the SAE no longer provides a faithful, interpretable decomposition of the activations, even though nothing appears wrong at the aggregate level. Feature Absorption (Chanin et al., 2025) detects this failure by testing whether the SAE maintains clean feature separation after pruning. Using a first-letter classification task, it identifies cases where an expected latent fails to activate and checks whether a semantically unrelated latent has absorbed its information. A lower absorption rate indicates better feature separation. Thus, the absorption metric goes beneath the aggregate surface and checks whether a feature that should represent “starts with ${ \mathrm { S } } ^ { \prime \prime }$ still fires on S-tokens, or whether its information has been absorbed by a semantically unrelated latent. Without this metric, we could observe perfect Core scores while the SAE has been silently rearranged into an uninterpretable state.

Failure Mode 3: Interventions Stop Working. The primary use of SAEs in interpretability is causal intervention: ablating specific latents to test whether they causally mediate a concept or behavior. Pruning can break this connection: a latent may still activate on the right inputs but no longer causally influence the model’s output as expected, or its ablation may collaterally affect unrelated concepts. Spurious Correlation Removal (SCR) (Karvonen et al., 2024) detects this failure by testing whether latent ablations still produce their intended causal effect after pruning: if ablating latents $S _ { c }$ removed a spurious “gender → profession” correlation in the dense model, SCR checks whether the same ablation achieves the same effect after pruning. Without SCR, we could observe latents that activate correctly yet have lost their causal influence: the latent fires on the right inputs, but ablating it no longer changes the model’s behavior. Targeted Probe Perturbation (TPP) (Karvonen et al., 2024) complements SCR by testing selectivity: ablating latents for concept c should degrade performance on class c but leave other classes unaffected. Without TPP, we could conclude that interventions remain effective while missing that they have become non-specific, damaging concepts they were never intended to affect.

Joint Sufficiency. Together, these metrics form a logical chain that mirrors the structure of our perturbation theory. Core verifies that $\varepsilon ^ { 2 }$ is small enough for aggregate reconstruction to be preserved (Thm. 3.2). Feature Absorption verifies that $\varepsilon ^ { 2 }$ is small enough relative to activation margins, ensuring individual latents remain intact. SCR verifies that surviving features retain their causal role, so that latent-level interventions still produce the intended effect. TPP verifies that these causal effects remain selective, that concepts stay disentangled, not merely detectable. This progression from aggregate fidelity, to feature-level correctness, to interventional validity, to interventional selectivity covers the full stack of properties that make SAEs useful as an interpretability tool. If all four are preserved under pruning, every standard SAE-based analysis transfers from the dense model to the pruned model.

## 5 Experimental Setup

Models and SAEs. To isolate the effect of pruning from SAE training variability, we hold all SAE weights fixed and use pretrained checkpoints from SAELens (Bloom et al., 2024). Our model selection is governed by the availability of high-quality pretrained SAEs. By fixing pretrained SAEs, we ensure that any observed degradation is attributable solely to pruning and not to SAE training choices. This consideration leads us to four model–SAE pairs spanning roughly two orders of magnitude in parameter count: pythia-70m (Biderman et al., 2023; Jedryszek & Crook, 2026), gemma-2-2b (Team et al., 2024; Lieberum et al., 2024), gemma-2-9b (Team et al., 2024; Lieberum et al., 2024), and mistral-7b (Jiang et al., 2023; Engels, 2024), all of which have publicly available, well-validated SAE checkpoints. Our main experiments focus on gemma-2-2b, which benefits from the most comprehensive SAE coverage among the four models. This model has pretrained SAEs available at every residual-stream layer (layers 0–25), enabling the fine-grained layer-level analysis that is central to our investigation. Results for all three remaining models are reported in Sec. 6 as a cross-model generalisation study, where we verify that the theoretical ordering and layer sensitivity patterns observed on gemma-2-2b hold across architectures and scales.

Pruning methods. We apply three pruning methods post-hoc to each dense checkpoint: MAGNITUDE pruning (Han et al., 2016), WANDA (Sun et al., 2024), and SPARSEGPT (Frantar & Alistarh, 2023) (see Sec. 2.2 for an overview). No retraining or weight updates are performed after pruning. Calibration data for WANDA and SPARSEGPT consists of 128 samples from OpenWebText (Gokaslan et al., 2019). We conduct experiments at 50% sparsity, a practically relevant operating point at which all three methods remain functional and differences in SAE degradation are clearly visible. To characterise how SAE degradation scales with the degree of compression, we additionally evaluate at 25% and 40% sparsity, allowing us to assess whether the layer sensitivity patterns and method ordering identified at 50% are consistent across sparsity levels or specific to that operating point.

Evaluation. For each pruned model variant, we compute the four SAEBench (Karvonen et al., 2025) metric categories described in Sec. 4: Core (KL divergence score, CE loss score, explained variance, cosine similarity, MSE), Feature Absorption, SCR, and TPP. Evaluations are run on the residual output of every layer (layers 0–25 for gemma-2-2b) to expose layerlevel sensitivity patterns. For SCR and TPP, we adopt k = 10 as the canonical ablation threshold, as it lies in the stable regime of the threshold sensitivity curve and produces the clearest separation between pruning methods (see Appendix A.2). All results are reported both as raw metric values and as percentage change from the dense baseline.

## 6 Results

Impact of pruning methods is consistent and theory-aligned. Table 1 and Figure 1 together reveal a clear and consistent ordering across all SAEBench metric categories: MAGNITUDE pruning causes the most severe SAE degradation, WANDA is substantially more robust, and SPARSEGPT is equally conservative, closely tracking the dense baseline. This ordering holds without exception across all eight metrics and all 26 layers of gemma-2-2b at 50% sparsity.

The pattern precisely validates our perturbation-theoretic analysis: MAGNITUDE pruning minimises the wrong objective: the Frobenius norm, which equals perturbation energy only under an identity covariance. While WANDA and SPARSEGPT progressively better approximate the true perturbation energy ε<sup>2</sup>. The separation is particularly stark on interventional metrics: MAGNITUDE pruning degrades SCR by up to 60% and TPP by up to 80% relative to the dense baseline in the worst-affected layers, whereas WANDA and SPARSEGPT remain within 15–25% of baseline on the same metrics (Refer to Figure 1).

Critically, the Core metrics, which aggregate over all latents, substantially understate the damage inflicted by MAGNITUDE pruning. For instance, KL divergence scores for MAG-NITUDE pruning in middle layers appear superficially reasonable (≥0.96), yet the corresponding SCR and TPP scores collapse, confirming our theoretical prediction that aggregate reconstruction fidelity is insufficient to certify SAE reliability. Appendix A.4 closes the loop directly: measured $\varepsilon ^ { 2 }$ values confirm the ordering of Corollary 3.3 at every layer group, and reveal that middle layers have the lowest local $\varepsilon ^ { 2 }$ despite the highest degradation, implicating accumulated upstream perturbation.

<table><tr><td rowspan="2"></td><td rowspan="2">Layers Method</td><td rowspan="2">Spar. (%)</td><td colspan="5">Core</td><td colspan="2">Absorption</td><td>SCR</td><td>TPP</td></tr><tr><td>KL Score↑</td><td>CE Score↑</td><td>Expl. Var.↑</td><td>Cos Sim↑</td><td>MSE ↓</td><td>Abs. Frac.↓</td><td>Full Abs.↓</td><td>Top-10 ↑</td><td>Top-10 ↑</td></tr><tr><td rowspan="4">Eary (8-0)</td><td>Baseline</td><td>0</td><td>0.9944</td><td>0.9941</td><td>0.8976</td><td>0.9497</td><td>0.4268</td><td>0.0674</td><td>0.0354</td><td>0.1714</td><td>0.0270</td></tr><tr><td>MAGNITUDE</td><td>50</td><td>0.9847</td><td>0.9818</td><td>0.8416</td><td>0.9175</td><td>0.6536</td><td>0.0764</td><td>0.0486</td><td>0.1130</td><td>0.0187</td></tr><tr><td>WANDA</td><td>50</td><td>0.9935</td><td>0.9873</td><td>0.8793</td><td>0.9401</td><td>0.4648</td><td>0.0691</td><td>0.0353</td><td>0.1487</td><td>0.0188</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9918</td><td>0.9906</td><td>0.8728</td><td>0.9353</td><td>0.4886</td><td>0.0717</td><td>0.0402</td><td>0.1329</td><td>0.0196</td></tr><tr><td rowspan="5">Mdde )(-1)</td><td>Baseline</td><td>0</td><td>0.9887</td><td>0.9876</td><td>0.8763</td><td>0.9301</td><td>2.0582</td><td>0.0668</td><td>0.0586</td><td>0.3047</td><td>0.0424</td></tr><tr><td>MAGNITUDE</td><td>50</td><td>0.9644</td><td>0.9531</td><td>0.7782</td><td>0.8780</td><td>3.3021</td><td>0.0794</td><td>0.0730</td><td>0.1301</td><td>0.0089</td></tr><tr><td>WANDA</td><td>50</td><td>0.9894</td><td>0.9854</td><td>0.8507</td><td>0.9175</td><td>2.2214</td><td>0.0660</td><td>0.0564</td><td>0.2399</td><td>0.0304</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9868</td><td>0.9893</td><td>0.8277</td><td>0.9071</td><td>2.5616</td><td>0.0731</td><td>0.0662</td><td>0.2103</td><td>0.0258</td></tr><tr><td>Baseline</td><td>0</td><td>0.9776</td><td>0.9765</td><td>0.8682</td><td>0.9307</td><td>14.2891</td><td>0.1208</td><td>0.1369</td><td>0.3885</td><td>0.0681</td></tr><tr><td rowspan="4">(18-25) Late</td><td>MAGNITUDE</td><td></td><td></td><td>0.9510</td><td>0.7544</td><td>0.8701</td><td>29.4766</td><td>0.0831</td><td>0.1177</td><td>0.3192</td><td></td></tr><tr><td>WANDA</td><td>50 50</td><td>0.9535 0.9780</td><td>0.9773</td><td>0.8569</td><td>0.9253</td><td>14.7422</td><td>0.1045</td><td>0.1179</td><td>0.3743</td><td>0.0117 0.0519</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9768</td><td>0.9815</td><td>0.8389</td><td>0.9150</td><td>17.6016</td><td>0.1054</td><td>0.1293</td><td>0.3605</td><td>0.0547</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1: SAEBench metrics at 50% sparsity for gemma-2-2b, aggregated by layer group (Early: layers 0–8; Middle: 9–17; Late: 18–25). Each group reports the unpruned dense baseline (italicised) alongside all three pruning methods. ↑ higher is better; ↓ lower is better. Refer to App. A.3 for per-layer SAEBench metrics from which these values are aggregated.

![](images/6e05cb66b0df1eb6f04ad5cc14b7f7a8c95e2ace0efa237930e392c4d143653e.jpg)  
Figure 1: Per-layer percentage degradation from the dense baseline across all eight SAEBench metric categories for gemma-2-2b at 50% sparsity. Each panel corresponds to one metric; MAGNITUDE (red), SPARSEGPT (blue), and WANDA (green) across layers 0–25. These plots emphasize per-layer trends and direct cross-method comparison at each layer; see Figure 2 for the same data as a heatmap.

Middle layers exhibit systematic structural vulnerability. A striking and consistent finding across all methods and metrics is the elevated sensitivity of middle layers (9–17) to pruning, relative to both early layers (0–8) and late layers (18–25). Figure 3 makes this explicit: averaging the percentage degradation across all methods and metrics, the middle-layer profile rises sharply above that of early and late layers at every sparsity level tested (25%, 40%, 50%), and this pattern is monotone in sparsity, confirming it is not an artifact of a particular operating point. At 50% sparsity, the heatmap in Figure 2 shows that middle layers carry the darkest cells across nearly every metric–method combination, with MAGNITUDE pruning on middle layers producing degradations of 10–14% in explained variance, 5–7% in cosine similarity, and 40–60% in SCR and TPP, substantially exceeding the corresponding values for early and late layers.

Layer Sensitivity Profile: Mean % change averaged across all methods and metrics  
% Change Heatmap: Methods × Layers (sparsity=50%)  
![](images/c240669c19393ffc2a9722131e02273ebc9a6ec02abe00c4f633bf0ad629ad55.jpg)

![](images/5c1b25eeb712ffed2e3376893f9d3135b38ffeda0f80b9c056b44e702b671383.jpg)

![](images/7b8638242d01116cb7d04e7c342c4ea41ac3ca89d90b6b5c6a49893298c691c1.jpg)

![](images/a9d8866f980aa74cdca0ac6c23d3f118330304c8c0a6fd55961b990abbabff68.jpg)

![](images/873524e69d83b867dc44f85034d6aa6e9f733bdf4dc3d793f7bf3110917d6cd3.jpg)

![](images/0dd9095fc4932c7c6f9ad0f81330c01d5f0755d9a00ae7071450b56baaa0e331.jpg)

![](images/fe3c07d525e12b881e0f88f80c6185d63ae980498b05e5e8f10ce74c6071fb12.jpg)

![](images/bf52a70e8775d9f6e969ffebfadefff743ca983bceec40b133f3018ae1743f35.jpg)  
Figure 2: Heatmap of per-layer, per-method percentage degradation from the dense baseline for gemma-2-2b at 50% sparsity. Each row corresponds to a pruning method; each column to a layer; colour intensity encodes the magnitude of degradation (darker = larger change). This heatmap presents the same underlying data as Figure 1, reorganized to expose spatial patterns across the full method × layer grid simultaneously.

![](images/86b7d53f9072d5dd1f9a1e3aad97527368350c6d7666a4ece5c98f8f956c9772.jpg)

![](images/8fc58159c97ba5283b2a01bc18c923db5533a183c690fce0d9b46317cacabad4.jpg)  
Figure 3: Layer sensitivity profile for gemma-2-2b, computed as the mean absolute percentage degradation averaged across all three pruning methods and all eight SAEBench metrics. (a) Sensitivity profiles at three sparsity levels (25%, 40%, 50%). (b) Sensitivity profile at 50% sparsity, coloured by layer group (Early: 0–8, Middle: 9–17, Late: 18–25).

We hypothesize this reflects the asymmetric propagation of representational damage across depth. Early layers construct foundational features that are heavily inherited by middle layers via residual connections, making middle layers the primary recipients of perturbation energy $\varepsilon ^ { 2 }$ introduced by pruning upstream weights. Late layers, by contrast, operate on an already fully composed representation and specialize in output-facing transformations, leaving them with little residual upstream dependency to corrupt. This asymmetry directly motivates a layer-wise sparsity schedule.

<table><tr><td rowspan="2">Model Method</td><td rowspan="2"></td><td rowspan="2">Spar. (%)</td><td colspan="5">Core</td><td rowspan="2"></td><td colspan="2">Absorption</td><td>SCR</td><td>TPP</td></tr><tr><td>KL Score↑</td><td>CE Score↑</td><td>Expl. Var.↑</td><td>Cos Sim↑</td><td>MSE ↓</td><td>Abs. Frac.↓</td><td>Full Abs.↓</td><td>Top-10 ↑</td><td>Top-10 ↑</td></tr><tr><td rowspan="5">&amp;emm2 2b</td><td>Baseline</td><td>0</td><td>0.9873</td><td>0.9864</td><td>0.8812</td><td>0.9370</td><td>5.2568</td><td>0.0836</td><td></td><td>0.0746</td><td>0.2843</td><td>0.0450</td></tr><tr><td>MAGNITUDE</td><td>50</td><td>0.9681</td><td>0.9624</td><td>0.7928</td><td>0.8893</td><td>10.4390</td><td>0.0795</td><td>0.0783</td><td></td><td>0.1824</td><td>0.0131</td></tr><tr><td>WANDA</td><td>50</td><td>0.9873</td><td>0.9835</td><td>0.8625</td><td>0.9277</td><td>5.4659</td><td>0.0789</td><td>0.0680</td><td></td><td>0.2497</td><td>0.0330</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9855</td><td>0.9874</td><td>0.8468</td><td>0.9193</td><td>6.4717</td><td>0.0826</td><td></td><td>0.0766</td><td>0.2297</td><td>0.0325</td></tr><tr><td>Baseline</td><td></td><td>00.9914</td><td>0.9900</td><td></td><td>0.8882</td><td>0.9380</td><td>7.7362</td><td>0.0997</td><td>0.0654</td><td>0.3237</td><td>0.0603</td></tr><tr><td rowspan="4">Semmm2 96</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0619</td><td></td><td></td></tr><tr><td>MAGNITUDE WANDA</td><td>50 50</td><td>0.9826 0.9911</td><td>0.9757 0.9869</td><td>0.8506 0.8772</td><td>0.9173 0.9322</td><td>12.4004 7.7391</td><td>0.0919 0.0905</td><td></td><td>0.0608</td><td>0.2307 0.2921</td><td>0.0298 0.0409</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9901</td><td>0.9892</td><td>0.8627</td><td>0.9245</td><td>9.6890</td><td>0.0910</td><td></td><td>0.0606</td><td>0.2706</td><td>0.0395</td></tr><tr><td>Baseline</td><td>0</td><td>0.9262</td><td>0.9240</td><td>0.9476</td><td>0.9572</td><td>0.0118</td><td>N/A⁺</td><td></td><td>N/A+</td><td></td><td>0.0505</td></tr><tr><td rowspan="4">pyhia 7m</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0726</td><td></td></tr><tr><td>MAGNITUDE</td><td>50</td><td>0.6422</td><td>0.8740</td><td>0.5592</td><td>0.8765 0.9244</td><td>0.0615 0.0195</td><td>N/A+ N/A⁺</td><td></td><td>N/A+ N/A+</td><td>0.0054</td><td>0.0420</td></tr><tr><td>WANDA SPARSEGPT</td><td>50 50</td><td>0.8798 0.8819</td><td>0.9091 0.9198</td><td>0.8705 0.9025</td><td>0.9328</td><td>0.0204</td><td>N/A+</td><td></td><td>N/A+</td><td>0.0150 0.0321</td><td>0.0405</td></tr><tr><td></td><td></td><td></td><td>0.9753</td><td>0.9844</td><td>0.8490</td><td>0.0150</td><td></td><td></td><td></td><td></td><td>0.0409</td></tr><tr><td rowspan="4">mranl 7b</td><td>Baseline</td><td>0</td><td>0.9722</td><td></td><td></td><td></td><td></td><td></td><td>0.4292</td><td>0.4150</td><td>0.3627</td><td>0.0501</td></tr><tr><td>MAGNITUDE</td><td>50</td><td>0.9672</td><td>0.9758</td><td>0.9792</td><td>0.8034</td><td>0.0177</td><td></td><td>0.4510</td><td>0.4371</td><td>0.3634</td><td>0.0208</td></tr><tr><td>WANDA</td><td>50</td><td>0.9742</td><td>0.9720</td><td>0.9857</td><td>0.8477</td><td>0.0128 0.0131</td><td></td><td>0.4023 0.3848</td><td>0.3805 0.3715</td><td>0.3810</td><td>0.0463</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9749</td><td>0.9734</td><td>0.9870</td><td>0.8477</td><td></td><td></td><td></td><td></td><td>0.3617</td><td>0.0396</td></tr></table>

<sup>†</sup> Feature absorption evaluation requires sufficient first-letter features to be detectable in the model. SAEBench (Karvonen et al., 2025) issues a warning that this evaluation is only reliable for models ≥2B parameters; pythia-70m (Biderman et al., 2023) (70M parameters) does not produce enough first-letter features for a valid absorption measurement (Chanin et al., 2025; Jedryszek & Crook, 2026), so these cells are excluded from all absorption analyses and averages.  
<sup>‡</sup> Pretrained SAEs for mistral-7b are available for only one layer per depth group (early, middle, late) in SAELens (Bloom et al., 2024; Engels, 2024); mistral-7b values are therefore averaged over 3 layers. The method ordering is consistent with the full-coverage models, but cross-model magnitude comparisons involving mistral-7b should be interpreted with this coverage difference in mind.

Table 2: Cross-architecture generalisation at 50% sparsity. Each cell reports the metric value averaged across all evaluated layers of a given model. ↑ higher is better; ↓ lower is better.

Layer-wise sparsity allocation. The layer sensitivity profile identified earlier suggests that a uniform sparsity schedule is suboptimal. Motivated by this hypothesis, we experiment with redistributing the global compression budget unevenly across depth: assigning lower sparsity to early layers to preserve the activations that middle layers depend on, and higher sparsity to late layers where upstream perturbation has a negligible downstream consequence. Concretely, given a target global sparsity s¯, we assign layer-wise sparsity s<sub>ℓ</sub> according to a schedule that ramps linearly from $s _ { \mathrm { m i n } }$ at the first layer to a plateau $s _ { \mathrm { m i d } }$ at the middle-layer boundary, and holds at $s _ { \mathrm { m i d } }$ through the remaining layers.

As a preliminary evaluation, we compare the perplexity of the layer-wise schedule against that of uniform sparsity at equivalent average sparsity on gemma-2-2b, using both WANDA and SPARSEGPT. Perplexity serves as a coarse measure to evaluate the model as a whole. As seen in Table 3, the layer-wise schedule yields lower perplexity in both cases, suggesting that the layer sensitivity profile is a practically useful signal for compression scheduling. We view this as an encouraging preliminary result rather than a fully developed method.

Non-uniform layerwise sparsity has previously been explored for general LLM compression. OWL (Yin et al., 2025) allocates sparsity in proportion to each layer’s activation-outlier ratio, validated against perplexity and downstream task accuracy. Our layer-wise schedule is motivated by a different, interpretability-specific diagnostic. The two criteria are not guaranteed to coincide: outlier concentration and SAE-degradation sensitivity are distinct properties of a layer’s activation distribution, though a systematic comparison is orthogonal to this paper’s central contribution, and we leave it to future work.

Cross-model generalisation confirms the theoretical ordering. Table 2 reports results aggregated over all layers for four model architectures spanning roughly two orders of magnitude in parameter count: pythia-70m, gemma-2-2b, gemma-2-9b, and mistral-7b. Across all models for which valid SAE evaluations are available, the method ordering SPARSEGPT ≻ WANDA ≻ MAGNITUDE is preserved on Core metrics, with WANDA and

<table><tr><td></td><td></td><td colspan="2">Wanda</td><td colspan="2">SparseGPT</td></tr><tr><td>Model</td><td>Avg. Sparsity Uniform Layer-wise Uniform Layer-wise</td><td></td><td></td><td></td><td></td></tr><tr><td>gemma-2-2b</td><td>50%</td><td>212</td><td>148</td><td>141</td><td>98</td></tr></table>

Table 3: Perplexity on gemma-2-2b under uniform versus layer-wise sparsity allocation evaluated for WANDA and SPARSEGPT. The ramp schedule assigns 20% sparsity to early layers, increasing linearly to 60% at the middle-layer and holding at 60% until the last layer.

SPARSEGPT achieving near-baseline KL and CE scores while MAGNITUDE pruning causes marked degradation. The result is especially pronounced for pythia-70m, where MAGNI-TUDE pruning reduces the KL divergence score from 0.926 to 0.642, a collapse not observed under WANDA (0.880) or SPARSEGPT (0.882). Mistral-7b also displays comparatively mild absolute degradation across all three methods on Core metrics.

## 7 Conclusion

This paper studies the interaction between weight pruning and mechanistic interpretability via sparse autoencoders (SAE), showing that, for a fixed SAE, pruning-induced SAE degradation is governed by perturbation energy, a covariance-weighted norm that measures how strongly pruning distorts the activation distribution, modulated by the SAE’s Lipschitz constant. This framework explains the consistent empirical ordering of pruning methods: MAGNITUDE pruning, which ignores activation geometry, induces large perturbation energy and significantly degrades SAE behavior, while activation-aware methods such as WANDA and SPARSEGPT better preserve interpretability by controlling perturbation energy. We validate this theory through a comprehensive evaluation across gemma-2-2b, gemma-2-9b, mistral-7b, and pythia-70m, using pretrained SAEs and the SAEBench evaluation suite. Additionally, we identify a structural vulnerability in middle layers and show that layerwise sparsity allocation, protecting these layers better, preserves representations under equivalent compression.

Limitations and Future Work. Our study has several limitations that point toward promising directions for future work. First, we hold SAE weights fixed after pruning. An important complementary direction is to retrain SAEs from scratch on the pruned model’s activations and examine whether the recovered features remain consistent with those of the dense model. Second, our conclusions rest on four model–SAE pairs drawn from a relatively homogeneous set of architectures. It remains unclear whether the results generalize to other model families, instruction-tuned models, mixture-of-experts architectures, or SAEs trained on MLP and attention-head outputs rather than the residual stream. Third, our analysis is confined to unstructured pruning; developing analogous theory for structured pruning, quantization, and knowledge distillation is important as these methods grow more prevalent in production deployments. Fourth, our layer-wise sparsity allocation is a preliminary illustration that the sensitivity profile is a useful scheduling signal, not a proposed allocation method in its own right: we do not benchmark it against more non-intuitive methods or dedicated layerwise-allocation methods such as OWL Yin et al. (2025), since a rigorous comparison is a separate undertaking from the SAE-degradation analysis that is this paper’s focus. Finally, while the middle-layer sensitivity finding is empirically robust, our framework does not yet explain it theoretically. A formal account of how the structure of internal representations evolves with depth, and how this motivates principled rather than heuristic sparsity allocation, remains an open problem.

## Acknowledgments

This work was funded in part by the National Science Foundation under award number IIS2202699, IIS-2416895, IIS-2301599, CMMI2301601, DMS-2529302. We used large language models (Claude by Anthropic, ChatGPT by OpenAI) to assist with writing and paraphrasing portions of this paper. No LLMs were used to originate research ideas, generate data or plots, or perform evaluation.

## References

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.

Joseph Bloom, Curt Tigges, Anthony Duong, and David Chanin. Saelens. https://github. com/decoderesearch/SAELens, 2024.

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nicholas L. Turner, Cem Anil, Carson Denison, Amanda Askell, et al. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. URL https://transformer-circuits.pub/2023/monosemantic-features.

Zhongteng Cai, Xueru Zhang, and Mohammad Mahdi Khalili. Privacy-aware randomized quantization via linear programming. In The 40th Conference on Uncertainty in Artificial Intelligence, 2024. URL https://openreview.net/forum?id=vWsf4L7rHq.

David Chanin, James Wilken-Smith, Toma´s Dulka, Hardik Bhatnagar, Satvik Golechha, andˇ Joseph Bloom. A is for absorption: Studying feature splitting and absorption in sparse autoencoders, 2025. URL https://arxiv.org/abs/2409.14507.

Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adria\` Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. In Advances in Neural Information Processing Systems, volume 36, pp. 16318–16352, 2023.

Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models, 2023. URL https: //arxiv.org/abs/2309.08600.

Kazuki Egashira, Robin Staab, Thibaud Gloaguen, Mark Vero, and Martin Vechev. Fewer weights, more problems: A practical attack on llm pruning, 2026. URL https://arxiv. org/abs/2510.07985.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition, 2022. URL https://arxiv.org/abs/2209.10652.

Josh Engels. Mistral-7b residual stream sparse autoencoders. https://huggingface.co/ JoshEngels/Mistral-7B-Residual-Stream-SAEs, 2024. Hugging Face repository.

Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot, 2023. URL https://arxiv.org/abs/2301.00774.

Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. Openwebtext corpus. http://Skylion007.github.io/OpenWebTextCorpus, 2019.

Jialong Guo, Xinghao Chen, Yehui Tang, and Yunhe Wang. Slimllm: Accurate structured pruning for large language models. 2025. URL https://arxiv.org/abs/2505.22689.

Suchit Gupte, Vishnu Kabir Chhabra, and Mohammad Mahdi Khalili. On the transferability of sparse autoencoders for interpreting compressed models, 2025. URL https://arxiv. org/abs/2507.15977.

Song Han, Huizi Mao, and William J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding, 2016. URL https: //arxiv.org/abs/1510.00149.

Piotr Jedryszek and Oliver M. Crook. Stable and steerable sparse autoencoders with weight regularization, 2026. URL https://arxiv.org/abs/2603.04198.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut´ Lavril, Thomas Wang, Timothee Lacroix, and William El Sayed. Mistral 7b, 2023. URL´ https://arxiv.org/abs/2310.06825.

Adam Karvonen, Can Rager, Samuel Marks, and Neel Nanda. Evaluating sparse autoencoders on targeted concept erasure tasks, 2024. URL https://arxiv.org/abs/2411.18895.

Adam Karvonen, Can Rager, Johnny Lin, Curt Tigges, Joseph Bloom, David Chanin, Yeu-Tong Lau, Eoin Farrell, Callum McDougall, Kola Ayonrinde, Demian Till, Matthew Wearden, Arthur Conmy, Samuel Marks, and Neel Nanda. Saebench: A comprehensive benchmark for sparse autoencoders in language model interpretability, 2025. URL https: //arxiv.org/abs/2503.09532.

Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, Janos Kram´ ar, Anca Dragan, Rohin Shah, and Neel Nanda. Gemma´ scope: Open sparse autoencoders everywhere all at once on gemma 2, 2024. URL https: //arxiv.org/abs/2408.05147.

Jack Lindsey. Sparse crosscoders for cross-layer features and model diffing. Transformer Cir cuits Thread, 2024. URL https://transformer-circuits.pub/2024/crosscoders/index. html.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models, 2023. URL https://arxiv.org/abs/2305.11627.

Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, and Pavlo Molchanov. Compact language models via pruning and knowledge distillation, 2024. URL https://arxiv.org/abs/2407.14679.

Lee Sharkey, Bilal Chughtai, Joshua Batson, Jack Lindsey, Jeff Wu, Lucius Bushnaq, Nicholas Goldowsky-Dill, Stefan Heimersheim, Alejandro Ortega, Joseph Bloom, Stella Biderman, Adria Garriga-Alonso, Arthur Conmy, Neel Nanda, Jessica Rumbelow, Martin Wattenberg, Nandi Schoots, Joseph Miller, Eric J. Michaud, Stephen Casper, Max Tegmark, William Saunders, David Bau, Eric Todd, Atticus Geiger, Mor Geva, Jesse Hoogland, Daniel Murfet, and Tom McGrath. Open problems in mechanistic interpretability, 2025.

Dong Shu, Xuansheng Wu, Haiyan Zhao, Daking Rai, Ziyu Yao, Ninghao Liu, and Mengnan Du. A survey on sparse autoencoders: Interpreting the internal mechanisms of large language models. In Findings of EMNLP, 2025.

Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. A simple and effective pruning approach for large language models, 2024. URL https://arxiv.org/abs/2306.11695.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Leonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre´ Rame, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos,´ Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Piotr Stanczyk, Sertan Girgin, Nikola Momchev, Matt Hoffman, Shantanu Thakoor, Jean-Bastien Grill, Behnam Neyshabur, Olivier Bachem, Alanna Walton, Aliaksei Severyn, Alicia Parrish, Aliya Ahmad, Allen Hutchison, Alvin Abdagic, Amanda Carl, Amy Shen, Andy Brock, Andy Coenen, Anthony Laforge, Antonia Paterson, Ben Bastian, Bilal Piot, Bo Wu, Brandon Royal, Charlie Chen, Chintu Kumar, Chris Perry, Chris Welty, Christopher A. Choquette-Choo, Danila Sinopalnikov, David Weinberger, Dimple Vijaykumar, Dominika Rogozinska, Dustin Herbison, Elisa Bandy, Emma Wang, Eric Noland, Erica ´ Moreira, Evan Senter, Evgenii Eltyshev, Francesco Visin, Gabriel Rasskin, Gary Wei, Glenn Cameron, Gus Martins, Hadi Hashemi, Hanna Klimczak-Plucinska, Harleen Batra, Harsh´ Dhand, Ivan Nardini, Jacinda Mein, Jack Zhou, James Svensson, Jeff Stanway, Jetha Chan, Jin Peng Zhou, Joana Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fernandez, Joost van Amersfoort, Josh Gordon, Josh Lipschultz, Josh Newlan, Ju yeong Ji, Kareem Mohamed,

Kartikeya Badola, Kat Black, Katie Millican, Keelin McDonell, Kelvin Nguyen, Kiranbir Sodhia, Kish Greene, Lars Lowe Sjoesund, Lauren Usui, Laurent Sifre, Lena Heuermann, Leticia Lago, Lilly McNealus, Livio Baldini Soares, Logan Kilpatrick, Lucas Dixon, Luciano Martins, Machel Reid, Manvinder Singh, Mark Iverson, Martin Gorner, Mat Velloso,¨ Mateo Wirth, Matt Davidow, Matt Miller, Matthew Rahtz, Matthew Watson, Meg Risdal, Mehran Kazemi, Michael Moynihan, Ming Zhang, Minsuk Kahng, Minwoo Park, Mofi Rahman, Mohit Khatwani, Natalie Dao, Nenshad Bardoliwalla, Nesh Devanathan, Neta Dumai, Nilay Chauhan, Oscar Wahltinez, Pankil Botarda, Parker Barnes, Paul Barham, Paul Michel, Pengchong Jin, Petko Georgiev, Phil Culliton, Pradeep Kuppala, Ramona Comanescu, Ramona Merhej, Reena Jana, Reza Ardeshir Rokni, Rishabh Agarwal, Ryan Mullins, Samaneh Saadat, Sara Mc Carthy, Sarah Cogan, Sarah Perrin, Sebastien M. R.´ Arnold, Sebastian Krause, Shengyang Dai, Shruti Garg, Shruti Sheth, Sue Ronstrom, Susan Chan, Timothy Jordan, Ting Yu, Tom Eccles, Tom Hennigan, Tomas Kocisky, Tulsee Doshi, Vihan Jain, Vikas Yadav, Vilobh Meshram, Vishal Dharmadhikari, Warren Barkley, Wei Wei, Wenming Ye, Woohyun Han, Woosuk Kwon, Xiang Xu, Zhe Shen, Zhitao Gong, Zichuan Wei, Victor Cotruta, Phoebe Kirk, Anand Rao, Minh Giang, Ludovic Peran, Tris Warkentin, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, D. Sculley, Jeanine Banks, Anca Dragan, Slav Petrov, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Sebastian Borgeaud, Noah Fiedel, Armand Joulin, Kathleen Kenealy, Robert Dadashi, and Alek Andreev. Gemma 2: Improving open language models at a practical size, 2024. URL https://arxiv.org/abs/2408.00118.

Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, et al. Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. Transformer Circuits Thread, 2024. URL https://transformer-circuits.pub/2024/scaling-monosemanticity.

Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In International Conference on Learning Representations, 2023.

Lu Yin, You Wu, Zhenyu Zhang, Cheng-Yu Hsieh, Yaqing Wang, Yiling Jia, Gen Li, Ajay Jaiswal, Mykola Pechenizkiy, Yi Liang, Michael Bendersky, Zhangyang Wang, and Shiwei Liu. Outlier weighed layerwise sparsity (owl): A missing secret sauce for pruning llms to high sparsity, 2025. URL https://arxiv.org/abs/2310.05175.

Ding Zhu, Zhiqun Zuo, and Mohammad Mahdi Khalili. An efficient training algorithm for models with block-wise sparsity. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=nay3Kvw8BD.

Xudong Zhu, Mohammad Mahdi Khalili, and Zhihui Zhu. Abstopk: Rethinking sparse autoencoders for bidirectional features. In International Conference on Learning Representations, volume 2026, pp. 50325–50352, 2026.

## A Appendix

## A.1 Detailed Proofs for the Perturbation-Theoretic Framework

This appendix provides fully detailed, self-contained proofs of the perturbation theory presented in Section 3.

## A.1.1 Setup and Notation

We recall the objects involved.

$W \in \mathbb { R } ^ { d \times p } ;$ the weight matrix of a single linear layer.

$x _ { \mathrm { i n } } \in \mathbb { R } ^ { p } ;$ : the random input to that layer, drawn from the data distribution D.

$x = W x _ { \mathrm { i n } } \in \mathbb { R } ^ { d } \colon$ the original (pre-pruning) activation.

$\Delta W \in \mathbb { R } ^ { d \times p }$ : the weight change induced by pruning. Because pruning zeroes entries, every non-zero entry $( i , j )$ of ∆W satisfies $\hat { ( \Delta W ) } _ { i j } = - W _ { i j }$

$x ^ { \prime } = ( W + \Delta W ) x _ { \mathrm { i n } } = x + \delta { \mathrm { : } }$ the pruned activation, where $\delta : = \Delta W x _ { \mathrm { i n } }$ is the perturbation vector.

$f _ { \theta } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d } \mathrm { : }$ : the trained SAE, which is L-Lipschitz (Assumption 2.2).

$\Sigma _ { x _ { \mathrm { i n } } } : = \mathbb { E } [ x _ { \mathrm { i n } } x _ { \mathrm { i n } } ^ { \top } ] \in \mathbb { R } ^ { p \times p }$ : the second-moment (input covariance) matrix.

We want to bound the expected squared reconstruction error on pruned activations:

$$
\mathcal { E } _ { \mathrm { p r u n e d } } : = \mathbb { E } \left[ \| x ^ { \prime } - f _ { \theta } ( x ^ { \prime } ) \| ^ { 2 } \right] .\tag{10}
$$

We will show that this is bounded in terms of the perturbation energy

$$
\begin{array} { r } { \varepsilon ^ { 2 } : = \mathbb { E } \left[ \| \delta \| ^ { 2 } \right] , } \end{array}\tag{11}
$$

and then show that $\varepsilon ^ { 2 } = \mathrm { t r } ( \Delta W \Sigma _ { x _ { \mathrm { i n } } } \Delta W ^ { \top } )$ .

## A.1.2 Proof of Lemma 3.1

Lemma (Lemma 3.1, restated). Let $f _ { \theta }$ be L-Lipschitz and let $x ^ { \prime } = x + \delta .$ Then:

$$
\begin{array} { r } { \| { \boldsymbol x } ^ { \prime } - f _ { \theta } ( { \boldsymbol x } ^ { \prime } ) \| \ \le \ \| { \boldsymbol x } - f _ { \theta } ( { \boldsymbol x } ) \| + ( 1 + L ) \| \delta \| . } \end{array}\tag{12}
$$

Proof. We want to bound the reconstruction error at $x ^ { \prime } .$ . We will relate it to the reconstruction error at x, which we know is small, and the perturbation $\delta ,$ which we want to bound.

Applying the triangle inequality: The triangle inequality states that for any three vectors $a , \dot { b } , c \dot { : } \ | | a - c | | \leq | | \overline { { a } } - b | | + | | b - c | |$ . We use it with $a \doteq x ^ { \prime } , c = f _ { \theta } ( x ^ { \prime } )$ , and b chosen to introduce $f _ { \theta } ( x )$

$$
\begin{array} { r } { \| \boldsymbol { x } ^ { \prime } - \boldsymbol { f } _ { \boldsymbol { \theta } } ( \boldsymbol { x } ^ { \prime } ) \| = \| ( \boldsymbol { x } ^ { \prime } - \boldsymbol { f } _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) ) + ( \boldsymbol { f } _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) - \boldsymbol { f } _ { \boldsymbol { \theta } } ( \boldsymbol { x } ^ { \prime } ) ) \| } \\ { \le \| \boldsymbol { x } ^ { \prime } - \boldsymbol { f } _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) \| + \| \boldsymbol { f } _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) - \boldsymbol { f } _ { \boldsymbol { \theta } } ( \boldsymbol { x } ^ { \prime } ) \| . } \end{array}\tag{13}
$$

We now bound each of these two terms separately.

Bounding the first term: Since $x ^ { \prime } = x + \delta ,$ , we apply the triangle inequality again:

$$
\begin{array} { r } { \| { \boldsymbol x } ^ { \prime } - f _ { \boldsymbol \theta } ( { \boldsymbol x } ) \| = \| ( { \boldsymbol x } + \delta ) - f _ { \boldsymbol \theta } ( { \boldsymbol x } ) \| } \\ { = \| ( { \boldsymbol x } - f _ { \boldsymbol \theta } ( { \boldsymbol x } ) ) + \delta \| } \\ { \le \| { \boldsymbol x } - f _ { \boldsymbol \theta } ( { \boldsymbol x } ) \| + \| \delta \| . } \end{array}\tag{14}
$$

Bounding the second term: We use the Lipschitz continuity assumption here. By Assumption 2.2, the SAE satisfies $\lVert f _ { \theta } ( u ) - f _ { \theta } ( v ) \rVert \stackrel { \cdot } { \le } L \lVert u - v \rVert$ for all u, v. Applying this with $u = x$ and $v = x ^ { \prime } = x + \delta \colon$

$$
\| f _ { \theta } ( x ) - f _ { \theta } ( x ^ { \prime } ) \| \ \leq \ L \| x - x ^ { \prime } \| \ = \ L \| \left( x \right) - \left( x + \delta \right) \| \ = \ L \| \delta \| .\tag{15}
$$

Substituting (14) and (15) into (13):

$$
\begin{array} { r l } & { \| x ^ { \prime } - f _ { \theta } ( x ^ { \prime } ) \| \leq \| x ^ { \prime } - f _ { \theta } ( x ) \| + \| f _ { \theta } ( x ) - f _ { \theta } ( x ^ { \prime } ) \| } \\ & { \qquad \leq \big ( \| x - f _ { \theta } ( x ) \| + \| \delta \| \big ) + L \| \delta \| } \\ & { \qquad = \| x - f _ { \theta } ( x ) \| + ( 1 + L ) \| \delta \| . } \end{array}
$$

Remark (Interpretation of $( 1 + L ) )$ . The factor $( 1 + L )$ has a precise meaning. The +1 accounts for the direct shift of the activation: moving from x to ${ \boldsymbol { x } } ^ { \prime } { \overset { \underset {  } { } } { = } } { \boldsymbol { x } } + \delta$ shifts the reconstruction target by ∥δ∥. The +L accounts for the ${ \tt S A E ^ { \prime } s }$ response to that shift: because the SAE is Lipschitz with constant $L ,$ its output can move by at most $L \| \delta \|$ when the input moves by $\lVert \dot { \delta } \rVert$ . Both movements contribute additively to the reconstruction error.

## Proof of Theorem 3.2 (Expected Bound and Covariance Decomposition)

Theorem (Theorem 3.2, restated). Under Assumption 2.2:

$$
\begin{array} { r } { \mathbb { E } \left[ \Vert x ^ { \prime } - f _ { \theta } ( x ^ { \prime } ) \Vert ^ { 2 } \right] \leq \mathbb { E } \left[ \Vert x - f _ { \theta } ( x ) \Vert ^ { 2 } \right] + ( 1 + L ) ^ { 2 } \varepsilon ^ { 2 } + 2 ( 1 + L ) \sqrt { \mathbb { E } [ \Vert x - f _ { \theta } ( x ) \Vert ^ { 2 } ] \cdot \varepsilon ^ { 2 } } , } \end{array}\tag{16}
$$

where

$$
\begin{array} { r } { \varepsilon ^ { 2 } : = \mathbb { E } \left[ \| \delta \| ^ { 2 } \right] = \operatorname { t r } \bigl ( \Delta W \Sigma _ { x _ { \mathrm { i n } } } \Delta W ^ { \top } \bigr ) . } \end{array}\tag{17}
$$

Proof. We prove the two parts separately.

## Part 1: Proving the bound (16).

From Lemma 3.1, we have for every sample:

$$
\| x ^ { \prime } - f _ { \theta } ( x ^ { \prime } ) \| \ \leq \ \| x - f _ { \theta } ( x ) \| + ( 1 + L ) \| \delta \| .\tag{18}
$$

Let us write $A : = \| x - f _ { \theta } ( x ) \|$ and $B : = ( 1 + L ) \| \delta \|$ to keep the algebra clean. The bound says $\lVert x ^ { \prime } - f _ { \theta } ( x ^ { \prime } ) \rVert \stackrel { . . . } { \le } A \stackrel { . . . } { + } B .$ , so since squaring is monotone on non-negative numbers:

$$
\| x ^ { \prime } - f _ { \theta } ( x ^ { \prime } ) \| ^ { 2 } \ \leq \ ( A + B ) ^ { 2 } \ = \ A ^ { 2 } + 2 A B + B ^ { 2 } .\tag{19}
$$

Expanding back:

$$
\begin{array} { r } { \| { \boldsymbol x } ^ { \prime } - f _ { \theta } ( { \boldsymbol x } ^ { \prime } ) \| ^ { 2 } \leq \| { \boldsymbol x } - f _ { \theta } ( { \boldsymbol x } ) \| ^ { 2 } + 2 ( 1 + L ) \| { \boldsymbol x } - f _ { \theta } ( { \boldsymbol x } ) \| \cdot \| \delta \| + ( 1 + L ) ^ { 2 } \| \delta \| ^ { 2 } . } \end{array}\tag{20}
$$

$\mathbb { E } [ \cdot ]$ is linear, so taking expectations on both sides of (20):

$$
\begin{array} { r l } & { \mathbb { E } \left[ \| x ^ { \prime } - f _ { \theta } ( x ^ { \prime } ) \| ^ { 2 } \right] \leq \mathbb { E } \left[ \| x - f _ { \theta } ( x ) \| ^ { 2 } \right] + ( 1 + L ) ^ { 2 } \mathbb { E } \left[ \| \delta \| ^ { 2 } \right] } \\ & { \qquad + 2 ( 1 + L ) \mathbb { E } \left[ \| x - f _ { \theta } ( x ) \| \cdot \| \delta \| \right] . } \end{array}\tag{21}
$$

The cross term $\mathbb { E } [ \| x - f _ { \theta } ( x ) \| \cdot \| \delta \| ]$ involves the expectation of a product of two random variables. We bound it using the Cauchy–Schwarz inequality for expectations, which states that for any two random variables U, V:

$$
{ \mathbb E } [ U V ] \le \sqrt { { \mathbb E } [ U ^ { 2 } ] { \mathbb E } [ V ^ { 2 } ] } .\tag{22}
$$

Setting $U = \| x - f _ { \theta } ( x ) \|$ and $V = \| \delta \|$

$$
\begin{array} { r } { \mathbb { E } \left[ \| x - f _ { \theta } ( x ) \| \cdot \| \delta \| \right] \leq \sqrt { \mathbb { E } \left[ \| x - f _ { \theta } ( x ) \| ^ { 2 } \right] \cdot \mathbb { E } \left[ \| \delta \| ^ { 2 } \right] } . } \end{array}\tag{23}
$$

Substituting (23) into (21) and writing $\varepsilon ^ { 2 } : = \mathbb { E } [ \| \delta \| ^ { 2 } ]$ :

$$
\begin{array} { r } { \mathbb { E } \left[ \Vert x ^ { \prime } - f _ { \theta } ( x ^ { \prime } ) \Vert ^ { 2 } \right] \leq \mathbb { E } \left[ \Vert x - f _ { \theta } ( x ) \Vert ^ { 2 } \right] + ( 1 + L ) ^ { 2 } \varepsilon ^ { 2 } + 2 ( 1 + L ) \sqrt { \mathbb { E } \left[ \Vert x - f _ { \theta } ( x ) \Vert ^ { 2 } \right] \cdot \varepsilon ^ { 2 } } . } \end{array}
$$

This is precisely (16).

(24)

<sup>□</sup>Part 1

## Part 2: Proving the covariance identity (17).

We show that $\mathbb { E } [ \| \delta \| ^ { 2 } ] = \mathrm { t r } ( \Delta W \Sigma _ { x _ { \mathrm { i n } } } \Delta W ^ { \top } )$

Recall $\delta = \Delta W x _ { \mathrm { i n } }$ . The squared norm of a vector v can be written as the dot product $\| \boldsymbol { v } \| ^ { 2 } = \boldsymbol { v } ^ { \top } \boldsymbol { v }$ . Therefore:

$$
\begin{array} { r } { \| \boldsymbol { \delta } \| ^ { 2 } = \boldsymbol { \delta } ^ { \top } \boldsymbol { \delta } = \left( \Delta \boldsymbol { W } \boldsymbol { x } _ { \mathrm { i n } } \right) ^ { \top } ( \Delta \boldsymbol { W } \boldsymbol { x } _ { \mathrm { i n } } ) = \boldsymbol { x } _ { \mathrm { i n } } ^ { \top } \Delta \boldsymbol { W } ^ { \top } \Delta \boldsymbol { W } \boldsymbol { x } _ { \mathrm { i n } } . } \end{array}\tag{25}
$$

For any deterministic matrix $M \in \mathbb { R } ^ { p \times p }$ and any random vector $v \in \mathbb { R } ^ { p }$ , following is a fundamental identity:

$$
\mathbb { E } \left[ v ^ { \top } M v \right] \ = \ \mathrm { t r } \big ( M \mathbb { E } [ v v ^ { \top } ] \big ) .\tag{26}
$$

Why does this identity hold? Writing out the quadratic form entry-by-entry: $v ^ { \top } M v ~ =$ $\Sigma _ { i , j } M _ { i j } v _ { i } v _ { j }$ . Taking expectations and using linearity: $\mathbb E [ v ^ { \top } M v ] \ : = \ : \sum _ { i , j } M _ { i j } \mathbb E [ v _ { i } v _ { j } ]$ . But $\mathbb E [ \bar { v } _ { i } v _ { j } ] = [ \mathbb E [ v v ^ { \top } ] ] _ { i j } ,$ so:

$$
\mathbb { E } [ { \boldsymbol { v } } ^ { \top } { \boldsymbol { M } } { \boldsymbol { v } } ] = \sum _ { i , j } M _ { i j } [ \mathbb { E } [ { \boldsymbol { v } } { \boldsymbol { v } } ^ { \top } ] ] _ { i j } = \sum _ { i } [ { \boldsymbol { M } } \mathbb { E } [ { \boldsymbol { v } } { \boldsymbol { v } } ^ { \top } ] ] _ { i i } = \operatorname { t r } \big ( { \boldsymbol { M } } \mathbb { E } [ { \boldsymbol { v } } { \boldsymbol { v } } ^ { \top } ] \big ) ,\tag{27}
$$

where the second equality uses the definition of matrix multiplication, and the third uses the definition of the trace of a matrix.

Applying (26) with $M = \Delta W ^ { \top } \Delta W$ and $v = x _ { \mathrm { i n } }$

$$
\begin{array} { r } { \mathbb { E } \left[ \| \delta \| ^ { 2 } \right] = \mathbb { E } \left[ x _ { \mathrm { i n } } ^ { \top } \Delta W ^ { \top } \Delta W x _ { \mathrm { i n } } \right] = \mathsf { t r } \big ( \Delta W ^ { \top } \Delta W \mathbb { E } [ x _ { \mathrm { i n } } x _ { \mathrm { i n } } ^ { \top } ] \big ) = \mathsf { t r } \big ( \Delta W ^ { \top } \Delta W \Sigma _ { x _ { \mathrm { i n } } } \big ) . } \end{array}\tag{28}
$$

The trace satisfies $\operatorname { t r } ( A B ) = \operatorname { t r } ( B A )$ for any matrices A, B of compatible dimensions (according to the cyclic property). Applying this with $A = \Delta W ^ { \top }$ and $B = { \Delta W } \Sigma _ { x _ { \mathrm { i n } } }$ :

$$
\mathrm { t r } \big ( \Delta W ^ { \top } \Delta W \Sigma _ { x _ { \mathrm { i n } } } \big ) \ = \ \mathrm { t r } \big ( \Delta W \Sigma _ { x _ { \mathrm { i n } } } \Delta W ^ { \top } \big ) .\tag{29}
$$

Chaining the equalities :

$$
\varepsilon ^ { 2 } = \mathbb { E } \left[ \| \delta \| ^ { 2 } \right] = \mathbb { E } \left[ \| \Delta W x _ { \mathrm { i n } } \| ^ { 2 } \right] = \mathrm { t r } \big ( \Delta W \Sigma _ { x _ { \mathrm { i n } } } \Delta W ^ { \top } \big ) .\tag{30}
$$

Remark (Eigenvalue expansion of $\varepsilon ^ { 2 } )$ . Writing the eigendecomposition $\begin{array} { r } { \Sigma _ { x _ { \mathrm { i n } } } = \sum _ { i = 1 } ^ { p } \lambda _ { i } v _ { i } v _ { i } ^ { \top } } \end{array}$ with $\lambda _ { 1 } \overset { \cdot } { \geq } \cdots \geq \lambda _ { p } > \overset { \cdot } { 0 }$ , the cyclic property gives:

$$
\varepsilon ^ { 2 } \ = \ \mathrm { t r } ( \Delta W \Sigma _ { x _ { \mathrm { i n } } } \Delta W ^ { \top } ) \ = \ \sum _ { i = 1 } ^ { p } \lambda _ { i } \| \Delta W v _ { i } \| ^ { 2 } .\tag{31}
$$

Each term $\lambda _ { i } \| \Delta W v _ { i } \| ^ { 2 }$ is the product of (i) the variance of the data in direction $v _ { i } ,$ and (ii) the squared magnitude of the activation perturbation in that direction. Equation (31) makes the covariance-weighting explicit: a pruning decision that perturbs a high-variance direction $( \mathrm { l a r g e } \lambda _ { i } )$ is penalised proportionally more in $\varepsilon ^ { 2 } ,$ and therefore causes proportionally more damage to the SAE.

## A.1.3 Proof of Corollary 3.3 (Paradigm Comparison)

Corollary (Corollary 3.3, restated). At any target sparsity $s ,$ the three pruning methods relate to $\varepsilon ^ { 2 }$ as follows:

1. MAGNITUDE pruning minimises $\| \Delta W \| _ { F } ^ { 2 } = \mathrm { t r } ( \Delta W I \Delta W ^ { \top } )$

2. WANDA minimises tr ∆W $\mathrm { d i a g } ( \mathbb { E } [ x _ { \mathrm { i n } } x _ { \mathrm { i n } } ^ { \top } ] ) \Delta W ^ { \top } )$ .

3. SparseGPT approximately minimises $\varepsilon ^ { 2 } = \mathrm { t r } ( \Delta W \Sigma _ { x _ { \mathrm { i n } } } \Delta W ^ { \top } )$ directly.

Proof. Each claim follows directly from the definition of the method (Definition 2.3) combined with the identity $\varepsilon ^ { 2 } = \mathrm { t r } ( \bar { \Delta W } \Sigma _ { x _ { \mathrm { i n } } } \Delta W ^ { \top } )$ ) from Theorem 3.2.

## Claim 1 (MAGNITUDE pruning).

MAGNITUDE pruning selects the pruning mask to minimise $\begin{array} { r } { \sum _ { ( i , j ) \in S } w _ { i j } ^ { 2 } = \lVert \Delta W \rVert _ { F } ^ { 2 } } \end{array}$ (since $( \Delta W ) _ { i j } = - w _ { i j }$ for pruned entries). By the covariance identity, $\lVert \Delta W \rVert _ { F } ^ { 2 } = \mathrm { t r } ( \Delta W I \Delta W ^ { \top } )$ which equals $\mathrm { t r } ( \Delta W \Sigma _ { x _ { \mathrm { i n } } } \Delta W ^ { \top } )$ only when $\Sigma _ { x _ { \mathrm { i n } } } = I$ . When $\Sigma _ { x _ { \mathrm { i n } } } \neq I$ (the typical case), MAGNITUDE pruning minimises the wrong objective.

Claim 2 (WANDA).

WANDA scores each entry $( i , j )$ by $w _ { i j } ^ { 2 } \cdot \mathbb { E } [ ( x _ { \mathrm { i n } } ) _ { j } ^ { 2 } ]$ and prunes the lowest-scoring entries. The summed score of all pruned entries is $\begin{array} { r } { \sum _ { ( i , j ) \in S } w _ { i j } ^ { 2 } \cdot { \mathbb { E } } \big [ \big ( { x _ { \mathrm { i n } } } \big ) _ { j } ^ { 2 } \big ] = \mathrm { t r } \big ( \Delta W D \Delta W ^ { \top } \big ) } \end{array}$ , where $D \ = \ \mathrm { d i a g } ( \mathbb { E } [ ( x _ { \mathrm { i n } } ) ^ { 2 } ] )$ is the diagonal matrix of per-coordinate input variances. Since $D _ { j j } = \mathbb { E } [ ( x _ { \mathrm { i n } } ) _ { j } ^ { 2 } ]$ , and noting that $[ \breve { \Sigma } _ { x _ { \mathrm { i n } } } ] _ { j j } = \mathbb { E } [ ( x _ { \mathrm { i n } } ) _ { j } ^ { 2 } ]$ , the WANDA objective is the diagonal ap proximation to $\mathrm { t r } ( \Delta W \Sigma _ { x _ { \mathrm { i n } } } \Delta W ^ { \top } )$ , obtained by discarding the off-diagonal (cross-covariance) terms of $\Sigma _ { x _ { \mathrm { i n } } }$

## Claim 3 (SparseGPT).

SparseGPT solves the layer-wise reconstruction problem $\begin{array} { r } { \operatorname* { m i n } _ { \Delta W : \mathrm { s p a r s i t y } \left( W + \Delta W \right) = s } \mathbb { E } \left[ \left\| \Delta W x _ { \mathrm { i n } } \right\| ^ { 2 } \right] } \end{array}$ via a second-order approximation using the empirical Hessian $\begin{array} { r } { \hat { \Sigma } _ { x _ { \mathrm { i n } } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } x _ { \mathrm { i n } } ^ { ( n ) } ( x _ { \mathrm { i n } } ^ { ( n ) } ) ^ { \top } } \end{array}$ . By Theorem 3.2, $\mathbb { E } [ \| \Delta W x _ { \mathrm { i n } } \| ^ { 2 } ] = \mathrm { t r } ( \Delta W \Sigma _ { x _ { \mathrm { i n } } } \Delta W ^ { \top } )$ , so SparseGPT directly (approximately) minimises $\varepsilon ^ { 2 }$ □

Remark A.1 (Hierarchy of approximations). The three methods form a natural hierarchy in how faithfully they approximate the true objective $\varepsilon ^ { 2 }$

$$
\underbrace { \mathrm { t r } ( \Delta W I \Delta W ^ { \top } ) } _ { \mathrm { M A G N I T U D E } } \prec \underbrace { \mathrm { t r } ( \Delta W \mathrm { d i a g } ( \Sigma _ { x _ { \mathrm { i n } } } ) \Delta W ^ { \top } ) } _ { \mathrm { W A N D A } } \prec \underbrace { \mathrm { t r } ( \Delta W \Sigma _ { x _ { \mathrm { i n } } } \Delta W ^ { \top } ) } _ { \mathrm { S p a r s e G P T } } ,\tag{32}
$$

where ≺ denotes “is a coarser approximation of the true perturbation energy than”. MAGNI-TUDE pruning ignores $\Sigma _ { x _ { \mathrm { i n } } }$ entirely; WANDA uses only its diagonal; SparseGPT uses the full matrix.

## A.2 SAEBench Evaluation Suite

This appendix gives a detailed overview of the SAEBench evaluation suite (Karvonen et al., 2025). We also provide intuitive understanding of each metric category (Chanin et al., 2025; Karvonen et al., 2024; 2025). Ultimately, we conclude the appendix by talking about the rationale behind the SCR and TPP threshold choices.

## A.2.1 Core metrics

KL divergence score. Let $\mathcal { D } _ { \mathrm { o r i g } }$ denote the output distribution of the original model and $\mathcal { D } _ { \mathrm { S A E } }$ the output distribution when the residual stream is replaced by the SAE reconstruction. The zero-ablation distribution $\mathcal { D } _ { z \mathrm { e r o } }$ is obtained by replacing the residual stream with zeros. The KL divergence score is defined as:

$$
\mathrm { K L } \mathrm { s c o r e } = 1 - \frac { D _ { \mathrm { K L } } ( \mathcal { D } _ { \mathrm { o r i g } } \parallel \mathcal { D } _ { \mathrm { S A E } } ) } { D _ { \mathrm { K L } } ( \mathcal { D } _ { \mathrm { o r i g } } \parallel \mathcal { D } _ { \mathrm { z e r o } } ) } ,\tag{33}
$$

normalized so that a perfect reconstruction yields a score of 1 and a zero-ablation yields a score of 0. This metric is sensitive to distributional shifts in the model’s predictions, not just pointwise reconstruction error.

CE loss score. Let $H _ { \mathrm { o r i g } } , H ^ { * } ,$ and $H _ { \mathrm { a b l a t i o n } }$ denote the cross-entropy loss of the model under the original activations, SAE-reconstructed activations, and zero-ablated activations, respectively. The CE loss score is defined as:

$$
\mathrm { C E ~ s c o r e } = \frac { H _ { \mathrm { a b l a t i o n } } - H ^ { * } } { H _ { \mathrm { a b l a t i o n } } - H _ { \mathrm { o r i g } } } .\tag{34}
$$

A score of 1 indicates that the SAE reconstruction perfectly recovers the original model’s cross-entropy loss, while a score of 0 corresponds to the zero-ablation baseline. Unlike the KL divergence score, this metric emphasizes the impact of reconstruction quality on next-token prediction performance.

Explained variance. For a layer with activation $\boldsymbol { x } \in \mathbb { R } ^ { d }$ and SAE reconstruction ${ \hat { x } } = f _ { \theta } ( x )$ the explained variance is:

$$
\mathrm { E V } = 1 - { \frac { \mathrm { V a r } ( x - { \hat { x } } ) } { \mathrm { V a r } ( x ) } } ,\tag{35}
$$

where variance is computed over the token dimension. A higher explained variance indicates that the SAE accounts for more of the variance in the original activations. This metric is sensitive to systematic reconstruction failures that affect large portions of the activation space.

Mean squared error (MSE). The MSE between the original activation x and its reconstruction xˆ is:

$$
\mathrm { M S E } = \mathbb { E } \left[ \| x - f _ { \theta } ( x ) \| ^ { 2 } \right] ,\tag{36}
$$

measured in the same units as the activation space. Unlike the normalized metrics above, MSE is an absolute measure and therefore varies substantially across layers and model sizes, as later layers tend to have activations with larger norms. We report MSE alongside the normalized metrics to provide a scale-sensitive view of reconstruction quality.

Cosine similarity. The mean cosine similarity between original and reconstructed activations is:

$$
\mathsf { C o s S i m } = \mathbb { E } \left[ \frac { \langle x , f _ { \theta } ( x ) \rangle } { \| x \| \| f _ { \theta } ( x ) \| } \right] .\tag{37}
$$

This metric captures directional alignment between original and reconstructed activations, independent of their magnitudes. It is particularly informative when activation norms vary substantially across layers, since magnitude-preserving but direction-distorting perturbations appear benign under MSE yet are detected here.

## A.2.2 Feature Absorption

Feature absorption happens when SAEs, due to sparsity pressure, fail to represent a general feature directly. Instead of having both a general latent (e.g., “starts with S”) and specific latents (e.g., “short”), the SAE only activates the specific ones. This makes the general latent look unreliable =⇒ The general latent may fire for many tokens but mysteriously may miss some cases, because those cases were absorbed into other latents.

## SAEBench Utilization

• Picking a ground-truth concept (e.g., “starts with S”).

• Identifying the SAE latent most responsible for that concept.

• Checking where that latent fails to fire.

• Seeing if other latents step in to cover the missing cases.

The final absorption score is the fraction of a concept’s cases that are not covered by its main latent but are covered by other latents.

Intuitive Example Suppose the set of all S-words is:

$$
\mathrm { s h o r t , ~ \mathsf { s m a l l } , ~ \mathsf { s a n d } , ~ \mathsf { s u n } , ~ \mathsf { s e a } , ~ \mathsf { s o a p } , ~ \mathsf { s t a r } , ~ \mathsf { s l o w } , ~ \mathsf { s o u n d } , ~ \mathsf { s h a r p } }
$$

SAEBench finds that latent ${ \mathsf { L } } { \mathsf { s } }$ is the best candidate for the concept starts with S. Let us assume that ${ \mathsf { L } } { \mathsf { s } }$ fires on $7$ out of 10 tokens: sand, sun, sea, soap, star, slow, sharp, and misses the tokens: short, small, sound. Thus, the recall of the latent $\begin{array} { r } { \mathsf { L } _ { \mathsf { S } } = \frac { 7 } { 1 0 } = 7 0 \% } \end{array}$

SAEBench then checks whether other latents cover the missing three tokens. Let us assume the latent L<sub>short</sub> fires on tokens short and small, and L<sub>sound</sub> fires on the token sound. Now all missing cases are covered, but by different latents.

Raw absorption = 30% (because 3/10 cases were absorbed away from the latent L<sub>S</sub>). Each missing case is explained elsewhere, so they are not pure false negatives. So the score reported would be 30% absorption, with full compensation of the missing cases by other latents. If none of the missing tokens had been covered elsewhere, the absorption score would flag this as a bad collapse.

Thus, the absorption score in SAEBench = “How often is a general feature incomplete, and to what extent do other latents pick up the missing pieces?”

Lower absorption scores → better interpretability! A low absorption score means the general latent is reliable. It means this latent fires consistently when the concept is present in the input. A tiny portion of any concept is fragmented into other specific latents.

Sparsity(L<sub>0</sub>) defines the number of latents that are allowed to fire per token. Low $L _ { 0 }$ (high sparsity) implies very few latents fire, high $L _ { 0 }$ (low sparsity) implies many latents can fire. From the SAEBench results, we can infer that at low $L _ { 0 } ^ { \setminus } ,$ SAEs aggressively minimize number of active latents =⇒ high absorption score.

## A.2.3 Spurious Correlation Coefficient

SCR score quantifies the capability of an SAE to find and remove spurious correlations (irrelevant signals) while preserving the signal we actually care about.

SAEBench Utilization SAEBench takes biased datasets. It uses probes and SAE latents to identify the spurious features (bias-creating features). Then it ablates those latents and checks if classifier accuracy on the actual task improves.

Intuitive Example Suppose we have a dataset with biased representation, featuring male professors and female nurses. The task is to predict a person’s profession based on their individual characteristics. Here, gender is the spurious feature.

We train a classifier on this biased data, which will learn the gender-profession shortcut. Let us assume:

$$
\mathrm { A c c u r a c y o n b i a s e d d a t a } ( A _ { b a s e } ) = 9 5 \%
$$

Next, we include female professors and male nurses in the dataset, and again train a classifier. Let us assume:

$$
\mathrm { A c c u r a c y o n u n b i a s e d d a t a } ( A _ { b e s t } ) = 7 5 \%
$$

Ultimately, we use SAE latents to ablate spurious features. We find gender-related latents, set them to zero, and retrain the classifier on the biased data. Let us assume:

$$
\mathrm { A c c u r a c y ~ o n ~ b i a s e d ~ d a t a ~ w i t h ~ a b l a t e d ~ l a t e n t s } ( A _ { a b l } ) = 8 0 ^ { \circ } { \circ }
$$

Finally, we compute the SCR score as follows:

$$
\begin{array} { c } { S C R = \frac { A _ { a b l } - A _ { b a s e } } { A _ { b e s t } - A _ { b a s e } } } \\ { = \frac { 8 0 - 9 5 } { 7 5 - 9 5 } = 0 . 7 5 } \end{array}
$$

Higher SCR scores → better SAE! A score of 1.0 indicates that SAE removed all spurious bias perfectly. Thus, SCR tells you whether an SAE can cut out spurious shortcuts like gender bias while keeping the real task intact.

## A.2.4 Targeted Probe Perturbation

TPP score quantifies the capability of an SAE to isolate individual concepts so that removing its latents only affects that concept and leaves others unchanged.

SAEBench Utilization SAEBench applies TPP to any multiclass dataset to test whether SAE features for one class are cleanly separated from others. We train probes (small classifiers) for each class. Next, we find the SAE latents most used by a specific probe and ablate them.

TPP is a generalization of SCR. Instead of needing biased datasets, it can test disentanglement on any multiclass dataset.

Intuitive Example Suppose we have an Amazon reviews dataset with books, electronics, home supplies and the goal is to check if book latents are isolated.

We train probes for each class. Let us assume:

$$
\mathrm { A c c u r a c y f o r B o o k s p r o b e = 8 5 \% }
$$

Accuracy for Electronics probe = 75%

Accuracy for Home supplies probe = 80%

After ablating book probes, let us assume:

$$
{ \mathrm { A c c u r a c y ~ f o r ~ B o o k s ~ p r o b e } } = 5 5 \% \implies \Delta = - 3 0 \%
$$

$$
{ \mathrm { A c c u r a c y ~ f o r ~ E l e c t r o n i c s ~ p r o b e } } = 7 0 \% \implies \Delta = - 5 \%
$$

$$
{ \mathrm { A c c u r a c y ~ f o r ~ H o m e ~ s u p p l i e s ~ p r o b e } } = 7 9 \% \implies \Delta = - 1 \%
$$

Since we are playing around with the target: books, we compute the TPP scores as follows:

$$
\begin{array} { l } { { T P P = \mathrm { m e a n } \Delta _ { i = j } - \mathrm { m e a n } \Delta _ { i \neq j } } } \\ { { \ = ( - 3 0 ) - ( { \frac { - 5 - 1 } { 2 } } ) } } \\ { { \ = - 3 0 + 3 = - 2 7 } } \end{array}
$$

Since TPP is normalized in practice, this would be reported as a high positive score showing the Book latents only hurt Book classification.

Higher TPP scores → better SAE! A higher TPP score is better, indicating that the SAE identified features specific to one class. Removing them tanks the target probe but leaves others intact. A lower TPP score means SAE latents are entangled.

## A.2.5 Threshold Sensitivity ofSCR and TPP

SCR and TPP are each computed at seven ablation thresholds $k \in \{ 2 , 5 , 1 0 , 2 0 , 5 0 , 1 0 0 , 5 0 0 \}$ corresponding to the number of top-scoring SAE latents ablated per concept direction. The choice of k is a hyperparameter of the evaluation, not of the model or the pruning method, and a valid evaluation protocol should not depend sensitively on it.

We verify this for gemma-2-2b at 50% sparsity in Figure 4, averaging across all layers. At every value of k the ordering SPARSEGPT ≥ WANDA ≥ MAGNITUDE holds for both SCR and TPP.

Figure 4 examines how SCR and TPP vary with the ablation threshold k. The raw metric values for all methods increase with k for both SCR and TPP, as ablating more latents captures more of the relevant concept signal. The method ordering is consistent across all evaluated thresholds: MAGNITUDE pruning shows the largest degradation throughout, while WANDA and SPARSEGPT remain close to each other and substantially closer to the dense baseline. The percentage change from the dense baseline stabilises for $k \geq 1 0$ for the activation-aware methods, while MAGNITUDE pruning shows larger and less stable degradation across thresholds. Based on these observations, we can confirm the choice of $k \stackrel { \sim } { = } 1 0$ as the canonical threshold, as it lies in the stable regime and produces the clearest separation between pruning paradigms.

Right: mean % change from dense

## SCR & TPP Threshold Comparison (sparsity=50%)

![](images/e72db046b9f20791288b5e8f36f14d89a90907bf7df9ce924e5c9a29bc818476.jpg)

![](images/416cd648868c75e35bd16f5748e2fdb486774f6381ee71d340812fd9f7557b5c.jpg)

![](images/5831a012bd29d14ef7dd06aa9a2488542671585c76194f63bfeba1287d69521b.jpg)

![](images/b1f23b5bdf1a2e2e9c14f13bdb6a2889dc960b15efd731467279449942cf96b1.jpg)  
Figure 4: SCR/TPP metric stability across ablation thresholds for gemma-2-2b at 50% sparsity. Left column: mean raw metric values as a function of ablation threshold k ∈ {2, 5, 10, 20, 50, 100, 500} for SCR (top)/ TPP (bottom), alongside the dense baseline (dotted). Right column: corresponding mean percentage change from the dense baseline.

## A.3 Per-Layer Results for gemma-2-2b

Full layer-by-layer SAEBench metrics for gemma-2-2b at 50% sparsity, split by depth group. These results support the aggregated group-level summaries reported in Table 1 and the per-layer profiles shown in Figures 1 and 2.

## A.4 Empirical Perturbation Energy

Table 7 reports the empirical perturbation energy $\varepsilon ^ { 2 } = \mathbb { E } \left\lceil \| \Delta W x _ { \mathrm { i n } } \| ^ { 2 } \right\rceil$ , averaged within each layer group, for each pruning method on gemma-2-2b at 50% sparsity, estimated over the calibration set of Sec. 5. The ordering $\varepsilon ^ { 2 }$ (SPARSEGPT) $< \varepsilon ^ { 2 } ( \dot { \mathrm { W A N D A } } ) < \varepsilon ^ { 2 }$ (MAGNITUDE) predicted by Corollary 3.3 is confirmed at every layer group: MAGNITUDE incurs roughly twice the perturbation energy of the activation-aware methods. This moderate $\varepsilon ^ { 2 }$ ratio is amplified into the large SAEBench gaps of Table 1 through the quadratic and cross terms of Theorem 3.2, consistent with the bound’s structure.

Notably, per-layer $\varepsilon ^ { 2 }$ does not mirror the layer sensitivity profile of Figure 3: middle layers exhibit the lowest local $\varepsilon ^ { 2 }$ yet the highest SAE degradation. Middle-layer vulnerability is therefore driven not by locally larger perturbation but by the accumulation of upstream perturbation from early layers propagating through residual connections. This further motivates the layer-wise allocation of Sec. 6: protecting early layers reduces the perturbation that cascades into vulnerable middle layers.

<table><tr><td rowspan="2">Layers Method</td><td rowspan="2"></td><td rowspan="2">Spar. (%)</td><td colspan="5">Core</td><td colspan="2">Absorption</td><td>SCR</td><td>TPP</td></tr><tr><td>KL Score↑</td><td>CE Score↑</td><td>Expl. Var.↑</td><td>Cos Sim↑</td><td>MSE ↓</td><td>Abs. Frac.↓</td><td>Full Abs.↓</td><td>Top-10 ↑</td><td>Top-10 ↑</td></tr><tr><td>0</td><td>Baseline</td><td>0</td><td>0.9966</td><td>0.9967</td><td>0.9453</td><td>0.9766</td><td>0.1396</td><td>0.0131</td><td>0.0324</td><td>0.1488</td><td>0.0272</td></tr><tr><td>Layer</td><td>MAGNITUDE</td><td>50</td><td>0.9915</td><td>0.9962</td><td>0.8984</td><td>0.9531</td><td>0.2754</td><td>0.0141</td><td>0.0432</td><td>0.1385</td><td>0.0327</td></tr><tr><td></td><td>WANDA</td><td>50</td><td>0.9964</td><td>0.9932</td><td>0.9375</td><td>0.9727</td><td>0.1582</td><td>0.0127</td><td>0.0339</td><td>0.1605</td><td>0.0242</td></tr><tr><td></td><td>SPARSEGPT</td><td>50</td><td>0.9953</td><td>0.9948</td><td>0.9375</td><td>0.9727</td><td>0.1611</td><td>0.0133</td><td>0.0346</td><td>0.1514</td><td>0.0206</td></tr><tr><td rowspan="2">1 Layyer</td><td>Baseline</td><td>0</td><td>0.9958</td><td>0.9951</td><td>0.9219</td><td>0.9648</td><td>0.2402</td><td>0.0989</td><td>0.0913</td><td>0.1788</td><td>0.0336</td></tr><tr><td>MAGNITUDE</td><td>50</td><td>0.9914</td><td>0.9962</td><td>0.8750</td><td>0.9375</td><td>0.3984</td><td>0.1130</td><td></td><td>0.1012 0.1456</td><td>0.0357</td></tr><tr><td rowspan="2"></td><td>WANDA</td><td>50</td><td>0.9956</td><td>0.9914</td><td>0.9141</td><td>0.9609</td><td>0.2793</td><td>0.0967</td><td>0.0820</td><td>0.1732</td><td>0.0241</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9942</td><td>0.9931</td><td>0.9102</td><td>0.9570</td><td>0.2793</td><td>0.0853</td><td>0.0807</td><td>0.1602</td><td>0.0271</td></tr><tr><td rowspan="2">2</td><td>Baseline</td><td>0</td><td>0.9960</td><td>0.9967</td><td>0.9141</td><td>0.9609</td><td>0.2578</td><td>0.0900</td><td>0.0224</td><td>0.1279</td><td>0.0184</td></tr><tr><td>MAGNITUDE</td><td>50</td><td>0.9912</td><td>0.9962</td><td>0.8516</td><td>0.9219</td><td>0.4512</td><td>0.1002</td><td>0.0458</td><td>0.1024</td><td>0.0195</td></tr><tr><td rowspan="2">Layye</td><td>WANDA</td><td>50</td><td>0.9956</td><td>0.9914</td><td>0.8945</td><td>0.9492</td><td>0.3008</td><td>0.0880</td><td>0.0246</td><td>0.1048</td><td>0.0105</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9943</td><td>0.9948</td><td>0.8867</td><td>0.9453</td><td>0.3086</td><td>0.0748</td><td>0.0226</td><td>0.0870</td><td>0.0120</td></tr><tr><td rowspan="2">3 Laye</td><td>Baseline</td><td></td><td>0 0.9930</td><td>0.9934</td><td>0.8828</td><td>0.9453</td><td>0.3906</td><td>0.1136</td><td>0.0570</td><td>0.1250</td><td>0.0190</td></tr><tr><td>MAGNITUDE</td><td>50</td><td>0.9843</td><td>0.9772</td><td>0.8125</td><td>0.9023</td><td>0.6211</td><td>0.1027</td><td>0.0680</td><td>0.0971</td><td>0.0108</td></tr><tr><td rowspan="2"></td><td>WANDA SPARSEGPT</td><td>50</td><td>0.9912</td><td>0.9829</td><td>0.8594</td><td>0.9336</td><td>0.4297</td><td>0.1145</td><td>0.0614</td><td>0.1203</td><td>0.0142</td></tr><tr><td></td><td>50</td><td>0.9899</td><td>0.9880</td><td>0.8516</td><td>0.9258</td><td>0.4512</td><td>0.1182</td><td>0.0704</td><td>0.1202</td><td>0.0153</td></tr><tr><td rowspan="2">4 Layye</td><td>Baseline</td><td></td><td>0 0.9964</td><td>0.9967</td><td>0.9062</td><td>0.9531</td><td>0.3320</td><td>0.1330</td><td>0.0387</td><td>0.1160</td><td>0.0278</td></tr><tr><td>MAGNITUDE</td><td>50</td><td>0.9876</td><td>0.9848</td><td>0.8398</td><td>0.9180</td><td>0.5352</td><td>0.1736</td><td>0.0736</td><td>0.0824</td><td>0.0144</td></tr><tr><td rowspan="3">5</td><td>WANDA</td><td>50</td><td>0.9957</td><td>0.9914</td><td>0.8828</td><td>0.9414</td><td>0.3711</td><td>0.1365</td><td>0.0319</td><td>0.1104</td><td>0.0154</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9939</td><td>0.9931</td><td>0.8789</td><td>0.9375</td><td>0.3848</td><td>0.1657</td><td>0.0552</td><td>0.0951</td><td>0.0155</td></tr><tr><td>Baseline</td><td></td><td>00.9930</td><td>0.9918</td><td>0.8711</td><td>0.9336</td><td>0.5312</td><td>0.0444</td><td>0.0212</td><td>0.1551</td><td>0.0159</td></tr><tr><td rowspan="3">Layer</td><td>MAGNITUDE</td><td>50</td><td>0.9752</td><td>0.9582</td><td>0.8203</td><td>0.9062</td><td>0.7812</td><td>0.0543</td><td>0.0316</td><td>0.1002</td><td>0.0194</td></tr><tr><td>WANDA</td><td>50</td><td>0.9916</td><td>0.9846</td><td>0.8516</td><td>0.9258</td><td>0.5547</td><td>0.0492</td><td>0.0261</td><td>0.1303</td><td>0.0182</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9888</td><td>0.9863</td><td>0.8477</td><td>0.9219</td><td>0.5859</td><td>0.0570</td><td>0.0302</td><td>0.1232</td><td>0.0182</td></tr><tr><td rowspan="2">6 Layer</td><td>Baseline</td><td></td><td>00.9937</td><td>0.9934</td><td>0.8750</td><td>0.9375</td><td>0.5352</td><td>0.0393</td><td>0.0134</td><td>0.1455</td><td>0.0334</td></tr><tr><td>MAGNITUDE</td><td>50</td><td>0.9817</td><td>0.9772</td><td>0.8203</td><td>0.9062</td><td>0.7656</td><td>0.0504</td><td></td><td>0.0252 0.0988</td><td>0.0226</td></tr><tr><td rowspan="2"></td><td>WANDA SPARSEGPT</td><td>50</td><td>0.9923</td><td>0.9846</td><td>0.8594</td><td>0.9297</td><td>0.5742</td><td>0.0454</td><td>0.0189</td><td>0.1360</td><td>0.0245</td></tr><tr><td></td><td>50</td><td>0.9906</td><td>0.9897</td><td>0.8516</td><td>0.9258</td><td>0.5977</td><td>0.0454</td><td>0.0217</td><td>0.1189</td><td>0.0288</td></tr><tr><td rowspan="2">7 Layer</td><td>Baseline</td><td>0</td><td>0.9928</td><td>0.9918</td><td>0.8828</td><td>0.9375</td><td>0.6602</td><td>0.0347</td><td>0.0164</td><td>0.2941</td><td>0.0324</td></tr><tr><td>MAGNITUDE WANDA</td><td>50</td><td>0.9794 0.9913</td><td>0.9734 0.9829</td><td>0.8281 0.8594</td><td>0.9062 0.9258</td><td>0.9688 0.7109</td><td>0.0471 0.0475</td><td>0.0259 0.0201</td><td>0.1569 0.2210</td><td>0.0084 0.0173</td></tr><tr><td rowspan="2">8</td><td>SPARSEGPT</td><td>50 50</td><td>0.9899</td><td>0.9880</td><td>0.8477</td><td>0.9180</td><td>0.7656</td><td>0.0508</td><td>0.0222</td><td>0.2015</td><td>0.0244</td></tr><tr><td>Baseline</td><td></td><td>00.9925</td></tr><tr><td>9</td><td>Baseline</td><td>0</td><td>0.9917</td><td>0.9901</td><td>0.8711</td><td>0.9297</td><td>0.9844</td><td>0.0555</td><td>0.0498</td><td>0.2386</td><td>0.0247</td></tr><tr><td>Layer</td><td>MAGNITUDE</td><td>50</td><td>0.9664</td><td>0.9544</td><td>0.7773</td><td>0.8789</td><td>1.3438</td><td>0.0555</td><td>0.0498</td><td>0.0858</td><td>0.0118</td></tr><tr><td></td><td>WANDA SPARSEGPT</td><td>50</td><td>0.9915</td><td>0.9846</td><td>0.8438</td><td>0.9141</td><td>1.0234 1.0938</td><td>0.0429 0.0527</td><td>0.0310 0.0414</td><td>0.1830</td><td>0.0170 0.0138</td></tr><tr><td></td><td>Baseline</td><td>50</td><td>0.9888</td><td>0.9880</td><td>0.8281</td><td>0.9062</td><td></td><td></td><td></td><td>0.1565</td><td></td></tr><tr><td>10</td><td>MAGNITUDE</td><td></td><td>0 0.9909</td><td>0.9901</td><td>0.8672</td><td>0.9258</td><td>1.1953</td><td>0.0708</td><td>0.0541</td><td>0.2430</td><td>0.0220</td></tr><tr><td>Layyer</td><td>WANDA</td><td>50 50</td><td>0.9733 0.9914</td><td>0.9620 0.9863</td><td>0.7812 0.8438</td><td>0.8789 0.9141</td><td>1.6719 1.2109</td><td>0.0967 0.0701</td><td>0.0769 0.0518</td><td>0.0921 0.1949</td><td>0.0081 0.0153</td></tr><tr><td></td><td>SPARSEGPT</td><td>50</td><td>0.9887</td><td>0.9897</td><td>0.8203</td><td>0.9062</td><td>1.3281</td><td>0.0834</td><td>0.0659</td><td>0.1593</td><td>0.0159</td></tr><tr><td>11</td><td>Baseline</td><td></td><td>0 0.9900</td><td>0.9885</td><td>0.8750</td><td>0.9258</td><td>1.4844</td><td>0.1061</td><td>0.0916</td><td>0.2557</td><td>0.0354</td></tr><tr><td>Layyer</td><td>MAGNITUDE</td><td>50</td><td>0.9649</td><td>0.9430</td><td>0.7695</td><td>0.8711</td><td>2.2344</td><td>0.1256</td><td>0.1183</td><td>0.0720</td><td>0.0064</td></tr><tr><td></td><td>WANDA SPARSEGPT</td><td>50</td><td>0.9908</td><td>0.9863</td><td>0.8516</td><td>0.9102</td><td>1.5000</td><td>0.1020</td><td>0.0885</td><td>0.2178</td><td>0.0265</td></tr><tr><td>12</td><td></td><td>50</td><td>0.9883</td><td>0.9897</td><td>0.8203</td><td>0.8984</td><td>1.7109</td><td>0.1181</td><td>0.1046</td><td>0.1827</td><td>0.0184</td></tr><tr><td></td><td>Baseline</td><td></td><td>0 0.9896</td><td>0.9885</td><td>0.8750</td><td>0.9219</td><td>1.5391</td><td>0.0992</td><td>0.0840</td><td>0.2924</td><td>0.0250</td></tr><tr><td>Layer</td><td>MAGNITUDE</td><td>50</td><td>0.9736</td><td>0.9696</td><td>0.7773</td><td>0.8750</td><td>2.0781</td><td>0.1212</td><td>0.0971</td><td>0.1018</td><td>0.0056</td></tr><tr><td></td><td>WANDA</td><td>50</td><td>0.9904</td><td>0.9863</td><td>0.8516</td><td>0.9102</td><td>1.5703</td><td>0.0987</td><td>0.0825</td><td>0.2181</td><td>0.0242</td></tr><tr><td></td><td>SPARSEGPT</td><td>50</td><td>0.9879</td><td>0.9897</td><td>0.8164</td><td>0.8984</td><td>1.8281</td><td>0.1112</td><td>0.0966</td><td>0.2046</td><td>0.0206</td></tr><tr><td>13</td><td>Baseline</td><td></td><td>00.9885</td><td>0.9868</td><td>0.8750</td><td>0.9297</td><td>1.9766</td><td>0.1344</td><td>0.1252</td><td>0.2864</td><td>0.0275</td></tr><tr><td>Layye</td><td>MAGNITUDE</td><td>50</td><td>0.9692</td><td>0.9582</td><td>0.7812</td><td>0.8828</td><td>3.0156</td><td>0.1609</td><td>0.1517</td><td>0.2095</td><td>0.0125</td></tr><tr><td></td><td>WANDA</td><td>50</td><td>0.9893</td><td>0.9846</td><td>0.8477</td><td>0.9180</td><td>2.1250</td><td>0.1279</td><td>0.1183</td><td>0.2008</td><td>0.0318</td></tr><tr><td></td><td>SPARSEGPT</td><td>50</td><td>0.9871</td><td>0.9897</td><td>0.8242</td><td>0.9062</td><td>2.5156</td><td>0.1422</td><td>0.1401</td><td>0.1783</td><td>0.0152</td></tr><tr><td>14</td><td>Baseline</td><td></td><td>00.9885</td><td>0.9868</td><td>0.8711</td><td>0.9297</td><td>2.1719</td><td></td><td>0.0288</td><td>0.3756</td><td>0.0440</td></tr><tr><td>Layer</td><td>MAGNITUDE</td><td></td><td>0.9562</td><td></td><td></td><td></td><td></td><td>0.0333</td><td></td><td></td><td>0.0102</td></tr><tr><td></td><td>WANDA</td><td>50 50</td><td>0.9893</td><td>0.9316 0.9863</td><td>0.7734 0.8438</td><td>0.8750 0.9180</td><td>3.3438 2.3594</td><td>0.0399 0.0354</td><td>0.0390 0.0291</td><td>0.1078 0.2755</td><td>0.0439</td></tr><tr><td>15</td><td>SPARSEGPT Baseline</td><td>50</td><td>0.9869</td><td>0.9897</td><td>0.8281</td><td>0.9102</td><td>2.7812</td><td>0.0401</td><td>0.0351</td><td>0.2309</td><td>0.0380</td></tr><tr><td></td><td>MAGNITUDE</td><td></td><td>0 0.9872</td><td>0.9868</td><td>0.8906</td><td>0.9375</td><td>2.3594</td><td>0.0314</td><td>0.0274</td><td>0.3937</td><td>0.0584</td></tr><tr><td>Layer</td><td>WANDA</td><td>50 50</td><td>0.9700 0.9881</td><td>0.9734 0.9846</td><td>0.8047 0.8594</td><td>0.8945 0.9258</td><td>4.0938 2.6719</td><td>0.0382 0.0308</td><td>0.0390 0.0262</td><td>0.1860 0.2847</td><td>0.0090 0.0378</td></tr><tr><td></td><td>SPARSEGPT</td><td>50</td><td>0.9857</td><td>0.9897</td><td>0.8398</td><td>0.9141</td><td>3.1094</td><td>0.0377</td><td>0.0347</td><td>0.2341</td><td>0.0356</td></tr><tr><td>16</td><td>Baseline</td><td></td><td>0 0.9862</td><td>0.9852</td><td>0.8789</td><td>0.9336</td><td>3.2969</td><td>0.0318</td><td>0.0254</td><td>0.3583</td><td>0.0752</td></tr><tr><td>Layyer</td><td>MAGNITUDE</td><td>50</td><td>0.9566</td><td>0.9468</td><td>0.7812</td><td>0.8789</td><td>5.7500</td><td>0.0334</td><td>0.0319</td><td>0.2037</td><td>0.0116</td></tr><tr><td></td><td>WANDA</td><td>50</td><td>0.9868</td><td>0.9846</td><td>0.8555</td><td>0.9219</td><td>3.6250</td><td>0.0283</td><td>0.0221</td><td>0.3144</td><td>0.0377</td></tr><tr><td></td><td>SPARSEGPT</td><td>50</td><td>0.9842</td><td>0.9897</td><td>0.8320</td><td>0.9102</td><td>4.2188</td><td>0.0317</td><td>0.0295</td><td>0.3069</td><td>0.0398</td></tr><tr><td>17</td><td>Baseline</td><td></td><td></td><td>0.9852</td><td>0.8828</td><td>0.9375</td><td></td><td>0.0381</td><td>0.0409</td><td>0.2987</td><td></td></tr><tr><td></td><td></td><td></td><td>0 0.9859</td><td></td><td></td><td></td><td>3.5156</td><td></td><td></td><td></td><td>0.0693</td></tr><tr><td>Layer</td><td>MAGNITUDE</td><td>50</td><td>0.9496</td><td>0.9392</td><td>0.7578</td><td>0.8672</td><td>6.1875</td><td>0.0434</td><td>0.0538</td><td>0.1120</td><td>0.0048</td></tr><tr></tr><tr><td rowspan="2"></td><td rowspan="2">Layers Method</td><td rowspan="2">Spar. (%)</td><td colspan="5">Core</td><td colspan="2">Absorption</td><td>SCR</td><td>TPP</td></tr><tr><td>KL Score↑</td><td>CE Score↑</td><td>Expl. Var.↑</td><td>Cos Sim↑</td><td>MSE ↓</td><td>Abs. Frac.↓</td><td>Full Abs.↓</td><td>Top-10 ↑</td><td>Top-10 ↑</td></tr><tr><td>18</td><td>Baseline</td><td>0</td><td>0.9855</td><td>0.9852</td><td>0.8867</td><td>0.9375</td><td>4.4062</td><td>0.0951</td><td>0.1101</td><td>0.3244</td><td>0.0655</td></tr><tr><td></td><td>MAGNITUDE</td><td>50</td><td>0.9426</td><td>0.9278</td><td>0.7617</td><td>0.8672</td><td>6.9375</td><td>0.0795</td><td>0.1189</td><td>0.1193</td><td>0.0063</td></tr><tr><td rowspan="2">Layyer</td><td>WANDA</td><td>50</td><td>0.9854</td><td>0.9846</td><td>0.8672</td><td>0.9297</td><td>4.8750</td><td>0.0813</td><td>0.0921</td><td>0.2761</td><td>0.0511</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9831</td><td>0.9880</td><td>0.8477</td><td>0.9180</td><td>5.5000</td><td>0.0815</td><td>0.1070</td><td>0.2658</td><td>0.0573</td></tr><tr><td rowspan="2">19</td><td>Baseline</td><td></td><td>0 0.9844</td><td>0.9835</td><td>0.8828</td><td>0.9375</td><td>5.5312</td><td>0.0823</td><td>0.0935</td><td>0.3398</td><td>0.0660</td></tr><tr><td>MAGNITUDE</td><td>50</td><td>0.9600</td><td>0.9658</td><td>0.7773</td><td>0.8789</td><td>9.3750</td><td>0.0787</td><td>0.1092</td><td>0.1583</td><td>0.0126</td></tr><tr><td rowspan="2">Layyer</td><td>WANDA</td><td>50</td><td>0.9845</td><td>0.9846</td><td>0.8633</td><td>0.9297</td><td>6.0312</td><td>0.0861</td><td>0.0931</td><td>0.3061</td><td>0.0447</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9823</td><td>0.9880</td><td>0.8438</td><td>0.9141</td><td>6.8750</td><td>0.0767</td><td>0.0975</td><td>0.2888</td><td>0.0482</td></tr><tr><td rowspan="2">20</td><td>Baseline</td><td>0</td><td>0.9827</td><td>0.9819</td><td>0.8750</td><td>0.9336</td><td>7.2500</td><td>0.1144</td><td>0.1249</td><td>0.4253</td><td>0.0677</td></tr><tr><td>MAGNITUDE</td><td>50</td><td>0.9571</td><td>0.9506</td><td>0.7617</td><td>0.8711</td><td>12.5000</td><td>0.1091</td><td>0.1540</td><td>0.2972</td><td>0.0093</td></tr><tr><td rowspan="2">Layer</td><td>WANDA</td><td>50</td><td>0.9829</td><td>0.9829</td><td>0.8633</td><td>0.9297</td><td>7.6562</td><td>0.0949</td><td>0.1034</td><td>0.3849</td><td>0.0641</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9812</td><td>0.9863</td><td>0.8438</td><td>0.9180</td><td>8.8125</td><td>0.1039</td><td>0.1251</td><td>0.3876</td><td>0.0523</td></tr><tr><td rowspan="3">21 Layyer</td><td>Baseline</td><td>0</td><td>0.9798</td><td>0.9786</td><td>0.8750</td><td>0.9336</td><td>10.0625</td><td>0.1283</td><td>0.1407</td><td>0.4160</td><td>0.0846</td></tr><tr><td>MAGNITUDE</td><td></td><td>0.9551</td><td>0.9506</td><td>0.7891</td><td>0.8867</td><td>18.7500</td><td>0.1043</td><td></td><td></td><td>0.0144</td></tr><tr><td>WANDA</td><td>50 50</td><td>0.9800</td><td>0.9795</td><td>0.8672</td><td>0.9297</td><td>10.3750</td><td>0.1150</td><td>0.1465 0.1263</td><td>0.4425 0.4694</td><td>0.0528</td></tr><tr><td rowspan="2">22</td><td>SPARSEGPT</td><td>50</td><td>0.9788</td><td>0.9828</td><td>0.8516</td><td>0.9219</td><td>12.1250</td><td>0.1182</td><td>0.1399</td><td>0.4662</td><td>0.0694</td></tr><tr><td>Baseline</td><td>0</td><td>0.9777</td><td>0.9769</td><td>0.8633</td><td>0.9297</td><td>13.9375</td><td>0.1217</td><td>0.1252</td><td>0.4038</td><td>0.0619</td></tr><tr><td rowspan="2">Layye</td><td>MAGNITUDE</td><td>50</td><td>0.9520</td><td>0.9468</td><td>0.7578</td><td>0.8750</td><td>27.2500</td><td>0.0751</td><td>0.0979</td><td>0.4423</td><td>0.0034</td></tr><tr><td>WANDA</td><td>50</td><td>0.9782</td><td>0.9777</td><td>0.8594</td><td>0.9258</td><td>14.3750</td><td>0.1108</td><td>0.1132</td><td>0.3853</td><td>0.0440</td></tr><tr><td rowspan="2">23</td><td>SPARSEGPT</td><td>50</td><td>0.9774</td><td>0.9828</td><td>0.8398</td><td>0.9180</td><td>17.0000</td><td>0.1170</td><td>0.1314</td><td>0.3808</td><td>0.0524</td></tr><tr><td>Baseline</td><td>0</td><td>0.9752</td><td>0.9736</td><td>0.8555</td><td>0.9258</td><td>18.3750</td><td>0.1120</td><td>0.1156</td><td>0.3858</td><td>0.0634</td></tr><tr><td rowspan="2">Layyer</td><td>MAGNITUDE</td><td>50</td><td>0.9541</td><td>0.9468</td><td>0.7422</td><td>0.8672</td><td>37.5000</td><td></td><td></td><td></td><td>0.0098</td></tr><tr><td>WANDA</td><td>50</td><td>0.9757</td><td>0.9743</td><td>0.8516</td><td>0.9219</td><td>18.6250</td><td>0.0672 0.1053</td><td>0.0836 0.1080</td><td>0.4309 0.3862</td><td>0.0496</td></tr><tr><td rowspan="2">24</td><td>SPARSEGPT</td><td>50</td><td>0.9749</td><td>0.9794</td><td>0.8320</td><td>0.9141</td><td>22.3750</td><td>0.1042</td><td>0.1163</td><td>0.3495</td><td>0.0397</td></tr><tr><td>Baseline</td><td>0</td><td>0.9695</td><td>0.9671</td><td>0.8438</td><td>0.9219</td><td>24.8750</td><td>0.1410</td><td>0.1928</td><td>0.3933</td><td>0.0638</td></tr><tr><td rowspan="3">Layyer</td><td>MAGNITUDE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WANDA</td><td>50 50</td><td>0.9528 0.9698</td><td>0.9468 0.9675</td><td>0.7266 0.8398</td><td>0.8633</td><td>55.0000 25.2500</td><td>0.0747</td><td>0.1230</td><td>0.3507 0.4123</td><td>0.0136 0.0453</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9698</td><td>0.9725</td><td>0.8164</td><td>0.9219 0.9102</td><td>30.8750</td><td>0.1184 0.1087</td><td>0.1606 0.1578</td><td>0.3605</td><td>0.0469</td></tr><tr><td rowspan="4">25 Layyer</td><td>Baseline</td><td></td><td></td><td>0.9654</td><td>0.8633</td><td>0.9258</td><td>29.8750</td><td>0.1717</td><td>0.1925</td><td>0.4197</td><td>0.0722</td></tr><tr><td></td><td></td><td>0 0.9662</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MAGNITUDE WANDA</td><td>50 50</td><td>0.9545 0.9679</td><td>0.9734 0.9675</td><td>0.7188 0.8438</td><td>0.8516 0.9141</td><td>68.5000 30.7500</td><td>0.0761 0.1239</td><td>0.1081 0.1463</td><td>0.3124 0.3742</td><td>0.0240 0.0636</td></tr><tr><td>SPARSEGPT</td><td>50</td><td>0.9671</td><td>0.9725</td><td>0.8359</td></tr></table>

Table 4: Per-layer SAEBench metrics for gemma-2-2b at 50% sparsity, layers 0–8 (Early group). The dense baseline is italicised for each layer. ↑ higher is better; ↓ lower is better.

Table 5: Per-layer SAEBench metrics for gemma-2-2b at 50% sparsity, layers 9–17 (Middle group). The dense baseline is italicised for each layer. ↑ higher is better; ↓ lower is better.

Table 6: Per-layer SAEBench metrics for gemma-2-2b at 50% sparsity, layers 18–25 (Late group). The dense baseline is italicised for each layer. ↑ higher is better; ↓ lower is better.

<table><tr><td>Layer Group</td><td>MAGNITUDE  $\varepsilon ^ { 2 }$ </td><td> ${ \mathrm { W A N D A } } \varepsilon ^ { 2 }$ </td><td> $\mathtt { S P A R S E G P T } \varepsilon ^ { 2 }$ </td></tr><tr><td>Early (0–8)</td><td>57.36</td><td>31.87</td><td>26.66</td></tr><tr><td>Middle (9–17)</td><td>47.13</td><td>21.16</td><td>20.70</td></tr><tr><td>Late (18–25)</td><td>53.69</td><td>26.45</td><td>24.95</td></tr></table>

Table 7: Empirical perturbation energy $\varepsilon ^ { 2 }$ by layer group for gemma-2-2b at 50% sparsity. Lower is better; the ordering predicted by Corollary 3.3 holds in every group.