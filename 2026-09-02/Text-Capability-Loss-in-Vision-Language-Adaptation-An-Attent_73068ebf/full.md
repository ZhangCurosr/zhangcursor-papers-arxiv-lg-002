# Text Capability Loss in Vision-Language Adaptation: An Attention-Sink Diagnosis

Minsik Choi<sup>\*</sup> Korea University mszzang2002@korea.ac.kr

Geewook Kim NAVER Cloud AI KAIST AI gwkim.rsrch@gmail.com

Young Geun Kim<sup>†</sup> Korea University younggeun\_kim@korea.ac.kr

## Abstract

Fine-tuning a pretrained LLM into a visionlanguage model (VLM) can erode the backbone’s text capability, with the damage concentrated on tasks that require following exact output rules, such as instruction following, chainof-thought reasoning graded on a strictly parsed final answer, and similar evaluations with strict graders. We trace this gap to attention-sink corruption: VL fine-tuning perturbs the early sink position that anchors a large fraction of attention probability, and how well the base LLM preserves its sink tracks how much of the affected capability survives adaptation. Building on this view, we introduce Sink Strength, a single scalar computed on the base LLM in a few seconds on a single GPU that predicts post-VL degradation without any VL training. It consistently tracks relative degradation across the six VLM–LLM pairs and multiple formatsensitive tasks. Complementing this diagnostic, we find that post-pretraining QK-RMSNORM injection fails to reproduce the protection of native QK-RMSNORM, while several off-theshelf weight-merging settings fail to recover the lost capability after VL training. These negative results underscore the value of screening backbones with Sink Strength before VL training and narrow the intervention space toward head-selective training-time protection.

## 1 Introduction

Vision-language models (VLMs) are commonly built by attaching a vision encoder and projector to a pretrained large language model (LLM) and jointly fine-tuning on multimodal data. We find that this adaptation can erode the LLM’s text capability, with the damage concentrated on tasks we call format-sensitive: those that require following exact output rules, such as instruction following or chain-of-thought reasoning whose final answer is parsed under a strict rule. Across the released VLM–LLM pairs we study, the text-capability gap reaches double digits on multiple such tasks. A text-only further-training counterpart shows substantially smaller degradation, while a separately matched VL-versus-text training trajectory isolates the modality difference (Sections 3.1 and 5). Existing remedies each help in part but at a cost, and none explains why the loss happens. The existing remedies include re-blending text-only data into the multimodal supervised fine-tuning (SFT) mix (Lin et al., 2024; Tong et al., 2024; Tu et al., 2025), freezing the LLM (Zhu et al., 2024), lowrank adapters (LoRA) (Liu et al., 2024; Hu et al., 2022), and post-hoc neuron re-merging (Yu and Ananiadou, 2025).

We trace this loss to attention-sink corruption. A modern LLM concentrates a large fraction of its attention onto a small number of early positions (Xiao et al., 2024; Barbero et al., 2025), and the hidden states at those positions are dominated by a few fixed feature dimensions carrying massive activations (Sun et al., 2024). This sink absorbs diffuse attention mass that would otherwise spread across non-informative positions (Xiao et al., 2024). We hypothesize that this stabilizes per-position attention to the surface tokens that format-sensitive evaluators grade, and the rest of the paper tests that hypothesis. We show that VL fine-tuning perturbs the query and key projections, $W _ { q }$ and $W _ { k }$ , that read those dimensions; because the induced logit shift scales with input magnitude, the perturbation is amplified precisely at the sink, collapsing the perhead sink (Figure 1). How well the base LLM’s sink survives this perturbation predicts how much post-VL damage we see on format-sensitive tasks. Base LLMs whose query and key projections are less sensitive to input magnitude exhibit greater sink preservation under VL adaptation (Section 4). Architectures bound this sensitivity by normalizing the projections, but the granularity matters. Applied per head, the normalization removes the rawmagnitude dependence outright. Applied across a whole layer, it instead ties each head’s scale to that head’s share of the layer’s energy, so a weak head is compressed rather than normalized. Such a backbone reaches VL training with its per-head sink already weak. Predicting post-VL damage therefore requires accounting for corruption inherited from pre-training as well as corruption induced by VL fine-tuning. Preserved-sink and collapsed-sink backbones separate cleanly on instruction following: every backbone that preserves its sink stays within 5.4 pt of its reference LLM, while every backbone whose sink collapses loses at least 7.9 pt.

![](images/041a965a6db351ddf0750337227dd9f981e3957e6a6aa34bf05526bf5ca69c6e.jpg)  
Figure 1: Attention-sink corruption in VL fine-tuning: the base LLM concentrates attention on an early sink position; whether this concentration survives VL finetuning varies sharply across language backbones.

Building on this account, we introduce Sink Strength S (Section 3), a single scalar computed entirely on the base LLM using a small number of inference-only forward passes and no VL training. S quantifies the head-level sink concentration, ranks backbones by their post-VL damage (Spearman $\rho = 0 . 9 7 )$ , and the ranking remains strongly correlated across multiple format-sensitive tasks. Because S is measured before VL training, it exposes base-side sink weakness independently of the subsequent VL update; Section 4 then combines this quantity with the post-VL gap perturbation to separate base-side vulnerability from updateinduced corruption.

Given a high-risk backbone, can simple interventions prevent or recover the damage? We test two complementary negative controls (Section 5): post-pretraining injection of the QK-RMSNORM module into a matched base LLM followed by VL training, and several post-VL weight-merging methods spanning linear, spectral, and sparsified families (Ilharco et al., 2023; Yadav et al., 2023; Yu et al., 2024; Gargiulo et al., 2025; Lee et al., 2025b; Wortsman et al., 2022). The former fails to reproduce the protection associated with native QK-RMSNORM, while the latter fails to recover the lost capability under the settings we test, narrowing the intervention space toward headselective training-time protection.

Our contributions are: (i) Sink Strength S, a base-LLM scalar predicting post-VL damage on format-sensitive tasks from inference-only forward passes (rank correlations remain strong across our six VLM–LLM pairs and four tasks, $\rho \ge 0 . 8 8 )$ providing a pre-VL screen; (ii) a sink-anchored mechanism (Theorem 1) showing how per-head QK-RMSNORM on the query and key projections removes explicit raw-input-magnitude sensitivity from the sink-gap perturbation bound, supported by margin calibration across the five headline pairs and two layerwise-QK-RMSNORM controls; (iii) confound and negative controls: text-only versus VL controls supporting modality-specific degradation, together with two complementary negative controls—post-pretraining QK-RMSNORM injection and post-VL weight-space merging.

## 2 Background & Related Work

VL fine-tuning and text-side erosion. Standard VLM recipes (Liu et al., 2023, 2024) update the LLM decoder weights during adaptation, typically over multiple stages of alignment and instruction tuning. A recurrent observation across this paradigm is that the decoder loses some of the underlying LLM’s text-side capability (Zhang et al., 2024; Zhai et al., 2024; Lee et al., 2025a; Ratzlaff et al., 2025; Srivastava et al., 2026), with instruction-following and other format-sensitive behaviors among the hardest hit. Existing remedies are data-side (text-only re-blending (Lin et al., 2024; Tong et al., 2024; Tu et al., 2025)), trainingside (frozen LLM (Zhu et al., 2024) or low-rank adapter (Hu et al., 2022; Liu et al., 2024)), or posthoc (neuron-level re-merging (Yu and Ananiadou, 2025)), each closing part of the gap. Recent work has also explored weight merging within multimodal instruction-tuning pipelines (Choi and Kim, 2026). Our primary diagnostic analysis leaves the released VLM training recipes untouched and instead asks which base-LLM property predicts how much text capability survives adaptation.

![](images/28c087c3a65aefefe8110a78d36438590a670a6522fd25a26b2c019bbeda8bb8.jpg)  
VLM-LM − reference LLM (pt)  
Figure 2: Text-capability gap across our headline VLM–LLM pairs on instruction-following and three further format-sensitive tasks. Color: Sink Strength S measured on the reference LLM (Section 3).

Attention sinks and mechanistic position. The attention-sink phenomenon (Xiao et al., 2024) has been traced to massive activations (Sun et al., 2024), given an over-mixing rationale (Barbero et al., 2025), linked to embedding-level bookkeeping (Zhang et al., 2025), and its emergence conditions characterized (Gu et al., 2025), with the vision analogue motivating register tokens (Darcet et al., 2024). While these characterize the sink of a fixed network at inference time, we study what happens to it during fine-tuning. VL fine-tuning is known to shift attention patterns within the adapted decoder (Zhang et al., 2024), and we operationalize this shift as corruption of the early-position sink that anchors text-side stability. QK-RMSNORM replaces the L2 normalization of the original query–key normalization (Henry et al., 2020) with RMSNorm, motivated by entropy-collapse stability arguments (Zhai et al., 2023). QK-RMSNORM adoption varies across base LLMs: present in Qwen3 (Yang et al., 2025), absent in Qwen2.5 (Yang et al., 2024b) and Llama-3 (Grattafiori et al., 2024). This architectural axis is examined throughout our analysis. Our bound and saturation lemma are in the spirit of small-circuit attention analyses (Olsson et al., 2022; Geva et al., 2021) and fine-tuning circuit-stability work (Wang et al., 2025b).

## 3 Sink Strength: A Pre-VL Diagnostic

## 3.1 Problem Setup

An LM-only further-training counterpart shows substantially smaller text-capability degradation than the corresponding VL endpoint across the four format-sensitive tasks; Section 5 provides a separately matched VL-versus-text training trajectory from a shared base LLM.

<table><tr><td>Pair</td><td>Reference LLM</td><td>Post-training</td><td>QK</td></tr><tr><td>Qwen3-VL</td><td>Qwen3-8B</td><td>SFT+RL</td><td>per-head</td></tr><tr><td>InternVL3.5</td><td>Qwen3-8B</td><td>SFT+RL (CascadeRL)</td><td>per-head</td></tr><tr><td>Qwen2.5-VL</td><td>Qwen2.5-7B-Instruct SFT+DPO</td><td></td><td>none</td></tr><tr><td>InternVL3</td><td>Qwen2.5-7B-Instruct SFT+MPO</td><td></td><td>none</td></tr><tr><td>LLaVA-OV</td><td>Qwen2-7B-Instruct</td><td>SFT</td><td>none</td></tr><tr><td colspan="4">Non-headline control (Appendix D.2)</td></tr><tr><td>Molmo2-O</td><td>Olmo-3-7B-Instruct</td><td>SFT (PixMo)</td><td>layerwise</td></tr></table>

Table 1: Five headline VLM–LLM pairs and the Molmo2-O control. "QK" reports the per-head / layerwise / none QK-RMSNORM configuration. Per-head QK-RMSNORM normalizes each attention head by its own RMS, while the layerwise variant applies a single RMS across all concatenated heads.

Setup. The five headline pairs come from three model families: Qwen, InternVL, and LLaVA-OneVision. Full per-pair metadata is in Table 1.<sup>1</sup> All released VLMs in the headline panel update the language backbone rather than relying on LoRAonly adaptation, removing the adapter-merge confound. For each released VLM, we extract its language backbone as a standalone causal LM (the VLM-LM hereafter; Appendix B.2 and score it on a nine-task text suite (Appendix B.1) against the corresponding text-only reference LLM, a comparison point rather than an assumed immediate predecessor of multimodal training.

Format-sensitive tasks. We call IFEval, EQ-Bench, GSM8K-CoT, and GPQA-Diamond-CoT format-sensitive because each grades a strict, parseable surface gate: per-prompt format compliance for IFEval (Zhou et al., 2023) (our headline metric), score-block parsing for EQ-Bench, and CoT answer-token extraction for GSM8K-CoT and GPQA-Diamond-CoT. Because these gates are parsed rather than scored semantically, small attention-pattern drift can disproportionately flip the outcome. Concretely, the regressing prompts we inspect often fail parseable rules: length constraints (a 1200-word essay truncated), exact-phrase requirements ("My answer is maybe" reduced to "My answer"), single-language constraints, and format markers. A per-category breakdown across pairs is in Appendix C.2.

<table><tr><td>Pair</td><td>QK</td><td>Both√</td><td></td><td></td><td>Lost Gained Both × Net (%)</td><td></td></tr><tr><td>Qwen3-VL</td><td>per-head</td><td>394</td><td>56</td><td>27</td><td>64</td><td>5.4</td></tr><tr><td>InternVL3.5</td><td>per-head</td><td>407</td><td>43</td><td>33</td><td>58</td><td>1.8</td></tr><tr><td>Qwen2.5-VL</td><td>none</td><td>299</td><td>91</td><td>39</td><td>112</td><td>9.6</td></tr><tr><td>InternVL3</td><td>none</td><td>294</td><td>96</td><td>48</td><td>103</td><td>8.9</td></tr><tr><td>LLaVA-OV</td><td>none</td><td>195</td><td>83</td><td>40</td><td>223</td><td>7.9</td></tr><tr><td>Molmo2-O</td><td>layerwise</td><td>317 126</td><td></td><td>25</td><td>73</td><td>18.7</td></tr></table>

Table 2: Per-prompt regression breakdown on the 541- prompt IFEval set, partitioning prompts by reference-LLM and VLM-LM pass/fail. Net = (Lost − Gained)/541.

Per-pair text-capability gap. Figure 2 (IFEval row) shows a clear VLM–LLM text-capability gap across our five headline pairs: two pairs show gaps of at most 5.4 pt (InternVL3.5 at −1.8, Qwen3- VL at −5.4), while three show gaps of at least 7.9 pt (LLaVA-OV −7.9, InternVL3 −8.9, Qwen2.5- VL −9.6). The non-headline pair (Molmo2-O on Olmo-3) shows a −18.7-pt gap and is analyzed separately as a control (Section 4).

Per-prompt regression. On the 541 IFEval validation prompts each backbone evaluates, we tag every prompt as both\_pass (LLM and VLM-LM both score ≥ 0.5), regression (LLM pass, VLM-LM fail), recovered (LLM fail, VLM-LM pass), or both\_fail (Table 2). The net regression rate reproduces the IFEval delta pair-by-pair. The regressions are spread across 43–126 distinct prompts per pair, so the behavioral split reflects broad instructionfollowing degradation rather than a few outlier prompts.

Modality attribution of the loss. A natural question is whether the observed gap reflects further optimization in general rather than VL training specifically. We first compare two Qwen2.5-family endpoints against the same Qwen2.5-7B-Instruct text-only reference: the text-only Qwen2.5-7B-Instruct-1M variant and Qwen2.5-VL-7B-Instruct. Under the same instruct protocol (0-shot with the chat template applied, Appendix B.1), the text-only variant matches the reference on IFEval promptstrict while the VL endpoint shows a 9.6-pt gap; across the four format-sensitive tasks the text-only variant does not regress on IFEval or GSM8K and loses at most 0.42× the VL gap on EQ-Bench and GPQA (Table 3; details in Appendix F.1). The same comparison also indicates why the VL endpoint perturbs the sink more strongly. Across every layer of the late-layer window, its displacement from the reference is larger in operator norm than that of the text-only endpoint, with a median ratio of 4.44× on the query projection and 3.65× on the key projection. The sink-to-other input contrast is also ≈ 1.12× larger once vision tokens are present. This endpoint contrast supports modality as an important source of the gap rather than further optimization alone, with a separately matched training-time control provided in Section 5. This frames the predictor question: which candidate LLM backbones are most vulnerable to text-side degradation under VL adaptation?

<table><tr><td>Bench</td><td>Text-only ∆</td><td>VL∆</td></tr><tr><td>IFEval prompt-strict</td><td>0.00</td><td>-9.6</td></tr><tr><td>GSM8K-CoT (flex)</td><td>+5.46</td><td>-14.3</td></tr><tr><td>GPQA-Diamond-CoT (flex)</td><td>-4.04</td><td>-9.8</td></tr><tr><td>EQ-Bench v2.1</td><td>-2.24</td><td>-11.2</td></tr></table>

Table 3: Endpoint modality comparison against the same Qwen2.5-7B-Instruct text-only reference: the textonly endpoint is Qwen2.5-7B-Instruct-1M, while the VL endpoint is Qwen2.5-VL-7B-Instruct.

## 3.2 Sink Strength

Sink Strength S, measured on each text-only reference LLM, captures the per-pair capability-gap split, whose clustering further aligns with per-head QK-RMSNORM in the reference LLM.

Definition. We capture per-head sink concentration on the base LLM as a stand-alone metric, the Sink Strength:

$$
S : = \mathrm { m e d i a n } _ { ( \ell , h , q ) } \log \frac { a _ { p _ { \mathrm { s i n k } } } ^ { ( \ell , h , q ) } } { 1 - a _ { p _ { \mathrm { s i n k } } } ^ { ( \ell , h , q ) } } ,
$$

where $a _ { p _ { \mathrm { s i n k } } } ^ { ( \ell , h , q ) }$ is the attention probability at the perhead argmax key position $p _ { \mathrm { s i n k } }$ at layer ℓ, head h, query position $q ,$ and the median is taken over the last ∼ 10 layers (design ablations in Appendix E.4). S is computed entirely on the base LLM (no VLM forward pass, no VL training) with 15 inferenceonly forward passes on calibration prompts disjoint from the IFEval test set. For a 7B backbone, this takes ∼ 8 seconds on a single A6000 GPU. Intuitively, S quantifies how concentrated the base LLM’s attention is on its sink position: high S means a strong, well-formed sink, while $S \le 0$ means the sink does not dominate the aggregate non-sink attention mass. Section 4 formalizes why this scalar captures vulnerability to VL-induced sink corruption via a sink-anchored perturbation bound.

![](images/e996af51d8dbac522b0ffdc0c51900249161ed2d531fb2f19f6e667a311ccb7d.jpg)  
Figure 3: Sink Strength S on the reference LLM versus the VLM–LLM IFEval gap. Color denotes the QK-RMSNORM configuration. The dashed line is a global linear fit for visualization; reported prediction errors use leave-one-pair-out fitting.

Analysis. Color in Figure 2 indicates per-pair S measured on the reference LLM, and the same backbones that lose more on IFEval have smaller S. S separates the six pairs into non-overlapping ranges along both axes: the two lower-loss pairs share $S ~ = ~ + 2 . 3 6$ (sharing the Qwen3-8B reference), the three higher-loss pairs fall in $S \in$ $[ + 1 . 1 8 , + 1 . 4 4 ]$ , and the Molmo2-O boundary case sits at S = +0.24 (Figure 3). The ordering induced by S strongly tracks the IFEval-gap ordering across the six pairs.

From sink collapse to format-sensitive loss. Low S in a candidate base LLM indicates a sink that fails to dominate non-informative positions: a sink collapse waiting to surface under VL finetuning. Format-sensitive tasks such as instruction following, score-block extraction, and CoT answer parsing grade strict, parseable rules over surface tokens that require stable per-position attention; the early-position sink anchors this stability (Xiao et al., 2024; Barbero et al., 2025), so sink collapse breaks surface-token tracking and flips format-checker outcomes. Knowledge tasks, which rely on facts encoded in MLP weights (Geva et al., 2021), do not depend on the same attentionpattern stability and are correspondingly preserved (Appendix C.1). The same holds on a broader capability panel spanning coding, long-context retrieval, and advanced reasoning, where the mean degradation remains substantially smaller than on the format-sensitive tasks (Appendix C.4).

Architectural correlate. The S clusters align with per-head QK-RMSNORM presence in the reference LLM, with finer resolution: perhead QK-RMSNORM pairs occupy the high-S range, no-QK-RMSNORM pairs the mid-S range, and Molmo2-O is a boundary case where QK-RMSNORM is present but applied layerwise. A coarse "has any QK-RMSNORM module" classifier would flag Molmo2-O as safe; S correctly identifies it as the highest-risk backbone among the six pairs, matching the largest measured drop in that set (−18.7 pt on IFEval). Because S is measured on the base LLM alone, this distinction is available before any VL training is committed, framing per-head sink preservation as a pretraining architectural-design concern. Section 4.2 gives the mechanism: per-head normalization removes the raw-magnitude dependence from the perturbation bound, while layerwise normalization leaves each head’s scale tied to its share of the layer’s energy (Remark 1). Section 4.3 confirms this empirically on Molmo2-O. Within pairs sharing the same reference LLM, and hence the same S, the drop still varies by up to 3.5 pt (Qwen3-VL vs. InternVL3.5), which S cannot resolve and we attribute to VL recipe variation.

Empirical validation. We use S in two modes: rank ordering across backbones requires no outcome calibration, while magnitude prediction adds a single 1-D linear fit calibrated on known post-VL outcomes. Across the six pairs of Table 1, S strongly tracks the VLM–LLM IFEval gap (Spearman $\rho = 0 . 9 7 )$ ; under leave-one-pair-out validation, the average held-out prediction error is 2.54 pt on a 17-pt outcome range (best −1.8 pt to worst −18.7 pt), against 4.40 pt for a constant-mean baseline (per-pair detail in Appendix E.1). With six datapoints this is a sanity check on the linear relation rather than prospective evidence on unseen backbones. The one large headline residual is Qwen3- VL (4.4 pt held-out): it shares $S ~ = ~ 2 . 3 6$ with InternVL3.5 but the two differ by 3.5 pt in the observed gap, and no single-feature fit can separate backbones with identical S, which sets the resolution floor of the predictor.

<table><tr><td></td><td>IFEval</td><td>EQ-Bench</td><td>GSM8K</td><td>GPQA-D</td></tr><tr><td>Spearman  $\rho , \mathrm { \boldsymbol { n } } \mathrm { = } 6$ </td><td>0.971</td><td>0.971</td><td>0.971</td><td>0.883</td></tr><tr><td>Error, 6 pairs (pt)</td><td>2.54</td><td>19.09</td><td>7.91</td><td>1.96</td></tr><tr><td>Error, 5 headline pairs (pt)</td><td>1.63</td><td>0.95</td><td>1.03</td><td>1.69</td></tr></table>

Table 4: Sink Strength S as a cross-task predictor on the four format-sensitive tasks (held-out 1-D linear regression). The 5-headline row excludes Molmo2-O, the boundary case where the linear extrapolation breaks down (task-specific Molmo collapse, e.g., EQ-Bench −64.6); rank correlation holds on all six pairs.

## 3.3 Generalization and Practical Use

Generalization across format-sensitive tasks. The same S, measured once on the reference LLM, also ranks the backbones on each of the four formatsensitive tasks (Table 4). Across six pairs, Spearman $\rho \in [ 0 . 8 8 , 0 . 9 7 ]$ on every task. On the five headline pairs (excluding Molmo2-O), the average held-out error stays under 2 pt on each task. S generalizes as a rank predictor across tasks without re-measurement; per-task magnitude prediction additionally requires a per-task linear fit but reuses the same reference-LLM S.

Predictor extension. We extend the predictor along two complementary axes. At fixed IFEval, a 17-pair panel spanning dense and MoE architectures, four reference-LLM families plus OLMoE, and 1.5B–32B scale achieves Spearman $\rho = 0 . 8 8$ (Appendix E.2). On a 9-pair subset evaluated across all four format-sensitive tasks, the rank correlation remains $\rho \ge 0 . 8 2$ (Appendix E.3).

Practical use. Given a candidate base LLM, S can be measured with 15 calibration prompts in seconds on a single GPU and used to rank expected post-VL IFEval damage. In our panels, every pair at $S \geq 2$ stays within 5.4 pt on IFEval, whereas pairs near S ≈ 1.2 span −8.9 to −17.9 pt.

From diagnosis to mechanism. Section 4 explains why a quantity measured before VL training predicts what survives it. There we bound the logitgap perturbation $B _ { \mathrm { g a p } }$ that an update of bounded size can inflict (Theorem 1), and show that the surviving sink mass is governed by the margin

$G _ { \mathrm { b a s e } } - B _ { \mathrm { g a p } }$ (Lemma 1). Since S is the median of $G _ { \mathrm { b a s e } }$ , it measures the base-side term of that margin: a high-S backbone enters VL training with more sink lead than a bounded update can close. The other term, $B _ { \mathrm { g a p } } ,$ , is recipe-dependent and observable only after adaptation, which is why S alone suffices to rank backbones beforehand while the full margin refines the analysis post hoc. In our sample the base-side term alone is in fact the stronger signal, with S tracking the IFEval gap better than the full margin $( \rho = 0 . 9 7 \mathrm { v s } . 0 . 8 9 )$ , so we use S as a leading-order, low-cost proxy for it.

## 4 Mechanism

A sink survives as long as its attention logit stays clearly ahead of the logits at the other key positions. This section makes that lead quantitative and bounds how much of it a fine-tuning update of a given size can close.

## 4.1 Setup and Notation

Setup. Fix a single attention head (hidden size d, head dim $d _ { h } ;$ the layer has H heads), a single layer, and a single query token, with base-LLM hidden states $\{ \xi _ { q } , \xi _ { p } \} _ { p }$ of the surrounding tokens, where $\xi _ { q } = \gamma _ { \mathrm { i n } } \odot \mathrm { R M S N o r m } ( x )$ for inputlayernorm scale $\gamma _ { \mathrm { i n } }$ and RMSNorm denotes rootmean-square normalization. The attention logit at key position p is

$$
\begin{array} { l } { \displaystyle { l _ { p } = \frac { Q \cdot K _ { p } } { \sqrt { d _ { h } } } , } } \\ { \displaystyle { Q = W _ { q } \xi _ { q } , } \quad \quad \kappa _ { p } = W _ { k } \xi _ { p } . } \end{array}\tag{1}
$$

Under QK-RMSNORM each projection is divided by its own RMS before the scale is applied,

$$
Q = \gamma _ { q } \odot \frac { Q _ { \mathrm { p r e } } } { \mathrm { R M S } ( Q _ { \mathrm { p r e } } ) } ,\tag{2}
$$

where $Q _ { \mathrm { p r e } } ~ = ~ W _ { q } \xi _ { q } ,$ and analogously for $K _ { p }$ Here $\| \gamma _ { * } \| _ { \infty } : = \operatorname* { m a x } _ { d } | \gamma _ { * } [ d ] |$ denotes the channelwise $\ell _ { \infty }$ norm of a QK-RMSNORM scale vector. Let $l _ { p } ^ { \prime } = l _ { p } + \delta _ { p }$ be the post-perturbation logit, and let $p _ { \mathrm { s i n k } }$ be the base LLM’s per-head sink position (argmax key on the unperturbed forward pass). Two quantities anchored at $p _ { \mathrm { s i n k } }$ drive the bound,

$$
\begin{array} { r l } & { G _ { \mathrm { b a s e } } : = \log \cfrac { a _ { p _ { \mathrm { s i n k } } } } { 1 - a _ { p _ { \mathrm { s i n k } } } } , } \\ & { B _ { \mathrm { g a p } } : = \underset { p \neq p _ { \mathrm { s i n k } } } { \operatorname* { s u p } } ~ \lvert \delta _ { p } - \delta _ { p _ { \mathrm { s i n k } } } \rvert , } \end{array}\tag{3}
$$

the aggregate sink gap on the unperturbed logits and the sink-anchored gap perturbation over the fixed key positions of the calibration prompt.

## Perturbation model.

Assumption 1 (Perturbation model). At the layer of interest, fine-tuning perturbs only the query and key projections, by $\Delta W _ { \ast }$ with $\| \Delta W _ { * } \| _ { \mathrm { o p } } \leq \varepsilon$ , and leaves the QK-RMSNORM scales $\gamma _ { q } , \gamma _ { k }$ and the input-layernorm scale $\gamma _ { \mathrm { i n } }$ fixed. We hold the base hidden states $\{ \xi _ { q } , \xi _ { p } \}$ at their LLM baseline values, so the bound isolates the effect of the projection drift at that layer.

Both parts have a plain reading. The operator-norm cap ε limits how far fine-tuning can move either projection. Holding the $\gamma \mathbf { \dot { s } }$ fixed is a close approximation for the Qwen3-VL pair, where the relativenorm drift $\| \Delta \gamma \| / \| \gamma \|$ is at most 2.9% across the 36 q-norm and k-norm layers.

Notation summary. We use four sink-related quantities: $G _ { \mathrm { b a s e } }$ (per-head log-odds of sink attention, base LLM), $B _ { \mathrm { g a p } }$ (post-VL gap perturbation, Theorem 1), the margin $G _ { \mathrm { b a s e } } - B _ { \mathrm { g a p } }$ (controls sink survival, Lemma 1), and Sink Strength S (Section 3, median of $G _ { \mathrm { b a s e } }$ over $( \ell , h , q ) )$

## 4.2 Perturbation Bound and Margin Saturation

Assumption 2 (Projection-RMS non-degeneracy). Let $ { \boldsymbol { S } } _ { q }$ be the finite set of query-side inputs $\xi _ { q }$ over the calibration prompts at the layer and head of interest, and analogously $S _ { k }$ for key-side inputs $\xi _ { p }$ over all p. Assume $\| { \xi } \| _ { 2 } ~ > ~ 0$ and $\mathrm { R M S } ( W _ { * } \xi ) ~ > ~ 0$ for every $\xi$ in the relevant set, and define $m _ { q } : =$ min $\xi \in S _ { q }$ $\mathrm { R M S } ( W _ { q } \xi ) / \| \xi \| _ { 2 }$ and $\begin{array} { r } { m _ { k } \ : = \ \operatorname* { m i n } _ { \xi \in { \cal S } _ { k } } \mathrm { R M S } ( \dot { W } _ { k } \xi ) / \| \xi \| _ { 2 } } \end{array}$ Then $m _ { q } , m _ { k } > 0$ in the finite calibration set we measure.

Intuition. The sink margin is threatened when a small $\Delta W$ produces a large logit shift $\delta _ { p }$ , and the three regimes differ in how the input ξ enters. Without QK-RMSNORM, $\delta _ { p } \sim \Delta W \cdot \xi$ carries two input factors. The first is the query magnitude $\| \xi _ { q } \|$ , present at any query position whether or not the sink falls there. The second is the keyside magnitude term, which includes the sink input $\| \xi _ { p _ { \mathrm { s i n k } } } \|$ and can therefore be amplified when the sink carries a massive-activation channel (Sun et al., 2024). Per-head QK-RMSNORM divides $Q$ and $K _ { p }$ by their own RMS, which absorbs $\| \xi \|$ , so the bound depends only on the frozen $\| \gamma \| _ { \infty }$ and the projection-RMS lower bounds, removing the explicit dependence on raw input magnitude. Layerwise QK-RMSNORM also absorbs ∥ξ∥, through a single RMS over $Q _ { \mathrm { c a t } } = [ Q _ { 1 } ; \cdot \cdot \cdot ; Q _ { H } ]$ , but the scale it leaves behind depends on each head’s share of the layer’s energy, so a weak head is compressed rather than normalized.

Theorem 1 (Logit-gap perturbation). Under Assumptions 1 and 2, in the regime where the firstorder Taylor expansion of $\tilde { q } , \tilde { k }$ near $W _ { q } , W _ { k }$ is valid (Appendix $A . I ) ,$ we have, with an absolute constant c and $O ( \varepsilon ^ { 2 } )$ constants depending on the fixed weights $W _ { q } , W _ { k } ,$ , scales $\gamma _ { q } , \gamma _ { k } , \gamma _ { \mathrm { i n } } ,$ calibration inputs $\{ \xi _ { q } , \xi _ { p } \}$ , and RMS lower bounds $m _ { q } , m _ { k }$ , No QK-RMSNORM:

$$
\begin{array} { r l } & { B _ { \mathrm { g a p } } \leq c \varepsilon d _ { h } ^ { - 1 / 2 } \| \xi _ { q } \| \left( \| W _ { q } \| _ { \mathrm { o p } } + \| W _ { k } \| _ { \mathrm { o p } } \right) } \\ & { \qquad \cdot \underset { p \neq p _ { \mathrm { s i n k } } } { \operatorname* { s u p } } ( \| \xi _ { p _ { \mathrm { s i n k } } } \| + \| \xi _ { p } \| ) + O ( \varepsilon ^ { 2 } ) . } \end{array}\tag{4}
$$

With QK-RMSNORM:

$$
\begin{array} { c } { { B _ { \mathrm { g a p } } \leq c \varepsilon \| \gamma _ { q } \| _ { \infty } \| \gamma _ { k } \| _ { \infty } ( m _ { q } ^ { - 1 } + m _ { k } ^ { - 1 } ) } } \\ { { + O ( \varepsilon ^ { 2 } ) . } } \end{array}\tag{5}
$$

The two bounds make this precise, with explicit raw-input-magnitude factors surviving only in the no-QK-RMSNORM case.

Remark 1 (Layerwise QK-RMSNORM extension; semi-formal). Here the H heads share a single RMS denominator over the concatenated $H \cdot d _ { h } .$ dimensional $Q _ { \mathrm { { c a t } } }$ , with the layer’s $\gamma _ { q }$ applied afterwards. Extracting head $h$ from $\tilde { Q } _ { \mathrm { c a t } } = \gamma _ { q } \odot$ $Q _ { \mathrm { c a t } } / \mathrm { R M S } ( Q _ { \mathrm { c a t } } )$ gives

$$
\lVert \tilde { Q } _ { h } \rVert \leq \lVert \gamma _ { q } ^ { ( h ) } \rVert _ { \infty } \lVert Q _ { h } \rVert \frac { \sqrt { H d _ { h } } } { \lVert Q _ { \mathrm { c a t } } \rVert } ,\tag{6}
$$

which replaces the head-energy-independent envelope $\| \gamma _ { q } ^ { ( h ) } \| _ { \infty } \sqrt { d _ { h } }$ that per-head normalization provides; the same applies to keys. Writing $\rho _ { h } : =$ $\| Q _ { h } \| / \| Q _ { \mathrm { c a t } } \|$ for the head’s share of layer energy, the two agree when $\rho _ { h } = 1 / \sqrt { H }$ , while a head carrying less than its share is compressed and its attention logits, including the sink’s lead, are attenuated with it. The margin can therefore fail from the $G _ { \mathrm { b a s e } }$ side rather than through a larger $B _ { \mathrm { g a p } } ,$ and Molmo2-O is the empirical analogue with the weakest per-head $G _ { \mathrm { b a s e } }$ among the calibrated pairs of Table 5 (full derivation in Appendix A.5; empirical control in Appendix D.2).

Lemma 1 (Saturation transfer). For sink position $p _ { \mathrm { s i n k } }$ with pre-perturbation aggregate gap $G _ { \mathrm { b a s e } }$ and gap perturbation $B _ { \mathrm { g a p } }$

$$
1 - a _ { p _ { \mathrm { s i n k } } } ^ { \prime } \ \leq \ \sigma \big ( B _ { \mathrm { g a p } } - G _ { \mathrm { b a s e } } \big ) ,\tag{7}
$$

where $\sigma ( x ) = e ^ { x } / ( 1 + e ^ { x } )$ is the logistic function.

The margin $G _ { \mathrm { b a s e } } \ : - \ : B _ { \mathrm { g a p } }$ controls an upper bound on the non-sink mass through $\sigma (  { B _ { \mathrm { g a p } } }$ $G _ { \mathrm { b a s e } } )$ . Preserving the sink at level $1 - \eta$ requires $G _ { \mathrm { b a s e } } - B _ { \mathrm { g a p } } \geq \log ( ( 1 - \eta ) / \eta )$ , and negative margins are consistent with the observed sink collapse.

## 4.3 Empirical Sink Behavior

We calibrate the margin quantities on the seven pairs, measure the cross-pair sink-mass distribution, and use the layerwise controls to distinguish aggregate from per-head sink.

Calibration across the seven pairs. We compute $( G _ { \mathrm { b a s e } } , B _ { \mathrm { g a p } } )$ across all five headline pairs and the two layerwise controls on a 15-prompt calibration set, aggregating over (layer, head, query position) in the last ∼10 layers where sink behavior is established beyond the first few local-context layers (Xiao et al., 2024). $G _ { \mathrm { b a s e } }$ is computed on the reference LLM at the per-head argmax key $p _ { \mathrm { s i n k } } .$ , and $B _ { \mathrm { g a p } }$ is reconstructed from the change in attention-probability ratios between the reference LLM and the VLM-LM at the same p<sub>sink</sub>. The empirical $B _ { \mathrm { g a p } }$ is the realized reference-to-VLM-LM gap change and thus includes hiddenstate and other parameter drift; it enters the margin of Lemma 1 but is not a direct numerical evaluation of Theorem 1’s projection-only bound. For released pairs, the reference LLM is a comparison point rather than necessarily the immediate pre-VL checkpoint. Within each Qwen generation, the two pairs share the same reference LLM and therefore the same $G _ { \mathrm { b a s e } } ;$ ; per-pair values are reported in Table 5. The marginal proxy median(G) − median(B) is near zero on both perhead QK-RMSNORM pairs and strongly negative on all three no-QK-RMSNORM pairs. Both layerwise QK-RMSNORM pairs land at the negative end of the sample, with Molmo2-O driven by the weakest base sink in the set. The ordering tracks the behavioral split of Figure 2. Calibration samplesize sensitivity is reported in Appendix D.1.

Cross-pair sink and outcome bucketing. We partition each VLM-LM’s IFEval prompts by perprompt strict accuracy (pass at $\geq 0 . 5 )$ . For each bucket we measure position-0 attention, averaging over late-layer × head × trailing-query positions with bootstrap 95% CIs over 100 prompts per outcome bucket (Appendix C.3). The withinpair pass−fail gap is small $( \le ~ 0 . 0 2$ per-prompt mean), but the cross-pair span is large. The two per-head QK-RMSNORM pairs concentrate $\geq 0 . 7 0$ of attention on position 0, while the no-QK-RMSNORM pairs span $5 \times 1 0 ^ { - 5 }$ to 0.50 and Molmo2-O (layerwise) sits at 0.25. Within the same OpenGVLab InternVL lineage, InternVL3 (no-QK-RMSNORM, $5 \times 1 0 ^ { - 5 } )$ and InternVL3.5 (per-head QK-RMSNORM, 0.74) span four orders of magnitude on this measure; the Qwen-VL family shows the same direction at smaller magnitude $( 0 . 0 3  0 . 7 0$ across Qwen2.5-VL → Qwen3-VL). Within-vendor framing reduces vendor-level confounding and strengthens the association with the architectural change, although recipe differences remain. The within-pair near-equality is consistent with the broken-sink state being a model-level head configuration rather than a prompt-level outcome marker. The cross-pair span is what tracks the behavioral split, so effective interventions should operate at the head level.

<table><tr><td>Pair</td><td>QK</td><td> $G _ { \mathrm { b a s e } }$ </td><td> $B _ { \mathrm { g a p } }$ </td><td>G-B</td></tr><tr><td>Qwen3-VL</td><td>per-head</td><td>+2.36</td><td>+2.06</td><td>+0.31</td></tr><tr><td>InternVL3.5</td><td>per-head</td><td>+2.36</td><td> $+ 2 . 4 7$ </td><td>-0.11</td></tr><tr><td>Qwen2.5-VL</td><td>none</td><td>+1.18</td><td>+5.01</td><td>-3.83</td></tr><tr><td>InternVL3</td><td>none</td><td> $+ 1 . 1 8$ </td><td>+3.40</td><td>-2.22</td></tr><tr><td> $\mathrm { L L a V A } .$  -OV</td><td>none</td><td>+1.44</td><td>+2.53</td><td>-1.09</td></tr><tr><td>Molmo2-O</td><td>layerwise</td><td>+0.24</td><td>+3.25</td><td>-3.01</td></tr><tr><td>MolmoE-1B</td><td>layerwise</td><td>+1.34</td><td>+4.19</td><td>-2.84</td></tr></table>

Table 5: Per-pair $G _ { \mathrm { b a s e } } ,$ $B _ { \mathrm { g a p } } .$ , and the marginal proxy $G \mathrm { ~ - ~ } B : = $ median $\left( G _ { \mathrm { b a s e } } \right) -$ median $\left( B _ { \mathrm { g a p } } \right)$ of the Lemma 1 margin, on a 15-prompt calibration set.

Per-head vs. aggregate sink. Molmo2-O (Olmo-3 base, layerwise QK-RMSNORM; Remark 1) separates aggregate from per-head sink: aggregate position-0 attention (0.25) is of the same order as LLaVA-OV’s (0.50) yet IFEval damage is far worse, while per-head sink is the weakest among the calibrated pairs and already weak in the Olmo-3 reference LLM (Appendix D.2). The marginal proxy $G - B$ is strongly negative (−3.01, second only to the no-QK-RMSNORM Qwen2.5-VL at −3.83; Table 5), so Lemma 1’s per-head margin protection is not delivered head-by-head. MolmoE-1B, a second layerwise backbone on an MoE base, shows the same pattern, so both layerwise pairs carry strongly negative margins despite differing in base family and architecture. Together, the evidence supports per-head QK-RMSNORM as the protective configuration in our sample, with the layerwise variant insufficient.

## 4.4 Channel and Direction Ablations

Ablating the sink amplifier. For each of Qwen3- VL’s normalization-scale tensors $( \gamma _ { \mathrm { i n } } , \gamma _ { q } , \gamma _ { k } )$ , replacing the top-10 channels per layer by the layerwise mean disables the amplifier without touching the residual stream or projections (Appendix D.3): joint kill costs 44 pt on IFEval against 20 pt as the sum of singles, a 24-pt super-additive interaction that supports sink amplification as a redundant multi-norm system contributing to format-sensitive capability. For reference, a norm-matched isotropic random perturbation causes a ∼ 6× larger IFEval drop than the Qwen2.5-VL gap (Appendix D.4). Thus, weight-space distance alone does not explain the degradation; perturbation direction matters.

## 5 Controls and Discussion

Within-vendor comparison. The same direction appears within each vendor’s own model line: Qwen3-VL vs. Qwen2.5-VL, and InternVL3.5 vs. InternVL3. In both comparisons the QK-RMSNORM backbone loses less text capability, so the association does not require comparing models built by different teams under different VL pipelines. However, this is not a controlled comparison, since successive releases differ in much more than QK-RMSNORM, and we report it as one alternative explanation ruled out rather than as evidence that QK-RMSNORM is the cause; isolating the architectural axis requires the ablation we leave to future work (See Limitations).

Modality control (VL vs. Tülu trajectory). The Section 3.1 comparison contrasts released 7B endpoints against a common text-only reference. We additionally run a matched 3B trajectory, comparing vision fine-tuning with text-only Tülu SFT from the same Qwen2.5-3B-Instruct base (Lambert et al., 2025). The vision run ends with a weaker sink (S : +0.41 → +0.20) as SEEDBench rises 60.1 → 64.8, whereas the text-only run holds S at +0.58–+0.65 and loses only ∼5 IFEval points from the shared base (Appendix F.2). Because S is a base-model property, it predicts the regime a run settles into rather than the step at which capability is lost. IFEval takes most of its loss by step 1k, before S declines, and then partially recovers while S keeps falling.

Post-pretraining QK-RMSNORM injection (negative). A natural question is whether the protection associated with native per-head

QK-RMSNORM can be reproduced by injecting the module after LLM pretraining. We add perhead QK-RMSNORM with unit-scale initialization to a clean Qwen2.5-3B-Instruct and then run the matched two-stage VL recipe (Appendix F.3). The injected variant lags vanilla throughout training, scoring only 11 on IFEval at the first evaluated checkpoint against the 59.9 base. Combined with the Molmo2-O control (Section 4), this supports native-pretraining co-adaptation of the projections, rather than the module’s mere presence, as an important component of the protective regime.

Weight-space merging (negative). Merging the VLM’s language backbone back toward its reference LLM is the natural post-hoc repair: seven off-the-shelf merging settings spanning linear taskvector mixes (Ilharco et al., 2023), spectral decompositions (Gargiulo et al., 2025; Lee et al., 2025b), and sparsified merges (Yadav et al., 2023; Yu et al., 2024) on Qwen2.5-VL and Qwen3-VL fail to recover IFEval on the low-S side: only the two mildest mixes avoid substantial additional degradation (Appendix F.4).

What the negative results do and do not show. Our matched Tülu trajectory provides a mechanistic rationale for the data-side remedy: text-only optimization preserves the sink whereas VL optimization disrupts it. That remedy is applied uniformly and paid for in multimodal data budget, and it is chosen after the backbone is already fixed; S instead identifies which backbones need it before training begins. The two negative controls argue against simpler shortcuts: post-pretraining QK-RMSNORM injection does not reproduce native protection, and post-VL merging does not recover the lost capability under our settings. Both leave the intervention space narrower than the remedy literature suggests.

## 6 Conclusion

Across five headline VLM–LLM pairs from three model families, format-sensitive text-capability gaps track Sink Strength of the corresponding reference LLM, with per-head QK-RMSNORM as the strongest architectural correlate in our sample. The relation extends from the six-pair diagnostic set to a 17-pair panel spanning MoE and multiple LLM families. Negative injection and merging controls further motivate pre-VL screening and head-selective training-time protection.

## Limitations

To our knowledge, the multimodal literature has not previously characterized the phenomenon we surface, in which VL adaptation erodes the base LLM’s per-head attention sink and carries a downstream cost to format-sensitive capability. Within our sample, per-head QK-RMSNORM covaries with vendor, recipe, and pretraining data. Within-vendor comparisons (Qwen2.5→Qwen3, InternVL3→InternVL3.5) reduce but do not isolate the architectural axis. We validate S against VL adaptation outcomes, so whether it also predicts instruction-following drift under adaptation regimes we did not sample, such as continual pretraining, remains open and is a natural next test of the diagnostic’s reach.

The post-VL merging methods we test do not recover the lost capability, while post-pretraining QK-RMSNORM injection fails to reproduce the native protective regime, leaving head-selective training-time intervention as an open direction (Section 5). The bound of Section 4 suggests one concrete candidate: freeze the layers below the first sink layer and project the sink-head $W _ { q } , W _ { k }$ updates away from the sink direction, which should slow how fast VL training closes the $G _ { \mathrm { b a s e } } \mathrm { ~ - ~ }$ $B _ { \mathrm { g a p } }$ margin. A direct causal intervention would separately test sufficiency, for instance disabling per-head QK-RMSNORM during VL training of Qwen3 to see whether IFEval collapses. We leave both to future work.

## Acknowledgments

This work was supported by National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-24534857 and RS-2026-25522655), Institute of Information & Communications Technology Planning & Evaluation(IITP)-ICT Creative Consilience Program grant funded by the Korea government (MSIT) (IITP-2026-RS-2020-II201819, 10%), IITP-ITRC(Information Technology Research Center) grant funded by the Korea government (MSIT) (IITP-2026-RS-2023-00260091, 10%), and IITP grant funded by the Korea government (MSIT) (RS-2024-00398353, Development of Countermeasure Technologies for Generative AI Security Threats). We also thank the members of the Korea University Intelligent Computer Architecture & Systems research Lab, Vision Understanding Team at NAVER Cloud AI, and the anonymous reviewers for their

helpful feedback.

## References

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. 2021. Program Synthesis with Large Language Models. arXiv preprint arXiv:2108.07732.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025a. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025b. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.

Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Velickoviˇ c, and Razvan Pascanu. 2025. Why do´ LLMs attend to the first token? In Second Conference on Language Modeling.

Minsik Choi and Geewook Kim. 2026. Decentralized Instruction Tuning: Conflict-Aware Splitting and Weight Merging. In International Conference on Machine Learning.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions. In Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936.

Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Rohun Tripathi, Sangho Lee, Mohammadreza Salehi, Jason Ren, Chris Dongjoo Kim, Yinuo Yang, et al. 2026. Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28652–28668.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168.

Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. 2024. Vision Transformers Need Registers. In International Conference on Learning Representations.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza

Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. 2025. Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 91–104.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and 5 others. 2024. The Language Model Evaluation Harness.

Antonio Andrea Gargiulo, Donato Crisostomi, Maria Sofia Bucarelli, Simone Scardapane, Fabrizio Silvestri, and Emanuele Rodolà. 2025. Task Singular Vectors: Reducing Task Interference in Model Merging. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18695–18705.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. 2021. Transformer Feed-Forward Layers Are Key-Value Memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5484–5495.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.

Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, and Min Lin. 2025. When Attention Sink Emerges in Language Models: An Empirical View. In International Conference on Learning Representations.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring Massive Multitask Language Understanding. In International Conference on Learning Representations.

Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. 2020. Query-Key Normalization for Transformers. In Findings ofthe Association for Computational Linguistics: EMNLP 2020, pages 4246–4253.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. RULER: What’s the Real Context Size of Your Long-Context Language Models? In First Conference on Language Modeling.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. 2023. Editing models with task arithmetic. In International Conference on Learning Representations.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7B. Preprint, arXiv:2310.06825.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Xinxi Lyu, et al. 2025. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. In Second Conference on Language Modeling.

Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. 2024. Building and better understanding vision-language models: insights and future directions. In NeurIPS 2024 Workshops: RBFM.

Seongyun Lee, Geewook Kim, Jiyeon Kim, Hyunji Lee, Hoyeon Chang, Sue Hyun Park, and Minjoon Seo. 2025a. How Does Vision-Language Adaptation Impact the Safety of Vision Language Models? In International Conference on Learning Representations.

Yu-Ang Lee, Ching-Yun Ko, Tejaswini Pedapati, I-Hsin Chung, Mi-Yen Yeh, and Pin-Yu Chen. 2025b. STAR: Spectral Truncation and Rescale for Model Merging. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 496–505.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2025. LLaVA-OneVision: Easy Visual Task Transfer. Transactions on Machine Learning Research.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023. SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension. arXiv preprint arXiv:2307.16125.

Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. 2024. VILA: On Pre-training for Visual Language Models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26689– 26699.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved Baselines with Visual Instruction Tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26296–26306.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual Instruction Tuning. Advances in

Neural Information Processing Systems, 36:34892– 34916.

Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Han-Jia Ye. 2024. Ovis: Structural Embedding Alignment for Multimodal Large Language Model. arXiv preprint arXiv:2405.20797.

Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, et al. 2025. OLMoE: Open Mixture-of-Experts Language Models. In International Conference on Learning Representations.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. 2022. In-context Learning and Induction Heads. arXiv preprint arXiv:2209.11895.

Samuel J. Paech. 2023. EQ-Bench: An Emotional Intelligence Benchmark for Large Language Models. arXiv preprint arXiv:2312.06281.

Neale Ratzlaff, Man Luo, Xin Su, Vasudev Lal, and Phillip Howard. 2025. Training-Free Mitigation of Language Reasoning Degradation After Multimodal Instruction Tuning. In Proceedings of the AAAI Symposium Series, volume 5, pages 384–388.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. 2024. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. In First Conference on Language Modeling.

Shikhar Srivastava, Md Yousuf Harun, Robik Singh Shrestha, and Christopher Kanan. 2026. Improving Multimodal Large Language Models Using Continual Learning. In Proceedings ofThe 4th Conference on Lifelong Learning Agents, volume 330 of Proceedings of Machine Learning Research, pages 736–755. PMLR.

Mingjie Sun, Xinlei Chen, J. Zico Kolter, and Zhuang Liu. 2024. Massive Activations in Large Language Models. In First Conference on Language Modeling.

Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, et al. 2025. Olmo 3. arXiv preprint arXiv:2512.13961.

Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. 2024. Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs. Advances in Neural Information Processing Systems, 37:87310– 87356.

Jianhong Tu, Zhuohao Ni, Nicholas Crispino, Zihao Yu, Michael Bendersky, Beliz Gunel, Ruoxi Jia, Xin Liu, Lingjuan Lyu, Dawn Song, and Chenguang Wang. 2025. MLAN: Language-Based Instruction Tuning Preserves and Transfers Knowledge in Multimodal Language Models. In Proceedings ofthe 3rd Workshop on Towards Knowledgeable Foundation Models (KnowFM), pages 59–74.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. 2025a. InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency. arXiv preprint arXiv:2508.18265.

Xu Wang, Yan Hu, Wenyu Du, Reynold Cheng, Benyou Wang, and Difan Zou. 2025b. Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 63088–63112. PMLR.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. 2024. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. Advances in Neural Information Processing Systems, 37:95266–95290.

Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. 2022. Model soups: averaging weights of multiple finetuned models improves accuracy without increasing inference time. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 23965–23998. PMLR.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Efficient Streaming Language Models with Attention Sinks. In International Conference on Learning Representations.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. 2023. TIES-Merging: Resolving Interference When Merging Models. Advances in Neural Information Processing Systems, 36:7093– 7115.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. 2024a. Qwen2 Technical Report. arXiv preprint arXiv:2407.10671.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,

Fei Huang, Haoran Wei, et al. 2024b. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115.

Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. 2024. Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 57755–57775. PMLR.

Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang Huang, Yuanqian Zhao, et al. 2025. MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipe. arXiv preprint arXiv:2509.18154.

Zeping Yu and Sophia Ananiadou. 2025. Locate-then-Merge: Neuron-Level Parameter Fusion for Mitigating Catastrophic Forgetting in Multimodal LLMs. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pages 7065–7078.

Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu, and Joshua M. Susskind. 2023. Stabilizing Transformer Training by Preventing Attention Entropy Collapse. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 40770–40803. PMLR.

Yuexiang Zhai, Shengbang Tong, Xiao Li, Mu Cai, Qing Qu, Yong Jae Lee, and Yi Ma. 2024. Investigating the Catastrophic Forgetting in Multimodal Large Language Model Fine-Tuning. In Conference on Parsimony and Learning, volume 234 of Proceedings of Machine Learning Research, pages 202–227. PMLR.

Stephen Zhang, Mustafa Khan, and Vardan Papyan. 2025. Attention Sinks: A ’Catch, Tag, Release Mechanism for Embeddings. Advances in Neural Information Processing Systems, 38:83140–83181.

Yi-Kai Zhang, Shiyin Lu, Yang Li, Yanqing Ma, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, De-Chuan Zhan, and Han-Jia Ye. 2024. Wings: Learning Multimodal LLMs without Text-only Forgetting. Advances in Neural Information Processing Systems, 37:31828–31853.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-Following Evaluation for Large Language Models. arXiv preprint arXiv:2311.07911.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2024. MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models. In International Conference on Learning Representations.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan,

Weijie Su, Jie Shao, et al. 2025. InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models. arXiv preprint arXiv:2504.10479.

## Table of Contents

A. Theory and Proofs . . p. 15   
A.1. Single-Head Perturbation Bound (Theorem 1) p. 15   
A.2. Aggregate-Gap Saturation Transfer (Lemma 1) p. 16   
A.3. Frobenius / Stable-Rank Restatement p. 17   
A.4. Multi-Layer Composition Remark p. 17   
A.5. Why Layerwise QK-RMSNorm Does Not Provide the Same Per-Head Guarantee p. 18   
A.6. Numerical Sanity Checks p. 18   
B. Evaluation Setup p. 20   
B.1. Evaluation Protocols p. 20   
B.2. Backbone Extraction p. 20   
B.3. Implementation Details and Reproducibility p. 20   
B.4. Models and Datasets . p. 20   
C. Per-Task and Behavioral Evidence p. 22   
C.1. Capability Preservation vs. Format-Sensitive Degradation p. 22   
C.2. Failure-Category Breakdown p. 23   
C.3. Outcome-Bucketed Sink Probe p. 23   
C.4. Extended Capability Panel p. 23   
D. Mechanism Details p. 24   
D.1. Calibration Sample-Size Robustness p. 24   
D.2. Layerwise Backbones: Aggregate vs. Per-Head Sink . p. 25   
D.3. Channel Ablation Sweep p. 26   
D.4. Random-∆W Control p. 26   
E. Sink Strength Predictor . p. 26   
E.1. Per-Pair Sink Strength Predictions p. 26   
E.2. IFEval-Specific Predictor (17 pairs) p. 26   
E.3. Multi-Task Generalization (9 pairs) p. 26   
E.4. Sink Strength Design Ablations p. 26   
F. Controls p. 27   
F.1. Modality Comparison: Text-Only Endpoint . p. 27   
F.2. Modality Control: VL vs. Text-Only Trajectory p. 27   
F.3. Injection Experiment Training Recipe p. 29   
F.4. Weight-Merging Sweep on Format-Sensitive Benches p. 29

## A Theory and Proofs

Notation as in Section 4. We work with a single attention head, hidden size $d ,$ head dim $d _ { h }$ . The pre-norm inputs are $\xi _ { q } , \xi _ { p } = \gamma _ { \mathrm { i n } } \odot \mathrm { R M S N o r m } ( x _ { q } ) , \gamma _ { \mathrm { i n } } \odot$ RMSNorm $( x _ { p } ) \in \mathbb { R } ^ { d }$ at the query and keyposition-p tokens, and the pre-QK-RMSNORM projections are $Q _ { \mathrm { p r e } } = W _ { q } \xi _ { q } , K _ { \mathrm { p r e } , p } = W _ { k } \xi _ { p }$ with $W _ { q } , W _ { k } \in \mathbb { R } ^ { d _ { h } \times d }$ . Under QK-RMSNORM we set $\tilde { q } = Q _ { \mathrm { p r e } } / \mathrm { R M S } ( Q _ { \mathrm { p r e } } )$ and $Q = \gamma _ { q } \odot \tilde { q }$ (similarly $K _ { p } )$ For clarity, the main text omits RoPE from the notation. Here, let $R _ { q }$ and $R _ { p }$ denote the orthogonal RoPE transformations at the query and key positions. Orthogonality preserves the norm of each rotated vector, while differences across positions are bounded by the triangle inequality; this yields the sum-of-norms term in Equation 4. The sink-anchored gap perturbation is $\begin{array} { r } { B _ { \mathrm { g a p } } : = \operatorname* { s u p } _ { p \neq p _ { \mathrm { s i n k } } } | \delta _ { p } - \delta _ { p _ { \mathrm { s i n k } } } | } \end{array}$ , and the aggregate sink gap is $\begin{array} { r } { G : = l _ { p _ { \mathrm { s i n k } } } - \log \sum _ { p \neq p _ { \mathrm { s i n k } } } e ^ { l _ { p } } } \end{array}$

## A.1 Single-Head Perturbation Bound (Theorem 1)

ProofofTheorem 1. Let $\Delta W _ { q } , \Delta W _ { k }$ be the projection perturbations with $\Vert \Delta W _ { * } \Vert _ { \mathrm { o p } } \ \leq \ \varepsilon$ . The two cases are treated separately because the dependence of $Q , K _ { p }$ on the projections is linear without QK-RMSNORM and nonlinear (through the RMS denominator) with QK-RMSNORM: the no-QK-RMSNORM case is exactly linear up to the explicit second-order cross term, whereas the QK-RMSNORM case requires a local Taylor expansion through the RMS denominator.

No QK-RMSNORM. Let $R _ { q }$ and $R _ { p }$ denote the orthogonal RoPE transformations at the query position and key position $p ,$ respectively. The attention logit is

$$
\ell _ { p } = \frac { ( R _ { q } W _ { q } \xi _ { q } ) ^ { \top } ( R _ { p } W _ { k } \xi _ { p } ) } { \sqrt { d _ { h } } } .
$$

Since the projections are linear, the first-order perturbation is

$$
\delta _ { p } = \frac { 1 } { \sqrt { d _ { h } } } \Big [ ( R _ { q } \Delta W _ { q } \xi _ { q } ) ^ { \top } ( R _ { p } W _ { k } \xi _ { p } ) + ( R _ { q } W _ { q } \xi _ { q } ) ^ { \top } ( R _ { p } \Delta W _ { k } \xi _ { p } ) \Big ] + O ( \varepsilon ^ { 2 } ) .
$$

Let $s = p _ { \mathrm { s i n k } }$ . The sink-anchored gap perturbation between s and any $p \neq s$ is

$$
\begin{array} { c } { \displaystyle \delta _ { p } - \delta _ { s } = \frac { 1 } { \sqrt { d _ { h } } } \Big [ \big ( R _ { q } \Delta W _ { q } \xi _ { q } \big ) ^ { \top } \big ( R _ { p } W _ { k } \xi _ { p } - R _ { s } W _ { k } \xi _ { s } \big ) } \\ { \displaystyle + \big ( R _ { q } W _ { q } \xi _ { q } \big ) ^ { \top } \big ( R _ { p } \Delta W _ { k } \xi _ { p } - R _ { s } \Delta W _ { k } \xi _ { s } \big ) \Big ] + O ( \varepsilon ^ { 2 } ) . } \end{array}\tag{8}
$$

Because each $R _ { p }$ is orthogonal,

$$
\begin{array} { r l } & { \quad \quad \| R _ { p } W _ { k } \xi _ { p } - R _ { s } W _ { k } \xi _ { s } \| \leq \| W _ { k } \| _ { \mathrm { o p } } \big ( \| \xi _ { p } \| + \| \xi _ { s } \| \big ) , } \\ & { \quad \| R _ { p } \Delta W _ { k } \xi _ { p } - R _ { s } \Delta W _ { k } \xi _ { s } \| \leq \varepsilon \big ( \| \xi _ { p } \| + \| \xi _ { s } \| \big ) . } \end{array}\tag{9}
$$

(10)

Also,

$$
\lVert R _ { q } \Delta W _ { q } \xi _ { q } \rVert \le \varepsilon \lVert \xi _ { q } \rVert , \qquad \lVert R _ { q } W _ { q } \xi _ { q } \rVert \le \lVert W _ { q } \rVert _ { \mathrm { o p } } \lVert \xi _ { q } \rVert .
$$

Applying Cauchy–Schwarz therefore gives

$$
| \delta _ { p } - \delta _ { s } | \leq \frac { \varepsilon } { \sqrt { d _ { h } } } \| \xi _ { q } \| \big ( \| W _ { q } \| _ { \mathrm { o p } } + \| W _ { k } \| _ { \mathrm { o p } } \big ) \big ( \| \xi _ { s } \| + \| \xi _ { p } \| \big ) + O ( \varepsilon ^ { 2 } ) .
$$

Taking the supremum over $p \neq s$ and absorbing fixed numerical factors into the absolute constant c yields Equation 4.

With QK-RMSNORM. The logit is $l _ { p } = ( \gamma _ { q } \odot \tilde { q } ) \cdot ( \gamma _ { k } \odot \tilde { k } _ { p } ) / \sqrt { d _ { h } }$ where $\tilde { q } = Q _ { \mathrm { p r e } } / \mathrm { R M S } ( Q _ { \mathrm { p r e } } )$ and analogously ${ \tilde { k } } _ { p } .$ . The map from $W _ { q } , W _ { k }$ to $\tilde { q } , \tilde { k } _ { p }$ is differentiable on the open set where ${ \mathrm { R M S } } ( Q _ { \mathrm { p r e } } )$ $\mathrm { R M S } ( K _ { \mathrm { p r e } , p } ) > 0 .$ , which Assumption 2 ensures. We work with the mathematical RMS here, and production implementations add a small stabilizer $\tau > 0$ inside the square root (distinct from the perturbation magnitude ε). Since $\mathrm { R M S } _ { \tau } \geq \mathrm { R M S }$ , the unstabilized bound used below is conservative. To first order in ε on a small neighborhood the logit perturbation decomposes into a Q-side and K-side term:

$$
\begin{array} { r } { \delta _ { p } = \frac { 1 } { \sqrt { d _ { h } } } \big [ \big ( \gamma _ { q } \odot \Delta \tilde { q } \big ) \cdot \big ( \gamma _ { k } \odot \tilde { k } _ { p } \big ) + \big ( \gamma _ { q } \odot \tilde { q } \big ) \cdot \big ( \gamma _ { k } \odot \Delta \tilde { k } _ { p } \big ) \big ] + O ( \varepsilon ^ { 2 } ) . } \end{array}
$$

We bound each term separately. For the Q-side perturbation,

$$
\Delta \tilde { q } = \frac { \Delta W _ { q } \xi _ { q } } { \mathrm { R M S } ( Q _ { \mathrm { p r e } } ) } - \frac { Q _ { \mathrm { p r e } } \Delta \mathrm { R M S } } { \mathrm { R M S } ^ { 2 } ( Q _ { \mathrm { p r e } } ) } ,
$$

with $| \Delta \mathrm { R M S } | \le \| \Delta W _ { q } \| _ { \mathrm { o p } } \| \xi _ { q } \| / \sqrt { d _ { h } }$ . Both terms are $O ( \varepsilon \| \xi _ { q } \| / \mathrm { R M S } ( Q _ { \mathrm { p r e } } ) )$ , and by Assumption 2 $\mathrm { R M S } ( Q _ { \mathrm { p r e } } ) \geq m _ { q } \lvert \lvert \xi _ { q } \rvert \rvert$ on the calibration distribution, so

$$
\lVert \Delta \tilde { q } \rVert \leq \frac { c _ { 1 } \varepsilon } { m _ { q } } ,
$$

with $c _ { 1 }$ an absolute constant. The crucial point is that $\| \xi _ { q } \|$ has cancelled, since numerator and denominator both scale linearly with $\| \xi _ { q } \|$

For the sink-anchored gap perturbation we then bound the Q-side contribution after RoPE. Since each $R _ { p }$ is orthogonal and mathematical unit-RMS gives $\| \widetilde { k } _ { p } \| _ { 2 } = \sqrt { d _ { h } }$ (with the stabilizer τ, $\| \tilde { k } _ { p } \| _ { 2 } \leq \sqrt { d _ { h } } )$ we have

$$
\begin{array} { r } { \left\| { R _ { p _ { \mathrm { s i n k } } } \big ( \gamma _ { k } \odot \tilde { k } _ { p _ { \mathrm { s i n k } } } \big ) - R _ { p } \big ( \gamma _ { k } \odot \tilde { k } _ { p } \big ) } \right\| \le 2 \| \gamma _ { k } \| _ { \infty } \sqrt { d _ { h } } . } \end{array}
$$

Therefore,

$$
\begin{array} { r l } & { \left| R _ { q } ( \gamma _ { q } \odot \Delta \tilde { q } ) \cdot \left[ R _ { p _ { \mathrm { s i n k } } } ( \gamma _ { k } \odot \tilde { k } _ { p _ { \mathrm { s i n k } } } ) - R _ { p } ( \gamma _ { k } \odot \tilde { k } _ { p } ) \right] \right| } \\ & { \leq \| \gamma _ { q } \| _ { \infty } \| \Delta \tilde { q } \| \cdot 2 \| \gamma _ { k } \| _ { \infty } \sqrt { d _ { h } } } \\ & { \leq \frac { 2 c _ { 1 } \varepsilon \| \gamma _ { q } \| _ { \infty } \| \gamma _ { k } \| _ { \infty } \sqrt { d _ { h } } } { m _ { q } } . } \end{array}
$$

A symmetric argument for the K-side gives the $1 / m _ { k }$ term. Summing the two sides, dividing by the outer $\sqrt { d _ { h } }$ in the logit definition, and taking the supremum over $p \neq p _ { \mathrm { s i n k } }$ yields

$$
\begin{array} { r } { B _ { \mathrm { g a p } } \leq c \varepsilon \| \gamma _ { q } \| _ { \infty } \| \gamma _ { k } \| _ { \infty } \big ( \frac { 1 } { m _ { q } } + \frac { 1 } { m _ { k } } \big ) + O ( \varepsilon ^ { 2 } ) , } \end{array}
$$

which is Equation 5.

Input-magnitude cancellation. The $\| \xi _ { q } \|$ that the no-QK-RMSNORM bound carries is cancelled in the QK-RMSNORM pathway by the RMS denominator. What replaces it is the projection-RMS lower bound $m _ { q } .$ , and if a calibration prompt puts $\xi _ { q }$ in or near the null space of $W _ { q } , m _ { q }$ degrades and the bound loosens. Assumption 2 excludes exact degeneracy; near-null cases appear as small $m _ { q }$ or $m _ { k }$ and therefore loosen the bound, consistent with the empirical $m _ { q } , m _ { k } > 0$ we measure on the calibration set.

## A.2 Aggregate-Gap Saturation Transfer (Lemma 1)

Proof of Lemma 1. Write the perturbed softmax probability at position p as $a _ { p } ^ { \prime } = e ^ { l _ { p } ^ { \prime } } / Z ^ { \prime }$ with $l _ { p } ^ { \prime } = l _ { p } + \delta _ { p }$ and $\begin{array} { r } { Z ^ { \prime } = \sum _ { q } e ^ { l _ { q } ^ { \prime } } } \end{array}$ . Then for $p \neq p _ { \mathrm { s i n k } }$

$$
\frac { a _ { p } ^ { \prime } } { a _ { p _ { \mathrm { s i n k } } } ^ { \prime } } = \exp ( l _ { p } - l _ { p _ { \mathrm { s i n k } } } + \delta _ { p } - \delta _ { p _ { \mathrm { s i n k } } } ) .
$$

Summing over $p \neq p _ { \mathrm { s i n k } }$

$$
\begin{array} { r l } {  { \frac { 1 - a _ { p _ { \mathrm { s i n k } } } ^ { \prime } } { a _ { p _ { \mathrm { s i n k } } } ^ { \prime } } = \sum _ { p \neq p _ { \mathrm { s i n k } } } \exp \bigl ( l _ { p } - l _ { p _ { \mathrm { s i n k } } } \bigr ) \exp \bigl ( \delta _ { p } - \delta _ { p _ { \mathrm { s i n k } } } \bigr ) } } \\ & { \leq \exp \bigl ( B _ { \mathrm { g a p } } \bigr ) \sum _ { p \neq p _ { \mathrm { s i n k } } } \exp \bigl ( l _ { p } - l _ { p _ { \mathrm { s i n k } } } \bigr ) } \\ & { = \exp \bigl ( B _ { \mathrm { g a p } } - G \bigr ) , } \end{array}
$$

using the definition $\begin{array} { r } { G = l _ { p _ { \mathrm { s i n k } } } - \log \sum _ { p \neq p _ { \mathrm { s i n k } } } e ^ { l _ { p } } } \end{array}$ and $| \delta _ { p } - \delta _ { p _ { \mathrm { s i n k } } } | \leq B _ { \mathrm { g a p } }$ for all $p \neq p _ { \mathrm { s i n k } }$ . Let $u : = 1 - a _ { p _ { \mathrm { s i n k } } } ^ { \prime }$ and write the inequality as ${ \hat { u } } / ( 1 - u ) \leq e ^ { B _ { \mathrm { g a p } } - G }$ . Solving for u gives $u \leq e ^ { B _ { \mathrm { g a p } } - G } / ( 1 +$ $e ^ { B _ { \mathrm { g a p } } - G } ) \stackrel { \cdot } { = } \sigma ( B _ { \mathrm { g a p } } - G )$ , which is the logistic form of Equation 7. The weaker exponential bound $u \leq e ^ { B _ { \mathrm { g a p } } - G }$ follows from $\sigma ( x ) \leq e ^ { x }$ □

Comparison to top-2 gap arguments. If one tried to state the same lemma using the top-2 gap $T = l _ { p _ { \mathrm { s i n k } } } - \operatorname* { m a x } _ { p \neq p _ { \mathrm { s i n k } } } l _ { p }$ instead of the aggregate G, one would need an extra log N term to control the sum $\sum _ { p \neq p _ { \mathrm { s i n k } } } e ^ { l _ { p } }$ , giving $1 - a _ { p _ { \mathrm { s i n k } } } ^ { \prime } \leq N \exp ( - T + B _ { \mathrm { g a p } } )$ or equivalently exp $( - T + B _ { \mathrm { g a p } } + \log N )$ which is vacuous unless $T > \dot { \log { N } } + B _ { \mathrm { g a p } }$ . The aggregate-gap form sidesteps this and matches the measured quantity directly.

## A.3 Frobenius / Stable-Rank Restatement of Theorem 1

The bound is stated in terms of $\| \Delta W _ { q } \| _ { \mathrm { o p } }$ and $\| \Delta W _ { k } \| _ { \mathrm { o p } }$ . Practitioners often report the Frobenius norms $\| \Delta W _ { q } \| _ { F } , \| \Delta W _ { k } \| _ { F }$ instead, and the conversion is via per-projection stable ranks.

The restatement. Let $r _ { q } : = \| \Delta W _ { q } \| _ { F } ^ { 2 } / \sigma _ { \operatorname* { m a x } } ^ { 2 } ( \Delta W _ { q } )$ and analogously $r _ { k }$ . By definition $\| \Delta W _ { * } \| _ { \mathrm { o p } } =$ $\sigma _ { \mathrm { m a x } } ( \Delta W _ { * } ) = \| \Delta W _ { * } \| _ { F } / \sqrt { r _ { * } }$ . Keeping the two operator norms separate in the proof before upperbounding both by a common $\varepsilon ,$ , and substituting $\lVert \Delta W _ { * } \rVert _ { \mathrm { o p } } = \lVert \Delta W _ { * } \rVert _ { F } / \sqrt { r _ { * } }$ , gives the Frobenius/stablerank restatement

$$
B _ { \mathrm { g a p } } \ \leq \ c \| \gamma _ { q } \| _ { \infty } \| \gamma _ { k } \| _ { \infty } \left( \frac { \| \Delta W _ { q } \| _ { F } } { \sqrt { r _ { q } } m _ { q } } + \frac { \| \Delta W _ { k } \| _ { F } } { \sqrt { r _ { k } } m _ { k } } \right) + O ( \varepsilon ^ { 2 } ) ,\tag{11}
$$

with empirical inputs $\left\{ \boldsymbol { r } _ { q } , \boldsymbol { r } _ { k } , \left\| \Delta W _ { q } \right\| _ { F } , \left\| \Delta W _ { k } \right\| _ { F } , m _ { q } , m _ { k } , \left\| \gamma _ { q } \right\| _ { \infty } , \left\| \gamma _ { k } \right\| _ { \infty } \right\}$ . This is a rewriting using stable ranks, not a tightening. If one perturbation is zero, its corresponding term is taken to be zero; otherwise the expression applies with the stable rank defined above.

Role of the restatement. The Frobenius form makes the structural observation explicit: the QK-RMSNORM side has no explicit raw-input-magnitude factor, only the stable ranks $r _ { q } , r _ { k }$ and the γ, m constants. The number that enters the main text is the directly measured $B _ { \mathrm { g a p } }$ on the calibration set (Section 4), and the measured value on Qwen3 (2.06 nats) is about 2.4× smaller than on Qwen2.5 (5.01 nats) on the same 15-prompt calibration set (Table 5). The Qwen3-versus-Qwen2.5 comparison is consistent with the theorem’s structural prediction that the QK-RMSNORM side is less sensitive to raw input magnitude, but we do not claim it as a causal isolation of QK-RMSNORM alone, since Qwen3 and Qwen2.5 differ in pre-training data, RL stage, and tokenizer in addition to QK-RMSNORM.

## A.4 Multi-Layer Composition Remark

Theorem 1 and Lemma 1 are single-layer statements. A formal multi-layer composition requires additional control of the value vectors, residual stream, MLP, and post-attention LayerNorm, which we do not provide here. We instead state an informal chaining intuition. If at every layer ℓ the per-layer base aggregate gap $G _ { \mathrm { b a s e } , \ell }$ exceeds the per-layer gap perturbation by a margin $G _ { \mathrm { b a s e , } \ell } - B _ { \mathrm { g a p , } \ell } \geq \mu ,$ , then by Lemma 1 the perturbed non-sink mass at that layer is at most $e ^ { - \mu }$ , and the next layer is approximately fed a hidden state close to the LLM baseline under additional Lipschitz assumptions on the intervening operators. We do not formalize these assumptions and report this remark only as motivation for the per-layer diagnostic.

Caveat. In our calibration measurements, the marginal proxy is negative in the top late layers of Qwen2.5 (Section 4). A tight multi-layer theorem (in the spirit of (Olsson et al., 2022)’s circuit composition) is future work.

## A.5 Why Layerwise QK-RMSNorm Does Not Provide the Same Per-Head Guarantee

This subsection derives the claim of Remark 1, that a shared RMS denominator does not provide the same per-head margin guarantee as per-head QK-RMSNORM.

Per-head normalization. Under per-head QK-RMSNORM, head $h$ is divided by its own RMS, $\mathrm { R M S } ( Q _ { h } ) = \lVert Q _ { h } \rVert / \sqrt { d _ { h } } \mathrm { . }$ , so

$$
\Vert \tilde { Q } _ { h } \Vert = \Big \Vert \gamma _ { q } ^ { ( h ) } \odot \frac { Q _ { h } } { \mathrm { R M S } ( Q _ { h } ) } \Big \Vert \leq \Vert \gamma _ { q } ^ { ( h ) } \Vert _ { \infty } \sqrt { d _ { h } } .\tag{12}
$$

The input magnitude $\| Q _ { h } \|$ cancels, which is exactly why the QK-RMSNORM branch of Theorem 1 carries no ∥ξ∥ factor.

Layerwise normalization. Under layerwise QK-RMSNORM, all H heads share one denominator over the concatenated vector $Q _ { \mathrm { c a t } } = [ Q _ { 1 } ; \cdot \cdot \cdot ; Q _ { H } ]$ , so $\mathrm { R M S } ( Q _ { \mathrm { c a t } } ) = \lVert Q _ { \mathrm { c a t } } \rVert / \sqrt { H d _ { h } }$ . Extracting the h-th block of $\tilde { Q } _ { \mathrm { c a t } } = \gamma _ { q } \odot Q _ { \mathrm { c a t } } / \mathrm { R M S } ( Q _ { \mathrm { c a t } } )$ gives

$$
\| \tilde { Q } _ { h } \| \leq \| \gamma _ { q } ^ { ( h ) } \| _ { \infty } \| Q _ { h } \| \frac { \sqrt { H d _ { h } } } { \| Q _ { \mathrm { c a t } } \| } ,\tag{13}
$$

which replaces the per-head envelope of Equation 12. The same derivation applies to keys.

What the shared denominator costs. Write $\rho _ { h } : = \| Q _ { h } \| / \| Q _ { \mathrm { c a t } } \|$ for the share of layer energy carried by head h, so that Equation 13 reads $\| \gamma _ { q } ^ { ( h ) } \| _ { \infty } \rho _ { h } \sqrt { H d _ { h } }$ . Both normalizations remove the input magnitude, since $\rho _ { h }$ is unchanged when $\xi _ { q }$ is rescaled. What per-head normalization additionally provides is a scale independent of the head energy, namely $\sqrt { d _ { h } }$ up to the learned factor $\| \gamma _ { q } ^ { ( h ) } \| _ { \infty }$ . The layerwise scale instead additionally tracks $\rho _ { h }$ . Under equal per-head energy $\rho _ { h } = 1 / \sqrt { H }$ and Equation 13 collapses to Equation 12, but a head carrying less than its share has $\rho _ { h } < 1 / \sqrt { H }$ and is compressed relative to its neighbours.

Consequence for the margin. Compression attenuates that head’s attention logits, and with them the sink’s logit lead, so $G _ { \mathrm { b a s e } }$ can be small at heads that carry little of the layer’s energy. The margin $G _ { \mathrm { b a s e } } - B _ { \mathrm { g a p } }$ can then fail from the base side rather than through a larger perturbation, which is the opposite of the no-QK-RMSNORM failure mode where $\| \xi \|$ enters the bound on $B _ { \mathrm { g a p } }$ directly. Aggregate sink mass is a sum over heads and can stay high while individual compressed heads carry almost no lead, so preservation of the aggregate does not imply preservation of the per-head margin. Appendix D.2 reports the empirical counterpart, where Molmo2-O has the lowest per-head $G _ { \mathrm { b a s e } }$ among the calibrated pairs at an unremarkable $B _ { \mathrm { g a p } }$ , while the no-QK-RMSNORM Qwen2.5-VL shows the reverse profile.

## A.6 Numerical Sanity Checks

We verify two pieces on small synthetic cases.

Theorem 1, $d = 1 6 , d _ { h } = 4 ,$ no QK-RMSNORM. We consider a RoPE-free synthetic case with Gaussian-initialized $W _ { q } , W _ { k }$ . We instantiate a sequence of $N = 8$ key positions with i.i.d. unit-norm Gaussian backgrounds, then form $\xi _ { p _ { \mathrm { s i n k } } }$ by adding a massive-activation-style channel at index 3 with value 10 on top of its background, while $\xi _ { p }$ for $p \neq p _ { \mathrm { s i n k } }$ retain only the unit-norm background. The sink hidden state thus has a sink/non-sink magnitude gap dominated by the size-10 channel rather than being constrained to unit norm. We draw Gaussian $\Delta W _ { \ast }$ with $\| \Delta W _ { * } \| _ { \mathrm { o p } } = 0 . 1$ . The measured sink-anchored quantity is $\begin{array} { r } { \operatorname* { s u p } _ { p \neq p _ { \mathrm { s i n k } } } | \delta _ { p } - \delta _ { p _ { \mathrm { s i n k } } } | = 0 . 1 6 9 } \end{array}$ . Because this synthetic case omits RoPE, the proof admits the sharper specialization with $\| \xi _ { p _ { \mathrm { s i n k } } } - \xi _ { p } \|$ before the triangle-inequality relaxation used in Equation 4. This specialized bound is 0.197, giving a ratio of 0.86. Since $\| \xi _ { p _ { \mathrm { s i n k } } } - \xi _ { p } \| \le \| \xi _ { p _ { \mathrm { s i n k } } } \| + \| \xi _ { p } \|$ , the more general RoPE-aware bound in Equation 4 is also satisfied.

Lemma 1, $G = 3 . 0 , B _ { \mathrm { g a p } } = 0 . 4 , N = 6 4 ,$ . Direct softmax computation with $l _ { p _ { \mathrm { s i n k } } } = 5$ and the $N - 1$ non-sink positions equally distributed to give $G = 3 . 0 \AA$ . To realize the worst-case gap perturbation, we set $\delta _ { p } = + B _ { \mathrm { { g a p } } } / 2$ for every $p \neq p _ { \mathrm { s i n k } }$ and $\delta _ { p _ { \mathrm { s i n k } } } = - B _ { \mathrm { g a p } } / 2$ , so that sup $_ { p \neq p _ { \mathrm { s i n k } } } \ : | \delta _ { p } - \delta _ { p _ { \mathrm { s i n k } } } | = B _ { \mathrm { g a p } }$ is attained. Empirical $1 - a _ { p _ { \mathrm { s i n k } } } ^ { \prime } = 0 . 0 6 9$ . The logistic bound from Equation 7 is $\sigma ( 0 . 4 - 3 . 0 ) = \sigma ( - 2 . 6 ) \approx$ 0.069. Ratio 1.00 (the bound is tight under the worst-case perturbation).

## B Evaluation Setup

This appendix collects evaluation infrastructure used across the paper: the harness protocols (Appendix B.1), the backbone-extraction pipeline (Appendix B.2), and hardware/software details for reproducibility (Appendix B.3).

## B.1 Evaluation Protocols

We evaluate all text-side capability with lm-evaluation-harness v0.4.12 (Gao et al., 2024) under two protocols whose disagreement on a single backbone is itself diagnostic.

Instruct protocol. –apply\_chat\_template, –num\_fewshot 0, –batch\_size 8, bf16. This matches the intended chat-style usage of instruction-tuned models.

Community-default protocol. The harness’s per-task default few-shot count (mmlu=5, gsm8k=8, etc.) with the chat template disabled. This provides a common leaderboard-style comparison.

Nine-task suite. MMLU, MMLU-Pro, BoolQ, MBPP (code, pass@1), RULER (long-context), GSM8K-CoT, IFEval (prompt\_level\_strict\_acc), GPQA-Diamond-CoT, and EQ-Bench (raw points). IFEval is the headline metric for sink corruption because its checker grades surface format directly (Zhou et al., 2023).

Pipeline sanity. Our LLM-side numbers reproduce the Qwen blog values on Qwen2.5-7B-Instruct GSM8K-CoT to within 0.93 pt (90.67 vs. 91.6) and MMLU-Pro to within 0.46 pt (55.84 vs. 56.30) when run under matched OpenCompass configurations. The deltas reported throughout this paper compare a reference LLM to its VLM-LM under the same protocol, reducing sensitivity to framework-specific evaluation differences.

## B.2 Backbone Extraction

To evaluate text-only behavior of a VLM with lm-evaluation-harness, we extract its language backbone as a standalone causal LM. For Qwen2.5- VL and the LLaVA-OneVision / LLaVA-Llama3 families, we remap the language-backbone parameters to the corresponding <Family>ForCausalLM namespace while preserving the VLM’s original input embeddings and output language-model head, including the original lm\_head.weight. For

Qwen3-VL, the same procedure additionally preserves the q\_norm / k\_norm scale parameters used in per-head normalization. For our QK-RMSNorminjected Qwen2.5-3B variant (Section 5), we use a self-contained trust\_remote\_code module so that the extracted backbone follows the same numerical pathway as during training. All extracted backbones are saved to safetensors, and the round-trip reproduces the original language module’s logits on a held-out text-only probe.

## B.3 Implementation Details and Reproducibility

Hardware. Model training (the Section 5 QK-RMSNORM injection experiment) ran on four NVIDIA H100 GPUs (80 GB), with two GPUs allocated to each variant. All text-side evaluation (lm-evaluation-harness across the nine-task suite for all 6 pairs and the text-only endpoint control, with MMLU-Pro and RULER measured on the five headline pairs and Molmo2-O excluded for those two), the Sink Strength calibration forward passes (Section 3, ∼ 8 s per 7B backbone), and the $G _ { \mathrm { b a s e } } , B _ { \mathrm { g a p } }$ measurement on the 15-prompt calibration set (∼ 52 s/pair, full pipeline including VLM-LM forward and $m _ { q } / m _ { k }$ compute) ran on a single NVIDIA A6000 (48 GB) in bf16.

Software. PyTorch 2.10+cu128, transformers 4.57.6, DeepSpeed 0.19.0, lm-evaluation-harness 0.4.12, VLMEvalKit.

## B.4 Models and Datasets

This subsection collects Hugging Face URLs and paper references for every model and dataset used in the paper.

Vision-language models (headline). The five headline VLMs and the Molmo2-O control (Section 3.1):

• Qwen3-VL-8B-Instruct (Bai et al., 2025a): https://huggingface.co/Qwen/ Qwen3-VL-8B-Instruct

• InternVL3.5-8B (Wang et al., 2025a): https://huggingface.co/OpenGVLab/ InternVL3\_5-8B

• Qwen2.5-VL-7B-Instruct (Bai et al., 2025b): https://huggingface.co/Qwen/Qwen2. 5-VL-7B-Instruct

• InternVL3-8B (Zhu et al., 2025): https:// huggingface.co/OpenGVLab/InternVL3-8B • LLaVA-OneVision-Qwen2-7B-OV (Li et al., 2025): https://huggingface.co/lmms-lab/ llava-onevision-qwen2-7b-ov

• Molmo2-O-7B (Clark et al., 2026): https:// huggingface.co/allenai/Molmo2-O-7B

Vision-language models (extended 17-pair panel). The eleven additional VLMs in the IFEval-specific 17-pair panel (Appendix E.2):

• Qwen3-VL-4B-Instruct (Bai et al., 2025a): https://huggingface.co/Qwen/ Qwen3-VL-4B-Instruct

• Qwen3-VL-2B-Instruct (Bai et al., 2025a): https://huggingface.co/Qwen/ Qwen3-VL-2B-Instruct

• Qwen3-VL-30B-A3B-Instruct (Bai et al., 2025a): https://huggingface.co/Qwen/ Qwen3-VL-30B-A3B-Instruct

• Ovis2-8B (Lu et al., 2024): https:// huggingface.co/AIDC-AI/Ovis2-8B

• Ovis2-34B (Lu et al., 2024): https:// huggingface.co/AIDC-AI/Ovis2-34B

• MolmoE-1B-0924 (Deitke et al., 2025): https://huggingface.co/allenai/ MolmoE-1B-0924

• InternVL3-2B (Zhu et al., 2025): https:// huggingface.co/OpenGVLab/InternVL3-2B

• InternVL3-14B (Zhu et al., 2025): https:// huggingface.co/OpenGVLab/InternVL3-14B

• MiniCPM-V-4.5 (Yu et al., 2025): https:// huggingface.co/openbmb/MiniCPM-V-4\_5

• LLaVA-NeXT-Mistral-7B (Liu et al., 2024): https://huggingface.co/llava-hf/ llava-v1.6-mistral-7b-hf

• Idefics3-8B-Llama3 (Laurençon et al., 2024): https://huggingface.co/HuggingFaceM4/ Idefics3-8B-Llama3

Text-only reference LLMs. The text-only reference models evaluated alongside the extracted VLM-LMs:

• Qwen3-8B (Yang et al., 2025): https:// huggingface.co/Qwen/Qwen3-8B

• Qwen3-4B (Yang et al., 2025): https:// huggingface.co/Qwen/Qwen3-4B

• Qwen3-1.7B (Yang et al., 2025): https:// huggingface.co/Qwen/Qwen3-1.7B

• Qwen3-30B-A3B-Instruct-2507 (Yang et al., 2025): https://huggingface.co/Qwen/ Qwen3-30B-A3B-Instruct-2507

• Qwen2.5-7B-Instruct (Yang et al., 2024b): https://huggingface.co/Qwen/Qwen2. 5-7B-Instruct

• Qwen2.5-1.5B-Instruct (Yang et al., 2024b): https://huggingface.co/Qwen/Qwen2.5-1. 5B-Instruct

• Qwen2.5-14B-Instruct (Yang et al., 2024b): https://huggingface.co/Qwen/Qwen2. 5-14B-Instruct

• Qwen2.5-32B-Instruct (Yang et al., 2024b): https://huggingface.co/Qwen/Qwen2. 5-32B-Instruct

• Qwen2-7B-Instruct (Yang et al., 2024a): https://huggingface.co/Qwen/ Qwen2-7B-Instruct

• Qwen2.5-3B-Instruct (Yang et al., 2024b): https://huggingface.co/Qwen/Qwen2. 5-3B-Instruct

• Olmo-3-7B-Instruct (Team Olmo et al., 2025): https://huggingface.co/allenai/ Olmo-3-7B-Instruct

• OLMoE-1B-7B-0924-Instruct (Muennighoff et al., 2025): https://huggingface.co/ allenai/OLMoE-1B-7B-0924-Instruct

• Mistral-7B-Instruct-v0.2 (Jiang et al., 2023): https://huggingface.co/mistralai/ Mistral-7B-Instruct-v0.2

• Llama-3.1-8B-Instruct (Grattafiori et al., 2024): https://huggingface.co/meta-llama/ Llama-3.1-8B-Instruct

Vision tower and projector donor. The injection experiment of Appendix F.3 takes its vision tower and projector from Qwen2.5-VL-3B-Instruct (Bai et al., 2025b), https://huggingface.co/Qwen/Qwen2. 5-VL-3B-Instruct.

Modality-control variant. The text-only further-training endpoint used in the 7B modality comparison (Appendix F.1): Qwen2.5-7B-Instruct-1M, https://huggingface.co/Qwen/Qwen2. 5-7B-Instruct-1M.

Text evaluation datasets. Four format-sensitive tasks and five capability tasks reported in the paper, all run via lm-evaluation-harness (Gao et al., 2024) (Appendix B.1):

• IFEval (Zhou et al., 2023): https:// huggingface.co/datasets/google/IFEval

• EQ-Bench v2.1 (Paech, 2023): https:// github.com/EQ-bench/EQ-Bench

• GSM8K (Cobbe et al., 2021): https:// huggingface.co/datasets/openai/gsm8k

• GPQA-Diamond (Rein et al., 2024): https://huggingface.co/datasets/ Idavidrein/gpqa

• MMLU (Hendrycks et al., 2021): https:// huggingface.co/datasets/cais/mmlu

• BoolQ (Clark et al., 2019): https: //huggingface.co/datasets/google/boolq

• MMLU-Pro (Wang et al., 2024): https: //huggingface.co/datasets/TIGER-Lab/ MMLU-Pro

• MBPP (Austin et al., 2021): https://huggingface.co/datasets/ google-research-datasets/mbpp

• RULER (Hsieh et al., 2024): https://github. com/NVIDIA/RULER

Calibration prompts. The 15-prompt set used for Sink Strength (Section 3) and the $G _ { \mathrm { b a s e } } , B _ { \mathrm { g a p } }$ measurement (Section 4) consists of short instruction-following prompts we wrote in the style of IFEval. No IFEval instance is used, so the calibration set is disjoint from the data we evaluate on by construction.

Licenses and terms of use. All Hugging Face checkpoints and evaluation datasets listed above are used under their published licenses, predominantly Apache 2.0 and MIT, with a small number under model-specific licenses (the Llama 3.1 Community License for Llama-3.1-8B-Instruct, and the research-only Qwen Research License for Qwen2.5-3B-Instruct and Qwen2.5-VL-3B-Instruct, which we use for research purposes only). We download each artifact under its stated terms and do not redistribute the underlying weights or data. The evaluation harness lm-evaluation-harness (Gao et al., 2024) is MIT-licensed.

## C Per-Task and Behavioral Evidence

This appendix expands the per-task behavioral split surfaced in Section 3.1: capability vs. formatsensitive coverage (Appendix C.1), IFEval failurecategory counts (Appendix C.2), an outcomebucketed sink probe (Appendix C.3), and an extended capability panel on coding, reasoning, and long-context retrieval (Appendix C.4).

## C.1 Capability Preservation vs. Format-Sensitive Degradation

Section 3.1 shows that several non-format-sensitive capabilities are comparatively preserved while format-following degrades. We verify this here on the knowledge- and comprehension-oriented benchmarks surfaced in the main text, and Appendix C.4 extends the check to coding, long-context retrieval, and advanced reasoning.

Knowledge and comprehension. Table 6 reports the per-pair ∆ on MMLU and BoolQ alongside the four format-sensitive anchors under the same instruct protocol. MMLU and BoolQ drops never exceed 2.3 pt across the six pairs, and on three pairs they even improve. On the same backbones the headline IFEval gaps reach −9.6 pt (no-QK-RMSNORM) and −18.7 pt (layerwise).

<table><tr><td rowspan="2">Pair</td><td rowspan="2">Knowledge</td><td colspan="2">Format-sensitive</td></tr><tr><td>MMLU BoolQ IFEval</td><td>EQ GSM8K GPQA</td></tr><tr><td>Qwen3-VL InternVL3.5</td><td>+1.6 +6.0 +3.0 +7.6</td><td>-5.4 -1.8</td><td>-2.4 -2.1 -1.9 -0.8 -1.5 -0.6</td></tr><tr><td>Qwen2.5-VL InternVL3</td><td>-2.1 -1.2 +3.0 +4.5</td><td>-9.6 -8.9</td><td>-11.2 -14.3 -9.8 -10.3 -12.5 -7.1</td></tr><tr><td>LLaVA-OV</td><td>-2.3 +0.1</td><td>-7.9</td><td>-8.6 -11.9 -8.4</td></tr><tr><td>Molmo2-O</td><td>-0.0 -1.5</td><td>-18.7</td><td>-64.6 -42.7 -19.7</td></tr><tr><td>Mean</td><td>+0.5 +2.6</td><td>-8.7</td><td>-16.3 -14.2 -7.9</td></tr></table>

Table 6: Per-pair ∆ on knowledge- and comprehensionoriented benchmarks (MMLU, BoolQ) vs. formatsensitive benchmarks (shaded). Same instruct protocol throughout, except that the Molmo2-O GPQA baseline is the published Olmo-3-7B-Instruct score rather than our own matched run, which would give −10.6. Pairlevel QK category in Table 1. EQ is EQ-Bench v2.1 (raw points); GSM8K is GSM8K-CoT flexible-extract; GPQA is GPQA-Diamond-CoT flexible-extract.

Ruling out a format confound. A sceptical reader could attribute the contrast to MCQAvs-open-generation format rather than capability: MMLU and BoolQ preserve because they are loglikelihood-graded letter-picks, and the formatsensitive tasks degrade because they are open generation. Two facts in our data rule out this simple reading. First, GPQA-Diamond-CoT is

MCQA in format (the answer is one of A/B/C/D, scored by flexible-extract on a free-text rationale), yet it drops by −7.1 to −9.8 pt on the three no-QK-RMSNORM pairs, almost the full IFEval-scale magnitude. Second, GSM8K-CoT is open generation with a multi-step reasoning trajectory, yet it stays within ±2.1 pt on the two perhead QK-RMSNORM pairs. Preservation therefore tracks the capability being probed (knowledge/comprehension vs. instruction-following / reasoning-trajectory) and the QK protection of the reference LLM, not the task’s MCQA-vsgeneration format.

## C.2 Failure-Category Breakdown

For each VLM-LM we partition the IFEval set into the four buckets reported in Table 2 (both\_pass, regression, recovered, both\_fail). The regression bucket (prompts the LLM passes but the VLM-LM fails) is the locus of the per-pair degradation. Table 7 reports the dominant failing IFEval instruction categories per pair (counts of failed instructions within the regression bucket). "Punctuation" is largely no\_comma, "combination" is largely repeat\_prompt or two\_responses, and "length\_constraints" is largely number\_words or number\_paragraphs. "Language" is the response\_language constraint, "startend" covers quotation and end\_checker, and "detectable\_format" covers multiple\_sections and number\_bullet\_lists. "Change\_case" covers english\_capital and english\_lowercase, and "keywords" covers existence and letter\_frequency.

## C.3 Outcome-Bucketed Sink Probe

For each VLM-LM we partition the IFEval prompts by prompt\_level\_strict\_acc into a pass bucket (≥ 0.5) and a fail bucket (< 0.5), then run a single forward pass per prompt with output\_attentions=True and aggregate position-0 attention probability over (late layer × head × last-8 query positions). We report the perprompt mean (so the bootstrap is over prompts, not flattened head/layer observations) and a 95% percentile CI from bootstrap over 100 prompts sampled per outcome bucket (200 per pair).

## C.4 Extended Capability Panel

Beyond the knowledge/comprehension benchmarks and the four format-sensitive tasks, the ninetask suite of Appendix B.1 covers coding (MBPP, execution pass@1), advanced multi-domain reasoning (MMLU-Pro), and long-context retrieval (RULER). Table 8 reports these three on the five headline pairs; Molmo2-O, the layerwise control, is not included in this panel.

<table><tr><td colspan="2"></td><td colspan="2"></td><td colspan="2"><img src="images/f63406ea255ed8fa9858fcd9e0fd11f6d977dde2c9cf2ddff35d5d814186c09d.jpg"/></td><td></td><td></td></tr><tr><td>length_constraints</td><td>9</td><td>8</td><td>17</td><td>16</td><td>7</td><td>20</td></tr><tr><td>detectable_format</td><td>8</td><td>5</td><td>10</td><td>8</td><td>20</td><td>30</td></tr><tr><td>keywords</td><td>9</td><td>6</td><td>13</td><td>18</td><td>4</td><td>19</td></tr><tr><td>change_case</td><td>5</td><td>8</td><td>11</td><td>17</td><td>16</td><td>8</td></tr><tr><td>combination</td><td>14</td><td>=</td><td>6</td><td>13</td><td>7</td><td>22</td></tr><tr><td>startend</td><td>7</td><td>4</td><td>4</td><td>7</td><td>12</td><td>18</td></tr><tr><td>language</td><td>5</td><td>=</td><td>7</td><td>5</td><td>8</td><td>10</td></tr><tr><td>punctuation</td><td>3</td><td>1</td><td>28</td><td>21</td><td>8</td><td>3</td></tr><tr><td>detectable_content</td><td>3</td><td>3</td><td>3</td><td>5</td><td>1</td><td>11</td></tr><tr><td>Total</td><td>56</td><td>43</td><td>91</td><td>96</td><td>83</td><td>126</td></tr></table>

Table 7: Regressed IFEval prompts, counted per violated instruction category, per pair. A prompt that violates several categories is counted in each, and a prompt whose violations fall outside the listed categories is counted in none, so the columns need not sum to the Total row. The Total row is the regressed-prompt count and matches the Lost column of Table 2. "−" marks absence; bold marks the row-max.

The picture across the three is mixed rather than uniform. Coding rises (mean +7.3), longcontext retrieval is roughly flat (mean −1.1), and advanced reasoning declines mildly (mean −2.7, worst −7.4 on Qwen2.5-VL). What separates them from the format-sensitive cluster is the comparison on the same five pairs, where the four formatsensitive means run −5.6 to −8.5 pt. None of these three benchmarks separates by per-head QK-RMSNORM, unlike the format-sensitive four. The loss therefore tracks the demand for controlled output formatting rather than the task’s domain or difficulty alone, which is what the mechanism of Section 4 predicts.

<table><tr><td>Pair</td><td>QK</td><td>MBPP</td><td>MMLU-Pro</td><td>RULER</td></tr><tr><td>Qwen3-VL</td><td>per-head</td><td>+7.6</td><td>-3.4</td><td>-0.2</td></tr><tr><td>InternVL3.5</td><td>per-head</td><td>-2.4</td><td>+0.6</td><td>-0.4</td></tr><tr><td>Qwen2.5-VL</td><td>none</td><td>+14.2</td><td>-7.4</td><td>+0.3</td></tr><tr><td>InternVL3</td><td>none</td><td>+21.4</td><td>-0.2</td><td>+0.6</td></tr><tr><td>LLaVA-OV</td><td>none</td><td>-4.4</td><td>-3.2</td><td>-6.0</td></tr><tr><td>Mean</td><td></td><td>+7.3</td><td>-2.7</td><td>-1.1</td></tr></table>

Table 8: Extended capability panel: per-pair ∆ (VLM-LM − reference LLM, pp) on coding (MBPP, execution pass@1), advanced multi-domain reasoning (MMLU-Pro), and long-context retrieval (RULER), across the five headline pairs; Molmo2-O excluded. Same instruct protocol; per-pair QK category as in Table 1.

## D Mechanism Details

This appendix expands the mechanism analyses of Section 4: the calibration-set size choice (Appendix D.1), the two layerwise controls (Appendix D.2), channel-level ablation $( \mathsf { A p - }$ pendix D.3), and a random-direction perturbation control (Appendix D.4).

## D.1 Calibration Sample-Size Robustness

The $( G _ { \mathrm { b a s e } } , B _ { \mathrm { g a p } } )$ measurement of Section 4 and the Sink Strength S of Section 3 both use a default of $N = 1 5$ calibration prompts. We verify this choice is converged by sweeping $N \in$ $\{ 5 , 1 0 , 1 5 , 2 5 , 5 0 , 1 0 0 \}$ with 3 random seeds per cell, drawing each subset uniformly from a 100- prompt calibration pool disjoint from the IFEval test set. At $N = 1 0 0$ every seed draws the entire pool, so the seed scatter vanishes there by construction. This produces 18 measurements per pair across the 6 pairs, 108 filled cells in total. The hyperparameter is measurement-time only: it never trains a model and never sees benchmark labels, so it controls only the variance of the estimator.

Observations. (i) Plateau from $N \approx 1 0 .$ . Seedmean shifts are $\leq 0 . 0 5$ in $G _ { \mathrm { b a s e } } , \leq 0 . 0 5$ in $B _ { \mathrm { g a p } } ,$ and ≤ 0.06 in $G - B$ between $N { = } 1 0$ and $N { = } 1 0 0$ for every pair; by $N = 1 5$ the curves are indistinguishable from the $N = 1 0 0$ asymptote within seed scatter. (ii) Seed scatter contracts as expected. $\sigma ( G - B )$ drops from $\sim 0 . 1 5$ at N = 5 (worst case, InternVL3.5) $\mathrm { t o } \le 0 . 0 7$ by $N = 1 5$ and $\leq 0 . 0 2$ by $N = 5 0$ . (iii) Pair ordering invariant. Across all 108 (pair, N, seed) cells, the sign of $G - B$ is preserved (positive only for Qwen3-VL, negative for the other five), and the rank ordering of pairs on $G - B$ does not flip at any cell. The behavioral split and the magnitude ordering used in Section 4 therefore do not depend on the calibration set choice.

Sink Strength S. The sweep above tracks the margin $G _ { \mathrm { b a s e } } - B _ { \mathrm { g a p } }$ . The predictor of Section 3 uses S alone, which is less stable under small N, so we report it separately. Moving from $N = 1 5$ to $N = 5$ shifts S on the six pairs by at most 0.11 (mean 0.09), against the $\leq 0 . 0 6$ margin shift above. The predictor is nevertheless unchanged, with $\rho ( S , \Delta _ { \mathrm { I F E v a l } } ) = 0 . 9 7 1$ at both sizes and no reordering of the six pairs. Since $N = 5$ also runs about 3× faster $( \sim 3$ s versus ∼ 8 s per 7B backbone on an A6000), it reproduces the headline predictor at a third of the cost. We keep $N { = } 1 5$ as the default because the extra prompts are inexpensive and buy a margin against seed variance on broader panels, where the smaller set is not a safe drop-in replacement.

![](images/2578344a16aa71f0600b1783e0239d03ce1f3a944eba1294db34f534a9db42cd.jpg)

![](images/43dfe72ebf36545ba6d188da9bad69a88be3e53e9bac6436a7cdd1827bc650f3.jpg)

![](images/3000720814478fdc899f1b0094598b314165919d768995f0117df4557442fae9.jpg)

Figure 4: $G _ { \mathrm { b a s e } } , B _ { \mathrm { g a p } } ,$ and the $G _ { \mathrm { b a s e } } - B _ { \mathrm { g a p } }$ margin versus calibration size N for each of the six pairs. Lines are seed means, shaded bands are ±σ across 3 seeds. The vertical dashed line marks the paper-default $N = 1 5 ;$ the horizontal line in the right panel marks the $G - B = 0$ sign-flip boundary.
<table><tr><td>Pair</td><td>N=5</td><td>N=10</td><td>N=15</td><td>N=25</td><td> $N = 5 0$ </td><td> $N = 1 0 0$ </td></tr><tr><td>Qwen2.5-VL-7B</td><td> $- 3 . 9 2 \pm 0 . 0 7$ </td><td> $- 3 . 8 6 \pm 0 . 0 2$ </td><td> $\mathbf { - 3 . 9 0 \pm 0 . 0 2 }$ </td><td> $- 3 . 8 8 \pm 0 . 0 4$ </td><td> $- 3 . 8 8 \pm 0 . 0 2$ </td><td> $- 3 . 8 6 \pm 0 . 0 0$ </td></tr><tr><td>InternVL3-8B</td><td> $- 2 . 2 9 \pm 0 . 0 8$ </td><td> $- 2 . 2 7 \pm 0 . 0 3$ </td><td> $\mathbf { - 2 . 2 9 \pm 0 . 0 3 }$ </td><td> $- 2 . 2 9 \pm 0 . 0 1$ </td><td> $- 2 . 3 0 \pm 0 . 0 2$ </td><td> $- 2 . 2 9 \pm 0 . 0 0$ </td></tr><tr><td>LLaVA-OneVision-7B</td><td> $- 1 . 2 3 \pm 0 . 1 1$ </td><td> $- 1 . 2 1 \pm 0 . 0 6$ </td><td> $\mathbf { - 1 . 3 0 \pm 0 . 0 7 }$ </td><td> $- 1 . 2 4 \pm 0 . 0 4$ </td><td> $- 1 . 2 5 \pm 0 . 0 2$ </td><td> $- 1 . 2 3 \pm 0 . 0 0$ </td></tr><tr><td>Qwen3-VL-8B</td><td> $+ 0 . 2 2 \pm 0 . 1 0$ </td><td> $+ 0 . 2 3 \pm 0 . 0 6$ </td><td> $\mathbf { + 0 . 2 3 \pm 0 . 0 4 }$ </td><td> $+ 0 . 2 5 \pm 0 . 0 6$ </td><td> $+ 0 . 2 5 \pm 0 . 0 2$ </td><td> $+ 0 . 2 4 \pm 0 . 0 0$ </td></tr><tr><td>InternVL3.5-8B</td><td> $- 0 . 2 9 \pm 0 . 1 5$ </td><td> $- 0 . 2 3 \pm 0 . 1 1$ </td><td> $\mathbf { - 0 . 2 8 \pm 0 . 0 6 }$ </td><td> $- 0 . 2 7 \pm 0 . 0 5$ </td><td> $- 0 . 2 5 \pm 0 . 0 2$ </td><td> $- 0 . 2 6 \pm 0 . 0 0$ </td></tr><tr><td>Molmo2-O-7B</td><td> $- 3 . 1 3 \pm 0 . 0 5$ </td><td> $- 3 . 0 9 \pm 0 . 0 5$ </td><td> $\mathbf { - 3 . 1 6 \pm 0 . 0 6 }$ </td><td> $- 3 . 1 1 \pm 0 . 0 3$ </td><td> $- 3 . 1 1 \pm 0 . 0 1$ </td><td> $- 3 . 1 1 \pm 0 . 0 0$ </td></tr></table>

Table 9: $G _ { \mathrm { b a s e } } - B _ { \mathrm { g a p } }$ margin $( \mu \pm \sigma$ across 3 seeds) over the full N grid; the paper-default N = 15 column is bold. Each cell averages 3 random N-prompt subsets drawn from the 100-prompt sweep pool, which is separate from the fixed 15-prompt calibration set used in Table 5 and throughout the paper. This is why the $N { = } 1 5$ column differs slightly from that table. The pair ordering is identical in both.

## D.2 Layerwise Backbones: Aggregate vs. Per-Head Sink

Molmo2-O-7B is built on Olmo-3, which applies QK-RMSNORM layerwise: at each layer, RMS is computed across the full H · d<sub>h</sub>-dimensional concatenated Q (and K) vector before $\gamma _ { q } , \gamma _ { k }$ are applied, rather than across each head’s $d _ { h } .$ dimensional slice independently (as in Qwen3 / InternVL3.5). At the model-mean level the layerwise scale preserves a strong sink: position-0 attention is 0.250 (95% CI [0.242, 0.258]), over 8× larger than on the no-QK-RMSNORM Qwen2.5- VL backbone (0.030). At the per-head level the distribution is different. On the same calibration set (15-prompt, top-∼ 10 late layers), median $G _ { \mathrm { b a s e } }$ across (layer, head, query position) is +0.24 nats (the lowest among the six pairs), strong-sink prevalence $( G _ { \mathrm { b a s e } } > 2$ , equivalently $a _ { \mathrm { s i n k } } ~ > ~ 0 . 8 8 1 )$ over non-degenerate late-layer (prompt, layer, head, query) observations is 8.5%, against 58% on the per-head QK-RMSNORM backbone (Qwen3-8B). The per-head margin protection of Lemma 1 is therefore not delivered head-by-head, and IFEval drops by 18.7 pt, GSM8K-CoT by 42.7 pt, EQ-

Bench by 64.6 pt, GPQA-Diamond-CoT by 19.7 pt against the Olmo-3-7B-Instruct reference (the GPQA figure is measured against the published Olmo-3 score; our own matched run gives 10.6 pt). The reading is that aggregate model-level sink mass is not sufficient and that the evidence points to per-head sink strength as the relevant quantity. We therefore exclude Molmo2-O from the headline 5-pair comparison and report it separately as a distinguishing control.

A second layerwise pair. MolmoE-1B, built on OLMoE-1B-7B-0924 and scored here against the released OLMoE-1B-7B-0924-Instruct, applies QK-RMSNORM across all heads at once on an MoE backbone and is a second natural layerwise case. It reaches $S = + 1 . 3 4$ with $G _ { \mathrm { b a s e } } - B _ { \mathrm { g a p } } =$ −2.84 and an IFEval drop of 9.4 pt, so it follows the same pattern as Molmo2-O $( S = + 0 . 2 4$ $G _ { \mathrm { b a s e } } - B _ { \mathrm { g a p } } = - 3 . 0 1 , 1 8 . 7 \ : \mathrm { p t ) }$ with the lower-S pair taking the larger drop. Both layerwise pairs sit well below the per-head pairs, which share $S = + 2 . 3 6$ , and both carry strongly negative margins while differing in base family and in dense versus MoE architecture. The two are not independent of recipe, since both come from the Molmo and PixMo line, so the recipe-independent derivation of Appendix A.5 is what carries the argument and these two pairs stand as convergent empirical support rather than as a controlled comparison. Neither is a headline pair, and both are reported here as layerwise controls.

## D.3 Channel Ablation Sweep

For each of the three normalization-scale tensors in Qwen3-VL-8B’s language backbone (input\_layernorm $\gamma _ { \mathrm { i n } } .$ , q\_norm $\gamma _ { q } ,$ k\_norm γ<sub>k</sub>), we identify the top-K channels by |γ| at each layer and replace those entries with the layer-wise channel mean. This disables the amplifier while leaving the residual stream and projections untouched. We fix K = 10 for all three tensors, so the ablation targets the largest-magnitude amplifier channels at each layer. Joint kill (all three norms simultaneously) costs 44.36 pt on IFEval, against a sum of individual costs of 19.96 pt, a 24.40-pt superadditive interaction.

## D.4 Random-∆W Control

We add Gaussian noise $G _ { W } \sim \mathcal { N } ( 0 , \sigma _ { W } ^ { 2 } I )$ to each attention / MLP projection W of Qwen2.5-7B-Instruct with $\sigma _ { W }$ chosen so that the per-sub-module relative Frobenius norm $\| G _ { W } \| _ { F } / \| W \| _ { F }$ exactly matches the corresponding ratio observed between Qwen2.5-VL-7B’s extracted LM and Qwen2.5-7B-Instruct. The γ parameters of every RMSNorm are left untouched. The result is a model whose persubmodule relative Frobenius distances from the reference LLM match those observed for Qwen2.5- VL, but whose perturbation direction is random. IFEval under the instruct protocol drops to 10.72 $( \Delta = - 6 1 . 3 7$ against the reference LLM at 72.09), compared with the observed Qwen2.5-VL gap of −9.6 pt.

## E Sink Strength Predictor

## E.1 Per-Pair Sink Strength Predictions

Per-pair numerical detail behind Figure 3 (Section 3): the reference-LLM Sink Strength S, the observed VLM–reference IFEval gap, and the leaveone-pair-out linear prediction.

## E.2 IFEval-Specific Predictor (17 pairs)

The IFEval-specific predictor panel (Table 11) measures Sink Strength S against the VLM–reference IFEval prompt-strict ∆ on an extended 17-pair set: the 6 pairs plus 11 additional VLMs that span MoE architectures (Qwen3-VL-30B-A3B-Instruct, MolmoE-1B), four reference-LLM families (Qwen, Olmo, Mistral, Llama-3.1) plus the MoE OL-MoE, and backbone sizes 1.5B–32B (full pair-toreference-LLM mapping with Hugging Face URLs in Appendix B.4). The panel therefore covers MoE and dense architectures, four reference-LLM families, and 1.5B–32B scale. Over it, Spearman $\rho ( S , \Delta _ { \mathrm { I F E v a l } } ) = + 0 . 8 8$ . Two caveats bound what a single rank correlation can carry here. Pairs that share a reference LLM share S by construction, so the 17 pairs supply 12 distinct S values, and 13 of the 17 are Qwen-family, so the four non-Qwen backbones are single points rather than a familylevel sample. Two entries illustrate the resolution limit of a reference-LLM statistic. Ovis2-8B shares $S = + 1 . 1 8$ with Qwen2.5-VL and InternVL3 but loses −17.9 pt against their −9.6 and −8.9, and Qwen3-VL-30B-A3B is the one pair whose VLM-LM exceeds its reference (+1.7 pt). S orders backbones by risk; it does not predict the magnitude a particular VL recipe will realize.

<table><tr><td>Pair</td><td>QK</td><td>S</td><td>Actual ∆</td><td>Held-out pred ∆</td></tr><tr><td>Qwen3-VL</td><td>per-head</td><td>+2.36</td><td>-5.4</td><td>-0.92</td></tr><tr><td>InternVL3.5</td><td>per-head</td><td>+2.36</td><td>-1.8</td><td>-3.41</td></tr><tr><td>Qwen2.5-VL</td><td>none</td><td>+1.18</td><td>-9.6</td><td>-10.78</td></tr><tr><td>InternVL3</td><td>none</td><td>+1.18</td><td>-8.9</td><td>-10.95</td></tr><tr><td>LLaVA-OV</td><td>none</td><td>+1.44</td><td>-7.9</td><td>-9.04</td></tr><tr><td>Molmo2-O</td><td>layerwise</td><td>+0.24</td><td>-18.7</td><td>-13.74</td></tr></table>

Table 10: Per-pair Sink Strength S measured on the reference LLM, the observed IFEval gap between the VLM-LM and reference LLM, and the leave-one-pairout linear prediction.

## E.3 Multi-Task Generalization (9 pairs)

Where Appendix E.2 widens the model set at a fixed task, this panel widens the task set instead. It adds three sub-10B Qwen-family pairs to the six (Qwen3-VL-4B, Qwen3-VL-2B, InternVL3- 2B), which extends the panel down to 1.5B reference LLMs while keeping all four format-sensitive tasks on the unified protocol. Table 12 reports perpair Sink Strength S and the four VLM–reference deltas. The IFEval correlation is +0.945 against +0.971 on the six, and the other three tasks fall between +0.82 and +0.88.

## E.4 Sink Strength Design Ablations

Section 3 defines S as a median taken over the last ∼ 10 decoder layers. This subsection isolates the three choices that definition makes.

<table><tr><td>Pair</td><td>Reference LLM</td><td>QK class</td><td>S</td><td>∆IFEval</td></tr><tr><td>Qwen2.5-VL</td><td>Qwen2.5-7B-Instruct</td><td>none</td><td>+1.18</td><td>-9.6</td></tr><tr><td>Qwen3-VL-8B</td><td>Qwen3-8B</td><td>per-head</td><td>+2.36</td><td>-5.4</td></tr><tr><td>InternVL3-8B</td><td>Qwen2.5-7B-Instruct</td><td>none</td><td>+1.18</td><td>-8.9</td></tr><tr><td>InternVL3.5-8B</td><td>Qwen3-8B</td><td>per-head</td><td>+2.36</td><td>-1.8</td></tr><tr><td>LLaVA-OV-7B</td><td>Qwen2-7B-Instruct</td><td>none</td><td>+1.44</td><td>-7.9</td></tr><tr><td>Molmo2-O-7B</td><td>Olmo-3-7B-Instruct</td><td>layerwise</td><td>+0.24</td><td>-18.7</td></tr><tr><td>Qwen3-VL-4B</td><td>Qwen3-4B</td><td>per-head</td><td>+2.36</td><td>-5.0</td></tr><tr><td>Qwen3-VL-2B</td><td>Qwen3-1.7B</td><td>per-head</td><td>+1.84</td><td>-5.2</td></tr><tr><td>Ovis2-8B</td><td>Qwen2.5-7B-Instruct</td><td>none</td><td>+1.18</td><td>-17.9</td></tr><tr><td>Ovis2-34B</td><td>Qwen2.5-32B-Instruct</td><td>none</td><td>+1.39</td><td>-6.7</td></tr><tr><td>MolmoE-1B</td><td>OLMoE-1B-7B-0924-Instruct</td><td>layerwise (MoE)</td><td>+1.34</td><td>-9.4</td></tr><tr><td>InternVL3-2B</td><td>Qwen2.5-1.5B-Instruct</td><td>none</td><td>+0.90</td><td>-12.2</td></tr><tr><td>InternVL3-14B</td><td>Qwen2.5-14B-Instruct</td><td>none</td><td>+1.30</td><td>-5.2</td></tr><tr><td>MiniCPM-V-4.5</td><td>Qwen3-8B</td><td>per-head</td><td>+2.36</td><td>-4.4</td></tr><tr><td>LLaVA-NeXT-Mistral-7B</td><td>Mistral-7B-Instruct-v0.2</td><td>none</td><td>-0.19</td><td>-10.9</td></tr><tr><td>Idefics3-8B</td><td>Llama-3.1-8B-Instruct</td><td>none</td><td>-0.21</td><td>-22.7</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>Qwen3-30B-A3B-Instruct-2507</td><td>per-head (MoE)</td><td>+1.66</td><td>+1.7</td></tr><tr><td colspan="3"> $\rho _ { \mathrm { S p e a r m a n } } ( S , \Delta _ { \mathrm { I F E v a l } } )$  on 17 pairs</td><td></td><td>+0.88</td></tr></table>

Table 11: IFEval-specific 17-pair predictor panel. Sink Strength S measured on the reference LLM versus the VLM-LM–reference-LLM IFEval prompt-strict ∆. The panel spans dense and MoE architectures, all three QKnorm classes (per-head / layerwise / none), four reference-LLM families (Qwen, Olmo, Mistral, Llama-3.1) plus the MoE OLMoE, and backbone sizes 1.5B–32B. Hugging Face URLs and paper references for every pair are in Appendix B.4.

Why the late layers. We split each reference LLM’s decoder into equal early, middle, and late thirds and recompute S within each band (Table 13). Sink concentration and its correlation with the outcome rise together with depth. The early third carries almost no signal, while the late third alone reproduces the full predictor at $\rho = 0 . 9 7$ The late third overlaps the last-∼ 10-layer default for every headline pair, so the emphasis on late layers reflects where the sink lives rather than a tuned window.

Sensitivity to the window size. Recomputing S over the last K layers for K ∈ {3, 5, 10, 15, 20} leaves the rank correlation unchanged at ρ = 0.971 for every K. The magnitudes of S shift with K, but no pairwise ordering reverses, so the last-∼ 10- layer window is a robust choice rather than a tuned one.

Why the median. The statistic should report whether the typical head forms a sink, which is what a median over (ℓ, h, q) measures, whereas a mean is inflated by a few strong heads. Substituting the mean preserves both the sign and the pair ordering, so this choice changes none of the conclusions drawn from S. Appendix D.2 is the case that motivates it, where a strong model-level aggregate coexists with a weak per-head distribution that only the per-head statistic exposes.

## F Controls

This appendix details the modality comparison of Section 3.1 and the controls of Section 5: a text-only Qwen2.5-family endpoint (Appendix F.1), a step-wise VL versus text-only trajectory from a shared 3B base (Appendix F.2), a post-pretraining QK-RMSNORM injection experiment (Appendix F.3), and a weight-merging sweep (Appendix F.4).

## F.1 Modality Comparison: Text-Only Endpoint

The Section 3.1 modality comparison (Table 3) uses Qwen2.5-7B-Instruct-1M, the Qwen team’s official long-context text-only release, as a Qwen2.5- family further-training endpoint (Hugging Face URL in Appendix B.4).

Both this model and Qwen2.5-VL-7B-Instruct are evaluated against the same Qwen2.5-7B-Instruct text-only reference. The text-only variant is evaluated under the same 0-shot instruct protocol as the rest of the paper (Appendix B.1), with batch size 8 on a single A6000.

## F.2 Modality Control: VL vs. Text-Only Trajectory

The Appendix F.1 comparison contrasts released 7B endpoints against a common text-only reference.

<table><tr><td>Pair</td><td>Reference LLM</td><td>QK</td><td>S</td><td>∆IFEval</td><td>∆GSM8K</td><td>∆EQ-Bench</td><td>∆GPQA-D</td></tr><tr><td>Qwen2.5-VL</td><td>Qwen2.5-7B-Instruct</td><td>none</td><td>+1.18</td><td>-9.6</td><td>-14.3</td><td>-11.2</td><td>-9.8</td></tr><tr><td>Qwen3-VL-8B</td><td>Qwen3-8B</td><td>per-head</td><td>+2.36</td><td>-5.4</td><td>-2.1</td><td>-2.4</td><td>-1.9</td></tr><tr><td>InternVL3-8B</td><td>Qwen2.5-7B-Instruct</td><td>none</td><td>+1.18</td><td>-8.9</td><td>-12.5</td><td>-10.3</td><td>-7.1</td></tr><tr><td>InternVL3.5-8B</td><td>Qwen3-8B</td><td>per-head</td><td>+2.36</td><td>-1.8</td><td>-1.5</td><td>-0.8</td><td>-0.6</td></tr><tr><td>LLaVA-OV-7B</td><td>Qwen2-7B-Instruct</td><td>none</td><td>+1.44</td><td>-7.9</td><td>-11.9</td><td>-8.6</td><td>-8.4</td></tr><tr><td>Molmo2-O-7B</td><td>Olmo-3-7B-Instruct</td><td>layerwise</td><td>+0.24</td><td>-18.7</td><td>-42.7</td><td>-64.6</td><td>-19.7</td></tr><tr><td>Qwen3-VL-4B*</td><td>Qwen3-4B</td><td>per-head</td><td>+2.36</td><td>-5.0</td><td>+13.9</td><td>+1.4</td><td>-2.0</td></tr><tr><td>Qwen3-VL-2B*</td><td>Qwen3-1.7B</td><td>per-head</td><td>+1.84</td><td>-5.2</td><td>+27.5</td><td>-0.7</td><td>-6.1</td></tr><tr><td>InternVL3-2B*</td><td>Qwen2.5-1.5B-Instruct</td><td>none</td><td>+0.90</td><td>-12.2</td><td>-18.0</td><td>-9.0</td><td>-6.1</td></tr><tr><td colspan="4">ρSpearman(S, ∆) on 9 pairs</td><td>+0.945</td><td>+0.877</td><td>+0.860</td><td>+0.821</td></tr></table>

Table 12: Extended 9-pair predictor panel. ∆ denotes VLM-LM minus the corresponding reference LLM (negative = lower performance relative to the reference) on IFEval prompt-strict, GSM8K-CoT flexible-extract, EQ-Bench v2.1, and GPQA-Diamond-CoT flexible-extract; all evaluated under the unified protocol (Appendix B.1), except that the Molmo2-O GPQA baseline is the published Olmo-3-7B-Instruct score rather than our own matched run. <sup>⋆</sup> marks the three pairs added beyond the six, which span S ∈ [0.90, 2.36] and reference-LLM sizes 1.5B–4B. Bold ∆IFEval column is the headline metric.

![](images/2b11e6a30b25de67b9e1eeb73420c0d888c16c61238beae2b8a8046b6237ae36.jpg)  
Figure 5: IFEval prompt-strict for Qwen2.5-VL (non-protected) and Qwen3-VL (per-head QK-RMSNORM) under seven weight-merging settings. Dashed line is each pair’s reference LLM.

<table><tr><td>Third</td><td>S mean</td><td> $\rho ( S , \Delta _ { \mathrm { I F E v a l } } )$ </td></tr><tr><td>Early</td><td>+0.20</td><td>+0.09</td></tr><tr><td>Middle</td><td>+0.91</td><td>+0.79</td></tr><tr><td>Late</td><td>+1.56</td><td>+0.97</td></tr></table>

Table 13: Sink Strength recomputed within equal thirds of the decoder, on the six pairs. Both the magnitude of S and its rank correlation with the VLM–reference IFEval gap increase with depth.

This subsection instead tracks the two modalities step by step on the 3B base of the injection experiment, so the sink can be observed while it changes. Starting from Qwen2.5-3B-Instruct, we run vision fine-tuning on image-text instruction data (the Stage-2 recipe of Appendix F.3) and a matched textonly run on Tülu general instruction SFT (Lambert et al., 2025), at the same effective batch size, learning rate, and number of steps, matching the optimization settings while varying the training modality and data. At each checkpoint we measure Sink Strength S on the trained backbone, vision understanding on SEEDBench-IMG (Li et al., 2023), and IFEval prompt-strict.

Table 14 reports the trajectories. The vision run gains vision understanding while its sink collapses, with SEEDBench-IMG rising from 60.1 to 64.8 and S falling from +0.41 at the shared base to +0.20 at step 10k, and IFEval sitting in the 34– 44 band throughout. The text-only run moves the sink in the opposite direction, holding S between +0.58 and +0.65, and gives up about 5 pt of IFEval. The step-0 IFEval of 59.9 reproduces the published Qwen2.5-3B-Instruct score of 58.2 (Yang et al., 2024b) to within 1.7 pt, so the drops start from a correctly measured baseline. Within the vision run the two curves are not step-by-step aligned. IFEval takes most of its loss by step 1k, before S has fallen, and then recovers slightly while S continues to decline, so S tracks the regime the run ends in rather than the step-level fluctuation.

<table><tr><td rowspan="2">Step</td><td colspan="3">VL adaptation</td><td colspan="2">Tülu (text-only)</td></tr><tr><td>S</td><td>SEED</td><td>IFEval</td><td>S</td><td>IFEval</td></tr><tr><td>0</td><td>+0.41</td><td></td><td>59.9</td><td>+0.41</td><td>59.9</td></tr><tr><td>1k</td><td>+0.54</td><td>60.1</td><td>36.8</td><td>+0.65</td><td>53.1</td></tr><tr><td>2k</td><td>+0.49</td><td>60.9</td><td>37.2</td><td>+0.58</td><td>53.6</td></tr><tr><td>3k</td><td>+0.35</td><td>62.1</td><td>34.2</td><td>+0.61</td><td>55.3</td></tr><tr><td>4k</td><td>+0.30</td><td>63.0</td><td>40.5</td><td>+0.60</td><td>56.9</td></tr><tr><td>5k</td><td>+0.24</td><td>63.9</td><td>39.7</td><td>+0.58</td><td>54.3</td></tr><tr><td>6k</td><td>+0.22</td><td>63.9</td><td>38.6</td><td>+0.58</td><td>55.1</td></tr><tr><td>7k</td><td>+0.19</td><td>64.3</td><td>42.3</td><td>+0.58</td><td>55.3</td></tr><tr><td>8k</td><td>+0.24</td><td>64.7</td><td>40.7</td><td>+0.58</td><td>53.8</td></tr><tr><td>9k</td><td>+0.20</td><td>65.0</td><td>43.1</td><td>+0.58</td><td>54.9</td></tr><tr><td>10k</td><td>+0.20</td><td>64.8</td><td>43.6</td><td>+0.58</td><td>55.8</td></tr></table>

Table 14: Step-wise modality control on Qwen2.5-3B-Instruct. S is Sink Strength on the trained backbone, SEED is SEEDBench-IMG overall accuracy, and IFEval is prompt-strict. Both runs start from the same base and use the same batch size, learning rate, and step count, while differing in training modality and data.

## F.3 Injection Experiment Training Recipe

The injection experiment trains two 3B VLMs from the same Qwen2.5-3B-Instruct base under the same LLaVA Stage-1 / Stage-2 recipe. The two variants are vanilla (no QK-RMSNORM) and QK-RMSNorm injected with unit-scale initialization $\gamma _ { q } = \gamma _ { k } = 1$ . Vision tower and projector come from Qwen2.5-VL-3B-Instruct. Both runs proceed on dedicated 2× NVIDIA H100 nodes with DeepSpeed ZeRO-2. Total walltime is ∼ 47 h per variant (∼ 7 h Stage 1, ∼ 40 h Stage 2).

<table><tr><td>Hyperparameter</td><td>Stage 1 (align)</td><td>Stage 2 (instruct)</td></tr><tr><td>Trainable params</td><td>projector only</td><td>full LM + projector</td></tr><tr><td>Dataset</td><td>LLaVA-Pretrain (558K)</td><td>LLaVA-Mix (665K)</td></tr><tr><td>Image resolution</td><td>3362 (112,896 px)</td><td>6722 (451,584 px)</td></tr><tr><td>Per-device batch</td><td>32</td><td>4</td></tr><tr><td>Grad accumulation</td><td>2 (NGPU= 2)</td><td>16</td></tr><tr><td>Effective batch</td><td>128</td><td>128</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>1×10−4</td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>LR schedule</td><td>const + warmup</td><td>cosine</td></tr><tr><td>Warmup ratio</td><td>0.03</td><td>0.03</td></tr><tr><td>Weight decay</td><td>0.0</td><td>0.0</td></tr><tr><td>Max grad norm</td><td>1.0</td><td>1.0</td></tr><tr><td>Epochs</td><td>1</td><td>1</td></tr><tr><td>Steps</td><td>4361</td><td>~5195</td></tr><tr><td>Precision</td><td>bf16</td><td>bf16</td></tr><tr><td>DeepSpeed</td><td>ZeRO-2</td><td>ZeRO-2</td></tr><tr><td>Save every</td><td>end of stage</td><td>500 steps</td></tr><tr><td>Save total limit</td><td></td><td>2</td></tr></table>

Table 15: Training recipe for the post-pretraining QK-RMSNORM injection experiment (vanilla and injected variants are matched on all rows; the QK-RMSNORM module is the only difference).

![](images/ad43dc1c5ea56f953323574642e57bebe39357febe83d6e2aaeeb0d981dfe7d0.jpg)  
Figure 6: IFEval prompt-strict over the LLaVA Stage-2 training epoch for the controlled post-pretraining QK-RMSNORM injection pair (vanilla vs. injected, matched in all other settings).

## F.4 Weight-Merging Sweep on Format-Sensitive Benches

The body reports IFEval on the two pairs (Qwen2.5- VL, Qwen3-VL) for seven weight-merging settings: the asymmetric 95% VLM + 5% LLM mix, uniform averaging, full Task Arithmetic (λ=1), Task Singular Vectors (α=1), STAR (η=40%), TIES (density 0.2, λ=1), and DARE (mask rate 0.9, λ=1). We also ran the same settings on EQ-Bench, GSM8K-CoT flexible-extract, and GPQA-Diamond-CoT flexible-extract, and summarize the pattern here rather than reproducing three further figures. It matches IFEval, in that the mild mixes (95/5, uniform averaging) avoid substantial additional degradation, full-strength operators (TA, TSV, STAR) drop the non-protected pair by tens of points, and sparsified merges (TIES, DARE) are the most damaging across both pairs.