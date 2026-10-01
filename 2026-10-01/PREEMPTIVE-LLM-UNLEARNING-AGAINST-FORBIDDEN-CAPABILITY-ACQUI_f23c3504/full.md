# PREEMPTIVE LLM UNLEARNING AGAINST FORBIDDEN CAPABILITY ACQUISITION VIA GRADIENT SEALING

Kemou Li<sup>1</sup> Qizhou Wang<sup>2</sup> Yue Wang<sup>3</sup> Fengpeng Li<sup>4</sup> Zhuan Shi<sup>5,6</sup> Negar Rostamzadeh<sup>5,6,7</sup> Golnoosh Farnadi<sup>5,6</sup> Masashi Sugiyama<sup>2,8</sup> Jiantao Zhou<sup>1</sup>

<sup>1</sup>State Key Laboratory of Internet of Things for Smart City, University of Macau <sup>2</sup>RIKEN Center for Advanced Intelligence Project <sup>3</sup>The University of Melbourne <sup>4</sup>King Abdullah University of Science and Technology <sup>5</sup>Mila – Québec AI Institute <sup>6</sup>McGill University <sup>7</sup>Google Research <sup>8</sup>The University of Tokyo

## ABSTRACT

Open-weight LLMs are released not only as fixed products but also as substrates for downstream fine-tuning. This openness, however, creates legal and ethical risks because users may misuse fine-tuning to instill illicit knowledge or enable hostile operations. Model providers therefore need apre-release defense against such acquisition, motivating the problem of preemptive unlearning. Unlike retrospective unlearning, which removes capabilities already present in a fixed model, preemptive unlearning seeks to prevent their acquisition under unseen attack data and future fine-tuning procedures. Despite its practical importance, this setting remains largely unexplored, presents distinct challenges, and is therefore the central focus of our work. We first verify that existing retrospective methods provide insufficient pre-release protection. Even when forbidden capabilities are suppressed in current outputs, forbidden-domain data can still induce gradients through internal pathways, enabling later acquisition. Motivated by this finding, we propose a gradient-sealing principle that blocks these pathways by pushing relevant preactivations into the negative region, where ReLU-family activations exhibit zero or near-zero derivatives. Experiments across multiple LLM families demonstrate our stronger resistance to downstream acquisition than retrospective baselines, validating gradient sealing as an effective mechanism for pre-release protection.

## 1 INTRODUCTION

Open-weight large language models (LLMs) (Jiang et al., 2023; Grattafiori et al., 2024; Gemma Team et al., 2025; Yang et al., 2025a) have become central to modern AI research and practice. Their acces sible parameters enable cost-effective, privacy-preserving deployment and domain-specific adaptation with modest compute. However, the same openness creates legal and ethical risks, as malicious users may fine-tune released models on harmful data to acquire capabilities for disinformation, explicit content generation, dangerous operational guidance, or privacy violations (Qi et al., 2024; Rosati et al., 2024; Kaunismaa et al., 2026), threatening both individuals and society. Model providers therefore need a pre-release defense that designates aforbidden domain and makes the corresponding capabilities resistant to acquisition after release. This motivates a critical yet underexplored question:

## Can we precondition a model to resist the acquisition of forbidden-domain capabilities through downstream fine-tuning?

A closely related line of research is LLM unlearning (Yao et al., 2024; Li et al., 2024; Zhang et al., 2024). Given a designated forget set, unlearning methods modify model parameters to suppress content already learned, as shown in Fig. 1a (Jang et al., 2023). However, these methods are inherently retrospective. Our pre-release defense is instead preemptive: rather than removing knowledge already embedded in the model, it aims to make forbidden-domain capabilities resistant to acquisition through downstream fine-tuning, as shown in Fig. 1b. Preemptive unlearning remains largely underexplored and particularly challenging because users have full access to released model parameters, making external guardrails (Inan et al., 2023a; Han et al., 2024) extremely difficult to enforce.

![](images/022efb07b05451314795ef8082898ac4b84d3277b7bc55ed8f41e38ca00ad337.jpg)  
Figure 1: Retrospective unlearning vs. preemptive unlearning. (a) Retrospective unlearning suppresses an already learned capability but may leave its gradient pathway open. (b) Preemptive unlearning instead uses a forbidden-domain proxy set before release to seal that pathway, preventing future acquisition from fine-tuning.

We further distinguish retrospective from preemptive unlearning by showing why retrospective methods provide insufficient pre-release protection. The key insight is that existing retrospective methods typically suppress current outputs while leaving the underlying gradients unconstrained (Fan et al., 2025a). Because downstream fine-tuning updates model parameters through these gradients, such open gradient pathways can still enable the acquisition of forbidden-domain capabilities. Our theoretical analysis in §3.1 formalizes this intuition, showing that forbidden-domain data can induce gradients along domain-sensitive pathways even when the released model exhibits little forbiddendomain capability. Empirical results in §3.2 further show that low release capability can hide high acquisition susceptibility, while proxy look-ahead can locate and targeted attenuation can weaken those responsible pathways. Together, these findings reinforce known robustness limitations of retrospective unlearning (Wang et al., 2025a; Lang et al., 2026).

These findings suggest that pre-release defense must seal acquisition pathways beyond output suppression. Directly penalizing acquisition-gradient norms, however, requires costly second-order differentiation. We instead exploit ReLU-family activation gates with zero or near-zero derivatives for sufficiently negative inputs. Pushing selected pre-activations into this regime attenuates their local gradient factors using only first-order updates. Accordingly, we propose Gradient-Sealed Unlearning (GSU), which uses proxy learning on a disposable copy to identify acquisition-sensitive gates from upward pre-activation shifts under matched contexts. Restarting from the original weights, GSU jointly optimizes a seal loss, response suppression, and retain supervision. The seal loss in Eq. (4) pushes selected forbidden-domain pre-activations below a shared negative-tail threshold, while the other terms suppress release-time responses and preserve normal utility.

To our knowledge, GSU represents one of the earliest systematic efforts to formulate and evaluate preemptive unlearning as a pre-release defense. It may also help strengthen retrospective unlearning against post-unlearning attacks (Hu et al., 2025a). In §5, we extensively evaluate GSU on WMDP (Li et al., 2024) and TOFU (Maini et al., 2024) across multiple LLM families, where it shows stronger resistance to downstream acquisition of forbidden capabilities than competitive retrospective unlearning baselines. These results support GSU’s gradient-sealing principle and activation-gating mechanism.

## 2 PROBLEM STATEMENT

Open-weight LLMs face legal and ethical risks similar to those of closed-weight LLMs (Bommasani et al., 2021; Kapoor et al., 2024). Releasing model parameters introduces an additional challenge: users can fine-tune a released model to acquire a designated forbidden capability, even after retrospective unlearning (Yao et al., 2024; Wang et al., 2025c), as discussed in §3. This motivates our study of preemptive unlearning, an equally important yet largely unexplored setting. Unlike retrospective unlearning, which removes capabilities already present in a model, preemptive unlearning conditions a model before release to resist the acquisition of forbidden capabilities through future fine-tuning.

Let $\mathcal { F }$ denote the forbidden domain and R the normal domain whose utility should be preserved, with corresponding distributions $\mathbb { P } _ { \mathcal { F } }$ and $\mathbb { P } _ { \mathcal { R } }$ over prompt–response pairs. For a performance functional Perf and model parameters θ, define $\mathsf { P e r f } _ { \mathcal { F } } ( \pmb { \theta } ) \overset { \cdot } { : } = \dot { \mathsf { P e r f } } ( \pmb { \theta } ; \mathbf { \bar { \mathbb { P } } } _ { \mathcal { F } } )$ and $\mathsf { P e r f } _ { \mathcal { R } } ( \pmb { \theta } ) \dot { : } = \mathsf { P e r f } ( \pmb { \theta } ; \mathbb { P } _ { \mathcal { R } } )$ , where higher values indicate stronger forbidden-domain capability and normal-domain utility, respectively. We formalize preemptive unlearning through the following attack–defense framework.

## 2.1 POST-RELEASE ATTACK

After release, an attacker obtains white-box access to the released model parameters $\theta _ { \mathrm { r e l } }$ and an unseen forbidden-domain dataset $\mathcal { D } _ { a } \sim \mathbb { P } _ { \mathcal { F } } ^ { \otimes n _ { a } }$ . An attack A from a class A produces $\theta _ { \mathrm { a t k } } = \mathcal { A } ( \theta _ { \mathrm { r e l } } ; \mathcal { D } _ { a } )$ Within A, the attacker may choose the training objective, optimizer, schedule, and other configurations, with the goal of acquiring the forbidden capability while retaining normal utility, as formalized below.

Definition 2.1 (Successful acquisition attack). Given a maximum acceptable forbidden-capability threshold $\epsilon _ { f }$ and a minimum usable-utility threshold $\epsilon _ { u } ,$ an attack ${ \mathcal { A } } \in { \mathfrak { A } }$ is successful against $\theta _ { \mathrm { r e l } }$ if the resulting $\pmb { \theta } _ { \mathrm { a t k } }$ satisfies both (i) E $[ \mathsf { P e r f } _ { \mathcal { F } } ( \pmb { \theta } _ { \mathrm { a t k } } ) ] > \epsilon _ { f }$ and (ii) $\mathbb { E } [ \mathsf { P e r f } _ { \mathcal { R } } ( \pmb { \theta } _ { \mathrm { a t k } } ) ] > \epsilon _ { u }$ . The expectations are taken over the sampling of $\mathcal { D } _ { a }$ and the internal randomness of $\mathcal { A }$

Def. 2.1 excludes attacks that acquire the forbidden capability only by severely degrading normaldomain utility. In such cases, high forbidden-domain performance may result primarily from fitting $\mathcal { D } _ { a }$ rather than from reusing information or capabilities retained in $\theta _ { \mathrm { r e l } }$ . Such attacks therefore fall outside our intended scope and do not constitute a failure of the defense introduced below.

## 2.2 PRE-RELEASE DEFENSE

Before release, the defender has access to the original trained model parameters $\theta _ { o }$ and a forbiddendomain proxy set $\mathcal { D } _ { f } \sim \mathbb { P } _ { \mathcal { F } } ^ { \otimes n _ { f } }$ . Remaining agnostic to the future attack dataset $\mathcal { D } _ { a }$ and attack procedure A, we seek a reliable preemptive unlearning algorithm U that produces defense-enhanced parameters $\pmb { \theta } _ { \mathrm { r e l } } = \mathcal { U } ( \pmb { \theta } _ { o } ; \mathcal { D } _ { f } )$ resistant to every ${ \mathcal { A } } \in { \mathfrak { A } }$ applied using any relevant $\mathcal { D } _ { a }$

Definition 2.2 (Preemptive unlearning). Fix an attack class A, thresholds $\left( \epsilon _ { f } , \epsilon _ { u } \right)$ from Def. 2.1, and an allowed release-time utility degradation $\epsilon _ { r } \geq 0$ such that $\epsilon _ { u } < \mathsf { P e r f } _ { \mathcal { R } } \mathopen { } \mathclose \bgroup \left( \theta _ { o } \aftergroup \egroup \right) - \epsilon _ { r } . \mathrm { ~ A ~ }$ released model $\theta _ { \mathrm { r e l } } \mathrm { i s } \left( \epsilon _ { f } , \epsilon _ { r } , \epsilon _ { u } \right)$ -preemptively unlearned against A w.r.t. $( \mathcal { F } , \mathcal { R } ) \mathrm { i f } \mathrm { ( i ) }$ no attack ${ \mathcal { A } } \in { \mathfrak { A } }$ is successful against $\theta _ { \mathrm { r e l } }$ under Def. 2.1, and (ii) $\theta _ { \mathrm { r e l } }$ satisfies Perf $( \pmb { \theta } _ { \mathrm { r e l } } ) \geq \mathsf { P e r f } _ { \mathcal { R } } ( \pmb { \theta } _ { o } ) - \epsilon _ { r }$

As with retrospective unlearning, a preemptive defense must preserve the model’s overall utility. Achieving robustness by substantially degrading normal performance is unacceptable, particularly because even modest performance gains often require considerable training time and compute. We therefore seek a released model that both resists post-release attacks and retains strong utility in normal use. In the next section, we further explain, both theoretically and empirically, why retrospective unlearning methods fall short in our preemptive setting.

## 3 WHY RETROSPECTIVE UNLEARNING FALLS SHORT

To see why directly repurposing retrospective unlearning may fail to provide the pre-release defense defined in §2, we first revisit what retrospective methods actually optimize. Their objectives suppress target behavior at the current parameters, whereas preemptive unlearning is evaluated after an attacker updates those parameters using unseen forbidden-domain data. Retrospective objectives therefore do not account for the gradient induced by post-release fine-tuning on such data. §3.1 formalizes this mismatch, and $\ S 3 . 2$ tests its predictions empirically.

## 3.1 FROM LIKELIHOOD SUPPRESSION TO FUTURE ACQUISITION GRADIENTS

We first formalize how retrospective unlearning suppresses the likelihood of undesirable responses, and then show why this suppression alone does not guarantee resistance to future acquisition.

Retrospective likelihood suppression. Given a prompt x and response $\mathbf { y } = ( y ^ { 1 } , \ldots , y ^ { | \mathbf { y } | } )$ , an LLM with parameters $\pmb { \theta }$ assigns the autoregressive likelihood $\textstyle \pi _ { \pmb { \theta } } ( \mathbf { y } \mid \mathbf { x } ) = \prod _ { i = 1 } ^ { | \mathbf { y } | } \pi _ { \pmb { \theta } } ( y ^ { i } \mid \mathbf { x } , \mathbf { y } ^ { < i } )$ and negative log-likelihood (NLL) $\ell _ { \pmb { \theta } } ( \mathbf { x } , \mathbf { y } ) = - \log \pi _ { \pmb { \theta } } ( \mathbf { y } \mid \mathbf { x } )$ . Starting from an original mode $\theta _ { o }$ that already contains the target behavior, retrospective unlearning uses a known forget set $\mathcal { D } _ { u }$ and a retain set $\mathcal { D } _ { r }$ <sub>r</sub> to produce an unlearned model $\theta _ { u }$ . It seeks to lower $\pi _ { \pmb { \theta } _ { u } } ( \mathbf { y } \mid \mathbf { x } )$ for $( \mathbf { x } , \mathbf { y } ) \in \mathcal { D } _ { u }$ while preserving normal behavior on $\mathcal { D } _ { r }$ . A broad class of methods can be written schematically as

$$
\operatorname* { m i n } _ { \theta } \ \mathcal { L } _ { \mathrm { R U } } ( \theta ) = \underbrace { \mathcal { L } _ { \mathrm { s u p } } ( \theta ; \mathcal { D } _ { u } ) } _ { \mathrm { s u p p r e s s t a r g e t r e s p o n s e s } } + \lambda _ { r } \underbrace { \mathcal { L } _ { \mathrm { r e t } } ( \theta ; \mathcal { D } _ { r } , \theta _ { o } ) } _ { \mathrm { p r e s e r v e n o r m a l b e h a v i o r } } \ ,
$$

where $\lambda _ { r } \geq 0$ controls the forget–retain trade-off. Within this common framework, gradient ascent (GA) (Jang et al., 2023; Yao et al., 2024) maximizes the forget-set NLL; GradDiff (Maini et al., 2024) and NPO (Zhang et al., 2024) add retain- or reference-model constraints; and RMU (Li et al., 2024) performs representation-level misdirection. Robustness-oriented work further uses latent adversarial training (Sheshadri et al., 2025), smoothness or invariance regularization (Fan et al., 2025a; Wang et al., 2025a), or optimizer simplification (Lang et al., 2026). Despite their different mechanisms, these methods are centered on suppressing a known target at release; a fuller review appears in $\ S _ { \mathrm { B } }$

The missing post-release gradient. When a retrospective method is repurposed for pre-release defense, the proxy set $\mathcal { D } _ { f }$ replaces $\mathcal { D } _ { u } .$ , and the optimized model is released as $\theta _ { \mathrm { r e l } }$ . The retrospective objective above may suppress forbidden responses at release, but it does not constrain the update induced by a fresh, unseen attacker set $\mathcal { D } _ { a }$ . This is the central mismatch: likelihood suppression controls the model’s current state at $\theta _ { \mathrm { r e l } }$ , whereas acquisition resistance depends on the directions in which downstream training can move that state. To isolate this distinction, we theoretically show below that the post-release gradient is the key diagnostic quantity: it explains why retrospective suppression can fail and identifies what a pre-release defense must control beyond current outputs.

Following common malicious fine-tuning settings (Tamirisa et al., 2025), we assume the attacker minimizes the supervised fine-tuning loss $\mathcal { L } _ { a } ( \bar { \theta ; \mathcal { D } } _ { a } ) : = \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \sim \mathcal { D } _ { a } } [ \ell _ { \pmb { \theta } } ( \mathbf { x } , \mathbf { y } ) ]$ ]. Our local analysis considers one gradient step $\pmb { \theta } ^ { + } = \pmb { \theta } _ { \mathrm { r e l } } - \eta \mathbf { g } _ { a }$ , where $\eta > 0$ and $\mathbf { g } _ { a } : = \nabla _ { \pmb { \theta } } \mathcal { L } _ { a } ( \pmb { \theta } _ { \mathrm { r e l } } ; \mathcal { D } _ { a } )$ . Since forget-related behavior and learning signals can concentrate in localized pathways (Cloud et al., 2024), consider the local orthogonal decomposition $\mathbb { R } ^ { d } = \pmb { S } \oplus \pmb { S } ^ { \perp }$ , where $s$ contains forbiddensensitive directions and $\mathcal { S } ^ { \perp }$ is its orthogonal complement. Let $\Pi _ { \mathcal { S } }$ denote the orthogonal projector onto $s$ and $S _ { \mathcal { F } }$ a differentiable surrogate of $\mathsf { P e r f } _ { \mathcal F } .$ . We adopt the idealized local-separation condition $\Pi _ { S } \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { \mathrm { r e l } } ) = \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { \mathrm { r e l } } )$ , so directions in $S ^ { \perp }$ have no first-order effect on the forbidden score. The following proposition links one-step acquisition to the component of the attacker gradient in ${ \mathcal { S } } .$

Proposition 3.1 (Future acquisition gain bound; proof deferred to §C.1). Suppose $S _ { \mathcal { F } }$ is L<sub>F</sub>-Lipschitz and $\beta _ { \mathcal { F } }$ -smooth in a neighborhood of $\theta _ { \mathrm { r e l } } ,$ , with $\mathrm { i } _ { S } \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { \mathrm { r e l } } ) = \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \bar { \pmb { \theta } _ { \mathrm { r e l } } } )$ . Forfixed $\mathcal { D } _ { a } ,$ define $\Delta S _ { \mathcal { F } } : = S _ { \mathcal { F } } ( \pmb { \theta } ^ { + } ) - S _ { \mathcal { F } } ( \pmb { \theta } _ { \mathrm { r e l } } )$ . For all sufficiently small $\eta > 0$ , this one-step acquisition gain satisfies:

$$
\Delta S _ { \mathcal { F } } = - \eta \left. \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { \mathrm { r e l } } ) , \mathbf { g } _ { a } \right. + O ( \eta ^ { 2 } ) \leq \eta L _ { \mathcal { F } } \left\| \Pi _ { S } \mathbf { g } _ { a } \right\| _ { 2 } + O ( \eta ^ { 2 } ) .
$$

Remark. Prop. 3.1 separates release-time suppression from susceptibility to future acquisition. Retrospective suppression can make $S _ { \mathcal { F } } ( \pmb { \theta } _ { \mathrm { r e l } } )$ small while leaving $\| \Pi _ { S } \mathbf { g } _ { a } \| _ { 2 }$ large; reducing this norm tightens the first-order upper bound on how rapidly fresh data can increase the forbidden score. This gap need not reflect a proxy–attacker data mismatch: even when $\mathcal { D } _ { a } = \mathcal { D } _ { f }$ , likelihood suppression alone does not ensure a small projected gradient. An example in §C.2 makes this separation explicit: two released models have identical forbidden and retain outputs, yet their projected gradients and one-step gains differ by controllable factors. Thus, gradient pathways can remain receptive to future acquisition after output suppression. §3.2 next investigates these predictions empirically.

## 3.2 RELEASE-TIME SUPPRESSION LEAVES FUTURE ACQUISITION PATHWAYS OPEN

Prior studies show that suppressed capabilities can re-emerge after downstream fine-tuning (Hu et al., 2025a). Echoing these robustness concerns, Prop. 3.1 highlights that a low release score need not imply a small acquisition gradient along forbidden-sensitive directions. We examine this gap in knowledge-absent preemptive setting by release failure, pathway prediction, and targeted intervention.

Common setup. On TOFU (Maini et al., 2024), we fine-tune Qwen3.5-2B (Qwen Team, 2026) on retain90 to obtain a retain-only Reference. For each of 20 target authors, we allocate 4/8/8 facts to $\mathcal { D } _ { f } , \mathcal { D } _ { a }$ , and holdout, respectively, yielding row-disjoint, same-author splits. Reference is never optimized on these target splits, isolating knowledge-absent preemptive unlearning: the defender uses $\mathcal { D } _ { f }$ to identify and suppress pathways that disjoint $\mathcal { D } _ { a }$ may later recruit. Following Wang et al. (2025b) and OpenUnlearning (Dorna et al., 2025), we measure empirical forbidden capability using extraction strength (ES ↑), defining $\widehat { \sf P e r f } _ { \mathcal F } ( \pmb \theta ; \mathcal { D } _ { a } ) : = \mathsf E \mathsf E ( \mathcal { D } _ { a } )$ . Full protocols appear in §E.1. Fig. 2 therefore moves from failure in Fig. 2a, to prediction in Fig. 2b, and to intervention in Fig. 2c.

![](images/4f007d327a917450e420d6dd21c30ab8a78ac8d0736d5d378cc15a9a66efaa79.jpg)  
(a) Release score is not resistance

![](images/4ef978cf8ed31a2360bcf689cc3f8d901fb48dd6973151118988e475a8b86beb.jpg)  
(b) Proxy ranks predict acquisition

![](images/23ab6412bb0b80ef761aee54410709b7a2a6871b71cce42bb2678c75fd30197a.jpg)  
(c) Attenuation reduces acquisition  
Figure 2: Empirical motivation for gradient sealing. (a) Release and post-acquisition ES for unlearning methods under identical acquisition. (b) Within-layer channel-rank alignment between $\mathcal { D } _ { f }$ look-ahead and $\mathcal { D } _ { a }$ acquisition. (c) Acquisition-gain reduction across selected-channel fractions and gradient-attenuation strengths.

Release-time suppression does not confer acquisition resistance. Starting from Reference, Sim-NPO (Fan et al., 2025b), SatImp (Yang et al., 2025b), NPO, and RMU produce distinct release states on $\mathcal { D } _ { f }$ retaining at least 95% of Reference utility. Fig. 2a connects release ES to its value after the same $\mathcal { D } _ { a }$ acquisition. Despite widely separated release scores, all method shows substantial acquisition: NPO rises from 0.112 to 0.326, and RMU from 0.032 to 0.257. Although their endpoints remain lower, suppression does not prevent subsequent acquisition, consistent with Prop. 3.1. Fig. 2b next asks where this residual trainability resides and whether $\mathcal { D } _ { f }$ can expose it before $\mathcal { D } _ { a }$ arrives.

A proxy look-ahead reveals where future acquisition will flow. Let ℓ index an MLP layer, j a channel, and $x \in \{ f , a \}$ a data split; $\hat { A } _ { x , \ell j } ^ { s }$ denotes that channel’s mean answer-token activation on $\mathcal { D } _ { x }$ in state s. A short $\mathcal { D } _ { f }$ look-ahead on a disposable copy defines $\Delta _ { \ell j } ^ { f } = | \bar { A } _ { f , \ell j } ^ { \mathrm { l o o k } } - \bar { A } _ { f , \ell j } ^ { 0 } |$ . After resetting to the Reference, independent $\mathcal { D } _ { a }$ acquisition defines $\Delta _ { \ell j } ^ { a } = | \bar { A } _ { a , \ell j } ^ { \mathrm { a \bar { c } q } } - \bar { A } _ { a , \ell j } ^ { \mathrm { 0 } } |$ . Here 0, look, and acq denote the Reference, look-ahead, and post-acquisition states. Fig. 2b plots their within-layer ranks on the horizontal and vertical axes, respectively, with darker hexagons containing more channels. The dark upper-right mass thus shows that channels most sensitive to proxy learning also tend to change most during actual acquisition. The pooled $\rho = 0 . 6 1 7$ , positive across all 24 layers, establishes a consistent predictive alignment between proxy sensitivity and subsequent acquisition. Fig. 2c next tests whether this alignment identifies a functional acquisition pathway rather than merely a correlate.

Targeted attenuation converts prediction into control. Using only the $\Delta _ { \ell j } ^ { f }$ ranking, we select the top p% channels per layer and scale their backward/update paths by $m = 1 - \alpha$ during $\mathcal { D } _ { a }$ acquisition, while preserving the forward pass and matching per-step update norms to Full-FT. Fig. 2c varies attenuation strength α horizontally and channel coverage p vertically. Each cell reports $\bar { R ( \alpha , p ) } = 1 0 0 ( { G _ { \mathrm { F T } } } - { G _ { \alpha , p } } ) \bar { / } { G _ { \mathrm { F T } } }$ , where $G = \mathsf E \mathsf E _ { \mathrm { { a f t e r } } } - \mathsf E \mathsf E _ { \mathrm { { b e f o r e } } }$ is the acquisition gain. Moving right strengthens attenuation, moving upward controls more predicted channels, and darker cells indicate less acquisition than Full-FT. The reduction generally grows toward the upper right, reaching 24.35% at $p = 1 0 \%$ and $\alpha = 1 0 0 \%$ ; under full attenuation, targeted masks also outperform layer- and count-matched random masks at every coverage. These interventions turn the predictive alignment in Fig. 2b into causal evidence that the localized pathways materially support acquisition.

Takeaways. Taken together, Fig. 2 forms a single chain: Fig. 2a identifies the failure of static suppression, Fig. 2b locates the residual acquisition route in advance, and Fig. 2c verifies that the localized route is a functional control point. A pre-release defense should therefore control where future optimization can flow, not only what the model currently expresses. Following this logic, GSU uses $\mathcal { D } _ { f }$ to expose acquisition-induced changes, localize sensitive gates, and seal their gradient pathways. $\ S \bar { 4 }$ next develops this full expose–localize–seal defense.

![](images/fceaf5cba01c9481a633e493c18c7dac9b9f8c1087eb25248d527425ad151baa.jpg)  
Figure 3: Overview of GSU. Expose reveals learning-induced changes by comparing the original model with a disposable proxy-fitted copy under matched contexts. Localize selects a fixed gate set G based on upward pre-activation shifts. Seal restarts from $\theta _ { o }$ and pushes selected pre-activations into the SiLU negative tail to attenuate local gradient factors, alongside response suppression and retain supervision, yielding $\pmb { \theta } _ { \mathrm { r e l } }$

## 4 GRADIENT-SEALED UNLEARNING

Our findings in $\ S 3$ motivate sealing acquisition pathways beyond suppressing release-time outputs. A direct approach penalizes acquisition-gradient norms, but differentiating this penalty introduces second-order derivatives and computational overhead (Zhao et al., 2022). GSU avoids this cost by controlling pre-activations, exploiting the zero or near-zero derivatives of ReLU-family activations for sufficiently negative inputs. Pushing pre-activations into this regime therefore attenuates local gradient factors using only first-order updates. Because indiscriminate control risks harming normal utility, we first identify acquisition-sensitive gates and restrict sealing to them. As shown in Fig. 3, we expose changes induced by proxy learning in §4.1, localize gates with upward pre-activation shifts in §4.2, and seal the selected gates jointly with response suppression and retain supervision in §4.3.

## 4.1 EXPOSE ACQUISITION-INDUCED CHANGES

Learning as a diagnostic. Under the formulation in §2, $\mathcal { D } _ { f }$ and $\mathcal { D } _ { a }$ are sampled from the same forbidden-domain distribution $\mathbb { P } _ { \mathcal { F } }$ but need not overlap. Since $\mathcal { D } _ { a }$ is unavailable before release, we use proxy learning on $\mathcal { D } _ { f }$ to expose gates that unseen acquisition may also recruit. Starting from $\theta _ { o } ,$ we briefly fine-tune a disposable copy using the supervised objective analyzed in §3.1:

$$
\begin{array} { r } { \operatorname* { m i n } _ { \pmb { \theta } } \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \sim \mathcal { D } _ { f } } \left[ \ell _ { \pmb { \theta } } ( \mathbf { x } , \mathbf { y } ) \right] . } \end{array}\tag{1}
$$

Let $\theta _ { \mathrm { l a } }$ denote the resulting look-ahead parameters. To determine which gates to target during sealing, we examine how their pre-activations respond to proxy learning. We therefore compare $\theta _ { o }$ and $\theta _ { \mathrm { l a } }$ on the same proxy examples under matched teacher-forcing contexts: both models receive $( \mathbf x , \mathbf y ^ { < _ { i } } )$ when predicting $y ^ { i }$ . For each such context, we write $u _ { \ell j } ^ { i } ( \pmb { \theta } ) \overline { { : } } = u _ { \ell j } ^ { i } ( \mathbf { x } , \mathbf { y } ^ { < i } ; \pmb { \theta } )$ for the gate pre-activation of channel $j$ in MLP layer ℓ used to predict $y ^ { i }$ . Matching the contexts isolates learning-induced pre-activation changes from differences in generated continuations, providing a consistent signal for selecting acquisition-sensitive gates in the next localization stage.

## 4.2 LOCALIZE DIRECTIONALLY RECRUITED GATES

Directional recruitment. To identify which gates to seal, we consider not only how much their pre-activations change, but also whether the changes oppose the intended intervention. Sealing aims to push pre-activations into sufficiently negative regions, where ReLU and smooth variants such as SiLU (Elfwing et al., 2017) have zero or near-zero derivatives. We instantiate this principle with the SiLU gates in the SwiGLU (Shazeer, 2020) MLPs of Llama and Qwen. Accordingly, we focus on gates whose pre-activations increase under proxy learning, counteracting the downward shift targeted by sealing. Their directional recruitment score is defined by

$$
r _ { \ell j } : = \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \sim \mathcal { D } _ { f } } \left[ \frac { 1 } { | \mathbf { y } | } \sum _ { i = 1 } ^ { | \mathbf { y } | } \mathrm { R e L U } \big ( u _ { \ell j } ^ { i } ( \theta _ { \mathrm { l a } } ) - u _ { \ell j } ^ { i } ( \theta _ { o } ) \big ) \right] .\tag{2}
$$

Applying ReLU before averaging prevents decreases at other positions from canceling increases. The score captures upward shifts even when both pre-activations remain negative. We next use these scores to select gates for sealing under layer and channel budgets.

Hierarchical selection. To compare recruitment across layers without favoring larger numerical scales, we normalize each layer’s total score by its original pre-activation magnitude. For a layer ℓ of width $d _ { \ell } .$ , define its magnitude $m _ { \ell }$ and relative recruitment score $R _ { \ell }$ as

$$
m _ { \ell } : = \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \sim \mathcal { D } _ { f } } \left[ \frac { 1 } { | \mathbf { y } | } \sum _ { i = 1 } ^ { | \mathbf { y } | } \sum _ { j = 1 } ^ { d _ { \ell } } \big | u _ { \ell j } ^ { i } ( \pmb { \theta } _ { o } ) \big | \right] , \qquad R _ { \ell } : = \frac { 1 } { m _ { \ell } } \sum _ { j = 1 } ^ { d _ { \ell } } r _ { \ell j } .\tag{3}
$$

The K layers with the highest positive $R _ { \ell }$ among those with $m _ { \ell } > 0$ form $\mathcal { T } ^ { \star }$ . Within each selected layer, the $\lceil p d _ { \ell } \rceil$ channels with the highest positive $r _ { \ell j }$ form $\mathcal { N } _ { \ell } ^ { \star }$ , where $p \in ( 0 , 1 ]$ ] controls the selected fraction. The resulting layer–channel index set $\mathcal { G } : = \{ ( \ell , j ) : \ell \in \mathcal { T } ^ { \star } , \ j \in \mathcal { N } _ { \ell } ^ { \star } \}$ identifies the gates selected for intervention, enabling us to turn localization into gradient control in the next stage.

## 4.3 SEAL SELECTED GRADIENT PATHWAYS

With G fixed, we discard $\theta _ { \mathrm { l a } }$ and restart full-parameter optimization from $\pmb { \theta } _ { o } .$ Proxy learning therefore contributes only the gate selection, not the fitted parameter changes. We now constrain the selected pre-activations to induce local gradient attenuation in the released model.

Sealing local gradients. For gate weights w and MLP input h, let $u = \mathbf { w } ^ { \top }$ h. By the chain rule, the gradient contribution through this gate is $c \phi ^ { \prime } ( u ) \mathbf { h }$ , where ϕ is SiLU and c combines the parallel-branch activation with the downstream gradient. Since $\vert \phi ^ { \prime } ( u ) \vert  0$ as $u  - \infty$ , pushing u sufficiently far into the negative tail makes this local derivative factor small, as illustrated in Fig. 3. This provides a first-order mechanism for gradient control through pre-activations. We therefore choose a shared threshold $\tau < 0$ in this tail and softly penalize violations of $u _ { \ell j } ^ { i } ( \pmb { \theta } ) \leq \tau \colon$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s e a l } } ( \pmb { \theta } ; \mathcal { D } _ { f } ) = \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \sim \mathcal { D } _ { f } } \left[ \frac { 1 } { | \mathbf { y } | } \sum _ { i = 1 } ^ { | \mathbf { y } | } \mathrm { R e L U } ^ { 2 } \big ( u _ { \ell j } ^ { i } ( \pmb { \theta } ) - \tau \big ) \right] . } \end{array}\tag{4}
$$

The shared threshold τ translates the small-derivative requirement into a common target in the negative tail. For each selected pre-activation u, the scalar penalty has derivative $2 \mathrm { R e L U } ( u - \tau )$ with respect to u, providing a downward signal proportional to the threshold violation. Once $u \leq \tau$ this contribution vanishes, so the seal term exerts no further direct downward pressure. The loss thus encourages entry into the low-derivative region through a soft constraint rather than hard clipping, allowing sealing to be balanced with utility preservation.

Coupling pathways and behavior. Sealing targets local gradient factors that output suppression does not explicitly constrain. To also suppress release-time forbidden responses and preserve normaldomain utility, we combine it with response suppression and retain supervision. The resulting GSU objective extends the suppression–retention formulation in §3.1, yielding

$$
\mathcal { L } _ { \mathrm { G S U } } ( \pmb \theta ; \mathcal { D } _ { f } , \mathcal { D } _ { r } ) = \mathcal { L } _ { \mathrm { s u p } } ( \pmb \theta ; \mathcal { D } _ { f } ) + \lambda _ { s } \mathcal { L } _ { \mathrm { s e a l } } ( \pmb \theta ; \mathcal { D } _ { f } ) + \lambda _ { r } \mathcal { L } _ { \mathrm { r e t } } ( \pmb \theta ; \mathcal { D } _ { r } ) ,\tag{5}
$$

where $\lambda _ { s } , \lambda _ { r } \geq 0$ control the strength of sealing and retention. We optimize this joint objective with first-order updates to obtain $\theta _ { \mathrm { r e l } }$ , without differentiating through the look-ahead stage.

GSU thus complements release-time suppression with targeted control of local gradient factors. When unseen acquisition reuses selected gates whose pre-activations remain in the negative tail, these factors remain small. §5 evaluates the resulting acquisition resistance and normal-domain utility.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Protection scenarios and benchmark construction. 1) Robust prevention: A model provider may wish to prevent downstream fine-tuning from incorporating private information that the released model has never learned. We simulate this setting with TOFU (Maini et al., 2024), using a retainonly checkpoint trained on the 180 retain90 authors. The remaining 20 forget10 authors define the forbidden domain; each contributes ten QA pairs to each of two pools, $B _ { 1 }$ and $B _ { 2 }$ (200 QA pairs each). The attacker uses $B _ { 2 } .$ , while the defender uses $B _ { 1 }$ in the disjoint setting and $B _ { 2 }$ in the identical setting. 2) Robust removal: A model may already contain hazardous knowledge that must be removed and kept from being restored through downstream fine-tuning. We study this setting on WMDP-Bio and WMDP-Cyber (Li et al., 2024), starting from a knowledgeable checkpoint. Each domain’s forget corpus is split into source-document pools $B _ { 1 }$ for defense and $B _ { 2 }$ for attack, with the full corresponding retain corpus supporting defense. We report only the disjoint setting: since the target knowledge is already present, identical defense and attack data would degrade to conventional robust unlearning.

Table 1: TOFU results of Llama3 and Qwen3.5 families under disjoint and identical prevention.
<table><tr><td></td><td></td><td colspan="3">Llama-3.2-1B</td><td colspan="3">Llama-3.2-3B</td><td colspan="3">Llama-3.1-8B</td><td colspan="3">Qwen3.5-2B</td><td colspan="3">Qwen3.5-4B</td><td colspan="3">Qwen3.5-9B</td></tr><tr><td></td><td>Method</td><td> $\overline { { \mathsf { E S } _ { 3 } } } \downarrow$ </td><td>ES5↓</td><td> $\mathsf { U A g } _ { 0 } \downarrow$ </td><td></td><td></td><td> $\overline { { { \mathsf { E S } } } } _ { 3 } \downarrow \mathsf { E S } _ { 5 } \downarrow \mathsf { U A } _ { 9 0 } \downarrow$ </td><td></td><td>ES3 ↓ ES5 ↓</td><td>UA90 ↓</td><td>ES3 ↓ ES5↓</td><td></td><td> $\mathsf { U A g o \downarrow }$ </td><td> $\overline { { \mathsf { E S } } } _ { 3 } \downarrow$ </td><td></td><td>ES5 ↓ UA90 ↓</td><td></td><td>ES3 ↓ ES5 ↓</td><td> $\mathsf { U A g o \downarrow }$ </td></tr><tr><td></td><td>No Defense</td><td>9.04</td><td>16.33</td><td>50.08</td><td>9.36</td><td>19.41</td><td>50.75</td><td>9.51</td><td>19.63</td><td>53.68</td><td>6.48</td><td>6.69</td><td>6.84</td><td>9.55</td><td>26.36</td><td>66.12</td><td>19.44 90.92</td><td></td><td>98.53</td></tr><tr><td>NPO</td><td>GradDiff</td><td>8.62</td><td>15.12</td><td>51.41</td><td>9.10</td><td>20.62</td><td>35.48</td><td>8.69</td><td>18.83</td><td>44.56</td><td>5.78</td><td>6.25</td><td>6.96</td><td>9.45</td><td>26.87</td><td>80.24</td><td>19.88 93.40</td><td></td><td>98.72</td></tr><tr><td>RMU</td><td></td><td>8.04</td><td>13.57</td><td>47.06</td><td>8.34</td><td>19.99</td><td>4.42</td><td>7.84</td><td>20.42</td><td>95.80</td><td>4.85</td><td>5.07</td><td>5.74</td><td>9.06</td><td>25.57</td><td>64.99</td><td>19.72</td><td>93.15</td><td>98.86</td></tr><tr><td>Disfn rnson</td><td></td><td>9.02</td><td>16.17</td><td>49.59</td><td>9.42</td><td>19.37</td><td>50.40</td><td>9.35</td><td>19.77</td><td>86.55</td><td>6.47</td><td>6.71</td><td>6.88</td><td>9.53</td><td>26.48</td><td>68.21</td><td>19.75</td><td>92.90</td><td>98.32</td></tr><tr><td>WGA</td><td></td><td>8.31</td><td>14.34</td><td>48.19</td><td>8.61</td><td>20.57</td><td>48.01</td><td>8.40</td><td>21.93</td><td>89.77</td><td>4.55</td><td>5.46</td><td>6.07</td><td>9.27</td><td>25.28</td><td>69.94</td><td>19.79</td><td>93.10</td><td>99.16</td></tr><tr><td>SatImp</td><td></td><td>9.02</td><td>17.22</td><td>55.32</td><td>9.48</td><td>23.06</td><td>55.90</td><td>8.52</td><td>24.66</td><td>91.11</td><td>5.93</td><td>6.07</td><td>6.35</td><td>9.26</td><td>26.94</td><td>94.69</td><td>20.85 93.05</td><td></td><td>99.36</td></tr><tr><td>RepNoise</td><td></td><td>8.67</td><td>16.69</td><td>53.16</td><td>9.05</td><td>23.08</td><td>40.49</td><td>8.80</td><td>21.60</td><td>86.93</td><td>5.54</td><td>6.12</td><td>6.84</td><td>9.48</td><td>24.76</td><td>63.72</td><td>19.71 92.95</td><td></td><td>98.53</td></tr><tr><td>ILU</td><td></td><td>7.14</td><td>12.93</td><td>30.52</td><td>8.15</td><td>20.42</td><td>4.47</td><td>8.42</td><td>21.60</td><td>87.12</td><td>4.93</td><td>5.40</td><td>5.95</td><td>8.24</td><td>25.06</td><td>61.86</td><td>19.8093.30</td><td></td><td>98.80</td></tr><tr><td>NPO+SAM</td><td></td><td>7.45</td><td>13.73</td><td>45.63</td><td>8.35</td><td>21.39</td><td>4.38</td><td>8.36</td><td>21.33</td><td>86.74</td><td>4.73</td><td>5.07</td><td>5.34</td><td>8.47</td><td>22.80</td><td>62.22</td><td>19.67</td><td>93.03</td><td>98.10</td></tr><tr><td></td><td>GSU (Ours)</td><td>6.60</td><td>12.40</td><td>27.87</td><td>8.21</td><td>19.89</td><td>3.77</td><td>7.78</td><td>18.03</td><td>86.17</td><td>4.06</td><td>4.44</td><td>5.03</td><td>8.28</td><td>21.81</td><td>59.40</td><td>19.63</td><td>92.85</td><td>98.39</td></tr><tr><td>Idennl Pon</td><td>GradDiff</td><td>7.42</td><td>11.62</td><td>49.31</td><td>7.94</td><td>13.57</td><td>13.57</td><td>8.76</td><td>16.46</td><td>47.82</td><td>4.75</td><td>5.36</td><td>5.68</td><td>8.26</td><td>17.80</td><td>69.17</td><td></td><td>17.50 84.07</td><td>98.60</td></tr><tr><td>NPO</td><td></td><td>5.51</td><td>8.51</td><td>20.60</td><td>5.94</td><td>9.71</td><td>2.81</td><td>5.73</td><td>8.93</td><td>2.74</td><td>3.38</td><td>3.62</td><td>4.15</td><td>6.30</td><td>9.35</td><td>38.22</td><td>11.31 66.67</td><td></td><td>98.22</td></tr><tr><td>RMU</td><td></td><td>9.03</td><td>15.85</td><td>48.43</td><td>9.25</td><td>19.44</td><td>50.75</td><td>9.39</td><td>19.70</td><td>54.91</td><td>6.50</td><td>6.65</td><td>6.84</td><td>9.60</td><td>26.67</td><td>67.12</td><td>19.2686.95</td><td></td><td>98.68</td></tr><tr><td>WGA</td><td></td><td>6.85</td><td>11.51</td><td>49.86</td><td>7.22</td><td>13.76</td><td>21.64</td><td>7.41</td><td>13.12</td><td>68.37</td><td>1.13</td><td>1.67</td><td>2.44</td><td>7.82</td><td>15.18</td><td>81.74</td><td>16.58 83.30</td><td></td><td>98.75</td></tr><tr><td>SatImp</td><td></td><td>8.86</td><td>17.08</td><td>54.27</td><td>9.23</td><td>23.88</td><td>56.27</td><td>8.65</td><td>22.53</td><td>89.14</td><td>5.86</td><td>6.15</td><td>6.36</td><td>9.32</td><td>27.46</td><td>90.31</td><td></td><td>20.14 85.18</td><td>98.66</td></tr><tr><td>RepNoise</td><td></td><td>8.37</td><td>14.89</td><td>44.87</td><td>8.44</td><td>18.20</td><td>51.19</td><td>8.78</td><td>17.92</td><td>76.43</td><td>5.48</td><td>5.85</td><td>6.24</td><td>9.39</td><td>23.58</td><td>78.31</td><td>19.09</td><td>86.67</td><td>98.53</td></tr><tr><td>ILU</td><td></td><td>5.90</td><td>9.30</td><td>20.54</td><td>6.24</td><td>11.48</td><td>3.19</td><td>5.86</td><td>8.92</td><td>5.66</td><td>3.37</td><td>3.61</td><td>4.39</td><td>5.80</td><td>9.83</td><td>3.21</td><td>11.23</td><td>66.56</td><td>95.60</td></tr><tr><td>NPO+SAM</td><td></td><td>5.28</td><td>8.17</td><td>17.56</td><td>5.48</td><td>8.61</td><td>2.90</td><td>5.79</td><td>7.28</td><td>3.04</td><td>3.04</td><td>3.60</td><td>4.04</td><td>6.21</td><td>8.09</td><td>30.75</td><td>9.43</td><td>48.08</td><td>95.20</td></tr><tr><td></td><td>GSU (Ours)</td><td>5.24</td><td>8.34</td><td>20.52</td><td>5.91</td><td>9.39</td><td>2.49</td><td>5.75</td><td>8.88</td><td>2.64</td><td>2.73</td><td>3.13</td><td>3.95</td><td>6.17</td><td>9.31</td><td>30.70</td><td>9.04</td><td>45.08</td><td>96.11</td></tr></table>

Models and baselines. On TOFU, we evaluate instruction-tuned Llama3-1B/3B/8B (Grattafiori et al., 2024; Meta AI, 2024) and Qwen3.5-2B/4B/9B (Qwen Team, 2026); on WMDP, we use Zephyr-7B-β (Tunstall et al., 2023). No Defense denotes the shared pre-defense checkpoint: the retain-only base on TOFU and the original checkpoint on WMDP. We compare 5 conventional unlearning methods—GradDiff (Maini et al., 2024), NPO (Zhang et al., 2024), RMU (Li et al., 2024), WGA (Wang et al., 2025c), and SatImp (Yang et al., 2025b)—and 3 robust defenses, RepNoise (Rosati et al., 2024), ILU (Wang et al., 2025a), and NPO+SAM (Fan et al., 2025a), all with retention.

Defense and attack settings. Defense uses 10 epochs with a base learning rate of $1 0 ^ { - 5 }$ on TOFU, and 80 updates at $4 \times 1 0 ^ { - 6 }$ on WMDP. All defense and attack runs use an effective batch size of 16. Following fine-tuning-based evaluations of tamper resistance and robust unlearning (Tamirisa et al., 2025; Fan et al., 2025a), we attack released models through supervised full-parameter fine-tuning on $\mathcal { D } _ { a } .$ TOFU attacks run for 10 epochs, using a learning rate calibrated once per model on No Defense and shared across methods; WMDP attacks run for 150 updates at $4 \times 1 0 ^ { - 6 }$

Evaluation metrics. We report early-mean and fifth-checkpoint forbidden scores: $\overline { { \mathsf { E S } } } _ { 3 }$ and $\mathsf { E S } _ { 5 }$ on TOFU use extraction strength (Wang et al., 2025b; Dorna et al., 2025) at attack epochs 1–3 and $5 ; \overline { { \mathsf { F } } } _ { 3 }$ and $\mathsf { F } _ { 5 }$ on WMDP use domain accuracy at steps 25/50/75 and 125. $\mathsf { U A } _ { 9 0 }$ is the maximum evaluated forbidden score with utility $\geq 0 . 9 U _ { \mathrm { r e f } }$ . For fair comparison, UWC (Wang et al., 2025b) calibrates all main-comparison releases to utility $\geq 0 . 9 5 U _ { \mathrm { r e f } }$ , so release utility is omitted from the tables. Scores are percentages (lower is better); bold and underline mark the best and second-best defended methods using unrounded values. §E provides full definitions and configurations.

## 5.2 MAIN COMPARISONS

Results on TOFU. GSU ranks first or second in 34 of 36 model–setting–metric comparisons in Tab. 1. Under disjoint prevention, it achieves the lowest $\mathsf { E S } _ { 5 }$ on five of six models and the lowest $\mathsf { U A } _ { 9 0 }$ on four, indicating resistance both at a fixed attack checkpoint and among utility-preserving attacks. For example, on Llama-3.2-1B, GSU reduces $\mathsf { U A } _ { 9 0 }$ from 50.08 without defense to 27.87, compared with 30.52 for the strongest baseline. Under identical prevention, GSU remains first or second in 17 of 18 comparisons. On Qwen3.5-9B, it achieves an $\mathsf { E S } _ { 5 }$ of 45.08, versus 48.08 for NPO+SAM and 90.92 without defense. These results demonstrate broad improvements across models and both prevention settings.

![](images/a3de9d99e29f4ec0f810df4a9a3452020e56edd7a9ce1791de069f8cfc1ed3cd.jpg)

![](images/ba0dec0aca9ec7f7fc1c1209e8efaa27cd0bb89904433b486575e113a3b2b445.jpg)

![](images/c4ebf0d3c0ee4f7c6af0ca9353d66a8c89cd05c58f009133123cd4f4ed92b372.jpg)  
Figure 4: Acquisition resistance, gate response, and benign learnability. (a) ES over ten attack epochs. (b) Mean absolute SiLU derivative over selected gates at release on $B _ { 1 }$ , $B _ { 2 }$ , and retain inputs. (c) Validation NLL reduction after two epochs of benign fine-tuning, interpreted relative to initial loss. Panels (a,b) use Qwen3.5-2B under disjoint TOFU; all results are single runs.

Results on WMDP. GSU ranks first or second across both domains and all three metrics in Tab. 2. It achieves the lowest $\mathsf { F } _ { 5 }$ on both Bio and Cyber, reducing accuracy to 34.13 and 30.23, respectively, compared with 65.36 and 43.83 without defense. These scores also improve over the strongest competing results of 34.83 on Bio and 31.08 on Cyber. GSU further attains the lowest ${ \overline { { \mathsf { F } } } } _ { 3 }$ on Cyber and the second-lowest $\mathsf { U A } _ { 9 0 }$ in both domains. Together, these results extend the acquisition-resistance gains observed on TOFU to hazardous-knowledge restoration under disjoint attacks.

Table 2: WMDP results under disjoint attacks.
<table><tr><td>Bio</td></tr><tr><td>Cyber Method  $\overline { { \mathsf { F } } } _ { 3 } \downarrow$   $\mathsf { F } _ { 5 } \downarrow$   $\mathsf { U A } _ { 9 0 }$  →  $\overline { { \mathsf { F } } } _ { 3 } \downarrow$   $\mathsf { F } _ { 5 } \downarrow$ </td></tr><tr><td> $\mathsf { U A } _ { 9 0 } \ \downarrow$  No Defense 65.23 65.36 65.99 43.7843.83 44.14</td></tr><tr><td></td></tr><tr><td>GradDiff 37.54 37.72 37.87 33.21 33.71 35.51</td></tr><tr><td>NPO 35.44 36.39 36.48 32.64 31.08 33.19</td></tr><tr><td>RMU 34.04 34.83 34.24 35.09 32.48 32.57 WGA 36.89 37.01 37.18 30.99 31.72 30.11</td></tr><tr><td>SatImp 39.03 39.19 39.22 33.93 34.31 33.84</td></tr><tr><td>RepNoise 36.18 35.69 35.71 34.42 234.99 34.37</td></tr><tr><td>ILU 39.81 39.98 39.93 31.98 33.09 31.87</td></tr><tr><td>NPO+SAM 38.28 38.47 38.53 35.51 35.56 34.99</td></tr><tr><td>GSU (Ours) 34.67 34.13 34.98 30.16 30.23 30.92</td></tr></table>

## 5.3 FURTHER ANALYSES

We connect acquisition resistance, selected-gate responses, and benign learnability. Under disjoint TOFU, the first two analyses use Qwen3.5-2B, while benign fine-tuning covers three models; §F details the frozen settings.

Acquisition under attack. In Fig. 4a, GSU has the lowest ES among the five methods at all eleven observed checkpoints. At epoch 5, it reaches 4.713%, versus 5.056% for NPO and 5.102% for NPO+SAM; at epoch 10, GSU and NPO+SAM reach 5.377% and 5.533%, respectively. This supports lower extraction throughout the measured attack budget. The gap narrows and GSU starts lower, so the comparison does not establish uniformly slower learning.

Release-stage gate response. Fig. 4b probes selected-gate responses at release. GSU reduces mean absolute SiLU derivatives relative to the original control by approximately 24% on $B _ { 1 }$ and 23% on $B _ { 2 }$ , versus 11% on retain inputs; without sealing stays closer to the control. This is consistent with targeted local attenuation; matched post-attack probes in §F.5 examine its partial reversal and residual differences.

Benign learnability. Fig. 4c tests whether benign learning remains possible. After two economics fine-tuning epochs, GSU on Llama-3.2-1B/3B and Qwen3.5-2B reduces validation NLL to within 0.005 of the No Defense controls. This supports retained benign optimization ability; larger reductions also reflect higher initial losses, not improved learning efficiency.

Additional results. §F.1 reports attack-time utility. §F.2 and §F.3 provide component ablations and sealing trajectories. §F.4 and §F.5 examine gate responses at release and after attack. §F.6 reports benign accuracy and NLL trajectories, while §F.7 tests protection after benign fine-tuning.

## 6 CONCLUSION

Preemptive unlearning requires controlling not only released behavior, but also what downstream optimization can acquire. GSU addresses this gap by exposing proxy-induced changes, localizing recruited gates, and sealing their local gradient pathways. Restarting from the pre-defense weights, it pushes selected pre-activations toward a shared SiLU negative-tail threshold alongside response suppression and retain supervision. The framework distinguishes prevention and reacquisition resistance from benign adaptability. Across the evaluated settings, GSU limits forbidden acquisition while retaining benign fine-tunability, with component ablations and matched gate probes supporting targeted sealing beyond output suppression alone.

## REFERENCES

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction. In NeurIPS, 2024.

Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, and Stella Biderman. LEACE: Perfect linear concept erasure in closed form. In NeurIPS, 2023.

Rishi Bommasani et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

Lucas Bourtoule, Varun Chandrasekaran, Christopher A. Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. Machine unlearning. In IEEE S&P, 2021.

ZiXuan Chen, Weikai Lu, Xin Lin, and Ziqian Zeng. SDD: Self-degraded defense against malicious fine-tuning. In ACL, 2025.

Alex Cloud, Jacob Goldman-Wetzler, Evžen Wybitul, Joseph Miller, and Alexander Matt Turner. Gradient routing: Masking gradients to localize computation in neural networks. arXiv preprint arXiv:2410.04332, 2024.

Vineeth Dorna, Anmol Mekala, Wenlong Zhao, Andrew McCallum, Zachary C. Lipton, J. Zico Kolter, and Pratyush Maini. OpenUnlearning: Accelerating LLM unlearning via unified benchmarking of methods and metrics. In NeurIPS D&B, 2025.

Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. arXiv preprint arXiv:1702.03118, 2017.

Chongyu Fan, Jinghan Jia, Yihua Zhang, Anil Ramakrishna, Mingyi Hong, and Sijia Liu. Towards LLM unlearning resilient to relearning attacks: A sharpness-aware minimization perspective and beyond. In ICML, 2025a.

Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, and Sijia Liu. Simplicity prevails: Rethinking negative preference optimization for LLM unlearning. In NeurIPS, 2025b.

Liam Fowl, Micah Goldblum, Ping-yeh Chiang, Jonas Geiping, Wojciech Czaja, and Tom Goldstein. Adversarial examples make strong poisons. In NeurIPS, 2021.

Shaopeng Fu, Fengxiang He, Yang Liu, Li Shen, and Dacheng Tao. Robust unlearnable examples: Protecting data privacy against adversarial learning. In ICLR, 2022.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. WildGuard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of LLMs. In NeurIPS, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In ICLR, 2021.

Robert Hönig, Javier Rando, Nicholas Carlini, and Florian Tramèr. Adversarial perturbations cannot reliably protect artists from generative AI. In ICLR, 2025.

Chia-Yi Hsu, Yu-Lin Tsai, Chih-Hsun Lin, Pin-Yu Chen, Chia-Mu Yu, and Chun-Ying Huang. Safe LoRA: The silver lining of reducing safety risks when finetuning large language models. In NeurIPS, 2024.

Shengyuan Hu, Yiwei Fu, Steven Wu, and Virginia Smith. Unlearning or obfuscating? jogging the memory of unlearned LLMs via benign relearning. In ICLR, 2025a.

Shengyuan Hu, Neil Kale, Pratiksha Thaker, Yiwei Fu, Steven Wu, and Virginia Smith. BLUR: A benchmark for LLM unlearning robust to forget-retain overlap. arXiv preprint arXiv:2506.15699, 2025b.

Hanxun Huang, Xingjun Ma, Sarah Monazam Erfani, James Bailey, and Yisen Wang. Unlearnable examples: Making personal data unexploitable. In ICLR, 2021.

Tiansheng Huang, Sihao Hu, and Ling Liu. Vaccine: Perturbation-aware alignment for large language models against harmful fine-tuning attack. In NeurIPS, 2024.

Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, and Ling Liu. Booster: Tackling harmful fine-tuning for large language models via attenuating harmful perturbation. In ICLR, 2025.

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama Guard: LLM-based input-output safeguard for human-AI conversations. arXiv preprint arXiv:2312.06674, 2023a.

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations. arXiv preprint arXiv:2312.06674, 2023b.

Joel Jang, Dongkeun Yoon, Sohee Yang, Sungmin Cha, Moontae Lee, Lajanugen Logeswaran, and Minjoon Seo. Knowledge unlearning for mitigating privacy risks in language models. In ACL, 2023.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7B, 2023.

Sayash Kapoor, Rishi Bommasani, Kevin Klyman, Shayne Longpre, Ashwin Ramaswami, Peter Cihon, Aspen K. Hopkins, Kevin Bankston, Stella Biderman, Miranda Bogen, et al. Position: On the societal impact of open foundation models. In ICML, 2024.

Jackson Kaunismaa, John Hughes, Christina Q. Knight, Avery Griffin, Mrinank Sharma, and Erik Jones. Eliciting harmful capabilities by fine-tuning on safeguarded outputs. In ICLR, 2026.

Yicheng Lang, Yihua Zhang, Chongyu Fan, Changsheng Wang, Jinghan Jia, and Sijia Liu. Downgrade to upgrade: Optimizer simplification enhances robustness in LLM unlearning. In ICLR, 2026.

Simon Lermen, Charlie Rogers-Smith, and Jeffrey Ladish. LoRA fine-tuning efficiently undoes safety training in Llama 2-Chat 70B. arXiv preprint arXiv:2310.20624, 2023.

Kemou Li, Qizhou Wang, Yue Wang, Fengpeng Li, Jun Liu, Bo Han, and Jiantao Zhou. LLM unlearning with LLM beliefs. In ICLR, 2026.

Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, Justin D. Li, Ann-Kathrin Dombrowski, Shashwat Goel, Gabriel Mukobi, et al. The WMDP benchmark: Measuring and reducing malicious use with unlearning. In ICML, 2024.

Chumeng Liang, Xiaoyu Wu, Yang Hua, Jiaru Zhang, Yiming Xue, Tao Song, Zhengui Xue, Ruhui Ma, and Haibing Guan. Adversarial example does good: Preventing painting imitation from diffusion models via adversarial examples. In ICML, 2023.

Junfeng Liao, Qizhou Wang, Shanshan Ye, Xin Yu, Ling Chen, and Zhen Fang. Explainable LLM unlearning through reasoning. In ICLR, 2026.

Aengus Lynch, Phillip Guo, Aidan Ewart, Stephen Casper, and Dylan Hadfield-Menell. Eight methods to evaluate robust unlearning in LLMs. arXiv preprint arXiv:2402.16835, 2024.

Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, and J. Zico Kolter. TOFU: A task of fictitious unlearning for LLMs. In COLM, 2024.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In NeurIPS, 2022.

Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a transformer. In ICLR, 2023.

Meta AI. Llama 3.2: Revolutionizing edge AI and vision with open, customizable models. Meta AI release documentation, 2024.

Kyle O’Brien, Stephen Casper, Quentin Anthony, Tomek Korbak, Robert Kirk, Xander Davies, Ishan Mishra, Geoffrey Irving, Yarin Gal, and Stella Biderman. Deep ignorance: Filtering pretraining data builds tamper-resistant safeguards into open-weight LLMs. arXiv preprint arXiv:2508.06601, 2025.

Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to! In ICLR, 2024.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen. ai/blog?id=qwen3.5.

Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. Null it out: Guarding protected attributes by iterative nullspace projection. In ACL, 2020.

Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, David Atanasov, Robie Gonzales, Subhabrata Majumdar, Carsten Maple, Hassan Sajjad, and Frank Rudzicz. Representation noising: A defence mechanism against harmful finetuning. In NeurIPS, 2024.

Shawn Shan, Emily Wenger, Jiayun Zhang, Huiying Li, Haitao Zheng, and Ben Y. Zhao. Fawkes: Protecting privacy against unauthorized deep learning models. In USENIX Security, 2020.

Shawn Shan, Jenna Cryan, Emily Wenger, Haitao Zheng, Rana Hanocka, and Ben Y. Zhao. Glaze: Protecting Artists from Style Mimicry by Text-to-Image Models. In USENIX Security, 2023.

Shawn Shan, Wenxin Ding, Josephine Passananti, Stanley Wu, Haitao Zheng, and Ben Y. Zhao. Nightshade: Prompt-specific poisoning attacks on text-to-image generative models. In IEEE S&P, 2024.

Noam Shazeer. GLU variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

William Shen, Xinchi Qiu, Meghdad Kurmanji, Alexandru-Andrei Iacob, Lorenzo Sani, Yihong Chen, Nicola Cancedda, and Nicholas Lane. Llm unlearning via neural activation redirection. In NeurIPS, 2026.

Abhay Sheshadri, Aidan Ewart, Phillip Huang Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, et al. Latent adversarial training improves robustness to persistent harmful behaviors in LLMs. Transactions on Machine Learning Research, 2025.

Weijia Shi, Jaechan Lee, Yangsibo Huang, Sadhika Malladi, Jieyu Zhao, Ari Holtzman, Daogao Liu, Luke Zettlemoyer, Noah A. Smith, and Chiyuan Zhang. MUSE: Machine unlearning six-way evaluation for language models. In ICLR, 2025.

Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, et al. Tamper-resistant safeguards for open-weight LLMs. In ICLR, 2025.

Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, et al. Zephyr: Direct distillation of LM alignment. arXiv preprint arXiv:2310.16944, 2023.

Thanh Van Le, Hao Phung, Thuan Hoang Nguyen, Quan Dao, Ngoc Tran, and Anh Tran. Anti-DreamBooth: Protecting users from personalized text-to-image synthesis. In ICCV, 2023.

Changsheng Wang, Yihua Zhang, Jinghan Jia, Parikshit Ram, Dennis Wei, Yuguang Yao, Soumyadeep Pal, Nathalie Baracaldo, and Sijia Liu. Invariance makes LLM unlearning resilient even to unanticipated downstream fine-tuning. In ICML, 2025a.

Qizhou Wang, Bo Han, Puning Yang, Jianing Zhu, Tongliang Liu, and Masashi Sugiyama. Towards effective evaluations and comparisons for LLM unlearning methods. In ICLR, 2025b.

Qizhou Wang, Jin Peng Zhou, Zhanke Zhou, Saebyeol Shin, Bo Han, and Kilian Q. Weinberger. Rethinking LLM unlearning objectives: A gradient perspective and go beyond. In ICLR, 2025c.

Yaxuan Wang, Jiaheng Wei, Chris Yuhao Liu, Jinlong Pang, Quan Liu, Ankit Parag Shah, Yujia Bao, Yang Liu, and Wei Wei. LLM unlearning via loss adjustment with only forget data. In ICLR, 2025d.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Nakyeong Yang, Dong-Kyum Kim, Jea Kwon, Minsung Kim, Kyomin Jung, and Meeyoung Cha. Erase or hide? suppressing spurious unlearning neurons for robust unlearning. In ICLR, 2026.

Puning Yang, Qizhou Wang, Zhuo Huang, Tongliang Liu, Chengqi Zhang, and Bo Han. Exploring criteria of loss reweighting to enhance LLM unlearning. In ICML, 2025b.

Yuanshun Yao, Xiaojun Xu, and Yang Liu. Large language model unlearning. In NeurIPS, 2024.

Chenlong Zhang, Zhuoran Jin, Hongbang Yuan, Jiaheng Wei, Tong Zhou, Kang Liu, Jun Zhao, and Yubo Chen. RULE: Reinforcement unlearning achieves forget-retain pareto optimality. In NeurIPS, 2025.

Ruiqi Zhang, Licong Lin, Yu Bai, and Song Mei. Negative preference optimization: From catastrophic collapse to effective unlearning. In COLM, 2024.

Yang Zhao, Hao Zhang, and Xiuyuan Hu. Penalizing gradient norm for efficiently improving generalization in deep learning. In ICML, 2022.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to AI transparency. arXiv preprint arXiv:2310.01405, 2023.

## Appendix of Gradient-Sealed Unlearning

## CONTENTS

A Notations 15   
B Detailed Related Work 15   
B.1 Retrospective LLM Unlearning and Robustness 16   
B.2 Open-Weight Safety and Tamper-Resistant Safeguards 16   
B.3 Data-Side Preemptive Protection 16   
B.4 Representation Localization and Pathway Control 16   
C Proofs and Supporting Analysis 17   
C.1 Proof of Proposition 3.1 . 17   
C.2 A Simple ReLU Counterexample 18   
D Pseudocode 19   
E Further Experimental Setup 20   
E.1 Mechanism Study for Figure 2 20   
E.2 Benchmark Construction 21   
E.3 Evaluation Metrics 21   
E.4 Models and Initialization 22   
E.5 Common Defense and Attack Configuration 22   
F Additional Results 22   
F.1 Utility during Acquisition . 23   
F.2 Complete Component Ablation 24   
F.3 Paired Sealing Ablation . 25   
F.4 Release-Stage Gate Distributions and Negative-Tail Occupancy . 25   
F.5 Selected-Gate Responses before and after Attack 27   
F.6 Benign Adaptation: Accuracy and Validation Loss . 28   
F.7 Resistance after Benign Fine-Tuning 29   
G Limitation and Future Work 30

## OVERVIEW OF THE APPENDIX

This appendix supplies notation, related work, proofs, pseudocode, experimental settings, and supplementary analyses supporting the main paper.

• §A summarizes the notation.

• §B reviews the four closest research threads.

• §C proves Prop. 3.1 and gives a minimal counterexample.

• §D presents GSU pseudocode.

• §E details data splits, metrics, implementations, and optimization.

• §F connects acquisition and utility to component ablations and gate responses, then examines benign learning and protection after adaptation.

• §G discusses scope and limitations.

## A NOTATIONS

This section summarizes the main notations in Tab. 3.

Table 3: Core notation used in the paper.
<table><tr><td>Notation</td><td>Description</td></tr><tr><td> $\mathbf x , \mathbf y$ </td><td>Prompt and autoregressive response.</td></tr><tr><td> $\pi _ { \boldsymbol { \theta } } , \ell _ { \boldsymbol { \theta } }$ </td><td>Response likelihood and response NLL.</td></tr><tr><td> $\mathcal { F } , \mathcal { R } ; \mathbb { P } _ { \mathcal { F } } , \mathbb { P } _ { \mathcal { R } }$ </td><td>Forbidden and normal domains, and their data distributions.</td></tr><tr><td> $\mathcal { D } _ { f } , \mathcal { D } _ { a } , \mathcal { D } _ { r }$ </td><td>Defender proxy, unseen attacker, and retain sets.</td></tr><tr><td> $\theta _ { o } , \theta _ { \mathrm { l a } } , \theta _ { \mathrm { r e l } } , \theta _ { \mathrm { a t k } }$ </td><td>Original, look-ahead, released, and post-attack parameters.</td></tr><tr><td> $\mathcal { U } , A , \mathfrak { A }$ </td><td>Pre-release defense, acquisition attack, and attack class.</td></tr><tr><td> $\mathsf { P e r f } _ { \mathcal { F } } , \mathsf { P e r f } _ { \mathcal { R } }$ </td><td>Forbidden capability and normal utility; hats denote empirical estimates.</td></tr><tr><td> $\epsilon _ { f } , \epsilon _ { u } , \epsilon _ { r }$ </td><td>Forbidden-capability ceiling, usable-utility floor, and allowed release-time utility drop.</td></tr><tr><td> $\mathcal { U } _ { T } , \cup \mathsf { A } _ { T }$ </td><td>Usable attack checkpoints through T and their maximum empirical forbidden capability.</td></tr><tr><td> $\mathcal { L } _ { a } , \mathbf { g } _ { a }$ </td><td>Attacker loss and its gradient at  $\pmb { \theta } _ { \mathrm { r e l } } .$ </td></tr><tr><td> $S _ { \mathcal { F } } , \Delta S _ { \mathcal { F } }$ </td><td>Differentiable forbidden-domain score and its one-step acquisition gain.</td></tr><tr><td> $ { \boldsymbol { S } } , \Pi _ {  { \boldsymbol { s } } }$ </td><td>Forbidden-sensitive parameter subspace and its orthogonal projector.</td></tr><tr><td> $u _ { \ell j } ^ { i } ( \pmb { \theta } )$ </td><td>Gate pre-activation at layer l, channel j, under context  $( \mathbf { x } , \mathbf { y } ^ { < i } )$  used to predict  $y ^ { i }$ </td></tr><tr><td> $\phi , \phi ^ { \prime }$ </td><td>SiLU activation and its derivative.</td></tr><tr><td> $d _ { \ell } , K _ { \mathrm { l a } } , K , p$ </td><td>Layer width, look-ahead steps, selected layer count, and within-layer channel fraction.</td></tr><tr><td> $r _ { \ell j } , m _ { \ell } , R _ { \ell }$ </td><td>Directional channel score, original layer magnitude, and normalized layer score.</td></tr><tr><td> $\mathcal { T } ^ { \star } , \mathcal { N } _ { \ell } ^ { \star } , \mathcal { G }$ </td><td>Selected layers, selected channels within layer  $\ell ,$  and the fixed layer-channel gate set.</td></tr><tr><td> $\tau < 0$ </td><td>Shared pre-activation threshold in the SiLU negative tail.</td></tr><tr><td> $\mathcal { L } _ { \mathrm { s u p } } , \mathcal { L } _ { \mathrm { r e t } }$ </td><td>Response suppression and normal-domain retention losses.</td></tr><tr><td> $\mathcal { L } _ { \mathrm { s e a l } } , \mathcal { L } _ { \mathrm { G S U } }$ </td><td>Selected-gate seal penalty and joint defense objective.</td></tr><tr><td> $\lambda _ { s } , \lambda _ { r }$ </td><td>Sealing and retention weights.</td></tr></table>

## B DETAILED RELATED WORK

This section separates four neighboring research threads. §B.1 reviews retrospective LLM unlearning and its robustness, §B.2 covers open-weight and tamper-resistant safeguards, §B.3 discusses dataside preemptive protection, and §B.4 summarizes representation localization and pathway control. Together, these threads motivate resisting unseen future learning in released weights.

## B.1 RETROSPECTIVE LLM UNLEARNING AND ROBUSTNESS

Retrospective LLM unlearning extends the broader goal of removing a designated training influence from an already trained model (Bourtoule et al., 2021) to knowledge and behaviors encoded by LLMs. Most methods optimize a release-time forget–retain trade-off. At the output level, GA maximizes forget-set loss (Jang et al., 2023; Yao et al., 2024), GradDiff couples forgetting with retain training (Maini et al., 2024), and NPO uses a reference-relative preference objective (Zhang et al., 2024). Subsequent variants simplify or rebalance these losses, including SimNPO (Fan et al., 2025b), WGA (Wang et al., 2025c), SatImp (Yang et al., 2025b), forget-only loss adjustment (Wang et al., 2025d), and reinforcement-based unlearning (Zhang et al., 2025). A complementary line intervenes inside the network: RMU misdirects target representations (Li et al., 2024), activation redirection changes internal states (Shen et al., 2026), Ssiuu regularizes spurious unlearning neurons (Yang et al., 2026), and reasoning- or belief-guided objectives target model-generated alternatives rather than only labeled answers (Liao et al., 2026; Li et al., 2026).

Benchmarks such as TOFU (Maini et al., 2024), WMDP (Li et al., 2024), and MUSE (Shi et al., 2025) evaluate privacy, copyright, and hazardous-knowledge removal, while overlap-aware settings such as BLUR test whether forgetting remains meaningful when forget and retain distributions intersect (Hu et al., 2025b). However, a low target score immediately after unlearning can reflect suppression or obfuscation rather than durable removal. Prompting, probing, relearning, and benign downstream fine-tuning can recover apparently forgotten behavior (Lynch et al., 2024; Hu et al., 2025a), motivating robustness-enhancing approaches based on latent adversarial training (Sheshadri et al., 2025), sharpness-aware optimization (Fan et al., 2025a), downstream invariance (Wang et al., 2025a), and optimizer simplification (Lang et al., 2026). Our WMDP regime directly evaluates this removal-plusresistance problem. Preemptive unlearning is broader in one crucial respect: it also covers a target capability absent from the original checkpoint and asks whether unseen future data can acquire it after release, rather than only whether previously encoded behavior can be recovered.

## B.2 OPEN-WEIGHT SAFETY AND TAMPER-RESISTANT SAFEGUARDS

Weight access allows downstream users to bypass inference-time moderation by directly changing the model. Aligned LLMs can lose safety after malicious or even benign fine-tuning (Qi et al., 2024; Lermen et al., 2023), and safeguarded outputs may themselves provide supervision for eliciting harmful capabilities (Kaunismaa et al., 2026). Input–output guard models such as Llama Guard and WildGuard (Inan et al., 2023b; Han et al., 2024) remain useful at deployment but cannot prevent weight-level adaptation. Model-side defenses include representation noising (Rosati et al., 2024), tamper-resistant safeguards (Tamirisa et al., 2025), perturbation-aware alignment (Huang et al., 2024; 2025), procedure-constrained defenses (Hsu et al., 2024; Chen et al., 2025), and filtered pretraining (O’Brien et al., 2025). These works mainly preserve broad safety alignment or refusal behavior. Preemptive unlearning instead asks whether clean, disjoint future data can acquire a designated capability while the released model retains normal utility.

## B.3 DATA-SIDE PREEMPTIVE PROTECTION

Unlearnable examples perturb training data so that a learner cannot easily acquire their semantics (Huang et al., 2021; Fowl et al., 2021; Fu et al., 2022). Related privacy and copyright defenses include Fawkes for face recognition (Shan et al., 2020), Anti-DreamBooth for personalized generation (Van Le et al., 2023), and artwork protections such as Glaze and Nightshade (Liang et al., 2023; Shan et al., 2023; 2024). Adaptive evaluations show that perturbation-based protection can be brittle under robust preprocessing or retraining (Hönig et al., 2025). The key distinction is control: data-side methods assume that the protected examples are those later used for training, whereas our defender cannot modify or even observe the attacker’s future data. GSU therefore encodes resistance into the released weights using a separate proxy set.

## B.4 REPRESENTATION LOCALIZATION AND PATHWAY CONTROL

Model editing localizes and modifies internal computations associated with facts (Meng et al., 2022; 2023); concept erasure removes linearly represented information (Ravfogel et al., 2020; Belrose et al., 2023); and representation engineering identifies low-dimensional directions associated with high-level behavior (Zou et al., 2023; Arditi et al., 2024). Gradient routing further shows that data-dependent learning signals can be localized to computational pathways (Cloud et al., 2024). GSU differs in objective: it does not merely change a current output or erase a linearly decodable feature, but targets the gate derivatives through which future domain data would update the model.

## C PROOFS AND SUPPORTING ANALYSIS

This section provides the proof and supporting analysis for Prop. 3.1. §C.1 first derives the futureacquisition bound with an explicit second-order remainder; §C.2 then gives a minimal ReLU counterexample showing how identical release-time scores can conceal different acquisition receptivity and how activation suppression reduces it.

## C.1 PROOF OF PROPOSITION 3.1

Proposition 3.1 (Future acquisition gain bound; proof deferred to $\ S { \bf C } . 1 )$ . Suppose $S _ { \mathcal { F } }$ is $L _ { \mathcal { F } ^ { - } } L _ { l }$ pschitz and $\beta _ { \mathcal { F } }$ -smooth in a neighborhood of $\theta _ { \mathrm { r e l } } ,$ , with $\mathrm { i } _ { S } \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { \mathrm { r e l } } ) = \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \bar { \pmb { \theta } _ { \mathrm { r e l } } } )$ . Forfixed $\mathcal { D } _ { a } ,$ define $\Delta S _ { \mathcal { F } } : = S _ { \mathcal { F } } ( \pmb { \theta } ^ { + } ) - S _ { \mathcal { F } } \mathrm { ( } \pmb { \theta } _ { \mathrm { r e l } } )$ . For all sufficiently small $\eta > 0$ , this one-step acquisition gain satisfies:

$$
\Delta S _ { \mathcal { F } } = - \eta \left. \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { \mathrm { r e l } } ) , \mathbf { g } _ { a } \right. + O ( \eta ^ { 2 } ) \leq \eta L _ { \mathcal { F } } \left\| \Pi _ { S } \mathbf { g } _ { a } \right\| _ { 2 } + O ( \eta ^ { 2 } ) .
$$

Proof. Fix $\mathcal { D } _ { a }$ and abbreviate

$$
\begin{array} { r } { \pmb { \theta } _ { 0 } : = \pmb { \theta } _ { \mathrm { r e l } } , \qquad \mathbf { g } : = \mathbf { g } _ { a } . } \end{array}
$$

Since $\mathbf { g }$ is evaluated at $\pmb { \theta } _ { 0 }$ for the fixed dataset $\mathcal { D } _ { a }$ , it is independent of the step size $\eta .$ . Let $\mathcal { N }$ be a neighborhood of $\pmb { \theta } _ { 0 }$ on which $S _ { \mathcal { F } }$ is both L<sub>F</sub>-Lipschitz and $\beta _ { \mathcal { F } }$ -smooth. Because g is fixed, there exists $\eta _ { 0 } > 0$ such that, for every $\eta \in ( 0 , \eta _ { 0 } ]$ , the entire line segment

$$
\{ \pmb \theta _ { 0 } - t \eta \mathbf g : t \in [ 0 , 1 ] \}
$$

is contained in ${ \mathcal { N } } .$

Applying the fundamental theorem of calculus along this segment gives

$$
\begin{array} { r l } & { \Delta S _ { \mathcal F } = S _ { \mathcal F } ( \pmb \theta _ { 0 } - \eta \mathbf g ) - S _ { \mathcal F } ( \pmb \theta _ { 0 } ) } \\ & { \qquad = - \eta \displaystyle \int _ { 0 } ^ { 1 } \langle \nabla _ { \pmb \theta } S _ { \mathcal F } ( \pmb \theta _ { 0 } - t \eta \mathbf g ) , \mathbf g \rangle \ \mathrm d t } \\ & { \qquad = - \eta \langle \nabla _ { \pmb \theta } S _ { \mathcal F } ( \pmb \theta _ { 0 } ) , \mathbf g \rangle + R _ { \eta } , } \end{array}
$$

where

$$
R _ { \eta } : = - \eta \int _ { 0 } ^ { 1 } \langle \nabla _ { \theta } S _ { \mathcal { F } } ( \pmb \theta _ { 0 } - t \eta \mathbf { g } ) - \nabla _ { \theta } S _ { \mathcal { F } } ( \pmb \theta _ { 0 } ) , \mathbf { g } \rangle \ \mathrm d t .
$$

By the Cauchy–Schwarz inequality and the $\beta _ { \mathcal { F } }$ -smoothness of $S _ { \mathcal { F } }$ ,

$$
\begin{array} { l } { | R _ { \eta } | \leq \eta \displaystyle \int _ { 0 } ^ { 1 } \| \nabla _ { \pmb \theta } S _ { \mathcal F } ( \pmb \theta _ { 0 } - t \eta \mathbf g ) - \nabla _ { \pmb \theta } S _ { \mathcal F } ( \pmb \theta _ { 0 } ) \| _ { 2 } \| \mathbf g \| _ { 2 } \mathrm { d } t } \\ { \displaystyle \leq \eta \displaystyle \int _ { 0 } ^ { 1 } \beta _ { \mathcal F } \| t \eta \mathbf g \| _ { 2 } \| \mathbf g \| _ { 2 } \mathrm { d } t } \\ { \displaystyle = \frac { \beta _ { \mathcal F } } { 2 } \eta ^ { 2 } \| \mathbf g \| _ { 2 } ^ { 2 } . } \end{array}
$$

Since $\mathbf { g } = \mathbf { g } _ { a }$ is fixed with respect to η, the preceding estimate implies that $R _ { \eta } = O ( \eta ^ { 2 } )$ . Therefore,

$$
\Delta S _ { \mathcal { F } } = - \eta \left. \nabla _ { \theta } S _ { \mathcal { F } } ( \pmb { \theta } _ { \mathrm { r e l } } ) , \mathbf { g } _ { a } \right. + O ( \eta ^ { 2 } ) ,
$$

which proves the first-order expansion.

We next bound its linear term. Since $S _ { \mathcal { F } }$ is differentiable and locally L<sub>F</sub>-Lipschitz at $\pmb { \theta } _ { 0 }$ , for every unit vector v and all sufficiently small nonzero h,

$$
\frac { | S _ { \mathcal { F } } ( \pmb \theta _ { 0 } + h \mathbf { v } ) - S _ { \mathcal { F } } ( \pmb \theta _ { 0 } ) | } { | h | } \le L _ { \mathcal { F } } .
$$

Taking $h  0$ and using differentiability yields

$$
| \langle \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { 0 } ) , \mathbf { v } \rangle | \leq L _ { \mathcal { F } } .
$$

Taking the supremum over all $\| \mathbf { v } \| _ { 2 } = 1$ gives

$$
\begin{array} { r } { \| \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { 0 } ) \| _ { 2 } \leq L _ { \mathcal { F } } . } \end{array}
$$

By the local-separation condition,

$$
\Pi _ { S } \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { 0 } ) = \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { 0 } ) .
$$

Moreover, because $\Pi _ { \cal S }$ is an orthogonal projector, it is self-adjoint. Hence,

$$
\begin{array} { r } { \langle \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { 0 } ) , \mathbf { g } \rangle = \langle \Pi _ { S } \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { 0 } ) , \mathbf { g } \rangle } \\ { = \langle \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { 0 } ) , \Pi _ { S } \mathbf { g } \rangle . } \end{array}
$$

It follows from the Cauchy–Schwarz inequality that

$$
\begin{array} { r l } & { - \left. \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { 0 } ) , \mathbf { g } \right. = - \left. \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { 0 } ) , \Pi _ { s } \mathbf { g } \right. } \\ & { \quad \quad \quad \leq \left| \left. \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { 0 } ) , \Pi _ { s } \mathbf { g } \right. \right| } \\ & { \quad \quad \quad \leq \left\| \nabla _ { \pmb { \theta } } S _ { \mathcal { F } } ( \pmb { \theta } _ { 0 } ) \right\| _ { 2 } \left\| \Pi _ { s } \mathbf { g } \right\| _ { 2 } } \\ & { \quad \quad \quad \leq L _ { \mathcal { F } } \left\| \Pi _ { s } \mathbf { g } \right\| _ { 2 } . } \end{array}
$$

Combining this inequality with the explicit remainder estimate gives

$$
\Delta S _ { \mathcal { F } } \leq \eta L _ { \mathcal { F } } \left. \Pi _ { S } \mathbf { g } _ { a } \right. _ { 2 } + \frac { \beta _ { \mathcal { F } } } { 2 } \eta ^ { 2 } \Vert \mathbf { g } _ { a } \Vert _ { 2 } ^ { 2 } .
$$

For fixed $\mathcal { D } _ { a } .$ , the second term is $O ( \eta ^ { 2 } )$ , and therefore

$$
\begin{array} { r } { \Delta S _ { \mathcal { F } } \leq \eta L _ { \mathcal { F } } \left. \Pi _ { S } \mathbf { g } _ { a } \right. _ { 2 } + O ( \eta ^ { 2 } ) , } \end{array}
$$

which completes the proof.

## C.2 A SIMPLE RELU COUNTEREXAMPLE

We give a minimal construction where a zero release score coexists with a nonzero acquisition gradient, while activation suppression reduces that gradient.

Let $\mathrm { R e L U } ( z ) = \operatorname* { m a x } \{ z , 0 \}$ act element-wise and consider

$$
\mathbf { h } _ { u } ( \mathbf { x } ) = \mathrm { R e L U } ( \mathbf { A } _ { u } \mathbf { x } ) , \qquad f _ { \theta } ( \mathbf { x } ) = \mathbf { w } ^ { \top } \mathbf { h } _ { u } ( \mathbf { x } ) , \qquad \mathbf { A } _ { u } = \left( \begin{array} { l l } { u } & { - 1 } \\ { - 1 } & { 1 } \end{array} \right) ,
$$

where $\mathbf { w } = ( w _ { f } , w _ { r } ) ^ { \top }$ and $\pmb { \theta } = ( w _ { f } , u , w _ { r } ) ^ { \top }$ . As illustrated in Fig. 5a, the first hidden unit forms the forbidden-sensitive pathway, parameterized by $( w _ { f } , u )$ , while the second forms an independent retain pathway controlled by $w _ { r } .$ . The fixed entries −1 isolate these pathways on the two basis inputs below and keep inactive pre-activations away from the ReLU kink. Accordingly, let

$$
S = \{ ( \alpha , \beta , 0 ) ^ { \top } : \alpha , \beta \in \mathbb { R } \} .
$$

For the forbidden and retain inputs ${ \bf x } _ { f } = { \bf e } _ { 1 }$ and ${ \bf x } _ { r } = { \bf e } _ { 2 }$ , respectively, any $u > 0$ gives

$$
\mathbf { h } _ { u } ( \mathbf { x } _ { f } ) = ( u , 0 ) ^ { \top } , \qquad \mathbf { h } _ { u } ( \mathbf { x } _ { r } ) = ( 0 , 1 ) ^ { \top } ,
$$

and therefore

$$
S _ { \mathcal { F } } ( \pmb \theta ) : = f _ { \pmb \theta } ( \mathbf { x } _ { f } ) = w _ { f } u , \qquad f _ { \pmb \theta } ( \mathbf { x } _ { r } ) = w _ { r } .
$$

The two rows of Fig. 5b trace output suppression and additional activation suppression from their released states to their one-step gains:

$$
\theta _ { \mathrm { o u t } } = ( 0 , 1 , 1 ) ^ { \top } , \qquad \theta _ { \mathrm { a c t } } = ( 0 , \delta , 1 ) ^ { \top } , \qquad 0 < \delta < 1 .
$$

Both released states satisfy

$$
S _ { \mathcal { F } } ( \pmb { \theta } _ { \mathrm { o u t } } ) = S _ { \mathcal { F } } ( \pmb { \theta } _ { \mathrm { a c t } } ) = 0 , \qquad f _ { \pmb { \theta } _ { \mathrm { o u t } } } ( \mathbf { x } _ { r } ) = f _ { \pmb { \theta } _ { \mathrm { a c t } } } ( \mathbf { x } _ { r } ) = 1 ,
$$

although their forbidden-sensitive activations are 1 and δ, respectively.

For this illustrative regression model, the attacker minimizes squared loss, equivalent up to an additive constant to the NLL of a unit-variance Gaussian with mean $f _ { \pmb { \theta } } ( \mathbf { x } _ { f } ) ;$

$$
\mathcal { L } _ { a } ( \pmb \theta ) = \frac { 1 } { 2 } \left( f _ { \pmb \theta } ( \mathbf { x } _ { f } ) - 1 \right) ^ { 2 } = \frac { 1 } { 2 } ( w _ { f } u - 1 ) ^ { 2 } .
$$

For $u > 0 ,$ , its gradient is

$$
{ \bf g } _ { a } ( \pmb \theta ) = ( w _ { f } u - 1 ) \left( \begin{array} { c } { { u } } \\ { { w _ { f } } } \\ { { 0 } } \end{array} \right) .
$$

Consequently,

$$
\mathbf { g } _ { a } ^ { \mathrm { o u t } } = ( - 1 , 0 , 0 ) ^ { \top } , \qquad \mathbf { g } _ { a } ^ { \mathrm { a c t } } = ( - \delta , 0 , 0 ) ^ { \top } .
$$

Both gradients lie in $s ,$ yielding the projected norms shown in the third column of Fig. 5b:

$$
\left\| \Pi _ { S } \mathbf { g } _ { a } ^ { \mathrm { o u t } } \right\| _ { 2 } = 1 , \qquad \left\| \Pi _ { S } \mathbf { g } _ { a } ^ { \mathrm { a c t } } \right\| _ { 2 } = \delta .
$$

After one gradient step with the same step size $\eta ,$

$$
\theta _ { \mathrm { o u t } } ^ { + } = ( \eta , 1 , 1 ) ^ { \top } , \qquad \theta _ { \mathrm { a c t } } ^ { + } = ( \eta \delta , \delta , 1 ) ^ { \top } ,
$$

giving the exact acquisition gains shown in the final column of Fig. 5b:

$$
\Delta S _ { \mathcal { F } } ^ { \mathrm { o u t } } = \eta , \qquad \Delta S _ { \mathcal { F } } ^ { \mathrm { a c t } } = \eta \delta ^ { 2 } .
$$

The first factor of $\delta$ comes from the smaller attacker gradient and the second from the remaining activation; both released models preserve a retain output of one.

Finally, for $u > 0 ,$

$$
\begin{array} { r } { \nabla _ { \pmb \theta } S _ { \mathcal { F } } ( \pmb \theta ) = ( u , w _ { f } , 0 ) ^ { \top } \in \mathcal { S } , } \end{array}
$$

so the local-separation condition holds. Both released states lie strictly away from the ReLU kink; in sufficiently small neighborhoods around them, $S _ { \mathcal { F } } ( \pmb { \theta } ) = w _ { f } u$ is locally Lipschitz and 1-smooth. Thus, this regression construction satisfies the local regularity and separation conditions of Prop. 3.1, while showing that identical release-time scores can conceal different acquisition receptivity. The choice $0 < \delta < 1$ gives a continuous $\delta ^ { 2 }$ separation while remaining away from the ReLU kink; moving the forbidden gate strictly below zero would additionally set its ReLU derivative to zero, matching the low-slope sealing mechanism that GSU applies during defense optimization.

## D PSEUDOCODE

Algorithm 1: Gradient-Sealed Unlearning (GSU)   
Input: Original parameters $\theta _ { o } ;$ proxy and retain sets $\mathcal { D } _ { f } , \mathcal { D } _ { r } ;$ look-ahead steps $K _ { \mathrm { l a } } ;$ selection parameters   
$K ,$ p; threshold $\tau < 0 ;$ loss weights $\lambda _ { s } , \lambda _ { r } .$   
Output: Released parameters $\pmb { \theta } _ { \mathrm { r e l } } .$   
// Expose   
1 Fit a disposable copy of $\pmb { \theta } _ { o }$ on $\mathcal { D } _ { f }$ for $K _ { \mathrm { l a } }$ steps using Eq. (1), obtaining $\theta _ { \mathrm { l a } } ;$   
2 Collect both models’ pre-activations under matched teacher-forcing contexts $( \mathbf { x } , \mathbf { y } ^ { < i } ) ;$   
// Localize   
3 Compute $r _ { \ell j } , m _ { \ell } ,$ and $R _ { \ell }$ using Eqs. (2) and (3);   
4 Select up to K layers with the highest positive $R \ell$ among those with m<sub>ℓ</sub> $> 0 ,$ forming $\boldsymbol { \mathcal { T } ^ { \star } } ;$   
5 Within each selected layer, select up to $\lceil p d _ { \ell } \rceil$ channels with the highest positive $r _ { \ell j }$ , forming $\mathcal { N } _ { \ell } ^ { \star }$   
6 $\mathcal { G }  \{ ( \ell , j ) : \ell \in \mathcal { T } ^ { \star } , \ \bar { j } \in \mathcal { N } _ { \ell } ^ { \star } \} ;$   
// Seal   
7 Fix G, discard $\theta _ { \mathrm { l a } } ,$ , and reset $\mathbf { \nabla } \theta \gets \theta _ { o } ;$   
8 for each defense update do   
9 Sample minibatches $\boldsymbol { B } _ { f } \subseteq \mathcal { D } _ { f }$ and $\boldsymbol { B } _ { r } \subseteq \mathcal { D } _ { r } ;$   
10 Compute the joint loss in Eq. (5) on $( B _ { f } , B _ { r } )$ , with $\mathcal { L } _ { \mathrm { s e a l } } = 0 \mathrm { i f } \mathcal { G } = \emptyset ;$   
11 Update all parameters θ with a first-order optimizer step;   
12 return $\theta _ { \mathrm { r e l } }  \theta ;$

(a) Two-path ReLU network  
![](images/4b7ab4df012b8126289dd0c7adb1330ebd0e55908e9f15d6c7319c34da5f9622.jpg)

(b) One-step acquisition from identical release outputs  
![](images/a419785125d193887cf68442d4be76987cd16b684a8d10008c2ebf7f40999634.jpg)  
Figure 5: A minimal ReLU counterexample. Panel (a) shows the forbidden-sensitive path in red and the retain path in gray; dashed arrows are fixed inhibitory connections. For $\mathbf { x } _ { f } = \mathbf { e } _ { 1 }$ and ${ \bf x } _ { r } = { \bf e } _ { 2 }$ , both released states in (b) have identical forbidden and retain outputs. Reducing the forbidden activation from 1 to δ reduces the projected attacker-gradient norm by δ and the exact one-step acquisition gain by $\delta ^ { 2 }$

## E FURTHER EXPERIMENTAL SETUP

## E.1 MECHANISM STUDY FOR FIGURE 2

Reference, splits, and metrics. This mechanism study uses the official Qwen3.5-2B snapshot and a retain-only Reference trained for five epochs on all 3,600 retain90 examples (AdamW, peak learning rate $1 0 ^ { - 5 }$ , seed 0). The Reference receives no optimizer exposure to $\mathcal { D } _ { f } , \mathcal { D } _ { a } .$ , or holdout. Within each of the 20 target authors, a fixed $4 / 8 / 8$ split gives 80 proxy, 160 attacker, and 160 holdout facts; the splits have no exact row overlap but share authors and domain structure. Forbidden capability is evaluated only on $\mathcal { D } _ { a }$ using ES.

Panel (a): SimNPO, SatImp, NPO, and RMU start from the same Reference and run for 10 release epochs (50 optimizer steps, learning rate $1 0 ^ { - 5 }$ , effective batch size 16, AdamW, weight decay 0.01, seed 0) using raw checkpoints without UWC or interpolation. The acquisition learning rate is calibrated once on the Reference and then applied unchanged to every plotted release state: 10 epochs/100 steps, constant learning rate $4 \times 1 0 ^ { - 6 }$ , effective batch size 16, and seed 0, with no early stopping or best-epoch selection.

Panels (b,c): For Panel (b), a disposable Reference copy receives one epoch (10 steps) of $\mathcal { D } _ { f }$ supervised look-ahead at learning rate $5 \times 1 0 ^ { - 6 }$ and effective batch size 8. We record answer-token mean gate\_proj pre-activations before and after look-ahead, discard the copy, reload the unchanged Reference, and independently acquire on $\mathcal { D } _ { a }$ for 10 epochs at learning rate $\mathrm { \dot { 1 } 0 ^ { - 6 } }$ and effective batch size 16. Both sensitivity arrays are converted to within-layer ranks before pooling. Panel (c) uses the same $\mathcal { D } _ { a }$ schedule (100 steps, seed 0, weight decay 0) and nested layer-quota masks at $0 . 5 / 1 / 2 / 5 / 1 0 \%$ coverage. For each selected channel, attenuation scales the corresponding gate\_proj/up\_proj rows, down\_proj column, and post-Adam update slice by $1 - a ;$ every attenuated arm is matched step-wise to Full-FT in whole-model update norm. The grid comprises one Full-FT control and 20 localized arms; the 0% column repeats that control. Let $G _ { p , a }$ be the 10-epoch increase in $F$ for coverage $p$ and attenuation a. Each cell reports $R _ { p , a } = 1 0 \dot { 0 } ( G _ { \mathrm { F T } } - G _ { p , a } \big ) / G _ { \mathrm { F T } }$ , the percentage reduction in acquisition gain relative to Full-FT.

## E.2 BENCHMARK CONSTRUCTION

TOFU. We use the official forget10/retain90 partition: 20 forbidden authors with 400 QA pairs and 180 retain authors with 3,600 QA pairs. A fixed seed-0 ordering partitions each forbidden author’s 20 QA pairs equally between $B _ { 1 }$ and $B _ { 2 } ,$ yielding 200 examples per pool. In the disjoint setting, $( \mathcal { D } _ { f } , \mathcal { D } _ { a } \bar { ) } = ( \bar { B _ { 1 } } , \bar { B _ { 2 } } )$ ; in the identical setting, $( \bar { D _ { f } } , \bar { D _ { a } } ) = \bar { ( B _ { 2 } , B _ { 2 } ) }$ . Thus, attack training and ES evaluation use the same $B _ { 2 }$ across settings, while disjoint defense uses non-overlapping QA records about the same authors. The disjoint split does not establish unseen-author or semantically disjoint-fact generalization. All 3,600 retain examples form $\mathcal { D } _ { r }$

WMDP. We split each domain’s forget corpus in source order: the first $\lfloor N / 2 \rfloor$ documents form the defender pool $B _ { 1 } .$ , and the remainder form the attacker pool $B _ { 2 }$ . The Bio pools contain 12,226 and 12,227 documents; the Cyber pools contain 500 documents each. We evaluate only disjoint attacks, with $\left( \mathcal { D } _ { f } , \mathcal { D } _ { a } \right) = \left( B _ { 1 } , B _ { 2 } \right)$ . Since the target knowledge is already present, identical defense and attack data would reduce this setting to conventional robust unlearning. Defense uses the full corresponding retain corpus, containing 60,887 Bio or 4,473 Cyber documents. Each pool is independently concatenated and tokenized into 512-token sequences using the OpenUnlearning preprocessing pipeline. The official Bio and Cyber test sets contain 1,273 and 1,987 multiple-choice questions, respectively; utility is evaluated on all 14,042 MMLU test questions across 57 subjects.

## E.3 EVALUATION METRICS

Extraction strength. For an answer of L tokens, extraction strength (ES) is the length of its longest suffix whose tokens are all correctly predicted by teacher-forced argmax decoding, divided by $L .$ We average this score over the 200 attacker examples. Let ES denote the score after attack epoch $t ,$ with t = 0 denoting release. The main TOFU table reports

$$
\overline { { \mathsf { E S } } } _ { 3 } = \frac { 1 } { 3 } \sum _ { t = 1 } ^ { 3 } \mathsf { E S } _ { t } , \qquad \mathsf { E S } _ { 5 } ,
$$

which summarize early acquisition and extraction after five attack epochs.

WMDP accuracy. We evaluate domain accuracy and sample-weighted MMLU accuracy at attack steps s ∈ {25, 50, 75, 100, 125, 150}. Writing $\mathsf { F } _ { j }$ for domain accuracy at step $2 5 j$ , we report

$$
\overline { { \mathsf { F } } } _ { 3 } = \frac { 1 } { 3 } \sum _ { j = 1 } ^ { 3 } \mathsf { F } _ { j } , \qquad \mathsf { F } _ { 5 } .
$$

Thus, the early mean uses steps 25, 50, and 75, while the fifth-checkpoint score uses step 125.

Utility and release calibration. TOFU utility is HM(MU, Fluency), where MU aggregates answer probability, ROUGE-L recall, and truth ratio over the retain, real-author, and world-fact evaluation sets. These sets contain 400, 100, and 117 examples, respectively; fluency uses the 200 canonical $B _ { 2 }$ questions. WMDP utility is sample-weighted MMLU accuracy (Hendrycks et al., 2021). The reference utility $U _ { \mathrm { r e f } }$ is measured on the shared retain-only base for TOFU and the original checkpoint for WMDP. For the main comparisons, Unlearning with Control (UWC) (Wang et al., 2025b) calibrates each final defense checkpoint against its reference checkpoint to satisfy $\mathrm { \bar { \it U } _ { r e l } } \geq 0 . 9 5 { \it U } _ { \mathrm { r e f } }$ before attack. Release utility is therefore omitted from the main tables. The separate mechanism and further-analysis protocols specify their own checkpoint handling in §E.1 and $\ S \mathrm { F }$

Usable acquisition. For both benchmarks, we report

$$
\mathsf { U A } _ { 9 0 } = \operatorname* { m a x } _ { \substack { t \in \mathscr { T } } } C _ { t } ,
$$

where $C _ { t }$ is the forbidden score and $U _ { t }$ is utility at the same checkpoint. For TOFU, $C _ { t } = \mathsf { E } S _ { t }$ and $\mathcal { T } = \{ 0 , 1 , \ldots , 1 0 \}$ indexes attack epochs, including release. For WMDP, $C _ { t }$ is domain accuracy and $\tilde { \mathcal { T } } = \{ 2 5 , 5 0 , \tilde { 7 } 5 , 1 0 0 , 1 2 5 , 1 5 0 \}$ indexes attack steps. The 90% threshold determines attackcheckpoint eligibility and is distinct from the 95% release-calibration requirement. The reference utility remains fixed throughout the attack, and attack checkpoints are not recalibrated. All main-table scores are percentages, with lower values indicating stronger protection.

## E.4 MODELS AND INITIALIZATION

The TOFU checkpoints are Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, and Qwen3.5-2B/4B/9B. For each model, we construct one retain-only base using five epochs on retain90 (1,125 updates), AdamW, learning rate $1 0 ^ { - 5 }$ , batch size 16, weight decay 0.01, and a one-epoch warmup followed by linear decay. This base is shared across methods and both settings. WMDP defenses share the original zephyr-7b-beta checkpoint.

## E.5 COMMON DEFENSE AND ATTACK CONFIGURATION

The following settings describe the main comparisons unless otherwise specified. All runs use seed 0, bfloat16, a maximum sequence length of 512, and effective batch size 16, with gradient accumulation as needed. TOFU defense and attack each use ten epochs over 200 examples, giving 130 optimizer updates with the final partial batch retained. Defense uses a base learning rate of $\bar { 1 } 0 ^ { - 5 }$ and draws paired retain batches from $\mathcal { D } _ { r }$ . Both stages use AdamW with weight decay 0.01, a constant learning rate, and no warmup. The fixed attack learning rates for Llama-3.2-1B/3B and Llama-3.1-8B are $( 1 0 ^ { - 5 } , 1 0 ^ { - 5 } , 5 \times 1 \dot { 0 } ^ { - 6 } )$ , and those for Qwen3. $5 { - } 2 \tt B / 4 \tt B / 9 \tt B$ are $( 1 0 ^ { - 6 } , 5 \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } )$ . We evaluate the release checkpoint and every attack epoch.

WMDP uses paged 32-bit AdamW with zero weight decay. Defense runs for 80 updates at learning rate $4 \times 1 0 ^ { - 6 }$ , with 16 warmup updates followed by a constant rate; the final defense checkpoint is UWC-calibrated before release. Attack runs for 150 updates on the attacker corpus at constant learning rate $4 \times 1 0 ^ { - 6 }$ without warmup, with evaluation every 25 updates.

## F ADDITIONAL RESULTS

Analysis roadmap. We connect acquisition performance to component contributions, selected-gate responses, and benign adaptability. §F.1 first places the extraction advantage alongside utility. §F.2 and §F.3 then examine the component design and sealing trajectories. §F.4 and §F.5 probe the corresponding gate responses at release and after attack. Finally, §F.6 and §F.7 evaluate benign learning and the protection retained after that adaptation.

Shared analysis protocol. The analyses in Fig. 4 and this section use disjoint TOFU, with $\mathcal { D } _ { f } = B _ { 1 }$ and $\mathcal { D } _ { a } = B _ { 2 }$ Each pool contains 200 non-overlapping QA pairs from the same 20 forbidden authors; the retain set contains 3,600 QA pairs from 180 other authors. This record-disjoint split does not establish unseen-author or semantically disjoint-fact generalization. Attack and gate analyses use Qwen3.5-2B; benign fine-tuning additionally covers Llama-3.2-1B/3B. All attack trajectories include their pre-attack checkpoint and ten evaluated attack epochs, using full-parameter fine-tuning on $B _ { 2 } .$ , learning rate $1 0 ^ { - 6 }$ , and effective batch size 16. All observations are single runs, without multi-seed error bars or significance tests. Shared controls and different measurements from the same runs are not independent replications.

Metrics and recorded fields. The definitions in §E.3 apply throughout. ES is read from the stored forbidden.components.es field and multiplied by 100 for percentage display. Utility is the perf\_r composite score, not a single task accuracy. Trajectory figures and their endpoint tables retain the 100× utility display scale; §F.2 instead reports raw composite scores. These scales represent the same utility metric and should not be read as task accuracies. The trajectory plots and complete ablation do not apply a reference-utility threshold or compute $\mathsf { U A } _ { 9 0 }$

Frozen configurations. For each model, GSU uses the configuration selected by mean ES over attack epochs 1–5 among completed disjoint candidates when the analysis plan was frozen. Configurations are not retuned for individual panels; baselines use the analysis plan’s defaults. These fixed configurations can differ from the main table’s metric-specific candidate selections, so the corresponding numerical results need not coincide. Because selection uses existing $B _ { 2 }$ attack evaluations, these are development-stage diagnostics, not independent-test or equal-tuning-budget comparisons. Tab. 4 lists the frozen settings. All use a defense learning rate of $1 0 ^ { - 5 }$ and 100 look-ahead updates.

Table 4: Frozen configurations for the supplementary analyses. “Seal alpha” retains the terminology of the experiment records.
<table><tr><td>Model</td><td> $K$ </td><td> $p$ </td><td>Seal alpha</td><td>T</td></tr><tr><td>Llama-3.2-1B</td><td>4</td><td>5%</td><td>0.01</td><td>-9</td></tr><tr><td>Qwen3.5-2B</td><td>4</td><td>5%</td><td>0.03</td><td>-4</td></tr><tr><td>Llama-3.2-3B</td><td>2</td><td>5%</td><td>0.01</td><td>-4</td></tr></table>

## F.1 UTILITY DURING ACQUISITION

Evaluating extraction alongside utility. We first ask whether the extraction advantage in Fig. 4a is accompanied by retained utility. Fig. 6 adds utility at the same checkpoints to the existing five-method ES trajectories; it is a complementary measurement of those runs. All eleven observed checkpoints are retained without smoothing, utility filtering, or checkpoint selection.

![](images/8137d1071712f65e84102c10f1c00a17cd7304dac61c1f62455ee0bba91f51a2.jpg)  
Figure 6: Utility accompanying the acquisition trajectory. Left: the ES observations from Fig. 4a, with GSU below all displayed controls at every checkpoint. Right: utility at the same checkpoints on the 100× composite-score scale, without utility filtering.

Lower extraction with retained utility. GSU maintains lower ES than all four displayed controls throughout the measured attack, while its utility stays above No Defense and RMU. At epoch 10, these three methods have utility scores of 69.955, 68.485, and 68.511 on the plotted scale, respectively, as reported in Tab. 5. This supports an acquisition advantage that is not explained by uniformly lower measured utility. Relative to NPO+SAM, GSU trades slightly lower final utility (69.955 versus 70.599) for lower final ES (5.377% versus 5.533%); it does not dominate both metrics. GSU also begins with lower extraction: its ES increases by 1.997 percentage points over the attack, versus 0.803 for NPO+SAM. The evidence concerns lower attainable extraction within this budget, not a slower acquisition rate or a $\mathsf { U A } _ { 9 0 }$ ranking. We next examine which components contribute to this behavior.

Table 5: Extraction and utility during the shared attack. ES entries are percentages; $\overline { { \mathsf { E S } } } _ { 3 }$ averages epochs 1–3. U is shown on the 100× composite-score scale, not as task accuracy.
<table><tr><td>Method</td><td> $\mathsf { E S } _ { 0 }$ </td><td> $\overline { { \mathsf { E S } } } _ { 3 }$ </td><td> $\mathsf { E S } _ { 5 }$ </td><td> $\mathsf { E S } _ { 1 0 }$ </td><td> $\mathrm { U _ { 0 } }$ </td><td> $\mathrm { U } _ { 1 0 }$ </td></tr><tr><td>No Defense</td><td>6.016</td><td>6.463</td><td>6.659</td><td>6.868</td><td>69.871</td><td>68.485</td></tr><tr><td>NPO</td><td>4.599</td><td>4.838</td><td>5.056</td><td>5.782</td><td>70.691</td><td>69.220</td></tr><tr><td>RMU</td><td>6.127</td><td>6.446</td><td>6.616</td><td>6.851</td><td>69.944</td><td>68.511</td></tr><tr><td>NPO+SAM</td><td>4.730</td><td>4.965</td><td>5.102</td><td>5.533</td><td>71.663</td><td>70.599</td></tr><tr><td>GSU</td><td>3.379</td><td>3.889</td><td>4.713</td><td>5.377</td><td>70.169</td><td>69.955</td></tr></table>

## F.2 COMPLETE COMPONENT ABLATION

Setup and metrics. To examine the design behind the acquisition advantage, we compare five completed variants under the frozen Qwen3.5-2B, disjoint TOFU, seed-0 setting: full GSU, without sealing, without suppression, without retention, and random gates. The variants follow the component definitions in $\ S 4$ . Their GSU and without-seal controls are shared with the trajectory analysis in $\ S \mathrm { F } . 3 ,$ , rather than independently repeated. Fig. 7 reports release utility and $\overline { { \mathsf { E S } } } _ { 5 } = \textstyle { \frac { 1 } { 5 } } \sum _ { t = 1 } ^ { 5 } \mathsf { E S } _ { t }$ . This early-attack mean excludes epoch 0 and differs from both the main table’s three-epoch mean and $\mathsf { U A } _ { 9 0 }$ . Utility is the raw perf\_r composite score, not an accuracy or a percentage; Tab. 6 also reports release and final-attack endpoints.

![](images/49bc8581f67f2130f64fb0669d9f1021d1239e7158de24de43f15f2ea660a964.jpg)  
Figure 7: Complete component ablation on Qwen3.5-2B under disjoint TOFU. Left: mean ES over attack epochs 1–5, in percent. Right: release utility as a raw composite score. Full GSU lowers early extraction relative to the three nonzero-utility ablations; without retention, utility is zero, so low ES alone is not evidence of a successful utility-preserving defense. Results use one training seed.

Complementary component contributions. Full GSU achieves a mean ES of 4.1740%, compared with 4.9598% without sealing, 5.0298% without suppression, and 5.2196% with random gates. The reductions of 0.786, 0.856, and 1.046 percentage points support the contributions of suppression, sealing, and the frozen gate selector to lower early acquisition. GSU also has lower ES than these three variants at epoch 10, although the gaps narrow. Its release utility is comparable to, but slightly below, without-seal and random-gate utility (0.701687 versus 0.705906 and 0.707940), and above without suppression (0.683498). These results support lower early acquisition at comparable measured utility, not strict dominance across every metric; without suppression has higher final utility.

Table 6: Complete ablation measurements. ES entries are percentages; $\mathsf { U } _ { 0 }$ and $\mathrm { U } _ { 1 0 }$ are unscaled composite utility scores, not accuracies. $\overline { { \mathsf { E S } } } _ { 5 }$ averages attack epochs 1–5.
<table><tr><td>Variant</td><td> $\mathsf { E S } _ { 0 }$ </td><td> $\overline { { \mathsf { E S } } } _ { 5 }$ </td><td> $\mathsf { E S } _ { 1 0 }$ </td><td> $\mathrm { U } _ { \mathrm { 0 } }$ </td><td> $\mathrm { U } _ { 1 0 }$ </td></tr><tr><td>GSU</td><td>3.3793</td><td>4.1740</td><td>5.3768</td><td>0.701687</td><td>0.699545</td></tr><tr><td>Without seal</td><td>4.6104</td><td>4.9598</td><td>5.7530</td><td>0.705906</td><td>0.695012</td></tr><tr><td>Without suppression</td><td>4.1924</td><td>5.0298</td><td>5.5130</td><td>0.683498</td><td>0.705820</td></tr><tr><td>Without retain</td><td>0.0000</td><td>0.1997</td><td>0.7997</td><td>0.000000</td><td>0.000000</td></tr><tr><td>Random gates</td><td>4.3565</td><td>5.2196</td><td>5.7881</td><td>0.707940</td><td>0.695478</td></tr></table>

Why retention matters. Removing retention yields utility of zero at both endpoints. Its low ES therefore illustrates the need to assess acquisition suppression jointly with utility, rather than an advantage over full GSU. Zero composite utility does not imply that every individual capability is absent. The random-gate comparison supports the selected configuration but uses only one random realization, not repeated-sampling evidence. The component evidence is specific to this single-seed setting; it does not establish statistical significance or transfer to other models, identical data, or WMDP. The next subsection resolves the sealing comparison across the full attack trajectory.

## F.3 PAIRED SEALING ABLATION

From component summaries to attack trajectories. We now examine when the sealing contribution in §F.2 is visible. Fig. 8 expands the same GSU and without-seal controls into release evaluation and ten attack epochs on Qwen3.5-2B. The shared disjoint protocol is unchanged; this is a temporal view of the paired comparison, not an additional independent ablation.

![](images/bd7d89a5d656ad518ec1479d90e448e1a2d24f880f935d57c816e54bc76a2c79.jpg)

![](images/d863d70f2ef07b07d5b3ea916bf32482adf200842583c8648120c2c32705448a.jpg)  
Figure 8: Sealing contributions across the attack trajectory. The GSU and without-seal runs from §F.2 are shown at all eleven observed checkpoints. GSU retains lower ES (left), while the narrowing gap and accompanying utility (right, 100× composite-score scale) show how the comparison evolves.

A benefit throughout the observed budget. GSU and without sealing begin at 3.379% and 4.610% ES, respectively, reach 4.713% and 5.323% at epoch 5, and finish at 5.377% and 5.753%. Thus, the sealing advantage remains present at every observed checkpoint, including a 0.376-percentage-point gap after ten attack epochs. Final utility is also slightly higher for GSU (69.955 versus 69.501 on the plotted scale). The trajectories complement the early-mean ablation by showing that its benefit is not confined to one selected checkpoint. Because GSU already starts lower and the gap narrows, this supports lower extraction over the measured budget, rather than a smaller acquisition rate or permanent resistance. We next probe the selected gates for internal changes consistent with this contribution.

## F.4 RELEASE-STAGE GATE DISTRIBUTIONS AND NEGATIVE-TAIL OCCUPANCY

Measurement scope and aggregation. To examine the local changes accompanying the sealing benefit, we supplement Fig. 4b with gate-wise pre-activation distributions and negative-tail occupancy at release. Measurements cover Original, released GSU, and the released without-seal variant on $B _ { 1 }$ $B _ { 2 }$ , and retain inputs, before any attacker updates. “Original” denotes the control checkpoint entering this analysis, not necessarily original pretrained weights. Each group contains 1,232 recorded selected gates. Both $B _ { 1 }$ and $B _ { 2 }$ use all 200 examples, while retain uses all 3,600 examples. For each gate, we first average over answer-token positions within an example, then weight examples equally. Writing $\mathcal { G } _ { \mathrm { r e c } }$ for the recorded gates and S for an input set,

$$
\bar { u } _ { g , S } ( \pmb \theta ) = \frac { 1 } { | S | } \sum _ { ( \mathbf x , \mathbf y ) \in S } \frac { 1 } { | \mathbf y | } \sum _ { t = 1 } ^ { | \mathbf y | } u _ { g } ^ { t } ( \pmb \theta ) ,\tag{6}
$$

$$
D _ { S } ( \pmb { \theta } ) = \frac { 1 } { | \mathcal { G } _ { \mathrm { r e c } } | } \sum _ { g \in \mathcal { G } _ { \mathrm { r e c } } } \frac { 1 } { | S | } \sum _ { ( \mathbf { x } , \mathbf { y } ) \in S } \frac { 1 } { | \mathbf { y } | } \sum _ { t = 1 } ^ { | \mathbf { y } | } \mathinner { | { \phi ^ { \prime } \left( u _ { g } ^ { t } ( \pmb { \theta } ) \right) } | } ,\tag{7}
$$

where $\boldsymbol { u } _ { g } ^ { t }$ is the pre-activation at the corresponding answer-token context and $\phi ^ { \prime } ( u ) = \sigma ( u ) + $ $u \sigma ( u ) ( 1 - \sigma ( u ) )$ . The derivative statistic averages absolute token-level derivatives, not the derivative evaluated at the mean pre-activation.

![](images/438ea3ccbbcab818bb14c1625e4f6a90e4447c1daa886934010660fee120eb73.jpg)  
Figure 9: Gate-wise pre-activation distributions at release. Each violin contains 1,232 gate-wise means, obtained by averaging answer-token pre-activations within examples and then examples equally. White markers and dark segments show the median and interquartile range across gates; the density outlines describe betweengate variation, not uncertainty across training runs.

A shift in selected-gate operating regions. Fig. 9 shows a shift toward more negative gate-wise means under GSU. On $B _ { 1 } ,$ the median moves from −0.2668 in Original to −0.5374 in GSU; on $B _ { 2 } .$ , it moves from −0.2656 to −0.5159. Retain also shifts, from −0.2877 to −0.4358. The withoutseal medians (−0.3149, −0.3094, and −0.3045, respectively) remain closer to Original, consistent with a contribution from sealing. These are distributions of $\bar { u } _ { g , S }$ across gates, not individual token activations.

![](images/9f7f1b67b0b4214e2be7abde42a71bef64234c1242a0e91c200269617b3abdbb.jpg)  
Figure 10: Local response and negative-tail occupancy at release. Left: mean absolute SiLU derivative, equally averaged across recorded gates. Right: mean negative-tail fraction at $\tau = - 4$ (colored), with the complement in gray. Both statistics change more on target inputs than retain inputs; they measure selected-gate responses before attack, not full-network gradients.

Stronger attenuation on target inputs. Fig. 10 and Tab. 7 connect this shift to local derivatives and negative-tail occupancy. The occupancy is the mean of the recorded gate-wise fractions, multiplied by 100, using $\tau = - 4$ rather than simply $u < 0$ and the same answer-token, example, and gate weighting. GSU raises occupancy on $B _ { 1 } / B _ { 2 }$ from approximately 0.41% to 3.68%/3.35%, versus 0.45% to 1.03% on retain; without sealing remains near Original. The corresponding reduction in $D _ { S }$ is approximately 23–24% on $B _ { 1 } / B _ { 2 }$ and 11% on retain. Together, the distributions and local responses are consistent with targeted attenuation rather than an equally large change across all input sets. Retain is not unchanged, and the small tail fractions do not imply that most gates or tokens are saturated. These release-stage probes are local mechanism evidence, not full-network gradient measurements or proof of causal isolation. §F.5 next checks how the same gate responses change under attack.

Table 7: Release-stage gate statistics. Derivatives and negative-tail fractions are equally averaged across recorded gates; tail fractions use $\tau = - 4$
<table><tr><td></td><td colspan="3">Mean  $\left| \phi ^ { \prime } ( u ) \right|$ </td><td colspan="3">Negative-tail fraction (%)</td></tr><tr><td>Inputs</td><td>Original</td><td>GSU</td><td>w/o seal</td><td>Original</td><td>GSU</td><td>w/o seal</td></tr><tr><td> $B _ { 1 }$ </td><td>0.3994</td><td>0.3018</td><td>0.3881</td><td>0.411</td><td>3.684</td><td>0.487</td></tr><tr><td> $B _ { 2 }$ </td><td>0.3997</td><td>0.3079</td><td>0.3904</td><td>0.408</td><td>3.348</td><td>0.463</td></tr><tr><td>Retain</td><td>0.3929</td><td>0.3484</td><td>0.3890</td><td>0.451</td><td>1.029</td><td>0.477</td></tr></table>

## F.5 SELECTED-GATE RESPONSES BEFORE AND AFTER ATTACK

Matched gate probes. We extend the release-only probes in §F.4 to attack epoch 10 on the same frozen Qwen3.5-2B disjoint setting. Fig. 11 compares the original base probe state with GSU and the without-seal variant at release and after attack. Original is not a No Defense checkpoint after ten attack epochs. All five states use the same 1,232 selected channels, with 308 channels in each of four layers, and the 200-example $B _ { 2 }$ probe set. The release measurements are shared with the earlier gate analyses, not independent training repetitions.

Measurements. We report mean pre-activation, mean absolute SiLU derivative, and mean occupancy of $u \leq - 4 .$ . For each gate, statistics first average valid answer-prediction positions within each example, then examples equally, and finally the 1,232 channels equally. Thus, tail occupancy is a hierarchical average, not the global fraction obtained by pooling all tokens; only this occupancy is multiplied by 100. The derivative uses $| \phi ^ { \prime } ( u ) | = | \sigma ( u ) \dot { + } \dot { u } \sigma ( u ) \dot { ( } 1 - \sigma ( u ) ) |$ at each valid prediction position, rather than evaluating the derivative at the mean pre-activation. Pre-activations here are means, not the gate-wise medians reported in §F.4.

![](images/3f63e465bfe6215eda6aff41d59a9e06023c8d60c3f8e6cf49f23dd9d12a4a64.jpg)

![](images/ece57510ffa840a768c4163f5692f0d8c9d06066ce5240d1da6f031b5d69db89.jpg)

![](images/37497ee1b62fb6b43f801a62e6398c63b805334647aa325f8689b5919f54e91f.jpg)  
Figure 11: Selected-gate responses before and after acquisition attacks. GSU and without sealing are probed at release and after ten attack epochs on $B _ { 2 } ;$ dashed lines denote the original base model. Panels show mean pre-activation, mean absolute SiLU derivative, and occupancy at u $\leq - 4$ over the same 1,232 channels. Statistics average valid answer-prediction positions within examples, then examples and channels.

Residual attenuation after attack. After ten attack epochs, GSU retains a lower mean absolute derivative than without sealing (0.373816 versus 0.419492), more negative mean pre-activations, and higher negative-tail occupancy, as shown in Tab. 8. This residual difference complements the lower extraction in the paired ablation. At the same time, attack partially reverses the release-stage changes: GSU’s mean pre-activation moves from −0.973011 to −0.456793, its derivative rises from 0.307875 to 0.373816, and its tail occupancy falls from 3.3482% to 0.5722%. The resulting picture is residual rather than irreversible attenuation: the selected gates move back toward the original state, but remain different from the attacked without-seal control.

Table 8: Matched $B _ { 2 }$ gate-response measurements. All states use the same selected channels and aggregation. Tail occupancy uses u $\leq - 4$ and is reported in percent.
<table><tr><td>Model state</td><td>Mean u</td><td>Mean  $\left| \phi ^ { \prime } ( u ) \right|$ </td><td>Tail (%)</td></tr><tr><td>Original base</td><td>-0.240670</td><td>0.399727</td><td>0.4084</td></tr><tr><td>GSU, release</td><td>-0.973011</td><td>0.307875</td><td>3.3482</td></tr><tr><td>GSU, attack epoch 10</td><td>-0.456793</td><td>0.373816</td><td>0.5722</td></tr><tr><td>Without seal, release</td><td>-0.302069</td><td>0.390441</td><td>0.4628</td></tr><tr><td>Without seal, attack epoch 10</td><td>-0.148507</td><td>0.419492</td><td>0.3177</td></tr></table>

Mechanism interpretation. The component controls and matched probes provide complementary empirical support for sealing: lower measured acquisition is accompanied by lower selected-gate response, including a residual difference after attack. The scope remains local. Even at release, only about 3.35% of valid answer-prediction positions fall in the negative tail under this aggregation; most gates or tokens cannot be described as saturated. Local derivatives do not give whole-network gradients or Hessians and do not directly measure resistance to optimization; 1,232 channels are not independent training replicates. These observations support the intended mechanism without a complete causal proof or a guarantee of irreversible forgetting. Having examined target-side behavior, we next test whether benign learning remains possible.

## F.6 BENIGN ADAPTATION: ACCURACY AND VALIDATION LOSS

Setup and metrics. We now assess benign adaptability using economics accuracy and full validationloss trajectories from the same six runs summarized in Fig. 4c. Each model has paired No Defense and GSU initializations. Training uses two epochs of full-parameter AdamW, learning rate $1 0 ^ { - 5 }$ effective batch size 16, weight decay 0.01, bfloat16 precision, maximum sequence length 512, a constant learning-rate schedule, and seed 0. The learning rate is used directly without calibration; no defense hooks or sealing penalties are active during benign training. Within each model, validation NLL uses the same 128 examples for both initializations; token counts can differ across model tokenizers. Economics accuracy is the equally weighted mean of the two MMLU subject accuracies, high\_school\_macroeconomics and high\_school\_microeconomics; it is distinct from validation NLL and utility. Both metrics are evaluated at epochs 0, 1, and 2.

![](images/b3a88f6dc1ac5bca87c8fab6c4abe2489327697e4292b5f9d0d249007d4f5078.jpg)

![](images/2902c1abe5485e43d3b687f1db6c7e4e00cbe9003c86897a97e3454549c2008a.jpg)

![](images/041d881c0536662a131ae8c79ea2019c713e4300ce080e66bfc2fd7f1610c775.jpg)  
Figure 12: Economics accuracy during benign fine-tuning. Accuracy is equally averaged over two economics MMLU subjects. The two Llama GSU models improve and narrow their gaps to No Defense; Qwen3.5-2B declines for both initializations. The three checkpoints belong to the same benign runs used for the NLL analysis.

Behavioral adaptation. The two Llama GSU models improve their economics accuracy and approach their undefended controls in Fig. 12. Llama-3.2-1B rises from 43.529% to 44.591%, and Llama-3.2-3B from 60.918% to 63.613%; their gaps to No Defense shrink from 1.143 to 0.420 and from 1.773 to 0.210 percentage points, respectively. Qwen3.5-2B instead declines under both GSU (65.971% to 64.105%) and No Defense (65.166% to 64.397%). Thus, the accuracy measurements support useful adaptation in the two Llama models, while also showing that NLL improvement need not produce accuracy gains on every model.

![](images/ded0a1f4323d319a58a421a7e316c9b310fd3a1e25eaffcdd8254ca8a8b5b004.jpg)

![](images/0824a1b8a460830975d9ed13cea908782fc1b3a190ebe250d892086592e0aa0d.jpg)

![](images/7e5560d1ed50eec4c2790db8a5483d46540e062b2639211d1f0fb4450113aaa2.jpg)  
Figure 13: Benign validation-loss trajectories. NLL at epochs 0, 1, and 2 for the runs summarized in Fig. 4c. All three GSU models approach the final loss of their undefended controls. GSU starts and finishes slightly higher, so larger reductions do not imply better final performance or higher learning efficiency.

Retained benign optimization ability. All three GSU models reduce validation NLL and finish within 0.005 of their paired No Defense controls, as shown in Fig. 13 and Tab. 9. The table reports the endpoints and $\Delta \mathsf { N L L } = \mathsf { N L L _ { 0 } } - \mathsf { N L L _ { 2 } }$ , the absolute reduction in the main figure. This provides a consistent optimization-level observation across the three models: normal fine-tuning can bring GSU close to the undefended controls’ validation loss. GSU’s larger reductions also reflect its higher initial losses, and its final losses remain slightly higher; they do not establish superior learning efficiency or statistical equivalence. Together with the accuracy measurements, this supports retained benign adaptability within the tested task, rather than uniformly improved downstream performance. The final subsection tests whether protection also remains after this adaptation.

Table 9: Benign validation-loss endpoints. ∆NLL is the absolute decrease over two epochs; displayed values are rounded independently.
<table><tr><td>Model</td><td>Initialization</td><td> ${ \mathsf { N L L } } _ { 0 }$ </td><td> ${ \mathsf { N L L } } _ { 2 }$ </td><td>∆NLL</td></tr><tr><td>Llama-3.2-1B</td><td>No Defense GSU</td><td>2.7635 2.7771</td><td>2.4991 2.5033</td><td>0.2645</td></tr><tr><td></td><td>No Defense</td><td>2.4733</td><td>2.2123</td><td>0.2738 0.2610</td></tr><tr><td>Qwen3.5-2B</td><td>GSU</td><td>2.5408</td><td>2.2171</td><td>0.3237</td></tr><tr><td>Llama-3.2-3B</td><td>No Defense</td><td>2.5707</td><td>2.2999</td><td>0.2707</td></tr><tr><td></td><td>GSU</td><td>2.5899</td><td>2.3030</td><td>0.2869</td></tr></table>

## F.7 RESISTANCE AFTER BENIGN FINE-TUNING

From benign adaptation to renewed attack. We finish by testing whether the benign learning in §F.6 eliminates GSU’s protection advantage on Qwen3.5-2B. No Defense and GSU are compared under two sequences: direct attack, and economics fine-tuning followed by attack. The benign stage uses two epochs, learning rate 10<sup>−5</sup>, and effective batch size 16, with the full settings given in §F.6. Each resulting model then receives the shared ten-epoch $B _ { 2 }$ attack. Fig. 14 shows all four trajectories; attack epoch 0 is immediately before attack, after benign fine-tuning where applicable. The direct-attack controls are shared with Fig. 4a.

![](images/ea9a1387fec880c055aa32937467bf13fb0d6ad716d21496abdd70e0b3247227.jpg)  
Figure 14: Resistance after benign fine-tuning. ES (left) and utility (right, 100× composite-score scale) during direct attacks (dashed) and attacks after two epochs of benign fine-tuning (solid). GSU retains lower final extraction and higher final utility than the corresponding undefended control after benign adaptation on Qwen3.5-2B.

Protection retained after adaptation. After benign fine-tuning and ten attack epochs, GSU reaches 5.063% ES, compared with 6.502% for No Defense, a 1.439-percentage-point advantage. Its final utility is also higher: 68.894 versus 67.772 on the plotted scale. Thus, in this experiment, normal benign adaptation does not eliminate GSU’s relative protection advantage. Prior benign fine-tuning lowers both final ES (5.377% to 5.063%) and utility (69.955 to 68.894) relative to directly attacking GSU, so adaptation is not a uniform improvement over the unadapted defense. The result concerns this benign task and attack budget, not arbitrary subsequent training.

Summary of the supplementary evidence. Across the measured setting, the analyses connect lower forbidden acquisition at retained utility, contributions from the component design, and selected-gate changes consistent with sealing. The benign experiments additionally show that normal learning remains possible and that a relative protection advantage survives the tested adaptation. Together, these observations support GSU’s goal of limiting target acquisition without eliminating benign adaptability, within the scope of the reported protocols.

## G LIMITATION AND FUTURE WORK

Our evaluation covers practical downstream fine-tuning rather than unrestricted retraining from scratch with arbitrary compute. The analysis is local, and GSU relies on the defender proxy set sharing acquisition pathways with future data; broader domain shifts may therefore require richer proxy coverage. Extending the study to additional architectures, domains, and post-release training procedures is a natural direction for future work.