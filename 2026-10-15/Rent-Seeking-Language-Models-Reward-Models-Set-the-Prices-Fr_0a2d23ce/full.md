# Rent-Seeking Language Models: Reward Models Set the Prices, Frequent Audits Curb the Rent

Zixiao Wang, Xingbo Du, Duzhen Zhang, Le Song Mohamed bin Zayed University of Artificial Intelligence {zixiao.wang,xingbo.du,duzhen.zhang,le.song}@mbzuai.ac.ae

## Abstract

Reward model overoptimization in RLHF is usually described as reward hacking. We recast it as rent-seeking: the policy, a firm, earns reward from a regulator, the proxy reward model, that true quality does not back. The lens turns overoptimization into measurable quantities, namely rent, a capture index and deadweight loss, and into testable predictions from the economics of regulation. In a fixed, versioncontrolled testbed with three model families, accuracy-matched proxy reward models and two gold reward models, RL turns 41% to 66% of its proxy gain into rent while gold reward plateaus. Rent follows each regulator’s prices. Neither proxy verifies facts, but one pays for confident specifics and the other for displays of diligence, so RL policies learn to fabricate details about nonexistent entities (97% to 100% of probes) or to deflect factual questions (67% to 98%). Kinship between policy and regulator adds a fixed premium of about 0.1 standard deviations but gives the policy no extra leverage under optimization. At a fixed expected fine, frequent mild audits beat rare severe ones, and random audits are as efficient per audit as targeted ones. Bringing the bunching estimator of tax economics to RL training, we also find a first sign that policies learn to stay just below a targeted audit’s threshold.

## 1 Introduction

Reinforcement learning from human feedback (RLHF) optimizes a policy against a learned reward model, a proxy for human preferences (Christiano et al., 2017; Stiennon et al., 2020; Ouyang et al., 2022). Pushed far enough, the optimization raises proxy reward without raising true quality, a regularity known as reward model overoptimization (Gao et al., 2023). Prior work catalogs the symptoms, such as growing length (Singhal et al., 2024) and sycophancy (Sharma et al., 2024), and proposes repairs such as reward model ensembles (Coste et al.,

2024; Eisenstein et al., 2024). Less is known about where the excess reward comes from, and about which institutional choices control it.

We study overoptimization as rent-seeking (Tullock, 1967; Krueger, 1974; Posner, 1975). The policy is a firm, the proxy reward model (proxy RM) is its regulator, and a gold reward stands for social welfare. Proxy reward that gold reward does not back is rent, and the fall in gold reward that rentseeking causes is deadweight loss. The two need not move together, because a firm can extract rent while welfare merely stops improving. Section 5 shows that our policies behave this way.

The lens matters only if it predicts something that the reward-hacking literature does not already test. It yields three hypotheses and three measurements. First, firms invest where the regulator pays, so rent should follow each proxy RM’s specific prices rather than a universal bias toward length. Second, regulation theory asks whether regulators recruited from the industry they oversee are captured or merely expert (Stigler, 1971; Dal Bó, 2006), which maps onto proxy RMs that share the policy’s pretraining. Third, deterrence theory asks whether the certainty or the severity of punishment deters (Becker, 1968; Polinsky and Shavell, 1979; Nagin, 2013). Audit theory adds that rules an agent can anticipate invite bunching below the threshold (Gans and Holden, 2026), as taxpayers bunch at kinks of the tax schedule (Saez, 2010) and firms below thresholds of stricter enforcement (Almunia and Lopez-Rodriguez, 2018). The measurements are a counterfactual decomposition of rent into strategies, a capture index, the share of proxy gain that gold reward does not back, and a bunching test for audit evasion.

We test these hypotheses in a controlled testbed built from open models (Figure 1). Policies and proxy RMs come from three pretraining families: Qwen3, Llama 3.2 and OLMo 2. All proxy RMs train on the same preference data, which the three families generate in equal shares, and a matched tier equalizes their held-out accuracy. Two gold reward models from different families define true quality, and a judge from a fourth family checks it. We optimize with best-of-n selection (BoN) and with RL, which separates a regulator’s intrinsic bias from a policy’s learned exploitation. We froze hypotheses, metrics and decision rules before running each experiment, and we log every later change with its information state (Appendix A).

Our findings are fourfold. First, RL turns 41% to 66% of its final proxy gain into rent, yet gold reward plateaus instead of collapsing. We therefore read rent under standard KL regularization as mostly a transfer. Second, rent follows prices. Neither proxy RM verifies facts, but the Qwen proxy pays for confident specifics and the Llama proxy for displays of diligence, and RL policies learn either to fabricate details about nonexistent entities or to deflect factual questions. Sycophancy carries a large rent premium yet does not grow, because no regulator pays more for it.

Third, proxy RMs rate outputs from their own family about 0.1 standard deviations higher, but neither BoN nor RL extracts more rent from a samefamily regulator. Kinship shifts the intercept, not the slope. Last, at a fixed expected fine, raising the audit probability from 2% to 20% raises the judge win rate by about 4 points per tenfold step, while rare severe audits barely beat no audit. Random audits match targeted audits per audit, and their cost does not drift as the policy changes. To our knowledge, we are also the first to apply bunching estimation to RL training, and under targeted audits it finds a small excess of training samples just below the audit threshold, a first sign that policies learn to evade a rule they can anticipate. We will release code, proxy RMs and all generations.

## 2 Related Work

Reward overoptimization. Proxy reward diverges from true reward under optimization in predictable ways (Gao et al., 2023; Skalse et al., 2022), and iterating RLHF shrinks it over rounds with diminishing gains (Wolf et al., 2025). Repairs include ensembles (Coste et al., 2024; Eisenstein et al., 2024), uncertainty penalties (Zhang et al., 2024), and reward models that disentangle, reweight or neutralize known spurious features such as length (Chen et al., 2024; Bu et al., 2025; Wang et al., 2025). Reward models also favor formats such as lists (Zhang et al., 2025), the behavior that carries most rent under our Qwen proxy (Section 5). Recent work audits rewards with an adversarially trained detector (Beigi et al., 2026), and a survey organizes the field (Wang et al., 2026). These methods detect or repair exploitation, whereas we measure which strategies carry rent, under which regulator, and how much of it optimization seeks.

Evaluator bias and kinship. Language-model judges favor their own generations (Zheng et al., 2023; Panickssery et al., 2024), partly through familiarity measured as perplexity (Wataoka et al., 2024), and they favor students trained on data from related models (Li et al., 2026). When one model both writes and evaluates, its evaluations drift from human judgment (Pan et al., 2024). Reward models trained on the same preference data still inherit different values from pretraining (Christian et al., 2026), and members of reward model ensembles that share pretraining make correlated errors (Gleave and Irving, 2022; Eisenstein et al., 2024), which suggests, but does not test, that a policy sharing a reward model’s pretraining could exploit its blind spots. Closest in design, Pinto (2026) crosses monitor calibration lineage with attacker lineage and finds a consistent own-lineage advantage. We test kinship between policy and proxy RM under optimization, with accuracy-matched proxies and shared training data.

Economics of oversight. Alignment has been framed as incomplete contracting (Hadfield-Menell and Hadfield, 2019) and through variants of Goodhart’s law (Manheim and Garrabrant, 2018). Recent work applies law-and-economics deterrence to solver and auditor agents (Agarwal et al., 2026), analyzes when randomized oversight deters agents that can conceal (Gans and Holden, 2026), and proposes rent-seeking as a diagnostic for modules inside an institutionally designed AI system (Chai, 2026). Wang and Huang (2026) model reward hacking as a principal-agent equilibrium in which the policy under-invests in what the evaluator does not measure, a theoretical counterpart to our capture index. To our knowledge, we are the first to model reward model overoptimization in RLHF as rent-seeking against a regulator and to test hypotheses from the economics of regulation in controlled RLHF experiments.

Monitoring and auditing. AI control protocols audit the most suspicious outputs under a fixed budget (Greenblatt et al., 2024), and optimizing against a chain-of-thought monitor teaches models to hide their intent (Baker et al., 2025). Auditing games test whether humans and agents can uncover hidden objectives (Marks et al., 2025; Sheshadri et al., 2026). Our audits are exogenous gold checks inside the reward channel, which lets us vary their probability, severity and targeting independently. To test whether a policy evades a learnable audit rule, we borrow bunching estimators from tax economics (Saez, 2010; Kleven, 2016).

![](images/a791c4a66a3139ffb697773f9925270f0ea2e1f664e3d65bd2e3cfad0e43c399.jpg)  
Figure 1: Overoptimization as rent-seeking. The policy earns proxy reward that gold reward does not back (rent). Rent follows the prices that each proxy RM sets.

## 3 Framework

Figure 1 maps the economic concepts onto RLHF, and Table 4 in Appendix B lists the correspondence. We z-normalize every reward model against one reference distribution, the pooled outputs of the three SFT policies on held-out prompts, so that proxy and gold rewards share a unit. We write $z _ { p }$ for the normalized proxy reward and $z _ { g }$ for the average of the two normalized gold rewards.

Rent and capture. The rent of a response y to a prompt x is

$$
\rho ( x , y ) = z _ { p } ( x , y ) - z _ { g } ( x , y ) .\tag{1}
$$

Optimization changes rent relative to a starting point, the SFT policy for RL and a single sample (n = 1) for BoN, and we call this change excess rent. The capture index compares the gold gain with the proxy gain over the same span,

$$
\mathrm { C I } = 1 - { \Delta z _ { g } } / { \Delta z _ { p } } ,\tag{2}
$$

and we compute it only when $\Delta z _ { p } \geq 0 . 2 5$ , because the ratio is unstable for small proxy gains. A capture index of 0 means that all proxy gain is real, 1 means that all of it is rent, and values above 1 mean that true quality falls.

Deadweight loss. Deadweight loss (DWL) is the drop in gold reward from its peak to the end of training, measured on a smoothed curve so that noise does not inflate the peak (Appendix B).

Favoritism and capture. We write cells as policy × RM and call a cell same-family when both share a pretraining family. Favoritism is the rent difference between same-family and cross-family cells before any optimization. The revolving-door effect is the difference in capture index between same-family and cross-family cells at equal optimization pressure,

$$
\Delta = \overline { { \mathrm { C I } } } _ { \mathrm { s a m e } } - \overline { { \mathrm { C I } } } _ { \mathrm { c r o s s } } .\tag{3}
$$

The capture hypothesis (Stigler, 1971) predicts $\Delta > 0 ,$ , and the expertise hypothesis (Che, 1995) predicts $\Delta < 0$ . Favoritism shifts the intercept of rent, whereas capture would steepen its slope in optimization pressure, so the two can occur separately.

Audits. Following models of income tax evasion (Allingham and Sandmo, 1972), each rollout is audited with probability p. An unaudited rollout receives $z _ { p } ,$ and an audited one receives

$$
R = z _ { g } - F \cdot \operatorname* { m a x } ( 0 , \rho - \tau ) ,\tag{4}
$$

which replaces proxy reward with gold reward and fines rent above a tolerance τ , the 90th percentile of rent on SFT outputs. Holding the expected fine $c = p \cdot F$ fixed traces an equal-expected-penalty line, along which Becker’s model predicts equal deterrence (Becker, 1968). A targeted audit (T2) instead checks a rollout whenever the primary proxy scores it above the cross-family proxy by more than a fixed threshold, set to audit 5% of SFT outputs.

## 4 Experimental Setup

Models. Policies and proxy RMs build on Qwen3-1.7B-Base (Yang et al., 2025), Llama-3.2- 1B (Grattafiori et al., 2024; Meta AI, 2024) and OLMo-2-0425-1B (Team OLMo et al., 2024). All three are dense decoder-only Transformers of similar scale, so the family contrast is primarily a pretraining contrast. We exclude hybrid-attention models such as Qwen3.5 from the core design, because an architecture difference would confound the family comparison. Each family receives the same supervised fine-tuning (SFT) on single-turn conversations from the Tulu 3 mixture (Lambert et al., 2025), under one shared plain-text chat template. Proxy RMs add a scalar head to the same bases and train with a Bradley-Terry loss (Bradley and Terry, 1952) for one epoch.

Data. Prompts come from UltraFeedback (Cui et al., 2024), split into disjoint sets of 20,000 prompts for reward modeling, 10,000 for RL and 1,000 for evaluation. Every family’s SFT policy writes two responses per reward-modeling prompt, and the pairs that the gold reward separates form 111,515 training pairs (Appendix C). Every proxy RM trains on this one pool, so a same-family cell never coincides with in-distribution training data. Proxy RMs come in three data tiers (5,000, 20,000 and 80,000 pairs) with three seeds per family and tier.

Accuracy matching. At equal data the Qwen proxy is markedly stronger, with held-out accuracy 0.866 at 20,000 pairs against 0.797 for Llama and 0.786 for OLMo. A matched tier therefore trains the Qwen proxy on 3,000 pairs, which brings its mean accuracy to 0.795 against a target of 0.792. RL uses the matched seed whose Qwen and Llama proxies are closest in accuracy (both 0.798).

Gold reward and judges. Two gold RMs define true quality: Skywork-Reward-V2 built on Qwen3- 8B and on Llama-3.1-8B (Liu et al., 2026). We z-normalize both and average them. They correlate at 0.925 on reference outputs, and the gold reward does not favor length, since the shorter response wins 55.6% of preference pairs. Gemma 4 judges (Gemma Team et al., 2026), from a family unrelated to the policies, check the gold reward through pairwise comparisons in both orders. On 444 calibration pairs, the 26B-A4B and 31B judges agree with the gold reward on 85.4% and 86.4% of pairs, and the two gold RMs agree with each other on 91.4%. An API judge from a fifth family, Claude Haiku 5.5, agrees with the gold reward on 87.0% of the pairs it decides (Appendix E). We average the two Gemma 4 judges in all judge-based results.

Optimization. BoN draws N = 256 samples per prompt from each SFT policy on all 1,000 evaluation prompts. Any $n \leq N$ then has an exact expectation without resampling, and its KL divergence from the SFT policy is at most log $n - ( n - 1 ) / n .$ about 4.5 nats at $n = 2 5 6$ (Stiennon et al., 2020; Beirami et al., 2025). RL uses RLOO (Ahmadian et al., 2024) in verl (Sheng et al., 2025), with KL divergence to the SFT policy as a reward penalty $( \beta = 0 . 0 0 5 )$ . Each of the 600 steps uses 64 prompts and 8 samples per prompt, with a constant learning rate of $2 \times 1 0 ^ { - 6 }$ . Every 10 steps the policy answers the first 500 evaluation prompts, which all reward models then score. RL covers the Qwen × Llama 2 × 2 grid with three seeds per cell, plus KL penalties of 0.01 and 0.05 with one seed each.

Statistics and analysis plan. We report 95% bootstrap confidence intervals over prompts, and over seeds where a design has several. We prespecified the main analyses, metrics and gates, and a decision log records every later change with its information state (Appendix A).

## 5 Anatomy of Rent (RQ1)

RL raises proxy reward steadily while gold reward stops improving (Figure 6 in Appendix D). In the first Qwen × Qwen seed, gold reward peaks at step 370 and then stays flat, and the per-prompt drop after the peak is not significant (lower bound −0.002). Over the same run the capture index rises from 0.07 at step 50 to 0.39 at step 600, so nearly all proxy gain after the plateau is rent. At step 600 the capture index is 0.41 for Qwen × Qwen, 0.43 for Qwen × Llama, 0.61 for Llama × Qwen and 0.66 for Llama × Llama.

Deadweight loss stays between 0.00 and 0.10 standard deviations, and it grows only in an exploratory run with a looser KL penalty (0.067 at $\beta = 0 . 0 0 1$ , against at most 0.007 at $\beta = 0 . 0 1$ and 0.05). We attribute the plateau to the KL penalty, which bounds how far a policy can drift, and we find that under standard regularization rent-seeking is mostly a transfer.

## 5.1 Candidate strategies

We first let the data propose candidate strategies. Pairing two responses to the same prompt with nearly equal gold reward but very different proxy reward isolates what the proxy pays for and the gold reward does not. A Gemma 4 judge describes how 228 such pairs differ, and clustering the descriptions yields 25 candidate categories (Appendix G). Five appear mainly on the high-rent side: structured formatting, bulleted key points, concluding summaries, lists of external resources and disclaimers. Self-praise is rare, and text addressed to the grader does not occur.

## 5.2 Counterfactual decomposition

We measure how much rent each strategy carries by removing it and scoring the result again. Eight deterministic rewrites each remove one behavior and keep the rest of the text verbatim: formatting, repeated sentences, restating the prompt, template phrases, self-evaluation, concluding summaries, disclaimers and resource lists. The rent that a rewrite removes is $\Delta \rho _ { k } = \rho ( \mathrm { o r i g i n a l } ) - \rho ( \mathrm { r e w r i t e } )$ and its rent share is the mean of $\Delta \rho _ { k }$ divided by the mean excess rent. Applying all rules in sequence gives the share that rules explain jointly. Before scoring, we validated every rule by hand and by full-population checks (Appendix H).

Figure 2 shows the shares at step 600. First, rules explain 6% to 30% of excess rent under RL. Second, the explained rent depends on the regulator. Formatting carries 35% of the rent in Qwen × Qwen [32%, 38%] and 8% in Llama × Qwen, but at most 4% under the Llama proxy, where resource lists carry 5.3% [4.5%, 6.1%] and 4.6% [3.9%, 5.3%] instead.

Third, RL learns these strategies. The formatting share in Qwen × Qwen rises from −0.20 at step 100 to 0.35 at step 600, whereas BoN, which only selects from the SFT distribution, shows no formatting rent, and rules explain little of its rent in any cell. Last, concluding summaries carry negative shares in most cells. The gold reward values them more than the proxy does, except in Llama × Qwen (+0.04).

## 5.3 Length

Rules barely change length, so we estimate the length channel from RL outputs directly. At the same prompt and validation step the three seeds write responses of different lengths, and the slope of rent on log length, times the growth in log length over training, gives the share of excess rent associated with length (Appendix I).

This share is 0.22 [0.15, 0.29] for Qwen × Qwen, 0.30 [0.22, 0.39] for Qwen × Llama, 0.14 [0.11, 0.17] for Llama × Qwen and 0.22 [0.18, 0.26] for Llama × Llama. The largest share belongs to the cell whose responses grow most, from 218 to 847 tokens. Gold reward rises slightly with length for Qwen policies and falls for Llama policies, so we conjecture that Llama policies pad where Qwen policies elaborate. The estimate is correlational and overlaps with formatting. Together, rules and length account for up to half of excess rent, and the probes below identify semantic strategies that carry rent beyond them.

## 5.4 Sycophancy and fabrication

UltraFeedback prompts rarely contain false premises or questions without an answer, so we probe two semantic strategies directly on the two SFT policies and the 12 final RL policies. The sycophancy probe asks 300 factual questions from Sharma et al. (2024) in four forms: neutral, suggesting a wrong answer, doubting the right answer, and suggesting the right answer. The fabrication probe asks about 300 nonexistent studies, laws, novels, prizes, towns and companies, for which an honest response says that it cannot verify the item. A Gemma 4 31B judge labels each greedy response, and a manual check of 50 labels found one borderline error (Appendix J).

We find both proxies blind to correctness. Within each model, responses that follow a suggested wrong answer receive about the same proxy reward as responses that do not (−0.21 to +0.18 standard deviations), while their gold reward is 0.40 to 1.43 lower. Sycophancy therefore carries a rent premium of 0.37 to 1.37 standard deviations in every model. For fabrication the proxies differ. On SFT outputs the Qwen proxy pays for invented specifics over honest uncertainty (+0.46 and +0.73). The Llama proxy pays nothing for them on Qwen outputs (+0.00) and, after RL, prefers deflection (−0.47 to −0.59 in Qwen × Llama).

RL follows these prices (Table 1). Policies trained against the Qwen proxy answer nearly everything, leave only 10% of neutral questions unanswered, and invent details about nonexistent entities in 97% to 100% of probes. Policies trained against the Llama proxy stop answering. They leave 67% and 98% of neutral factual questions unanswered, typically by explaining how to look the answer up, and fabricate in only 17% and 30% of probes. Sycophancy, by contrast, does not change: among responses that commit to one of the two answers, the shift toward a suggested wrong answer stays within 0.09 of the SFT policy in every cell. We conclude that rent the regulator merely fails to detect persists, while rent the regulator pays for is sought.

![](images/905204828a65238c2a64f4dbdf14641e3eeaa60309a381a5d023aad48dfc2704.jpg)  
Figure 2: Rent shares of rule-detected strategies. Left: RL cells over training. Right: BoN cells. “All” applies every rule in sequence. Shares of single rules overlap and do not sum.

<table><tr><td>Model</td><td>Correct</td><td>No ans.</td><td>Follow</td><td>Fabricate</td></tr><tr><td>SFT Qwen</td><td>0.41</td><td>0.10</td><td>0.57</td><td>0.83</td></tr><tr><td>SFT Llama</td><td>0.35</td><td>0.21</td><td>0.64</td><td>0.95</td></tr><tr><td>Qwen × Qwen</td><td>0.35</td><td>0.10</td><td>0.49</td><td>0.97</td></tr><tr><td>Llama × Qwen</td><td>0.30</td><td>0.10</td><td>0.67</td><td>1.00</td></tr><tr><td>Qwen × Llama</td><td>0.21</td><td>0.67</td><td>0.52</td><td>0.17</td></tr><tr><td>Llama × Llama</td><td>0.02</td><td>0.98</td><td>0.60†</td><td>0.30</td></tr></table>

Table 1: Probe results, means of three seeds for RL cells. Correct and No answer refer to neutral questions. Follow is the rise in the share of committed answers that adopt a suggested wrong answer. †Few committed answers, confidence interval wider than 0.5.

## 5.5 Public checkpoints

We run the rule detectors on the SFT, DPO and final checkpoints of Tulu 3 8B (Lambert et al., 2025), OLMo 2 7B (Team OLMo et al., 2024) and OLMo 3 7B (Team Olmo et al., 2025) (Appendix K). Formatting rises at the preferenceoptimization stage in all three series, by 0.24, 0.23 and 0.15 in prevalence from SFT to DPO, and stays flat through the final stage, which optimizes verifiable rewards without a reward model. The behavior that carries the most rent under the Qwen proxy is thus the one that preference optimization amplifies in practice.

## 6 The Revolving Door (RQ2)

Static capture. We use BoN to reveal a regulator’s intrinsic bias without any learning. Proxy RMs from the policy’s own family assign its responses 0.10 to 0.17 standard deviations more rent than cross-family proxies do (Figure 8 in Appendix L). This favoritism is already present at n = 1 (0.11 to 0.14), before any selection, and it barely changes as n grows to 256. It holds under either single gold RM, which rules out affinity between a gold RM and its family as its source. The capture index, which removes the level at n = 1, differs by at most 2.7 points between same-family and cross-family cells, and by 0.6 points [0.0, 1.2] in the matched tier. The judge finds at most small differences, with win-rate gaps of at most 1.9 points across n.

Familiarity explains part of this favoritism, and after controlling for gold reward it remains in two of the three families (Appendix L).

We find that regulator capacity matters far more than kinship. At n = 64 the mean capture index over the nine cells falls from 0.42 for proxies trained on 5,000 pairs to 0.25 with 20,000 pairs and 0.17 with 80,000 pairs, and the three policy families stay within 0.37 to 0.44, 0.22 to 0.27 and 0.15 to 0.18. Across this range, the same-family difference never exceeds 2.7 points.

Dynamic capture. RL lets the policy learn to exploit its regulator. We compare RL with BoN at the same KL divergence of 4 nats, the largest value that BoN reaches, and interpolate each RL curve in KL between neighboring validation points. At this point the revolving-door effect is +0.002 [−0.100, 0.101] under RL and −0.030 [−0.052, −0.008] under BoN, and their difference in differences is +0.032 [−0.072, 0.131] (Figure 3). We therefore find no amplification by $\mathrm { R L , }$ and the data rule out an amplification larger than about 0.13. Beyond 4 nats, the RL effect stays within about ±0.02 up to 64 nats.

![](images/375db9eb4f0d3c03d92efb8ff087de905b7dc7ec8ae8fc3880b61e861f8bae1a.jpg)

![](images/183fb35e4aac1f343876c07f8253d4b059847bb3a5f4c1478c8c54e547b23fd9.jpg)  
Figure 3: Revolving-door effect $\Delta .$ Left: $\Delta$ against KL for RL and BoN. Right: estimates at 4 nats with 95% intervals.

Favoritism survives RL. At 4 nats the samefamily rent premium is 0.13 [0.09, 0.16] under RL, whereas BoN selection lowers it to 0.07 [0.04, 0.09].

The cells differ mostly for reasons other than kinship. At step 600 the two Llama-policy cells reach capture indices of 0.61 and 0.66 against 0.41 and 0.43 for the Qwen-policy cells, which we attribute to the weaker Llama SFT policy that leaves less real quality for RL to gain (Appendix L).

Interpretation. We conclude that kinship breeds favoritism, not capture. Same-family regulators add a fixed premium to their own family’s outputs, but the premium does not raise the return on optimization, and we find no consistent expertise advantage either. Pinto (2026) reports that AI-control monitors detect attacks from the lineage on which they were calibrated better than attacks from other lineages. Our proxy RMs all train on data that the three families generate in equal shares, which closes that channel, so our null effect isolates shared pretraining. We read the two results together as consistent with expertise that comes from experience rather than from shared origin.

## 7 Audits (RQ3)

We use audits to let gold reward enter training in part, and we run them on the Qwen × Qwen cell. Table 2 lists the conditions. Each runs for 600 steps with the RL setup of Section 4, and the no-audit reference reuses the three Qwen × Qwen seeds of the 2 × 2 grid.

<table><tr><td>Condition</td><td>p</td><td>F</td><td>Seeds</td></tr><tr><td>No audit</td><td>0</td><td>n/a</td><td>3</td></tr><tr><td>Equal-exp. penalty</td><td>0.02 / 0.05 / 0.2</td><td> $5 0 / 2 0 / 5$ </td><td>3 each</td></tr><tr><td>Back tax only</td><td>0.05</td><td>0</td><td>3</td></tr><tr><td>Full gold</td><td>1</td><td>0</td><td>2</td></tr><tr><td>Targeted audit (T2)</td><td>0.05*</td><td>20</td><td>3</td></tr><tr><td>Length penalty</td><td> $\lambda = 1 . 3$ </td><td></td><td>2</td></tr><tr><td>Cross-family ens.</td><td>min of two proxies</td><td></td><td>2</td></tr></table>

Table 2: Audit conditions. The expected fine is c = $p \cdot F = 1$ . ∗Nominal budget, set on SFT outputs.

Calibration. Pilots of 150 steps fixed the expected fine at $c = 1$ , the only tested value that closed between 25% and 75% of the rent gap to full gold reward, and an interpolated length penalty of $\lambda = 1 . 3$ that keeps responses within 4% of SFT length (Appendix M).

Outcome. Audits feed gold reward into training, so the gold reward is no longer a neutral evaluator for these runs. The primary outcome is therefore the win rate of the final checkpoint over SFT responses on 300 prompts, judged in both orders and averaged over the two Gemma 4 judges. Each judge alone serves as a robustness check (Appendix M). Along the equal-expected-penalty line we regress this win rate on $\log _ { 1 0 } p$ with a random intercept per seed. A positive slope whose interval excludes zero supports the certainty hypothesis, and an interval inside ±0.03 is consistent with Becker’s model.

Certainty versus severity. At a fixed expected fine, frequent mild audits outperform rare severe ones (Figure 4). The judge win rate rises by 0.039 per tenfold increase in audit probability [0.016, 0.065], which supports the certainty hypothesis under our pre-specified decision rule. Rare severe audits $( p = 0 . 0 2 $ , F = 50) barely beat no audit (0.672 against 0.666), although they halve the capture index (0.20 against 0.41). Each judge alone gives the same result (0.038 [0.009, 0.066] and 0.041 [0.010, 0.075]).

We attribute part of the effect to the variance of the training signal. Without standard-deviation normalization, a rare large fine produces extreme negative advantages, with a minimum of −28.7 at $p = 0 . 0 2$ against −7.5 at p = 0.2 and −1.3 without audits, so a few fined samples dominate each update. Because the share of gold reward in training also grows with p, we checked a back-tax condition without fines. $\mathbf { A t } p = 0 . 0 5$ it adds only 0.014 to the win rate (0.680), which suggests that deterrence rather than the extra gold signal drives most of the effect.

![](images/a0497b44e70958962394319107dc14de3ef71675ca1186ccec4f695356091fee.jpg)  
Audit probability p (expected fine p·F held fixed along the line)  
Figure 4: Equal-expected-penalty line $( p \cdot F = 1 )$ . Left: judge win rate over SFT. Middle and right: gold reward and capture index at step 600, which audits partly optimize. Grey lines are seeds.

Targeted audits and bunching. Random audits are as efficient as targeted ones. Per thousand audits, both raise the win rate by about the same amount (0.0028 and 0.0026), and at the nominal budget T2 leads random audits at $p \ : = \ : 0 . 0 5$ by 0.023 [−0.010, 0.053]. Random audits also keep their cost fixed, whereas policy drift widened the disagreement between the two proxies and pushed T2 to audit 8.2% of rollouts instead of 5%.

The bunching test gives a first sign that the policy learns to stay just below the T2 threshold (Figure 5). It compares the density of disagreement scores in a window below the threshold with a smooth fit outside the window, as bunching studies of tax data do (Kleven, 2016). The excess mass of disagreement scores in the window below the threshold is 3.7% [0.1%, 7.5%] under T2 and 0.1% [−1.1%, 1.3%] under random audits, which have no threshold to avoid. Their difference is 3.6% [−0.2%, 7.6%], in the predicted direction with an interval covering zero. The overall audit rate rose rather than fell, so the policy did not evade targeted audits at scale, and any avoidance stays local to the threshold.

Cost and baselines. Among partial audits, T2 reaches the highest win rate (0.731) and lies on the cost frontier, whereas audits at $p = 0 . 2$ spend 2.4 times as many gold calls for a lower win rate of 0.714 (Figure 9 and Table 12 in Appendix M). Two common mitigations fall below no audit on the judge measure. A length penalty shortens responses to 152 tokens but lowers the win rate to 0.604, and a cross-family ensemble that rewards the minimum of two proxies lengthens responses to 728 tokens and lowers it to 0.614.

![](images/2b7a84d896ddbd4795c2096903d722fbab85f10e72b86f49cc2c7d965ec63961.jpg)

![](images/d0f590efd12330fc6450355c029a450d08cbddedf848f8e0e476c23677cabe14.jpg)  
Figure 5: Disagreement scores near the audit threshold after step 100. Orange bins form the window below the threshold, and the curve is the counterfactual fit outside it. Left: targeted audits. Right: random audits as a placebo.

## 8 Conclusion

We recast reward model overoptimization in RLHF as rent-seeking and tested predictions from the economics of regulation under a fixed, versioncontrolled testbed. Three findings stand out. Rent follows prices: each proxy RM pays for a different signal that it cannot verify, and RL policies learn to supply it, inventing specifics under one regulator and deflecting under the other. Regulator capacity matters more than kinship, because same-family reward models add a fixed premium but give the policy no extra leverage. Certainty deters better than severity, because at a fixed expected fine frequent mild audits raise judge-rated quality while rare severe audits barely move it. A bunching test, new to RL training, adds a first sign that policies learn to stay just below a targeted audit’s threshold.

Neither proxy RM can verify facts, and every strategy we can price exploits that gap: confident specifics, displays of diligence, formatting and length. Rent that a regulator merely fails to detect, such as sycophancy, persists without growing. We conjecture that it would grow under a regulator that pays for agreement. In practice, investing in what a reward model can verify matters more than avoiding same-family reward models, although same-family scores still carry a level bias.

These results give practice three levers, namely checking what a reward model pays for, investing in its capacity, and auditing often, mildly and at random. Generic fixes miss these prices, since a length penalty and a cross-family ensemble both lowered judge-rated quality below no audit. At a fixed expected fine, frequent mild audits beat rare severe ones, favoring thinly spread gold-reward budgets. More broadly, the economic lens turns overoptimization into quantities that can be measured and predictions that can be tested.

## Limitations

Our policies and proxy RMs have 1 to 2 billion parameters. The public checkpoints of Section 5 extend the behavioral findings to 7B and 8B models. Owing to computational constraints, we did not scale to larger models. We leave such verification to better-resourced industry labs or institutions. Our prompts are English only. Our results come from a computational sandbox. We use economic theory to generate hypotheses and measurements, and we make no causal claim about real economies or real regulators.

## References

Rohit Agarwal, Joshua Lin, Mark Braverman, and Elad Hazan. 2026. AI alignment via incentives and correction. Preprint, arXiv:2605.01643.

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. 2024. Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12248–12267, Bangkok, Thailand. Association for Computational Linguistics.

Michael G. Allingham and Agnar Sandmo. 1972. Income tax evasion: A theoretical analysis. Journal of Public Economics, 1(3–4):323–338.

Miguel Almunia and David Lopez-Rodriguez. 2018. Under the radar: The effects of monitoring firms on tax compliance. American Economic Journal: Economic Policy, 10(1):1–38.

Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y. Guan, Aleksander Madry, Wojciech Zaremba, Jakub Pachocki, and David Farhi. 2025. Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. Preprint, arXiv:2503.11926.

Gary S. Becker. 1968. Crime and punishment: An economic approach. Journal of Political Economy, 76(2):169–217.

Mohammad Beigi, Ming Jin, Junshan Zhang, Qifan Wang, and Lifu Huang. 2026. Adversarial reward auditing for active detection and mitigation of reward hacking. In Third Conference on Language Modeling.

Ahmad Beirami, Alekh Agarwal, Jonathan Berant, Alexander D’Amour, Jacob Eisenstein, Chirag Nagpal, and Ananda Theertha Suresh. 2025. Theoretical guarantees on the best-of-n alignment policy. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings

of Machine Learning Research, pages 3580–3602. PMLR.

Ralph Allan Bradley and Milton E. Terry. 1952. Rank analysis of incomplete block designs: I. The method of paired comparisons. Biometrika, 39(3/4):324– 345.

Yuyan Bu, Liangyu Huo, Yi Jing, and Qing Yang. 2025. Beyond excess and deficiency: Adaptive length bias mitigation in reward models for RLHF. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pages 3091–3098, Albuquerque, New Mexico. Association for Computational Linguistics.

Rui Chai. 2026. Alignment as institutional design: From behavioral correction to transaction structure in intelligent systems. Preprint, arXiv:2604.13079.

Yeon-Koo Che. 1995. Revolving doors and the optimal tolerance for agency collusion. The RAND Journal ofEconomics, 26(3):378–397.

Lichang Chen, Chen Zhu, Jiuhai Chen, Davit Soselia, Tianyi Zhou, Tom Goldstein, Heng Huang, Mohammad Shoeybi, and Bryan Catanzaro. 2024. ODIN: Disentangled reward mitigates hacking in RLHF. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 7935–7952. PMLR.

Brian Christian, Jessica A. F. Thompson, Elle Michelle Yang, Vincent Adam, Hannah Rose Kirk, Christopher Summerfield, and Tsvetomira Dumbalska. 2026. Reward models inherit value biases from pretraining. In The Fourteenth International Conference on Learning Representations (ICLR), pages 36013–36041.

Paul F. Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. 2024. Reward model ensembles help mitigate overoptimization. In The Twelfth International Conference on Learning Representations (ICLR).

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2024. UltraFeedback: Boosting language models with scaled AI feedback. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 9722–9744. PMLR.

Ernesto Dal Bó. 2006. Regulatory capture: A review. Oxford Review ofEconomic Policy, 22(2):203–225.

Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D’Amour, DJ Dvijotham, Adam Fisch, Katherine Heller, Stephen Pfohl, Deepak Ramachandran, Peter Shaw, and Jonathan Berant. 2024.

Helping or herding? Reward model ensembles mitigate but do not eliminate reward hacking. In First Conference on Language Modeling (COLM).

Joshua S. Gans and Richard Holden. 2026. When does randomized oversight align AI agents that can conceal? Preprint, arXiv:2609.38262.

Leo Gao, John Schulman, and Jacob Hilton. 2023. Scaling laws for reward model overoptimization. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 10835–10866. PMLR.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, et al. 2026. Gemma 4 technical report. Preprint, arXiv:2607.02770.

Adam Gleave and Geoffrey Irving. 2022. Uncertainty estimation for language reward models. Preprint, arXiv:2203.07472.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, et al. 2024. The Llama 3 herd of models. Preprint, arXiv:2407.21783.

Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan, and Fabien Roger. 2024. AI control: Improving safety despite intentional subversion. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 16295–16336. PMLR.

Dylan Hadfield-Menell and Gillian K. Hadfield. 2019. Incomplete contracting and AI alignment. In Proceedings ofthe 2019 AAAI/ACM Conference on AI, Ethics, and Society (AIES), pages 417–422. ACM.

Henrik Jacobsen Kleven. 2016. Bunching. Annual Review ofEconomics, 8(1):435–464.

Anne O. Krueger. 1974. The political economy of the rent-seeking society. The American Economic Review, 64(3):291–303.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, and 4 others. 2025. Tülu 3: Pushing frontiers in open language model post-training. In Second Conference on Language Modeling (COLM).

Dawei Li, Renliang Sun, Yue Huang, Ming Zhong, Bohan Jiang, Jiawei Han, Xiangliang Zhang, Wei Wang, and Huan Liu. 2026. Preference leakage: A contamination problem in LLM-as-a-judge. In The Fourteenth International Conference on Learning Representations (ICLR).

Chris Yuhao Liu, Liang Zeng, Yuzhen Xiao, Jujie He, Jiacai Liu, Chaojie Wang, Rui Yan, Wei Shen, Fuxiang Zhang, Jiacheng Xu, Yang Liu, and Yahui Zhou. 2026. Skywork-Reward-V2: Scaling preference data curation via human-AI synergy. In The Fourteenth International Conference on Learning Representations (ICLR).

David Manheim and Scott Garrabrant. 2018. Categorizing variants of Goodhart’s law. Preprint, arXiv:1803.04585.

Samuel Marks, Johannes Treutlein, Trenton Bricken, Jack Lindsey, Jonathan Marcus, Siddharth Mishra-Sharma, Daniel Ziegler, Emmanuel Ameisen, Joshua Batson, Tim Belonax, Samuel R. Bowman, Shan Carter, Brian Chen, Hoagy Cunningham, Carson Denison, Florian Dietz, Satvik Golechha, Akbir Khan, Jan Kirchner, and 16 others. 2025. Auditing language models for hidden objectives. Preprint, arXiv:2503.10965.

Meta AI. 2024. Llama 3.2: Revolutionizing edge AI and vision with open, customizable models. Meta AI blog post, September 25, 2024.

Daniel S. Nagin. 2013. Deterrence in the twenty-first century. Crime and Justice, 42(1):199–263.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc.

Jane Pan, He He, Samuel R. Bowman, and Shi Feng. 2024. Spontaneous reward hacking in iterative selfrefinement. Preprint, arXiv:2407.04549.

Arjun Panickssery, Samuel R. Bowman, and Shi Feng. 2024. LLM evaluators recognize and favor their own generations. In Advances in Neural Information Processing Systems, volume 37, pages 68772–68802.

Lucas Pinto. 2026. Calibration-family overfit: Why trusted sabotage monitors don’t transfer across lineages. Preprint, arXiv:2607.06596. COLM 2026 Workshop on AI Measurement Science.

A. Mitchell Polinsky and Steven Shavell. 1979. The optimal tradeoff between the probability and magnitude of fines. American Economic Review, 69(5):880– 891.

Richard A. Posner. 1975. The social costs of monopoly and regulation. Journal of Political Economy, 83(4):807–827.

Emmanuel Saez. 2010. Do taxpayers bunch at kink points? American Economic Journal: Economic Policy, 2(3):180–212.

Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R. Johnston, Shauna Kravec, Timothy Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan, Miranda Zhang, and Ethan Perez. 2024. Towards understanding sycophancy in language models. In The Twelfth International Conference on Learning Representations (ICLR).

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. HybridFlow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems (EuroSys ’25), pages 1279–1297, Rotterdam, Netherlands. ACM.

Abhay Sheshadri, Aidan Ewart, Elias Kempf, Kai Fronsdal, Isha Gupta, Simon Schrodi, Arthur Conmy, Samuel R. Bowman, Sara Price, Samuel Marks, and Rowan Wang. 2026. AuditBench: Evaluating alignment auditing techniques on models with hidden behaviors. Preprint, arXiv:2602.22755.

Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. 2024. A long way to go: Investigating length correlations in RLHF. In First Conference on Language Modeling (COLM).

Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. 2022. Defining and characterizing reward hacking. In Advances in Neural Information Processing Systems, volume 35, pages 9460–9471. Curran Associates, Inc.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F. Christiano. 2020. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems, volume 33, pages 3008–3021.

George J. Stigler. 1971. The theory of economic regulation. The Bell Journal of Economics and Management Science, 2(1):3–21.

Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, et al. 2025. Olmo 3. Preprint, arXiv:2512.13961.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, et al. 2024. 2 OLMo 2 furious. Preprint, arXiv:2501.00656. A shorter version appeared at COLM 2025.

Gordon Tullock. 1967. The welfare costs of tariffs, monopolies, and theft. Western Economic Journal, 5(3):224–232.

Chaoqi Wang, Zhuokai Zhao, Yibo Jiang, Zhaorun Chen, Chen Zhu, Yuxin Chen, Jiayi Liu, Lizhu Zhang, Xiangjun Fan, Hao Ma, and Sinong Wang. 2025. Beyond reward hacking: Causal rewards

for large language model alignment. Preprint, arXiv:2501.09620.

Jiacheng Wang and Jinbin Huang. 2026. Reward hacking as equilibrium under finite evaluation. Preprint, arXiv:2603.28063.

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. 2024. Large language models are not fair evaluators. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9440–9450, Bangkok, Thailand. Association for Computational Linguistics.

Xiaohua Wang, Muzhao Tian, Yuqi Zeng, Zisu Huang, Jiakang Yuan, Bowen Chen, Jingwen Xu, Mingbo Zhou, Wenhao Liu, Muling Wu, Zhengkang Guo, Qi Qian, Yifei Wang, Feiran Zhang, Ruicheng Yin, Shihan Dou, Changze Lv, Tao Chen, Kaitao Song, and 4 others. 2026. Reward hacking in the era of large models: Mechanisms, emergent misalignment, challenges. Preprint, arXiv:2604.13602.

Koki Wataoka, Tsubasa Takahashi, and Ryokan Ri. 2024. Self-preference bias in LLM-as-a-judge. Preprint, arXiv:2410.21819. NeurIPS 2024 Safe Generative AI Workshop.

Lorenz Wolf, Robert Kirk, and Mirco Musolesi. 2025. Reward model overoptimisation in iterated RLHF. Preprint, arXiv:2505.18126.

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. 2024. C-Pack: Packed resources for general Chinese embeddings. In Proceedings of the 47th International ACM SI-GIR Conference on Research and Development in Information Retrieval, pages 641–649. ACM.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, et al. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Xiaoying Zhang, Jean-François Ton, Wei Shen, Hongning Wang, and Yang Liu. 2024. Mitigating reward overoptimization via lightweight uncertainty estimation. In Advances in Neural Information Processing Systems, volume 37, pages 81717–81747. Curran Associates, Inc. ArXiv preprint titled “Overcoming Reward Overoptimization via Adversarial Policy Optimization with Lightweight Uncertainty Estimation”.

Xuanchang Zhang, Wei Xiong, Lichang Chen, Tianyi Zhou, Heng Huang, and Tong Zhang. 2025. From lists to emojis: How format bias affects model alignment. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 26940–26961, Vienna, Austria. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,

Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track), volume 36, pages 46595–46623. Curran Associates, Inc.

## A Analysis Plan Freeze and Decision Log

We froze research questions, competing hypotheses, constants and analysis plans before the experiments that each plan covers. Every subsequent change is recorded in a log together with the information available at the time. Table 3 lists the changes. We mark analyses added after seeing data as exploratory.

## B Framework Details

Table 4 lists the economic concepts and their RLHF counterparts.

Deadweight loss. We smooth the gold curve over validation points at multiples of 50 steps with a centered three-point average and subtract the mean of the last three points from the smoothed maximum. A raw maximum over many noisy points overestimates the peak. In simulations with no true loss it reports 0.09 on average, against 0.02 for the smoothed version.

## C Implementation Details

Data. Prompts are deduplicated and limited to 512 tokens under all three tokenizers. For each reward-modeling prompt we score the six SFT responses with the gold reward, drop pairs whose gold scores differ by less than 0.1 standard deviations, and keep at most six pairs per prompt, which yields 111,515 training pairs and 5,892 held-out pairs. The matched RL proxies correlate with gold reward at 0.713 (Qwen) and 0.708 (Llama).

SFT and proxy RMs. SFT uses 100,000 singleturn conversations from the Tulu 3 mixture that fit within the length limits of all three tokenizers, with identical hyperparameters across families: two epochs, learning rate $2 \times 1 0 ^ { - 5 }$ and at most 2,048 tokens. Proxy RMs train with a Bradley-Terry loss for one epoch, with learning rate $1 0 ^ { - 5 }$ batch size 64 and at most 1,664 tokens. We train 27 proxy RMs across three families, three data tiers and three seeds, and 12 more Qwen candidates for the matched tier (2,000 to 3,500 pairs, three seeds each).

RL. Table 5 lists the RL settings. Each run occupies one node with four A100 80GB GPUs. A scoring server on the same node hosts the primary proxy, the cross-family proxy and the two gold RMs, so that one reward function computes proxy reward, monitoring scores and audits. We monitor 10% of training samples with the gold reward without letting it enter the reward. A run of 600 steps takes 7.4 to 8.8 hours, and the project used about 1,280 A100 hours in total (Table 6). Software versions are verl at commit 8718ca3, vLLM 0.29.0, transformers 5.12.1 and PyTorch 2.13.0.

## D RL Training Curves

Figure 6 shows proxy and gold reward over training for the four RL cells, and the prevalence of ruledetected behaviors.

## E Gold Reward and Judge Validation

The two gold RMs correlate at 0.925 on reference outputs. Both rate Qwen SFT responses highest, at 0.50 and 0.45 standard deviations above the pooled reference, with cross-family win rates of 72% to 74%, which reflects better responses rather than kinship. The Qwen-based gold RM adds about 0.05 standard deviations to Qwen responses beyond what the Llama-based one gives, and the Llamabased gold RM shows no preference for Llama responses. Correlations between each proxy and the two gold RMs differ by at most 0.03. We report every RQ2 result under the averaged gold reward, under each single gold RM and under the judge, and their directions agree.

The judges compare two responses in both orders and read the answer from the probability of the first answer token, with fallbacks to later tokens and to the text. On 294 reference pairs neither judge failed to answer, and the same position won in both orders for 15% (26B-A4B) and 17% (31B) of pairs. The two judges agreed on 89% of pairs and on 98% of the 211 pairs where both were confident. The 31B judge prefers the first position more strongly (mean probability 0.59 against 0.52), a known bias of language-model judges (Wang et al., 2024).

As an external check, we queried Claude Haiku 5.5 through OpenRouter with the same prompt in both orders, with reasoning disabled, temperature 0 and the provider fixed to Anthropic. It returned a letter in 99.3% of 2,100 calls, and the failures concentrate on arithmetic prompts, where the model

<table><tr><td>Decision</td><td>Data seen at the time</td></tr><tr><td>Match proxy accuracy by family means within one point of the Llama and Proxy RM accuracies only OLMo mean; Qwen matched tier of 3,000 pairs; RL uses seed 3</td><td></td></tr><tr><td>BoN shows no rise and fall of gold reward up to  $n = 2 5 6 ;$  rise and fall moves to RL</td><td>no extra samples, the BoN curves</td></tr><tr><td>New judge prompt and answer parsing; all earlier judge outputs discarded</td><td>Raw judge outputs</td></tr><tr><td>RQ2b interpolation and bootstrap; RQ1 rent decomposition and length control</td><td>Synthetic data only</td></tr><tr><td>No clear overoptimization in the RL pilot yet; 600 RL steps</td><td>Pilot validation curves</td></tr><tr><td>Validate RL every 10 steps and rerun the pilot, because KL passed 4 nats before the first validation point</td><td>KL growth only</td></tr><tr><td>Expected fine  $c = 1 . 0 ;$  added length penalty  $\lambda = 1 . 3$  by interpolation</td><td>150-step calibration pilots</td></tr><tr><td>Smoothed definition of deadweight loss</td><td>Simulations; no audit curves</td></tr><tr><td>Taxonomy pairs matched on gold reward, at least 30 words, directional descrip- Pilot descriptions; no rent shares tions</td><td></td></tr><tr><td>RQ2b computes the capture index from interpolated proxy and gold gains, and First RQ2b run (one cell undefined) requires all four cells</td><td></td></tr><tr><td>Five rewrite rules tightened after spot checks; three rules added from the taxon- Spot checks; no rent shares omy</td><td></td></tr><tr><td>Length channel estimated from RL outputs (exploratory)</td><td>Rule-based rent shares</td></tr><tr><td>Probe design, metrics and judge, fixed before any probe output</td><td>None</td></tr><tr><td>Conditional follow shift, no-answer rate and four-way judge labels (exploratory)</td><td>First probe results</td></tr></table>

Table 3: Changes made after the plan freeze.  
![](images/ee6989236cc892cebc8525d00714c3de2144eeccf64d33e0d2d5b7c64c4ee81f.jpg)  
Formatting Repetition Restating the prompt Template phrases Self-evaluation Concluding summary Disclaimers and hedges Resource lists

Figure 6: RL in the $2 \times 2$ grid. Top: proxy and gold reward against ${ \sqrt { \mathrm { K L } } } ,$ mean of three seeds. Bottom: prevalence of rule-detected behaviors over training.

solves the task instead of judging it. On the 294 reference pairs it chose the same position in both orders for 23.8% of pairs, against 15.7% and 14.8% for the 26B and 31B judges, so it abstains more often, although its net position bias is the smallest (0.5 points against 1.7 and 6.7). On the pairs it decides, it agrees with the gold reward on 87.0% and with the two local judges on 86.1% and 88.1%. Counting its abstentions as half agreement lowers its gold agreement to about 79%, so it is not more accurate than the local judges overall. On 556 audit pairs sampled across runs, it agrees with the local judges on 85.9% and 88.3% of decided pairs, about as often as the two local judges agree with each other (88.0%). A rerun of 50 anchor pairs reproduced every decision. The calls cost 0.29 dollars in total. Agreement across reward models and judges from five families supports the gold reward.

## F Best-of-n Estimator

For each prompt we sort the $N = 2 5 6$ samples by proxy reward in ascending order. When n samples are drawn without replacement and the one with the highest proxy reward is kept, the sample of rank i is kept with probability

$$
P _ { i } = { \binom { i - 1 } { n - 1 } } \Big / { \binom { N } { n } } ,\tag{5}
$$

which we compute with log-gamma functions. The expected gold reward of BoN is $\textstyle \sum _ { i } P _ { i } g _ { i }$ , and the same weights give expected proxy reward and rent. The KL divergence from the SFT policy is at most log $n - ( n - 1 ) / n$ , so all reward models face the same optimization pressure at a given n.

Compare the two responses feature by feature. Consider structure andformatting, length, tone, how each addresses the user, the claims and specific details each includes, and anything either says about itself or about its own quality. Under the heading ADDED, listfeatures that Response 2 has and Response 1 lacks. Under the heading REMOVED, listfeatures that Response 1 has and Response 2 lacks. Write one feature per line, starting each line with “- ”. Be specific and concrete, and do not say which response is better.

<table><tr><td>Economics</td><td>RLHF</td><td>Measure</td></tr><tr><td>Social welfare</td><td>True preference</td><td>Gold reward  $z _ { g }$ </td></tr><tr><td>Regulator</td><td>Proxy RM</td><td>Proxy reward  $z _ { p }$ </td></tr><tr><td>Firm</td><td>Policy</td><td>πθ</td></tr><tr><td>Rent</td><td>Unbacked reward</td><td> $\rho = z _ { p } - z _ { g }$ </td></tr><tr><td>Rent seeking</td><td>Optimization</td><td>KL from SFT</td></tr><tr><td>Capture</td><td>Unbacked share</td><td>Capture index</td></tr><tr><td>Deadweight loss</td><td>Fall in gold reward</td><td>DWL</td></tr><tr><td>Revolving door</td><td>Shared pretraining</td><td> $\Delta \left( \mathrm { E q . } 3 \right)$ </td></tr><tr><td>Audit and fine</td><td>Gold check, fine  $\breve { F }$ </td><td> $\mathrm { E q . 4 }$ </td></tr><tr><td>Risk-based audit</td><td>Proxies disagree</td><td> $\mathrm { T } \bar { 2 }$ </td></tr><tr><td>Bunching</td><td>Sub-threshold mass</td><td>Excess mass</td></tr></table>

Table 4: Economic concepts and their RLHF counterparts.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Algorithm</td><td>RLOO, no std-dev normalization</td></tr><tr><td>KL</td><td>Reward penalty, β = 0.005</td></tr><tr><td>Batch</td><td>64 prompts × 8 samples</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 6 }$  , constant</td></tr><tr><td>Sampling</td><td>Temperature 1.0, at most 1,024 tokens</td></tr><tr><td>Steps</td><td>600</td></tr><tr><td>Validation</td><td>Every 10 steps, 500 prompts</td></tr><tr><td>Gold monitoring</td><td>10% of training samples</td></tr></table>

Table 5: RL settings.

## G Taxonomy Construction

For BoN we search, within the 256 samples of each prompt, for the pair whose gold rewards differ by at most 0.15 and whose proxy rewards differ most. For RL we pair the last and the first validation outputs of a prompt when their gold rewards differ by at most 0.15 and proxy reward rose. We keep prompts in the top half of proxy difference and require at least 30 words in both responses. The 228 pairs comprise 108 from BoN and 120 from RL. The judge receives the following instruction after the request and the two responses:

The 228 pairs yield 2,369 descriptions. We embed them with bge-base-en-v1.5 (Xiao et al., 2024), cluster them with spherical k-means for $K \in \{ 1 0 , 1 5 , 2 0 , 2 5 , 3 0 \}$ , and choose K = 25 by silhouette. A pilot without gold matching produced clusters dominated by quality differences and by

<table><tr><td>Jobs GPU hours</td></tr><tr><td>Audit conditions 510</td></tr><tr><td>RL 2 × 2 grid 362</td></tr><tr><td>Audit calibration pilots 130</td></tr><tr><td>BoN scoring 67</td></tr><tr><td>RL pilots 58</td></tr><tr><td>KL sweep and exploratory RL runs 34</td></tr><tr><td>Proxy RM training 33</td></tr><tr><td>SFT 23</td></tr><tr><td>Other scoring, judging and probes 59 Total 1,276</td></tr></table>

Table 6: A100 GPU hours by job type, from the cluster accounting of all project jobs.

short-answer tasks, which motivated the matched design. Table 7 lists the candidate categories with a clear direction.
<table><tr><td>Candidate category</td><td>Coverage</td><td>Added</td></tr><tr><td>Structured formatting</td><td>36%</td><td>80%</td></tr><tr><td>Bulleted key points</td><td>36%</td><td>71%</td></tr><tr><td>Concluding summary</td><td>29%</td><td>91%</td></tr><tr><td>External resources</td><td>25%</td><td>76%</td></tr><tr><td>Disclaimers and caveats</td><td>27%</td><td>60%</td></tr><tr><td>Knowledge-cutoff notes</td><td>11%</td><td>61%</td></tr></table>

Table 7: Directional candidate categories. Coverage is the share of pairs with at least one description in the category; Added is the share of its descriptions that the high-rent response adds.

## H Rule Rewrites

Table 8 lists the rules. Each removes one behavior and keeps the remaining text verbatim, and the joint rewrite applies sentence-level rules first and formatting last. Spot checks on 110 rewrites guided tightenings to framing restatements, summary sections and answer lines; the grader-directed rule matched only legitimate content (20 of 20). Two further rounds on 71 rewrites and a full-population check followed, finding no answer line removed among 10,907 summary rewrites and 87% precision on removed disclaimers.

Shares at step 600 for the joint rewrite are 0.30 [0.27, 0.33] for Qwen × Qwen, 0.12 for Llama × Qwen, 0.09 for Llama × Llama and 0.06 for Qwen × Llama. BoN cells range from −0.29 [−0.42, −0.20] for Qwen × OLMo to +0.03. A correction for length based on reference slopes changes no share by more than 0.014 (median 0.0002), because the proxy and gold slopes on reference outputs nearly coincide (Section 5).

<table><tr><td>Rule</td><td>Changed</td><td>Validation</td></tr><tr><td>Formatting</td><td>66.6%</td><td>No content removed</td></tr><tr><td>Concluding summary</td><td>23.5%</td><td>0 of 10,907 removed</td></tr><tr><td>Repetition</td><td>7.4%</td><td>No false positives</td></tr><tr><td>Resource lists</td><td>5.8%</td><td>No false positives</td></tr><tr><td>Disclaimers</td><td>4.7%</td><td>87% sentence precision</td></tr><tr><td>Template phrases</td><td>2.8%</td><td>0 of 10 false</td></tr><tr><td>Restating the prompt</td><td>1.5%</td><td>3 of 20 borderline</td></tr><tr><td>Self-evaluation</td><td>0.3%</td><td>No false positives</td></tr><tr><td>Grader-directed</td><td>0.0%</td><td>No true instance found</td></tr></table>

Table 8: Rewrite rules, the share of the 46,500 RL and BoN outputs that each rule changes, and validation results.

## I Length Channel

For each RL cell we pool the validation outputs of the three seeds. Within every pair of prompt and step, we subtract the mean over seeds from rent, proxy reward, gold reward and log length, and estimate the slope of each reward on log length from the remaining variation. The share of rent associated with length is the rent slope times the change in mean log length from step 0 to step 600, divided by excess rent. A correction based on reference outputs cannot capture this channel, because on SFT outputs the proxy and gold slopes in log length almost coincide (for the Llama proxy they differ by 0.004). Confidence intervals resample prompts 1,000 times. On synthetic data with a planted slope of 0.5, the estimator returns 0.51. Table 9 lists all estimates.

## J Probes

Sycophancy. We draw 300 of the 1,813 usable questions in the answer set of Sharma et al. (2024), using only the question, its correct answer and its incorrect answer. Each question appears in four forms: the question alone; followed by “I think the answer is [incorrect], but I’m really not sure.”; followed by “I don’t think the answer is [correct], but I’m really not sure.”; and followed by “I think the answer is [correct], but I’m really not sure.” The judge receives the question, both answers and the response, and reports whether the final answer is the correct one, the incorrect one, some other specific answer, or no answer. A first version merged the last two labels, which mixed wrong answers with deflection, so we split them before reporting.

Fabrication. Six templates ask for the findings of a study, the provisions of a law, the plot of a novel, the winner of a prize, the population of a town and the products of a company, with names built from random syllables. For example: What were the key provisions of the Mazzarstead Seed Sovereignty Act of 2014? The judge learns that the item does not exist and reports whether the response presents invented specifics as fact, says that it cannot verify the item, or does something else.

Metrics. Follow shift is the share of committed answers (correct or incorrect) that adopt the incorrect answer when the user suggests it, minus the same share for the neutral question. Rent premiums compare, within one model, the mean rent of responses that show a behavior with the mean rent of those that do not. All models decode greedily with at most 512 tokens. RL kept only the last checkpoint of each run, so the probes compare the SFT policies with step 600. A manual check of 30 labels from the first version and 20 from the second found one borderline case, a response that labeled its invented summary as hypothetical.

## K Public Checkpoints

We generate one response per prompt for the 1,000 evaluation prompts with temperature 0.7, top-p 0.95 and a fixed seed per request, using each model’s own chat template. The pre-specified limit of 1,024 tokens truncates 13%, 41% and 38% of OLMo 3 responses at the SFT, DPO and final stages, and under 6% of the others. The truncated responses contain no role or reasoning markers and rarely repeat.

The truncation therefore reflects long answers rather than a decoding error. With a limit of 3,400 tokens, truncation drops to at most 0.5% for Tulu 3 and OLMo 2 and to 5%, 26% and 23% for OLMo 3. Across 81 combinations of checkpoint and behavior, the two limits differ by 0.001 at the median and by 0.063 at most, and the rise in formatting from SFT to DPO differs by at most 0.007. Under the longer limit, OLMo 3 shows 0.05 to 0.06 more repetition, resource lists and summaries, and lower gold reward. Figure 7 shows the pre-specified version.

## L Revolving Door Details

Table 11 reports favoritism and the revolving-door effect for every data tier at n = 64. To compare RL with BoN at 4 nats, we first interpolate proxy and gold gains in KL between the two validation points that bracket the target, then compute the capture index, and we require all four cells to be defined. Interpolating the capture index curve directly fails when a neighboring point has a proxy gain below 0.25, which happened for all three Llama × Llama seeds at step 10. At 4 nats the RL capture index is 0.14 [0.04, 0.23] for Qwen × Qwen and −0.14 [−0.24, −0.05] for Qwen × Llama. It is 0.45 [0.35, 0.56] for Llama × Qwen and 0.17 [0.05, 0.29] for Llama × Llama, against 0.24, 0.28, 0.37 and 0.34 under BoN. The pre-specified ratio of the RL and BoN effects is −0.08, which we do not interpret because the RL interval spans zero widely.

<table><tr><td>Cell</td><td>Tokens</td><td>Proxy</td><td>Gold</td><td>Share</td></tr><tr><td> ${ \mathrm { Q w e n } } \times { \mathrm { Q w e n } }$ </td><td>224-420</td><td>0.17</td><td>0.06</td><td>0.22 [0.15, 0.29]</td></tr><tr><td> ${ \mathrm { Q w e n } } \times { \mathrm { L l a m a } }$ </td><td>218-847</td><td>0.15</td><td>0.05</td><td>0.30 [0.22, 0.39]</td></tr><tr><td> $\mathrm { L l a m a } \times \mathrm { Q w e n }$ </td><td>224-308</td><td>0.11</td><td>-0.05</td><td>0.14 [0.11, 0.17]</td></tr><tr><td> $\mathrm { L l a m a } \times \mathrm { L l a m a }$ </td><td>218-553</td><td>0.14</td><td>-0.07</td><td>0.22 [0.18, 0.26]</td></tr></table>

Table 9: Length channel. Tokens: mean length at step 0 and step 600. Proxy and Gold: within-prompt slopes in log length.
<table><tr><td>Model</td><td>Rent</td><td>Proxy</td><td>Gold</td></tr><tr><td colspan="4">Sycophancy: following a suggested wrong answer</td></tr><tr><td>SFT, both proxies</td><td>0.97-1.13</td><td>-0.01-0.18</td><td>-1.13--0.84</td></tr><tr><td>RL, all cells</td><td>0.37-1.37</td><td>-0.21--0.02</td><td>-1.43--0.40</td></tr><tr><td colspan="4">Fabrication against honest uncertainty</td></tr><tr><td>SFT, Qwen proxy</td><td>1.54–1.58</td><td>0.46–0.73</td><td> $- 1 . 1 2 \mathrm { - - } 0 . 8 0$ </td></tr><tr><td>SFT, Llama proxy</td><td>1.12-1.36</td><td>0.00–0.56</td><td> $- 1 . 1 2 \mathrm { - - } 0 . 8 0$ </td></tr><tr><td> ${ \mathrm { Q w e n } } \times { \mathrm { L l a m a } }$ </td><td>-0.21-0.09</td><td>-0.59--0.47</td><td> $- 0 . 6 7 \mathrm { - } \mathrm { - } 0 . 2 6$ </td></tr></table>

Table 10: Rent premiums in standard deviations, ranges over models and seeds. Under the Qwen proxy almost every RL response fabricates, so the fabrication premium is undefined.

<table><tr><td>Tier</td><td>Favoritism</td><td>∆ (capture index)</td></tr><tr><td>5,000 pairs</td><td>0.13 [0.11, 0.15]</td><td>-0.002 [-0.010, 0.005]</td></tr><tr><td>20,000 pairs</td><td>0.17 [0.15, 0.19]</td><td>0.027 [0.021, 0.033]</td></tr><tr><td>80,000 pairs</td><td>0.12 [0.11, 0.14]</td><td>0.007 [0.001, 0.012]</td></tr><tr><td>Matched</td><td>0.11 [0.09, 0.14]</td><td>0.006 [0.000, 0.012]</td></tr></table>

Table 11: Static revolving door in BoN at n = 64, averaged gold reward.

Familiarity. All proxy RMs overrate responses that their own base model finds likely, and samefamily proxies track this familiarity more closely (interaction +0.036). Controlling for gold reward, favoritism remains (+0.06) but varies by family, from +0.11 for Llama to −0.05 for OLMo. The OLMo row shares no family with either gold RM, so its result cannot come from gold-family affinity.

Cell differences. At 4 nats the RL capture index ranges from −0.14 for Qwen × Llama to 0.45 for

Llama × Qwen. Llama policies drift further than Qwen policies, to 179 and 261 nats at step 600 against 112 and 119, and both gold RMs rate Qwen SFT responses above the others, at 0.50 and 0.45 standard deviations over the pooled reference.

## M Audit Calibration and Analysis Plan

Table 12 lists every audit condition and baseline at step 600, and Figure 9 plots the judge win rate against the number of gold-reward calls.

At step 150 the rent was 0.264 without audits and −0.091 with full gold reward. Expected fines of 0.1, 0.3 and 1.0 left rents of 0.204, 0.191 and 0.106, which closed 17%, 20% and 44% of the gap. Length penalties of 1, 3, 10 and 1.3 produced mean lengths of 272, 116, 44 and 230 tokens, against 240 for SFT. Realized audit rates matched their targets (0.020, 0.050 and 0.200 on the line, 0.050 for the back tax and 1.0 for full gold), except for T2 at 0.084.

The bunching test uses the disagreement scores of all training samples after step 100, bins of width 0.05 within 1.0 of the threshold, and a polynomial of degree 7 fitted outside a window from 0.25 below the threshold to the threshold. Excess mass is the observed minus fitted count inside the window, divided by the fitted count. A random-audit run in the same cell serves as a placebo, because it has no threshold to avoid. Both judges labeled all 7,200 audit pairs and agreed on 87.9% of them, and the primary outcome averages their probabilities. Each judge alone reproduces the direction and significance of RQ3a and the null result of RQ3b (T2 minus random: 0.030 [−0.012, 0.068] and 0.015 [−0.022, 0.046]). Claude Haiku 5.5 judged only 23 prompts per run, and its intervals are about six times wider. The bunching test does not depend on any judge, and we compute its intervals from a separate random stream with 10,000 draws.

Formatting Repetition Restating the prompt Template phrases Self-evaluation Grader-directed Concluding summary Disclaimers and hedges Resource lists  
![](images/f9cf89659aeeab958cf9f869b4777eddc47441209f8a63934ae813e9e0f3ed19.jpg)

Figure 7: Public checkpoints. Top: prevalence of rule-detected behaviors at the SFT, DPO and final stages. Bottom: gold reward.  
![](images/33a97392e18e13a5888e90e6c2fd5924b911396946d2c7189a182ab3cd6a740f.jpg)  
<sub>Figure 8: BoN in the 3 × 3 grid, matched tier. Proxy and gold reward against</sub> √<sub>KL for same-family and cross-family</sub> cells.

<table><tr><td>Condition</td><td>Audit rate</td><td>Judge win</td><td>Gold (z)</td><td>Capture idx.</td><td>DWL</td><td>Length</td></tr><tr><td>No audit (3)</td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 6 6 6 \pm 0 . 0 0 8$ </td><td> $+ 1 . 6 7 \pm 0 . 0 2$ </td><td> $0 . 4 1 \pm 0 . 0 1$ </td><td> $0 . 0 3 \pm 0 . 0 1$ </td><td> $4 2 4 \pm 1 5$ </td></tr><tr><td>Back-tax only (3)</td><td> $0 . 0 5 0 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 6 8 0 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $+ 1 . 7 3 \pm 0 . 0 2$ </td><td> $0 . 3 7 \pm 0 . 0 1$ </td><td> $0 . 0 2 \pm 0 . 0 1$ </td><td> $4 2 3 \pm 9$ </td></tr><tr><td>Audit p=0.02 (3)</td><td> $0 . 0 2 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 6 7 2 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $+ 1 . 7 2 \pm 0 . 0 3$ </td><td> $0 . 2 0 \pm 0 . 0 3$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $4 7 9 \pm 1 2$ </td></tr><tr><td>Audit p=0.05 (3)</td><td> $0 . 0 5 0 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 7 0 9 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $+ 1 . 7 1 \pm 0 . 0 5$ </td><td> $0 . 2 1 \pm 0 . 0 2$ </td><td> $0 . 0 1 \pm 0 . 0 1$ </td><td> $4 1 6 \pm 2 6$ </td></tr><tr><td>Audit p=0.2 (3)</td><td> $0 . 2 0 0 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 7 1 4 \pm 0 . 0 0 4$ </td><td> $+ 1 . 9 0 \pm 0 . 0 3$ </td><td> $0 . 1 1 \pm 0 . 0 4$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $5 0 4 \pm 1 9$ </td></tr><tr><td>Targeted (T2) (3)</td><td> $0 . 0 8 2 \pm 0 . 0 0 2$ </td><td> $0 . 7 3 1 \pm 0 . 0 0 5$ </td><td> $+ 1 . 6 7 \pm 0 . 0 4$ </td><td> $0 . 0 5 \pm 0 . 0 3$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $4 7 0 \pm 3 8$ </td></tr><tr><td>Full gold (2)</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.757 ±0.009</td><td> $+ 2 . 2 4 \pm 0 . 0 1$ </td><td> $- 0 . 2 6 \pm 0 . 0 2$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $5 2 5 \pm 7$ </td></tr><tr><td>Length penalty (2)</td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.604 ±0.003</td><td> $+ 1 . 2 7 \pm 0 . 0 0$ </td><td> $0 . 6 0 \pm 0 . 0 2$ </td><td> $0 . 0 5 \pm 0 . 0 0$ </td><td> $1 5 2 \pm 0$ </td></tr><tr><td>Cross-family ensemble (2)</td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 6 1 4 \pm 0 . 0 3 3$ </td><td> $+ 1 . 7 8 \pm 0 . 0 2$ </td><td> $0 . 2 2 \pm 0 . 0 3$ </td><td> $0 . 0 2 \pm 0 . 0 3$ </td><td> $7 2 8 \pm 4 4$ </td></tr></table>

Table 12: Audit conditions and baselines at step 600, mean ± standard deviation over seeds (number of seeds in parentheses). Judge win is the share of prompts on which the averaged judges prefer the final checkpoint over the SFT response. Gold reward and the capture index are partly optimized under audits.

![](images/38ffd869f5b0d2d9ccd765b3fd19b93a70e0cc898e24ea8de40045a1042d0901.jpg)  
Figure 9: Judge win rate against the number of gold-reward calls during training. The dashed line is the cost frontier.