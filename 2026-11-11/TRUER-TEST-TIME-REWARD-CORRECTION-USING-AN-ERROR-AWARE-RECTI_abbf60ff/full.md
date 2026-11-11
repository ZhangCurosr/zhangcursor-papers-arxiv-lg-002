# TRUER: TEST-TIME REWARD CORRECTION USING AN ERROR-AWARE RECTIFIER FOR REINFORCEMENT LEARNING

Mingxiang Shen Jinhao Li Jiaming Xu Yibo Han Guohao Dai Yijia Zhang Ningyi Xu Shanghai Jiao Tong University

## ABSTRACT

Test-time reinforcement learning enables language models to adapt using rewards derived from their own answers. However, majority voting can reinforce incorrect solutions even when correct alternatives are available. Our training audit shows that false positives—incorrect responses receiving positive rewards—dominate reward errors. We focus on a practical regime where ground-truth rewards are difficult to obtain at scale, while a limited labeled source set is available. We introduce TRUER, a framework that uses limited supervision to train a reward rectifier that distinguishes correctness from consensus. To achieve this goal, we train the rectifier with an emphasis on incorrect majorities and correct minorities, targeting cases where consensus and correctness disagree. After that, we freeze the rectifier and use it to correct rewards during subsequent test-time adaptation. By improving reward accuracy and thereby model performance, TRUER consistently outperforms naive TTRL in both metrics across all evaluated configurations.

## 1 INTRODUCTION

Reliable rewards are central to reinforcement learning for language model reasoning (Shao et al., 2024; DeepSeek-AI et al., 2025). Yet ground-truth rewards can be difficult to obtain at scale. Testtime reinforcement learning (TTRL) addresses this challenge by deriving pseudo-rewards from the model’s own responses (Zuo et al., 2025). A common approach samples several solutions and uses the most frequent answer as a training target, extending the idea of self-consistency from inference to learning (Wang et al., 2022).

But consensus does not ensure correctness. Even when a correct answer appears among the model’s 32 sampled responses, majority voting can favor an incorrect answer, as illustrated by the gap between Pass@32 and Majority@32 in Figure 2(a). Majority-based rewards can then reinforce an incorrect majority while withholding reward from a correct minority. The challenge is therefore to distinguish correct answers from false consensus and use them to guide subsequent learning.

Ground-truth answers provide direct supervision for this distinction, but obtaining them at scale for target problems can be costly. Prior work uses limited supervision to assess pseudo-label quality (Li et al., 2024) and guide reinforcement learning on unlabeled problems (Yang et al., 2025b). We therefore focus on a practical regime with limited labeled source data, disjoint from the target adaptation questions, while target ground-truth rewards remain unavailable. We investigate how this source supervision can help correct erroneous consensus without accessing target ground-truth answers.

To this end, we introduce Test-Time Reward Correction Using an Error-Aware Rectifier (TRUER), which turns limited source supervision into reusable reward correction. The rectifier is trained on candidate solutions organized by both correctness and consensus membership, with emphasis on incorrect majorities and correct minorities. After training, it is frozen. During test-time adaptation, the rectifier verifies the leading answer, checks a second-ranked alternative after explicit rejection, and abstains when neither provides a reliable target. This design aims to improve reward accuracy while retaining useful learning signals.

![](images/817db8a608bfeac5012b5fdde70844412e5aef9d0f645e226832cc8e4234c81a.jpg)  
Figure 1: TRUER overview and learning dynamics. Top: limited source labels train an erroraware rectifier, which is then frozen to correct rewards during test-time adaptation. Bottom: starting from the same sampled answer distribution, naive TTRL can reinforce a wrong majority, whereas TRUER can recover a correct minority and support policy learning.

Our experiments evaluate reward quality and final model performance across several mathematical reasoning configurations, comparing TRUER with naive TTRL and an oracle reference trained with ground-truth rewards using GRPO. Our contributions are threefold.

• Reward reliability in majority-vote TTRL. We analyze the reliability of majority-based rewards in TTRL. Despite high Pass@32, consensus can miss correct candidates; our training audit further shows that reward accuracy does not improve consistently and that false positives dominate reward errors.

• Limited-supervision reward rectification. We introduce TRUER, which emphasizes incorrect majorities and correct minorities when training a rectifier on a disjoint labeled source set. The frozen rectifier rejects incorrect consensus, recovers alternatives, and abstains when needed, correcting rewards without querying target ground truth.

• Empirical evaluation and analysis. Across all evaluated configurations, TRUER improves reward accuracy and final model performance over naive TTRL, narrowing the gap to GRPO trained with ground-truth rewards. Reward audits, training trajectories, and ablations further examine how reward correction changes supervision and supports policy learning.

## 2 RELATED WORK

Test-time learning with self-generated rewards. TTRL (Zuo et al., 2025) uses majority voting to estimate labels for reinforcement learning. Its “lucky hit” effect can maintain high reward accuracy because incorrect answers differing from an incorrect pseudo-label still receive correct zero rewards. However, consensus does not ensure correctness, and rewarding incorrect consensus may reinforce the model’s existing reasoning errors.

Pseudo-label selection through consistency and reflection. Self-Harmony (Wang et al., 2025) uses a reframer to reformulate questions, then selects answers receiving consistent support across original and reframed versions. SR-TTRL (Wu et al., 2026) instead groups and summarizes candidate solutions, then prompts the model to compare their reasoning and select a pseudo-label. Mean while, SCRL (Yan et al., 2026) restricts positive rewards to sufficiently strong consensus and uses entropy-gated negative feedback to suppress unlikely answers. These methods derive corrective signals from the model’s answer distributions or self-assessment. When these signals share the model’s reasoning errors, they may provide insufficient evidence to reject incorrect majorities or recover correct minorities.

Verification training with consensus-derived supervision. CoVerRL (Pan et al., 2026) jointly trains generation and verification: consensus-derived contrastive examples supervise the verifier, whose judgments then filter pseudo-labels. Similarly, TTRL-CoCoV (Li et al., 2026) uses highconfidence majority answers to train verification and applies the resulting judgments to select pseudo-labels for low-confidence problems. Although this interaction can improve supervision, verification targets still depend on model consensus. Consequently, incorrect consensus may propagate into verification training, reinforcing existing biases and limiting corrective feedback.

Reward correction with limited supervision. CARE (Wang et al., 2026) selects problems for annotation using estimated reliability and correction value, but requires ground-truth access during adaptation. Despite labeled supervision, its acquisition classifiers use policy-derived features that may reflect model biases. Our method instead trains a reward rectifier on a labeled source set disjoint from the target problems, emphasizing incorrect majorities and correct minorities. The rectifier is then frozen and reused to correct rewards without acquiring ground-truth labels during target adaptation.

## 3 WHEN CONSENSUS FAILS TO PROVIDE RELIABLE SUPERVISION

## 3.1 CORRECT SOLUTIONS CAN BE MISSED BY CONSENSUS

Candidate coverage and answer selection are distinct aspects of inference-time scaling (Brown et al., 2024). For each question, Pass@32 measures whether a correct answer appears in a fixed pool of sampled responses, whereas Majority@32 measures whether voting selects it. Their difference identifies questions on which a correct candidate is available but overlooked. Figure 2(a) shows this gap across the inspected naive-TTRL endpoints. It is particularly pronounced for the smaller Qwen3 model on AMC23. Correct solutions are reachable, but answer frequency does not reliably identify them.

This gap motivates checking correctness before turning a popular answer into supervision, consistent with the logical-minority diagnosis of Wu et al. (2026).

## 3.2 FALSE POSITIVES DOMINATE RESIDUAL REWARD ERRORS

Let $a _ { i }$ be the answer extracted from response $y _ { i }$ , m the most frequent answer, and $a ^ { * }$ the groundtruth answer. We use majority to mean the largest answer cluster, which need not contain more than half the responses. Majority and correctness rewards are

$$
r _ { i } ^ { \mathrm { m v } } = \mathcal { k } [ a _ { i } \equiv m ] , \qquad r _ { i } ^ { * } = \mathcal { k } [ a _ { i } \equiv a ^ { * } ] ,\tag{1}
$$

where ≡ denotes mathematical equivalence. A false positive (FP) rewards an incorrect response; a false negative (FN) withholds reward from a correct response. Overall reward accuracy measures agreement between these binary rewards and true correctness.

![](images/46e956ab486e46660317a54693f7c62e8861ee2ee0bd03155c73f1b17aac0a18.jpg)

(b) Residual errors are mostly FP  
![](images/db37cec1642e38177f40c9fac35d1b0a8e2fc6865d014baf216fd4cd9c7d13d4.jpg)  
Figure 2: Available correct answers do not guarantee reliable rewards. (a) Pass@32 and Ma jority@32 use the same candidate pool at each naive-TTRL endpoint. The gap represents correct candidates missed by voting. (b) In the Qwen2.5-Math-1.5B/MATH500 training audit, false positives dominate reward errors across disjoint training windows, including late training.

Our MATH500 audit shows that continued training does not consistently improve reward accuracy. Across consecutive 15-update windows, accuracy fluctuates and ends at 94.21%, below the initial 94.67%. Meanwhile, false positives dominate reward errors in every window, accounting for 82% in the final window (Figure 2(b)). Incorrect answers outside the selected cluster still receive the correct zero reward, keeping aggregate accuracy high despite these erroneous positive rewards.

## 3.3 IMPLICATIONS FOR REWARD CORRECTION

With group-relative policy optimization (GRPO), a rewarded response receives positive outcome advantage when its group contains mixed rewards (Shao et al., 2024). A wrong majority can therefore be reinforced while a correct minority is discouraged. Constant-reward groups have no such outcome contrast.

For example, suppose that 18 of 32 responses share an incorrect answer, 10 give the correct answer, and the remaining four give other incorrect answers. Majority voting assigns reward 1 to the 18 incorrect responses and reward 0 to the correct minority. Under GRPO, this reward contrast can increase the probability of the incorrect answer, making it more likely to dominate subsequent samples. Repeated updates can thus reinforce the wrong majority and make the correct answer harder to recover.

A useful correction policy must address both sides of this failure. Rejecting an incorrect majority removes false-positive rewards; accepting a correct alternative restores a learning target. Rejection alone may leave no outcome signal. The bottom panel of Figure 1 illustrates these contrasting learning dynamics. We therefore evaluate reward accuracy together with correct-reward retention, active training groups, and model performance.

## 4 METHOD

## 4.1 PROBLEM SETUP AND OVERVIEW

We assume a labeled source set $\mathcal { D } _ { s } = \{ ( x , a ^ { * } ) \}$ and a target set $\mathcal { D } _ { t }$ whose ground-truth answers are unavailable for reward computation. Rectifier fitting and subsequent test-time adaptation use disjoint sets of questions. TRUER first trains a rectifier $V _ { \phi }$ on source candidates, then freezes $\phi$ while adapting a language model policy $\pi _ { \theta }$ on target inputs (Figure 1).

Table 1: TRUER target adaptation. The rectifier is trained on source labels before this procedure. Input: policy $\pi _ { \theta } .$ frozen rectifier $V _ { \phi }$ , target questions $\mathcal { D } _ { t }$ For each policy update:

1. Sample a batch of target questions and candidate solutions from $\pi _ { \theta }$

2. Group equivalent answers and rank the clusters by frequency.

3. Initialize $\hat { a } = \perp$ . Verify rank1 when its route is eligible and untied.

4. If rank1 is accepted, select it. If explicitly rejected, verify an eligible, untied rank2 and select it only if accepted.

5. Assign rewards using Equation 3 and update θ with GRPO.

Output: adapted policy and the unchanged rectifier.

## 4.2 LEARNING THE REWARD RECTIFIER

Each rectifier example contains a question, a candidate solution, and its extracted answer. Source ground truth determines correctness, while the sampled candidate pool determines majority member ship. Their cross-product gives four categories: majority-correct, majority-wrong, minority-correct, and minority-wrong. We emphasize the two categories where consensus and correctness disagree, while retaining examples of correct majorities and incorrect minorities.

The rectifier produces a short justification and an integer score $s \in \{ 0 , \ldots , 9 \}$ . With $p = s / 9$ and correctness label $c \in \{ 0 , 1 \}$ , its training reward is

$$
R _ { V } = 1 - 2 ( p - c ) ^ { 2 } - 0 . 1 { \Vdash } [ \mathrm { e x p l i c i t r e a s o n – s c o r e c o n f i c t } ] .\tag{2}
$$

Malformed or truncated outputs receive a fixed penalty. GRPO optimizes the rectifier using multiple judgments per candidate. Category weights emphasize incorrect majorities and correct minorities; they are applied to the policy loss after within-group advantage normalization. Neither the gold answer nor the category label is included in the rectifier input. The weighting formula and recorded training settings are given in Appendix A.3.

## 4.3 RECTIFIED REWARDS FOR TARGET ADAPTATION

The model samples responses and groups equivalent final answers. TRUER checks an eligible representative of the largest cluster, rank1. A valid score $s \geq \tau$ accepts that answer, with threshold $\tau = 5$ in our experiments. An explicit rejection $( s < \tau )$ triggers a rank2 check; an unresolved judgment does not. If neither branch selects an answer, TRUER abstains. Ties and candidate eligibility are handled by fixed rules described in Appendix A.3.

For selected answer ${ \hat { a } } ,$ with ⊥ denoting abstention, the reward and outcome advantage are

$$
r _ { i } ^ { \mathrm { T R U E R } } = \mathbb { k } [ \hat { a } \neq \bot ] \mathbb { k } [ a _ { i } \equiv \hat { a } ] , \qquad A _ { i } = \frac { r _ { i } ^ { \mathrm { T R U E R } } - \overline { { r } } ^ { \mathrm { T R U E R } } } { \mathrm { s t d } ( r ^ { \mathrm { T R U E R } } ) + 1 0 ^ { - 6 } } .\tag{3}
$$

The policy is updated with GRPO while the rectifier remains frozen (Table 1). Abstention assigns all-zero rewards and removes the group’s direct outcome advantage.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Models and tasks. We evaluate Qwen2.5-Math-1.5B and Qwen3 models with 1.7B or 4B parameters (Yang et al., 2024; 2025a), as well as Llama-3.1-8B-Instruct (Grattafiori et al., 2024), using Qwen3 rectifiers with 1.7B or 4B parameters. Table 2 contains nine configurations across MATH500 and MATH-L5, both drawn from MATH (Hendrycks et al., 2021), as well as AMC23 and AIME24. The MATH500 experiments reserve 130 questions for rectifier development and use the remaining 370 for target adaptation and evaluation. MATH-L5 uses a separate set of 600 questions. For Qwen models on AMC23 and AIME24, rectifiers use source questions from AMC2024

Table 2: TRUER main results (%). Columns report the model/task, rectifier, Pass@1, Pass@32, and reward accuracy. In each Pass@1 bar, the gray line spans the TTRL–Oracle gap and the blue line shows the TTRL–TRUER gain. Labels give the gain in percentage points (above), and the full gap in percentage points and the fraction closed (below).
<table><tr><td rowspan="2">Model / task</td><td rowspan="2">Rectifier</td><td>Pass@1</td><td></td><td colspan="2">Pass@32</td><td rowspan="2"></td><td colspan="2">Reward acc.</td></tr><tr><td>Base TTRL TRUER Oracle TTRL TRUER Oracle TTRL TRUER</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Qwen2.5-Math 1.5B / MATH500</td><td rowspan="2">Qwen3-4B</td><td>21.85 69.62 70.46</td><td>70.87 92.97</td><td></td><td>91.89</td><td>93.24</td><td>95.53</td><td>96.36</td></tr><tr><td>+0.84 pp TTRL 1.25 pp / 67.6%</td><td>Oracle</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Qwen3-1.7B AMC23</td><td rowspan="2">Qwen3-1.7B</td><td>46.09 56.09 57.11 +1.02 pp</td><td></td><td>64.6177.50</td><td>92.50</td><td></td><td>92.5074.84</td><td>86.09</td></tr><tr><td>TTRL • 8.52 pp / 11.9%</td><td>Oracle</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Qwen3-1.7B AMC23</td><td rowspan="2">Qwen3-4B</td><td>46.09 56.09 +3.28 pp</td><td>59.38</td><td>64.6177.50</td><td>92.50</td><td></td><td>92.5074.84</td><td>89.22</td></tr><tr><td>TTRL</td><td>8.52 pp / 38.5%</td><td>Oracle</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Qwen3-1.7B AIME24 Qwen3-1.7B</td><td rowspan="2">Qwen3-1.7B</td><td>13.9620.21 +1.25 pp</td><td>21.46 24.69</td><td>46.67</td><td>50.00</td><td></td><td>46.6761.41</td><td>86.80</td></tr><tr><td>TTRL 4.48 pp / 27.9%</td><td></td><td>Oracle</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">AIME24 Qwen3-4B</td><td rowspan="2">Qwen3-4B</td><td>13.9620.21 +4.58 pp</td><td>24.79 24.69</td><td>46.67</td><td>43.33</td><td></td><td>46.6761.41</td><td>88.28</td></tr><tr><td>TTRL</td><td>4.48 pp / 102.3%</td><td>Oracle</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">AMC23</td><td rowspan="2">Qwen3-4B</td><td>65.39 79.69 +2.66 pp</td><td>82.34 85.63</td><td>92.50</td><td>97.50</td><td>97.50</td><td>87.73</td><td>97.66</td></tr><tr><td>TTRL</td><td>5.94 pp / 44.7%</td><td>Oracle</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Qwen3-4B MATH-L5</td><td rowspan="2">Qwen3-4B</td><td>69.43 79.82</td><td>80.68</td><td>83.7692.50</td><td>93.00</td><td>94.83</td><td>87.81</td><td>89.92</td></tr><tr><td>TTRL</td><td>+0.86 pp 3.94 pp / 21.8%</td><td>Oracle</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Llama-3.1-8B Instruct / MATH500</td><td>Qwen3-4B</td><td>37.02 63.93</td><td>73.61</td><td>83.6867.03</td><td>85.95</td><td></td><td>91.62 75.66</td><td>96.12</td></tr><tr><td>TTRL</td><td>+9.69 pp 19.76 pp / 49.0%</td><td>Oracle</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Llama-3.1-8B Instruct / AMC23</td><td>Qwen3-4B†</td><td>12.42 49.45 60.23</td><td></td><td>88.2050.00</td><td>75.00</td><td>95.0061.48</td><td></td><td>89.48</td></tr><tr><td></td><td>+10.78 pp</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TTRL</td><td>38.75 pp / 27.8%</td><td>Oracle</td><td></td><td></td><td></td><td></td><td></td></tr></table>

<sup>†</sup>Same rectifier as the Llama/MATH500 configuration above.

and AIME2025, respectively. Llama on AMC23 reuses a frozen rectifier trained solely on MATH.   
Detailed experimental protocols are provided in Appendix A.

Comparisons and metrics. Naive TTRL implements the majority-vote reward mechanism of Zuo et al. (2025) under our experimental protocol; TRUER uses rectified rewards; Oracle is a supervised reference using target ground-truth rewards with GRPO (Shao et al., 2024) under the corresponding policy training configuration. Base is the model before adaptation. Pass@1 estimates single-sample accuracy from the saved responses, Pass@32 measures candidate coverage, and Majority@32 measures voting accuracy on the same pool (Chen et al., 2021). Reward accuracy is the fraction of binary rewards agreeing with response correctness, aggregated over the training windows specified in Appendix A. The main table reports each model’s final checkpoint; the mechanism audit instead compares reward policies on one fixed response pool.

Evaluation scope. All reported scores are measured in our own experiments, including the Base, naive TTRL, and Oracle references. We evaluate model performance on the target adaptation sets. Target labels are excluded from TRUER’s reward decisions.

![](images/1c063c4c2b193184d50fae05681ee35e4ed8a8f434021665d215c3702a57e349.jpg)

![](images/5add53eb918eb0dc987c02bd72323d7b1b0b77577c9de9211ed82abb51304c5b.jpg)  
Figure 3: Reward correction on MATH500. Qwen2.5-Math-1.5B with a Qwen3-4B rectifier. (a) FP/FN counts on identical responses; retention is relative to majority voting. (b) Reward accuracy over 15-update windows and endpoint Pass@1 on each method’s own trajectory.

## 5.2 MAIN RESULTS

Across all nine configurations in Table 2, TRUER improves reward accuracy and final Pass@1 over naive TTRL, narrowing the Pass@1 gap to the Oracle reference trained with ground-truth rewards. For example, using a Qwen3-4B rectifier, Qwen2.5-Math-1.5B on MATH500 improves reward accuracy from 95.53% to 96.36% and closes 67.6% of the TTRL–Oracle gap; Qwen3-1.7B on AMC23 improves reward accuracy from 74.84% to 89.22% and closes 38.5% of the gap. The final Llama-3.1-8B-Instruct/AMC23 row reuses a MATH-trained rectifier without AMC-specific rectifier training, suggesting some transferability across datasets.

Pass@32 varies less consistently. Where it changes little despite higher Pass@1, reinforcement learning may sharpen the answer distribution toward correct solutions already reachable by the base model, improving sampling efficiency without necessarily expanding its reasoning coverage. This interpretation is consistent with prior observations on RLVR (Yue et al., 2025).

## 5.3 CASE STUDY: REWARD CORRECTION AND MODEL ADAPTATION

We investigate Qwen2.5-Math-1.5B with a Qwen3-4B rectifier on MATH500 to examine how reward correction relates to model adaptation.

Reward accuracy and model performance. We first apply naive majority rewards and TRUER to the same 92,160 responses from the TRUER training run. As Figure 3(a) shows, total reward errors fall from 5,100 to 3,365, with 73.1% fewer false positives, although false negatives increase. TRUER retains 97.9% of the correct-positive reward count, including correct rewards recovered through rank2, and 82.3% of active training groups relative to majority voting. Vote-margin filters retain fewer correct rewards at comparable false-positive removal (Appendices C and C.2).

Across training, TRUER achieves higher reward accuracy in every 15-update window and higher Pass@1 at all six checkpoints (Figure 3(b)). Appendix B provides additional learning curves and uncertainty estimates.

A concrete case: correct minority becomes majority. Figure 4 tracks both methods on the same question at matched training steps. TRUER rejects an incorrect majority and rewards the correct minority, which later becomes dominant, while naive TTRL retains the wrong consensus. Recovery is not guaranteed: repeated abstention can leave the model incorrect. Appendix D gives both examples and their selection rules, while Appendix B.3 reports aggregate answer-distribution changes.

![](images/e5de21983fbb6cae0f90a81114185e0aa6a072160d0f5c19cfcc59ce2870a4c9.jpg)

![](images/a230591a7b2f45e7a6bcbdff28f948a4a7cddc592ad6e9c9fa2873877f83e18b.jpg)

Figure 4: From correct minority to majority. (a) Each training visit pairs naive TTRL (left, hatched) with TRUER (right, solid); stars mark correct rank2 selection. (b) Separately sampled endpoint responses. Each bar contains 32 responses; “Other responses” includes incorrect and unparsed answers.  
![](images/fee8f7c331ecdaf0fe1b47390f2dfe06a5e71d792fa05c0692fb2b0acf05363c.jpg)  
Figure 5: Rectifier reuse at deployment. For TTRL, TRUER, and Oracle, darker segments end at Pass@1 and lighter extensions reach Majority@32. Purple bars show TRUER with score-weighted answer aggregation on identical 32-candidate pools. Panels use different truncated y-axis ranges. Four configurations are shown; Appendix E reports all six.

## 5.4 ADDITIONAL EVALUATION: RECTIFIER REUSE AT DEPLOYMENT

Scoring candidate solutions is a common use of test-time computation (Cobbe et al., 2021; Snell et al., 2024). We therefore evaluate whether the rectifier trained for reward correction can also support answer selection at deployment. Given 32 sampled candidates, we group equivalent answers and sum their thresholded rectifier scores, normalized by the square root of the number of valid judgments in each group. We select the answer with the largest aggregate support, with majority fallback if no candidate passes. Figure 5 compares this strategy with single-sample answering and majority voting for naive TTRL, TRUER, and Oracle. Under this shared rule, selected through exploratory target-set replay, rectifier selection improves over TRUER majority voting in all six evaluated configurations, by 0.17–6.67 percentage points. With 32 candidates, it exceeds Oracle single-sample accuracy in all six configurations. Appendix E specifies the aggregation rule, its selection procedure, matched naive-TTRL results, and inference cost.

## 5.5 ABLATIONS AND LIMITATIONS

Effect of rectifier training. In our fixed-pool comparisons, the trained rectifier achieves higher judgment accuracy and rejects more incorrect answers than its pretrained counterpart. For example, on responses generated by naive TTRL, training increases judgment accuracy from 71.85% to 86.52% and reduces wrong-answer acceptance from 52.11% to 18.35%. Detailed results and evaluation settings are provided in Appendix C.4. For adaptation, TRUER with a trained rectifier achieves higher final Pass@1 than its untrained-rectifier variant in the three comparisons reported in Table 14.

On AIME24 with the 1.7B rectifier, the untrained variant even falls below naive TTRL (18.96% vs.   
20.21%), whereas the trained variant reaches 21.46%.

Training overhead. Although deploying the adapted model alone incurs no additional rectifier computation, reward correction increases training time. For Qwen2.5-Math-1.5B on MATH500, the recorded mean time per policy update increases from 177.2 seconds for naive TTRL to 219.3 seconds for TRUER, an overhead of approximately 23.7% with a dedicated rectifier GPU, excluding rectifier pretraining (Appendix F).

Training-budget controls. We extend naive TTRL to approximately match TRUER’s adaptationstage GPU-hours, and separately extend TRUER to match the original naive TTRL run’s cumulative number of groups with nonzero outcome advantages, to assess TRUER’s performance relative to naive TTRL under comparable computation and cumulative counts of advantage-bearing groups, respectively. Additional naive training narrows the gap in some settings, but TRUER retains higher final Pass@1 in all five GPU-hour-matched comparisons. The active-group-matched controls likewise preserve TRUER’s advantage over the original naive TTRL checkpoints. Full results and protocols are provided in Appendix F.

Evaluation scope. Our evaluation focuses on mathematical reasoning; extending reward rectification to open-ended tasks with less structured answers remains an interesting direction.

## 6 CONCLUSION

We introduced TRUER, which uses limited source supervision to correct unreliable consensus rewards in test-time reinforcement learning without access to target ground-truth labels. By emphasizing incorrect majorities and correct minorities during rectifier training, our method improves reward accuracy, which in turn leads to better model performance. Across all evaluated configurations, TRUER improves model performance over naive TTRL and narrows the gap to the Oracle reference trained with ground-truth rewards. This offers a practical path toward adaptation where ground-truth feedback for new problems is costly or infeasible.

## AI USE STATEMENT

We used generative AI tools to assist with experimental design, code implementation, and selected aspects of result analysis, as well as manuscript formatting checks, verification of cross-references and bibliographic metadata, and editorial revisions. The authors take full responsibility for the experimental design, code, reported results and their interpretation, and the final manuscript.

## REFERENCES

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Re, and Aza-´ lia Mirhoseini. Large Language Monkeys: Scaling Inference Compute with Repeated Sampling. arXiv:2407.21787, 2024. URL https://arxiv.org/abs/2407.21787.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, et al. Evaluating Large Language Models Trained on Code. arXiv:2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168, 2021. URL https://arxiv.org/abs/2110.14168.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, et al. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, 2025. URL https://arxiv.org/abs/2501.12948.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, et al. The Llama 3 Herd of Models. arXiv:2407.21783, 2024. URL https://arxiv.org/abs/2407.21783.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring Mathematical Problem Solving With the MATH Dataset. arXiv:2103.03874, 2021. URL https://arxiv.org/abs/2103.03874.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685, 2021. URL https://arxiv.org/abs/2106.09685.

Jiahui Li, Jianfeng Shan, Wenpei Chen, Shunyu Wu, Jian Lou, Wenjie Feng, Dan Li, and See-Kiong Ng. Exploiting Verification-Generation Gap: Test-Time Reinforcement Learning with Confidence-Conditioned Verification. arXiv:2606.03608, 2026. URL https://arxiv.org/ abs/2606.03608.

Siyuan Li, Weiyang Jin, Zedong Wang, Fang Wu, Zicheng Liu, Cheng Tan, and Stan Z. Li. SemiReward: A General Reward Model for Semi-supervised Learning. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ 6d1fbfed3f5758c1e5a5a6357a6aa3b1-Abstract-Conference.html.

Teng Pan, Yuchen Yan, Zixuan Wang, Ruiqing Zhang, Guiyang Hou, Wenqi Zhang, Weiming Lu, Jun Xiao, and Yongliang Shen. CoVerRL: Breaking the Consensus Trap in Label-Free Reasoning via Generator-Verifier Co-Evolution. arXiv:2603.17775, 2026. URL https://arxiv.org/ abs/2603.17775.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024. URL https://arxiv.org/abs/2402.03300.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters. arXiv:2408.03314, 2024. URL https://arxiv.org/abs/2408.03314.

Li Wang, Xiaodong Lu, Xiaohan Wang, Yikun Ban, Jiajun Chai, Wei Lin, Tianhao Peng, and Guojun Yin. When Self-Belief Misleads: Active Label Acquisition for Reinforcement Learning with Verifiable Rewards. arXiv:2605.25864, 2026. URL https://arxiv.org/abs/2605.25864.

Ru Wang, Wei Huang, Qi Cao, Yusuke Iwasawa, Yutaka Matsuo, and Jiaxian Guo. Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning. arXiv:2511.01191, 2025. URL https://arxiv.org/abs/2511.01191.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv:2203.11171, 2022. URL https://arxiv.org/abs/2203.11171.

Sitong Wu, Haoru Tan, Xichen Zhang, Bin Xia, Shaofeng Zhang, Xiaojuan Qi, Bei Yu, and Jiaya Jia. Beyond Majority Voting: Self-Reflective Test-Time Reinforcement Learning for LLM Reasoning. Author manuscript, ICML 2026, 2026. URL https://www.cse.cuhk.edu.hk/<sub>˜</sub>byu/ papers/C357-ICML2026-SR-TTRL.pdf.

Dong Yan, Jian Liang, Yanbo Wang, Shuo Lu, Ran He, and Tieniu Tan. What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time. arXiv:2603.19880, 2026. URL https://arxiv.org/abs/2603.19880.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement. arXiv:2409.12122, 2024. URL https://arxiv.org/ abs/2409.12122.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, et al. Qwen3 Technical Report. arXiv:2505.09388, 2025a. URL https://arxiv.org/ abs/2505.09388.

Shenzhi Yang, Guangcheng Zhu, Xing Zheng, Yingfan Ma, Zhongqi Chen, Bowen Song, Weiqiang Wang, Junbo Zhao, Gang Chen, and Haobo Wang. TraPO: A Semi-Supervised Reinforcement Learning Framework for Boosting LLM Reasoning. arXiv:2512.13106, 2025b. URL https: //arxiv.org/abs/2512.13106.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? In Advances in Neural Information Processing Systems, volume 38, pp. 57654–57689, 2025. doi: 10.52202/ 085713-1933. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/537d5aa768c2d534016a4d06f87bc8fb-Paper-Conference.pdf.

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, Biqing Qi, Youbang Sun, Zhiyuan Ma, Lifan Yuan, Ning Ding, and Bowen Zhou. TTRL: Test-Time Reinforcement Learning. arXiv:2504.16084, 2025. URL https://arxiv.org/abs/2504.16084.

## A EXPERIMENTAL SETUP AND PROTOCOLS

## A.1 CONFIGURATIONS AND SUPERVISION

Table 3 records the nine main-table configurations. The six matched deployment comparisons use the same score-weighted answer aggregation rule with 32 candidates and majority fallback (denoted Weighted in Table 3). Source-label budgets vary across rows.

Table 3: Configurations in the nine-row main table. Seeds describe policy training. Rectifier steps identify the frozen checkpoint, not additional policy updates. A dash denotes a configuration outside the matched deployment comparison.
<table><tr><td>Model</td><td>Target</td><td>Seed</td><td>Updates</td><td>Rectifier</td><td>Step</td><td>Deploy</td></tr><tr><td>Math-1.5B</td><td>MATH500</td><td>42</td><td>90</td><td>Qwen3-4B</td><td>150</td><td>Weighted</td></tr><tr><td>Qwen3-1.7B</td><td>AMC23</td><td>42</td><td>120</td><td>Qwen3-1.7B</td><td>198</td><td>Weighted</td></tr><tr><td>Qwen3-1.7B</td><td>AMC23</td><td>42</td><td>120</td><td>Qwen3-4B</td><td>25</td><td>Weighted</td></tr><tr><td>Qwen3-1.7B</td><td>AIME24</td><td>42</td><td>120</td><td>Qwen3-1.7B</td><td>260</td><td>Weighted</td></tr><tr><td>Qwen3-1.7B</td><td>AIME24</td><td>42</td><td>120</td><td>Qwen3-4B</td><td>312</td><td></td></tr><tr><td>Qwen3-4B</td><td>AMC23</td><td>43</td><td>120</td><td>Qwen3-4B</td><td>175</td><td>Weighted</td></tr><tr><td>Qwen3-4B</td><td>MATH-L5</td><td>43</td><td>240</td><td>Qwen3-4B</td><td>50</td><td>Weighted</td></tr><tr><td>Llama-3.1-8B</td><td>MATH500</td><td>43</td><td>120</td><td>Qwen3-4B</td><td>175</td><td></td></tr><tr><td>Llama-3.1-8B</td><td>AMC23</td><td>43</td><td>120</td><td>Qwen3-4B</td><td>175</td><td></td></tr></table>

Qwen2.5-Math-1.5B on MATH500. We reserve 130 MATH500 questions for rectifier development and use the remaining 370 for test-time adaptation and evaluation. The eligible source pool contains 3,648 candidates from 124 questions. The selected rectifier consumes 2,400 distinct candidates and generates 21,016 judgment trajectories through update 150. Target questions are excluded from rectifier fitting by ID, normalized text, and the recorded question-family check.

Source candidates were generated by a model after one warm-up round on all MATH500 inputs, without access to any ground-truth answers. Rectifier training and checkpoint selection were then performed exclusively on source data. The models used for evaluation restart from pretrained weights.

AMC23. The configurations using the 1.7B model fit rectifiers on 528 candidates from 20 AMC2024 questions, with ten further AMC2024 questions for source calibration. Test-time adaptation and evaluation use the same 40 AMC23 questions. The source and target files have no overlapping IDs, normalized texts, or digit-template families. The 4B rectifier is frozen at step 25. The main adaptation run with the 4B rectifier uses seed 42; its naive TTRL and Oracle references are shared with the 1.7B-rectifier configuration.

The stable 1.7B rectifier uses the same source data and category weights, learning rate 10<sup>−6</sup> with ten-step warmup, and KL coefficient 0.02. Its step 198 checkpoint is frozen for test-time adaptation. The model uses seed 42, with historical Naive and Oracle controls; its random stream differs from those controls. This is not an isolated rectifier-size comparison.

Training repetitions. For the Qwen3-1.7B model with the 4B rectifier on AMC23, naive TTRL, TRUER, and Oracle were each independently trained from the same pretrained policy weights for each seed (42, 43, and 44), yielding nine runs evaluated at the final update 120. The rectifier was held fixed across seeds. Table 4 reports all three methods for each seed. TRUER improves Pass@1 over the independently trained naive TTRL run with the same seed by 3.28, 5.78, and 4.84 percentage points, respectively (4.64 points on average). The main table reports seed 42 for consistency with the historical naive TTRL and Oracle controls; this is the most conservative of the three runs in terms of Pass@1 gain.

Table 4: AMC23 training repetitions: Qwen3-1.7B with the fixed 4B rectifier. Pass@1 (%) at update 120. Each row contains independently trained policies using the same training seed. ∆ is TRUER minus naive TTRL in percentage points, computed before rounding.
<table><tr><td>Seed</td><td>Naive TTRL</td><td>TRUER</td><td>Oracle</td><td> $\Delta$ </td></tr><tr><td>42</td><td>56.09</td><td>59.38</td><td>64.61</td><td>+3.28</td></tr><tr><td>43</td><td>55.23</td><td>61.02</td><td>64.30</td><td>+5.78</td></tr><tr><td>44</td><td>54.84</td><td>59.69</td><td>65.31</td><td>+4.84</td></tr><tr><td>Mean</td><td>55.39</td><td>60.03</td><td>64.74</td><td>+4.64</td></tr></table>

AIME24. Both AIME24 configurations use a Qwen3-1.7B model, 30 AIME2025 source questions for rectifier fitting, and all 30 AIME2024 questions for test-time adaptation and evaluation. The 1.7B and 4B rectifiers are frozen at epoch 5 (step 260) and epoch 6 (step 312), respectively. The two configurations share the same Base, naive TTRL, and Oracle references. All policy training runs use seed 42 and 40 epochs (120 updates), with ten questions per update and 32 responses per question. Response generation uses temperature 1, a 2,048-token prompt limit, and a 4,096-token response limit. Main-table reward accuracy uses updates 117–120: out of 1,280 assignments per arm, Naive has 786 correct, and TRUER has 1,111 with the 1.7B rectifier and 1,130 with the 4B rectifier. The latter model’s Pass@1 is 238/960, compared with 237/960 for Oracle, yielding 102.3% gap closure; Oracle is an empirical reference rather than a strict performance ceiling.

Additional 4B configurations. The 4B AMC row uses the recorded mixed-100 source rectifier. MATH-L5 uses a separate 600-question target set. The retained summary does not itemize the complete source-label budgets for these two rectifiers; neither inherits the MATH130 or AMC20 budget by assumption.

Llama-3.1-8B-Instruct on MATH500. The Llama configuration uses the fixed MATH500 split of 100 source-fitting questions, 30 source-calibration questions, and 370 target questions. We generate 64 source candidates per question with Llama before fitting the Qwen3-4B rectifier. Rectifier step 175 is selected using source calibration and frozen for test-time adaptation. All three adaptation runs use seed 43, 120 updates over ten epochs, batches of up to 32 questions with 32 responses per question, learning rate ${ \bar { 5 } } \times 1 0 ^ { - 7 }$ , three warmup steps, KL coefficient 0.001, a 4,096-token response limit, and full-parameter training.

All endpoint outputs are scored with the same retained scorer. Evaluation uses four logical shards, eight-question batches, and question-specific seeds. Deployment verification has not been measured.

Llama-3.1-8B-Instruct on AMC23. This configuration reuses the frozen Qwen3-4B step 175 rectifier from the Llama/MATH500 experiment without AMC-specific retraining. Rectifier fitting uses 100 MATH500 source questions, with 64 candidates per question generated by the original Llama model and labeled using source ground truth. All three AMC adaptation runs start from the original Llama-3.1-8B-Instruct model, use seed 43, and complete 120 updates on eight GPUs. Adaptation and endpoint evaluation use all 40 AMC23 questions, with 32 responses per question for the sampled evaluation. These are model endpoint scores; deployment rectifier selection has not been measured for this configuration.

Experimental coverage. Hardware availability and time constraints limited our coverage of the full matrix of language models, rectifiers, tasks, and training seeds. Nevertheless, TRUER achieves higher model Pass@1 and reward accuracy than naive TTRL in all nine configurations reported in Table 2. We view these results, together with the three-seed AMC23 comparison, as supporting the effectiveness of the method within the evaluated settings.

## A.2 METRICS AND RECORDED PROTOCOLS

For c<sub>q</sub> correct answers among $c _ { q }$ $N = 3 2$ responses to question q,

$$
\begin{array} { l l } { { \displaystyle { \widehat { \mathrm { P a s s @ 1 } } } = \frac { 1 } { Q } \sum _ { q } \frac { c _ { q } } { 3 2 } , \qquad ~ } } & { { \displaystyle { \widehat { \mathrm { P a s s @ 3 2 } } } = \frac { 1 } { Q } \sum _ { q } \mathbb { k } [ c _ { q } > 0 ] , } } \\ { { \displaystyle { \widehat { \mathrm { P a s s @ 2 } } } = \frac { 1 } { Q } \sum _ { q } \left[ 1 - \frac { ( 3 2 - c _ { q } ) ( 3 1 - c _ { q } ) } { 3 2 \cdot 3 1 } \right] . } } \end{array}\tag{4}
$$

(5)

Majority@32 uses equivalent-answer clusters from the same responses. Unparsed responses remain in the denominator. The main table uses the final model checkpoint, with no deployment verification in the Pass metrics. Gap closure is $( \mathrm { P a s s @ 1 _ { T R U E R } - P a s s @ 1 _ { N a i v e } } ) / ( \mathrm { P a s s @ 1 _ { O r a c l e } - P a s s @ 1 _ { N a i v e } } )$ computed before rounding.

Main-table reward accuracy covers each Qwen arm’s final four updates: 87–90 for Qwen2.5- Math-1.5B/MATH500, 117–120 for AMC23 and AIME24, and 237–240 for L5. These contain 4,096 responses per Qwen2.5-Math arm and 1,280 per other Qwen arm, including constant-reward groups. The Llama/MATH500 row uses the complete 120-update audit, with 118,400 responses per arm: naive TTRL has 69,295 TP, 27,737 FP, 20,284 TN, and 1,084 FN; TRUER has 75,309 TP, 2,208 FP, 38,500 TN, and 2,383 FN. Reward accuracy is 75.66% and 96.12%, respectively. The Llama/AMC23 row also uses all 120 updates, with 38,400 responses per arm: naive TTRL has 15,710 TP, 14,571 FP, 7,898 TN, and 221 FN; TRUER has 15,215 TP, 1,085 FP, 19,144 TN, and 2,956 FN, giving reward accuracies of 61.48% and 89.48%, respectively. These counts come from each arm’s own trajectory; fixed-pool audits instead compare labeling rules on identical saved responses.

The Qwen2.5-Math-1.5B settings appear in Table 5. AMC adaptation runs use 120 updates, ten questions per update, 32 responses per question, a 4,096-token response cap, and a 5,120-token context. The rectifier uses greedy decoding with a 4,096-token output cap and 8,192-token context. The L5 endpoint is update 240. Reward selection never receives target gold; Oracle uses it explicitly.

Table 5: Qwen2.5-Math-1.5B/MATH500 training settings.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Model</td><td>Qwen2.5-Math-1.5B, original pretrained weights</td></tr><tr><td>Policy trainable parameters</td><td>All parameters; four-GPU FSDP2</td></tr><tr><td>Policy optimizer</td><td>AdamW; weight decay 0.01; gradient clipping 1.0</td></tr><tr><td>Policy learning rate</td><td>Peak  $5 \times 1 0 ^ { - 7 } $  ; three warmup steps; cosine to step 90</td></tr><tr><td>Reference policy / KL</td><td>Frozen initial policy; separate KL loss coefficient 0.001</td></tr><tr><td>Training batch / responses</td><td>32 questions / 32 responses per question</td></tr><tr><td>Generation settings</td><td>Temperature 1; top-p 1; top-k disabled</td></tr><tr><td>Generation lengths Rectifier</td><td>Prompt 1,024; response 3,072; context 4,096 tokens</td></tr><tr><td></td><td>Qwen3-4B; LoRA (Hu et al., 2021) rank 64, scaling alpha 128</td></tr><tr><td>Rectifier optimizer</td><td>Learning rate  $5 \times 1 0 ^ { - 6 } ;$  weight decay 0.01</td></tr><tr><td>Rectifier training groups</td><td>16 candidate prompts per batch; initially 8 judgments per candidate</td></tr><tr><td>Rectifier extra sampling</td><td>Same-policy expansion to 16/32 for homogeneous- reward groups that are not all successful</td></tr><tr><td>Rectifier PPO / KL</td><td>Clip 0.2; KL coefficient 0.01</td></tr><tr><td>Rectifier decoding</td><td>Training temperature 1; native reasoning; output limit 4,096, context 8,192</td></tr><tr><td>Rectifier deployment</td><td>Frozen step 150; greedy reasoning; score  $\geq 5$  accepts</td></tr><tr><td>Evaluation</td><td>Sampled  $\bar { N } = 3 2 ^ { \circ }$  , temperature 1; greedy separately at temperature 0</td></tr><tr><td>Evaluation scheduling</td><td>Four-question batches; fixed question-specific hash seeds</td></tr></table>

MATH TTRL and TRUER resume at step 30 with optimizer, scheduler, and random states restored; rank1-only and Oracle run continuously. TTRL resolves top-count ties by sample order, whereas rectifier policies abstain. The rank1-only/TRUER ablation shares the tie rule.

Scoring and uncertainty. We retain the frozen mathematical scorer across arms. Known edge cases involving intervals, units, equations, and numeric bases are not silently corrected in the main results. An audit affected sixteen rectifier candidate labels, including four training candidates; the selected MATH rectifier was not retrained after this audit. Reviewed examples exclude known scoring errors. Paired bootstrap intervals resample question IDs 10,000 times and do not include training seed or checkpoint-selection uncertainty.

## A.3 RECTIFIER TRAINING AND SELECTION RULES

The rectifier receives the problem, candidate solution, and extracted answer. Its prompt asks for an independent check and a final JSON judgment containing a reason and score. It neither executes candidate code nor receives gold answers or category labels. Scores are learned ratings rather than calibrated probabilities. Valid outputs use Equation 2; malformed or truncated outputs receive −1.2.

For the rectifier used with Qwen2.5-Math-1.5B on MATH500, category k with $n _ { k }$ candidates and priority $\alpha _ { k }$ has weight

$$
w _ { k } = \frac { M \alpha _ { k } } { n _ { k } \sum _ { j } \alpha _ { j } } , \qquad M = 3 , 6 4 8 .\tag{6}
$$

The weight multiplies the policy loss after advantage normalization. Table 6 describes the eligible candidate pool.

Table 6: Rectifier supervision for Qwen2.5-Math-1.5B/MATH500. Correctness labels come from source ground truth; category membership is not provided as input.
<table><tr><td>Category</td><td>Eligible candidates</td><td>Priority</td><td>Loss weight</td></tr><tr><td>Majority-correct</td><td>1,621</td><td>1</td><td>0.2813</td></tr><tr><td>Majority-wrong</td><td>440</td><td>3</td><td>3.1091</td></tr><tr><td>Minority-correct</td><td>153</td><td>3</td><td>8.9412</td></tr><tr><td>Minority-wrong</td><td>1,434</td><td>1</td><td>0.3180</td></tr></table>

During test-time adaptation, rank1 is checked using its first eligible representative in sampling order. Eligibility uses parsing and length. A valid score $s \geq 5$ accepts the answer; $s \ < \ 5$ permits a rank2 check if there is no boundary tie. Missing representatives, unresolved judgments, and rejected alternatives yield abstention. Rank1-only removes the second check. Abstained groups have zero outcome advantage; the separate KL term remains active.

## B ADDITIONAL RESULTS AND LEARNING DYNAMICS

Original figure legends labeled $\mathrm { \ " { O u r s } } ^ { \prime \mathrm { { \bullet } } }$ or “Full cascade” refer to TRUER.

## B.1 MATH500 LEARNING CURVES AND CHECKPOINTS

The MATH results below retain all recorded checkpoints and the rank1 ablation. The extra update 110 continuation has a longer training budget and a changed learning-rate phase; it is supplementary to the update 90 main comparison.

Figure 6 compares the recorded learning curves, and Figure 7 provides paired uncertainty estimates.   
Table 7 lists the checkpoint scores; the rank1-only contrasts are reported in Appendix C.3.

![](images/44b5f72768b91f304a6a34c454dcfdb024bb5a5efc063cf418ea4fac28057bcc.jpg)

![](images/d028790a7e0e728819805b0455d381c0eb6d9877806a241aee0238fb4c187d9f.jpg)

![](images/89bc13323a46980ecb700f2684b6f48520a1ebeb16b928ac688132ba1d87a95b.jpg)  
Figure 6: Reward intervention and policy learning. TRUER moves Pass@1 and Majority@32 toward the ground-truth-reward reference, while Pass@32 shows a different ordering. Each line is one training seed. The figure shows all recorded checkpoints and does not select the best intermediate result.

![](images/d17814a3f7f47b6ff937df07acc91209fec1e7c2286c35263787bff83cf23460.jpg)

![](images/3f2bd94db44dadb98fd35d5881b7b8fbe27e0a78de97bca2cac6d9d8618954f0.jpg)  
Figure 7: Paired uncertainty in the MATH500 learning curves. Left: model Pass@1. Right: TRUER minus TTRL with pointwise 95% question-bootstrap intervals, conditional on the saved samples and one training seed.

Table 7: All recorded model checkpoints. Percentages, frozen scorer. No best-checkpoint selection is used for the main endpoint.
<table><tr><td>Step</td><td>Method</td><td>Greedy</td><td>Pass@1</td><td>Pass@2</td><td>Majority@32</td><td>Pass@32</td></tr><tr><td>15</td><td>TTRL</td><td>73.51</td><td>58.46</td><td>72.43</td><td>81.89</td><td>92.70</td></tr><tr><td>15</td><td>Rank1-only</td><td>72.70</td><td>57.90</td><td>72.33</td><td>82.16</td><td>92.43</td></tr><tr><td>15</td><td>TRUER</td><td>72.43</td><td>59.33</td><td>73.07</td><td>80.27</td><td>91.89</td></tr><tr><td>15</td><td>Oracle</td><td>72.16</td><td>58.56</td><td>72.49</td><td>81.62</td><td>92.16</td></tr><tr><td>30</td><td>TTRL</td><td>72.43</td><td>65.49</td><td>76.99</td><td>84.05</td><td>92.97</td></tr><tr><td>30</td><td>Rank1-only</td><td>72.97</td><td>65.46</td><td>76.77</td><td>82.70</td><td>93.24</td></tr><tr><td>30</td><td>TRUER</td><td>74.59</td><td>66.10</td><td>77.16</td><td>84.05</td><td>93.78</td></tr><tr><td>30</td><td>Oracle</td><td>74.59</td><td>66.08</td><td>77.35</td><td>83.51</td><td>93.24</td></tr><tr><td>45</td><td>TTRL</td><td>75.68</td><td>67.76</td><td>78.22</td><td>84.05</td><td>92.70</td></tr><tr><td>45</td><td>Rank1-only</td><td>76.22</td><td>68.52</td><td>78.75</td><td>84.05</td><td>93.24</td></tr><tr><td>45</td><td>TRUER</td><td>75.68</td><td>68.51</td><td>78.87</td><td>84.59</td><td>93.51</td></tr><tr><td>45</td><td>Oracle</td><td>76.76</td><td>68.55</td><td>79.07</td><td>84.32</td><td>93.24</td></tr><tr><td>60</td><td>TTRL</td><td>77.30</td><td>68.74</td><td>78.81</td><td>83.24</td><td>92.43</td></tr><tr><td>60</td><td>Rank1-only</td><td>75.68</td><td>69.21</td><td>79.04</td><td>83.78</td><td>93.51</td></tr><tr><td>60</td><td>TRUER</td><td>75.95</td><td>69.76</td><td>79.44</td><td>84.59</td><td>93.24</td></tr><tr><td>60</td><td>Oracle</td><td>78.11</td><td>70.11</td><td>79.99</td><td>85.95</td><td>92.70</td></tr><tr><td>75</td><td>TTRL</td><td>78.92</td><td>69.15</td><td>78.68</td><td>82.70</td><td>92.16</td></tr><tr><td>75</td><td>Rank1-only</td><td>77.57</td><td>69.97</td><td>79.27</td><td>83.51</td><td>93.78</td></tr><tr><td>75</td><td>TRUER</td><td>75.95</td><td>70.27</td><td>79.73</td><td>84.59</td><td>93.24</td></tr><tr><td>75</td><td>Oracle</td><td>78.65</td><td>70.46</td><td>80.03</td><td>85.95</td><td>92.97</td></tr><tr><td>90</td><td>TTRL</td><td>76.22</td><td>69.62</td><td>79.33</td><td>83.24</td><td>92.97</td></tr><tr><td>90</td><td>Rank1-only</td><td>76.22</td><td>70.04</td><td>79.51</td><td>83.78</td><td>94.32</td></tr><tr><td>90</td><td>TRUER</td><td>76.49</td><td>70.46</td><td>79.76</td><td>84.86</td><td>91.89</td></tr><tr><td>90</td><td>Oracle</td><td>77.30</td><td>70.87</td><td>80.32</td><td>85.41</td><td>93.24</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## B.2 AMC23 LEARNING CURVES AND ENDPOINTS

The following AMC figures use the seed 42 run with the 1.7B model and 4B rectifier reported in the main table. This run is distinct from the stable 1.7B-rectifier configuration. Only TRUER intermediate sampled checkpoints were retained; Naive and Oracle lines mark their endpoints.

Figure 8 shows the recorded learning curves; Table 8 reports the endpoint metrics.

![](images/27ffdfecc10f66b61b3a723bb6fc8b7f422e2e34df98491345d11102ce16483e.jpg)  
Figure 8: All 30 AMC23 TRUER checkpoints. Solid curves are separately completed, matched evaluations of each checkpoint without rectifier-based selection. Dashed lines mark the final epoch 30 TTRL and Oracle scores; their intermediate sampled checkpoints were not retained. Greedy uses the same complete evaluation pipeline rather than the separate inline training validation.

Table 8: AMC23 endpoints: 1.7B model, 4B rectifier, seed 42 (%).
<table><tr><td>Method</td><td>Greedy</td><td>Pass@1</td><td>Pass@2</td><td>Majority@32</td><td>Pass@32</td></tr><tr><td>Base</td><td>45.00</td><td>46.09</td><td>59.97</td><td>67.50</td><td>90.00</td></tr><tr><td>TTRL</td><td>57.50</td><td>56.09</td><td>61.67</td><td>65.00</td><td>77.50</td></tr><tr><td>TRUER</td><td>60.00</td><td>59.38</td><td>66.67</td><td>67.50</td><td>92.50</td></tr><tr><td>Oracle</td><td>55.00</td><td>64.61</td><td>72.91</td><td>77.50</td><td>92.50</td></tr></table>

## B.3 ANSWER-DISTRIBUTION CHANGES

The base model already samples a correct answer for 330 of the 370 MATH questions. Most additional correct samples after adaptation come from this already-covered set. Figure 9 tracks each question’s correct-answer count; no questions are selected by improvement. TRUER newly covers fifteen questions and loses five, alongside a larger increase in the frequency of correct answers. These are finite-sample coverage observations.

![](images/4bdb68fb85870159c0e82671fb9ab377a9aed36c5e77ef7b90fb21bccf5cca80.jpg)

![](images/5032977bacca7694bd2deda0b54608b9aa3f5c69674ae248d4d052c6bca9b60f.jpg)  
Figure 9: Where the same questions move after adaptation. Rows are base correct-response counts; columns are counts at update 90. Both panels contain all 370 questions and use the same bins and color scale. Movement to the right indicates more frequent correct answers.

Table 9 records parsing rates and generation lengths for the same endpoint comparison.

Table 9: Parsing and generation lengths under the base comparison. All scores retain the full denominator of 11,840 responses; unparsed responses are not discarded.
<table><tr><td>Model</td><td>Unparsed</td><td>Length-limit hits</td><td>Mean output tokens</td></tr><tr><td>Base</td><td>32.97%</td><td>11.25%</td><td>1,102.7</td></tr><tr><td>TTRL90</td><td>1.78%</td><td>2.58%</td><td>682.6</td></tr><tr><td>TRUER90</td><td>1.98%</td><td>2.52%</td><td>684.1</td></tr></table>

## C REWARD ANALYSIS AND ABLATIONS

## C.1 FIXED-POOL REWARD CORRECTION

The fixed MATH TRUER trajectory contains 92,160 responses and 2,880 question visits. Majority labeling produces 4,012 FP and 1,088 FN rewards; rectification produces 1,081 FP and 2,284 FN rewards. Reward accuracy rises from 94.47% to 96.35%. TRUER retains 55,039 originally correct positive rewards, removes 1,557, and recovers 361, leaving 55,400 in total. Active groups fall from 2,557 to 2,105.

At the question-visit level, 48 wrong labels become correct, 357 become abstentions, and 99 correct majorities are withheld. The rank2 branch accepts 48 correct and 25 wrong alternatives. Figure 10 relates these decisions to retained advantage mass and candidate availability. Counts are visits, not distinct questions.

![](images/78a4dc636da413c9e1af13d30a2a63a18ce0aa1c3b55baee95dd9c9642107f40.jpg)

![](images/451c1717a0f3793bf29e655c3ba89d17309fcb94989396fde554c32617a72b3f.jpg)  
Figure 10: Cleaner rewards do not automatically supply more corrective signal. Both panels use TRUER’s fixed training response pool. Left: retained advantage mass and active groups relative to raw majority labeling. Right: the nested conditions needed to rescue a correct rank2. Counts are question visits, not unique questions.

Table 10 separately reports the full-run reward audit on each method’s own responses.

Table 10: Training reward audit over all 90 updates, evaluated on each arm’s own responses. For an isolated labeling comparison, use the shared-pool numbers in the main text.
<table><tr><td>Quantity</td><td>TTRL</td><td>TRUER</td></tr><tr><td>TP</td><td>56,183</td><td>55,400</td></tr><tr><td>FP</td><td>4,363</td><td>1,081</td></tr><tr><td>TN</td><td>30,498</td><td>33,395</td></tr><tr><td>FN</td><td>1,116</td><td>2,284</td></tr><tr><td>Overall reward accuracy</td><td>94.05%</td><td>96.35%</td></tr><tr><td>FPR</td><td>12.52%</td><td>3.14%</td></tr><tr><td>Positive-reward precision</td><td>92.79%</td><td>98.09%</td></tr><tr><td>Correct-response reward recall</td><td>98.05%</td><td>96.04%</td></tr><tr><td>Accepted question-group coverage</td><td>100.00%</td><td>84.17%</td></tr></table>

## C.2 COMPARISON WITH VOTE FILTERS

We replay vote-count and vote-margin thresholds on the same response pool (Table 11 and Figure 11). At comparable FP removal, the rectifier retains more correct rewards and active groups than these frequency-only filters. The margin rule requires discarding about 2.7 times as many correct positive rewards to match TRUER’s FP count. At aggressive thresholds it retains less wrong-positive advantage mass, but also loses more correct-positive mass.

(c) Training signal kept  
Table 11: Label-free confidence filters on the frozen pool of Section 5.3. Only vote counts enter the selection. Precision is $\mathrm { T P } / ( \mathrm { T P } + \mathrm { F P } )$ ; the last two columns are relative to unfiltered TTRL majority. The learned cascade removes 73.1% of false positives while keeping 97.9% of correct positive rewards and 82.3% of active groups.
<table><tr><td>Policy</td><td>TP</td><td>FP</td><td>Active groups</td><td>Precision (%)</td><td>TP kept (%)</td><td>FP removed (%)</td></tr><tr><td>TTRL (no filter)</td><td>56,596</td><td>4,012</td><td>2,557</td><td>93.38</td><td>100.0</td><td>0.0</td></tr><tr><td>Vote margin δ=8</td><td>54,133</td><td>1,299</td><td>1,815</td><td>97.66</td><td>95.6</td><td>67.6</td></tr><tr><td>Vote margin δ=10</td><td>53,143</td><td>1,014</td><td>1,714</td><td>98.13</td><td>93.9</td><td>74.7</td></tr><tr><td>Vote margin δ=12</td><td>52,110</td><td>928</td><td>1,636</td><td>98.25</td><td>92.1</td><td>76.9</td></tr><tr><td>Absolute support  $\geq 2 4$ </td><td>43,692</td><td>496</td><td>1,179</td><td>98.88</td><td>77.2</td><td>87.6</td></tr><tr><td>TRUER (ours)</td><td>55,400</td><td>1,081</td><td>2,105</td><td>98.09</td><td>97.9</td><td>73.1</td></tr></table>

![](images/90823313faab1ed55b111f7127ba42d99a96dc012c25523b545deedd491dcfe0.jpg)

![](images/b1fa28a82acef5a98654f24a2484b19a1f9d962129cf2d0196ae3f9e09a406b3.jpg)

![](images/cca8cd4301b1a08782e3f343554105464054d4ccecf2d9cd43cdb6cb883160ab.jpg)  
Figure 11: Label-free confidence filters against the learned rectifier. Each curve sweeps one filter family over its threshold. At comparable FP removal, TRUER retains more correct rewards and active groups. Panel (b) restricts the pool to updates 61–90, where false positives dominate reward errors.

## C.3 RANK1-ONLY ABLATION

Table 12 reports paired endpoint contrasts between TRUER, rank1-only, and naive TTRL. The checkpoint scores are listed in Table 7.

Table 12: Paired 90-step contrasts involving rank1-only. Differences and intervals are percentage points. These were recomputed from the saved question-level scores after the rank1 run completed.
<table><tr><td>Contrast</td><td>Metric</td><td>Difference</td><td>95% interval</td></tr><tr><td>TRUER – rank1 TRUER – rank1 TRUER – rank1</td><td>Greedy Pass@1 Pass@2</td><td>+0.27 +0.42 +0.25</td><td>[-2.43, +2.97] [-0.37, +1.28] [-0.54, +1.11]</td></tr><tr><td>TRUER – rank1 TRUER – rank1</td><td>Majority@32 Pass@32</td><td>+1.08 -2.43</td><td>[-1.35, +3.51] [-4.32, -0.81]</td></tr><tr><td>Rank1 - TTRL Rank1 – TTRL</td><td>Greedy Pass@1</td><td>+0.00 +0.42</td><td>[-2.70, +2.70] [-0.37, +1.17]</td></tr></table>

## C.4 EFFECT OF RECTIFIER TRAINING

We compare the original Qwen3-4B and the trained rectifier on identical AMC23 candidate pools, using the same tokenized prompts, greedy decoding, parser, eligibility rules, and score threshold. The pools come from the base model, naive TTRL, and Oracle. Invalid outputs count as errors. Table 13 reports the numerical comparison, visualized in Figure 12.

Table 13: Original versus trained rectifier on identical AMC23 responses. Each entry is original→trained, in percent. Accuracy includes invalid outputs as errors. Wrong-answer acceptance is conditional on a wrong candidate; correct-answer withholding includes explicit rejection and invalid outputs on correct candidates. The rectifier was trained on disjoint AMC2024 questions.
<table><tr><td>Response source</td><td></td><td>Accuracy ↑ Wrong accepted ↓ Correct withheld ↓</td><td></td><td>Invalid ↓</td></tr><tr><td>Base model</td><td> $7 4 . 3 9  8 8 . 0 5$ </td><td> $3 8 . 4 7  1 4 . 8 0$ </td><td> $7 . 6 5  8 . 1 6$ </td><td> $3 . 0 9  0 . 3 3$ </td></tr><tr><td>TTRL model</td><td> $7 1 . 8 5  8 6 . 5 2$ </td><td> $5 2 . 1 1  1 8 . 3 5$ </td><td> $4 . 8 9  9 . 6 4 $ </td><td> $3 . 9 7  0 . 1 6$ </td></tr><tr><td>Oracle model</td><td> $8 2 . 8 9  9 0 . 7 9 $ </td><td> $4 8 . 3 1  1 6 . 9 2$ </td><td> $3 . 1 9  5 . 8 9$ </td><td> $1 . 8 4 \to 0 . 2 6$ </td></tr></table>

![](images/21b8303a21625503cd0f7d648940473fed20ecb1d9b2a9b2c2a057b402afdc7a.jpg)  
Figure 12: Task-specific training improves the 4B rectifier’s judgment tradeoff. Original and trained models judge the same candidates. Training reduces wrong-answer acceptance but also withholds more correct answers.

Training improves judgment accuracy and reduces wrong-answer acceptance across these pools, with some increase in correct-answer withholding. Paired question-bootstrap intervals exclude zero for accuracy improvements and wrong-answer acceptance reductions. This control evaluates judgments; it does not isolate category weighting or replace a matched adaptation run using the untrained rectifier.

Adaptation with an untrained rectifier. Table 14 compares adaptation using the original pretrained rectifier and its trained counterpart. All three reported configurations start from the original Qwen3-1.7B policy and use seed 42, 120 policy updates, and the original endpoint evaluation with 32 responses per question. The trained-rectifier results use the original endpoints rather than the budget-control continuations. The trained rectifier yields higher final Pass@1 in each comparison; the untrained 1.7B rectifier on AIME24 also underperforms naive TTRL.

Table 14: Adaptation with and without rectifier training. Final model Pass@1 (%) without rectifier-based answer selection. All policies are Qwen3-1.7B; untrained denotes the original pretrained rectifier without task-specific training.
<table><tr><td>Target</td><td>Rectifier</td><td>Naive TTRL</td><td>Untrained</td><td>Trained</td></tr><tr><td>AMC23</td><td>Qwen3-4B</td><td>56.09</td><td>59.22</td><td>59.38</td></tr><tr><td>AIME24</td><td>Qwen3-1.7B</td><td>20.21</td><td>18.96</td><td>21.46</td></tr><tr><td>AIME24</td><td>Qwen3-4B</td><td>20.21</td><td>23.02</td><td>24.79</td></tr></table>

These comparisons have execution differences: the AMC23/4B and AIME24/4B untrained runs use policy data parallelism of four rather than two, while AIME24/1.7B changes from RTX 5090 with vLLM 0.23 to A800 with vLLM 0.25.1. They therefore do not isolate rectifier training under identical hardware and runtime settings.

## D CASE STUDIES

We select the successful example by the largest endpoint correct-sample gap among reviewed questions where naive TTRL is unanimously wrong and TRUER has at least 75% correct responses. The failure example has the largest combined wrong-mode share among reviewed cases where both models have at least 75% wrong consensus. Ties use question IDs. Known scorer errors and mixed-truth clusters are excluded.

Recovering a correct minority. For MATH algebra/2023, the question asks for the smallest integer in the domain of $\sqrt { x - 1 } / { ( x - 2 ) }$ . The constraints are $x \geq 1$ and $x \neq 2 ,$ , so the answer is 1; a common mistake gives 3. TRUER rejects 3 and selects the minority answer 1 at updates 10, 13, and 34. Later visits have a correct majority. Endpoint samples contain 24 correct answers for TRUER, while naive TTRL answers 3 in all 32 samples.

Filtering without recovery. For intermediate algebra/1279, summing the roots of $\textstyle \sum _ { j = 1 } ^ { 4 } j / ( x ^ { 2 } -$ $j ) = 2 0 1 0 x - 4$ gives 2/1005 by Vieta’s formula after clearing denominators. Both models commonly answer zero. TRUER removes the wrong majority reward on all eight visits but abstains each time; the few correct candidates are blocked by rank2 tie rules. Neither model produces a correct endpoint sample (Figure 13). Cleaner rewards can prevent an erroneous update without producing a corrective one.

![](images/3daa3691ad409180ea36859c9b05003bb724ff49cbdaaf236cbec826f3157e35.jpg)

![](images/c545233d71583d0977b0db6f85a27ba2b19e81bdb69168370771668eb2921b14.jpg)  
Figure 13: Filtering without recovery. (a) Answer distributions at all eight recorded TRUER training visits; crosses mark abstention. (b) Separately sampled endpoint responses for Base, naive TTRL, and TRUER. The named wrong answer is zero. Rejecting this answer supplies no replacement target, and the model remains incorrect.

An incorrect fallback. At update 64 on intermediate algebra/1849, the rectifier rejects rank1 answer 2 but accepts rank2 answer 1 with score 9. The true answer is 501, absent from the candidate pool. The judgment misses a family of negative integer cases. Rejecting one wrong answer therefore does not certify its alternative.

## E DEPLOYMENT SELECTION AND COST

## E.1 SHARED SELECTION RULE

The current deployment comparison uses one score-weighted answer aggregation rule for all six configurations. We retain each final model’s original pool of 32 responses and apply the configuration’s frozen rectifier to both naive TTRL and TRUER. A response is eligible when its final answer can be parsed and clustered, the model response is not truncated, and the rectifier prompt fits the 4,096-token input limit. Rectifier judgments use greedy decoding, a 4,096-token output cap, and an 8,192-token context. A valid integer score of at least 5 passes; malformed or truncated judgments do not pass.

For each answer a, let $\nu _ { a }$ contain the eligible candidates with that answer and a valid rectifier score $s _ { i } ,$ , and let $n _ { a } = | \gamma _ { a } |$ . We compute

$$
S ( a ) = \frac { \sum _ { i \in \mathcal { V } _ { a } } \operatorname* { m a x } ( s _ { i } - 4 , 0 ) } { \sqrt { n _ { a } } } , \qquad n _ { a } > 0 .\tag{7}
$$

Valid rejected judgments contribute zero to the numerator but remain in $n _ { a } ;$ invalid judgments are excluded from both. The weights are used for ranking and are not calibrated probabilities. We select the answer with the largest positive support. Ties are resolved by the highest passing score in the answer cluster, then its original total candidate count, and finally the earliest index among its highest-scoring passing candidates. If no candidate passes, we return the original majority answer. Every question remains in the denominator.

Selector choice and scope. We selected this shared aggregation rule after comparing alternative selectors on the target response pools. The resulting deployment comparisons are exploratory and do not constitute independent validation of the selector choice. Candidate selection itself receives no target labels. The model checkpoints, rectifiers, candidate pools, eligibility criteria, and answer scorers remain fixed; this replay changes only how the saved judgments are aggregated. In particular, the AMC23 configuration with a 1.7B model and 4B rectifier uses the original seed 42 candidate pools and judgments.

Table 15 reports the paired naive TTRL and TRUER results. The shared aggregation rule improves over each TRUER model’s own majority voting in all six configurations. With matched rectifier postprocessing, TRUER is higher than naive TTRL in three configurations, tied in one, and lower in two. Historical selection controls below retain their original rules and results.

Table 15: Deployment under one shared selection rule (%). Parentheses identify the rectifier size. Both adaptation methods use the same frozen rectifier within each row. “+Rectifier” applies Equation 7 to the corresponding model’s fixed 32-response pool, with majority fallback.
<table><tr><td colspan="4">Naive TTRL</td><td colspan="2">TRUER</td></tr><tr><td>Model / target (rectifier)</td><td>N</td><td>Majority</td><td>+Rectifier</td><td>Majority</td><td>+Rectifier</td></tr><tr><td>Math-1.5B / MATH500 (4B)</td><td>370</td><td>83.24</td><td>88.11</td><td>84.86</td><td>86.76</td></tr><tr><td>Qwen3-1.7B / AMC23 (1.7B)</td><td>40</td><td>65.00</td><td>65.00</td><td>67.50</td><td>70.00</td></tr><tr><td>Qwen3-1.7B / AMC23 (4B)</td><td>40</td><td>65.00</td><td>70.00</td><td>67.50</td><td>70.00</td></tr><tr><td>Qwen3-1.7B / AIME24 (1.7B)</td><td>30</td><td>26.67</td><td>30.00</td><td>26.67</td><td>33.33</td></tr><tr><td>Qwen3-4B / AMC23 (4B)</td><td>40</td><td>82.50</td><td>85.00</td><td>90.00</td><td>95.00</td></tr><tr><td>Qwen3-4B / MATH-L5 (4B)</td><td>600</td><td>84.83</td><td>87.17</td><td>86.00</td><td>86.17</td></tr></table>

## E.2 HISTORICAL SELECTION CONTROLS

Table 16 uses the same AMC1.7B/4B-rectifier seed 42 pools as the main table and the deployment comparison. It additionally tests top2 with one representative per cluster and top2 with all eligible responses in those clusters. These policies and the All32 selector rank passing candidates by score,

cluster size, and sample order. None of the historical AMC selection gains establishes a statistically significant improvement under the recorded paired tests.

Table 16: Historical AMC23 final answer-selection accuracy (%). Every row uses the same 32 candidates for each model, and full question coverage. Changes do not alter the models’ sampled Pass@1 or Pass@32.
<table><tr><td>Selection rule</td><td>Base</td><td>TTRL</td><td>TRUER</td><td>Oracle</td></tr><tr><td>Original majority</td><td>67.50</td><td>65.00</td><td>67.50</td><td>77.50</td></tr><tr><td>Sequential rank1-to-rank2</td><td>72.50</td><td>67.50</td><td>67.50</td><td>80.00</td></tr><tr><td>Top2: one representative</td><td>72.50</td><td>67.50</td><td>65.00</td><td>80.00</td></tr><tr><td>Top2: all candidates</td><td>70.00</td><td>65.00</td><td>70.00</td><td>80.00</td></tr><tr><td>All32: all candidates</td><td>70.00</td><td>65.00</td><td>75.00</td><td>80.00</td></tr></table>

Table 17 reports the historical MATH fixed-pool selection controls, including the additional continuation described in Appendix B.

Table 17: Historical MATH fixed-pool selection controls (percent). Oracle receives matched postprocessing here.
<table><tr><td>Selection rule</td><td>TRUER90</td><td>TRUER110</td><td>Oracle90</td></tr><tr><td>Original majority</td><td>84.86</td><td>85.14</td><td>85.41</td></tr><tr><td>Rank1-to-rank2 cascade</td><td>85.68</td><td>86.22</td><td>86.22</td></tr><tr><td>Top2, score first</td><td>85.68</td><td>85.95</td><td>86.76</td></tr><tr><td>Top3, score first</td><td>85.68</td><td>85.68</td><td>87.84</td></tr></table>

## E.3 INFERENCE COST

The weighted aggregation rule reuses the same at most 32 candidate judgments per question as All32. No new judgments are generated for this replay. Table 18 records the judgments and rectifier output tokens for the current paired evaluations; these totals exclude prompt tokens and the unchanged response generation.

Table 18: Rectifier inference cost for the unified deployment comparison. Counts cover all eligible responses in each fixed pool.
<table><tr><td colspan="3">Naive TTRL</td><td colspan="2">TRUER</td></tr><tr><td>Model / target (rectifier)</td><td>Judgments</td><td>Output tokens</td><td>Judgments</td><td>Output tokens</td></tr><tr><td>Math-1.5B / MATH500 (4B)</td><td>11,463</td><td>7,409,697</td><td>11,457</td><td>7,382,694</td></tr><tr><td>Qwen3-1.7B / AMC23 (1.7B)</td><td>1,261</td><td>1,501,618</td><td>1,207</td><td>1,366,736</td></tr><tr><td>Qwen3-1.7B / AMC23 (4B)</td><td>1,261</td><td>1,110,284</td><td>1,173</td><td>939,509</td></tr><tr><td>Qwen3-1.7B / AIME24 (1.7B)</td><td>864</td><td>208,525</td><td>768</td><td>169,730</td></tr><tr><td>Qwen3-4B / AMC23 (4B)</td><td>1,194</td><td>936,227</td><td>1,170</td><td>925,414</td></tr><tr><td>Qwen3-4B / MATH-L5 (4B)</td><td>17,801</td><td>11,997,996</td><td>17,359</td><td>8,656,022</td></tr></table>

On historical AMC TRUER candidates, top2 representatives require 67 judgments and 58,074 rectifier output tokens, while All32 requires 1,173 judgments and 939,509 output tokens. These counts exclude input tokens and response generation. Figure 14 shows the recorded tradeoff.

AMC23 / Ours epoch 30: identical 32 candidates, full coverage  
![](images/ccbf3d0e55272e57cc8f905a4454ef1e02a1323d7e431353674bf8c0e41b5b36.jpg)  
Figure 14: Selection quality and additional rectifier cost on AMC23. The same frozen rectifier scores TRUER epoch 30 candidates. Output-token counts are sums from the cached judgments, excluding rectifier input tokens and the unchanged response generation. Calls count candidate judgments. All policies retain all 40 questions.

## F TRAINING-BUDGET CONTROLS

We conduct two independent continuation controls on configurations from the main table. The GPU-hour control extends naive TTRL and compares it with the original TRUER checkpoint. The active-group control extends TRUER and compares it with the original naive TTRL checkpoint; we also report its change from the original TRUER endpoint. The two extended endpoints do not simultaneously match computation and effective training exposure.

Training overhead. The recorded MATH policy updates take approximately 219.3 seconds for TRUER versus 177.2 for naive TTRL, with a dedicated rectifier GPU; this excludes rectifier training. Matching policy updates is not total-compute matching.

## F.1 GPU-HOUR-MATCHED NAIVE TTRL

Each run starts from the original naive TTRL endpoint and continues until its cumulative adaptationstage GPU-hours approximately match those of the original TRUER run. The stopping budget is fixed before endpoint evaluation. Table 19 reports all five configurations. The final naive budgets are 1.1–1.4% above the corresponding TRUER budgets, reflecting the budget margin and stopping at complete updates. Additional naive training improves Pass@1 in three configurations and reduces it in two; the original TRUER checkpoints remain higher in all five comparisons.

Table 19: GPU-hour-matched naive continuation. $N _ { 0 }$ and $T _ { 0 }$ are the original naive TTRL and TRUER endpoints; $N _ { + }$ is the extended naive TTRL endpoint. GPU-hours cover adaptation and online rectifier inference, excluding rectifier pretraining, saving, export, and evaluation.
<table><tr><td rowspan="2">Model / task</td><td rowspan="2">Seed</td><td colspan="3">Pass@1 (%)</td><td colspan="3">GPU-hours</td></tr><tr><td> $N _ { 0 }$ </td><td> $N _ { + }$ </td><td> $T _ { 0 }$ </td><td> $N _ { 0 }$ </td><td> $N _ { + }$ </td><td> $T _ { 0 }$ </td></tr><tr><td>Qwen3-1.7B / AMC23</td><td>42</td><td>56.09</td><td>59.14</td><td>59.38</td><td>9.78</td><td>27.65</td><td>27.31</td></tr><tr><td>Qwen3-1.7B / AIME24</td><td>42</td><td>20.21</td><td>23.75</td><td>24.79</td><td>19.70</td><td>39.3938.96</td><td></td></tr><tr><td>Qwen3-4B / AMC23</td><td>43</td><td>79.69</td><td>78.52</td><td>82.34</td><td>27.10</td><td>30.64 30.21</td><td></td></tr><tr><td>Qwen3-4B / MATH-L5</td><td>43</td><td>79.82</td><td>80.36</td><td>80.68</td><td>56.45</td><td>68.75</td><td>67.84</td></tr><tr><td>Llama-3.1-8B / AMC23</td><td>43</td><td>49.45</td><td>47.50</td><td>60.23</td><td>29.26</td><td>63.69 62.97</td><td></td></tr></table>

## F.2 ACTIVE-GROUP-MATCHED TRUER

An active group contains 32 responses with nonconstant binary rewards and hence nonzero grouprelative outcome advantages. We extend each original TRUER checkpoint until its cumulative active-group count reaches that of the original naive TTRL run, stopping at the first complete update that crosses the target. This matches effective group exposure, not optimizer-step counts or accumulated parameter-gradient norms. Counts are independently checked against the saved per-response rewards at every update. Table 20 reports the active-group-matched results.

Table 20: Active-group-matched TRUER continuation. $T _ { + }$ is the extended TRUER endpoint. Group counts include the original training; the target is the original naive TTRL count. GPU-hours are cumulative. Seeds and original scores are given in Table 19.
<table><tr><td rowspan="2">Model / task</td><td colspan="3">Active groups</td><td colspan="3"></td></tr><tr><td> $T _ { 0 }$ </td><td>Target</td><td> $T _ { + }$ </td><td></td><td>Added steps Pass@1 (%)</td><td>GPU-hours</td></tr><tr><td>Qwen3-1.7B / AMC23</td><td>636</td><td>930</td><td>935</td><td>65</td><td>61.02</td><td>43.46</td></tr><tr><td>Qwen3-1.7B / AIME24</td><td>530</td><td>1,179</td><td>1,180</td><td>144</td><td>27.92</td><td>121.80</td></tr><tr><td>Qwen3-4B / AMC23</td><td>569</td><td>651</td><td>653</td><td>21</td><td>81.72</td><td>38.00</td></tr><tr><td>Qwen3-4B / MATH-L5</td><td>1,094</td><td>1,197</td><td>1,197</td><td>24</td><td>80.74</td><td>76.15</td></tr><tr><td>Llama-3.1-8B / AMC23</td><td>415</td><td>630</td><td>630</td><td>139</td><td>64.53</td><td>140.13†</td></tr></table>

<sup>†</sup>The Llama GPU-hour entry is a lower bound; the recorded cumulative cost is 140.13–140.60 GPUhours, including discarded computation after checkpoint recovery. Active-group counts include only the retained training trajectory.

All five active-group-matched endpoints remain above the original naive TTRL scores. Relative to the original TRUER checkpoints, Pass@1 improves in four configurations, with gains of 3.13 percentage points on Qwen3-1.7B/AIME24 and 4.30 points on Llama/AMC23. Qwen3-4B/AMC23 decreases by 0.63 points, while Qwen3-4B/MATH-L5 changes little (+0.06 points). Thus, additional effective training benefits most configurations, but does not uniformly improve TRUER.

## F.3 CONTINUATION AND EVALUATION PROTOCOLS

Both controls restart Adam from the corresponding final model weights while retaining the original base reference model. The TRUER continuations retain the frozen rectifier. Budgets determine stopping independently of evaluation scores. Endpoint evaluation retains each original configuration’s questions, scoring rules, and full 32-sample plus greedy protocol. GPU-hours count active training time multiplied by the participating physical GPUs, including online rectifier services and counting shared GPUs only once. They exclude rectifier pretraining, checkpoint saving, export, and endpoint evaluation. Hardware allocations and parallel topologies differ in some continuations, so these are approximate adaptation-compute controls rather than hardware-normalized FLOP comparisons or from-scratch runs under a common total budget.

These controls use one training seed per configuration. Conditional question-bootstrap 95% intervals for the TRUER-minus-extended-naive Pass@1 difference are [0.94, 7.03] points on Qwen3- 4B/AMC23, [−10.23, 11.17] on Qwen3-1.7B/AMC23, and [−1.88, 27.81] on Llama/AMC23. For extended-minus-original TRUER, the corresponding available intervals are [−0.55, 3.67] on Qwen3- 1.7B/AMC23 and [−2.19, 1.09] on Qwen3-4B/AMC23. These intervals condition on the retained runs and do not include training-seed variation.

## G REPRODUCIBILITY

Figures and tables use retained question-level scores, reward audits, and response histories. Maintable data, reward-window counts, figure inputs, and source hashes are preserved with the analysis. Historical configurations and exploratory selection comparisons are identified separately. The continuation controls additionally retain budget ledgers and complete endpoint evaluations.