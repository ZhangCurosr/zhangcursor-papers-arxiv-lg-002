![](images/adfd9299a71da294d56118dc68a853ea001eedab7fa78429c0fca4b7a1a722cc.jpg)

# RATIO: REASONING ANALYSIS AND TOKEN-LEVEL INFERENCE OPTIMIZATION FOR QUANTIZED REASON-ING MODELS

Chengzhu Bao<sup>1\*</sup>, Xianglong Yan<sup>1\*</sup>, Tianao Zhang<sup>1</sup>, Jiaqi Chen<sup>1</sup>, Shaoqiu Zhang<sup>1</sup>, Yulun Zhang<sup>1†</sup> <sup>1</sup>Shanghai Jiao Tong University

## ABSTRACT

Post-training quantization (PTQ) has become a widely adopted technique for reducing the memory footprint and inference cost of large language models (LLMs). However, recent studies reveal that when applied to reasoning models, PTQ not only degrades reasoning performance but also exacerbates overthinking, leading to longer reasoning trajectories. These issues may offset the efficiency gains expected from lower-precision inference. Existing approaches mainly rely on complex optimization procedures. More recent lightweight inference strategies instead use predefined overthinking markers, limiting their adaptability across quantized models. To address these issues, we propose Reasoning Analysis and Tokenlevel Inference Optimization (RATIO), a framework that identifies model-specific overthinking tokens and assigns each a tailored penalty. RATIO first introduces Quantization-aware Reasoning Behavior Analysis (QRBA) to identify overthinking tokens by analyzing discrepancies between full-precision and quantized models. It then adopts Token-Specific Penalty Determination (TSPD), which leverages full-precision guidance to derive token-specific penalties without additional training. Extensive experiments show that RATIO achieves a better accuracy-efficiency trade-off than existing token-level interventions. Specifically, RATIO achieves up to 9.8 points accuracy improvement and reduces chain-of-thought (CoT) length by up to 51.3% compared with quantized baselines. The code will be available at https://github.com/steven-bao1/RATIO.

## 1 INTRODUCTION

Large language models (LLMs) have achieved remarkable performance across a wide range of tasks, with increasing model size playing an important role in their progress. Therefore, modern LLMs (Kimi Team, 2026; DeepSeek-AI et al., 2026; GLM-5 Team, 2026) continue to grow to further improve performance. However, this growth increases memory requirements and inference costs, which hinder the practical deployment of LLMs. Post-training quantization (PTQ) (Frantar et al., 2023; Lin et al., 2024; Ashkboos et al., 2024b; Yan et al., 2026a) mitigates these costs by representing model parameters at lower precision, reducing memory usage and per-step decoding costs. These benefits have made PTQ a widel

Figure 1: Performance comparison of different strategies across three quantized reasoning models. RATIO consistently achieves higher accuracy gains and larger CoT length reductions.

costs. These benefits have made PTQ a widely adopted approach to LLM deployment.

However, applying PTQ to reasoning models introduces new challenges, as large reasoning models (LRMs) rely on extended inference-time computation and generate long reasoning trajectories to solve complex tasks. First, low-bit quantization can introduce representation errors that degrade the final accuracy of reasoning models (Liu et al., 2025). More importantly, recent studies (Lotfi et al.,

![](images/70edda5dbc2ff6bc0bf81504007b2b3078764a1c878f5311e410d222ab747a01.jpg)  
(a) Selected reasoning tokens

![](images/64bf55f76ce160c198d892cd9668ff02495487282536637aac6db9b71b6da41a.jpg)  
(b) Token-specific penalties  
Figure 2: Target tokens and calibrated penalties for Qwen-1.5B. (a) Word cloud of the 21 selected tokens, repeated for visualization. (b) Token-specific logit penalties in descending order.

2026; Lian et al., 2026) have found that quantization can exacerbate overthinking behaviors, such as hesitation and repeated verification. This can lead to longer chains of thought and greater token usage, potentially offsetting the inference efficiency gains expected from quantization.

To address these issues, recent studies have explored various approaches, which can be broadly categorized into optimization-based methods and inference-time interventions. Optimization-based approaches typically adopt sophisticated strategies, such as task-specific fine-tuning (Li et al., 2025b), full-precision model guidance (Alimaskina et al., 2026), or dynamic precision adjustment (Chen et al., 2026), to recover the reasoning capability of quantized models. Recently, a lightweight decoding-time strategy (Lotfi et al., 2026) has been proposed to mitigate inefficient reasoning behaviors by penalizing manually selected overthinking markers. However, such marker-based strategies rely on predefined token lists and cannot adapt to model-specific quantization-induced reasoning changes, as overthinking tokens may vary across models. Moreover, tokens exhibit different degrees of reasoning deviation, making uniform penalties insufficient to capture token-level differences.

In this work, we propose Reasoning Analysis and Token-level Inference Optimization (RATIO), a framework that discovers model-specific overthinking tokens and performs token-specific calibration for quantized reasoning models. RATIO introduces Quantization-aware Reasoning Behavior Analysis (QRBA) to discover overthinking tokens through a two-stage process: Quantization-Sensitive Token Identification (QSTI), which identifies candidate tokens with token-level shifts, and Reasoning-context-aware Token Validation (RTV), which further selects tokens associated with inefficient reasoning behaviors based on actual quantized models’ trajectories. Based on the selected tokens, Token-Specific Penalty Determination (TSPD) leverages the full-precision model as guidance to derive token-specific penalties according to token-level generation discrepancies. As shown in Fig. 2, QRBA identifies a set of model-specific reasoning tokens for Qwen-1.5B, while TSPD assigns a distinct penalty to each token. By adapting token selection and penalty strength to quantized models, RATIO suppresses unnecessary reasoning tokens while preserving or improving reasoning accuracy. As shown in Fig. 1, on DeepSeek-R1-Distill-Qwen-1.5B under GPTQ-W3, RATIO improves average accuracy by 3.58 percentage points and reduces CoT length by 25.7%, outperforming the best fixed-penalty baseline on both metrics.

## Our main contributions are summarized as follows:

• We propose RATIO, a training-free framework for quantized reasoning models that enables token-level calibration without additional inference overhead.

• We introduce Quantization-aware Reasoning Behavior Analysis (QRBA), which discovers model-specific overthinking tokens through token-level discrepancy analysis and reasoning-context-aware validation on actual reasoning trajectories.

• We develop Token-Specific Penalty Determination (TSPD), which leverages the fullprecision model as a reference to derive token-specific penalties and calibrates quantized models’ reasoning behaviors with tailored strengths.

• Extensive experiments across multiple reasoning benchmarks, model scales and quantization settings demonstrate that RATIO effectively reduces unnecessary reasoning while preserving or improving the reasoning accuracy, achieving a better accuracy-efficiency tradeoff over the corresponding quantized baselines.

## 2 RELATED WORKS

## 2.1 POST-TRAINING QUANTIZATION

Post-training quantization (PTQ) has become a widely adopted technique for efficient LLM deployment. By converting high-precision parameters into low-bit representations, PTQ reduces memory and computation costs without requiring additional training. Existing PTQ methods can be broadly categorized into mixed-precision, compensation-based, and transformation-based approaches. Mixed-precision methods (Zhao et al., 2024; Bai et al., 2025; Huang et al., 2025; Saxena et al., 2025) adaptively allocate varying bit-widths based on quantization sensitivity. For example, Quik (Ashkboos et al., 2024a) retains outlier channels at higher precision while quantizing most weights and activations to 4 bits. Compensation-based methods (Li et al., 2025a; Kim et al., 2025; Arai & Ichikawa, 2025) such as GPTQ (Frantar et al., 2023) reduce reconstruction error by employing Hessian-based optimization to adjust full-precision weights. Transformation-based methods, including SmoothQuant (Xiao et al., 2023), AWQ (Lin et al., 2024), and QuaRot (Ashkboos et al., 2024b), apply equivalent transformations or rotations to alleviate outliers and improve low-bit quantization robustness. Beyond methods developed for integer quantization, recent works (Chen et al., 2025; Bao et al., 2026; Yan et al., 2026b) have investigated microscaling floating-point formats such as MXFP4 and NVFP4. For instance, MR-GPTQ (Egiazarian et al., 2026) further improves FP4 quantization through discrete scaling search and block-wise Hadamard rotation. However, conventional quantization evaluation mainly focuses on accuracy and per-token generation speed, which is incomplete for reasoning models because the number of generated tokens is itself a major component of the total computational cost.

## 2.2 EFFECT OF QUANTIZATION ON REASONING

Recent studies have begun to investigate the effects of quantization on reasoning models. Empirical analyses (Li et al., 2025b; Liu et al., 2025) report degradation in reasoning accuracy and show that its severity varies with the model, quantization setting, and task. Further work (Lian et al., 2026) finds that quantization can also increase the number of reasoning tokens, even when answer accuracy is preserved. To improve the reasoning capability of quantized models, existing approaches have explored different recovery and optimization strategies. Some methods (Li et al., 2025b; Zhang et al., 2026) leverage additional training signals, such as task-specific adaptation or fine-tuning-guided quantization, to recover quantized reasoning performance. Other approaches focus on inferencetime adaptation, where ASTRO (Chen et al., 2026) reduces reasoning costs through adaptive precision allocation and termination strategies, while FP16 planning and loop rescue (Alimaskina et al., 2026) mitigate generation failures in extremely low-bit reasoning models. More closely related to our work, a recent decoding-time method (Lotfi et al., 2026) reduces unnecessary reasoning by penalizing a manually specified set of overthinking markers with a shared logit bias. However, such fixed marker-based strategies overlook how strongly a token is associated with inefficient reasoning in different contexts and how much its probability shifts under quantization across models. To address this, RATIO identifies model-specific reasoning tokens and assigns tailored penalties using full-precision guidance, enabling token-specific calibration without additional training.

## 3 METHOD

## 3.1 OVERVIEW OF RATIO

In this section, we introduce our method RATIO, as illustrated in Fig. 3. To analyze and mitigate quantization-induced changes in reasoning behavior, for each base model, RATIO compares its full-precision version with two quantized variants produced by AWQ and GPTQ. RATIO first per forms Quantization-aware Reasoning Behavior Analysis (QRBA), as detailed in Sec. 3.2, to discover model-specific overthinking tokens by combining token-level discrepancies and reasoning-context information. Based on the identified tokens, RATIO further performs Token-Specific Penalty De termination (TSPD), as described in Sec. 3.3, to derive token-specific penalties for inference-time calibration under full-precision guidance.

![](images/c7702f9f13b22c1e236fa5242fe852ae3f79446c46c6f05185e92562059ea7fb.jpg)  
Figure 3: Overview of RATIO. The two panels on the left illustrate QRBA, which discovers model-specific quantization-sensitive reasoning tokens by combining token-level discrepancies with reasoning-context information. The rightmost panel presents TSPD, which leverages full-precision guidance to derive token-specific penalties for inference-time calibration.

## 3.2 QUANTIZATION-AWARE REASONING BEHAVIOR ANALYSIS

## 3.2.1 QUANTIZATION-SENSITIVE TOKEN IDENTIFICATION.

To identify quantization-sensitive tokens, we perform a controlled comparison between fullprecision and quantized models under identical reasoning contexts. Given a fixed reference reasoning trajectory, the full-precision model and its quantized counterparts are evaluated with the same input prefix using teacher forcing, eliminating the influence of different generation trajectories. For each reasoning position, we measure the next-token probability shift between a quantized model Q and the full-precision model F:

$$
\delta _ { Q , i } ( k ) = \log p _ { Q } ( k | x _ { < t _ { i } } ) - \log p _ { F } ( k | x _ { < t _ { i } } ) ,\tag{1}
$$

where k denotes a token retained from the union of the two models’ top-p candidate sets at position i, and $x _ { < t _ { i } }$ represents their shared prefix before the i-th position. Further details of the candidate sets are provided in the Appendix C.1.

For each token k, we aggregate its occurrence-level probability shifts across reasoning positions and compute the average shift separately for each quantized model:

$$
\bar { \delta } _ { Q } ( k ) = \frac { 1 } { N _ { Q } ( k ) } \sum _ { i } \delta _ { Q , i } ( k ) ,\tag{2}
$$

where $N _ { Q } ( k )$ denotes the number of response positions where token k is retained from the union of the top-p candidate sets for Q and F. To reduce quantizer-specific effects, we compare the average shifts from AWQ and GPTQ and retain tokens with positive shifts in both models, subject to additional support and stability criteria detailed in the Appendix C.1, for subsequent reasoning-contextaware validation. Note that this stage only captures quantization-related token-level distribution changes; the identified shifts alone do not necessarily indicate inefficient reasoning behaviors.

## 3.2.2 REASONING-CONTEXT-AWARE TOKEN VALIDATION

Token shifts identified on fixed reference trajectories reveal how quantization changes next-token distributions, but they do not determine whether the affected tokens are associated with inefficient reasoning behaviors. We therefore further validate the tokens obtained from QSTI using freely generated reasoning trajectories from the quantized models.

For each quantized model, we first generate reasoning trajectories on quantized models, while an independent full-precision rollout on the same questions only provides correctness information. We then compare the quantized and full-precision models on the generated quantized trajectories: the quantized model follows its own generated trajectory, while the full-precision model evaluates the same prefixes through teacher forcing. This enables token-level probability comparison under identical reasoning states while avoiding the influence of different free-generation trajectories.

Numerical Evidence in RTV  
![](images/8e76af020e12c4433d569f5efb00a9df193f86520ddd77740f35484d4ba5976f.jpg)  
Figure 4: Numerical evidence construction in RTV. (A) Quantitative behavioral validation compares token probabilities under the same quantized prefixes. (B) Trajectory-level validation examines associations with incorrect answers and repetitive reasoning. Here, Q and F denote the quantized and full-precision models, respectively.

Quantitative Behavioral Validation. We first examine whether the quantization-related shifts identified on fixed trajectories persist during actual generation. For each token obtained from QSTI, we measure token-level probability deviations during quantized responses from two perspectives. Actual-token analysis evaluates the probability difference assigned to the token generated by the quantized model, while next-token preview analysis compares the top-p candidate distributions of the quantized and full-precision models at the same generation position. The former captures deviations of sampled tokens, while the latter captures tokens that are favored by quantization even when they are not selected during generation.

Trajectory-level Validation. Beyond quantitative behavioral validation, we further examine whether the identified tokens are associated with inefficient reasoning trajectories. Specifically, we analyze whether these tokens exhibit stronger associations with incorrect reasoning cases by comparing their occurrences between quantized-failure trajectories and successful reasoning trajectories. We additionally measure their association with repetitive reasoning patterns, including explicit loops and other forms of repetition. Together with quantitative behavioral validation, these trajectory-level statistics provide numerical evidence linking candidate tokens to inefficient reasoning outcomes. Figure 4 illustrates the construction of these numerical signals. Detailed formulations of these trajectory-level measurements are provided in the Appendix C.2.

Contextual Validation. Both preceding components rely on numerical measurements of token shifts and trajectory outcomes. However, such evidence may still be insufficient to distinguish inefficient deliberation from useful reasoning operations, such as verification and self-correction. We therefore inspect the surrounding reasoning contexts of candidate tokens to better understand their roles within reasoning trajectories. Importantly, contextual validation is only applied after tokens satisfy the corresponding numerical criteria in each selection stage. It determines token retention o exclusion but does not introduce additional tokens beyond the numerically qualified candidates.

Selection Strategy. Based on the above validation signals, Reasoning-context-aware Token Validation (RTV) adopts a staged token selection strategy. Tokens that satisfy strong quantitative behavioral criteria and sufficient support across AWQ and GPTQ are first considered as high-confidence candidates, followed by contextual validation to determine their retention. For borderline tokens with weaker quantitative behavioral evidence, additional trajectory-level and contextual validation signals are considered to determine whether they should be retained. Through this process, RTV obtains a model-specific token set for subsequent Token-Specific Penalty Determination.

## 3.3 TOKEN-SPECIFIC PENALTY DETERMINATION

After identifying the final quantization-sensitive reasoning token set, RATIO performs Token-Specific Penalty Determination (TSPD) to adjust the generation preference of each token during inference. Unlike previous approaches that apply a shared penalty, TSPD derives token-specific calibration penalties according to the deviation between quantized and full-precision models.

TSPD derives its calibration signal from the fixed reference trajectories used in QSTI. Let K denote the final model-specific token set obtained by QRBA. Given a selected token $k \in K$ , the fullprecision model and each quantized model are evaluated under the same reference prefix $x _ { < t _ { i } }$ using teacher forcing. For a quantized model Q and the full-precision model F, we obtain:

$$
p _ { Q } ( k | x _ { < t _ { i } } ) , \qquad p _ { F } ( k | x _ { < t _ { i } } ) ,\tag{3}
$$

where $Q$ denotes either the AWQ or GPTQ variant. Since TSPD only suppresses tokens whose preference is increased after quantization, we retain events satisfying:

$$
\begin{array} { r } { p _ { Q } ( k | x _ { < t _ { i } } ) > p _ { F } ( k | x _ { < t _ { i } } ) . } \end{array}\tag{4}
$$

Instead of directly using probability differences, TSPD derives the correction magnitude in the logit space:

$$
c _ { Q , i } ( k ) = \mathrm { l o g i t } ( p _ { Q } ( k | x _ { < t _ { i } } ) ) - \mathrm { l o g i t } ( p _ { F } ( k | x _ { < t _ { i } } ) ) ,\tag{5}
$$

where

$$
{ \mathrm { l o g i t } } ( p ) = \log { \frac { p } { 1 - p } } .\tag{6}
$$

The derivation of this logit-space correction is provided in the Appendix C.3. This quantity represents the token-level correction magnitude required to align the quantized preference of token k with its full-precision preference.

We use the median of correction magnitudes across occurrences to obtain the token-level correction strength, and further combine the values from AWQ and GPTQ through conservative selection:

$$
\begin{array} { r } { m _ { Q } ( k ) = \operatorname* { m e d i a n } _ { i \colon p _ { Q } ( k | x _ { < { t _ { i } } } ) > p _ { F } ( k | x _ { < { t _ { i } } } ) } c _ { Q , i } ( k ) , \qquad r _ { k } = \operatorname* { m i n } \{ m _ { \mathrm { A W Q } } ( k ) , m _ { \mathrm { G P T Q } } ( k ) \} . } \end{array}\tag{7}
$$

Finally, we normalize the correction magnitudes within the selected token set:

$$
M _ { K } = \mathrm { m e d i a n } _ { j \in K } r _ { j } , \qquad \lambda _ { k } = \frac { r _ { k } } { M _ { K } } .\tag{8}
$$

During inference, the corresponding token logits are adjusted by:

$$
z _ { k } ^ { \prime } = z _ { k } - \lambda _ { k } , \quad k \in K .\tag{9}
$$

By assigning different calibration strengths to different tokens, TSPD provides a lightweight inference-time mechanism to correct quantization-induced token preference shifts.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Models and Quantization Settings. We evaluate RATIO on five reasoning models, including DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-14B, DeepSeek-R1-Distill-Llama-8B, and Qwen3-4B (Yang et al., 2025) in thinking mode. The three Qwen-based distilled models and DeepSeek-R1-Distill-Llama-8B are built on the Qwen2.5 (Yang et al., 2024) and Llama 3.1 (Grattafiori et al., 2024) architectures. For each model, we consider the full-precision BF16 model and two representative post-training quantization settings: AWQ (Lin et al., 2024) and GPTQ (Frantar et al., 2023). Both AWQ and GPTQ use 3-bit weight quantization with a group size of 128. Additional details about quantization configurations are provided in the Appendix B.2.

Dataset Analyses. We collect reasoning trajectories from several reasoning datasets for token identification and validation. For QSTI, we use fixed MATH-CoT reference trajectories (Hendrycks et al., 2021) to provide controlled reasoning contexts. For RTV, we generate reasoning trajectories on AIME (Dekoninck et al., 2026), GPQA-Diamond (Rein et al., 2024), MATH-500 (Lightman et al., 2024), and GSM8K (Cobbe et al., 2021), with 50 questions randomly selected from each benchmark for each quantized model. The analysis subsets of MATH-500 and GSM8K are sampled from their corresponding training splits. During RTV, the models are provided only with the input questions, without access to any reference answers.

Table 1: Accuracy (%) and CoT length (k tokens) for BF16 and W3-quantized models using GPTQ or AWQ, with shared-penalty decoding (+ Lotfi et al. (2026)) or our method (+RATIO). The ∆ columns report the accuracy change in percentage points and the relative CoT length change (%) compared with the corresponding quantized model without intervention. For each model, the fixedpenalty baseline uses the penalty strength that achieves the highest mean accuracy across AWQ and GPTQ, averaged over the five benchmarks.
<table><tr><td colspan="2"></td><td>AIME</td><td>MATH</td><td>GSM8K Acc (%) / Len (k)</td><td>GPQA</td><td>HumanEval</td><td>Avg. Acc / Len</td><td>∆Acc</td><td>∆Len%</td></tr><tr><td colspan="2">BF16</td><td colspan="6"></td><td colspan="2">vs. baseline</td></tr><tr><td rowspan="6">wen--.5B</td><td>GPTQ</td><td>20.83/25.11</td><td>85.60/6.06</td><td>84.69/2.81</td><td>36.36/9.97 31.82/23.67</td><td>73.17/6.50</td><td>60.13/10.09</td><td></td><td></td></tr><tr><td></td><td>4.17/51.53</td><td>54.00/24.27 58.20/18.48</td><td>68.76/9.54 70.81/6.22</td><td>27.27/21.14</td><td>10.37/39.31 20.73/32.16</td><td>33.82/29.66 36.24/25.11</td><td></td><td></td></tr><tr><td>+ Lotfi et al. (2026)</td><td>4.17/47.53 5.00/42.55</td><td>61.40/14.06</td><td>71.49/3.11</td><td>30.81/20.64</td><td>18.29/29.90</td><td>37.40/22.05</td><td>+2.42 +3.58</td><td>-15.34</td></tr><tr><td>+RATIO</td><td></td><td></td><td></td><td>23.23/36.56</td><td>17.68/39.81</td><td></td><td></td><td>-25.66</td></tr><tr><td>AWQ</td><td>7.50/55.53</td><td>44.80/36.13 56.20/19.36</td><td>61.18/22.16</td><td>28.28/28.76</td><td>29.27/31.43</td><td>30.88/38.04 37.79/25.86</td><td></td><td></td></tr><tr><td>+ Lotfi et al. (2026) +RATIO</td><td>6.67/41.60 6.67/32.52</td><td>62.80/13.86</td><td>68.54/8.17 72.18/3.63</td><td>26.26/19.87</td><td>35.37/22.68</td><td>40.66/18.51</td><td>+6.91 +9.78</td><td>-32.02 -51.34</td></tr><tr><td rowspan="7">ew-B</td><td>BF16</td><td>42.50/14.27</td><td>93.60/3.83</td><td>91.96/1.50</td><td>52.02/8.60</td><td>85.98/4.87</td><td>73.21/6.61</td><td></td><td></td></tr><tr><td>GPTQ</td><td>25.00/33.84</td><td>86.80/11.06</td><td>88.63/3.55</td><td>39.39/11.50</td><td>71.95/9.49</td><td>62.35/13.89</td><td></td><td></td></tr><tr><td>+ Lotfi et al. (2026)</td><td>24.17/34.57</td><td>84.80/9.25</td><td>88.25/2.43</td><td>42.93/11.74</td><td>81.10/6.70</td><td>64.25/12.94</td><td>+1.90</td><td>-6.84</td></tr><tr><td>+RATIO</td><td>25.83/29.15</td><td>89.60/6.30</td><td>88.40/2.37</td><td>45.96/9.81</td><td>75.00/7.36</td><td>64.96/11.00</td><td>+2.61</td><td>-20.81</td></tr><tr><td>AWQ</td><td>30.83/19.91</td><td>87.80/6.40</td><td>89.69/2.33</td><td>43.43/9.96</td><td>76.83/8.34</td><td>65.72/9.39</td><td></td><td></td></tr><tr><td>+ Lotfi et al. (2026)</td><td>31.67/17.62</td><td>89.00/5.50</td><td>90.14/1.85</td><td>46.46/9.16</td><td>81.10/7.76</td><td>67.67/8.38</td><td>+1.95</td><td>-10.76</td></tr><tr><td>+RATIO</td><td>32.50/16.41</td><td>88.60/5.07</td><td>89.46/1.82</td><td>45.96/8.45</td><td>81.10/4.98</td><td>67.52/7.34</td><td>+1.80</td><td>-21.83</td></tr><tr><td rowspan="7">Owen-114B</td><td>BF16</td><td>52.50/13.01</td><td>95.20/3.74</td><td>94.47/1.44</td><td>62.12/8.21</td><td>93.29/3.18</td><td>79.52/5.92</td><td></td><td></td></tr><tr><td>GPTQ</td><td>36.67/14.92</td><td>92.20/4.06</td><td>92.34/1.42</td><td>48.48/7.18</td><td>93.90/4.23</td><td>72.72/6.36</td><td></td><td></td></tr><tr><td>+ Lotfi et al. (2026)</td><td>43.33/13.27</td><td>91.00/3.92</td><td>92.95/1.30</td><td>53.54/7.48</td><td>95.12/3.77</td><td>75.19/5.95</td><td>+2.47</td><td>-6.45</td></tr><tr><td>+RATIO</td><td>35.00/11.55</td><td>91.80/3.21</td><td>92.95/1.10</td><td>57.58/6.19</td><td>92.68/2.74</td><td>74.00/4.96</td><td>+1.28</td><td>-22.01</td></tr><tr><td>AWQ</td><td>39.17/14.29</td><td>93.20/4.52</td><td>93.33/1.77</td><td>54.55/8.65</td><td>92.68/4.50</td><td>74.59/6.75</td><td></td><td></td></tr><tr><td>+ Lotfi et al. (2026)</td><td>38.33/16.09</td><td>92.00/4.43</td><td>93.63/1.68</td><td>50.51/8.44</td><td>91.46/4.36</td><td>73.19/7.00</td><td>-1.40</td><td>+3.70</td></tr><tr><td>+RATIO</td><td>40.83/14.08</td><td>92.40/3.65</td><td>93.48/1.28</td><td>55.05/7.32</td><td>90.85/3.20</td><td>74.52/5.91</td><td>-0.07</td><td>-12.44</td></tr><tr><td rowspan="7">I1a-8B</td><td>BF16</td><td>35.83/16.26</td><td>89.00/4.75</td><td>87.87/1.81</td><td>46.97/9.20</td><td>81.10/3.88</td><td>68.15/7.18</td><td></td><td></td></tr><tr><td>GPTQ</td><td>15.00/24.57</td><td>73.60/8.62</td><td>74.37/1.73</td><td>34.85/10.80</td><td>70.73/7.99</td><td>53.71/10.74</td><td></td><td></td></tr><tr><td>+ Lotfi et al. (2026)</td><td>15.00/24.81</td><td>73.20/6.30</td><td>75.06/1.38</td><td>37.88/10.60</td><td>68.90/7.11</td><td>54.01/10.04</td><td>+0.30</td><td>-6.52</td></tr><tr><td>+RATIO</td><td>15.00/23.58</td><td>71.40/6.33</td><td>73.77/1.52</td><td>37.37/9.64</td><td>75.00/6.80</td><td>54.51/9.57</td><td>+0.80</td><td>-10.89</td></tr><tr><td>AWQ</td><td>18.33/16.40</td><td>78.00/5.39</td><td>83.40/2.43</td><td>34.34/8.14</td><td>72.56/5.99</td><td>57.33/7.67</td><td></td><td></td></tr><tr><td>+ Lotfi et al. (2026)</td><td>15.83/12.93</td><td>75.40/5.78</td><td>82.11/1.93</td><td>33.33/7.82</td><td>73.78/4.94</td><td>56.09/6.68</td><td>-1.24</td><td>-12.91</td></tr><tr><td>+RATIO</td><td>16.67/17.87</td><td>76.80/4.95</td><td>83.32/2.06</td><td>36.36/8.03</td><td>73.78/3.77</td><td>57.39/7.33</td><td>+0.06</td><td>-4.43</td></tr><tr><td rowspan="7">Oe-4B</td><td>BF16</td><td>71.67/16.91</td><td>97.40/5.35</td><td>94.39/2.32</td><td>52.02/6.60</td><td>90.85/4.60</td><td>81.27/7.15</td><td></td><td>一</td></tr><tr><td>GPTQ</td><td>16.67/32.10</td><td>81.20/12.05</td><td>90.52/3.41</td><td>35.35/15.30</td><td>54.88/20.20</td><td>55.72/16.61</td><td>一</td><td>一</td></tr><tr><td>+ Lotfi et al. (2026)</td><td>20.00/31.08</td><td>80.20/11.70</td><td>90.90/3.27</td><td>32.32/15.19</td><td>57.93/18.36</td><td>56.27/15.92</td><td>+0.55</td><td>-4.15</td></tr><tr><td>+RATIO</td><td>19.17/28.53</td><td>80.40/10.72</td><td>90.52/2.98</td><td>35.86/12.80</td><td>70.12/13.65</td><td>59.21/13.74</td><td>+3.49</td><td>-17.28</td></tr><tr><td>AWQ</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ Lotfi et al. (2026)</td><td>26.67/23.90 25.00/22.75</td><td>85.60/7.94 87.40/7.30</td><td>89.69/3.65 91.43/3.11</td><td>38.38/9.62 40.91/10.34</td><td>71.95/12.28 76.22/10.82</td><td>62.46/11.48 64.19/10.87</td><td></td><td>-5.31</td></tr><tr><td>+RATIO</td><td>26.67/22.36</td><td>86.60/7.01</td><td>90.60/2.95</td><td>37.88/8.81</td><td>76.22/8.89</td><td>63.59/10.00</td><td>+1.73 +1.13</td><td>-12.89</td></tr></table>

Evaluation and baseline. We evaluate RATIO on five reasoning benchmarks, including AIME (Dekoninck et al., 2026), GPQA-Diamond (Rein et al., 2024), MATH-500 (Lightman et al., 2024), GSM8K (Cobbe et al., 2021), and HumanEval (Chen et al., 2021). We report answer accuracy to measure reasoning performance. To evaluate inference efficiency, we additionally measure the generated chain-of-thought length, defined as the number of generated tokens before the final answer. We compare RATIO with quantized baselines without calibration and the decoding-time intervention method (Lotfi et al., 2026) based on manually specified overthinking markers. For fixedpenalty baselines, the same penalty strength is applied to both AWQ and GPTQ for each model, consistent with the shared calibration setting of RATIO. All experiments use default decoding with T = 0.6 and top-p = 0.95, following the setting used by prior work (Liu et al., 2025) on quantization of reasoning models. All experiments are conducted on NVIDIA RTX A6000 GPUs, each with 48GB of memory, and use a maximum generation budget of 65,536 tokens.

![](images/c7a4da7d47ee8ab1bc436dfb166207b611ab75d3c6937bbb75cdb781689cd508.jpg)  
(a) Accuracy–efficiency trade-off

![](images/473da25d94c6981bacad6423128ec28c8268aa4040c0c9114f626e40c7918b0d.jpg)  
(b) Selected-token occurrence rates  
Figure 5: Accuracy–efficiency trade-off and selected-token occurrence analysis. (a) Changes in average accuracy and CoT length for fixed-penalty decoding with different penalty strengths and RATIO on Qwen-7B, measured relative to the corresponding uncalibrated AWQ-W3 and GPTQ-W3 baselines. RATIO occupies the upper-left region (shorter CoT and higher accuracy) under both quantizers. (b) Benchmark-wise occurrence rates of QRBA-selected tokens on Qwen-1.5B with AWQ-W3 under RATIO, its component ablations, and the fixed-penalty baseline. RATIO consistently produces lower occurrence rates across benchmarks, showing that it effectively suppresses the selected target tokens during reasoning.

## 4.2 MAIN RESULTS

Overall Accuracy. We first evaluate the impact of RATIO on reasoning accuracy. As shown in Table 1, low-bit quantization leads to accuracy degradation, while the effectiveness of the fixedpenalty method varies across models and quantizers. In contrast, RATIO generally preserves or improves the accuracy of quantized models. For example, on Qwen-1.5B, RATIO improves the average accuracy of AWQ-W3 from 30.88% to 40.66% and that of GPTQ-W3 from 33.82% to 37.40%. Similar improvements are observed on larger models under different quantization settings. On Qwen3-4B under GPTQ-W3, RATIO improves average accuracy from 55.72% to 59.21%. These results demonstrate that RATIO effectively mitigates quantization-induced reasoning degradation across diverse reasoning models.

Reasoning Efficiency. We further evaluate the effect of RATIO on reasoning length. As shown in Table 1, RATIO consistently reduces the average CoT length of the evaluated models under both AWQ and GPTQ. On Qwen-1.5B, for example, it achieves reductions of 51.34% under AWQ-W3 and 25.66% under GPTQ-W3 relative to their uncalibrated counterparts. Reductions are also observed across the other model scales, demonstrating the general effectiveness of RATIO in controlling reasoning length. Moreover, RATIO generally produces shorter reasoning trajectories than the fixed-penalty baseline, improving the inference efficiency of quantized reasoning models. For instance, on Qwen-7B under GPTQ-W3, RATIO reduces average CoT length by 14.99% relative to the fixed-penalty baseline while also achieving higher average accuracy. These results support RATIO’s effectiveness in reducing CoT length across models and quantizers.

Overall, RATIO achieves a more favorable accuracy–efficiency trade-off for quantized reasoning models. As illustrated in Fig. 5, RATIO substantially shortens reasoning trajectories while generally preserving or improving accuracy over uncalibrated quantized models, providing a better overall balance than fixed-penalty strategies. These results demonstrate the effectiveness of RATIO for accurate and efficient quantized reasoning.

## 4.3 ABLATION STUDY

Effect of QRBA. We first assess the contribution of QRBA by replacing the model-specific token set identified by QRBA with the 50 manually selected overthinking markers from prior work (Lotfi et al., 2026), while retaining TSPD. As shown in Table 2, this variant consistently underperforms the complete RATIO framework. For example, under AWQ-W3, this variant reduces average accuracy by 3.27 percentage points and increases average CoT length by 49.27% compared with the complete RATIO framework. These results demonstrate the importance of QRBA in identifying model-specific tokens that can be effectively calibrated.

Table 2: Effect of QRBA and TSPD. Component-wise ablation on Qwen-1.5B under AWQ-W3 and GPTQ-W3. QRBA applies uniform penalties to the identified tokens, while TSPD applies tokenspecific penalties to the manually selected markers from prior work. Each entry reports accuracy (%) and CoT length (k tokens), with ∆ columns showing changes relative to the corresponding quantized baseline.
<table><tr><td></td><td></td><td>GPQA</td><td>AIME</td><td>GSM8K Acc (%) / Len (k)</td><td>MATH</td><td>HumanEval</td><td>Avg. Acc / Len</td><td>∆Acc vs. baseline</td><td>∆Len%</td></tr><tr><td rowspan="10">AWQ</td><td></td><td>23.23/36.56</td><td>7.50/55.53</td><td>61.18/22.16</td><td>44.80/36.13</td><td>17.68/39.81</td><td>30.88/38.04</td><td></td><td></td></tr><tr><td>QRBA (λ = 0.5)</td><td>29.80/32.37</td><td>4.17/48.73</td><td>69.29/9.17</td><td>59.40/20.21</td><td>34.76/29.86</td><td>39.48/28.07</td><td>+8.60</td><td>-26.21</td></tr><tr><td>QRBA (λ = 1.0)</td><td>29.29/26.25</td><td>4.17/42.11</td><td>70.36/4.08</td><td>62.60/14.10</td><td>31.10/26.39</td><td>39.50/22.59</td><td>+8.62</td><td>-40.62</td></tr><tr><td>TSPD</td><td>20.71/31.74</td><td>7.50/47.74</td><td>66.19/10.20</td><td>59.00/20.21</td><td>33.54/28.25</td><td>37.39/27.63</td><td>+6.51</td><td>-27.37</td></tr><tr><td>RATIO</td><td>26.26/19.87</td><td>6.67/32.52</td><td>72.18/3.63</td><td>62.80/13.86</td><td>35.37/22.68</td><td>40.66/18.51</td><td>+9.78</td><td>-51.34</td></tr><tr><td>GPTQ</td><td>31.82/23.67</td><td>4.17/51.53</td><td>68.76/9.54</td><td>54.00/24.27</td><td>10.37/39.31</td><td>33.82/29.66</td><td></td><td></td></tr><tr><td>QRBA (λ = 0.5)</td><td>30.30/24.70</td><td>5.83/48.91</td><td>72.18/4.79</td><td>57.20/18.74</td><td>17.68/31.72</td><td>36.64/25.77</td><td>+2.82</td><td>-13.12</td></tr><tr><td>QRBA (λ = 1.0)</td><td>30.30/22.81</td><td>6.67/40.04</td><td>72.25/3.13</td><td>61.00/15.29</td><td>23.17/28.29</td><td>38.68/21.91</td><td>+4.86</td><td>-26.13</td></tr><tr><td>TSPD</td><td>33.84/19.76</td><td>2.50/46.07</td><td>71.04/5.93</td><td>57.80/19.63</td><td>19.51/32.20</td><td>36.94/24.72</td><td>+3.12</td><td>-16.66</td></tr><tr><td>RATIO</td><td>30.81/20.64</td><td>5.00/42.55</td><td>71.49/3.11</td><td>61.40/14.06</td><td>18.29/29.90</td><td>37.40/22.05</td><td>+3.58</td><td>-25.66</td></tr></table>

Effect of TSPD. We then replace TSPD with a uniform penalty applied to all tokens identified by QRBA. Overall, the uniform-penalty variants yield a less favorable accuracy–efficiency trade-off than the complete RATIO framework, and their performance varies between AWQ and GPTQ. For example, under AWQ-W3, the strongest uniform-penalty variant reduces average accuracy by 1.16 percentage points and increases average CoT length by 22.04% compared with the complete RATIO framework. Also, a uniform penalty of λ = 1 yields lower accuracy and longer CoT than RATIO under AWQ-W3, but higher accuracy and slightly shorter CoT under GPTQ-W3. These results show that token-specific penalties provide more reliable calibration than applying a shared penalty strength to all selected tokens.

Overall, these results highlight the complementary roles of model-specific token selection and tokenspecific penalty determination. As shown in Fig. 5(b), RATIO produces lower occurrence rates of the selected tokens than either component ablation and the fixed-penalty baseline across all five benchmarks, further illustrating their combined effect on reasoning behavior.

## 5 DISCUSSION AND FUTURE WORKS

While RATIO assigns token-specific penalties based on model-level quantization behaviors, these penalties remain static during inference. However, the same token may play different roles under different reasoning states, and a fixed penalty may not always be optimal for every occurrence. A promising future direction is to develop dynamic token calibration strategies that adapt the penalty strength according to the current reasoning context. For example, penalties can be relaxed during early occurrences of a token to preserve useful reasoning behaviors, while being strengthened when the token appears repeatedly within a short context window to suppress potential repetitive reasoning patterns. This would allow calibration to distinguish useful self-correction from repeated reconsideration that contributes little to solving the problem. Such state-aware calibration may further improve the balance between reasoning efficiency and model capability.

## 6 CONCLUSION

In this paper, we investigate the challenges of applying post-training quantization to reasoning models, and focus on the limitations of fixed token-level interventions. To address these issues, we propose RATIO, a framework that identifies model-specific target tokens and assigns each a tailored penalty. RATIO introduces Quantization-aware Reasoning Behavior Analysis (QRBA) to identify quantization-sensitive reasoning tokens through token-level distribution analysis and reasoningcontext validation, and further develops Token-Specific Penalty Determination (TSPD) to derive token-specific penalties with full-precision guidance. Extensive experiments across multiple reasoning benchmarks and quantization settings demonstrate that RATIO effectively reduces reasoning length while preserving or improving accuracy, achieving a favorable accuracy-efficiency trade-off compared with existing token-level intervention strategies. This work establishes a behavior-aware calibration framework for quantized reasoning models, advancing efficient low-bit reasoning and providing a foundation for future research.

## REFERENCES

Ekaterina Alimaskina, Darya Rudas, Denis Shveykin, Gleb Molodtsov, Pavel Vasiliev, and Aleksandr Beznosikov. Extreme low-bit inference in reasoning models: Failure modes and targeted recovery. arXiv preprint arXiv:2606.02011, 2026.

Yamato Arai and Yuma Ichikawa. Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization. In NeurIPS, 2025.

Saleh Ashkboos, Ilia Markov, Elias Frantar, Tingxuan Zhong, Xincheng Wang, Jie Ren, Torsten Hoefler, and Dan Alistarh. Quik: Towards end-to-end 4-bit inference on generative large language models. In EMNLP, 2024a.

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. In NeurIPS, 2024b.

Runsheng Bai, Bo Liu, and Qiang Liu. Skim: Any-bit quantization pushing the limits of posttraining quantization. In ICML, 2025.

Chengzhu Bao, Xianglong Yan, Zhiteng Li, Guangshuo Qin, Guanghua Yu, and Yulun Zhang. Soar: Scale optimization for accurate reconstruction in nvfp4 quantization. arXiv preprint arXiv:2605.12245, 2026.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Tianle Chen, Pengyu Cheng, Qiyuan Zhu, Jiacheng Wang, Bei Liu, Hao Gu, Ruijie Shen, Xiaofeng Hou, Sirui Han, and Jiacheng Liu. Adaptive spatial and temporal redundancy optimization for efficient reasoning in large language models. In ACL, 2026.

Yuzong Chen, Xilai Dai, Jake Hyun, Chi-Chih Chang, Wonsuk Jang, Yuheng Wu, Thierry Tambe, Jae sun Seo, and Mohamed S. Abdelfattah. Razer: Pushing the limits of nvfp4 quantization with redundant zero remapping. arXiv preprint arXiv:2501.04052, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. DeepSeek-V4: Towards highly efficient milliontoken context intelligence. arXiv preprint arXiv:2606.19348, 2026.

Jasper Dekoninck, Nikola Jovanovic, Tim Gehrunger, K´ ari R´ ognvaldsson, Ivo Petrov, Chenhao Sun,¨ and Martin Vechev. Beyond benchmarks: MathArena as an evaluation platform for mathematics with LLMs. arXiv preprint arXiv:2605.00674, 2026.

Vage Egiazarian, Roberto L. Castro, Denis Kuznedelev, Andrei Panferov, Eldar Kurtic, Shubhra Pandit, Alexandre Marques, Mark Kurtz, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Bridging the gap between promise and performance for microscaling fp4 quantization. In ICLR, 2026.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. In ICLR, 2023.

GLM-5 Team. GLM-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. Thirtyfifth Conference on NeurIPS Datasets and Benchmarks Track (Round 2), 2021.

Wei Huang, Haotong Qin, Yangdong Liu, Yawei Li, Qinshuo Liu, Xianglong Liu, Luca Benini, Michele Magno, Shiming Zhang, and Xiaojuan Qi. SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models. In ICML, 2025.

Junhan Kim, Ho-young Kim, Eulrang Cho, Chungman Lee, Joonyoung Kim, and Yongkweon Jeon. BOA: Attention-aware Post-training Quantization without Backpropagation. In ICML, 2025.

Kimi Team. Kimi K3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

Yuhang Li, Ruokai Yin, Donghyun Lee, Shiting Xiao, and Priyadarshini Panda. GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration. In ICML, 2025a.

Zhen Li, Yupeng Su, Runming Yang, Congkai Xie, Zheng Wang, Zhongwei Xie, Ngai Wong, and Hongxia Yang. Quantization meets reasoning: Exploring llm low-bit quantization degradation for mathematical reasoning. arXiv preprint arXiv:2501.03035, 2025b.

Xinyu Lian, Walid Krichene, Beichen Huang, Masahiro Tanaka, Olatunji Ruwase, Li Zhang, and Minjia Zhang. Quantization inflates reasoning: Token inflation as a hidden cost of low-bit reasoning models. In EMNLP, 2026.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In ICLR, 2024.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. In MLSys, 2024.

Ruikang Liu, Yuxuan Sun, Manyi Zhang, Haoli Bai, Xianzhi Yu, Tiezheng Yu, Chun Yuan, and Lu Hou. Quantization hurts reasoning? an empirical study on quantized reasoning models. In COLM, 2025.

Sanae Lotfi, Polina Kirichenko, Steven Li, and Zechun Liu. Quantized reasoning models think they need to think longer, but they do not. In NeurIPS, 2026.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof Q&A benchmark. In COLM, 2024.

Utkarsh Saxena, Sayeh Sharify, Kaushik Roy, and Xin Wang. Resq: Mixed-precision quantization of large language models with low-rank residuals. In ICML, 2025.

Yuxuan Sun, Ruikang Liu, Haoli Bai, Han Bao, Kang Zhao, Yuening Li, Jiaxin Hu, Xianzhi Yu, Lu Hou, Chun Yuan, et al. Flatquant: Flatness matters for llm quantization. In ICML, 2025.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. In ICML, 2023.

Xianglong Yan, Chengzhu Bao, Zhiteng Li, Tianao Zhang, Shaoqiu Zhang, Ruobing Xie, Samm Sun, and Yulun Zhang. D<sup>2</sup>Quant: Accurate low-bit post-training weight quantization for llms. In NeurIPS, 2026a.

Xianglong Yan, Hong Liu, Chengzhu Bao, Tianao Zhang, Guanghua Yu, Jianchen Zhu, and Yulun Zhang. Focus: Fp4 optimization via coupled-relaxation and dual-granularity scaling. arXiv preprint arXiv:2608.01847, 2026b.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Nan Zhang, Eugene Kwek, Yusen Zhang, Muyu Pan, Suhang Wang, Prasenjit Mitra, and Rui Zhang. Quantlrm: Quantization of large reasoning models via fine-tuning signals. arXiv preprint arXiv:2602.02581, 2026.

Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, and Baris Kasikci. Atom: Low-bit quantization for efficient and accurate llm serving. In MLSys, 2024.

## A MORE EXPERIMENTAL RESULTS

## A.1 PERFORMANCE ON OTHER QUANTIZATION METHODS

To evaluate whether RATIO remains effective beyond weight-only quantization, we further apply it to DeepSeek-R1-Distill-Qwen-1.5B quantized with FlatQuant (Sun et al., 2025) under the W4A4KV4 setting. As shown in Table 3, RATIO improves average accuracy from 45.27% to 48.48% while reducing average CoT length by 12.90%. It also provides a more favorable accuracy– efficiency trade-off than fixed-penalty decoding, demonstrating its applicability to the joint low-bit quantization of weights, activations, and the KV cache.

Table 3: Performance of RATIO on Qwen-1.5B with FlatQuant W4A4KV4. Accuracy (%) and CoT length (k tokens) are reported as Acc/Len. The ∆ columns report changes relative to uncalibrated FlatQuant. For each metric, the highest accuracy and shortest CoT length are in bold.
<table><tr><td></td><td></td><td>GPQA</td><td>AIME</td><td>GSM8K Acc (%) / Len (k)</td><td>MATH</td><td>HumanEval</td><td>Avg. Acc / Len</td><td>∆Acc</td><td>∆Len% vs. baseline</td></tr><tr><td></td><td>FlatQuant</td><td>34.34/25.68</td><td>8.33/37.20</td><td>74.37/4.06</td><td>65.40/16.47</td><td>43.90/14.64</td><td>45.27/19.61</td><td></td><td></td></tr><tr><td>w-.5</td><td>+ Lotfi et al. (2026)</td><td>30.30/20.81</td><td>11.67/39.26</td><td>75.66/2.83</td><td>69.80/13.30</td><td>48.17/11.75</td><td>47.12/17.59</td><td>+1.85</td><td>-10.30</td></tr><tr><td></td><td>+RATIO</td><td>30.81/22.16</td><td>11.67/37.77</td><td>75.51/1.69</td><td>73.20/11.87</td><td>51.22/11.90</td><td>48.48/17.08</td><td>+3.21</td><td>-12.90</td></tr></table>

## B ADDITIONAL EXPERIMENTAL DETAILS

## B.1 SELECTED TOKENS AND PENALTIES

Tables 4 and 5 report the final model-specific token sets produced by QRBA and their normalized TSPD penalties. The same model-specific token set and penalty values are applied to its AWQ-W3 and GPTQ-W3 variants.

Table 4: Model-specific tokens and normalized penalties used by RATIO. A leading underscore denotes a whitespace prefix in the decoded token.
<table><tr><td colspan="2">Qwen-1.5B</td><td colspan="2">Qwen-7B</td><td colspan="2">Llama-8B</td><td colspan="2">Qwen-14B</td></tr><tr><td>Token</td><td>λk</td><td>Token</td><td>λk</td><td>Token</td><td>λk</td><td>Token</td><td> $\lambda _ { k }$ </td></tr><tr><td>_but</td><td>1.22</td><td>_but</td><td>1.06</td><td>_back</td><td>1.00</td><td>_but</td><td>1.00</td></tr><tr><td>_what</td><td>1.21</td><td>_back</td><td>0.58</td><td>-try</td><td>0.90</td><td>_what</td><td>1.04</td></tr><tr><td>_try</td><td>0.73</td><td>_check</td><td>0.48</td><td>_well</td><td>0.86</td><td>_think</td><td>0.71</td></tr><tr><td>_again</td><td>1.16</td><td>_But</td><td>1.15</td><td>_check</td><td>0.93</td><td>_But</td><td>1.23</td></tr><tr><td>_well</td><td>0.88</td><td>_something</td><td>1.09</td><td>_But</td><td>1.12</td><td>_something</td><td>1.57</td></tr><tr><td>_But</td><td>1.09</td><td>_might</td><td>0.75</td><td>_something</td><td>1.34</td><td>_question</td><td>0.80</td></tr><tr><td>_something</td><td>1.12</td><td>-question</td><td>1.32</td><td>_thought</td><td>0.85</td><td>_actually</td><td>1.05</td></tr><tr><td>_might</td><td>1.03</td><td>_wait</td><td>0.92</td><td>-question</td><td>1.10</td><td>_wait</td><td>0.92</td></tr><tr><td>No</td><td>0.74</td><td>But</td><td>1.00</td><td>_actually</td><td>0.95</td><td>But</td><td>1.14</td></tr><tr><td>_thought</td><td>0.78</td><td>_trying</td><td>0.64</td><td>_wait</td><td>0.96</td><td>_correct</td><td>0.92</td></tr><tr><td>But</td><td>1.09</td><td>_ensure</td><td>0.67</td><td>But</td><td>1.07</td><td>_trying</td><td>0.71</td></tr><tr><td>_correct</td><td>1.00</td><td>but</td><td>1.04</td><td>_correct</td><td>1.13</td><td>_seems</td><td>0.88</td></tr><tr><td>_ensure</td><td>1.00</td><td>_Wait</td><td>1.18</td><td>_seems</td><td>1.11</td><td>_maybe</td><td>0.77</td></tr><tr><td>Yes</td><td>0.90</td><td>Wait</td><td>1.12</td><td>_maybe</td><td>0.93</td><td>_confirm</td><td>0.69</td></tr><tr><td>However</td><td>1.04</td><td>_Thus</td><td>0.37</td><td>-perhaps</td><td>0.96</td><td>-perhaps</td><td>0.96</td></tr><tr><td>_Wait</td><td>0.96</td><td>Thus</td><td>0.36</td><td>_Wait</td><td>1.14</td><td>_Wait</td><td>1.33</td></tr><tr><td>Wait</td><td>1.12</td><td>Hmm</td><td>1.03</td><td>something</td><td>1.29</td><td>Wait</td><td>1.32</td></tr><tr><td>_conclude</td><td>1.02</td><td></td><td></td><td></td><td></td><td>something</td><td>1.78</td></tr><tr><td>Thus</td><td>0.48</td><td></td><td></td><td></td><td></td><td>Thus</td><td>0.36</td></tr><tr><td>Hmm</td><td>0.97</td><td></td><td></td><td></td><td></td><td>Hmm</td><td>1.23</td></tr><tr><td>Alternatively</td><td>0.97</td><td></td><td></td><td></td><td></td><td>_Hmm</td><td>1.25</td></tr></table>

Table 5: Selected tokens and normalized penalties for Qwen3-4B in thinking mode. A leading underscore denotes a whitespace prefix in the decoded token.
<table><tr><td>Token</td><td> $\lambda _ { k }$ </td><td>Token</td><td> $\lambda _ { k }$ </td></tr><tr><td>_but</td><td>1.11</td><td>-question</td><td>1.11</td></tr><tr><td>_what</td><td>1.03</td><td>But</td><td>1.06</td></tr><tr><td>_back</td><td>0.83</td><td>_However</td><td>0.94</td></tr><tr><td>_think</td><td>0.81</td><td>but</td><td>1.00</td></tr><tr><td>_But</td><td>1.05</td><td>-perhaps</td><td>0.84</td></tr><tr><td>_something</td><td>1.00</td><td>Wait</td><td>1.02</td></tr><tr><td>_looking</td><td>0.82</td><td>_mistake</td><td>0.79</td></tr></table>

## B.2 QUANTIZATION AND IMPLEMENTATION DETAILS

Quantization Settings. We provide the detailed quantization and inference configurations used in our experiments. For AWQ, we follow the standard AWQ recipe and use 128 sequences with a sequence length of 512 sampled from the Pile validation set as the calibration data. For GPTQ, we use WikiText-2 as the calibration dataset, with 128 sequences of length 2048. All quantized models use a group size of 128 for weight quantization.

Implementation Details. Unlike the 65,536-token generation budget specified in the main text, Qwen3-4B is subject to a total context limit of 40,960 tokens.

## C ADDITIONAL METHOD DETAILS

## C.1 CANDIDATE-SET CONSTRUCTION AND QSTI CRITERIA

Candidate-Set Construction. At each reasoning position i, we first construct a raw candidate pool from the union of the top-p candidate sets produced by the quantized model Q and the full-precision model F:

$$
\mathcal { U } _ { Q , i } = \mathcal { N } _ { Q , i } \cup \mathcal { N } _ { F , i } ,\tag{10}
$$

where $\mathcal { N } _ { Q , i }$ and $\mathcal { N } _ { F , i }$ denote the corresponding top-p candidate sets under the shared prefix $x _ { < t _ { i } }$ . We set $p = 0 . 9 5$ , consistent with the decoding configuration used in our other experiments. For tokens appearing in both candidate sets, we directly use the probabilities produced by the two models. For a token appearing in only one candidate set, we use a probability floor $\epsilon = 1 0 ^ { - 3 }$ for the missing side.

A probability floor may introduce an unreliable shift when it is larger than the observed probability on the available side. We therefore apply a filtering criterion to single-sided candidate events. Specifically, the retained candidate set is defined as

$$
\mathcal { C } _ { Q , i } = ( \mathcal { N } _ { Q , i } \cap \mathcal { N } _ { F , i } ) \cup \mathcal { S } _ { Q , i } \cup \mathcal { S } _ { F , i } ,\tag{11}
$$

where

$$
S _ { Q , i } = \left\{ k \in \mathcal { N } _ { Q , i } \setminus \mathcal { N } _ { F , i } : p _ { Q } ( k | x _ { < t _ { i } } ) > \epsilon \right\} ,\tag{12}
$$

and

$$
S _ { F , i } = \left\{ k \in \mathcal { N } _ { F , i } \setminus \mathcal { N } _ { Q , i } : p _ { F } ( k | x _ { < t _ { i } } ) > \epsilon \right\} .\tag{13}
$$

For each retained token $k \in \mathcal { C } _ { Q , i } ,$ , we compute the token-level probability shift as

$$
\delta _ { Q , i } ( k ) = \log p _ { Q } ( k | x _ { < t _ { i } } ) - \log p _ { F } ( k | x _ { < t _ { i } } ) .\tag{14}
$$

For a retained single-sided event, the probability on the missing side is set to ϵ only when computing this shift. For example, consider a token that appears only in the quantized candidate set and satisfies

$$
p _ { Q } ( k | x _ { < t _ { i } } ) = 0 . 0 0 3 > \epsilon .\tag{15}
$$

Using $p _ { F } ( k | x _ { < t _ { i } } ) = \epsilon = 0 . 0 0 1$ for shift estimation gives

$$
\delta _ { Q , i } ( k ) = \log 0 . 0 0 3 - \log 0 . 0 0 1 > 0 .\tag{16}
$$

In contrast, if the observed probability is $1 0 ^ { - 5 } \le \epsilon$ , the event is excluded before shift estimation. In this case, assigning ϵ to the missing side would make the direction of the resulting shift primarily determined by the artificial floor rather than reliable model evidence. Only events retained in $\mathcal { C } _ { Q , i }$ are included in the occurrence-level aggregation and subsequent QSTI statistics.

Support and Stability Criteria. After obtaining valid occurrence-level shifts, we further apply support and stability criteria to identify reliable quantization-sensitive tokens. For each token $k ,$ we denote $N _ { Q } ( k )$ as the number of valid shift events under quantizer $Q , S _ { Q } ( k )$ as the number of reference trajectories containing these events, and $\bar { \delta } _ { Q } ( k )$ as the average occurrence-level shift.

We first retain tokens that are observed under both AWQ and GPTQ and satisfy the token-cleaning rule. The cleaning rule removes special tokens, punctuation, non-English pieces, and uninterpretable tokenizer fragments. We then require sufficient event support from both quantizers:

$$
N _ { \mathrm { A W Q } } ( k ) > 1 0 0 , \qquad N _ { \mathrm { G P T Q } } ( k ) > 1 0 0 .\tag{17}
$$

To ensure that the identified preference changes are not specific to a single quantization method, we further require the average shifts of AWQ and GPTQ to have the same direction:

$$
\bar { \delta } _ { \mathrm { A W Q } } ( k ) \cdot \bar { \delta } _ { \mathrm { G P T Q } } ( k ) > 0 .\tag{18}
$$

Finally, we require sufficient trajectory coverage:

$$
S _ { \mathrm { A W Q } } ( k ) > 1 0 5 , \qquad S _ { \mathrm { G P T Q } } ( k ) > 1 0 5 .\tag{19}
$$

The tokens satisfying these requirements form the shared candidate pool $K _ { \mathrm { b a s e } }$

We then perform stability filtering on the positive-shift candidates in $K _ { \mathrm { b a s e } }$ . Since RATIO aims to suppress tokens whose preference is increased by quantization, negative-shift tokens are not included in the final calibration set. For each positive token, we compute three complementary statistics. First, the conservative shift magnitude is defined as:

$$
R _ { \Delta } ( k ) = \operatorname* { m i n } \left( | \bar { \delta } _ { \mathrm { A W Q } } ( k ) | , | \bar { \delta } _ { \mathrm { G P T Q } } ( k ) | \right) .\tag{20}
$$

Second, we measure directional consistency across occurrence-level events:

$$
R _ { \mathrm { d i r } } ^ { + } ( k ) = \operatorname* { m i n } \left( \mathrm { P r } _ { \mathrm { A W Q } } ( \delta _ { i } > 0 ) , \mathrm { P r } _ { \mathrm { G P T Q } } ( \delta _ { i } > 0 ) \right) .\tag{21}
$$

Here, $\mathrm { P r } _ { \mathrm { A W Q } } ( \delta _ { i } > 0 )$ represents the proportion of shift events where AWQ assigns a higher probability to token k than the full-precision model. Third, we measure the proportion of events where the token appears in both the quantized and full-precision top-p candidate sets:

$$
R _ { \mathrm { b o t h } } ( k ) = \operatorname* { m i n } \left( \operatorname* { P r } _ { \mathrm { A W Q } } ( k \in \mathcal { N } _ { Q , i } \cap \mathcal { N } _ { F , i } ) , \operatorname* { P r } _ { \mathrm { G P T Q } } ( k \in \mathcal { N } _ { Q , i } \cap \mathcal { N } _ { F , i } ) \right) .\tag{22}
$$

These three statistics characterize the strength, consistency, and reliability of the observed quantization-related shifts. The corresponding thresholds are determined adaptively from the lower quartiles of the positive candidate distribution:

$$
\tau _ { \Delta } ^ { + } = \mathrm { Q u a n t i l e } _ { 0 . 2 5 } ( R _ { \Delta } ) ,\tag{23}
$$

$$
\tau _ { \mathrm { d i r } } ^ { + } = \mathrm { Q u a n t i l e } _ { 0 . 2 5 } ( R _ { \mathrm { d i r } } ^ { + } ) ,\tag{24}
$$

$$
\tau _ { \mathrm { b o t h } } ^ { + } = \mathrm { Q u a n t i l e } _ { 0 . 2 5 } ( R _ { \mathrm { b o t h } } ) .\tag{25}
$$

A token is retained as a numerically stable candidate when:

$$
R _ { \Delta } ( k ) \geq \tau _ { \Delta } ^ { + } ,\tag{26}
$$

$$
R _ { \mathrm { d i r } } ^ { + } ( k ) \geq \tau _ { \mathrm { d i r } } ^ { + } ,\tag{27}
$$

$$
R _ { \mathrm { b o t h } } ( k ) \geq \tau _ { \mathrm { b o t h } } ^ { + } .\tag{28}
$$

The resulting set is denoted as $K _ { \mathrm { n u m } }$ 1·

To improve recall, we additionally consider a small reasoning-state lexical recovery branch within $K _ { \mathrm { b a s e } }$ This branch targets tokens with strong associations with reasoning-related behaviors but weaker numerical stability. Such tokens are not directly included in the calibration set; instead, they are passed to subsequent reasoning-context-aware validation for further verification. The resulting set is denoted as $K _ { \mathrm { l e x } } ^ { \bar { } }$

The final QSTI candidate pool is defined as:

$$
K _ { \mathrm { Q S T I } } = K _ { \mathrm { n u m } } \cup K _ { \mathrm { l e x } } .\tag{29}
$$

All tokens in $K _ { \mathrm { Q S T I } }$ are further validated by RTV before being used for token-specific calibration.

## C.2 REASONING-CONTEXT-AWARE TOKEN VALIDATION

## C.2.1 NUMERICAL EVIDENCE CONSTRUCTION

Given an input $x _ { r } .$ , each quantized model $Q \in \{ \mathrm { A W Q } , \mathrm { G P T Q } \}$ freely generates a reasoning trajectory $y _ { r } .$ . The full-precision model $F$ then evaluates the same quantized prefixes through teacher forcing. This allows all token-level comparisons to be performed under identical reasoning states. For each candidate $k \in K _ { \mathrm { Q S T I } }$ , we construct the following quantitative evidence.

Actual-token Scoring. Suppose that the quantized model generates token $y _ { r , t } = k$ at position t in trajectory r. We measure its probability shift under the same quantized prefix as

$$
\delta _ { Q , r , t } ^ { \mathrm { s c o r e } } ( k ) = \log p _ { Q } ( k | y _ { r , < t } , x _ { r } ) - \log p _ { F } ( k | y _ { r , < t } , x _ { r } ) .\tag{30}
$$

A positive value indicates that the quantized model assigns a higher probability to the generated token than the full-precision model under the same reasoning state. Let

$$
\mathcal { O } _ { Q } ( k ) = \{ ( r , t ) : y _ { r , t } = k \}\tag{31}
$$

denote the set of actual-token events for k. We compute its event-weighted mean anomaly as

$$
A _ { Q } ( k ) = \frac { 1 } { | \mathcal { O } _ { Q } ( k ) | } \sum _ { ( r , t ) \in \mathcal { O } _ { Q } ( k ) } \delta _ { Q , r , t } ^ { \mathrm { s c o r e } } ( k ) .\tag{32}
$$

Every occurrence receives equal weight in $A _ { Q } ( k )$ . Consequently, trajectories in which the token occurs frequently contribute more strongly to this statistic.

Trajectory-level Aggregation. To measure whether the observed shift is consistent across different reasoning trajectories, let

$$
I _ { Q } ( k ) = \{ r : \exists t , ( r , t ) \in \mathcal { O } _ { Q } ( k ) \}\tag{33}
$$

denote the set of trajectories containing at least one actual-token event for k. Its trajectory support is defined as

$$
S _ { \cal Q } ( k ) = | I _ { \cal Q } ( k ) | .\tag{34}
$$

Each trajectory contributes at most one unit to $S _ { Q } ( k )$ , regardless of how frequently the token occurs within that trajectory. Let $n _ { Q , r } ( k )$ denote the number of occurrences of k in trajectory r. We first compute its within-trajectory mean anomaly:

$$
\bar { a } _ { Q , r } ( k ) = \frac { 1 } { n _ { Q , r } ( k ) } \sum _ { t : y _ { r , t } = k } \delta _ { Q , r , t } ^ { \mathrm { s c o r e } } ( k ) ,\tag{35}
$$

and then average equally across trajectories:

$$
T _ { Q } ( k ) = \frac { 1 } { S _ { Q } ( k ) } \sum _ { r \in I _ { Q } ( k ) } \bar { a } _ { Q , r } ( k ) .\tag{36}
$$

Unlike the event-weighted statistic $A _ { Q } ( k ) , T _ { Q } ( k )$ assigns equal weight to each trajectory and therefore reduces the influence of individual long or repetitive responses. We define the corresponding cross-quantizer evidence as

$$
E _ { \mathrm { t r a j e c t o r y } } ( k ) = [ T _ { \mathrm { A W Q } } ( k ) > 0 ] \wedge [ T _ { \mathrm { G P T Q } } ( k ) > 0 ] .\tag{37}
$$

This condition indicates that the positive anomaly persists across reasoning trajectories under both quantizers.

Next-token Preview. Actual-token scoring considers only tokens selected during generation. To capture tokens whose probabilities are increased by quantization even when they are not sampled, we additionally compare the top-p candidate distributions under the same quantized prefix. For a token k appearing in the quantized top-p set, its preview shift is

$$
\delta _ { Q , r , t } ^ { \mathrm { p r e v i e w } } ( k ) = \log p _ { Q } ( k | y _ { r , < t } , x _ { r } ) - \log p _ { F } ( k | y _ { r , < t } , x _ { r } ) .\tag{38}
$$

If k is absent from the full-precision top-p set, we use the same probability floor as in QSTI:

$$
p _ { \mathrm { f l o o r } } = 0 . 0 0 1 .\tag{39}
$$

This value is used only to approximate the missing-side probability and is unrelated to the inferencetime penalty. We aggregate preview events similarly to actual-token events, obtaining trajectory support $S _ { Q } ^ { \mathrm { p r e v i e w } } ( k )$ and mean anomaly $A _ { Q } ^ { \mathrm { p r e v i e w } } ( k )$ . Preview evidence is activated when

$$
E _ { \mathrm { p r e v i e w } } ( k ) = \bigwedge _ { Q \in \{ \mathrm { A W Q , G P T Q } \} } \left( [ S _ { Q } ^ { \mathrm { p r e v i e w } } ( k ) \geq 2 0 ] \wedge [ A _ { Q } ^ { \mathrm { p r e v i e w } } ( k ) > 0 ] \right) .\tag{40}
$$

Correctness-conditioned Evidence. We use independent full-precision rollouts to divide the quantized trajectories into two primary groups:

$$
\mathcal { E } _ { Q } = \{ r : Q \mathrm { ~ i s ~ i n c o r r e c t } { \operatorname { a n d } F } \mathrm { { i s } ~ c o r r e c t } \} ,\tag{41}
$$

$$
{ \mathcal { C } } _ { Q } = \{ r : Q { \mathrm { ~ i s ~ c o r r e c t ~ a n d ~ } } F { \mathrm { ~ i s ~ c o r r e c t } } \} .\tag{42}
$$

The first group represents reasoning failures introduced by quantization, while the second serves as a control group. For evidence type $u \in \{ \mathrm { s c o r e , p r e v i e w } \}$ , let $\bar { a } _ { Q , r } ^ { u } ( k )$ denote the corresponding trajectory-level mean anomaly. We define the correctness-conditioned difference as

$$
\begin{array} { r } { \Delta _ { Q } ^ { u } ( k ) = \mathrm { m e a n } _ { r \in \mathcal { E } _ { Q } } \bar { a } _ { Q , r } ^ { u } ( k ) - \mathrm { m e a n } _ { r \in \mathcal { C } _ { Q } } \bar { a } _ { Q , r } ^ { u } ( k ) . } \end{array}\tag{43}
$$

The means are taken over trajectories in which the corresponding event for k is observed. A positive value indicates that the token exhibits a stronger quantization-related anomaly when the quantized model fails on a problem solved correctly by the full-precision model. We define

$$
E _ { \mathrm { s c o r e - c o r r e c t } } ( k ) = \bigwedge _ { Q \in \{ \mathrm { A W Q , G P T Q } \} } [ \Delta _ { Q } ^ { \mathrm { s c o r e } } ( k ) > 0 ] ,\tag{44}
$$

and

$$
E _ { \mathrm { p r e v i e w - c o r r e c t } } ( k ) = \bigwedge _ { Q \in \{ \mathrm { A W Q } , \mathrm { G P T Q } \} } [ \Delta _ { Q } ^ { \mathrm { p r e v i e w } } ( k ) > 0 ] .\tag{45}
$$

The two indicators are combined as

$$
E _ { \mathrm { c o r r e c t } } ( k ) = E _ { \mathrm { s c o r e - c o r r e c t } } ( k ) \vee E _ { \mathrm { p r e v i e w - c o r r e c t } } ( k ) .\tag{46}
$$

Thus, $E _ { \mathrm { c o r r e c t } } ( k )$ indicates that either actual-token or preview anomalies are consistently stronger in quantized-model failures than in shared successes under both quantizers.

Loop and Repetition Evidence. We construct two complementary forms of repetition evidence: token enrichment in explicit-loop trajectories and localized high-repeat patterns.

Explicit-loop detection. We first label a trajectory as an explicit loop if it satisfies

$$
\left( { \mathrm { r e p e a t 4 } } \geq 8 0 \% \land { \mathrm { d u p l i c a t e - l i n e s } } \geq 5 0 \% \right) \lor \left( { \mathrm { m a x ~ s a m e ~ t o k e n ~ r u n } } \geq 6 4 \right) .\tag{47}
$$

Here, we enumerate all contiguous 4-token windows in the trajectory. A window is treated as repeated if its exact 4-token sequence occurs at least twice within that trajectory, and all occurrences of such a sequence, including its first occurrence, are counted as repeated windows. The value of repeat4 is the number of repeated-window occurrences divided by the total number of 4-token windows. Thus, repeat4 $\ge ~ \hat { 8 0 \% }$ means that at least 80% of the trajectory’s 4-token windows have an identical counterpart elsewhere in the same trajectory. To compute duplicate-lines, we split the decoded trajectory at newline characters and discard empty lines and lines shorter than a predefined length threshold. We then compare the remaining lines using exact string matching. A retained line is counted as duplicated if an identical line appears elsewhere in the same trajectory, and duplicate-lines is the proportion of such duplicated line occurrences among all retained lines. Finally, max same token run denotes the longest consecutive repetition of a single token. An explicit loop that produces an incorrect answer is assigned to the error-loop set $L _ { Q } ^ { \mathrm { e r r o r } }$ , whereas a trajectory that does not satisfy the loop criterion is assigned to the non-loop set $L _ { Q } ^ { \mathrm { n o n l o o p } }$

Token enrichment in explicit loops. After labeling the trajectories, we determine whether candidate token k occurs more frequently in error-loop trajectories than in non-loop trajectories. For evidence type $u \in \{ \mathrm { s c o r e } , \mathrm { p r e v i e w } \}$ , its length-normalized occurrence rate in trajectory r is

$$
f _ { Q , r } ^ { u } ( k ) = 1 0 0 0 \frac { \mathrm { c o u n t } _ { Q , r } ^ { u } ( k ) } { \mathrm { l e n g t h } ( y _ { r } ) } .\tag{48}
$$

We then compute the loop-enrichment ratio

$$
R _ { Q } ^ { \mathrm { { l o o p } } , u } ( k ) = \frac { \mathrm { m e a n } _ { r \in L _ { Q } ^ { \mathrm { e r r o r } } } f _ { Q , r } ^ { u } ( k ) } { \mathrm { m e a n } _ { r \in L _ { Q } ^ { \mathrm { n o n l o o p } } } f _ { Q , r } ^ { u } ( k ) } .\tag{49}
$$

The corresponding cross-quantizer indicators are

$$
E _ { \mathrm { s c o r e - l o o p } } ( k ) = \bigwedge _ { Q \in \{ \mathrm { A W Q } , \mathrm { G P T Q } \} } [ R _ { Q } ^ { \mathrm { l o o p } , \mathrm { s c o r e } } ( k ) > 1 ] ,\tag{50}
$$

$$
E _ { \mathrm { p r e v i e w - l o o p } } ( k ) = \bigwedge _ { Q \in \{ \mathrm { A W Q } , \mathrm { G P T Q } \} } [ R _ { Q } ^ { \mathrm { l o o p } , \mathrm { p r e v i e w } } ( k ) > 1 ] .\tag{51}
$$

If the mean occurrence rate in non-loop trajectories is zero while that in error-loop trajectories is positive, the ratio is treated as infinity. If both rates are zero, the ratio is treated as missing and does not activate the corresponding indicator.

Localized high-repeat evidence. The trajectory-level loop criterion may overlook localized repetition. We therefore inspect all 4-token windows covering each occurrence of k. An occurrence is marked as a high-repeat event if any such exact 4-token sequence occurs at least 8 times within the same trajectory. Let $H _ { Q } ( k )$ denote the number of trajectories containing at least one high-repeat event for k. We define

$$
E _ { \mathrm { r e p e a t } } ( k ) = [ H _ { \mathrm { A W Q } } ( k ) > 0 ] \wedge [ H _ { \mathrm { G P T Q } } ( k ) > 0 ] .\tag{52}
$$

Because this condition is intentionally permissive, it is used only as supporting evidence.

Evidence Consolidation. Finally, we consolidate the raw indicators into four distinct evidence categories. We retain $E _ { \mathrm { t r a j e c t o r y } } ( \boldsymbol { k } )$ and $E _ { \mathrm { p r e v i e w } } ( k )$ as separate categories, and define

$$
E _ { \mathrm { l o o p / r e p e a t } } ( k ) = E _ { \mathrm { s c o r e - l o o p } } ( k ) \vee E _ { \mathrm { p r e v i e w - l o o p } } ( k ) \vee E _ { \mathrm { r e p e a t } } ( k ) .\tag{53}
$$

Together with $E _ { \mathrm { c o r r e c t } } ( k )$ defined above, the number of distinct evidence categories is

$$
N _ { C } ( k ) = E _ { \mathrm { t r a j e c t o r y } } ( k ) + E _ { \mathrm { p r e v i e w } } ( k ) + E _ { \mathrm { c o r r e c t } } ( k ) + E _ { \mathrm { l o o p / r e p e a t } } ( k ) .\tag{54}
$$

Each category contributes at most one count, preventing closely related indicators from being counted repeatedly.

## C.2.2 STAGED TOKEN SELECTION

Based on the evidence constructed above, RTV progressively selects tokens through a strict core and two expansion stages. All stages operate exclusively on $K _ { \mathrm { Q S T I } }$ , ensuring that contextual validation cannot introduce tokens that have not passed QSTI.

Strict core. We first construct a reference pool containing candidates with sufficient trajectory support and positive actual-token anomalies under both quantizers:

$$
K _ { \mathrm { r e f } } = \left\{ k \in K _ { \mathrm { Q S T I } } : \bigwedge _ { Q \in \{ \mathrm { A W Q } , \mathrm { G P T Q } \} } [ S _ { Q } ( k ) \geq 2 0 \land A _ { Q } ( k ) > 0 ] \right\} .\tag{55}
$$

For each quantizer, we compute the lower quartile of its anomaly values within the reference pool:

$$
\tau _ { Q } = \mathrm { Q u a n t i l e } _ { 0 . 2 5 } \left( \{ A _ { Q } ( j ) : j \in K _ { \mathrm { r e f } } \} \right) , \qquad Q \in \{ \mathrm { A W Q } , \mathrm { G P T Q } \} .\tag{56}
$$

A candidate passes the strict numerical gate if

$$
G _ { \mathrm { s t r i c t } } ( k ) = [ k \in K _ { \mathrm { Q S T I } } ] \wedge \bigwedge _ { Q \in \{ \mathrm { A W Q } , \mathrm { G P T Q } \} } [ S _ { Q } ( k ) \geq 2 0 \wedge A _ { Q } ( k ) \geq \tau _ { Q } ] .\tag{57}
$$

The thresholds $\tau _ { Q }$ are derived separately for each model and quantizer rather than shared across models. This gate therefore requires substantial positive anomalies under both AWQ and GPTQ, rather than deviations that are only marginally above zero. Candidates passing the gate are further examined using reasoning-state lexical information. A token is retained in the strict core only when its observed use is directly associated with reasoning operations such as restarting, backtracking, verification, hesitation, self-questioning, self-correction, uncertainty, or metacognitive control.

First-round expansion (R1). To recover candidates with moderate support or asymmetric anomaly strength across quantizers, R1 applies a relaxed numerical gate $G _ { \mathrm { R 1 } } ( \bar { k } )$ defined by:

$$
[ k \in K _ { \mathrm { Q S T } } ] \land \bigwedge _ { Q \in \{ \mathrm { A W Q , G P T Q } \} } [ S _ { Q } ( k ) \geq 1 0 ] \land \bigvee _ { Q \in \{ \mathrm { A W Q , G P T Q } \} } [ A _ { Q } ( k ) > 0 ] \land [ N _ { C } ( k ) \geq 1 ] \land \neg G _ { \mathrm { s t r i c t } } ( k ) .\tag{58}
$$

All R1 candidates are first examined using the same reasoning-state lexical information as the strict core. We additionally require contextual review when the candidate is supported by only one evidence category and that category is either trajectory-level or preview evidence. The contextualreview trigger is

$$
[ N _ { C } ( k ) = 1 ] \wedge [ E _ { \mathrm { t r a j e c t o r y } } ( k ) \vee E _ { \mathrm { p r e v i e w } } ( k ) ] \wedge \neg E _ { \mathrm { c o r r e c t } } ( k ) \wedge \neg E _ { \mathrm { l o o p / r e p e a t } } ( k ) .\tag{59}
$$

This condition is deterministic. It is activated because trajectory-level and preview evidence primarily reflect distributional deviations; without correctness-conditioned or repetition-related support, the surrounding reasoning context must be inspected to distinguish inefficient reasoning from valid verification or self-correction.

Second-round expansion (R2). R2 considers near-miss candidates with weaker numerical support but evidence from multiple complementary categories:

$$
G _ { \mathrm { R 2 } } ( k ) = [ k \in K _ { \mathrm { Q S T I } } ] \wedge \bigwedge _ { \substack { Q \in \{ \mathrm { A W Q } , \mathrm { G P T Q } \} } } [ S _ { Q } ( k ) \geq 5 ] \wedge [ N _ { C } ( k ) \geq 2 ] \wedge \neg G _ { \mathrm { s t r i c t } } ( k ) \wedge \neg G _ { \mathrm { R 1 } } ( k ) .\tag{60}
$$

Because R2 uses the most relaxed support threshold, every candidate passing this gate undergoes contextual review. We inspect its occurrences in the original quantized trajectories to determine whether the token is consistently associated with inefficient deliberation, rather than necessary verification, correction, or other valid reasoning behavior.

Let $K _ { \mathrm { s t r i c t } } , K _ { \mathrm { R 1 } }$ , and $K _ { \mathrm { R 2 } }$ denote the tokens retained by the three stages. The final model-specific token set is

$$
K = K _ { \mathrm { s t r i c t } } \cup K _ { \mathrm { R 1 } } \cup K _ { \mathrm { R 2 } } .\tag{61}
$$

Contextual review is applied only after the corresponding numerical gate has been satisfied. It determines whether a numerically qualified candidate is retained or excluded, but never introduces tokens outside the gated candidate pools.

## C.3 DERIVATION OF TOKEN-SPECIFIC LOGIT CORRECTION

For a target token $k ,$ its probability under the softmax function can be written as:

$$
p ( k ) = \frac { e ^ { z _ { k } } } { e ^ { z _ { k } } + \sum _ { j \neq k } e ^ { z _ { j } } } ,\tag{62}
$$

where $z _ { k }$ denotes the logit of token k. Let

$$
S = \sum _ { j \neq k } e ^ { z _ { j } } ,\tag{63}
$$

then the probability of token k can be rewritten as:

$$
p ( k ) = \frac { e ^ { z _ { k } } } { e ^ { z _ { k } } + S } .\tag{64}
$$

The corresponding logit transformation is defined as:

$$
\mathrm { l o g i t } ( p ( k ) ) = \log \frac { p ( k ) } { 1 - p ( k ) } .\tag{65}
$$

Substituting the probability expression into the logit transformation gives:

$$
\mathrm { l o g i t } ( p ( k ) ) = \log \frac { \frac { e ^ { z _ { k } } } { e ^ { z _ { k } } + S } } { \frac { S } { e ^ { z _ { k } } + S } } = \log \frac { e ^ { z _ { k } } } { S } = z _ { k } - \log S .\tag{66}
$$

Under the independent token calibration assumption, only the target token logit is adjusted while all other quantized logits remain unchanged. Let $z _ { Q , k } ^ { \prime }$ denote the corrected logit whose probability matches the full-precision preference. The required correction magnitude is therefore:

$$
c _ { Q , i } ( k ) = z _ { Q , k } - z _ { Q , k } ^ { \prime } = \log \mathrm { i t } ( p _ { Q } ( k | x _ { < t _ { i } } ) ) - \log \mathrm { i t } ( p _ { F } ( k | x _ { < t _ { i } } ) ) .\tag{67}
$$

## D DIALOGUE EXAMPLES

Table 6 shows how RATIO curbs repetition and erroneous continuations while preserving correct answers.

Table 6: Dialogue examples on Qwen-1.5B with GPTQ-W3. Red marks repetitive or erroneous reasoning; green marks correct RATIO calculations and answers. Ellipses indicate omissions.

<table><tr><td colspan="2">Case 1: Repetitive reasoning after reaching the correct result</td></tr><tr><td>Prompt</td><td>From her science class study, Brady learned that each whale has 40 gallons of blood and that a shark has three times as much blood as a whale. How many gallons of blood do ten sharks have?</td></tr><tr><td>GPTQ-W3</td><td>Each shark has  $3 \times 4 0 = 1 2 0$  gallons, so ten sharks have  $1 0 \times 1 2 0 = 1 2 0 0$  gallons. Wait, let me double-check that. ... Wait, that seems correct. Let me think if I interpreted the question correctly. ... Alternatively, perhaps I can think of the problem as ten sharks each having three times as much as a whale, so  $1 0 \times ( 3 \times W ) = \hat { 3 } \times ( 1 0 \times W )$  Therefore, ten sharks have  $3 \times 1 0 \times 4 0 = 1 2 0 0$  gallons. Alternatively, perhaps I can think of it as</td></tr><tr><td>RATIO</td><td> $1 0 \times ( 3 \times W ) = 3 \times ( 1 0 \times W ) . . . .$  Wait, 3 W. ... 3 W3 W3 W. A shark has three times as much blood as a whale. Therefore, each shark has  $3 \times 4 0 = 1 2 0$  gallons of blood. Ten such sharks have  $1 0 \times 1 2 0 = 1 2 0 0$  gallons of blood. The final answer is 1200</td></tr><tr><td>Compute  $9 9 ^ { 2 } + 9 9 + 1 \nonumber$  in your head.</td><td>Case 2: Repetition followed by an incorrect reasoning trajectory</td></tr><tr><td>Prompt GPTQ-W3</td><td>First,  $9 9 ^ { 2 } \ : = \ : ( 1 0 0 \ : - \ : 1 ) ^ { 2 } \ : = \ : 1 0 , 0 0 0 \ : - \ : 2 0 0 + 1 \ : = \ : 9 , 8 0 1$  Adding 99 gives 9,900, and adding 1 gives 9,901. . .. Let me verify each step again.  $9 9 ^ { 2 }$  is 9,801, then  $9 , 8 0 1 + 9 9 =$  9,900, and adding 1 gives 9,901. . . . Alternatively, maybe I can factor it as 99 × (99 + 1) + 1. That gives 99  $\times ~ 1 0 0 + 1 = 9 { , } 9 0 1$  . Alternatively, if I think of  $9 9 ^ { 2 } + 9 9 { \dot { + } } 1$  as  $( \stackrel { \cdot } { 9 } 9 + 1 ) ^ { 2 } - \stackrel { \cdot } { 1 } ,$  that is  $1 0 0 ^ { 2 } - 1 = 9 { , } 9 9 9 .$  Wait, no, that is incorrect. ... So, 9999.</td></tr><tr><td>RATIO</td><td>1099999999999999999999999999999999999999. First,  $9 9 ^ { 2 } = ( 1 0 0 - 1 ) ^ { 2 } = 1 0 , 0 0 0 - 2 0 0 + 1 = 9 , 8 0 1 . { \mathrm { ~ N e x t , ~ } } 9 , 8 0 1 + 9 9 = 9 , 9 0 0 { \mathrm { ~ } }$  and  $9 , 9 0 0 + 1 = 9 , 9 0 1 . \mathrm { T h u s } ,$  the final answer is 9901</td></tr></table>