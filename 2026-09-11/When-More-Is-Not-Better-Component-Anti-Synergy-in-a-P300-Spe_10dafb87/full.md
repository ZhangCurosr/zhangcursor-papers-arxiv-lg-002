# When More Is Not Better: Component Anti-Synergy in a P300 Speller

Lucas Yang

Rui Liu

Parkland High School

Dept. of Computer Science

Fusheng Wang

Allentown, PA, USA

yanglucas2028@gmail.com

Stony Brook University

Stony Brook, NY, USA

Dept. of Computer Science

ruiliu1@cs.stonybrook.edu

Dept. of Biomedical Informatics

Stony Brook University

Stony Brook, NY, USA

fusheng.wang@stonybrook.edu

Abstract—P300 brain-computer interface (BCI) spellers can provide hands-free communication for people with severe motor impairments. Modern pipelines combine multiple individually promising components, often assuming that “more-is-better.” We tested this assumption using a four-component full-factorial experiment varying the inclusion of Euclidean Alignment (EA), xDAWN spatial filtering, subject calibration, and language model priors on a public P300 dataset. Performance was evaluated using accuracy, repetitions, and information transfer rate (ITR) with mixed-effects models. Results show that the value of components is conditional rather than additive. Calibration was the strongest singular contributor, while EA compensated for its absence in zero-calibration settings. Adding independently useful components could also reduce performance, revealing component antisynergy. Contrary to conventional wisdom, LM support was not universally beneficial: its effect depends strongly on the strength of the underlying EEG pipeline, while results from a larger LM showed a similar pattern. Together, these findings challenge maximal “all-on” pipeline design and highlight the value of selecting spatial and language-support components according to the quality of available EEG evidence.

Index Terms—Brain–computer interface, P300 speller, Euclidean Alignment, xDAWN, subject calibration

## I. INTRODUCTION

For people with severe communication and motor impairments, including individuals with amyotrophic lateral sclerosis (ALS), loss of speech and movement can make conventional input devices difficult or impossible to use [1]. Brain–computer interface (BCI) spellers offer a hands-free alternative by translating brain activity into character selections [2]. In a P300 speller using the Farewell-Donchin grid, rows or columns of a character matrix flash while the user attends to a target. The resulting P300 event-related potential (ERP) in the electroencephalography (EEG) signal is used to infer the intended character [3]. Practical systems must balance communication accuracy and efficiency, system calibration complexities, and robustness across distinct users.

Modern P300 pipelines combine multiple components to address these goals: Euclidean Alignment (EA) supports crosssubject/session transfer, xDAWN (XD) enhances P300 spatial representation, subject calibration (CB) personalizes the decoder, and language models (LM) offer contextual priors [4]– [7]. Although each one has demonstrated value, prior studies typically evaluate individual modules or aggregate pipeline performance, leaving their interactions unclear. Individually useful components may become complementary, redundant, or interfering when combined.

This uncertainty is especially notable for zero- or lowcalibration settings, which reduce setup burden but also limit subject-specific adaptability [8] and require other components to compensate for missing calibration. Moreover, although LM and, more recently, large language model (LLM) priors may improve contextual prediction, it remains unclear whether their effects depend on the strength of the underlying EEG evidence shaped by other pipeline components [7], [9]. We therefore ask: (1) Are component effects additive, redundant, or antisynergistic? (2) How do component behaviors differ with versus without calibration, and can any partially compensate for its absence? (3) How do LM and LLM priors behave when EEG decoding is well-aligned, personalized, or weak?

We address these questions using a public P300 matrixspeller dataset containing 10 subjects across three sessions [10], [11]. During processing, EA, xDAWN, calibration, and LM support were independently enabled or disabled in a 2<sup>4</sup> = 16 full-factorial design. Performance was evaluated using accuracy, repetitions per character, and information transfer rate (ITR) with linear mixed-effects models. Robustness was assessed using a larger LM, an alternative random-effects specification, and cross-session curation.

Our contributions are:

• We show through full-factorial analysis that P300 component value is conditional rather than additive and identify configurations with redundancy penalties.

• We clarify zero-calibration design: calibration is the dominant contributor, while xDAWN benefits uncalibrated decoding only when paired with EA and can reduce efficiency once EA and calibration are both present.

• We establish a boundary condition for LM/LLM support: language priors are harmful when EEG evidence is insufficiently aligned or personalized, whereas calibration largely neutralizes their interference.

## II. RELATED WORK

Previous P300 research has established benefits for components in different stages of the decoding pipeline. However, individual-component studies estimate stand-alone benefits, ● 16 EEG Channelswhereas complete-system studies emphasize aggregate perfor-<sup>Example</sup> <sup>Raw</sup> <sup>EEG</sup>mance. Neither directly reveals whether adding one component <sup>Epoching</sup> <sup>(relative</sup> <sup>to</sup> <sup>flash</sup> <sup>onset)</sup>changes the marginal value of another.

![](images/8c675da6c0ba4be5dea932bb96790d6f2c6f5145e03f25695e4ffee9d4221130.jpg)  
<sup>Decision</sup> <sup>Making</sup>Fig. 1. P300 Speller Pipeline and Full-Factorial Design

<sup>(yes</sup> <sup>P300)</sup>These interactions are particularly important for zero-<sup>(no</sup> <sup>P300) Select</sup> <sup>Char</sup>calibration decoding, where successful systems may rely on <sup>Future</sup> <sup>live</sup> <sup>LSL</sup> <sup>input</sup>compensatory mechanisms elsewhere in the pipeline. For (EA)example, Kindermans et al. combined inter-subject transfer, continuous adaptation, language modeling, and dynamic stopping [8]. EA and xDAWN may exhibit a similar compensatory interaction: EA improves cross-subject ERP transfer without labeled target-user data [4], while xDAWN enhances P300 spatial separation [12]. Yet it remains unclear whether EA can compensate for missing calibration, whether it is needed for xDAWN to remain beneficial under zero calibration, or whether the two become redundant once calibration is available.

A similar ambiguity applies to language support. Studies reporting benefits from LM- or LLM-assisted P300 spelling may have generally used trained or calibrated EEG decoders before evaluating language assistance: Oken et al. calibrated participants before LM-EEG fusion, while Speier et al. used training trials before online LM evaluation [13], [14]. Consequently, the independent effect of an LM under unaligned or unpersonalized EEG remains unclear, and a detrimental effect could be masked by stronger upstream decoding. These observations motivate jointly testing EA, xDAWN, calibration, and language support on their individual and interaction effects.

## III. METHODS

## A. Dataset and Simulated Online Decoding

We evaluated the P300 speller using the public BNCI2014 009 dataset distributed through the Mother of All BCI Benchmarks (MOABB) framework [10], [11]. The dataset contains EEG from 10 subjects across three sessions from 16 channels at 256 Hz during a visual P300 matrix-speller task. Recorded EEG was reorganized for simulated online evaluation. During decoding, characters were processed sequentially, allowing previously decoded text to provide LM context while preserving the original EEG responses associated with each stimulus presentation. <sup>ld</sup> <sup>comparison</sup>The primary analyses evaluated decoding within session, with cross-session decoding examined as a robustness test.

## B. P300 Speller Pipeline

The decoding pipeline is summarized in Fig. 1. Raw EEG was bandpass filtered (0.1-20 Hz) and epoched from −100 to +800 ms around each flash. Depending on the experimental condition, Euclidean Alignment (EA) [4] and xDAWN spatial filtering [5] were optionally applied before linear discriminant analysis (LDA) classification.

With calibration, the population LDA was adapted using empirical-Bayes shrinkage that blended population and subject-specific statistics [6] with weight,

$$
\alpha = \frac { n _ { \mathrm { c a l } } } { n _ { \mathrm { c a l } } + \kappa } ,\tag{1}
$$

where more calibration data increased subject-specific weighting [8]. Without calibration, the population model was used directly.

LDA evidence accumulated across repeated flashes for each candidate character. When enabled, the LM generated a character-level prior from the previously decoded text context [7]. This prior was introduced once at the beginning of each character selection and combined with subsequently accumulated EEG evidence [13]. After each repetition, character scores were converted to softmax probabilities; decoding stopped when the maximum probability exceeded the selection threshold; otherwise, another repetition followed [8].

## C. Full-Factorial Design and Evaluation Measures

EA, xDAWN (XD), calibration (CB), and LM were independently enabled or disabled, yielding 2<sup>4</sup> = 16 configurations while all other procedures remained fixed.

Performance was assessed using three outcomes: accuracy, the proportion of correctly decoded characters; repetitions, the mean number of stimulus repetitions per character, with lower value indicating greater efficiency; and information transfer rate (ITR), measured in bits/min and jointly reflecting accuracy and efficiency using Wolpaw’s definition [15]:

$$
B = \log _ { 2 } { N } + A \log _ { 2 } { A } + ( 1 - A ) \log _ { 2 } { \left( { \frac { 1 - A } { N - 1 } } \right) }\tag{2}
$$

$$
\mathrm { I T R } = { \frac { 6 0 B } { R t } }\tag{3}
$$

where B is bits/selection, A is accuracy, N the number of choices, R repetitions per selection, and t seconds per repetition.

## D. Primary Mixed-Effects Analysis

Within-session outcomes were analyzed separately using linear mixed-effects models with a subject random intercept. For outcome $Y _ { i j t }$ from subject i, sentence j, and configuration t, the four factors (CB, EA, XD, and LM) $X _ { k t }$ were effectcoded as −1 (off) and +1 (on). The model is

$$
\begin{array} { l } { { \displaystyle Y _ { i j t } = \beta _ { 0 } + \sum _ { k = 1 } ^ { 4 } \beta _ { k } X _ { k t } + \sum _ { k < \ell } \beta _ { k \ell } X _ { k t } X _ { \ell t } } } \\ { { \displaystyle ~ + \sum _ { k < \ell < m } \beta _ { k \ell m } X _ { k t } X _ { \ell t } X _ { m t } } } \\ { { \displaystyle ~ + \beta _ { 1 2 3 4 } \prod _ { k = 1 } ^ { 4 } X _ { k t } + u _ { i } + \epsilon _ { i j t } , } } \end{array}\tag{4}
$$

where the $\beta$ terms represent factorial effects, $u _ { i } \sim N ( 0 , \sigma _ { u } ^ { 2 } )$ is the subject random intercept, and $\epsilon _ { i j t } ~ \sim ~ N ( 0 , \sigma ^ { 2 } )$ is residual error. The four-way term preserved model hierarchy, while interpretation emphasized lower-order interactions. FDR-adjusted $p < . 0 5$ indicated significance.

## E. Robustness Check

Three robustness tests were conducted. First, we reestimated the within-session models with subject-and-sentence random intercepts, adding a sentence random intercept to account for variation across text items. Second, we applied the primary subject-random-intercept model to cross-session accuracy, repetitions, and ITR to test generalization under session transfer. Third, we replaced the primary LM with a larger contemporary LLM [9] and repeated the withinsession analysis using the primary mixed-effects specification. Together, these tests evaluated robustness to the random-effects structure, session transfer, and language-prior choice.

## IV. RESULTS

## A. Main Effects

Table I summarizes the estimated effects and FDR-adjusted p-values from the primary within-session mixed-effects models. Calibration produced the largest favorable main effects across accuracy, repetitions, and ITR, followed by EA. xDAWN showed no significant average main effect, whereas LM support significantly worsened all three outcomes. These contrasting main effects motivated examination of component interactions and possible anti-synergy.

MIXED-EFFECTS MODEL ESTIMATES FOR WORD-LEVEL OUTCOMES  
TABLE I
<table><tr><td>Contrast</td><td>Accuracy</td><td>Repetition</td><td>ITR</td></tr><tr><td>Main Effects</td><td></td><td></td><td></td></tr><tr><td>CB</td><td>0.288**</td><td>-20.158*</td><td>16.728**</td></tr><tr><td>EA</td><td>0.085**</td><td>-7.621*</td><td>5.654**</td></tr><tr><td>XD</td><td>0.045</td><td>-1.685</td><td>0.601</td></tr><tr><td>LM</td><td>-0.035*</td><td>3.455**</td><td>-2.013*</td></tr><tr><td>Two-way Interactions</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { X D } \times { \dot { \mathrm { C B } } }$ </td><td>-0.115**</td><td>12.908**</td><td>-9.624**</td></tr><tr><td> $\mathrm { X D } \times \mathrm { E A }$ </td><td>0.155*</td><td>-9.903*</td><td>4.361*</td></tr><tr><td> $\mathbf { L M } \times \mathbf { C B }$ </td><td>0.086**</td><td>-7.191**</td><td>4.369**</td></tr><tr><td> $\mathrm { L M } \times \mathrm { E A }$ </td><td>0.061*</td><td>-2.780*</td><td>1.089</td></tr><tr><td> $\mathrm { X D } \times \mathrm { L M }$ </td><td>0.064*</td><td>-1.695</td><td>1.037</td></tr><tr><td>CB × EA</td><td>-0.058</td><td>4.655</td><td>-0.428</td></tr><tr><td>Three-way Interactions</td><td></td><td></td><td></td></tr><tr><td> $\mathbf { \mathrm { X D } } \times \mathbf { \mathrm { C B } } \times \mathbf { \mathrm { E A } }$ </td><td>-0.410*</td><td>41.800**</td><td>-26.148**</td></tr><tr><td> $\mathbf { L M } \times \mathbf { C B } \times \mathbf { E A }$ </td><td>-0.118*</td><td>4.411*</td><td>-1.650</td></tr><tr><td> $\mathrm { X D } \ \times \ \mathrm { L M } \ \times \ \mathrm { C B }$ </td><td>-0.148*</td><td>3.948</td><td>-2.466</td></tr><tr><td> $\mathrm { X D } \times \mathrm { L M } \times \mathrm { E A }$ </td><td> $- 0 . 1 6 4 ^ { * * }$ </td><td>15.314**</td><td>-8.872**</td></tr><tr><td>Four-way Interaction</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { X D } \times \mathrm { L M } \times \mathrm { C B } \times \mathrm { E A }$ </td><td>0.320**</td><td>-30.274**</td><td>15.836**</td></tr></table>

Note: Subject-random-intercept mixed-effects models on with-session outcomes. CB = calibration, EA = Euclidean Alignment, XD = xDAWN, and LM = language model. Negative estimates indicate fewer repetitions but lower accuracy/ITR. FDR-corrected significance: $\textit { \textbf { * } } p < 0 . 0 5 , \textit { \textbf { * * } } p < \boldsymbol { \hat { 0 . } } 0 1 , \textit { \textbf { * * } } p < 0 . 0 0 1$

## B. Interaction Effects

Table I shows that component effects were conditional: xDAWN interacted with both calibration and EA across all outcomes, while LM interacted strongly with calibration. The focal three-way interactions are shown in Fig. 2, illustrating the effects of enabling xDAWN and LM on accuracy, repetitions, and ITR across EA and calibration conditions. Values represent component-on minus component-off mean differences.

1) xDAWN Anti-Synergy Depends on Calibration and EA: The EA × xDAWN × calibration interaction was significant across all three outcomes. Without calibration, xDAWN improved accuracy/ITR by +0.279/ + 13.96 when paired with EA, but worsened them by −0.075/ − 3.09 without EA; repetitions changed by −1.959 versus +0.578. After calibration, xDAWN added little without EA (+0.017 accuracy, +0.25 ITR) and reduced efficiency significantly when EA was also enabled (+0.821 repetitions, −9.13 ITR). Thus, xDAWN was beneficial in an EA-aligned zero-calibration pipeline but showed a clear redundancy penalty after alignment and calibration were both present.

2) LM Interference Depends on EEG Support: The EA × LM × calibration interaction was significant for accuracy and repetitions, but not ITR. Without calibration, LM harm was greatest without EA (accuracy/ITR: −0.148/ − 5.09) and smaller with EA $( - 0 . 0 2 1 / - 3 . 0 2 ) ;$ repetitions increased by 0.722 and 0.342. After calibration, LM effects were nearly neutral across EA conditions (accuracy $- 0 . 0 0 1 / \mathrm { ~ - ~ } 0 . 0 0 1$ repetitions −0.097/ − 0.087; ITR +0.93/ + 0.50). Thus, EA partially mitigated LM interference under zero calibration, while calibration largely neutralized it.

![](images/9c44972a46f46480d188d9d0dd6fcd8e4a680539221650dec0e23d804fdc48ff.jpg)  
Fig. 2. xDAWN and LM Effects Conditional on Calibration and Euclidean Alignment Note: $^ { * } p _ { \mathrm { F D R } } < . 0 5 ; ^ { * * } p _ { \mathrm { F D R } } < . 0 0 5$

## C. Robustness Analyses

The robustness tests largely preserved the core findings (Table II). First, adding a sentence random intercept retained the focal xDAWN interaction across all outcomes and the LM interaction for accuracy and repetitions. Second, cross-session analysis again supported xDAWN’s dependence on EA and calibration. It also retained the negative LM main effect and the LM × calibration interaction across all outcomes, although the EA × LM × calibration interaction remained significant only for repetitions. Third, replacing the LM with a larger LLM reproduced the negative language-prior main effect and its attenuation by calibration; its three-way interaction was significant for accuracy. Overall, the most stable results were calibration’s dominant benefit, xDAWN’s dependence on EA under zero calibration, the negative average effect of language priors, and calibration’s mitigation of that interference.

## V. DISCUSSION

Our key contribution is the identification of component anti-synergy: components that are beneficial individually can become redundant or even harmful when combined. Calibration was the dominant contributor, while EA enabled xDAWN to improve zero-calibration decoding. Once both EA and calibration were present, however, xDAWN increased repetitions and reduced ITR. A plausible explanation is that EA and calibration already improve the signal representation through alignment and personalization, leaving xDAWN with little additional value while adding redundant spatial filtering [4], [5]. Future work should test this mechanism directly.

The language-model results reveal a second important boundary condition. Previous studies reporting benefits from language information generally evaluated trained or calibrated EEG decoders [14], [17]. In contrast, both LM and LLM priors in our study were harmful on average, especially when neither EA nor calibration was present. EA partly mitigated this interference, while calibration largely neutralized it. One possible explanation is that Weak EEG evidence may allow contextual priors to dominate and propagate decoding errors. The larger LLM showed the same pattern, suggesting that model scale alone does not resolve the fusion problem [9], [13]. Future systems may therefore benefit from adjusting LM/LLM influence based on EEG quality.

Table III translates the interaction findings into practical configuration guidance. The calibrated recommendation (EA+CB) achieved the best overall performance, with the highest accuracy (0.933) and ITR (35.5 bits/min) and the fewest repetitions (3.20), outperforming both naive baselines. Under zero calibration, EA+xDAWN substantially improved over the minimalist baseline (0.819 vs. 0.731 accuracy; 25.3 vs. 16.1 ITR; 3.67 vs. 4.82 repetitions) and also required fewer repetitions than the maximalist baseline. This suggests a practical low-burden configuration when subject-specific calibration is unavailable. After calibration, retaining EA while removing xDAWN avoids the observed efficiency penalty. Overall, these results support selective rather than maximal component configuration.

Limitations include the use of a single public dataset with simulated online EEG recording. The system was also not tested with real ALS patients, a key target population for P300 communication systems. Future work should therefore validate the observed interactions across additional datasets and headsets, and test genuine online use with clinical participants.

TABLE II  
ROBUSTNESS TESTS FOR MIXED-EFFECTS MODEL ESTIMATES
<table><tr><td></td><td colspan="3">Robustness Test 1: Alternative Mixed-Effects Model</td><td colspan="3">Robustness Test 2: Cross-Session Outcomes</td><td colspan="3">Robustness Test 3: Alternative LLM Model</td></tr><tr><td>Contrast</td><td>Accuracy</td><td>Repetition</td><td>ITR</td><td>Accuracy</td><td>Repetition</td><td>ITR</td><td>Accuracy</td><td>Repetition</td><td>ITR</td></tr><tr><td colspan="10">Main Effects</td></tr><tr><td>CB</td><td> $0 . 2 8 8 ^ { \ast \ast \ast }$ </td><td> $- 2 0 . 1 5 8 ^ { * * * }$ </td><td>16.728***</td><td>0.235</td><td>-20.158*</td><td>14.987*</td><td>0.287**</td><td>-20.307*</td><td>16.926**</td></tr><tr><td>EA</td><td> $0 . 0 8 5 ^ { \ast \ast \ast }$ </td><td> $- 7 . 6 2 1 ^ { \ast \ast \ast }$ </td><td>5.654***</td><td>0.091**</td><td>-7.477**</td><td>5.169**</td><td>0.087**</td><td>-7.345*</td><td>5.518**</td></tr><tr><td>XD</td><td> $0 . 0 4 5 ^ { * * * }$ </td><td> $- 1 . 6 8 5 ^ { * * * }$ </td><td>0.601*</td><td>0.044*</td><td>-3.256**</td><td>1.123</td><td>0.046</td><td>-1.980</td><td>0.696</td></tr><tr><td>LM</td><td> $- 0 . 0 3 5 ^ { * * * }$ </td><td> $3 . 4 5 5 ^ { \ast \ast \ast \ast }$ </td><td> $- 2 . 0 1 3 ^ { \ast \ast \ast }$ </td><td>-0.075**</td><td>3.875**</td><td>-3.319**</td><td>-0.043**</td><td>2.643**</td><td>-1.668**</td></tr><tr><td colspan="10">Two-way Interactions</td></tr><tr><td> $\mathrm { X D } \ \times \ \mathrm { C B }$ </td><td> $- 0 . 1 1 5 ^ { * * * }$ </td><td> $1 2 . 9 0 8 ^ { * * * }$ </td><td> $- 9 . 6 2 4 ^ { \ast \ast \ast }$ </td><td>-0.093**</td><td>9.985**</td><td>-7.118**</td><td>-0.111**</td><td>12.608**</td><td>-9.470**</td></tr><tr><td> $\mathrm { X D } \times \mathrm { E A }$ </td><td> $0 . 1 5 5 ^ { \ast \ast \ast }$ </td><td> $- 9 . 9 0 3 ^ { * * * }$ </td><td> $4 . 3 6 1 ^ { \ast \ast \ast }$ </td><td>0.167**</td><td>-9.373*</td><td>4.401</td><td>0.152*</td><td>-9.700*</td><td>4.232*</td></tr><tr><td> $\mathbf { L M } \times \mathbf { C B }$ </td><td> $0 . 0 8 6 ^ { \ast \ast \ast }$ </td><td> $- 7 . 1 9 1 ^ { \ast \ast \ast }$ </td><td>4.369***</td><td>0.087**</td><td>-7.905**</td><td>3.909*</td><td>0.084**</td><td>-7.489**</td><td>4.765**</td></tr><tr><td> $\mathbf { L M } \times \mathbf { E A }$ </td><td> $0 . 0 6 1 ^ { \ast \ast \ast }$ </td><td> $- 2 . 7 8 0 ^ { * * }$ </td><td>1.089</td><td>0.064*</td><td>-2.804*</td><td>1.288</td><td>0.064*</td><td>-2.226</td><td>0.818</td></tr><tr><td colspan="10">Three-way Interactions</td></tr><tr><td> $\mathrm { X D } \times \mathrm { C B } \times \mathrm { E A }$ </td><td> $- 0 . 4 1 0 ^ { * * * }$ </td><td> $4 1 . 8 0 0 ^ { * * * }$ </td><td> $- 2 6 . 1 4 8 ^ { * * * }$ </td><td>-0.408**</td><td>41.428*</td><td> $- 2 4 . 7 3 3 ^ { \ast \ast }$ </td><td>-0.404</td><td>41.488**</td><td>-25.618**</td></tr><tr><td> $\mathbf { L M } \times \mathbf { C B } \times \mathbf { E A }$ </td><td> $- 0 . 1 1 8 ^ { * * * }$ </td><td> $4 . 4 1 1 ^ { \ast * }$ </td><td>-1.650</td><td>-0.129</td><td>5.223*</td><td>-1.424</td><td>-0.127*</td><td>4.675</td><td>-2.503</td></tr></table>

Note: Only focal interactions are reported. Tests 1 and 3 use within-session outcomes; Tests 2 and 3 use a subject-random-intercept model. Tests 1–2 use the LM, and Test 3 uses the LLM. Significance (FDR-corrected): ${ } ^ { * } \ p < 0 . 0 5 ,$ \*\* $p < 0 . 0 1 ,$ 1, \*\*\* $p < 0 . 0 0 1$

TABLE III  
INTERACTION-GUIDED CONFIGURATION STRATEGIES
<table><tr><td>Strategy</td><td>EA, XD, CB, LM</td><td>Acc.</td><td>Rep.</td><td>ITR</td></tr><tr><td>Naive minimalist</td><td>(All not included)</td><td>0.731</td><td>4.82</td><td>16.1</td></tr><tr><td>Naive maximalist</td><td>(All included)</td><td>0.905</td><td>4.02</td><td>27.3</td></tr><tr><td>Calibrated recommendation</td><td>(EA, CB)</td><td>0.933</td><td>3.20</td><td>35.5</td></tr><tr><td>Zero-calib. recommendation</td><td>(EA, XD)</td><td>0.819</td><td>3.67</td><td>25.3</td></tr></table>

Configuration order: EA/xDAWN/calibration/LM. Higher accuracy and ITR and lower repetitions are preferable.

## VI. CONCLUSION

We evaluated EA, xDAWN, subject calibration, and LM support across 16 full-factorial configurations. Three key findings emerged: component anti-synergy, where adding individually useful modules can reduce performance; negative language-model effects, particularly when neither calibration nor EA is present; and EA as a potential compensatory mechanism under zero calibration, especially by enabling xDAWN. Together, these results suggest potential value in conditionally configuring P300 pipelines, with spatial and language support adapted to the available EEG evidence.

## ACKNOWLEDGMENT

The authors would like to thank Dr. Fusheng Wang and the CSIRE program at Stony Brook University for supporting this research.

## REFERENCES

[1] F. Nijboer, E. W. Sellers, J. Mellinger, M. A. Jordan, T. Matuz, A. Furdea, S. Halder, U. Mochty, D. J. Krusienski, T. M. Vaughan, J. R. Wolpaw, N. Birbaumer, and A. Kubler, “A P300-based brain-computer¨ interface for people with amyotrophic lateral sclerosis,” Clinical Neurophysiology, vol. 119, no. 8, pp. 1909–1916, 2008.

[2] E. W. Sellers and E. Donchin, “A P300-based brain-computer interface: initial tests by ALS patients,” Clinical Neurophysiology, vol. 117, no. 3, pp. 538–548, 2006.

[3] L. A. Farwell and E. Donchin, “Talking off the top of your head: toward a mental prosthesis utilizing event-related brain potentials,” Electroencephalography and Clinical Neurophysiology, vol. 70, no. 6, pp. 510–523, 1988.

[4] H. He and D. Wu, “Transfer learning for brain-computer interfaces: A Euclidean space data alignment approach,” IEEE Transactions on Biomedical Engineering, vol. 67, no. 2, pp. 399–410, 2020.

[5] B. Rivet, A. Souloumiac, V. Attina, and G. Gibert, “xDAWN algorithm to enhance evoked potentials: application to brain-computer interface,” IEEE Transactions on Biomedical Engineering, vol. 56, no. 8, pp. 2035– 2043, 2009.

[6] P.-J. Kindermans, H. Verschore, D. Verstraeten, and B. Schrauwen, “A P300 BCI for the masses: Prior information enables instant unsupervised spelling,” in Advances in Neural Information Processing Systems 25 (NIPS 2012), P. Bartlett, F. C. N. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger, Eds. Lake Tahoe, NV, USA: NIPS Foundation, 2012, pp. 719–727.

[7] W. Speier, C. Arnold, J. Lu, A. Deshpande, and N. Pouratian, “Integrating language information with a hidden Markov model to improve communication rate in the P300 speller,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 22, no. 3, pp. 678–684, 2014.

[8] P.-J. Kindermans, M. Tangermann, K.-R. Muller, and B. Schrauwen,¨ “Integrating dynamic stopping, transfer learning and language models in an adaptive zero-training ERP speller,” Journal of Neural Engineering, vol. 11, no. 3, p. 035005, 2014.

[9] J. Hong, W. Wang, and L. Najafizadeh, “ChatBCI, a P300 speller BCI with context-driven word prediction leveraging large language models, from concept to evaluation,” Scientific Reports, vol. 16, no. 1, p. 6379, 2026.

[10] P. Arico, F. Aloise, F. Schettini, S. Salinari, D. Mattia, and F. Cin-\` cotti, “Influence of P300 latency jitter on event related potential-based brain-computer interface performance,” Journal of Neural Engineering, vol. 11, no. 3, p. 035008, 2014.

[11] V. Jayaram and A. Barachant, “MOABB: trustworthy algorithm benchmarking for BCIs,” Journal of Neural Engineering, vol. 15, no. 6, p. 066011, 2018.

[12] F. Li, Y. Xia, F. Wang, D. Zhang, X. Li, and F. He, “Transfer learning algorithm of P300-EEG signal based on XDAWN spatial filter and Riemannian geometry classifier,” Applied Sciences, vol. 10, no. 5, p. 1804, 2020.

[13] W. Speier, C. Arnold, N. Chandravadia, D. Roberts, S. Pendekanti, and N. Pouratian, “Improving P300 spelling rate using language models and predictive spelling,” Brain-Computer Interfaces, vol. 5, no. 1, pp. 13–22, 2018.

[14] B. S. Oken, U. Orhan, B. Roark, D. Erdogmus, A. Fowler, A. Mooney, B. Peters, M. Miller, and M. B. Fried-Oken, “Brain-computer interface with language model-electroencephalography fusion for locked-in syndrome,” Neurorehabilitation and Neural Repair, vol. 28, no. 4, pp. 387–394, 2014.

[15] J. R. Wolpaw, H. Ramoser, D. J. McFarland, and G. Pfurtscheller, “EEGbased communication: improved accuracy by response verification,” IEEE Transactions on Rehabilitation Engineering, vol. 6, no. 3, pp. 326–333, 1998.