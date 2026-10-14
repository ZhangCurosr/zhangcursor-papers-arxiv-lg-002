# Learning When to Transfer: Reinforcement Learning for Carbon-Footprint Modeling in New Manufacturing Domains

Muhammad Umar Farooq Daniel R. Cooper Mechanical Engineering Department, University of Michigan 2350 Hayward Street, Ann Arbor, 48109, MI, United States Correspondence: muf@umich.edu

## Abstract

Manufacturers often need to characterize a new material, process, or product before enough operating data are available to train reliable predictive models for this new, target domain. Existing manufacturing operations contain potentially useful knowledge to accelerate this characterization, but indiscriminate reuse of data from existing source domains can cause negative transfer in learning of the new, target domain when the manufacturing domains differ sufficiently. This work proposes a reinforcement-learning method for selective knowledge transfer, where its reward scores each opportunity to transfer against supervised alternatives, so negative transfer is avoided and controlled during forward prediction instead of being diagnosed afterwards. Given limited data from a new target domain, the learned policy chooses among predictors based on individual source operations, subsets of the sources, or the supervised alternatives: fully pooled sources, or target data alone. A validation gate deploys the transferred prediction only when it improves upon the supervised alternatives. The method is designed for learning manufacturing variables such as process energy and material consumption, which populate life-cycle inventories and support carbon accounting, as well as performance and quality outcomes. We demonstrate the approach by quantifying the embodied greenhouse-gas emissions of parts produced via material-extrusion additive manufacturing, using data from 729 experiments spanning five materials: ABS, PETG, TPU, PLA, and recycled PLA. The response combines measured fabrication electricity with polymer-specific cradle-to-gate production factors, while 22 material, process, and toolpath features characterize each experiment. In leave-one-material-out evaluation, the proposed method achieves a mean test RMSE of 5.2 g.CO e per part, 20.0% lower than target-only training (32.5% lower when only 5–10% of the target training data are available) and 10.3% lower than indiscriminate source-target pooling. The validation gate reduces the observed negative-transfer rate from 33% to 17%, compared with 49% for pooling. These results demonstrate how manufacturers can selectively reuse data from existing operations to accelerate carbon-footprint modeling for a newly introduced material while limiting harmful transfer.

## 1 Introduction

Manufacturing and production activities account for approximately one-fifth of global carbon emissions [1]. Manufacturing decarbonization depends on the timely development of accurate life-cycle inventories (LCIs), which underpin cradle-to-gate life-cycle assessments (LCAs) and enable informed, environmentally motivated decision-making. Decisions about a new material, process, or product therefore require foreground data on energy requirements, material consumption, yield, and the generation of wastes and by-products. Yet these measurements are local to the operating domain, and compiling domain-specific LCIs remains costly and time-consuming [2].

Mechanistic simulations and measurements from existing operations can reduce this burden. Previous work has already shown how correlation-aware multi-fidelity learning can combine low-fidelity physics-based priors with a small number of experiments (providing sparse high-fidelity data) to accelerate development of manufacturing LCIs [3]. Historical operating data create a complementary opportunity: predictors already fitted to prior domains can encode useful process–response structure. However, a new domain may alter material properties, feasible settings, toolpaths, or dominant energy mechanisms. Pooling all historical observations can then introduce bias, while choosing a source by descriptive similarity alone may overlook response-level compatibility [4]. Therefore, using data and/or models from existing manufacturing domains to accelerate LCI model development on a new domain requires explicit decisions on whether, what, and how much of this existing knowledge to transfer.

This issue is visible in additive manufacturing (AM). Data-driven LCA and G-code models estimate energy and environmental performance earlier in design [5, 6], but reported models are commonly tied to one material, machine, geometry, or sensing configuration. AM datasets are often small because experiments are slow and costly [7]; transfer learning can reduce this burden, including cross-material quality monitoring [8]. More generally, manufacturing studies use source screening, instance reweighting, representation alignment, and multi-source ensembles to improve transfer [9, 10, 11]. These methods establish that source usefulness is heterogeneous, but typically optimize within a predetermined transfer pipeline. They do not treat target-only learning and alternative transfer configurations as competing deployment choices.

This work addresses the gap by framing low-shot manufacturing prediction as a contextual, one-step reinforcement-learning problem. One policy, trained across source domains, weights individualsource, source-subsets, fully pooled sources, and target-only predictors. A separate validation gate deploys its mixture only when it improves on the supervised (target-only or fully pooled) baseline alternatives. Reinforcement learning is used because the output is a decision: the correct transfer choice for a new material is never labelled, its value is defined relative to a supervised alternative, and the policy replaces that alternative only when held-out evidence favours it [12, 13]. The contributions are: (i) a material-agnostic decision formulation that represents domain distance, source competence, predictor disagreement, and target-data budget; (ii) a controlled leave-one-material-out protocol that separates fitting, model-selection, and test evidence; and (iii) an empirical analysis of accuracy and negative transfer across five material (polymer AM) domains and seven target-data budgets, against direct learning (the supervised target-only alternative) and policy ablations.

## 2 Problem formulation and method

Figure 1 illustrates the manufacturing-domain policy workflow, with its key components summarized below and described in greater detail in the Appendix.

## 2.1 Experimentation and carbon inventory

The dataset contains 729 controlled material-extrusion prints: 145 each for ABS, PETG, PLA, and recycled PLA (rPLA), and 149 for TPU. All trials use the same specimen geometry and harmonized schema. For each material, a design of experiments is conducted over the AM process settings. Electrical power is recorded at 1 Hz; process settings and G-code are archived. The 22 predictors comprise 11 material descriptors, six controllable process parameters, and five toolpath aggregates. For experiment i using polymer j, the per-part response (in $\mathrm { k g C O _ { 2 } e ) }$ is

$$
C _ { i } = C _ { E } E _ { i } + C _ { m , j } m _ { i } ,\tag{1}
$$

where $E _ { i }$ is the measured fabrication electricity (kWh), $m _ { i }$ is the filament mass used $( \ker ) , C _ { E } =$ 0.587 $\mathrm { k g C O _ { 2 } e / k W h }$ is the location-specific electricity factor [14], and $C _ { m , j } ~ \mathrm { ( k g C O _ { 2 } e / k g ) }$ is a polymer-specific cradle-to-gate filament production factor [15, 16, 17, 18]. Equation 1 is an operational-plus-material inventory indicator, not a complete comparative LCA. Figure 2 shows the printing setup, the resulting carbon-response distributions, and a two-dimensional projection of the material descriptors.

![](images/56c90324cedd3d6671e4a41f385584c196b97ee4c39d63b374296978d64d9b93.jpg)  
Figure 1: Manufacturing-development context and transfer-policy workflow. Top: conceptual links between process development, supporting data, and manufacturing domains. Bottom: target-domain evidence informs an actor that assigns weights to candidate predictors. A critic supports rewardbased policy training, and a validation gate selects between the policy mixture and the stronger supervised alternative.

## 2.2 Material-agnostic transfer policy

For target domain t and fitting budget b, the candidate bank $\mathcal { C } _ { t } ( b ) = \{ g _ { 1 } , . . . , g _ { C } \}$ contains predictors trained on each source separately, selected source subsets plus target data, all sources plus target data, and target data alone. The policy state records $b ,$ target-set summaries, fold-standardized material distances, source-expert fitting errors, prediction disagreement, and validation performance of the target-only and pooled supervised models. Material-agnostic means that one decision rule is shared across materials; material differences remain explicit state variables.

An action $a ~ \in ~ \Delta ^ { C - 1 }$ assigns nonnegative weights summing to one to the candidate predictors, giving $\begin{array} { r } { g _ { a } ( x ) = \sum _ { j = 1 } ^ { C } a _ { j } g _ { j } ( x ) } \end{array}$ . Let $\widehat { \boldsymbol { a } } _ { \boldsymbol { \theta } } ( \boldsymbol { s } )$ denote the weights produced by the policy for deployment in state $s ;$ the policy predictor is $g _ { \mathrm { p o l } } ( x ) = g _ { \widehat { a } _ { \theta } ( s ) } ( x )$ . Behaviour cloning initializes the actor toward validation-optimal convex mixtures. Actor–critic refinement rewards low validation normalized RMSE (NRMSE) and improvement over the stronger supervised alternative, while penalizing underperformance relative to that alternative and the validation-optimal mixture. Each target–budget case has one state, one model-selection action, and one validation reward: this is contextual reinforcement learning, not sequential process control.

## 2.3 Evaluation and validation gate

Policy generalization is evaluated using leave-one-material-out: four materials form policy-training cases and the fifth is the unseen target. Each material is partitioned before transfer into mutually exclusive training (50%), validation (20%), and test (30%) sets. Target fitting uses 5, 10, 20, 40, 60, 80, or 100% of the training partition, with eleven repetitions per budget; below 100% each repetition draws a different fitting subset, and at 100% the repetitions share the full training partition and differ only in learner random seeds. Policy-training rewards use the validation data of source materials treated as pseudo-targets. For the held-out target, validation data provide the supervised modelselection evidence, validation-state features, and deployment-gate comparison, without updating the policy parameters; thus, the budgets quantify scarce fitting data, not total labelled data. The untouched target test set measures generalization after all choices are fixed.

At deployment, the baseline model $g _ { \mathrm { f b } }$ is defined by selecting the better-performing model $( \mathrm { i . e . }$ , the one with lower target-validation NRMSE) among the target-only model and the model trained on all sources plus the target fitting subset. The complete policy (actor–critic + validation gate) deploys

$$
g _ { \mathrm { f i n a l } } = \left\{ \begin{array} { l l } { g _ { \mathrm { p o l } } , } & { \mathrm { N R M S E _ { \mathrm { p o l , v a l } } \leq N R M S E _ { \mathrm { f b , v a l } } , } } \\ { g _ { \mathrm { f b } } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{2}
$$

This validation-based deployment heuristic does not guarantee lower test error.

![](images/c8f8ded192856fdbeb0a5efade0a2f4961918fbba97b9197d24ae14bce08b5b3.jpg)

![](images/b748f07daa20cb94c2563933c61196086e6e642cc34df1707c72ea62f2f18c27.jpg)

![](images/30d9ce8f6e1fe2a5674d21a84c8dd2a850a07f5a5e3e3199e9cc027178a21644.jpg)  
Figure 2: Experimental setup and characterization of the five material domains:(a) materialextrusion printing; (b) estimated density curves of the per-part carbon response defined in Eq. 1, in $\mathrm { k g C O _ { 2 } e / p a r t } ;$ and (c) material descriptors projected onto the first two principal components, PC1 and PC2.

## 3 Results and discussion

## 3.1 Transfer structure and overall performance

Source-expert transfer is directional: PETG and TPU transfer effectively in both directions; however, PETG transfers effectively to rPLA but not vice versa. Overall, ABS is the most difficult target. Material distance (the Wasserstein distance between standardized material-descriptor distributions) summarizes differences in material descriptors. For fixed empirical distributions, this distance is symmetric, whereas predictive transfer errors can differ between directions. Moreover, similar material descriptors do not imply similar relationships between process inputs and carbon response. The policy therefore uses source-expert errors on the target fitting subset, disagreement among sourceexpert predictions for the same inputs, and target-data availability alongside distance.

Across held-out materials, fitting budgets, and subset draws, the complete policy ("actor-critic + gate" in Table 1) achieves the lowest mean test RMSE, 5.2 $\mathrm { g C O _ { 2 } e / p a r t }$ (Table 1). This is 20.0% lower than target-only fitting and 10.3% lower than fully pooled model. The relative benefit of the complete policy grows as target data become scarce. The ablations separate the actor-critic policy learning from the effect of the validation gate: raw actor–critic obtains 5.6 $\mathrm { g C O _ { 2 } e / p a r t } .$ behaviour cloning with the gate obtains 5.3 $\mathrm { g C O _ { 2 } e / p a r t }$ , and actor–critic with the gate obtains $5 . 2 ~ \mathrm { g C O _ { 2 } e / p a r t }$ The ordering indicates that the central advantage comes from learning over a diverse candidate bank while retaining a strong supervised alternative. Under the gate, actor–critic refinement lowers the negative-transfer rate from 28% to 17% (McNemar $p = \bar { 4 } \times 1 0 ^ { - 5 }$ across repetitions for these materials), while mean RMSE reduces by only 0.1 $\mathrm { g } C O _ { 2 } e / p a r t$

## 3.2 Target-data budget and source composition

Relative predictive performance varies with both the target material and the target fitting-data budget (Table 1; Fig. 3). When target data are scarce $( b ~ \leq ~ 0 . 1 0 )$ , the complete policy lowers RMSE by 32.5% relative to target-only fitting, with negative-transfer rates of 2–4%, against 15–20% for pooling. TPU shows the clearest transition: target-only fitting is weak at a target-data budget of 5–10% but becomes preferable after approximately 40% of the target training partition. Across all 35 material–budget cells, the complete policy results in mean RMSE on the target test set within 15% of the best fixed rule chosen in hindsight; whereas, behaviour cloning with the gate, and the raw actor–critic are up to 86% and 134% higher, respectively.

Table 1: Held-out test RMSE $\mathrm { ( k g C O _ { 2 } e / p a r t ) }$ and negative-transfer rate by target fitting budget b. Negative-transfer rate is 100 times the number of cases in which a method’s test RMSE exceeds target-only test RMSE, divided by the number of evaluated cases. Each budget contains 5 materials × 11 repetitions = 55 cases against each budget; overall results contain $5 \times 7 \times 1 1 = 3 8 5$ cases. RMSE entries are means; the Avg. RMSE and Test $R ^ { 2 }$ entries report mean ± standard deviation across all cases. AC: actor–critic; BC: behaviour cloning; Pooled: all sources + target.
<table><tr><td></td><td colspan="5">Test RMSE (kgCO2e/part)</td><td colspan="4">Negative-transfer rate (%)</td></tr><tr><td>b</td><td>AC + gate</td><td>BC + gate</td><td>Raw AC</td><td>Pooled</td><td>Target only</td><td>AC + gate BC + gate Raw AC</td><td></td><td></td><td>Pooled</td></tr><tr><td>0.05</td><td>0.0072</td><td>0.0076</td><td>0.0076</td><td>0.0075</td><td>0.0113</td><td>4</td><td>11</td><td>11</td><td>15</td></tr><tr><td>0.10</td><td>0.0065</td><td>0.0067</td><td>0.0069</td><td>0.0069</td><td>0.0090</td><td>2</td><td>16</td><td>22</td><td>20</td></tr><tr><td>0.20</td><td>0.0058</td><td>0.0059</td><td>0.0061</td><td>0.0062</td><td>0.0069</td><td>18</td><td>29</td><td>33</td><td>38</td></tr><tr><td>0.40</td><td>0.0047</td><td>0.0049</td><td>0.0051</td><td>0.0054</td><td>0.0053</td><td>22</td><td>33</td><td>38</td><td>51</td></tr><tr><td>0.60</td><td>0.0044</td><td>0.0044</td><td>0.0047</td><td>0.0053</td><td>0.0046</td><td>27</td><td>31</td><td>42</td><td>69</td></tr><tr><td>0.80</td><td>0.0039</td><td>0.0041</td><td>0.0043</td><td>0.0048</td><td>0.0042</td><td>25</td><td>33</td><td>45</td><td>65</td></tr><tr><td>1.00</td><td>0.0037</td><td>0.0039</td><td>0.0043</td><td>0.0047</td><td>0.0040</td><td>24</td><td>40</td><td>42</td><td>82</td></tr><tr><td>Avg.  $R ^ { 2 }$ </td><td>0.0052±0.0021</td><td>0.0053±0.0022</td><td>0.0056±0.0022</td><td>0.0058±0.0023</td><td>0.0065±0.0033</td><td>17</td><td>28</td><td>33</td><td>49</td></tr><tr><td>Test</td><td>0.776±0.215</td><td>0.726±0.213</td><td>0.720±0.189</td><td>0.663±0.265</td><td>0.493±0.661</td><td></td><td></td><td></td><td></td></tr></table>

Useful information is also compositional rather than purely pairwise. Frequently, competitive subsets include TPU+rPLA, PETG+TPU, and ABS+PLA. A source that is weak in isolation can still contribute after joint training with other sources and target data. For example, ABS appears in useful multi-source subsets despite being a poor stand-alone source for several targets. Selective subset transfer can therefore exploit complementary structure that is lost under both nearest-source selection (the source with the smallest Wasserstein distance to the target) by material distance and indiscriminate pooling.

## 3.3 Validation-gated deployment and manufacturing relevance

The validation gate converts this target-dependent transfer structure into a deployment decision. Raw actor–critic exceeds target-only test RMSE in 33% of evaluated cases; the gate reduces the observed rate to 17%, compared with 49% for all-sources-plus-target pooling (Appendix Fig. 4).

The manufacturing relevance of this work is that it enables manufacturers to develop more accurate LCAs of emerging manufacturing domains—including materials, processes, and products—by modeling the response in Eq. 1. These improved assessments support better-informed environmental decision-making early in development, when design and process choices can have the greatest influence on long-term environmental impacts. Ultimately, this approach helps identify and evaluate pathways toward deep decarbonization.

## 4 Conclusion

Across 729 material-extrusion experiments, the new material-agnostic policy lowers mean test RMSE relative to target-only fitting and indiscriminate pooling, while its validation gate approximately halves the raw policy’s observed negative-transfer rate. The results indicate that selective reuse of existing manufacturing data supports accurate and rapid carbon-footprint modeling when manufacturers introduce a new material, with the largest gains when target data are scarce.

## References

[1] World Economic Forum. Reducing the carbon footprint of the manufacturing industry through data sharing, 2025.

[2] Muhammad Umar Farooq and Sidi Deng. Data-driven sustainability assessment of advanced manufacturing processes. The International Journal ofAdvanced Manufacturing Technology, 142(9):4245–4288, February 2026.

[3] Muhammad Umar Farooq and Daniel R. Cooper. Accelerating life cycle inventory development through correlation-aware multi-fidelity transfer learning: Case studies in cradle-to-gate manufacturing energy modeling. Journal ofManufacturing Systems, 88:544–561, 2026.

[4] Zirui Wang, Zihang Dai, Barnabás Póczos, and Jaime Carbonell. Characterizing and avoiding negative transfer. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11285–11294. IEEE, 2019.

[5] Ahmed Z. Naser, Fantahun Defersha, Xun Xu, and Sheng Yang. Automating life cycle assessment for additive manufacturing with machine learning: Framework design, dataset buildup, and a case study. Journal of Manufacturing Systems, 71:504–526, 2023.

[6] Shujun Yu, Heng Liu, Gang Zhao, Hua Zhang, Feng Hou, and Kuankuan Xu. A code-based method for carbon emission prediction of 3D printing: A case study on the fused deposition modeling (FDM) 3D printing and comparison with conventional approach. Journal of Cleaner Production, 484:144341, December 2024.

[7] Durant Fullington, Emmanuel Yangue, Mahathir Mohammad Bappy, Chenang Liu, and Wenmeng Tian. Leveraging small-scale datasets for additive manufacturing process modeling and part certification: Current practice and remaining gaps. Journal of Manufacturing Systems, 75:306–321, August 2024.

[8] Jingchang Li, Jiexiang Hu, Qi Zhou, and Yahui Zhang. Transfer learning-based quality monitoring of laser powder bed fusion across materials. Expert Systems with Applications, 252:124150, October 2024.

[9] Wentao Mao, Wen Zhang, Ke Feng, Michael Beer, and Chunsheng Yang. Tensor representation-based transferability analytics and selective transfer learning of prognostic knowledge for remaining useful life prediction across machines. Reliability Engineering & System Safety, 242:109695, February 2024.

[10] Biyao Qiang, Kaining Shi, Junxue Ren, and Yaoyao Shi. Multi-source online transfer learning based on hybrid physics-data model for cross-condition tool health monitoring. Journal ofManufacturing Systems, 77:1–17, 2024.

[11] Zhilie Gao, Ni Chen, Yingfei Yang, and Liang Li. An innovative multisource multibranch metric ensemble deep transfer learning algorithm for tool wear monitoring. Advanced Engineering Informatics, 62:102659, 2024.

[12] Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh. High confidence policy improvement. In Proceedings of the 32nd International Conference on Machine Learning, pages 2380–2388, 2015.

[13] Romain Laroche, Paul Trichelair, and Remi Tachet des Combes. Safe policy improvement with baseline bootstrapping. In Proceedings of the 36th International Conference on Machine Learning, pages 3652– 3661, 2019.

[14] Climatiq. Emission factor: electricity supplied from grid, supplier DTE Energy, Michigan, US. https://www.climatiq.io/data/emission-factor/ 01dbd61d-53b8-427d-9b44-eb568a30f427. Accessed 15 March 2026.

[15] Jakub Kocí. Introducing Prusament PETG recycled with calculated life cycle assessment!, March 2022.ˇ

[16] American Chemistry Council. Cradle-to-Gate Life Cycle Analysis of Acrylonitrile Butadiene Styrene (ABS) Resin, November 2022.

[17] Spot Material. Plastic, TPU, material report. https://spotmaterial.com/material/ thermoplastic-polyurethane-tpu/plastic-tpu/. Accessed 15 March 2026.

[18] TotalEnergies Corbion. Environmental footprint of polylactic acid production. Cradle-to-gate life cycle assessment of Luminy PLA, ISO 14040/44/67.

[19] Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3):229–256, 1992.

[20] Po-Wei Chou, Daniel Maturana, and Sebastian Scherer. Improving stochastic policy gradients in continuous control with deep reinforcement learning using the Beta distribution. In Proceedings of the 34th International Conference on Machine Learning, pages 834–843, 2017.

## A Experimental protocol and evaluation

## A.1 Domains, partitions, and target-data budgets

Each polymer is treated as a distinct manufacturing domain. The dataset contains 145 experiments each for ABS, PETG, PLA, and recycled PLA (rPLA), and 149 experiments for TPU. Before any transfer experiment, every material is divided into mutually exclusive training (50%), validation (20%), and test (30%) partitions. Source experts and target-side candidates are fitted using training data; validation data are used for model selection, policy rewards, and the deployment gate; the test partition is accessed only after all model and policy decisions are fixed.

For target material t, the fitting subset at budget b is

$$
\begin{array} { r l r } { \mathcal { A } _ { t } ( b ) \subset \mathcal { S } _ { \mathrm { t r a i n } } ^ { ( t ) } , } & { } & { \mathcal { B } = \{ 0 . 0 5 , 0 . 1 0 , 0 . 2 0 , 0 . 4 0 , 0 . 6 0 , 0 . 8 0 , 1 . 0 0 \} . } \end{array}\tag{3}
$$

Only $\mathcal { A } _ { t } ( b )$ is used to fit target-dependent candidates. The full 20% validation partition is retained at every budget and is not counted in b. Consequently, the experiment measures scarcity of target fitting data, rather than scarcity of all labelled target observations. Table 2 makes this distinction explicit.

Table 2: Approximate target fitting-set sizes. Ranges reflect the difference between the 145-sample and 149-sample material domains and integer rounding.
<table><tr><td>Budget b</td><td>0.05</td><td>0.10</td><td>0.20 0.40</td><td>0.60</td><td>0.80 1.00</td></tr><tr><td>Target fitting observations Fixed validation observations</td><td>4 7-8</td><td>14-15 approximately 29–30 at every budget</td><td>29-30</td><td>43-45 58-60</td><td>72-75</td></tr></table>

Eleven repetitions are used at each budget, giving 385 held-out cases. $\mathrm { A t } \ b \ < \ 1$ , the repetitions differ in which training observations form $\mathcal { A } _ { t } \bar { ( } b ) . \dot { \mathrm { A t } } b = 1 , \mathcal { A } _ { t } ( b )$ is the full training partition, so the eleven repetitions share identical fitting data and differ only in the random seeds of the stochastic tree learners (bootstrap and feature subsampling), whose effect propagates through the candidate predictions to the policy state, action, and gate. Variation at $b = 1$ therefore reflects model randomness rather than data sampling and is correspondingly small: the within-material standard deviation of the proposed method across repetitions is 0.08–0.24 g per part at $b = 1$ , against 0.18–2.0 g at $b = 0 . 0 5$ . In the leave-one-material-out evaluation, four materials generate policy-training cases and the fifth material is held out as an unseen target. Each case is defined by a target material, a fitting-data budget, and a repetition. Reported aggregate metrics combine target materials, budgets, and repetitions.

## A.2 Fold-specific preprocessing and material distance

Material distance is recalculated within every outer fold. For material descriptor $f$ in fold k,

$$
\widetilde { x } _ { f } ^ { k } = \frac { x _ { f } - \mu _ { f , k } ^ { \mathrm { m e t a } } } { \sigma _ { f , k } ^ { \mathrm { m e t a } } } ,\tag{4}
$$

where $\mu _ { f , k } ^ { \mathrm { m e t a } }$ and $\sigma _ { f , k } ^ { \mathrm { m e t a } }$ are fitted using only the policy-training materials; zero-variance features receive unit scale. The dimensionless material distance is

$$
d _ { \mathrm { m a t } } ^ { ( k ) } ( a , b ) = \frac { 1 } { | F _ { \mathrm { m a t } } | } \sum _ { f \in F _ { \mathrm { m a t } } } W _ { 1 } \Big ( \widetilde { P } _ { a , k } ^ { ( f ) } , \widetilde { P } _ { b , k } ^ { ( f ) } \Big ) .\tag{5}
$$

Material descriptors are constant within a material domain, so the feature-wise Wasserstein distance reduces to the absolute difference between standardized descriptor values. The same fold-specific distance is used in the policy state, the similarity-weighted baseline, and nearest-source pretrain– finetune. State features are transformed using Yeo–Johnson and standardization operations fitted only on the policy-training cases of the outer fold.

## B Candidate models and comparison methods

## B.1 Source experts and candidate prediction bank

The source-expert model family is

M = {Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost}

(6)

For each source s, the expert is the model with the lowest source-validation RMSE:

$$
m _ { s } ^ { \star } = \arg \operatorname* { m i n } _ { m \in \mathcal { M } _ { \mathrm { t r e e } } } \mathrm { R M S E } \Big ( y _ { \mathrm { v a l } } ^ { ( s ) } , \widehat { m } _ { s , m } ( X _ { \mathrm { v a l } } ^ { ( s ) } ) \Big ) , \qquad f _ { s } ( x ) = \widehat { m } _ { s , m _ { s } ^ { \star } } ( x ) .\tag{7}
$$

Source experts use process-control and toolpath predictors. Material properties are domain constants and are therefore used as domain descriptors, rather than as sample-varying inputs to a withinmaterial source expert.

For target t and budget b, the candidate bank $\mathcal { C } _ { t } ( b ) = \{ g _ { 1 } , . . . , g _ { C } \}$ contains: (i) each single-source expert; (ii) supervised models retrained on selected source subsets of size at least two together with $A _ { t } ( b ) ;$ (iii) target-only scratch; and (iv) all sources plus target. Test predictions from these candidates are stored but are not used during candidate construction, policy training, or deployment selection.

## B.2 Baseline definitions

Table 3 defines every comparison method. Each method uses the fitting and selection data specified below; no method uses the held-out test partition for fitting or selection.

Table 3: Evaluated baselines and the aspect of transfer tested by each method.
<table><tr><td>Method</td><td>Transfer basis</td><td>Rule</td><td>Purpose</td></tr><tr><td>Uniform average</td><td>Prediction</td><td>Equal-weight average of all source experts</td><td>Tests whether naive source ensem- bling is sufficient.</td></tr><tr><td>Similarity-weighted</td><td>Prediction</td><td>Inverse weighting by  $d _ { \mathrm { m a t } } ^ { ( k ) }$ </td><td>Tests whether material proximity alone identifies useful sources.</td></tr><tr><td>Best single source</td><td>Prediction</td><td>Source with lowest error on  $\boldsymbol { A } _ { t } ( \boldsymbol { b } )$ </td><td>Hard source selection using limited tar- get fitting evidence.</td></tr><tr><td>Adaptation simplex</td><td>Prediction</td><td>Nonnegative convex stacking of source predictions fitted on  $\boldsymbol { A } _ { t } ( \boldsymbol { b } )$ </td><td>Strong source-only prediction- combination baseline.</td></tr><tr><td>Target-only scratch</td><td>Data</td><td>Best tree model fitted only on  $\boldsymbol { A } _ { t } ( \boldsymbol { b } )$ </td><td>Non-transfer reference.</td></tr><tr><td>All sources + target</td><td>Data</td><td>Best tree model fitted on all source training data plus  $\boldsymbol { A } _ { t } ( \boldsymbol { b } )$ </td><td>Indiscriminate supervised transfer ref- erence.</td></tr><tr><td>Nearest-source pretrain- Parameters finetune</td><td></td><td>Pretrain an MLP on the nearest source and fine-tune Parameter-transfer baseline using the on At (b)</td><td>same distance as the policy.</td></tr><tr><td>Pooled pretrain-finetune</td><td>Parameters</td><td>Pretrain the same MLP on pooled sources and fine- Parameter-transfer baseline using all tune on  $\boldsymbol { A } _ { t } ( \boldsymbol { b } )$ </td><td>source data.</td></tr><tr><td>Behaviour cloning + gate</td><td>Model selection</td><td>Imitate validation-optimal candidate weights, then Isolates supervised policy initializa- apply the gate</td><td>tion without actor-critic refinement.</td></tr><tr><td>Raw actor-critic</td><td>Model selection</td><td>Refine candidate weights using the one-step reward, Isolates reinforcement-learning refine- without the gate</td><td>ment.</td></tr><tr><td>Actor-critic + gate</td><td>back</td><td>Selection + fall- Raw policy followed by comparison with the Proposed deployed method. stronger supervised reference</td><td></td></tr></table>

The four prediction-level source baselines can be written

$$
{ \widehat { y } } ( X ) = \sum _ { s \neq t } w _ { s } f _ { s } ( X ) , \qquad w _ { s } \geq 0 , \quad \sum _ { s \neq t } w _ { s } = 1 .\tag{8}
$$

Uniform averaging uses $w _ { s } = 1 / S$ , where $S$ is the number of source domains. Similarity weighting uses

$$
w _ { s } = \frac { \operatorname* { m a x } \{ d _ { \mathrm { m a t } } ^ { ( k ) } ( t , s ) , 1 0 ^ { - 8 } \} ^ { - 1 } } { \sum _ { j \neq t } \operatorname* { m a x } \{ d _ { \mathrm { m a t } } ^ { ( k ) } ( t , j ) , 1 0 ^ { - 8 } \} ^ { - 1 } } .\tag{9}
$$

Best-single-source chooses the source with the lowest RMSE on the target fitting subset. Adaptation simplex solves

$$
\boldsymbol { w } ^ { \star } = \arg \operatorname* { m i n } _ { \boldsymbol { w } \in \Delta ^ { s - 1 } } \left\| \boldsymbol { y } _ { A } - \widehat { Y } _ { A } \boldsymbol { w } \right\| _ { 2 } ^ { 2 } ,\tag{10}
$$

where $\widehat { Y } _ { A }$ stacks source-expert predictions on $\mathcal { A } _ { t } ( b )$ . The neural pretrain–finetune baselines use the full material, process, and toolpath vector after fold-specific Yeo–Johnson preprocessing, Huber loss, early stopping on target validation data, and gradient clipping.

## C Material-agnostic transfer policy

## C.1 State, action, and behaviour-cloning initialization

The state contains: (i) the target fitting-data budget; (ii) means and standard deviations of the target fitting subset over the material, process, and toolpath feature groups; (iii) fold-specific standardized source–target material distances; (iv) source-expert errors on the fitting subset; (v) disagreement among source-expert predictions; and (vi) validation performance of the target-only and all-sourcesplus-target references.

The action is a vector $a \in \Delta ^ { C - 1 }$ of nonnegative candidate weights, producing

$$
g _ { a } ( x ) = \sum _ { j = 1 } ^ { C } a _ { j } g _ { j } ( x ) .\tag{11}
$$

An action mask gives unavailable candidates exactly zero probability and zero mixture weight. For available-candidate set $\nu _ { t }$

$$
p _ { \theta , j } ( s ) = \frac { \mathbb { I } ( j \in \mathcal { V } _ { t } ) \exp ( z _ { \theta , j } ( s ) ) } { \sum _ { \ell \in \mathcal { V } _ { t } } \exp ( z _ { \theta , \ell } ( s ) ) } .\tag{12}
$$

The behaviour-cloning target is the validation-optimal convex mixture

$$
a ^ { \star } = \arg \operatorname* { m i n } _ { a \in \Delta ^ { C - 1 } , a _ { j } = 0 } \forall j \notin \mathcal { V } _ { t }  \left\| y _ { \mathrm { v a l } } - P _ { \mathrm { v a l } } a \right\| _ { 2 } ^ { 2 } ,\tag{13}
$$

and the actor is initialized with

$$
\mathcal { L } _ { \mathrm { B C } } = - \sum _ { j \in \mathcal { V } _ { t } } a _ { j } ^ { \star } \log p _ { \theta , j } ( s ) .\tag{14}
$$

Here s is the state of a target–budget case, $p _ { \theta , j } ( s )$ are the masked, normalized candidate weights of Eq. 12, and $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { a } \mid \boldsymbol { s } )$ in Appendix C.2 denotes the Dirichlet density over the continuous weight vector $^ { a ; }$ both are computed from the same actor network. Because each case is a single decision, no time index is needed.

## C.2 One-step actor–critic refinement

The actor and critic are two-hidden-layer multilayer perceptrons. Positive masked actor outputs parameterize a Dirichlet distribution over candidate weights, and a non-reparameterized action $a \sim$ Dirichlet $\left( \alpha _ { \theta } ( s ) \right)$ is sampled during refinement. The critic predicts the scalar state value. Because every target–budget case contains one state, one model-selection action, and one validation reward, the method is contextual one-step reinforcement learning rather than sequential process control.

For validation prediction ${ \widehat { y } } ,$ normalized RMSE is

$$
\mathrm { N R M S E } ( \widehat { y } ) = \frac { \sqrt { n _ { \mathrm { v a l } } ^ { - 1 } \sum _ { i } ( y _ { i } - \widehat { y } _ { i } ) ^ { 2 } } } { \operatorname* { m a x } _ { i } y _ { i } - \operatorname* { m i n } _ { i } y _ { i } } .\tag{15}
$$

Let $g _ { \mathrm { r e f } }$ be the better of target-only scratch and all sources plus target on validation data, and let $g _ { \mathrm { b e s t } }$ be the validation-optimal available mixture (best expert weights). The reward is (explained in Table 4):

$$
\begin{array} { r l } & { r = \mathrm { - N R M S E _ { p o l } + 1 . 5 [ N R M S E _ { r e f } - N R M S E _ { p o l } ] _ { + } } } \\ & { \phantom { r } - 3 . 0 [ \mathrm { N R M S E _ { p o l } - N R M S E _ { r e f } } ] _ { + } - 0 . 7 5 [ \mathrm { N R M S E _ { p o l } - N R M S E _ { b e s t } } ] _ { + } , } \end{array}\tag{16}
$$

where $[ z ] _ { + } = \operatorname* { m a x } ( z , 0 )$ . The actor and critic objectives are

$$
\mathcal { L } _ { \mathrm { a c t o r } } = - \log \pi _ { \boldsymbol { \theta } } ( \boldsymbol { a } \mid \boldsymbol { s } ) \hat { A } - \beta \mathcal { H } ( \pi _ { \boldsymbol { \theta } } ( \cdot \mid \boldsymbol { s } ) ) + \lambda _ { \mathrm { B C } } \mathcal { L } _ { \mathrm { B C } } ,\tag{17}
$$

$$
{ \mathcal { L } } _ { \mathrm { c r i t i c } } = ( V _ { \phi } ( s ) - r ) ^ { 2 } ,\tag{18}
$$

with one-step advantage $\hat { A } = r - V _ { \phi } ( s )$ . Hyperparameters are listed in Table 7.

Table 4: Reward components and their purposes.
<table><tr><td>Reward Component</td><td>Formulation</td><td>Purpose</td></tr><tr><td>Negative validation</td><td> $- \mathrm { N R M S E } _ { p o l }$ </td><td>Penalizes prediction error</td></tr><tr><td>Improvement bonus</td><td> $+ 1 . 5 [ \mathrm { N R M S E } _ { r e f } \mathrm { ~ - ~ } \mathrm { N R M S E } _ { p o l } ] _ { + }$ </td><td>Rewards improvement over reference</td></tr><tr><td>Underperformance penalty</td><td> $- 3 . 0 \dot { [ } \mathrm { N R M S E } _ { p o l } - \mathrm { N R M S E } _ { r e f } \dot { ] } +$ </td><td>Penalizes worse performance than reference</td></tr><tr><td>Expert-gap penalty</td><td> $- 0 . 7 \dot { 5 } [ \mathrm { N R M S E } _ { p o l } - \mathrm { N R M S E } _ { b e s t } ] _ { + }$ </td><td>Penalizes worse performance than the validation-optimal mixture</td></tr></table>

## C.3 Validation gate

The deployed supervised fallback $g _ { \mathrm { f b } }$ is the better of target-only scratch and all sources plus target on the target validation split. For policy predictor $g _ { \mathrm { p o l } }$ ,

$$
g _ { \mathrm { f i n a l } } = \left\{ \begin{array} { l l } { g _ { \mathrm { p o l } } , } & { \mathrm { N R M S E _ { \mathrm { p o l , v a l } } } \leq \mathrm { N R M S E _ { \mathrm { f b , v a l } } } - \delta , } \\ { g _ { \mathrm { f b } } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{19}
$$

The experiments use $\delta = 0$ . The raw actor–critic variant omits Eq. 19; the behaviour-cloning variant omits actor–critic refinement but retains the same gate.

## D Extended comparison and diagnostic results

Table 5: Overall performance across held-out target materials, fitting-data budgets, and subset draws. RMSE is in $\mathrm { k g C O _ { 2 } e / p a r t } ;$ values are mean ± standard deviation.
<table><tr><td>Method</td><td> $\mathrm { V a l } . R ^ { 2 }$ </td><td>Val. RMSE</td><td> $\mathrm { T e s t } R ^ { 2 }$ </td><td>Test RMSE</td></tr><tr><td>Actor-critic + gate</td><td> $0 . 8 5 8 0 \pm 0 . 1 4 9 6$ </td><td> $0 . 0 0 4 4 \pm 0 . 0 0 1 5$ </td><td> $0 . 7 7 5 7 \pm 0 . 2 1 5 1$ </td><td> $0 . 0 0 5 2 \pm 0 . 0 0 2 1$ </td></tr><tr><td>Behaviour cloning + gate</td><td> $0 . 8 3 0 3 \pm 0 . 2 7 7 7$ </td><td> $0 . 0 0 4 7 \pm 0 . 0 0 2 0$ </td><td> $0 . 7 2 6 4 \pm 0 . 2 1 3 0$ </td><td> $0 . 0 0 5 3 \pm 0 . 0 0 2 2$ </td></tr><tr><td>Raw actor-critic</td><td> $0 . 8 0 6 4 \pm 0 . 2 5 3 5$ </td><td> $0 . 0 0 5 3 \pm 0 . 0 0 2 0$ </td><td> $0 . 7 1 9 5 \pm 0 . 1 8 9 3$ </td><td> $0 . 0 0 5 6 \pm 0 . 0 0 2 2$ </td></tr><tr><td>All sources + target</td><td> $0 . 8 0 9 8 \pm 0 . 2 6 7 9$ </td><td> $0 . 0 0 5 0 \pm 0 . 0 0 2 0$ </td><td> $0 . 6 6 3 3 \pm 0 . 2 6 5 4$ </td><td>0.0058 ± 0.0023</td></tr><tr><td>Target only</td><td> $0 . 6 7 0 3 \pm 0 . 3 6 8 4$ </td><td> $0 . 0 0 6 6 \pm 0 . 0 0 3 3$ </td><td> $0 . 4 9 2 9 \pm 0 . 6 6 0 9$ </td><td> $0 . 0 0 6 5 \pm 0 . 0 0 3 3$ </td></tr><tr><td>Adaptation simplex</td><td>0.4851 ± 0.6410</td><td>0.0077 ± 0.0036</td><td> $0 . 5 0 0 5 \pm 0 . 3 3 1 1$ </td><td> $0 . 0 0 7 7 \pm 0 . 0 0 3 9$ </td></tr><tr><td>Best single source</td><td> $0 . 4 6 4 8 \pm 0 . 6 2 9 2$ </td><td> $0 . 0 0 8 2 \pm 0 . 0 0 3 6$ </td><td> $0 . 4 6 7 8 \pm 0 . 2 9 8 8$ </td><td> $0 . 0 0 8 2 \pm 0 . 0 0 3 7$ </td></tr><tr><td>Pooled pretrain-finetune</td><td> $0 . 5 1 7 0 \pm 0 . 5 6 9 7$ </td><td> $0 . 0 0 8 5 \pm 0 . 0 0 4 0$ </td><td> $0 . 0 6 9 6 \pm 1 . 1 6 9 4$ </td><td> $0 . 0 0 9 4 \pm 0 . 0 0 4 9$ </td></tr><tr><td>Similarity-weighted</td><td> $- 0 . 2 3 4 0 \pm 1 . 3 6 3 8$ </td><td> $0 . 0 1 1 9 \pm 0 . 0 0 4 9$ </td><td> $0 . 0 9 2 1 \pm 0 . 6 1 3 6$ </td><td> $0 . 0 1 0 2 \pm 0 . 0 0 5 5$ </td></tr><tr><td>Uniform average</td><td> $- 0 . 2 0 8 6 \pm 1 . 4 2 1 6$ </td><td> $0 . 0 1 1 4 \pm 0 . 0 0 5 3$ </td><td> $- 0 . 0 0 9 5 \pm 0 . 7 2 4 6$ </td><td> $0 . 0 1 0 3 \pm 0 . 0 0 5 9$ </td></tr><tr><td>Nearest-source pretrain-finetune</td><td> $- 3 5 . 9 2 3 4 \pm 1 1 6 . 1 2 9 9$ </td><td> $0 . 0 5 3 0 \pm 0 . 0 7 8 8$ </td><td> $- 6 4 . 6 4 7 8 \pm 2 2 3 . 8 5 2 7$ </td><td> $0 . 0 5 2 8 \pm 0 . 0 7 6 9$ </td></tr></table>

The full benchmark places the policy against prediction-, data-, and parameter-transfer alternatives. Simple averaging and similarity weighting do not resolve incompatible sources, while the learned variants benefit from case-specific candidate evidence. The behaviour-cloning and raw actor–critic rows act as complementary ablations: together with the gated result, they show how validationoriented initialization, reward-based refinement, and supervised fallback contribute to the deployed method.

## D.1 Budget-dependent performance

![](images/a195250c9d9411eb00ac8d32e398c086522248e2a931978945298cd57c060e26.jpg)

![](images/1bcc53376587fc5f6f489a33b086aed7cc609424b34391958574791c8e3ca3e0.jpg)

![](images/5dc76706eeea0a2edb7f7e35077046338bcdd454602dd6ced2e0d3b37fa4afa2.jpg)

![](images/34b327833880c0f43efb6becdae0e77d3e3d2bf05e0d5b385d63dae104dbd02c.jpg)

![](images/eb569243820b84aeb9c7b03255712d3250a280e0d2b6acf309f1d2dfb2868877.jpg)  
Figure 3: Test RMSE versus target fitting budget for (a) PETG, (b) ABS, (c) PLA, (d) rPLA, and (e) TPU. The budget b is the fraction of the target training partition used for fitting; the separate validation partition remains fixed. Curves compare the complete policy, its ablations, target-only fitting, and all-sources-plus-target pooling. RMSE is expressed in $\mathrm { k g C O _ { 2 } e / p a r t }$ . Lines show mean test $\mathrm { R M S E } ;$ shaded bands show standard deviation.

No method dominates for every target and budget. Transfer-aware methods are most useful when target fitting data are scarce, but target-only fitting can become preferable as the target relationship becomes estimable. PETG shows a large low-budget gap: target-only RMSE is approximately 0.010 at $b = 0 . 0 5$ , while transfer-aware methods are near 0.006. ABS methods converge at larger budgets, consistent with weak source-expert transfer into ABS. TPU reverses from poor target-only performance at 5–10% to target-only superiority after approximately 40%. These results motivate conditioning the decision on both domain evidence and target fitting-data budget.

![](images/c08b426b29e2ef8ed813e4db74af6cd28ee273317430ccc6b4906ef5b89f046d.jpg)

![](images/064c78fbdeaf01a20b80ebe27976dab1df44759f9692300a0fadb3e32b5f90b7.jpg)  
Figure 4: Deployment evidence: (a) negative-transfer rate relative to target-only fitting; and (b) frequency with which the validation gate replaces the actor–critic prediction with the stronger supervised model.

## D.2 Distance, negative transfer, and failure cases

![](images/8ab899ae765d664118bb635565ce501b024a1562fdb715c9f71bd14c86639dad.jpg)  
Figure 5: Source-expert test RMSE against standardized material distance for representative source– target pairs; statistics over all 20 directed pairs are given in Appendix E.

Material proximity is not a reliable stand-alone transfer rule. ABS–PETG has a standardized distance of 0.810 but poor transfer error near 0.019–0.024 kgCO e/part, whereas PETG–TPU has a larger distance of 1.803 and transfers well in both directions. Distance is therefore retained as one state component rather than used as a deterministic decision.

The negative-transfer rate is approximately 0.17 for actor–critic plus gate, 0.28 for behaviour cloning plus gate, 0.33 for raw actor–critic, 0.49 for all sources plus target, 0.68 for pooled pretrain–finetune, and 0.98 for nearest-source pretrain–finetune. Thus, the gate reduces the raw policy’s observed harmful-transfer rate by about 48%. Fallback frequency is itself a result: ABS increasingly uses a supervised reference and reaches a fallback rate of 1.0 by b = 0.80; TPU uses fallback in all cases at every budget. More target data can therefore make transfer less appropriate when the target-specific supervised relationship becomes stronger.

Residual error is target-specific and often coincides with disagreement among source experts. PLA shows the clearest sensitivity: mean absolute error rises from approximately $\mathrm { { \bar { 2 } . 1 \times 1 0 ^ { - 3 } } }$ in the first disagreement quantile to $8 . { \dot { 4 } } \times 1 0 ^ { - 3 }$ in the third. ABS also deteriorates in the highest quantile.

![](images/1c487b36a7547d7a9f1257c86c0954ff0cbcd8dec00915bcf261a2f73eda8ca1.jpg)

![](images/37c334452a9e8d9d2b8939509eba38c1d18c31a0653af6eb8ca24e39a201164e.jpg)  
Figure 6: Failure analysis using (a) absolute prediction error and (b) mean absolute error by sourceexpert disagreement group.

This pattern supports the inclusion of expert disagreement in the policy state and identifies highdisagreement cases as priorities for additional target measurements.

## D.3 Manufacturing interpretation

For implementation, a manufacturer needs a common predictor schema across historical and new operations, a small target fitting set, and a separate target validation set. The candidate bank then turns historical operations into auditable alternatives: individual experts expose directional compatibility, subset candidates capture complementary sources, and the target-only and pooled candidates anchor deployment. The selected predictor supplies an inventory-linked response before a full characterization campaign is complete, while disagreement and fallback frequency indicate where additional target measurements are most valuable.

The controlled cross-material setting holds geometry, printer, and response construction consistent, so the observed differences can be attributed to the transfer decision rather than changes in system boundaries. The demonstrated benefit is improved prediction of an operational-plus-material carbon indicator; realized emissions reductions depend on using those predictions to choose lower-footprint, technically feasible alternatives.

## E Decision formulation, comparison with direct learning, and policy specification

## E.1 Prediction versus decision

Direct supervised learning fits one predictor by minimizing prediction error on the data it is given; whether reusing source data helps or harms a new target becomes visible only when alternatives are compared. The proposed method learns that comparison as a decision. Its action selects and weights candidates, including target-only learning; its reward (Eq. 16) scores the resulting mixture against the stronger supervised reference; and the gate (Eq. 19) deploys the policy only when validation evidence favours it, using a validation-based comparison with a supervised fallback [12, 13].

Validation-optimal mixture weights provide behaviour-cloning targets on policy-training materials, followed by reward-based refinement; the trained policy is then evaluated on a held-out material. Each case has one state, one action, and one reward, so the problem is a contextual bandit, and one-step actor–critic, i.e. REINFORCE with a learned baseline [19], is the standard estimator. The score-function estimator requires only reward evaluations, which keeps the candidate bank, reward, and gate interchangeable. The Dirichlet distribution is the natural policy class on the simplex of mixture weights, as Beta policies are for bounded actions [20].

Reward weights. The weights in Eq. 16 encode the deployment preference. Underperforming the trusted supervised reference costs twice the credit given for improving on it (3.0 versus 1.5), and distance from the validation-optimal mixture is a weaker shaping term (0.75). They were set from this preference rather than tuned on held-out data, so the test results carry no weight-selection bias. The coefficients change the reward’s relative emphasis on improvement and underperformance during policy training. Other processes, response scales, or risk tolerances may call for recalibration, ideally on validation data, and a systematic study of this choice is left to future work.

Direct optimization. When labelled target validation data exist, mixture weights can also be optimized directly on the target validation set (Eq. 13). We use this solution as the imitation target rather than as a deployed predictor. The learned policy instead provides one decision rule that is shared across materials, evaluated on materials it has not seen, and combined with a gate that guards against negative transfer. Direct differentiation through the mixture reward is also a possible approach to training a shared policy; the present experiments do not establish an advantage of the score-function estimator over this alternative. Comparisons with this approach and with per-target direct optimization are left to future work.

## E.2 Comparison with direct learning

Table 6 adds 95% intervals and paired tests over the 385 matched held-out cases. All cases of a material share its test set, trained policy, and validation set; within a material–budget combination they differ only in the repetition. Case-level tests therefore treat repetitions as replicates: they assess whether a difference is consistent across repetitions for the evaluated materials. Because the repetitions at $b = 1$ share their fitting data, their cases are near-replicates; excluding them, the reduction in negative transfer relative to behaviour cloning with the gate remains consistent (53 versus 23 discordant cases, McNemar $p = 8 \times 1 0 ^ { - 4 } )$ . Generalization to new materials is assessed by a hierarchical bootstrap that resamples materials and then cases; with five materials, these intervals are approximate. The proposed method lowers RMSE by 0.64 g per part (95% interval 0.36–0.87 g) relative to all sources plus target and by 1.31 g (0.63–1.90 g) relative to target-only fitting. It lowers the negative-transfer rate by 31 percentage points (95% interval: 18–44 percentage points) relative to pooling. The hierarchical bootstrap intervals support an average improvement across the evaluated materials, although inference beyond these materials remains limited by the five-material sample. Relative to behaviour cloning with the gate, the reduction in negative transfer is consistent across repetitions (Table 6) and concentrated in TPU and ABS, the targets with the weakest source transfer; for PLA and rPLA, behaviour cloning with the gate has slightly lower rates, so this comparison holds across repetitions rather than across materials.

The decision is also robust across conditions. In each of the 35 material–budget cells, the proposed method was compared with the better of the two fixed rules, target-only fitting and pooling, chosen in hindsight from test results. The proposed method is within 5% of, or better than, that hindsight choice in 29 cells, never more than 15% worse, and 3.2% better on average. Behaviour cloning with the gate and raw actor–critic are up to 86% and 134% worse, respectively. The spread of the average row of Table 1 reflects the conditions tested more than instability. A one-way analysis of variance of the 385 per-case test RMSE values of the proposed method, grouped by the 35 material–budget combinations, attributes 83% of the variance to differences between combinations $( \eta ^ { 2 } ;$ material alone 27%, budget alone 34%, the remainder their interaction) and 17% to variation across repetitions within a combination. Per-part responses average 0.051 $\mathrm { k g C O _ { 2 } e }$ across the 729 prints, so the overall RMSE of the proposed method is about 10% of a typical part’s footprint, against 13% for target-only fitting.

Table 6: Held-out results with 95% intervals (385 cases; intervals resample cases within material– budget cells) and paired tests against the proposed method (Wilcoxon signed-rank for RMSE, Mc-Nemar for negative transfer). Intervals and p-values are case-level and treat repetitions as replicates for the evaluated materials.
<table><tr><td>Method</td><td>RMSE  $\mathrm { ( k g C O _ { 2 } e / p a r t ) }$ </td><td>Neg. transfer (%)</td><td>p (RMSE)</td><td>p (neg. transfer)</td></tr><tr><td>Actor-critic + gate (proposed)</td><td>0.0052 [0.0051, 0.0053]</td><td>17 [14, 20]</td><td></td><td></td></tr><tr><td>Behaviour cloning + gate</td><td>0.0053 [0.0053, 0.0054]</td><td>28 [25, 30]</td><td>0.13</td><td> $4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Raw actor-critic</td><td>0.0056 [0.0055, 0.0057]</td><td>33 [30, 36]</td><td> $1 \times 1 0 ^ { - 9 }$ </td><td> $2 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>All sources + target</td><td>0.0058 [0.0057, 0.0059]</td><td>49 [45, 52]</td><td> $4 \times 1 0 ^ { - 3 0 }$ </td><td> $5 \times 1 0 ^ { - 3 5 }$ </td></tr><tr><td>Target only</td><td>0.0065 [0.0063, 0.0067]</td><td></td><td> $3 \times 1 0 ^ { - 3 1 }$ </td><td></td></tr></table>

## E.3 Policy specification

For a held-out target, the bank contains 17 candidates: four single-source experts, eleven source subsets, target-only scratch, and all sources plus target. For a policy-training case, candidates involving the pseudo-target are unavailable, leaving nine. The state has 81 features: the budget and fitting-set size; means and standard deviations of the 22 predictors over the fitting subset; per-source material distances, fitting-subset errors, and availability masks; summaries of expert disagreement; validation errors of the two supervised references; and aggregate summaries of these quantities. Policy-training and held-out cases use separate fitting-subset draws. Across all 20 directed source–target pairs, material distance and source-expert error are not significantly rank-correlated (Spearman $\rho = 0 . 2 8$ $p = 0 . 2 4$ on target validation data; $\rho = 0 . 2 1 , p = 0 . 3 8$ on test data), and validation and test errors agree closely $( \rho = 0 . 8 7 )$ . The distance finding therefore does not depend on test data.

Table 7: Policy hyperparameters.
<table><tr><td>Actor and critic</td><td>2 × 32 MLP, dropout 0.1</td><td>Optimizer</td><td>Adam, lr 5 × 10−4, weight decay 10−4</td></tr><tr><td>Behaviour-cloning epochs</td><td>500</td><td>Refinement epochs</td><td>500</td></tr><tr><td>Entropy weight β</td><td>0.1</td><td>Critic loss weight</td><td>0.5</td></tr><tr><td>λBC</td><td>0.3</td><td>Gradient-norm clip</td><td>1.0</td></tr><tr><td>Dirichlet concentration floor</td><td>0.1</td><td>Gate margin δ</td><td>0</td></tr><tr><td>Reward weights (Eq. 16)</td><td colspan="3">1.5, 3.0, 0.75</td></tr></table>