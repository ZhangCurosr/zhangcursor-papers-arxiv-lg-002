# Towards Reasonable Molecular Structure Elucidation from Infrared Spectroscopy with Chemical Feedback

Yusen Tan<sup>1</sup>, Hongyu Zhan<sup>1</sup>, Hai-tao Yu<sup>1</sup>, Changxi Chi<sup>2</sup>, Wenjie Du<sup>3</sup>, Jun Xia<sup>1,4,†</sup>

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>2</sup>Westlake University

<sup>3</sup>University of Science and Technology of China <sup>4</sup>The Hong Kong University of Science and Technology {ytan277, hzhan701, hyu382}@connect.hkust-gz.edu.cn chichangxi@westlake.edu.cn duwenjie@mail.ustc.edu.cn junxia@hkust-gz.edu.cn

## Abstract

Infrared (IR) spectra provide characteristic signals of molecular structure, which are often interpreted by experts via functional-group identification or library matching, making the process time-consuming and ambiguous. Recent machine learning methods have made progress in molecular structure elucidation using molecular formulas and IR spectra. However, these models often infer unreasonable candidate molecular structures, including top-ranked predictions. More specifically, the molecular formula implied by a candidate structure often fails to match the input molecular formula, and the candidate’s theoretical IR spectrum is often inconsistent with the observed IR spectrum. To address these issues, we propose Formula- and IR-Matched Preference Optimization (FIRMPO), a general and plug-and-play chemical feedback-driven preference optimization framework for molecular structure elucidation. FIRMPO incorporates chemical feedback as preference signals based on exact molecular formula matching and IR spectral consistency to guide reasonable structure predictions. Unlike generic preference optimization methods, FIRMPO is tailored to molecular structure elucidation while remaining modelagnostic, enabling it to be readily integrated with different structure prediction models in this class. This encourages models to prioritize structures that satisfy the chemical feedback, leading to a substantial improvement in the accuracy of top-ranked predictions. Extensive experiments on three widely used IR datasets show that FIRMPO significantly improves molecular structure elucidation accuracy over existing baselines.

## 1 Introduction

Analytical spectroscopic techniques, including Infrared (IR) spectroscopy, Mass Spectrometry (MS), and Nuclear Magnetic Resonance (NMR), are widely used in chemical analysis [20, 1]. By providing complementary information about molecular composition, functional groups, and atomic connectivity, they play a central role in molecular structure elucidation [20, 17]. Although IR spectroscopy typically provides less chemical detail than MS and NMR with respect to molecular weight, elemental composition, and stereochemistry, its low cost, minimal sample preparation, and rapid data acquisition make it a practical tool for preliminary characterization and high-throughput analysis [21]. However, extracting complete structural information from IR spectra remains challenging due to complex spectral patterns [22, 6]. In most cases, interpretation of IR spectra still relies on expert-driven identification of characteristic functional groups or matching against reference libraries. This process is time-consuming and often ambiguous [22, 6, 23].

![](images/6e6b416fa7b3c0e300c9bafd60ed2ceb8857f791335dc5326ccb46a37b1bccc0.jpg)  
Figure 1: Illustration of a ranking failure in molecular structure elucidation from molecular formulas and IR spectra: although the correct structure appears at rank 8, higher-ranked predictions violate chemical feedback, showing (i) molecular formula mismatch or (ii) inconsistency between the theoretical and observed IR spectra.

Recently, machine learning methods have been explored to directly infer molecular structures from molecular formulas and IR spectra. For example, IRtoMol employs a Transformer-based architecture conditioned on molecular formulas and IR spectra to enable end-to-end molecular structure elucidation [2]. Subsequent studies have further advanced IR-based molecular structure prediction by introducing patch-based self-attention spectrum embeddings and improved Transformer architectures with refined spectral representations and decoding strategies[25, 4]. Despite this progress, current models can still suffer from ranking failures: even when a plausible molecular structure is included among the generated candidates, an unreasonable candidate may be assigned the top-ranked position, as illustrated in Figure 1. This leads to a pronounced gap between top-1 and top-k accuracy. Concretely, a top-ranked candidate may imply a molecular formula that differs from the input molecular formula, and its theoretical IR spectrum may be inconsistent with the observed spectrum.

To address these issues, we propose Formula- and IR-Matched Preference Optimization (FIRMPO), a general and plug-and-play chemical feedback-driven preference optimization framework for molecular structure elucidation. FIRMPO is model-agnostic and can be readily integrated with different IR-based molecular structure prediction models. The core idea is to incorporate chemical feedback, namely exact molecular formula matching and IR spectral consistency, as preference signals to re-rank candidate molecular structures into a chemically preferred order, thereby constructing a preference dataset. Building on this preference dataset, FIRMPO encourages the model to prioritize candidate structures that exactly match the input molecular formula and exhibit higher IR spectral consistency with the observed spectrum, leading to improved accuracy of top-ranked molecular structure predictions.

Overall, our main contributions are summarized as follows:

1) We introduce Formula- and IR-Matched Preference Optimization (FIRMPO), a general and plug-and-play chemical feedback-driven preference optimization framework for molecular structure elucidation. FIRMPO incorporates exact molecular formula matching and IR spectral consistency as preference signals to guide models toward chemically reasonable structure predictions.

2) FIRMPO is model-agnostic and can be readily integrated with different IR-based molecular structure prediction models. Unlike generic preference optimization methods, FIRMPO is tailored to molecular structure elucidation by introducing chemistry-aware preference construction and two additional weighting terms, which provide fine-grained optimization signals for ranking chemically preferred candidate structures.

3) Extensive experiments on three public IR spectral datasets show that FIRMPO significantly improves molecular structure elucidation accuracy over existing baselines under multiple evaluation metrics, especially for top-ranked predictions.

## 2 Related Work

## 2.1 Molecular Structure Elucidation from Infrared Spectroscopy

IR spectroscopy is widely used for molecular characterization due to its sensitivity to functional groups and vibrational patterns. Traditional IR analysis often relies on expert interpretation or spectral database matching, which can limit throughput and may not generalize across diverse molecular systems [23]. To mitigate these limitations, recent studies have applied machine learning methods to automate the interpretation of IR spectra [10, 7, 18]. Early efforts mainly focused on functional group identification. These methods used convolutional neural networks, random forests, and multilayer perceptrons to extract discriminative spectral features for classification [11, 8]. While such models support preliminary analysis and compound screening, they provide only partial structural information and do not recover molecular connectivity or topology. This has motivated research on complete molecular structure elucidation directly from molecular formulas and IR spectra [24].

Complete molecular structure elucidation from molecular formulas and IR spectra is more challenging than functional group identification because it requires capturing richer chemical information [12]. Recent work has explored end-to-end structural elucidation from IR spectra, often with additional molecular formula information. IRtoMol introduced a Transformer-based approach that conditions SMILES generation on molecular formulas and IR spectra for end-to-end molecular structure elucidation [2]. Subsequent studies have advanced this paradigm by improving spectral representations with patch-based self-attention [25] and refining Transformer architectures with enhanced augmentation and decoding strategies [4]. Despite this progress, existing models can still assign high ranks to chemically unreasonable candidate structures. In particular, the molecular formula implied by a high-ranked candidate may not exactly match the provided molecular formula, and the candidate’s theoretical IR spectrum may be inconsistent with the observed spectrum. Such ranking failures lead to a pronounced gap between top-1 and top-k accuracy, suggesting that chemically plausible structures may be generated but are not consistently prioritized. This motivates us to construct preference signals from chemical feedback, namely molecular formula matching and IR spectral consistency, to guide models toward ranking chemically plausible candidate structures at the top positions.

## 2.2 Preference Optimization (PO)

Preference optimization (PO) is a paradigm for aligning model behavior with preference signals, including human judgments and domain-specific criteria. More broadly, recent alignment research increasingly emphasizes PO as an efficient alternative to reinforcement learning from human feedback, where learning signals come from comparative feedback such as preferences over multiple candidate outputs rather than explicit target texts [16, 19]. In practice, PO methods are often categorized into pointwise, pairwise, and listwise objectives, with recent work placing greater emphasis on the latter two [19, 9, 14, 5]. Pairwise PO learns from comparisons between two candidates under the same context. For instance, Direct Preference Optimization (DPO) optimizes a closed-form, KL-regularized objective over preferred and dispreferred response pairs, enabling stable offline alignment without an explicit reward model [19]. Kahneman–Tversky Optimization (KTO) instead uses pointwise good and bad labels inspired by prospect theory, reducing reliance on strictly constructed pairs [9]. While pairwise PO is simple and effective, it can underutilize supervision available in multi-candidate rankings. Listwise PO addresses this by learning from preferences defined over candidate sets, which naturally matches ranking tasks. Listwise Preference Optimization (LiPO) formulates alignment as learning to rank with listwise supervision over multiple candidates [14]. K-order Ranking Preference Optimization (KPO) further emphasizes top-K ranking consistency and introduces query-adaptive K selection and curriculum learning [5].

Despite their success in large language model alignment, existing PO methods are not readily applicable to molecular structure elucidation. In particular, these PO objectives do not explicitly prioritize candidates that are identical to the ground-truth structure, which is precisely the goal of molecular structure elucidation. Consequently, the correct structure can be ranked below a large candidate set of highly similar structural isomers, limiting improvements in molecular structure elucidation accuracy. This motivates us to develop a preference optimization method tailored to molecular structure elucidation that explicitly promotes ground-truth-matching candidates.

## 2.3 Theoretical Infrared Spectra and Spectral Similarity Metrics

A practical way to assess whether a candidate molecular structure is consistent with the observed IR spectrum is to predict its theoretical IR spectrum from the structure and then measure its similarity to the observed IR spectrum. Chemprop-IR [15] provides a learned forward model that predicts IR spectra from SMILES (Simplified Molecular Input Line Entry System) strings using message passing neural networks.

In this work, we quantify spectral consistency using cosine similarity between the predicted theoretical spectrum and the observed spectrum. Given vectorized spectra $\dot { \mathcal X _ { \mathrm { t h e o r y } } }$ and $\mathcal { X } _ { \mathrm { o b s } }$ , cosine similarity is defined as:

$$
\mathrm { C o s S i m } ( \mathcal { X } _ { \mathrm { t h e o r y } } , \mathcal { X } _ { \mathrm { o b s } } ) : = \frac { \mathcal { X } _ { \mathrm { t h e o r y } } \cdot \mathcal { X } _ { \mathrm { o b s } } } { \left\| \mathcal { X } _ { \mathrm { t h e o r y } } \right\| _ { 2 } \left\| \mathcal { X } _ { \mathrm { o b s } } \right\| _ { 2 } } .\tag{1}
$$

Cosine similarity is scale-invariant and captures consistency in spectral shape, making it suitable for comparing theoretical and observed spectra.

## 3 Methodology

## 3.1 Preliminaries

Let $\boldsymbol { \mathcal { X } } \in \mathbb { R } ^ { L }$ denote an IR spectrum discretized on L wavenumber points, and let $\mathcal { F }$ denote the corresponding molecular formula. Let Y denote the set of molecular structures represented by SMILES sequences. For the task of molecular structure elucidation from molecular formulas and IR spectra, we denote the input by $x = ( \mathcal { F } , \mathcal { X } )$ and the predicted output by $y \in \mathcal { V }$ , with the goal of generating y identical to the ground-truth SMILES $y ^ { \ast } \in \mathcal { \dot { y } }$ given x. However, this task is ill-posed in the sense that multiple distinct isomers, which are often structurally similar, can explain the same input x. Therefore, in practice, the output is often represented as an ordered list of M candidate structures $\mathbf { y } ( x ) = ( y _ { 1 } , \dots , y _ { M } )$ , where $y _ { i } \in \mathcal { V }$

Following Alberts et al. [2], we formulate molecular structure elucidation from molecular formulas and IR spectra as learning a conditional generation policy:

$$
\pi _ { \theta } ( \cdot \mid x ) : x \to \Delta ( y ) ,\tag{2}
$$

where $\Delta ( \mathcal { V } )$ denotes the set of probability distributions over $\mathcal { V }$ . With the policy $\pi _ { \boldsymbol { \theta } } ( \cdot \mid x ) , \mathbf { y } ( x )$ can be inferred via beam search given x.

## 3.2 Supervised Learning for Molecular Structure Elucidation

In prior work (e.g., IRtoMol [2]), $\pi _ { \boldsymbol { \theta } } ( \cdot \mid x )$ is obtained via supervised learning on a dataset $\mathcal { D } =$ $\{ ( x , y ^ { * } ) \}$ . Representing the ground-truth SMILES as $y ^ { * } = ( s _ { 1 } , \ldots , s _ { T } )$ , θ is learned by minimizing the token-level negative log-likelihood:

$$
\mathcal { L } _ { \operatorname { s u p } } ( \theta ) = \mathbb { E } _ { ( x , y ^ { * } ) \sim \mathcal { D } } \Big [ - \sum _ { t = 1 } ^ { T } \log \pi _ { \theta } ( s _ { t } \mid s _ { < t } , x ) \Big ] .\tag{3}
$$

This objective encourages $\pi _ { \theta }$ to assign high probability to the ground-truth structure $y ^ { * }$ , enabling it to perform molecular structure elucidation to a certain degree. However, the supervised policy $\pi _ { \theta }$ often decodes chemically implausible candidate structures, as evidenced by violations of key chemical constraints such as exact molecular formula matching and IR spectral consistency.

## 3.3 Preference Dataset Construction from Chemical Feedback

To achieve more chemically plausible molecular structure elucidation, we employ preference optimization to update the supervised policy $\pi _ { \boldsymbol { \theta } } ( \cdot \mid x )$ , yielding an improved policy $\pi _ { \theta ^ { \prime } } ( \cdot \mid x )$ . A key prerequisite is a preference dataset that encodes chemically meaningful rankings over candidate structures. Accordingly, for each $x \in \mathcal { D }$ , we first obtain an ordered list of candidate structures by decoding from the supervised policy $\pi _ { \boldsymbol { \theta } } ( \cdot \mid x )$ via beam search. We then evaluate each candidate y using chemical feedback from the input x, namely molecular formula matching and IR spectral consistency, and re-rank the candidates according to a chemistry-informed tiered lexicographic preference scheme. This produces a chemically preferred ordered list $\mathbf { y } ^ { * } ( x )$ , and collecting such lists over all inputs yields the desired preference dataset ${ \mathcal { D } } ^ { * } = \{ ( x , \mathbf { y } ^ { * } ( x ) ) \}$ . We next specify the chemical feedback metrics, the tiered preference scheme, and the construction procedure for $\mathcal { D } ^ { * }$

Chemical feedback. We define chemical feedback as the combination of two chemically motivated metrics that can be computed from an input x and a candidate structure $y .$ The first is whether the molecular formula matches, which can be expressed as:

$$
m _ { \hat { \mathcal F } } ( y , \mathcal { F } ) : = \mathbb { I } \Big [ \hat { \mathcal F } ( y ) = \mathcal { F } \Big ] \in \{ 0 , 1 \} ,\tag{4}
$$

where $\hat { \mathcal { F } } ( y )$ denotes the molecular formula implied by the candidate structure $y .$ . The second measures how consistent the theoretical IR spectrum is with the input spectrum, which can be formulated as:

$$
s _ { \mathrm { I R } } ( y , \mathcal { X } ) : = \mathrm { C o s S i m } ( g ( y ) , \mathcal { X } ) ,\tag{5}
$$

where $g ( y )$ is a fixed predictor that maps a molecular structure $y$ to its theoretical IR spectrum. In this work, we adopt Chemprop-IR [15] as $g ( \cdot )$

Chemistry-informed tiered preference scheme. Chemical feedback provides criteria for assessing the quality of candidate molecular structures. Based on this feedback, we develop a tiered preference scheme with three priority levels: (i) highest priority $\kappa _ { \mathrm { 1 } } \mathrm { : }$ candidates whose molecular structures are identical to the ground-truth molecular structure; (ii) intermediate priority $\kappa _ { 2 } \mathrm { : }$ : among the remaining candidates, those whose molecular formulas exactly match the input molecular formula are preferred; within this formula-matched subset, candidates with higher IR spectral consistency are preferred; (iii) lowest priority $\kappa _ { 3 } ;$ among candidates that do not match the input molecular formula, those with higher IR spectral consistency are preferred.

Concretely, this tiered preference scheme can be implemented by assigning each candidate $y \textrm { a }$ lexicographic key, which can be expressed as:

$$
k _ { x } ( y ) : = \Big ( m _ { \mathrm { M o l } } ( y , y ^ { * } ) , m _ { \hat { \mathcal { F } } } ( y , \mathcal { F } ) , s _ { \mathrm { I R } } ( y , \mathcal { X } ) \Big ) ,\tag{6}
$$

where $m _ { \mathrm { M o l } } ( y , y ^ { \ast } )$ indicates whether the candidate molecular structure $y$ is identical to the groundtruth molecular structure $y ^ { * }$ , which can be formulated as:

$$
m _ { \mathrm { M o l } } ( y , y ^ { \ast } ) : = \mathbb { I } [ y = y ^ { \ast } ] \in \{ 0 , 1 \} .\tag{7}
$$

Construction of preference dataset $\mathcal { D } ^ { * }$ . We define the preference dataset as ${ \mathcal { D } } ^ { * } = \{ ( x , \mathbf { y } ^ { * } ( x ) ) \}$ where each input x is drawn from the supervised training dataset $\mathcal { D } _ { : }$ , and $\mathbf { y } ^ { * } ( x ) = ( y _ { 1 } ^ { * } , \ldots , y _ { M } ^ { * } )$ denotes a chemically preferred ordered list of M candidate structures for x. More specifically, for a given $x , \mathbf { y } ^ { * } ( x )$ is constructed by sorting the candidate list $\mathbf { y } ( x )$ in decreasing lexicographic order of $\bar { k } _ { x } ( \cdot )$ , which can be expressed as:

$$
\mathbf { y } ^ { * } ( x ) : = { \mathrm { ~ s o r t \_ d e s c e n d i n g } } ( \mathbf { y } ( x ) , k e y = k _ { x } ) .\tag{8}
$$

The complete procedure for constructing $\mathbf { y } ^ { * } ( x )$ is provided in Appendix A.

## 3.4 Formula- and IR-Matched Preference Optimization (FIRMPO) for Reasonable Molecular Structure Elucidation

To more effectively exploit the preference signal encoded in $\mathcal { D } ^ { * }$ for policy updates and thereby promote reasonable molecular structure elucidation, we propose FIRMPO. FIRMPO explicitly incorporates our chemistry-informed tiered preferences, i.e., exact molecular formula matching and IR spectral consistency, and its loss $\mathcal { L } _ { \mathrm { F I R M P O } } ^ { \mathcal { K } ( x ) }$ is formulated as:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { F I R M P O } } ^ { K ( x ) } ( \pi _ { \theta ^ { \prime } } ; \pi _ { \theta } ) = - \mathbb { E } _ { ( x , y _ { 1 } , \dots , y _ { M } ) \sim \mathcal { D } ^ { * } } \Bigg [ \displaystyle \sum _ { i = 1 } ^ { K _ { 2 } ( x ) } \log \sigma \Bigg ( - \log \sum _ { j = \operatorname* { m a x } ( K _ { 1 } ( x ) , i ) + 1 } ^ { K _ { 3 } ( x ) } \left[ { \mathbf w } _ { 1 } \right] _ { i } \exp ( - \left[ \frac { \pi ( x ) } { K _ { 2 } ( x ) + 1 } \right] _ { i } ) } \\ & { \Bigg ( ( \beta [ W _ { 2 } ] _ { i , j } \log \frac { \pi _ { \theta ^ { \prime } } ( y _ { j } | x ) } { \pi _ { \theta } ( y _ { j } | x ) } - \beta [ W _ { 2 } ] _ { i , j } \log \frac { \pi _ { \theta ^ { \prime } } ( y _ { i } | x ) } { \pi _ { \theta } ( y _ { i } | x ) } \Bigg ) \Bigg ) \Bigg ] , } \end{array}\tag{9}
$$

![](images/17fdec9d55eabfd8166b2b6e3094cb50988846bb283142111762c12c7f9391b1.jpg)  
Figure 2: Overview of the proposed method.

where $\sigma ( \cdot )$ is the sigmoid function and $\beta$ controls the strength of the regularization that keeps the updated policy $\pi _ { \theta ^ { \prime } }$ close to the reference policy $\pi _ { \theta } . \ K _ { 1 } ( x )$ denotes the number of candidates in the highest-priority tier $\mathcal { K } _ { 1 } , \mathcal { K } _ { 2 } ( x )$ denotes the number of candidates in tiers $\kappa _ { 1 }$ and $\kappa _ { 2 } .$ , and $\kappa _ { 3 } ( x )$ denotes the total number of candidates in tiers $\kappa _ { 1 } , \kappa _ { 2 }$ , and $\displaystyle { { \cal { K } } _ { 3 } }$ . The weight vector $\mathbf { w } _ { 1 }$ and the pairwise strength coefficients $[ W _ { 2 } ] _ { i , j }$ (collected in a matrix $W _ { 2 } )$ are defined as follows:

$$
\begin{array} { c } { { \displaystyle { \bf w } _ { 1 } = \overbrace { \left[ K _ { 2 } ( x ) K _ { 2 } ( x ) \cdots K _ { 2 } ( x ) \right] } ^ { K _ { 1 } ( x ) } \overbrace { 1 \cdots 1 ] } ^ { K _ { 3 } ( x ) - K _ { 1 } ( x ) } \ , } } \\ { { \displaystyle { [ W _ { 2 } ] _ { i , j } : = \left\{ \begin{array} { l l } { { 0 , } } & { j \leq i , } \\ { \frac { 1 } { K _ { 2 } ( x ) } , } & { i < j \leq K _ { 2 } ( x ) , } \\ { 1 , } & { K _ { 2 } ( x ) < j \leq K _ { 3 } ( x ) . } \end{array} \right. } } } \end{array}
$$

Here, the weight vector $\mathbf { w } _ { 1 }$ upweights comparisons anchored at candidates in the highest-priority tier $\kappa _ { 1 }$ . The pairwise strength coefficients $[ W _ { 2 } ] _ { i , j }$ assign weaker strength to comparisons among topranked, formula-matched candidates $( i \stackrel { . } { < } j \stackrel { . } { \leq } \mathcal { K } _ { 2 } ( x ) )$ ) and stronger strength to comparisons against lower-priority candidates $( \mathcal { K } _ { 2 } ( x ) < j \le \bar { \mathcal { K } _ { 3 } ( x ) } )$ . The lower summation bound max $( \dot { \mathcal { K } } _ { 1 } ( x ) , i ) \dot { + } 1$ in Eq. (9) avoids imposing an arbitrary within-tier ordering among $\kappa _ { 1 }$ candidates.

Finally, the improved policy $\pi _ { \theta ^ { \prime } } ( \cdot \mid x )$ is obtained by optimizing the FIRMPO objective:

$$
\theta ^ { \prime } = \arg \operatorname* { m i n } _ { \theta ^ { \prime } } \mathcal { L } _ { \mathrm { F I R M P O } } ^ { K ( x ) } ( \pi _ { \theta ^ { \prime } } ; \pi _ { \theta } ; \mathcal { D } ^ { * } ) .\tag{10}
$$

An overview of our method is illustrated in Fig. 2.

Advantages of FIRMPO. Unlike other listwise preference optimization methods, FIRMPO is tailored to reasonable molecular structure elucidation by guiding the model to rank candidate molecular structures using chemical feedback. Compared with generic listwise methods, FIRMPO offers three key advantages:

1) Tier-aware optimization without imposing unnecessary within-tier order. Instead of optimizing a fixed top-K prefix that implicitly enforces a strict total order, FIRMPO respects our tiered preference structure. In particular, candidates in the top-priority tier $\kappa _ { 1 }$ correspond to the same ground-truth molecule and therefore should be treated as equally preferred, rather than being assigned an arbitrary within-tier precedence.

2) Explicit upweighting for top-priority reference candidates. FIRMPO explicitly upweights comparisons anchored at top-priority candidates $( y _ { i } \in K _ { 1 } )$ as reference points, thereby placing greater emphasis on the highest-priority tier and more effectively improving top-ranked predictions.

3) Non-uniform pairwise strength across tiers to emphasize decisive contrasts. Rather than applying a uniform comparison strength to all pairs, FIRMPO assigns weaker strength to comparisons within the formula-matched region (tiers $\kappa _ { 1 }$ and $\kappa _ { 2 } )$ and stronger strength to contrasts against lower-priority, formula-mismatched candidates (tier $\kappa _ { 3 } )$ . This design encourages cross-tier separation dominate training while avoiding overly harsh constraints on subtle within-region distinctions.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate on three IR spectrum–structure benchmarks covering both simulated and experimental spectra. For simulated spectra, we use two datasets: (i) the QM9S synthetic IR dataset introduced by Zou et al. [26], and (ii) the IBM synthetic IR dataset introduced by Alberts et al. [3]. These benchmarks provide paired IR spectra and molecular structures with known ground truth. For experimental spectra, we use gas-phase IR spectra from the National Institute of Standards and Technology (NIST) Chemistry WebBook [13], which reflects realistic measurement noise and artifacts absent from simulated data. For all datasets, we randomly split the data into training, validation, and test sets with an 85/5/10% ratio.

Baselines. We evaluate FIRMPO on three supervised IR-based structure elucidation backbones. (i) IRtoMol [2] is a Transformer-based spectrum-to-SMILES model. (ii) PatchIR [25] improves spectral representation learning through patch-based self-attention. (iii) IR-Bench [4] further strengthens Transformer-based IR modeling with improved spectral representations, data augmentation, and formula-constrained decoding.

For each backbone, we compare the original supervised model (Base) with its FIRMPO-optimized counterpart (+FIRMPO). This paired evaluation setup enables a direct assessment of whether FIRMPO consistently improves molecular structure elucidation performance across different architectures and datasets.

Evaluation metrics. We report top-k molecular accuracy (%), formula accuracy (%), and IR similarity for $k \in \{ 1 , 5 , 1 0 \}$ . All metrics are computed over the top-k candidate set: an instance is counted as correct if any candidate among the top-k satisfies the corresponding criterion. Specifically, the top-k molecular accuracy is $\bar { \mathrm { A c c @ } k } = 1 0 0 \cdot \mathbb { P } [ \exists i \leq k : m _ { \mathrm { M o l } } ( y _ { i } , y ^ { \bar { \star } } ) = 1 ]$ , where P[·] denotes the empirical probability over the test set. The top-k formula accuracy is $\operatorname { A c c } _ { \mathrm { F } } @ k =$ $1 0 0 \cdot \mathbb { P } \Big [ \exists i \leq k : \hat { \mathcal { F } } ( y _ { i } ) = \mathcal { F } \Big ]$ . The top-k IR similarity is $\mathrm { S i m @ } k = \mathbb { E } [ \operatorname* { m a x } _ { 1 \leq i \leq k } \mathrm { S i m } ( \mathcal { X } , g ( y _ { i } ) ) ]$ where $\bar { \mathbb { E } } [ \cdot ]$ denotes the average over the test set.

## 4.2 Main Results

Table 1 reports overall performance on molecular structure elucidation from molecular formulas and IR spectra for QM9S [26], IBM [3], and the NIST Chemistry WebBook [13]. Across all three datasets and all evaluated backbones, applying FIRMPO consistently improves molecular accuracy, particularly at top-1. The improvements generally persist at top-5, indicating that FIRMPO enhances the overall ranking quality of generated candidates rather than improving only a single cutoff. The gains are accompanied by consistent improvements in both formula accuracy and IR spectrum similarity across different backbones and datasets, indicating that FIRMPO more effectively prioritizes chemically consistent candidates whose molecular formulas and theoretical IR spectra better match the input observations. Overall, the improvements in formula accuracy and IR spectrum similarity indicate that FIRMPO produces candidate structures that better satisfy chemical feedback, leading to more chemically reasonable molecular structure elucidation. More importantly, the consistent gains in molecular accuracy show that FIRMPO recovers the ground-truth structure more frequently by ranking chemically correct candidates higher.

Table 1: Overall performance of supervised IR-based structure elucidation backbones and their FIRMPO-optimized variants on QM9S [26], IBM [3], and NIST Chemistry WebBook [13]. We report top-k molecular accuracy (%), formula accuracy (%), and IR spectrum similarity. The best result for each metric within each dataset is highlighted in bold.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Backbone Variant</td><td rowspan="2"></td><td colspan="3">Molecule accuracy</td><td colspan="3">Formula accuracy</td><td colspan="3">IR spectrum similarity</td></tr><tr><td>Acc@1</td><td>Acc@5</td><td>Acc@10</td><td></td><td>AcCF@1 AcCF@5 AcCF@10</td><td></td><td>Sim@1</td><td>Sim@5</td><td>Sim@10</td></tr><tr><td rowspan="4">QM9S</td><td rowspan="2">IRtoMol</td><td rowspan="2">Base +FIRMPO</td><td>4.68</td><td>14.26</td><td>18.01</td><td>96.97</td><td>99.69</td><td>99.84</td><td>0.551</td><td>0.639</td><td>0.657</td></tr><tr><td>6.48</td><td>16.53</td><td>19.96</td><td>99.88</td><td>99.88</td><td>99.88</td><td>0.662</td><td>0.663</td><td>0.665</td></tr><tr><td rowspan="2">PatchIR</td><td rowspan="2">Base +FIRMPO</td><td>0.28 0.29</td><td>1.28</td><td>2.43</td><td>99.84</td><td>99.94</td><td>99.95</td><td>0.197</td><td>0.249</td><td>0.271</td></tr><tr><td></td><td>1.29</td><td>2.43</td><td>99.95</td><td>99.95</td><td>99.95</td><td>0.270</td><td>0.271</td><td>0.271</td></tr><tr><td rowspan="4"></td><td rowspan="2">IR-Bench</td><td>Base +FIRMPO</td><td>2.35 4.75</td><td>12.32 15.77</td><td>22.65 22.65</td><td>98.75 99.98</td><td>99.94 99.98</td><td>99.98 99.98</td><td>0.199 0.268</td><td>0.251 0.269</td><td>0.270 0.270</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Base IRtoMol +FIRMPO</td><td>15.42 25.45</td><td>39.82 41.12</td><td></td><td>42.18 44.97</td><td>64.11 95.24</td><td>86.87 95.24</td><td>91.07 95.24</td><td>0.609 0.719</td><td>0.717 0.747</td><td>0.735 0.754</td></tr><tr><td></td><td>1.23</td><td>6.93</td><td>11.75</td><td>75.63</td><td></td><td>98.28</td><td>0.355</td><td></td><td>0.411</td></tr><tr><td rowspan="4"></td><td rowspan="2">PatchIR</td><td>Base +FIRMPO Base</td><td>4.31</td><td>9.81</td><td>11.75</td><td>98.28</td><td>95.65 98.28</td><td>98.28</td><td>0.402</td><td>0.397 0.407</td><td>0.411</td></tr><tr><td></td><td>69.48</td><td>86.10</td><td>88.90</td><td>95.23</td><td>99.62</td><td>99.90</td><td>0.367</td><td>0.392</td><td>0.405</td></tr><tr><td rowspan="2">IR-Bench</td><td>+FIRMPO</td><td>70.79</td><td>86.94</td><td>88.90</td><td>99.90</td><td>99.90</td><td>99.90</td><td>0.400</td><td>0.402</td><td>0.405</td></tr><tr><td>Base</td><td>36.51</td><td>54.33</td><td>56.68</td><td>67.33</td><td>82.05</td><td>85.15</td><td>0.728</td><td>0.856</td><td>0.877</td></tr><tr><td rowspan="5">NIST WebBook PatchIR</td><td rowspan="2">IRtoMol</td><td>+FIRMPO</td><td>55.94</td><td>56.93</td><td>57.05</td><td>86.88</td><td>86.88</td><td>86.88</td><td>0.761</td><td>0.879</td><td>0.888</td></tr><tr><td>Base</td><td>15.35</td><td>37.25</td><td>47.28</td><td>74.50</td><td>87.62</td><td>91.34</td><td>0.783</td><td>0.886</td><td>0.910</td></tr><tr><td rowspan="2"></td><td>+FIRMPO</td><td>38.12</td><td>45.42</td><td>47.28</td><td>91.34</td><td>91.34</td><td>91.34</td><td>0.896</td><td>0.907</td><td>0.910</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">IR-Bench</td><td>Base +FIRMPO</td><td>50.99 62.38</td><td>76.61 79.58</td><td>83.42 83.42</td><td>96.04 98.89</td><td>98.64 98.89</td><td>98.89 98.89</td><td>0.878 0.917</td><td>0.925 0.928</td><td>0.933</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.933</td></tr><tr><td></td><td colspan="8"></td><td colspan="3"></td></tr><tr><td></td><td rowspan="2"></td><td colspan="4"></td><td colspan="2"></td><td colspan="3" rowspan="2"></td><td rowspan="2"></td></tr><tr><td rowspan="2"></td><td rowspan="2" colspan="3"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td><img src="images/43bb88e65293e221960f53dac3c231c01122fd02ed009dee3da01a5c2541c10b.jpg"/></td></tr></table>

Figure 3: Case study on a NIST Chemistry WebBook molecule (SMILES: CCCCC(CC)CC(=O)O) [13], showing 2D depictions of the top-5 predicted candidate structures. Candidates matching the ground-truth structure are highlighted with green boxes.

## 4.3 Case Study

To qualitatively evaluate the effect of FIRMPO on molecular structure elucidation, we conduct a case study on a NIST WebBook molecule with SMILES CCCCC(CC)CC(=O)O. Figure 3 shows the top-5 predicted candidate structures generated by IRtoMol and IRtoMol+FIRMPO. Across both models, most candidates preserve the carboxylic-acid scaffold, indicating that IR spectra strongly constrain the functional-group identity. The primary differences lie in the ranking of the ground-truth structure and the chemical plausibility of alternative candidates.

IRtoMol. IRtoMol ranks two related carboxylic-acid isomers ahead of the ground-truth structure, placing the correct molecule at rank 3. This behavior suggests that while the model successfully captures the dominant carboxylic-acid spectral signature, it struggles to distinguish subtle alkylchain branching patterns that produce highly similar IR spectra. Beyond the correct candidate, the model generates less plausible hypotheses, including fluorinated analogs, indicating relatively weak enforcement of molecular-formula consistency during decoding.

Table 2: Ablation of preference signals on the NIST Chemistry WebBook dataset [13], reporting top-1, top-5, and top-10 molecular accuracy (%), formula accuracy (%), and IR similarity. The best performance for each metric is in bold.
<table><tr><td rowspan="2">Model</td><td colspan="3">Molecule accuracy</td><td colspan="3">Formula accuracy</td><td colspan="3">IR spectrum similarity</td></tr><tr><td>Acc@1 Acc@5</td><td></td><td>Acc@10</td><td></td><td>AcCF@1 AcCF@5</td><td>AccF@10 Sim@1</td><td></td><td></td><td>Sim@5 Sim@10</td></tr><tr><td>IRtoMol</td><td>36.51</td><td>54.58</td><td>56.93</td><td>67.45</td><td>82.18</td><td>85.27</td><td>0.728</td><td>0.856</td><td>0.877</td></tr><tr><td>IRtoMol+FIRMPO w/o IR</td><td>41.38</td><td>56.31</td><td>57.05</td><td>86.88</td><td>86.88</td><td>86.88</td><td>0.755</td><td>0.873</td><td>0.887</td></tr><tr><td>IRtoMol+FIRMPO w/o Formula</td><td>52.97</td><td>56.56</td><td>57.05</td><td>68.81</td><td>83.54</td><td>86.39</td><td>0.761</td><td>0.877</td><td>0.887</td></tr><tr><td>IRtoMol+FIRMPO</td><td>55.94</td><td>56.93</td><td>57.05</td><td>86.88</td><td>86.88</td><td>86.88</td><td>0.762</td><td>0.879</td><td>0.888</td></tr></table>

IRtoMol+FIRMPO (Ours). IRtoMol+FIRMPO places the ground-truth structure at rank 1, demonstrating improved discrimination among closely related alkyl-branching isomers. Compared with the original IRtoMol model, the remaining candidates remain chemically plausible and formulaconsistent, while avoiding spurious heteroatom substitutions.

## 4.4 Ablations on Preference Signals

We further examine the contribution of the preference signals used by FIRMPO to construct the chemically preferred ordered list $\mathbf { y } ^ { * } ( x )$ . Since the main experiments evaluate FIRMPO across multiple supervised backbones, we use IRtoMol as a representative backbone in this analysis. This isolates the effect of different preference definitions while keeping the backbone architecture, candidate generation procedure, and preference optimization pipeline fixed.

Variants. All variants use the same IRtoMol backbone and the same candidate generation procedure, and differ only in how the ordered preference list is constructed during preference optimization. The IRtoMol baseline denotes the original supervised model without FIRMPO optimization, while IRtoMol+FIRMPO denotes the full model using the complete tiered preference definition in Eq. (6). We further consider two variants derived from the full model: (i) IRto-Mol+FIRMPO w/o IR, which removes the IR-consistency preference signal, i.e., Eq. (6) becomes $k _ { x } ( y ) : = ( m _ { \mathrm { M o l } } ( y , y ^ { \star } ) , m _ { \hat { \mathcal { F } } } ( y , \mathcal { F } ) )$ ; and (ii) IRtoMol+FIRMPO w/o Formula, which removes the formula-matching preference signal, i.e., Eq. (6) becomes $k _ { x } ( y ) : = \left( m _ { \mathrm { M o l } } ( y , y ^ { \star } ) , s _ { \mathrm { I R } } ( x , y ) \right)$

Results and interpretation. We conduct this analysis on the NIST Chemistry WebBook dataset. Table 2 shows that all FIRMPO variants improve Acc@1 over the original IRtoMol baseline, indicating the effectiveness of preference optimization. Removing formula matching substantially reduces formula accuracy, while removing IR consistency preserves formula accuracy but weakens molecular ranking and IR similarity. The full IRtoMol+FIRMPO model achieves the best performance, and matches the best formula accuracy across all k. These results show that formula matching and IR consistency provide complementary signals for chemistry-informed preference optimization.

## 5 Conclusion

This work narrows the gap between top-1 and top-k molecular structure elucidation from molecular formulas and IR spectra. We propose Formula- and IR-Matched Preference Optimization (FIRMPO), a general and plug-and-play chemical feedback-driven preference optimization framework for molecular structure elucidation. FIRMPO is model-agnostic and integrates with IR-based structure prediction models. It constructs chemistry-informed preferences from exact molecular formula matching and IR spectral consistency, and optimizes models to prioritize chemically reasonable candidates that better satisfy these feedback signals. Experiments on two simulated datasets and one experimental IR dataset show that FIRMPO consistently improves molecular structure elucidation accuracy across different backbone models, with gains persisting at top-5 and top-10.

## References

[1] Mike J Adams. Chemometrics in analytical spectroscopy. Number 8. Royal Society of Chemistry, 2004.

[2] Marvin Alberts, Teodoro Laino, and Alain C Vaucher. Leveraging infrared spectroscopy for automated structure elucidation. Communications Chemistry, 7(1):268, 2024.

[3] Marvin Alberts, Oliver Schilter, Federico Zipoli, Nina Hartrampf, and Teodoro Laino. Unraveling molecular structure: A multimodal spectroscopic dataset for chemistry. Advances in Neural Information Processing Systems, 37:125780–125808, 2024.

[4] Marvin Alberts, Federico Zipoli, and Teodoro Laino. Setting new benchmarks in ai-driven infrared structure elucidation. Digital Discovery, 4(7):1936–1943, 2025.

[5] Shihao Cai, Chongming Gao, Yang Zhang, Wentao Shi, Jizhi Zhang, Keqin Bao, Qifan Wang, and Fuli Feng. K-order ranking preference optimization for large language models. arXiv preprint arXiv:2506.00441, 2025.

[6] John Coates et al. Interpretation of infrared spectra, a practical approach. Encyclopedia of analytical chemistry, 12:10815–10837, 2000.

[7] Vu Hoang Minh Doan, Cao Duong Ly, Sudip Mondal, Thi Thuy Truong, Tan Dung Nguyen, Jaeyeop Choi, Byeongil Lee, and Junghwan Oh. Fcg-former: identification of functional groups in ftir spectra using enhanced transformer-based model. Analytical Chemistry, 96(30): 12358–12369, 2024.

[8] Abigail A Enders, Nicole M North, Chase M Fensore, Juan Velez-Alvarez, and Heather C Allen. Functional group identification for ftir spectra using image-based machine learning models. Analytical Chemistry, 93(28):9711–9718, 2021.

[9] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.

[10] Jonathan A Fine, Anand A Rajasekar, Krupal P Jethava, and Gaurav Chopra. Spectral deep learning for prediction and prospective validation of functional groups. Chemical science, 11 (18):4618–4630, 2020.

[11] Kevin Judge, Chris W Brown, and Lutz Hamel. Sensitivity of infrared spectra to chemical functional groups. Analytical chemistry, 80(11):4186–4192, 2008.

[12] Peter Larkin. Infrared and Raman spectroscopy: principles and spectral interpretation. Elsevier, 2017.

[13] Peter J Linstrom and William G Mallard. The nist chemistry webbook: A chemical data resource on the internet. Journal ofChemical & Engineering Data, 46(5):1059–1063, 2001.

[14] Tianqi Liu, Zhen Qin, Junru Wu, Jiaming Shen, Misha Khalman, Rishabh Joshi, Yao Zhao, Mohammad Saleh, Simon Baumgartner, Jialu Liu, et al. Lipo: Listwise preference optimization through learning-to-rank. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2404–2420, 2025.

[15] Charles McGill, Michael Forsuelo, Yanfei Guan, and William H Green. Predicting infrared spectra with message passing neural networks. Journal of Chemical Information and Modeling, 61(6):2594–2609, 2021.

[16] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

[17] Donald L Pavia, Gary M Lampman, George S Kriz, and James R Vyvyan. Introduction to spectroscopy. Belmont, USA, page 13, 2001.

[18] Dev Punjabi, Yu-Chieh Huang, Laura Holzhauer, Pierre Tremouilhac, Pascal Friederich, Nicole Jung, and Stefan Bräse. Infrared spectrum analysis of organic molecules with neural networks using standard reference data sets in combination with real-world data. Journal of Cheminformatics, 17(1):1–13, 2025.

[19] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

[20] Robert M Silverstein and G Clayton Bassler. Spectrometric identification of organic compounds. Journal ofChemical Education, 39(11):546, 1962.

[21] Douglas A Skoog, F James Holler, and Timothy A Nieman. Principles of instrumental analysis, volume 158. Saunders College Philadelphia, 1980.

[22] Brian C Smith. Infrared spectral interpretation: a systematic approach. CRC press, 2018.

[23] Barbara H Stuart. Infrared spectroscopy: fundamentals and applications. John Wiley & Sons, 2004.

[24] Rianne E van Outersterp, Jonathan Martens, Giel Berden, Valerie Koppen, Filip Cuyckens, and Jos Oomens. Mass spectrometry-based identification of ortho-, meta-and para-isomers using infrared ion spectroscopy. Analyst, 145(18):6162–6170, 2020.

[25] Wenjin Wu, Ales Leonardis, Jianbo Jiao, Jun Jiang, and Linjiang Chen. Transformer-based models for predicting molecular structures from infrared spectra using patch-based self-attention. The Journal ofPhysical Chemistry A, 129(8):2077–2085, 2025.

[26] Zihan Zou, Yujin Zhang, Lijun Liang, Mingzhi Wei, Jiancai Leng, Jun Jiang, Yi Luo, and Wei Hu. A deep learning model for predicting selected organic molecular spectra. Nature Computational Science, 3(11):957–964, 2023.

## A Detailed Procedure for Constructing Preference Candidates

The detailed procedure for constructing preference candidates $\mathbf { y } ^ { * } ( x )$ is provided in this section.

Algorithm 1 Chemical-feedback re-ranking with a tiered lexicographic key for constructing a   
preferred ordered list   
1: Input: $x = ( { \mathcal { F } } , { \mathcal { X } } ) ;$ ; ground-truth $y ^ { * } ;$ candidate list $\mathbf { y } ( x ) = ( y _ { 1 } , \dots , y _ { M } )$   
2: Output: Chemically preferred ordered list $\mathbf { y } ^ { * } ( x ) = ( y _ { 1 } ^ { * } , \dots , y _ { M } ^ { * } )$   
3: for $\bar { i } = 1$ to M do   
4: if y can be parsed into a valid molecule then   
5: $\mathbf { \bar { \mu } } _ { a _ { i } }  \mathbb { I } [ y _ { i } = y ^ { * } ]$   
6: $b _ { i }  \mathbb { I } \bigg [ \bigg . \bigg | \bigg . \bigg | \bigg . \bigg . \bigg ] = \mathcal { F } \bigg ]$   
7: $c _ { i } \gets \bar { \mathrm { C o s S i m } } ( g ( y _ { i } ) , \mathcal { X } )$   
8: $k _ { i } \gets ( a _ { i } , b _ { i } , c _ { i } )$   
9: else   
10: $k _ { i } \gets ( - 1 , - 1 , - \infty )$   
11: end if   
12: end for   
13: $\mathbf { y } ^ { \ast } ( x ) \gets$ sort\_descending $( \mathbf { y } ( x )$ , key = k<sub>x</sub>)   
14: return $\mathbf { y } ^ { * } ( x )$

## B Experimental Details

All experiments are conducted on a single NVIDIA H100 GPU (80GB). Spectra are resampled over the range of 400– $4 0 0 0 \mathrm { c m } ^ { - 1 }$ and normalized using model-specific resolutions. For all three datasets, we use the same 85/5/10% random split into training, validation, and test sets as in the main text. The validation set is used to select the best checkpoint and tune hyperparameters, including the learning rate, batch size, and preference-optimization strength β. For IRtoMol and IR-Bench, we initialize the models using the checkpoints released by the original authors and follow their recommended experimental setup.

## C Additional Analysis of Preference Optimization

In the main experiments, we evaluate FIRMPO by applying it to multiple supervised IR-based structure elucidation backbones and comparing each original supervised model with its FIRMPOoptimized counterpart. To complement these results, we include an additional analysis comparing our chemical feedback-driven preference optimization framework with a generic listwise preference optimization baseline.

Specifically, we compare IRtoMol+FIRMPO with IRtoMol+KPO. Both methods use the same supervised IRtoMol model as the initial backbone and are trained using the same preference data construction pipeline. The difference lies in the optimization objective: IRtoMol+KPO applies a generic listwise preference optimization objective, whereas IRtoMol+FIRMPO incorporates chemical feedback based on molecular formula matching and IR spectral consistency through the task-specific weighting terms $\mathbf { w } _ { 1 }$ and $W _ { 2 }$ . These terms align the training objective with the tiered preference structure used in our task, which accounts for molecular correctness, formula consistency, and IR spectral similarity.

Figure 4 reports the top-k molecular accuracy for $k \leq 1 0$ on QM9S, IBM, and the NIST Chemistry WebBook. Across all three datasets, IRtoMol+FIRMPO consistently achieves higher top-k molecular accuracy than IRtoMol+KPO for every evaluated value of k. This result suggests that the improvement of FIRMPO is not merely due to the use of preference optimization in a generic form. Instead, the chemistry-informed weighting design provides additional benefits by better aligning the ranking objective with the evaluation criteria and the underlying structure of IR-based molecular prediction.

![](images/f9bc24a43c27f2d6c4931426d804d708dfd11555b5b4cc473ce30b50206963db.jpg)  
(a) QM9S.

![](images/42de3a921bc7a049125bf3a288731a96e5bc31bc351765a6fd05d98a55ae5895.jpg)  
(b) IBM.

![](images/7fe422a5f86705ff5253a15772b9e1568ef37e27a18c4cbb18bef8186f4c7064.jpg)  
(c) NIST Chemistry WebBook.  
Figure 4: Top-10 molecular accuracy (%) of IRtoMol+FIRMPO and IRtoMol+KPO on QM9S [26], IBM [3], and NIST Chemistry Webbook [13].

## D Additional Case Studies

To provide additional qualitative comparisons for the QM9S [26] and IBM [3] datasets, this section presents case studies of molecular structure elucidation for several molecules from each dataset, focusing on the predictions generated by IRtoMol and IRtoMol+FIRMPO.

Figures 5 and 6 show 2D depictions of the top-5 predicted candidate structures for two QM9S molecules with SMILES strings C#CC1C2OC(=N)C12O and N=C1Cn2nccc2O1, respectively. First, for the molecule with SMILES C#CC1C2OC(=N)C12O, IRtoMol ranks the ground-truth structure at rank 3, while the remaining top predictions are closely related heterocyclic structural isomers. IRto-Mol+FIRMPO ranks the ground-truth structure at rank 1, and the remaining candidates remain chemically plausible and scaffold-consistent. Second, for the molecule with SMILES N=C1Cn2nccc2O1, IRtoMol does not recover the ground-truth structure within the top-5 and instead predicts strained small-ring isomers that preserve similar functional groups but alter the core connectivity. IRto-Mol+FIRMPO ranks the ground-truth structure at rank 2, and the remaining candidates remain chemically close while preserving the same functional-group inventory and overall ring topology.

![](images/ee68cd6400d201d5a4b74d0fc6f9d76b7539b559bcaef60366173678270b6200.jpg)  
Figure 5: Case study on a QM9S molecule (SMILES: C#CC1C2OC(=N)C12O) [26], showing 2D depictions of the top-5 predicted candidate structures.

![](images/7b37cb3997fd715e70a8612b980b5cefbe7da49559a5e3bf85c5720c71b34715.jpg)  
Figure 6: Case study on a QM9S molecule (SMILES: N=C1Cn2nccc2O1) [26], showing 2D depictions of the top-5 predicted candidate structures.

Figures 7 and 8 show 2D depictions of the top-5 predicted candidate structures generated by IRtoMol and IRtoMol+FIRMPO for two IBM [3] molecules with SMILES strings Cc1nc(-c2cccnc2)cn1N(C)C(=O)OC(C)(C)C and COC(=O)CC1CC2COCC(C1)N2, respectively. First, for the molecule with SMILES Cc1nc(-c2cccnc2)cn1N(C)C(=O)OC(C)(C)C, IRtoMol ranks the ground-truth structure at rank 4, while the remaining top-5 candidates are closely related regioisomers that preserve the same pyridyl imidazole carbamate scaffold but vary substituent placement and local connectivity. IRtoMol+FIRMPO ranks the ground-truth structure at rank 1, and the remaining candidates remain chemically plausible and scaffold-consistent. Second, for the molecule with SMILES COC(=O)CC1CC2COCC(C1)N2, IRtoMol does not recover the ground-truth structure within the top-5 and instead predicts bicyclic amine and lactone-like variants with altered ring connectivity. IRtoMol+FIRMPO ranks the ground-truth structure at rank 2, and the remaining candidates remain chemically close while preserving the same functional groups and overall bicyclic framework.  
![](images/fe6e72084606f452702c6f66386724d60093b9537d24f4f8f567a7996fff4e0c.jpg)  
Figure 8: Case study on a IBM molecule (SMILES: COC(=O)CC1CC2COCC(C1)N2) [3] , showing 2D depictions of the top-5 predicted candidate structures.

## E Limitations and Future Directions

Although FIRMPO consistently improves top-ranked molecular structure prediction, its performance still depends on the quality and diversity of the candidate structures generated by the underlying backbone model. If the correct structure is absent from the candidate set, the resulting preference supervision signals may be insufficient for effectively optimizing the model. Future work could investigate tighter integration between chemical feedback and candidate generation to further improve end-to-end molecular structure elucidation.