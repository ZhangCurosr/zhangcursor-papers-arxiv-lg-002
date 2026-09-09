# SIM: Subspace Interaction-based Method for Token-Level Text Anomaly Detection

Kehan Yan<sup>1</sup>, Yue Tan<sup>2</sup>, Qingfeng Chen<sup>†1</sup>, Shiyuan Li<sup>2</sup>, Yu Zheng<sup>2</sup>, and Yixin Liu<sup>†2</sup>

<sup>1</sup> Guangxi University, Guangxi, China, <sup>2</sup> Griffith University, Queensland, Australia

2413301048@st.gxu.edu.cn, {yue.tan, shiyuan.li, yu.zheng, yixin.liu}@griffith.edu.au, qingfeng@gxu.edu.cn

Abstract—Token-level text anomaly detection, as an emerging trend of text anomaly detection, moves beyond coarse-grained document-level detection by localizing anomalous tokens within text. By providing fine-grained abnormality prediction, tokenlevel text anomaly detection plays a critical role in various real-world applications, such as spam filtering and fake news detection. However, existing methods still rely on the global distance calculation for scoring, during which the local anomaly signals are severely diluted by numerous redundant normal feature dimensions. Moreover, pre-trained language models used in these methods inevitably smooth out surface anomalies, further limiting their effectiveness in token-level anomaly detection. To address these limitations, we propose a Subspace Interactionbased Method (SIM for short) for token-level text anomaly detection. To prevent local signal dilution, SIM adopts a subspace interaction-based anomaly detector, which decouples highdimensional token embeddings into multiple low-dimensional ones, amplifying localized anomaly signals hidden within specific dimensions. To counteract the over-smoothing effect, we design a hard pseudo-anomaly generation module to construct pseudoanomalous tokens, simulating the subtle anomalies obscured by semantic smoothing. Also, a probabilistic boundary loss is developed to standardize anomaly scores into statistical distances, effectively enforcing anomalous instances to deviate significantly from the normal distribution center. Extensive experiments on multiple benchmark datasets verify the effectiveness of SIM and demonstrate its remarkable efficiency, robustness, and interpretability. The source code is available at: https://github.com/ yankehan/SIM-TAD.

Index Terms—text anomaly detection, token-level anomaly detection, document-level anomaly detection.

## I. INTRODUCTION

Anomaly detection is a foundational research problem that plays a critical role in numerous practical application scenarios [1]–[4]. For example, in tabular [5], [6] and graph data [7], [8], anomaly detection is commonly utilized to identify financial fraud or malicious nodes within social networks. However, textual data possesses highly unstructured, discrete, and complex semantic characteristics, limiting early progress in anomaly detection specifically targeting textual data [9]–[11]. These characteristics have motivated recent research on text anomaly detection, which aims to identify text instances that deviate from normal semantic or syntactic patterns [12], [13]. Due to the ubiquity of textual data, text anomaly detection has become increasingly important in scenarios such as spam filtering, fake news detection, and machine-generated content identification [14], [15].

![](images/92abd5249948d039d2ce9e37de637ad8c0dfcdbd1ba174f841ddb380d5d68bbc.jpg)  
(a) SMS Spam

![](images/3f4a8b21b579df72334d05528e0dfc1f9e6f77a7967c810b22320dab4531b3e7.jpg)  
Fig. 1. t-SNE visualization of token embeddings on two datasets corresponding to different anomaly types: SMS Spam (SMS gibberish corruption) and Grammar (grammatical anomalies).  
(b) Grammar

In recent years, the rapid advancement of pre-trained language models (PLMs) [16], [17] has provided text anomaly detection with powerful contextual representations that encode rich semantic and syntactic information. By combining these high-quality text embeddings with mainstream anomaly detectors, existing methods have achieved promising performance in solving text anomaly detection problems [18]. However, most existing studies are restricted to document-level anomaly detection tasks that assign a single anomaly score to each document to indicate its overall abnormality [9], [10], [19]. This coarse-grained detection makes it difficult for users to understand the basis for anomaly determination and prevents the precise localization of problematic segments. In fact, practical applications are in urgent demand for fine-grained anomaly localization [20]. For example, practitioners may need to locate anomalous instructions that cause program crashes in massive website backend logs, while security analysts need to identify deceptive illegal links in phishing emails. In these scenarios, fine-grained anomaly localization not only facilitates rapid troubleshooting but also significantly enhances the transparency and trustworthiness of the detection system.

To achieve fine-grained text anomaly detection, Cao et al. [20] proposed the first token-level text anomaly detection framework, named TokenCore. This method utilizes PLMs to encode each token in a document into a high-dimensional numerical embedding, and then derives anomaly scores by calculating the nearest neighbor distance between the test token and the set of normal tokens. Because PLMs can capture rich semantic and syntactic features, anomalous tokens that deviate from normal patterns exhibit a significant distance shift from the distribution of normal tokens within the embedding space. Consequently, these anomalous tokens are assigned higher anomaly scores. Overall, TokenCore opens up a new direction for text anomaly detection by extending anomaly scoring from the document level to the token level.

![](images/a4dd85bdad49108d9ef7a47f0ca688003a8f9e8fc1a3401338bd0c2af8b9ab14.jpg)  
Fig. 2. Visualization of BERT embedding similarities. Grammatical variants (“am” vs. “is”) show high latent similarity, whereas semantically distinct tokens (“good” vs. “stupid”) show lower similarity scores.

Although TokenCore [20] shows promising potential in token-level anomaly detection, it highly relies on a distancebased scoring mechanism, which directly relies on the distance of embedding vectors in the feature space, inevitably introducing two inherent limitations. Limitation 1: Local anomaly signal dilution. While a small number of obvious anomalies, such as tokens with extreme contextual mismatches or strong negative sentiments, can be easily detected in the global embedding space, most subtle anomalies, such as text corruption, are only reflected in a few specific embedding dimensions, i.e., subspaces. During global distance calculation, these localized anomaly signals are severely diluted by numerous redundant normal feature dimensions [21]. Consequently, as illustrated in Figure 1, anomalous tokens remain close to normal-token clusters in the feature space, making them difficult to distinguish through global distance-based scoring. Limitation 2: Oversmoothing effect. Because PLMs such as BERT were originally designed to optimize semantic similarity rather than anomaly sensitivity [16], [22], their representations inevitably smooth out surface anomalies, such as grammatical errors. Taking the representation-level similarity in Figure 2 as an example, surface anomalous tokens (e.g., is) exhibit high similarity to normal tokens (e.g., am) in the latent space and are difficult to distinguish, leading to weak anomaly separability for such types of anomalous tokens. Motivated by these limitations, a natural question arises: Can we develop a token-level text anomaly detection framework capable of capturing local anomaly signals while counteracting the over-smoothing effect inherent in PLMs?

To answer this question, we propose SIM, a Subspace Interaction-based Method for token-level text anomaly detection. To address Limitation 1, we design a subspace interaction-based anomaly detector, which decouples highdimensional token embeddings into multiple low-dimensional feature subspaces and introduces a cross-subspace selfattention mechanism to dynamically capture interactions between subspaces. This subspace interaction mechanism endows the model with the ability to amplify local anomaly signals hidden within specific dimensions, enhancing its sensitivity to subtle token-level anomalies. To handle Limitation 2, we train the anomaly detector with two complementary strategies, namely hard pseudo-anomaly generation and probabilistic boundary loss, improving its sensitivity to surface anomalies that are smoothed out in the representation space. Specifically, given that smoothed surface anomalies are highly similar to normal tokens in the representation space, the hard pseudo-anomaly generation introduces a distance perturbation mechanism to artificially construct pseudo-anomalous tokens. These tokens are highly similar to normal tokens in the representation space, yet they lack meaningful semantic and syntactic structures; therefore, they can serve as suitable pseudo-anomalies. Training on these hard pseudo-anomalies discourages the detector from relying solely on semantic similarity and encourages it to capture subtle anomaly signals beyond global representation distance. Furthermore, we design a probabilistic boundary loss to guide anomaly score prediction. This loss utilizes the mean and variance of the anomaly scores of normal data to construct a statistical boundary, thereby enforcing that the scores of anomalous samples significantly deviate from normal instances in a statistical sense.

In summary, the contributions of this paper are threefold:

• New Perspective: Going beyond the traditional perspective of global token embeddings, we are the first to explore the utilization of token subspace interaction information to achieve token-level text anomaly detection.

• Novel Method: We propose SIM, a lightweight model using a subspace interaction-based anomaly detector to capture subspace anomaly signals. Moreover, we introduce a hard pseudo-anomaly generation module to overcome the over-smoothing effect, and design a probabilistic boundary loss to guide the learning of anomaly scores.

• Extensive Experiments: Extensive experiments on three real-world datasets demonstrate that SIM achieves superior anomaly detection performance at both token and document levels compared to state-of-the-art approaches, while delivering remarkable efficiency, robustness, and interpretability.

## II. RELATED WORK

## A. Anomaly Detection

Anomaly detection aims to identify samples that deviate from standard data distributions [1], [2], [23]. Early research relied on explicit assumptions about data-space structures to formulate interpretable detection criteria. Specifically, LOF [24] assumes local consistency, identifying anomalies through local density deviations. iForest [25] assumes anomalies are easier to isolate in the feature space and measures abnormality through random partitioning. ECOD [26] focuses on overall distribution geometry and quantifies anomalies by modeling tail events. With the advancement of representation learning, the paradigm has shifted toward learning normative representation structures. For instance, AutoEncoder [27] uses reconstruction errors as anomaly signals, assuming normal data can be reconstructed more accurately. DeepSVDD [28] learns a compact representation of normal data and identifies samples that deviate significantly from it. LUNAR [29] leverages local neighborhood information to identify complex anomalous patterns.

Despite being widely adopted and strong baselines in domains such as tabular [30] and graph data [8], [31]–[33], these methods primarily evaluate anomalies in the global feature space. Consequently, when anomalous patterns are sparse and localized, localized anomaly signals are severely diluted by redundant normal feature dimensions [34]. This motivates fine-grained anomaly detectors that capture localized anomaly signals for improved anomaly detection performance.

## B. Document-Level Text Anomaly Detection

Document-level Text Anomaly Detection aims to identify text instances that deviate from normal semantic or syntactic patterns [12], [13], [18]. Existing methods can be primarily categorized into two technical routes: end-to-end methods and two-stage methods. End-to-end methods directly take raw text as input and output anomaly scores. For instance, CVDD [9] introduces multiple learnable contextual prototype vectors and quantifies abnormality based on the distance between sample features and these prototypes. DATE [10] applies substitution perturbations to text and constructs self-supervised tasks to model normal text patterns and identify anomalies. FATE [19] introduces a deviation learning mechanism that uses a minimal amount of labeled anomalous samples to optimize anomaly scores. In contrast, two-stage methods [35] first use PLMs [16], [17] to encode raw text into dense embedding vectors, then apply traditional anomaly detection algorithms for continuous vector spaces (e.g., LOF [24] and iForest [25]) for outlier detection.

Although both categories have made significant progress, they only predict anomaly scores for the entire text and cannot localize specific anomalous tokens. However, in practical tasks such as grammatical error correction, merely identifying an entire sentence as anomalous is meaningless. Therefore, it is essential to shift from coarse-grained document-level discrimination to fine-grained token-level anomaly detection.

## C. Token-Level Text Anomaly Detection

Token-level text anomaly detection aims to assess the global abnormality of a document while precisely localizing anomalous tokens. This task differs fundamentally from traditional supervised token-level tasks (e.g., spell checking, grammatical error detection, or Named Entity Recognition [36]). First, it operates under a one-class setting where only normal documents are available during training to learn normal token distributions, lacking supervisory signals for actual anomalies.

Second, rather than focusing on predefined anomaly types, it identifies diverse and unknown anomaly variants, increasing its complexity. As a pioneering attempt, Cao et al. [20] proposed the representative baseline TokenCore. Specifically, TokenCore maps individual tokens into high-dimensional numerical representations via PLMs, then quantifies abnormality using the nearest neighbor distance between a test token and a set of normal tokens. This paradigm establishes an effective baseline for token-level text anomaly detection.

While TokenCore achieves promising performance, its reliance on spatial distance measurements in the global feature space exposes two critical limitations [37]. First, weak anomaly signals in specific subspaces are severely diluted by redundant normal dimensions during global distance computation. Second, the intrinsic over-smoothing effect of PLMs causes surface-level anomalies to map closely to normal tokens in the latent space, making them nearly indistinguishable. To overcome these limitations, we propose SIM, a Subspace Interaction-based Method for token-level text anomaly detection. By integrating a subspace interaction-based anomaly detector and a hard pseudo-anomaly generation, SIM resolves local signal dilution and representation over-smoothing. Furthermore, we introduce a probabilistic boundary loss to guide the model in acquiring reliable anomaly scores.

## III. PRELIMINARIES

Pre-trained Language Model (PLM)-Generated Embeddings. To perform fine-grained text anomaly detection, we utilize a mainstream PLM (such as BERT [16]) to generate token-level embeddings for subsequent anomaly detection. Concretely, let $\mathcal D = \{ \mathcal X _ { i } \} _ { i = 1 } ^ { N }$ be a text dataset consisting of N documents. Each document $\mathcal { X } _ { i }$ comprises a variable-length token sequence ${ \mathcal { X } } _ { i } ~ = ~ \{ x _ { i , 1 } , x _ { i , 2 } , . ~ . ~ . ~ , x _ { i , T _ { i } } \}$ . Given a PLM $f ( \cdot )$ , each document $\mathcal { X } _ { i }$ is mapped to a sequence of token embeddings:

$$
\mathbf Z _ { i } = f ( \mathcal X _ { i } ) = \{ \mathbf z _ { i , 1 } , \mathbf z _ { i , 2 } , \dots , \mathbf z _ { i , T _ { i } } \} .\tag{1}
$$

Token-Level Text Anomaly Detection. The goal of tokenlevel text anomaly detection is to learn a token-level anomaly scoring function $s ( \cdot )$ that computes an anomaly score $s _ { i , t }$ for each token embedding ${ \bf z } _ { i , t } \mathrm { : }$

$$
s _ { i , t } = s ( \mathbf { z } _ { i , t } ) ,\tag{2}
$$

where a higher score indicates a greater degree of anomaly for the corresponding token.

Document-Level Text Anomaly Detection. The goal of document-level text anomaly detection is to learn a documentlevel anomaly score $S _ { i }$ for each document $\mathcal { X } _ { i } .$ Leveraging the obtained token-level anomaly scores, we can derive the document-level anomaly score $S _ { i }$ by applying an aggregation strategy Aggregate(·) over the individual token-level scores within $X _ { i } { \mathrm { : } }$

$$
\begin{array} { r } { { \cal S } _ { i } = \mathrm { A g g r e g a t e } ( \{ s _ { i , t } \} _ { t = 1 } ^ { T _ { i } } ) . } \end{array}\tag{3}
$$

Crucially, besides providing a document-level anomaly score, these token-level anomaly scores inherently offer fine-grained

interpretability by identifying the exact tokens that trigger document-level anomalies and quantifying their corresponding anomaly intensities.

## IV. METHOD

In this section, we introduce the proposed token-level text anomaly detection framework, SIM, in detail. As illustrated in Figure 3, to capture localized anomaly signals across specific dimensions, we shift our perspective from global token embeddings to token subspaces and design a subspace interaction-based anomaly detector (Section IV-A). Furthermore, to mitigate the over-smoothing effect inherent in pretrained language models (PLMs), we introduce a hard pseudoanomaly generation that synthesizes pseudo-anomalous tokens highly similar to normal data but lacking meaningful semantic structures (Section IV-B). Additionally, we introduce a probabilistic boundary loss to ensure that the scores of anomalous tokens significantly deviate from normal instances (Section IV-C).

## A. Subspace Interaction-based Anomaly Detector

Conventional text anomaly detection methods usually leverage global embeddings as the evidence to determine whether anomalies exist or not. However, as observed in the t-SNE visualization of token embeddings (Figure 1), local anomalies typically do not cause significant shifts in the global representation; rather, their anomalous features are mainly concentrated in a few specific dimensions [38]. Therefore, when evaluating anomaly scores directly within the global feature space, these subtle local deviations are inevitably masked and diluted by redundant normal dimensions [39], [40]. To achieve effective token-level text anomaly detection, the key challenge is to address the issue of local anomaly signal dilution. To this end, our core idea is to shift the detection paradigm from “coarsegrained global token representation analysis” to “fine-grained token subspace evaluation.” Built upon this idea, we design a subspace interaction-based anomaly detector. Specifically, we partition the high-dimensional embeddings into multiple independent subspaces and model the interactive information among them. The ultimate goal is to enable the model to explicitly isolate anomalous subspaces and adaptively suppress irrelevant normal dimensions.

To mitigate the dilution of local anomaly signals by highdimensional global representations, we first split the highdimensional embeddings to capture anomaly signals at the local subspace level. In practical, we uniformly partition the input token embedding $\textbf { z } \in \ \mathbb { R } ^ { d }$ into m distinct subspaces, represented as:

$$
\mathbf { z } = [ \mathbf { z } ^ { ( 1 ) } \parallel \mathbf { z } ^ { ( 2 ) } \parallel \cdot \cdot \cdot \parallel \mathbf { z } ^ { ( m ) } ] ,\tag{4}
$$

where ∥ denotes the concatenation operation, $\mathbf { z } ^ { ( i ) }$ denotes the representation of each subspace where $\mathbf { z } ^ { ( i ) } \in \mathbb { R } ^ { d _ { s u b } }$ and $d _ { s u b } = d / m$

Although such partitioning enables the model to examine token representations at a finer granularity, merely partitioning the representation structurally is still insufficient to capture the underlying correlation structures among different subspaces. Therefore, we introduce a linear projection step inspired by the self-attention mechanism to generate Query, Key, and Value vectors for each subspace representation independently. This provides a unified representation space foundation for subsequent cross-subspace relationship modeling:

$$
\mathbf { q } ^ { ( i ) } = \mathbf { z } ^ { ( i ) } \mathbf { W } _ { q } , \quad \mathbf { k } ^ { ( i ) } = \mathbf { z } ^ { ( i ) } \mathbf { W } _ { k } , \quad \mathbf { v } ^ { ( i ) } = \mathbf { z } ^ { ( i ) } \mathbf { W } _ { v } ,\tag{5}
$$

where $\mathbf { W } _ { q } , \mathbf { W } _ { k } , \mathbf { W } _ { v } \ \in \ \mathbb { R } ^ { d _ { s u b } \times d _ { s u b } }$ are learnable projection matrices. Next, we further utilize the scaled dot-product attention mechanism to allow each subspace to aggregate information from all other subspaces, thereby explicitly modeling the relationships among different subspaces:

$$
\mathbf { z } _ { a t t n } ^ { ( i ) } = \sum _ { j = 1 } ^ { m } \mathrm { S o f t m a x } \left( \frac { \mathbf { q } ^ { ( i ) } \mathbf { k } ^ { ( j ) } } { \sqrt { d _ { s u b } } } \right) \mathbf { v } ^ { ( j ) } .\tag{6}
$$

Driven by the cross-subspace self-attention mechanism, the model can adaptively emphasize anomaly-related subspace information while suppressing redundant or irrelevant subspace information. More importantly, for an anomalous token, its abnormality stems not only from local feature deviations within a single subspace but also from structural inconsistencies across subspaces. Therefore, we further concatenate the updated m subspaces back to the original dimension and feed them into a two-layer MLP to quantify this degree of inconsistency, which serves as the final token-level anomaly score:

$$
s = \mathbf { W } _ { 2 } \mathbf { L e a k y R e L U } ( \mathbf { W } _ { 1 } \mathbf { z } _ { a t t n } + b _ { 1 } ) + b _ { 2 } ,\tag{7}
$$

where $\mathbf { z } _ { a t t n } = [ \mathbf { z } _ { a t t n } ^ { ( 1 ) } \parallel \cdot \cdot \cdot \parallel \mathbf { z } _ { a t t n } ^ { ( m ) } ] \in \mathbb { R } ^ { d }$ is the representation after concatenating all updated subspace embeddings, $\mathbf { W } _ { 1 } ~ \in ~ \mathbb { R } ^ { d \times d }$ and $\mathbf { W } _ { 2 } ~ \in ~ \mathbb { R } ^ { \bar { d } \times 1 }$ are the weight matrices of the MLP layers, and $b _ { 1 } , b _ { 2 }$ are the corresponding bias terms. This scoring mechanism effectively prevents the local anomaly signal being diluted by operations in global redundant dimensions, allowing the final scalar score s to be more directly determined by anomaly-related subspace information.

## B. Hard Pseudo-Anomaly Generation

Although the subspace interaction-based anomaly detector can capture anomaly signals at the subspace level, the model struggles to be trained effectively due to the lack of genuine anomaly labels under the one-class setting [41]. Beyond this, a more intractable problem is that the optimization objective of PLMs prioritizes semantic similarity over anomaly sensitivity; hence, surface-level anomalies that preserve similar contextual meanings can still be mapped close to normal tokens in the representation space. This inevitably causes an oversmoothing effect on surface-level anomalies. To address these two issues, an intuitive approach is to artificially construct pseudo-anomalous tokens to simulate these anomalies hidden by semantic smoothing. By generating pseudo-anomalies through distance perturbation in the feature space, we construct pseudo-anomaly samples that are extremely close to the normal distribution in the feature space, yet inherently lack any actual semantic and syntactic structure. Through introducing challenging pseudo-anomaly samples into the training process, we can effectively mitigate the negative impact of the oversmoothing phenomenon, discouraging the model from relying solely on semantic similarity while encouraging it to capture subtle anomaly signals beyond global representation distance. Therefore, the model can better identify surface-level anomalies that are semantically close to normal tokens but violate normal semantic or syntactic structures.

![](images/76bdd9053d1bacdfa829d4799a6d5251efcc9ff85439a388c3ccb3f5bc72a2f5.jpg)  
Fig. 3. The overall pipeline of SIM for token-level text anomaly detection.

To practically construct these pseudo-anomalies, we propose a distance perturbation mechanism based on local density. Here, the local density of each normal token is estimated by the average distance to its K nearest neighbors, where a smaller average distance indicates a denser local region and thus a higher local density, and vice versa. Considering the difference in local density among different normal tokens, we utilize the K-nearest neighbors algorithm to dynamically determine the perturbation direction and perturbation magnitude for each sample. Specifically, given a mini-batch of normal token embeddings $\left\{ \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { B } \right\}$ , we first calculate the batch center $\begin{array} { r } { { \pmb \mu } _ { B } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } { \bf z } _ { i } } \end{array}$ . Subsequently, we randomly select a proportion of normal samples to be transformed into anomalies. The proportion is denoted as $\alpha ,$ a hyperparameter that adjusts the pseudo-anomaly generation ratio. For each selected target sample $\mathbf { z } _ { i } ,$ we retrieve its local K nearest neighbors $\mathcal { N } _ { K } ( { \bf z } _ { i } )$ within the current batch to estimate its local density and determine its perturbation direction and magnitude. To disrupt the original semantic anchor of the target token in the feature space without generating meaningless outliers, we aggregate all vectors pointing from the neighboring samples to $\mathbf { z } _ { i }$ to obtain the repulsion vector $\mathbf { r } _ { i }$

used for perturbation:

$$
\mathbf { r } _ { i } = \sum _ { \mathbf { z } _ { j } \in \mathcal { N } _ { K } ( \mathbf { z } _ { i } ) } ( \mathbf { z } _ { i } - \mathbf { z } _ { j } ) .\tag{8}
$$

Next, we normalize the repulsion vector $\mathbf { r } _ { i }$ to obtain the unit perturbation direction $\frac { \mathbf { r } _ { i } } { \| \mathbf { r } _ { i } \| }$ . To ensure that the generated anomalous samples simulate subtle deviations rather than trivial out-of-distribution noise, we dynamically set the perturbation magnitude based on the local density. Specifically, we calculate the average distance between $\mathbf { z } _ { i }$ and its K neighbors as the base perturbation magnitude. Then, we apply this magnitude along the perturbation direction to $\mathbf { z } _ { i }$ , obtaining the intermediate feature $\mathbf { z } _ { i } ^ { \prime } \mathrm { ; }$

$$
\mathbf { z } _ { i } ^ { \prime } = \mathbf { z } _ { i } + \beta \cdot \frac { \mathbf { r } _ { i } } { \lVert \mathbf { r } _ { i } \rVert } \cdot \left( \frac { 1 } { K } \sum _ { \mathbf { z } _ { j } \in \mathcal { N } _ { K } ( \mathbf { z } _ { i } ) } \lVert \mathbf { z } _ { i } - \mathbf { z } _ { j } \rVert \right) ,\tag{9}
$$

where $\beta$ is a hyperparameter that controls the overall repulsion strength. A larger β produces stronger perturbations, pushing the generated pseudo-anomalies farther away from their original semantic anchors.

Although the intermediate feature $\mathbf { z } _ { i } ^ { \prime }$ is no longer constrained by the correct local semantics, it still risks drifting away from the global manifold. To guarantee that the generated pseudo-anomalies consistently maintain extremely high similarity to the normal distribution in the feature space, we further apply a hypersphere projection to $\mathbf { z } _ { i } ^ { \prime }$ to constrain it onto the global manifold. Specifically, we project $\mathbf { z } _ { i } ^ { \prime }$ onto a hypersphere centered at the batch center $\pmb { \mu } _ { B }$ with a radius equal to the original distance $\| \mathbf { z } _ { i } - \mu _ { B } \|$ , so that the generated sample preserves the original radial distance. The final generated pseudo-anomaly sample $\tilde { \mathbf { z } } _ { i }$ is formalized as:

$$
\tilde { \mathbf { z } } _ { i } = \pmb { \mu } _ { B } + ( \mathbf { z } _ { i } ^ { \prime } - \pmb { \mu } _ { B } ) \frac { \lVert \mathbf { z } _ { i } - \pmb { \mu } _ { B } \rVert } { \lVert \mathbf { z } _ { i } ^ { \prime } - \pmb { \mu } _ { B } \rVert } .\tag{10}
$$

In summary, since these generated samples are entirely derived from pure physical displacement in the feature space, they inherently lack any actual semantic or syntactic structure, making them reasonable pseudo-anomalies. Meanwhile, under the strict constraints of the hypersphere projection, they will not turn into easily identifiable extreme outliers. The discriminative signals provided by these high-quality synthesized negative samples can effectively overcome the over-smoothing effect in pre-trained representations.

## C. Probabilistic Boundary Loss

After generating pseudo-samples for training and deriving anomaly scores via the subspace interaction-based anomaly detector, the remaining question is how to design a suitable optimization objective to guide model learning. A conventional approach is to utilize the Binary Cross-Entropy (BCE) loss for supervised training. However, since BCE inherently relies on the empirical fitting of positive and negative samples, it is prone to overfitting to specific pseudo-anomaly generation patterns, thereby weakening generalization to unseen anomalies. To address this issue, we aim to guide the detector with a distribution-aware objective rather than directly fitting the generated pseudo-anomalies as a fixed positive class. Specifically, we map the anomaly scores into a standardized space defined by the score distribution of normal samples, so that the optimization focuses on whether a sample statistically deviates from the normal distribution instead of matching specific pseudo-anomaly patterns. To characterize the statistics of the standardized space, we compute the mean and variance of the anomaly scores from normal instances and standardize the scores into a Z-score format, transforming the anomaly rating into a standardized distance that deviates from the mean of the normal distribution. To further exploit this statistical characterization, we design a probabilistic boundary optimization objective to constrain normal samples near the distribution center while enforcing pseudo-anomalous samples to lie several standard deviations away from the distribution center.

Score Standardization. A primary challenge in standardizing anomaly scores is how to sample the mean and variance of normal instances. Currently, there are two main approaches, i.e., sampling from real data and sampling from a prior distribution. We ultimately opt for the prior distribution because sampling from real data exhibits significant volatility, bringing about additional training instability, and related studies show that the Gaussian distribution can well fit anomaly scores across a range of datasets, making it suitable for modeling the normal score distribution [42], [43]. In this paper, we sample 5000 instances from a standard normal distribution to obtain the reference mean $\mu _ { r e f }$ and standard deviation $\sigma _ { r e f }$ for the anomaly scores of normal samples. Then, we convert the raw anomaly score s of each token into a standard Z-score:

$$
d e v ( s ) = \frac { s - \mu _ { r e f } } { \sigma _ { r e f } } ,\tag{11}
$$

which maps the raw anomaly scores into a standardized statistical space. Here, dev(s) explicitly quantifies the number of standard deviations by which a token’s score deviates from the normal data center, thereby providing a statistically interpretable measure of anomaly degree.

Probabilistic Boundary Optimization. To strengthen the generalization ability of the model, we design a probabilistic boundary optimization objective for training. It encourages the detector to learn anomaly scores based on statistically meaningful deviations from the normal distribution, rather than overfitting to specific pseudo-anomaly patterns. For a minibatch of size B, the overall probabilistic boundary loss $\mathcal { L } _ { P B L }$ is calculated as follows:

$$
\mathcal { L } _ { P B L } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left[ ( 1 - y _ { i } ) | d e v ( s _ { i } ) | + y _ { i } \operatorname* { m a x } ( 0 , a - d e v ( s _ { i } ) ) \right] ,\tag{12}
$$

where $y _ { i } = 1$ denotes a pseudo-anomalous sample, $y _ { i } = 0$ denotes a normal sample, and a is the confidence boundary parameter for the Z-score. This loss function forces the scores of normal tokens to tightly cluster around the distribution center $( \mathrm { i . e . , ~ } | d e v ( s ) | ~  ~ 0 )$ , while enforcing the generated anomalous samples to stay at least a standard deviations away from the normal center, thereby establishing a clear probabilistic boundary between normal and pseudo-anomalous samples.

In summary, the probabilistic boundary loss transforms the originally uninterpretable anomaly scores into a standardized distance metric with clear statistical significance and formulates an optimization objective constrained by a confidence boundary. This design not only avoids overfitting to the pseudo-anomaly distribution but also better guides the model in learning robust anomaly scores.

Document-Level Score Aggregation. After obtaining tokenlevel anomaly scores, it is necessary to aggregate them into a document-level score for document-level anomaly detection [20]. Specifically, given a document $\mathcal { X } _ { i }$ containing $T _ { i }$ tokens, the token-level anomaly detector assigns an anomaly score $s _ { i , t }$ to each token t. The key question is how to aggregate these token-level scores into an overall documentlevel anomaly score.

Existing methods typically employ mean pooling for aggregation since raw scores are inherently unstable and vulnerable to noise, and averaging can yield a more robust overall esti mation [20]. In practice, however, we argue that this approach is highly suboptimal for fine-grained detection tasks, and we instead utilize max pooling as an alternative aggregation function to address the limitations of mean pooling. The rationale is that in token-level anomaly detection datasets, the proportion of anomalous tokens is extremely small, often accounting for less than 1%. Under these circumstances, mean pooling inevitably averages the few strong local anomaly signals with the overwhelming majority of normal tokens, causing the critical anomaly information to be significantly diluted. Therefore, we adopt max pooling as the aggregation function, and the document-level anomaly score $S _ { i }$ for document $\mathcal { X } _ { i }$ is calculated as follows:

$$
S _ { i } = \operatorname* { m a x } _ { 1 \leq t \leq T _ { i } } s _ { i , t } .\tag{13}
$$

This design allows the document-level score to be dominated by the most suspicious token-level evidence, thereby preserving sparse but critical anomaly signals that would otherwise be smoothed out by averaging.

## D. Computational Complexity

We analyze the computational complexity of SIM for a document with T tokens and embedding dimension d, assuming a training mini-batch size of B tokens. Consistent with standard protocols, token-level embeddings are extracted once via the pretrained language model and cached [20], excluding the backbone forward pass from our analysis. The subspace interaction-based anomaly detector first partitions embeddings into m subspaces and performs cross-subspace attention, costing $O ( T d ^ { 2 } / m + T m d )$ , and subsequently employs a twolayer MLP for token-level scoring with complexity $O ( T d ^ { 2 } )$ The hard pseudo-anomaly generation is exclusively activated during training; it computes the batch center and conducts in-batch K-nearest neighbor retrieval and neighborhood repulsion, incurring a training-only overhead of $O ( B ^ { 2 } d + B K d )$ per mini-batch. Finally, both score standardization and documentlevel max-pooling aggregation scale linearly, costing O(T) per document. Consequently, the overall inference complexity per document scales as $O ( T d ^ { 2 } + T m d )$ . Because $m \ll d$ in practice, the overhead is dominated by the linear projections and MLP layers, which remain strictly linear with respect to the document length T, ensuring that SIM is highly efficient for practical deployment.

## V. EXPERIMENTS

## A. Experiment Setup

Datasets. We conduct experiments on three public benchmark datasets encompassing diverse anomaly patterns, including a grammatical error dataset (Grammar), a negative sentiment dataset (Review), and a text corruption dataset (SMS Spam). Following the standard protocol of TokenCore [20], we allocate 50% of the normal instances for training. The remaining 50% of the normal instances, together with all anomalous instances, form the test set.

Baselines. We compare SIM with representative baselines, including LOF [24], iForest [25], ECOD [26], DeepSVDD [28], AutoEncoder [27], LUNAR [29], and TokenCore [20]. To evaluate the performance of Large Language Model (LLM) on token-level text anomaly detection, we include GPT-4.1- nano<sup>1</sup> as an additional baseline.

Evaluation and Implementation. We report AUROC and AUPRC as the main metrics. For all methods, we report the average results across 3 random seeds. All methods use embeddings extracted from BERT-base-uncased<sup>2</sup> [16]. Since the tokenizer typically splits a word into multiple subwords (e.g., splitting “playing” into “play” and “##ing”), we apply max pooling to aggregate the subword embeddings, obtaining word-level representations that align with token annotations [20].

## B. Main Results

Table I reports the comparison results in terms of AUROC and AUPRC. We have the following observations. ❶ At the document level, SIM achieves the best AUROC and AUPRC across all three datasets. Compared to the strongest baseline, our method obtains a relative performance gain of over 10%. This demonstrates its consistent and effective performance for text anomaly detection under diverse anomaly patterns. ❷ At the token level, SIM obtains the best average AUROC and AUPRC. Particularly on the SMS Spam dataset, it reaches an impressive AUROC of 98.44, even outperforming GPT-4.1- nano (92.19). This demonstrates that our model is capable of capturing subtle anomaly signals, thereby proving its effectiveness in token-level text anomaly detection. ❸ Existing methods (such as TokenCore and LUNAR) demonstrate competitive performance on certain specific datasets; however, their performance varies drastically across different datasets. This reflects the inherent characteristics of pre-trained language models (PLMs) like BERT, which are primarily optimized for semantic similarity. In the Review dataset, semantic anomalies are distinctly separated within the embedding space, enabling various methods to achieve high detection performance. However, for text corruptions in SMS Spam and grammatical errors in Grammar, embeddings from PLMs inevitably smooth out these surface-level anomalies, leading to suboptimal detection results. ❹ Notably, the LLM baseline generally underperforms at both token and document levels. Although GPT-4.1-nano demonstrates certain zero-shot capabilities on the SMS Spam dataset, its overall performance still lags significantly behind. This highlights the necessity of designing specialized architectures tailored for token-level text anomaly detection.

## C. Ablation Study

To validate the key components of SIM, we evaluate three variants: ❶ w/o Sub-Int, which replaces the cross-subspace self-attention mechanism with an MLP; ❷ w/o Hard-Gen, which uses Gaussian noise instead of distance perturbation to generate pseudo-anomalies; and ❸ w/o Prob-Loss, which trains the anomaly detector with Binary Cross-Entropy (BCE) loss instead of probabilistic boundary loss.

The results are summarized in Table II, which shows that all components consistently contribute to the final performance. ❶ w/o Sub-Int causes a consistent performance drop. This indicates that treating high-dimensional token embeddings as indivisible black boxes severely dilutes local anomaly signals with redundant normal feature dimensions. ❷ w/o Hard-Gen yields the largest drop across most settings, e.g., $7 2 . 1 4  4 8 . 6 6$ on Grammar at the token level. This shows that directionless Gaussian noise fails to generate effective hard pseudo-anomalies to counteract the over-smoothing effect inherent in PLMs, making it difficult for the model to break free from its reliance on semantic similarity. $\otimes$ w/o Prob-Loss results in a severe decrease, especially when identifying text gibberish, e.g., $8 8 . 5 3 \  \ 5 4 . 7 5$ on SMS Spam at the document level. Without a statistical boundary constructed from the mean and variance of normal data, the standard BCE objective easily overfits to pseudo-anomalies, limiting its generalization to unseen anomalies.

TABLE I  
MAIN RESULTS ON AUROC AND AUPRC FOR THREE DATASETS. BEST RESULTS ARE HIGHLIGHTED IN BOLD AND SHADED.
<table><tr><td rowspan="2">Level</td><td rowspan="2">Methods</td><td colspan="2">SMS_Spam</td><td colspan="2">Review</td><td colspan="2">Grammar</td><td colspan="2">Average</td></tr><tr><td>AUROC</td><td>AUPRC</td><td>AUROC</td><td>AUPRC</td><td>AUROC</td><td>AUPRC</td><td>AUROC</td><td>AUPRC</td></tr><tr><td rowspan="9">Token-level</td><td>LOF</td><td>44.31</td><td>0.77</td><td>66.35</td><td>1.99</td><td>52.26</td><td>2.88</td><td>54.31</td><td>1.88</td></tr><tr><td>iForest</td><td>76.18</td><td>2.10</td><td>67.57</td><td>6.70</td><td>41.91</td><td>2.37</td><td>61.89</td><td>3.72</td></tr><tr><td>ECOD</td><td>82.57</td><td>2.86</td><td>69.49</td><td>7.32</td><td>47.15</td><td>2.61</td><td>66.40</td><td>4.26</td></tr><tr><td>DeepSVDD</td><td>58.74</td><td>1.10</td><td>59.88</td><td>2.20</td><td>50.52</td><td>3.00</td><td>56.38</td><td>2.10</td></tr><tr><td>AutoEncoder</td><td>36.64</td><td>0.67</td><td>63.01</td><td>1.74</td><td>63.01</td><td>3.68</td><td>54.22</td><td>2.03</td></tr><tr><td>LUNAR</td><td>66.22</td><td>1.26</td><td>81.78</td><td>4.49</td><td>60.80</td><td>3.47</td><td>69.60</td><td>3.07</td></tr><tr><td>TokenCore</td><td>70.26</td><td>1.43</td><td>81.89</td><td>4.81</td><td>64.00</td><td>3.80</td><td>72.05</td><td>3.35</td></tr><tr><td>GPT-4.1-nano</td><td>92.19</td><td>38.79</td><td>47.48</td><td>7.36</td><td>42.41</td><td>2.75</td><td>60.69</td><td>16.30</td></tr><tr><td>SIM</td><td>98.44</td><td>28.74</td><td>74.78</td><td>17.82</td><td>72.14</td><td>5.22</td><td>81.79</td><td>17.26</td></tr><tr><td rowspan="9">Doc-level</td><td>LOF</td><td>55.72</td><td>18.49</td><td>91.42</td><td>43.83</td><td>57.55</td><td>19.91</td><td>68.23</td><td>27.41</td></tr><tr><td>iForest</td><td>45.98</td><td>14.40</td><td>87.55</td><td>33.98</td><td>60.85</td><td>22.82</td><td>64.79</td><td>23.73</td></tr><tr><td>ECOD</td><td>47.10</td><td>14.75</td><td>87.07</td><td>31.34</td><td>61.88</td><td>23.90</td><td>65.35</td><td>23.33</td></tr><tr><td>DeepSVDD</td><td>41.12</td><td>13.17</td><td>78.93</td><td>20.95</td><td>65.32</td><td>28.68</td><td>61.79</td><td>20.93</td></tr><tr><td>AutoEncoder</td><td>46.43</td><td>14.56</td><td>88.45</td><td>37.36</td><td>68.07</td><td>27.72</td><td>67.65</td><td>26.55</td></tr><tr><td>LUNAR</td><td>58.78</td><td>19.14</td><td>95.66</td><td>65.04</td><td>67.05</td><td>26.72</td><td>73.83</td><td>36.97</td></tr><tr><td>TokenCore</td><td>60.59</td><td>20.06</td><td>95.68</td><td>65.10</td><td>64.51</td><td>25.69</td><td>73.59</td><td>36.95</td></tr><tr><td>GPT-4.1-nano</td><td>54.99</td><td>16.48</td><td>43.04</td><td>9.12</td><td>62.05</td><td>23.82</td><td>53.36</td><td>16.47</td></tr><tr><td>SIM</td><td>88.53</td><td>48.28</td><td>95.75</td><td>83.13</td><td>70.10</td><td>35.35</td><td>84.79</td><td>55.59</td></tr></table>

TABLE II  
ABLATION RESULTS ON AUROC FOR THREE DATASETS. BEST RESULTS ARE HIGHLIGHTED IN BOLD AND SHADED.
<table><tr><td>Level</td><td>Variants</td><td>SMS_Spam</td><td>Review</td><td>Grammar</td></tr><tr><td rowspan="4"> $\frac { \frac { \overline { { \lambda } } } { 2 } } { \underline { { \lambda } } }$ </td><td>SIM</td><td>98.44</td><td>74.78</td><td>72.14</td></tr><tr><td>w/o Sub-Int</td><td>49.39</td><td>71.35</td><td>59.77</td></tr><tr><td>w/o Hard-Gen</td><td>49.37</td><td>43.19</td><td>48.66</td></tr><tr><td>w/o Prob-Loss</td><td>60.60</td><td>59.50</td><td>62.86</td></tr><tr><td rowspan="4"> $\lesseqqgtr$ </td><td>SIM w/o Sub-Int</td><td>88.53 45.62</td><td>95.75</td><td>70.10</td></tr><tr><td></td><td></td><td>95.23</td><td>57.56</td></tr><tr><td>w/o Hard-Gen</td><td>62.89</td><td>73.16</td><td>62.46</td></tr><tr><td>w/o Prob-Loss</td><td>54.75</td><td>92.68</td><td>57.58</td></tr></table>

## D. Hyperparameter Analysis

We study the sensitivity of SIM to the hard pseudo-anomaly generation parameters, the anomaly ratio α and perturbation scale $\beta ,$ as shown in Figures 4 and 5. Across all datasets, optimal performance is consistently achieved with a relatively large $\alpha ,$ demonstrating the benefit of pseudo-anomaly samples for model training. Furthermore, the optimal perturbation scale $\beta$ depends on the specific anomaly type. On Grammar and Review, a larger $\beta$ is preferred, as a small $\beta$ produces negligible noise that fails to disrupt the original semantic and syntactic structures. Conversely, SMS Spam favors a smaller $\beta ,$ as an excessively large $\beta$ pushes generated tokens into overly distant feature-space regions, turning them into trivial outliers rather than hard pseudo-anomalies and degrading detection precision.

![](images/386e5a62b467081ee839cce77f0b513eb00ae358b527b40160d512e6e852aeaf.jpg)  
(a) Grammar

![](images/7ac244f4da9ce253af62e9330495484f0e12d16ee5521ecff4cfc764d48aa282.jpg)  
(b) Review

![](images/3b093a91da79917b726649d94acf6dab458a9feb1426fb619323307490a44c2a.jpg)  
(c) SMS Spam  
Fig. 4. Hyperparameter sensitivity analysis of token-level AUROC.

![](images/23d95204d53c9006ba8720161007c52429d5c0db9f7ff14171d8b509f09fd795.jpg)

![](images/bd00332dad845f378ac8948edc3ccd221fb12faf71b6ee4d997f475ee6a43f74.jpg)  
(a) Grammar

![](images/ed4bedf081cec51fb137a32b203f8ca380ec9a6b1a040927265c2f5625d15d70.jpg)  
(b) Review  
(c) SMS Spam  
Fig. 5. Hyperparameter sensitivity analysis of document-level AUROC.

## E. Efficiency Analysis

To assess the runtime efficiency and performance trade-off of SIM, Figure 6a compares token-level AUROC and total runtime with representative baselines on Grammar. Overall,

![](images/783d014b948b9a2562c81e88e024eeb374cd6fda0f245c8052a676fb06678515.jpg)  
(a)

![](images/90f4f9e35a1cc26d02b946e8de53a2feb85f369458f5d46c8d2b4f7ca958d40d.jpg)  
(b)  
Fig. 6. Performance evaluation on the Grammar dataset. (a) Efficiency analysis comparing various methods in terms of token-level AUROC and runtime. (b) Robustness analysis assessing AUROC stability under varying training data contamination ratios.

SIM achieves the highest AUROC with consistently low runtime. Specifically, SIM reduces execution time by an order of magnitude compared to GPT-4.1-nano and is substantially faster than AE, DeepSVDD, and LUNAR, achieving more than a twofold speedup with higher detection accuracy. Notably, despite being a deep learning-based model, SIM (0.77s) is faster than the training-free traditional baseline ECOD (0.86s). Although TokenCore and LOF are slightly faster, their detection performance is substantially inferior; for example, SIM outperforms TokenCore by 8.14% in token-level AUROC with negligible efficiency trade-off. These results demonstrate that SIM achieves a favorable balance between accuracy and efficiency without prohibitive computational overhead.

## F. Robustness Analysis

To evaluate the robustness of SIM against contaminated training data, we artificially corrupt varying proportions of normal token embeddings with zero-mean Gaussian noise scaled to three times the dataset’s standard deviation, as shown in Figure 6b. Overall, as the contamination ratio increases, the detection performance of all models exhibits a predictable downward trend. However, SIM preserves strong performance under increasing contamination. Most notably, even under the most severe scenario (10% contamination), the degraded performance of SIM still significantly surpasses the peak performance of all baselines in the strictly clean (0% contamination) setting, maintaining a clear margin over strong baselines. Ultimately, these results compellingly demonstrate that SIM can effectively mitigate the adverse impact of the inevitable data contamination found in real-world scenarios.

## G. Interpretability Analysis

To investigate the interpretability of our proposed framework, we visualize the token-level anomaly scores generated by SIM across three distinct datasets in Figure 7. The visualizations reveal that SIM effectively mitigates the oversmoothing effect of PLMs and successfully captures finegrained local anomaly signals. For instance, in the Grammar dataset, the model overcomes latent representation smoothing to precisely pinpoint the surface anomaly preposition “at” (score: 0.48), the exact token disrupting the sentence structure. In the Review dataset, it accurately assigns a high score to the strong negative descriptor “Tasteless” (0.47), while successfully ignoring neutral semantic tokens such as “food” and “Drinks”. Similarly, in the SMS Spam dataset, the model effectively detects local structural corruption by assigning the highest score (0.61) to the anomalous string “Kjjjjggjgjytfd”, thereby accurately pinpointing this irregular character sequence.

<table><tr><td>DATASET</td><td colspan="8">SAMPLE</td></tr><tr><td rowspan="2">Grammar</td><td colspan="2">The patient (0.04)</td><td colspan="2">is (0.21)</td><td colspan="2">recovering (0.18)</td><td>à (0.29)</td><td colspan="2">surgery (0.21)</td></tr><tr><td>at (0.48)</td><td>(0.28) a (0.37)</td><td colspan="2">hospital (0.23)</td><td colspan="2">(0.27) (0.25)</td><td></td><td></td></tr><tr><td rowspan="2">Review</td><td colspan="8">Tasteless Appetizers came</td></tr><tr><td colspan="2">food (0.47) (0.18) Drinks (0.10) (0.25) (0.20)</td><td colspan="2">(0.12) and vibes</td><td colspan="2">(0.22) are good</td><td colspan="2">after (0.17) (0.24)</td><td colspan="2">mains (0.29)</td></tr><tr><td colspan="2">SMS_Spam</td><td colspan="8">(0.28) (0.16) (0.24)</td></tr><tr><td rowspan="5"></td><td colspan="5">I have had two</td><td colspan="2">more letters from</td><td colspan="2">Kjigjgjytfd</td></tr><tr><td>(0.24)</td><td colspan="2">(0.24) (0.23) will</td><td colspan="2">(0.33)</td><td colspan="2">(0.30) (0.31) you</td><td>(0.36)</td><td colspan="2">(0.61)</td></tr><tr><td>(0.31)</td><td colspan="2">(0.32) (0.27)</td><td colspan="2">copy them (0.29) (0.28)</td><td colspan="2">for (0.37) (0.33)</td><td>cos (0.42)</td><td colspan="2">one has (0.29) (0.28)</td></tr><tr><td>4 (0.30)</td><td>message (0.36)</td><td>for (0.37)</td><td>you (0.30)</td><td>(0.42)</td><td>Speak (0.30)</td><td colspan="2">soon (0.35)</td><td></td></tr></table>

Fig. 7. Interpretability visualization of token-level anomaly attributions across different datasets. Darker shading corresponds to higher anomaly scores.

## VI. CONCLUSION

In this paper, we focus on token-level text anomaly detection, aiming to identify anomalous tokens within a document and provide fine-grained anomaly localization results. To this end, we propose SIM, a Subspace Interaction-based Method for token-level text anomaly detection. To address the dilution of local anomaly signals, SIM adopts a subspace interactionbased anomaly detector to dynamically amplify localized anomaly signals hidden in specific dimensions. To counteract the inherent over-smoothing effect of pre-trained language models (PLMs), we introduce hard pseudo-anomaly generation to construct pseudo-anomalous tokens and design a probabilistic boundary loss to standardize anomaly scores into statistical distances, enforcing anomalous instances to deviate from the normal distribution center. Extensive experiments demonstrate that SIM achieves state-of-the-art performance at both token and document levels on multiple benchmark datasets, with remarkable efficiency, robustness, and interpretability.

## ACKNOWLEDGMENT

The work of Qingfeng Chen was partially supported by the Specific Research Project of Guangxi for Research Bases and Talents under Grant No. GuiKe AD24010011 and the Key Research & Development Program Project of Guangxi under Grant No. GuiKe AB25069095. The work of Kehan Yan was partially supported by the Innovation Project of Guangxi Graduate Education under Grant No. YCSW2026145.

[1] G. Pang, C. Shen, L. Cao, and A. V. D. Hengel, “Deep learning for anomaly detection: A review,” ACM computing surveys (CSUR), vol. 54, no. 2, pp. 1–38, 2021.

[2] R. Chalapathy and S. Chawla, “Deep learning for anomaly detection: A survey,” arXiv preprint arXiv:1901.03407, 2019.

[3] S. Han, X. Hu, H. Huang, M. Jiang, and Y. Zhao, “Adbench: Anomaly detection benchmark,” Advances in neural information processing systems, vol. 35, pp. 32 142–32 159, 2022.

[4] Y. Zheng, M. Jin, Y. Liu, L. Chi, K. T. Phan, and Y.-P. P. Chen, “From unsupervised to few-shot graph anomaly detection: A multi-scale contrastive learning approach,” Transactions on Graph Intelligence and Network Applications (TGINA), 2026.

[5] J. Yin, Y. Qiao, Z. Zhou, X. Wang, and J. Yang, “Mcm: Masked cell modeling for anomaly detection in tabular data,” in The Twelfth International Conference on Learning Representations, 2024.

[6] H. Ye, H. Zhao, W. Fan, M. Zhou, D. Guo, and Y. Chang, “Drl: Decomposed representation learning for tabular anomaly detection,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 24 554–24 589.

[7] J. Pan, Y. Liu, Y. Zheng, and S. Pan, “Prem: A simple yet effective approach for node-level graph anomaly detection,” in 2023 IEEE International Conference on Data Mining (ICDM). IEEE, 2023.

[8] Y. Zhao, Y. Liu, S. Li, Q. Chen, Y. Zheng, and S. Pan, “Freegad: A training-free yet effective approach for graph anomaly detection,” in Proceedings of the 34th ACM International Conference on Information and Knowledge Management, 2025, pp. 4379–4389.

[9] L. Ruff, Y. Zemlyanskiy, R. Vandermeulen, T. Schnake, and M. Kloft, “Self-attentive, multi-context one-class classification for unsupervised anomaly detection on text,” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019, pp. 4061–4071.

[10] A. Manolache, F. Brad, and E. Burceanu, “Date: Detecting anomalies in text via self-supervision of transformers,” in Proceedings of the 2021 conference of the North American chapter of the association for computational linguistics: Human language technologies, 2021, pp. 267–277.

[11] J. Pan, Y. Liu, Y. Zheng, L. Chi, A. W.-C. Liew, and S. Pan, “Camera: Adapting to semantic camouflage in unsupervised text-attributed graph fraud detection,” in International Joint Conference on Artificial Intelligence, 2026.

[12] Y. Cao, S. Yang, C. Li, H. Xiang, L. Qi, B. Liu, R. Li, and M. Liu, “Tadbench: A comprehensive benchmark for embedding-based text anomaly detection,” arXiv preprint arXiv:2501.11960, 2025.

[13] Y. Cao, S. Yang, Y. Yang, L. Qi, and M. Liu, “Text anomaly detection with simplified isolation kernel,” arXiv preprint arXiv:2510.13197, 2025.

[14] Y. Qian, Y. Tan, Y. Liu, W. Yu, and S. Pan, “Dynhd: Hallucination detection for diffusion large language models via denoising dynamics deviation learning,” in Findings of the Association for Computational Linguistics: EMNLP 2026, 2026.

[15] B. Chen, W. Wongso, X. Hu, Y. Tan, and F. D. Salim, “Multi-stage verification-centric framework for mitigating hallucination in multimodal rag,” in 2025 KDD Cup Workshop for Multimodal Retrieval Augmented Generation, 2025.

[16] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the associationfor computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[17] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “Roberta: A robustly optimized bert pretraining approach,” arXiv preprint arXiv:1907.11692, 2019.

[18] Y. Li, J. Li, Z. Xiao, T. Yang, Y. Nian, X. Hu, and Y. Zhao, “Nlp-adbench: Nlp anomaly detection benchmark,” arXiv preprint arXiv:2412.04784, 2024.

[19] A. S. Das, A. Ajay, S. Saha, and M. Bhuyan, “Few-shot anomaly detection in text with deviation learning,” in International Conference on Neural Information Processing. Springer, 2023, pp. 425–438.

[20] Y. Cao, B. Yu, S. Yang, M. Liu, and Y. Yang, “Towards token-level text anomaly detection,” in Proceedings of the ACM Web Conference 2026, 2026, pp. 8733–8736.

[21] J. Tu, H. Liu, and C. Li, “Weighted subspace anomaly detection in highdimensional space,” Pattern Recognition, vol. 146, p. 110056, 2024.

[22] H. Shi, J. GAO, H. Xu, X. Liang, Z. Li, L. Kong, S. Lee, and J. Kwok, “Revisiting over-smoothing in bert from the perspective of graph,” in International Conference on Learning Representations, 2022.

[23] Y. Cao, H. Xiang, H. Zhang, Y. Zhu, and K. M. Ting, “Anomaly detection based on isolation mechanisms: A survey,” Machine Intelligence Research, vol. 22, no. 5, pp. 849–865, 2025.

[24] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, “Lof: identifying density-based local outliers,” in Proceedings of the 2000 ACM SIGMOD international conference on Management of data, 2000, pp. 93–104.

[25] F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation forest,” in 2008 eighth ieee international conference on data mining. IEEE, 2008, pp. 413–422.

[26] Z. Li, Y. Zhao, X. Hu, N. Botta, C. Ionescu, and G. H. Chen, “Ecod: Unsupervised outlier detection using empirical cumulative distribution functions,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 12, pp. 12 181–12 193, 2022.

[27] C. Zhou and R. C. Paffenroth, “Anomaly detection with robust deep autoencoders,” in Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, 2017, pp. 665–674.

[28] L. Ruff, R. Vandermeulen, N. Goernitz, L. Deecke, S. A. Siddiqui, A. Binder, E. Muller, and M. Kloft, “Deep one-class classification,”¨ in International conference on machine learning. PMLR, 2018, pp. 4393–4402.

[29] A. Goodge, B. Hooi, S.-K. Ng, and W. S. Ng, “Lunar: Unifying local outlier detection methods via graph neural networks,” in Proceedings of the AAAI conference on artificial intelligence, vol. 36, no. 6, 2022, pp. 6737–6745.

[30] S. Li, Y. Zhao, Y. Tan, Q. Chen, Y. Liu, and S. Pan, “Towards anomaly detection on relational data,” arXiv preprint arXiv:2606.18621, 2026.

[31] Y. Zhao, Y. Liu, Q. Chen, S. Li, Y. Tan, and S. Pan, “Fedcigar: A personalized reconstruction approach for federated graph-level anomaly detection,” in IJCAI, 2026.

[32] Y. Liu, S. Li, Y. Zheng, Q. Chen, C. Zhang, P. S. Yu, and S. Pan, “From few-shot to zero-shot: Towards generalist graph anomaly detection,” IEEE Transactions on Knowledge and Data Engineering, 2026.

[33] S. Li, Y. Liu, Y. Zheng, M. Li, Q. V. H. Nguyen, and S. Pan, “Ofamas: One-for-all multi-agent system topology design based on mixtureof-experts graph generative models,” in Proceedings of the ACM Web Conference 2026, 2026, pp. 1333–1344.

[34] S. Li, Y. Liu, Y. Zheng, X. Cao, S. Pan, and H. T. Shen, “Towards onefor-all anomaly detection for tabular data,” in International Conference on Machine Learning, 2026.

[35] Y. Liu, K. Yan, S. Li, Q. Chen, and S. Pan, “Beyond a single perspective: Text anomaly detection with multi-view language representations,” in Joint European Conference on Machine Learning and Knowledge Discovery in Databases, 2026.

[36] J. Li, A. Sun, J. Han, and C. Li, “A survey on deep learning for named entity recognition,” IEEE transactions on knowledge and data engineering, vol. 34, no. 1, pp. 50–70, 2020.

[37] Y. Liu, Y. Liu, Y. Zheng, A. W.-C. Liew, X. Cao, and S. Pan, “Rethinking feature alignment in generalist graph anomaly detection: A relational fingerprint-based approach,” in International Conference on Machine Learning, 2026.

[38] X. Shen, Y. Liu, Y. Wang, R. Miao, Y. Dai, S. Pan, Y. Chang, and X. Wang, “Raising the bar in graph ood generalization: Invariant learning beyond explicit environment modeling,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[39] Y. Tan, C. Chen, W. Zhuang, X. Dong, L. Lyu, and G. Long, “Taming heterogeneity to deal with test-time shift in federated learning,” in International Workshop on Federated Learning for Distributed Data Mining, 2023.

[40] Y. Tan, G. Long, J. Jiang, and C. Zhang, “Influence-oriented personalized federated learning,” in IEEE International Conference on Data Mining, 2026.

[41] H. Xu, Y. Wang, S. Jian, Q. Liao, Y. Wang, and G. Pang, “Calibrated one-class classification for unsupervised time series anomaly detection,” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 11, pp. 5723–5736, 2024.

[42] H.-P. Kriegel, P. Kroger, E. Schubert, and A. Zimek, “Interpreting and unifying outlier scores,” in Proceedings of the 2011 SIAM International Conference on Data Mining. SIAM, 2011, pp. 13–24.

[43] G. Pang, C. Shen, and A. Van Den Hengel, “Deep anomaly detection with deviation networks,” in Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, 2019, pp. 353–362.