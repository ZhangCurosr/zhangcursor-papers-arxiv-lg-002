# TDDM-Melatt: A Decoupled Memory and Diffusion Framework for Generalizable Encrypted Traffic Classification

Ze Chen National University of Defense Technology Hefei, Anhui, China chenze@nudt.edu.cn

Qiming Yu   
National University of   
Defense Technology   
Hefei, Anhui, China   
yuqiming24@nudt.edu.cn   
Guozheng Yang\*   
National University of   
Defense Technology   
Hefei, Anhui, China   
yangguozheng17@nudt.edu.cn

Zijia Song National University of Defense Technology Hefei, Anhui, China songzijia@nudt.edu.cn

Wei Yan   
National University of   
Defense Technology   
Hefei, Anhui, China   
yan.wei2023@nudt.edu.cn

## Abstract

The widespread adoption of encrypted traffic poses severe challenges to current security situational awareness systems based on network traffic monitoring. In existing dataset-driven training and testing studies, limitations such as shortcut learning induced by spurious feature correlations and sample imbalance caused by the longtail distribution of real-world traffic result in weak generalization of traffic identification performance to real-world network traffic. To address these limitations, we propose TDDM-Melatt, a disentangled memory-based traffic classification framework with diffusion-based data augmentation. First, we design Melatt, a memory-decoupled traffic representation model, which employs Competitive Gating Long Short-Term Memory (CG-LSTM) to construct the encoder and decoder. We design a spurious-correlation-free pre-training and inference paradigm, employing strict topology anonymization and a frozen pre-trained encoder strategy to cut off the model's learning pathways for spurious features. During pre-training, computing the reconstruction loss between different traffic classes and each memory prototype forces the model to learn prototype-aligned representations. During inference, classification is performed efficiently by a downstream classifier on the frozen representations. Second, we propose a Traffic Denoising Diffusion Model (TDDM) tailored to the characteristics of traffic data. Addressing the structured nature of high-dimensional sparsity and strong feature correlations in traffic data, we design a lightweight residual noise prediction network and a compact noise sampling mechanism, solving the problem of having only class labels as conditional guidance for traffic diffusion models. Extensive experiments are conducted on 4 representative public benchmark datasets. Under strict flow-level splitting and anonymization, TDDM-Melatt outperforms 6 basic classification models and 6 SOTA representation learning models. The proposed method provides a new and effective technical pathway for encrypted traffic classification in real-world network environments.

• Security and privacy → Network security.

Keywords Network Security, Deep Learning, Encrypted Traffic Identification

## 1 Introduction

Object recognition in the human brain relies on prototype memories stored in the hippocampus, forming a closed-loop cognitive process of “memory storage-feature matching-reconstructive recognition." Visual information captured by the eyes is encoded into abstract features by the visual cortex and subsequently undergoes precise matching and reconstruction with memory prototypes via the hippocampus. High matching confidence leads to successful recognition, whereas low confidence results in classification as unknown or anomalous. This cognitive mechanism offers profound inspiration for network traffic inspection. If a “memory module" storing behavioral prototypes of traffic can be constructed, the reconstruction error naturally serves as a training objective when traffic of a given class cannot be accurately reconstructed by its own class-specific prototypes. Motivated by this insight, this paper focuses on encrypted traffic classification—a challenge in network security—and proposes a novel solution.

In recent years, with the increasing sophistication of cybersecurity threats and the strengthening of privacy protection regulations the adoption of encryption protocols in Internet communications has exceeded 80% [41]. While encryption effectively safeguards user privacy, it also introduces substantial technical obstacles to network traffic monitoring and security auditing. Traditional Deep Packet Inspection (DPI) [5, 14] techniques are rendered ineffective due to the inability to decrypt encrypted payloads, compelling security researchers to shift toward deep learning approaches.

Explicit identifiers commonly retained in public datasets [13, 30, 49, 61], such as IP, Port, and flow IDs, exhibit strong yet non-causal statistical correlations with traffic labels. Models tend to prioritize learning such “shortcut features”. This reliance on spurious correlations severely degrades generalization capability, manifesting as inflated accuracy on public test sets followed by precipitous performance drops upon real-world deployment [64, 72]. Furthermore, most studies [30, 75] adopt packet-level sample partitioning, whereby packets from the same flow are distributed across both training and test sets. This practice causes models to learn flow-specific idiosyncrasies instead of class-generalizable patterns further amplifying the detrimental effects of spurious correlations. In the context of fine-tuning traffic classification models, high accuracy on encrypted datasets has led to the controversial claim that pre-training enables models to extract patterns from encrypted payloads.

In real-world network environments, traffic exhibits highly imbalanced class distributions. Extreme data skewness causes models to be dominated by majority classes during training. Generative Adversarial Networks (GANs) [1, 12, 40] have been explored to mitigate data imbalance. However, GANs are prone to training instability and mode collapse. Diffusion models, as an emerging class of generative models, have achieved remarkable success in image synthesis [11, 22, 66], yet their application to network traffic remains in its infancy and struggles to accommodate the structured nature of traffic data.

To address the aforementioned limitations, we propose TDDM-Melatt, a decoupled memory and diffusion framework for generalizable encrypted traffic classification. The main contributions are summarized as follows:

• We propose Melatt, a modular traffic memory model. The architecture comprises a CG-LSTM-based encoder and a memory-augmented multi-head cross-attention decoder, enhancing the capacity to extract and reconstruct dynamic traffic information.

• We establish a pre-training and inference paradigm free of spurious correlations. During data preprocessing, explicit flow identifiers are rigorously removed, and timestamp information is generalized. Freezing the parameters of the pre-trained encoder during downstream classification tasks ensures the cross-scenario generalization capability in real network environments.

• We design TDDM, a traffic denoising diffusion model tailored to the structural characteristics of traffic data. In light of the structured characteristics of traffic data, we construct a lightweight diffusion generation architecture that achieves a reduction of more than 30% in the dimensionality of the features. By adopting a class-wise training paradigm, we propose a condition-guided compact noise sampling mechanism to improve class convergence during generation.

• We conduct comprehensive experimental evaluations on 4 public datasets. Under strict flow-level sample partitioning and topology anonymization, TDDM-Melatt outperforms 6 baseline classifiers and 6 SOTA models.

## 2 Problem Statement

Encrypted traffic classification constitutes a fundamental capability within cyberspace security defense frameworks, and its technological evolution is deeply intertwined with the upgrade of encryption protocols, the mutation of network attack patterns, and the shifting landscape of regulatory requirements. With the widespread deployment of next-generation encryption protocols such as TLS 1.3 and QUIC, traditional payload-based DPI [5, 14] techniques have become increasingly ineffective, prompting a pivot toward representation learning-based classification methods as Table 1. ET-BERT [30], YaTC [71], NetMamba [60], and TrafficFormer [75] have achieved improved classification performance. Nevertheless, when migrating from laboratory settings to real-world environments, existing approaches frequently encounter precipitous performance degradation.

Table 1: Comparison of the Traffic Analysis Methods.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">ML Models</td><td rowspan="2">Modality</td><td colspan="2">Detection Abilities</td></tr><tr><td>Encrypted</td><td>Non IP/Port</td></tr><tr><td>netFound</td><td>BERT</td><td>Flow</td><td>√</td><td>√</td></tr><tr><td>YaTC</td><td>ViT</td><td>Flow</td><td>√</td><td>√</td></tr><tr><td>NetMamba</td><td>Mamba</td><td>Flow</td><td>√</td><td>X</td></tr><tr><td>ET-BERT</td><td>BERT</td><td>Packet</td><td>√</td><td>√</td></tr><tr><td>TrafficFormer</td><td>BERT</td><td>Packet</td><td>√</td><td>X</td></tr><tr><td>Pcap-Encoder</td><td>T5</td><td>Flow</td><td>√</td><td>X</td></tr></table>

The prevailing research paradigm in encrypted traffic classification relies on supervised learning over labeled datasets, implicitly assuming the existence of stable causal relationships between features and labels. This assumption, however, engenders a severe “shortcut learning" phenomenon, which has become a critical bottleneck restricting model generalization. Public datasets routinely retain explicit identifiers such as IP, Port, and flow IDs. These features exhibit strong yet non-causal correlations with traffic labels [2, 45]. For instance, specific applications often utilize well-known ports, and particular attack traffic frequently originates from specific IP prefixes. During training, supervised models preferentially learn these “low-cost, high-discriminability"shortcut features. Such reliance on spurious correlations yields outstanding performance on test sets drawn from the same distribution as the training data [64, 72], yet once deployed in real networks where IP and Port diverge, model performance collapses dramatically. The packetlevel sample partitioning commonly adopted in existing studies [30, 75] further amplifies the detrimental effects of spurious correlations. Several works split distinct packets belonging to the same flow across training and test sets, thereby enabling models to learn flow-specific idiosyncrasies. This partitioning scheme, which inherently involves data leakage, leads to severe overestimation of actual model performance in experimental evaluations.

More critically, full fine-tuning (unfrozen encoder) of existing representation learning models severely erodes the generic features acquired during pre-training [27]. When deployment environments shift, the model forfeits both the generalization benefits of pretrained knowledge and becomes excessively reliant on training-setspecific shortcut features, ultimately resulting in sharp performance attenuation in real-world network contexts. This phenomenon underscores a fundamental tension between theoretical innovation and practical deployment in encrypted traffic classification.

Another major challenge is the long-tail distribution of traffic in real network environments [56, 74]. In operational networks, benign traffic and flows from popular applications overwhelmingly dominate the total volume. In contrast, malicious traffic and niche applications flows are not only scarce in aggregate but also exhibit sample counts that may differ by several orders of magnitude across categories. Such extreme imbalance causes models to be dominated by majority class samples during training, leading to decision boundaries severely skewed against minority classes [38]. In pursuit of high overall accuracy, models tend to classify samples as majority classes, resulting in unacceptably low recognition rates for minority classes. In cybersecurity defense scenarios, however, it is precisely these exceedingly rare minority samples that carry the highest security value [53].

To mitigate the scarcity of minority class samples [17], data augmentation techniques have been extensively explored in encrypted traffic classification. In recent years, GANs [1, 12, 40] have been applied to address data imbalance. However, GANs are plagued by training instability and mode collapse. Diffusion models, as an emerging class of generative models, have achieved breakthrough successes in image synthesis and natural language processing [11, 22, 66], offering a promising new avenue for traffic data augmentation. Nonetheless, the application of diffusion models to network traffic remains in its infancy, and existing efforts have merely performed a straightforward model transfer by converting traffic data into images. Moreover, existing traffic classification models [8, 10, 28, 30, 44, 55, 60, 71, 75] are typically designed with a deep coupling among core components such as the encoder, the pretrained backbone, and the classification head. When transitioning across tasks, extensive architectural redesign and model retraining are often required, incurring prohibitively high development and tuning costs [42]. This “one-task-one-model” development paradigm impedes the iterative evolution of traffic classification systems and hinders their capacity to adapt to rapidly evolving cybersecurity threats [6].

This section systematically delineates major challenges confronting contemporary traffic classification research: (1) the trap of shortcut learning and spurious correlations under the supervised learning paradigm; (2) the class imbalance of real-world network traffic and the limitations of existing data augmentation methods; (3) the scenario-binding nature of model architectures and deficiencies in cross-task adaptability.

In light of the aforementioned challenges, the core research question of our work is formulated as follows: How can we construct an encrypted traffic classification framework that is capable of escaping spurious correlations, accommodating extreme data distributions, supporting rapid cross-scenario adaptation, and generating high-fidelity traffic samples?To address this question, we draw inspiration from the cognitive mechanisms of the human brain and propose a decoupled memory and diffusion framework for generalizable encrypted traffic classification.

## 3 Methodology Overview

The overall framework proposed in our work is illustrated in Figure 1(a). It comprises three core components: a memory representation model termed Melatt, a traffic data augmentation module based on denoising diffusion probabilistic models termed TDDM, and a downstream task classifier. The traffic memory representation model, which serves as a pre-training architecture, consists of an encoder, a memory module, and a decoder. This model performs a sequence of operations on input traffic features, including encoding, memory bank addressing, and reconstruction decoding The conditional diffusion augmentation module functions as an independent generative component capable of synthesizing highfidelity minority class samples conditioned on specified class labels. The downstream task classifier is designed as a flexible and extensible classification engine. Embracing a modular design philosophy, it supports seamless transitions from lightweight traditional models to complex deep architectures, thereby accommodating diverse deployment scenarios (e.g., resource-constrained edge devices or high-performance servers). To highlight the decoupled nature of the framework, and unless otherwise specified, the classifier is implemented using a straightforward XGBoost [7] model.

We construct an independent external memory module, the Melatt architecture illustrated in Figure 1(b). Emulating the cognitive process of “perception-memory-decision” in the human brain, this module realizes the decoupling of feature extraction and knowledge storage. The module consists of multiple learnable memory matrices and a content-based addressing mechanism [20]. Each memory slot corresponds to the prototype features of a specific traffic class, supporting independent pre-training and flexible insertion or deletion. To overcome the prevalent “fuzzy memory” problem in neural networks, we introduce a negative entropy-based regularization constraint that enforces sharpening of the attention distribution. To address the dual nature of network traffic characterized by both periodicity and burstiness, we design a Competitive Gating Long Short-Term Memory (CG-LSTM) in both the encoder and the decoder. By placing its 3 gates within a shared softmax competitive framework, CG-LSTM forces the model to allocate a finite total “attentional budget” at each time step, thereby enhancing its discriminative capacity for dynamic traffic patterns. Furthermore, we integrate a multi-head cross-attention mechanism [8, 34, 46] within the decoder, which enables dynamic querying of multiple prototype patterns stored in the memory module at each decoding step, ultimately generating more accurate reconstructed sequences.

Traffic data is characterized by high dimensionality, sparsity and strong feature interdependencies [59], while generic generative models often suffer from distribution drift and feature distortion when applied to this domain. To address these issues, we propose TDDM, a denoising diffusion model specifically tailored for structured traffic data. First, we design a feature selection method based on Random Forest Gini importance. This method prunes redundant features that exhibit high zero-value ratios or low classification contributions, achieving a dimensionality reduction of over 30% while preserving 95% of the critical information. Second, we eschew the complex convolutional and attention-based architectures commonly employed in image diffusion models. Instead, we adopt a lightweight MLP-based residual network as the noise predictor and adopt a class-wise training paradigm to avoid unnecessary cross-modal computational overhead. To further enhance the distributional consistency of generated samples, we propose a compact noise sampling mechanism. Finally, a nearest-neighbor feature padding strategy is employed to restore the complete feature dimensionality by leveraging the manifold consistency in the low-dimensional latent space.

## 4 Methodological Details

## 4.1 Encoder and Decoder

Encoder Based on Competitive Gating LSTM.

![](images/b05ef0640631257db80db31ae63643b3b06dc87b50d5d07434ef5e2563bbe11a.jpg)  
Figure 1: Framework Overview.

Traditional LSTM networks [19, 51] employ 3 independent sigmoidal gating mechanisms, allowing the forget gate, input gate, and output gate to simultaneously approach unity. This results in a lack of selectivity in the information flow through the memory cell, making it difficult to decisively switch between bursty traffic and quiescent background states. To address these challenges, we design a CG-LSTM, the structure of which is depicted in Figure 2.

![](images/0862893b511fac55ad8a6180134d19e034fde46b3ee26cfd9c3baa1e994bcdcf.jpg)  
Figure 2: Structure of CG-LSTM.

In conventional LSTM networks [19, 23, 51], the forget gate $f _ { t } ,$ input gate $i _ { t } ,$ and output gate $o _ { t }$ are each computed via independent sigmoidal activation functions. Consequently, all 3 gating values may simultaneously approach unity, leading to scenarios where the model erases substantial portions of historical information while concurrently writing new content, thereby inducing memory confusion. The CG-LSTM replaces the independent sigmoidal activations with a joint softmax operation, thereby constraining 3 gates to form a probability distribution at each time step:

$$
[ f _ { t } , i _ { t } , o _ { t } ] = \mathrm { s o f t m a x } \left( W _ { g } [ h _ { t - 1 } ; \boldsymbol { x } _ { t } ] + b _ { g } \right) ,\tag{1}
$$

where $W _ { q } \in \mathbb { R } ^ { 3 \times ( d _ { h } + d _ { x } ) }$ hidden state of previous step $\pmb { h } _ { t - 1 }$ , traffic vector $x _ { t } ,$ and $b _ { q } \in \mathbb { R } ^ { 3 }$ . This formulation enforces the following constraints:

$$
f _ { t } + i _ { t } + o _ { t } = 1 , \quad f _ { t } , i _ { t } , o _ { t } \geq 0 .\tag{2}
$$

Under this design, the model allocates a finite “gating resource budget” across the forget, input, and output operations at each time step.

When encountering a traffic burst, the model may allocate the majority of its budget to the input gate it to rapidly incorporate novel features. During quiescent periods, the budget is preferentially assigned to the forget gate $f _ { t }$ to maintain the baseline representation of benign traffic. When the model needs to propagate long-range periodic patterns, the budget is concentrated on the output gate $o _ { t } ,$ enabling high-fidelity propagation of the memory state. Define the element-wise multiplication as ©.The candidate memory cell state $\tilde { C } _ { t } = \operatorname { t a n h } \left( W _ { c } [ \boldsymbol { h } _ { t - 1 } ; \boldsymbol { x } _ { t } ] + \boldsymbol { b } _ { c } \right)$ . The update equations for the CG-LSTM cell state $C _ { t }$ and hidden state $\pmb { h } _ { t }$ are given by:

$$
C _ { t } = f _ { t } \odot C _ { t - 1 } + i _ { t } \odot \tilde { C } _ { t } , \quad h _ { t } = o _ { t } \odot \operatorname { t a n h } ( C _ { t } ) .\tag{3}
$$

Compared with standard LSTM, CG-LSTM shares a single weight matrix across 3 gates, reducing the parameter count by approximately one-third. Furthermore, the extreme values (0 or 1) produced by the softmax output are more conducive to gradient propagation than the intermediate values typically output by sigmoidal functions, thereby alleviating the vanishing gradient problem in long sequences. When minority class traffic appears, the model is compelled to shift its gating budget from the forget gate toward the input gate in order to record novel features. This mechanism inherently amplifies the learning weight assigned to minority class samples, thereby enhancing robustness in class-imbalanced scenarios.

The encoder is constructed by stacking multiple CG-LSTM layers. For an input traffic feature sequence $\tilde { X } = \left\{ x _ { 1 } , x _ { 2 } , \ldots , x _ { T } \right\} ( T$ denotes the sequence length), the encoder iteratively updates the cell state and hidden state at each time step according to Equation (3). We adopt the hidden state $h _ { T }$ from the final time step as the global temporal fingerprint of the traffic sequence. This fingerprint is subsequently transformed via a linear projection to yield a latent vector of dimension L. A batch normalization layer and a LeakyReLU activation function are appended to the encoder output Ultimately, the encoder produces a latent vector $\boldsymbol { z } _ { t } \in \mathbb { R } ^ { L }$ , which serves dually as the initial state for the decoder and as the query vector for the subsequent memory module.

Decoder Based on Cross-Attention .

The decoder takes the memory vector $z _ { m }$ output by the memory module as its initial state. It employs a multi-layer stack of CG-LSTM units as its backbone and incorporates a multi-head cross-attention mechanism [57] to facilitate the fusion of global memory information with locally generated features. The overall architecture of the decoder comprises a CG-LSTM module, a multi-head cross-attention module, a residual connection, and layer normalization module and a fully connected reconstruction layer.

The CG-LSTM structure within the decoder is identical to that of the encoder. The multi-head cross-attention module serves as the core enhancement component of the decoder. By dynamically retrieving canonical traffic patterns from the global memory matrix, it mitigates the information closure problem. In this module, the intermediate feature sequence output by the CG-LSTM serves as the Query, representing the local temporal characteristics at the current generation step. The external memory matrix $\pmb { M } \in \mathbb { R } ^ { N \times L }$ serves simultaneously as both the Key and the Value, where N denotes the number of memory slots and L denotes the dimension of each memory slot. A multi-head attention mechanism [8, 34, 46] is employed to concurrently retrieve the most pertinent global prototype patterns from multiple distinct feature subspaces conditioned on the current local features:

$$
\mathrm { M u l t i H e a d } ( Q , M , M ) = \mathrm { C o n c a t } ( \mathrm { h e a d } _ { 1 } , . . . , \mathrm { h e a d } _ { H } ) W ^ { O } ,\tag{4}
$$

$$
\mathbf { h e a d } _ { i } = \mathbf { A t t e n t i o n } ( Q W _ { i } ^ { Q } , M W _ { i } ^ { K } , M W _ { i } ^ { V } ) .\tag{5}
$$

where $W ^ { O }$ is the learnable projection matrix for the output.

$$
\mathrm { A t t e n t i o n } ( Q , K , V ) = \mathrm { s o f t m a x } \left( { \frac { Q K ^ { T } } { \sqrt { d _ { k } } } } \right) V .\tag{6}
$$

The outputs of multiple attention heads are concatenated and subsequently transformed via a linear projection to yield the attentionaugmented feature sequence. To alleviate the vanishing gradient problem during the training of deep neural networks, residual connections and layer normalization are introduced after the multihead cross-attention module. These mechanisms fuse the original CG-LSTM output with the attention-enhanced features. Finally, a fully connected layer maps the fused feature sequence back to the original D-dimensional feature space of the input, producing the final reconstructed traffic sequence.

## 4.2 Memory Model and Pre-training Constraints Memory Module.

The Melatt emulates the cognitive process of“perception-memorydecision" in the human brain by introducing an independent external memory unit, thereby realizing the decoupling of feature extraction and knowledge storage. This liberates the encoder from the burdensome task of parameter storage, allowing it to concentrate exclusively on extracting temporal features, while long-term knowledge retention is delegated to the external memory module.

The memory module constitutes the core component of Melatt. It consists of multiple learnable memory matrices $\pmb { M } \in \mathbb { R } ^ { N \times L }$ and a content-based addressing mechanism. Each M comprises N memory slots, each corresponding to a prototype feature of a target traffic class. These memory slots are iteratively optimized and updated during the pre-training process. Notably, each memory matrix can be pre-trained independently, and different pre-trained memory matrices can be flexibly inserted into or removed from the memory bank, provided that their dimensionality remains consistent.

Given the latent vector $z _ { i }$ output by the encoder, the memory module computes its similarity with all memory slots via a contentbased addressing mechanism. The relevance score e for the memory slots is expressed as $\pmb { e } = z _ { i } \cdot \pmb { M } ^ { T } / \sqrt { L }$ . The attention weight $\omega = \mathrm { S o f t m a x } ( e )$ is computed via the softmax function, where the weight represents the “memory strength" of the input traffic with respect to each memory prototype. To address the prevalent “fuzzy memory” problem in neural networks (requiring that a given traffic flow correspond unambiguously to a single distinct memory prototype), we introduce a regularization constraint based on negative entropy. During training, the entropy of the attention distribution is minimized:

$$
\mathcal { L } _ { \mathrm { e n t } } = - \sum \omega _ { i } \log \omega _ { i } .\tag{7}
$$

This constraint enforces the attention weight distribution to become sharper and sparser, compelling the model to match a specific memory prototype with high precision during reconstruction. Ultimately, the model reconstructs the sequence using the attention-weighted memory content $z _ { m } = \omega M ,$ yielding the encoder output.

## Loss Function.

The efficacy of Melatt is highly contingent upon the constraints imposed during the pre-training phase. A conventional, singular reconstruction error criterion is insufficient to guarantee that the latent space acquires structured semantic information. To this end, we devise 4 pre-training constraints.

Reconstruction Loss. We employ the Smooth L1 Loss as the reconstruction loss to mitigate the influence of outliers on gradient updates. Compared with the traditional MSE loss, Smooth L1 exhibits L1 robustness in large error regions, making it less susceptible to domination by outliers, and smoothly transitions to L2 loss in small error regions, thereby ensuring convergence stability.

$$
\mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { S m o o t h L } 1 ( \tilde { X } _ { i } , \tilde { X } _ { m i } ) ,\tag{8}
$$

where ${ \tilde { X } } _ { m }$ denotes the traffic reconstructed by the decoder.

Compactness Constraint. To prevent the latent vector $z _ { i }$ output by the encoder from deviating substantially from the memory readout $z _ { m } .$ we introduce a compactness constraint. This constraint enforces the feature vectors in the latent space to closely adhere to the memory prototypes, thereby enhancing the continuity of the feature manifold:

$$
\mathcal { L } _ { \mathrm { c o m p } } = \mathbb { E } \| z _ { i } - z _ { m } \| _ { 2 } ^ { 2 } .\tag{9}
$$

By minimizing the distance between the latent vector and the memory readout vector, the model is encouraged to encode inputs into a subspace consistent with the memory prototypes, thus preventing excessive dispersion of latent features.

Entropy Regularization and Diversity Constraints. Equation (7) prescribes a negative entropy-based regularization constraint, $\mathcal { L } _ { \mathrm { e n t } }$ , which compels the model to match a specific memory prototype with high precision during traffic reconstruction.

Concurrently, redundant prototype features may appear in the memory bank, where multiple memory slots learn similar or identical feature patterns, causing “mode collapse". To prevent this, we maximize the distance between any pair of distinct memory slots $\mathbf { \nabla } m _ { i }$ and $\mathbf { \delta } _ { m _ { j } }$ . This enforces each memory slot to learn a unique and distinguishable traffic pattern, thereby augmenting the capacity and richness of feature representations:

$$
\mathcal { L } _ { \mathrm { d i v } } = - \frac { 1 } { N ( N - 1 ) } \sum _ { i \neq j } \| \pmb { m } _ { i } - \pmb { m } _ { j } \| _ { 2 } ^ { 2 } .\tag{10}
$$

The negative sign ensures that minimizing ${ \mathcal { L } } _ { \mathrm { d i v } }$ is equivalent to maximizing the pairwise distances among memory slots.

Total Loss. The aggregate loss function during the pre-training phase is formulated as a weighted sum of the losses defined in Equations (7), (8), (9), and (10). In practice, while the reconstruction loss retains its original scale (weight of 1), we set the weighting coefficients as $\alpha _ { \mathrm { c o m p } } = 0 . 0 5 , \alpha _ { \mathrm { e n t } } = 0 . 0 5$ , and $\alpha _ { \mathrm { d i v } } = 0 . 0 0 0 5$ This configuration yielded favorable outcomes across all our experiments.

It is important to emphasize that the nearest-prototype matching paradigm, driven by $\scriptstyle { \mathcal { L } } _ { \mathrm { r e c } } ,$ is the pre-training objective. During inference, the decoder outputs the reconstructed feature vector $\tilde { \mathbf { X } } _ { m i } \in \mathbb { R } ^ { D }$ , which is concatenated with the encoder's latent vector $\mathbf { z } _ { i } \in \mathbb { R } ^ { L }$ to form a compact representation $[ \tilde { \mathbf { X } } _ { m i } ; \mathbf { z } _ { i } ]$ . This representation is provided to downstream classifier for final classification. The pre-training process ensures that $\tilde { \mathbf { X } } _ { m i }$ inherently encodes the quality of prototype matching, making the inference pipeline efficient and decoupled from the memory module's per-prototype computational overhead.

## 4.3 Feature Selection and Padding

Preprocessed Flow-Level Statistical Features. We extract a total of 76 flow-level statistical features from each bidirectional flow, all derived exclusively from packet headers (IP and TCP/UDP layers) without accessing any encrypted payload. These features span five functional categories: length/size statistics, temporal and IAT features, packet/byte count features, rate features, and flag/initialization features. The complete list of all 76 features, organized by category, is provided in Appendix Table 8.

## Feature Selection Based on Gini Importance.

Traffic data differs fundamentally from image data [59], the domain in which diffusion models have been most extensively applied A substantial proportion of traffic features assume zero values, as summarized in Table 2. The table presents, for each of the four datasets, the fraction of features whose zero proportion exceeds the threshold γ. In each dataset, over 10% of the features are identically zero across all samples. A higher zero-value ratio for a given feature indicates a greater prevalence of zero entries across all classes within the dataset. This phenomenon may correspond to one of two scenarios: (1) the feature constitutes a distinguishing characteristic of a specific traffic class relative to others; (2) the feature exhibits consistent behavior across different classes, manifesting as a uniformly high zero-value ratio. In the latter case, the feature conveys negligible information, contributes virtually nothing to downstream tasks, and simultaneously degrades the training efficiency of the diffusion model while increasing the risk of overfitting.

To address this issue, we introduce a feature selection method based on Random Forest importance scores. Specifically, a Random Forest classifier is first trained on the training partition designated for the diffusion model. The Gini Importance [73] of each feature is subsequently computed, which quantifies the contribution of a feature to classification performance via the mean reduction in node impurity. The Gini impurity at a given node is defined as: Gin $\begin{array} { r } { \mathrm { i } ( i ) = 1 - \sum _ { k } p _ { k } ^ { 2 } } \end{array}$ where $\scriptstyle { \mathcal { P } } k$ denotes the proportion of samples belonging to the k-th class at forest node i. For a node with $n _ { i }$ total samples, the reduction in Gini impurity resulting from a split is given by:

Table 2: Statistics of Feature Zero-Value Ratios
<table><tr><td>Dataset</td><td> $\gamma = 1 0 0 \%$ </td><td> $\gamma > 9 5 \%$ </td><td> $\gamma > 9 0 \%$ </td><td> $\gamma > 8 0 \%$ </td></tr><tr><td>CIC-IDS-2017</td><td>10.39%</td><td>19.48%</td><td>23.38%</td><td>27.27%</td></tr><tr><td>ISCX-VPN-2016</td><td>11.84%</td><td>27.63%</td><td>46.05%</td><td>64.47%</td></tr><tr><td>USTC-TFC2016</td><td>11.84%</td><td>18.42%</td><td>21.05%</td><td>38.16%</td></tr><tr><td>CSTNET-TLS1.3</td><td>10.53%</td><td>15.79%</td><td>19.74%</td><td>21.05%</td></tr></table>

$$
\Delta \mathrm { G i n i } = \mathrm { G i n i } ( i ) - \frac { n _ { \mathrm { l e f t } } } { n _ { i } } \mathrm { G i n i } ( i _ { \mathrm { l e f t } } ) - \frac { n _ { \mathrm { r i g h t } } } { n _ { i } } \mathrm { G i n i } ( i _ { \mathrm { r i g h t } } ) .\tag{11}
$$

After normalization, the Gini Importance for each feature is obtained. Features are then ranked in descending order of their importance. We retain the minimal subset of features whose cumulative importance reaches a predefined threshold, while the remaining features are deemed low-contribution and are consequently pruned.

![](images/120de2f4e4359124f9615ed8d4f41df5c1ce3fac05e0099be878cfc61e59cc44.jpg)  
Figure 3: Feature Importance Cumulative Curves.

As illustrated in Figure 3, experimental results across 8 tasks spanning 4 datasets demonstrate that setting the cumulative importance threshold to 95% yields a reduction in feature dimension exceeding 30%. We present the feature importance bar charts for 4 tasks in Appendix Figure 12.

Features that are pruned are not simply discarded. Instead, after the diffusion model synthesizes samples in the reduced feature space, we restore the full feature dimension using a nearestneighbor padding strategy.

## Feature Padding Based on Nearest-Neighbor.

After feature selection, the diffusion model is trained and generates samples exclusively within the important feature subspace $\mathbb { R } ^ { d }$ However, downstream classification tasks require the complete D-dimensional feature vectors. Directly discarding the pruned features leads to inevitable information loss, whereas naive random padding disrupts the intrinsic correlations among features, thereby inducing uncontrolled deviation between the distribution of generated samples and that of real data. To address this issue, we propose a nearest-neighbor-based padding strategy. The key insight is that the manifold structure of feature vectors in the original high-dimensional space is approximately preserved under the lowdimensional projection. Consequently, if two samples are proximate in the low-dimensional space, they should also exhibit similar values for the pruned features in the full-dimensional space.

Let $X _ { c } ^ { \mathrm { f u l l } } \in \mathbb { R } ^ { n _ { c } \times D }$ denote the matrix of authentic data belonging to class c. Its low-dimensional projection is given by: $X _ { c } ^ { \mathrm { l o w } } = X _ { c } ^ { \mathrm { \bar { f u l l } } } [ :$ $\mathcal { K } ] \in \mathbb { R } ^ { n _ { c } \times d }$ , where K denotes the index set of retained features. For a generated low-dimensional sample $\tilde { \pmb y } \in \mathbb R ^ { d }$ , we define the index of its nearest neighbor as:

$$
i ^ { * } = \arg \operatorname* { m i n } _ { i \in \{ 1 , \ldots , n _ { c } \} } \left\| \tilde { \pmb { y } } - \pmb { X } _ { c } ^ { \mathrm { l o w } } [ i , : ] \right\| _ { 2 } .\tag{12}
$$

The complete generated sample $\tilde { \boldsymbol { x } } \in \mathbb { R } ^ { D }$ is then constructed as: $\tilde { \boldsymbol { x } } [ \mathcal { K } ] = \bar { \boldsymbol { y } } , \quad \tilde { \boldsymbol { x } } [ \mathcal { D } ] = X _ { c } ^ { \mathrm { f u l l } } [ i ^ { * } , \mathcal { D } ]$ , where D denotes the index set of pruned features.

## 4.4 Traffic Feature Generation via Diffusion

The overall architecture of our proposed TDDM, is built upon the denoising diffusion probabilistic framework. It comprises two core modules: the forward noise-adding diffusion process and the reverse denoising network [22, 54]. TDDM is specifically tailored to accommodate the characteristics of traffic feature data, thereby addressing the generative requirements of non-image, structured traffic data. The detailed structure is illustrated in Figure 1(c) and Figure 4.

![](images/bcfd7c08f332a1e6fecfad725c4e8c66a895f568f6868a55c330a4af07a83f9e.jpg)  
Figure 4: Details of TDDM Architecture.

Traffic feature data differs intrinsically from image data. In the image domain, an encoder is typically employed to map pixel-space representations into a lower-dimensional latent space to reduce the computational burden of diffusion models. In contrast, traffic features are inherently structured and comparatively low in volume; thus, a similar dimensionality reduction mapping is unnecessary, and diffusion modeling can be performed directly in the original feature space [18]. Given a feature sample $\pmb { x } _ { 0 } \sim q ( \pmb { x } _ { 0 } )$ belonging to class c, the forward diffusion process progressively injects Gaussian noise over T steps, yielding a sequence of increasingly noisy latent variables. The transition probability at each step is defined as:

$$
q ( \pmb { x } _ { t } \ | \ \pmb { x } _ { t - 1 } ) = N ( \pmb { x } _ { t } ; \sqrt { 1 - \beta _ { t } } \ \pmb { x } _ { t - 1 } , \beta _ { t } \mathbf { I } ) ,\tag{13}
$$

where $\beta _ { t } \in ( 0 , 1 )$ represents a predefined noise schedule. We adopt a linear schedule defined by $\begin{array} { r } { \beta _ { t } = \beta _ { \mathrm { s t a r t } } + \frac { t - 1 } { T - 1 } ( \beta _ { \mathrm { e n d } } - \beta _ { \mathrm { s t a r t } } ) } \end{array}$ . In our experiments, we set $\beta _ { \mathrm { s t a r t } } = 1 0 ^ { - 4 }$ and $\beta _ { \mathrm { e n d } } = 0 . 0 2$ Leveraging the reparameterization trick, the noisy sample at an arbitrary timestep t can be directly expressed as: ${ \pmb x } _ { t } = \sqrt { \bar { \alpha } _ { t } } { \pmb x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } { \pmb \varepsilon } $ where $\varepsilon \sim { \cal N } ( 0 , \mathbf { I } ) , \alpha _ { t } = 1 - \beta _ { t }$ , and $\textstyle { \bar { \alpha } } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i }$

We eschew complex convolutional and attention-based architectures for the noise prediction network and instead adopt a lightweight structure for noise estimation and denoising. The network takes as input the current noisy sample $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { d }$ and a time-step embedding temb $\in \mathbb { R } ^ { 1 2 8 }$ produced by a two-layer linear projection. The main network body consists of an input MLP layer $( d  5 1 2 )$ five cascaded residual blocks, and an output MLP layer $( 5 1 2  d )$ Each residual block contains two fully connected layers and integrates the time-step embedding and the residual path via a linear layer. The diffusion model is trained according to Algorithm 1, minimizing the following simplified loss function:

$$
\mathcal { L } _ { \mathrm { s i m p l e } } = \mathbb { E } _ { t , x _ { 0 } , \varepsilon } \left[ \| \varepsilon - \varepsilon _ { \theta } ( x _ { t } , t ) \| ^ { 2 } \right] .\tag{14}
$$

Algorithm 1 Class-Wise Diffusion Model Training   
Input: Low-dimensional features Xc of class c, number of diffusion   
steps T, learning rate η, batch size B, patience p   
Output: Trained noise prediction network εθ   
1: Initialize network parameters θ   
2: repeat   
3: Sample a mini-batch $\{ \pmb { x } _ { 0 } ^ { ( i ) } \} _ { i = 1 } ^ { B }$ randomly from $X _ { c }$   
4: for each sample i in the batch do   
5: Sample timestep t ∼ Uniform( $\left. 1 , . . . , T \right.$   
6: Sample noise $\pmb { \varepsilon } ^ { ( i ) } \sim { \cal N } ( 0 , \mathbf { I } )$   
7: Compute $\pmb { x } _ { t } ^ { ( i ) } = \sqrt { \bar { \alpha } _ { t } } \pmb { x } _ { 0 } ^ { ( i ) } + \sqrt { 1 - \bar { \alpha } _ { t } } \pmb { \varepsilon } ^ { ( i ) }$   
8: end for   
9: Compute loss $\begin{array} { r } { \mathcal { L } _ { \mathrm { s i m p l e } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left\| \pmb { \varepsilon } ^ { ( i ) } - \pmb { \varepsilon } _ { \theta } ( \pmb { x } _ { t } ^ { ( i ) } , t ) \right\| ^ { 2 } } \end{array}$   
10: Update θ using gradient descent   
11: until validation loss does not decrease for p consecutive epochs   
12: return εθ

## 4.5 Condition-Guided Compact Noise Sampling

In Algorithm 1, we adopt a class-wise training paradigm for the noise prediction network rather than a joint training scheme. Classical text-guided diffusion models require the introduction of crossmodal cross-attention mechanisms and rely on pre-trained text encoders to establish alignment between textual descriptions and generated content. In contrast, the diffusion model oriented toward traffic generation only utilizes class labels as conditional guidance. The limited semantic information inherent in class labels is insufficient to support effective training of a text encoder, while simultaneously leading to inefficient utilization of computational resources.

Nevertheless, a diffusion generative model that lacks conditional guidance, even when trained independently per class, still cannot avoid the drift of synthetic data away from the real distribution. To address this issue, we design a centroid-guided compact sampling mechanism during the denoising phase to enhance the distributional consistency of the generated samples.

The core idea of the compact sampling mechanism is to guide the samples toward the authentic data distribution of their corresponding class at each reverse denoising step. Since feature selection has already been performed during the forward diffusion process, and the compact sampling directly operates on the predicted noise this method exerts only a minor influence on the diversity of the generated data. Let the feature centroid of class c be defined as: $\begin{array} { r } { \pmb { \mu _ { c } } = \frac { 1 } { n _ { c } } \sum _ { i = 1 } ^ { n _ { c } } \pmb { x _ { i } ^ { ( c ) } } } \end{array}$ . During sampling, starting from $\mathbf { \boldsymbol { x } } _ { T } \sim \mathcal { N } ( \mathbf { \boldsymbol { 0 } } , \mathbf { \boldsymbol { I } } )$ and proceeding in reverse temporal order, the noise prediction network first estimates the current noise $\hat { \pmb { \varepsilon } } = \pmb { \varepsilon } _ { \theta } ( \pmb { x } _ { t } , t )$ . Subsequently, a guidance correction term is introduced:

$$
\hat { \pmb { \varepsilon } } ^ { \prime } = \hat { \pmb { \varepsilon } } - \gamma \cdot \left( 1 - \bar { \alpha } _ { t } \right) ^ { \frac { 1 } { 2 } } \big ( \pmb { x } _ { t } - \pmb { \mu } _ { c } \big ) ,\tag{15}
$$

where $\gamma \geq 0$ is a hyperparameter that controls the guidance strength (we set $\gamma = 0 . 3$ in our experiments). The corrected noise prediction steers the update direction toward the real sample distribution thereby yielding more compactly distributed samples. The sample update formula is given by:

$$
\pmb { x } _ { t - 1 } = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( \pmb { x } _ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \hat { \pmb { \varepsilon } } ^ { \prime } \right) + \sqrt { \beta _ { t } } \pmb { u } ,\tag{16}
$$

where $\pmb { u } \sim \mathcal { N } ( 0 , \mathbf { I } )$ is standard Gaussian noise. When $\pmb { x } _ { t }$ deviates from the centroid $\mu _ { c } ,$ the term $( 1 - \bar { \alpha } _ { t } ) ^ { 1 / 2 }$ acts as a noise-leveldependent weight: during the early stages of denoising (large t strong noise), the guidance is relatively weak to avoid disrupting the global structure; during the later stages (small t, weak noise) the guidance intensifies and forces the sample to converge toward the authentic class distribution.

## 5 Evaluation

## 5.1 Experimental Setup

Experimental Environment. All experiments were conducted on a server equipped with an AMD EPYC 7343 CPU @ 3.2 GHz, 256 GB of RAM, and an NVIDIA GeForce RTX 4090 GPU with 24 GB of memory. The software stack comprised Python 3.9 and the deep learning framework PyTorch 2.8.0.

Datasets. Our evaluation employs four representative public benchmark datasets. Each dataset is partitioned into training, validation, and test sets following a 7:1:2 ratio.

• CIC-IDS-2017 (denoted IDS) [49]. Collected under emulated real-world network conditions, this dataset contains benign traffic alongside 14 categories of attack traffic. It supports binary classification (2-class) and 15-class classification tasks.

• ISCX-VPN-2016 (denoted VPN) [13]. This dataset focuses on encrypted traffic classification under both VPN and non-VPN environments. It supports binary classification (VPN vs. non-VPN), 6-class classification (service types), and 16-class classification (specific application types).

• USTC-TFC2016 (denoted USTC) [61]. This dataset is dedicated to Tor anonymous network traffic classification. It supports binary classification (Tor vs. non-Tor) and 20-class classification.

• CSTNET-TLS1.3 (denoted TLS) [30]. This dataset is dedicated to the analysis of TLS 1.3 encrypted traffic. It encompasses traffic from 120 mainstream websites and supports a 120-class classification task.

Data Preprocessing. Our experiments adopt strict flow-level partitioning, using flow-level statistical features as the fundamental unit of analysis. The specific preprocessing pipeline is as follows:

• Parsing of raw traffic files is performed such that each flow sample retains only statistical features derived from transportlayer interactions.

• Rigorous topological information anonymization is enforced. Identifiers that directly locate network entities—including flow IDs, IP, and Port—are removed.

• Timestamp generalization is applied.

• Missing values arising from statistical features are uniformly filled with zero to maintain tensor integrity.

It is worth emphasizing that the constraints described above apply exclusively to our own model. Unless otherwise specified, the baseline models described subsequently adopt the data preparation procedures recommended in their original publications

Baseline Models. To comprehensively validate the superiority of the proposed TDDM-Melatt framework in encrypted traffic classification scenarios, we select SOTA methods as comparative baselines.

• Classification Baselines. We compare our model against SOTA traffic classification models, including ET-BERT [30], YaTC [71], NetMamba [60], TrafficFormer [75], netFound [21], and Pcap-Encoder [72]. All baseline models adhere as closely as possible to the data preparation procedures and hyperparameter settings recommended in their original publications. Our requirement is strict flow-level partitioning. For pre-trained models, we freeze the backbone parameters and ine-tune only the downstream classification head. Furthermore, we require that all models support recognition of classes containing only a small number of samples; discarding any class due to insufficient sample size is forbidden, as artificially modifying the original data distribution introduces bias that may adversely affect real-world deployment performance.

• Data Generation Baselines. We compare the proposed TDDM against current SOTA traffic generation models, including the diffusion-based models NetDiffus [52] and LRT-DDPM [58], as well as GANs [1] and their variants ACGAN [12] and WGAN-GP [40]. All data generation procedures are applied exclusively to the training set, while the test set remains unaltered in its original state.

Evaluation Metrics. We adopt Accuracy (AC), Precision (PR), and F1-Score (F1) as the primary evaluation metrics. In addition to classification metrics, we adopt Maximum Mean Discrepancy (MMD) and Average Pairwise Distance (APD) as complementary metrics to directly evaluate the quality of generated samples.

## 5.2 Analysis of Classification Performance

To evaluate the generalization capability and feature representation quality of the TDDM-Melatt across diverse network traffic classification scenarios, we construct 6 classification tasks spanning three datasets: ISCX-VPN-2016, USTC-TFC2016, and CSTNET-TLS1.3. These tasks cover varying levels of difficulty, including binary classification, multi-class classification, and fine-grained classification. 6 baseline classifiers and 6 SOTA representation models are selected for comparative validation. Given the sufficient sample sizes in binary classification tasks, TDDM is not applied to those tasks. For the remaining classification tasks, a class is defined as a "long-tail" (minority) class if its sample size is less than 15% of that of the largest class. For such classes, TDDM is employed to augment their training set by a factor of 1.5×. The full set of experimental results is summarized in Table 3.

The experimental results demonstrate that TDDM-Melatt achieves consistently superior performance across multi-scenario traffic classification tasks, exhibiting a significant performance advantage and strong generalization capability. The baseline classifiers, when unconstrained, perform reasonably on binary classification tasks but suffer precipitous performance drops on multi-class tasks of higher difficulty, revealing severely insufficient generalization. Although the representation learning models generally exhibit better generalization than the baseline classifiers, their performance still degrades substantially compared to the results originally reported in their respective publications. Notably, this performance attenuation stems from merely two constraints: (1) requiring that packets from the same flow not appear simultaneously in both the training and test sets; (2) freezing the encoder parameters of pre-trained models. In contrast, the memory-decoupled design of TDDM-Melatt enables the model to focus on learning the intrinsic behavioral characteristics of traffic rather than relying on spurious topological identifiers. As a result, it maintains stable, high performance under stringent experimental conditions—including topology anonymization and timestamp generalization.

As shown in Table 3, Pcap-Encoder and TDDM-Melatt both achieve outstanding performance and appear comparable. To further simulate common real-world network scenarios involving traffic encryption and anonymization, we conduct a generalization robustness study in the context of multi-class classification tasks. We progressively apply the data preprocessing constraints adopted by TDDM-Melatt to the input of Pcap-Encoder. We first remove payload data to simulate scenarios where payload encryption renders payload inspection infeasible. Subsequently, we remove explicit identifiers such as IP and Port to emulate traffic classification under an anonymized network environment. Using the F1-score, which better reflects true performance under class imbalance, as the evaluation metric, the results are reported in Table 4. As the level of data generalization increases, the detection performance of Pcap-Encoder exhibits a pronounced downward trend. In particular, after the removal of explicit identifiers, its F1-score on the TLS-120class task plummets from 0.637 to 0.130, ultimately degrading to a level comparable to the remaining baseline models. Under identical generalization scenarios, TDDM-Melatt consistently achieves the best performance, validating the superiority of its memory decoupling and diffusion-based data augmentation mechanisms in coping with the uncertainties intrinsic to real-world network environments.

## 5.3 Generation Performance Analysis

Examining the experimental results in Table 3, it can be observed that, with the exception of the VPN-16class task, TDDM consistently improves the classification performance of Melatt across the remaining five tasks. To further validate the generation quality of TDDM, we conduct comparative experiments against NetDiffus, LRT-DDPM, and a suite of GAN-based models (GAN, WGAN-GP, ACGAN). A unified experimental setup is adopted for all comparisons: Melatt serves as the underlying pre-trained classification backbone, and the original training set is augmented exclusively with synthetically generated data. The experimental results are presented in Table 5.

![](images/a8cc36ff3b4eefbe621a477c914b528097b593e147640b69c169e7b866333292.jpg)

![](images/6f624667e6023c23020a178019157d56d4468f3e2e7695c5eea871d822bac419.jpg)  
(b) Augmented F1 Score  
Figure 5: Per-class Performance on USTC-20class.

Analysis of the experimental results reveals that, in VPN-16class and USTC-20class tasks, the performance gains contributed by all generative models are negligible, and several models even induce performance degradation. To further investigate this phenomenon, we conduct additional verification experiments, as illustrated in Figure 5 (and in Appendix Figure 11). Taking USTC-20class as an example, certain classes with abundant samples still suffer from relatively low classification F1 (e.g., Neris 0.8973), whereas the improvement for classes with scarce samples is limited (e.g., Others from 0.9644 to 0.9664). Consequently, data augmentation targeted at minority classes struggles to yield a marked improvement in overall classification performance. Instead, the synthetic data may introduce noise that adversely affects the training dynamics of the majority classes.

In contrast, on the VPN-6class and TLS-120class tasks, TDDM is the sole method that achieves statistically significant performance improvements. The F1-score gains reach 9.98% on TLS-120class and 5.56% on VPN-6class, respectively. Conversely, all other baseline generative models fail to deliver any notable performance enhancement on these two tasks. These results indicate that the generalization capacity of existing data generation baselines is extremely limited. Most of these baselines do not adequately account for the high-dimensional sparsity and temporal dependencies inherent in network traffic data, and they may introduce substantial noise that disrupts the classifier's decision boundaries.

To further verify the necessity and effectiveness of TDDM, we conduct a systematic comparative experiment by incorporating five classical imbalance handling methods based on existing GANvariant-based generative schemes, including Oversampling, Simple Copy, SMOTE, and ADASYN. We adopt MMD and APD as direct generation quality evaluation metrics, and carry out all experiments on the IDS-15class task. The experimental results are presented in Table 6. It is worth noting that although the classical imbalance handling methods achieve favorable MMD and APD scores, they contribute little to downstream classification performance, or even produce negative effects. Compared with vanilla GAN, TDDM achieves competitive MMD and APD performance, while delivering the strongest classification gain. This demonstrates that the diffusion model achieves a better balance between sample fidelity and task-relevant signal preservation.

Table 3: Classification Performance Comparison of TDDM-Melatt against Baseline and SOTA Models across 6 Tasks.
<table><tr><td rowspan="2">Model</td><td colspan="2">VPN-2class</td><td colspan="2">VPN-6class</td><td colspan="2">VPN-16class</td><td colspan="2">USTC-2class</td><td colspan="2">USTC-20class</td><td colspan="2">TLS-120class</td></tr><tr><td>AC</td><td>F1</td><td>AC</td><td>F1</td><td>AC</td><td>F1</td><td>AC</td><td>F1</td><td>AC</td><td>F1</td><td>AC</td><td>F1</td></tr><tr><td>MLP [4]</td><td>0.925</td><td>0.798</td><td>0.657</td><td>0.140</td><td>0.309</td><td>0.032</td><td>0.966</td><td>0.966</td><td>0.154</td><td>0.030</td><td>0.008</td><td>0.000</td></tr><tr><td>1D-CNN [50]</td><td>0.876</td><td>0.558</td><td>0.683</td><td>0.136</td><td>0.311</td><td>0.030</td><td>0.915</td><td>0.915</td><td>0.136</td><td>0.021</td><td>0.005</td><td>0.000</td></tr><tr><td>2D-CNN [47]</td><td>0.918</td><td>0.804</td><td>0.681</td><td>0.140</td><td>0.308</td><td>0.035</td><td>0.986</td><td>0.986</td><td>0.062</td><td>0.033</td><td>0.014</td><td>0.000</td></tr><tr><td>LSTM [33]</td><td>0.955</td><td>0.895</td><td>0.540</td><td>0.364</td><td>0.337</td><td>0.277</td><td>1.000</td><td>1.000</td><td>0.854</td><td>0.844</td><td>0.574</td><td>0.541</td></tr><tr><td>GRU [69]</td><td>0.959</td><td>0.902</td><td>0.643</td><td>0.371</td><td>0.354</td><td>0.296</td><td>1.000</td><td>1.000</td><td>0.852</td><td>0.843</td><td>0.579</td><td>0.545</td></tr><tr><td>Transformer [35]</td><td>0.972</td><td>0.924</td><td>0.579</td><td>0.226</td><td>0.355</td><td>0.089</td><td>0.992</td><td>0.992</td><td>0.395</td><td>0.336</td><td>0.258</td><td>0.210</td></tr><tr><td>netFound [21]</td><td>0.760</td><td>0.619</td><td>0.473</td><td>0.365</td><td>0.329</td><td>0.153</td><td>0.994</td><td>0.994</td><td>0.580</td><td>0.307</td><td>0.019</td><td>0.005</td></tr><tr><td>YaTC [71]</td><td>0.839</td><td>0.839</td><td>0.692</td><td>0.601</td><td>0.609</td><td>0.443</td><td>0.995</td><td>0.995</td><td>0.852</td><td>0.780</td><td>0.155</td><td>0.096</td></tr><tr><td>NetMamba [60]</td><td>0.750</td><td>0.745</td><td>0.569</td><td>0.490</td><td>0.396</td><td>0.284</td><td>0.976</td><td>0.975</td><td>0.725</td><td>0.577</td><td>0.088</td><td>0.045</td></tr><tr><td>ET-BERT [30]</td><td>0.847</td><td>0.846</td><td>0.717</td><td>0.642</td><td>0.592</td><td>0.437</td><td>1.000</td><td>1.000</td><td>0.849</td><td>0.796</td><td>0.109</td><td>0.067</td></tr><tr><td>TrafficFormer [75]</td><td>0.909</td><td>0.909</td><td>0.765</td><td>0.694</td><td>0.677</td><td>0.544</td><td>1.000</td><td>1.000</td><td>0.720</td><td>0.650</td><td>0.297</td><td>0.240</td></tr><tr><td>Pcap-Encoder [72]</td><td>0.999</td><td>0.999</td><td>0.921</td><td>0.898</td><td>0.835</td><td>0.710</td><td>1.000</td><td>1.000</td><td>0.910</td><td>0.871</td><td>0.710</td><td>0.637</td></tr><tr><td>Melatt</td><td>0.999</td><td>0.999</td><td>0.838</td><td>0.833</td><td>0.826</td><td>0.811</td><td>1.000</td><td>1.000</td><td>0.966</td><td>0.966</td><td>0.586</td><td>0.581</td></tr><tr><td>TDDM-Melatt</td><td></td><td></td><td>0.889</td><td>0.889</td><td>0.824</td><td>0.810</td><td></td><td></td><td>0.967</td><td>0.966</td><td>0.684</td><td>0.681</td></tr></table>

Note: The best results are highlighted in bold, and the second-best results are indicated with underlining. For the binary classification tasks (VPN-2class, USTC-20class), TDDM-Melatt does not employ data augmentation and its performance is identical to that of Melatt.

Table 4: Performance under Data Generalization Scenarios.
<table><tr><td>Model</td><td>Processing</td><td>VPN-16class</td><td>TLS-120class</td></tr><tr><td rowspan="3">Pcap-Encoder</td><td>Retain (Original)</td><td>0.710</td><td>0.637</td></tr><tr><td>Remove Payload</td><td>0.667</td><td>0.630</td></tr><tr><td>Remove IPs/Ports</td><td>0.525</td><td>0.130</td></tr><tr><td rowspan="2">Melatt TDDM-Melatt</td><td>-</td><td>0.811</td><td>0.581</td></tr><tr><td>1</td><td>0.810</td><td>0.681</td></tr></table>

![](images/26fcad787f763d8706a7e5aba66df3eeacc9e3a198701acd9f094b5ffd3de95d.jpg)

![](images/6ccd2335f07eb00501e061afca9a4cf89e25b939e3fff19845313ac8cc76ac3f.jpg)

![](images/9023435091b051af06bb240ab3d42ab9d7b6e72c7e3e712cc94989904d6faf8a.jpg)  
Figure 6: Convergence Analysis in Training Sample Size.

To determine the minimum number of samples required to train a reliable class-wise diffusion model, we conduct an experiment where we progressively increase the training sample size from 20 to 600 (in steps of 5) for a single minority class and observe the MMD and APD curves, as shown in Figure 6. To identify the optimal minimum sample size, we adopt a unified marginal improvement criterion combining the 50% convergence rule and standard deviation constraints. As shown in the marginal improvement rate curves in Figure 6, the inflection point is identified at 150 training samples, where the MMD improvement reaches 40.1% of its total reduction, and the APD improvement reaches 55.7% of its total increase. Beyond this point, the marginal gains diminish significantly, and the cross-class standard deviations of MMD and APD converge to stable levels (0.1533 and 1.2766, respectively). Based on these results, we recommend a minimum of 150 samples per class for training a reliable single-class diffusion model. This threshold ensures that the model captures sufficient intra-class diversity while avoiding overfitting to limited data. When the training sample size reaches 600, the MMD and APD scores of TDDM reach levels comparable to those of classical oversampling methods.

Through targeted enhancements to the diffusion generative process—including feature selection, class-wise training, centroidguided compact sampling, and nearest-neighbor padding—TDDM provides an effective technical pathway for network traffic classification in scenarios characterized by limited samples and class imbalance.

## 5.4 Condition-Guided Noise Sampling Analysis

To thoroughly investigate the impact of the condition-guided compact noise sampling mechanism on the performance of TDDM, we design a targeted ablation study. In the experiment, TDDM first undergoes a forward diffusion process of T = 1000 timesteps, followed by 500 epochs of training for the noise prediction network. During the sample generation phase, we compare the distributional characteristics of the generated data under 2 settings: with or without condition-guided compact noise sampling. The denoising process is visualized using t-SNE dimensionality reduction [37]. The results are presented in Figure 7 (and in Appendix Figure 13). It is worth noting that t-SNE transforms local similarities among data points in high-dimensional space into probability distributions, effectively revealing cluster structures. However, t-SNE cannot accurately reflect the true distances between clusters, and the visualization outcome is inherently stochastic.

Table 5: Performance Comparison of Data Augmentation Methods across 4 Tasks.
<table><tr><td rowspan="2">Model</td><td colspan="3">VPN-6class</td><td colspan="3">VPN-16class</td><td colspan="3">USTC-20class</td><td colspan="3">TLS-120class</td></tr><tr><td>PR</td><td>AC</td><td>F1</td><td>PR</td><td>AC</td><td>F1</td><td>PR</td><td>AC</td><td>F1</td><td>PR</td><td>AC</td><td>F1</td></tr><tr><td>Melatt</td><td>0.8502</td><td>0.8381</td><td>0.8329</td><td>0.8282</td><td>0.8258</td><td>0.8113</td><td>0.9677</td><td>0.9664</td><td>0.9656</td><td>0.5855</td><td>0.5863</td><td>0.5810</td></tr><tr><td>Melatt + GAN [1]</td><td>0.8490</td><td>0.8393</td><td>0.8331</td><td>0.8281</td><td>0.8256</td><td>0.8107</td><td>0.9687</td><td>0.9670</td><td>0.9661</td><td>0.5753</td><td>0.5742</td><td>0.5698</td></tr><tr><td>Melatt + WGAN-GP [40]</td><td>0.8499</td><td>0.8385</td><td>0.8333</td><td>0.8288</td><td>0.8246</td><td>0.8105</td><td>0.9694</td><td>0.9682</td><td>0.9673</td><td>0.5790</td><td>0.5819</td><td>0.5752</td></tr><tr><td>Melatt + ACGAN [12]</td><td>0.8494</td><td>0.8377</td><td>0.8325</td><td>0.8273</td><td>0.8245</td><td>0.8095</td><td>0.9690</td><td>0.9688</td><td>0.9687</td><td>0.5827</td><td>0.5821</td><td>0.5776</td></tr><tr><td>Melatt + LRT-DDPM [58]</td><td>0.8494</td><td>0.8387</td><td>0.8332</td><td>0.8274</td><td>0.8252</td><td>0.8109</td><td>0.9689</td><td>0.9672</td><td>0.9664</td><td>0.5679</td><td>0.5672</td><td>0.5626</td></tr><tr><td>Melatt + NetDiffus [52]</td><td>0.8500</td><td>0.8376</td><td>0.8320</td><td>0.8276</td><td>0.8243</td><td>0.8102</td><td>0.9694</td><td>0.9681</td><td>0.9672</td><td>0.5618</td><td>0.5646</td><td>0.5587</td></tr><tr><td>TDDM-Melatt</td><td>0.9022</td><td>0.8889</td><td>0.8885</td><td>0.8272</td><td>0.8242</td><td>0.8102</td><td>0.9682</td><td>0.9667</td><td>0.9659</td><td>0.6834</td><td>0.6844</td><td>0.6808</td></tr></table>

![](images/9ab9e87e7911d358a53be860b61590d18b0e68186e97d0012ee5e7900080e756.jpg)  
Figure 7: Comparison of Reverse Denoising Processes in IDS and TLS Datasets (UG: unguided, G: guided).

Table 6: Quality Metrics of Imbalance Handling Methods.
<table><tr><td>Method</td><td>MMD↓</td><td>APD↑</td></tr><tr><td>GAN</td><td>0.3930</td><td>2.4315</td></tr><tr><td>WGAN-GP</td><td>0.3310</td><td>2.6534</td></tr><tr><td>ACGAN</td><td>0.3962</td><td>2.4721</td></tr><tr><td>Oversampling</td><td>0.0591</td><td>3.0292</td></tr><tr><td>SMOTE</td><td>0.0518</td><td>3.0356</td></tr><tr><td>ADASYN</td><td>0.0506</td><td>2.9781</td></tr><tr><td>SimpleCopy</td><td>0.0545</td><td>3.0558</td></tr><tr><td>TDDM</td><td>0.2667</td><td>2.9633</td></tr></table>

Observing the experimental results reveals that, in the unguided reverse denoising process, the noise predicted by the model exhibits ambiguity, which ultimately leads to inferior quality of the final generated samples. After the introduction of condition-guided compact noise sampling, the model is able to effectively steer the reverse denoising process. As the timestep decreases, the generated samples progressively converge toward the distribution of authentic samples. Considering the configured noise schedule parameters $( \beta _ { \mathrm { s t a r t } } = 1 0 ^ { - 4 } , \ \beta _ { \mathrm { e n d } } = 0 . 0 2 )$ , the guidance strength is relatively weak during the initial stages of denoising; as the process progresses toward the final stages, the guidance strength intensifies significantly. From a theoretical perspective, by embedding class information into the noise sampling procedure, the conditional guidance mechanism effectively constrains the output space of the denoising generation, thereby reducing the risk of mode collapse while simultaneously enhancing the diversity of the generated samples.

## 5.5 Ablation Analysis of the Memory Model

![](images/612fcb40feb24e3a5a191486ffda3c5f595bfdd8fb1e23c970d9be34593cc048.jpg)  
Figure 8: Feature Extraction Performance in 6 Classifiers.

To validate the actual contribution of the memory model Melatt in feature pre-training and downstream classification, we design and conduct targeted ablation experiments. By comparing the generalization performance of different feature extraction strategies across multiple classifiers, we analyze the advantages of Melatt over a conventional Autoencoder (AE) [25, 70] and the original raw traffic features. The experiments are conducted on the 6-class and 16-class classification tasks of the ISCX-VPN-2016 dataset, comparing six classifiers: XGBoost [63], CatBoost [43], RandomForest [3], AdaBoost [15], GaussianNB [16], and KNN [9]. All comparison schemes share a unified experimental configuration, with the feature extraction method serving as the sole independent variable. Each experimental configuration is repeated independently five times, and the mean and standard deviation are reported to eliminate the influence of random fluctuations. The results are illustrated in Figure 8, along with the error bars of the repeated trials.

Baseline does not incorporate any deep learning feature extraction module and retains only preprocessing. AE denotes a MLP autoencoder without the memory module. It extracts the latent vector and computes reconstruction errors (i.e., the squared differences between the input and reconstructed). These errors, concatenated with the latent vector, serve as the classification features. Melatt refers to our full model. Across all 6 classifiers, the features extracted by Melatt achieve the highest AC and F1, with notable improvements. Although AE adopts a pre-training paradigm based on reconstruction error, its lack of a content-based addressing mechanism for the memory bank causes its performance on certain classifiers (e.g., XG-Boost, GaussianNB) to fall even below that of the Baseline. Features derived purely from reconstruction error may introduce additional noise or redundant information during the encoding and decoding processes. In contrast, by virtue of the memory module, Melatt can effectively capture and match the deep behavioral characteristics and anomalous patterns of traffic, thereby supplying more discriminative feature representations to downstream classifiers.

Further analysis of the performance across different classifiers reveals that the performance trends of the three feature extraction strategies remain consistent across all 6 classifiers. Gradient boosting tree models (XGBoost, CatBoost) exhibit the best overall performance. RandomForest is a parallel ensemble learning method and slightly underperforms gradient boosting trees. This is primarily attributable to its voting-based aggregation mechanism, whose residual fitting precision is inferior to serial iterative optimization of gradient boosting. AdaBoost, as an earlier boosting algorithm delivers relatively limited performance on complex traffic classification tasks; its high sensitivity to noisy samples renders the model prone to overfitting when confronted with traffic data characterized by feature overlap and outliers. The probabilistically-driven GaussianNB and the distance-based KNN perform poorly, indicating the inherent limitations of linear assumptions and distance metrics in high-dimensional traffic feature spaces. In addition, the performance of CatBoost and RandomForest exhibits minor fluctuations across the five repeated runs. It is worth noting that, although our model was not specifically designed with a sophisticated classifier, the insights gained from this experiment provide valuable directions for the design of downstream classifiers in future work.

![](images/395873701a601374003990ed1a853509e1a481b9b5436b0692b3808111bcb822.jpg)  
Figure 9: Component-level Ablation Study of Melatt.

To precisely identify the contribution of each core component in Melatt, we conduct additional ablation experiments on two representative tasks: USTC-20class (relatively balanced, Macro F1 = 96.56%) and TLS-120class (highly imbalanced fine-grained task, Macro F1 = 58.10%). We evaluate six variants: (A1) replacing CG-LSTM with standard LSTM, (A2) removing the multi-head crossattention module, (B1) removing the compactness loss, (B2) removing the diversity loss, (B3) removing the entropy loss, and (B4) retaining only the reconstruction loss (i.e., removing the other three losses). Figure 9 presents the F1 scores of all variants on both tasks.

The ablation results reveal a clear pattern: the effectiveness of each Melatt component is strongly correlated with task difficulty. On the relatively simple USTC-20class task (96.56% Macro F1), removing any single component causes only marginal performance degradation (F1 drop < 0.2%). In contrast, on the challenging TLS-120class task (58.10% Macro F1), the contributions of all components become substantially more pronounced. This component-wise effectiveness aligns with the inspiration of our framework: in simple scenarios, the memory module acts as a robust yet non-critical auxiliary regulator, whereas in challenging fine-grained classification tasks, each component becomes indispensable for maintaining high-fidelity prototype representations.

![](images/1703e71d6b23583745e1558440e4ad9e97479b885ce6d0767cfed3f0491e978d.jpg)  
Figure 10: Efficiency Evaluation of Feature Extraction.

## 5.6 Model Efficiency

Feature extraction serves as the critical link between data preprocessing and downstream classification, and its computational efficiency directly determines the overall system throughput and end-to-end detection latency. While the ablation study has confirmed the significant performance advantages of Melatt in classification, we further quantify the computational overhead introduced by the memory module through an efficiency comparison experiment, evaluating the time performance of Melatt across different traffic classification scenarios. The experiment encompasses eight classification tasks of varying scale and complexity across the four datasets. All experiments are conducted on an edge computing platform equipped with NVIDIA GeForce RTX 3050 Laptop GPU. To eliminate the influence of system caching and initialization overhead, five warm-up runs are performed first, followed by ten consecutive test runs, from which the arithmetic mean is reported.

The experimental results are presented in Figure 10. The additional overhead introduced by the memory module ranges between 50% and 110%. For tasks with smaller sample sizes but higher class complexity (e.g., TLS-120class), the overhead reaches 104.59%; for all other tasks, it remains below 80%. This indicates that the memory mechanism incurs a relatively higher computational cost when handling complex tasks, yet the overhead remains within an acceptable range. Notably, task complexity exerts a minimal impact on the per-sample processing time. From the 2-class classification tasks to the 120-class task, the per-sample processing time increases by merely approximately 0.005 ms. This demonstrates that the computational complexity of Melatt is largely decoupled from the number of classes, exhibiting excellent scalability.

Table 7: Inference Latency Comparison.
<table><tr><td>Model</td><td>Architecture</td><td>Latency (ms)</td></tr><tr><td>Melatt + XGBoost</td><td>CG-LSTM + Memory</td><td>&lt; 12</td></tr><tr><td>NetMamba</td><td>Mamba</td><td>&gt; 15</td></tr><tr><td>ET-BERT</td><td>BERT</td><td>&gt; 50</td></tr><tr><td>TrafficFormer</td><td>BERT</td><td>&gt; 50</td></tr><tr><td>Pcap-Encoder</td><td>T5</td><td>&gt; 180</td></tr><tr><td>netFound</td><td>BERT</td><td>&gt; 104</td></tr></table>

To evaluate the practical inference efficiency of the proposed Melatt framework, we conduct a quantitative latency comparison with the SOTA traffic classification models, with detailed results summarized in Table 7. The Melatt framework integrated with XGBoost achieves optimal inference efficiency, with a per-sample feature extraction latency lower than 12 ms. This prominent efficiency advantage benefits from the framework's decoupled design, which employs a lightweight XGBoost classifier on fixed pre-trained representations. This design avoids the necessity of end-to-end full network execution for each inference, fundamentally reducing inference overhead and enabling Melatt to deliver far superior practical deployment efficiency compared with all competing SOTA models.

## 6 Discussion

The core insight underpinning the proposed TDDM-Melatt is that the generalization bottleneck in encrypted traffic classification stems not from insufficient model capacity, but rather from a fundamental misalignment of learning objectives. Existing research has predominantly pursued performance gains in specific datasets while overlooking the dynamic variations in feature distributions within real-world network environments. Our experiments compellingly demonstrate that once spurious features such as IP and Port are rigorously removed, the performance of SOTA models suffers a precipitous decline. This phenomenon reveals a pervasive “benchmark overfitting" problem in the current research landscape. Through its memory-decoupled architecture, TDDM-Melatt fundamentally shifts the learning paradigm of traffic classification models from “fitting the training data distribution" to “learning universal behavioral prototypes of traffic.

Regarding data augmentation, our study indicates that generic generative models cannot be directly transplanted to network traffic data. The reason TDDM outperforms GANs and other diffusionbased models lies critically in the fact that we did not blindly transfer architectures designed for image generation. Instead, we proceeded from the intrinsic characteristics of traffic data and devised a suite of targeted strategies, including feature selection, class-wise training, centroid-guided sampling, and nearest-neighbor padding. Our experiments also reveal that data augmentation is not a panacea. When performance on the majority classes approaches saturation, merely increasing minority-class samples yields limited improvement in overall performance. This suggests that future research should not only focus on data augmentation techniques themselves, but should also explore more effective imbalance-aware loss functions and training strategies. Furthermore, we observe that the effectiveness of TDDM is not uniformly transferable across different base models. This disparity suggests that TDDM's compatibility depends on the base model's representation learning paradigm. We discuss this limitation in detail in Appendix C.

Although TDDM-Melatt has achieved outstanding performance on public datasets, we are fully aware that the complexity of realworld network environments far exceeds that of any laboratory dataset. Future work should address issues such as the inability of memory to adapt to dynamic traffic patterns and the insufficiency of incremental learning capabilities. We believe that interpretability and generalizability will be the defining characteristics of nextgeneration AI systems for cybersecurity. The decoupled memory architecture and the spurious-correlation-free training paradigm proposed in this paper lay the foundation for building such systems.

## 7 Related Work

Encrypted traffic classification constitutes a highly active research area in contemporary network security. In recent years, deep learning methods have emerged as the predominant paradigm. CNNs [36, 48] have been employed to capture local packet length patterns while LSTM and GRU [31] have been applied to model temporal dependencies. ET-BERT [30] draws inspiration from pre-trained language models. YaTC [71] adopts the Transformer, leveraging selfattention mechanisms to capture long-range dependencies within traffic flows. NetMamba [60] introduces state space models to preserve the capacity for long-sequence modeling. However, explicit identifiers commonly retained in public datasets exhibit spurious correlations with traffic labels [24, 64, 72]. This phenomenon causes the generalization capability of these models to deteriorate severely when deployed in real-world network environments.

Data augmentation techniques have been extensively explored as a means to address the problem of class imbalance. Conventional methods such as SMOTE [29] and ADASYN [32] generate synthetic samples via linear interpolation in the feature space, yet they struggle to capture the complex temporal dependencies and nonlinear characteristics inherent in traffic data. The generated samples often lack semantic coherence and may even introduce noise, thereby degrading model performance. In the realm of deep generative models, GANs [1] and their variants, including WGAN-GP [40] and ACGAN [12], have been widely investigated for traffic generation [26, 67]. Nevertheless, these methods suffer from intrinsic deficiencies such as training instability, mode collapse, and vanishing gradients [39]. More recently, diffusion models have begun to be applied to traffic data augmentation [68]. NetDiffus [52] maps traffic data into two-dimensional grayscale images and then directly adopts an image-oriented diffusion architecture. LRT-DDPM [58] attempts to leverage diffusion models for encrypted traffic classification. However, both of these methods directly transplant generic generative model architectures without adequately considering the distinctive characteristics of traffic data—namely, high dimensionality, sparsity, strong feature interdependencies, and discrete structured nature [65]. The generated samples consequently suffer from issues such as feature distribution drift and semantic information deficiency [62, 76], which may, in turn, introduce noise that interferes with model training.

## 8 Conclusion

In this paper, to address the insufficient generalization capability caused by spurious correlations and the class imbalance problem in real-world encrypted traffic classification, we propose TDDM-Melatt, a decoupled memory and diffusion model framework for generalizable encrypted traffic classification. The modular decoupled memory representation model based on the CG-LSTM separates feature extraction from knowledge storage. A pre-training and inference paradigm free of spurious correlations is established to suppress shortcut learning, and a denoising diffusion model tailored to the structural properties of traffic data is designed to alleviate the class imbalance issue. Experiments on four public datasets demonstrate that TDDM-Melatt achieves superior performance across multiple classification tasks. The TDDM data augmentation method yields up to a 9.98% improvement in F1-score, and ablation studies validate the effectiveness of each core component. This paper provides a new paradigm for encrypted traffic classification in real network environments, characterized by high accuracy, strong generalization, and rapid adaptability.

## References

[1] Giuseppina Andresini, Annalisa Appice, Luca De Rose, and Donato Malerba. 2021. GAN augmentation to deal with imbalance in imaging-based intrusion detection. Future Generation Computer Systems 123 (2021), 108–127.

[2] Roman Beltiukov, Wenbo Guo, Arpit Gupta, and Walter Willinger. 2023. In Search of netUnicorn: A Data-Collection Platform to Develop Generalizable ML Models for Network Security Problems. In CCS '23. 2217–2231.

[3] Leo Breiman. 2001. Random Forests. Machine Learning 45 (2001).

[4] Zhiyong Bu, Bin Zhou, Pengyu Cheng, Kecheng Zhang, and Zhen-Hua Ling. 2020. Encrypted Network Traffic Classification Using Deep and Parallel Networkin-Network Models. IEEE ACCESS 8 (2020), 132950–132959.

[5] Tomasz Bujlow, Valentin Carela-Espanol, and Pere Barlet-Ros. 2015. Independent comparison of popular DPI tools for traffic classification. Computer Networks 76 (2015), 75–89.

[6] Francesco Cerasuolo, Alfredo Nascita, Giampaolo Bovenzi, Giuseppe Aceto, Domenico Ciuonzo, Antonio Pescape, and Dario Rossi. 2024. MEMENTO: A novel approach for class incremental learning of encrypted traffic. Computer Networks 245 (2024), 110374.

[7] Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In KDD '16. 785–794.

[8] Xu-Yang Chen, Lu Han, De-Chuan Zhan, and Han-Jia Ye. 2025. MIETT: Multi-Instance Encrypted Traffic Transformer for Encrypted Traffic Classification. In AAAI '25. 15922–15929.

[9] Thomas M. Cover and Peter E. Hart. 1967. Nearest Neighbor Pattern Classification. IEEE Transactions on Information Theory 13 (1967), 21–27.

[10] Jianbang Dai, Xiaolong Xu, Honghao Gao, Xinheng Wang, and Fu Xiao. 2023. SHAPE: A Simultaneous Header and Payload Encoding Model for Encrypted Traffic Classification. IEEE Transactions on Network and Service Management 20 (2023), 1993–2012.

[11] Prafulla Dhariwal and Alex Nichol. 2021. Diffusion models beat GANs on image synthesis. In NeurIPS '21.

[12] Hongwei Ding, Leiyang Chen, Liang Dong, Zhongwang Fu, and Xiaohui Cui. 2022. Imbalanced data classification: A KNN and generative adversarial networks-based hybrid approach for intrusion detection. Future Generation Computer Systems 131 (2022), 240 – 254.

[13] Gerard Draper-Gil, Arash Habibi Lashkari, Mohammad Saiful Islam Mamun, and Ali A. Ghorbani. 2016. Characterization of Encrypted and VPN Traffic using Time-related Features. In ICISSP '16.

[14] Michael Finsterbusch, Chris Richter, Eduardo Rocha, Jean-Alexander Muller, and Klaus Hanssgen. 2014. A Survey of Payload-Based Traffic Classification Approaches. IEEE Communications Surveys and Tutorials 16 (2014), 1135-1156.

[15] Yoav Freund and Robert E. Schapire. 1997. A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting. J. Comput. System Sci. 55 (1997), 119–139.

[16] Nir Friedman, Dan Geiger, and Moises Goldszmidt. 1997. Bayesian Network Classifiers. Machine learning 29 (1997), 131–163.

[17] Chuanpu Fu, Qi Li, Elisa Bertino, and Ke Xu. 2025. Training with Only 1.0% Samples: Malicious Traffic Detection via Cross-Modality Feature Fusion. In CCS '25. 3930–3944.

[18] Chuanpu Fu, Qi Li, and Ke Xu. 2023. Detecting Unknown Encrypted Malicious Traffic in Real Time via Flow Interaction Graph Analysis. In NDSS '23.

[19] Felix A. Gers, Jürgen Schmidhuber, and Fred Cummins. 2000. Learning to Forget: Continual Prediction with LSTM. Neural Computation 12 (2000), 2451–2471.

[20] Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton Van Den Hengel. 2019. Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection. In ICCV '19. 1705–1714.

[21] Satyandra Guthula, Roman Beltiukov, Navya Battula, Wenbo Guo, Arpit Gupta, and Inder Monga. 2025. netFound: Foundation Model for Network Security. arXiv:2310.17025 [cs.NI]

[22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. In NeurIPS '20.

[23] Sepp Hochreiter and Jurgen Schmidhuber. 1997. Long short-term memory. Neural Computation 9 (1997), 1735–1735.

[24] Kamil Jerabek, Jan Luxemburk, Richard Plny, Josef Koumar, Jaroslav Pesek, and Karel Hynek. 2025. When Simple Model Just Works: Is Network Traffic Classification in Crisis? arXiv:2506.08655 [cs.LG]

[25] Jizhe Jia, Meng Shen, Qingjun Yuan, Yong Liu, Jing Wang, Jian Kong, Liang Huang, Haotian He, and Liehuang Zhu. 2025. Adaptive detection of encrypted malware traffic via fully convolutional masked autoencoders. Frontiers of Computer Science 20 (2025).

[26] Qimin Jin, Rongheng Lin, and Fangchun Yang. 2020. E-WACGAN: Enhanced Generative Model of Signaling Data Based on WGAN-GP and ACGAN. IEEE Systems Journal 14 (2020), 3289–3300.

[27] Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang. 2022. Fine-Tuning Can Distort Pretrained Features and Underperform Out-of-Distribution. In ICLR '22.

[28] Xiaochang Li, Chen Qian, Qineng Wang, Jiangtao Kong, Yuchen Wang, Ziyu Yao, Bo Ji, Long Cheng, Gang Zhou, and Huajie Shao. 2026. Lens: A Knowledge-Guided Foundation Model for Network Traffic. arXiv:2402.03646 [cs.LG]

[29] Lawrence Chuin Ming Liaw, Shing Chiang Tan, Pey Yun Goh, and Chee Peng Lim. 2025. A histogram SMOTE-based sampling algorithm with incremental learning for imbalanced data classification. Information Sciences 686 (2025)

[30] Xinjie Lin, Gang Xiong, Gaopeng Gou, Zhen Li, Junzheng Shi, and Jing Yu. 2022. ET-BERT: A Contextualized Datagram Representation with Pre-training Transformers for Encrypted Traffic Classification. In WWW '22. 633–642.

[31] Chang Liu, Longtao He, Gang Xiong, Zigang Cao, and Zhen Li. 2019. FS-Net: A Flow Sequence Network for Encrypted Traffic Classification. In INFOCOM '19. 1171-1179.

[32] Jingmei Liu, Yuanbo Gao, and Fengjie Hu. 2021. A fast network intrusion detection system using adaptive synthetic oversampling and LightGBM. Computers and Security 106 (2021)

[33] Kaixuan Liu, Yuyang Zhang, Xiaoya Zhang, Wenxuan Qiao, and Ping Dong. 2023. Network Traffic Classification Based on LSTM plus CNN and Attention Mechanism. In ICENAT '22, Vol. 1696. 545–556.

[34] Ya Liu, Xiao Wang, Bo Qu, and Fengyu Zhao. 2024. ATVITSC: A Novel Encrypted Traffic Classification Method Based on Deep Learning. IEEE Transactions on Information Forensics and Security 19 (2024), 9374–9389.

[35] Ziao Liu, Yuanyuan Xie, Yanyan Luo, Yuxin Wang, and Xiangmin Ji. 2025. TransECA-Net: A Transformer-Based Model for Encrypted Traffic Classification. Applied Sciences-Basel 15 (2025).

[36] Mohammad Lotfollahi, Mahdi Jafari Siavoshani, Ramin Shirali Hossein Zade, and Mohammdsadegh Saberian. 2020. Deep packet: a novel approach for encrypted traffic classification using deep learning. Soft Computing 24 (2020), 1999–2012.

[37] Laurens Van Der Maaten. 2014. Accelerating t-SNE using tree-based algorithms. Journal of Machine Learning Research 15 (2014), 3221–3245.

[38] Yisroel Mirsky, Tomer Doitshman, Yuval Elovici, and Asaf Shabtai. 2018. Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection. In NDSS '18.

[39] Ziyu Mu, Xiyu Shi, and Safak Dogan. 2026. GMA-SAWGAN-GP: A Novel Data Generative Framework to Enhance IDS Detection Performance. arXiv:2603.28838 [cs.CR]

[40] Ziyu Mu, Xiyu Shi, and Safak Dogan. 2026. A Novel Solution for Zero-day Attack Detection in IDS using Self-attention and Jensen-Shannon divergence in WGAN-GP. Computer Networks 282 (2026).

[41] Eva Papadogiannaki and Sotiris Ioannidis. 2021. A Survey on Encrypted Network Traffic Analysis Applications, Techniques, and Countermeasures. ACM Comput Surv. 54 (2021), 123:1–123:35.

[42] Julien Piet, Dubem Nwoji, and Vern Paxson. 2023. GGFAST: Automating Generation of Flexible Network Traffic Classifiers. In SIGCOMM '23. 850-866.

[43] Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. 2018. CatBoost: unbiased boosting with categorical features. In NeurIPS '18. 6639–6649.

[44] Tian Qin, Guang Cheng, Yuyang Zhou, Zihan Chen, and Xing Luan. 2026. SNAKE: A sustainable and multi-functional traffic analysis system utilizing specialized large-scale models with a mixture of experts architecture. Computer Networks 276 (2026).

[45] Yuqi Qing, Qilei Yin, Xinhao Deng, Xiaoli Zhang, Peiyang Li, Zhuotao Liu, Kun Sun, Ke Xu, and Qi Li. 2025. Training Robust Classifiers for Classifying Encrypted Traffic under Dynamic Network Conditions. In CCS '25. 3563–3577.

[46] Xing Qiu, Guang Cheng, Weizhou Zhu, Dandan Niu, and Nan Fu. 2025. Dual-Channel Interactive Graph Transformer for Traffic Classification with Message-Aware Flow Representation. In AAAI '25. 685–693.

[47] Tal Shapira and Yuval Shavitt. 2019. FlowPic: Encrypted Internet Traffic Classification is as Easy as Image Recognition. In INFOCOM WKSHPS '19. 680–687.

[48] Tal Shapira and Yuval Shavitt. 2021. FlowPic: A Generic Representation for Encrypted Traffic Classification and Applications Identification. IEEE Transactions on Network and Service Management 18 (2021), 1218–1232.

[49] Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani. 2018. Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. In ICISSP '18. 108–116.

[50] KaiChao Shi, Yong Zeng, Baihe Ma, Zhihong Liu, and Jianfeng Ma. 2023. MT-CNN: A Classification Method of Encrypted Traffic Based on Semi-Supervised Learning. In GLOBECOM '23. 7538–7543.

[51] Zhaolei Shi, Nurbol Luktarhan, Yangyang Song, and Huixin Yin. 2023. TSFN: A Novel Malicious Traffic Classification Method Using BERT and LSTM. Entropy 25 (2023).

[52] Nirhoshan Sivaroopan, Dumindu Bandara, Chamara Madarasingha, Guillaume Jourjon, Anura P. Jayasumana, and Kanchana Thilakarathna. 2024. NetDiffus: Network traffic generation by diffusion models through time-series imaging. Computer Networks 251 (2024).

[53] Robin Sommer and Vern Paxson. 2010. Outside the closed world: On using machine learning for network intrusion detection. In IEEE SP '10. 305–316.

[54] Jiaming Song, Chenlin Meng, and Stefano Ermon. 2022. Denoising Diffusion Implicit Models. arXiv:2010.02502 [cs.LG]

[55] Zijia Song, Zhengyi Ma, Qiming Yu, Xiaohui Xie, Yelin Wang, and Guozheng Yang. 2026. PaT: An enhanced pretrained framework via Mamba-2 for network traffic analysis. Computer Networks 283 (2026).

[56] Saravanan Thirumuruganathan, Fatih Deniz, Issa Khalil, Ting Yu, Mohamed Nabeel, and Mourad Ouzzani. 2024. Detecting and Mitigating Sampling Bias in Cybersecurity with Unlabeled Data. In USENIX Security '24. 1741–1758.

[57] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All You Need. In NIPS '17.

[58] Chaoling Wang, Jiang Wu, and Liang Wang. 2025. LRT-DDPM: A Diffusion Model-Based Approach for Network Traffic Data Generation in Intrusion Detection. IEEE Access 13 (2025), 149054–149070.

[59] Kai Wang, Zhiliang Wang, Dongqi Han, Wenqi Chen, Jiahai Yang, Xingang Shi, and Xia Yin. 2023. BARS: Local Robustness Certification for Deep Learning based Traffic Analysis Systems. In NDSS '23

[60] Tongze Wang, Xiaohui Xie, Wenduo Wang, Chuyi Wang, Youjian Zhao, and Yong Cui. 2024. NetMamba: Efficient Network Traffic Classification via Pre-training Unidirectional Mamba. In ICNP '24.

[61] Wei Wang, Ming Zhu, Xuewen Zeng, Xiaozhou Ye, and Yiqiang Sheng. 2017. Malware Traffic Classification Using Convolutional Neural Network for Representation Learning. In ICOIN '17. 712–717.

[62] Tonglong Wei, Youfang Lin, Shengnan Guo, Yan Lin, Yiheng Huang, Chenyang Xiang, Yuqing Bai, and Huaiyu Wan. 2024. Diff-RNTraj: A Structure-Aware Diffusion Model for Road Network-Constrained Trajectory Generation. IEEE Transactions on Knowledge and Data Engineering 36 (2024), 7940–7953.

[63] Nimesha Wickramasinghe, Arash Shaghaghi, Elena Ferrari, and Sanjay Jha. 2025. Less is More: Simplifying Network Traffic Classification Leveraging RFCs. In WWW '25. 1398-1401.

[64] Nimesha Wickramasinghe, Arash Shaghaghi, Gene Tsudik, and Sanjay Jha. 2025. SoK: Decoding the Enigma of Encrypted Network Traffic Classifiers. In IEEE SP '25.1825 - 1843.

[65] Haoran Yu, Wenchuan Yang, Baojiang Cui, Runqi Sui, and Xuedong Wu. 2024. Enhanced anomaly traffic detection framework using BiGAN and contrastive learning. Cybersecurity 7 (2024).

[66] Jiwen Yu, Xuanyu Zhang, Youmin Xu, and Jian Zhang. 2023. CRoSS: diffusion model makes controllable, robust and secure image steganography. In NeurIPS '23.

[67] Jiangtao Zhai, Peng Lin, Yongfu Cui, Lilong Xu, and Ming Liu. 2023. GraphCWGAN-GP: A Novel Data Augmenting Approach for Imbalanced Encrypted Traffic Classification. Computer Modeling in Engineering and Sciences 136 (2023), 2069–2092.

[68] Shiyuan Zhang, Tong Li, Depeng Jin, and Yong Li. 2024. NetDiff: A Service-Guided Hierarchical Diffusion Model for Network Flow Trace Generation. Proc. ACM Netw. 2 (2024).

[69] Huiqi Zhao, Yaowen Ma, Fang Fan, and Huajie Zhang. 2024. Anomaly Detection Method for Integrated Encrypted Malicious Traffic Based on RFCNN-GRU. In FCS '23. 457–471.

[70] Ruijie Zhao, Xianwen Deng, Mingwei Zhan, Fangqi Li, Yanhao Wang, Yijun Wang, Guan Gui, and Zhi Xue. 2024. A Novel Self-Supervised Framework Based on Masked Autoencoder for Traffic Classification. IEEE/ACM Trans. Netw. 32 (2024), 2012–2025.

[71] Ruijie Zhao, Mingwei Zhan, Xianwen Deng, Yanhao Wang, Yijun Wang, Guan Gui, and Zhi Xue. 2023. Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation. In AAAI '23. 5420-5427.

[72] Yuqi Zhao, Giovanni Dettori, Matteo Boffa, Luca Vassio, and Marco Mellia. 2025. The Sweet Danger of Sugar: Debunking Representation Learning for Encrypted Traffic Classification. In SIGCOMM '25. 296–310.

[73] Ming Zheng, Xiaowen Hu, Ying Hu, Xiaoyao Zheng, and Yonglong Luo. 2025. Fed-UGI: Federated Undersampling Learning Framework With Gini Impurity for Imbalanced Network Intrusion Detection. IEEE Transactions on Information Forensics and Security 20 (2025), 1262–1277.

[74] Andy Zhou, Xiaojun Xu, Ramesh Raghunathan, Alok Lal, Xinze Guan, Bin Yu, and Bo Li. 2024. KnowGraph: Knowledge-Enabled Anomaly Detection via Logical

Reasoning on Graph Data. In CCS '24. 168–182.

[75] Guangmeng Zhou, Xiongwen Guo, Zhuotao Liu, Tong Li, Qi Li, and Ke Xu. 2025. TrafficFormer: An Efficient Pre-trained Model for Traffic Data. In IEEE SP '25.

[76] Haowei Zhu, Ling Yang, Jun-Hai Yong, Hongzhi Yin, Jiawei Jiang, Meng Xiao, Wentao Zhang, and Bin Wang. 2024. Distribution-aware data expansion with diffusion models. In NeurIPS '24

## A Preprocessed Statistical Features

This appendix provides the complete set of preprocessed features, as listed in Table 8. It also presents the full feature importance bar charts for 4 classification tasks (VPN-16class, TLS-120class, VPN-6class, and USTC-20class) that were generated using the TDDM, as shown in Figure 12.

Table 8: Complete List of 76 Preprocessed Statistical Features.
<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Features</td></tr><tr><td rowspan=1 colspan=1>Length / Size(16 features)</td><td rowspan=1 colspan=1>Total Length of Fwd Packet, Total Length of BwdPacket, Fwd Packet Length Max/Min/Mean/Std,Bwd Packet Length Max/Min/Mean/Std, PacketLength Min/Max/Mean/Std/Variance, AveragePacket Size, Fwd Segment Size Avg, Bwd SegmentSize Avg, Fwd Seg Size Min</td></tr><tr><td rowspan=1 colspan=1>Temporal / IAT(20 features)</td><td rowspan=1 colspan=1>Flow Duration, Flow IAT Mean/Std/Max/Min,Fwd IAT Total/Mean/Std/Max/Min, Bwd IAT To-tal/Mean/Std/Max/Min, Active Mean/Std/Max/Min,Idle Mean/Std/Max/Min</td></tr><tr><td rowspan=1 colspan=1>Packet / Byte Count(12 features)</td><td rowspan=1 colspan=1>Total Fwd Packet, Total Bwd packets, Subflow FwdPackets, Subflow Bwd Packets, Subflow Fwd Bytes,Subflow Bwd Bytes, Fwd Act Data Pkts, Down/UpRatio, Fwd Header Length, Bwd Header Length</td></tr><tr><td rowspan=1 colspan=1>Rate Features(10 features)</td><td rowspan=1 colspan=1>Flow Bytes/s, Flow Packets/s, Fwd Packets/s, BwdPackets/s, Fwd Bulk Rate Avg, Bwd Bulk Rate Avg,Fwd Bytes/Bulk Avg, Fwd Packet/Bulk Avg, BwdBytes/Bulk Avg, Bwd Packet/Bulk Avg</td></tr><tr><td rowspan=1 colspan=1>Flag / Window Init(18 features)</td><td rowspan=1 colspan=1>FIN Flag Count, SYN Flag Count, RST Flag Count,PSH Flag Count, ACK Flag Count, URG Flag Count,CWR Flag Count, ECE Flag Count, Fwd PSH Flags,Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags,FWD Init Win Bytes, Bwd Init Win Bytes</td></tr></table>

## B Supplementary Figures and Tables

This section collects figures and tables that cannot be fully presented in the main text due to page constraints. It streamlines the main manuscript and enhances the clarity of core illustrations therein.

## B.1 Per-class Performance on USTC-20class

In the main text, Figure 5 only presents the per-class F1-scores of TDDM-Melatt on the USTC-20class dataset after TDDM-based data augmentation. As a supplementary counterpart to Figure 5, Figure 11 further reports both Accuracy (AC) and F1-score metrics.

## B.2 Comparison of Reverse Denoising Processes

Figure 7 visualizes the denoising diffusion process with and without guidance for partial classes on the IDS and TLS datasets. For comprehensive comparison, Figure 13 further provides the contrast results of guided and unguided denoising diffusion across more timesteps over three datasets: IDS, USTC, and TLS.

![](images/d94fcacc860f319d338f3abefa424c6a66039e0f624407ccf39f24a3ad7ea35d.jpg)

![](images/6d5b3ffb46d5a3d57f2799117213e6155ea24c447633f5bbc43b7aed13c459dc.jpg)

(a) Original Accuracy  
![](images/b8cc4af5692b599d7ed04ec73f3be4b5c3f5703b6f947d60ec6c7e769940eae0.jpg)  
(c) Augmented Accuracy

(b) Original F1 Score  
![](images/7c148a5c9353537255937a03bf0230f8d606b739e1127c188b29df3146fa01aa.jpg)  
(d) Augmented F1 Score  
Figure 11: Per-class Classification Performance on USTC-20class (AC and F1).

## C Discussion on Model Generalization Limitations

Table 9: AC and F1 Changes after Integrating TDDM on Different Models and Datasets.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Network</td><td colspan="2">TLS-120class</td><td colspan="2">VPN-16class</td></tr><tr><td>ΔAC</td><td>∆F1</td><td>ΔAC</td><td>∆F1</td></tr><tr><td>ET-BERT</td><td>BERT</td><td>+0.0154</td><td>+0.0190</td><td>+0.1975</td><td>+0.1412</td></tr><tr><td>NetMamba</td><td>Mamba</td><td>-0.0623</td><td>-0.0346</td><td>+0.0054</td><td>-0.0126</td></tr><tr><td>YaTC</td><td>ViT</td><td>-0.1193</td><td>-0.0729</td><td>-0.0981</td><td>-0.0661</td></tr></table>

While TDDM-Melatt demonstrates strong generalization through its memory-decoupled architecture, we acknowledge that the effectiveness of the TDDM data augmentation module is not uniformly transferable across all base models, as TDDM is originally designed to adapt to Melatt. We integrate TDDM with three representative SOTA models—ET-BERT, NetMamba, and YaTC—on the TLS-120class and VPN-16class tasks, and report the performance changes in Table 9.

The results reveal a significant disparity: TDDM yields substantial improvements on ET-BERT (F1 +1.90% on TLS-120class, +14.12% on VPN-16class), yet causes notable degradation on NetMamba (F1 -3.46% on TLS-120class) and YaTC (F1 -7.29% on TLS-120class, - 6.61% on VPN-16class). We attribute this disparity to the differences in preprocessing paradigms and feature representations among these models. ET-BERT employs a BERT-style pre-training that learns contextualized byte-level representations; the additional augmented samples from TDDM provide complementary statistical patterns that enrich this representation space. In contrast, YaTC and NetMamba rely on statistical features or raw payload sequences that are more tightly coupled to the specific feature distributions of the training dataset. When TDDM introduces synthetic samples that slightly deviate from the original training distribution, these models, being more sensitive to the exact feature manifold, suffer from degraded decision boundaries.

VPN-16class Feature Importance  
![](images/2d58fc0093c413d8c2beaa23612572bfc005d85a2278c2ee61ec4daa93a55d50.jpg)

TLS-120class Feature Importance  
![](images/796210a4c2eeb05ab179894c7475d23f29ef315d5ad6645debc229f172c94e8a.jpg)

VPN-6class Feature Importance  
![](images/ba551a08f97fffb1068345cd1e21dc95c76628ea4c9cfeb2c41c80f26dc54e69.jpg)

USTC-20class Feature Importance  
![](images/9750e4d2244931d27d650594472b5406eb12451157b2b1ad9f5a58439f05bc56.jpg)  
Figure 12: Feature Importance Bar Charts for TDDM-Using Tasks.

![](images/a1c15e163e480502e913942884fbd58d8a97e7dc1b48a73187d9567e035c8c0f.jpg)  
Figure 13: Comparison of Reverse Denoising Processes in IDS, TLS, and USTC Datasets (UG: unguided, G: guided).

This observation highlights a critical limitation of our approach: the effectiveness of TDDM is contingent on the compatibility between the base model's representation learning paradigm and the nature of the augmented features. For models that already incorporate robust pre-training on diverse data distributions, TDDM serves as an effective complementary augmentation strategy. However, for models that rely heavily on task-specific shortcut features or have limited capacity to accommodate distributional shifts, naive application of TDDM may introduce noise that undermines performance. This finding underscores the importance of considering the interaction between data augmentation and model architecture when designing generalizable traffic classification systems, and points to the need for model-aware augmentation strategies in future work.