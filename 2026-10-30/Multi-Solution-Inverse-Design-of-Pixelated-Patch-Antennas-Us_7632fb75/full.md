# Multi-Solution Inverse Design of Pixelated Patch Antennas Using an Autoregressive Transformer

Young-Mok Yun, Seung-Won Shin, Jaeyong Chung, Member, IEEE, and Kyung-Sik Choi, Member, IEEE

Abstract—Deep-learning-based inverse design has emerged as a promising framework for pixelated patch antenna synthesis by enabling broader exploration of discrete geometry spaces and reducing design time compared with conventional templatebased methods. Existing methods, including surrogate-assisted metaheuristic optimization, tandem inverse networks, and diffusion models, have advanced the field but still face practical limitations. Metaheuristic methods such as binary particle swarm optimization (BPSO) are sensitive to initialization and search hyperparameters and are often time-consuming, tandem networks have difficulty capturing the one-to-many nature of inverse antenna design, and diffusion models may require excessive complexity and data for moderate-scale binary layout synthesis. In this paper, an autoregressive (AR) transformer is presented for the inverse design of 10 × 10 pixelated patch antennas. Its sequential discrete generation process enables the synthesis of multiple feasible structures for a given target response. The pretrained model is further fine-tuned for dual-band design using step-shaped target responses with sharp spectral transitions, allowing intuitive band specification without manually constructing detailed S curves. Compared with BPSO, our proposed method generates valid and diverse antenna candidates more efficiently. Fabricated and measured examples confirm that distinct pixelated layouts can realize the desired antenna responses, thereby demonstrating the multi-solution capability of the proposed framework.

Index Terms—Antennas, generative AI, design automation, microstrip antennas, inverse design, multifrequency antennas.

## I. INTRODUCTION

N recent years, machine learning has played an increasingly important role in the design of electromagnetic (EM) and photonic structures, offering new possibilities beyond conventional simulation-driven and template-based methodologies [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]. In antenna engineering, conventional synthesis typically relies on repeated electromagnetic simulations and substantial domain knowledge to tune geometry parameters, making it difficult to scale to large design spaces and complex target specifications. This challenge becomes particularly critical in multiband antenna design, where multiple impedance-matching notches must be precisely aligned with the desired operating frequencies while ensuring sufficient bandwidth and practical radiation characteristics. To overcome the limited flexibility of predefined templates, pixelated antenna design has been explored as a high-degree-of-freedom approach by representing the radiating structure as a binary pixel array [12], [13], [14], [15]. However, this flexibility introduces a large discrete combinatorial design space, making direct trial-and-error or purely simulation-driven optimization inefficient. To address this challenge, deep-learning-based inverse design has been widely investigated, including surrogate-assisted metaheuristic optimization [16], [17], [18], tandem inverse networks [4], [19], [20] for one-shot geometry prediction, and diffusionbased generative models [21], [22], [23] capable of synthesizing multiple feasible candidates for a given target response.

![](images/204d7b9ff7478332f7753d0aca6a23400d422f2b48dc9a71fca5bfc8da3bed25.jpg)  
Fig. 1. LLM-inspired autoregressive framework for antenna inverse design.

Despite recent progress, existing inverse design approaches still have several limitations. Surrogate-assisted iterative optimization can be sensitive to model parameters and may become trapped in local optima [24], [25], while tandem inverse networks typically provide only a single representative solution for a given target, limiting design diversity in inherently nonunique inverse antenna design problems. This limitation is critical in practical antenna design because geometries with similar $S _ { 1 1 }$ responses can differ in radiation characteristics, bandwidth, gain and robustness to fabrication-induced variations. Thus, generating multiple feasible candidates enables designers to rapidly compare alternative layouts and select a final structure that satisfies the target frequency response while also providing favorable bandwidth, radiation performance, and fabrication robustness [26]. While diffusion-based models can also provide multiple candidates, they typically require large datasets and multi-stage training, which is heavier than necessary for moderate-scale pixelated layouts.

To address these issues, this work adopts an autoregressive (AR) transformer that models the conditional distribution of > This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

![](images/2fe993ce82da1410f76e5bf6f72b8c3d0d9df965d1cc0eafb0a44c0423601506.jpg)  
Fig. 2. Overview of the proposed framework.

discrete antenna layouts under a desired target response. As illustrated in Fig. 1, the proposed framework is inspired by large language models, where a sequence of tokens is generated autoregressively conditioned on a given prompt; similarly, the AR transformer performs inverse antenna design through a conditional pixel-by-pixel generation scheme. In this scheme, each antenna pixel is sequentially determined based on the target reflection spectrum and the previously generated pixels, allowing the model to account for inter-pixel dependencies during geometry construction. Unlike deterministic inverse models that directly map a target response to a single layout, this probabilistic generation process learns the conditional joint distribution of the full pixelated structure and can produce multiple valid antenna geometries corresponding to the same target spectrum. Therefore, the proposed pixel generation method naturally reflects the one-to-many solution space inherent in inverse antenna design and provides multiple design alternatives for user-specified single- and dual-band responses. The model is further fine-tuned using step-shaped target responses, enabling intuitive band-level target specification without manually constructing detailed $S _ { 1 1 }$ curves. In this way, the desired operating frequency bands can be specified directly, and multiple antenna candidates can be generated in a short time for subsequent selection based on both impedance matching and practical antenna performance. The proposed framework is validated through single- and dual-band antenna synthesis, comparison with binary particle swarm optimization (BPSO)-based optimization, and fabricated measurement.

The rest of this paper is organized as follows. Section II describes the architectures and training procedures of the proposed inverse and forward models. Section III compares the proposed AR transformer with BPSO in terms of generation efficiency and design performance. Section IV reports fabrication and measurement results of generated antennas. Section V discusses model design considerations and further analysis. Finally, Section VI concludes this paper.

## II. PROPOSED MACHINE LEARNING MODELS

As shown in Fig. 2, the proposed framework consists of candidate generation (inverse network) and performance evaluation (forward network). The inverse network is implemented using an AR transformer, which is well suited for the oneto-many nature of antenna inverse design because multiple pixelated geometries can produce similar reflection responses. Instead of directly mapping a target spectrum to a single deterministic layout, the AR transformer generates the antenna geometry pixel by pixel, where each pixel is conditioned on both the target reflection spectrum and the previously generated pixels. This sequential conditional generation allows the model to capture inter-pixel dependencies and learn the joint distribution of physically meaningful antenna layouts.

![](images/026812adca5747e961f521304b25bca1a7be9fc92166acceaddd679a804b3c81.jpg)  
Fig. 3. Pixel-spectrum paired dataset for training.

A step-function-like $S _ { 1 1 }$ spectrum is used as the input condition to represent the desired operating bands. Based on this target response, the AR transformer generates multiple pixelated antenna candidates. These candidates are subsequently evaluated by a forward surrogate model, which rapidly predicts their reflection spectra and compares them with the target response without requiring full-wave EM simulation for every design. Finally, the candidates that best satisfy the target specification are selected as valid antenna designs.

To construct the training dataset, 500k random pixelated antenna layouts are generated, as illustrated in Fig. 3. Each generated layout is then analyzed using HFSS full-wave EM simulation to obtain the corresponding $S _ { 1 1 }$ response, forming a pixel–spectrum paired dataset. Simulations are parallelized across multiple accounts on two CPU-based computing servers: an Intel Xeon Silver 4310 system at 2.10 GHz with 12 cores and 24 logical processors, and a dual-socket Intel Xeon Gold 6242R system at 3.10 GHz with 40 cores and 80 logical processors. This parallelized dataset generation produces approximately 10k samples per day. The dataset is divided into training, validation, and test sets with a ratio of 80%, 10%, and 10%, respectively, and is used to train both the inverse and forward surrogate models. Detailed simulation settings for dataset generation are described in Section III.

## A. Inverse Model

Fig. 4 illustrates the sequential generation process of the proposed autoregressive transformer. Instead of predicting the entire antenna layout in a single step, the inverse model sequentially generates the antenna layout pixel by pixel. At each generation step shown in Fig. 4, the red pixel denotes the current prediction target, the blue region represents the previously generated pixels, and the gray region indicates the > This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

![](images/f065f680391df43f5fe881908bbe6df2ecdc23a0fd62c30c142335f54259a8e1.jpg)  
Autoregressive generation of pixel x<sub>i</sub>

![](images/285ba6599d42553a300e9618937ba99c7946a3aa9e19588cd52c0960f5d52456.jpg)  
Fig. 4. Illustration of the autoregressive pixel generation process.

pixels that remain ungenerated. Given the target spectrum and the previously generated pixel configuration, the model determines whether the current pixel should be present or absent in the antenna layout. The selected value is then appended to the partial layout and used as additional context for the next prediction step.

Fig. 5 illustrates the architecture of the proposed inverse model. A target input reflection coefficient $( S _ { 1 1 } )$ spectrum is first encoded by a 1-D ResNet-based spectral encoder into a compact condition vector. Through residual blocks and global average pooling operation, the encoder extracts spectral features that represent the desired antenna response. The encoded condition vector is then broadcast to all sequence positions and added to the token embeddings and positional embeddings. In this way, the spectral condition is injected into the transformer at every pixel-generation step, so that each next-pixel prediction is guided by both the target response and the partial pixel layout generated up to the current step.

Since the inverse model generates the antenna layout in an autoregressive manner, the two-dimensional pixel grid must be converted into an ordered one-dimensional sequence before training. This serialization defines the generation order of the pixels, allowing the model to predict each pixel based on the target spectrum and the previously generated pixels. Thus, the model learns layout generation as sequential next-pixel prediction rather than one-shot prediction of the entire pixel pattern. The $1 0 \times 1 0$ layout is represented as a one-dimensional sequence of 100 binary values, denoted as $[ x _ { 1 } , x _ { 2 } , \dotsc , x _ { 1 0 0 } ]$ ], where each $x _ { i }$ represents the state of the i-th pixel after serialization. Here, $x _ { i } = 1$ and $x _ { i } = 0$ indicate the presence and absence of a metallic patch element at the i-th pixel position, respectively.

![](images/a034d0eb990c9c8900aba241d07c77e449bf09dc9e43855f0331234e621cd7a9.jpg)  
Fig. 5. Detailed architecture of the AR inverse model.

Before serialization, canonicalization [27], [28] under left– right symmetry is applied to the two-dimensional pixel layout. For each binary layout, its horizontally flipped counterpart is generated. The two layouts are then converted into binary sequences, and the one that comes first in lexicographic order is selected as the canonical target. Note that this procedure differs from data augmentation. Instead, it reduces ambiguity in the target space by mapping symmetry-equivalent layouts to a single representative form. This is especially important in the early generation steps, because two left–right symmetric antenna layouts can produce identical spectral responses, but they may correspond to different binary sequences. By using canonicalization, the model receives consistent supervision and the ambiguity of the inverse mapping is greatly reduced.

Bit-level accuracy is defined as the ratio of correctly predicted pixels in the binary layout sequence and is used to evaluate how accurately the inverse model predicts each pixel position. As shown in Fig. $^ { 6 , }$ the bit-level accuracy progressively improves during training and converges to a > This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

![](images/f81ea70cb5ee2bbb529c4f4a06d3ed329f03be3dabe21a53bf03d81bd0a89395.jpg)  
Fig. 6. Bit-level accuracy of the AR inverse model during training.

validation accuracy of approximately 0.70 as the AR inverse model learns the sequential pixel distribution of the antenna layouts. This indicates that the model gradually captures the conditional relationship between the target spectrum and the sequential generation of the corresponding pixel layout.

Following the canonicalization stage in Fig. 5, the binary layout is serialized into a one-dimensional sequence using an ordering strategy designed to preserve spatial relationships between neighboring pixels in the original two-dimensional grid. Raster, snake, and Hilbert-curve orderings are compared [29], and the best-performing strategy is adopted in the final model. This allows the model to learn local geometric dependencies between neighboring pixels while still modeling the global sequential dependency of the entire layout.

During training, a beginning-of-sequence (BOS) token is added at the beginning of the input sequence. The BOS token is a special start token which informs the model that pixel generation is about to begin. Since the first pixel $x _ { 1 }$ has no previous pixel before it, the model uses the BOS token as the initial input when predicting $x _ { 1 }$ . The training input sequence is therefore constructed as

$$
[ \mathrm { B O S } , x _ { 1 } , x _ { 2 } , \dots , x _ { 9 9 } ] ,
$$

and the corresponding target sequence is

$$
[ x _ { 1 } , x _ { 2 } , x _ { 3 } , \ldots , x _ { 1 0 0 } ] .
$$

In other words, the model learns to predict the next pixel at each sequence position. For example, the BOS token is used to predict $x _ { 1 } .$ , the previous sequence $[ \mathrm { B O S } , x _ { 1 } ]$ is used to predict $x _ { 2 } .$ , and the previous sequence $\left[ \mathrm { B O S } , x _ { 1 } , x _ { 2 } , \dots , x _ { 9 9 } \right]$ is used to predict the final pixel $x _ { 1 0 0 }$

The serialized input sequence is represented by the sum of token embeddings, positional embeddings, and the spectral condition vector. The token embedding represents whether the previous input token corresponds to the presence or absence of a metallic patch element, or to the BOS token. The positional embedding provides information about the location of each token in the serialized sequence. The spectral condition vector is added to every token position so that each pixel prediction is conditioned on the same target $S _ { 1 1 }$ spectrum.

The resulting embeddings are processed by 15 stacked causal transformer layers with a 512-dimensional hidden representation. Each layer employs 4-head self-attention to capture dependencies among previously generated pixels from multiple representation subspaces. The self-attention output is then refined through residual connection, layer normalization, and a position-wise feed-forward network. In the trained model, the feed-forward network has an internal dimension of 768, allowing the 512-dimensional token representation at each pixel position to be expanded and nonlinearly transformed before being mapped back to the transformer hidden dimension. This stacked causal structure enables the model to generate each pixel based on the target spectrum and the previously generated pixels, without using information from future pixel positions [30].

![](images/691a8d4ad6f73c5df6c473372383727b15af6e79374049b951a6bfcc75d655ce.jpg)  
Fig. 7. BCE losses of the AR inverse model.

The model is trained by minimizing the binary cross-entropy (BCE) loss [31] between the predicted bit probabilities and the ground-truth binary pixel sequence. As shown in Fig. 7, the training loss, represented by the blue curve, steadily decreases over 50 epochs, while the validation loss, represented by the red curve, decreases from approximately 0.64 to 0.55. These trends indicate that the autoregressive model successfully learns the conditional pixel-generation process and generalizes to unseen validation samples. The training was performed on an NVIDIA RTX 4000 Ada Generation GPU and took approximately 18 hours.

For dual-band synthesis, the autoregressive model is further fine-tuned using step-function-like target responses with abrupt spectral transitions. This is motivated by the practical observation that users often know the desired operating bands, but may not know the exact spectral transition shape, notch profile, or perturbation level that should be embedded in the target response [32]. By representing the design objective with simplified step-function-like targets, the proposed approach allows users to specify the operating bands more intuitively, without manually constructing detailed $S _ { 1 1 }$ curves. As a result, the model can be adapted to dual-band inverse design in a more practical and user-friendly manner while maintaining the sequential generation framework learned from the initially trained model.

Fig. 8 illustrates the fine-tuning process of the proposed AR inverse model. From the original HFSS dataset, antenna layouts exhibiting single- or dual-band resonance charac-> This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

![](images/c603800d191bf66174e357ce1f438a1dc6bde07768201df334fc6eaa6a95b4de.jpg)  
Fig. 8. Fine-tuning process of the AR inverse model.

teristics are selected. The corresponding $S _ { 1 1 }$ responses are then converted into simplified step-function targets while the associated pixel layouts remain unchanged. For each resonance band, a rectangular notch with a fixed width of 100 MHz and a magnitude of −20 dB is assigned, while the remaining frequency region is set to 0 dB. To improve robustness to small variations in the target-frequency specification, the step window is not always centered exactly at the original resonance minimum. Instead, its position is randomly perturbed such that the original resonance minimum lies within the central 25% of the step window.

The pretrained AR transformer is subsequently adapted to the resulting step targets. Since the pretrained model has already learned the autoregressive dependencies among antenna pixels from continuous $S _ { 1 1 }$ responses, fine-tuning is performed using different learning rates for the spectral condition encoder and the AR generation body. Specifically, a larger learning rate of $2 \times 1 0 ^ { - 5 }$ is applied to the condition encoder to accommodate the domain shift from continuous spectra to simplified step representations, whereas a smaller learning rate of $5 \times 1 0 ^ { - 6 }$ is applied to the pretrained Transformer and output layers to preserve the previously learned pixel-generation knowledge. The same canonicalization and snake serialization scheme used during pretraining is retained.

The model is fine-tuned using the BCE loss for up to 100 epochs with a batch size of 128. A 2-epoch learning-rate warm-up followed by cosine decay is employed, together with gradient clipping of 1.0 and early stopping with a patience of 15 epochs. The training converged through early stopping at epoch 37 and required approximately one hour. This finetuning strategy enables the pretrained AR model to adapt from original spectral conditioning to intuitive step specifications.

![](images/b5198b96fa96fe1d0b045094cd914b3daa8857bcd946f1031b7745646a9f9d09.jpg)  
Fig. 9. Forward surrogate model architecture.

![](images/d66b3ff260eb8e6ab4fae141c5570854339f6dcb831f670958e634c7465e6317.jpg)  
(a)

![](images/3c4550e7bd24140c0969a096b1a62ef9ed4c661f848ac72c17ef1119baba9adf.jpg)  
(b)  
Fig. 10. Forward surrogate model training: (a) Loss curves (b) $S _ { 1 1 }$ heatmap.

## B. Forward Surrogate Model

In contrast to the inverse problem, the forward problem aims to predict the $S _ { 1 1 }$ spectrum from a given binary pixel layout. Since this task is a deterministic mapping from geometry to response, the main requirement is to efficiently extract useful local spatial features from the layout. For this reason, a residual convolutional neural network (CNN) [33] is employed as the forward surrogate model.

As shown in Fig. 9, the input binary layout is represented as a single-channel $1 0 \times 1 0$ image. The network consists of multiple residual convolutional stages, where each residual block contains two $3 \times 3$ convolution layers, batch normalization, and nonlinear activation, together with a projection shortcut when the channel dimension changes. Through these stacked residual blocks, the model learns hierarchical spatial features from the pixelated antenna geometry. After feature extraction, adaptive global average pooling (GAP) is applied, and the resulting global feature vector is mapped to the full $S _ { 1 1 }$ spectrum through a fully connected (FC) layer.

The motivation for using a CNN in the forward surrogate model is fundamentally different from that of the transformer in the inverse model. In the forward problem, useful local geometric patterns that appear in one region of the pixel layout are also expected to be useful in other regions. Convolution is well suited to this role because it exploits local receptive fields and shared filters, thereby providing translation-aware feature extraction with relatively high parameter efficiency. Therefore, while the inverse model requires explicit modeling of pixelto-pixel dependencies and conditional joint probability, the forward model mainly requires robust extraction of repeated local spatial features for deterministic spectrum regression.

The forward surrogate is trained in a supervised manner using the mean squared error (MSE) between the predicted and target spectra. Specifically, for a given pixelated layout, the network predicts the full $S _ { 1 1 }$ response, and the MSE loss is computed over all frequency points. The model is trained for 110 epochs using the same GPU environment as the inverse > This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

![](images/d76dbe75e244a1cf9f2c0305c10a2d921885e44004fd9bb4acb41fe4b7d25558.jpg)  
Fig. 11. Pixelated patch antenna geometry.

network training. The AdamW optimizer is used with an initial learning rate of $1 \times 1 0 ^ { - 3 }$ and a weight decay of $1 \times 1 0 ^ { - 4 }$ A learning-rate decay strategy is applied to prevent the model from converging to local minima and to guide it toward the global minimum. A batch size of 128 is used during training.

The training and validation loss curves are shown in Fig. 10(a). The validation loss decreases from an initial value of 2.66 to approximately 0.46, and the training is terminated at 110 epochs as no further meaningful improvement in the validation loss is observed, indicating that the residual CNN successfully learns the mapping from the binary pixel layout to the corresponding $S _ { 1 1 }$ spectrum. Although a slight degree of overfitting is observed, increasing the number of model parameters tends to improve the performance in this forward regression task. This suggests that sufficient model capacity is beneficial for capturing the complex relationship between the pixelated antenna geometry and the resulting EM response. The total training time is approximately 15 hours.

The prediction accuracy of the trained forward surrogate is further evaluated using the heatmap in Fig. 10(b), where the horizontal axis represents the predicted $S _ { 1 1 }$ value and the vertical axis represents the simulated $S _ { 1 1 }$ value. As shown in the heatmap, the high-density region is well aligned around this diagonal line, especially in the major response region, indicating that the model has learned the overall spectral trend sufficiently. The overall coefficient of determination reaches $R ^ { 2 } = 0 . 9 2$ , confirming a strong correlation between the predicted and simulated spectra. In addition, the notchregion mean absolute error (MAE) is approximately 2.13 dB, showing that the surrogate model provides a reasonably accurate approximation of the resonance-depth behavior. Once trained, the forward surrogate provides a fast approximation of the EM response of candidate pixel layouts and can therefore be used to efficiently evaluate or rank designs generated by the inverse model.

## III. IMPLEMENTATION DETAILS

## A. EM Simulation Setup and Dataset Generation

To construct the pixel–spectrum paired dataset, full-wave EM simulations are performed using the pixelated patch antenna structure shown in Fig. 11. The antenna is designed around a center frequency of 6 GHz using Rogers 4350B as the substrate material, with a relative dielectric constant of $\epsilon _ { r } = 3 . 6 6$ and a thickness of 1.52 mm. The overall substrate size is set to 20.48 mm × 20.48 mm, and the pixelated patch region is defined as an 11 mm × 11 mm square region. This patch size is determined through a preliminary patch-size scaling analysis, in which 1000 EM simulations are conducted for each scaling condition to form a resonance near 6 GHz. The feed width of 3.142 mm is employed to achieve 50-Ω characteristic impedance. Note that an overlap of 0.25 mm is introduced between neighboring pixels to ensure electrical connectivity [17], [34], [35].

![](images/87375cd2741cb14012d51a59a7e2c3e8616e366891141dbb18dc7062885c2b1b.jpg)  
Fig. 12. Resonance ratio by density and spectrum distribution at 60% density.

To select an appropriate pixel density for dataset generation, the resonance occurrence rate is analyzed under different density conditions, as shown in Fig. 12. Similar to the patch-size scaling analysis, 1000 EM simulations are conducted for each density condition. Among the tested density values, the 60% density condition exhibits one of the highest resonance ratios. Therefore, this density is selected to increase the number of resonant antenna patterns in the dataset, which is important for enabling the model to effectively learn meaningful notch responses [36]. Based on this condition, each pixelated antenna sample is generated by randomly selecting 60 active pixels among the 100 pixel positions.

For each generated layout, a Python script automatically controls the HFSS EM simulation process by repeatedly executing the required simulation macro. The simulated $S _ { 1 1 }$ is extracted over the frequency range from 5 to 7 GHz with 201 frequency points. The resonance statistics of the generated 60%-density dataset are also summarized in Fig. 12. As can be observed from the figure, 73.29% of the generated samples are classified as non-resonant responses, 26.02% as singleband responses, and 0.69% as dual-band responses.

## B. Comparison of Searching Algorithms

To determine an effective inference strategy for the proposed AR transformer, three searching algorithms are compared: random sampling [37], beam search [38], and multiarmed bandit (MAB) [39] based search. Random sampling generates antenna layouts probabilistically from the pixelwise output distribution of the AR transformer, which pro-> This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

![](images/f2690f266cab6ff666e2871337e83b9332eb663c4a1ba835b1f879c851ffa761.jpg)  
Fig. 13. Conceptual illustration of the multi-armed bandit (MAB) problem.

vides high candidate diversity but does not directly guide the generation toward high-performance antenna responses. Beam search retains multiple high-probability generation paths during autoregressive decoding, allowing likely layouts to be generated efficiently. However, because it repeatedly favors high-likelihood paths, the generated candidates can become less diverse [40], [41].

The MAB-based search is adopted to guide candidate generation more adaptively. As illustrated in Fig. 13, the general MAB problem consists of an agent and multiple selectable arms. At each trial, the agent selects one arm, observes the corresponding reward, and updates its future selection strategy based on the accumulated rewards. This framework is useful for balancing exploration of less-tested options and exploitation of options that have already shown favorable results [42], [43]. In this work, the MAB concept is applied to the prefix selection of the AR transformer. Several initial binary prefix patterns are treated as selectable arms, and the remaining pixels are generated by stochastic autoregressive sampling. Therefore, the MAB-based search adaptively allocates more trials to prefix regions that are likely to produce high-quality antenna candidates. The upper confidence bound criterion is used to balance exploration and exploitation.

Each generated candidate is evaluated using the forward surrogate model, and the candidate MSE is computed over the full $S _ { 1 1 }$ spectrum. A candidate is regarded as valid when its predicted $S _ { 1 1 }$ satisfies the target notch condition at −12 dB, stricter than the conventional −10 dB, to ensure a practical bandwidth margin. To quantitatively compare the three searching algorithms, the same 334 unseen dual-band test cases are used for comparison. Since multiple algorithms can find valid candidates for the same target, a simple binary criterion based only on whether a valid candidate is found is insufficient. Therefore, each algorithm is evaluated using the best candidate found within the same 600-second search budget. For each target case, the lowest full-band MSE among the valid candidates is used as the representative performance, and the algorithms were compared using this metric. A lower MSE indicates that the algorithm finds a candidate whose predicted response more closely matches the target.

![](images/6b146502ac953d0fd027d736e89cb4260409748574d3269251ba4dc7303b59bb.jpg)  
(a)

![](images/0aab38efa04809703bdc62ddc72af552f3c8e6869029c41ef88eb338951b0589.jpg)  
(b)

![](images/292af871a2561e22223fc6a862cf460a915ed04839f1941c8e4a01d359a7d7d4.jpg)  
(c)  
Fig. 14. Heatmaps of unique valid-candidate counts for searching algorithms: (a) multi-armed bandit, (b) random sampling, and (c) beam search.

TABLE I  
PERFORMANCE COMPARISON OF SEARCHING ALGORITHMS
<table><tr><td rowspan=1 colspan=1>Algorithm</td><td rowspan=1 colspan=1>Number oftop-ranked cases</td><td rowspan=1 colspan=1>Average full-band MSEof best valid candidate</td></tr><tr><td rowspan=1 colspan=1>Multi-Armed Bandit</td><td rowspan=1 colspan=1>234 / 334</td><td rowspan=1 colspan=1>0.92</td></tr><tr><td rowspan=1 colspan=1>Random Sampling</td><td rowspan=1 colspan=1>78 / 34</td><td rowspan=1 colspan=1>1.19</td></tr><tr><td rowspan=1 colspan=1>Beam Search</td><td rowspan=1 colspan=1>22 / 334</td><td rowspan=1 colspan=1>1.67</td></tr></table>

Fig. 14 compares the number of unique valid candidates generated by the three searching algorithms for the same dualband test cases. Here, $f _ { 1 }$ and $f _ { 2 }$ denote the lower and higher target resonance frequencies, respectively. Since each dualband target is defined such that $f _ { 1 } < f _ { 2 }$ , the lower triangular region corresponding to $f _ { 1 } \geq f _ { 2 }$ does not represent valid dualband combinations and is therefore shown as a hatched region. In the valid upper triangular region, a higher candidate count indicates that a larger number of distinct valid antenna layouts are obtained for the corresponding target, whereas black points indicate cases in which no unique valid candidate is found. The MAB-based search exhibits broader coverage and fewer unsuccessful cases than random sampling and beam search.

Table I summarizes the comparison results. Here, “Number of top-ranked cases” denotes the number of target cases in which each algorithm achieved the lowest full-band MSE. The MAB-based search achieves the largest number of best cases, with 234 out of 334 cases. It also achieves the lowest average full-band MSE of the best valid candidate at 0.92, indicating that the MAB-based search can identify high-quality candidates more effectively than other searching algorithms.

## C. Comparison of BPSO and the Proposed AR Transformer

To quantitatively evaluate the candidate generation capability of the proposed inverse design model, the generation performance of BPSO and the proposed AR transformer is compared. BPSO is a widely used population-based optimization algorithm for discrete design problems, but its search performance is strongly affected by the initial condition and parameter tuning [25], [44], [45]. In particular, multiple independent optimization runs are often required to obtain several valid antenna geometries for a given target response, which can increase the overall generation time. Therefore, the comparison is conducted to numerically demonstrate the difference in generation efficiency.

> This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

![](images/b13bdad2e03a194af7b573eb5d77439b07e3d07456dcf5b289371099fa4c0f6f.jpg)  
(a)

![](images/28250a8167b0af5c23aa2fdfee0962a97fee58852face660d35a26351441e2a2.jpg)  
(b)

Fig. 15. Inverse design workflows of (a) BPSO-based iterative optimization and (b) AR transformer-based direct generation.  
![](images/c431430935f049ba1e3676cb65c30fe3a89d5e06e0d6b3a5b0a20e4019768211.jpg)  
(a)

![](images/e21341e2e2f8b2cd0cbf6e8c076a25f0ee708e79495a69e288c128abb91f6fdc.jpg)  
(b)

![](images/7432cf47e4a18673c64649b5eb169fe3251acb806529e3dd4c371d501b63e806.jpg)  
(c)  
Fig. 16. Heatmaps of unique valid-candidate counts over 334 dual-band test cases: (a) pretrained AR model, (b) BPSO best, and (c) BPSO worst.

Fig. 15 illustrates the difference between the two inverse design workflows. In the BPSO-based approach, candidate antenna geometries are obtained through iterative particle updates, and multiple optimization runs are generally required to obtain diverse valid solutions for the same target spectrum. In contrast, the proposed AR transformer directly synthesizes pixelated antenna layouts conditioned on the target $S _ { 1 1 }$ response through sequential pixel generation, enabling multiple distinct candidates to be generated without repetitive optimization.

For a comprehensive quantitative comparison, the pretrained AR model is compared with conventional BPSO and its variants on 334 unseen dual-band targets from the held-out test set using the same budget of 20,000 evaluated candidates per target. To account for the parameter sensitivity of conventional BPSO, 36 hyperparameter settings are evaluated by varying the inertia weight w, cognitive acceleration coefficient $c _ { 1 }$ and social acceleration coefficient $c _ { 2 }$ within the recommended ranges based on widely adopted PSO tuning guidelines [46], [47], [48]. Each setting is evaluated over all test cases, and the settings achieving the highest and lowest coverage are reported as BPSO (best) and BPSO (worst), respectively. Vshaped BPSO (VBPSO) [12] and Modified V-shaped BPSO (MVBPSO) [49] are additionally evaluated.

TABLE II  
COMPARISON OF THE PRETRAINED AR MODEL AND BPSO VARIANTS
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Successfulcases</td><td rowspan=1 colspan=1>Coverage(%)</td><td rowspan=1 colspan=1>Redundant validcandidates (%)</td><td rowspan=1 colspan=1>Average bestfull-band MSE</td></tr><tr><td rowspan=1 colspan=1>Pretrained AR</td><td rowspan=1 colspan=1>322/334</td><td rowspan=1 colspan=1>96.4</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.89</td></tr><tr><td rowspan=1 colspan=1>BPSO (best)</td><td rowspan=1 colspan=1>202/334</td><td rowspan=1 colspan=1>60.5</td><td rowspan=1 colspan=1>20.4</td><td rowspan=1 colspan=1>7.67</td></tr><tr><td rowspan=1 colspan=1>BPSO (worst)</td><td rowspan=1 colspan=1>171/334</td><td rowspan=1 colspan=1>51.2</td><td rowspan=1 colspan=1>34.2</td><td rowspan=1 colspan=1>9.05</td></tr><tr><td rowspan=1 colspan=1>VBPSO</td><td rowspan=1 colspan=1>184/334</td><td rowspan=1 colspan=1>55.1</td><td rowspan=1 colspan=1>13.8</td><td rowspan=1 colspan=1>4.18</td></tr><tr><td rowspan=1 colspan=1>MVBPSO</td><td rowspan=1 colspan=1>192/334</td><td rowspan=1 colspan=1>57.5</td><td rowspan=1 colspan=1>71.3</td><td rowspan=1 colspan=1>3.50</td></tr></table>

As summarized in Fig. 16 and Table II, the pretrained AR model provides substantially broader target-space coverage than conventional BPSO, VBPSO, and MVBPSO, while producing non-redundant valid solutions. It also achieves the lowest average best full-band spectrum MSE among the evaluated methods, demonstrating more accurate and diverse multi-solution generation across the dual-band target space.

## D. Fine-Tuned Model Evaluation

The fine-tuned model is evaluated using 153 dual-band stepfunction targets with band centers from 5.1 to 6.9 GHz at 0.1- GHz intervals. Adjacent centers are excluded so that two 100- MHz notches remain separated; each notch is fixed at −20 dB with 0 dB elsewhere. As in the pretrained-model evaluation, the 36 BPSO hyperparameter settings are evaluated over all 153 targets, and the settings with the highest and lowest coverage are reported as BPSO (best) and BPSO (worst), respectively. VBPSO and MVBPSO are additionally evaluated using the same target set and candidate budget.

As summarized in Fig. 17 and Table III, the fine-tuned AR model achieves substantially broader target-space coverage than conventional BPSO, VBPSO, and MVBPSO while producing non-redundant valid solutions. It also provides the > This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

![](images/115ae218516ca7756c883d0c1f08aaf03ddedab6f450780800aacaab7e190757.jpg)  
(a)

![](images/71a935bf1d210889d8c6446cf2370b774809db16c4e71e10caf9d709b8874dfe.jpg)  
(b)

![](images/38fff87b6eb8876a1cb3d5e7a44b18c1930814e09cb639e36fedfa13b0be2b5c.jpg)  
(c)

Fig. 17. Heatmaps of unique valid-candidate counts for step-function targets: (a) fine-tuned AR model, (b) BPSO best, and (c) BPSO worst.  
![](images/d081a56775b1d2cc59df0689ee0cddd3e7ec622ef21c664a39590b42d5f4f6c2.jpg)

TABLE III  
COMPARISON OF THE FINE-TUNED AR MODEL AND BPSO VARIANTS
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Successfulcases</td><td rowspan=1 colspan=1>Coverage(%)</td><td rowspan=1 colspan=1>Redundant validcandidates (%)</td><td rowspan=1 colspan=1>Average bestnotch MSE</td></tr><tr><td rowspan=1 colspan=1>Fine-tuned AR</td><td rowspan=1 colspan=1>111/153</td><td rowspan=1 colspan=1>72.5</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>40.15</td></tr><tr><td rowspan=1 colspan=1>BPSO (best)</td><td rowspan=1 colspan=1>59/153</td><td rowspan=1 colspan=1>38.6</td><td rowspan=1 colspan=1>3.5</td><td rowspan=1 colspan=1>55.66</td></tr><tr><td rowspan=1 colspan=1>BPSO (worst)</td><td rowspan=1 colspan=1>41/153</td><td rowspan=1 colspan=1>26.8</td><td rowspan=1 colspan=1>37.0</td><td rowspan=1 colspan=1>48.31</td></tr><tr><td rowspan=1 colspan=1>VBPSO</td><td rowspan=1 colspan=1>43/153</td><td rowspan=1 colspan=1>28.1</td><td rowspan=1 colspan=1>27.0</td><td rowspan=1 colspan=1>59.78</td></tr><tr><td rowspan=1 colspan=1>MVBPSO</td><td rowspan=1 colspan=1>61/153</td><td rowspan=1 colspan=1>39.9</td><td rowspan=1 colspan=1>74.6</td><td rowspan=1 colspan=1>52.91</td></tr></table>

lowest average best notch-region MSE among the evaluated methods, demonstrating more accurate and diverse multisolution generation for simplified step-function specifications.

## E. Prediction and Simulation Comparison

The generation capability of the pretrained AR transformer is first evaluated using single- and dual-band target spectra. Three representative single-band and three representative dualband test cases are evaluated, as shown in Figs. 18 and 19, respectively. The generated layouts exhibit resonant notches close to the target frequencies, while the predicted spectra show good agreement with the full-wave EM simulation results.

![](images/66955f06f398eb23a0b4edcd957ae2f47cd78a0bfc449286b2de5f45a0bb58ed.jpg)

To further verify the one-to-many generation capability, the fine-tuned AR transformer is evaluated using step-function targets. The single-band target is specified at 6 GHz, while the dual-band is specified at 5.9 and 6.9 GHz. For each selected target, three distinct antenna candidates are generated, as shown in Figs. 20 and 21. Although the generated layouts have different pixel configurations, they produce similar responses corresponding to the same target specification. These results demonstrate that the fine-tuned model can generate multiple distinct solutions for both single- and dual-band step targets.

## IV. MEASUREMENT RESULTS

To experimentally validate the antenna designs, the antennas are fabricated according to the optimized geometries obtained from the HFSS simulation setup. The reflection coefficients are measured using an Anritsu MS46522B 40 GHz vector network analyzer (VNA). The radiation characteristics are measured in an anechoic chamber using the measurement setup shown in Fig. 22. In this setup, a transmitting horn antenna is used as the reference source, while the fabricated antenna under test (AUT) is placed at the receiving position.

Figs. 23–25 show the measured characteristics of the three fabricated 6-GHz single-band antennas. The fabrication and measurement of these antennas are conducted to experimentally verify the multiple single-band solutions obtained from the simulation results in Fig. 20. The measured $S _ { 1 1 }$ responses exhibit resonances near 6 GHz and show good agreement with the HFSS simulation results. The measured radiation patterns > This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

![](images/e7fe41428d1cfa7bcd4d8231d67cd67c28dd6cd055062db551b62bf9f15850c9.jpg)

![](images/a2faa0bebbf37f744a90495a10e0982c9257469f5413bcde2b6f1c362d21e3a0.jpg)

![](images/f43410cc13b1faef2abe261f70e97788ef26a1fe8b3d6b5746b317aab8487d19.jpg)

![](images/4d71718d626a28183d2827385009a77f519e2d0f70e11c6e2174a63dd7d86540.jpg)  
Fig. 18. Single-band antenna responses generated by the pretrained AR model.

![](images/fb0503fb3975d4769d86ce7f9a85945b1bdf06d308ac29836d04cca38f3d7edf.jpg)

![](images/e1d0dd474bf4092a40b4e84765cea4e68ef0fb35501235eaa2ba5bc36b4e0a22.jpg)

![](images/97d2cba3ca7bd469070bce1a02606b91e89766fcd69ae009202eef7ed2350ee1.jpg)

![](images/d1ccafcfdb607b93e21428821907301eebfd9f2a41e36d4d159e2a3589d97f62.jpg)  
Fig. 19. Dual-band antenna responses generated by the pretrained AR model.

![](images/dd9e709648f2670fd21a7724997f13ac87e9c786273e32e3c0e759bb53f336bf.jpg)

![](images/5744495171a9c9d7080bfd4ebf8c582ad6ebb444ee8c8dbdd21dd4e7185d0e2d.jpg)

![](images/2429b90cfa706265d7b92ced74f55bafc1deb575fb0d0d2de4b88ea6a323cb4d.jpg)

![](images/4387daef123b9c76ca8c5093a336badb5a30e409aa8d3d8183bfd88e51236d50.jpg)

![](images/87ebe52bfc2cf678dea07697e8ade39cc6b40c4b76737da7a38907928f102524.jpg)

![](images/17bcea485443ecbe7a086fa83d5cd0a07a0ef459a1e6b3634702ff59aa58447d.jpg)

![](images/464be3bc8407cb812966cea1214e48fdc63f65c0b3edd8af03aa33f70853df33.jpg)  
(a)  
Fig. 20. Multiple solutions for a single-band step-function target.

![](images/55f38c5701cbcc324d617970fef59123d86563e79367621076a6840a331b6f6d.jpg)

![](images/5716f5c0c8dc097ff2f53b895aca315a37d5d11edcfa2ef97b57d311cea55e0e.jpg)

![](images/2a0efefa38af7125e9212b1b0842b94373e7bf987ddb903a78f643b0ccb328bd.jpg)

![](images/4930e42f4f1861ae07253e6c2673cbeddd40e72fde34529cb9d16352c3530811.jpg)  
Fig. 21. Multiple solutions for a dual-band step-function target.

at $\phi \ : = \ : 0 ^ { \circ }$ and $\phi \ : = \ : 9 0 ^ { \circ }$ also follow the simulated results and exhibit broadside radiation characteristics. The measured resonant frequency $f _ { r }$ , bandwidth BW, and realized gain of each design are summarized in Table IV, confirming that all three antennas maintain consistent single-band performance

Fig. 22. Anechoic chamber setup for radiation pattern measurement.  
(b)  
![](images/cb1ce5527b278fd6574e10ff3647dd815f27f79a074bf7a3f576d432f4757378.jpg)

![](images/964928d0e6859fa9b8db4f9c6db1e17c130fff3e5bad730eb886b97a41dea999.jpg)

![](images/611acef4ea167cf64c168b76091b15e65264ff0fdda8b19fe7381a2973c0eaef.jpg)  
Fig. 23. Measured results of the first 6-GHz single-band antenna: (a) fabricated antenna, (b) $S _ { 1 1 }$ response, (c) Elevation radiation pattern $( \phi = 0 ^ { \circ } )$ at 6.01 GHz, (d) Elevation radiation pattern $( \phi = 9 0 ^ { \circ } )$ at 6.01 GHz.

TABLE IV  
MEASURED PERFORMANCE OF 6-GHZ SINGLE-BAND ANTENNAS
<table><tr><td rowspan=2 colspan=1>Design</td><td rowspan=2 colspan=1> $f _ { r }$ (GHz)</td><td rowspan=2 colspan=1>BW (MHz)</td><td rowspan=1 colspan=2>Realized Gain (dBi)</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \phi = 0 ^ { \circ } } }$ </td><td rowspan=1 colspan=1> $\overline { { \phi = 9 0 ^ { \circ } } }$ </td></tr><tr><td rowspan=1 colspan=1>First</td><td rowspan=1 colspan=1>6.01</td><td rowspan=1 colspan=1>170.20</td><td rowspan=1 colspan=1>5.38</td><td rowspan=1 colspan=1>5.67</td></tr><tr><td rowspan=1 colspan=1>Second</td><td rowspan=1 colspan=1>5.99</td><td rowspan=1 colspan=1>192.13</td><td rowspan=1 colspan=1>5.52</td><td rowspan=1 colspan=1>5.74</td></tr><tr><td rowspan=1 colspan=1>Third</td><td rowspan=1 colspan=1>6.02</td><td rowspan=1 colspan=1>139.36</td><td rowspan=1 colspan=1>5.32</td><td rowspan=1 colspan=1>5.98</td></tr></table>

around the target frequency.

To experimentally validate the dual-band solutions obtained from the fine-tuned AR model, three generated dual-band antennas shown in Fig. 21 are fabricated and measured. Figs. 26–28 show the measured characteristics of the first, second, and third generated dual-band antennas, respectively. The measured $S _ { 1 1 }$ responses exhibit two impedance-matching notches near the intended operating bands and show good agreement with the HFSS simulation results. The measured radiation patterns generally follow the simulated results and > This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

(a)  
![](images/324b152fe4116f1b4a20115df0612954d64511f2ef3dbc564a5b279a3281ed67.jpg)  
(b)

![](images/089e8c77c4dd657adb7a66cbb081e7626afe47a702b1fd91a9bc4ecf6f98da83.jpg)  
(a)

![](images/a3400af4c73772c3c818e2f63ae71565c0db58d403671e14846c52dac13fe726.jpg)  
Fig. 24. Measured results of the second 6-GHz single-band antenna: (a) fabricated antenna, (b) $S _ { 1 1 }$ response, (c) Elevation radiation pattern $( \phi = 0 ^ { \circ } )$ at 5.99 GHz, (d) Elevation radiation pattern $( \phi = 9 0 ^ { \circ } )$ at 5.99 GHz.

(a)  
![](images/ea1644380f074be488a81f06156464c4c1e0264b180708c236525bd1675b6891.jpg)

(b)  
![](images/bfa99b74d4776b0f5a21b8f4c6fae62dc292590dc7b887f7bcad79e2c3294b65.jpg)

![](images/e23962086cb73c54140399d97b45eef49b36d2576034e55e90b99a827f046434.jpg)  
Fig. 25. Measured results of the third 6-GHz single-band antenna: (a) fabricated antenna, $( \mathsf { b } ) \ S _ { 1 1 }$ response, (c) Elevation radiation pattern $( \phi = 0 ^ { \circ } )$ at 6.02 GHz, (d) Elevation radiation pattern $( \phi = 9 0 ^ { \circ } )$ at 6.02 GHz.

TABLE V  
MEASURED PERFORMANCE OF DUAL-BAND ANTENNAS
<table><tr><td rowspan=2 colspan=1>Design</td><td rowspan=2 colspan=1>Band</td><td rowspan=2 colspan=1> $f _ { r }$ (GHz)</td><td rowspan=2 colspan=1>BW (MHz)</td><td rowspan=1 colspan=2>Realized Gain (dBi)</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \phi = 0 ^ { \circ } } }$ </td><td rowspan=1 colspan=1> $\phi = 9 0 ^ { \circ }$ </td></tr><tr><td rowspan=2 colspan=1>First</td><td rowspan=1 colspan=1>Lower</td><td rowspan=1 colspan=1>5.91</td><td rowspan=1 colspan=1>106.85</td><td rowspan=1 colspan=1>5.81</td><td rowspan=1 colspan=1>5.91</td></tr><tr><td rowspan=1 colspan=1>Upper</td><td rowspan=1 colspan=1>6.85</td><td rowspan=1 colspan=1>152.08</td><td rowspan=1 colspan=1>4.46</td><td rowspan=1 colspan=1>4.35</td></tr><tr><td rowspan=2 colspan=1>Second</td><td rowspan=1 colspan=1>Lower</td><td rowspan=1 colspan=1>5.94</td><td rowspan=1 colspan=1>89.53</td><td rowspan=1 colspan=1>5.26</td><td rowspan=1 colspan=1>5.33</td></tr><tr><td rowspan=1 colspan=1>Upper</td><td rowspan=1 colspan=1>6.88</td><td rowspan=1 colspan=1>169.03</td><td rowspan=1 colspan=1>4.54</td><td rowspan=1 colspan=1>4.36</td></tr><tr><td rowspan=2 colspan=1>Third</td><td rowspan=1 colspan=1>Lower</td><td rowspan=1 colspan=1>5.87</td><td rowspan=1 colspan=1>173.76</td><td rowspan=1 colspan=1>5.43</td><td rowspan=1 colspan=1>5.50</td></tr><tr><td rowspan=1 colspan=1>Upper</td><td rowspan=1 colspan=1>6.86</td><td rowspan=1 colspan=1>163.68</td><td rowspan=1 colspan=1>4.53</td><td rowspan=1 colspan=1>4.18</td></tr></table>

(a)

(b)  
![](images/655fc71021b1728d3f10b7d9aef994ff34634f794dab6a1a16ec0d8efb0f06d7.jpg)

![](images/1ae3ba4c49b875ab4f88c5a0e4fd32b734eb16a26276e590d29e269f696fe939.jpg)

exhibit broadside radiation characteristics, although increased back radiation is observed in the higher-frequency resonance. This discrepancy can be attributed in part to diffraction from the finite ground-plane edges, which is known to affect the radiation characteristics of microstrip patch antennas [50]. The measured resonant frequency $f _ { r } ,$ bandwidth BW, and realized gain of each dual-band design are summarized in Table V.

![](images/ab994a0f1aa5655167839d7f9062bc2b4cfc3ecec02933ae4096624564e3ecbc.jpg)

![](images/525e3cc76bd3468f6d69ddd7f1f06a91491ee06f2d99018c18a963ffb2cb4b14.jpg)

![](images/a60d14bcf82a9d4f0c66fcaf22882f31cb39b522fd82c574eec1ff4bd574c6f0.jpg)

(f)  
![](images/a99c0dce978fa66f2fb9ae7cdd0b2657027ff43cdb4cc7a3138108a2ee35bc73.jpg)  
Fig. 26. Measured results of the first dual-band antenna: (a) fabricated antenna, (b) $S _ { 1 1 }$ response, Elevation radiation patterns at $f _ { 1 } = 5 . 9 1$ GHz (c) $\phi = 0 ^ { \circ }$ , (d) $\phi = 9 0 ^ { \circ }$ , and at $f _ { 2 } = 6 . 8 5$ GHz (e) $\phi = 0 ^ { \circ }$ , (f) $\phi = 9 0 ^ { \circ }$

![](images/1bd1c7ab6cf691804b090d5f35e4a02a13ba717c8a6db9085ea4b12f48daa645.jpg)

(b)  
![](images/67484799775e10a3215ba11c5f4d5b873bada4d81abed307d6cc721614a7e122.jpg)

![](images/c4a9685952174ff9cb87938b9385769dde61cc060686044e0257380cc653c67f.jpg)  
Fig. 27. Measured results of the second dual-band antenna: (a) fabricated antenna, (b) $S _ { 1 1 }$ response, Elevation radiation patterns at $f _ { 1 } = 5 .$ .94 GHz (c) $\phi = 0 ^ { \circ }$ , (d) $\phi = 9 0 ^ { \circ }$ , and at $f _ { 2 } = 6 . 8 8$ GHz (e) $\phi = 0 ^ { \circ } ,$ , (f) $\phi = 9 0 ^ { \circ }$

> This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

(a)  
![](images/c7ce5c4703ce3284bc7032c9b1e216c45970908f3035653c7452a8753a29c849.jpg)

(b)  
![](images/af3f865c3c6e7597cf703e90ad440ffbeee3a71bbcc6d18a59d79cce0c810585.jpg)

![](images/a3a738c29cb92af45611403aedb176cdab7f9094b315e310c0cd89ea27a170ca.jpg)  
Fig. 28. Measured results of the third dual-band antenna: (a) fabricated antenna, (b) $S _ { 1 1 }$ response, Elevation radiation patterns at $f _ { 1 } = 5$ .87 GHz (c) $\phi = 0 ^ { \circ }$ , (d) $\phi = 9 0 ^ { \circ }$ , and at $f _ { 2 } = 6 . 8 6$ GHz (e) $\phi = 0 ^ { \circ } , ( \mathrm { f } ) \phi = 9 0 ^ { \circ }$

## V. DISCUSSION

## A. Quantitative Diversity of Generated Designs

In addition to the number of unique valid candidates and the redundant-candidate ratio, the structural diversity of the generated layouts is quantitatively evaluated using the normalized Hamming distance. The redundant-candidate ratio identifies exact repetitions, whereas the Hamming distance measures how two nonidentical pixel layouts differ from each other [51]. For two binary layouts $\mathbf { x } ^ { ( i ) }$ and $\mathbf { x } ^ { ( j ) }$ containing $P = 1 0 0$ pixels, the normalized Hamming distance is defined as

$$
d _ { \mathrm { H } } \left( \mathbf { x } ^ { ( i ) } , \mathbf { x } ^ { ( j ) } \right) = \frac { 1 } { P } \sum _ { p = 1 } ^ { P } \mathbf { 1 } \left[ x _ { p } ^ { ( i ) } \neq x _ { p } ^ { ( j ) } \right] , \qquad P = 1 0 0 .
$$

For each target case, exact duplicate layouts are first removed, and the normalized Hamming distance is calculated for every unordered pair of unique valid candidates. The mean distance is obtained for each case containing at least two unique valid candidates and is then averaged across such cases, preventing cases with a large number of generated candidates from dominating the dataset-level metric.

As shown in Table VI, both the pretrained and fine-tuned AR models achieve substantially higher normalized Hamming distances than conventional BPSO, VBPSO, and MVBPSO. Together with the low redundant-candidate ratios, this indicates that the proposed AR framework generates valid antenna layouts with greater structural diversity and less repetition across both original and step-shaped target sets.

TABLE VI  
STRUCTURAL DIVERSITY BASED ON NORMALIZED HAMMING DISTANCE
<table><tr><td rowspan=1 colspan=1>Target set</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Mean normalized Hamming distance</td></tr><tr><td rowspan=5 colspan=1>Original targets</td><td rowspan=1 colspan=1>Pretrained AR</td><td rowspan=1 colspan=1>0.4540</td></tr><tr><td rowspan=1 colspan=1>BPSO (best)</td><td rowspan=1 colspan=1>0.1351</td></tr><tr><td rowspan=1 colspan=1>BPSO (worst)</td><td rowspan=1 colspan=1>0.1174</td></tr><tr><td rowspan=1 colspan=1>VBPSO</td><td rowspan=1 colspan=1>0.2680</td></tr><tr><td rowspan=1 colspan=1>MVBPSO</td><td rowspan=1 colspan=1>0.1678</td></tr><tr><td rowspan=5 colspan=1>Step-shaped targets</td><td rowspan=1 colspan=1>Fine-tuned AR</td><td rowspan=1 colspan=1>0.4605</td></tr><tr><td rowspan=1 colspan=1>BPSO (best)</td><td rowspan=1 colspan=1>0.1606</td></tr><tr><td rowspan=1 colspan=1>BPSO (worst)</td><td rowspan=1 colspan=1>0.0895</td></tr><tr><td rowspan=1 colspan=1>VBPSO</td><td rowspan=1 colspan=1>0.2159</td></tr><tr><td rowspan=1 colspan=1>MVBPSO</td><td rowspan=1 colspan=1>0.1156</td></tr></table>

## B. Model Scaling and Computational Complexity

The present framework uses a 10 × 10 binary pixel grid, corresponding to an autoregressive sequence of 100 pixels. For an $M \times M$ grid, the sequence length becomes $N = M ^ { 2 }$ , and the self-attention operation scales quadratically with sequence length. Thus, increasing the resolution to $2 0 \times 2 0 ~ \mathrm { o r } ~ 3 0 \times 3 0$ increases the number of attention interactions by 16 and 81 times, respectively. Although a finer grid provides additional geometric degrees of freedom, the heatmap results over unseen dual-band targets show that the current 10 × 10 representation already achieves high target-space coverage. Even within this $1 0 \times 1 0$ design space, the BPSO variants have difficulty covering a wide range of target specifications, whereas the AR transformer maintains substantially broader coverage.

Increasing the grid resolution does not necessarily improve the performance of the AR transformer, because a larger grid expands the discrete design space and makes the inverse mapping more difficult. Therefore, simply increasing the grid size does not guarantee higher generation success or convergence to a valid solution. The $1 0 \times 1 0$ representation may instead provide a practical balance between sufficient geometric flexibility and manageable computational complexity for the dual-band design problem considered in this work.

## C. Hyperparameter Selection and Model Tuning

The architectural hyperparameters of both the forward surrogate and the AR inverse model were determined through systematic hyperparameter searches rather than being selected arbitrarily. Weights & Biases [52] sweeps with Bayesian optimization were used to explore candidate configurations while continuously monitoring the corresponding validation performance. For the forward surrogate, this process included the number and arrangement of residual blocks and channel dimensions. For the AR transformer, major parameters including the number of Transformer layers, hidden dimension, attention heads, feed-forward dimension, dropout ratio, and learning rate were examined. The final configuration was selected based on the best validation behavior observed during these sweeps.

## D. Effect of the Fixed Pixel Density

Another design choice in the present framework is the use of a fixed 60% metallic-pixel density. The density was selected based on the preliminary statistical analysis, which showed > This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

that this region provides the highest probability of obtaining resonant antenna responses. Concentrating the training data on this resonance-rich region reduces the number of non-resonant samples presented to the model and allows the available training budget to be used more efficiently for learning useful antenna geometries.

However, fixing the density also restricts the accessible design space [36]. Low-density or extremely dense layouts may exhibit electromagnetic behavior that is not represented by the present model, and the current results should therefore not be interpreted as demonstrating optimality over all possible pixel densities. Extending the training set to multiple density levels would provide a wider design space, although this would also increase the amount of training data and model capacity. Such an extension is particularly attractive for future multiobjective inverse design, where $S _ { 1 1 }$ can be optimized together with additional objectives such as gain, radiation efficiency, polarization, or bandwidth rather than using the reflection response alone.

## E. Comparison with Diffusion-Based Models

Diffusion models have recently been widely investigated for generative inverse-design problems because of their ability to represent complex multimodal design distributions [53], [54]. However, the present antenna geometry consists of only a 10 × 10 binary pixel grid, for which an autoregressive representation provides a more direct formulation. Each pixel can be generated sequentially as a discrete binary variable conditioned on the target spectrum and previously generated pixels, naturally enforcing the binary design constraint without an additional continuous-to-binary conversion or iterative denoising process.

In contrast, diffusion-based generation generally requires multiple denoising steps for each candidate, introducing additional sampling complexity that may be unnecessary for this relatively low-dimensional discrete design space. Therefore, the AR architecture was selected as a practical balance between multi-solution generation capability and computational complexity for the $1 0 \times 1 0$ pixelated antenna problem. Diffusion-based approaches remain an interesting direction for future work, particularly for higher-dimensional or more complex geometric representations.

## F. Extension to Other Antenna Configurations

The present model is developed for a fixed antenna configuration, including the substrate and feed geometry. Based on previous studies applying transfer learning to antenna and metasurface inverse design [16], [55], [56], the proposed framework could potentially be adapted to other substrate properties, feed structures, or antenna dimensions by finetuning the pretrained model with a relatively small amount of additional data from the new configuration. By reusing the features and design relationships learned from the original dataset, transfer learning can reduce the amount of newly generated training data and shorten the retraining process compared with training a new model from scratch. Nevertheless, the amount of additional data required is expected to depend on the degree of difference between the original and target configuration.

## G. Pixelated-Patch Antenna Design Considerations

Pixelated antennas provide a large discrete design space and enable compact and multiband geometries that are difficult to obtain using only a small number of conventional geometric parameters. At the same time, irregular and asymmetric metallic distributions introduce additional design considerations. In particular, asymmetry can degrade co- and crosspolarization isolation and therefore reduce polarization purity. Imposing strict geometric symmetry during generation is one possible way to improve polarization purity, but doing so substantially reduces the available pixel design space. Kumari et al. demonstrated that the high cross-polarization associated with an asymmetric inverse-designed pixelated antenna can be significantly suppressed by arranging mirror-symmetric elements with differential feeding, improving the broadside cross-polarization level from −5 to −50 dB [57].

Another practical issue in pixelated antenna design is the electrical connectivity between neighboring conducting pixels. Infinitesimal point contacts that are well defined in ideal EM simulations may be difficult to reproduce reliably because of fabrication tolerances. Previous studies have therefore introduced finite overlaps between adjacent pixels to ensure electrical connectivity [13], [58]. Jayasinghe et al. further investigated the effect of different overlap sizes, showing that the overlap configuration can also affect the resulting antenna performance [59]. Accordingly, in this work, a 0.25-mm overlap is introduced between neighboring pixels to ensure reliable electrical connectivity and improve manufacturability.

In addition to the issues arising from pixelated structures, microstrip patch antennas inherently exhibit a relatively narrow impedance bandwidth because of their resonant nature [60]. This limitation has traditionally been alleviated through structural and feeding modifications. For example, aperturecoupled stacked patches have been employed to broaden the operating bandwidth through coupled resonances [61], while defected ground structures (DGSs) have been used to modify the ground-current distribution and enhance impedance bandwidth [62]. More recently, geometry optimization has provided another approach to broadband patch design; Wang et al. demonstrated machine-learning-assisted bandwidth optimization of pixelated patch antennas, achieving measured relative bandwidths of 24.5% and 32.5% [14]. These studies demonstrate that the inherent bandwidth limitation of patch antennas can be substantially mitigated through appropriate structural and geometry optimization.

Table VII compares the proposed method with state-ofthe-art dual-band antenna design works in terms of design methodology, target specification capability, synthesis time, and antenna performance. Many prior inverse-design methods either return a single representative geometry for a given target response or do not explicitly support step-function-based target specifications. In contrast, our work directly generates multiple 10 × 10 pixelated patch antenna layouts from a desired dualband step-function target within approximately one minute. Therefore, this work provides a practical balance among automated inverse design, target-driven synthesis, multiplesolution generation, and experimental feasibility.

> This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

TABLE VII  
COMPARISON WITH STATE-OF-THE-ART DUAL-BAND ANTENNA WORKS IN TERMS OF DESIGN METHODOLOGY AND ANTENNA PERFORMANCE
<table><tr><td rowspan=1 colspan=1>Ref.</td><td rowspan=1 colspan=1>DesignMethod</td><td rowspan=1 colspan=1>Multiplesolutions</td><td rowspan=1 colspan=1>Step-functiontarget</td><td rowspan=1 colspan=1>Averagesynthesis time</td><td rowspan=1 colspan=1>Antennatype</td><td rowspan=1 colspan=1>Substrate /height</td><td rowspan=1 colspan=1>f1/f2(GHz)</td><td rowspan=1 colspan=1>Realizedgain (dBi)</td><td rowspan=1 colspan=1>Antennaefficiency</td></tr><tr><td rowspan=1 colspan=1>[63]</td><td rowspan=1 colspan=1>Conventionalheuristic design</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>Coupled-resonatorpatch antenna</td><td rowspan=1 colspan=1>Rogers 4003 /11 mm</td><td rowspan=1 colspan=1>3.24–3.83 /4.74–5.30</td><td rowspan=1 colspan=1>8.1 / 8.8</td><td rowspan=1 colspan=1>NR</td></tr><tr><td rowspan=1 colspan=1>[64]</td><td rowspan=1 colspan=1>Conventionalheuristic design</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>Differential-fedslotted-patch</td><td rowspan=1 colspan=1>Rogers RTduroid 5880 /7.35 mm</td><td rowspan=1 colspan=1>2.40–2.52 /5.09–5.41</td><td rowspan=1 colspan=1>8.5 / 9.2*</td><td rowspan=1 colspan=1>≥80 / ≥80%*</td></tr><tr><td rowspan=1 colspan=1>[65]</td><td rowspan=1 colspan=1>Conventionalheuristic design</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>Frequency selective-surface based sharedaperture patch</td><td rowspan=1 colspan=1>Rogers RO4003  /25.6 mm</td><td rowspan=1 colspan=1>2.3–2.7 /3.3-3.8</td><td rowspan=1 colspan=1>9.1 / 8.6</td><td rowspan=1 colspan=1>85 / 76%</td></tr><tr><td rowspan=1 colspan=1>[19]</td><td rowspan=1 colspan=1>Tandem NNinverse design</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>&lt;1 s</td><td rowspan=1 colspan=1>12 × 12 pixelatedmicrostrip antenna</td><td rowspan=1 colspan=1>Air /0.61 mm</td><td rowspan=1 colspan=1>14 / 18.37</td><td rowspan=1 colspan=1>9.73 / 9.43*</td><td rowspan=1 colspan=1>92.3 / 91.4%*</td></tr><tr><td rowspan=1 colspan=1>[4]</td><td rowspan=1 colspan=1>TGVAE-FGNNinverse design</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>Multiband slotteddipole antenna</td><td rowspan=1 colspan=1>Brass sheet /NR</td><td rowspan=1 colspan=1>Target-dependent0.3-3.0</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td></tr><tr><td rowspan=1 colspan=1>[16]</td><td rowspan=1 colspan=1>Transfer-learningsurrogate + BPSOinverse design</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>40 min</td><td rowspan=1 colspan=1>12 × 12 pixelatedmicrostrip antenna</td><td rowspan=1 colspan=1>FR4/3.2 mm</td><td rowspan=1 colspan=1>4.0 / 4.7</td><td rowspan=1 colspan=1>4.12 / 4.31</td><td rowspan=1 colspan=1>NR</td></tr><tr><td rowspan=1 colspan=1>[17]</td><td rowspan=1 colspan=1>CNN surrogate +BPSO inverse design</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>30 min</td><td rowspan=1 colspan=1>10 × 10 pixelatedmicrostrip antenna</td><td rowspan=1 colspan=1>Ideal lossless, $\epsilon _ { r } = 2 . 2 ~ /$ 2.5 mm</td><td rowspan=1 colspan=1>2.8 / 3.7</td><td rowspan=1 colspan=1>7.07 / 6.85*</td><td rowspan=1 colspan=1>95.3 / 98.8%*</td></tr><tr><td rowspan=1 colspan=1>[12]</td><td rowspan=1 colspan=1>VBPSO-basedEM optimization</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>38.47 h</td><td rowspan=1 colspan=1>Triangular- / square-pixel stacked patch</td><td rowspan=1 colspan=1>Rogers 4003 /3.04 mm</td><td rowspan=1 colspan=1>5.16–5.21 /5.79-5.86</td><td rowspan=1 colspan=1>3.82 / 3.46</td><td rowspan=1 colspan=1>70 / 82%*</td></tr><tr><td rowspan=1 colspan=1>[66]</td><td rowspan=1 colspan=1>SADRL inverse design(IBPSO+BCNN1+DQN)</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>15.43 h</td><td rowspan=1 colspan=1>12 × 10 pixelatedmonopole antenna</td><td rowspan=1 colspan=1>FR4/1.6 mm</td><td rowspan=1 colspan=1>3.3-3.8 /5.75–5.85</td><td rowspan=1 colspan=1>1.89±0.23 /1.35±0.11</td><td rowspan=1 colspan=1>NR</td></tr><tr><td rowspan=1 colspan=1>[58]</td><td rowspan=1 colspan=1>DC-SADEAwith XGBoostinverse design</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>9.3 days</td><td rowspan=1 colspan=1>Digitally codedfeeding-crosseddipole antenna</td><td rowspan=1 colspan=1>FR4/18.8 mm</td><td rowspan=1 colspan=1>3.3-3.8 /4.8-5.0</td><td rowspan=1 colspan=1>6.47 / 5.31*</td><td rowspan=1 colspan=1>NR</td></tr><tr><td rowspan=1 colspan=1>[67]</td><td rowspan=1 colspan=1>ONN-assistedEM optimization</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>4 h</td><td rowspan=1 colspan=1>Coaxial probe-fedslotted patch</td><td rowspan=1 colspan=1>Rogers RTduroid 5880 /3.175 mm</td><td rowspan=1 colspan=1>1.9 / 2.4</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td></tr><tr><td rowspan=1 colspan=1>[68]</td><td rowspan=1 colspan=1>FGNN-basedinverse design</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>Variable-slotmonopole patch</td><td rowspan=1 colspan=1>Brass sheet /NR</td><td rowspan=1 colspan=1>Target-dependent0.3-3.0</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>NR</td></tr><tr><td rowspan=1 colspan=1>[23]</td><td rowspan=1 colspan=1>Diffusion model +BCNN2-CMAinverse design</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>Circularly polarizedmicrostrip antenna</td><td rowspan=1 colspan=1>NR</td><td rowspan=1 colspan=1>3.63 / 6.91</td><td rowspan=1 colspan=1>6.61 / 3.12</td><td rowspan=1 colspan=1>NR</td></tr><tr><td rowspan=1 colspan=1>Thiswork</td><td rowspan=1 colspan=1>AR transformerinverse design</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>~1 min</td><td rowspan=1 colspan=1>10 × 10 pixelatedmicrostrip antenna</td><td rowspan=1 colspan=1>Rogers 4350B /1.52 mm</td><td rowspan=1 colspan=1>5.80–5.97 /6.78–6.95</td><td rowspan=1 colspan=1>5.43 / 4.53</td><td rowspan=1 colspan=1>88.0 / 87.7%*</td></tr></table>

<sup>∗</sup>Simulated value. NR: Not reported. NN: Neural network. TGVAE: Tandem generalized variational autoencoder. FGNN: Flex-form generative neural network. VBPSO: V-shaped transfer function based binary particle swarm optimization. SADRL: Surrogate model-assisted deep reinforcement learning. IBPSO: Improved binary particle swarm optimization. BCNN<sub>1</sub>: Bayesian convolutional neural network. DQN: Deep Q-network. DC-SADEA: Digitally coded antenna-oriented surrogate model-assisted evolutionary algorithm. XGBoost: Extreme gradient boosting. ONN: Optimization assisted by a neural network. BCNN<sub>2</sub>: Bilinear convolutional neural network. CMA: Characteristic mode analysis.

## VI. CONCLUSION

This paper demonstrates a fine-tuned autoregressive transformer for efficient inverse design of pixelated patch antennas. While conventional metaheuristic algorithms such as BPSO require iterative searches and are highly affected by initial conditions and hyperparameter tuning, the proposed AR transformer learns the conditional distribution between target $S _ { 1 1 }$ responses and antenna layouts, enabling direct generation of valid antenna candidates. The comparison with BPSO shows that the proposed method achieves more efficient valid-candidate generation and provides diverse antenna solutions. Furthermore, step-shaped target spectrum is introduced for fine-tuning, allowing practical dual-band inverse design through simple band-level specifications. The fabricated antennas show resonant responses and broadside radiation characteristics consistent with the intended designs. These results confirm that the proposed AR transformer overcomes key limitations of metaheuristic-based inverse design and offers an efficient framework for multi-solution antenna synthesis.

## ACKNOWLEDGEMENT

The authors would like to thank Prof. Jaeho Im of Hongik University for providing the measurement equipment used in this work.

## REFERENCES

[1] E. A. Karahan, Z. Liu, A. Gupta, Z. Shao, J. Zhou, U. Khankhoje, and K. Sengupta, “Deep-learning enabled generalized inverse design of multi-port radio-frequency and sub-terahertz passives and integrated circuits,” Nature Communications, vol. 15, no. 1, p. 10734, 2024.

[2] J. Zhou, E. A. Karahan, S. Ghozzy, and K. Sengupta, “AI-enabled end-to-end design in RFICs with controllable architectural style from ‘Classical’ to ‘Non-Intuitive’ for mm-Wave/sub-THz LNAs,” in 2026 IEEE International Solid-State Circuits Conference (ISSCC), vol. 69. IEEE, 2026, pp. 1–3.

[3] J. Zhou, E. A. Karahan, S. Ghozzy, Z. Liu, H. Jalili, and K. Sengupta, “25.3 AI-enabled design space discovery and end-to-end synthesis for RFICs with reinforcement learning and inverse methods demonstrating mm-Wave/sub-THz PAs between 30 and 120 GHz,” in 2025 IEEE International Solid-State Circuits Conference (ISSCC), vol. 68. IEEE, 2025, pp. 1–3.

[4] T. Yuan, Y. Li, and X. Yang, “Tandem generalized variational autoencoder network for multisolutions inverse design,” IEEE Antennas and Wireless Propagation Letters, vol. 25, no. 1, pp. 354–358, 2025.

[5] E. A. Karahan, Z. Liu, and K. Sengupta, “Deep-learning-based inversedesigned millimeter-wave passives and power amplifiers,” IEEE Journal of Solid-State Circuits, vol. 58, no. 11, pp. 3074–3088, 2023.

[6] E. A. Karahan, A. Gupta, U. K. Khankhoje, and K. Sengupta, “Deep learning based modeling and inverse design for arbitrary planar antenna structures at RF and millimeter-wave,” in 2022 IEEE International Symposium on Antennas and Propagation and USNC-URSI Radio Science Meeting (AP-S/URSI). IEEE, 2022, pp. 499–500.

[7] E. A. Karahan, Z. Shao, and K. Sengupta, “Deep learning aided modelling and inverse design for multi-port antennas,” in 2024 IEEE International Symposium on Antennas and Propagation and INC/USNC-URSI Radio Science Meeting (AP-S/INC-USNC-URSI). IEEE, 2024, pp. 799–800.

[8] F. Svelto, C. Chu, E. Liu, and H. Wang, “AI-assisted complex load mismatch prediction by mm-wave reconfigurable Doherty power amplifier and input/output power sensors,” in 2025 16th German Microwave Conference (GeMiC). IEEE, 2025, pp. 475–478.

[9] C. Chu, J. Xu, Y. Liu, J. Zeng, A. Wang, T. Torii, S. Shinjo, K. Yamanaka, and H. Wang, “AI-assisted template-seeded pixelated design for multi-metal-layer high-coupling EM structures: A Ku-band 6G FR3 PA in 22nm FDX+,” in 2025 IEEE/MTT-S International Microwave Symposium-IMS 2025. IEEE, 2025, pp. 922–925.

[10] H. Chae, H. Yu, S. Li, and D. Z. Pan, “PulseRF: Physics augmented ML modeling and synthesis for high-frequency RFIC design,” in Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design, 2024, pp. 1–9.

[11] H. Chae, S. H. Chai, T. Chi, S. Li, and D. Z. Pan, “ML-assisted RF IC design enablement: The new frontier of AI for EDA,” in Proceedings of the 30th Asia and South Pacific Design Automation Conference, 2025, pp. 683–689.

[12] M. A. Ullah, R. Keshavarz, M. Abolhasan, J. Lipman, and N. Shariati, “Multiservice compact pixelated stacked antenna with different pixel shapes for IoT applications,” IEEE Internet of Things Journal, vol. 10, no. 22, pp. 19 883–19 897, 2023.

[13] H. Chen, K. Fu, X. Cai, L. Yue, and L. Gao, “Surrogate-assisted level set topology optimization for pixelated antenna design,” IEEE Antennas and Wireless Propagation Letters, 2026.

[14] Q. Wang, Z. Pang, D. Gao, P. Liu, X. Pang, and X. Yin, “Machine learning-assisted quasi-bisection method for pixelated patch antenna bandwidth optimization,” IEEE Antennas and Wireless Propagation Letters, vol. 23, no. 12, pp. 4807–4811, 2024.

[15] V. Gjokaj, J. Doroshewitz, J. Nanzer, and P. Chahal, “A design study of 5G antennas optimized using genetic algorithms,” in 2017 IEEE 67th Electronic Components and Technology Conference (ECTC). IEEE, 2017, pp. 2086–2091.

[16] A. Gupta and U. K. Khankhoje, “Transfer learning based rapid design of frequency and dielectric agile antennas,” IEEE Journal on Multiscale and Multiphysics Computational Techniques, vol. 10, pp. 47–57, 2024.

[17] S. Chen, G.-H. Sun, and K. Wang, “Inverse design of microstrip antennas based on deep learning,” Electronics, vol. 14, no. 13, p. 2510, 2025.

[18] K. Fu, X. Cai, B. Yuan, Y. Yang, and X. Yao, “An efficient surrogate assisted particle swarm optimization for antenna synthesis,” IEEE Transactions on Antennas and Propagation, vol. 70, no. 7, pp. 4977–4984, 2022.

[19] A. Gupta, E. A. Karahan, C. Bhat, K. Sengupta, and U. K. Khankhoje, “Tandem neural network based design of multiband antennas,” IEEE Transactions on Antennas and Propagation, vol. 71, no. 8, pp. 6308– 6317, 2023.

[20] W.-Q. Deng, S. Xu, Z. Xu, and S.-Y. Zhu, “Automated design of millimeter-wave dielectric transmitarray antenna using tandem network,” IEEE Antennas and Wireless Propagation Letters, vol. 23, no. 11, pp. 3549–3553, 2024.

[21] M. Kim, E. Shin, H. Kim, T. Yoon, and J. Oh, “Diffusion model driven inverse design of microstrip patch antenna enabling extendable FOM modeling,” in 2025 Asia-Pacific Microwave Conference (APMC). IEEE, 2025, pp. 1–3.

[22] Y. Guo, E. A. Karahan, Z. Li, Z. Shao, Z. Zhang, M. Wang, and K. Sengupta, “DALL-EM: Generative AI with diffusion models for new design space discovery and target-to-electromagnetic structure synthesis,” in 2025 IEEE/MTT-S International Microwave Symposium-IMS 2025. IEEE, 2025, pp. 926–929.

[23] Q. Pang, J. Ouyang, F. Yang, S. Yang, and J. Hu, “Inverse design method of antenna based on generative artificial intelligence,” IEEE Antennas and Wireless Propagation Letters, 2025.

[24] P. Ghewari and V. Patil, “Advancements in microstrip patch antenna design using nature-inspired metaheuristic optimization algorithms: A systematic review,” Archives of Computational Methods in Engineering, vol. 32, no. 6, pp. 3687–3732, 2025.

[25] J. Kim, J.-Y. Kim, J. Kim, Y. Hyeong, B. Neseli, J.-B. You, J. Shim, J. Shin, H.-H. Park, and H. Kurt, “Inverse design of nanophotonic devices enabled by optimization algorithms and deep learning: recent achievements and future prospects,” Nanophotonics, vol. 14, no. 2, pp. 121–151, 2025.

[26] Z. N. Chen, P. Liu, Y. Kuang, Y. Lyu, B. Zhang, D. Yang, Y. He, and J.- M. Jin, “Deep learning-enabled generative antenna design: Review and prospects,” IEEE Transactions on Antennas and Propagation, 2026.

[27] S.-O. Kaba, A. K. Mondal, Y. Zhang, Y. Bengio, and S. Ravanbakhsh, “Equivariance with learned canonicalization functions,” in International Conference on Machine Learning. PMLR, 2023, pp. 15 546–15 566.

[28] G. Ma, Y. Wang, D. Lim, S. Jegelka, and Y. Wang, “A canonicalization perspective on invariant and equivariant learning,” Advances in Neural Information Processing Systems, vol. 37, pp. 60 936–60 979, 2024.

[29] D. Kutscher, D. Chan, Y. Bai, T. Darrell, and R. Gupta, “Reordering patches improves vision models,” Advances in Neural Information Processing Systems, vol. 38, pp. 115 268–115 294, 2026.

[30] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in Neural Information Processing Systems, vol. 30, 2017.

[31] J. A. Nelder and R. W. Wedderburn, “Generalized linear models,” Journal of the Royal Statistical Society Series A: Statistics in Society, vol. 135, no. 3, pp. 370–384, 1972.

[32] Y. Li, T. Kim, A. Zhang, Z. Wang, and Y. Liu, “On-demand inverse design for narrowband nanophotonic structures based on generative model and tandem network,” arXiv preprint arXiv:2507.14761, 2025.

[33] S. Lathuiliere, P. Mesejo, X. Alameda-Pineda, and R. Horaud, “A\` comprehensive analysis of deep regression,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 9, pp. 2065– 2081, 2019.

[34] L. Wakrim, S. Ibnyaich, and M. M. Hassani, “Multiband operation and performance enhancement of the PIFA antenna by using particle swarm optimization and overlapping method,” Applied Computational Intelligence and Soft Computing, vol. 2017, no. 1, p. 3481709, 2017.

[35] L. Wakrim, A. E. Yassini, A. Khabba, S. Ibnyaich, and M. M. Hassani, “Novel design of a triple band PIFA antenna by using a binary genetic algorithm,” Journal of Computational Electronics, vol. 20, no. 3, pp. 1373–1386, 2021.

[36] A. Khaireh-Walieh, D. Langevin, P. Bennet, O. Teytaud, A. Moreau, and P. R. Wiecha, “A newcomer’s guide to deep learning for inverse design in nano-photonics,” Nanophotonics, vol. 12, no. 24, pp. 4387–4414, 2023.

[37] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, “The curious case of neural text degeneration,” arXiv preprint arXiv:1904.09751, 2019.

[38] M. Freitag and Y. Al-Onaizan, “Beam search strategies for neural machine translation,” in Proceedings of the First Workshop on Neural Machine Translation, 2017, pp. 56–60.

[39] P. Auer, N. Cesa-Bianchi, and P. Fischer, “Finite-time analysis of the multiarmed bandit problem,” Machine Learning, vol. 47, no. 2, pp. 235– 256, 2002.

[40] D. Ippolito, R. Kriz, J. Sedoc, M. Kustikova, and C. Callison-Burch, “Comparison of diverse decoding methods from conditional language models,” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019, pp. 3752–3762

[41] C. Shi, H. Yang, D. Cai, Z. Zhang, Y. Wang, Y. Yang, and W. Lam, “A thorough examination of decoding methods in the era of LLMs,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 8601–8629.

[42] J.-Y. Audibert, R. Munos, and C. Szepesvari, “Exploration–exploitation´ tradeoff using variance estimates in multi-armed bandits,” Theoretical Computer Science, vol. 410, no. 19, pp. 1876–1902, 2009.

[43] A. Carpentier, A. Lazaric, M. Ghavamzadeh, R. Munos, and P. Auer, “Upper-confidence-bound algorithms for active learning in multi-armed bandits,” in International Conference on Algorithmic Learning Theory. Springer, 2011, pp. 189–203.

[44] Y.-L. Zheng, L.-H. Ma, L.-Y. Zhang, and J.-X. Qian, “On the convergence analysis and parameter selection in particle swarm optimization,” in Proceedings of the 2003 International Conference on Machine Learning and Cybernetics (IEEE Cat. No. 03EX693), vol. 3. IEEE, 2003, pp. 1802–1807.

[45] J. Liu, Y. Mei, and X. Li, “An analysis of the inertia weight parameter for binary particle swarm optimization,” IEEE Transactions on Evolutionary Computation, vol. 20, no. 5, pp. 666–681, 2015.

[46] M. S. Innocente and J. Sienz, “Coefficients’ settings in particle swarm optimization: insight and guidelines,” Mecanica Computacional ´ , vol. 29, no. 94, pp. 9253–9269, 2010.

[47] T. M. Shami, A. A. El-Saleh, M. Alswaitti, Q. Al-Tashi, M. A. Summakieh, and S. Mirjalili, “Particle swarm optimization: A comprehensive survey,” IEEE Access, vol. 10, pp. 10 031–10 061, 2022.

[48] A. Rezaee Jordehi and J. Jasni, “Parameter selection in particle swarm optimisation: a survey,” Journal of Experimental & Theoretical Artificial Intelligence, vol. 25, no. 4, pp. 527–542, 2013.

[49] A. Kumar, M. G. Krishna, and D. Sarkar, “Automated design and inverse modelling of metasurfaces using modified V-shaped binary particle swarm optimization and convolutional neural networks,” IEEE Transactions on Antennas and Propagation, 2026.

[50] J. Huang, “The finite ground plane effect on the microstrip antenna radiation patterns,” IEEE Transactions on Antennas and Propagation, vol. 31, no. 4, pp. 649–653, 1983.

[51] M. Wineberg and F. Oppacher, “The underlying similarity of diversity measures used in evolutionary computation,” in Genetic and Evolutionary Computation Conference (GECCO). Springer, 2003, pp. 1493– 1504.

[52] L. Biewald, “Experiment tracking with Weights and Biases,” 2020, software available from wandb.com.

[53] M. Kim, E. Shin, H. Kim, T. Yoon, S. Lee, and J. Oh, “MR2P: Diffusion model-driven pixelated-antenna inverse design framework generalizable across extensible electromagnetic figures of merit,” IEEE Journal on Multiscale and Multiphysics Computational Techniques, 2026.

[54] J. Zhang, Z. Zhang, and H. Li, “Synthesis of sparse antenna arrays using denoising diffusion probabilistic models,” IEEE Internet of Things Journal, 2025.

[55] Z. Fan, C. Qian, Y. Jia, M. Chen, J. Zhang, X. Cui, E.-P. Li, B. Zheng, T. Cai, and H. Chen, “Transfer-learning-assisted inverse metasurface design for 30% data savings,” Physical Review Applied, vol. 18, no. 2, p. 024022, 2022.

[56] T. B. Kanmaz, E. Ozturk, H. V. Demir, and C. Gunduz-Demir, “Deeplearning-enabled electromagnetic near-field prediction and inverse design of metasurfaces,” Optica, vol. 10, no. 10, pp. 1373–1382, 2023.

[57] P. Kumari, P. Karthik, A. Gupta, P. Shukla, and U. K. Khankhoje, “Polarization purity enhancement of asymmetric microstrip pixelated antennas,” in 2025 IEEE Microwaves, Antennas, and Propagation Conference (MAPCON). IEEE, 2025, pp. 1–4.

[58] T. Wu, B. Liu, Q. Hua, M. O. Akinsolu, D. P. Buch, J. J. Adams, M. Imran, P. Lazaridis, R. Pei, and P. Excell, “An efficient method for complex digitally coded antenna design based on evolutionary computation and machine learning techniques,” IEEE Transactions on Antennas and Propagation, 2025.

[59] J. J. W. Jayasinghe, J. Anguera, D. N. Uduwawala, and A. Andujar,´ “Nonuniform overlapping method in designing microstrip patch antennas using genetic algorithm optimization,” International Journal of Antennas and Propagation, vol. 2015, no. 1, p. 805820, 2015.

[60] K.-F. Lee and K.-F. Tong, “Microstrip patch antennas—basic characteristics and some recent advances,” Proceedings of the IEEE, vol. 100, no. 7, pp. 2169–2180, 2012.

[61] S. Targonski, R. Waterhouse, and D. Pozar, “Design of wide-band aperture-stacked patch microstrip antennas,” IEEE Transactions on Antennas and Propagation, vol. 46, no. 9, pp. 1245–1251, 1998.

[62] D. S. Marotkar and P. Zade, “Bandwidth enhancement of microstrip patch antenna using defected ground structure,” in 2016 International Conference on Electrical, Electronics, and Optimization Techniques (ICEEOT). IEEE, 2016, pp. 1712–1716.

[63] X. Liu, B. Sanz-Izquierdo, H. Zhang, S. Gao, W. Hu, X.-X. Yang, and J. T. S. Sumantyo, “Differentially fed dual-band base station antenna with multimode resonance and high selectivity for 5G applications,” IEEE Transactions on Antennas and Propagation, vol. 72, no. 1, pp. 256–266, 2023.

[64] T. H. Bui, H. T. Vu, S. X. Ta, K. K. Nguyen, C. Dao-Ngoc, and N. Nguyen-Trong, “Dual-band dual-polarized slotted-patch antenna for in-band full-duplex applications,” IEEE Antennas and Wireless Propagation Letters, vol. 22, no. 6, pp. 1286–1290, 2023.

[65] S. J. Yang and X. Y. Zhang, “Frequency selective surface-based dualband dual-polarized high-gain antenna,” IEEE Transactions on Antennas and Propagation, vol. 70, no. 3, pp. 1663–1671, 2021.

[66] J. Dou, D. Li, S. Wei, T. Shen, and J. Song, “A multi-objective antenna optimization method based on the surrogate model-assisted deep reinforcement learning,” IEEE Antennas and Wireless Propagation Letters, 2026.

[67] A. Papathanasopoulos, P. A. Apostolopoulos, and Y. Rahmat-Samii, “Optimization assisted by neural network-based machine learning in electromagnetic applications,” IEEE Transactions on Antennas and Propagation, vol. 72, no. 1, pp. 160–173, 2023.

[68] T. Yuan, J. Xie, Y. Li, X. Yang, X. Yang, and E. Wen, “Flex-form generative neural network for multi-frequency modes antennas,” IEEE Transactions on Antennas and Propagation, 2025.

![](images/c5bbe53b33060f2df8fb67134e4101d4e02edf27d8932db43d500346d790efb2.jpg)

Young-Mok Yun received the B.S. degree in electrical and electronic engineering from Hongik University, Seoul, South Korea, in 2025. He is currently working toward the M.S. degree in system semiconductor engineering at Yonsei University. His current research interests include AI-assisted inverse design and optimization of RF/millimeter-wave components and high-speed interface systems.

![](images/9c6d2e22e049604eab4f00afdfa505c8cd07776fd81a9cab55ce008dace3a56b.jpg)

Seung-Won Shin received the B.S. degree in electrical and electronic engineering from Yonsei University, Seoul, South Korea, in 2026. He is currently working toward the M.S. degree in system semiconductor engineering at Yonsei University. His current research interests include AI-assisted integrated circuit design and machine-learning-based electronic design automation.

![](images/d81efd96c2d77d11351123da983ce70af9aea9cc90f42a9a94611bc46b8db43f.jpg)

Jaeyong Chung (Member, IEEE) received the B.S. degree in electrical engineering from Yonsei University, Seoul, South Korea, in 2006, and the M.S. and Ph.D. degrees in electrical and computer engineering from the Department of Electrical and Computer Engineering, University of Texas at Austin, Austin, TX, USA, in 2008 and 2011, respectively.

He worked with the Strategic CAD Lab (SCL), Intel, and the IBM Thomas J. Watson Research Center in Summer 2008 and Summer 2010, respectively. From 2011 to 2013, he was with the Design

Compiler Team, Synopsys, Inc., Mountain View, CA, USA. He is currently an Associate Professor with Yonsei University, Seoul, South Korea. His current research interests include efficient deep learning and AI-based hardware design.

![](images/6cd8dfe55bfdc368f63d8ea3729e8338b35a224873a1dc49c0342e12f0155d66.jpg)

Kyung-Sik Choi (Member, IEEE) received the B.S., M.S., and Ph.D. degrees in electrical engineering from Korea Advanced Institute of Science and Technology (KAIST), Daejeon, South Korea, in 2016, 2018, and 2022, respectively.

From 2022 to 2024, he was a Post-Doctoral Researcher with ETH Zurich, Zurich, Switzerland.¨ From 2024 to 2025, he was a tenure-track Assistant Professor with the Department of Electronic Engineering, Seoul National University of Science and Technology (SeoulTech), Seoul, South Korea. Since

2025, he has been a tenure-track Assistant Professor with the Department of System Semiconductor Engineering, Yonsei University, Seoul, South Korea. His research interests include radio frequency (RF) integrated circuits and systems, as well as high-frequency terahertz integrated circuits and systems based on complementary metal–oxide–semiconductor (CMOS) technology.