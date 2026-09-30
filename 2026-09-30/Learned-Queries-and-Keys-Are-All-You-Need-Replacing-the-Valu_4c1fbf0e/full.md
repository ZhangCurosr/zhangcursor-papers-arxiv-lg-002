# Learned Queries and Keys Are All You Need: Replacing the Value Projection with Structured Transforms

Ene Meco, Emadeldeen Hamdan, A. Enis Cetin

University of Illinois Chicago, Chicago, IL 60607

September 30, 2026

## Abstract

To reduce the number of parameters and cache memory requirements of transformers we introduce dual-headed transformers instead of three heads. We studied Walsh-Hadamard Transform (WHT), Discrete Cosine Transform (DCT), Discrete Fourier Transform, filterbank based Shearlet Transform, and Multiplication-Avoiding (MA) operators to construct dual heads. We combine spatial patches and their orthogonal transforms (or Shearlet and MA operators) in a structure similar to the attention block. We obtained better results than triple headed transformers in ImageNet. Extensive simulation examples are presented.

## 1 Introduction

The Vision Transformer (ViT) introduced a major shift in computer vision [4], transitioning the field from CNNs to Transformer architectures adapted from Natural Language Processing [21, 8]. Transformers now are the main machine learning (ML) method in core visual tasks such as classification, detection, segmentation, and generation [12]. They also power modern large vision-language models. By converting images into sequences compatible with textual tokens, this flexible architecture seamlessly handles tasks like visual question answering, serving as a universal backbone for multimodal AI.

The attention mechanism is the core of transformer networks, Computing the multiheaded attention requires a computational complexity quadratic in sequence length [1], [Tay et al., 2022] [14]. This bottleneck is particularly severe in computer vision, where token counts scale quadratically with spatial image resolution. The computational burden intensifies further in multimodal settings, as long sequences of visual tokens are concatenated with textual inputs. Therefore, reducing the cost of attention has become a critical necessity and an important area of research [Tay et al., 2022].

To reduce the number of parameters and cache memory requirements of transformers we developed dual-headed transformers instead of three heads. We studied Walsh-Hadamard Transform (WHT), Discrete Cosine Transform (DCT), Discrete Fourier Transform, filterbank based Shearlet Transform, and Multiplication-Avoiding (MA) operators to construct dual heads. We combine spatial patches and their orthogonal transforms (or Shearlet and MA) in a structure similar to the attention block. We obtained better results than triple headed transformers in ImageNet. Extensive simulation examples are presented.

To mitigate the above mentioned computational bottlenecks, we propose a streamlined dual-headed transformer framework that replaces conventional multi-head designs. Our approach systematically eliminates unnecessary projection overhead while preserving expressive representations by pairing spatial input patches with transformed domains.

To construct these efective dual heads, we evaluate a wide spectrum of orthogonal transforms and computationally eficient operators:

• Walsh-Hadamard Transform (WHT) [18, 15]: Enables fast, addition-only frequency domain projections.

• Discrete Cosine Transform (DCT) and Discrete Fourier Transform (DFT): Capture high- and low-frequency spatial patterns with high energy compaction. We extensively studied the use of DCT as a part of the attention mechanism in [14].

• Filterbank-based Shearlet Transform [2], [11]: Captures multi-scale and directional visual features [1] with high geometric sensitivity.

• Multiplication-Avoiding (MA) Operators [6, 20]: Minimizes hardware complexity by replacing costly floating-point multiplications with low-overhead operations.

Structurally, we integrate spatial image patches alongside their corresponding transform representations directly into a unified block inspired by the standard attention mechanism. By interacting cross-domain features without requiring a redundant third weight projection, the architecture achieves a richer contextual encoding with a significantly smaller memory footprint. In fact, cross-correlations and region covariances have been used in image recognition, description and tracking [16, 17, 20, 5]

Experimental evaluations on the CIFAR-10 benchmark demonstrate that our dualheaded design consistently outperforms conventional multi-headed baselines. Furthermore, the reduction in Key-Value (KV) cache memory requirements accelerates throughput during both training and inference. We provide extensive simulation results, ablation studies, and comparative analysis to validate the eficiency and scala-

bility of our proposed method.

## 2 Two Learned Channels

In this section, we define the dual-headed attention units considered in this work. The term dual-headed refers to the use of two learned projections, Q and K, instead of the three learned projections Q, K, and V employed by conventional scaled dot-product attention. This terminology is distinct from the number of attention heads used in multi-head attention.

Given an input token matrix

$$
X \in \mathbb { R } ^ { N \times d } ,\tag{1}
$$

the two learned representations are obtained as

$$
Q = X W _ { Q } , \qquad K = X W _ { K } ,\tag{2}
$$

where

$$
W _ { Q } , W _ { K } \in \mathbb { R } ^ { d \times d } .\tag{3}
$$

For comparison, conventional scaled dot-product attention computes

$$
Q = X W _ { Q } , \qquad K = X W _ { K } , \qquad V = X W _ { V } ,\tag{4}
$$

followed by

$$
Y _ { \mathrm { Q K V } } = \mathrm { s o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { d _ { k } } } \right) V .\tag{5}
$$

Our proposed units eliminate the independently learned value projection V . Instead, the output representation is constructed from K itself or from a fixed transform of K.

For convenience, throughout this section we define

$$
A ( Q , K ) = \mathrm { s o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { d _ { k } } } \right) .\tag{6}
$$

Thus, the general form of the proposed dual-headed unit is

$$
Y = A ( Q , K ) \mathcal { T } ( K ) ,\tag{7}
$$

where $\tau ( \cdot )$ may denote the identity mapping, an orthogonal transform such as DFT, DWT, Walsh-Hadamard Transform, or a filterbank-based representation.

This formulation preserves the token-to-token interaction produced by $Q K ^ { T }$ , while

eliminating the third learned projection used to construct V .

## 2.1 Key-as-Value Dual-Head Unit

The simplest dual-headed formulation directly uses K in place of the conventional value representation. The resulting unit is

$$
Y _ { \mathrm { Q K K } } = A ( Q , K ) K = \mathrm { s o f t m a x } \left( { \frac { Q K ^ { T } } { \sqrt { d _ { k } } } } \right) K .\tag{8}
$$

This formulation requires only the learned projections $W _ { Q }$ and $W _ { K }$ . It therefore serves as the basic dual-headed baseline for the transform-based units introduced below.

## 2.2 Orthogonal-Transform Dual-Head Units

Rather than applying the attention map directly to $K$ , fixed orthogonal transforms can be used to construct the output representation. In this case,

$$
Y _ { T } = A ( Q , K ) \mathcal { T } ( K ) .\tag{9}
$$

Because the transform is fixed, this operation does not introduce an additional learned projection analogous to $W _ { V }$

## 2.2.1 Discrete Cosine Transform

Let $\mathcal { D } ( \cdot )$ denote the orthonormal Discrete Cosine Transform (DCT) [19, 7]. The DCTbased dual-headed unit is defined as

$$
Y _ { \mathrm { D C T } } = A ( Q , K ) \mathcal { D } ( K ) = \mathrm { s o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { d _ { k } } } \right) \mathcal { D } ( K ) .\tag{10}
$$

The DCT replaces the independently learned value representation with a fixed transform-domain representation of the key features.

## 2.2.2 Walsh–Hadamard Transform

Similarly, let $\mathcal { H } ( \cdot )$ denote the orthonormal Walsh–Hadamard Transform (WHT). The corresponding dual-headed unit is

$$
Y _ { \mathrm { W H T } } = A ( Q , K ) \mathcal { H } ( K ) = \mathrm { s o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { d _ { k } } } \right) \mathcal { H } ( K ) .\tag{11}
$$

The WHT is particularly attractive because it can be implemented using only additions and subtractions when an appropriate fast transform is used.

## 2.2.3 Discrete Fourier Transform

The same dual-headed formulation can be extended to the Discrete Fourier Transform (DFT). Let $\mathcal F ( \cdot )$ denote the Fourier transform representation used by the model. The corresponding unit is

$$
\begin{array} { r } { Y _ { \mathrm { D F T } } = A ( Q , K ) \mathcal { F } ( K ) . } \end{array}\tag{12}
$$

DFT is implemented using FFT and it is complex. Therefore we compute the absolute value of the DFT. This formulation provides a frequency-domain representation without introducing a third learned projection.

## 2.3 Normalized MF–Tanh Attention (QK-MFQK)

In a conventional multi-head self-attention layer, the query, key, and value tensors are obtained through three learned projections. For one attention head, this computation is

$$
\mathbf { Q } = \mathbf { X } \mathbf { W } _ { Q } , \mathbf { K } = \mathbf { X } \mathbf { W } _ { K } , \mathbf { V } = \mathbf { X } \mathbf { W } _ { V } ,\tag{13}
$$

followed by

$$
\mathbf { Y } _ { \mathrm { V i T } } = \operatorname { s o f t m a x } \left( { \frac { \mathbf { Q } \mathbf { K } ^ { \mathsf { T } } } { \sqrt { d } } } \right) \mathbf { V } ,\tag{14}
$$

where d is the head dimension. The learned value projection incurs $D ^ { 2 } + D$ parameters and $N D ^ { 2 }$ multiply–accumulate operations per transformer block, where D and N denote the embedding dimension and the number of tokens, respectively.

We remove the learned value projection and construct its replacement directly from the query and key tensors using an element-wise multiplication-free (MF) operator [6, 13]. Given $\mathbf { Q } , \mathbf { K } \in \mathbb { R } ^ { B \times H \times N \times d }$ , we define

$$
\widetilde { \mathbf { V } } = \mathrm { M F } _ { \mathrm { e l e m } } ( \mathbf { Q } , \mathbf { K } ) = \mathrm { s i g n } ( \mathbf { Q } ) \odot | \mathbf { K } | + \mathrm { s i g n } ( \mathbf { K } ) \odot | \mathbf { Q } | .\tag{15}
$$

Because the operation is element-wise, $\widetilde { \mathbf { V } }$ has exactly the shape required by the attention value tensor. It uses sign extraction, absolute values, and additions, without general multiplications or learned parameters.

The proposed layer replaces softmax with a signed hyperbolic-tangent attention map. We first compute the conventional scaled similarity matrix,

$$
\mathbf { S } = { \frac { \mathbf { Q } \mathbf { K } ^ { \mathsf { T } } } { \sqrt { d } } } ,\tag{16}
$$

and normalize each row of the resulting signed weights by its $\ell _ { 1 }$ norm:

$$
\widehat { A } _ { i j } = \frac { \operatorname { t a n h } ( S _ { i j } ) } { \epsilon + \sum _ { k } \left| \operatorname { t a n h } ( S _ { i k } ) \right| } .\tag{17}
$$

The output of the proposed attention layer is therefore

$$
\mathbf { Y } _ { \mathrm { M F - T a n h } } = \hat { \mathbf { A } } \tilde { \mathbf { V } } .\tag{18}
$$

The signed normalization controls the magnitude of each attention row while retaining negative interactions, which are not available in a softmax probability distribution. The query–key projections, scaled query–key product, output projection, and feedforward network remain unchanged. Thus, relative to conventional ViT attention, the proposed layer removes only the value projection and replaces softmax by normalized tanh attention.

## 2.4 Shearlet-Inspired Fourier Filterbank

To exploit the two-dimensional geometry of image patches, we construct a fixed multiscale directional filterbank inspired by cone-adapted Shearlets [10]. For each attention head, the non-class tokens are reshaped to their original spatial patch grid and transformed with a two-dimensional FFT,

$$
\widehat { Z } ( \omega _ { x } , \omega _ { y } ) = \mathcal { F } _ { 2 D } \{ Z \} .\tag{19}
$$

The frequency plane is divided into horizontal and vertical cones,

$$
C _ { H } = \mathbf { 1 } \big ( | \omega _ { x } | \geq | \omega _ { y } | \big ) , \qquad C _ { V } = \mathbf { 1 } \big ( | \omega _ { y } | \geq | \omega _ { x } | \big ) .\tag{20}
$$

For each scale s, a radial Gaussian bandpass window is defined as

$$
R _ { s } ( \omega _ { x } , \omega _ { y } ) = \exp \left[ - \frac { ( r - s ) ^ { 2 } } { 2 \sigma _ { r } ^ { 2 } } \right] , \qquad r = \sqrt { \omega _ { x } ^ { 2 } + \omega _ { y } ^ { 2 } } .\tag{21}
$$

Directional selectivity is introduced through the frequency slopes

$$
p _ { H } = \frac { \omega _ { y } } { \omega _ { x } + \epsilon } , \qquad p _ { V } = \frac { \omega _ { x } } { \omega _ { y } + \epsilon } ,\tag{22}
$$

with directional windows

$$
D _ { k } ^ { c } = \exp \left[ - \frac { ( p _ { c } - k ) ^ { 2 } } { 2 \sigma _ { s } ^ { 2 } } \right] , \qquad c \in \{ H , V \} .\tag{23}
$$

The resulting directional filters are

$$
H _ { s , k } ^ { H } = R _ { s } D _ { k } ^ { H } C _ { H } , \qquad H _ { s , k } ^ { V } = R _ { s } D _ { k } ^ { V } C _ { V } .\tag{24}
$$

In our implementation, three scales and five shear values are used, producing $3 \times$ $5 \times 2 = 3 0$ directional filters, together with one low-pass filter.

Each attention head learns normalized fusion coeficients $\alpha _ { h , m }$ over the fixed filterbank,

$$
\alpha _ { h , m } = \frac { \exp ( a _ { h , m } ) } { \sum _ { \ell } \exp ( a _ { h , \ell } ) } ,\tag{25}
$$

and forms an efective filter

$$
H _ { h } ^ { \mathrm { f u s e d } } = \sum _ { m } \alpha _ { h , m } H _ { m } .\tag{26}
$$

The filtered representation is then obtained as

$$
\begin{array} { r } { S ( Z ) = \mathcal { F } _ { 2 D } ^ { - 1 } \left[ \widehat { Z } H _ { h } ^ { \mathrm { f u s e d } } \right] . } \end{array}\tag{27}
$$

Thus, the FFT is used only as an eficient implementation of the multiscale directional filterbank; the Fourier transform itself is not used as a separate attention representation.

## 2.4.1 Shearlet Representation as the Output

The first formulation retains the original $Q$ and K representations for computation of the attention map, while replacing the output representation by the Shearlet transform of K:

$$
Y _ { \mathrm { Q K - S h e a r K } } = A ( Q , K ) S ( K ) .\tag{28}
$$

Equivalently,

$$
Y _ { \mathrm { Q K - S h e a r K } } = \mathrm { s o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { d _ { k } } } \right) S ( K ) .\tag{29}
$$

This configuration leaves the attention-score computation unchanged and introduces directional and multiscale information only into the representation aggregated by the attention map.

## 2.4.2 Shearlet Key in the Attention Map

The second formulation replaces K by its Shearlet representation when computing the attention weights and also uses the transformed key as the output representation:

$$
Y _ { \mathrm { Q - S h e a r K - S h e a r K } } = \mathrm { s o f t m a x } \left( \frac { Q S ( K ) ^ { T } } { \sqrt { d _ { k } } } \right) S ( K ) .\tag{30}
$$

This formulation allows the directional features extracted by the Shearlet filterbank to influence both the token similarity measure and the representation propagated to the next layer.

## 2.4.3 Shearlet Query and Key with Spatial Output

We next transform both learned representations before computing the attention map, while retaining the original K as the output representation:

$$
Y _ { \mathrm { S h e a r Q - S h e a r K - K } } = \mathrm { s o f t m a x } \left( \frac { \mathcal { S } ( Q ) \mathcal { S } ( K ) ^ { T } } { \sqrt { d _ { k } } } \right) K .\tag{31}
$$

Here the Shearlet domain is used only to determine token-to-token relationships, while the original learned key features are propagated through the attention unit.

## 2.4.4 Fully Shearlet-Domain Dual-Head Unit

Finally, both the similarity computation and the output representation can be formed in the Shearlet domain:

$$
Y _ { \mathrm { S h e a r Q - S h e a r K - S h e a r K } } = \mathrm { s o f t m a x } \left( \frac { S ( Q ) S ( K ) ^ { T } } { \sqrt { d _ { k } } } \right) S ( K ) .\tag{32}
$$

These four Shearlet configurations allow us to separately study the efect of directional and multiscale representations on (i) the computation of the attention weights and (ii) the representation aggregated by those weights.

## 2.5 Parameter Reduction

For an embedding dimension $d ,$ conventional QKV attention requires three learned projection matrices,

$$
W _ { Q } , \ W _ { K } , \ W _ { V } \in \mathbb { R } ^ { d \times d } ,\tag{33}
$$

corresponding to approximately $3 d ^ { 2 }$ projection parameters, excluding biases and the final output projection.

The proposed dual-headed units require only

$$
W _ { Q } , \ W _ { K } , \in \mathbb { R } ^ { d \times d } ,\tag{34}
$$

corresponding to approximately $2 d ^ { 2 }$ projection parameters. Therefore, the orthogonal projection stage eliminates $d ^ { 2 }$ learned parameters, or one third of the conventional $Q .$ $K .$ , and V projection parameters.

The DCT, WHT, and DFT representations are fixed transforms and therefore do not require a learned value projection. Likewise, the filterbank coeficients of the Shearlet representation can be fixed, with only a small number of optional fusion parameters required when multiple directional subbands are combined.

For the Mini-ViT architecture used in our experiments, the standard QKV model contains 546,186 trainable parameters. The QKK, DCT, and WHT dual-headed variants contain 480,138 parameters, while the Shearlet-inspired variants contain 480,634 parameters due to the additional learned filter-fusion coeficients. Thus, the dualheaded variants reduce the total model parameter count by approximately 12% relative to the standard QKV model.

## 2.6 Computational Complexity

Let N be the number of tokens, d the embedding dimension, and $d _ { k } \ = \ d / H$ the dimension of each attention head.

Standard QKV attention requires three input projections and one output projection, giving

$$
C _ { \mathrm { Q K V } } = O ( 4 N d ^ { 2 } + 2 N ^ { 2 } d ) ,\tag{35}
$$

where the $2 N ^ { 2 } d$ term corresponds to $Q K ^ { T }$ and the subsequent attention–value multiplication.

By removing the learned value projection, the proposed dual-headed QKK unit reduces the complexity to

$$
C _ { \mathrm { Q K K } } = O ( 3 N d ^ { 2 } + 2 N ^ { 2 } d ) .\tag{36}
$$

The transform-based variants add only the cost of the corresponding fixed transform:

$$
C _ { \mathrm { { D C T } } } = O \left( \frac { N d ^ { 2 } } { H } \right) , \qquad C _ { \mathrm { { W H T } } } = O ( N d \log d _ { k } ) .\tag{37}
$$

For the Shearlet-inspired Fourier filterbank,

$$
C _ { S } = O \left( d N _ { p } \log N _ { p } + H M N _ { p } \right) ,\tag{38}
$$

where $N _ { p } = N - 1$ is the number of image patches and M is the number of filterbank

elements.

Thus, the proposed units reduce the projection cost while retaining the $O ( N ^ { 2 } d )$ token-interaction complexity of standard attention.

Table 1: Computational complexity of the attention units.
<table><tr><td>Unit</td><td>Complexity</td></tr><tr><td>QKV</td><td rowspan="3"> $O ( 4 N d ^ { 2 } + 2 N ^ { 2 } d )$   $O ( 3 N d ^ { 2 } + 2 N ^ { 2 } d )$   $O ( 3 N d ^ { 2 } + 2 N ^ { 2 } d + N d ^ { 2 } / H )$ </td></tr><tr><td>QKK QK-DCTK</td></tr><tr><td>QK-WHTK  $O ( 3 N d ^ { 2 } + 2 N ^ { 2 } d + N d \log d _ { k } )$ </td></tr></table>

## 3 Experimental Results

## 3.1 Transform-Based Dual-Head Results

All transform-based experiments were conducted on the CIFAR-10 dataset using a compact Mini-ViT architecture. The model uses 32×32 input images with 4×4 patches, an embedding dimension of 128, four Transformer blocks, four attention heads, and an MLP ratio of 2. All models were trained for 30 epochs under the same training settings to ensure a fair comparison between the attention configurations.

Table 2 reports the classification accuracy obtained using the regular attention unit of the Transformer and the proposed dual-headed variants based on orthogonal transforms and the filterbank-based Shearlet representation. Results are reported as mean test accuracy and standard deviation across repeated runs.

Table 2 reports the classification accuracy obtained using the regular attention unit of the transformer and various orthogonal transforms, and the filterbank-based Shearlet transform in diferent locations of the proposed dual-headed attention unit. Results are reported as mean test accuracy and standard deviation across repeated runs.

Among the tested Shearlet configurations, $A ( Q , K ) { \mathcal { S } } ( K )$ achieved the highest mean test accuracy of 77.320%, with a standard deviation of 0.354. A nearly identical mean accuracy of 77.310% was obtained when the Shearlet-transformed key representation was used both in the attention-score computation and in the output. Transforming both Q and K also produced competitive results, although the corresponding configurations exhibited somewhat larger run-to-run variation.

These results suggest that the directional and multiscale information provided by the filterbank representation can be efectively incorporated into a two-projection attention unit. In particular, using the original Q and K to determine the attention weights while replacing the conventional value representation by ${ \mathcal { S } } ( K )$ provided the highest mean accuracy among the tested Shearlet configurations.

Table 2: Test accuracy and parameter count of the evaluated attention configurations.
<table><tr><td>Configuration</td><td>Parameters</td><td>Test Accuracy (%)</td></tr><tr><td rowspan="2">Regular Attention:  $A ( Q , K ) V$   $A ( Q , K ) K$ </td><td>546,186</td><td> $7 6 . 0 3 5 \pm 0 . 5 3 0$ </td></tr><tr><td>480,138</td><td> $7 6 . 1 5 5 \pm 0 . 0 3 5$ </td></tr><tr><td rowspan="2"> $A ( Q , K ) \operatorname { D C T } ( K )$   $A ( Q , K ) \operatorname { H T } ( K )$ </td><td>480,138</td><td> $7 6 . 1 4 0 \pm 0 . 8 0 6$ </td></tr><tr><td>480,138</td><td> $7 6 . 8 3 0 \pm 0 . 7 6 4$ </td></tr><tr><td> $A ( Q , K ) { \mathcal { S } } ( K )$ </td><td>480,634</td><td> $\mathbf { 7 7 . 3 2 0 \pm 0 . 3 5 4 }$ </td></tr><tr><td> $A ( Q , S ( K ) ) S ( K )$ </td><td>480,634</td><td> $7 7 . 3 1 0 \pm 0 . 9 7 6$ </td></tr><tr><td> $A ( \mathcal { S } ( Q ) , \mathcal { S } ( K ) ) K$ </td><td>480,634</td><td> $7 6 . 9 7 0 \pm 1 . 0 7 5$ </td></tr><tr><td> $A ( \mathcal { S } ( Q ) , \mathcal { S } ( K ) ) \mathcal { S } ( K )$ </td><td>480,634</td><td> $7 7 . 1 4 5 \pm 0 . 7 5 7$ </td></tr></table>

Table 3: Comparison of conventional ViT and normalized MF–tanh attention. Top-1 accuracy is the best accuracy on the fixed internal validation split from one training run (seed 0). “Mult.” counts general multiplications and excludes sign, absolute-value, addition, tanh, reduction, and division operations. C10 represent the CIFAR10 dataset and Tiny-IN represent Tiny ImageNet dataset.
<table><tr><td>Dataset</td><td>Attention</td><td>Top-1 (%)</td><td>Top-5 (%)</td><td>Params (M)</td><td>Mult. (M)</td></tr><tr><td>C10</td><td>Softmax ViT Normalized MF-tanh</td><td>81.92 77.80</td><td>98.82 98.70</td><td>2.694 2.471</td><td>182.85 168.47</td></tr><tr><td rowspan="2">Tiny-IN</td><td></td><td>40.39</td><td>65.35</td><td></td><td></td></tr><tr><td>Softmax ViT Normalized MF-tanh</td><td>35.40</td><td>60.32</td><td>2.758 2.536</td><td>184.66 170.28</td></tr></table>

Walsh-Hadamard transform (WHT) based dual-headed attention unit is the computationally most eficient one because the WHT is binary, i.e., $H T ( K )$ can be implemented without performing any multiplications. Furthermore, it achieves better results than the ordinary attention unit based transformer.

In Table 3, we use the same compact ViT [4] configuration in all comparisons: six transformer blocks, embedding dimension D = 192, three attention heads, and an MLP hidden dimension of 768. CIFAR-10 [9] uses 4 × 4 patches, whereas Tiny-ImageNet [3] uses $8 \times 8$ patches; both settings produce 65 tokens including the class token. All models are trained from scratch for 100 epochs with AdamW, a batch size of 128, an initial learning rate of $3 \times 1 0 ^ { - 4 }$ , weight decay 0.05, five warm-up epochs, and cosine learning-rate decay. We use fixed 45,000/5,000 and 90,000/10,000 training/selection splits for CIFAR-10 and Tiny-ImageNet, respectively. The oficial test sets are not used for model selection.

Across both datasets, normalized MF–tanh attention removes 222,336 parameters and 14.38 million general multiplications from the six-block model. Its accuracy is lower than that of softmax ViT in the present single-seed experiments, with gaps of 4.12 and 4.99 percentage points on CIFAR-10 and Tiny-ImageNet, respectively. These results establish a consistent eficiency–accuracy tradeof: the proposed construction eliminates every learned value projection while retaining most of the baseline accuracy. Multi-seed evaluation on the oficial evaluation sets is required before reporting final mean and standard-deviation results.

## 4 Conclusion

We introduced a dual-headed Transformer formulation that removes the independently learned value projection and constructs the output from the key representation or from fixed transforms of the key. This reduces the number of learned projection parameters while preserving the standard token-to-token attention mechanism.

Among the investigated DCT, WHT, QKK, and Shearlet-inspired configurations, the multiscale directional filterbank produced the highest mean accuracy. In particular, $A ( Q , K ) { \mathcal { S } } ( K )$ achieved 77.32% test accuracy, indicating that directional and multiscale information can provide a useful alternative to a learned value projection. The results also show that the transform can be introduced without modifying the conventional $Q K ^ { T }$ attention map. WHT based projection method is the computationally most eficient one.

These findings motivate further investigation of fixed and filterbank-based operators for reducing the parameter and memory requirements of Transformer attention. Future work will evaluate the proposed dual-headed units on larger image datasets and architectures, and will investigate more eficient filterbank designs and hardware-oriented implementations.

## References

[1] Rashid Ansari, A Enis Cetin, and Sang H Lee. Sub-band coding of images using nonrectangular filter banks. In Applications of Digital Image Processing XI, volume 974, pages 315–323. SPIE, 1988.

[2] Mariantonia Cotronei, Milvia Rossini, Tomas Sauer, and Elena Volont\`e. Filters for anisotropic wavelet decompositions. Journal of Computational and Applied Mathematics, 349:316–330, 2019.

[3] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[4] Alexey Dosovitskiy and et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[5] Y Hakan Habiboglu, Osman Gunay, and A Enis Cetin. Real-time wildfire detection using correlation descriptors. In 2011 19th European Signal Processing Conference, pages 894–898. IEEE, 2011.

[6] Emadeldeen Hamdan and Ahmet Enis Cetin. Htma-net: Towards multiplicationavoiding neural networks via hadamard transform and in-memory computing. arXiv preprint arXiv:2509.23103, 2025.

[7] Emadeldeen Hamdan, Yingyi Luo, Ryan Forelli, Mengzhan Liufu, Nan Zhou, Sameera Shridhar, Ellie Quattrocchi, Zachary Leveroni, Seda Ogrenci, Nhan Tran, et al. Real-time instantaneous phase estimation using a deep dual-branch complex neural network. IEEE Transactions on Biomedical Engineering, 2025.

[8] Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah. Transformers in vision: A survey. ACM computing surveys (CSUR), 54(10s):1–41, 2022.

[9] Alex Krizhevsky, Geofrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[10] Gitta Kutyniok, Wang-Q Lim, and Xiaosheng Zhuang. Digital shearlet transforms. In Shearlets: Multiscale analysis for multivariate data, pages 239–282. Springer, 2012.

[11] Gitta Kutyniok, Morteza Shahram, and Xiaosheng Zhuang. Shearlab: A rational design of a digital parabolic scaling algorithm. SIAM Journal on Imaging Sciences, 5(4):1291–1332, 2012.

[12] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer us-

ing shifted windows. In 2021 IEEE/CVF international conference on computer vision (ICCV), pages 9992–10002. Ieee, 2021.

[13] Shamma Nasrin, Diaa Badawi, Ahmet Enis Cetin, Wilfred Gomes, and Amit Ranjan Trivedi. Mf-net: Compute-in-memory sram for multibit precision inference using memory-immersed data conversion and multiplication-free operators. IEEE Transactions on Circuits and Systems I: Regular Papers, 68(5):1966–1978, 2021.

[14] Hongyi Pan, Emadeldeen Hamdan, Xin Zhu, Ahmet Enis Cetin, and Ulas Bagci. Discrete cosine transform based decorrelated attention for vision transformers. IJCAI-ECAI 2026 (the 35th International Joint Conference on Artificial Intelligence), arXiv preprint arXiv:2405.13901, 2024.

[15] Hongyi Pan, Xin Zhu, Salih Furkan Atici, and Ahmet Cetin. A hybrid quantumclassical approach based on the hadamard transform for the convolutional layer. In International Conference on Machine Learning, pages 26891–26903. PMLR, 2023.

[16] Fatih Porikli. Sensitivity characteristics of cross-correlation distance metric and model function. In Proceedings of 37th CISS, 2003.

[17] Fatih Porikli, Oncel Tuzel, and Peter Meer. Covariance tracking using model update based on lie algebra. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06), volume 1, pages 728–735. IEEE, 2006.

[18] Hakob Sarukhanian, Sos S Agaian, Jaakko T Astola, and Karen O Egiazarian. Binary matrices, decomposition, and multiply-add architectures. In Image Processing: Algorithms and Systems II, volume 5014, pages 111–122. SPIE, 2003.

[19] Athanassios Skodras, Charilaos Christopoulos, and Touradj Ebrahimi. The jpeg 2000 still image compression standard. IEEE Signal processing magazine, 18(5):36–58, 2001.

[20] Hakan Tuna, Ibrahim Onaran, and A Enis Cetin. Image description using a multiplier-less operator. IEEE Signal Processing Letters, 16(9):751–753, 2009.

[21] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.