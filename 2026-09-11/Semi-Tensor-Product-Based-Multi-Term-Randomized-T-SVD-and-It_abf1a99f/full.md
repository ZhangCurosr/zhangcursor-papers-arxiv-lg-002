# Semi-Tensor Product-Based Multi-Term Randomized T-SVD and Its Visual Applications

Xingchen Xiao<sup>a</sup>, Feng Zhang<sup>a,∗</sup>, Wenjin Qin<sup>a</sup>, Jianjun Wang<sup>a</sup>

<sup>a</sup>School ofMathematics and Statistics, Southwest University, Chongqing, 400715, China

## Abstract

Tensor singular value decomposition (T-SVD), which is built upon the tensor-tensor product (t-product), has emerged as a powerful tool for processing high-dimensional visual data such as color images and videos. However, the standard t-product imposes strict dimensional compatibility constraints. Although extensions based on the semitensor product (STP) relax this restriction, their single-term formulations still sufer from limited approximation accuracy. Moreover, these deterministic methods incur high computational costs when processing large-scale tensor data. To address these issues, this paper introduces a novel semi-tensor product for third-order tensors under the t-product framework induced by arbitrary invertible linear transforms. The resulting tensor semi-tensor product breaks the rigid dimension matching requirement of the standard t-product, while retaining the closed-form property of T-SVD. Based on this construction, we develop a multi-term semi-tensor product singular value decomposition (MSTP-SVD), which integrates multiple orthogonal decomposition terms to significantly improve low-rank approximation accuracy compared with single-term schemes. To reduce the computational cost of multi-term modeling, we incorporate randomized projection and power iteration techniques into the MSTP-SVD framework, yielding an accelerated multi-term randomized semi-tensor product SVD (MRSTP-SVD) algorithm that achieves a balance between reconstruction accuracy and computational eficiency. Experiments on image and video compression and completion tasks demonstrate the efectiveness of the proposed method.

Keywords: Tensor singular value decomposition, Semi-tensor product, Multi-term Kronecker product decomposition, Randomized algorithm, Tensor compression and completion

## 1. Introduction

The rapid advancement of data acquisition technologies has led to an explosion of high-dimensional visual data, including color images, color videos, hyperspectral remote sensing images, and medical imaging data. Tensors, as natural higher-order extensions of matrices, have become the standard representation for such multidimensional data and found widespread applications in computer vision [1, 2, 3, 4], machine learning [5, 6], signal processing [7, 8], and data mining [9, 10]. Compared with flattening multidimensional signals into matrices or vectors, tensor representations preserve inherent cross-dimensional structural correlations, which are crucial for extracting meaningful features and achieving superior performance in various analytical tasks [11, 12, 13, 14, 15]. Nevertheless, the development of eficient and accurate tensor decomposition algorithms for large-scale visual data remains a fundamental challenge.

Among numerous tensor decomposition techniques, CANDECOMP/PARAFAC (CP) [16], Tucker decomposition [17], Tensor Train (TT) [18], and Tensor Ring (TR) [19] have been extensively studied and successfully applied. However, these decompositions either sufer from NP-hard rank determination (CP), lack optimal truncation properties (Tucker), or involve unbalanced matricization schemes that may not capture global information efectively [20, 21]. In this context, the tensor singular value decomposition (T-SVD) pioneered by Kilmer et al. [22, 23] has emerged as a theoretically attractive alternative. Built upon the tensor-tensor product (t-product), T-SVD provides a closed-form factorization (see Theorem 2.1). Moreover, truncated T-SVD achieves optimal approximation in the Frobenius norm sense for any unitary-invariant tensor norm [24], analogous to the classical Eckart–Young–Mirsky theorem for matrices. These theoretical advantages, combined with natural parallelizability across frontal slices, have made T-SVD particularly successful in image compression [25], video completion [26], face recognition [27], and background modeling [28].

Despite its elegance, the standard t-product framework imposes strict dimensional compatibility requirements. Specifically, for two third-order tensors $\boldsymbol { X } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ and $\boldsymbol { y } \in \mathbb { R } ^ { m _ { 1 } \times m _ { 2 } \times n _ { 3 } }$ , their t-product $\chi _ { * } y$ is well-defined only when $n _ { 2 } = m _ { 1 }$ . This constraint severely limits the applicability of T-SVD to real-world tensors whose modes do not satisfy such rigid matching conditions. To address this limitation, researchers have explored alternative algebraic structures. Notably, the matrix semi-tensor product (STP) proposed by Cheng [29] supports dimensionally mismatched operations by introducing block Kronecker products, ofering greater flexibility than conventional matrix multiplication. Recent work by Chen et al. [30] has extended this idea to the tensor setting, defining a tensor semi-tensor product that relaxes dimension constraints while preserving certain algebraic properties.

However, a critical observation is that existing tensor STP-based decompositions adopt a single-term formulation, which represents the target tensor using only one set of orthogonal factors. This single-component structure inherently restricts the representation power, particularly when the underlying tensor exhibits complex multi-mode correlations or does not possess rapidly decaying singular values. In such scenarios, single-term approximations inevitably sufer from limited accuracy, failing to capture fine-grained structural information efectively [31]. This observation motivates the exploration of multi-term decomposition strategies, wherein multiple orthogonal components are combined to achieve more accurate and flexible tensor representations—a paradigm that has proven successful in classical matrix decompositions and other tensor frameworks, yet remains largely underexplored in the STP-based context.

Furthermore, deterministic tensor decomposition algorithms, whether based on exact SVD computations or iterative optimization procedures, incur substantial computational costs. As shown in [32], computing a k-term truncated T-SVD requires $O \left( n _ { 1 } n _ { 2 } n _ { 3 } l o g ( n _ { 3 } ) + m _ { 1 } m _ { 2 } k \right)$ operations, which becomes prohibitively expensive for high-resolution videos (e.g., $1 9 2 0 \times 1 0 8 0 \times 1 0 0 0 )$ or large-scale hyperspectral datasets. This challenge is further exacerbated in multi-term decomposition scenarios, where multiple factorizations must be performed, multiplying the computational burden accordingly. Although randomized algorithms have been successfully developed for

Tucker decomposition [33], CP approximation [34], and even T-SVD [35, 36] to reduce complexity, their application to STP-based frameworks remains largely unexplored.

Motivated by these observations, we aim to develop a comprehensive framework that simultaneously addresses the aforementioned three challenges: (i) dimensional rigidity of traditional t-product operations, (ii) approximation accuracy limitations of single-term decomposition schemes, and (iii) computational ineficiency of deterministic methods for large-scale data. These three challenges are not isolated; addressing one often exacerbates another. For instance, relaxing dimensional constraints through semi-tensor products may introduce additional algebraic complexity, while improving accuracy via multi-term modeling inevitably increases computational cost. Moreover, we observe that existing tensor STP constructions rely exclusively on specific transform bases (e.g. discrete Fourier transform (DFT)), which may not be optimal for all application scenarios. A framework based on arbitrary invertible linear transforms could provide greater flexibility and potentially superior approximation properties for diverse data characteristics. Therefore, a unified approach that carefully balances these competing objectives is highly desirable.

To this end, the main contributions of this paper are summarized as follows:

• We propose a novel semi-tensor product for third-order tensors under a generalized t-product framework induced by arbitrary invertible linear transforms. Unlike prior work that relies exclusively on DFT, our construction accommodates any unitary transform, providing a more flexible algebraic foundation while retaining the closed-form property of T-SVD and essential algebraic properties.

• We establish a multi-term semi-tensor product singular value decomposition (MSTP-SVD) model that integrates multiple orthogonal decomposition terms, significantly improving low-rank approximation accuracy compared with existing single-term schemes. We provide theoretical analysis of the approximation error bounds.

• We develop an accelerated multi-term randomized STP-SVD (MRSTP-SVD) algorithm by incorporating random projection and power iteration techniques, which achieves substantial speedup over deterministic counterparts with controllable approximation error. We derive expected error bounds and discuss the efects of key parameters.

• We conduct extensive experiments on image compression, video compression, and tensor completion tasks, demonstrating that the proposed method achieves superior or comparable performance against state-of-the-art baselines in both reconstruction quality and computational eficiency.

The remainder of this paper is organized as follows. Section 2 reviews preliminaries on tensor notations, t-product operations, and matrix semi-tensor product. In Section 3, we present our novel tensor semi-tensor product definition along with its theoretical properties and connections to existing frameworks. Section 4 develops the MSTP-SVD decomposition model. The randomized MRSTP-SVD algorithm is deferred to Section 5. Section 6 reports experimental results on various tasks, and Section 7 concludes the paper with discussions on future work.

## 2. Notations and Preliminaries

In this section, some notations and basic preliminaries adopted throughout this paper are summarized. Vectors are represented by bold lowercase letters, matrices by bold uppercase letters, and tensors by calligraphic letters. R and C denote the real and complex Euclidean spaces, respectively. For a third-order tensor $\mathcal { A } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ its (i<sub>,</sub> j<sub>,</sub> k)-th entry is written as $\mathcal { A } _ { i j k }$ . We adopt $\mathcal { A } ^ { ( i ) }$ to represent the i-th frontal slices. For two matrices A<sub>,</sub> $\mathbf { B } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } }$ , their inner product is defined as $\langle \mathbf { A } , \mathbf { B } \rangle = \operatorname { T r } \left( \mathbf { A } ^ { \top } \mathbf { B } \right)$ , where $\mathbf { A } ^ { \top }$ denotes the transpose of A and Tr(·) denotes the matrix trace operator. Additionally, $\mathbf { A } ^ { \mathrm { H } }$ denotes the conjugate transpose of matrix A. For a arbitrary tensor A, its Frobenius norm is defined as $\| \mathcal { A } \| _ { F } = \sqrt { \sum _ { i , j , k } | \mathcal { A } _ { i j k } | ^ { 2 } }$

To facilitate the understanding of what follows, we first briefly review the definitions and key properties of vectors and the STP of matrices, which serve as the essential mathematical foundation for this study. Note that this paper considers only the left semi-tensor product. The right semi-tensor product and its formulation for

general dimensions, as well as other related aspects, are discussed in detail in [37].   
Hereafter, the term “semi-tensor product” refers to the left semi-tensor product.

Definition 2.1 (Semi-tensor product of vectors [38]). Let $\mathbf { x } \in \mathbb { R } ^ { 1 \times n p }$ be a row vector and $\mathbf { y } \in \mathbb { R } ^ { p \times 1 }$ be a column vector. We split x into p blocks as $\mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \cdots , \mathbf { x } _ { p } ,$ , each block is a $1 \times n$ row vector. Then we can define the STP oftwo vectors, denoted by ⋉, as

$$
\mathbf { x } \ltimes \mathbf { y } = \sum _ { i = 1 } ^ { p } \mathbf { x } _ { i } y _ { i } \in \mathbb { R } ^ { 1 \times n } , \quad \mathbf { y } ^ { \top } \ltimes \mathbf { x } ^ { \top } = \sum _ { i = 1 } ^ { p } y _ { i } ( \mathbf { x } _ { i } ) ^ { \top } \in \mathbb { R } ^ { n \times 1 } .
$$

Definition 2.2 (Semi-tensor product of matrices [38]). Given two matrices $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ and $\mathbf { B } \in \mathbb { R } ^ { p \times q } . \ I f n = t p o r p = t n , t \in \mathbb { Z } ^ { + }$ , then we can define the semi-tensor product ofA and B, denoted by $\mathbf { C } = \mathbf { A }$ ⋉ B. Here C is a block matrix which has $m \times q$ blocks, each block can be represented as

$$
\mathbf { C } ^ { i j } = \mathbf { A } ^ { i } \times \mathbf { B } _ { j } ,
$$

where $\mathbf { A } ^ { i }$ is the i-th row of A and $\mathbf { B } _ { j }$ is the j-th column of B.

Lemma 2.1 (The associative law of semi-tensor product of matrix [29]). Let A B C be matrices of compatible dimensions, The following properties of the semi-tensor product of matrices hold:

$$
( \mathbf { A } \ltimes \mathbf { B } ) \ltimes \mathbf { C } = \mathbf { A } \ltimes ( \mathbf { B } \ltimes \mathbf { C } ) .
$$

Let ${ \mathbf I } _ { n }$ denote an $n \times n$ identity matrix. Then, the matrix STP defined in Definition 2.2 admits the following representation in terms of the Kronecker product (see Supplementary Material for its definition).

Lemma 2.2. [29] $L e t \mathbf { A } \in \mathbb { R } ^ { m \times n }$ , and $\mathbf { B } \in \mathbb { R } ^ { p \times q }$ . Then, the following properties hold:

$$
\mathbf { A } \ltimes \mathbf { B } = ( \mathbf { A } \otimes \mathbf { I } _ { t / n } ) ( \mathbf { B } \otimes \mathbf { I } _ { t / p } ) \in \mathbb { R } ^ { m ( t / n ) \times q ( t / p ) } ,
$$

where t is the least common multiple ofn and $p , i . e . , t = \operatorname { l c m } ( n , p )$

Remark 2.1. when $p = n , \mathbf { A } \ltimes \mathbf { B } = ( \mathbf { A } \otimes \mathbf { I } _ { 1 } ) ( \mathbf { B } \otimes \mathbf { I } _ { 1 } ) = \mathbf { A } \mathbf { B } .$ . That $i s ,$ the matrix $S T P$ can degenerate into the standard matrix multiplication.

Existing work in [23] proposes the generalized definition of the t-product under arbitrary invertible linear transforms. In our paper, we consider a linear transform $L : \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }  \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ <sup>3</sup>, which is defined as

$$
\begin{array} { r } { \bar { \mathcal { A } } = L ( \mathcal { A } ) = \mathcal { A } \times _ { 3 } \mathbf { L } , } \end{array}\tag{1}
$$

where $" \mathrm { \times } _ { 3 } "$ represents mode-3 product (Definition 2.5 in [23]) and $\mathbf { L } \in \mathbb { C } ^ { n _ { 3 } \times n _ { 3 } }$ can be any invertible transform matrix. Clearly, L is invertible with the inverse transform defined as $\mathcal { A } = L ^ { - 1 } ( \bar { \mathcal { A } } ) = \bar { \mathcal { A } } \times _ { 3 } \mathbf { L } ^ { - 1 }$

we construct a block-diagonal matrix as follows:

$$
\bar { \bf A } = { \mathrm { b d i a g } } ( \bar { \mathcal { A } } ) : = \left[ \begin{array} { c c c c } { \bar { \mathcal { A } } ^ { ( 1 ) } } & & & \\ & { \bar { \mathcal { A } } ^ { ( 2 ) } } & & \\ & & { \ddots } & \\ & & & { \bar { \mathcal { A } } ^ { ( n _ { 3 } ) } } \end{array} \right] .
$$

The block-diagonal matrix constructed via the block-diagonalization operation can be converted back into the original tensor by the fold operator: fold $\begin{array} { r } { \vert ( \mathbf { b } \mathbf { d i a g } ( \bar { \mathcal { A } } ) ) = \bar { \mathcal { A } } } \end{array}$

Based on the definition of fold and bdiag operators and invertible linear transform $L ,$ the definition of t-product can be given as follows.

Definition 2.3 (T-product [23]). Let L be any invertible linear transform in (1), and A be an $n _ { 1 } \times n _ { 2 } \times n _ { 3 }$ third-order tensor and B be an $n _ { 2 } \times n _ { 4 } \times n _ { 3 }$ third-order tensor, then the product ofA ∗ B is an $n _ { 1 } \times n _ { 4 } \times n _ { 3 }$ tensor C which can be represented as

$$
C = \mathcal { A } * _ { L } \mathcal { B } = L ^ { - 1 } [ \operatorname { f o l d } ( \operatorname { b d i a g } ( \bar { \mathcal { A } } ) \times \operatorname { b d i a g } ( \bar { \mathcal { B } } ) ) ] ,
$$

where “×” denotes the standard matrix product.

Definition 2.4 (Tensor transpose [23]). Let L be any invertible linear transform in (1). Let A be an $n _ { 1 } \times n _ { 2 } \times n _ { 3 }$ tensor, then the transpose ofA under L, denoted as $\mathcal { A } ^ { \top }$ satisfies $L ( \mathcal { A } ^ { \top } ) ^ { ( i ) } = L ( \mathcal { A } ^ { ( i ) } ) ^ { \top } , i = 1 , \dots , n _ { 3 }$

Definition 2.5 (Identity tensor [23]). Let L be any invertible linear transform in (1). Let I be an $n \times n \times n _ { 3 }$ tensor so that each frontal slices of $L ( \mathcal { T } ) = \bar { \mathcal { T } }$ is an $n \times n$ sized identity matrix. Then $\bar { \mathcal { I } } = L ^ { - 1 } ( \bar { \bar { \mathcal { I } } } )$ gives the identity tensor under L.

Definition 2.6 (F-diagonal tensor [22]). A third-order tensor tensor calledf-diagonal tensor ifeach ofitsfrontal slices is a diagonal matrix.

Definition 2.7 (Orthogonal tensor [23]). Let L be any invertible linear transform in (1). Q is an $n \times n \times n _ { 3 }$ orthogonal tensor $i f Q ^ { \top } * _ { L } Q = Q * _ { L } Q ^ { \top } = \bar { J }$

Theorem 2.1 (T-SVD [23]). Let L be any invertible linear transform in (1), and $\mathcal { A } \in$ $\mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , then it can be factorized as

$$
\mathcal { A } = \mathcal { U } * _ { L } S * _ { L } \mathcal { V } ^ { \top } ,
$$

where $\mathcal { U } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 1 } \times n _ { 3 } } , \ \mathcal { V } \in \mathbb { R } ^ { n _ { 2 } \times n _ { 2 } \times n _ { 3 } }$ are orthogonal, and $S \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ is an fdiagonal tensor.

## 3. Semi-tensor product of tensors via arbitrary invertible linear transforms

In this section, we define a novel tensor STP induced by arbitrary invertible linear transforms and investigate several theoretical properties of this newly proposed multiplication. The definitions and properties regarding Kronecker product are provided in the Supplementary Material. Building on these auxiliary prerequisites, we now define the tensor STP via arbitrary invertible linear transforms.

Definition 3.1 (Semi-tensor product of tensors). Let L be any invertible linear transform in (1). Suppose that $\mathcal { A } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ and $\mathcal { B } \in \mathbb { R } ^ { m _ { 1 } \times m _ { 2 } \times n _ { 3 } }$ . We have

$$
\begin{array} { r l } & { { \mathcal { R } } \ltimes _ { L } { \mathcal { B } } = ( { \mathcal { R } } \otimes { { T } _ { t / n _ { 2 } } } ) * _ { L } \left( { \mathcal { B } } \otimes { { T } _ { t / m _ { 1 } } } \right) } \\ & { \qquad = L ^ { - 1 } \bigg [ \mathrm { f o l d } \Big ( \mathrm { b d i a g } ( { \overline { { \mathcal { A } \otimes { { T } _ { t / n _ { 2 } } } } } } ) \times \mathrm { b d i a g } ( { \overline { { \mathcal { B } \otimes { { T } _ { t / m _ { 1 } } } } } } ) \Big ) \bigg ] \in { \mathbb { R } } ^ { n _ { 1 } ( t / n _ { 2 } ) \times m _ { 2 } ( t / m _ { 1 } ) \times n _ { 3 } } , } \end{array}
$$

where $t = \mathrm { l c m } ( n _ { 2 } , m _ { 1 } ) , \bar { J } _ { t / n _ { 2 } } \in \mathbb { R } ^ { t / n _ { 2 } \times t / n _ { 2 } \times 1 }$ and I<sub>t/m1</sub> ∈ R<sup>t/m1×t/m1×1</sup> are identity tensors.

Remark 3.1. We can see that $i f n _ { 2 } = m _ { 1 } , \mathcal { A } \ltimes _ { L } \mathcal { B } = \left( \mathcal { A } \otimes \bar { J } _ { 1 } \right) \ast _ { L } \left( \mathcal { B } \otimes \bar { J } _ { 1 } \right) = \mathcal { A } \ast _ { L } \mathcal { B } .$ That is, the tensor STP can reduce to the t-product.

The following equivalent representation follows directly from Definition 3.1.

Lemma 3.1. Let L be any invertible linear transform in (1). Suppose that $\mathcal { A } \in$ $\mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ and $\mathcal { B } \in \mathbb { R } ^ { m _ { 1 } \times m _ { 2 } \times n _ { 3 } }$ are third-order tensors. The following property holds:

$$
\mathcal { A } \ltimes _ { L } \mathcal { B } = L ^ { - 1 } \biggl [ \mathrm { f o l d } \biggl ( \mathrm { b d i a g } ( \bar { \mathcal { A } } ) \ltimes \mathrm { b d i a g } ( \bar { \mathcal { B } } ) \biggr ) \biggr ] .
$$

Proof. Based on Lemma 2.2 and Definition 3.1, and the definitions of the fold, bdiag operators as well as the Kronecker product, we have

$$
\begin{array} { r l } & { { \mathcal A } \ltimes _ { L } { \mathcal B } = L ^ { - 1 } \Bigg [ \mathrm { f o l d } \Big ( \mathrm { b d i a g } \big ( { \mathcal { \bar { A } } } \otimes { \bar { I } } _ { t / n _ { 2 } } \big ) \times \mathrm { b d i a g } \big ( { \mathcal { \bar { B } } } \otimes { \bar { I } } _ { t / m _ { 1 } } \big ) \Big ) \Bigg ] } \\ & { \qquad = L ^ { - 1 } \Bigg [ \mathrm { f o l d } \Big ( \mathrm { b d i a g } \big ( { \bar { \mathcal { A } } } \otimes { \bar { I } } _ { t / n _ { 2 } } \big ) \times \mathrm { b d i a g } \big ( { \bar { \mathcal { B } } } \otimes { \bar { I } } _ { t / m _ { 1 } } \big ) \Big ) \Bigg ] } \\ & { \qquad = L ^ { - 1 } \Bigg [ \mathrm { f o l d } \Big ( \big ( \mathrm { b d i a g } ( { \bar { \mathcal { A } } } ) \otimes \mathrm { I } _ { t / n _ { 2 } } \big ) \times \big ( \mathrm { b d i a g } ( { \bar { \mathcal { B } } } ) \otimes \mathrm { I } _ { t / m _ { 1 } } \big ) \Big ) \Bigg ] } \\ & { \qquad = L ^ { - 1 } \Bigg [ \mathrm { f o l d } \Big ( \mathrm { b d i a g } ( { \bar { \mathcal { A } } } ) \ltimes \mathrm { b d i a g } ( { \bar { \mathcal { B } } } ) \Big ) \Bigg ] . } \end{array}
$$

Remark 3.2. Lemma 3.1 establishes that the tensor STP is equivalent to the STP of block-diagonal matrices in the transform domain, i.e., $L ( \mathcal { A } \ltimes _ { L } \mathcal { B } ) = \mathrm { f o l d } ( \mathrm { b d i a g } ( \bar { \mathcal { A } } ) \ltimes$ bdiag(B<sup>¯</sup>). Accordingly, $\mathcal { A } \ltimes _ { L } \mathcal { B }$ can be computed via slice-wise matrix STP operations on A<sup>¯</sup> and B<sup>¯</sup>,followed by the inverse linear transform $L ^ { - 1 }$

Next, we establish a fundamental property pertaining to the STP of tensors.

Theorem 3.1 (The associative law of semi-tensor product of tensors). Let A B C be third-order tensors with compatible dimensions such that their semi-tensor products are well-defined. Then thefollowing equation holds.

$$
( \mathcal { A } \ltimes _ { L } \mathcal { B } ) \ltimes _ { L } C = \mathcal { A } \ltimes _ { L } ( \mathcal { B } \ltimes _ { L } C ) .
$$

Proof. By Lemma 3.1,

$$
\mathcal { A } \ltimes \ l _ { L } \mathcal { B } = L ^ { - 1 } \bigg [ \mathrm { f o l d } \big ( \mathrm { b d i a g } ( \bar { \mathcal { A } } ) \ltimes \mathrm { b d i a g } ( \bar { \mathcal { B } } ) \big ) \bigg ] , \mathcal { B } \ltimes \ l _ { L } C = L ^ { - 1 } \bigg [ \mathrm { f o l d } \big ( \mathrm { b d i a g } ( \bar { \mathcal { B } } ) \ltimes \mathrm { b d i a g } ( \bar { C } ) \big ) \bigg ] .
$$

Then,

$$
\begin{array} { r l } & { ( \mathcal { R } \ltimes \ l _ { L } \mathcal { B } ) \ltimes \ l _ { L } C = L ^ { - 1 } \Big [ \mathrm { f o l d } \Big ( \mathrm { b d i a g } \big ( L \big ( L ^ { - 1 } \big ( \mathrm { f o l d } \big ( \mathrm { b d i a g } ( \bar { \mathcal { A } } ) \ltimes \mathrm { b d i a g } ( \bar { \mathcal { B } } ) \big ) \big ) \big ) \big ) \ltimes \mathrm { b d i a g } ( \bar { C } ) \Big ) \Big ] } \\ & { \qquad = L ^ { - 1 } \bigg [ \mathrm { f o l d } \Big ( \big ( \mathrm { b d i a g } ( \bar { \mathcal { A } } ) \ltimes \mathrm { b d i a g } ( \bar { \mathcal { B } } ) \big ) \ltimes \mathrm { b d i a g } ( \bar { C } ) \Big ) \bigg ] , } \end{array}
$$

Similarly,

$$
\mathcal { A } \ltimes \ l _ { L } \left( \mathcal { B } \ltimes \ l _ { L } C \right) = L ^ { - 1 } \bigg [ \mathrm { f o l d } \Big ( \mathrm { b d i a g } ( \bar { \mathcal { A } } ) \ltimes \left( \mathrm { b d i a g } ( \bar { \mathcal { B } } ) \ltimes \mathrm { b d i a g } ( \bar { C } ) \right) \Big ) \bigg ] .
$$

The conclusion holds trivially by Lemma 2.1.

## 4. Tensor singular value decomposition via semi-tensor product

## 4.1. Single-term STP-SVD ofmatrices

Let

$$
\mathbf { A } = \left[ \begin{array} { c c c } { \mathbf { A } _ { 1 , 1 } } & { \cdots } & { \mathbf { A } _ { 1 , n _ { 1 } } } \\ { \mathbf { A } _ { 2 , 1 } } & { \cdots } & { \mathbf { A } _ { 2 , n _ { 1 } } } \\ { \vdots } & { \ddots } & { \vdots } \\ { \mathbf { A } _ { m _ { 1 } , 1 } } & { \cdots } & { \mathbf { A } _ { m _ { 1 } , n _ { 1 } } } \end{array} \right] \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } } ,
$$

where each block $\mathbf { A } _ { i , j }$ with $i = 1 , 2 , \cdots , m _ { 1 }$ and $j = 1 , 2 , \cdots , n _ { 1 }$ is an $m _ { 2 } \times n _ { 2 }$ matrix. Next, we define the rearrangement operator ${ \mathcal { R } } ,$ applied to the matrix A, as follows:

$$
\begin{array} { r } { \mathcal { R } ( { \bf A } ) = [ { \bf A } _ { 1 } { \bf A } _ { 2 } \cdot \cdot \cdot \cdot { \bf A } _ { n _ { 1 } } ] ^ { \top } \in \mathbb { R } ^ { m _ { 1 } n _ { 1 } \times m _ { 2 } n _ { 2 } } , } \end{array}\tag{2}
$$

where $\mathbf { A } _ { j } = \left[ \mathrm { v e c } ( \mathbf { A } _ { 1 , j } ) , \mathrm { v e c } ( \mathbf { A } _ { 2 , j } ) , \cdot \cdot \cdot , \mathrm { v e c } ( \mathbf { A } _ { m _ { 1 } , j } ) \right] ^ { \top }$

From the definition of $\mathcal { R }$ in (2), we obtain the following result on optimal Kronecker product approximation.

Lemma 4.1. [39] Given $\mathbf { A } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } }$ , there exist matrices $\mathbf { B } \in \mathbb { R } ^ { m _ { 1 } \times n _ { 1 } }$ and $\mathbf { C \in }$ R<sup>m2×n2</sup> such that vec $\mathbf { \rho } ( \mathbf { B } ) = \mathbf { \rho } \sqrt { \sigma _ { 1 } } \mathbf { U } ( : , 1 )$ , vec $\mathbf { \bar { \mathbf { \rho } } } ( \mathbf { C } ) = \sqrt { \sigma _ { 1 } } \mathbf { V } ( : , 1 )$ which minimize

$$
\lVert \mathbf { A } - \mathbf { B } \otimes \mathbf { C } \rVert _ { F } ,
$$

where $\sigma _ { 1 }$ denotes the largest singular value of $\mathcal { R } ( { \bf A } )$ given in (2), and $\mathbf { U } ( : , 1 )$ and $\mathbf { V } ( : , 1 )$ are the corresponding left and right singular vectors, respectively.

By virtue of Lemma 4.1, an SVD-like approximate matrix decomposition using the STP, which is referred to in the literature as STP-SVD.

Theorem 4.1. [30] Given $\mathbf { A } \ \in \ \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } }$ . Then there exist orthogonal matrices $\mathbf { U } \in \mathbb { R } ^ { m _ { 1 } \times m _ { 1 } }$ and $\mathbf { V } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 1 } }$ <sup>1</sup> such that

$$
\mathbf { A } = \mathbf { U } \ltimes \Sigma \ltimes \mathbf { V } ^ { \top } + \mathbf { E } ,\tag{3}
$$

where $\begin{array} { r } { \sum \ = \ \mathrm { b l o c k d i a g } ( \mathbf S _ { 1 } , \mathbf S _ { 2 } , \cdot \cdot \cdot \mathbf S _ { p } ) \ \in \ \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } } } \end{array}$ is a block-diagonal matrix with blocks ${ \bf S } _ { i } \in \mathbb { R } ^ { m _ { 2 } \times n _ { 2 } } f o r i = 1 , 2 , \cdot \cdot \cdot , p$ and $p = \min \{ m _ { 1 } , n _ { 1 } \}$ , satisfying $\Vert \mathbf { S } _ { 1 } \Vert _ { F } \geq \Vert \mathbf { S } _ { 2 } \Vert _ { F } \geq$

$\cdots \geq \| \mathbf { S } _ { p } \| _ { F }$ . The approximation error satisfies

$$
\| { \bf E } \| _ { F } = \sqrt { \sum _ { i = 2 } ^ { \nu } \sigma _ { i } ^ { 2 } } ,
$$

where $\sigma _ { 2 } \geq \cdot \cdot \cdot \geq \sigma _ { \nu } \geq 0$ are the singular values of $\mathcal { R } ( \mathbf { A } ) \in \mathbb { R } ^ { m _ { 1 } n _ { 1 } \times m _ { 2 } n _ { 2 } }$ defined in (2), and $\nu = \operatorname* { m i n } \{ m _ { 1 } n _ { 1 } , m _ { 2 } n _ { 2 } \}$

Algorithm 1 presents the complete procedure for Theorem 4.1.

Remark 4.1. By applying truncated SVD to the matrix B, a truncated STP-SVD algorithm applicable to general matrices can be derived. It follows the identical workflow except replacing full svd by rank-truncated svds with given rank parameter r. For brevity, its pseudocode is deferred to Supplementary Material. Correspondingly, we can obtain the upper bound ofthe error matrix E in this case [30],

$$
\| \mathbf { E } \| _ { F } \leq \sqrt { \sum _ { i = 2 } ^ { \nu } \sigma _ { i } ^ { 2 } } + \sqrt { \sum _ { j = r + 1 } ^ { p } \| \mathbf { S } _ { j } \| _ { F } ^ { 2 } } .
$$

Algorithm 1: STP-SVD of matrices [30]   
Input: A ∈ R<sup>m1m2×n1n2</sup> .   
Output: U Σ V.   
1 Calculate matrices $\mathbf { B } \in \mathbb { R } ^ { m _ { 1 } \times n _ { 1 } }$ and $\mathbf { C } \in \mathbb { R } ^ { m _ { 2 } \times n _ { 2 } }$ via Lemma 4.1, such that   
$\mathbf { A } \approx \mathbf { B } \otimes \mathbf { C } ;$   
2 Perform full SVD on B: $[ \mathbf { U _ { B } } , \boldsymbol { \Sigma _ { B } } , \mathbf { V _ { B } } ] = \mathrm { s v d } ( \mathbf { B } ) ;$   
3 return $\mathbf { U } = \mathbf { U _ { B } } , \boldsymbol { \Sigma } = \boldsymbol { \Sigma } _ { \mathbf { B } } \otimes \mathbf { C } , \mathbf { V } = \mathbf { V _ { B } } .$

## 4.2. Multi-term STP-SVD of matrices

While the single-term STP-SVD (Subsection 4.1) establishes an STP-based matrix factorization framework, its approximation accuracy is inherently limited by a single Kronecker component. To address this bottleneck and achieve more flexible, precise approximation, we propose a multi-term STP decomposition scheme that integrates multiple orthogonal components. This approach substantially improves approximation fidelity while preserving the structural properties and computational eficiency of the original single-term formulation. We first present a supporting lemma before detailing the multi-term decomposition.

Lemma 4.2. [39] Given $\mathbf { A } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } }$ and let k be a given positive integer. Then there exist matrices $\mathbf { B } _ { i } \in \mathbb { R } ^ { m _ { 1 } \times n _ { 1 } }$ and $\mathbf { C } _ { i } \in \mathbb { R } ^ { m _ { 2 } \times n _ { 2 } } ( i = 1 , 2 , \cdot \cdot \cdot$ k) such that $\begin{array} { r l } { \mathrm { v e c } ( \mathbf { B } _ { i } ) = } \end{array}$ ${ \sqrt { \sigma _ { i } } } \mathbf { U } ( : , i ) , { \mathrm { v e c } } ( \mathbf { C } _ { i } ) = { \sqrt { \sigma _ { i } } } \mathbf { V } ( : , i )$ which minimize

$$
\left\| \mathbf { A } - \sum _ { i = 1 } ^ { k } \mathbf { B } _ { i } \otimes \mathbf { C } _ { i } \right\| _ { F } ,\tag{4}
$$

where $\sigma _ { i }$ denotes the i-th singular value of $\mathcal { R } ( { \bf A } )$ given in $( 2 ) ,$ and $\mathbf { U } ( : , i )$ and $\mathbf { V } ( : , i )$ are the corresponding left and right singular vectors, respectively.

We formally introduce this improved decomposition in Theorem 4.2, along with a rigorous theoretical analysis of its approximation error.

Theorem 4.2 (Multi-term STP-SVD of matrices). Let $\mathbf { A } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } }$ and k be a given positive integer. Then A can befactorized as

$$
\mathbf { A } = \sum _ { i = 1 } ^ { k } \mathbf { U } _ { i } \ltimes \Sigma _ { i } \ltimes \mathbf { V } _ { i } ^ { \top } + \mathbf { E } _ { k } ,\tag{5}
$$

where $\mathbf { U } _ { i } \in \mathbb { R } ^ { m _ { 1 } \times m _ { 1 } }$ and $\mathbf { V } _ { i } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 1 } } ( i = 1 , 2 , \cdots , k )$ are orthogonal, and each $\pmb { \Sigma } _ { i } \in$ R<sup>m</sup>1<sup>m</sup>2<sup>×n</sup>1<sup>n</sup>2 $( i = 1 , 2 , \cdots , k )$ is block-diagonal with blocks $\mathbf { S } _ { i j } \in \mathbb { R } ^ { m _ { 2 } \times n _ { 2 } } ( j = 1 , 2 , \cdots , p )$ and $p = \min \{ m _ { 1 } , n _ { 1 } \}$ , satisfying $\| \mathbf { S } _ { i 1 } \| _ { F } \geq \| \mathbf { S } _ { i 2 } \| _ { F } \geq \ldots \geq \| \mathbf { S } _ { i p } \| _ { F }$ . The approximation error satisfies

$$
\| \mathbf { E } _ { k } \| _ { F } ^ { 2 } = \sum _ { i = k + 1 } ^ { \nu } \sigma _ { i } ^ { 2 } ,
$$

where $\sigma _ { k + 1 } \geq \sigma _ { k + 2 } \geq \cdot \cdot \cdot \geq \sigma _ { \nu } \geq 0$ are the singular values of $\mathcal { R } ( { \bf A } ) \in \mathbb { R } ^ { m _ { 1 } n _ { 1 } \times m _ { 2 } n _ { 2 } }$ defined in (2), and $\nu = \operatorname* { m i n } \{ m _ { 1 } n _ { 1 } , m _ { 2 } n _ { 2 } \}$

Proof. The proof can be found in the Supplementary Material.

Remark 4.2. Compared with the STP-SVD (Theorem $4 . l ) _ { : }$ , the multi-term formulation in Theorem 4.2 provides a more accurate approximation of the target matrix. In particular, when $k = 1$ , the multi-term decomposition collapses to the single-term case. As k increases, the approximation error $\| \mathbf { E } _ { k } \| _ { F }$ decreases monotonically, since additional Kronecker components are included to capture more information from the original matrix. We now present Algorithm 2for the multi-term STP-SVD.

Algorithm 2: Multi-term STP-SVD of matrices   
Input: $\overline { { \mathbf { A } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } } } }$ , the number of terms k.   
Output: $\mathbf { U } _ { i } , \pmb { \Sigma } _ { i } , \mathbf { V } _ { i } .$   
1 Calculate matrices $\mathbf { B } _ { i } \in \mathbb { R } ^ { m _ { 1 } \times n _ { 1 } }$ and $\mathbf { C } _ { i } \in \mathbb { R } ^ { m _ { 2 } \times n _ { 2 } } ( i = 1 , \dots , k )$ via Lemma 4.2,   
such that $\begin{array} { r } { \mathbf { A } \approx \sum _ { i = 1 } ^ { k } \mathbf { B } _ { i } \otimes \mathbf { C } _ { i } ; } \end{array}$   
2 for i = 1 to k do   
3 Perform full SVD on $\mathbf { B } _ { i } \colon [ \mathbf { U } _ { i } , \boldsymbol { \Sigma } _ { \mathbf { B } _ { i } } , \mathbf { V } _ { i } ] = \mathrm { s v d } ( \mathbf { B } _ { i } ) ;$   
4 $\pmb { \Sigma } _ { i } = \pmb { \Sigma } _ { \mathbf { B } _ { i } } \otimes \mathbf { C } _ { i } ,$ ;   
5 return $\mathbf { U } _ { i } , \pmb { \Sigma } _ { i } , \mathbf { V } _ { i } .$

Remark 4.3. Theorem 4.2 gives the full multi-term STP-SVD of matrices. To reduce computational overhead, we develop a truncated variant that retains only the leading r singular components of each $\mathbf { B } _ { i } .$ . The resulting decomposition preserves the same factorization structure and admits thefollowing error bound:

$$
\| \mathbf { E } _ { k , r } \| _ { F } ^ { 2 } \leq \sum _ { i = k + 1 } ^ { \nu } \sigma _ { i } ^ { 2 } + \sum _ { i = 1 } ^ { k } \sum _ { j = r + 1 } ^ { p } \| \mathbf { S } _ { i j } \| _ { F } ^ { 2 } ,\tag{6}
$$

where $\mathbf { E } _ { k , r }$ denotes the approximation error induced by the truncated multi-term STP-SVD. The first term on the right-hand side originates from the rank-k truncated SVD over $\mathcal { R } ( \mathbf { A } )$ , while the second term arises from truncated SVD for each matrix $\mathbf { B } _ { i } .$ Here, r standsfor the number ofretained singular componentsfor truncation on each $\mathbf { B } _ { i } .$ For brevity, its detailed computational procedure is deferred to Supplementary Material.

## 4.3. Multi-term STP-SVD oftensors

Theorem 4.2 establishes the multi-term STP-SVD for general real matrices, which approximates a target matrix by summing multiple orthogonal STP factorization components. Benefiting from the tensor STP defined under arbitrary invertible linear trans-

forms in Definition 3.1, this matrix decomposition paradigm can be naturally generalized to third-order tensors, giving rise to the tensor MSTP-SVD formulation shown in Theorem 4.3 with structurally consistent factorization form.

Theorem 4.3 (MSTP-SVD). Let L be any invertible linear transform in (1) and the transform matrix L satisfies ${ \bf L } ^ { \mathrm { H } } { \bf L } = { \bf L } { \bf L } ^ { \mathrm { H } } = \rho { \bf I } _ { l }$ and ${ \bf L } ^ { - 1 } = { \bf L } ^ { \mathrm { H } } / \rho$ for some constant $\rho > 0 ,$ , and $\mathcal { A } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } \times l }$ . Then it can befactorized as

$$
\mathcal { A } = \sum _ { i = 1 } ^ { k } \mathcal { U } _ { i } \ltimes _ { L } S _ { i } \ltimes _ { L } \mathcal { V } _ { i } ^ { \top } + \mathcal { E } _ { k } ,\tag{7}
$$

where $\mathcal { U } _ { i } \in \mathbb { R } ^ { m _ { 1 } \times m _ { 1 } \times l }$ and $\mathcal { V } _ { i } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 1 } \times l } \left( i = 1 , 2 , \cdots , k \right)$ are orthogonal, each frontal slice of $S _ { i } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } \times l }$ is a block-diagonal matrix, and $\mathcal { E } _ { k }$ is an error tensor, its squared Frobenius norm satisfies

$$
\| \mathcal { E } _ { k } \| _ { F } ^ { 2 } = \frac { 1 } { \rho } \sum _ { j = 1 } ^ { l } \sum _ { i = k + 1 } ^ { \nu } ( \hat { \sigma } _ { i } ^ { ( j ) } ) ^ { 2 } ,\tag{8}
$$

where $\hat { \sigma } _ { i } ^ { ( j ) }$ is the i-th singular value of $\mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } )$ and $\nu = \operatorname* { m i n } \{ m _ { 1 } n _ { 1 } , m _ { 2 } n _ { 2 } \}$

Proof. The proof is provided in Supplementary Material.

Remark 4.4. As implied by the constructive proof of Theorem 4.3, the MSTP-SVD of tensor A can be computed by performing the matrix MSTP-SVD on each frontal slice of A<sup>¯</sup>. Algorithm 3 summarizes this procedure.

Remark 4.5. In Subsection 4.2, we introduced the truncated MSTP-SVDfor matrices, and the framework can be naturally generalized to third-order tensors herein. For brevity, its detailed computational procedure is deferred to Supplementary Material

The core idea of truncated MSTP-SVD (TMSTP-SVD) is to perform a truncated matrix MSTP-SVD on each $\mathbf { B } _ { i } ^ { ( j ) }$ when decomposing the $\bar { \mathcal { A } } ^ { ( j ) }$ . For this purpose, let ${ \bf R } = [ R _ { i j } ] \in \mathbb { N } _ { + } ^ { k \times l }$ be a matrix of positive integers, where $R _ { i j }$ denotes the truncation rank for the SVD of $\mathbf { B } _ { i } ^ { ( j ) }$ . For the tensor factorization (7), the number of diagonal blocks within the j-th frontal slice $S _ { i } ^ { ( j ) }$ precisely equals $R _ { i j }$ . Based on this correspondence, we formally define the rank-truncation matrix $\mathbf { R } \in \mathbb { N } _ { + } ^ { k \times l }$ below.

Algorithm 3: MSTP-SVD method of tensors   
Input: A ∈ R<sup>m1m2×n1n2×l</sup>, the number of terms k.   
Output: $\mathcal { U } _ { i } , S _ { i } , \mathcal { V } _ { i } .$   
1 Obtain $\bar { \mathcal { A } }$ by applying an invertible linear transform L on $\mathcal { A } ;$   
2 for $j = 1$ to l do   
3 Approximate the j-th frontal slice by Lemma $\begin{array} { r } { \mathbf { \ell . 2 : } \bar { \mathcal { A } } ^ { ( j ) } \approx \sum _ { i = 1 } ^ { k } \mathbf { B } _ { i } ^ { ( j ) } \otimes \mathbf { C } _ { i } ^ { ( j ) } } \end{array}$   
4 for $i = 1$ to k do   
5 Compute the full SVD of $\mathbf { B } _ { i } ^ { ( j ) } \colon [ \mathbf { U } _ { i } ^ { ( j ) } , \boldsymbol { \Sigma } _ { \mathbf { B } _ { i } } ^ { ( j ) } , \mathbf { V } _ { i } ^ { ( j ) } ] = \mathrm { s v d } \left( \mathbf { B } _ { i } ^ { ( j ) } \right) ;$   
6 $\pmb { \Sigma } _ { i } ^ { ( j ) } = \pmb { \Sigma } _ { \pmb { \mathrm { B } } _ { i } } ^ { ( j ) } \otimes \mathbf { C } _ { i } ^ { ( j ) } ;$   
7 Store $\mathbf { U } _ { i } ^ { ( j ) } , \pmb { \Sigma } _ { i } ^ { ( j ) } , \mathbf { V } _ { i } ^ { ( j ) }$ into $\mathcal { U } _ { i } , S _ { i } , \mathcal { V } _ { i } ,$ respectively;   
8 return $\mathcal { U } _ { i } = L ^ { - 1 } ( \mathcal { U } _ { i } ) , S _ { i } = L ^ { - 1 } ( S _ { i } ) , \mathcal { V } _ { i } = L ^ { - 1 } ( \mathcal { V } _ { i } ) .$

Definition 4.1. Consider $\mathcal { A } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } \times l }$ that obeys the full MSTP-SVD factorization given in (7). We define the truncation rank matrix ${ \bf R } = [ R _ { i j } ] \in \mathbb { N } _ { + } ^ { k \times l }$ , whose entry $R _ { i j }$ stands for the retained singular rank adopted for matrix $\mathbf { B } _ { i } ^ { ( j ) }$ during the decomposition of the $\bar { \mathcal { A } } ^ { ( j ) }$ . For the truncated variant with rank matrix R, we denote the corresponding approximation error tensor as $\mathcal { E } _ { k , \mathrm { I } }$ <sub>R</sub>.

Remark 4.6. Suppose that the transform matrix L satisfies ${ \bf L } ^ { \mathrm { H } } { \bf L } = { \bf L } { \bf L } ^ { \mathrm { H } } = \rho { \bf I } _ { l }$ and ${ \bf L } ^ { - 1 } = { \bf L } ^ { \mathrm { H } } / \rho$ for some constant $\rho > 0 .$ . For the TMSTP-SVD with k multi-terms and truncation rank matrix R, the approximation error tensor $\mathcal { E } _ { k , { \bf R } }$ obeys the following squared Frobenius norm upper bound:

$$
\begin{array} { r } { \displaystyle \| \mathscr { E } _ { k , \mathbf { R } } \| _ { F } ^ { 2 } = \left( \frac { 1 } { \sqrt { \rho } } \| \mathsf { b } \mathrm { d i a g } ( \bar { \mathscr { E } } _ { k , \mathbf { R } } ) \| _ { F } \right) ^ { 2 } = \frac { 1 } { \rho } \displaystyle \sum _ { j = 1 } ^ { l } \| \bar { \mathscr { E } } _ { k , \mathbf { R } } ^ { ( j ) } \| _ { F } ^ { 2 } } \\ { \leq \frac { 1 } { \rho } \displaystyle \sum _ { j = 1 } ^ { l } \biggl ( \displaystyle \sum _ { i = k + 1 } ^ { \nu } ( \hat { \sigma } _ { i } ^ { ( j ) } ) ^ { 2 } + \displaystyle \sum _ { i = 1 } ^ { k } \displaystyle \sum _ { t = R _ { i j } + 1 } ^ { p } \| \mathbf { S } _ { i t } ^ { ( j ) } \| _ { F } ^ { 2 } \biggr ) , } \end{array}\tag{9}
$$

where the last inequality isfrom (6). The two summation terms inside the parentheses correspond to two independent sources ofapproximation error:

• The term $\textstyle \sum _ { i = k + 1 } ^ { \nu } ( \hat { \sigma } _ { i } ^ { ( j ) } ) ^ { 2 }$ arises from truncating the trailing singular values of the rearranged matrix $\mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } )$ , where $\hat { \sigma } _ { i } ^ { ( j ) }$ are the singular values of $\mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } )$ and

![](images/90905abfa04f50685cfcbf3882f7341be560695a59c9e454a5efdba607defaef.jpg)  
Fig. 1: Runtime comparison of SVD computations for $\mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } )$ (Case 1) and submatrices $\mathbf { B } _ { i } ^ { ( j ) }$ (Case 2) under MSTP-SVD (k = 1).

$$
\nu = \operatorname* { m i n } \{ m _ { 1 } n _ { 1 } , m _ { 2 } n _ { 2 } \} .
$$

• The term $\begin{array} { r } { \sum _ { i = 1 } ^ { k } \sum _ { t = R _ { i j } + 1 } ^ { p } | | \mathbf { S } _ { i t } ^ { ( j ) } | | _ { F } ^ { 2 } } \end{array}$ originates from the rank- $\mathbf { \nabla } \cdot R _ { i j }$ truncated SVD of each matrix $\mathbf { B } _ { i } ^ { ( j ) }$ , where $\mathbf { S } _ { i t } ^ { ( j ) }$ are the diagonal blocks located in the j-th frontal slice of ${ \bar { S } } _ { i } ,$ and $p = \min \{ m _ { 1 } , n _ { 1 } \}$

## 5. Fast randomized tensor singular value decomposition via semi-tensor product

Despite improved approximation accuracy, deterministic MSTP-SVD (Subsection 4.3) incurs heavy computational cost dominated by SVD on the large rearranged matrix $\mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } )$ , as verified in Fig. 1 with four high-resolution RGB images. We therefore adopt a randomized acceleration scheme, replacing the costly exact SVD with its randomized counterpart for notable speedup with negligible accuracy loss.

Randomized T-SVD projects high-dimensional tensors onto a low-dimensional subspace via random projection, performs eficient T-SVD in the subspace, and reconstructs results in the original space, achieving significant complexity reduction with bounded approximation error.

As the t-product under any invertible linear transform decouples into slice-wise matrix multiplications [23], we analyze randomized T-SVD entirely in the transform domain. We generate a Gaussian random tensor $\mathcal { G }$ (each frontal slice $\bar { g } ^ { ( j ) }$ has independent standard-normal entries [40]) and apply transform L to both A and $\mathcal { G }$ to obtain $\bar { \mathcal { A } }$ and ${ \bar { g } } .$ . Direct full SVD on each large-scale frontal slice $\bar { \mathcal { A } } ^ { ( j ) }$ incurs prohibitive cost. Naive random projection ${ \bf Y } _ { 0 } = \bar { \mathcal { A } } ^ { ( j ) } \bar { \mathcal { G } } ^ { ( j ) }$ sufers from large approximation bias when the singular-value gap of $\bar { \mathcal { A } } ^ { ( j ) }$ is narrow, requiring heavy oversampling [41]. We thus

adopt power-enhanced random projection:

$$
\mathbf { Y } = ( \bar { \mathcal { A } } ^ { ( j ) } ( \bar { \mathcal { A } } ^ { ( j ) } ) ^ { \top } ) ^ { q } \bar { \mathcal { A } } ^ { ( j ) } \bar { \mathcal { G } } ^ { ( j ) } .
$$

Remark 5.1. Substituting the SVD of $\bar { \mathcal { A } } ^ { ( j ) }$ into the power term shows that singular values are raised to the $2 q + 1$ . This widens the singular-value gap $\tau _ { k } ^ { \left( j \right) } = \hat { \sigma } _ { k + 1 } ^ { \left( j \right) } / \hat { \sigma } _ { k } ^ { \left( j \right) }$ ， suppresses trivial components, and mitigates oversampling-induced bias, yielding a tighter error bound with only marginal extra computational cost.

After obtaining Y, thin QR factorization yields an orthonormal basis $\mathbf { Q } _ { j }$ for the principal subspace of $\bar { \mathcal { A } } ^ { ( j ) }$ . Projecting $\bar { \mathcal { A } } ^ { ( j ) }$ onto $\mathbf { Q } _ { j }$ gives a compact matrix ${ \textbf { B } } =$ $\mathbf { Q } _ { i } ^ { \top } \bar { \mathcal { A } } ^ { ( j ) }$ , whose SVD is far cheaper. We retain the top-k singular components, store truncated factors $\mathbf { U } , \mathbf { S } , \mathbf { V } ^ { \top }$ in the transform domain, and apply $L ^ { - 1 }$ to recover T-SVD factors in the original space. The full procedure is outlined in Algorithm 4.

Algorithm 4: Randomized T-SVD with power iteration (RT-SVD) [32]   
Input: $\mathcal { A } \in \mathbb { R } ^ { m \times n \times l }$ , truncation term $k ,$ oversampling parameter $s \geq 0 ,$   
iteration parameter $q \geq 0 .$   
Output: $\mathcal { U } _ { k } , S _ { k } , \mathcal { V } _ { k } .$   
1 Generate a Gaussian random tensor $\mathcal { G } \in \mathbb { R } ^ { n \times ( k + s ) \times l } ;$   
2 Compute $\bar { \mathcal { A } } = L ( \mathcal { A } )$ and $\bar { \mathcal { G } } = L ( \mathcal { G } ) ;$   
3 for $j = 1$ to l do   
4 Compute ${ \bf Y } = ( \bar { \mathcal { A } } ^ { ( j ) } ( \bar { \mathcal { A } } ^ { ( j ) } ) ^ { \top } ) ^ { q } \bar { \mathcal { A } } ^ { ( j ) } \bar { \mathcal { G } } ^ { ( j ) } ;$   
5 Compute thin-QR factorization $\mathbf { Y } = \mathbf { Q } _ { j } \mathbf { R } ;$   
6 Compute $\begin{array} { r } { \mathbf { B } = \mathbf { Q } _ { j } ^ { \top } \bar { \mathcal { A } } ^ { ( j ) } ; } \end{array}$   
7 Compute the SVD of B: $\mathbf B = \mathbf { U } \mathbf { S } \mathbf { V } ^ { \top }$   
8 Form $\mathbf { U } _ { k } , \mathbf { V } _ { k } , \mathbf { S } _ { k }$ by truncating Q<sub>j</sub>U, V, S with k;   
9 Set $\bar { \mathcal { U } } _ { k } ^ { \mathrm { ~ } ( j ) } = \mathbf { U } _ { k } , \bar { S _ { k } } ^ { ( j ) } = \mathbf { S } _ { k } , \bar { \mathcal { V } } _ { k } ^ { ( j ) } = \mathbf { V } _ { k } ;$   
10 return $\mathcal { U } _ { k } = L ^ { - 1 } ( \bar { \mathcal { U } } _ { k } ) , S _ { k } = L ^ { - 1 } ( \bar { S } _ { k } ) , \mathcal { V } _ { k } = L ^ { - 1 } ( \bar { \mathcal { V } } _ { k } ) .$

Based on the runtime results in Fig. 1, which verifies that SVD on $\mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } )$ dominates the computational cost of deterministic MSTP-SVD, we embed the above randomized subspace extraction scheme into the MSTP-SVD framework. Concretely, we replace $\bar { \mathcal { A } } ^ { ( j ) }$ adopted in standard randomized T-SVD with the rearranged matrix $\mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } )$ defined for multi-term semi-tensor modeling, leading to our accelerated MRSTP-SVD algorithm. Its complete pipeline is provided in Algorithm 5.

We further derive the unified expected error bound for the proposed MRSTP-SVD algorithm, which is stated in the following theorem.

Theorem 5.1. Let L be any invertible linear transform in (1), and the transform matrix L satisfies ${ \bf L } ^ { \mathrm { H } } { \bf L } = { \bf L } { \bf L } ^ { \mathrm { H } } = \rho { \bf I } _ { l }$ and ${ \bf L } ^ { - 1 } = { \bf L } ^ { \mathrm { H } } / \rho$ for some constant $\rho > 0 .$ . Suppose that $\mathcal { A } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } \times l }$ and $\mathcal { G } \in \mathbb { R } ^ { m _ { 2 } n _ { 2 } \times ( k + s ) \times l }$ is a Gaussian random tensor with $k + s \leq$ min{m<sub>1</sub>n<sub>1,</sub> m<sub>2</sub>n<sub>2</sub>}. $\begin{array} { r } { I f \widetilde { \mathcal { A } } = \sum _ { i = 1 } ^ { k } \mathcal { U } _ { i } \ltimes _ { L } S _ { i } \ltimes _ { L } \mathcal { V } _ { i } ^ { \top } } \end{array}$ , where $\mathcal { U } _ { i } , S _ { i } ,$ and $\mathcal { N } _ { i }$ are obtained from Algorithm 5, then,

$$
\mathbb { E } \Vert \mathcal { A } - \widetilde { \mathcal { A } } \Vert _ { F } ^ { 2 } \leq \frac { 2 } { \rho } \sum _ { j = 1 } ^ { l } \left[ \left( 2 + \frac { k } { s - 1 } ( \tau _ { k } ^ { ( j ) } ) ^ { 4 q } \right) \left( \sum _ { i = k + 1 } ^ { \nu } ( \hat { \sigma } _ { i } ^ { ( j ) } ) ^ { 2 } \right) \right] ,\tag{10}
$$

where $\nu = \operatorname* { m i n } \{ m _ { 1 } n _ { 1 } , m _ { 2 } n _ { 2 } \}$ , k is the target truncation term, $s \geq 2$ denotes the oversampling parameter , $q$ is the number of power iteration steps, $\hat { \sigma } _ { i } ^ { ( j ) }$ denotes the i-th singular value of $\mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } )$ $\tau _ { k } ^ { ( j ) } = \hat { \sigma } _ { k + 1 } ^ { ( j ) } / \hat { \sigma } _ { k } ^ { ( j ) } \ll 1$ is the singular value gap.

Proof. The detailed proof is provided in Supplementary Material.

Theorem 5.1 reveals a delicate three-way trade-of among hyperparameters $k ,$ s, and $q .$ Inceasing the truncation term k reduces residual energy from truncated trailing singular components but amplifies random-sampling bias. We can mitigate this sampling error by adopting a larger oversampling parameter s. Especially for slowlydecaying singular values of $\mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } )$ , more power-iteration steps q suppress the detrimental efect of the singular-value gap $\tau _ { k } ^ { \left( j \right) }$ , albeit with additional computational overhead.

Remark 5.2. The above error bound is derived based onfull MSTP-SVD, ifwe adopt the TMSTP-SVD scheme defined in Supplementary Material, the deterministic residual $\| \mathcal { A } - \mathcal { A } _ { M S T P } \| _ { F } ^ { 2 }$ is updated to (9). Substituting this truncated deterministic error into the proof flow of Theorem 5.1, the overall expected squared Frobenius error of truncated MRSTP-SVD (TMRSTP-SVD) reads

$$
\mathbb { E } \Vert \mathcal { R } - \widetilde { \mathcal { M } } _ { T r u n c } \Vert _ { F } ^ { 2 } \le \frac { 2 } { \rho } \sum _ { j = 1 } ^ { l } \left[ \left( 2 + \frac { k } { s - 1 } ( \tau _ { k } ^ { ( j ) } ) ^ { 4 q } \right) \left( \sum _ { i = k + 1 } ^ { \nu } ( \hat { \sigma } _ { i } ^ { ( j ) } ) ^ { 2 } \right) + \sum _ { i = 1 } ^ { k } \sum _ { t = R _ { i j } + 1 } ^ { p } \Vert \mathbf { S } _ { i t } ^ { ( j ) } \Vert _ { F } ^ { 2 } \right] .
$$

Compared with (10), the extra term $\begin{array} { r } { \sum _ { i = 1 } ^ { k } \sum _ { t = R _ { i j } + 1 } ^ { p } \left\| S _ { i t } ^ { ( j ) } \right\| _ { F } ^ { 2 } } \end{array}$ originates from rank- $\mathbf { \nabla } \cdot R _ { i j }$ truncated SVD for each block matrix $B _ { i } ^ { ( j ) }$ , where $S _ { i t } ^ { ( j ) }$ are diagonal blocks in the $j -$ thfrontal slice of ${ \bar { S } } _ { i } ,$ and $p = \min \{ m _ { 1 } , n _ { 1 } \}$

Correspondingly, we outline the full procedure of TMRSTP-SVD, which serves as a simplified truncated counterpart to Algorithm 5. For brevity, its pseudocode is deferred to Supplementary Material.

## 6. Numerical experiments

In this section, we conduct numerical experiments on real-world data and compare with other algorithms: truncated T-SVD (TT-SVD) [22], STP-SVD and truncated STP-SVD (TSTP-SVD) [30] to substantiate the superiority and efectiveness of our methods. All simulations are performed on a laptop computer with 2.50GHz Intel(R) Core(TM) i5-10300H CPU and 24GB memory. The Peak Signal-to-Noise Ratio (PSNR), the structural similarity (SSIM), and the CPU runtime are employed to evaluate the performance of the proposed algorithm. Let X, $\hat { X } \in \mathbb { R } ^ { m \times n \times l }$ be the original tensor and its reconstruction. The PSNR and SSIM are defined as

$$
\mathrm { P S N R } = 1 0 \log _ { 1 0 } \left( \frac { \| \boldsymbol { \chi } \| _ { \infty } ^ { 2 } } { \frac { 1 } { m n l } \| \boldsymbol { \chi } - \hat { \boldsymbol { \chi } } \| _ { F } ^ { 2 } } \right) , \mathrm { S S I M } = \frac { \left( 2 \mu _ { \boldsymbol { \chi } } \mu _ { \hat { \boldsymbol { \chi } } } + C _ { 1 } \right) \left( 2 \sigma _ { \boldsymbol { \chi } \hat { \boldsymbol { \chi } } } + C _ { 2 } \right) } { \left( \mu _ { \boldsymbol { \chi } } ^ { 2 } + \mu _ { \hat { \boldsymbol { \chi } } } ^ { 2 } + C _ { 1 } \right) \left( \sigma _ { \boldsymbol { \chi } } ^ { 2 } + \sigma _ { \hat { \boldsymbol { \chi } } } ^ { 2 } + C _ { 2 } \right) } ,
$$

where $\| X \| _ { \infty }$ denotes the maximum absolute value of all entries in $\chi , \sigma _ { \chi \hat { \chi } }$ is the crosscovariance between X and $\hat { X } , \mu _ { X } , \mu _ { \hat { X } }$ represent the average values of X and $\hat { X } , \sigma _ { X } , \sigma _ { \hat { X } }$ are the standard deviations, $C _ { 1 } , C _ { 2 }$ are constants. In general, larger PSNR and SSIM values correspond to superior reconstruction quality.

## 6.1. Compression on the image data

In this section, we employ the proposed algorithm for image compression. First, we select three distinct linear transforms: DFT, discrete cosine transform (DCT), and random orthogonal transform (ROT), and test their impacts on the performance of our proposed algorithm on four RGB benchmark images (Lake, Night, Road, and Fruit) downloaded from the ISO Republic website<sup>1</sup>. Their corresponding resolutions are 6016 × 4016 × 3, 4920 × 3280 × 3, 8192 × 5464 × 3, and $6 0 0 0 \times 4 0 0 0 \times 3$ , respectively. Fig. 2 reports PSNR, SSIM and runtime (in seconds) under diferent transform schemes, showcasing results for the lake and road images. The three invertible linear transforms deliver comparable PSNR and SSIM, while DFT exhibits persistently lower computational overhead. Averaged across two test images, DFT reduces total runtime by over 10% compared with DCT and ROT. We therefore select DFT as the default linear transform for all subsequent experiments. Additional results for other test images in the Supplementary Material further validate this finding.

Algorithm 5: MRSTP-SVD method of tensors   
Input: $\mathcal { A } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } \times l } ,$ the number of terms $k ,$ oversampling parameter   
$s \geq 0 ,$ iteration parameter $q \geq 0 ,$ truncated rank matrix $\mathbf { R } .$   
Output: $\mathcal { U } _ { i } , S _ { i } , \mathcal { V } _ { i } .$   
1 Generate a Gaussian random tensor $\mathcal { G } \in \mathbb { R } ^ { n _ { 1 } n _ { 2 } \times ( k + s ) \times l } ;$   
2 Compute $\bar { \mathcal { A } } = L ( \mathcal { A } )$ and $\bar { \mathcal { G } } = L ( \mathcal { G } ) ;$   
3 for $j = 1$ to l do   
4 Obtain $\mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } )$ by reorganizing the blocks of $\bar { \mathcal { A } } ^ { ( j ) } ;$   
5 Compute ${ \bf Y } = ( \mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } ) \mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } ) ^ { \top } ) ^ { q } \mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } ) \bar { \mathcal { G } } ^ { ( j ) } ;$   
6 Compute thin-QR factorization $\mathbf { Y } = \mathbf { Q } _ { j } \mathbf { R } ;$   
7 Compute $\mathbf { B } = \mathbf { Q } _ { j } ^ { \top } \mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } ) ;$   
8 Compute the SVD of B: $\mathbf B = \mathbf { U } \mathbf { S } \mathbf { V } ^ { \top }$ ;   
9 Form $\mathbf { U } _ { k } , \mathbf { V } _ { k } , \mathbf { S } _ { k }$ by truncating Q U, V, S with $k ;$   
10 for i = 1 to k do   
11 $\mathrm { v e c } ( \mathbf { B } _ { i } ^ { ( j ) } ) = \sqrt { \mathbf { S } _ { k } ( i , i ) } \mathbf { U } _ { k } ( : , i ) ,$ vec $( \mathbf { C } _ { i } ^ { ( j ) } ) = \sqrt { \mathbf { S } _ { k } ( i , i ) } \mathbf { V } _ { k } ( : , i )$ such that   
$\begin{array} { r } { \bar { \mathcal { A } } ^ { ( j ) } \approx \sum _ { i = 1 } ^ { k } \mathbf { B } _ { i } ^ { ( j ) } \otimes \mathbf { C } _ { i } ^ { ( j ) } ; } \end{array}$   
12 Compute the full SVD of $\mathbf { B } _ { i } ^ { ( j ) } \colon [ \mathbf { U } _ { i } ^ { ( j ) } , \boldsymbol { \Sigma } _ { B _ { i } } ^ { ( j ) } , \mathbf { V } _ { i } ^ { ( j ) } ] = \mathrm { s v d } \left( \mathbf { B } _ { i } ^ { ( j ) } \right) ;$   
13 $\pmb { \Sigma } _ { i } ^ { ( j ) } = \pmb { \Sigma } _ { \pmb { \mathrm { B } } _ { i } } ^ { ( j ) } \otimes \mathbf { C } _ { i } ^ { ( j ) } ;$   
14 Store $\mathbf { U } _ { i } ^ { ( j ) } , \pmb { \Sigma } _ { i } ^ { ( j ) } , \mathbf { V } _ { i } ^ { ( j ) }$ into $\mathcal { U } _ { i } , S _ { i } , \mathcal { V } _ { i } ,$ , respectively;   
15 return $\mathcal { U } _ { i } = L ^ { - 1 } ( \mathcal { U } _ { i } ) , S _ { i } = L ^ { - 1 } ( S _ { i } ) , \mathcal { V } _ { i } = L ^ { - 1 } ( \mathcal { V } _ { i } ) .$

0.8   
3   
<sub>S</sub>IM <sub>S</sub>N<sup>R</sup> 15<sub>me</sub> (<sub>s</sub>)   
10   
MSTP-SVD MSTP-SVD MSTP-SVD MRSTP-SVD MRSTP-SVD MRSTP-SVD MSTP-SVD MSTP-SVD MSTP-SVD MRSTP-SVD MRSTP-SVD MRSTP-SVD MSTP-SVD MSTP-SVD MSTP-SVD MRSTP-SVD MRSTP-SVD MRSTP-SVD   
1-5 513 1= 51=3 1F-5 1- 1-3 51=1 51=2 51=3 51=-5 3 51F 51=2 51=3   
0.   
2   
0.<sub>S</sub>I<sub>M</sub> <sub>S</sub>N<sup>R</sup> Time (s)   
10   
0.2   
MSTP-SVD MSTP-SVD MSTP-SVD MRSTP-SVD MRSTP-SVD MRSTP-SVD MSTP-SVD MSTP-SVD MSTP-SVD MRSTP-SVD MRSTP-SVD MRSTP-SVD MSTP-SVD MSTP-SVD MSTP-SVD MRSTP-SVD MRSTP-SVD MRSTP-SVD   
k=1 k=2 k=3 k=1 k=2 k=3 k-1 k=2 k-3 k=1 k=2 k=3 k=1 k=2 k=3 RSTP k=2 k=3   
DFT DCT ROT  
Fig. 2: Quantitative metrics (PSNR, SSIM, runtime) of MSTP-SVD and randomized MRSTP-SVD for image compression under invertible transforms (DFT, DCT, ROT). Top: lake; Bottom: road.

![](images/1eccf2db56916f915b05aee32e9473314b85fe3e64c7c0bc009ee1a345e3fd84.jpg)  
Fig. 3: Visual reconstruction performance, PSNR and runtime of competing and proposed methods on two representative test images.

Visual outputs together with per-image PSNR and runtime measurements for Lake and Road are presented in Fig. 3, whereas the corresponding results for Night and Fruit are provided in the Supplementary Material. As shown in Fig. 3, our proposed methods can better preserve local structural details of original images. By contrast, baseline methods tend to produce over-smoothed outputs and lose subtle image contents. For the four test images, STP block partition sizes $( m _ { 2 } , n _ { 2 } )$ are adaptively assigned according to spatial resolution to balance approximation flexibility and computational eficiency. Specifically, $( m _ { 2 } , n _ { 2 } ) = ( 4 , 4 )$ for Lake and Night, (4 6) for Fruit, and (8 8) for the higher-resolution Road image. Frontal-slice truncation ranks of TT-SVD are 50, 100, 100 and 250 for Lake, Night, Fruit and Road, respectively. Within MSTP-SVD and MRSTP-SVD, the truncation rank matrix is $\mathbf { R } = c \mathbf { J } _ { k \times l } ,$ where c denotes a uniform rank shared over all terms and frontal slices, and $\mathbf { J } _ { k \times l }$ is the $k \times l$ all-ones matrix. We set c equal to the TT-SVD truncation rank for each image to guarantee fair comparisons under identical rank budgets. For the randomized MRSTP-SVD variant, we use oversampling $s = 5$ and power iterations $q = 1$ , achieving a desirable trade-of between reconstruction accuracy and computational overhead.

We further conduct quantitative image compression experiments on 20 RGB test images. Fig. 4 presents per-image PSNR, SSIM and runtime, and Table 1 summarizes the averaged performance. As shown in Table 1, deterministic MSTP-SVD $( k = 3 )$ achieves 33.67 dB average PSNR and 0.959 average SSIM, outperforming singleterm STP-SVD by 4.96 dB and baseline TT-SVD by 8.07 dB in PSNR, validating the superior representation capability of multi-term semi-tensor decomposition. The randomized MRSTP-SVD $( k = 3 )$ delivers nearly identical reconstruction quality (33.64 dB PSNR) while cutting average runtime from 7.57 s to 5.31 s, achieving a favorable accuracy-eficiency trade-of.

Table 1: Average PSNR, SSIM and runtime comparison of competing and proposed algorithms over twenty test images.
<table><tr><td rowspan="2">Metric</td><td rowspan="2"></td><td rowspan="2">TT-SVD STP-SVD TSTP-SVD</td><td rowspan="2"></td><td colspan="2">MSTP-SVD</td><td colspan="2">TMSTP-SVD</td><td colspan="3">MRSTP-SVD</td><td colspan="3">TMRSTP-SVD</td></tr><tr><td>k = 2</td><td>k = 3</td><td>k = 2</td><td>k = 3</td><td>k = 1</td><td>k = 2</td><td>k = 3</td><td>k = 1</td><td>k = 2</td><td>k = 3</td></tr><tr><td>PSNR</td><td>25.60</td><td>28.71</td><td>25.01</td><td>31.13</td><td>33.67</td><td>25.47</td><td>25.74</td><td>28.71</td><td>31.13</td><td>33.64</td><td>25.01</td><td></td><td>25.47 25.74</td></tr><tr><td>SSIM</td><td>0.830</td><td>0.902</td><td>0.816</td><td>0.937</td><td>0.959</td><td>0.824</td><td>0.828</td><td>0.902</td><td>0.937</td><td>0.959</td><td>0.816</td><td>0.824</td><td>0.828</td></tr><tr><td>Time (s)</td><td>7.58</td><td>6.18</td><td>5.86</td><td>6.74</td><td>7.57</td><td>6.25</td><td>6.88</td><td>4.03</td><td>4.66</td><td>5.31</td><td>3.74</td><td>4.22</td><td>4.65</td></tr></table>

![](images/007c3304c76d42c4c1570f873e39dd9f1b0e940b545300ed0d5dd15af017127c.jpg)  
Fig. 4: Per-image quantitative comparisons of PSNR, SSIM and runtime over twenty test images.

## 6.2. Compression on the video data

For video compression experiments, we validate the proposed algorithm on four representative test sequences from the derf video dataset<sup>2</sup>: Crosswalk, Market, Narrator, and Aerial. DFT is employed for its excellent reconstruction performance. Owing to computational constraints, we extract the first 40 frames from each sequence, yielding third-order tensors of size $2 1 6 0 \times 4 0 9 6 \times 4 0$ . For the baseline TT-SVD, the frontal-slice truncation rank is fixed at $r = 5 0 .$ . To ensure a fair comparison, the rank parameter c in our proposed MSTP-SVD and MRSTP-SVD frameworks is set identically, i.e., all entries of the truncation rank matrix R take the value 50. The block partition sizes are $( m _ { 2 } , n _ { 2 } ) = ( 4 , 4 )$ for Market, Narrator, and Aerial, and (8 8) for the more spatially complex Crosswalk sequence. Detailed configurations of the randomized algorithms, including oversampling parameter s and power iteration count q, are summarized in Table 2.

Fig. 5 plots frame-wise PSNR and SSIM over the first 40 frames for all methods. Our multi-term schemes consistently outperform baselines, and higher k yields steady improvements. The randomized variant achieves accuracy comparable to its deterministic counterpart with substantial acceleration and negligible performance loss. Fig. 6 presents visual and quantitative comparisons on sampled frames from Crosswalk and Market (results for Narrator and Aerial are in Supplementary Material). Our methods recover richer textures and finer local details, while baselines produce over-smoothed outputs. MRSTP-SVD notably reduces runtime across all sequences with negligible accuracy degradation relative to deterministic MSTP-SVD. Table 3 summarizes average PSNR, SSIM and runtime. MRSTP-SVD (k = 2 3) substantially outperforms TT-SVD and TSTP-SVD on all high-resolution videos, with over 5 dB PSNR gain on Market, nearly 4 dB on Aerial, and over 5 s runtime reduction. TMRSTP-SVD achieves further speedup with only minor acceptable accuracy loss. Overall, the proposed framework strikes a favorable accuracy-eficiency trade-of, ofering a practical solution for high-resolution video compression.

Table 2: Hyperparameter configurations (power iteration q, oversampling s) for MRSTP-SVD and TMRSTP-SVD on video sequences.
<table><tr><td rowspan="2">Video</td><td rowspan="2">Algorithm</td><td rowspan="2">q</td><td colspan="3">S</td></tr><tr><td> $k = 1$ </td><td> $k = 2$ </td><td> $k = 3$ </td></tr><tr><td>Crosswalk</td><td>(truncated) MRSTP-SVD</td><td>1</td><td>7</td><td>6</td><td>5</td></tr><tr><td>Market</td><td>(truncated) MRSTP-SVD</td><td>1</td><td>5</td><td>4</td><td>3</td></tr><tr><td>Narrator</td><td>(truncated) MRSTP-SVD</td><td>1</td><td>5</td><td>4</td><td>3</td></tr><tr><td>Aerial</td><td>(truncated) MRSTP-SVD</td><td>1</td><td>5</td><td>4</td><td>3</td></tr></table>

![](images/69367de391a12783abbdc28d37cd905e0b3f6f6704541fce9a398d727dc8524c.jpg)  
Fig. 5: Frame-wise PSNR and SSIM curves over the first 40 frames of four benchmark videos for competing baselines and our approaches (original and truncated variants). From top to bottom: Crosswalk, Market, Narrator, Aerial.

![](images/b3bd3c7264ec9a531a74d8cba05621d5c3c52e75b93aec28a26d13ab1d932a09.jpg)  
Fig. 6: Visual reconstruction examples and quantitative PSNR-SSIM comparisons of competing baselines and our approaches (original and truncated variants) on randomly sampled frames from two test video sequences.

## 6.3. Image and video completion

For image and video completion tasks, we adopt DFT as the invertible linear transform for its best reconstruction performance. Consider $\boldsymbol { X } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ that contains missing entries, and Ω the index set of observed elements. The tensor completion problem is formulated as

$$
\operatorname* { m i n } _ { \chi } \left\| P _ { \Omega } ( \chi ) - P _ { \Omega } ( \mathcal { M } ) \right\| _ { F } ^ { 2 } , \mathrm { ~ s . t . ~ R a n k } ( \chi ) = \mathbf { R } ,\tag{11}
$$

where M is the observed tensor and R controls truncation ranks across terms and slices. Following [42], we introduce an auxiliary variable $z$ to reformulate the prob-

Table 3: Average PSNR, SSIM and runtime comparison of competing baselines and our approaches (original and truncated multi-term variants) over all frames of four test videos.
<table><tr><td rowspan="2">Metric</td><td rowspan="2"></td><td rowspan="2">TT-SVD STP-SVD TSTP-SVD</td><td rowspan="2">k = 2</td><td colspan="2">MSTP-SVD</td><td colspan="2">TMSTP-SVD</td><td colspan="3">MRSTP-SVD</td><td colspan="3">TMRSTP-SVD</td></tr><tr><td>k = 3</td><td>k = 2</td><td></td><td>k = 3</td><td>k = 1</td><td>k = 2</td><td>k = 3</td><td>k = 1</td><td>k = 2</td><td>k = 3</td></tr><tr><td colspan="10">Crosswalk</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PSNR</td><td>35.84</td><td>36.76</td><td>33.98</td><td>39.10 39.73 34.8635.0636.2737.0938.87 33.72 34.10 34.70</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SSIM Time (s)</td><td>0.972 63.45</td><td>0.934</td><td>0.923</td><td>0.954 0.954 0.937 0.9370.933 0.935 0.953 0.922 0.9230.935</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">22.26 21.05</td><td colspan="3">26.7830.8622.42 26.18 13.42 16.42 18.5212.5213.51 14.73</td></tr><tr><td></td><td></td><td></td><td></td><td>Market</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PSNR</td><td>27.58</td><td>33.55</td><td>27.14</td><td colspan="10">37.5240.7727.4627.5833.0536.3038.4827.0527.3627.48</td></tr><tr><td>SSIM</td><td>0.815</td><td>0.896</td><td>0.774</td><td>0.954 0.9780.7870.7940.8880.9430.9670.772 0.7840.791</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Time (s)</td><td>63.36</td><td>61.84</td><td>54.77</td><td colspan="10">68.4976.43 60.41 64.48</td></tr><tr><td colspan="10">Narrator</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td>PSNR SSIM</td><td>32.26 0.936</td><td>34.01 0.937</td><td>30.43 0.893</td><td></td><td></td><td>0.955 0.975 0.902 0.912 0.934 0.933 0.949 0.892 0.889 0.897</td><td></td><td></td><td>35.39 36.72 30.82 31.10 35.02 35.66 36.90 30.86 31.03 31.29</td></tr><tr><td>Time (s)</td><td>63.12</td><td>57.38</td><td>53.62</td><td colspan="10">70.87 78.73 60.85 </td></tr><tr><td></td><td></td><td></td><td></td><td colspan="10">63.6840.2547.0860.5338.8442.4147.78 Aerial</td></tr><tr><td></td><td>24.39</td><td>28.14</td><td>24.09</td><td colspan="10">30.5433.0624.3624.5228.2530.7333.6824.1124.3724.54</td></tr><tr><td>PSNR SSIM</td><td>0.594</td><td>0.741</td><td>0.551</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Time (s)</td><td>87.37</td><td>62.53</td><td>58.48</td><td colspan="10">0.850 0.910 0.581 0.594 0.745 0.854 0.919 0.552 0.582 0.596 70.8181.2767.4674.9040.0553.0059.4939.5648.0552.54</td></tr></table>

lem (11) as

$$
\operatorname* { m i n } _ { \chi } \left\| \chi - \mathcal { Z } \right\| _ { F } ^ { 2 } , \mathrm { ~ s . t . ~ R a n k } ( \chi ) = \mathbf { R } , ~ P _ { \Omega } ( \mathcal { Z } ) = P _ { \Omega } ( \mathcal { M } ) .\tag{12}
$$

The solution is approximated iteratively by

$$
\begin{array} { r } { X ^ { ( n ) }  \pi _ { r } ( \mathcal Z ^ { ( n ) } ) , } \end{array}\tag{13}
$$

$$
\mathcal { Z } ^ { ( n + 1 ) } \gets {  { \mathcal { M } } _ { \Omega } } + {  { \mathcal { X } } _ { \Omega ^ { c } } ^ { ( n ) } } ,\tag{14}
$$

where $\pi _ { r } ( \cdot )$ returns the rank-r low-rank approximation. We initialize with the incomplete tensor $\chi ^ { ( 0 ) }$ and iterate until convergence. Note that $M _ { \Omega }$ equals $\chi ^ { ( 0 ) }$ and need not be recomputed each iteration. To further boost performance, we apply smoothing to ${ \mathcal { Z } } ^ { ( n + 1 ) }$ before the low-rank approximation step. Since low-rank approximation dominates computational cost for large-scale data and many iterations, we replace the TMSTP-SVD with TMRSTP-SVD in each iterative, fixing the term parameter $k = 2$ throughout all completion experiments. This randomized substitution yields nearly equivalent reconstruction quality with substantially lower per-iteration overhead.

![](images/c96602a5eaf2eee48d11fdc72b8cd40a74c587b2e9aea36513901c761ac21cdd.jpg)

Fig. 7: Visual reconstruction examples, PSNR and runtime comparisons of competing baselines and our proposed approaches on four representative test images for image completion.  
![](images/c9db0cfc93ce13b307ab645170b39c06d64a8c5605ff68a4668b450b3c152898.jpg)  
Fig. 8: Per-image PSNR, SSIM and runtime comparisons of competing baselines and our proposed approaches over twenty test images for image completion.

We evaluate image completion on 20 test images (four selected for visual comparison), with 70% of pixels randomly removed. The oversampling parameter is set to 3, and all other settings follow Subsection 6.1. Fig. 7 presents visual reconstruction results along with PSNR and runtime metrics obtained by competing baselines and our proposed approaches on these four representative images. Visually, our approaches recover richer textures and finer local structural details. Baseline methods, in contrast, tend to leave visible artifacts in the recovered regions. In terms of quantitative metrics, our approaches yield higher PSNR values. The randomized scheme maintains competitive reconstruction quality. It greatly reduces computational burden for iterative completion procedures. Fig. 8 summarizes per-image PSNR, SSIM and runtime comparisons across all twenty test images. The overall results further confirm the superiority of our framework over competing baselines in both reconstruction fidelity and computational eficiency. It renders our approach well-suited for large-scale image completion tasks requiring iterative optimization.

Video completion experiments are conducted on the four sequences from Subsection from Subsection 6.2, with 70% of pixels randomly masked in each video tensor. All experimental parameters follow the settings in Subsection 6.2. Fig. 9 shows framewise PSNR/SSIM curves, runtime statistics and visual reconstructions for Crosswalk and Market; results for Narrator and Aerial are provided in Supplementary Material. Our TMSTP-SVD and TMRSTP-SVD outperform baselines in reconstruction fidelity. The randomized TMRSTP-SVD achieves comparable PSNR and SSIM with negligible quality loss, and reduces total completion runtime by around 25% relative to its deterministic counterpart. Visually, our methods recover finer textures in missing regions while baselines produce noticeable artifacts. Consistent performance gains can be observed for the two evaluated sequences, which are further validated by the supplementary results. Overall, the video completion results corroborate the findings from image completion. The proposed randomized algorithm maintains reconstruction quality comparable to the deterministic version while achieving substantial computational savings, making it well suited for iterative completion on large-scale video data.

![](images/9a517c579c7a6e601d4882f3b07b2b44194d07b9290411a09e3f30d27f475eb9.jpg)

![](images/493a7d0ec79b7b3299068bedd5d462fbc5356f37c00178a906b466ae1fa52a59.jpg)

![](images/0b6bb3eaebb6501178eea08a695b3fc5654f698162757c37738225f468de1a65.jpg)

![](images/32a53a14749084738358954e1f5f96ebbce2bb4c87d9186c295c83bf6c32500d.jpg)

![](images/4168802965af2897d1275a6a87afd2a1b5f177b987fc20d009963aef579651db.jpg)

![](images/1d3a05ab220fbf808315b56db136611fd52b136c86b2897e38fe1ec0c31bef0f.jpg)  
Fig. 9: Objective and visual comparisons of video recovery methods on two test sequences, including PSNR-SSIM curves, runtime, and reconstruction results under 70% missing pixels.

## 7. Conclusion

This paper proposes a novel semi-tensor product for third-order tensors under a generalized t-product framework with arbitrary invertible linear transforms. Unlike fixed-transform alternatives, our construction preserves the closed-form T-SVD structure while accommodating any unitary transform for enhanced flexibility. On this basis, we develop a multi-term decomposition model (MSTP-SVD) with multiple orthogonal components, which notably improves low-rank approximation accuracy over single-term schemes. To address the computational bottleneck in large-scale applications, we further introduce a randomized variant (MRSTP-SVD) combining projection and power iteration, achieving a practical balance between reconstruction fidelity and eficiency. Theoretical error bounds are derived for both deterministic and randomized formulations to clarify the roles of key parameters. Image/video compression experiments validate the proposed method, and tensor completion experiments confirm its efectiveness as a low-rank prior, with the randomized variant delivering substantial acceleration at negligible accuracy cost. This work opens avenues for further algorithmic acceleration, adaptive parameter tuning, and extensions to higher-order tensors and broader algebraic structures.

## CRediT authorship contribution statement

Xingchen Xiao: Writing – original draft, Visualization, Methodology, Software, Validation, Conceptualization. Feng Zhang: Writing – review & editing, Supervision, Resources, Funding acquisition, Conceptualization. Wenjin Qin: Writing – review & editing, Formal analysis, Investigation, Data curation. Jianjun Wang: Writing – review & editing, Supervision, Project administration.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgment

This work was supported in part by the National Key Research and Development Program of China under Grant 2023YFA1008502; in part by Fundamental Research Funds for the Central Universities under Grant SWU-KR25013; and in part by National Natural Science Foundation of China under Grant 12101512.

## Data availability

The datasets used in this study are publicly available from the sources cited in the numerical experiments section.

## References

[1] Z. Song, Y. Chen, Z. Weihua, Adaptively robust high-order tensor factorization for low-rank tensor reconstruction, Pattern Recognit. 165 (2025) 111600. doi: 10.1016/j.patcog.2025.111600.

[2] M. Yang, Q. Luo, W. Li, M. Xiao, Nonconvex 3d array image data recovery and pattern recognition under tensor framework, Pattern Recognit. 122 (2022) 108311. doi:10.1016/j.patcog.2021.108311.

[3] F. Zhang, J. Wang, W. Wang, C. Xu, Low-tubal-rank plus sparse tensor recovery with prior subspace information, IEEE Trans. Pattern Anal. Mach. Intell. 43 (10) (2020) 3492–3507. doi:10.1109/TPAMI.2020.2986773.

[4] J. Hou, F. Zhang, H. Qiu, J. Wang, Y. Wang, D. Meng, Robust low-tubal-rank tensor recovery from binary measurements, IEEE Trans. Pattern Anal. Mach. Intell. 44 (8) (2021) 4355–4373. doi:10.1109/TPAMI.2021.3063527.

[5] Q. Xie, Q. Zhao, D. Meng, Z. Xu, Kronecker-basis-representation based tensor sparsity and its applications to tensor recovery, IEEE Trans. Pattern Anal. Mach. Intell. 40 (8) (2017) 1888–1902. doi:10.1109/TPAMI.2017.2734888.

[6] J. Xue, Y. Zhao, W. Liao, J. C.-W. Chan, S. G. Kong, Enhanced sparsity prior model for low-rank tensor completion, IEEE Trans. Neural Netw. Learn. Syst. 31 (11) (2020) 4567–4581. doi:10.1109/TNNLS.2019.2956153.

[7] N. D. Sidiropoulos, L. De Lathauwer, X. Fu, K. Huang, E. E. Papalexakis, C. Faloutsos, Tensor decomposition for signal processing and machine learning, IEEE Trans. Signal Process. 65 (13) (2017) 3551–3582. doi:10.1109/ TSP.2017.2690524.

[8] A. Cichocki, D. Mandic, H. A. Phan, L. D. Lathauwer, G. Zhou, Q. Zhao, C. Caiafa, Tensor decompositions for signal processing applications: From two-way to multiway component analysis, IEEE Signal Process. Mag. 32 (2) (2015) 145– 163. doi:10.1109/MSP.2013.2297439.

[9] E. E. Papalexakis, C. Faloutsos, N. D. Sidiropoulos, Tensors for data mining and data fusion: Models, applications, and scalable algorithms, ACM Trans. Intell. Syst. Technol. 8 (2) (2016) 1–44. doi:10.1145/2915921.

[10] J. Sun, S. Papadimitriou, C.-Y. Lin, N. Cao, S. Liu, W. Qian, Multivis: Contentbased social network exploration through multi-way visual analysis, in: Proceedings of the 2009 SIAM International Conference on Data Mining (SDM), 2009, pp. 1064–1075. doi:10.1137/1.9781611972795.91.

[11] Q. Zhu, S. Fang, S. Wu, X. Li, S. Xie, S. Agaian, Dwt-based tensor robust principal component analysis for dynamic high-dimensional signals, Pattern Recognit. 180 (2026) 114295. doi:10.1016/j.patcog.2026.114295.

[12] Y. Luo, X. Zhao, Z. Li, M. K. Ng, D. Meng, Low-rank tensor function representation for multi-dimensional data recovery, IEEE Trans. Pattern Anal. Mach. Intell. 46 (5) (2023) 3351–3369. doi:10.1109/TPAMI.2023.3341688.

[13] H. Wang, J. Peng, W. Qin, J. Wang, D. Meng, Guaranteed tensor recovery fused low-rankness and smoothness, IEEE Trans. Pattern Anal. Mach. Intell. 45 (9) (2023) 10990–11007. doi:10.1109/TPAMI.2023.3259640.

[14] T. Wu, B. Gao, Y. Zhang, J. Xue, Y. Yu, W. Woo, Eficient low average rank tensor recovery with implicit low-rank subspace regularization, IEEE Trans. Multimed. (2026) 1–15doi:10.1109/TMM.2026.3660114.

[15] W. Kong, F. Zhang, W. Qin, J. Wang, Low-tubal-rank tensor recovery with multilayer subspace prior learning, Pattern Recognit. 140 (2023) 109545. doi: 10.1016/j.patcog.2023.109545.

[16] F. L. Hitchcock, Multiple invariants and generalized rank of a p-way matrix or tensor, J. Math. Phys. 7 (1-4) (1928) 39–79. doi:10.1002/sapm19287139.

[17] L. R. Tucker, Some mathematical notes on three-mode factor analysis, Psychometrika 31 (3) (1966) 279–311. doi:10.1007/BF02289464.

[18] I. V. Oseledets, Tensor-train decomposition, SIAM J. Sci. Comput. 33 (5) (2011) 2295–2317. doi:10.1137/090752286.

[19] Q. Zhao, G. Zhou, S. Xie, L. Zhang, A. Cichocki, Tensor ring decomposition, arXiv preprint (2016). doi:10.48550/arXiv.1606.05535.

[20] C. J. Hillar, L.-H. Lim, Most tensor problems are NP-hard, J. ACM 60 (6) (2013) 45. doi:10.1145/2512329.

[21] J. A. Bengua, H. N. Phien, H. D. Tuan, M. N. Do, Eficient tensor completion for color image and video recovery: Low-rank tensor train, IEEE Trans. Image Process. 26 (5) (2017) 2466–2479. doi:10.1109/TIP.2017.2672439.

[22] M. E. Kilmer, C. D. Martin, Factorization strategies for third-order tensors, Linear Algebra Appl. 435 (3) (2011) 641–658. doi:10.1016/j.laa.2010.09. 020.

[23] E. Kernfeld, M. Kilmer, S. Aeron, Tensor–tensor products with invertible linear transforms, Linear Algebra Appl. 485 (2015) 545–570. doi:10.1016/j.laa. 2015.07.021.

[24] M. E. Kilmer, L. Horesh, H. Avron, E. Newman, Tensor-tensor algebra for optimal representation and compression of multiway data, Proc. Natl. Acad. Sci. 118 (28) (2021) e2015851118. doi:10.1073/pnas.2015851118.

[25] S. Ahmadi-Asl, A.-H. Phan, A. Cichocki, A randomized algorithm for tensor singular value decomposition using an arbitrary number of passes, J. Sci. Comput. 98 (1) (2024) 23. doi:10.1007/s10915-023-02411-2.

[26] Y.-Y. Liu, X.-L. Zhao, G. Vivone, Rank-revealing fully-connected tensor network decomposition and its application to tensor completion, Pattern Recognit. 165 (2025) 111610. doi:10.1016/j.patcog.2025.111610.

[27] P. Zhou, C. Lu, J. Feng, Z. Lin, S. Yan, Tensor low-rank representation for data recovery and clustering, IEEE Trans. Pattern Anal. Mach. Intell. 43 (5) (2021) 1718–1732. doi:10.1109/TPAMI.2019.2954874.

[28] J. Lin, T.-Z. Huang, X.-L. Zhao, T.-Y. Ji, Q. Zhao, Tensor robust kernel pca for multidimensional data, IEEE Trans. Neural Netw. Learn. Syst. 36 (2) (2025) 2662–2674. doi:10.1109/TNNLS.2024.3356228.

[29] D.-Z. Cheng, Y. Zhao, An introduction to semi-tensor product of matrices and its applications, World Scientific, 2012.

[30] Z.-R. Chen, S.-W. Vong, Z.-J. Xie, A tensor svd-like decomposition based on the semi-tensor product of tensors, arXiv preprint (2023). doi:10.48550/arXiv. 2301.05937.

[31] K. Batselier, N. Wong, A constructive arbitrary-degree kronecker product decomposition of tensors, Numer. Linear Algebra Appl. 24 (5) (2017) e2097. doi:10.1002/nla.2097.

[32] J. Zhang, A. K. Saibaba, M. E. Kilmer, S. Aeron, A randomized tensor singular value decomposition based on the t-product, Numer. Linear Algebra Appl. 25 (5) (2018) e2179. doi:10.1002/nla.2179.

[33] M. Che, Y. Wei, H. Yan, Eficient algorithms for tucker decomposition via approximate matrix multiplication, Adv. Comput. Math. 51 (3) (2025) 20. doi: 10.1007/s10444-025-10232-0.

[34] O. A. Malik, S. Becker, Fast randomized matrix and tensor interpolative decomposition using countsketch, Adv. Comput. Math. 46 (6) (2020) 76. doi: 10.1007/s10444-020-09816-9.

[35] X. Wang, K. Wang, C. Mo, Subspace-orbit randomized algorithms for low rank approximations of third-order tensors in t-product format, Pattern Recognit. 170 (2026) 112066. doi:10.1016/j.patcog.2025.112066.

[36] Y.-Y. Liu, X.-L. Zhao, Y.-B. Zheng, T.-H. Ma, H. Zhang, Hyperspectral image restoration by tensor fibered rank constrained optimization and plug-and-play regularization, IEEE Trans. Geosci. Remote Sens. 60 (2021) 1–17. doi:10. 1109/TGRS.2020.3045169.

[37] D. Cheng, Matrix and polynomial approach to dynamic control systems, Science Press, 2002.

[38] D. Cheng, H. Qi, A. Xue, A survey on semi-tensor product of matrices, J. Syst. Sci. Complex. 20 (2) (2007) 304–322. doi:10.1007/s11424-007-9027-0.

[39] C. F. Van Loan, N. Pitsianis, Approximation with kronecker products, in: Linear algebra for large scale and real-time applications, Springer, 1993, pp. 293–314. doi:10.1007/978-94-015-8196-7\_17.

[40] W. Qin, H. Wang, F. Zhang, W. Ma, J. Wang, T. Huang, Nonconvex robust highorder tensor completion using randomized low-rank approximation, IEEE Trans. Image Process. 33 (2024) 2835–2850. doi:10.1109/TIP.2024.3385284.

[41] N. Halko, P. G. Martinsson, J. A. Tropp, Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions, SIAM Rev. 53 (2) (2011) 217–288. doi:10.1137/090771806.

[42] S. Ahmadi-Asl, M. G. Asante-Mensah, A. Cichocki, A. H. Phan, I. Oseledets, J. Wang, Fast cross tensor approximation for image and video completion, Signal Process. 213 (2023) 109121. doi:10.1016/j.sigpro.2023.109121.

[43] R. Bellman, Introduction to matrix analysis, SIAM, 1997.

## Supplementary Material of Semi-Tensor Product-Based Multi-Term Randomized T-SVD and Its Visual Applications

This supplementary material accompanies the main paper by providing supporting mathematical preliminaries, full proofs of all theoretical results, complete algorithm pseudocodes, and additional experimental validations. All results presented here are included for completeness and do not afect the core contributions of the main work.

## Appendix A. Auxiliary definitions and properties of Kronecker product

We first review key definitions and properties of the Kronecker product for matrices and tensors. These are standard results from the literature, reproduced here to keep the main text concise and to support the theoretical developments in later sections.

Definition S1.1. [43] $I f \mathbf { A } \in \mathbb { R } ^ { m _ { 1 } \times m _ { 2 } }$ and $\mathbf { B } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } }$ , the Kronecker product between them is defined as

$$
\mathbf { A } \otimes \mathbf { B } = { \left[ \begin{array} { l l l } { \mathbf { A } _ { 1 1 } } & { \cdots } & { \mathbf { A } _ { 1 m _ { 2 } } } \\ { \mathbf { A } _ { 2 1 } } & { \cdots } & { \mathbf { A } _ { 2 m _ { 2 } } } \\ { \vdots } & { \ddots } & { \vdots } \\ { \mathbf { A } _ { m _ { 1 1 } } } & { \cdots } & { \mathbf { A } _ { m _ { 1 } m _ { 2 } } } \end{array} \right] } \otimes \mathbf { B } = { \left[ \begin{array} { l l l l } { \mathbf { A } _ { 1 1 } \mathbf { B } } & { \cdots } & { \mathbf { A } _ { 1 m _ { 2 } } \mathbf { B } } \\ { \mathbf { A } _ { 2 1 } \mathbf { B } } & { \cdots } & { \mathbf { A } _ { 2 m _ { 2 } } \mathbf { B } } \\ { \vdots } & { \ddots } & { \vdots } \\ { \mathbf { A } _ { m _ { 1 } 1 } \mathbf { B } } & { \cdots } & { \mathbf { A } _ { m _ { 1 } m _ { 2 } } \mathbf { B } } \end{array} \right] } \in \mathbb { R } ^ { m _ { 1 1 } \times m _ { 2 } n _ { 2 } } .
$$

Lemma S1.1. [29] Let A B C D be matrices of compatible dimensions, and let α denote a scalar. The following properties of the Kronecker product hold:

(i) $\mathbf { A B } \otimes \mathbf { C D } = ( \mathbf { A } \otimes \mathbf { C } ) ( \mathbf { B } \otimes \mathbf { D } )$

(ii) A ⊗ (B ± C) = (A ⊗ B) ± (A ⊗ C) and (B ± C) ⊗ A = B ⊗ A ± C ⊗ A

(iii) $( \mathbf { A } \otimes \mathbf { B } ) ^ { \top } = \mathbf { A } ^ { \top } \otimes \mathbf { B } ^ { \top }$

(iv) $( \mathbf { A } \otimes \mathbf { B } ) ^ { - 1 } = \mathbf { A } ^ { - 1 } \otimes \mathbf { B } ^ { - 1 }$ , provided A and B are invertible.

(v) $( \mathbf { A } \otimes \mathbf { B } ) \otimes \mathbf { C } = \mathbf { A } \otimes ( \mathbf { B } \otimes \mathbf { C } )$

(vi) (αA) ⊗ B = A ⊗ (αB) = α(A ⊗ B)

Definition S1.2 (Kronecker product of tensors [31]). Consider two p-order tensors $\mathcal { A } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times \cdots \times n _ { p } } a n d \mathcal { B } \in \mathbb { R } ^ { m _ { 1 } \times m _ { 2 } \times \cdots \times m _ { p } }$ . Their tensor Kronecker product $C = \mathcal { A } \otimes \mathcal { B } \in$

R<sup>n1m1×···×npmp</sup> is defined entrywise by

$$
C _ { [ i _ { 1 } j _ { 1 } ] \cdots [ i _ { p } j _ { p } ] } = \mathcal { A } _ { i _ { 1 } \cdots i _ { p } } \mathcal { B } _ { j _ { 1 } \cdots j _ { p } } .
$$

Several fundamental properties of the tensor Kronecker product are summarized below without elaborate derivations.

Lemma S1.2. [31] Let A, B, C be p-order tensors and let α is a scalar. Thefollowing identities hold:

(i) ${ \mathcal { A } } \otimes ( { \mathcal { B } } \pm C ) = { \mathcal { A } } \otimes { \mathcal { B } } \pm { \mathcal { A } } \otimes C ;$

(ii) $\left( \mathcal { B } \pm C \right) \otimes \mathcal { A } = \mathcal { B } \otimes \mathcal { A } \pm C \otimes \mathcal { A } ;$

(iii) $\begin{array} { r } { ( \mathcal { A } \otimes \mathcal { B } ) \otimes C = \mathcal { A } \otimes ( \mathcal { B } \otimes C ) , } \end{array}$

(iv) $( \alpha \mathcal { A } ) \otimes \mathcal { B } = \mathcal { A } \otimes ( \alpha \mathcal { B } ) = \alpha ( \mathcal { A } \otimes \mathcal { B } ) .$

## Appendix B. Proofs of theorems

## Appendix B.1. Proof of Theorem 4.2

To keep the main text focused on algorithmic frameworks and experimental validation, we defer the detailed theoretical derivations to this section. Specifically, we first prove Theorem 4.2, which establishes the multi-term STP-SVD decomposition for matrices and characterizes its truncation error in terms of singular values. We then extend the result to the higher-order tensor setting and prove Theorem 4.3, which presents the MSTP-SVD decomposition with verified orthogonality of factor tensors and a closed-form error bound. Finally, we provide the full proof of Theorem 5.1, which derives the expected reconstruction error bound for the randomized MRSTP-SVD algorithm by decomposing the total error into a deterministic truncation component and a randomized approximation component.

Proof. Based on Lemma 4.2, for any matrix $\mathbf { A } \ \in \ \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } }$ , there exist matrices $\mathbf { B } _ { i } \in \mathbb { R } ^ { m _ { 1 } \times n _ { 1 } }$ and $\mathbf { C } _ { i } ~ \in ~ \mathbb { R } ^ { m _ { 2 } \times n _ { 2 } }$ that matrix A can be decomposed as the sum of k Kronecker product terms plus an error term, i.e.,

$$
\mathbf { A } = \sum _ { i = 1 } ^ { k } \mathbf { B } _ { i } \otimes \mathbf { C } _ { i } + \mathbf { E } _ { k } ,\tag{S2.1}
$$

where $\mathbf { E } _ { k } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } }$ denotes the approximation error matrix, whose squared Frobenius norm satisfies

$$
\| \mathbf { E } _ { k } \| _ { F } ^ { 2 } = \sum _ { i = k + 1 } ^ { \nu } \sigma _ { i } ^ { 2 } ,\tag{S2.2}
$$

where $\sigma _ { k + 1 } \geq \sigma _ { k + 2 } \geq \cdot \cdot \cdot \geq \sigma _ { \nu } \geq 0$ are the singular values of $\mathcal { R } ( { \bf A } ) \in \mathbb { R } ^ { m _ { 1 } n _ { 1 } \times m _ { 2 } n _ { 2 } }$ and $\nu = \operatorname* { m i n } \{ m _ { 1 } n _ { 1 } , m _ { 2 } n _ { 2 } \}$ . Next, computing the SVD of each $\mathbf { B } _ { i }$ yields $\mathbf { B } _ { i } = \mathbf { U } _ { i } \pmb { \Sigma } _ { \mathbf { B } _ { i } } \mathbf { V } _ { i } ^ { \top }$ where $\mathbf { U } _ { i } \in \mathbb { R } ^ { m _ { 1 } \times m _ { 1 } } , \boldsymbol { \Sigma } _ { \mathbf { B } _ { i } } \in \mathbb { R } ^ { m _ { 1 } \times n _ { 1 } } , \mathbf { V } _ { i } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 1 } }$ . Accordingly, (S2.1) can be rewritten as

$$
\mathbf { A } = \sum _ { i = 1 } ^ { k } ( \mathbf { U } _ { i } \pmb { \Sigma } _ { \mathbf { B } _ { i } } \mathbf { V } _ { i } ^ { \top } ) \otimes \mathbf { C } _ { i } + \mathbf { E } _ { k } ,\tag{S2.3}
$$

where $\pmb { \Sigma } _ { \mathbf { B } _ { i } }$ is a diagonal matrix whose diagonal entries are the singular values of $\mathbf { B } _ { i } .$ Let $\sigma _ { i 1 } , \sigma _ { i 2 } , \cdots , \sigma _ { i p }$ with $p = \min \{ m _ { 1 } , n _ { 1 } \}$ denote the singular values of $\mathbf { B } _ { i }$ sorted in decreasing order such that $\sigma _ { i 1 } \geq \sigma _ { i 2 } \geq \cdot \cdot \cdot \geq \sigma _ { i p }$ . Then $\pmb { \Sigma } _ { \mathbf { B } _ { i } } = \mathrm { d i a g } ( \sigma _ { i 1 } , \sigma _ { i 2 } , \cdots , \sigma _ { i p } )$ With Lemma S1.1 and Lemma 2.2, (S2.3) can be reformulated as

$$
\begin{array} { l } { \displaystyle { \mathbf { A } = \sum _ { i = 1 } ^ { k } ( \mathbf { U } _ { i } \Sigma _ { \mathbf { B } _ { i } } \mathbf { V } _ { i } ^ { \top } ) \otimes ( \mathbf { I } _ { m _ { 2 } } \mathbf { C } _ { i } \mathbf { I } _ { n _ { 2 } } ) + \mathbf { E } _ { k } } } \\ { \displaystyle \quad = \sum _ { i = 1 } ^ { k } ( \mathbf { U } _ { i } \otimes \mathbf { I } _ { m _ { 2 } } ) ( \Sigma _ { \mathbf { B } _ { i } } \otimes \mathbf { C } _ { i } ) ( { \mathbf { V } _ { i } ^ { \top } \otimes \mathbf { I } _ { n _ { 2 } } } ) + \mathbf { E } _ { k } } \\ { \displaystyle \quad = \sum _ { i = 1 } ^ { k } \mathbf { U } _ { i } \ltimes \Sigma _ { i } \ltimes { \mathbf { V } _ { i } ^ { \top } + \mathbf { E } _ { k } } , } \end{array}
$$

where $\pmb { \Sigma } _ { i } = \pmb { \Sigma } _ { \mathbf { B } _ { i } } \otimes \mathbf { C } _ { i } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } }$ is a block-diagonal matrix whose diagonal blocks are given by $\mathbf { S } _ { i 1 } = \sigma _ { i 1 } \mathbf { C } _ { i } , \mathbf { S } _ { i 2 } = \sigma _ { i 2 } \mathbf { C } _ { i } , \cdots , \mathbf { S } _ { i p } = \sigma _ { i p } \mathbf { C } _ { i }$ . Since $\sigma _ { i j } ( j = 1 , 2 , \cdots , p )$ are non-negative scalars, norm properties give $\| \sigma _ { i j } \mathbf { C } _ { i } \| _ { F } = \sigma _ { i j } \| \mathbf { C } _ { i } \| _ { F }$ . Consequently,

$$
\lVert \sigma _ { i 1 } \mathbf { C } _ { i } \rVert _ { F } \geq \lVert \sigma _ { i 2 } \mathbf { C } _ { i } \rVert _ { F } \geq \cdots \geq \lVert \sigma _ { i p } \mathbf { C } _ { i } \rVert _ { F } ,
$$

or equivalently

$$
\Vert \mathbf { S } _ { i 1 } \Vert _ { F } \geq \Vert \mathbf { S } _ { i 2 } \Vert _ { F } \geq \cdots \geq \Vert \mathbf { S } _ { i p } \Vert _ { F } .
$$

## Appendix B.2. Proof of Theorem 4.3

Proof. We first assume that each frontal slice $\bar { \mathcal { A } } ^ { ( j ) } \left( j = 1 , 2 , \ldots , l \right)$ admits the multiterm STP-SVD decomposition derived in Theorem 4.2, i.e.,

$$
\bar { \mathcal { A } } ^ { ( j ) } = \sum _ { i = 1 } ^ { k } \bar { \mathcal { U } } _ { i } ^ { ( j ) } \ltimes \bar { S } _ { i } ^ { ( j ) } \ltimes ( \bar { \mathcal { V } } _ { i } ^ { ( j ) } ) ^ { \top } + \bar { \mathcal { E } } _ { k } ^ { ~ ( j ) } .
$$

Based on the definition of fold and bdiag operators, we have

$$
\begin{array} { r l } & { \mathcal { \beta } _ { 1 } - \mathrm { c o l i a l e l ~ } \mathrm { E q } _ { 2 } \langle \hat { Z } , \hat { Z } \rangle } \\ & { \quad = \mathrm { ~ E q a l l ~ } | \begin{array} { l } { \frac { 1 } { \mathcal { N } ^ { 2 } } \mathrm { t r } ^ { 2 } } \\ { \mathcal { \beta } ^ { 2 } } \\ { \quad - \mathrm { c h a l l ~ } } \\ { \quad - \mathrm { c h a l l ~ } } \\ { \quad - \mathrm { c h a r l e l ~ } } \end{array} | } \\ & { \quad \quad - \mathrm { t e n d } [ \begin{array} { l } { \frac { 1 } { \mathcal { N } ^ { 3 } } } \\ { \vdots } \\ { \mathcal { N } ^ { 5 } } \\ { \vdots } \\ { \mathcal { N } ^ { 6 } } \end{array} \mathrm { t r a l ~ } { \mathcal { N } } \mathrm { t a b } ( \hat { Z } ) \cdot \mathrm { b e ~ h a l g a } ( \hat { V } ^ { \top } ) \cdot \mathrm { b e ~ h a r g } ( \hat { V } ^ { \top } ) \cdot \mathrm { b e ~ h a r g } ( \hat { V } ) } \\ { \quad - \mathrm { t e n d } [ \begin{array} { l } { \frac { 1 } { \mathcal { N } ^ { 3 } } \mathrm { t r } ( \mathrm { d i a g } ( \hat { Z } ) \mathrm { t s } ) } \\ { \vdots } \\ { \mathcal { N } ^ { 6 } } \end{array} ] \mathrm { t r } ( \hat { Z } ) \cdot \mathrm { b e l a g } ( \hat { Z } ) } \\ & { \quad \quad - \mathrm { t e n d } [ \begin{array} { l } { \frac { 1 } { \mathcal { N } ^ { 3 } } \mathrm { t r } ( \mathrm { d i a g } ( \hat { Z } ) \mathrm { t s } ) } \\ { \vdots } \\ { \mathcal { N } ^ { 5 } } \end{array} ] \cdot \mathrm { b e l a g } ( \hat { Z } ) \cdot \mathrm { b e l a g } ( \hat { V } ) \cdot \mathrm { b e l a g } ( \hat { V } ^ { \top } ) \cdot \mathrm { b e l a g } ( \hat { V } ) \cdot \mathrm { b e l a g } ( \hat { Z } ) } \\ &  \quad \quad - \mathrm { t e n d } [ \begin{array} { l }   \end{array} \end{array}
$$

Then,

$$
\mathcal { A } = L ^ { - 1 } ( \bar { \mathcal { A } } ) = \sum _ { i = 1 } ^ { k } \mathcal { U } _ { i } \ltimes _ { L } S _ { i } \ltimes _ { L } \mathcal { V } _ { i } ^ { \top } + \mathcal { E } _ { k } .
$$

Since $( \bar { \mathcal { U } } _ { i } ^ { ( j ) } ) ^ { \top }$ is orthogonal, we have,

$$
\begin{array} { r l } { \| A \mathcal { H } _ { \sigma } ^ { * } ( x , q , t ) \| _ { 2 } = \| A \mathcal { H } _ { \sigma } ^ { * } ( x , q ) \| } \\ & { = \operatorname* { i n d } | \operatorname* { s i n d } | \operatorname* { s i n d } \widehat { \mathcal { H } } | ^ { 2 } \rangle \times \operatorname { v i n d } \widehat { \mathcal { H } } | } \\ & { \qquad [ \bigoplus \widehat { \mathcal { H } } ^ { * , \gamma } \times \widehat { \mathcal { H } } ^ { 0 , \gamma }  } \\ & { = \operatorname* { i n d } | } \\ & { \qquad - \operatorname { i n d } | \operatorname* { d i v } ^ { \mathcal { H } }   } \\ & { \qquad  ( \widehat { \mathcal { H } } _ { \sigma } ^ { \mathcal { H } , \gamma } ) \times \widehat { \mathcal { H } } _ { \sigma } ^ { \mathcal { H } , \gamma }   } \\ & { = \operatorname* { i n d } | \operatorname* { c i n d } ^ { \mathcal { H } } ( x , q )   } \\ & { \qquad   \langle \widehat { \mathcal { H } } _ { \sigma } ^ { \mathcal { H } , \gamma } \times \widehat { \mathcal { H } } _ { \sigma } ^ { \mathcal { H } , \gamma }   } \\ & { \qquad - \operatorname { i n d } | [ \operatorname* { d i v } ^ { \mathcal { H } , \gamma } \times \widehat { \mathcal { H } } _ { \sigma } ^ { \mathcal { H } , \gamma }    } \\ & { \qquad   \langle \widehat { \mathcal { H } } _ { \sigma } ^ { \mathcal { H } , \gamma } \times \widehat { \mathcal { H } } _ { \sigma } ^ { \mathcal { H } , \gamma } ] | } \\ & { = \operatorname* { m a x } _ { \sigma , \epsilon \in \mathcal { H } , \epsilon } |  } \\ & { \qquad  \langle \widehat { \mathcal { H } } _ { \sigma } ^ { \mathcal { H } , \gamma } \times \widehat { \mathcal { H } } _ { \sigma } ^ { \mathcal { H } , \gamma }   } \\ &  = \operatorname* { m a x } _   \end{array}
$$

By analogous arguments, $L ( \mathcal { U } _ { i } * _ { L } \mathcal { U } _ { i } ^ { \top } ) = \bar { \mathcal { I } } _ { m _ { 1 } m _ { 1 } l }$ , which verifies that $\mathcal { U } _ { i }$ is an orthogonal tensor. Following identical reasoning, $\mathcal { N } _ { i }$ is also orthogonal.

Suppose that the transform matrix L satisfies ${ \bf L } ^ { \mathrm { H } } { \bf L } = { \bf L } { \bf L } ^ { \mathrm { H } } = \rho { \bf I } _ { l }$ and ${ \bf L } ^ { - 1 } = { \bf L } ^ { \mathrm { H } } / \rho$ for some constant $\rho > 0$ . For the approximation error tensor $\mathcal { E } _ { k }$ , its squared Frobenius norm satisfies

$$
\lVert \bar { \boldsymbol { E } } _ { k } \rVert _ { F } ^ { 2 } = \left( \frac { 1 } { \sqrt { \rho } } \lVert \mathrm { b d i a g } ( \bar { \boldsymbol { \mathcal { E } } } _ { k } ) \rVert _ { F } \right) ^ { 2 } = \frac { 1 } { \rho } \sum _ { j = 1 } ^ { l } \lVert \bar { \boldsymbol { E } } _ { k } ^ { ( j ) } \rVert _ { F } ^ { 2 } = \frac { 1 } { \rho } \sum _ { j = 1 } ^ { l } \sum _ { i = k + 1 } ^ { \nu } ( \hat { \sigma } _ { i } ^ { ( j ) } ) ^ { 2 } ,\tag{S2.4}
$$

where the last equality is from (S2.2), and $\hat { \sigma } _ { i } ^ { ( j ) }$ is the i-th singular value of $\mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } )$ and $\nu = \operatorname* { m i n } \{ m _ { 1 } n _ { 1 } , m _ { 2 } n _ { 2 } \}$ □

## Appendix B.3. Proof of Theorem 5.1

Proof. Let $\mathcal { A } _ { \mathrm { M S T P } }$ denotes the exact MSTP-SVD approximation of tensor A, which obeys the deterministic error bound from Theorem 4.2:

$$
\left. \mathcal { A } - \mathcal { A } _ { \mathrm { M S T P } } \right. _ { F } ^ { 2 } = \frac { 1 } { \rho } \sum _ { j = 1 } ^ { l } \sum _ { i = k + 1 } ^ { \nu } ( \hat { \sigma } _ { i } ^ { ( j ) } ) ^ { 2 } .\tag{S2.5}
$$

The tensor $\tilde { \mathcal { A } }$ produced by the MRSTP-SVD algorithm is built upon the exact deterministic MSTP-SVD approximation $\mathcal { A } _ { \mathrm { M S T P } }$ . To cut the heavy computational cost of full SVD on each rearranged matrix $\mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } )$ , we substitute the exact SVD routine with randomized projection and power iteration. Given this two-stage construction flow, we can naturally decompose the overall reconstruction error into two additive residual parts as

$$
\mathcal { A } - \tilde { \mathcal { A } } = \left( \mathcal { A } - \mathcal { A } _ { \mathrm { M S T P } } \right) + \left( \mathcal { A } _ { \mathrm { M S T P } } - \tilde { \mathcal { A } } \right) ,\tag{S2.6}
$$

where the first term corresponds to fixed truncation residual of multi-term decomposition, and the second term represents the error induced by randomized subspace approximation. As given in Theorem 2 of [40], the error bound for the standalone randomized T-SVD approximation satisfies

$$
\mathbb { E } \left. \mathcal { R } _ { \mathrm { M S T P } } - \tilde { \mathcal { A } } \right. _ { F } ^ { 2 } \leq \frac { 1 } { \rho } \sum _ { j = 1 } ^ { l } \left( 1 + \frac { k } { p - 1 } ( \tau _ { k } ^ { ( j ) } ) ^ { 4 q } \right) \left( \sum _ { i = k + 1 } ^ { \nu } ( \hat { \sigma } _ { i } ^ { ( j ) } ) ^ { 2 } \right) .\tag{S2.7}
$$

Combining (S2.5), (S2.6) and (S2.7), we apply the parallelogram identity for the Frobenius norm and linearity of expectation to expand the total expected error:

$$
\begin{array} { r l } { \mathbb { E } \left\| \mathcal { A } - \tilde { \mathcal { A } } \right\| _ { F } ^ { 2 } = \mathbb { E } \left\| ( \mathcal { A } - \mathcal { A } _ { \mathrm { M S T P } } ) + \left( \mathcal { A } _ { \mathrm { M S T P } } - \tilde { \mathcal { A } } \right) \right\| _ { F } ^ { 2 } } & { } \\ { \leq 2 \mathbb { E } \left\| \mathcal { A } - \mathcal { A } _ { \mathrm { M S T P } } \right\| _ { F } ^ { 2 } + 2 \mathbb { E } \left\| \mathcal { A } _ { \mathrm { M S T P } } - \tilde { \mathcal { A } } \right\| _ { F } ^ { 2 } } & { } \\ { = \displaystyle \frac { 2 } { \rho } \sum _ { j = 1 } ^ { l } \left[ \left( 2 + \frac { k } { p - 1 } \left( \tau _ { k } ^ { ( j ) } \right) ^ { 4 q } \right) \left( { \sum _ { i = k + 1 } ^ { \nu } \left( \hat { \sigma } _ { i } ^ { ( j ) } \right) ^ { 2 } } \right) \right] . } \end{array}\tag{S2.8}
$$

## Appendix C. Pseudocodes for all truncated variants

For completeness and ease of reproducibility, we collect the full pseudocodes for all truncated variants of the STP-SVD framework. Specifically, we detail the truncated STP-SVD for matrices, the truncated multi-term STP-SVD for matrices, the truncated MSTP-SVD for tensors, and the truncated randomized MRSTP-SVD for tensors.

Algorithm 6: Truncated STP-SVD of matrices [30]   
Input: $\overline { { \mathbf { A } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } } } }$ , truncated parameter r.   
Output: U Σ V.   
1 Calculate matrices $\mathbf { B } \in \mathbb { R } ^ { m _ { 1 } \times n _ { 1 } }$ and $\mathbf { C } \in \mathbb { R } ^ { m _ { 2 } \times n _ { 2 } }$ via Lemma 4.1, such that   
$\mathbf { A } \approx \mathbf { B } \otimes \mathbf { C } ;$   
2 Perform truncated SVD on B: $[ \mathbf { U _ { B } } , { \boldsymbol \Sigma _ { B } } , \mathbf { V _ { B } } ] = \operatorname { s v d s } ( \mathbf { B } , r ) ;$   
3 return $\mathbf { U } = \mathbf { U _ { B } } , \boldsymbol { \Sigma } = \boldsymbol { \Sigma } _ { \mathbf { B } } \otimes \mathbf { C } , \mathbf { V } = \mathbf { V _ { B } } .$

Algorithm 7: Truncated multi-term STP-SVD of matrices   
Input: $\overline { { \mathbf { A } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } } } }$ , the number of terms k, truncated parameter r.   
Output: $\mathbf { U } _ { i } , \pmb { \Sigma } _ { i } , \mathbf { V } _ { i } .$   
1 Calculate matrices $\mathbf { B } _ { i } \in \mathbb { R } ^ { m _ { 1 } \times n _ { 1 } }$ and $\mathbf { C } _ { i } \in \mathbb { R } ^ { m _ { 2 } \times n _ { 2 } } ( i = 1 , \dots , k )$ via Lemma 4.2,   
such that $\begin{array} { r } { \mathbf { A } \approx \sum _ { i = 1 } ^ { k } \mathbf { B } _ { i } \otimes \mathbf { C } _ { i } ; } \end{array}$   
2 for i = 1 to k do   
3 Perform truncated SVD on $\mathbf { B } _ { i } \colon [ \mathbf { U } _ { i } , \boldsymbol { \Sigma } _ { \mathbf { B } _ { i } } , \mathbf { V } _ { i } ] = \operatorname { s v d s } ( \mathbf { B } _ { i } , r ) ;$   
4 $\pmb { \Sigma } _ { i } = \pmb { \Sigma } _ { \mathbf { B } _ { i } } \otimes \mathbf { C } _ { i } ;$   
5 return $\mathbf { U } _ { i } , \pmb { \Sigma } _ { i } , \mathbf { V } _ { i } .$

Algorithm 8: Truncated MSTP-SVD method of tensors (TMSTP-SVD)   
Input: $\mathcal { A } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } \times l }$ , the number of terms k, truncated rank matrix R.   
Output: $\mathcal { U } _ { i } , S _ { i } , \mathcal { V } _ { i } .$   
1 Obtain A<sup>¯</sup> by applying an invertible linear transform L on A;   
2 for j = 1 to l do   
3 Approximate the j-th frontal slice by Lemma 4.2: $\begin{array} { r } { \bar { \mathcal { A } } ^ { ( j ) } \approx \sum _ { i = 1 } ^ { k } \mathbf { B } _ { i } ^ { ( j ) } \otimes \mathbf { C } _ { i } ^ { ( j ) } , } \end{array}$   
4 for i = 1 to k do   
5 Compute the truncated SVD of $\mathbf { B } _ { i } ^ { ( j ) } \colon [ \mathbf { U } _ { i } ^ { ( j ) } , \boldsymbol { \Sigma } _ { \mathbf { B } _ { i } } ^ { ( j ) } , \mathbf { V } _ { i } ^ { ( j ) } ] = \operatorname { s v d s } \big ( \mathbf { B } _ { i } ^ { ( j ) } , R _ { i j } \big ) ;$   
6 $\pmb { \Sigma } _ { i } ^ { ( j ) } = \pmb { \Sigma } _ { \pmb { \mathrm { B } } _ { i } } ^ { ( j ) } \otimes \mathbf { C } _ { i } ^ { ( j ) } ;$   
7 Store $\mathbf { U } _ { i } ^ { ( j ) } , \pmb { \Sigma } _ { i } ^ { ( j ) } , \mathbf { V } _ { i } ^ { ( j ) }$ into $\mathcal { U } _ { i } , S _ { i } , \mathcal { V } _ { i } ,$ , respectively;   
8 return $\mathcal { U } _ { i } = L ^ { - 1 } ( \mathcal { U } _ { i } ) , S _ { i } = L ^ { - 1 } ( S _ { i } ) , \mathcal { V } _ { i } = L ^ { - 1 } ( \mathcal { V } _ { i } ) .$

Algorithm 9: Truncated MRSTP-SVD method of tensors (TMRSTP-SVD)   
Input: $\begin{array} { r } { \overline { { \mathcal { A } \in \mathbb { R } ^ { m _ { 1 } m _ { 2 } \times n _ { 1 } n _ { 2 } \times l } } } , } \end{array}$ , the number of terms $k ,$ oversampling parameter   
$s \geq 0 ,$ iteration parameter $q \geq 0 ,$ truncated rank matrix R.   
Output: $\mathcal { U } _ { i } , S _ { i } , \mathcal { V } _ { i } .$   
1 Generate a Gaussian random tensor $\mathcal { G } \in \mathbb { R } ^ { n _ { 1 } n _ { 2 } \times ( k + s ) \times l } ;$   
2 Compute $\bar { \mathcal { A } } = L ( \mathcal { A } )$ and $\bar { \mathcal { G } } = L ( \mathcal { G } ) ;$   
3 for $j = 1$ to l do   
4 Obtain $\mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } )$ by reorganizing the blocks of $\bar { \mathcal { A } } ^ { ( j ) } ;$   
5 Compute ${ \bf Y } = ( \mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } ) \mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } ) ^ { \top } ) ^ { q } \mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } ) \bar { \mathcal { G } } ^ { ( j ) } ;$   
6 Compute thin-QR factorization $\mathbf { Y } = \mathbf { Q } _ { j } \mathbf { R } ;$   
7 Compute $\mathbf { B } = \mathbf { Q } _ { j } ^ { \top } \mathcal { R } ( \bar { \mathcal { A } } ^ { ( j ) } ) ;$   
8 Compute the SVD of B: $\mathbf B = \mathbf { U } \mathbf { S } \mathbf { V } ^ { \top }$   
9 Form $\mathbf { U } _ { k } , \mathbf { V } _ { k } , \mathbf { S } _ { k }$ by truncating $\mathbf { Q } _ { j } \mathbf { U }$ , V, S with $k ;$   
10 for $i = 1$ to k do   
11 vec $( \mathbf { B } _ { i } ^ { ( j ) } ) = \sqrt { \mathbf { S } _ { k } ( i , i ) } \mathbf { U } _ { k } ( : , i ) ,$ vec $( \mathbf { C } _ { i } ^ { ( j ) } ) = \sqrt { \mathbf { S } _ { k } ( i , i ) } \mathbf { V } _ { k } ( : , i )$ such that   
$\begin{array} { r } { \bar { \mathcal { A } } ^ { ( j ) } \approx \sum _ { i = 1 } ^ { k } \mathbf { B } _ { i } ^ { ( j ) } \otimes \mathbf { C } _ { i } ^ { ( j ) } ; } \end{array}$   
12 Compute the truncated SVD of $\mathbf { B } _ { i } ^ { ( j ) } \colon [ \mathbf { U } _ { i } ^ { ( j ) } , \boldsymbol { \Sigma } _ { \mathbf { B } _ { i } } ^ { ( j ) } , \mathbf { V } _ { i } ^ { ( j ) } ] = \operatorname { s v d s } \big ( \mathbf { B } _ { i } ^ { ( j ) } , R _ { i j } \big ) ;$   
13 $\pmb { \Sigma } _ { i } ^ { ( j ) } = \pmb { \Sigma } _ { \pmb { \mathrm { B } } _ { i } } ^ { ( j ) } \otimes \mathbf { C } _ { i } ^ { ( j ) } ;$   
14 Store $\mathbf { U } _ { i } ^ { ( j ) } , \pmb { \Sigma } _ { i } ^ { ( j ) } , \mathbf { V } _ { i } ^ { ( j ) }$ into $\mathcal { U } _ { i } , S _ { i } , \mathcal { V } _ { i } ,$ respectively;   
15 return $\mathcal { U } _ { i } = L ^ { - 1 } ( \mathcal { U } _ { i } ) , S _ { i } = L ^ { - 1 } ( S _ { i } ) , \mathcal { V } _ { i } = L ^ { - 1 } ( \mathcal { V } _ { i } ) .$

## Appendix D. Supplementary experimental results

This section presents supplementary experimental results to further validate the efectiveness and eficiency of the proposed method for compression and completion tasks. We extend the empirical evaluation in the main text to additional benchmark datasets and provide extra qualitative visual comparisons. All results presented here complement the conclusions of the main manuscript and deliver a more comprehensive empirical verification of our approach.

Fig. S4.1presents the impact of three invertible linear transforms (DFT, DCT, and ROT) on the compression performance of both deterministic MSTP-SVD and randomized MRSTP-SVD on the Night and Fruit test images. Consistent with the observations in the main text, all three transforms yield comparable reconstruction quality in terms of PSNR and SSIM, while DFT consistently achieves the lowest computational overhead across both test images. These supplementary results further validate the rationality of selecting DFT as the default transform in all subsequent experiments.

![](images/9ced7d3d783975cc83f0906253dec106200187f6e2eae859416dbf994f652422.jpg)  
Fig. S4.1: Quantitative metrics (PSNR, SSIM, runtime) of MSTP-SVD and randomized MRSTP-SVD for image compression under invertible transforms (DFT, DCT, ROT). Top: Night; Bottom: Fruit.

Fig. S4.2 provides supplementary visual reconstruction comparisons and corresponding quantitative PSNR and runtime measurements on two additional representative test images, covering all competing baselines and the proposed multi-term, truncated, and randomized variants. The block partition sizes and truncation rank settings are identical to those in the main text to ensure fair comparison. Quantitatively, MSTP-SVD (k=3) achieves 7 9 dB and 3 1 dB PSNR improvements over the single-term STP-SVD on the two images, respectively, and outperforms the TT-SVD baseline by 5 8 − 7 3 dB. The randomized MRSTP-SVD yields comparable reconstruction fidelity to its deterministic counterpart, with PSNR loss less than 0 08 dB, while reducing the runtime by approximately 22%. It can be observed that the multiterm schemes consistently outperform single-term baselines and preserve finer local structural details, whereas baseline methods tend to produce over-smoothed outputs. These results corroborate the efectiveness and generalization capability of the proposed methods across diverse image content.

![](images/37ef71f3a6d32d7c5ff17a2bdf5ad739a82037544e4f708235f91515d20d3227.jpg)  
Fig. S4.2: Supplementary comparisons of reconstruction performance, PSNR, and runtime for competing methods and the proposed approach on two additional representative test images.

Fig. S4.3 shows supplementary visual reconstruction examples and quantitative PSNR-SSIM comparisons on randomly sampled frames from two additional test video sequences, including all competing baselines as well as the original and truncated variants of the proposed methods. All experimental configurations, including block partition sizes, truncation ranks, and randomized hyperparameters, follow the settings used in the main text. Quantitative results demonstrate that MSTP-SVD (k=3) im-

proves PSNR by 2.9 dB and 5.2 dB over single-term STP-SVD on the two sequences, respectively, and achieves around 8 dB of PSNR gain over the TT-SVD baseline for both sequences. The randomized MRSTP-SVD reduces average runtime by 15%-25% with negligible quality degradation. Visually, the proposed methods recover richer textures and finer details than baseline approaches. These supplementary video samples further verify the robustness of the proposed framework across diferent video content.  
![](images/78901d78fe7af6139eed96c7fe0b4364a8f4c2c842d8099d6241d801545418de.jpg)  
Fig. S4.3: Supplementary visual reconstruction examples and quantitative PSNR-SSIM comparisons between competing baselines and our approaches (original and truncated variants) on randomly sampled frames from two additional test video sequences.

Fig. S4.4 reports supplementary video completion results on two additional test sequences under 70% randomly missing pixels, including frame-wise PSNR-SSIM curves, total runtime statistics, and visual reconstruction examples on sampled frames. The experiments adopt the same iterative completion framework as the main text, with DFT as the default transform and 70% pixel missing ratio. Quantitative results show that the proposed methods outperform baseline competitors in reconstruction accuracy. Compared with the deterministic TMSTP-SVD, the randomized TMRSTP-SVD substantially cuts down the overall completion runtime, while TMRSTP-SVD incurs nearly negligible reconstruction quality loss. Visually, the proposed methods recover finer textures and introduce fewer visible artifacts in missing regions. These supplementary results further confirm the generalization ability of the proposed methods for video completion tasks.

![](images/fa41f38908ce02d325cf319b5a21951b4efd68f0efcf41673c122401756d3db5.jpg)

![](images/bb3aec1e4131ace179e5d930525835f32d0f45c464f8bd631090a951c424fb06.jpg)

![](images/5c2bf76ceaa553682a1b4056d50cda40d2cf2dff9c70246921e981fdc21e3569.jpg)

![](images/e6c479be667d40c1cb9e2c9342dd54ff9787e1b2671d3d62a93a60424aced3f1.jpg)  
Fig. S4.4: Supplementary objective and visual comparisons of video recovery methods on two additional test sequences, including PSNR-SSIM curves, runtime measurements, and reconstruction results under 70% missing pixels.