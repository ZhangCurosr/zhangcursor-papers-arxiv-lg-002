# Neural Dimensional Analysis: Foundations of Dimensionless and Scale-Consistent Computation

Bum Jun Kim<sup>∗</sup>

Makoto Kawano Yusuke Iwasawa The University of Tokyo

Yutaka Matsuo

## Abstract

Modern Transformers rely on many scalecontrol mechanisms, including normalization, residual scaling, attention temperatures, loss weighting, and adaptive updates. To explain how these mechanisms maintain scale consistency, we introduce Neural Dimensional Analysis (NDA), a unified framework based on dimensional consistency. We define neural units that specify how tensors rescale and which quantities can be added, just as physical units do in physics. Our rules match residual units, nondimensionalize attention and gate inputs, combine losses across units, and convert gradient units into parameter units. We implement NDA through dimensionless Unit Violation Indices (UVI) for diagnosing scale mismatches within Transformers, a unit-consistent multi-loss objective, and the Gauge-Canonical Transformer (GCT) with normalized coordinates and orthogonalized residual additions. In language-model experi ments spanning 51M to 1B parameters, GCT remains stable during training across a broad sixteenfold learning-rate range, outperforming Pre-LN, LLaMA-style, and QK-Norm in 124M validation loss at tuned learning rates. GCT’s normalized residual stream also shows improved robustness to quantization. At 1B parameters, calibrated per-tensor int8 quantization increases GCT’s validation loss by only 0.009 nats, compared with up to 5.51 nats for the baselines. Vision and physicsinformed neural network experiments further confirm NDA’s benefits beyond language modeling. These results establish dimensional consistency as a neural network design principle.

## 1 INTRODUCTION

Deep learning employs a broad family of mechanisms for controlling internal scale. Batch normalization, layer normalization (LayerNorm), and root-mean-square normalization (RMSNorm) normalize activations (Iofe and Szegedy, 2015; Ba et al., 2016; Zhang and Sennrich, 2019). DeepNet, ReZero, LayerScale, and Fixup control the scale of residual additions (Wang et al., 2024; Bachlechner et al., 2021; Touvron et al., 2021b; Zhang et al., 2019). Attention scaling and query–key normalization (QK-Norm) regulate softmax inputs (Vaswani et al., 2017; Henry et al., 2020; Dehghani et al., 2023; Team, 2024). Loss-balancing methods and adaptive optimizers manage scale imbalances in objectives and parameter updates (Kendall et al., 2018; Chen et al., 2018; Wang et al., 2021; Kingma and Ba, 2015; You et al., 2020). These scale-control mechanisms play a critical role in improving training stability and performance (Xiong et al., 2020; Zhai et al., 2023; Wortsman et al., 2024). All of these mechanisms manage how quantities of diferent scales interact within a neural network (Figure 1).

We interpret these scale-control mechanisms through dimensional consistency. In physics, terms added together must have compatible units, and the arguments of exp and log must be dimensionless. A physical unit, such as a centimeter or a meter, provides a reference scale for expressing a measured quantity. Inspired by physical units and dimensions, we define a neural unit as the rule governing how an internal tensor scales when the computation graph is rescaled. Neural units describe internal scaling behavior even when the data have no physical units. In terms of neural units, each scale-control mechanism in deep learning becomes a unit operation. Normalization removes units, residual branches restore stream units, attention temperatures convert logits to dimensionless quantities, loss weights convert loss units, and optimizers map gradient units back to parameter units. Building on this view, we propose Neural Dimensional Analysis (NDA), a framework that unifies these unit operations under one set of local typing rules.

![](images/003e7b01d1bca7ed0a171529152f00a3b6d41b392e0196fe1854d215cd49a3fb.jpg)  
Figure 1: NDA at a glance. (Left) Scale-control mechanisms in deep learning, including normalization, residual scaling, attention temperature, loss weighting, and adaptive updates, manage scale at diferent interfaces of a Transformer. Unit Violation Indices (UVI) flag deviations from the intended scale regime, locating the interface where a scale mismatch arises. (Middle) NDA expresses these mechanisms through unit-consistent operations and a factorization into scales and canonical coordinates. (Right) The Gauge-Canonical Transformer (GCT) builds the whole network on this factorization. Every block computes on dimensionless coordinates (blue), ensuring matching units at every interface, including attention and multilayer perceptron (MLP) branches. The scale (gold) is reattached only when an output needs units.

Positioning. Prior work on scale control has addressed individual aspects of scale for specific purposes. For example, units-equivariant learning handles the physical units of input and output data (Villar et al., 2023), maximal-update parameterization (µP) prescribes parameterization and update scales for hyperparameter transfer (Yang and Hu, 2021; Yang et al., 2021), unit scaling chooses scales for low-precision training (Blake et al., 2023), and the normalized Transformer (nGPT) constrains activations and weights to the unit hypersphere (Loshchilov et al., 2025). In contrast, our NDA framework ofers a unified view of scale control over the whole computation graph, covering activations, objectives, gradients, and optimizers, and additionally provides diagnostics and local corrections. App. C.6 develops these connections.

## Contributions. We make three contributions:

1. A graph-level unit calculus covering activations, objectives, gradients, and updates, together with a theoretical analysis. This calculus links forward computation and backpropagation through a single set of unit rules (Prop. 1). To guide unit-consistent network design and training, we prove theorems on loss weighting, optimizer updates, canonical factorization, and quantization error (App. E).

2. Diagnostics and designs derived from the calculus. To make the unit rules testable in trained networks, we introduce dimensionless UVI and a weight-space audit with per-block rescaling predictions (Prop. 6). GCT computes every block on dimensionless coordinates and uses an orthogonal projection to control scale growth during residual updates. The calculus also provides local scale corrections for existing architectures, along with unit-consistent training objectives and parameter updates.

3. Language-model experiments from 51M to 1B parameters under matched recipes. GCT training remains stable across a 16× learning-rate range, and the GCT hyperparameters transfer unchanged to other GCT models up to 6× as deep, with clean validation loss comparable to strong baselines (Figure 2; Tables 1 and 4). The orthogonal projection also improves nGPT in the depth comparison, reducing nGPT’s loss by 0.275 nats at L = 48. GCT’s normalized residual stream also improves robustness to low-precision storage by keeping token scales uniform. At 1B, calibrated per-tensor 8-bit integer (int8) quantization of this stream increases GCT’s validation loss by only 0.009 nats, compared with 5.51 nats for the worst baseline (Table 2). Vision and physics-informed neural network (PINN) studies validate the same principles beyond language modeling (App. I).

## 2 NEURAL UNIT CONSISTENCY

This section formulates local rules for managing scale on a neural network’s computation graph. The rules specify which quantities can be added and which inputs must be dimensionless. The rules also determine how loss weights must compensate for changes in loss units and how gradients must rescale when activations or parameters are rescaled. These rules provide the basis for the diagnostics in Sec. 3 and the architecture in Sec. 4.

## 2.1 Units as transformation laws

Let $\mathcal { G }$ be a computation graph whose node v carries a tensor $x _ { v }$ in a finite-dimensional real inner-product space $V _ { v }$ . We declare a rescaling group $G : = ( \mathbb { R } _ { + } ) ^ { K }$ ， with one factor for each independent internal scale degree of freedom. A neural unit of node v is a representation

$$
\rho _ { v } : G \to G L ( V _ { v } ) , \qquad x _ { v } \mapsto \rho _ { v } ( g ) x _ { v } .\tag{1}
$$

A tensor is dimensionless if and only if $\rho _ { v } \equiv I .$ . The scalar units considered here are the continuous characters $\begin{array} { r } { \chi _ { u } ( g ) : = \prod _ { k = 1 } ^ { K } g _ { k } ^ { u _ { k } } , u \in \mathbb { R } ^ { K } } \end{array}$ , with $\rho _ { v } = \chi _ { u _ { v } } I .$ We then write $u ( x ) : = \chi _ { u _ { x } }$ and use the multiplicative notation $u ( x y ) = u ( x ) u ( y )$

The unit type $u ( x )$ specifies how x transforms under $G ,$ whereas the empirical scale $s _ { x }$ measures magnitude and shares that type, making $x / s _ { x }$ dimensionless. The root-mean-square (RMS) value is one such scale on a shared-unit block; mixed-unit tensors require separate scales or a unit-compatible metric (App. F.7).

For an operation $y : = f ( x _ { 1 } , \ldots , x _ { n } )$ , the declared transformation laws are $\rho _ { 1 } , \ldots , \rho _ { n }$ for the input tensors and $\rho _ { y }$ for the output tensor. We call the operation unitconsistent when the following equivariance identity holds for every $g \in G$

$$
f ( \rho _ { 1 } ( g ) x _ { 1 } , \ldots , \rho _ { n } ( g ) x _ { n } ) = \rho _ { y } ( g ) f ( x _ { 1 } , \ldots , x _ { n } ) .\tag{2}
$$

In words, rescaling the inputs before the operation must give the same result as rescaling the output afterward.

Parameters follow the same rule. In $y : = f _ { \phi } ( x )$ , we treat ϕ as another input and declare the parameter transformation $\phi \mapsto \rho _ { \phi } ( g ) \phi$ . App. F.7 details linear maps and explicit scale converters.

We also need gradient diagnostics that can be compared across blocks with diferent units. The next proposition shows that the forward unit declarations fix how every backpropagated gradient transforms.

Proposition 1 (Reverse-mode dual typing). Let G be a directed acyclic graph (DAG) whose nodes carry declared units $\rho _ { v }$ and whose every primitive operation is unit-consistent with respect to those declarations. If the loss node is typed dimensionless, then, at every point where the relevant derivatives exist, every reverse-mode gradient node transforms by the dual law $\rho _ { v } ^ { - \top }$ . The proof is given in App. G.1.

Under the declared scalar rescaling, $x \mapsto c x$ gives $\nabla _ { x } \mathcal { L } \mapsto c ^ { - 1 } \nabla _ { x } \mathcal { L }$ , meaning that multiplying the activation by c divides the activation gradient by c. These factors cancel in the primal–dual RMS product in Eq. 5, making the product comparable across rescalings of a shared scalar-unit block. To keep a scalar diagnostic unchanged under a change of units, construct the diagnostic’s dimensionless output from unit-consistent operations.

## 2.2 Local typing rules

Addition and residuals. $z : = x + y$ requires $u ( x ) =$ $u ( y ) = u ( z )$ . In $h _ { \ell + 1 } = h _ { \ell } + r _ { \ell }$ , the branch must return the residual-stream unit. In pre-layer-normalization (Pre-LN) blocks, $r _ { \ell } : = F _ { \ell } ( N ( h _ { \ell } ) )$ with $N : U _ { \mathrm { r e s } }  1$ The branch $F _ { \ell } : 1  U _ { \mathrm { r e s } }$ restores the stream unit, a role played implicitly by output projections and gains in vanilla blocks.

Normalization as nondimensionalization. With $s _ { x } : = \mathrm { R M S } ( x )$ on a nonzero shared-unit block, $N ( x ) : =$ $x / s _ { x }$ satisfies $N ( c x ) = N ( x )$ for $c > 0$ . This division removes the unit and the overall magnitude. For outputs that need units, Sec. 4 retains explicit scales alongside the dimensionless coordinates. App. F.7 gives the gauge interpretation and afine-normalization examples.

Softmax and attention. Softmax depends on logit diferences and is unchanged by a common shift within each row. A logit diference must be dimensionless because exponentiating the logit diference gives a probability ratio, which must not change merely because the unit system changes. If raw scores $z _ { i j } : = q _ { i } ^ { \top } k _ { j }$ carry unit $u ( q ) u ( k )$ , the temperature is a unit converter with $u ( \tau ) = u ( q ) u ( k )$ . With per-head coordinate scales $s _ { Q }$ and $s _ { K }$ , the calibrated temperature is

$$
\tau _ { \ell , h } : = \sqrt { d _ { h } } s _ { Q , \ell , h } s _ { K , \ell , h } ,\tag{3}
$$

of which the standard $\sqrt { d _ { h } }$ is the $s _ { Q } = s _ { K } = 1$ special case. Value aggregation preserves $u ( V )$ . The output projection must map $U _ { V }$ to $U _ { \mathrm { r e s } } . ~ \mathrm { A p p }$ . F.2 gives the variance assumptions and empirical scale estimates.

Activations and gates. Standard non-homogeneous activations, including sigmoid, tanh, the Gaussian error linear unit (GELU), and the sigmoid linear unit (SiLU), require dimensionless arguments. In gated $\mathrm { M L P s } ~ z : =$ $v \odot g .$ units multiply. To retain the value-branch unit $u ( v )$ at the output, the gate must be dimensionless, $u ( g ) = 1$ . The raw gate $a \sigma ( a )$ of a Swish-gated linear unit (SwiGLU) silently assumes that a is dimensionless, whereas the normalized gate $\tilde { g } : = ( a / s _ { a } ) \sigma ( a / s _ { a } )$ is typed. App. F.7 gives the homogeneity and transitionscale examples.

Loss aggregation. To keep the objective value unchanged under a change of unit system, we construct a dimensionless objective using unit-consistent operations. A dimensionless objective has transformation factor one. Cross-entropy is dimensionless when the input logits are dimensionless, whereas squared error has $u ( \| y - \hat { y } \| ^ { 2 } ) = U _ { y } ^ { 2 }$ . In $\begin{array} { r } { \mathcal { L } : = \sum _ { i } \lambda _ { i } \mathcal { L } _ { i } . } \end{array}$ consistency forces $u ( \lambda _ { i } ) = u ( \mathcal { L } _ { i } ) ^ { - 1 }$ . Each loss weight acts as an inverse-unit converter. We factor each loss weight into a dimensionless preference $\alpha _ { i }$ and an inverse characteristic scale $1 / S _ { i }$

$$
\lambda _ { i } = \alpha _ { i } / S _ { i } , \qquad u ( S _ { i } ) = u ( \mathcal { L } _ { i } ) , \quad u ( \alpha _ { i } ) = 1 ,\tag{4}
$$

where $1 / S _ { i }$ converts each loss to a dimensionless value and $\alpha _ { i }$ sets the corresponding task preference. App. E.2 gives characteristic-scale examples and characterizes invariant linear weighting rules.

Gradients and dual units. If L is dimensionless and $u ( x ) \ = \ U$ , then $\mathrm { d } \mathcal { L } ~ = ~ \langle \nabla _ { x } \mathcal { L } , \mathrm { d } x \rangle$ forces $u ( \nabla _ { x } \mathcal { L } ) = U ^ { - 1 }$ . A raw gradient norm changes with the chosen unit. Comparing blocks with diferent units requires unit cancellation. Dimensionless sensitivity scales include $x \odot \nabla _ { x } { \mathcal { L } }$ elementwise, or on shared-unit blocks

$$
\Pi _ { x } : = \operatorname { R M S } ( x ) \operatorname { R M S } ( \nabla _ { x } \mathcal { L } ) ,\tag{5}
$$

the primal–dual RMS product.

Optimizer updates and primal units. The update $\theta _ { t + 1 } ~ = ~ \theta _ { t } + \Delta \theta _ { t }$ needs $u ( \Delta \theta ) ~ = ~ U _ { \theta }$ , while $u ( \nabla _ { \theta } \mathcal { L } ) = U _ { \theta } ^ { - 1 }$ . With a dimensionless learning rate, a typed preconditioner supplies the missing factor through $u ( P _ { \theta } ) = U _ { \theta } ^ { 2 } . \mathrm { A p p }$ . E.3 gives relative-update examples and the covariance and quotient-geometry results.

## 2.3 Residual scale dynamics

Residual additions can amplify scale even when both terms carry the same unit. Here $r _ { \ell }$ is the branch contribution added to the hidden state during the forward pass. We track this amplification using the branch-tostream RMS ratio and the branch’s alignment with the stream.

Let $s _ { \ell } : = \| h _ { \ell } \| _ { 2 } / \sqrt { d }$ and assume $h _ { \ell } \neq 0 .$ . For $r _ { \ell } \neq $ 0, define the dimensionless residual step size $\alpha _ { \ell } : =$ $\mathrm { R M S } ( r _ { \ell } ) / \mathrm { R M S } ( h _ { \ell } ) > 0$ . Expanding $\| h _ { \ell } + r _ { \ell } \| _ { 2 } ^ { 2 }$ gives

$$
s _ { \ell + 1 } ^ { 2 } = s _ { \ell } ^ { 2 } ( 1 + \alpha _ { \ell } ^ { 2 } + 2 \alpha _ { \ell } c _ { \ell } ) , \quad c _ { \ell } : = \frac { \langle h _ { \ell } , r _ { \ell } \rangle } { \| h _ { \ell } \| \| r _ { \ell } \| } .\tag{6}
$$

For persistent positive alignment $c _ { \ell } \geq c _ { 0 } > 0 .$ , bounded scale requires $\textstyle \sum _ { \ell } \alpha _ { \ell } = \bar { O ( 1 ) }$ , giving $O ( L ^ { - 1 } )$ per layer for uniformly comparable steps. Orthogonalization removes the linear term and permits $\begin{array} { r } { \sum _ { \ell } \alpha _ { \ell } ^ { 2 } = O ( 1 ) } \end{array}$ with uniform $\Theta ( L ^ { - 1 / 2 } )$ steps, as in depth-wise $\mu \mathrm { P }$ (Yang et al., 2024). Sec. 4 implements this projection. App. F.4 discusses zero branches, nonuniform schedules, and near-orthogonal contributions.

App. E collects the full loss-weighting, optimizer, and residual-stability results, while App. F derives the attention temperature and residual recursion.

## 3 UVI

To locate scale mismatches within a neural network during training, we propose UVI. The indices measure dimensionless log-deviations from a target or logdispersion across depth. Large values point to the corresponding interface for inspection. Each index is defined relative to a declared unit assignment and reference regime. With a small dimensionless floor ϵ, the residual and attention indices are

$$
\begin{array} { r } { \mathrm { U V I } _ { \mathrm { r e s } } ( \ell ) : = \Big | \log \frac { \alpha _ { \ell } + \epsilon } { \alpha ^ { \star } + \epsilon } \Big | , \qquad \alpha ^ { \star } : = \frac { 1 } { \sqrt { 2 L } } , } \end{array}\tag{7}
$$

$$
\begin{array} { r } { \mathrm { U V I } _ { \mathrm { a t t n } } ( \ell , h ) : = \Big | \log \frac { \bar { v } _ { \ell , h } + \epsilon } { v ^ { \star } + \epsilon } \Big | , \qquad v ^ { \star } : = 1 , } \end{array}\tag{8}
$$

where $\alpha _ { \ell }$ is the measured residual step size, and $\bar { v } _ { \ell , h }$ is the batch-averaged row-centered logit variance after all model-specific scaling.

App. E.4 defines the activation, gradient, and optimizer indices and the auxiliary drift diagnostics. All five indices and the two additional diagnostics can be computed inexpensively on small batches at intervals during training.

## 4 GCT ARCHITECTURE

The goal is to make the block’s dimensionless computation independent of arbitrary changes in internal scale. We apply the NDA rules in two designs, the Unit-Consistent Transformer (UCT) and GCT. UCT adds scale-correction modules to a conventional Transformer, while GCT normalizes the residual stream after every residual addition. App. H.4 specifies UCT’s three corrective modules.

GCT computes every block on normalized, dimensionless coordinates and keeps any required magnitude and units separate. The name Gauge-Canonical reflects our choice to fix the scale gauge through RMS normalization. In the exact construction, all positive rescalings of a nonzero vector share one RMS-one representative, which is the canonical coordinate. Each unitful quantity is factorized as

unitful = unit scale × canonical coordinate. (9)

## 4.1 Canonical state and residual block

State factorization. We separate a token state’s magnitude and direction by writing $h _ { \ell } = s _ { \ell } \bar { h } _ { \ell }$ , with $s _ { \ell } = \mathrm { R M S } ( h _ { \ell } )$ and $u ( s _ { \ell } ) = U _ { \mathrm { r e s } }$ . For $h _ { \ell } \neq 0$ , dividing by $s _ { \ell }$ gives $\bar { h } _ { \ell } = h _ { \ell } / s _ { \ell }$ with $\mathrm { R M S } ( \bar { h } _ { \ell } ) = 1$ . We call $s _ { \ell }$ the scale state, which carries the magnitude and units separately from the vector used for computation. The dimensionless vector $\bar { h } _ { \ell }$ is the canonical coordinate, which preserves the direction of $h _ { \ell }$ and has RMS one. For numerical stability, we use the canonicalizer

$$
\mathcal { C } ( x ) : = \frac { x } { \sqrt { \mathrm { R M S } ( x ) ^ { 2 } + \epsilon _ { x } } } .\tag{10}
$$

For $x \neq 0$ , setting $\epsilon _ { x } = 0$ gives exact RMS normalization. The finite-floor implementation and its deviations from exact unit consistency are described in App. F.1.

Orthogonalized residual addition. An attention or MLP branch produces an output vector $\bar { a } _ { \ell }$ , which we call the branch proposal to denote the branch’s suggested contribution to the current state. GCT projects, normalizes, and scales the branch proposal before adding the resulting contribution to the residual stream. To remove the first-order scale growth identified in Sec. 2.3, we subtract the component of the branch output that points along the current state. The analysis uses the exact orthogonal projector

$$
P _ { \bar { h } } ^ { \perp } a : = a - \frac { \langle a , \bar { h } \rangle } { \| \bar { h } \| ^ { 2 } } \bar { h } , \qquad \bar { h } \neq 0 ,
$$

and learnable dimensionless scalars $\alpha _ { \ell } ,$ initialized at $1 / \sqrt { 2 L }$ . Each block applies the attention sub-block followed by the MLP sub-block, using the following update for each:

$$
\begin{array} { r l } & { \bar { a } _ { \ell } : = \Phi _ { \ell } ( \bar { h } _ { \ell } ) , \qquad \hat { a } _ { \ell } : = \mathcal { C } ( P _ { \bar { h } _ { \ell } } ^ { \perp } \bar { a } _ { \ell } ) , } \\ & { \bar { h } _ { \ell ^ { + } } : = \mathcal { C } ( \bar { h } _ { \ell } + \alpha _ { \ell } \hat { a } _ { \ell } ) . } \end{array}\tag{11}
$$

Here $\Phi _ { \ell }$ is the attention or MLP branch, and $\ell ^ { + }$ denotes the state after that sub-block. With a nonzero projected proposal and zero canonicalization floors, $\hat { a } _ { \ell } \perp \bar { h } _ { \ell }$ and both vectors have RMS one. Before the final normalization (retraction), the contribution $\alpha _ { \ell } \hat { a } _ { \ell }$ is orthogonal to the stream and has relative RMS |α<sub>ℓ</sub>|; retraction returns the state to RMS one. Orthogonality applies before retraction, and the final displacement need not be orthogonal. App. F.4 gives the exact rotation and limiting cases.

## 4.2 Canonical interfaces and guarantees

Attention in canonical coordinates. Queries and keys are canonicalized per head vector, $\bar { q } : = \mathcal { C } ( q )$ and $\bar { k } : = \mathcal { C } ( k )$ . A per-head dimensionless gain $g _ { h }$ , initialized at 1, tunes the attention entropy within these canonical coordinates. App. F.2 gives the variance assumptions.

Gates in canonical coordinates. The gated MLP canonicalizes preactivations per token to ensure that the sigmoid receives a dimensionless input. The MLP uses the normalized gate $\tilde { g } : = \mathcal { C } ( a ) \sigma ( \mathcal { C } ( a ) )$ to multiply the value branch.

Heads, scales, and parameters. The classification head produces dimensionless logits from $\mathcal { C } ( \bar { h } _ { L } )$ with a learnable dimensionless logit gain. Outputs with units use $\hat { y } : = s _ { y } \hat { \bar { y } } $ , where the explicit scale $s _ { y }$ restores the magnitude and units. For tasks with dimensionless outputs, such as language modeling, the computation does not depend on the scale state, and we implement the residual stream using only canonical coordinates. App. H.4 gives the scale parameterization.

Scope and guarantees. On nonzero inputs with zero canonicalization floors, ${ \mathcal { C } } ( c h ) = { \mathcal { C } } ( h )$ for $c > 0$ makes the dimensionless network output invariant at every block-input boundary, including $h _ { 0 }$ , and at the final block-output boundary (App. F.6). Canonical coordinates preserve continuous invariant expressivity under independent positive rescalings; targets depending on absolute magnitude or dimensionless scale ratios require retaining that information (Apps. E.5 and E.6).

## 5 EXPERIMENTS

We test whether GCT’s explicit scale control makes learning-rate and depth transfer more reliable, keeps predictions unchanged under internal rescaling, and preserves performance when residual states are stored at low precision. The practical goal is to obtain these benefits while retaining competitive clean validation loss.

## 5.1 Setup

We evaluate decoder-only Transformers with 51M, 124M, 350M, and 1B parameters to test GCT across a broad range of model sizes. The baselines include blocks modeled on Large Language Model Meta AI (LLaMA). Within each comparison, we train the architectures on FineWeb-Edu using the same data order, token budget, and optimization recipe. We test higher learning rates in the 51M models and transfer the 51M hyperparameters to deeper networks. We compare clean loss and internal-rescaling invariance at 124M, and repeat quantization tests at 350M and 1B. At 124M, GCT ranks third of eight architectures by unrounded three-seed mean loss. GCT’s mean loss is within 0.005 nats of LayerScale, nGPT, and Pre-LN and 0.023 nats above QK-Norm, which makes these models comparable in clean validation performance before the robustness tests (Table 4). To test transfer across model widths and compatibility with another optimizer, we also evaluate width transfer and the momentum orthogonalized by Newton–Schulz (Muon) optimizer. To test the principles beyond language modeling, we include two vision benchmarks and PINNs. App. H specifies the architectures, recipes, seeds, and evaluation protocols.

![](images/b6af323b4193138e6159dd0dfdd05355a32baf7a9acc7324e5c13d72e0dd1d67.jpg)  
Figure 2: Final validation loss as a function of peak learning rate for primary-seed 51M models trained on 600M tokens. GCT and nGPT retain low loss at high rates that worsen the other three architectures, showing a wider useful learning-rate range.

## 5.2 Learning-rate robustness

Internal scale drift can make the same learning rate behave diferently during training. GCT limits this variation by keeping the residual state normalized before each attention and MLP computation. We test whether GCT can train stably at higher learning rates while continuing to reduce validation loss.

GCT and nGPT remain stable across the full 16× learning-rate range and retain low validation loss at the highest rates (Table 5 and Figure 2). Pre-LN and LLaMA-style deteriorate beyond the respective best sampled rates. QK-Norm finishes training at every sampled rate without numerical divergence, but QK-Norm’s upper-end loss is 0.18 nats above the architecture’s best sampled loss. At the additional rate $9 . 6 \times 1 0 ^ { - 3 }$ , GCT and nGPT improve further, while QK-Norm’s loss is 0.197 nats above the architecture’s best sampled loss. At this rate, we stop the Pre-LN and LLaMA-style runs at step 100, about 8% of the training budget, when the runs’ early UVI scores exceed the preset stopping threshold (App. H). Both normalized models continue improving through this highest tested rate, placing both models’ best measured results at the upper boundary of the sweep. GCT maintains low validation loss across a wide range of high learning rates, including rates that worsen validation performance in the other three architectures.

Table 1: Validation loss in fixed-budget depth transfer with the primary seed, 400M tokens, and hyperparameters transferred unchanged from $L = 8 .$ . GCT achieves the lowest loss at both depths, and adding the GCT projection also improves nGPT at $L = 4 8 .$
<table><tr><td>architecture</td><td> $L = 2 4$ </td><td> $L = 4 8$ </td></tr><tr><td>Pre-LN</td><td>4.397</td><td>4.598</td></tr><tr><td>LLaMA-style</td><td>4.322</td><td>4.513</td></tr><tr><td>QK-Norm</td><td>4.206</td><td>4.322</td></tr><tr><td>nGPT</td><td>3.996</td><td>4.243</td></tr><tr><td>GCT (ours)</td><td>3.973</td><td>3.903</td></tr><tr><td>GCT without  $P ^ { \perp }$ </td><td>3.983</td><td>3.938</td></tr><tr><td>nGPT with  $P ^ { \perp }$ </td><td>4.005</td><td>3.968</td></tr></table>

GCT improves by 0.125 nats and nGPT by 0.127 nats when the learning rate selected at 51M is applied unchanged at 124M, whereas the other baselines improve by 0.052 to 0.072 nats (Table 7). Selection uses the original five-point $1 6 \times \mathrm { g r i d }$ . The additional $9 . 6 \times 1 0 ^ { - 3 }$ point is a separate stress test. Every architecture receives a separate sweep and transfer. QK-Norm reaches 3.091 at the QK-Norm-selected rate of $1 . 2 \times 1 0 ^ { - 3 }$ , while nGPT and GCT reach 3.043 and 3.045, respectively, at $4 . 8 \times 1 0 ^ { - 3 }$ . The normalized architectures achieve the lowest losses under this transfer protocol. For GCT, this transfer result shows that a useful rate can be selected on the smaller model and reused at the larger size without retuning.

## 5.3 Depth transfer and architectural ablations

nGPT is the closest architectural baseline because both models compute with normalized states and are scaleinvariant in the zero-floor formulations of GCT and nGPT. We test the direction of the added residual contribution because the analysis in Sec. 2.3 shows that subtracting the component along the state removes first-order magnitude growth. This orthogonalization permits per-layer steps of order $L ^ { - 1 / 2 }$ while keeping the cumulative squared step size bounded. To test depth transfer without retuning, we apply the $L = 8$ hyperparameters unchanged at $L = 2 4$ and $L = 4 8 .$ GCT remains stable in this primary-seed probe and reaches a validation loss of 3.903 at $L = 4 8$ with a fixed 400M-token budget, compared with 4.243 for nGPT (Table 1).

We remove proposal canonicalization or the orthogonal projection $P ^ { \perp }$ from GCT to measure how each component afects validation loss. At $L = 8 ,$ removing proposal canonicalization costs +0.6 nats, whereas removing $P ^ { \perp }$ has negligible efect (Table 6). In the primary-seed depth cells, removing $P ^ { \perp }$ increases loss by 0.010 nats at $L = 2 4$ and 0.035 nats at $L = 4 8$ (Table 1). Proposal normalization is important already in the shallow model, while the projection’s benefit grows with depth in these probes. Figure 3 connects this depth comparison to the measured residual step sizes and alignments.

To test whether the projection also helps an existing normalized architecture, we add $P ^ { \perp }$ to nGPT’s update direction before nGPT’s per-dimension step, removing the direction’s component along the current state. Adding $P ^ { \perp }$ changes $\mathrm { n G P T ^ { \circ } s }$ loss by approximately +0.010 nats at $L = 2 4$ and reduces nGPT’s loss by 0.275 nats at $L = 4 8$ , from 4.243 to 3.968 (Table 1). The gain at $L = 4 8$ shows that this NDA-derived component can also improve depth transfer beyond GCT under the same fixed-budget protocol.

## 5.4 Invariance under internal rescaling

We test sensitivity to internal rescaling by multiplying a hidden state by a positive factor, $h  c h$ , at the embedding output or at mid-depth. This rescaling changes the state’s magnitude while preserving the state’s direction and leaves the trained weights fixed. For GCT and nGPT, validation loss is unchanged to the reported precision and 32-bit floating-point (fp32) block invariance error remains near the numerical floor over $c \in [ 1 0 ^ { - 2 } , 1 0 ^ { 2 } ]$ (App. I.2, Figure 5, and Table 8). The trained GCT model retains validation performance across these internal scale changes without retraining.

Weight-space fingerprint. Prop. 6 predicts zero radial sensitivity $\langle W , \nabla _ { W } \mathcal { L } \rangle _ { F }$ when downstream computation cancels rescaling of a matrix’s output. All 54 of 54 architecture-by-role predictions match across the eight trained 124M models, and the 1B audit likewise matches 34 of 34 cells. Predicted-null cells lie at the fp32 floor. App. I.2 gives the measured sensitivities and rolewise interpretation.

## 5.5 Post-training quantization (PTQ) robustness

Scale invariance removes pure changes in magnitude, but quantization can also change a hidden state’s direction through rounding (Thm. 6 and Cor. 5). We test whether GCT’s normalized residual stream (Sec. 4) limits this additional error and preserves validation performance when states are stored at low precision. NDA predicts that reducing token-to-token scale variation lowers the error associated with a shared quantization scale. To test this prediction, we apply PTQ to the residual stream at every block boundary, as in hidden-state caching, ofloading, or transmission. In a pre-norm network, the skip path carries the stored tensor, including the rounding errors, into subsequent residual additions.

Prop. 5 and Thm. 7 show that, under the stated assumptions and high-resolution regime, the worst-case downstream output error grows as $\Theta ( \kappa _ { \mathrm { s c a l e } } / q )$ , where $\kappa _ { \mathrm { s c a l e } }$ is the ratio of the largest token RMS to the smallest and $q = 2 ^ { b - 1 } - 1$ . This dependence remains even when subsequent normalization cancels pure changes in magnitude exactly. In efective-resolution terms, crosstoken scale variation consumes $\log _ { 2 } \kappa _ { \mathrm { s c a l e } }$ bits. The zero-floor GCT stream fixes every token’s RMS at one, setting $\kappa _ { \mathrm { s c a l e } } = 1$ and removing this source of resolution loss. We compare one scale for the entire tensor with one scale per token to test whether adapting the quantization spacing to each token’s magnitude reduces the baselines’ error and narrows the error gap to GCT.

We use weight-only, linear-input activation, and key– value (KV) cache quantization as controls to identify where scale causes the largest errors. The strongest separation between normalized streams and the other baselines occurs when the residual stream uses a shared tensor scale (Tables 2 and 10). Per-token scaling sharply reduces the baselines’ int8 loss increases and narrows the gap to GCT, showing that explicit per-token scale management compensates for much of the measured error in the unnormalized streams.

At 1B, calibrated per-tensor stream int8 increases validation loss by 0.009 nats for GCT, while the largest baseline increase is 5.51 nats. Under per-token 4-bit integer (int4) quantization, GCT limits the increase to +0.739 nats, compared with +4.661 for nGPT and at least +5.230 for the other baselines (Table 2). On the independent second 1B seed, static per-tensor stream int8 changes the worst baseline, LLaMA-style, by +5.67 nats and GCT by +0.008. Under per-token int4, GCT changes by +0.47, compared with at least +5.13 for the other non-nGPT baselines.

Table 2: PTQ at 1B with $d = 1 7 9 2$ $L = 2 4$ , 4B tokens, and the primary seed. Base is the unquantized 16-bit brain floating-point (bf16) loss on a fixed 32-batch subset, while other entries are loss increases in nats. Per-channel 4-bit weight quantization (W4) changes only weights. In 8-bit weight-and-activation quantization (W8A8), linear-input activations use static scales. Stream columns use static per-layer tensor scales or dynamic per-token scales at every block boundary. The logit head remains in full precision. GCT has the smallest stream loss increase with per-tensor int8 and per-token int4 quantization, while per-tensor int4 causes large loss increases for every architecture.
<table><tr><td colspan="2"></td><td colspan="2">weights and activations</td><td colspan="2">stream h → int8</td><td colspan="2">stream h → int4</td></tr><tr><td>architecture</td><td>base</td><td>W4</td><td>W8A8</td><td>tensor</td><td>token</td><td>tensor</td><td>token</td></tr><tr><td>Pre-LN</td><td>2.759</td><td>+0.045</td><td>+0.028</td><td>+3.018</td><td>+0.049</td><td>+6.390</td><td>+7.101</td></tr><tr><td>LLaMA-style</td><td>2.744</td><td>+0.032</td><td>+0.348</td><td>+5.513</td><td>+0.030</td><td>+7.117</td><td>+5.474</td></tr><tr><td>QK-Norm</td><td>2.737</td><td>+0.029</td><td>+0.833</td><td>+5.426</td><td>+0.021</td><td>+8.074</td><td>+5.230</td></tr><tr><td>nGPT</td><td>2.751</td><td>+0.038</td><td>+0.075</td><td>+0.032</td><td>+0.008</td><td>+19.083</td><td>+4.661</td></tr><tr><td>GCT (ours)</td><td>2.756</td><td>+0.028</td><td>+0.079</td><td>+0.009</td><td>+0.001</td><td>+9.517</td><td>+0.739</td></tr></table>

GCT combines the int8 benefit of normalized token states with substantially lower per-token int4 degradation than nGPT in this test. Both models have $\kappa _ { \mathrm { s c a l e } } \approx 1$ , and $\kappa _ { \mathrm { s c a l e } } = 1$ exactly in the zero-floor limit. Equal token RMS does not control within-vector amplitude, downstream directional sensitivity, or error accumulation across blocks. The cross-token bound alone cannot explain the models’ diferent int4 loss increases. GCT achieves the smaller int4 loss increase with per-token scales. Per-tensor int4 causes large loss increases for every architecture.

On the independently trained second-seed 1B models, we evaluate the same residual-stream sites and granularities through zero-shot accuracy averaged over four tasks (Table 11). Dynamic per-tensor int8 changes mean accuracy by −0.1 points for GCT and −0.8 for nGPT, compared with −19.6 to −22.7 for the other baselines. Per-token int8 limits those baseline changes to −0.5 to −1.9 points. Under per-token int4, GCT changes by −7.4 points, nGPT by −17.4, and the other baselines by −22.6 to −23.1.

## 5.6 PINNs under changes of physical units

We use PINNs to test NDA’s loss-aggregation rule on data-fitting and partial diferential equation (PDE) residual losses, which can carry diferent physical units. Following Sec. 2.2 and Thm. 2, we divide each loss by a characteristic scale with matching units to keep the total objective unchanged under a unit conversion. We derive these scales from the measured magnitude of the data or forcing and the units introduced by the PDE operator, and use the scaled output head of Sec. 4. We then re-express the same Burgers and Helmholtz problems in diferent units to test whether this construction preserves physical solution accuracy without retuning the loss weights.

Across the $1 0 ^ { 4 }$ unit sweep, NDA’s scaled output head and loss weights keep the worst relative $L _ { 2 }$ error at 0.015, whereas equal or adaptive weights degrade for at least one rescaling (Table 17, five seeds). Figure 9 visualizes the recovered Burgers fields. These results show that the loss-aggregation rule, applied together with the scaled output head, preserves accurate physical solutions across the tested unit systems.

Additional diagnostic, width-transfer, optimizer, vision, and deployment results are collected in App. I.

## 6 CONCLUSION

Our NDA framework establishes a typing principle for scale management across the computation graph. The NDA rules unify the scale-control mechanisms used in Transformers, including normalization, residual scaling, attention temperature, loss weighting, and parameter updates. UVI diagnostics and the weightspace audit make these rules testable and actionable by locating scale mismatches and checking the rescaling behavior predicted by NDA. GCT implements these scale-control rules in a Transformer architecture, and the typed multi-loss objective makes each loss dimensionless before combining the terms. Experiments spanning 51M to 1B parameters show that GCT achieves learning-rate and depth transfer, internal-rescaling invariance, and improved residual-stream quantization robustness while retaining competitive clean validation performance. Vision and PINN studies support the use of NDA beyond language modeling. These results demonstrate the benefits of explicit scale management grounded in NDA for neural network design.

## AI USE STATEMENT

Generative AI tools were used as assistive tools for writing and editing, mathematical proofreading and consistency checking, and code development and debugging. Any AI-assisted outputs incorporated into this work were reviewed and validated by the authors as appropriate. The authors take full responsibility for the final content of this work, including any text, claims, or artifacts produced with the assistance of generative AI.

## References

Arora, S., Li, Z., and Lyu, K. (2019). Theoretical Analysis of Auto Rate-Tuning by Batch Normalization. In ICLR.

Ashkboos, S., Mohtashami, A., Croci, M. L., Li, B., Cameron, P., Jaggi, M., Alistarh, D., Hoefler, T., and Hensman, J. (2024). QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. In NeurIPS.

Ba, L. J., Kiros, J. R., and Hinton, G. E. (2016). Layer Normalization. CoRR, abs/1607.06450.

Bachlechner, T., Majumder, B. P., Mao, H. H., Cottrell, G., and McAuley, J. J. (2021). ReZero is all you need: fast convergence at large depth. In UAI, volume 161, pages 1352–1361.

Bakarji, J., Callaham, J., Brunton, S. L., and Kutz, J. N. (2022). Dimensionally consistent learning with Buckingham Pi. Nat. Comput. Sci., 2(12):834–844.

Bernstein, J. and Newhouse, L. (2025). Modular Duality in Deep Learning. In ICML, volume 267.

Bisk, Y., Zellers, R., Bras, R. L., Gao, J., and Choi, Y. (2020). PIQA: Reasoning about Physical Commonsense in Natural Language. In AAAI, pages 7432–7439.

Blake, C., Eichenberg, C., Dean, J., Balles, L., Prince, L. Y., Deiseroth, B., Cruz-Salinas, A. F., Luschi, C., Weinbach, S., and Orr, D. (2025). u-µP: The Unit-Scaled Maximal Update Parametrization. In ICLR.

Blake, C., Orr, D., and Luschi, C. (2023). Unit Scaling: Out-of-the-Box Low-Precision Training. In ICML, volume 202, pages 2548–2576.

Bondarenko, Y., Nagel, M., and Blankevoort, T. (2023). Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing. In NeurIPS.

Brock, A., De, S., Smith, S. L., and Simonyan, K. (2021). High-Performance Large-Scale Image Recognition Without Normalization. In ICML, volume 139, pages 1059–1071.

Buckingham, E. (1914). On physically similar systems; illustrations of the use of dimensional equations. Physical review, 4(4):345.

Chen, Z., Badrinarayanan, V., Lee, C., and Rabinovich, A. (2018). GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. In ICML, volume 80, pages 793–802.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. (2018). Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. CoRR, abs/1803.05457.

Cohen, T., Weiler, M., Kicanaoglu, B., and Welling, M. (2019). Gauge Equivariant Convolutional Networks and the Icosahedral CNN. In ICML, volume 97, pages 1321–1330.

De, S. and Smith, S. L. (2020). Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks. In NeurIPS.

Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., Steiner, A. P., Caron, M., Geirhos, R., Alabdulmohsin, I., Jenatton, R., Beyer, L., Tschannen, M., Arnab, A., Wang, X., Ruiz, C. R., Minderer, M., Puigcerver, J., Evci, U., Kumar, M., van Steenkiste, S., Elsayed, G. F., Mahendran, A., Yu, F., Oliver, A., Huot, F., Bastings, J., Collier, M., Gritsenko, A. A., Birodkar, V., Vasconcelos, C. N., Tay, Y., Mensink, T., Kolesnikov, A., Pavetic, F., Tran, D., Kipf, T., Lucic, M., Zhai, X., Keysers, D., Harmsen, J. J., and Houlsby, N. (2023). Scaling Vision Transformers to 22 Billion Parameters. In ICML, volume 202, pages 7480–7512.

Dettmers, T., Lewis, M., Belkada, Y., and Zettlemoyer, L. (2022). GPT3.int8(): 8-bit Matrix Multiplication for Transformers at Scale. In NeurIPS.

Dinh, L., Pascanu, R., Bengio, S., and Bengio, Y. (2017). Sharp Minima Can Generalize For Deep Nets. In ICML, volume 70, pages 1019–1028.

Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. (2023). OPTQ: Accurate Quantization for Generative Pre-trained Transformers. In ICLR.

He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep Residual Learning for Image Recognition. In CVPR, pages 770–778.

Hendrycks, D. and Dietterich, T. G. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. In ICLR.

Henry, A., Dachapally, P. R., Pawar, S. S., and Chen, Y. (2020). Query-Key Normalization for Transformers. In EMNLP (Findings), volume EMNLP 2020, pages 4246–4253.

Hofer, E., Banner, R., Golan, I., and Soudry, D. (2018). Norm matters: eficient and accurate normalization

schemes in deep networks. In NeurIPS, pages 2164– 2174.

Hofmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Vinyals, O., Rae, J. W., and Sifre, L. (2022). An empirical analysis of compute-optimal large language model training. In NeurIPS.

Iofe, S. and Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In ICML, volume 37, pages 448–456.

Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A. G., Adam, H., and Kalenichenko, D. (2018). Quantization and Training of Neural Networks for Eficient Integer-Arithmetic-Only Inference. In CVPR, pages 2704–2713.

Jordan, K., Jin, Y., Boza, V., You, J., Cesista, F., Newhouse, L., and Bernstein, J. (2024). Muon: An optimizer for hidden layers in neural networks, 2024. URL https://kellerjordan. github. io/posts/muon, 6(3):4.

Kaba, S., Mondal, A. K., Zhang, Y., Bengio, Y., and Ravanbakhsh, S. (2023). Equivariance with Learned Canonicalization Functions. In ICML, volume 202, pages 15546–15566.

Karpathy, A. (2022). nanoGPT. https://github. com/karpathy/nanoGPT. GitHub repository.

Karras, T., Aittala, M., Lehtinen, J., Hellsten, J., Aila, T., and Laine, S. (2024). Analyzing and Improving the Training Dynamics of Difusion Models. In CVPR, pages 24174–24184.

Kendall, A., Gal, Y., and Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. In CVPR, pages 7482–7491.

Kennedy, A. (2009). Types for Units-of-Measure: Theory and Practice. In CEFP, volume 6299, pages 268–305.

Kingma, D. P. and Ba, J. (2015). Adam: A Method for Stochastic Optimization. In ICLR.

Kunin, D., Sagastuy-Breña, J., Ganguli, S., Yamins, D. L. K., and Tanaka, H. (2021). Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics. In ICLR.

Large, T., Liu, Y., Huh, J., Bahng, H., Isola, P., and Bernstein, J. (2024). Scalable Optimization in the Modular Norm. In NeurIPS.

Li, Z. and Arora, S. (2020). An Exponential Learning Rate Schedule for Deep Learning. In ICLR.

Lin, J., Tang, J., Tang, H., Yang, S., Chen, W., Wang, W., Xiao, G., Dang, X., Gan, C., and Han, S. (2024). AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration. In ML-Sys.

Liu, Z., Yuan, J., Jin, H., Zhong, S. H., Xu, Z., Braverman, V., Chen, B., and Hu, X. (2024). KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. In ICML, volume 235, pages 32332–32344.

Loshchilov, I., Hsieh, C., Sun, S., and Ginsburg, B. (2025). nGPT: Normalized Transformer with Representation Learning on the Hypersphere. In ICLR.

Loshchilov, I. and Hutter, F. (2019). Decoupled Weight Decay Regularization. In ICLR.

Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2017). Pointer Sentinel Mixture Models. In ICLR.

Micikevicius, P., Narang, S., Alben, J., Diamos, G. F., Elsen, E., García, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., and Wu, H. (2018). Mixed Precision Training. In ICLR.

Nagel, M., Fournarakis, M., Amjad, R. A., Bondarenko, Y., van Baalen, M., and Blankevoort, T. (2021). A White Paper on Neural Network Quantization. CoRR, abs/2106.08295.

Neyshabur, B., Salakhutdinov, R., and Srebro, N. (2015). Path-SGD: Path-Normalized Optimization in Deep Neural Networks. In NIPS, pages 2422–2430.

Paperno, D., Kruszewski, G., Lazaridou, A., Pham, Q. N., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., and Fernández, R. (2016). The LAMBADA dataset: Word prediction requiring a broad discourse context. In ACL (1).

Penedo, G., Kydlícek, H., Allal, L. B., Lozhkov, A., Mitchell, M., Rafel, C. A., von Werra, L., and Wolf, T. (2024). The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. In NeurIPS.

Puny, O., Atzmon, M., Smith, E. J., Misra, I., Grover, A., Ben-Hamu, H., and Lipman, Y. (2022). Frame Averaging for Invariant and Equivariant Network Design. In ICLR.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. J. Comput. Phys., 378:686–707.

Salimans, T. and Kingma, D. P. (2016). Weight Normalization: A Simple Reparameterization to Accelerate

Training of Deep Neural Networks. In NIPS, page 901.

Shleifer, S., Weston, J., and Ott, M. (2021). Norm-Former: Improved Transformer Pretraining with Extra Normalization. CoRR, abs/2110.09456.

Su, J., Ahmed, M. H. M., Lu, Y., Pan, S., Bo, W., and Liu, Y. (2024). RoFormer: Enhanced transformer with Rotary Position Embedding. Neurocomputing, 568:127063.

Sun, M., Chen, X., Kolter, J. Z., and Liu, Z. (2024). Massive Activations in Large Language Models. CoRR, abs/2402.17762.

Team, G. (2024). Gemma 2: Improving Open Language Models at a Practical Size. CoRR, abs/2408.00118.

Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., and Jégou, H. (2021a). Training dataeficient image transformers & distillation through attention. In ICML, volume 139, pages 10347–10357.

Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., and Jégou, H. (2021b). Going deeper with Image Transformers. In ICCV, pages 32–42.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. (2023). LLaMA: Open and Eficient Foundation Language Models. CoRR, abs/2302.13971.

van Laarhoven, T. (2017). L2 Regularization versus Batch and Weight Normalization. CoRR, abs/1706.05350.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is All you Need. In NIPS, pages 5998–6008.

Villar, S., Yao, W., Hogg, D. W., Blum-Smith, B., and Dumitrascu, B. (2023). Dimensionless machine learning: Imposing exact units equivariance. J. Mach. Learn. Res., 24:109:1–109:32.

Wang, H., Ma, S., Dong, L., Huang, S., Zhang, D., and Wei, F. (2024). DeepNet: Scaling Transformers to 1,000 Layers. IEEE Trans. Pattern Anal. Mach. Intell., 46(10):6761–6774.

Wang, S., Sankaran, S., Wang, H., and Perdikaris, P. (2023). An Expert’s Guide to Training Physicsinformed Neural Networks. CoRR, abs/2308.08468.

Wang, S., Teng, Y., and Perdikaris, P. (2021). Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks. SIAM J. Sci. Comput., 43(5):A3055–A3081.

Wang, S., Yu, X., and Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. J. Comput. Phys., 449:110768.

Wortsman, M., Liu, P. J., Xiao, L., Everett, K. E., Alemi, A. A., Adlam, B., Co-Reyes, J. D., Gur, I., Kumar, A., Novak, R., Pennington, J., Sohl-Dickstein, J., Xu, K., Lee, J., Gilmer, J., and Kornblith, S. (2024). Small-scale proxies for large-scale Transformer training instabilities. In ICLR.

Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S. (2023). SmoothQuant: Accurate and Eficient Post-Training Quantization for Large Language Models. In ICML, volume 202, pages 38087–38099.

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. (2024). Eficient Streaming Language Models with Attention Sinks. In ICLR.

Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu, T. (2020). On Layer Normalization in the Transformer Architecture. In ICML, volume 119, pages 10524– 10533.

Xu, J. and Durrett, G. (2018). Spherical Latent Spaces for Stable Variational Autoencoders. In EMNLP, pages 4503–4513.

Yang, G. and Hu, E. J. (2021). Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks. In ICML, volume 139, pages 11727–11737.

Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., and Gao, J. (2021). Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. In NeurIPS, pages 17084–17097.

Yang, G., Yu, D., Zhu, C., and Hayou, S. (2024). Tensor Programs VI: Feature Learning in Infinite Depth Neural Networks. In ICLR.

You, Y., Gitman, I., and Ginsburg, B. (2017). Scaling SGD Batch Size to 32K for ImageNet Training. CoRR, abs/1708.03888.

You, Y., Li, J., Reddi, S. J., Hseu, J., Kumar, S., Bhojanapalli, S., Song, X., Demmel, J., Keutzer, K., and Hsieh, C. (2020). Large Batch Optimization for Deep Learning: Training BERT in 76 minutes. In ICLR.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. (2019). HellaSwag: Can a Machine Really Finish Your Sentence? In ACL (1), pages 4791–4800.

Zhai, S., Likhomanenko, T., Littwin, E., Busbridge, D., Ramapuram, J., Zhang, Y., Gu, J., and Susskind, J. M. (2023). Stabilizing Transformer Training by Preventing Attention Entropy Collapse. In ICML, volume 202, pages 40770–40803.

Zhang, B. and Sennrich, R. (2019). Root Mean Square Layer Normalization. In NeurIPS, pages 12360– 12371.

Zhang, H., Dauphin, Y. N., and Ma, T. (2019). Fixup Initialization: Residual Learning Without Normalization. In ICLR.

Zhao, B., Dehmamy, N., Walters, R., and Yu, R. (2022). Symmetry Teleportation for Accelerated Optimization. In NeurIPS.

Zhu, J., Chen, X., He, K., LeCun, Y., and Liu, Z. (2025). Transformers without Normalization. In CVPR, pages 14901–14911.

## CHECKLIST

1. For all models and algorithms presented, check if you include:

(a) A clear description of the mathematical setting, assumptions, algorithm, and/or model. [Yes] See Secs. 2–4.

(b) An analysis of the properties and complexity (time, space, sample size) of any algorithm. [Yes] See Appendices E and I.3.

(c) (Optional) Anonymized source code, with specification of all dependencies, including external libraries. [Yes] See Appendix H.5.

2. For any theoretical claim, check if you include:

(a) Statements of the full set of assumptions of all theoretical results. [Yes] See Sec. 2 and Appendices E and I.2.

(b) Complete proofs of all theoretical results. [Yes] See Appendices F and G.

(c) Clear explanations of any assumptions. [Yes] See Sec. 2 and Appendices E–F.

3. For all figures and tables that present empirical results, check if you include:

(a) The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). [Yes] See Appendix H.5.

(b) All the training details (e.g., data splits, hyperparameters, how they were chosen). [Yes] See Appendix H.

(c) A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). [Yes] See Appendix H.4 and the captions.

(d) A description of the computing infrastructure used. (e.g., type of GPUs, internal cluster, or cloud provider). [Yes] See Appendix H.1.

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:

(a) Citations of the creator If your work uses existing assets. [Yes] See Appendix H.

(b) The license information of the assets, if applicable. [No] Licenses are not listed.

(c) New assets either in the supplemental material or as a URL, if applicable. [No] No separate asset release.

(d) Information about consent from data providers/curators. [Not Applicable] Existing public datasets only.

(e) Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. [No] Not discussed in this paper.

5. If you used crowdsourcing or conducted research with human subjects, check if you include:

(a) The full text of instructions given to participants and screenshots. [Not Applicable] No crowdsourcing or human subjects.

(b) Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. [Not Applicable] No human subjects.

(c) The estimated hourly wage paid to participants and the total amount spent on participant compensation. [Not Applicable] No participants.

Neural Dimensional Analysis: Foundations of Dimensionless and Scale-Consistent Computation Appendix

## APPENDIX TABLE OF CONTENTS

A Notation 14   
B Limitations and scope 14   
C Related work 15   
D Discussion 17   
E Additional theoretical statements 18   
F Derivations 26   
G Proofs 30   
H Experimental details 37   
I Full results 44

Table 3: List of notation. Recurring symbols are grouped by the part of the calculus that introduces each symbol. A bar denotes a dimensionless canonical coordinate, and a star denotes a target value of a diagnostic.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td colspan="2">Units and the rescaling group</td></tr><tr><td> $G$ </td><td>declared rescaling group  $( \mathbb { R } _ { + } ) ^ { K }$  of the graph</td></tr><tr><td> $g , c$ </td><td>group element; a single positive rescaling factor</td></tr><tr><td> $\rho _ { v }$ </td><td>neural unit of graph node  $v ,$  a representation of  $G$ </td></tr><tr><td> $\chi _ { u }$ </td><td>scalar character  $\begin{array} { r } { { \bar { g ^ { \prime } } } \mapsto \prod _ { k } g _ { k } ^ { u _ { k } } } \end{array}$  with exponent u</td></tr><tr><td> $u ( x )$ </td><td>unit type of  $x ; u ( x ) = 1 { \overset { \cdot } { } }$  means dimensionless</td></tr><tr><td> $U _ { \mathrm { r e s } } , U _ { \theta }$ </td><td>residual-stream unit and parameter unit</td></tr><tr><td> $s _ { x }$ </td><td>empirical scale  $\operatorname { o f } x ,$  with  $u ( s _ { x } ) = u ( x )$ </td></tr><tr><td> $\mathrm { R M S } ( x )$ </td><td>RMS over a shared-unit block</td></tr><tr><td> $\delta _ { v } , K _ { v u }$ </td><td>local equivariance defect and Lipschitz gain at node v</td></tr><tr><td colspan="2">Canonical gauge and architecture</td></tr><tr><td> $d , d _ { h } , L$ </td><td></td></tr><tr><td> $h _ { \ell } , r _ { \ell }$ </td><td>model width, head dimension, number of blocks</td></tr><tr><td> $s _ { \ell } , h _ { \ell }$ </td><td>residual stream and branch contribution at block l</td></tr><tr><td> ${ \mathcal { C } } , { \mathcal { C } } _ { 0 }$ </td><td>scale state and canonical coordinate,  $h _ { \ell } = s _ { \ell } \bar { h } _ { \ell }$ </td></tr><tr><td> $\mathbb { S } , d _ { \mathbb { S } }$ </td><td>canonicalizer and zero-floor limit  $x / \mathrm { R M S } ( x )$ </td></tr><tr><td></td><td>RMS-one canonical section and chord metric on that section</td></tr><tr><td> $P _ { \bar { h } } ^ { \perp }$ </td><td>projector removing the component along  $\bar { h }$ </td></tr><tr><td> $\alpha \ell , \alpha ^ { \star }$ </td><td>dimensionless residual step and target step size  $1 / \sqrt { 2 L }$ </td></tr><tr><td> $c _ { \ell }$ </td><td>cosine similarity between the branch contribution  $r _ { \ell }$  and stream  $h _ { \ell }$ </td></tr><tr><td> $\tau _ { \ell , h }$ </td><td>attention temperature  $\sqrt { d _ { h } } s _ { Q } s _ { K }$  , a logit nondimensionalizer</td></tr><tr><td> $s Q , s _ { K }$   $A , N , \mathcal { Q }$ </td><td>per-head query and key coordinate scales</td></tr><tr><td></td><td>unit-exponent matrix, basis of ker A, complete invariant</td></tr><tr><td colspan="2">Losses, gradients, and optimizer dimensionless objective and constituent ith loss term</td></tr><tr><td> ${ \mathcal { L } } , { \mathcal { L } } _ { i }$   $\lambda _ { i }$ </td><td></td></tr><tr><td></td><td>loss weight, a preference  $\alpha _ { i }$  over a characteristic scale  $S _ { i }$ </td></tr><tr><td> $\nabla _ { x } \mathcal { L }$ </td><td>gradient of the loss, carrying the dual unit  $u ( x ) ^ { - 1 }$ </td></tr><tr><td> $\Pi _ { x }$ </td><td>primal–dual RMS product  $\breve { \mathrm { R M S } } ( x ) \mathrm { R M S } ( \nabla _ { x } \dot { \mathcal { L } } )$ </td></tr><tr><td> $\theta , \Delta \theta$ </td><td>parameters and optimizer displacement</td></tr><tr><td> $P _ { \theta , \eta }$ </td><td>preconditioner with  $u ( P _ { \theta } ) = U _ { \theta } ^ { 2 }$  ; dimensionless step</td></tr><tr><td> $p ( r )$ </td><td>radial preconditioner; gauge-covariant iff  $p ( r ) \propto r ^ { 2 }$ </td></tr><tr><td> $\vartheta$ </td><td>rotation angle of the exact great-circle quotient update</td></tr><tr><td> $\rho _ { b }$ </td><td>relative update of parameter block b</td></tr><tr><td colspan="2">Diagnostics, perturbation, and quantization</td></tr><tr><td> $\mathrm { U V I _ { r e s } , U V I _ { a t t n } }$ </td><td>log deviation of  $\alpha _ { \ell }$  and of logit variance from target</td></tr><tr><td> $\mathrm { U V I } _ { \mathrm { a c t } } , \mathrm { U V I } _ { \mathrm { g r a d } }$ </td><td>depth dispersion of preactivation scale and of  $\Pi _ { \ell }$ </td></tr><tr><td> $\mathrm { U V I _ { o p t } }$ </td><td>cross-block imbalance of the relative updates  $\rho _ { b }$ </td></tr><tr><td> $e , a , \rho$ </td><td>perturbation, relative radial coefficient, and tangential size</td></tr><tr><td> $b , q$ </td><td>bit width and half-range  $2 ^ { b - 1 } - 1$  of the integer grid</td></tr><tr><td> $\Delta , Q _ { \Delta }$ </td><td>shared quantization step and associated absmax quantizer</td></tr><tr><td> $\kappa _ { \mathrm { s c a l e } }$ </td><td>cross-token spread maxt  ${ \boldsymbol { r } } _ { t } /$  mint  $r _ { t }$  of the stream RMS  $r _ { t }$ </td></tr><tr><td> $\kappa _ { \mathrm { a m p } }$ </td><td>within-vector amplitude ratio maxt  $\| h _ { t } \| _ { \infty } / r _ { t }$ </td></tr><tr><td></td><td></td></tr><tr><td> $\kappa _ { \mathrm { d e p t h } }$ </td><td>span of the mean token RMS across block inputs</td></tr><tr><td> $\mathfrak { E } _ { q } ( \Gamma )$ </td><td>worst-case quotient error at scale spread at most Γ</td></tr><tr><td> $\kappa$ </td><td>rescaling of the data&#x27;s unit system in the PINN study</td></tr></table>

## A NOTATION

Table 3 collects the symbols that recur across the paper. The subscripts $\ell , h , t , i ,$ and b index blocks, attention heads, tokens, loss terms, and parameter blocks, respectively.

## B LIMITATIONS AND SCOPE

The theory applies to the declared rescaling group, and the experiments test the theory’s consequences in controlled settings. The language-model study spans decoder-only Transformers from 51M to 1.0B parameters trained on FineWeb-Edu under one shared recipe, with the main controlled comparison at 124M and 2.75B tokens, and the generality probes cover two vision benchmarks and two PINN problems. Generality beyond these datasets and model families remains to be tested. The larger models support the deployment claim of Sec. 5.5. Compute-optimal scaling requires a separate study. Each setting compares typed and untyped models at a matched budget. Extending these comparisons to further corpora, scales, and architectures is natural future work.

## C RELATED WORK

## C.1 Normalization, residual scaling, and attention temperature

Batch normalization established the practice of restandardizing internal activations (Iofe and Szegedy, 2015), and LayerNorm and RMSNorm carry that practice into the Transformer (Ba et al., 2016; Zhang and Sennrich 2019), where Pre-LN placement stabilizes depth (Xiong et al., 2020) and NormFormer adds post-attention and mid-MLP normalization with head scaling (Shleifer et al., 2021). Residual connections make depth trainable in the first place (He et al., 2016), and DeepNet, ReZero, SkipInit, LayerScale, and Fixup control residual-branch magnitude to sustain that trainability (Wang et al., 2024; Bachlechner et al., 2021; De and Smith, 2020; Touvron et al., 2021b; Zhang et al., 2019). A complementary line removes the normalizer instead of tuning the normalizer’s placement, through a learnable elementwise squashing (Zhu et al., 2025) or through analytically scaled residua blocks with adaptive gradient clipping (Brock et al., 2021).

The same accounting governs the attention interface. The $1 / \sqrt { d _ { h } }$ factor is the $s _ { Q } = s _ { K } = 1$ case of the logit nondimensionalizer $\tau = \sqrt { d _ { h } } s _ { Q } s _ { K }$ (Vaswani et al., 2017), QK-Norm and its use in a 22-billion-parameter Vision Transformer (ViT) make softmax inputs dimensionless by construction (Henry et al., 2020; Dehghani et al., 2023), σReparam bounds logit growth through weight reparameterization (Zhai et al., 2023), and controlled small-scale reproductions trace large-model training instabilities to attention-logit growth exposed through learning-rate sensitivity (Wortsman et al., 2024).

NDA interprets these mechanisms as scale controls at diferent interfaces. Normalization nondimensionalizes from $U _ { \mathrm { r e s } }$ to 1. Residual scaling controls the dimensionless step size $\alpha _ { \ell } ,$ , and attention scaling implements the softmax rule of Sec. 2.2. In a normalizer-free block, a bounded nonlinearity and a calibrated input scale can provide the corresponding nondimensionalization without a measured normalization statistic.

## C.2 Optimizers, parameterizations, and typed optimization

Adaptive moment estimation (Adam) normalizes per-coordinate gradient scale (Kingma and Ba, 2015), and layer-wise adaptive moments for batch training (LAMB) and layer-wise adaptive rate scaling (LARS) apply relative trust-ratio updates per block, which restore primal units in our terminology (You et al., 2020, 2017). $\mu \mathrm { P }$ chooses width-dependent multipliers to keep the update efect invariant across widths and enable zero-shot hyperparameter transfer (Yang and Hu, 2021; Yang et al., 2021). Depth-wise $\mu \mathrm { P }$ scales residual-branch contributions as $1 / \sqrt { L }$ to enable the same transfer across depth (Yang et al., 2024). Unit scaling chooses scales for low-precision formats (Blake et al., 2023), generalizing the loss-scaling constant of mixed-precision training, which is an inserted unit conversion that keeps half-precision gradients representable (Micikevicius et al., 2018), and unit-scaled $\mu \mathrm { P }$ composes the two disciplines (Blake et al., 2025).

A parallel line makes the metric on weight space explicit. The modular norm equips a network with a norm under which well-scaled updates transfer across width and depth (Large et al., 2024), and modular duality with the eficient Muon realization casts the optimizer as a duality map that sends dual gradient vectors to primal update directions (Bernstein and Newhouse, 2025; Jordan et al., 2024).

NDA uses the same primal–dual relation, in which a gradient carries the inverse unit of the diferentiated variable and an optimizer restores the primal unit. NDA expresses this primal–dual relation for scalar rescaling groups $G = ( \mathbb { R } _ { + } ) ^ { K }$ as a typing discipline covering the full computation graph, including objectives and diagnostics. A modular norm can supply a shape-aware duality map, while NDA further requires the complete displacement, including the scalar step or reference scale of the update, to obey Prop. 2. Combining Muon with a scalar scale that obeys the required transformation law can make the full update unit-covariant. Our optimizer contribution is the covariance criterion, while the proposed constructions are the UVI diagnostics and the canonical architecture.

## C.3 Rescaling symmetries and normalized architectures

A classical line studies weight rescalings that leave a network’s function invariant. For positive-homogeneous activations, paired rescalings of incoming and outgoing weights form a symmetry group. Path-norm methods and path-normalized variants of stochastic gradient descent (SGD) optimize in the induced quotient (Neyshabur et al., 2015), and the same reparameterizations show that sharpness measures are not invariant (Dinh et al., 2017), imprint conservation laws on gradient flow (Kunin et al., 2021), and define the orbits used by teleportation methods (Zhao et al., 2022).

Weight normalization exploits the same freedom constructively by splitting each weight vector into a direction and an explicit magnitude (Salimans and Kingma, 2016), which is the radial coordinate that becomes irrelevant under the downstream-nondimensionalization hypothesis of Prop. 6. A normalizer creates this freedom by making the upstream weight block scale-invariant. The scale-invariant weight block’s norm then acts only through an efective learning rate (van Laarhoven, 2017; Hofer et al., 2018; Arora et al., 2019), which $\operatorname { A p p . }$ E.3 recovers as the $p \equiv 1$ case of Prop. 3, and weight decay on such a block is equivalent to a learning-rate schedule (Li and Arora, 2020). NDA follows how activation rescalings propagate through the graph. In an untyped graph, a rescaling can change the computed function. The typing rules identify the interfaces that absorb or propagate the rescaling. The two views meet at interfaces, because a weight-space rescaling of adjacent layers induces an activation rescaling between those layers.

The same normalized gauge also appears at the architecture level. nGPT constrains hidden states and weight rows to the unit hypersphere, with large reported convergence speedups (Loshchilov et al., 2025), following earlier work that placed learned latent representations on the unit hypersphere (Xu and Durrett, 2018), and the magnitude-preserving difusion layers of Karras et al. (2024) hold activations and weights on fixed-norm shells throughout the backbone, an independent cross-domain instance of computing only in a normalized coordinate.

GCT shares the normalized-state mechanics but difers in derivation and content. GCT follows from a typing principle that also produces UVI diagnostics, UCT corrective modules, and a treatment of losses and optimizers, and the architecture adds an orthogonalizing projection $P ^ { \perp }$ motivated by the residual-drift analysis, a learnable dimensionless step initialized at $1 / \sqrt { 2 L }$ , and dimensionless gates. For unitful outputs such as regression and PINN predictions, GCT retains an explicit scale state to restore the output magnitude and units. The component ablations evaluate proposal canonicalization and the orthogonal projection (Tables 1 and 6).

## C.4 Quantization and activation-scale management

Quantization research documents substantial variation in the scale of internal language-model activations. Integer inference represents tensors as scales times integer grids (Jacob et al., 2018; Nagel et al., 2021), connecting integer inference to the factorization in Sec. 4. Relevant phenomena include outlier features (Dettmers et al., 2022), massive activations (Sun et al., 2024), and attention sinks (Xiao et al., 2024). These observations motivate examining both cross-token scale variation and within-vector concentration.

Methods for low-bit inference include mixed-precision decomposition (Dettmers et al., 2022), error-compensated weight rounding (Frantar et al., 2023), activation-scale migration (Xiao et al., 2023), activation-aware weight quantization (Lin et al., 2024), and rotations that reduce coordinate outliers (Ashkboos et al., 2024). Logit soft-capping provides a complementary architectural control at attention and output interfaces (Team, 2024). These methods address diferent aspects of the scale and precision problem.

An architectural approach is taken by Bondarenko et al. (2023), who associate activation outliers with attention heads implementing a no-op and use clipped softmax and gated attention to improve int8 quantization.

NDA controls cross-token scale variation by constructing a canonical residual stream. Within-vector outliers remain a separate factor, $\kappa _ { \mathrm { a m p } } ,$ in Prop. 5. Thm. 7 shows that the cross-token penalty persists in a worst-case functional bound after radial rejection. The experiments compare this architectural control with per-token and per-channel stream scales and SmoothQuant migration. Normalized streams reduce the need for fine-grained scale management in these settings while retaining the ordinary per-tensor quantization cost.

## C.5 Typed objectives and units of physical data

Uncertainty weighting (Kendall et al., 2018), gradient normalization (Chen et al., 2018), and gradient-statistics annealing for PINNs (Wang et al., 2021) adapt loss weights during training, and practitioner guides for PINNs also recommend nondimensionalizing the problem before training (Wang et al., 2023). NDA instead types the objective. A weight $\lambda _ { i }$ factorizes into a dimensionless preference $\alpha _ { i }$ and an inverse-unit converter $1 / S _ { i }$ with $u ( S _ { i } ) = u ( \mathcal { L } _ { i } )$ . Thm. 2 shows that every invariant linear rule whose weights depend on the characteristic scales carries this inverse-unit factor, with the remaining scale dependence confined to dimensionless π groups, and Cor. 1 shows that the preferences are constant within this class when the loss units vary independently. Deriving $S _ { i }$ from observable characteristic scales makes the objective invariant to the data’s unit system without adaptive weighting. Adaptive methods can then adjust dimensionless preferences among the normalized terms.

This typing connects to the broader use of units in scientific machine learning, where dimensional analysis constrains learned models of physical systems (Villar et al., 2023), Buckingham–π style reductions are classical (Buckingham, 1914), and recent work learns the dimensionless groups from data (Bakarji et al., 2022). Outside machine learning, units-of-measure type systems check the dimensional consistency of a program at compile time and infer unit-polymorphic types (Kennedy, 2009). All of these lines attach externally declared physical units to inputs, outputs, and program variables. NDA instead assigns units to internal tensors as transformation laws under internal rescalings, a gauge structure of the computation graph that applies when no physical unit exists. The gauge terminology follows geometric deep learning, where equivariance is stated relative to a choice of loca frame rather than to a single global symmetry (Cohen et al., 2019).

## C.6 Positioning summary

The related methods address complementary parts of scale management. Parameterization methods such as $\mu \mathrm { P }$ and unit scaling prescribe initialization, multipliers, and update scales (Yang and Hu, 2021; Yang et al., 2024; Blake et al., 2023). Normalized architectures control activation magnitude (Loshchilov et al., 2025; Karras et al., 2024). Weight-space symmetry analyses describe function-preserving reparameterizations (Neyshabur et al., 2015; Dinh et al., 2017; Kunin et al., 2021). Modular norms and duality maps organize optimizer updates (Large et al., 2024; Bernstein and Newhouse, 2025). Loss-balancing and quantization methods address scale in objectives and finite-precision computation, respectively (Kendall et al., 2018; Chen et al., 2018; Dettmers et al., 2022; Xiao et al., 2023). Units-of-measure type systems and units-equivariant learning provide the closest conceptual parallels for externally declared units (Kennedy, 2009; Villar et al., 2023).

Our NDA framework unifies these perspectives in one graph-level description. Units are transformation laws under declared internal rescalings. This definition lets NDA apply to tensors without physical units. Forward consistency determines reverse-mode units (Prop. 1), while UVI diagnostics and the parameter-gauge audit make the declarations testable on trained networks. Through NDA, we also connect training-time scale control to PTQ. Our contribution is this unified treatment and the resulting diagnostics and constructions.

## D DISCUSSION

Our NDA framework connects a declared rescaling group to local computation rules, diagnostics, and architectural choices. The following discussion explains how to interpret these declarations and how the resulting constructions relate to UCT, nGPT, parameterization methods, and optimizers.

The status of a unit declaration. A unit assignment specifies a rescaling group and the intended transformation of each quantity. UVI values describe deviations from the corresponding reference regime. The same declaration organizes the scale-control mechanisms, identifies mismatches, and guides the construction of GCT. The diagnostics characterize that regime, while the relation between UVI values and task quality is an empirical question.

Role of UCT in the comparison. UCT tests local application of the typing rules through three corrective modules: an RMS-matching residual adapter, an empirical attention temperature, and normalized gates. At 124M, UCT’s per-site tensor int8 loss increase is +0.007 nats (Table 10), but UCT’s clean loss of 3.256 is the highest among the evaluated architectures (Table 4). UCT also loses invariance under the strongest down-scaling intervention. The hidden state shrinks, but the small constants in UCT’s normalization denominators and the stored UCT scale estimates remain fixed. Dividing by these unchanged quantities no longer cancels the change in magnitude. This comparison distinguishes the benefits of local converters from the guarantees of the canonical construction.

Relation to nGPT. Because GCT and nGPT both use normalized states, we evaluate $\mathrm { n G P T }$ as a direct baseline across the sweep, depth, perturbation, quantization, and vision studies under the identical recipe. The language-model comparison between GCT and nGPT contrasts GCT’s orthogonalizing projection, proposal canonicalization, and scalar step initialized at $1 / \sqrt { 2 L }$ with $\mathrm { n G P T ^ { \circ } s }$ per-dimension eigen learning rate. This language-model comparison concerns dimensionless computation, for which the implementation of GCT omits $s _ { \ell } ,$ as stated in Sec. 4. Under the fixed-budget depth-transfer protocol at L = 48 and 400M tokens, GCT reaches 3.903, compared with 4.243 for nGPT (Table 1). Under 1B int4 per-token residual-stream quantization, the two architectures separate more strongly (Table 2), as discussed in Sec. 5.5. Our contribution beyond nGPT is the derivation through typing rules that also cover losses, gradients, and optimizers, the UVI diagnostic layer, and the local-correction and canonical-design variants.

Relation to $\mu \mathbf { P } .$ µP-style width transfer is orthogonal to and composable with NDA typing, and depth-wise $\mu \mathrm { P }$ (Yang et al., 2024) plays the analogous role along depth. The width experiments of Sec. I.4 test this composition directly.

Optimizers and open extensions. The optimizer contribution is a covariance law for complete updates. Modular-duality optimizers (Bernstein and Newhouse, 2025; Jordan et al., 2024) realize the shape-aware dualitymap component at scale, but satisfying NDA’s rescaling law additionally requires a typed scalar step or reference scale. The Muon pairing experiment in Sec. I.4 evaluates compatibility with GCT under a shared training recipe. On the architecture side, the zero-floor construction fixes each token’s pre-retraction RMS ratio to $| \alpha _ { \ell } | ,$ and the chord displacement of the canonical state transition is likewise token-independent. Per-token dimensionless gates on α are a natural extension for tasks requiring residual step sizes that vary strongly across tokens. $\mathrm { A p p }$ . B states the scope of the empirical study.

## E ADDITIONAL THEORETICAL STATEMENTS

## E.1 Quantitative local-to-global typing stability

Real graphs also contain sources of inexact equivariance, including finite normalization floors, approximate kernels, and quantizers. Thm. 1 bounds the output error by propagating each local equivariance defect through the sum of the downstream path gains from the defective node. For a frozen canonicalization floor, the local defect is explicitly $O ( \epsilon / \mathrm { R M S } ( x ) ^ { 2 } )$ , whereas a correctly co-transforming floor has zero defect. The theorem gives quantitative control for an arbitrary DAG by separating a node’s own typing defect from the downstream gains that can amplify that defect.

Fix $g \in G$ and one evaluation $x : = ( x _ { v } ) _ { \ast }$ of a finite DAG with source set $s$ and output node o. Equip every node space with a norm $\| \cdot \| _ { v } .$ A nonsource node v with parents $\mathrm { p a } ( v )$ computes $x _ { v } : = f _ { v } ( ( x _ { u } ) _ { u \in \mathrm { p a } ( v ) } )$ . Let $\widetilde { x }$ be the graph evaluation obtained by replacing every source value $x _ { s }$ by $\rho _ { s } ( g ) x _ { s }$ . Suppose that for each nonsource v there are constants $K _ { v u } \ge 0$ such that

$$
\| f _ { v } ( z ) - f _ { v } ( z ^ { \prime } ) \| _ { v } \leq \sum _ { u \in \mathrm { p a } ( v ) } K _ { v u } \| z _ { u } - z _ { u } ^ { \prime } \| _ { u } .\tag{E.1}
$$

This bound need only hold on a set containing the two parent tuples used in the proof, $z : = ( \widetilde { x } _ { u } ) _ { u \in \mathrm { p a } ( v ) }$ and $z ^ { \prime } : = ( \rho _ { u } ( g ) x _ { u } ) _ { u \in \mathrm { p a } ( v ) }$ . A global Lipschitz bound is suficient but not required. Define the local equivariance defect

$$
\begin{array} { r } { \delta _ { v } ( g , x ) : = \| f _ { v } ( ( \rho _ { u } ( g ) x _ { u } ) _ { u \in \mathrm { p a } ( v ) } ) - \rho _ { v } ( g ) x _ { v } \| _ { v } . } \end{array}\tag{E.2}
$$

Let $\mathcal { P } ( v , o )$ be the set of directed paths from v to o. This set includes the empty path when $v = o$ . Then Theorem 1 (Quantitative local-to-global typing stability). Under the preceding setup,

$$
\| \widetilde x _ { o } - \rho _ { o } ( g ) x _ { o } \| _ { o } \le \sum _ { v \notin { \mathcal S } } \delta _ { v } ( g , x ) \sum _ { p \in { \mathcal P } ( v , o ) } \prod _ { ( a  b ) \in p } K _ { b a } .\tag{E.3}
$$

The proof is given in App. G.2.

This theorem also quantifies the finite-floor implementation error in Sec. 4. For the floored canonicalizer $\mathcal { C } _ { \epsilon } ( x ) : = x / \sqrt { \mathrm { R M S } ( x ) ^ { 2 } + \epsilon }$ with a fixed numerical $\epsilon \geq 0 ,$ a correctly typed floor transforms as $\epsilon \mapsto c ^ { 2 } \epsilon$ and makes the local equivariance defect exactly zero. A frozen implementation floor instead leaves an explicit $O ( \epsilon / r ^ { 2 } )$ local defect at $r : = \mathrm { R M S } ( x ) > 0$ , bounded in $\mathrm { E q . ~ F . 1 }$ 1 and derived in $\mathrm { A p p . ~ F . 1 }$ , which $\operatorname { E q . }$ E.3 propagates to the output. The bound separates an architectural mismatch from the numerical error caused by a frozen floor.

## E.2 Invariant loss scalarization

Characteristic loss scales. In Eq. 4, each factor $1 / S _ { i }$ converts the corresponding loss to a dimensionless value. The converter can be derived from characteristic scales of the data and operators. For PDE residuals, $S _ { i }$ must account for the operator’s output unit, including the units introduced by diferentiation. For example, a squared first-order-in-time residual loss uses the characteristic loss scale $S _ { i } : = ( s _ { u } / t ^ { * } ) ^ { 2 }$

Thm. 2 characterizes all loss weights that preserve this invariance within the class of linear objectives whose weights depend only on the characteristic scales. Every such weight must contain the factor $1 / S _ { i }$ , with any remaining scale dependence restricted to dimensionless combinations of scales called Buckingham–π groups. When the loss units vary independently, Cor. 1 fixes the preferences to constants and gives $w _ { i } ( S ) = { \alpha _ { i } } / { S _ { i } }$

To characterize every invariant linear scalarization whose weights depend only on the declared characteristic scales, let $G : = ( \mathbb { R } _ { + } ) ^ { K }$ . Write $g : = e ^ { t }$ coordinatewise, let $a _ { i } \in \mathbb { R } ^ { K }$ be the character exponent of loss i, let $A \in \mathbb { R } ^ { K \times n }$ m be the matrix with columns $a _ { 1 } , \ldots , a _ { m }$ , and set $D ( t ) : = \mathrm { d i a g } ( e ^ { a _ { 1 } ^ { \top } t } , \dots , e ^ { a _ { m } ^ { \top } t } )$ . Suppose the loss vector $\ell \in \mathbb { R } ^ { m }$ and the corresponding positive characteristic-scale vector $S \in \mathbb { R } _ { + } ^ { m }$ both transform as $( \ell , S ) \mapsto ( D ( t ) \ell , D ( t ) S )$ Consider an arbitrary scale-dependent linear scalarization $\begin{array} { r } { J _ { S } ( \ell ) : = \sum _ { i = 1 } ^ { m } w _ { i } ( S ) \ell _ { i } } \end{array}$

Theorem 2 (Characterization of characteristic-scale-dependent invariant loss scalarization). Let $s : = \operatorname { r a n k } A$ and let the columns of $\boldsymbol { N } \in \mathbb { R } ^ { m \times ( m - s ) }$ be a basis of ker A. Under the preceding setup, $J _ { S }$ is unit-system invariant for every loss vector,

$$
J _ { D ( t ) S } ( D ( t ) \ell ) = J _ { S } ( \ell ) , \ f o r \ a l l \ S \in \mathbb { R } _ { + } ^ { m } , \ell \in \mathbb { R } ^ { m } , t \in \mathbb { R } ^ { K } ,
$$

$i f$ and only if the scalarization weights have the form

$$
w _ { i } ( S ) = \frac { 1 } { S _ { i } } \psi _ { i } ( N ^ { \top } \log S ) , \qquad i = 1 , \dots , m ,\tag{E.4}
$$

for arbitrary functions $\psi _ { i } : \mathbb { R } ^ { m - s }  \mathbb { R }$ , where log $S$ is componentwise. The proof is given in $A p p$ . G.3.

The quotient coordinates $N ^ { \top }$ log S form a basis for the logarithms of all dimensionless monomials $\Pi _ { j } S _ { j } ^ { n _ { j } }$ with $n \in$ ker A.

Any scale dependence of a unit-invariant adaptive preference must pass through these dimensionless π groups.

Corollary 1 (Independent loss units force inverse-scale weights). Under the assumptions of Thm. 2, if the m loss-unit characters are independent, meaning rank $A = m$ , the quotient has no nonconstant dimensionless scale group and

$$
w _ { i } ( S ) = { \alpha _ { i } } / { S _ { i } }\tag{E.5}
$$

with constants $\alpha _ { i }$ is the unique invariant linear rule in this S-dependent class. The proof is given in App. G.3.

## E.3 Optimizer covariance and quotient geometry

Relative optimizer updates. Adam’s $m _ { t } / ( \sqrt { v _ { t } } + \epsilon _ { g } )$ is dimensionless when $u ( \epsilon _ { g } ) = U _ { \theta } ^ { - 1 }$ . Multiplying this ratio by a parameter scale restores the update unit, giving the relative form $\Delta \theta = - \eta s _ { \theta } m _ { t } / ( \sqrt { v _ { t } } + \epsilon _ { g } )$ with $u ( s _ { \theta } ) = U _ { \theta }$ . LARS (You et al., 2017) and LAMB (You et al., 2020) implement this relative-scaling idea through trust ratios based on parameter norms. The comparable diagnostic across blocks is the relative update $\rho _ { b } : = \| \Delta \theta _ { b } \| / ( s _ { \theta , b } ^ { \mathrm { r e f } } + \epsilon _ { \theta , b } )$

These rules enable us to distinguish a change of parameter units from a change in the optimization trajectory. For an invertible parameter rescaling R, displacement covariance is equivalent to the preconditioner law $\dot { P } ^ { g } = \dot { R } P R ^ { \top }$ (Prop. 2). On a scale-invariant block, Prop. 3 shows that the induced quotient update is an exact great-circle step, and Thm. 3 and Cor. 2 show that $p ( r ) \propto r ^ { 2 }$ is the unique isotropic radial preconditioner that removes dependence on the arbitrary parameter gauge. In this setting, the squared-scale factor makes the update take the same angular step at every rescaling of the parameter vector.

To make the optimizer rule precise, let $R \in G L ( \Theta )$ be an invertible rescaling of a finite-dimensional real innerproduct parameter space. Suppose corresponding gradients satisfy $q ^ { g } = R ^ { - \top } q ,$ as the gradients do when a scalar objective is invariant under $\theta ^ { g } = R \theta$ . Consider preconditioned displacements u $: = - \eta P q$ and $u ^ { g } : = - \eta P ^ { g } q ^ { g }$ with the same dimensionless learning rate η.

Proposition 2 (Covariance law for preconditioned updates). Under the preceding setup, $u ^ { g } = R u$ for every q if and only if

$$
P ^ { g } = R P R ^ { \top } .\tag{E.6}
$$

The proof is given in App. $G . 4 .$

Carrying squared parameter units is both necessary and suficient for the preconditioner to produce a covariant displacement.

The same distinction applies to nonlinear direction maps. A map D from gradients to displacements is unitcovariant precisely when the transformed direction map satisfies $\mathcal { D } ^ { g } ( R ^ { - \top } q ) = R \mathcal { D } ( q )$ . A duality map satisfies this law only when the resulting full displacement, including any scalar step, co-transforms with the parameters. For example, under a scalar rescaling $R = c I .$ , a scale-zero-homogeneous orthogonalization obeys $\mathcal { D } ( c ^ { - 1 } q ) = \mathcal { D } ( q )$ whereas displacement covariance requires $c D ( q )$ . Such an update becomes typed only after supplying an explici primal-scale factor or an equivalently transformed scalar step.

Two immediate consequences clarify the operator law. Ordinary Euclidean SGD with a fixed common learning rate satisfies that law for every gradient only when $R R ^ { \top } = I$ . A scalar parameter rescaling $R = c I$ instead requires the transformed learning rate $\eta ^ { g } = c ^ { 2 } \eta$

The preconditioner law in Prop. 2 is an operator statement. For the common case of a scale-invariant parameter block and a scalar radial preconditioner, the following proposition and theorem show that the quotient update is an exact geodesic step. These results also show that the squared parameter scale is the unique radial scalar correction that makes that step independent of the arbitrary parameter gauge.

Let V be a finite-dimensional real inner-product space of dimension at least two and let $J : V \setminus \{ 0 \}  \mathbb { R }$ be continuously diferentiable and scale-invariant, $J ( c w ) = J ( w )$ for every $c > 0$ . Writing $g ( w ) : = \nabla J ( w )$ diferentiation with respect to w and to c gives

$$
g ( c w ) = c ^ { - 1 } g ( w ) , \qquad \langle w , g ( w ) \rangle = 0 .\tag{E.7}
$$

Fix $a > 0 , \eta > 0$ , and a scalar radial preconditioner $p : ( 0 , \infty ) \to ( 0 , \infty )$ . Writing $r : = \| w \|$ , consider

$$
w ^ { + } : = a w - \eta p ( r ) g ( w ) .\tag{E.8}
$$

Proposition 3 (Exact geometry of a radial quotient update). Under the preceding setup, at every nonstationary point, with w¯ $: = w / \| w \|$ and $\bar { g } : = g ( w ) / \| g ( w ) \|$ , the canonical update is exactly

$$
\frac { w ^ { + } } { \| w ^ { + } \| } = \cos \vartheta \bar { w } - \sin \vartheta \bar { g } , \qquad \tan \vartheta = \frac { \eta p ( r ) \| g ( w ) \| } { a r } , \quad 0 < \vartheta < \frac { \pi } { 2 } .\tag{E.9}
$$

The proof is given in App. G.5.

The update in $\operatorname { E q . }$ . E.8 follows a great circle on the scale quotient exactly, including at finite step sizes. The same orthogonality separates the complementary raw radial evolution:

$$
\lVert \boldsymbol { w } ^ { + } \rVert ^ { 2 } = a ^ { 2 } r ^ { 2 } + \eta ^ { 2 } p ( r ) ^ { 2 } \lVert \boldsymbol { g } ( \boldsymbol { w } ) \rVert ^ { 2 } .\tag{E.10}
$$

Theorem 3 (Equivalent gauge-covariance criteria for radial updates). Under the assumptions and notation of Prop. 3, the following three conditions are equivalent: For every scale-invariant J, every nonstationary w, and every $c > 0 .$ , the updates from w and cw have the same canonical direction; the raw update is equivariant, $( c w ) ^ { + } = c w ^ { + }$ , for every such $J , w , c ;$ and the preconditioner has quadratic covariance, $p ( c r ) = c ^ { 2 } p ( r )$ for all $c , r > 0$ . The proof is given in App. G.5.

Corollary 2 (Uniqueness of squared-scale radial preconditioning). Under the same assumptions, the equivalent conditions of Thm. 3 hold if and only if there is a constant $\kappa > 0$ such that

$$
p ( r ) = \kappa r ^ { 2 } .\tag{E.11}
$$

For this unique family, tan $\vartheta = ( \eta \kappa / a ) r \Vert g ( w ) \Vert$ is dimensionless and gauge-invariant. The proof is given in App. G.5.

The radial qualifier is essential. For a general direction-dependent scalar preconditioner ${ \widetilde { p } } : V \setminus \{ 0 \} \to ( 0 , \infty )$ gauge equivariance requires and is implied by $\widetilde { p } ( c w ) = c ^ { 2 } \bar { \widetilde { p } } ( w )$ , whose complete family of solutions is $\widetilde { p } ( w ) =$ $\bar { | | \boldsymbol { w } | | ^ { 2 } \boldsymbol { \varphi } ( \boldsymbol { w } / | | \boldsymbol { w } | | ) }$ for an arbitrary positive function $\varphi$ on the unit sphere, as derived in App. G.5. Cor. 2 establishes uniqueness only within isotropic radial rules, where $\varphi$ must be constant.

Ordinary Euclidean gradient descent has $p \equiv 1$ and obeys tan $\vartheta ( c w ) = c ^ { - 2 } \tan \vartheta ( w )$ . The Euclidean gradient step on the quotient depends quadratically on an arbitrary parameter rescaling. This quadratic dependence is the exact form of the efective learning rate that a normalizer induces on the normalizer’s upstream block, where the angular progress of plain gradient descent scales as $\eta / r ^ { 2 }$ and the parameter norm silently retunes the step (Hofe et al., 2018; Arora et al., 2019). Cor. 2 identifies $p ( r ) = \kappa r ^ { 2 }$ as the unique isotropic radial rule that removes this dependence instead of tracking or annealing the resulting efective learning rate.

Together, Prop. 3, Thm. 3, and Cor. 2 make the optimizer rule operational at block level. Writing $d _ { V } : = \dim V$ we have $r \Vert g ( w ) \Vert = d _ { V } \operatorname { R M S } ( w ) \operatorname { R M S } ( g ( w ) ) = d _ { V } \Pi _ { w } .$ , making this invariant a fixed-dimension multiple of the primal–dual RMS product in Eq. 5. The normalized update in Eq. E.9 has exactly the rotation form used by GCT.

## E.4 UVI calibration and depth-uniform stability

Additional UVI definitions. In addition to the residual and attention indices in Sec. 3, we define, with a small dimensionless floor $\epsilon ,$

$$
\mathrm { U V I } _ { \mathrm { a c t } } : = \mathrm { V a r } _ { \ell } [ \log ( \hat { s } _ { a , \ell } + \epsilon ) ] ,\tag{E.12}
$$

$$
\mathrm { U V I } _ { \mathrm { g r a d } } : = \mathrm { V a r } _ { \ell } [ \log ( \Pi _ { \ell } + \epsilon ) ] ,\tag{E.13}
$$

$$
\begin{array} { r } { \mathrm { U V I } _ { \mathrm { o p t } } : = \frac { 1 } { B } \sum _ { b } [ \log \frac { \rho _ { b } + \epsilon } { \rho ^ { \star } + \epsilon } ] ^ { 2 } , } \end{array}\tag{E.14}
$$

where $\hat { s } _ { a , \ell } : = \mathrm { R M S } ( a _ { \ell } ) / s _ { a , \ell } ^ { \mathrm { r e f } }$ is the gate and activation preactivation RMS expressed in a declared reference scale with u $\mathfrak { z } ( s _ { a , \ell } ^ { \mathrm { r e f } } ) = u ( a _ { \ell } )$ . The quantity $\Pi _ { \ell }$ is the primal–dual RMS product from Eq. 5 at the block output, and $\rho _ { b }$ is the relative update of parameter block b. The target $\rho ^ { \star }$ is the median of the $\rho _ { b }$ values in log space. Eq. E.14 measures how far the blocks’ relative updates deviate from this common target.

The activation and gradient indices in $\operatorname { E q }$ . E.12 and Eq. E.13 measure the log-scale dispersion of normalized preactivation RMS and primal–dual RMS products across depth, respectively. A common log-scale ofset cancels in both index variances because adding the same ofset to all log-values shifts the mean log-value by the same amount, leaving each deviation from the mean unchanged. To monitor residual-stream scale growth across depth, we additionally measure $\textstyle \sum _ { \ell } \alpha _ { \ell } ^ { 2 }$ and the aligned drift $\Sigma _ { \ell } \operatorname* { m a x } ( 2 \alpha _ { \ell } c _ { \ell } , 0 )$ from Eq. 6. All five indices and the two additional diagnostics can be computed inexpensively on small batches at intervals during training.

Attention entropy regime. The calibrated attention regime uses dimensionless, row-centered logit variance $v _ { i } = O ( 1 )$ . This paragraph gives the entropy expansion and the limitation of using logit variance to infer saturation. For row logits $z _ { i j }$ with row-centered variance $\begin{array} { r } { v _ { i } : = ( 1 / n ) \sum _ { j } ( z _ { i j } - \bar { z } _ { i } ) ^ { 2 } } \end{array}$ , a second-order expansion of softmax entropy gives

$$
\begin{array} { r } { H ( p _ { i } ) = \log n - \frac { 1 } { 2 } v _ { i } + O ( \underset { j } { \operatorname* { m a x } } | z _ { i j } - \bar { z } _ { i } | ^ { 3 } ) , } \end{array}\tag{E.15}
$$

showing that attention is nearly uniform for $v _ { i } \ll 1$ in this local regime. The calibrated dimensionless regime is $v _ { i } = O ( 1 )$ . Large variance lies outside the Taylor regime and often accompanies saturation. Inferring low entropy from variance additionally requires an assumption on the logit shape. Entropy and maximum probability measure actual saturation directly. The temperature in Eq. 3 targets the calibrated variance regime, and $\mathrm { U V I } _ { \mathrm { a t t n } }$ measures deviation from that regime.

Depth-uniform residual drift. $\mathrm { U V I _ { r e s } }$ connects the measured step budget to stream-scale drift (Sec. 2.3). $\mathrm { U V I _ { r e s } }$ also yields a depth-uniform drift guarantee in the near-orthogonal regime. Prop. 4 shows that $| c _ { k } | \le \kappa \alpha _ { k }$ and bounded target-relative step violation control $\log ( s _ { N } / s _ { 0 } )$ independently of depth. This depth-uniform drif guarantee motivates combining $P ^ { \perp }$ with a depth-scaled residual step in GCT.

Consider any nondegenerate residual trajectory obeying Eq. 6 over N residual additions, with $s _ { k } > 0$ , steps $\alpha _ { k } > 0$ , and alignments $c _ { k } \in [ - 1 , 1 ]$ . Suppose every addition satisfies $\mathrm { U V I } _ { \mathrm { r e s } } ( k ) = | \log ( \alpha _ { k } / \alpha ^ { \star } ) | \leq B$ in the zero-floor limit $\epsilon  0 .$ . This bound implies $\alpha _ { k } \leq a : = \alpha ^ { \star } e ^ { B }$ with $\alpha ^ { \star } : = 1 / \sqrt { N }$ . For a Transformer with attention and MLP sub-blocks, $N = 2 L$ , giving the target $\alpha ^ { \star } = 1 / \sqrt { 2 L }$ in Eq. 7.

The exact projector of Sec. 4 enforces this near-orthogonal regime for the GCT pre-retraction proposal.

Proposition 4 (Depth-uniform drift under near-orthogonal residual additions). Under the preceding setup, suppose additionally that $| c _ { k } | \le \kappa \alpha _ { k }$ with $0 \le \kappa < 1 / 2$ . Then the stream can only grow and

$$
0 \leq \log ( s _ { N } / s _ { 0 } ) \leq \frac { 1 + 2 \kappa } { 2 } \sum _ { k } \alpha _ { k } ^ { 2 } \leq \frac { ( 1 + 2 \kappa ) e ^ { 2 B } } { 2 } .
$$

The proof is given in App. G.6.

The bound is independent of N. For fixed B, the drift upper bound is attained to leading order as $N \to \infty$ by $\alpha _ { k } \equiv a$ and $c _ { k } = \kappa \alpha _ { k }$ . In this limit, $a = e ^ { B } / \sqrt { N } \to 0$

The near-orthogonal stream’s log-scale growth is bounded by $( 1 + 2 \kappa ) e ^ { 2 B } / 2$ at every depth under these assumptions, motivating orthogonalization and depth-scaled residual steps in GCT.

## E.5 Gauge-canonical factorization

What scale separation preserves. For nonzero inputs with independent positive rescalings, canonical coordinates preserve continuous invariant expressivity (Thm. 4; Cor. 3). Targets that depend on absolute magnitude still require the relevant scale information. Coupled unit declarations can also leave meaningfu dimensionless scale ratios, which must be retained (Thm. 5). Apps. E.5 through E.7 give the factorization and perturbation-stability details.

Factorization and expressivity. For nonzero inputs that can each be rescaled independently by a positive factor, Thm. 4 shows that every equivariant computation factors into a function of RMS-normalized inputs and explicit scale factors that supply the required output unit. Choosing RMS-one coordinates makes this factorization unique. For targets that are unchanged by these rescalings, Cor. 3 shows that normalized coordinates retain all the information needed to represent any continuous invariant function. Targets that depend on absolute magnitude still require the relevant scale information.

On nonzero shared-unit blocks, every map respecting independent positive rescalings has a unique factorization once the RMS-one section is chosen. The following theorem establishes this normal form, and the accompanying corollary shows that gauge fixing preserves continuous invariant expressivity.

For $j = 1 , \ldots , m$ , let $V _ { j }$ be a nonzero finite-dimensional real inner-product space of dimension $d _ { j }$ , set $r _ { j } ( x _ { j } ) : =$ $\| x _ { j } \| _ { j } / \sqrt { d _ { j } } = \mathrm { R M S } ( x _ { j } )$ , and define $V _ { j } ^ { \times } : = V _ { j } \setminus \{ 0 \} , \mathbb { S } _ { j } : = \{ z \in V _ { j } : r _ { j } ( z ) = 1 \}$ , and $\mathcal { C } _ { 0 } ( x _ { j } ) : = \overset { \cdot } { x } _ { j } / r _ { j } ( \overset { \cdot } { x } _ { j } )$ . Let $G : = ( \mathbb { R } _ { + } ) ^ { m }$ act on $\begin{array} { r } { X : = \prod _ { j } V _ { j } ^ { \times } } \end{array}$ by $g \cdot x : = ( g _ { 1 } x _ { 1 } , \dots , g _ { m } x _ { m } )$ , and let the action of G on a finite-dimensional normed output space W be the scalar character $\begin{array} { r } { \chi _ { q } ( g ) : = \prod _ { j } g _ { j } ^ { q _ { j } } } \end{array}$ for a fixed $q \in \mathbb { R } ^ { m }$

Theorem 4 (Gauge-canonical factorization). Under the preceding setup, a map $F : X  W$ is equivariant,

$$
F ( g \cdot x ) = \chi _ { q } ( g ) F ( x ) \qquad ( g \in G ) ,\tag{E.16}
$$

$i f$ and only if there is a unique map $\begin{array} { r } { f : \prod _ { j } \mathbb { S } _ { j }  W } \end{array}$ such that

$$
F ( x _ { 1 } , \dots , x _ { m } ) = \left( \prod _ { j = 1 } ^ { m } r _ { j } ( x _ { j } ) ^ { q _ { j } } \right) f ( \mathcal C _ { 0 } ( x _ { 1 } ) , \dots , \mathcal C _ { 0 } ( x _ { m } ) ) .\tag{E.17}
$$

The proof is given in App. G.7.

For $q \neq 0 .$ , Eq. E.17 shows exactly which explicit scale monomial must restore the output unit.

Corollary 3 (Invariant expressivity is preserved by gauge fixing). Under the assumptions of Thm. $^ { 4 , }$ for $q = 0$ pullback by $\mathcal { C } _ { 0 }$ is an isometric bijection, in the uniform norm, between continuous functions on the compact product $\Pi _ { j } \mathbb { S } _ { j }$ and continuous G-invariant functions on X. The proof is given in App. G.7.

Any model class uniformly dense on the canonical coordinates is uniformly dense in the entire continuous invariant function class after composition with $\mathcal { C } _ { 0 }$ , which means that gauge fixing preserves the full continuous invariant function class. The mechanism is the scale-group counterpart of frame averaging and canonicalization for spatial symmetry groups, where a symmetry is likewise discharged by mapping each input to a canonical representative before an unconstrained network is applied (Puny et al., 2022; Kaba et al., 2023). The rescaling group here is abelian and acts by a positive scalar on each block. The RMS-one section is a single global slice with a unique canonical representative for each orbit. No frame averaging is needed.

For the single residual-stream action, the invariant case is $F ( h ) = f ( \mathcal { C } _ { 0 } ( h ) )$ and the degree-one equivariant case is $F ( h ) = \mathrm { R M S } ( h ) f ( \mathcal { C } _ { 0 } ( h ) )$ . The dimensionless head and the explicit scale-state reconstruction are the $q = 0$ and $q = 1$ instances of the same representation theorem. The corollary covers targets that are unchanged by rescaling. Targets that depend on absolute magnitude require retaining the relevant scale information. Sec. I.5 discusses magnitude information as one possible explanation for the additive-noise results.

## E.6 Coupled neural units and the complete quotient

When unit declarations couple diferent tensors, relative magnitudes can remain meaningful because two tensors with the same unit have a dimensionless RMS ratio. Thm. 5 identifies all the scale combinations that must be retained, namely the dimensionless scale monomials generated by ker A, where A collects the declared unit exponents. Keeping these combinations prevents normalization from discarding information that is unchanged by the declared rescalings.

Thm. 4 uses independent rescalings, one rescaling for each block. Neural-unit declarations can instead couple several blocks to the same base units. In that general case, the quotient can contain both canonical directions and dimensionless ratios of empirical scales. Use the following notation for the coupled action.

For $j = 1 , \ldots , m$ , let $V _ { j }$ be a nonzero finite-dimensional real inner-product space, let $V _ { j } ^ { \times } : = V _ { j } \setminus \{ 0 \}$ , and define $r _ { j } ( x _ { j } ) : = \mathrm { R M S } ( x _ { j } ) , z _ { j } ( x _ { j } ) : = { \mathcal C } _ { 0 } ( x _ { j } ) = x _ { j } / r _ { j } ( x _ { j } )$ , and $\mathbb { S } _ { j } : = \{ z : r _ { j } ( z ) = 1 \}$ . Write $g : = { { e } ^ { t } } \in { { \cal G } } = ( \mathbb { R } _ { + } ) ^ { K }$ and let the coupled scalar-unit action on $\begin{array} { r } { X = \prod _ { j } V _ { j } ^ { \times } } \end{array}$ be

$$
( e ^ { t } \cdot x ) _ { j } : = e ^ { a _ { j } ^ { \top } t } x _ { j } , \qquad A : = \left[ a _ { 1 } \quad \cdot \cdot \cdot \quad a _ { m } \right] \in \mathbb { R } ^ { K \times m } .\tag{E.18}
$$

Put $s : = { }$ rank A and let the columns of $\textbf { \textit { N } } \in \mathbb { \mathbb { R } } ^ { m \times ( m - s ) }$ be any basis of ker A. With $\log r ( x ) \ : = $ (log $r _ { 1 } ( x _ { 1 } ) , \ldots , \log r _ { m } ( x _ { m } ) )$ , define

$$
\begin{array} { r } { \mathcal { Q } ( \boldsymbol { x } ) : = ( z _ { 1 } ( x _ { 1 } ) , \ldots , z _ { m } ( x _ { m } ) , N ^ { \top } \log r ( \boldsymbol { x } ) ) . } \end{array}\tag{E.19}
$$

The next theorem gives the complete quotient. The following corollary determines when nonzero equivariant outputs exist and factorizes those outputs when the output character is represented by the input characters.

Theorem 5 (Complete quotient for coupled neural units). Under the preceding setup, Q is a complete invariant: For all $x , x ^ { \prime } \in X$

$$
\mathcal { Q } ( x ) = \mathcal { Q } ( x ^ { \prime } ) \quad \iff \quad x ^ { \prime } = g \cdot x f o r s o m e g \in G .\tag{E.20}
$$

The proof is given in App. G.8.

The $N ^ { \top } \log r ( x )$ coordinates are logarithms of dimensionless scale monomials generated by ker A. When rank $A =$ $m ,$ , no scale coordinate remains and canonical directions form the complete quotient. Otherwise, the $m - s$ Buckingham–π coordinates in Eq. E.19 must be retained. The independent action of Thm. 4 is the special case $A = I _ { m }$

Corollary 4 (Factorization of equivariant outputs). Under the assumptions of Thm. 5, let a finite-dimensional output space W carry the scalar character $w \mapsto e ^ { q ^ { \top } t } w$ $H q \notin$ im A, the only equivariant map $F : X  W$ is $F \equiv 0$ . If q ∈ im A, fix any $p \in \mathbb { R } ^ { m }$ satisfying $A p = q$ . Then all and only the equivariant maps $F : X  W$ are

$$
F ( x ) = \left( \prod _ { j = 1 } ^ { m } r _ { j } ( x _ { j } ) ^ { p _ { j } } \right) f ( z _ { 1 } ( x _ { 1 } ) , \ldots , z _ { m } ( x _ { m } ) , N ^ { \top } \log r ( x ) ) ,\tag{E.21}
$$

where, for the fixed choices of p and N, the map $\begin{array} { r } { f : ( \prod _ { i } \mathbb { S } _ { j } ) \times \mathbb { R } ^ { m - s }  W } \end{array}$ is unique. Moreover, F is continuous $i f$ and only if f is continuous. The proof is given in App. G.8.

Thm. 5 and Cor. 4 give a design rule against over-canonicalization. A common scale is gauge, but a ratio of two quantities carrying the same unit is dimensionless and may be predictive. The ratio must either remain in the canonical computation or be retained as an explicit log-ratio state. The coordinates $N ^ { \top }$ log S in Thm. 2 are exactly the scale-only part of this maximal invariant. The independent per-token residual gauge used by GCT is the N-empty case in which no dimensionless scale coordinate remains.

## E.7 Perturbation stability on the canonical quotient

Perturbation stability. RMS normalization cancels a positive multiplier because the vector and the corresponding RMS are multiplied by the same factor. For $x \neq 0$ and $\epsilon _ { x } = 0$ , this cancellation gives ${ \mathcal C } ( c x ) = { \mathcal C } ( x )$ for every $c > 0$ . A perturbation that also changes direction can afect the result. Thm. 6 separates a perturbation into a radial component along the input and a tangential component perpendicular to the input. As long as the radial component does not reverse the input’s direction, normalization removes pure radial perturbations exactly and the remaining error is bounded by the tangential component measured relative to the radially rescaled input.

Factorization identifies the information available to an invariant map. The next result quantifies the efect of a perturbation after canonicalization, assuming that the perturbation’s radial component preserves orientation, meaning a $, > - 1$ in the notation below. Pure radial perturbations are removed exactly. For a mixed perturbation, the bound uses the tangential ratio $\rho / ( 1 + a )$ , which measures directional error after accounting for radial rescaling.

Let V be a real inner-product space of dimension $d \geq 2 , r ( x ) : = \| x \| / \sqrt { d } , \mathbb { S } : = \{ z : r ( z ) = 1 \}$ , and $\mathcal { C } _ { 0 } ( x ) : = x / r ( x )$ on $V ^ { \times } : = V \setminus \{ 0 \}$ . Equip S with the RMS chord metric $d _ { \mathbb { S } } ( z , z ^ { \prime } ) : = r ( z - z ^ { \prime } )$ . For $x \neq 0$ and a perturbation e, define the relative radial coeficient and tangential component of e by $a : = \langle x , e \rangle / \| x \| ^ { 2 } , e _ { \perp } : = e - a x$ , and $\rho : = \| e _ { \perp } \| / \| x \|$

Theorem 6 (Radial rejection and sharp tangential displacement). Under the preceding setup, $i f a > - 1$ , then

$$
d _ { \mathbb { S } } ( \mathcal { C } _ { 0 } ( x + e ) , \mathcal { C } _ { 0 } ( x ) ) = \left[ 2 \left( 1 - \frac { 1 + a } { \sqrt { ( 1 + a ) ^ { 2 } + \rho ^ { 2 } } } \right) \right] ^ { 1 / 2 } \leq \frac { \rho } { 1 + a } .\tag{E.22}
$$

The proof is given in App. G.9.

The unit constant in the upper bound is asymptotically sharp as $\rho / ( 1 + a )  0$ . For an orientation-preserving pure radial error $e = a x$ with $a > - 1$ , we have $\rho = 0$ and the error is rejected exactly.

Corollary 5 (Functional stability under coordinatewise quantization). Under the notation and assumptions of Thm. 6, let $f : \mathbb { S }  W$ be K-Lipschitz into a normed space W and set $F : = f \circ \mathcal { C } _ { 0 } . \quad I f ,$ in a fixed orthonormal coordinate system, a nonclipping coordinatewise quantizer produces $e : = Q ( x ) - x$ with $| e _ { i } | \le \Delta / 2$ and $\delta : = \Delta / ( 2 r ( x ) ) < 1$ , then $Q ( x ) \neq 0$ and

$$
\Vert F ( Q ( x ) ) - F ( x ) \Vert _ { W } \leq K { \frac { \delta } { 1 - \delta } } .\tag{E.23}
$$

The proof is given in App. G.9.

For a canonical stream $r ( x ) = 1$ , this bound is independent of the token’s original scale. For a shared quantization step on an unmanaged stream, the bound worsens as $1 / r ( x )$ and exposes exactly the cross-token scale-spread penalty of Prop. 5.

The perturbation theorem and the quantization corollary separate two robustness claims that are otherwise easy to conflate. Gauge invariance removes multiplicative scale distortion with no Lipschitz constant. Finiteprecision rounding generally introduces both radial and tangential components, but only the tangential component can change the canonical computation. The sensitivity to rounding is determined by the dimensionless ratio $\mathrm { R M S } ( e ) / \mathrm { R M S } ( x )$ . The rescaling experiment of Sec. 5.4 isolates the first claim, while residual-stream quantization in Sec. 5.5 exercises the second claim.

## E.8 Shared-scale quantization on the quotient

We next bound the nonclipping rounding error under shared-scale quantization. In symmetric b-bit absmax quantization with a shared per-tensor scale, the largest absolute coordinate in the tensor sets the quantization step. Smaller-scale tokens have a larger relative rounding-error bound.

Two conventions fix the scope of the bound. In the following results, the index t ranges only over vectors that share the same quantization step. If a deployment uses a separate tensor scale at each block site, the result applies sitewise with a separate $\kappa _ { \mathrm { s c a l e } , \ell } .$ The stated bound applies directly to dynamic absmax. For a static nonclipping range $A _ { \mathrm { c a l } } \geq A$ , the same proof replaces A by $A _ { \mathrm { c a l } }$ and multiplies the dynamic-absmax right-hand sides by the calibration-slack factor $\gamma _ { \mathrm { c a l } } : = A _ { \mathrm { c a l } } / A$ . If $A _ { \mathrm { { c a l } } } < A$ , clipping contributes an additional error not covered by the proposition.

Formally, let $b \geq 2 , q : = 2 ^ { b - 1 } - 1$ , and let nonzero vectors $h _ { 1 } , \ldots , h _ { T } \in \mathbb { R } ^ { d }$ have $r _ { t } : = \mathrm { R M S } ( h _ { t } )$ . Set $A : =$ ma $\begin{array} { r } { \mathrm { x } _ { t , i } | h _ { t , i } | , \Delta : = A / q , } \end{array}$ , and $\kappa _ { \mathrm { a m p } } : = \operatorname* { m a x } _ { t } \| h _ { t } \| _ { \infty } / r _ { t }$ . Let $Q _ { \Delta }$ be coordinatewise symmetric absmax quantization to the grid $\Delta \{ - q , \ldots , q \}$

Proposition 5 (Shared-scale absmax rounding bound). Under the preceding setup, no coordinate clips and, for every token,

$$
\frac { \| Q _ { \Delta } ( h _ { t } ) - h _ { t } \| _ { 2 } } { \| h _ { t } \| _ { 2 } } \leq \frac { \kappa _ { \mathrm { a m p } } } { 2 q } \frac { \operatorname* { m a x } _ { s } r _ { s } } { r _ { t } } \leq \kappa _ { \mathrm { a m p } } 2 ^ { - ( b - 1 ) } \frac { \operatorname* { m a x } _ { s } r _ { s } } { r _ { t } } .
$$

The proof is given in App. G.11.

For fixed normalized directions, the worst tokenwise multiplier in Prop. 5 is the scale spread $\kappa _ { \mathrm { s c a l e } } : =$ max $r _ { t } / \AA$ min $r _ { t }$ . A zero-floor canonical stream has $r _ { t } \ \equiv \ 1$ under Eq. 11, while per-token absmax scaling removes the shared-token factor for an arbitrary stream. With a finite frozen floor, an input of RMS r is mapped to RMS $( 1 + \epsilon / r ^ { 2 } ) ^ { - 1 / 2 }$ . Equal output scales are only approximate when the input scales difer. The associated rescaling defect is bounded in Eq. F.1. Under the usual high-resolution power-of-two convention, the shared-scale factor costs $\log _ { 2 } \kappa _ { \mathrm { s c a l e } }$ efective bits. The factor is sharp in general.

The preceding sharpness statement concerns representation error. To characterize functional error after radial rejection, we next bound the displacement on the canonical quotient. The matching lower bound shows that shared-scale rounding can produce an unavoidable tangential error even when within-token amplitudes are bounded.

Let $q \geq 2$ be an integer, let $h _ { 1 } , \ldots , h _ { T } \in \mathbb { R } ^ { d } \setminus \left\{ 0 \right\}$ , and use the shared absmax quantizer of Prop. 5, with $r _ { t } : = { }$ $\mathrm { R M S } ( h _ { t } ) , r _ { \operatorname* { m i n } } : =$ min<sub>t</sub> $r _ { t } , A : = \operatorname* { m a x } _ { t } \| h _ { t } \| _ { \infty } $ , and $\begin{array} { r } { \Delta : = A / q . \mathrm { \ P u t \ } \kappa _ { \mathrm { a m p } } : = \operatorname* { m a x } _ { t } \| h _ { t } \| _ { \infty } / r _ { t } , \kappa _ { \mathrm { s c a l e } } : = \operatorname* { m a x } _ { t } r _ { t } / r _ { \mathrm { m i n } } , } \end{array}$ $\delta _ { * } : = A / ( 2 q r _ { \operatorname* { m i n } } )$ , and $\mathbb { S } _ { d } : = \{ z : \mathrm { R M S } ( z ) = 1 \}$

Lemma 1 (Shared-scale quotient upper bound). Under the preceding setup, if $\ : : \delta _ { * } < 1 \ :$ , then every $Q _ { \Delta } ( h _ { t } )$ is nonzero and, for every K-Lipschitz $f : \mathbb { S } _ { d } \to W$ into a normed space W under the RMS chord metric,

$$
\operatorname* { m a x } _ { t } \| f ( \mathcal C _ { 0 } ( Q _ { \Delta } ( h _ { t } ) ) ) - f ( \mathcal C _ { 0 } ( h _ { t } ) ) \| _ { W } \le K \frac { \delta _ { * } } { 1 - \delta _ { * } } , \qquad \delta _ { * } \le \frac { \kappa _ { \mathrm { a m p } } \kappa _ { \mathrm { s c a l e } } } { 2 q } .\tag{E.24}
$$

The proof is given in App. G.12.

For $q \geq 4$ and $2 \leq \Gamma \leq q / 2$ , define the worst-case quotient error

$$
\mathfrak { E } _ { q } ( \Gamma ) : = \operatorname* { s u p } _ { \begin{array} { c } { { d _ { \ge 2 , T \ge 2 } } } \\ { { h _ { 1 } , \ldots , h _ { T } \ne 0 } } \\ { { \kappa _ { \mathrm { s c a l e } } \le \Gamma , \kappa _ { \mathrm { a m p } } \le \sqrt { 2 } } } \\ { { f \colon \mathbb { S } _ { d }  \mathbb { R } , \mathrm { L i p } ( f ) \le 1 } } \end{array} } \operatorname* { m a x } _ { t } | f ( \mathcal { C } _ { 0 } ( Q _ { \Delta } ( h _ { t } ) ) ) - f ( \mathcal { C } _ { 0 } ( h _ { t } ) ) | ,
$$

where, for each tensor in the supremum, r<sub>t</sub>, κ<sub>scale</sub>, κ<sub>amp</sub>, A, $\Delta : = A / q$ , and $Q _ { \Delta }$ are defined as in Lem. 1. These constraints imply $\delta _ { * } \leq \sqrt { 2 } \Gamma / ( 2 q ) \leq \sqrt { 2 } / 4 < 1$ . Lem. 1 ensures that every canonicalized quantized vector in this definition is nonzero and the supremum is well-defined.

Theorem 7 (Worst-case efective-bit law on the scale quotient). The dependence in Lem. 1 is worst-case sharp up to universal constants even for scalar outputs and $\kappa _ { \mathrm { a m p } } \leq { \sqrt { 2 } } .$ For every $q \geq 4$ and $2 \leq \Gamma \leq q / 2$

$$
\frac { \Gamma } { 6 \pi q } \leq \mathfrak { E } _ { q } ( \Gamma ) \leq \frac { \sqrt { 2 } \Gamma } { 2 q - \sqrt { 2 } \Gamma } .\tag{E.25}
$$

The proof is given in App. G.12.

Thus $\mathfrak { E } _ { q } ( \Gamma ) = \Theta ( \Gamma / q )$ in the stated high-resolution regime even when $\kappa _ { \mathrm { a m p } }$ is bounded. The lower-bound construction has $\kappa _ { \mathrm { s c a l e } } \in [ \Gamma / 2 , \Gamma ] .$ , so the same order is equivalently linear in the attained worst-case scale spread up to universal constants. Since $\dot { q } = 2 ^ { b - 1 } - 1$ , the high-resolution interpretation of the worst-case shared-scale law is a loss of $\log _ { 2 } \kappa _ { \mathrm { s c a l e } }$ efective bits even after exact radial rejection. At finite $b ,$ the exact statement is the factor in q of Eq. E.25. A zero-floor canonical stream sets $\kappa _ { \mathrm { s c a l e } } = 1$ and leaves only the within-vector term $O ( \kappa _ { \mathrm { a m p } } 2 ^ { - b } )$

The deployment protocol assigns a separate scale to every block site. The theoretical bound applies across tokens within each site. The zero-floor GCT formulation enforces $r _ { \ell , t } = 1$ for every token and hence $\kappa _ { \mathrm { s c a l e } , \ell } = 1$ . A finite-floor canonicalization instead maps an input RMS $r _ { \ell , t , \mathrm { i n } }$ to $( 1 + \epsilon / r _ { \ell , t , \mathrm { i n } } ^ { 2 } ) ^ { - 1 / 2 }$ and has the intervention defect of Eq. F.1. An unmanaged stream imposes no such constraint. The static-quantization columns directly measure the resulting deployment error, including calibration slack and any clipping, and the per-token control removes the shared-token-scale factor empirically.

Prop. 5 bounds how much rounding error a shared scale creates. Thm. 6 shows that the downstream canonica graph discards the radial part of the rounding error. Thm. 7 establishes that the remaining tangential functional error still has the same scale-spread order in the worst case. Together, these results identify two contributors to the deployment gap. An unmanaged stream inflates $\mathrm { R M S } ( e ) / \mathrm { R M S } ( h _ { t } )$ for small-scale tokens, while a canonical stream fixes the denominator and removes the scale direction.

In the high-resolution interpretation, unmanaged scale consumes $\log _ { 2 } \kappa _ { \mathrm { s c a l e } }$ efective bits. This cost occupies a larger fraction of the budget as the bit width decreases. In the int4 stream columns of Table 10, the worst baseline changes by 7.71 nats with a per-tensor scale, compared with +0.828 for GCT. Per-token scales reduce the worst change $\mathrm { t o \ t i { 1 . 0 4 4 } }$ , while GCT changes by +0.107 and nGPT by +0.52. This remaining gap is consistent with within-vector amplitude, downstream directional sensitivity, and accumulation across sites, none of which canonicalization removes.

## F DERIVATIONS

## F.1 Finite floors of the canonical gauge

Numerical stabilizers are also typed. A denominator $s _ { x } + \epsilon _ { s }$ requires $u ( \epsilon _ { s } ) = u ( x )$ , and $\sqrt { s _ { x } ^ { 2 } + \epsilon _ { \mathrm { v a r } } }$ requires $u ( \epsilon _ { \mathrm { v a r } } ) = u ( x ) ^ { 2 }$ . Adding a fixed dimensionless ϵ to a unitful quantity breaks exact equivariance in the floordominated regime and serves only as numerical regularization.

For the canonicalizer in Sec. 4, setting $\epsilon _ { x } = 0$ recovers exact division by RMS for $x \neq 0$ . In this case, positive multiples such as x and 2x give the same RMS-one vector, and applying C again leaves that vector unchanged. The implementation adds a small positive $\epsilon _ { x }$ to prevent division by zero. This numerical floor makes the output RMS less than one. Exact unit consistency with a finite floor requires $u ( \epsilon _ { x } ) = u ( x ) ^ { 2 }$ and $\epsilon _ { x } \mapsto c ^ { 2 } \epsilon _ { x }$ when x 7→ cx. Keeping $\epsilon _ { x }$ numerically fixed during our rescaling interventions can change the normalized vector. The following bound quantifies the rescaling error in Eq. F.1.

For $\mathcal { C } _ { \epsilon } ( x ) : = x / \sqrt { \mathrm { R M S } ( x ) ^ { 2 } + \epsilon }$ with a fixed numerica $\epsilon \geq 0 , r : = \mathrm { R M S } ( x ) > 0$ , and $c > 0$ , direct calculation and the mean-value theorem give

$$
\mathrm { R M S } ( \mathcal { C } _ { \epsilon } ( c x ) - \mathcal { C } _ { \epsilon } ( x ) ) = \left| \frac { 1 } { \sqrt { 1 + \epsilon / ( c ^ { 2 } r ^ { 2 } ) } } - \frac { 1 } { \sqrt { 1 + \epsilon / r ^ { 2 } } } \right| \leq \frac { \epsilon } { 2 r ^ { 2 } } | c ^ { - 2 } - 1 | .\tag{F.1}
$$

A correctly typed floor transforms as $\epsilon \mapsto c ^ { 2 } \epsilon$ and makes this defect exactly zero, whereas a frozen implementation floor leaves the explicit $O ( \epsilon / r ^ { 2 } )$ local defect that Eq. E.3 propagates to the output.

The same floor also prevents an exact RMS of one. If $r = \mathrm { R M S } ( x ) > 0$ , then $\mathrm { R M S } ( \mathcal { C } ( x ) ) = r / \sqrt { r ^ { 2 } + \epsilon _ { x } } =$ $( 1 + \epsilon _ { x } / r ^ { 2 } ) ^ { - 1 / 2 }$ . The canonicalizer of Sec. 4 returns RMS one only in the zero-floor limit.

Finite-floor implementation. For numerical regularization, the implementation uses $\| \bar { h } \| ^ { 2 } + \epsilon _ { \perp }$ in the projection denominator. The display below derives the resulting residual alignment. The exact orthogonality statement in Sec. 4 and the rotation identity in App. F.4 apply to $P _ { \bar { h } } ^ { \perp }$ or to the zero-projection-floor limit. Comparing relative errors for finite floors additionally requires a nondegenerate tangential proposal.

This projection floor behaves analogously to the canonicalization floor. Replacing $\| \bar { h } \| ^ { 2 }$ by $\| \bar { h } \| ^ { 2 } + \epsilon _ { \perp }$ generally leaves a residual component along h<sup>¯</sup>:

$$
\left. \bar { h } , a - \frac { \langle a , \bar { h } \rangle } { \| \bar { h } \| ^ { 2 } + \epsilon _ { \perp } } \bar { h } \right. = \frac { \epsilon _ { \perp } } { \| \bar { h } \| ^ { 2 } + \epsilon _ { \perp } } \langle a , \bar { h } \rangle .
$$

## F.2 Dot-product variance and the calibrated temperature in Eq. 3

Suppose each $q _ { a }$ is independent of $k _ { a } .$ , both coordinates are zero-mean with $\mathrm { V a r } ( q _ { a } ) = s _ { Q } ^ { 2 }$ and $\mathrm { V a r } ( k _ { a } ) = s _ { K } ^ { 2 }$ , and the products $q _ { a } k _ { a }$ are uncorrelated across channels. Channelwise independence is one suficient condition. Then $\begin{array} { r } { \mathrm { V a r } ( q ^ { \top } k ) = \bar { \sum _ { a } } \mathrm { V a r } ( q _ { a } k _ { a } ) = d _ { h } s _ { Q } ^ { 2 } s _ { K } ^ { 2 } } \end{array}$ exactly. With merely weak cross-channel correlations, the same equality is an approximation with the omitted cross-covariances added explicitly. Dividing logits by $\tau = \sqrt { d _ { h } } s _ { Q } s _ { K }$ yields dimensionless logits of unit variance under the exact assumptions and $O ( 1 )$ variance in the weak-correlation regime. In the zero-floor GCT formulation, per-head-vector canonicalization enforces $\lVert \bar { q } \rVert = \lVert \bar { k } \rVert = \sqrt { d _ { h } }$ . For weakly dependent directions, $\bar { q } ^ { \top } \bar { k } / \sqrt { d _ { h } }$ again has $O ( 1 )$ variance, and the learnable per-head gain tunes the regime multiplicatively.

Attention in canonical coordinates. Queries and keys are canonicalized per head vector, $\bar { q } : = \mathcal { C } ( q )$ and $\bar { k } : = \mathcal { C } ( k )$ . When $\epsilon _ { x } = 0$ , RMS normalization removes the magnitude of each nonzero query and key before the dot product. The remaining $1 / \sqrt { d _ { h } }$ factor accounts for the head dimension. If the query and key directions are approximately isotropic and weakly dependent at initialization, we have $\mathrm { V a r } ( \bar { q } ^ { \top } \bar { k } / \sqrt { d _ { h } } ) \approx 1$ , giving softmax dimensionless logits on an $O ( 1 )$ scale. A per-head dimensionless gain $g _ { h }$ , initialized at 1, tunes the attention entropy within these canonical coordinates.

Empirical calibration. If the coordinates are centered with $\mathrm { V a r } ( q _ { a } ) = s _ { Q } ^ { 2 }$ and $\mathrm { V a r } ( k _ { a } ) = s _ { K } ^ { 2 }$ , each $q _ { a }$ is independent of $k _ { a } .$ and the products $q _ { a } k _ { a }$ are uncorrelated across channels, then $\mathrm { V a r } ( q ^ { \top } k ) = d _ { h } s _ { Q } ^ { 2 } s _ { K } ^ { 2 }$ exactly. Empirical per-head centered-RMS estimates give a calibrated approximation. Centered RMS is the RMS after subtracting the mean, $\mathrm { R M S } ( x - { \bar { x } } \mathbf { 1 } )$ , with x¯ computed over the same entries. Under that averaging convention, centered RMS equals the standard deviation. Ordinary RMS serves as a proxy when coordinate means are negligible.

## F.3 Gradient dual units

For dimensionless $\mathcal { L }$ and $u ( x ) = U$ , consider $x \mapsto \chi _ { U } ( g ) x$ . Invariance of $\mathrm { d } \mathcal { L } = \langle \nabla _ { x } \mathcal { L } , \mathrm { d } x \rangle$ forces $\nabla _ { x } \mathcal { L } \mapsto$ $\chi _ { U } ( g ) ^ { - 1 } \nabla _ { x } \mathcal { L }$ . The products $x \odot \nabla _ { x } { \mathcal { L } }$ and $\mathrm { R M S } ( x ) \mathrm { R M S } ( \nabla _ { x } \mathcal { L } )$ are dimensionless invariants, whereas the raw gradient norm scales inversely with x. The same argument for parameters gives $u ( \nabla _ { \theta } \mathcal { L } ) = U _ { \theta } ^ { - 1 }$ and types Adam moments as $u ( m ) = U _ { \theta } ^ { - 1 }$ and $u ( v ) = U _ { \theta } ^ { - 2 }$ . The ratio $m / ( \sqrt { v } + \epsilon _ { g } )$ is dimensionless when $u ( \epsilon _ { g } ) = U _ { \theta } ^ { - 1 }$

## F.4 Residual scale dynamics for Eq. 6

Let $s _ { \ell } : = \| h _ { \ell } \| _ { 2 } / \sqrt { d }$ and assume $h _ { \ell } \neq 0 . \mathrm { ~ H ~ } r _ { \ell } = 0$ , then $h _ { \ell + 1 } = h _ { \ell }$ and $s _ { \ell + 1 } = s _ { \ell }$ , with no change in stream scale. For $r _ { \ell } \neq 0$ , write $\begin{array} { r } { r _ { \ell } = \alpha _ { \ell } s _ { \ell } z _ { \ell } } \end{array}$ with $\mathrm { R M S } ( z _ { \ell } ) = 1$ . The coeficient $\alpha _ { \ell } = \mathrm { R M S } ( r _ { \ell } ) / \mathrm { R M S } ( h _ { \ell } ) > 0$ is the dimensionless residual step size. Expanding $\| h _ { \ell } + r _ { \ell } \| _ { 2 } ^ { 2 }$ shows how the branch’s squared magnitude and alignment with the stream change the squared stream scale in $\operatorname { E q . 6 }$

With $h _ { \ell + 1 } = h _ { \ell } + r _ { \ell } , \| h _ { \ell + 1 } \| ^ { 2 } = \| h _ { \ell } \| ^ { 2 } + \| r _ { \ell } \| ^ { 2 } + 2 \langle h _ { \ell } , r _ { \ell } \rangle$ . Dividing by d and using $s _ { \ell } = \| h _ { \ell } \| / \sqrt { d } .$ , the definition $\| r _ { \ell } \| = \alpha _ { \ell } \| h _ { \ell } \|$ , and the cosine similarity $c _ { \ell }$ gives Eq. 6.

The cosine similarity $c _ { \ell }$ measures the directional alignment between the stream and the branch contribution. A positive value means that the contribution points partly along the stream, zero means that the two vectors are perpendicular, and a negative value means that the contribution points partly against the stream. If branch contributions align persistently with the stream, say $c _ { \ell } \geq c _ { 0 } > 0$ , bounded stream scale forces $\textstyle \sum _ { \ell } \alpha _ { \ell } = O ( 1 )$ . If all $L$ steps are equal or remain uniformly comparable, each must be $O ( L ^ { - 1 } )$ . This per-layer bound need not hold when only a few layers have appreciable contributions. Those layers may take larger steps, provided that $\textstyle \sum _ { \ell }$ α stays bounded. If $c _ { \ell } = 0$ exactly, boundedness follows from $\textstyle \sum _ { \ell } \alpha _ { \ell } ^ { 2 } = O ( 1 )$ and permits the uniform schedule $\alpha _ { \ell } = \Theta ( L ^ { - 1 / 2 } )$ , which is the branch scale that depth-wise $\mu \mathrm { P }$ prescribes for depth transfer (Yang et al., 2024). Prop. 4 extends this order to near-orthogonal contributions with $| c _ { \ell } | \leq \kappa \alpha _ { \ell }$ and $0 \le \kappa < 1 / 2$ . The alignment term $2 \alpha _ { \ell } c _ { \ell }$ then has magnitude at most $2 \kappa \alpha _ { \ell } ^ { 2 }$ . Exact orthogonality is unnecessary as long as alignment shrinks with the residual step size.

For persistent positive alignment $c _ { \ell } \geq c _ { 0 } > 0 .$ , the term $2 \alpha _ { \ell } c _ { \ell }$ causes scale growth that is first order in the residual step size. Orthogonalizing the branch contribution removes this linear term and leaves $\alpha _ { \ell } ^ { 2 }$ , permitting larger uniform residual steps of order $L ^ { - 1 / 2 }$ under the same bounded scale budget. Geometrically, an orthogonal contribution changes the hidden state’s direction, while the resulting change in the state’s magnitude begins only at second order in $\alpha _ { \ell }$ . Sec. 4 implements this orthogonalization by projecting each branch output perpendicular to the stream.

If $c _ { \ell } \geq c > 0$ for all $\ell ,$ then $s _ { L } ^ { 2 } \ge s _ { 0 } ^ { 2 } \Pi _ { \ell } ( 1 + 2 c \alpha _ { \ell } )$ . Uniform boundedness of this product also bounds every $\alpha _ { \ell }$ . On that bounded interval, $\log ( 1 + 2 c \alpha ) \ \geq$ c<sup>′</sup>α for some $c ^ { \prime } > 0$ . Boundedness therefore forces $\textstyle \sum _ { \ell } \alpha _ { \ell } = O ( 1 )$ . Only a uniform or uniformly comparable schedule implies the per-layer order $O ( L ^ { - 1 } )$ . If $c _ { \ell } = 0$ then $\begin{array} { r } { s _ { L } ^ { 2 } = s _ { 0 } ^ { 2 } \prod _ { \ell } ( 1 + \alpha _ { \ell } ^ { 2 } ) \le s _ { 0 } ^ { 2 } e ^ { \sum _ { \ell } \alpha _ { \ell } ^ { 2 } } } \end{array}$ , so $\textstyle \sum _ { \ell } \alpha _ { \ell } ^ { 2 } = O ( 1 )$ sufices. With uniform steps, the bounded quadratic step budget permits $\overset { \cdot } { \alpha } _ { \ell } = \Theta \overset { \cdot } { ( } L ^ { - 1 / 2 } )$ instead of $\Theta ( L ^ { - 1 } )$ . With the exact projector, a nonzero projected branch output, and zero canonicalization floors, GCT adds a contribution orthogonal to the current state. The subsequent canonicalization normalizes the sum of the state and this contribution, giving $\bar { h } _ { \ell ^ { + } } = ( \bar { h } _ { \ell } + \alpha _ { \ell } \hat { a } _ { \ell } ) / \sqrt { 1 + \alpha _ { \ell } ^ { 2 } } ,$ . The finite projection-floor qualification is stated in $\mathrm { A p p }$ . F.1.

Canonical residual update. In Eq. 11, $\Phi _ { \ell }$ is the attention or MLP branch, and $\ell ^ { + }$ denotes the state after that sub-block. When the projected branch output $P _ { \bar { h } } ^ { \perp } \bar { a }$ is nonzero and the canonicalization floors are set to zero, $\hat { a } \perp \bar { h }$ and both vectors have RMS one. The final normalization, which we call the retraction, returns the updated vector to RMS one. The next canonical state is the exact rotation

$$
\bar { h } _ { \ell ^ { + } } = \frac { \bar { h } _ { \ell } + \alpha _ { \ell } \hat { a } _ { \ell } } { \sqrt { 1 + \alpha _ { \ell } ^ { 2 } } } ,
$$

which changes direction while keeping RMS fixed. Before retraction, the vector actually added to the state is $r _ { \ell } ^ { \mathrm { p r e } } : = \alpha \ell \hat { a } _ { \ell }$ . This contribution is orthogonal to ${ \bar { h } } _ { \ell } ,$ matching the orthogonal residual case analyzed in Sec. 2.3. When $\alpha _ { \ell } \neq 0$ , the pre-retraction contribution has cosine similarity $c _ { \ell } ^ { \mathrm { p r e } } = 0$ with the stream and an RMS ratio to the stream of exactly $\left| \alpha _ { \ell } \right|$ . The learned scalar directly controls the relative size of the added contribution. When $\alpha _ { \ell } = 0$ , no branch contribution is added, and exact RMS normalization leaves the state unchanged. The final displacement $\bar { h } _ { \ell ^ { + } } - \bar { h } _ { \ell }$ generally includes a component along $\bar { h } _ { \ell }$ because retraction rescales the sum. With zero canonicalization floors, the resulting state still satisfies $\mathrm { R M S } ( \bar { h } _ { \ell ^ { + } } ) = 1$ exactly.

## F.5 Attention entropy expansion for Eq. E.15

Let $z _ { j } = \bar { z } + \delta _ { j }$ with $\begin{array} { r } { ( 1 / n ) \sum _ { i } \delta _ { j } = 0 . } \end{array}$ , put $v : = ( 1 / n ) \textstyle \sum _ { j } \delta _ { j } ^ { 2 }$ , and write $M : = \operatorname* { m a x } _ { j } | \delta _ { j } |$ . Then $\begin{array} { r } { p _ { j } = e ^ { \delta _ { j } } / \sum _ { k } e ^ { \delta _ { k } } } \end{array}$ and a Taylor expansion of $\begin{array} { r } { H ( \bar { p } ) = - \sum _ { j } p _ { j } \log p _ { j } } \end{array}$ in δ gives $H = \log n - v / 2 + O ( M ^ { 3 } )$ . To obtain this expansion, write $\textstyle \sum _ { k } e ^ { \delta _ { k } } = n ( 1 + { v } / { 2 } + { O } ( M ^ { 3 } ) )$ and $\log p _ { j } = \delta _ { j } - \log n - v / 2 + O ( M ^ { 3 } )$ . Substituting these expansions gives

$$
H = \log n + \frac { v } { 2 } - \frac { 1 } { n } \sum _ { j } \left( 1 + \delta _ { j } + \frac { \delta _ { j } ^ { 2 } } { 2 } \right) \delta _ { j } + O ( M ^ { 3 } ) = \log n - \frac { v } { 2 } + O ( M ^ { 3 } ) .
$$

Centered logit variance controls the local attention entropy to leading order. Small v gives near-uniform attention. Large v flags over-dispersed logits. Outside the Taylor regime, entropy and maximum probability identify whether attention is saturated. $\mathrm { U V I } _ { \mathrm { a t t n } }$ measures under- and over-dispersion symmetrically in log space.

## F.6 Exact gauge invariance of GCT

Boundary invariance. Assume that the canonicalization floors are zero and every vector being normalized is nonzero. Replacing a block input h by ch for $c > 0$ then leaves the network’s dimensionless output unchanged because ${ \mathcal { C } } ( c h ) = { \mathcal { C } } ( h )$ . This invariance holds at every block-input boundary, including $h _ { 0 }$ , and at the final block-output boundary because the stated head also canonicalizes the final block output.

Derivation. In the zero-floor limit, positive homogeneity of RMS gives ${ \mathcal { C } } ( c h ) = { \mathcal { C } } ( h )$ for every $c > 0$ . A GCT block including entry canonicalization has the same output on h and ch wherever the block’s canonicalizations are defined. Applying this identity at the first downstream block shows that a network with the stated dimensionless head is invariant to positive rescaling at every block-input boundary, including $h _ { 0 }$ . The head’s entry canonicalization also removes a rescaling at the final block-output boundary.

For an output with units, the output value must change with the declared scale. Under $( s _ { \ell } , \bar { h } _ { \ell } ) \mapsto ( c s _ { \ell } , \bar { h } _ { \ell } )$ , the reconstructed output changes from $s _ { \ell } \bar { h } _ { \ell }$ to $c s _ { \ell } \bar { h } _ { \ell }$ , while the normalized coordinate and the canonical-coordinate computation stay unchanged. This behavior is called equivariance.

For unitful outputs, the corresponding statement is equivariance rather than invariance. Under the scale-state action $\left( s _ { \ell } , \bar { h } _ { \ell } \right) \mapsto \left( c s _ { \ell } , \bar { h } _ { \ell } \right)$ , the reconstruction map $R ( s , \bar { h } ) : = s \bar { h }$ obeys $R ( c s , \bar { h } ) = c R ( s , \bar { h } )$ , while the canonical coordinate is fixed. The reconstruction $s _ { \ell } \bar { h } _ { \ell }$ follows the declared scale change exactly.

## F.7 Further examples of local typing rules

Independent scale factors. Each element $g = ( g _ { 1 } , \dotsc , g _ { K } ) \in G$ assigns a multiplier $g _ { k }$ to each chosen scale. Each multiplier can be chosen independently as any positive real number. Examples include the residual-stream scale of each stage, the target scale of each task head, and per-block parameter scales.

A single residual-stream unit. The simplest rescaling group has one independent scale factor, $K = 1$ . Let the single factor g rescale the residual stream of a Pre-LN block, and write $U _ { \mathrm { r e s } }$ for the resulting stream unit. Under $h _ { \ell } \mapsto g h _ { \ell }$ , the identity $N ( g h _ { \ell } ) = N ( h _ { \ell } )$ shows that the normalizer output stays unchanged and is dimensionless. To produce an output that can be added to $h _ { \ell } .$ , the branch $F _ { \ell }$ must restore the stream unit through a map $1  U _ { \mathrm { r e s } }$ . This declaration concerns the chosen activation rescaling and applies even when the input data, such as token identifiers, have no physical units. Sec. 2.2 generalizes the example into local typing rules.

Unit types and empirical scales. We distinguish a tensor’s unit type from the tensor’s empirical scale. The type u(x) specifies how x transforms under $G$ and can be read from the graph. The empirical scale $s _ { x }$ is a measured magnitude with the same type as the tensor, $u ( s _ { x } ) = u ( x )$ , making $x / s _ { x }$ dimensionless. For example, multiplying $h _ { \ell }$ by $g$ also multiplies the tensor’s RMS value by g, giving $\mathrm { R M S } ( g h _ { \ell } ) = g \mathrm { R M S } ( h _ { \ell } )$ . The tensor and the corresponding RMS share the unit $U _ { \mathrm { r e s } } .$ , while a forward pass supplies the actual RMS value. The attention scales $s _ { Q } , s _ { K }$ , stream scale $s _ { \ell } ,$ and characteristic loss scales $S _ { i }$ used below are empirical scales in this sense. A single RMS scale applies to a block whose coordinates share one scalar unit. Mixed-unit tensors require separate scales for shared-unit blocks, or a unit-compatible metric.

Parameters and explicit converters. Parameters follow the same rule. In $y : = f _ { \phi } ( x )$ , we treat $\phi$ as another input and declare the parameter transformation $\phi \mapsto \rho _ { \phi } ( g ) \phi$ . For example, in a linear map $y = W x$ between scalar-unit blocks, the weights must convert the input unit $U _ { \mathrm { i n } }$ into the output unit $U _ { \mathrm { o u t } } .$ which requires $u ( W ) = U _ { \mathrm { o u t } } U _ { \mathrm { i n } } ^ { - 1 }$ . Fixing these units before checking the graph makes mismatches detectable because an addition with unequal input units requires a converter. We keep unit conversion explicit by separating dimensionless mixing weights W<sup>¯</sup> from scale converters s. The factorization $W = s { \bar { W } }$ has $u ( \bar { W } ) = 1$ and $u ( s ) = U _ { \mathrm { o u t } } U _ { \mathrm { i n } } ^ { - 1 }$ , as developed in Sec. 4.

Normalization and output units. Consider $N ( x ) : = x / \mathrm { R M S } ( x )$ on a nonzero shared-unit block. Here a gauge is a convention for representing tensors that difer only by the declared rescaling. Gauge fixing chooses one representative from each such family. Here we divide by RMS, which maps every cx, $c > 0 .$ , to the same RMS-one tensor. We call this mapping to a fixed representative canonicalization. RMSNorm applies this division, yielding dimensionless coordinates before the learnable RMSNorm gain. LayerNorm also removes a common coordinate shift by subtracting the mean. Learnable gains specify the output unit. For an afine normalization $\gamma \odot N ( x ) + \beta$ the output unit is the unit of $\gamma ,$ , provided that the bias $\beta$ is assigned the same unit.

Activations and transition scales. The rectified linear unit (ReLU) is positively homogeneous and satisfies ReL $. \mathrm { U } ( c x ) = c \mathrm { R e L U } ( x )$ for $c \geq 0$ . Under the positive rescalings in G, the ReLU output scales by the same factor as the ReLU input, preserving the scalar unit.

Standard non-homogeneous activations, including sigmoid, tanh, GELU, and SiLU, require dimensionless arguments. The formula for each function contains a fixed transition scale. Changing a unitful input’s units can move the same underlying value to a diferent part of the nonlinear response curve. Evaluating the nonlinear response at $x / s$ avoids this change because x and s rescale together. Unit-consistent forms are $\mathrm { G E L U } _ { s } ( x ) : = x \Phi ( x / s )$ and $\mathrm { S i L U } _ { s } ( x ) : = x \sigma ( x / s )$ with $u ( s ) = u ( x )$

## G PROOFS

## G.1 Proof of Proposition 1

Proof of Proposition 1. Evaluating the graph on source values transformed by the declared source actions places every node at the corresponding transformed state, by topological substitution through the unit-consistent primitives. Fix $g \in G$ and abbreviate $R _ { i } : = \rho _ { i } ( g )$ and $R _ { y } : = \rho _ { y } ( g )$ . Here ⊤ denotes the adjoint with respect to the fixed inner products on the node spaces, or the usual matrix transpose in orthonormal coordinates. At a diferentiability point, diferentiate the local equivariance identity in Eq. 2 with respect to the primitive’s ith argument. With $x ^ { \prime } : = ( R _ { 1 } x _ { 1 } , \ldots , R _ { m } x _ { m } )$ , the chain rule gives

$$
D _ { i } f ( { x } ^ { \prime } ) R _ { i } = R _ { y } D _ { i } f ( { x } ) , \qquad \mathrm { h e n c e } \qquad D _ { i } f ( { x } ^ { \prime } ) = R _ { y } D _ { i } f ( { x } ) R _ { i } ^ { - 1 } .\tag{G.1}
$$

Let $\bar { y }$ denote the reverse-mode cotangent at $y .$ The cotangent at the dimensionless loss is the scalar 1, which is fixed by the dual action. Suppose, in reverse topological order, that $\bar { y } ^ { \prime } = R _ { y } ^ { - \top } \bar { y }$ . The contribution of the edge $x _ { i } \to y$ to the transformed parent cotangent follows from Eq. G.1:

$$
D _ { i } f ( \boldsymbol { x } ^ { \prime } ) ^ { \top } \bar { \boldsymbol { y } } ^ { \prime } = R _ { i } ^ { - \top } D _ { i } f ( \boldsymbol { x } ) ^ { \top } R _ { y } ^ { \top } R _ { y } ^ { - \top } \bar { \boldsymbol { y } } = R _ { i } ^ { - \top } D _ { i } f ( \boldsymbol { x } ) ^ { \top } \bar { \boldsymbol { y } } .
$$

If a node has several children, the node’s reverse-mode cotangent is the sum of the child contributions, and every summand has the same factor $R _ { i } ^ { - \top }$ . Reverse induction yields $\bar { x } _ { v } ^ { \prime } = \rho _ { v } ( g ) ^ { - \top } \bar { x } _ { v }$ at every node, which is the claimed dual transformation law for $\nabla _ { x _ { v } } { \mathcal { L } } .$ □

## G.2 Proof of Theorem 1

Proof of Theorem 1. For every node, define the equivariance error $E _ { v } : = \| \widetilde { x } _ { v } - \rho _ { v } ( g ) x _ { v } \| _ { v }$ . At a source $s \in S ,$ the transformed evaluation is defined by $\widetilde { x } _ { s } : = \rho _ { s } ( g ) x _ { s }$ , hence $E _ { s } = 0$ . For a nonsource node, insert and subtract the value of the node’s primitive on the ideally transformed parent tuple. The triangle inequality, Eq. E.1, and Eq. E.2 give

$$
\begin{array} { r l } & { E _ { v } = \| f _ { v } ( ( \widetilde { x } _ { u } ) _ { u } ) - \rho _ { v } ( g ) x _ { v } \| _ { v } } \\ & { \quad \le \| f _ { v } ( ( \widetilde { x } _ { u } ) _ { u } ) - f _ { v } ( ( \rho _ { u } ( g ) x _ { u } ) _ { u } ) \| _ { v } + \delta _ { v } ( g , x ) } \\ & { \quad \le \delta _ { v } ( g , x ) + \displaystyle \sum _ { u \in \mathrm { p a } ( v ) } K _ { v u } E _ { u } . } \end{array}\tag{G.2}
$$

All parent indices in this recurrence precede v in a topological ordering.

We now unroll Eq. G.2. We claim, for every node v, that

$$
E _ { v } \le \sum _ { w \notin S } \delta _ { w } ( g , x ) \sum _ { p \in \mathcal { P } ( w , v ) } \prod _ { ( a  b ) \in p } K _ { b a } ,\tag{G.3}
$$

where a sum over an empty path set is zero and the empty path from v back to v has product one. The claim is immediate for a source. Suppose the claim holds for all parents of a nonsource v. Substitution into $\operatorname { E q . }$ G.2 appends the edge $u  v$ to every path ending at each parent u. Every nonempty directed path ending at v has a unique last edge and is obtained exactly once by this path extension. The term $\delta _ { v }$ supplies the empty path. This construction proves Eq. G.3 by topological induction. Setting v = o yields Eq. E.3. □

## G.3 Proofs of Theorem 2 and Corollary 1

Proof of Theorem 2. Expanding the invariance identity and comparing the coeficient of each $\ell _ { i }$ gives

$$
w _ { i } ( D ( t ) S ) = e ^ { - a _ { i } ^ { \top } t } w _ { i } ( S ) .
$$

We now solve this covariance equation completely. Define $\beta _ { i } ( S ) : = S _ { i } w _ { i } ( S )$ . Because the ith coordinate of $D ( t ) S$ is $e ^ { a _ { i } ^ { \top } t } S _ { i }$ , the covariance equation is equivalent to $\beta _ { i } ( D ( t ) S ) = \beta _ { i } ( S )$ for all $S \in \mathbb { R } _ { + } ^ { m }$ and $t \in \mathbb { R } ^ { K }$ . Pass to logarithmic coordinates $y : = \log S \in \mathbb { R } ^ { m }$ and put $b _ { i } ( y ) : = \beta _ { i } ( e ^ { y } )$ , with the exponential also componentwise. Since $\dot { \log ( D ( t ) S ) } = y + A ^ { \top } t$ , this identity says precisely that $b _ { i }$ is constant on every afine orbit $y + \operatorname { i m } A ^ { \top }$

We now identify those orbits without any regularity assumption on w. The columns of N span ker $A ,$ so ker $N ^ { \top } = ( \operatorname { i m } N ) ^ { \bot } = ( \ker A ) ^ { \bot } = \operatorname { i m } A ^ { \top }$ , where the last equality is the finite-dimensional fundamental theorem of linear algebra. Therefore

$$
N ^ { \top } y = N ^ { \top } y ^ { \prime } \quad \Longleftrightarrow \quad y ^ { \prime } - y \in \mathrm { i m } A ^ { \top } .
$$

Thus the fibers of $y \mapsto N ^ { \top } y$ are exactly the G-orbits. Since N has full column rank, $N ^ { \top } : \mathbb { R } ^ { m }  \mathbb { R } ^ { m - s }$ is surjective. We may define, for each $z \in \mathbb { R } ^ { m - s } , \psi _ { i } ( z ) : = b _ { i } ( y )$ for any y satisfying $N ^ { \top } y = z$ . The orbit characterization makes this definition independent of the chosen $y ,$ and the definition yields

$$
\beta _ { i } ( S ) = \psi _ { i } ( N ^ { \top } \log S ) , \qquad w _ { i } ( S ) = \frac { 1 } { S _ { i } } \psi _ { i } ( N ^ { \top } \log S ) ,
$$

which proves necessity of $\operatorname { E q }$ . E.4.

Conversely, let the weights have the form in Eq. E.4. Since $A N = 0$ , we have $N ^ { \top } \log ( D ( t ) S ) = N ^ { \top }$ log $S +$ $N ^ { \top } A ^ { \top } t = N ^ { \top }$ log S. The argument of every $\psi _ { i }$ is invariant, while the prefactor transforms as $\begin{array} { r } { ( D ( t ) S ) _ { i } ^ { - 1 } = } \end{array}$ $e ^ { - a _ { i } ^ { \top } t } S _ { i } ^ { - 1 }$ . Hence the coeficient covariance holds, and expanding the scalarization proves the invariance identity. This direction establishes suficiency and completes the characterization.

Finally, for any $n \in$ ker A,

$$
\prod _ { j = 1 } ^ { m } ( D ( t ) S ) _ { j } ^ { n _ { j } } = \exp ( t ^ { \top } A n ) \prod _ { j = 1 } ^ { m } S _ { j } ^ { n _ { j } } = \prod _ { j = 1 } ^ { m } S _ { j } ^ { n _ { j } } .
$$

Taking logarithms gives $n ^ { \top }$ log S. Conversely, every invariant linear combination of the log-scales has coeficient vector in ker $A ,$ so the coordinates supplied by a basis N exhaust the dimensionless monomial groups. □

Proof of Corollary 1. If rank $A = m$ , then $m - s = 0$ and $\mathbb { R } ^ { m - s } = \mathbb { R } ^ { 0 }$ is a one-point space. Every $\psi _ { i }$ is a constant $\alpha _ { i }$ , giving Eq. E.5. The coeficient comparison at the start of the theorem proof shows there are no other invariant linear rules. □

## G.4 Proof of Proposition 2

Proof of Proposition 2. For the preconditioned update, the displacement computed in the transformed run is $u ^ { g } = - \eta P ^ { g } R ^ { - \top } q ,$ whereas equivariance requires the displacement to equal $R u = - \eta R P q .$ Equality for every covector q is equivalent to the operator identity $P ^ { g } R ^ { - \top } = R P$ . Right multiplication by $R ^ { \top }$ gives $\begin{array} { r } { P ^ { \acute { g } } = R P R ^ { \acute { \intercal } } } \end{array}$ proving both necessity and suficiency of Eq. E.6.

Ordinary SGD with a common learning rate has $P ^ { g } = P = I$ , so Eq. E.6 reduces to $I = R R ^ { \top }$ . If scalar learning rates may difer, the corresponding calculation is $\eta ^ { g } I = \eta R R ^ { \top }$ . For $R = c I$ , the condition is exactly $\eta ^ { g } = c ^ { 2 } \eta$ .□

## G.5 Proofs of Proposition 3, Theorem 3, and Corollary 2

Proof of Proposition 3. Suppose $g ( w ) \neq 0$ . The unit vectors $\bar { w } = w / r$ and $\bar { g } = g ( w ) / \| g ( w ) \|$ are orthogonal by Eq. E.7, and the update can be written

$$
w ^ { + } = a r \bar { w } - \eta p ( r ) \| g ( w ) \| \bar { g } .
$$

Pythagoras gives Eq. E.10. Define $\vartheta \in ( 0 , \pi / 2 )$ by

$$
\cos \vartheta = \frac { a r } { \| w ^ { + } \| } , \qquad \sin \vartheta = \frac { \eta p ( r ) \| g ( w ) \| } { \| w ^ { + } \| } .
$$

The ratio of the displayed sine and cosine is the tangent in Eq. E.9, and division of the preceding expression for $w ^ { + }$ by the norm of $w ^ { + }$ proves that equation. Because w¯ and $- \bar { g }$ span a two-dimensional plane through the origin, the curve cos $\theta \bar { w } - \sin \theta \bar { g }$ is the unit sphere’s great circle through w¯ in direction $- \bar { g }$ . The step is exactly geodesic. □

Proof of Theorem 3. The gradient scaling law gives the update from cw as

$$
( c w ) ^ { + } = a c w - \eta p ( c r ) c ^ { - 1 } g ( w ) .
$$

If $p ( c r ) = c ^ { 2 } p ( r )$ , the right side of this update equals $c [ a w - \eta p ( r ) g ( w ) ] = c w ^ { + }$ . Thus quadratic covariance implies raw equivariance, which immediately implies canonical-direction invariance.

Conversely, at a nonstationary point the normalized gradient direction is unchanged by positive rescaling, and Eq. E.9 applied at cw gives

$$
\tan \vartheta ( c w ) = \frac { \eta p ( c r ) \| c ^ { - 1 } g ( w ) \| } { a \| c w \| } = \frac { p ( c r ) } { c ^ { 2 } p ( r ) } \tan \vartheta ( w ) .
$$

Both angles lie in $( 0 , \pi / 2 )$ and use the same ordered orthonormal pair $( { \bar { w } } , - { \bar { g } } )$ , so equality of the canonical directions forces equality of the angles. Thus canonical-direction invariance implies $p ( c r ) = c ^ { 2 } p ( r )$ wherever the test point is nonstationary. The invariance hypothesis applies at every radius. Because dim $V \geq 2$ , for any $r > 0$ choose orthonormal $u , v .$ , put $w = r u$ , and use the smooth scale-invariant objective $J _ { v } ( x ) : = \langle v , x \rangle / \| x \|$ . At w, the objective’s gradient is $v / r \ne 0$ . Hence the angle identity proves quadratic covariance for every $c , r > 0$ . This implication closes the cycle and proves the equivalence of the three conditions. □

Proof of Corollary 2. Quadratic covariance with $r = 1$ and $c = r ^ { \prime } \mathrm { g i v e s } p ( r ^ { \prime } ) = r ^ { \prime 2 } p ( 1 )$ . Thus quadratic covariance implies the squared-scale form with $\kappa : = p ( 1 ) > 0$ , and the reverse implication is immediate. For $p ( r ) = \kappa r ^ { 2 }$ , the scaled-update calculation proves raw equivariance directly and Eq. E.9 reduces to tan $\vartheta = ( \eta \kappa / a ) r \| g ( w ) \|$ . The product is invariant because $\| c w \| \| g ( c w ) \| = r \| g ( w ) \|$ . This computation proves the claimed unique family and the gauge invariance of the quotient angle. □

Direction-dependent extension and Euclidean case. For $p \equiv 1$ , the scaled-angle identity in the theorem proof gives tan $\vartheta ( c w ) = c ^ { - 2 } \tan \vartheta ( w )$ , proving the Euclidean comparison of App. E.3. Now replace the radial rule $p ( r )$ by a general scalar rule $\widetilde { p } ( w ) > 0$ . The same scaled-update calculation shows that raw gauge equivariance is equivalent to $\widetilde { p } ( c w ) = c ^ { 2 } \widetilde { p } ( w )$ Writing $w = r u$ with $r = \| w \|$ and $\lVert u \rVert = 1$ then gives ${ \widetilde { p } } ( w ) = r ^ { 2 } { \widetilde { p } } ( u ) =$ $\| \boldsymbol { w } \| ^ { 2 } \varphi ( \boldsymbol { w } / \| \boldsymbol { w } \| )$ , where $\varphi$ is an arbitrary positive function on the unit sphere. Every such rule satisfies the covariance law. This derivation proves the qualification following Cor. 2 and shows why radiality forces $\varphi$ to be constant.

## G.6 Proof of Proposition 4

Proof of Proposition $\it 4 .$ The UVI hypothesis implies $\alpha _ { k } \le \alpha ^ { \star } e ^ { B } = a$ . Iterating the exact recursion in Eq. 6 gives

$$
\biggl ( \frac { s _ { N } } { s _ { 0 } } \biggr ) ^ { 2 } = \prod _ { k = 1 } ^ { N } ( 1 + \alpha _ { k } ^ { 2 } + 2 \alpha _ { k } c _ { k } ) ,\tag{G.4}
$$

and hence, for a nondegenerate trajectory,

$$
2 \log { \frac { s _ { N } } { s _ { 0 } } } = \sum _ { k = 1 } ^ { N } \log ( 1 + \alpha _ { k } ^ { 2 } + 2 \alpha _ { k } c _ { k } ) .\tag{G.5}
$$

The near-orthogonality assumption gives the two-sided estimate

$$
1 + ( 1 - 2 \kappa ) \alpha _ { k } ^ { 2 } \le 1 + \alpha _ { k } ^ { 2 } + 2 \alpha _ { k } c _ { k } \le 1 + ( 1 + 2 \kappa ) \alpha _ { k } ^ { 2 } .\tag{G.6}
$$

Because $\kappa < 1 / 2$ , the left side is at least one, so every factor in Eq. G.4 is at least one and $s _ { N } \geq s _ { 0 }$ . For the upper bound, use Eq. G.5, the right side of Eq. G.6, and again $\log ( 1 + x ) \leq x { \mathrm { : } }$

$$
\begin{array} { l } { 2 \log \displaystyle \frac { s _ { N } } { s _ { 0 } } \le \sum _ { k = 1 } ^ { N } \log ( 1 + ( 1 + 2 \kappa ) \alpha _ { k } ^ { 2 } ) \qquad } \\ { \displaystyle \le ( 1 + 2 \kappa ) \sum _ { k = 1 } ^ { N } \alpha _ { k } ^ { 2 } \le ( 1 + 2 \kappa ) N a ^ { 2 } = ( 1 + 2 \kappa ) e ^ { 2 B } . } \end{array}
$$

Division by two proves the claim.

Finally, choose $\alpha _ { k } = a$ and $c _ { k } = \kappa a$ for every k. This choice is admissible for all suficiently large N at fixed B. The right inequality in $\mathrm { E q . ~ G . 6 }$ is then exact and

$$
\log \frac { s _ { N } } { s _ { 0 } } = \frac { N } { 2 } \log ( 1 + ( 1 + 2 \kappa ) a ^ { 2 } ) = \frac { 1 + 2 \kappa } { 2 } e ^ { 2 B } + O ( N a ^ { 4 } ) .
$$

Since $N a ^ { 4 } = e ^ { 4 B } / N \to 0$ , the chosen steps and alignments attain the stated upper constant to leading order.

## G.7 Proofs of Theorem 4 and Corollary 3

Proof of Theorem $\it 4 .$ Assume first that F satisfies Eq. E.16. On $\mathbb { S } : = \Pi _ { i } \mathbb { S } _ { j }$ , define f by restriction, $f ( z ) : = F ( z )$ Given an arbitrary $x \in X$ , put $s _ { j } ~ : = ~ r _ { j } ( x _ { j } ) ~ > ~ 0$ and $z _ { j } : = x _ { j } / \check { s _ { j } } = \mathcal { C } _ { 0 } ( x _ { j } ) \in \mathbb { S } _ { j }$ . Then $ { \boldsymbol { { x } } } \ = \  { \boldsymbol { { s } } } \cdot  { \boldsymbol { z } }$ for $s : = ( s _ { 1 } , \ldots , s _ { m } ) \in G$ , so equivariance gives

$$
F ( x ) = F ( s \cdot z ) = \chi _ { q } ( s ) F ( z ) = \left( \prod _ { j = 1 } ^ { m } r _ { j } ( x _ { j } ) ^ { q _ { j } } \right) f ( { \mathcal C } _ { 0 } ( x _ { 1 } ) , \dots , { \mathcal C } _ { 0 } ( x _ { m } ) ) ,
$$

which is Eq. E.17.

Conversely, suppose Eq. E.17 holds. Positive homogeneity of RMS implies, for $g _ { j } > 0 , r _ { j } ( g _ { j } x _ { j } ) = g _ { j } r _ { j } ( x _ { j } )$ and $\mathcal { C } _ { 0 } ( g _ { j } x _ { j } ) = \mathcal { C } _ { 0 } ( x _ { j } )$ . Substituting these identities into Eq. E.17 yields

$$
F ( g \cdot x ) = \left( \prod _ { j } ( g _ { j } r _ { j } ( x _ { j } ) ) ^ { q _ { j } } \right) f ( \mathcal { C } _ { 0 } ( x _ { 1 } ) , \ldots , \mathcal { C } _ { 0 } ( x _ { m } ) ) = \chi _ { q } ( g ) F ( x ) ,
$$

so $F$ is equivariant. Uniqueness is forced by restriction to the canonical section. ${ \mathrm { I f ~ } } z \in { \mathbb { S } } .$ , then every $r _ { j } ( z _ { j } ) = 1$ and Eq. E.17 gives $f ( z ) = F ( z )$ □

Proof of Corollary 3. Now take $q = 0$ . The product S is compact because each $V _ { j }$ is finite-dimensional, and $\mathcal { C } _ { 0 } : X  \mathbb { S }$ is continuous and surjective. Define the pullback $P : C ( \mathbb { S } , W ) \to ^ { \cdot } C ( X , W ) ^ { G }$ by $( P f ) ( x ) : =$ $f ( \mathcal { C } _ { 0 } ( x _ { 1 } ) , \dots , \mathcal { C } _ { 0 } ( x _ { m } ) )$ . The already-proved factorization says that restriction to S is the inverse of P, so $P$ is a bijection. Moreover, for any $f , h \in C ( \mathbb { S } , W )$ , surjectivity of $\mathcal { C } _ { 0 }$ gives

$$
\begin{array} { r l } & { \| P f - P h \| _ { \infty , X } = \underset { x \in X } { \operatorname* { s u p } } \| f ( \mathcal { C } _ { 0 } ( x ) ) - h ( \mathcal { C } _ { 0 } ( x ) ) \| _ { W } } \\ & { \qquad = \underset { z \in \mathbb { S } } { \operatorname* { s u p } } \| f ( z ) - h ( z ) \| _ { W } = \| f - h \| _ { \infty , \mathbb { S } } . } \end{array}
$$

Thus P is an isometric bijection. If a class H is uniformly dense in $C ( \mathbb { S } , W )$ , approximate the unique restriction $f : = F | _ { \mathbb { S } }$ of any continuous invariant F by elements of $\mathcal { H } .$ The preceding norm identity transfers the same approximation error unchanged to all of X. This transfer proves the expressivity assertion. □

## G.8 Proofs of Theorem 5 and Corollary 4

Proof of Theorem 5. Write $y ( x ) : = \log r ( x ) \in \mathbb { R } ^ { m }$ . Positive homogeneity of RMS and Eq. E.18 give $y ( e ^ { t } \cdot x ) =$ $y ( x ) + A ^ { \top } t$ and $z _ { j } ( ( e ^ { t } \cdot x ) _ { j } ) = z _ { j } ( x _ { j } )$ . Since $A N = 0$ , multiplication of the first identity by $N ^ { \top }$ shows that $\mathcal { Q } ( e ^ { t } \cdot x ) = \mathcal { Q } ( x )$ . Thus Q is invariant.

To prove completeness, suppose $\mathcal { Q } ( x ) = \mathcal { Q } ( x ^ { \prime } )$ and put $y : = y ( x ) , y ^ { \prime } : = y ( x ^ { \prime } )$ . Equality of the last coordinates gives $N ^ { \top } ( y ^ { \prime } - y ) = 0$ . The columns of N span ker $A _ { i }$ , hence the finite-dimensional fundamental theorem of linear algebra gives ker $N ^ { \top } = ( \ker A ) ^ { \bot } = \operatorname { i m } A ^ { \top }$ . There is therefore a $t \in \mathbb { R } ^ { K }$ such that $y ^ { \prime } - y = A ^ { \top } t$ . Equality of the canonical directions and the identity $x _ { j } = e ^ { y _ { j } } z _ { j } ( x _ { j } )$ now imply

$$
\begin{array} { r } { x _ { j } ^ { \prime } = e ^ { y _ { j } ^ { \prime } } z _ { j } ( x _ { j } ^ { \prime } ) = e ^ { a _ { j } ^ { \intercal } t } e ^ { y _ { j } } z _ { j } ( x _ { j } ) = e ^ { a _ { j } ^ { \intercal } t } x _ { j } } \end{array}
$$

for every $j ,$ so $x ^ { \prime } = e ^ { t } \cdot x$ . The reverse implication is the invariance already proved, establishing Eq. E.20.

Proof of Corollary 4. If $q \not \in$ im $A ,$ finite-dimensional orthogonality gives a vector $t \in$ ker $A ^ { \top }$ with $q ^ { \top } t \neq 0$ . Then $e ^ { t } \cdot x = x$ for every $x \in X$ , while equivariance would require $F ( x ) = F ( e ^ { t } \cdot x ) = e ^ { q ^ { \top } t } F ( x )$ . Since $e ^ { q ^ { \intercal } t } \neq 1$ , the equivariance identity forces $F \equiv 0$

Now suppose $q \in$ im A. Write $y ( x ) : = \log r ( x )$ , fix p with $A p = q .$ and define the positive scale monomial $\begin{array} { r } { M _ { p } ( x ) : = e ^ { p ^ { \top } y ( x ) } = \prod _ { i = 1 } ^ { m } r _ { j } ( x _ { j } ) ^ { p _ { j } } } \end{array}$ . The scale transformation above yields $M _ { p } ( e ^ { t } \cdot x ) = e ^ { p ^ { \top } A ^ { \top } t } M _ { p } ( x ) = e ^ { q ^ { \top } t } M _ { p } ( x )$ If F is equivariant, $H \dot { : } = F / M _ { p }$ is consequently invariant. By the completeness of $\mathcal { Q } ,$ H is constant on each fiber of Q, so there is a unique function $f$ on the image of Q such that $H = f \circ \mathcal { Q }$

That image is the entire stated domain. Indeed, when $m - s > 0$ , let $C : = N ( N ^ { \top } N ) ^ { - 1 } \in \mathbb { R } ^ { m \times ( m - s ) }$ , which satisfies $N ^ { \top } C = I _ { m - s }$ . When $m - s = 0$ , interpret C as the unique empty linear map. Given arbitrary $\begin{array} { r } { z = ( z _ { 1 } , \ldots , z _ { m } ) \in \prod _ { j } \mathbb { S } _ { j } } \end{array}$ and $u \in \mathbb { R } ^ { m - s }$ , set $y : = C u$ and $x _ { j } : = e ^ { y _ { j } } z _ { j }$ . Then $z _ { j } ( x _ { j } ) = z _ { j }$ and $N ^ { \top } y = u$ , proving $\boldsymbol { \mathcal { Q } } ( \boldsymbol { x } ) = ( z , u )$ . This construction establishes Eq. E.21 and uniqueness of $f$ on all of $\begin{array} { r } { ( \prod _ { j } \mathbb { S } _ { j } ) \times \mathbb { R } ^ { m - s } } \end{array}$

Conversely, if Eq. E.21 holds for any $f ,$ then $\mathcal { Q }$ is invariant and the preceding transformation identity shows immediately that $F ( e ^ { t } \cdot x ) = e ^ { q ^ { \top } t } F ( x )$ . This converse direction completes the necessary-and-suficient factorization.

Continuity in the forward direction follows from the explicit continuous section just constructed:

$$
f ( z , u ) = e ^ { - p ^ { \top } C u } F ( e ^ { ( C u ) _ { 1 } } z _ { 1 } , \dots , e ^ { ( C u ) _ { m } } z _ { m } ) .
$$

The displayed formula makes $f$ continuous whenever $F$ is continuous. The converse follows directly from Eq. E.21, since RMS, canonicalization on nonzero inputs, logarithm, and the exponential prefactor are continuous. □

## G.9 Proofs of Theorem 6 and Corollary 5

Proof of Theorem 6. By construction, $e \perp \perp x$ and $x + e = ( 1 + a ) x + e _ { \perp }$ . Writing $\rho = \| e _ { \perp } \| / \| x \|$ , we obtain

$$
\| x + e \| ^ { 2 } = \| x \| ^ { 2 } ( ( 1 + a ) ^ { 2 } + \rho ^ { 2 } ) , \qquad { \frac { \langle x , x + e \rangle } { \| x \| \| x + e \| } } = { \frac { 1 + a } { \sqrt { ( 1 + a ) ^ { 2 } + \rho ^ { 2 } } } } .
$$

The assumption $a > - 1$ makes $x + e$ nonzero and fixes the positive orientation of the radial component of $x + e$ Both $\mathcal { C } _ { 0 } ( x )$ and $\mathcal { C } _ { 0 } ( x + e )$ have squared norm d. The polarization identity gives

$$
\begin{array} { l } { \displaystyle { d _ { \mathbb { S } } ( \mathscr { C } _ { 0 } ( x + e ) , \mathscr { C } _ { 0 } ( x ) ) ^ { 2 } = \frac { 1 } { d } \| \mathscr { C } _ { 0 } ( x + e ) - \mathscr { C } _ { 0 } ( x ) \| ^ { 2 } } } \\ { \displaystyle { \quad \quad = 2 \left( 1 - \frac { \langle x , x + e \rangle } { \| x \| \| x + e \| } \right) = 2 \left( 1 - \frac { 1 + a } { \sqrt { ( 1 + a ) ^ { 2 } + \rho ^ { 2 } } } \right) , } } \end{array}
$$

which proves the equality in $\mathrm { E q . \ E . 2 2 }$

Set $t : = \rho / ( 1 + a ) \geq 0$ . For $s \geq 0$ , the function $\phi ( s ) : = s - 2 + 2 / \sqrt { 1 + s }$ satisfies $\phi ( 0 ) = 0$ and $\phi ^ { \prime } ( s ) =$ $1 - ( 1 + s ) ^ { - 3 / 2 } \geq 0$ . Thus $2 ( 1 - ( 1 + t ^ { 2 } ) ^ { - 1 / 2 } ) \leq t ^ { 2 }$ . Taking square roots proves the inequality in Eq. E.22. Moreover, $[ 2 ( 1 - ( 1 + t ^ { 2 } ) ^ { - 1 / 2 } ) ] ^ { 1 / 2 } = t + { \cal O } ( t ^ { 3 } ) \mathrm { ~ a s ~ } t  0 ^ { + }$ , and every suficiently small $t > 0$ is realizable in

dimension $d \geq 2$ by taking $a = 0$ and a tangential perturbation with $\| e _ { \bot } \| / \| x \| = t .$ . Hence no constant smaller than one can replace the unit constant in Eq. E.22 uniformly. When $e = a x$ with $a > - 1 , e _ { \perp } = 0$ and the exact formula is zero, proving radial rejection without an approximation. □

Proof of Corollary 5. The coordinatewise error hypothesis implies

$$
\operatorname { R M S } ( e ) = { \frac { \| e \| } { \sqrt { d } } } \leq { \frac { \Delta } { 2 } } , \qquad { \frac { \| e \| } { \| x \| } } \leq { \frac { \Delta } { 2 r ( x ) } } = \delta .
$$

Because $\delta < 1$ , the same bound also gives $\| e \| < \| x \|$ and hence $Q ( x ) = x + e \neq 0$ . Cauchy–Schwarz and orthogonal projection then give $| a | \le \delta$ and $\rho \leq \delta$ . Since $\delta < 1$ , we have $a > - 1$ and $1 + a \geq 1 - \delta$ . The K-Lipschitz property of $f$ and Thm. 6 give

$$
\| F ( Q ( x ) ) - F ( x ) \| _ { W } \leq K d _ { \mathbb { S } } ( \mathcal { C } _ { 0 } ( Q ( x ) ) , \mathcal { C } _ { 0 } ( x ) ) \leq K \frac { \delta } { 1 - \delta } ,
$$

which is $\operatorname { E q }$ . E.23.

## G.10 Proof of Proposition 6

Proof of Proposition 6. Fix the input batch and all parameters other than $W ,$ and set $W _ { t } : = e ^ { t } W$ . Independence of u from W implies $x _ { t } = W _ { t } u = e ^ { t } W u = e ^ { t } x$ . The assumed downstream invariance gives $\mathcal { L } ( W _ { t } ) = \mathcal { L } ( W )$ for every $t \in \mathbb { R }$ . This equality is a functional identity along the whole positive ray and uses no stationarity assumption.

Let $f ( t ) : = \mathcal { L } ( W _ { t } )$ . Wherever the first derivative exists, the chain rule for the Frobenius inner product gives

$$
0 = f ^ { \prime } ( t ) = \langle \nabla _ { W } \mathcal { L } ( W _ { t } ) , W _ { t } \rangle _ { F } .
$$

At $t = 0$ , the identity is the claimed radial Euler pairing.

Derivation of the untyped comparison. When downstream invariance is absent, because $x = W u$ is the only dependence on W, matrix calculus gives $\nabla _ { W } \mathcal { L } = ( \nabla _ { x } \mathcal { L } ) u ^ { \top }$ . Taking the Frobenius pairing with W gives

$$
\langle W , \nabla _ { W } \mathcal { L } \rangle _ { F } = \operatorname { t r } ( W ^ { \top } ( \nabla _ { x } \mathcal { L } ) u ^ { \top } ) = ( \nabla _ { x } \mathcal { L } ) ^ { \top } W u = \langle x , \nabla _ { x } \mathcal { L } \rangle ,
$$

which is the corresponding activation-space radial pairing.

## G.11 Proof of Proposition 5

Proof of Proposition 5. Let rounding ties be resolved arbitrarily. By definition of A, every coordinate lies in $[ - A , A ] = [ - q \Delta , q \Delta ]$ , so absmax quantization does not clip. Rounding to the nearest grid point gives the coordinatewise estimate $| Q _ { \Delta } ( h _ { t , i } ) - h _ { t , i } | \leq \Delta / 2$ . Summing the squared coordinate errors yields

$$
\| Q _ { \Delta } ( h _ { t } ) - h _ { t } \| _ { 2 } \leq \frac { \sqrt { d } \Delta } { 2 } = \frac { \sqrt { d } A } { 2 q } .\tag{G.7}
$$

On the other hand,

$$
A = \operatorname* { m a x } _ { s } \| h _ { s } \| _ { \infty } \leq \left( \operatorname* { m a x } _ { s } \frac { \| h _ { s } \| _ { \infty } } { r _ { s } } \right) \operatorname* { m a x } _ { s } r _ { s } = \kappa _ { \mathrm { a m p } } \operatorname* { m a x } _ { s } r _ { s } .\tag{G.8}
$$

Since $\| h _ { t } \| _ { 2 } = \sqrt { d } r _ { t }$ , division of $\operatorname { E q } .$ . G.7 by this norm followed by $\operatorname { E q } .$ . G.8 gives

$$
\frac { \| Q _ { \Delta } ( h _ { t } ) - h _ { t } \| _ { 2 } } { \| h _ { t } \| _ { 2 } } \leq \frac { \kappa _ { \mathrm { a m p } } } { 2 q } \frac { \operatorname* { m a x } _ { s } r _ { s } } { r _ { t } } .
$$

For $b \geq 2 , [ 2 ( 2 ^ { b - 1 } - 1 ) ] ^ { - 1 } \leq 2 ^ { - ( b - 1 ) }$ , proving the second bound in the proposition.

Sharpness and efective-bit interpretation. The scale-spread factor is attained by a simple construction. Take $d = 1$ , choose $A > 0$ , set $h _ { 1 } = A$ , and for any integer $m \in \{ 0 , \ldots , q - 1 \}$ set $h _ { 2 } = ( m + 1 / 2 ) \Delta$ with $\Delta = A / q$ The first token fixes the shared absmax and the second token lies halfway between two grid points, hence

$$
\frac { | Q _ { \Delta } ( h _ { 2 } ) - h _ { 2 } | } { | h _ { 2 } | } = \frac { 1 } { 2 m + 1 } = \frac { 1 } { 2 q } \frac { A } { | h _ { 2 } | } = \frac { \kappa _ { \mathrm { s c a l e } } } { 2 q } .
$$

This construction attains the linear dependence on $\kappa _ { \mathrm { s c a l e } } ,$ establishing the sharpness of the scale-spread factor. In the power-of-two relaxation $[ 2 ( 2 ^ { b - 1 } - 1 ) ] ^ { - 1 } \leq 2 ^ { - ( b - 1 ) }$ , multiplying the resolution by $\kappa _ { \mathrm { s c a l e } } = 2 ^ { \log _ { 2 } \kappa _ { \mathrm { s c a l e } } }$ is algebraically the same as reducing the efective resolution budget by $\log _ { 2 } \kappa _ { \mathrm { s c a l e } }$ bits. For finite $b ,$ the exact level-count statement remains the factor $\kappa _ { \mathrm { s c a l e } } / ( 2 q )$ above.

## G.12 Proofs of Lemma 1 and Theorem 7

Proof of Lemma 1. For token $t ,$ write $e _ { t } : = Q _ { \Delta } ( h _ { t } ) - h _ { t }$ . Nearest-grid rounding and the absmax choice imply $| ( e _ { t } ) _ { i } | \le \Delta / 2$ coordinatewise. Summing the squared coordinate errors gives

$$
\mathrm { R M S } ( e _ { t } ) \leq \frac { \Delta } { 2 } , \qquad \frac { \| e _ { t } \| _ { 2 } } { \| h _ { t } \| _ { 2 } } = \frac { \mathrm { R M S } ( e _ { t } ) } { r _ { t } } \leq \delta _ { t } : = \frac { \Delta } { 2 r _ { t } } \leq \delta _ { * } .\tag{G.9}
$$

If $Q _ { \Delta } ( h _ { t } ) = 0$ , the middle ratio in Eq. G.9 equals one, contradicting $\delta _ { * } < 1$ . Hence every canonicalized quantized token is well-defined.

Use the radial–tangential decomposition from Thm. 6, $a _ { t } : = \langle h _ { t } , e _ { t } \rangle / \| h _ { t } \| _ { 2 } ^ { 2 } , ( e _ { t } ) _ { \bot } : = e _ { t } - a _ { t } h _ { t } ,$ and $\rho _ { t } : =$ $\| ( e _ { t } ) _ { \perp } \| _ { 2 } / \| h _ { t } \| _ { 2 }$ . Cauchy–Schwarz and contractivity of orthogonal projection give $| a _ { t } | \leq \| e _ { t } \| _ { 2 } / \| h _ { t } \| _ { 2 } \leq \delta _ { t }$ and $\rho _ { t } \leq \| e _ { t } \| _ { 2 } / \| h _ { t } \| _ { 2 } \leq \delta _ { t }$ . Thus $a _ { t } > - 1$ and $1 + a _ { t } \geq 1 - \delta _ { t }$ . Applying the exact quotient perturbation bound in Eq. E.22 and then the Lipschitz property of $f$ yields

$$
\Vert f ( \mathcal C _ { 0 } ( Q _ { \Delta } ( h _ { t } ) ) ) - f ( \mathcal C _ { 0 } ( h _ { t } ) ) \Vert _ { W } \le K \frac { \rho _ { t } } { 1 + a _ { t } } \le K \frac { \delta _ { t } } { 1 - \delta _ { t } } \le K \frac { \delta _ { * } } { 1 - \delta _ { * } } .
$$

Taking the maximum over t proves the first inequality in Eq. E.24. Moreover, $A = \operatorname* { m a x } _ { t } \| h _ { t } \| _ { \infty } \leq \kappa _ { \mathrm { a m p } } \operatorname* { m a x } _ { t } r _ { t } .$ and therefore

$$
\delta _ { * } = \frac { A } { 2 q \operatorname* { m i n } _ { t } r _ { t } } \leq \frac { \kappa _ { \mathrm { a m p } } \kappa _ { \mathrm { s c a l e } } } { 2 q } ,
$$

which proves the second inequality.

Proof of Theorem 7. We first upper-bound the worst-case quantity. Every tensor in the definition of $\mathfrak { E } _ { q } ( \Gamma )$ satisfies $\delta _ { * } \leq \sqrt { 2 } \Gamma / ( 2 q ) \leq \sqrt { 2 } / 4 < 1$ . The already-proved bound with $K = 1$ , together with monotonicity of $x \mapsto x / ( 1 - x )$ on [0, 1), gives

$$
\mathfrak { E } _ { q } ( \Gamma ) \le \frac { \sqrt { 2 } \Gamma / ( 2 q ) } { 1 - \sqrt { 2 } \Gamma / ( 2 q ) } = \frac { \sqrt { 2 } \Gamma } { 2 q - \sqrt { 2 } \Gamma } .
$$

For the matching lower bound, fix $q \geq 4$ and $2 \le \Gamma \le q / 2$ , and put $k : = \lceil q / \Gamma \rceil$ . Then $2 \leq k \leq q - 1$ . For an arbitrary $\Delta > 0$ , take $d = T = 2$ and

$$
h _ { 1 } = ( q \Delta , 0 ) , \qquad h _ { 2 } = ( \Delta / 3 , k \Delta ) .\tag{G.10}
$$

The shared absmax $A = q \Delta$ gives a shared quantization step of exactly $\Delta .$ . Nearest-grid rounding is unambiguous at $\Delta / 3$ , and

$$
Q _ { \Delta } ( h _ { 1 } ) = h _ { 1 } , \qquad Q _ { \Delta } ( h _ { 2 } ) = ( 0 , k \Delta ) .\tag{G.11}
$$

The two RMS scales are $r _ { 1 } = q \Delta / \sqrt { 2 }$ and $r _ { 2 } = \Delta \sqrt { k ^ { 2 } + 1 / 9 } / \sqrt { 2 }$ . The ratio of these RMS scales gives the scale spread of the two tokens,

$$
\kappa _ { \mathrm { s c a l e } } = { \frac { r _ { 1 } } { r _ { 2 } } } = { \frac { q } { \sqrt { k ^ { 2 } + 1 / 9 } } } .\tag{G.12}
$$

Because $k \geq q / \Gamma ,$ , we have $\kappa _ { \mathrm { s c a l e } } \leq q / k \leq \Gamma$ . For the reverse comparison, set $x : = q / \Gamma \geq 2$ . Since $k \leq x + 1$ and $( x + 1 ) ^ { 2 } + 1 / \overset { \cdot } { 9 } \leq 4 x ^ { 2 }$ for $x \geq 2 .$ , we obtain $\sqrt { k ^ { 2 } + 1 / 9 } \leq 2 q / \Gamma$ , hence $\Gamma / 2 \le \kappa _ { \mathrm { s c a l e } } \le \Gamma$ . The amplitude factor satisfies $\| h _ { 1 } \| _ { \infty } / r _ { 1 } = \sqrt { 2 }$ and $\| h _ { 2 } \| _ { \infty } / r _ { 2 } = \sqrt { 2 } k / \sqrt { k ^ { 2 } + 1 / 9 } < \sqrt { 2 } ,$ so $\kappa _ { \mathrm { a m p } } = \sqrt { 2 }$ and the constructed tensor belongs to the supremum defining $\mathfrak { E } _ { q } ( \Gamma )$

We now exhibit a scale-invariant Lipschitz head sensitive to the rounding error. Let $z : = \mathcal { C } _ { 0 } ( h _ { 2 } )$ and $z ^ { \prime } : =$ $\mathcal { C } _ { 0 } ( Q _ { \Delta } ( h _ { 2 } ) )$ . In two dimensions, canonicalization multiplies the Euclidean unit direction by ${ \sqrt { 2 } } .$ so the RMS chord distance between z and $z ^ { \prime }$ is the ordinary chord distance between those unit directions. The angle between those directions is $\theta = \arctan ( 1 / ( 3 k ) )$ , and hence $d _ { \mathbb { S } } ( z , z ^ { \prime } ) = 2 \sin ( \theta / 2 )$ . For $0 \le y \le \pi / 2$ , sin $y \ge 2 y / \pi$ , and for $0 \leq u \leq 1$ , arctan u $\iota \geq u / 2$ . Since $1 / ( 3 k ) \le 1$ , these inequalities and Eq. G.12 give

$$
d _ { \mathbb { S } } ( z , z ^ { \prime } ) \geq \frac { 2 \theta } { \pi } \geq \frac { 1 } { 3 \pi k } \geq \frac { \kappa _ { \mathrm { s c a l e } } } { 3 \pi q } \geq \frac { \Gamma } { 6 \pi q } ,
$$

where the last step uses $\kappa _ { \mathrm { s c a l e } } \geq \Gamma / 2$

Finally, put $v : = ( z ^ { \prime } - z ) / \| z ^ { \prime } - z \| _ { 2 }$ and define $f : \mathbb { S } _ { 2 } \to \mathbb { R } \mathrm { ~ b y ~ } f ( u ) : = \langle v , u \rangle / \sqrt { 2 }$ . For all $u , u ^ { \prime } \in \mathbb { S } _ { 2 }$ , we have $| f ( u ) - f ( u ^ { \prime } ) | \leq \| u - u ^ { \prime } \| _ { 2 } / \sqrt { 2 } = d _ { \mathbb { S } } ( u , u ^ { \prime } )$ , so $f$ is 1-Lipschitz, and the choice of v makes $| f ( z ^ { \prime } ) - f ( z ) | =$ $\| z ^ { \prime } - z \| _ { 2 } / \sqrt { 2 } = d _ { \mathbb { S } } ( z , z ^ { \prime } )$ . Therefore the scale-invariant map $f \circ \mathcal { C } _ { 0 }$ attains a functional error at least $\Gamma / ( 6 \pi q )$ on the admissible tensor in Eq. G.10. This construction proves the lower inequality in Eq. E.25. Together with the upper inequality, the lower bound establishes the claimed worst-case order. □

## H EXPERIMENTAL DETAILS

## H.1 Language-model training and controlled comparisons

Under matched controls, we test how unit consistency afects training robustness, functional rescaling invariance, deployment quantization, and unit-system-invariant multi-loss optimization. All runs use a single NVIDIA GH200-96GB GPU. Within each controlled comparison, architectures share data order, budget, and recipe. Only the declared intervention changes, which in the main comparisons is the block structure.

Main language models. The main language models are decoder-only Transformers with $d = 7 6 8 , L = 1 2$ , 12 heads, and context length 1024. These models have approximately 124M parameters, with all variants matched to within 0.3%. We train the models from scratch on FineWeb-Edu (Penedo et al., 2024) with the Generative Pre-trained Transformer 2 (GPT-2) byte-pair encoding (BPE) tokenizer (Radford et al., 2019) for 2.75B tokens. This budget is approximately Chinchilla-optimal for the model size (Hofmann et al., 2022). A disjoint shard supplies a held-out 43M-token validation split.

Training recipe. The training recipe follows standard practice at this scale (Karpathy, 2022; Touvron et al., 2023). We use Adam with decoupled weight decay (AdamW), setting $\beta = ( 0 . 9 , 0 . 9 5 )$ and weight decay 0.1 on matrices only (Loshchilov and Hutter, 2019). The peak learning rate is $6 \times 1 0 ^ { - 4 }$ , the warmup occupies 2.7% of training and lasts 151 steps at 124M, and cosine decay ends at 10% of the peak rate. The global batch contains 491,520 tokens, the gradient clip is 1.0, and training uses bf16 autocast and weight tying. Every variant uses rotary positional encoding (RoPE) (Su et al., 2024), which strengthens the Pre-LN baseline relative to learned positions.

All eight architectures are replicated with three seeds at 124M. Unless a table explicitly reports a multi-seed aggregate, perturbation and quantization analyses evaluate one model per architecture trained with the primary seed. The remaining seeds quantify training variability. App. H.4 states the seed protocol for every scale.

Architectures compared. The comparison includes Pre-LN with LayerNorm and GELU in the GPT-2 style (Radford et al., 2019; Xiong et al., 2020). LLaMA-style blocks use RMSNorm and SwiGLU (Touvron et al., 2023). QK-Norm adds per-head RMS normalization of Q and K (Henry et al., 2020; Dehghani et al., 2023). We also include NormFormer (Shleifer et al., 2021) and LayerScale initialized at 0.1 (Touvron et al., 2021b).

Three further variants control scale through normalized or explicitly corrected representations. nGPT is the closest normalized-representation prior to GCT and uses a sphere-normalized state, per-dimension eigen-learningrate linear-interpolation updates, unit-norm weights, and QK-Norm (Loshchilov et al., 2025). UCT uses the local-correction modules of App. H.4, and GCT is the canonical design derived in Sec. 4.

Learning-rate robustness and seeds at 51M. The robustness study uses $d = 5 1 2 , L = 8$ models with parameter counts from 50.9M to 51.5M, referred to as 51M models. Small-scale learning-rate sweeps are used to study training instabilities that also arise at larger scales (Wortsman et al., 2024). Pre-LN, LLaMA-style, QK-Norm, nGPT, and GCT train for 600M tokens with learning rates in $\{ 3 , 6 , 1 2 , 2 4 , 4 8 \} \times 1 0 ^ { - 4 }$ , spanning a factor of 16. The protocol includes the strong QK-Norm baseline in every stress test. Each architecture also has three seeds at the default learning rate.

UVI-based early stopping. At the additional learning rate $9 . 6 \times 1 0 ^ { - 3 } ,$ , we apply the same preset stopping rule to Pre-LN, LLaMA-style, nGPT, and GCT. At diagnostic steps within the first 10% of the training budget, excluding initialization, the monitor terminates a run if $\mathrm { U V I _ { r e s } + U V I _ { a t t n } > 2 . 7 9 }$ . Pre-LN and LLaMA-style cross this threshold at step 100 of 1220, while nGPT and GCT complete training under the same rule. These early stops record threshold crossings, not observed numerical divergence.

Depth probes and ablations. Separate fixed-budget depth probes use $L = 2 4$ and $L = 4 8$ , which are respectively $3 \times$ and 6× the tuning depth. These probes use 400M tokens with unchanged hyperparameters for the same five architectures, and the depth table reports the primary seed. The GCT component ablations remove $P ^ { \perp }$ , proposal canonicalization, or both components under the 600M-token $L = 8$ protocol. We additionally repea the $P ^ { \perp }$ ablation under the 400M-token depth protocol to measure how the projection’s efect changes during depth transfer.

Larger language models. The same shared recipe uses learning rate $6 \times 1 0 ^ { - 4 }$ with an identical schedule and data order at larger scales. The 350M models use d = 1024, L = 24, 2B tokens, and three seeds. The 1.0B models use d = 1792, L = 24, 4B tokens, and two seeds. Pre-LN, LLaMA-style, QK-Norm, nGPT, and GCT are all represented. These models extend the residual-stream quantization and parameter-fingerprint studies beyond the 124M setting.

Scale perturbation. Using the trained 124M models, we measure validation loss under $h $ ch at the embedding output and at mid-depth for $c \in [ 1 0 ^ { - 2 } , 1 0 ^ { 2 } ]$ . We also measure per-block equivariance or invariance error in fp32 (Figure $5 ;$ Table 8).

## H.2 Quantization and deployment evaluations

PTQ. We apply PTQ to the same trained models to evaluate sensitivity to finite-precision storage and computation. The protocols include symmetric per-channel weight quantization at 8 and 4 bits and int8 activation quantization at every linear input, using both dynamic and calibration-based static scales in the standard integer-inference format (Jacob et al., 2018). The per-tensor, per-channel, and per-token granularities compared in Sec. 5.5 are the standard PTQ options (Nagel et al., 2021). Weight quantization uses round-to-nearest rather than error-compensated rounding (Frantar et al., 2023), because the comparison isolates activation- and stream-scale management rather than weight-rounding quality.

We also quantize the residual stream at every block boundary to int8 and int4. This setting arises when hidden states are cached, ofloaded, or transmitted across pipeline stages. The stream protocols use static or dynamic pertensor scales and per-token scales. We additionally evaluate scale migration through SmoothQuant with $\alpha = 0 . 5$ and per-site static calibration on the linear path (Xiao et al., 2023), together with calibrated per-channel static scales for the stored stream. The logit head remains in full precision as specified in App. H.6. A microbenchmark compares the deployment cost of per-token and per-tensor activation-scale management on the same device.

Of-the-shelf audit. To examine the quantization pattern in independently trained models, we record the depth-scale span $\kappa _ { \mathrm { d e p t h } } : = \operatorname* { m a x } _ { \ell } \bar { r } _ { \ell } /$ min<sub>ℓ</sub> $\bar { r } _ { \ell }$ , where $\bar { r } _ { \ell }$ is the mean token RMS at block input ℓ. We then apply identical int8 and int4 quantization hooks to the four publicly available pretrained OpenAI GPT-2 models from 124M to 1.5B parameters (Radford et al., 2019). The depth-scale span $\kappa _ { \mathrm { d e p t h } }$ is a descriptive depth-drift statistic and is distinct from the within-tensor $\kappa _ { \mathrm { s c a l e } }$ of Prop. 5. We evaluate these pretrained models on the same FineWeb-Edu validation bins with the same GPT-2 BPE through an independent $\mathrm { P y }$ Torch reimplementation. The implementation reproduces the published WikiText-103 (Merity et al., 2017) validation perplexities 27.7, 20.3, 17.5, and 16.0 for the 124M, 355M, 774M, and 1.5B pretrained models on non-overlapping blocks, confirming faithful weight loading.

Quantized downstream accuracy. The 1B models trained independently with a second seed are evaluated before and after residual-stream quantization on four tasks. Language Modeling Broadened to Account for Discourse Aspects (LAMBADA) uses the OpenAI variant and greedy final-word accuracy (Paperno et al., 2016). Harder Endings, Longer Contexts, and Low-shot Activities for Situations With Adversarial Generations (HellaSwag) uses byte-length-normalized four-way accuracy (Zellers et al., 2019). The remaining tasks are Physical Interaction: Question Answering (PIQA) (Bisk et al., 2020) and the easy subset of the Allen Institute for AI Reasoning Challenge (ARC-Easy) (Clark et al., 2018). The evaluation uses lm-evaluation-harness-compatible prompts and log-likelihood scoring under the training tokenizer.

## H.3 Vision, optimizer, width, and multi-loss evaluations

First vision benchmark. The benchmark on the 100-class Canadian Institute for Advanced Research (CIFAR) dataset uses ViTs at the scale of the tiny Data-eficient Image Transformer (DeiT) variant, denoted DeiT-Ti, with $d = 1 9 2 , L = 1 2$ , and patch size 4. Models train from scratch for 300 epochs with the standard heavy-augmentation recipe (Touvron et al., 2021a). The recipe uses AdamW with learning rate $1 0 ^ { - 3 }$ , batch size 512, weight decay 0.05, a 10-epoch warmup, cosine decay, RandAugment(2, 9), mixup 0.8, CutMix 1.0, label smoothing 0.1, and random erasing 0.25. Variants are Pre-LN, QK-Norm, nGPT, UCT, and GCT. Evaluation covers clean top-1 accuracy, the Gaussian-noise, contrast, and brightness corruptions of the standard common-corruption robustness protocol (Hendrycks and Dietterich, 2019), and multiplicative rescaling of the normalized input as an input-unit shift.

Vision benchmark on Tiny-ImageNet. A second benchmark trains small ViT (ViT-S) models from scratch on Tiny-ImageNet-200 for 200 epochs. These models use $d = 3 8 4 , L = 1 2$ , patch size 8, images of size $6 4 \times 6 4$ 100k training images, and 10k validation images. All eight architectures use the same optimizer and augmentation recipe with a 15-epoch warmup.

Optimizer and width evaluations. Two further axes test generality beyond one optimizer and one width. The optimizer-pairing axis uses a 51M experiment that replaces AdamW on hidden matrices with the shape-aware orthogonalized Muon optimizer (Jordan et al., 2024). We test $\eta _ { \mathrm { M u o n } } \ \in \ \{ 0 . 0 1 , 0 . 0 2 \}$ for GCT, Pre-LN, and QK-Norm, while auxiliary parameters retain the AdamW recipe. This comparison tests compatibility with a duality-map optimizer.

The width axis repeats the learning-rate robustness protocol at $d \in \{ 2 5 6 , 1 0 2 4 \}$ with L = 8 and fixed $d _ { h } = 6 4$ The same axis also composes a learning-rate-only $\mu \mathrm { P }$ rule with GCT and Pre-LN at the respective d = 512-tuned learning rates (Yang et al., 2021). Hidden-matrix learning rates scale by $5 1 2 / d ,$ while initialization and multipliers remain unchanged.

PINN multi-loss. The PINN study covers Burgers with $\nu = 0 . 0 1 / \pi$ and a Cole–Hopf reference, together with Helmholtz using $a _ { 1 } = 1$ and $a _ { 2 } = 4$ (Raissi et al., 2019; Wang et al., 2021). Both problems use a $4 \times 6 4$ tanh MLP, Adam with learning rate $1 0 ^ { - 3 }$ , 20K steps with a cosine tail, and 10K collocation points. The field is re-expressed in rescaled units as $\hat { u } : = \kappa u$ for $\kappa \in \{ 1 0 ^ { - 2 } , 1 , 1 0 ^ { 2 } \}$ . The PDE is rewritten accordingly, leaving the physical problem unchanged.

Methods include equal weights and the common manual choice $\lambda _ { \mathrm { s u p } } = 1 0 0$ , which is tuned at $\kappa = 1$ and then transferred. We also evaluate uncertainty weighting (Kendall et al., 2018), gradient-statistics annealing, known as learning-rate annealing (LRA) (Wang et al., 2021), neural tangent kernel (NTK) trace balancing (Wang et al., 2022), and the NDA-typed objective. NDA types the whole graph through the output head ${ \hat { u } } : = s _ { \mathrm { o u t } } { \hat { \bar { u } } }$ from Sec. 4 and the objective in Eq. 4. All characteristic scales are derived from observables such as data RMS, forcing RMS and forcing length scale, and operator units. The method requires no tuning, adaptivity, or access to the solution. App. H.6 gives the characteristic scales and baseline settings. We use five seeds per cell and report relative $L _ { 2 }$ error of the recovered physical field $\hat { u } / \kappa$

## H.4 Controls and model implementation

Seed protocol. Every language-model family draws training seeds from one shared list, and each architecture inside a controlled comparison reuses that same list to isolate the efect of the declared intervention. At 51M and at 124M, each architecture trains three seeds at the default learning rate, the 350M family trains three seeds, and the 1.0B family trains two seeds. Learning-rate sweep points, component ablations, and depth probes are single-seed. One member of the shared list serves as the primary seed at every scale.

Analyses that report one result per architecture evaluate the model trained with the primary seed unless the table states otherwise. These analyses include the scale-perturbation study of Table 8, the PTQ results of Tables 2 and 10, the parameter-gauge fingerprint of Table 9, the fixed-budget depth probe of Table 1, and the width and $\mu \mathrm { P }$ grid of Table 14. The remaining seeds supply the multi-seed aggregates and quantify training variability.

We use models trained independently with a second seed for the 350M column of Table 13, the repeat of the 1B stream-quantization and parameter-fingerprint results in Sec. 5.5, the downstream-accuracy evaluation of Table 11, and the serving-latency benchmark. The PINN study uses five seeds per cell and reports the mean across those five seeds, and the post-layer-normalization (Post-LN) reference at $L = 8$ in App. I.1 uses five seeds.

Weight-decay controls. Every architecture in a controlled comparison uses weight decay 0.1 on matrices. The efect of weight decay can nevertheless difer across architectures, because nGPT normalizes weight rows and GCT canonicalizes branch proposals. On scale-invariant parameter blocks, weight decay afects the efective learning rate through the parameter norm (van Laarhoven, 2017; Li and Arora, 2020).

We assess sensitivity to this shared setting with weight-decay controls. Re-running nGPT without weight decay, as recommended by Loshchilov et al. (2025), changes nGPT’s 51M loss by +0.012 nats, and setting weight decay to zero changes the other 51M baselines by at most 0.004 nats, with −0.004 for Pre-LN and +0.001 for LLaMA-style, while GCT changes by +0.003, leaving the clean-loss ordering unchanged. Removing both weight decay and warmup, as in nGPT’s original recipe, while retaining our other settings increases $\mathrm { n G P T ^ { \circ } s }$ best loss at the two tested high learning rates by 0.093 nats, from 3.500 to 3.593. These controls do not indicate a disadvantage from the shared recipe in this setting.

UCT corrective modules. UCT makes residual additions unit-consistent and gives softmax and gates dimensionless inputs through three added modules: an RMS-matching residual adapter $h + \alpha _ { \ell } [ \mathrm { R M S } ( h ) / \mathrm { R M S } ( a ) ] a$ the empirical attention temperature from Eq. 3, and normalized gates. We use UCT to test whether these three local corrections improve a conventional Transformer’s performance and robustness to changes in scale.

UCT details. Empirical attention scales $s _ { Q }$ and $s _ { K }$ are per-head stop-gradient RMS batch statistics with an exponential moving average (EMA) of momentum 0.99 used at evaluation. The gate scale $s _ { a }$ is a per-layer scalar EMA. The residual adapter uses per-token RMS matching with a learnable scalar $\alpha _ { \ell }$ for each branch. The pre-normalizers and configured converter denominators use fixed numerical floors of $\epsilon = 1 0 ^ { - 6 }$ . Both these floors and the evaluation-time EMA scales stay numerically fixed under an injected rescaling. When the hidden state is scaled down, the fixed floors contribute more to the normalization denominators, and the stored EMA scales no longer match the activations. Dividing rescaled activations by these unchanged quantities does not undo the rescaling, which explains the loss of equivariance under the most severe down-scaling in Figure 5.

GCT details. Canonicalizer floors are fixed numerically at $1 0 ^ { - 1 2 }$ inside the square root. The floors have the formal squared unit $u ( \epsilon _ { x } ) = u ( x ) ^ { 2 }$ . The numerical canonicalizer floors stay fixed under an injected rescaling, producing the finite defect in Eq. F.1. Exact invariance refers to the zero-floor or co-transforming-floor statement. The implementation uses the projection denominator $\| \bar { h } \| ^ { 2 } + \epsilon _ { \perp }$ with $\epsilon _ { \perp } = 1 0 ^ { - 6 }$ , whereas the exact orthogonality and rotation results use $\epsilon _ { \perp } = 0$ as stated in App. F.1. Per-head gains and the logit gain initialize at 1.0. Attention and MLP sub-blocks are sequential, and each sub-block has a separate $\alpha _ { \ell } .$ Embeddings are canonicalized at entry, followed by a final canonicalization before the tied head.

Heads, scales, and parameters. The classification head produces dimensionless logits from $\mathcal { C } ( \bar { h } _ { L } )$ with a learnable dimensionless logit gain. For regression and PINN predictions that require units, the head first produces a dimensionless value ${ \hat { \bar { y } } } .$ Multiplying by an explicit output scale gives $\hat { y } : = s _ { y } \hat { \bar { y } }$ , which restores the magnitude and units. We parameterize each learned scale as $s : = s _ { 0 } e ^ { \ell _ { s } }$ , where the fixed reference $s _ { 0 }$ has the required unit and the trainable log multiplier $\ell _ { s }$ is dimensionless. Within each block, linear maps mix dimensionless coordinates using dimensionless weights. Any required unit conversion is performed by an explicit scale factor, such as $s _ { y }$ at the output. For tasks with dimensionless outputs, such as language modeling, the computation does not depend on the scale state, and we implement the residual stream using only canonical coordinates.

## H.5 Reference PyTorch implementation of GCT

Listing 1 is a compact extraction of the implementation used in the experiments. The reference block fixes the reported configuration, which uses scalar branch steps and enables both the tangential projection and proposal canonicalization. Only diagnostic collection, ablation switches, and standard initialization boilerplate are removed. The listing retains every scale-sensitive operation: fp32 canonicalization with a return to the input dtype, per-head canonical $Q$ and K, the dimensionless normalized-SiLU gate, the regularized tangential projection, and the two sequential canonical residual additions. RoPE is included and is norm preserving, while scaled\_dot\_product\_attention supplies the standard $1 / \sqrt { d _ { h } }$ factor.

The two floors in Listing 1 are the reported numerical values. Setting eps\_canon and eps\_perp to zero recovers the exact canonicalizer and projector on the respective nondegenerate domains. The displayed finite-floor code has precisely the defects derived in App. F.1. At network level, the embedding output is canonicalized before the first block, the last block output is canonicalized again before the tied language-model head, and a learnable dimensionless scalar multiplies the logits. These two wrapper operations are omitted from the listing because the wrapper operations repeat canon and contain no additional scale-sensitive mechanism.

## H.6 Task-specific and audit protocols

Parameter-gauge audit predictions. GCT nondimensionalizes every probed output. The architecture applies C to q, k, and the gate preactivation, and applies $\mathcal { C } \circ P ^ { \perp }$ to both branch proposals. All seven roles are null. Normalizing every weight row in nGPT’s forward pass makes all roles null. QK-Norm applies per-head RMSNorm only to q and k. In NormFormer, LayerNorm after the attention out-projection types v and the out-projection. NormFormer’s mid-MLP LayerNorm acts after the GELU, leaving the GELU’s input scale uncorrected. Pre-LN, LLaMA-style, and LayerScale have no nulls because none places a nondimensionalizer between a probed matrix and the residual stream. In evaluation mode, UCT’s live-RMS residual adapter types both branch outputs, covering v, the attention output, the up-projection, and the MLP output. The q, k, and gate interfaces retain radial sensitivity because the corrections at these interfaces use frozen EMA statistics.

Parameter-gauge audit protocol. We use fp32 evaluation mode and the first 8 fixed validation batches. One backward pass yields every $\langle W , \nabla _ { W } { \mathcal { L } } \rangle$ , with dot products accumulated in 64-bit floating-point precision and reported as the maximum over layers for each role. A central finite diference at the middle layer uses $c = e ^ { \pm 0 . 1 }$ and cross-checks two roles per architecture. This secant over ±0.1 log-units includes curvature. For nonnull cells, the secant checks the sign and order of magnitude, while for null cells, the pairing and secant both vanish. The identical audit runs on the five trained 1B models, yielding 34 cells because UCT is absent at 1B and Pre-LN’s GELU MLP has no up-projection.

Blocks that realize the same activation rescaling must carry equal pairings. The q and k rows both rescale the logits, the v rows and out-projection both rescale the attention write, and the up- and down-projections both rescale the MLP write. The measured columns agree, providing an internal consistency check on the pairing measurement. For stop-gradient scale estimators, as in UCT with batch statistics active, the finite diference and the autograd pairing measure function invariance and the optimizer-visible gradient, respectively. The two measurements are reported separately in App. I.2.

PINN details. Both problems use the standard two-term split between the PDE residual and combined supervision to give the adaptive balancers a well-conditioned pair. Burgers uses $N _ { f } = 1 0 ^ { 4 }$ uniformly sampled interior points and combines 200 initial-condition points with 200 boundary-condition points into one supervision term. The Burgers reference is computed by Cole–Hopf with 100-node Gauss–Hermite quadrature on a $2 5 6 \times 1 0 1$ grid. Helmholtz uses $N _ { f } = 1 0 ^ { 4 }$ , 400 boundary points, and the manufactured solution sin(πx) sin(4πy).

The adaptive baselines follow the published method specifications. LRA follows Wang et al. (2021). Every 10 steps, $\lambda _ { \mathrm { s u p } }$ tracks $\| \nabla _ { \theta } \mathcal { L } _ { \mathrm { p d e } } \| / \| \nabla _ { \theta } \mathcal { L } _ { \mathrm { s u p } } \|$ as an EMA with weight 0.1, clamped to $[ 1 0 ^ { - 2 } , 1 0 ^ { \bar { 2 } } ]$ . Uncertainty weighting optimizes $\sum _ { i } e ^ { - s _ { i } } \mathcal { L } _ { i } / 2 \overset { . } { + } s _ { i } / 2$ . NTK balancing follows Wang et al. (2022). Every 100 steps, $\lambda _ { k } : = \mathrm { t r } ( K ) / \mathrm { t r } ( K _ { k } )$ with tr $( K _ { k } )$ estimated as $N _ { k } \times$ the mean squared parameter-gradient norm of 16 randomly sampled per-point residuals of term $k .$

```python
import torch
2 import torch.nn as nn
3 import torch.nn.functional as F
4
5 def canon(x, eps=1e-12):
6 xf = x.float()
7 inv_rms = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
8 return (xf * inv_rms).type_as(x)
9
10 def apply_rope(x, cos, sin):
11 x1, x2 = x[..., 0::2], x[..., 1::2]
12 c, s = cos[None, None], sin[None, None]
13 y = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
14 return y.flatten(-2)
15
16 class GCTBlock(nn.Module):
17 def __init__(self, dim, n_heads, mlp_width, depth,
18 eps_canon=1e-12, eps_perp=1e-6):
19 super().__init__()
20 assert dim % n_heads == 0
21 self.dim, self.n_heads = dim, n_heads
22 self.head_dim = dim // n_heads
23 self.eps_canon, self.eps_perp = eps_canon, eps_perp
24 self.qkv = nn.Linear(dim, 3 * dim, bias=False)
25 self.attn_out = nn.Linear(dim, dim, bias=False)
<sup>26</sup> <sub>27</sub> self.gain = nn.Parameter(torch.ones(n_heads))
self.w_gate = nn.Linear(dim, mlp_width, bias=False)
28 self.w_up = nn.Linear(dim, mlp_width, bias=False)
29 self.mlp_out = nn.Linear(mlp_width, dim, bias=False)
30 alpha0 = 1.0 / (2.0 * depth) ** 0.5
31 self.alpha_att = nn.Parameter(torch.tensor(alpha0))
32 self.alpha_mlp = nn.Parameter(torch.tensor(alpha0))
33
34 def _attention(self, x, cos, sin, causal):
35 B, T, D = x.shape
36 H, dh = self.n_heads, self.head_dim
37 qkv = self.qkv(x).view(B, T, 3, H, dh)
38 q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
39 q = canon(q, self.eps_canon) * self.gain.view(1, H, 1, 1)
40 k = canon(k, self.eps_canon)
41 q = apply_rope(q, cos, sin).type_as(v)
42 k = apply_rope(k, cos, sin).type_as(v)
43 y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
44 y = y.transpose(1, 2).contiguous().view(B, T, D)
45 return self.attn_out(y)
46
47 def _mlp(self, x):
48 a = canon(self.w_gate(x), self.eps_canon)
49 gate = a * torch.sigmoid(a)
50 return self.mlp_out(gate * self.w_up(x))
51
52 def _update(self, h, proposal, alpha):
53 hf, pf = h.float(), proposal.float()
54 coeff = (pf * hf).sum(-1, keepdim=True)
55 coeff = coeff / (hf.square().sum(-1, keepdim=True)
56 + self.eps_perp)
57 tangent = (pf - coeff * hf).type_as(proposal)
58 tangent = canon(tangent, self.eps_canon)
59 return canon(h + alpha * tangent, self.eps_canon)
60
61 def forward(self, h, cos, sin, causal=True):
62 h = canon(h, self.eps_canon)
63 h = self._update(
64 h, self._attention(h, cos, sin, causal), self.alpha_att)
65 h = self._update(h, self._mlp(h), self.alpha_mlp)
66 return h
```  
Listing 1: Finite-floor PyTorch reference implementation of the GCT block used in the experiments.

The NDA-typed method types both the head and the objective. Characteristic scales are computed from observables only. For Burgers, s<sub>u</sub> := RMS(initial-condition data), which gives output scale $s _ { \mathrm { o u t } } : = s _ { u } .$ , combinedsupervision scale $S _ { \mathrm { s u p } } : = s _ { u } ^ { 2 } ,$ , and $S _ { \mathrm { p d e } } : = ( \dot { s } _ { u } / t ^ { * } ) ^ { 2 }$ with domain time scale t<sup>∗</sup> = 1.

Because the Helmholtz boundary data are zero, we estimate the field scale from the forcing magnitude and forcing length scale. Define $s _ { q } : = \mathrm { R M S } ( q )$ and $\kappa _ { q } ^ { 2 } : = \mathbb { E } _ { \boldsymbol { x } } \| \nabla q ( \boldsymbol { x } ) \| _ { 2 } ^ { 2 } / \mathbb { E } _ { \boldsymbol { x } } | q ( \boldsymbol { x } ) | ^ { 2 }$ , with the expectations estimated over the forcing sample. Our single-mode manufactured forcing is nonresonant and satisfies $\kappa _ { q } ^ { 2 } \neq k ^ { 2 }$ . The operator balance is $| \Delta u + k ^ { 2 } u | \sim | \kappa _ { q } ^ { 2 } - k ^ { 2 } | u ^ { * } \sim s _ { q }$ , giving $u ^ { * } = s _ { q } / | \kappa _ { q } ^ { 2 } - k ^ { 2 } |$ . For a representative forcing sample, this observable estimate is 0.515. Across the five experimental seeds, the estimate ranges from 0.488 to 0.523, around the true continuous-field RMS of 0.5. We then set $s _ { \mathrm { o u t } } : = u ^ { * } , S _ { \mathrm { s u p } } : = u ^ { * 2 }$ , and $S _ { \mathrm { p d e } } : = s _ { q } ^ { 2 } .$ The scales come directly from data and operator statistics, without using the reference solution or tuning the loss weights.

Quantization details. All schemes use simulated quantization in the standard PTQ protocol. Evaluation uses the first 32 fixed batches of the validation stream, which are the same deterministic batches as in the perturbation study. The shared base points agree to numerical precision, as noted in the accompanying discussion of Table 8. Weights use symmetric per-output-channel absmax scaling at 8 or 4 bits, applied to every linear layer except the tied logit head. For nGPT, whose forward pass normalizes weight rows, quantization applies to the efective normalized weights that a deployment would store.

W8A8 activations use symmetric per-tensor int8 at every linear input. The dynamic variant recomputes the absmax per call, while the static variant fixes per-call-site scales from absmax calibration over 8 held-out training batches. Residual-stream quantization maps h to int8 at every block input using dynamic per-tensor absmax, calibrated per-layer static absmax, or per-token absmax scales. We intercept calls at the torch-function level to treat functional and module linear calls identically across architectures.

Migration variants. SmoothQuant is applied at each linear site with $\alpha = 0 . 5$ . Per-channel activation absmax values $a _ { j }$ are calibrated over the same 8 batches, and migration uses $s _ { j } : = a _ { j } ^ { \alpha } / \operatorname* { m a x } _ { i } | W _ { i j } | ^ { 1 - \alpha }$ . Weights W diag(s) are requantized per output channel. Activations $x \mathrm { d i a g } ( s ) ^ { - 1 }$ use per-tensor static quantization with the scale implied by the calibrated $a _ { j } / s _ { j }$ . For the stored stream, we calibrate per-channel static absmax scales at each block site, which is the storage analogue of migration. The logit head stays in full precision throughout.

Quantized downstream-accuracy details. Table 11 uses standard lm-evaluation-harness formats. LAMBADA-OpenAI accuracy requires greedy decoding of every token in the final word. HellaSwag validation uses the activity label and context, preprocessed endings, and byte-length-normalized scoring. PIQA validation uses the prompt template Question:...Answer:. ARC-Easy test choices are scored as completions. We use right padding under causal masking, truncate contexts from the left to 1024 tokens, and score with fp32 log-softmax over bf16 logits.

Breadth-evaluation details. Muon uses five Newton–Schulz iterations and orthogonalized Nesterov momentum with $\mu = 0 . 9 5$ on hidden two-dimensional matrices. The optimizer applies shape-aware step scaling $\sqrt { \operatorname* { m a x } ( 1 , d _ { \mathrm { o u t } } / d _ { \mathrm { i n } } ) }$ and decoupled weight decay 0.1. Embeddings, gains, and scalars use AdamW with the recipe settings. Both groups follow the shared cosine schedule. We report the better result over $\eta _ { \mathrm { M u o n } } \in \{ 0 . 0 1 , 0 . 0 2 \}$

## H.7 Reproducibility and numerical verification

Reproducibility. Reported values are computed from the corresponding run logs, and protocol descriptions follow the executed configurations. Experimental protocols are described in this appendix, and Listing 1 provides a compact reference implementation of the GCT block. The parameter-gauge audit of Prop. 6 uses one backward pass to measure radial sensitivity by weight-matrix role.

Numerical consistency checks. Automated fp32 tests check the numerical idempotency of C. The maximum coordinate error is below $1 0 ^ { - 5 }$ , and the maximum deviation of the mean square from 1 is below $1 0 ^ { - 4 }$ . The tests verify GCT block invariance under h → ch with relative error below $1 0 ^ { - 4 }$ . The same tests also verify full-network invariance at the embedding output for $c = 1 0 ^ { 2 }$ and in the middle of the stream for $c = 1 0 ^ { - 2 }$ , with maximum logit error below $1 0 ^ { - 3 }$ . UCT block equivariance under the tested up-scaling $c = 1 0 ^ { 2 }$ has relative error below $1 0 ^ { - 3 }$ Attention logit variance at GCT initialization remains between $0 . 5$ and 2.0. The NDA-typed PINN objective is numerically invariant at initialization, with relative spread below $1 0 ^ { - 3 }$ for $\kappa \in \{ 1 0 ^ { - 2 } , 1 , 1 0 ^ { 2 } \}$ . The small observed variation across κ is consistent with nondeterministic GPU reductions in second-order autograd rather than systematic unit dependence.

Table 4: Clean validation loss for the matched 124M models trained on 2.75B tokens. Each row reports the mean and standard deviation over three seeds under the shared recipe. Values are rounded to three decimal places. A displayed standard deviation of 0.000 denotes variation below that precision.
<table><tr><td>architecture</td><td>validation loss</td></tr><tr><td>Pre-LN</td><td> $3 . 1 7 2 \pm 0 . 0 0 2$ </td></tr><tr><td>LLaMA-style</td><td> $3 . 1 7 5 \pm 0 . 0 0 3$ </td></tr><tr><td>QK-Norm</td><td> $\mathbf { 3 . 1 4 7 \pm 0 . 0 0 2 }$ </td></tr><tr><td>NormFormer</td><td> $3 . 1 9 8 \pm 0 . 0 0 8$ </td></tr><tr><td>LayerScale</td><td> $3 . 1 6 7 \pm 0 . 0 0 3$ </td></tr><tr><td>nGPT</td><td> $3 . 1 7 0 \pm 0 . 0 0 3$ </td></tr><tr><td>Corrective UCT (ours)</td><td> $3 . 2 5 6 \pm 0 . 0 0 4$ </td></tr><tr><td>GCT (ours)</td><td> $3 . 1 7 0 \pm 0 . 0 0 0$ </td></tr></table>

Table 5: Final validation loss across a 16× learning-rate range for 51M models trained on 600M tokens. The additional octave at $9 . 6 \times 1 0 ^ { - 3 }$ is reported wherever training runs to completion. abort marks a run terminated early because the run’s UVI score exceeded the preset threshold, as specified in App. H.
<table><tr><td rowspan="2">architecture</td><td colspan="6">peak learning rate</td></tr><tr><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $6 \times 1 0 ^ { - 4 }$ </td><td> $1 . 2 \times 1 0 ^ { - 3 }$ </td><td> $2 . 4 \times 1 0 ^ { - 3 }$ </td><td> $4 . 8 \times 1 0 ^ { - 3 }$ </td><td> $9 . 6 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Pre-LN</td><td>4.232</td><td>3.943</td><td>3.750</td><td>3.876</td><td>3.881</td><td>ABORT</td></tr><tr><td>LLaMA-style</td><td>4.248</td><td>3.892</td><td>3.714</td><td>3.802</td><td>3.883</td><td>ABORT</td></tr><tr><td>QK-Norm</td><td>4.153</td><td>3.840</td><td>3.700</td><td>3.740</td><td>3.879</td><td>3.897</td></tr><tr><td>nGPT</td><td>4.383</td><td>3.893</td><td>3.649</td><td>3.563</td><td>3.507</td><td>3.500</td></tr><tr><td>GCT (ours)</td><td>4.373</td><td>3.885</td><td>3.658</td><td>3.548</td><td>3.517</td><td>3.513</td></tr></table>

Table 6: Seed variability and GCT component ablations. Both tables follow the Table 5 protocol at learning rate $6 \times 1 0 ^ { - 4 }$ . The seed table reports mean ± standard deviation over three seeds, while the ablations use the primary seed.
<table><tr><td colspan="2">Seed variability at 51M architecture val loss</td></tr><tr><td>Pre-LN</td><td> $3 . 9 4 0 5 \pm 0 . 0 0 3 9$ </td></tr><tr><td>LLaMA-style</td><td> $3 . 9 0 2 4 \pm 0 . 0 1 6 7$ </td></tr><tr><td>QK-Norm</td><td> $3 . 8 4 4 5 \pm 0 . 0 0 3 7$ </td></tr><tr><td>nGPT GCT (ours)</td><td> $3 . 8 9 3 4 \pm 0 . 0 0 0 4$   $3 . 8 9 4 4 \pm 0 . 0 0 9 6$ </td></tr></table>

<table><tr><td colspan="2">GCT component ablations variant val loss</td></tr><tr><td>GCT (full)</td><td>3.885</td></tr><tr><td>without  $P ^ { \perp }$ </td><td>3.882</td></tr><tr><td>without proposal canon.</td><td>4.486</td></tr><tr><td>without either component</td><td>4.083</td></tr></table>

## I FULL RESULTS

This section collects the detailed training, intervention, deployment, breadth, and parameter-gauge results summarized in the main text. Table 9 reports the per-role parameter-gauge fingerprint of Prop. 6, Table 14 reports the width-transfer and $\mu \mathrm { P }$ composition grid, and Figure 8 reports the CIFAR-100 robustness grid.

Results guide. Figure 4 shows how UVI distinguishes attention-scale control from residual-stream scale growth. The useful high-rate behavior across widths is reported in Table 14, and App. I.4 also evaluates compatibility with Muon. Tables 15 and 16 report the two vision benchmarks. Additional quantization evidence covers downstream accuracy, publicly available pretrained GPT-2 models, and the 350M models (Tables 11, 12, and 13). Sec. 5.5 reports the independent-seed PTQ repeat and downstream accuracy. App. I.3 collects kernel verification, scale-management costs, serving measurements, migration controls, and the KV-cache control. App. I.2 details the UCT finite-floor failure and the full rescaling results.

![](images/11dc7c993a9685257dc5553c0087f5ac550bd022126c826bba3acf57af759302.jpg)

![](images/3e5da48ea969dab765cfe358da3eff15f1b4e1ed3d9fea1c6bcfdc9b114c33a1.jpg)

![](images/cac2e009905a0bfc658f9590a21c8f96b0d1117800d4d41bb9460e9fd904d52b.jpg)  
Post-LN Pre-LN nGPT nGPT + P <sup>⟂</sup> GCT w/o P <sup>⟂</sup> GCT (c) tangential ®\`q 1 ¡ c 2\` (c) radial ®\`jc\`j  
Figure 3: Measured residual steps and alignments $( \alpha _ { \ell } , c _ { \ell } )$ during depth transfer to $L = 4 8 ,$ six times the tuning depth. Results use primary-seed fixed-budget probes. The shared legend below the panels names every architecture. (a) Measured step size against alignment. Grey contours give the per-sub-block stream-scale gain $s _ { \ell + 1 } / s _ { \ell } = ( 1 + \alpha _ { \ell } ^ { 2 } + 2 \alpha _ { \ell } c _ { \ell } ) ^ { 1 / 2 }$ of Eq. 6, the dashed ×1 contour being the scale-preserving locus, and the dotted line marks $\alpha ^ { \star } = 1 / \sqrt { 2 L }$ . (b) Removing $P ^ { \perp }$ from GCT increases alignment, whereas adding the projection to nGPT reduces alignment and changes loss from 4.243 to 3.968. (c) Tangential–radial step decomposition, with the final validation loss of each run in the right-hand column. Points are medians over the final three measurements. Hyperparameters transfer unchanged from $L = 8$

## I.1 Learning-rate tables, seeds, ablations, and transfer

Seed variability. The three-seed results in Table 6 show similar mean losses for GCT and nGPT relative to the observed seed variability of both models. The observed seed variability leaves the ordering of the two architectures unresolved.

Residual schedules in the evaluated models. Sec. 2.3 identifies the stability-relevant quantity as the positive measured RMS ratio $\alpha _ { \ell } .$ In the zero-floor formulation of GCT, proposal canonicalization makes that pre-retraction ratio exactly the magnitude $\left| \alpha _ { \ell } \right|$ of the learned scalar coeficient, initialized at $1 / \sqrt { 2 L }$ . With the fixed $1 0 ^ { - 1 2 }$ implementation floor, that identity is exact only up to the finite-floor defect documented in Eq. F.1.

Residual-scaling methods control the measured ratio $\alpha _ { \ell } ,$ often indirectly through skip or branch coeficients. These coeficients should not be identified with the RMS ratio unless branch and stream scales are matched. The original Post-LN addition uses raw coeficients of one. ReZero and SkipInit initialize a branch gate at zero (Bachlechner et al., 2021; De and Smith, 2020). DeepNet uses the skip coeficient $( 2 N ) ^ { 1 / 4 }$ together with a matching initialization down-scale calibrated for the very-deep regime (Wang et al., 2024). We next evaluate the coeficient-one Post-LN reference and report the corresponding depth sensitivity.

In Post-LN, the conventional raw coeficient of one need not produce a measured ratio of $\alpha _ { \ell } = 1$ . Under the shared recipe, the five $L = 8$ anchor runs span losses from 4.14 to 7.60, with mean $5 . 4 7 \pm 1 . 6 4$ . The $L = 2 4$ and $L = 4 8$ runs reach 7.62 and 7.61. GCT reaches 3.973 and 3.903 under the same fixed-budget depth protocol (Table 1). These values illustrate why the coeficient-one anchor requires a diferent depth schedule or training recipe.

Our experiments use the coeficient-one Post-LN anchor and do not evaluate the full DeepNet recipe. The theoretical comparison tracks the measured residual-to-stream RMS ratio, which depends on both the raw coeficients and the branch and stream scales.

UVI fingerprint. Figure 4 illustrates how the diagnostics localize scale mismatches in matched stress runs. QK Norm controls attention variance and preserves attention entropy, while QK-Norm’s residual-stream RMS grows by orders of magnitude. GCT controls both readouts. Residual-stream RMS is an auxiliary drift measurement, distinct from $\mathrm { U V I _ { r e s } }$ . Together with the attention diagnostic, the RMS measurement distinguishes local attention control from control of the whole stream.

![](images/c44d952d0c47adf8031352b79b38a5780c042da1fd1331b883e3114a51f28313.jpg)

(a) attention logit variance v  
![](images/87e08138a46cce7a9a9638ed5a0625d4af5bd10b5d8239fa5136b8d3457bd8ca.jpg)  
(b) residual-stream RMS  
RMS = 1:6 10<sup>2</sup>–3:3 10<sup>2</sup>  
RMS = 1.0

Figure 4: UVI fingerprint for matched 51M stress runs at seed 1337 and 600M tokens, without warmup or clipping, at peak learning rate $9 . 6 \times 1 0 ^ { - 3 }$ . (a) QK-Norm repairs attention variance and entropy, while (b) QK-Norm’s residual RMS still reaches values between $1 . 6 \times 1 0 ^ { \bar { 2 } }$ and $3 . 3 \times 1 0 ^ { 2 }$ . GCT keeps both readouts controlled. The annotations report final-step statistics: the layer median of v, the mean attention entropy as a fraction of the maximum attention entropy, and the across-layer range of the stream RMS. This figure is a single-seed localization case, while Figure 2 and Table 5 provide the sweep evidence.  
Table 7: Sweep-to-scale transfer. Each architecture selects a learning rate from the original five-point 16× grid in Table 5 and applies the selected learning rate unchanged at 124M. The $9 . 6 \times 1 0 ^ { - 3 }$ stress point is excluded. The selected-rate GCT entry reports mean ± standard deviation over two seeds, while other loss entries use the primary seed. ∆ is computed from unrounded values relative to the primary-seed result at $6 \times 1 0 ^ { - 4 }$
<table><tr><td colspan="2"></td><td colspan="2">124M val loss</td><td rowspan="2">∆</td></tr><tr><td>architecture</td><td>51M-selected rate</td><td> $\mathrm { a t ~ 6 \times 1 0 ^ { - 4 } }$ </td><td>at selected rate</td></tr><tr><td>Pre-LN</td><td rowspan="3"> $1 . 2 \times 1 0 ^ { - 3 }$ </td><td>3.172</td><td>3.119</td><td>-0.052</td></tr><tr><td>LLaMA-style</td><td>3.178</td><td>3.105</td><td>-0.072</td></tr><tr><td>QK-Norm</td><td>3.149</td><td>3.091</td><td>-0.058</td></tr><tr><td>nGPT</td><td rowspan="2"> $4 . 8 \times 1 0 ^ { - 3 }$ </td><td>3.171</td><td>3.043</td><td>-0.127</td></tr><tr><td>GCT (ours)</td><td>3.170</td><td> $3 . 0 4 5 \pm 0 . 0 0 2$ </td><td>-0.125</td></tr></table>

Interpreting learning-rate transfer. The sweep is consistent with the scale-management interpretation in Sec. 5.2. Canonicalization controls residual-stream magnitude, making the forward computation less sensitive to one source of variation across depth and training time. Both GCT and nGPT retain low validation loss across a wide range of high learning rates (Table 5). The transfer experiments test whether a rate selected from this range remains efective at a larger model size.

Sweep-to-scale transfer. We select a learning rate for each architecture using the original five-point 51M sweep and apply the selected learning rate unchanged at 124M (Table 7). Under this protocol, the transferred learning rate brings GCT to 3.045, within 0.002 nats of nGPT at 3.043.

![](images/7501acf5533b2b7f003044c809b0449c7aee150a45b4ee44bd666f4407234910.jpg)  
Figure 5: Validation loss under internal rescaling h → ch at the embedding output and mid-depth, with fp32 block error. The intervention changes the hidden state’s magnitude while preserving the state’s direction. GCT and $\mathrm { n G P T }$ preserve validation loss, with block invariance errors near the numerical floor, showing insensitivity to the imposed magnitude change. UCT loses invariance at $c = 1 0 ^ { - 2 }$ because the hidden state shrinks while the small constants in UCT’s normalization denominators and the stored evaluation-time scale estimates remain fixed. The fixed constants account for a larger share of the normalization denominators, and the stored scales no longer match the activations. These mismatches prevent exact cancellation of the rescaling. Baselines with unnormalized residual streams also degrade.

Table 8: Validation loss of the trained 124M models at the $c = 0 . 1$ and $c = 1 0$ points of Figure 5, together with block equivariance error at $c = 1 0 .$
<table><tr><td colspan="2"></td><td colspan="2">rescaled at embedding</td><td colspan="2">rescaled at mid-depth</td><td rowspan="2">eq error</td></tr><tr><td>architecture</td><td>base</td><td> $c = 0 . 1$ </td><td> $c = 1 0$ </td><td> $c = 0 . 1$ </td><td> $c = 1 0$ </td></tr><tr><td>Pre-LN</td><td>3.118</td><td>7.901</td><td>3.964</td><td>10.581</td><td>5.239</td><td> $3 . 7 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>LLaMA-style</td><td>3.126</td><td>7.317</td><td>4.884</td><td>8.118</td><td>5.592</td><td> $3 . 4 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>QK-Norm</td><td>3.098</td><td>7.474</td><td>4.299</td><td>8.909</td><td>5.333</td><td> $4 . 0 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>NormFormer</td><td>3.139</td><td>3.141</td><td>3.277</td><td>10.153</td><td>5.370</td><td> $4 . 7 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>LayerScale</td><td>3.120</td><td>8.723</td><td>22.860</td><td>11.438</td><td>7.875</td><td> $3 . 3 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>nGPT</td><td>3.119</td><td>3.119</td><td>3.119</td><td>3.119</td><td>3.119</td><td> $3 . 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Corrective UCT (ours)</td><td>3.204</td><td>3.205</td><td>3.204</td><td>3.204</td><td>3.204</td><td> $1 . 1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>GCT (ours)</td><td>3.118</td><td>3.118</td><td>3.118</td><td>3.118</td><td>3.118</td><td> $3 . 7 \times 1 0 ^ { - 5 }$ </td></tr></table>

## I.2 Additional rescaling and weight-space audits

The corrective UCT is nearly flat at the $c \in \{ 0 . 1 , 1 0 \}$ points of Table 8. At the stronger down-scaling $c = 1 0 ^ { - 2 }$ in Figure 5, UCT’s loss changes by +1.406 nats at the embedding intervention and +0.211 nats at mid-depth, with block equivariance error 0.164. At this smaller activation scale, the fixed constants account for a larger share of the normalization denominators, while the stored evaluation scales do not adjust to the smaller activations. UCT’s local corrections no longer cancel the injected rescaling.

The baselines without normalized residual streams are more sensitive to the injected rescaling. The largest baseline increase is up to 19.7 nats among the tabulated points and 43.1 nats over the full sweep in Figure 5.<sup>1</sup>

The weight-space shadow. A bias-free matrix product $x : = W u$ is 1-homogeneous in W. When this product is the block’s only dependence on W, with no tied use or additive W-dependent path, the ray $W \mapsto e ^ { t } W$ realizes the activation rescaling $x \mapsto e ^ { t } x$ . If the downstream graph nondimensionalizes that activation, the same ray becomes an exact symmetry of the loss.

Proposition 6 (Weight-space shadow of activation typing). Let W be a parameter block whose only efect on the loss is through $x : = W u$ , where u is independent of W. Suppose the graph nondimensionalizes x such that the downstream computation is invariant to x 7→ cx $f o r \ c > 0$ when all other quantities are fixed. Then $\mathcal { L } ( e ^ { t } W ) = \mathcal { L } ( W )$ for all $t \in \mathbb { R }$ , and the radial Euler pairing vanishes exactly wherever the derivative is defined:

$$
\langle W , \nabla _ { W } \mathcal { L } \rangle _ { F } = 0 .
$$

The proof is given in App. G.10, and the architecture-specific predictions and measurement protocol are given in App. H.6.

If instead the graph fails to nondimensionalize x, the weight-space radial pairing equals the corresponding activation-space pairing $\langle x , \nabla _ { x } { \mathcal { L } } \rangle$ and is generally nonzero. App. G.10 gives the derivation.

Measured fingerprint. We also identify weight matrices whose magnitude can change without changing the loss. Prop. 6 predicts zero radial sensitivity $\langle W , \nabla _ { W } \mathcal { L } \rangle _ { F }$ when the downstream computation cancels rescaling of a matrix’s output. We measure this pairing to check whether multiplying the matrix by a common factor afects the loss. Under $W \mapsto e ^ { t } W$ , this pairing is the loss derivative at $t = 0$ , measured in nats per e-fold. One e-fold is a factor of $e \approx 2 . 7 1 8$ in weight magnitude, corresponding to a one-unit increase in log-scale. Table 9 reports the predicted zero and nonzero roles and the largest measured sensitivity across layers for each architecture at 124M. All 54 of 54 architecture-by-role predictions match across the eight trained models, with predicted-null cells at the fp32 floor of at most $2 . 4 \times 1 0 ^ { - 7 }$ nats per e-fold. The 1B audit likewise matches 34 of 34 cells, with predicted-null values at most $3 . 3 \times 1 0 ^ { - 9 }$ nats per e-fold. This agreement shows that NDA predicts which weight matrices can be rescaled without changing the loss. For example, QK-Norm cancels rescaling of the query and key projections but retains sensitivity to the other probed matrices.

Sec. 5.4 reports the aggregate fingerprint. Table 9 gives the full rolewise audit, reporting the worst layer in every architecture-by-role cell. GCT and nGPT are null on every role. QK-Norm is null exactly on q and k and nowhere else. NormFormer’s post-attention LayerNorm types NormFormer’s attention write through v and the output projection, but no other role. Pre-LN, LLaMA-style, and LayerScale have no null direction, with pairings from $3 . 5 \times 1 0 ^ { - 3 } \mathrm { ~ t o ~ } 2 . 6 \times 1 0 ^ { - 2 }$ nats per e-fold.

The local-correction UCT is null where UCT’s live-RMS adapter acts, but the q, k and gate interfaces corrected with frozen EMA statistics lose nullity at deployment. When batch statistics are active, the finite-diference slope vanishes at $1 . 5 \times 1 0 ^ { - 7 }$ while the stop-gradient autograd pairing remains $1 . 1 \times 1 0 ^ { - 3 }$ . Here local correction restores forward invariance with live batch statistics, while the stop-gradient training computation and the frozen deployment graph retain radial sensitivity.

The following PTQ evaluations test how the same architectures respond to rounding errors during finite-precision storage and computation.

## I.3 Extended PTQ results

Integer quantization stores integer coordinates together with a scale that restores the activation magnitudes. A shared tensor scale must cover large activation values, making the integer grid relatively coarse for tokens with smaller magnitudes. Per-token scaling assigns a separate range of representable values to each token. Tokens with smaller activations can then use more closely spaced values instead of the coarse spacing set by the largest activations in the tensor.

Quantizing the same trained functions in Table 10 and Figure 6 diferentiates architectures along the scalemanagement axis predicted by the typing rules. Static per-site, per-tensor int8 changes the normalized-gauge models by at most +0.005 nats, while the other 124M baselines change by +0.034 to +1.099 nats. Per-token scales reduce the worst 124M change to +0.002 nats. At 1B under per-token int4, GCT has the smallest increase, at +0.739 nats.

Table 9: The Prop. 6 fingerprint on the trained 124M models in fp32 evaluation mode. Entries report max<sub>ℓ</sub> $| \langle W _ { \ell } , \nabla _ { W _ { \ell } } \mathcal { L } \rangle |$ in nats per e-fold for each weight-matrix role. The symbol † marks a cell predicted to be an exact null by the architecture’s typing, with the proof in App. G.10 and the rolewise predictions in App. H.6. Every marked cell sits at the fp32 floor, while every other cell is orders of magnitude above that floor. Equal-pairing roles agree as the theory requires for q and k, for v and the attention output, and for the up-projection and MLP output. A double dash denotes an absent role because a GELU MLP has no separate up-projection.
<table><tr><td></td><td colspan="4">attention</td><td colspan="3">MLP</td></tr><tr><td>architecture</td><td>q</td><td>k</td><td>v</td><td>out</td><td>in</td><td>up</td><td>out</td></tr><tr><td>Pre-LN</td><td> $4 . 1 \times 1 0 ^ { - 3 }$ </td><td> $4 . 1 \times 1 0 ^ { - 3 }$ </td><td> $3 . 5 \times 1 0 ^ { - 3 }$ </td><td> $3 . 5 \times 1 0 ^ { - 3 }$ </td><td> $2 . 6 \times 1 0 ^ { - 2 }$ </td><td>一</td><td> $1 . 6 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>LLaMA-style</td><td> $4 . 2 \times 1 0 ^ { - 3 }$ </td><td> $4 . 2 \times 1 0 ^ { - 3 }$ </td><td> $1 . 2 \times 1 0 ^ { - 2 }$ </td><td> $1 . 2 \times 1 0 ^ { - 2 }$ </td><td> $2 . 3 \times 1 0 ^ { - 2 }$ </td><td> $1 . 6 \times 1 0 ^ { - 2 }$ </td><td> $1 . 6 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>QK-Norm</td><td> $6 . 4 \times 1 0 ^ { - 9 } \dag$ </td><td> $1 . 5 \times 1 0 ^ { - 8 } \dagger$ </td><td> $4 . 6 \times 1 0 ^ { - 3 }$ </td><td> $4 . 6 \times 1 0 ^ { - 3 }$ </td><td> $2 . 1 \times 1 0 ^ { - 2 }$ </td><td> $1 . 3 \times 1 0 ^ { - 2 }$ </td><td> $1 . 3 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>NormFormer</td><td> $4 . 3 \times 1 0 ^ { - 3 }$ </td><td> $4 . 3 \times 1 0 ^ { - 3 }$ </td><td> $2 . 4 \times 1 0 ^ { - 7 } \dagger$ </td><td> $2 . 4 \times 1 0 ^ { - 7 } \dagger$ </td><td> $9 . 2 \times 1 0 ^ { - 3 }$ </td><td></td><td> $1 . 3 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>LayerScale</td><td> $4 . 5 \times 1 0 ^ { - 3 }$ </td><td> $4 . 5 \times 1 0 ^ { - 3 }$ </td><td> $6 . 7 \times 1 0 ^ { - 3 }$ </td><td> $6 . 7 \times 1 0 ^ { - 3 }$ </td><td> $2 . 0 \times 1 0 ^ { - 2 }$ </td><td> $1 . 1 \times 1 0 ^ { - 2 }$ </td><td> $1 . 1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>nGPT</td><td> $2 . 9 \times 1 0 ^ { - 1 0 } \dagger$ </td><td> $3 . 9 \times 1 0 ^ { - 1 0 } \dag$ </td><td> $5 . 0 \times 1 0 ^ { - 1 0 } \dagger$ </td><td> $1 . 0 \times 1 0 ^ { - 9 } \dag$ </td><td> $3 . 5 \times 1 0 ^ { - 1 0 } \dag$ </td><td> $3 . 7 \times 1 0 ^ { - 1 0 } \dagger$ </td><td> $8 . 4 \times 1 0 ^ { - 1 0 } \dagger$ </td></tr><tr><td>Corrective UCT (ours)</td><td> $1 . 3 \times 1 0 ^ { - 2 }$ </td><td> $1 . 3 \times 1 0 ^ { - 2 }$ </td><td> $4 . 3 \times 1 0 ^ { - 8 } \dagger$ </td><td> $4 . 4 \times 1 0 ^ { - 8 } \dot { \mathsf { I } }$ </td><td> $1 . 9 \times 1 0 ^ { - 2 }$ </td><td> $3 . 0 \times 1 0 ^ { - 8 } \dagger$ </td><td> $2 . 7 \times 1 0 ^ { - 8 } \dagger$ </td></tr><tr><td>GCT (ours)</td><td> $1 . 1 \times 1 0 ^ { - 9 } \dagger$ </td><td> $1 . 9 \times 1 0 ^ { - 9 } \dagger$ </td><td> $3 . 7 \times 1 0 ^ { - 9 } \dot { \mathsf { I } }$ </td><td> $1 . 1 \times 1 0 ^ { - 9 } \dagger$ </td><td> $2 . 0 \times 1 0 ^ { - 9 } \dagger$ </td><td> $3 . 7 \times 1 0 ^ { - 9 } \dot { \mathsf { I } }$ </td><td> $2 . 5 \times 1 0 ^ { - 9 } \dot { + }$ </td></tr></table>

Table 10: PTQ of the trained 124M models. Values are changes in validation loss relative to bf16, measured in nats, and positive values are worse. The 8-bit weight-only setting is not tabulated because that setting changes loss by 0.000 nats for every architecture, both here and at 1B. W8A8 uses static absmax calibration at every linear input. The h → int8 and h → int4 columns quantize the residual stream at every block boundary with per-layer static tensor scales or per-token scales. The logit head remains in full precision.
<table><tr><td></td><td></td><td colspan="2">weights and activations</td><td colspan="2">stream h → int8</td><td colspan="2">stream h → int4</td></tr><tr><td>architecture</td><td>base</td><td>W4</td><td>W8A8</td><td>tensor</td><td>token</td><td>tensor</td><td>token</td></tr><tr><td>Pre-LN</td><td>3.118</td><td>+0.067</td><td>+0.016</td><td>+0.085</td><td>+0.002</td><td>+6.723</td><td>+0.912</td></tr><tr><td>LLaMA-style</td><td>3.126</td><td>+0.063</td><td>+0.270</td><td>+1.058</td><td>+0.001</td><td>+6.212</td><td>+1.044</td></tr><tr><td>QK-Norm</td><td>3.098</td><td>+0.061</td><td>+0.237</td><td>+1.099</td><td>+0.001</td><td>+7.715</td><td>+0.386</td></tr><tr><td>NormFormer</td><td>3.139</td><td>+0.125</td><td>+0.234</td><td>+0.034</td><td>+0.001</td><td>+4.164</td><td>+0.761</td></tr><tr><td>LayerScale</td><td>3.120</td><td>+0.061</td><td>+0.178</td><td>+0.896</td><td>+0.002</td><td>+6.214</td><td>+0.691</td></tr><tr><td>nGPT</td><td>3.119</td><td>+0.070</td><td>+0.024</td><td>+0.005</td><td>+0.001</td><td>+2.951</td><td>+0.522</td></tr><tr><td>Corrective UCT (ours)</td><td>3.204</td><td>+0.104</td><td>+0.056</td><td>+0.007</td><td>+0.000</td><td>+2.033</td><td>+0.152</td></tr><tr><td>GCT (ours)</td><td>3.118</td><td>+0.056</td><td>+0.026</td><td>+0.001</td><td>+0.000</td><td>+0.828</td><td>+0.107</td></tr></table>

The control schemes distinguish sensitivity to the quantization site and scale granularity. Weight-only int8 changes loss by 0.000 nats at the reported precision at both model scales, whereas weight-only int4 increases loss by at most 0.125 nats. Static W8A8 produces larger, architecture-dependent changes, including +0.833 nats for QK-Norm at 1B (Table 2). The KV-cache control below has much smaller changes across all architectures. The strongest separation between normalized streams and the other baselines occurs under per-tensor residual-stream int8. Per-token scales substantially reduce this separation, supporting cross-token scale variation as an important contributor.

Kernel fidelity. Simulated quantization followed by dequantization and a floating-point general matrix multiplication (GEMM) could diverge from deployed integer arithmetic. To check fidelity, we reran the W8A8 setting of Table 10 through the actual int8 tensor-core path, torch.\_int\_mm, with int8×int8 products accumulated in 32-bit integers and identical scale computation. Every architecture’s loss delta agrees with the simulated delta for the same architecture to within 0.002 nats across all 8 trained 124M models, while the deltas span up to 0.233 nats. This agreement supports the fidelity of the simulated W8A8 evaluation. The residual-stream protocols evaluate quantization and dequantization without replacing matrix multiplication.

Cost of fine-grained scale management. On the same device, the per-token quantization and dequantization step takes 1.02× as long as the per-tensor step and 72% as long as the feed-forward GEMM surrounded by the quantization and dequantization operations. Scale metadata for cached or ofloaded activations grows from 48 B for the per-tensor path to 48 KB per 1024 tokens for a 24-layer stream. A normalized-gauge stream still incurs the per-tensor round trip but avoids this additional fine-grained management cost.

![](images/19d82937f35cc5cc77180d321686b11a69b10a32391c60347405126869d83462.jpg)  
Figure 6: Quantization degradation by scheme on a logarithmic scale. Per-tensor int8 on the residual stream separates the architectures by orders of magnitude. Per-token scales remove token-to-token scale spread from the quantizer’s relative resolution and substantially reduce the int8 gap.

The migration alternative. Scale migration provides another way to manage activation ranges, as reviewed in App. C. SmoothQuant moves calibrated per-channel activation scales into adjacent weights ofline (Xiao et al., 2023), while calibrated per-channel scales provide analogous range control for a stored stream. Rotation-based schemes redistribute coordinate magnitudes by changing the basis (Ashkboos et al., 2024). All of these methods can be applied to trained models.

Migration reduces the measured quantization error on the same trained models. SmoothQuant with α = 0.5 and per-site static calibration reduces the worst-baseline W8A8 envelope of Table 10 from +0.270 to +0.061 nats, with QK-Norm worst after migration. Per-channel static scales reduce the worst-baseline residual-stream envelope from 1.10 to +0.003 nats, with LLaMA-style worst after migration. At 1B the worst value is +0.007, compared with 5.51 before this treatment. For GCT, the two treatments produce loss changes of +0.008 and +0.000 nats, respectively. At 1B, the stream treatment changes GCT’s loss by +0.000 nats.

The serving stack can manage scale externally, or the architecture can control scale before deployment. Static migration and channel scales require a calibration pass for each deployment target, using 8 held-out batches here, and may require recalibration under distribution drift. The dynamic per-token route carries fine-grained metadata on every cache and ofload path, with the 1.02× runtime ratio and 48 KB per 1024 tokens measured above. Both routes require kernel support for the chosen granularity. The normalized-gauge architectures meet the int8 loss threshold with per-tensor scales in this setting, while retaining the ordinary quantization and dequantization cost.

The KV cache gives a negative control. KV-cache compression is a separate deployment target (Liu et al., 2024). Every architecture computes keys and values from normalized block inputs, removing accumulated radial stream drift before the projections. Equal projected key and value norms require additional conditions because a learned projection can still produce direction-dependent norms. We measure $\kappa _ { \mathrm { s c a l e } } \approx 1$ empirically.

Consistent with that observation, int8 KV-cache quantization is near-lossless for all eight architectures. The worst increases are +0.0004 nats with per-tensor scales and +0.0001 with per-token scales, with no separation between normalized-gauge and baseline blocks. The control holds unchanged at 1B, where the worst increase is +0.0002 nats. These controls localize the measured deployment gap of Table 10 to residual-stream storage.

Persistence at scale. We repeat the study on the trained 1B models (Table 2). The separation persists over the tested 8× parameter-count increase, with GCT at 0.009 nats compared with 5.51 nats for LLaMA-style under static per-tensor stream int8.

Table 13 and Figure 7 add the 350M point between 124M and 1B. The largest baseline loss increase under static per-tensor stream int8 rises from +1.10 nats at 124M to +2.45 at 350M and +5.51 at 1B, while GCT increases by +0.001, +0.003, and +0.009 nats, respectively. Across these model sizes, the largest baseline loss increase grows with parameter count, while nGPT and GCT retain small int8 loss increases.

The int4 results also reflect factors beyond cross-token scale spread. In the zero-floor theory, canonicalization fixes GCT’s cross-token RMS spread at $\kappa _ { \mathrm { s c a l e } } \equiv 1$ , and the finite-floor implementation fixes this spread to numerical precision. The remaining error can depend on within-vector amplitude $\kappa _ { \mathrm { a m p } }$ , which the canonical gauge leaves untouched, together with downstream directional sensitivity and accumulation across sites, neither of which the proposition quantifies. Measured per-tensor int4 losses increase substantially for every stream. Under per-token int4, GCT has the smallest measured increase, at +0.739 nats (Table 2). Explaining this empirical ordering requires analysis of the additional factors beyond cross-token scale spread. On the independent second 1B seed, the parameter-fingerprint audit again matches all 34 of 34 predictions; the corresponding PTQ repeat is reported in Sec. 5.5.

Exact internal-rescaling invariance of the zero-floor construction removes the radial component of a scale distortion. Robustness to quantization additionally depends on the tangential and within-vector terms quantified by Thm. 6 and Prop. 5. The deployment tables measure the combined efects of radial rejection, tangential perturbations, and within-vector amplitude.

Residual-stream storage at fixed loss thresholds. The residual-stream cache occupies 5.25 GiB in bf16 for an illustrative 1B configuration with 32 concurrent sequences and context length 2048. At a 0.05-nat threshold, nGPT and GCT use calibrated per-site int8 tensor scales and occupy 2.63 GiB, whereas the other baselines require per-token int8 scales. At a 1-nat threshold, GCT reaches 4× compression at 1.32 GiB with a +0.739-nat increase, whereas the other evaluated architectures exceed that threshold under int4. These precision choices use the measured loss increases in Table 2.

These storage figures include only the residual-stream cache and allow per-token scaling for every architecture. Weight and KV-cache storage are excluded. KV-cache quantization is evaluated separately above. The 4× result uses per-token int4 and has the measured +0.739-nat loss increase. The option to use per-tensor scales applies to the 2× int8 configuration.

Measured latency. We time the dynamic-absmax residual-stream storage-and-retrieval round trip on device during a forward pass of the independent second-seed 1B model, with d = 1792, L = 24, B = 32 sequences, and context length T = 2048. The preceding accuracy budget uses the calibrated static primary-seed column of Table 2. Separately measured dynamic per-tensor deltas on both 1B seeds induce the same 0.05-nat tier assignment. GCT and nGPT meet this threshold, whereas the other baselines exceed the loss budget.

At this near-lossless granularity, the per-tensor paths add 6.2% and 7.4% forward latency for GCT and nGPT, respectively. The per-token paths required by Pre-LN, LLaMA-style, and QK-Norm add 12.3%, 12.9%, and 9.9%. Every path reduces the bf16 stream cache from 5.25 to 2.63 GiB. Across these configurations, the measured latency overhead ranges from 6% to 13%.

At the stated loss threshold, the normalized-gauge streams need only coarse per-tensor scales, whereas the other baselines require per-token scales. The primary-seed int4 per-token deltas are +0.74 nats for GCT, +4.66 for nGPT, and +5.23 to +7.10 for the other baselines. Timing and accuracy use diferent protocols because static calibration and dynamic absmax are diferent operations.

Table 11: Downstream accuracy under residual-stream quantization on the 1B models trained independently with a second seed. We report zero-shot accuracy with the harness formatting for LAMBADA, HellaSwag, PIQA and ARC-Easy. The stream is stored at the same block-boundary sites and granularities as in Table 2, using dynamic scales without calibration, and the evaluation is inference-only, with no retraining or fine-tuning. The bf16 column gives mean accuracy over the four tasks in percent, and the remaining columns give accuracy changes in points, where negative values indicate lower accuracy.
<table><tr><td rowspan="2">architecture</td><td rowspan="2">bf16 (%)</td><td colspan="2">int8</td><td>int4</td></tr><tr><td>tensor</td><td>token</td><td>token</td></tr><tr><td>Pre-LN</td><td>48.6</td><td>-20.5</td><td>-1.9</td><td>-22.7</td></tr><tr><td>LLaMA-style</td><td>48.7</td><td>-22.7</td><td>-0.8</td><td>-22.6</td></tr><tr><td>QK-Norm</td><td>49.4</td><td>-19.6</td><td>-0.5</td><td>-23.1</td></tr><tr><td>nGPT</td><td>49.0</td><td>-0.8</td><td>-0.3</td><td>-17.4</td></tr><tr><td>GCT (ours)</td><td>49.0</td><td>-0.1</td><td>+0.2</td><td>-7.4</td></tr></table>

Table 12: External-validity audit on of-the-shelf GPT-2. The OpenAI GPT-2 family uses Pre-LN and was trained by a third party on WebText. We evaluate the family on the same validation bins and residual-stream fake-quantization hooks as our trained models, using dynamic scales, with loss increases reported in nats relative to each unquantized model. The quantity $\kappa _ { \mathrm { d e p t h } }$ is the ratio between the maximum and minimum mean residualstream RMS across block inputs, not the within-tensor spread of Prop. 5. Every size has a large per-tensor loss increase and a near-lossless per-token result. For comparison, our 124M GCT model has $\kappa _ { \mathrm { d e p t h } } \approx 1$ and a +0.001-nat dynamic-int8 loss increase.
<table><tr><td rowspan="2">model</td><td rowspan="2">Kdepth</td><td colspan="2">int8</td><td>int4</td></tr><tr><td>tensor</td><td>token</td><td>token</td></tr><tr><td>GPT-2 (124M)</td><td>48</td><td>+5.79</td><td>+0.008</td><td>+4.25</td></tr><tr><td>GPT-2 (355M)</td><td>236</td><td>+7.91</td><td>+0.046</td><td>+8.02</td></tr><tr><td>GPT-2 (774M)</td><td>223</td><td>+9.03</td><td>+0.034</td><td>+6.02</td></tr><tr><td>GPT-2 (1.5B)</td><td>596</td><td>+7.30</td><td>+0.035</td><td>+8.05</td></tr></table>

Table 13: Calibrated per-site, per-tensor int8 residual-stream quantization along the 124M, 350M, and 1B ladder. Columns report one trained model per architecture, the primary seed at 124M and 1B and the second seed at 350M. nGPT and GCT remain near the int8 floor. The largest baseline loss increase grows with parameter count, although individual trajectories need not be monotone.
<table><tr><td rowspan="2">architecture</td><td colspan="3">model size</td></tr><tr><td>124M</td><td>350M</td><td>1.0B</td></tr><tr><td>Pre-LN</td><td>+0.085</td><td>+1.009</td><td>+3.018</td></tr><tr><td>LLaMA-style</td><td>+1.058</td><td>+2.453</td><td>+5.513</td></tr><tr><td>QK-Norm</td><td>+1.099</td><td>+0.325</td><td>+5.426</td></tr><tr><td>nGPT</td><td>+0.005</td><td>+0.010</td><td>+0.032</td></tr><tr><td>GCT (ours)</td><td>+0.001</td><td>+0.003</td><td>+0.009</td></tr></table>

Table 11 gives the full architecture comparison for the downstream accuracy results reported in Sec. 5.5.

The publicly available pretrained GPT-2 models show a similar dependence on scale granularity (Table 12). Across the four sizes, $\kappa _ { \mathrm { d e p t h } }$ ranges from 48 to 596. Dynamic per-tensor int8 residual storage increases loss by 5.79 to 9.03 nats, while per-token scaling limits the increase to 0.046 nats. This audit extends the observed quantization pattern to independently trained models. The depth statistic is descriptive and difers from the within-tensor spread used in the rounding bound.

## I.4 Additional optimizer and width evaluations

Two checks test whether the comparison generalizes beyond one optimizer and one width. Table 14 gives the width results.

![](images/3c5c7aa7e17b74db167893fbe168a2d509657646016e0d88f02a652f1c6224db.jpg)  
Figure 7: Table 13 on log–log axes. Along the tested model-size ladder, the worst-baseline increase reaches +5.51 nats at 1B, while nGPT and GCT remain near the int8 floor.

Optimizer pairing. The optimizer rule in Sec. 2.2 requires the full displacement to transform from dual to primal units. Muon supplies a competitive shape-aware orthogonalized direction with a fixed scalar step. The experiment tests compatibility rather than the covariance law. All three architectures improve at matched tokens, with similar gains for the typed and untyped blocks. Relative to the shared-recipe AdamW point at matched tokens, the changes are −0.309 nats for GCT, −0.329 for Pre-LN, and −0.242 for QK-Norm. The result shows that the typed architecture and this shape-aware optimizer compose without changing the architecture.

Width transfer and $\mu \mathbf { P }$ composition. Two experiments probe the parameterization axis in Table 14. The learning-rate robustness protocol of Table 5 is repeated at $d \in \{ 2 5 6 , 1 0 2 4 \}$ with $d _ { h } = 6 4$ and the rest of the recipe unchanged. Raising the rate from $6 \times 1 0 ^ { - 4 }$ to $4 . 8 \times 1 0 ^ { - 3 }$ improves GCT at every width, with changes of −0.592, −0.369, and −0.245 nats for $d \in \{ 2 5 6 , 5 1 2 , 1 0 2 4 \}$ . At the higher rate, Pre-LN is +0.141, +0.131, and +2.039 nats above Pre-LN’s best sampled value at the respective width. We then compose both architectures with the learning-rate-only $\mu \mathrm { P }$ rule at the respective $d = 5 1 2 \mathrm { - t u n e d }$ learning rates. This rule scales hidden-matrix learning rates by $5 1 2 / d$ and leaves initialization and multipliers unchanged. The rule changes GCT loss by at most 0.006 nats at either width, namely −0.005 at d = 256 and +0.006 at $d = 1 0 2 4$ , while the Pre-LN changes are $+ 0 . 0 2 0$ at $d = 2 5 6$ and −0.068 at d = 1024. These results support combining the tested width-dependent learning-rate rule with canonical activation scaling.

## I.5 Vision: robustness to photometric and input-unit shifts

GCT has the highest clean top-1 accuracy among the five variants at 74.2%, compared with 70.7% for QK-Norm (Table 15). All models are trained to convergence with the standard DeiT-Ti recipe. Robustness is then compared on contrast, brightness, and noise shifts, with the full accuracy grid in Figure 8. The sharpest comparison is multiplicative rescaling of the normalized input, which is the vision analogue of an input-unit shift.

Table 14: Width transfer and learning-rate-only $\mu \mathrm { P }$ composition for the 51M-centered family, with $L = 8 , d _ { h } = 6 4$ and the primary seed under the Table 5 token budget. The standard-parameterization (SP) columns use the listed learning rate. The $\mu \mathrm { P }$ columns scale hidden-matrix learning rates by 512/d from the architecture’s d = 512-tuned rate, namely $4 . 8 \times 1 0 ^ { - 3 }$ for GCT and $1 . 2 \times 1 0 ^ { - 3 }$ for Pre-LN. At the tuning width, the rule reduces to SP.
<table><tr><td colspan="4">GCT</td><td colspan="4">Pre-LN</td></tr><tr><td>width</td><td> $6 \times 1 0 ^ { - 4 }$ </td><td> $4 . 8 \times 1 0 ^ { - 3 }$ </td><td> $\mu \mathrm { P }$ </td><td> $6 \times 1 0 ^ { - 4 }$ </td><td> $1 . 2 \times 1 0 ^ { - 3 }$ </td><td> $4 . 8 \times 1 0 ^ { - 3 }$ </td><td> $\mu \mathrm { P }$ </td></tr><tr><td> $d = 2 5 6$ </td><td>4.366</td><td>3.773</td><td>3.768</td><td>4.419</td><td>4.166</td><td>4.307</td><td>4.186</td></tr><tr><td> $d = 5 1 2$ </td><td>3.885</td><td>3.517</td><td>3.517</td><td>3.943</td><td>3.750</td><td>3.881</td><td>3.750</td></tr><tr><td> $d = 1 0 2 4$ </td><td>3.590</td><td>3.345</td><td>3.351</td><td>3.613</td><td>3.617</td><td>5.652</td><td>3.549</td></tr></table>

Table 15: CIFAR-100 from scratch with the DeiT-Ti recipe for 300 epochs. Entries report top-1 accuracy in percent. Robustness columns average over shift grids with noise $\sigma \in \{ 0 . 0 5 , 0 . 1 \}$ , contrast {0.5, 2.0}, brightness {0.7, 1.3}, and input scale {0.25, 0.5, 2, 4}.
<table><tr><td></td><td></td><td colspan="4">robustness</td></tr><tr><td>architecture</td><td>clean</td><td>noise</td><td>contrast</td><td>bright</td><td>in-scale</td></tr><tr><td>Pre-LN</td><td>69.81</td><td>37.49</td><td>63.27</td><td>68.45</td><td>51.58</td></tr><tr><td>QK-Norm</td><td>70.68</td><td>36.92</td><td>64.80</td><td>69.58</td><td>56.00</td></tr><tr><td>nGPT</td><td>73.98</td><td>34.34</td><td>68.92</td><td>73.00</td><td>63.54</td></tr><tr><td>Corrective UCT (ours)</td><td>73.28</td><td>33.66</td><td>68.00</td><td>72.25</td><td>62.25</td></tr><tr><td>GCT (ours)</td><td>74.17</td><td>34.36</td><td>69.23</td><td>72.93</td><td>64.13</td></tr></table>

![](images/d6513f20805983df1c3445070dfd10a19c86bdd2d308d82632f4fccfe518a207.jpg)  
Figure 8: ViT robustness $\mathrm { g r i d }$ on CIFAR-100 accompanying Table 15. The figure reports top-1 accuracy under the photometric and input-scale shift grid.

Additive-noise robustness. GCT scores 34.4% on the averaged Gaussian-noise shift, against 37.5% for the best baseline in the noise column of Table 15. One possible explanation is that canonicalization removes magnitude information that can help distinguish signal from additive noise. The CIFAR-100 results are consistent with this explanation, although the causal role of magnitude removal remains to be tested.

Table 16: Tiny-ImageNet-200 from scratch with ViT-S, patch size 8, the DeiT recipe, and 200 epochs. Accuracy is reported as in Table 15, and the noise, contrast, and input-scale columns average over the corresponding shift grids.
<table><tr><td></td><td></td><td colspan="3">robustness</td></tr><tr><td>architecture</td><td>clean</td><td>noise</td><td>contrast</td><td>in-scale</td></tr><tr><td>Pre-LN</td><td>54.94</td><td>44.67</td><td>48.53</td><td>37.32</td></tr><tr><td>LLaMA-style</td><td>55.14</td><td>45.11</td><td>49.66</td><td>40.91</td></tr><tr><td>QK-Norm</td><td>56.30</td><td>46.24</td><td>51.30</td><td>43.25</td></tr><tr><td>NormFormer</td><td>43.18</td><td>34.49</td><td>35.52</td><td>21.27</td></tr><tr><td>LayerScale</td><td>56.38</td><td>45.48</td><td>50.87</td><td>39.73</td></tr><tr><td>nGPT</td><td>57.81</td><td>48.18</td><td>53.31</td><td>47.15</td></tr><tr><td>Corrective UCT (ours)</td><td>57.74</td><td>47.05</td><td>52.22</td><td>45.64</td></tr><tr><td>GCT (ours)</td><td>59.25</td><td>48.42</td><td>54.18</td><td>49.47</td></tr></table>

Table 17: PINN relative $L _ { 2 }$ error of the recovered physical field, averaged over five seeds, when the data’s unit system is rescaled by κ. The problem is physically identical across columns, and only the units change. The last column gives the worst case over κ.
<table><tr><td>problem</td><td>method</td><td> $\kappa = 1 0 ^ { - 2 }$ </td><td> $\kappa = 1$ </td><td> $\kappa = 1 0 ^ { 2 }$ </td><td>worst</td></tr><tr><td rowspan="7">Burgers</td><td>equal</td><td>0.848</td><td>0.014</td><td>0.337</td><td>0.848</td></tr><tr><td>manual  $\lambda _ { \mathrm { s u p } } = 1 0 0$ </td><td>0.491</td><td>0.081</td><td>0.431</td><td>0.491</td></tr><tr><td>uncertainty</td><td>0.520</td><td>0.006</td><td>0.618</td><td>0.618</td></tr><tr><td>LRA</td><td>0.812</td><td>0.006</td><td>0.230</td><td>0.812</td></tr><tr><td>NTK balance</td><td>0.560</td><td>0.214</td><td>0.550</td><td>0.560</td></tr><tr><td>NDA-typed (ours)</td><td>0.008</td><td>0.009</td><td>0.008</td><td>0.009</td></tr><tr><td>NDA-typed + LRA</td><td>0.005</td><td>0.004</td><td>0.004</td><td>0.005</td></tr><tr><td rowspan="7">Helmholtz</td><td>equal</td><td>0.111</td><td></td><td></td><td>2.973</td></tr><tr><td>manual  $\lambda _ { \mathrm { s u p } } = 1 0 0$ </td><td></td><td>0.110</td><td>2.973</td><td></td></tr><tr><td></td><td>0.035</td><td>0.005</td><td>2.002</td><td>2.002</td></tr><tr><td>uncertainty</td><td>0.054</td><td>0.002</td><td>2.175</td><td>2.175</td></tr><tr><td>LRA</td><td>0.048</td><td>0.006</td><td>1.620</td><td>1.620</td></tr><tr><td>NTK balance</td><td>0.036</td><td>0.390</td><td>0.971</td><td>0.971</td></tr><tr><td>NDA-typed (ours)</td><td>0.010</td><td>0.015</td><td>0.014</td><td>0.015</td></tr></table>

A second dataset at larger scale. We repeat the from-scratch comparison on Tiny-ImageNet-200 with a ViT-S, images with 4× the CIFAR pixel area, and 200 classes (Table 16). Patch size 8 preserves the same $8 \times 8$ patch-token grid as the CIFAR model with patch size 4 instead of increasing the token count. GCT has the highest clean top-1 accuracy, 59.2%, compared with 56.4% for LayerScale. The contrast and input-scale results follow the pattern in Table 15. The noise comparison difers. GCT scores 48.4%, compared with 48.2% for nGPT and 46.2% for QK-Norm. The noise ordering changes under the combined change in dataset, resolution, and backbone.

NormFormer reaches 43.2% clean top-1 accuracy, about 12 percentage points below most other baselines under this shared recipe. We report this result under the shared recipe, without architecture-specific retuning. The cause of the gap remains unresolved. GCT has the highest measured accuracy in each column of Table 16.

Across both vision datasets, GCT combines the highest measured clean accuracy with strong photometric and input-scale robustness. These observations complement the internal-rescaling tests in language modeling, while the dataset-dependent noise results distinguish scale robustness from robustness to arbitrary input perturbations.

## I.6 Multi-loss objectives: unit-system invariance without adaptivity

Table 17 reports recovered-field error as the data’s unit system is rescaled by $\kappa \in \{ 1 0 ^ { - 2 } , 1 , 1 0 ^ { 2 } \}$ . The physical problem is identical across columns. The NDA-typed objective and the NDA-typed + LRA composition are the only evaluated methods that remain in the same low-error range across this $1 0 ^ { 4 }$ unit sweep, whereas the other adaptive methods degrade for at least one rescaling.

![](images/85788efa8fd3a4f6b78830cd572cf643503a032fd80b778abde15dc8ea6da114.jpg)

![](images/a11d1f8130754fa0b74735fdbeebbdd18c91e225ba41b18c37076a083812ba1d.jpg)

![](images/11bf7adefdd779b44fd79d3e3368eff84bbb3b6a7c4ed3189affaf8a23ca800d.jpg)  
Figure 9: Same Burgers flow, diferent numerical units: qualitative field recovery at seed 0 after 20,000 steps. Columns use $\kappa \in \{ 1 0 ^ { - 2 } , 1 , 1 0 ^ { 2 } \}$ and are converted back to the same physical field $\hat { u } / \kappa . ~ ( \mathrm { a } , \mathrm { b } , \mathrm { c } , \mathrm { e } , \mathrm { f } , \mathrm { g } )$ Recovered field over space and time, with the recovered field’s relative $L _ { 2 }$ error above each panel, and black contours show the Cole–Hopf solution. (d, h) The same fields sliced at $t = 0 . 5$ against the exact solution (grey). Equal weighting in the top row changes the recovered field of-reference, whereas the NDA-typed head and objective in the bottom row recover consistent fields. Table 17 gives five-seed results.

Equal weighting is the unit-agnostic reference. The manual $\lambda _ { \mathrm { s u p } } = 1 0 0$ rule is tuned at $\kappa = 1$ and transferred. Uncertainty weighting, gradient-norm balancing, and NTK-trace balancing adapt from scale-sensitive training statistics. None of these baselines explicitly performs unit conversion, and each baseline degrades for at least one of-reference unit system. Equal weighting reaches a worst-case relative $L _ { 2 }$ error of 2.973 across Burgers and Helmholtz because equal loss weights conflate unit conversion with task preference.

The NDA-typed objective types the output head and divides each term by an observable characteristic scale from App. H. The NDA-typed objective is invariant by construction and remains in the low-error range across κ, with worst case 0.015. The same objective also performs comparably at the reference scale without tuning, adaptivity, or access to the solution. This benchmark applies the loss-aggregation rule of Sec. 2.2 end-to-end, typing both the output head and every loss term.

Practitioner guides for PINNs already recommend nondimensionalizing a problem before training (Wang et al., 2023). The typed objective derives this nondimensionalization step directly from the loss-aggregation rule. Thm. 2 shows that every invariant linear alternative whose weights depend only on these characteristic scales must use the same inverse-unit factors. Any remaining scale dependence is restricted to dimensionless groups. Cor. 1 shows that the fixed-α rule is unique in this class when the declared loss units vary independently.

Typing and adaptivity also compose. Applying LRA on top of the nondimensionalized terms, denoted NDA-typed + LRA in Table 17, retains unit-system invariance with worst case 0.005 across κ. This combination lets adaptation adjust dimensionless task preferences after unit conversion, demonstrating compatibility between typing and adaptive weighting.