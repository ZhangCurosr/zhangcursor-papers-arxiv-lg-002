# Same Function, Diferent Trajectory: No Standard Optimizer Is Equivariant to Transformer Reparameterization

Bum Jun Kim<sup>∗</sup> The University of Tokyo bumjun.kim@weblab.t.u-tokyo.ac.jp

## Abstract

We show that two Transformers representing the same function can follow diferent training trajectories. For example, doubling a query matrix and halving its key matrix preserves the model’s function but changes the gradient scales seen by the optimizer. Our theoretical analysis establishes that standard block-local first-order optimizers fail to scale updates with parameters under a shared learning rate. For AdamW, transforming the firstand second-moment estimates alone does not correct the mismatch between parameter and update scales. To make runs restarted from equivalent checkpoints follow the same trajectory, we balance matrix pairs and transform their moment estimates together, bringing distinct but equivalent training states to the same starting point. To correct scale imbalance between paired Transformer matrices without changing predictions, we propose Symmetry-Aware Gradient Geometry Stabilization (SAGGS), which periodically equalizes the Frobenius norms of exact symmetry pairs, and its damped variant, Soft SAGGS. SAGGS reduces perplexity (PPL) by $0 . 3 7 4 \pm 0 . 0 2 6$ at 101M parameters on WikiText-103, and Soft SAGGS reduces PPL by 0.780±0.040 in a 31M modern-architecture replication, both under scale imbalance. In a 123.8M checkpoint-resume experiment, reciprocally rescaling matrix pairs barely changes validation loss at restart but increases final PPL by $3 . 7 8 \pm 0 . 4 2 ;$ hard and soft rebalancing recover to within 0.02 PPL of the control.

## 1 INTRODUCTION

Transformers (Vaswani et al., 2017) serve as the standard architecture for large language models (Radford et al., 2019; Touvron et al., 2023). Each Transformer layer consists of a multi-head self-attention block and a multi-layer perceptron (MLP). The attention block is built from query, key, value, and output projections, and the MLP is based on rectified linear units (ReLU) or Swish-gated linear units (SwiGLU) (Shazeer, 2020).

Several of these weight matrices form pairs whose scales can be traded against each other without changing the output. For example, attention compares queries and keys through the product $( \mathbf { \Delta } W ^ { Q } ) ^ { \top } \mathbf { W } ^ { K }$ of the query and key matrices, so replacing $( \dot { W } ^ { Q } , W ^ { K } )$ with $( \mathsf { \bar { c } } W ^ { \bar { Q } } , c ^ { - 1 } W ^ { \bar { K } } )$ for any $c > 0$ leaves every attention score, and hence every prediction, unchanged. We call a pair of matrices $( A , B )$ an exact symmetry pair when replacing the pair with $( c A , c ^ { - 1 } B )$ preserves the model’s function for every $c > 0$ . Rescaling the exact symmetry pairs of a model in this way yields a whole set of parameterizations that all compute the same function, which we call a symmetry orbit.

These parameterizations represent the same function but lead to diferent training trajectories. The reason is that the gradients scale inversely to the weights: Multiplying A by c multiplies its gradient by $c ^ { - 1 }$ , and multiplying B by $c ^ { - 1 }$ multiplies its gradient by c. A gradient-descent step moves the enlarged matrix $c ^ { 2 }$ times less, relative to its size, than in the original model and its shrunken partner $c ^ { 2 }$ times more, so the two models no longer compute the same function after the step. Figure 1 illustrates this efect.

Concretely, symmetry orbits in a Transformer arise from attention query–key (Q–K) and value–output (V– O) pairs, ReLU up–down projections, and the linear value–output path of SwiGLU (Sec. 2). This inverse gradient-scaling law holds for all of these pairs and follows from the chain rule (Dinh et al., 2017, Theorem 3). We use this law to determine how a rescaling distorts the optimizer’s updates and what is needed for rescaled models to train identically.

![](images/dfab9dee6ef667a12ee6e7cd1bc63cb0d7d007419d21dd32260a309c5aa9ce0f.jpg)  
Figure 1: Same function, diferent trajectory. (1) Multiplying A by c and dividing its partner B by c leaves every prediction unchanged. The rectangle’s sides are the Frobenius norms of A and B, so the rectangle changes shape but keeps its area. (2) The same numerical update causes unequal relative changes in the two blocks. (3) Resuming training from the rescaled checkpoint, whose pair norms are imbalanced but whose function is unchanged, ends with much worse perplexity (PPL) than resuming from the original. (4) Hard and Soft SAGGS rebalance the pairs without changing the current function and recover from this degradation.

Standard optimizers already include mechanisms such as adaptive gradient normalization, but these mechanisms still do not remove this dependence on the parameterization. Adam normalizes gradients using moving averages of gradients and squared gradients (Kingma and Ba, 2015), which cancels constant-factor scaling of the gradients: Both models receive the same numerical update (Figure 1(2)). To keep the two models equivalent, however, the update of the enlarged matrix would have to be c times larger and the update of its shrunken partner c times smaller. The same update is too small for the enlarged matrix and too large for its shrunken partner. Transforming Adam’s moment estimates along with the weights does not fix this mismatch, because both models still use the same learning rate (Sec. 3). Appendix A discusses in detail how this work relates to and difers from prior studies of rescaling-aware optimization, weight balancing, and symmetry-induced learning dynamics (Neyshabur et al., 2015; Stock et al., 2019; Kunin et al., 2021).

To remove the scale imbalance between paired matrices without changing the function, we introduce Symmetry-Aware Gradient Geometry Stabilization (SAGGS), which periodically rescales each exact symmetry pair so that its two matrices have equal Frobenius norms (Sec. 4). To avoid disrupting optimizer state that has adapted to imbalanced scales, Soft SAGGS applies only a fraction of this correction at each rebalancing step. Both operations leave the current function unchanged and only rebalance the weights from which training continues. SAGGS also gives an exact restart procedure: Applied together with a matching transformation of the moment estimates, SAGGS moves any two checkpoints on the same symmetry orbit to the same balanced state, from which training follows the same trajectory (Corollary 4). We evaluate both variants in deliberate rescaling tests, which rescale the pairs of layer ℓ by $c _ { \ell } = e ^ { \lambda \ell }$ to impose pair-scale imbalance without changing the function, as well as in checkpoint-resume workflows and in training without such rescaling (Sec. 5). In our seven-seed comparisons, rebalancing restores near-baseline PPL after an imposed rescaling, including a checkpoint edit, and leaves PPL nearly unchanged when no imbalance is imposed.

Our contributions are fourfold.

• We identify exact symmetry orbits in Transformer architectures (Sec. 2) and show their efect on training.

• Our theoretical analysis proves that, under a shared learning rate, no standard optimizer, including gradient descent, AdamW, and Lion, keeps two equivalent models equivalent during training (Theorem 5). Even with correctly transformed moment estimates, AdamW does so only if its learning rate, ϵ, and weight decay are also rescaled coordinate by coordinate (Theorem 2 and Example 3). We further prove that canonicalizing both weights and moments at restart makes the training resumed from any two checkpoints on the same symmetry orbit identical (Corollary 4).

• We propose SAGGS, which periodically moves each exact symmetry pair to the unique point on its orbit where the two matrices have equal Frobenius norms, regardless of where on the orbit the pair starts (Proposition 1). Its damped variant, Soft SAGGS, moves each pair only part of the way there, shrinking the log ratio of the two norms by a constant factor at each rebalancing step (Sec. 4.2).

• Under deliberate rescaling, seven-seed paired comparisons show consistent PPL reductions from rebalancing, of $0 . 3 7 4 \pm 0 . 0 2 6$ for a 101M WikiText-103 model and of $0 . 7 8 0 \pm 0 . 0 4 0$ for a 31M modernarchitecture replication, and also for training with Lion (Sec. 5.2). In a 123.8M checkpoint-resume workflow, an edit that barely changes validation loss at restart increases final PPL by $3 . 7 8 \pm 0 . 4 2$ and both rebalancing variants recover to within 0.02 PPL of the control (Sec. 5.3).

## 2 SYMMETRIES AND GRADIENTS

This section identifies which pairs of weight matrices in a Transformer can trade scale without changing the model’s function, and how such a trade changes their gradients. SAGGS rebalances exactly these pairs (Sec. 4), and the optimizer analysis of Sec. 3 starts from this change in the gradients.

## 2.1 Exact Reparameterization Symmetries

Consider an L-layer Transformer (Vaswani et al., 2017). At layer $\ell \in \{ 1 , \ldots , L \}$ , the self-attention uses projections $W _ { \ell } ^ { Q } , \dot { W } _ { \ell } ^ { K } , W _ { \ell } ^ { \bar { V } }$ , and $W _ { \ell } ^ { O }$ , and the MLP uses $\boldsymbol { W } _ { \ell } ^ { 1 }$ and $W _ { \ell } ^ { 2 }$ . Let $\mathcal { L } ( \pmb { \theta } )$ denote the data loss evaluated at parameters θ. The equations omit biases.

Table 1 records the scalar pair symmetries used in the optimizer analysis. The Q–K and V–O rows are scalar instances of the multi-head attention symmetries characterized by Tran et al. (2025); the ReLU up–down row uses standard positive homogeneity (Dinh et al., 2017; Neyshabur et al., 2015). The Gaussian error linear unit (GELU) (Hendrycks and Gimpel, 2016) is not positively homogeneous, so rescaling its up–down pair changes the function, and SAGGS excludes this pair. The SwiGLU MLP (Shazeer, 2020) is defined as

$$
\mathrm { S w i G L U } _ { \ell } ( \pmb { x } ) : = W _ { \ell } ^ { 2 } [ \mathrm { S w i s h } ( W _ { \ell } ^ { \mathrm { g a t e } } \pmb { x } ) \odot ( W _ { \ell } ^ { \mathrm { u p } } \pmb { x } ) ] ,\tag{1}
$$

where ⊙ denotes element-wise multiplication. For a fixed gate, the output is linear in the value path $W _ { \ell } ^ { \mathrm { u p } } x .$ so rescaling $W _ { \ell } ^ { \mathrm { u p } }$ by c and $W _ { \ell } ^ { 2 }$ by $c ^ { - 1 }$ leaves the output unchanged; the gate $W _ { \ell } ^ { \mathrm { g a t e } }$ is not rescaled.

Table 1: Scalar pair rescalings $( A , B ) \mapsto ( c A , c ^ { - 1 } B )$ $c > 0$ . Rows marked $\checkmark$ Yes preserve the forward pass exactly and are used by SAGGS.
<table><tr><td>Sub-layer</td><td>A</td><td>B</td><td>Mechanism</td><td>Exact</td></tr><tr><td>Self-attention</td><td> $W _ { \ell } ^ { Q }$   $W _ { \ell } ^ { V }$ </td><td> $W _ { \ell } ^ { K }$   $\bar { \boldsymbol { W _ { \ell } ^ { O } } }$ </td><td>Bilinear in logits Linear composition</td><td>√ Yes √ Yes</td></tr><tr><td>MLP (ReLU)</td><td>W1</td><td> $W _ { \ell } ^ { 2 }$ </td><td>Positive homogeneity</td><td>√Yes</td></tr><tr><td>MLP (GELÚ)</td><td> $\bar { \boldsymbol { W } _ { \ell } ^ { 1 } }$ </td><td> $\bar { \boldsymbol { W } _ { \ell } ^ { 2 } }$ </td><td>Non-homogeneous</td><td>xNo</td></tr><tr><td>SwiGLU</td><td> $W _ { \ell } ^ { \mathrm { u p } }$ </td><td> $W _ { \ell } ^ { 2 }$ </td><td>Linear in value path</td><td> ${ \swarrow \mathrm { ~ Y e s ~ } }$ </td></tr></table>

The same pairs remain exact in the modern architecture that we use for the 31M replication and the checkpointresume experiments (Sec. 5). This architecture combines SwiGLU with grouped-query attention (GQA), root mean square layer normalization (RMSNorm), and rotary position embeddings (RoPE); under GQA, the Q–K and V–O rescalings use one factor per group (Appendix B.2).

## 2.2 Gradient Covariance Identity

For an exact scalar symmetry pair, the standard chainrule identity for reciprocal rescaling (Dinh et al., 2017, Theorem 3) gives, at diferentiable points,

$$
\nabla _ { A ^ { \prime } } \mathcal { L } = c ^ { - 1 } \nabla _ { A } \mathcal { L } , \qquad \nabla _ { B ^ { \prime } } \mathcal { L } = c \nabla _ { B } \mathcal { L } ,\tag{2}
$$

where $( A ^ { \prime } , B ^ { \prime } ) : = ( c A , c ^ { - 1 } B )$ and each gradient of the same data loss is evaluated at its own parameterization. The two parameterizations compute the same function yet give the optimizer diferent gradients. In Sec. 3, we show that, under a shared learning rate, no standard optimizer scales its updates with the weights, so two equivalent models stop computing the same function after an update. We also show that canonicalizing both weights and moments at restart makes two such models train identically.

Appendix B gives the matrix shapes, the single-head attention and MLP equations, the bias co-transforms, and the argument that layer normalization leaves the pairs exact; the same appendix also discusses crossattention, mixture-of-experts, and state-space blocks, which our experiments do not use. Lemma 7 in $\mathrm { A p - }$ pendix C.1 states the general diagonal-map form of Eq. 2 used in the proofs.

## 3 EQUIVARIANCE AND RESTARTS

Two models on the same orbit compute the same function but give the optimizer diferent gradients (Eq. 2). This section asks whether an optimizer can compensate for this diference and keep the two models equivalent after every update. If the optimizer cannot, the position of a model on its orbit afects training, which motivates rebalancing (Sec. 4). We fix one balanced representative of each orbit (Sec. 3.1), derive what AdamW needs to keep two runs equivalent (Sec. 3.2), show that restarting both runs from the same balanced state makes the resumed runs identical (Sec. 3.3), and prove that no standard optimizer keeps two runs equivalent under a shared learning rate (Sec. 3.4).

## 3.1 Scalar Orbit Canonicalization

All points on an orbit compute the same function, so any one of these points can represent the orbit. We use the point at which the two matrices of each exact pair have equal Frobenius norms. The exact restart (Sec. 3.3) and SAGGS (Sec. 4.1) both move a model to this point.

Proposition 1 (Unique balanced representative of a scalar orbit). For an exact symmetry pair $\left( A _ { p } , B _ { p } \right)$ with positive Frobenius norms, let $A _ { p } ^ { \star } : = A _ { p } / c _ { p } ^ { \star }$ and $B _ { p } ^ { \star } : = c _ { p } ^ { \star } B _ { p }$ with $c _ { p } ^ { \star } : = ( \| A _ { p } \| _ { \mathrm { F } } / \| B _ { p } \| _ { \mathrm { F } } ) ^ { 1 / 2 }$ . These matrices form the unique Frobenius-balanced point on the orbit, with $\left. \mathbfcal { A } _ { p } ^ { \star } \right. _ { \mathrm { F } } = \left. \mathbfcal { B } _ { p } ^ { \star } \right. _ { \mathrm { F } } .$ . The balancing map $( A _ { p } , B _ { p } ) \mapsto ( A _ { p } ^ { \dot { \star } } , \dot { B _ { p } ^ { \dot { \star } } } )$ is independent of the starting orbit representative, idempotent, and forward-preserving.

Balancing sends every point of an orbit to the same point, so two checkpoints that difer only by an orbit move end up with identical weights after balancing.

## 3.2 AdamW Trajectory Equivariance

The gradient normalization built into Adam cannot make parameter updates equivariant. With covariantly transformed moments, Adam’s normalized direction cancels the gradient scale when $\epsilon _ { i } = 0 ;$ a common positive $\epsilon _ { i }$ generally prevents exact cancellation. Even when this cancellation is exact, two runs that start on the same orbit do not stay equivalent. Adam then applies the same update to both runs, whereas staying on the same orbit requires each parameter to move in proportion to its own scale, that is, c times more for the enlarged block and c times less for the shrunken block.

AdamW setup. Let $S : = \mathrm { d i a g } ( s _ { i } ) , s _ { i } > 0 .$ , act on disjoint exact symmetry pairs as $( A , B ) \mapsto ( c A , c ^ { - 1 } B )$ and as the identity elsewhere. Every sampled data loss $\mathcal { L } _ { t }$ is assumed diferentiable at the iterates and satisfies $\mathcal { L } _ { t } ( S \pmb { \theta } ) = \mathcal { L } _ { t } ( \pmb { \theta } )$ Consider AdamW, with common $( \beta _ { 1 } , \beta _ { 2 } )$ and bias-correction counters. Let $m _ { t }$ and $v _ { t }$ denote its moving averages of the data gradients and their elementwise squares, respectively. These averages estimate the first moment and second raw moment of the gradient; hats denote bias-corrected estimates.

We refer to these estimates simply as moments. After updating the moments, the parameter update is

$$
\theta _ { t + 1 , i } : = ( 1 - \eta _ { t , i } \omega _ { t , i } ) \theta _ { t , i } - \eta _ { t , i } \frac { \hat { m } _ { t + 1 , i } } { \sqrt { \hat { v } _ { t + 1 , i } } + \epsilon _ { t , i } } .\tag{3}
$$

Here $\eta _ { t , i } > 0 , \omega _ { t , i } , \epsilon _ { t , i } \geq 0$ , all denominators are positive, and the gradients entering the moments are the data gradients, without clipping (Pascanu et al., 2013) or an additional parameter penalty. The following theorem gives the exact conditions under which AdamW, including its moments, keeps two runs on the same orbit.

Theorem 2 (AdamW trajectory equivariance). Under the AdamW setup above and the state transformation $( \pmb \theta ^ { \prime } , m ^ { \prime } , v ^ { \prime } ) : = ( S \pmb \theta , S ^ { - 1 } m , S ^ { - 2 } v )$ , the parameter-update formula commutes with S for every state with positive denominators $i f$ and only $i f ,$ at every step and coordinate,

$$
\begin{array} { r } { \eta _ { t , i } ^ { \prime } = s _ { i } \eta _ { t , i } , \qquad \epsilon _ { t , i } ^ { \prime } = s _ { i } ^ { - 1 } \epsilon _ { t , i } , \qquad \omega _ { t , i } ^ { \prime } = s _ { i } ^ { - 1 } \omega _ { t , i } . } \end{array}\tag{4}
$$

Consequently, with these rules and a shared sequence of sampled losses, orbit-related initial states remain orbit-related at every step.

In words, when a coordinate is multiplied by $s _ { i } ,$ its learning rate must be multiplied by $s _ { i }$ as well, while its ϵ and weight decay must be divided by $s _ { i }$ . A learning rate shared by the two runs violates this rule for every nontrivial move $S \neq I .$ The following example shows how large the resulting error can be.

Example 3 (Moment covariance alone is insuficient). If the two runs keep the same hyperparameters, the trajectory equivariance of Theorem 2 fails in general even when $\epsilon = 0$ and $\omega = 0$ . In particular, let $f ( a , b ) : =$ ab and $\mathcal { L } ( a , b ) : = ( a b ) ^ { 2 } / 2$ . One Adam step from zero moments, with common learning rate $\eta > 0$ , starting at (1, 1) and $( c , c ^ { - 1 } )$ , respectively, gives

$$
| f ( a _ { 1 } ^ { \prime } , b _ { 1 } ^ { \prime } ) - f ( a _ { 1 } , b _ { 1 } ) | = \eta ( c + c ^ { - 1 } - 2 ) .\tag{5}
$$

The primed loss after that step is unbounded as $c \to$ $\infty .$ , although both initial losses equal $1 / 2$ and their moments transform correctly.

Both runs move each weight by $\eta ,$ whereas equivalence would require the primed run to move $a ^ { \prime }$ by $c \eta$ and $b ^ { \prime }$ by $\eta / c \colon$ The step is too small for the enlarged weight and too large for the shrunken one. The failure arises even without moment lag, clipping, weight decay, or an ϵ floor. With c set by the depth schedule $c _ { \ell } = e ^ { \lambda \ell }$ of our deliberate rescaling tests (Sec. 5), the discrepancy is $2 \eta [ \cosh ( \lambda \ell ) - 1 ]$ and grows exponentially with the layer index ℓ.

## 3.3 Exact Canonical Restart

With shared hyperparameters, AdamW cannot keep two equivalent runs equivalent (Theorem 2). A restart avoids this problem: If two runs resume from the same numerical state, both runs follow the same trajectory. For the weights, balancing already provides this common state, because the balancing map sends every point of an orbit to the same point (Proposition 1); the map below extends this canonicalization to the moments.

Canonical state map. Let S and the disjoint exact pairs be as in Theorem 2. Assume each paired matrix has positive Frobenius norm. Let $R _ { \theta }$ multiply each A block by $\rho ^ { - 1 / 2 }$ and each B block by $\rho ^ { 1 / 2 }$ , where $\rho : =$ $\| A \| _ { \mathrm { F } } / \| B \| _ { \mathrm { F } }$ , and leave other coordinates unchanged. Define the state canonicalizer

$$
\mathcal { C } ( \pmb \theta , m , v ) : = ( R _ { \pmb \theta } \pmb \theta , R _ { \pmb \theta } ^ { - 1 } m , R _ { \pmb \theta } ^ { - 2 } v ) .\tag{6}
$$

Corollary 4 (Exact canonical restart). For the state canonicalizer C and exact-pair map S defined above, with positive norms on every pair,

$$
\mathcal { C } ( S \theta , S ^ { - 1 } m , S ^ { - 2 } v ) = \mathcal { C } ( \theta , m , v ) .\tag{7}
$$

Thus canonicalizing both orbit-related checkpoints and their moments before the first resumed update, and then using common hyperparameters, counters, and randomness, gives identical subsequent training trajectories. The same conclusion holds when both sets of moments are reset to zero before canonicalization.

In other words, if the weights and moments of both checkpoints are canonicalized before training resumes, the two runs are identical from the first resumed update onward. The restart needs no equivariant optimizer, because canonicalization removes the diference between the two runs before the optimizer sees either run. The checkpoint-resume experiments of Sec. 5.3 test a practical version of this idea: These experiments rebalance an edited branch periodically and compare its final PPL with the final PPL of a control.

## 3.4 Limits of Block-Local Updates

AdamW becomes trajectory-equivariant only after its hyperparameters are transformed coordinate by coordinate (Theorem 2). We now show that no standard first-order optimizer keeps two equivalent runs equivalent under a shared learning rate. The obstruction comes from block locality: Standard optimizers update each block using only the gradient history of that block. Write $\pmb \theta : = ( \{ ( A _ { p } , B _ { p } ) \} _ { p \in \mathcal { P } } , \pmb \xi )$ for the disjoint exact pairs together with the remaining parameters, and let $\mathcal { G } : = \{ S _ { c } : \pmb { c } \in ( 0 , \infty ) ^ { p } \}$ be the group of orbit moves $S _ { c } \pmb \theta : = ( \{ ( c _ { p } \pmb A _ { p } , c _ { p } ^ { - 1 } \pmb B _ { p } ) \} _ { p } , \pmb \xi )$ . For the exact pairs in

Table 1, every sampled loss satisfies $\mathcal { L } _ { t } \circ S = \mathcal { L } _ { t }$ for all $S \in { \mathcal { G } }$

A first-order method is a sequence of maps U<sub>t</sub> with $\pmb \theta _ { t + 1 } : = \pmb \theta _ { t } + \mathscr { U } _ { t } ( \pmb \theta _ { 1 : t } , \pmb \operatorname { \mathscr { g } } _ { 1 : t } )$ and $g _ { \tau } : = \nabla \mathcal { L } _ { \tau } ( \pmb \theta _ { \tau } )$ ; hyperparameters, step counters, and optimizer state with a fixed initialization are absorbed into $\mathcal { U } _ { t }$ . The method is orbit-equivariant if

$$
\mathcal { U } _ { t } ( S \pmb { \theta } _ { 1 : t } , S ^ { - 1 } g _ { 1 : t } ) = S \mathcal { U } _ { t } ( \pmb { \theta } _ { 1 : t } , g _ { 1 : t } ) ,\tag{8}
$$

for all $S \in { \mathcal { G } }$ and all $t ,$ wherever both sides are defined. Here $S ^ { - 1 } g _ { 1 : t }$ is exactly the gradient history produced at orbit-related iterates (Lemma 7). For a pair $p ,$ we write $x _ { A } : = g _ { 1 : t } ^ { A _ { p } }$ and $x _ { B } : = g _ { 1 : t } ^ { B _ { p } }$ for the gradient histories of its two blocks.

Theorem 5 (Block-local obstruction to orbit equivariance). Fix a pair $p$ and suppose that the increment $o f$ $A _ { p }$ has the form

$$
\begin{array} { r } { \Delta A _ { p , t } : = - \eta _ { t } \Psi _ { t } ( x _ { A } ) - \eta _ { t } \omega _ { t } A _ { p , t } , } \end{array}\tag{9}
$$

with $\eta _ { t } > 0 , \omega _ { t } \geq 0 \quad$ , and $\Psi _ { t } \textbf { } a$ function of the gradient history of the block $A _ { p }$ alone. If U is orbit-equivariant, then

$$
\Psi _ { t } ( c ^ { - 1 } x ) = c \Psi _ { t } ( x ) ,\tag{10}
$$

for all $c > 0$ and all histories x. Consequently $\Psi _ { t } \equiv 0$ whenever $\Psi _ { t }$ is positively homogeneous of some degree k $\neq - 1$ , or is bounded on a neighborhood of the zero history. The same holds for $B _ { p }$

By Eq. 10, an equivariant block-local rule must turn gradients that are c times smaller into an update that is c times larger, since the block itself is c times larger. Standard optimizers never meet this requirement: When the gradients get smaller, their updates get smaller or stay the same. Gradient descent and heavy-ball momentum have degree 1 and scale their updates with the gradients. Degree-zero rules ignore the gradient scale: AdaGrad, RMSProp, Adam, and AdamW with a zero stabilizer $\epsilon = 0 ;$ the sign-based rules signSGD and Lion; and the zero-stabilizer limit of Shampoo. Adam and AdamW with $\epsilon > 0$ keep their updates bounded as the gradients vanish. None of these optimizers is orbit-equivariant under a shared learning rate. Block locality and the homogeneity or boundedness assumption are both needed: A block-local rule of degree −1, or a rule that couples the two blocks of a pair, can be equivariant; Appendix C.7 gives these examples and the precise conditions for each optimizer above.

Relative-update scaling. Theorem 5 shows that equivalence breaks; the next result measures by how much, in terms of how far each block moves relative to its own size. The gradient covariance in Eq. 2, combined with homogeneity of the block-local direction, gives the following scaling law.

Corollary 6 (Relative-update scaling for homogeneous rules). Let both blocks of pair p follow Eq. 9 with $\omega _ { t } = 0$ and with $\Psi _ { t }$ positively homogeneous of degree k. Assume both current block norms are positive. For the orbit-related histories $\left( \pmb { \theta } _ { 1 : t } , g _ { 1 : t } \right)$ and $( S _ { c } \pmb { \theta } _ { 1 : t } , S _ { c } ^ { - 1 } g _ { 1 : t } )$ , where $S _ { c }$ acts on pair p with factor $^ { c , }$ the increments satisfy $\Delta A ^ { \prime } = c ^ { - k } \Delta A$ and $\Delta B ^ { \prime } = c ^ { k } \Delta B$ , so that

$$
{ \frac { \| \Delta { \cal A } ^ { \prime } \| _ { \mathrm { F } } } { \| { \cal A } ^ { \prime } \| _ { \mathrm { F } } } } = c ^ { - ( 1 + k ) } { \frac { \| \Delta { \cal A } \| _ { \mathrm { F } } } { \| { \cal A } \| _ { \mathrm { F } } } } ,
$$

$$
{ \frac { \left\| \Delta \boldsymbol { B } ^ { \prime } \right\| _ { \mathrm { F } } } { \left\| \boldsymbol { B } ^ { \prime } \right\| _ { \mathrm { F } } } } = c ^ { 1 + k } { \frac { \left\| \Delta \boldsymbol { B } \right\| _ { \mathrm { F } } } { \left\| \boldsymbol { B } \right\| _ { \mathrm { F } } } } .\tag{11}
$$

For gradient descent with $k = 1$ , the relative update of the enlarged block shrinks by $c ^ { 2 }$ and the relative update of the shrunken block grows by $c ^ { 2 } ;$ degree-zero rules such as Lion or Adam with $\epsilon = 0$ reduce both factors to c. Under the depth schedule $c _ { \ell } = e ^ { \lambda \ell }$ , the B-to-A log-ratio of relative updates shifts by $2 ( 1 + k ) \lambda \ell \colon$ 4λℓ for gradient descent and 2λℓ for degree-zero rules. Gradient normalization at best halves this shift, from 4λℓ to 2λℓ, and does not remove the shift. SAGGS instead removes this shift at each rebalance: Measured from the balanced point of Proposition 1, each rebalance resets c to 1 for every pair (Sec. 4).

Proofs for this section are in Appendix C.

## 4 SAGGS REBALANCING

No standard optimizer keeps two equivalent models equivalent under a shared learning rate (Theorem 5); thus, the position of a model on its orbit afects training. SAGGS leaves the optimizer unchanged and instead periodically moves the model along its orbit to the balanced point, where the two matrices of each exact pair have equal Frobenius norms (Proposition 1). The current function stays the same, and training continues from balanced pairs. Hard SAGGS makes this move at once (Sec. 4.1); its damped variant, Soft SAGGS, makes only part of the move at each rebalance so as not to disrupt optimizer state that has adapted to the imbalance (Sec. 4.2).

## 4.1 Hard SAGGS

Hard SAGGS applies the balancing map of Proposition 1 to every exact pair once every $T _ { \mathrm { b a l } }$ optimization steps (Algorithm 1). Each application shrinks the larger matrix of a pair and enlarges the smaller one by the same factor, so that both norms become the geometric mean of the two; in Figure 1(4), the rectangle becomes a square of the same area. Because each application is an exact orbit move, the forward map and the current data loss are unchanged at each rebalance.

Algorithm 1 Symmetry-Aware Rebalancing   
Require: Parameters θ, interval $T _ { \mathrm { b a l } }$ , disjoint exact   
symmetry pairs $( A _ { p } , B _ { p } ) , p \in \mathcal { P }$   
1: for each optimization step t do   
2: Perform standard optimizer update on $\pmb { \theta }$   
3: if t mod $T _ { \mathrm { b a l } } = 0$ then   
4: for each pair $p \in \mathcal P$ do   
5: if $\Vert A _ { p } \Vert _ { \mathrm { F } } = 0$ or $\Vert B _ { p } \Vert _ { \mathrm { F } } = 0$ then   
6: Skip this pair, since no positive norm ratio   
is defined   
7: else   
8: $\rho _ { p }  \| A _ { p } \| _ { \mathrm { F } } / \| B _ { p } \| _ { \mathrm { F } }$   
9: $A _ { p }  \rho _ { p } ^ { - 1 / 2 } A _ { p }$   
10: $B _ { p }  \rho _ { p } ^ { 1 / 2 } B _ { p }$   
11: end if   
12: end for   
13: end if   
14: end for

## 4.2 Soft SAGGS

Hard SAGGS moves each pair to its balanced point in a single step but leaves the optimizer state, which was built from gradients at the old scales, unchanged. Lion, for example, takes the sign of a combination of the current gradient and a momentum bufer, an exponential moving average (EMA) of past gradients (Chen et al., 2023). After a hard rebalance, the gradient of the shrunken block grows and the gradient of the enlarged block shrinks (Eq. 2), while the bufer keeps the old scales, so the sign of the combination, and hence the update, can change abruptly. In a Lion ablation without imposed rescaling, hard SAGGS was less consistent across seeds, whereas state-covariant SAGGS, which also rescales the bufer by the same factors as the gradients, removed most of this variation (Appendix F.5).

To avoid abrupt changes in the update without modifying the optimizer state, Soft SAGGS makes each correction smaller: At each rebalance, Soft SAGGS moves each pair only part of the way to its balanced point, so the gradient scales change in several smaller steps rather than in one jump. Concretely, Soft SAGGS keeps the schedule and the norm ratio $\rho _ { p }$ of Algorithm 1 but replaces its full correction with the damped rebalancing map

$$
\begin{array} { c } { { A _ { p }  c _ { p } ^ { \mathrm { e f f } } ( \alpha ) ^ { - 1 } A _ { p } , } } \\ { { B _ { p }  c _ { p } ^ { \mathrm { e f f } } ( \alpha ) B _ { p } , } } \end{array}\tag{12}
$$

where $c _ { p } ^ { \mathrm { e f f } } ( \alpha ) : = \rho _ { p } ^ { \alpha / 2 }$ and $\alpha \in ( 0 , 1 ]$ controls how much of the full correction is applied. Forward equivalence is preserved exactly for any α, because this map is an exact orbit move with factor $1 / c _ { p } ^ { \mathrm { e f f } }$ on a pair from Table 1. Setting α = 1 recovers hard SAGGS, and as $\alpha ~ \to ~ 0 ^ { + }$ the map approaches the identity. We refer to Eq. 12 with $\alpha < 1$ as Soft SAGGS, and we adopt $\alpha = 0 . 5$ as the default whenever a single value is required.

Soft and hard SAGGS. Each Soft SAGGS rebalance multiplies the log-norm ratio log $\rho _ { p }$ of a pair by $1 - \alpha ,$ , so repeated rebalances shrink the imbalance geometrically instead of removing the imbalance at once. For $\alpha < 1$ , however, the rebalanced weights still depend on where on the orbit the pair starts, and applying the map again moves the weights further; only hard SAGGS sends every point of an orbit to the same point, as the exact restart of Corollary 4 requires.

Appendix C.8 derives these properties, Appendix D.5 discusses optimizer-state handling and defines statecovariant SAGGS, Appendix B.1 gives the bias cotransforms for afine projections, Appendix D.3 provides a reference implementation of hard and Soft SAGGS, and Appendix D.6 covers computational cost, the interaction with weight decay, and numerical guards.

## 5 EXPERIMENTS

All points on an orbit compute the same function, yet the analysis of Sec. 3 predicts that the position of a model on its orbit afects training. The experiments below measure how much an imbalanced position degrades language-model training and how much of this degradation rebalancing recovers. We create such a position with an orbit edit, a deliberate rescaling of exact pairs, at initialization (Sec. 5.2) and in a checkpoint before training resumes (Sec. 5.3), and we also train without an edit to check whether rebalancing costs anything when no imbalance is imposed. Case studies on released checkpoints show imbalance arising in ordinary training and test exact canonical restart (Appendix I).

## 5.1 Experimental Setup and Statistics

Unless otherwise stated, we train a GPT-2-style decoder-only Transformer with pre-layer normalization (Pre-LN) on WikiText-103 using AdamW (Loshchilov and Hutter, 2019) with weight decay and gradient clipping. The primary 101M model uses ReLU. The 31M replication and the checkpoint experiments of Sec. 5.3 use a modern architecture with RMSNorm, RoPE, SwiGLU, and GQA, together with T5 SentencePiece tokenization.

The orbit edit rescales exact pairs of layer ℓ by $c _ { \ell } = e ^ { \lambda \ell }$ which leaves the function unchanged but imbalances deeper pairs more strongly. We compare training without an edit, at $\lambda = 0 ,$ with training after a deliberate edit at $\lambda \ : = \ : 0 . 2$ . The experiments index layers as $\ell = 0 , \ldots , L - 1$ , so the largest scale is $e ^ { \lambda ( L - 1 ) }$ , about 100 for a 24-layer model at $\lambda = 0 . 2$ . Vanilla uses the same optimizer and hyperparameters without rebalancing.

Table 2: 101M WikiText-103 comparison with $n =$ 7. Method columns report mean $\mathrm { P P L } ; \bar { \Delta }$ reports the mean ± sample SD of paired Vanilla-minus-SAGGS diferences. d is paired Cohen’s d and $p _ { \mathrm { s f } } ^ { + }$ is the onesided sign-flip p-value.
<table><tr><td> $\lambda$ </td><td>Vanilla</td><td>SAGGS</td><td> $\bar { \Delta }$ </td><td> $d$ </td><td> $p _ { \mathrm { s f } } ^ { + }$ </td></tr><tr><td>0.0</td><td>22.881</td><td>22.906</td><td>−0.025 ± 0.034</td><td>-0.75</td><td>0.9453</td></tr><tr><td>0.2</td><td>23.277</td><td>22.904</td><td>+0.374 ± 0.026</td><td>14.21</td><td>0.0078</td></tr></table>

We report the mean paired PPL diference $\bar { \Delta }$ over seven paired seeds. Unless stated otherwise, a diference is Vanilla minus the method, so a positive value favors the method, and ± denotes a sample standard deviation (SD). The one-sided sign-flip p-value $p _ { \mathrm { s f } } ^ { + }$ takes its smallest possible value, 0.0078, exactly when all seven diferences are positive.

## 5.2 Paired Language-Model Comparisons

## 5.2.1 101M WikiText-103 Comparison

In our main comparison, we train the 101M ReLU Transformer with $L = 2 4$ for 30K AdamW steps, starting either without an edit or after a Q–K edit at $\lambda = 0 . 2 .$ Hard SAGGS rebalances all exact pairs every 100 steps. At $\lambda = 0 . 2 .$ , SAGGS reduces PPL by $0 . 3 7 4 \pm 0 . 0 2 6$ across seven paired seeds, with $p _ { \mathrm { s f } } ^ { + } = 0 . 0 0 7 8 \ ( \mathrm { T a b l e \ 2 } )$ At $\lambda = 0$ , the diference is $- 0 . 0 2 5 \pm 0 . 0 3 4$ PPL. With SAGGS, the edit no longer matters: PPL is 22.904 with the edit and 22.906 without the edit. Without an edit, Vanilla ends with a Q–K log-norm ratio of only about 0.15, compared with about 3 after the edit, and removing this small imbalance barely changes PPL.

## 5.2.2 Architecture and Tokenizer Replication

To test whether these results depend on the GPT-2 architecture or the GPT-2 tokenizer, we repeat the comparison on a 31M model with the modern architecture and T5 SentencePiece tokenization. Over seven paired seeds, Soft SAGGS reduces PPL by 0.780±0.040 at $\lambda = 0 . 2$ , with $p _ { \mathrm { s f } } ^ { + } = 0 . 0 0 7 8$ , while the diference without an edit is $+ 0 . 0 1 9 \pm 0 . 0 2 4 ~ \mathrm { P P L }$ (Table 7). As in the 101M comparison, rebalancing helps after the edit and barely changes PPL without one, so its benefit is not specific to the GPT-2 architecture or the GPT-2 tokenizer.

Table 3: 123.8M checkpoint-surgery resume with $n = 7 .$ Values are final PPL, reported as mean ± sample SD across seeds. Paired contrasts appear in Table 11.
<table><tr><td>Resume condition</td><td>Final PPL</td></tr><tr><td>Control resume</td><td> $2 0 . 8 4 \pm 0 . 2 5$ </td></tr><tr><td>Surgery resume, no SAGGS</td><td> $2 4 . 6 2 \pm 0 . 5 8$ </td></tr><tr><td> $\mathrm { S u r g e r y + S A G G S }$ </td><td> $2 0 . 8 5 \pm 0 . 2 5$ </td></tr><tr><td> $\mathrm { S u r g e r y + S o f t ~ S A G G S }$ </td><td> $2 0 . 8 5 \pm 0 . 2 5$ </td></tr></table>

## 5.2.3 Lion Comparisons

Lion (Chen et al., 2023) uses only the sign of each update coordinate and therefore ignores the gradient scale, but, like Adam, Lion is still not orbit-equivariant (Theorem 5). We train the 101M ReLU model for 15K Lion steps with an all-pairs edit and seven paired seeds. We rebalance with Soft SAGGS, using $\alpha = 0 . 5$ every 100 steps, because a hard rebalance can abruptly change Lion’s updates (Sec. 4.2). At λ = 0.2, all seven seeds favor Soft SAGGS: ∆<sup>¯</sup> = +34.465 ± 25.170 PPL and $p _ { \mathrm { s f } } ^ { + } = 0 . 0 0 7 8$ (Table 9). At λ = 0, the diference i $\mathrm { ~ s ~ } - 0 . 0 5 5 \pm 0 . 2 3 2 \mathrm { ~ P P L }$ . Without rebalancing, the edit more than doubles the mean PPL of Lion, from 24.15 to 58.32, whereas the edited runs with Soft SAGGS reach 23.86.

## 5.3 Checkpoint Recovery and Diagnostics

Training often resumes from a checkpoint with a fresh optimizer, for example when the checkpoint is imported without its optimizer state. To measure how much an imbalanced checkpoint degrades training in this setting, we train the 123.8M model for 10K steps with each of seven seeds, reset its optimizer, and clone the checkpoint into four 5K-step continuations: control, edited, edited with SAGGS, and edited with Soft SAGGS. The three edited branches receive the same exact all-pairs edit at $\lambda = 0 . 2 .$ , and SAGGS and Soft SAGGS rebalance at the restart and every 50 steps thereafter.

At restart, the edited checkpoints are indistinguishable from the original by their loss: The edit and any first rebalance change the validation loss by less than $7 . 4 \times$ $1 0 ^ { - 5 }$ . Nevertheless, the edited run without rebalancing finishes $3 . 7 7 5 \pm 0 . 4 1 6$ PPL worse than the control, with $p _ { \mathrm { s f } } ^ { + } = 0 . 0 0 7 8$ (Table 11). Hard and Soft SAGGS finish within 0.02 PPL of the control (Table 3), so rebalancing recovers almost all of the degradation caused by the edit.

Instead of being reset, the AdamW moments can be kept and transformed along with the weights. On the 31M model, keeping and transforming the moments leaves a nonzero gap of $0 . 0 5 8 \pm 0 . 0 1 7$ PPL relative to a control that also keeps its optimizer state (Table 12), consistent with Theorem 2: Under a shared learning rate, transformed moments alone do not keep the two runs equivalent.

Case studies on public checkpoints (Appendix I) show that pair imbalance also arises in ordinary training: In Pythia 410M, the query and key norms drift apart during pre-training by a factor of up to about 6 in deep layers. The case studies also test canonical restart on GPT-2 774M, fine-tuned from the released checkpoint and from an exact Q–K edit of this checkpoint: Canonicalizing both checkpoints at restart makes the two runs identical, whereas without canonicalization the edited run predicts a diferent top-1 token than the canonicalized runs at 7.3% of held-out positions after 200 updates.

Appendix D gives the training configurations, hardware, and seed protocol, and Appendix E defines the paired statistics, states their test assumptions, and separates primary from supporting comparisons. Appendix F gives the full protocols, tables, and paired contrasts for Secs. 5.2 and 5.3, together with optimizer-state diagnostics for reset and stale moments. Appendix G verifies the predicted gradient-norm scaling, measures natural drift, and reports stress tests and replications across corpora, architectures, and model sizes; some of these experiments use SAGGS+P, which adds to SAGGS an auxiliary penalty on pair imbalance (Appendix D.1).

## 6 CONCLUSION

Even two Transformers that represent the same function can train diferently. Our analysis characterizes exact AdamW equivariance, identifies the obstruction for standard block-local updates, and establishes exact restart equivalence through canonicalization of weights and moments. SAGGS and Soft SAGGS provide simple function-preserving corrections for pair-scale imbalance. The normalization layers of a Transformer, weight decay, and gradient clipping do not restore pair balance (Appendix H). A rebalance rescales each pair only as much as the pair is imbalanced, so rebalancing can be kept on throughout training. Across paired languagemodel experiments and checkpoint-resume workflows, these corrections recover the performance lost to this imbalance, in particular after an exact orbit edit. These findings show that the pair-scale imbalance introduced by a parameterization strongly afects training, whether training starts from initialization or resumes from a checkpoint.

## AI USE STATEMENT

Generative AI tools were used as assistive tools for writing and editing, mathematical proofreading and consistency checking, and code development and debugging. Any AI-assisted outputs incorporated into this work were reviewed and validated by the authors as appropriate. The authors take full responsibility for the final content of this work, including any text, claims, or artifacts produced with the assistance of generative AI.

## References

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., and Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. In EMNLP, pages 4895– 4901.

Ainsworth, S. K., Hayase, J., and Srinivasa, S. S. (2023). Git Re-Basin: Merging Models modulo Permutation Symmetries. In ICLR.

Bernstein, J., Wang, Y., Azizzadenesheli, K., and Anandkumar, A. (2018). SIGNSGD: Compressed Optimisation for Non-Convex Problems. In ICML, volume 80, pages 559–568.

Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Pham, H., Dong, X., Luong, T., Hsieh, C., Lu, Y., and Le, Q. V. (2023). Symbolic Discovery of Optimization Algorithms. In NeurIPS.

Cohen, J. (2013). Statistical power analysis for the behavioral sciences.

Dinh, L., Pascanu, R., Bengio, S., and Bengio, Y. (2017). Sharp Minima Can Generalize For Deep Nets. In ICML, volume 70, pages 1019–1028.

Du, S. S., Hu, W., and Lee, J. D. (2018). Algorithmic Regularization in Learning Deep Homogeneous Models: Layers are Automatically Balanced. In NeurIPS, pages 382–393.

Entezari, R., Sedghi, H., Saukh, O., and Neyshabur, B. (2022). The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks. In ICLR.

Fedus, W., Zoph, B., and Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Eficient Sparsity. J. Mach. Learn. Res., 23:120:1–120:39.

Glorot, X. and Bengio, Y. (2010). Understanding the dificulty of training deep feedforward neural networks. In AISTATS, volume 9, pages 249–256.

Gokaslan, A. and Cohen, V. (2019). OpenWeb-Text Corpus. http://Skylion007.github.io/ OpenWebTextCorpus.

Gu, A. and Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. CoRR, abs/2312.00752.

Gupta, V., Koren, T., and Singer, Y. (2018). Shampoo: Preconditioned Stochastic Tensor Optimization. In ICML, volume 80, pages 1837–1845.

He, K., Zhang, X., Ren, S., and Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. In ICCV, pages 1026–1034.

Hendrycks, D. and Gimpel, K. (2016). Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units. CoRR, abs/1606.08415.

Henry, A., Dachapally, P. R., Pawar, S. S., and Chen, Y. (2020). Query-Key Normalization for Transformers. In EMNLP (Findings), volume EMNLP 2020, pages 4246–4253.

Huang, X. S., Pérez, F., Ba, J., and Volkovs, M. (2020). Improving Transformer Optimization Through Better Initialization. In ICML, volume 119, pages 4475– 4483.

Kingma, D. P. and Ba, J. (2015). Adam: A Method for Stochastic Optimization. In ICLR.

Kunin, D., Sagastuy-Breña, J., Ganguli, S., Yamins, D. L. K., and Tanaka, H. (2021). Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics. In ICLR.

Lakens, D. (2013). Calculating and reporting efect sizes to facilitate cumulative science: a practical primer for t-tests and ANOVAs. Frontiers in psychology, 4:863.

Liu, L., Liu, X., Gao, J., Chen, W., and Han, J. (2020). Understanding the Dificulty of Training Transformers. In EMNLP (1), pages 5747–5763.

Loshchilov, I. and Hutter, F. (2019). Decoupled Weight Decay Regularization. In ICLR.

Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2017). Pointer Sentinel Mixture Models. In ICLR.

Neyshabur, B., Salakhutdinov, R., and Srebro, N. (2015). Path-SGD: Path-Normalized Optimization in Deep Neural Networks. In NIPS, pages 2422–2430.

Pascanu, R., Mikolov, T., and Bengio, Y. (2013). On the dificulty of training recurrent neural networks. In ICML (3), volume 28, pages 1310–1318.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Salimans, T. and Kingma, D. P. (2016). Weight Normalization: A Simple Reparameterization to Accelerate

Training of Deep Neural Networks. In NIPS, page 901.

Shazeer, N. (2020). GLU Variants Improve Transformer. CoRR, abs/2002.05202.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q. V., Hinton, G. E., and Dean, J. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. In ICLR.

Stock, P., Graham, B., Gribonval, R., and Jégou, H. (2019). Equi-normalization of Neural Networks. In ICLR.

Su, J., Ahmed, M. H. M., Lu, Y., Pan, S., Bo, W., and Liu, Y. (2024). RoFormer: Enhanced transformer with Rotary Position Embedding. Neurocomputing, 568:127063.

Tanaka, H. and Kunin, D. (2021). Noether’s Learning Dynamics: Role of Symmetry Breaking in Neural Networks. In NeurIPS, pages 25646–25660.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. (2023). LLaMA: Open and Eficient Foundation Language Models. CoRR, abs/2302.13971.

Tran, H. V., Vo, T., The, A. N., Huu, T. T., Nguyen-Nhat, M., Tran, T., Pham, D., and Nguyen, T. M. (2025). Equivariant Neural Functional Networks for Transformers. In ICLR.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is All you Need. In NIPS, pages 5998–6008.

Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu, T. (2020). On Layer Normalization in the Transformer Architecture. In ICML, volume 119, pages 10524– 10533.

Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., and Gao, J. (2021). Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. In NeurIPS, pages 17084–17097.

Zhai, S., Likhomanenko, T., Littwin, E., Busbridge, D., Ramapuram, J., Zhang, Y., Gu, J., and Susskind, J. M. (2023). Stabilizing Transformer Training by Preventing Attention Entropy Collapse. In ICML, volume 202, pages 40770–40803.

Zhang, B. and Sennrich, R. (2019). Root Mean Square Layer Normalization. In NeurIPS, pages 12360– 12371.

Zhang, G., Wang, C., Xu, B., and Grosse, R. B. (2019). Three Mechanisms of Weight Decay Regularization. In ICLR.

Zhao, B., Dehmamy, N., Walters, R., and Yu, R. (2022). Symmetry Teleportation for Accelerated Optimization. In NeurIPS.

## CHECKLIST

1. For all models and algorithms presented, check if you include:

(a) A clear description of the mathematical setting, assumptions, algorithm, and/or model. [Yes] See Secs. 2, 3, and 4.

(b) An analysis of the properties and complexity (time, space, sample size) of any algorithm. [Yes] See Sec. 3 and Appendix D.6.

(c) (Optional) Anonymized source code, with specification of all dependencies, including external libraries. [Yes] See Appendix D.3.

2. For any theoretical claim, check if you include:

(a) Statements of the full set of assumptions of all theoretical results. [Yes] See Sec. 3 and Appendix C.

(b) Complete proofs of all theoretical results. [Yes] See Appendix C.

(c) Clear explanations of any assumptions. [Yes] See Secs. 2 and 3.

3. For all figures and tables that present empirical results, check if you include:

(a) The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). [Yes] See Appendix D.3.

(b) All the training details (e.g., data splits, hyperparameters, how they were chosen). [Yes] See Appendices D, F, G, and I.6.

(c) A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). [Yes] See Appendix E.2.

(d) A description of the computing infrastructure used. (e.g., type of GPUs, internal cluster, or cloud provider). [Yes] See Appendix D.2.

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:

(a) Citations of the creator If your work uses existing assets. [Yes] See Appendices D and I.6.

(b) The license information of the assets, if applicable. [No] Asset licenses are not listed.

(c) New assets either in the supplemental material or as a URL, if applicable. [Not Applicable] No new datasets or model checkpoints are released.

(d) Information about consent from data providers/curators. [Not Applicable] No new participant data are collected.

(e) Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. [No] Sensitive corpus content is not analyzed.

5. If you used crowdsourcing or conducted research with human subjects, check if you include:

(a) The full text of instructions given to participants and screenshots. [Not Applicable] No crowdsourcing or human-subject research.

(b) Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. [Not Applicable] No human-subject research.

(c) The estimated hourly wage paid to participants and the total amount spent on participant compensation. [Not Applicable] No human participants.

## Same Function, Diferent Trajectory: No Standard Optimizer Is Equivariant to Transformer Reparameterization Supplementary Materials

## APPENDIX TABLE OF CONTENTS

A Related Work 14   
B Notation and Architectural Details 15   
B.1 Attention, MLPs, and Afine Projections 15   
B.2 Grouped-Query Attention and Rotary Embeddings 15   
B.3 Interaction with Layer Normalization . 16   
B.4 Other Architectural Variants 16   
C Proofs and Optimizer Analysis 16   
C.1 Gradient Covariance from Forward Invariance 16   
C.2 Unique Balanced Representative 17   
C.3 Recovery of a Balanced Checkpoint . 17   
C.4 AdamW Moment Covariance at Corresponding Orbit Points . 18   
C.5 AdamW Equivariance and Canonical Restart 18   
C.6 Orbit-Equivariant First-Order Updates . 19   
C.7 Normalized Directions and the Scope of the Obstruction 20   
C.8 Soft Rebalancing: Contraction and Orbit Dependence 21   
C.9 Pair-Norm Dynamics . 21   
C.10 Derivations for Pair-Norm Dynamics 22   
D Experimental and Implementation Details 23   
D.1 Models and Training Protocols 23   
D.2 Seeds and Computing Infrastructure 24   
D.3 Implementation 24   
D.4 Model Configurations 25   
D.5 Optimizer-State Co-Transformation . 26   
D.6 Cost, Weight Decay, and Numerical Guards 26   
D.7 Code Availability and Reproducibility 27   
E Statistical Methodology and Summary of Results 27 27   
E.1 Paired Comparisons 27   
E.2 Statistics and Sign-Flip Tests 27   
E.3 Summary of Primary and Supporting Comparisons 28   
F Primary Comparisons and Optimizer-State Ablations 29   
F.1 101M WikiText-103 Comparisons 29   
F.2 Architecture and Tokenizer Replication . 29   
F.3 Hard Rebalancing with Lion 30   
F.4 Soft Rebalancing with Lion 31   
F.5 Optimizer-State Co-Transformation with Lion 31   
F.6 Branched Checkpoint-Surgery Resume 32   
F.7 Optimizer-State Diagnostics 33   
F.8 State-Preserved Checkpoint-Surgery Diagnostic 33   
F.9 Stale Optimizer-State Checkpoint Edit 34   
F.10 Partial Optimizer-State Checkpoint Edit . 34   
G Additional Experiments 35   
G.1 Gradient Law and Orbit Imbalance . 35   
G.2 Supporting Ablations and Replications 36   
G.3 FP32 Gradient-Law Verification . 36   
G.4 Naturally Occurring Pair Imbalance 37   
G.5 Pair Imbalance in a Pre-Trained Model 38   
G.6 Deep-Model Sensitivity to Rescaling 38   
G.7 Rescaling a 354M-Parameter Model 39   
G.8 SwiGLU Sensitivity to Rescaling 40   
G.9 1.2B-Parameter Gradient Law Verification 41   
G.10 Rescaling Strength at 101M Parameters 42   
G.11 Multi-Seed Comparisons Across Scales 42   
G.12 1.2B-Parameter Training . 43   
G.13 Exact Contraction and Synthetic Optimizer Diagnostics 44   
G.14 Cross-Corpus Replication on OpenWebText 45   
G.15 123.8M Modern-Architecture Scale Replication 45   
G.16 296.7M Modern-Architecture Scale Replication 46   
H Further Discussion 46   
H.1 Adaptive Normalization and Trajectory Equivariance . 47   
H.2 Sign-Based Optimizers and Soft Rebalancing 47   
H.3 Natural Drift and Weight Decay 47   
H.4 Implications for Initialization 47   
H.5 Training Stability . 48   
I Released-Checkpoint Case Studies 49   
I.1 Pair Scales in Public Checkpoints 49   
I.2 Fine-Tuning from Equivalent Checkpoints 50   
I.3 Depth-Resolved Evolution under Lion 51   
I.4 Token-Level Diferences after Restart . 52   
I.5 Summary 53   
I.6 Protocols 53   
J List of Notation 55   
K Limitations 55

## A RELATED WORK

Parameterization and initialization. Variance-preserving initialization (Glorot and Bengio, 2010; He et al., 2015) fixes a single scale at the start of training. The Maximal Update Parametrization (µP) of Yang et al. (2021) provides a principled framework for hyperparameter transfer across widths. T-Fixup (Huang et al., 2020) instead rescales residual branches so that deep Transformers can be trained more stably at a fixed width. Our work is complementary: µP governs how to scale across widths, and T-Fixup chooses a particular residual scale, while reparameterization symmetry reveals a continuous family of forward-equivalent configurations at fixed width and depth.

Weight normalization and reparameterization. Weight normalization (WeightNorm) (Salimans and Kingma, 2016) decouples the direction and magnitude of weight vectors, implicitly addressing some scale ambiguities. Dinh et al. (2017) explicitly derive the inverse gradient-coordinate law under function-preserving reciprocal rescaling. We use that law as background for the full-state optimizer analysis. Neyshabur et al. (2015) introduced path-regularization to account for rescaling invariances in ReLU networks. Transformer-NFN (Tran et al., 2025) characterizes more general matrix symmetries of multi-head attention; the scalar Q–K and V–O rescalings used here are special cases. Our focus is the response of first-order optimizer states and resumed trajectories to these edits. SAGGS instead selects the Frobenius-balanced representative of an exact scalar pair orbit. Standard WeightNorm has a trainable magnitude and does not impose this pairwise canonicalization.

Depth-dependent techniques. Pre-LN (Xiong et al., 2020) places the normalization before the attention and MLP sub-layers, improving gradient flow relative to post-layer normalization (Post-LN). Admin (Liu et al., 2020) diagnoses unbalanced residual-branch magnitudes in Transformers and rescales those magnitudes at initialization. σReparam (Zhai et al., 2023) constrains Q–K spectral scales to prevent attention entropy collapse, a failure mode adjacent to unconstrained motion along the Q–K orbit. Query-Key Normalization (Henry et al., 2020) removes that free Q–K scale by unit-normalizing the two factors. These techniques act on initialization, the residual stream, or spectral constraints, whereas our analysis concerns reciprocal rescalings inside attention and MLP blocks. The existence of an internal symmetry alone does not determine a training recipe’s stability or final PPL.

Weight balancing and equivariance. Du et al. (2018) proved conservation of diferences of squared layer norms under gradient flow in homogeneous networks and established discrete-time balancing results for matrix factorization. Conservation alone does not drive an arbitrary initial imbalance to zero. Equi-normalization (Stock et al., 2019) uses the same ReLU scale freedom to iteratively rebalance hidden units toward a minimum-norm representative; the procedure does not canonicalize first-order optimizer state. Symmetry Teleportation (Zhao et al., 2022) moves along a function-preserving orbit to change the local descent geometry. SAGGS uses a fixed orbit-selection criterion: the Frobenius-balanced representative of an exact scalar pair.

Symmetry and optimization dynamics. Kunin et al. (2021) and Tanaka and Kunin (2021) analyze how continuous network symmetries constrain learning dynamics and how finite-step and adaptive updates break associated conservation laws. Discrete permutation symmetries, whose role in linear mode connectivity was isolated by Entezari et al. (2022) and later used for model merging (Ainsworth et al., 2023), identify a diferent orbit: Those methods rearrange hidden units rather than rescaling a bilinear pair. Our focus is on complete discrete optimizer states and canonicalized restarts along the scalar rescaling orbit. The moment-scaling identities follow from the Adam recursions (Kingma and Ba, 2015). Theorem 2 and Corollary 4 give the corresponding discrete-time AdamW and restart rules, complementing the continuous-time analysis of Kunin et al. (2021).

## B NOTATION AND ARCHITECTURAL DETAILS

## B.1 Attention, MLPs, and Afine Projections

Consider an L-layer Transformer (Vaswani et al., 2017) with hidden dimension $d ,$ attention head dimension $d _ { k }$ and MLP intermediate dimension $d _ { \mathrm { f } }$ . At layer $\ell \in \{ 1 , \ldots , L \}$ , the self-attention sub-layer computes:

$$
\mathrm { A t t n } _ { \ell } ( X ) : = W _ { \ell } ^ { \cal O } \left[ \mathrm { s o f t m a x } \left( \frac { ( W _ { \ell } ^ { \cal Q } X ) ^ { \top } ( W _ { \ell } ^ { \cal K } X ) } { \sqrt { d _ { k } } } \right) ( W _ { \ell } ^ { \cal V } X ) ^ { \top } \right] ^ { \top } ,\tag{13}
$$

where $\ b { W _ { \ell } ^ { Q } } , \ b { W _ { \ell } ^ { K } } \in \mathbb { R } ^ { d _ { k } \times d } , \ \ b { W _ { \ell } ^ { V } } \in \mathbb { R } ^ { d _ { k } \times d } .$ , and $W _ { \ell } ^ { O } \in \mathbb { R } ^ { d \times d _ { k } }$ . The equation is for a single head; multi-head attention concatenates the per-head matrices.

The MLP sub-layer with ReLU activation is:

$$
\mathrm { M L P } _ { \ell } ( \pmb { x } ) : = \pmb { W } _ { \ell } ^ { 2 } \sigma ( \pmb { W } _ { \ell } ^ { 1 } \pmb { x } ) ,\tag{14}
$$

where $W _ { \ell } ^ { 1 } \in \mathbb { R } ^ { d _ { \mathrm { f f } } \times d } , W _ { \ell } ^ { 2 } \in \mathbb { R } ^ { d \times d _ { \mathrm { f f } } }$ , and σ denotes the activation function.

Let $\mathcal { L } ( \pmb \theta )$ denote the data loss evaluated at parameters $\pmb \theta : = \{ { \pmb W } _ { \ell } ^ { Q } , { \pmb W } _ { \ell } ^ { K } , { \pmb W } _ { \ell } ^ { V } , { \pmb W } _ { \ell } ^ { O } , { \pmb W } _ { \ell } ^ { 1 } , { \pmb W } _ { \ell } ^ { 2 } \} _ { \ell = 1 } ^ { L }$ . The equations omit biases.

For afine projections, scale the Q–K biases with their respective matrices, the V bias with V, and the ReLU up-projection or SwiGLU value bias with its up matrix; leave output biases and the SwiGLU gate unchanged. These co-transforms preserve the forward and gradient-coordinate identities. Without these co-transforms, matrix-only rescaling need not be a symmetry of a biased layer.

## B.2 Grouped-Query Attention and Rotary Embeddings

Modern variants. The pairs of Table 1 remain exact in the modern architecture that we use for the 31M replication and the checkpoint-resume experiments (Sec. 5). This architecture combines SwiGLU with GQA, RMSNorm, and RoPE. RMSNorm acts on the residual stream rather than on the paired matrices, and RoPE only rotates queries and keys, so the factors c and $c ^ { - 1 }$ still cancel. GQA shares each key and value head across a group of query heads, so the Q–K and V–O rescalings use one factor per group.

Grouped-query attention. GQA (Ainslie et al., 2023) shares one $W ^ { K }$ and one $W ^ { V }$ across $H / G$ query heads, where G is the group count. The Q–K orbit becomes group-tied: Any rescaling $W _ { \ell , h } ^ { Q } \mapsto c W _ { \ell , h } ^ { Q }$ for the $H / G$ query heads in a group must be paired with a single $W _ { \ell , g } ^ { K } \mapsto c ^ { - 1 } W _ { \ell , g } ^ { K }$ for the group’s shared key. The dimension of the orbit is reduced from H scalars per layer in the multi-head case to G scalars per layer; the gradient covariance in Eq. 2 applies to each group-pair independently. The per-pair Frobenius rebalance of SAGGS extends by treating each group’s $H / G \mathrm { ~ Q ~ }$ heads and shared K head as one balancing unit: The Q-side norm is $\begin{array} { r l } & { ( \sum _ { h \in g } \left. W _ { \ell , h } ^ { Q } \right. _ { \mathrm { F } } ^ { 2 } ) ^ { 1 / 2 } } \end{array}$ For $_ \mathrm { V - O }$ , scale the shared V head by c and all $H / G$ corresponding output-projection blocks by $c ^ { - 1 }$ , using their joint Frobenius norm on the output side.

Residual-stream normalization and rotary embeddings. RMSNorm (Zhang and Sennrich, 2019) prenormalization acts on the residual stream and therefore leaves the internal pair identities unchanged. RoPE (Su et al., 2024) applies a position-dependent linear orthogonal map to queries and keys, so $( c W ^ { Q } \mathbf { \check { x } } ) ^ { \top } ( c ^ { - 1 } W ^ { \dot { K } } \mathbf { x } )$

is invariant before and after the rotary map. The LLaMA-style block combining RMSNorm, RoPE, and SwiGLU (Touvron et al., 2023) with GQA (Ainslie et al., 2023) remains inside the catalog for the $\mathrm { Q \mathrm { - } K , \ V \mathrm { - } O _ { : } }$ and SwiGLU value–output pairs used by SAGGS.

## B.3 Interaction with Layer Normalization

Layer-normalization interaction. Pre-LN normalizes the input to each sub-layer, which might seem to eliminate the efect of weight norm imbalance. However, layer normalization acts on the residual stream, i.e., the sub-layer input, not on the internal weight matrices. The gradient transformation law operates on the internal bilinear structure, and layer normalization does not interact with this symmetry. Specifically, the layer-normalization output $\hat { \pmb x } : = \mathrm { L N } ( { \pmb x } )$ is a fixed function of the residual stream regardless of the internal parameterization of the sub-layer, so the reparameterization $( A _ { \ell } , B _ { \ell } ) \to ( c A _ { \ell } , c ^ { - 1 } B _ { \ell } )$ preserves $\mathrm { S u b L a y e r } _ { \ell } ( \hat { \pmb x } )$ and hence the gradient transformation law continues to hold.

To make the layer-normalization argument precise, write $f _ { A , B } ( X )$ for the sub-layer map with all other sub-layer parameters fixed. Every exact pair in Table 1 satisfies

$$
f _ { c A , c ^ { - 1 } B } ( { \boldsymbol { X } } ) = f _ { A , B } ( { \boldsymbol { X } } ) .
$$

The form $B g ( A X )$ describes a ReLU MLP, but does not represent a Q–K attention pair: Both Q and K enter the logits, with $( c \pmb { A } \pmb { X } ) ^ { \top } ( c ^ { - 1 } \pmb { B } \pmb { X } ) = ( \pmb { A } \pmb { X } ) ^ { \top } ( \pmb { B } \pmb { X } )$ . In Pre-LN, the residual-block map is $X + f _ { A , B } ( \mathrm { L N } ( X ) )$ in Post-LN, the residual-block map is $\mathrm { L N } ( X + f _ { A , B } ( X ) )$ ). Substituting the same sub-layer identity into either expression proves forward invariance of the whole block. Lemma 7 then gives the same inverse gradient-coordinate factors in both cases.

## B.4 Other Architectural Variants

The following extensions and boundaries complete the catalog in Sec. 2.1. Cross-attention, mixture-of-experts, and state-space blocks are not used in the primary training comparisons.

Cross-attention in encoder and decoder models. The attention pairs in Table 1 also apply when queries, keys, and values come from diferent sources: Their bilinear and linear compositions do not require the two input sequences to coincide.

Mixture-of-experts models. Each expert in a sparse mixture-of-experts MLP (Shazeer et al., 2017; Fedus et al., 2022) with ReLU experts contains an independent $( W _ { e } ^ { 1 } , W _ { e } ^ { 2 } )$ pair. Conditional routing means a given token is processed by only k experts, but the symmetry $\begin{array} { r } { \dot { \boldsymbol { W } } _ { e } ^ { 1 \prime } : = \boldsymbol { c } _ { e } \boldsymbol { W } _ { e } ^ { 1 } , \boldsymbol { W } _ { e } ^ { 2 \prime } : = \boldsymbol { c } _ { e } ^ { - 1 } \boldsymbol { W } _ { e } ^ { 2 } } \end{array}$ is per-expert and forward-equivalent regardless of the routing decision: The routing gate is shared across experts and unafected. The catalog adds E independent scalar orbits per layer, where E is the expert count. For SwiGLU experts, the exact pair is V–O with the gate fixed. GELU up–down rescaling is not an exact symmetry.

State-space models. Mamba’s selective state-space block (Gu and Dao, 2023) contains an input projection W<sup>in</sup> and an output projection $W ^ { \mathrm { o u t } }$ around a state-space recurrence. The input-dependent discretization step ∆ enters nonlinearly, so the recurrence is not generally homogeneous in its input. Reciprocal rescaling of the input and output projections need not preserve the function, and state-space blocks are excluded from Table 1. Any sub-block that has the form $B g ( A x )$ with g positively homogeneous inherits the corresponding pair. ReLU has this property; in SwiGLU, the same pair inheritance holds for the value path with the gate fixed, not for joint scaling of the gate and value projections.

## C PROOFS AND OPTIMIZER ANALYSIS

## C.1 Gradient Covariance from Forward Invariance

We record the general chain-rule identity underlying Eq. 2 and the optimizer-state proofs. The reciprocal-scaling instance is standard (Dinh et al., 2017, Theorem 3).

Lemma 7 (Forward invariance implies the gradient law). Let $F : \mathbb { R } ^ { n }  \mathbb { R }$ and let $J : = \mathrm { d i a g } ( s _ { 1 } , \ldots , s _ { n } )$ with all $s _ { i } > 0$ . Suppose that $F ( J \pmb { \theta } ) = F ( \pmb { \theta } )$ for every $\pmb { \theta } \in \mathbb { R } ^ { n }$ . Then F is diferentiable at θ if and only if F is diferentiable at Jθ, and at every such point

$$
\nabla F ( J \pmb \theta ) = J ^ { - 1 } \nabla F ( \pmb \theta ) , \qquad i . e . , \qquad \partial _ { i } F ( J \pmb \theta ) = s _ { i } ^ { - 1 } \partial _ { i } F ( \pmb \theta ) \quad f o r \ i = 1 , \dots , n .\tag{15}
$$

Proof. Let $G : = F \circ J , i . e . , G ( \pmb \theta ) = F ( J \pmb \theta )$ . By hypothesis, $G = F$ as functions on $\mathbb { R } ^ { n }$ . Since J is an invertible linear map, G is diferentiable at θ if and only if F is diferentiable at $J \theta ;$ for the converse write $F = G \circ J ^ { - 1 }$ In that case, the chain rule gives $\nabla G ( \pmb \theta ) = J ^ { \top } \nabla F ( J \pmb \theta ) = J \nabla F ( J \pmb \theta )$ . Because $G = F$ , G is diferentiable at θ exactly when $F { \mathrm { ~ i s } } ,$ with $\nabla G ( \pmb { \theta } ) = \nabla F ( \pmb { \theta } )$ . Combining the two statements, F is diferentiable at θ if and only if F is diferentiable at Jθ, and then $J \nabla F ( J \pmb { \theta } ) = \nabla F ( \pmb { \theta } )$ ; multiplying by $J ^ { - 1 }$ yields Eq. 15. □

For a symmetry pair, take $J = \mathrm { d i a g } ( c I _ { A } , c ^ { - 1 } I _ { B } )$ and leave all other coordinates unchanged. Forward invariance supplies the hypothesis, and the two blocks of Eq. 15 $\mathrm { g i v e ~ E q . ~ 2 }$ . This hypothesis holds for the data loss L but not for a loss that also includes penalties on the weights, such as L2 regularization, because rescaling a pair changes such penalties. Disjoint exact pairs can be transformed simultaneously by the same argument with a diagonal map containing their respective scales.

## C.2 Unique Balanced Representative

Proof of Proposition 1. Throughout, $\| A _ { p } \| _ { \mathrm { F } } > 0$ and $\| B _ { p } \| _ { \mathrm { F } } > 0$ , so that $\rho _ { p } : = \left. A _ { p } \right. _ { \mathrm { F } } / \left. B _ { p } \right. _ { \mathrm { F } }$ and $c _ { p } ^ { \star } : = \rho _ { p } ^ { 1 / 2 }$ are well defined, and $T _ { c } ( A , B ) : = ( c A , c ^ { - 1 } { \dot { B } } )$ denotes the orbit move with factor $c > 0$

$$
\begin{array} { r } { \big \| A _ { p } ^ { \star } \big \| _ { \mathrm { F } } = \big \| A _ { p } \big \| _ { \mathrm { F } } / c _ { p } ^ { \star } = \big \| A _ { p } \big \| _ { \mathrm { F } } ^ { 1 / 2 } \big \| B _ { p } \big \| _ { \mathrm { F } } ^ { 1 / 2 } \mathrm { ~ a n d ~ } \big \| B _ { p } ^ { \star } \big \| _ { \mathrm { F } } = c _ { p } ^ { \star } \| B _ { p } \| _ { \mathrm { F } } = \big \| A _ { p } \big \| _ { \mathrm { F } } ^ { 1 / 2 } \big \| B _ { p } \big \| _ { \mathrm { F } } ^ { 1 / 2 } , \mathrm { ~ s o ~ } \big \| A _ { p } ^ { \star } \big \| _ { \mathrm { F } } = c _ { p } ^ { \star } \big \| _ { \mathrm { F } } \big \| _ { \mathrm { F } } \big \| _ { \mathrm { F } } \big \| _ { \mathrm { F } } \big \| _ { \mathrm { F } } = c _ { p } ^ { \star } \big \| _ { \mathrm { F } } \big \| _ { \mathrm { F } } \big \| _ { \mathrm { F } } \big \| _ { \mathrm { F } } \big \| _ { \mathrm { F } } \big \| _ { \mathrm { F } } \big \| _ { \mathrm { F } } , } \end{array}
$$

$$
\left\| B _ { p } ^ { \star } \right\| _ { \mathrm { F } } .
$$

Uniqueness and independence of the orbit representative. The orbit of $\left( A _ { p } , B _ { p } \right)$ is $\mathcal { O } _ { p } : = \{ T _ { c } ( A _ { p } , B _ { p } )$ $c > 0 \}$ , and $c \mapsto T _ { c } ( A _ { p } , B _ { p } )$ is injective because $c A _ { p } = \tilde { c } A _ { p }$ with $\mathbf { A } _ { p } \neq \mathbf { 0 }$ forces $c = { \tilde { c } }$ . The point $T _ { c } ( A _ { p } , B _ { p } )$ is Frobenius-balanced if and only if $c \| \mathbfcal { A } _ { p } \| _ { \mathrm { F } } = c ^ { - 1 } \| \dot { B } _ { p } \| _ { \mathrm { F } } , \dot { \iota } . e .$ , if and only if $c ^ { 2 } = \| B _ { p } \| _ { \mathrm { F } } / \| A _ { p } \| _ { \mathrm { F } } ,$ , whose unique positive solution is $c = 1 / c _ { p } ^ { \star }$ . Hence $( { \cal A } _ { p } ^ { \star } , { \bar { \cal B } } _ { p } ^ { \star } ) = T _ { 1 / c _ { p } ^ { \star } } ( { \bar { \cal A } } _ { p } , { \cal B } _ { p } )$ is the unique balanced point of $\mathcal { O } _ { p }$ . The balanced point depends only on the orbit and not on the point from which the representative is computed: If $( \tilde { A } , \tilde { B } ) =$ $T _ { c } ( A _ { p } , B _ { p } )$ , then $\tilde { c } ^ { \star } = ( c \| A _ { p } \| _ { \mathrm { F } } / ( c ^ { - 1 } \| B _ { p } \| _ { \mathrm { F } } ) ) ^ { 1 / 2 } = c c _ { p } ^ { \star } , \mathrm { s o } \tilde { A } / \tilde { c } ^ { \star } = c A _ { p } / ( c c _ { p } ^ { \star } ) = A _ { p } ^ { \star }$ and $\tilde { c } ^ { \star } \tilde { B } = c c _ { p } ^ { \star } c ^ { - 1 } B _ { p } = B _ { p } ^ { \star }$ In particular, the step is idempotent: $\operatorname { A t } \ \left( A _ { p } ^ { \star } , B _ { p } ^ { \star } \right)$ one has $\rho _ { p } = 1$ and $c _ { p } ^ { \star } = 1$ , so a second application is the identity.

Forward pass. Since $( A _ { p } ^ { \star } , B _ { p } ^ { \star } ) \in \mathcal { O } _ { p }$ , the exact-pair assumption ensures that the step leaves the forward map unchanged. □

## C.3 Recovery of a Balanced Checkpoint

Recovery after an orbit edit of a balanced checkpoint. If the imbalance was created from a balanced pair by an exact orbit move $( A , B ) \mapsto ( c A , c ^ { - 1 } B )$ , this step applies the inverse orbit move and removes the associated diagonal coordinate factor. Suppose $( A _ { p } , B _ { p } ) = T _ { c } ( \bar { A } _ { p } ^ { 0 } , \bar { B } _ { p } ^ { 0 } )$ for some balanced pair with $\left. \mathbf { A } _ { p } ^ { 0 } \right. _ { \mathrm { F } } = \left. \mathbf { B } _ { p } ^ { 0 } \right. _ { \mathrm { F } }$ and some $c > 0$ . Then $c _ { p } ^ { \star } = ( c \left\| A _ { p } ^ { 0 } \right\| _ { \mathrm { F } } / ( c ^ { - 1 } \left\| B _ { p } ^ { 0 } \right\| _ { \mathrm { F } } ) ) ^ { 1 / 2 } = c ,$ so the rebalancing step is $T _ { 1 / c _ { n } ^ { \star } } = T _ { c ^ { - 1 } } = T _ { c } ^ { - 1 }$ and returns $( A _ { p } ^ { 0 } , B _ { p } ^ { 0 } )$ exactly. By Eq. 2, returning to the original balanced point also restores its gradient coordinates.

One-shot canonicalization. Algorithm 1 applies exactly $A _ { p }  \rho _ { p } ^ { - 1 / 2 } A _ { p }$ and $B _ { p }  \rho _ { p } ^ { 1 / 2 } B _ { p } ,$ , which is $T _ { 1 / c _ { v } ^ { \star } }$ By Proposition 1, a single application reaches the unique balanced representative of the orbit without changing the forward pass, and any further application is the identity; this fact is the one-shot canonicalizer property for pair-norm imbalance on the symmetry orbit.

Canonicalization preserves the current function and data loss. At a fixed checkpoint, every point of its exact symmetry orbit has the same validation PPL. Proposition 1 selects the balanced representative; subsequent training from that point is evaluated empirically.

## C.4 AdamW Moment Covariance at Corresponding Orbit Points

For a fixed sequence of corresponding orbit points, the Adam moment recursions determine the moments in the transformed coordinates. The following lemma supplies the identities used in Theorem $2 ;$ the lemma does not assume that common hyperparameters preserve the correspondence during training.

Corresponding-history setup. Let $g _ { t } \in \mathbb { R } ^ { n }$ denote the data-loss gradient at step t and let $\hat { m } _ { t } , \hat { v } _ { t }$ be the bias-corrected first- and second-moment estimates of AdamW with $\beta _ { 1 } \in [ 0 , 1 )$ and $\beta _ { 2 } ~ \in ~ ( 0 , 1 )$ . Under the reparameterization $\pmb { \theta } ^ { \prime } : = T _ { c } ( \pmb { \theta } )$ with diagonal Jacobian $J _ { c } : = \mathrm { d i a g } ( s _ { 1 } , \ldots , s _ { n } )$ , where $s _ { i } : = c$ on $A , s _ { i } : = c ^ { - 1 }$ on $B _ { ; }$ , and $s _ { i } : = 1$ elsewhere, let the primed quantities be computed from the gradients in the new coordinates at the corresponding orbit points $\pmb { \theta } _ { \tau } ^ { \prime } : = T _ { c } ( \pmb { \theta } _ { \tau } ) , \tau \leq t .$ , with the same hyperparameters and zero-initialized moment bufers.

Lemma 8 (Adam moment covariance). Under the corresponding-history setup above, the new-coordinate gradients satisfy $g _ { t , i } ^ { \prime } = s _ { i } ^ { - 1 } g _ { t , i }$ <sub>i</sub> for every i and every t, so $\hat { m } _ { t , i } ^ { \prime } = s _ { i } ^ { - 1 } \hat { m } _ { t , i }$ and $\hat { v } _ { t , i } ^ { \prime } = s _ { i } ^ { - 2 } \hat { v } _ { t , i }$

Proof of Lemma 8. With all parameters outside the pair held fixed, the data loss satisfies $\mathcal { L } \circ T _ { c } = \mathcal { L }$ by the exact-pair assumption, and $T _ { c }$ is the linear map $J _ { c } .$ . Lemma 7 therefore gives, at every pair of corresponding points, $g _ { \tau } ^ { \prime } = J _ { c } ^ { - 1 } g _ { \tau } , i . e . , g _ { \tau , i } ^ { \prime } = s _ { i } ^ { - 1 } g _ { \tau , i }$ for all $\tau \leq t$ and all i. The moment recursions $m _ { \tau } : = \beta _ { 1 } m _ { \tau - 1 } + ( 1 - \beta _ { 1 } ) g _ { \tau }$ and $v _ { \tau } : = \beta _ { 2 } v _ { \tau - 1 } + ( 1 - \beta _ { 2 } ) g _ { \tau } ^ { \odot 2 }$ with $m _ { 0 } : = v _ { 0 } : = 0$ are, coordinate-wise, linear in the sequence $( g _ { \tau , i } ) _ { \ i }$ <sub>τ</sub> and in $( g _ { \tau , i } ^ { 2 } ) _ { \tau }$ , respectively. By induction on $\tau \colon ~ m _ { 0 , i } ^ { \prime } = 0 = s _ { i } ^ { - 1 } m _ { 0 , i } ,$ and if $m _ { \tau - 1 , i } ^ { \prime } = s _ { i } ^ { - 1 } m _ { \tau - 1 , i }$ then $m _ { \tau , i } ^ { \prime } =$ $\beta _ { 1 } s _ { i } ^ { - 1 } m _ { \tau - 1 , i } + ( 1 - \beta _ { 1 } ) s _ { i } ^ { - 1 } g _ { \tau , i } = s _ { i } ^ { - 1 } m _ { \tau , i } ;$ likewise $v _ { \tau , i } ^ { \prime } = \beta _ { 2 } s _ { i } ^ { - 2 } v _ { \tau - 1 , i } + ( 1 - \beta _ { 2 } ) s _ { i } ^ { - 2 } g _ { \tau , i } ^ { 2 } = s _ { i } ^ { - 2 } v _ { \tau , i }$ . The bias corrections $\hat { m } _ { t } : = m _ { t } / ( 1 - \beta _ { 1 } ^ { t } )$ and $\hat { v } _ { t } : = v _ { t } / ( 1 - \beta _ { 2 } ^ { t } )$ are positive scalars that do not depend on the coordinate, so $\hat { m } _ { t , i } ^ { \prime } = s _ { i } ^ { - 1 } \hat { m } _ { t , i }$ and $\hat { v } _ { t , i } ^ { \prime } = s _ { i } ^ { - 2 } \hat { v } _ { t , i }$ . Decoupled weight decay acts separately on the parameters and does not enter the moments. □

## C.5 AdamW Equivariance and Canonical Restart

We prove the coordinate criterion, the functional counterexample, and the canonical restart identity separately.   
All calculations are in exact arithmetic.

Proof of Theorem 2. For the moment recursions and the coordinate criterion, suppose at step t that ${ \pmb \theta } _ { t } ^ { \prime } = S { \pmb \theta } _ { t } .$ $m _ { t } ^ { \prime } = S ^ { - 1 } m _ { t }$ , and $v _ { t } ^ { \prime } = S ^ { - 2 } v _ { t }$ . Since the sampled loss is the same in the two runs and $\mathcal { L } _ { t } \circ S = \mathcal { L } _ { t }$ , diferentiating this identity gives $S ^ { \top } \nabla { \mathcal { L } } _ { t } ( S \pmb { \theta } _ { t } ) = \nabla { \mathcal { L } } _ { t } ( \pmb { \theta } _ { t } )$ The diagonal matrix S is symmetric, so $g _ { t } ^ { \prime } \ = \ S ^ { - 1 } g _ { t }$ , where $g _ { t } : = \nabla \mathcal { L } _ { t } ( \theta _ { t } )$ . The Adam moment recursions therefore give

$$
\begin{array} { r l } & { m _ { t + 1 } ^ { \prime } = \beta _ { 1 } S ^ { - 1 } m _ { t } + ( 1 - \beta _ { 1 } ) S ^ { - 1 } g _ { t } = S ^ { - 1 } m _ { t + 1 } , } \\ & { ~ v _ { t + 1 } ^ { \prime } = \beta _ { 2 } S ^ { - 2 } v _ { t } + ( 1 - \beta _ { 2 } ) ( S ^ { - 1 } g _ { t } ) ^ { \odot 2 } = S ^ { - 2 } v _ { t + 1 } . } \end{array}
$$

Here $x ^ { \odot 2 }$ denotes componentwise squaring. Shared bias-correction counters multiply each moment by the same scalar in both runs, so the same identities hold for mˆ and vˆ. This argument is inductive: The induction assumes corresponding parameters only at the current step, not at future iterates.

Fix a coordinate and a step, suppress their indices, and put $s : = s _ { i } , a : = \hat { m } _ { t + 1 , i } , w : = \sqrt { \hat { v } _ { t + 1 , i } }$ . Because $s > 0 ,$ the primed adaptive term is $\eta ^ { \prime } a / ( w + s \epsilon ^ { \prime } )$ . Thus universal commutation of the parameter-update formula is precisely the identity

$$
( 1 - \eta ^ { \prime } \omega ^ { \prime } ) s \theta - \frac { \eta ^ { \prime } a } { w + s \epsilon ^ { \prime } } = s ( 1 - \eta \omega ) \theta - \frac { s \eta a } { w + \epsilon }\tag{16}
$$

for every $\theta , a$ and every positive $w ,$ and by continuity where $w = 0$ and the denominators remain positive. First set $a = 0$ and vary θ. This substitution yields $\eta ^ { \prime } \omega ^ { \prime } = \eta \omega$ . With these decay terms equal, take $a \neq 0$ and cancel a to obtain

$$
\eta ^ { \prime } ( w + \epsilon ) = s \eta ( w + s \epsilon ^ { \prime } ) \qquad \mathrm { f o r ~ e v e r y ~ } w > 0 .
$$

Equality of the coeficients of w gives $\eta ^ { \prime } = s \eta$ . Equality of the constant coeficients, together with $s \eta > 0$ , then gives $\epsilon ^ { \prime } = \epsilon / s$ . Substituting the learning-rate identity into the decay identity gives $\omega ^ { \prime } = \omega / s$ . This calculation proves necessity for the update formula on its full state domain; two particular trajectories may still agree by accident without these identities. Conversely these three identities make both sides of Eq. 16 equal, proving suficiency. Combined with the moment calculation above, suficiency completes an induction from the initia state and proves trajectory equivariance. The reasoning also permits a deterministic time-dependent schedule: The three identities must hold at each step of that schedule. □

Verification of Example 3. For $\mathcal { L } ( a , b ) : = ( a b ) ^ { 2 } / 2$ , the gradients are $g _ { a } = a b ^ { 2 }$ and $g _ { b } = a ^ { 2 } b$ . The gradients are $( 1 , 1 )$ at (1, 1) and $( c ^ { - 1 } , c )$ at $( c , c ^ { - 1 } )$ . With zero initial moments, the first bias-corrected moments are $\hat { m } _ { 1 } = g$ and $\hat { v } _ { 1 } = g ^ { \odot 2 }$ , for any $\beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ . Every gradient component is positive, so all four denominators are positive and the normalized updates with $\epsilon = 0$ are exactly 1. With no weight decay, this update rule gives

$$
\begin{array} { r l } & { ( a _ { 1 } , b _ { 1 } ) = ( 1 - \eta , 1 - \eta ) , } \\ & { ( a _ { 1 } ^ { \prime } , b _ { 1 } ^ { \prime } ) = ( c - \eta , c ^ { - 1 } - \eta ) . } \end{array}
$$

Hence

$$
\begin{array} { c } { { a _ { 1 } b _ { 1 } = 1 - 2 \eta + \eta ^ { 2 } , } } \\ { { a _ { 1 } ^ { \prime } b _ { 1 } ^ { \prime } = 1 - \eta ( c + c ^ { - 1 } ) + \eta ^ { 2 } , } } \\ { { a _ { 1 } ^ { \prime } b _ { 1 } ^ { \prime } - a _ { 1 } b _ { 1 } = - \eta ( c + c ^ { - 1 } - 2 ) . } } \end{array}
$$

The inequality $c + c ^ { - 1 } \geq 2$ proves Eq. 5. For fixed $\eta > 0 , a _ { 1 } ^ { \prime } b _ { 1 } ^ { \prime } \to - \infty { \mathrm { ~ a s ~ } } c \to \infty$ , so $\mathcal { L } ( a _ { 1 } ^ { \prime } , b _ { 1 } ^ { \prime } ) = ( a _ { 1 } ^ { \prime } b _ { 1 } ^ { \prime } ) ^ { 2 } / 2  \infty$ The initial product is 1 in both runs. The first and second moments satisfy exactly the relations in the proof of Theorem 2, so the discrepancy cannot be attributed to stale moments. Finally, $\scriptstyle c \ = \ e ^ { \lambda \ell }$ implies $c + c ^ { - 1 } - 2 = 2 [ cosh ( \lambda \ell ) - 1 ]$ , giving the stated depth-schedule consequence. □

Proof of Corollary $\it 4 .$ For one pair, the norm ratio at Sθ is $c ^ { 2 } \rho .$ Consequently its A-side balancing multiplier is $( c ^ { 2 } \rho ) ^ { - 1 / 2 } = c ^ { - 1 } \rho ^ { - 1 / 2 }$ , and its B-side multiplier is $( c ^ { 2 } \rho ) ^ { 1 / 2 } = c \rho ^ { \dot { 1 } / 2 }$ . The pairs are disjoint and all other coordinates are unchanged, so

$$
R _ { S \pmb { \theta } } S = R _ { \pmb { \theta } } .\tag{17}
$$

All matrices here are positive diagonal matrices and commute. These commutation relations imply that

$$
\begin{array} { c } { { R _ { S } \pmb { \theta } S \pmb { \theta } = R _ { \pmb { \theta } } \pmb { \theta } , } } \\ { { \qquad R _ { S } ^ { - 1 } S ^ { - 1 } m = R _ { \pmb { \theta } } ^ { - 1 } m , } } \\ { { \qquad R _ { S } ^ { - 2 } S ^ { - 2 } v = R _ { \pmb { \theta } } ^ { - 2 } v . } } \end{array}
$$

These three identities are exactly the three components of Eq. 7. The transformations are well-defined because both norms in every pair are positive. Each parameter balancing step is an exact symmetry and hence preserves the forward map.

After the two canonicalizations, the numerical parameters and moments are identical. With equal counters, common hyperparameters, and the same next minibatch and random choices, the next gradients, moments, and parameter updates are identical. Induction proves equality at every subsequent step. This implication also holds for common deterministic clipping applied after canonicalization, because the two gradients are then identical; clipping need not commute with an arbitrary S for this conclusion. If moments are reset, the same proof applies since $S ^ { - 1 } 0 = S ^ { - 2 } 0 = 0$ . No identity with an uncanonicalized reference or an arbitrary stale state is used or implied. □

## C.6 Orbit-Equivariant First-Order Updates

Throughout, $S _ { c }$ denotes the element of G that acts on the fixed pair p with factor $c > 0$ and as the identity on every other block; that is, $S _ { c }$ multiplies the $A _ { p } .$ -coordinates by c and the $B _ { p } .$ -coordinates by $c ^ { - 1 }$ , while $S _ { c } ^ { - 1 }$ multiplies those coordinates by $c ^ { - 1 }$ and $c ,$ respectively.

Proof of Theorem 5. For the degree −1 condition and the stated obstructions, fix $c > 0$ , a step t, and an admissible history $( \theta _ { 1 : t } , g _ { 1 : t } )$ , and write $x : = x _ { A } : = g _ { 1 : t } ^ { A _ { p } }$ . The $A _ { p } .$ -coordinates of $S _ { c } ^ { - 1 } g _ { 1 : t }$ form the history $c ^ { - 1 } x .$ , and the A<sub>p</sub>-block of $S _ { c } \pmb \theta _ { t }$ is $c A _ { p , t }$ . By Eq. 9, the $A _ { p } .$ -block of the left-hand side of Eq. 8 equals $- \eta _ { t } \Psi _ { t } ( c ^ { - 1 } x ) - \eta _ { t } \omega _ { t } c A _ { p , t } ,$ whereas the $A _ { p } .$ -block of the right-hand side equals $c ( - \eta _ { t } \Psi _ { t } ( x ) - \eta _ { t } \omega _ { t } \pmb { A } _ { p , t } )$ . Equating the two, canceling the decay terms, and dividing by $- \eta _ { t } < 0$ yields Eq. 10. The history x was arbitrary in the domain of $\Psi _ { t }$

Homogeneous case. Suppose $\Psi _ { t } ( s x ) = s ^ { k } \Psi _ { t } ( x )$ for all $s > 0$ with $k \neq - 1$ . Applying Eq. 10 with $c = 2$ and homogeneity with $s = 1 / 2$ gives $2 \Psi _ { t } ( x ) = \Psi _ { t } ( x / 2 ) = 2 ^ { - k } \Psi _ { t } ( x )$ , so $( 2 - 2 ^ { - k } ) \Psi _ { t } ( x ) = 0$ . Because $k \neq - 1$ we have $2 ^ { - k } \neq 2$ , hence $\Psi _ { t } ( x ) = 0$

Bounded case. Suppose $\| \Psi _ { t } ( y ) \| \leq M$ whenever $\| y \| < \delta .$ Given any $x ,$ choose $c > \| x \| / \delta ;$ then $\left\| c ^ { - 1 } x \right\| < \delta$ and Eq. 10 gives $\| \Psi _ { t } ( x ) \| = c ^ { - 1 } \left\| \Psi _ { t } ( c ^ { - 1 } x ) \right\| \le M / c$ . Letting $c \to \infty$ shows $\Psi _ { t } ( x ) = 0$

For the $B _ { p }$ -block, $S _ { c } ^ { - 1 }$ multiplies the gradient history by c and $S _ { c }$ multiplies the parameters by $c ^ { - 1 }$ , so the same computation yields $\bar { \Psi } _ { t } ^ { B } ( c x _ { B } ) \bar { = } c ^ { - 1 } \Psi _ { t } ^ { \bar { B } } ( x _ { B } )$ , which is Eq. 10 with c replaced by $c ^ { - 1 }$ ; the two consequences follow verbatim.

Coverage. Gradient descent, $\Psi _ { t } ( x ) : = x _ { t } ,$ , and heavy-ball momentum, $\begin{array} { r } { \Psi _ { t } ( x ) : = \sum _ { \tau < t } \beta ^ { t - \tau } x _ { \tau } } \end{array}$ , are homogeneous of degree 1. AdaGrad, RMSProp, and Adam with $\epsilon = 0$ divide a linear function of the history by the square root of a quadratic one and are homogeneous of degree 0, as are signSGD and Lion, since sgn $( s y ) = \operatorname { s g n } ( y )$ for $s > 0$ For Shampoo, put $\begin{array} { r } { L _ { t } : = \sum _ { \tau < t } x _ { \tau } \bar { x } _ { \tau } ^ { \top } } \end{array}$ and $\begin{array} { r } { R _ { t } : = \dot { \sum } _ { \tau < t } x _ { \tau } ^ { \top } x _ { \tau } } \end{array}$ . These Gram factors can be singular, including at the first step for a rectangular gradient. The zero-stabilizer limiting direction is

$$
( L _ { t } ^ { \dag } ) ^ { 1 / 4 } x _ { t } ( R _ { t } ^ { \dag } ) ^ { 1 / 4 } ,
$$

where † denotes the Moore–Penrose pseudoinverse; equivalently, the inverse fourth root is zero on each nullspace. This pseudoinverse formula agrees with $L _ { t } ^ { - 1 / 4 } x _ { t } R _ { t } ^ { - 1 / 4 }$ when both factors are positive definite. This pseudoinverse direction is the limit of the regularized direction because ker $L _ { t } \subseteq \ker x _ { t } ^ { \top }$ and ker $R _ { t } \subseteq \ker x _ { t } \colon$ No component of $x _ { t }$ lies in the divergent nullspace factors. Scaling the history by $s > 0$ scales each Gram factor by $s ^ { 2 }$ , so the limiting direction scales as $s ^ { - 1 / 2 } \cdot s \cdot s ^ { - 1 / 2 } = s ^ { 0 }$ . For Adam or AdamW with $\epsilon > 0$ , the bias-corrected first moment is a convex combination of the past gradients of the coordinate, so $\begin{array} { r } { \left| \Psi _ { t , i } ( x ) \right| \leq \operatorname* { m a x } _ { \tau \leq t } \left| x _ { \tau , i } \right| / \epsilon } \end{array}$ , which tends to 0 with the history; thus $\Psi _ { t }$ is bounded near the zero history. Decoupled weight decay is already included in Eq. 9, and Lion’s decay term $\eta \omega \pmb \theta$ has the same form. None of these rules is identically zero, so none is orbit-equivariant.

Proof of Corollary 6. For the orbit-related histories, the $A _ { p } .$ -history is $c ^ { - 1 } x _ { A }$ and the $B _ { p } .$ -history is $c x _ { B }$ . By homogeneity, $\Delta \dot { A } ^ { \prime } = - \eta _ { t } \Psi _ { t } ( c ^ { - 1 } x _ { A } ) = c ^ { - k } ( - \eta _ { t } \Psi _ { t } ( x _ { A } ) ) = \dot { c } ^ { - k } \Delta A$ and $\Delta B ^ { \prime } = - \eta _ { t } \Psi _ { t } ^ { B } ( { \dot { c } } x _ { B } ) = c ^ { k } \Delta B$ . Since $\| A ^ { \prime } \| _ { \mathrm { F } } = c \| A \| _ { \mathrm { F } }$ and $\left\| \boldsymbol B ^ { \prime } \right\| _ { \mathrm { F } } = c ^ { - 1 } \left\| \boldsymbol B \right\| _ { \mathrm { F } } ,$ Eq. 11 follows. □

Depth-schedule specialization. For the nonzero increments in Corollary 6, taking logarithms of the ratio of the two nonzero relative increments gives a shift of log $c ^ { 1 + k } - \log c ^ { - ( 1 + k ) } = \bar { 2 } ( 1 + k )$ log c, which equals $2 ( 1 + k ) \lambda \ell$ for $c = e ^ { \lambda \ell } ;$ ; the same diference is 4λℓ for $k = 1$ and 2λℓ for $k = 0$ . For $\lambda \ell \neq 0$ , the shift is zero only for $k = - 1 ;$ an identity orbit move gives zero for every k.

Full increments with decoupled decay. When $\omega _ { t } \neq 0$ , the corresponding full increments instead satisfy

$$
\begin{array} { r l } & { \Delta A ^ { \prime } = - \eta _ { t } c ^ { - k } \Psi _ { t } ( x _ { A } ) - \eta _ { t } \omega _ { t } c A , } \\ & { \Delta B ^ { \prime } = - \eta _ { t } c ^ { k } \Psi _ { t } ^ { B } ( x _ { B } ) - \eta _ { t } \omega _ { t } c ^ { - 1 } B . } \end{array}
$$

The data and decay terms carry diferent powers of c for $k \neq - 1$ , so their sum has no single scaling exponent in general. For example, at the scalar pair $a = b = 1$ with loss $( a b ) ^ { 2 } / 2$ , a first zero-ϵ AdamW step has data directions $u _ { a } = u _ { b } = 1$ . With $c = 2 , \omega = 1$ , and a common $\eta > 0$ , the multipliers of the full relative increments are $3 / 4$ on A and $3 / 2$ on $B _ { ; }$ , instead of the data-only values $1 / 2$ and 2. The B-to-A multiplier is 2, whereas the data-only law gives $c ^ { 2 } = 4 .$

## C.7 Normalized Directions and the Scope of the Obstruction

Adam normalized direction. For corresponding gradient histories under $\theta _ { i } ^ { \prime } : = s _ { i } \theta _ { i } , s _ { i } > 0$ , and covariantly transformed initial moments, including zero initialization, Adam’s moment recursions give $\hat { m } _ { i } ^ { \prime } = \hat { m } _ { i } / s _ { i }$ and

$\hat { v } _ { i } ^ { \prime } = \hat { v } _ { i } / s _ { i } ^ { 2 }$ . With a common additive stabilizer $\epsilon _ { i } ,$ its normalized data direction is

$$
\begin{array} { c } { { u _ { i } ^ { \prime } : = \displaystyle \frac { \hat { m } _ { i } / s _ { i } } { \sqrt { \hat { v } _ { i } } / s _ { i } + \epsilon _ { i } } } } \\ { { = \displaystyle \frac { \hat { m } _ { i } } { \sqrt { \hat { v } _ { i } } + s _ { i } \epsilon _ { i } } , } } \\ { { u _ { i } : = \displaystyle \frac { \hat { m } _ { i } } { \sqrt { \hat { v } _ { i } } + \epsilon _ { i } } . } } \end{array}\tag{18}
$$

Thus the gradient-scale cancellation (Kingma and Ba, 2015) is exact when $\epsilon _ { i } = 0$ and the denominators are positive; the cancellation generally fails for a common positive $\epsilon _ { i } .$ Even when $u _ { i } ^ { \prime } = u _ { i }$ , preserving an orbit-related training trajectory requires the parameter displacement to scale with the parameters.

Block-locality examples. Block locality and the stated regularity alternatives are essential. For example, the unbounded block-local rule that sets $\dot { \Psi } ( x ) : = x _ { t } / \left\| x _ { t } \right\| _ { \mathrm { F } } ^ { 2 }$ for $x _ { t } \neq 0$ and $\Psi ( x ) : = 0$ at $x _ { t } = 0$ has degree −1 and satisfies the required scaling. Moreover, the parameter-free, pair-coupled directions Ψ $_ A : = \left\| g _ { t } ^ { B } \right\| _ { \mathrm { F } } ^ { 2 } g _ { t } ^ { A }$ and $\Psi _ { B } : = \left\| g _ { t } ^ { A } \right\| _ { \mathrm { F } } ^ { 2 } g _ { t } ^ { B }$ are polynomial and equivariant: These directions transform by c and $c ^ { - 1 }$ , respectively. Thus the obstruction does not cover all parameter-free first-order methods.

Standard optimizer instances. Theorem 5 applies to gradient descent with or without heavy-ball momentum, both homogeneous of degree 1. The degree-zero class includes AdaGrad and RMSProp with zero additive stabilizers, Adam with $\epsilon = 0$ , signSGD (Bernstein et al., 2018), and Lion. This class also includes per-tensor preconditioned updates such as the zero-stabilizer limit of Shampoo (Gupta et al., 2018), with inverse roots taken on the ranges of the Gram factors. Adam and AdamW with $\epsilon > 0$ are bounded near the zero history and also fal within the theorem’s scope. These rules are not orbit-equivariant under a common learning rate, with or without decoupled weight decay.

## C.8 Soft Rebalancing: Contraction and Orbit Dependence

Contraction of the log-ratio. The residual log-imbalance after a single damped event contracts geometrically:

$$
\log \rho _ { p } ^ { \mathrm { p o s t } } : = \log \frac { \left\| A _ { p } ^ { \mathrm { n e w } } \right\| _ { \mathrm { F } } } { \left\| B _ { p } ^ { \mathrm { n e w } } \right\| _ { \mathrm { F } } } = ( 1 - \alpha ) \log \rho _ { p } ,\tag{19}
$$

so K events with no intervening optimizer updates multiply the log-imbalance by $( 1 - \alpha ) ^ { K }$ . During training, optimizer updates can introduce new imbalance between events.

Dependence on the starting representative. The soft map is not itself an orbit canonicalizer for $\alpha < 1$ Writing Φ<sub>α</sub> for the damped rebalancing map of Eq. 12 and $T _ { c } ( { \cal A } , { \cal B } ) : = ( c { \cal A } , c ^ { - 1 } { \cal B } )$ , direct substitution of $\rho ( T _ { c } ( A , B ) ) = c ^ { 2 } \rho ( A , B )$ gives

$$
\Phi _ { \alpha } ( T _ { c } ( { \cal A } , { \cal B } ) ) = T _ { c ^ { 1 - \alpha } } ( \Phi _ { \alpha } ( { \cal A } , { \cal B } ) ) .\tag{20}
$$

Thus one soft event retains dependence on the starting representative; away from balance the soft map is also not idempotent, since a second event changes $( 1 - \alpha )$ log ρ to $( 1 - \alpha ) ^ { 2 } \log \rho$ . Hard rebalancing with $\alpha = 1$ has both canonicalizer properties. Corollary 4 uses that hard map. The seven-seed Lion comparison in Sec. 5.2 evaluates the damped variant on Lion training.

## C.9 Pair-Norm Dynamics

For one matrix pair, let $( U _ { t } ^ { A } , U _ { t } ^ { B } )$ denote the optimizer direction: the data gradient for gradient descent, the normalized moment direction for AdamW, or a sign vector for Lion. With $\eta _ { t } > 0$ and $0 \leq \eta _ { t } \omega _ { t } < 1$ , write

$$
\begin{array} { r } { A _ { t + 1 } : = ( 1 - \eta _ { t } \omega _ { t } ) A _ { t } - \eta _ { t } U _ { t } ^ { A } , \qquad B _ { t + 1 } : = ( 1 - \eta _ { t } \omega _ { t } ) B _ { t } - \eta _ { t } U _ { t } ^ { B } . } \end{array}\tag{21}
$$

Let $\Delta _ { t } : = \left\| A _ { t } \right\| _ { \mathrm { F } } ^ { 2 } - \left\| B _ { t } \right\| _ { \mathrm { F } } ^ { 2 }$ be the balance defect, and let $r _ { t } : = \log ( \left. A _ { t } \right. _ { \mathrm { F } } / \left. B _ { t } \right. _ { \mathrm { F } } )$ be the log-ratio underlying the experimental diagnostics wherever both norms are positive. The numerical stabilizer used for measurements is specified in Appendix G.4. For the optimizer direction in Eq. 21, define the radial forcing and squared-direction norm diference by $F _ { t } : = \langle \boldsymbol { A } _ { t } , \boldsymbol { U } _ { t } ^ { A } \rangle _ { \mathrm { F } } - \langle \boldsymbol { B } _ { t } , \boldsymbol { U } _ { t } ^ { B } \rangle _ { \mathrm { F } }$ and $Q _ { t } : = \mathsf { \bar { \| } } U _ { t } ^ { A } \big \| _ { \mathrm F } ^ { 2 } - \big \| U _ { t } ^ { B } \big \| _ { \mathrm F } ^ { 2 }$

Exact defect recursion. For any direction $( U _ { t } ^ { A } , U _ { t } ^ { B } )$ in Eq. 21, expanding the squared norms gives

$$
\Delta _ { t + 1 } = ( 1 - \eta _ { t } \omega _ { t } ) ^ { 2 } \Delta _ { t } - 2 \eta _ { t } ( 1 - \eta _ { t } \omega _ { t } ) F _ { t } + \eta _ { t } ^ { 2 } Q _ { t } .\tag{22}
$$

This algebraic recursion does not require a symmetry assumption.

From the balance defect to the log-ratio. For positive norms, decay alone leaves $r _ { t }$ unchanged. The balance defect bounds the log-ratio as $| r _ { t } | \leq \left| \Delta _ { t } \right| / ( 2 \operatorname* { m i n } \{ \left\| \mathbf { \bar { A } } _ { t } \right\| _ { \mathrm { F } } ^ { 2 } , \left\| B _ { t } \right\| _ { \mathrm { F } } ^ { 2 } \} )$ ). Appendix C.10 derives this bound and the exact recursion.

Radial parameter and update components. Balancing equalizes the weight norms of a pair, not the norms of its data gradients. Even at Frobenius balance, whether at initialization or right after a SAGGS rebalance, the matrices $\pmb { A } _ { \ell }$ and $\mathbf { \delta } _ { B _ { \ell } }$ in a symmetry pair can receive gradients with diferent magnitudes due to their positions in the computation graph. For instance, $W _ { \ell } ^ { Q }$ receives gradients that depend on $W _ { \ell } ^ { K }$ through the attention logit Jacobian, and vice versa. Gradient magnitudes alone do not determine norm drift. Under Eq. 21, the expansion at a fixed iterate, with fixed directions and decay coeficient, is

$$
r _ { t + 1 } - r _ { t } = - \eta _ { t } \left( \frac { \langle A _ { t } , U _ { t } ^ { A } \rangle _ { \mathrm { F } } } { \left. A _ { t } \right. _ { \mathrm { F } } ^ { 2 } } - \frac { \langle B _ { t } , U _ { t } ^ { B } \rangle _ { \mathrm { F } } } { \left. B _ { t } \right. _ { \mathrm { F } } ^ { 2 } } \right) + O ( \eta _ { t } ^ { 2 } ) .\tag{23}
$$

The common decay contribution cancels from this first-order log-ratio increment.

Appendix C.10 gives the exact logarithmic recursion. At Frobenius balance, with both squared norms equal to $N > 0$ , the first-order term is $- \eta _ { t } F _ { t } / N$ . The corresponding exact balance-defect recursion is Eq. 22. Interpreting a measured drift requires the radial components of the actual optimizer directions.

Common weight decay preserves the pair ratio. AdamW applies weight decay as $\theta  ( 1 - \eta \omega ) \theta$ uniformly to all parameters. This uniform decay multiplies both matrices of a pair by the same factor, so a decay step leaves the pair ratio unchanged and cannot restore balance by itself. The exact one-step defect recursion in Eq. 22 records the decay factor and data-update terms. The log-ratio increment depends on the radial components of the actual optimizer directions, as in Eq. 23.

## C.10 Derivations for Pair-Norm Dynamics

All inner products and norms are Frobenius; we suppress the step index when no confusion can arise.

Derivation of the exact defect recursion. Expanding the squared norms of Eq. 21,

$$
\begin{array} { r l } & { \| A _ { t + 1 } \| _ { \mathrm { F } } ^ { 2 } = ( 1 - \eta _ { t } \omega _ { t } ) ^ { 2 } \left\| A _ { t } \right\| _ { \mathrm { F } } ^ { 2 } - 2 \eta _ { t } ( 1 - \eta _ { t } \omega _ { t } ) \langle A _ { t } , U _ { t } ^ { A } \rangle + \eta _ { t } ^ { 2 } \left\| U _ { t } ^ { A } \right\| _ { \mathrm { F } } ^ { 2 } , } \\ & { \| B _ { t + 1 } \| _ { \mathrm { F } } ^ { 2 } = ( 1 - \eta _ { t } \omega _ { t } ) ^ { 2 } \left\| B _ { t } \right\| _ { \mathrm { F } } ^ { 2 } - 2 \eta _ { t } ( 1 - \eta _ { t } \omega _ { t } ) \langle B _ { t } , U _ { t } ^ { B } \rangle + \eta _ { t } ^ { 2 } \left\| U _ { t } ^ { B } \right\| _ { \mathrm { F } } ^ { 2 } , } \end{array}
$$

and subtracting gives Eq. 22.

Decay and log-ratio diagnostics. For the positive block norms of Appendix C.9, multiplying both matrices by the same positive scalar $1 - \eta _ { t } \omega _ { t }$ leaves the ratio of their norms, hence $r _ { t } ,$ unchanged. For the bound on $r _ { t } ,$ put $x : = \bar { \Delta } _ { t } / \left\| \boldsymbol B _ { t } \right\| _ { \mathrm F } ^ { 2 } > - 1$ , so that $r _ { t } = ( 1 / 2 ) \log ( 1 + x )$ . If $x \geq 0$ then $\left\| B _ { t } \right\| _ { \mathrm { F } } \leq \left\| A _ { t } \right\| _ { \mathrm { F } }$ and $\log ( 1 + x ) \leq$ $x = \left| \Delta _ { t } \right| / \operatorname* { m i n } \{ \left\| A _ { t } \right\| _ { \mathrm { F } } ^ { 2 } , \left\| B _ { t } \right\| _ { \mathrm { F } } ^ { 2 } \}$ . I $\dot { \mathbf { \eta } } - 1 < x < 0$ then $\| A _ { t } \| _ { \mathrm { F } } < \| B _ { t } \| _ { \mathrm { F } }$ and $- \log ( 1 + x ) = \log ( 1 / ( 1 + x ) ) \leq$ $1 / ( 1 + x ) - 1 = - x / ( 1 + x ) = | \Delta t | / ( \| B _ { t } \| _ { \mathrm { F } } ^ { 2 } + \Delta _ { t } ) = | \Delta _ { t } | / \| A _ { t } \| _ { \mathrm { F } } ^ { 2 } ,$ , using log $u \leq u - 1$ . In both cases, $\left. r _ { t } \right. \leq \left. \Delta _ { t } \right. / ( 2 \operatorname* { m i n } \{ \left. A _ { t } \right. _ { \mathrm { F } } ^ { 2 } , \left. B _ { t } \right. _ { \mathrm { F } } ^ { 2 } \} )$

Exact log-ratio increments. For positive block norms before and after the update, define $\nu _ { t } ^ { A } : =$ $\langle { \pmb A } _ { t } , { \pmb U } _ { t } ^ { A } \rangle _ { \mathrm { F } } / \left\| { \pmb A } _ { t } \right\| _ { \mathrm { F } } ^ { 2 }$ and $\chi _ { t } ^ { A } : = \left\| U _ { t } ^ { A } \right\| _ { \mathrm { F } } ^ { 2 } / \left\| A _ { t } \right\| _ { \mathrm { F } } ^ { 2 }$ , with the analogous definitions for $B ,$ and set $\kappa _ { t } : = 1 - \eta _ { t } \omega _ { t }$ Dividing the squared-norm expansions above by the respective old squared norms gives

$$
r _ { t + 1 } - r _ { t } = \frac { 1 } { 2 } \log \frac { \kappa _ { t } ^ { 2 } - 2 \eta _ { t } \kappa _ { t } \nu _ { t } ^ { A } + \eta _ { t } ^ { 2 } \chi _ { t } ^ { A } } { \kappa _ { t } ^ { 2 } - 2 \eta _ { t } \kappa _ { t } \nu _ { t } ^ { B } + \eta _ { t } ^ { 2 } \chi _ { t } ^ { B } } .\tag{24}
$$

At a fixed iterate with fixed directions and decay coeficient, the two arguments have expansions $1 - 2 \eta _ { t } ( \omega _ { t } +$ $\nu _ { t } ^ { A } ) + O ( \eta _ { t } ^ { 2 } )$ and $1 - 2 \eta _ { t } ( \omega _ { t } + \nu _ { t } ^ { B } ) + O ( \eta _ { t } ^ { 2 } )$ . Using $\log ( 1 + x ) = x + O ( x ^ { 2 } )$ proves Eq. 23. The terms involving ω<sub>t</sub> cancel, and at balance $\nu _ { t } ^ { A } - \nu _ { t } ^ { B } = F _ { t } / N$

Numerical verification of the optimizer identities. A standalone numerical suite checks the AdamW state identities over 80 updates of a matrix regression with two coupled symmetry pairs, counterexamples obtained by omitting each required hyperparameter transformation, canonicalized restarts with common gradient clipping, and the zero-ϵ scalar counterexample. The numerical suite also checks the relative-increment exponents of Corollary 6 and the exact defect recursion in Eq. 22. These checks complement the proofs above.

## D EXPERIMENTAL AND IMPLEMENTATION DETAILS

## D.1 Models and Training Protocols

Architecture. The original small GPT-style experiments use Pre-LN, with normalization before each sub-layer, learned positional embeddings, tied input and output embeddings, bias-free linear projections, and context length 256. Later experiments specify their architecture and context-length changes. The vocabulary size |V| is 50,257, as determined by the GPT-2 byte-pair encoding (BPE) tokenizer. Linear and embedding weights in the GPT-style model use $\mathcal { N } ( 0 , 0 . 0 2 ^ { 2 } )$ ; normalization gains retain their default initialization. These $\mathrm { G P T - s t y }$ le runs apply no additional $1 / \sqrt { 2 L }$ factor to residual-output projections.

The four original GPT-style configurations range from 22M to 354M parameters. Table 4 lists their widths, depths, head counts, and peak learning rates under the names small, medium, deep, and xlarge.

Training. We use AdamW (Loshchilov and Hutter, 2019) with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 , { \epsilon } = 1 0 ^ { - 8 }$ , weight decay 0.1, and a cosine learning rate schedule with linear warmup over the first 2,000 steps and a minimum learning-rate ratio of 0.1. Gradient clipping is applied with max norm 1.0. The default batch size is 64 for small and deep models and 32 for medium models. The xlarge configuration uses batches of 16 with gradient accumulation over 4 micro-batches, giving an efective batch size of 64. Training uses bfloat16 (BF16) mixed precision for eficiency.

Datasets. WikiText-103 (Merity et al., 2017) is the primary dataset, with approximately 103M training words; GPT-2 BPE token counts and processed training-token budgets difer. OpenWebText (Gokaslan and Cohen, 2019) is used for multi-dataset validation. OpenWebText contains approximately 8B tokens, and we use a subset for training.

SAGGS hyperparameters. Hard SAGGS rebalances all symmetry pairs every 100 steps. Soft SAGGS uses the reported interval and $\alpha = 0 . 5$ unless a sweep specifies otherwise. $\mathrm { S A G G S + P } ,$ used only in some supplementary experiments, adds to SAGGS an auxiliary penalty on the training loss: the mean of $[ \log ( \Vert A \Vert _ { \mathrm { F } } + \delta ) - \log ( \Vert B \Vert _ { \mathrm { F } } + \delta ) ] ^ { 2 }$ over the rebalanced pairs, with $\delta : = 1 0 ^ { - 8 }$ , weighted by 0.01 unless a protocol specifies otherwise. The penalty is not part of SAGGS or Soft SAGGS. We rebalance Q–K, V–O, and MLP pairs in ReLU and SwiGLU models. GELU models use only Q–K and V–O pairs, since GELU up–down rescaling is not an exact symmetry.

Reparameterization procedure. The exponential reparameterization is applied after standard initialization by multiplying $\begin{array} { r } { { \bf A } _ { \ell }  e ^ { \lambda \ell } { \bf A } _ { \ell } } \end{array}$ and $B _ { \ell } \gets e ^ { - { \lambda } \ell } B _ { \ell }$ for each symmetry pair. For exact pairs, this rescaling preserves the initial forward pass in exact arithmetic across all λ values. GELU MLP rescaling in the approximation diagnostic is excluded from this identity.

The mathematical sections index layers from 1 to L; the training code uses $0 , \ldots , L - 1$ , so its largest applied scale is $c _ { L - 1 } = e ^ { \lambda ( L - 1 ) }$ . Numerical examples use the code convention. Small and deep models use 30K steps, medium models 25K to 30K steps, and 354M models 15K to 20K steps unless their protocols specify otherwise. The logged gradients in SAGGS+P experiments include the auxiliary penalty; the exact gradient-coordinate law concerns the data loss alone.

Evaluation. Validation PPL is computed on the WikiText-103 validation set or the corresponding OpenWebText validation set, using non-overlapping windows at the experiment’s configured sequence length, usually 256 or 512. Standard evaluation traverses the validation loader; any experiment-specific evaluation limits are specified in

the corresponding protocol. Most experiments report mean ± SD over the paired seeds stated in each protocol;   
Appendix D.2 lists the number of seeds in every experiment.

## D.2 Seeds and Computing Infrastructure

Random seeds. Paired comparisons use the same optimization seed for both arms, so the two arms share initialization, data order, and dropout masks. All seed pools were fixed before the corresponding experiments, and no seed was selected retrospectively from a larger pool.

A three-seed core pool was fixed before the multi-seed cross-scale comparisons and the three-seed 101M comparison. Three-seed experiments use this pool: the three-seed 101M comparison, the hard-rebalancing Lion comparison, the Lion state ablation, the OpenWebText replication, the 123.8M and 296.7M scale replications, the 354M stress test at $\lambda = 0 . 2$ , the 22M multi-seed cross-scale comparison, and the 101M one at $\lambda = 0 . 2$ . Four additional seeds, fixed before the primary seven-seed comparisons, extend the core pool to seven seeds for the seven-seed 101M comparison, the 31M architecture and tokenizer replication, and the soft-rebalancing Lion comparison. The branched checkpoint-surgery resume and the optimizer-state diagnostics use a separate seven-seed pool, which also contains the core pool and was fixed before those experiments. Two seeds from the core pool are used in the natural-drift experiment, the deep-model stress test, the 101M multi-seed cross-scale comparison at $\lambda = 0 .$ , and the 354M one. A single seed from the core pool is used in the 354M stress test at $\lambda \in \{ 0 , 0 . 1 \}$ , the SwiGLU stress test, the 101M rescaling-strength sweep, and the 1.2B training comparison.

Compute. The main experimental suite was run on NVIDIA H100 80GB GPUs, using 4×H100 single-node runs with distributed data parallel (DDP) training for the largest configurations. The implementation also supports Intel Data Center GPU Max devices through the PyTorch XPU backend; DDP runs use NCCL on CUDA devices and XCCL on XPU devices. The 31M architecture and tokenizer replication, the 123.8M scale replication and branched checkpoint-surgery resume, the 31M optimizer-state diagnostics, and the 296.7M scale replication were executed on Aurora Intel Data Center GPU Max 1550 XPU tiles. Runtime depends on the recipe and hardware. For example, the 1.2B training comparison records 2,494 to 2,518 seconds per configuration on four H100s, equivalent to 41.6 to 42.0 minutes or approximately 2.8 H100-hours per configuration. These values are training timers, including periodic evaluation, rather than end-to-end times including data and model setup these timers should not be extrapolated from parameter count alone.

## D.3 Implementation

Implementation checks. We verify the implementation on CPU and replay the same suite on Aurora using an XPU device. The checks confirm exact forward preservation, the closed-form gradient law, and SAGGS forward preservation for Q–K, V–O, and MLP pairs on both the GPT-style model and the RMSNorm, RoPE, SwiGLU, and GQA model of the architecture and tokenizer replication. The checks also verify the optimizer-state co-transform used in the Lion state ablation by confirming that Adam-family first- and second-moment bufers receive the chain-rule scale factors after a hard SAGGS orbit move and after an explicit exact reparameterization surgery. Both runs pass all checks: The largest relative gradient-law error is below $3 \times 1 0 ^ { - 7 }$ , and the optimizer-state relative error is zero.

Reference implementation. Listing 1 reproduces the rebalancing operator of our implementation, trimmed to the bias-free projections used in the primary training experiments. The pair map follows Table 1: For SwiGLU the MLP pair is the value and output matrices and the gate is untouched, and GELU models use only the Q–K and V–O pair types. For each pair p the function computes $\rho _ { p } = \left\| A _ { p } \right\| _ { \mathrm { F } } / \left\| B _ { p } \right\| _ { \mathrm { F } }$ and applies the damped rebalancing map of Eq. 12 with $c _ { p } ^ { \mathrm { e f f } } ( \alpha ) = \rho _ { p } ^ { \alpha / 2 } ; \alpha = 1$ gives the hard rebalancing of Algorithm 1. Pairs with a norm below $\delta _ { \mathrm { s k i p } } = 1 0 ^ { - 1 2 }$ are skipped, as analyzed in Eq. 25. The optional moment co-transform is the state-covariant variant of Appendix D.5, used only in the Lion state ablation and the optimizer-state diagnostics; default hard and Soft SAGGS leave the bufers unchanged. For afine projections the full implementation additionally applies the bias co-transforms of Appendix B.1. The training loop calls the operator on the 32-bit floating-point (FP32) parameters, outside the BF16 autocast region, immediately after the optimizer update whenever t mod $T _ { \mathrm { b a l } } = 0$

Listing 1: Hard SAGGS with alpha=1 and Soft SAGGS with alpha<1 for bias-free projections, with optional optimizer-state co-transformation.

```python
import torch
2
3
4 def get_pair_modules(block, pair_type):
5 """Modules (A, B) of one exact pair in a block (Table 1)."""
6 if pair_type == 'qk':
return block.attn.W_Q, block.attn.W_K
8 if pair_type == 'vo':
9 return block.attn.W_V, block.attn.W_O
10 if pair_type == 'mlp':
11 if hasattr(block.mlp, 'W1'): # ReLU up and down
12 return block.mlp.W1, block.mlp.W2
13 if hasattr(block.mlp, 'W_value'): # SwiGLU value and output; gate kept
14 return block.mlp.W_value, block.mlp.W_out
15 return None, None
16
17
18 def rescale_optimizer_state(optimizer, param, grad_scale):
19 """Buffer co-transform for param ← s * param, grad_scale = 1 / s:
20 gradient averages scale linearly, squared-gradient averages quadratically."""
21 state = optimizer.state.get(param)
22 if not state:
23 return
24 for key, buf in state.items():
25 if not torch.is_tensor(buf) or buf.shape != param.shape:
26 continue
27 if key == 'exp_avg': # Adam and AdamW first moment; Lion momentum
28 buf.mul_(grad_scale)
29 elif key == 'exp_avg_sq': # Adam and AdamW second-moment estimate
30 buf.mul_(grad_scale * grad_scale)
31
32
33 @torch.no_grad()
34 def saggs_rebalance_soft(model, pair_types=('qk', 'vo', 'mlp'),
35 alpha=0.5, optimizer=None,
36 transform_optimizer_state=False):
37 """Soft SAGGS with alpha in (0, 1]; alpha = 1 is Algorithm 1."""
38 for block in model.blocks:
39 for pt in pair_types:
40 A_mod, B_mod = get_pair_modules(block, pt)
41 if A_mod is None:
42 continue
43 A, B = A_mod.weight, B_mod.weight
44 norm_A, norm_B = A.norm(), B.norm() # Frobenius norms
45 if norm_A < 1e-12 or norm_B < 1e-12: # delta_skip guard
46 continue
47 c_eff = (norm_A / norm_B).pow(alpha * 0.5) # rho^(alpha/2)
48 A.div_(c_eff) # A ← A / c_eff
49 B.mul_(c_eff) # B ← c_eff * B
50 if transform_optimizer_state: # state-covariant only
51 rescale_optimizer_state(optimizer, A, c_eff)
52 rescale_optimizer_state(optimizer, B, c_eff.reciprocal())
53
54
55 def saggs_rebalance(model, pair_types=('qk', 'vo', 'mlp')):
56 """Hard SAGGS (Algorithm 1): balanced orbit representative."""
57 saggs_rebalance_soft(model, pair_types, alpha=1.0)
```

## D.4 Model Configurations

Table 4 summarizes the architectures and peak learning rates. Experiment-specific protocols specify any departures from these defaults.

The GPT-2-style models use context length 256 unless otherwise stated; the xlarge, 1B, and later 101M replication recipes use 512. At the same widths, the SwiGLU stress test uses three-projection variants with 25.5M parameters

Table 4: Model configurations. LR is the peak learning rate; modern-architecture head counts are pairs $\left( h _ { q } , h _ { k v } \right)$ of query heads and key-value (KV) heads. The first four parameter counts assume two-projection ReLU or GELU MLPs. Context lengths and SwiGLU variants are specified in the text.
<table><tr><td>Name</td><td> $d$ </td><td> $d _ { \mathrm { f f } }$ </td><td>L</td><td>Heads</td><td>Params</td><td>LR</td></tr><tr><td> $\mathrm { G P T - 2 { \cdot } s t y l e }$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Small</td><td>256</td><td>1024</td><td>12</td><td>4</td><td>22M</td><td> $6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Medium</td><td>512</td><td>2048</td><td>24</td><td>8</td><td>101M</td><td> $6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Deep</td><td>256</td><td>1024</td><td>48</td><td>4</td><td>51M</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Xlarge</td><td>1024</td><td>4096</td><td>24</td><td>16</td><td>354M</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>1B (SwiGLU)</td><td>2048</td><td>5504</td><td>22</td><td>16</td><td>1.22B</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Modern architecture</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Modern-small</td><td>384</td><td>1024</td><td>12</td><td>(6,2)</td><td>31M</td><td> $4 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Modern-medium-plus</td><td>640</td><td>1728</td><td>24</td><td>(10,2)</td><td>123.8M</td><td> $2 . 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Modern-large</td><td>896</td><td>2432</td><td>32</td><td>(14, 2)</td><td>296.7M</td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Pre-trained</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-2 (pre-trained)</td><td>768</td><td>3072</td><td>12</td><td>12</td><td>124M</td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr></table>

at $L = 1 2$ and 63.3M at $L = 4 8$ . Modern-architecture models use RMSNorm, RoPE, SwiGLU, GQA, and T5 SentencePiece tokenization. Their context length is 256 for the 31M model in the architecture and tokenizer replication and the optimizer-state diagnostics, and 512 for the larger models.

## D.5 Optimizer-State Co-Transformation

Optimizer-state co-transform ablation. SAGGS is a coordinate change on the symmetry orbit, whereas practical optimizers also maintain state variables. To separate the efect of the orbit move from the possible perturbation of optimizer state, we also evaluate a state-covariant variant. If a rebalance applies $A ^ { \prime } : = A / c$ and $\pmb { B } ^ { \prime } : = c \pmb { B }$ , then the data-gradient coordinates transform as $\nabla _ { A ^ { \prime } } \mathcal { L } = c \nabla _ { A } \mathcal { L }$ and $\nabla _ { B ^ { \prime } } \mathcal { L } = c ^ { - 1 } \nabla _ { B } \mathcal { L }$ . Accordingly, for an Adam-family first-moment bufer or a Lion momentum bufer $( m _ { A } , m _ { B } )$ we map $\pmb { m } _ { A } ^ { \prime } : = c \pmb { m } _ { A }$ and $\pmb { m } _ { B } ^ { \prime } : = c ^ { - 1 } \pmb { m } _ { B }$ , and for Adam-family second-moment estimates $( { \pmb v } _ { A } , { \pmb v } _ { B } )$ we map $\pmb { v } _ { A } ^ { \prime } : = c ^ { 2 } \pmb { v } _ { A }$ and $\pmb { v } _ { B } ^ { \prime } : = c ^ { - 2 } \pmb { v } _ { B }$ By default, both hard and Soft SAGGS leave the optimizer bufers unchanged, so neither method needs access to the internal state of the optimizer. We use the state-covariant variant only as a diagnostic: In the Lion state ablation (Appendix F.5), this variant measures how leaving Lion’s momentum bufer in the old coordinates afects the cost of hard rebalancing.

## D.6 Cost, Weight Decay, and Numerical Guards

Computational cost. Each rebalance computes two Frobenius norms and rescales all entries, costing $O ( n _ { A } + n _ { B } )$ per pair with O(1) tensor operations. At interval $T _ { \mathrm { b a l } } .$ , this rebalancing adds amortized work of $O ( ( n _ { A } + n _ { B } ) / T _ { \mathrm { b a l } } )$ per pair per optimization step. With full-layer $\mathrm { Q { - } K , \ V { - } O }$ , and MLP pairs there are 3L pairs; independent headwise Q–K and V–O pairs give (2H + 1)L for standard H-head attention.

Interaction with weight decay. A common decoupled decay factor multiplies both matrices equally and leaves their norm ratio unchanged (Zhang et al., 2019). The one-step balance-defect recursion in Eq. 22 records the contribution of this decay factor; we discuss the measured drift in Appendix H.3.

Numerical skip guard. The implementation skips a pair when either norm is below $\delta _ { \mathrm { s k i p } } : = 1 0 ^ { - 1 2 }$ , whereas Algorithm 1 states the ideal positive-norm map. For a scheduled soft event k with positive norms, let $r _ { k } : = { }$ $\log ( \left. A \right. _ { \mathrm { F } } / \left. B \right. _ { \mathrm { F } } )$ before the event and let $z _ { k }$ be the log-ratio afterward. The implemented correction obeys, in exact arithmetic,

$$
\alpha _ { k } ^ { \mathrm { e f f } } : = \alpha { \bf 1 } \{ \mathrm { m i n } \{ \| A \| _ { \mathrm { F } } , \| B \| _ { \mathrm { F } } \}  \ge \delta _ { \mathrm { s k i p } } \} , \qquad z _ { k } = ( 1 - \alpha _ { k } ^ { \mathrm { e f f } } ) r _ { k } .\tag{25}
$$

A skipped positive-norm pair has $z _ { k } = r _ { k }$ , so the hard canonicalizer and fixed-α contraction identities apply to executed corrections.

## D.7 Code Availability and Reproducibility

Appendix C gives self-contained proofs under the stated assumptions. The experimental protocols and table captions specify model configurations, datasets, optimizers, training budgets, schedules, and hardware. $\mathrm { A p - }$ pendix D.2 describes the seed protocol and seed counts, and Appendix E.2 defines the paired statistics and multiple-comparison conventions.

Appendix D.3 gives a reference implementation of the rebalancing operator (Listing 1) and the checks used to verify this implementation. WikiText-103 and OpenWebText are public datasets.

Reproduction uses fixed seeds and the documented hardware and backend. Because torch.nn.functional. scaled\_dot\_product\_attention can introduce backend non-determinism, reruns can difer from the reported values within numerical tolerance.

## E STATISTICAL METHODOLOGY AND SUMMARY OF RESULTS

The primary comparisons and checkpoint-resume experiments use seven paired seeds; OpenWebText and the larger modern architectures use three. We report the mean paired PPL diference ∆<sup>¯</sup> , its sample SD $\sigma _ { \Delta }$ , paired Cohen’s $d ,$ and exact sign-flip p-values. Unless stated otherwise, a diference is Vanilla minus the method, so a positive value favors the method, and ± denotes a sample SD. With seven seeds, the one-sided sign-flip p-value $p _ { \mathrm { s f } } ^ { + }$ takes its smallest possible value, 0.0078, exactly when all seven diferences are positive; with three seeds, its smallest value is 0.125.

## E.1 Paired Comparisons

For every paired comparison of Vanilla with a method at fixed λ and fixed model size, we use the same optimization seed s for both arms, covering data ordering, weight initialization, and dropout mask, so the paired diference $\Delta ^ { ( s ) } : = { \cal M } ^ { \mathrm { V a n i l l a } } ( s ) - { \cal M } ^ { \mathrm { M e t h o d } } ( s )$ controls these stochastic factors. Checkpoint-branch diagnostics share the base checkpoint and use the continuation protocol specified in the corresponding experiment. The within-seed design reduces variation due to these shared stochastic factors. The primary 101M WikiText-103, 31M architecture replication, and Lion comparisons use seven paired seeds (Appendix D.2).

## E.2 Statistics and Sign-Flip Tests

Reported statistics. The paired summaries use the mean $\bar { \Delta } .$ , the paired SD $\sigma _ { \Delta } : = [ ( 1 / ( n - 1 ) ) \sum _ { s } ( \Delta ^ { ( s ) } - \bar { \Delta } ) ^ { 2 } ] ^ { 1 / 2 }$ and paired Cohen’s $d : = \bar { \Delta } / \sigma _ { \Delta }$ . Tables report the subset of these statistics relevant to each comparison. Unless explicitly labeled otherwise, primary ± values are sample SDs, not standard errors. Earlier small-n tables use a diferent dispersion convention: Those summaries use population SDs, while the later paired analyses use the $n - 1$ denominator. We do not pool these summaries. Diferences are computed before rounding; subtracting two rounded means can difer by one last rounded digit from the reported paired diference. Later table captions state only experiment-specific details. Unless a caption specifies otherwise, a positive paired $\Delta$ is the left-minus-right or Vanilla-minus-method contrast on validation PPL, and $p _ { \mathrm { s f } } ^ { + }$ is the one-sided sign-flip p-value for a positive mean.

Sign-flip tests. For finite-sample significance, we use exact paired sign-flip randomization tests as the primary p-value whenever the claim is a paired $n = 7$ contrast. Their finite-sample validity requires invariance of the joint null distribution under independent sign flips of the paired diferences. For example, the condition holds for independent diferences symmetric about zero. Pairing seeds or assuming only a zero mean does not establish this invariance. At $n = 7 ,$ , the smallest one-sided sign-flip $p$ is $1 / 2 ^ { 7 } = 0 . 0 0 7 8 ;$ at $n = 3$ the p-value floor is only 0.125, so three-seed comparisons can establish efect direction and scale but not a $p < 0 . 0 5$ sign-flip claim. We also report the paired statistic $t : = \sqrt { n } \bar { \Delta } / \sigma _ { \Delta }$ as a supplementary descriptive summary; significance claims use the sign-flip test.

For paired diferences $\Delta _ { 1 } , \ldots , \Delta _ { n }$ , we enumerate all $2 ^ { n }$ sign assignments and compute the one-sided value

$$
p _ { \mathrm { s f } } ^ { + } : = 2 ^ { - n } \sum _ { \sigma \in \{ \pm 1 \} ^ { n } } { \bf 1 } \left\{ { \frac { 1 } { n } } \sum _ { i } \sigma _ { i } \Delta _ { i } \geq \bar { \Delta } \right\} .
$$

The two-sided zero-efect test compares the absolute permuted mean against |∆<sup>¯</sup> |. These tests do not establish equivalence against a prespecified practical margin.

Efect-size thresholds. We use the conventional Cohen thresholds adapted to the paired design (Cohen, 2013; Lakens, 2013): $| d | < 0 . 2$ negligible, $0 . 2 \leq | d | < 0 . 5$ small, $0 . 5 \leq | d | < 0 . 8$ medium, $| d | \geq 0 . 8$ large, $| d | \geq 2 . 0$ very large. The PPL efect at $\lambda = 0 . 2$ $n = 7$ on the $1 0 1 \mathrm { M }$ medium model is d ≈ +14.2 in the primary comparison (Sec. 5.2 and Table 2), with all seven paired diferences positive. We interpret paired efect sizes together with absolute metric changes and sample size, rather than relying on the p-value alone.

Multiple-comparison handling. The primary hypothesis that SAGGS protects the language-model objective under a forward-equivalent $\lambda = 0 . 2$ orbit imbalance is tested by the 101M WikiText-103 paired seven-seed contrast in Sec. 5.2. The 31M modern-architecture seven-seed replication and the Lion and Soft SAGGS seven-seed comparison are primary replications of the architecture, tokenizer, and optimizer variants. Secondary contrasts including other λ values, scale variants, and workflow diagnostics, provide supporting evidence. Reported p-values are unadjusted unless labeled otherwise; the unadjusted p-values must not be interpreted as family-wise significance claims. The single pre-declared core PPL contrast is separate from these secondary families.

Single-seed 1.2B comparison. The 1.2B comparison in Appendix G.12 uses a single seed because of compute constraints. Its training timers are 2,494 to 2,518 seconds per configuration on four H100s, or approximately 2.8 H100-hours per configuration, excluding data and model setup. We report this comparison’s observed diferences without an inferred p-value or Cohen’s d (Table 23). The seven-seed 101M, modern-architecture, Lion, and checkpoint-resume comparisons are the primary statistical evidence; the three-seed 123.8M and 296.7M comparisons test transfer across scales.

Pre-specification of λ and seed choices. Throughout the paper, pre-specified means fixed in our experiment plan before inspecting the corresponding held-out result; the phrase does not imply external public registration. The stress-test setting $\lambda = 0 . 2$ was selected from the empirical sensitivity observed in initial single-seed exploratory runs and then fixed in advance for all subsequent paired confirmation experiments with $\lambda > 0 .$ The paired seed pools were likewise fixed before the corresponding experiments were executed (Appendix D.2). The primary paired tables include all seeds of the corresponding pools and the λ values fixed in those configurations.

## E.3 Summary of Primary and Supporting Comparisons

Role of each comparison. The main protection result is $\bar { \Delta } _ { \mathrm { P P L } } = + 0 . 3 7 4 \pm 0 . 0 2 6$ at $\lambda = 0 . 2 , n = 7$ , with paired Cohen’s $d = 1 4 . 2 1$ (Table 2), measured under deliberate reparameterization. At λ = 0, mean diferences are small in the core AdamW and selected Soft and Lion comparisons; the three-seed Lion state ablation (Appendix F.5) is the main adverse unperturbed case.

SAGGS and Soft SAGGS are orbit-rebalancing steps for exact bilinear symmetries, complementary to the optimizer: Hard SAGGS selects the balanced representative, and Soft SAGGS contracts the distance to that representative in log-ratio coordinates. Deliberate rescaling tests measure sensitivity to orbit position; checkpoint-surgery and stale-state branches test recovery after an edit. Table 5 records the statistical support of each comparison. Paired $n = 7$ comparisons are the primary evidence; the $n = 3$ modern-architecture comparisons have a sign-flip floor of 0.125, and the 1.2B run is a single-seed scale check.

Table 5: Primary PPL comparisons and checkpoint-resume diagnostics. $\bar { \Delta }$ is the paired left-minus-right diference, reported as mean ± sample SD. Rows marked <sup>†</sup> use three paired seeds; all others use seven. Soft denotes Soft SAGGS, WT-103 denotes WikiText-103, and V denotes Vanilla.
<table><tr><td>Setting</td><td>λ</td><td>Contrast</td><td> $\bar { \Delta }$ </td><td> $p _ { \mathrm { s f } } ^ { + }$ </td><td>Purpose</td></tr><tr><td colspan="6">Primary</td></tr><tr><td>101M WT-103</td><td>0.2</td><td> $\mathrm { ~ V ~ - ~ S A G G S }$ </td><td> $+ 0 . 3 7 4 \pm 0 . 0 2 6$ </td><td>0.0078</td><td>Core language-model protection</td></tr><tr><td>31M modern</td><td>0.2</td><td> $\mathrm { V } \mathrm { ~ - ~ } \mathrm { S o f t }$ </td><td> $+ 0 . 7 8 0 \pm 0 . 0 4 0$ </td><td>0.0078</td><td>Architecture and tokenizer</td></tr><tr><td>101M Lion</td><td>0.2</td><td> $\mathrm { V } \mathrm { ~ - ~ } \mathrm { S o f t }$ </td><td> $+ 3 4 . 4 6 5 \pm 2 5 . 1 7 0$ </td><td>0.0078</td><td>Optimizer transfer</td></tr><tr><td colspan="6">Workflow</td></tr><tr><td>123.8M surgery</td><td></td><td>No-SAGGS - control</td><td> $+ 3 . 7 7 5 \pm 0 . 4 1 6$ </td><td>0.0078</td><td>State-reset resume</td></tr><tr><td colspan="6">Mechanism</td></tr><tr><td>31M reset</td><td></td><td>Reset – reset ctrl</td><td> $+ 0 . 2 7 7 \pm 0 . 0 1 5$ </td><td>0.0078</td><td>Reset penalty isolated</td></tr><tr><td>31M reset</td><td></td><td>Soft – reset ctrl</td><td> $+ 0 . 0 0 7 \pm 0 . 0 0 6$ </td><td>0.0156</td><td>Residual after Soft</td></tr><tr><td>31M stale</td><td></td><td> $\mathrm { S t a l e } - \mathrm { S o f t }$ </td><td> $+ 0 . 0 6 2 \pm 0 . 0 1 6$ </td><td>0.0078</td><td>Stale-state recovery</td></tr><tr><td>31M partial</td><td></td><td> $\mathrm { D e e p { - } s t a l e - S o f t }$ </td><td> $+ 0 . 0 6 2 \pm 0 . 0 0 8$ </td><td>0.0078</td><td>Partial-state recovery</td></tr><tr><td colspan="6">Scale transfer</td></tr><tr><td>123.8M modern†</td><td>0.2</td><td> $\mathrm { V } \mathrm { ~ - ~ } \mathrm { S o f t }$ </td><td> $+ 2 1 . 1 5 6 \pm 0 . 6 6 7$ </td><td>0.1250</td><td>Larger modern architecture</td></tr><tr><td>296.7M modern†</td><td>0.2</td><td> $\mathrm { V } \mathrm { ~ - ~ } \mathrm { S o f t }$ </td><td> $+ 8 5 . 8 1 3 \pm 4 . 7 3 0$ </td><td>0.1250</td><td>300M-class modern</td></tr></table>

Table 6: Three-seed 101M WikiText-103 comparison. Table 2 gives the seven-seed extension. Method columns report PPL as mean ± sample SD; $\bar { \Delta }$ reports paired Vanilla-minus-SAGGS diferences in the same format.
<table><tr><td> $\lambda$ </td><td>Vanilla</td><td>SAGGS</td><td> $\bar { \Delta }$ </td></tr><tr><td>0.0</td><td> $2 2 . 8 8 \pm 0 . 0 6 0$ </td><td> $2 2 . 9 2 \pm 0 . 0 3 3$ </td><td> $- 0 . 0 4 4 \pm 0 . 0 2 6$ </td></tr><tr><td>0.2</td><td> $2 3 . 2 5 \pm 0 . 0 2 4$ </td><td> $2 2 . 8 7 \pm 0 . 0 3 6$ </td><td> $+ 0 . 3 8 0 \pm 0 . 0 2 5$ </td></tr></table>

## F PRIMARY COMPARISONS AND OPTIMIZER-STATE ABLATIONS

The subsections below group related comparisons by purpose.

## F.1 101M WikiText-103 Comparisons

The three-seed comparison uses a paired $2 \times 2 \times 3$ design on WikiText-103: $\lambda \in \{ 0 , 0 . 2 \}$ , Vanilla or SAGGS, and three paired seeds. The medium ReLU Transformer has $L = 2 4 , d = 5 1 2 .$ , 8 heads, about 101M parameters, and context length 512. Each run uses 30K AdamW steps, learning rate $6 \times 1 0 ^ { - 4 } , \beta _ { 2 } = 0 . 9 5$ , weight decay 0.1, 2K-step warmup, cosine decay to 10% of peak, gradient clipping at 1.0, BF16 mixed precision, and batch size 32. At λ = 0.2, Q–K pairs receive the depth-exponential edit at initialization. SAGGS rebalances Q–K, V–O, and ReLU up–down pairs every 100 steps. Data sampling and initialization are shared within seed. The primary comparison in Table 2 extends this protocol to seven paired seeds.

Table 6 reports paired diferences $\Delta _ { s } = \mathrm { P P L } _ { \mathrm { V a n i l l a } } ( s ) - \mathrm { P P L } _ { \mathrm { S A G G S } } ( s )$ . At λ = 0, the mean is $- 0 . 0 4 4 \pm 0 . 0 2 6$ PPL: a small observed cost on a base near 22.9 PPL. At λ = 0.2, the mean is $+ 0 . 3 8 0 \pm 0 . 0 2 5$ PPL and all three seeds favor SAGGS. The paired efect sizes are $d = - 1 . 6 9$ and $d = 1 5 . 3 5$ , respectively. With $n = 3 ,$ the one-sided exact sign-flip p-value floor is 0.125.

Pair-norm measurements in the seven-seed comparison. SAGGS keeps the final pair log-ratio imbalance below $1 0 ^ { - 3 }$ in these comparisons, whereas Vanilla Q–K imbalance is about 0.15 at $\lambda = 0$ and about 3 at $\lambda = 0 . 2$ Rebalancing controls pair norms in both conditions; the PPL benefit appears under deliberate rescaling.

## F.2 Architecture and Tokenizer Replication

This experiment tests transfer across architecture and tokenizer choices. The experiment replaces the GPT-style layer normalization, learned absolute positions, and standard multi-head attention with RMSNorm, RoPE, SwiGLU, and GQA, and replaces GPT-2 BPE with T5 SentencePiece tokenization. We evaluate the resulting

Table 7: Seven-seed modern architecture and cross-tokenizer replication. The paired block is Vanilla minus Soft SAGGS with $\alpha = 0 . 5$
<table><tr><td></td><td colspan="2">Mean PPL</td><td colspan="5">Paired difference</td></tr><tr><td>λ</td><td>Vanilla</td><td> $\mathrm { S o f t } , \alpha = 0 . 5$ </td><td> $\bar { \Delta }$ </td><td>σ∆</td><td>t</td><td>d</td><td> $p _ { \mathrm { s f } } ^ { + }$ </td></tr><tr><td>0.0</td><td> $2 0 . 8 3 \pm 0 . 0 6$ </td><td> $2 0 . 8 1 \pm 0 . 0 6$ </td><td>+0.019</td><td>0.024</td><td>+2.11</td><td>+0.80</td><td>0.0391</td></tr><tr><td>0.2</td><td> $2 1 . 5 8 \pm 0 . 0 7$ </td><td> $2 0 . 8 0 \pm 0 . 0 6$ </td><td>+0.780</td><td>0.040</td><td>+52.18</td><td>+19.72</td><td>0.0078</td></tr></table>

31M model over seven paired seeds.

We train a 12-layer causal language model with $d = 3 8 4 ,$ six query heads, two KV heads, SwiGLU hidden width 1024, tied input and output embeddings, no biases, RMSNorm pre-normalization, RoPE, and T5 SentencePiece vocabulary. We re-tokenize WikiText-103 with the T5 tokenizer.

The optimizer is AdamW with $\eta = 4 \times 1 0 ^ { - 4 }$ , weight decay 0.1, cosine decay, 10% warmup, gradient clipping at 1.0, BF16 XPU autocast, batch size 48, sequence length 256, and 12K steps. The paired design compares Vanilla with Soft SAGGS at $\alpha = 0 . 5$ , crossed with $\lambda = 0$ and $\lambda = 0 . 2$ , over seven paired seeds. The reparameterization acts on $\mathrm { Q \mathrm { - } K , \ V \mathrm { - } O } ,$ , and SwiGLU value–output pairs, and Soft SAGGS rebalances those same pairs every 100 steps. A pre-training verification on a WikiText-103 batch confirms exact forward equivalence, with loss diference 0 and maximum logit diference below $2 . 4 \times 1 0 ^ { - 6 }$ , and gradient-law relative errors below $5 \times 1 0 ^ { - 7 }$ for Q–K, V–O, and SwiGLU value–output pairs.

The results reproduce both patterns observed in the 101M comparison (Table 7). First, the unperturbed $\lambda = 0$ comparison has a small observed diference: Vanilla and Soft SAGGS difer by only +0.019 ± 0.024 PPL, taking Vanilla minus Soft, relative to a baseline near 20.8 PPL. Second, the $\lambda = 0 . 2$ reparameterization creates a consistent degradation in Vanilla on all seven seeds, while Soft SAGGS remains near its unperturbed baseline: Vanilla is 21. $5 8 \pm 0 . 0 7$ PPL and Soft SAGGS is $2 0 . 8 0 \pm 0 . 0 6 ~ \mathrm { P P L }$ , with paired $\bar { \Delta } = + 0 . 7 8 0 \pm 0 . 0 4 0 , t = 5 2 . 1 8$ Cohen $d = 1 9 . 7 2 \ :$ , and exact one-sided sign-flip $p = 0 . 0 0 7 8 .$

The final pair-norm statistics show corresponding balance control: $\mathrm { A t } \ \lambda = 0 . 2$ , Vanilla ends with Q–K max absolute log-ratio about 2.65, V–O about 1.21, and MLP about 1.81, while Soft SAGGS keeps all three below $8 \times 1 0 ^ { - 4 }$ . The scale replications in Appendices G.15 and G.16 extend the same architecture to 123.8M and 296.7M parameters. PPL depends on the tokenizer, so these values are not comparable with the values of the GPT-2-tokenized models; all performance comparisons here use the same tokenizer and paired seeds.

## F.3 Hard Rebalancing with Lion

Lion (Chen et al., 2023) uses sign-based updates. Positive-scale invariance of its sign direction does not imply orbit equivariance of its parameter trajectory (Theorem 5). We test whether hard SAGGS protects Lion training under an exact orbit tilt.

Design. The protocol uses the medium ReLU Transformer with L = 24, d = 512, 8 heads, and 101M parameters, WikiText-103 with GPT-2 BPE, context length 512, and three paired seeds. We compare Vanilla and SAGGS at $\lambda = 0 . 2$ , applied to Q–K, V–O, and ReLU up–down pairs at initialization. Training uses 15K Lion steps with learning rate $2 \times 1 0 ^ { - 4 } , \beta = ( 0 . 9 , 0 . 9 9 )$ , weight decay 0.1, batch size 32, gradient clipping at 1.0, and BF16 mixed precision. The cosine schedule uses 1.5K warmup steps and decays to 10% of peak learning rate. SAGGS rebalances all three pair types every 100 steps.

Results. All three $\lambda = 0 . 2$ pairs favor SAGGS, with $\bar { \Delta } _ { \mathrm { V a n i l l a - S A G G S } } = + 5 2 . 8 9 6 \pm 3 1 . 5 4 3$ PPL, paired Cohen’s $d = 1 . 6 8 { \mathrm { . } }$ , and paired t = 2.90 (Table 8). With $n = 3 .$ , the one-sided exact sign-flip p-value floor is 0.125. The soft-rebalancing comparison in Appendix F.4 evaluates Soft SAGGS with seven paired seeds under both unperturbed and deliberately tilted starts. The Lion state ablation in Appendix F.5 compares hard, soft, and state-covariant rebalancing under matched Lion controls.

Table 8 records the Lion validation PPL of each paired seed in this comparison.

Table 8: Three-seed hard-rebalancing Lion comparison at $\lambda = 0 . 2 \mathrm { : }$ : 101M parameters, $L = 2 4$ , and 15K steps. Rows index the paired seeds. V denotes Vanilla, S denotes SAGGS, and $\Delta = \mathrm { P P L _ { V } - P P L _ { S } }$
<table><tr><td>Seed</td><td> $\mathrm { P P L v }$ </td><td> $\mathrm { P P L _ { S } }$ </td><td> $\Delta$ </td></tr><tr><td>1</td><td>55.992</td><td>23.954</td><td>+32.038</td></tr><tr><td>2</td><td>113.248</td><td>24.065</td><td>+89.183</td></tr><tr><td>3</td><td>61.429</td><td>23.963</td><td>+37.466</td></tr></table>

Table 9: Seven-seed Lion comparison on the 101M WikiText-103 model. PPL is reported as mean ± sample SD; the paired block reports Vanilla-minus-Soft SAGGS diferences.
<table><tr><td></td><td colspan="2">Mean PPL</td><td colspan="5">Paired difference</td></tr><tr><td> $\lambda$ </td><td>Vanilla</td><td>Soft SAGGS</td><td> $\bar { \Delta }$ </td><td>σ∆</td><td>t</td><td> $d$ </td><td> $p _ { \mathrm { s f } } ^ { + }$ </td></tr><tr><td>0.0</td><td> $2 4 . 1 5 \pm 0 . 3 9$ </td><td> $2 4 . 2 0 \pm 0 . 5 8$ </td><td>-0.055</td><td>0.232</td><td>-0.62</td><td>-0.24</td><td>0.6953</td></tr><tr><td>0.2</td><td> $5 8 . 3 2 \pm 2 5 . 2 0$ </td><td> $2 3 . 8 6 \pm 0 . 0 9$ </td><td>+34.465</td><td>25.170</td><td>+3.62</td><td>+1.37</td><td>0.0078</td></tr></table>

## F.4 Soft Rebalancing with Lion

Design. We use the 101M ReLU Transformer and 15K-step Lion recipe of the hard-rebalancing comparison: learning rate $2 \times 1 0 ^ { - 4 } , \beta = ( 0 . 9 , 0 . 9 9 )$ , weight decay 0.1, 1.5K-step warmup, cosine decay to 10% of peak learning rate, batch size 32, context length 512, gradient clipping at 1.0, and BF16 mixed precision on WikiText-103. The paired design compares Vanilla and Soft SAGGS at $\lambda \in \{ 0 , 0 . 2 \}$ , with all-pairs reparameterization and Soft $\alpha = 0 . 5$ every 100 steps. The hard and state-covariant comparisons in the Lion state ablation use three paired seeds.

Table 9 provides the seven-seed summaries. At $\lambda = 0 . 2 .$ every one of the seven paired seeds favors Soft SAGGS, giving $\bar { \Delta } _ { \mathrm { V a n i l l a - S o f t } } = + 3 4 . 4 6 5$ PPL and exact one-sided sign-flip $p _ { \mathrm { s f } } ^ { + } = 0 . 0 0 7 8$ . At $\lambda = 0$ , the paired mean is $- 0 . 0 5 5 \pm 0 . 2 3 2$ PPL, including one seed where Soft is 0.51 PPL worse than Vanilla. The results support protection under the deliberate tilt, with a small observed unperturbed mean cost.

## F.5 Optimizer-State Co-Transformation with Lion

A hard rebalance changes gradient coordinates while Lion retains a momentum bufer storing an EMA of past gradients. To measure the efect of this state mismatch, we compare Vanilla, hard SAGGS, Soft SAGGS, and the state-covariant variant of Appendix D.5 under matched medium-scale Lion protocols.

Design. The state-covariant experiment comprises six runs: $\lambda \in \{ 0 . 0 , 0 . 2 \}$ , three paired seeds, 101M parameters and $L = 2 4$ , WikiText-103, Lion $\eta = 2 \times 1 0 ^ { - 4 } , \beta = ( 0 . 9 , 0 . 9 9 )$ , weight decay 0.1, batch 32, 15K steps, all-pairs reparameterization, and $T _ { \mathrm { b a l } } = 1 0 0$ The only changed variable relative to hard SAGGS is that the Lion momentum bufer (exp\_avg) is multiplied by the same chain-rule factor as the gradient coordinate after every rebalance. The Vanilla, hard SAGGS, and Soft SAGGS baselines use the matched three-seed Lion baselines from the hard-rebalancing comparison. Their means and sample SDs are reported alongside the state-covariant variant in Table 10.

Interpretation. At $\lambda = 0$ , state-covariant SAGGS has a lower mean and less seed variation than hard SAGGS: $2 4 . 1 4 \pm 0 . 1 5$ PPL compared with 30.12 ± 10.56 PPL in Table 10. State-covariant SAGGS still has an observed mean cost of 0.225 PPL relative to Vanilla, with $\bar { \Delta } _ { \mathrm { V a n i l l a - S t a t e } } = - 0 . 2 2 5$ PPL and $n = 3 .$ . These results support an optimizer-state contribution to the cost of hard rebalancing, while showing that co-transforming the momentum bufer does not eliminate the measured unperturbed mean cost.

At $\lambda = 0 . 2$ , the state-covariant variant retains the full protective efect: Vanilla minus state-covariant SAGGS is $+ 5 2 . 9 1 7 \pm 3 1 . 5 3 9$ PPL, with $d = + 1 . 6 8$ (Table 10), matching the direction and order of magnitude of the matched Vanilla-minus-hard and Vanilla-minus-Soft contrasts in Table 10. State-covariant SAGGS and hard SAGGS difer by only −0.021 PPL on average at $\lambda = 0 . 2$ , while Soft SAGGS is slightly lower than state-covariant SAGGS by 0.117 PPL on average. These results motivate Soft SAGGS as the default when optimizer bufers are left unchanged. State-covariant hard rebalancing provides a complementary diagnostic when momentum-bufer transformations are available.

Table 10: Lion state-covariant ablation on the 101M model with $n = 3 .$ . The upper block reports PPL as mean ± sample SD, and the lower block reports paired Vanilla-minus-State diferences. Soft denotes Soft SAGGS; State denotes hard SAGGS with covariantly transformed momentum bufers.
<table><tr><td> $\lambda$ </td><td colspan="2">Vanilla</td><td>SAGGS</td><td> $\operatorname { S o f t }$ </td><td></td><td>State</td></tr><tr><td>0.0 0.2</td><td colspan="2"> $2 3 . 9 2 \pm 0 . 0 2$   $7 6 . 8 9 \pm 3 1 . 6 0$ </td><td> $3 0 . 1 2 \pm 1 0 . 5 6$   $2 3 . 9 9 \pm 0 . 0 6$ </td><td colspan="2"> $2 3 . 9 7 \pm 0 . 1 1$   $2 3 . 8 6 \pm 0 . 1 2$ </td><td> $2 4 . 1 4 \pm 0 . 1 5$   $2 3 . 9 7 \pm 0 . 0 8$ </td></tr><tr><td></td><td>λ</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Δ</td><td> $\sigma _ { \Delta }$ </td><td>t</td><td>d</td><td> $p _ { \mathrm { s f } } ^ { + }$ </td></tr><tr><td></td><td>0.0</td><td>-0.225</td><td>0.140</td><td>-2.78</td><td>-1.60</td><td>1.000</td></tr><tr><td></td><td>0.2</td><td>+52.917</td><td>31.539</td><td>+2.91</td><td>+1.68</td><td>0.125</td></tr></table>

Table 11: Branched checkpoint-surgery full paired contrasts on the 123.8M modern architecture with $n = 7 .$
<table><tr><td>Contrast</td><td></td><td>Δ</td><td> $\sigma _ { \Delta }$ </td><td>t</td><td>d</td><td> $p _ { \mathrm { s f } } ^ { + }$ </td></tr><tr><td>Surgery no SAGGS – control</td><td></td><td>+3.775</td><td>0.416</td><td>+24.00</td><td>+9.07</td><td>0.0078</td></tr><tr><td>Surgery no SAGGS – SAGGS</td><td></td><td>+3.765</td><td>0.413</td><td>+24.14</td><td>+9.12</td><td>0.0078</td></tr><tr><td>Surgery no SAGGS – Soft</td><td></td><td>+3.766</td><td>0.414</td><td>+24.06</td><td>+9.09</td><td>0.0078</td></tr><tr><td>Control - SAGGS</td><td></td><td>-0.010</td><td>0.008</td><td>-3.53</td><td>-1.33</td><td>0.9922</td></tr><tr><td>Control – Soft</td><td></td><td>-0.009</td><td>0.007</td><td>-3.36</td><td>-1.27</td><td>0.9844</td></tr></table>

## F.6 Branched Checkpoint-Surgery Resume

Motivation. The 123.8M scale replication in Appendix G.15 measures sensitivity to orbit imbalance at initialization. We now test a checkpoint-resume setting in which weights are edited and training continues with a fresh optimizer. For each seed, one shared phase-1 checkpoint is cloned into four continuation branches, isolating the efect of the edit and subsequent rebalancing.

Design. For each of the seven paired seeds, we train the same RMSNorm, RoPE, SwiGLU, and GQA T5 SentencePiece model used in the 123.8M scale replication for 10K normal WikiText-103 AdamW steps. We then reset the optimizer and branch into four 5K-step resumes: control, surgery without SAGGS, surgery with hard SAGGS, and surgery with Soft SAGGS. The non-control branches receive an exact all-pairs λ = 0.2 orbit surgery at the handof; the SAGGS branches then rebalance Q–K, V–O, and SwiGLU value–output pairs every 50 steps. Hard SAGGS applies an initial exact rebalance at the handof; Soft SAGGS applies $\alpha = 0 . 5$ at the handof and during the resume. The optimizer state is deliberately not carried across the handof. Each phase uses a fresh cosine schedule with 10% warmup: 1K steps in phase 1 and 500 steps in phase 2. The base learning rate, weight decay, batch size, context length, and precision match the 123.8M scale replication

The handof uses an exact symmetry edit: Across all edited branches, the absolute validation-loss change after surgery and any initial rebalance is below $7 . 4 \times 1 0 ^ { - 5 }$ . These changes are floating-point residuals of the full handof operation.

The optimizer-state-reset resume exposes a large coordinate efect. Surgery with no SAGGS reaches $2 4 . 6 2 \pm 0 . 5 8$ PPL, while the control resume reaches 20.84 ± 0.25 PPL, a paired degradation of +3.775 ± 0.416 PPL with all seven seeds in the same direction, with $p _ { \mathrm { s f } } ^ { + } = 0 . 0 0 7 8$ . Hard SAGGS and Soft SAGGS recover the final validation PPL closely: Their final PPLs are $2 0 . 8 5 \pm 0 . 2 5$ and $2 0 . 8 5 \pm 0 . 2 5$ , only 0.010 and 0.009 PPL above the control resume, respectively. The final Q–K log-ratio makes the mechanism visible: No-SAGGS surgery ends near 6.14, whereas both SAGGS variants end below $1 . 7 \times 1 0 ^ { - 4 }$

The experiment demonstrates protection in a controlled checkpoint-resume workflow with a deliberately introduced symmetry-orbit displacement and an optimizer-state reset. The comparison measures recovery of the final PPL; identical resumed trajectories require canonicalizing both checkpoints (Corollary 4).

Table 12: State-preserved checkpoint-surgery diagnostic on the 31M modern-architecture WikiText-103 model with $n = 7 .$ . The lower block reports paired contrasts.
<table><tr><td></td><td colspan="2">Resume condition</td><td>Final PPL</td><td colspan="2"></td></tr><tr><td></td><td colspan="2">Control, optimizer reset Surgery, optimizer reset Surgery reset + Soft SAGGS</td><td colspan="2"> $2 3 . 2 7 \pm 0 . 0 5$   $2 3 . 5 4 \pm 0 . 0 6$   $2 3 . 2 7 \pm 0 . 0 6$ </td></tr><tr><td colspan="3">Control, optimizer preserved Surgery, state-covariant optimizer</td><td colspan="2"> $2 3 . 9 2 \pm 0 . 0 6$ </td></tr><tr><td colspan="3"></td><td colspan="2"> $2 3 . 9 7 \pm 0 . 0 7$ </td></tr><tr><td colspan="2">Contrast</td><td>Δ  $\sigma \Delta$ </td><td>t</td><td>d</td><td> $p _ { \mathrm { s f } } ^ { + }$ </td></tr><tr><td colspan="2">Surgery reset – control reset</td><td>+0.277 0.015</td><td>+50.21</td><td>+18.98</td><td>0.0078</td></tr><tr><td colspan="2">Surgery reset – Soft</td><td>+0.270</td><td>0.013 +56.23</td><td>+21.25</td><td>0.0078</td></tr><tr><td colspan="2">Soft – control reset</td><td>+0.007</td><td>0.006 +2.98</td><td>+1.13</td><td>0.0156</td></tr><tr><td colspan="2">State-covariant - state control</td><td>+0.058</td><td>0.017 +8.83</td><td>+3.34</td><td>0.0078</td></tr><tr><td colspan="2">Reset control - state control</td><td>-0.650 0.019</td><td>-88.91</td><td>-33.60</td><td>1.0000</td></tr></table>

## F.7 Optimizer-State Diagnostics

Instead of being reset, the AdamW moments can be kept and transformed along with the weights, or loaded unchanged, which leaves the moments stale. Seven-seed experiments on the 31M model compare these cases, including moments that are stale only in the deeper half of the layers. With reset moments, the edit adds $0 . 2 7 7 \pm 0 . 0 1 5$ PPL relative to the matched control (Table 12). Keeping and transforming the moments leaves a smaller but nonzero gap of $0 . 0 5 8 \pm 0 . 0 1 7 \mathrm { P P I }$ relative to a control that also keeps its optimizer state, consistent with Theorem 2: Under a shared learning rate, transformed moments alone do not keep the two runs equivalent. Fully stale and half-stale moments yield penalties of $0 . 0 5 6 \pm 0 . 0 1 5$ and $0 . 0 5 6 \pm 0 . 0 0 7 \mathrm { P P L }$ , respectively (Tables 13 and 14). In the reset and stale cases, Soft SAGGS returns PPL to approximately the PPL of the control, with a residual of at most $0 . 0 0 7 \pm 0 . 0 0 6 ~ \mathrm { P P L }$ .

## F.8 State-Preserved Checkpoint-Surgery Diagnostic

The branched checkpoint-surgery resume deliberately resets optimizer state at the handof, matching the common workflow of importing a checkpoint without its AdamW moments. We now measure the residual penalty when optimizer moments are preserved and transformed covariantly with the exact orbit edit.

Design. For each of the seven paired seeds, we train the 31M modern RMSNorm, RoPE, SwiGLU, and GQA T5 SentencePiece model for 5K WikiText-103 AdamW steps. We then clone the same checkpoint into five 2.5K-step branches: control reset, surgery reset, surgery reset + Soft SAGGS, control state, and surgery state-covariant. The model uses context length 256, batch size 48, and peak learning rate $4 \times 1 0 ^ { - 4 }$ . Phase 1 follows a 7.5K-step cosine schedule with 750-step warmup. The reset branches create a fresh AdamW optimizer and a 2.5K-step cosine schedule with 250-step warmup; the state branches load the phase-1 AdamW and scheduler state. In the state-covariant surgery branch, the exact λ = 0.2 all-pairs orbit move also maps AdamW first moments by the chain-rule factor and second moments by its square. This branch applies the covariant moment map without periodic rebalancing during phase 2.

Table 12 reports the seven-seed summaries. The reset branch reproduces the checkpoint-surgery mechanism at a smaller model size. Surgery with optimizer reset is $+ 0 . 2 7 7 \pm 0 . 0 1 5$ PPL worse than the reset control over seven paired seeds, with exact sign-flip $p _ { \mathrm { s f } } ^ { + } = 0 . 0 0 7 8$ . Soft SAGGS recovers the penalty almost entirely, leaving only $+ 0 . 0 0 7 \pm 0 . 0 0 6$ PPL relative to the reset control. The largest immediate absolute validation-loss change after the edit, including the initial Soft rebalance where used, is below $6 . 7 \times 1 0 ^ { - 5 }$

When optimizer state is preserved and co-transformed, the surgery gap falls to $+ 0 . 0 5 8 \pm 0 . 0 1 7 \mathrm { P P L }$ relative to the state-preserved control. Moment co-transformation alone does not enforce trajectory equivariance: The shared learning rates already violate the coordinate rule in Theorem 2, even if ϵ = 0. Global clipping, finite precision, and state history can introduce further discrepancies. The residual is about one fifth of the reset-surgery gap. This comparison changes both moment history and the learning-rate continuation, so the roughly 79% reduction cannot be attributed to moment preservation alone or used to partition the 123.8M checkpoint-surgery penalty at a diferent model scale. The small measured handof loss change separately supports forward preservation. The state-preserved control itself is worse than the reset control in this short-horizon diagnostic. The experiment characterizes the residual penalty under the state-preserved continuation.

Table 13: Stale AdamW-state diagnostic on the 31M model with $n = 7 :$ final PPL as mean ± sample SD and paired contrasts after an exact $\lambda = 0 . 2$ edit. Stale branches retain the pre-edit moments.
<table><tr><td>Resume condition</td><td></td><td></td><td>Final PPL</td></tr><tr><td>Control, preserved state Surgery + stale AdamW state  $\mathrm { S u r g e r y + c o v a r i a n t \ A d a m W }$ </td><td>state  $\mathrm { S u r g e r y + s t a l e \ s t a t e + S o f t \ S A G G S }$ </td><td> $2 3 . 9 1 \pm 0 . 0 5$ </td><td> $2 3 . 9 6 \pm 0 . 0 7$   $2 3 . 9 6 \pm 0 . 0 7$   $2 3 . 9 0 \pm 0 . 0 5$ </td></tr><tr><td>Contrast</td><td>Δ  $\sigma _ { \Delta }$ </td><td>t</td><td>d</td><td> $p _ { \mathrm { s f } } ^ { + }$ </td></tr><tr><td>Stale – control</td><td></td><td></td><td> $+ 3 . 7 3$ </td><td>0.0078</td></tr><tr><td>Stale - covariant</td><td>+0.056 +0.002</td><td>0.015 0.001</td><td>+9.86  $+ 6 . 4 2$ </td><td> $+ 2 . 4 2$ </td></tr><tr><td>Stale – Soft</td><td>+0.062</td><td>0.016</td><td>+10.52</td><td>0.0078  $+ 3 . 9 8$ </td></tr><tr><td></td><td></td><td></td><td></td><td>0.0078</td></tr><tr><td>Covariant - control</td><td>+0.054</td><td>0.015</td><td>+9.21</td><td> $+ 3 . 4 8$  0.0078</td></tr><tr><td></td><td></td><td></td><td>-5.45</td><td></td></tr><tr><td>Soft — control</td><td>-0.005</td><td>0.003</td><td>-2.06</td><td>1.0000</td></tr></table>

## F.9 Stale Optimizer-State Checkpoint Edit

The state-preserved diagnostic compares optimizer reset against a carefully co-transformed AdamW state. A second checkpoint-resume condition arises when a checkpoint is edited by an exact symmetry transform, but its AdamW moments are loaded unchanged. This mismatch can occur whenever model weights are imported, rescaled, or surgically modified while optimizer internals are treated as opaque checkpoint metadata.

Design. We reuse the 31M modern RMSNorm, RoPE, SwiGLU, and GQA T5 SentencePiece WikiText-103 setting of the state-preserved diagnostic. For each of the seven paired seeds, we train the model for 5K phase-1 steps, then clone the resulting checkpoint’s model, AdamW, and scheduler states into four 2.5K-step branches: control state, surgery state stale, surgery state-covariant, and surgery state stale + Soft SAGGS. All surgery branches receive the exact λ = 0.2 all-pairs orbit edit. The stale branch leaves AdamW first and second moments in the old coordinates; the covariant branch maps first moments by the chain-rule factor and second moments by its square; the Soft branch uses the same stale optimizer state but applies Soft SAGGS at the handof and every 50 steps.

Table 13 reports the seven-seed summaries. The stale-state edit creates a small but highly consistent penalty: $+ 0 . 0 5 6 \pm 0 . 0 1 5 ~ \mathrm { P P L }$ relative to the state-preserved control over seven paired seeds, with $p _ { \mathrm { s f } } ^ { + } = 0 . 0 0 7 8$ , despite an immediate handof validation-loss change below $7 . 3 \times 1 0 ^ { - 5 }$ . Co-transforming AdamW moments is slightly better than stale reuse by only $+ 0 . 0 0 2 \pm 0 . 0 0 1 \mathrm { \ P P L }$ , and still remains $+ 0 . 0 5 4 \pm 0 . 0 1 5 ~ \mathrm { P P L }$ worse than control. This residual is consistent with Theorem 2: Keeping common learning rates while co-transforming moments is already insuficient for trajectory equivariance. Clipping, finite precision, and the ϵ floor can add to this efect; the contrast does not isolate their shares.

Soft SAGGS removes the stale-state penalty. The stale-minus-Soft contrast is $+ 0 . 0 6 2 \pm 0 . 0 1 6 ~ \mathrm { P P L }$ over all seven seeds, with $p _ { \mathrm { s f } } ^ { + } = 0 . 0 0 7 8$ , while Soft itself is $- 0 . 0 0 5 \pm 0 . 0 0 3 ~ \mathrm { P P L }$ relative to the state-preserved control. The efect size is much smaller than in the optimizer-reset workflow of the branched checkpoint-surgery resume, but the exact checkpoint edit still changes subsequent training when moments are stale. Periodic Soft SAGGS recovers the control PPL in these runs.

## F.10 Partial Optimizer-State Checkpoint Edit

The stale-state diagnostic treats the AdamW state as either fully stale or fully co-transformed. Real checkpoint workflows can fail in a more granular way: Optimizer-state shards may be missing or stale only for a subset o layers. This experiment tests that intermediate failure mode while keeping the forward edit exact.

Table 14: Partial AdamW-state diagnostic on the 31M model with $n = 7 :$ final PPL as mean ± sample SD and paired contrasts. In deep-half-stale branches, only shallow-layer moments are co-transformed.
<table><tr><td colspan="2">Resume condition</td><td></td><td>Final PPL</td><td></td><td></td><td rowspan="3"> $2 3 . 9 0 \pm 0 . 0 7$ </td></tr><tr><td colspan="9">Control state preserved Exact edit, covariant state</td></tr><tr><td colspan="5">Exact edit, all state stale</td><td colspan="2"> $2 3 . 9 6 \pm 0 . 0 7$   $2 3 . 9 6 \pm 0 . 0 7$ </td></tr><tr><td colspan="5">Exact edit, deep-half stale</td><td colspan="2"> $2 3 . 9 6 \pm 0 . 0 7$ </td></tr><tr><td colspan="5">Deep-half stale + Soft SAGGS</td><td colspan="2"> $2 3 . 9 0 \pm 0 . 0 6$ </td></tr><tr><td colspan="5"></td><td colspan="2"></td></tr><tr><td colspan="5">Contrast</td><td colspan="2"></td><td colspan="2"></td></tr><tr><td colspan="2">All stale – control</td><td>Δ</td><td> $\sigma _ { \Delta }$ </td><td>t</td><td></td><td>d</td><td> $p _ { \mathrm { s f } } ^ { + }$ </td></tr><tr><td colspan="2">Deep stale - control</td><td>+0.056</td><td>0.008</td><td></td><td>+19.19</td><td>+7.25</td><td>0.0078</td></tr><tr><td colspan="2">Deep stale - covariant</td><td>+0.056</td><td>0.007</td><td></td><td>+19.86 +4.45</td><td>+7.51</td><td>0.0078</td></tr><tr><td colspan="2">All stale – deep stale</td><td>+0.003</td><td>0.002 0.001</td><td>+0.71</td><td></td><td>+1.68</td><td>0.0078</td></tr><tr><td colspan="2"></td><td>+0.000</td><td></td><td></td><td></td><td>+0.27</td><td>0.2500</td></tr><tr><td colspan="2">Deep stale – Soft</td><td>+0.062</td><td>0.008</td><td></td><td>+19.92</td><td>+7.53</td><td>0.0078</td></tr><tr><td colspan="2"> $\mathrm { S o f t } \mathrm { ~ - ~ } \mathrm { c o n t r o l }$ </td><td>-0.005</td><td>0.002</td><td></td><td>-8.05</td><td>-3.04</td><td>1.0000</td></tr></table>

Design. We again use the 31M modern RMSNorm, RoPE, SwiGLU, and GQA T5 SentencePiece WikiText-103 setting of the state-preserved and stale-state diagnostics. For each of the seven paired seeds, one 5K-step phase-1 checkpoint is cloned into five 2.5K-step branches: control state, surgery covariant, surgery all-stale, surgery deep-stale, and surgery deep-stale + Soft SAGGS. All surgery branches receive the exact $\lambda = 0 . 2$ all-pairs orbit edit. The covariant branch maps every AdamW first and second moment into the new coordinates. The all-stale branch leaves every moment in the old coordinates. The deep-stale branch maps only the shallow half of layers and leaves layers 6 through 11 stale, mimicking a partially missing optimizer-state shard. The Soft branch starts from this same partial-state condition and uses Soft SAGGS at handof and every 50 steps.

Table 14 reports the seven-seed summaries. The partial-state condition changes subsequent training despite preserving the current function: The immediate handof validation-loss change, including the initial Soft rebalance where used, is below $6 . 2 \times 1 0 ^ { - 5 }$ , but the deep-stale branch is $+ 0 . 0 5 6 \pm 0 . 0 0 7 \mathrm { P P I }$ worse than the state-preserved control over seven paired seeds, with $p _ { \mathrm { s f } } ^ { + } = 0 . 0 0 7 8$ . This partial-state penalty is essentially the same penalty as leaving all moments stale, $+ 0 . 0 5 6 \pm 0 . 0 0 8 ~ \mathrm { P P L }$ . Leaving the additional shallow-half moments stale changes mean PPL by only 0.00027; this diference measures the incremental cost of stale moments, not the entire surgery penalty. Relative to the fully covariant branch, the deep-stale condition is still +0.003 ± 0.002 PPL worse, so the partial shard mismatch is measurable but much smaller than the orbit edit itself under the practical protocol.

Soft SAGGS again removes the penalty. The deep-stale-minus-Soft contrast is +0.062 ± 0.008 PPL, with all seven paired seeds in the same direction, and Soft is −0.005 ± 0.002 PPL relative to the control. Together with the reset and stale-state comparisons, these results show that periodic rebalancing recovers control-level PPL across the tested optimizer-state conditions.

## G ADDITIONAL EXPERIMENTS

These experiments evaluate numerical identities, naturally occurring imbalance, and transfer across architectures, corpora, and model sizes. Seed counts and method variants are stated for each protocol.

## G.1 Gradient Law and Orbit Imbalance

## G.1.1 Gradient-Law Verification

We evaluate the gradient-norm scaling implied by Eq. 2 on the same batch at orbit-related points, using FP32, or BF16 at 1.2B parameters. The exact pairs agree with the predicted scaling within floating-point error, whereas GELU MLP pairs show substantially larger deviations. Appendix G.3 defines the residual and slope measurements. Table 15 reports their maxima.

Table 15: Maximum relative gradient-norm residual $\varepsilon _ { \mathrm { r e l } }$ (Eq. 26) in FP32, by depth, tilt, and pair type. GELU MLPs are included as a non-exact comparison.
<table><tr><td>Pair</td><td>L</td><td> $\lambda = 0 . 0 1$ </td><td> $\lambda = 0 . 0 5$ </td><td> $\lambda = 0 . 1$ </td></tr><tr><td rowspan="3">Q-K (ReLU)</td><td>12</td><td> $4 . 7 \times 1 0 ^ { - 7 }$ </td><td> $2 . 7 \times 1 0 ^ { - 7 }$ </td><td> $2 . 4 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>24</td><td> $7 . 4 \times 1 0 ^ { - 5 }$ </td><td> $4 . 9 \times 1 0 ^ { - 5 }$ </td><td> $1 . 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>48</td><td> $1 . 8 \times 1 0 ^ { - 4 }$ </td><td> $2 . 2 \times 1 0 ^ { - 4 }$ </td><td> $1 . 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td rowspan="3">V-O (ReLU)</td><td>12</td><td> $1 . 2 \times 1 0 ^ { - 7 }$ </td><td> $9 . 8 \times 1 0 ^ { - 8 }$ </td><td> $1 . 0 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>24</td><td> $8 . 6 \times 1 0 ^ { - 5 }$ </td><td> $7 . 6 \times 1 0 ^ { - 5 }$ </td><td> $8 . 6 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>48</td><td> $1 . 1 \times 1 0 ^ { - 5 }$ </td><td> $4 . 7 \times 1 0 ^ { - 5 }$ </td><td> $5 . 2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td rowspan="3">MLP (ReLU)</td><td>12</td><td> $8 . 7 \times 1 0 ^ { - 8 }$ </td><td> $1 . 6 \times 1 0 ^ { - 7 }$ </td><td> $6 . 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>24</td><td> $1 . 9 \times 1 0 ^ { - 5 }$ </td><td> $6 . 1 \times 1 0 ^ { - 5 }$ </td><td> $6 . 2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>48</td><td> $5 . 7 \times 1 0 ^ { - 5 }$ </td><td> $5 . 5 \times 1 0 ^ { - 5 }$ </td><td> $7 . 4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td rowspan="3">MLP (GELU)</td><td>12</td><td> $1 . 2 \times 1 0 ^ { - 2 }$ </td><td> $7 . 0 \times 1 0 ^ { - 2 }$ </td><td> $1 . 4 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>24</td><td> $3 . 1 \times 1 0 ^ { - 2 }$ </td><td> $1 . 6 \times 1 0 ^ { - 1 }$ </td><td> $2 . 0 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>48</td><td> $6 . 3 \times 1 0 ^ { - 2 }$ </td><td> $1 . 5 \times 1 0 ^ { - 1 }$ </td><td> $1 . 1 \times 1 0 ^ { - 1 }$ </td></tr><tr><td rowspan="3">MLP (SwiGLU)</td><td>12</td><td> $1 . 6 \times 1 0 ^ { - 7 }$ </td><td> $1 . 5 \times 1 0 ^ { - 7 }$ </td><td> $1 . 1 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>24</td><td> $1 . 3 \times 1 0 ^ { - 7 }$ </td><td> $1 . 9 \times 1 0 ^ { - 7 }$ </td><td> $1 . 3 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>48</td><td> $1 . 5 \times 1 0 ^ { - 7 }$ </td><td> $1 . 1 \times 1 0 ^ { - 7 }$ </td><td> $1 . 6 \times 1 0 ^ { - 7 }$ </td></tr></table>

## G.1.2 Natural Drift

We test whether pair-scale imbalance also develops without a deliberate edit. We train three ReLU models at $\lambda = 0$ for 30K steps with two seeds: a small model with 22M parameters and $L = 1 2$ , a medium model with 101M parameters and $L = 2 4$ , and a deep model with 51M parameters and $L = 4 8 .$ . We track the maximum absolute log-ratio across layers and measure its growth from initialization to the last logged step. Every reported seed shows positive growth. Q–K pairs have the largest mean increases, ranging from 0.14 to 0.20 (Table 16). Appendix G.4 defines the measurement and its numerical stabilizer. These results establish naturally occurring imbalance; its efect on PPL is assessed by the paired training comparisons in Sec. 5.2.

## G.1.3 Additional Stress and Scale Checks

Supplementary experiments examine deeper models, SwiGLU, 354M models, and a single-seed 1.2B training run. These experiments generally show larger PPL efects under imposed rescaling than during unperturbed training, with exceptions documented in the experiments below. These comparisons complement the primary seven-seed results.

## G.2 Supporting Ablations and Replications

Three-seed replications give Vanilla-minus-Soft PPL diferences of +18.278 on OpenWebText, +21.156 at 123.8M parameters, and $+ 8 5 . 8 1 3 { \pm } 4 . 7 3 0$ at 296.7M under $\lambda = 0 . 2$ . Appendices G.14, G.15, and G.16 give the results. Their smaller seed counts support comparisons of direction and magnitude. Synthetic contraction and optimizer-state ablations provide separate diagnostics of the rebalancing operation, as reported in Appendices G.13 and F.5.

## G.3 FP32 Gradient-Law Verification

We verify the gradient transformation law of Eq. 2 by computing gradients at the original parameterization and the reparameterized configuration for the same batch. Table 15 uses FP32; the 1.2B check uses BF16 mixed precision. The verification compares gradient norms. Writing $g _ { A } : = \| \nabla _ { A } \mathcal { L } \| _ { \mathrm { F } }$ and $g _ { A } ^ { \prime } : = \| \nabla _ { A ^ { \prime } } \mathcal { L } \| _ { \mathrm { F } }$ , its relative residual is

$$
\varepsilon _ { A } : = \frac { \big \vert g _ { A } ^ { \prime } - c ^ { - 1 } g _ { A } \big \vert } { \operatorname* { m a x } \big \{ c ^ { - 1 } g _ { A } , \epsilon _ { \mathrm { n u m } } \big \} } , \qquad \varepsilon _ { B } : = \frac { \big \vert g _ { B } ^ { \prime } - c g _ { B } \big \vert } { \operatorname* { m a x } \big \{ c g _ { B } , \epsilon _ { \mathrm { n u m } } \big \} } , \qquad \epsilon _ { \mathrm { n u m } } : = 1 0 ^ { - 1 2 } .\tag{26}
$$

The reported $\varepsilon _ { \mathrm { r e l } }$ is the maximum over tested blocks and layers. This residual verifies the norm consequence of the law; small norm residuals alone do not verify gradient directions.

Table 16: Increase in maximum absolute symmetry-pair log-ratio during standard ReLU training at $\lambda = 0 .$ . The reported $G = \Delta _ { \mathrm { m a x } } ^ { ( 2 9 . 5 \mathrm { K } ) } - \Delta _ { \mathrm { m a x } } ^ { ( 0 ) }$ uses $\delta = 1 0 ^ { - 1 2 }$ as in Appendix G.4. Values are mean ± population SD over two seeds.
<table><tr><td>Model</td><td>Params</td><td>L</td><td>Q-K G</td><td> $\mathbf { V } { - } \mathbf { O } \ G$ </td><td>MLP G</td></tr><tr><td>Small</td><td>22M</td><td>12</td><td> $0 . 2 0 0 \pm 0 . 0 2 6$ </td><td> $0 . 0 6 8 \pm 0 . 0 0 4$ </td><td> $0 . 0 9 9 \pm 0 . 0 0 2$ </td></tr><tr><td>Medium</td><td>101M</td><td>24</td><td> $0 . 1 3 6 \pm 0 . 0 0 3$ </td><td> $0 . 0 4 7 \pm 0 . 0 0 2$ </td><td> $0 . 0 3 7 \pm 0 . 0 0 2$ </td></tr><tr><td>Deep</td><td>51M</td><td>48</td><td> $0 . 1 6 3 \pm 0 . 0 2 1$ </td><td> $0 . 1 2 0 \pm 0 . 0 2 6$ </td><td> $0 . 0 9 3 \pm 0 . 0 0 3$ </td></tr></table>

The reported gradient-ratio depth slopes fit $h _ { \ell , \delta _ { g } } : = \log ( g _ { B , \ell } + \delta _ { g } ) - \log ( g _ { A , \ell } + \delta _ { g } )$ against layer index, with $\delta _ { g } : = 1 0 ^ { - 1 2 }$ . The exact 2 log c shift under an orbit move applies to the unstabilized log-ratio; a fixed additive $\delta _ { g }$ can change this shift when a gradient norm is near the floor. We test across depths $L \in \{ 1 2 , 2 4 , 4 8 \}$ , tilt parameters $\lambda \in \{ 0 . 0 1 , 0 . 0 5 , 0 . 1 \}$ , and all symmetry pairs Q–K, V–O, and MLP.

For the exact symmetries, $i . e . , \mathrm { Q - K }$ and V–O across all activations, ReLU MLP, and SwiGLU value–output, the reported relative errors range from approximately $1 0 ^ { - 7 }$ to $2 . 2 \times 1 0 ^ { - 4 }$ across the reported FP32 configurations. In particular, the $L = 1 2$ ReLU $\mathrm { M L P }$ row reaches $6 . 0 \times 1 0 ^ { - 5 } \mathrm { ~ a t ~ } \lambda = 0 . 1 $ ; the largest Q–K residual is $2 . 2 \times 1 0 ^ { - 4 }$ at $L = 4 8$ . These residuals are consistent with floating-point error in an exact symmetry. GELU MLP shows substantially larger errors with a non-monotone dependence on λ in the deepest row, up to $2 \times 1 0 ^ { - 1 }$ , confirming that the deviation is dominated by GELU’s lack of exact homogeneity rather than numerical precision. In the 1.2B gradient-law verification (Table 20), the law continues to hold with relative error below 0.16% under BF16 mixed precision. The SwiGLU value–output rows have norm residuals near $1 0 ^ { - 7 }$

## G.4 Naturally Occurring Pair Imbalance

At each logging step, before the optimizer update, we track the maximum stabilized log-ratio imbalance across all layers:

$$
r _ { \ell , \delta } ^ { ( t ) } : = \log \frac { \left\| A _ { \ell } ^ { ( t ) } \right\| _ { \mathrm { F } } + \delta } { \left\| B _ { \ell } ^ { ( t ) } \right\| _ { \mathrm { F } } + \delta } , \qquad \Delta _ { \operatorname* { m a x } } ^ { ( t ) } : = \operatorname* { m a x } _ { \ell } | r _ { \ell , \delta } ^ { ( t ) } | , \qquad \delta : = 1 0 ^ { - 1 2 } .\tag{27}
$$

We record this statistic for each symmetry pair type Q–K, V–O, and MLP. This statistic is the absolute imbalance at step t. Table 16 instead reports its increase between the first and last logged states:

$$
\begin{array} { r } { G : = \Delta _ { \mathrm { m a x } } ^ { ( 2 9 . 5 \mathrm { K } ) } - \Delta _ { \mathrm { m a x } } ^ { ( 0 ) } . } \end{array}\tag{28}
$$

The reported G is a diference of maxima. The growth statistic G is distinct from both the endpoint $\Delta _ { \mathrm { m a x } } ^ { ( \mathrm { 2 9 . 5 K } ) }$ and the maximum layerwise change max<sub>ℓ</sub> $| r _ { \ell , \delta } ^ { ( 2 9 . 5 \mathrm { K } ) } - r _ { \ell , \delta } ^ { ( 0 ) } |$

For positive norms $a : = \| A \| _ { \mathrm { F } }$ and $b : = \| B \| _ { \mathrm { F } }$ , the ideal log-ratio $r : = \log ( a / b )$ used in the theory satisfies $r _ { \delta } - r = \log ( 1 + \delta / a ) - \log ( 1 + \delta / b )$ . Thus the stabilized measurement approximates r when both norms are large relative to δ, but the stabilized measurement does not obey the exact orbit-shift or contraction identities near the floor. Experimental references to measured log-ratios use this numerical convention. Since initialization is not exactly balanced, a nonzero endpoint alone does not establish growth from initialization; positive G does establish an increase in the maximum absolute imbalance. We train for 30,000 steps with two seeds, context length 256, 2K-step warmup, and logging every 500 steps. Peak learning rate is $6 \times 1 0 ^ { - 4 }$ for the small and deep models and $3 \times 1 0 ^ { - 4 }$ for the medium model. The last logged drift state is at step 29.5K, before the final 500 updates.

The log-ratio increment depends on parameter and update inner products, as in Eq. 23; diferent gradient magnitudes alone do not determine the direction or persistence of drift.

![](images/e931188352a0f2e431fa300778c360ae2bb9d7f80676a9a089e3d8269f2fc2a1.jpg)  
Figure 2: Growth G in maximum absolute pair log-ratio from step 0 to 29.5K during standard ReLU training. Q–K pairs show the largest mean growth. The ordering of MLP and V–O difers across model configurations. Error bars show ±1 population SD over two seeds.

## G.5 Pair Imbalance in a Pre-Trained Model

Motivation. The natural-drift experiment measures changes in pair imbalance during training from scratch. Here we inspect a released GPT-2 checkpoint to determine whether its learned attention pairs have unequal norms.

Checkpoint measurement. We load GPT-2 small (Radford et al., 2019), with 124M parameters, $d = 7 6 8$ $L = 1 2 ,$ and GELU, from Hugging Face and measure the symmetry pair-norm balance. The recorded maximum absolute log-ratios are 0.221 for Q–K and 0.931 for V–O, so V–O has the larger imbalance in this checkpoint. These pre-fine-tuning measurements use stabilizer $\delta : = 1 0 ^ { - 8 }$ . Without the corresponding initial norms, these endpoint measurements do not determine how much imbalance accumulated during pre-training. We report the attention-pair maxima; GELU up–down rescaling is not an exact symmetry. Exact attention-pair edits of this biased model also require the bias co-transforms stated in Appendix B.1. These observations motivate studying attention-pair canonicalization during fine-tuning.

## G.6 Deep-Model Sensitivity to Rescaling

Motivation. We test whether rebalancing protects deep Transformer training from deliberate exact Q–K orbit tilts. The deepest layer’s scale factor is $c _ { 4 7 } : = e ^ { 4 7 \lambda }$ ; the gradient-coordinate identity alone does not predict the resulting PPL or a critical value of λ.

Table 17: Deep ReLU model with $L = 4 8$ under Q–K rescaling over 30K steps. Values are best validation $\mathrm { P P L } ,$ reported as mean ± population SD over two seeds; $\Delta$ denotes Vanilla minus SAGGS+P.
<table><tr><td>λ</td><td> $c _ { 4 7 }$ </td><td>Vanilla</td><td> $\mathbf { S } \mathbf { A } \mathbf { G } \mathbf { G } \mathbf { S } { + } \mathbf { P }$ </td><td> $\Delta$ </td></tr><tr><td>0.0</td><td>1</td><td> $3 0 . 3 4 \pm 0 . 0 2$ </td><td> $3 0 . 2 7 \pm 0 . 0 1$ </td><td> $+ 0 . 0 7$ </td></tr><tr><td>0.2</td><td> $\approx 1 . 2 \times 1 0 ^ { 4 }$ </td><td> $6 1 . 4 6 \pm 2 . 9 9$ </td><td> $3 0 . 3 1 \pm 0 . 0 4$ </td><td> $+ 3 1 . 1 5$ </td></tr><tr><td>0.3</td><td> $\approx 1 . 3 \times 1 0 ^ { 6 }$ </td><td> $1 5 0 . 5 8 \pm 0 . 9 6$ </td><td> $3 0 . 3 5 \pm 0 . 0 6$ </td><td> $+ 1 2 0 . 2 3 $ </td></tr></table>

Table 18: 354M GELU model with Q–K rescaling over 15K steps. Values are best validation PPL, with ∆ denoting Vanilla minus ${ \mathrm { S A G G S } } { \mathrm { + P } } ;$ gradient slopes are seed means at the last logged state, step 14.5K. The row marked <sup>†</sup> uses three seeds, with PPL as mean ± sample SD; the other rows use a single seed.
<table><tr><td></td><td></td><td colspan="3">PPL</td><td colspan="2">Gradient slope</td></tr><tr><td>λ</td><td> $c _ { 2 3 }$ </td><td>Vanilla</td><td>SAGGS+P</td><td>∆</td><td>Vanilla</td><td>SAGGS+P</td></tr><tr><td>0.0</td><td>1</td><td>23.53</td><td>23.59</td><td>-0.06</td><td>0.018</td><td>0.020</td></tr><tr><td>0.1</td><td>≈10</td><td>23.89</td><td>23.56</td><td>+0.33</td><td>0.097</td><td>0.018</td></tr><tr><td>0.2†</td><td>≈100</td><td> $2 5 . 8 4 \pm 0 . 0 4$ </td><td> $2 3 . 5 8 \pm 0 . 0 6$ </td><td>+2.26</td><td>0.187</td><td>0.017</td></tr></table>

Setup. We train the deep ReLU model with $d = 2 5 6$ and $L = 4 8$ on WikiText-103 for 30,000 AdamW steps with two seeds, applying Q–K rescaling at $\lambda \in \{ 0 , 0 . 2 , 0 . 3 \}$ . The recipe uses peak learning rate $3 \times 1 0 ^ { - 4 }$ , batch size 64, context length 256, and 2K-step warmup. We compare Vanilla against SAGGS+P, with rebalancing every 100 steps and auxiliary-penalty weight 0.01.

Results. Table 17 shows the results. At λ = 0.2, where $c _ { 4 7 } \approx 1 . 2 \times 1 0 ^ { 4 }$ , Vanilla PPL approximately doubles from 30.34 to 61.46, while SAGGS+P maintains near-baseline PPL of 30.31. At λ = 0.3, where $c _ { 4 7 } \approx 1 . 3 \times 1 0 ^ { 6 }$ Vanilla reaches PPL 150.58, while SAGGS+P achieves 30.35, close to the unperturbed baseline in this two-seed stress test. The PPL gap of +120.23 at $\lambda = 0 . 3$ is a stress-test result: Orbit rebalancing removes the injected symmetry-coordinate distortion.

## G.7 Rescaling a 354M-Parameter Model

Motivation. We test whether the protection observed in the deep-model stress experiment also occurs at 354M parameters. The experiment compares unperturbed training with exact Q–K reparameterization at $\lambda \in \{ 0 . 1 , 0 . 2 \}$ under the same training recipe.

Setup. We train the xlarge GELU model with $d = 1 0 2 4$ and $L = 2 4$ for 15,000 AdamW steps on a single NVIDIA H100 GPU, applying Q–K rescaling at $\lambda \in \{ 0 , 0 . 1 , 0 . 2 \}$ . The sequence length is 512, and batch size 8 with 4× gradient accumulation gives an efective batch size of 32. SAGGS+P applies only to Q–K and V–O pairs, where the bilinear symmetry is exact regardless of activation, with interval 200 and auxiliary penalty weight 0.005. Warmup is 1.5K steps. The reported PPL is the best validation value; gradient slopes use the last logged state at 14.5K.

Results. Table 18 reports one seed at λ = 0 and 0.1, and three seeds at λ = 0.2. At λ = 0, the PPL values are 23.53 and 23.59, a 0.06 increase for SAGGS+P in this single-seed comparison.

$\mathrm { A t } ~ \lambda = 0 . 1$ , where $c _ { 2 3 } = e ^ { 0 . 1 \times 2 3 } \approx 1 0$ , Vanilla achieves 23.89 while SAGGS+P maintains 23.56, giving $\mathrm { a + 0 . 3 3 }$ PPL gap at this scale factor. $\mathrm { A t } ~ \lambda = 0 . 2$ , where $c _ { 2 3 } = e ^ { 0 . 2 \times 2 3 } \approx 1 0 0$ , Vanilla PPL degrades to $2 5 . 8 4 \pm 0 . 0 4$ over 3 seeds, $\mathrm { a + 2 . 3 1 }$ increase over the 23.53 Vanilla baseline, while SAGGS+P maintains $2 3 . 5 8 \pm 0 . 0 6$ , essentially unchanged from the unperturbed case. All three seed-paired diferences favor SAGGS+P, with small dispersion in these observed runs. The final logged gradient slope is 0.187 for Vanilla at $\lambda = 0 . 2$ and 0.017 for SAGGS+P.

Table 19: Single-seed SwiGLU MLP rescaling: best validation PPL over 30K steps, with MLP pair imbalance and gradient-ratio depth slope measured at the last logged state, step 29.5K.
<table><tr><td>λ</td><td>Method</td><td>PPL</td><td>MLP imbalance</td><td>Gradient slope</td></tr><tr><td colspan="5">Deep, L = 48</td></tr><tr><td>0.0</td><td>Vanilla (baseline)</td><td>28.48</td><td>0.052</td><td>0.001</td></tr><tr><td></td><td>SAGGS+P</td><td>28.42</td><td> $< 1 0 ^ { - 4 }$ </td><td>0.001</td></tr><tr><td>0.2</td><td>Vanilla</td><td>63.24</td><td> $8 . 7 0$ </td><td>0.212</td></tr><tr><td></td><td> $\mathrm { S A G G S { + } P }$ </td><td>28.46</td><td> $< 1 0 ^ { - 4 }$ </td><td>0.0002</td></tr><tr><td colspan="5">Small,  $L = 1 2$ </td></tr><tr><td>0.0</td><td>Vanilla</td><td>31.79</td><td>0.061</td><td>0.015</td></tr><tr><td></td><td> $\mathrm { S A G G S { + } P }$ </td><td>31.88</td><td> $< 1 0 ^ { - 4 }$ </td><td>0.015</td></tr><tr><td>0.1</td><td>Vanilla</td><td>31.98</td><td> $_ { 0 . 1 9 3 }$ </td><td>0.026</td></tr><tr><td></td><td> $\mathrm { S A G G S { + } P }$ </td><td>31.80</td><td> $< 1 0 ^ { - 4 }$ </td><td>0.011</td></tr><tr><td>0.2</td><td>Vanilla</td><td>32.48</td><td> $0 . 7 1 8$ </td><td>0.071</td></tr><tr><td></td><td>SAGGS+P</td><td>31.77</td><td> $< 1 0 ^ { - 4 }$ </td><td>0.012</td></tr><tr><td>0.3</td><td>Vanilla</td><td>33.37</td><td>1.694</td><td>0.165</td></tr><tr><td></td><td> $\mathrm { S A G G S { + } P }$ </td><td>31.85</td><td> $< 1 0 ^ { - 4 }$ </td><td>0.009</td></tr></table>

## G.8 SwiGLU Sensitivity to Rescaling

Motivation. The gradient-law verification in Appendix G.3 confirmed SwiGLU’s exact V–O symmetry with the nonlinear gate held fixed. SwiGLU is used in modern Transformer architectures, motivating a test of this activation. We vary the rescaling strength for SwiGLU, following the ReLU comparison in the deep-model stress test.

Setup. We compare two SwiGLU models with d = 256 and d = 1024: a 12-layer model with 25.5M parameters and a 48-layer model with 63.3M parameters. Rescaling acts on the MLP value–output pair, with $\lambda \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 3 \}$ for the smaller model and $\lambda \in \{ 0 , 0 . 2 \}$ for the deeper model. Each configuration uses 30,000 WikiText-103 training steps and a single seed. The MLP-only runs use learning rate $6 \times 1 0 ^ { - 4 }$ at $L = 1 2$ and $3 \times 1 0 ^ { - 4 }$ at $L = 4 8 .$ , context 256, batch 64, and 2K-step warmup. The table reports best validation $\mathrm { P P L } ;$ pair imbalance and gradient slope use the last logged state at step 29.5K.

Results. Table 19 reports the results.

Deep model, $L = 4 8$ . At λ = 0.2, where $c _ { 4 7 } = e ^ { 0 . 2 \times 4 7 } \approx 1 . 2 \times 1 0 ^ { 4 }$ , Vanilla PPL degrades severely to 63.24 with an MLP log-ratio imbalance of 8.70 and gradient slope 0.212, while SAGGS+P maintains 28.46 PPL with pair imbalance below $1 0 ^ { - 4 }$ and slope 0.0002. Its PPL is close to the λ = 0 baseline of 28.48, recovering nearly all of the degradation caused by the imposed rescaling. SAGGS+P at λ = 0 achieves 28.42, compared with the Vanilla baseline of 28.48, a 0.06 PPL diference in this single-seed protocol.

Small model, L = 12. The lambda sweep reveals a smooth degradation curve: Vanilla PPL increases from 31.79 at λ = 0 to 31.98 at λ = 0.1 to 32.48 at λ = 0.2 to 33.37 at λ = 0.3, with final logged MLP imbalance growing from 0.061 to 0.193 to 0.718 to 1.694 across the four tilts. SAGGS+P consistently maintains baseline performance across all λ values: 31.80 at λ = 0.1, 31.77 at λ = 0.2, and 31.85 at λ = 0.3, all with pair imbalance below $1 0 ^ { - 4 }$ . At λ = 0.3, the increase above the unperturbed Vanilla baseline of 31.79 is +1.58 PPL for Vanilla, which reaches 33.37, and +0.06 for SAGGS+P.

The +34.78 PPL gap between Vanilla and SAGGS+P at λ = 0.2 in the deep model shows substantial sensitivity to the exact SwiGLU value–output reparameterization in this training recipe. Figure 3 plots the corresponding validation-PPL curves at both depths.

![](images/5317a07a940143fe94a833cc20ee0b0d31e005d79611111c57dd29d4a29815c1.jpg)

![](images/c33dd148cbe9898eb23ea7aad947ec26a9daaed3886072931b119aecb691b4f7.jpg)  
Figure 3: Single-seed SwiGLU MLP rescaling at (a) L = 12 and (b) $L = 4 8 \colon$ best validation PPL for Vanilla and $\mathrm { S A G G S { + } P }$ (Table 19). The deeper model uses only $\lambda \in \{ 0 , 0 . 2 \}$ . Vanilla markers shift left and $\mathrm { S A G G S { + } P }$ markers shift right by 0.007 for visibility; the bracket in (b) gives Vanilla minus SAGGS+P at $\lambda = 0 . 2$

Table 20: 1.2B-parameter gradient law verification in BF16 mixed precision. Values are the maximum relative error across all 22 layers for each symmetry pair type and λ. All errors are below 0.16%.
<table><tr><td rowspan="2">Pair type</td><td colspan="3">Maximum relative error (%)</td></tr><tr><td> $\lambda = 0 . 0 5$   $\lambda = 0 . 1$ </td><td> $\lambda = 0 . 2$ </td><td> $\lambda = 0 . 3$ </td></tr><tr><td>Q-K</td><td>0.152 0.118</td><td>0.129</td><td>0.146</td></tr><tr><td>V-0</td><td>0.110 0.107</td><td>0.110</td><td>0.156</td></tr><tr><td>MLP (SwiGLU)</td><td>0.030 0.042</td><td>0.038</td><td>0.054</td></tr></table>

## G.9 1.2B-Parameter Gradient Law Verification

Motivation. We measure gradient-law residuals at billion-parameter scale under BF16 mixed precision.

Setup. We construct a 1.22B-parameter SwiGLU Transformer with $d = 2 0 4 8 , L = 2 2 , n _ { \mathrm { h e a d s } } = 1 6$ , and $d _ { \mathrm { f f } } = 5 5 0 4$ , with gradient checkpointing, and measure gradient norms at a single minibatch under BF16 mixed precision. For each of the symmetry pair types Q–K, V–O, and MLP, and $\lambda \in \{ 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 3 \}$ , we reparameterize from a common baseline state and compare the measured gradient norms against the theoretical predictions $\left\| \nabla _ { A _ { \ell } ^ { \prime } } \mathcal { L } \right\| = c _ { \ell } ^ { - 1 } \left\| \nabla _ { A _ { \ell } } \mathcal { L } \right\| \mathrm { ~ a n d ~ } \left\| \nabla _ { B _ { \ell } ^ { \prime } } \mathcal { L } \right\| = c _ { \ell } \left\| \nabla _ { B _ { \ell } } \mathcal { L } \right\|$

Results. Table 20 reports the maximum relative error across all layers for each pair type and λ value. Across all 12 configurations, the maximum relative error remains below 0.16%, with the MLP SwiGLU value–output pair achieving particularly precise agreement below 0.055%. $\mathrm { A t } \ \lambda = 0 . 3$ , the deepest layer’s scale factor is $c _ { 2 1 } = e ^ { 0 . 3 \times 2 1 } \approx 5 4 5$ . The extreme coordinate multiplier ratio is $c _ { 2 1 } ^ { 2 } = e ^ { 1 2 . 6 } \approx 2 . 9 7 \times 1 0 ^ { 5 } ;$ ; the actual gradient-norm spread also depends on the baseline gradients. These measurements show that the norm-scaling identity remains numerically accurate in this billion-parameter BF16 configuration.

Table 21: Single-seed rescaling-strength sweep at 101M parameters and $L = 2 4$ over 30K steps. Here $\Delta$ is SAGGS+P minus Vanilla. The $\lambda = 0 . 3$ diference is −2.9954 before rounding.
<table><tr><td> $\lambda$ </td><td>Vanilla PPL</td><td>SAGGS+P PPL</td><td> $\Delta$ </td></tr><tr><td>0.1</td><td>22.91</td><td>22.90</td><td>-0.01</td></tr><tr><td>0.2</td><td>23.24</td><td>22.89</td><td>-0.35</td></tr><tr><td>0.3</td><td>25.91</td><td>22.92</td><td>-3.00</td></tr></table>

## G.10 Rescaling Strength at 101M Parameters

Motivation. We test whether the advantage of SAGGS+P on the medium model with d = 512, $L = 2 4$ , and 101M parameters grows at higher λ values where the scale factor $c _ { 2 3 } = e ^ { 2 3 \lambda }$ becomes large.

Setup. We compare Vanilla and SAGGS+P on the medium ReLU model with Q–K rescaling at $\lambda \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ Training uses a single seed, 30,000 WikiText-103 steps, learning rate $\eta = 6 \times 1 0 ^ { - 4 }$ , context length 512, batch size 32, and a 2K-step warmup. SAGGS+P uses rebalancing every 100 steps and auxiliary-penalty weight 0.01.

Results. Table 21 reports the results. At $\lambda = 0 . 1$ , where c ≈ 10, SAGGS+P and Vanilla difer by 0.01 PPL, reaching 22.90 and 22.91, respectively. $\mathrm { A t } \ \lambda = 0 . 2$ , where $c _ { 2 3 }$ ≈ 100, SAGGS+P begins to show benefit: SAGGS+P reaches 22.89 and Vanilla reaches 23.24, so $\Delta = - 0 . 3 5$ . At λ = 0.3, where $c _ { 2 3 }$ ≈ 1000, the gap widens substantially: SAGGS+P reaches 22.92 and Vanilla reaches 25.91, so $\Delta = - 3 . 0 0$ . SAGGS+P at $\lambda = 0 . 3$ remains within 0.01 PPL of the $\lambda = 0 . 1$ Vanilla reference of 22.91, despite the roughly thousand-fold rescaling of the deepest pair.

## G.11 Multi-Seed Comparisons Across Scales

Motivation. We assess consistency across random seeds at 22M, 101M, and 354M parameters, using two or three paired seeds at $\lambda = 0$ and $\lambda = 0 . 2 .$

Setup. We compare Vanilla and SAGGS+P at $\lambda \in \{ 0 , 0 . 2 \}$ in three configurations. The 22M ReLU model uses $L = 1 2 .$ , context length 256, and 3 seeds. The 101M ReLU model uses $L = 2 4$ , context length 512, 3 seeds at $\lambda = 0 . 2 .$ , and 2 seeds at $\lambda = 0 .$ . The 354M GELU model uses $L = 2 4$ , context length 512, 2 seeds, and gradient accumulation over 4 micro-batches for an efective batch size of 32. The 22M and 101M models train for 30K steps; the 354M model trains for 20K steps. The medium and 354M comparisons use Q–K-only tilts, learning rate $6 \times 1 0 ^ { - 4 }$ , 2K-step warmup, rebalancing every 100 steps, and auxiliary-penalty weight 0.01. In particular, this 354M recipe uses a higher learning rate and more frequent rebalancing than the 354M stress test in Appendix G.7.

Results. Table 22 reports the multi-seed validation. At λ = 0, SAGGS+P shows negligible diference from Vanilla across these scales, with absolute mean diferences below 0.05 PPL.

At λ = 0.2, SAGGS+P protects the medium and 354M models, while the small-model comparison favors Vanilla. The medium-scale gap of $\Delta = - 0 . 3 8 \pm 0 . 0 5$ is consistent with the 101M rescaling-strength sweep (Table 21), and the 354M gap of $\Delta = - 1 . 1 7 \pm 0 . 0 1$ has a large paired efect size in this protocol. This 20K-step cross-scale protocol is distinct from the 15K-step 354M stress experiment (Table 18), where the same mechanism yields a larger +2.26 PPL gap; the comparisons agree in sign and difer in horizon and recipe.

Table 22: Multi-seed validation across scales. Rows marked <sup>†</sup> use two seeds; the others use three. ∆ is SAGGS+P minus Vanilla; paired SD uses within-seed diferences. Values are mean ± SD: Medium and 354M entries use sample SD over the complete specified seed sets, while small-model entries retain their original recorded summaries.
<table><tr><td>Scale</td><td>λ</td><td>Vanilla PPL</td><td> $\mathbf { S A G G S + P \Phi + P P P L }$ </td><td> $\Delta$ </td></tr><tr><td rowspan="2">Small, 22M</td><td>0</td><td> $3 4 . 3 9 \pm 0 . 0 8$ </td><td> $3 4 . 3 8 \pm 0 . 0 8$ </td><td> $- 0 . 0 0 \pm 0 . 0 6$ </td></tr><tr><td>0.2</td><td> $3 4 . 2 0 \pm 0 . 0 4$ </td><td> $3 4 . 3 0 \pm 0 . 0 6$ </td><td> $+ 0 . 1 1 \pm 0 . 0 5$ </td></tr><tr><td rowspan="2">Medium, 101M</td><td>0†</td><td> $2 2 . 8 4 \pm 0 . 0 5$ </td><td> $2 2 . 8 8 \pm 0 . 0 4$ </td><td> $+ 0 . 0 4 2 \pm 0 . 0 0 3$ </td></tr><tr><td>0.2</td><td> $2 3 . 2 5 \pm 0 . 0 3$ </td><td> $2 2 . 8 7 \pm 0 . 0 3$ </td><td> $- 0 . 3 8 \pm 0 . 0 5$ </td></tr><tr><td rowspan="2">354M</td><td>0†</td><td> $2 0 . 6 6 \pm 0 . 0 3$ </td><td> $2 0 . 6 7 \pm 0 . 0 4$ </td><td> $+ 0 . 0 1 2 \pm 0 . 0 0 6$ </td></tr><tr><td> $0 . 2 ^ { \dagger }$ </td><td> $2 1 . 8 2 \pm 0 . 0 2$ </td><td> $2 0 . 6 5 \pm 0 . 0 3$ </td><td> $- 1 . 1 7 \pm 0 . 0 1$ </td></tr></table>

Table 23: Single-seed 1.2B-parameter SwiGLU training over 10K steps with DDP on 4×H100 GPUs. Gradient norms are measured before clipping at 1.0.
<table><tr><td rowspan="3">Metric</td><td> $\lambda = 0$ </td><td colspan="2"> $\lambda = 0 . 2$ </td></tr><tr><td>Vanilla</td><td>Vanilla</td><td>SAGGS</td></tr><tr><td>Final validation PPL</td><td>23.81</td><td>221.24</td><td>23.70</td></tr><tr><td>Best validation PPL</td><td>23.81</td><td>221.24</td><td>23.70</td></tr><tr><td>Mean gradient norm</td><td>0.68</td><td>27.79</td><td>0.68</td></tr><tr><td>Q-K max-abs log-ratio</td><td>0.032</td><td>4.222</td><td> $0 . 5 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>V-O max-abs log-ratio</td><td>0.033</td><td>4.117</td><td> $1 . 2 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>MLP max-abs log-ratio</td><td>0.037</td><td>4.138</td><td> $0 . 4 3 \times 1 0 ^ { - 4 }$ </td></tr></table>

## G.12 1.2B-Parameter Training

Motivation. The 1.2B gradient-law verification in Appendix G.9 confirmed the gradient transformation law via a single forward-backward pass. A separate empirical issue is whether the sensitivity seen in smaller-scale training persists at billion-parameter scale. To test this persistence, we train a 1.22B-parameter SwiGLU model for 10,000 optimizer steps using DDP across 4×H100 GPUs.

Setup. The SwiGLU Transformer has d = 2048, L = 22, $n _ { \mathrm { h e a d s } } = 1 6 .$ , and $d _ { \mathrm { f f } } = 5 5 0 4$ , with gradient checkpointing and BF16 mixed precision. We train on WikiText-103 using AdamW with $\eta = 3 \times 1 0 ^ { - 4 } , \beta = ( 0 . 9 , 0 . 9 5 )$ , weight decay 0.1, a cosine schedule with 1,000-step warmup, and gradient clipping at 1.0. A per-GPU batch size of 4 on 4 GPUs with 2 accumulation steps gives an efective batch size of 32. We compare three configurations: unperturbed Vanilla at λ = 0, Vanilla with all-pairs rescaling at λ = 0.2, and SAGGS with the same rescaling and rebalancing every 100 steps.

Results. Table 23 reports the results. The 1.2B baseline achieves PPL = 23.81 after the specified 10K-step training budget. At λ = 0.2, where $c _ { 2 1 } = e ^ { 0 . 2 \times 2 1 } \approx 6 6 . 7$ and the all-pairs reparameterization afects Q–K, V–O, and MLP simultaneously, Vanilla PPL severely degrades to 221.24, a 9.3× increase. The training loss plateaus above 2.7, compared with a baseline of 1.5, and does not recover within the 10K-step budget. Before clipping at 1.0, the mean gradient norm reaches 27.8, compared with a baseline of 0.68, with a peak of about 4,700. The final maximum absolute log-ratios are 4.22 for Q–K, 4.12 for V–O, and 4.14 for MLP.

In this single-seed 10K-step run, SAGGS protects against this degradation, achieving PPL = 23.70, slightly better than the baseline, while keeping the final maximum absolute log-ratio below $1 . 3 \times 1 0 ^ { - 4 }$ for every pair type. V–O is largest at $1 . 2 3 \times 1 0 ^ { - 4 }$ . Figure 4 shows how Q–K and MLP imbalance evolve during this run. This single-seed comparison extends the observed sensitivity to rescaling to billion-parameter training.

Interpretation. Together with the 1.2B gradient-law verification (Appendix G.9), this run shows that both the structural symmetry and the associated optimizer sensitivity appear at billion-parameter size.

![](images/66c7939e0f8af6e9d46ad79f6c848be6fe4e966c4a2fce6a7c786df80104a31a.jpg)  
Figure 4: Maximum absolute pair log-ratio across layers during 1.2B training: (a) Q–K and (b) MLP. The shared vertical scale is linear from 0 to $1 0 ^ { - 4 }$ and logarithmic above; the break marks the transition. Post-rebalance SAGGS samples are at most $4 . 8 \times 1 0 ^ { - 7 }$ and appear at zero. Samples align with rebalancing events and end at 9.9K; Table 23 reports the final 10K-step measurements after the intervening updates.

## G.13 Exact Contraction and Synthetic Optimizer Diagnostics

This synthetic study tests the exact disturbance-free contraction law and separately measures Lion sign changes after a soft orbit edit. The sign diagnostic concerns the sampled optimizer state.

Disturbance-free Soft SAGGS contraction. Starting from a random pair $\left( A _ { 0 } , B _ { 0 } \right)$ with controlled log-ratio $r _ { 0 } = \log ( \lVert A _ { 0 } \rVert _ { \mathrm { F } } / \lVert B _ { 0 } \rVert _ { \mathrm { F } } )$ , we apply the damped rebalancing map of Eq. 12 repeatedly without intervening optimizer updates. The contraction law in Eq. 19 predicts $r _ { K } = ( 1 - \alpha ) ^ { K } r _ { 0 }$ exactly. In double precision over $\alpha \in \{ 0 . 1 0 , 0 . 2 5 , 0 . 5 0 , 0 . 7 5 , 1 . 0 \}$ and three initial imbalances, the reported worst scored single-step relative error is $7 . 8 \times 1 0 ^ { - 4 }$ . This error is scored only when both $\left| r _ { k } \right|$ and $| ( 1 - \alpha ) r _ { k } |$ exceed $1 0 ^ { - 1 2 }$ ; tiny denominators can amplify rounding error. For $\alpha = 1$ , the predicted residual is zero, so this relative-error score has no eligible steps. The 200-event absolute residual is zero for all three starts at $\alpha = 1$ . The theoretical factor at $\alpha = 0 . 1 0$ is $0 . 9 ^ { \bar { 2 } 0 0 } \simeq 7 . 0 6 \times 1 0 ^ { - 1 0 }$ after 200 events and $0 . 9 ^ { 1 5 0 } \simeq 1 . 3 7 \times 1 0 ^ { - 7 }$ after 150 events. These factors describe repeated applications of the damped rebalancing map; training-time balance is measured separately in the language-model experiments.

Lion update-sign response to damping. After a 500-step warmup with $\lambda = 0 . 2$ initial reparameterization, we branch the optimizer state into six parallel runs, each applying Soft SAGGS with a diferent α, and compare the immediate Lion sign update to the $\alpha = 0$ branch. The fraction of update components that flip sign relative to baseline is 0.65% at $\alpha = 0 . 1 0$ , 1.82% at α = 0.25, 3.91% at $\alpha = 0 . 5 0$ , 5.34% at $\alpha = 0 . 7 5$ , and 7.03% at $\alpha = 1 . 0 0$ The measured relationship is monotone in α and roughly linear. Hard rebalancing with $\alpha = 1$ flips about 11 times as many components as $\alpha = 0 . 1 0$ in this diagnostic; this pattern is an empirical observation about the sampled state.

Table 24: OpenWebText paired comparison with $n = 3$ seeds. The paired block is Vanilla minus Soft SAGGS with $\alpha = 0 . 5$ and reports the two-sided sign-flip p-value $p _ { \mathrm { s f } } .$
<table><tr><td></td><td colspan="2">Mean PPL</td><td colspan="5">Paired difference</td></tr><tr><td>λ</td><td>Vanilla</td><td> $\mathrm { S o f t } , \alpha = 0 . 5$ </td><td> $\bar { \Delta }$ </td><td> $\sigma \Delta$ </td><td>t</td><td> $d$ </td><td>psf</td></tr><tr><td>0.0</td><td> $5 5 . 9 2 \pm 0 . 0 7$ </td><td> $5 5 . 9 5 \pm 0 . 0 3$ </td><td>-0.032</td><td>0.081</td><td>-0.69</td><td>-0.40</td><td>0.500</td></tr><tr><td>0.2</td><td> $7 4 . 2 8 \pm 0 . 4 1$ </td><td> $5 6 . 0 0 \pm 0 . 1 4$ </td><td>+18.278</td><td>0.437</td><td>+72.37</td><td>+41.79</td><td>0.250</td></tr></table>

Table 25: 123.8M modern-architecture scale replication, 15K AdamW steps, $n = 3 .$ The paired block is Vanilla minus Soft SAGGS with $\alpha = 0 . 5 .$
<table><tr><td></td><td colspan="2">Mean PPL</td><td colspan="5">Paired difference</td></tr><tr><td>λ</td><td>Vanilla</td><td> $\mathrm { S o f t } , \alpha = 0 . 5$ </td><td> $\bar { \Delta }$ </td><td> $\sigma _ { \Delta }$ </td><td>t</td><td> $d$ </td><td> $p _ { \mathrm { s f } } ^ { + }$ </td></tr><tr><td>0.0</td><td> $1 9 . 8 6 \pm 0 . 0 6$ </td><td> $1 9 . 8 1 \pm 0 . 0 6$ </td><td>+0.055</td><td>0.028</td><td>+3.40</td><td>+1.96</td><td>0.1250</td></tr><tr><td>0.2</td><td> $4 1 . 1 2 \pm 0 . 6 5$ </td><td> $1 9 . 9 6 \pm 0 . 0 9$ </td><td>+21.156</td><td>0.667</td><td>+54.90</td><td>+31.70</td><td>0.1250</td></tr></table>

## G.14 Cross-Corpus Replication on OpenWebText

The medium-scale paired comparisons in the 101M WikiText-103, multi-seed cross-scale, and Lion experiments all run on a single corpus of about 103M words tokenized with the GPT-2 vocabulary. We repeat the paired comparison of Vanilla and Soft SAGGS with $\alpha = 0 . 5$ , crossed with $\lambda = 0$ and $\lambda = 0 . 2$ , on OpenWebText (Gokaslan and Cohen, 2019), using a 100M-token GPT-2-tokenized subset, with the same medium-scale architecture as the 101M WikiText-103 comparisons: $L = 2 4 , d = 5 1 2 , H = 8 .$ and 101M parameters, trained with AdamW at $\eta = 3 \times 1 0 ^ { - 4 } , \beta = ( 0 . 9 , 0 . 9 5 )$ , weight decay 0.1, a cosine schedule with 1.5K warmup, gradient clipping at 1.0, batch size 32, $T = 5 1 2$ , and BF16 autocast, for 15K steps with all-pairs reparameterization. The learning rate, warmup, horizon, and use of all-pairs rather than Q–K-only tilts difer from the 30K-step recipe of the 101M WikiText-103 comparisons, and the optimizer difers from the optimizer of the Lion comparisons. The replication changes both corpus and training recipe. Each configuration uses three paired seeds.

Table 24 reports $5 5 . 9 2 \pm 0 . 0 7$ PPL for unperturbed Vanilla training and $7 4 . 2 8 \pm 0 . 4 1$ after rescaling. At $\lambda = 0 . 2 .$ the paired Vanilla-minus-Soft diference is +18.278 PPL; at λ = 0, the diference is −0.032 PPL. The protection has the same direction as the WikiText-103 comparison, but its magnitude is not directly attributable to the corpus change because the optimizer recipe, horizon, and edited pairs also difer. The two-sided sign-flip p-value is 0.25 at $\lambda = 0 . 2$ for three seeds. This experiment provides supporting evidence on another corpus under a diferent training recipe.

This OpenWebText comparison keeps the GPT-2 BPE vocabulary and the same dense Transformer without GQA or RMSNorm. The architecture and tokenizer replication in Appendix F.2 instead tests a smaller but deliberately diferent LLaMA-like stack (Touvron et al., 2023; Ainslie et al., 2023): RMSNorm, RoPE, SwiGLU, GQA, and a T5 SentencePiece tokenizer.

## G.15 123.8M Modern-Architecture Scale Replication

The 31M replication in Appendix F.2 changed the architecture and tokenizer at a scale that admits seven paired seeds. We next test the same mechanism in the 100M-parameter regime. We keep the RMSNorm, RoPE, SwiGLU, and GQA block and T5 SentencePiece tokenizer, but use a 24-layer model with $d = 6 4 0$ , ten query heads, two KV heads, SwiGLU hidden width 1728, sequence length 512, and tied embeddings, for 123.8M trainable parameters.

We use the same paired design comparing Vanilla and Soft SAGGS with $\alpha = 0 . 5$ , crossed with λ = 0 and $\lambda = 0 . 2 ,$ over three paired seeds. The optimizer is AdamW with $\eta = 2 . 5 \times 1 0 ^ { - 4 }$ , weight decay 0.1, cosine decay, 10% warmup, gradient clipping at 1.0, BF16 XPU autocast, batch size 12, sequence length 512, and 15K steps. The reparameterization and Soft SAGGS act on Q–K, V–O, and SwiGLU value–output pairs, with rebalancing every 100 steps. Training uses Intel Data Center GPU Max 1550 XPU tiles.

Table 25 reports the three-seed summaries. The 123.8M replication follows the same pattern as the 31M comparison. At λ = 0, the mean Vanilla-minus-Soft diference is $+ 0 . 0 5 5 \pm 0 . 0 2 8 ~ \mathrm { P P L }$

Table 26: 296.7M modern-architecture scale replication, 12K AdamW steps, $n = 3 .$ . The paired block is Vanilla minus Soft SAGGS with $\alpha = 0 . 5$
<table><tr><td></td><td colspan="2">Mean PPL</td><td colspan="5">Paired difference</td></tr><tr><td>λ</td><td>Vanilla</td><td> $\mathrm { S o f t } , \alpha = 0 . 5$ </td><td> $\bar { \Delta }$ </td><td> $\sigma \Delta$ </td><td>t</td><td> $d$ </td><td> $p _ { \mathrm { s f } } ^ { + }$ </td></tr><tr><td>0.0</td><td> $2 5 . 5 1 \pm 0 . 0 4$ </td><td> $2 5 . 4 2 \pm 0 . 1 0$ </td><td>+0.088</td><td>0.052</td><td>+2.96</td><td> $+ 1 . 7 1$ </td><td>0.1250</td></tr><tr><td>0.2</td><td> $1 1 1 . 4 5 \pm 4 . 8 3$ </td><td> $2 5 . 6 4 \pm 0 . 1 0$ </td><td>+85.813</td><td>4.730</td><td>+31.42</td><td>+18.14</td><td>0.1250</td></tr></table>

At $\lambda = 0 . 2 .$ , Vanilla ends at PPL 41. $1 2 \pm 0 . 6 5$ , whereas Soft SAGGS ends at $1 9 . 9 6 \pm 0 . 0 9 $ ; the paired contrast is $\bar { \Delta } = + 2 1 . 1 5 6 \pm 0 . 6 6 7$ PPL with all three seeds favoring Soft SAGGS. With $n = 3$ , the one-sided exact sign-flip p-value floor is 0.125.

## G.16 296.7M Modern-Architecture Scale Replication

The modern-architecture replications at 31M and 123.8M parameters show sensitivity to rescaling. We next test the same paired design at 296.7M parameters, increasing both the width and depth of the RMSNorm, ${ \mathrm { R o P E } } .$ SwiGLU, and GQA architecture.

Design. The model uses the same T5 SentencePiece tokenization and causal RMSNorm, RoPE, SwiGLU, and GQA block, with $d = 8 9 6$ , 14 query heads, two KV heads, SwiGLU hidden width 2432, 32 layers, tied embeddings, sequence length 512, and 296.7M trainable parameters. We train the paired grid of Vanilla and Soft SAGGS with $\alpha = 0 . 5$ , crossed with $\lambda = 0$ and $\lambda = 0 . 2$ , over three paired seeds for 12K AdamW steps on WikiText-103 with learning rate $2 . 0 \times 1 0 ^ { - 4 }$ , weight decay 0.1, 10% warmup, cosine decay, gradient clipping at 1.0, BF16 XPU autocast, batch size $\begin{array} { r } { 8 , } \end{array}$ and rebalancing every 100 steps. As in the 123.8M scale replication and checkpoint-surgery resume, the intervention acts on $\mathrm { Q { - } K , V { - } O }$ , and SwiGLU value–output pairs. The run uses all 12 Intel Data Center GPU Max 1550 XPU tiles.

Table 26 reports the three-seed summaries. At $\lambda = 0 ,$ Soft SAGGS achieves slightly lower PPL in all three paired seeds, with $\bar { \Delta } _ { \mathrm { V a n i l l a - S o f t } } = + 0 . 0 8 8 \pm 0 . 0 5 2$ . Under the same exact $\lambda = 0 . 2$ all-pairs reparameterization, Vanilla ends at $1 1 1 . 4 5 \pm 4 . 8 3 ~ \mathrm { P P L }$ , while Soft SAGGS ends at $2 5 . 6 4 \pm 0 . 1 0 \mathrm { P P L } ;$ the paired protection is $+ 8 5 . 8 1 3 \pm 4 . 7 3 0$ PPL with all three seeds favoring Soft SAGGS.

The final pair-norm statistics show balance control: Vanilla runs at $\lambda = 0 . 2$ retain large maximum absolute log-ratios, Q–K about 7.5, V–O about 5.4, and MLP about 6.4, whereas Soft SAGGS keeps all three below $2 . 5 \times 1 0 ^ { - 4 }$ . Absolute PPL is tokenizer- and budget-dependent; the contrast is the within-tokenizer, seed-paired PPL recovery under deliberate rescaling at a 300M-class modern-architecture scale.

## H FURTHER DISCUSSION

The results distinguish invariance of the current function from equivariance of an optimizer trajectory. Standard adaptive and sign-based rules can cancel gradient scales while retaining a dependence on parameter scales. Transforming moments alone does not correct the update-scaling mismatch under shared hyperparameters. Full-state canonicalization addresses this dependence at restart by bringing orbit-related checkpoints to the same numerical state.

Periodic rebalancing serves a diferent practical role: controlling pair-scale imbalance as training proceeds. Its strongest measured benefits occur after deliberately imposed rescaling, including the checkpoint-resume workflows. Imbalance also arises naturally, in both trained-from-scratch and public models, and rebalancing can be kept on throughout training to control this imbalance: A rebalance rescales each pair only as much as the pair is imbalanced, correcting a large imbalance and leaving pairs that are already close to balance almost untouched. Soft SAGGS brings the same control to Lion, whose update depends on a momentum bufer of past gradients: Soft SAGGS leaves the optimizer state unchanged and applies the correction in several smaller steps, so the update changes gradually, and training recovers from the degradation caused by an edit.

The normalization layers of a Transformer, either layer normalization or RMSNorm, do not remove the pair symmetries, because these layers act on the residual stream rather than on the paired matrices. Weight decay does not restore pair balance either, because the decay shrinks both matrices of a pair by the same factor. Global gradient clipping rescales all gradients by one common factor, set by their total norm, so the gradients enlarged by an orbit edit can shrink the gradients of all other blocks. The following subsections expand on this discussion, covering adaptive and sign-based optimizers, natural drift and weight decay, initialization, and training stability.

## H.1 Adaptive Normalization and Trajectory Equivariance

Adam and AdamW normalize gradients using per-parameter moment estimates. With covariantly transformed moments and $\epsilon = 0$ , their normalized directions are unchanged by reciprocal rescaling. Equivariance requires the parameter displacements to scale with the parameters, which a shared learning rate does not provide (Theorem 2). Theorem 5 extends this obstruction to the standard block-local rules considered here.

For ideal degree-zero rules on corresponding histories, Corollary 6 shows that the distortion in the log-ratio of relative data updates is half the distortion of gradient descent. This factor concerns the data-update term under the stated assumptions; full increments can also reflect weight decay, a fixed positive ϵ, and diferences in clipping.

Theorem 2 supplies a mechanism for a residual surgery gap even with correctly transformed moments. The depth and λ sweeps measure the training response, while the state-preserved, stale-state, and partial-state experiments (Appendices F.8, F.9, and F.10) examine the additional role of moment history.

Adaptive-optimizer qualifications. Reset or stale moments and time-varying orbit moves can break the correspondence between current gradients and past statistics. A fixed positive ϵ must also be transformed according to Theorem 2 to preserve trajectory equivariance. Global gradient clipping generally changes its common clipping multiplier under a nonuniform diagonal rescaling of the gradient. Decoupled weight decay alone commutes with a fixed diagonal rescaling when ηω is unchanged. If learning rates are transformed to restore update equivariance, the decay coeficients must also be transformed to keep this product fixed.

## H.2 Sign-Based Optimizers and Soft Rebalancing

Lion computes its update sign from a weighted combination of the current gradient and its momentum bufer, an EMA of past gradients. A positive scalar rescaling of both the gradient and the bufer leaves this sign unchanged. As in Example 3, this sign invariance produces the same numerical displacement, while a rescaled parameter trajectory requires a rescaled displacement. Sign invariance does not imply immunity to a symmetry tilt, even before accounting for clipping or a stale momentum bufer. Theorem 5 covers Lion directly as a degree-zero rule. The hard- and soft-rebalancing Lion comparisons (Appendices F.3 and F.4) measure the resulting training sensitivity: The $\lambda = 0 . 2$ tilt degrades Vanilla Lion training, and rebalancing recovers PPL in those comparisons.

The Lion state ablation in Appendix F.5 reports a higher unperturbed mean PPL and greater seed variation for hard SAGGS than for Soft or state-covariant rebalancing. Soft SAGGS limits the correction at each event by applying only an α fraction of the log-norm correction, as in Eq. 19. Optimizer updates can introduce new imbalance between events. The synthetic study in Appendix G.13 separately reports a sampled sign-response diagnostic. For sign-based optimizers, we use α = 0.5 as the default, retain the λ = 0.2 protection, and assess the unperturbed cost with paired controls at λ = 0.

## H.3 Natural Drift and Weight Decay

Pair imbalance develops during ordinary training, but the direction of that drift depends on the radial components of the optimizer updates, not just their magnitudes. Appendix C.9 states the exact defect recursion and the first-order log-ratio identity. Appendix C.10 gives the derivations. Common weight decay scales both matrices equally and therefore leaves the ratio unchanged. Hard SAGGS restores balance at each executed event, while Soft SAGGS contracts the log-ratio. The efect on validation loss depends on the subsequent training trajectory. Layer normalization preserves the internal pair identities, as shown in Appendix B.3.

## H.4 Implications for Initialization

Our analysis reveals that standard initialization schemes, e.g., Xavier and Kaiming, implicitly choose a specific point in the continuous family of forward-equivalent parameterizations. The gradient geometry at this point need not be the best one for a fixed optimizer, horizon, and regularization recipe. Specifically, if $\boldsymbol { W } _ { \ell } ^ { Q }$ and $W _ { \ell } ^ { K }$ are initialized with the same variance, as is standard, the squared Frobenius norms have the same expectation when their shapes match; independent samples are not exactly balanced. Training can then change the imbalance. Residual-output initialization and $\mathrm { G Q A }$ also give systematic norm diferences, so $\lambda = 0$ means no deliberate tilt, not exact balance of every pair.

SAGGS enforces balance at executed hard events. Soft SAGGS contracts the current log-ratio. Their efects on validation PPL are evaluated against the unperturbed training controls.

## H.5 Training Stability

Gradient geometry distortion can also afect training stability. When the reparameterization amplifies gradients of $\scriptstyle B _ { \ell }$ by $e ^ { \lambda \ell }$ while attenuating the gradients of $\pmb { A } _ { \ell }$ by $e ^ { - \lambda \ell }$ , the spread of gradient norms can change in either direction, depending on the baseline gradients. The amplified $\scriptstyle B _ { \ell }$ gradients may dominate the global norm and cause clipping to scale the current gradients of every block. This clipping leaves every nonzero pair-gradient norm ratio unchanged. For gradient descent, the clipping scales the data-update direction by the same factor, but the efect on AdamW depends on both moment histories. Theorem 2 excludes clipping, whose multiplier can change under an orbit edit. Decoupled decay is separate from this gradient clipping.

We track gradient norm spikes, defined as steps where the total gradient norm exceeds $\mu + 3 \sigma$ , across our experiments. While the absolute number of spikes does not change substantially with moderate λ under AdamW in the reported runs, the distribution of gradient mass across parameter groups shifts systematically, which can be detrimental for training eficiency. SAGGS controls pair-norm imbalance; the resulting gradient-norm distribution still depends on the training trajectory.

(a) Shape-adjusted pair log-ratio in released checkpoints; all layers; zero = equal RMS entries  
![](images/e86548d7cfbdee441d48cfe47b47b5e2eba94ab295834d3e2828a196d1c95132.jpg)

(b) Frobenius log-ratio during Pythia pre-training; equal-size pairs have matching RMS and Frobenius ratios  
![](images/0abeb070d7f44517e01c7c7812b040a6a90b818ddab025f413b05525910f6072.jpg)

![](images/1274c69138dd7f0033094adc0a6f0f14a9339716f0269c8911480271a512a394.jpg)

![](images/eab3910093009b33a9abd83a5158076b7aabf4e410561c0cec0d50add878eae9.jpg)  
Figure 5: Pair-scale variation in released checkpoints. (a) Eleven checkpoints, using the shape-adjusted log-ratio $\widetilde { r } _ { \ell } = \log ( \| A _ { \ell } \| _ { \mathrm { F } } / \| B _ { \ell } \| _ { \mathrm { F } } ) - ( 1 / 2 ) \log ( n _ { A } / n _ { B } )$ , where $n _ { A }$ and $n _ { B }$ count matrix entries. Red denotes larger root mean square (RMS) entry magnitudes in $A _ { \ell }$ and blue denotes larger RMS entry magnitudes in $B _ { \ell } ;$ zero means equal RMS, which difers from Frobenius balance for unequal-size GQA pairs. Blank SwiGLU rows denote GELU MLPs. (b) Four Pythia pre-training runs and layerwise Q–K log-ratios for 410M. Equal matrix sizes make RMS and Frobenius log-ratios coincide. Appendix I.6 gives the protocols.

## I RELEASED-CHECKPOINT CASE STUDIES

The paired experiments establish protection and recovery in controlled training runs. We complement these experiments with four case studies on public checkpoints: naturally occurring pair-scale variation, coordinatesensitive fine-tuning, the internal evolution of a Lion stress test, and token-level restart diferences on real text. These measurements are independent of the primary training runs. Appendix I.5 summarizes the case studies, and Appendix I.6 gives the protocols.

## I.1 Pair Scales in Public Checkpoints

Figure 5 extends the GPT-2 endpoint measurement of Appendix G.5 with a cross-model survey and intermediate Pythia checkpoints. The intermediate checkpoints distinguish initialization ofsets from changes accumulated during training: Q–K starts nearly balanced, but reaches a maximum absolute log-ratio of 1.86 in Pythia 410M, concentrated in deep layers. $\mathrm { P y }$ thia 1B remains within 0.04, and V–O initially moves toward balance. Drift is model- and pair-dependent. These observations show pair-scale variation during ordinary training, without measuring its efect on loss. Training also changes the function, so these checkpoints are not points on a single fixed symmetry orbit.

![](images/362815700476188ec766367bd4559b989b75f235b1045df631916a819ddb9550.jpg)  
(c) Relative movement: W<sub>Q</sub> (solid), W<sub>K</sub> (dashed)

(b) Divergence from the canonicalized trajectory  
![](images/a17156cd7622b50d4f2375431973c8e3dd264175f1695a70eefafb8855b4197f.jpg)  
(d) Q–K imbalance persists; Soft contracts it

![](images/e3e36172410d6c58a98b702c78d5cc63531fe244d2db9ec81c6f4ab87c50093a.jpg)

![](images/a257b0dc6b08eb04fe65bd3ab04d869b671cff3e7149e6ce2844254cf4ab0cc5.jpg)  
released coordinates  
released, other training-window seed

edit + Soft SAGGS (® = 0:5, every 10 steps) canonicalized at handoff (either start)

Figure 6: Fine-tuning from forward-identical GPT-2 774M checkpoints: 200 AdamW updates with fresh moments and 16,384 held-out targets. (a) Top-1 disagreement and (b) mean absolute token-NLL diference, relative to the canonicalized branch. Canonicalizing either start gives the same trajectory, shown in green. The dashed reference uses diferent sampled training windows; the reference is not an uncertainty band. (c) Net Q–K movement relative to each branch’s starting norm, a cumulative rather than one-step measurement. (d) Q–K log-ratios: layer mean and min–max band. Soft SAGGS uses $\alpha = 0 . 5$ at handof and every ten steps. Appendix I.6 gives the protocols.

## I.2 Fine-Tuning from Equivalent Checkpoints

We next branch from the released GPT-2 774M checkpoint with fresh AdamW moments (Figure 6). An exact reciprocal edit $W _ { Q } \mapsto W _ { Q } / 8 , W _ { K } \mapsto 8 W _ { K }$ preserves the current function but leads to 7.3% held-out top-1 disagreement with the canonicalized trajectory after 200 updates. Canonicalizing either starting representative gives identical subsequent predictions and token negative log-likelihoods (NLLs) at every evaluation. Soft SAGGS leaves 1.1% disagreement. Continuing from the released coordinates gives 0.65% disagreement with the same canonicalized reference. This comparison illustrates Corollary 4 with reset moments and distinguishes exact restart equivalence from the approximate recovery provided by Soft SAGGS. The reciprocal edit is imposed for this controlled comparison; trajectory diferences do not establish a long-run loss advantage.

(a) Relative first step: $W _ { Q }$ (solid), $W _ { K }$ (dashed)  
![](images/785de82414067b5a8724b297c8a1cf30f227141bf4c7a4b5d539555b2f72c89c.jpg)

![](images/7798073bf6b94797e601dd50948016462dad94555a4b3873d50843c889d6de97.jpg)

(c) Attention entropy after 400 steps  
![](images/2b31065da6b032234f14482cc9ea649e163c3e8ad7a5ffd9c1793805aadccc19.jpg)

(d) Early loss improvement reverses  
![](images/5cf13a6e6b0e2bda42c87b6e697748934e2f94b5009e90f7046bf0beee0a91c2.jpg)

$$
( c _ { \ell } = e ^ { 0 . 2 \ell }
$$

tilt + hard, every 10 steps

sign-term prediction ´ ${ \sqrt { n } } / \left\| W \right\| _ { F }$ (tilt)

Figure 7: Depth-resolved response to an exact attention-pair tilt $c _ { \ell } = e ^ { 0 . 2 \ell } \ \mathrm { i n }$ GPT-2 355M with L = 24 during 400 Lion steps. (a) First-step relative Q–K movement and the sign-term prediction $\eta { \sqrt { n } } / \left. W \right. _ { \mathrm { F } } ;$ full updates also include decay. (b) Direction change $1 - \cos ( W _ { t } , W _ { 0 } )$ for tilted $W _ { K }$ and $W _ { O }$ , and $W _ { K }$ with tilt + Soft every ten steps. Columns are 15 evaluation steps on a nonlinear time axis. (c) Attention-entropy change from handof after 400 steps, averaged over 16 fixed sequences, heads, and query positions. (d) Mean held-out NLL. Rebalancing includes handof and leaves bufers unchanged; Soft uses $\alpha = 0 . 5$ . The figure shows a single-seed stress test; protocols are in Appendix I.6.

## I.3 Depth-Resolved Evolution under Lion

Figure 7 connects the degree-zero relative-update law to the time course of a public-model stress test. The deepest key block receives a relative first step of about 1% under the imposed tilt, compared with 0.01% in released coordinates. Its cosine similarity with the starting matrix falls to 0.46 after 400 steps, accompanied by a larger attention entropy decrease in deep layers and a reversal of the early loss advantage. The cosine diagnostic is invariant to positive scalar rescalings and therefore separates directional change from the balancing operation itself. Soft SAGGS limits these changes, and hard rebalancing also performs well in this setting. The preference for damping is supported by the unperturbed Lion state ablation in Appendix F.5. Here, attention entropy and loss are measured jointly, without isolating a causal relation between the two quantities.

![](images/83e27d1e8b6a360cc6b2aec99ef9bd5603290bc29b247db3c47c8f278db07f6b.jpg)

![](images/b8e84fdfcf345da722c375c5faaeee6d3880b1e92ac393eaf322fa84ed5ad0bf.jpg)

![](images/00cd34dfe1dc0464784ca0a1813bacd40d96c92a8ad45bcc95173b540dd27021.jpg)  
Figure 8: Token-level restart diferences in GPT-2 124M. (a), (b), and (c) Outcome-independent excerpts from three WikiText-103 validation articles. Columns are ground-truth BPE tokens conditioned on preceding text; a leading dot denotes a space and @-@ is the corpus hyphen. Rows show branch-minus-control token NLL at handof or after 30 shared AdamW updates. Blue denotes increased and red denotes decreased target probability on a shared signed scale. The edit co-transforms moments; the canonical branch additionally balances weights and moments, while Soft SAGGS with α = 0.5 leaves bufers unchanged at rebalancing. Appendix I.6 gives the protocols.

## I.4 Token-Level Diferences after Restart

Figure 8 makes the full-state distinction visible at individual tokens. After a moment-covariant edit, the probability of Marshal rises from the control’s 0.0582 to 0.0628, while commander falls from 0.0242 to 0.0215 after resuming. Full-state canonicalization restores both, within finite-precision residuals. The edited branch’s mean NLL is 0.01136 nats lower on this evaluation set, illustrating that trajectory divergence can also improve predictions. Soft SAGGS leaves a small residual diference from the control; full-state canonicalization recovers the reference trajectory.

## I.5 Summary

Case studies on public checkpoints complement the paired training comparisons. The case studies show that pair imbalance also arises in ordinary training, to a degree that varies across models and pairs. In Pythia 410M, for example, the query and key matrices start with nearly equal norms, but during pre-training their norms drift apart by a factor of up to about 6 in deep layers, which corresponds to a log-norm ratio of 1.86.

The case studies also test canonical restart on a released model. We fine-tune GPT-2 774M with fresh AdamW moments, starting from the released checkpoint and from an exact Q–K edit of this checkpoint that maps $W _ { Q } \mapsto W _ { Q } / 8$ and $W _ { K } \mapsto 8 W _ { K }$ . Canonicalizing both checkpoints at restart makes the two runs identical, with the same predictions and token NLLs at every evaluation. Without canonicalization, the edited run predicts a diferent top-1 token than the canonicalized runs at 7.3% of held-out positions after 200 updates. Periodic Soft SAGGS reduces this fraction to 1.1% but not to zero. Canonical restart also works when the moments are kept rather than reset: In a GPT-2 124M case whose edit transforms the moments along with the weights, canonicalizing both weights and moments fully restores the trajectory of the unedited run.

## I.6 Protocols

The four figures in this appendix use supplementary measurements on public checkpoints, independently of the primary training runs. Training case studies use a single fixed seed, FP32, deterministic $\mathrm { P y }$ Torch operations, eager attention, and no dropout, TensorFloat-32, or gradient clipping. Within each comparison, all branches use the same fixed training batches, except for the explicitly marked alternative-sampling reference. The training runs use one GPU; the checkpoint-norm survey reads weight tensors on CPU. The case studies are descriptive, single-seed measurements, not additional multi-seed PPL comparisons. Evaluation subsets contain teacher-forced next-token targets, not sampled generations or full-validation PPL measurements.

Public checkpoint survey (Figure 5). Figure 5(a) includes GPT-2 124M, 355M, 774M and 1.5B; Pythia 6.9B; Qwen2.5 3B and 7B; Mistral 7B v0.1; Hermes-3-Llama-3.1-8B; a ViT-B/16 CIFAR-10 classifier; and the DistilBERT SST-2 classifier. All layers and available exact matrix pairs are retained, with biases excluded from norms. Fused GPT-2 and Pythia projections are split into Q, K, and V. Pythia stores these projections grouped by head. SwiGLU uses only its up–down value path. GELU MLPs are excluded. Figure 5(a) removes the shape ofset $( 1 / 2 ) \log ( n _ { A } / n _ { B } )$ , and therefore measures equal-RMS rather than equal-Frobenius balance for GQA. The measured Qwen2.5-3B-Instruct row is omitted because its log-ratios difer from the base by at most $1 . 9 \times 1 0 ^ { - 4 }$

Figure 5(b) uses public Pythia 160M, 410M, 1B and 1.4B checkpoints from step 0 through step 143000: 20 each for 160M and 410M and 11 each for 1B and 1.4B. Each training step represents 2,097,152 tokens. Pythia 2.8B was excluded because its early checkpoints contained the same already-trained weights as the endpoint, preventing an initialization comparison. Q–K starts within 0.003 of balance in all four plotted models. V–O starts near log(0.316L), consistent with the initialization ofset. The model sizes difer in architecture dimensions; no monotone size trend or performance consequence is inferred from these observations.

Fine-tuning and Lion protocols. Figures 6 and 7 use the released GPT-2 774M (openai-community/ gpt2-large) and 355M (openai-community/gpt2-medium) checkpoints, respectively. Both use batch size four, context length 128, and fixed random windows from a prefix of the WikiText-103 raw training split containing just over 600,000 tokens, with end-of-text separators. The evaluation set comprises the first 128 non-heading validation paragraphs with at least 129 GPT-2 tokens, truncated to 129 tokens each, yielding 16,384 targets.

The 774M run starts each branch with fresh AdamW moments, learning rate $2 \times 1 0 ^ { - 5 } , ( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ $\epsilon = 1 0 ^ { - 8 }$ , decay 0.1, and 200 updates at constant learning rate without warmup. Q–K reciprocal edits co-transform their biases and leave the attention scaling operation unchanged. Canonicalization and Soft operate on Q–K and V–O. Q, K, and V biases follow their weight scales, while attention output biases are untouched. Soft uses $\alpha = 0 . 5$ at handof and every ten steps without moment transformation. The alternative-sampling branch uses a diferent seed to draw new training windows. In Figure 6(a) and Figure 6(b), this branch is compared to the canonicalized branch, just like the other curves, and is a single reference, not a variability estimate. The measured handof maximum logit diference is $8 . 8 \times 1 0 ^ { - 5 }$ on the first 16 evaluation sequences, with zero top-1 disagreements over all evaluation targets. The two canonicalized branches have identical predictions and token NLLs at all 12 evaluation steps.

The canonicalized branch’s mean held-out NLL falls from 3.389 to 2.975 nats. At step 200, the reciprocal-edit, released, and Soft SAGGS branches exceed this NLL by $1 . 3 4 \times 1 0 ^ { - 2 } , 6 . 1 \times 1 0 ^ { - 4 }$ , and $2 . 3 \times 1 0 ^ { - 5 }$ nats, respectively. Median Q-to-K relative net movement is about 58 for the edited branch and near one after canonicalization.

The 355M run uses Lion with a zero-initialized momentum bufer, constant learning rate $1 0 ^ { - 5 } , \beta = ( 0 . 9 , 0 . 9 9 )$ decay 0.1 on matrices and no decay on vectors, and 400 updates without warmup. The tilt acts on $\mathrm { Q - K }$ and V–O in all 24 layers; biases follow the exact transformations above. Branches comprise released, released+Soft, tilt, tilt+Soft every ten or 100 steps, and tilt+hard every ten steps. Every rebalancing branch also corrects at handof and leaves bufers unchanged. In particular, the interval-100 Soft branch already has a smaller initial imbalance than the uncorrected tilt branch. Relative steps are measured before each periodic rebalance, and cosine changes are measured against each branch’s own post-handof matrices. The handof maximum logit residual is below 0.013 on the logit probe, with zero disagreements on all evaluation targets. For K and O, measured step norms agree with the sign-only prediction within 0.11% throughout the run; decay contributes more for the inflated Q blocks. The large imposed scale is a stress test, not an estimate of typical naturally occurring imbalance.

At layer 24, the relative first-step magnitude of $W _ { K }$ divided by the relative first-step magnitude of $W _ { Q }$ is 7,166 under the tilt and 1.03 in released coordinates. Final NLL is 3.350 for the tilted branch, 3.151 for released, 3.153 for $\mathrm { t i l t } + \mathrm { S o f t }$ , and 3.151 for tilt + hard.

Token-level restart protocol. Figure 8 uses the released GPT-2 124M model (openai-community/gpt2) and its tokenizer. We first perform 20 AdamW adaptation steps, then exactly balance attention pairs and co-transform moments to obtain a shared balanced starting state. This complete state is cloned into the control and the edited branches. The edit uses $c _ { \ell } = e ^ { 0 . 2 \ell }$ for $\ell = 0 , \ldots , 1 1$ and maps $m \mapsto S ^ { - 1 } m , v \mapsto S ^ { - 2 } v$ . The full-state branch additionally canonicalizes both weights and moments before resuming. Soft applies $\alpha = 0 . 5$ at handof and every five steps, leaving bufers unchanged at its own events. This design distinguishes full-state canonical restart from the default bufer-preserving periodic method.

All branches resume for 30 updates with batch four, context 128, constant learning rate $2 \times 1 0 ^ { - 5 }$ , AdamW betas $( 0 . 9 , 0 . 9 5 ) , \epsilon = 1 0 ^ { - 8 }$ , and decay 0.1. Fixed training windows are drawn from a prefix of the WikiText-103 raw training split containing just over 200,000 tokens. The evaluation set takes the first 64 non-heading validation paragraphs with at least 65 tokens and scores their first 64 next-token targets. Displayed cases are the first qualifying paragraphs of the first three distinct validation articles, located at source rows 3, 49, and 80, using their first 33 tokens; the initial token provides context. Selection does not depend on the branch outcomes. Original WikiText punctuation is retained. For the 96 displayed target positions, the maximum handof logit discrepancy is $2 . 2 9 \times 1 0 ^ { - 4 } \colon$ after resuming, the discrepancy is 3.08 for the edited branch and $1 . 5 3 \times 1 0 ^ { - 4 }$ for full-state canonicalization. The three passages are illustrative cases, not independent training replicates.

Across all 4,096 evaluation targets, the branches initially agree on top-1 predictions. Final disagreement with control is 3.32% with mapped moments alone, 0% with full-state canonicalization, and 0.20% with Soft SAGGS. These comparisons measure restoration of the reference trajectory, not a general quality advantage.

## J LIST OF NOTATION

Table 27 lists the main mathematical notation.

## K LIMITATIONS

Table 27: Main mathematical notation used in the paper.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td>θ  $\mathcal { L } ( \pmb \theta )$   $L , \ell$   $d , d _ { k } , d _ { \mathrm { f f } }$   $W _ { \ell } ^ { Q } , W _ { \ell } ^ { K } , W _ { \ell } ^ { V } , W _ { \ell } ^ { O }$   $W _ { \ell } ^ { 1 } , W _ { \ell } ^ { 2 }$   $W _ { \ell } ^ { \mathrm { { \bar { u p } } } } , W _ { \ell } ^ { \mathrm { { g a t e } } }$ </td><td>model parameters data loss depth and layer index hidden, head, and MLP widths attention projections ReLU MLP up and down projections SwiGLU value and gate Frobenius norm</td></tr><tr><td> $( { \overset { \cdot } { A } } , B )$   $c > 0$   $T _ { c } ( A , B )$   $S \doteq \mathrm { d i a g } ( s _ { i } )$   $\lambda$   $c _ { \ell } : = e ^ { \lambda \ell }$   $\rho _ { p }$   $c _ { p } ^ { \star }$   $R _ { \theta }$ </td><td>exact scalar symmetry pair reciprocal scale factor orbit map  $( c A , c ^ { - 1 } B )$  coordinatewise orbit map depth-schedule tilt layerwise orbit scale pair-norm ratio  $\| A \| _ { \mathrm { F } } / \| B \| _ { \mathrm { F } }$  /2 balancing scale  $\rho _ { p } ^ { \mathbf { \scriptscriptstyle 1 } }$  pair-balancing diagonal map</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td> $\mathcal { C }$ </td><td>full-state canonicalizer</td></tr><tr><td></td><td></td></tr><tr><td> $\mathcal { P } , p$ </td><td>set and index of disjoint exact pairs</td></tr><tr><td></td><td></td></tr><tr><td>Optimizer updates</td><td></td></tr><tr><td></td><td></td></tr><tr><td> $\eta _ { t }$ </td><td>learning rate</td></tr><tr><td> $\omega _ { t }$ </td><td>decoupled weight-decay coefficient</td></tr><tr><td> $\epsilon$ </td><td>Adam additive stabilizer</td></tr><tr><td> $\beta _ { 1 } , \beta _ { 2 }$ </td><td></td></tr><tr><td> $m , \hat { m }$ </td><td>Adam moment-estimate decay rates</td></tr><tr><td> $v , \hat { v }$ </td><td>first-moment estimate and bias-corrected version</td></tr><tr><td> $\tilde { U } _ { t } ^ { A } , U _ { t } ^ { B }$ </td><td>second-raw-moment estimate and bias-corrected version</td></tr><tr><td> $\Psi _ { t }$ </td><td>pair optimizer directions</td></tr><tr><td> $k$ </td><td>block-local direction map</td></tr><tr><td> $\Delta A , \Delta B$ </td><td>homogeneity degree of  $\bar { \Psi } _ { t }$ </td></tr><tr><td></td><td>parameter increments</td></tr><tr><td>Diagnostics and SAGGS</td><td></td></tr><tr><td> $\Delta _ { t }$ </td><td>balance defect  $\left\| A \right\| _ { \mathrm { F } } ^ { 2 } - \left\| B \right\| _ { \mathrm { F } } ^ { 2 }$ </td></tr><tr><td> $r _ { t }$ </td><td>pair log-norm ratio</td></tr><tr><td> $F _ { t }$ </td><td>radial forcing</td></tr><tr><td> $Q _ { t }$ </td><td></td></tr><tr><td> $T _ { \mathrm { b a l } }$ </td><td>squared-direction-norm difference</td></tr><tr><td> $\alpha$ </td><td>rebalancing interval</td></tr><tr><td></td><td>Soft SAGGS damping</td></tr><tr><td> $c _ { p } ^ { \mathrm { e f f } } ( \alpha )$ </td><td>damped orbit factor</td></tr><tr><td>Experiments</td><td></td></tr><tr><td></td><td></td></tr><tr><td> $\bar { \Delta } _ { \mathrm { P P L } }$ </td><td>mean paired PPL difference</td></tr><tr><td> $\sigma _ { \Delta }$ </td><td></td></tr><tr><td></td><td>paired SD</td></tr><tr><td> $d$ </td><td>paired Cohen&#x27;s  $d$ </td></tr><tr><td></td><td></td></tr><tr><td> $p _ { \mathrm { s f } } ^ { + }$ </td><td>one-sided sign-flip p-value</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td> $n$ </td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>number of paired seeds</td></tr></table>

The experiments evaluate a selected collection of Transformer architectures, optimizers, and language-modeling datasets. The experiments do not cover every model family or every corpus, so the empirical findings should be interpreted within this experimental scope.