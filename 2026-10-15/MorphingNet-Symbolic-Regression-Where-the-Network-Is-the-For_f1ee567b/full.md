# MorphingNet: Symbolic Regression Where the Network Is the Formula

Jincheng Gao <sup>1</sup> Jiahe Yi <sup>1</sup> Shuo Sun <sup>1</sup>

## Abstract

We introduce MorphingNet, a network for symbolic regression (SR) whose units change their function continuously. Instead of searching over expressions built from a fixed operator library, MorphingNet trains the form of each function as a parameter, so SR becomes one continuous optimization problem. Its units are either polefree rational functions or complex exponential– logarithm functions, which move between exponentials, sinusoids, damped oscillations, power laws and logarithms as their parameters change. The network is a readable formula at every step of training, and no operator library is needed to train it; a structure-recovery step afterwards rewrites the trained formula into short elementary expressions. Training is dense linear algebra, so it is GPU-friendly and scales to large data. MorphingNet achieves state-of-the-art accuracy on SRBench, recovers most generating formulas on synthetic benchmarks, and on large data is up to two orders of magnitude faster than the strongest search method. By turning the search for a formula into the training of a network, MorphingNet offers a new paradigm for SR and for scientific discovery from data.

## 1. Introduction

Symbolic regression (SR) looks for a closed-form expression $y = f ( \mathbf { x } )$ that fits data. It chooses both the structure of the expression and its constants. Because the answer is a formula, people can read it, check it against theory and reuse it. SR has recovered physical laws from measurements (Schmidt & Lipson, 2009; Udrescu & Tegmark, 2020) and extracted equations from trained networks in astrophysics (Cranmer et al., 2020). In materials science it has rediscovered the Johnson–Mehl–Avrami–Kolmogorov transformation law and a Landau free energy (Wang et al., 2019).

Most SR methods are genetic programming (GP) (Koza, 1992). GP evolves a population of expression trees built from an operator library, such as {+, ×, sin, exp}. Each generation mutates and recombines the trees and scores every one on the data. This search is slow. Its cost grows with the number of samples, and data sets of 10<sup>5</sup> to 10<sup>6</sup> samples are now common. On a million samples, Operon (Burlacu et al., 2020), one of the fastest GP systems, takes a median of 11.8 hours per problem in our experiments.

A natural idea is to run SR on a GPU, as deep learning does. This is harder than it sounds. A GPU is fast when it repeats one operation on large, uniformly shaped arrays. A GP population holds many trees of different shapes, and each tree is a different program. Recent systems still move evaluation, constant fitting and even variation onto the GPU (Zhang et al., 2022; de Vries et al., 2025; Wu et al., 2025; Mao et al., 2026). These systems make each evaluation faster, but the search itself stays discrete. Every step proposes new trees, and each must be evaluated on all the data. A GPU speeds up the arithmetic, not the number of trees the search must try. What a GPU does best is continuous optimization: one model, updated by one dense step on all the data, repeated until convergence. Deep learning scales because its training has this form, and GP does not.

So the search should become training. A line of work does exactly this by making the network itself the formula. EQL (Martius & Lampert, 2016; Sahoo et al., 2018) and its successors (Li et al., 2024; Wu et al., 2024) build networks whose units are elementary operators and train them by gradient descent. ParFam (Scholl et al., 2025) fits rational functions composed with preselected elementary functions. These are elegant steps toward SR as continuous optimization. They optimize continuously, but the functions available to their units are chosen in advance. In scientific discovery, we may not know which functions to choose. An unfamiliar relationship may have no simple expression in the chosen library, even when its shape is simple. This motivates a different approach: learn the shape of each function first, then seek a familiar symbolic description. The learned shape remains available for inspection even when no such description is found.

The Kolmogorov–Arnold Network (KAN) (Liu et al., 2025; 2024) takes this route. It places a learnable spline on every edge. Splines are flexible, and their shape is optimized continuously, like any network weight. The library is used only at the end, to replace each spline by a matching symbolic function. But a spline gains flexibility only by adding knots, and every knot is a constant. So accuracy and length grow together. Written out, a KAN is a spline soup: hundreds of piecewise constants, nested layer by layer. The symbolic matching that shortens it is approximate, so the exported formula loses accuracy.

This raises a question. Can SR be a continuous optimization problem that runs at scale, and still return a formula that people can read: a short expression, or one whose structure, such as an inner product, is visible?

Our answer is a network whose units change their function continuously. We call it MorphingNet. A unit’s function is not picked from a list. It morphs from one form into another, for example from an exponential into a damped sinusoid, as training moves its parameters. So the form of the function is learned like any weight, and the network stays a closed-form expression throughout. Like KAN, Mor phingNet learns function shapes from data, but each learned unit is itself a closed-form expression that can be plotted, inspected and then simplified. Such units need a basis as flexible as the B-spline, but one whose elements are closed form functions. We use two families. The first is the rational function. Rationals approximate far better than polynomials of the same size: they reach |x| at the root-exponential rate $e ^ { - c { \sqrt { n } } }$ where polynomials reach it at an algebraic one (New man, 1964), and Pade approximants represent analytic func-´ tions with few constants (Baker & Graves-Morris, 1996). Rational activations have already been used inside neural networks (Molina et al., 2020; Aghaei, 2024). The second family is the exponential of a complex logarithm. One for mula with a few continuous parameters covers exponentials, sinusoids, damped oscillations and power laws. Its closest relative is the soft exponential activation (Godfrey & Gash ler, 2016), which interpolates between logarithm, identity and exponential and becomes sinusoidal when its parameter is complex. Our unit composes its logarithmic and exponen tial branches, so that power laws and damped oscillations appear in the same unit. Unlike a spline, each of these functions is global: a few parameters describe it on the whole domain. Each unit first projects its inputs onto a learned direction, as in projection pursuit (Friedman & Stuetzle, 1981), and then applies a rational or complex exponential– logarithm function. Product ports let units multiply as well as add. The whole network is a smooth least-squares problem, so we train it with a second-order method and prune it with a sparse group penalty. Every training step is a dense matrix operation, so it runs on CPU and GPU alike. The trained network is already a real-valued formula and exports without approximation.

As trained, these formulas are accurate but long. No library is used during training, so a rational unit cannot write sin directly. To our surprise, the units learn approximation schemes of their own. A rational unit represents a sinusoid or an exponential by a Pade-type rational function, and´ redundant branches cancel each other. These long formulas hide short ones. This motivates a structure-recovery step that rewrites the trained formula under a fixed error bound. It replaces approximations by elementary functions and recovers inner products. A library enters only here, and its role differs from its role in GP. It does not search for the formula; the formula already exists and is accurate. The library only rewrites it in a shorter vocabulary, and users can extend it for their domain. In one case, a trained formula with 48 operations and 519 constants, fitted to a million samples in 50 dimensions, became $0 . 7 9 6 e ^ { z _ { 2 } }$ sin $z _ { 1 }$ . Here $z _ { 1 }$ and $z _ { 2 }$ are learned linear combinations of the inputs, and the 45 irrelevant inputs dropped out of them.

The results bear out all three requirements. On the 130 ground-truth problems of SRBench (La Cava et al., 2021), over ten seeds, MorphingNet reaches $R ^ { 2 } > 0 . 9 9 9$ on 96 to 97% of problems at the three lower noise levels, against at most 88% for the published methods, and on 70% at the highest level, against at most 68%. On 104 targets whose functions lie in the library, structure recovery returns the generating formula in 96 cases, with median 7 operations and 10 constants against 16 and 16 for Operon. On a million samples, MorphingNet runs in about five minutes on a GPU, 123 times faster than Operon.

## Our contributions are:

1. Morphing units. A network of projections followed by rational and complex exponential–logarithm units, joined by product ports. The form of each function is optimized continuously, and the trained network is a formula without loss (Section 3.1).

2. Accuracy. The highest accuracy among the SRBench methods on its ground-truth problems, at every noise level, with ablations that trace the gain to projection, and product ports to multiplicative targets (Sections 4.1 and 4.2).

3. Structure recovery. A post-training rewrite that shortens formulas under a fixed error bound and recovers the generating formula in most cases. A library is used only here, and can be extended by the user (Sections 3.4 and 4.3).

4. Scale. Second-order training on CPU and GPU that matches GP at small scale and is one to two orders of magnitude faster on large data (Sections 3.5 and 4.4).

## 2. Related Work

Search over expressions. Genetic programming (Koza, 1992) evolves expression trees built from an operator library. It found free-form physical laws (Schmidt & Lipson, 2009), and modern systems include PySR (Cranmer, 2023), Operon (Burlacu et al., 2020) and GP-GOMEA (Virgolin et al., 2021); the last two are among the strongest methods on SRBench (La Cava et al., 2021). AI Feynman (Udrescu & Tegmark, 2020) adds physics-inspired decompositions before the search. Neural networks guide the search (Landajuela et al., 2022), pretrained transformers propose expressions directly (Biggio et al., 2021; Kamienny et al., 2022), and large language models propose them inside an evolutionary loop (Shojaee et al., 2025). These methods return short formulas, and they return them in the vocabulary of the library. A target outside the library can only be approximated. The exact solution of the Colebrook equation for pipe friction needs the Lambert W function (Brkic´, 2011), and a GP search without W in its library returned logarithmic approximations (Praks & Brkic´, 2018). A transformer trained on recurrent sequences, asked for a Bessel function, returned its asymptotic form (d’Ascoli et al., 2022). Benchmarks build the right library in: for its ground-truth track, SRBench adds to every method the operators these problems need (La Cava et al., 2021). Every candidate is a discrete object, so constants are fitted inside a combinatorial search, and the cost of that search grows with the size of the data. MorphingNet does not search over expressions. It fits one smooth model, and a library appears only when the fitted formula is shortened. Any library from this line of work can be used at that step.

Networks of fixed operators and parametric families. EQL (Martius & Lampert, 2016; Sahoo et al., 2018) turned symbolic regression into network training. Its units are elementary operators, it is trained with a sparsity penalty, and the pruned network is the formula. DySymNet (Li et al., 2024) searches the layout of such networks with a learned controller, and PruneSymNet (Wu et al., 2024), Symbol-Net (Tsoi et al., 2025) and LIES (Montazerin et al., 2025) prune them toward compact expressions. ParFam (Scholl et al., 2025) fits rational functions composed with preselected elementary functions by global continuous optimization. TIR (de Franc¸a, 2022) evolves models of a similar rational form. MorphingNet keeps the pipeline these methods share: a network that is a formula, a sparsity penalty, pruning and refitting. What it changes is the unit. In all of these methods the elementary function of a unit is chosen from a list, either fixed in advance or picked by an outer search. The choice of function is still discrete, even when the weights are trained by gradients. In MorphingNet the function itself has continuous parameters, so no list is needed during training.

Continuous function forms. KAN (Liu et al., 2025) learns a spline on every edge, and KAN 2.0 (Liu et al., 2024) adds multiplication nodes; softly symbolified KANs mix symbolic and spline terms during training (Bagrow & Bongard, 2025). Rational activations (Molina et al., 2020; Aghaei, 2024) and the soft exponential (Godfrey & Gashler, 2016) make the shape of an activation trainable inside an ordinary network. GINN-LP (Ranasinghe et al., 2024) learns the exponents of products of powers continuously, a family limited to Laurent polynomials. Symbolic metamodels (Alaa & van der Schaar, 2019) fit Meijer G-functions, a closed-form family with continuous parameters, and Symbolic Pursuit (Crabbe et al.´ , 2020) fits sums of them over projected inputs, both to explain a trained black-box model. These methods all optimize the form of a function continuously. They differ in what they produce at the end. A spline can be written out as a formula: a piecewise polynomial with one constant per knot. With many knots across many edges, that formula is too long to read. KAN therefore replaces each spline by a matching symbolic function, and this step loses accuracy. Rational and soft-exponential activations were not designed to be exported as formulas. Symbolic Pursuit adds one ridge function at a time and has no products or composition. GINN-LP does return a formula, but only a Laurent polynomial in the raw inputs, with no projections. MorphingNet combines the three ingredients these methods have separately: closed-form units with continuous parameters, projections of the inputs, and products between units. It trains them jointly as one formula.

## 3. Method

We start with an example, highlighted in Figure 1a. Suppose the data come from $y = 2 \sin ( x _ { 1 } ) x _ { 2 } + 1$ . MorphingNet can represent this function in two layers. In the first layer, one unit reads the direction $z = w x _ { 1 }$ and computes a function close to sin z. In the second layer, a product port multiplies that unit by x<sub>2</sub>. The output head scales the product by 2 and adds 1. Every operation is an ordinary real operation, so the trained network can be written down as this formula.

The method has four steps, summarized in Algorithm 1. (1) Build a network whose units are closed-form functions with continuous parameters (Section 3.1). (2) Train it so that unneeded units become exactly zero, then prune and refit at several levels (Section 3.2). (3) Choose one formula from the resulting candidates (Section 3.3). (4) Rewrite that formula into a shorter one with the same accuracy (Section 3.4). We follow the example through all four steps.

## 3.1. The network

Figure 1a shows the full network; the dark path is the example above. Inputs are scaled to [−1, 1] by their declared ranges, $\tilde { x } = ( x - c ) / h$ . The network keeps a bank: the list of all quantities computed so far. It starts as $b ^ { ( 0 ) } = \tilde { x }$ Each layer reads the whole bank through one affine map, $z = A b ^ { \dot { ( \ell - 1 ) } } + a$ . Its outputs are of two kinds. A unit takes one port $z _ { j }$ and applies a closed-form function $h _ { j } = \varphi _ { \theta _ { j } } ( z _ { j } )$ with its own parameters $\theta _ { j } . \operatorname { A }$ product port takes two ports and multiplies them, $z _ { a } z _ { b }$ . The layer appends all unit outputs and products to the bank, so a later layer can read the raw inputs as well as every earlier result. The output is an affine function of the final bank, $\hat { y } = m + s \left( v ^ { \top } b ^ { ( \bar { L } ) } + v _ { 0 } \right)$

![](images/22714f79570c7274c3c9538766cd68f4df040d364c770d7fec664ae1d1d541d6.jpg)  
Figure 1. MorphingNet at a glance. (a) A two-layer network. Each layer reads the whole bank through an affine map; its ports feed units (blue) and product ports (orange), and their outputs are appended to the bank. Dark lines show the part that survives training on $y = 2 \sin ( x _ { 1 } ) x _ { 2 } + 1 ;$ gray parts were set to zero. (b) The complex exponential–logarithm unit J at five settings of $( \alpha , \omega , \beta )$ , computed from its definition. One formula moves continuously between these shapes.

Algorithm 1 MorphingNet. Hyperparameters and capacities   
are listed in Appendix B.   
1: for capacity c = 1, 2, 3 layers do   
2: Fit: proximal LM on the loss (4), penalty off for the   
warm-up find the structure   
3: for each threshold pair (t, u) do   
4: Mask coefficients below t and units below u; refit   
with LM one candidate per level   
5: end for   
6: Export and replay every candidate; record length and   
selection error   
7: if some candidate so far has selection error $\leq \tau$ then   
8: break capacity ladder   
9: end if   
10: end for   
11: Choose the shortest candidate on the length–error fron  
tier within max(1.1 $e _ { \operatorname* { m i n } } , \tau )$   
12: Apply rewrites (Table 1) while the error stays within   
max $( 1 . 1 e _ { 0 } , \tau )$

where m and s undo the scaling of $y .$

Each piece has a reason. Because each unit reads a learned direction rather than a single coordinate, a function such as sin $( x _ { 1 } + x _ { 2 } )$ needs one unit, not a composition of onedimensional pieces. Products give multiplication and inner products $( u ^ { \top } x ) ( v ^ { \top } x )$ in one step. The bank lets a later layer reuse an input or an earlier result without computing it again, so a formula like sin u + vw does not need a copy of u in every layer.

Rational unit (R). The rational unit is

$$
\varphi ( z ) = \frac { P ( z ) } { 1 + Q ( z ) ^ { 2 } } ,\tag{1}
$$

where P and $Q$ are polynomials of degree D. A plain ratio $P / Q$ would be more flexible, but it has poles, and a formula with a pole can divide by zero on new data. The denominator $1 + Q ^ { 2 }$ is at least one, so the unit is defined and smooth on the whole real line. It represents $1 / ( 1 + z ^ { 2 } )$ and polynomials exactly, and approximates other shapes in a Pade-like way (´ Baker & Graves-Morris, 1996). In the example, a rational unit cannot write sin z; it fits a rational approximation of it, which Section 3.4 later replaces.

Complex exponential–logarithm unit (C). The second family is built from one complex atom. Let $L _ { \beta } ( z ) = \log ( 1 +$ $\beta z ) / \beta$ , with $L _ { 0 } ( z ) = z ,$ and let $q = \alpha +$ iω be complex. The unit is

$$
J ( z ) = \frac { e ^ { q L _ { \beta } ( z ) } - 1 } { q } ,\tag{2}
$$

and both its real and imaginary parts enter the bank. Its parameters move it continuously between familiar functions:

$$
J ( z ) = \{ \begin{array} { l l } { ( e ^ { \alpha z } - 1 ) / \alpha , } & { \beta = 0 , \omega = 0 , } \\ { ( \sin \omega z + i ( 1 - \cos \omega z ) ) / \omega , } & { \beta = 0 , \alpha = 0 , } \\ { ( e ^ { \alpha z } e ^ { i \omega z } - 1 ) / q , } & { \beta = 0 , } \\ { ( ( 1 + \beta z ) ^ { q / \beta } - 1 ) / q , } & { \beta \neq 0 , } \\ { \log ( 1 + \beta z ) / \beta , } & { q  0 . } \end{array}\tag{3}
$$

From top to bottom these are an exponential, a sinusoid, a damped oscillation, a power law and a logarithm. The division by $q$ keeps the atom finite as $q \to 0 .$ , so the transitions are smooth. A library that contains these five functions represents the same shapes. The difference is how one moves between them. In a library, going from exp to a damped oscillation is a discrete swap. Here it is a small change of $\omega ,$ so the form of the function is trained by the same steps as every weight. In the example, a complex unit with $\alpha = \beta = 0$ gives sin(ωz)/ω exactly.

A run trains three capacities, with one, two and three layers (Appendix B). Units and product ports are deliberately more than needed; training removes the unused ones.

## 3.2. Training

Training has to do two things at once. It has to fit the data to high precision, because recovering an exact formula needs errors far below those usual in neural network training. And it has to make unneeded units exactly zero, so that they can be deleted from the formula. The direct approach is Adam with an $\ell _ { 1 }$ penalty, as in EQL. It meets neither goal well. Subgradient steps leave many coefficients small but nonzero, and first-order methods slow down sharply near the optimum of an ill-conditioned least-squares problem. Our networks are small, with a hundred to two thousand parameters, so we can afford a second-order method instead.

The objective is

$$
\mathcal { L } ( \theta ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \bigl ( \hat { y } _ { i } - y _ { i } \bigr ) ^ { 2 } + \lambda _ { 1 } \| \theta _ { S } \| _ { 1 } + \lambda _ { g } \sum _ { j } \| \theta _ { \mathrm { o u t } ( j ) } \| _ { 2 } ,\tag{4}
$$

with $y$ in scaled units. The $\ell _ { 1 }$ term acts on affine weights and unit coefficients. The group term acts on the outgoing weights of each unit j: every weight with which a later layer or the head reads that unit. When the group reaches zero, nothing reads the unit, and it is removed. In the example, the smallest capacity that fits has 6 units per layer and the formula needs one. The group term drives the outgoing weights of the other five to zero.

We minimize (4) with a proximal Levenberg–Marquardt (LM) method (Levenberg, 1944; Marquardt, 1963). We want a step that decreases the least-squares error quickly and lands exactly on zero where the penalty asks for it. Linearizing the residual, $r ( \theta + \delta ) \approx r + J \delta$ , gives the subproblem

$$
\operatorname* { m i n } _ { \delta } \ \frac { 1 } { 2 } \delta ^ { \top } ( J ^ { \top } J + \mu I ) \delta + g ^ { \top } \delta + \Omega ( \theta + \delta ) ,\tag{5}
$$

where $\boldsymbol { g } = \boldsymbol { J } ^ { \top } \boldsymbol { r } , \boldsymbol { \Omega }$ is the penalty and $\mu$ is the damping. The quadratic part is the usual LM model, and $\mu$ acts as a trust region: it shrinks when the predicted decrease is matched and grows when a step is rejected. The penalty part is not differentiable at zero, which is what makes exact zeros possible. We solve the subproblem with ADMM (Boyd et al., 2011), whose proximal step returns exact zeros (Appendix C).

A fit first runs 200 steps without the penalty. If the penalty were on from the start, it could remove a unit before the data had shown that the unit is needed. The fit stops when the control error reaches a declared tolerance τ, when the proximal gradient vanishes, when the control error stops improving, or after 2000 steps. The control error is measured on a separate split, not on the training data. Snapshots that change the error or the structure are saved as candidates.

Pruning and refitting. After the fit, many coefficients are exactly zero but some are small and nonzero. In the example, the first-layer port may keep a small weight on $x _ { 2 }$ (a gray edge in Figure 1a). Whether such a weight is noise or structure differs from problem to problem, so no single pruning level is right for all. We therefore prune at a grid of levels and refit each. A coefficient threshold t removes coefficients below t times the largest magnitude in their row, and a unit threshold u removes units whose group norm is below u times the largest. Both thresholds are measured on the same fit, so larger thresholds give nested masks. Each pair gives one refit, which trains the surviving coefficients with plain LM under a fixed mask. The grid has 31 refits per capacity (Algorithm 1); identical masks share one. A single refit at a middle level instead leaves fewer problems at the tolerance and longer formulas (Section 4.2).

## 3.3. Choosing the formula

Every candidate is exported and replayed: its formula is evaluated again and must agree with the network. Each candidate has a complexity and an error on a selection split, which is used for nothing else. We keep the candidates that no other candidate beats on both, the length–error frontier. On the frontier, every candidate with selection error at most max $( 1 . 1 e _ { \mathrm { m i n } } , \tau )$ counts as tied, and the tied candidate with the fewest unit operations, then the fewest constants, is chosen. The factor 1.1 treats small differences as noise. The tolerance τ treats all errors below it as equally good on noise-free data, so extra constants are not spent on accuracy that has no meaning.

Capacities are trained in order. When the fit and refits of a capacity are finished and a candidate of that or a smaller capacity has selection error at most τ, larger capacities are not trained, and their candidates do not enter selection. This capacity ladder only fires when a small network already fits to τ. In the example, a one-layer network cannot multiply a unit by $x _ { 2 } .$ , so its error stays above τ . The two-layer network reaches $\tau ,$ and the three-layer network is never trained. Within the two-layer network, the refit that removed the $x _ { 2 }$ weight and the one that kept it are both below τ. They tie, and the shorter one is chosen. On noisy data, where the error cannot reach $\tau ,$ all three capacities run. For noisy data the tie factor is applied after removing an estimate of the noise variance from the selection error (Appendix D). The test split is read only after the choice is saved.

Table 1. Rewrites used by structure recovery. Each is refitted jointly and kept only under the error bound.
<table><tr><td>Rewrite</td><td>Action</td></tr><tr><td>Unit</td><td>Replace a unit by  $A \phi ( z ) + \gamma z + d \mathrm { w i t h } \ \phi$  from the library.  $\textrm { \textbf { A C } }$  unit with  $\beta = 0$  be- comes  $K e ^ { \alpha z }$  COS  $\omega z + M e ^ { \alpha z }$  sin  $\omega z - K$  ex-</td></tr><tr><td>Merge</td><td>actly. Replace several terms that depend on the same direction by one function.</td></tr><tr><td>Quadratic</td><td>Collect products into a quadratic form and rewrite it with the fewest products by eigen-</td></tr><tr><td>Delete</td><td>decomposition. Delete terms; absorb linear terms into the bias of a product factor.</td></tr><tr><td>Swap</td><td>Swap one elementary function for another if the error halves, e.g. tanh → exp.</td></tr><tr><td>Template</td><td>If the gradient outer product has rank two, fit  $g ( z _ { 1 } ) + h ( z _ { 2 } ) \operatorname { o r } g ( { \bar { z } } _ { 1 } ) h ( z _ { 2 } )$ </td></tr></table>

## 3.4. Structure recovery

The chosen formula is accurate but long. A unit that cannot represent a function exactly approximates it, several units may share one direction, and an inner product may be spread over several product ports. Structure recovery rewrites the formula into a shorter one with the same accuracy. Its ingredients are standard: candidate replacements, a joint refit and an error check. What differs from search-based SR is the starting point. Recovery starts from a formula that already fits, so it only has to shorten it, never to find it.

Each rewrite is a transaction. It proposes a replacement, refits all remaining constants jointly with LM on the training split, and is accepted only if the selection error stays below max $( 1 . 1 e _ { 0 } , \tau )$ , where $e _ { 0 }$ is the error of the original formula, and the formula does not grow. The bound is set once from the original formula, so a sequence of rewrites cannot drift away from it. The rewrites are listed in Table 1; they are applied in rounds until none is accepted.

In the example with rational units, the Unit rewrite replaces the rational approximation by $A \sin ( z ) + \gamma z + d ;$ after the joint refit γ and d are negligible, the Delete rewrite removes them, and the formula becomes $2 \sin ( x _ { 1 } ) x _ { 2 } + 1$ . Exact canonicalizations, such as merging an affine map into its only reader, run after every step. The library is used only here. It contains the usual elementary functions and a few compositions of them (Appendix E); a user can extend it.

## 3.5. Cost

One LM step costs $O ( n p ^ { 2 } )$ for the Gram matrix $J ^ { \top } J$ and $O ( p ^ { 3 } )$ for its factorization, where n is the number of training rows and p the number of parameters. Every operation is dense and has a fixed shape, so the cost is linear in n and maps directly onto a GPU. The fit and the refits are independent and run in parallel. CPU and GPU runs select the same formulas on the same data. Appendix F describes both implementations, including an exact integer scheme that computes the Gram matrix on GPUs with weak doubleprecision arithmetic (Ozaki et al., 2012).

## 4. Experiments

The experiments answer four questions, one for each contribution. Does MorphingNet fit benchmark targets more accurately than published methods (Section 4.1)? Does the gain come from projections and products (Section 4.2)? Does structure recovery find the generating formula (Section 4.3)? How does the cost scale with data (Section 4.4)? Section 4.5 compares with the neural and parametric methods of Section 2 on a common suite.

## 4.1. SRBench ground-truth problems

We use the 130 ground-truth problems of SR-Bench (La Cava et al., 2021), 116 from the Feynman lectures and 14 from Strogatz, at the four noise levels of the benchmark and ten random splits. Each split has a training set of at most 10 000 rows, and separate control, selection and test sets. We report the fraction of problems with test $R ^ { 2 } > 0 . 9 9 9$ against the noise-free target, the standard SR-Bench accuracy criterion. Published results are those of the SRBench release.

Table 2 shows the result. At the three lower noise levels the rational network reaches $R ^ { 2 } > 0 . 9 9 9$ on 96 to $97 \%$ of the problems, against 85 to 88% for the best published methods. At the highest noise level the strong methods are within a few points of each other, between 66% and 70%. The spread over ten splits is one to three points. Every rational formula is finite on every test set. About 3% of the confluent formulas are not finite at some test points, which counts as a miss.

Table 2. Fraction of SRBench ground-truth problems with test $R ^ { 2 } > 0 . 9 9 9$ , by noise level. MorphingNet: mean ± standard deviation over ten splits. Other methods: published SRBench results; MRGP (Arnaldo et al., 2014), SBP-GP (Virgolin et al., 2019), AI Feynman 2.0 (Udrescu et al., 2020). Best in each column in bold, second underlined.
<table><tr><td>Method</td><td>0</td><td>0.001</td><td>0.01</td><td>0.1</td></tr><tr><td>MorphingNet-R MorphingNet-C</td><td>97.0±0.9 95.3±1.5</td><td> ${ \bf 9 6 . 8 \pm 1 . 1 }$  94.8±1.0</td><td> ${ \bf 9 6 . 3 \pm 1 . 0 }$  92.8±1.6</td><td>70.1±2.5 66.4±3.5</td></tr><tr><td>MRGP</td><td>88</td><td>87</td><td>83</td><td>3</td></tr><tr><td>Operon</td><td>85</td><td>86</td><td>85</td><td>68</td></tr><tr><td>SBP-GP</td><td>75</td><td>74</td><td>74</td><td>50</td></tr><tr><td>AI Feynman</td><td>74</td><td>73</td><td>69</td><td>12</td></tr><tr><td>GP-GOMEA</td><td>71</td><td>72</td><td>72</td><td>67</td></tr></table>

## 4.2. Where the accuracy comes from

We change one ingredient at a time and keep the rest of the pipeline fixed. The suite has eight target families with five instances each, with and without noise (Table 3). Projection is the ingredient that matters most. If each first-layer port may read only one input coordinate, fewer noise-free problems reach the tolerance: 16 instead of 24 of 40 with rational units, and 18 instead of 24 with confluent units. The median test error rises by 5.2× and 2.2×. The formulas also grow, because the network must rebuild each projection from single coordinates. The median rational formula goes from 36 to 67 unit operations, and the selected capacity moves to the largest network on 31 of 40 problems instead of 11. On noisy data the error is set by the noise, and the two versions differ little (1.3× and 1.1×).

Product ports matter only where the target multiplies terms. Without them, about as many noise-free problems reach the tolerance (22 and 23 of 40), and the median formula is shorter (22 against 36 operations for rational units, 31 against 60 for confluent units). Both unit families can emulate a product, for example as ${ \scriptstyle { \frac { 1 } { 4 } } } \left( ( u + v ) ^ { 2 } - ( u - v ) ^ { 2 } \right)$ or as a sum of exponentials, so most targets do not need a dedicated port. The exceptions are targets built around a product. On sin $u + v w .$ , rational units without products reach the tolerance on 1 of 5 instances instead of 4, and the median error rises by 62×. On damped oscillations $e ^ { - a ( u + 1 ) } \sin ( b v + c )$ confluent units reach it on 2 instead of $^ { 4 , }$ and the formula more than doubles its constants (842 against 370). We keep the ports, because such targets are common in science. On the other targets the price is a longer trained formula, not a less accurate one. The pruning grid matters on noise-free data. With a single refit at a middle level, 16 instead of 24 rational and 13 instead of 24 confluent problems reach the tolerance, the median error rises by 1.9× and 2.3×, and the formulas grow (56 against 36 operations for rational units, 160 against 60 for confluent units). On noisy data the error barely changes (1.2×), but the formulas are still longer. The grid costs about three times the CPU of a single refit.

Table 3. Ablations on the current engine, eight families × five instances per condition. Reached: problems whose validation error is within $\dot { \tau } = 1 0 ^ { - 5 }$ . Ratio: median paired test-error ratio to the full model. Ops: median unit operations of the selected formula.
<table><tr><td></td><td></td><td colspan="3">Noise-free</td><td colspan="2">Noisy</td></tr><tr><td>Unit</td><td>Variant</td><td>Reached Ratio</td><td></td><td>Ops</td><td></td><td>Ratio Ops</td></tr><tr><td>Rational</td><td>full first layer on axes</td><td>24/40 16/40</td><td>1 5.2</td><td>36 67</td><td>1 1.3</td><td>18 30</td></tr><tr><td></td><td>no products</td><td>22/40</td><td>0.72</td><td>22</td><td>0.86</td><td>10</td></tr><tr><td></td><td>one prune level</td><td>16/40</td><td>1.9</td><td>56</td><td>1.2</td><td>27</td></tr><tr><td>Confluent full</td><td></td><td>24/40</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>1</td><td>60</td><td>1</td><td>17</td></tr><tr><td></td><td>first layer on axes</td><td>18/40</td><td>2.2</td><td>73</td><td>1.1</td><td>23</td></tr><tr><td></td><td>no products</td><td>23/40</td><td>1.0</td><td>31</td><td>0.89</td><td>10</td></tr><tr><td></td><td>one prune level</td><td>13/40</td><td>2.3</td><td>160</td><td>1.2</td><td>30</td></tr></table>

## 4.3. Structure recovery

To measure recovery, we need targets whose generating formula is known and expressible in the library. We use 48 synthetic data sets in four families (response $( a + b u ) / ( 1 +$ $c v ^ { 2 } )$ , oscillation $e ^ { - a ( u + 1 ) } \sin ( b v + c ) $ , reuse sin u + vw, special erf $: ( u ) + c J _ { 0 } ( v ) )$ , where $u , v ,$ w are fixed projections of a few of the $d \in \{ 5 , 2 0 , 5 0 \}$ inputs, and n runs from $1 0 ^ { 3 }$ to $1 0 ^ { 6 }$ rows. We apply recovery to the 138 formulas chosen by the CPU and GPU runs, with the original error bound and no access to the test set.

Recovery returned the generating formula in 96 of the 104 cases whose functions are in the library, with test error around $1 0 ^ { - 1 6 }$ (Table 4). All 36 response formulas were recovered, including one with 92 unit operations and 1478 constants, which became a formula with 6 operations and 9 constants. The irrelevant inputs were removed in every recovered case. For example, an oscillation formula with 48 operations and 519 constants on $1 0 ^ { 6 }$ rows and 50 inputs became

$$
\begin{array} { r l } & { y = 0 . 7 9 6 e ^ { z _ { 2 } } \sin z _ { 1 } , } \\ & { z _ { 1 } = 0 . 6 2 1 - 0 . 2 3 4 x _ { 2 } - 0 . 1 7 3 x _ { 3 } + 1 . 7 2 x _ { 4 } , } \\ & { z _ { 2 } = 0 . 0 6 4 7 x _ { 1 } - 0 . 1 2 1 x _ { 2 } + 0 . 0 4 2 3 x _ { 3 } . } \end{array}
$$

The eight misses outside the special family stop at a nearby form: five oscillation formulas keep a saturating tanh in place of exp, with test error $2 \times 1 0 ^ { - 7 } \mathrm { ~ t o ~ 2 ~ } \times 1 0 ^ { - 5 }$ . The special family uses erf and $J _ { 0 } .$ , which are not in the library. There the recovered formulas approximate them, with 8 to 13 operations and test error $6 \times 1 0 ^ { - 8 } \mathrm { t o } 7 \times 1 0 ^ { - 7 } .$ ; Operon reaches $1 0 ^ { - 4 }$

Across all 138 formulas, the median recovered formula has 7 unit operations and 10 constants, against 16 and 16 for Operon on the same data. Paired by data set, the recovered formula has no more constants than Operon in 108 cases, no more operations in 127, and lower test error in 137.

Table 4. Structure recovery on 138 formulas. Exact: test nRMSE $< 1 0 ^ { - 1 2 }$ . Medians of unit operations / constants, before and after recovery, and of Operon on the same data. Bold: shorter than Operon in both counts.
<table><tr><td>Family</td><td>Exact</td><td>Before</td><td>After</td><td>Operon</td></tr><tr><td>Response</td><td>36/36</td><td>26/175</td><td>7/9</td><td>16/18</td></tr><tr><td>Oscillation</td><td>29/34</td><td>26/250</td><td>7/8</td><td>17/14</td></tr><tr><td>Reuse</td><td>31/34</td><td>15/45</td><td>5/11</td><td>17/21</td></tr><tr><td>Special</td><td>0/34</td><td>21/115</td><td>13/23</td><td>13/14</td></tr><tr><td>All</td><td>96/138</td><td>23/107</td><td>7/10</td><td>16/16</td></tr></table>

Table 5. Cost on 48 data sets. Medians of per-data-set ratios Operon / MorphingNet; above one favors MorphingNet. CPU at $1 0 ^ { \overset { \bullet } { } }$ rows has two pairs.
<table><tr><td>Rows</td><td> $1 0 ^ { 3 }$ </td><td> $1 0 ^ { 4 }$ </td><td> $\mathrm { 1 0 ^ { 5 } }$ </td><td> ${ 1 0 } ^ { 6 }$ </td></tr><tr><td>Wall time, CPU</td><td>1.0</td><td>3.3</td><td>4.9</td><td>11.4</td></tr><tr><td>Wall time, GPU</td><td></td><td>4.4</td><td>24.9</td><td>122.6</td></tr><tr><td>Test error</td><td>22</td><td>11</td><td>16</td><td>18</td></tr></table>

## 4.4. Cost and scale

On the same 48 data sets we compare wall time with Operon, run with four threads, and PySR. MorphingNet uses 16 single-threaded jobs on CPU, or three GPUs with four jobs each. Table 5 gives paired ratios. On CPU, MorphingNet matches Operon at $1 0 ^ { 3 }$ rows and is three to five times faster at $1 0 ^ { 4 }$ and $1 0 ^ { 5 }$ . PySR is one to two orders of magnitude slower than both. On GPU the advantage grows with the data: 25 times at $1 0 ^ { 5 }$ rows and 123 times at $1 0 ^ { 6 }$ , where Operon takes a median of 11.8 hours and MorphingNet 4.5 minutes for R and 6.3 for C. At every size the median test error of MorphingNet is 11 to 22 times lower than that of Operon. On the 8 000-row development problem I.12.2, the GPU pipeline takes 50 seconds for R and 42 for C.

## 4.5. Comparison with neural and parametric methods

We ran the closest methods of Section 2 on a common suite of eight problem families, with the same splits and the same selection rule for every method. Table 6 gives median test errors. On noise-free data MorphingNet is two to three orders of magnitude more accurate than ParFam, EQL, KAN and DySymNet. KAN loses about a factor of three between its splines and its exported formulas. Brush (Aldeia et al., 2026) and Operon are the strongest search methods. Paired by problem, MorphingNet is more accurate than Brush on five of the eight in both conditions, and more accurate than Operon on all eight noise-free problems and five of the eight noisy ones.

Table 6. Median test nRMSE on the common suite, eight problems per condition, and median unit operations / constants on noise-free data. Lowest error in bold, second underlined.
<table><tr><td>Method</td><td>Noise-free</td><td>Noisy</td><td>Ops / const.</td></tr><tr><td>MorphingNet-R MorphingNet-C</td><td>7.9e-6 1.3e-5</td><td>4.0e-3 4.8e-3</td><td>30/130 34/130</td></tr><tr><td>Brush</td><td>3.9e-5</td><td>6.5e-3</td><td>20/17</td></tr><tr><td>Operon</td><td>9.7e-5</td><td>5.5e-3</td><td>13/16</td></tr><tr><td>PySR</td><td>2.4e-3</td><td>9.1e-3</td><td>20/13</td></tr><tr><td>ParFam</td><td>9.4e-4</td><td>4.5e-3</td><td>10/62</td></tr><tr><td>EQL</td><td>9.5e-3</td><td>2.4e-2</td><td>31/46</td></tr><tr><td>KAN 2.0 (formula)</td><td>1.5e-2</td><td>1.7e-2</td><td>33/34</td></tr><tr><td>KAN 2.0 (spline)</td><td>4.9e-3</td><td>8.2e-3</td><td></td></tr><tr><td>DySymNet</td><td>1.5e-2</td><td>2.5e-2</td><td>10/15</td></tr></table>

## 5. Discussion

Why trained formulas are long, and why that is not the end. A network of flexible units approximates what it cannot represent and spreads one structure over several branches. This is a property of continuous structure learning, not of MorphingNet alone: a wider KAN has more redundancy, not less. What matters is whether the short formula can be recovered. Because the trained network is already exact, recovery starts from an accurate formula and can be held to its error, which is not possible when export itself is an approximation.

The role of the library. MorphingNet needs no operator library to train or to export. The library enters only during recovery, and it decides how the result is named, not whether a structure is found. The special family shows both sides: without erf and $J _ { 0 }$ the formulas are approximations, but they are two to three orders of magnitude more accurate than a search method with its library. A user who expects such functions can add them.

What a formula learned from data can claim. Outside the range of the data, a formula found by MorphingNet can move far from the generating function. We do not see this as a failure. A data-driven method learns from the data and reports a formula that is faithful to them, and a finite sample is matched by many functions. If f fits the training points $x _ { 1 } , \ldots , x _ { n } ,$ , so does $\begin{array} { r } { g _ { c } ( x ) = f ( x ) + c \prod _ { i = 1 } ^ { n } ( x - x _ { i } ) } \end{array}$ for every c. The same holds within a tolerance on the whole domain: $f ( x ) + \varepsilon \left( ( x - m ) / r \right) ^ { N }$ stays within ε of f for $| x -$ $| m | \leq r$ and grows like $| x | ^ { N }$ outside. Different behaviour outside the domain is therefore a necessary consequence of learning from finite data.

Limitations. We did not study heavy-tailed noise, because SRBench uses Gaussian noise; this is next. At the highest noise level MorphingNet leads by two points, within the spread over splits; the gap at lower noise does not carry over because the selection split cannot resolve candidates below the noise. The domain of the C unit with $\beta \neq 0$ is checked on sampled points, not certified on the whole input box. Structure recovery is greedy, takes a median of 215 seconds per formula on one core, and is sensitive to the last digits of its inputs; a different execution order can end at a nearby form. The fixed-operator variant of the network was not ported to the accelerated implementation.

## 6. Conclusion

MorphingNet makes the form of each function a continuous parameter, so symbolic regression becomes one smooth least-squares problem. The trained network is the formula, it is accurate on standard benchmarks, and a recovery step brings most formulas back to their generating form. Because training is dense linear algebra, it runs at the speed of conventional symbolic regression on a CPU and one to two orders of magnitude faster on a GPU when the data are large.

We end with why we think symbolic regression is promising for scientific discovery. Our conjecture is that understanding the world is itself a large reverse-engineering problem. Newton’s law of gravitation was read off observations and checked against other theories, and its contribution is beyond question. Yet suppose a creator made the world, gravity included, by some program. Would Newton’s law be that program? We think it need not be. This takes nothing away from the law: within its domain it is excellent, and it fits with the rest of physics. Much of science consists of such laws, valid under certain conditions or in a certain range, and this is the normal state of knowledge. We hold discovered formulas to the same standard: they need not be the true formula, if one exists, but should be accurate within the domain of the data and consistent with established theory in the field. Judging the second condition, and choosing among the candidates from different seeds, is the work of domain experts; MorphingNet is meant to give them candidates, not to replace that judgement.

## Impact Statement

This paper presents work whose goal is to advance the field of machine learning. Symbolic regression is used to build interpretable models in science and engineering. Formulas found from data can look authoritative while being wrong outside the range of the data; we report the domain of every formula and recommend that discovered formulas be checked by domain experts before use.

## References

Aghaei, A. A. rKAN: Rational kolmogorov-arnold networks. arXiv preprint arXiv:2406.14495, 2024.

Alaa, A. M. and van der Schaar, M. Demystifying blackbox models with symbolic metamodels. In Advances in Neural Information Processing Systems, 2019.

Aldeia, G. S. I., Romano, J. D., de Franc¸a, F. O., Herman, D. S., and La Cava, W. G. Towards symbolic regression for interpretable clinical decision scores. Philosophical Transactions ofthe Royal Society A, 2026. doi: 10.1098/ rsta.2024.0588.

Arnaldo, I., Krawiec, K., and O’Reilly, U.-M. Multiple regression genetic programming. In Proceedings of the Genetic and Evolutionary Computation Conference, 2014.

Bagrow, J. and Bongard, J. Softly symbolifying Kolmogorov-Arnold networks. arXiv preprint arXiv:2512.07875, 2025.

Baker, G. A. and Graves-Morris, P. Pade Approximants´ . Cambridge University Press, 2 edition, 1996.

Biggio, L., Bendinelli, T., Neitz, A., Lucchi, A., and Parascandolo, G. Neural symbolic regression that scales. In International Conference on Machine Learning, 2021.

Boyd, S., Parikh, N., Chu, E., Peleato, B., and Eckstein, J. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine Learning, 3(1):1–122, 2011.

Brkic, D. W solutions of the CW equation for flow friction.´ Applied Mathematics Letters, 24(8):1379–1383, 2011.

Burlacu, B., Kronberger, G., and Kommenda, M. Operon C++: An efficient genetic programming framework for symbolic regression. In Proceedings of the Genetic and Evolutionary Computation Conference Companion, pp. 1562–1570, 2020.

Crabbe, J., Zhang, Y., Zame, W., and van der Schaar, M.´ Learning outside the black-box: The pursuit of interpretable models. In Advances in Neural Information Processing Systems, 2020.

Cranmer, M. Interpretable machine learning for science with PySR and SymbolicRegression.jl. arXiv preprint arXiv:2305.01582, 2023.

Cranmer, M., Sanchez-Gonzalez, A., Battaglia, P., Xu, R., Cranmer, K., Spergel, D., and Ho, S. Discovering symbolic models from deep learning with inductive biases. In Advances in Neural Information Processing Systems, 2020.

d’Ascoli, S., Kamienny, P.-A., Lample, G., and Charton, F. Deep symbolic regression for recurrent sequences. In International Conference on Machine Learning, 2022.

de Franc¸a, F. O. Transformation-interaction-rational representation for symbolic regression. In Proceedings of the Genetic and Evolutionary Computation Conference, 2022.

de Vries, S., Keemink, S. W., and van Gerven, M. A. J. Kozax: Flexible and scalable genetic programming in JAX. In Proceedings of the Genetic and Evolutionary Computation Conference Companion, pp. 603–606, 2025.

Friedman, J. H. and Stuetzle, W. Projection pursuit regression. Journal ofthe American Statistical Association, 76 (376):817–823, 1981.

Godfrey, L. B. and Gashler, M. S. A continuum among logarithmic, linear, and exponential functions, and its potential to improve generalization in neural networks. arXiv preprint arXiv:1602.01321, 2016.

Kamienny, P.-A., d’Ascoli, S., Lample, G., and Charton, F. End-to-end symbolic regression with transformers. In Advances in Neural Information Processing Systems, 2022.

Koza, J. R. Genetic Programming: On the Programming of Computers by Means of Natural Selection. MIT Press, 1992.

La Cava, W., Orzechowski, P., Burlacu, B., de Franc¸a, F. O., Virgolin, M., Jin, Y., Kommenda, M., and Moore, J. H. Contemporary symbolic regression methods and their relative performance. In NeurIPS Track on Datasets and Benchmarks, 2021.

Landajuela, M., Lee, C. S., Yang, J., Glatt, R., Santiago, C. P., Aravena, I., Mundhenk, T., Mulcahy, G., and Petersen, B. K. A unified framework for deep symbolic regression. In Advances in Neural Information Processing Systems, 2022.

Levenberg, K. A method for the solution of certain nonlinear problems in least squares. Quarterly ofApplied Mathematics, 2(2):164–168, 1944.

Li, W., Li, W., Yu, L., Wu, M., Sun, L., Liu, J., Li, Y., Wei, S., Deng, Y., and Hao, M. A neural-guided dynamic symbolic network for exploring mathematical expressions from data. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of PMLR, pp. 28222–28242, 2024.

Liu, Z., Ma, P., Wang, Y., Matusik, W., and Tegmark, M. KAN 2.0: Kolmogorov-arnold networks meet science. arXiv preprint arXiv:2408.10205, 2024.

Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljaciˇ c, M., Hou, T. Y., and Tegmark, M. KAN:´ Kolmogorov–arnold networks. In International Conference on Learning Representations, 2025.

Mao, H., Liu, X. T., Lu, S., Zhao, P., Jiang, W., and Chen, Y. Efficient constant optimization for symbolic regression with GPU-accelerated tree-based genetic programming. In IEEE High Performance Extreme Computing Conference (HPEC), 2026.

Marquardt, D. W. An algorithm for least-squares estimation of nonlinear parameters. Journal ofthe Societyfor Industrial and Applied Mathematics, 11(2):431–441, 1963.

Martius, G. and Lampert, C. H. Extrapolation and learning equations. arXiv preprint arXiv:1610.02995, 2016.

Molina, A., Schramowski, P., and Kersting, K. Pade acti-´ vation units: End-to-end learning of flexible activation functions in deep networks. In International Conference on Learning Representations, 2020.

Montazerin, M., Al Aawar, M., Ortega, A., and Srivastava, A. Sparse interpretable deep learning with LIES networks for symbolic regression. arXiv preprint arXiv:2506.08267, 2025.

Newman, D. J. Rational approximation to |x|. Michigan Mathematical Journal, 11:11–14, 1964.

Ootomo, H., Ozaki, K., and Yokota, R. DGEMM on integer matrix multiplication unit. The International Journal of High Performance Computing Applications, 38(4):297– 313, 2024.

Ozaki, K., Ogita, T., Oishi, S., and Rump, S. M. Errorfree transformations of matrix multiplication by using fast routines of matrix multiplication and its applications. Numerical Algorithms, 59(1):95–118, 2012.

Praks, P. and Brkic, D. Symbolic regression-based genetic´ approximations of the Colebrook equation for flow friction. Water, 10(9):1175, 2018.

Ranasinghe, N., Senanayake, D., Seneviratne, S., Premaratne, M., and Halgamuge, S. GINN-LP: A growing interpretable neural network for discovering multivariate Laurent polynomial equations. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 2024.

Sahoo, S. S., Lampert, C. H., and Martius, G. Learning equations for extrapolation and control. In International Conference on Machine Learning, 2018.

Schmidt, M. and Lipson, H. Distilling free-form natural laws from experimental data. Science, 324(5923):81–85, 2009.

Scholl, P., Bieker, K., Hauger, H., and Kutyniok, G. Par-Fam – (neural guided) symbolic regression via continuous global optimization. In International Conference on Learning Representations, 2025.

Shojaee, P., Meidani, K., Gupta, S., Farimani, A. B., and Reddy, C. K. LLM-SR: Scientific equation discovery via programming with large language models. In International Conference on Learning Representations, 2025.

Tsoi, H. F., Loncar, V., Dasu, S., and Harris, P. SymbolNet: Neural symbolic regression with adaptive dynamic pruning for compression. Machine Learning: Science and Technology, 6(1):015021, 2025.

Uchino, Y., Ozaki, K., and Imamura, T. Performance enhancement of the Ozaki scheme on integer matrix multiplication unit. The International Journal ofHigh Performance Computing Applications, 39(3):462–476, 2025.

Udrescu, S.-M. and Tegmark, M. AI Feynman: A physicsinspired method for symbolic regression. Science Advances, 6(16):eaay2631, 2020.

Udrescu, S.-M., Tan, A., Feng, J., Neto, O., Wu, T., and Tegmark, M. AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity. In Advances in Neural Information Processing Systems, 2020.

Virgolin, M., Alderliesten, T., and Bosman, P. A. N. Linear scaling with and within semantic backpropagation-based genetic programming for symbolic regression. In Proceedings ofthe Genetic and Evolutionary Computation Conference, 2019.

Virgolin, M., Alderliesten, T., Witteveen, C., and Bosman, P. A. N. Improving model-based genetic programming for symbolic regression of small expressions. Evolutionary Computation, 29(2):211–237, 2021.

Wang, Y., Wagner, N., and Rondinelli, J. M. Symbolic regression in materials science. MRS Communications, 9 (3):793–805, 2019.

Wu, M., Li, W., Yu, L., Li, W., Liu, J., Li, Y., and Hao, M. PruneSymNet: A symbolic neural network and pruning algorithm for symbolic regression. arXiv preprint arXiv:2401.15103, 2024.

Wu, Z., Wang, L., Sun, K., Li, Z., and Cheng, R. Enabling population-level parallelism in tree-based genetic programming for GPU acceleration. arXiv preprint arXiv:2501.17168, 2025.

Zhang, R., Lensen, A., and Sun, Y. Speeding up genetic programming based symbolic regression using GPUs. In Pacific Rim International Conference on Artificial Intelligence (PRICAI), 2022.

## A. Details of the units

Rational unit. Each unit has numerator and denominator coefficient $p _ { 0 } , \ldots , p _ { D }$ and $q _ { 0 } , \ldots , q _ { D }$ and computes $P ( z ) / ( 1 +$ $Q ( z ) ^ { 2 } )$ by Horner’s rule. The denominator is at least one for every real $z ,$ so the unit needs no domain check. At initialization $p _ { k } \sim \mathcal N ( 0 , 0 . 0 2 5 ^ { 2 } )$ with $p _ { 1 }$ shifted by one, so each unit starts close to the identity, and $q _ { k } \sim \mathcal { N } ( 0 , 0 . 0 8 ^ { 2 } )$ . The denominator coefficients must not start at zero, because the derivative of $1 + Q ^ { 2 }$ with respect to Q vanishes at $Q = 0$

One formula, no branches. The C unit is written with two functions that are analytic at the origin,

$$
G ( u ) = \frac { \log ( 1 + u ) } { u } , \qquad E ( t ) = \frac { e ^ { t } - 1 } { t } , \qquad G ( 0 ) = E ( 0 ) = 1 ,
$$

as $L _ { \beta } ( z ) = z G ( \beta z )$ and $J ( z ) = L E ( q L )$ . This is the definition of Section 3.1 with the two divisions moved inside G and $E .$ Both limits of that section follow at once: $\beta  0$ gives $L \to z ,$ , and $q \to 0$ gives $J \to L$ . So the five cases listed there are values of one smooth function of $( z , q , \beta )$ , and training moves through them without a switch.

Stable evaluation. Near the origin the quotients G and E lose all accuracy to cancellation. We evaluate them by their Taylor series of degree 16 when $| u | \leq 1 / 1 6 \ \mathrm { o r } \ | t | \leq 1 / 2$ and otherwise by $\log 1 \mathrm { p } ( u ) / u$ and expm1 $( t ) / t ;$ both branches are accurate to double rounding. The derivatives used by LM are analytic:

$$
\frac { \partial J } { \partial z } = \frac { e ^ { q L } } { 1 + \beta z } , \qquad \frac { \partial J } { \partial q } = L ^ { 2 } E ^ { \prime } ( q L ) , \qquad \frac { \partial J } { \partial \beta } = e ^ { q L } z ^ { 2 } G ^ { \prime } ( \beta z ) ,
$$

with $E ^ { \prime }$ and $G ^ { \prime }$ evaluated by the same two-branch rule. At initialization $\alpha \sim \mathcal { N } ( 0 , 0 . 2 ^ { 2 } ) , | \omega | = e ^ { s }$ with s uniform on $[ - 1 . 4 , 1 . 4 ]$ and a random sign, and $\beta \sim \mathcal { N } ( 0 , 0 . 0 2 5 ^ { 2 } )$ .

Export. The exported formula is real. With $\ell = L _ { \beta } ( z ) , A = \mathrm { e x p m 1 } ( \alpha \ell ) \cos \omega \ell - 2 \sin ^ { 2 } ( \omega \ell / 2 )$ and $B = e ^ { \alpha \ell } \sin \omega \ell ,$ which are the real and imaginary parts of $e ^ { q \ell } - 1$

$$
\mathrm { R e } J = \frac { \alpha A + \omega B } { \alpha ^ { 2 } + \omega ^ { 2 } } , \qquad \mathrm { I m } J = \frac { \alpha B - \omega A } { \alpha ^ { 2 } + \omega ^ { 2 } } .
$$

A unit with $\beta = 0$ exports $\ell = z ,$ and a unit with $q = 0$ exports $J = \ell ;$ these are the exact limits, not small substitutes.

Domain. For $\beta \neq 0$ the logarithm needs $1 + \beta z > 0$ . Every accepted step is checked on the training rows and on 256 probes in the declared input box, uniform points of which each has one coordinate set to its lower or upper bound. A step is rejected if $1 + \beta z \le 0 \mathrm { o r } \alpha \ell > 5 0$ at any of these points. This guard is sampled, not a certificate over the box (Section 5). Rational units need no guard.

## B. Hyperparameters

Table 7 lists the three capacities and Table 8 the settings. They are the same for every problem in the paper. Affine weights start from $\mathcal { N } ( 0 , 0 . 0 4 / k )$ for a layer that reads k quantities, and biases from $\mathcal { N } ( 0 , 0 . 0 3 ^ { 2 } )$ . Each run uses one initialization seed; the ten SRBench repetitions differ in their data splits.

Table 7. Capacities trained in each run, from small to large. Units and products are counted per layer.
<table><tr><td>Layers</td><td>Units</td><td>Products</td><td>Degree  $D$ </td></tr><tr><td>1</td><td>6</td><td>2</td><td>3</td></tr><tr><td>2</td><td>6</td><td>2</td><td>3</td></tr><tr><td>3</td><td>12</td><td>4</td><td>5</td></tr></table>

## C. The proximal step

The LM subproblem has a quadratic part and a nonsmooth penalty part. We split the two with ADMM (Boyd et al., 2011). The quadratic part needs one Cholesky factorization of $J ^ { \top } J + ( \mu + \rho ) I$ per LM step. The penalty part is a proximal map, a

Table 8. Training and selection settings.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td> $\ell _ { 1 }$  weight  $\lambda _ { 1 }$  / group weight  $\lambda _ { g }$ </td><td> $3 \times 1 0 ^ { - 6 } / 1 0 ^ { - 3 }$ </td></tr><tr><td>Warm-up without penalty</td><td> $2 0 0 \mathrm { L M }$  steps</td></tr><tr><td>Steps per fit and per refit</td><td>at most 2000</td></tr><tr><td>Control error</td><td>checked after every step</td></tr><tr><td>Stagnation</td><td>40 steps without a relative improvement of  $1 0 ^ { - 5 }$ </td></tr><tr><td>Proximal gradient tolerance</td><td> $1 0 ^ { - 8 }$ </td></tr><tr><td>ADMM</td><td>residual tolerance  $1 0 ^ { - 7 }$  , at most 400 inner iterations</td></tr><tr><td>Tolerance τ (control and selection nRMSE)</td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>Domain probes</td><td>256</td></tr><tr><td>Coefficient thresholds t</td><td>0, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9</td></tr><tr><td>Unit thresholds u</td><td>0, 0.35, 0.65, 0.9</td></tr><tr><td>Selection tie factor</td><td>1.1 (on excess error for noisy data, Appendix D)</td></tr></table>

soft threshold followed by a group shrinkage, and it returns exact zeros. The ADMM penalty $\rho$ is set by residual balancing. Consecutive subproblems differ little, so we carry the dual variable and $\rho$ from one LM step to the next. With this warm start the inner solver needs about ten iterations per step, and it costs less than a tenth of the Jacobian and Gram computation.

## D. Selection on noisy data

On noisy data no candidate can have a selection error below the noise level $\sigma _ { : }$ , measured in the same scaled units. The rule of Section 3.3 then loses its meaning. Take $\sigma = 0 . 1$ and a best candidate with $e _ { \mathrm { m i n } } = 0 . 1 0 0 5$ , whose error on the signal is $\sqrt { e _ { \mathrm { m i n } } ^ { 2 } - \sigma ^ { 2 } } = 0 . 0 1$ . The tie band 1.1 $e _ { \mathrm { m i n } }$ admits a candidate whose error on the signal is 0.046, five times larger. We therefore apply the factor to the error that remains after the noise is removed.

The noise is estimated from the training split alone. Inputs are standardized; for each row i we fit a linear function to its $k = \operatorname* { m a x } ( 4 ( d + 1 ) , 2 0 )$ nearest neighbours, excluding the row itself, and take the residual $r _ { i }$ of y at $x _ { i }$ . The local fit has error of its own, so the mean squared residual is divided by one plus the average leverage $( d + 1 ) / k$ of the fit:

$$
\hat { \sigma } ^ { 2 } = \frac { 1 } { 1 + ( d + 1 ) / k } \cdot \frac { 1 } { n } \sum _ { i } r _ { i } ^ { 2 } ,
$$

divided by the output scale. With $s = \operatorname* { m i n } ( \hat { \sigma } , e _ { \operatorname* { m i n } } )$ , a frontier candidate with error e is tied with the best if

$$
e ^ { 2 } - s ^ { 2 } \leq 1 . 2 1 ( e _ { \mathrm { m i n } } ^ { 2 } - s ^ { 2 } ) ,
$$

and the shortest tied candidate is chosen as before. When ${ \hat { \sigma } } \geq e _ { \operatorname* { m i n } }$ the band closes and the most accurate candidate is chosen. A local linear fit overestimates the noise where the target curves strongly within a neighbourhood. So when $\hat { \sigma } > 1 . 5 e _ { \mathrm { m i n } }$ the estimate is not trusted and the rule of Section 3.3 is used. On SRBench this fallback fires on most problems at noise 0.001, where curvature dominates the local residual, and on fewer at higher noise.

The rule matters at the highest noise level. There the plain rule reaches $R ^ { 2 } > 0 . 9 9 9$ on 33% of the problems with rational units and 23% with confluent units, against 70% and 66% with this rule. At noise 0.01 and below the two rules agree within one point (Table 9).

## E. Structure recovery

Library. A one-dimensional piece $g ( z )$ of the formula is replaced by $A \phi ( z ) + \gamma z + d ,$ with ϕ one of sin $( a z + b ) , e ^ { a z }$ $( z + a ) ^ { 2 } , 1 / ( z + a ) , \log ( z + a ) , \sqrt { z + a } , \operatorname { t a n h } ( a z + b ) , e ^ { - ( a z + b ) ^ { 2 } } , 1 / ( 1 + ( a z + b ) ^ { 2 } ) , ( z + a ) ^ { b }$ and $e ^ { a z } \sin ( b z + c )$ , or by a constant or a linear function. For each form the linear coefficients A, γ, d are solved by least squares inside a nonlinear least-squares fit of the inner parameters, from several starts. Candidates are tried in a fixed order of preference, first constan and linear, then the other forms, with $( z + a ) ^ { b }$ and the damped sine last, and within each group by fit error. A user adds a form by giving its expression and its starting points.

Acceptance. A rewrite removes the parts of the formula it leaves unused and is accepted if the formula stays finite, has no more operations and no more constants than before, with at least one of the two fewer, and has selection nRMSE at most $\operatorname* { m a x } ( 1 . 1 e _ { 0 } , \tau )$ . Every candidate is first checked as proposed. If none passes, the first three are refitted jointly with all other constants (LM, 4000 rows, 50 iterations) and checked again. Refits that do not change the structure are kept only if they lower the selection error.

Rewrites. Unit replaces one unit. A C unit with $\beta = 0$ becomes the exact real form of Appendix A. Merge collects all terms of one consumer that depend on the same input direction, such as several rational units that together approximate one sinusoid, into one function of that direction. Quadratic collects the products of affine ports read by one consumer into a symmetric matrix $Q ,$ , a linear part and a constant. Its eigendecomposition gives the fewest products: a positive and a negative eigenvector $e _ { + } , e _ { - }$ share one product $( e _ { + } + e _ { - } ) ^ { \top } x ( e _ { + } - e _ { - } ) ^ { \top } x$ , so Q needs as many products as the larger of its numbers of positive and negative eigenvalues. Small eigenvalues are dropped at relative thresholds from $1 0 ^ { - 9 } \mathrm { t o } 1 0 ^ { - 2 }$ , each tried in turn. If the consumer also holds a function of a direction u in the range of $Q = V \Lambda V ^ { \top }$ , then removing $\kappa u u ^ { \top }$ with $\kappa = 1 / ( c ^ { \top } \Lambda ^ { - 1 } c ) , c = V ^ { \top } \boldsymbol { \imath }$ , lowers the rank of $Q$ by one. The square $\kappa ( \boldsymbol { u } ^ { \intercal } \boldsymbol { x } ) ^ { 2 }$ is kept as a function of u, where Merge can combine it with the other terms on u. Delete removes affine terms one at a time with a local refit, and absorbs a linear term $w \left( p ^ { \intercal } x \right)$ into the bias of a product factor when the shift reproduces it. Swap replaces one elementary function by another when the jointly refitted and pruned formula halves the selection error and is not longer. Template applies when the averaged gradient outer product $\mathbb { E } [ \nabla f \nabla f ^ { \top } ]$ on 4000 rows has rank two. In that plane, the Hessians of an additive $g _ { 1 } ( a _ { 1 } ^ { \top } x ) + g _ { 2 } ( a _ { 2 } ^ { \top } x )$ lie in the span of $a _ { 1 } a _ { 1 } ^ { \top }$ and $\overline { { a _ { 2 } a _ { 2 } ^ { \top } } }$ , and so do those of log |f| for a product $g _ { 1 } g _ { 2 }$ . This fixes $a _ { 1 }$ and $^ { a _ { 2 } ; }$ each $g$ is chosen from the linear function and the first nine forms of the library, and the template is kept if, refitted and pruned, it is within the bound and shorter than the current formula.

Order and cost. A round applies Unit, Merge, Quadratic, absorption, a joint refit, Delete, another refit and Swap, and then Template if the formula has a length not seen before. Rounds repeat until none changes the formula, at most four times. A last pass applies Swap to forms accepted in the final round, and biases below $1 0 ^ { - 1 2 }$ of the output scale, left over by exact rewrites, are set to zero. Recovery takes a median of 215 seconds per formula on one CPU core.

## F. Implementation

On CPU, a C++ core computes the analytic Jacobian for all rows at once, and the fit and refits, which are independent, execute in parallel as single-threaded processes. Sixteen single-threaded jobs use five times less CPU time than four jobs with four BLAS threads each, at a similar wall time. On GPU, one fused kernel computes the forward pass and the Jacobian, and the data, Gram matrix, factorization and ADMM stay on the device. Several fits share one GPU through MPS, which lets their kernels run concurrently.

Gram matrix on integer tensor cores. Consumer GPUs have weak double-precision arithmetic: one 8000 × 1745 Gram matrix takes 30 ms in FP64 on an RTX 5090. We compute it with the Ozaki scheme (Ozaki et al., 2012), as on integer matrix units in Ootomo et al. (2024); Uchino et al. (2025). Each row of $J ^ { \top }$ , one parameter over all data rows, is scaled by a power of two so that its largest entry lies in [64, 128), and split into eight slices of seven bits, stored as 8-bit integers. The products of slices t and u with the same weight $t + u$ are summed on INT8 tensor cores with 32-bit integer accumulation. The data rows are processed in blocks of 16 384, since $8 \cdot 1 6 3 8 4 \cdot 1 2 7 ^ { 2 } < 2 ^ { 3 1 }$ , so every accumulation is exact. Symmetry halves the work, $G = H + H ^ { \top } + D$ with H the pairs $t < u$ and D the pairs $t = u ,$ and weights $t + u \geq 8 ,$ which fall below double rounding, are dropped. The levels are combined in double precision. On Jacobians from real fits, the relative spectral-norm error is 4 $\times 1 0 ^ { - 1 \bar { 7 } } \mathrm { t o } 9 \times 1 0 ^ { - 1 5 }$ , against $2 \times 1 0 ^ { - 1 6 } t o 2 \times \bar { 1 } 0 ^ { - 1 5 }$ for an FP64 matrix product and about $1 0 ^ { - 7 }$ in FP32. This is $1 0 ^ { - 5 } ~ \mathrm { t o } ~ 0 . 0 5$ times the smallest LM damping used, so it does not change the steps: a 300-step refit reaches the same best control error as FP64 to seven digits. The scheme is 8 to 10 times faster than FP64: 3.7 against 30 ms for $8 0 0 0 \times 1 7 4 5$ , and 28 against 241 ms for 65 536 rows. An FP32 Gram matrix is faster still, but its error exceeds the damping by five orders of magnitude, and two thirds of the refit steps had to be repeated in FP64.

## G. Full results

Table 9 breaks down the SRBench result by problem set and gives the result of the plain selection rule.

Table 9. SRBench ground-truth problems: percentage with test $R ^ { 2 } > 0 . 9 9 9$ , mean ± standard deviation over ten splits, for all 130 problems, the 116 Feynman and the 14 Strogatz problems, and with the plain $1 . 1 e _ { \mathrm { m i n } }$ rule instead of the noise-aware rule. Last column: median unit operations / constants of the chosen formulas, before structure recovery.
<table><tr><td>Unit</td><td>Noise</td><td>All</td><td>Feynman</td><td>Strogatz</td><td>Plain rule</td><td> $\mathrm { { O p s } / \mathrm { { c o n s t . } } }$ </td></tr><tr><td>Rational</td><td>0</td><td> $9 7 . 0 { \pm } 0 . 9 $ </td><td> $9 7 . 8 { \pm } 0 . 4 $ </td><td> $9 0 . 0 { \pm } 8 . 0 \ $ </td><td> $9 7 . 0 { \pm } 0 . 9 $ </td><td> $5 9 / 2 1 6$ </td></tr><tr><td></td><td>0.001</td><td> $9 6 . 8 { \pm } 1 . 1 $ </td><td> $9 7 . 8 { \pm } 0 . 6 $ </td><td> $8 8 . 6 { \pm } 9 . 1 $ </td><td> $9 6 . 8 { \pm } 1 . 1 $ </td><td>39 / 94</td></tr><tr><td></td><td>0.01</td><td> $9 6 . 3 { \pm } 1 . 0 $ </td><td> $9 8 . 0 { \pm } 0 . 4 \ $ </td><td> $8 2 . 1 { \pm } 8 . 0 $ </td><td> $9 6 . 3 { \pm } 1 . 0 $ </td><td>36 / 83</td></tr><tr><td></td><td>0.1</td><td> $7 0 . 1 { \pm } 2 . 5 $ </td><td> $7 6 . 5 { \pm } 2 . 9 $ </td><td> $1 7 . 1 \pm 5 . 7$ </td><td> $3 3 . 2 \pm 3 . 5 $ </td><td>30/ 61</td></tr><tr><td>Confluent</td><td>0</td><td> $9 5 . 3 { \pm } 1 . 5 $ </td><td> $9 6 . 5 { \pm } 1 . 0 \ $ </td><td> $8 5 . 7 { \pm } 9 . 6 $ </td><td> $9 5 . 3 { \pm } 1 . 5 $ </td><td>50 / 196</td></tr><tr><td></td><td>0.001</td><td> $9 4 . 8 { \pm } 1 . 0 $ </td><td> $9 5 . 7 { \pm } 0 . 8 $ </td><td> $8 7 . 9 { \pm } 6 . 4 $ </td><td> $9 4 . 8 { \pm } 1 . 0 $ </td><td>37 / 100</td></tr><tr><td></td><td>0.01</td><td> $9 2 . 8 { \pm } 1 . 6 $ </td><td> $9 4 . 2 { \pm } 1 . 4 $ </td><td> $8 1 . 4 \pm 5 . 7$ </td><td> $9 2 . 2 { \pm } 2 . 4 $ </td><td>35 / 92</td></tr><tr><td></td><td>0.1</td><td> $6 6 . 4 \pm 3 . 5$ </td><td> $7 1 . 2 { \pm } 4 . 0 $ </td><td> $2 6 . 4 { \pm } 8 . 5 $ </td><td> $2 2 . 8 { \pm } 2 . 4 $ </td><td>27 / 60</td></tr></table>