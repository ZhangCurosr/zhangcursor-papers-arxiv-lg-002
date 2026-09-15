# Structured Features Overfit Where Random Features Grok

Chon-Fai Kam

dubussygauss@gmail.com

University Paris City & University of Reunion, Paris, France; Dipartimento di Fisica e Chimica Emilio Segr\`e, Universit\`a degli Studi di Palermo, Italy

Miloud Bessafi

miloud.bessafi@gmail.com

EnergyLab, University of Reunion, Saint-Denis, France

Frederic Cadet

frederic.cadet.run@gmail.com

University Paris City & University of Reunion, Paris, France; PEACCEL, AI for Biologics, Paris, France

## Abstract

Xu, Vardi and Safran (ICML 2026) prove that over-parameterized ridge regression over an unstructured random Gaussian feature map groks, with the delay between memorization and generalization growing as 1/λ in the weight decay. We show that on a structured feature map the same delay does not appear. For a band-limited Fourier feature map over $\mathbb { Z } _ { p } ^ { 2 }$ carrying a single-character target that lies inside the expressible class, enlarging the band at fixed positive weight decay drives peak held-out accuracy monotonically from 1.00 to 0.07, with no memorize-then-generalize regime anywhere along the sweep. The degradation is not an interpolation efect. It sets in at capacity ratio q/n = 0.638, far below the interpolation threshold, on separate grounds from the exact null space that appears above it. What does have a sharp boundary is the active support. Holding the nominal dimension fixed and masking the band back to 1089 active modes restores held-out accuracy of 1.000 with zero variance across seeds, while the full 4225-mode band collapses to 0.185. The number of active modes acts through the teacher-weighted spectrum of the empirical Gram matrix and not through the capacity ratio, which makes this a statement about feature geometry and not a restatement of double descent.

## 1. Introduction

Grokking, the onset of generalization long after a network has fit its training data, was first observed by Power et al. (2022) on small algorithmic datasets. On modular arithmetic it has since been traced to the emergence of a Fourier-based algorithm in the trained weights (Nanda et al., 2023), to a representation-learning transition (Liu et al., 2022), and to a shift from lazy to rich training dynamics (Kumar et al., 2024). In at least one clean setting it is now understood as a consequence of over-parameterization. Xu et al. (2026) (XVS) prove an end-to-end grokking guarantee for over-parameterized ridge regression trained by gradient descent with weight decay, and give quantitative bounds on the generalization delay $t _ { 2 } - t _ { 1 }$ in terms of the training hyperparameters, with $t _ { 2 } \propto 1 / \lambda$ as the weight decay $\lambda  0 .$ . Their analysis holds for any realizable teacher over any fixed feature map, and their experiments use an unstructured feature map, either the identity or i.i.d. Gaussian random features. In that regime the null space created by over-parameterization is isotropic. Every extra dimension carries a little signal, and weight decay resolves the delayed transition that defines grokking.

A natural question is whether the same picture holds when the feature map is structured and not random. Kam et al.’s solvable holomorphic model (Kam et al., 2026) shows that a modular target is representable if and only if its discrete Fourier support lies inside the class the architecture can express. We work with a symmetric-Laurent-band extension of that class (Section 2), which enlarges it from a positive Fourier line to a signed band so that rules such as $a - b$ become representable. That raises the dynamical sequel. Once a target lies inside the band, does training find it immediately, only after memorization, or not at all?

The natural expectation is grokking. XVS establish that a fixed feature map produces the memorize-then-generalize delay whenever the problem is over-parameterized and realizable, and a band-representable target on an over-parameterized band feature map appears to satisfy both conditions, so it should be memorized first and generalized later. They do not grok. At fixed positive weight decay, enlarging the band produces overfitting collapse and no delayed transition, and the outcome follows from which modes are active, not from the capacity ratio. We characterize an absence, meaning the conditions under which the memorize-then-generalize delay fails to appear, and trace it to a structural property of the feature map.

Representability on this feature map is decidable in closed form, inherited from the exact Fourier analysis of Kam’s holomorphic model and verified numerically at $\delta ^ { 2 } \sim 1 0 ^ { - 1 2 }$ for representable targets and $\delta ^ { 2 } = 1$ for the full-plane target ab (Section 2 and Appendix B), so the static question is settled before any dynamics are run. Enlarging the band for a representable target then degrades generalization monotonically instead of inducing grokking, with held-out accuracy falling from 1.00 to near chance as $q / n$ grows. A masking experiment at fixed nominal dimension isolates the active support as the variable that carries the efect (Section 3).

The band-limited character map is the right object for this test, not just a convenient one. XVS assume realizability over a fixed feature map, and on an unstructured map that assumption cannot be audited. Here the expressible class is a span of group characters, so realizability is decided by a Fourier support condition in closed form, with no training run. We can hold the hypothesis of their theorem fixed and vary only the geometry of the features, which is the comparison the question demands.

The structure of a feature map is a control variable for whether grokking occurs at all, and not a detail of the experimental setting.

## 2. Setup

Feature map. Fix an odd prime p and $\omega = e ^ { 2 \pi i / p }$ . A modular pair $( a , b ) \in \mathbb { Z } _ { p } ^ { 2 }$ is encoded through its Laurent harmonics: for a band radius M, the feature vector collects the characters $\chi _ { r , s } ( a , b ) = \omega ^ { r a + s b }$ for $- M \leq r , s \leq M$ , realized in real coordinates as cos and sin channels. The band is symmetric, $B _ { M } = - B _ { M }$ , and p is odd, so the conjugate pair $\pm ( r , s )$ satisfies cos $\langle - r , - s , \theta \rangle = \cos \langle r , s , \theta \rangle$ and sin $\langle - r , - s , \theta \rangle = - \sin \langle r , s , \theta \rangle$ . Each pair contributes two independent real functions, not four, and the $2 ( 2 M + 1 ) ^ { 2 }$ implemented channels span a function space of dimension only

$$
q = ( 2 M + 1 ) ^ { 2 } .\tag{1}
$$

The design matrix has numerical rank q at every M tested. We report the capacity ratio as $q / n$ throughout, since it is $q ,$ and not the number of implemented columns, that decides interpolation. Every feature entry has unit modulus. The symmetric Laurent band is introduced here. It admits negative frequencies through the two-channel realification $\chi = A + i B , \overline { { { \chi } } } = A - i B$ with $A = \cos$ and $B = \sin$ , and closes to $k B _ { M } = [ - k M , k M ] ^ { d }$ under a degree-k monomial. This real, signed-frequency realization is what places the model inside the real-valued ridge framework we use, and what makes signed rules such as $a - b$ representable where a positive Fourier line would exclude them.

Representability. A target f is representable if its Fourier support lies inside the band $A = \{ ( r , s ) : | r | , | s | \leq M \}$ , i.e. if the spectral gap

$$
\delta ^ { 2 } : = \sum _ { ( u , v ) \notin A } | \widehat { \omega ^ { f } } ( u , v ) | ^ { 2 } = 0 .\tag{2}
$$

Because the features are group characters, $\delta ^ { 2 }$ is a static, training-free quantity, decidable in closed form with no training run, a property the Gaussian feature map does not have. A linear rule $m a + n b$ is a single Fourier mode, representable as soon as $| m | , | n | \leq M$ , so the signed rule $a - b = { \mathrm { m o d e ~ } } ( 1 , - 1 )$ is representable at $M \geq 1$ , confirmed numerically at $\delta ^ { 2 } \sim 1 0 ^ { - 1 2 }$ . The multiplication rule ab has full-plane Fourier support and is representable at no finite M (a consequence of the quadratic Gauss sum). Every failure reported below is a failure to find a rule the class contains, never a failure to contain it. The derivation and the numerical check across targets and band radii are in Appendix B. The closed-form structure of modular-arithmetic solutions has been studied for two-layer networks by Gromov (2023) and extended to modular polynomials by Doshi et al. (2024). Our band feature map inherits this Fourier structure but freezes it into a fixed, checkable expressible class (Kam et al., 2026).

Model and training. We train a linear read-out $\hat { y } = \Phi \theta$ over the fixed feature map, minimizing mean-squared error with $\ell _ { 2 }$ regularization (ridge), by gradient descent with weight decay λ. The target phase $\omega ^ { f ( a , b ) }$ is regressed in (cos, sin) coordinates and decoded by angle. We train on a fraction of the $p ^ { 2 }$ pairs and evaluate held-out accuracy. We use $p = 9 7$ throughout. We train only the linear read-out, because the monomial activation of the parent holomorphic model acts as a frequency dilation $( B _ { M } \to k B _ { M } )$ and is not a trainable object. The two-layer non-convex case is outside our scope (Section 4). Following $\mathrm { X V S } , t _ { 1 }$ is the step at which training accuracy first exceeds threshold and $t _ { 2 }$ the step at which held-out accuracy does. A positive gap $t _ { 2 } - t _ { 1 }$ is the grokking signature. Full experimental detail, seeds, and code availability are in Appendix D.

## 3. Over-parameterization overfits, it does not grok

We train the linear read-out by gradient descent and vary the band radius M, equivalently the capacity ratio $q / n$ of Eq. (1), for the representable target $a - b$ , holding $\lambda = 1 0 ^ { - 2 }$ fixed. With $n = 3 7 6 3$ , over three seeds (Figure 1 left, values in Appendix D).

The XVS picture predicts the opposite. There, deeper over-parameterization amplifies grokking, so the delay grows and generalization eventually succeeds. Here it destroys generalization, and peak held-out accuracy falls monotonically to 0.066 at $q / n \approx 1 . 7$ , barely above the chance level $1 / 9 7$

![](images/179b5a99baad62f1af501221db5a9c42628a71c45e08c148be8105ca2f79ee16.jpg)

![](images/ad1c339a6a8ebe1690fea478d7a5fe0abce1cade44854aed6ae4d9f21f73de4a.jpg)  
active modes (nominal dimension fixed, M = 32)  
Figure 1: Left. Peak held-out accuracy versus the capacity ratio $q / n$ for the representable target $a - b .$ , with the interpolation threshold and the chance level $1 / 9 7$ marked. Accuracy has already fallen to 0.506 at $q / n = 0 . 8 6 3$ , to the left of the threshold. Right. Peak held-out accuracy versus the number of active modes at fixed nominal dimension, $M = 3 2$ , with n marked. Restricting the active support to 1089 modes gives perfect and zero-variance generalization, while the full band collapses. Both panels use the functional dimension of Eq. (1) and plot the values of Tables 2 and 3. Three seeds, with error bars where visible.

Generalization is badly degraded at $M = 2 4$ and $M = 2 8$ , where $q / n = 0 . 6 3 8$ and 0.863 and the model is under-parameterized. The degradation begins below the interpolation threshold and cannot be attributed to interpolation. Only from $M = 3 2$ does q exceed $n ,$ forcing a null space of dimension $q - n$ . The table is consistent with two regimes and not one transition, attenuation of small positive directions below the threshold and an irreducible floor above it, which we do not attempt to separate. The $M = 2 4$ row resembles grokking, with training accuracy saturating at 1.00 while held-out accuracy stalls at 0.91. It is not a delay, and the reason is structural. At $M = 2 4$ we have $q < n ,$ , so $\Phi _ { \mathrm { t r } }$ has full column rank q and the orthogonal complement of its row space inside the $2 q$ implemented coordinates is precisely the gauge redundancy of the duplicated realification of Eq. (1). Those directions carry no function at all, so the correct solution that weight decay is supposed to surface cannot be hiding in them. Appendix A gives the rank audit and the trajectory.

The mechanism: which modes are active. Why does the structured feature map overfit where the Gaussian one groks? The extra dimensions are not isotropic. The target $a - b$ occupies one Fourier mode, every other band mode is orthogonal to it on the population, and only the finite sample mixes them into the target direction. We test the role of the active support by fixing the nominal dimension at $M = 3 2$ and masking all but the modes within radius k of the origin (Figure 1 right).

At $k = 1 6$ the model is nominally 4225-dimensional but efectively 1089-dimensional and generalizes perfectly, while at $k = 3 2$ the same nominal dimension collapses. Overparameterization per se is not the cause. A large active band of signal-free structured modes is.

The count is a capacity proxy rather than the exact variable. Take an in-band singlecharacter teacher $e _ { t }$ and let $G = \Phi _ { \mathrm { t r } } ^ { \dagger } \Phi _ { \mathrm { t r } } / n$ be the empirical character Gram matrix. The terminal population error of the ridge fit is then the teacher-weighted spectral integral

$$
\lambda ^ { 2 } e _ { t } ^ { \dagger } ( G + \lambda I ) ^ { - 2 } e _ { t } = \int { \left( \frac { \lambda } { \mu + \lambda } \right) ^ { 2 } } \mathrm { d } \nu _ { t } ^ { G } ( \mu ) ,\tag{3}
$$

where $\nu _ { t } ^ { G }$ is the spectral measure of G viewed from the teacher (immediate from the normal equations and diagonalization). The number of active modes enters only through $\nu _ { t } ^ { G }$ , since enlarging the active support moves teacher-weighted mass towards zero, where the factor $( \lambda / ( \mu + \lambda ) ) ^ { 2 }$ attenuates it. Masking changes that measure decisively. It does not make the count the control variable, and a formula in $( q , n , \lambda )$ alone cannot be exact for every teacher.

Equation (3) also settles something the accuracy curves alone cannot, namely whether the collapse is an artefact of a finite training budget. It is not. The quantity in Eq. (3) is the error at the fixed point of the regularized objective and not an error after some number of steps, so at positive λ it bounds what any amount of further training can reach. Once teacher-weighted mass sits at eigenvalues $\mu \ll \lambda .$ , the factor $( \lambda / ( \mu + \lambda ) ) ^ { 2 }$ leaves that mass almost untouched and the optimum itself is wrong. Gradient descent converging to that fixed point converges to a predictor that never generalizes, which is why the delay is absent and not long.

## 4. Discussion

In the XVS setting the feature map is random and isotropic, so the null-space directions created by over-parameterization are statistically aligned with the target, and weight decay resolves them into the grokking transition. In the band setting those directions are orthogonal group characters carrying no target energy by construction. Weight decay removes them without surfacing a delayed correct solution, because no delayed correct solution is hiding there. The over-parameterization that grokking needs, structured features convert into overfitting. The sensitivity of grokking to experimental setting is well known, and what this adds is a cause for it.

The collapse curve of Figure 1 (left) can be read as a restatement of double descent, with over-parameterization past the interpolation threshold degrading test error and the mechanism reducing to classical bias–variance. That signal-free directions can hurt generalization is classical, and we do not claim it. The specific claim here is dynamical. XVS establish that in the same regime an unstructured feature map produces a memorize-then-generalize delay of order $1 / \lambda$ . A structured feature map, at fixed realizability and over-parameterization, does not shift that curve. It removes the delay, leaving either gapless generalization or collapse and nothing in between (Appendix A). A teacher-relevant direction with small positive $\mu$ is a slow clock and not a permanent obstruction, and positive weight decay attenuates such directions before they emerge. We therefore expect the delay to return as $\lambda  0 .$ , on a timescale set by the smallest teacher-weighted eigenvalue and not by the weight decay, and preliminary observation at $\lambda = 0$ is consistent with a long memorize-then-generalize separation at the band radii that collapse here. Our claim concerns the positive-ridge regime, where the delay is absent and not merely long. The zero-ridge limit is treated in the full-length version. Double descent describes where the test-error curve sits. This concerns whether the delay appears at all, which makes the two compatible but distinct.

The static representability framework is inherited from Kam et al.’s holomorphic model (Kam et al., 2026), and the symmetric-Laurent-band extension is a contribution of the present work. Solvable models of grokking have been studied in linear estimators (Levi et al., 2024) and for modular addition in two-layer quadratic networks (Mohamadi et al., 2024), but these operate in the regime where memorization is always possible and the feature map is unstructured or learned. The present result is complementary to XVS (Xu et al., 2026), delimiting the reach of their Gaussian-feature theorem by exhibiting a structured feature map on which its central phenomenon does not appear.

## 5. Conclusion

On a band-limited character feature map, a target the class provably contains is not learned by a memorize-then-generalize transition. Under positive weight decay it is learned at once when the active band is small and not at all when the active band is large, and the switch is governed by which modes are active, not by how many parameters there are.

The read-out is linear over a fixed feature map, so the two-layer jointly trained problem is untouched. The target carries a single mode, and a multi-mode target such as 2a − 3b would activate a broader signal-relevant support, moving the threshold in a direction we have not mapped. Every run uses one weight decay, $\lambda = 1 0 ^ { - 2 }$ , leaving open the zero-ridge limit, where slow teacher-relevant directions stop being attenuated. The threshold, which Appendix C brackets between 2401 and 3249 active modes at this $( \lambda , n )$ , is observed, not derived. Deriving it from λ, n, and band geometry is the next step, and Eq. (3) says where to look, since a closed form needs a limiting description of the teacher-weighted spectral measure.

## References

D. Doshi, T. He, A. Das, and A. Gromov. Grokking modular polynomials. In ICLR 2024 Workshop on Bridging the Gap between Practice and Theory in Deep Learning, 2024. arXiv:2406.03495.

A. Gromov. Grokking modular arithmetic. arXiv:2301.02679, 2023.

C.-F. Kam, X. Cadet, M. Bessafi, and F. Cadet. Algebraic representability as the limiting regime of grokking: An exactly solvable model with holomorphic activations. arXiv:2607.13749, 2026.

T. Kumar, B. Bordelon, S. J. Gershman, and C. Pehlevan. Grokking as the transition from lazy to rich training dynamics. In International Conference on Learning Representations (ICLR), 2024. arXiv:2310.06110.

N. Levi, A. Beck, and Y. Bar-Sinai. Grokking in linear estimators—a solvable model that groks without understanding. In International Conference on Learning Representations (ICLR), 2024. arXiv:2310.16441.

Z. Liu, O. Kitouni, N. Nolte, E. J. Michaud, M. Tegmark, and M. Williams. Towards understanding grokking: An efective theory of representation learning. arXiv:2205.10343, 2022.

M. A. Mohamadi, Z. Li, L. Wu, and D. J. Sutherland. Why do you grok? a theoretical analysis of grokking modular addition. In International Conference on Machine Learning (ICML), 2024. arXiv:2407.12332.

N. Nanda, L. Chan, T. Lieberum, J. Smith, and J. Steinhardt. Progress measures for grokking via mechanistic interpretability. In International Conference on Learning Representations (ICLR), 2023. arXiv:2301.05217.

A. Power, Y. Burda, H. Edwards, I. Babuschkin, and V. Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv:2201.02177, 2022.

J. A. Tropp. An introduction to matrix concentration inequalities. Foundations and Trends in Machine Learning, 8(1–2):1–230, 2015.

M. Xu, G. Vardi, and I. Safran. To grok grokking: Provable grokking in ridge regression. In International Conference on Machine Learning (ICML), 2026.

## Appendix A. The transient at M = 24 is onset of degradation, not grokking

The single point in Figure 1 (left) that superficially resembles grokking is $M = 2 4 ~ ( q / n =$ 0.638), where training accuracy still reaches 1.00 while held-out accuracy has fallen to 0.91. A train–test gap is the nominal signature of grokking, so the transient has to be separated from a memorize-then-generalize delay. This appendix does that in three steps, an operational criterion that already rules the point out, a rank argument showing that the mechanism XVS rely on cannot operate at this band radius, and the trajectory of Figure 2, which confirms both.

The criterion is not met. The grokking signature defined in Section 2 is a finite positive gap $t _ { 2 } - t _ { 1 }$ at the 0.99 threshold. At $M = 2 4$ the peak held-out accuracy over the whole run is 0.912, which never crosses the threshold, so $t _ { 2 }$ does not exist and $t _ { 2 } - t _ { 1 }$ is undefined. A delayed transition that never arrives is not a long delay. The same holds at every larger band radius, where peak accuracy is lower still. The nominal train–test gap at $M = 2 4$ is therefore a gap in the sense of two numbers difering, not in the sense the grokking literature uses.

Two null spaces, only one of which exists here. Write $\Phi _ { \mathrm { t r } } = U S V ^ { \top }$ for the thin SVD and $P _ { \parallel } = V _ { r } V _ { r } ^ { \top }$ for the projector onto the row space of rank $r ,$ so that the component invisible to the training data is $\theta _ { \perp } = ( I - P _ { \| } ) \theta$ . Under XVS, grokking is driven by the slow decay of a signal-bearing $\theta _ { \perp }$ , the delayed generalization occurring as weight decay resolves a correct solution hidden there. Whether such a solution can exist depends on where the null space comes from, and on this feature map there are two separate sources.

The first is the duplicated realification of $\operatorname { E q . }$ (1). The implementation instantiates $2 q$ channels spanning a space of dimension $q ,$ so $q$ coordinate directions are pure gauge and carry no function whatever the sample. The second is sampling, and it appears only once $q > n$ , with dimension $q - n$

At M = 24 only the first source is present. Here $q = 2 4 0 1 < n = 3 7 6 3$ . We verify numerically that $\Phi _ { \mathrm { t r } }$ attains full column rank 2401 on the deduplicated basis, with smallest singular value 2.08 against a largest of 70.7, so the rank is resolved with a wide margin and is not a threshold artefact. The complement of the row space inside the 4802 implemented coordinates therefore has dimension $4 8 0 2 - 2 4 0 1 = 2 4 0 1$ , which is the gauge dimension. Every direction weight decay grinds down is a direction in which the predictor does not move. No delayed solution can be hiding in them, and the XVS mechanism has nothing to act on. The situation changes only above the threshold, where $M = 3 2$ gives $q - n = 4 6 2$ genuinely unsampled functional directions, and those are the ones that produce the irreducible floor discussed in Section 3.

The decay of $\lVert \theta _ { \perp } \rVert$ at $M = 2 4$ is therefore a statement about coordinates and not about predictions, so it cannot be read as evidence of a resolving null space. The transient is likewise not a consequence of over-parameterization, since at $M = 2 4$ the model is under-parameterized on the axis that matters.

The trajectory. Figure 2 makes this concrete for the single-mode target $a - b$ at $M = 2 4$ $( \lambda = 1 0 ^ { - 2 } , \nu = 5$ , learning rate 0.05). The null-space mass $\lVert \theta _ { \perp } \rVert$ decays monotonically from 228 to $1 0 ^ { - 3 }$ over roughly 25,000 steps, while held-out accuracy reaches its ceiling of 0.91 by step $\sim 1 0 { , } 0 0 0$ and then does not move. Throughout the second half of training, while $\theta _ { \perp }$ is still being ground to zero, held-out accuracy is flat. There is no late upswing coinciding with the resolution of the null space, and that upswing is the defining signature of grokking.

The reading is falsifiable in the obvious way. Had the curve risen late, in the window where $\lVert \theta _ { \perp } \rVert$ crosses through the values that weight decay is removing, the XVS account would have been the right one even at this band radius. It does not, and the rank arithmetic above says why it could not have. We read the M = 24 gap as the onset of degradation, with training accuracy saturating first, held-out accuracy already short of one, and both worsening as the band grows. The transient is not reported as grokking anywhere in the main text.

Scope. The diagnostic is run at one band radius, one seed, and one weight decay, with the rank convention recorded in Appendix D. The gauge argument is exact and needs neither of these choices. The trajectory is a single realization and is ofered as confirmation, not as the load-bearing evidence.

## Appendix B. Closed-form representability in the character basis

This appendix records why the spectral gap $\delta ^ { 2 }$ is cheaply computable, a property specific to the group-character feature map, and reports the numerical check summarized in Section 2.

Let $G = \mathbb { Z } _ { p } ^ { 2 }$ with characters $\chi _ { u , v } ( a , b ) = \omega ^ { u a + v b }$ , orthonormal under the normalized inner product $\begin{array} { r } { \langle f , g \rangle = | G | ^ { - 1 } \sum _ { ( a , b ) } f ( a , b ) \overline { { g ( a , b ) } } } \end{array}$ . The band feature map spans the subspace ${ \mathcal { F } } _ { A } = \operatorname { s p a n } \{ \chi _ { u , v } ~ : ~ ( u , v ) \in A \}$ with $A = \{ | u | , | v | \leq M \}$ . Because the characters are orthonormal, for any target $\omega ^ { f }$ the best band approximation is the orthogonal projection onto ${ \mathcal { F } } _ { A }$ , and the residual energy is the out-of-band Fourier mass:

$$
\delta ^ { 2 } = \bigl \| \omega ^ { f } - \mathrm { P r o j } _ { \mathcal { F } _ { A } } \omega ^ { f } \bigr \| ^ { 2 } = \sum _ { ( u , v ) \notin A } \widehat { \left| \omega ^ { f } ( u , v ) \right| ^ { 2 } } .\tag{4}
$$

![](images/45a2e9f37f3457809147d29304c7e39f6221b71ee677c12142063c877654a647.jpg)  
Figure 2: Single-mode target $a - b$ at $M = 2 4 \ ( q / n = 0 . 6 3 8 )$ . The null-space mass $\lVert \theta _ { \perp } \rVert$ (red, log scale) decays to numerical zero, but held-out accuracy (blue dashed) plateaus at 0.91 and never reaches one, so the resolution of the null space is not accompanied by a delayed rise in generalization. This distinguishes the transient from grokking, in which the correct solution surfaces as the null-space component decays.

<table><tr><td>Target</td><td>M</td><td>q</td><td> $\delta ^ { 2 }$ </td><td>train acc</td><td>held-out acc</td></tr><tr><td> $a + b$ </td><td>1</td><td>9</td><td> $7 . 0 \times 1 0 ^ { - 1 3 }$ </td><td>1.000</td><td>1.000</td></tr><tr><td> $a + b$ </td><td>24</td><td>2401</td><td> $1 . 7 \times 1 0 ^ { - 1 7 }$ </td><td>1.000</td><td>1.000</td></tr><tr><td> $a - b$ </td><td>1</td><td>9</td><td> $1 . 3 \times 1 0 ^ { - 1 2 }$ </td><td>1.000</td><td>1.000</td></tr><tr><td> $a - b$ </td><td>24</td><td>2401</td><td> $1 . 8 \times 1 0 ^ { - 1 7 }$ </td><td>1.000</td><td>1.000</td></tr><tr><td> $a b$ </td><td>1</td><td>9</td><td>1.00</td><td>0.013</td><td>0.012</td></tr><tr><td> $a b$ </td><td>3</td><td>49</td><td>1.00</td><td>0.011</td><td>0.012</td></tr><tr><td> $a b$ </td><td>24</td><td>2401</td><td>2.54</td><td>0.109</td><td>0.038</td></tr></table>

Table 1: Closed-form ridge solution across targets and band radii. Representable targets have $\delta ^ { 2 }$ at numerical zero. The full-plane target ab has $\delta ^ { 2 } = 1$ at every band, and the $\delta ^ { 2 } > 1$ entry is the finite-sample efect discussed at the end of this appendix. Dimensions are functional (q), per Eq. (1).

The quantity is static and training-free, and representability $\left( \delta ^ { 2 } = 0 \right)$ is decided by whether $\operatorname { s u p p } ( { \widehat { \omega } } ^ { f } ) \subseteq A$

A point of bookkeeping applies to every row of Table 1 and not only to the anomalous one. Equation (4) is the residual of the exact orthogonal projection onto ${ \mathcal { F } } _ { A }$ , computed against the full population of $p ^ { 2 }$ pairs. The tabulated numbers are instead the residual of the ridge solution fitted on a subsample of n points at $\lambda = 1 0 ^ { - 6 }$ and then evaluated on all $p ^ { 2 }$ pairs, normalized by the target energy. The two agree to numerical precision when the target is representable, because the projection is then exact and the fit recovers it. For an unrepresentable target the two separate, and the diference is the over-sample error discussed at the end of this appendix. We report the fitted quantity because it is what the experiments compute, and we give the population values below wherever they difer.

The accuracy columns are angular. A predicted phase is decoded to the nearest class, so $\delta ^ { 2 }$ and accuracy are diferent functionals of the same residual and need not move together in general. On the rows tabulated here they do. The residual is either at numerical zero, which gives exact recovery, or comparable to the total target energy, which gives chance.

Linear rules. A linear-phase target $m a + n b$ is the single character $\chi _ { m , n } ,$ , so its Fourier support is the point $( m , n )$ and $\delta ^ { 2 } = 0 \ \mathrm { i f f } \ | m | , | n | \leq M$ . The signed rule $a - b = \chi _ { 1 , - 1 }$ is representable at every $M \geq 1$ , the extension that inverse phases (the two-channel Laurent construction) provide over a positive Fourier line, and Table 1 confirms it at $\delta ^ { 2 } \sim 1 0 ^ { - 1 2 }$ already at $M = 1$ . The main text relies on one consequence. Any dynamical failure of $a - b$ at small M, and we do observe such failures under a nonlinear read-out, is an optimization artifact and not a representability barrier.

The multiplication rule. For $f = a b .$ , the Fourier coeficient is

$$
\widehat { \omega ^ { a b } } ( u , v ) = \frac 1 { p ^ { 2 } } \sum _ { a , b } \omega ^ { a b - u a - v b } = \frac { \omega ^ { - u v } } { p } ,\tag{5}
$$

using $\begin{array} { r } { \sum _ { a } \omega ^ { a ( b - u ) } = p [ b \equiv u ] } \end{array}$ and then summing over b. Every coeficient has modulus $1 / p$ regardless of $( u , v )$ , so the spectrum of $\omega ^ { a b }$ is flat over the whole plane $\mathbb { Z } _ { p } ^ { 2 }$ . The derivation uses nothing beyond character orthogonality, and the flatness is the discrete counterpart of a chirp being spread across all frequencies.

First, the in-band energy fraction is $| A | / p ^ { 2 } = q / p ^ { 2 }$ , so the population residual is

$$
\delta _ { \mathrm { p o p } } ^ { 2 } ( a b ) = 1 - \frac { q } { p ^ { 2 } } ,\tag{6}
$$

which equals 0.9990, 0.9948 and 0.7448 at $M = 1 , 3 , 2 4$ . Second, $\delta _ { \mathrm { p o p } } ^ { 2 } = 0$ requires $q = p ^ { 2 }$ that is $M = ( p - 1 ) / 2 = 4 8$ and a band equal to the entire character basis. There is no partial remedy. A model that represents ab on $\mathbb { Z } _ { p } ^ { 2 }$ must be able to represent every function on $\mathbb { Z } _ { p } ^ { 2 } ,$ which is why ab is the irreducible obstruction of the parent holomorphic model and not a target that a wider band would eventually reach.

Why $\delta ^ { 2 }$ can exceed one. Comparing the fitted values against Eq. (6) shows that the excess is present on every unrepresentable row and merely becomes conspicuous on one of them.
<table><tr><td>M</td><td>q</td><td>fitted  $\delta ^ { 2 }$ </td><td>population  $\delta _ { \mathrm { p o p } } ^ { 2 }$ </td></tr><tr><td>1</td><td>9</td><td>1.0004</td><td>0.9990</td></tr><tr><td>3</td><td>49</td><td>1.0038</td><td>0.9948</td></tr><tr><td>24</td><td>2401</td><td>2.5365</td><td>0.7448</td></tr></table>

The fitted residual exceeds the population residual at all three band radii, by $1 0 ^ { - 3 }$ at $M = 1$ and by a factor of 3.4 at $M = 2 4$ , and the gap widens with q. The reason is that the fitted coeficients minimize error on a subsample of n points and not on the population, so they difer from the orthogonal projection, and the of-sample error they incur adds to rather than subtracts from the residual of Eq. (4). The measured quantity is an over-sample error and is not bounded by the target energy, which is why a number above one is not a contradiction.

This is not an interpolation efect. At M = 24 the functional dimension is $q = 2 4 0 1 <$ $n = 3 7 6 3$ , so the system is overdetermined and the training set cannot be interpolated. What grows with $q$ is the number of in-band directions available to absorb subsample-specific structure, and at $\lambda = 1 0 ^ { - 6 }$ almost nothing restrains them. The same mechanism is what Section 3 exhibits dynamically for a representable target, where it costs generalization instead of showing up as an inflated residual.

## Appendix C. A heuristic for the collapse threshold

We give an order-of-magnitude account of where the collapse threshold should sit, and then bracket it against the data. This is a heuristic, not a theorem, and a rigorous treatment is deferred.

Let $m _ { \mathrm { a c t } }$ be the number of active modes, of which one is signal-relevant (the target mode) and $m _ { \mathrm { a c t } } - 1$ are signal-free. On the training subsample of size $n ,$ the ridge solution fits both the target mode and the empirical projections of the signal-free modes, and the latter contribute out-of-sample error controlled by their finite-sample aliasing. For orthonormal characters $\chi _ { j }$ and $\chi _ { t }$ with $\begin{array} { r } { j \neq t , } \end{array}$ , the empirical inner product $\begin{array} { r } { n ^ { - 1 } \sum _ { x \in S } \chi _ { j } ( x ) \overline { { \chi _ { t } ( x ) } } } \end{array}$ has mean zero, since each summand has mean zero over the population. Its second moment is $n ^ { - 2 }$ times the variance of the sum. The summands are uncorrelated and of unit modulus, so that variance is $n ,$ giving

$$
\mathbb { E } { \left| \langle \chi _ { j } , \chi _ { t } \rangle _ { S } \right| } ^ { 2 } = \frac { 1 } { n } \left( 1 - \frac { n - 1 } { N - 1 } \right) \leq \frac { 1 } { n } ,\tag{7}
$$

where the bracket is the finite-population correction for sampling without replacement and equals 0.60 at our $n / N$ . Summing over the $m _ { \mathrm { a c t } } - 1$ signal-free modes gives an aliasing energy of order $( m _ { \mathrm { a c t } } - 1 ) / n$ . Weight decay λ shrinks each spurious coeficient, suppressing this energy by a factor that saturates once $m _ { \mathrm { a c t } } / n$ is order one. The transition to collapse is therefore expected near

$$
m _ { \mathrm { a c t } } ^ { \star } \sim { } c ( \lambda ) n ,\tag{8}
$$

with $c ( \lambda )$ an $O ( 1 )$ factor increasing with the regularization strength.

Bracketing the constant. The masking sweep alone leaves a wide window, since it samples only 1089 (perfect) and 4225 (collapsed). The capacity sweep supplies intermediate points at the same λ and the same n, and the two can be placed on a single axis of active modes. Doing so is not neutral. It presumes the claim of Section 3, that only the active support matters and the nominal dimension does not, so a masked band with $m _ { \mathrm { a c t } }$ active modes should behave like an unmasked band with $q = m _ { \mathrm { a c t } }$ , and where the two sweeps nearly meet they do.

<table><tr><td> $m _ { \mathrm { a c t } }$ </td><td>peak held-out acc</td><td>source</td></tr><tr><td>1089</td><td>1.000</td><td>masked, k = 16</td></tr><tr><td>1369</td><td>0.999</td><td>unmasked, M = 18</td></tr><tr><td>1849</td><td>0.992</td><td>unmasked, M = 21</td></tr><tr><td>2401</td><td>0.912</td><td>unmasked, M = 24</td></tr><tr><td>3249</td><td>0.506</td><td>unmasked, M = 28</td></tr><tr><td>4225</td><td>0.185</td><td>both, M = 32</td></tr></table>

The two independent routes to 4225 agree, and 1089 masked sits beside 1369 unmasked with a diference of 0.001. The combined curve narrows the window considerably. Measurable degradation begins near $m _ { \mathrm { a c t } } = 2 4 0 1$ , which is 0.638 n, and half-height is reached near 3249, which is 0.863 n. These are the same two ratios that appear as $q / n$ in Table 2, since the axis is the same. Reading the half-height point as the threshold gives $c ( 1 0 ^ { - 2 } ) \approx 0 . 8 6$ , which is the order-unity constant the argument predicts without fixing.

What this does not establish. The scaling $m _ { \mathrm { a c t } } ^ { \star } \propto n$ is untested, since every run here uses one n. The dependence of c on λ is likewise untested, since every run uses $\lambda = 1 0 ^ { - 2 }$ so the claim that c increases with regularization is an expectation from Eq. (7) and not a measurement. The same character second-moment computation controls both the aliasing energy and the conditioning of the empirical Gram matrix, and we expect a full-length analysis, via matrix concentration bounds on the empirical Gram (Tropp, 2015), to yield $c ( \lambda )$ in closed form.

Finally, the count is a proxy. The quantity appearing above is $m _ { \mathrm { a c t } } .$ the number of activated signal-free modes, and not the nominal dimension, which is the distinction the masking experiment isolates. By Eq. (3), however, the count acts on the error only through the teacher-weighted spectrum, and Eq. (7) averages over modes where the exact object does not. That is why we present this as a scaling and not as a law, and why the bracket above is a description of one (λ, n, band) configuration rather than a threshold formula.

## Appendix D. Experimental detail and measured values

Tables 2 and 3 are the numbers plotted in Figure 1, with seed-to-seed standard deviations, and the protocol that produced them is recorded below.

Setup. All experiments use $p = 9 7$ . Features are the real $( \cos / \sin )$ channels of the band characters. The implementation instantiates all $2 ( 2 M + 1 ) ^ { 2 }$ channels over the full symmetric band. By Eq. (1) these span a space of functional dimension $q = ( 2 M + 1 ) ^ { 2 }$ , and every capacity figure quoted in this paper is the functional value q. The masked variant zeros all channels outside keep-radius k, symmetrically in $\pm ( r , s )$ , while preserving the number of instantiated columns. A single train/test split at fraction 0.4 is drawn per seed. The seed is set once, a uniform permutation of the $p ^ { 2 } = 9 4 0 9 $ cells is drawn, and its first $n = \lfloor 0 . 4 p ^ { 2 } \rfloor = 3 7 6 3$ entries form the training set. The seed therefore fixes both the split and the initialization, and every run is reproducible from it alone. Held-out accuracy is measured on the remaining 5646 cells.

<table><tr><td>M</td><td>q</td><td> $q / n$ </td><td>peak held-out acc  $( \mathrm { m e a n } \pm \mathrm { s t d } )$ </td></tr><tr><td>18</td><td>1369</td><td>0.364</td><td> $0 . 9 9 9 \pm 0 . 0 0 1$ </td></tr><tr><td>21</td><td>1849</td><td>0.491</td><td> $0 . 9 9 2 \pm 0 . 0 0 2$ </td></tr><tr><td>24</td><td>2401</td><td>0.638</td><td> $0 . 9 1 2 \pm 0 . 0 0 6$ </td></tr><tr><td>28</td><td>3249</td><td>0.863</td><td> $0 . 5 0 6 \pm 0 . 0 0 6$ </td></tr><tr><td>32</td><td>4225</td><td>1.123</td><td> $0 . 1 8 5 \pm 0 . 0 0 6$ </td></tr><tr><td>40</td><td>6561</td><td>1.744</td><td> $0 . 0 6 6 \pm 0 . 0 0 4$ </td></tr></table>

Table 2: Peak held-out accuracy versus capacity for the representable target $a - b ,$ with $n = 3 7 6 3$ . Accuracy falls monotonically toward chance as $q / n$ grows. The first two collapse points, $M = 2 4$ and $M = 2 8$ , lie below the interpolation threshold $q / n = 1$

<table><tr><td>keep radius k active modes</td><td></td><td>s peak held-out acc  $( \mathrm { m e a n } \pm \mathrm { s t d } )$ </td></tr><tr><td>1</td><td>9</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>2</td><td>25</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>4</td><td>81</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>8</td><td>289</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>16</td><td>1089</td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>32</td><td>4225</td><td> $0 . 1 8 5 \pm 0 . 0 0 6$ </td></tr></table>

Table 3: Masking at fixed nominal dimension, $M = 3 2$ , with masked channels zeroed and not removed. Active modes are counted functionally as $( 2 k + 1 ) ^ { 2 }$ . Restricting the active support to any radius up to 1089 modes, while keeping the target mode, gives perfect and zero-variance generalization. Only the full 4225-mode band collapses.

Null-space diagnostic. The trajectory of Appendix A is a separate run at $M = 2 4$ with the same hyperparameters and seed 0. The thin SVD of $\Phi _ { \mathrm { t r } }$ is taken once before training, and $\lVert \theta _ { \perp } \rVert$ is recorded on the same 100-step grid as the accuracies. The rank r is fixed by a relative threshold of $1 0 ^ { - 8 }$ on the leading singular value.

Penalty normalization. Weight decay is applied uniformly to all $2 ( 2 M + 1 ) ^ { 2 }$ instantiated coordinates. Because each function is represented by a conjugate pair of coordinates that gradient descent loads symmetrically, a coeficient of size c in the deduplicated orthonormal basis is penalized as $2 ( c / 2 ) ^ { 2 } = c ^ { 2 } / 2$ in the implemented one. The quoted $\lambda = 1 0 ^ { - 2 }$ is therefore the value in the implemented parameterization and corresponds to $5 \times 1 0 ^ { - 3 }$ in the deduplicated basis. The two parameterizations span the same predictors, but numerical values of λ are not comparable across them, and we carry the implemented value throughout. Closed-form results solve $( \Phi _ { \mathrm { t r } } ^ { \top } \Phi _ { \mathrm { t r } } + \lambda I ) \theta = \Phi _ { \mathrm { t r } } ^ { \top } Y$ with $\lambda = 1 0 ^ { - 6 }$ and report the population residual over all $p ^ { 2 }$ points. Dynamical results use full-batch gradient descent on the meansquared error. Weight decay is supplied by the optimizer and not by an explicit penalty term (SGD, learning rate 0.05, weight decay $\lambda = 1 0 ^ { - 2 }$ , no momentum). The read-out is a single matrix $\theta \in \mathbb { R } ^ { 2 ( 2 M + 1 ) ^ { 2 } \times 2 }$ with independent $\mathcal { N } ( 0 , \nu ^ { 2 } )$ entries at $\nu = 5$ and no bias term. The target phase is regressed in (cos, sin) coordinates, and a prediction counts as correct when its angle lies within $\pi / p$ of the target angle, the half-width of the decision wedge for $p$ classes. Figure 1 uses three seeds {0, 1, 2}. Peak held-out accuracy is the best value attained at any evaluation point over the whole run, sampled every 100 steps. It is an upper envelope and not a terminal value, so it lies slightly above the ridge fixed point of Eq. (3). The two panels use diferent budgets: 60,000 steps for the capacity sweep and 30,000 for the masking sweep. The budgets are consistent where they overlap: the $k = 3 2$ masking configuration is the unmasked $M = 3 2$ band and reaches 0.185 under both, indicating that 30,000 steps already sufice for convergence in that regime. Every number reported in this paper follows from these settings and the listed seeds, with no per-configuration tuning. Code and seeds are available from the authors upon request.