# Provable Adversarial Detection: Prime Quantization meets Gromov–Wasserstein

Manohar Kaul, Sadbhavana Babar, Aditya Ramesh, Aditya Saibewar, Fumiyoshi Kasahara, Maki Inui, Hiroaki Kurihara

Fujitsu Research, Japan Fujitsu Research of India Pvt. Ltd. kaul.manohar@fujitsu.com

## Abstract

Adversarial vulnerability persists across modern vision architectures from CNNs to vision language models (VLMs), yet existing detection methods rely on heuristics without theoretical guarantees. We address the fundamental question of when adversarial perturbations can be provably detected from a geometric perspective. Our key insight is that adversarial perturbations cannot simultaneously preserve geometric structure across spaces with fundamentally different properties. Accordingly, we construct two such complementary metric spaces. First, we use a standard CNN embedding space Z, where adversarial samples exhibit significant displacement patterns. Second, we build a novel prime-quantized space P, that absorbs small perturbations through number-theoretic discretization, resulting in minimal displacement, while preserving discriminability. We then leverage the geometric discrepancies across spaces Z and P to detect adversarial samples. To the best of our knowledge, we establish the first rigorous separation theory for adversarial detection, proving that adversarial samples create unavoidable geometric inconsistencies across both spaces. Our framework provides theoretical guarantees including pixel-level absorption bounds, neighborhood diameter concentration, Gromov-Wasserstein (GW) separation theorems, and practical risk control. Extensive experiments validate our theoretical predictions and achieve consistently strong detection performance across a wide range of attack types and model families.

## 1 Introduction

Vision systems have rapidly progressed from CNNs He et al. (2016) to Vision–Language Models (VLMs) Radford et al. (2021) and multimodal architectures Hurst et al. (2024), yet adversarial vulnerability persists across all these paradigms. As these increasingly capable models are deployed at scale, the consequences of undetected adversarial attacks also scale, making detection a core safety requirement.

Defenses fall into three main families. Adversarial training (Madry et al., 2018; Zhang et al., 2019; Gowal et al., 2021) augments models with adversarial examples. While effective in restricted scenarios, it requires expensive retraining and often fails to generalize across diverse attacks, including gradient-based (Goodfellow et al., 2014; Madry et al., 2018; Carlini & Wagner, 2017), physical (Brown et al., 2017), and natural corruptions (Hendrycks & Dietterich, 2019b; Engstrom et al., 2019). Detection methods (Metzen et al., 2017; Feinman et al., 2017; Ma et al., 2018; Lee et al., 2018; Ma et al., 2019; Meng & Chen, 2017; Lee et al., 2024) rely on auxiliary classifiers or statistical tests, but remain heuristic and easily broken by adaptive adversaries (Athalye et al., 2018). Certifiable robustness (Raghunathan et al., 2018; Cohen et al., 2019; Salman et al., 2019) provides provable invariance regions, but targets robust classification rather than detection, and is computationally intensive. Trade-off frameworks such as TRADES (Zhang et al., 2019) deepened our theoretical understanding, but left unanswered the key question: “Can detection itselfbe endowed with guarantees, and what properties make adversarial examples inherently detectable?"

Our insight. Clean and adversarial samples leave distinct geometric traces across two complementary spaces, namely, the CNN embedding space Z and a prime-quantized space P. In $Z ,$ , clean samples form tight neighborhoods, while adversarial ones exhibit characteristic displacements that disrupt local structure. In $P ,$ each pixel is discretized by rounding to nearby primes under a secret bit mask, so small perturbations are either absorbed within prime gaps or forced into discrete jumps. This mechanism preserves overall discriminability yet creates systematic cross-space inconsistencies, making adversarial inputs detectable through geometric analysis.

To the best of our knowledge, we establish the first rigorous geometric separation theoryfor adversarial detection, spanning four levels of guarantees: (i) pixel-level absorption bounds proving when perturbations vanish in $P$ or cross prime-gap boundaries, (ii) K-NN diameter envelopes showing clean and adversarial neighborhoods diverge differently in Z and P, (iii) cross-space separation theorems based on Gromov–Wasserstein (GW) distances that yield a non-vanishing gap scaling with dimension and perturbation strength, and (iv) risk control guarantees establishing that simple thresholding achieves bounded misclassification rates. Together, these results provide a principled foundation for adversarial detection, addressing the open question of when and why detection must succeed.

Contributions. (i) We introduce prime quantization, a cryptographically inspired discretization that generalizes across CNN, VLM, and multimodal architectures, and could extend to other one-way transforms. (ii) We present a unified theoretical framework proving that adversarial perturbations necessarily induce cross-space inconsistencies, with guarantees from pixel absorption up through GW-based separation. (iii) We empirically validate our method on a broad suite of attacks, VLM zero shot settings, and adaptive adversaries, demonstrating consistent and strong performance compared to state-of-the-art defenses.

## 2 Related Work

Adversarial attacks are categorized by attacker knowledge into: (i) white-box (FGSM (Goodfellow et al., 2014), PGD (Madry et al., 2018), C&W (Carlini & Wagner, 2017)), (ii) black-box (ZOO (Chen et al., 2017), Square (Andriushchenko et al., 2020)), and (iii) adaptive attacks that exploit defense mechanisms (Athalye et al., 2018), often defeating methods that appear robust under non-adaptive evaluation.

Detection methods include: (i) autoencoder-based reconstruction (MagNet (Meng & Chen, 2017), PixelDefend (Song et al., 2018)), (ii) distributional analysis (Mahalanobis (Lee et al., 2018)), (iii) prediction differences (Feature Squeezing (Xu et al., 2018)), and (iv) learned classifiers (MetaAdvDet (Ma et al., 2019)). These approaches remain heuristic and are routinely bypassed by adaptive adversaries, with no guarantees on when detection must succeed.

Robust training and certification methods such as adversarial training (Madry et al., 2018), TRADES (Zhang et al., 2019), and certified defenses based on randomized smoothing (Cohen et al., 2019) or patch-based strategies (Xiang et al., 2022) aim at robust classification rather than detection, often requiring retraining and incurring accuracy trade-offs.

In contrast, our work provides the first theoretical guarantees that adversarial perturbations create unavoidable cross-space inconsistencies, yielding a principled basis for detection with quantifiable confidence. Unlike most prior defenses, we further evaluate on vision–language models in zero-shot settings, highlighting robustness beyond CNN benchmarks.

## 3 Preliminaries

We introduce notation and the prime-quantized space used by our detector.

Let $X \in [ 0 , 1 ] ^ { N \times d }$ be N images, each $x _ { i } \in [ 0 , 1 ] ^ { d }$ a flattened vector of d normalized pixels with label $y _ { i } \in \dot { Y } \dot { ( | Y | = C ) }$ . A classifier $f _ { \theta } = g _ { \theta } \circ h _ { \theta }$ (where $h _ { \theta }$ is the feature extractor and $g _ { \theta }$ the classification head) has embedding $Z _ { i } = h _ { \theta } ( x _ { i } ) \in \mathbb R ^ { m }$ , with $Z = [ \dot { Z } _ { 1 } , \dots , Z _ { N } ] ^ { T }$ . With a slight abuse of notation, we also use $Z$ to denote the ambient embedding space R<sup>m</sup> equipped with the Euclidean metric $d _ { Z } ( z _ { i } , z _ { j } ) = \| z _ { i } - z _ { j } \| _ { 2 }$ . Thus, each $Z _ { i }$ is both a row of the embedding matrix and a point in metric space $( \breve { Z } , d _ { Z } )$

Definition 1 (Adversarial perturbation). Given $x _ { i }$ , a perturbation $\eta \in \mathbb { R } ^ { d }$ yields $\tilde { x } _ { i } = x _ { i } + \eta .$ . It is ϵ-bounded $i f \| \eta \| _ { \infty } \leq \epsilon$

To defend against such perturbations, we transform images into a discrete prime space via three steps: (i) scale to integers, (ii) round to primes, and (iii) rescale.

Definition 2 (Scaling). For $k \in \mathbb { N } ,$ let $\mathbb { P } _ { k } = \{ p \le 1 0 ^ { k }$ : p prime}. Define $S _ { k } ( x _ { i , j } ) = \left\lfloor x _ { i , j } 1 0 ^ { k } \right\rfloor$ and $S _ { k } ^ { - 1 } ( n ) = n \cdot 1 0 ^ { - k }$

Definition 3 (Prime rounding). Given secret bit $b _ { j } \in \{ 0 , 1 \}$ , map integer n to the nearest prime in $\mathbb { P } _ { k } . \ I f p _ { \ell } < n < p _ { \ell + 1 }$ , where $p _ { \ell } , p _ { \ell + 1 } \in \mathbb { P } _ { k } ,$ , then $R _ { k } ^ { ( 0 ) } ( n ) = p _ { \ell } , R _ { k } ^ { ( 1 ) } ( n ) = p _ { \ell + 1 }$

Definition 4 (Prime quantization). The pixel transform is $T _ { k } ^ { ( b _ { j } ) } ( x _ { i , j } ) \ : = \ : S _ { k } ^ { - 1 } ( R _ { k } ^ { ( b _ { j } ) } ( S _ { k } ( x _ { i , j } ) ) )$ Extending component-wise yields $T _ { k } ^ { ( b ) } : [ 0 , 1 ] ^ { d } \to [ 0 , 1 ] ^ { d }$

Example 1. For $k = 2 , x _ { i , j } = 0 . 3 8$ gives $S _ { 2 } = 3 8 ,$ , between 37 and 41. Then $T _ { 2 } ^ { ( 0 ) } = 0 . 3 7$ $T _ { 2 } ^ { ( 1 ) } = 0 . 4 1$ . A perturbation 0.385 still maps to 38, hence quantization is unchanged.

Space transformation. $T _ { k } ^ { ( b ) }$ maps images into discrete $P ,$ , where prime gaps and secret $( b , k )$ yield irregular, attacker-unpredictable rounding. Unlike uniform quantization, prime rounding introduces structured but unpredictable discretization.

Problem statement. Given $x \in [ 0 , 1 ] ^ { d }$ , construct a detector $D ( x ) \in \{ \mathrm { c l e a n } , \mathrm { a d v } \}$ by comparing $h _ { \theta } ( x ) \in Z$ and $T _ { k } ^ { ( b ) } ( x ) \in { \cal P }$ , ensuring w.h.p. that clean inputs agree across spaces while adversarial ones create detectable discrepancies.

## 4 Multiscale Gromov-Wasserstein (GW) adversarial detector

Our method compares neighborhood behaviors of samples across the embedding space $Z$ and the prime-quantized space P, exploiting their complementary geometries.

Clean neighborhoods. In $Z ,$ clean samples cluster by class, so a sample’s local neighborhood is dominated by its true label and its global neighborhood aligns with class centroids. In contrast, prime quantization scatters samples uniformly in $P ,$ , destroying spatial coherence, resulting in neighborhoods that show nearly uniform label distributions without clustering.

Adversarial neighborhoods. In $Z ,$ adversarial samples jump from their true cluster toward a wrong class, shifting both local and global label distributions. In $P ,$ prime quantization often reduces the impact of small perturbations, depending on the quantization gaps and perturbation strength. Therefore, adversarial neighborhoods tend to resemble their clean counterparts more closely. This creates systematic cross-space discrepancies between Z and $P .$

We next formalize these ideas via neighborhood maps and quantify their mismatches using Gromov-Wasserstein (GW) distances.

![](images/c93e8f3d9a24169717105c409b05c9f46c87a1ad30b9864fb4be5f4f019bf982.jpg)  
Figure 1: Geometry in Z (clusters) vs. P (dispersed).

Definition 5 (Neighborhood map and induced distributions). Let $( \mathsf { M } , d _ { \mathsf { M } } )$ be a metric space. For a query $q \in { \mathsf { M } }$ and integer $K \geq 1$ , a neighborhood map $\dot { \mathcal { N } } _ { K } ^ { \sf M } : \mathsf { M } \to 2 ^ { \mathsf { M } }$ returns a set ofK reference points, defining local neighborhoods when K corresponds to nearest neighbors or global neighborhoods when K corresponds to class centroids. The corresponding spatial distribution is $\begin{array} { r } { \mu _ { K } ^ { \mathsf { M } } ( q ) \ = \ \frac { 1 } { K } \sum _ { z \in \mathcal { N } _ { \mathsf { K } } ^ { \mathsf { M } } ( q ) } \delta _ { z } \ \in \mathcal { P } ( \mathsf { M } ) } \end{array}$ (i.e., the space ofprobability measures over M). $I f c : { \mathsf { M } } \to \{ 1 , \ldots , C \}$ is a class-label map, the corresponding semantic distribution is defined by the pushforward $\begin{array} { r } { \psi _ { K } ^ { \mathsf { M } } ( q ) = c _ { \# } \mu _ { K } ^ { \mathsf { M } } ( q ) = \frac { 1 } { K } \sum _ { z \in { \mathcal { N } } _ { \kappa } ^ { \mathsf { M } } ( q ) } \delta _ { c ( z ) } \in { \mathcal { P } } ( \{ 1 , \ldots , C \} ) } \end{array}$ .

The GW distance compares probability distributions that are supported on possibly distinct metric spaces via alignment.

Definition 6 (Gromov–Wasserstein Distance). Let $( X , d _ { X } , \mu )$ and $( Y , d _ { Y } , \nu )$ be metric measure spaces, and define $L ^ { 2 } ( x , x ^ { \prime } , y , y ^ { \prime } ) : = | d _ { X } ( x , \dot { x ^ { \prime } } ) - \dot { d _ { Y } } ( y , \dot { y ^ { \prime } } ) | ^ { 2 }$ . The squared Gromov–Wasserstein distance is

$$
\mathrm { G W } ^ { 2 } ( \mu , \nu ) : = \operatorname * { i n f } _ { \gamma \in \Pi ( \mu , \nu ) } \int L ^ { 2 } ( x , x ^ { \prime } , y , y ^ { \prime } ) d ( \gamma \otimes \gamma ) .
$$

Here, $\Pi ( \mu , \nu )$ denotes the set of couplings between $\mu$ and $\nu .$ The integral is taken over $( X \times Y ) ^ { 2 }$ with respect to $\gamma \otimes \gamma$

As exact computation of $\mathrm { G W ^ { 2 } }$ involves a quadratic assignment problem (QAP), which is NPhard Loiola et al. (2007), several tractable relaxations have been proposed. We focus on the entropic Gromov–Wasserstein distance Peyré et al. (2016), defined as

$$
\mathrm { G W } _ { \lambda } ^ { 2 } ( \mu , \nu ) : = \operatorname* { i n f } _ { \gamma \in \Pi ( \mu , \nu ) } \Big [ \int L ^ { 2 } ( x , x ^ { \prime } , y , y ^ { \prime } ) d ( \gamma \otimes \gamma ) + \lambda \mathrm { K L } ( \gamma \parallel \mu \otimes \nu ) \Big ] .
$$

Choice of scales (lo, gl). We use lo and $\mathtt { g l }$ as labels for two neighborhood scales. The lo scale corresponds to a local k-NN neighborhood of the query in $Z$ and ${ \check { P } } .$ . The $\mathtt { g l }$ scale corresponds to a global neighborhood induced by a fixed set of k-means centroids, computed offline on clean training data, which define the support for cross-space GW coupling. This local/global decomposition follows standard practice in GW-based analysis.

Algorithm. Our detector takes an image x, extracts its CNN embedding $z \ = \ h _ { \theta } ( x ) \in { \cal Z }$ and prime-quantized version $p \ = \ T _ { k } ^ { ( b ) } ( x ) \ \in \ P$ , and compares neighborhoods at two scales $( s \in$ $\{ \log _ { \mathfrak { g l } } \} )$ . For each scale, we compute (i) spatial distributions $\mu _ { s } ^ { Z } , \mu _ { s } ^ { P } , \mathrm { ( i i ) }$ semantic distributions $\psi _ { s } ^ { Z } , \psi _ { s } ^ { P }$ , and derive $g _ { 1 } = \mathrm { G W } _ { \lambda } ^ { 2 } ( \mu _ { s } ^ { \bar { Z } } , \mu _ { s } ^ { P } ) , g _ { 2 } = \mathrm { G W } _ { \lambda } ^ { 2 } ( \psi _ { s } ^ { Z } , \psi _ { s } ^ { P } )$ , and a semantic mismatch score $h =$ ENTROPY $\smash { ( \psi _ { s } ^ { Z } , \psi _ { s } ^ { P } ) }$ , defined as the Jensen–Shannon divergence $\mathrm { E N T R O P Y } ( \alpha , \beta ) : = \mathrm { J S } ( \alpha , \beta ) =$ $\begin{array} { r } { \frac 1 2 \mathrm { K L } ( \alpha \| m ) + \frac { 1 } { 2 } \mathrm { K L } ( \beta \| m ) } \end{array}$ with $m = { \textstyle { \frac { 1 } { 2 } } } ( \alpha + \beta )$ The resulting six-dimensional feature vector $\bar { \mathbf { f } } ( x ) = [ g _ { 1 , \mathrm { l o } } , \bar { g _ { 2 , \mathrm { l o } } } , h _ { \mathrm { l o } } , g _ { 1 , \mathrm { g l } } , g _ { 2 , \mathrm { g l } } , h _ { \mathrm { g l } } ]$ encodes cross-space discrepancies, which are classified by an SVM. Full pseudocode is provided in Algorithm 1 in Appendix J.1.

## 5 Geometric Foundations and Theoretical Guarantees

We begin by analyzing the stability of the prime quantization map $T _ { k } ^ { ( b ) }$ , which is central to our cross-space detector. The key question is, "when does a perturbation vanish into quantization noise, and when does it inevitably alter the output?" Our results formalize two complementary phenomena: pixel-level local stability and image-level injectivity. Prime quantization related proofs are deferred to Appendix B. All our formal results are stated for local neighborhoods. While the framework naturally extends to global neighborhoods, we leave the full theoretical treatment of that case to future work.

## 5.1 Pixel-Level Absorption

The absorption radius (Definition 7) captures the largest perturbation at a pixel that leaves its quantized value unchanged.

Definition 7 (Absorption radius). For $x _ { j } \in [ 0 , 1 ]$ with $S _ { k } ( x _ { j } ) = n \in ( p _ { \ell } , p _ { \ell + 1 } )$ , the absorption radius is $\begin{array} { r } { r _ { \mathrm { a b s } } ( x _ { j } , k ) = \frac { \operatorname* { m i n } \left\{ n - p _ { \ell } , p _ { \ell + 1 } - n \right\} } { 1 0 ^ { k } } } \end{array}$

Proposition 1 (Absorption guarantee). $I f | \eta _ { j } | \leq r _ { \mathrm { a b s } } ( x _ { j } , k )$ , then $T _ { k } ^ { ( b _ { j } ) } ( x _ { j } + \eta _ { j } ) = T _ { k } ^ { ( b _ { j } ) } ( x _ { j } ) f o r$ all bits $b _ { j }$

Lemma 1 (Absorption bounds). For any $\begin{array} { r } { x _ { j } \in [ 0 , 1 ] , \frac { 1 } { 1 0 ^ { k } } \leq r _ { \mathrm { a b s } } ( x _ { j } , k ) \leq \frac { 1 } { 2 } } \end{array}$

Remarks. These results formalize pixel-level stability: perturbations smaller than $r _ { \mathrm { a b s } }$ vanish under prime quantization, while larger ones necessarily cause a quantization change. Although Lemma 1 permits $r _ { \mathrm { a b s } } { \leq } 1 / 2$ , practical values are tiny $( \mathrm { e . g . , } \breve { \leq } 1 . 8 \times 1 \bar { 0 } ^ { - 3 } \mathrm { f o r } k = 4 )$ . Since adversarial budget in vision $( \epsilon \ge 1 / 2 5 \bar { 5 } \approx 3 . 9 \times 1 0 ^ { - 3 } )$ typically exceed these radii, most attacks cross prime boundaries and induce detectable discrepancies between Z and P.

![](images/16791127cafb418ba5fead355db4dfb93469183502ca7037b79aa12c84382a5b.jpg)  
Figure 2: Geometric inconsistency detection and gap theorem validation. 1st panel: clean neighborhoods; 2nd panel: adversarial separation $( \gamma c = 3 . 5 6 7 ) ;$ 3rd panel: gap theorem across $\varepsilon \in$ {2/255, 4/255, 6/255, 8/255}; 4th panel: risk control thresholding (left:clean; right:adversarial).

Even when perturbations lie near or below $r _ { \mathrm { a b s } } ,$ clean and adversarial samples seldom quantize identically. Because prime gaps are irregular, even a $1 / 2 5 5$ change can cross a prime-interval midpoint under the same bit-vector $b ,$ yielding different prime assignments. And when some coordinates do round identically, the $Z .$ -space embedding remains sensitive while $P$ stays piecewise constant, producing a measurable $Z { - } P$ mismatch. Larger prime resolution k further shrinks $r _ { \mathrm { a b s } }$ and increases the likelihood of such discrepancies.

## 5.2 Image-Level Injectivity

While prime quantization is many-to-one per pixel, we must ensure it does not collapse distinct images globally. Lemma 2 shows that such collisions are exponentially unlikely.

Lemma 2 (Collision probability). $F i x k \ge 2$ and let $N : = 1 0 ^ { k } . L e t x , x ^ { \prime } \in [ 0 , 1 ] ^ { d }$ be two independent random images with i.i.d. pixel marginals whose densities are bounded by Λ on [0, 1] (in particular, $\Lambda = 1 f o r$ the uniform distribution). For a fixed secret bit vector $b \in \{ 0 , 1 \} ^ { d }$ , let $T _ { k } ^ { ( b ) }$ be the prime–quantization transform (Definition 4). Assume the prime–gap envelope $G _ { k }$ from Assumption 1. Then, $\begin{array} { r } { \mathrm { P r } \big [ T _ { k } ^ { ( b ) } ( x ) = T _ { k } ^ { ( b ) } ( x ^ { \prime } ) \big ] \ \leq \ \left( \Lambda ^ { 2 } \frac { G _ { k } } { N } \right) ^ { d } } \end{array}$ . In particular, for $\Lambda = 1$ , the collision probability decays as $\left( G _ { k } / 1 0 ^ { k } \right) ^ { d }$ in the number ofpixels.

Remarks. Lemma 2 establishes that global collisions are vanishingly rare. Even with pixel-level absorption, distinct images remain separable: $\mathrm { e . g . }$ , for $\mathrm { C I F A R - } 1 0 ( d = \bar { 3 } 0 7 2 )$ and $k = 4 , ( G _ { k } / 1 0 ^ { k } ) ^ { d }$ is effectively zero. Thus, quantization is locally many-to-one but globally almost injective, ensuring discriminability while dampening small perturbations.

## 5.3 Bridge to GW Separation

Pixel-level absorption (Def. 7, Prop. 1) and image-level injectivity (Lemma 2) set the boundary conditions: $\mathrm { i f } \epsilon \le r _ { \mathrm { a b s } }$ , perturbations vanish in $P$ while Z still moves; if $\epsilon > r _ { \mathrm { a b s } } .$ , quantization shifts and $P$ changes, so in both regimes $Z$ and $P$ neighborhoods diverge. To formalize these divergences, we embed samples into spatial–semantic product spaces $\mathcal { C } = ( \breve { Z } \times Y , d _ { \mathcal { C } } )$ and $\mathcal { K } = ( P \times \mathbf { \breve { Y } } , d _ { \mathcal { K } } )$ representing each image as $( h _ { \theta } ( x ) , y )$ and $( T _ { k } ^ { ( b ) } ( x ) , y )$ . By Theorem 5 and Corollary 2, the $\ell _ { \infty }$ product is the tightest among admissible component metrics, so any discrepancy in geometry or label mass yields separation in $\mathcal { C }$ and K. This construction underlies the GW envelope and gap theorems that follow.

## 5.4 Diameter Envelopes in C and K

We now summarize the behavior of k-nearest neighbor diameters in the two product spaces C (CNNbased) and K (prime-quantized). For clean samples, diameters concentrate tightly around a median distance; for adversarially perturbed samples, explicit additive expansion terms appear. The full technical statements and proofs are deferred to Appendix D.

Lemma 3 (Unified local diameter envelopes in C and K). For any confidences $\delta _ { \mathrm { c l e a n } } , \delta _ { \mathrm { e n v } } , \delta _ { \mathrm { g r a d } } \in$ (0, 1) and local neighborhood size $K _ { \mathrm { l o } } \ \geq \ 2 ,$ , and under the variance proxy (Assumption 2) and prime-gap sensitivity (Assumption 3) conditions (see Appendix D), define the clean envelopes as $\begin{array} { r } { U _ { \mathrm { c l e a n } } ^ { \mathcal { C } } : = 2 \mu _ { \mathcal { C } } + 2 \left( \sqrt { \frac { 2 \log K _ { l o } } { d } } + \sqrt { \frac { 2 \log ( 2 / \delta ) } { d } } \right) } \end{array}$ and $U _ { \mathrm { c l e a n } } ^ { \mathcal { K } } : = 2 A \mu _ { K } ^ { \operatorname* { m a x } } ( q ) + C _ { k } \sqrt { d } \Bigl ( \sqrt { 2 \log K _ { l o } } +$ $\sqrt { 2 \log ( 2 / \delta _ { \mathrm { e n v } } ) } )$ . Additionally, we define the Jacobian drift as $\begin{array} { r } { J : = \frac { 2 \sqrt { d } \sigma } { \sqrt { \delta } } \| \eta \| _ { \infty } } \end{array}$ . Then, for clean queries q and adversarial queries $\tilde { q } = q + \eta ,$ , thefollowing bounds hold:

$$
\mathrm { d i a m } \left( \mathcal { N } _ { K _ { \mathrm { l o } } } ^ { \mathcal { C } } ( q ) \right) \leq U _ { \mathrm { c l e a n } } ^ { \mathcal { C } } ,\tag{1}
$$

$$
\mathrm { d i a m } \left( \mathcal N _ { K _ { \mathrm { l o } } } ^ { \mathcal { C } } ( \tilde { q } ) \right) \le U _ { \mathrm { c l e a n } } ^ { \mathcal { C } } + J + 2 \mathbf { 1 } _ { \{ y \neq \hat { y } \} } ,\tag{2}
$$

$$
\mathrm { d i a m } \left( \mathcal { N } _ { K _ { \mathrm { l o } } } ^ { \mathcal { K } } ( q ) \right) \leq U _ { \mathrm { c l e a n } } ^ { \mathcal { K } } ,\tag{3}
$$

$$
\mathrm { d i a m } \left( \mathcal N _ { K _ { \mathrm { l o } } } ^ { \mathcal K } ( \tilde { q } ) \right) \le U _ { \mathrm { c l e a n } } ^ { \mathcal K } + J + 2 { \bf 1 } _ { \{ y \ne \hat { y } \} } .\tag{4}
$$

Here $\mu { c }$ is the clean median pairwise distance in ${ \mathcal { C } } , \mu _ { \mathcal { K } } ^ { \operatorname* { m a x } } ( q )$ is the maximum key-annealed median distance in K, σ is the variance proxyfrom Assumption 2, and $C _ { k } = 2 G _ { k }$ is the prime-gap sensitivity constant from Assumption 3.

Geometric Insight. In C, clean neighborhood diameters concentrate around a median distance (App. Lemma 6), while adversarial perturbations add a Jacobian-driven drift and a possible label-flip penalty. In K, clean neighborhoods are stabilized by prime-gap sensitivity (App. Thm. 8), whereas adversarial perturbations add terms from key sensitivity, perturbation norms, and label flips. Together these define the diameter envelopes that underlie the cross-space GW theorems. Fig. 2 (1st–2nd panels) empirically confirms this: clean samples form compact clusters consistent with App. Thms. 6, 8, while adversarial queries induce the predicted separation gap $\gamma c = 3 . 5 6 7 ( \mathrm { T h m } . 2 )$

## 5.5 GW Bounds: Clean vs. Adversarial

We now move from local geometry (K-NN diameter bounds in C and K, Lemma 3) to a distributional geometry comparison across spaces. The Gromov–Wasserstein (GW) distance aligns pairwise distance structures, allowing us to bound: (i) in the clean case, similarity of C and K, and (ii) in the adversarial case, a provable increase when perturbations inflate diameters differently across spaces. This separation underlies our detection framework.

Theorem 1 (Clean cross–space GW upper bound via K–NN radii). Fix a clean query x and consider its local neighborhoods $\mathcal { \hat { N } } _ { K } ^ { \mathcal { C } } ( x ) \subset \mathcal { C } \mathrm { ~ } \overset { \cdot \cdot } { a n d } \mathcal { N } _ { K } ^ { \mathcal { K } } ( x ) \subset K$ , each endowed with the uniform probability measure on K points.

Let $R _ { { \mathcal { C } } } : = \operatorname* { m a x } _ { 1 \leq i \leq K } d _ { { \mathcal { C } } } ( x _ { i } , q _ { \mathcal { C } } )$ amd $R _ { { \sf K } } : = \operatorname* { m a x } _ { 1 \leq i \leq K } d _ { K } ( y _ { i } , q _ { K } )$ ) denote the clean K–NN radii in C and $\kappa ,$ , respectively. Assume the clean radius envelopes $R c \leq U _ { \star } ^ { \mathcal { C } }$ and $R \kappa \leq U _ { \star } ^ { \kappa }$ hold with probabilities at least $1 - \delta _ { C }$ and $1 - \delta _ { K }$ , as established in Appendix D.

Then, with probability at least $1 - ( \delta c + \delta \kappa )$ , the quadratic Gromov–Wasserstein discrepancy between the clean neighborhoods satisfies

$$
\begin{array} { r } { \mathrm { G W } ^ { 2 } \Big ( \mathcal { N } _ { K } ^ { \mathcal { C } } ( x ) , \mathcal { N } _ { K } ^ { \mathcal { K } } ( x ) \Big ) \ \leq \ 4 \Big ( 1 - \frac { 1 } { K } \Big ) \left( U _ { \star } ^ { \mathcal { C } } + U _ { \star } ^ { K } \right) ^ { 2 } . } \end{array}
$$

Notation (adversarial queries, radii, and gap). For a clean input x and perturbation $\eta ,$ the adversarial query in $\mathsf { M } \in \{ \mathcal { C } , \mathcal { K } \}$ is $\widetilde { q } _ { \mathsf { M } } = z ^ { \mathsf { M } } ( x + \eta )$ . Its K nearest neighbors form $\widetilde { \mathcal { N } } _ { K } ^ { \mathsf { M } } ( x + \eta ) =$ $\{ z _ { 1 } ^ { \mathsf { M } } , \ldots , z _ { K } ^ { \mathsf { M } } \}$ with radii $r _ { i } ^ { \mathsf { M } } = d _ { \mathsf { M } } ( z _ { i } ^ { \mathsf { M } } , \widetilde { q } _ { \mathsf { M } } )$ and maximum $R _ { \mathsf { M } } ^ { \mathrm { a d v } } = \operatorname* { m a x } _ { i } r _ { i } ^ { \mathsf { M } }$ . We partition the K neighbors into an inner set L of size $( 1 - \theta ) K$ and outer set H of size θK, and define the separation gap as $\begin{array} { r } { \gamma _ { \mathsf { M } } = \operatorname* { m i n } _ { i \in H } r _ { i } ^ { \mathsf { M } } - \operatorname* { m a x } _ { j \in L } r _ { j } ^ { \mathsf { M } } } \end{array}$

Theorem 2 (Adversarial cross–space GW lower bound). Fix a query x and perturbation $\eta ,$ and consider the adversarial neighborhoods $\widetilde { \mathcal { N } } _ { K } ^ { \mathcal { C } } ( x + \eta )$ and $\widetilde { \mathcal { N } } _ { K } ^ { \mathcal { K } } ( x + \eta )$ , each with uniform measure on K points. Let $\gamma _ { \mathcal { C } }$ be the separation gap and let diam $\left( \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { K } } ( x + \eta ) \right)$ denote the adversarial K–NN diameter in K, bounded as in Appendix Theorem 9. Define the adversarial K–diameter $D _ { \mathcal { K } } ^ { \mathrm { a d v } } : = \mathrm { d i a m } \big ( \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { K } } ( x + \eta ) \big )$ . Then, with probability at least $1 - \delta _ { K } ^ { \mathrm { e n v } }$

$$
\mathrm { G W } ^ { 2 } \Big ( \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { C } } ( x + \eta ) , \widetilde { \mathcal { N } } _ { K } ^ { K } ( x + \eta ) \Big ) \ \geq \ 2 \theta ( 1 - \theta ) \left( \gamma _ { \mathcal { C } } - D _ { K } ^ { \mathrm { a d v } } \right) _ { + } ^ { 2 } .
$$

Full proof is provided in Appendix F.

Mirror results. For brevity, we omit the symmetric (i) clean lower bounds and (ii) adversarial upper bounds on GW, but detailed proofs are provided in Theorem 10 and Theorem 11 in Appendix F.

## 5.6 GW Gap and Risk Control

Theorem 3 (Cross–space GW gap). With probability at least $1 - ( \delta c + \delta \kappa + \delta _ { K } ^ { \mathrm { e n v } } + \delta _ { \mathrm { a u x } } )$ , the clean and adversarial GW discrepancies satisfy $\mathrm { | G W _ { a d v } ^ { 2 } - G W _ { c l e a n } ^ { 2 } | \geq \tau : = }$ max $\{ \tau _ { \mathrm { a d v } } , \tau _ { \mathrm { c l e a n } } , 0 \}$ where $\tau _ { \mathrm { a d v } } = L _ { \mathrm { a d v } } - U _ { \mathrm { c l e a n } }$ and $\tau _ { \mathrm { c l e a n } } = L _ { \mathrm { c l e a n } } - U _ { \mathrm { a d v } } . ~ U n d e r$ Assumption 2, for fixed K and perturbation $\| \eta \| _ { \infty } = \varepsilon ,$ , we obtain $\tau = \Omega ( d \sigma ^ { 2 } \varepsilon ^ { 2 } ) \ - \ O \big ( ( R _ { K } ^ { \mathrm { c l e a n } } ) ^ { 2 } \big )$

Lemma 4 (Risk control via GW margin). If the gap event holds with margin $\tau > 0$ and an estimator $\widehat { \mathrm { G W } _ { \lambda } ^ { 2 } }$ satisfies Pr $\left( | \widehat { \mathrm { G W } _ { \lambda } ^ { 2 } } - \mathrm { G W } ^ { 2 } | \leq \tau / 3 \right) \geq 1 - \delta _ { \mathrm { e s t } }$ , then thresholding $\widehat { \mathrm { G W } _ { \lambda } ^ { 2 } }$ at the midpoint between clean and adversarial envelopes makes no error on this event. Thus Pr(misclassification) ≤ $\mathrm { P r } ( E _ { \mathrm { g a p } } ^ { c } ) + \delta _ { \mathrm { e s t } }$

From GW to entropic GW. All bounds above were stated for quadratic $\mathrm { G W ^ { 2 } }$ . For entropic GW $\mathrm { G W _ { \lambda } ^ { 2 } }$ with $\lambda > 0$ , the lower bounds remain unchanged, while the upper bounds incur only an additive 2λ log K (Corollaries 7–8). Hence, the clean/adversarial separation guarantees extend seamlessly to the entropic case used in practice. Theorem 3 certifies a provable margin: the clean GW discrepancy is uniformly controlled by the clean envelopes, while adversarial perturbations induce a lower bound whose leading term scales as $\Omega ( d \sigma ^ { 2 } \varepsilon ^ { 2 } )$ (up to constants). Lemma 4 translates this into a statistical guarantee: once $\widehat { \mathrm { G W _ { \lambda } ^ { 2 } } }$ concentrates within $\tau / 3 .$ , thresholding achieves negligible error. Empirical evidence (Fig. 2, panel 4) confirms the theory: clean and adversarial discrepancy distributions separate cleanly, validating the predicted risk bound. Proofs and full derivations are in App. F.

## 6 Empirical Analysis

## 6.1 Experimental Setup

Adversarial Attacks. We evaluate a broad suite spanning gradient-based, optimization, spatial, and perceptual perturbations: Auto-Attack (AA) (Croce & Hein, 2020), Carlini–Wagner (CW) (Carlini & Wagner, 2017), Patch (PT) (Brown et al., 2017), Projected Gradient Descent (PGD) (Madry et al., 2018), Spatial (SA) (Engstrom et al., 2019), Square (SQ) (Andriushchenko et al., 2020), Universal Perturbations (UP) (Moosavi-Dezfooli et al., 2017), Auto-PGD (AP) (Croce & Hein, 2020), Fast Gradient Sign (FG) (Goodfellow et al., 2014), Frequency (FA) (Yin et al., 2019), Gaussian Blur (GB) (Zhang et al., 2022), Pixel Flip (PF) (Su et al., 2019), Semantic Rotation (SR) (Hosseini & Poovendran, 2018), AdvAD (AAD) (Li et al., 2024), Penalizing Gradient Norm (PGN) (Ge et al., 2023), and Block Shuffle and Rotation (BSR) (Wang et al., 2024). We use these boldface abbreviations throughout tables and figures for brevity. Refer to Table 5 in Appendix G.5 for attack hyperparameter settings and defaults.

Baseline Defenses. We benchmark against representative detection methods: Mahalanobis Detector (MD) (Lee et al., 2018), Feature Squeezing (FS) (Xu et al., 2018), Meta-Adversarial-Detect (MAD) (Ma et al., 2019), MagNet (MN) (Meng & Chen, 2017), Multiple Perturbation Detector (EA) (Zhang et al., 2023), and Be Your Own Neighborhood (BY) (He et al., 2024).

Evaluation Metrics. We report the following complementary metrics. (1) Binary detection accuracy: overall accuracy of classifying inputs as clean or adversarial. (2) True Positive Rate (TPR): fraction of adversarial samples correctly flagged as adversarial, i.e., $\mathrm { T P R } \ \stackrel { \cdot } { = } \ \frac { \mathrm { d e t e c t e d a d v e r s a r i a l } } { \mathrm { a l l a d v e r s a r i a l } }$ . (3)

<table><tr><td>Attack</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td></tr><tr><td>AA</td><td>97.9</td><td>68.9</td><td>82.7</td><td>52.0</td><td>74.1</td></tr><tr><td>CW</td><td>97.0</td><td>73.6</td><td>86.0</td><td>51.4</td><td>56.7</td></tr><tr><td>PT</td><td>98.0</td><td>86.4</td><td>67.8</td><td>50.7</td><td>57.3</td></tr><tr><td>PGD</td><td>97.8</td><td>91.3</td><td>74.4</td><td>51.1</td><td>81.0</td></tr><tr><td>SA</td><td>96.8</td><td>78.1</td><td>74.4</td><td>41.1</td><td>54.9</td></tr><tr><td>SQ</td><td>97.6</td><td>89.2</td><td>88.5</td><td>51.0</td><td>44.4</td></tr><tr><td>UP</td><td>97.8</td><td>66.4</td><td>53.7</td><td>50.7</td><td>47.9</td></tr><tr><td>AP</td><td>97.6</td><td>68.3</td><td>81.4</td><td>50.2</td><td>73.6</td></tr><tr><td>FG</td><td>98.0</td><td>73.8</td><td>60.9</td><td>49.6</td><td>44.7</td></tr><tr><td>FA</td><td>95.1</td><td>49.9</td><td>50.0</td><td>49.7</td><td>49.8</td></tr><tr><td>GB</td><td>85.7</td><td>49.8</td><td>51.7</td><td>48.2</td><td>48.6</td></tr><tr><td>PF</td><td>97.0</td><td>51.7</td><td>51.4</td><td>49.5</td><td>49.4</td></tr><tr><td>SR</td><td>95.9</td><td>50.3</td><td>52.7</td><td>50.1</td><td>49.1</td></tr></table>

Table 1: Binary detection accuracy (%). Best results are in bold and second best are underlined.

<table><tr><td>Attack</td><td>Model</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td><td>EA</td><td>BY</td></tr><tr><td>PGD</td><td>ResNet-18 ViT</td><td>0.97 0.95</td><td>0.91 0.65</td><td>0.71 0.75</td><td>0.25 0.57</td><td>0.78 0.01</td><td>0.96 0.95</td><td>0.70 0.80</td></tr><tr><td>SQ</td><td>ResNet-18 ViT</td><td>0.96 0.95</td><td>0.89 0.66</td><td>0.85 0.86</td><td>0.24 0.56</td><td>0.01 0.02</td><td>0.90 0.89</td><td>0.57 0.61</td></tr><tr><td>PT</td><td>ResNet-18 ViT</td><td>0.98 0.95</td><td>0.86 0.66</td><td>0.54 0.67</td><td>0.26 0.54</td><td>0.03 0.01</td><td>0.89 0.89</td><td>0.80 0.81</td></tr><tr><td>AAD</td><td>ResNet-18 ViT</td><td>0.95 0.93</td><td>0.31 0.67</td><td>0.46 0.57</td><td>0.52 0.64</td><td>0.13 0.01</td><td>0.91 0.92</td><td>0.59 0.82</td></tr><tr><td>PGN</td><td>ResNet-18 ViT</td><td>0.95 0.96</td><td>0.63 0.67</td><td>0.62 0.62</td><td>0.24 0.64</td><td>0.01 0.02</td><td>0.90 0.93</td><td>0.77 0.77</td></tr><tr><td>BSR</td><td>ResNet-18 ViT</td><td>0.95 0.98</td><td>0.67 0.67</td><td>0.52 0.56</td><td>0.23 0.64</td><td>0.40 0.02</td><td>0.92 0.92</td><td>0.81 0.78</td></tr></table>

Table 2: F1-score comparison on CIFAR-10 across multiple attacks using ResNet-18 and ViT. Best results are in bold, second best are underlined.

End-to-End accuracy: proportion of clean samples correctly classified and passed by the detector, plus adversarial samples correctly blocked; this reflects system-level robustness under attack. (4) Precision: fraction of samples flagged as adversarial that are truly adversarial. (5) Recall: identical to TPR—the fraction of adversarial samples correctly detected. (6) F1-score: harmonic mean of preci sion and recall, summarizing detection quality under imbalance. (7) AUROC: area under the ROC curve, measuring threshold-independent separability between clean and adversarial distributions.

## 6.2 Adversarial Detection Accuracy

Setup. We evaluate detection on the datasets CIFAR-10 Krizhevsky et al. (2009), FMNIST Xiao et al. (2017), KMNIST Clanuwat et al. (2018), and ImageNet Deng et al. (2009) using models ResNet18 He et al. (2016) and ViT Dosovitskiy et al. (2021), with adversarial datasets generated from the attack suite in Sec. 6.1. Detector/classifier hyperparameters and attack configurations appear in Appendix H and Appendix G.5, respectively. We note that ResNet18 on CIFAR-10 is employed as our default configuration.

## Results and analysis.

Table 1 shows that our detector achieves $\geq \mathbf { 9 5 \% }$ binary detection on 12 of 13 attacks, with Gaussian blur (85.7%) as the only exception. Accuracy remains consistently high across attack families: (i) Gradient/optimization (AA, CW, PGD, AP): 96–98%, with margins of +10–20 points, since small-norm shifts in $Z$ are often absorbed in $P ,$ , producing sharp cross-space mismatches; (ii) Spatial/patch (SA, PT): 97–98%, where local structural changes disrupt geometry differently in each space; (iii) Transfer/decision-based (SQ, UP): 97–98%, where transfer-induced distortions misalign $Z$ and $P$ far more than gradient-based attacks, yielding especially large gains (+31 points on UP); and (iv) Perceptual/frequency (FA, PF, SR): 95–97%, where frequency and semantic shifts perturb $P ^ { * } { \bf s }$ discrete neighborhoods and $Z \ ' _ { \mathbf { s } }$ embeddings in complementary ways, creating highly detectable discrepancies. Gaussian blur is the hardest case because it averages neighboring pixels, suppressing edges and textures, inducing similar distortions in both $Z$ and $P .$ This reduces the crossspace discrepancy that our detector exploits. Nevertheless, uneven quantization in $P$ ensures residual separation, and we still outperform all baselines on blur. Full per-dataset results and additional metrics, including TPR/FPR heatmaps and end-to-end-accuracy, are reported in Table 12(Appendix K) and Appendix L respectively. Across all six attacks and both architectures (ResNet-18 and ViT), our method consistently achieves the highest F1-scores, typically exceeding 0.95. EA generally emerges as the strongest baseline yet remains noticeably weaker than our detector, especially under patchbased and structure-altering attacks such as SQ and PT. Modern attacks such as AAD, PGN, and BSR also show the same trend: while EA or BY occasionally achieve strong second-best performance, our approach maintains a clear advantage across architectures. These results highlight the robustness and model-agnostic behavior of the proposed Z–P discrepancy framework.

<table><tr><td>Dataset</td><td> $C _ { \mathrm { c r o s s } }$ </td><td> $C _ { \mathrm { m s } }$ </td></tr><tr><td>CIFAR-10</td><td>86.7</td><td>84.5</td></tr><tr><td>FMNIST</td><td>89.6</td><td>87.9</td></tr><tr><td>KMNIST</td><td>88.2</td><td>86.8</td></tr></table>

(a) Adaptive attack detection (%).

<table><tr><td>Features</td><td>AA</td><td>CW</td><td>PT</td><td>PGD</td><td>FG</td></tr><tr><td>Local only</td><td>67.7</td><td>76.2</td><td>73.2</td><td>62.5</td><td>79.1</td></tr><tr><td>Global only</td><td>83.5</td><td>66.5</td><td>75.6</td><td>65.0</td><td>71.3</td></tr><tr><td>Both</td><td>97.9</td><td>97.0</td><td>98.0</td><td>97.8</td><td>98.0</td></tr></table>

(b) Ablation on CIFAR-10 (%).  
Table 3: Adaptive robustness and feature ablation. (a) Our method resists adaptive white-box attacks despite defense-aware optimization. (b) Combining local and global GW features yields the strongest detection across attacks.

## 6.3 Adaptive Attack Resistance and Ablation

Adaptive attacks. We test two white-box adaptive formulations: (i) cross-space $( C _ { \mathrm { c r o s s } } )$ and (ii) multi-scale $( C _ { \mathrm { m s } } )$ , where the adversary knows the architecture but not the secret bit vector b<sup>⋆</sup>. The complete formulation of the adversary’s objective, the prior distribution over unknown secret bits, the consistency penalties, and the optimization procedure are provided in Appendix I. As shown in Table 3a, our method maintains strong detection (84–90%) across CIFAR-10, FMNIST, and KMNIST, demonstrating robustness even when defenses are explicitly targeted. This accuracy drop relative to non-adaptive attacks arises because the adversary now explicitly optimizes to minimize cross-space discrepancies (CNN vs. crypto features). By enforcing feature consistency under a prior over b, they can partially reduce the mismatches our detector relies on.

Ablation study. To quantify feature contributions, we compare detectors using only local GW features, only global GW features, or both. Table 3b shows that while local or global features alone yield moderate performance (65–83%), combining them achieves 97–98% across all attacks. This confirms that local fine-grained cues and global structural signals are complementary.

## 7 Conclusion

We introduced a principled framework for adversarial detection based on geometric inconsistencies between the embedding space Z and a prime-quantized space P. Our theory shows that adversarial perturbations inevitably create detectable cross-space discrepancies, providing the first guarantees for when detection must succeed. Experiments confirm consistently high detection accuracy across diverse attacks, strong generalization to zero-shot VLMs, and robustness to adaptive adversaries. These results demonstrate that geometric reasoning offers a solid foundation for adversarial robustness. An immediate direction is to adapt our framework to multimodal models, where both adversarial pressure and generalization demands are higher. Extending our theoretical guarantess from local to global neighborhoods also forms an interesting direction for future work.

## 8 Limitations and Scope:

The framework is most effective when perturbations induce non-isometric distortions between Z and P; approximately low-frequency or near-isometric transformations, such as Gaussian blur, produce weaker geometric discrepancies and reduced detection margins. In addition, the use of Gromov– Wasserstein distances incurs higher computational cost relative to simpler detectors (Appendix K.8), favoring high-assurance over ultra-low-latency deployments. The theoretical guarantees rely on explicit assumptions, including the validity of a local Jacobian proxy (Assumption 2) and neighborhood regularity, and should be interpreted as local geometric rather than global manifold-level results.

## References

Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, and Matthias Hein. Square attack: a query-efficient black-box adversarial attack via random search. In European conference on computer vision, pp. 484–501. Springer, 2020.

Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In International conference on machine learning, pp. 274–283. PMLR, 2018.

Aharon Ben-Tal, Laurent El Ghaoui, and Arkadi Nemirovski. Robust Optimization. Princeton University Press, Princeton, NJ, 2009.

Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with random forests. In European conference on computer vision, pp. 446–461. Springer, 2014.

Tom B Brown, Dandelion Mané, Aurko Roy, Martín Abadi, and Justin Gilmer. Adversarial patch. arXiv preprint arXiv:1712.09665, 2017.

Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy (SP), pp. 39–57. IEEE Computer Society, 2017.

Pin-Yu Chen, Huan Zhang, Yash Sharma, Jinfeng Yi, and Cho-Jui Hsieh. Zoo: Zeroth order optimization based black-box attacks to deep neural networks without training substitute models. In Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security, AISec ’17, pp. 15–26. Association for Computing Machinery, 2017.

Tarin Clanuwat, Mikel Bober-Irizar, Asanobu Kitamoto, Alex Lamb, Kazuaki Yamamoto, and David Ha. Deep learning for classical japanese literature. arXiv preprint arXiv:1812.01718, 2018.

Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In international conference on machine learning, pp. 1310–1320. PMLR, 2019.

Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In International conference on machine learning, pp. 2206–2216. PMLR, 2020.

Xuanming Cui, Alejandro Aparcedo, Young Kyun Jang, and Ser-Nam Lim. On the robustness of large multimodal models against image adversarial attacks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24625–24634, 2024.

Herbert A David and Haikady N Nagaraja. Order statistics. John Wiley & Sons, 2004.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

Pierre Dusart. Estimates of some functions over primes without rh. arXiv preprint arXiv:1002.0442, 2010.

Logan Engstrom, Brandon Tran, Dimitris Tsipras, Ludwig Schmidt, and Aleksander Madry. Exploring the landscape of spatial robustness. In International conference on machine learning, pp. 1802– 1811. PMLR, 2019.

Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In 2004 conference on computer vision and pattern recognition workshop, pp. 178–178. IEEE, 2004.

Reuben Feinman, Ryan R Curtin, Saurabh Shintre, and Andrew B Gardner. Detecting adversarial samples from artifacts. arXiv preprint arXiv:1703.00410, 2017.

Zhijin Ge, Hongying Liu, Wang Xiaosen, Fanhua Shang, and Yuanyuan Liu. Boosting adversarial transferability by achieving flat local maxima. Advances in Neural Information Processing Systems, 36:70141–70161, 2023.

Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.

Sven Gowal, Sylvestre-Alvise Rebuffi, Olivia Wiles, Florian Stimberg, Dan Andrei Calian, and Timothy A Mann. Improving robustness using generated data. Advances in neural information processing systems, 34:4218–4233, 2021.

Gregory Griffin, Alex Holub, Pietro Perona, et al. Caltech-256 object category dataset. Technical report, Technical Report 7694, California Institute of Technology Pasadena, 2007.

Godfrey Harold Hardy and Edward Maitland Wright. An introduction to the theory of numbers. Oxford university press, 1979.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Zhiyuan He, Yijun Yang, Pin-Yu Chen, Qiang Xu, and Tsung-Yi Ho. Be your own neighborhood: Detecting adversarial examples by the neighborhood relations built on self-supervised learning. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 18063–18080. PMLR, 21–27 Jul 2024.

Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261, 2019a.

Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In International Conference on Learning Representations, 2019b.

Hossein Hosseini and Radha Poovendran. Semantic adversarial examples. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 1614–1619, 2018.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings ofthe 32nd International Conference on Machine Learning, volume 37 of Proceedings ofMachine Learning Research, pp. 448–456. PMLR, 07–09 Jul 2015.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

Anders Krogh and John Hertz. A simple weight decay can improve generalization. In J. Moody, S. Hanson, and R.P. Lippmann (eds.), Advances in Neural Information Processing Systems, volume 4. Morgan-Kaufmann, 1991.

Hyeyoon Lee, Kanghyun Choi, Dain Kwon, SunJong Park, Mayoore Selvarasa Jaiswal, Noseong Park, Jonghyun Choi, and Jinho Lee. Datafreeshield: Defending adversarial attacks without training data. In Forty-first International Conference on Machine Learning, 2024.

Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. Advances in neural information processing systems, 31, 2018.

Jin Li, Ziqiang He, Anwei Luo, Jian-Fang Hu, Z Jane Wang, and Xiangui Kang. Advad: Exploring non-parametric diffusion for imperceptible adversarial attacks. Advances in Neural Information Processing Systems, 37:52323–52353, 2024.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, volume 36, pp. 34892–34916. Curran Associates, Inc., 2023.

Eliane Maria Loiola, Nair Maria Maia de Abreu, Paulo Oswaldo Boaventura-Netto, Peter Hahn, and Tania Querido. A survey for the quadratic assignment problem. European Journal ofOperational Research, 176(2):657–690, 2007. ISSN 0377-2217. doi: https://doi.org/10.1016/j.ejor.2005.09.032. URL https://www.sciencedirect.com/science/article/pii/S0377221705008337.

Chen Ma, Chenxu Zhao, Hailin Shi, Li Chen, Junhai Yong, and Dan Zeng. Metaadvdet: Towards robust detection of evolving adversarial attacks. pp. 692–701. Association for Computing Machinery, 2019.

Xingjun Ma, Bo Li, Yisen Wang, Sarah M. Erfani, Sudanthi Wijewickrema, Grant Schoenebeck, Michael E. Houle, Dawn Song, and James Bailey. Characterizing adversarial subspaces using local intrinsic dimensionality. In International Conference on Learning Representations, 2018.

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018.

Dongyu Meng and Hao Chen. Magnet: a two-pronged defense against adversarial examples. In Proceedings of the 2017 ACM SIGSAC conference on computer and communications security, pp. 135–147, 2017.

Jan Hendrik Metzen, Tim Genewein, Volker Fischer, and Bastian Bischoff. On detecting adversarial perturbations. In International Conference on Learning Representations, 2017.

Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, and Pascal Frossard. Universal adversarial perturbations. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1765–1773, 2017.

Gabriel Peyré, Marco Cuturi, and Justin Solomon. Gromov-wasserstein averaging of kernel and distance matrices. In International conference on machine learning, pp. 2664–2672. PMLR, 2016.

Ben Poole, Subhaneil Lahiri, Maithra Raghu, Jascha Sohl-Dickstein, and Surya Ganguli. Exponential expressivity in deep neural networks through transient chaos. Advances in neural information processing systems, 29, 2016.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Aditi Raghunathan, Jacob Steinhardt, and Percy Liang. Certified defenses against adversarial examples. In International Conference on Learning Representations, 2018.

Reuven Y Rubinstein and Dirk P Kroese. Simulation and the Monte Carlo method. John Wiley & Sons, 2016.

Hadi Salman, Jerry Li, Ilya Razenshteyn, Pengchuan Zhang, Huan Zhang, Sebastien Bubeck, and Greg Yang. Provably robust deep learning via adversarially trained smoothed classifiers. In Advances in Neural Information Processing Systems, volume 32, 2019.

Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry. How does batch normalization help optimization? Advances in neural information processing systems, 31, 2018.

Samuel S. Schoenholz, Justin Gilmer, Surya Ganguli, and Jascha Sohl-Dickstein. Deep information propagation. In International Conference on Learning Representations, 2017.

Yang Song, Taesup Kim, Sebastian Nowozin, Stefano Ermon, and Nate Kushman. Pixeldefend: Leveraging generative models to understand and defend against adversarial examples. In International Conference on Learning Representations, 2018.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(56):1929–1958, 2014.

Jiawei Su, Danilo Vasconcellos Vargas, and Kouichi Sakurai. One pixel attack for fooling deep neural networks. IEEE Transactions on Evolutionary Computation, 23(5):828–841, 2019.

Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2018.

Stefan Wager, Sida Wang, and Percy S Liang. Dropout training as adaptive regularization. In Advances in Neural Information Processing Systems, volume 26, 2013.

Kunyu Wang, Xuanran He, Wenxuan Wang, and Xiaosen Wang. Boosting adversarial transferability by block shuffle and rotation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24336–24346, 2024.

Chong Xiang, Saeed Mahloujifar, and Prateek Mittal. Patchcleanser: Certifiably robust defense against adversarial patches for any image classifier. In 31st USENIX security symposium (USENIX Security 22), pp. 2065–2082, 2022.

Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.

Weilin Xu, David Evans, and Yanjun Qi. Feature squeezing: Detecting adversarial examples in deep neural networks. In 25th Annual Network and Distributed System Security Symposium, NDSS 2018, San Diego, California, USA, February 18-21, 2018, 2018.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024. URL https://arxiv.org/abs/2407.10671.

Dong Yin, Raphael Gontijo Lopes, Jon Shlens, Ekin Dogus Cubuk, and Justin Gilmer. A fourier perspective on model robustness in computer vision. Advances in Neural Information Processing Systems, 32, 2019.

Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric Xing, Laurent El Ghaoui, and Michael Jordan. Theoretically principled trade-off between robustness and accuracy. In International conference on machine learning, pp. 7472–7482. PMLR, 2019.

Shuhai Zhang, Feng Liu, Jiahao Yang, Yifan Yang, Changsheng Li, Bo Han, and Mingkui Tan. Detecting adversarial data by probing multiple perturbations using expected perturbation score. In International conference on machine learning, pp. 41429–41451. PMLR, 2023.

Yaoyuan Zhang, Yu-an Tan, Mingfeng Lu, Tian Chen, Yuanzhang Li, and Quanxin Zhang. Boosting cross-task adversarial attack with random blur. International journal ofintelligent systems, 37(10): 8139–8154, 2022.

## A Notation

For ease of reference, we summarize in Table 4 the main symbols and spaces used throughout the paper. Unless otherwise noted, all notation is consistent across sections.

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td>Data &amp; embeddings</td><td></td></tr><tr><td> $X = \{ x _ { i } \} _ { i = 1 } ^ { N } , \ x _ { i } \in$   $[ 0 , 1 ] ^ { d }$ </td><td>Dataset of N images (dimension  $d ) .$ </td></tr><tr><td> $\mathbf { \Lambda } _ { Y } ^ { Y } = \{ 1 , \dots , C \} , \ y _ { i } \in$ </td><td>Label set and label of  $x _ { i } .$ </td></tr><tr><td> $h _ { \theta } : [ 0 , 1 ] ^ { d } \to \mathbb { R } ^ { m }$ </td><td>CNN feature extractor.</td></tr><tr><td> $z _ { i } = h _ { \theta } ( x _ { i } ) \in \mathbb { R } ^ { m }$ </td><td>Embedding of  $x _ { i } .$ </td></tr><tr><td> $T _ { k } ^ { ( b ) }$ </td><td>Prime quantization map with key b and resolution  $k .$ </td></tr><tr><td> $Q ^ { k , b } ( x )$ </td><td>Prime-quantized embedding of x.</td></tr><tr><td> $P$ </td><td>Prime-quantized space (Euclidean metric).</td></tr><tr><td>Product spaces &amp; metrics</td><td></td></tr><tr><td> $\mathcal { C } = ( Z \times Y , d c )$ </td><td>CNN-label product space;</td></tr><tr><td> $\mathcal { K } = ( P \times Y , d \kappa )$ </td><td> $d _ { \mathcal { C } } ( ( z , y ) , ( \dot { z } ^ { \prime } , y ^ { \prime } ) ) \stackrel { * } { = } \operatorname* { m a x } \{ \| z - z ^ { \prime } \| _ { 2 } , 1 [ y \neq y ^ { \prime } ] \} .$  Prime-label product space (analogous metric).</td></tr><tr><td></td><td></td></tr><tr><td>Neighborhoods &amp; radii</td><td></td></tr><tr><td> $\mathcal { N } _ { K } ^ { \mathsf { M } } ( q )$ </td><td>K-nearest neighbors of q in  $\mathsf { M } \in \{ \mathcal { C } , \mathcal { K } \}$ </td></tr><tr><td> $R _ { \mathsf { M } } , R _ { \mathsf { M } } ^ { \mathrm { a d v } }$ </td><td>Clean / adversarial K-NN radii in M.</td></tr><tr><td> $r _ { i } ^ { \mathsf { M } }$   $\gamma _ { \mathsf { M } }$ </td><td>Distance of i-th neighbor to q in M.</td></tr><tr><td>GW quantities &amp; envelopes</td><td>Separation gap between outer and inner neighbor groups.</td></tr><tr><td> $\mathrm { G W ^ { 2 } , \ G W _ { \lambda } ^ { 2 } }$ </td><td>Quadratic and entropic GW discrepancies.</td></tr><tr><td> $D _ { \mathsf { M } }$ </td><td>Distance matrix in space M.</td></tr><tr><td>π</td><td></td></tr><tr><td></td><td>Coupling (transport plan) in GW.</td></tr><tr><td> $U . , L .$   $\tau$ </td><td>GW upper / lower envelopes. Two-sided GW margin.</td></tr><tr><td>Perturbations &amp; constants</td><td></td></tr><tr><td></td><td>Adversarial perturbation and budget.</td></tr><tr><td> $\eta , \| \eta \| _ { \infty } \leq \epsilon$ </td><td></td></tr><tr><td> $\mu c$ </td><td>Median pairwise distance in C.</td></tr><tr><td> $\mu _ { \mathcal { K } } ^ { \mathrm { k e y } } , \mu _ { \mathcal { K } } ^ { \mathrm { m a x } }$ </td><td>Key-annealed median; maximum over dataset.</td></tr><tr><td>σ</td><td>Variance proxy (Assumption A1).</td></tr><tr><td> $G _ { k } , C _ { k } = 2 G _ { k }$ </td><td>Prime gap bound (Dusart) and sensitivity constant.</td></tr><tr><td>δ</td><td>Confidence parameters (clean/env/grad/est/gap).</td></tr><tr><td> $d , m , C , N , K$ </td><td>Input dim, embedding dim, #classes, #samples, #neighbors.</td></tr><tr><td> $b \in \{ 0 , 1 \} ^ { d } , \ k$ </td><td>Secret key bits; quantization resolution</td></tr></table>

Table 4: Notation summary used throughout the paper.

## B Proofs for Prime Quantization Results

Proof Roadmap. The auxiliary results in this section establish the robustness of the prime quantization transform under bounded perturbations. We begin with Lemma 5, which shows how a perturbation of size ϵ translates into an integer drift in the scaled domain. This feeds directly into Theorem 4, which proves that whenever the perturbation budget exceeds the distance to a prime boundary, one can construct a feasible perturbation that crosses the gap, thereby changing the quantized value. To complement this, we define the absorption radius (Definition 7), derive the guarantee that perturbations below this radius are absorbed (Proposition 1), and bound the possible size of this radius in Lemma 1. Finally, Corollary 1 ties these ingredients together, yielding a crisp detection condition: perturbations below the absorption radius leave quantization unchanged, while those above it necessarily induce a detectable change.

Lemma 5 (Perturbation budget constraint on scaled pixel values). Let $x \in [ 0 , 1 ] ^ { d }$ be an image and $x _ { j } \in [ 0 , 1 ]$ its j-th pixel. Consider a perturbation vector $\eta \in \mathbb { R } ^ { d }$ with $\| \eta \| _ { \infty } \le \epsilon$ . Then for each pixel $x _ { j }$

$$
\begin{array} { r } { \left| S _ { k } ( x _ { j } + \eta _ { j } ) - S _ { k } ( x _ { j } ) \right| \ \leq \ \lfloor \epsilon \cdot 1 0 ^ { k } \rfloor + 1 . } \end{array}
$$

Proof. We have $S _ { k } ( x _ { j } + \eta _ { j } ) = \left\lfloor ( x _ { j } + \eta _ { j } ) \cdot 1 0 ^ { k } \right\rfloor = \left\lfloor x _ { j } \cdot 1 0 ^ { k } + \eta _ { j } \cdot 1 0 ^ { k } \right\rfloor$ and $S _ { k } ( x _ { j } ) = \lfloor x _ { j } \cdot 1 0 ^ { k } \rfloor$ Since $\| \eta \| _ { \infty } \le \epsilon ,$ it follows that $| \eta _ { j } | \le \epsilon$ and hence $| \eta _ { j } \cdot 1 0 ^ { k } | \le \epsilon \cdot 1 0 ^ { k }$ . By the floor inequality $| \lfloor a + \dot { b } \rfloor - \lfloor a \rfloor | \leq \lceil | b | \rceil$ ⌉, it follows that,

$$
\begin{array} { r l } & { | S _ { k } ( x _ { j } + \eta _ { j } ) - S _ { k } ( x _ { j } ) | = | \lfloor x _ { j } \cdot 1 0 ^ { k } + \eta _ { j } \cdot 1 0 ^ { k } | - \lfloor x _ { j } \cdot 1 0 ^ { k } \rfloor | } \\ & { \qquad \leq \ | \eta _ { j } \cdot 1 0 ^ { k } | + 1 } \\ & { \qquad \leq \ \lfloor \epsilon \cdot 1 0 ^ { k } \rfloor + 1 . } \end{array}
$$

Theorem 4 (ϵ-Dependent Gap-Crossing Detection). Let $x _ { j } \in [ 0 , 1 ]$ be the j-th pixel of an image, with $S _ { k } ( x _ { j } ) = n \in ( p _ { l } , p _ { l + 1 } )$ , and let $\epsilon > 0$ be a perturbation budget such that

$$
\epsilon \cdot 1 0 ^ { k } > \operatorname* { m i n } \{ n - p _ { l } , \ : p _ { l + 1 } - n \} .
$$

Then there exists a perturbation $\eta _ { j }$ with $| \eta _ { j } | \le \epsilon$ such that $S _ { k } ( x _ { j } )$ and $S _ { k } ( x _ { j } + \eta _ { j } )$ lie in different prime gap intervals, and hence

$$
T _ { k } ^ { ( b _ { j } ) } ( x _ { j } + \eta _ { j } ) \neq T _ { k } ^ { ( b _ { j } ) } ( x _ { j } )
$$

for any secret bit $b _ { j } \in \{ 0 , 1 \}$

Proof. From Lemma 5, any perturbation $| \eta _ { j } | \le \epsilon$ induces an integer drift in the scaled domain of at most $\lfloor \epsilon \cdot 1 0 ^ { k } \rfloor + 1$ . Thus, whenever $\epsilon \cdot 1 0 ^ { k }$ exceeds the distance from n to the nearest prime boundary, some perturbation $\eta _ { j }$ exists that pushes $S _ { k } ( x _ { j } )$ across that boundary.

Since $S _ { k } ( x _ { j } ) = n \in ( p _ { l } , p _ { l + 1 } )$ , two cases arise:

(i) Closer to p<sub>l</sub>. If $n - p _ { l } \le p _ { l + 1 } - n$ and $\epsilon \cdot 1 0 ^ { k } > n - p _ { l }$ , choose $\eta _ { j } < 0$ with $- \epsilon \leq \eta _ { j } <$ $- ( n - p _ { l } ) / 1 0 ^ { k }$ . Then $S _ { k } ( x _ { j } + \eta _ { j } ) \le \lfloor n + \eta _ { j } \cdot 1 0 ^ { k } \rfloor < p _ { l } \mathrm { { } }$ , placing the perturbed value in $( p _ { j - 1 } , p _ { l } )$ (ii) Closer to $p _ { l + 1 } . \mathrm { I f } p _ { l + 1 } - n < n - p _ { l }$ and $\epsilon \cdot 1 0 ^ { k } > p _ { l + 1 } - n$ , choose $\eta _ { j } > 0$ with $( p _ { l + 1 } - n ) / 1 0 ^ { k } <$ $\eta _ { j } \leq \epsilon .$ . Then $S _ { k } ( x _ { j } + \eta _ { j } ) \ge \lfloor n + \eta _ { j } \cdot 1 0 ^ { k } \rfloor > p _ { l + 1 }$ , placing the perturbed value in $( p _ { l + 1 } , p _ { j + 2 } )$

In both cases, $S _ { k } ( x _ { j } + \eta _ { j } )$ and $S _ { k } ( x _ { j } )$ lie in different prime-gap intervals. Since $R _ { k } ^ { ( b _ { j } ) }$ rounds each integer to one of the two primes bracketing its interval, the images

$$
R _ { k } ^ { ( b _ { j } ) } ( S _ { k } ( x _ { j } ) ) \in \{ p _ { l } , p _ { l + 1 } \} , \qquad R _ { k } ^ { ( b _ { j } ) } ( S _ { k } ( x _ { j } + \eta _ { j } ) ) \in \{ p _ { \ell } , p _ { \ell + 1 } \} , \ \ell \neq j \qquad 
$$

must map to disjoint prime sets. Hence

$$
T _ { k } ^ { ( b _ { j } ) } ( x _ { j } ) = S _ { k } ^ { - 1 } ( R _ { k } ^ { ( b _ { j } ) } ( n ) ) \enspace \ne \enspace S _ { k } ^ { - 1 } ( R _ { k } ^ { ( b _ { j } ) } ( m ) ) = T _ { k } ^ { ( b _ { j } ) } ( x _ { j } + \eta _ { j } ) .
$$

Therefore, if $\epsilon \cdot 1 0 ^ { k } > \mathrm { m i n } \{ n - p _ { l } , p _ { l + 1 } - n \}$ , some perturbation $| \eta _ { j } | \le \epsilon$ necessarily changes the prime quantization output, regardless of the secret bit $b _ { j }$ □

Remark 1. The condition in Theorem 4 is sufficient: it ensures that some perturbation of size $\leq \epsilon$ crosses a prime boundary, though not every direction must. This simplification is enough for our later GW separation results.

Definition 7 (Absorption radius). For $x _ { j } \in [ 0 , 1 ]$ with $S _ { k } ( x _ { j } ) = n \in ( p _ { \ell } , p _ { \ell + 1 } )$ , the absorption radius is $\begin{array} { r } { r _ { \mathrm { a b s } } ( x _ { j } , k ) = \frac { \operatorname* { m i n } \left\{ n - p _ { \ell } , p _ { \ell + 1 } - n \right\} } { 1 0 ^ { k } } } \end{array}$

Proposition 1 (Absorption guarantee). $I f | \eta _ { j } | \leq r _ { \mathrm { a b s } } ( x _ { j } , k )$ , then $T _ { k } ^ { ( b _ { j } ) } ( x _ { j } + \eta _ { j } ) = T _ { k } ^ { ( b _ { j } ) } ( x _ { j } )$ for all bits b $b _ { j }$

Proof. By Definition 7 of absorption radius $r _ { \mathrm { a b s } } ( x _ { j } , k )$ , we have $S _ { k } ( x _ { j } ) , S _ { k } ( x _ { j } + \eta _ { j } ) \in \left( p _ { \ell } , p _ { \ell + 1 } \right)$ for the same prime gap interval. Since both $S _ { k } ( \bar { x } _ { j } )$ and $S _ { k } ( x _ { j } + \eta _ { j } )$ lie in the same prime gap $\left( p _ { \ell } , p _ { \ell + 1 } \right)$ , the prime rounding operator $R _ { k } ^ { ( b _ { j } ) }$ maps both to the same prime: $R _ { k } ^ { ( b _ { j } ) } ( S _ { k } ( x _ { j } + \eta _ { j } ) ) =$ $R _ { k } ^ { ( b _ { j } ) } ( S _ { k } ( x _ { j } ) )$

Applying $S _ { k } ^ { - 1 }$ to both sides:

$$
T _ { k } ^ { ( b _ { j } ) } ( x _ { j } + \eta _ { j } ) = S _ { k } ^ { - 1 } ( R _ { k } ^ { ( b _ { j } ) } ( S _ { k } ( x _ { j } + \eta _ { j } ) ) ) = S _ { k } ^ { - 1 } ( R _ { k } ^ { ( b _ { j } ) } ( S _ { k } ( x _ { j } ) ) ) = T _ { k } ^ { ( b _ { j } ) } ( x _ { j } ) .
$$

Lemma 6 (Elementary Prime Gap Bound Hardy & Wright (1979)). For any two consecutive primes $p _ { l } < p _ { l + 1 }$ , we have the prime gap as $p _ { l + 1 } - p _ { l } \le p _ { l }$

Assumption 1 (Prime-gap envelope for all $k \geq 2 )$ . Let $N = 1 0 ^ { k }$ . There exists an absolute constant $C _ { 0 } > 0$ covering $N < \overline { { x _ { 0 } } } = 3 9 6 , \overline { { 7 } } 3 8$ such that

$$
G _ { k } : = C _ { 0 } + \frac { N } { 2 5 ( \ln N ) ^ { 2 } }
$$

satisfies $p _ { \ell + 1 } - p _ { \ell } \leq G _ { k }$ for all consecutive primes $p _ { \ell } < p _ { \ell + 1 } \leq N .$ . This is a direct consequence of Proposition 6.8 in Dusart Dusart (2010).

Lemma 1 (Absorption bounds). For any $\begin{array} { r } { x _ { j } \in [ 0 , 1 ] , \frac { 1 } { 1 0 ^ { k } } \leq r _ { \mathrm { a b s } } ( x _ { j } , k ) \leq \frac { 1 } { 2 } . } \end{array}$

Proof. By Definition 7, for the j-th pixel $x _ { j } \in [ 0 , 1 ]$ with $S _ { k } ( x _ { j } ) = n$ and consecutive primes $p _ { \ell } < p _ { \ell + 1 }$ satisfying $p _ { \ell } < n < p _ { \ell + 1 }$ , the absorption radius is

$$
r _ { \mathrm { a b s } } ( x _ { j } , k ) = { \frac { \mathrm { m i n } \{ n - p _ { \ell } , p _ { \ell + 1 } - n \} } { 1 0 ^ { k } } } .
$$

We will now proceed to prove each bound separately.

(i) Lower bound. Since $n \in ( p _ { \ell } , p _ { \ell + 1 } )$ is an integer, it follows that $p _ { \ell + 1 } - p _ { \ell } \ge 2$ , and hence $n - p _ { \ell } \ge 1$ and $p _ { \ell + 1 } - n ~ \geq ~ 1$ . Therefore min $\{ n - p _ { \ell } , ~ p _ { \ell + 1 } - n \} ~ \geq ~ 1$ , which implies $r _ { \mathrm { a b s } } ( x _ { j } , k ) \ \geq \ { \frac { 1 } { 1 0 ^ { k } } }$

(ii) Upper bound. The maximum of min $\{ n - p \ell , \ p _ { \ell + 1 } - n \}$ occurs when n is at the midpoint of the prime gap, i.e., min $\{ n - p _ { \ell } , p _ { \ell + 1 } - n \} \ \leq \ { \frac { p _ { \ell + 1 } - p _ { \ell } } { 2 } }$ . Since $p _ { \ell + 1 } \leq 1 0 ^ { k }$ by construction, it follows that $\begin{array} { r } { r _ { \mathrm { a b s } } ( x _ { j } , k ) \ \leq \ \frac { 1 0 ^ { k } / 2 } { 1 0 ^ { k } } \ = \ \frac 1 2 } \end{array}$ . Hence, we obtain $\begin{array} { r } { \frac { 1 } { 1 0 ^ { k } } \ \leq \ r _ { \mathrm { a b s } } ( x _ { j } , k ) \ \leq \ \frac 1 2 } \end{array}$ , which completes the proof. □

Corollary 1 (Absorption vs. Gap-Crossing Condition). Let $x _ { j } \in [ 0 , 1 ]$ be a pixel and $\epsilon > 0$ a perturbation budget. Then:

(i) Absorption. $I f \epsilon \le r _ { \mathrm { a b s } } ( x _ { j } , k )$ , every $| \eta _ { j } | \le \epsilon$ is absorbed, i.e. $T _ { k } ^ { ( b _ { j } ) } ( x _ { j } + \eta _ { j } ) = T _ { k } ^ { ( b _ { j } ) } ( x _ { j } )$

(i) Gap crossing. $I f \epsilon > r _ { \mathrm { a b s } } ( x _ { j } , k )$ , there exists some $| \eta _ { j } | \le \epsilon f o r$ which $T _ { k } ^ { ( b _ { j } ) } ( x _ { j } + \eta _ { j } ) \neq T _ { k } ^ { ( b _ { j } ) } ( x _ { j } )$

Proof. Part (i) is an immediate consequence of Proposition 1. For part (ii), Theorem 4 ensures that whenever $\epsilon \cdot 1 0 ^ { k } >$ min $\{ n - p _ { \ell } , p _ { \ell + 1 } - n \}$ , equivalently $\epsilon > r _ { \mathrm { a b s } } ( x _ { j } , k )$ , one can construct a perturbation $\eta _ { j }$ that shifts $S _ { k } ( x _ { j } )$ into a different prime-gap interval, thereby altering the prime quantization output. □

Lemma 2 (Collision probability). Fix $k \geq 2$ and let $N : = 1 0 ^ { k }$ . Let x, $x ^ { \prime } \in [ 0 , 1 ] ^ { d }$ be two independent random images with i.i.d. pixel marginals whose densities are bounded by Λ on [0, 1] (in particular, $\Lambda = 1$ for the uniform distribution). For a fixed secret bit vector $b \in \{ 0 , 1 \} ^ { d }$ , let $T _ { k } ^ { ( b ) }$ be the prime–quantization transform (Definition 4). Assume the prime–gap envelope $G _ { k }$ from Assumption 1. Then, Pr $\begin{array} { r } { \left[ T _ { k } ^ { ( b ) } ( x ) = T _ { k } ^ { ( b ) } ( x ^ { \prime } ) \right] \ \leq \ \left( \Lambda ^ { 2 } \frac { G _ { k } } { N } \right) ^ { d } } \end{array}$ . In particular, for $\Lambda = 1$ , the collision probability decays as $\left( G _ { k } / 1 0 ^ { k } \right) ^ { d }$ in the number of pixels.

Proof. Fix two independent images $x , x ^ { \prime } \in [ 0 , 1 ] ^ { d }$ with i.i.d. pixel marginals of density at most Λ on $[ 0 , 1 ]$ . Let $N = 1 0 ^ { k }$ , and for each pixel index $i \in \{ 1 , \ldots , d \}$ define the scaled integers

$$
U _ { i } : = S _ { k } ( x _ { i } ) = \lfloor N x _ { i } \rfloor , \qquad U _ { i } ^ { \prime } : = S _ { k } ( x _ { i } ^ { \prime } ) = \lfloor N x _ { i } ^ { \prime } \rfloor .
$$

Partition $\{ 0 , 1 , \ldots , N - 1 \}$ into prime-gap intervals $I _ { j } = ( p _ { j } , p _ { j + 1 } ) \cap \{ 0 , \dots , N - 1 \}$ with lengths $g _ { j } = | I _ { j } |$ . Since each pixel marginal has densit $r \leq \Lambda$ , the probability of landing in any integer bin is $\dot { \leq } \Lambda / \dot { N }$ . Therefore, for any gap $I _ { j }$

$$
\operatorname* { P r } [ U _ { i } \in I _ { j } ] \le \sum _ { u \in I _ { j } } \frac { \Lambda } { N } = \Lambda \frac { g _ { j } } { N } , \qquad \operatorname* { P r } [ U _ { i } ^ { \prime } \in I _ { j } ] \le \Lambda \frac { g _ { j } } { N } .
$$

For a fixed secret bit $b _ { i } .$ , collision occurs at pixel i if both $U _ { i }$ and $U _ { i } ^ { \prime }$ fall in the same gap $I _ { j }$ , since then $R _ { k } ^ { ( b _ { i } ) }$ maps both to the same prime. By independence of $U _ { i }$ and $U _ { i } ^ { \prime }$

$$
\operatorname* { P r } [ \mathrm { c o l l i s i o n ~ a t ~ p i x e l ~ } i ] \ = \ \sum _ { j } \operatorname* { P r } [ U _ { i } \in I _ { j } ] \operatorname* { P r } [ U _ { i } ^ { \prime } \in I _ { j } ] \ \leq \ \sum _ { j } \Big ( \Lambda \frac { g _ { j } } { N } \Big ) ^ { 2 } .
$$

Pixels are i.i.d. across $i ,$ so collisions at all d coordinates occur with probability

$$
\operatorname* { P r } \left[ T _ { k } ^ { ( b ) } ( x ) = T _ { k } ^ { ( b ) } ( x ^ { \prime } ) \right] \ \leq \ \Big ( \Lambda ^ { 2 } \sum _ { j } ( g _ { j } / N ) ^ { 2 } \Big ) ^ { d } .
$$

Finally, note that $\begin{array} { r } { \sum _ { j } ( g _ { j } / N ) ^ { 2 } \le ( \operatorname* { m a x } _ { j } g _ { j } / N ) \cdot \sum _ { j } g _ { j } / N \le G _ { k } / N } \end{array}$ , where $G _ { k }$ is the prime-gap envelope from Assumption 1. Hence, $\begin{array} { r } { \mathrm { P r } \big [ T _ { k } ^ { ( b ) } ( x ) = T _ { k } ^ { ( b ) } ( x ^ { \prime } ) \big ] \ \leq \ \left( \Lambda ^ { 2 } \frac { G _ { k } } { N } \right) ^ { d } } \end{array}$ , yielding the stated bound. □

Remark 2 (Numerics and scope). For $k = 3 ( N = 1 0 ^ { 3 } )$ with empirical maximum gap $G _ { k } =$ 36, the per-pixel factor is $3 6 / \bar { 1 } 0 0 0 = 0 . 0 3 6$ , so for $C I F A R - I O \ : ( d = 3 0 7 2 )$ the bound is at most $( 0 . 0 3 6 ) ^ { 3 0 7 2 } \ \approx \ 1 0 ^ { - 4 . 4 \times 1 0 ^ { 3 } }$ . For $k = 4 ( N = 1 0 ^ { 4 } , G _ { k } = 3 6 )$ , the per-pixel factor is $3 . 6 \times 1 0 ^ { - 3 }$ and the overall bound is even smaller. This result is distributional, i.e., it certifies that collisions are exponentially unlikely for two independent draws with bounded pixel densities. It does not claim that $T _ { k } ^ { ( b ) }$ is injective on $[ 0 , 1 ] ^ { d }$ (the map is many-to-one by construction). Rather, it quantifies that image-level collisions are negligible under natural sampling.

## C Proofs for bounds in $\ell _ { p }$ product metric spaces

Roadmap. In Section 5, we introduced the spatial–semantic product spaces C and K, both endowed with the $\ell _ { \infty }$ metric. The purpose of this appendix is to justify that choice. We first establish in Theorem 5 that upper bounds in an $\ell _ { p }$ product space always imply corresponding bounds in the component spaces. We then prove in Corollary 2 that among all $\ell _ { p }$ metrics, $\ell _ { \infty }$ achieves the tightest possible uniform upper bound. Together these results explain why $\ell _ { \infty }$ is the natural metric for $\bar { \boldsymbol { c } }$ and $\kappa ,$ ensuring that perturbations in either spatial geometry or class distribution immediately translate into separation in the product space.

Let $( X , d _ { X } )$ and $( Y , d _ { Y } )$ be metric spaces. We consider their product space $W = X \times Y$ endowed with a standard $\ell _ { p }$ product metric. For $p \in [ 1 , \infty )$ , this metric is defined by

$$
d _ { W , p } ( ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) ) : = \bigl ( d _ { X } ( x _ { 1 } , x _ { 2 } ) ^ { p } + d _ { Y } ( y _ { 1 } , y _ { 2 } ) ^ { p } \bigr ) ^ { 1 / p } ,\tag{5}
$$

and for $p = \infty$ by

$$
d _ { W , \infty } ( ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) ) : = \operatorname* { m a x } \left\{ d _ { X } ( x _ { 1 } , x _ { 2 } ) , d _ { Y } ( y _ { 1 } , y _ { 2 } ) \right\} .\tag{6}
$$

We now establish a general theorem relating upper bounds in the product space $W$ to upper bounds in the component spaces X and $Y$

Theorem 5 (Component-wise Upper Bounds from $\ell _ { p }$ Product Metrics). $L e t \left( X , d _ { X } \right)$ and $( Y , d _ { Y } )$ be metric spaces, and let $W = \bar { X ^ { \times } } Y$ with the $\ell _ { p }$ metric $d _ { W , p } f o r$ some $p \in [ 1 , \infty ]$ . Suppose there exists a constant $M \geq 0$ such that

$$
d _ { W , p } ( ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) ) \leq M , \quad \forall ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) \in W .\tag{7}
$$

Then thefollowing component-wise bounds hold:

$$
d _ { X } ( x _ { 1 } , x _ { 2 } ) \leq M , \quad d _ { Y } ( y _ { 1 } , y _ { 2 } ) \leq M , \quad \forall x _ { 1 } , x _ { 2 } \in X , y _ { 1 } , y _ { 2 } \in Y .\tag{8}
$$

Proof. We consider two cases:

$( i ) 1 \leq p < \infty$ . By definition, for any $( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) \in W$

$$
d _ { W , p } ( ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) ) = \left( d _ { X } ( x _ { 1 } , x _ { 2 } ) ^ { p } + d _ { Y } ( y _ { 1 } , y _ { 2 } ) ^ { p } \right) ^ { 1 / p } .
$$

Since $d _ { X } ( x _ { 1 } , x _ { 2 } ) ^ { p } \geq 0$ and $d _ { Y } ( y _ { 1 } , y _ { 2 } ) ^ { p } \geq 0$ , it immediately follows that

$$
d _ { X } ( x _ { 1 } , x _ { 2 } ) ^ { p } \leq d _ { X } ( x _ { 1 } , x _ { 2 } ) ^ { p } + d _ { Y } ( y _ { 1 } , y _ { 2 } ) ^ { p } = d _ { W , p } ( ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) ) ^ { p } .
$$

Taking the p-th root on both sides gives $d _ { X } ( x _ { 1 } , x _ { 2 } ) \leq d _ { W , p } ( ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) ) \leq M$ . An identical argument applies to $d _ { Y } ( y _ { 1 } , y _ { 2 } )$

(ii) $p = \infty$ . By definition,

$$
d _ { W , \infty } ( ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) ) = \operatorname* { m a x } \{ d _ { X } ( x _ { 1 } , x _ { 2 } ) , d _ { Y } ( y _ { 1 } , y _ { 2 } ) \} .
$$

Hence, by properties of the maximum function,

$$
d _ { X } ( x _ { 1 } , x _ { 2 } ) \leq d _ { W , \infty } ( ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) ) \leq M , \quad d _ { Y } ( y _ { 1 } , y _ { 2 } ) \leq d _ { W , \infty } ( ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) ) \leq M .
$$

Combining the two cases, the theorem follows.

Corollary 2 (Tightest Upper Bound in $\ell _ { p }$ Product Spaces). Let $( X , d _ { X } )$ and $( Y , d _ { Y } )$ have known upper bounds $M _ { X }$ and $M _ { Y }$ respectively, $i . e .$

$$
d _ { X } ( x _ { 1 } , x _ { 2 } ) \leq M _ { X } , \quad d _ { Y } ( y _ { 1 } , y _ { 2 } ) \leq M _ { Y } , \quad \forall x _ { 1 } , x _ { 2 } \in X , y _ { 1 } , y _ { 2 } \in Y .
$$

Then the corresponding upper bound for the product space $( W , d _ { W , p } )$ is

$$
\begin{array} { r } { d _ { W , p } ( ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) ) \leq \left\{ \begin{array} { l l } { ( M _ { X } ^ { p } + M _ { Y } ^ { p } ) ^ { 1 / p } , } & { 1 \leq p < \infty , } \\ { \operatorname* { m a x } \{ M _ { X } , M _ { Y } \} , } & { p = \infty . } \end{array} \right. } \end{array}
$$

Moreover, among all $\ell _ { p }$ product metrics, the $\ell _ { \infty }$ metric achieves the tightest upper bound, $i . e . ,$

$$
\operatorname* { m a x } \{ M _ { X } , M _ { Y } \} \leq ( M _ { X } ^ { p } + M _ { Y } ^ { p } ) ^ { 1 / p } , \quad \forall p \in [ 1 , \infty ) ,
$$

and is therefore optimal when minimizing the guaranteed upper bound in the product space.

Proof. The bound for $1 \leq p < \infty$ follows directly from the monotonicity of the $\ell _ { p }$ norm:

$$
d _ { W , p } ( ( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) ) \leq ( M _ { X } ^ { p } + M _ { Y } ^ { p } ) ^ { 1 / p } .
$$

For $p = \infty .$ , by definition $d _ { W , \infty } = \operatorname* { m a x } \{ d _ { X } , d _ { Y } \} \leq \operatorname* { m a x } \{ M _ { X } , M _ { Y } \}$ . To see that $\ell _ { \infty }$ is the tightest, observe that for any $p < \infty , ( M _ { X } ^ { p } + M _ { Y } ^ { p } ) ^ { 1 / p } \geq \operatorname* { m a x } \{ M _ { X } , M _ { Y } \}$ . Equality occurs only if one of $M _ { X }$ or $M _ { Y }$ is zero. Hence, $\ell _ { \infty }$ gives the smallest guaranteed upper bound over all $\ell _ { p }$ norms.

Definition 8 (Spatial, semantic, and product metric spaces). Let $X \subset [ 0 , 1 ] ^ { d }$ be the image space and $Y = \{ 1 , \dots , \bar { C } \}$ the label set. For a representation map $f : X \to { \dot { \mathbb { R } } } ^ { m }$ , define the spatial metric space $\grave { (} M ^ { X } , d _ { X } \grave { ) }$ ) with $M ^ { X } = \{ f ( x ) : x { \dot { \in } } X \}$ and $d _ { X } ( x _ { 1 } , x _ { 2 } ) = \| x _ { 1 } - x _ { 2 } \| _ { 2 } .$ . The semantic metric space is $( M ^ { Y } , d _ { Y } )$ with $M ^ { Y } = Y$ and $d _ { Y } ( y _ { 1 } , y _ { 2 } ) = \mathbf { 1 } [ y _ { 1 } \neq y _ { 2 } ]$

Their $\ell _ { \infty }$ product is the metric space

$$
\begin{array} { r } { \boldsymbol { M } ^ { X Y } = ( \boldsymbol { M } ^ { X } \times \boldsymbol { M } ^ { Y } , \ d \boldsymbol { x } \boldsymbol { r } ) , \qquad \boldsymbol { d } _ { X Y } \big ( ( \boldsymbol { x } _ { 1 } , y _ { 1 } ) , ( \boldsymbol { x } _ { 2 } , y _ { 2 } ) \big ) = \operatorname* { m a x } \{ \boldsymbol { d } _ { X } ( \boldsymbol { x } _ { 1 } , \boldsymbol { x } _ { 2 } ) , \ d _ { Y } ( y _ { 1 } , y _ { 2 } ) \} . } \end{array}
$$

Each image $x \in X$ embeds as $( f ( x ) , y )$ where $y \in Y$ is its class label. Projections are defined by $\Pi _ { X } ( x , y ) = x$ and $\Pi _ { Y } ( x , y ) = y$ . Instantiating $f = h _ { \theta }$ yields the space C with spatial component $Z ,$ and instantiating $f = T _ { k } ^ { ( b ) }$ yields K with spatial component P.

Instantiating $f ( x ) = h _ { \theta } ( x ) { \mathrm { o r } } f ( x ) = T _ { k } ^ { ( b ) } ( x )$ yields the product spaces C and $\kappa ,$ respectively.   
These will serve as the foundation for the GW bounds in Section 5.

## D Proofs for Diameter Bounds in C and K

Proposition 2 (Concentration in $\ell _ { \infty }$ product spaces). Let $\{ ( X _ { i } , d _ { i } ) \} _ { i = 1 } ^ { n }$ be metric spaces and let $W =$ $\textstyle \prod _ { i = 1 } ^ { n } X _ { i }$ be endowed with the $\ell _ { \infty }$ product metric $d _ { \infty } \big ( ( x _ { i } ) _ { i = 1 } ^ { n } , ( y _ { i } ) _ { i = 1 } ^ { n } \big ) : =$ max $\iota \leq i \leq n \ : d _ { i } ( x _ { i } , y _ { i } )$ Let ${ \dot { \bf X } } = ( X _ { 1 } , \ldots , X _ { n } )$ be a random element of W and fix reference points $m _ { i } \in X _ { i } \mathrm { ~ } ( e . g .$ ., means or Fréchet means), writing $\mathbf { n } = ( m _ { 1 } , \ldots , m _ { n } )$

Assume that each coordinate concentrates around its reference point, i.e., there exist tailfunctions $\psi _ { i } : ( 0 , \infty ) \to [ 0 , 1 ]$ such thatfor all $t > 0$

$$
\operatorname* { P r } \left\{ d _ { i } ( X _ { i } , m _ { i } ) \geq t \right\} \ \leq \ \psi _ { i } ( t ) \qquad ( i = 1 , \ldots , n ) .
$$

Then the product random element concentrates around m in $( W , d _ { \infty } ) .$ : for all $t > 0$

$$
\operatorname* { P r } \left\{ d _ { \infty } \big ( \mathbf { X } , \mathbf { m } \big ) \ge t \right\} \ = \ \operatorname* { P r } \Big \{ \operatorname* { m a x } _ { 1 \le i \le n } d _ { i } \big ( X _ { i } , m _ { i } \big ) \ge t \Big \} \ \le \ \sum _ { i = 1 } ^ { n } \psi _ { i } ( t ) .
$$

Proof. The event $\{ d _ { \infty } ( \mathbf { X } , \mathbf { m } ) \geq t \}$ equals $\{ \operatorname* { m a x } _ { i } d _ { i } ( X _ { i } , m _ { i } ) \geq t \}$ , which is contained in the union $\mathsf { U } _ { i } \{ \bar { d } _ { i } ( X _ { i } , m _ { i } ) \geq \bar { t } \}$ . Apply the union bound and the assumed coordinate-wise tail bounds. □

Corollary 3. (1) For $n = 2$ and real-valued coordinates with $d _ { i } ( x , m ) = | x - m |$ , letting $M =$ max $\{ X , { \dot { Y } } \}$ and $m = \operatorname* { m a x } \{ \mathbb { E } X , \mathbb { E } Y \}$ gives

$$
\operatorname* { P r } \{ | M - m | \ge t \} \le \operatorname* { P r } \{ | X - \mathbb { E } X | \ge t \} + \operatorname* { P r } \{ | Y - \mathbb { E } Y | \ge t \} .
$$

Theorem 6 (CNN Product Space Clean Diameter Bound). Let $\mathcal { C } = ( Z \times Y , d _ { \mathcal { C } } )$ be the CNN product space (with $d _ { \mathcal { C } }$ the $\ell _ { \infty }$ product metric; cf. Proposition $^ { 2 ) , }$ , and let d denote the spatial feature dimension of $Z .$ . Fix a clean query z and let $\dot { \mathcal { N } } _ { K } ^ { \mathcal { C } } ( z )$ denote its $K$ -nearest neighbors among i.i.d. clean samples. Thenfor any $\bar { K } \ge 2$ and any $\delta \in ( \mathrm { 0 , 1 } )$ , under a standard $K { \cdot } N N$ sampling regime, with probability at least $1 - \delta ,$

$$
\mathrm { d i a m } \left( { \cal N } _ { K } ^ { \mathcal { C } } ( z ) \right) \ \le \ 2 \mu c + \ 2 \left( \sqrt { \frac { 2 \log K } { d } } + \sqrt { \frac { 2 \log ( 2 / \delta ) } { d } } \right) ,
$$

where $\mu { c }$ is the clean median pairwise distance in $\mathcal { C } .$

Proof. We first establish that distances between clean embeddings in the product space $\mathcal { C }$ satisfy sub-Gaussian concentration properties. This will serve as the foundation for bounding $K { \cdot } \mathbf { N N }$ diameters. Recall the definition of a sub-Gaussian random variable. Throughout this section, the spatial metric $d _ { Z }$ is assumed to be normalized, ensuring concentration at scale $O ( 1 / \sqrt { d } )$

Definition (Sub-Gaussian random variable). A real random variable X is called sub-Gaussian with parameter $\sigma ^ { 2 }$ if for all $t \in \mathbb { R }$

$$
\begin{array} { r } { \mathbb { E } [ e ^ { t X } ] \ \leq \ \exp \left( \frac { \sigma ^ { 2 } t ^ { 2 } } { 2 } \right) . } \end{array}
$$

Equivalently, its tail probabilities satisfy $\begin{array} { r } { \mathbb { P } [ | X - \mathbb { E } [ X ] | \geq t ] \ \leq \ 2 \exp \left( - \frac { t ^ { 2 } } { 2 \sigma ^ { 2 } } \right) } \end{array}$

For a random vector $Z = ( Z _ { 1 } , \ldots , Z _ { d } ) \in \mathbb { R } ^ { d }$ , we say Z is sub-Gaussian if every linear functional is sub-Gaussian:

$$
\| Z \| _ { \psi _ { 2 } } \ = \ \operatorname* { s u p } _ { u \in S ^ { d - 1 } } \| \langle Z , u \rangle \| _ { \psi _ { 2 } } \ < \ \infty ,
$$

where for a random variable $Y ,$ the sub-Gaussian norm is $\| Y \| _ { \psi _ { 2 } } = \operatorname* { i n f } \{ t > 0 : \mathbb { E } [ e ^ { Y ^ { 2 } / t ^ { 2 } } ] \leq 2 \}$

Each image x maps to the product space via $z = ( h _ { \theta } ( x ) , y ) \in \mathcal { C }$ , where $h _ { \theta } ( x ) \in Z$ is its CNN embedding and $y \in Y$ its class label. For two clean images $x _ { i } , x _ { j }$ , we denote their embeddings by $z _ { i } , z _ { j }$ and define the product space distance as $D _ { i j } = d _ { C } ( z _ { i } , z _ { j } )$

We now analyze the spatial and semantic components of $D _ { i j }$

Spatial component. The embedding $\Pi _ { Z } ( z ) \in \mathbb { R } ^ { d }$ has sub-Gaussian coordinates due to several architectural and statistical effects. Namely, batch normalization enforces near unit variance and zero mean across feature activations Ioffe & Szegedy (2015); Santurkar et al. (2018). In Poole et al. (2016); Schoenholz et al. (2017), the authors demonstrate CLT effects arise from weighted sums of many independent activations, yielding approximately Gaussian tails. Moreover, Regularization techniques $( \mathrm { e . g . }$ , weight decay Krogh & Hertz (1991), dropout Srivastava et al. (2014)) further constrain magnitudes, supporting sub-Gaussian tails Wager et al. (2013).

Formally, if $\sigma _ { Z } ^ { 2 }$ is the empirical variance of a coordinate in $Z ,$ , then for all $t > 0 ,$

$$
\begin{array} { r } { \mathbb { P } ( | [ \Pi _ { Z } ( z ) ] _ { \ell } - \mathbb { E } [ \Pi _ { Z } ( z ) _ { \ell } ] | \ge t ) \le 2 \exp \left( - \frac { t ^ { 2 } } { 2 \sigma _ { Z } ^ { 2 } } \right) . } \end{array}
$$

Thus $\Pi _ { Z } ( z )$ is sub-Gaussian with $\Vert \Pi _ { Z } ( z ) \Vert _ { \psi _ { 2 } } \leq K _ { Z }$ , where $K _ { Z } = O ( \sigma _ { Z } { \sqrt { d } } )$ . By standard results (see (Vershynin, 2018, Thm. 3.1.1)), Euclidean distances between embeddings in $Z$ concentrate sharply around their mean.

Semantic component. The label projection $\Pi _ { Y } ( z )$ contributes

$$
d _ { Y } ( \Pi _ { Y } ( z _ { i } ) , \Pi _ { Y } ( z _ { j } ) ) = { \bf 1 } \{ y _ { i } \neq y _ { j } \} ,
$$

which is bounded in $\{ 0 , 1 \}$ and deterministic once class labels are fixed.

Product space concentration. Recall that

$$
d _ { \mathcal { C } } ( z _ { i } , z _ { j } ) = \operatorname* { m a x } \{ d _ { Z } ( \Pi _ { Z } ( z _ { i } ) , \Pi _ { Z } ( z _ { j } ) ) , \mathbf { 1 } [ y _ { i } \neq y _ { j } ] \} .
$$

Applying Proposition 2 with $n = 2$ shows that concentration of the spatial distance transfers to the product metric, up to the bounded semantic coordinate. Accordingly, we summarize the resulting clean pairwise-distance behavior via the tail proxy

$$
\mathbb { P } ( | d _ { \mathcal { C } } ( z _ { i } , z _ { j } ) - \mu _ { \mathcal { C } } | \ge t ) \le C \exp ( - c d t ^ { 2 } ) ,\tag{9}
$$

which we use in the remainder of the proof.

K-NN order statistics to diameter bound. Fix a clean query z and let $D _ { i } = d _ { \mathcal { C } } ( z , z _ { i } )$ denote the distance between z and the i.i.d. clean samples $\{ z _ { i } \} _ { i = 1 } ^ { n } .$ . Let $D _ { ( 1 ) } \leq \cdots \leq D _ { ( n ) }$ denote the order statistics. For any threshold τ , the classical characterization of order statistics (David & Nagaraja, 2004, Eq. 2.1.3) gives

$$
\big \{ D _ { ( K ) } \geq \tau \big \} \iff \Big \{ \# \{ i : D _ { i } \geq \tau \} \geq n - K + 1 \Big \} .\tag{10}
$$

The event on the right means that there are at least $n - K + 1$ indices for which $D _ { i } \geq \tau$ . Equivalently, there exists a subset $S \subseteq \{ 1 , \dots , n \}$ with $| S | = n - K + 1$ such that

$$
D _ { i } \geq \tau \quad \forall i \in S .
$$

That is,

$$
\{ D _ { ( K ) } \geq \tau \} \subseteq \bigcup _ { S \subseteq \{ 1 , \ldots , n \} } \bigcap _ { i \in S } \{ D _ { i } \geq \tau \} .\tag{11}
$$

Applying the union bound to equation 11 yields

$$
\mathbb { P } \{ D _ { ( K ) } \geq \tau \} \ \leq \ \sum _ { \tiny \begin{array} { c } { S \subseteq \{ 1 , \ldots , n \} } \\ { | S | = n - K + 1 } \end{array} } \mathbb { P } \Big ( \bigcap _ { i \in S } \{ D _ { i } \geq \tau \} \Big ) .\tag{12}
$$

Since the $D _ { i } \mathrm { s }$ are i.i.d., the probability for any fixed $S$ factors as

$$
\mathbb { P } \Big ( \bigcap _ { i \in S } \{ D _ { i } \geq \tau \} \Big ) = \prod _ { i \in S } \mathbb { P } ( D _ { i } \geq \tau ) = \big ( \mathbb { P } ( D _ { i } \geq \tau ) \big ) ^ { n - K + 1 } .
$$

There are $\binom { n } { n - K + 1 }$ such subsets S. Hence equation 12 simplifies to

$$
\begin{array} { r } { \mathbb { P } \{ D _ { ( K ) } \ge \tau \} \le \binom { n } { n - K + 1 } \big ( \mathbb { P } ( D _ { i } \ge \tau ) \big ) ^ { n - K + 1 } . } \end{array}\tag{13}
$$

We now shift our focus to the next part, where we bound the diameter via pairwise bounds on the K-nearest neighbor set. Let $\mathcal { N } _ { K } ^ { \mathcal { C } } ( z ) \overset { \cdot } { = } \left\{ z _ { ( 1 ) } , \dots , z _ { ( K ) } \right\}$ be the K nearest neighbors of z (ties broken arbitrarily), and write $D _ { ( K ) } : = d _ { \mathcal { C } } ( z , z _ { ( K ) } )$ . By the triangle inequality, for any $1 \leq i < j \leq K$

$$
d \mathcal { C } \big ( z _ { ( i ) } , z _ { ( j ) } \big ) \ \leq \ d _ { \mathcal { C } } \big ( z _ { ( i ) } , z \big ) + d _ { \mathcal { C } } \big ( z , z _ { ( j ) } \big ) \ \leq \ D _ { ( K ) } + D _ { ( K ) } .
$$

Taking the maximum over $i < j$ yields the deterministic bound

$$
\dim \left( { \mathcal { N } } _ { K } ^ { \mathcal { C } } ( z ) \right) \ \leq \ 2 D _ { ( K ) } .\tag{14}
$$

Fix $\tau = \mu c + t$ with $t > 0$ . By equation 13 and the single-distance tail equation 9,

$$
\mathbb { P } \{ D _ { ( K ) } \geq \mu _ { C } + t \} \ \leq \ \left( { n \atop n - K + 1 } \right) \left( \mathbb { P } ( D _ { i } \geq \mu _ { C } + t ) \right) ^ { n - K + 1 } \leq \binom { n } { n - K + 1 } \left( C e ^ { - c d t ^ { 2 } } \right) ^ { n - K + 1 } .
$$

To obtain a high-probability bound, we select $t > 0$ such that the preceding upper bound is at most a prescribed failure probability δ. Taking logarithms of the bound

$$
\binom { n } { n - K + 1 } \left( C e ^ { - c d t ^ { 2 } } \right) ^ { n - K + 1 } ,
$$

this requirement is satisfied whenever

$$
c d t ^ { 2 } \ \geq \ \log C \ + \ \frac { \log \binom { n } { n - K + 1 } + \log ( 1 / \delta ) } { n - K + 1 } .
$$

We work in a standard K-NN sampling regime in which the candidate pool is sufficiently large compared to $K$ (for example, $n \geq K ^ { \tilde { 2 } } )$ , so that $\textstyle { \frac { 1 } { n - K + 1 } } \log { \binom { n } { n - K + 1 } } = O ( \log K )$ . Absorbing absolute constants (including log $C$ and the $O ( \log K )$ term) into c, it suffices to choose

$$
t = \sqrt { \frac { 2 \log K } { d } } + \sqrt { \frac { 2 \log ( 2 / \delta ) } { d } } ,
$$

which yields

$$
\mathbb { P } \{ D _ { ( K ) } \geq \mu c + t \} \ \leq \ \delta .
$$

Combining this bound with equation 14 gives, with probability at least $1 - \delta ,$

$$
\mathrm { d i a m } \left( { \cal N } _ { K } ^ { \mathcal { C } } ( z ) \right) \ \le \ 2 \mu _ { \mathcal { C } } \ + \ 2 \left( \sqrt { \frac { 2 \log K } { d } } + \sqrt { \frac { 2 \log ( 2 / \delta ) } { d } } \right) .
$$

This completes the proof.

Corollary 4 (95% confidence bound). For confidence level $\delta = 0 . 0 5$ , the K-nearest neighbor diameter in C satisfies, under the assumptions ofTheorem 6,

$$
\mathrm { d i a m } \big ( \mathcal { N } _ { K } ^ { \mathcal { C } } ( z ) \big ) \ \le \ 2 \mu _ { \mathcal { C } } \ + \ \frac { 2 } { \sqrt { d } } \Big ( \sqrt { 2 \log K } + 2 . 7 1 7 \Big ) ,
$$

with probability at least 95%. In particular, the excess diameter above the clean median scale decays as $O ( \sqrt { \log K / d } )$

Notation. We adopt the metric space setup of Definition 8. In particular, $\mathcal { C } = ( Z \times Y , d _ { \mathcal { C } } )$ denotes the CNN product space and $\mathcal { K } \doteq ( P \times \bar { Y } , d \kappa )$ the prime-quantized product space. Let $\mu _ { \mathcal { C } }$ and $\mu \kappa$ denote the clean median pairwise distances in $\mathcal { C }$ and $\kappa$ respectively. We consider adversarial perturbations $\eta \in \mathbb { R } ^ { d }$ with $\| \eta \| _ { \infty } \le \epsilon .$ , where $\epsilon > 0$ is the fixed attack budget.

Definition 9 (Adversarial query). Let $x \in X$ be a clean input with ground-truth label $y \in Y$ . For an ϵ-bounded perturbation $\eta \in \mathbb { R } ^ { d }$ , the adversarial query under a representation map $f : X  \mathbb { R } ^ { m }$ is

$$
\big ( f ( x + \eta ) , y \big ) \in \mathcal { F } \times Y ,
$$

where $\mathcal { F }$ denotes the feature space associated with $f .$ . Thus, adversarial queries live in the same feature–label product space as clean points. Note that a classifier may produce a prediction $\hat { y } \neq y ;$ but yˆ is not part of the definition. Instantiating $f = h _ { \theta }$ gives ${ \mathcal { F } } = Z$ and hence adversarial queries in the CNN product space ${ \mathcal { C } } = Z \times Y$ , while instantiating $f = T _ { k } ^ { ( b ) }$ gives ${ \mathcal { F } } = P$ and hence adversarial queries in the prime-quantized product space $\kappa = \overset { \smile } { P } \times \overset { \vartriangle } { Y }$

Assumption 2 (A1: variance-only control via Jacobian proxy). Let $\Delta ( \eta , x ) : = \| f ( x + \eta ) - f ( x ) \| .$ 2 be the feature displacement under perturbation $\eta .$ . For small perturbations with $\| \eta \| _ { \infty } \le \epsilon ,$ , we assume there exists a constant $\sigma ^ { 2 } > 0$ (a variance proxy) such that

$$
\mathbb { E } \big [ \Delta ( \eta , x ) ^ { 2 } \big ] \ \leq \ d \sigma ^ { 2 } \epsilon ^ { 2 } .\tag{15}
$$

Interpretation. Byfirst-order Taylor expansion, $f ( x + \eta ) - f ( x ) \approx J ( x ) \eta ,$ , where $J ( x ) = \nabla f ( x ) \in$ $\mathbb { R } ^ { m \times d }$ is the Jacobian of f at x (with m the feature dimension). Thus,

$$
\Delta ( \eta , x ) ^ { 2 } \approx \| J ( x ) \eta \| _ { 2 } ^ { 2 } = \sum _ { r = 1 } ^ { m } \langle J _ { r , . } ( x ) , \eta \rangle ^ { 2 } .
$$

Assumption equation 15 requires that each row $J _ { r , \cdot } ( x )$ has second moment bounded by $\sigma ^ { 2 }$ , so that the expected squared shift across m features grows at most linearly with d (via $\| \eta \| _ { \infty } \le \epsilon )$ and quadratically with ϵ.

By Chebyshev’s inequality, for any $\delta _ { \mathrm { g r a d } } \in ( 0 , 1 )$

$$
\Delta ( \eta , x ) \ \leq \ \frac { \sqrt { d } \sigma } { \sqrt { \delta _ { \mathrm { g r a d } } } } \epsilon \quad w i t h p r o b a b i l i t y a t l e a s t 1 - \delta _ { \mathrm { g r a d } } .\tag{16}
$$

Here $\delta _ { \mathrm { g r a d } }$ acts as a tolerance parameter: it specifies the probability mass we are willing to allocate to rare large deviations in feature shifts. Smaller values of $\cdot \delta _ { \mathrm { g r a d } }$ yield higher-probability guarantees but make the bound looser. This provides a high-probability control ofadversarialfeature shifts using only variance information, without assuming Lipschitz continuity or sub-Gaussianity ofthe Jacobian.

Remark 3 (Relating $\ell _ { \infty }$ and $\ell _ { 2 }$ budgets). An $\ell _ { \infty }$ budget ϵ implies an $\ell _ { 2 }$ budget $\epsilon _ { 2 } \leq \sqrt { d } \epsilon ,$ , and conversely an $\ell _ { 2 }$ budget $\epsilon _ { 2 }$ implies $\ell _ { \infty } b u d g e t \geq \epsilon _ { 2 } / \sqrt { d } .$ . This allows translating $\ell _ { 2 } \cdot$ –based results to our $\ell _ { \infty }$ setting and vice versa.

In the adversarial setting, the k-NN neighborhood can enlarge primarily through displacement of the query relative to its clean location. Thus, bounding the query’s displacement (via Assumption 2) allows us to extend the clean k-NN diameter bound to the adversarial case.

Theorem 7 (Adversarial K-NN diameter bound in ${ \mathcal { C } } ) .$ . Fix $K \_ { \mathrm { ~ \scriptsize ~ 2 ~ } }$ and confidence levels $\delta _ { \mathrm { c l e a n } } , \delta _ { \mathrm { g r a d } } \ \in \ ( 0 , 1 )$ . Let x be a clean query with label y, and let x + η be an adversarially perturbed query with $\varepsilon = \| \eta \| _ { \infty } . L e t z = ( f ( x ) , y )$ and let $\boldsymbol { q } = ( f ( \boldsymbol { x } + \eta ) , \hat { y } )$ denote the adversarial query (with yˆ the classifier prediction).

Assume: (i) Theorem 6 (clean K-NN diameter concentration); (ii) Assumption 2 (variance-controlled feature displacement); (iii) the K-NN gap condition

$$
D _ { ( K + 1 ) } ( z ) - D _ { ( K ) } ( z ) > 2 d _ { \mathcal { C } } ( z , q ) ,\tag{17}
$$

where $D _ { ( r ) } ( z )$ denotes the r-th nearest-neighbor distance from z (ties excluded).

Then, with probability at least $1 - ( \delta _ { \mathrm { c l e a n } } + \delta _ { \mathrm { g r a d } } )$

$$
\mathrm { d i a m } \left( \mathcal { N } _ { K } ^ { \mathcal { C } } ( x + \eta ) \right) \leq 2 \mu _ { \mathcal { C } } + 2 \left( \sqrt { \frac { 2 \log K } { d } } + \sqrt { \frac { 2 \log ( 2 / \delta _ { \mathrm { c l e a m } } ) } { d } } \right) + \frac { 2 \sqrt { d } \sigma } { \sqrt { \delta _ { \mathrm { g r a d } } } } \varepsilon + 2 \mathbf { 1 } _ { \{ y \neq \hat { y } \} } .\tag{18}
$$

Proof. By Theorem 6, for any fixed clean query $z = ( f ( x ) , y )$ , we have

$$
\mathrm { d i a m } \left( \mathcal { N } _ { K } ^ { \mathcal { C } } ( z ) \right) \ \leq \ 2 \mu c \ + \ 2 \left( \sqrt { \frac { 2 \log K } { d } } + \sqrt { \frac { 2 \log ( 2 / \delta _ { c l e a n } ) } { d } } \right) ,
$$

with probability at least $1 - \delta _ { \mathrm { c l e a n } }$ over the i.i.d. clean sample.

Assumption 2 bounds the feature shift $\Delta ( \eta , x ) = \| f ( x + \eta ) - f ( x ) \| _ { 2 }$ . Since $d _ { \mathcal { C } }$ is an $\ell _ { \infty }$ product metric, a label change contributes at most $\mathbf { 1 } _ { \{ y \neq \hat { y } \} }$ , and therefore

$$
d _ { \mathcal { C } } ( z , q ) \ \leq \ \| \boldsymbol { f } ( \boldsymbol { x } + \boldsymbol { \eta } ) - \boldsymbol { f } ( \boldsymbol { x } ) \| _ { 2 } + \mathbf { 1 } _ { \{ \boldsymbol { y } \neq \boldsymbol { \hat { y } } \} } .
$$

For any dataset point u, the triangle inequality implies $\begin{array} { r } { \left| d _ { \mathcal { C } } ( q , u ) - d _ { \mathcal { C } } ( z , u ) \right| ~ \le ~ d _ { \mathcal { C } } ( z , q ) } \end{array}$ , so all query–dataset distances are perturbed by at most $d \boldsymbol { c } ( \boldsymbol { z } , \boldsymbol { q } )$ . Let $\{ u _ { 1 } , \ldots , u _ { n } \}$ denote the dataset, and let $u _ { ( 1 ) } , \ldots , u _ { ( n ) }$ denote the same points reordered so that $d _ { \mathcal { C } } \left( z , u _ { \left( 1 \right) } \right) \leq \ldots \leq d _ { \mathcal { C } } ( z , u _ { \left( n \right) } )$ , with $D _ { ( r ) } ( z ) : = d c ( z , u _ { ( r ) } )$

Under the gap condition equation 17, we now show that the K-nearest-neighbor set is stable under the perturbation from z to q. For any dataset point u, the triangle inequality implies

$$
\left| d _ { \mathcal { C } } ( q , u ) - d _ { \mathcal { C } } ( z , u ) \right| \ \leq \ d _ { \mathcal { C } } ( z , q ) .
$$

Let $u _ { ( 1 ) } , \ldots , u _ { ( n ) }$ denote the dataset points ordered by increasing distance from the clean query z, so that $d _ { C } ( z , u _ { ( r ) } ) = D _ { ( r ) } ( z )$ . In particular,

$$
d _ { \mathcal { C } } ( \boldsymbol { q } , \boldsymbol { u } _ { ( K ) } ) \leq D _ { ( K ) } ( \boldsymbol { z } ) + d _ { \mathcal { C } } ( \boldsymbol { z } , \boldsymbol { q } ) , \qquad d _ { \mathcal { C } } ( \boldsymbol { q } , \boldsymbol { u } _ { ( K + 1 ) } ) \geq D _ { ( K + 1 ) } ( \boldsymbol { z } ) - d _ { \mathcal { C } } ( \boldsymbol { z } , \boldsymbol { q } ) .
$$

Subtracting these two inequalities yields

$$
d _ { \mathcal { C } } ( \boldsymbol { q } , u _ { ( K + 1 ) } ) - d _ { \mathcal { C } } ( \boldsymbol { q } , u _ { ( K ) } ) \geq \big ( D _ { ( K + 1 ) } ( \boldsymbol { z } ) - D _ { ( K ) } ( \boldsymbol { z } ) \big ) - 2 d _ { \mathcal { C } } ( \boldsymbol { z } , \boldsymbol { q } ) .
$$

By the gap condition equation 17, the right-hand side is strictly positive, which implies $d _ { \cal C } \big ( q , u _ { ( \cal K ) } \big ) <$ $d _ { C } \big ( q , u _ { ( K + 1 ) } \big )$ . Since $D _ { ( j ) } ( z ) \geq D _ { ( K + 1 ) } ( z )$ for all $j \ge K + 1$ , the above inequality implies that no index $j \geq K + 1$ can enter the K-NN set under the perturbation. Consequently, the identity of the $K$ nearest neighbors is preserved when the query moves $\mathrm { f r o m } \ z \mathrm { t o } \ q$ , and therefore $\mathcal { N } _ { K } ^ { \mathcal { C } } ( q ) \doteq \mathcal { N } _ { K } ^ { \mathcal { C } } ( z )$

Let $S = \mathcal { N } _ { K } ^ { \mathcal { C } } ( z ) = \mathcal { N } _ { K } ^ { \mathcal { C } } ( q )$ . For any $u , v \in S$ , the triangle inequality yields

$$
d _ { \mathcal { C } } ( u , v ) \ \leq \ d _ { \mathcal { C } } ( u , z ) + d _ { \mathcal { C } } ( z , v ) \ \leq \ d _ { \mathcal { C } } ( u , q ) + d _ { \mathcal { C } } ( q , v ) + 2 d _ { \mathcal { C } } ( z , q ) .
$$

Taking maxima over $u , v \in S$ gives diam $\left( \mathcal { N } _ { K } ^ { \mathcal { C } } ( q ) \right) \ \leq \ \mathrm { d i a m } \left( \mathcal { N } _ { K } ^ { \mathcal { C } } ( z ) \right) \ + \ 2 d _ { \mathcal { C } } ( z , q )$ . Intersecting the clean diameter event and the displacement event and applying a union bound yields probability at least $1 - ( \delta _ { \mathrm { c l e a n } } + \delta _ { \mathrm { g r a d } } )$ . On this event, we arrive at the claimed bound. □

Assumption 3 (Prime-gap sensitivity under bit flips). $F i x \ k \geq 2$ and let $G _ { k }$ be the prime-gap envelope from Assumption 1. Then, for every coordinate $j \in \{ 1 , \ldots , d \}$ , every two images x, $x ^ { \prime } \in X$ and any two secret keys b, $b ^ { \prime } \in \{ 0 , 1 \} ^ { d }$ that differ only at bit j, we have

$$
\begin{array} { r l r } { \boxed { \underbrace { \left[ T _ { k } ^ { ( b ) } ( x ) - T _ { k } ^ { ( b ) } ( x ^ { \prime } ) \right] _ { j } } } _ { d i f f r e n c e } - \underbrace { \left[ T _ { k } ^ { ( b ^ { \prime } ) } ( x ) - T _ { k } ^ { ( b ^ { \prime } ) } ( x ^ { \prime } ) \right] _ { j } } _ { d i f f e r e n c e } \ \le } & { 2 G _ { k } . } \end{array}\tag{19}
$$

Explanation. Flipping one bit can move the j-th coordinate by at most one local prime $g a p \leq G _ { k } .$ For two inputs, both coordinates may shift, so by the triangle inequality their pairwise difference changes by at most $2 G _ { k } ,$ , regardless of the absolute difference.

Definition 10 (Key-annealed (data-quenched) median in K). Fix a clean query $x \in X$ and afixed clean image $x ^ { \prime } \in { \dot { X } }$ (or, more generally, a fixed dataset and query). Let $( b , k { \bar { ) } }$ denote the environment, where $b \sim \operatorname { U n i f } ( \{ 0 , 1 \} ^ { d } )$ and k is either fixed or drawn independently from a prescribed distribution $\mathcal { P } _ { k }$ . Define

$$
D ^ { ( b , k ) } ( x , x ^ { \prime } ) : = d _ { \mathcal { K } } \big ( z ^ { ( b , k ) } ( x ) , z ^ { ( b , k ) } ( x ^ { \prime } ) \big ) .
$$

The key-annealed (data-quenched) medianfor the $p a i r \left( x , x ^ { \prime } \right)$ is

$$
\mu _ { K } ^ { \ker } ( x , x ^ { \prime } ) : = \operatorname* { i n f } \left\{ m : \mathbb { P } _ { ( b , k ) } \big [ D ^ { ( b , k ) } ( x , x ^ { \prime } ) \leq m \big ] \geq \frac { 1 } { 2 } \right\} .
$$

When a single symbol is used, we write µ<sub>K</sub> for $\mu _ { \mathcal { K } } ^ { \mathrm { k e y } }$ with the convention that the probability is over $( b , k )$ only (the data are heldfixed)

For brevity, we denote the maximal key-annealed median by

$$
\mu _ { \mathcal { K } } ^ { \mathrm { m a x } } ( x ) : = \operatorname* { m a x } _ { 1 \leq i \leq n } \mu _ { \mathcal { K } } ^ { \mathrm { k e y } } ( x , x _ { i } ) .
$$

Theorem 8 (Annealed-over-keys clean K-NN diameter in $\kappa )$ . Assume the prime-gap envelope (Assumption 1). Let the key $b \overset { \cdot } { \sim } \operatorname { U n i f } ( \{ 0 , 1 \} ^ { d } )$ and let the granularity k be either fixed or drawn independently from a prescribed distribution $\mathcal { P } _ { k }$ . Fix a clean query x and afixed dataset $\{ x _ { i } \} _ { i = 1 } ^ { n }$ (data quenched). Define

$$
C _ { \mathrm { m a x } } : = 2 \operatorname* { s u p } _ { k \in \mathrm { s u p p } ( \mathcal { P } _ { k } ) } G _ { k } .
$$

Thenfor any integer $K \geq 2$ and any $\delta _ { \mathrm { e n v } } \in ( 0 , 1 )$ , with probability at least $1 - \delta _ { \mathrm { e n v } }$ over the draw of $( b , k )$

$$
\mathrm { d i a m } \left(  { N _ { K } ^ { \ K } } ( x ; b , k ) \right) \ \le \ 2 A \mu _ { K } ^ { \operatorname* { m a x } } ( x ) \ + \ C _ { \operatorname* { m a x } } \sqrt { d } \left( \sqrt { 2 \log K } + \sqrt { 2 \log ( 2 / \delta _ { \operatorname { e n v } } ) } \right) ,
$$

where $\begin{array} { r } { \mu _ { \mathcal { K } } ^ { \mathrm { m a x } } ( x ) : = \mathrm { m a x } _ { 1 \leq i \leq n } \ \mu _ { \mathcal { K } } ^ { \mathrm { k e y } } ( x , x _ { i } ) } \end{array}$

Proof. Fix the clean dataset $\{ x _ { i } \} _ { i = 1 } ^ { n }$ and the clean query $x ;$ these are held deterministic in this theorem. We first condition on an arbitrary realization of k. All probabilistic statements below are with respect to $\mathbb { P } _ { b \mid k }$ and $\mathbb { E } _ { b \mid k }$ . Since all bounds depend on k only through $G _ { k }$ and hence uniformly through $C _ { \mathrm { m a x } } : = { \mathrm { 2 } } \operatorname* { s u p } _ { k \in \operatorname { s u p p } ( \mathcal { P } _ { k } ) } G _ { k }$ , the resulting bounds hold uniformly in k and therefore also with respect to the joint draw $( b , k )$ . Assume moreover that log ${ \binom { n } { 2 } } = O ( \log K )$ (a standard regime where n is polynomial in $K )$ . For notational simplicity, we continue to write $\mathbb { P } _ { b }$ for $\mathbb { P } _ { b \mid k }$

Fix any pair $( i , j )$ of dataset indices. For simplicity of notation, let $Q _ { i } ^ { b } : = T _ { k } ^ { ( b ) } ( x _ { i } ) \in \mathbb { R } ^ { d }$ . Define the spatial distance under key b as

$$
R _ { i j } ( b ) : = \big \| Q _ { i } ^ { b } - Q _ { j } ^ { b } \big \| _ { 2 } = \Big ( \sum _ { m = 1 } ^ { d } \big ( [ Q _ { i } ^ { b } ] _ { m } - [ Q _ { j } ^ { b } ] _ { m } \big ) ^ { 2 } \Big ) ^ { 1 / 2 } .\tag{20}
$$

Now flip a single key bit $b _ { m } \mapsto b _ { m } ^ { \prime }$ while keeping all other bits fixed. By Assumption 3, the m-th coordinate difference can change by at most $2 G _ { k }$ (a single prime-gap shift per image, hence a $2 G _ { k }$ change for a difference), while all other coordinates remain unchanged:

$$
| [ Q _ { i } ^ { b ^ { \prime } } ] _ { m } - [ Q _ { j } ^ { b ^ { \prime } } ] _ { m } - \left( [ Q _ { i } ^ { b } ] _ { m } - [ Q _ { j } ^ { b } ] _ { m } \right) \mid \ \leq \ 2 G _ { k } , \qquad [ Q _ { i } ^ { b ^ { \prime } } ] _ { \ell } - [ Q _ { j } ^ { b ^ { \prime } } ] _ { \ell } = [ Q _ { i } ^ { b } ] _ { \ell } - [ Q _ { j } ^ { b } ] _ { \ell } \quad ( \ell \neq m ) .
$$

From a vector viewpoint, let

$$
v ( b ) : = Q _ { i } ^ { b } - Q _ { j } ^ { b } \in \mathbb { R } ^ { d } , \qquad v ( b ^ { \prime } ) = v ( b ) + \Delta e _ { m } , \quad | \Delta | \leq 2 G _ { k } ,
$$

where $e _ { m }$ is the m-th standard basis vector. Then the Euclidean norm changes by at most

$$
| \ | v ( b ^ { \prime } ) | | _ { 2 } - | | v ( b ) | | _ { 2 } \ | \ \le \ | | v ( b ^ { \prime } ) - v ( b ) | | _ { 2 } \ = \ | \Delta | \ \le \ 2 G _ { k } \ \le \ C _ { \operatorname* { m a x } } .\tag{21}
$$

Therefore, $R _ { i j } ( b )$ is coordinate-wise $C _ { \mathrm { m a x } } { - } \mathrm { I }$ Lipschitz in each bit $b _ { m }$

Let $b = ( b _ { 1 } , \ldots , b _ { d } ) \in \{ 0 , 1 \} ^ { d }$ be uniformly random with independent bits. By McDiarmid’s inequality,

$$
\mathbb { P } _ { b } \big ( \big | R _ { i j } ( b ) - \mathbb { E } _ { b } [ R _ { i j } ( b ) ] \big | \ge t \big ) \ \le \ 2 \exp \left( - \frac { 2 t ^ { 2 } } { d C _ { \operatorname* { m a x } } ^ { 2 } } \right) .\tag{22}
$$

Write the product embedding as $z ^ { ( b , k ) } ( x ) : = ( T _ { k } ^ { ( b ) } ( x ) , y ) \in \mathcal { K }$ . Define the product distance

$$
D _ { i j } ( b ) : = d _ { K } \big ( z ^ { ( b , k ) } ( x _ { i } ) , z ^ { ( b , k ) } ( x _ { j } ) \big ) = \operatorname* { m a x } \{ R _ { i j } ( b ) , \mathbf { 1 } [ y _ { i } \neq y _ { j } ] \} .\tag{23}
$$

Since the label indicator is deterministic, $\left| D _ { i j } ( b ) - \mathbb { E } _ { b } [ D _ { i j } ( b ) ] \right| \ \leq \ \left| R _ { i j } ( b ) - \mathbb { E } _ { b } [ R _ { i j } ( b ) ] \right|$ . Thus,

$$
\mathbb { P } _ { b } \big ( \big | D _ { i j } ( b ) - \mathbb { E } _ { b } [ D _ { i j } ( b ) ] \big | \ge t \big ) \ \le \ 2 \exp \left( - \frac { 2 t ^ { 2 } } { d C _ { \operatorname* { m a x } } ^ { 2 } } \right) .\tag{24}
$$

Applying a union bound over all  <sup>n</sup> dataset pairs yields

$$
\mathbb { P } _ { b } \bigg ( \operatorname* { m a x } _ { 1 \leq i < j \leq n } \left| D _ { i j } ( b ) - \mathbb { E } _ { b } [ D _ { i j } ( b ) ] \right| > t \bigg ) \ \leq \ \binom { n } { 2 } \cdot 2 \exp \bigg ( - \frac { 2 t ^ { 2 } } { d C _ { \operatorname* { m a x } } ^ { 2 } } \bigg ) .\tag{25}
$$

Fix $\delta _ { \mathrm { e n v } } \in ( 0 , 1 )$ and choose $t : = C _ { \mathrm { m a x } } \sqrt { \textstyle { \frac { d } { 2 } } } \sqrt { \log { \binom { n } { 2 } } + \log ( 2 / \delta _ { \mathrm { e n v } } ) }$ . Then equation 25 implies equation 26 with this t.

If we additionally work in a regime where log ${ \binom { n } { 2 } } \ = \ O ( \log K )$ , the same t satisfies $t \_ <$ $C _ { \mathrm { m a x } } \sqrt { d } \big ( \sqrt { 2 \log K } + \sqrt { 2 \log ( 2 / \delta _ { \mathrm { e n v } } ) } \big )$ after absorbing absolute constants.

With this choice, we obtain the uniform bound

$$
\mathbb { P } _ { b } \Bigg ( \operatorname* { m a x } _ { 1 \leq i < j \leq n } \left. D _ { i j } ( b ) - \mathbb { E } _ { b } [ D _ { i j } ( b ) ] \right. \leq C _ { \operatorname* { m a x } } \sqrt { \frac { d } { 2 } } \sqrt { \log \binom { n } { 2 } + \log ( 2 / \delta _ { \operatorname* { m v } } ) } \Bigg ) \geq 1 - \delta _ { \mathrm { e n v } } .\tag{26}
$$

Since $\mathcal { N } _ { K } ^ { \kappa } ( x ; b , k ) \subseteq \{ x _ { 1 } , . . . , x _ { n } \}$ , the maximum over neighbor pairs is bounded by the maximum over all dataset pairs; hence $\begin{array} { r } { \operatorname* { m a x } _ { u < v \in \mathcal { N } _ { K } } D _ { u v } ( b ) \leq \operatorname* { m a x } _ { i < j } \bar { D } _ { i j } \bar { ( b ) } } \end{array}$

By definition, diam $\begin{array} { r } { \left( \mathcal { N } _ { K } ^ { \mathcal { K } } ( x ; b , k ) \right) = \operatorname* { m a x } _ { u < v } D _ { u v } ( b ) } \end{array}$ . Therefore,

$$
\mathrm { d i a m } \left( \mathcal { N } _ { K } ^ { K } ( x ; b , k ) \right) \ \leq \ \Gamma + C _ { \operatorname* { m a x } } \sqrt { \frac { d } { 2 } } \sqrt { \log \binom { n } { 2 } + \log ( 2 / \delta _ { \mathrm { e n v } } ) } ,\tag{27}
$$

where $\begin{array} { r } { \Gamma : = \operatorname* { m a x } _ { 1 \leq i < j \leq n } \mathbb { E } _ { b \mid k } [ D _ { i j } ( b ) ] } \end{array}$

Let $q = z ^ { ( b , k ) } ( x )$ be the embedded query and let $z _ { ( 1 ) } , \dotsc , z _ { ( K ) }$ be its K nearest neighbors. We now impose the following scale condition (data quenched): there exists an absolute constant $A \geq 1$ such that $\mathrm { m a x } _ { 1 \leq i < j \leq n } \ \bar { \mathbb { E } _ { b | k } } [ D _ { i j } ( b ) ] \ \leq \ 2 A \mu _ { K } ^ { \mathrm { m a \hat { x } } } ( x )$ . Under this condition, $\Gamma \le 2 A \mu _ { \mathcal { K } } ^ { \operatorname* { m a x } } ( x )$

Substituting and simplifying yields, with probability at least $1 - \delta _ { \mathrm { e n v } }$ ，

$$
\begin{array} { r } { \mathrm { d i a m } \left(  { \mathcal { N } } _ { K } ^ { K } ( x ; b , k ) \right) \ \leq \ 2 A \mu _ { K } ^ { \operatorname* { m a x } } ( x ) + C _ { \operatorname* { m a x } } \sqrt { d } \Big ( \sqrt { 2 \log K } + \sqrt { 2 \log ( 2 / \delta _ { \mathrm { e n v } } ) } \Big ) . } \end{array}
$$

This completes the proof.

Corollary 5 (Quenched-in-key clean diameter bound). Under the assumptions ofTheorem 8, there exists a set ofkeys $\mathcal { G } \subseteq \{ 0 , 1 \} ^ { d }$ with $\mathbb { P } _ { b } ( \mathcal { G } ) \geq 1 - \delta _ { \mathrm { e n v } }$ such thatfor every $b \in { \mathcal { G } }$ (and the given $k )$ the bound in Theorem 8 holdsfor thefixed key b and the given clean dataset and query x:

$$
\mathrm { d i a m } \left( \mathcal { N } _ { K } ^ { K } ( x ; b , k ) \right) \leq 2 A \mu _ { K } ^ { \operatorname* { m a x } } ( x ) + C _ { \operatorname* { m a x } } \sqrt { d } \left( \sqrt { 2 \log K } + \sqrt { 2 \log ( 2 / \delta _ { \mathrm { e n v } } ) } \right) .
$$

Proof sketch. The set G is the (key, k)-event on which the union bound in equation 26 holds; this event has probability $\ge 1 - \delta _ { \mathrm { e n v } }$ . On G, the derivation ofTheorem 8 is deterministic, hence the bound is validfor every $b \in { \mathcal { G } }$ (quenched).

Theorem 9 (Adversarial K-NN diameter in K (concise reuse)). Fix $K \geq 2$ and $\delta _ { \mathrm { e n v } } , \delta _ { \mathrm { g r a d } } \in ( 0 , 1 )$ Let $q = ( Q ^ { k , b } ( x ) , y )$ be the clean query and $q _ { \eta } = ( Q ^ { k , b } ( x + \eta ) , \hat { y } )$ the adversarial query (predicted label yˆ may differ from y). Assume the prime-gap sensitivity bound (Assumption 3) at granularity $k ,$ and define $C _ { k } : = 2 G _ { k }$ .

Assume moreover the same data-quenched scale condition as in Theorem 8: there exists an absolute constant $A \geq 1$ such that

$$
\operatorname* { m a x } _ { 1 \leq i < j \leq n } \mathbb { E } _ { b \mid k } [ D _ { i j } ( b ) ] \leq 2 A \mu _ { K } ^ { \operatorname* { m a x } } ( x ) , \qquad \mu _ { K } ^ { \operatorname* { m a x } } ( x ) : = \operatorname* { m a x } _ { 1 \leq i \leq n } \mu _ { K } ^ { \mathrm { k e y } } ( x , x _ { i } ) ,
$$

where $D _ { i j } ( b ) : = d _ { K } ( z ^ { ( b , k ) } ( x _ { i } ) , z ^ { ( b , k ) } ( x _ { j } ) )$ and $z ^ { ( b , k ) } ( x ) : = ( Q ^ { k , b } ( x ) , y ( x ) )$ . The K-nearestneighbor set is stable under the perturbation, $i . e . , N _ { K } ^ { \mathcal { K } } ( q _ { \eta } ) = \mathcal { N } _ { K } ^ { \mathcal { K } } ( q )$ . Finally, assume the varianceonly adversarial displacement control (Assumption 2), i.e., with probability at least $1 - \delta _ { \mathrm { g r a d } }$ $\begin{array} { r } { \| Q ^ { k , b } ( x + \eta ) - Q ^ { k , b } ( x ) \| _ { 2 } \ \leq \ \frac { \sqrt { d } \sigma } { \sqrt { \delta _ { \mathrm { g r a d } } } } \ \| \eta \| _ { \infty } } \end{array}$ . Then, with probability at least $1 - ( \delta _ { \mathrm { e n v } } + \delta _ { \mathrm { g r a d } } )$ over the draw ofthe key b (and k, ifrandom),

$$
\mathrm { d i a m } \left( \mathcal { N } _ { K } ^ { \mathcal { K } } ( q _ { \eta } ) \right) \leq 2 A \mu _ { \mathcal { K } } ^ { \operatorname* { m a x } } ( x ) + C _ { k } \sqrt { d } \Big ( \sqrt { 2 \log K } + \sqrt { 2 \log ( 2 / \delta _ { \mathrm { e n v } } ) } \Big ) + \frac { 2 \sqrt { d } \sigma } { \sqrt { \delta _ { \mathrm { g r a d } } } } \| \eta \| _ { \infty } + 2 \mathbf { 1 } _ { \{ y \neq \bar { y } \} } .
$$

Proof. The proof follows that of Theorem 8 verbatim for the clean case (with clean query x and fixed dataset $\{ x _ { i } \} _ { i = 1 } ^ { n } )$ . In particular, the McDiarmid concentration for pairwise distances under random keys, the union bound over dataset pairs, and the resulting high-probability envelope equation 26 are unchanged. The only modification is that the K-NN set is now formed with respect to the adversarial query $x + \eta$ (instead of the clean query x), and we relate the resulting pairwise distances to the clean anchor via a triangle inequality.

Let

$$
\begin{array} { r } { q : = \big ( Q ^ { k , b } ( \boldsymbol { x } ) , \boldsymbol { y } \big ) \in \mathcal { K } , \qquad q _ { \eta } : = \big ( Q ^ { k , b } ( \boldsymbol { x } + \eta ) , \hat { \boldsymbol { y } } \big ) \in \mathcal { K } , } \end{array}
$$

where $\hat { y }$ is the classifier prediction on $x + \eta$ . Let $z _ { ( 1 ) } , \dotsc , z _ { ( K ) }$ denote the $K$ nearest neighbors of $q _ { \eta }$ in $\kappa$ . For any pair $z _ { ( u ) } , z _ { ( v ) }$ among these neighbors, the triangle inequality gives

$$
d \kappa ( z _ { ( u ) } , z _ { ( v ) } ) \leq d \kappa ( z _ { ( u ) } , q ) + d \kappa ( q , z _ { ( v ) } ) + 2 d \kappa ( q , q _ { \eta } ) .
$$

The first two terms are controlled exactly as in the clean proof (applied with the clean anchor $q )$ , yielding the $2 A \mu _ { \mathcal { K } } ^ { \mathrm { m a x } } ( x )$ term together with the key-fluctuation term $C _ { k } \sqrt { d } \big ( \sqrt { 2 \log K } + \sqrt { 2 \log ( 2 / \delta _ { \mathrm { e n v } } ) } \big )$ For the adversarial displacement term, by the $\ell _ { \infty }$ product metric,

$$
d _ { K } ( q , q _ { \eta } ) \ \leq \ \| Q ^ { k , b } ( x + \eta ) - Q ^ { k , b } ( x ) \| _ { 2 } \ + \ \mathbf { 1 } _ { \{ y \neq \hat { y } \} } .
$$

By Assumption 2 and Chebyshev’s inequality, with probability at least $1 - \delta _ { \mathrm { g r a d } }$

$$
d _ { K } ( q , q _ { \eta } ) \leq \frac { \sqrt { d } \sigma } { \sqrt { \delta _ { \mathrm { g r a d } } } } \| \eta \| _ { \infty } + { \bf 1 } _ { \{ y \neq \hat { y } \} } .
$$

Combining these bounds, and intersecting the clean-key event (probability $\ge 1 - \delta _ { \mathrm { e n v } } )$ with the displacement event (probability $\geq 1 - \delta _ { \mathrm { g r a d } } )$ , yields the stated inequality. □

## E A Brief Primer on Metric Measure Spaces and Gromov–Wasserstein Distances

## E.1 Metric Measure Spaces

A metric measure space (mm-space) is a triple $( X , d _ { X } , \mu )$ , where X is a Polish space, $d _ { X }$ is a metric on $X$ , and $\mu$ is a Borel probability measure on X. Intuitively, an mm-space encodes both the geometry (via $d _ { X } )$ and the distribution of mass (via $\mu )$

Two mm-spaces $( X , d _ { X } , \mu )$ and $( Y , d _ { Y } , \nu )$ are considered equivalent if there exists a measurepreserving isometry $\varphi : X  Y , \mathrm { i . e . } d _ { X } ( x , x ^ { \prime } ) = d _ { Y } ( \varphi ( x ) , \varphi ( \bar { x } ^ { \prime } ) )$ and $\nu = \varphi \# \mu$ . This quotienting ensures that we compare spaces only up to relabeling of points.

The classical notion of distance between mm-spaces is the Gromov–Hausdorff distance, which measures how well two spaces can be embedded into a common metric space with small distortion. However, it is highly combinatorial and not well-suited to data applications.

## E.2 The Gromov–Wasserstein Distance

The Gromov–Wasserstein (GW) distance relaxes Gromov–Hausdorff by using optimal transport ideas. For two mm-spaces $( X , d _ { X } , \mu )$ and $( Y , d _ { Y } , \nu )$ , the squared GW distance is defined as

$$
\mathrm { G W } ^ { 2 } \big ( ( X , d _ { X } , \mu ) , ( Y , d _ { Y } , \nu ) \big ) : = \operatorname* { m i n } _ { \pi \in \Pi ( \mu , \nu ) } \iint | d _ { X } ( x , x ^ { \prime } ) - d _ { Y } ( y , y ^ { \prime } ) | ^ { 2 } d \pi ( x , y ) d \pi ( x ^ { \prime } , y ^ { \prime } ) ,\tag{28}
$$

where $\Pi ( \mu , \nu )$ is the set of couplings with marginals $\mu , \nu .$ Thus GW finds a soft correspondence π between X and Y and penalizes discrepancies between their intra-space distances.

## Properties.

• GW is a metric on the space of mm-spaces up to equivalence.

• If $X = Y$ and $d _ { X } = d _ { Y }$ , then $\mathrm { G W } = 0$ regardless of labeling.

• GW generalizes Wasserstein distance: if $X = Y$ as sets with the same underlying metric, then GW reduces to $W _ { 2 }$

Statistical viewpoint. For empirical datasets $X = \{ x _ { i } \} _ { i = 1 } ^ { n } , Y = \{ y _ { j } \} _ { j = 1 } ^ { m }$ , the metric structure is given by pairwise distance matrices $\left( d _ { X } ( x _ { i } , x _ { i ^ { \prime } } ) \right)$ and $\left( d _ { Y } ( y _ { j } , y _ { j ^ { \prime } } ) \right)$ . The GW distance then becomes a quadratic assignment problem over couplings $\bar { \pi } \in \mathbb { R } ^ { \bar { n } \times m }$ with row/column marginals $1 / n , 1 / m$

## E.3 Entropic Gromov–Wasserstein Distance

The GW optimization in equation 28 is computationally hard due to its quadratic objective. To address this, Peyré et al. (2016) introduced the entropic regularized GW distance, defined as

$$
\mathrm { G W } _ { \gamma } ^ { 2 } ( X , Y ) : = \operatorname* { m i n } _ { \pi \in \Pi ( \mu , \nu ) } \Bigg \{ \iint | d _ { X } ( x , x ^ { \prime } ) - d _ { Y } ( y , y ^ { \prime } ) | ^ { 2 } d \pi ( x , y ) d \pi ( x ^ { \prime } , y ^ { \prime } ) - \gamma H ( \pi ) \Bigg \} ,\tag{29}
$$

where $\begin{array} { r } { H ( \pi ) = - \sum _ { i , j } { \pi } _ { i j } \log { \pi } _ { i j } } \end{array}$ is the Shannon entropy and $\gamma > 0$ is the regularization parameter.

Effects of entropic regularization.

• Computational: The problem becomes smooth and solvable by Sinkhorn-like iterations, scaling to tens of thousands of points.

• Statistical: GW<sub>γ</sub> inherits concentration bounds and enjoys faster empirical convergence (regularization reduces variance).

• Geometric: The optimal coupling π becomes diffuse, capturing probabilistic alignments between X and Y.

Connections to our work. In our setting, the product spaces C (CNN) and K (Crypto) each define mm-spaces under their product metrics and empirical measures. Our clean vs. adversarial concentration bounds on k-NN diameters directly control the intra-space geometry terms in equation 28. Thus, these results serve as building blocks for bounding clean/adversarial GW and entropic GW distances, providing rigorous separation guarantees for detection.

Abuse of notation. Throughout the paper, we use $\mathrm { G W } ^ { 2 } ( X , Y ) , \mathrm { G W } ^ { 2 } ( \mu , \nu )$ , and $\mathrm { G W } ^ { 2 } \big ( ( X , d _ { X } , \mu ) , ( Y , d _ { Y } , \nu ) \big )$ interchangeably when the underlying metric spaces and probability measures are clear from context. In particular, $\operatorname { G W } ^ { 2 } ( X , Y )$ denotes the Gromov–Wasserstein distance between the metric measure spaces induced by X and Y equipped with their respective metrics and empirical measures.

## F Proofs for Gromov–Wasserstein bounds

Definition 11 (Quadratic Gromov–Wasserstein discrepancy). Let $( X , d _ { X } , \mu )$ and $( Y , d _ { Y } , \nu )$ be metric–measure spaces. A coupling $\pi \in \Pi ( \mu , \nu )$ is a probability measure on $X \times Y$ whose marginals are µ and ν, i.e.

$$
\pi ( A \times Y ) = \mu ( A ) , \qquad \pi ( X \times B ) = \nu ( B ) \quad f o r a l l m e a s u r a b l e A \subseteq X , B \subseteq Y .
$$

The quadratic GW discrepancy is defined as

$$
\operatorname { G W } ^ { 2 } { \bigl ( } ( X , d _ { X } , \mu ) , ( Y , d _ { Y } , \nu ) { \bigr ) } : = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \mathbb { E } _ { ( x , y ) \sim \pi } { \Bigl [ } \left( d _ { X } ( x , x ^ { \prime } ) - d _ { Y } ( y , y ^ { \prime } ) \right) ^ { 2 } { \Bigr ] } .
$$

Remark 4 (Upper bound by identity coupling). For brevity, we write $\mathrm { G W } ^ { 2 } ( X , Y )$ as the quadratic GW discrepancy between two metric–measure spaces $( X , d _ { X } , \mu )$ and $( Y , d _ { Y } , \nu )$ . Let $\begin{array} { r } { \bar { \mu } ~ = ~ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { x _ { i } } } \end{array}$ and $\begin{array} { r } { \nu \stackrel { \cdot } { = } \frac { 1 } { n } \textstyle \sum _ { i = 1 } ^ { n } \delta _ { y _ { i } } } \end{array}$ , with distance matrices $D _ { X } [ i , j ] ~ = ~ d _ { X } ( x _ { i } , x _ { j } )$ and $D _ { Y } [ i , j ] = d _ { Y } ( y _ { i } , y _ { j } )$ . Consider the identity coupling $\begin{array} { r } { \pi _ { 0 } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { ( x _ { i } , y _ { i } ) } } \end{array}$ . Since $\mathrm { G W } ^ { 2 }$ is an infimum over couplings, evaluating at any feasible π gives an upper bound:

$$
\operatorname { G W } ^ { 2 } ( X , Y ) \leq \frac { 1 } { n ^ { 2 } } \left\| D _ { X } - D _ { Y } \right\| _ { F } ^ { 2 } .
$$

More generally, for any permutation σ (with permutation matrix $P ) ,$ , the coupling $\pi _ { \sigma } =$ $\begin{array} { r } { \frac { 1 } { n } \sum _ { i } \delta _ { ( x _ { i } , y _ { \sigma ( i ) } ) } } \end{array}$ yields

$$
\mathrm { G W } ^ { 2 } ( X , Y ) \leq \frac { 1 } { n ^ { 2 } } \left\| D _ { X } - P D _ { Y } P ^ { \top } \right\| _ { F } ^ { 2 } .
$$

These bounds are typically loose but serve as alignment-dependent certificates.

Theorem 1 (Clean cross–space GW upper bound via K–NN radii). Fix a clean query x and consider its local neighborhoods $\dot { \mathcal { N } } _ { K } ^ { \mathcal { C } } ( x ) \subset \mathcal { C }$ and $\mathcal { N } _ { K } ^ { \mathcal { K } } ( x ) \subset \mathcal { K }$ , each endowed with the uniform probability measure on K points.

Let $R c : = \operatorname* { m a x } _ { 1 \leq i \leq K } d _ { \mathcal { C } } ( x _ { i } , q _ { \mathcal { C } } )$ amd $R _ { { \sf K } } : = \operatorname* { m a x } _ { 1 \leq i \leq K } d _ { K } ( y _ { i } , q _ { K } )$ denote the clean K–NN radii in C and K, respectively. Assume the clean radius envelopes $R c \leq U _ { \star } ^ { \mathcal { C } }$ and $R \kappa \leq U _ { \star } ^ { \kappa }$ hold with probabilities at least $1 - \delta _ { \scriptstyle { \mathcal { C } } }$ and $1 - \delta \kappa$ , as established in Appendix D.

Then, with probability at least $1 - ( \delta \boldsymbol { c } + \delta \boldsymbol { \kappa } )$ , the quadratic Gromov–Wasserstein discrepancy between the clean neighborhoods satisfies

$$
\begin{array} { r } { \mathrm { G W } ^ { 2 } \Big ( \mathcal { N } _ { K } ^ { \mathcal { C } } ( x ) , \mathcal { N } _ { K } ^ { K } ( x ) \Big ) \ \leq \ 4 \Big ( 1 - \frac { 1 } { K } \Big ) \left( U _ { \star } ^ { \mathcal { C } } + U _ { \star } ^ { K } \right) ^ { 2 } . } \end{array}
$$

Proof. Let $X : = \mathcal { N } _ { K } ^ { \mathcal { C } } ( x ) = \{ x _ { 1 } , \ldots , x _ { K } \}$ and $Y : = \mathcal { N } _ { K } ^ { K } ( x ) = \{ y _ { 1 } , . . . , y _ { K } \}$ be the K neighbors in C and $\kappa .$ respectively, both with the uniform measure $K ^ { - 1 } \sum _ { i = 1 } ^ { K } \delta _ { ( \cdot ) }$ . We denote their distance matrices as $D c [ i , j ] : = d c ( x _ { i } , x _ { j } )$ and $D \kappa [ i , j ] : = d _ { K } ( y _ { i } , y _ { j } )$ , which satisfy $D c [ i , i ] = D \kappa [ i , i ] = 0$ $\boldsymbol { \mathrm { B y } }$ Remark 4, evaluating the quadratic GW objective at the identity coupling $\begin{array} { r } { \pi _ { 0 } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \delta _ { ( x _ { i } , y _ { i } ) } } \end{array}$ yields

$$
\operatorname { G W } ^ { 2 } ( X , Y ) \leq \frac { 1 } { K ^ { 2 } } \left\| D c - D \kappa \right\| _ { F } ^ { 2 } .\tag{30}
$$

Let $q _ { \mathcal { C } }$ be the clean query center in C and $q \kappa$ the center in $\kappa .$ Define the K–NN radii as $R c : =$ I $\scriptstyle \operatorname { n a x } _ { 1 \leq i \leq K } d _ { \mathcal { C } } ( x _ { i } , q _ { \mathcal { C } } )$ and $R _ { { \sf K } } : = \operatorname* { m a x } _ { 1 \leq i \leq { K } } d _ { K } ( y _ { i } , q _ { K } )$ . By the triangle inequality in each product metric, we have

$$
d _ { \mathcal { C } } ( x _ { i } , x _ { j } ) \leq d _ { \mathcal { C } } ( x _ { i } , q _ { \mathcal { C } } ) + d _ { \mathcal { C } } ( q _ { \mathcal { C } } , x _ { j } ) \leq 2 R _ { \mathcal { C } } , d _ { K } ( y _ { i } , y _ { j } ) \leq 2 R _ { K } , ( i \neq j ) .\tag{31}
$$

Hence every off-diagonal entry of ${ \cal D } c \left( \mathrm { r e s p . } D _ { K } \right)$ is bounded by $2 R c \ ( \mathrm { r e s p . } 2 R \kappa )$

There are exactly $K ( K - 1 )$ off-diagonal entries. Using equation 31 we arrive at

$$
\| D _ { \cal C } \| _ { { \cal F } } ^ { 2 } \leq K ( K - 1 ) ( 2 R _ { \cal C } ) ^ { 2 } , \qquad \| D _ { \cal K } \| _ { { \cal F } } ^ { 2 } \leq K ( K - 1 ) ( 2 R _ { K } ) ^ { 2 } .
$$

Therefore, by the triangle inequality for $\| \cdot \| _ { F }$

$$
\begin{array} { r l r } { \| D c - D \kappa \| _ { F } } & { \leq } & { \| D c \| _ { F } + \| D \kappa \| _ { F } } \\ & { } & { \leq 2 \sqrt { K ( K - 1 ) } ( R c + R \kappa ) . } \end{array}\tag{32}
$$

Substituting equation 32 into equation 30 yields

$$
\mathrm { G W } ^ { 2 } ( X , Y ) \leq \frac { 1 } { K ^ { 2 } } \Big ( 2 \sqrt { K ( K - 1 ) } ( R _ { \mathcal { C } } + R _ { \mathcal { K } } ) \Big ) ^ { 2 } = 4 \Big ( 1 - \frac { 1 } { K } \Big ) ( R _ { \mathcal { C } } + R _ { \mathcal { K } } ) ^ { 2 } .
$$

On the event $E : = \{ R c \leq U _ { \star } ^ { c } \} \cap \{ R \kappa \leq U _ { \star } ^ { \kappa } \}$ we have $\smash { ( R c + R _ { K } ) ^ { 2 } \leq \bigl ( U _ { \star } ^ { \mathscr { C } } + U _ { \star } ^ { \mathscr { K } } \bigr ) ^ { 2 } }$ , and therefore $\mathrm { G W } ^ { 2 } ( X , Y ) \ \leq \ 4 \Big ( 1 - { \textstyle { \frac { 1 } { K } } } \Big ) \left( U _ { \star } ^ { \mathcal { C } } + U _ { \star } ^ { K } \right) ^ { 2 }$ . Finally, by the union bound, $\mathrm { P r } ( E ) \geq 1 - ( \delta c + \delta \kappa )$ , which concludes the proof. □

Notation (adversarial queries, radii, and separation gap). Let x be a clean image and η a perturbation. For each space $\mathsf { M } \in \{ \mathcal { C } , \dot { \mathcal { K } } \}$ , define the adversarial query as $\widetilde q _ { \mathsf { M } } : = z ^ { \mathsf { M } } ( \overline { { x } } + \eta )$ , the embedding of the perturbed image x + η into M. As the dataset is fixed and only the query moves, so the K–NN neighborhood may change membership relative to the clean case. We define the adversarial neighborhood as $\widetilde { \mathcal { N } } _ { K } ^ { \mathsf { M } } ( x + \eta ) : = \{ z _ { 1 } ^ { \mathsf { M } } , \dots , z _ { K } ^ { \mathsf { M } } \}$ , which are the K nearest neighbors to q˜<sub>M</sub> under $d _ { \mathsf { M } }$ . The associated adversarial radii are $r _ { i } ^ { \mathsf { M } } : = d _ { \mathsf { M } } ( z _ { i } ^ { \mathsf { M } } , \widetilde { q } _ { \mathsf { M } } )$ , and $R _ { \mathsf { M } } ^ { \mathrm { a d v } } : = \operatorname* { m a x } _ { 1 \leq i \leq K } r _ { i } ^ { \mathsf { M } }$

Adversarial shifts often produce a “cluster split” in these radii, i.e., some neighbors become unusually close to $\tilde { q } _ { \cal M }$ , while others remain farther away. To capture this structure, we partition $\widetilde { \mathcal { N } } _ { K } ^ { \mathsf { M } } ( x + \eta )$ into an inner group L and an outer group H of sizes $( 1 - \theta ) K$ and θK by thresholding $\{ r _ { i } ^ { \mathsf { M } } \}$ . The adversarial separation ${ g a p }$ in M is

$$
\gamma _ { \mathsf { M } } : = \operatorname* { m i n } _ { i \in H } r _ { i } ^ { \mathsf { M } } - \operatorname* { m a x } _ { j \in L } r _ { j } ^ { \mathsf { M } } ,
$$

which is positive when the inner and outer sets are well separated.

Theorem 2 (Adversarial cross–space GW lower bound). Fix a query x and perturbation $\eta ,$ and consider the adversarial neighborhoods $\widetilde { \mathcal { N } } _ { K } ^ { \mathcal { C } } ( x + \eta )$ and $\widetilde { \mathcal { N } } _ { K } ^ { \mathcal { K } } ( x + \eta )$ , each with uniform measure on K points. Let $\gamma _ { \mathcal { C } }$ be the separation gap and let diam $\left( \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { K } } ( x + \eta ) \right)$ denote the adversarial $K { - } N \bar { N }$ diameter in $\mid \mathcal { K } ,$ bounded as in Appendix Theorem 9. Define the adversarial K–diameter $D _ { \mathcal { K } } ^ { \mathrm { a d v } } : = \mathrm { d i a m } \big ( \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { K } } ( x + \eta ) \big )$ . Then, with probability at least $1 - \delta _ { K } ^ { \mathrm { e n v } }$

$$
\mathrm { G W } ^ { 2 } \Big ( \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { C } } ( x + \eta ) , \widetilde { \mathcal { N } } _ { K } ^ { K } ( x + \eta ) \Big ) \ \geq \ 2 \theta ( 1 - \theta ) \left( \gamma _ { \mathcal { C } } - D _ { K } ^ { \mathrm { a d v } } \right) _ { + } ^ { 2 } .
$$

Proof. In any metric space $( \mathsf { M } , d )$ with center $q ,$ one has for all $u , v ,$

$$
| d ( u , q ) - d ( v , q ) | \ \leq \ d ( u , v ) \ \leq \ d ( u , q ) + d ( v , q ) .
$$

We apply this to the adversarial neighborhoods. For the $\mathcal { C } \mathrm { - s p a c e }$ , each $x _ { i } , x _ { j } \in \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { C } } ( x + \eta )$ has radii $\boldsymbol { r } _ { i } ^ { \mathcal { C } } = d _ { \cal C } ( x _ { i } , \tilde { q } _ { \mathcal { C } } )$ , while for the K-space, each $y _ { a } , y _ { b } \in \widetilde { \mathcal { N } } _ { K } ^ { \kappa } ( x + \eta )$ has radii $r _ { a } ^ { \kappa } = d _ { \kappa } ( y _ { a } , \tilde { q } \kappa )$

$$
d _ { \mathcal { C } } ( x _ { i } , x _ { j } ) \geq | r _ { i } ^ { \mathcal { C } } - r _ { j } ^ { \mathcal { C } } | , \qquad d _ { K } ( y _ { a } , y _ { b } ) \leq \mathrm { d i a m } \big ( \widetilde { \mathcal { N } } _ { K } ^ { K } ( x + \eta ) \big ) .\tag{33}
$$

We now identify which neighbor pairs give us a guaranteed discrepancy. Recall that the adversarial neighborhood in C is partitioned into an inner set L of size $( 1 - \theta ) \bar { K }$ and an outer set $H$ of size $\theta K$ with separation gap $\begin{array} { r } { \dot { \gamma _ { \mathcal { C } } } : = \operatorname* { m i n } _ { i \in H } r _ { i } ^ { \mathcal { C } } - \operatorname* { m a x } _ { j \in L } r _ { j } ^ { \mathcal { C } } > 0 } \end{array}$

A natural question arises, namely, Whyfocus on cross pairs? If both indices come from H (outer– outer) or both from L (inner–inner), the corresponding radii may be very close, and no nontrivial separation is guaranteed. However, whenever one index $i \in \mathcal { \bar { H } }$ and the other $j \in L$ , we know $r _ { i } ^ { \bar { \mathcal { C } } } - r _ { j } ^ { \mathcal { C } } ~ \ge ~ \gamma _ { \mathcal { C } }$

By the triangle inequality bound equation 33,

$$
d _ { \mathcal { C } } ( \boldsymbol { x } _ { i } , \boldsymbol { x } _ { j } ) \geq | r _ { i } ^ { \mathcal { C } } - r _ { j } ^ { \mathcal { C } } | \geq \gamma _ { \mathcal { C } } , \qquad d _ { K } ( y _ { a } , y _ { b } ) \leq \mathrm { d i a m } \big ( \widetilde { \mathcal { N } } _ { K } ^ { K } ( \boldsymbol { x } + \boldsymbol { \eta } ) \big ) .
$$

Therefore, for any cross pair $( i , j ) \in H \times L \mathrm { o r } ( j , i ) \in L \times H$ , and for all choices of $( a , b )$ in $\kappa ,$

$$
\big | d _ { \mathcal { C } } ( x _ { i } , x _ { j } ) - d _ { K } ( y _ { a } , y _ { b } ) \big | \ge \gamma _ { \mathcal { C } } - \mathrm { d i a m } \big ( \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { K } } ( x + \eta ) \big ) .\tag{34}
$$

How many such pairs exist? Define $S : = ( H \times L ) \cup ( L \times H ) \subset \{ 1 , \dots , K \} ^ { 2 }$ . Then

$$
| S | = | H | | L | + | L | | H | = 2 | H | | L | = 2 \left( \theta K \right) \left( ( 1 - \theta ) K \right) = 2 \theta ( 1 - \theta ) K ^ { 2 } .
$$

Thus, exactly $2 \theta ( 1 - \theta ) K ^ { 2 }$ ordered pairs $( i , j )$ enjoy the guaranteed discrepancy equation 34, which will next drive our GW lower bound.

For any coupling $\pi \in \Pi ( \tilde { \mu } , \tilde { \nu } )$ with uniform marginals, the GW objective can be written as

$$
\mathrm { G W } ^ { 2 } ( \widetilde { \mathcal { C } } , \widetilde { K } ) = \operatorname* { i n f } _ { \pi } \sum _ { i , i ^ { \prime } } \sum _ { j , j ^ { \prime } } \pi [ i , j ] \pi [ i ^ { \prime } , j ^ { \prime } ] \Big ( d _ { \mathcal { C } } ( x _ { i } , x _ { i ^ { \prime } } ) - d _ { \mathcal { K } } ( y _ { j } , y _ { j ^ { \prime } } ) \Big ) ^ { 2 } .
$$

Fix any ordered cross pair $( i , i ^ { \prime } ) \in S$ . By equation 34, for every pair $( j , j ^ { \prime } )$ in $\widetilde { \mathcal { N } } _ { K } ^ { \mathcal { K } } ( x + \eta )$ we have $\left| ~ d _ { \mathcal { C } } ( x _ { i } , x _ { i ^ { \prime } } ) - d _ { K } ( y _ { j } , y _ { j ^ { \prime } } ) ~ \right| ~ \geq ~ \gamma c -$ diam $\left( \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { K } } ( x + \eta ) \right)$ , where we used that $d _ { \mathcal { K } } ( y _ { j } , y _ { j ^ { \prime } } ) \leq$ diam $\left( \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { K } } ( x + \eta ) \right)$ holds uniformly for all $( j , j ^ { \prime } )$ . Squaring both sides yields

$$
\Big ( d _ { \mathcal { C } } \big ( x _ { i } , x _ { i ^ { \prime } } \big ) - d _ { K } \big ( y _ { j } , y _ { j ^ { \prime } } \big ) \Big ) ^ { 2 } \ \geq \ \Big ( \gamma _ { \mathcal { C } } - \mathrm { d i a m } \big ( \widetilde { \mathcal { N } } _ { K } ^ { K } ( x + \eta ) \big ) \Big ) ^ { 2 } .
$$

Since the coupling π has uniform marginals, the total weight assigned to the block $\{ i \} \times \{ i ^ { \prime } \}$ after summing over $j , \breve { j ^ { \prime } }$ is fixed:

$$
\sum _ { j , j ^ { \prime } } \pi [ i , j ] \pi [ i ^ { \prime } , j ^ { \prime } ] = \Big ( \sum _ { j } \pi [ i , j ] \Big ) \Big ( \sum _ { j ^ { \prime } } \pi [ i ^ { \prime } , j ^ { \prime } ] \Big ) = \frac { 1 } { K } \cdot \frac { 1 } { K } = \frac { 1 } { K ^ { 2 } } .
$$

Therefore for each $\begin{array} { r l r } { ( i , i ^ { \prime } ) \in S , \sum _ { j , j ^ { \prime } } \pi [ i , j ] \pi [ i ^ { \prime } , j ^ { \prime } ] ( \cdots ) } & { \geq } & { \frac { 1 } { K ^ { 2 } } \left( \gamma _ { \mathcal { C } } - \dim \bigl ( \widetilde { \mathcal { N } } _ { K } ^ { K } ( x + \eta ) \bigr ) \right) _ { + } ^ { 2 } . } \end{array}$ Summing over all $( i , i ^ { \prime } ) \in S$ gives

$$
\mathbb { E } _ { \pi \otimes \pi } \Big [ \big ( d _ { \mathcal { C } } ( x , x ^ { \prime } ) - d _ { K } ( y , y ^ { \prime } ) \big ) ^ { 2 } \Big ] \ \geq \ \frac { | S | } { K ^ { 2 } } \Big ( \gamma _ { \mathcal { C } } - \mathrm { d i a m } \big ( \widetilde { \mathcal { N } } _ { K } ^ { K } ( x + \eta ) \big ) \Big ) _ { + } ^ { 2 } ,
$$

because contributions from pairs outside S are nonnegative and can be dropped. Finally, recalling $\begin{array} { r } { | S | = 2 \theta ( 1 - \theta ) K ^ { 2 } } \end{array}$ , we arrive at ${ \mathbb E } _ { \pi \otimes \pi } \Big [ \big ( d _ { \mathcal { C } } ( x , x ^ { \prime } ) - d _ { K } ( y , y ^ { \prime } ) \big ) ^ { 2 } \Big ] \ \ge \ 2 \theta ( 1 - \theta ) \left( \gamma _ { \mathcal { C } } - \mathrm { d i a m } \big ( \widetilde { \mathcal { N } } _ { K } ^ { K } ( x + x ^ { \prime } ) - 2 \widetilde { \mathcal { N } } _ { K } ^ { K } ( y , y ^ { \prime } ) \big ) ^ { 2 } \right) \ .$

$\eta ) \big ) \bigg _ { + } ^ { 2 }$ . As this bound is independent of the choice of coupling, it continues to hold after taking the

infimum over π. Finally, substituting the high–probability envelope for diam $\left( \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { K } } ( x + \eta ) \right)$  from Appendix Theorem 9 yields the explicit form of the bound, which completes the proof. □

Remark 5 (Instantiating $\gamma _ { \mathcal { C } }$ from clean geometry). The abstract separation parameter γ in Theorem 2 can be linked to earlier C results. Let q and q˜ be the clean and adversarial queries, with the clean K–NN neighborhood partitioned into inner set L and outer set H. The clean separation gap is

$$
\Delta _ { \mathrm { c l e a n } } : = \operatorname* { m i n } _ { i \in { \cal H } } d _ { { \mathcal { C } } } ( x _ { i } , q ) { \mathbf \alpha } - \operatorname* { m a x } _ { j \in { \cal L } } d _ { { \mathcal { C } } } ( x _ { j } , q ) .
$$

By our earlier adversarial absorption radius analysis in ${ \mathcal { C } } ,$ adversarial perturbations shift each star distance by at most $\frac { \sqrt { d } \sigma } { \sqrt { \delta _ { \mathrm { g r a d } } } } \varepsilon + \mathbf { 1 } _ { \{ c \neq \hat { c } \} }$ . Hence the induced adversarial gap satisfies

$$
\begin{array} { r } { \gamma c \geq \Big ( \Delta _ { \mathrm { c l e a n } } - 2 \Big ( \frac { \sqrt { d } \sigma } { \sqrt { \delta _ { \mathrm { g r a d } } } } \varepsilon + \mathbf { 1 } _ { \{ y \neq \hat { y } \} } \Big ) \Big ) _ { + } . } \end{array}
$$

Thus $\gamma _ { \mathcal { C } }$ is not an arbitrary constant: it can be certified from clean geometry plus the perturbation shift and possible label flip.

Corollary 6 (Explicit clean–adversarial gap across C and K). Combining Theorem 2 with Remark 5, with probability at least $1 - ( \delta _ { \mathcal { K } } ^ { \mathrm { e n v } } + \delta _ { \mathrm { g r a d } } )$

$$
\begin{array} { r } { \mathrm { G W } ^ { 2 } \Big ( \tilde { \mathcal { N } } _ { K } ^ { \sigma } ( x + \eta ) , \tilde { \mathcal { N } } _ { K } ^ { K } ( x + \eta ) \Big ) \geq 2 \theta ( 1 - \theta ) \left( \Delta _ { \mathrm { c l e a n } } - 2 \Big ( \frac { \sqrt { d } \sigma } { \sqrt { \tilde { \delta } _ { \operatorname* { g r a d } } } } \varepsilon + \mathbf { 1 } _ { \{ y \neq \tilde { y } \} } \Big ) - \dim ( \tilde { \mathcal { N } } _ { K } ^ { K } ( x + \eta ) ) \right) _ { + } ^ { 2 } . } \end{array}
$$

This bound separates the clean structure, the C–side adversarial shift, and the K–side absorption, making the cross–space adversarial gap explicit

## F.1 Mirror theorems for upper and lower bounds on GW

Theorem 10 (Clean cross–space GW lower bound). Fix a clean query x and its clean neighborhoods $\mathcal { N } _ { K } ^ { \mathcal { C } } ( x )$ and $\dot { \mathcal { N } } _ { K } ^ { \mathcal { K } } ( x )$ , each with uniform measure on K points. Suppose the clean C–radii around the clean center q<sub>C</sub> exhibit a separation gap

$$
\gamma _ { \mathcal { C } } ^ { \mathrm { c l e a n } } : = \operatorname* { m i n } _ { i \in H } d _ { \mathcal { C } } ( x _ { i } , q _ { \mathcal { C } } ) - \operatorname* { m a x } _ { j \in L } d _ { \mathcal { C } } ( x _ { j } , q _ { \mathcal { C } } ) > 0 ,
$$

for a partition into inner L and outer H of sizes $( 1 - \theta ) K$ and θK. Let $R _ { K } ^ { \mathrm { c l e a n } }$ denote the clean K–NN radius in $\kappa ,$ bounded by the clean K envelope. Then

$$
\mathrm { G W } ^ { 2 } \Big ( \mathcal { N } _ { K } ^ { \mathcal { C } } ( x ) , \mathcal { N } _ { K } ^ { K } ( x ) \Big ) \ \geq \ 2 \theta ( 1 - \theta ) \Big ( \gamma _ { \mathcal { C } } ^ { \mathrm { c l e a n } } - 2 R _ { K } ^ { \mathrm { c l e a n } } \Big ) ^ { 2 } .
$$

Proof sketch. Identical to Theorem 2 (Steps 1–4) with “adversarial” replaced by “clean” and $R _ { K } ^ { \mathrm { a d v } }$ replaced by $R _ { \mathcal { K } } ^ { \mathrm { c l e a n } }$ . The cross-pairs $( \mathsf { \tilde { H } } \times L ) \cup ( L \times H )$ enforce an entrywise gap ofat least $\gamma _ { c } ^ { \mathrm { c l e a n } } - 2 R _ { \mathcal { K } } ^ { \mathrm { c l e a n } }$ ; uniform marginals then yield the factor $2 \theta ( 1 - \theta )$ after averaging over couplings. Theorem 11 (Adversarial cross–space GW upper bound via K-NN star radii). Fix a query x and perturbation η, and consider the adversarial neighborhoods

$$
\begin{array} { r } { \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { C } } ( \boldsymbol { x } + \boldsymbol { \eta } ) , \qquad \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { K } } ( \boldsymbol { x } + \boldsymbol { \eta } ) , } \end{array}
$$

each with uniform measure on K points. Let $R _ { C } ^ { \mathrm { a d v } }$ and $R _ { K } ^ { \mathrm { a d v } }$ denote the adversarial K–NN radii in C and K, respectively (each bounded by the adversarial envelopes in those spaces). Then

$$
\begin{array} { r } { \mathrm { G W } ^ { 2 } \Big ( \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { C } } ( x + \eta ) , \widetilde { \mathcal { N } } _ { K } ^ { K } ( x + \eta ) \Big ) \ \leq \ 4 \Big ( 1 - \frac { 1 } { K } \Big ) \Big ( R _ { \mathcal { C } } ^ { \mathrm { a d v } } + R _ { K } ^ { \mathrm { a d v } } \Big ) ^ { 2 } . } \end{array}
$$

Proof sketch. Copy Theorem 1 verbatim, replacing clean radii by adversarial radii. The identity coupling and Frobenius argument give $\begin{array} { r } { \mathrm { G W } ^ { 2 } \leq K ^ { - 2 } \| D c - D \kappa \| _ { F } ^ { 2 } } \end{array}$ . Each off-diagonal entry is at most $2 R ^ { \mathrm { a d v } }$ by triangle inequality via the adversarial centers, so $\| D c - D \kappa \| _ { F } \leq 2 \sqrt { K ( K - 1 ) } ( R _ { \mathcal { C } } ^ { \mathrm { a d v } } +$ $R _ { K } ^ { \mathrm { a d v } } )$ , which yields the claim.

## F.2 Gap theorem in GW

We now combine the clean and adversarial envelopes established in Theorems 1, 2, 10, and 11 to show that their GW discrepancies are separated by a margin that is nonvanishing in high dimension.

Theorem 3 (Cross–space GW gap). With probability at least $1 - ( \delta c + \delta \kappa + \delta _ { K } ^ { \mathrm { e n v } } + \delta _ { \mathrm { a u x } } )$ , the clean and adversarial GW discrepancies satisfy $| \mathrm { G W _ { a d v } ^ { 2 } - G W _ { c l e a n } ^ { 2 } } | \geq \tau : = \operatorname* { m a x } \{ \tau _ { \mathrm { a d v } } , \tau _ { \mathrm { c l e a n } } , 0 \}$ where $\tau _ { \mathrm { a d v } } = L _ { \mathrm { a d v } } - U _ { \mathrm { c l e a n } }$ and $\tau _ { \mathrm { c l e a n } } = L _ { \mathrm { c l e a n } } - U _ { \mathrm { a d v } } .$ Under Assumption 2, for fixed K and perturbation $\| \eta \| _ { \infty } = \varepsilon ,$ , we obtain $\tau = \Omega ( d \sigma ^ { 2 } \varepsilon ^ { 2 } ) \ - \ O \big ( ( R _ { K } ^ { \mathrm { c l e a n } } ) ^ { 2 } \big )$

Proof. On the joint event where all four bounds hold:

$$
\mathrm { G W _ { a d v } ^ { 2 } } \geq L _ { \mathrm { a d v } } , \mathrm { G W _ { c l e a n } ^ { 2 } } \leq U _ { \mathrm { c l e a n } } ,
$$

which implies

$$
\mathrm { G W } _ { \mathrm { a d v } } ^ { 2 } - \mathrm { G W } _ { \mathrm { c l e a n } } ^ { 2 } \ \geq \ L _ { \mathrm { a d v } } - U _ { \mathrm { c l e a n } } .
$$

Similarly,

$$
\mathrm { G W } _ { \mathrm { c l e a n } } ^ { 2 } \ \geq \ L _ { \mathrm { c l e a n } } , \qquad \mathrm { G W } _ { \mathrm { a d v } } ^ { 2 } \ \leq \ U _ { \mathrm { a d v } } ,
$$

which implies

$$
\mathrm { G W } _ { \mathrm { c l e a n } } ^ { 2 } - \mathrm { G W } _ { \mathrm { a d v } } ^ { 2 } \ \geq \ L _ { \mathrm { c l e a n } } - U _ { \mathrm { a d v } } .
$$

Taking the maximum of these two margins yields the result.

Discussion. Clean side. For fixed $K ,$ , as $d \to \infty$ , the clean K–NN radii in $\mathcal { C }$ concentrate according to Theorem 6, yielding

$$
\begin{array} { r } { R _ { \small \mathscr { C } \mathrm { ~ = ~ } } O \Bigl ( \mu _ { \small \mathscr { C } } + \sqrt { \frac { \log K } { d } } \Bigr ) , } \end{array}
$$

so the excess clean radius above the median scale contracts at rate $O ( \sqrt { \frac { \log K } { d } } )$ . On the K side, the clean envelope from Theorem 8 includes key–induced fluctuations scaling as $C _ { \mathrm { m a x } } \sqrt { d }$ (up to $\sqrt { \log K }$ factors), and therefore $R _ { K } ^ { \mathrm { c l e a n } }$ need not vanish with dimension under the present assumptions. Accordingly, the clean cross–space GW upper bound remains uniformly controlled but does not rely on contraction of K radii.

Adversarial side. A perturbation $\| \eta \| _ { \infty } = \varepsilon$ shifts the K–NN star distances in $\mathcal { C }$ by

$$
\begin{array} { r } { \delta c ( \varepsilon ) = \Theta \Big ( \frac { \sqrt { d } } { \sqrt { \delta _ { \mathrm { g r a d } } } } \sigma \varepsilon \Big ) , } \end{array}
$$

which induces a separation gap $\gamma _ { \mathcal { C } }$ between inner and outer neighbors. Meanwhile, absorption in K is controlled by the adversarial envelope $R _ { K } ^ { \mathrm { a d v } } = O ( C _ { k } \sqrt { d } )$ , as established in Theorem 9. Let $L _ { \mathrm { a d v } }$ denote the lower bound in Theorem 2. Substituting the bounds on $\gamma _ { \mathcal { C } }$ from Remark 5 and on diam $( \widetilde { \mathcal { N } } _ { K } ^ { \mathcal { K } } )$ from Theorem 9 yields

$$
L _ { \mathrm { a d v } } = \Omega ( d \sigma ^ { 2 } \varepsilon ^ { 2 } ) \quad ( \mathrm { u p t o c o n s t a n t s a n d c o n f i d e n c e p a r a m e t e r s } ) .
$$

Gap scaling. Combining the clean upper envelope with the adversarial lower envelope, we obtain a margin of the form

$$
\tau = \Omega \big ( d \sigma ^ { 2 } \varepsilon ^ { 2 } \big ) - O \big ( R _ { K } ^ { \mathrm { c l e a n 2 } } \big ) ,
$$

where the second term reflects the clean K–side geometry. In particular, even though $R _ { K } ^ { \mathrm { c l e a n } }$ does not contract with $d ,$ the adversarial contribution grows linearly in dimension, so the gap is asymptotically nonvanishing whenever $l \sigma ^ { 2 } \varepsilon ^ { 2 } \gg R _ { K } ^ { \mathrm { c l e a n 2 } }$ . Thus, robustness in high dimension is achieved through separation growth on the C side rather than contraction on the K side.

Implication for entropic solvers. Because the GW gap remains bounded away from zero in this regime, entropic relaxations that preserve relative ordering inherit the same discriminative power, justifying our detector design.

## F.3 Entropic GW Corollaries and Risk Control

Corollary 7 (Entropic relaxation preserves lower bounds). For any metric–measure spaces $( X , d _ { X } , \mu )$ and $( Y , \dot { d } _ { Y } , \nu )$ and any $\lambda > 0$

$$
\mathrm { G W } _ { \lambda } ^ { 2 } ( X , Y ) \ge \mathrm { G W } ^ { 2 } ( X , Y ) .
$$

Hence the lower bounds of Theorems 2 and 10 remain valid verbatim under entropic GW.

Corollary 8 (Entropic slack in upper bounds). For uniform marginals on K points,

$$
\mathrm { G W } _ { \lambda } ^ { 2 } ( X , Y ) \le \mathrm { G W } ^ { 2 } ( X , Y ) + 2 \lambda \log K .
$$

Thus the upper bounds of Theorems 1 and 11 hold with additive slack 2λ log $K .$

Corollary 9 (Quenched key version). All high–probability envelopes on the K–side radii (Theorems 8, 2, 11) were stated in the annealed sense, averaging over random keys $( b , k )$ . By conditioning, the same inequalities holdfor anyfixed $( b , k )$ with identical probability bounds over the randomness of image sampling and adversarial perturbations.

Risk control The GW gap theorem (Theorem 3) ensures that, with high probability, the clean and adversarial discrepancies $\mathrm { G W } _ { \mathrm { c l e a n } } ^ { 2 }$ and $\mathrm { G W _ { a d v } ^ { 2 } }$ are separated by a margin $\tau > 0$ . In practice, however, we only observe the empirical, regularized estimator $\widehat { \mathrm { G W _ { \lambda } ^ { 2 } } } .$ which deviates from the truth due to (i) statistical sampling noise and (ii) entropic bias. If these deviations exceed $\tau ,$ , the detector may fail. The next lemma formalizes that controlling the estimation error to $\tau / 3$ suffices.

Lemma 4 (Risk control via GW margin). If the gap event holds with margin $\tau > 0$ and an estimator $\widehat { \mathrm { G W } _ { \lambda } ^ { 2 } }$ satisfies $\mathrm { P r } \big ( \vert \widehat { \mathrm { G W } _ { \lambda } ^ { 2 } } - \mathrm { G W } ^ { 2 } \vert \le \tau / 3 \big ) \ge 1 - \delta _ { \mathrm { e s t } }$ , then thresholding $\overline { { \mathrm { G W } _ { \lambda } ^ { 2 } } }$ at the midpoint between clean and adversarial envelopes makes no error on this event. Thus Pr(misclassification) $\leq$ $\mathrm { P r } ( E _ { \mathrm { g a p } } ^ { c } ) + \delta _ { \mathrm { e s t } }$

Proof. On the event $E _ { \mathrm { g a p } }$ , the clean and adversarial discrepancies satisfy conditions $\mathrm { { G W _ { c l e a n } ^ { 2 } } \leq }$ $U _ { \mathrm { c l e a n } }$ and $\mathrm { G W } _ { \mathrm { a d v } } ^ { 2 } ~ \geq ~ L _ { \mathrm { a d v } } ,$ , with $L _ { \mathrm { a d v } } - U _ { \mathrm { c l e a n } } \ge \tau _ { \mathrm { a d v } } .$ , and symmetrically $\mathrm { G W } _ { \mathrm { c l e a n } } ^ { 2 } \ \geq \ L$ clean and $\mathrm { G W } _ { \mathrm { a d v } } ^ { 2 } ~ \leq ~ U _ { \mathrm { a d v } } ,$ , with $L _ { \mathrm { c l e a n } } - U _ { \mathrm { a d v } } \geq \tau _ { \mathrm { c l e a n } }$ . By definition, $\tau = \operatorname* { m a x } \{ \tau _ { \mathrm { a d v } } , \tau _ { \mathrm { c l e a n } } \} > 0 .$ , so there exists a threshold t<sup>∗</sup> lying strictly between the clean and adversarial ranges, with a buffer of at least $\frac { \tau } { 2 }$ to each side.

In the case of a clean instance, on $E _ { \mathrm { e s t } }$ we have that $\widehat { \mathrm { G W _ { \lambda } ^ { 2 } } } \le \mathrm { G W } _ { \mathrm { c l e a n } } ^ { 2 } + \textstyle { \frac { \tau } { 3 } } \le U _ { \mathrm { c l e a n } } + \textstyle { \frac { \tau } { 3 } } ,$ . Since $\begin{array} { r } { t ^ { * } \geq U _ { \mathrm { c l e a n } } + \frac { \tau } { 2 } } \end{array}$ , we conclude $\begin{array} { r } { \widehat { \mathrm { G W } _ { \lambda } ^ { 2 } } \ \leq \ t ^ { * } - \frac { \tau } { 6 } \ < \ t ^ { * } } \end{array}$ , so the classifier correctly outputs “clean.”

For an adversarial instance, on $E _ { \mathrm { e s t } }$ , we have $\begin{array} { r c l r } { \widehat { \mathrm { G W } _ { \lambda } ^ { 2 } } } & { \ge } & { \mathrm { G W } _ { \mathrm { a d v } } ^ { 2 } - \frac { \tau } { 3 } \ge L _ { \mathrm { a d v } } - \frac { \tau } { 3 } } \end{array}$ . Since $\begin{array} { r } { t ^ { * } \le L _ { \mathrm { a d v } } - \frac { \tau } { 2 } } \end{array}$ , we similarly conclude $\widehat { \mathrm { G W } _ { \lambda } ^ { 2 } } \ \geq \ t ^ { * } + \frac { \tau } { 6 } \ > \ t ^ { * }$ , so the classifier correctly outputs <sup>2</sup>“adversarial.”

Thus, on $E _ { \mathrm { g a p } } \cap E _ { \mathrm { e s t } }$ , the plug–in classifier is error–free. Finally, since $E _ { \mathrm { g a p } }$ holds with probability at least $1 - \bar { \delta _ { \mathrm { g a p } } }$ (from Theorem 3) and $E _ { \mathrm { e s t } }$ with probability at least $1 - \delta _ { \mathrm { { e s t } } }$ , a union bound yields P(misclassification) $\leq \delta _ { \mathrm { g a p } } + \delta _ { \mathrm { e s t } }$ □

Discussion. Lemma 4 formalizes the transition from a theoretical gap to a practical detector. Theorem 3 ensures a margin τ exists between clean and adversarial discrepancies. The lemma shows that if the empirical entropic GW estimator concentrates within $\tau / 3$ of the truth, then a midpoint threshold $t ^ { * }$ separates the two classes with zero error. The factor $1 / 3$ is convenient: it splits the error budget evenly, allowing statistical variance and entropic bias to each consume at most $\tau / 6 .$ . This provides a direct analogue to margin-based classifiers in statistical learning: once the theoretical gap is positive, robust classification depends only on estimator concentration, not on further geometric properties of C or K. In particular, higher dimension amplifies τ, so the limiting risk is controlled primarily by solver accuracy and sample complexity rather than geometry itself.

## G Details of Adversarial Image Generation

We focus on the following white-box and black-box attacks in this work across the supervised and zero-shot settings:

## G.1 White-Box Attacks

White-box attacks assume access to the internal parameters of the target model.

• Auto Attack (Croce & Hein, 2020): A parameter-free ensemble attack combining four complementary attacks: APGD-CE, APGD-DLR, FAB-T, and Square Attack. The ensemble automatically selects optimal hyperparameters and provides reliable robustness evaluation without manual tuning.

• Carlini & Wagner (C&W) Attack Carlini & Wagner (2017): An optimization-based attack that formulates adversarial example generation as:

$$
\operatorname* { m i n } _ { \delta } \| \delta \| _ { p } + c \cdot f ( x + \delta )\tag{35}
$$

where $f ( x + \delta ) = \mathrm { m a x } ( \mathrm { m a x } \{ Z ( x + \delta ) _ { i } : i \neq t \} - Z ( x + \delta ) _ { t } , - \kappa )$ with $Z$ representing logits, t the target class, and κ the confidence parameter.

• Projected Gradient Descent (PGD) Attack Madry et al. (2018): An iterative first-order adversarial attack using projected gradient descent:

$$
x _ { t + 1 } = \Pi _ { S } ( x _ { t } + \alpha \cdot \mathrm { s i g n } ( \nabla _ { x } \ell ( \theta , x _ { t } , y ) ) )\tag{36}
$$

where $\Pi _ { S }$ denotes projection onto the constraint set $S = \{ x ^ { \prime } : \| x ^ { \prime } - x \| _ { \infty } \leq \epsilon \}$ and ℓ is the loss function.

• Auto-PGD (APGD) Attack (Croce & Hein, 2020): An enhanced version of PGD with automatic step size adaptation and momentum. The step size is dynamically adjusted based on the loss trajectory:

$$
\alpha _ { t } = \alpha _ { 0 } \cdot \rho ^ { k _ { t } }\tag{37}
$$

where $k _ { t }$ counts the number of step size reductions and $\rho = 0 . 7 5$

• Fast Gradient Sign Method (FGSM) (Goodfellow et al., 2014): A single-step attack that generates adversarial examples using:

$$
x _ { a d v } = x + \epsilon \cdot \mathrm { s i g n } ( \nabla _ { x } J ( \theta , x , y ) )\tag{38}
$$

where $J$ is the cost function used to train the neural network, θ are the model parameters, and ϵ controls the perturbation magnitude.

• Universal Adversarial Perturbation (Moosavi-Dezfooli et al., 2017): Generates imageagnostic perturbations that fool classifiers across different inputs:

$$
\operatorname* { m i n } _ { v } \| v \| _ { p } { \mathrm { ~ s u b j e c t ~ t o ~ } } \mathbb { P } _ { x \sim \mu } [ { \hat { k } } ( x + v ) \neq { \hat { k } } ( x ) ] \geq 1 - \delta\tag{39}
$$

where v is the universal perturbation, $\mu$ is the data distribution, and $\delta$ is the desired fooling rate.

• Adversarial Patch Attack Brown et al. (2017): Generates printable adversarial patches that can cause misclassification in the physical world:

$$
\hat { p } = \arg \operatorname* { m a x } _ { p } \mathbb { E } _ { x , t , l } [ \log \operatorname* { P r } ( \hat { y } | A ( p , x , l , t ) ) ]\tag{40}
$$

where $A ( p , x , l , t )$ applies patch $p$ to image x at location l with transformation $t ,$ and $\hat { y }$ is the target class.

## G.2 Black-Box Attacks

Black-box attacks operate without knowledge of internal model parameters.

• Frequency Attack Yin et al. (2019): Exploits the vulnerability of neural networks in the frequency domain by applying perturbations to the Fourier transform:

$$
\mathcal { F } ( x _ { a d v } ) = \mathcal { F } ( x ) + \delta _ { f }\tag{41}
$$

where $\mathcal { F }$ denotes the Fourier transform and $\delta _ { f }$ represents frequency-domain perturbations.

• Square Attack Andriushchenko et al. (2020): A query-efficient score-based black-box attack that uses random search within $\ell _ { p }$ balls:

$$
x _ { t + 1 } = x _ { t } + \eta _ { t } \cdot h _ { t }\tag{42}
$$

where $h _ { t }$ is a random direction sampled uniformly from $\{ - 1 , + 1 \} ^ { d }$ and $\eta _ { t }$ is the step size adapted based on the attack success.

• Gaussian Blur Attack (Zhang et al., 2022): Applies Gaussian blur to exploit the frequency bias of deep neural networks:

$$
G _ { \sigma } ( x , y ) = \frac { 1 } { 2 \pi \sigma ^ { 2 } } \exp { \left( - \frac { x ^ { 2 } + y ^ { 2 } } { 2 \sigma ^ { 2 } } \right) }\tag{43}
$$

where $\sigma$ controls the blur intensity and the convolution $x _ { b l u r } = x * G _ { \sigma }$ generates the adversarial example.

• Semantic Rotation Attack (Hosseini & Poovendran, 2018): Applies geometric transformations including rotations that preserve semantic content while causing misclassification:

$$
R _ { \theta } = \left[ \begin{array} { c c } { \cos \theta } & { - \sin \theta } \\ { \sin \theta } & { \cos \theta } \end{array} \right]\tag{44}
$$

where $\theta$ represents the rotation angle applied to the input image coordinates.

• Pixel Flip Attack (Su et al., 2019): A sparse attack that modifies only a few pixels to cause misclassification:

$$
\operatorname* { m i n } | S | \operatorname { s u b j e c t } \operatorname { t o } f ( x \oplus \delta _ { S } ) \neq f ( x )\tag{45}
$$

where $S$ is the set of modified pixel locations, $\delta _ { S }$ represents the pixel modifications, and ⊕ denotes the modification operation.

## G.3 Adversarial Image Generation in the Supervised Setting

In the supervised setting, adversarial examples are generated against traditional classification models trained on labeled datasets. The model produces logits through a standard forward pass:

$$
z = f _ { \boldsymbol { \theta } } ( x ) ,\tag{46}
$$

where $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ denotes the neural network with parameters $\theta , x \in \mathbb { R } ^ { H \times W \times C }$ is the input image, and $z \in \mathbb { R } ^ { K }$ are the raw logits for $K$ classes.

The final classification layer is typically a linear transformation:

$$
\boldsymbol { z } = \boldsymbol { W } ^ { T } \boldsymbol { h } + \boldsymbol { b } ,\tag{47}
$$

where $h$ is the penultimate layer representation, $W \in \mathbb { R } ^ { d \times K }$ is the weight matrix, and $b \in \mathbb { R } ^ { K }$ is the bias vector.

The predicted class probabilities are obtained via the softmax function:

$$
P ( y = c \mid x ) = \frac { \exp ( z _ { c } ) } { \sum _ { j = 1 } ^ { K } \exp ( z _ { j } ) } ,\tag{48}
$$

where $z _ { c }$ is the logit for class c.

During adversarial attack generation, the commonly used loss function is the cross-entropy loss:

$$
\mathcal { L } ( x , y ) = - \log P ( y \mid x ) = - z _ { y } + \log \left( \sum _ { j = 1 } ^ { K } \exp ( z _ { j } ) \right) .\tag{49}
$$

## G.4 Adversarial Image Generation in the Zero-Shot Setting

In the zero-shot setting, adversarial examples are generated against Vision-Language Models (VLMs) using its image encoder model (in our case CLIP Radford et al. (2021)), which do not require training on the target classes. The model consists of separate image and text encoders that project inputs into a shared embedding space.

Given an input image $x \in \mathbb { R } ^ { H \times W \times C }$ and a set of K class names $\{ c _ { 1 } , c _ { 2 } , \ldots , c _ { K } \}$ , the zero-shot classification process proceeds as follows:

Image Encoding: The image encoder $E _ { I } \ : \ \mathbb { R } ^ { H \times W \times C } \ \to \ \mathbb { R } ^ { d }$ maps the input image to an $\ell _ { 2 ^ { - } }$ normalized embedding in the shared representation space:

$$
{ v _ { I } } = \frac { { E _ { I } } ( x ) } { { \| E _ { I } ( x ) \| _ { 2 } } } ,\tag{50}
$$

where $v _ { I } \in \mathbb { R } ^ { d }$ represents the normalized image embedding with unit norm.

Text Encoding: For each class $c _ { i } .$ , a text prompt is constructed using the template “A photo of $\boldsymbol { c _ { i } } ^ { \flat }$ The text encoder $E _ { T } : \mathcal { V } ^ { * } \to \mathbb { R } ^ { d }$ maps each prompt to a normalized embedding in the same shared space:

$$
v _ { T , i } = \frac { E _ { T } ( ^ { * } \mathrm { { A p h o t o o f } } c _ { i } { ^ { * } } ) } { \lVert E _ { T } ( ^ { * } \mathrm { { A p h o t o o f } } c _ { i } { ^ { * } } ) \rVert _ { 2 } } ,\tag{51}
$$

where $\nu ^ { * }$ denotes the vocabulary space and $v _ { T , i } \in \mathbb { R } ^ { d }$ is the normalized text embedding for class $c _ { i }$ Logit Computation: The logits are computed as the temperature-scaled cosine similarities between the image embedding and each text embedding:

$$
z _ { i } = \tau \cdot v _ { I } ^ { T } v _ { T , i } = \tau \cdot \cos ( v _ { I } , v _ { T , i } ) ,\tag{52}
$$

where $\tau > 0$ is a temperature parameter that controls the sharpness of the similarity distribution. The complete logit vector is:

$$
z = \tau \cdot \left[ v _ { I } ^ { T } v _ { T , 1 } , \quad v _ { I } ^ { T } v _ { T , 2 } , \quad \ldots , \quad v _ { I } ^ { T } v _ { T , K } \right] ^ { T } \in \mathbb { R } ^ { K } .\tag{53}
$$

Classification Decision: The predicted class is obtained by selecting the class with the maximum logit value:

$$
\hat { y } = \arg \operatorname* { m a x } _ { i \in \{ 1 , \ldots , K \} } z _ { i } = \arg \operatorname* { m a x } _ { i \in \{ 1 , \ldots , K \} } v _ { I } ^ { T } v _ { T , i } .\tag{54}
$$

The class posterior probabilities are obtained through softmax normalization:

$$
P ( y = c _ { i } \mid x ) = \frac { \exp { \left( \tau \cdot v _ { I } ^ { T } v _ { T , i } \right) } } { \sum _ { j = 1 } ^ { K } \exp { \left( \tau \cdot v _ { I } ^ { T } v _ { T , j } \right) } } .\tag{55}
$$

## G.5 Adversarial Attack Hyperparameter Selection

In this section, we provide the detailed configuration of adversarial attack methods used in our experiments (Section 6). Table 5 summarizes the set of hyperparameters chosen for each attack. These values are selected following common practice in the adversarial robustness literature to ensure a fair comparison across methods.

We consider a diverse set of attack strategies, including optimization-based, gradient-based, scorebased, and patch-based approaches. For gradient-based methods such as PGD, Auto-PGD, FGSM, and Square Attack, we evaluate under multiple perturbation budgets with $\epsilon \in \lbrace 4 / 2 5 5 , 8 / 2 5 5 \rbrace$ Universal Perturbation is evaluated with a wider range of perturbation strengths, namely $\epsilon \in$ {4/255, 8/255, 12/255}.

For optimization-based attacks, the Carlini & Wagner (CW) attack is configured with confidence parameter $\kappa = 0 . 0$ , following the default setting to generate minimally perturbed adversarial examples. AutoAttack is evaluated under $\epsilon \in \{ 4 / 2 5 5 , \check { 8 } / 2 \check { 5 } 5 \}$ , consistent with its standardized benchmark protocol.

Patch-based and spatial transformations are included to account for more physically realizable adversarial scenarios. The Patch Attack is tested with square patches of shape (3, 8, 8) and (3, 16, 16), while the Spatial Attack allows for up to $3 0 ^ { \circ }$ rotation and translations of up to 10% of the image dimensions.

In addition to the attacks described above, we also include several specialized perturbations and image corruptions to probe robustness across different perturbation modalities: a Frequency Attack with noise strength noise\_strength = 0.05, Gaussian Blur with σ = 1.0 (blur\_type set to “uniform”), Pixel Flip with num\_pixel = 5 and attack mode set to “random”, and a Semantic Rotation with angle = 8<sup>◦</sup>. These parameter choices are summarized in Table 5 and were chosen to reflect common settings used in prior work while providing a broad coverage of perturbation types.

These configurations ensure that the evaluation captures a broad spectrum of attack types, ranging from small-norm pixel perturbations to structured and geometric transformations. See Table 5 for an overview of the attack methods and the full set of parameter values used in our experiments.

<table><tr><td>Attack Method</td><td>Parameter Values</td></tr><tr><td>AutoAttack (AA)</td><td> $\overline { { \epsilon = 4 / 2 2 5 , 8 / 2 2 5 } }$ </td></tr><tr><td>C&amp;W (CW)</td><td> $\kappa = \mathbf { 0 . 0 }$ </td></tr><tr><td>Patch Attack (PT)</td><td> $\mathrm { p a t c h \_ s h a p e = ( { \bf 3 } , { \bf 8 } , { \bf 8 } ) , ( { \bf 3 } , 1 6 , 1 6 ) }$ </td></tr><tr><td>PGD Spatial Attack (SA)</td><td> $\epsilon = { \bf 4 } / { \bf 2 5 5 } , 8 / 2 5 5$ </td></tr><tr><td>Square Attack (SQ)</td><td> $\operatorname* { m a x } \operatorname { r o t a t i o n } ; { \mathbf { 3 0 } } ^ { \circ }$  , max translation: 10% of the image size  $\epsilon = { \bf 4 } / { \bf 2 5 5 } , 8 / 2 5 5$ </td></tr><tr><td>Universal Perturbation (UP)</td><td></td></tr><tr><td>Auto-PGD (AP)</td><td> $\epsilon = { \bf 4 } / { \bf 2 5 5 , } 8 / { \bf 2 5 5 , } 1 2 / { \bf 2 5 5 }$ </td></tr><tr><td>FGSM (FG)</td><td> $\epsilon = { \bf 4 } / { \bf 2 5 5 } , 8 / 2 5 5$ </td></tr><tr><td>Frequency Attack (FA)</td><td> $\epsilon = { \bf 4 } / { \bf 2 5 5 } , 8 / 2 5 5$ </td></tr><tr><td>Gaussian Blur (GB)</td><td> $\mathrm { n o i s e \_ s t r e n g i h } = \mathbf { 0 . 0 5 }$ </td></tr><tr><td>Pixel Flip (PF)</td><td> $\sigma = \mathbf { 1 . 0 } , \mathrm { b l u r \_ t y p e } = \mathrm { u n i f o r m }$ </td></tr><tr><td></td><td>num_pixel = 5, attack_mode = random</td></tr><tr><td>Semantic Rotation (SR)</td><td>angle = 8</td></tr></table>

Table 5: Overview of adversarial attack methods and their parameter settings. Parameter value written with bold represents the default value of the corresponding attack among its parameter configurations.

## G.6 Examples of Adversarial Images Supervised Setting

## G.6.1 Gaussian Blur Attack

![](images/667867e07cfbfff190f90d31c16aa95f33e3d86041f9ba640a2251b801ed5922.jpg)  
Figure 3: Comparison between Original and Perturbed Images using Gaussian blur attack. Left: Original Image with True Label car, Center: Adversarial Image with Predicted Label truck and Right: Perturbation

![](images/b7bfefc9e839bbf32ee7aa1a7b916f62339a83e11904639f1a64513c48f985fa.jpg)

![](images/6892141a40d800ab68e65f33f72be05b7b588a2bd3d540abdcca969f8f29c735.jpg)

![](images/3ac8cb4f631ba6692ec4317b8622c83003b7ab6088c9a9bd779c4f6ae1c0648d.jpg)  
Figure 4: Comparison between Original and Perturbed Images using Gaussian blur attack. Left: Original Image with True Label plane, Center: Adversarial Image with Predicted Label bird and Right: Perturbation

## G.6.2 Patch Attack

![](images/e7ecc128d618e22503358502c9013a206631dbcae5b082249933fdb0697044f5.jpg)

![](images/a0c9933c77ed51bf6996b125dd68d5871fa6b9c9dcfc9aaa3890fdaf9ec38cd9.jpg)

![](images/93cfb76aaaa9073f16c9405d3a1a4ef37f6139fc40d6a0c2c95eef6e180c2674.jpg)  
Figure 5: Comparison between Original and Perturbed Images using Patch attack. Left: Original Image with True Label car, Center: Adversarial Image with Predicted Label truck and Right: Perturbation

![](images/dc75589515c56834ff00b49876189ccf7c58c2bdeb0f4da6cca87b0630f90f39.jpg)

![](images/d1fc7ca63d04d9f5a4c1e05b171782c640e0fce3262b01c931fded56f89bc682.jpg)

![](images/2cbdc2ef423dfcd5ae54b3e05557005101435b882dfeb0d7c96ae4c8239a7f0e.jpg)  
Figure 6: Comparison between Original and Perturbed Images using Patch attack. Left: Original Image with True Label plane, Center: Adversarial Image with Predicted Label bird and Right: Perturbation

## G.7 Examples of Adversarial Images Zero Shot Setting

## G.7.1 APGD Attack

![](images/569bb9f38bbddbe7aefa0070a1bda7b94bc69f3425a8229caa4352b1519eb37f.jpg)  
Figure 7: Comparison between Original and Perturbed Images using APGD attack. Left: Original Image with True Label Dolphin, Center: Adversarial Image with Predicted Label Camera and Right: Perturbation

## G.7.2 PGD Attack

![](images/cc9ee69a6ac726d87a20df5790c0e8002c20fd817b99aa21f177941418413ecd.jpg)  
Figure 8: Comparison between Original and Perturbed Images using APGD attack. Left: Original Image with True Label Trilobite, Center: Adversarial Image with Predicted Label Helicopter and Right: Perturbation

## H Detection Methods Configuration

This section provides detailed descriptions of the adversarial detection methods evaluated in our experiments, with configurations specified in Table 6. Further, we also mention the hyperparameter settings used in our defense approach.

## H.1 Mahalanobis Detector

The Mahalanobis detector (Lee et al., 2018) leverages the Mahalanobis distance to measure distributional deviations of test samples from training data in the neural network’s feature space. For a given sample x and its feature representation f(x) at layer l, the method computes class-conditional Gaussian distributions $\mathcal { N } ( \mu _ { c } ^ { ( l ) } , \pmb { \Sigma } ^ { ( l ) } )$ from clean training data. The Mahalanobis distance is defined as:

$$
M _ { c } ^ { ( l ) } ( \mathbf { x } ) = ( \mathbf { f } ^ { ( l ) } ( \mathbf { x } ) - { \pmb { \mu } } _ { c } ^ { ( l ) } ) ^ { T } ( \pmb { \Sigma } ^ { ( l ) } ) ^ { - 1 } ( \mathbf { f } ^ { ( l ) } ( \mathbf { x } ) - { \pmb { \mu } } _ { c } ^ { ( l ) } )
$$

The minimum distance across all classes serves as the confidence score for adversarial detection, exploiting the property that adversarial perturbations typically push samples away from the natural data manifold.

## H.2 Feature Squeezing

Feature Squeezing (Xu et al., 2018) reduces the degrees of freedom available to adversarial perturbations by applying input transformations that compress the feature space. Our implementation uses median smoothing with a $2 \times 2$ kernel and $L _ { 1 }$ distance metric for comparing predictions. For an input x and its squeezed version $\mathbf { x } ^ { \prime }$ , the detection score is computed as:

$$
\mathrm { s c o r e } ( \mathbf { x } ) = \| \mathbf { p } ( \mathbf { x } ) - \mathbf { p } ( \mathbf { x } ^ { \prime } ) \| _ { 1 }
$$

where $\mathbf { p } ( \cdot )$ represents the model’s prediction probabilities. A threshold is determined using the training false positive rate (FPR) of 0.2, assuming legitimate inputs remain robust to minor spatial transformations while adversarial examples exhibit significant prediction changes.

## H.3 MetaDetect

MetaDetect (Ma et al., 2019) formulates adversarial detection as a few-shot learning problem using meta-learning principles. The method employs episodic training with support sets ${ \cal S } = \{ ( { \bf x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { s } }$ and query sets $\mathcal { Q } = \{ ( \mathbf { x } _ { j } , y _ { j } ) \} _ { j = 1 } ^ { N _ { q } }$ , where $y _ { i } \in \{ 0 , 1 \}$ indicates clean (0) or adversarial (1) samples. Our configuration uses $N _ { s } ~ = ~ 1$ support example and $N _ { q } ~ = ~ 1 5$ query examples with a conv3 architecture. The meta-detector learns a function $f _ { \theta }$ that maps from support-query episode pairs to detection decisions, optimizing over episode distributions to generalize across different attack types.

## H.4 MagNet

MagNet (Meng & Chen, 2017) combines detection and defense mechanisms using autoencoder-based reconstruction and probability estimation. The method trains an autoencoder $\mathcal { E } _ { \phi } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ on clean data to approximate the natural data manifold. For detection, it computes the reconstruction error:

$$
\mathcal { L } _ { \mathrm { r e c } } ( \mathbf { x } ) = \| \mathbf { x } - \mathcal { E } _ { \phi } ( \mathbf { x } ) \| _ { 1 } ^ { 2 }
$$

Additionally, MagNet estimates the probability density using the Jensen-Shannon divergence between the original and reconstructed inputs’ predicted distributions. The underlying assumption is that adversarial examples, lying off the natural manifold, will exhibit higher reconstruction errors and lower probability estimates compared to legitimate inputs.

Detection Method Parameters   
Mahalanobis Detector train $\overline { { \mathrm { f p r } = 0 . 1 5 } }$   
Feature Squeezing distance metric: $L _ { 1 }$ , squeezer: median smoothing $( 2 \times 2 ) .$ , train\_fpr=0.2   
MetaDetect num\_support = 1, num\_query = 15, arch = conv3   
MagNet $l _ { 1 }$ norm reconstruction error, train\_fpr = 0.15   
Table 6: Parameters and configurations for different adversarial detection methods. Methods refer  
enced: Mahalanobis (Lee et al., 2018), Feature Squeezing (Xu et al., 2018), MetaDetect (Ma et al.,   
2019), and MagNet (Meng & Chen, 2017).

H.5 Hyper Parameter Configuration for the Proposed Method
<table><tr><td>Hyperparameter</td><td>Candidate Values</td><td>Optimal Value</td><td>Description</td></tr><tr><td> $k _ { \mathrm { l o c a l } }$ </td><td>{8, 10, 12, 15}</td><td>8</td><td>Local Gromov-Wasserstein features</td></tr><tr><td> $k _ { \mathrm { g l o b a l } }$ </td><td>{3, 5, 7}</td><td>3</td><td>Global Gromov-Wasserstein features</td></tr><tr><td> $\epsilon _ { \mathrm { g w } }$ </td><td>{0.2, 0.5, 0.8}</td><td>0.5</td><td>Entropic regularization strength</td></tr><tr><td>Kernel</td><td>{linear, rbf, poly}</td><td>rbf</td><td>Choice of kernel function</td></tr><tr><td> $C$ </td><td>{0.1, 1, 10, 100}</td><td>1</td><td>Regularization parameter</td></tr><tr><td>γ</td><td>{scale, auto, 0.01, 0.001 }</td><td>scale</td><td>Kernel Coeff. for RBF</td></tr></table>

Table 7: Hyperparameter search for GW features: We performed a grid search over the candidate values of each hyperparameter and chose the values that achieved the best trade-off between robustness and model usability. Based on this search, we selected the optimal parameters as $k _ { \mathrm { l o c a l } } = 8 , k _ { \mathrm { g l o b a l } } = 3 .$ and $\epsilon _ { \mathrm { g w } } = 0 . 5$

Table 8: Hyperparameter search for SVM: We performed a grid search over the candidate values of each hyperparameter and selected the optimal configuration based on validation accuracy. The chosen parameters are Kernel = rbf, C = 1, γ = scale.

## I Adaptive Attack Formulation

Evaluating the robustness of a defense mechanism against an adaptive adversary is crucial. We consider an adversary who possesses complete knowledge of the defense’s architecture, including the classifier $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ , the CNN feature extractor $\phi _ { \mathrm { c n n } } ( x )$ , and the crypto feature extractor $\phi _ { \mathrm { c r } } ^ { ( b ) } ( x )$ with its associated transform $T _ { b }$ . However, the adversary is unaware of the defender’s specific, fixed secret bit vector $b ^ { \star } \in \{ 0 , 1 \} ^ { D }$ used in deployment. This section formalizes the adversary’s objective and optimization strategy to generate adversarial examples under this realistic uncertainty, focusing on two distinct consistency-based attacks. This approach is typical for evaluating defenses against strong, adaptive attackers Athalye et al. (2018).

## I.1 Attacker’s Prior over Secret Bits

To account for the unknown secret $b ^ { \star }$ , the adversary models it as a random variable b drawn from a prior distribution $p ( b )$ . This prior is constructed as a mixture model over a set of plausible Bernoulli distributions, $\mathcal { M } .$ , reflecting the adversary’s uncertainty about the specific statistical properties of $b ^ { \star } \colon$

$$
p ( b ) \ = \ \frac { 1 } { | \mathcal { M } | } \sum _ { m \in \mathcal { M } } \prod _ { j = 1 } ^ { D } \mathrm { B e r n o u l l i } ( b _ { j } ; p _ { m } ) ,
$$

where $p _ { m }$ corresponds to the individual Bernoulli success probability for each distribution type in $\mathcal { M } \left( \mathrm { e . g . , } p _ { m } = 0 . 5 \right.$ for uniform or Gaussian-threshold components, and specific probabilities like $0 . 3 , 0 . 7$ for biased Bernoulli components). This mixture prior allows the adversary to account for various possibilities of how the defender might have generated $b ^ { \star }$ . Modeling unknown parameters in this manner is a standard robust optimization technique Ben-Tal et al. (2009).

## I.2 Attacker’s Objective Function

The adversary’s goal is to craft an adversarial example x from a benign input $x _ { 0 }$ that achieves misclassification by $f _ { \theta } ( x )$ while simultaneously maintaining a high degree of feature consistency with $x _ { 0 }$ . The latter ensures the adversarial example does not trip the defense’s detection mechanisms, particularly those relying on the crypto features. Since the specific $b ^ { \star }$ is unknown, the adversary targets an average consistency, minimizing the expected penalty under their prior $p ( b )$

The general adversarial objective is:

$$
\displaystyle \operatorname* { m a x } _ { x \in \mathcal { X } } \mathcal { L } ( x ; x _ { 0 } , y ) = \ell ( f _ { \theta } ( x ) , y ) - \lambda \mathbb { E } _ { b \sim p ( b ) } [ C ( x , x _ { 0 } ; b ) ] ,
$$

where $\ell ( f _ { \theta } ( x ) , y )$ is the cross-entropy loss for the true label $y ,$ which the adversary seeks to maximize; $\lambda > 0$ is a weighting factor that balances the misclassification objective against the consistency penalty — a formulation commonly used in adversarial attacks to trade off attack success and imperceptibility or stealth Carlini $\&$ Wagner (2017); $C ( x , x _ { 0 } ; b )$ quantifies the discrepancy between features of x and $x _ { 0 }$ for a given $b ,$ with lower values of C implying better stealth against consistency checks; and X defines the allowed perturbation space, typically restricted to a range $[ a , b ] ^ { H \times W \times \check { C } }$ for pixel values.

## I.3 Consistency Penalties $( C ( x , x _ { 0 } ; b ) )$

We define the consistency penalty $C ( x , x _ { 0 } ; b )$ using an OT-like discrepancy metric, $D _ { \mathrm { O T } } ( \cdot , \cdot )$ . This metric compares feature vectors $( L _ { 2 }$ distance if dimensions match) or their statistical summaries

$( L _ { 2 }$ distance between mean, std, min, max, skewness, kurtosis if dimensions differ or comparison of statistics is explicitly requested).

For this study, we consider two specific attack formulations based on distinct consistency penalties:

## I.3.1 Cross-Space Consistency Attack $\left( C _ { \mathrm { c r o s s } } \right)$

This attack targets the defense by imposing consistency across both the standard CNN feature space and the specialized crypto feature space. The adversary aims to ensure that the features extracted from the adversarial example x remain similar to those from the clean input $x _ { 0 }$ in both domains. The penalty term is defined as the sum of discrepancies in each feature space:

$$
C _ { \mathrm { c r o s s } } ( x , x _ { 0 } ; b ) = D _ { \mathrm { O T } } { \big ( } \phi _ { \mathrm { c n n } } ( x ) , \phi _ { \mathrm { c n n } } ( x _ { 0 } ) { \big ) } + D _ { \mathrm { O T } } { \big ( } \phi _ { \mathrm { c r } } ^ { ( b ) } ( x ) , \phi _ { \mathrm { c r } } ^ { ( b ) } ( x _ { 0 } ) { \big ) } .
$$

By minimizing this penalty, the adversarial example is constrained to modify the input in a way that, on average over $b ,$ preserves the inherent characteristics captured by both $\phi _ { \mathrm { c n n } }$ and $\phi _ { \mathrm { c r } } ^ { ( b ) }$

## I.3.2 Multi-Scale Consistency Attack $( C _ { \mathrm { m s } } )$

This attack extends the cross-space consistency by introducing an additional constraint on global CNN feature similarity. This reflects a defense that might perform multi-scale or global consistency checks specifically on CNN features. The $C _ { \mathrm { m s } }$ penalty is structured as:

$$
C _ { \mathrm { m s } } ( x , x _ { 0 } ; b ) = C _ { \mathrm { l o c a l } } ( x , x _ { 0 } ; b ) + C _ { \mathrm { g l o b a l } } ( x , x _ { 0 } ) ,
$$

where:

$C _ { \mathrm { l o c a l } } ( x , x _ { 0 } ; b ) = C _ { \mathrm { c r o s s } } ( x , x _ { 0 } ; b )$ represents the local, cross-space consistency across CNN and crypto features.

$C _ { \mathrm { g l o b a l } } ( x , x _ { 0 } ) = D _ { \mathrm { O T } } { \big ( } \phi _ { \mathrm { c n n } } ( x ) , \phi _ { \mathrm { c n n } } ( x _ { 0 } ) { \big ) }$ ensures global consistency focusing solely on CNN features. Note that $C _ { \mathrm { g l o b a l } }$ does not depend on b, as the global consistency check is assumed to be deterministic based on CNN features, which are not secrets-dependent.

The combined penalty

$$
C _ { \mathrm { m s } } ( x , x _ { 0 } ; b ) = 2 \cdot D _ { \mathrm { O T } } \bigl ( \phi _ { \mathrm { c n n } } ( x ) , \phi _ { \mathrm { c n n } } ( x _ { 0 } ) \bigr ) + D _ { \mathrm { O T } } \bigl ( \phi _ { \mathrm { c r } } ^ { ( b ) } ( x ) , \phi _ { \mathrm { c r } } ^ { ( b ) } ( x _ { 0 } ) \bigr )
$$

effectively doubles the weight on CNN feature consistency, making the adversarial example potentially harder to detect by defenses performing aggregated checks on CNN features.

To optimize the objective function, the adversary utilizes an iterative PGD Madry et al. (2018) is used. Since the objective involves an expectation over the unknown b, a Monte Carlo (MC) approximation is employed Rubinstein & Kroese (2016).

## J Implementation Details

## J.1 Pseudocode for Cross-Space Detector

Algorithm 1 Multi-Scale Cross-Space GW Detector   
Require: Image $\quad \ : \because z  h _ { \theta } ( x ) ; p  T _ { k } ^ { ( b ) } ( x )$   
1 for all $s \in \bar { \{ 1 0 , \mathrm { g l } \} }$ do   
2 Build $\mathcal { N } _ { s } ^ { \vec { Z } } ( z ) , \mathcal { N } _ { s } ^ { P } ( p ) ;$ compute $\mu _ { s } ^ { Z } , \mu _ { s } ^ { P } , \psi _ { s } ^ { Z } , \psi _ { s } ^ { P }$   
3 $g _ { 1 } \gets \mathrm { G W } _ { \lambda } ^ { 2 } ( \mu _ { s } ^ { Z } , \mu _ { s } ^ { P } ) ; g _ { 2 } \gets \mathrm { G W } _ { \lambda } ^ { 2 } ( \psi _ { s } ^ { Z } , \psi _ { s } ^ { P } ) .$   
4 $h  \mathrm { E N T R O P Y } ( \psi _ { s } ^ { Z } , \psi _ { s } ^ { P } ) ; \mathbf { f }  \mathbf { f } \parallel [ g _ { 1 } , g _ { 2 } , h ]$   
5 end for   
6 return SVM(f) ∈ {clean, adv}

We provide the hyperparmeter selection details for GW features in Table 7 and for SVM classifier in Table 8.

## K Additional Experiments

## K.1 Additional Baseline Defences and Attacks

To evaluate performance against imperceptible and optimization-free attacks, we extend our experiments to three recent low-magnitude adversarial methods, i.e., AdvAD, PGN, and BSR, which explicitly target the small-perturbation regime. These attacks probe the limits of pixel-level stability and serve as stringent tests for detectors relying on fine-grained geometric discrepancies.

In addition to our default ResNet-18 backbone, we generate all attacks using a Vision Transformer (ViT) to assess robustness across fundamentally different architectural families. Table 9 reports binary detection accuracy for all attacks and defenses. Table 10 reports corresponding AUROC values.

Across all attack types—including classical gradient-based attacks (PGD, SQ, PT) and modern diffusion-based or non-parametric attacks (AdvAD, PGN, BSR)—our detector achieves the strongest performance on both ResNet-18 and ViT. Competing baselines degrade substantially under stronger or low-magnitude attacks, whereas our Z–P discrepancy remains highly separable across architectures, perturbation magnitudes, and attack mechanisms.

<table><tr><td>Attack</td><td>Model</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td><td>EA</td><td>BY</td></tr><tr><td>PGD</td><td>ResNet-18</td><td>97.8</td><td>91.7</td><td>74.4</td><td>46.3</td><td>81.9</td><td>97.5</td><td>69.5</td></tr><tr><td></td><td>ViT</td><td>95.7</td><td>49.8</td><td>70.3</td><td>57.5</td><td>49.4</td><td>93.2</td><td>77.5</td></tr><tr><td>SQ</td><td>ResNet-18</td><td>97.6</td><td>89.1</td><td>88.5</td><td>45.8</td><td>91.9</td><td>91.9</td><td>59.5</td></tr><tr><td></td><td>ViT</td><td>96.2</td><td>51.0</td><td>87.3</td><td>56.3</td><td>49.7</td><td>90.5</td><td>62.5</td></tr><tr><td>PT</td><td>ResNet-18 ViT</td><td>98.0 95.4</td><td>86.4</td><td>67.3</td><td>46.5</td><td>50.1</td><td>90.1</td><td>78.0 78.5</td></tr><tr><td></td><td></td><td></td><td>51.3</td><td>71.2</td><td>54.9</td><td>49.3</td><td>89.5</td><td></td></tr><tr><td>AAD</td><td>ResNet-18</td><td>96.4</td><td>54.4</td><td>41.3</td><td>49.6</td><td>53.2</td><td>94.6</td><td>61.0</td></tr><tr><td></td><td>ViT</td><td>93.7</td><td>52.6</td><td>52.7</td><td>53.6</td><td>50.1</td><td>92.7</td><td>80.0</td></tr><tr><td>PGN</td><td>ResNet-18</td><td>96.5</td><td>70.8</td><td>61.0</td><td>45.6</td><td>49.6</td><td>94.9</td><td>75.0</td></tr><tr><td></td><td>ViT</td><td>96.9</td><td>52.2</td><td>65.8</td><td>53.7</td><td>49.9</td><td>94.9</td><td>75.5</td></tr><tr><td>BSR</td><td>ResNet-18</td><td>95.0</td><td>73.0</td><td>42.4</td><td>45.5</td><td>62.1</td><td>92.9</td><td>78.5</td></tr><tr><td></td><td>ViT</td><td>98.7</td><td>51.9</td><td>53.9</td><td>53.7</td><td>50.0</td><td>93.2</td><td>76.0</td></tr></table>

Table 9: Detection accuracy (%) on CIFAR-10 for a range of classical and low-magnitude attacks, evaluated using ResNet-18 and ViT. Best results are in bold; second best are underlined.

<table><tr><td>Attack</td><td>Model</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td><td>EA</td><td>BY</td></tr><tr><td rowspan="2">PGD</td><td>ResNet-18</td><td>0.99</td><td>0.81</td><td>0.69</td><td>0.49</td><td>0.51</td><td>0.99</td><td>0.72</td></tr><tr><td>ViT</td><td>0.99</td><td>0.72</td><td>0.72</td><td>0.57</td><td>0.49</td><td>0.96</td><td>0.89</td></tr><tr><td rowspan="2">SQ</td><td>ResNet-18</td><td>0.99</td><td>0.55</td><td>0.87</td><td>0.48</td><td>0.49</td><td>0.95</td><td>0.54</td></tr><tr><td>ViT</td><td>0.99</td><td>0.40</td><td>0.88</td><td>0.56</td><td>0.50</td><td>0.95</td><td>0.60</td></tr><tr><td rowspan="2">PT</td><td>ResNet-18</td><td>0.99</td><td>0.93</td><td>0.52</td><td>0.50</td><td>0.54</td><td>0.95</td><td>0.89</td></tr><tr><td>ViT</td><td>0.99</td><td>0.47</td><td>0.63</td><td>0.54</td><td>0.52</td><td>0.94</td><td>0.90</td></tr><tr><td rowspan="2">AAD</td><td>ResNet-18</td><td>0.99</td><td>0.61</td><td>0.56</td><td>0.49</td><td>0.51</td><td>0.97</td><td>0.55</td></tr><tr><td>ViT</td><td>0.98</td><td>0.88</td><td>0.55</td><td>0.53</td><td>0.53</td><td>0.97</td><td>0.94</td></tr><tr><td rowspan="2">PGN</td><td>ResNet-18</td><td>0.99</td><td>0.79</td><td>0.59</td><td>0.48</td><td>0.50</td><td>0.98</td><td>0.80</td></tr><tr><td>ViT</td><td>0.97</td><td>0.54</td><td>0.61</td><td>0.55</td><td>0.51</td><td>0.98</td><td>0.81</td></tr><tr><td rowspan="2">BSR</td><td>ResNet-18</td><td>0.99</td><td>0.64</td><td>0.53</td><td>0.48</td><td>0.51</td><td>0.95</td><td>0.87</td></tr><tr><td>ViT</td><td>0.99</td><td>0.59</td><td>0.58</td><td>0.55</td><td>0.54</td><td>0.96</td><td>0.84</td></tr></table>

Table 10: AUROC comparison on CIFAR-10 for various attacks generated using ResNet-18 and ViT. Best results are in bold; second best are underlined.

## K.2 Experimental evaluations with additional metrics

<table><tr><td>Attacks</td><td>Model</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td><td>EA</td><td>BY</td></tr><tr><td rowspan="2">PGD</td><td>ResNet-18</td><td>0.97/0.98</td><td>0.89/0.95</td><td>0.71/0.72</td><td>0.17/0.54</td><td>0.97/0.65</td><td>0.98/0.95</td><td>0.67/0.74</td></tr><tr><td>ViT</td><td>1.00/0.90</td><td>0.49/0.95</td><td>0.73/0.78</td><td>0.57/0.57</td><td>0.25/0.01</td><td>0.98/0.93</td><td>0.72/0.90</td></tr><tr><td rowspan="2">SQ</td><td>ResNet-18</td><td>0.96/0.97</td><td>0.88/0.89</td><td>0.83/0.88</td><td>0.16/0.52</td><td>0.22/0.01</td><td>0.89/0.92</td><td>0.60/0.54</td></tr><tr><td>ViT</td><td>0.95/0.94</td><td>0.50/0.98</td><td>0.93/0.80</td><td>0.56/0.57</td><td>0.40/0.01</td><td>0.88/0.90</td><td>0.63/0.60</td></tr><tr><td rowspan="2">PT</td><td>ResNet-18</td><td>0.97/0.98</td><td>0.87/0.84</td><td>0.62/0.49</td><td>0.17/0.57</td><td>0.55/0.01</td><td>0.91/0.88</td><td>0.72/0.91</td></tr><tr><td>ViT</td><td>0.99/0.93</td><td>0.50/0.98</td><td>0.75/0.61</td><td>0.53/0.56</td><td>0.25/0.01</td><td>0.89/0.87</td><td>0.72/0.92</td></tr><tr><td rowspan="2">AAD</td><td>ResNet-18</td><td>0.98/0.93</td><td>0.63/0.21</td><td>0.55/0.41</td><td>0.49/0.54</td><td>0.87/0.07</td><td>0.89/0.93</td><td>0.61/0.57</td></tr><tr><td>ViT</td><td>0.93/0.92</td><td>0.51/1.00</td><td>0.55/0.61</td><td>0.82/0.53</td><td>0.41/0.01</td><td>0.89/0.95</td><td>0.73/0.95</td></tr><tr><td rowspan="2">PGN</td><td>ResNet-18</td><td>0.97/0.93</td><td>0.83/0.51</td><td>0.67/0.59</td><td>0.15/0.51</td><td>0.34/0.01</td><td>0.90/0.91</td><td>0.70/0.85</td></tr><tr><td>ViT</td><td>0.98/0.95</td><td>0.51/0.99</td><td>0.67/0.58</td><td>0.82/0.53</td><td>0.44/0.01</td><td>0.92/0.94</td><td>0.71/0.86</td></tr><tr><td rowspan="2">BSR</td><td>ResNet-18</td><td>0.99/0.91</td><td>0.84/0.55</td><td>0.45/0.62</td><td>0.15/0.51</td><td>0.94/0.25</td><td>0.96/0.88</td><td>0.72/0.92</td></tr><tr><td>ViT</td><td>0.99/0.98</td><td>0.50/0.98</td><td>0.51/0.64</td><td>0.82/0.53</td><td>0.42/0.01</td><td>0.93/0.91</td><td>0.71/0.87</td></tr></table>

Table 11: Precision / Recall comparison of adversarial detection on CIFAR-10 under various attacks generated using ResNet-18 and ViT, with best results in bold and second best underlined.

Table 11 reports the precision and recall of different adversarial detection methods across six attack types using both ResNet-18 and ViT. Across nearly all attacks and architectures, our method achieves the highest precision and recall values (highlighted in bold), often reaching near-perfect scores close to 1.00, demonstrating its strong ability to accurately detect adversarial inputs. In contrast, EA generally appears as the second-best performer, but still lags behind our approach, particularly on challenging attacks such as AAD, PGN, and BSR. Overall, these results show that our detector is highly robust and consistent, maintaining superior numerical performance across all attack families and both model architectures.

![](images/9eca4affa5cb6fa3c98cc22b1281b57ead1f644dd69ebe97f97c51c04dfdb4cd.jpg)  
Figure 9: PR curves for our method on attacks generated via ResNet-18 model on CIFAR10 dataset. The small inset box displays a zoomed-in view of the upper-right corner of the PR curves.

In Figure 9, the Precision–Recall (PR) curves show that our detector maintains consistently high precision and recall across all six attack types. All curves remain close to the top-right region, indicating strong overall detection performance. The zoomed inset further illustrates that even at high recall values (close to 1.0), the precision for each attack remains above 0.95 with only minimal degradation. The high Average Precision (AP) scores (0.987 − −0.998) confirm the robustness of our method, demonstrating reliable performance even against stronger modern attacks such as AdvAD, PGN, and BSR. Overall, the PR curves highlight that our approach is highly accurate, stable, and generalizes well across diverse adversarial attacks.

Table 12 shows that our method consistently achieves the highest end-to-end detection accuracy across all evaluated attacks. For every attack type, our detector outperforms all competing defenses, often by margins of 10–40%. FS and MD occasionally achieve the second-best performance, but they remain substantially weaker overall, while MAD and MN lag far behind on most attacks. These results demonstrate that our approach generalizes robustly across a broad spectrum of adversarial perturbations and maintains reliable detection performance even against diverse and challenging attack strategies.

<table><tr><td>Ours MD FS MAD MN 71.6</td></tr><tr><td>AA 94.9 67.2 82.0 48.3 CW 93.8 72.1 85.3 48.9 54.2 PT 94.4 84.9 67.1 49.5 54.8 PGD 94.8 90.0 73.7 49.0 78.5 SA 93.4 76.7 73.7 39.2 52.5 SQ 93.4 87.7 87.8 48.5 41.9 UP 94.4 65.0 52.9 49.0 45.4 AP 93.8 66.9 80.7 48.6 71.1 FG 94.3 72.3 60.2 49.1 42.2 FA 94.6 48.5 49.3 48.7 47.3 GB 73.2 48.5 50.9 48.1 46.1</td></tr></table>

Table 12: End-to-end accuracy (%). Best results are in bold and second best are underlined.

<table><tr><td>Attacks</td><td>Model</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td><td>EA</td><td>BY</td></tr><tr><td>PGD</td><td>ResNet-18</td><td>95.6</td><td>50.00</td><td>68.9</td><td>45.8</td><td>91.00</td><td>94.9</td><td>-</td></tr><tr><td>SQ</td><td>ResNet-18</td><td>96.7</td><td>50.00</td><td>84.2</td><td>45.5</td><td>49.00</td><td>89.6</td><td>一</td></tr><tr><td>PT</td><td>ResNet-18</td><td>96.2</td><td>50.00</td><td>73.7</td><td>45.0</td><td>50.50</td><td>88.2</td><td>一</td></tr><tr><td>AAD</td><td>ResNet-18</td><td>97.6</td><td>50.00</td><td>55.9</td><td>49.6</td><td>59.38</td><td>95.4</td><td>一</td></tr><tr><td>PGN</td><td>ResNet-18</td><td>96.3</td><td>50.00</td><td>63.2</td><td>46.3</td><td>49.06</td><td>93.2</td><td>一</td></tr><tr><td>BSR</td><td>ResNet-18</td><td>97.8</td><td>50.00</td><td>59.4</td><td>46.35</td><td>50.31</td><td>93.6</td><td></td></tr></table>

Table 13: Detection accuracy (%) comparison on ImageNet + ResNet-18 with best (bold) and second best (underlined).

## K.3 Evaluation on the ImageNet Dataset on default ResNet-18 model

In this section, we additionally evaluate our method on the large-scale ImageNet Deng et al. (2009) dataset using the default ResNet-18 backbone. This allows us to verify that the detector remains effective when applied to high-resolution images and a significantly more challenging data distribution.

Note: The BY He et al. (2024) baseline does not provide a publicly available codebase, so we reproduce the method following the details reported in the original paper. Running BY requires a self-supervised model trained on the target dataset; while we trained such a model for CIFAR-10, training an equivalent backbone for ImageNet is computationally prohibitive. Consequently, we omit BY from the ImageNet experiments due to the intractable training cost and scale of the dataset.

Tables 13, 14, and 15 jointly show that our detector achieves the strongest overall performance on ImageNet across all six adversarial attacks. In terms of detection accuracy and AUC, our method consistently reaches top performance—typically in the 0.98–0.99 range—while the next-best baseline, EA, trails by 1 − 5% depending on the attack. The precision and recall results further reinforce this trend: our detector attains the best or second-best PR scores in nearly all settings, maintaining high recall even for challenging attacks such as AdvAD, PGN, and BSR. Competing defenses such as MD, FS, MAD, and MN perform substantially worse across all metrics. Together, these results confirm that our approach scales robustly to high-resolution ImageNet data and preserves strong discriminative ability under a wide range of adversarial perturbations.

<table><tr><td>Attacks</td><td>Model</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td><td>EA</td><td>BY</td></tr><tr><td>PGD</td><td>ResNet-18</td><td>0.98</td><td>0.53</td><td>0.73</td><td>0.503</td><td>0.68</td><td>0.98</td><td>一</td></tr><tr><td>SQ</td><td>ResNet-18</td><td>0.97</td><td>0.55</td><td>0.79</td><td>0.489</td><td>0.49</td><td>0.94</td><td>一</td></tr><tr><td>PT</td><td>ResNet-18</td><td>0.99</td><td>0.52</td><td>0.71</td><td>0.45</td><td>0.50</td><td>0.95</td><td>一</td></tr><tr><td>AAD</td><td>ResNet-18</td><td>0.99</td><td>0.49</td><td>0.55</td><td>0.495</td><td>0.45</td><td>0.96</td><td>一</td></tr><tr><td>PGN</td><td>ResNet-18</td><td>0.98</td><td>0.50</td><td>0.66</td><td>0.488</td><td>0.44</td><td>0.95</td><td>一</td></tr><tr><td>BSR</td><td>ResNet-18</td><td>0.99</td><td>0.50</td><td>0.62</td><td>0.488</td><td>0.43</td><td>0.95</td><td>一</td></tr></table>

Table 14: AUC comparison on ImageNet with best (bold) and second best (underlined).

<table><tr><td>Attacks</td><td>Model</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td><td>EA</td><td>BY</td></tr><tr><td>PGD</td><td>ResNet-18</td><td>1/0.93</td><td>0.50/1.00</td><td>0.65/0.82</td><td>0.05/0.55</td><td>0.92/1.00</td><td>0.98/0.97</td><td>一</td></tr><tr><td>SQ</td><td>ResNet-18</td><td>0.98/0.96</td><td>0.50/1.00</td><td>0.84/0.76</td><td>0.05/0.52</td><td>0.42/0.06</td><td>0.89/0.91</td><td></td></tr><tr><td>PT</td><td>ResNet-18</td><td>0.96/0.95</td><td>0.50/1.00</td><td>0.82/0.65</td><td>0.04/0.45</td><td>0.52/0.09</td><td>0.87/0.89</td><td></td></tr><tr><td>AAD</td><td>ResNet-18</td><td>0.99/0.96</td><td>0.50/1.00</td><td>0.51/0.55</td><td>0.49/0.54</td><td>0.87/0.21</td><td>0.93/0.96</td><td></td></tr><tr><td>PGN</td><td>ResNet-18</td><td>0.94/0.97</td><td>0.50/1.00</td><td>0.61/0.68</td><td>0.16/0.52</td><td>0.28/0.01</td><td>0.95/0.92</td><td></td></tr><tr><td>BSR</td><td>ResNet-18</td><td>0.97/0.96</td><td>0.50/1.00</td><td>0.50/0.73</td><td>0.16/0.52</td><td>0.54/0.03</td><td>0.96/0.91</td><td></td></tr></table>

Table 15: Precision / Recall comparison on ImageNet under various attacks, with best values in bold and second best underlined.

<table><tr><td>Attacks</td><td>Model</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td><td>EA</td><td>BY</td></tr><tr><td>PGD</td><td>ResNet-18</td><td>0.96</td><td>0.66</td><td>0.72</td><td>0.10</td><td>0.96</td><td>0.97</td><td>-</td></tr><tr><td>SQ</td><td>ResNet-18</td><td>0.96</td><td>0.66</td><td>0.79</td><td>0.09</td><td>0.10</td><td>0.90</td><td></td></tr><tr><td>PT</td><td>ResNet-18</td><td>0.95</td><td>0.66</td><td>0.72</td><td>0.08</td><td>0.15</td><td>0.88</td><td></td></tr><tr><td>AAD</td><td>ResNet-18</td><td>0.97</td><td>0.66</td><td>0.52</td><td>0.51</td><td>0.35</td><td>0.95</td><td></td></tr><tr><td>PGN</td><td>ResNet-18</td><td>0.95</td><td>0.66</td><td>0.64</td><td>0.24</td><td>0.02</td><td>0.93</td><td></td></tr><tr><td>BSR</td><td>ResNet-18</td><td>0.96</td><td>0.66</td><td>0.59</td><td>0.24</td><td>0.07</td><td>0.93</td><td></td></tr></table>

Table 16: F1 score comparison on ImageNet under various adversarial attacks. Best results are shown in bold, second best underlined.

## K.4 Sensitivity to the Prime-Resolution Parameter k

![](images/f9ce98f0750dc1c3d5d5a611c1ae1496812e8829d6bf8a32f2f9098e68e17107.jpg)  
Figure 10: Sensitivity of the detector to the prime-resolution parameter k.

Figure 10 shows how detection accuracy varies as the prime-resolution parameter k is changed. Across the entire tested range $( k = 3 \tan { k } = 8 )$ , the mean detection accuracy remains extremely stable, fluctuating only within a narrow interval of approximately 0.954–0.957. The shaded region captures variability across repeated runs and likewise remains tightly concentrated.

These results indicate that the detector is not sensitive to the specific choice of k. Even when k is varied over a relatively broad range, the performance remains effectively unchanged. This robustness reflects an intrinsic property of our method, i.e., once the prime gaps are sufficiently fine to induce stable absorption and injectivity behavior, further refinement provides little additional benefit. In practice, this means that k does not require fine-tuning to achieve strong performance, simplifying deployment across different datasets and architectures.

## K.5 Comparison with Alternative Quantization Schemes

To assess whether the advantages of our quantization scheme can be replicated by other non-uniform discretization strategies, we compare it against three alternatives: Fibonacci quantization, logarithmic quantization, and standard uniform quantization. All approaches are evaluated under the same experimental setup and across five random seeds (1, 3, 5, 21, 42). Table 17 reports detection accuracy for each seed, together with the mean and standard deviation.

Across all seeds, our method achieves the highest mean accuracy and the lowest variance, demonstrating both improved performance and greater stability compared to all alternative discretization schemes.

<table><tr><td rowspan="2">Quantization</td><td colspan="5">Seed</td><td rowspan="2"> $\mathbf { M e a n } \pm \mathbf { S t d }$ </td></tr><tr><td>1</td><td>3</td><td>5</td><td>21</td><td>42</td></tr><tr><td>Ours</td><td>0.94</td><td>0.92</td><td>0.95</td><td>0.94</td><td>0.96</td><td> ${ \bf 0 . 9 4 \pm 0 . 0 1 }$ </td></tr><tr><td>Fibonacci</td><td>0.62</td><td>0.61</td><td>0.55</td><td>0.60</td><td>0.68</td><td> $0 . 6 1 \pm 0 . 0 4$ </td></tr><tr><td>Logarithmic</td><td>0.70</td><td>0.61</td><td>0.61</td><td>0.56</td><td>0.58</td><td> $0 . 6 2 \pm 0 . 0 5$ </td></tr><tr><td>Uniform</td><td>0.65</td><td>0.65</td><td>0.60</td><td>0.56</td><td>0.65</td><td> $\underline { { 0 . 6 2 } } \pm 0 . 0 3$ </td></tr></table>

Table 17: Detection accuracy of different quantization schemes across five random seeds. Values reported using a ResNet backbone and CIFAR-10 under mixed attacks (PGD, FGSM, APGD).

Notably, replacing our scheme with Fibonacci or logarithmic quantization does not yield comparable results, despite also introducing non-uniform discretization. Uniform quantization performs similarly poorly. These results suggest that the structural properties of our secret prime quantization strategy are essential for producing a reliable Z–P discrepancy signal, and that generic discretization methods do not replicate this behavior in practice.

## K.6 Sensitivity Analysis of $k _ { \mathbf { l o c a l } }$ and $k _ { \mathbf { g l o b a l } }$

<table><tr><td> $k _ { \mathrm { l o c a l } }$ </td><td>Accuracy</td><td> $k _ { \mathbf { g l o b a l } }$ </td><td>Accuracy</td></tr><tr><td>8</td><td>97.90</td><td>3</td><td>97.90</td></tr><tr><td>10</td><td>86.15</td><td>5</td><td>95.60</td></tr><tr><td>12</td><td>92.57</td><td>7</td><td>83.45</td></tr><tr><td>15</td><td>93.24</td><td>10</td><td>85.35</td></tr></table>

Table 18: Detection accuracy as a function of the local neighborhood size $\overline { { k } } _ { \mathrm { l o c a l } }$ (left) and the global neighborhood size $k _ { \mathrm { g l o b a l } } \mathrm { ( r i g h t ) }$

We evaluate the sensitivity of our detector to the neighborhood parameters $k _ { \mathrm { l o c a l } }$ and $k _ { \mathrm { g l o b a l } }$ , which govern the local consistency scale and the global support for GW coupling, respectively.

For the local neighborhood size, we test $k _ { \mathrm { l o c a l } } \in \{ 8 , 1 0 , 1 2 , 1 5 \}$ . Accuracy remains consistently high across all settings, with the best performance (97.9%) obtained at $k _ { \mathrm { l o c a l } } = 8$ . While performance dips for $k _ { \mathrm { l o c a l } } = 1 0 .$ , it recovers for larger values (12 and 15), indicating that the method is broadly robust to the choice of local scale.

For the global neighborhood size, we test $k _ { \mathrm { g l o b a l } } \ \in \ \{ 3 , 5 , 7 , 1 0 \}$ . Smaller global neighborhoods yield the best performance, with $k _ { \mathrm { g l o b a l } } = 3$ achieving 97.9%. Larger values gradually degrade performance, suggesting that excessively large global supports may introduce noise or dilute the structural alignment captured during cross-space GW coupling.

Overall, the detector displays stable performance across a wide range of neighborhood sizes, with optimal performance achieved at smaller values. Based on this analysis, we adopt $k _ { \mathrm { l o c a l } } = 8$ and $k _ { \mathrm { g l o b a l } } = 3$ for all main experiments.

## K.7 Robustness Against Common Corruptions

To assess whether the detector responds specifically to adversarial perturbations, rather than generic input noise, we evaluate its behaviour under benign corruptions using the CIFAR-C benchmark Hendrycks & Dietterich (2019a). CIFAR-C includes a diverse set of naturally occurring degradation types, such as Gaussian, shot, and impulse noise; blur corruptions (defocus, frosted glass, motion, zoom); weather effects (snow, frost, fog); brightness and contrast shifts; elastic distortions; pixelation; and JPEG compression.

Table 19 reports results for the Gaussian noise corruption. The detector maintains high accuracy and AUROC, indicating that benign perturbations do not trigger the characteristic Z–P discrepancies associated with adversarial attacks. This supports our claim that the method does not misclassify natural corruptions as adversarial.

<table><tr><td>Corruption Type</td><td>Detection Accuracy</td><td>AUROC</td><td>Precision / Recall / F1</td></tr><tr><td>Gaussian Noise</td><td>96.43%</td><td>0.99</td><td>0.97 / 0.96 / 0.96</td></tr></table>

Table 19: Performance of our detector under Gaussian noise (CIFAR-C).

<table><tr><td>Method</td><td>Runtime (s)</td><td>CPU (GB)</td><td>GPU (GB)</td></tr><tr><td>BY</td><td>0.152</td><td>2.00</td><td>0.60</td></tr><tr><td>EA</td><td>0.115</td><td>0.20</td><td>11.80</td></tr><tr><td>MD</td><td>0.141</td><td>0.68</td><td>0.10</td></tr><tr><td>FS</td><td>0.005</td><td>0.25</td><td>0.17</td></tr><tr><td>MAD</td><td>0.105</td><td>3.30</td><td>0.00</td></tr><tr><td>MN</td><td>0.012</td><td>0.78</td><td>0.10</td></tr><tr><td>Ours</td><td>0.120</td><td>0.62</td><td>2.04</td></tr></table>

Table 20: Runtime and memory usage of different adversarial detection methods on CIFAR-10. Best values in bold, second-best underlined.

## K.8 Runtime and Memory Efficiency

Table 20 compares inference-time and memory footprint across several recent adversarial detection baselines. Our detector achieves a competitive runtime of 0.12 seconds per sample while maintaining a moderate CPU and GPU memory footprint. Methods such as BY He et al. (2024) require substantially larger memory because they process multiple transformed copies of each input, resulting in expanded intermediate activations. In contrast, our detector relies on a single forward pass through a ResNet backbone, leading to a modest GPU footprint.

Lightweight approaches such as FS Xu et al. (2018) and MN Meng & Chen (2017) offer faster runtimes but exhibit either higher memory usage or weaker robustness. Overall, our method strikes an effective balance among runtime efficiency, memory consumption, and detection performance.

## K.9 Robustness and Generalization Capabilities

## K.9.1 Reliability Analysis

We study detector reliability across a range of perturbation levels using TPR/FPR heatmaps (Fig. 11). With dataset and backbone fixed, only adversarial conditions vary, revealing how consistently detectors identify adversarial inputs while avoiding false positives on clean samples.

![](images/b55eb68c1b24d22613f715757ebd71122ecec3fb3f0b41dc3c428deb4044f97b.jpg)  
(a) TPR Heatmap (higher is better).

![](images/572cdb58941d2108ad99ed1e52a49bb15b5f7fc9bd473cbb971df8bdc7afe57c.jpg)  
(b) FPR Heatmap (lower is better).  
Figure 11: Reliability via TPR/FPR heatmaps. X-axis: attack types with perturbation levels, Y-axis: defenses. Thresholds are calibrated only on clean samples, so baselines yield constant FPR across attacks.

## K.9.2 Cross-Attack Generalization

To evaluate generalization to unseen attacks, we train on ResNet18 adversarial samples from FG and PGD, then test on CW, SQ, SA, and PT. Results (Table 21) report accuracy, adversarial recall, and adversarial precision. Our method consistently outperforms baselines, demonstrating robustness to unseen attack families.

<table><tr><td>Attack</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td></tr><tr><td>CW</td><td>97.5/100/95</td><td> $7 1 / 5 1 / 8 6$ </td><td>90/90/90</td><td>49.5/69/46.6</td><td> $6 0 / 4 5 / 6 3$ </td></tr><tr><td>SQ</td><td>93.5/98/89</td><td>88/85/91</td><td>94.5/99/90.8</td><td>48.5/63/47</td><td>45/15/46</td></tr><tr><td>SP</td><td>95.5/99/92</td><td>86/80/90</td><td>92.5/95/90.5</td><td>49/67/49</td><td>48/21/46</td></tr><tr><td>PT</td><td>95/100/90</td><td>91/90/91</td><td>89.5/89/89.9</td><td>49/67/49</td><td>45/15/37</td></tr></table>

Table 21: Cross-Attack Generalization. Accuracy / $\overline { { \mathrm { R e c a l l } _ { \mathrm { a d v } } } }$ / Precision<sub>adv</sub> (%).

## K.9.3 Cross-Model Generalization

We also test transfer robustness by training on ResNet18 attacks and evaluating on FourierNet (FNet) adversarial samples. Table 22 reports detection accuracy, recall, and precision. Our method again outperforms baselines, showing resilience to model transfer attacks.

<table><tr><td>Attack</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td></tr><tr><td>CW</td><td>95/97/94</td><td>54/17/68</td><td>73.5/91/67.4</td><td>50/69/50</td><td>50/25/52</td></tr><tr><td>SQ</td><td>87/93/83</td><td>51/10/55</td><td>77.0/98/69.0</td><td>49/67/49</td><td>52/29/54</td></tr><tr><td>SP</td><td>94/93/95</td><td>64/37/82</td><td>74/92/67.6</td><td>49/67/49</td><td>53/31/56</td></tr><tr><td>PT</td><td>96/95/97</td><td>57/22/73</td><td>72.5/89/66.9</td><td>47/64/48</td><td>49/22/47</td></tr><tr><td>FG</td><td>93/99/89</td><td>55/18/69</td><td>70.0/84/65.6</td><td>46/61/47</td><td>50/25/50</td></tr><tr><td>PGD</td><td>98/99/99</td><td>56/21/72</td><td>74.5/93/67.9</td><td>50/69/50</td><td>50/26/52</td></tr></table>

Table 22: Cross-Model Generalization. Accuracy $\overline { { / \ R e c a l l _ { \mathrm { a d v } } } }$ / Precision<sub>adv</sub> (%).

## K.10 True Positive Rate (TPR) Analysis

Table 23 reports the true positive rate (TPR) across adversarial attack types, i.e., the fraction of adversarial inputs correctly detected as adversarial. The formula is, $\begin{array} { r } { \mathrm { T P R } = \frac { \mathrm { d e t e c t e d a d v e r s a r i a l } } { \mathrm { a l l a d v e r s a r i a l } } } \end{array}$ . Bold entries denote the best-performing method, and underlined entries denote the second best. Our method consistently achieves the highest TPR in 12 out of 13 attacks, showing large margins especially for transfer-based (UP) and perceptual/frequency attacks (FA, PF, SR). The only exception is Gaussian blur (GB), where all detectors struggle, but our method still provides a clear advantage over baselines. These results highlight that our cross-space framework is particularly effective in reliably flagging adversarial samples, even under challenging attack families.

<table><tr><td>Attack / Method</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td></tr><tr><td>AA</td><td> $\mathbf { 9 7 . 4 0 \pm 1 . 0 7 }$ </td><td> $\overline { { 5 0 . 6 0 \pm 1 . 9 9 } }$ </td><td> $\overline { { 8 6 . 5 0 \pm 1 . 2 4 } }$ </td><td> $\overline { { 5 7 . 9 6 \pm 3 . 1 0 } }$ </td><td> $\overline { { 7 4 . 0 9 \pm 1 . 2 6 } }$ </td></tr><tr><td>CW</td><td> ${ \bf 9 6 . 3 3 \pm 0 . 5 7 }$ </td><td> $5 9 . 9 0 \pm 0 . 6 5$ </td><td> $\overline { { 9 3 . 2 0 \pm 0 . 1 4 } }$ </td><td> $5 6 . 8 2 \pm 1 . 5 2$ </td><td> $3 9 . 3 0 \pm 1 . 6 0$ </td></tr><tr><td>PT</td><td> ${ \bf 9 7 . 7 3 \pm 0 . 3 4 }$ </td><td> $8 5 . 5 0 \pm 0 . 8 5$ </td><td> $\overline { { 5 6 . 7 0 \pm 2 . 4 9 } }$ </td><td> $5 5 . 3 3 \pm 3 . 9 2$ </td><td> $4 0 . 5 0 \pm 1 . 6 6$ </td></tr><tr><td>PGD</td><td> ${ \bf 9 7 . 2 0 \pm 1 . 0 7 }$ </td><td> $9 5 . 3 0 \pm 0 . 6 2$ </td><td> $6 9 . 9 0 \pm 2 . 6 8$ </td><td> $5 6 . 2 2 \pm 1 . 1 8$ </td><td> $8 7 . 9 0 \pm 0 . 2 9$ </td></tr><tr><td>SA</td><td> ${ \bf 9 6 . 6 7 \pm 0 . 2 3 }$ </td><td> $\overline { { 6 9 . 0 0 \pm 1 . 9 1 } }$ </td><td> $6 9 . 9 0 \pm 0 . 1 7$ </td><td> $3 6 . 2 0 \pm 0 . 6 2$ </td><td> $3 5 . 8 0 \pm 2 . 8 6$ </td></tr><tr><td>SQ</td><td> $9 6 . 6 7 \pm 0 . 9 0$ </td><td> $9 1 . 2 0 \pm 0 . 7 7$ </td><td> ${ \bf 9 8 . 2 0 \pm 1 . 1 2 }$ </td><td> $5 5 . 9 0 \pm 3 . 3 4$ </td><td> $1 4 . 8 0 \pm 1 . 1 2$ </td></tr><tr><td>UP</td><td> $\pm \overline { { \mathbf { 7 . 0 0 } \pm \mathbf { 1 . 1 4 } } }$ </td><td> $4 5 . 6 0 \pm 3 . 0 6$ </td><td> $2 8 . 5 0 \pm 2 . 5 2$ </td><td> $5 5 . 8 0 \pm 2 . 1 2$ </td><td> $2 1 . 7 0 \pm 1 . 1 7$ </td></tr><tr><td>AP</td><td> ${ \bf 9 6 . 2 2 \pm 1 . 1 1 }$ </td><td> $4 9 . 3 0 \pm 1 . 6 5$ </td><td> $8 3 . 9 0 \pm 1 . 4 4$ </td><td> $\overline { { 5 4 . 2 8 \pm 1 . 5 9 } }$ </td><td> $7 3 . 1 0 \pm 1 . 5 1$ </td></tr><tr><td>FG</td><td> ${ \bf 9 6 . 8 7 \pm 1 . 5 7 }$ </td><td> $6 0 . 3 0 \pm 2 . 6 4$ </td><td> $4 2 . 9 0 \pm 3 . 2 6$ </td><td> $5 3 . 0 9 \pm 0 . 7 2$ </td><td> $1 5 . 3 0 \pm 1 . 5 7$ </td></tr><tr><td>FA</td><td> ${ \bf 9 4 . 3 3 \pm 1 . 2 3 }$ </td><td> $\overline { { 1 2 . 5 0 \pm 0 . 8 8 } }$ </td><td> $2 1 . 1 0 \pm 1 . 2 3$ </td><td> $5 3 . 3 4 \pm 3 . 0 8$ </td><td> $2 5 . 5 0 \pm 1 . 7 8$ </td></tr><tr><td>GB</td><td> $\mathbf { 7 3 . 0 0 \pm 1 0 . 2 7 }$ </td><td> $1 2 . 4 0 \pm 1 . 9 2$ </td><td> $2 4 . 5 0 \pm 0 . 5 3$ </td><td> $5 0 . 4 2 \pm 2 . 3 8$ </td><td> $2 3 . 1 0 \pm 2 . 0 6$ </td></tr><tr><td>PF</td><td> ${ \bf 9 6 . 3 3 \pm 0 . 5 7 }$ </td><td> $1 6 . 1 0 \pm 1 . 2 0$ </td><td> $2 4 . 0 0 \pm 0 . 4 6$ </td><td> $\overline { { 5 3 . 0 1 \pm 3 . 6 5 } }$ </td><td> $2 4 . 7 0 \pm 1 . 2 1$ </td></tr><tr><td>SR</td><td> $\mathbf { 9 5 . 3 3 \pm 1 . 0 7 }$ </td><td> $1 3 . 3 0 \pm 0 . 1 9$ </td><td> $2 6 . 6 0 \pm 1 . 2 4$ </td><td> $\overline { { 5 4 . 2 0 \pm 2 . 3 7 } }$ </td><td> $2 4 . 4 0 \pm 2 . 2 9$ </td></tr></table>

Table 23: True positive rate (%) on adversarial samples: This table shows the results of TPR measured on adversarial samples.

<table><tr><td rowspan=1 colspan=1>Attack</td><td rowspan=1 colspan=5>Ours           MD           MN</td></tr><tr><td rowspan=1 colspan=1>AA4</td><td rowspan=1 colspan=5> $\mathbf { \overline { { 9 7 . 6 0 \pm 0 . 2 2 } } }$    $\overline { { 5 0 . 0 0 \pm 0 . 5 6 } }$    $\overline { { 4 9 . 9 0 \pm 1 . 5 9 } }$ </td></tr><tr><td rowspan=1 colspan=1>AA8</td><td rowspan=1 colspan=2> ${ \bf 9 6 . 4 3 \pm 0 . 5 9 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 0 . 4 0 \pm 0 . 8 6 } }$ </td><td rowspan=1 colspan=2> $5 0 . 1 0 \pm 1 . 7 5$ </td></tr><tr><td rowspan=2 colspan=1>CWPT7</td><td rowspan=1 colspan=2> ${ \bf 9 6 . 1 0 \pm 1 . 6 3 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 1 . 6 0 \pm 0 . 6 7 } }$ </td><td rowspan=1 colspan=2> $5 4 . 0 5 \pm 1 . 4 3$ </td></tr><tr><td rowspan=1 colspan=2> ${ \bf 9 8 . 6 3 \pm 0 . 1 7 }$ </td><td rowspan=1 colspan=1> $8 9 . 4 5 \pm 0 . 4 5$ </td><td rowspan=1 colspan=2> $\overline { { 9 0 . 2 0 \pm 0 . 3 2 } }$ </td></tr><tr><td rowspan=1 colspan=1>PT14</td><td rowspan=1 colspan=2> ${ \bf 9 4 . 1 0 \pm 0 . 4 9 }$ </td><td rowspan=1 colspan=1> $9 0 . 2 0 \pm 0 . 4 3$ </td><td rowspan=1 colspan=2> $\overline { { 9 0 . 0 9 \pm 0 . 3 2 } }$ </td></tr><tr><td rowspan=1 colspan=1>PGD</td><td rowspan=1 colspan=2> ${ \bf 9 7 . 6 3 \pm 0 . 5 3 }$ </td><td rowspan=1 colspan=1> $\overline { { 4 9 . 8 0 \pm 0 . 6 7 } }$ </td><td rowspan=1 colspan=2> $4 9 . 9 0 \pm 1 . 8 0 $ </td></tr><tr><td rowspan=1 colspan=1>PGD8</td><td rowspan=1 colspan=2> ${ \bf 9 5 . 3 3 \pm 0 . 9 7 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 1 . 4 5 \pm 0 . 6 9 } }$ </td><td rowspan=1 colspan=1> $5 0 . 3 0 \pm 1 . 8 2$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SA</td><td rowspan=1 colspan=2> ${ \bf 9 7 . 2 1 \pm 1 . 3 6 }$ </td><td rowspan=1 colspan=1> $\overline { { 8 0 . 8 0 \pm 2 . 0 1 } }$ </td><td rowspan=1 colspan=1> $5 3 . 1 0 \pm 0 . 1 5$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SQ</td><td rowspan=1 colspan=2> ${ \bf 9 7 . 2 0 \pm 0 . 5 7 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 0 . 0 5 \pm 0 . 5 5 } }$ </td><td rowspan=1 colspan=1> $5 0 . 0 0 \pm 0 . 1 8$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SQ8</td><td rowspan=1 colspan=2> $\mathbf { 9 6 . 4 7 \pm 4 . 0 1 }$ </td><td rowspan=1 colspan=1> $5 2 . 2 0 \pm 0 . 6 7$ </td><td rowspan=1 colspan=1> $5 0 . 7 3 \pm 0 . 5 2$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>UP</td><td rowspan=1 colspan=1> ${ \bf 9 9 . 8 7</td><td rowspan=1 colspan=1>\pm 0 . 0 5 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 0 . 0 5 \pm 0 . 7 1 } }$ </td><td rowspan=1 colspan=1> $5 0 . 2 3 \pm 1 . 8 5$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>UP8</td><td rowspan=1 colspan=1> ${ \bf 9 8 . 3 0</td><td rowspan=1 colspan=1>\pm 1 . 2 7 }$ </td><td rowspan=1 colspan=1> $5 1 . 1 5 \pm 0 . 5 5$ </td><td rowspan=1 colspan=1> $\overline { { 5 0 . 6 5 \pm 1 . 5 0 } }$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>UP12</td><td rowspan=1 colspan=1> ${ \bf 9 9 . 2 3</td><td rowspan=1 colspan=1>\pm 0 . 6 8 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 4 . 7 5 \pm 1 . 3 7 } }$ </td><td rowspan=1 colspan=1> $5 1 . 8 0 \pm 1 . 2 3 $ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>AP</td><td rowspan=1 colspan=1>98.63</td><td rowspan=1 colspan=1>± 0.34</td><td rowspan=1 colspan=1> $\underline { { 4 9 . 7 0 \pm 0 . 6 5 } }$ </td><td rowspan=1 colspan=1> $4 9 . 4 9 \pm 1 . 7 4$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>AP8</td><td rowspan=1 colspan=1> $\mathbf { 9 9 .</td><td rowspan=1 colspan=1>5 0 \pm 0 . 3 6 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 0 . 0 5 \pm 0 . 9 9 } }$ </td><td rowspan=1 colspan=1> $5 0 . 2 0 \pm 1 . 8 3$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FG</td><td rowspan=1 colspan=1> ${ \bf 9 7 . 2 7</td><td rowspan=1 colspan=1>\pm 3 . 4 4 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 9 . 9 5 \pm 0 . 6 7 } }$ </td><td rowspan=1 colspan=1> $4 9 . 9 5 \pm 1 . 6 6$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FG8</td><td rowspan=1 colspan=1> ${ \bf 9 5 . 5 3</td><td rowspan=1 colspan=1>\pm 0 . 6 2 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 1 . 8 5 \pm 0 . 8 1 } }$ </td><td rowspan=1 colspan=1> $5 0 . 7 8 \pm 1 . 7 3$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FA</td><td rowspan=1 colspan=1> ${ \bf 9 7 . 6 5</td><td rowspan=1 colspan=1>\pm 0 . 7 8 }$ </td><td rowspan=1 colspan=1> $\overline { { 6 8 . 9 0 \pm 0 . 3 0 } }$ </td><td rowspan=1 colspan=2> $4 5 . 6 5 \pm 0 . 9 0$ </td></tr><tr><td rowspan=1 colspan=1>GB</td><td rowspan=1 colspan=1> ${ \bf 8 7 . 2 7</td><td rowspan=1 colspan=1>\pm 0 . 3 2 }$ </td><td rowspan=1 colspan=1> $\overline { { 6 8 . 1 2 \pm 0 . 4 9 } }$ </td><td rowspan=1 colspan=2> $4 1 . 6 0 \pm 0 . 1 0$ </td></tr><tr><td rowspan=1 colspan=1>PF</td><td rowspan=1 colspan=1> ${ \bf 9 2 . 2 2</td><td rowspan=1 colspan=1>\pm 0 . 4 5 }$ </td><td rowspan=1 colspan=1> $6 9 . 6 8 \pm 0 . 4 2$ </td><td rowspan=1 colspan=2> $4 8 . 8 5 \pm 1 . 1 7$ </td></tr><tr><td rowspan=1 colspan=1>SR</td><td rowspan=1 colspan=2> ${ \bf 9 7 . 6 5 \pm 0 . 7 6 }$ </td><td rowspan=1 colspan=1> $\overline { { 7 0 . 3 0 \pm 0 . 7 7 } }$ </td><td rowspan=1 colspan=2> $4 1 . 9 9 \pm 1 . 4 3$ </td></tr></table>

Table 24: The results of binary accuracy on FMNIST using ResNet18.

<table><tr><td rowspan=1 colspan=1>Attack</td><td rowspan=1 colspan=6>Ours           MD           MN</td></tr><tr><td rowspan=1 colspan=1>AA</td><td rowspan=1 colspan=2> $\mathbf { 9 7 . 3 3 \pm 0 . 3 3 }$ </td><td rowspan=1 colspan=4> $\overline { { 5 0 . 1 0 \pm 1 . 9 1 } }$     $\overline { { 5 0 . 0 0 \pm 1 . 0 0 } }$ </td></tr><tr><td rowspan=1 colspan=1>AA8</td><td rowspan=1 colspan=2> ${ \bf 9 8 . 0 8 \pm 0 . 2 6 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 0 . 3 0 \pm 1 . 8 4 } }$ </td><td rowspan=1 colspan=3> $5 0 . 0 0 \pm 0 . 9 8$ </td></tr><tr><td rowspan=1 colspan=1>PT7</td><td rowspan=1 colspan=2> ${ \bf 9 6 . 8 7 \pm 0 . 3 5 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 3 . 3 5 \pm 0 . 8 6 } }$ </td><td rowspan=1 colspan=3> $7 5 . 8 \pm 0 . 7 2$ </td></tr><tr><td rowspan=1 colspan=1>PT14</td><td rowspan=1 colspan=2> ${ \bf 9 7 . 6 3 \pm 0 . 4 6 }$ </td><td rowspan=1 colspan=1> $5 2 . 2 0 \pm 0 . 3 1$ </td><td rowspan=1 colspan=3> $8 6 . 2 7 \pm 0 . 2 7$ </td></tr><tr><td rowspan=1 colspan=1>PGD</td><td rowspan=1 colspan=2> ${ \bf 9 8 . 1 0 \pm 0 . 4 5 }$ </td><td rowspan=1 colspan=1> $5 0 . 9 5 \pm 1 . 7 2$ </td><td rowspan=1 colspan=3> $\overline { { 4 9 . 4 5 \pm 0 . 6 8 } }$ </td></tr><tr><td rowspan=1 colspan=1>PGD8</td><td rowspan=1 colspan=2> ${ \bf 9 8 . 6 0 \pm 0 . 3 6 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 1 . 8 0 \pm 1 . 4 7 } }$ </td><td rowspan=1 colspan=2> $4 9 . 2 5 \pm 0 . 6 9$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SA</td><td rowspan=1 colspan=2> ${ \bf 9 8 . 1 0 \pm 1 . 8 5 }$ </td><td rowspan=1 colspan=1> $5 0 . 4 5 \pm 0 . 9 8$ </td><td rowspan=1 colspan=2> $5 6 . 5 0 \pm 0 . 3 9$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SQ</td><td rowspan=1 colspan=2> ${ \bf 9 8 . 7 7 \pm 0 . 0 5 }$ </td><td rowspan=1 colspan=1> $5 1 . 1 5 \pm 1 . 4 9$ </td><td rowspan=1 colspan=2> $\overline { { 4 8 . 8 5 \pm 0 . 9 6 } }$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SQ8</td><td rowspan=1 colspan=2> ${ \bf 9 5 . 3 2 \pm 3 . 5 1 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 2 . 2 5 \pm 1 . 4 8 } }$ </td><td rowspan=1 colspan=2> $4 8 . 8 5 \pm 0 . 8 5$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>UP</td><td rowspan=1 colspan=1> ${ \bf 9 9 . 8 7</td><td rowspan=1 colspan=1>\pm 0 . 0 5 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 1 . 1 5 \pm 0 . 6 1 } }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $4 9 . 6 0 \pm 0 . 6 7$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>UP8</td><td rowspan=1 colspan=1> ${ \bf 9 8 . 3 0</td><td rowspan=1 colspan=1>\pm 1 . 2 7 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 2 . 0 5 \pm 1 . 4 1 } }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $4 9 . 9 0 \pm 0 . 1 0$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>UP12</td><td rowspan=1 colspan=1> ${ \bf 9 9 . 4</td><td rowspan=1 colspan=1>1 \pm 0 . 0 5 }$ </td><td rowspan=1 colspan=1>53.10 ± 1.47</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $5 0 . 0 5 \pm 0 . 7 0$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>AP</td><td rowspan=1 colspan=1> ${ \bf 9 8 . 6 3</td><td rowspan=1 colspan=1>\pm 0 . 3 4 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 0 . 6 0 \pm 1 . 7 8 } }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $4 9 . 6 5 \pm 0 . 6 8$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>AP8</td><td rowspan=1 colspan=1> $\mathbf { 9 9</td><td rowspan=1 colspan=1>. 5 0 \pm 0 . 3 6 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 1 . 3 5 \pm 1 . 3 6 } }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $4 9 . 2 5 \pm 0 . 6 9$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FG</td><td rowspan=1 colspan=2> ${ \bf 9 7 . 5 7 \pm 1 . 5 4 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 1 . 1 5 \pm 1 . 5 4 } }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $4 9 . 6 5 \pm 0 . 6 8$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FG8</td><td rowspan=1 colspan=2> ${ \bf 9 9 . 4 7 \pm 0 . 2 5 }$ </td><td rowspan=1 colspan=1>52.05 ± 1.71</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $4 8 . 7 0 \pm 0 . 1 0$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FA</td><td rowspan=1 colspan=1> ${ \bf 9 8 . 1</td><td rowspan=1 colspan=1>6 \pm 1 . 7 6 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 0 . 0 0 \pm 1 . 8 5 } }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $4 9 . 9 5 \pm 0 . 9 1$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>GB</td><td rowspan=1 colspan=1> ${ \bf 9 3 . 8</td><td rowspan=1 colspan=1>6 \pm 0 . 3 4 }$ </td><td rowspan=1 colspan=1> $3 8 . 8 5 \pm 1 . 5 6$ </td><td rowspan=1 colspan=2> $\underline { { 4 1 . 1 5 \pm 1 . 0 4 } }$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>PF</td><td rowspan=1 colspan=2> ${ \bf 9 6 . 8 7 \pm 0 . 6 5 }$ </td><td rowspan=1 colspan=1> $4 9 . 0 0 \pm 1 . 9 6$ </td><td rowspan=1 colspan=2> ${ \underline { { 5 2 . 2 0 \pm 0 . 7 4 } } }$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SR</td><td rowspan=1 colspan=2> ${ \bf 9 8 . 1 0 \pm 1 . 8 5 }$ </td><td rowspan=1 colspan=3> $4 8 . 9 5 \pm 1 . 5 3$    $\overline { { 5 2 . 1 2 \pm 0 . 6 6 } }$ </td><td rowspan=1 colspan=1></td></tr></table>

Table 25: The results of binary accuracy on KMNIST using ResNet18.

## K.11 Experiments on FMNIST and KMNIST

Table 24 and Table 25 show the result of the detection accuracy on FMNIST and KMNIST respectively. Regarding the performance of MD on FMNIST and KMNIST, we observed that the distributions of Mahalanobis scores for clean and adversarial samples did not show significant difference. This is because clean and adversarial features for FMNIST and KMNIST are fairly similar in the feature space of ResNet18, making it difficult for MD to distinguish between clean and adversarial samples. The similar feature representations in these datasets limit the separability of the two distributions of Mahalanobis scores. A similar phenomenon regarding the distributions of clean and adversarial features used for detection was observed in MN.

## K.12 Zero-Shot Setting

Adversarial robustness in large-scale Vision–Language Models (VLMs) remains relatively underexplored, especially in the zero-shot regime where models are accessed only through APIs and adversaries rely on transfer attacks. This provides a natural testbed for evaluating cross-model generalization, since neither gradients nor model parameters are available.

We evaluate the zero-shot transferability of our detector across two recent $\mathrm { V L M s \_ L L a V A } { - 1 . 5 - }$ 7B (Liu et al., 2023) and Qwen-2.7B-VL Yang et al. (2024)—on three diverse datasets: CalTech-101 Fei-Fei et al. (2004), Food-101 Bossard et al. (2014), and CalTech-256 Griffin et al. (2007).

<table><tr><td rowspan="2">Attack</td><td rowspan="2">Dataset</td><td colspan="2">Detection Accuracy / AUC</td><td colspan="2">Precision / Recall / F1</td></tr><tr><td>LLaVA-1.5</td><td>Qwen-2.7B-VL</td><td>LLaVA-1.5</td><td>Qwen-2.7B-VL</td></tr><tr><td rowspan="3">APGD</td><td>CalTech-101</td><td>89.50 / 0.99</td><td>89.63 / 0.99</td><td>0.89 / 0.89 / 0.89</td><td>0.89 / 0.89 / 0.89</td></tr><tr><td>Food-101</td><td>90.44 / 0.99</td><td>87.72 / 0.99</td><td>0.90 / 0.90 / 0.90</td><td>0.87 / 0.87 / 0.87</td></tr><tr><td>CalTech-256</td><td>88.60 / 0.95</td><td>87.83 / 0.99</td><td>0.88 / 0.88 / 0.88</td><td>0.87 / 0.87 / 0.87</td></tr><tr><td rowspan="3">PGD</td><td>CalTech-101</td><td>89.50 / 0.99</td><td>87.13 / 0.99</td><td>0.89 / 0.89 / 0.89</td><td>0.87 / 0.87 / 0.86</td></tr><tr><td>Food-101</td><td>87.13 / 0.99</td><td>88.42 / 0.99</td><td>0.87 / 0.87 / 0.87</td><td>0.88 / 0.88 / 0.88</td></tr><tr><td>CalTech-256</td><td>83.00 / 0.91</td><td>88.76 / 0.99</td><td>0.83 / 0.83 / 0.82</td><td>0.88 / 0.88 / 0.88</td></tr><tr><td rowspan="3">FGSM</td><td>CalTech-101</td><td>90.80 / 0.99</td><td>89.20 / 0.99</td><td>0.90 / 0.90 / 0.90</td><td>0.89 / 0.89 / 0.89</td></tr><tr><td>Food-101</td><td>87.08 / 0.94</td><td>90.74 / 0.99</td><td>0.84 / 0.81 / 0.81</td><td>0.90 / 0.90 / 0.90</td></tr><tr><td>CalTech-256</td><td>85.50 / 0.92</td><td>88.07 / 0.99</td><td>0.85 / 0.85 / 0.85</td><td>0.88 / 0.88 / 0.88</td></tr></table>

Table 26: Zero-shot adversarial detection performance on LLaVA-1.5 and Qwen-2.7B-VL across multiple datasets and attacks.

Adversarial examples are generated using PGD, APGD, and FGSM following (Cui et al., 2024), and CLIP (Radford et al., 2021) embeddings define the Z-space. Table 26 reports detection accuracy, AUROC, and precision/recall/F1 metrics for all model–dataset combinations.

Across all attacks and datasets, the detector achieves strong zero-shot transferability: detection accuracies are consistently $\geq 8 3 \%$ , AUROC values $\geq 0 . 9 4 \ : ( \mathrm { { \bar { o } f t e n } \geq 0 . 9 9 ) }$ , and precision/recall/F1 scores typically remain $\ge 0 . 8 7$ . These results indicate robust generalization across VLM architectures without requiring access to model internals.

The robustness stems from adversarial perturbations disrupting semantic alignment in Z while being unevenly absorbed in P, yielding a persistent cross-space discrepancy detectable even under transfer. Additional robustness and generalization results—including TPR/FPR heatmaps, cross-attack transfer, and cross-model generalization—are provided in Appendix K.

## L Comparison of Our Proposed Defence Against Baseline Defences on CIFAR-10 Dataset

## L.1 Binary Accuracy Comparison

In this section we present the binary accuracy results for detecting adversarial attacks on CIFAR-10. This metric measures how well each defense method can correctly classify samples as either adversarial or clean. Our proposed method consistently outperforms all baselines across all attack types, achieving accuracy rates between 96.97% and 97.83%. The Mahalanobis Detector (MD) shows the second-best performance for most attacks, while Feature Squeezing (FS) performs well on specific attack types like CW and PGD4. The MAD and MN methods show poor performance with accuracies around 50%, essentially equivalent to random guessing. The standard deviations for our method are consistently low (0.22% to 0.65%), indicating stable and reliable performance.

<table><tr><td>Attack / Metric</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td></tr><tr><td>AA8</td><td> $\mathbf { 9 7 . 7 3 \pm 0 . 4 7 }$ </td><td> $\overline { { 9 0 . 4 0 \pm 1 . 4 } }$ </td><td> $7 9 . 3 5 \pm 0 . 8 4$ </td><td> $\overline { { 5 0 . 8 5 \pm 0 . 2 0 } }$ </td><td> $\overline { { 7 9 . 0 4 \pm 1 . 0 3 } }$ </td></tr><tr><td>CW</td><td> ${ \bf 9 6 . 9 7 \pm 0 . 6 1 }$ </td><td> $7 3 . 5 5 \pm 2 . 1 3$ </td><td> $8 6 . 0 0 \pm 0 . 5 6$ </td><td> $5 1 . 4 3 \pm 2 . 1 6$ </td><td> $5 6 . 6 5 \pm 1 . 7 1$ </td></tr><tr><td>PT8</td><td> ${ \bf 9 7 . 3 0 \pm 0 . 2 2 }$ </td><td> $5 2 . 8 0 \pm 1 . 9 1 $ </td><td> $\overline { { 5 2 . 0 0 \pm 0 . 5 5 } }$ </td><td> $4 3 . 4 3 \pm 2 . 3 2$ </td><td> $5 1 . 0 9 \pm 0 . 9 5$ </td></tr><tr><td>PGD</td><td> ${ \bf 9 7 . 5 7 \pm 0 . 5 4 }$ </td><td> $7 1 . 2 5 \pm 1 . 6 0$ </td><td> $\underline { { 8 1 . 1 0 \pm 1 . 0 7 } }$ </td><td> $4 8 . 9 1 \pm 1 . 3 3$ </td><td> $7 6 . 0 5 \pm 0 . 8 3$ </td></tr><tr><td>SQ</td><td> ${ \bf 9 7 . 3 0 \pm 0 . 6 5 }$ </td><td> $8 6 . 8 5 \pm 1 . 8 1$ </td><td> $8 4 . 7 0 \pm 1 . 2 3$ </td><td> $4 9 . 1 1 \pm 1 . 3 2$ </td><td> $4 4 . 4 5 \pm 1 . 4 8$ </td></tr><tr><td>UP</td><td> $\mathbf { 9 7 . 8 3 \pm 0 . 5 0 }$ </td><td> $\overline { { 6 6 . 4 0 \pm 2 . 4 0 } }$ </td><td> $5 3 . 6 5 \pm 1 . 2 3$ </td><td> $5 0 . 9 2 \pm 1 . 8 6$ </td><td> $4 7 . 8 5 \pm 1 . 8 3$ </td></tr><tr><td>AP8</td><td> ${ \bf 9 7 . 5 0 \pm 0 . 2 2 }$ </td><td> $9 0 . 0 \pm 2 . 1 7$ </td><td> $7 9 . 2 0 \pm 1 . 4 9$ </td><td> $5 0 . 8 7 \pm 1 . 3$ </td><td> $7 9 . 3 0 \pm 0 . 4 1$ </td></tr><tr><td>FG8</td><td> ${ \bf 9 7 . 7 3 \pm 0 . 4 2 }$ </td><td> $8 5 . 8 5 \pm 1 . 3 4$ </td><td> $6 3 . 3 5 \pm 2 . 1 8$ </td><td> $5 0 . 6 8 \pm 2 . 0 7$ </td><td> $4 4 . 4 0 \pm 1 . 5 4$ </td></tr></table>

Table 27: Comparison of detection performance (%) under different adversarial attacks: Bold values indicate the best performance, and underlined values denote the second-best.

<table><tr><td>Attack / Metric</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td></tr><tr><td>AA8</td><td> $\mathbf { \overline { { 9 6 . 6 0 \pm 0 . 8 6 } } }$ </td><td> $\overline { { 9 3 . 5 6 \pm 1 . 5 1 } }$ </td><td>79.90 ± 1.72</td><td> $\overline { { 5 5 . 6 7 \pm 2 . 4 9 } }$ </td><td>84.10 ± 1.03</td></tr><tr><td>CW</td><td> ${ \bf 9 6 . 3 3 \pm 0 . 5 7 }$ </td><td> $\overline { { 5 9 . 9 0 \pm 0 . 6 5 } }$ </td><td> $9 3 . 2 0 \pm 0 . 1 4$ </td><td> $5 6 . 8 2 \pm 1 . 5 2$ </td><td> $3 9 . 3 0 \pm 1 . 6 0$ </td></tr><tr><td>PT8</td><td> ${ \bf 9 6 . 0 7 \pm 0 . 5 }$ </td><td> $1 8 . 4 0 \pm 1 . 8 0$ </td><td> $\overline { { 2 5 . 2 0 \pm 0 . 3 9 } }$ </td><td> $\underline { { 4 0 . 8 3 } } \pm 1 . 9 0 $ </td><td> $2 8 . 2 0 \pm 1 . 0 3$ </td></tr><tr><td>PGD</td><td> ${ \bf 9 7 . 0 7 \pm 1 . 3 4 }$ </td><td> $5 5 . 3 0 \pm 1 . 1 2$ </td><td> $\underline { { 8 3 . 4 0 \pm 2 . 2 2 } }$ </td><td> $5 1 . 7 9 \pm 1 . 7 8$ </td><td> $7 8 . 1 0 \pm 0 . 3 0$ </td></tr><tr><td>SQ</td><td> ${ \bf 9 6 . 0 0 \pm 0 . 7 5 }$ </td><td> $8 6 . 5 0 \pm 2 . 7 6$ </td><td> $9 0 . 6 0 \pm 1 . 4 8$ </td><td> $5 2 . 1 9 \pm 1 . 5 6$ </td><td> $1 4 . 9 0 \pm 1 . 2 3 $ </td></tr><tr><td>UP</td><td> $\mathbf { 9 7 . 0 0 \pm 1 . 1 4 }$ </td><td> $4 5 . 6 0 \pm 3 . 0 6$ </td><td> $2 8 . 5 0 \pm 2 . 5 2$ </td><td> $5 5 . 8 0 \pm 2 . 1 2$ </td><td> $2 1 . 7 0 \pm 1 . 1 7$ </td></tr><tr><td>AP8</td><td> ${ \bf 9 6 . 7 3 \pm 1 . 0 9 }$ </td><td> $9 2 . 6 0 \pm 0 . 9 6$ </td><td> $7 9 . 6 0 \pm 3 . 5 0$ </td><td> $5 5 . 6 9 \pm 2 . 1 7$ </td><td> $8 4 . 6 0 \pm 2 . 7 3$ </td></tr><tr><td>FG8</td><td> ${ \bf 9 6 . 8 7 \pm 0 . 9 6 }$ </td><td> $8 4 . 5 0 \pm 2 . 1 9$ </td><td> $4 7 . 9 0 \pm 3 . 6 3$ </td><td> $5 5 . 3 2 \pm 1 . 2 7$ </td><td> $1 4 . 8 0 \pm 1 . 4 9$ </td></tr></table>

Table 28: The results of TPR measured on 1k adversarial samples. TPR := (the number of adv correctly detected) / (the number of adv) ×100 (%). The best results are written in bold, and the second-best results are written with underlines.

## L.2 TPR on Adversarial Samples Comparison

In this section we focus on True Positive Rate (TPR), which specifically measures how well each method detects adversarial samples (the percentage of adversarial samples correctly identified as adversarial). Our method maintains excellent TPR performance (96.00% to 97.07%) across all attack types. Feature Squeezing (FS) shows strong TPR for CW and SQ4 attacks (93.20% and 90.60% respectively) but performs poorly on P8 and UP4 attacks. The Mahalanobis Detector (MD) demonstrates good TPR for AA8 and AP8 attacks (92.3% and 91.4%) but fails significantly on P8 attack (16.8%). The MAD method shows moderate TPR (40.83% to 56.82%) but with high variance, while MagNet (MN) fails with low TPR for most attacks, indicating it cannot effectively detect adversarial samples.

## L.3 End To End Accuracy Comparison

In this section we present the end-to-end accuracy, which is a comprehensive metric that considers both correct detection of adversarial samples by the detector and correct detection of clean samples and correct classification of clean samples. Our method achieves excellent end-to-end accuracy (93.9% to 94.8%) across all attack types, demonstrating consistent and robust performance. The Mahalanobis Detector (MD) shows moderate performance (51.0% to 89.0%) with good results on AA8 and AP8 attacks but struggling with P8 attack. Feature Squeezing (FS) shows reasonable performance (51.30% to 85.30%) but with significant variation across different attacks, performing well on CW and SQ4 attacks but struggling with P8 and UP4 attacks. The MAD method shows poor performance (42.03% to 49.18%) with accuracies around random guessing level. MagNet (MN) performs with low accuracies for most attacks, indicating it cannot provide effective end-to-end protection. The consistency of our method across different attack types demonstrates its robustness and reliability in maintaining both detection accuracy and classification performance under adversarial conditions.

<table><tr><td>Attack / Metric</td><td>Ours</td><td>MD</td><td>FS</td><td>MAD</td><td>MN</td></tr><tr><td>AA8 CW</td><td>94.1 94.8</td><td>89.0</td><td>78.64</td><td>49.18</td><td>76.60</td></tr><tr><td rowspan="4">PT8 PGD SQ</td><td></td><td>72.0</td><td>85.30</td><td>48.92</td><td>54.20</td></tr><tr><td>94.4</td><td>51.0</td><td>51.30</td><td>42.03</td><td>48.65</td></tr><tr><td>94.4 93.9</td><td>70.0 86.0</td><td>80.40 84.00</td><td>48.72</td><td>73.60</td></tr><tr><td></td><td></td><td></td><td>47.99</td><td>42.00</td></tr><tr><td>UP AP8</td><td>94.4 94.6</td><td>65.0 89.0</td><td>52.94 78.50</td><td>48.99 48.58</td><td>45.40 76.85</td></tr><tr><td>FG8</td><td>94.5</td><td>84.0</td><td>62.64</td><td>49.04</td><td>41.95</td></tr></table>

Table 29: The results of end-to-end accuracy: ((the number of correctly detected adversarial samples by the detector)+ (the number of correctly detected clean samples and correctly classified samples))/(1000 + 1000).

## M Key Leakage Experiments

In this section, we measure the detection rate under a fully white-box adaptive attack where b is known. Table 30 shows the detection rates. Detection degrades gradually as key knowledge increases, remaining at 65.31% at 75% recovery and reaching near-chance (55.17%) only under complete key compromise.

<table><tr><td>Fraction of known</td><td>Detection rate (CIFAR-10)</td></tr><tr><td>0%</td><td>86.7%</td></tr><tr><td>25%</td><td>81.23%</td></tr><tr><td>50%</td><td>76.22%</td></tr><tr><td>75%</td><td>65.31%</td></tr><tr><td>100%</td><td>55.17%</td></tr></table>

Table 30: Comparison of detection rates on CIFAR-10 as the fraction of known information varies from 0% to 100%.