# When Does Perturbation Consistency Add Information Beyond Softmax Confidence? Theory and Evidence for the Decision Consistency Framework

Amit Kalita<sup>a</sup>, Moirangthem Tiken Singh<sup>a,∗</sup>

<sup>a</sup>Department ofComputer Science and Engineering, Dibrugarh University Institute ofEngineering and Technology, Dibrugarh University, Dibrugarh, Assam, India

## Abstract

Perturbation consistency scores judge whether a prediction survives small input changes. The Decision Consistency Framework (DCF) scores each image by label stability (LS) and confidence deviation (CD) under six mild operators, combines them into a consistency score (CS), and summarises a model by the gap between accuracy and mean consistency (CAG). We ask when these scores carry information that the clean softmax output lacks. Across 201 evaluations with three seeds per configuration, CS detected misclassifications on clean data less well than the maximum softmax probability (MSP) in every model family. For frozen CNNs the deficit was 0.057 in area under the ROC curve (AUROC; 95% CI 0.043 to 0.073), at 8 to 15 times the compute. A label flip requires the logit margin to be at most the square root of two times the logit displacement, and at a fixed margin the displacement predicted errors at AUROC 0.515. The same inequality gives a label-free split-conformal no-flip certificate, never violated in 84 runs. Consistency did add information where confidence is misled: under noise corruptions of CIFAR-10-C it beat the margin, and under projectedgradient attacks it separated clean from attacked inputs with AUROC 0.83 against 0.37 for MSP at a budget of 2/255, as pre-specified. The gain under noise did not transfer to unseen shift families. The large negative CAG on OCTMNIST persists under full fine-tuning and in-domain training and is partly a class-prior efect: correcting the prior recovers about 40% of the gap.

Keywords: decision consistency, misclassification detection, conformal prediction, distribution shift, label shift, adversarial detection

## 1. Introduction

Accuracy on a held-out test set says how often a classifier is right on average. It does not say which of its predictions should be trusted. In deployment the second question matters more. A radiology triage tool or an inspection camera has to decide, image by image, whether to act on the output or pass the case to a person [1, 2]. The usual answer is a confidence score. The maximum softmax probability (MSP) [3], the logit margin, the predictive entropy and the energy score [4] are all computed from one forward pass and are dificult to outperform for misclassification detection [5, 6, 7].

These scores read only the clean output. They cannot show whether a prediction would survive a small, harmless change to the input. A second family of scores addresses this gap. Test-time augmentation agreement [8, 9], the flip probabilities of ImageNet-P [10] and feature squeezing for adversarial detection [11] all follow this idea. The Decision Consistency Framework (DCF) is one instance of it. For each test image, DCF applies six mild operators: rotation, brightness, Gaussian noise, contrast, blur and JPEG compression. It records how often the predicted label is kept (label stability, LS) and how much the top-class confidence moves (confidence deviation, CD). It then combines the two into a consistency score (CS). At the model level, DCF reports the consistency-accuracy gap (CAG), the diference between clean accuracy and mean CS.

The appeal of such scores is intuitive. A prediction that flips under a rotation of two degrees looks fragile, and one that never moves looks dependable. The cost is also clear, because each operator needs one extra forward pass. That cost is justified only if consistency carries information that the clean output does not already contain. Arguments for such scores usually rest on Lipschitz-style bounds and on flip frequencies. They seldom rest on a comparison with the confidence the scores are meant to improve on. The central question is therefore:

When does perturbation consistency add information about the correctness of a prediction beyond what the clean softmax output already provides?

We answer it with a theoretical argument and a pre-specified set of experiments. The answer is mostly negative, with two exceptions. The contributions follow the order of the argument.

1. A mechanism for redundancy. We prove that a label flip under any operator implies that the clean logit margin is at most the square root of two times the norm of the logit displacement. It follows that LS and CD can add information only through the dependence of the displacement on correctness at a fixed clean output. Measured directly within margin bins, that dependence is negligible on clean data.

2. A finite-sample no-flip certificate. The same inequality yields a split-conformal rule that certifies an image from its clean margin alone. The calibration set needs no labels. The probability that an image is certified and still flips is bounded by a chosen level, and this bound held in every run at every level tested. Unlike a Lipschitz flip-probability bound, which is vacuous in this setting, the rule needs neither a Lipschitz constant nor a shared perturbation budget.

3. A broad comparison against confidence baselines. The mechanism predicts that consistency is redundant on clean data, and the comparison tests this prediction. It covers frozen and natively trained CNNs, vision transformers, a model zoo and last-layer ensembles, evaluated on several natural-image and medical-image datasets. CS was below MSP on average for clean error detection in every model family. LS alone was close to chance among high-confidence predictions. No reformulation of CD and no choice of the weight between LS and CD closed the gap.

4. A boundary condition. The same mechanism indicates where redundancy should fail, namely where the clean output is itself misled. Consistency signals did add information in two such settings. The first is noise-type corruptions of natural images, where the gain grew with MSP over-confidence. The second is adversarial inputs, where a pre-specified prediction was confirmed in every run. The gain does not transfer to shift families not seen in training. An attacker who penalises instability can evade the detector, at the price of a much lower attack success rate.

5. A partial explanation of “stable but wrong” predictions. The last finding concerns CAG. Models trained on a retinal imaging task had a large negative CAG, which means that many wrong predictions were stable. The pattern survives longer head training, full fine-tuning and training from scratch on the target domain, and at matched accuracy the in-domain model is no better. Part of it is a class-prior shift between the oficial splits. Correcting the prior recovers part of the gap, and inducing the same kind of shift on a natural-image benchmark reproduces it.

These findings suggest a rule for practice. Use MSP or the margin for clean error detection. Use the conformal certificate when a cheap guarantee against small perturbations is needed. Reserve consistency checks for settings where confidence itself is known to be corrupted.

The rest of the paper is organised as follows. Section 2 reviews related work. Section 3 defines the DCF quantities and states the theory. Section 4 describes the protocol, including the perturbation audit and the pre-specified predictions. Section 5 reports the results, Section 6 discusses their implications and limitations, and Section 7 concludes.

## 2. Related Work

This section follows the structure of the argument. It first reviews the single-pass confidence scores that serve as baselines. It then turns to perturbation-based scores, to the behaviour of uncertainty estimates under shift, and to the margin and conformal tools that underlie the theory. It ends with adversarial detection and label shift, two settings in which confidence can be misleading.

## 2.1. Misclassification detection from a single forward pass

Hendrycks and Gimpel [3] showed that MSP separates correct from incorrect predictions well above chance. It has remained a strong baseline since then. Later scores reuse the same logits. They include the margin between the two largest logits, the predictive entropy, the energy score [4], the maximum logit [12] and DOCTOR [13]. Temperature scaling [14] improves calibration without changing the ranking of MSP. Careful evaluations found that elaborate alternatives rarely outperform these baselines for failure detection [5, 6]. They also found that the remaining weaknesses of MSP can often be fixed post hoc [7, 15].

Other families of methods address related problems. Feature-space scores such as the Mahalanobis distance [16] and deterministic uncertainty methods [17] target out-of-distribution (OOD) inputs [18] more than misclassification. Training-time methods such as OpenMix [19] change the model and fall outside the posthoc setting considered here. Recent work applies failure detection to vision-language models [20] and to concept-based explanations of errors [21]. A benchmark of confidence scores on medical image classification under distribution shift found that no single score dominated across settings [22]. The single-pass scores therefore form the baselines that any perturbation-based score must outperform.

## 2.2. Perturbation consistency and test-time augmentation

Perturbation-based scores start from a diferent idea. They ask whether a prediction survives a change to the input. Averaging predictions over augmented views, known as test-time augmentation (TTA), can improve accuracy and uncertainty estimates [8, 9]. ImageNet-C and ImageNet-P [10] measure average-case robustness to corruptions and the probability that a prediction flips along a perturbation sequence. The motivation is that convolutional networks can change their output under shifts of a few pixels or small rotations [23, 24].

Consistency under perturbation is also used in other roles. It serves as a training signal [25, 26, 27] and as an objective for test-time adaptation [28]. For detection, feature squeezing [11] flags adversarial inputs whose prediction changes under bit-depth reduction or smoothing. DCF belongs to this family. It difers from TTA in using the agreement and the confidence change as a reliability score rather than averaging the predictions. TTA and the single-pass scores are therefore both relevant comparisons.

## 2.3. Uncertainty under distribution shift

Reliability scores are most needed when the data change, and uncertainty estimates degrade under shift [29]. Deep ensembles [30] and Monte Carlo dropout [31] remain common references. At the model level, accuracy under shift is often linearly related to in-distribution accuracy [32, 33]. Agreement between models, or the average confidence, can predict accuracy on unlabelled shifted data [34, 35, 36, 37]. A clean-data score that is claimed to predict behaviour under shift therefore has to outperform clean accuracy itself as a predictor. A correlation with robustness is not suficient.

## 2.4. Margins, Lipschitz bounds and conformal guarantees

Guarantees about prediction changes rest on the logit margin. Lipschitz-margin arguments give deterministic certificates. If the logit margin exceeds a constant times the product of the Lipschitz constant and the perturbation budget, no perturbation within the budget can flip the label [38, 39, 40, 41]. Randomised smoothing gives probabilistic certificates for a smoothed classifier [42]. Both approaches need either a Lipschitz constant or a change to the classifier.

Conformal prediction ofers a diferent route. It gives distribution-free, finite-sample guarantees under exchangeability [43, 44, 45], and its behaviour when exchangeability fails is well studied [46, 47]. The certificate proposed here combines the two approaches. The margin inequality is deterministic. The threshold is set by a conformal quantile of measured displacements, so no Lipschitz constant is needed.

## 2.5. Adversarial detection and adaptive attacks

Adversarial inputs are a setting in which confidence is deliberately misled. Projected gradient descent (PGD) [48] is the standard first-order attack. Ensembles of parameter-free attacks are used when robustness itself is the claim [49]. Detectors that look convincing against fixed attacks often fail once the attacker knows the detector [50, 51, 52]. Non-diferentiable components such as JPEG compression must be attacked through approximations such as the backward pass diferentiable approximation (BPDA) [51]. Evidence for a detector therefore requires an adaptive attack as well as a standard one.

## 2.6. Label shift

Class priors are a second source of misleading confidence. When class priors difer between training and test data while the class-conditional distributions do not, a classifier’s posteriors are biased towards the training prior. The prior can be re-estimated without labels by expectation-maximisation [53] or by blackbox shift estimation [54]. Calibrated maximum likelihood is a strong choice [55, 56]. Class imbalance more generally leaves minority classes with low recall [57]. This literature makes label shift a candidate explanation for confident but wrong predictions.

## 3. Decision consistency: definitions and theory

## 3.1. Notation

Let $z ( \mathbf { x } ) \in \mathbb { R } ^ { J }$ be the logits of a classifier with J classes for an input x, $p ( \mathbf { x } ) = \operatorname { s o f t m a x } ( z ( \mathbf { x } ) )$ the class probabilities, ${ \hat { y } } ( \mathbf { x } ) = \arg \operatorname* { m a x } _ { j } z _ { j } ( \mathbf { x } )$ the prediction and $c ( \mathbf { x } ) = \mathrm { m a x } _ { j } p _ { j } ( \mathbf { x } )$ the MSP. The clean logit margin is

$$
\gamma ( \mathbf { x } ) = z _ { \hat { y } ( \mathbf { x } ) } ( \mathbf { x } ) - \operatorname* { m a x } _ { j \neq \hat { y } ( \mathbf { x } ) } z _ { j } ( \mathbf { x } ) \ \geq 0 .\tag{1}
$$

A perturbation set $\mathcal { T } = \{ T _ { 1 } , \ldots , T _ { K } \}$ contains K operators. An operator may be random, as Gaussian noise is. In that case its randomness is drawn independently for every image. For image $\mathbf { X } _ { i }$ we write $\hat { y } _ { i } = \hat { y } ( \mathbf x _ { i } )$ $c _ { i } = c ( \mathbf { x } _ { i } ) , \gamma _ { i } = \gamma ( \mathbf { x } _ { i } ) , \mathbf { x } _ { i } ^ { ( k ) } = T _ { k } ( \mathbf { x } _ { i } ) , \hat { y } _ { i } ^ { ( k ) } = \hat { y } ( \mathbf { x } _ { i } ^ { ( \hat { k } ) } ) , c _ { i } ^ { ( k ) } = \bar { c } ( \mathbf { x } _ { i } ^ { ( k ) } )$ , and

$$
\Delta _ { i , k } = z ( \mathbf { x } _ { i } ^ { ( k ) } ) - z ( \mathbf { x } _ { i } ) \in \mathbb { R } ^ { J }\tag{2}
$$

for the logit displacement caused by operator k. We say that operator k flips image i if $\hat { y } _ { i } ^ { ( k ) } \neq \hat { y } _ { i }$ . With ties, a flip means that some class $j \neq \hat { y } _ { i }$ satisfies $z _ { j } ( \mathbf { x } _ { i } ^ { ( k ) } ) \geq z _ { \hat { y } _ { i } } ( \mathbf { x } _ { i } ^ { ( k ) } )$ . Correctness is $Z _ { i } = \mathbf { 1 } [ \hat { y } _ { i } = y _ { i } ]$ , where $1 [ \cdot ]$ is the indicator function and $y _ { i }$ is the true label.

## 3.2. The DCF quantities

DCF summarises the behaviour of a prediction under the K operators with three image-level quantities and one model-level quantity. Label stability (LS) is the fraction of operators that keep the clean prediction,

$$
S _ { i } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbf { 1 } \Big [ \hat { y } _ { i } ^ { ( k ) } = \hat { y } _ { i } \Big ] \in [ 0 , 1 ] .\tag{3}
$$

Confidence deviation (CD) measures the complementary efect on confidence. It is the mean absolute change of the top-class confidence,

$$
D _ { i } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \Bigl | c _ { i } - c _ { i } ^ { ( k ) } \Bigr | \in [ 0 , 1 ] .\tag{4}
$$

Equation (4) compares maximum probabilities even when the winning class changes. A prediction can therefore flip from one class to another with the same confidence and still have $D _ { i } ~ \approx ~ 0$ . Equation (4) is the reference definition used throughout.

The consistency score (CS) combines the two quantities as a convex combination,

$$
C _ { i } = \alpha S _ { i } + ( 1 - \alpha ) ( 1 - D _ { i } ) , \qquad \alpha \in [ 0 , 1 ] ,\tag{5}
$$

with $\alpha = 0 . 5$ unless stated otherwise. A single value of $C _ { i }$ does not determine $S _ { i }$ and $D _ { i } ,$ because many pairs give the same score. LS and CD are therefore always reported alongside CS, and CS is read only as a ranking score.

At the model level, the consistency-accuracy gap (CAG) is

$$
\mathbf { C } \mathbf { A } \mathbf { G } = \mathbf { A } \mathbf { c c } - \bar { C } , \qquad \bar { C } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } C _ { i } ,\tag{6}
$$

where Acc is the accuracy on the full test set and $\bar { C }$ is averaged over $N$ evaluation images. Accuracy measures agreement with the true label, while CS measures agreement of the model with itself. The diference of the two is therefore not a calibration measure, and it carries no guarantee. CAG is used only as a descriptive flag. For a tolerance δ, we call a model accuracy-dominant if $\mathrm { C A G } > \delta ,$ consistency-dominant if $\mathrm { C A G } < - \delta$ and balanced otherwise. The label “balanced” makes no calibration claim. Calibration refers to the agreement between confidence and accuracy [14].

## 3.3. What a flip implies about the margin

The first result connects a label flip to the clean margin. It holds for every image and every operator, and it requires no assumption about the classifier.

Proposition 1 (Flip implication). For every image x<sub>i</sub> and operator $T _ { k } , i f T _ { k } f i p s { \bf x } _ { i }$ then

$$
\gamma ( \mathbf { x } _ { i } ) ~ \leq ~ \left| \Delta _ { i , k , \hat { y } _ { i } } \right| + \operatorname* { m a x } _ { j \neq \hat { y } _ { i } } \left| \Delta _ { i , k , j } \right| ~ \leq ~ \sqrt { 2 } \left\| \Delta _ { i , k } \right\| _ { 2 } .\tag{7}
$$

Consequently $\begin{array} { r } { 1 - S _ { i } \le \frac { 1 } { K } \sum _ { k } \mathbf { 1 } \big [ \gamma ( \mathbf { x } _ { i } ) \le \sqrt { 2 } \| \Delta _ { i , k } \| _ { 2 } \big ] . } \end{array}$ , and $S _ { i } = 1$ whenever $\gamma ( \mathbf { x } _ { i } ) > \ \sqrt { 2 } \operatorname* { m a x } _ { k } \| \Delta _ { i , k } \| _ { 2 }$

Proof. Write $\hat { y } = \hat { y } _ { i } , z = z ( \mathbf { x } _ { i } ) , z ^ { \prime } = z ( \mathbf { x } _ { i } ^ { ( k ) } )$ and $\Delta = z ^ { \prime } - z$ . If $T _ { k }$ flips $\mathbf { X } _ { i } ,$ , there is a class $j ^ { * } \neq \hat { y }$ with $z _ { j ^ { \ast } } ^ { \prime } \geq z _ { \hat { y } } ^ { \prime }$ By the definition of the margin, $z _ { j ^ { * } } \leq$ ma $\mathrm { x } _ { j \neq \widehat { y } } z _ { j } = z _ { \widehat { y } } - \gamma ( \mathbf { x } _ { i } )$ . Hence

$$
0 \ \geq \ z _ { \hat { y } } ^ { \prime } - z _ { \hat { y } ^ { * } } ^ { \prime } \ = \ ( z _ { \hat { y } } - z _ { \hat { y } ^ { * } } ) + ( \Delta _ { \hat { y } } - \Delta _ { \hat { y } ^ { * } } ) \ \geq \ \gamma ( \mathbf { x } _ { i } ) + \Delta _ { \hat { y } } - \Delta _ { \hat { y } ^ { * } } ,
$$

so $\begin{array} { r } { \gamma ( \mathbf { x } _ { i } ) \leq \Delta _ { j ^ { * } } - \Delta _ { \hat { y } } \leq | \Delta _ { \hat { y } } | + | \Delta _ { j ^ { * } } | \leq | \Delta _ { \hat { y } } | + \operatorname* { m a x } _ { j \neq \hat { y } } | \Delta _ { j } | } \end{array}$ , which is the first inequality. For the second, let $a = | \Delta _ { \hat { \nu } } |$ and $b = \mathrm { m a x } _ { j \neq \hat { y } } | \Delta _ { j } | = | \Delta _ { j ^ { \prime } } |$ for some $j ^ { \prime } \neq \hat { y }$ . By the Cauchy-Schwarz inequality, $a + b \leq \sqrt { 2 } \sqrt { a ^ { 2 } + b ^ { 2 } }$ and $a ^ { 2 } + b ^ { 2 } = \Delta _ { \hat { \nu } } ^ { 2 } + \Delta _ { j \prime } ^ { 2 } \leq \| \Delta \| _ { 2 } ^ { 2 }$ because $\hat { y } \ne j ^ { \prime }$ are two distinct coordinates. Taking square roots gives $a + b \leq \sqrt { 2 } \vert \vert \Delta \vert \vert _ { 2 }$ . The statement about $S _ { i }$ follows by applying the implication to each operator and averaging the indicators over k. □

Bounding the change of the winning logit and of the runner-up separately by $\| \Delta _ { i , k } \| .$ <sub>2</sub> gives the constant 2. Combining the two coordinates as in the proof gives ${ \sqrt { 2 } } ,$ , which is tight (take ∆ supported on ˆy and one competitor with equal and opposite entries). If the logit map is L-Lipschitz and $\begin{array} { r } { \| \mathbf { x } _ { i } ^ { ( k ) } - \mathbf { x } _ { i } \| _ { 2 } \leq \varepsilon , } \end{array}$ then $\| \Delta _ { i , k } \| _ { 2 } \leq L \varepsilon { \mathrm { a n d } } \mathrm { E q . } ( 7 )$ gives the familiar certificate $\gamma > \sqrt { 2 } L \varepsilon \Rightarrow \mathrm { n o }$ flip [39].

A probabilistic version of this certificate is not informative in this setting.

Remark 1 (Limitations of a probabilistic Lipschitz bound). A bound oftheform $P ( \hat { y } _ { i } ^ { ( k ) } \neq \hat { y } _ { i } ) \leq \operatorname* { m i n } ( 1 , 2 L \varepsilon / \gamma _ { i } )$ obtainedfrom Markov’s inequality, is vacuous: when $\gamma _ { i } \leq 2 L \varepsilon$ it equals one, and when $\gamma _ { i } > 2 L \varepsilon$ ε the deterministic certificate already gives zero. It also needs one budget ε shared by all operators, which operators of diferent kinds do not generally share. In addition, checking that images with $\gamma _ { i } > \sqrt { 2 } \vert \vert \Delta _ { i , k } \vert \vert _ { 2 }$ do notflip, with $\Delta _ { i , k }$ measured on the same pair, cannot validate the bound, because Proposition 1 guarantees the outcome. This check serves only as an implementation test. The conformal certificate avoids these limitations.

## 3.4. A split-conformal no-flip certificate

Proposition 1 needs the displacement of the image being certified, and that displacement is available only after the perturbed image has been processed. A more practical rule uses only the clean margin. Its threshold is calibrated on the worst-case scaled displacement of an image,

$$
R ( \mathbf { x } _ { i } ) = \sqrt { 2 } \operatorname* { m a x } _ { k } \left\| \Delta _ { i , k } \right\| _ { 2 } ,\tag{8}
$$

which depends on the image and on the draw of the random operators, but not on its label.

Proposition 2 (Conformal no-flip certificate). $L e t { \bf x } _ { 1 } , \ldots , { \bf x } _ { n }$ be calibration images and ${ \bf X } _ { n + 1 }$ a test image such that $R ( \mathbf { x } _ { 1 } ) , \ldots , R ( \mathbf { x } _ { n + 1 } )$ are exchangeable. For $\cdot \beta \in ( 0 , 1 )$ let $m = \lceil ( n + 1 ) ( 1 - \beta ) \rceil$ and let q be the m-th smallestˆ $o f R ( \mathbf { x } _ { 1 } ) , \ldots , R ( \mathbf { x } _ { n } )$ , with $\hat { q } = \infty i f m > n . \ C e r t i f y { \bf x } _ { n + 1 } i f \gamma ( { \bf x } _ { n + 1 } ) > \hat { q } .$ . Then

$$
P ( \mathbf { x } _ { n + 1 } \ i s \ c e r t i f i e d \ a n d \ s o m e \ o p e r a t o r \ f i p s \ i t ) \ \le \ P ( R ( \mathbf { x } _ { n + 1 } ) > \hat { q } ) \ \le \ \beta ,\tag{9}
$$

where the probability is over the calibration images, the test image and the operator randomness.

Proof. If ${ \bf X } _ { n + 1 }$ is certified and some operator k flips it, Proposition 1 gives $\hat { q } < \gamma ( \mathbf { x } _ { n + 1 } ) \leq \sqrt { 2 } \| \Delta _ { n + 1 , k } \| _ { 2 } \leq$ $R ( \mathbf { x } _ { n + 1 } )$ . The event on the left of Eq. (9) is therefore contained in $\{ R ( \mathbf { x } _ { n + 1 } ) ~ > ~ \hat { q } \}$ , which proves the first inequality. For the second, by exchangeability the rank of $R ( \mathbf { x } _ { n + 1 } )$ among the $n + 1$ values is uniformly distributed on $\{ 1 , \ldots , n + 1 \}$ when there are no ties, and ties only make the event $R ( \mathbf { x } _ { n + 1 } ) > \hat { q }$ less likely. The event $R ( \mathbf { x } _ { n + 1 } ) > \hat { q }$ requires $R ( \mathbf { x } _ { n + 1 } )$ to be larger than at least m of the calibration values, so its rank is at least $m + 1$ , which has probability at most $( n + 1 - m ) / ( n + 1 ) \le \beta$ by the choice of m [45, 44]. If $m > n , { \hat { q } } = \infty$ and nothing is certified. □

The certificate has three properties. First, the threshold is fixed on separate data, and at test time only the clean margin is used, so the perturbed versions of the test image are never computed. Second, the certificate needs no labels, because R is computed from logits alone, and any unlabelled sample from the deployment distribution can serve as the calibration set. Third, the guarantee is marginal and depends on exchangeability. Under distribution shift the calibration quantile may be too small. $\mathbf { A }$ per-sample flip bound can be built in the same way from the empirical distribution of $\sqrt { 2 } | | \Delta _ { i , k } | | _ { 2 }$ on the calibration split, and the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality [58] makes it uniform in finite samples.

## 3.5. When can consistency add information?

The flip implication also bears on the central question of whether LS and CD can add information beyond the clean output. Proposition 1 says that a flip depends on the margin and on the size of the displacement. The clean output already contains the margin. Consistency can therefore add information only through the displacement, which the next proposition makes precise.

Proposition 3 (Redundancy condition). Write Z, S and Dfor the correctness, LS and CD ofan image drawn at random, and $\Delta _ { k }$ for the displacement caused by operator k (the image index is dropped). LS and CD are deterministic functions of the clean logits z(x) and the displacements $( \Delta _ { 1 } , \ldots , \Delta _ { K } )$ . If the displacements are conditionally independent of correctness given the clean logits, $( \Delta _ { 1 } , \dots , \Delta _ { K } ) \ \perp \ Z \ | \ z ( \mathbf { x } )$ , then $P ( Z = 1 \ |$ $z , S , D ) = P ( Z = 1 \mid z ) .$ . In that case no error-detection score built from $( z , S , D )$ has a higher area under the ROC curve (AUROC) than the posterior $P ( Z = 1 \mid z ) ,$ , which uses the clean logits alone.

Proof. $( S , D )$ is a function of $( z , \Delta _ { 1 } , . . . , \Delta _ { K } )$ , so conditional independence of the displacements from $Z$ given z implies $( S , D ) \bot Z \mid z .$ , and hence $P ( Z = 1 \mid z , S , D ) = P ( Z = 1 \mid z )$ . For any feature vector, the ROC curve of the posterior probability of the positive class dominates the ROC curve of every other score computed from the same features (the Neyman-Pearson lemma applied to the likelihood ratio, which is a monotone function of the posterior). Since the posterior given (z S D) equals the posterior given z, the best achievable ROC curve is the same with or without $( S , D )$ □

Proposition 3 makes the question of added information testable. Consistency can only help if two images with the same clean logits, one classified correctly and one not, tend to move diferently under the operators. The cleanest test therefore fixes the margin and asks whether the displacement still separates correct from incorrect predictions. The condition is expected to fail when the clean output itself is corrupted, for example by input noise or by an adversarial perturbation designed to raise confidence. In such cases the displacement caused by a smoothing operator can reveal what the clean logits hide [11].

## 3.6. Evaluation pipeline

Fig. 1 shows the pipeline and where the three theoretical results enter. Computing LS, CD and CS for N images costs $N ( K + 1 )$ forward passes. The certificate costs one forward pass per test image once $\hat { q }$ has been computed on the calibration split.

## 4. Experimental protocol

## 4.1. Datasets

The experiments use four image classification datasets (Table 1). They are CIFAR-10 [59], STL-10 [60], the Intel Image Classification scene dataset [61] and OCTMNIST [62, 63]. The first three contain natural images, and OCTMNIST contains retinal optical coherence tomography images. For each run we draw an evaluation set of $N = 1 { , } 0 0 0$ images and a disjoint calibration set of 1 000 images. On CIFAR-10, STL-10 and Intel-Scene, both sets come from the oficial test split. OCTMNIST has only 1,000 test images. The whole test split is therefore the evaluation set, and the calibration set is drawn from the oficial validation split. The OCTMNIST training and validation splits share one class prior, while the test split is balanced. Accuracy and expected calibration error (ECE, 15 equal-width bins) are computed on the full test split. DCF quantities are computed on the evaluation set.

![](images/2c02629078bbdc71be43c1aebea44a0fb6d26c5d77d089078947541aef8708e3.jpg)  
Figure 1: Evaluation pipeline and where the theory enters. The left column computes the DCF quantities from K + 1 forward passes per image. The right column shows the three results of Section 3: the exact flip implication (Proposition 1), the conformal certificate that needs only the clean margin of a test image and an unlabelled calibration set (Proposition 2), and the condition under which LS and CD can carry information beyond the clean logits (Proposition 3).

Table 1: Datasets. Images are resized to 224 × 224 for the ImageNet backbones after the perturbation is applied at native resolution. OCTMNIST class priors are given as (choroidal neovascularization, diabetic macular edema, drusen, normal).
<table><tr><td>Dataset</td><td>Resolution</td><td>Classes</td><td>Split used for evaluation / calibration</td></tr><tr><td>CIFAR-10</td><td>32×32</td><td>10</td><td>test / test (disjoint)</td></tr><tr><td>STL-10</td><td>96×96</td><td>10</td><td>test / test (disjoint)</td></tr><tr><td>Intel-Scene</td><td>150× 150</td><td>6</td><td>test / test (disjoint)</td></tr><tr><td>OCTMNIST</td><td>28 × 28</td><td>4</td><td>test (balanced) / validation, prior (0.344, 0.105, 0.080, 0.472)</td></tr></table>

## 4.2. Models and adaptation

We evaluate five groups of models. Together they cover frozen and fully trained classifiers, convolutional and transformer architectures, and single models and ensembles.

The first group contains frozen CNNs: ResNet-18, ResNet-50 [64], VGG-16 [65] and MobileNetV2 [66]. Each uses frozen ImageNet weights and one linear head trained on the target data (SGD, learning rate 10<sup>−3</sup>, momentum 0.9, batch size 32, 5 epochs). The same head produces accuracy, ECE and all DCF quantities. This gives 16 configurations, CNN-001 to CNN-016.

The second group contains frozen vision transformers: ViT-B/16 [67], DeiT-B [68] and Swin-B [69]. Each is trained with a linear head by exactly the same recipe on each target dataset, which gives 12 configurations. The transformers are always evaluated with a trained head, because their ImageNet output space does not match the target classes.

The third group contains fourteen publicly released CIFAR-10 networks that were trained end to end at 32 × 32. They are ResNet-20/32/44/56, VGG-11/13/16/19 with batch normalisation, MobileNetV2 ×0 5 and ×1 4, ShufleNetV2 ×0 5 and ×2 0, and RepVGG-A0 and A2<sup>1</sup>. These networks contrast with the frozen backbones, which use only a short linear probe. For them the seed changes the evaluation images and the operator noise, not the training.

The fourth group is a 13-model zoo for model-level questions. It contains the four frozen CNNs and three vision transformers, plus ResNet-34, DenseNet-121, EficientNet-B0, ConvNeXt-T [70], RegNetY-800MF and ShufleNetV2. All use linear heads trained on CIFAR-10 by the same recipe.

Table 2: Operator audit. Each cell gives the mean $\ell _ { 2 }$ distance between clean and perturbed images at the 224 × 224 network input (pixel values in [0 1]), followed by the flip rate in percent for frozen CNNs (mean over four models and three seeds).
<table><tr><td>Operator</td><td>CIFAR-10</td><td>STL-10</td><td>Intel-Scene</td><td>OCTMNIST</td></tr><tr><td>Rotation  $2 ^ { \circ }$ </td><td>15.2/9.8</td><td> $3 9 . 0 / 3 . 6 $ </td><td> $5 0 . 9 / 5 . 2$ </td><td>0.0 / 0.0</td></tr><tr><td>Brightness ×1.1</td><td>19.1 / 2.8</td><td>18.0/0.9</td><td>18.1 / 1.2</td><td>8.3 / 3.7</td></tr><tr><td>Gaussian noise  $\sigma = 0 . 0 1$ </td><td>2.6 / 6.1</td><td>2.6 / 1.8</td><td>2.6 / 2.0</td><td>2.6 / 21.9</td></tr><tr><td>Contrast ×1.1</td><td>7.6 /2.5</td><td>7.7 / 0.7</td><td>8.0 / 1.1</td><td>5.4/ 3.6</td></tr><tr><td>Blur  $3 \times 3 , \sigma = 0 . 5$ </td><td>5.5 /9.1</td><td>5.3 / 2.2</td><td>7.6 /2.5</td><td> $6 . 1 / 1 4 . 9$ </td></tr><tr><td> $\mathrm { J P E G } q = 7 5$ </td><td>9.3 / 21.7</td><td>8.5 / 5.1</td><td>7.5 /3.2</td><td>0.6 / 2.3</td></tr></table>

The fifth group consists of adaptation controls on OCTMNIST and Intel-Scene. Linear probing and full fine-tuning can produce diferent features and failure modes [71, 72]. The controls therefore vary the training regime. They comprise three conditions. The first is the frozen head of each of the four CNNs, trained for up to 30 epochs with early stopping on a held-out 10% of the training data (patience 3). The second is full fine-tuning of ResNet-18 and MobileNetV2 (AdamW, backbone learning rate $1 0 ^ { - 4 }$ , head $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , batch $^ { 6 4 , }$ at most 15 epochs with early stopping). The third is ResNet-18 trained from random initialisation on the target data only, which serves as the in-domain control. Because of a 6 GB GPU limit, full fine-tuning on OCTMNIST used a stratified subset of 20,000 training images and 15 rather than 30 epochs. For a matched-accuracy comparison, we took from each fine-tuning or from-scratch trajectory the epoch whose test accuracy was closest to that of the paired frozen model.

Every configuration was run with seeds $^ { 4 2 , }$ 43 and 44. The seed controls head training where there is training, the choice of evaluation and calibration images, and the draw of the noise operator.

## 4.3. Perturbation operators and audit

DCF perturbs each image with six operators. Rotation is by $2 ^ { \circ }$ (nearest-neighbour, no expansion, uncovered corners set to zero). Brightness and contrast use factors of 1.1 (clipped to [0 1]). Gaussian noise has $\sigma = 0 . 0 1$ on pixels in [0 1] (clipped). Gaussian blur uses a $3 \times 3$ kernel with $\sigma = 0 . 5$ (reflect padding). JPEG re-encoding uses quality 75. All operators are applied at native resolution, before resizing.

An audit characterises how strongly each operator acts. Table 2 gives the mean $\ell _ { 2 }$ distance between clean and perturbed images at the network input, together with the flip rate of the frozen CNNs. The distances difer between operators on the same dataset. On OCTMNIST the $2 ^ { \circ }$ rotation leaves every image unchanged in the pipeline, so it acts as a null operator there, and only five of the six operators can change a prediction. The spread of the distance across images is small relative to the diferences between operators. Excluding the null operator on OCTMNIST, the 95th percentile is at most 1.65 times the median for every operator and dataset. The largest distances occur for rotation on Intel-Scene (median 50.0, 95th percentile 73.3, maximum 89.2). Gaussian noise is almost constant (CIFAR-10: median 2.65, maximum 2.76).

A screening check assessed label preservation. A vision-language model judged 420 images (360 perturbed, 20 per operator and dataset for CIFAR-10, STL-10 and Intel-Scene, plus 60 clean controls) without seeing the original label. It assigned the original class to all 420.

## 4.4. Baselines and evaluation measures

The consistency scores are compared with single-pass confidence scores and with other perturbationbased scores. For misclassification detection we score every evaluation image with MSP, MSP after temperature scaling, the logit margin of Eq. (1), negative entropy, maximum logit [12], the energy score [4], DOCTOR [13], LS, CS and the CD variants of Section 3.2. The natively trained models receive three further scores. They are the Mahalanobis score on penultimate features [16], two feature-space consistency scores (the KL divergence between clean and perturbed predictive distributions, and the displacement of the penultimate embedding), and TTA with horizontal flips and two-pixel shifts [8]. The TTA operator set is deliberately diferent from the DCF operators. The zoo adds TTA agreement and TTA mean confidence. The frozen CNNs on CIFAR-10, Intel-Scene and OCTMNIST add a five-head last-layer ensemble [30] with ensemble MSP, entropy, mutual information and member agreement.

The main measure is the AUROC for separating correct from incorrect predictions. Higher scores mean “more likely correct”, so that 0.5 is chance. We also report the area under the risk-coverage curve (AURC;

lower is better) [73]. Under shift we report AUROC on the shifted data, the error of label-free accuracy estimators (average thresholded confidence, ATC [36], and diference of confidences, DoC [37]) and, at the model level, rank correlations with accuracy under shift. Cost is measured in forward passes and in wall-clock time with mixed precision on one RTX 4050 laptop GPU.

## 4.5. Shift, adversarial and out-of-distribution tests

The scores are also tested where confidence is expected to be less reliable. We use three kinds of shift. The first is CIFAR-10-C-style corruptions generated in this study (six corruptions, five severities) on the CIFAR-10 frozen CNNs. The second is the oficial CIFAR-10-C files [10] (19 corruptions, five severities) on the 14 natively trained models. The third is an operator-strength sweep in which every DCF operator is applied at five strengths, level 1 being the setting of Table 2, with the level chosen on the calibration split.

For adversarial inputs we ran $\ell _ { \infty } \mathrm { \ P G D }$ [48] with 10 steps, step size $\epsilon _ { \infty } / 4$ and a random start, at budgets $\epsilon _ { \infty } \in \{ 2 , 4 , 8 \} / 2 5 5$ , on 500 test images per run for the 14 natively trained models. This attack has no knowledge of DCF. An adaptive attack evaluates detection against an attacker who knows the detector. It was run on four of the models (ResNet-20, ResNet-56, VGG-16-BN, RepVGG-A0) with 20 steps of size $\epsilon _ { \infty } / 8$ at $\epsilon _ { \infty } \in \{ 4 , 8 \} / 2 5 5$ . The attack maximises

$$
\mathcal { L } ( \mathbf { x } ^ { \prime } ) = \mathrm { C E } ( z ( \mathbf { x } ^ { \prime } ) , y ) - \lambda \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \bigl [ \mathrm { C E } ( z ( T _ { k } \mathbf { x } ^ { \prime } ) , \hat { y } ( \mathbf { x } ^ { \prime } ) ) + \bigl | c ( T _ { k } \mathbf { x } ^ { \prime } ) - c ( \mathbf { x } ^ { \prime } ) \bigr | \bigr ] ,\tag{10}
$$

where CE is the cross-entropy and the weight $\lambda \in \{ 0 , 1 , 5 \}$ controls the stability term. The first term pushes the prediction away from the true label, and the second makes the new prediction stable under the DCF operators. Rotation, brightness, contrast and blur are diferentiable. JPEG is passed through BPDA with an identity gradient [51]. The attacker uses its own noise draw, which the defender does not share. For OOD detection we used 1,000 images each from SVHN [74] (far OOD) and CIFAR-100 (near OOD).

## 4.6. Statistics, pre-specification and reproducibility

Every number is a mean over seeds, and models trained once are evaluated on three seeds of sampling. For comparisons between scores we use a hierarchical bootstrap that resamples configurations, then seeds within configurations, then images within runs [75] (500 replicates). We report the paired diference with its 95% interval, Cohen’s $d _ { z }$ over configurations, and Wilcoxon signed-rank tests with Holm correction [76] within each family of comparisons. The unit of analysis is the configuration, not the image.

Five predictions were specified before the adversarial and OOD experiments were run. They were: (i) consistency scores would detect adversarial inputs and errors under attack better than MSP and margin; (ii) they would not beat energy or Mahalanobis on far OOD; (iii) they would not beat MSP on near OOD; (iv) no consistency-type score would beat MSP for clean error detection; and (v) Eq. (5) would leave under-trained and stochastic-dropout heads inside or near the normal range, while at least one CD variant would expose them. The grids for α (0 to 1 in steps of 0.25), δ (0.02, 0.05, 0.10) and K (1 to 6) were fixed before the test results were examined.

## 5. Results

## 5.1. Reference measurements

Table 3 lists the measured DCF quantities for the 28 frozen configurations. Two observations follow. First, the numbers are stable across seeds: the standard deviation of mean CS is at most 0.005. Second, every CAG is negative. $\mathrm { A t } ~ \delta = 0 . 0 5 $ , 11 of the 16 CNN configurations are consistency-dominant and 5 are balanced, while 8 of the 12 transformer configurations are balanced. The transformers are more consistent than the CNNs on CIFAR-10 (CS 0.945 to 0.961 against 0.911 to 0.936). CS values are therefore not comparable across architectures without qualification.

Mean CS is not independent of accuracy. Across the 16 CNN configurations the Pearson correlation between Acc and CS is $r = 0 . 5 8 6 ( 9 5 \% \mathrm { C I } 0 . 3 6$ to 0.87, $p = 0 . 0 1 7 ;$ Spearman 0.71), and CS is not related to ECE $( r = - 0 . 2 1 , p = 0 . 4 3 )$ ). Brier scores averaged 0.077 on STL-10, 0.123 on Intel-Scene, 0.198 on CIFAR-10 and 0.485 on OCTMNIST. Like ECE, they follow accuracy. Flip rates depend on the dataset more than on the operator (Table 2). JPEG dominates on CIFAR-10, while noise (21.9%) and blur (14.9%) dominate on OCTMNIST. The input distance is a poor guide to the efect on the model. On Intel-Scene the mean distance difers by a factor of about 20 between operators (rotation 50.9, noise 2.6). JPEG moves CIFAR-10 images less than brightness does (9.3 against 19.1) but flips 21.7% of predictions rather than 2.8%.

Table 3: Reference values of the DCF quantities (mean over three seeds; ± is the standard deviation of mean CS across seeds). Acc and ECE are computed on the full test split; LS, CD and CS on $N = 1 { , } 0 0 0$ evaluation images. Regime labels use δ = 0 05 and are descriptive only.
<table><tr><td>Dataset</td><td>Model</td><td>Acc</td><td>ECE</td><td>LS</td><td>CD</td><td>CS</td><td>CAG</td><td>Regime</td></tr><tr><td colspan="9">Convolutional networks, frozen backbone and linear head</td></tr><tr><td>CIFAR-10</td><td>ResNet-18</td><td>0.864</td><td>0.009</td><td>0.912</td><td>0.067</td><td> $0 . 9 2 2 \pm 0 . 0 0 5$ </td><td>-0.058</td><td>consistency-dom.</td></tr><tr><td>CIFAR-10</td><td>ResNet-50</td><td>0.883</td><td>0.009</td><td>0.929</td><td>0.057</td><td> $0 . 9 3 6 \pm 0 . 0 0 2$ </td><td>-0.053</td><td>consistency-dom.</td></tr><tr><td>CIFAR-10</td><td>VGG-16</td><td>0.841</td><td>0.008</td><td>0.896</td><td>0.075</td><td> $0 . 9 1 1 \pm 0 . 0 0 1$ </td><td>-0.070</td><td>consistency-dom.</td></tr><tr><td>CIFAR-10</td><td>MobileNetV2</td><td>0.865</td><td>0.017</td><td>0.913</td><td>0.064</td><td> $0 . 9 2 4 \pm 0 . 0 0 3$ </td><td>-0.059</td><td>consistency-dom.</td></tr><tr><td>STL-10</td><td>ResNet-18</td><td>0.948</td><td>0.048</td><td>0.974</td><td>0.028</td><td> $0 . 9 7 3 \pm 0 . 0 0 2$ </td><td>-0.025</td><td>balanced</td></tr><tr><td>STL-10</td><td>ResNet-50</td><td>0.966</td><td>0.042</td><td>0.981</td><td>0.023</td><td> $0 . 9 7 9 \pm 0 . 0 0 1$ </td><td>-0.013</td><td>balanced</td></tr><tr><td>STL-10</td><td>VGG-16</td><td>0.951</td><td>0.022</td><td>0.975</td><td>0.027</td><td> $0 . 9 7 4 \pm 0 . 0 0 3$ </td><td>-0.023</td><td>balanced</td></tr><tr><td>STL-10</td><td>MobileNetV2</td><td>0.945</td><td>0.059</td><td>0.974</td><td>0.030</td><td> $0 . 9 7 2 \pm 0 . 0 0 0$ </td><td>-0.027</td><td>balanced</td></tr><tr><td>Intel-Scene</td><td>ResNet-18</td><td>0.911</td><td>0.014</td><td>0.978</td><td>0.024</td><td>0.977 ± 0.002</td><td>-0.066</td><td>consistency-dom.</td></tr><tr><td>Intel-Scene</td><td>ResNet-50</td><td>0.919</td><td>0.016</td><td>0.977</td><td>0.025</td><td> $0 . 9 7 6 \pm 0 . 0 0 3$ </td><td>-0.057</td><td>consistency-dom.</td></tr><tr><td>Intel-Scene</td><td>VGG-16</td><td>0.922</td><td>0.014</td><td>0.973</td><td>0.032</td><td> $0 . 9 7 0 \pm 0 . 0 0 1$ </td><td>-0.048</td><td>balanced</td></tr><tr><td>Intel-Scene</td><td>MobileNetV2</td><td>0.917</td><td>0.019</td><td>0.971</td><td>0.026</td><td> $0 . 9 7 2 \pm 0 . 0 0 1$ </td><td>-0.055</td><td>consistency-dom.</td></tr><tr><td>OCTMNIST</td><td>ResNet-18</td><td>0.629</td><td>0.115</td><td>0.913</td><td>0.053</td><td> $0 . 9 3 0 \pm 0 . 0 0 1$ </td><td>-0.301</td><td>consistency-dom.</td></tr><tr><td>OCTMNIST</td><td>ResNet-50</td><td>0.623</td><td>0.135</td><td>0.932</td><td>0.042</td><td> $0 . 9 4 5 \pm 0 . 0 0 2$ </td><td>-0.322</td><td>consistency-dom.</td></tr><tr><td>OCTMNIST</td><td>VGG-16</td><td>0.627</td><td>0.119</td><td>0.928</td><td>0.049</td><td> $0 . 9 3 9 \pm 0 . 0 0 2$ </td><td>-0.312</td><td>consistency-dom.</td></tr><tr><td>OCTMNIST</td><td>MobileNetV2</td><td>0.651</td><td>0.135</td><td>0.918</td><td>0.054</td><td> $0 . 9 3 2 \pm 0 . 0 0 2$ </td><td>-0.281</td><td>consistency-dom.</td></tr><tr><td colspan="9">Vision transformers, frozen backbone and linear head (same recipe)</td></tr><tr><td>CIFAR-10</td><td>ViT-B/16</td><td>0.951</td><td>0.011</td><td>0.959</td><td>0.036</td><td> $0 . 9 6 1 \pm 0 . 0 0 3$ </td><td>-0.011</td><td>balanced</td></tr><tr><td>CIFAR-10</td><td>DeiT-B</td><td>0.943</td><td>0.013</td><td>0.957</td><td>0.042</td><td> $0 . 9 5 8 \pm 0 . 0 0 1$ </td><td>-0.015</td><td>balanced</td></tr><tr><td>CIFAR-10</td><td>Swin-B</td><td>0.926</td><td>0.018</td><td>0.943</td><td>0.052</td><td> $0 . 9 4 5 \pm 0 . 0 0 5$ </td><td>-0.019</td><td>balanced</td></tr><tr><td>STL-10</td><td>ViT-B/16</td><td>0.980</td><td>0.050</td><td>0.993</td><td>0.016</td><td> $0 . 9 8 9 \pm 0 . 0 0 1$ </td><td>-0.008</td><td>balanced</td></tr><tr><td>STL-10</td><td>DeiT-B</td><td>0.982</td><td>0.052</td><td>0.990</td><td>0.020</td><td> $0 . 9 8 5 \pm 0 . 0 0 1$ </td><td>-0.004</td><td>balanced</td></tr><tr><td>STL-10</td><td>Swin-B</td><td>0.980</td><td>0.081</td><td>0.991</td><td>0.026</td><td> $0 . 9 8 2 \pm 0 . 0 0 1$ </td><td>-0.002</td><td>balanced</td></tr><tr><td>Intel-Scene</td><td>ViT-B/16</td><td>0.932</td><td>0.027</td><td>0.984</td><td>0.017</td><td> $0 . 9 8 4 \pm 0 . 0 0 2$ </td><td>-0.052</td><td>consistency-dom.</td></tr><tr><td>Intel-Scene</td><td>DeiT-B</td><td>0.932</td><td>0.022</td><td>0.981</td><td>0.020</td><td> $0 . 9 8 0 \pm 0 . 0 0 1$ </td><td>-0.048</td><td>balanced</td></tr><tr><td>Intel-Scene</td><td>Swin-B</td><td>0.929</td><td>0.036</td><td>0.971</td><td>0.028</td><td> $0 . 9 7 1 \pm 0 . 0 0 2$ </td><td>-0.042</td><td>balanced</td></tr><tr><td>OCTMNIST</td><td>ViT-B/16</td><td>0.656</td><td>0.118</td><td>0.935</td><td>0.039</td><td> $0 . 9 4 8 \pm 0 . 0 0 0$ </td><td>-0.292</td><td>consistency-dom.</td></tr><tr><td>OCTMNIST</td><td>DeiT-B</td><td>0.611</td><td>0.161</td><td>0.935</td><td>0.044</td><td> $0 . 9 4 6 \pm 0 . 0 0 1$ </td><td>-0.335</td><td>consistency-dom.</td></tr><tr><td>OCTMNIST</td><td>Swin-B</td><td>0.571</td><td>0.167</td><td>0.924</td><td>0.053</td><td> $0 . 9 3 6 \pm 0 . 0 0 0$ </td><td>-0.364</td><td>consistency-dom.</td></tr></table>

## 5.2. Error detection against confidence baselines

The central comparison is with single-pass confidence scores. Table 4 gives error-detection AUROC and AURC for the frozen CNNs. MSP, its temperature-scaled version and the margin are the best scores on every dataset except OCTMNIST, where negative entropy is marginally ahead. CS is below MSP on all four datasets and in all 16 configurations. LS alone is much weaker (0.700). The hierarchical bootstrap puts the diference CS − MSP at −0 057 (95% CI −0 073 to −0 043; Holm-corrected $p = 1 . 2 { \times } 1 0 ^ { - 4 } ; d _ { z } = - 1 . 9 0 )$ . The diference CS − margin, computed on unrounded values, is $- 0 . 0 5 6 \ ( - 0 . 0 7 0 \ \mathrm { t o } \ - 0 . 0 4 1 )$ , and CS − negative entropy is −0 051 (−0 069 to −0 036). CS does beat the energy score on average (+0 034), but the interval includes zero (−0 009 to +0 082; p = 0 18). Energy is itself a weak misclassification detector here because the linear heads leave the logit scale uncalibrated.

The same ordering holds beyond the frozen-backbone setting (Fig. 2, Table 5). With the transformers, CS reached 0.855 against 0.891 for MSP and 0.895 for the margin. It was ahead of MSP only in 6 of 36 runs, all on STL-10, where it still trailed the margin. With the natively trained CIFAR-10 models the gap narrowed to 0.016 but did not change sign: CS was better than MSP in 3 of 42 runs. The deficit is therefore smaller for natively trained models than for linear probes, but it has the same sign. In the 13-model zoo, CS (0.778) was again below MSP (0.865). TTA mean confidence matched MSP exactly (0.865), while TTA label agreement (0.742) was below CS. The five-head ensembles were barely better than a single head (ensemble MSP 0.854 against 0.851), but CS lost to ensemble MSP in all 36 runs.

Table 4: Misclassification detection for the 16 frozen CNN configurations (48 runs). Each cell gives AUROC / AURC, averaged over models and seeds. Higher AUROC and lower AURC are better.
<table><tr><td>Score</td><td>CIFAR-10</td><td>STL-10</td><td>Intel-Scene</td><td>OCTMNIST</td><td>All</td></tr><tr><td>MSP</td><td>0.883 / 0.029</td><td>0.950 / 0.004</td><td>0.892 / 0.014</td><td>0.778 / 0.174</td><td>0.876 / 0.055</td></tr><tr><td>MSP, temperature-scaled</td><td>0.883 / 0.029</td><td>0.951 / 0.004</td><td>0.892 / 0.014</td><td>0.778 /0.174</td><td>0.876 / 0.055</td></tr><tr><td>Logit margin</td><td>0.883 / 0.029</td><td>0.950 / 0.004</td><td>0.891 / 0.014</td><td>0.773 / 0.175</td><td>0.874 / 0.056</td></tr><tr><td>Negative entropy</td><td>0.873 / 0.031</td><td>0.940 / 0.004</td><td>0.884 / 0.015</td><td>0.782 / 0.173</td><td>0.870 / 0.056</td></tr><tr><td>Max logit</td><td>0.813 / 0.044</td><td>0.910 / 0.006</td><td>0.722 / 0.040</td><td>0.773 / 0.182</td><td>0.804 / 0.068</td></tr><tr><td>Energy</td><td>0.794 / 0.048</td><td>0.890 / 0.008</td><td>0.686 / 0.045</td><td>0.767 / 0.185</td><td>0.784 / 0.071</td></tr><tr><td>CS, Eq. (5)</td><td>0.802 / 0.045</td><td>0.918 / 0.006</td><td>0.861 / 0.018</td><td>0.693 / 0.215</td><td>0.819 / 0.071</td></tr><tr><td>CS, rescaled CD</td><td>0.820 / 0.041</td><td>0.928 / 0.005</td><td>0.870 / 0.017</td><td>0.713 / 0.204</td><td>0.833 / 0.067</td></tr><tr><td>LS, Eq. (3)</td><td>0.737 / 0.076</td><td>0.766 / 0.021</td><td>0.683 / 0.059</td><td>0.616 / 0.304</td><td>0.700 / 0.115</td></tr></table>

![](images/98478a38b71182bc978d2434140e7e18bcc4f905fce2421d58af4e7fa7d1b1cd.jpg)  
Figure 2: Misclassification-detection AUROC on clean data for four model families. Bars are means over runs; whiskers span the 10th to 90th percentiles across runs. CS (Eq. (5)) is below MSP and the margin in every family.

Among the consistency-type scores computed on the natively trained models, TTA mean confidence was the only one to beat MSP (0.939 against 0.920). It averages predictions over views instead of measuring their disagreement. This is consistent with Proposition 3: averaging can improve the estimate of the clean posterior, while agreement only adds a second, noisier view of the same quantity.

Three further analyses examine whether LS or CD contribute beyond confidence. Among predictions with MSP ≥ 0 90 (about 600 images per run with 15 errors), LS was at chance (AUROC 0.508), CS reached 0.692 and the margin 0.773. A cross-validated logistic model on MSP and margin changed by +0 001 when LS was added and by −0 001 when LS and CD were both added. The Spearman correlation between margin and LS was 0.40 to 0.68 across configurations (mean 0.55), but the partial correlation given MSP fell to −0 03 to 0.16 (mean 0.07). Stability is mostly a re-expression of confidence.

The cost of DCF is also higher than that of the baselines. DCF needs seven forward passes per image, and the measured slowdown with mixed precision was 7 9× (VGG-16) to 15 2× (MobileNetV2). The ranking of configurations by mean CS also depends on the operator set. Kendall’s $\tau _ { \mathrm { K } }$ between the ranking from all six operators and the ranking from a subset was 0.44 on average for one operator, 0.68 for three and 0.88 for five. Across the five severities of the self-generated CIFAR-10 corruptions (Section 5.6), the ranking of the four frozen CIFAR-10 CNNs by mean CS did not change (τ = 1 for every pair of severities). With only four models this is a weak test, and it concerns corruption severity, not operator strength.

Table 5: Clean misclassification-detection AUROC beyond the frozen CNNs (means over runs). $\mathrm { ^ { * } C S } > \mathrm { M S P ^ { * } }$ counts the runs in which CS had the higher AUROC.
<table><tr><td>Family (runs)</td><td>MSP</td><td>Margin</td><td>Best other</td><td>CS</td><td>LS</td><td> $\mathrm { C S } > \mathrm { M S P }$ </td></tr><tr><td>Transformers (36)</td><td>0.891</td><td>0.895</td><td>entropy 0.878</td><td>0.855</td><td>0.707</td><td>6/36 (all STL-10)</td></tr><tr><td>Native CIFAR-10 (42)</td><td>0.920</td><td>0.921</td><td>TTA mean conf. 0.939</td><td>0.904</td><td>0.781</td><td>3/42</td></tr><tr><td>13-model zoo (39)</td><td>0.865</td><td>0.870</td><td>TTA mean conf. 0.865</td><td>0.778</td><td>0.704</td><td>0/39</td></tr><tr><td>Last-layer ensembles (36)</td><td>0.851</td><td>n/a</td><td>ensemble MSP 0.854</td><td>0.786</td><td>0.678</td><td>0/36 vs ens. MSP</td></tr></table>

Table 6: Upper part: error-detection AUROC of CS with four definitions of CD, with $\alpha = 0 . 5$ and with α chosen on the calibration split (48 frozen-CNN runs; MSP reaches 0.876). Lower part: ResNet-20 features with retrained heads on CIFAR-10 (mean of three seeds).
<table><tr><td>CD definition</td><td colspan="2"> $\alpha = 0 . 5$ </td><td colspan="2">α from calibration</td><td>α = 0.5 chosen</td></tr><tr><td>Eq. (4), max-probability</td><td colspan="2">0.819</td><td colspan="2">0.819</td><td>58% of runs</td></tr><tr><td>Rescaled to [0, 1] on calibration</td><td colspan="2">0.833</td><td colspan="2">0.833</td><td>21%</td></tr><tr><td>Clean-class probability</td><td colspan="2">0.817</td><td colspan="2">0.817</td><td>44%</td></tr><tr><td>Total variation</td><td colspan="2">0.828</td><td colspan="2">0.828</td><td>46%</td></tr><tr><td>Head</td><td>Acc</td><td>Mean conf.</td><td>LS</td><td>CS, Eq. (5)</td><td>CS, total variation</td></tr><tr><td>Fully trained</td><td>0.921</td><td>0.963</td><td>0.943</td><td>0.955</td><td>0.942</td></tr><tr><td>Stochastic dropout rate 0.5</td><td>0.867</td><td>0.892</td><td>0.819</td><td>0.843</td><td>0.801</td></tr><tr><td>Stochastic dropout rate 0.9</td><td>0.429</td><td>0.508</td><td>0.249</td><td>0.477</td><td>0.346</td></tr><tr><td>Under-trained, 50 SGD steps</td><td>0.793</td><td>0.255</td><td>0.909</td><td>0.945</td><td>0.937</td></tr></table>

## 5.3. Reformulating CD and choosing α

Eq. (4) can reward a model that switches classes at constant confidence. Three alternative definitions are evaluated. The first is the change in the probability of the clean class, $| p _ { \hat { y } _ { i } } ( \mathbf x _ { i } ) - p _ { \hat { y } _ { i } } ( \mathbf x _ { i } ^ { ( k ) } ) |$ . The second is the total-variation distance between the clean and perturbed probability vectors. The third is Eq. (4) rescaled to [0 1] on the calibration split. The first two respond to a change of class. The rescaled version keeps the maximum-probability form and therefore keeps this property. Each is evaluated with α selected on the calibration split (Table 6, upper part). The total-variation and rescaled versions improve CS slightly (0.828 and 0.833 against 0.819), the clean-class version does not, and none comes within 0.04 of MSP. Choosing α on the calibration split gave the same test AUROC as $\alpha = 0 . 5$ to three decimals. Even the best α chosen on the test data improved AUROC by less than 0.001. The value $\alpha = 0 . 5$ is therefore used throughout, and CS is treated as a ranking score and not as an improvement over its components. Logit displacement is not used as a CD variant, because it is unbounded and depends on the logit scale. Rescaling changes the range of CD, not its calibration, and no temperature-scaled CD is evaluated.

The lower part of Table 6 reports heads retrained on ResNet-20 features. These heads test whether the variants expose abnormal models. With dropout rate 0.9 kept active at test time, LS is 0.249, and Eq. (4) lifts CS to 0.477. The total-variation variant gives 0.346. All variants separate stochastic-dropout heads from normal ones, since the 14 natively trained models have mean CS between 0.956 and 0.977. The opposite case is harder. A head trained for only 50 SGD steps has accuracy 0.793 but a mean confidence of 0.255, close to uniform. Its confidence barely moves under perturbation, so CD is near zero and CS stays at 0.937 to 0.945 under every variant. This is within 0.02 of the fully trained head (0.942 to 0.955) and inside the range of the 28 reference configurations in Table 3. The model is under-confident, not unstable. No consistency score detects it, and only the gap between accuracy and confidence does. The same occurred in the zoo, where an under-trained ShufleNetV2 head with ECE 0.59 had CS 0.949. Prediction (v) was therefore partly confirmed. The variants expose stochastic dropout, but nothing in DCF exposes under-confidence.

## 5.4. Why consistency is redundant on clean data

The clean-data results raise the question of why consistency adds so little. Proposition 3 predicts that LS can only beat the margin if the logit displacement separates correct from incorrect images at a fixed margin. We therefore binned the evaluation images of each of the 84 frozen runs (48 CNN, 36 transformer) into margin deciles. Within each bin with at least five correct and five incorrect images, we computed the

Table 7: Conformal no-flip certificate (Proposition 2) on 84 frozen runs. The guarantee is P(certified ∧ flip) ≤ β.
<table><tr><td> $\beta$ </td><td>Share certified</td><td>P(cert. ∧ flip)</td><td>Worst run</td><td>Runs violating</td><td>Acc. certified / not</td></tr><tr><td>0.01</td><td>12.6%</td><td>0.0000</td><td>0.001</td><td>0/84</td><td>0.977 / 0.841</td></tr><tr><td>0.05</td><td>23.2%</td><td>0.0002</td><td>0.003</td><td>0/84</td><td>0.989 / 0.826</td></tr><tr><td>0.10</td><td>29.5%</td><td>0.0005</td><td>0.005</td><td>0/84</td><td>0.992 / 0.814</td></tr></table>

(a) Coverage of the certificate  
![](images/9fd514a639c34a31781f8b3ad0f43347ee7ec057f68fbc6b795e82ac07d1597d.jpg)

(b) Realised violation rate  
![](images/95213c3a8c138f686400026a1c75fb31d8c67d614cfdd1924ae8fd5e3258dcd1.jpg)  
Figure 3: Conformal no-flip certificate. (a) Share of evaluation images certified at $\beta = 0 . 0 5 ,$ , by dataset and model family. (b) Realised P(certified ∧ flip) against the guarantee β (dashed), as the mean over 84 runs and for the worst run.

AUROC of the negative mean displacement. The size-weighted average was 0.515. It exceeded 0.5 in 37 of 84 runs, and the run-level Wilcoxon test against 0.5 gave $p = 0 . 6 7$ . LS reached 0.535 within bins. Adding the displacement and LS to a cross-validated logistic model on the margin changed AUROC by +0 003 on average. Over the whole evaluation set the displacement is almost uncorrelated with the margin (Spearman −0 02 for CNNs). The images that flip are therefore mostly those with small margins, and the margin already ranks them. This accounts for the negative results of Section 5.2. On clean data the operators move correct and incorrect images by similar amounts, and whether a move causes a flip is decided by a quantity the clean output already contains.

## 5.5. The certificate and the bounds

The certificate rests on the flip implication, which can be checked directly. The margin is a ${ \sqrt { 2 } } -$ -Lipschitz function of the logits, and the measured ratio $\lvert \gamma ( \mathbf { x } _ { i } ^ { ( k ) } ) - \gamma ( \mathbf { x } _ { i } ) \rvert / \lvert \lvert \Delta _ { i , k } \rvert$ ∥ never exceeded $\sqrt { 2 }$ (largest value 1.414). No image with $\gamma ( \mathbf { x } _ { i } ) > \ \sqrt { 2 } \vert \vert \Delta _ { i , k } \vert \vert _ { 2 }$ flipped. This confirms only that the logits were recorded and processed correctly (Remark 1).

Table 7 and Fig. 3 report the conformal certificate of Proposition 2 on all 84 frozen runs. The guarantee held in every run at every $\beta .$ The largest realised value of P(certified ∧ flip) was 0.003 at $\beta = 0 . 0 5$ . Certified images were also far more often correct (0.989 against 0.826 at $\beta = 0 . 0 5 )$ , so the certificate doubles as a cheap high-precision filter. Its coverage depends strongly on the setting. $\mathrm { A t } \beta = 0 . 0 5$ it certified 1.2% of CIFAR-10 images for CNNs but 63.5% of STL-10 images for transformers, where the margins of the models are large relative to their displacements. As a check of exchangeability, we also tested $P ( R ( \mathbf { x } ) > \hat { q } ) \leq \beta$ directly. It exceeded $\beta$ by more than 1.645 standard errors in 6 of 84 runs at each $\beta ,$ close to the 4.2 expected by chance at that level. Under shift the certificate certifies few images. On CIFAR-10 corruptions the calibration quantile computed on clean data certified only 0.02% to 0.8% of shifted images (against 0.1% to 3.3% of clean images in the same runs), and none of them flipped.

The certificate is a guarantee about flips, so a natural comparison is with per-sample flip bounds. We compared these bounds on tightness, calibration and validity. For each operator we bounded the flip probability of an image from its margin, using quantities measured on the calibration split only. We compared each bound with the observed flip rate in margin deciles of the evaluation split $( 2 , 8 8 0$ decile cells on the 48 CNN runs; mean observed flip rate 0.054). Brier scores are computed per decile cell unless marked per image. A bound shaped like a Lipschitz flip lemma equals one when the margin is below the largest calibration displacement and zero otherwise. It averaged 0.599, was trivial (equal to one) in 55% of cells and had a Brier score of 0.494. First- and second-moment Markov bounds averaged 0.484 and 0.388 (Brier 0.273 and 0.225). The empirical tail $P _ { \mathrm { c a l } } ( { \sqrt { 2 } } | | \Delta _ { k } | | _ { 2 } \geq \gamma )$ averaged 0.229 with Brier 0.098 but was exceeded in 0.07% of cells, because it is not a finite-sample guarantee. The DKW correction (95% confidence) gives a finite-sample guarantee. The corrected bound averaged 0.268 (Brier 0.148 per image) in this comparison. Evaluated on all 84 frozen runs, CNNs and transformers, it averaged 0.235 (Brier 0.124 per image) and was exceeded in none of 5,040 cells. The calibrated bounds are informative in a way the lemma-shaped bound is not, although they remain upper bounds about four to five times the observed flip rate.

![](images/0978dca7db689a40937d109827d75cd31e56def2e8289d30775b7b1b967677f3.jpg)  
MSP over-confidence (mean conf. − acc.)

![](images/63543bd27b6fc61aaa0cb3a57cfe616ee1127228b598b8e16462f761bee5e19a.jpg)  
MSP over-confidence (mean conf. − acc.)  
Figure 4: Oficial CIFAR-10-C, 14 natively trained models, one point per corruption (mean over models, seeds and severities). (a) AUROC of CS minus AUROC of the margin. (b) Gain in cross-validated AUROC from adding DCF features to confidence features when the combiner is fitted on labelled data from the same corruption. Both grow with MSP over-confidence; noise corruptions are the only family where CS beats the margin outright.

## 5.6. Distribution shift: where consistency helps and where it does not

The first question is whether the clean-data ordering survives a shift. Under the self-generated CIFAR-10 corruptions, CS detected errors less well than MSP (0.737 against 0.815; margin 0.807, energy 0.767). Adding LS and CD to a confidence-and-margin model left the AUROC unchanged at 0.814. On the oficial CIFAR-10-C with the 14 natively trained models, pooled over all 19 corruptions and five severities, CS (0.828) was again slightly below MSP (0.835), DOCTOR (0.836) and negative entropy (0.837).

Pooled results hide diferences between corruption families (Fig. 4a). On the four noise corruptions (Gaussian, shot, speckle, impulse), CS was better than the margin by 0.05 to 0.11 AUROC. On blur it was worse by about 0.04, and on weather and digital corruptions it was close to or below the margin. The diference follows how badly MSP is misled. Across the 19 corruptions, the gain from adding DCF features to confidence features grew with MSP over-confidence (mean confidence minus accuracy) with Spearman $\rho = 0 . 8 8 \ : ( p = 6 . 5 \times 1 0 ^ { - 7 }$ ; Fig. 4b). Noise is where confidence is most inflated (over-confidence 0.38 on average) and where the gain is largest (0.13 when the combiner is fitted on the same corruption). The most useful single operator under noise was blur. The score computed from blur alone reached 0.878 against 0.753 for the margin. Blur partly removes the noise, so a prediction that changes after blurring is likely to have been driven by the noise. This is the mechanism of feature squeezing [11]. It is also the situation in which the condition of Proposition 3 fails: at the same corrupted clean logits, correct and incorrect predictions respond diferently to a denoising operator.

The gain in Fig. 4b uses labelled examples of the corruption being tested, and deployed systems typically lack such examples. When the combiner was trained on clean data only, the gain on noise corruptions fell to +0 009. When it was trained on all other corruption families and tested on noise, the gain was +0 002 (not significant). For blur it became −0 046 and for digital corruptions −0 012. Only when other noise corruptions were in the training set (leave-one-corruption-out) did a gain survive on noise (+0 067). Consistency features therefore help only under a shift that has been observed and labelled.

Changing the operators does not remove this limitation. Increasing the strength of all six operators made CS worse, not better. Its AUROC fell from 0.903 at level 1 to 0.755 at level 5, while MSP stayed at 0.920.

Table 8: Pre-specified adversarial and OOD tests on 14 natively trained CIFAR-10 models (42 runs). AUROC for separating clean from attacked or OOD images (higher is better). The attack is non-adaptive $\ell _ { \infty }$ PGD.
<table><tr><td>Score</td><td>PGD 2/255</td><td>PGD  $4 / 2 5 5$ </td><td>PGD 8/255</td><td>SVHN</td><td>CIFAR-100</td></tr><tr><td>MSP</td><td>0.370</td><td>0.105</td><td>0.023</td><td>0.890</td><td>0.853</td></tr><tr><td>Energy</td><td>0.366</td><td>0.106</td><td>0.022</td><td>0.889</td><td>0.853</td></tr><tr><td>Mahalanobis</td><td>0.723</td><td>0.800</td><td>0.875</td><td>0.889</td><td>0.801</td></tr><tr><td>CS, all six operators</td><td>0.829</td><td>0.800</td><td>0.726</td><td>0.841</td><td>0.844</td></tr><tr><td>CS, noise operator only</td><td>0.583</td><td>0.224</td><td>0.061</td><td>0.910</td><td>0.846</td></tr><tr><td>LS</td><td>0.761</td><td>0.774</td><td>0.709</td><td>0.684</td><td>0.725</td></tr><tr><td>Embedding drift</td><td>0.914</td><td>0.936</td><td>0.888</td><td>0.768</td><td>0.811</td></tr></table>

When the level was chosen on the calibration split, level 1 (the setting of Table 2) was selected in 40 of 42 runs.

Finally, we asked whether clean-data scores predict accuracy under shift. For label-free accuracy estimation under the self-generated shift, average thresholded confidence with MSP had a mean absolute error of 0.049, against 0.124 when CS replaced MSP and 0.091 for average confidence. At the model level, in the 13-model zoo, clean accuracy predicted accuracy on held-out corruption families almost perfectly (Spearman 0.99), and clean CS added nothing once accuracy was controlled for (partial Spearman +0 08). Clean CS did correlate with efective robustness, the residual from the accuracy-on-the-line fit [32] (Spearman 0.79). The correlation was driven by one anomalous model, the ShufleNetV2 head with ECE 0.59. With that model removed and the line refitted on a logit scale, it fell to 0.19 $( p = 0 . 5 6 )$ .

## 5.7. Thresholds and regimes

The same question applies to the regime labels of CAG. The labels of Eq. (6) depend on δ. Moving δ from 0.02 to 0.10 changes the label of 11 of the 16 CNN configurations. $\operatorname { A t } \delta = 0 . 0 2$ , 15 are consistency-dominant, and at $\delta = 0 . 1 0$ only the four OCTMNIST models are. No configuration is accuracy-dominant at any δ in the grid. The clean-data quantities are also evaluated as predictors of behaviour under shift. A shift alarm with a threshold chosen on some corruptions and tested on others reached a balanced accuracy of 0.72 with CS, 0.76 with LS, 0.91 with the margin and 0.94 with MSP. Across the CIFAR-10 runs, clean CAG did not predict the drop in accuracy under shift (Spearman 0.09, $p = 0 . 7 8 )$ . A consistency floor τ on mean CS, used to separate unstable from stable models, would not bind. Every configuration in Table 3 has mean CS above 0.91, and the only heads below any reasonable floor are the artificial stochastic-dropout heads of Table 6. Neither δ nor τ is validated on independent data, so δ is a descriptive tolerance without an operational interpretation.

## 5.8. Adversarial and out-of-distribution inputs

Adversarial and OOD inputs test the settings in which confidence is expected to fail. Table 8 summarises the pre-specified tests on the 14 natively trained models. Prediction (i) was confirmed. Against non-adaptive PGD, CS separated clean from attacked images with AUROC 0.83, 0.80 and 0.73 at $\epsilon _ { \infty } = 2 $ , 4 and 8/255, while MSP fell to 0.370, 0.105 and 0.023, because the attack makes the wrong class very confident. CS beat MSP and the margin in 42 of 42 runs at every budget. It also ranked errors on the mixed clean-plus-attacked set far better $_ { ( 0 . 8 7 }$ against 0.32 at $\epsilon _ { \infty } = 2 / 2 5 5 )$ ). CS was not the best detector, however. The displacement of the penultimate embedding under the same operators reached 0.89 to 0.94, and the Mahalanobis score overtook CS at $\epsilon _ { \infty } = 8 / 2 5 5$ (0.88 against 0.73). Predictions (ii) and (iii) were also confirmed. On SVHN, CS (0.841) was below energy and Mahalanobis (both 0.889), and on CIFAR-100 it was below MSP (0.844 against 0.853). One result was not predicted: a CS computed from the noise operator alone detected SVHN better than MSP (0.910 against 0.890, 39 of 42 runs). Prediction (iv) failed in one respect, since TTA mean confidence beat MSP on clean data (0.939 against 0.920). This score averages views rather than measuring their disagreement (Section 5.2).

Detection against an attack that ignores the detector is weak evidence [52]. Under the adaptive attack of Eq. (10) (Table 9, Fig. 5b), the attacker can evade DCF. With λ = 5 the detection AUROC of CS drops to 0.09 at $\epsilon _ { \infty } = 4 / 2 5 5$ and 0.05 at $\epsilon _ { \infty } = 8 / 2 5 5$ , below chance, because the attacked images become more stable than clean ones. Evasion has a cost. With λ = 5 only 6% (at 4/255) and 9% (at 8/255) of attacks still change the prediction, against 98% and 100% for plain PGD. At every λ, CS ranked the resulting errors well (AUROC 0.79 to 0.87), while MSP did not (0.06 to 0.50). Under this one objective, the attacker can either fool the model or stay stable under the operators, but rarely both. Stronger adaptive objectives were not tested, and no claim of adversarial robustness is made.

weight λ on the DCF-evasion term  
Table 9: Adaptive PGD against DCF $\left( \mathrm { E q . } \right.$ (10)) on four natively trained models, three seeds each. Detection: clean versus attacked. Error detection: correct versus incorrect predictions on attacked images.
<table><tr><td> $\epsilon _ { \infty }$ </td><td>λ</td><td>Attack success</td><td>CS detection</td><td>CS error det.</td><td>MSP error det.</td></tr><tr><td>4/255</td><td>0</td><td>0.98</td><td>0.81</td><td>0.85</td><td>0.10</td></tr><tr><td>4/255</td><td>1</td><td>0.70</td><td>0.67</td><td>0.87</td><td>0.23</td></tr><tr><td>4/255</td><td>5</td><td>0.06</td><td>0.09</td><td>0.86</td><td>0.50</td></tr><tr><td>8/255</td><td>0</td><td>1.00</td><td>0.76</td><td>0.80</td><td>0.06</td></tr><tr><td>8/255</td><td>1</td><td>0.86</td><td>0.67</td><td>0.79</td><td>0.11</td></tr><tr><td>8/255</td><td>5</td><td>0.09</td><td>0.05</td><td>0.82</td><td>0.43</td></tr></table>

![](images/9c14cb4f15cd2527c29a082528dcfe9c8365b0e58c7baab05d1b10d59624c9c1.jpg)

![](images/547d2b40ba12d4006994d9c56f54c49f100da347b6b24a94d670e46c7c22864d.jpg)  
Figure 5: Adversarial inputs. (a) Clean-versus-attacked AUROC under non-adaptive PGD; MSP falls below chance because the attack raises confidence. (b) Adaptive PGD (Eq. (10)): as the weight λ on stability grows, CS detection collapses, but so does the attack success rate.

## 5.9. Stable but wrong: OCTMNIST

The negative CAG on OCTMNIST is far larger than on the other datasets (Table 3). One reading is backbone-domain mismatch: an ImageNet backbone with a short linear probe may produce stable features that do not separate retinal classes. Adaptation controls evaluate this reading directly (Table 10).

If weak adaptation caused the gap, stronger adaptation should remove it. It does not: CAG stayed below −0 10 in all 33 OCTMNIST runs. These are 11 configurations with three seeds each. Eight are frozen CNNs: each of the four CNNs of Table 3 with a head trained for 5 epochs, and with a head trained for up to 30 epochs with early stopping. The other three are ResNet-18 and MobileNetV2 with full fine-tuning, and ResNet-18 trained from scratch. The transformers are not among these runs. Table 10 lists the configurations involving ResNet-18 and MobileNetV2. Longer head training changed nothing. Full fine-tuning raised accuracy by about 0.1 and improved CAG by 0 145 ± 0 077 (ResNet-18) and 0 158 ± 0 012 (MobileNetV2), but the gap stayed at −0 12 to −0 16. On Intel-Scene the gap stayed between −0 05 and −0 07 for frozen and fine-tuned models and was larger (−0 11) for the less accurate model trained from scratch.

Backbone-domain mismatch is also tested directly. A ResNet-18 trained only on OCTMNIST, which has no ImageNet features to mismatch, still had CAG −0 206. At matched accuracy the comparison is sharper. At the epoch where the in-domain model reached the accuracy of the frozen ImageNet model (0.620), its CAG was −0 331, against −0 308 for the frozen model. On Intel-Scene, a fine-tuned model at matched accuracy had the same gap as the frozen one (−0 068 against −0 067). These controls do not support backbone-domain mismatch as the sole explanation. They do not exclude a contribution from it, because each matched-accuracy comparison uses one point per trajectory.

Table 10: Adaptation controls (mean of three seeds). Frozen: ImageNet backbone with a linear head. ES: early stopping. The scratch model sees no ImageNet data. Validation-split CAG uses the oficial OCTMNIST validation split, which has the training prior.
<table><tr><td>Dataset</td><td>Model, adaptation</td><td>Acc</td><td>LS</td><td>CS</td><td>CAG (test)</td><td>CAG (val)</td></tr><tr><td rowspan="5">OCTMNIST</td><td>ResNet-18, frozen, 5 epochs</td><td>0.629</td><td>0.913</td><td>0.930</td><td>-0.301</td><td>-0.128</td></tr><tr><td>ResNet-18, frozen, ≤30 epochs + ES</td><td>0.622</td><td>0.917</td><td>0.931</td><td>-0.310</td><td>-0.124</td></tr><tr><td>ResNet-18, full fine-tuning</td><td>0.742</td><td>0.877</td><td>0.898</td><td>-0.156</td><td>-0.021</td></tr><tr><td>ResNet-18, from scratch (in-domain)</td><td>0.753</td><td>0.953</td><td>0.959</td><td>-0.206</td><td>-0.078</td></tr><tr><td>MobileNetV2, frozen, 5 epochs MobileNetV2, full fine-tuning</td><td>0.651 0.739</td><td>0.918 0.813</td><td>0.932</td><td>-0.281</td><td>-0.098</td></tr><tr><td rowspan="4">Intel-Scene</td><td></td><td></td><td></td><td>0.862</td><td>-0.123</td><td>+0.002</td></tr><tr><td>ResNet-18, frozen, 5 epochs</td><td>0.911</td><td>0.978</td><td>0.977</td><td>-0.066</td><td>n/a</td></tr><tr><td>ResNet-18, full fine-tuning</td><td>0.924</td><td>0.981</td><td>0.981</td><td>-0.058</td><td>n/a</td></tr><tr><td>ResNet-18, from scratch (in-domain)</td><td>0.865</td><td>0.975</td><td>0.977</td><td>-0.112</td><td>n/a</td></tr></table>

Table 11: Induced label shift on CIFAR-10 (ResNet-18, frozen, three seeds). Two classes are kept at the given fraction of their training images; the test set is balanced. The 100% row is a separate run and difers slightly from the value in Table 3.
<table><tr><td>Kept</td><td>Acc</td><td>Rare recall</td><td>Rare LS</td><td>CAG</td><td>CAG, prior corrected</td><td>CAG, re-weighted</td></tr><tr><td>100%</td><td>0.863</td><td>0.77</td><td>0.87</td><td>-0.060</td><td>-0.060</td><td>-0.060</td></tr><tr><td>10%</td><td>0.821</td><td>0.48</td><td>0.81</td><td>-0.101</td><td>-0.070</td><td>-0.040</td></tr><tr><td>2%</td><td>0.749</td><td>0.09</td><td>0.82</td><td>-0.175</td><td>-0.083</td><td>-0.031</td></tr></table>

A second explanation is label shift, which accounts for part of the gap. The OCTMNIST training and validation splits have the class prior CNV 0.344, DME 0.105, drusen 0.080 and normal 0.472, while the oficial test split is balanced. Models trained on the skewed prior rarely predict the minority classes. Drusen recall was 12% for the frozen ResNet-18 and 33% after full fine-tuning, and the wrong predictions for drusen images were as stable as any others. On the validation split, which shares the training prior, the gap nearly vanishes for fine-tuned models (−0 021 and +0 002) and is much smaller for frozen ones (about −0 10 to −0 13; Table 10 and Fig. 6a). Two experiments test this causally.

First, we corrected the saved test logits for the prior shift by adding the diference log $\pi _ { \mathrm { t e s t } } - \log \pi _ { \mathrm { t r a i n } }$ between the log class priors of the test data $( \pi _ { \mathrm { t e s t } } )$ and of the training data $( \pi _ { \mathrm { t r a i n } } )$ [53]. With the true (balanced) test prior, mean CAG over all OCTMNIST models and modes improved from −0 264 to −0 162 (32 of 33 runs improved), and drusen recall rose from 14% to 53%. With a prior estimated without labels by expectationmaximisation (EM), CAG improved to −0 224 (29 of 33 runs). Prior correction thus recovers about 40% of the gap on OCTMNIST. The rest remains on the test split, and a residual of about −0 10 to −0 13 is present on the validation split for frozen models. There, CS saturates above accuracy.

Second, we induced the same kind of shift on CIFAR-10, where the frozen ResNet-18 has a small gap. Heads were trained with two classes (bird and cat) cut to 10% or 2% of their images and tested on the balanced test set (Fig. 6b and Table 11). At 2%, the rare classes were recognised 9% of the time, yet their LS stayed at 0.82, and CAG fell from −0 060 to −0 175. Correcting the prior brought it back to −0 083, and re-weighting the test set to the training prior brought it to −0<sub>.</sub>031. Stable but wrong behaviour can thus be induced by label shift and largely removed by correcting it.

## 6. Discussion

## 6.1. What the results say about perturbation consistency

On clean data and under shifts that the system has not seen, perturbation consistency was a costlier and noisier re-expression of confidence. This held for frozen and fully trained CNNs, for transformers, for a varied zoo of architectures and against small ensembles. Proposition 3 and the within-decile test of Section 5.4 explain why. Whether a mild operator flips a prediction is decided mostly by the clean margin, and at a fixed margin the size of the displacement says almost nothing about correctness. A practitioner who wants to rank predictions by trustworthiness should therefore use MSP, the margin or the entropy. TTA mean confidence is a further option if a few extra forward passes are afordable.

![](images/dc24515f7505a5b8f55cc8798f49ae793ae28b9f99eff7434aabaccc29d24437.jpg)

![](images/ee4af20ce7e1621c05d6bd57edcdf6ff8ea04532100d7913279836a7aa6d0f22.jpg)  
Figure 6: Stable but wrong. (a) CAG of ResNet-18 on OCTMNIST under four adaptation modes, on the balanced test split and on the validation split that shares the training prior. (b) Induced label shift on CIFAR-10: CAG worsens as two classes become rare, and prior correction or re-weighting removes most of the efect. The dotted line marks CAG = −0 05, the boundary for δ = 0 05.

The exceptions follow the same logic. Proposition 3 fails when the clean output is itself corrupted, so that two images with the same clean logits difer in how much of that output is real. Additive noise and adversarial perturbations both raise confidence in the wrong class, and a smoothing operator partly undoes them. In those settings consistency carried information that confidence lacked, the size of the gain followed the degree of over-confidence, and a pre-specified prediction was confirmed. DCF is therefore best applied as a check on inputs whose confidence is known to be unreliable, such as noise corruptions and adversarial perturbations. It is preferably computed in feature space rather than from predicted labels. Its limits against adaptive attackers are stated in the results.

## 6.2. What CAG measures

Rewriting CAG clarifies its behaviour. Because $\mathbf { C A G } = \mathbf { A c c - } \bar { C } = ( 1 - \bar { C } ) - ( 1 - \mathbf { A c c } )$ , the gap is the mean instability minus the error rate. For the models in Table 3, mean instability 1 − C<sup>¯</sup> was 0.01 to 0.09, so CAG is negative whenever the error rate exceeds that, which is almost always. A large negative CAG therefore says that many predictions are wrong but stable. Section 5.9 shows one common reason: a class-prior mismatch that makes the model confidently and consistently predict the majority classes. CAG is thus best read as a prompt to check class priors and per-class recall, and not as a summary of reliability. It does not predict accuracy loss under shift, and its regime labels depend on an arbitrary δ.

## 6.3. Scope and limitations

The conclusions apply to a defined regime. Most experiments use images at CIFAR-10 to STL-10 scale, and ImageNet-scale backbones serve as frozen feature extractors. The six operators are mild by design, and stronger settings made CS worse (Section 5.6). Label preservation under these operators was screened automatically with a vision-language model on three of the four datasets. It was not validated by human raters, and the study does not claim that every perturbed image preserves its label.

Several conditions limit how far the model comparisons extend. The natively trained models and the zoo backbones were trained once by others, so for them the seeds vary the sampling and the noise but not the training. The ensembles are last-layer ensembles on shared features, which are less diverse than independently trained deep ensembles. The adaptive attack uses one objective with 20 steps. Stronger or diferently weighted attacks may break the trade-of observed here, and no claim of adversarial robustness is made. The conformal certificate gives a marginal guarantee under exchangeability. On OCTMNIST the calibration images come from the validation split, whose class prior difers from the balanced test split, so the guarantee there is not formally covered, even though no violations occurred.

The corruption and OCTMNIST analyses have their own scope. The zoo analysis uses self-generated corruptions and not the oficial CIFAR-10-C files. Flip rates are measured for the six operators, and ImageNet-P, which exists only for ImageNet, is not used. Prior correction recovers part of the OCTMNIST gap, and the remainder is described and not explained. The in-domain control is a network trained from scratch, not a backbone pretrained on retinal images such as RETFound [77].

## 7. Conclusion

We asked when the decision consistency of an image classifier, measured under six mild perturbations, says something about the correctness of a prediction that the clean softmax output does not.

On clean data the answer was negative. Across 201 evaluations, the consistency score was below MSP for misclassification detection in every model family. It did not add information when combined with MSP, and it cost 8 to 15 times more compute. An exact inequality explains this redundancy: a label flip requires the clean logit margin to be at most the square root of two times the norm of the logit displacement. The same inequality yields a label-free conformal certificate, whose guarantee held in every run.

The negative answer has two exceptions. Consistency did add information where confidence is misled, namely under noise corruptions and adversarial perturbations. The gain did not transfer to unseen kinds of shift, and an adaptive attacker can trade attack success for stability.

A separate finding concerns the consistency-accuracy gap. The gap is large and negative on OCTMNIST, and backbone mismatch alone does not explain it. Part of it is a class-prior efect. Prior correction recovers about 40% of the gap, and inducing label shift on CIFAR-10 reproduces the pattern. The protocol provides a baseline-controlled way to evaluate future perturbation-based reliability scores.

## CRediT authorship contribution statement

Amit Kalita: Conceptualization, Methodology, Software, Investigation, Formal analysis, Visualization, Writing (original draft). Moirangthem Tiken Singh: Conceptualization, Methodology, Supervision, Validation, Writing (review and editing).

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of Generative AI and AI Assisted Technologies in the Writing Process

During the preparation of this work the authors used generative artificial intelligence tools to improve language clarity and to help with analysis code. After using these tools, the authors reviewed and edited the content and code as needed and take full responsibility for the content of the publication.

## Data Availability

All datasets used in this study are publicly available from their original creators: CIFAR-10, CIFAR-10-C, CIFAR-100, STL-10, SVHN, the Intel Image Classification dataset and OCTMNIST (MedMNIST v2).

## Code Availability

The code for every experiment, the logged clean and perturbed logits of every run, the per-sample tables and the operator audit files are available from the corresponding author and will be released in a public repository upon publication.

## References

[1] D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, and D. Mané, “Concrete Problems in AI Safety,” 2016. [Online]. Available: https://arxiv.org/abs/1606.06565

[2] E. Tjoa and C. Guan, “A Survey on Explainable Artificial Intelligence (XAI): Toward Medical XAI,” IEEE Transactions on Neural Networks and Learning Systems, vol. 32, no. 11, pp. 4793–4813, 2021. [Online]. Available: https://doi.org/10.1109/TNNLS.2020.3027314

[3] D. Hendrycks and K. Gimpel, “A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks,” in International Conference on Learning Representations, 2017. [Online]. Available: https://openreview.net/forum?id=Hkg4TI9xl

[4] W. Liu, X. Wang, J. Owens, and Y. Li, “Energy-based Out-of-distribution Detection,” in Advances in Neural Information Processing Systems, vol. 33. Curran Associates, Inc., 2020, pp. 21 464–21 475. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2020/hash/ f5496252609c43eb8a3d147ab9b9c006-Abstract.html

[5] P. F. Jaeger, C. T. Lüth, L. Klein, and T. J. Bungert, “A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification,” in International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=YnkGMIh0gvX

[6] I. Galil, M. Dabbah, and R. El-Yaniv, “What Can We Learn From the Selective Prediction and Uncertainty Estimation Performance of 523 ImageNet Classifiers?” in International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=p66AzKi6Xim

[7] L. F. Cattelan and D. Silva, “How to Fix a Broken Confidence Estimator: Evaluating Post-hoc Methods for Selective Classification with Deep Neural Networks,” in Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence, ser. Proceedings of Machine Learning Research, vol. 244. PMLR, 2024, pp. 547–584. [Online]. Available: https://proceedings.mlr.press/v244/cattelan24a.html

[8] M. S. Ayhan and P. Berens, “Test-time Data Augmentation for Estimation of Heteroscedastic Aleatoric Uncertainty in Deep Neural Networks,” in Medical Imaging with Deep Learning, 2018. [Online]. Available: https://openreview.net/forum?id=rJZz-knjz

[9] D. Shanmugam, D. Blalock, G. Balakrishnan, and J. Guttag, “Better Aggregation in Test-Time Augmentation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 1194–1203. [Online]. Available: https://doi.org/10.1109/ICCV48922.2021.00125

[10] D. Hendrycks and T. Dietterich, “Benchmarking Neural Network Robustness to Common Corruptions and Perturbations,” in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview.net/forum?id=HJz6tiCqYm

[11] W. Xu, D. Evans, and Y. Qi, “Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks,” in Network and Distributed System Security Symposium (NDSS), 2018. [Online]. Available: https://doi.org/10.14722/ndss.2018.23198

[12] D. Hendrycks, S. Basart, M. Mazeika, A. Zou, J. Kwon, M. Mostajabi, J. Steinhardt, and D. Song, “Scaling Out-of-Distribution Detection for Real-World Settings,” in Proceedings of the 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 162. PMLR, 2022, pp. 8759–8773. [Online]. Available: https: //proceedings.mlr.press/v162/hendrycks22a.html

[13] F. Granese, M. Romanelli, D. Gorla, C. Palamidessi, and P. Piantanida, “DOCTOR: A Simple Method for Detecting Misclassification Errors,” in Advances in Neural Information Processing Systems, vol. 34. Curran Associates, Inc., 2021, pp. 5669–5681. [Online]. Available: https://proceedings.neurips.cc/ paper\_files/paper/2021/hash/2cb6b10338a7fc4117a80da24b582060-Abstract.html

[14] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” in Proceedings of the International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 70. PMLR, 2017, pp. 1321–1330. [Online]. Available: https://proceedings.mlr.press/v70/guo17a.html

[15] E. D. C. Gomes, M. Romanelli, G. Pichler, and P. Piantanida, “A Data-Driven Measure of Relative Uncertainty for Misclassification Detection,” in International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=ruGY8v10mK

[16] K. Lee, K. Lee, H. Lee, and J. Shin, “A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks,” in Advances in Neural Information Processing Systems, vol. 31. Curran Associates, Inc., 2018. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper 2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html

[17] J. Mukhoti, A. Kirsch, J. van Amersfoort, P. H. S. Torr, and Y. Gal, “Deep Deterministic Uncertainty: A New Simple Baseline,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 24 384–24 394. [Online]. Available: https://doi.org/10.1109/CVPR52729.2023.02336

[18] J. Yang, K. Zhou, Y. Li, and Z. Liu, “Generalized Out-of-Distribution Detection: A Survey,” International Journal of Computer Vision, vol. 132, no. 12, pp. 5635–5662, 2024. [Online]. Available: https://doi.org/10.1007/s11263-024-02117-4

[19] F. Zhu, Z. Cheng, X.-Y. Zhang, and C.-L. Liu, “OpenMix: Exploring Outlier Samples for Misclassification Detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12 074–12 083. [Online]. Available: https://doi.org/10.1109 CVPR52729.2023.01162

[20] F. Zeng, Z. Cheng, F. Zhu, and X.-Y. Zhang, “Towards Eficient and General-Purpose Few-Shot Misclassification Detection for Vision-Language Models,” 2025. [Online]. Available: https://arxiv.org/abs/2503.20492

[21] K. X. Nguyen, T. Li, and X. Peng, “Interpretable Failure Detection with Human-Level Concepts,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, 2025, pp. 26 326–26 334. [Online]. Available: https://doi.org/10.1609/aaai.v39i25.34831

[22] P. Steinmetz, F. Frouin, V. Morard, and I. Buvat, “Failure Detection in Medical Image Classification under Realistic Distribution Shifts: A Large-Scale Benchmark,” medRxiv preprint, 2026. [Online]. Available: https://doi.org/10.64898/2026.05.04.26350496

[23] A. Azulay and Y. Weiss, “Why Do Deep Convolutional Networks Generalize So Poorly to Small Image Transformations?” Journal of Machine Learning Research, vol. 20, no. 184, pp. 1–25, 2019. [Online]. Available: https://jmlr.org/papers/v20/19-519.html

[24] L. Engstrom, B. Tran, D. Tsipras, L. Schmidt, and A. Madry, “Exploring the Landscape of Spatial Robustness,” in Proceedings of the 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97. PMLR, 2019, pp. 1802–1811. [Online]. Available: https://proceedings.mlr.press/v97/engstrom19a.html

[25] T. Miyato, S.-I. Maeda, M. Koyama, and S. Ishii, “Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 8, pp. 1979–1993, 2019. [Online]. Available: https://doi.org/10.1109/TPAMI.2018.2858821

[26] D. Hendrycks, N. Mu, E. D. Cubuk, B. Zoph, J. Gilmer, and B. Lakshminarayanan, “AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty,” in International Conference on Learning Representations, 2020. [Online]. Available: https://openreview.net/forum?id=S1gmrxHFvB

[27] M. Zhang, S. Levine, and C. Finn, “MEMO: Test Time Robustness via Adaptation and Augmentation,” in Advances in Neural Information Processing Systems, ser. NIPS ’22, vol. 35. Curran Associates, Inc., 2022, pp. 38 629–38 642. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper 2022/hash/fc28053a08f59fccb48b11f2e31e81c7-Abstract-Conference.html

[28] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell, “Tent: Fully Test-Time Adaptation by Entropy Minimization,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=uXl3bZLkr3c

[29] Y. Ovadia, E. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin, J. Dillon, B. Lakshminarayanan, and J. Snoek, “Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift,” in Advances in Neural Information Processing Systems, vol. 32. Curran Associates, Inc., 2019. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2019/ hash/8558cb408c1d76621371888657d2eb1d-Abstract.html

[30] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles,” in Advances in Neural Information Processing Systems, vol. 30. Curran Associates, Inc., 2017, pp. 6402–6413. [Online]. Available: https://proceedings.neurips.cc/ paper\_files/paper/2017/hash/9ef2ed4b7fd2c810847fa5fa85bce38-Abstract.html

[31] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,” in Proceedings of the International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 48. PMLR, 2016, pp. 1050–1059. [Online]. Available: https://proceedings.mlr.press/v48/gal16.html

[32] J. P. Miller, R. Taori, A. Raghunathan, S. Sagawa, P. W. Koh, V. Shankar, P. Liang, Y. Carmon, and L. Schmidt, “Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 139. PMLR, 2021, pp. 7721–7735. [Online]. Available: https://proceedings.mlr.press/v139/miller21b.html

[33] R. Taori, A. Dave, V. Shankar, N. Carlini, B. Recht, and L. Schmidt, “Measuring robustness to natural distribution shifts in image classification,” in Advances in Neural Information Processing Systems, ser. NIPS ’20, vol. 33. Curran Associates, Inc., 2020, pp. 18 583–18 599. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2020/hash d8330f857a17c53d217014ee776bfd50-Abstract.html

[34] Y. Jiang, V. Nagarajan, C. Baek, and J. Z. Kolter, “Assessing Generalization of SGD via Disagreement,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/forum?id=WvOGCEAQhxl

[35] C. Baek, Y. Jiang, A. Raghunathan, and J. Z. Kolter, “Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift,” in Advances in Neural Information Processing Systems, vol. 35. Curran Associates, Inc., 2022, pp. 19 274–19 289. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ 7a8d388b7a17df480856df1cc079b08-Abstract-Conference.html

[36] S. Garg, S. Balakrishnan, Z. C. Lipton, B. Neyshabur, and H. Sedghi, “Leveraging Unlabeled Data to Predict Out-of-Distribution Performance,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/forum?id=o\_HsiMPYh\_x

[37] D. Guillory, V. Shankar, S. Ebrahimi, T. Darrell, and L. Schmidt, “Predicting with Confidence on Unseen Distributions,” in Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 2021, pp. 1114–1124. [Online]. Available: https://doi.org/10.1109/ICCV48922.2021.00117

[38] M. Hein and M. Andriushchenko, “Formal Guarantees on the Robustness of a Classifier against Adversarial Manipulation,” in Advances in Neural Information Processing Systems, vol. 30. Curran Associates, Inc., 2017. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2017/ hash/e077e1a544eec4f0307cf5c3c721d944-Abstract.html

[39] Y. Tsuzuku, I. Sato, and M. Sugiyama, “Lipschitz-Margin Training: Scalable Certification of Perturbation Invariance for Deep Neural Networks,” in Advances in Neural Information Processing Systems, vol. 31. Curran Associates, Inc., 2018. [Online]. Available: https://proceedings.neurips.cc paper\_files/paper/2018/hash/485843481a7edacbfce101ecb1e4d2a8-Abstract.html

[40] T.-W. Weng, H. Zhang, P.-Y. Chen, J. Yi, D. Su, Y. Gao, C.-J. Hsieh, and L. Daniel, “Evaluating the Robustness of Neural Networks: An Extreme Value Theory Approach,” in International Conference on Learning Representations, 2018. [Online]. Available: https://openreview.net/forum?id=BkUHlMZ0b

[41] M. Fazlyab, T. Entesari, A. Roy, and R. Chellappa, “Certified robustness via dynamic margin maximization and improved lipschitz regularization,” in Advances in Neural Information Processing Systems, vol. 36. Curran Associates, Inc., 2023, pp. 34 451–34 464. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2023/hash 6c7ca1889f01a9b767c631686fb5fd24-Abstract-Conference.html

[42] J. M. Cohen, E. Rosenfeld, and J. Z. Kolter, “Certified Adversarial Robustness via Randomized Smoothing,” in Proceedings of the International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97. PMLR, 2019, pp. 1310–1320. [Online]. Available: https://proceedings.mlr.press/v97/cohen19c.html

[43] A. N. Angelopoulos, S. Bates, M. I. Jordan, and J. Malik, “Uncertainty Sets for Image Classifiers using Conformal Prediction,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=eNdiU\_DbM9

[44] A. N. Angelopoulos and S. Bates, “Conformal Prediction: A Gentle Introduction,” Foundations and Trends in Machine Learning, vol. 16, no. 4, pp. 494–591, 2023. [Online]. Available: https://doi.org/10.1561/2200000101

[45] J. Lei, M. G’Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman, “Distribution-Free Predictive Inference for Regression,” Journal of the American Statistical Association, vol. 113, no. 523, pp. 1094–1111, 2018. [Online]. Available: https://doi.org/10.1080/01621459.2017.1307116

[46] R. J. Tibshirani, R. F. Barber, E. Candès, and A. Ramdas, “Conformal Prediction Under Covariate Shift,” in Advances in Neural Information Processing Systems, vol. 32. Curran Associates, Inc., 2019. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2019/ hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html

[47] R. F. Barber, E. J. Candès, A. Ramdas, and R. J. Tibshirani, “Conformal Prediction Beyond Exchangeability,” The Annals of Statistics, vol. 51, no. 2, pp. 816–845, 2023. [Online]. Available: https://doi.org/10.1214/23-AOS2276

[48] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards Deep Learning Models Resistant to Adversarial Attacks,” in International Conference on Learning Representations, 2018. [Online]. Available: https://openreview.net/forum?id=rJzIBfZAb

[49] F. Croce and M. Hein, “Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks,” in Proceedings of the 37th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 119. PMLR, 2020, pp. 2206–2216. [Online]. Available: https://proceedings.mlr.press/v119/croce20b.html

[50] N. Carlini and D. Wagner, “Towards Evaluating the Robustness of Neural Networks,” in Proceedings of the IEEE Symposium on Security and Privacy, 2017, pp. 39–57. [Online]. Available: https://doi.org/10.1109/SP.2017.49

[51] A. Athalye, N. Carlini, and D. Wagner, “Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples,” in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 80. PMLR, 2018, pp. 274–283. [Online]. Available: https://proceedings.mlr.press/v80/athalye18a.html

[52] F. Tramèr, N. Carlini, W. Brendel, and A. Madry, “On Adaptive Attacks to Adversarial Example Defenses,” in Advances in Neural Information Processing Systems, vol. 33. Curran Associates, Inc., 2020, pp. 1633–1645. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2020/hash 11f38f8ecd71867b42433548d1078e38-Abstract.html

[53] M. Saerens, P. Latinne, and C. Decaestecker, “Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure,” Neural Computation, vol. 14, no. 1, pp. 21–41, 2002. [Online]. Available: https://doi.org/10.1162/089976602753284446

[54] Z. C. Lipton, Y.-X. Wang, and A. J. Smola, “Detecting and Correcting for Label Shift with Black Box Predictors,” in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 80. PMLR, 2018, pp. 3122–3130. [Online]. Available: https://proceedings.mlr.press/v80/lipton18a.htm

[55] A. Alexandari, A. Kundaje, and A. Shrikumar, “Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation,” in Proceedings of the 37th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 119. PMLR, 2020, pp. 222–232. [Online]. Available: https://proceedings.mlr.press/v119/alexandari20a.html

[56] S. Garg, Y. Wu, S. Balakrishnan, and Z. C. Lipton, “A Unified View of Label Shift Estimation,” in Advances in Neural Information Processing Systems, vol. 33. Curran Associates, Inc., 2020, pp. 3290–3300. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2020/hash 219e052492f4008818b8adb6366c7ed6-Abstract.html

[57] Y. Zhang, B. Kang, B. Hooi, S. Yan, and J. Feng, “Deep Long-Tailed Learning: A Survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 10 795–10 816, 2023. [Online]. Available: https://doi.org/10.1109/TPAMI.2023.3268118

[58] P. Massart, “The Tight Constant in the Dvoretzky–Kiefer–Wolfowitz Inequality,” The Annals of Probability, vol. 18, no. 3, pp. 1269–1283, 1990. [Online]. Available: https://doi.org/10.1214/aop/ 1176990746

[59] A. Krizhevsky, “Learning multiple layers of features from tiny images,” University of Toronto, Tech. Rep., 2009. [Online]. Available: https://www.cs.toronto.edu/\~kriz/learning-features-2009-TR.pdf

[60] A. Coates, A. Ng, and H. Lee, “An Analysis of Single-Layer Networks in Unsupervised Feature Learning,” in Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, vol. 15. PMLR, 2011, pp. 215–223. [Online]. Available: https://proceedings.mlr.press/v15/coates11a.html

[61] P. Bansal, “Intel Image Classification Dataset.” [Online]. Available: https://www.kaggle.com/datasets puneet6060/intel-image-classification

[62] J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, and B. Ni, “MedMNIST v2 - A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification,” Scientific Data, vol. 10, no. 1, p. 41, 2023. [Online]. Available: https://doi.org/10.1038/s41597-022-01721-8

[63] D. S. Kermany, M. Goldbaum, W. Cai et al., “Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning,” Cell, vol. 172, no. 5, pp. 1122–1131.e9, 2018. [Online]. Available: https://doi.org/10.1016/j.cell.2018.02.010

[64] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778. [Online]. Available: https://doi.org/10.1109/CVPR.2016.90

[65] K. Simonyan and A. Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” in International Conference on Learning Representations, 2015. [Online]. Available: https://arxiv.org/abs/1409.1556

[66] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 4510–4520. [Online]. Available: https://doi.org/10.1109/CVPR.2018.00474

[67] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=YicbFdNTTy

[68] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jégou, “Training Data-Eficient Image Transformers & Distillation through Attention,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 139. PMLR, 2021, pp. 10 347–10 357. [Online]. Available: https://proceedings.mlr.press/v139/touvron21a.html

[69] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 9992–10 002. [Online]. Available: https://doi.org/10.1109/ICCV48922.2021.00986

[70] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A ConvNet for the 2020s,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 11 966–11 976. [Online]. Available: https://doi.org/10.1109/CVPR52688.2022.01167

[71] A. Kumar, A. Raghunathan, R. Jones, T. Ma, and P. Liang, “Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/forum?id=UYneFzXSJWh

[72] P. Kirichenko, P. Izmailov, and A. G. Wilson, “Last Layer Re-Training is Suficient for Robustness to Spurious Correlations,” in International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=Zb6c8A-Fghk

[73] Y. Geifman and R. El-Yaniv, “Selective Classification for Deep Neural Networks,” in Advances in Neural Information Processing Systems, vol. 30. Curran Associates, Inc., 2017, pp. 4878–4887. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2017/hash 4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html

[74] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng, “Reading digits in natural images with unsupervised feature learning,” in NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011, 2011. [Online]. Available: http://ufldl.stanford.edu/housenumbers/nips2011\_ housenumbers.pdf

[75] V. Saravanan, G. J. Berman, and S. J. Sober, “Application of the Hierarchical Bootstrap to Multi-Level Data in Neuroscience,” Neurons, Behavior, Data Analysis, and Theory, vol. 3, no. 5, 2020. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290/

[76] S. Holm, “A Simple Sequentially Rejective Multiple Test Procedure,” Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65–70, 1979. [Online]. Available: https://www.jstor.org/stable/4615733

[77] Y. Zhou, M. A. Chia, S. K. Wagner, M. S. Ayhan, D. J. Williamson, R. R. Struyven, T. Liu, M. Xu, M. G. Lozano, P. Woodward-Court, Y. Kihara, A. Altmann, A. Y. Lee, E. J. Topol, A. K. Denniston, D. C. Alexander, and P. A. Keane, “A Foundation Model for Generalizable Disease Detection from Retinal Images,” Nature, vol. 622, no. 7981, pp. 156–163, 2023. [Online]. Available: https://doi.org/10.1038/s41586-023-06555-x