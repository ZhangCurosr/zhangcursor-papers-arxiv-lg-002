# On Task Scope and Information Retention in Source Coding

Alireza Furutanpey<sup>∗</sup>

NeverBlink, Warszawa, Poland

a.furutanpey@neverblink.eu

Kerstin Bunte

Rijksuniversiteit Groningen, Groningen, Netherlands

k.bunte@rug.nl

## Abstract

We argue that dividing codec design into Coding for Machines (CfM) and Coding for Humans (CfH) is a misleading distinction for deciding what information a codec may discard. Receiver identity does not determine admissible information loss. The required rate depends on task scope, including the predictions to support, their losses and tolerated risks, the encoder observation, and the permitted decoding procedures. Notably, a machine task may have a higher minimum rate than a restricted human decision. Rate savings on selected machine tasks apply only to the stated requirements, not to an intrinsic ordering by receiver type. We extend source and feature coding to finite task families, derive when restricting the encoder observation preserves the minimum rate, and show that equality between source and split-feature coding rates can no longer hold as the task scope expands.

## 1 Introduction

The line of work in Coding for Machines (CfM) studies compression when transmitted data are used primarily for downstream machine analysis (Duan et al., 2020; Choi & Bajić, 2022; Feng et al., 2022; Shao et al., 2022; Gündüz et al., 2023; Harell et al., 2025). A common motivation is the growing volume of data that must be transmitted over bandwidth-limited links in mobile applications and remote sensing. The corresponding system model assumes a resource-constrained client that continuously acquires high-dimensional data and transmits a compact representation to a better-provisioned remote server shared by multiple clients. The server performs prediction tasks and may return or relay their results. A cloud provider ofering a feature codec to application developers may not know their tasks at codec design time. Such a service must specify the task scope it supports, since savings on designer-selected tasks do not establish suitability for its clients. Compression must ofset codec overhead and, in time-critical applications, reduce end-to-end request latency. At the information-theoretic level, CfM is a source-coding problem in which fidelity is specified by the information that must be preserved to satisfy downstream prediction-risk requirements.

While CfM may provide a sound formal model for theoretical analysis, we argue that the implied division of codec design into Coding for Machines and Coding for Humans is misleading, particularly as a basis for deciding which source information a codec should preserve to reduce rate.

Receiver identity alone provides no indication of which source distinctions may be discarded. Rather, the required rate depends primarily on the task scope, i.e., what must be predicted, to what tolerance, and subject to what observation and decoding constraints. Therefore, two machine tasks may require diferent source information, while a restricted human decision may require less information than either faithful reconstruction or a machine task with broader predictive requirements.

Task-specific rate savings establish an advantage only for the stated requirements and do not imply a general ordering of minimum rates by receiver type. A machine task may instead have the higher minimum rate. Consider that prediction models can exploit source distinctions that human observers cannot reliably perceive (Ilyas et al., 2019; Geirhos et al., 2019; Furutanpey et al., 2025). If those distinctions reduce prediction

Original

risk, preserving that lower risk forces the coded message to retain information that a human-limited task may discard, i.e., a model with better prediction performance through such information is likely to require more bits to preserve that performance. Moreover, reconstruction serves diferent purposes for genera human viewing and for expert inspection of a prediction. The former may require broad perceptual fidelity, whereas the latter requires only that the evidence relevant to the decision remain interpretable. If the coded message preserves that evidence, the same message can support a human-interpretable reconstruction without introducing a separate reconstruction channel. Source details irrelevant to the decision need not be preserved solely because a human views the reconstruction. A simple preliminary experiment with two machine tasks illustrates this dependence on task scope (Figure 1).

![](images/5e0c6b2230d7480c7b935767132c6a9e4ec063a4660df3fb29585c4e1bacd74b.jpg)

![](images/dd8dfb8b2e80858f9753635141abde83feeb6e6da61a449a082a0b6891d70a87.jpg)  
Figure 1: ImageNet ResNet-50 inversions for Indigo Bunting (top) and Blue Grosbeak (bottom). Boxes mark the wing regions enlarged below. The chestnut patch becomes less visible at later blocks. Species-pair accuracy uses 200-way CUB readers restricted to these species, with 30 test photographs each. Filled points show two fits per observation and open points their means (Appendix B.7).

The frozen remaining layers of the ImageNet-pretrained ResNet-50 reproduce the original ImageNet classification decisions at every shown block. The color cue distinguishing Indigo Bunting from Blue Grosbeak becomes less visible in the later reconstruction. At block 16, species-pair accuracy is 93.3% from native features and 82.5% from reconstructed RGB, averaged over the fitted readers (Wah et al., 2011). The original ImageNet decisions remain unchanged, while species-pair accuracy difers between prediction from native features and prediction from reconstructed RGB. A reconstruction constraint does not remove this dependence on task scope because it preserves only the distinctions penalized by its distortion criterion and tolerance. Guaranteeing human verification outside the stated task scope requires preserving distinctions relevant to those additional inspections. The listed task-risk constraints do not, in general, guarantee their preservation when those inspections are unspecified. This is precisely the setting that motivates CfM. Yet, the CfM/CfH distinction risks imposing a misleading design prior, directing attention toward receiver identity rather than the task scope that ultimately determines what information may be discarded. Why, then, should codec design be divided into Coding for Machines and Coding for Humans when admissible information loss is determined by the supported task scope rather than receiver identity?

Through theoretical analysis and empirical studies, we show how extrapolating coding equivalence beyond the original task can mislead codec design. We extend a recent rate–distortion formulation of coding for machines (Harell et al., 2025) to finite task families with uncertain reference outputs. We characterize when a reduced encoder observation preserves the coding optimum and bound the penalty when it does not. Coding from a feature can match source coding for the original task yet require more bits when another task is added, despite unchanged optimal uncompressed predictions. Learned finite codes demonstrate this penalty in completely transmitted messages. Image-compression studies further show how preservation losses and decoding procedures afect the predictions obtainable from a message. All code and reported results are openly available in https://github.com/rezafuru/Task-Scope-and-Information-Retention.

## 2 Task scope and compression requirements

Task scope specifies the required predictions, their losses and tolerated risks, the encoder observation, and the permitted joint predictions. Reconstruction requirements and decoder choices also restrict message reuse.

## 2.1 Task scope and rate–distortion theory

Let X be a source variable and $\widehat { X }$ a decoded output. We evaluate their discrepancy with a distortion function $d ( x , { \hat { x } } )$ and restrict $\widehat { X }$ to a reconstruction alphabet $\widehat { \chi }$ , which may contain images, features, or predictions. For independent and identically distributed samples on finite alphabets, additive distortion, and unrestricted block coding, the minimum asymptotic rate is

$$
R ( D ) = \operatorname* { m i n } _ { p ( { \hat { x } } \mid x ) } \left\{ I ( X ; { \widehat { X } } ) \Big \vert \operatorname { \mathbb { E } } [ d ( X , { \widehat { X } } ) ] \leq D \right\} ,\tag{1}
$$

where D is the tolerated expected distortion, and mutual information is measured in bits. Suficiently long block codes can approach this rate. Two compression problems with the same source and receiver can have diferent minimum rates when they impose diferent reconstruction alphabets or distortion requirements. Accordingly, a rate comparison must specify what the decoded message must preserve.

To represent several predictions from the same message, we model X jointly with finite-alphabet reference outputs $T _ { 1 } , \ldots , T _ { m }$ . The index $q \in Q = \{ 1 , \dots , m \}$ identifies a task, and $T _ { q }$ denotes its reference output. For a task asking whether an image contains a vehicle, $T _ { q }$ is the binary reference label. A reference output may remain uncertain even when $X$ is known, as with noisy labels. Lowercase x and $t _ { q }$ denote their realizations.

The encoder may have access either to the full source X or to a restricted observation $Y = g ( X )$ , where g is a fixed observation or feature map applied before coding and is not adapted to the particular task $q .$ We denote the encoder’s available observation by O. For a block of n independent samples, the encoder sends one message $M _ { n } = e _ { n } ( O ^ { n } )$ at rate R bits per sample. Decoder q computes $\widehat { T } _ { q } ^ { n } = \psi _ { q , n } ( M _ { n } )$ from that same message. We assume no source-correlated side information at the decoders. Restricting O through g can merge source states before coding. If g merges states with diferent conditional target distributions to the same observation, the smallest attainable prediction risk can increase. A fixed viewing interface for a human operator can impose the same restriction when it renders two measurements indistinguishable despite their diferent evidence for the task. A predictor observing X can then predict with lower risk, and preserving that risk can require a higher rate.

For a single sample, let $\widehat { T } = ( \widehat { T } _ { 1 } , \ldots , \widehat { T } _ { m } )$ denote the joint decoded prediction and let $\hat { t }$ exhibit a particular prediction tuple. We fix a finite, nonempty set $\widehat { \tau }$ of permitted tuples. Separate task decoders may permit every combination of individual predictions, whereas prediction functions applied to one common reconstruction may allow only a subset of combinations. All channels, sums, and optimizations over t<sup>ˆ</sup> are taken over $\widehat { \tau }$ which need not equal the set of possible reference-output tuples.

We evaluate $\hat { t } _ { q }$ against $t _ { q }$ using a task distortion $d _ { T _ { q } } ( t _ { q } , \hat { t } _ { q } ) \in [ 0 , 1 ]$ . Its expectation is the prediction risk. For each task, we require $\mathbb { E } [ d _ { T _ { q } } ( T _ { q } , \widehat { T } _ { q } ) ] \leq D _ { q } .$ , and write $\mathbf { D } = ( D _ { 1 } , \ldots , D _ { m } )$ for the vector of task tolerances. Because the target $T _ { q }$ may remain uncertain even when the observation is fixed at $O = o$ , we average the distortion over its conditional distribution and define

$$
c _ { q , O } ( o , \hat { t } ) = \mathbb { E } \big [ d _ { T _ { q } } ( T _ { q } , \hat { t } _ { q } ) \ | \ O = o \big ] .\tag{2}
$$

A prediction channel based on O satisfies $\mathbb { E } [ d _ { T _ { q } } ( T _ { q } , \widehat { T } _ { q } ) ] = \mathbb { E } [ c _ { q , O } ( O , \widehat { T } ) ]$ . We apply indirect rate-distortion theory with one expected-loss constraint for each required task (Witsenhausen, 1980; Ivry, 2026). The minimum asymptotic rate is

$$
R _ { O } ( \mathbf { D } ; Q ) = \operatorname* { m i n } _ { p ( \hat { t } | o ) } \left\{ I ( O ; \widehat { T } ) \left| \mathbb { E } [ d _ { T _ { q } } ( T _ { q } , \widehat { T } _ { q } ) ] \leq D _ { q } \mathrm { ~ f o r ~ e v e r y ~ } q \in Q \right. \right\} .\tag{3}
$$

We set the rate to infinity when no permitted prediction channel satisfies all requirements and omit Q from the notation when the family is fixed.

The Bayes risk $D _ { q , O } ^ { \star }$ is the smallest expected task loss achievable from observation O. If $D _ { q } < D _ { q , O } ^ { \star }$ , the risk requirement cannot be achieved from O at any rate. We measure excess risk relative to the Bayes risk for a specified reference observation, X or Y. A feature Y is suficient for target $T _ { q }$ when $T _ { q }$ and X are conditionally independent given Y. The same conditional target distribution is then available from either observation, so either permits an optimal prediction for any loss (Blackwell, 1953; van Rooyen & Williamson, 2014).

For a fixed loss, preserving an optimal prediction can require fewer distinctions than preserving the posterior. Assuming binary zero-one loss, posterior probabilities 0.6 and 0.9 for class 1 both give the Bayes prediction 1. Predicting 0 instead increases conditional error by 0.2 and 0.8, respectively. An encoder that distinguishes these states can place more coding errors on the state with posterior 0.6, where each error has lower cost. An observation that merges the states removes that choice. Multiple tasks can further restrict the error allocation. If another task assigns high error cost to states that are inexpensive for the first task, satisfying both requirements can require a higher rate even when both tasks share the same Bayes prediction.

## 2.2 Related Work

Receiver labels and mixed fidelity requirements. Researchers in coding for machines, supervised compression, and task-oriented semantic communication specify fidelity through selected inference objectives (Duan et al., 2020; Matsubara et al., 2022; 2023; Shao et al., 2022; Gündüz et al., 2023). Zhang et al. (2026) motivate their framework by describing machine vision as requiring less information than human vision while evaluating selected detection, segmentation, and reconstruction objectives. Task-specific compression can also preserve human decisions at reduced rates (Reddy et al., 2021). JPEG AI specifies a shared representation for visualization, image processing, and computer vision (ISO/IEC JTC 1/SC 29/WG 1, 2026). Stavrou & Kountouris (2023) analyze joint semantic and observation reconstruction subject to separate fidelity constraints.

Task families and preserved decisions. Dubois et al. (2021) formulate preservation through excess prediction risk over task families defined by shared invariances. Ivry (2026) derives a finite-family rate expression with one tuple of query-specific predictions and separate excess-risk constraints. For a fixed loss, Sevetlidis (2026) characterizes representations that preserve Bayes risk by recovering an optimal prediction. When a statistic preserves the target posterior, Armstrong (2026) proves information-bottleneck and log-loss equivalence through conditional averaging. Wang & Dai (2026) obtain equal source and posterior coding rates when distortion depends on the source through its target posterior. Optimal predictions can agree across tasks even when coding errors have diferent conditional costs.

Coding interfaces and conditional error costs. Choi & Bajić (2022) compare source and feature reconstructions given a prescribed task distortion and discuss the possible inadequacy of a trained representation for later tasks. Harell et al. (2025) add direct-feature coding and evaluate distortion against a fixed model output $T = f ( X ) = h ( Y )$ . They give output conditions for rate equalities. With multiple targets, each may remain uncertain given X and need not be determined by Y. Martinian et al. (2008) study source coding with observed distortion costs, and Enttsel & Corlay (2026) compare indirect coding with compressing a model’s prediction. Finite Shannon optimization and conditions on the predictions used by an optimum are established in rational inattention (Caplin et al., 2019; Armenter et al., 2024).

Reusable learned representations. Researchers learn features for reuse across vision tasks and exploit dependencies between task representations during coding (Feng et al., 2022; Guo et al., 2025; Huang et al., 2026). Other methods use common and task-specific messages (de Andrade et al., 2026), auxiliary reconstruction for secondary tasks (de Andrade & Bajić, 2024), task-derived importance weights (Esfahanizadeh et al., 2026), or transformed features from diferent tasks and architectures (Gao et al., 2025). With Franken-Split and FOOL, researchers compress shallow features and evaluate their reuse across downstream models (Furutanpey et al., 2024; 2025). Reusing a codec across tasks, obtaining an additional prediction from an existing message, and fitting a new decoder after coding impose diferent requirements. Rate comparisons must state which of these requirements are included.

## 3 Comparing source and feature coding

An encoder observing X can use source distinctions that are absent from $Y = g ( X )$ . We compare three coding arrangements with identical task-risk tolerances and permitted prediction tuples.

In source coding, we encode X and predict from a source reconstruction ${ \widehat { X } } \in { \widehat { \mathcal { X } } } .$ . In split-feature coding, we encode Y and predict from a feature reconstruction $\widehat { Y }$ . In direct-feature coding, we encode X and predict from a feature reconstruction $\widetilde { Y }$ . We denote their minimum rates by $R _ { X } , R _ { Y }$ , and $R _ { X Y }$ , respectively. Both feature reconstructions take values in $\widehat { \mathcal { V } }$

## 3.1 Jointly attainable predictions

Let $F ( \hat { x } ) = ( f _ { 1 } ( \hat { x } ) , \ldots , f _ { m } ( \hat { x } ) )$ and $H ( \hat { y } ) = ( h _ { 1 } ( \hat { y } ) , \dots , h _ { m } ( \hat { y } ) )$ be fixed prediction functions of the source and feature reconstructions, respectively. Suppose their finite reconstruction alphabets satisfy

$$
F ( { \widehat { \mathcal { X } } } ) = H ( { \widehat { \mathcal { Y } } } ) .\tag{4}
$$

We use this common range as the permitted prediction tuples in equation 3. Every permitted tuple is obtainable from both reconstruction alphabets. Matching the output values available for each task separately does not ensure that every combination of those outputs is attainable from a single reconstruction.

At task tolerances D, we obtain each rate by minimizing the corresponding mutual information,

$$
\begin{array} { r l } { I ( X ; \widehat { X } ) , \ } & { { } X \longrightarrow \widehat { X } \stackrel { F } { \longrightarrow } \widehat { T } , \ \ R _ { X } ( \mathbf { D } ; Q ) , } \\ { I ( Y ; \widehat { Y } ) , \ } & { { } Y \longrightarrow \widehat { Y } \stackrel { H } { \longrightarrow } \widehat { T } , \quad R _ { Y } ( \mathbf { D } ; Q ) , } \\ { I ( X ; \widetilde { Y } ) , \ } & { { } X \longrightarrow \widetilde { Y } \stackrel { H } { \longrightarrow } \widehat { T } , \ \ R _ { X Y } ( \mathbf { D } ; Q ) , } \end{array}\tag{5}
$$

over reconstruction channels satisfying $\mathbb { E } [ d _ { T _ { q } } ( T _ { q } , \widehat { T } _ { q } ) ] \leq D _ { q }$ for every task.

Replacing a reconstruction with its prediction tuple preserves all task risks and cannot increase mutual infor mation. Conversely, by equation 4, we can choose one representative reconstruction for each permitted tuple. From any prediction channel, we can reconstruct that representative with the same mutual information and risks. We can therefore compute the three rates in equation 5 using the prediction optimization in equation 3, with $O = X$ for source and direct-feature coding and $O = Y$ for split-feature coding (Appendix A.1).

## 3.2 Encoder observation and task risk

For an input x and prediction tuple $\hat { t } ,$ write $c _ { q } ( x , \hat { t } ) = c _ { q , X } ( x , \hat { t } )$ for the conditional task distortion defined in equation 2. Averaging over inputs with the same feature gives

$$
\bar { c } _ { q } ( y , \hat { t } ) = c _ { q , Y } ( y , \hat { t } ) = \mathbb { E } [ c _ { q } ( X , \hat { t } ) \mid Y = y ] .\tag{6}
$$

An encoder observing X can compute Y, so $R _ { X Y } ( \mathbf { D } ) = R _ { X } ( \mathbf { D } ) \leq R _ { Y } ( \mathbf { D } )$ . For Harell’s target $T = f ( X ) =$ $h ( Y )$ , every conditional cost depends on X only through Y. Averaging a source prediction channel over inputs with the same feature preserves task distortion and cannot increase rate. Thus $R _ { X Y } = R _ { Y }$ , and $R _ { X }$ is also equal when equation 4 holds (Harell et al., 2025). For additional targets, the conditional costs of diferent predictions can vary among inputs with the same feature. Whether those diferences afect the minimum rate depends on the tolerated risks.

## 3.3 Exact equality and rate penalty

For finite nonnegative multipliers λ, write $\begin{array} { r } { c _ { \lambda , O } = \sum _ { q } \lambda _ { q } c _ { q , O } } \end{array}$ and define

$$
J _ { O } ( \mathbf { \lambda } \mathbf { \lambda } ) = \operatorname* { m i n } _ { p ( \hat { t } | o ) } \left\{ I ( O ; \widehat { T } ) + \mathbb { E } c _ { \mathbf { \lambda } , O } ( O , \widehat { T } ) \right\} .\tag{7}
$$

Each $\lambda _ { q }$ is the marginal cost of increasing task $q \mathrm { ^ s }$ risk. For a distribution r over permitted prediction tuples, the standard rate-distortion variational formula gives the conditional distribution

$$
\pi _ { O } ^ { r } ( \hat { t } \mid o ) = \frac { r ( \hat { t } ) 2 ^ { - c _ { \mathsf { \pm } , O } ( o , \hat { t } ) } } { \sum _ { \hat { u } } r ( \hat { u } ) 2 ^ { - c _ { \mathsf { \pm } , O } ( o , \hat { u } ) } } .\tag{8}
$$

At an optimum, $r$ is the marginal distribution of the decoded prediction. The variational reduction and its support conditions are given in (Caplin et al., 2019; Armenter et al., 2024). Comparing the two observations at the same r gives

$$
G _ { \lambda } ( \boldsymbol { r } ) = \mathbb { E } _ { X } D _ { \mathrm { K L , 2 } } ( \pi _ { Y } ^ { r } ( \cdot \mid Y ) \parallel \pi _ { X } ^ { r } ( \cdot \mid X ) ) ,\tag{9}
$$

where $D _ { \mathrm { K L , 2 } }$ uses base-two logarithms. This quantity measures the divergence between the prediction channels caused by averaging the conditional costs within each feature value.

Proposition 1 (Prescribed observation). For the finite model, assume equation 4. Then $J _ { X } ( \lambda ) = J _ { Y } ( \lambda )$ if and only if some globally optimal source marginal r satisfies

$$
c _ { \lambda , X } ( x , \hat { t } ) - c _ { \lambda , X } ( x , \hat { u } ) = c _ { \lambda , X } ( x ^ { \prime } , \hat { t } ) - c _ { \lambda , X } ( x ^ { \prime } , \hat { u } )\tag{10}
$$

whenever $g ( x ) = g ( x ^ { \prime } ) , p _ { X } ( x ) p _ { X } ( x ^ { \prime } ) > 0$ , and $r ( \hat { t } ) r ( \hat { u } ) > 0$ . For any globally optimal marginals $r _ { X } , r _ { Y }$ for the respective objectives,

$$
G _ { \lambda } ( r _ { Y } ) \leq J _ { Y } ( \lambda ) - J _ { X } ( \lambda ) \leq G _ { \lambda } ( r _ { X } ) .\tag{11}
$$

The equality condition involves only cost diferences between predictions with positive probability in an optimal channel. Adding the same source-dependent cost to every used prediction leaves their relative costs unchanged. Posteriors can therefore vary among inputs with the same feature value while the minimum objective values still remain equal. Identical Bayes predictions alone do not ensure equality, because they do not specify the relative costs of coding-errors. The predictions with positive probability in an optima channel can change with the tolerance. Requiring a global optimum excludes arbitrary singleton supports, which satisfy equation 10 vacuously (Appendix A.2).

The rate penalty at the same task-risk tolerances follows by evaluating the two objectives at their respective dual-optimal multipliers. If D is feasible from $Y$ and finite multipliers $\lambda _ { X } , \lambda _ { Y }$ attain the respective constrained dual optima, then

$$
\Delta J ( \lambda _ { X } ) \le R _ { Y } ( { \bf D } ) - R _ { X } ( { \bf D } ) \le \Delta J ( \lambda _ { Y } ) , \qquad \Delta J = J _ { Y } - J _ { X } .\tag{12}
$$

Strict feasibility of the risk constraints sufices for attainment by finite multipliers. Equal multipliers need not include equal task risks. Combining equation 11 and equation 12 gives quantitative bounds on the rate penalty at the required componentwise tolerances. For Harell’s fixed target, the conditional costs satisfy equation 10 on the entire prediction range. The conditional costs for an expanded task family can violate this criterion even when the original task and its risk tolerance remain unchanged.

The encoder’s observation and the training objective must also be specified separately, since the latter does not by itself determine the information retained by the encoder. When a source-observing encoder is trained using distortion against $T = f ( X )$ , it can still retain distinctions between inputs with the same T. We therefore cannot infer the Markov chain $X ^ { n } \longrightarrow T ^ { n } \longrightarrow M _ { n }$ from that training objective alone. When the distortion depends on X only through T, we can construct a target-dependent channel at the unrestricted optimum by averaging the encoder channel over X given T. A fitted encoder need not implement this target-dependent channel, even if it achieves the same training risk.

## 4 Finite families at nonzero risk

## 4.1 Noisy threshold tasks

Let S be uniform on $\{ 0 , \ldots , 5 \}$ , let $X = ( S , N )$ include an independent finite-valued sensor variation $N _ { \ast }$ , and let $Y = \lfloor S / 2 \rfloor$ . We require predictions of two noisy threshold labels. The underlying decisions are

$$
C = { \bf 1 } \{ S \geq 2 \} , \qquad B = { \bf 1 } \{ S \geq 3 \} .\tag{13}
$$

We denote task indices as C and B, with tolerances ordered as $\mathbf { D } = \left( D _ { C } , D _ { B } \right)$ . The targets are $T _ { C } = C \oplus E _ { C }$ and $T _ { B } = B \oplus E _ { B }$ , where $\oplus$ adds bits modulo two, flipping the threshold bit when $E _ { q } = 1$ . The noise variables $E _ { C } , E _ { B }$ are independent Bernoulliϵ variables with parameter $\epsilon \in ( 0 , 1 / 2 )$ , independent of S. The sensor variation N is independent of $( S , E _ { C } , E _ { B } )$ . Both tasks use zero–one loss, $d _ { T _ { q } } ( t _ { q } , \hat { t } _ { q } ) = \mathbf { 1 } \{ t _ { q } \neq \hat { t } _ { q } \}$ for $q \in \{ C , B \}$

The source threshold functions and the feature outputs $h _ { C } ( y ) = \mathbf { 1 } \{ y \geq 1 \}$ and $h _ { B } ( y ) = \mathbf { 1 } \{ y \geq 2 \}$ permit exactly the same joint predictions $( \widehat { T } _ { C } , \widehat { T } _ { B } ) \in \{ ( 0 , 0 ) , ( 1 , 0 ) , ( 1 , 1 ) \}$ . The prediction (0, 1) is excluded because the higher threshold cannot be positive when the lower threshold is negative. Independent label noise still permits the reference-output tuple $( T _ { C } , T _ { B } ) = ( 0 , 1 )$

We can recover C from Y. Recovering B without error requires distinguishing $S = 2$ from $S = 3$ , which share $Y = 1$ . These equally likely values occur with total probability $1 / 3 ,$ giving minimum underlying-bit error $1 / 6$ from Y. If a decoded prediction has underlying-bit error $\textstyle e _ { q } .$ , its target risk is $\epsilon + ( 1 - 2 \epsilon ) e _ { q }$ . The split-observation Bayes risk for the additional task is therefore $D _ { B , Y } ^ { * } \bar { = } \epsilon + ( 1 - 2 \epsilon ) / 6$ . Both full-observation Bayes risks are ϵ.

At the original task’s Bayes risk, every code must recover C, requiring $H _ { C } = h _ { 2 } ( 2 / 3 )$ bits per sample, where $h _ { 2 }$ is binary entropy. At or above the reduced-observation risk floor, transmitting C also satisfies the additional task requirement. Below that floor, source-observing encoders additionally encode B within the event $C = 1$ , which has probability $2 / 3$ . The independent sensor variation N can be discarded. No code based on the reduced observation has additional-task risk below this floor. Proposition 2 in Appendix A.5 gives the exact rates and conditional binary-coding construction.

Keeping encoder access fixed at X, we can compare requirements specified by the reference observations X and $V = Y$ . We keep the original-task tolerance at ϵ and add the same nonnegative excess-risk allowance to each reference’s additional-task Bayes risk. With the coarser reference, the minimum rate is $H _ { C }$ bits per sample. With X as reference, the minimum rate is higher while this allowance is less than $( 1 - 2 \epsilon ) / 6$ (Figure 2a). This increment is smaller than $H ( X \mid V ) = 1 + H ( N )$ because the independent sensor variation remains irrelevant to both tasks. Appendix A.5 gives the rate diference.

Replacing B by $B ^ { \prime } = { \bf 1 } \{ S \geq 4 \}$ retains two tasks and makes both underlying decisions deterministic functions of Y. Encoding their joint Bayes predictions requires $\log _ { 2 } 3 = 1 . 5 8 4 9 6$ bits per sample, compared with 1.45915 bits per sample for $( C , B )$ . Using $Y ,$ , we can attain the full-observation Bayes risks of this higher-rate family. We cannot attain those of the lower-rate family from Y. Task count and the minimum rate from X therefore do not determine whether a particular reduced observation preserves the required predictions.

Full observation X Reduced observation Y

![](images/9a0d60e4c4bc78379e7c83ab6f51bcb8ece239c269414f45645027cb4970bbe3.jpg)

![](images/e39b7e5675a333217cb739921d818ae016fe18aae5815a8abbdd5609167e171a.jpg)

![](images/c387a29947c6a940a879044a370538c05e9616869abce8829d3dc641ca9a3406.jpg)  
Figure 2: Exact asymptotic rates for two noisy task families. (a) Label noise and original-task error are 0.05. Minimum additional-task error from Y is 0.20. At or above this floor, the rates coincide. (b) At noise probabilities (0.05, 0.45), both observations have Bayes prediction Z and minimum error 0.25, with diferent rates at intermediate tolerances. (c) Reversing these probabilities for a complementary task makes the rates equal at $D _ { 1 } = D _ { 2 } = D$ , with Bayes tuple $( Z , Z )$ . Circles mark $D = 0 . 3 0$ . Single-task curves specialize Martinian et al. (2008)’s formula. The pair rate is equation 15.

## 4.2 Retaining the original task while adding a requirement

A feature can preserve the original task’s entire rate-distortion function while increasing the minimum rate required by an expanded task family. Let $Z \in \{ 0 , 1 , 2 \}$ have probabilities $( 0 . 3 5 , 0 . 3 5 , 0 . 3 0 )$ , let $K \in \{ - 1 , 1 \}$ be an independent fair sign, and set $X = ( Z , K )$ and $Y = Z$ . The original task is the noiseless target $T _ { 0 } = Z$ An additional noisy ternary target $T _ { 1 }$ has the unique Bayes-optimal prediction $Z$ from either observation and Bayes risk 0.45. Both tasks use zero–one loss.

For $Z = 0$ or 1, predicting the other of these two classes has conditional excess risk 0.325, irrespective of K. Predicting class 2 instead increases risk by $0 . 3 2 5 + 0 . 2 7 6 2 5 K$ . The encoder observing K can therefore allocate some prediction errors to states where its cost is 0.04875, rather than 0.60125. At $Z = 2$ , either wrong prediction has conditional excess cost 0.325. Appendix A.3 gives the complete target law. The minimum Bayes margin is 0.04875, so the Bayes prediction is unique.

For the original task alone, all three coding arrangements achieve the same minimum rate-distortion function. Keep its tolerance at $D _ { 0 } = 0 . 3 5$ and add $D _ { 1 } = 0 . 4 9 9$ . The resulting family rates are

$$
R _ { X } ( 0 . 3 5 , 0 . 4 9 9 ) = 0 . 6 3 5 4 0 , \qquad R _ { Y } ( 0 . 3 5 , 0 . 4 9 9 ) = 0 . 8 1 8 7 6\tag{14}
$$

bits per sample. The optimal source and feature channels have original-task risks 0.29456 and 0.15077, respectively. Both satisfy the unchanged original requirement. The gap persists when all nine prediction tuples are permitted. The additional task alone yields a lower bound attained by setting $\widehat { T } _ { 0 } = \widehat { T } _ { 1 }$ in these feasible channels.

The gap is 0.18336 bits per sample, lying within the interval [0.16642, 0.19660] from equation 12. With exact original-task recovery, $D _ { 0 } = 0$ , both encoders instead require $H ( Z )$ bits per sample and can attain the additional task Bayes risk 0.45 by transmitting Z. Thus the penalty depends on the allowed risks as well as the added task. A feature selected using the original fixed-target equality can restrict the encoder’s ability to allocate errors to states with lower conditional excess costs, even without changing either task’s best uncompressed prediction.

## 4.3 Task-dependent error costs

Complementary tasks can eliminate an encoder’s rate advantage from observing confidence. Let Z and K be independent fair bits, set $X = ( Z , K , N )$ and $Y = Z$ , and let N be finite-valued sensor variation independent of all task variables. The two zero–one tasks have targets $T _ { q } = Z \oplus E _ { q }$ . Conditional noise probabilities in states $K = 0 , 1$ are $( \epsilon _ { 0 } , \epsilon _ { 1 } )$ for task 1 and reversed for task 2, independently of Z, with $0 \le \epsilon _ { 0 } < \epsilon _ { 1 } < 1 / 2$ Both Bayes predictions are $Z ,$ , with risk $p = ( \epsilon _ { 0 } + \epsilon _ { 1 } ) / 2$ . The readouts $f _ { q } ( z , k , n ) = z$ and $h _ { q } ( y ) = y$ therefore yield the same decoded bit U for both tasks.

Write $w _ { k } = 1 - 2 \epsilon _ { k }$ and $e _ { k } \ = \ \operatorname* { P r } ( U \ \neq \ Z \ | \ K = \ k )$ . The tasks’ excess risks are $( w _ { 0 } e _ { 0 } + w _ { 1 } e _ { 1 } ) / 2$ and $( w _ { 1 } e _ { 0 } + w _ { 0 } e _ { 1 } ) / 2$ . For task 1 alone, observing K permits a larger error probability in the less costly state $K = 1$ , as in distortion-side-information coding (Martinian et al., 2008). An encoder observing only Y cannot condition its error probability on $K$ , so $e _ { 0 } = e _ { 1 }$

At a common tolerance D, averaging the two risk constraints gives $( e _ { 0 } + e _ { 1 } ) / 2 \le ( D - p ) / ( 1 - 2 p )$ . Averaging the coding channel over N and symmetrizing it with its bit-complemented counterpart, preserves the risks and cannot increase the rate, giving rate $1 - [ h _ { 2 } ( e _ { 0 } ) + h _ { 2 } ( e _ { 1 } ) ] / 2$ . Concavity of binary entropy then implies that the constant error allocation is optimal. Encoders observing X or Y can use this constant error allocation, so for $p \le D \le 1 / 2$

$$
R _ { X } ^ { ( 1 , 2 ) } ( D , D ) = R _ { Y } ^ { ( 1 , 2 ) } ( D , D ) = 1 - h _ { 2 } \bigg ( \frac { D - p } { 1 - 2 p } \bigg ) .\tag{15}
$$

The sum of the two conditional excess costs is independent of $K$ , satisfying equation 10. Proposition 3 in Appendix A.6 extends this criterion to arbitrary finite confidence states and task families, including unrestricted joint predictions and unequal tolerances.

For $( \epsilon _ { 0 } , \epsilon _ { 1 } ) = ( 0 . 0 5 , 0 . 4 5 )$ and $D = 0 . 3 0$ , task 1 alone requires 0.33680 bits per sample from X and 0.53100 bits per sample from Y. The source-optimal channel for task 1 alone gives task 2 risk 0.44486. Requiring both risks to be at most 0.30 raises the minimum source rate to 0.53100 (Figure 2b,c). The Bayes tuple remains $( Z , Z )$ , yet the rate advantage of observing confidence disappears once both tasks are imposed. Counting optimal predictions or comparing uncompressed accuracy therefore does not determine the rate required when prediction errors are allowed.

![](images/aea8e2fe718ab27191b2c4c46ad8891e83a0fa6a7fafcdc91db2c0b07e284a41.jpg)  
Figure 3: Finite codes at 0.81251 bits per symbol, below the 0.81876 minimum for satisfying both requirements from Y. (a,b) Codes whose encoders observe class and confidence satisfy both requirements. (c) Their errors concentrate in negative-confidence states, where they incur less additional-task loss.

## 5 Conditional error costs in finite coding

We fit three codebooks for the three-class family, each with 8,192 ternary words of length 16. Both decoders return the codeword indexed by the transmitted message. Full and reduced selectors use either the known conditional costs or estimates of these costs fitted from independent noisy labels. We evaluate the codes on fresh source blocks using the known target law, with tolerances $D _ { 0 } = 0 . 3 5$ and $D _ { 1 } = 0 . 4 9 9$ . We count framing and padding bits when computing complete rates (Appendix B.2).

For all three full-observation codes, the simultaneous one-sided 99% Hoefding upper risk bounds are below their respective task tolerances (Figure 3a,b). Their complete rate of 0.81251 bits per source symbol is below the 0.81876-bit analytical minimum rate for reduced-observation coding under both requirements, irrespective of how the selector is fitted (equation 14). The strict rate advantage of observing the full source in Section 4.2 therefore persists for finite codebooks, including framing and padding.

The full encoder allocates more original-task errors to the negative-confidence states, where predicting class 2 incurs less additional-task loss (Figure 3c). Selecting codewords using fitted conditional costs likewise allocates more original-task errors in the negative-confidence states and satisfies both risk requirements. On the same codebooks, reduced-observation selectors cannot distinguish the confidence states, and their additional-task risks exceed the tolerance. Preserving the Bayes prediction therefore does not preserve the available allocation of coding errors across confidence states.

Two controls preserve the class distribution and Bayes predictions. With only the original deterministic task, full and reduced selection give identical packets. Full and reduced selection likewise produce identical packets when the additional task’s target law is averaged over confidence. Observing confidence changes these codes only when the conditional error costs relevant to the coding objective vary with confidence. In the binary family, tighter complementary requirements eliminate the benefit of concentrating errors in either state. The single-task saving disappears when corruption makes the observed confidence independent of the underlying state, i.e. at $\tau = 1 / 2$ (Appendix B.3). These efects follow Proposition 3.

![](images/0270fbd1953b46b090eeee230bce5a44bce06145f8f7b12ebeeae2fa61cf0b6d.jpg)  
Figure 4: ImageNet accuracy drops from uncompressed references. Rows list fitting losses and complete validation rates. Open and filled markers show validation and selected test means. Small dots show individual fits, and bars show conditional 95% paired-image intervals. Dashed lines mark the allowances.

## 6 Requirements for reusable image codes

Image-code rates are measured in bits per pixel (bpp) using complete messages, including the main latent, hyperlatent, and header. All reported predictions are computed from entropy-decoded messages. Shared model parameters are stored separately (Appendix B). Classification requirements allow at most three percentage points of coarse-accuracy loss and five points of fine-accuracy loss relative to uncompressed references.

## 6.1 Coarse supervision and fine prediction

ImageNet’s ENTITY-30 grouping has 30 superclasses and 240 fine classes (Santurkar et al., 2021). Codecs fitted with either objective encode the same layer-2 tensor from an ImageNet-pretrained ResNet-50 at 224×224 resolution and reconstruct it for the frozen sufix. Starting from a shared feature-reconstruction initialization, coarse-only fitting minimizes KL between original and decoded coarse probabilities plus rate. Added-fine fitting additionally minimizes normalized fine-label cross entropy, holding the architecture, observation, coarse-loss and rate terms, and fitting exposure fixed (Appendix B.4).

We use validation data to select readout routes and minimum mean complete rate among paired codec fits. We compare minimum rates under the coarse-only and joint mean accuracy-drop requirements within the same candidate pool, and retain a selection satisfying both requirements for each fitting objective.

Coarse-only and joint validation selections coincide. On 6,000 test images, the selected coarse-only codec’s coarse- and fine-accuracy drops exceed their respective allowed losses (Fig. 4). With added-fine coding, the simultaneous 95% upper limits on both accuracy drops are below their allowances, at a complete rate of 0.26717 bpp, compared with 0.19772 bpp for the coarse-only objective. The original objective and encoder observation alone do not determine whether an added task’s risk requirement will be satisfied. Both codecs observe the same early tensor, so this experiment isolates the efect of the fitted preservation objective rather than the observation restriction studied in Section 3.2.

## 6.2 Preservation and adapted decoding

CIFAR-100 has 20 coarse and 100 fine classes (Krizhevsky, 2009). A ResNet-18 (He et al., 2016) teacher is trained on coarse labels. The teacher’s early 128 × 16 × 16 feature tensor serves as the encoder input. A tensor with the same dimensions is reconstructed from each message. The preservation loss is either squared error on the early features or squared error on the 20 coarse logits produced by the frozen teacher sufix. Fine labels are not used in either preservation loss.

Coarse predictions are computed with the frozen sufix. The inherited fine classifier is kept fixed after training on uncompressed early tensors. Adapted classifiers are refitted using decoded tensors or reconstructed logits, respectively. The coarse and fine accuracy-drop allowances are the same as for ImageNet, with fine predictions obtained from refitted tensor readouts. For each of three teachers and each preservation loss, the lowest-rate paired candidate satisfying both mean requirements is fixed before test assessment (Appendix B.5).

![](images/e1bf8b00a2f17b32130773db36824d48bc407662f12f5a9143d55cf10664f6b3.jpg)  
Figure 5: Test results for validation-selected CIFAR codes and nearby lower-rate feature-preserving candidates. (a,b) Accuracy drops from uncompressed references, with dashed allowance lines. (c) Fine readouts applied to each unchanged logit-preserving message. In (a,b), diamonds show means over paired codec fits, hollow circles show individual runs, and bars show conditional 95% paired-image intervals.

Across teachers, the selected logit-preserving codes achieve smaller coarse-accuracy drops at lower complete rates than the selected feature-preserving codes (Fig. 5a). Refitted fine accuracies are similar (Fig. 5b). The 95% intervals for the selected logit-minus-feature diferences fall within the one-percentage-point equivalence margin, conditional on the fitted models (Appendix B.5). For the nearby feature-preserving candidates at lower-rates, rates remain higher than those of the logit-preserving selections and coarse-accuracy drops approach or exceed the allowance. Teacher C’s selected feature-preserving group includes one codec fit that failed, and its interval for coarse-drop crosses the allowance threshold.

Refitting the fine classifier on decoded tensors improves fine accuracy without changing the logit-preserving message (Fig. 5c). The classifier refitted on reconstructed logits is less accurate than either tensor readout. Therefore, the inherited classifier’s errors do not establish that the message lacks information useful for distinguishing fine labels. The encoder input remains the early tensor, and a loss on coarse logits alone does not require the message to retain information only about those logits (Section 3.2). Fine distinctions can therefore remain recoverable without fine-label supervision during codec fitting, provided the decoder is adapted to the coded representation.

Measured rates characterize tested codecs and fitted readouts. Appendix A.4 compares prediction-only transmission and the rate needed to limit residual information.

## 6.3 Task losses and decoder choices

Taskonomy contains aligned indoor RGB images, depth maps, and semantic pseudo-labels (Zamir et al., 2018). The same RGB inputs are encoded with codecs fitted using combinations of depth (D), semantic (S), and RGB-derived edge (E) losses, or with RGB reconstruction loss. The matched task decoders in Fig. 6 are fitted after freezing each codec, with a common architecture, matched training exposure, and the same validation checkpoint rule.

We evaluate log-depth Smooth-L1 loss, weighted semantic cross entropy, and edge mean squared error. Stringent, moderate, and loose requirements permit at most 10%, 25%, and 50%, respectively, of the loss increase from each validation reference to its constant predictor. The thresholds remain fixed when evaluated on 2,000 test images from five unseen buildings. We plot risks using the same normalization in Figure 6 and assess feasibility from point estimates. Appendix B.6 reports the corresponding absolute tolerances.

![](images/24d753707e13f055f610ce0dee10c86bd06e6f130a3507b772f6235647499e4e.jpg)  
Figure 6: Normalized Taskonomy test risks for twelve codecs. Filled symbols show risks with matched task decoders. Arrows connect risks to those obtained from the same RGB-preserving message by reconstructing RGB and applying frozen reference predictors (hollow symbols). The latent edge decoder is unchanged. Dashed lines mark tolerated risks, and bars show conditional 95% image-bootstrap intervals.

Near 0.03 bpp, the depth-semantic and depth–edge codes have similar depth risks (Fig. 6a). Semantic loss is lower for depth-semantic coding, and edge loss is lower for depth-edge coding (Fig. 6b,c). The displayed depth-edge candidate satisfies all three moderate requirements without semantic supervision in codec fitting. For decoders selected using the available-decoder rule, the mean risks over three fits also satisfy these requirements, although only one individual fit satisfies all three (Appendix B.6.2). Hence, the task losses used for codec fitting do not specify the complete set of predictions obtainable from the resulting message.

Reconstructing RGB and applying the frozen reference predictors reduces depth and semantic risks without changing the transmitted message, as indicated by the vertical arrows in Fig. 6a,b. Depth risk falls from above the moderate threshold to below the stringent threshold when the frozen reference predictor is applied to reconstructed RGB. All three stringent requirements are satisfied when the latent edge decoder is retained.

All three fits of the D, 30 depth-only setting satisfy the loose semantic cross-entropy requirement. Their mean object intersection-over-union (mIoU) is below 0.01. In the validation examples, depth-only semantic maps omit labeled objects or assign them incorrect categories (Fig. 9). Therefore, a probability-loss requirement alone does not establish categorical agreement with the reference labels. We set the additional moderate mIoU floor at 0.75 times the validation reference mIoU. The uncompressed reference’s test mIoU falls below this floor. Its mIoU exceeds the floor in a retrospective diagnostic restricted to classes present in the validation set, thereby excluding the newly appearing test class (Appendix B.6.4). Categorical requirements must specify both the metric and the set of classes to be evaluated.

## 7 Conclusion

Receiver identity alone does not determine which source information a codec may discard. We extended source and feature coding to finite task families, characterized when restricting encoder observation leaves the minimum rate unchanged, and bounded the resulting rate penalty. Our theoretical constructions and learned finite codes demonstrate that restricting observation can preserve the minimum rate for one task while increasing it for an expanded task family.

The image studies show that information useful for additional predictions can remain available after fitting with a narrower loss. With adapted decoding, we can satisfy additional risk requirements without changing the transmitted message or its rate. Compression savings must therefore be interpreted relative to the specified predictions, losses, tolerances, encoder observations, and decoding constraints. Preserving each task’s optimal uncompressed prediction need not preserve the distinctions required to allocate coding errors eficiently under the tasks’ risk constraints.

## References

Alessandro Achille and Stefano Soatto. Emergence of invariance and disentanglement in deep representations. Journal ofMachine Learning Research, 19(50):1–34, 2018. URL https://jmlr.org/papers/v19/17-646. html.

Roc Armenter, Michèle Müller-Itten, and Zachary R. Stangebye. Geometric methods for finite rational inattention. Quantitative Economics, 15(1):115–144, 2024. doi: 10.3982/QE2050.

Joss Armstrong. Source-side suficiency for the information bottleneck: Exact reduction and finite-block equivalence. arXiv preprint arXiv:2604.26744, 2026. URL https://arxiv.org/abs/2604.26744.

Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. Variational image compression with a scale hyperprior. In International Conference on Learning Representations, 2018. URL https://arxiv.org/abs/1802.01436.

Jean Bégaint, Fabien Racapé, Simon Feltman, and Akshay Pushparaja. CompressAI: A PyTorch library and evaluation platform for end-to-end compression research. arXiv preprint arXiv:2011.03029, 2020. URL https://arxiv.org/abs/2011.03029.

David Blackwell. Equivalent comparisons of experiments. The Annals of Mathematical Statistics, 24(2): 265–272, 1953. doi: 10.1214/aoms/1177729032.

Andrew Caplin, Mark Dean, and John Leahy. Rational inattention, optimal consideration sets, and stochastic choice. The Review of Economic Studies, 86(3):1061–1094, 2019. doi: 10.1093/restud/rdy037.

Hyomin Choi and Ivan V. Bajić. Scalable image coding for humans and machines. IEEE Transactions on Image Processing, 31:2739–2754, 2022. doi: 10.1109/TIP.2022.3160602.

Anderson de Andrade and Ivan V. Bajić. Towards task-compatible compressible representations. In IEEE International Conference on Multimedia and Expo Workshops, pp. 1–6, 2024. doi: 10.1109/ICMEW63481. 2024.10645459.

Anderson de Andrade, Alon Harell, and Ivan V. Bajić. Lossy common information in a learnable Gray–Wyner network. In International Conference on Learning Representations, 2026.

Lingyu Duan, Jiaying Liu, Wenhan Yang, Tiejun Huang, and Wen Gao. Video coding for machines: A paradigm of collaborative compression and intelligent analytics. IEEE Transactions on Image Processing, 29:8680–8695, 2020. doi: 10.1109/TIP.2020.3016485.

Yann Dubois, Benjamin Bloem-Reddy, Karen Ullrich, and Chris J. Maddison. Lossy compression for lossless prediction. In Advances in Neural Information Processing Systems, volume 34, pp. 14014–14028, 2021.

Andriy Enttsel and Vincent Corlay. Model-aware rate-distortion limits for task-oriented source coding. arXiv preprint arXiv:2602.12866, 2026. URL https://arxiv.org/abs/2602.12866.

Homa Esfahanizadeh, Matin Mortaheb, Jinfeng Du, and Harish Viswanathan. UniTAC: Universal taskaware compression via weighted distortion measures. arXiv preprint arXiv:2608.16696, 2026. URL https: //arxiv.org/abs/2608.16696.

Ruoyu Feng, Xin Jin, Zongyu Guo, Runsen Feng, Yixin Gao, Tianyu He, Zhizheng Zhang, Simeng Sun, and Zhibo Chen. Image coding for machines with omnipotent feature learning. In European Conference on Computer Vision, 2022.

Alireza Furutanpey, Philipp Raith, and Schahram Dustdar. FrankenSplit: Eficient neural feature compression with shallow variational bottleneck injection for mobile edge computing. IEEE Transactions on Mobile Computing, 23(12):10770–10786, 2024. doi: 10.1109/TMC.2024.3381952.

Alireza Furutanpey, Qiyang Zhang, Philipp Raith, Tobias Pfandzelter, Shangguang Wang, and Schahram Dustdar. FOOL: Addressing the downlink bottleneck in satellite computing with neural feature compression. IEEE Transactions on Mobile Computing, 24(8):6747–6764, 2025. doi: 10.1109/TMC.2025.3544516.

Changsheng Gao, Zijie Liu, Li Li, Dong Liu, Xiaoyan Sun, and Weisi Lin. DT-UFC: Universal large model feature coding via peaky-to-balanced distribution transformation. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 5198–5207, 2025. doi: 10.1145/3746027.3755814. URL https://arxiv.org/abs/2506.16495.

Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, and Wieland Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. In International Conference on Learning Representations, 2019.

Deniz Gündüz, Zhijin Qin, Iñaki Estella Aguerri, Harpreet S. Dhillon, Zhaohui Yang, Aylin Yener, Kai Kit Wong, and Chan-Byoung Chae. Beyond transmitting bits: Context, semantics, and taskoriented communications. IEEE Journal on Selected Areas in Communications, 41(1):5–41, 2023. doi: 10.1109/JSAC.2022.3223408.

Sha Guo, Jing Chen, Zixuan Hu, Zhuo Chen, Wenhan Yang, Yu Lin, Xing Jiang, and Lingyu Duan. Which tasks should be compressed together? a causal discovery approach for eficient multi-task representation compression. In International Conference on Learning Representations, 2025.

Alon Harell, Yalda Foroutan, Nilesh Ahuja, Parual Datta, Bhavya Kanzariya, V. Srinivasa Somayazulu, Omesh Tickoo, Anderson de Andrade, and Ivan V. Bajić. Rate-distortion theory in coding for machines and its applications. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(7):5501–5519, 2025. doi: 10.1109/TPAMI.2025.3548516.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016. doi: 10.1109/CVPR.2016.90. URL https://arxiv.org/abs/1512.03385.

Zhimeng Huang, Rongao Yuan, Junlong Gao, Qi Mao, Siwei Ma, Wen Gao, and Chuanmin Jia. Discovering adaptive task dependencies for eficient multi-task representation compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5326–5336, 2026.

Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan Engstrom, Brandon Tran, and Aleksander Madry. Adversarial examples are not bugs, they are features. Advances in neural information processing systems, 32, 2019.

ISO/IEC JTC 1/SC 29/WG 1. JPEG AI use cases and requirements, version 3.0. Document WG1N101397, 110th JPEG Meeting, 2026. URL https://ds.jpeg.org/documents/jpegai/ wg1n101397-110-REQ-JPEG\_AI\_Use\_Cases\_and\_Requirements\_v3\_0.pdf.

Amir Ivry. Task-aware answer preservation under audio compression for large audio language models. arXiv preprint arXiv:2605.06631, 2026. URL https://arxiv.org/abs/2605.06631.

Alon Kipnis, Stefano Rini, and Andrea J. Goldsmith. Indirect rate-distortion function of a binary i.i.d. source. arXiv preprint arXiv:1505.04875, 2015. URL https://arxiv.org/abs/1505.04875.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009. URL https://www.cs.toronto.edu/\~kriz/learning-features-2009-TR.pdf.

Emin Martinian, Gregory W. Wornell, and Ram Zamir. Source coding with distortion side information. IEEE Transactions on Information Theory, 54(10):4638–4665, 2008. doi: 10.1109/TIT.2008.928983.

Yoshitomo Matsubara, Ruihan Yang, Marco Levorato, and Stephan Mandt. Supervised compression for resource-constrained edge computing systems. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 2685–2695, January 2022.

Yoshitomo Matsubara, Ruihan Yang, Marco Levorato, and Stephan Mandt. SC2 benchmark: Supervised compression for split computing. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=p28wv4G65d.

David Minnen, Johannes Ballé, and George Toderici. Joint autoregressive and hierarchical priors for learned image compression. In Advances in Neural Information Processing Systems, volume 31, 2018. URL https://arxiv.org/abs/1809.02736.

Siddharth Reddy, Anca D. Dragan, and Sergey Levine. Pragmatic image compression for human-in-the-loop decision-making. In Advances in Neural Information Processing Systems, volume 34, pp. 26499–26510, 2021.

Shibani Santurkar, Dimitris Tsipras, and Aleksander Madry. BREEDS: Benchmarks for subpopulation shift. In International Conference on Learning Representations, 2021. URL https://openreview.net/pdf/ 267e1b0387f6edaaa3b1145def1009b6803d55b6.pdf.

Vasileios Sevetlidis. Bayes-suficient representations in supervised learning. arXiv preprint arXiv:2606.04045, 2026. URL https://arxiv.org/abs/2606.04045.

Jiawei Shao, Yuyi Mao, and Jun Zhang. Learning task-oriented communication for edge inference: An information bottleneck approach. IEEE Journal on Selected Areas in Communications, 40(1):197–211, 2022. doi: 10.1109/JSAC.2021.3126087.

Photios A. Stavrou and Marios Kountouris. The role of fidelity in goal-oriented semantic communication: A rate distortion approach. IEEE Transactions on Communications, 2023. doi: 10.1109/TCOMM.2023. 3274122. URL https://www.eurecom.fr/en/publication/6943.

Brendan van Rooyen and Robert C. Williamson. Le Cam meets LeCun: Deficiency and generic feature learning. arXiv preprint arXiv:1402.4884, 2014. URL https://arxiv.org/abs/1402.4884.

Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The Caltech-UCSD Birds-200-2011 dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011. URL https://www.vision.caltech.edu/datasets/cub\_200\_2011/.

Yi Wang and Linglong Dai. From source reconstruction to predictive state preservation: An informationtheoretic framework for AI-native communication. arXiv preprint arXiv:2609.01131, 2026. URL https: //arxiv.org/abs/2609.01131.

Hans S. Witsenhausen. Indirect rate distortion problems. IEEE Transactions on Information Theory, 26(5): 518–521, 1980. doi: 10.1109/TIT.1980.1056251.

Amir R. Zamir, Alexander Sax, William Shen, Leonidas J. Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018.

Zifu Zhang, Shengxi Li, Xiancheng Sun, Mai Xu, Zhengyuan Liu, and Jingyuan Xia. Machines serve human: A novel variable human-machine collaborative compression framework. IEEE Transactions on Image Processing, 35:7828–7842, 2026. doi: 10.1109/TIP.2026.3712352.

## A Coding results and proofs

The coding model has finite alphabets, independent and identically distributed samples, componentwise losses, and the permitted prediction tuples of Section 2.1.

## A.1 Jointly attainable predictions and equality of coding rates

For a source reconstruction channel, define ${ \widehat { T } } = F ( { \widehat { X } } )$ . Every task risk is unchanged, and data processing gives $I ( X ; { \widehat { T } } ) \leq I ( X ; { \widehat { X } } )$ . Given equation 4, choose a representative $s ( \hat { t } ) \in \widehat { \mathcal { X } }$ with $F ( s ( \hat { t } ) ) = \hat { t }$ for every permitted prediction tuple t<sup>ˆ</sup>. Given a prediction channel, set $\widehat { X } = s ( \widehat { T } )$ . Because the representative and its prediction tuple determine one another, $I ( X ; { \widehat { X } } ) = I ( X ; { \widehat { T } } )$ , with the same task risks. Applying the same construction with representatives in $\widehat { \mathcal { V } }$ proves the reduction for the split and direct rates.

For the deterministic target $T = f ( X ) = h ( g ( X ) )$ in Harell et al. (2025), the conditional expected task distortion depends on X only through $Y = g ( X )$ ). Averaging a source-dependent reconstruction channel over X given Y preserves that distortion and does not increase mutual information. Hence the direct and split rates are equal. Source equality additionally uses an attainable-output condition or a replacement output with no greater distortion. For several tasks, one replacement must have no greater distortion for any task. Separate task replacements may form a tuple that no reconstruction can produce.

## A.2 Prescribed observation

Fix λ. All sums range over the permitted prediction tuples in equation 4. For $O = X , Y$ , put

$$
Z _ { O } ^ { r } ( o ) = \sum _ { \hat { t } } r ( \hat { t } ) 2 ^ { - c _ { \mathbf { \lambda } , O } ( o , \hat { t } ) } , \qquad \Phi _ { O } ( r ) = - \mathbb { E } \log _ { 2 } Z _ { O } ^ { r } ( O ) .\tag{16}
$$

Finite costs make $Z _ { O } ^ { r }$ positive throughout the compact probability simplex. For any prediction channel $p$ with marginal $p _ { \widehat { T } }$

$$
\begin{array} { r l } & { \mathbb { E } D _ { \mathrm { K L , 2 } } ( p ( \cdot  { | \mathbf {  { O } } ) | } | r ) = I ( { \cal O } ; \widehat { T } ) + D _ { \mathrm { K L , 2 } } ( p _ { \widehat { T } } | | r ) , } \\ & { \mathbb { E } D _ { \mathrm { K L , 2 } } ( p ( \cdot  { | \mathbf {  { O } } ) | } | r ) + \mathbb { E } c _ { \lambda , { \cal O } } ( { \cal O } , \widehat { T } ) = \Phi _ { { \cal O } } ( r ) + \mathbb { E } D _ { \mathrm { K L , 2 } } ( p ( \cdot  { | \mathbf {  { O } } ) | } | \pi _ { { \cal O } } ^ { r } ( \cdot  { | \mathbf {  { O } } ) } ) . } \end{array}\tag{17}
$$

Minimizing jointly over $p , r$ gives $J _ { O } = \operatorname* { m i n } _ { r } \Phi _ { O } ( r )$ . At a global minimum, $\pi _ { O } ^ { r }$ has marginal $r .$ . These are the finite Shannon variational identities underlying the support conditions of Caplin et al. (2019); Armenter et al. (2024).

On the support of $r ,$ the logarithm of the ratio $\pi _ { Y } ^ { r } ( \hat { t } \mid y ) / \pi _ { X } ^ { r } ( \hat { t } \mid x )$ equals

$$
c _ { \lambda , X } ( x , \hat { t } ) - c _ { \lambda , Y } ( y , \hat { t } ) + \log _ { 2 } { Z _ { X } ^ { r } ( x ) } - \log _ { 2 } { Z _ { Y } ^ { r } ( y ) } .
$$

Average with probabilities $p _ { X } ( x ) \pi _ { Y } ^ { r } ( \hat { t } \mid g ( x ) )$ . Conditional expectation given Y cancels the cost diference, proving

$$
\begin{array} { r } { \Phi _ { Y } ( r ) - \Phi _ { X } ( r ) = G _ { \lambda } ( r ) . } \end{array}\tag{18}
$$

Evaluating this identity at $r _ { Y }$ and $r _ { X }$ proves equation 11.

If equation 10 holds at an X-optimal r, then on its support $\boldsymbol { c } _ { \lambda , X } ( \boldsymbol { x } , \hat { t } ) = \boldsymbol { u } ( \boldsymbol { x } ) + \boldsymbol { v } ( \boldsymbol { g } ( \boldsymbol { x } ) , \hat { t } )$ . The factor $2 ^ { - u ( x ) }$ cancels in equation 8, so $\pi _ { X } ^ { r } = \pi _ { Y } ^ { r }$ and $J _ { X } = J _ { Y }$ . Conversely, equality gives $J _ { X } \le \Phi _ { X } ( r _ { Y } ) \le \Phi _ { Y } ( r _ { Y } ) = J _ { Y }$ Hence $r _ { Y }$ is also X-optimal and its two Gibbs channels coincide. For supported tuples, the ratio of the Gibbs probabilities satisfies

$$
\frac { \pi _ { X } ^ { r _ { Y } } ( \hat { t } \mid x ) } { \pi _ { X } ^ { r _ { Y } } ( \hat { u } \mid x ) } = \frac { r _ { Y } ( \hat { t } ) } { r _ { Y } ( \hat { u } ) } 2 ^ { - \left[ c _ { \lambda , X } ( x , \hat { t } ) - c _ { \lambda , X } ( x , \hat { u } ) \right] } ,
$$

which proves equation 10. With finite costs, the Gibbs probabilities are positive on the support of $r _ { Y }$ , so these ratios are defined. Global optimality additionally requires

$$
\mathbb { E } _ { X } \frac { 2 ^ { - c _ { \lambda , X } ( X , \hat { t } ) } } { Z _ { X } ^ { r } ( X ) } \leq 1 \quad \mathrm { f o r ~ e v e r y ~ p e r m i t t e d ~ } \hat { t } , \qquad \mathrm { w i t h ~ e q u a l i t y ~ i f ~ } r ( \hat { t } ) > 0 .\tag{19}
$$

$G _ { \lambda } ( r _ { Y } )$ can equal zero when $J _ { Y } > J _ { X }$ , because $J _ { Y } - J _ { X } = G _ { \lambda } ( r _ { Y } ) + \Phi _ { X } ( r _ { Y } ) - J _ { X }$

For equation 12, use ${ \cal R } _ { { \cal O } } ( { \bf D } ) = J _ { { \cal O } } ( \pmb { \lambda } _ { { \cal O } } ) - \pmb { \lambda } _ { { \cal O } } \cdot \mathbf { D }$ . Evaluating the Y dual at $\lambda _ { X }$ gives the lower bound, and evaluating the X dual at λ<sub>Y</sub> gives the upper bound. At any D feasible from Y , the rates are equal exactly when some constrained X-optimal prediction channel factors through Y . If the rates are equal, a Y -optimal channel composed with $g$ is also X-optimal. Conversely, an X-optimal channel that depends on X only through Y is feasible from $Y$ at the same rate. If finite source multipliers $\lambda _ { X }$ attain the dual optimum, an equivalent certificate is an X-optimal Gibbs marginal r at these multipliers satisfying equation 10, whose channel has risks $D _ { q } ^ { r } \leq D _ { q }$ and $\lambda _ { X , q } ( D _ { q } - D _ { q } ^ { r } ) = 0$ for every q. At feasible boundaries, compactness gives $R _ { O } ( { \bf D } + \epsilon { \bf 1 } )  R _ { O } { \bf \bar { ( } D ) }$ as $\epsilon \downarrow 0$ . Applying the finite-multiplier bounds at these strictly feasible relaxed tolerances provides boundary bounds by limits. If D is infeasible from $Y$ , then $R _ { Y } ( { \bf D } ) = + \infty$

## A.3 Three-class conditional error costs

Let $P ( Z ) = ( 0 . 3 5 , 0 . 3 5 , 0 . 3 0 )$ , let K be an independent fair sign, and set $X = ( Z , K ) , Y = Z$ . The original target is $T _ { 0 } = Z . \mathrm { ~ W r i t e ~ } a = 0 . 5 5 , b = 0 . 2 2 5 , h = a - b = 0 . 3 2 5 , \rho = 0 . 8 5 h$ , and $s = K \rho / 3$ . The additional target has conditional law

$$
\begin{array} { r l } & { P ( T _ { 1 } = \cdot \mid Z = 0 , K ) = ( a + s , b + s , b - 2 s ) , } \\ & { P ( T _ { 1 } = \cdot \mid Z = 1 , K ) = ( b + s , a + s , b - 2 s ) , } \\ & { P ( T _ { 1 } = \cdot \mid Z = 2 , K ) = ( b , b , a ) . } \end{array}\tag{20}
$$

For zero–one loss, both observations have unique Bayes tuple $( Z , Z )$ and risks (0, 0.45). Every conditional class probability is positive, and the minimum Bayes margin is $h - \rho = 0 . 0 4 8 7 5$

For the additional task alone, put $\beta = \lambda h$ . Symmetry and convexity allow a marginal with $r ( 0 ) = r ( 1 ) =$ $( 1 - r ( 2 ) ) / 2 . \mathrm { A t } r ( 2 ) = 0$ , the derivative of $\Phi _ { X }$ with respect to $r ( 2 )$ is $( 1 - S _ { X } ) / \ln 2$ , where

$$
S _ { X } ( \beta ) = 0 . 7 \frac { 2 ^ { - 0 . 1 5 \beta } + 2 ^ { - 1 . 8 5 \beta } } { 1 + 2 ^ { - \beta } } + 0 . 3 2 ^ { \beta } .\tag{21}
$$

For $\beta > 0$ , prediction 2 is unused exactly when $S _ { X } \leq 1$ . For $t = 2 ^ { \beta }$ , multiplying $S _ { X } - 1$ by $t + 1$ gives $f ( t ) = 0 . 3 t ^ { 2 } - 0 . 7 t - 1 + 0 . 7 ( t ^ { 0 . 8 5 } + t ^ { - 0 . 8 5 } )$ . On $t \ge 1 , f ( 1 ) = 0 , f ^ { \prime } ( 1 ) = - 0 . 1$ , and $f ^ { \prime \prime } ( t ) \geq 0 . 5 1 0 7 5 .$ Since $f ( t )  \infty$ , it has exactly one further root. Prediction 2 has positive probability in the source-optimal marginal for $\beta > \beta _ { X } \simeq 0 . 1 8 1 5 2 9 9 0$ . Averaging the costs replaces $t ^ { 0 . 8 5 } + t ^ { - 0 . { \bar { 8 } } 5 } \ \mathrm { b y \ 2 }$ . Prediction 2 has positive probability in the feature-optimal marginal for $\beta > \beta _ { Y } = \log _ { 2 } ( 4 / 3 ) \simeq 0 . 4 1 5 0 3 7 5 0$

For $\beta > 0$ , the exponential-cost matrices have full column rank, so their marginal objectives are strictly convex. Divide each source row by its correct-prediction entry. Subtracting the exchanged class rows forces the first two coordinates of a null vector to agree. Subtracting the confidence-sign rows then forces its third coordinate to vanish, and the first two vanish as well. The feature matrix has diagonal 1 and of-diagonal $2 ^ { - \beta }$ . For $0 ~ < ~ \beta ~ \le ~ \beta _ { X }$ , only predictions 0, 1 have positive probability in the source-optimal marginal. Their conditional cost diference is independent of K, so $J _ { X } = J _ { Y }$ . For $\beta > \beta _ { X }$ , the source-optimal marginal assigns positive probability to prediction 2 and at least one other prediction. Their conditional cost diference depends on K, so $J _ { X } < J _ { Y }$ . A singleton prediction-2 optimum is excluded by its larger constant-prediction risk.

At the finite-code confirmation’s tolerance $D _ { 1 } = 0 . 4 9 9$ , minimizing the strictly convex marginal objective gives source multiplier $\lambda _ { X } \simeq 8 . 9 3 6 0 3 3 2 4 7 6 5 5 .$ , rate 0.635401148729, and original-label error 0.294561875080. Writing $p = ( 0 . 3 5 , 0 . 3 5 , 0 . 3 0 )$ and $e = ( D _ { 1 } - 0 . 4 5 ) / 0 . 3 2 5$ , Fano’s inequality for a ternary variable gives a matching lower bound for

$$
R _ { Y } ^ { ( 1 ) } ( D _ { 1 } ) = H ( p ) - h _ { 2 } ( e ) - e , \qquad 0 \leq e \leq 0 . 6 .\tag{22}
$$

To attain it, choose $r _ { i } = ( p _ { i } - e / 2 ) / ( 1 - 3 e / 2 )$ and the backward channel $P ( Z = i \mid \widehat { T } _ { 1 } = j ) = 1 - e$ for $i = j$ , and $e / 2$ otherwise. Mixing these conditional distributions with weights $r _ { j }$ gives marginal p for $Z$ and conditional entropy $H ( Z \mid { \widehat { T } } _ { 1 } ) = h _ { 2 } ( e ) + e$ . At $D _ { 1 } = 0 . 4 9 9$ , the feature rate is 0.818759706492 and its original-label error is $e \simeq 0 . 1 5 0 7 6 9 2 3 0 7 6 9$

Set $D _ { 0 } = 0 . 3 5$ and permit all nine prediction pairs. Projection onto $\widehat { T } _ { 1 }$ gives the standalone additional-task rate as a family lower bound. Setting $\widehat { T } _ { 0 } = \widehat { T } _ { 1 }$ attains it from either observation while satisfying $D _ { 0 }$ . Thus the family-rate gap at $( D _ { 0 } , D _ { 1 } ) = ( 0 . 3 5 , 0 . 4 9 9 )$ is approximately 0.183358557763 bits per sample.

The interval of equal rates for the additional task alone ends near 0.00199467 bits per sample. For the common-prediction construction on this interval, the original-task error is $0 . 3 0 + 0 . 7 0 / ( 1 + 2 ^ { \beta } ) \geq 0 . 6 2 8 0 0 9 2 9$ above $D _ { 0 } = 0 . 3 5$ . Even with separate predictions, the original-task Fano bound is $H ( p ) - h _ { 2 } ( 0 . 3 5 ) - 0 . 3 5 \simeq$ 0.29722284. Finally, if $D _ { 0 } = 0$ , every feasible message determines $Z$ and requires at least $H ( Z )$ bits per sample. Sending Z attains additional risk 0.45, so both family rates equal $H ( Z )$ for every feasible $D _ { 1 } \geq 0 . 4 5$

## A.4 Information beyond a preservation target

The chain-rule argument of Achille & Soatto (2018, Proposition 3.1) bounds information beyond a deterministic preservation target. With lossy target reconstruction, the bound depends on complete coded length and the target’s rate–distortion function.

Let $( X _ { i } , U _ { i } )$ be iid and finite, and let $T _ { i } = f ( X _ { i } )$ . A complete binary prefix-coded message M is generated from $X ^ { n }$ using randomness independent of the source and targets. A decoder using only M reconstructs $T ^ { n }$ at mean additive distortion at most D. Write $r = \mathbb { E } | M | / n$ and let $R _ { T } ( D )$ be the ordinary target rate–distortion function assuming the same distortion and reconstruction alphabet. Then

$$
\frac { 1 } { n } I ( U ^ { n } ; M \mid T ^ { n } ) \le r - R _ { T } ( D ) .\tag{23}
$$

By the prefix-code converse, the chain rule, the target rate–distortion converse, and conditional data processing,

$$
\begin{array} { r l } & { n r \geq H ( M ) \geq I ( X ^ { n } ; M ) } \\ & { \quad = I ( T ^ { n } ; M ) + I ( X ^ { n } ; M \mid T ^ { n } ) } \\ & { \quad \geq n R _ { T } ( D ) + I ( U ^ { n } ; M \mid T ^ { n } ) . } \end{array}\tag{24}
$$

The last step uses $U ^ { n } \longrightarrow X ^ { n } \longrightarrow M$ and deterministic $T ^ { n }$ . Source-independent shared model parameters can be conditioned on throughout. A rate close to $R _ { T } ( D )$ limits additional information beyond $T ^ { n }$ . The image-code rates are not compared with an estimated $R _ { T } ( D )$ .

If only the known coarse and fine prediction tuple is required, computing both predictions at the encoder permits transmission in 11 fixed-length bits per image before framing.

## A.5 Noisy threshold tasks

Proposition 2 (Overlapping thresholds). Fix $D _ { C } = \epsilon ,$ , write $e = ( D _ { B } - \epsilon ) / ( 1 - 2 \epsilon )$ , and let $H _ { C } = h _ { 2 } ( 2 / 3 )$ ， where $h _ { 2 } ( u ) = - u \log _ { 2 } u - ( 1 - u ) \log _ { 2 } ( 1 - u )$ is binary entropy, with $0 \log _ { 2 } 0 = 0$ . For $D _ { B } \geq \epsilon$

$$
R _ { X } ( \epsilon , D _ { B } ) = R _ { X Y } ( \epsilon , D _ { B } ) = \left\{ \begin{array} { l l } { H _ { C } + \frac 2 3 \big [ h _ { 2 } ( 1 / 4 ) - h _ { 2 } ( 3 e / 2 ) \big ] , } & { e < 1 / 6 , } \\ { H _ { C } , } & { e \ge 1 / 6 . } \end{array} \right.\tag{25}
$$

The split-feature rate $R _ { Y } ( \epsilon , D _ { B } )$ is infinite below $D _ { B , Y } ^ { * }$ and equals $H _ { C }$ at or above it. All three rates are infinite if either task requires risk below ϵ.

For Proposition 2, independent label noise gives target risk $\epsilon + ( 1 - 2 \epsilon ) e _ { q }$ , where $e _ { q }$ is the error in predicting underlying threshold bit $q \in \{ C , B \}$ . For the target risk to converge to $\epsilon ,$ the average error in C must converge to zero. The pair $( C , B )$ has probabilities $( 1 / 3 , 1 / 6 , 1 / 2 )$ on (00, 10, 11). On the event $C = 0$ $B = 0$ . Conditional on $C = 1$ , B is Bernoulli with parameter $3 / 4$

For a length-n encoded message $M _ { n }$ and average error $e _ { C }$ in the decoded coarse predictions, binary Fano’s inequality gives

$$
I ( C ^ { n } ; M _ { n } ) \ge n H _ { C } - n h _ { 2 } ( e _ { C } ) .\tag{26}
$$

If the additional threshold has average error $e _ { B } < 1 / 6$ , its error conditional on $C = 1$ is at most 3e $_ B / 2$ . The conditional entropy bound and concavity of binary entropy give

$$
I ( B ^ { n } ; M _ { n } \mid C ^ { n } ) \ge \frac { 2 n } { 3 } \big [ h _ { 2 } ( 1 / 4 ) - h _ { 2 } ( 3 e _ { B } / 2 ) \big ] .\tag{27}
$$

For this bound, expand $H ( B ^ { n } \mid M _ { n } , C ^ { n } )$ over positions, retain only the conditioning on $C _ { i }$ and decoded prediction $\widehat { T } _ { B , i }$ , and apply binary Fano’s inequality when $C _ { i } = 1$ . The total error on those positions is at most the overall error. Averaging over positions gives Eq. 27. Since both $C ^ { n }$ and $B ^ { n }$ are functions of $X ^ { n }$ the sum of these two mutual informations is at most $I ( X ^ { n } ; M _ { n } )$ . Letting $e _ { C } \to 0$ proves the converse in Eq. 25. When $e _ { B } \geq 1 / 6$ , the coarse bound alone gives $H _ { C }$

For achievability, encode $C ^ { n }$ losslessly at asymptotic rate $H _ { C }$ and apply a binary Hamming rate–distortion code to the subsequence with $C _ { i } = 1$ . Its rate per original sample is the second term in Eq. 25. For $e _ { B } \geq 1 / 6$ send only C and predict $\widehat { T } _ { B } = C$ . The three joint predictions are obtained from source representatives $\widehat { S } = 0 , 2 , 3 .$ , with any fixed permitted value of ${ \widehat { N } } ,$ or feature representatives $\widetilde { Y } = 0 , 1 , 2$ . Independent N afects neither task’s distortion and can be discarded.

The feature value $Y ~ = ~ 1$ corresponds to equally likely $S \ = \ 2 , 3 .$ , so every prediction of B from Y has underlying-bit error at least $( 1 / 3 ) ( 1 / 2 ) = 1 / 6$ . At or above the resulting target-risk floor, transmitting C attains rate $H _ { C }$ , which the coarse task also requires. This proves the split-feature formula.

We can use the same family to compare reference prediction risks while keeping encoder access fixed at X. Let $V = Y$ be a coarser observation used only to specify those risks, and let $\eta \geq 0$ be the tolerated excess risk for the additional task. With tolerances equal to the reference risks from $V$ plus allowance $( 0 , \eta )$ , the minimum rate is $H _ { C }$ bits per sample. Using the reference risks from X with the same allowance requires an additional

$$
\Delta R ( \eta ) = \frac { 2 } { 3 } \left[ h _ { 2 } ( 1 / 4 ) - h _ { 2 } \biggl ( \frac { 3 \eta } { 2 ( 1 - 2 \epsilon ) } \biggr ) \right]\tag{28}
$$

for $0 \leq \eta < ( 1 - 2 \epsilon ) / 6$ , and zero thereafter. $\mathrm { A t ~ } \epsilon = 0 . 0 5$ and zero allowance, the diference is 0.54085 bits per sample (Figure 2a). Preserving the lower errors for these two noisy decisions requires fewer additional bits than $H ( X \mid V ) = 1 + H ( N )$ , which includes the independent sensor variation.

With encoder observation fixed at X, the reference risk vectors are $( \epsilon , \epsilon )$ from X and $( \epsilon , \epsilon + ( 1 - 2 \epsilon ) / 6 )$ from $V = Y$ . Adding $( 0 , \eta )$ to each and substituting into Eq. 25 gives Eq. 28.

For arbitrary tolerances on both decisions, we obtain the complete region by solving the finite-alphabet rate–distortion problem with two constraints. Order the prediction columns as (00, 10, 11). Index the source rows by the underlying pairs $( C , B ) = ( 0 0 , 1 0 , 1 1 )$ with probabilities $( 1 / 3 , 1 / 6 , 1 / 2 )$ , and the feature rows by $Y = 0 , 1 , 2$ with probabilities $( 1 / 3 , 1 / 3 , 1 / 3 )$ . The corresponding underlying-bit distortion matrices are

$$
d _ { C } = \left( { \begin{array} { c c c } { 0 } & { 1 } & { 1 } \\ { 1 } & { 0 } & { 0 } \\ { 1 } & { 0 } & { 0 } \end{array} } \right) , \qquad d _ { B , X } = \left( { \begin{array} { c c c } { 0 } & { 0 } & { 1 } \\ { 0 } & { 0 } & { 1 } \\ { 1 } & { 1 } & { 0 } \end{array} } \right) , \qquad d _ { B , Y } = \left( { \begin{array} { c c c } { 0 } & { 0 } & { 1 } \\ { 1 / 2 } & { 1 / 2 } & { 1 / 2 } \\ { 1 } & { 1 } & { 0 } \end{array} } \right) .\tag{29}
$$

The coarse-task matrix is the same for both observations. For the noisy targets, transform every matrix entry d into $\epsilon + ( 1 - 2 \epsilon ) d$ to obtain the conditional expected target loss (Witsenhausen, 1980; Kipnis et al., 2015).

## A.6 Conditional error profiles

Let Z be a fair bit independent of finite $K \in \mathcal { K }$ , with $\operatorname* { P r } ( K = k ) > 0$ for every k. Set $X = ( Z , K , N )$ and $Y = Z$ , with finite sensor variation N independent of all task variables. Each task has target $T _ { q } = Z \oplus E _ { q } ,$ zero–one loss, and

$$
\operatorname* { P r } ( E _ { q } = 1 \mid Z , K = k ) = \epsilon _ { q } ( k ) < 1 / 2 .
$$

An encoder observing X has access to K but not $E _ { q }$ . Every task has the unique Bayes prediction Z from either observation. Its Bayes risk and conditional excess cost for predicting $1 - Z$ are

$$
p _ { q } = \mathbb { E } \epsilon _ { q } ( K ) , \qquad w _ { q } ( k ) = 1 - 2 \epsilon _ { q } ( k ) > 0 , \qquad \bar { w } _ { q } = \mathbb { E } w _ { q } ( K ) = 1 - 2 p _ { q } .\tag{30}
$$

Observing K permits more coding errors where $w _ { q } ( k )$ is smaller, as in distortion-side-information coding (Martinian et al., 2008).

For a tolerated risk vector $\mathbf { D } = ( D _ { 1 } , \ldots , D _ { m } )$ with $D _ { q } \geq p _ { q }$ , define

$$
e ^ { * } = \operatorname* { m i n } \left\{ \frac { 1 } { 2 } , \operatorname* { m i n } _ { q \in Q } \frac { D _ { q } - p _ { q } } { \bar { w } _ { q } } \right\} , \qquad { \bf v } _ { q } = \left( \frac { w _ { q } ( k ) } { \bar { w } _ { q } } \right) _ { k \in \mathcal { K } } .\tag{31}
$$

Here $e ^ { * }$ is the largest common crossover probability in $[ 0 , 1 / 2 ]$ that satisfies all risk constraints when the encoder observes only $Z .$ The vector $\mathbf { v } _ { q }$ records task $q \mathrm { { ^ { * } s } }$ normalized error costs across confidence states and has expectation one over $K$ . Let $J \subseteq Q$ contain exactly the tasks attaining the inner minimum, and let 1 denote the vector of ones indexed by $\kappa .$

Proposition 3 (Conditional error profiles). Allow every joint prediction in $\{ 0 , 1 \} ^ { m }$ . The minimum rates for observations X and $Y$ are

$$
R _ { X } ( \mathbf { D } ) = 1 - \operatorname* { m a x } _ { e } \mathbb { E } \big [ h _ { 2 } ( e ( K ) ) \big ] ,\tag{32}
$$

$$
R _ { Y } ( \mathbf { D } ) = 1 - h _ { 2 } ( e ^ { * } ) .\tag{33}
$$

We maximize over functions e: $\mathcal { K }  [ 0 , 1 / 2 ]$ satisfying

$$
\begin{array} { r } { \mathbb { E } \big [ w _ { q } ( K ) e ( K ) \big ] \leq D _ { q } - p _ { q } , \qquad q \in Q . } \end{array}
$$

Here $e ( k )$ is the conditional probability of predicting $1 - Z$ at state $k .$ Each minimum rate is attained by a channel predicting the same bit for every task, although all joint prediction tuples are permitted. For $0 < e ^ { * } < 1 / 2$

$$
R _ { X } ( \mathbf { D } ) = R _ { Y } ( \mathbf { D } ) \quad \Longleftrightarrow \quad \mathbf { 1 } \in \mathrm { c o n v } \{ \mathbf { v } _ { q } : q \in J \} .\tag{34}
$$

The convex hull contains all weighted averages with nonnegative weights summing to one. If the condition fails, $R _ { X } < R _ { Y }$ . Both rates are infinite $i f$ any $D _ { q } < p _ { q }$ , one $i f$ some $D _ { q } = p _ { q }$ and all other requirements are feasible, and zero $i f$ every $D _ { q } \geq 1 / 2$

Restricting the permitted tuples to $( 0 , \ldots , 0 )$ and $( 1 , \ldots , 1 )$ leaves both minimum rates unchanged. These are exactly the joint predictions obtainable from the readouts $f _ { q } ( z , k , n ) = z { \mathrm { ~ a n d ~ } } h _ { q } ( y ) = y$ . The reduction from the full product $\{ 0 , 1 \} ^ { m }$ to these two tuples therefore lets us identify the optimized prediction rates with the source and split-feature rates for these readouts. Direct-feature coding has the same rate as source coding, $R _ { X Y } ( \mathbf { D } ) = R _ { X } ( \mathbf { D } )$ , because it can reconstruct either of the two feature values.

Both task content and tolerated risk determine which profiles enter the equality condition. At a constant optimum, tasks with normalized allowance above $e ^ { * }$ have slack risk constraints and zero multipliers. If every task permits the same fraction $\alpha \in ( 0 , 1 )$ of the interval from its Bayes risk to the constant-prediction risk, then $D _ { q } = p _ { q } + \alpha ( 1 / 2 - p _ { q } ) , e ^ { * } = \alpha / 2$ , and every task is active. Rate equality then requires 1 to belong to the convex hull of the entire family. Repeating a task changes neither its constraint nor this convex hull.

The full joint prediction alphabet. For Proposition 3, write $r _ { q } = D _ { q } - p _ { q }$ for the permitted excess risk. For a joint prediction $\hat { t } \in \{ 0 , 1 \} ^ { m }$ , task $q$ has conditional excess distortion $w _ { q } ( k ) \mathbf { 1 } \{ \hat { t } _ { q } \ \neq \ z \}$ . We can remove independent N by averaging the coding channel conditionally on $( Z , K )$ , without changing any task risk or increasing mutual information. For nonnegative multipliers, put $\begin{array} { r } { L _ { k } = \sum _ { q } \lambda _ { q } w _ { q } ( k ) } \end{array}$ and $\begin{array} { r } { u _ { k } ( \hat { t } ) = \sum _ { q } \lambda _ { q } w _ { q } ( k ) \hat { t } _ { q } } \end{array}$ . By the variational formula in equation 17, we minimize

$$
- \mathbb { E } _ { Z , K } \log _ { 2 } \sum _ { \hat { t } \in \{ 0 , 1 \} ^ { m } } \pi ( \hat { t } ) 2 ^ { - \sum _ { q } \lambda _ { q } w _ { q } ( K ) \mathbf { 1 } \{ \hat { t } _ { q } \neq Z \} }\tag{35}
$$

over probability distributions $\pi$ on all binary prediction tuples. To obtain the constrained dual objective, subtract $\sum _ { q } \lambda _ { q } r _ { q }$

The objective is convex in $\pi .$ . Complementing Z and all predictions preserves the source law and costs, so averaging π with its complement cannot increase the objective. A complement pair of total mass θ then contributes

$$
\frac { \theta } { 2 } \left[ 2 ^ { - u _ { k } ( \hat { t } ) } + 2 ^ { - \left( L _ { k } - u _ { k } ( \hat { t } ) \right) } \right] \leq \frac { \theta } { 2 } ( 1 + 2 ^ { - L _ { k } } )\tag{36}
$$

to the partition sum for either source bit. For $0 \leq u \leq L _ { k }$ , the inequality follows from

$$
1 + 2 ^ { - L _ { k } } - 2 ^ { - u } - 2 ^ { - ( L _ { k } - u ) } = ( 1 - 2 ^ { - u } ) ( 1 - 2 ^ { - ( L _ { k } - u ) } ) \geq 0 .\tag{37}
$$

We can therefore assign each pair’s probability to the all-zero and all-one pair without decreasing any partition sum. The argument applies separately at every confidence state. This extends the single-task symmetrization of Martinian et al. (2008, Appendix I) without requiring symmetry among tasks or confidence states. Multiple-distortion Gibbs channels also appear in Stavrou & Kountouris (2023).

Both the full-product and common-prediction Lagrangian infima are consequently

$$
1 - \mathbb { E } \log _ { 2 } ( 1 + 2 ^ { - L _ { K } } ) ,\tag{38}
$$

attained by crossover probabilities $e ( k ) = 1 / ( 1 + 2 ^ { L _ { k } } )$ . When every $r _ { q } > 0$ , mixing the exact common prediction with a suficiently small positive probability of uniform joint predictions gives full support and strictly satisfies every risk constraint. In the problem restricted to a common prediction, a suficiently smal positive constant crossover probability lies in (0, 1/2) and strictly satisfies every risk constraint. Slater’s condition holds for both convex programs. Their equal Lagrangian infima for every nonnegative multiplier vector imply equal constrained optima.

If some $r _ { q } = 0$ , positivity of $w _ { q } ( k )$ forces $\widehat { T } _ { q } = Z$ , and data processing requires at least one bit. Sending Z uses one bit per sample and gives every task zero excess risk, satisfying all nonnegative excess-risk allowances. Negative allowances are infeasible. If every $D _ { q } \geq 1 / 2$ , a constant common prediction gives zero rate.

Error allocation and restricted observations. Let U be the common binary prediction, so $\widehat { T } _ { q } = U$ for every task. Averaging the channel with its bit-complemented counterpart gives a binary symmetric channel conditional on K. Its output is fair and independent of K, so

$$
I ( Z , K ; U ) = 1 - \mathbb { E } h _ { 2 } ( e ( K ) ) , \qquad \mathbb { E } d _ { T _ { q } } ( T _ { q } , U ) = p _ { q } + \mathbb { E } [ w _ { q } ( K ) e ( K ) ] .\tag{39}
$$

Replacing any conditional crossover probability above one half by its complement preserves its entropy and reduces every task risk. This proves Eq. 32 on the stated interval. When the encoder observes only $Y = Z$ the conditional error weights are $\bar { w } _ { q }$ . The same complement-pair argument proves optimality of a common prediction. Its tightest normalized requirement gives Eq. 33.

The active-profile criterion. Suppose $0 < e ^ { * } < 1 / 2$ . If nonnegative coeficients $\rho _ { q } ,$ indexed by $q \in J ,$ sum to one and satisfy $\textstyle \sum _ { q \in J } \rho _ { q } \mathbf { v } _ { q } = \mathbf { 1 }$ , the same weighted sum of the normalized risk constraints gives $\mathbb { E } e ( K ) \le e ^ { * }$ . Jensen’s inequality and monotonicity of binary entropy on $[ 0 , 1 / 2 ]$ give

$$
\begin{array} { r } { \mathbb { E } h _ { 2 } ( e ( K ) ) \leq h _ { 2 } ( \mathbb { E } e ( K ) ) \leq h _ { 2 } ( e ^ { * } ) . } \end{array}\tag{40}
$$

The constant allocation $e ( k ) = e ^ { * }$ is feasible, and both inequalities hold with equality.

Conversely, the entropy objective is strictly concave, and $e ^ { * }$ is inside the error interval. The constant allocation $e ^ { * } / 2$ strictly satisfies every constraint. The Karush–Kuhn–Tucker conditions are therefore necessary and suficient. At the constant optimum, the constraints outside J have strict slack and zero multipliers. Stationarity gives

$$
h _ { 2 } ^ { \prime } ( e ^ { * } ) { \bf 1 } = \sum _ { q \in J } \mu _ { q } { \bf v } _ { q } , \qquad \mu _ { q } \geq 0 .\tag{41}
$$

All confidence-state probabilities are positive, so they cancel from the coordinate equations. Taking expectation over K gives $\begin{array} { r } { \sum _ { q \in J } \mu _ { q } = h _ { 2 } ^ { \prime } ( e ^ { * } ) > 0 } \end{array}$ . Dividing each multiplier by this sum gives the required convex combination. If 1 is outside that convex hull, the constant allocation fails a necessary optimality condition. The feasible set is compact, so a maximizing allocation exists and has $\mathbb { E } h _ { 2 } ( e ( K ) ) > h _ { 2 } ( e ^ { * } )$ . Hence the rate from X is strictly smaller.

Two-state and three-state families. For two equiprobable confidence states and first-task weights $( w _ { 0 } , w _ { 1 } ) , \mathrm { E q } .$ 32 reduces to the single-task expression of Martinian et al. (2008, Section IV-E, Eqs. 40–44),

$$
R _ { X } ^ { ( 1 ) } ( D ) = 1 - \operatorname* { m a x } _ { e _ { 0 } , e _ { 1 } } { \frac { h _ { 2 } ( e _ { 0 } ) + h _ { 2 } ( e _ { 1 } ) } { 2 } } ,\tag{42}
$$

where $e _ { k } = e ( k )$ and $p$ is the task’s Bayes risk. We maximize over $e _ { 0 } , e _ { 1 } \in [ 0 , 1 / 2 ]$ satisfying

$$
\frac { w _ { 0 } e _ { 0 } + w _ { 1 } e _ { 1 } } { 2 } \leq D - p .
$$

The Gibbs crossover probabilities satisfy $e _ { k } = 1 / ( 1 + 2 ^ { \lambda w _ { k } } )$ . Swapping the weights for the second task makes the two normalized profiles average to 1. At equal tolerances, the active-profile criterion gives Eq. 15. For a single nonconstant profile, its convex hull does not contain 1, so $R _ { X } ^ { ( 1 ) } ( D ) < R _ { Y } ^ { ( 1 ) } ( D )$ for $p < D < 1 / 2$

For three equiprobable confidence states, choose three tasks with rows of conditional error costs

$$
\left( { \begin{array} { l l l } { 0 . 9 } & { 0 . 5 } & { 0 . 1 } \\ { 0 . 1 } & { 0 . 9 } & { 0 . 5 } \\ { 0 . 5 } & { 0 . 1 } & { 0 . 9 } \end{array} } \right) .\tag{43}
$$

Since $\epsilon _ { q } ( k ) = ( 1 - w _ { q } ( k ) ) / 2$ , each task’s noise probabilities are a permutation of (0.05, 0.25, 0.45). Their mean is the Bayes risk $p _ { q } = 0 . 2 5$ . The chosen component tolerance $D _ { q } = 0 . 3 0$ leaves excess risk 0.05. Using Eq. 32, we maximize $\textstyle { \frac { 1 } { 3 } } \sum _ { k = 1 } ^ { 3 } h _ { 2 } ( e _ { k } )$ subject to $\textstyle \sum _ { k = 1 } ^ { 3 } w _ { q } ( k ) e _ { k } \leq 0 . 1 5$ for every included task. The numerical optima are

<table><tr><td>Required tasks</td><td> $\mathrm { O p t i m a l } \left( e _ { 1 } , e _ { 2 } , e _ { 3 } \right)$ </td><td> $R _ { X }$  (bits per sample)</td></tr><tr><td>First</td><td>(0.0398, 0.1458, 0.4125)</td><td>0.394</td></tr><tr><td>First two</td><td>(0.1210, 0.0423, 0.1996)</td><td>0.498</td></tr><tr><td>All three</td><td>(0.1,0.1,0.1)</td><td>0.531</td></tr></table>

Cyclic permutation gives the same rates for every singleton or pair. With all three rows, summing the constraints bounds mean error by 0.1, attained by the constant allocation. The three normalized profiles average to 1, which lies on no pair’s segment. For an encoder observing only $Y = Z$ , the error probability e is independent of K and each task’s mean error weight is 0.5. Thus $0 . 5 e \leq 0 . 0 5$ gives $R _ { Y } = 1 - h _ { 2 } ( 0 . 1 ) \approx 0 . 5 3 1$ for every nonempty task subset. Every task’s uncompressed Bayes prediction remains Z.

## B Empirical methods and supporting evidence

## B.1 Finite-model numerical comparisons

We separately minimize mutual information over the full product prediction alphabet and over commonprediction error allocations. Sixteen cases include cyclic subsets, unequal normalized allowances, nonuniform confidence probabilities and families with up to four tasks. The two minima difer by at most $4 . 8 4 \times 1 0 ^ { - 8 }$ bits. We also test the whole-family and active-family convex-hull conditions separately.

## B.2 Three-class operational coding

The law in equation 20 has five distinct conditional-cost rows because the confidence states at $Z = 2$ have identical costs. Each codebook starts from 8,192 distinct ternary words of length 16. Three Lloyd updates assign 131,072 training blocks by total conditional loss and replace each word position with its minimumloss class. We use 16,384 development blocks for the first fit and 32,768 for each replication, and fix the additional-task tolerance at 0.499 before independent confirmation.

Known-loss encoding minimizes the sum of additional conditional losses. Fitted-loss encoding estimates five rows from $2 ^ { 2 0 }$ independent observations with noisy targets sampled per codebook. For reduced-observation encoding, we pool these estimates by class. We round fitted costs to multiples of $2 ^ { - 1 6 }$ for integer selection scores, with per-entry error at most $2 ^ { - 1 7 }$ . Each confirmation packet contains $2 ^ { 2 1 }$ 13-bit indices, padding, and a 48-byte header, totaling 3,407,920 bytes for 33,554,432 source symbols. The shared codebook and packet sufice for decoding.

We assess the codes by averaging conditional target losses within independent fresh blocks. For N blocks and loss range $b ,$ the one-sided Hoefding allowance is $b \sqrt { \ln ( 2 2 / 0 . 0 1 ) / ( 2 N ) }$ . A union bound gives simultaneous

99% coverage for 22 component means across the known and fitted full selectors, fitted reduced selectors, and two controls. The original loss range is one, and the additional range follows from the source table. The largest full-observation upper risk bounds are 0.25248 (original task) and 0.49846 (additional task).

The original-task control minimizes class errors. The confidence-removed control minimizes additional loss averaged over $K ,$ , retaining the marginal target law and unique Bayes prediction. Each uses one fixed codebook and 32,768 paired blocks. Full and reduced encoders produce identical packets. The confidenceremoved control has additional risk 0.51689, above the 0.499 tolerance.

## B.3 Binary learned block coding

## B.3.1 Source laws, fitting, and messages

Decision bits and confidence states are independent and fair. The first task has noise probabilities (0.05, 0.45) and tolerated risk 0.30. The additional task is a duplicate or has reversed noise probabilities, with tolerance 0.46 (loose) or 0.30 (tight). Confidence is observed through independent bit corruption at probability $\tau \in$ $\{ 0 , 1 / 8 , 1 / 2 \}$ . The first-task conditional error costs become $( 0 . 9 - 0 . 8 \tau , 0 . 1 + 0 . 8 \tau )$ . Three-state profiles are the rows of Eq. 43.

We use 4–10 index bits for length 12 and 5–13 for length 16. Eight weighted Lloyd rounds start from distinct random words, assigning 65,536 fresh blocks per round and updating bits by weighted majority. Fitting costs are uniform, original-task costs, or the first-two-profile average for three states. We compare both observations for each frozen codebook and select across fitting origins. All decoders return the same word.

Reference encoding minimizes known conditional loss. Fitted encoding learns two or three positive state scores, normalized to the same total cost, from soft known-cost assignments on a separate 64-word, length-12 codebook, using 250 batches of 512 independent blocks. Selection is exhaustive. Reduced-observation encoding uses constant scores and independent dummy states.

Other second-task limits are 0.34, 0.38, and 0.42. Corrupted-state comparisons retain the single and tightpair requirements. We set each task’s tolerance to 0.30 in the three-state single, first-two, and cyclic families.

We generate candidate assignments from scalarized profiles and minimize mean validation payload rate over their nonnegative frequencies, with each component risk 0.0015 below its tolerance. We convert frequencies to integer block counts in a shufled, source-independent schedule shared with the decoder. For candidate $c ,$ we transmit an eight-byte header and packed $b _ { c } { - } \mathrm { b i t }$ indices for $N _ { c } > 0$ blocks. The complete rate is

$$
r = \frac { 8 } { n \sum _ { c } N _ { c } } \sum _ { c } \left( 8 + \left\lceil \frac { N _ { c } b _ { c } } { 8 } \right\rceil \right) \quad \mathrm { b i t s ~ p e r ~ d e c i s i o n } .\tag{44}
$$

Codebooks and the source-independent schedule are shared with the decoder.

## B.3.2 Assessment and score precision

Each of three fits uses 65,536 validation and 131,072 assessment blocks, separate from fitting and calibration. We use independent-block standard errors and normal multipliers 1.96 for one requirement and 2.394 for two or three to form intervals with approximate simultaneous 95% coverage conditional on fitting and selection. The rate-equivalence margin is $1 / 6 4$ bit per decision.

Fitted-score diferences of order $1 0 ^ { - 7 }$ alter assignments at exact known-cost ties. We round fitted scores to five decimal places, retain parameters and codebooks, and use fresh validation, assessment, and forecast samples.

## B.3.3 Finite-code sensitivity and fixed-codebook comparisons

The single-task saving persists at both block lengths (Figure 7). Single-task, duplicate, and loose-pair rates overlap. Corrupting confidence reduces the saving until it vanishes at $\tau = 1 / 2 ,$ while tight-pair rates remain equivalent at all three corruption levels. With exact confidence, single-task coding concentrates errors in

(c) Three confidence states

Fitted, with confidence

![](images/2e16b5ee7dc410db0ece509e09b9fd9a7f1ecd2b98101becdc20dec56d3eac8a.jpg)

![](images/d43c9aff6a6bcadbf7f009570ac313f9c4601688a83e0276f4aee2cdf7ae4dc4.jpg)  
Asymptotic optimum

![](images/c08f9138abd2456f3db9ea43a2ef4672780e18d2a7f5fbe9334495e9a29992b4.jpg)  
Figure 7: Finite-code sensitivity. Filled circles show three fits per condition and hollow diamonds the unrestricted asymptotic rates. (a) Single-task rate above its asymptotic minimum at lengths 12 and 16. (b) Length-16 rates as the second-task limit varies, with the first fixed at 0.30. (c) Three-state families at length 16. Horizontal ofsets separate fits. We compute rates from index, header and padding bits.

the state with lower conditional error cost. When both complementary requirements are tight, the two conditional error probabilities are nearly equal.

As three-state profiles are added, the full-observation rate increases. For the cyclic family, the full- and reduced-observation rates are equivalent, while the optimal uncompressed predictions remain unchanged. Across both lengths and all conditions, fitted and reference rates difer by at most 0.000448 bits per decision. For all 252 selected policies, the joint upper risk bounds are at or below their respective task tolerances.

On fixed length-16 codebooks with eight-bit indices, observing confidence reduces single-task risk by 0.03779– 0.03802, with block standard error 0.000105. These codebooks coincide across fitting origins. With ten-bit indices, full-observation risks span 0.272730–0.272888 across both fitting origins, compared with 0.308054– 0.308321 without confidence. The advantage therefore persists on codebooks fitted for either observation. For the constant-score complementary candidate, predictions and component risks agree exactly across observations on each codebook.

## B.3.4 Additional-task risk forecasts

For a fixed binary code with conditional error probabilities $e ( k )$ , another known task has risk

$$
L _ { q } = p _ { q } + \sum _ { k } \operatorname* { P r } ( K = k ) w _ { q } ( k ) e ( k ) .\tag{45}
$$

We substitute conditional error probabilities estimated on 131,072 calibration blocks into equation 45 to forecast additional-task risks for each frozen single-task code without refitting. Across 252 forecasts, the largest absolute component-risk error on independent assessment is 0.001128. Forecast and assessment classifications agree for both the mean-threshold and joint-interval rules, including the two uncertain cases at tolerance 0.42. The length-16 codes have complementary-task risk of about 0.420, below the loose tolerance and above the tight tolerance, with unchanged messages. Stricter requirements can therefore require another code even though the uncompressed Bayes predictions agree.

## B.4 ImageNet task requirements

ENTITY-30 contains 307,828 training images across 240 fine classes (Santurkar et al., 2021). We split oficial validation data into 6,000 validation and 6,000 test images, each with 25 per fine class. Preprocessing uses a 232-pixel resize, 224-pixel center crop and channel normalization. We normalize the selected 240 logits into fine probabilities, then sum them within each superclass for coarse probabilities.

The $5 1 2 \times 2 8 \times 2 8$ input tensor is normalized using 4,096 training images. Analysis widths are 512, 192, 64. We pad the spatial dimensions to $3 2 \times 3 2$ , then compute a 64 $\times 8 \times 8$ latent and $3 2 \times 2 \times 2$ hyperlatent with the analysis transforms. Synthesis restores the input dimensions. We compute complete rates from the bits used for both entropy-coded latents and 40 framing bytes per image.

We fit the common initializer by minimizing normalized feature MSE plus 0.003 times estimated bpp with Adam at $1 0 ^ { - 3 }$ . Task fits use fresh Adam optimizers at $1 0 ^ { - 4 }$ . Each stage uses 5,000 updates, batches of 128 and 639,976 image exposures, with matched image order across objectives. Across both objectives, we retain all eight final checkpoints from fits using seeds 17 and 23 and rate multipliers 0.03 and 0.3. Coarse KL and fine cross entropy are normalized by ln 30 and ln 240. Neither objective includes feature MSE.

Adapted readouts use 2,048-dimensional penultimate sufix features, a 512-unit hidden layer and separate coarse and fine outputs. We fit two readouts with diferent seeds on the same 23,040 training images, 96 per fine class, for 20 AdamW epochs at learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ and batch size 256. On validation, we select checkpoints by summed normalized cross entropy and each task’s route by comparing inherited accuracy with the two-fit adapted mean, for raw and coded inputs alike. We average rates and accuracies over the two codec fits per objective and multiplier. We test the union of validation selections from Section 6.1.

We calculate intervals from 10,000 common paired resamples within fine classes, conditional on fitted models and classes. One-sided 97.5% upper limits give approximate 95% Bonferroni coverage for each group’s coarseand fine-accuracy drops. Validation sensitivity comparisons use fine allowances of three and ten points.

The selected added-fine code’s simultaneous upper drop limits are 2.575 coarse and 4.367 fine points, below the three- and five-point allowances.

## B.5 CIFAR-100

## B.5.1 Fitting and complete rates

We reserve 250 training images per coarse class for validation, leaving 45,000 for fitting. The 10,000 test images are excluded from fitting. Earlier test results had been inspected. For the extension in Section 6.2, we select candidates on validation. We exclude fine labels from data splitting and teacher selection.

The three ResNet-18 teachers use a $3 \times 3 ,$ , stride-one input convolution, no max pooling, independent initializations and 200 epochs with padded random crops and reflections. We select checkpoints by validation coarse accuracy, breaking ties by cross entropy.

Early-observation and logit-observation codecs use strided convolutional and MLP analysis, respectively, with 64 $\times 4 \times 4$ latents, $3 2 \times 1 \times 1$ hyperlatents and common transposed-convolution synthesis. Reconstructions are unconstrained. A mean-and-scale hyperprior models conditional Gaussian main latents and factorized hyperlatents without autoregressive context (Ballé et al., 2018; Minnen et al., 2018; Bégaint et al., 2020).

Preservation MSEs are normalized by training constant-predictor errors. For codec fitting, we use fine labels only in the family-supervised control, averaging coarse and fine cross entropies normalized by training-label entropies. Fine predictions are computed by a frozen early-feature MLP. We train randomly initialized codecs for 60 epochs. We fit paired preservation codecs for 60 epochs from a shared codec pretrained for 60 epochs with early-feature MSE. The earlier three-fit Teacher A groups and six reused controlled fits use final epoch-60 checkpoints. For single-fit controls, the reused warm control and new extension fits, we select checkpoints minimizing validation distortion plus β times estimated bpp.

Fine readouts are linear classifiers or MLPs with two 512-unit hidden layers on tensors pooled to 4×4 or on 20 logits. We normalize coordinates using training data, fit for 100 epochs and select checkpoints by validation accuracy. Training inputs are deterministically quantized, and assessment inputs are entropy-decoded. We compute rates from both entropy-coded bitstream lengths and the 40-byte header (0.3125 bpp).

## B.5.2 Preservation selection and test assessment

Each paired candidate uses two codecs per teacher and two MLP readouts per codec and input type, with matched examples and fitting rules. The earlier reconstructed-logit diagnostic uses a fixed raw-logit classifier, refitted in the extension.

![](images/b644b0b9d13804d24755c78d0db2a9e16dd177445bc84905cae663407bea6583.jpg)  
Figure 8: CIFAR validation candidates by teacher. Dashed lines mark the coarse and fine allowances. Open circles show codec fits, diamonds paired means, and stars the six selections. Single-fit controls have no paired mean. Fine accuracies average two refitted tensor readouts.

Table 1: Uncompressed CIFAR-100 reference accuracies (%). Fine-readout means and ranges summarize two fits. We use the early-tensor mean as the fine-accuracy reference.
<table><tr><td colspan="3"></td><td colspan="2">Fine, early tensor</td><td colspan="2">Fine, logits</td></tr><tr><td>Split</td><td>Teacher</td><td>Coarse</td><td>Mean</td><td>Range</td><td>Mean</td><td>Range</td></tr><tr><td rowspan="3">Validation</td><td>A</td><td>85.68</td><td>63.370</td><td>[63.36, 63.38]</td><td>45.970</td><td>[45.84, 46.10]</td></tr><tr><td>B</td><td>85.86</td><td>62.810</td><td>[62.40, 63.22]</td><td>46.410</td><td>[46.38, 46.44]</td></tr><tr><td>C</td><td>86.16</td><td>62.430</td><td>[62.30, 62.56]</td><td>45.390</td><td>[45.38, 45.40]</td></tr><tr><td rowspan="3">Test</td><td>A</td><td>85.38</td><td>62.615</td><td>[62.43, 62.80]</td><td>44.485</td><td>[44.28, 44.69]</td></tr><tr><td>B</td><td>85.27</td><td>62.480</td><td>[62.46, 62.50]</td><td>45.060</td><td>[44.65, 45.47]</td></tr><tr><td>C</td><td>85.55</td><td>63.060</td><td>[63.05, 63.07]</td><td>43.765</td><td>[43.67, 43.86]</td></tr></table>

We require an additional matched readout or codec pair when validation ranges across readout fits or codec means exceed one point and could afect selection. For candidates that could afect selection, readout ranges are at most 0.88 point.

We compare 14 paired groups and six single-fit controls (34 codecs) on validation. For each teacher and loss, we select the paired candidate minimizing mean rate subject to both mean accuracy-drop allowances, using two refitted tensor readouts.

We test the six selected groups and three feature-preserving candidates at the next lower rates, fixed using validation data (18 codecs, Fig. 5). References are in Table 1. We calculate paired 95% intervals from 2,000 common bootstrap resamples within fine classes, averaging per-image correctness across fits, conditional on teachers, codecs and readouts. For an approximate 95% Bonferroni joint assessment of each group’s mean risks, both upper endpoints must be at or below their respective allowances.

The selected feature-preserving group for Teacher C and the feature-preserving candidates with the next lower rates for teachers A and C fail the joint assessment, with coarse-drop intervals [2.310, 3.135], [2.470, 3.365] and [2.805, 3.705] points. Teacher C’s selected group and the candidate with the next lower rate contain one and two failed codec fits, respectively. Other assessed mean accuracy drops are within the allowances.

Intervals for the selected logit-minus-feature fine-accuracy diferences for teachers A, B and C are [−0.640, 0.280], [−0.835, 0.055] and [0.050, 0.928] points, within the one-point equivalence margin. Corresponding rate reductions are 26.35%, 36.38% and 25.73%.

## B.5.3 Observation and initialization controls

The earlier Teacher A comparison uses three codec fits and one fine-readout initialization. Randomly initialized early-observation and logit-observation codecs have test fine accuracies of 61.43% and 41.22% at 2.770 and 0.724 bpp with refitted readouts. Their diferent analysis architectures prevent isolating the observation cost. We fitted the early-observation logit-loss codec without early-feature MSE pretraining.

Earlier paired preservation fits have 59.88% fine accuracy at 2.750 bpp with early-feature loss and 60.14% at 1.999 bpp with logit loss. Logit-loss fits have 3.4812 times the early-feature MSE of the early-feature-loss fits. At a separate near-rate logit-loss setting, refitting improves fine accuracy from 40.89% to 60.27%.

Another randomly initialized logit-loss fit has validation coarse accuracy of 66.00%, compared with the 85.68% reference. Fine-supervised controls have lower coarse accuracy than the focal settings, so we cannot infer an advantage at matched coarse risk.

## B.6 Taskonomy

## B.6.1 Data, losses, and fitting

We use 256 × 256 images, with 24,000 training images balanced across 24 buildings, 500 validation images from five other buildings, and 2,000 test images balanced across five further buildings. All candidates use convolutional hyperprior codecs and width-48 residual upsampling decoders.

Depth loss is unit-transition Smooth-L1 between log predictions and log targets, with depths clamped to at least $1 0 ^ { - 4 }$ before taking logarithms. Semantic targets are Taskonomy’s FCIS pseudo-labels.<sup>1</sup> We use 17 labels including background and exclude uncertain pixels. Cross-entropy weights are inverse square roots of class frequencies in 2,400 balanced training images. We divide each image’s depth loss by its valid-pixel count and semantic loss by its valid-pixel weight sum, then average across images. During training, we normalize over valid pixels within each batch. We compute object mIoU over non-background classes present in the ground truth. Edge targets are clipped, rescaled Sobel magnitudes of Gaussian-smoothed grayscale images derived from RGB. Edge and RGB losses are MSE.

We fit each codec for 24,000 updates with batches of 12 images, minimizing estimated bpp times a multiplier plus task losses weighted by 10, 1, 10, and 100 for depth, semantics, edges, and RGB. After freezing codecs, we fit decoder sets with equal exposure and select checkpoints by weighted-DSE validation loss. For availabledecoder results, we select each task’s original or retained newly fitted width-48 decoder on validation. For matched-decoder results, we use the common weighted-sum checkpoint. We fix decoders before testing and use one message per decoding procedure.

Absolute requirements are fixed from validation reference and constant-predictor losses,

$$
D _ { q } ( \alpha ) = L _ { q , \mathrm { r e f } } + \alpha ( L _ { q , \mathrm { c o n s t } } - L _ { q , \mathrm { r e f } } ) , \qquad \alpha \in \{ 0 . 1 , 0 . 2 5 , 0 . 5 \} .\tag{46}
$$

We fit the uncompressed reference with the same architecture families and exposure, without quantization or rate penalty. The constant predictions are geometric-mean depth, weighted smoothed class probabilities, and mean edge intensity, computed from training data. For constant depth RMSE, we use arithmetic-mean depth. Additional mIoU floors are (1 − α) times validation reference mIoU, giving 0.16573044, 0.13810870, and 0.09207246, with cross-entropy requirements retained.

## B.6.2 Candidate selection and fit variation

We fit one codec and its task decoders for each of twelve settings. We select D, 30, DS, 3, and DE, 3 for three-fit replication by mean validation rates and risks. We compute pointwise 95% intervals from 2,000 paired image resamples within the five test buildings, conditional on buildings and fitted models. Matched newly fitted decoders give the same descriptive test minima as the available-decoder rule in Table 2.

Table 2: Complete Taskonomy test pool with task decoders selected on validation. We compute rates from complete message lengths and exclude background from object mIoU. Depth RMSE is in metres.
<table><tr><td>Family</td><td>Multiplier</td><td>Rate (bpp)</td><td>Depth  $L _ { D }$ </td><td>Semantics  $L _ { S }$ </td><td>Edges  $L _ { E }$ </td><td>Object mIoU</td><td>Depth RMSE (m)</td></tr><tr><td>D</td><td>0.3</td><td>0.07096</td><td>0.07669</td><td>0.84068</td><td>0.02827</td><td>0.0444</td><td>1.251</td></tr><tr><td></td><td>3</td><td>0.01752</td><td>0.07551</td><td>0.92659</td><td>0.03642</td><td>0.0284</td><td>1.223</td></tr><tr><td></td><td>30</td><td>0.00718</td><td>0.08833</td><td>1.03715</td><td>0.04183</td><td>0.0082</td><td>1.350</td></tr><tr><td>DS</td><td>0.3</td><td>0.12416</td><td>0.07607</td><td>0.66453</td><td>0.02421</td><td>0.1303</td><td>1.228</td></tr><tr><td></td><td>3</td><td>0.03346</td><td>0.07688</td><td>0.66444</td><td>0.03263</td><td>0.1345</td><td>1.257</td></tr><tr><td>DE</td><td>0.3</td><td>0.13277</td><td>0.07419</td><td>0.78960</td><td>0.00299</td><td>0.0591</td><td>1.201</td></tr><tr><td></td><td>1</td><td>0.06386</td><td>0.07250</td><td>0.82183</td><td>0.00402</td><td>0.0485</td><td>1.195</td></tr><tr><td></td><td>3</td><td>0.03142</td><td>0.07703</td><td>0.87632</td><td>0.00758</td><td>0.0367</td><td>1.260</td></tr><tr><td>DSE</td><td>0.3</td><td>0.19188</td><td>0.07777</td><td>0.62568</td><td>0.00425</td><td>0.1384</td><td>1.248</td></tr><tr><td></td><td>1</td><td>0.09410</td><td>0.07364</td><td>0.62297</td><td>0.00543</td><td>0.1441</td><td>1.208</td></tr><tr><td></td><td>3</td><td>0.05095</td><td>0.07676</td><td>0.65327</td><td>0.00770</td><td>0.1204</td><td>1.265</td></tr><tr><td>RGB</td><td>0.3</td><td>0.18272</td><td>0.07755</td><td>0.67093</td><td>0.00055</td><td>0.1262</td><td>1.211</td></tr><tr><td>Reference</td><td></td><td></td><td>0.07375</td><td>0.62329</td><td>0.00189</td><td>0.1361</td><td>1.184</td></tr></table>

Table 3: Taskonomy test means [ranges] over three fits with available-decoder selection. All fits satisfy the moderate depth requirement. The last row counts fits satisfying moderate DS, DE, and DSE requirements.
<table><tr><td>Metric</td><td></td><td>D,30</td><td></td><td>DS,3</td><td>DE,3</td></tr><tr><td>Rate (bpp)</td><td>0.00725 [0.00718, 0.00736]</td><td></td><td>0.03199 [0.03077, 0.03346]</td><td></td><td>0.02808 [0.02429, 0.03142]</td></tr><tr><td>Depth  $L _ { D }$ </td><td>0.08963 [0.08833, 0.09099]</td><td></td><td>0.07598 [0.07543, 0.07688]</td><td></td><td>0.07706 [0.07675, 0.07740]</td></tr><tr><td>Semantics  $L _ { S }$ </td><td>1.05059 [1.03715, 1.05881]</td><td></td><td>0.64969 [0.63882, 0.66444]</td><td></td><td>0.88166 [0.87632, 0.88692]</td></tr><tr><td>Edges  $L _ { E }$ </td><td>0.04171 [0.04155, 0.04183]</td><td></td><td>0.03232 [0.03180, 0.03263]</td><td></td><td>0.01054 [0.00758, 0.01439]</td></tr><tr><td>Object mIoU</td><td>0.00864 [0.00819, 0.00917]</td><td></td><td>0.13315 [0.12376, 0.14116]</td><td></td><td>0.03844 [0.03589, 0.04273]</td></tr><tr><td>Depth RMSE (m)</td><td></td><td>1.355 [1.350, 1.364]</td><td>1.255 [1.251, 1.257]</td><td></td><td>1.271 [1.260, 1.282]</td></tr><tr><td colspan="5">Fits: DS/DE/DSE 0/0/0 3/0/0</td><td>2/2/1</td></tr></table>

With the mIoU floor, we select DS, 3 and DSE, 3 on validation for loose DS and DSE requirements. No candidate satisfies stringent test requirements including semantics. Both selected codes and the uncompressed reference have test object mIoU below the moderate floor.

In the hardest building, all fits exceed the moderate depth-loss tolerance. For every building and fit, DS has lower semantic and higher edge risk than DE. We double newly fitted decoder width from 48 to 96 for the three base fits, keeping fitting exposure and selection unchanged. Primary validation and test feasibility decisions are unchanged.

## B.6.3 Input-independent mixtures

We select frequencies $a _ { c }$ for a shared image-independent schedule by minimizing validation rate subject to the required task risks

$$
\operatorname* { m i n } _ { a } \sum _ { c } a _ { c } r _ { c , \mathrm { v a l } } , \qquad a _ { c } \ge 0 , \quad \sum _ { c } a _ { c } = 1 , \qquad \sum _ { c } a _ { c } L _ { c q , \mathrm { v a l } } \le D _ { q } \quad \mathrm { f o r ~ e a c h ~ r e q u i r e d ~ } q .\tag{47}
$$

We use the twelve original candidates and validation-selected readouts, then apply weights unchanged on test. The encoder and decoder share the schedule, so no per-image codec identifier need be transmitted.

Edge predictions  
Table 4: Validation-selected codes and image-independent mixtures. We count all transmitted bits. We select D, 30 for depth only at every tolerance. Its test risk exceeds only the stringent tolerance.
<table><tr><td rowspan="2">α</td><td rowspan="2">Required tasks</td><td rowspan="2">Single code</td><td colspan="3">Validation</td><td colspan="2">Mixture test</td></tr><tr><td>Single rate</td><td>Mixture rate</td><td>Reduction</td><td>Rate</td><td>Unsatisfied requirement</td></tr><tr><td rowspan="3">0.10</td><td>DS</td><td>DS,3</td><td>0.03376</td><td>0.03123</td><td>7.50%</td><td>0.03095</td><td>None</td></tr><tr><td>DE</td><td>DE,1</td><td>0.06241</td><td>0.03791</td><td>39.26%</td><td>0.03837</td><td>Edge</td></tr><tr><td>DSE</td><td>DSE,1</td><td>0.09171</td><td>0.04897</td><td>46.60%</td><td>0.04991</td><td>Edge</td></tr><tr><td rowspan="3">0.25</td><td>DS</td><td>DS,3</td><td>0.03376</td><td>0.02459</td><td>27.18%</td><td>0.02436</td><td>None</td></tr><tr><td>DE</td><td>DE,3</td><td>0.03123</td><td>0.02692</td><td>13.81%</td><td>0.02706</td><td>Edge</td></tr><tr><td>DSE</td><td>DSE,3</td><td>0.04994</td><td>0.03199</td><td>35.95%</td><td>0.03206</td><td>Edge</td></tr><tr><td rowspan="2">0.50</td><td>DS</td><td>D,3</td><td>0.01745</td><td>0.01351</td><td>22.56%</td><td>0.01338</td><td>None</td></tr><tr><td>DE,DSE</td><td>DE,3</td><td>0.03123</td><td>0.01889</td><td>39.50%</td><td>0.01895</td><td>Edge</td></tr></table>

![](images/e771586f628cf11866dd1e3b972e9e2ea5f5a1a83c9e11f86341d6c8dca37083.jpg)  
<sup>bed microwave refrigeratorchair bed microwave refrigerator</sup>Figure 9: Semantic and edge predictions from five decoded validation images. RGB references and class colors are identical across methods. Gray target pixels are excluded from loss. Edge intensity scale is [0, 1].

Five of twelve mixtures and eleven of twelve single-code selections satisfy their test requirements. For all six mixtures with an edge requirement, the test edge risks exceed the tolerance. The three DS mixtures and the depth-only choices at moderate and loose tolerances satisfy their test requirements.

For the moderate DS mixture, we use D, 30 with frequency approximately 0.346 and DS, 3 with frequency approximately 0.654. Test depth and semantic risks are (0.080843, 0.793438).

## B.6.4 Semantic class support

Validation contains fourteen object classes. Test additionally contains book, while toaster is absent from both. Background occupies 92.7103% and 93.5162% of valid validation and test pixels. We compute original mIoU over object classes present in each split.

Restricting test mIoU retrospectively to the fourteen validation-present classes raises reference mIoU from 0.136058 to 0.143284 and mean DS, 3 mIoU from 0.133147 to 0.141463, both above the unchanged moderate floor 0.13810870. Two of three DS fits have mIoU at or above this floor. The DSE, 3 base fit has mIoU 0.127146, below the floor. Test reference mIoU remains below the validation reference value of 0.184145.

## B.6.5 Complete message lengths and decoded examples

Complete messages contain entropy-coded latents and a 40-byte header (67.32% of the depth-only message length). We compute rates from these lengths, excluding model storage and decoder computation.

We apply base D, 30, DS, 3, and DE, 3 codes to the first validation image per building in Fig. 9.

## B.7 Feature preservation and visual decoding

We use ImageNet ResNet-50 IMAGENET1K\_V2 features. Block 13 ends Stage 3 with a 1024 × 14 × 14 tensor. Blocks 14–16 form Stage 4, each with a 2048 × 7 × 7 tensor. The unchanged sufix reproduces the original ImageNet logits. At each depth, we input the same complete tensor to the inverse and native-feature classifier.

Photographs and species prediction. We square-crop CUB photographs around the annotated bird box, with side length ⌈1.2 max(w, h)⌉, mid-gray boundary padding and bilinear resizing to 224 × 224 pixels (Wah et al., 2011). We fit readers on 4,794 photographs from 200 species, validating on six per species. We evaluate species-pair accuracy on 30 oficial test photographs per species by restricting the fitted 200-class output to Indigo Bunting and Blue Grosbeak. Both displayed photographs belong to development.

Native readers use the pretrained fourth stage at block 13 and final residual block at blocks 14–16, followed by global pooling and a new 200-class linear output. All reader parameters are fitted with feature extraction fixed. For RGB readers, we adapt a pretrained ResNet-50 to each inverse’s reconstructions, using horizontal reflections. We fit two readers per observation for 50 AdamW epochs with class-weighted cross entropy, learning rate and weight decay 10<sup>−4</sup>, and batch size 64. We select checkpoints by validation macro accuracy.

Visual inversion. Convolutional inverses use 5 × 5 kernels, 128 hidden channels, ReLUs, linear RGB outputs, and four upsampling layers at block 13 or five at blocks 14–16. We minimize RGB MSE for 100,000 Adam updates at learning rate 10<sup>−4</sup>, clipping gradient norms at one. We sample 64-image ImageNet training batches with replacement, shared across depths, and select checkpoints by MSE on 5,000 fixed ImageNet validation photographs. Pixels are clipped and rounded for eight-bit export. Using CUB part annotations, we locate enlarged 96 × 48 wing regions at identical coordinates across reconstructions.

Table 5: Species-pair accuracy (%) for each of the two fitted readers.
<table><tr><td>Observation</td><td>Block 13</td><td>Block 14</td><td>Block 15</td><td>Block 16</td></tr><tr><td>Native feature, fit 1</td><td>96.7</td><td>95.0</td><td>93.3</td><td>93.3</td></tr><tr><td>Native feature, fit 2</td><td>96.7</td><td>96.7</td><td>93.3</td><td>93.3</td></tr><tr><td>Reconstructed RGB, fit 1</td><td>91.7</td><td>86.7</td><td>88.3</td><td>83.3</td></tr><tr><td>Reconstructed RGB, fit 2</td><td>88.3</td><td>85.0</td><td>86.7</td><td>81.7</td></tr></table>