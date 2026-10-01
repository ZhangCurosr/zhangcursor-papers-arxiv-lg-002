# LEARNING BEYOND FULL IMITATION: TASK-PRESERVING KNOWLEDGE DISTILLATION

Qianfeng Yuan Wenbing Tao

Huazhong University of Science and Technology

## ABSTRACT

Knowledge distillation transfers knowledge by encouraging a student to match a teacher’s predicted class probabilities. These probabilities express not only confidence in the correct class, but also relations among incorrect alternatives. Yet closer imitation does not necessarily yield a better student. Our key observation is that a student may already distinguish the correct class more sharply than its teacher, so further imitation can require giving back discrimination it has acquired. Our main result is an exact separation between full imitation and conditional learning. When the correct class’s score advantage over each alternative must be preserved, full teacher-to-student KL minimization is blocked exactly when the student assigns no more probability than the teacher to every incorrect class. Crucially, the teacher’s relative probabilities among incorrect classes remain fully learnable. We characterize the exact price of this transfer: a minimum increase in correct-class log-odds that compensates for the largest conditional-probability mismatch. Label fitting and conditional matching can therefore be completed even as full teacher KL diverges. This separation motivates task-preserving knowledge distillation (TPKD), which keeps the label gradient intact and minimally corrects the conditional gradient so that its output update preserves the label step’s gains against every incorrect alternative. The corrected conditional direction retains more than half of the original first-order conditional descent at the same step size, with a tight bound. For a fixed positive conditional target and sufficiently small constant output steps, label and conditional errors vanish together. Experiments trace this learning from exact head updates to ordinary network training. TPKD reaches 88.05% accuracy on CIFAR-100 and 93.81% on CLINC150, improving over standard distillation by 0.47 and 0.35 percentage points across three seeds.

Keywords: knowledge distillation, task preservation, conditional knowledge transfer, constrained optimization, learning dynamics.

## 1 INTRODUCTION

Knowledge distillation (KD) transfers knowledge from a teacher to a student through the teacher’s predictive distribution (Hinton et al., 2015). Besides confidence in the correct class, this distribution expresses how plausible the incorrect alternatives are relative to one another. These relative probabilities can encode similarities and confusions absent from hard labels, making them an important source of transferable knowledge (Zhao et al., 2022).

Progress in KD has been driven largely by empirical advances in objectives and training procedures, while theoretical analyses remain comparatively limited (Phuong & Lampert, 2019; Menon et al., 2021; Harutyunyan et al., 2023). A closer imitator is not necessarily a better student (Cho & Hariharan, 2019; Stanton et al., 2021). Existing theory studies capacity, teacher quality, regularization, and data geometry or optimization bias. We study a complementary question: when further imitation conflicts with the student’s task progress, what can the teacher still teach?

Our key observation is that the student is not merely a passive copy of the teacher. Label learning may make it more discriminative on an example, assigning lower probability than the teacher to every incorrect class. Moving its full prediction closer to the teacher then requires giving back some of that discrimination. Yet the teacher may still describe a different relative ordering among the incorrect alternatives. Reproducing the teacher’s complete prediction and learning the structure within it are therefore different objectives.

This leads to the central question of this work: What can a student still learnfrom its teacher without sacrificing any ofits existing task discrimination, and at what cost?

Preserving all existing task discrimination requires more than keeping the answer correct. For example, a smaller score lead can make a previously accepted prediction require human confirmation (Liang et al., 2024). Since different confusions may require different acceptance thresholds, we preserve each correct-versus-incorrect score gap—its margin—while allowing the teacher to reshape relations among incorrect classes.

Our answer is an exact separation between full imitation and conditional learning. We prove that full imitation is blocked precisely when the student assigns no more probability than the teacher to every incorrect class. Its current prediction is then already the closest point to the teacher within the entire task-preserving region. Crucially, this does not mean that the teacher has nothing left to teach. The student can still recover the teacher’s entire conditional distribution over incorrect classes, not merely their ranking, without decreasing any correct-class margin.

Learning this structure has an exact compensation cost. Matching a different conditional distribution raises some incorrect alternatives relative to others. Preserving their margins requires a corresponding increase in correct-class confidence. The price is greater confidence, not weaker discrimination. We derive the minimum increase in correct-class log-odds needed for exact, task-preserving transfer: the largest teacher-to-student conditional-probability ratio determines the cost, and a continuous safe path attains it. This yields an important consequence: the student can approach perfect label fitting while exactly matching the teacher’s conditional structure, even as full teacher KL diverges. Learning the teacher’s knowledge need not mean reproducing its confidence.

This separation suggests how to keep learning. We propose Task-Preserving Knowledge Distillation (TPKD). Our main contributions are: Separation and transfer cost. To our knowledge, we provide the first joint characterization of exact full-KL blocking, complete conditional reachability, and minimum confidence compensation under per-competitor margin preservation. We prove a necessary-and-sufficient condition for blocking and derive the attainable minimum correct-class log-odds cost. A protected update with tight retention. TPKD keeps the label gradient intact and adds the nearest safe conditional direction. It preserves every margin gain of the label step from the same start and with the same step size. The correction retains over half the first-order conditional descent with a tight bound, and a nonzero, aligned conditional component whenever error remains. Joint learning and parameter-space guarantees. For fixed positive targets and sufficiently small constant output steps, label and conditional errors vanish together. For nonuniform targets, this differs from CE’s uniform conditional limit. Exact fixed-feature head updates and computable backpropagation conditions connect the output construction to network parameters.

Experiments test the mechanism and its use in full training. Across the evaluated visual CE states, blocking rises from 50.00% to 58.90%, with conditional knowledge still unlearned. All 64 exact-head batches gain additional conditional knowledge while retaining label-step margins; native updates carry this gain into the network and accumulate it over successive iterations. Full training reaches 88.05% on CIFAR-100 and 93.81% on CLINC150, exceeding standard KD by 0.47 and 0.35 percentage points across three seeds. Removing the CE gradient or replacing conditional learning with protected full imitation reduces accuracy in both domains, while removing projection gives closely matched performance.

## 2 RELATED WORK

Distillation targets. Classical KD matches softened teacher probabilities (Hinton et al., 2015), while DKD separates target-class and non-target supervision (Zhao et al., 2022). These approaches identify useful teacher signals; our analysis asks which signals remain attainable when existing task discrimination must be preserved.

Theoretical perspectives. KD theory addresses capacity mismatch (Mirzadeh et al., 2020), teacher quality and statistical supervision (Menon et al., 2021), self-distillation as regularization (Mobahi et al., 2020), and data geometry or optimization-induced deviations (Phuong & Lampert, 2019;

Nagarajan et al., 2023). We instead characterize what remains learnable, and at what cost, while preserving the student’s acquired class comparisons.

Preservation and gradient coordination. MPT reduces prediction regressions during model updates through margin calibration and dual-source distillation (Ricci et al., 2026). PCGrad projects conflicting gradients (Yu et al., 2020); DeepKD decouples momentum updates and filters non-target knowledge (Huang et al., 2025); DTO-KD balances task and distillation gradients (Hayder et al., 2026). Our analysis first characterizes blocking and transfer cost in the task-preserving output region. TPKD then keeps the complete label gradient and minimally corrects each example’s conditional increment, with tight bounds on retained learning.

## 3 WHAT TO LEARN AND WHAT TO PRESERVE

Learning targets. Labels specify the correct answer; the teacher also describes relations among alternatives. We distinguish label fitting from two teacher-learning objectives: reproducing the full prediction or learning only its conditional structure among incorrect classes.

Consider an example with label $y \in \{ 1 , . . . , K \} , K \geq 3$ . The student produces logits $z \in \mathbb { R } ^ { K }$ and probabilities $p = \operatorname { s o f t m a x } ( z )$ , where $\begin{array} { r } { p _ { j } = e ^ { z _ { j } } / \sum _ { k = 1 } ^ { K } e ^ { z _ { k } } } \end{array}$ ; the teacher supplies probabilities t. Both distributions are strictly positive. Define their correct-class confidences $\rho = p _ { y } , \tau = t _ { y } ,$ , and conditional probabilities $r _ { j } = p _ { j } / ( 1 - \rho ) , q _ { j } = t _ { j } / ( 1 - \tau ) \operatorname { f o r } j \neq y$ . The vectors $r , q$ each sum to one over incorrect classes. With coordinate $y$ first and the other class indices aligned, $p = ( \rho , ( 1 - \rho ) r )$ $t = ( \tau , ( 1 - \tau ) q )$ . For positive distributions $a , b ,$ let $\begin{array} { r } { \mathrm { K L } ( { a } \| { b } ) = \sum _ { i } \tilde { a _ { j } } \mathrm { \bar { l o g } } ( \tilde { a _ { j } } / b _ { j } ) ; \mathrm { \bar { k l } } _ { \mathrm { B e r } } ( \bar { \tau } \| \rho ) } \end{array}$ denotes the KL between $( \tau , 1 - \tau )$ and $( \rho , 1 - \rho )$ . Label fitting minimizes $L _ { \mathrm { C E } } ( p , y ) = - \log \rho .$ full imitation minimizes $L _ { T } ( p ) = \mathrm { K L } ( t \| p )$ , and conditional learning minimizes $L _ { \mathrm { c o n d } } ( p ) = \mathrm { K L } ( q | | r )$ The teacher loss decomposes as (Zhao et al., 2022)

$$
L _ { T } ( p ) = \mathrm { k l } _ { \mathrm { B e r } } ( \tau \| \rho ) + ( 1 - \tau ) L _ { \mathrm { c o n d } } ( p ) .\tag{1}
$$

Full imitation matches both teacher confidence and conditional structure. Increasing $\rho$ with $r$ fixed improves label fitting without changing these relations; label progress alone therefore does not measure conditional learning. We use natural logarithms and Euclidean inner products, norms and projections. Logit arguments mean evaluation at $p = \operatorname { s o f t m a x } ( z )$ , with $y .$ , t fixed.

Preserving learned comparisons. The student may change its prediction, but should not give back its acquired advantage over any competitor. For each $j \neq y ,$ , define the margin $M _ { j } ( p ) = \log ( p _ { y } / p _ { j } ) =$ $z _ { y } - z _ { j }$ . Let $\Delta _ { m } ^ { \circ } = \{ a \in \bar { \mathbb { R } } ^ { m } : \acute { a _ { j } } > 0 , \sum _ { i = 1 } ^ { m } a _ { j } = 1 \}$ be the positive probability simplex. From reference $p ,$ the task-preserving predictions are

$$
\begin{array} { r } { S ( p ) = \{ p ^ { \prime } \in \Delta _ { K } ^ { \circ } : M _ { j } ( p ^ { \prime } ) \geq M _ { j } ( p ) \mathrm { ~ f o r ~ a l l ~ } j \neq y \} . } \end{array}\tag{2}
$$

This protects individual comparisons as well as the label loss. For nonnegative competitor weights $\omega = ( \omega _ { j } ) _ { j \neq y }$ , define $\begin{array} { r } { L _ { \omega } ( z , y ) = \log ( 1 + \sum _ { i \neq u } \omega _ { j } e ^ { - M _ { j } ( z ) } ) } \end{array}$ . Preserving all margins is equivalent to not increasing any $L _ { \omega } \mathrm { : }$ unit weights give CE; a single unit weight isolates one competitor (Appendix $\mathrm { A . 1 } )$ . CE alone can improve while one comparison worsens; Appendix $_ { \mathrm { A } . 6 }$ gives its constrained imitation optimum.

How learning and protection interact. Since $p _ { j } = ( 1 - \rho ) r _ { j }$ , we have $M _ { j } ( p ) = \log [ \rho / ( 1 -$ $\rho ) ] - \log r _ { j }$ . Matching a different conditional target raises some $r _ { j }$ , shrinking its margin at fixed $\rho .$ Increasing correct-class confidence can compensate. Conditional learning leaves this confidence free; full imitation also seeks the teacher’s confidence. We next determine which goal remains attainable within $S ( p )$ , and at what minimum cost.

## 4 WHEN IMITATION STOPS, WHAT CAN THE TEACHER STILL TEACH?

## 4.1 WHY FULL IMITATION CAN BECOME BLOCKED

Our first result identifies when closer imitation must undo task progress. Once the student has suppressed every incorrect class at least as strongly as the teacher, its current prediction is already the best full imitation permitted by task preservation.

Theorem 1 (Full-imitation blocking). For positive p, t, the inequalities $p _ { j } \leq t _ { j }$ for all $j \neq y$ hold if and only ifp uniquely minimizes $L _ { T }$ over $S ( p )$ . For any $p ^ { \prime } \in \mathcal { S } ( p )$ , define $\Delta \bar { M _ { j } } = M _ { j } ( p ^ { \prime } ) - M _ { j } ( p )$ Under the blocking condition $p _ { j } \leq t _ { j }$ for all $j \neq y ,$

$$
L _ { T } ( p ^ { \prime } ) - L _ { T } ( p ) = \mathrm { K L } ( p \| p ^ { \prime } ) + \sum _ { j \neq y } ( t _ { j } - p _ { j } ) \Delta M _ { j } \geq \mathrm { K L } ( p \| p ^ { \prime } ) .\tag{3}
$$

The teacher asks for at least as much probability on each incorrect class, while protection permits only nonnegative margin gains. Every term on the right of Eq. (3) is therefore nonnegative: any change from p strictly increases full imitation error. Yet conditional knowledge can remain unlearned $( r \neq q )$ throughout a nonempty region of blocked states (Appendix A.2).

## 4.2 LEARNING THE REMAINING STRUCTURE, AND ITS PRICE

Blocking full imitation does not exhaust the teacher’s knowledge. The student can still match the teacher’s relative probabilities among incorrect classes. Raising a competitor’s conditional probability requires enough extra correct-class confidence to preserve its margin. The next result gives the minimum increase needed for complete transfer.

For a candidate confidence $a \in ( 0 , 1 )$ , let $p ^ { \prime } = ( a , ( 1 - a ) q )$ match the full conditional target. Define $\mathrm { l o g i t } ( a ) = \log [ a / ( 1 - a ) ]$ , the log-odds of the correct class, $\mathcal { R } _ { \infty } = \operatorname* { m a x } _ { j \neq y } q _ { j } / r _ { j }$ , and $D _ { \infty } ( q \| r ) = \log \mathcal { R } _ { \infty }$ . Let $a _ { \mathrm { m i n } }$ be the smallest confidence permitting task-preserving transfer.

Theorem 2 (Minimum compensation for exact transfer). The prediction $p ^ { \prime }$ belongs to $S ( p )$ if and only $i f$

$$
\mathrm { l o g i t } ( a ) - \mathrm { l o g i t } ( \rho ) \geq D _ { \infty } ( q \| r ) , \qquad a _ { \mathrm { m i n } } = \frac { \rho \mathcal { R } _ { \infty } } { 1 - \rho + \rho \mathcal { R } _ { \infty } } < 1 .\tag{4}
$$

A continuous path attains $a _ { \mathrm { m i n } }$ while every margin remains nondecreasing. $I f r \neq q ;$ , then $a _ { \mathrm { m i n } } > \rho$ and conditional KL decreases strictly along the path to zero.

The most underestimated competitor sets the price. To attain it, for $s \in [ 0 , 1 ]$ define $r ( s ) = ( 1 - s ) r +$ $s q ,$ choose $a ( s )$ by logit $a ( s ) \overset { \cdot } { = } \log \mathrm { i t } \rho + \log ( 1 - s + s \mathcal { R } _ { \infty } )$ , and set $\boldsymbol { \bar { p ( s ) } } \bar { = } ( a ( s ) , ( 1 - a ( s ) ) \boldsymbol { r ( s ) } )$ This moves the whole conditional distribution toward the teacher while compensating just enough at each point (Appendices A.3–A.4); Appendix A.7 gives optimal partial transfer at smaller budgets.

A concrete example. In Figure 1, $y \ = \ 1 , \ p \ = \ ( 0 . 9 0 , 0 . 0 7 , 0 . 0 3 )$ and $t = ( 0 . 4 0 , 0 . 2 5 , 0 . 3 5 )$ Both incorrect-class probabilities are below the teacher’s, so full imitation is blocked. Yet the student favors class 2 over class $3 \ : ( 7 { : } 3 )$ , whereas the teacher favors class 3 (5:7). At the transfer endpoint $p ^ { * } = ( a _ { \mathrm { m i n } } , ( 1 - a _ { \mathrm { m i n } } ) q ) , a _ { \mathrm { m i n } } = 3 5 / 3 7 \approx 0 . 9 4 6 \colon$ the class-3 margin stays fixed and the class-2 margin grows. The plot uses conditional coordinate $x = r _ { 3 }$ and confidence increment $\Delta = \mathrm { l o g i t } ( a ) - \mathrm { \bar { l o g i t } } ( \rho )$ . Appendix $\mathrm { A . 4 }$ works through the calculation.

![](images/e724f25e605bc40a10cece55b0eda6b8ea1e261ddb8090d631d0cce27b3680bd.jpg)

![](images/98dab6e7b0cff630419ff9a82a8f9e64c183a25eebd663538e088941bd678f47.jpg)  
Figure 1: Full imitation can worsen while conditional knowledge is learned. The safe path attains the minimum confidence $a _ { \mathrm { m i n } } \approx 0 . 9 4 6$ . Label CE and conditional KL fall while full KL rises; losses are normalized by their initial values.

The separation is strongest near perfect label fitting. For $\varepsilon \in ( 0 , 1 )$ ), define $p ^ { ( \varepsilon ) } = ( 1 - \varepsilon , \varepsilon q )$ whose conditional distribution is $r ^ { ( \varepsilon ) } = q .$ Then, $\mathsf { a s } \varepsilon \downarrow 0$

$$
{ \cal L } _ { \mathrm { C E } } ( p ^ { ( \varepsilon ) } , y ) \to 0 , \qquad \mathrm { K L } ( q \| r ^ { ( \varepsilon ) } ) = 0 , \qquad \mathrm { K L } ( t \| p ^ { ( \varepsilon ) } ) \to \infty .\tag{5}
$$

Learning the label and the teacher’s conditional structure therefore need not make the student a closer full imitator. Theorem 2 identifies the minimum total confidence compensation required to complete this transfer. We now turn to the local learning problem: how can each update acquire conditional knowledge while preserving all the progress of the corresponding label step? TPKD addresses this question by making the smallest necessary correction to the combined learning direction.

## 5 TASK-PRESERVING KNOWLEDGE DISTILLATION

## 5.1 KEEP LABEL LEARNING INTACT AND CORRECT THE TEACHER SIGNAL

The separation suggests a simple design: keep the progress of ordinary label learning and correct only the additional teacher signal that would interfere with it. The correction should be minimal, so protection does not unnecessarily discard teacher knowledge.

Let $e _ { j } \in \mathbb { R } ^ { K }$ have one in coordinate j and zeros elsewhere. The two logit gradients are $h =$ $\nabla _ { z } L _ { \mathrm { C E } } = p - e _ { y }$ and $u = \nabla _ { z } L _ { \mathrm { c o n d } } = ( 0 , r - q )$ , where the zero occupies coordinate y. Let $\mathbf { 1 } \in \mathbb { R } ^ { K }$ denote the all-ones vector. Conditional gradients lie in the subspace $\mathcal { L } _ { y }$ . Directions that can be subtracted without reducing any margin form the safe cone $\mathcal { K } _ { y } .$

$$
\mathcal { L } _ { y } = \{ \boldsymbol { x } \in \mathbb { R } ^ { K } : x _ { y } = 0 , \mathbf { 1 } ^ { \top } \boldsymbol { x } = 0 \} , \qquad \mathcal { K } _ { y } = \{ \boldsymbol { x } \in \mathbb { R } ^ { K } : x _ { j } - x _ { y } \geq 0 \mathrm { ~ f o r ~ a l l ~ } j \neq y \} .
$$

A nonzero direction in $\mathcal { L } _ { y }$ cannot belong to $\kappa _ { y } \mathrm { . }$ changing only the relative wrong-class scores must favor some competitor. Thus $\mathcal { L } _ { y } \cap \mathcal { K } _ { y } \overset { = } { = } \{ 0 \}$ . Write $\bar { \Pi } _ { \mathcal { K } _ { y } } { }$ for Euclidean projection onto $\mathcal { K } _ { y } ,$ let d be the corrected conditional direction and v the complete update direction. With output step size $\eta \geq 0$ TPKD uses

$$
\begin{array} { r } { d = \Pi _ { \mathcal { K } _ { y } } ( u ) , \qquad \Big | \boldsymbol { v } = \boldsymbol { h } + \frac { 1 } { 2 } \boldsymbol { d } , \qquad \boldsymbol { z } ^ { + } = \boldsymbol { z } - \eta \boldsymbol { v } . \Big | } \end{array}\tag{6}
$$

Equivalently, v is the nearest direction to $h + u / 2$ that preserves every label-step margin gain. The projection costs $O ( K \log K )$ per example (Appendix B.1).

The reference is the progress that the label step would have achieved on its own. Starting from the same logits and using the same step size, adding the corrected conditional signal preserves every margin gain of that label step:

$$
\begin{array} { r l } & { M _ { j } ( z - \eta v ) \geq M _ { j } ( z - \eta h ) \geq M _ { j } ( z ) , \qquad j \neq y , } \\ & { L _ { \mathrm { C E } } ( z - \eta v ) \leq L _ { \mathrm { C E } } ( z - \eta h ) \leq L _ { \mathrm { C E } } ( z ) . } \end{array}\tag{7}
$$

The same ordering holds for every weighted label loss $L _ { \omega }$ with $\omega \ge 0$ . Thus TPKD preserves not only the student’s existing discrimination, but also the additional discrimination that the corresponding CE step would have gained.

Training uses ordinary backpropagation. Let $\boldsymbol { \theta } \in \mathbb { R } ^ { P }$ collect the P network parameters; subscript i indexes the $N$ batch examples. Define the output Jacobian $J _ { i } = \partial z _ { i } / \partial \theta \in \mathbb { R } ^ { \dot { K } \times P }$ . Holding $v _ { i }$ fixed during differentiation with stopgrad, we inject it through the surrogate loss

$$
L _ { \mathrm { s u r } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left. \mathrm { s t o p g r a d } ( v _ { i } ) , z _ { i } \right. , \qquad \nabla _ { \theta } L _ { \mathrm { s u r } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } J _ { i } ^ { \top } v _ { i } .\tag{8}
$$

For teacher logits $z ^ { T }$ , let $z _ { \lnot y } ^ { T }$ denote the non-target coordinates. The fixed teacher supplies its original $q = \mathrm { s o f t m a x } ( z _ { \lnot y } ^ { T } )$ . Figure 2 summarizes the two learning signals and their combination.

## 5.2 PROTECTION RETAINS REAL CONDITIONAL LEARNING

What does the correction change? It caps the teacher’s strongest requests to raise competing classes and turns the clipped mass into correct-class compensation. Let $\widetilde { q }$ be the retained teacher masses and

![](images/2a25b0a689fc867023ab90588f429a22897975a9f8e5c5d233fd3862d37e373a.jpg)  
Figure 3: Protection preserves a genuine conditional-learning direction. The corrected direction combines confidence compensation with a pure conditional component (a), which remains closely aligned with the original signal (b). The four-class example is constructed in Appendix B.5.

![](images/4eaec49f0c5f144ebcc2ed24d1c3367cf44bc410798d6f1a5c2a7cbacf6bd31a.jpg)  
Figure 2: TPKD keeps the label signal intact. Only the teacher-conditional direction is corrected. The combined direction is passed to ordinary backpropagation through Eq. (8).

$\ell = - d _ { y } \geq 0$ the compensation. With $[ a ] _ { + } = \operatorname* { m a x } ( a , 0 )$ , ℓ uniquely solves $\begin{array} { r } { \ell = \sum _ { j \neq y } [ q _ { j } - r _ { j } - \ell ] _ { + } } \end{array}$ giving

$$
\widetilde { q } _ { j } = \operatorname* { m i n } ( q _ { j } , r _ { j } + \ell ) , \qquad d = ( - \ell , r - \widetilde { q } ) , \qquad \sum _ { j \neq y } \widetilde { q } _ { j } = 1 - \ell .\tag{9}
$$

Freezing the clipping at the current output gives a local loss whose gradient equals d at that output: fit the normalized clipped teacher and increase correct-class log-odds (Appendix B.6). The extra margin over the CE endpoint is $\begin{array} { r } { \frac { \eta } { 2 } [ r _ { j } - q _ { j } + \ell ] _ { + } \colon } \end{array}$ the most demanding competitors keep exactly the CE margin; the others gain more.

The important question is whether protection preserves genuine conditional learning or merely increases confidence in the correct class. To distinguish these effects, define the symmetric confidence direction $a _ { 0 } = ( - 1 , 1 / ( K - 1 ) , \ldots , 1 / ( K - 1 ) )$ and the remaining component $w _ { \mathrm { c o n d } } =$ $d - \ell a _ { 0 }$ . Then $d = \ell a _ { 0 } + w _ { \mathrm { c o n d } } , w _ { \mathrm { c o n d } } \in \mathcal { L } _ { y } , w _ { \mathrm { c o n d } } \ \perp \ a _ { 0 }$ . A step along $- a _ { 0 }$ increases every correct-class margin equally and leaves the relative probabilities among incorrect classes unchanged. The component $w _ { \mathrm { c o n d } }$ changes those relations. The next theorem shows that this component remains nonzero and aligned with the teacher’s conditional signal whenever conditional error remains.

Theorem 3 (Retained conditional learning). For $u \in { \mathcal { L } } _ { y } ,$ , define $\gamma _ { K } = K / [ 2 ( K - 1 ) ]$ ] and $c _ { K } =$ $2 \sqrt { \gamma _ { K } } / ( 1 + \gamma _ { K } )$ . Then

$$
u ^ { \top } d = \| d \| ^ { 2 } \geq \gamma _ { K } \| u \| ^ { 2 } , \qquad \| w _ { \mathrm { c o n d } } \| ^ { 2 } \geq \gamma _ { K } \| d \| ^ { 2 } \geq \gamma _ { K } ^ { 2 } \| u \| ^ { 2 } ,\tag{10}
$$

$$
\cos { \angle } ( u , w _ { \mathrm { c o n d } } ) \geq c _ { K } \geq { \frac { 2 \sqrt { 2 } } { 3 } } \qquad ( u \neq 0 ) .\tag{11}
$$

Both norm bounds are tight and can be attained simultaneously.

Here $u ^ { \top } d$ measures first-order conditional loss reduction along −d. The first bound retains more than half the reduction along −u at the same step size; the remaining bounds ensure a nonzero, aligned conditional component whenever $r \neq q ( { \mathrm { F i g u r e ~ } } 3 )$ . Appendix B quantifies the removed signal and compensation, and establishes optimality under local quadratic and fixed-displacement budgets.

## 5.3 BOTH KINDS OF LEARNING CAN BE COMPLETED TOGETHER

Can the learning retained in each step accumulate until both objectives are achieved? We prove that it can: for a fixed positive teacher-conditional target and sufficiently small constant output steps,

label fitting and conditional matching converge together. To track their joint progress, define the potential $\Phi ( z ) = 4 L _ { \mathrm { C E } } ( z , y ) + L _ { \mathrm { c o n d } } ( z )$ and its logit gradient $\begin{array} { r } { g _ { \Phi } = \nabla _ { z } \Phi = 4 h + u . } \end{array}$ . A subscript k denotes evaluation at output $z _ { k }$ , for example $p _ { k } =$ softmax(z<sub>k</sub>) and $v _ { k } = v ( z _ { k } )$

Theorem 4 (Joint descent and completed learning). The complete direction obeys the tight alignment bound

$$
g _ { \Phi } ^ { \top } v \ge \| v \| ^ { 2 } + \frac { \gamma _ { K } } { 4 } \| g _ { \Phi } \| ^ { 2 } , \qquad \cos \angle ( g _ { \Phi } , v ) \ge \sqrt { \gamma _ { K } } .\tag{12}
$$

For fixed positive q, finite initial logits, and $z _ { k + 1 } = z _ { k } - \eta v _ { k }$ with $0 < \eta \leq 4 / 5$

$$
\Phi ( z _ { k + 1 } ) \leq \Phi ( z _ { k } ) - \frac { \eta \gamma _ { K } } { 4 } \| g _ { \Phi , k } \| ^ { 2 } - \eta ( 1 - 5 \eta / 4 ) \| v _ { k } \| ^ { 2 } .\tag{13}
$$

Consequently, $p _ { k , y }  \ .$ 1, $r _ { k }  q ,$ and $v _ { k } \to 0 .$

The retained conditional signal therefore supports completed learning, not just a favorable local direction. Joint and component gradient residuals have vanishing time-averaged squared norms (Appendix C.2); Appendix C.4 gives conditions for a complete step to lower its own conditional KL. For comparison, define the uniform non-target vector by $\nu _ { j } = 1 / ( K - 1 )$ for $j \neq y .$ . CE output gradient flow, or fixed steps $0 < \eta \le 1 / 2 0$ , fits the label but drives $r  \nu .$ . For nonuniform q, both rules fit the label, but TPKD retains the teacher’s conditional structure rather than a uniform distribution (Appendix C.3).

## 5.4 TURNING OUTPUT PROGRESS INTO PARAMETER LEARNING

The output construction becomes useful for training through two connections: exact execution by a classification head, and conditional learning beyond compensation under ordinary backpropagation.

Proposition 1 (Exact head update). Ifthefrozen batchfeature matrix hasfull row rank, a unique minimum-Frobenius-norm head displacement realizes Eq. (6) exactlyfor every example. It therefore inherits the step’s margin and loss orderings in Eq. (7) (Appendix D.1).

Learning beyond compensation. Compare the parameter steps from Eq. (8) for TPKD and $h _ { i } +$ $\ell _ { i } a _ { 0 , i } / 2 \colon$ the same confidence compensation without direct conditional adjustment. Backpropagation can distort these signals. Let $\chi _ { c } , \chi \Phi$ be its largest-to-smallest squared stretch ratios on the spans of the batch-concatenated pairs $( u _ { i } , w _ { \mathrm { c o n d } , i } )$ and $( g _ { \Phi , i } , v _ { i } )$ , respectively. Zero minimum stretch gives an infinite ratio. Define $\kappa _ { \Phi } = ( 1 + \sqrt { \gamma _ { K } } ) / ( 1 - \sqrt { \gamma _ { K } } )$ and $\kappa _ { c } = \kappa _ { \Phi } ^ { 2 }$

Theorem 5 (Conditional learning through network parameters). At a differentiable parameter state, for sufficiently small equal positive parameter steps: (i) if some $u _ { i } \neq$ 0 and $\chi _ { c } < \kappa _ { c } ,$ TPKD has a nonzero conditional parameter increment over compensation only and achieves strictly lower batch-average conditional KL; (ii) if both joint batch vectors are nonzero and $\chi _ { \Phi } < \kappa _ { \Phi }$ , TPKD strictly decreases the batch mean ofΦ.

The first conclusion guarantees a useful conditional increment, not merely extra confidence; the second guarantees progress of the complete update. These results establish both a parameter implementation and, under the stated conditions, genuine learning from the retained teacher signal. Appendix D gives the equivalent formulas, proofs, quantitative bounds, computation and measured-angle refinements.

## 6 EXPERIMENTS

The mechanism study uses CIFAR-100 (Krizhevsky, 2009), with a VOLO-D2 teacher and PiT-B student (Yuan et al., 2023; Heo et al., 2021), under the main training protocol. CE20/40/60 denote label-only students after 20/40/60 epochs. Margins measure the correct class’s advantage over each competitor; conditional KL measures the error in learning the teacher’s relations among incorrect classes.

Blocking in ordinary training. We first test whether ordinary label training can block full imitation while leaving conditional knowledge unlearned. Each CE state is evaluated on 2,000 fixed images under eight views; an image–view pair is blocked when the student’s probability for every incorrect class is no greater than the teacher’s. Blocking rises from 50.00% to 58.90%, yet every blocked pair has positive conditional error (Table 1). Thus the theoretical obstruction occurs in ordinary training while teacher knowledge remains available.

Table 1: Full-imitation blocking leaves conditional knowledge unlearned. Conditional error and transfer cost are means within the blocked set (nats).
<table><tr><td>Student state</td><td>Blocked (%)</td><td>Remaining conditional KL</td><td>Minimum log-odds cost</td></tr><tr><td>CE20</td><td>50.00</td><td>1.696</td><td>7.520</td></tr><tr><td>CE40</td><td>55.29</td><td>1.766</td><td>7.632</td></tr><tr><td>CE60</td><td>58.90</td><td>1.803</td><td>7.569</td></tr></table>

Protection without discarding knowledge. Exact head updates on frozen features isolate whether the prescribed correction protects the task without discarding conditional learning. At CE20 and CE60, we compare TPKD with CE, compensation only, unprojected learning, and CE plus full KL from the same start. Compensation only keeps the correct-class push but removes changes among incorrect classes. TPKD gains conditional knowledge beyond CE on all 64 batches and all 4,096 example states while preserving every CE-step margin to numerical precision. Compensation only has CE’s conditional endpoint; unprojected learning loses at least one CE-step margin on every example. TPKD therefore combines the two desired effects. Minimum retention and complete-update cosine exceed the 100-class bounds of approximately 50.51% and 0.7107 (Theorems 3–4); all joint-descent checks pass (Table 2).

Table 2: Exact head steps learn conditional structure while preserving label-step progress.
<table><tr><td>State</td><td>Gain  $( 1 0 ^ { - 3 } )$ </td><td>KL change  $( 1 0 ^ { - 3 } )$ </td><td>Min. retained (%)</td><td>Min. cosine</td><td>Φ change  $( 1 0 ^ { - 3 } )$ </td></tr><tr><td>CE20</td><td>1.195</td><td>-1.359</td><td>52.13</td><td>0.7220</td><td>-3.302</td></tr><tr><td>CE60</td><td>1.213</td><td>-1.283</td><td>52.13</td><td>0.7221</td><td>-1.831</td></tr></table>

KL denotes conditional KL. Gain is CE minus TPKD at the endpoint; changes are TPKD endpoint minus common start. Loss columns are means; minima are over individual examples. Retention is $u ^ { \top } d / \| u \| ^ { 2 } ;$ cosine compares v with $g _ { \Phi }$ . Each state: $3 2 \times 6 4$ examples; output step 0.01.

Conditional learning through the network. We next test whether this conditional increment remains useful under ordinary full-network optimization. At CE20 and CE60, 16 native-optimizer pairs per state compare TPKD with compensation only from identical model, optimizer and random states. For $K = 1 0 0$ , Theorem 5 gives distortion-ratio limits of approximately 34.96 for conditional learning and 5.91 for joint descent. All 32 batches pass both geometric tests, and every native pair has lower conditional KL under TPKD (Table 3). These paired gains isolate conditional learning beyond compensation.

Table 3: The conditional advantage survives ordinary backpropagation.
<table><tr><td>State</td><td>Conditional test</td><td>Joint descent test</td><td>Positive pairs</td><td>Mean gain  $( 1 0 ^ { - 3 } )$  1</td></tr><tr><td>CE20</td><td>16/16</td><td>16/16</td><td>16/16</td><td>8.252</td></tr><tr><td>CE60</td><td>16/16</td><td>16/16</td><td>16/16</td><td>9.886</td></tr></table>

Gain is compensation-only endpoint conditional KL minus TPKD endpoint conditional KL.

Accumulation over successive updates. Finally, we test whether the single-step advantage accumulates. From CE60, CE, compensation only and TPKD follow the same data sequence for 128 native-optimizer iterations. On 2,048 fixed observation images separate from the update images, TPKD lowers conditional KL from 1.800 to 1.557. Its endpoint KL is 0.244 nats below CE’s and 0.320 below compensation only’s: the extra conditional learning persists over successive updates. Protocols are in Appendix E.

## 6.1 FULL TRAINING IN VISION AND TEXT

TPKD is not limited to vision: its update uses class probabilities and labels, not modality-specific features. We therefore also evaluate full training on CLINC150 text intent classification (Larson et al., 2019), with BERT-large as teacher and BERT-Mini as student (Devlin et al., 2019; Turc et al., 2019).

Table 4 compares CE and nine distillation methods (Hinton et al., 2015; Zhao et al., 2022; Roth et al., 2024; Yang et al., 2025; Hayder et al., 2026) using shared within-domain protocols and final-epoch evaluation fixed in advance over three seeds (Appendix E.1).

TPKD reaches 88.05% on CIFAR-100 and 93.81% on CLINC150, improving over standard KD by 0.47 and 0.35 percentage points and over CE by 0.68 and 0.56 points. Retained conditional learning thus benefits full training in both domains.

Table 4: Full-training accuracy (%). Mean ± sample standard deviation over three seeds.
<table><tr><td>Method</td><td>CIFAR-100 60 epochs</td><td>CLINC150 4 epochs</td><td>Method</td><td>CIFAR-100 60 epochs</td><td>CLINC150 4 epochs</td></tr><tr><td>CE</td><td> $8 7 . 3 7 \pm 0 . 0 7$ </td><td> $9 3 . 2 4 \pm 0 . 2 8$ </td><td>DP-S</td><td> $8 3 . 6 5 \pm 0 . 0 1$ </td><td> $9 3 . 4 4 \pm 0 . 2 8$ </td></tr><tr><td>KD</td><td> $8 7 . 5 8 \pm 0 . 0 7$ </td><td> $9 3 . 4 6 \pm 0 . 3 4$ </td><td>XE-KL</td><td> $8 4 . 1 7 \pm 0 . 0 5$ </td><td> $9 3 . 4 1 \pm 0 . 2 2$ </td></tr><tr><td>KL-Dist</td><td> $8 2 . 2 6 \pm 0 . 0 5$ </td><td> $9 3 . 3 8 \pm 0 . 1 9$ </td><td>DHKD</td><td> $8 5 . 8 3 \pm 0 . 1 0$ </td><td> $9 3 . 2 9 \pm 0 . 3 8$ </td></tr><tr><td>DKD</td><td> $8 4 . 5 3 \pm 0 . 0 9$ </td><td> $9 3 . 5 9 \pm 0 . 2 0$ </td><td>DTO-KD*</td><td> $8 7 . 8 1 \pm 0 . 1 1$ </td><td> $9 3 . 4 6 \pm 0 . 2 3$ </td></tr><tr><td>DP-U</td><td> $8 3 . 1 4 \pm 0 . 0 4$ </td><td> $9 3 . 3 8 \pm 0 . 1 9$ </td><td>TPKD</td><td> ${ \bf 8 8 . 0 5 \pm 0 . 1 2 }$ </td><td> ${ \bf 9 3 . 8 1 \pm 0 . 2 7 }$ </td></tr></table>

<sup>∗</sup>Text DTO-KD includes multi-layer feature distillation. Full configurations are in Appendix E.1.

## 6.2 WHICH COMPONENTS MAKE THE DIFFERENCE?

Independent label learning. Matching the conditional target makes $d = 0 ,$ , but the CE gradient continues fitting the label. Removing it lowers accuracy by 2.72 points in vision and 2.21 in text (Table 5).

Learning the right teacher object. Safe full KL keeps CE and projection but uses $p - t$ instead of the conditional gradient. In a blocked state, its projected teacher increment is zero, whereas TPKD’s remains nonzero when $r \neq q$ (Appendix A.5). Accuracy falls to 87.21% and 93.33%, close to CE. Protecting full imitation alone does not recover the benefit of conditional learning.

The cost of protection. Without projection, accuracy is 87.83% and 93.83%: TPKD is 0.22 points higher in vision and differs by only 0.02 points in text. The correction thus provides the demonstrated margin protection while retaining closely matched predictive performance.

Table 5: Component ablations. Accuracy (%), mean ± sample standard deviation over three seeds.
<table><tr><td>Condition</td><td>Direction</td><td>CIFAR-100</td><td>CLINC150</td></tr><tr><td>CE</td><td>h</td><td> $8 7 . 3 7 \pm 0 . 0 7$ </td><td> $9 3 . 2 4 \pm 0 . 2 8$ </td></tr><tr><td>Without CE gradient</td><td> $d / 2$ </td><td> $8 5 . 3 3 \pm 0 . 0 5$ </td><td> $9 1 . 6 0 \pm 0 . 0 7$ </td></tr><tr><td>Safe full-KL</td><td> $h + \Pi _ { \mathcal { K } _ { u } } ( p - t ) / 2$ </td><td> $8 7 . 2 1 \pm 0 . 0 4$ </td><td> $9 3 . 3 3 \pm 0 . 2 9$ </td></tr><tr><td>Without projection</td><td> $h + u / 2$ </td><td> $8 7 . 8 3 \pm 0 . 0 3$ </td><td> $9 3 . 8 3 \pm 0 . 2 3$ </td></tr><tr><td>TPKD</td><td> $h + { d } / { 2 }$ </td><td> $8 8 . 0 5 \pm 0 . 1 2$ </td><td> $9 3 . 8 1 \pm 0 . 2 7$ </td></tr></table>

## 7 CONCLUSION

Full imitation can become blocked before the teacher’s knowledge is exhausted. We prove that its conditional structure remains fully transferable under task preservation and determine the exact confidence compensation required. TPKD implements this separation by retaining the label gradient and adding the nearest safe conditional direction. Tight retention and joint-learning results explain why genuine conditional learning survives; parameter-space results connect it to network updates. Mechanism experiments locate blocking in ordinary training and follow conditional gains through successive updates, while full training and ablations demonstrate their contribution in vision and text. Learning from a teacher need not mean reproducing its full prediction: conditional knowledge can be acquired without surrendering label-learning progress.

## AI USE STATEMENT

No generative AI tools were used in conducting this research or preparing this manuscript.

## REPRODUCIBILITY STATEMENT

Appendices A–D prove the results stated in the main text; Appendix E specifies the reported experiments. The source bundle includes the reported accuracy summaries, table-generation scripts and analytical checks.

## REFERENCES

Jang Hyun Cho and Bharath Hariharan. On the efficacy of knowledge distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4794–4802, 2019.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–4186, 2019.

Hrayr Harutyunyan, Ankit Singh Rawat, Aditya Krishna Menon, Seungyeon Kim, and Sanjiv Kumar. Supervision complexity and its role in knowledge distillation. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2301.12245.

Zeeshan Hayder, Ali Cheraghian, Lars Petersson, Mehrtash Harandi, and Richard Hartley. DTO-KD: Dynamic trade-off optimization for effective knowledge distillation. In International Conference on Learning Representations, 2026.

Byeongho Heo, Sangdoo Yun, Dongyoon Han, Sanghyuk Chun, Junsuk Choe, and Seong Joon Oh. Rethinking spatial dimensions of vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11936–11945, 2021.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Haiduo Huang, Jiangcheng Song, Yadong Zhang, and Pengju Ren. DeepKD: A deeply decoupled and denoised knowledge distillation trainer. In Advances in Neural Information Processing Systems, volume 38, pp. 27138–27167, 2025. doi: 10.52202/085713-0915.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, and Jason Mars. An evaluation dataset for intent classification and out-of-scope prediction. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 1311–1316, 2019.

Hengyue Liang, Le Peng, and Ju Sun. Selective classification under distribution shifts. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id= dmxMGW6J7N.

Aditya K. Menon, Ankit Singh Rawat, Sashank Reddi, Seungyeon Kim, and Sanjiv Kumar. A statistical perspective on distillation. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pp. 7632–7642, 2021.

Seyed Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, and Hassan Ghasemzadeh. Improved knowledge distillation via teacher assistant. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34, pp. 5191–5198, 2020.

Hossein Mobahi, Mehrdad Farajtabar, and Peter L. Bartlett. Self-distillation amplifies regularization in Hilbert space. In Advances in Neural Information Processing Systems, volume 33, pp. 3351–3361, 2020.

Vaishnavh Nagarajan, Aditya K. Menon, Srinadh Bhojanapalli, Hossein Mobahi, and Sanjiv Kumar. On student-teacher deviations in distillation: Does it pay to disobey? In Advances in Neural Information Processing Systems, volume 36, pp. 5961–6000, 2023. doi: 10.52202/075280-0261. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ 12d286282e1be5431ea05262a21f415c-Abstract-Conference.html.

Neal Parikh and Stephen Boyd. Proximal algorithms. Foundations and Trends in Optimization, 1(3): 127–239, 2014.

Mary Phuong and Christoph H. Lampert. Towards understanding knowledge distillation. In Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 5142–5151, 2019.

Simone Ricci, Niccolò Biondi, Federico Pernici, and Alberto Del Bimbo. Mitigating negative flips via margin preserving training. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 8721–8730, 2026. doi: 10.1609/aaai.v40i11.37825.

Karsten Roth, Lukas Thede, Almut Sophia Koepke, Oriol Vinyals, Olivier Hénaff, and Zeynep Akata. Fantastic gains and where to find them: On the existence and prospect of general knowledge transfer between any pretrained model. In International Conference on Learning Representations, 2024.

Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A. Alemi, and Andrew Gordon Wilson. Does knowledge distillation really work? In Advances in Neural Information Processing Systems, volume 34, pp. 6906–6919, 2021.

Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Well-read students learn better: On the importance of pre-training compact models. arXiv preprint arXiv:1908.08962, 2019.

Penghui Yang, Chen-Chen Zong, Sheng-Jun Huang, Lei Feng, and Bo An. Dual-head knowledge distillation: Enhancing logits utilization with an auxiliary head. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, pp. 3530–3541, 2025.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems, volume 33, pp. 5824–5836, 2020.

Li Yuan, Qibin Hou, Zihang Jiang, Jiashi Feng, and Shuicheng Yan. VOLO: Vision outlooker for visual recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5): 6575–6586, 2023. doi: 10.1109/TPAMI.2022.3206108.

Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, and Jiajun Liang. Decoupled knowledge distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11953–11962, 2022.

## A PROOFS SUPPORTING TASK PRESERVATION AND THE SEPARATION

This appendix proves the equivalence in Section 3, Theorems 1–2, and the two component consequences stated in Section 6.2.

## A.1 WEIGHTED TASK LOSSES AND PER-MARGIN PRESERVATION

For nonnegative competitor weights ω, let $\begin{array} { r } { L _ { \omega } ( z , y ) = \log ( 1 + \sum _ { j \neq y } \omega _ { j } e ^ { - M _ { j } ( z ) } ) } \end{array}$ . The all-ones choice is cross-entropy; a single nonzero weight isolates one competitor.

Proof of the equivalence in Section 3. For logits $z , z ^ { \prime } \in \mathbb { R } ^ { K }$ , if $M _ { j } ( z ^ { \prime } ) \geq M _ { j } ( z )$ for all $j \neq y ,$ then $e ^ { - M _ { j } ( z ^ { \prime } ) } \leq e ^ { - M _ { j } ( z ) } ;$ ; any $\omega _ { j } \geq 0$ preserves the termwise inequality, and summing and using the monotonicity of log gives $L _ { \omega } ( z ^ { \prime } , y ) \overset { \cdot } { \leq } L _ { \omega } ( z , y )$ . Conversely, for any $j \neq y$ take $\omega _ { j } = 1$ and all other weights zero, which gives log( $1 + e ^ { - M _ { j } ( z ^ { \prime } ) } ) \leq \log ( 1 + e ^ { - M _ { j } ( z ) } ) ;$ since $M \mapsto \log ( 1 + e ^ { - M } )$ is strictly decreasing, $M _ { j } ( { \bar { z } } ^ { \prime } ) \geq M _ { j } ( z )$ . Taking $z ^ { \prime }$ and the reference to be $z - \eta v$ and $z - \eta h$ yields the ordering of TPKD against the label base under every competitor weighting.

## A.2 PROOF OF THEOREM 1

Let $\Delta M _ { j } = M _ { j } ( p ^ { \prime } ) - M _ { j } ( p )$ . Since $\begin{array} { r } { \sum _ { j } ( t _ { j } - p _ { j } ) = 0 } \end{array}$ , the common quantity $\log ( p _ { y } / p _ { y } ^ { \prime } )$ may be subtracted from every term without changing the sum, so

$$
\begin{array} { r l } & { \mathrm { K L } ( t \| p ^ { \prime } ) - \mathrm { K L } ( t \| p ) - \mathrm { K L } ( p \| p ^ { \prime } ) = \displaystyle \sum _ { j } ( t _ { j } - p _ { j } ) \log \frac { p _ { j } } { p _ { j } ^ { \prime } } } \\ & { \qquad = \displaystyle \sum _ { j \ne y } ( t _ { j } - p _ { j } ) \Big [ \log \frac { p _ { j } } { p _ { j } ^ { \prime } } - \log \frac { p _ { y } } { p _ { y } ^ { \prime } } \Big ] = \sum _ { j \ne y } ( t _ { j } - p _ { j } ) \Delta M _ { j } . } \end{array}\tag{A.1}
$$

This gives the identity in Eq. (3) before imposing the sign conditions. When $p ^ { \prime } \in \mathcal { S } ( p )$ and $p _ { j } \leq t _ { j }$ for all non-target coordinates, the right-hand side is nonnegative; for strictly positive probabilities $\mathrm { K L } ( p \| p ^ { \prime } )$ is strictly positive when $\boldsymbol { p ^ { \prime } } \ne p ,$ so p is the unique minimizer and $\operatorname { E q . } \ ( 3 )$ follows. Conversely, if $p _ { j } > t _ { j }$ for some $j ,$ let $z ^ { \prime } = z - \eta e _ { i }$ and $p ^ { \prime } = \mathrm { s o f t m a x } ( z ^ { \prime } )$ : this leaves the other margins unchanged and increases the $j \mathrm { - t h } .$ , so $p ^ { \prime } \in \mathsf { \bar { S } } ( p ) ;$ since $\partial _ { z _ { j } } \mathrm { K L } ( t | | \mathrm { s o f t m a x } ( z ) ) = p _ { j } - t _ { j } ,$ the derivative of the full KL in η at zero equals $- ( p _ { j } - \dot { t } _ { j } ) < 0 .$ , and p is not a minimizer.

Strict blocking with residual conditional error. For any fixed positive $r \neq q$ and $\tau < 1$ , every ρ with

$$
0 < 1 - \rho < ( 1 - \tau ) \operatorname* { m i n } _ { j \neq y } q _ { j } / r _ { j }
$$

is strictly blocked and has $\mathrm { K L } ( q \| r ) > 0$ . Thus blocking with unlearned conditional structure has nonempty interior.

## A.3 PROOF OF THEOREM 2

With $\mathcal { R } _ { \infty } = \operatorname* { m a x } _ { j \neq y } q _ { j } / r _ { j }$ , the smallest feasible correct-class probability is

$$
a _ { \mathrm { m i n } } = \frac { \rho \mathcal { R } _ { \infty } } { 1 - \rho + \rho \mathcal { R } _ { \infty } } < 1 .\tag{A.2}
$$

For $p ^ { \prime } = ( a , ( 1 - a ) q )$ , the j-th margin constraint $M _ { j } ( p ^ { \prime } ) \geq M _ { j } ( p )$ is equivalent to

$$
\log { \frac { a } { ( 1 - a ) q _ { j } } } \geq \log { \frac { \rho } { ( 1 - \rho ) r _ { j } } } \iff \log \mathrm { i t } ( a ) - \log \mathrm { i t } ( \rho ) \geq \log { \frac { q _ { j } } { r _ { j } } } .
$$

Taking the maximum over j gives $D _ { \infty } ( q \| r )$ ; solving logi $\dot { \mathbf { \rho } } ( a ) = \mathrm { l o g i t } ( \rho ) + \mathrm { l o g } \mathcal { R } _ { \infty }$ for a gives Eq. (4). Normalization $\begin{array} { r } { \sum _ { j } q _ { j } = \sum _ { j } r _ { j } = 1 } \end{array}$ implies $\mathcal { R } _ { \infty } \geq 1$ with equality if and only if $q = r ;$ positivity ensures $\mathcal { R } _ { \infty } < \dot { \infty }$ , hence $a _ { \operatorname* { m i n } } < 1$

## A.4 THE MINIMUM-COMPENSATION PATH

For $s \in [ 0 , 1 ]$ , define

$$
r ( s ) = ( 1 - s ) r + s q , \qquad a ( s ) = \frac { \rho ( 1 - s + s \mathcal { R } _ { \infty } ) } { 1 - \rho + \rho ( 1 - s + s \mathcal { R } _ { \infty } ) } .\tag{A.3}
$$

Proof. Write $l _ { j } = q _ { j } / r _ { j } \le \mathcal { R } _ { \infty }$ . By Eq. (A.3), $r _ { j } ( s ) / r _ { j } = 1 - s + s l _ { j }$ and logit $a ( s ) - \log { \mathrm { i t } } \rho =$ log $( 1 - s + s \mathcal { R } _ { \infty } )$ , so

$$
M _ { j } ( p ( s ) ) - M _ { j } ( p ) = \log ( 1 - s + s \mathcal { R } _ { \infty } ) - \log ( 1 - s + s l _ { j } ) \geq 0 ,\tag{A.4}
$$

$$
\frac { d } { d s } M _ { j } ( p ( s ) ) = \frac { \mathcal { R } _ { \infty } - l _ { j } } { ( 1 - s + s \mathcal { R } _ { \infty } ) ( 1 - s + s l _ { j } ) } \ge 0 .\tag{A.5}
$$

With $\Delta _ { j } = q _ { j } - r _ { j }$ we have $q _ { j } = r _ { j } ( s ) + ( 1 - s ) \Delta _ { j }$ and $\textstyle \sum _ { j } \Delta _ { j } = 0$ , hence

$$
\frac { d } { d s } \operatorname { K L } ( q \| r ( s ) ) = - \sum _ { j } \frac { q _ { j } \Delta _ { j } } { r _ { j } ( s ) } = - ( 1 - s ) \sum _ { j } \frac { \Delta _ { j } ^ { 2 } } { r _ { j } ( s ) } .\tag{A.6}
$$

If $q \neq r$ this derivative is strictly negative for $s < 1 ; \mathrm { a t } s = 1 , r ( 1 ) = q \mathrm { a n d } a ( 1 ) = a _ { \mathrm { m i n } }$

The numerical illustration in Figure 1. The correct label is class 1, with $p = ( 0 . 9 0 , 0 . 0 7 , 0 . 0 3 )$ and $t = ( 0 . 4 0 , 0 . 2 5 , 0 . 3 5 )$ . Since $0 . 0 7 < 0 . 2 5$ and $0 . 0 3 < 0 . 3 5$ , Theorem 1 applies. The conditional distributions are nevertheless different:

$$
\rho = { \frac { 9 } { 1 0 } } , \quad \tau = { \frac { 2 } { 5 } } , \quad \quad r = \left( { \frac { 7 } { 1 0 } } , { \frac { 3 } { 1 0 } } \right) , \quad \quad q = \left( { \frac { 5 } { 1 2 } } , { \frac { 7 } { 1 2 } } \right) .
$$

The largest relative mismatch is on class 3:

$$
\mathcal { R } _ { \infty } = \operatorname* { m a x } \left\{ \frac { 2 5 } { 4 2 } , \frac { 3 5 } { 1 8 } \right\} = \frac { 3 5 } { 1 8 } , \qquad D _ { \infty } ( q \| r ) = \log \frac { 3 5 } { 1 8 } \approx 0 . 6 6 4 9 7 6 .
$$

Inserting this value into Eq. (4) gives

$$
a _ { \mathrm { m i n } } = \frac { ( 9 / 1 0 ) ( 3 5 / 1 8 ) } { 1 / 1 0 + ( 9 / 1 0 ) ( 3 5 / 1 8 ) } = \frac { 3 5 } { 3 7 } , \qquad p ^ { * } = \left( \frac { 3 5 } { 3 7 } , \frac { 5 } { 2 2 2 } , \frac { 7 } { 2 2 2 } \right) .
$$

The endpoint’s non-target ratio is 5:7, exactly the teacher’s. Its correct-versus-incorrect probability ratios are

$$
\frac { p _ { 1 } } { p _ { 2 } } = \frac { 9 0 } { 7 } \longrightarrow \frac { p _ { 1 } ^ { * } } { p _ { 2 } ^ { * } } = 4 2 , \qquad \frac { p _ { 1 } } { p _ { 3 } } = 3 0 \longrightarrow \frac { p _ { 1 } ^ { * } } { p _ { 3 } ^ { * } } = 3 0 .
$$

Thus the class-2 margin increases and the class-3 margin is unchanged. Along the path,

$$
r _ { 2 } ( s ) = \frac { 7 } { 1 0 } - \frac { 1 7 s } { 6 0 } , \quad r _ { 3 } ( s ) = \frac { 3 } { 1 0 } + \frac { 1 7 s } { 6 0 } , \qquad a ( s ) = \frac { \frac { 9 } { 1 0 } ( 1 + \frac { 1 7 s } { 1 8 } ) } { \frac { 1 } { 1 0 } + \frac { 9 } { 1 0 } ( 1 + \frac { 1 7 s } { 1 8 } ) } .
$$

The left plot uses $x = r _ { 3 } ( s )$ and $\Delta = \mathrm { l o g i t } ( a ( s ) ) - \mathrm { l o g i t } ( \rho )$ . For an arbitrary positive conditional vector $( 1 - x , x )$ , its safe boundary is $\Delta = \dot { D _ { \infty } } ( ( 1 - x , x ) \| r )$ . The teacher is at $x = 7 / 1 2$ and $\Delta = \mathrm { l o g i t } ( 2 / 5 ) - \mathrm { l o g i t } ( 9 / 1 0 ) = \mathrm { l o g } ( 2 / 2 7 ) \approx - 2$ .60269, below this region.

The right plot divides each loss by its own initial value. Substitution into the loss definitions gives

$$
\begin{array} { r l } & { \frac { p } { L _ { \mathrm { { C E } } } } \qquad - \log ( 9 / 1 0 ) \qquad \quad - \log ( 3 5 / 3 7 ) } \\ & { \frac { 5 } { 1 2 } \log \frac { 2 5 } { 4 2 } + \frac { 7 } { 1 2 } \log \frac { 3 5 } { 1 8 } \qquad \quad 0 } \\ & { \frac { L _ { T } } { L _ { T } } \quad \left| \frac { { \small \frac { 4 } { 2 } } } { 5 } \log \frac { 4 } { 9 } + \frac { 1 } { 4 } \log \frac { 2 5 } { 7 } + \frac { 7 } { 2 0 } \log \frac { 3 5 } { 3 } \quad \frac { 2 } { 5 } \log \frac { 7 4 } { 1 7 5 } + \frac { 3 } { 5 } \log \frac { 1 1 1 } { 1 0 } \right. } \end{array}
$$

Numerically, label CE changes from approximately 0.105361 to 0.055570, conditional KL from 0.171739 to zero, and full teacher KL from 0.853727 to 1.099879. These are evaluations of the analytic path, whose margin monotonicity and conditional descent were established above.

## A.5 THE TWO COMPONENT CONSEQUENCES USED IN THE ABLATION

For the control without $h , r = q$ implies $u = ( 0 , r - q ) = 0$ and hence $d = \Pi _ { K _ { u } } ( 0 ) = 0$ . This need not imply that $p _ { y } = 1$ , because r is independent of the correct-class probability. The ordinary label gradient has $h _ { y } = p _ { y } - 1$ , so it remains nonzero at every positive prediction with $p _ { y } < 1$

For the safe full-KL control, define the polar cone $\mathcal { K } _ { y } ^ { \circ } = \left\{ g \in \mathbb { R } ^ { K } : g ^ { \top } x \leq 0 \right.$ for every $x \in \kappa _ { y } \}$ Put $g = p - t$ and suppose $p _ { j } \leq t _ { j }$ for $j \neq y .$ . For every $x \in \mathcal { K } _ { y }$ , using $\mathbf { 1 } ^ { \top } g = 0$

$$
g ^ { \top } x = \sum _ { j \neq y } ( t _ { j } - p _ { j } ) ( x _ { y } - x _ { j } ) \leq 0 .
$$

Thus g belongs to the polar cone $\mathcal { K } _ { y } ^ { \circ }$ and $\Pi _ { { \mathcal K } _ { u } } ( g ) \ = \ 0 . \ \mathrm { ~ I f ~ } r \ \ne \ q .$ , Theorem 3 instead gives $\Vert \Pi _ { \mathcal { K } _ { u } } ( u ) \Vert ^ { 2 } \geq \gamma _ { K } \Vert u \Vert ^ { 2 } > 0$ . This proves the loss of the full-KL teacher increment at a blocked state, while the conditional increment remains available.

## A.6 THE IMITATION OPTIMUM WHEN ONLY CE IS PROTECTED

The comparison in Section 3 can be solved exactly. Preserving only the label cross-entropy requires $p _ { y } ^ { \prime } \ge \rho .$ . Write $p ^ { \prime } = ( a , ( 1 - a ) x )$ , where x is a conditional probability vector. The KL decomposition gives

$$
\mathrm { K L } ( t \| p ^ { \prime } ) = \mathrm { k l } _ { \mathrm { B e r } } ( \tau \| a ) + ( 1 - \tau ) \mathrm { K L } ( q \| x ) .
$$

For any feasible a, the second term is uniquely minimized by $x = q$ . Since

$$
\frac \partial { \partial a } \mathrm { k l _ { B e r } } ( \tau \| a ) = \frac { a - \tau } { a ( 1 - a ) } ,
$$

the unique optimum over a $\geq \rho$ is $a _ { * } = \operatorname* { m a x } ( \rho , \tau )$ . Thus

$$
p _ { \mathrm { C E } } ^ { * } = ( a _ { * } , ( 1 - a _ { * } ) q ) , \qquad \operatorname* { m i n } _ { p _ { y } ^ { \prime } \geq \rho } { \mathrm { K L } } ( t \| p ^ { \prime } ) = { \mathrm { k l } } _ { \mathrm { B e r } } ( \tau \| a _ { * } ) .\tag{A.7}
$$

This solution matches the conditional target without lowering the correct-class probability. It can still reduce an individual competitor margin, which the stronger requirement in Eq. (2) preserves. For the example in Figure $1 , \bar { p _ { \mathrm { C E } } } ^ { * } = ( 0 . 9 , \bar { 1 } / 2 4 , 7 / 1 2 0 )$ has unchanged CE, but its third-class probability exceeds the initial 0.03 and hence its third-class margin is smaller.

## A.7 THE FINITE-COMPENSATION FRONTIER

Theorem 2 gives the budget needed for complete conditional transfer. With a smaller budget, the best partial transfer also has a closed form. Fix $\bar { B } \geq 0$ as the maximum allowed increase in correct-class log-odds. The attainable conditional distributions are

$$
\mathcal { Q } _ { B } ( r ) = \left\{ x \in \mathbb { R } _ { > 0 } ^ { K - 1 } : \sum _ { j \neq y } x _ { j } = 1 , \quad x _ { j } \leq e ^ { B } r _ { j } \ ( j \neq y ) \right\} .\tag{A.8}
$$

Indeed, a prediction $( a , ( 1 - a ) x )$ preserves the jth margin exactly when $x _ { j } / r _ { j } \leq \exp ( \log \mathrm { i t } ( a ) -$ $\log \mathrm { i t } ( \rho ) )$ . Any x in Eq. (A.8) is attained safely with $\mathrm { l o g i t } ( a ) = \mathrm { l o g i t } ( \rho ) + \bar { B } .$

Define the least remaining conditional error by

$$
\mathcal { V } ( B ) = \operatorname* { m i n } _ { x \in \mathcal { Q } _ { B } ( r ) } \mathrm { K L } ( q \| x ) .
$$

Let $\lambda > 0$ enforce normalization. The unique minimizer $x ^ { * }$ is positive and satisfies

$$
x _ { j } ^ { * } = \operatorname* { m i n } \{ e ^ { B } r _ { j } , q _ { j } / \lambda \} , \quad \quad \sum _ { j \neq y } x _ { j } ^ { * } = 1 ,\tag{A.9}
$$

Moreover,

$$
\begin{array} { r } { \mathcal V ( 0 ) = \mathrm { K L } ( q \| r ) , \qquad \mathcal V ( B ) = 0 \quad \Longleftrightarrow \quad B \geq D _ { \infty } ( q \| r ) . } \end{array}\tag{A.10}
$$

The function $\nu$ is nonincreasing and convex in B.

Proof. The closure of the feasible set is compact and contains the positive vector r. Since $q$ is positive, the objective diverges when any coordinate approaches zero; a positive minimizer therefore exists. Strict convexity of $- \sum _ { j } q _ { j }$ log x<sub>j</sub> gives uniqueness. For $B > 0 .$ , let $\mu _ { j } \geq 0$ be the multipliers for $x _ { j } \le e ^ { B } r _ { j }$ . The KKT conditions are

$$
- q _ { j } / x _ { j } + \lambda + \mu _ { j } = 0 , \qquad \mu _ { j } \geq 0 , \qquad \mu _ { j } ( x _ { j } - e ^ { B } r _ { j } ) = 0 .
$$

At least one coordinate is uncapped because the caps sum to $e ^ { B } > 1 \AA$ ; hence $\lambda = q _ { j } / x _ { j } > 0$ there. Uncapped coordinates equal $q _ { j } / \lambda .$ , while capped coordinates equal $e ^ { B } r _ { j }$ , yielding Eq. (A.9). At $B = 0$ the only feasible vector is r; any $0 < \lambda \leq$ min<sub>j</sub> $q _ { j } / r _ { j }$ gives the same formula. Equation (A.10) follows from KL $\boldsymbol { \mathscr { s } } ( \boldsymbol { q } \| \boldsymbol { x } ) = 0$ exactly at $x = q$ and the feasibility of that vector.

The sets $\mathcal { Q } _ { B } ( r )$ grow with $B ,$ so V is nonincreasing. For convexity, define $Y ~ = ~ ( Y _ { j } ) _ { j \neq y }$ by $Y _ { j } = \log ( x _ { j } / \ ' r _ { j } )$ and use the equivalent problem

$$
{ \mathcal { V } } ( B ) = \operatorname* { m i n } _ { Y } \left\{ { \mathrm { K L } } ( q \| r ) - q ^ { \top } Y : \sum _ { j } r _ { j } e ^ { Y _ { j } } \leq 1 , \quad Y _ { j } \leq B \right\} .\tag{A.11}
$$

The normalization constraint is active at an optimum: otherwise a coordinate below its cap can be increased, strictly reducing the objective, since $q _ { j } > 0$ and the caps sum to at least one. The objective is affine and the constraints are jointly convex in $( Y , B )$ . A convex combination of feasible minimizers at two budgets is feasible at their averaged budget and has the averaged objective, proving convexity of V.

## B PROOFS SUPPORTING THE SAFE CORRECTION AND RETAINED LEARNING

These proofs support the update in Section 5.1, the clipped representation in Section 5.2, and Theorem 3. We use Euclidean projection and its Moreau decomposition (Parikh & Boyd, 2014).

## B.1 THE PROJECTION AND TASK ORDERING

The general star-cone projection is useful for both TPKD and safe full KL. For an input $\boldsymbol { x } \in \mathbb { R } ^ { K }$ , sort its non-target coordinates as $x _ { ( 1 ) } \leq \cdot \cdot \cdot \leq x _ { ( K - 1 ) }$ . With an empty prefix sum for $k = 0$

$$
\zeta = \operatorname* { m i n } _ { 0 \leq k \leq K - 1 } \frac { x _ { y } + \sum _ { j = 1 } ^ { k } x _ { ( j ) } } { k + 1 } , \qquad ( \Pi _ { K _ { y } } x ) _ { y } = \zeta , \quad ( \Pi _ { K _ { y } } x ) _ { j } = \operatorname* { m a x } ( x _ { j } , \zeta ) .\tag{B.1}
$$

For TPKD, $x = u$ has $u _ { y } = 0 ;$ , so $\zeta \leq 0$ . Computing the sorted prefixes costs O(K log K). To prove the formula, fix the projected target coordinate at $\zeta .$ . The other optimal coordinates are $\operatorname* { m a x } ( x _ { j } , \zeta )$ reducing the problem to

$$
F ( \zeta ) = \textstyle { \frac { 1 } { 2 } } ( \zeta - x _ { y } ) ^ { 2 } + \textstyle { \frac { 1 } { 2 } } \sum _ { j \not = y } [ \zeta - x _ { j } ] _ { + } ^ { 2 } .
$$

Its derivative $\begin{array} { r } { F ^ { \prime } ( \zeta ) = \zeta - x _ { y } + \sum _ { j \neq y } [ \zeta - x _ { j } ] _ { + } } \end{array}$ is continuous and strictly increasing. If k coordinates are below its root, then $\begin{array} { r } { ( k + 1 ) \zeta = \overset { \cdot } { x } _ { y } + \sum _ { j < k } x _ { ( j ) } } \end{array}$ . The minimum-prefix expression selects precisely such a consistent root, including ties, and strict convexity gives uniqueness. This formula does not assume $x _ { y } = 0$ when used for the full-KL gradient.

Nearest correction of the complete direction. Let $w \in \mathbb { R } ^ { K }$ be a candidate complete direction. The update v in Eq. (6) is the unique solution of

$$
\operatorname* { m i n i m i z e } _ { w \in \mathbb { R } ^ { K } : \ w - h \in K _ { y } } \quad \frac 1 2 \| w - ( h + u / 2 ) \| ^ { 2 } .
$$

Indeed, the constraint preserves every margin gain of the corresponding label step, and translation by h followed by the positive homogeneity of projection onto the cone gives $v = \bar { h } + \Pi _ { \mathcal { K } _ { y } } ( u ) / 2$

Full task ordering. From $h = ( 1 - \rho ) ( - 1 , r )$ we get $h _ { j } - h _ { y } = ( 1 - \rho ) ( 1 + r _ { j } ) \geq 0$ for every $j \neq y ,$ , so $h \in \mathcal { K } _ { y }$ . Since d $\in \mathcal { K } _ { y }$ by construction and $v - \bar { h } = \bar { d } / 2 \in { \mathcal K } _ { y }$ , for every $\eta \geq 0$

$$
M _ { j } ( z - \eta v ) \geq M _ { j } ( z - \eta h ) \geq M _ { j } ( z ) .
$$

The cross-entropy ordering in Eq. (7) follows from $\begin{array} { r } { L _ { \mathrm { C E } } ( z , y ) = \log ( 1 + \sum _ { j \neq y } e ^ { - M _ { j } ( z ) } ) } \end{array}$

## B.2 CLIPPED TEACHER AND THE TIGHT COMPENSATION BOUNDS

For the compensation in Eq. (9), let $\begin{array} { r } { \mathrm { T V } ( q , r ) = \frac { 1 } { 2 } \| q - r \| _ { 1 } } \end{array}$ , where $\| \cdot \| _ { 1 }$ <sub>1</sub> sums absolute coordinates. The tight size bounds are $\mathrm { T V } ( q , r ) / ( K - 1 ) \le \ell \le \mathrm { T V } ( q , r ) / 2 \colon$ compensation is controlled by the mismatch between the two conditional distributions.

Proof. From the threshold form of the projection, Eq. (B.1), $d _ { y } = - \ell$ and $d _ { j } = \operatorname* { m a x } ( r _ { j } - q _ { j } , - \ell )$ and from $\mathbf { 1 } ^ { \top } d = 0 ( \mathbf { A }$ ppendix B.3),

$$
\ell = \sum _ { j \neq y } \bigl [ q _ { j } - r _ { j } - \ell \bigr ] _ { + } , \qquad d _ { j } = r _ { j } - \operatorname* { m i n } ( q _ { j } , r _ { j } + \ell ) .
$$

The function $\begin{array} { r } { g ( a ) = a - \sum _ { j } [ q _ { j } - r _ { j } - a ] _ { + } } \end{array}$ is continuous and strictly increasing. ${ \mathrm { ~ f ~ } } q \neq r$ then $g ( 0 ) = - \mathrm { T V } ( q , r ) < 0$ and $\bar { g } ( \mathrm { T V } ( q , r ) ) > 0 ; \mathrm { i f } q = r$ the unique root is zero. Hence ℓ is unique. Summing the clipped mass gives $\begin{array} { r } { \sum _ { j } \widetilde { q } _ { j } = 1 - \ell . } \end{array}$

Let $T = \mathrm { T V } ( q , r )$ and let $k \le K - 2$ be the number of coordinates with positive deviation $( q _ { j } > r _ { j } )$ The root equation gives

$$
T = \ell + \sum _ { q _ { j } > r _ { j } } \operatorname* { m i n } ( q _ { j } - r _ { j } , \ell ) \ \leq \ ( k + 1 ) \ell \ \leq \ ( K - 1 ) \ell .
$$

If $T > 0 ,$ the set $A = \left\{ j : q _ { j } - r _ { j } > \ell \right\}$ is nonempty; with $m = | A |$

$$
T \geq \sum _ { j \in A } ( q _ { j } - r _ { j } ) = ( m + 1 ) \ell \geq 2 \ell .
$$

This proves the two compensation bounds. One positive deviation a shared as negative deviation by the other coordinates gives $\ell = T / 2 ; K - 2$ equal positive deviations with the last coordinate carrying all the negative deviation gives $\ell = T / ( \bar { K } - \bar { 1 } )$ ). Sufficiently small perturbations of the uniform distribution make both examples strictly positive, so both bounds are tight, and $\ell < 1 / 2$

Finally $d _ { j } - d _ { y } = [ r _ { j } - q _ { j } + \ell ] _ { + }$ . Substituting this into the margin change relative to the CE endpoint gives the extra-margin formula in Section 5.2.

## B.3 PROOF OF THEOREM 3

Let $n = K - 1$ . The Moreau decomposition gives $u = d + e$ with $e \in \mathcal { K } _ { u } ^ { \circ } , d ^ { \intercal } e = 0 .$ , and $\| u \| ^ { 2 } = \| d \| ^ { 2 } + \| e \| ^ { 2 }$ . Vectors of the polar cone $\mathcal { K } _ { y } ^ { \circ }$ have the form $( e ) _ { y } = \Lambda , ( e ) _ { j } = - \lambda _ { j } , \lambda _ { j } \ge 0$ $\begin{array} { r } { \Lambda = \sum _ { j \neq y } \lambda _ { j } } \end{array}$ . Put $\begin{array} { r } { Q = \sum _ { i } \lambda _ { j } ^ { 2 } \leq \Lambda ^ { 2 } , \operatorname { s o } \left\| e \right\| ^ { 2 } = \operatorname { \bar { Q } } + \Lambda ^ { 2 } } \end{array}$ . From $d ^ { \top } e = 0$ we have $\boldsymbol { u } ^ { \intercal } \boldsymbol { e } = \Vert \boldsymbol { e } \Vert ^ { 2 }$ ; since $u _ { y } = 0$ and $\textstyle \sum _ { j \neq y } u _ { j } { \dot { = } } 0$ , the $( \lambda _ { j } )$ may be centered inside the inner product:

$$
\| e \| ^ { 4 } = ( u ^ { \top } e ) ^ { 2 } = \left( \sum _ { j \neq y } u _ { j } \big ( \lambda _ { j } - \frac { \Lambda } { n } \big ) \right) ^ { 2 } \leq \| u \| ^ { 2 } \left( Q - \frac { \Lambda ^ { 2 } } { n } \right) \leq \frac { n - 1 } { 2 n } \| u \| ^ { 2 } ( Q + \Lambda ^ { 2 } ) ,\tag{B.2}
$$

where the last inequality is equivalent to $Q \leq \Lambda ^ { 2 } . \mathrm { I f } e \neq 0$ , dividing by $\left\| e \right\| ^ { 2 }$ gives $\textstyle \left\| e \right\| ^ { 2 } \leq { \frac { n - 1 } { 2 n } } \left\| u \right\| ^ { 2 }$ and hence $\begin{array} { r } { \left\| d \right\| ^ { 2 } \geq \frac { n + 1 } { 2 n } \left\| u \right\| ^ { 2 } = \gamma _ { K } \left\| u \right\| ^ { 2 } } \end{array}$ ; the case $e = 0$ is trivial. Finally $\boldsymbol { u } ^ { \top } \boldsymbol { d } = ( d + e ) ^ { \top } \boldsymbol { d } = \left\| \boldsymbol { d } \right\| ^ { 2 }$ which gives Eq. (10).

Pure conditional component. The polar representation gives $\mathbf { 1 } ^ { \top } e = 0$ , and with $\mathbf { 1 } ^ { \top } u = 0$ we get $\mathbf { 1 } ^ { \top } d = 0$ . Let $\ell = - d _ { y } ;$ then $\textstyle \sum _ { j \neq y } d _ { j } = { \dot { \ell } } .$ , and $w _ { \mathrm { c o n d } } = d - \ell a _ { 0 }$ satisfies $( w _ { \mathrm { c o n d } } ) _ { y } = 0$ and $\mathbf { 1 } ^ { \top } w _ { \mathrm { c o n d } } = 0 , \mathrm { i . e . } w _ { \mathrm { c o n d } } \in \mathcal { L } _ { y } ;$ moreover $a _ { 0 } ^ { \top } d = \ell ( 1 + 1 / n ) = \ell \left\| a _ { 0 } \right\| ^ { 2 }$ shows $w _ { \mathrm { c o n d } } \perp a _ { 0 }$ . Since $u \perp a _ { 0 } , u ^ { \top } w _ { \mathrm { c o n d } } = u ^ { \top } d = \left\| d \right\| ^ { 2 }$ . For u $\neq 0 ,$ , Cauchy–Schwarz gives

$$
\left\| w _ { \mathrm { c o n d } } \right\| ^ { 2 } \geq \frac { \left( u ^ { \top } w _ { \mathrm { c o n d } } \right) ^ { 2 } } { \left\| u \right\| ^ { 2 } } = \frac { \left\| d \right\| ^ { 4 } } { \left\| u \right\| ^ { 2 } } \geq \gamma _ { K } \left\| d \right\| ^ { 2 } \geq \gamma _ { K } ^ { 2 } \left\| u \right\| ^ { 2 } .
$$

Tightness. Take $a > 0 , u _ { y } = 0 .$ , one non-target coordinate $u _ { j _ { * } } = - a , $ , and the remaining $n - 1$ coordinates equal to $a / ( n - 1 )$ . By Eq. $( \mathrm { B } . 1 ) , \breve { \zeta } = - a / 2$ , and the projection is $d _ { y } = d _ { j _ { * } } = - a / 2$ $d _ { j } = a / ( n - 1 )$ for $j \neq y , j _ { * }$ . A direct computation gives $\| d \| ^ { 2 } / \| u \| ^ { 2 } = ( n + 1 ) / ( 2 n ) = \gamma _ { K }$ and $w _ { \mathrm { c o n d } } = \gamma _ { K } u .$ , so both bounds are attained simultaneously; for sufficiently small a this u is the difference of two strictly positive conditional distributions.

## B.4 THE ALIGNMENT PART OF THEOREM 3

Keep $u = d + e$ and the polar representation. Since $\begin{array} { r } { u _ { y } = 0 , ( e ) _ { y } = - d _ { y } = \ell = \sum _ { j } \lambda _ { j } , } \end{array}$ , so $\begin{array} { r } { \left\| e \right\| ^ { 2 } = \ell ^ { 2 } + \sum _ { j } \lambda _ { j } ^ { 2 } \le 2 \ell ^ { 2 } } \end{array}$ . Write $\gamma = \gamma _ { K }$ . From $w _ { \mathrm { c o n d } } \perp a _ { 0 } , u \perp a _ { 0 }$ , and $\left. a _ { 0 } \right. ^ { 2 } = ( n + 1 ) / n = 2 \gamma$

$$
\begin{array} { r } { \left\| \boldsymbol d \right\| ^ { 2 } = \left\| \boldsymbol w _ { \mathrm { c o n d } } \right\| ^ { 2 } + 2 \gamma \ell ^ { 2 } , \qquad \boldsymbol u ^ { \top } \boldsymbol w _ { \mathrm { c o n d } } = \left\| \boldsymbol d \right\| ^ { 2 } . } \end{array}
$$

Therefore

$$
\begin{array} { r l } & { \gamma \left. u \right. ^ { 2 } = \gamma \left. d \right. ^ { 2 } + \gamma \left. e \right. ^ { 2 } \leq \gamma \left. d \right. ^ { 2 } + 2 \gamma \ell ^ { 2 } } \\ & { \qquad = \gamma \left. d \right. ^ { 2 } + \left. d \right. ^ { 2 } - \left. w _ { \mathrm { c o n d } } \right. ^ { 2 } = \left( 1 + \gamma \right) u ^ { \top } w _ { \mathrm { c o n d } } - \left. w _ { \mathrm { c o n d } } \right. ^ { 2 } . } \end{array}\tag{B.3}
$$

Completing the square in this inequality gives the equivalent form

$$
\left. w _ { \mathrm { c o n d } } - \frac { 1 + \gamma } { 2 } u \right. ^ { 2 } \leq \frac { ( 1 - \gamma ) ^ { 2 } } { 4 } \left. u \right. ^ { 2 } .\tag{B.4}
$$

For $u \ne 0$ we have $w _ { \mathrm { c o n d } } \neq 0 .$ , and $( 1 + \gamma ) u ^ { \top } w _ { \mathrm { c o n d } } \geq \gamma \left. u \right. ^ { 2 } + \left. w _ { \mathrm { c o n d } } \right. ^ { 2 } \geq 2 \sqrt { \gamma } \left. u \right. \left. w _ { \mathrm { c o n d } } \right.$ gives Eq. (11). Since $\gamma \in \dot { ( 1 / 2 , 3 / 4 ] }$ and $2 \sqrt { \gamma } / ( 1 + \gamma )$ is increasing on $( 0 , 1 ]$ , the uniform lower bound is $2 { \sqrt { 2 } } / 3$

## B.5 THE GEOMETRIC EXAMPLE IN FIGURE 3

The illustration uses $K \ = \ 4$ , with the correct class first, $r ~ = ~ ( 0 . 2 0 , 0 . 3 0 , 0 . 5 0 )$ and $q =$ (0.30, 0.32, 0.38). These give

$$
u = ( 0 , - 0 . 1 0 , - 0 . 0 2 , 0 . 1 2 ) , \qquad d = ( - 0 . 0 5 , - 0 . 0 5 , - 0 . 0 2 , 0 . 1 2 ) , \qquad \ell = 0 . 0 5 .
$$

With $a _ { 0 } = ( - 1 , 1 / 3 , 1 / 3 , 1 / 3 )$ , the pure conditional component is

$$
w _ { \mathrm { c o n d } } = ( 0 , - 1 / 1 5 , - 1 1 / 3 0 0 , 3 1 / 3 0 0 ) .
$$

It is orthogonal to $a _ { 0 }$ and forms an angle of approximately $1 1 . 5 4 ^ { \circ }$ with u. Panel (a) uses an oblique view to display both orthogonal components; panel (b) shows the angle within the conditional plane. This is a concrete illustration of the decomposition and alignment in Theorem 3.

## B.6 A LOCAL LOSS REPRESENTATION OF THE CLIPPED TEACHER

The clipped direction in Eq. (9) has an exact local loss interpretation. Fix a reference output z<sub>0</sub>, compute ℓ and $\widetilde { q }$ there, and hold both fixed. Since $\begin{array} { r } { \sum _ { j } \widetilde { q } _ { j } = 1 { \stackrel { - } { - } } \ell > 0 } \end{array}$ , the vector $\bar { q } = \widetilde { q } / ( 1 - \ell )$ is a positive conditional distribution. Define

$$
\Lambda _ { z _ { 0 } } ( z ) = ( 1 - \ell ) \mathrm { K L } ( \bar { q } \| r ( z ) ) - \ell \log \mathrm { i } \ell \rho ( z ) .\tag{B.5}
$$

Then

$$
\nabla _ { z } \Lambda _ { z _ { 0 } } ( z ) \vert _ { z = z _ { 0 } } = d ( z _ { 0 } ) .\tag{B.6}
$$

The two terms learn the normalized clipped teacher and increase correct-class log-odds, respectively, with weights determined by the retained and clipped probability masses.

Proof. Using logit $\begin{array} { r } { \rho ( z ) = z _ { y } - \log { \sum _ { j \neq y } e ^ { z _ { j } } } } \end{array}$ and differentiating with the reference quantities frozen gives

$$
\partial _ { z _ { y } } \Lambda _ { z _ { 0 } } = - \ell , \qquad \partial _ { z _ { j } } \Lambda _ { z _ { 0 } } = ( 1 - \ell ) ( r _ { j } ( z ) - \bar { q } _ { j } ) + \ell r _ { j } ( z ) = r _ { j } ( z ) - \widetilde { q } _ { j } .
$$

$\mathrm { A t } ~ z = z _ { 0 }$ these are precisely the coordinates of $d ( z _ { 0 } )$ in Eq. (9).

## B.7 THE EXACT SIGNAL REMOVED AND THE SIZE OF COMPENSATION

The retention bound can be refined using the coordinates actually clipped. Let $\lambda = ( \lambda _ { j } ) _ { j \neq y }$ collect the clipped masses, defined by

$$
\lambda _ { j } = [ q _ { j } - r _ { j } - \ell ] _ { + } , \qquad e = u - d = ( \ell , - \lambda ) , \qquad \sum _ { j \neq y } \lambda _ { j } = \ell .
$$

Moreau orthogonality gives the exact amount removed from the conditional first-order signal:

$$
\| u \| ^ { 2 } - u ^ { \top } d = \| u \| ^ { 2 } - \| d \| ^ { 2 } = \ell ^ { 2 } + \sum _ { j \neq y } \lambda _ { j } ^ { 2 } .\tag{B.7}
$$

For $u \neq 0 ,$ , let $m = | \{ j : \lambda _ { j } > 0 \} |$ . Then

$$
\ell ^ { 2 } ( 1 + 1 / m ) \leq \| u \| ^ { 2 } - \| d \| ^ { 2 } \leq 2 \ell ^ { 2 } .\tag{B.8}
$$

The lower bound follows from Cauchy–Schwarz applied to the $m$ positive coordinates of $\lambda ;$ the upper bound follows from their nonnegativity and sum ℓ.

The extra task-side change is also controlled by the current conditional mismatch. Write $T =$ $\begin{array} { r } { \mathrm { T V } ( q , r ) = \frac { 1 } { 2 } \| q - r \| _ { 1 } } \end{array}$ and let $\Delta M _ { i } ^ { \mathrm { e x t r a } }$ be the TPKD margin minus the corresponding CE-step margin. Because $r _ { j } - q _ { j } \leq T$ and $\ell \le T / 2$ , Eq. (9) gives, for every $\eta \geq 0$

$$
0 \leq \Delta M _ { j } ^ { \mathrm { e x t r a } } = \frac { \eta } { 2 } [ r _ { j } - q _ { j } + \ell ] _ { + } \leq \frac { 3 \eta T } { 4 } .\tag{B.9}
$$

Let $z ^ { \prime } = z - \eta h$ and $\rho ^ { \prime } = \mathrm { s o f t m a x } ( z ^ { \prime } ) _ { y }$ . Convexity of CE and the margin ordering imply

$$
\begin{array} { r l } & { 0 \leq L _ { \mathrm { C E } } ( z ^ { \prime } ) - L _ { \mathrm { C E } } ( z ^ { \prime } - \eta d / 2 ) \leq \displaystyle \frac { \eta } { 2 } h ( z ^ { \prime } ) ^ { \top } d } \\ & { \quad \quad \quad = \displaystyle \frac { \eta } { 2 } ( 1 - \rho ^ { \prime } ) \sum _ { j \neq y } r _ { j } ( z ^ { \prime } ) ( d _ { j } - d _ { y } ) } \\ & { \quad \quad \quad \leq \displaystyle \frac { 3 \eta } { 4 } ( 1 - \rho ^ { \prime } ) T \leq \displaystyle \frac { 3 \eta } { 4 } ( 1 - \rho ) T . } \end{array}\tag{B.10}
$$

Thus, at a fixed output step size, the additional confidence effect vanishes as the conditional mismatch vanishes.

## B.8 OPTIMAL CONDITIONAL PROGRESS UNDER TWO LOCAL BUDGETS

The nearest-safe projection also maximizes conditional progress under two complementary comparisons.

Local quadratic model. Let $L > 0$ be a smoothness bound for $L _ { \mathrm { c o n d } }$ and let x be an output displacement, so that $z ^ { + } = z - x$ . Smoothness gives

$$
L _ { \mathrm { c o n d } } ( z ) - L _ { \mathrm { c o n d } } ( z - x ) \geq Q _ { u } ( x ) , \qquad Q _ { u } ( x ) = u ^ { \top } x - \frac { L } { 2 } \| x \| ^ { 2 } .
$$

Completing the square yields

$$
Q _ { u } ( x ) = \frac { \| u \| ^ { 2 } } { 2 L } - \frac { L } { 2 } \| x - u / L \| ^ { 2 } .
$$

The unconstrained maximizer is $u / L ;$ over $x \in \mathcal { K } _ { y }$ it is $d / L$ , by positive homogeneity of cone projection. Consequently,

$$
\operatorname* { m a x } _ { x \in { \mathcal { K } } _ { y } } Q _ { u } ( x ) = { \frac { \| d \| ^ { 2 } } { 2 L } } , \qquad \operatorname* { m a x } _ { x } Q _ { u } ( x ) = { \frac { \| u \| ^ { 2 } } { 2 L } } .\tag{B.11}
$$

For $u \neq 0 ,$ , the ratio of these optimal guaranteed decreases is $\| d \| ^ { 2 } / \| u \| ^ { 2 } \geq \gamma _ { K }$ , with equality in the tightness construction of Appendix B.3.

Fixed displacement length. For $u \ne 0$ and a budget $S > 0$

$$
\operatorname* { m a x } _ { \substack { x \in { \mathcal K } _ { y } , \| x \| \leq S } } u ^ { \top } x = S \| d \| , \qquad x ^ { * } = S \frac { d } { \| d \| } .\tag{B.12}
$$

To prove this, use $u = d + e $ , where $e \in \mathcal { K } _ { y } ^ { \circ }$ : for any feasible x,

$$
u ^ { \top } x = d ^ { \top } x + e ^ { \top } x \leq \| d \| \| x \| \leq S \| d \| .
$$

The stated $x ^ { * }$ attains equality. Without the cone constraint the optimum is $S \Vert u \Vert$ , so the safe-tounconstrained ratio is $\| \dot { d } \| / \| \dot { u } \| \ge \sqrt { \gamma _ { K } }$ , again tight. This is the comparison at equal displacement length; Eq. (10) compares the original and projected directions at equal step size.

## C PROOFS SUPPORTING JOINT FITTING AND THE CE COMPARISON

We prove Theorem 4, including its residual rate, and the uniform-limit comparison stated in Section 5.3.

## C.1 COMPLETE-UPDATE ALIGNMENT IN THEOREM 4

Write $u = d + e$ for the Moreau decomposition. Then $d ^ { \top } e = 0 , e \in \mathcal { K } _ { y } ^ { \circ }$ , and Theorem 3 implies

$$
\left\| e \right\| ^ { 2 } \leq \frac { 1 - \gamma _ { K } } { \gamma _ { K } } \left\| d \right\| ^ { 2 } .
$$

We have $h \in \mathcal { K } _ { y }$ , so $h ^ { \top } e \leq 0$ . Also, using the CE form $h = ( 1 - \rho ) ( - 1 , r )$

$$
h ^ { \top } d = ( 1 - \rho ) \sum _ { j \neq y } r _ { j } ( d _ { j } - d _ { y } ) \geq 0 .
$$

Set $g _ { \Phi } = 4 h + u$ and abbreviate $\gamma = \gamma _ { K }$ . Expanding gives

$$
\begin{array} { l } { { g _ { \Phi } ^ { \top } v - \| v \| ^ { 2 } - \frac { \gamma } { 4 } \| g _ { \Phi } \| ^ { 2 } = ( 3 - 4 \gamma ) \| h \| ^ { 2 } + 2 ( 1 - \gamma ) h ^ { \top } d + ( 1 - 2 \gamma ) h ^ { \top } e } } \\ { { \displaystyle \qquad +  \frac { 1 } { 4 } [ ( 1 - \gamma ) \| d \| ^ { 2 } - \gamma \| e \| ^ { 2 } ] \geq 0 . } } \end{array}\tag{C.1}
$$

Every term is nonnegative because $1 / 2 < \gamma \leq 3 / 4$ . The arithmetic–geometric mean inequality gives

$$
g _ { \Phi } ^ { \top } v \ge \left\| v \right\| ^ { 2 } + \frac { \gamma } { 4 } \left\| g _ { \Phi } \right\| ^ { 2 } \ge \sqrt { \gamma } \left\| g _ { \Phi } \right\| \left\| v \right\| ,
$$

which proves Eq. (12).

Sharpness. Fix any strictly positive pair $r , q$ realizing the equality construction in Appendix $\operatorname { B } . 3 ;$ for example, perturb the uniform conditional distribution by a sufficiently small one-negative-coordinate, equal-positive-remainder difference. Keep this $\boldsymbol { u } = ( 0 , r - q )$ fixed and let the student correct-class probability $\rho$ tend to one. Then $h  0 , g _ { \Phi } $ u and $v  d / 2$ , whence

$$
\cos \angle ( g _ { \Phi } , v ) \longrightarrow \frac { u ^ { \top } d } { \| u \| \| d \| } = \frac { \| d \| } { \| u \| } = \sqrt { \gamma } .
$$

Thus no larger uniform angle constant holds for strictly positive probabilities. This construction also makes the gap in Eq. (C.1) tend to zero.

Batch form. Let col denote vertical stacking. For $W _ { \Phi } = \mathrm { c o l } ( g _ { \Phi , i } ) / \sqrt { N }$ and $V = \mathrm { c o l } ( v _ { i } ) / \sqrt { N }$ summing the per-example inequality yields

$$
W _ { \Phi } ^ { \top } V \ge \left\| V \right\| ^ { 2 } + \frac { \gamma } { 4 } \left\| W _ { \Phi } \right\| ^ { 2 } \ge \sqrt { \gamma } \left\| W _ { \Phi } \right\| \left\| V \right\| .\tag{C.2}
$$

## C.2 DESCENT, CONVERGENCE AND THE RESIDUAL RATE

The logit Hessians of the cross-entropy and conditional KL are softmax covariance matrices, with spectral norm at most $1 / 2$ . Therefore $\Phi = 4 L _ { \mathrm { C E } } + L _ { \mathrm { c o n d } }$ is $5 / 2$ -smooth. For the specified output step,

$$
\begin{array} { l } { \displaystyle \Phi ( z - \eta v ) \leq \Phi ( z ) - \eta g _ { \Phi } ^ { \top } v + \frac 5 4 \eta ^ { 2 } \left. v \right. ^ { 2 } } \\ { \displaystyle \qquad \leq \Phi ( z ) - \frac { \eta \gamma _ { K } } { 4 } \left. g _ { \Phi } \right. ^ { 2 } - \eta \left( 1 - \frac 5 4 \eta \right) \left. v \right. ^ { 2 } . } \end{array}\tag{C.3}
$$

This proves Eq. (13). When $0 < \eta \le 4 / 5$ , the last term is nonpositive. For any integer $T \geq 1$ summing over $k = 0 , \ldots , T - 1$ and using $\Phi \ge 0$ gives

$$
\sum _ { k < T } \left. g _ { \Phi , k } \right. ^ { 2 } \leq \frac { 4 \Phi ( z _ { 0 } ) } { \eta \gamma _ { K } } .\tag{C.4}
$$

Thus $g _ { \Phi , k } \to 0$ . Since $u _ { k , y } = 0$ and $( g _ { \Phi , k } ) _ { y } = - 4 ( 1 - \rho _ { k } )$ , we obtain $\rho _ { k } \to 1$ . The identity $h _ { k } = ( 1 - \rho _ { k } ) ( - 1 , r _ { k } )$ then gives $h _ { k } \ \to \ 0 ,$ , so $u _ { k } = g _ { \Phi , k } - 4 h _ { k } \to 0$ and $r _ { k } \ \to \ q$ . Finally $\| d _ { k } \| \leq \| u _ { k } \|$ implies $v _ { k } \to 0$

Joint and component residuals. Dividing Eq. (C.4) by T gives

$$
\frac { 1 } { T } \sum _ { k < T } \| g _ { \Phi , k } \| ^ { 2 } \leq \frac { 4 \Phi ( z _ { 0 } ) } { \eta \gamma _ { K } T } .\tag{C.5}
$$

The two learning signals inherit quantitative bounds. Since $h = ( 1 - \rho ) ( - 1 , r )$ and $\begin{array} { r } { ( g _ { \Phi } ) _ { y } = } \end{array}$ $- 4 ( 1 - \rho )$

$$
\| h \| ^ { 2 } = ( 1 - \rho ) ^ { 2 } ( 1 + \| r \| ^ { 2 } ) \leq 2 ( 1 - \rho ) ^ { 2 } \leq \frac { 1 } { 8 } \| g _ { \Phi } \| ^ { 2 } .
$$

Also $u = g _ { \Phi } - 4 h$ implies $\| u \| ^ { 2 } \leq 2 \| g _ { \Phi } \| ^ { 2 } + 3 2 \| h \| ^ { 2 } \leq 6 \| g _ { \Phi } \| ^ { 2 }$ . Hence

$$
\frac { 1 } { T } \sum _ { k < T } \left( \| h _ { k } \| ^ { 2 } + \frac { 1 } { 1 2 } \| u _ { k } \| ^ { 2 } \right) \le \frac { 5 \Phi ( z _ { 0 } ) } { 2 \eta \gamma _ { K } T } .\tag{C.6}
$$

These bounds quantify the vanishing time-averaged squared gradient residuals of the joint, label and conditional signals.

## C.3 THE CE CONDITIONAL LIMIT STATED IN SECTION 5.3

Let $n = K - 1 , \mathbf { 1 } _ { n } = ( 1 , \ldots , 1 ) \in \mathbb { R } ^ { n }$ , and $\nu = \mathbf { 1 } _ { n } / n$ . From finite initial logits, CE gradient flow and the fixed-step rule with $0 < \eta \leq 1 / 2 0$ have limits $p _ { y }  1$ and $r  \nu .$ This differs from the TPKD conditional limit only when $q \neq \nu .$

Proof. Let $z _ { \lnot y } \in \mathbb { R } ^ { n }$ contain the non-target logits, set $a = z _ { \lnot y } - ( \mathbf { 1 } _ { n } ^ { \top } z _ { \lnot y } ) \mathbf { 1 } _ { n } / n$ , and define on the zero-sum subspace

$$
{ \cal F } ( a ) = \log \sum _ { j } e ^ { a _ { j } } - \frac { 1 } { n } \sum _ { j } a _ { j } - \log n = \mathrm { K L } ( \nu \| r ) , \qquad \nabla { \cal F } = r - \nu .
$$

Let $t \geq 0$ denote flow time, with dots denoting time derivatives. Along the cross-entropy gradient flow, $\dot { z } _ { y } = 1 - \rho$ and $\dot { z } _ { j } = - ( 1 - \rho ) r _ { j }$ , so

$$
\begin{array} { r } { \dot { \boldsymbol { a } } = - ( 1 - \rho ) \nabla F , \qquad \dot { \boldsymbol { F } } = - ( 1 - \rho ) \left\| \boldsymbol { r } - \boldsymbol { \nu } \right\| ^ { 2 } , } \end{array}
$$

which gives the conditional dynamics of CE. Let $\begin{array} { r } { s ( t ) = \int _ { 0 } ^ { t } ( 1 - \rho ( \tau ) ) d \tau . \operatorname { I f } s ( \infty ) < \infty , } \end{array}$ , every logit has finite total variation, all logits converge to finite values, and $1 - \rho$ converges to a positive number, contradicting the finiteness of the integral; hence $s ( t ) \to \infty$ . In the effective time $s , d a / d s = - \nabla F$ The sublevel sets of $F$ on the zero-sum subspace are bounded and its Hessian dia $\mathrm { g } ( \boldsymbol { r } ) - \boldsymbol { r } \boldsymbol { r } ^ { \top }$ , where $\mathrm { d i a g } ( r )$ is the diagonal matrix with diagonal $r ,$ is positive definite on that subspace, so on the compact sublevel set containing the whole trajectory there is $\mu > 0$ such that $F$ is µ-strongly convex. The unique minimizer is $a = 0$ , hence $r  \nu$ . Moreover $z _ { y } = z _ { y } ( 0 ) + s ( t ) \to \infty$ while all non-target logits are nonincreasing, so $\rho  1$

In the discrete case let $s _ { k } = \eta ( 1 - \rho _ { k } ) , \mathrm { s o } a _ { k + 1 } = a _ { k } - s _ { k } \nabla F ( a _ { k } )$ . The gradient of F is ${ \scriptstyle { \frac { 1 } { 2 } } - \mathrm { L i p s c h i t z } }$ hence

$$
F ( a _ { k + 1 } ) \leq F ( a _ { k } ) - s _ { k } ( 1 - { s _ { k } } / { 4 } ) \left\| \nabla F ( a _ { k } ) \right\| ^ { 2 } .
$$

For $0 < \eta \leq 1 / 2 0$ the sequence stays in the same compact sublevel set. $\begin{array} { r } { \operatorname { I f } \sum _ { k } s _ { k } < \infty } \end{array}$ , the same finite-total-variation argument gives a positive limit of $1 - \rho _ { k }$ , a contradiction; hence $\textstyle \sum _ { k } s _ { k } = \infty$ By strong convexity on that compact set and $\left\| \nabla F \right\| ^ { 2 } \geq 2 \mu F .$

$$
F ( a _ { k + 1 } ) \leq \left[ 1 - 2 \mu s _ { k } ( 1 - { s _ { k } } / { 4 } ) \right] F ( a _ { k } ) ,
$$

so $F ( a _ { k } )  0$ and $r _ { k } \to \nu .$ . At the same time $\begin{array} { r } { z _ { k , y } = z _ { 0 , y } + \sum _ { j < k } s _ { j } \to } \end{array}$ ∞ and all non-target logits are nonincreasing, so $\rho _ { k } \to 1$

## C.4 WHEN A COMPLETE STEP LOWERS ITS OWN CONDITIONAL ERROR

The effect of CE on the teacher’s conditional objective has the exact form

$$
u ^ { \top } h = ( 1 - \rho ) ( r - q ) ^ { \top } r = \frac { 1 - \rho } { 2 } \left( \| r - q \| ^ { 2 } + \| r \| ^ { 2 } - \| q \| ^ { 2 } \right) .\tag{C.7}
$$

This follows by expanding $\| r - q \| ^ { 2 }$ . The projected conditional contribution adds $\| d \| ^ { 2 } / 2 ,$ so for the complete direction $v = h + d / 2$

$$
u ^ { \top } v = u ^ { \top } h + \frac { 1 } { 2 } \| d \| ^ { 2 } \geq - \| h \| \| u \| + \frac { \gamma _ { K } } { 2 } \| u \| ^ { 2 } .\tag{C.8}
$$

In particular, $\| u \| > 0$ and $\| h \| / \| u \| < \gamma _ { K } / 2$ imply $c = u ^ { \top } v > 0$ . The conditional KL has logit Hessian dia $\mathrm { g } ( \ddot { r } ) - r r ^ { \top }$ on the non-target coordinates, with spectral norm at most $1 / 2$ . Thus

$$
L _ { \mathrm { c o n d } } ( z - \eta v ) \leq L _ { \mathrm { c o n d } } ( z ) - \eta c + \frac { \eta ^ { 2 } } { 4 } \| v \| ^ { 2 } .
$$

Consequently the sufficient conditions

$$
\| u \| > 0 , \qquad \frac { \| h \| } { \| u \| } < \frac { \gamma _ { K } } { 2 } , \qquad 0 < \eta < \frac { 4 u ^ { \top } v } { \| v \| ^ { 2 } }\tag{C.9}
$$

give $L _ { \mathrm { c o n d } } ( z - \eta v ) < L _ { \mathrm { c o n d } } ( z )$ . The complete update then reduces its own conditional error while retaining the label-step margin gains in Eq. (7).

## C.5 GENERAL POSITIVE CONDITIONAL WEIGHTS

For positive coefficients $\beta , w ,$ , consider the family

$$
v _ { \beta } = h + \beta d , \qquad \Phi _ { w } = w L _ { \mathrm { C E } } + L _ { \mathrm { c o n d } } .
$$

For every $\beta > 0 , v _ { \beta } - h = \beta d \in \mathcal { K } _ { y }$ , so the label-step margin ordering holds for all $\eta \geq 0$ . Moreover, $v _ { \beta }$ is the unique minimizer of $\begin{array} { r } { \frac { 1 } { 2 } \| a - ( h + \beta u ) \| ^ { 2 } } \end{array}$ over candidate directions $a \in \mathbb { R } ^ { K }$ with $a - h \in \mathcal K _ { y }$ A sufficient joint-learning condition is

$$
4 w \beta \gamma _ { K } > 1 .\tag{C.10}
$$

To make the resulting constants explicit, set

$$
c _ { w , \beta } = \frac { w + \beta \gamma _ { K } - \sqrt { ( w - \beta \gamma _ { K } ) ^ { 2 } + 1 } } { 2 } > 0 , \qquad C _ { \beta } = \operatorname * { m a x } \{ 1 , \beta ^ { 2 } \} .
$$

For a fixed positive conditional target and finite initial logits, every constant step

$$
0 < \eta \leq \frac { c _ { w , \beta } } { ( w + 1 ) C _ { \beta } }\tag{C.11}
$$

satisfies

$$
\Phi _ { w } ( z - \eta v _ { \beta } ) \leq \Phi _ { w } ( z ) - \frac { \eta c _ { w , \beta } } { 2 } \big ( \| h \| ^ { 2 } + \| u \| ^ { 2 } \big ) .\tag{C.12}
$$

It follows that

$$
\frac { 1 } { T } \sum _ { k < T } \bigl ( \| h _ { k } \| ^ { 2 } + \| u _ { k } \| ^ { 2 } \bigr ) \le \frac { 2 \Phi _ { w } ( z _ { 0 } ) } { \eta c _ { w , \beta } T } , \qquad p _ { k , y } \to 1 , \quad r _ { k } \to q , \quad v _ { \beta , k } \to 0 .
$$

Proof. Write $H = \left\| h \right\|$ and $S = \| u \|$ . The identities $h ^ { \top } d \geq 0$ and $u ^ { \top } d \geq \gamma _ { K } \| u \| ^ { 2 }$ give

$$
\begin{array} { r } { \nabla \Phi _ { w } ^ { \top } { v } _ { \beta } = w \| h \| ^ { 2 } + w \beta h ^ { \top } d + u ^ { \top } h + \beta u ^ { \top } d \geq w H ^ { 2 } - H S + \beta \gamma _ { K } S ^ { 2 } . } \end{array}
$$

The matrix of this quadratic form is $\left( \begin{array} { c c } { { w } } & { { - 1 / 2 } } \\ { { - 1 / 2 } } & { { \beta \gamma _ { K } } } \end{array} \right)$ . It is positive definite under Eq. (C.10), with smallest eigenvalue $c _ { w , \beta }$ , so the last expression is at least $c _ { w , \beta } ( H ^ { 2 } + S ^ { 2 } )$ . The potential is $( w + 1 ) / 2 \cdot$ smooth and

$$
\| v _ { \beta } \| ^ { 2 } \leq 2 H ^ { 2 } + 2 \beta ^ { 2 } S ^ { 2 } \leq 2 C _ { \beta } ( H ^ { 2 } + S ^ { 2 } ) .
$$

The descent lemma therefore gives

$$
\Phi _ { w } ( z - \eta v _ { \beta } ) \le \Phi _ { w } ( z ) - \eta \left[ c _ { w , \beta } - \frac { ( w + 1 ) C _ { \beta } \eta } { 2 } \right] ( H ^ { 2 } + S ^ { 2 } ) ,
$$

which implies $\mathrm { E q . } \left( \mathrm { C . } 1 2 \right)$ . Summation and $\Phi _ { w } \ge 0$ show that $h _ { k } , u _ { k } \to 0$ . The identity $h _ { k , y } =$ $p _ { k , y } - 1 \operatorname { g i v e s } p _ { k , y }  1$ , while $u _ { k } = ( 0 , r _ { k } - q )$ gives $r _ { k } \to q ;$ finally $\| d _ { k } \| \leq \| u _ { k } \|$ yields $v _ { \beta , k } \to 0 .$

## D PROOFS SUPPORTING EXACT REALIZATION AND PARAMETER LEARNING

This appendix proves Proposition 1, Theorem 5, and the measured-angle test stated in Section 5.4.

## D.1 EXACT HEAD REALIZATION

For N frozen features of dimension D, absorb the classifier bias into the last feature column. Let $X \in \mathbb { R } ^ { N \times D }$ be the feature matrix, $A \in \mathbb { R } ^ { D \times K }$ the classifier weights, and $Z = X A$ the batch logits. Let $V _ { \mathrm { r o w } }$ have rows $v _ { i } ^ { \top } , X ^ { \dag }$ denote the pseudoinverse, $I _ { N }$ the $N \times N$ identity, and $\Delta A$ the head-weight displacement. The exact construction in Proposition 1 is

$$
\Delta A = - \eta X ^ { \dagger } V _ { \mathrm { r o w } } , \qquad X ( A + \Delta A ) = Z - \eta V _ { \mathrm { r o w } } .\tag{D.1}
$$

Full row rank gives $X X ^ { \dag } = I _ { N }$ , so $X ( A - \eta X ^ { \dagger } V _ { \mathrm { r o w } } ) = Z - \eta V _ { \mathrm { r o w } }$ . Let $H \in \mathbb { R } ^ { D \times K }$ satisfy $X H = 0$ . Every other feasible displacement i $\mathsf { s } - \eta X ^ { \dagger } V _ { \mathrm { r o w } } + H$ . The columns of $X ^ { \dagger } V _ { \mathrm { r o w } }$ lie in the column space of $X ^ { \top }$ and are orthogonal to the columns of $H ;$ ; writing $\| \cdot \| _ { F }$ for the Frobenius norm gives

$$
\lVert - \eta X ^ { \dagger } V _ { \mathrm { r o w } } + H \rVert _ { F } ^ { 2 } = \eta ^ { 2 } \lVert X ^ { \dagger } V _ { \mathrm { r o w } } \rVert _ { F } ^ { 2 } + \lVert H \rVert _ { F } ^ { 2 } .
$$

This proves the unique minimum-norm claim and the exact inheritance of the same step’s output orderings.

Repeated realization on fixed features. Let the same full-row-rank X be used at every iteration, with fixed positive conditional targets for its rows. For $A _ { k + 1 } = A _ { k } - \eta X ^ { \dagger } V _ { \mathrm { r o w } , k }$

$$
Z _ { k + 1 } = X A _ { k + 1 } = Z _ { k } - \eta V _ { \mathrm { r o w } , k } .
$$

Thus every row follows the output iteration of Theorem 4. With finite initial logits and $0 < \eta \leq 4 / 5$ each row has $p _ { i , k , y _ { i } }  1$ and $r _ { i , k } \to q _ { i }$

Computational cost. For $N \leq D$ , forming and factorizing $X X ^ { \top }$ and applying $X ^ { \dagger } V _ { \mathrm { r o w } }$ costs $O ( N ^ { \bar { 2 } } D + N ^ { 3 } + N D K )$ . With fixed X, its pseudoinverse can be reused, leaving $O ( N D K )$ work per head displacement.

Parameter steps and batch objectives. Let $g _ { \mathrm { T } } = N ^ { - 1 } \sum _ { i } { J _ { i } ^ { \top } v _ { i } }$ and $\begin{array} { r } { g _ { \mathrm { C } } = N ^ { - 1 } \sum _ { i } J _ { i } ^ { \top } ( h _ { i } + } \end{array}$ $\ell _ { i } a _ { 0 , i } / 2 )$ be the respective parameter gradients of TPKD and compensation only. Define their conditional increment $\delta g = g _ { \mathrm { T } } - g _ { \mathrm { C } }$ , the batch-average conditional loss $\begin{array} { r } { \overline { { L } } _ { \mathrm { c o n d } } ( \theta ) = N ^ { - 1 } \sum _ { i } L _ { \mathrm { c o n d } } ( z _ { i } ( \theta ) ) } \end{array}$ , and the batch-average joint potential $\begin{array} { r } { \Psi ( \theta ) = \bar { N } ^ { - 1 } \sum _ { i } \Phi ( z _ { i } ( \theta ) ) } \end{array}$ . Under the respective conditions of Theorem 5, for sufficiently small $\eta > 0$ its conclusions are

$$
\begin{array} { r l } & { \mathrm { ( i ) } \quad \delta g \neq 0 , \qquad \overline { { L } } _ { \mathrm { c o n d } } ( \theta - \eta g _ { \mathrm { T } } ) < \overline { { L } } _ { \mathrm { c o n d } } ( \theta - \eta g _ { \mathrm { C } } ) , } \\ & { \mathrm { ( i i ) } \quad \Psi ( \theta - \eta g _ { \mathrm { T } } ) < \Psi ( \theta ) . } \end{array}\tag{D.2}
$$

## D.2 THE ANGLE–SPECTRUM ARGUMENT

Batch signals and restricted spectra. Here col denotes vertical concatenation. For the batch Jacobian $J = \mathrm { c o l } ( J _ { i } ) / \sqrt { N }$ , define $G = J J ^ { \top }$ and the normalized signal stacks

$$
( U , R ) = { \frac { 1 } { \sqrt { N } } } { \big ( } \mathrm { c o l } ( u _ { i } ) , \mathrm { c o l } ( w _ { \mathrm { c o n d } , i } ) { \big ) } , \qquad ( W , V ) = { \frac { 1 } { \sqrt { N } } } { \big ( } \mathrm { c o l } ( g _ { \Phi , i } ) , \mathrm { c o l } ( v _ { i } ) { \big ) } .
$$

The first pair describes conditional learning and the second joint descent. Equation (8) yields

$$
\delta \boldsymbol { g } = \boldsymbol { J } ^ { \top } \boldsymbol { R } / 2 , \qquad \nabla _ { \boldsymbol { \theta } } \overline { { \boldsymbol { L } } } _ { \mathrm { c o n d } } = \boldsymbol { J } ^ { \top } \boldsymbol { U } , \qquad \nabla _ { \boldsymbol { \theta } } \Psi = \boldsymbol { J } ^ { \top } \boldsymbol { W } , \qquad g _ { \mathrm { T } } = \boldsymbol { J } ^ { \top } \boldsymbol { V } .
$$

For either pair let Q have orthonormal columns spanning its subspace. The smallest and largest eigenvalues of $Q ^ { \top } G Q$ are the squared-stretch extrema m, M for $J ^ { \top }$ . Use $( m _ { c } , M _ { c } )$ for span{U, R} and $( m _ { \Phi } , M _ { \Phi } )$ for $\operatorname { s p a n } \{ W , V \}$ . The batch map in Section 5.4 is $J ^ { \top } / \sqrt { N } ;$ its squared stretches are $\dot { m } / N , M / \dot { N }$ , with the same ratio. Consequently $\chi _ { c } = M _ { c } / m _ { c }$ and $\chi _ { \Phi } = M _ { \Phi } / m _ { \Phi }$ when their minima are positive, and the ratios are infinite otherwise. The limits used in Theorem 5 are

$$
\kappa _ { c } = \left( \frac { 1 + \sqrt { \gamma _ { K } } } { 1 - \sqrt { \gamma _ { K } } } \right) ^ { 2 } , \qquad \kappa _ { \Phi } = \frac { 1 + \sqrt { \gamma _ { K } } } { 1 - \sqrt { \gamma _ { K } } } .\tag{D.3}
$$

With $K = 1 0 0$ , these give $\kappa _ { c }$ ≈ 34.957650 and $\kappa _ { \Phi } \approx 5 . 9 1 2 4 9 9$

Let $a , b \neq 0$ have cosine c, and suppose the compression of $G = J J ^ { \top }$ to their span has spectrum in $[ m , M ] , m > 0$ . For $- 1 < c < 1$ , the unit bisectors

$$
e \pm = { \frac { a / \| a \| \pm b / \| b \| } { \sqrt { 2 ( 1 \pm c ) } } }
$$

are orthonormal. Symmetry of G cancels the cross terms and gives

$$
\frac { a ^ { \top } G b } { \| a \| \| b \| } = \frac { 1 } { 2 } \big [ ( 1 + c ) e _ { + } ^ { \top } G e _ { + } - ( 1 - c ) e _ { - } ^ { \top } G e _ { - } \big ] \geq \frac { 1 } { 2 } [ ( 1 + c ) m - ( 1 - c ) M ] .\tag{D.4}
$$

Thus a positive measured cosine certifies $a ^ { \top } G b > 0$ whenever $M / m < ( 1 + c ) / ( 1 - c )$ . For $c = 1$ $a , b$ are positively collinear and the lower bound is $m \| a \| \| b \| > 0$ . This proves the measured-angle refinement described in Section 5.4. Using a universal lower bound on c gives the class-dependent tests instead.

## D.3 CONDITIONAL INCREMENT AND FINITE-STEP ADVANTAGE

Assume $U \neq 0$ and let the conditional signal subspace have spectral bounds $0 ~ < ~ m _ { c } ~ \le ~ M _ { c }$ Let $\delta g = g _ { \mathrm { T } } - g _ { \mathrm { C } } = J ^ { \top } R / 2$ . Averaging the per-example squared-norm bounds gives $\| R \| ^ { 2 } \geq$ $\gamma _ { K } ^ { 2 } \Vert \dot { U } \Vert ^ { 2 }$ . To transfer the angle bound to the stacked signals, average the stronger inequality proved in Appendix B.4:

$$
( 1 + \gamma _ { K } ) U ^ { \top } R \geq \gamma _ { K } \| U \| ^ { 2 } + \| R \| ^ { 2 } \geq 2 \sqrt { \gamma _ { K } } \| U \| \| R \| .\tag{D.5}
$$

This yields $R \neq 0$ and cos $\angle ( U , R ) \geq c _ { K }$ . Therefore

$$
\begin{array} { r } { \| \delta g \| ^ { 2 } = \frac { 1 } { 4 } R ^ { \top } G R \geq \frac { m _ { c } } { 4 } \| R \| ^ { 2 } \geq \frac { m _ { c } \gamma _ { K } ^ { 2 } } { 4 } \| U \| ^ { 2 } > 0 . } \end{array}\tag{D.6}
$$

Moreover, with $\beta _ { c } = ( 1 + c _ { K } ) m _ { c } - ( 1 - c _ { K } ) M _ { c } , \mathtt { E q . } ( \mathrm { D . 4 } ) \mathrm { y i e l d s }$

$$
\begin{array} { r } { t _ { c } : = \big \langle \nabla _ { \theta } \overline { { L } } _ { \mathrm { c o n d } } , \delta g \big \rangle = \frac { 1 } { 2 } U ^ { \top } G R \geq \frac { \beta _ { c } } { 4 } \| U \| \| R \| . } \end{array}
$$

The threshold $M _ { c } / m _ { c } < ( 1 + c _ { K } ) / ( 1 - c _ { K } )$ makes $t _ { c } > 0$ . Since $c _ { K } = 2 \sqrt { \gamma _ { K } } / ( 1 + \gamma _ { K } )$ , this threshold is exactly $\kappa _ { c } \sin \mathrm { E q . } \left( \mathrm { D } . 3 \right)$ . At a differentiable parameter state,

$$
L _ { \mathrm { c o n d } } ( \theta - \eta g _ { \mathrm { T } } ) - \overline { { L } } _ { \mathrm { c o n d } } ( \theta - \eta g _ { \mathrm { C } } ) = - \eta t _ { c } + o ( \eta ) < 0
$$

for sufficiently small positive η, proving the conditional conclusion in Eq. (D.2). More explicitly, if the loss gradient is $L _ { c } – \mathrm { L }$ ipschitz on both step segments, the difference is at most $- \eta t _ { c } + L _ { c } \eta ^ { \dot { 2 } } ( \lVert g _ { \mathrm { T } } \rVert ^ { 2 } +$ $\| g _ { \mathrm { C } } \| ^ { 2 } ) \big / 2 ;$ choosing $\eta \leq { t _ { c } } / [ L _ { c } ( \| g _ { \mathrm { T } } \| ^ { 2 } + \| g _ { \mathrm { C } } \| ^ { 2 } ) ]$ gives an upper bound of $- \eta t _ { c } / 2 .$

## D.4 JOINT-POTENTIAL DESCENT

Assume $W , V \neq 0$ and let the joint signal subspace have spectral bounds $0 < m _ { \Phi } \leq M _ { \Phi }$ . The batch form of Eq. (12) gives cos $\angle ( W , V ) \ge s _ { K } = \sqrt { \gamma _ { K } }$ . Applying Eq. (D.4) on span{W, V } gives

$$
\begin{array} { r } { t _ { \Phi } : = \langle \nabla _ { \theta } \Psi , g _ { \mathrm { T } } \rangle = W ^ { \top } G V \geq \frac { 1 } { 2 } [ ( 1 + s _ { K } ) m _ { \Phi } - ( 1 - s _ { K } ) M _ { \Phi } ] \| W \| \| V \| . } \end{array}
$$

Thus $M _ { \Phi } / m _ { \Phi } < \kappa _ { \Phi }$ makes $t _ { \Phi } > 0$ , and $\Psi ( \theta - \eta g _ { \mathrm { T } } ) = \Psi ( \theta ) - \eta t _ { \Phi } + o ( \eta ) < \Psi ( \theta )$ for sufficiently small positive $\eta .$ This proves Eq. (D.2).

## An explicit finite-step bound. Put

$$
E = \frac { 1 } { N } \sum _ { i } \left( \| h _ { i } \| ^ { 2 } + \frac { 1 } { 1 2 } \| u _ { i } \| ^ { 2 } \right) , \qquad \mu _ { \Phi } = \frac { 1 } { 2 } [ ( 1 + \sqrt { \gamma \kappa } ) m _ { \Phi } - ( 1 - \sqrt { \gamma \kappa } ) M _ { \Phi } ] .
$$

Under $M _ { \Phi } / m _ { \Phi } < \kappa _ { \Phi } , \mu _ { \Phi } > 0 ,$ . The output inequalities give

$$
\begin{array} { r } { ( 4 h + u ) ^ { \top } v \geq 4 \| h \| ^ { 2 } - \| h \| \| u \| + \frac { 1 } { 4 } \| u \| ^ { 2 } \geq \| h \| ^ { 2 } + \frac { 1 } { 1 2 } \| u \| ^ { 2 } , } \end{array}
$$

where $\| h \| \| u \| \leq 3 \| h \| ^ { 2 } + \| u \| ^ { 2 } / 1 2$ suffices. Hence $\| W \| \| V \| \ge W ^ { \top } V \ge E$ , while $\| V \| ^ { 2 } \leq 8 E$ . It follows that

$$
t _ { \Phi } \geq \mu _ { \Phi } E , \qquad \| g _ { \mathrm { T } } \| ^ { 2 } = V ^ { \top } G V \leq M _ { \Phi } \| V \| ^ { 2 } \leq 8 M _ { \Phi } E .
$$

If $\nabla _ { \theta }$ Ψ is L<sub>Ψ</sub>-Lipschitz on the current step segment, with $L _ { \Psi } > 0$ , then

$$
\Psi ( \theta - \eta g _ { \mathrm { T } } ) \leq \Psi ( \theta ) - \eta \mu _ { \Phi } E + 4 L _ { \Psi } M _ { \Phi } \eta ^ { 2 } E .
$$

Consequently,

$$
0 < \eta \leq \frac { \mu _ { \Phi } } { 8 M _ { \Phi } L _ { \Psi } } \quad \Longrightarrow \quad \Psi ( \theta - \eta g _ { \mathrm { T } } ) \leq \Psi ( \theta ) - \frac { \eta \mu _ { \Phi } } { 2 } E .\tag{D.7}
$$

## D.5 COMPUTING THE TWO-DIMENSIONAL CERTIFICATES

For either output pair, orthonormalize its span to obtain $Q = [ \xi _ { 1 } , \xi _ { 2 } ]$ , omitting a dependent column. Compute $b _ { i } = J ^ { \top } \xi _ { i }$ by vector–Jacobian products; the compressed matrix has entries $( Q ^ { \top } G Q ) _ { i j } =$ $b _ { i } ^ { \top } b _ { j }$ . Its smallest and largest eigenvalues are the required m, M. No full Jacobian or kernel matrix is formed. The direct conditional and joint inner products are $\langle J ^ { \top } U , J ^ { \top } R / 2 \rangle$ and $\langle J ^ { \top } W , J ^ { \top } V \rangle$ respectively. Each pair uses its own subspace, spectrum and measured angle.

## E EXPERIMENTAL PROTOCOLS

This appendix specifies the training and measurements reported in Section 6.

## E.1 MODELS, DATA, OPTIMIZATION AND EVALUATION

Vision. All visual experiments use CIFAR-100 with 50,000 training images and 10,000 test images, a VOLO-D2 teacher and a PiT-B student. The student combines a publicly pretrained backbone with a classifier fitted on features of the 50,000 training images by standardized logistic regression with regularization parameter $C = 0 . 0 1$ . All full-training methods use the same initial assets. Inputs use eight fixed augmented views: pad each 32-pixel image by four pixels, crop and horizontally flip, then bicubic resize to $2 2 4 \times 2 2 4$

Full training uses 60 epochs, effective batch size 64 (microbatch 32), native automatic mixed precision (AMP), and SGD with momentum 0.9 and constant learning rate $1 0 ^ { - 4 }$ , without warmup or decay. Weight decay is $1 0 ^ { - 3 }$ , except for KL-Dist and XE-KL, whose recorded baseline configurations use $1 0 ^ { - \overline { { 4 } } }$ . Every method is evaluated at epoch 60.

Text. CLINC150 has 150 in-domain classes. We combine its original 15,000 training examples and original 3,000 validation examples into one fixed 18,000-example training pool. The official 4,500-example in-domain test set remains separate. All methods use this same split and a fixed final-epoch evaluation.

The teacher is the existing five-epoch BERT-large. The student uses a publicly pretrained BERT-Mini backbone and a shared random classifier, initialized with truncated normal weights (standard deviation 0.02, limits ±0.04) and zero bias. The head is trained from the first update. Inputs have maximum length 128 with fixed padding and an attention mask; the classifier receives the pretrained pooler output.

Training uses four epochs, 1,125 optimizer iterations, effective batch size 64 (microbatch 32), FP32 and disabled TF32. Google AdamWeightDecay uses moment-decay coefficients $( \beta _ { 1 } , \beta _ { 2 } ) =$ (0.9, 0.999) and numerical stabilizer $\epsilon = 1 0 ^ { \bar { - } 6 }$ , weight decay 0.01, no bias correction, and global gradient-norm clipping at 1.0. Bias and LayerNorm parameters are excluded from weight decay. With zero-indexed iteration k, the shared learning-rate schedule is

$$
\eta _ { k } = 3 \times 1 0 ^ { - 4 } \left\{ { k / 1 1 2 , } \atop { 1 - k / 1 1 2 5 , } \right. \ : \ : k < 1 1 2 ,
$$

Thus $3 \times 1 0 ^ { - 4 }$ is the base rate of the common warmup and decay schedule.

Method-specific objectives and temperatures. The comparisons align data, teacher–student pair, initialization, training budget and evaluation within each domain. The loss used by each baseline retains its method-specific definition and prescribed scale. In particular, temperature is part of that definition, rather than a common training-budget parameter. The following choices are fixed in the reported runs.

KD and DKD use temperature $T = 4 ,$ , matching the KD and DKD settings in the official DKD implementation (Zhao et al., 2022). Its configuration declares KD.TEMPERATURE=4 and DKD. $\mathrm { { T } = 4 . 0 . ^ { 1 } }$ For DKD, target and non-target weights are 1 and 8, respectively; the recorded method warmup is retained (two epochs in text). Standard KD uses distillation coefficient 1 in the shared training setup.

DHKD uses temperature 2 for its binary-KL objective, following its released training commands (Yang et al., 2025); the official ImageNet command explicitly specifies $- \mathrm { B i n a r y K L \_ T } 2 . ^ { 2 }$ This is the temperature of DHKD’s binary-KL loss, rather than a replacement of its objective by the KD loss.

The recorded KL-Dist, XE-KL, DP-U, DP-S and DTO-KD objectives use unit logit scale $( T = 1$ where a temperature argument is present). TPKD also uses the original, unit-temperature conditional probabilities, with coefficient $1 / 2$ . CE has no distillation temperature. Text DTO-KD retains its multi-layer feature-distillation implementation. These objective settings are unchanged across the three seeds; the five component controls use the TPKD settings and alter only the direction shown in Table 5.

Accuracy reporting. All full-training entries use seeds 42, 43 and 44. Within each domain, the comparisons share initial assets, training objects, data-order rules and the final-epoch evaluation. Tables 4–5 report test accuracy in percent as the mean and sample standard deviation over $n = 3$ seeds (denominator n − 1). The final epoch is fixed before training; no validation, observation or best-test checkpoint is used for selection.

KL-Dist and DP-U on CLINC150. In the reported CLINC150 runs, DP-U selected the teacher target at a rate of 100% in every epoch for all three seeds (42, 43 and 44), making its training objective identical to that of KL-Dist in these runs. Under the shared initialization, data order and optimization settings, we verified that the final logits were elementwise identical between the two methods for each seed. This explains their identical per-seed accuracies and the same mean and sample standard deviation of $9 3 . 3 8 \pm 0 . 1 9 \%$ in Table 4.

## E.2 BLOCKING AND EXACT HEAD UPDATES

Blocking (Table 1). CE20, CE40 and CE60 are fixed seed-42 CE states under the visual protocol. Each uses 2,000 predetermined images with eight fixed views. The blocking criterion is $p _ { j } \leq t _ { j }$ for every $j \neq y .$ . Conditional KL and $D _ { \infty }$ are evaluated from log probabilities within the blocked set. Positive conditional error is checked at tolerance $1 0 ^ { - 1 2 }$ ; all 16,000 image–view pairs enter the measurement.

Exact updates (Table 2). Freeze the backbone features and absorb the head bias into the feature matrix. The head, direction calculation and SVD pseudoinverse use FP64. Each candidate starts from the same model and batch with output step $\eta = 0 . 0 1$ . The row-stacked direction specifies each example’s output displacement, so it is not divided by batch size. The full-row-rank solve uses the pseudoinverse directly without ridge regularization.

The five directions are CE h, compensation only $h + \ell a _ { 0 } / 2$ , unprojected conditional learning $h + u / 2$ TPKD $h + d / 2$ , and CE plus full KL $h + ( p - t )$ . The last uses the original teacher probabilities with coefficient 1. Two states and 32 batches of 64 examples per state give 320 candidate steps, with two repeat checks. Let $z _ { \mathrm { C E } } ^ { + }$ and $z _ { \mathrm { T P K D } } ^ { + }$ denote the respective endpoint logits. The paired gain is $G _ { \mathrm { C E } } = L _ { \mathrm { c o n d } } ( z _ { \mathrm { C E } } ^ { + } ) - L _ { \mathrm { c o n d } } ( z _ { \mathrm { T P K D } } ^ { + } )$ ; table losses and changes are means over examples.

Retention is $u ^ { \top } d / \Vert u \Vert ^ { 2 }$ and the complete-update cosine is cos $\angle ( g _ { \Phi } , v )$ . We also verify the alignment residual $R _ { h } = g _ { \Phi } ^ { \dagger } \ddot { v } - \| v \| ^ { 2 } - \gamma _ { K } \| \bar { g } _ { \Phi } \| ^ { 2 } / 4$ and the finite-step descent bound in Eq. (13). A tolerance of $1 0 ^ { - 1 2 }$ is used for inequalities and margin-order checks. The largest output execution error and CEreference margin discrepancy are both $1 . 4 2 \times 1 0 ^ { - 1 4 }$ , supporting the numerical-precision statement in Section 6.

## E.3 BACKPROPAGATION CONDITIONS AND NATIVE PAIRED UPDATES

Table 3 uses sixteen fixed batches at each of CE20 and CE60. TPKD uses $h + d / 2 ;$ compensation only uses $h + \ell a _ { 0 } / 2 .$ , preserving the common confidence contribution while deleting the pure conditional component. Batch-average losses and the $1 / \sqrt { 6 4 }$ normalization in Appendix D are used throughout.

Network and vector–Jacobian calculations use FP32, with FP64 geometric calculations and diagnostic microbatch size eight. The two restricted spectra are computed as in Appendix D.5, with relative rank tolerance $1 0 ^ { - 1 0 }$ and spectral tolerance $1 0 ^ { - 1 2 }$ . For 100 classes, the theoretical constants are $\gamma _ { 1 0 0 } = 1 0 0 / 1 9 8 , \sqrt { \gamma _ { 1 0 0 } } \approx 0 . 7 1 0 7 , \kappa _ { c } \approx 3 4 . 9 6$ and $\kappa _ { \Phi } \approx 5 . 9 1$

Each native candidate restores the same model, optimizer history, AMP scaler, buffers and random state, then executes the usual training iteration with microbatch size 32 and learning rate $1 0 ^ { - 4 }$ . Let $\theta _ { \mathrm { C } } ^ { + }$ and $\theta _ { \mathrm { T } } ^ { + }$ denote the resulting compensation-only and TPKD parameter endpoints. Their paired gain is

$$
G _ { \mathrm { n a t i v e } } = \overline { { L } } _ { \mathrm { c o n d } } ( \theta _ { \mathrm { C } } ^ { + } ) - \overline { { L } } _ { \mathrm { c o n d } } ( \theta _ { \mathrm { T } } ^ { + } ) .
$$

FP32 paired replay evaluates both endpoint losses consistently. The geometric tests establish a useful conditional parameter direction; the native pairs measure its additional benefit under the training optimizer.

## E.4 THE 128-ITERATION CONTINUATION

$\mathrm { C E , }$ compensation only and TPKD continue from the same CE60 state for 128 native-optimizer iterations at constant learning rate $1 0 ^ { - 4 }$ . They follow the same sequence of 8,192 update images, recomputing their directions from their own current student states. A fixed 2,048-image observation set is disjoint from these updates. Both sets are drawn from the original training pool, and observation is used only to measure conditional learning. We record conditional KL at iterations 0, 1, 8, 32, 64 and 128. Section 6 reports its starting value and TPKD’s endpoint advantages.

## E.5 THE FIVE COMPONENT CONTROLS

All five rows in Table 5 use the full-training protocols above. CE (h) and TPKD $( h + d / 2 )$ use the corresponding main-table runs. Removing projection gives $h + u / 2$ . Removing the CE gradient gives $d / 2 \colon$ it retains coefficient $1 / 2$ , true-label indexing of the non-target classes, and the same safe cone. Safe full KL gives $h + \Pi _ { \mathcal { K } _ { u } } ( p - t ) / 2$ , using the full unit-temperature gradient and the generic projection in Eq. (B.1). Every control recomputes its direction at each iteration. The initialization, teacher, data order, optimization schedule and endpoint selection are unchanged.

## E.6 IMPLEMENTATION OF THE PRESCRIBED DIRECTION

Keep the teacher fixed and detached. Compute $q = \mathrm { s o f t m a x } ( z _ { \lnot y } ^ { T } )$ and $r = \operatorname { s o f t m a x } ( z _ { \lnot y } )$ directly on non-target logits, form $u = ( 0 , r - q )$ , project to d, and set $v = h + d / 2$ . Detach v in Eq. (8); its batchaveraged surrogate supplies the complete parameter data gradient, including the label contribution. Log-softmax values are used for conditional losses and compensation statistics, avoiding division by small $1 - p _ { y }$ values. The native optimizer then applies its usual update to the student parameters.