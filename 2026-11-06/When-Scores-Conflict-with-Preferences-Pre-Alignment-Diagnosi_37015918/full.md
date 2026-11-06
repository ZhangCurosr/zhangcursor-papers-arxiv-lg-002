# When Scores Conflict with Preferences: Pre-Alignment Diagnosis and Adaptive Correction of Constraint Drift in DPO

Ping Liu   
Department of Computer Science   
and Engineering   
University of Nevada, Reno   
pino.pingliu@gmail.com   
Yan Yan<sup>∗</sup>   
School of Electrical Engineering   
and Computer Science   
Washington State University   
yan.yan1@wsu.edu

## Abstract

Direct Preference Optimization (DPO) is increasingly used in settings where pairwise preferences coexist with pointwise criteria such as response length, answer correctness, or safety scores. When the two disagree, standard DPO treats pairs whose preferred output scores worse on the pointwise criterion (conflict pairs) in the same way as the remaining (matching) pairs. We characterize a systematic failure mode, constraint drift: for sufficiently small updates, conflict pairs tend to increase constraint cost while matching pairs tend to repair it, and the net change depends on their balance. This balance yields a closed-form pre-alignment diagnostic, $\rho _ { \mathrm { e q } } ^ { * } ,$ computable from dataset statistics, that indicates whether a dataset is self-correcting or requires intervention. Building on this diagnostic, we propose Conflict-aware DPO (CoDPO), which down-weights conflict pairs with a strength set from $\rho _ { \mathrm { e q } } ^ { * }$ rather than tuned, and adds an adaptive margin shift that addresses residual drift during training. Empirically, drift grows monotonically with conflict exposure on the conflict-dominated domains, and the intervention decision given by $\rho _ { \mathrm { e q } } ^ { * }$ matches the observed drift on every domain, including one that it identifies as self-correcting. On all four domains of the main comparison, CoDPO reduces drift while largely preserving the preference learning of vanilla DPO. Compared with hard filtering and recent constrained and multi-objective baselines, CoDPO also achieves a more favorable drift-preference trade-off.

## 1 Introduction

Generative models are increasingly aligned with human objectives through preference optimization. Preference optimization methods such as RLHF [1, 2, 3] and DPO [4] have substantially improved helpfulness, reasoning quality, and instruction following [5, 6, 7, 8] by learning from pairwise comparisons between candidate outputs. A response preferred in such a comparison may nevertheless score poorly on a pointwise criterion, such as correctness or safety, that assesses each response on its own [9, 10]. Following PSDPO [11], we call this coexistence of pairwise preferences and pointwise criteria heterogeneous supervision.

Under heterogeneous supervision, standard preference optimization treats all training pairs uniformly, implicitly assuming that the preferred output also performs well on the pointwise criterion. On the domains we study, the preferred output scores worse on the pointwise criterion in a substantial fraction of pairs: where the criterion is response length, the preferred response is the longer one in 57% of HelpSteer2 pairs [12], and where it is correctness, the preferred response is the incorrect one in 71% of GSM8K pairs [13] and 58% of ARC pairs [14]. Such pairs, conflict pairs in the terminology of PSDPO [11], occur at comparable rates in the other conflict-dominated domains we study (Table 1); the remaining pairs are matching pairs. Conflict pairs tend to push preference optimization toward outputs that win the pairwise comparison but carry a higher cost on the pointwise criterion, a degradation that PSDPO [11] identifies as semantic drift in text-to-video alignment; we call its counterpart under a general pointwise criterion constraint drift.

![](images/6466be0f4a1afe958f490760f072c4d3c2ded37145bf84121f9218464ad16ccd.jpg)

![](images/fef17744ef8e703bfff589db238726f66f8022655babc7557a4b693686b697fd.jpg)

![](images/a49add9c81b2136966c4a265fb731eaf36c07b1e48e28bd307506cff3b7471e6.jpg)  
Figure 1: Overview of CoDPO. (a) Under DPO, conflict pairs push the constraint cost up and matching pairs pull it down; $\rho _ { \mathrm { e q } } ^ { * } ,$ computed before training, compares the two. If the push dominates $( \rho _ { \mathrm { e q } } ^ { * } < 1 )$ , CoDPO intervenes; otherwise the data are self-correcting and plain DPO is used. (b) Conflict pairs are down-weighted by severity, with $\tau ^ { * }$ set from $\rho _ { \mathrm { e q } } ^ { * } .$ (c) During training, the DPO margin is shifted, more strongly when the constraint-risk estimate rises. Curves in (b) and (c) are schematic.

Existing interventions against constraint drift differ in how they trade the pointwise criterion off against the preference. Constrained RLHF, constrained DPO [9, 15, 16, 17, 18], and multi-objective methods [19, 20] set this trade-off through a weight that is either fixed by the practitioner or, in Lagrangian methods, adjusted during training according to whether the estimated average cost exceed a target. Hard filtering, akin to preference-data selection [21], removes conflict pairs, which can discard much of the data when conflicts are frequent, while per-pair weighting [22, 11] attenuates them. In each case, the strength of the intervention is chosen by the practitioner, which leaves two questions open for a given dataset: whether it needs intervention at all, ideally known in advance, and if so, how strong the intervention should be, both before and during training.

We address both questions from a theoretical account of constraint drift under DPO. We show that, for sufficiently small updates, conflict pairs and matching pairs push the constraint in opposite directions: conflict pairs raise the expected constraint cost, while matching pairs repair it. Drift therefore depends not on the frequency of conflicts alone but on the balance between conflict-side pressure and matching-side repair. Under a first-order approximation, this balance can be estimated from dataset statistics alone, before any alignment step is taken. The estimate yields a closed-form diagnostic, $\rho _ { \mathrm { e q } } ^ { * } ,$ that indicates whether the dataset is self-correcting or needs intervention (Figure 1a).

When intervention is needed, our control method, Conflict-aware DPO (CoDPO), acts at two levels within a single loss (Figure 1b,c). At the coarse level, CoDPO down-weights conflict pairs by severity, rather than removing them all, through asymmetric conflict weighting [11]. Instead of tuning the weighting strength, CoDPO solves for it from dataset statistics through $\rho _ { \mathrm { e q } } ^ { * } ,$ so that conflict-side pressure and matching-side repair balance to first order. The diagnostic is computed at initialization, and the balance can shift as the policy moves away from the reference model during training, so some drift can remain. The fine level addresses this residual drift by shifting the DPO margin of each pair in proportion to its constraint-cost difference, and strengthens the shift when the drift measured during training rises. Together, the diagnostic tells the practitioner whether a dataset needs intervention, and CoDPO determines how strong that intervention should be.

We test the diagnostic and CoDPO on five conflict-dominated domains (HelpSteer2, GSM8K, ARC, Math-Calc, TL;DR) and a self-correcting control (PKU-SafeRLHF [10]). On these domains, drift grows monotonically with conflict exposure except on the control, where it stays negligible, and $\rho _ { \mathrm { e q } } ^ { * }$ assigns the correct intervention regime on all six. On the four domains of the main comparison, all of which call for intervention, CoDPO reduces drift while largely preserving the preference learning achieved by vanilla DPO. Compared with hard filtering and recent constrained and multi-objective baselines, CoDPO also achieves a more favorable drift-preference trade-off.

In summary, our contributions are as follows:

• We show that, for small updates, the change in constraint cost at each DPO step is a balance between conflict-side pressure and matching-side repair, and derive from it a closed-form diagnostic, $\rho _ { \mathrm { e q } } ^ { * } ,$ , indicating before training whether a dataset needs intervention.

• We propose CoDPO: its coarse level adopts PSDPO’s asymmetric weighting form but applies it only when $\rho _ { \mathrm { e q } } ^ { * } < 1$ , with the strength set from $\rho _ { \mathrm { e q } } ^ { * }$ rather than tuned; its fine level shifts the DPO margin and strengthens the shift when the drift measured during training rises.

• Empirically, the intervention decision given by $\rho _ { \mathrm { e q } } ^ { * }$ matches the observed drift on every domain we evaluate, and in our main comparison CoDPO reduces drift while largely preserving vanilla DPO’s preference learning, with a more favorable drift-preference tradeoff than the compared baselines.

## 2 Related Work

Preference and constrained alignment. RLHF [2, 1] and DPO [4] learn from pairwise preferences, and subsequent variants modify the preference objective [5, 6, 7, 23, 24, 25] or address conflicts among multiple pairwise dimensions [26, 27, 28, 29], while keeping the supervision purely pairwise. Constrained alignment adds a pointwise cost, through a cost model with Lagrangian updates in Safe RLHF [9] and through recent constrained DPO variants [15, 30, 16, 17, 18]. These methods share our supervision structure, pairwise preferences together with a pointwise criterion, and set the trade-off between the two through a parameter chosen by the practitioner or driven by an aggregate cost estimate. Our analysis addresses a question that precedes this trade-off: whether a given dataset needs constraint handling at all, and if so, how strong it should be, as determined by the conflic structure of its pairs.

Multi-objective optimization and preference-data selection. Multi-objective methods such as PCGrad [31], CAGrad [32], Rewarded Soups [19], and MODPO [20] address objective competition through gradient projection, model interpolation, or reward combination, and data-centric methods retain a high-margin subset of the training pairs [21] or select the informative tokens within a pair [33]; these methods act on objectives or on general data quality rather than on the conflict between a pairwise preference and a pointwise criterion within a pair. Closer to our coarse level, Uni-DPO [22] keeps every pair but reweights it by an external score margin and by the implicit reward margin, so that ambiguous and already-learned pairs contribute less. In text-to-video alignment, PSDPO [11] weights pairs by conflict direction instead, down-weighting conflict pairs with an asymmetric exponential weight that leaves matching pairs unchanged.

Our coarse level adopts this weighting form and differs in how its strength is set. PSDPO bounds the gradient contributed by conflict pairs and shows that the bound shrinks with stronger down-weighting, which motivates down-weighting but leaves its strength to be tuned. Section 3 instead tracks the constraint cost, which conflict pairs raise and matching pairs reduce. Because of this repair, the cost stops rising to first order once the conflict weight falls below a critical value, from which CoDPO sets the strength and decides whether down-weighting is needed. To limit drift that accumulates during training, PSDPO introduces conflict pairs only in a second training stage, whereas our fine level shifts the DPO margin throughout training and strengthens the shift when the measured drift rises.

## 3 Constraint Drift as a Balance of Pressure and Repair

## 3.1 Setup and Conflict Structure

Each training example is a preference pair $( x , y _ { w } , y _ { l } )$ , in which $y _ { w }$ is the output preferred by the pairwise annotation and $y _ { l }$ the rejected one. Let $\pi _ { \theta } ( y \mid x )$ be a generative policy initialized at the reference policy, $\pi _ { \theta _ { 0 } } = \pi _ { \mathrm { r e f } }$ , as is standard for DPO, and let $\bar { L _ { R } } ( \theta ) = - \mathbf { \bar { \mathbb { E } } } _ { \mathcal { D } } [ \log \sigma ( \beta \hat { r } _ { \theta } ) ]$ be the

DPO preference loss [4], where $\begin{array} { r } { \hat { r } _ { \theta } = \log \frac { \pi _ { \theta } \left( y _ { w } | x \right) } { \pi _ { \mathrm { r e f } } \left( y _ { w } | x \right) } - \log \frac { \pi _ { \theta } \left( y _ { l } | x \right) } { \pi _ { \mathrm { r e f } } \left( y _ { l } | x \right) } } \end{array}$ is the implicit reward margin. A separate scorer $s : \mathcal { X } \times \mathcal { Y }  \mathbb { R }$ evaluates each output independently of the pairwise signal, oriented so that larger values indicate better satisfaction of the pointwise criterion; for instance, s decreases with response length on a length-constrained domain [34, 35] and is a reward-model score [2, 1] on a safety-constrained domain. A constraint cost $c ( x , y ) \geq 0$ , strictly decreasing in $s ,$ defines the constraint risk $C ( \theta ) = \mathbb { E } _ { x , y \sim \pi _ { \theta } } [ c ( x , y ) ]$

Definition 1 (Conflict [11]) For a preference pair $( x , y _ { w } , y _ { l } )$ , let $\Delta s : = s ( x , y _ { w } ) - s ( x , y _ { l } )$ and $\Delta c : = c ( x , y _ { w } ) - c ( x , y _ { l } )$ . A conflict pair satisfies $\Delta s < 0$ (equivalently $\Delta c > 0 ) ,$ , meaning that the preferred output scores worse on the pointwise criterion than the rejected one. We define the conflict set $\mathcal { D } _ { \mathrm { c o n f } } : = \bar { \{ ( x , y _ { w } , y _ { l } ) : \Delta s < 0 \} }$ and the matching set ${ \mathcal { D } } _ { \operatorname* { m a t c h } } : = \{ ( x , y _ { w } , y _ { l } ) : \Delta s \geq 0 \}$ , with conflict rate $p _ { c } : = \mathrm { P r } ( \Delta s < 0 ) . ^ { 2 }$ 2

## 3.2 Pressure, Repair, and the Diagnostic

Following PSDPO [11], the conflict-matching partition splits the DPO gradient into contributions from the two subsets,

$$
\nabla L _ { R } ( \theta ) = ( 1 - p _ { c } ) g _ { \mathrm { { m a t c h } } } + p _ { c } g _ { \mathrm { { c o n f } } } ,
$$

where $g _ { \mathrm { m a t c h } } = \mathbb { E } _ { \mathcal { D } _ { \mathrm { m a t c h } } } [ \nabla \ell _ { R } ] , g _ { \mathrm { c o n f } } = \mathbb { E } _ { \mathcal { D } _ { \mathrm { c o n f } } } [ \nabla \ell _ { R } ]$ , and the per-sample loss is $\ell _ { R } = - \log \sigma ( \beta \hat { r } _ { \theta } )$ Whereas PSDPO [11] bounds the norm of the conflict term, we bound its first-order effect on the constraint risk, as in first-order analyses of DPO training dynamics [36, 37]: under regularity conditions on $C$ and the coupling argument of Appendix B.2, the conflict term can raise the constraint risk to first order,

$$
- \langle \nabla C , g _ { \mathrm { c o n f } } \rangle \ge \mathbb { E } _ { \mathcal { D } _ { \mathrm { c o n f } } } [ \beta ( 1 - \sigma ( \beta \hat { r } _ { \theta } ) ) \Delta c \mathcal { T } _ { \theta } ] \ge 0 ,
$$

where, for $\Delta c \neq 0 ,$ , the geometry term $\begin{array} { r } { \mathcal { T } _ { \theta } : = ( \Delta c ) ^ { - 1 } \langle \nabla _ { \theta } C _ { x } ( \theta ) , \nabla _ { \theta } \hat { r } _ { \theta } \rangle > 0 . } \end{array}$ , with $C _ { x } ( \theta ) =$ $\mathbb { E } _ { y \sim \pi _ { \theta } } [ c ( x , y ) ]$ , measures how much the expected cost on prompt x changes, per unit of the cost gap $\Delta c ,$ , when DPO learns the pair; it is named for the inner product between the two gradient directions. An analogous bound with the opposite sign holds for $g _ { \mathrm { m a t c h } }$ and captures the repair provided by matching pairs.

Theorem 1 (Constraint drift) Under the regularity conditions of Appendix B.1 (proof in $A p \cdot$ pendix B.3), gradient descent on $L _ { R }$ satisfies

$$
\begin{array} { r } { \mathbb { E } \left[ C ( \theta _ { t + 1 } ) - C ( \theta _ { t } ) \right] \geq - \eta ( 1 - p _ { c } ) \mu _ { 0 } + \eta p _ { c } \kappa _ { 0 } - \frac { 1 } { 2 } \eta ^ { 2 } L _ { C } G _ { R } ^ { 2 } , } \end{array}\tag{1}
$$

where $\eta$ is the step size, $L _ { C }$ is the smoothness constant of $C ,$ , and $G _ { R }$ bounds $\| \nabla \ell _ { R } \|$ . The terms

$$
\kappa _ { 0 } : = \mathbb { E } _ { \mathcal { D } _ { \mathrm { c o n t } } } [ \beta ( 1 - \sigma ( \beta \hat { r } _ { \theta } ) ) \Delta c \mathcal { T } _ { \theta } ] > 0 , \qquad \mu _ { 0 } : = \mathbb { E } _ { \mathcal { D } _ { \mathrm { m a t e h } } } [ \beta ( 1 - \sigma ( \beta \hat { r } _ { \theta } ) ) | \Delta c | \mathcal { Z } _ { \theta } ] > 0
$$

aggregate the conflict-side drift pressure and the matching-side repair, respectively. For sufficiently small $\eta ,$ whenever $p _ { c } \kappa _ { 0 } > ( 1 - p _ { c } ) \mu _ { 0 }$ , the lower bound is strictly positive and the expected constraint risk increases.

Theorem 1 thus expresses each DPO step as a balance between conflict-side pressure and matchingside repair: drift appears when pressure outweighs repair, and we call the reverse case, in which repair at least matches pressure, the self-correcting regime, which Section 3.3 makes precise.

Remark 1 (Pre-alignment diagnostic) The drift condition of Theorem 1, $p _ { c } \kappa _ { 0 } > ( 1 - p _ { c } ) \mu _ { 0 } $ , is equivalent to $( 1 - p _ { c } ) \mu _ { 0 } / ( p _ { c } \kappa _ { 0 } ) < 1$ , and this ratio admits a simple data-dependent approximation at the start ofalignment. Both $\mu _ { 0 }$ and κ<sub>0</sub> average the product ofthreefactors: the learning signal, the cost difference, and the geometry term. At initialization $\pi _ { \theta } = \pi _ { \mathrm { r e f } } , s o \hat { r } _ { \theta } = 0$ and the learning signal $\beta ( 1 - \sigma ( \beta \hat { r } _ { \theta } ) )$ ) takes the same value $\beta / 2$ on every pair, which cancels exactly in the ratio. Assumption (vi) (Appendix B.1) posits that, per unit ofcost gap, learning a conflict pair changes the expected cost by about as much as learning a matching pair; under this assumption, the geometry term cancels approximately as well, and the ratio reduces to the average cost differences $\overline { { \Delta c } } _ { \mathrm { c o n f } } : = \mathbb { E } _ { \mathcal { D } _ { \mathrm { c o n f } } } [ \Delta c ]$ and $\overline { { | \Delta c | } } _ { \mathrm { m a t c h } } : = \mathbb { E } _ { \mathcal { D } _ { \mathrm { m a t c h } } } [ | \Delta c | ]$ , giving the closed-form diagnostic

$$
\rho _ { \mathrm { e q } } ^ { * } = \frac { ( 1 - p _ { c } ) \overline { { | \Delta c | } } _ { \mathrm { m a t c h } } } { p _ { c } \overline { { \Delta c } } _ { \mathrm { c o n f } } } ,\tag{2}
$$

that $i s ,$ , the total cost reduction across matching pairs divided by the total cost increase across conflict pairs, so $\rho _ { \mathrm { e q } } ^ { * }$ depends not only on how often conflicts occur but also on how large the cost differences are. Under this approximation, $\rho _ { \mathrm { e q } } ^ { * } \geq 1$ indicates before alignment that the dataset is self-correcting and needs no intervention, while a value below 1 calls for intervention; Appendix B.5 gives the full derivation.

## 3.3 Down-Weighting Conflict Pairs

Theorem 1 suggests keeping matching pairs at full weight and down-weighting conflict pairs, as PSDPO [11] does in text-to-video alignment; the question we address is how much down-weighting is enough. We assign each pair a weight $w ( \Delta s )$ , equal to 1 on matching pairs and in $[ 0 , { 1 } ]$ on conflict pairs, and summarize how much of the conflict-side signal is retained by the average conflict weight $\rho : = \mathbb { E } _ { \mathcal { D } _ { \mathrm { c o n f } } } [ w ( \Delta s ) ]$ , so that $\rho = 1$ recovers plain DPO and $\rho = 0$ removes all conflict pairs. Theorem 2 answers this question with a critical value $\rho ^ { * }$ for the average conflict weight, below which the reduced conflict-side pressure is smaller than the matching-side repair.

Theorem 2 (Drift attenuation) Under the conditions ofTheorem $^ { l , }$ the modulated loss $L _ { R , w } ( \theta ) =$ $- \mathbb { E } [ w ( \Delta s )$ log $\sigma ( \beta \hat { r } _ { \theta } ) ]$ satisfies the upper bound

$$
\begin{array} { r } { \mathbb { E } [ C ( \theta _ { t + 1 } ) - C ( \theta _ { t } ) ] \le - \eta ( 1 - p _ { c } ) \mu _ { 0 } + \eta p _ { c } \kappa _ { w } + \frac { 1 } { 2 } \eta ^ { 2 } L _ { C } G _ { R } ^ { 2 } , } \end{array}
$$

where $\kappa _ { w } : = \mathbb { E } _ { \mathcal { D } _ { \mathrm { c o n f } } } [ w ( \Delta s ) \beta ( 1 - \sigma ( \beta \hat { r } _ { \theta } ) ) \Delta c \mathcal { T } _ { \theta } ]$ is the modulated counterpart of $\kappa _ { 0 }$ and satisfies $\kappa _ { w } \leq \rho \kappa _ { 0 }$ by Assumption (v) and Chebyshev’s sum inequality (Appendix B.4). Using $\kappa _ { w } \leq \rho \kappa _ { 0 }$ thefirst-order part ofthis bound is at most η $\left[ p _ { c } \rho \kappa _ { 0 } - \left( 1 - p _ { c } \right) \mu _ { 0 } \right]$ , which is negative when $\rho$ lies below the critical weight

$$
\rho ^ { * } = { \frac { ( 1 - p _ { c } ) \mu _ { 0 } } { p _ { c } \kappa _ { 0 } } } ,
$$

the model-dependent threshold at which conflict-side pressure and matching-side repair balance. For $\rho < \rho ^ { * }$ and a step size satisfying the explicit bound given in Appendix B.4, the expected per-step change in constraint risk is non-positive.

Down-weighting leaves the repair from matching pairs unchanged and reduces the pressure from conflict pairs to at most $\rho$ times its original value. Once $\rho$ falls below $\rho ^ { * }$ , the reduced pressure is smaller than the repair, so for a sufficiently small step each update is expected to lower the constraint risk or leave it unchanged. Plain DPO corresponds to $\rho = 1$ , so when $\rho ^ { * } > 1$ it already satisfies this condition, and the data are self-correcting.

To use this result, CoDPO needs a value of $\rho$ before training starts. Because $\rho ^ { * }$ depends on gradients of the model, CoDPO instead uses its estimate $\rho _ { \mathrm { e q } } ^ { * }$ from Remark 1, which needs only the cost gaps in the data. It sets the average conflict weight to $\rho = \mathrm { { \dot { \rho } } _ { e q } ^ { * } }$ , and keeps every weight at 1, that is, plain DPO, when $\rho _ { \mathrm { e q } } ^ { * } \geq 1$ . Some drift can nevertheless remain, for two reasons: $\rho$ is set at the estimated balance point rather than strictly below $\rho ^ { * }$ , and $\rho ^ { * }$ itself shifts as training proceeds. Section 4 implements this coarse level and adds a fine level that addresses this residual drift.

## 4 CoDPO: Controlling the Balance Before and During Training

CoDPO turns this analysis into two levels of control: a coarse level that sets the conflict weights before training (Section 4.1) and a fine level that corrects the drift remaining during training (Section 4.2).

## 4.1 Coarse Level: Setting the Weights Before Training

The coarse level assigns per-pair weights whose average conflict weight matches the pre-alignment estimate $\rho _ { \mathrm { e q } } ^ { * }$ of Remark 1.<sup>3</sup> We keep matching pairs at full weight and down-weight conflict pairs by severity through the weighting family

$$
\begin{array} { r } { w _ { \tau , p } ( \Delta s ) = \left\{ \begin{array} { l l } { 1 , } & { \Delta s \geq 0 , } \\ { \exp \bigl ( - ( | \Delta s | / \tau ) ^ { p } \bigr ) , } & { \Delta s < 0 , } \end{array} \right. } \end{array}\tag{3}
$$

where $\tau$ sets how strongly conflict pairs are down-weighted overall and $p$ sets how sharply the weight drops from 1 to near 0 as the conflict becomes more severe.<sup>4</sup> This family extends the asymmetric exponential weighting introduced by PSDPO [11] with the sharpness p and recovers it at $p = 1 \AA$ ; in PSDPO, τ is chosen as a hyperparameter. In CoDPO, τ is instead determined by the diagnostic. For fixed $p ,$ each τ yields an average conflict weight $\rho ( \tau ) : = \mathbb { E } _ { \mathcal { D } _ { \mathrm { c o n f } } } [ w _ { \tau , p } ( \Delta s ) ]$ , which increases strictly from 0 to 1 as τ grows from 0 to ∞: a small τ suppresses nearly all conflict pairs, and a large τ keeps them at nearly full weight. CoDPO picks the τ whose average conflict weight equals the diagnostic,

$$
\rho ( \tau ^ { * } ) = \mathrm { c l i p } ( \rho _ { \mathrm { e q } } ^ { * } , 0 , 1 ) .\tag{4}
$$

Because $\rho ( \tau )$ is strictly increasing, Eq. (4) has a unique solution for $0 < \rho _ { \mathrm { e q } } ^ { * } < 1$ , which we locate by bisection; when $\rho _ { \mathrm { e q } } ^ { * ^ { \star } } \geq 1 , \mathrm { C o D P O }$ keeps $w _ { i } = 1$ on every pair. The calibration uses only dataset statistics and requires no gradient computation.

## 4.2 Fine Level: Shifting the Margin During Training

The coarse weights are calibrated from $\rho _ { \mathrm { e q } } ^ { * } ,$ which assumes the learning signal $\beta ( 1 - \sigma ( \beta \hat { r } _ { \theta } ) )$ to be the same on every pair. This holds at initialization, where $\pi _ { \theta } = \pi _ { \mathrm { r e f } }$ gives every pair a zero margin $\hat { r } _ { \theta }$ . As training proceeds, the learning signal comes to differ from pair to pair, so the balance point that $\rho _ { \mathrm { e q } } ^ { * }$ approximates moves while the weights stay fixed, and the constraint risk can keep rising; we call this remaining drift residual drift. Lowering the conflict weights further would discard more of the preference signal that conflict pairs carry, and fixed weights cannot respond to drift as it emerges, so we add a second control that acts during training without changing the weights.

Shifting the margin. DPO pushes each pair with strength $\beta ( 1 - \sigma ( \beta \hat { r } _ { \theta } ) )$ , which is set by the pair’s margin ${ \hat { r } } _ { \theta } \colon$ the larger the margin, the more the pair is treated as already learned and the less it is pushed. Adding an offset to a pair’s margin therefore makes DPO update that pair more or less strongly, while leaving its coarse weight w unchanged. On the i-th pair $( x _ { i } , y _ { w , i } , y _ { l , i } )$ , DPO moves probability from $y _ { l , i }$ toward $y _ { w , i }$ . On a conflict pair $( \Delta c _ { i } > 0 ) , y _ { w , i }$ is the costlier response, so this raises the expected constraint cost; on a matching pair $( \Delta c _ { i } < 0 )$ , it lowers it. To push conflict pairs less and matching pairs more, we shift each pair’s margin in proportion to its cost gap,

$$
\hat { r } _ { \theta } \to \hat { r } _ { \theta } + \lambda \Delta c , \lambda \geq 0 .
$$

A conflict pair has its margin raised and is pushed less, most of all when its conflict is severe, so its effect on the constraint cost weakens while its preference is still learned; a matching pair has its margin lowered and is pushed more, so its repair strengthens. A single scalar λ thus adjusts each pair according to its own cost gap and, unlike the fixed margin offsets of ODPO [38], R-DPO [35], and SafeDPO [18], can be changed during training.

Implementing the shift. The margin shift $\hat { r } _ { \theta }  \hat { r } _ { \theta } + \lambda \Delta c$ needs no change to the DPO loss: standard DPO computes it when the reference is tilted against high-cost outputs,

$$
\begin{array} { r } { \tilde { \pi } _ { \mathrm { r e f } } ( y \mid x ) \propto \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \bigl ( - \lambda c ( x , y ) \bigr ) , } \end{array}\tag{5}
$$

since log $\tilde { \pi } _ { \mathrm { r e f } } ( y \mid x ) = \log \pi _ { \mathrm { r e f } } ( y \mid x ) - \lambda c ( x , y ) - \log Z ( x )$ and the normalizer $Z ( x )$ , shared by $y _ { w }$ and y<sub>l</sub>, cancels in the pairwise margin, leaving $\tilde { r } _ { \theta } = \hat { r } _ { \theta } + \lambda \Delta c \left( \mathrm { A } _ { } \right)$ ppendix C.2). Running standard DPO with $\tilde { \pi } _ { \mathrm { r e f } }$ thus replaces $\ell _ { R }$ by $\ell _ { \mathrm { t i l t } } = - \log { \sigma ( \beta \tilde { r } _ { \theta } ) }$ , and since $\lambda \Delta c$ does not depend on θ, the gradient direction is unchanged and only the argument of σ shifts:

$$
\nabla \ell _ { \mathrm { t i l t } } = - \beta \Big ( 1 - \sigma \big ( \beta \hat { r } _ { \theta } + \beta \lambda \Delta c \big ) \Big ) \nabla \hat { r } _ { \theta } .\tag{6}
$$

The tilt also links the shift to constrained alignment: because the optimal DPO policy is proportional to the reference times $\exp ( r / \beta )$ , training against $\tilde { \pi } _ { \mathrm { r e f } }$ targets the cost-penalized reward $r - \beta \lambda c$ , the form used by Lagrangian approaches [9, 16], which typically raise λ while an estimated cost exceeds a prescribed threshold; CoDPO instead raises λ when the constraint risk measured on the training pairs rises, as described next.

Setting the shift strength online. How much the constraint cost still rises under the coarse weights depends on how training unfolds, which cannot be known before training, so λ is adjusted during training. Adjusting λ requires deciding when to increase it and by how much. When to increase λ is decided by the change in constraint risk, measured every K training steps on the training pairs through the estimate

Algorithm 1 CoDPO training procedure   
Require: Preference dataset D, scorer s, cost c, reference policy $\pi _ { \mathrm { r e f } } .$ , temperature ${ \overline { { \beta , } } }$ learning rate $\eta ,$ steps T, interval K,   
step fraction α   
Ensure: Trained policy $\pi _ { \theta ^ { \ast } }$   
1: Compute $\Delta { { s } _ { i } }$ and $\Delta { { c } _ { i } }$ for all pairs   
2: Compute p<sub>c</sub> and $\rho _ { \mathrm { e q } } ^ { * } ,$ using the degenerate conventions of §4.1 when $p _ { c } \in \{ 0 , 1 \}$   
3: if $\dot { \rho } _ { \mathrm { e q } } ^ { * ^ { - } } \geq \bar { 1 }$ then   
4: Set w<sub>i</sub> ← 1 for all pairs   
5: else if $\rho _ { \mathrm { e q } } ^ { * } = 0$ then   
6: Set $w _ { i } \gets 1$ for matching pairs and $w _ { i } \gets 0$ for conflict pairs   
7: else   
8: Solve Eq. (4) for $\tau ^ { * }$ by bisection and set $w _ { i }$ using Eq. (3)   
9: end if   
10: Compute $\lambda _ { \mathrm { t i l t } }$ using Eq. (8)   
11: Initialize $\lambda  \alpha \lambda _ { \mathrm { t i l t } }$ and $C _ { \mathrm { p r e v } }  \widehat { C } _ { \mathcal { D } } ( \theta _ { 0 } )$   
12: for $t = 1 , \dots , T$ do   
13: Sample mini-batch B   
14: Update θ using $\nabla _ { \theta } \mathcal { L } _ { \mathrm { C o D P O } } ( \theta )$   
15: if t mod $K = 0$ 0 then   
16: $C _ { \mathrm { c u r r } } \gets \widehat { C } _ { \mathcal { D } } ( \theta _ { t } )$   
17: $\mathbf { i f } \ C _ { \mathrm { c u r r } } > C _ { \mathrm { p r e v } }$ then   
18: λ ← min $\left( \lambda + \alpha \lambda _ { \mathrm { t i l t } } , \lambda _ { \mathrm { t i l t } } \right)$   
19: end if   
20: C<sub>prev</sub> ← C<sub>curr</sub>   
21: end if   
22: end for

$$
\widehat { C } _ { \mathcal { D } } ( \theta ) = \frac { 1 } { \left| \mathcal { D } \right| } \sum _ { i } \left[ q _ { \theta , i } c ( x _ { i } , y _ { w , i } ) + \left( 1 - q _ { \theta , i } \right) c ( x _ { i } , y _ { l , i } ) \right] , \qquad q _ { \theta , i } = \frac { \pi _ { \theta } \left( y _ { w , i } \mid x _ { i } \right) } { \pi _ { \theta } \left( y _ { w , i } \mid x _ { i } \right) + \pi _ { \theta } \left( y _ { l , i } \mid x _ { i } \right) } ,\tag{7}
$$

where $q _ { \theta , \mathbf { \nu } }$ <sub>i</sub> is the probability the current policy assigns to $y _ { w , i }$ within the pair, a normalized policy probability rather than the Bradley–Terry probability $\sigma ( \beta \hat { r } _ { \theta } )$ . Because $\widehat { C } _ { \mathcal { D } }$ uses only the responses $y _ { w , i }$ and $y _ { l , i }$ already in the training data, rather than new responses sampled from the policy, it adds no generation cost; it rises when, across pairs, the policy shifts probability toward the costlier response of each pair on balance. How much to increase it is set by the same first-order analysis as $\rho _ { \mathrm { e q } } ^ { * }$ , which, under the same approximations and for the unmodulated loss, gives the shift strength that cancels first-order drift (Appendix C.4),

$$
\lambda _ { \mathrm { t i l t } } = \operatorname* { m a x } \left( 0 , \frac { 2 \mathbb { E } _ { \mathcal { D } } [ \Delta c ] } { \beta \mathbb { E } _ { \mathcal { D } } [ \Delta c ^ { 2 } ] } \right) .\tag{8}
$$

Because the coarse weights already remove part of this drift, setting $\lambda \ : = \ : \lambda _ { \mathrm { t i l t } }$ outright could overcorrect, so we increase λ gradually, in increments of $\alpha \lambda _ { \mathrm { t i l t } }$ , and cap it at $\lambda _ { \mathrm { t i l t } }$ . Concretely, λ starts at $\alpha \lambda _ { \mathrm { t i l t } }$ , so that the shift is active from the first step; whenever $\hat { C } _ { D }$ has increased between two consecutive evaluations, which indicates that the residual drift has grown and the current shift is too weak, λ is raised by $\alpha \lambda _ { \mathrm { t i l t } }$ , up to $\lambda _ { \mathrm { t i l t } }$ , and otherwise it is kept unchanged, since a stable or falling $\widehat { C } _ { \mathcal { D } }$ indicates that the current shift suffices. Taking the maximum with 0 in $\lambda _ { \mathrm { t i l t } }$ prevents a negative $\lambda ,$ which would reverse the shift. It also switches the fine level off on self-correcting data: there $\mathbb { E } _ { \mathcal { D } } [ \Delta c ] \leq 0$ , which is equivalent to $\rho _ { \mathrm { e q } } ^ { * } \geq 1$ , so $\lambda _ { \mathrm { t i l t } } = 0$ and, with all coarse weights at 1, CoDPO reduces to vanilla DPO.

## 4.3 Combining the Two Levels

CoDPO combines the coarse weights and the margin shift in a single loss:

$$
\mathcal { L } _ { \mathrm { C o D P O } } ( \theta ) = - \frac { 1 } { \vert \mathcal { B } \vert } \sum _ { i \in \mathcal { B } } w _ { i } \cdot \log \sigma \big ( \beta \hat { r } _ { \theta } ^ { ( i ) } + \beta \lambda \Delta c _ { i } \big ) ,\tag{9}
$$

where the weights $w _ { i }$ are fixed before training (Section 4.1) and the shift $\lambda \Delta c _ { i }$ is adjusted during training (Section 4.2). The complete training procedure is summarized in Algorithm 1. The offline calibration steps are performed once before training, in $O ( | \mathcal { D } | )$ time per bisection iteration. Each optimization step has the same cost as vanilla DPO because $w _ { i } , \Delta c _ { i } ,$ and λ enter the loss only as scalar coefficients. Additional overhead comes only from evaluating the constraint-risk estimate $\widehat { C } _ { \mathcal { D } } ( \theta _ { t } )$ to update λ, which uses forward passes on the fixed training pairs and no generation, and is needed only once every $K = 5 0 0$ steps.

Table 1: Experimental domains. $p _ { c } \colon$ conflict rate. $\rho _ { \mathrm { e q } } ^ { * }$ : pre-alignment diagnostic (clipped to $[ 0 , 1 ] )$ Zero-cross.: estimated value of $\rho$ at which drift changes sign in Table 2.
<table><tr><td>Domain</td><td>Pairwise preference</td><td>Pointwise constraint</td><td>Pairs</td><td> $p _ { c }$ </td><td> $\rho _ { \mathrm { e q } } ^ { * }$ </td><td>Zero-cross.</td></tr><tr><td>HelpSteer2</td><td>Helpfulness rating</td><td>Response length</td><td>7,220</td><td>57%</td><td>0.71</td><td>~0.76</td></tr><tr><td>GSM8K</td><td>Explanation detail</td><td>Answer correctness</td><td>14,284</td><td>71%</td><td>0.40</td><td>~0.27</td></tr><tr><td>ARC</td><td>Reward-model score</td><td>Answer correctness</td><td>12,580</td><td>58%</td><td>0.72</td><td>~0.61</td></tr><tr><td>Math-Calc</td><td>Calculation density</td><td>Answer correctness</td><td>10,387</td><td>56%</td><td>0.80</td><td>~0.50</td></tr><tr><td>TL;DR</td><td>Human preference</td><td>Summary length</td><td>92,534</td><td>58%</td><td>0.59</td><td>~0.43</td></tr><tr><td>PKU-SafeRLHF</td><td>Helpfulness</td><td>Safety score</td><td>73,907</td><td>2.5%</td><td>1.0</td><td></td></tr></table>

## 5 Experiments

We evaluate on a suite of domains spanning diverse constraint types (Table 1). Most are conflict-dominated, where preference and pointwise criterion frequently disagree (HelpSteer2 [12], GSM8K [13], ARC [14], Math-Calc, TL;DR [1]). PKU-SafeRLHF [10] serves as a self-correcting control where the diagnostic predicts no intervention $( \rho _ { \mathrm { e q } } ^ { * } \ \ge \ 1$ , clipped to 1.0). Math-Calc is constructed from GSM8K [13] with calculation density as the preference signal. We use the conflictdominated domains for drift-curve validation and a representative subset (HelpSteer2, GSM8K, ARC, Math-Calc) for full method comparison against baselines.

We use Qwen2.5-7B-Instruct [39] as the default base model with LoRA [40], DPO with $\beta { = } 0 . 1$ and CoDPO with p=8, K=500, α=0.1 (full hyperparameters in Appendix A). For robustness, we additionally test three base models (DeepSeek-R1-Distill-Llama-8B [8], Mistral-7B-Instructv0.3 [41], Qwen3-8B [42]) and three preference losses (IPO [5], SimPO [7], KTO [6]). We compare against five recent methods: SafeDPO [18], PD-DPO [16], C2-DPO [17], MODPO [20], and Rewarded Soups [19], with vanilla $\mathrm { D P O } \left( \rho { = } 1 \right)$ and hard filtering $( \rho { = } 0 )$ as references.

We evaluate each model on the two candidate outputs in each preference pair. The expected constraint cost is

$$
\mathbb { E } [ c ] = P _ { \theta } ( y _ { w } \succ y _ { l } \mid x ) c ( x , y _ { w } ) + P _ { \theta } ( y _ { l } \succ y _ { w } \mid x ) c ( x , y _ { l } ) ,
$$

where the pairwise probabilities are obtained from the policy’s likelihoods of the two candidates. Drift is measured as $\mathbb { E } [ c ( \theta ) ] - \mathbb { E } [ c ( \theta _ { \mathrm { r e f } } ) ]$ , preference learning as $P _ { \theta } ( \mathrm { c h o s e n } ) = P _ { \theta } ( y _ { w } \succ y _ { l } \mid x )$ and generation quality as pairwise LLM-as-judge win rates using JudgeLM-13B [43] with positiondebiased scoring.

## 5.1 Drift Curves and Diagnostic Validation

To test the balance in Theorem 1, we vary the average conflict weight $\rho$ over {0, 0.3, 0.5, 0.7, 1} using the modulation family of CoDPO, train a model at each value, and measure drift and P(chosen) on the training pairs (Table 2). The two ends of this range correspond to familiar settings: $\rho = 0$ removes all conflict pairs, as hard filtering does, and $\rho = 1$ is vanilla DPO.

Table 2 shows three patterns that hold in all five conflict-dominated domains. First, drift increases monotonically with $\rho .$ Second, drift is negative at $\rho = 0$ , where only matching pairs drive training, and positive at $\rho = 1$ , where conflict pairs keep full weight, consistent with the balance between repair and pressure in Theorem 1. Third, P(chosen) largely increases with $\rho ,$ , so removing conflict pairs to suppress drift also weakens preference learning. Hard filtering $( \rho = 0 )$ keeps only 7% to 41% of the gain in P(chosen) that vanilla DPO achieves over the base model on HelpSteer2, GSM8K, ARC, and Math-Calc, and on TL;DR it falls below the base model. Removing all conflict pairs thus suppresses drift at a large cost in preference learning, which motivates keeping $\rho$ near the value at which drift changes sign rather than at 0.

This value can be read off Table 2 (Zero-cross. in Table 1). Computed from dataset statistics alone, the diagnostic agrees with this behavior: $\rho _ { \mathrm { e q } } ^ { * } < 1$ in all five domains, matching the positive drift at $\rho = 1$ , and $\rho _ { \mathrm { e q } } ^ { * }$ lies within 0.05 to 0.30 of the zero crossing, a gap we attribute to the first-order approximation. On PKU-SafeRLHF, $\rho _ { \mathrm { e q } } ^ { * } \geq 1$ matches its negligible drift. The diagnostic thus decides whether to intervene and sets the strength of the coarse weights.

Table 2: Drift curves across six domains. Each row corresponds to a different average conflict weight $\rho ;$ Base denotes the reference model before alignment. The constraint cost is response length (characters) on HelpSteer2 and TL;DR, binary incorrectness probability on GSM8K, ARC, and Math-Calc, and based on the safety score on PKU-SafeRLHF (Appendix A).
<table><tr><td rowspan="2">ρ</td><td colspan="2">HelpSteer2 E[c] Drift</td><td rowspan="2">E[c]</td><td colspan="2">GSM8K</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">ARC Drift</td><td rowspan="2"></td><td colspan="3">Math-Calc</td><td colspan="3">TL;DR</td><td colspan="3">PKU-SafeRLHF</td></tr><tr><td>P(ch)</td><td>Drift</td><td>P(ch)</td><td>E[c]</td><td>P(ch)</td><td>E[c]</td><td>Drift</td><td>P(ch)</td><td>E[c]</td><td>Drift P(ch)</td><td>E[c]</td><td>Drift</td><td>P(ch)</td></tr><tr><td>Base</td><td>1227</td><td></td><td>.454</td><td>.363</td><td></td><td>.351</td><td>.525</td><td></td><td>.439</td><td>.363</td><td></td><td>.420</td><td>126.5</td><td></td><td>.511</td><td>.4956</td><td></td><td>.396</td></tr><tr><td>0.0</td><td>1182</td><td>-46</td><td>.489</td><td>.225</td><td>-.138</td><td>.358</td><td>.409</td><td>-.116</td><td>.473</td><td>.227</td><td>-.136</td><td>.516</td><td>121.4</td><td>-5.1</td><td>.479</td><td>.4915</td><td>-.004</td><td>.418</td></tr><tr><td>0.3</td><td>1197</td><td>-30</td><td>.532</td><td>.379</td><td>+.016</td><td>.423</td><td>.506</td><td>-.019</td><td>.514</td><td>.337</td><td>-.026</td><td>.597</td><td>125.9</td><td>-0.6</td><td>.556</td><td>.4922</td><td>-.003</td><td>.418</td></tr><tr><td>0.5</td><td>1212</td><td>-15</td><td>.550</td><td>.392</td><td>+.029</td><td>.434</td><td>.519</td><td>-.006</td><td>.523</td><td>.363</td><td>.000</td><td>.623</td><td>126.8</td><td>+0.3</td><td>.566</td><td>.4923</td><td>-.003</td><td>.418</td></tr><tr><td>0.7</td><td>1223</td><td>-5</td><td>.560</td><td>.404</td><td>+.041</td><td>.443</td><td>.530</td><td>+.005</td><td>.528</td><td>.375</td><td>+.012</td><td>.631</td><td>127.2</td><td>+0.7</td><td>.568</td><td>.4926</td><td>-.003</td><td>.417</td></tr><tr><td>1.0</td><td>1247</td><td>+20</td><td>.566</td><td>.422</td><td>+.059</td><td>.453</td><td>.544</td><td>+.019</td><td>.532</td><td>.389</td><td>+.026</td><td>.652</td><td>127.5</td><td>+1.0</td><td>.567</td><td>.4928</td><td>-.003</td><td>.417</td></tr></table>

Table 3: Comparison with prior work across four domains. Drift $= \mathbb { E } [ c ( \theta ) ] - \mathbb { E } [ c ( \theta _ { \mathrm { r e f } } ) ]$ . P(ch) $= P _ { \theta } ( y _ { w } \succ y _ { l } \mid x )$ . MODPO (w=1.0) and Soups (α=0.75) use the setting with the smallest |drift|. Full hyperparameter sweeps are in Appendix D. Implementation details are given in Appendix A.
<table><tr><td rowspan="2">Method</td><td colspan="2">HelpSteer2</td><td colspan="2">GSM8K</td><td colspan="2">ARC</td><td colspan="2">Math-Calc</td></tr><tr><td>Drift</td><td>P(ch)</td><td>Drift</td><td>P(ch)</td><td>Drift</td><td>P(ch)</td><td>Drift</td><td>P(ch)</td></tr><tr><td>Base</td><td></td><td>.454</td><td></td><td>.351</td><td></td><td>.439</td><td></td><td>.420</td></tr><tr><td>Vanilla</td><td>+59</td><td>.566</td><td>+.129</td><td>.453</td><td>+.055</td><td>.532</td><td>+.071</td><td>.652</td></tr><tr><td>SafeDPO</td><td>-132</td><td>.437</td><td>-.353</td><td>.284</td><td>-.493</td><td>.412</td><td>-.352</td><td>.442</td></tr><tr><td>PD-DPO</td><td>-128</td><td>.452</td><td>-.360</td><td>.289</td><td>-.509</td><td>.427</td><td>-.357</td><td>.449</td></tr><tr><td>C2-DPO</td><td>≈0</td><td>.457</td><td>+.004</td><td>.353</td><td>+.003</td><td>.442</td><td>≈0</td><td>.423</td></tr><tr><td>MODPO</td><td>-74</td><td>.502</td><td>−.180</td><td>.365</td><td>-.158</td><td>.466</td><td>−.175</td><td>.514</td></tr><tr><td>Soups</td><td>+3</td><td>.509</td><td>+.018</td><td>.406</td><td>-.010</td><td>.480</td><td>+.001</td><td>.540</td></tr><tr><td>CoDPO</td><td>+18.7</td><td>.640</td><td>+.082</td><td>.551</td><td>+.046</td><td>.622</td><td>+.066</td><td>.757</td></tr></table>

## 5.2 Comparison with Prior Work

Table 3 compares CoDPO with five recent baselines across four conflict-dominated domains, and the results show three patterns. First, SafeDPO [18] and PD-DPO [16], which enforce a constraint on the expected cost, lower the constraint cost well below the reference level, and their P(chosen) remains near or below that of the base model. C2-DPO [17], which constrains the update itself, keeps both the constraint cost and P(chosen) near their initial values. These methods are designed to satisfy a constraint rather than to retain preference learning at limited drift, which is what our evaluation measures. Second, MODPO [20] and Rewarded Soups [19] expose a trade-off hyperparameter, and sweeping it traces a frontier between drift and preference learning (Tables 6 and 7 in Appendix D). Third, among all methods and configurations with drift comparable to or below that of CoDPO, CoDPO attains the highest P(chosen) in all four domains, and the configurations with higher P(chosen), MODPO with constraint weight w ≤ 0.5, have roughly 1.5 to 5 times its drift. Relative to vanilla DPO, CoDPO also reduces drift while raising P(chosen) in all four domains.

## 5.3 Contributions of Coarse Weights and Margin Shift

Table 4 separates the contributions of the two components by comparing vanilla DPO, coarse weights alone (Eq. 3 with λ=0), and the full CoDPO (coarse weights with the adaptive margin shift) across four conflict-dominated domains. Coarse control alone reduces drift in all four domains. Adding the margin shift reduces drift further in every case. This additional reduction is largest on GSM8K (+17 percentage points) and smallest on HelpSteer2 (+2 percentage points).

## 5.4 Robustness Across Models and Losses

On HelpSteer2, across four base models (Qwen2.5-7B [39], DeepSeek-8B [8], Mistral-7B [41], Qwen3-8B [42]; Table 4), vanilla DPO drifts upward in every case and the coarse weights reduce

Table 4: Drift under vanilla training, coarse control only, and full CoDPO, across domains (Qwen2.5- 7B), base models (HelpSteer2), and preference losses (HelpSteer2; IPO with β=0.5, SimPO with $\beta { = } 2 , \gamma { = } 0 . 5 ;$ coarse control at $\rho _ { \mathrm { e q } } ^ { * } )$ . Coarse control reduces drift in every setting.
<table><tr><td></td><td colspan="4">Domains (Qwen2.5-7B)</td><td colspan="3">Base models (HelpSteer2)</td><td colspan="3">Losses (HelpSteer2)</td></tr><tr><td></td><td>HelpSteer2</td><td>GSM8K</td><td>ARC</td><td>Math-Calc</td><td>DeepSeek-8B</td><td>Mistral-7B</td><td>Qwen3-8B</td><td>IPO</td><td>SimPO</td><td>KTO</td></tr><tr><td>Vanilla</td><td>+59</td><td>+.129</td><td>+.055</td><td> $+ . 0 7 1$ </td><td>+18</td><td>+92</td><td>+34</td><td>+15</td><td>+74</td><td>+72</td></tr><tr><td>Coarse</td><td>+20</td><td>+.105</td><td>+.049</td><td>+.068</td><td>-7</td><td>+45</td><td>+0.4</td><td>-36</td><td>-1</td><td>+16</td></tr><tr><td>Combined</td><td>+18.7</td><td>+.082</td><td>+.046</td><td> $+ . 0 6 6$ </td><td>-7</td><td>+41</td><td>+0.4</td><td></td><td></td><td></td></tr></table>

Table 5: LLM-as-judge evaluation (JudgeLM-13B) on HelpSteer2: CoDPO compared pairwise against each method.
<table><tr><td></td><td>Vanilla DPO</td><td>SafeDPO</td><td>PD-DPO</td><td>C2-DPO</td><td>MODPO</td><td>Soups</td></tr><tr><td>CoDPO wins</td><td>14.1%</td><td>86.3%</td><td>73.9%</td><td>12.7%</td><td>52.2%</td><td>13.5%</td></tr><tr><td>Opp. wins</td><td>10.5%</td><td>0.0%</td><td>3.0%</td><td>10.7%</td><td>7.7%</td><td>10.0%</td></tr><tr><td>Tie</td><td>75.5%</td><td>13.6%</td><td>23.2%</td><td>76.6%</td><td>40.1%</td><td>76.5%</td></tr></table>

drift in every case, with the margin shift reducing it further on Qwen2.5 and Mistral. Applied as a data-level plug-in to IPO [5], SimPO [7], and KTO [6] (Table 4), the coarse weights also reduce drift in all three cases.

## 5.5 LLM-as-Judge Evaluation

To evaluate generation quality beyond probability-level metrics, we compare CoDPO pairwise with each method on all 7,220 HelpSteer2 prompts using JudgeLM-13B [43] with position-debiased scor ing (Table 5), and CoDPO wins more often than it loses against every compared method. Compared with SafeDPO [18], PD-DPO [16], and MODPO [20], CoDPO wins 52%–86% of comparisons, in line with their lower P(chosen) in Table 3. Compared with vanilla DPO [4], C2-DPO [17], and Rewarded Soups [19], most comparisons are ties (75%–77%), so CoDPO reduces drift relative to vanilla DPO without a change in generation quality detectable by this judge. GPT-4o as a second judge agrees: CoDPO ties vanilla DPO and is preferred over MODPO 44.9% to 3.9% (Appendix F.3).

## 5.6 Ablations

Steepness parameter p. The weighting family of Eq. 3 recovers the PSDPO weighting at $p { = } 1$ and approaches a hard threshold on conflict severity as p→∞. Table 8 sweeps p on HelpSteer2: drift reduction increases monotonically with p (22% to 75%), while P(chosen) remains essentially constant (.640–.644). Due to compute constraints, we use $p { = } 8$ in all experiments without tuning it per domain, although larger p lowers drift further here.

Dynamic diagnostic validity. To examine the initialization approximation $\sigma ( \beta \hat { r } _ { \theta } ) \approx 1 / 2$ , we track $\rho ^ { * } ( t )$ during vanilla DPO training on HelpSteer2, recomputing only the learning signal from the actual $\sigma ( \beta \hat { r } _ { \boldsymbol { \theta } } )$ on all 7,220 pairs: $\rho ^ { * } ( 0 ) = 0 . 7 1 0$ matches $\rho _ { \mathrm { e q } } ^ { * } = 0 . 7 0 9$ , and as $\sigma \to 0 . 9 4 6 , \rho ^ { * } ( t )$ moves to 0.603 (gap 0.11). This is consistent with the balance point shifting during training, which motivates the adaptive margin shift; Appendix E reports the trajectory of λ during training.

## 6 Conclusion

We studied constraint drift under heterogeneous supervision and showed that, for small updates, it depends on the balance between conflict-side pressure and matching-side repair. From this balance we derive a diagnostic, $\rho _ { \mathrm { e q } } ^ { * }$ , that is computed from dataset statistics before training and indicates whether a dataset needs intervention. When it does, CoDPO down-weights conflict pairs through the asymmetric weighting of PSDPO [11], with a strength set by $\rho _ { \mathrm { e q } } ^ { * }$ rather than tuned, and shifts the DPO margin during training to address the remaining drift. Empirically, the intervention decision given by $\rho _ { \mathrm { e q } } ^ { * }$ matches the observed drift on all evaluated domains, and in our main comparison CoDPO reduces drift while largely preserving the preference learning of vanilla DPO. A natural next step is to track the balance as the model changes during training, rather than only at initialization, and to extend the analysis to settings where the preference pairs themselves are generated during training [44].

## Acknowledgments and Disclosure of Funding

This work used the high-performance GPU cluster acquired with support from the U.S. National Science Foundation (NSF) under Award #2511989, “Research Infrastructure: MRI: Acquisition of a High-Performance GPU Cluster for Large-Scale AI.” This work also used Delta and DeltaAI at the National Center for Supercomputing Applications (NCSA) through allocation CIS260780 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program [45], which is supported by NSF grants #2138259, #2138286, #2138307, #2137603, and #2138296. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF. The authors declare no competing interests.

## References

[1] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. In NeurIPS, 2020.

[2] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In NeurIPS, 2022.

[3] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

[4] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS, 2023.

[5] Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Rémi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In AISTATS, 2024.

[6] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. KTO: Model alignment as prospect theoretic optimization. In ICML, 2024.

[7] Yu Meng, Mengzhou Xia, and Danqi Chen. SimPO: Simple preference optimization with a reference-free reward. In NeurIPS, 2024.

[8] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645:633–638, 2025.

[9] Juntao Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe RLHF: Safe reinforcement learning from human feedback. In ICLR, 2024.

[10] Jiaming Ji, Donghai Hong, Borong Zhang, Boyuan Chen, Josef Dai, Boren Zheng, Tianyi Alex Qiu, Jiayi Zhou, Kaile Wang, Boxun Li, et al. PKU-SafeRLHF: Towards multi-level safety alignment for LLMs with human preference. In ACL, 2025.

[11] Siwei Meng, Yawei Luo, Shu Zhang, and Ping Liu. When physical preferences meet semantic constraints: Physical and semantic direct preference optimization for text-to-video generation. In ACM MM, 2026.

[12] Zhilin Wang, Yi Dong, Olivier Delalleau, Jiaqi Zeng, Gerald Shen, Daniel Egert, Jimmy J Zhang, Makesh Narsimhan Sreedhar, and Oleksii Kuchaiev. HelpSteer 2: Open-source dataset for training top-performing reward models. In NeurIPS Datasets and Benchmarks Track, 2024.

[13] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[14] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

[15] Akifumi Wachi, Thien Q Tran, Rei Sato, Takumi Tanabe, and Youhei Akimoto. Stepwise alignment for constrained language model policy optimization. In NeurIPS, 2024.

[16] Yihan Du, Seo Taek Kong, and R. Srikant. Provably convergent primal-dual DPO for constrained LLM alignment. arXiv preprint arXiv:2510.05703, 2025.

[17] Kavosh Asadi, Xingzi Xu, Julien Han, Ege Beyazit, Idan Pipano, Dominique Perrault-Joncas, Shoham Sabach, Mohammad Ghavamzadeh, and Karim Bouyarmane. C2-DPO: Constrained controlled direct preference optimization. TMLR, 2026.

[18] Geon-Hyeong Kim, Yu Jin Kim, Byoungjip Kim, Honglak Lee, Kyunghoon Bae, Youngsoo Jang, and Moontae Lee. SafeDPO: A simple approach to direct preference optimization with enhanced safety. In ICLR, 2026.

[19] Alexandre Rame, Guillaume Couairon, Corentin Dancette, Jean-Baptiste Gaya, Mustafa Shukor, Laure Soulier, and Matthieu Cord. Rewarded soups: towards Pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards. In NeurIPS, 2023.

[20] Zhanhui Zhou, Jie Liu, Jing Shao, Xiangyu Yue, Chao Yang, Wanli Ouyang, and Yu Qiao. Beyond one-preference-fits-all alignment: Multi-objective direct preference optimization. In Findings of ACL, 2024.

[21] Xun Deng, Han Zhong, Rui Ai, Fuli Feng, Zheng Wang, and Xiangnan He. Less is more: Improving LLM alignment via preference data selection. In NeurIPS, 2025.

[22] Shangpin Peng, Weinong Wang, Zhuotao Tian, Senqiao Yang, Xing Wu, Haotian Xu, Chengquan Zhang, Takashi Isobe, Baotian Hu, and Min Zhang. Uni-DPO: A unified paradigm for dynamic preference optimization of LLMs. In ICLR, 2026.

[23] Teng Xiao, Yige Yuan, Huaisheng Zhu, Mingxiao Li, and Vasant G Honavar. Cal-DPO: Calibrated direct preference optimization for language model alignment. In NeurIPS, 2024.

[24] Sayak Ray Chowdhury, Anush Kini, and Nagarajan Natarajan. Provably robust DPO: Aligning language models with noisy feedback. In ICML, 2024.

[25] Junkang Wu, Yuexiang Xie, Zhengyi Yang, Jiancan Wu, Jinyang Gao, Bolin Ding, Xiang Wang, and Xiangnan He. β-DPO: Direct preference optimization with dynamic β. In NeurIPS, 2024.

[26] Moxin Li, Yuantao Zhang, Wenjie Wang, Wentao Shi, Zhuo Liu, Fuli Feng, and Tat-Seng Chua. Selfimprovement towards Pareto optimality: Mitigating preference conflicts in multi-objective alignment. In Findings ofACL, 2025.

[27] Zhihao Xu, Yongqi Tong, Xin Zhang, Jun Zhou, and Xiting Wang. Understanding conflicts in multiobjective alignment through reward consistency. In Findings ofACL, 2026.

[28] Jia Zhang, Yao Liu, Chen-Xi Zhang, Yi Liu, Yi-Xuan Jin, Lan-Zhe Guo, and Yu-Feng Li. Data selection for LLM alignment using fine-grained preferences. In ICLR, 2026.

[29] Peter Chen, Xiaopeng Li, Xi Chen, and Tianyi Lin. Reward-free alignment for conflicting objectives. In ICML, 2026.

[30] Wenxuan Zhang, Philip H. S. Torr, Mohamed Elhoseiny, and Adel Bibi. Bi-factorial preference optimization: Balancing safety-helpfulness in language models. In ICLR, 2025.

[31] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In NeurIPS, 2020.

[32] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. In NeurIPS, 2021.

[33] Kailai Yang, Zhiwei Liu, Qianqian Xie, Jimin Huang, Erxue Min, and Sophia Ananiadou. Selective preference optimization via token-level reward function estimation. In EMNLP, 2025.

[34] Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. A long way to go: Investigating length correlations in RLHF. In COLM, 2024.

[35] Ryan Park, Rafael Rafailov, Stefano Ermon, and Chelsea Finn. Disentangling length from quality in direct preference optimization. In Findings of ACL, 2024.

[36] Noam Razin, Sadhika Malladi, Adithya Bhaskar, Danqi Chen, Sanjeev Arora, and Boris Hanin. Unintentional unalignment: Likelihood displacement in direct preference optimization. In ICLR, 2025.

[37] Yi Ren and Danica J. Sutherland. Learning dynamics of LLM finetuning. In ICLR, 2025.

[38] Afra Amini, Tim Vieira, and Ryan Cotterell. Direct preference optimization with an offset. In Findings of ACL, 2024.

[39] An Yang et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

[40] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In ICLR, 2022.

[41] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, et al. Mistral 7B. arXiv preprint arXiv:2310.06825, 2023.

[42] An Yang et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[43] Lianghui Zhu, Xinggang Wang, and Xinlong Wang. JudgeLM: Fine-tuned large language models are scalable judges. In ICLR, 2025.

[44] Zixuan Xu, Haoxuan Chen, Jiangxin Sun, and Jian-Fang Hu. GPDPO: Learning from reward conflicts via group Pareto-dominance policy optimization. arXiv preprint arXiv:2610.35480, 2026.

[45] Timothy J. Boerner, Stephen Deems, Thomas R. Furlani, Shelley L. Knuth, and John Towns. ACCESS: Advancing innovation: NSF’s advanced cyberinfrastructure coordination ecosystem: Services & support. In Practice and Experience in Advanced Research Computing (PEARC ’23), 2023.