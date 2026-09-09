# WHY SHARED ATTENTION VECTORS FAIL: A CASE FOR OUTCOME-INDEXED TUNING

A PREPRINT

Lenard Dome

Department of Psychiatry and Psychotherapy, Faculty of Medicine University of Tübingen, Tübingen, Germany   
German Center for Mental Health (DZPG), Tübingen, Germany lenard.dome@uni-tuebingen.de

## ABSTRACT

Dimensional attention in learning is often implemented as a globally shared attention vector, where each stimulus dimension corresponds to a single scalar. These scalars are learned by models through gradientdescent on error, where predictive features acquire more salience. We show that under multi-outcome learning, where models predict more than one outcome, this shared vector becomes unstable; it collapses to its bounds and prevents the models from learning meaningful attentional tunings for learning and generalization. We address this by introducing an outcome-indexed attentional matrix that converts globally shared attentional tuning into an outcome-indexed representation. We present an analysis of the unstable shared vectors and derive the conditions under which it holds. Empirically, three synthetic experiments benchmark the proposed attention matrices and show that they converge to meaningful representations, something shared attention vectors fail to do. These results suggest that outcome-indexed attentional matrices are a general fix for gradient-based attentional processes, which improves models of learning under multi-outcome conditions.

Keywords outcome-indexed attention · feed-forward networks · attention shift · multi-outcome learning · gradient descent on error

## 1 Introduction

Attention is a quintessential part of learning. The capacity of an organism–or any information-processing system–to choose and direct itself towards diagnostic information, while disregarding irrelevant ones, is a powerful adaptive ability. As a driving explanatory mechanism, attention has been shown to account for a wide range of phenomena (Le Pelley et al., 2016; Don et al., 2021; Paskewitz and Jones, 2020, 2023; Livesey et al., 2025). Selective attention to features of a stimulus has also been proposed as a formal requirement for models of categorization (Kruschke, 1993).

One of the most influential attentional frameworks was proposed by Mackintosh (1975), who conceptualized attention as salience underlying each stimulus feature, represented by a feature-specific scalar – collected in a vector of saliences. These saliences are derived from experience, where attention to a given feature is determined by how well that feature predicts a given outcome. Eye-tracking data extensively corroborated this attentional allocation account, where eye-fixation proportions were taken to correspond to the mechanisms specified in attentional theories (e.g. Le Pelley et al., 2016; Easdale et al., 2019; Beesley et al., 2015; Don et al., 2019; Wills et al., 2007;

Walker et al., 2019; Stojic et al.´ , 2020). This framework has influenced a number of formal models (e.g. Kruschke, 2001, 1992; Paskewitz and Jones, 2020, 2023), which provide extensive formalism for the process of allocating attention to predictive features. In all of these instantiations, salience is feature-specific, clamped to [0, ∞), stored in a globally shared attention vector, and adjusted using gradient descent on error.

In a recent work, we have discovered that globally shared attention vectors become volatile under specific parameter and environmental constraints (Dome and Wills, 2025). Because attention weights are clamped between 0 and ∞, models reset attention weights to 0 every time the update pushes values below this boundary. This clamping mechanism is part of the model specification that discards genuinely non-diagnostic cues: the resetting of attentional weights is an intended consequence of large updates (Kruschke, 2001; Paskewitz and Jones, 2020). This resetting deploys a legitimate function in singleoutcome cases. Here, we show that this mechanism becomes unstable under multi-outcome learning, remains independent of the step-size, and prevents models to learn what feature to attend. We identify the summation-across-outcomes as the primary cause, which makes the collapse of scalar salience values indiscriminate with respect to how predictive those features actually are (a feature diagnostic for outcome A still nets a below-boundary displacement once gradients from outcome B, C, are summed onto the globally shared salience scalar). We provide a principled solution that enables the encoding of a much richer attentional mapping. Furthermore, our solution improves model’s capacity to account for complex real-world behavior.

## 2 Preliminaries

We consider the problem for a set of learning models that adjust salience weights to minimize error. These models use a nominal input representation that encodes presence or absence of a feature as 0 or 1, such that each stimulus is represented as vector $S = \{ s _ { 1 } , s _ { 2 } , . . . , s _ { n } \}$ , with n representing the number of possible features of which the stimulus can take. Following in the Mackintosh (1975) framework, each input to the system correspond to an underlying scalar salience, $\eta = \{ \eta _ { 1 } , \eta _ { 2 } , \dots , \eta _ { n } \}$ with a $[ 0 , \infty )$ bound. The model then combines S with $\eta$ to generate attention gains, $^ { g , }$ at the beginning of each trial:

$$
g _ { i } = \eta _ { i } s _ { i }\tag{1}
$$

These attention gains are normalized through their vector $p \textmd { - }$ norm:

$$
a _ { i } = \frac { g _ { i } } { \left( \sum _ { j = 1 } ^ { n } | g _ { j } | ^ { p } \right) ^ { \frac { 1 } { p } } }\tag{2}
$$

with $p$ representing a brutality parameter, controlling the degree of attentional competition between currently present input dimensions. Most often, these normalized attention strengths are combined with connection weights to produce model predictions, O, along $K$ outcomes (e.g. usually implemented as output units in feed-forward connectionist networks):

$$
o _ { k } = \sum _ { i } w _ { k i } a _ { i }\tag{3}
$$

After making a prediction, models receive feedback. That feedback is used to calculate the following error for the derivations:

$$
\delta _ { k } = \lambda _ { k } - o _ { k } ,\tag{4}
$$

$$
L = \frac { 1 } { 2 } \sum _ { k } \delta _ { k } ^ { 2 }\tag{5}
$$

where $\lambda$ is a teaching vector supplied to the model. Salience is adjusted via gradient descent on error: predictive stimuli acquires higher attention weights, whereas unpredictive stimuli reduces in salience. With the salience weights constrained to be non-negative via a hard projection (clamp), the attention shifts can be written as:

$$
\Delta g _ { i , j + 1 } ^ { \prime } = - \rho { \frac { \partial { \cal L } } { \partial g _ { i , j } } } , ~ \mathrm { r e i t e r a t e s ~ t e n ~ t i m e s }\tag{6}
$$

$$
g _ { i } ^ { \prime } = \operatorname* { m a x } \left[ 0 , g _ { j } ^ { \prime } \right]\tag{7}
$$

where $\rho$ is the step-size, t is the trial, $j$ is the current iteration for the descent, and $g ^ { \prime }$ is the updated attention gain for each iteration. Attention shifts are a non-linear function (gradient changes as attention changes), which means that the shift cannot be achieved with a single large step along the gradient. Therefore, Equation 6 reiterates 10 times. The starting value of $g _ { j = 1 } ^ { \prime }$ is the current $g$ on that particular trial. The expanded equation for Equation 6 for the attention shift yields

$$
\Delta g _ { j + 1 } ^ { \prime } = \rho s _ { i } \| g _ { j } ^ { \prime } \| _ { p } ^ { - 1 } \sum _ { k } ( W _ { k i } s _ { i } - a _ { i } ^ { \prime p - 1 } o _ { k } ^ { \prime } ) \delta _ { k } ^ { \prime }\tag{8}
$$

where $\boldsymbol { o } ^ { \prime } , \boldsymbol { a } ^ { \prime } , \boldsymbol { g } ^ { \prime }$ are recalculated on each $j$ iteration. Critically, on any of these iterations, $g ^ { \prime }$ values are clamped between $[ 0 , + \infty )$ . Finally, η is updated via state displacement, similar to reconstruction error in recirculation networks (Hinton and McClelland, 1987; O’Reilly, 1996), where α is a learning rate:

$$
\eta _ { t + 1 } = \operatorname* { m a x } \Bigl [ 0 , \eta _ { t } + \alpha ( g ^ { \prime } - g ) \Bigr ] .\tag{9}
$$

## 3 Problem Statement

Scalar salience values intended to live between $[ 0 , \infty )$ become unstable when aggressive updates force the salience to reset at 0. The attention shifts (settling dynamics) operate on a single shared state (vector) g with the aggregated loss:

$$
L = \sum _ { k } L _ { k }\tag{10}
$$

where $L _ { k }$ is the k-outcome loss: $\textstyle { \frac { 1 } { 2 } } \delta _ { k } ^ { 2 }$ . The gradient descent used in the ten iterations of the attention shift sums over all outcome-specific error signal:

$$
\frac { \partial L } { \partial g } = \sum _ { k } \frac { \partial L _ { k } } { \partial g }\tag{11}
$$

The attention shift updates on each j iteration is:

$$
g _ { i , j + 1 } ^ { \prime } = g _ { i , j } ^ { \prime } \ - \ \rho \sum _ { k } { \frac { \partial L _ { k } } { \partial g _ { i } } }\tag{12}
$$

After the settling, or stabilization as Kruschke (2001) called it, the global salience is updated via a state displacement and clamped to be non-negative:

$$
\eta _ { i , t + 1 } = \operatorname* { m a x } [ 0 , \eta _ { i , t } + \alpha ( g _ { i } ^ { \prime } - g _ { i } ) ]\tag{13}
$$

The displacement $( g _ { i } ^ { \prime } - g _ { i } )$ grows with the number of active outcomes because the summed gradient pulls $g _ { i } ^ { \prime }$ further from its initial value $g _ { i , j = 1 } ^ { \prime }$ . When

$$
- \alpha ( g _ { i } ^ { \prime } - g _ { i } ) > \eta _ { i , t }\tag{14}
$$

the clamp fires and $\eta _ { i }$ resets to zero; the projection maps the value to the boundary for the next trial:

$$
\eta _ { t + 1 } = 0 .\tag{15}
$$

For any model M with K simultaneously active outcomes, where $\lvert K \rvert > 1$ , the magnitude of the attention update scales as $O ( K )$ . This is true for all models satisfying the following conditions:

1. Additive loss across outcomes. The total loss decomposes across outcomes, $L = \sum _ { k } L _ { k }$ , where $L _ { k }$ is the error associated between stimulus and outcome k.

2. Gradient descent on attention. Attention shift updates via $\begin{array} { r } { \Delta g = - \rho \frac { \partial L } { \partial g } } \end{array}$ , where step-size is $\rho > 0$

3. Hard non-negativity. Attention weights are constrained to $\eta \geq 0$ via a hard projection.

4. Same-sign gradient reinforcement. For multiple simultaneously active outcomes k, the partial derivatives $\partial L / \partial g$ for non-predicted outcomes share a sign and is not offset by the minority of the predicted outcomes.

5. Multi-outcome activations. More than one outcome can be active on a single trial, $| K | > 1$

Remark 1. The same-sign reinforcement condition is a property of a feature under multi-outcome learning. On a trial with K co-active outcomes, let’s assume that feature i is predictive of some outcomes but not others. For each k outcome that i can predict, $\partial L _ { k } / \partial g _ { i }$ <sub>i</sub> increases $_ { g _ { i } ; }$ and for every outcome it does not predict, the shift lowers $g _ { i }$ . These descents share a sign within each group, so the summed $\textstyle \sum _ { k } \partial L _ { k } / \partial g _ { i }$ is dominated by whichever group is larger. A feature that is diagnostic for a minority of the co-active outcomes is therefore driven down by the non-predicting majority, and the magnitude of the suppressing group of outcomes grows with the number of outcomes the feature fails to inform. Once the growth exceeds the shared $\eta _ { i , t } .$ , the clamp fires and resets it to 0. The failure is fundamentally multi-outcome: at $K = 1$ there is no majority to outvote and a dropped feature on a single-outcome trial discards a genuinely uninformative feature (Kruschke, 2001). In multi-outcome learning, the mechanism strips attention from a cue in proportion to how many outcomes it is irrelevant to: it is punished for being selectively informative.

Remark 2. As a consequence of Remark 1, a shared salience can fail in a second, quieter way, even where the boundary is never approached. When features are predictive of some outcomes and not predictive of others, the summation-acrossoutcomes will have a mix of positive and negative signs, which cancel each other out. The mechanism destroys attention to a feature because it excessively punishes a feature for only being useful for some outcomes but not all. Both the settling attention shift and the salience updates are driven by the aggregated gradient $\textstyle \sum _ { k } \partial L _ { k } / \partial g _ { i }$ . Suppose cue i is diagnostic in opposing directions for two co-active outcomes — its influence should be amplified for outcome k but attenuated for outcome $k ^ { \prime } - { \bf s 0 }$ that $\partial { \cal L } _ { k } / \partial g _ { i }$ and $\partial L _ { k ^ { \prime } } / \partial g _ { i }$ carry opposite signs. Although each demand is individually large, $| \dot { \mathcal { \partial } } L _ { k } / { \partial g _ { i } } | , ~ | { \partial L _ { k ^ { \prime } } } / { \partial g _ { i } } | \gg 0$ their sum cancels:

$$
\sum _ { k } { \frac { \partial L _ { k } } { \partial g _ { i } } } \approx 0 .\tag{16}
$$

The shared salience then receives no net update and $\eta _ { i }$ is frozen, unable to acquire the outcome-specific attention the task requires because it is lost in the summation. This failure is therefore independent of the non-negativity constraint: it afflicts any rule that collapses signed, outcome-specific gradients onto a single shared state represented as a scalar.

## 4 Architectural Fix

Here, we propose outcome-indexed attention as an alternative mechanism, which removes the summation-over-outcomes from Equation 11. Individual features encode more granular information across outcomes that cannot be collapsed into a single scalar. In the event of multiple co-active outcomes, a single cue might be preferentially predictive of some while being quite uninformative for others. Similarly, the input configuration (compound cues) will determine how salient cues must be given what we are trying to predict. This means that the predictive value of a cue is not globally fixed, but locally determined – it is outcome-driven. This outcome-indexed attention weight follows from the problem of the cue being connected to multiple outcomes simultaneously. A single (scalar) salience per stimulus is the wrong level of description that cannot account for how informative this stimulus is for this outcome given this cue configuration. Different cue configurations produce different predictiveness profiles across outcomes, so attention has to be indexed by outcomes, not just the cue. Thus, in this model, attention weight is indexed by the connection it modulates. Below, we implement this architectural change.

Replace the shared state $\eta _ { i }$ with an outcome-indexed state $\eta _ { k i }$ where the normalized attention will be k-specific:

$$
g _ { k i } = \eta _ { k i } \times s _ { i }\tag{17}
$$

These attention gains are normalized through their k-specific vector p-norm:

$$
a _ { k i } = { \frac { g _ { k i } } { \left( \sum _ { j } { | g _ { k j } | ^ { p } } \right) ^ { \frac { 1 } { p } } } }\tag{18}
$$

Here, we can apply each outcome’s gradient independently:

$$
\Delta g _ { k i , j + 1 } ^ { \prime } = - \rho \frac { \partial L _ { k } } { \partial g _ { k i } }\tag{19}
$$

$$
= \rho s _ { i } \| g _ { k , j } ^ { \prime } \| _ { p } ^ { - 1 } \delta _ { k } ^ { \prime } \biggl ( W _ { k i } s _ { i } - a _ { k i } ^ { \prime p - 1 } o _ { k } ^ { \prime } \biggr )\tag{20}
$$

The summation over k is removed, which we extrapolate to the loss function:

$$
L _ { k } = \frac { 1 } { 2 } ( t _ { k } - o _ { k } ) ^ { 2 }\tag{21}
$$

Each row of the attention matrix settles under its own error signal. The attention update becomes:

$$
\eta _ { k i , t + 1 } = \operatorname* { m a x } [ 0 , \ \eta _ { k i , t } + \alpha ( g _ { k i } ^ { \prime } - g _ { k i } ) ]\tag{22}
$$

The displacement $( g _ { k i } ^ { \prime } - g _ { k i } )$ is now driven by a single outcome’s gradient, so its magnitude no longer scales with the number of active outcomes.

## 5 Simulations

We evaluate and compare the shared vector and attention matrices across three simulations procedurally increasing the inputoutput mapping complexity. In all simulations below, we will pretrain model weights with a delta-rule network (Gluck and Bower, 1988; Rescorla and Wagner, 1972) to develop non-zero and meaningful input-output representations before applying the attentional shift mechanism outlined above. All starting weights were initialized to a non-zero value by sampling from a normal distribution with a mean of 0 and standard deviation of 0.025, and the learning rate set to 0.1. The network was trained for 50 epochs, with each epoch comprising a single presentation of each stimulus. Order of presentation was randomized between each epoch. After this training, we apply the attention shift mechanism for globally shared salience vectors (Equation 6) and outcome-indexed attention weight matrices (Equation 19); the updates are further constrained through a squashing hyperbolic tangent functions, we acquired similar results without a squashing function. For the exact equations, see Appendix A. Condition 2 of Section 3 requires $\rho > 0 ;$ ; it places no upper bound on the step-size and the collapse is claimed to be for the class of shared attention vectors, not for a particular parameterization. Thus, we sweep ρ over a 0-2 range instead of fixing it across simulations. Below, we measure the stability of these mechanisms as proportion of stimulus that triggered the clamping mechanism; the boundary hit is defined as the trigger, which are all instances when attention to a feature moves below 0 as per Equation 6. Table 1 shows the abstract design for the three following simulations, including the stimulus representations and the set (T) of teaching vectors corresponding to each stimulus. We fixed the stimulus set, S, throughout the simulations.

## 5.1 Distinctive Singular: minimal case with no conflict

The first instance begins with the least demanding case. The problem is defined with a disjoint outcome space of $| K | =$ 3, with $| T _ { i } | ~ = ~ 1$ , where $T _ { i }$ is the target set for stimulus i with its teaching vector, and |T| is the cardinality of $T$ . Each present stimulus predicts exactly one positive outcome (receives excitation on a single output node) and faces $| K | - 1$ absent outcomes (receives no excitation), so for $| K | > 2$ the absent group strictly outnumbers the positively predictive one. Table 1 shows the set of $T$ used for the current simulations under Distinct Singular header. This experiment is designed to test for same-sign suppression and general boundary collapse in Remark 1.

The first row of Figure 1 shows the proportion of features driven to the boundary for each iteration of the gradient descent, across the $\rho$ sweep. The attention shift for shared vectors consistently resets all feature-specific scalar by the end of the attention shift, and its terminal state floors all values, resulting in $g ^ { \prime } = 0$ producing $0 - g = - g$ in the attention update; attention will decrease for all cues regardless of their informativeness. The outcome-indexed attention matrix settles on non-zero and informative values. For most iterations of the attention shift, a substantial proportion of outcome-indexed attention weights survive, resulting in an informative terminal state that encodes what the shared vector cannot: differential and selective informativeness.

The shared vector collapsed to floor value across all four features: $S _ { 1 }$ and $S _ { 4 }$ , which are unique to a single stimulus, are indistinguishable from $S _ { 2 }$ and $S _ { 3 } ^ { \mathrm { ~ ~ } } .$ , which appear in two; see first column of Figure 2. The vector carries no information about which features are informative because it carries no information at all. The attention weight matrix, run on the same input-output pairs, is structured: $\eta _ { k i }$ is elevated for 8 out of the 12 featureoutcome mappings and is at 0 for the remainder. For example, $S _ { 1 }$ is unique to stimulus A and is connected to $O _ { 2 }$ through excitatory and $O _ { 1 }$ through inhibitory connections. Interestingly, inhibitory connections are reliably amplified through the attentional weights. Its excitatory connection to $O _ { 3 }$ might seem surprising, but it is a function of the delta rule applied during learning. As the delta rule calculates error as the sum of errors across input nodes for each outcome, the positive update to $S _ { 1 }  O _ { 3 }$ is the result of the inhibitory connection developed from $S _ { 3 }  O _ { 3 } ,$ so that the output node activation matches the teaching signal of 0.

## 5.2 Distinctive Multi-Outcome: no outcome overlap with multiple excited output units

In the second instance, we increase the problem size to $| K | = 5$ while keeping the disjoint outcome space. In this problem setup, $| T _ { i } | \in \{ 1 , 2 \}$ . The feature’s demands remain same-signed for the most part, but now a majority suppression group can exist.

As before, the shared vector collapsed to floor across all four features, see middle column of Figure 2. Although features present in stimulus A retains more of its informativeness relative to the previous simulations for longer and for smaller step-sizes; see second row of Figure 1. Attention matrices however encode robust representations and retain where the feature is informative.

Table 1: Stimulus Input Patterns and Feedback Vectors Across Overlap Conditions. Rows correspond to stimulus-teacher pairs, such that stimulus A is horizontally followed by its respective teaching vector.
<table><tr><td rowspan="3">Stimulus</td><td colspan="4">Input Pattern (S)</td><td colspan="13">Feedback Vectors Used in the Three Simulations (T)</td></tr><tr><td rowspan="2"> $S _ { 1 } \quad S _ { 2 } \quad S _ { 3 } \quad S _ { 4 }$ </td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td colspan="5">Distinct Multi-OutcomeS</td><td colspan="5">Shared Outcome Space Distinct Singular</td><td colspan="3"></td></tr><tr><td> $O _ { 1 }$ </td><td> $O _ { 2 }$ </td><td> $O _ { 3 }$ </td><td> $O _ { 4 }$ </td><td> $O _ { 5 }$ </td><td> $O _ { 1 }$ </td><td> $O _ { 2 }$ </td><td> $O _ { 3 }$ </td><td> $O _ { 4 }$ </td><td> $O _ { 5 }$ </td><td> $O _ { 1 }$ </td><td> $O _ { 2 }$ </td><td> $O _ { 3 }$ </td></tr><tr><td>A</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td></td><td>0</td><td>1</td><td>0</td><td>1</td><td>0</td></tr><tr><td>B</td><td>0</td><td>1</td><td>1</td><td>0</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td><td>0</td><td></td><td>1</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td></tr><tr><td>C</td><td>0</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td></td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td></tr></table>

![](images/4689a2c74b7e23bfd4e8f86af42f50ac9835095b8aafcf50f2011575610a9bb4.jpg)  
Figure 1: The boundary-hit sweeps across ρ: rows are the three stimulus sets (ordered from top to bottom by how much they force stimuli to compete), columns show the three stimuli, x-axis show the iteration in the attention shift mechanism, y-axis shows the proportion of eligible attention shifts reset at zero after crossing the boundary. Colour shows step-size, $\rho ,$ one curve per value; the attention representation is shown as shape and line-type: solid with circles for the shared vector, dashed with crossed empty squares for the outcome-indexed attention matrix.

## 5.3 Shared Outcome Spaces: overlapping co-active outcomes

In this last instance, we further increase complexity by introducing a shared outcome space with $| T _ { i } | \in \{ 2 , 3 \}$ , such that each stimulus has a single output unit that is excited and overlaps with the target teaching vector for another. Table 1 shows the outcome set T for this simulation under Shared Outcome Space heading. $S _ { 3 }$ is present in both A and B, and must support $O _ { 2 }$ and $O _ { 5 }$ on $\mathbf { A }$ trials while supporting $O _ { 1 }$ and $O _ { 3 }$ on B trials; $S _ { 2 }$ is present in both B and $C ,$ , supporting $O _ { 1 }$ and $O _ { 3 }$ on one and $O _ { 3 } , O _ { 4 }$ and $O _ { 5 }$ on the other. $S _ { 2 }$ and $S _ { 3 }$ each carry demands of opposite sign within a trial; thus opposite signs coexist within a trial. This is the silent failure mode described in Remark 2. The shared salience for these features receives no net (or infinitesimal) updates, while the attention matrix stores the opposing demands on outcome-indexed rows; see Figure 2.

![](images/a641c58add46f91c53840bfb283e65e51d4752e1ab9682eb88c270e6fdd95123.jpg)  
Figure 2: The three stimulus sets side by side (columns, ordered by how much they force stimuli to compete), each with the final attention gains reached after the final iteration (top row; the per-outcome attention matrix above the single shared vector) and the sign of its pretrained input-to-output connection weights (bottom row). Attention weights is shown on a continuous color scale, and normalized to a common 0-1 scale across columns, so allocations are comparable between stimulus sets. Red color indicates that the connection between input-output pairs are excitatory, blue indicates that they are inhibitory.

## 6 Summary

Here we presented the conditions under which attentional learning fails in feed-forward network models of learning, and proposed outcome-driven attentional learning as a solution to the breakdown. This result has implications for a range of models using attention shift as an update mechanism, such as EXIT (Kruschke, 2001), RASHNL (Kruschke and Johansen, 1999), and their neural network derivatives (Paskewitz and Jones, 2020). The three simulations we ran established that: the collapse is not the by-product of step-size as it occurs across all explored range of $\rho > 0 ;$ the sign-cancellation failure described in Remark 2 is independent of the non-negativity constrains; lastly, our proposed model architecture (outcome-indexed attention weight matrices) are a sufficient solution – decoupling gradients by outcomes removes the failure modes without altering the underlying learning rule. The fix reduces to one structural change of

$$
\begin{array} { r l r } { \sum _ { k } \frac { \partial L _ { k } } { \partial g _ { j } } } & { { } \longrightarrow } & { \frac { \partial L _ { k } } { \partial g _ { k j } } } \end{array}\tag{23}
$$

applied independently per outcome k for the attention matrix $\eta \doteq \mathbb { R } _ { \ge 0 } ^ { K \times J }$ . This solution produces several theoretically interesting consequences. The outcome-indexed attention weights enable features to take on outcome-specific importance, so that different features are selectively activated depending on what the system is attempting to predict. Attentional reallocation within each trial is determined by what the system is trying to do rather than cue-specific associability independent of the system’s overall purpose. Features can be diagnostic of some outcomes, but not others, and this information is used depending on what available outcomes are excitable.

Across our simulations in Section 5.3 and 5.2, outcomes are shared between inputs. These scenarios have strong implications for extending similar attentional processes to reinforcement learning environment with probabilistic feature-outcome mappings (Jones and Canas, 2010; Canas and Jones, 2010). In this case, features are connected to more than one output nodes via excitatory connections, which will cause scalar salience to break down. The attention matrix is structurally incapable of having that instability.

We also observed an interesting mapping between attention and connection weights. In a matrix-like representation, dimensional attention vectors encode information about both excitatory and inhibitory connections separately, which are selectively amplified by the attentional gating procedure. In almost all simulations, inhibitory connections were always amplified, but excitatory connection were less likely to increase in salience. This configuration was sufficient for the model to learn the current input-output mappings, but it invokes interesting implications for early phases of learning. Features with close-to-zero initial weights become predictive of absence of an outcome early on and acquire strong attentional weight. Excitatory connections to outcomes then develop undisturbed with standard excitatory connection that do not need attentional augmentation. However, initializing connection weights close to zero is a choice, and principled alternative approaches exist (Spicer et al., 2021; Rumelhart et al., 1986).

We do not present evidence that prior models are inadequate. On the contrary, we kept their core computational principles (attention adjusted on gradient descent on error) intact and extended it to more challenging environment. Mackintosh (1975) formalizes salience, also called associability, as the property of a feature that is acquired through its history of predictive success. Our results do not contradict this position, as our attention matrix architecture becomes indistinguishable from the Mackintosh tradition under K = 1.

In psychology, we often stabilize our models by reducing the complexity of the experiments (“externally restricting the input environment”; Grossberg, 1976a,b). The problem than becomes one of model capacity and not adequacy. The increase in environmental complexity presented here makes cognitive models more ecologically valid, which in turn improves the mapping between the model, the experimental paradigms, and the real-world conditions. In the current case, improved model capacity allows the computational mechanism to generalize to multiple co-active K outcomes. In practice, we often predict more than a single event. If we diagnose a disease early, we predict not just the disease label, but also symptoms that are yet to be experienced, what potential treatment might be used given hospital resources or symptom combinations, and predict costs for the patient. In many instances, the number of outcomes we anticipate exceeds the number of features presently observed.

## 7 Conclusion

Selective attention implemented as a globally shared salience vector is unstable under multi-outcome learning. Its instability emerges from its architecture, persists across a range of stepsizes, and remains invisible under single-outcome paradigms which these models were designed to accommodate. We identified the instability to result from summing outcome-indexed gradients onto a single shared state. Any architecture implementing this summation is within scope. Indexing attention by outcome restores stable learning with one relatively small architectural change. Outcome-indexed attention weights also encodes more granular representations that scalar saliences cannot: informativeness of a feature is specific to the outcome and the configuration in which it appears. The architecture change proposed here remains minimal, and it is a precondition for extending attentional learning for richer and more demanding environments in which real-world predictions usually take place.

## Acknowledgements

I would like to thank Xin Sui for helpful discussions on how to approach gradient descent. I would like to further thank Maciek Szul and Ehsan Kakaei for helpful comments on the manuscript.

## Open Science

All simulation code is available on GitHub: https://github.   
com/lenarddome/tue010-attention-unstability.

## References

Tom Beesley, Katherine P Nguyen, Daniel Pearson, and Mike E Le Pelley. Uncertainty and predictiveness determine attention to cues during human associative learning. Quarterly Journal ofExperimental Psychology, 68(11):2175–2199, 2015.

Fabian Canas and Matt Jones. Attention and Reinforcement Learning: Constructing Representations from Indirect Feedback. Proceedings ofthe Annual Meeting ofthe Cognitive Science Society, 32(32), 2010.

Lenard Dome and Andy J. Wills. G-Distance: On the comparison of model and human heterogeneity. Psychological Review, 132(3):632–655, 2025. ISSN 0033-295X. doi:10.1037/rev0000550.

Hilary J Don, Tom Beesley, and Evan J Livesey. Learned predictiveness models predict opposite attention biases in the inverse base-rate effect. Journal of Experimental Psychology: Animal Learning and Cognition, 45(2):143, 2019.

Hilary J Don, Darrell A Worthy, and Evan J Livesey. Hearing hooves, thinking zebras: A review of the inverse base-rate effect. Psychonomic Bulletin & Review, 28(4):1142–1163, 2021.

Lara C Easdale, Mike E Le Pelley, and Tom Beesley. The onset of uncertainty facilitates the learning of new associations by increasing attention to cues. Quarterly journal of experimental psychology, 72(2):193–208, 2019.

Mark A. Gluck and Gordon H. Bower. From conditioning to category learning: an adaptive network model. Journal of Experimental Psychology: General, 117(3):227–47, 1988.

S. Grossberg. Adaptive pattern classification and universal recoding: I. Parallel development and coding of neural feature detectors. Biological Cybernetics, 23(3):121–134, 1976a. ISSN 0340-1200, 1432-0770. doi:10.1007/BF00344744.

Stephen Grossberg. Adaptive pattern classification and universal recoding: II. Feedback, expectation, olfaction, illusions. Biological Cybernetics, 23(4):187–202, 1976b. ISSN 0340- 1200, 1432-0770. doi:10.1007/BF00340335.

Geoffrey E Hinton and James McClelland. Learning Representations by Recirculation. In Neural Information Processing Systems, volume 0, pages 358 – 366. American Institute of Physics, 1987.

Matt Jones and Fabian Canas. Integrating Reinforcement Learning with Models of Representation Learning. Proceedings ofthe Annual Meeting ofthe Cognitive Science Society, 32 (32), 2010.

John K. Kruschke. ALCOVE: An exemplar-based connectionist model of category learning. Psychological Review, 99(1):22– 44, 1992. ISSN 1939-1471. doi:10.1037/0033-295X.99.1.22.

John K. Kruschke. Three principles for models of category learning. In Categorization by Human and Machines : The Psychology ofLearning and Motivation, volume 29, pages 57–90. Academic Press, 1993.

John K. Kruschke. Toward a Unified Model of Attention in Associative Learning. Journal of Mathematical Psychology, 45(6):812–863, 2001. ISSN 00222496. doi:10.1006/jmps.2000.1354.

John K Kruschke and Mark K Johansen. A model of probabilistic category learning. Journal of Experimental Psychology: Learning, Memory, and Cognition, 25(5):1083, 1999.

Mike E. Le Pelley, Chris J. Mitchell, Tom Beesley, David N. George, and Andy J. Wills. Attention and associative learning in humans: An integrative review. Psychological Bulletin, 142(10):1111–1140, October 2016. ISSN 1939-1455, 0033- 2909. doi:10.1037/bul0000064.

Evan J. Livesey, Yvonne Y. Chan, Shu Chen, and Hilary J. Don. Attention and prediction error as mechanisms for theory protection? Journal of Experimental Psychology: Animal Learning and Cognition, 51(4):169–180, 2025. ISSN 2329- 8464. doi:10.1037/xan0000408.

Nicholas J Mackintosh. A theory of attention: Variations in the associability of stimuli with reinforcement. Psychological Review, 82(4):276, 1975.

Randall C. O’Reilly. Biologically Plausible Error-Driven Learning Using Local Activation Differences: The Generalized Recirculation Algorithm. Neural Computation, 8(5):895–938, 1996. ISSN 0899-7667. doi:10.1162/neco.1996.8.5.895.

Samuel Paskewitz and Matt Jones. Dissecting EXIT. Journal ofMathematical Psychology, 97:102371, 2020. ISSN 00222496. doi:10.1016/j.jmp.2020.102371.

Samuel Paskewitz and Matt Jones. A statistical foundation for derived attention. Journal of Mathematical Psychology, 112:102728, February 2023. ISSN 0022-2496. doi:10.1016/j.jmp.2022.102728.

Robert A. Rescorla and Alan R. Wagner. A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement. In A. H. Black and W. F. Prokasy, editors, Classical Conditioning II: Current Research and Theory, pages 64–99. Appleton-Century-Crofts, 1972.

David E. Rumelhart, James L. McClelland, and PDP Research Group. Parallel Distributed Processing, Volume 1: Explorations in the Microstructure of Cognition: Foundations. The MIT Press, 1986. ISBN 978-0-262-29140-8. doi:10.7551/mitpress/5236.001.0001.

Stuart Spicer, Andy Wills, Peter Jones, Chris J. Mitchell, and Lenard Dome. Representing uncertainty in the Rescorla-Wagner model: Blocking, the redundancy effect, and outcome base rate. Open Journal of Experimental Psychology and Neuroscience., 2021. doi:10.46221/ojepn.2021.6623.

Hrvoje Stojic, Eric Schulz, Pantelis Analytis, and Maarten´ Speekenbrink. It’s new, but is it good? How generalization and uncertainty guide the exploration of novel options. Journal of Experimental Psychology: General, 149(10):1878– 1907, 2020. ISSN 1939-2222. doi:10.1037/xge0000749.

Adrian R. Walker, David Luque, Mike E. Le Pelley, and Tom Beesley. The role of uncertainty in attentional and choice exploration. Psychonomic Bulletin & Review, 26(6):1911– 1916, 2019. ISSN 1531-5320. doi:10.3758/s13423-019- 01653-2.

Andy J Wills, Aureliu Lavric, GS Croft, and Timothy L Hodgson. Predictive learning, prediction errors, and attention: Evidence from event-related potentials and eye tracking. Journal ofCognitive Neuroscience, 19(5):843–854, 2007.

## A Attention shift equations

Attention shift for globally shared salience vectors are defined as:

$$
\begin{array} { l } { \displaystyle \Delta g _ { i } ^ { \prime } = \operatorname { t a n h } \left( - \rho \frac { \partial E } { \partial g _ { i } } \right) } \\ { \displaystyle \qquad = \operatorname { t a n h } \left[ \rho s _ { i } \| g ^ { \prime } \| _ { p } ^ { - 1 } \sum _ { k } ( W _ { k i } s _ { i } - a _ { i } ^ { \prime p - 1 } o _ { k } ^ { \prime } ) \delta _ { k } ^ { \prime } \right] } \end{array}\tag{24}
$$

(25)

where $\rho$ is a positive constant, denoting the step size for the gradient descent, called the attention shift rate; tanh is a squashing hyperbolic tangent function that we apply to further constrain updates to lie between -1 and 1.

Attention shift for matrix-representation for outcome-indexed attention weights are defined as:

$$
\Delta g _ { k i } ^ { \prime } = \operatorname { t a n h } \left( - \rho \frac { \partial E _ { k } } { \partial g _ { k i } } \right)\tag{26}
$$

$$
= \operatorname { t a n h } \Big [ \rho s _ { i } \| g _ { k } ^ { \prime } \| _ { p } ^ { - 1 } \delta _ { k } ^ { \prime } \big ( W _ { k i } s _ { i } - a _ { k i } ^ { \prime p - 1 } o _ { k } ^ { \prime } \big ) \Big ]\tag{27}
$$