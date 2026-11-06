# Unifying Central Flow, Rod Flow, and the Edge Coupling

Eric Regis<sup>1</sup> and Sinho Chewi<sup>1</sup>

<sup>1</sup>Yale University {eric.regis, sinho.chewi}@yale.edu

## Abstract

At the edge of stability, the iterates of gradient descent oscillate along the sharp directions of the Hessian. This poses a significant challenge for continuous-time models as the usual continuoustime approximation—the gradient flow—admits no oscillations. Several continuous-time models have been proposed to capture the behavior at the edge of stability, among them central flow and rod flow. Although both models can be interpreted as performing a coarse-graining in time, the connection between them remains poorly understood. Here we show that both flows can be derived from a third object, the edge coupling: a functional on consecutive iterates whose critical points are the fixed points and period-two orbits of gradient descent. It is revealed that the two flows correspond to coarse-grainings at diferent time scales. Rod flow coarse-grains over only the fastest time scale, while central flow coarse-grains over both the fast and intermediate time scales. Furthermore, using the edge coupling, we derive rod flows and central flows for momentum methods such as heavy ball and Adam. We verify the derived flows experimentally on representative neural network architectures, where they track the discrete iterates throughout all phases of training—including the transition from the stable regime to the edge of stability.

## Contents

1 Introduction 3   
1.1 Contributions . 5   
2 Related Work 6   
3 Background 6   
3.1 Setup 6   
3.2 Central Flow 7   
3.3 Rod Flow . 8   
3.4 Edge Coupling 8   
4 Rod Flow and Edge Coupling 9   
4.1 Gradient Descent–Ascent 9   
4.2 The Zero Fixed Point 10   
4.3 Covariance Versus Second Moment . 11   
4.4 Endpoint Rod Flow 12   
4.5 RMSProp 14   
5 Rod Flow for Momentum Methods 15   
5.1 The Phase-Space Edge Coupling 15   
5.2 Heavy Ball Rod Flow 16   
5.3 Adam 17   
6 Central Flow and Edge Coupling 18   
6.1 The Truncated Edge Coupling . 19   
6.2 Covariance Matrix as Lagrange Multiplier 19   
6.3 Persistence 20   
6.4 RMSProp 21   
7 Central Flow for Momentum Methods 22   
7.1 Eliminating the Momentum Oscillation . 22   
7.2 The Truncated Reduced Edge Coupling 22   
7.3 Persistence 23   
7.4 Adam 24   
8 Rod Flow and Central Flow 25   
8.1 Fast, Intermediate, and Slow Time Scales 25   
8.2 Central Flow as Time-Averaged Rod Flow 27   
8.3 Turning Rate of the Rod . 28   
8.4 Turning Rate for Momentum Methods 30   
8.5 Summary 31   
9 Experiments 32   
9.1 Setup 32   
9.2 Results . 32   
10 Conclusion 36   
A Notation 41   
B Edge Couplings of Structured Gradient Methods 42   
B.1 Structured Gradient Methods 42   
B.2 Heavy Ball 45   
B.3 Other Structured Gradient Methods 47   
C Nesterov Momentum 47   
C.1 Setup 48   
C.2 Edge Coupling 48   
C.3 Rod Flow . 49   
C.4 Central Flow 49   
D AdamW 51   
D.1 Setup 51   
D.2 Edge of Stability 51   
D.3 Edge Coupling 52   
D.4 Rod Flow . 55   
D.5 Central Flow 56   
E Rod Flow 57   
E.1 Extent Matrix Formulation 57   
E.2 The Noise Floor . 58   
E.3 Bias Correction 62   
F Central Flow 63   
F.1 Convex Quadratic Programs . 63   
F.2 Central Flows as Projected Dynamical Systems 65   
G Experimental Details 67   
G.1 Stable Flows 67   
G.2 Runs . 68   
G.3 Measurements 70   
G.4 Numerical Integration 71   
H Additional Experiments 73   
H.1 Flow Comparisons . 73   
H.2 Cross-Entropy Loss . 80   
H.3 Rod Turning 83   
H.4 Position and Momentum Extents . 90   
H.5 Near Two-Periodicity . 93   
H.6 The Ring and the Onset 95

## 1 Introduction

Cohen et al. [2021] observed that gradient descent operates at the edge of stability. During training, the sharpness (the largest eigenvalue of the Hessian) rises in a phase called progressive sharpening. The sharpness then hovers at 2/η, where η is the learning rate. Once there, the iterates bounce across the valley of the loss landscape rather than rolling downhill smoothly. It’s been found that momentum methods such as Adam also operate at the edge of stability [Cohen et al., 2022].

Continuous-time models are a standard tool for analyzing discrete-time dynamics. However, the edge of stability poses a significant challenge for such models. At the edge of stability, the iterates oscillate along the sharp directions of the Hessian. The usual continuous-time limit of gradient descent—the gradient flow—cannot capture these oscillations.

There are several continuous-time models of gradient descent that aim to capture the dynamics at the edge of stability [Cohen et al., 2025, Regis and Chewi, 2026b, Chodron de Courcel, 2026, Marion, 2026]. Cohen et al. [2025] introduced central flow: a continuous-time model that aims to capture not the individual iterates, but their time-averaged position. It acts as a projected gradient flow—gradient flow that is constrained to the feasible region where the sharpness is at most 2/η. The oscillations are represented by a covariance matrix that is found by solving a semidefinite complementarity problem (SDCP). In the same paper, they also developed central flows for scalar RMSProp and RMSProp.

Rod flow [Regis and Chewi, 2026b] is a continuous-time model that represents consecutive iterates as an extended one-dimensional object—a “rod”—whose midpoint and extent evolve smoothly. In contrast with the covariance matrix of central flow, the extent matrix of rod flow evolves according to an ODE. Regis and Chewi [2026a] then lifted the rod from weight space to phase space, yielding rod flows for momentum methods such as heavy ball and Adam.

![](images/288db87149e2de8cde7dd33db80bc960730fd8fec7bc22bc64d69fb6645ed265.jpg)  
Figure 1: The Edge Coupling Generates Both Rod Flow and Central Flow. Rod flow is gradient descent–ascent on the centered edge coupling, with the oscillations tracked by an explicit state variable. Central flow is constrained descent on the same objective, with the covariance matrix acting as a Lagrange multiplier. Central flow is rod flow averaged over the intermediate time scale of self-stabilization.

Central flow and rod flow both aim to model the dynamics of gradient descent when coarsegrained in time. As a result, one would expect that the two flows must capture the same underlying structure. Yet the two flows were derived under very diferent modeling assumptions. Central flow was derived by time-averaging the iterates and imposing the empirically-observed constraint that the sharpness remains at 2/η. Meanwhile, rod flow was derived by assuming that the midpoint of consecutive iterates and the outer product of their half-displacement vary smoothly. Whether a deeper link exists between them has remained unclear.

Recently, Litman [2026] introduced the edge coupling: a functional on consecutive iterates whose critical points encode the fixed points and period-two orbits of gradient descent. With it, he sheds light on why the dynamics drive the sharpness to 2/η. Regis [2026] then extended the edge coupling to heavy ball and Nesterov momentum.

Here we establish connections between all three objects: central flow, rod flow, and the edge coupling (Figure 1).

Rod flow and edge coupling. Analogous to the relationship between gradient flow and the loss function, one can derive a rod flow by taking the centered edge coupling as the generating function. However, the resulting dynamics are not gradient descent, but gradient descent–ascent: descent on the midpoint w¯ and ascent on the extent δ. This alternative formulation of rod flow fixes a subtle error in the original, which allows it to track the discrete iterates through the transition to the edge of stability. We then generalize the construction to the other optimizers of the Adam family—deriving rod flows for RMSProp, heavy ball momentum, and Adam.

Central flow and edge coupling. One can also re-derive the central flow for gradient descent from the centered edge coupling. To do so, we replace the loss in the edge coupling with its local quadratic approximation, which makes the edge coupling linear in the covariance matrix Σ. Then, instead of being governed by an ODE, Σ is determined as the solution of a maximization problem parametrized by the midpoint w¯. By imposing a persistence condition on Σ, we recover precisely the SDCP of central flow. Using a similar approach, we derive novel central flows for momentum methods such as heavy ball and Adam (Figure 2).

![](images/a498375864abfa306d761d8c0db1083a21e801a9001907f040f40c1efdd59b58.jpg)  
Figure 2: Rod Flow and Central Flow Track AdamW. AdamW on an MLP $( \eta = 1 0 ^ { - 5 }$ $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \zeta = 1 0 )$ . Left: With both rod flow and central flow, the preconditioned sharpness self-stabilizes at the correct value. Right: Both flows track the discrete iterates more accurately than the stable flow does.

Because rod flow is also generated by the edge coupling, the two flows now share a common derivation. Mathematically, they difer only in whether the oscillation is treated as a state variable or as a Lagrange multiplier.

Rod flow and central flow. Berthier [2026] derived the central flow for gradient descent using the method of multiple scales, identifying three time scales. There’s the fast time scale on which the components of the iterates along sharp directions alternate in sign. There’s the intermediate time scale on which the oscillation amplitude self-stabilizes. And there’s the slow time scale that governs the drift of the center. Building of of Berthier’s observation, we show that rod flow and central flow correspond to coarse-grainings at diferent time scales. Rod flow coarse-grains over the fast time scale only. It models the intermediate time scale dynamics of the oscillation envelope, exhibiting fluctuations at the onset of self-stabilization. In contrast, central flow coarse-grains over both the fast and intermediate time scales.

We thus unify the three objects.

## 1.1 Contributions

• A rod flow for gradient descent, from the centered edge coupling (Section 4).

• An extension of rod flow to heavy ball and Adam, from the phase-space edge coupling (Section 5).

• A derivation of the central flow for gradient descent from the centered edge coupling (Section 6).

• A construction of novel central flows for heavy ball and Adam (Section 7).

• A demonstration that rod flow and central flow coarse-grain at diferent time scales (Section 8).

• An experimental verification of rod flow and central flow (Section 9).

## 2 Related Work

Edge of stability. Jastrzębski et al. [2020] documented a break-even point on the optimization trajectories of SGD. Cohen et al. [2021] observed that full-batch gradient descent operates at the edge of stability: the sharpness rises to $2 / \eta$ and then hovers there. Cohen et al. [2022] extended the picture to momentum and adaptive methods, and Islamov et al. [2026] to non-Euclidean gradient descent. Theoretical accounts include self-stabilization [Damian et al., 2023], the two-step analysis of period-two orbits [Ahn et al., 2023, Chen and Bruna, 2023], and convergence in the unstable regime [Ahn et al., 2022, Arora et al., 2022, Zhu et al., 2023]. Others study progressive sharpening in quadratic regression [Agarwala et al., 2023], the catapult mechanism [Lewkowycz et al., 2020], and its amplification by momentum [Phunyaphibarn et al., 2024]. Further lines include loss-landscape and bifurcation analyses [Ma et al., 2022, Song and Yun, 2023], the edge of stochastic stability of minibatch SGD [Andreyev and Beneventano, 2024] and of its momentum variants [Andreyev et al., 2026], chaotic convergence to a statistical distribution [Kong and Tao, 2020], and a fixed-point analysis with a period-doubling route to chaos [Kalra et al., 2025]. Litman [2026] introduced the edge coupling, a functional on consecutive iterates whose critical points are the fixed points and period-two orbits of gradient descent. Regis [2026] extended the edge coupling to heavy ball and Nesterov momentum.

Continuous-time models. Gradient flow is the classical continuous limit of gradient descent but cannot capture the oscillations of the edge of stability. Rosca et al. [2023] study its instabilities. Modified equations from backward error analysis [Barrett and Dherin, 2021, Smith et al., 2021, Hairer et al., 2006] do not reach the oscillatory regime either. Central flow [Cohen et al., 2025] was the first continuous-time model of gradient descent, scalar RMSProp and RMSProp at the edge of stability. Berthier [2026] derived central flow for gradient descent by the method of multiple scales, identifying three time scales. Rod flow [Regis and Chewi, 2026b] models consecutive iterates as an extended one-dimensional object. Regis and Chewi [2026a] lifted the rod to phase space, deriving rod flows for momentum and preconditioned methods such as Adam. Other continuous-time models at the edge of stability include edge flow [Marion, 2026] and the mean–fluctuation model [Chodron de Courcel, 2026].

Discrete mechanics. The edge coupling can be identified with the two-period Störmer–Verlet action [Verlet, 1967, Hairer et al., 2003], connecting the edge of stability to discrete mechanics [Marsden and West, 2001]. The linear stability of periodic orbits through the Hessian of a discrete action goes back to MacKay and Meiss [1983]. Symplectic integrators and their shadow Hamiltonians are treated by Hairer et al. [2006] and Benettin and Giorgilli [1994]. A separate line connects momentum methods to variational and conformally symplectic dynamics [Wibisono et al., 2016, França et al., 2020, Muehlebach and Jordan, 2021].

## 3 Background

We briefly recap central flow, rod flow, and the edge coupling.

## 3.1 Setup

Consider gradient descent with step size η on a loss function $L : \mathbb { R } ^ { d } $ R:

$$
w _ { t + 1 } = w _ { t } - \eta \nabla L ( w _ { t } ) .\tag{1}
$$

Let $H = \nabla ^ { 2 } L$ denote the Hessian of the loss, and let $\lambda _ { 1 } ( H )$ denote the sharpness—the largest eigenvalue of the Hessian. For quadratic objectives, the GD iterates diverge when the sharpness

exceeds $2 / \eta$

Gradient descent operates at the edge of stability (EoS) when the sharpness hovers at the threshold $2 / \eta$ . At EoS, the iterates oscillate along the top eigenvectors of the Hessian, with the sharp components of the gradient alternating in sign from one iteration to the next. These oscillatory dynamics pose a significant challenge for continuous-time models that aim to describe gradient descent at the edge of stability.

## 3.2 Central Flow

Central flow [Cohen et al., 2025] is a continuous-time model of gradient descent at the edge of stability. Instead of tracking the individual oscillating iterates, it models their time-averaged position w¯ and a covariance matrix Σ which represents the oscillations. The central flow is given by the ODE:

$$
\frac { \mathrm { d } \bar { w } } { \mathrm { d } t } = - \eta \Big [ \nabla L ( \bar { w } ) + \frac { 1 } { 2 } \nabla _ { \bar { w } } \langle H ( \bar { w } ) , \Sigma \rangle \Big ] .\tag{2}
$$

The first term is the familiar gradient flow. The second term acts as a sharpness-reduction force, pushing the flow toward flatter regions of the loss landscape. Indeed, when H has only one sharp eigenvalue and the oscillations are confined to the sharp direction, we have:

$$
\begin{array} { r } { \frac { 1 } { 2 } \nabla _ { \bar { w } } \langle H ( \bar { w } ) , \Sigma \rangle = \frac { \sigma ^ { 2 } } { 2 } \nabla _ { \bar { w } } \lambda _ { 1 } ( H ( \bar { w } ) ) , } \end{array}\tag{3}
$$

where $\sigma ^ { 2 }$ is the squared magnitude of the oscillations.

Central flow requires that the following three static conditions must hold along the trajectory:

1. The covariance matrix must be positive semidefinite $( \Sigma \succeq 0 )$

2. The slack $\begin{array} { r } { \Lambda = \frac { 2 } { \eta } I _ { d } - H } \end{array}$ must also be positive semidefinite $( \Lambda \succeq 0 )$

3. The oscillations must be confined to the sharp directions $( \langle \Lambda , \Sigma \rangle = 0 )$

However, these static conditions alone are not suficient to determine the covariance matrix. To do so, one must invoke persistence: we require that the dynamics also preserve the constraints.

Let U be a matrix whose columns form an orthonormal basis for the critical directions—directions whose sharpness equals $2 / \eta$ . Let k denote the number of critical directions. We can then express the covariance matrix as $\Sigma = U X U ^ { \top }$ where X is a $k \times k$ positive semidefinite matrix. Diferentiating in time yields the following dynamic constraints:

1. The slack cannot decrease along critical directions $( U ^ { \top } \dot { \Lambda } U \succeq 0 )$

2. There can be no oscillation in directions whose slack is increasing $( \langle U ^ { \top } \dot { \Lambda } U , X \rangle = 0 )$

Together with the condition that $X \succeq 0$ , these dynamic constraints define a semidefinite complementarity problem—which determines Σ.

Central flow can also be interpreted as a projected gradient flow. Let Ω denote the feasible region, the set of points whose sharpness is at most $2 / \eta \colon$

$$
\Omega = \left\{ w \in \mathbb { R } ^ { d } : \lambda _ { 1 } ( H ) \leq 2 / \eta \right\} .\tag{4}
$$

Central flow is gradient flow constrained to $\Omega ,$ with Σ acting as the Lagrange multiplier that enforces the constraint.

## 3.3 Rod Flow

Rod flow [Regis and Chewi, 2026b] is a continuous-time model of gradient descent where consecutive iterates are modeled as an extended one-dimensional object—a “rod”. Define the midpoint w¯ and half-displacement $\hat { \delta }$ of consecutive iterates:

$$
\begin{array} { r } { \bar { w } _ { t } = \frac { 1 } { 2 } ( w _ { t } + w _ { t + 1 } ) } \end{array}\tag{5}
$$

$$
\hat { \delta } _ { t } = \frac { 1 } { 2 } ( w _ { t } - w _ { t + 1 } )\tag{6}
$$

The core modeling assumption of rod flow is that, although the individual iterates oscillate at the edge of stability, the midpoint $\bar { w } _ { t }$ and outer product $\hat { \delta } _ { t } \otimes \hat { \delta } _ { t }$ vary smoothly—and therefore can be tracked by ODEs.

Let Σ denote the extent matrix, the continuous-time analog of $\hat { \delta } \otimes \hat { \delta }$ . And let δ denote the principal eigenvector of Σ scaled by the square root of its principal eigenvalue. Although the extent matrix is not constrained to have rank 1, it has been found empirically that it can safely be approximated as rank 1.

Let $\bar { g }$ denote the average of the gradients evaluated at the rod endpoints, and let $\overline { { g ^ { \otimes 2 } } }$ denote the average of their outer products:

$$
\begin{array} { r } { \bar { g } = \frac { 1 } { 2 } \big [ \nabla L ( \bar { w } + \delta ) + \nabla L ( \bar { w } - \delta ) \big ] , \qquad \overline { { g ^ { \otimes 2 } } } = \frac { 1 } { 2 } \big [ \nabla L ( \bar { w } + \delta ) ^ { \otimes 2 } + \nabla L ( \bar { w } - \delta ) ^ { \otimes 2 } \big ] . } \end{array}\tag{7}
$$

The rod flow ODEs for $\bar { w }$ and Σ are given as:

$$
\frac { \mathrm { d } \bar { w } } { \mathrm { d } t } = - \eta \bar { g } ,\tag{8}
$$

$$
\frac { \mathrm { d } \Sigma } { \mathrm { d } t } = \frac { \eta ^ { 2 } } { 2 } \overline { { g ^ { \otimes 2 } } } - 2 \Sigma .\tag{9}
$$

The midpoint w¯ moves with the mean gradient, while the extent Σ is driven by the second moment of the endpoint gradients and damped by the −2Σ term. Along directions flatter than the $2 / \eta$ threshold, damping dominates and Σ decays quickly. As a result, the extent matrix remains confined to the sharp directions. However, along critical directions, the source and damping terms nearly cancel. Along these directions, Σ grows when the sharpness exceeds the threshold and shrinks when the sharpness falls below it. Thus, unlike with central flow, Σ is not fixed by $\bar { w } ^ { - }$ —instead, it is set dynamically by self-stabilization.

## 3.4 Edge Coupling

The edge coupling [Litman, 2026] is a functional that takes two points in weight space as arguments:

$$
\begin{array} { r } { A _ { \eta } ( w , w ^ { \prime } ) = L ( w ) + L ( w ^ { \prime } ) - \frac { 1 } { 2 \eta } \| w - w ^ { \prime } \| ^ { 2 } . } \end{array}\tag{10}
$$

The critical points of the edge coupling encode the fixed points and period-two orbits of gradient descent.

The edge coupling can be thought of as a generalization of the loss function. The loss has the property that its critical points are precisely the fixed points of gradient descent:

$$
\nabla L ( w _ { t } ) = 0 \iff w _ { t + 1 } = w _ { t } .\tag{11}
$$

The edge coupling characterizes not only the fixed points but also the period-two orbits. Taking the partial derivative of $A _ { \eta }$ with respect to w and setting it equal to zero recovers the gradient descent update equation from w to $w ^ { \prime } \colon$

$$
\nabla _ { \boldsymbol { w } } \boldsymbol { \mathcal { A } } _ { \eta } = 0 \iff \boldsymbol { w ^ { \prime } } = \boldsymbol { w } - \eta \nabla L ( \boldsymbol { w } ) .\tag{12}
$$

And because $\mathcal { A } _ { \eta }$ is symmetric under swapping $w  w ^ { \prime }$ , we also have that:

$$
\nabla _ { \boldsymbol { w } ^ { \prime } } A _ { \eta } = 0 \iff w = w ^ { \prime } - \eta \nabla L ( w ^ { \prime } ) .\tag{13}
$$

There are only two circumstances in which both criticality conditions hold: $w = w ^ { \prime }$ is a fixed point of gradient descent, or w $\neq w ^ { \prime }$ and $( w , w ^ { \prime } )$ form a period-two orbit.

$$
\nabla \mathcal { A } _ { \eta } ( w _ { t } , w _ { t + 1 } ) = 0 \iff w _ { t + 2 } = w _ { t } .\tag{14}
$$

For any smooth, invertible change of coordinates, critical points correspond bijectively between the old and new coordinates. We may therefore express the edge coupling in whatever coordinates are most convenient. Consider the coordinate system:

$$
\bar { w } = \frac { w + w ^ { \prime } } { 2 } , \qquad \delta = \frac { w - w ^ { \prime } } { 2 } .\tag{15}
$$

We can then define the centered edge coupling:

$$
\begin{array} { r l } & { \Psi ( \bar { w } , \delta ) \equiv \frac { 1 } { 2 } \mathcal { A } _ { \eta } ( \bar { w } + \delta , \bar { w } - \delta ) } \\ & { \qquad = \frac { 1 } { 2 } \big [ L ( \bar { w } + \delta ) + L ( \bar { w } - \delta ) \big ] - \frac { 1 } { \eta } \| \delta \| ^ { 2 } . } \end{array}\tag{16}
$$

(17)

The factor of $\frac { 1 } { 2 }$ , while not strictly necessary, will prove convenient later.

The edge coupling encodes not only the fixed points and period-two orbits of gradient descent, but also their linear stability—through its Hessian. In the special case where the Hessians at the two arguments agree— ${ \cal H } ( \bar { w } + \delta ) = { \cal H } ( \bar { w } - \delta )$ —the cross block $\begin{array} { r } { \nabla _ { \bar { w } \delta } ^ { 2 } \Psi = \frac { 1 } { 2 } [ H ( \bar { w } + \delta ) - H ( \bar { w } - \delta ) ] } \end{array}$ vanishes, so the Hessian of the centered edge coupling is block-diagonal. We then have that a period-two orbit is linearly stable if and only if:

$$
\nabla _ { \bar { w } \bar { w } } ^ { 2 } \Psi \succ 0 \qquad \mathrm { a n d } \qquad \nabla _ { \delta \delta } ^ { 2 } \Psi \prec 0 .\tag{18}
$$

In this case, linearly-stable period-two orbits correspond to min–max critical points of the centered edge coupling.

## 4 Rod Flow and Edge Coupling

We show that one can derive a rod flow for gradient descent as continuous-time gradient descent– ascent on the centered edge coupling. We also derive a rod flow for RMSProp.

## 4.1 Gradient Descent–Ascent

Let $[ [ g ] ]$ denote the half-diference of the gradients of the endpoints of the rod.

$$
\begin{array} { r } { \mathbb { [ } g ] = \frac { 1 } { 2 } [ \nabla L ( \bar { w } + \boldsymbol { \delta } ) - \nabla L ( \bar { w } - \boldsymbol { \delta } ) ] } \end{array}\tag{19}
$$

Taking derivatives of the centered edge coupling with respect to $\bar { w }$ and $\delta ,$ , we obtain:

$$
\nabla _ { \bar { w } } \Psi = \bar { g }\tag{20}
$$

$$
\nabla _ { \delta } \Psi = \left[ \left[ g \right] \right] - \frac { 2 } { \eta } \delta .\tag{21}
$$

Examining the right-hand side of our equation for $\nabla _ { \boldsymbol { \bar { w } } } \Psi$ , we find a familiar quantity: the average gradient ${ \bar { g } } .$ Recall that the average gradient is precisely what drives the midpoint w¯ in rod flow. The expression for $\nabla _ { \boldsymbol { \delta } } \Psi$ is also reminiscent of the right-hand side of the Σ ODE in rod flow:

$$
\frac { \mathrm { d } \Sigma } { \mathrm { d } t } = \frac { \eta ^ { 2 } } { 2 } \overline { { g ^ { \otimes 2 } } } - 2 \Sigma .
$$

Even though $\delta$ is a vector while Σ is a matrix, they nonetheless share a similar form. In rod flow, Σ relaxes toward a function of the endpoint gradients, whereas our criticality condition for δ is a self-consistency condition: $\delta$ must equal the half-diference of the endpoint gradients multiplied by $\eta / 2$

Can one derive a rod flow for gradient descent by letting w¯ and $\delta$ evolve under gradient flow on the centered edge coupling?

Yes—though there is one subtlety. For gradient flow on the loss, the stable fixed points correspond to critical points of the loss that are local minima. However, recall from (18) that for the centered edge coupling—when the endpoint Hessians agree—stable period-two orbits correspond to critical points that are minima in w¯ and maxima in δ.

So instead of gradient descent, we must perform gradient descent–ascent: descending in $\bar { w }$ and ascending in $\delta .$ This yields the following rod flow:

$$
\boxed { \begin{array} { l } { \frac { \mathrm { d } } { \mathrm { d } t } \left( \begin{array} { l } { \bar { w } } \\ { \delta } \end{array} \right) = \eta \left( \begin{array} { l l } { - I _ { d } } & { 0 } \\ { 0 } & { I _ { d } } \end{array} \right) \nabla _ { ( \bar { w } , \delta ) } \Psi } \end{array} }\tag{22}
$$

Expressing the ODEs for w¯ and δ separately, we have:

$$
{ \frac { \mathrm { d } { \bar { w } } } { \mathrm { d } t } } = - \eta { \bar { g } }
$$

$$
\frac { \mathrm { d } \delta } { \mathrm { d } t } = \eta [ g ] - 2 \delta\tag{23}
$$

(24)

## 4.2 The Zero Fixed Point

Before the edge of stability, $\delta = 0$ is a strong attractor of rod flow. This can be seen by Taylor expanding the gradient half-diference to first order in $\delta \colon$

$$
\frac { \mathrm { d } \delta } { \mathrm { d } t } = \eta [ [ g ] ] - 2 \delta \approx [ \eta H - 2 I _ { d } ] \delta\tag{25}
$$

Thus, in the stable regime—when all eigenvalues of H lie below $2 / \eta$ —each component of $\delta$ decays exponentially, at a rate proportional to the gap between the corresponding Hessian eigenvalue and $2 / \eta$

This attraction to the zero fixed point has a practical consequence: even after the sharpness crosses the $2 / \eta$ threshold, it takes time for $\delta$ to recover from the small value it decayed to during the stable phase in order to grow large enough to trigger self-stabilization.

This phenomenon is not specific to rod flow. Gradient descent exhibits the same phenomenon. Let $\begin{array} { r } { \hat { \delta } _ { t } = \frac { 1 } { 2 } ( w _ { t } - w _ { t + 1 } ) = \frac { \eta } { 2 } \nabla L ( w _ { t } ) } \end{array}$ be the half-displacement between consecutive iterates, and let $\widetilde { H } _ { t }$ be the Hessian averaged along the gradient step:

$$
\widetilde { H } _ { t } = \int _ { 0 } ^ { 1 } \nabla ^ { 2 } L \big ( w _ { t } - s \eta \nabla L ( w _ { t } ) \big ) \mathrm { d } s .\tag{26}
$$

By the fundamental theorem of calculus:

$$
\nabla L ( w _ { t + 1 } ) - \nabla L ( w _ { t } ) = - \eta \widetilde { H } _ { t } \nabla L ( w _ { t } ) .\tag{27}
$$

We can then deduce:

$$
\hat { \delta } _ { t + 1 } = \left( I - \eta \tilde { H } _ { t } \right) \hat { \delta } _ { t } .\tag{28}
$$

Let u denote an eigenvector of $\widetilde { H } _ { t }$ with eigenvalue λ. The u-component of the half-displacement is multiplied by $1 - \eta \lambda$ at every step, so it contracts exactly when $\lambda \in ( 0 , 2 / \eta )$

We have that both gradient descent and rod flow are predicted to exhibit exponential decay in the stable regime for $\hat { \delta } _ { t }$ and δ, respectively. However, it’s been empirically observed that the sharp component of gradient descent hovers at a small value instead of decaying further (Figure 14). So why does the sharp component of GD’s half-step $\hat { \delta } _ { t }$ remain small but nonzero, whereas rod flow’s δ plummets toward zero?

The diference turns out to have nothing to do with the dynamics. Rather, it’s due to the imprecision of floating-point arithmetic on computers. Each time the weights are stored during gradient descent, rounding due to finite precision injects a small but nonzero noise into the sharp component of $\hat { \delta } _ { t }$ . In our setting, it’s on the order of $1 0 ^ { - 8 }$ for float32. This small rounding noise is what causes gradient descent to self-stabilize earlier than rod flow.

To replicate the efect of finite floating-point precision, we impose a floor on δ. With this floor in place, rod flow tracks the discrete iterates accurately, including the timing of the onset of self-stabilization.

There is precedent for adding a noise floor. Edge flow [Marion, 2026], an alternative continuoustime model of gradient descent at the edge of stability, implements its own version of a noise floor. More details are provided in Appendix E.2.

## 4.3 Covariance Versus Second Moment

As observed by Marion [2026], the original formulation of rod flow [Regis and Chewi, 2026b] was unable to track the onset of self-stabilization even though it could track the GD iterates if it was initialized in the steady-state edge of stability regime. But the new formulation of rod flow is able to track the iterates through all phases of training (Figure 3). What’s the key diference between the two formulations?

It helps to compare their respective fixed points for the extent. For the edge-coupling-derived formulation of rod flow, we have that the fixed point for δ is given by the half-diference of the gradient, scaled by $\eta / 2 .$ . We can then take the outer product to get the fixed point for $\Sigma _ { \mathrm { E C } } = \delta \otimes \delta$

$$
\begin{array} { r } { \Sigma _ { \mathrm { E C } } ^ { * } = \frac { \eta ^ { 2 } } { 4 } [ [ g ] ^ { \otimes 2 } } \end{array}\tag{29}
$$

For the original formulation of rod flow, we have that the fixed point of $\Sigma _ { \mathrm { S M } }$ is given by the second moment of the gradient.

$$
\begin{array} { r } { \Sigma _ { \mathrm { S M } } ^ { * } = \frac { \eta ^ { 2 } } { 4 } \overline { { g ^ { \otimes 2 } } } } \end{array}\tag{30}
$$

We can express the second moment of the gradient in terms of the average gradient and the gradient half-diference.

$$
\begin{array} { r } { \Sigma _ { \mathrm { S M } } ^ { * } = \frac { \eta ^ { 2 } } { 4 } [ \bar { g } ^ { \otimes 2 } + [ [ g ] ] ^ { \otimes 2 } ] } \end{array}\tag{31}
$$

This is the key diference between the two formulations: in the original rod flow, Σ tracks the second moment of the gradient, whereas in the formulation derived from the edge coupling, it tracks the gradient’s covariance.

![](images/786d7ab0516cbc39d6745e5855f8b49435f1dead981b04edfe5cefc2c6f5d555.jpg)  
Figure 3: Second Moment Versus Covariance. Second-moment and covariance rod flows on an MLP $( \eta = 0 . 0 5 )$ . Left: Second-moment rod flow fails to self-stabilize at the transition, whereas covariance rod flow succeeds. Middle and right: Alignment of each flow’s extent with the mean gradient $\bar { g } _ { t }$ and with the gradient half-diference $[ [ g ] ] _ { t }$ of the discrete iterates. The second-moment rod stays aligned with the mean gradient, while the covariance rod correctly aligns with the gradient half-diference at the onset.

Why might it be better for Σ to track the covariance of the gradient rather than its second moment? Recall that rod flow was motivated by the observation that the outer product $\hat { \delta } \otimes \hat { \delta }$ varies smoothly. But this is not strictly true. We can decompose the half-displacement into the average gradient g¯ and half-diference gradient g , which alternates each iteration:

$$
\begin{array} { r } { \hat { \delta } _ { t } \approx \frac { \eta } { 2 } \left( \bar { g } + ( - 1 ) ^ { t } [ [ g ] ] \right) } \end{array}\tag{32}
$$

Taking the outer product, we have:

$$
\begin{array} { r } { \hat { \delta } _ { t } \otimes \hat { \delta } _ { t } \approx \frac { \eta ^ { 2 } } { 4 } \left( \bar { g } ^ { \otimes 2 } + [ [ g ] ^ { \otimes 2 } + ( - 1 ) ^ { t } [ \bar { g } \otimes [ [ g ] ] + [ [ g ] ] \otimes \bar { g } ] \right) } \end{array}\tag{33}
$$

We have that $\hat { \delta } _ { t } \otimes \hat { \delta } _ { t }$ varies smoothly when the alternating terms are small relative to the nonalternating terms. That’s true when $\| \bar { g } \| \gg \| [ g ] \| \mathrm { o r } \| \bar { g } \| \ll \| [ g ] \|$ . But when $\bar { g }$ and g are of similar magnitude, as is the case at the onset of self-stabilization (Figure 4), the alternating terms are comparable in magnitude to the non-alternating terms, violating the assumption that $\hat { \delta } _ { t } \otimes \hat { \delta } _ { t }$ varies smoothly. The edge-coupling-derived rod flow does not have this issue, since its $\delta$ tracks only $[ [ g ] ]$

Removing the slow-drift component also means that the edge-coupling-derived rod flow only encodes the oscillations—which makes a certain amount of physical sense.

## 4.4 Endpoint Rod Flow

Recall the uncentered edge coupling:

$$
\begin{array} { r } { A _ { \eta } ( w , w ^ { \prime } ) = L ( w ) + L ( w ^ { \prime } ) - \frac { 1 } { 2 \eta } \| w - w ^ { \prime } \| ^ { 2 } . } \end{array}\tag{10}
$$

gradient descent, MLP, MSE: the alternating term

![](images/b268e2e543606453fccd60c4b0baf6a070de93f65572cc5b9e20d30df59ea8d8.jpg)  
Figure 4: The Alternating Term at the Onset. Gradient descent on an MLP $( \eta = 0 . 0 5 )$ Top: Norms of the mean $\bar { g } _ { t }$ and the half-diference $[ [ g ] ] _ { t }$ of consecutive gradients. Before the edge of stability, $\| \bar { g } _ { t } \| \gg \| \mathbb { J } \| _ { t } \|$ . In the steady-state edge-of-stability regime, $\lVert \bar { g } _ { t } \rVert \ll \lVert \mathbb { I } g \rVert _ { t } \rVert$ . Bottom: Norm of the alternating term $\bar { g } _ { t } \otimes \mathbb { [ } g \mathbb { ] } _ { t } + \mathbb { [ } g \mathbb { ] } _ { t } \otimes \bar { g } _ { t }$ in $\hat { \delta } _ { t } \otimes \hat { \delta } _ { t }$ , relative to that of the non-alternating terms $\bar { g } _ { t } ^ { \otimes 2 } + [ [ g ] ] _ { t } ^ { \otimes 2 }$ . The alternating term is negligible before the crossing but peaks just after the onset, when $\bar { g } _ { t }$ and $[ [ g ] ] _ { t }$ are similar in magnitude.

One can actually construct a rod flow using the uncentered edge coupling:

$$
\boxed { \begin{array} { c } { \mathrm { d } } \\ { \mathrm { d } t } \end{array} } \left( \begin{array} { l } { w } \\ { w ^ { \prime } } \end{array} \right) = - \eta \left( \begin{array} { l l } { 0 } & { I _ { d } } \\ { I _ { d } } & { 0 } \end{array} \right) \nabla _ { ( w , w ^ { \prime } ) } \mathcal { A }\tag{34}
$$

Expressing the ODEs for w and $w ^ { \prime }$ separately:

$$
\frac { \mathrm { d } w } { \mathrm { d } t } = - ( w - [ w ^ { \prime } - \eta \nabla L ( w ^ { \prime } ) ] )\tag{35}
$$

$$
\frac { \mathrm { d } w ^ { \prime } } { \mathrm { d } t } = - \big ( w ^ { \prime } - [ w - \eta \nabla L ( w ) ] \big )\tag{36}
$$

Rather than tracking the midpoint w¯ and the extent $\delta ,$ this formulation tracks the two endpoints of the rod directly. Therefore, an apt name for this formulation is endpoint rod flow. We can then call the $( \bar { w } , \delta )$ formulation midpoint rod flow.

Each endpoint regresses toward a “target” created by the other endpoint. This target simul taneously ensures that the two endpoints are the correct distance apart and that both are rolling downhill in accordance with the average gradient.

Although endpoint rod flow appears superficially diferent from the midpoint formulation of rod flow, the two are equivalent—as follows from a straightforward application of the chain rule. Consequently, they perform identically in experiments (Section 9.2).

## 4.5 RMSProp

RMSProp [Tieleman and Hinton, 2012] can be considered Adam without momentum. It is an adaptive modification of gradient descent: instead of updating the parameters using the raw gradient, RMSProp maintains an exponential moving average ν of the componentwise squared gradients and scales each component of the update by the inverse square root of $\nu .$ The preconditioner P is defined as

$$
P ( \nu ) = \mathrm { d i a g } ( \sqrt { \nu } ) + \epsilon I _ { d } ,\tag{37}
$$

where $\epsilon > 0$ is a small numerical constant.

Let $v ^ { \odot 2 }$ denote the componentwise square of a vector v. The update equations for RMSProp are

$$
\begin{array} { r l } & { \nu _ { t + 1 } = \beta _ { 2 } \nu _ { t } + \left( 1 - \beta _ { 2 } \right) \nabla L ( w _ { t } ) ^ { \odot 2 } , } \\ & { w _ { t + 1 } = w _ { t } - \eta P _ { t + 1 } ^ { - 1 } \nabla L ( w _ { t } ) , } \end{array}\tag{38}
$$

where $\beta _ { 2 } \in [ 0 , 1 )$ and $P _ { t } = P ( \nu _ { t } )$

We can define an edge coupling for RMSProp by slightly modifying the edge coupling for gradient descent. The main diference is that the preconditioner $P ( \nu )$ now acts as a metric tensor, changing the norm used to measure the displacement between consecutive iterates. Because P depends on the second-moment estimate $\nu ,$ the edge coupling must also take ν as an extra argument:

$$
\begin{array} { r } { \boldsymbol { \mathcal { A } } _ { \eta , \nu } ( w , w ^ { \prime } ) = L ( w ) + L ( w ^ { \prime } ) - \frac { 1 } { 2 \eta } ( w - w ^ { \prime } ) ^ { \top } P ( \nu ) ( w - w ^ { \prime } ) . } \end{array}\tag{39}
$$

Similarly, we can also derive a centered edge coupling for RMSProp:

$$
\begin{array} { r } { \Psi _ { \eta , \nu } ( \bar { w } , \delta ) = \frac { 1 } { 2 } \big [ L ( \bar { w } + \delta ) + L ( \bar { w } - \delta ) \big ] - \frac { 1 } { \eta } \delta ^ { \top } P ( \nu ) \delta . } \end{array}\tag{40}
$$

As with gradient descent, we can derive a rod flow for RMSProp from the centered edge coupling. However, with RMSProp, only the position variable has rod structure. Because squaring kills the sign flip, the second moment ν doesn’t oscillate at the edge of stability. So ν¯ evolves as a smooth auxiliary variable. Recall that

$$
\overline { { g ^ { \odot 2 } } } = \bar { g } ^ { \odot 2 } + [ [ g ] ] ^ { \odot 2 } .\tag{41}
$$

We can then derive the following rod flow for RMSProp:

$$
\begin{array} { c }  { { \displaystyle { \boxed { \begin{array} { c } { { \underline { { { \mathrm { d } } } } } \end{array} } \left( \bar { w } \right) = \eta \left( \begin{array} { c c } { { - P ( \bar { \nu } ) ^ { - 1 } } } & { { 0 } } \\ { { 0 } } & { { P ( \bar { \nu } ) ^ { - 1 } } } \end{array} \right) \nabla _ { ( \bar { w } , \delta ) } \Psi _ { \eta , \nu } } } } } \\  { { \hphantom { { \displaystyle { \frac { \mathrm { d } } { \mathrm { d } t } } \left( \begin{array} { c } { { \underline { { { \mathrm { d } } } } \bar { \nu } } } \\ { { \mathrm { d } t } } \end{array} \right) } } \displaystyle { \frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \Big [ \bar { g } ^ { \odot 2 } + [ [ g ] ] ^ { \odot 2 } - \bar { \nu } \Big ] . } } } \end{array}\tag{42}
$$

Notice how simple the interpretation of P as a metric tensor is. We have that the identity matrix in the rod flow ODEs for GD is replaced by $P ^ { - 1 }$ . Indeed, it must be the inverse that appears here because it acts on the gradient—which is a covector.

For the individual ODEs, we have:

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } \bar { w } } { \mathrm { d } t } = - \eta P ^ { - 1 } \bar { g } , } \\ { \displaystyle \frac { \mathrm { d } \delta } { \mathrm { d } t } = \eta P ^ { - 1 } \bigl [ g \bigr ] - 2 \delta , } \\ { \displaystyle \frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \Bigl [ \bar { g } ^ { \odot 2 } + \bigl [ g \bigr ] ^ { \odot 2 } - \bar { \nu } \Bigr ] . } \end{array}\tag{43}
$$

For the endpoint rod flow, we have:

$$
\begin{array} { r l } & { \boxed { \begin{array} { c } { \underline { { \mathrm { d } } } \left( \begin{array} { c } { w } \\ { w ^ { \prime } } \end{array} \right) = - \eta \left( \begin{array} { c c } { 0 } & { P ( \bar { \nu } ) ^ { - 1 } } \\ { P ( \bar { \nu } ) ^ { - 1 } } & { 0 } \end{array} \right) \nabla _ { ( w , w ^ { \prime } ) } A _ { \eta , \bar { \nu } } , } \\ { \underline { { \mathrm { d } } } \bar { \nu } } & { = ( 1 - \beta _ { 2 } ) \big [ \frac { 1 } { 2 } \big ( \nabla L ( w ) ^ { \odot 2 } + \nabla L ( w ^ { \prime } ) ^ { \odot 2 } \big ) - \bar { \nu } \big ] . } \end{array} } } \end{array}\tag{44}
$$

Writing out the individual ODEs:

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } w } { \mathrm { d } t } = - \big ( w - \big [ w ^ { \prime } - \eta P ^ { - 1 } \nabla L ( w ^ { \prime } ) \big ] \big ) , } \\ { \displaystyle \frac { \mathrm { d } w ^ { \prime } } { \mathrm { d } t } = - \big ( w ^ { \prime } - \big [ w - \eta P ^ { - 1 } \nabla L ( w ) \big ] \big ) , } \\ { \displaystyle \frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \big [ \frac { 1 } { 2 } \big ( \nabla L ( w ) ^ { \odot 2 } + \nabla L ( w ^ { \prime } ) ^ { \odot 2 } \big ) - \bar { \nu } \big ] . } \end{array}\tag{45}
$$

## 5 Rod Flow for Momentum Methods

We derive the phase-space edge coupling of heavy ball. We then use the edge coupling to obtain rod flows for heavy ball and Adam.

## 5.1 The Phase-Space Edge Coupling

Let w and m denote the position and momentum, respectively. The update equations for heavy ball momentum [Polyak, 1964] are given as:

$$
m _ { t + 1 } = \beta m _ { t } + ( 1 - \beta ) \nabla L ( w _ { t } )\tag{46}
$$

$$
w _ { t + 1 } = w _ { t } - \eta m _ { t + 1 }\tag{47}
$$

where $\beta \in [ 0 , 1 )$ is the momentum coeficient.

Let $z = ( w , m )$ denote a point in phase space. Heavy ball admits a phase-space edge coupling: a functional $B _ { \eta , \beta } ( z , z ^ { \prime } )$ of two phase-space points whose critical points encode the fixed points and period-two orbits of heavy ball [Regis, 2026]. It’s given as:

$$
\begin{array} { r } { \mathcal { B } _ { \eta , \beta } ( z , z ^ { \prime } ) = L ( w ) + L ( w ^ { \prime } ) - \frac { 1 } { 2 \eta } \| w - w ^ { \prime } \| ^ { 2 } + \frac { \beta } { 1 - \beta } ( m - m ^ { \prime } ) \cdot ( w - w ^ { \prime } ) - \frac { \eta \beta } { 1 - \beta } m \cdot m ^ { \prime } . } \end{array}\tag{48}
$$

While this property can be verified by direct computation, we will show how one can derive the phasespace edge coupling. Furthermore, our approach applies to all “structured gradient methods ${ \mathfrak { s } } _ { - \mathrm { a } }$ family of optimization methods that includes every optimizer in the Adam family.

Structured gradient methods. Consider a discrete-time optimizer whose update rule is of the form

$$
\boldsymbol { z } _ { t + 1 } = \boldsymbol { z } _ { t } + K \nabla \mathcal { L } ( \boldsymbol { z } _ { t } ) ,\tag{49}
$$

where $\mathcal { L }$ is the generating function and K is the structure matrix. We require that the structure matrix K be both constant and symmetric. We will call such an optimizer a structured gradient method. The edge coupling is given as:

$$
\begin{array} { r } { \boxed { \mathcal { B } ( z , z ^ { \prime } ) = \mathcal { L } ( z ) + \mathcal { L } ( z ^ { \prime } ) + \frac { 1 } { 2 } ( z - z ^ { \prime } ) ^ { \top } K ^ { - 1 } ( z - z ^ { \prime } ) . } } \end{array}\tag{50}
$$

Gradient descent corresponds to ${ \mathcal { L } } = L$ and $K = - \eta I _ { d }$ . RMSProp, with the second moment ν frozen, corresponds to $\mathcal { L } = L$ and $K = - \eta P ( \nu ) ^ { - 1 }$

Let Φ denote the update map:

$$
\Phi ( z ) = z + K \nabla { \mathcal { L } } ( z ) .\tag{51}
$$

The partial gradients of the edge coupling are

$$
\nabla _ { z } \mathcal { B } ( z , z ^ { \prime } ) = K ^ { - 1 } \big ( \Phi ( z ) - z ^ { \prime } \big ) , \qquad \nabla _ { z ^ { \prime } } \mathcal { B } ( z , z ^ { \prime } ) = K ^ { - 1 } \big ( \Phi ( z ^ { \prime } ) - z \big ) .\tag{52}
$$

By inspection, one can see that $\nabla _ { ( z , z ^ { \prime } ) } B = 0$ precisely when $z ^ { \prime } = \Phi ( z )$ and $z = \Phi ( z ^ { \prime } )$ . As desired, the critical points of B encode the fixed points and period-two orbits of the optimizer.

Now consider heavy ball momentum. Heavy ball can be expressed as a structured gradient method with generating function and structure matrix:

$$
\begin{array} { r } { \mathcal { L } ( w , m ) = L ( w ) - \frac { \eta \beta } { 2 ( 1 - \beta ) } \| m \| ^ { 2 } , \qquad K = \left( \begin{array} { c c } { - \eta ( 1 - \beta ) I _ { d } } & { ( 1 - \beta ) I _ { d } } \\ { ( 1 - \beta ) I _ { d } } & { \frac { ( 1 - \beta ) ^ { 2 } } { \eta \beta } I _ { d } } \end{array} \right) . } \end{array}\tag{53}
$$

By substituting this $\mathcal { L }$ and K into our formula for the edge coupling, we obtain the phase-space edge coupling for heavy ball.

Note that while heavy ball can be expressed as a damped Hamiltonian flow, the structure matrix in that formulation is not symmetric, so it doesn’t yield a valid edge coupling. In fact, for the case of heavy ball, the generating function and symmetric structure matrix above are unique up to a trivial simultaneous rescaling. More details on edge couplings for structured gradient methods are given in Appendix B.

## 5.2 Heavy Ball Rod Flow

Define the centered coordinates:

$$
\bar { z } = \binom { \bar { w } } { \bar { m } } = \frac { z + z ^ { \prime } } { 2 } , \qquad \Delta = \binom { \delta } { \gamma } = \frac { z - z ^ { \prime } } { 2 }\tag{54}
$$

and the centered phase-space edge coupling:

$$
\begin{array} { r l } & { \Upsilon _ { \eta , \beta } ( \bar { z } , \Delta ) : = \frac { 1 } { 2 } \mathcal { B } _ { \eta , \beta } ( \bar { z } + \Delta , \bar { z } - \Delta ) } \\ & { \qquad = \frac { 1 } { 2 } [ L ( \bar { w } + \delta ) + L ( \bar { w } - \delta ) ] - \frac { 1 } { \eta } \| \delta \| ^ { 2 } + \frac { 2 \beta } { 1 - \beta } \gamma \cdot \delta - \frac { \eta \beta } { 2 ( 1 - \beta ) } ( \| \bar { m } \| ^ { 2 } - \| \gamma \| ^ { 2 } ) } \end{array}\tag{55}
$$

(56)

For gradient descent, we obtained the rod flow by performing gradient descent–ascent on the centered edge coupling. Using a similar approach, we can derive the heavy ball rod flow—though we must now incorporate the structure matrix K. By applying $K$ to the gradient in the midpoint z¯ and −K to the gradient in the extent $\Delta .$ , we obtain the following rod flow:

$$
\boxed { \begin{array} { c c } { \boxed { \mathrm { d } } } \end{array} \left( \begin{array} { c c } { \bar { z } } \\ { \Delta } \end{array} \right) = \left( \begin{array} { c c } { K } & { 0 } \\ { 0 } & { - K } \end{array} \right) \nabla _ { ( \bar { z } , \Delta ) } \Upsilon }\tag{57}
$$

Writing out the individual ODEs:

$$
\frac { \mathrm { d } \bar { w } } { \mathrm { d } t } = - \eta \big [ \beta \bar { m } + ( 1 - \beta ) \bar { g } \big ] ,
$$

$$
\frac { \mathrm { d } \bar { m } } { \mathrm { d } t } = ( 1 - \beta ) \big [ \bar { g } - \bar { m } \big ] ,
$$

$$
\frac { \mathrm { d } \delta } { \mathrm { d } t } = - 2 \delta + \eta \big [ \beta \gamma + ( 1 - \beta ) [ g ] \big ] ,\tag{58}
$$

$$
\frac { \mathrm { d } \gamma } { \mathrm { d } t } = - ( 1 + \beta ) \gamma - ( 1 - \beta ) \mathbb { I } g \mathbb { I } .
$$

Note that at $\beta = 0$ the heavy ball rod flow reduces to the gradient descent rod flow.

For the endpoint rod flow, we have:

$$
\boxed { \frac { \mathrm { d } } { \mathrm { d } t } \left( \begin{array} { c } { z } \\ { z ^ { \prime } } \end{array} \right) = \left( \begin{array} { c c } { 0 } & { K } \\ { K } & { 0 } \end{array} \right) \nabla _ { ( z , z ^ { \prime } ) } B _ { \eta , \beta } }\tag{59}
$$

Writing out the individual ODEs:

$$
\frac { \mathrm { d } w } { \mathrm { d } t } = - \big ( w - \big [ w ^ { \prime } - \eta \big ( \beta m ^ { \prime } + ( 1 - \beta ) \nabla L ( w ^ { \prime } ) \big ) \big ] \big ) ,
$$

$$
\frac { \mathrm { d } m } { \mathrm { d } t } = - \big ( m - \big [ \beta m ^ { \prime } + ( 1 - \beta ) \nabla L ( w ^ { \prime } ) \big ] \big ) ,
$$

$$
\frac { \mathrm { d } w ^ { \prime } } { \mathrm { d } t } = - \big ( w ^ { \prime } - \big [ w - \eta \big ( \beta m + ( 1 - \beta ) \nabla L ( w ) \big ) \big ] \big ) ,\tag{60}
$$

$$
\frac { \mathrm { d } m ^ { \prime } } { \mathrm { d } t } = - \big ( m ^ { \prime } - \big [ \beta m + ( 1 - \beta ) \nabla L ( w ) \big ] \big ) .
$$

Each endpoint regresses toward the heavy-ball image of the other endpoint.

An analogous rod flow can be derived for Nesterov momentum using the Nesterov phase-space edge coupling. More details are provided in Appendix C.

## 5.3 Adam

Adam [Kingma and Ba, 2015] combines momentum with the adaptive preconditioner of RMSProp. Adam’s update equations are:

$$
\begin{array} { r l } & { m _ { t + 1 } = \beta _ { 1 } m _ { t } + ( 1 - \beta _ { 1 } ) \nabla L ( w _ { t } ) , } \\ & { \nu _ { t + 1 } = \beta _ { 2 } \nu _ { t } + ( 1 - \beta _ { 2 } ) \nabla L ( w _ { t } ) ^ { \odot 2 } , } \\ & { w _ { t + 1 } = w _ { t } - \eta P _ { t + 1 } ^ { - 1 } m _ { t + 1 } , } \end{array}\tag{61}
$$

where $P ( \nu ) = \mathrm { d i a g } ( \sqrt { \nu } ) + \epsilon I _ { d ^ { - } }$ —the same as in RMSProp. For simplicity, we omitted the bias correction, though it can be incorporated into the formalism without much dificulty. Details are provided in Appendix E.3.

Treating ν as fixed, Adam is a structured gradient method on phase space. Its generating function and structure matrix are those of heavy ball except that the preconditioner P replaces the identity matrix as the acting metric tensor.

$$
\begin{array} { r }  \mathcal { L } _ { \nu } ( w , m ) = L ( w ) - \frac { \eta \beta _ { 1 } } { 2 ( 1 - \beta _ { 1 } ) } m ^ { \top } P ^ { - 1 } m , \qquad K _ { \nu } = \left( \begin{array} { c c } { - \eta ( 1 - \beta _ { 1 } ) P ( \nu ) ^ { - 1 } } & { ( 1 - \beta _ { 1 } ) I _ { d } } \\ { ( 1 - \beta _ { 1 } ) I _ { d } } & { \frac { ( 1 - \beta _ { 1 } ) ^ { 2 } } { \eta \beta _ { 1 } } P ( \nu ) \right) . } \end{array} \end{array}\tag{62}
$$

For the edge coupling, we have:

$$
\begin{array} { r } { \mathcal { B } _ { \eta , \beta _ { 1 } , \nu } ( z , z ^ { \prime } ) = L ( w ) + L ( w ^ { \prime } ) - \frac { 1 } { 2 \eta } ( w - w ^ { \prime } ) ^ { \top } P ( w - w ^ { \prime } ) + \frac { \beta _ { 1 } } { 1 - \beta _ { 1 } } ( m - m ^ { \prime } ) \cdot ( w - w ^ { \prime } ) - \frac { \eta \beta _ { 1 } } { 1 - \beta _ { 1 } } m ^ { \top } P ^ { - 1 } m ^ { \prime } . } \end{array}\tag{63}
$$

And for the centered edge coupling, we have:

$$
\begin{array} { r } { \Upsilon _ { \eta , \beta _ { 1 } , \nu } ( \bar { z } , \Delta ) = \frac { 1 } { 2 } \big [ L ( \bar { w } + \delta ) + L ( \bar { w } - \delta ) \big ] - \frac { 1 } { \eta } \delta ^ { \top } P \delta + \frac { 2 \beta _ { 1 } } { 1 - \beta _ { 1 } } \gamma \cdot \delta - \frac { \eta \beta _ { 1 } } { 2 ( 1 - \beta _ { 1 } ) } \big [ \bar { m } ^ { \top } P ^ { - 1 } \bar { m } - \gamma ^ { \top } P ^ { - 1 } \gamma \big ] . } \end{array}\tag{64}
$$

As with RMSProp, we treat the second-moment ν¯ as a smooth auxiliary variable. We then obtain the following rod flow for Adam:

$$
\begin{array} { r l } & { \boxed { \begin{array} { c } { \underline { { \mathrm { d } } } } \\ { \overline { { \mathrm { d } } } t } \end{array} } \left( \begin{array} { c c } { \bar { z } } \\ { \Delta } \end{array} \right) = \binom { K _ { \bar { \nu } } } { 0 } \boxed { 0 } \nabla _ { ( \bar { z } , \Delta ) } \Upsilon _ { \eta , \beta _ { 1 } , \bar { \nu } } , } \\ & { \qquad \underline { { \mathrm { d } } } \bar { \underline { { \nu } } } = ( 1 - \beta _ { 2 } ) \big [ \bar { g } ^ { \odot 2 } + \big [ [ g ] ^ { \odot 2 } - \bar { \nu } \big ] . } \end{array}\tag{65}
$$

Writing out the individual ODEs:

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } \bar { w } } { \mathrm { d } t } = - \eta P ^ { - 1 } \big [ \beta _ { 1 } \bar { m } + ( 1 - \beta _ { 1 } ) \bar { g } \big ] , } \\ { \displaystyle \frac { \mathrm { d } \bar { m } } { \mathrm { d } t } = ( 1 - \beta _ { 1 } ) \big [ \bar { g } - \bar { m } \big ] , } \\ { \displaystyle \frac { \mathrm { d } \bar { v } } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \big [ \bar { g } ^ { \circledast 2 } + [ g ] ^ { \circledast 2 } - \bar { \nu } \big ] , } \\ { \displaystyle \frac { \mathrm { d } \bar { \delta } } { \mathrm { d } t } = - 2 \delta + \eta P ^ { - 1 } \big [ \beta _ { 1 } \gamma + ( 1 - \beta _ { 1 } ) [ g ] \big ] , } \\ { \displaystyle \frac { \mathrm { d } \bar { \gamma } } { \mathrm { d } t } = - ( 1 + \beta _ { 1 } ) \gamma - ( 1 - \beta _ { 1 } ) [ g ] . } \end{array}\tag{66}
$$

For the endpoint rod flow, we have:

$$
\begin{array} { r l r } {  { \frac { \mathrm { d } } { \mathrm { d } t } ( \begin{array} { l } { z } \\ { z ^ { \prime } } \end{array} ) = ( \begin{array} { l l } { 0 } & { K _ { \bar { \nu } } } \\ { K _ { \bar { \nu } } } & { 0 } \end{array} ) \nabla _ { ( z , z ^ { \prime } ) } \mathcal { B } _ { \eta , \beta _ { 1 } , \bar { \nu } } , } } \\ & { } & { \frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \big [ \frac { 1 } { 2 } \big ( \nabla L ( w ) ^ { \odot 2 } + \nabla L ( w ^ { \prime } ) ^ { \odot 2 } \big ) - \bar { \nu } \big ] . } \end{array}\tag{67}
$$

Writing out the individual ODEs:

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } w } { \mathrm { d } t } = - \big ( w - \big [ w ^ { \prime } - \eta P ^ { - 1 } \big ( \beta _ { 1 } m ^ { \prime } + ( 1 - \beta _ { 1 } ) \nabla L ( w ^ { \prime } ) ) \big ] \big ) , } \\ { \displaystyle \frac { \mathrm { d } m } { \mathrm { d } t } = - \big ( m - \big [ \beta _ { 1 } m ^ { \prime } + ( 1 - \beta _ { 1 } ) \nabla L ( w ^ { \prime } ) \big ] \big ) , } \\ { \displaystyle \frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = \big ( 1 - \beta _ { 2 } \big ) \big [ \frac { 1 } { 2 } \big ( \nabla L ( w ) ^ { \circledast 2 } + \nabla L ( w ^ { \prime } ) ^ { \circledast 2 } \big ) - \bar { \nu } \big ] , } \\ { \displaystyle \frac { \mathrm { d } w ^ { \prime } } { \mathrm { d } t } = - \big ( w ^ { \prime } - \big [ w - \eta P ^ { - 1 } \big ( \beta _ { 1 } m + ( 1 - \beta _ { 1 } ) \nabla L ( w ) \big ) \big ] \big ) , } \\ { \displaystyle \frac { \mathrm { d } m ^ { \prime } } { \mathrm { d } t } = - \big ( m ^ { \prime } - \big [ \beta _ { 1 } m + ( 1 - \beta _ { 1 } ) \nabla L ( w ) \big ] \big ) . } \end{array}\tag{68}
$$

One can also derive an edge coupling and rod flow for AdamW, a modification of Adam whose update rule decays the weights towards zero. The details are provided in Appendix D.

## 6 Central Flow and Edge Coupling

We derive central flow from the centered edge coupling. Whereas the extent of rod flow is governed by an ODE, the covariance matrix of the central flow acts as a Lagrange multiplier, constraining the dynamics to the feasible region.

## 6.1 The Truncated Edge Coupling

Losses in machine learning have been found to be well described by their local quadratic approximation [Meterez et al., 2026]. Applying this approximation about the midpoint $\bar { w }$ , we have that:

$$
\begin{array} { r } { \frac { 1 } { 2 } [ L ( \bar { w } + \delta ) + L ( \bar { w } - \delta ) ] \approx L ( \bar { w } ) + \frac { 1 } { 2 } \delta ^ { \top } H ( \bar { w } ) \delta . } \end{array}\tag{69}
$$

Substituting this approximation into the centered edge coupling, we obtain the truncated edge coupling:

$$
\begin{array} { r } { \widetilde { \Psi } ( \bar { w } , \delta ) = L ( \bar { w } ) + \frac { 1 } { 2 } \delta ^ { \top } H ( \bar { w } ) \delta - \frac { 1 } { \eta } \| \delta \| ^ { 2 } . } \end{array}\tag{70}
$$

Notice that the quadratic terms of the truncated edge coupling depend on $\delta$ only through $\delta \otimes \delta \colon$

$$
\begin{array} { r } { \frac 1 2 \delta ^ { \top } H ( \bar { w } ) \delta - \frac 1 \eta \| \delta \| ^ { 2 } = \frac 1 2 \langle H ( \bar { w } ) - \frac 2 \eta I _ { d } , \delta \otimes \delta \rangle . } \end{array}\tag{71}
$$

We will now replace the positive semidefinite rank-one matrix $\delta \otimes \delta$ with a general positive semidefinite matrix $\Sigma ,$ which we call the covariance matrix. The truncated edge coupling can then be written as a function of $\Sigma \colon$

$$
\begin{array} { r } { \widetilde { \Psi } ( \bar { w } , \Sigma ) = L ( \bar { w } ) + \frac { 1 } { 2 } \langle H ( \bar { w } ) - \frac { 2 } { \eta } I _ { d } , \Sigma \rangle . } \end{array}\tag{72}
$$

Why may we consider general positive semidefinite matrices $\Sigma$ rather than only positive semidefinite rank-one matrices? Recall that every positive semidefinite matrix can be written as a convex combination of positive semidefinite rank-one matrices. Since $\widetilde { \Psi }$ is linear in $\Sigma ,$ its supremum over all positive semidefinite matrices equals its supremum over positive semidefinite rank-one matrices, so the maximum value is unchanged.

There is also a physical reason to allow arbitrary positive semidefinite Σ. As we will see in Section 8.2, Σ arises as the time average $\langle \delta \otimes \delta \rangle$ over intermediate time scales—and a time average of rank-one positive semidefinite matrices can be arbitrary rank.

## 6.2 Covariance Matrix as Lagrange Multiplier

Recall that, when the endpoint Hessians agree, stable period-two orbits correspond to min–max critical points of the centered edge coupling in w¯ and δ. When we replace $\delta \otimes \delta$ by $\Sigma ,$ we retain the maximization criterion. However, unlike with rod flow, central flow does not determine the dynamics of Σ by an ODE. Instead, Σ is determined at each instant by a maximization problem parametrized by w¯.

Notice that the truncated edge coupling is linear in Σ. Let $\Lambda = \textstyle { \frac { 2 } { \eta } } I _ { d } - H$ denote the slack. Since $L ( \bar { w } )$ does not depend on Σ, the maximization over Σ is governed entirely by the linear term

$$
- { \textstyle \frac { 1 } { 2 } } \langle \Lambda , \Sigma \rangle .
$$

The argmax is determined by the sharpness $\lambda _ { 1 } ( H )$ . When the sharpness is less than $2 / \eta$ , Λ is positive definite and the argmax is the zero matrix. When the sharpness is greater than $2 / \eta$ , no argmax exists: the maximization problem is unbounded as any component of Σ along the top eigenspace of H can be increased without bound. When the sharpness is exactly $2 / \eta$ , the solution is not unique: any Σ supported on the kernel of Λ is a valid maximizer, since it contributes nothing to $\langle \Lambda , \Sigma \rangle$ . We then have that:

$$
\operatorname { a r g m a x } _ { \Sigma \subseteq 0 } \widetilde { \Psi } ( \bar { w } , \Sigma ) = \left\{ \begin{array} { l l } { \{ 0 \} , } & { \lambda _ { 1 } ( H ) < \frac { 2 } { \eta } , } \\ { \mathcal { K } ( \bar { w } ) , } & { \lambda _ { 1 } ( H ) = \frac { 2 } { \eta } , } \\ { \emptyset , } & { \lambda _ { 1 } ( H ) > \frac { 2 } { \eta } , } \end{array} \right.\tag{73}
$$

where

$$
\mathcal { K } ( \bar { w } ) : = \{ U X U ^ { \top } : X \succeq 0 \} , \qquad X \in \mathbb { R } ^ { k \times k } ,\tag{74}
$$

is the critical face. $U \in \mathbb { R } ^ { d \times k }$ is an orthonormal basis for the k-dimensional eigenspace of $H ( \bar { w } )$ at $2 / \eta$ . K represents the set of covariance matrices whose oscillations are confined to the critica subspace.

There are three conditions that Cohen et al. [2025] impose on the central flow:

$$
\Sigma \succeq 0 , \qquad \Lambda \succeq 0 , \qquad \langle \Lambda , \Sigma \rangle = 0 .\tag{75}
$$

The first condition requires the covariance matrix to be positive semidefinite. The second requires the slack to be positive semidefinite, which is equivalent to confining the flow to the feasible region Ω. The third condition requires the covariance matrix Σ to be orthogonal to the slack Λ. This encodes the fact that oscillations are confined to the critical subspace of H. We have that these three conditions are obtained by maximizing the truncated edge coupling over Σ.

Whereas rod flow encodes the oscillations in the state variable δ, the covariance matrix Σ of central flow acts as a Lagrange multiplier that constrains the dynamics to the feasible region. Thus, the dynamics of central flow correspond to a constrained minimization problem rather than a min–max problem.

## 6.3 Persistence

Central flow can be expressed as gradient flow on the truncated edge coupling:

$$
\frac { \mathrm { d } \bar { w } } { \mathrm { d } t } = - \eta \partial _ { \bar { w } } \widetilde { \Psi } ( \bar { w } , { \Sigma } )\tag{76}
$$

$$
\begin{array} { r } { = - \eta \nabla L ( \bar { w } ) - \frac { \eta } { 2 } \nabla _ { \bar { w } } \langle H ( \bar { w } ) , \Sigma \rangle , } \end{array}\tag{77}
$$

where Σ is a maximizer of $\tilde { \Psi } ( \bar { w } , \cdot )$ . The resultant system defines a diferential complementarity problem: an ordinary diferential equation coupled to complementarity conditions.

When the sharpness is below $2 / \eta$ , the maximizer is $\Sigma = 0$ —central flow reduces to gradient flow. However, when the sharpness equals $2 / \eta$ , the maximizer is no longer unique. How do we select Σ?

We select Σ by invoking persistence: feasibility and complementarity must hold not only at a single instant but along the entire trajectory. It’s these dynamic constraints that determine Σ.

Define the rate of change of the slack:

$$
\begin{array} { r } { \dot { \Lambda } = \frac { \mathrm { d } } { \mathrm { d } t } \Lambda ( \bar { w } ( t ) ) } \end{array}\tag{78}
$$

$$
= - \nabla _ { \bar { w } } H ( \bar { w } ) [ \dot { \bar { w } } ] .\tag{79}
$$

For the trajectory to remain feasible, we require that the slack not decrease along the sharp directions (where the slack is zero). These directions are given by the columns of U—which yields the condition:

$$
U ^ { \top } \dot { \Lambda } U \succeq 0 .\tag{80}
$$

This condition allows a critical eigenvalue to move below $2 / \eta$ , but prevents it from moving above $2 / \eta$

Complementarity requires that oscillations occur only in directions where the slack is zero. Persistence of complementarity then dictates that if a direction develops a positive slack, it can no longer carry oscillations. Any direction that carries oscillations must therefore not only have zero

slack, but also stay at zero slack to first order— $- \langle \dot { \Lambda } , \Sigma \rangle = 0$ . Rearranging, we obtain the following condition:

$$
\langle U ^ { \top } \dot { \Lambda } U , X \rangle = 0 .\tag{81}
$$

Altogether, we have that the following conditions define the central flow for gradient descent:

$$
\begin{array} { r } { \displaystyle \left| \frac { \mathrm { d } \bar { w } } { \mathrm { d } t } = - \eta \partial _ { \bar { w } } \widetilde { \Psi } ( \bar { w } , \Sigma ) , \qquad \Sigma = U X U ^ { \top } , \right. } \\ { \displaystyle \left. X \succeq 0 , \qquad U ^ { \top } \dot { \Lambda } U \succeq 0 , \qquad \langle U ^ { \top } \dot { \Lambda } U , X \rangle = 0 . \right| } \end{array}\tag{82}
$$

One can show that the SDCP is equivalent to a convex quadratic program for X. Further details are provided in Appendix F.1.

## 6.4 RMSProp

We can also derive a central flow for RMSProp. The truncated edge coupling is given as:

$$
\begin{array} { r } { \widetilde { \Psi } _ { \eta , \bar { \nu } } ( \bar { w } , \Sigma ) = L ( \bar { w } ) + \frac { 1 } { 2 } \langle H ( \bar { w } ) - \frac { 2 } { \eta } P ( \bar { \nu } ) , \Sigma \rangle . } \end{array}\tag{83}
$$

Define the slack:

$$
\begin{array} { r } { \Lambda _ { \mathrm { R M S P r o p } } : = \frac { 2 } { \eta } P ( \bar { \nu } ) - H ( \bar { w } ) . } \end{array}\tag{84}
$$

Maximization over $\Sigma \succeq 0$ gives the static conditions:

$$
\Sigma \succeq 0 , \qquad \Lambda _ { \mathrm { R M S P r o p } } \succeq 0 , \qquad \langle \Lambda _ { \mathrm { R M S P r o p } } , \Sigma \rangle = 0 .\tag{85}
$$

Let U be an orthonormal basis for the kernel of Λ<sub>RMSProp</sub>. Then $\Sigma = U X U ^ { \top }$ for some $X \succeq 0$

The midpoint and second-moment evolve according to:

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } \bar { w } } { \mathrm { d } t } = - \eta P ^ { - 1 } \Big [ \nabla L + \frac { 1 } { 2 } \nabla _ { \bar { w } } \langle H , \Sigma \rangle \Big ] , } \\ { \displaystyle \frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \Big [ ( \nabla L ) ^ { \odot 2 } + \mathrm { d i a g } ( H \Sigma H ) - \bar { \nu } \Big ] , } \end{array}\tag{86}
$$

where we make the approximation

$$
\begin{array} { r } { \bar { g } ^ { \odot 2 } + \mathbb { [ } g \mathbb { ] } ^ { \odot 2 } \approx ( \nabla L ) ^ { \odot 2 } + \mathrm { d i a g } ( H \Sigma H ) , } \end{array}\tag{87}
$$

which follows from the local quadratic approximation of the loss about w¯ and by replacing $\delta \otimes \delta$ with Σ. For the time derivative of the slack, we have:

$$
\begin{array} { r } { \dot { \Lambda } _ { \mathrm { R M S P r o p } } = \frac { 2 } { \eta } \nabla _ { \bar { \nu } } P ( \bar { \nu } ) [ \dot { \bar { \nu } } ] - \nabla _ { \bar { w } } H ( \bar { w } ) [ \dot { \bar { w } } ] } \end{array}\tag{88}
$$

At fixed ( ¯w, ν¯), both terms are afine in Σ. Persistence of feasibility and complementarity gives:

$$
\begin{array} { r } { U ^ { \top } \dot { \Lambda } _ { \mathrm { R M S P r o p } } U \succeq 0 , \qquad \langle U ^ { \top } \dot { \Lambda } _ { \mathrm { R M S P r o p } } U , X \rangle = 0 . } \end{array}\tag{89}
$$

These conditions define a semidefinite complementarity problem that determines the covariance Σ. Altogether, we have that the following conditions define the central flow for RMSProp:

$$
\begin{array} { r l } & { \displaystyle \frac { \mathrm { d } \bar { w } } { \mathrm { d } t } = - \eta P ( \bar { \nu } ) ^ { - 1 } \partial _ { \bar { w } } \widetilde { \Psi } _ { \eta , \bar { \nu } } ( \bar { w } , \Sigma ) , \qquad \Sigma = U X U ^ { \top } , } \\ & { \displaystyle \frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \Big [ ( \nabla L ) ^ { \odot 2 } + \mathrm { d i a g } ( H \Sigma H ) - \bar { \nu } \Big ] , } \\ & { \qquad X \succeq 0 , \qquad U ^ { \top } \dot { \Lambda } _ { \mathrm { R M S P r o p } } U \succeq 0 , \qquad \langle U ^ { \top } \dot { \Lambda } _ { \mathrm { R M S P r o p } } U , X \rangle = 0 . } \end{array}\tag{90}
$$

This is precisely the RMSProp central flow of Cohen et al. [2025].

## 7 Central Flow for Momentum Methods

Using the phase-space edge coupling, we derive novel central flows for heavy ball and Adam.

## 7.1 Eliminating the Momentum Oscillation

We would like to derive a heavy-ball central flow. To do so, it helps to first eliminate the momentum oscillations $\gamma ,$ so that the edge coupling is expressed only in terms of $( \bar { z } , \delta )$

Recall the centered phase-space edge coupling $\Upsilon _ { \eta , \beta }$ of heavy ball. The momentum half-diference $\gamma$ enters it through only two terms:

$$
{ \frac { 2 \beta } { 1 - \beta } } \gamma \cdot \delta + { \frac { \eta \beta } { 2 ( 1 - \beta ) } } \| \gamma \| ^ { 2 } = { \frac { \eta \beta } { 2 ( 1 - \beta ) } } \left\| \gamma + { \frac { 2 } { \eta } } \delta \right\| ^ { 2 } - { \frac { 2 \beta } { \eta ( 1 - \beta ) } } \| \delta \| ^ { 2 } .\tag{91}
$$

As period-two orbits are critical points of $\Upsilon _ { \eta , \beta }$ , they are stationary in $\gamma .$ . Since $\Upsilon _ { \eta , \beta }$ is strictly convex in $\gamma _ { : }$ , stationarity has the unique solution

$$
\gamma = - \frac { 2 } { \eta } \delta ,\tag{92}
$$

which minimizes $\Upsilon _ { \eta , \beta }$ over $\gamma .$ . For period-two orbits, the momentum oscillation is not an independent degree of freedom: it is fixed by the position oscillation.

Define the reduced edge coupling:

$$
\begin{array} { r l } & { \Upsilon _ { \eta , \beta } ^ { \mathrm { r e d } } ( \bar { z } , \delta ) : = \underset { \gamma } { \mathrm { m i n } } ~ \Upsilon _ { \eta , \beta } \big ( \bar { z } , ( \delta , \gamma ) \big ) } \\ & { \qquad = \frac { 1 } { 2 } \big [ L ( \bar { w } + \delta ) + L ( \bar { w } - \delta ) \big ] - \frac { 1 + \beta } { \eta ( 1 - \beta ) } \| \delta \| ^ { 2 } - \frac { \eta \beta } { 2 ( 1 - \beta ) } \| \bar { m } \| ^ { 2 } . } \end{array}\tag{93}
$$

Note that the critical points of the reduced edge coupling exactly correspond to those of the centered phase-space edge coupling.

## 7.2 The Truncated Reduced Edge Coupling

As with gradient descent, we approximate the loss about the midpoint w¯ by its local quadratic approximation:

$$
\begin{array} { r } { \frac { 1 } { 2 } [ L ( \bar { w } + \delta ) + L ( \bar { w } - \delta ) ] \approx L ( \bar { w } ) + \frac { 1 } { 2 } \delta ^ { \top } H ( \bar { w } ) \delta . } \end{array}\tag{94}
$$

Substituting into the reduced edge coupling and again replacing $\delta \otimes \delta$ with a general positive semidefinite matrix $\Sigma \succeq 0$ , we obtain the truncated reduced edge coupling:

$$
\begin{array} { r } { \widetilde { \Upsilon } _ { \eta , \beta } ^ { \mathrm { r e d } } ( \bar { z } , \Sigma ) = L ( \bar { w } ) - \frac { \eta \beta } { 2 ( 1 - \beta ) } \| \bar { m } \| ^ { 2 } - \frac { 1 } { 2 } \langle \Lambda _ { \mathrm { H B } } ( \bar { w } ) , \Sigma \rangle , } \end{array}\tag{95}
$$

where

$$
\begin{array} { r } { \Lambda _ { \mathrm { H B } } ( \bar { w } ) : = \frac { 2 ( 1 + \beta ) } { \eta ( 1 - \beta ) } I _ { d } - H ( \bar { w } ) . } \end{array}\tag{96}
$$

The truncated edge coupling of heavy ball is quite similar to that of gradient descent—though there are two diferences. First, the slack difers between heavy ball and gradient descent, reflecting the difering sharpness thresholds of the two methods. Second, the heavy-ball reduced edge coupling has an extra term proportional to the squared midpoint momentum.

Maximizing over $\Sigma \succeq 0$ , we obtain the following conditions:

$$
\Sigma \succeq 0 , \qquad \Lambda _ { \mathrm { H B } } \succeq 0 , \qquad \langle \Lambda _ { \mathrm { H B } } , \Sigma \rangle = 0 .\tag{97}
$$

As was the case with gradient descent, the covariance matrix Σ acts as a Lagrange multiplier—now constraining the dynamics to the feasible region of heavy ball.

Following a similar approach, one can also derive a central flow for Nesterov momentum. Details are provided in Appendix C.

## 7.3 Persistence

Taking derivatives of the truncated reduced edge coupling with respect to the midpoint variables gives:

$$
\begin{array} { r } { \partial _ { \bar { w } } \widetilde { \Upsilon } _ { \eta , \beta } ^ { \mathrm { r e d } } = \nabla L ( \bar { w } ) + \frac { 1 } { 2 } \nabla _ { \bar { w } } \langle H ( \bar { w } ) , \Sigma \rangle , } \end{array}\tag{98}
$$

$$
\begin{array} { r } { \partial _ { \bar { m } } \tilde { \Upsilon } _ { \eta , \beta } ^ { \mathrm { r e d } } = - \frac { \eta \beta } { 1 - \beta } \bar { m } . } \end{array}\tag{99}
$$

As with the heavy ball rod flow, we apply the structure matrix K to the gradient of the truncated reduced edge coupling to obtain the central flow ODE for heavy ball.

$$
\frac { \mathrm { d } \bar { z } } { \mathrm { d } t } = K \partial _ { \bar { z } } \widetilde { \Upsilon } _ { \eta , \beta } ^ { \mathrm { r e d } } ( \bar { z } , \Sigma ) .\tag{100}
$$

Writing the ODEs for w¯ and m¯ individually:

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } \bar { w } } { \mathrm { d } t } = - \eta \left[ \beta \bar { m } + ( 1 - \beta ) \Big ( \nabla L ( \bar { w } ) + \frac { 1 } { 2 } \nabla _ { \bar { w } } \langle H ( \bar { w } ) , \Sigma \rangle \Big ) \right] , } \\ { \displaystyle \frac { \mathrm { d } \bar { m } } { \mathrm { d } t } = ( 1 - \beta ) \left[ \nabla L ( \bar { w } ) + \frac { 1 } { 2 } \nabla _ { \bar { w } } \langle H ( \bar { w } ) , \Sigma \rangle - \bar { m } \right] . } \end{array}\tag{101}
$$

Note that the central flow ODEs for the midpoint variables are quite similar to the ODEs of continuous-time heavy-ball flow—with the only diference being that every gradient is now accompanied by an additional sharpness-reducing force.

As with the central flow for gradient descent, to determine Σ we invoke persistence: feasibility and complementarity must hold along the trajectory.

Let U be an orthonormal basis for the kernel of $\Lambda _ { \mathrm { H B } }$ . Static complementarity requires:

$$
\Sigma = U X U ^ { \top } , \qquad X \succeq 0 .\tag{102}
$$

Computing the rate of change of the slack:

$$
\dot { \Lambda } _ { \mathrm { H B } } = - \nabla _ { \bar { w } } H ( \bar { w } ) [ \dot { \bar { w } } ]\tag{103}
$$

Persistence of feasibility requires that the slack not decrease along sharp directions:

$$
U ^ { \top } \dot { \Lambda } _ { \mathrm { H B } } U \succeq 0 .\tag{104}
$$

Persistence of complementarity requires that there be no oscillations in directions whose slack is increasing:

$$
\langle U ^ { \top } \dot { \Lambda } _ { \mathrm { H B } } U , X \rangle = 0 .\tag{105}
$$

As with gradient descent, the persistence conditions define a semidefinite complementarity problem for X. Altogether, we have that the following conditions define the central flow for heavy ball:

$$
\begin{array} { r l } & { \left| \frac { \mathrm { d } \bar { z } } { \mathrm { d } t } = K \partial _ { \bar { z } } \widetilde { \Upsilon } _ { \eta , \beta } ^ { \mathrm { r e d } } ( \bar { z } , \Sigma ) , \qquad \Sigma = U X U ^ { \top } , \right. } \\ & { \left. X \succeq 0 , \qquad U ^ { \top } \dot { \Lambda } _ { \mathrm { H B } } U \succeq 0 , \qquad \langle U ^ { \top } \dot { \Lambda } _ { \mathrm { H B } } U , X \rangle = 0 . \right| } \end{array}\tag{106}
$$

Figure 5 illustrates the central flow for heavy ball. As with gradient descent, it can be interpreted as a projected flow: at the boundary of the feasible region, the sharpness-reduction force cancels the outward component of the motion. What is projected, however, is the heavy-ball step rather than the negative gradient.

![](images/598259cb45f7a7115153bc6358396d24b50f01afa10f361217ceabb3fcca9ba1.jpg)  
Figure 5: Heavy-Ball Central Flow. The gray region is the feasible region for heavy ball. While the stable flow leaves the feasible region, the central flow turns a corner at the boundary and follows it to a constrained minimum. Inset: The heavy-ball update points out of the feasible region. The sharpness-reduction force cancels its outward component, and what remains is the velocity of the central flow.

## 7.4 Adam

We can also obtain a central flow for Adam. With the second moment ν¯ held fixed, the centered phase-space edge coupling of Adam depends on γ as:

$$
\frac { 2 \beta _ { 1 } } { 1 - \beta _ { 1 } } \gamma \cdot \delta + \frac { \eta \beta _ { 1 } } { 2 ( 1 - \beta _ { 1 } ) } \gamma ^ { \top } P ^ { - 1 } \gamma = \frac { \eta \beta _ { 1 } } { 2 ( 1 - \beta _ { 1 } ) } \left( \gamma + \frac 2 \eta P \delta \right) ^ { \top } P ^ { - 1 } \left( \gamma + \frac 2 \eta P \delta \right) - \frac { 2 \beta _ { 1 } } { \eta ( 1 - \beta _ { 1 } ) } \delta ^ { \top } P \delta .\tag{107}
$$

Stationarity in γ gives $\begin{array} { r } { \gamma = - \frac { 2 } { \eta } P \delta } \end{array}$ . Eliminating γ from the centered phase-space edge coupling yields the reduced edge coupling:

$$
\begin{array} { r } { \Upsilon _ { \eta , \beta _ { 1 } , \bar { \nu } } ^ { \mathrm { r e d } } ( \bar { z } , \delta ) = \frac { 1 } { 2 } \big [ L ( \bar { w } + \delta ) + L ( \bar { w } - \delta ) \big ] - \frac { 1 + \beta _ { 1 } } { \eta ( 1 - \beta _ { 1 } ) } \delta ^ { \top } P \delta - \frac { \eta \beta _ { 1 } } { 2 ( 1 - \beta _ { 1 } ) } \bar { m } ^ { \top } P ^ { - 1 } \bar { m } . } \end{array}\tag{108}
$$

This closely parallels the reduced edge coupling for heavy ball, except that the preconditioner P now acts as the metric tensor. Replacing $\delta \otimes \delta$ with $\Sigma \succeq 0$ , the truncated edge coupling becomes:

$$
\begin{array} { r } { \widetilde \Upsilon _ { \eta , \beta _ { 1 } , \bar { \nu } } ^ { \mathrm { r e d } } ( \bar { z } , \Sigma ) = L ( \bar { w } ) - \frac { \eta \beta _ { 1 } } { 2 ( 1 - \beta _ { 1 } ) } \bar { m } ^ { \top } P ^ { - 1 } \bar { m } - \frac { 1 } { 2 } \langle \Lambda _ { \mathrm { A d a m } } , \Sigma \rangle , } \end{array}\tag{109}
$$

where the Adam slack is given as:

$$
\begin{array} { r } { \Lambda _ { \mathrm { A d a m } } ( \bar { w } , \bar { \nu } ) : = \frac { 2 ( 1 + \beta _ { 1 } ) } { \eta ( 1 - \beta _ { 1 } ) } P ( \bar { \nu } ) - H ( \bar { w } ) . } \end{array}\tag{110}
$$

Maximizing over $\Sigma \succeq 0$ gives the following static conditions:

$$
\Sigma \succeq 0 , \qquad \Lambda _ { \mathrm { A d a m } } \succeq 0 , \qquad \langle \Lambda _ { \mathrm { A d a m } } , \Sigma \rangle = 0 .\tag{111}
$$

Let U be an orthonormal basis for ker $\Lambda _ { \mathrm { A d a m } }$ . As before, we write:

$$
\Sigma = U X U ^ { \top } , \qquad X \succeq 0 .\tag{112}
$$

We can then obtain the central flow for Adam:

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } \bar { w } } { \mathrm { d } t } = - \eta P ^ { - 1 } \left[ \beta _ { 1 } \bar { m } + ( 1 - \beta _ { 1 } ) \Big ( \nabla L ( \bar { w } ) + \frac { 1 } { 2 } \nabla _ { \bar { w } } \langle H ( \bar { w } ) , \Sigma \rangle \Big ) \right] , } \\ { \displaystyle \frac { \mathrm { d } \bar { m } } { \mathrm { d } t } = ( 1 - \beta _ { 1 } ) \left[ \nabla L ( \bar { w } ) + \frac { 1 } { 2 } \nabla _ { \bar { w } } \langle H ( \bar { w } ) , \Sigma \rangle - \bar { m } \right] , } \\ { \displaystyle \frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \left[ ( \nabla L ( \bar { w } ) ) ^ { \odot 2 } + \mathrm { d i a g } \left( H ( \bar { w } ) \Sigma H ( \bar { w } ) \right) - \bar { \nu } \right] . } \end{array}\tag{113}
$$

As with RMSProp, the second-moment equation uses the approximation

$$
\bar { g } ^ { \odot 2 } + \mathbb { [ } g \mathbb { I } ^ { \odot 2 } \approx ( \nabla L ( \bar { w } ) ) ^ { \odot 2 } + \mathrm { d i a g } ( H \Sigma H ) ,\tag{114}
$$

which follows from the local quadratic approximation of the loss about w¯ and by replacing $\delta \otimes \delta$ with Σ.

We determine the covariance by imposing persistence of feasibility and complementarity. The time derivative of the slack is:

$$
\begin{array} { r } { \dot { \Lambda } _ { \mathrm { A d a m } } = \frac { 2 ( 1 + \beta _ { 1 } ) } { \eta ( 1 - \beta _ { 1 } ) } \nabla _ { \bar { \nu } } P ( \bar { \nu } ) [ \dot { \bar { \nu } } ] - \nabla _ { \bar { w } } H ( \bar { w } ) [ \dot { \bar { w } } ] . } \end{array}\tag{115}
$$

At fixed $( \bar { w } , \bar { \nu } )$ , both the $\dot { \bar { w } }$ and $\dot { \bar { \nu } }$ terms are afine in Σ. Persistence then requires:

$$
U ^ { \top } \dot { \Lambda } _ { \mathrm { A d a m } } U \succeq 0 , \qquad \left. U ^ { \top } \dot { \Lambda } _ { \mathrm { A d a m } } U , X \right. = 0 .\tag{116}
$$

These persistence conditions define a semidefinite complementarity problem for X. Altogether, we have that the following conditions define the central flow for Adam:

$$
\boxed { \frac { \mathrm { d } \bar { z } } { \mathrm { d } t } = K _ { \bar { \nu } } \partial _ { \bar { z } } \widetilde { \Upsilon } _ { \eta , \beta _ { 1 } , \bar { \nu } } ^ { \mathrm { r e d } } ( \bar { z } , \Sigma ) , \qquad \Sigma = U X U ^ { \top } , }
$$

$$
\frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \Big [ ( \nabla L ( \bar { w } ) ) ^ { \odot 2 } + \mathrm { d i a g } \big [ H ( \bar { w } ) \Sigma H ( \bar { w } ) \big ] - \bar { \nu } \Big ] ,
$$

$$
X \succeq 0 , \qquad U ^ { \top } \dot { \Lambda } _ { \mathrm { A d a m } } U \succeq 0 , \qquad \langle U ^ { \top } \dot { \Lambda } _ { \mathrm { A d a m } } U , X \rangle = 0 . \left| \right.\tag{117}
$$

One can also derive a central flow for AdamW. Details are provided in Appendix D.

## 8 Rod Flow and Central Flow

We show that the key diference between rod flow and central flow is the time scale at which they coarse-grain. Rod flow coarse-grains over only the fast time scale, whereas central flow coarse-grains over both the fast and the intermediate time scales.

## 8.1 Fast, Intermediate, and Slow Time Scales

Berthier [2026] derived central flow by the method of multiple scales, identifying three time scales. There’s the fast time scale, given by the iteration counter itself. On this time scale, the sharp components of the iterates alternate in sign. There’s the intermediate time scale of self-stabilization. When the oscillation amplitude departs from its steady-state value, this time scale governs how long it takes for the amplitude to re-equilibrate. Finally, there’s the slow time scale. It’s the time scale that governs the drift of the center of the oscillating iterates.

Under this framework, rod flow is an efective theory for gradient descent that coarse-grains over the fast time scale only. Recall that the half-displacement of the iterates decomposes as

$$
\begin{array} { r } { \widehat { \delta } _ { t } \approx \frac { \eta } { 2 } \big ( \bar { g } + ( - 1 ) ^ { t } [ [ g ] ] \big ) . } \end{array}\tag{32}
$$

The alternating component flips sign at every step of the iteration counter. By tracking the unsigned envelope of the oscillation, rod flow coarse-grains over this sign flip.

What are the dynamics of the envelope? It helps to consider the rod flow ODEs when expanded to leading order in δ:

$$
\frac { \mathrm { d } \delta } { \mathrm { d } t } = \bigl [ \eta H ( \bar { w } ) - 2 I _ { d } \bigr ] \delta , \qquad \frac { \mathrm { d } \bar { w } } { \mathrm { d } t } = - \eta \Bigl [ \nabla L ( \bar { w } ) + \frac { 1 } { 2 } \nabla _ { \bar { w } } \langle H ( \bar { w } ) , \delta \otimes \delta \rangle \Bigr ] .\tag{118}
$$

The ODE for $\delta$ says that oscillations grow along components sharper than the $2 / \eta$ threshold and decay along components that are less sharp than the threshold. The ODE for w¯ is reminiscent of the one for central flow: it is a gradient flow on the loss, plus a sharpness-reduction force whose strength is proportional to the squared magnitude of the oscillations.

Together, the two equations form a feedback loop. Excess sharpness causes the amplitude to grow along sharp directions. The larger oscillations strengthen the sharpness-reduction force, which pushes the sharpness below the threshold. The oscillations then decay, weakening the sharpness-reduction force. The sharpness then grows until it is back above the threshold. The result is self-stabilization: the iterates hover at the $2 / \eta$ sharpness threshold without diverging.

We can express the self-stabilization loop as a Lotka–Volterra-type system. Consider the case of a single sharp direction. Let $\delta = s \hat { u }$ where uˆ is the unit vector in the sharp direction. And let $x = \eta \lambda _ { 1 } - 2$ denote the excess sharpness. Then we can obtain the following ODEs for the amplitude s and the excess sharpness x:

$$
{ \frac { \mathrm { d } s } { \mathrm { d } t } } = x s , \qquad { \frac { \mathrm { d } x } { \mathrm { d } t } } = \eta \Bigl ( r - { \textstyle { \frac { \eta } { 2 } } } c s ^ { 2 } \Bigr ) ,\tag{119}
$$

where $r = - \eta \langle \nabla \lambda _ { 1 } , \nabla L \rangle$ and $c = \| \nabla \lambda _ { 1 } \| ^ { 2 }$ . Here $r$ is the progressive sharpening constant: it measures how much descending the loss tends to increase the sharpness. We have that $r > 0$ corresponds to progressive sharpening. Likewise, c is the sharpness reduction coeficient: it measures the strength of the sharpness-reduction force per unit of squared amplitude.

Where does central flow fit in? The solutions to the Lotka–Volterra-type system above trace out closed orbits. Averaging the excess sharpness and the squared amplitude over one orbit gives:

$$
\langle x \rangle = 0 , \qquad \langle s ^ { 2 } \rangle = \frac { 2 r } { \eta c } .\tag{120}
$$

Over one orbit, the average excess sharpness is zero. And the average squared amplitude is proportional to the progressive sharpening constant and inversely proportional to the sharpness reduction coeficient. We have that oscillations are larger when progressive sharpening is strong or when the third derivative of the loss is small.

Central flow corresponds to the approximation in which we replace the instantaneous value of the amplitude-squared with its average over one orbit. Instead of coarse-graining over just the fast time scale of alternation as with rod flow, central flow also coarse-grains over the intermediate time scale of self-stabilization.

One subtlety is that the orbits are closed only when r and c evolve on the slow time scale and can therefore be treated as constant. For r, this assumption holds only while the oscillation is small. Once the oscillation becomes large, the orbits are no longer closed: the period lengthens and the orbits decay. Importantly, however, the orbit averages remain valid even in this regime—which is what justifies central flow’s approximation.

gradient descent, MLP, MSE: two critical directions

![](images/8840e0ae0040a75ec8175d47d738374d70dc2af5f3ac13ab1868eee50933f9a8.jpg)

![](images/f1c14a8a02c18102ee029ba6300f3f577e45b18899d938c409c1d359f7bd1820.jpg)  
Figure 6: Central Flow as Time-Averaged Rod Flow. Top: The extent matrix of rod flow has rank one at any given instant. In the presence of multiple sharp eigenvalues, rod flow self-stabilizes by turning in the critical subspace. The covariance matrix of central flow can be interpreted as the time average of the extent matrix. It has rank $k ,$ where $k$ is the number of critical eigenvalues. Bottom: Gradient descent on an MLP $( \eta = 0 . 0 5 )$ . There are two sharp eigenvalues. The extent matrix of rod flow and the discrete iterates are averaged over windows of $T$ iterations. Left: As the width of the averaging window increases, their participation ratio rises from one to that of $\Sigma _ { \mathrm { c e n t r a l } } .$ Right: Their normalized Bures–Wasserstein distance to $\Sigma _ { \mathrm { c e n t r a l } }$ falls toward zero.

## 8.2 Central Flow as Time-Averaged Rod Flow

That the two flows coarse-grain over diferent time scales also helps explain another key diference between them—the rank of the object each uses to encode the oscillations.

Rod flow is rank one: it tracks a single vector δ. This modeling assumption is natural, as a single step of gradient descent displaces the iterate along a single vector. But it raises a question: how does rod flow continue to track the iterates when there are multiple sharp directions?

Central flow is rank $k ,$ where k is the number of critical eigenvalues. This makes sense too. Each oscillation direction self-stabilizes a diferent sharp direction, and all k directions are needed for the SDCP of central flow to have a solution. But it raises a question of its own: what observable of the discrete iterates does a rank-k covariance correspond to?

Let $\Sigma _ { \mathrm { r o d } } = \delta \otimes \delta$ denote the extent matrix of rod flow, and let $\Sigma _ { \mathrm { c e n t r a l } }$ denote the covariance matrix of central flow. We claim that the extent matrix of rod flow, averaged over the intermediate time scale, equals the covariance matrix of central flow:

$$
\Sigma _ { \mathrm { c e n t r a l } } ( t ) = \frac { 1 } { T } \int _ { t - T / 2 } ^ { t + T / 2 } \Sigma _ { \mathrm { r o d } } ( t ^ { \prime } ) \mathrm { d } t ^ { \prime } ,\tag{121}
$$

where the averaging window T spans one full swing of the rod through the critical subspace. The window is long on the intermediate time scale but short on the slow one, so w¯ barely moves across it. Empirically, we find that T corresponds to roughly two orbits of the self-stabilization loop.

Rod flow maintains self-stabilization by turning. Although the extent is rank one at any given instant, its axis swings back and forth within the critical subspace. Central flow, whose covariance matrix encodes the time-averaged oscillations, carries all of the sharp directions at once. That the covariance matrix has rank k is necessary. As Berthier [2026] notes, when the self-stabilization loop is generalized to more than one sharp direction, it generically has no rank-one fixed point.

We tested this hypothesis experimentally in a setting with two sharp eigenvalues (Figure 6). We averaged the extent matrix of rod flow over windows of increasing length and compared each average with the covariance matrix of central flow. Rather than comparing the ranks of these matrices, we compared their participation ratios (PR) as the participation ratio is less sensitive to small eigenvalues.

$$
\operatorname { P R } ( { \Sigma } ) = \frac { ( \operatorname { t r } { \Sigma } ) ^ { 2 } } { \operatorname { t r } ( { \Sigma } ^ { 2 } ) } = \frac { \left( \sum _ { i } \sigma _ { i } \right) ^ { 2 } } { \sum _ { i } \sigma _ { i } ^ { 2 } } ,\tag{122}
$$

where $\sigma _ { i }$ are the eigenvalues of Σ.

As expected, the extent matrix of rod flow averaged over a window of one step has rank one. As the window grows, the participation ratio rises. It completes 90% of its rise at a window length of about 185 iterations and levels of by 250, reaching 1.67 compared with 1.70 for central flow. The participation ratio stays below two because the two critical directions are unequally excited.

While the participation ratio tells us that the oscillations are spread out across directions to a similar degree, it doesn’t tell us whether those directions coincide. To probe this, we also computed the Bures–Wasserstein distance between the time-averaged extent matrix and the covariance matrix of central flow. We normalized the distance by $\sqrt { \mathrm { t r } \Sigma _ { \mathrm { c e n t r a l } } }$ , so that it does not depend on the overall size of the oscillations:

$$
\hat { d } _ { \mathrm { B W } } ( T ) = \frac { d _ { \mathrm { B W } } \left( \langle \Sigma _ { \mathrm { r o d } } \rangle _ { T } , \Sigma _ { \mathrm { c e n t r a l } } \right) } { \sqrt { \mathrm { t r } \Sigma _ { \mathrm { c e n t r a l } } } }\tag{123}
$$

where $\langle \Sigma _ { \mathrm { r o d } } \rangle _ { T }$ denotes the time-averaged extent matrix over a window of length T iterations.

Roughly, a normalized distance near one indicates very dissimilar matrices, whereas a distance near zero indicates nearly identical ones. When comparing $\langle \Sigma _ { \mathrm { r o d } } \rangle _ { T }$ and $\Sigma _ { \mathrm { c e n t r a l } }$ across averaging windows of increasing width, the normalized distance falls from 0.69 for a single step to 0.06 at $T = 2 5 0$ . This shows that the two matrices agree not only on the magnitude of the oscillations but also in their direction.

## 8.3 Turning Rate of the Rod

What determines how fast the rod turns? Write $\delta = s \hat { v }$ with $\| \hat { v } \| = 1$ . The rod flow ODE for $\delta$ then splits into separate ODEs for the amplitude s and the axis vˆ:

$$
\frac { \mathrm { d } s } { \mathrm { d } t } = \eta \langle [ g ] , \hat { v } \rangle - 2 s , \qquad \frac { \mathrm { d } \hat { v } } { \mathrm { d } t } = \frac { \eta } { s } [ [ g ] ] _ { \perp } ,\tag{124}
$$

where $[ [ g ] ] _ { \perp } : = [ [ g ] ] - \langle [ [ g ] ] , \hat { v } \rangle \hat { v }$ is the component of the gradient half-diference orthogonal to the current axis of the rod. All of the turning comes from this transverse component. Taylor-expanding

![](images/4ea992809d9a83f9575250aad0c67b48190dc3987505651cd3739f8067907275.jpg)

![](images/ddb14b0e47ffdb5468ff6d39fd0abb245300026f968ee0b3712d48e23722bb93.jpg)  
predicted turning rate (rad / iteration)

![](images/96b94ab5a79519e7f8d887c6d5aeb4e3e18ea8be3b71f03661f7cb593861725c.jpg)  
Figure 7: Turning Rate of the Rod. Gradient descent on an MLP $( \eta = 0 . 0 5 )$ . Left: Time series of the turning rate for rod flow and for the discrete oscillation, compared with the prediction of the linear turning law. Middle: Scatter plot of measured versus predicted turning rates for the discrete oscillation and for rod flow, showing strong agreement. Right: Participation ratio of the time-averaged extent matrix as a function of the swept angle.

the gradient half-diference and retaining only the Hessian term of $[ [ g ] ]$ , we obtain:

$$
\frac { \mathrm { d } \hat { v } } { \mathrm { d } t } \approx \eta \big ( H \hat { v } - \langle H \hat { v } , \hat { v } \rangle \hat { v } \big ) = \eta \big ( H \hat { v } \big ) _ { \perp } .\tag{125}
$$

As the right-hand side is orthogonal to $\hat { v } ,$ the flow preserves $\| \hat { v } \| = 1$ . By the Pythagorean theorem, $\| ( H \hat { v } ) _ { \perp } \| ^ { 2 } = \| H \hat { v } \| ^ { 2 } - \langle H \hat { v } , \hat { v } \rangle ^ { 2 }$ . And since H is symmetric, $\| H \hat { v } \| ^ { 2 } = \langle H ^ { 2 } \hat { v } , \hat { v } \rangle$ . Hence

$$
\lVert \dot { \hat { v } } \rVert \approx \eta \sqrt { \langle H ^ { 2 } \hat { v } , \hat { v } \rangle - \langle H \hat { v } , \hat { v } \rangle ^ { 2 } } .\tag{126}
$$

To help interpret this expression, write

$$
\hat { v } = \sum _ { i = 1 } ^ { d } c _ { i } \hat { u } _ { i }\tag{127}
$$

where the $\hat { u } _ { i }$ are orthonormal eigenvectors of the Hessian with corresponding eigenvalues $\lambda _ { i }$ . Note that $\textstyle \sum _ { i } c _ { i } ^ { 2 } = 1$ . We then have that:

$$
\| \dot { \hat { v } } \| \approx \eta \sqrt { \sum _ { i } c _ { i } ^ { 2 } ( \lambda _ { i } - \bar { \lambda } ) ^ { 2 } } ,\tag{128}
$$

where $\begin{array} { r } { \bar { \lambda } = \sum _ { i } c _ { i } ^ { 2 } \lambda _ { i } } \end{array}$ is the Rayleigh quotient of H at vˆ. The rod therefore turns faster when $\hat { v }$ has substantial components along several eigenvectors with diferent eigenvalues.

To make this concrete, consider the case where we have two sharp eigenvalues $\lambda _ { 1 } > \lambda _ { 2 }$ with respective eigenvectors $\hat { u } _ { 1 }$ and $\hat { u } _ { 2 }$ . Write vˆ = cos $\theta \hat { u } _ { 1 } +$ sin $\theta \hat { u } _ { 2 }$ . Then

$$
\dot { \hat { v } } = \dot { \theta } \hat { e } _ { \theta }\tag{129}
$$

where $\hat { e } _ { \theta } = -$ sin $\theta \hat { u } _ { 1 } +$ cos $\theta \hat { u } _ { 2 }$ is the unit vector perpendicular to vˆ in the two-dimensional critical plane. Using $H \hat { v } = \lambda _ { 1 }$ cos $\theta \hat { u } _ { 1 } + \lambda _ { 2 }$ sin $\theta \hat { u } _ { 2 }$ and $\langle H \hat { v } , \hat { v } \rangle = \lambda _ { 1 } \cos ^ { 2 } \theta + \lambda _ { 2 } \sin ^ { 2 } \theta$ , a direct computation gives:

$$
H \hat { v } - \langle H \hat { v } , \hat { v } \rangle \hat { v } = ( \lambda _ { 2 } - \lambda _ { 1 } ) \sin \theta \cos \theta \hat { e } _ { \theta } .
$$

![](images/f4b90404dd2447ed58b033b0ede2a91c38795df27dbff796dc8ef520d52112ff.jpg)

![](images/3da774fb8f186ec5ad4507587c8a4bdfcb7b51f57c78a314c6d629dccf666e7f.jpg)

![](images/7855e53cb7143efa8ca03534212dadca4dcc98c09d4eab30f97d02144e299a82.jpg)  
Figure 8: Position and Momentum Extents of Heavy Ball. Heavy ball on an MLP $( \eta = 0 . 3$ $\beta = 0 . 9 )$ . There are two critical eigenvalues. Left: The amplitude ratio $\| \gamma \| / \| \delta \|$ over time. For period-two orbits, the amplitude ratio is $2 / \eta$ (dashed line). The ratio hovers around this value but shows non-trivial fluctuations. Middle: The alignment cos $( - \gamma , \delta )$ between the position and momentum extents. Right: The normalized Bures–Wasserstein distance between the time averages $\langle \frac { \eta ^ { 2 } } { 4 } \gamma \gamma ^ { \top } \rangle _ { T }$ and $\langle \delta \delta ^ { \top } \rangle _ { T }$ . As the averaging window T grows, the time-averaged position and momentum extent matrices increasingly agree with the relation they satisfy on a period-two cycle.

Matching the coeficients of $\scriptstyle { \hat { e } } _ { \theta }$ yields:

$$
\frac { \mathrm { d } \theta } { \mathrm { d } t } = - \frac { \eta } { 2 } \left( \lambda _ { 1 } - \lambda _ { 2 } \right) \sin { 2 \theta } .\tag{130}
$$

The turning rate is thus proportional to the gap between the two critical eigenvalues. In particular, since $\lambda _ { 1 } > \lambda _ { 2 }$ , the rod turns toward $\theta = 0 \mathrm { - } \mathrm { t h e }$ direction of the sharper eigenvector $\hat { u } _ { 1 }$

We tested the turning-rate law by comparing the measured turning rate with its predicted value and found strong agreement between the two (Figure 7). Furthermore, in the case of two sharp eigenvalues, we found that the participation ratio of $\langle \Sigma _ { \mathrm { r o d } } \rangle _ { T }$ converged to that of $\Sigma _ { \mathrm { c e n t r a l } }$ after roughly π radians of turning—in line with expectations.

## 8.4 Turning Rate for Momentum Methods

We will now consider the turning rate for heavy ball. Let $\delta = s \hat { v }$ . As before, the rate of change of the direction vˆ is given by the orthogonal component of the update for $\delta ,$ divided by the amplitude s:

$$
\frac { \mathrm { d } \hat { v } } { \mathrm { d } t } = \frac { \eta } { s } \big [ ( 1 - \beta ) [ [ g ] ] + \beta \gamma \big ] _ { \perp } .\tag{131}
$$

In contrast with gradient descent, there is now a term proportional to the momentum extent γ—which must be handled with care. For a period-two orbit, the position and momentum half-diferences are in fixed proportion. However, this is not generically true of the position and momentum extents over the course of the dynamics (Figure 8).

To see why, consider heavy ball on the one-dimensional quadratic $L ( w ) = { \textstyle { \frac { 1 } { 2 } } } \lambda w ^ { 2 }$ . The update is linear in phase space:

$$
\binom { w _ { t + 1 } } { m _ { t + 1 } } = A \binom { w _ { t } } { m _ { t } } , \qquad A = \binom { 1 - ( 1 - \beta ) \eta \lambda } { ( 1 - \beta ) \lambda } \frac { - \eta \beta } { \beta } \Bigg ) .
$$

It helps to examine the eigenvalues and eigenvectors of A. Since $\operatorname { t r } A = 1 + \beta - ( 1 - \beta ) \eta \lambda$ and det $A = \beta ,$ the characteristic equation is

$$
\alpha ^ { 2 } - \tau \alpha + \beta = 0 , \qquad \tau = 1 + \beta - ( 1 - \beta ) \eta \lambda .\tag{132}
$$

The eigenvalues and corresponding eigenvectors are

$$
\alpha _ { \pm } = \frac { \tau \pm \sqrt { \tau ^ { 2 } - 4 \beta } } { 2 } , \qquad v _ { \pm } = \left( { \alpha _ { \pm } - \beta \atop ( 1 - \beta ) \lambda } \right) .\tag{133}
$$

One can see that the eigenvalues are complex when $\tau ^ { 2 } < 4 \beta$ . This is satisfied when:

$$
\frac { 1 - \sqrt { \beta } } { 1 + \sqrt { \beta } } < \eta \lambda < \frac { 1 + \sqrt { \beta } } { 1 - \sqrt { \beta } } .
$$

At $\beta = 0 . 9$ , the window is $0 . 0 2 6 < \eta \lambda < 3 7 . 9 7 -$ against a threshold value of 38. Because sharp eigenvalues hover around the sharpness threshold rather than staying exactly at it, for large momentum coeficients they spend a significant fraction of the time in the complex window. This causes the sharp components to rotate in the $( w , m )$ plane. As a result, their amplitude ratio fluctuates instead of staying at exactly $2 / \eta$

We must therefore treat $\gamma$ as a separate degree of freedom. Define $\tilde { \gamma } = \gamma / s$ , the momentum extent divided by the position amplitude. Linearizing $[ [ g ] ] \approx s H \hat { v }$ as before, the $\gamma$ ODE gives

$$
\frac { \mathrm { d } \tilde { \gamma } } { \mathrm { d } t } = \frac { \dot { \gamma } } { s } - \frac { \dot { s } } { s } \tilde { \gamma } \approx - ( 1 + \beta ) \tilde { \gamma } - ( 1 - \beta ) H \hat { v } - \frac { \dot { s } } { s } \tilde { \gamma } .
$$

The amplitude enters through the ratio $\dot { s } / s$ , which we can express in terms of vˆ and $\tilde { \gamma }$ using the ODE for $s \mathrm { : }$

$$
\frac { \dot { s } } { s } \approx - 2 + \eta \hat { v } ^ { \top } \big [ ( 1 - \beta ) H \hat { v } + \beta \tilde { \gamma } \big ] .\tag{134}
$$

Substituting this expression, we obtain a closed system of ODEs for the pair $( \hat { v } , \tilde { \gamma } )$ :

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } \hat { v } } { \mathrm { d } t } \approx \eta \big [ ( 1 - \beta ) H \hat { v } + \beta \tilde { \gamma } \big ] _ { \perp } , } \\ { \displaystyle \frac { \mathrm { d } \tilde { \gamma } } { \mathrm { d } t } \approx ( 1 - \beta ) \big ( \tilde { \gamma } - H \hat { v } \big ) - \eta \hat { v } ^ { \top } \big [ ( 1 - \beta ) H \hat { v } + \beta \tilde { \gamma } \big ] \tilde { \gamma } . } \end{array}\tag{135}
$$

We tested this turning law for heavy ball and again found good agreement with the measured turning rate. More details are provided in Appendices H.3 and H.4.

## 8.5 Summary

There are four main modeling diferences between central flow and rod flow.

1. Central flow treats the covariance matrix Σ as a Lagrange multiplier, whereas in rod flow the extent $\delta$ evolves according to its own ODE.

2. Central flow replaces the loss with its local quadratic approximation so that $\dot { \bar { w } }$ is afine in $\Sigma$ whereas rod flow evaluates the exact loss at the endpoints of the rod.

3. The covariance matrix of central flow is rank k, where k is the number of critical eigenvalues, whereas the extent matrix of rod flow is rank one.

4. For momentum methods, central flow represents only the position oscillations, whereas rod flow carries the position and momentum extents as distinct degrees of freedom.

This time-scale analysis reveals that these modeling choices are not independent: all four diferences follow from the fact that central flow coarse-grains over the intermediate time scale.

Averaging over the self-stabilization loop removes the amplitude as a dynamical variable. What survives is its orbit average, which is the Lagrange multiplier that holds the sharpness at $2 / \eta$ Because the covariance matrix of central flow is a Lagrange multiplier rather than a state variable, it must be determined by solving an optimization problem. To make this optimization problem tractable, it helps to replace the loss with its local quadratic approximation—which makes the dynamics of w¯ afine in Σ and yields an SDCP. Once Σ is fixed by the SDCP, persistence dictates that its rank must equal the number of critical directions. And while the momentum and position oscillations may be independent at any given instant, their time averages converge to the amplitude ratio of a period-two orbit—rendering them interdependent over intermediate time scales.

## 9 Experiments

We evaluate the rod flows and central flows on three representative neural network architectures: an MLP [Rumelhart et al., 1986], a CNN [LeCun et al., 1998], and a ViT [Dosovitskiy et al., 2021]. We compare them with both the discrete iterates and the stable flow, the naïve continuous-time limit of the discrete optimizer.

## 9.1 Setup

All three models were trained full-batch on the same subset of 5 000 examples from CIFAR-10 [Krizhevsky, 2009]. The loss function was mean squared error (MSE) on one-hot targets. The MLP had two hidden layers of width 50 (156 710 parameters). The CNN had two $3 \times 3$ convolutional layers with 32 channels (30 634 parameters). The ViT was a SimpleViT [Beyer et al., 2022] with width 64, depth 3, 8 attention heads of width 8, and MLP blocks of width 256 (153 322 parameters). The step sizes were chosen using probe runs, so that the edge of stability was reached after a few thousand iterations.

We ran experiments on the following optimizers: gradient descent, RMSProp, heavy-ball momentum, Nesterov momentum, Adam, and AdamW. In this section, we will focus on the results for gradient descent and Adam. The experimental results for all the optimizers can be found in Appendix H.

For each optimizer, we ran stable flow, midpoint and endpoint rod flow (mp-rod and ep-rod in the figures), and central flow. Both rod flows were run with noise floors. For gradient descent, we also ran two other continuous-time models: edge flow [Marion, 2026] and the mean–fluctuation model [Chodron de Courcel, 2026]. We implemented the mean–fluctuation model from the first version of Chodron de Courcel [2026], whose equations are unchanged in the second.

Each flow was initialized from the discrete iterates 1 000 iterations before the transition to the edge of stability and run until 3 000 iterations after it (500 before and 1 500 after for the ViT). The flows were integrated with four forward Euler substeps per iteration.

## 9.2 Results

Gradient descent. Figure 9 shows the results for gradient descent on the three architectures. Every flow except the stable flow self-stabilizes at the $2 / \eta$ sharpness threshold. The sharpness of central flow sits exactly at the threshold, while the sharpnesses of the other flows fluctuate about the threshold. Importantly, rod flow now successfully tracks the discrete iterates through the transition to the edge of stability—fixing a shortcoming of the previous formulation.

![](images/3f1565d7744a7fbad9a08b28bf6748774767510ddac049dfc348cd9ccbafda6e.jpg)  
Figure 9: Experimental Results for Gradient Descent. Left: MLP $( \eta = 0 . 0 5 )$ . Center: CNN $( \eta = 0 . 0 5 )$ . Right: ViT (η = 0.1). The flows are handed of from the discrete iterates 1 000 iterations before the crossing (500 for the ViT). All flows except the stable flow self-stabilize at the sharpness threshold, and they track the discrete iterates orders of magnitude more closely than the stable flow does.

Every flow tracks the center of the iterates more than an order of magnitude more closely than the stable flow does. Rod flow tracks best on the MLP and the CNN, staying three to four times closer to the iterates than the next-best flow. On the ViT, the five flows are within a factor of two of one another, with central flow tracking slightly better than the other flows. For all three architectures, the midpoint and endpoint rod flows coincide.

Several critical eigenvalues. It is of interest to test the performance of rod flow and central flow in the presence of multiple sharp eigenvalues. To do so, we conducted an extended run on the MLP—until three eigenvalues had reached the threshold (Figure 10). Each new critical eigenvalue raises the tracking error: about tenfold when the second largest eigenvalue becomes critical, and about twofold when the third largest one does. Furthermore, once a second eigenvalue becomes critical, the distance between the flows and the discrete iterates starts to wobble. Still, even in the presence of multiple sharp eigenvalues, both rod flow and central flow successfully track the discrete iterates, staying more than an order of magnitude closer to them than the stable flow does.

![](images/68691e925ed6d34fd3d24cf788e4dd87bf1dfbc5fa962ecf3982cdbdbb14a4ca.jpg)  
Figure 10: Tracking Through Three Critical Eigenvalues. Gradient descent on an MLP $( \eta = 0 . 0 5 )$ . Top: The top six eigenvalues of the Hessian at the discrete midpoint. $\lambda _ { 1 } , \lambda _ { 2 }$ , and $\lambda _ { 3 }$ reach the threshold one after another. Bottom: The distance from each flow’s center to the discrete midpoint. Rod flow and central flow stay close to the iterates through all three transitions, while the stable flow drifts away.

Adam. The previous formulation of the rod flow for Adam [Regis and Chewi, 2026a] had several shortcomings. Like the original rod flow for gradient descent, it could not track the transition to the edge of stability and had to be initialized in the steady-state edge of stability regime. But even there, it failed to track the iterates for large values of the momentum coeficient—the largest value demonstrated was $\beta _ { 1 } = 0 . 8$ for the MLP.

In our experimental runs, all three architectures use $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , and bias correction— hyperparameter settings that are actually used in practice. And as was the case with gradient descent, the flows are initialized before the transition.

Figure 11 shows the results for Adam on the three architectures. Let $\hat { P } _ { t }$ denote the bias-corrected preconditioner:

$$
\hat { P } _ { t } = P \bigg ( \frac { \nu _ { t } } { 1 - \beta _ { 2 } ^ { t } } \bigg ) .\tag{136}
$$

Every flow except the stable flow self-stabilizes the preconditioned sharpness $\lambda _ { 1 } ( \hat { P } ^ { - 1 / 2 } H \hat { P } ^ { - 1 / 2 } )$ at the $3 8 / \eta$ threshold—that of heavy ball at $\beta _ { 1 } = 0 . 9$ . The preconditioned sharpness of the central flow sits at the threshold, while those of the rod flow and the discrete iterates fluctuate within a few percent of it.

Every other flow also tracks the center of the iterates more closely than the stable flow does:

![](images/f62726a290e8f8727b6abd3730c8baeba38fea9c2dc18debe3fe8102afb3da7d.jpg)  
Figure 11: Experimental Results for Adam (with bias correction). Left: MLP $( \eta = 1 0 ^ { - 5 } )$ Center: CNN $( \eta = 5 \times 1 0 ^ { - 5 } )$ . Right: ViT $( \eta = 1 0 ^ { - 5 } )$ . All three with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ . All flows except the stable flow self-stabilize the preconditioned sharpness at its threshold $3 8 / \eta .$ , and they track the discrete iterates more closely than the stable flow does.

twenty to thirty times more closely on the MLP and the ViT, and about six times more closely on the CNN. However, on the MLP and the CNN, this is a far smaller margin than was the case for gradient descent. Also, the rod flow and central flow for Adam perform much more alike than they did for gradient descent. Rod flow still tracks better, but central flow is only ten to forty percent farther from the discrete iterates than rod flow.

Why does the edge-coupling-derived formulation of the Adam rod flow perform well while the original formulation did not? We conjecture that the discrepancy comes from how the two formulations handle the slow drift. As $\beta _ { 1 }$ grows, the drift makes up a larger and larger share of the momentum. Because the extent of the original formulation tracks the second moment of the momentum—drift included—it can no longer self-stabilize once the drift becomes comparable in size to the oscillations.

Regis and Chewi [2026a] had suggested that period-doubling bifurcations could be a possible cause for the failure of self-stabilization of rod flow at large values of $\beta _ { 1 }$ . However, we found that the oscillations of Adam are in fact mostly period-two—though with some bursty behavior. More details on the validity of the near-period-two assumption can be found in Appendix H.5.

## 10 Conclusion

We have established connections between central flow, rod flow, and the edge coupling. Using the edge coupling, we derived an alternative formulation of rod flow—which now accurately tracks the iterates throughout all phases of training. In addition, we used the edge coupling to re-derive the central flows for gradient descent and RMSProp. This allowed us to generalize our derivation to obtain central flows for heavy ball and Adam. We have also shown that central flow and rod flow represent diferent coarse-grained descriptions of the dynamics—which explains how they can difer while both remaining accurate.

Limitations. Rod flow and central flow currently apply only to the full-batch setting. Both are also more expensive than the optimizers they model, so they are tools for understanding optimization dynamics rather than substitutes for the optimizers. Finally, while we have provided extensive experimental verification, we have not formally proven that rod flow or central flow precisely tracks the discrete iterates.

Future work. Natural next steps include extending rod flow and central flow to the minibatch setting. One would also like to prove formal tracking bounds between the flows and the discrete iterates. Finally, extending this framework to optimizers outside the Adam family remains future work.

## Acknowledgements

We would like to thank Raphaël Berthier for discussions. We used various LLMs—Opus 5.5, Fable 5.1, and GPT-6 Astra—in the brainstorming, drafting, and editing of this paper. We thank the authors of the Central Flow paper for making their code publicly available. Our implementation builds on code from https://github.com/alex-damian/EOS and https://github.com/centralflows/ centralflows.

## References

A. Agarwala, F. Pedregosa, and J. Pennington. Second-order regression models exhibit progressive sharpening to the edge of stability. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR, 2023.

K. Ahn, J. Zhang, and S. Sra. Understanding the unstable convergence of gradient descent. In International Conference on Machine Learning, pages 247–257. PMLR, 2022.

K. Ahn, S. Bubeck, S. Chewi, Y. T. Lee, F. Suarez, and Y. Zhang. Learning threshold neurons via edge of stability. Advances in Neural Information Processing Systems, 36:19540–19569, 2023.

A. Andreyev and P. Beneventano. Edge of stochastic stability: Revisiting the edge of stability for SGD. arXiv preprint arXiv:2412.20553, 2024.

A. Andreyev, A. Ananthkumar, M. Walden, T. Poggio, and P. Beneventano. Momentum further constrains sharpness at the edge of stochastic stability. arXiv preprint arXiv:2604.14108, 2026.

S. Arora, Z. Li, and A. Panigrahi. Understanding gradient descent on the edge of stability in deep learning. In International Conference on Machine Learning, pages 948–1024. PMLR, 2022.

K. J. Arrow, L. Hurwicz, and H. Uzawa. Studies in linear and non-linear programming. Stanford University Press, 1958.

D. Barrett and B. Dherin. Implicit gradient regularization. In International Conference on Learning Representations, 2021.

G. Benettin and A. Giorgilli. On the Hamiltonian interpolation of near-to-the-identity symplectic mappings with application to symplectic integration algorithms. Journal of Statistical Physics, 74 (5–6):1117–1143, 1994.

R. Berthier. The multiple timescales of gradient descent on the edge of stability: a perturbative derivation of the central flow. arXiv preprint arXiv:2609.01034, 2026.

L. Beyer, X. Zhai, and A. Kolesnikov. Better plain ViT baselines for ImageNet-1k. arXiv preprint arXiv:2205.01580, 2022.

R. Bhatia, T. Jain, and Y. Lim. On the Bures–Wasserstein distance between positive definite matrices. Expositiones Mathematicae, 37(2):165–191, 2019.

H. Brézis. Opérateurs maximaux monotones et semi-groupes de contractions dans les espaces de Hilbert, volume 5 of North-Holland Mathematics Studies. North-Holland, 1973.

L. Chen and J. Bruna. Beyond the edge of stability via two-step gradient updates. In International Conference on Machine Learning, pages 4330–4391. PMLR, 2023.

A. Chodron de Courcel. Mean–fluctuation dynamics at the edge of stability. arXiv preprint arXiv:2606.05326, 2026. Version 2; version 1 titled “Gradient descent at the edge of stability: free energy model and kinetic description of the two-layer network”.

J. Cohen, S. Kaur, Y. Li, J. Z. Kolter, and A. Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. In International Conference on Learning Representations, 2021.

J. Cohen, A. Damian, A. Talwalkar, J. Z. Kolter, and J. D. Lee. Understanding optimization in deep learning with central flows. In The Thirteenth International Conference on Learning Representations, 2025.

J. M. Cohen, B. Ghorbani, S. Krishnan, N. Agarwal, S. Medapati, M. Badura, D. Suo, D. Cardoze, Z. Nado, G. E. Dahl, and J. Gilmer. Adaptive gradient methods at the edge of stability. arXiv preprint arXiv:2207.14484, 2022.

B. Cornet. Existence of slow solutions for a class of diferential inclusions. Journal of Mathematical Analysis and Applications, 96(1):130–147, 1983.

A. Damian, E. Nichani, and J. D. Lee. Self-stabilization: the implicit bias of gradient descent at the edge of stability. In The Eleventh International Conference on Learning Representations, 2023.

C. Daskalakis, A. Ilyas, V. Syrgkanis, and H. Zeng. Training GANs with optimism. In International Conference on Learning Representations, 2018.

A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An image is worth 16x16 words: transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

P. Dupuis and A. Nagurney. Dynamical systems and variational inequalities. Annals of Operations Research, 44(1):7–42, 1993.

Y. Fong and H. Yang. Adam at the edge of stability: adaptive feedback, provable oscillation, and gradient reversal. arXiv preprint arXiv:2608.20638, 2026.

G. França, J. Sulam, D. P. Robinson, and R. Vidal. Conformal symplectic and relativistic optimization. In Advances in Neural Information Processing Systems, 2020.

E. Hairer, C. Lubich, and G. Wanner. Geometric numerical integration illustrated by the Störmer– Verlet method. Acta Numerica, 12:399–450, 2003.

E. Hairer, C. Lubich, and G. Wanner. Geometric numerical integration: structure-preserving algorithms for ordinary diferential equations. Springer, 2006.

C. Henry. An existence theorem for a class of diferential equations with multivalued right-hand side. Journal of Mathematical Analysis and Applications, 41(1):179–186, 1973.

R. Islamov, M. Crawshaw, J. Cohen, and R. Gower. Non-Euclidean gradient descent operates at the edge of stability. In International Conference on Machine Learning, 2026.

S. Jastrzębski, M. Szymczak, S. Fort, D. Arpit, J. Tabor, K. Cho, and K. Geras. The break-even point on optimization trajectories of deep neural networks. In International Conference on Learning Representations, 2020.

D. S. Kalra, T. He, and M. Barkeshli. Universal sharpness dynamics in neural network training: Fixed point analysis, edge of stability, and route to chaos. In The Thirteenth International Conference on Learning Representations, 2025.

D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

L. Kong and M. Tao. Stochasticity of deterministic gradient descent: Large learning rate for multiscale objective function. In Advances in Neural Information Processing Systems 33, 2020.

A. Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 4 2009.

Y. LeCun, L. Bottou, Y. Bengio, and P. Hafner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

A. Lewkowycz, Y. Bahri, E. Dyer, J. Sohl-Dickstein, and G. Gur-Ari. The large learning rate phase of deep learning: the catapult mechanism. arXiv preprint arXiv:2003.02218, 2020.

E. Litman. The origin of edge of stability. arXiv preprint arXiv:2604.20446, 2026.

I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

C. Ma, D. Kunin, L. Wu, and L. Ying. Beyond the quadratic approximation: the multiscale structure of neural network loss landscapes. Journal of Machine Learning, 1(3):247–267, 9 2022.

R. S. MacKay and J. D. Meiss. Linear stability of periodic orbits in Lagrangian systems. Physics Letters A, 98(3):92–94, 1983.

P. Marion. Edge flow: a tractable and predictive continuous-time model for gradient descent at the edge of stability. arXiv preprint arXiv:2606.18080, 2026.

J. E. Marsden and M. West. Discrete mechanics and variational integrators. Acta Numerica, 10: 357–514, 2001.

A. Meterez, P. A. Nair, D. Morwani, C. Pehlevan, S. Kakade, and A. Damian. A defense of the quadratic model. arXiv preprint arXiv:2607.21716, 2026.

M. Muehlebach and M. I. Jordan. Optimization with momentum: dynamical, control-theoretic, and symplectic perspectives. Journal of Machine Learning Research, 22(73):1–50, 2021.

A. Nagurney and D. Zhang. Projected Dynamical Systems and Variational Inequalities with Applications. International Series in Operations Research & Management Science. Kluwer Academic Publishers, 1996.

Y. E. Nesterov. A method for solving the convex programming problem with convergence rate $O ( 1 / k ^ { 2 } )$ . Proceedings of the USSR Academy of Sciences, 269:543–547, 1983.

A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, volume 32, 2019.

P. Phunyaphibarn, J. Lee, B. Wang, H. Zhang, and C. Yun. Gradient descent with Polyak’s momentum finds flatter minima via large catapults. arXiv preprint 2311.15051, 2024.

B. T. Polyak. Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics, 4(5):1–17, 1964.

L. D. Popov. A modification of the Arrow–Hurwicz method for search of saddle points. Mathematical Notes of the Academy of Sciences of the USSR, 28(5):845–848, 1980.

E. Regis. Variational structure at the edge of stability. arXiv preprint arXiv:2608.21660, 2026.

E. Regis and S. Chewi. A rod flow model for Adam at the edge of stability. arXiv preprint arXiv:2605.06821, 2026a.

E. Regis and S. Chewi. Rod flow: a continuous-time model for gradient descent at the edge of stability. arXiv preprint arXiv:2602.01480, 2026b.

M. Rosca, Y. Wu, C. Qin, and B. Dherin. On a continuous time model of gradient descent dynamics and instability in deep learning. Transactions on Machine Learning Research, 2023.

D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by back-propagating errors. Nature, 323(6088):533–536, 1986.

B. Shi, S. S. Du, M. I. Jordan, and W. J. Su. Understanding the acceleration phenomenon via high-resolution diferential equations. Mathematical Programming, 195(1–2):79–148, 2022.

S. L. Smith, B. Dherin, D. Barrett, and S. De. On the origin of implicit regularization in stochastic gradient descent. In International Conference on Learning Representations, 2021.

M. Song and C. Yun. Trajectory alignment: understanding the edge of stability phenomenon via bifurcation theory. In Advances in Neural Information Processing Systems, volume 36, 2023.

I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the importance of initialization and momentum in deep learning. In International Conference on Machine Learning, pages 1139–1147, 2013.

T. Tieleman and G. Hinton. Lecture 6.5—RMSProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 2012.

L. Verlet. Computer “experiments” on classical fluids. I. Thermodynamical properties of Lennard-Jones molecules. Physical Review, 159(1):98–103, 1967.

A. Wibisono, A. C. Wilson, and M. I. Jordan. A variational perspective on accelerated methods in optimization. Proceedings of the National Academy of Sciences, 113(47):E7351–E7358, 2016.

X. Zhu, Z. Wang, X. Wang, M. Zhou, and R. Ge. Understanding edge-of-stability training dynamics with a minimalist example. In International Conference on Learning Representations, 2023.

## A Notation

Table 1: Notation used throughout the main body.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\eta$ </td><td>Step size</td></tr><tr><td> $\beta$ </td><td>Momentum coefficient</td></tr><tr><td> $\beta _ { 1 }$ </td><td>First-moment decay rate</td></tr><tr><td> $\beta _ { 2 }$ </td><td>Second-moment decay rate</td></tr><tr><td> $\epsilon$ </td><td>Numerical stability constant</td></tr><tr><td> $d$ </td><td>Number of parameters</td></tr><tr><td> $L$ </td><td>Loss</td></tr><tr><td> $H$ </td><td>Hessian of the loss</td></tr><tr><td> $\lambda _ { 1 } ( H )$ </td><td>Sharpness: the largest eigenvalue of H</td></tr><tr><td> $\lambda _ { i } , \hat { u } _ { i }$ </td><td>Eigenvalues and eigenvectors of H</td></tr><tr><td> $S ^ { \star }$ </td><td>Sharpness threshold</td></tr><tr><td> $t _ { c }$ </td><td>Crossing time: first iteration with sharpness above  $S ^ { \star }$ </td></tr><tr><td> $\bar { H } _ { t }$ </td><td>Hessian averaged along a gradient step</td></tr><tr><td> $w _ { t }$ </td><td>Position at iteration t</td></tr><tr><td> $m _ { t }$ </td><td>Momentum at iteration t</td></tr><tr><td> $\nu _ { t }$ </td><td>Second moment at iteration t</td></tr><tr><td> $z = ( w , m )$ </td><td>Phase-space point</td></tr><tr><td> $\Phi$ </td><td>Update map of the optimizer</td></tr><tr><td> $P ( \nu )$ </td><td>Preconditioner  $\mathrm { d i a g } ( \sqrt { \nu } ) + \epsilon I _ { d }$ </td></tr><tr><td> $\hat { P }$   $v ^ { \odot 2 }$ </td><td>Preconditioner at the bias-corrected second moment</td></tr><tr><td></td><td>Elementwise square of a vector v</td></tr><tr><td> $\bar { w }$ </td><td>Midpoint of consecutive positions</td></tr><tr><td> $\bar { m }$ </td><td>Midpoint of consecutive momenta</td></tr><tr><td> $\bar { \nu }$ </td><td>Second moment of the flows</td></tr><tr><td> $\bar { z } = ( \bar { w } , \bar { m } )$ </td><td>Phase-space midpoint</td></tr><tr><td> $\hat { \delta } _ { t }$ </td><td>Half-displacement of consecutive iterates</td></tr><tr><td> $a _ { t }$ </td><td>Oscillation envelope of the discrete iterates</td></tr><tr><td> $\delta$ </td><td>Position half-difference (extent of the rod)</td></tr><tr><td> $\gamma$ </td><td>Momentum half-difference</td></tr><tr><td> $\Delta = ( \delta , \gamma )$ </td><td>Phase-space half-difference</td></tr><tr><td> $\Sigma$ </td><td>Extent matrix (rod flow); covariance matrix (central flow)</td></tr><tr><td> $\bar { g }$ </td><td>Average of the gradients at the rod&#x27;s endpoints</td></tr><tr><td> $\frac { [ [ g ] ] } { g ^ { \otimes 2 } }$ </td><td>Half-difference of the gradients at the rod&#x27;s endpoints</td></tr><tr><td></td><td>Average outer product of the endpoint gradients</td></tr><tr><td> $\Sigma _ { \mathrm { E C } } ^ { * } , \Sigma _ { \mathrm { S M } } ^ { * }$ </td><td>Fixed points of the extent: covariance and second moment</td></tr><tr><td> $K$ </td><td>Structure matrix</td></tr><tr><td> $\mathcal { L }$ </td><td>Generating function of a structured gradient method</td></tr><tr><td> $\boldsymbol { B }$ </td><td>Edge coupling of a structured gradient method</td></tr><tr><td> $\varLambda _ { \eta } , \Psi$ </td><td>Edge coupling; centered edge coupling</td></tr><tr><td> $\mathcal { A } _ { \eta , \nu } , \Psi _ { \eta , \nu }$ </td><td>Edge couplings of RMSProp</td></tr><tr><td></td><td>Phase-space edge coupling; centered phase-space edge coupling</td></tr><tr><td> $B _ { \eta , \beta } , \Upsilon _ { \eta , \beta }$ </td><td></td></tr><tr><td> $B _ { \eta , \beta _ { 1 } , \nu } , \Upsilon _ { \eta , \beta _ { 1 } , \nu }$ </td><td>Phase-space edge couplings of Adam</td></tr><tr><td> $\widetilde { \Psi } , \widetilde { \Upsilon }$   $\Upsilon ^ { \mathrm { r e d } }$ </td><td>Truncated edge couplings</td></tr><tr><td></td><td>Reduced edge coupling</td></tr><tr><td> $\widetilde { \Upsilon } ^ { \mathrm { r e d } }$ </td><td>Truncated reduced edge coupling</td></tr></table>

Table 1, continued.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\Lambda$ </td><td>Slack  $\frac { 2 } { \eta } I _ { d } - H$ </td></tr><tr><td> $\dot { \Lambda }$ </td><td>Rate of change of the slack</td></tr><tr><td>ΛRMSProp,  $\Lambda _ { \mathrm { H B } } .$  ΛAdam</td><td>Slacks of RMSProp, heavy ball, and Adam</td></tr><tr><td> $U$ </td><td>Orthonormal basis of the critical directions</td></tr><tr><td> $k$ </td><td>Number of critical directions</td></tr><tr><td> $X$ </td><td>Covariance expressed in the basis U</td></tr><tr><td> $\sigma ^ { 2 }$ </td><td>Squared magnitude of the oscillation</td></tr><tr><td> $\kappa ( \bar { w } )$ </td><td>Critical face</td></tr><tr><td> $\Omega$ </td><td>Feasible region</td></tr><tr><td>S</td><td>Oscillation amplitude</td></tr><tr><td>x</td><td>Excess sharpness</td></tr><tr><td>r</td><td>Progressive-sharpening constant</td></tr><tr><td>C</td><td>Sharpness-reduction coefficient</td></tr><tr><td> $T$ </td><td>Averaging window</td></tr><tr><td> $\hat { v }$   $\Sigma _ { \mathfrak { c } }$ </td><td>Axis of the rod</td></tr><tr><td> $\Sigma _ { \mathrm { r o d } }$  entral</td><td>Extent matrix of rod flow; covariance matrix of central flow</td></tr><tr><td> $\mathrm { P R }$ </td><td>Participation ratio  $( \operatorname { t r } \Sigma ) ^ { 2 } / \operatorname { t r } ( \Sigma ^ { 2 } )$ </td></tr></table>

## B Edge Couplings of Structured Gradient Methods

This appendix provides further information about the edge coupling for arbitrary structured gradient methods.

## B.1 Structured Gradient Methods

Consider a linear optimizer of the form

$$
\boldsymbol { z } _ { t + 1 } = \boldsymbol { z } _ { t } + J \nabla \mathcal { G } ( \boldsymbol { z } _ { t } ) ,\tag{137}
$$

where J is a constant matrix and $\mathcal { G }$ is a generating function. We will call such an optimizer a linear gradient method, since its update is linear in the gradient of $\mathcal { G }$ . Importantly, we are not assuming that J is symmetric.

We will show that if an edge coupling generates the residuals as its derivatives, then the structure matrix must be symmetric. Let $B ( z , z ^ { \prime } )$ be a function that is symmetric under the exchange $z  z ^ { \prime }$ Suppose that $\boldsymbol { B }$ satisfies

$$
\nabla _ { z } B ( z , z ^ { \prime } ) = K ^ { - 1 } \big ( z + J \nabla \mathcal { G } ( z ) - z ^ { \prime } \big )\tag{138}
$$

for some constant invertible matrix K. By the symmetry of $B ,$ we also have that

$$
\nabla _ { z ^ { \prime } } \boldsymbol { B } ( z , z ^ { \prime } ) = K ^ { - 1 } \big ( z ^ { \prime } + J \nabla \mathcal { G } ( z ^ { \prime } ) - z \big ) .\tag{139}
$$

By diferentiating the first equation with respect to $z ^ { \prime }$ and the second equation with respect to $z ,$ we can obtain the mixed second derivatives of $B { \mathrm { : } }$

$$
\nabla _ { z ^ { \prime } } \nabla _ { z } \boldsymbol { B } = - K ^ { - 1 } , \qquad \nabla _ { z } \nabla _ { z ^ { \prime } } \boldsymbol { B } = - K ^ { - 1 } .\tag{140}
$$

Because mixed derivatives commute, we have that

$$
\nabla _ { z ^ { \prime } } \nabla _ { z } B = ( \nabla _ { z } \nabla _ { z ^ { \prime } } B ) ^ { T } \implies K = K ^ { T } .\tag{141}
$$

We have that K is symmetric.

Next, we can subtract from $\boldsymbol { B }$ the quadratic term of a structured edge coupling and take the derivative with respect to z:

$$
\begin{array} { r l } & { \nabla _ { z } \Big [ B ( z , z ^ { \prime } ) - \frac { 1 } { 2 } ( z - z ^ { \prime } ) ^ { \top } K ^ { - 1 } ( z - z ^ { \prime } ) \Big ] = K ^ { - 1 } \big ( z + J \nabla \mathcal { G } ( z ) - z ^ { \prime } \big ) - K ^ { - 1 } ( z - z ^ { \prime } ) } \\ & { \qquad = K ^ { - 1 } J \nabla \mathcal { G } ( z ) . } \end{array}\tag{142}
$$

The right-hand side doesn’t depend on $z ^ { \prime } .$ . That means there exists a function $\mathcal { L }$ such that

$$
K ^ { - 1 } J \nabla \mathcal { G } ( z ) = \nabla \mathcal { L } ( z )\tag{143}
$$

By taking the anti-derivative of both sides with respect to $z ,$ it then follows:

$$
B ( z , z ^ { \prime } ) - \textstyle { \frac { 1 } { 2 } } ( z - z ^ { \prime } ) ^ { \top } K ^ { - 1 } ( z - z ^ { \prime } ) = \mathcal { L } ( z ) + f ( z ^ { \prime } )\tag{144}
$$

for some function $f .$ By the symmetry of $B ,$ f equals $\mathcal { L }$ up to an additive constant $C .$ . Setting $C = 0$ , it then follows that:

$$
\begin{array} { r } { \mathcal { B } ( z , z ^ { \prime } ) = \mathcal { L } ( z ) + \mathcal { L } ( z ^ { \prime } ) + \frac { 1 } { 2 } ( z - z ^ { \prime } ) ^ { \top } K ^ { - 1 } ( z - z ^ { \prime } ) . } \end{array}\tag{145}
$$

The above shows that $i f$ an edge coupling exists for a linear gradient method, then the associated structure matrix must be symmetric. A reasonable follow-up question is: For a given linear gradient method, can we find a mathematically equivalent structured gradient method? Specifically, for a matrix $J$ and generating function $\mathcal { G }$ , do there exist a symmetric structure matrix K and a generating function $\mathcal { L }$ such that

$$
J \nabla \mathcal { G } = K \nabla \mathcal { L } ?\tag{146}
$$

Let $z = ( w , m ) \in \mathbb { R } ^ { 2 d }$ , where $w , m \in \mathbb { R } ^ { d }$ . Consider generating functions of the form

$$
\begin{array} { r } { \mathcal G ( z ) = L ( w ) + \frac { 1 } { 2 } z ^ { \top } Q z , } \end{array}\tag{147}
$$

where $Q$ is a constant symmetric matrix. Many common optimizers are generated by functions of this form, including heavy-ball momentum—whose generating function is its Hamiltonian. We will show that every linear gradient method generated by such a function can be put in structured gradient form.

Recall that, assuming $\mathcal { L }$ and K exist, we have

$$
\nabla \mathcal { L } ( z ) = K ^ { - 1 } J \nabla \mathcal { G } ( z ) .\tag{148}
$$

Diferentiating both sides with respect to z gives

$$
\nabla ^ { 2 } \mathcal { L } ( z ) = K ^ { - 1 } J \nabla ^ { 2 } \mathcal { G } ( z ) .\tag{149}
$$

Since mixed partial derivatives commute, $\nabla ^ { 2 } { \mathcal { L } } ( z )$ is symmetric. It follows that

$$
K ^ { - 1 } J \nabla ^ { 2 } \mathcal { G } ( z ) \mathrm { ~ i s ~ s y m m e t r i c ~ f o r ~ a l l ~ } z .\tag{150}
$$

We can use this constraint to solve for $K$

For the Hessian of $\mathcal { G }$ , we have

$$
\nabla ^ { 2 } \mathcal { G } ( z ) = \binom { H ( w ) } { 0 } + Q .\tag{151}
$$

As the structure matrix should not depend on the specific loss function, (150) must hold for every symmetric H—including $H = 0$ . It must therefore hold for the loss term and the quadratic term separately:

$$
K ^ { - 1 } J \left( \begin{array} { c c } { { H } } & { { 0 } } \\ { { 0 } } & { { 0 } } \end{array} \right) \mathrm { ~ i s ~ s y m m e t r i c } \qquad \mathrm { a n d } \qquad K ^ { - 1 } J Q \mathrm { ~ i s ~ s y m m e t r i c . }\tag{152}
$$

For the loss term, we have that:

$$
\begin{array}{c} K ^ { - 1 } J \left( \begin{array} { l l } { H } & { 0 } \\ { 0 } & { 0 } \end{array} \right) = \left( K ^ { - 1 } J ) _ { w w } H  & { 0 } \\ { ( K ^ { - 1 } J ) _ { m w } H } & { 0 } \end{array} \right) .\tag{153}
$$

The upper-right block of the matrix on the right-hand side is zero. For this matrix to be symmetric, the lower-left block must vanish as well. Since this must hold for every symmetric H, we can deduce that

$$
( K ^ { - 1 } J ) _ { m w } = 0 .\tag{154}
$$

Now consider the upper-left block $( K ^ { - 1 } J ) _ { w w } H$ , which must be symmetric for every symmetric H. The only matrices with this property are multiples of the identity, so

$$
( K ^ { - 1 } J ) _ { w w } = \alpha I _ { d }\tag{155}
$$

for some scalar α. Combining these two results, the first block column of $K ^ { - 1 } J$ is

$$
( K ^ { - 1 } J ) \left( \begin{array} { c } { { I _ { d } } } \\ { { 0 } } \end{array} \right) = \alpha \left( \begin{array} { c c } { { I _ { d } } } \\ { { 0 } } \end{array} \right) .\tag{156}
$$

By associativity, this is equivalent to

$$
K ^ { - 1 } \left( J \left( { \cal I } _ { d } \right) \right) = \alpha \left( { \cal I } _ { d } \right) .\tag{157}
$$

Assuming $\alpha \neq 0$ and multiplying both sides on the left by $K / \alpha$ , we obtain

$$
K \left( \begin{array} { c } { { I _ { d } } } \\ { { 0 } } \end{array} \right) = \frac 1 \alpha J \left( \begin{array} { c } { { I _ { d } } } \\ { { 0 } } \end{array} \right) .\tag{158}
$$

This fixes the first block column of K. And because K is symmetric, this also fixes the first block row. Note that for K to be symmetric we require that $J _ { w w }$ is symmetric. The only remaining unknown of K is the $( m , m )$ block:

$$
K = \left( \begin{array} { c c } { { \frac { 1 } { \alpha } J _ { w w } } } & { { \frac { 1 } { \alpha } J _ { m w } ^ { \top } } } \\ { { \frac { 1 } { \alpha } J _ { m w } } } & { { K _ { m m } } } \end{array} \right) .\tag{159}
$$

To solve for $K _ { m m }$ , we use the condition that $K ^ { - 1 } J Q$ must be symmetric. We assume that the sub-blocks of J and Q are symmetric and commute with one another. Without this assumption, the symmetry of $K ^ { - 1 } J Q$ would impose more scalar equations than $K _ { m m }$ has entries, and in general there would be no solution. We likewise seek $K _ { m m }$ among matrices that commute with these blocks.

Under these assumptions, we can invert K as we would a $2 \times 2$ matrix:

$$
K ^ { - 1 } = ( K _ { w w } K _ { m m } - K _ { w m } ^ { 2 } ) ^ { - 1 } \binom { K _ { m m } } { - K _ { w m } } \quad K _ { w w } ) .\tag{160}
$$

Every sub-block of $K ^ { - 1 } J Q$ is then built from commuting symmetric matrices, and is therefore itself symmetric. Hence $K ^ { - 1 } J Q$ is symmetric exactly when its two of-diagonal sub-blocks are equal. Setting them equal and cancelling the common factor $\left( K _ { w w } K _ { m m } - K _ { w m } ^ { 2 } \right) ^ { - 1 }$ , we obtain

$$
K _ { m m } ( J Q ) _ { w m } - K _ { w m } ( J Q ) _ { m m } = K _ { w w } ( J Q ) _ { m w } - K _ { w m } ( J Q ) _ { w w } .\tag{161}
$$

Recall that the first block column of $K$ is $1 / \alpha$ times the first block column of $^ { J , }$ so $K _ { w w } = J _ { w w } / \alpha$ and $K _ { m w } = J _ { m w } / \alpha$ . Substituting these expressions and using $K _ { w m } = K _ { m w } ,$ we obtain a linear equation for $K _ { m m }$

$$
( J Q ) _ { w m } K _ { m m } = { \frac { 1 } { \alpha } } \Bigl [ J _ { w w } ( J Q ) _ { m w } + J _ { m w } ( ( J Q ) _ { m m } - ( J Q ) _ { w w } ) \Bigr ] .\tag{162}
$$

$\mathrm { I f ~ } ( J Q ) _ { w m }$ is invertible, this equation has exactly one solution:

$$
K _ { m m } = \frac { 1 } { \alpha } ( J Q ) _ { w m } ^ { - 1 } \Big [ J _ { w w } ( J Q ) _ { m w } + J _ { m w } \big ( ( J Q ) _ { m m } - ( J Q ) _ { w w } \big ) \Big ] .\tag{163}
$$

The structure matrix is therefore

$$
\boxed { K = \cfrac { 1 } { \alpha } \left( J _ { w w } \left( \ J Q \right) _ { w m } ^ { - 1 } \left[ J _ { w w } ( J Q ) _ { m w } + J _ { m w } ( ( J Q ) _ { m m } - ( J Q ) _ { w w } ) \right] \right) \bigg | . }\tag{164}
$$

$\mathrm { A s }$ we have found $K .$ , we can integrate $K ^ { - 1 } J \nabla \mathcal { G }$ to obtain $\mathcal { L } .$ Substituting the gradient of ${ \mathcal { G } } ,$ , and using (158), we have that:

$$
\begin{array} { c } { { \nabla \mathcal { L } ( z ) = K ^ { - 1 } J \nabla \mathcal { G } } } \\ { { \ } } \\ { { \mathrm { ~ } = K ^ { - 1 } J \left( \begin{array} { c } { { \nabla L ( w ) } } \\ { { 0 } } \end{array} \right) + K ^ { - 1 } J Q z } } \\ { { \ } } \\ { { \mathrm { ~ } = \alpha \left( \begin{array} { c } { { \nabla L ( w ) } } \\ { { 0 } } \end{array} \right) + K ^ { - 1 } J Q z . } } \end{array}\tag{165}
$$

The first term is the gradient of $\alpha L ( w )$ . Since $K ^ { - 1 } J Q$ is symmetric, the second term is the gradient of ${ \scriptstyle { \frac { 1 } { 2 } } } z ^ { \top } K ^ { - 1 } J Q z$ . Taking the anti-derivative, we obtain the generating function:

$$
\begin{array} { r } { \boxed { \mathcal { L } ( z ) = \alpha L ( w ) + \frac { 1 } { 2 } z ^ { \top } K ^ { - 1 } J Q z . } } \end{array}\tag{166}
$$

Since the structure matrix $K$ is proportional to $1 / \alpha .$ , the product $K ^ { - 1 } J Q$ is proportional to $\alpha$ . It follows that $\mathcal { L }$ is also proportional to $\alpha _ { \mathrm { { ; } } }$ , while $K \nabla \mathcal L$ is independent of $\alpha .$ . We therefore obtain a one-parameter family of valid structured forms—and as long as $( J Q ) _ { w m }$ is invertible, there are no others. In the rest of the paper we take $\alpha = 1$ , so that the loss appears in $\mathcal { L }$ with coeficient one.

## B.2 Heavy Ball

Heavy ball can be expressed as a linear gradient method by using the Hamiltonian as the generating function [França et al., 2020]. Let $z = ( w , m )$ denote a point in phase space. For the Hamiltonian, we have:

$$
\begin{array} { r } { \mathcal { H } ( w , m ) = L ( w ) + \frac { \eta } { 2 ( 1 - \beta ) } \| m \| ^ { 2 } . } \end{array}\tag{167}
$$

We have that the loss plays the role of the potential energy, and the momentum-squared plays the role of the kinetic energy. Note that we have that

$$
Q = \left( { \begin{array} { c c } { 0 } & { 0 } \\ { 0 } & { { \frac { \eta } { 1 - \beta } } I _ { d } } \end{array} } \right) .\tag{168}
$$

The heavy ball update is given as $\boldsymbol { z } _ { t + 1 } = \boldsymbol { z } _ { t } + J \nabla \mathcal { H } ( \boldsymbol { z } _ { t } )$ , where

$$
\begin{array} { r l } & { J = - ( 1 - \beta ) \left( \begin{array} { l l } { \eta I _ { d } } & { \beta I _ { d } } \\ { - I _ { d } } & { \frac { 1 - \beta } { \eta } I _ { d } } \end{array} \right) } \\ & { \quad = - \frac { ( 1 - \beta ) ( 1 + \beta ) } { 2 } \left( \begin{array} { l l } { 0 } & { I _ { d } } \\ { - I _ { d } } & { 0 } \end{array} \right) - ( 1 - \beta ) \left( \begin{array} { c c } { \eta I _ { d } } & { - \frac { 1 - \beta } { 2 } I _ { d } } \\ { - \frac { 1 - \beta } { 2 } I _ { d } } & { \frac { 1 - \beta } { \eta } I _ { d } } \end{array} \right) . } \end{array}\tag{169}
$$

The first part of J is antisymmetric. Up to an overall factor, it is the symplectic matrix of Hamiltonian mechanics, which encodes how force drives changes in momentum and momentum drives changes in position. The second part is symmetric and negative definite—it supplies the damping. Since $J$ is not symmetric, the damped formulation is not in structured gradient form.

As $Q$ only involves $Q _ { m m }$ , we can actually derive a simplified expression for $K .$ . The product $J Q$ only has a second block column:

$$
J Q = \left( \begin{array} { c c } { { 0 } } & { { J _ { w m } Q _ { m m } } } \\ { { 0 } } & { { J _ { m m } Q _ { m m } } } \end{array} \right) .\tag{170}
$$

Substituting these blocks into (164), only the term $J _ { m w } ( J Q ) _ { m m }$ survives in the lower-right block. We then have that:

$$
\begin{array} { c } { { K _ { m m } } = \left( { J _ { w m } } { Q _ { m m } } \right) ^ { - 1 } J _ { m w } J _ { m m } Q _ { m m } } \\ { = J _ { w m } ^ { - 1 } J _ { m w } J _ { m m } . } \end{array}\tag{171}
$$

Recall that the first block column of K coincides with that of J. We also have that

$$
\begin{array} { r l } { { } } & { { K _ { w m } } = J _ { m w } } \\ { { } } & { { } } \\ { { } } & { { } } \end{array}\tag{172}
$$

Thus, the second block column of K is that of $^ { J , }$ right-multiplied by $J _ { w m } ^ { - 1 } J _ { m w }$ . Altogether,

$$
K = J \left( \begin{array} { c c } { { I _ { d } } } & { { 0 } } \\ { { 0 } } & { { J _ { w m } ^ { - 1 } J _ { m w } } } \end{array} \right) .\tag{173}
$$

Substituting, the generating function is

$$
\begin{array} { r } { \mathcal { L } ( z ) = L ( w ) + \frac { 1 } { 2 } m ^ { \top } J _ { m w } ^ { - 1 } J _ { w m } Q _ { m m } m . } \end{array}\tag{174}
$$

For the case of heavy ball, we have that

$$
\begin{array} { r } { J _ { w m } ^ { - 1 } J _ { m w } = \frac { 1 - \beta } { - \beta ( 1 - \beta ) } I _ { d } = - \frac { 1 } { \beta } I _ { d } . } \end{array}\tag{175}
$$

The structured form flips the sign of the second block column of J and divides it by $\beta \colon$

$$
\begin{array} { r }  K = J \left( \begin{array} { c c } { I _ { d } } & { 0 } \\ { 0 } & { - \frac { 1 } { \beta } I _ { d } \right) = \left( \begin{array} { c c } { - \eta ( 1 - \beta ) I _ { d } } & { ( 1 - \beta ) I _ { d } } \\ { ( 1 - \beta ) I _ { d } } & { \frac { ( 1 - \beta ) ^ { 2 } } { \eta \beta } I _ { d } } \end{array} \right) . } \end{array} \end{array}\tag{176}
$$

And for the generating function, we have that the kinetic energy is multiplied by $- \beta$ :

$$
\begin{array} { r } { \mathcal { L } ( w , m ) = L ( w ) - \frac { \eta \beta } { 2 ( 1 - \beta ) } \| m \| ^ { 2 } . } \end{array}\tag{177}
$$

## B.3 Other Structured Gradient Methods

We briefly cover some other structured gradient methods that admit edge couplings.

Gradient descent–ascent. Simultaneous gradient descent–ascent on $f ( u , v )$ [Arrow et al., 1958] descends in u and ascends in v:

$$
u _ { t + 1 } = u _ { t } - \eta \nabla _ { u } f ( u _ { t } , v _ { t } ) , \qquad v _ { t + 1 } = v _ { t } + \eta \nabla _ { v } f ( u _ { t } , v _ { t } ) .\tag{178}
$$

For the generating function and structure matrix, we have:

$$
\begin{array} { r } { \mathcal { L } ( u , v ) = f ( u , v ) , \qquad K = \left( \begin{array} { l l } { - \eta I } & { 0 } \\ { 0 } & { \eta I } \end{array} \right) . } \end{array}\tag{179}
$$

For the edge coupling, we have:

$$
\begin{array} { r } { \mathcal { B } ( u , v , u ^ { \prime } , v ^ { \prime } ) = f ( u , v ) + f ( u ^ { \prime } , v ^ { \prime } ) - \frac { 1 } { 2 \eta } \| u - u ^ { \prime } \| ^ { 2 } + \frac { 1 } { 2 \eta } \| v - v ^ { \prime } \| ^ { 2 } . } \end{array}\tag{180}
$$

Optimistic gradient descent. Optimistic gradient descent [Popov, 1980, Daskalakis et al., 2018] extrapolates from the previous gradient:

$$
w _ { t + 1 } = w _ { t } - 2 \eta \nabla L ( w _ { t } ) + \eta \nabla L ( w _ { t - 1 } ) .\tag{181}
$$

In the coordinates $( w _ { t } , w _ { t - 1 } )$ , optimistic gradient descent is not a structured gradient method, since the loss function is evaluated at two diferent points. We can fix this with a change of coordinates. Let $q _ { t } : = \nabla L ( w _ { t - 1 } )$ denote the previous gradient, and let $z = ( w , q )$ . Written in terms of diferences, the update in these coordinates is

$$
\begin{array} { r l } & { w _ { t + 1 } - w _ { t } = - 2 \eta \nabla L ( w _ { t } ) + \eta q _ { t } , } \\ & { ~ q _ { t + 1 } - q _ { t } = \nabla L ( w _ { t } ) - q _ { t } , } \end{array}\tag{182}
$$

In this form, we have that optimistic gradient descent is a linear gradient method with the following generating function and structure matrix:

$$
\begin{array} { r } { \mathcal { G } ( \boldsymbol { w } , \boldsymbol { q } ) = L ( \boldsymbol { w } ) + \frac { 1 } { 2 } \| \boldsymbol { q } \| ^ { 2 } , \qquad \boldsymbol { J } = \left( \begin{array} { c c } { - 2 \eta I _ { d } } & { \eta I _ { d } } \\ { I _ { d } } & { - I _ { d } } \end{array} \right) . } \end{array}\tag{183}
$$

Since the generating function is a loss function plus a quadratic form, optimistic gradient descent can be written as a structured gradient method, with $\mathcal { L }$ and K given by

$$
\begin{array} { r } { \mathcal { L } ( w , q ) = L ( w ) + \frac { \eta } { 2 } \| q \| ^ { 2 } , \qquad K = \left( \begin{array} { c c } { - 2 \eta I _ { d } } & { I _ { d } } \\ { I _ { d } } & { - \frac { 1 } { \eta } I _ { d } } \end{array} \right) . } \end{array}\tag{184}
$$

For the edge coupling, we have:

$$
\begin{array} { r l } & { \mathcal { B } ( w , q , w ^ { \prime } , q ^ { \prime } ) = L ( w ) + L ( w ^ { \prime } ) + \frac { \eta } { 2 } ( \| q \| ^ { 2 } + \| q ^ { \prime } \| ^ { 2 } ) } \\ & { \qquad - \frac { 1 } { 2 \eta } \| w - w ^ { \prime } \| ^ { 2 } - ( w - w ^ { \prime } ) \cdot ( q - q ^ { \prime } ) - \eta \| q - q ^ { \prime } \| ^ { 2 } . } \end{array}\tag{185}
$$

## C Nesterov Momentum

This appendix derives the edge coupling, rod flow, and central flow for Nesterov momentum.

## C.1 Setup

Nesterov momentum [Nesterov, 1983] is an alternative to heavy-ball momentum. The two difer only in where the gradient in the momentum update is evaluated—a diference that shows up in continuous time as a gradient-correction term [Shi et al., 2022]. Rather than evaluating the gradient at the current iterate, Nesterov momentum evaluates it at a look-ahead point: the location the iterate would reach if it were updated by the momentum term alone [Sutskever et al., 2013]. The update equations are:

$$
\begin{array} { r l } & { m _ { t + 1 } = \beta m _ { t } + ( 1 - \beta ) \nabla L ( w _ { t } - \eta \beta m _ { t } ) , } \\ & { } \\ & { w _ { t + 1 } = w _ { t } - \eta m _ { t + 1 } . } \end{array}\tag{186}
$$

Nesterov momentum is arguably more natural when expressed in look-ahead coordinates. Define the look-ahead point:

$$
\theta _ { t } = w _ { t } - \eta \beta m _ { t } .\tag{187}
$$

In the coordinates $( \theta , m )$ , the update equations no longer involve a look-ahead. The gradient is evaluated at the current iterate itself:

$$
\begin{array} { r l } & { m _ { t + 1 } = \beta m _ { t } + ( 1 - \beta ) \nabla L ( \theta _ { t } ) , } \\ & { \theta _ { t + 1 } = \theta _ { t } - \eta \big [ \beta m _ { t + 1 } + ( 1 - \beta ) \nabla L ( \theta _ { t } ) \big ] . } \end{array}\tag{188}
$$

We will work in look-ahead coordinates throughout.

## C.2 Edge Coupling

In look-ahead coordinates, Nesterov momentum is a structured gradient method. Written as diferences, the update is

$$
\begin{array} { r l } & { \theta _ { t + 1 } - \theta _ { t } = - \eta \big [ \beta ^ { 2 } m _ { t } + ( 1 - \beta ^ { 2 } ) \nabla L ( \theta _ { t } ) \big ] , } \\ & { m _ { t + 1 } - m _ { t } = ( 1 - \beta ) \big [ \nabla L ( \theta _ { t } ) - m _ { t } \big ] , } \end{array}\tag{189}
$$

which is $K \nabla _ { ( \theta , m ) } \mathcal { L }$ for the generating function and structure matrix

$$
\begin{array} { r } { \mathcal { L } ( \theta , m ) = L ( \theta ) - \frac { \eta \beta ^ { 2 } } { 2 ( 1 - \beta ) } \| m \| ^ { 2 } , \qquad K = \left( { \begin{array} { c c } { - \eta ( 1 - \beta ^ { 2 } ) I _ { d } } & { ( 1 - \beta ) I _ { d } } \\ { ( 1 - \beta ) I _ { d } } & { \frac { ( 1 - \beta ) ^ { 2 } } { \eta \beta ^ { 2 } } I _ { d } } \end{array} } \right) , } \end{array}\tag{190}
$$

The inverse of the structure matrix is

$$
\begin{array} { r } { K ^ { - 1 } = \left( \begin{array} { c c } { - \frac { 1 } { \eta } I _ { d } } & { \frac { \beta ^ { 2 } } { 1 - \beta } I _ { d } } \\ { \frac { \beta ^ { 2 } } { 1 - \beta } I _ { d } } & { \frac { \eta \beta ^ { 2 } ( 1 + \beta ) } { 1 - \beta } I _ { d } } \end{array} \right) . } \end{array}\tag{191}
$$

Let $z = ( \theta , m )$ and $z ^ { \prime } = ( \theta ^ { \prime } , m ^ { \prime } )$ . We then obtain the phase-space edge coupling of Nesterov momentum [Regis, 2026]:

$$
\begin{array} { r l } & { \mathcal { B } _ { \eta , \beta } ^ { \mathrm { N A G } } ( z , z ^ { \prime } ) = L ( \theta ) + L ( \theta ^ { \prime } ) - \frac { 1 } { 2 \eta } \| \theta - \theta ^ { \prime } \| ^ { 2 } + \frac { \beta ^ { 2 } } { 1 - \beta } ( m - m ^ { \prime } ) \cdot ( \theta - \theta ^ { \prime } ) } \\ & { \qquad + \frac { \eta \beta ^ { 3 } } { 2 ( 1 - \beta ) } \| m - m ^ { \prime } \| ^ { 2 } - \frac { \eta \beta ^ { 2 } } { 1 - \beta } m \cdot m ^ { \prime } . } \end{array}\tag{192}
$$

Its critical points are the fixed points and period-two orbits of Nesterov momentum.

Define the centered coordinates

$$
\bar { z } = \binom { \bar { \theta } } { \bar { m } } = \frac { z + z ^ { \prime } } { 2 } , \qquad \Delta = \binom { \delta } { \gamma } = \frac { z - z ^ { \prime } } { 2 } .\tag{193}
$$

In these coordinates, the centered phase-space edge coupling is

$$
\begin{array} { r l } & { \Upsilon _ { \eta , \beta } ^ { \mathrm { N A G } } ( \bar { z } , \Delta ) : = \frac { 1 } { 2 } \mathcal { B } _ { \eta , \beta } ^ { \mathrm { N A G } } ( \bar { z } + \Delta , \bar { z } - \Delta ) } \\ & { \quad \quad \quad \quad \quad \quad \quad = \frac { 1 } { 2 } \big [ L ( \bar { \theta } + \delta ) + L ( \bar { \theta } - \delta ) \big ] - \frac { 1 } { \eta } \| \delta \| ^ { 2 } + \frac { 2 \beta ^ { 2 } } { 1 - \beta } \gamma \cdot \delta } \\ & { \quad \quad \quad \quad \quad \quad + \frac { \eta \beta ^ { 2 } ( 1 + 2 \beta ) } { 2 ( 1 - \beta ) } \| \gamma \| ^ { 2 } - \frac { \eta \beta ^ { 2 } } { 2 ( 1 - \beta ) } \| \bar { m } \| ^ { 2 } . } \end{array}\tag{194}
$$

To lighten notation, we will drop the superscript and subscripts for the remainder of this appendix— writing B and Υ for the edge couplings of Nesterov momentum.

## C.3 Rod Flow

For the Nesterov midpoint rod flow, we have:

$$
\begin{array}{c} \boxed { \begin{array} { c c } { \boxed { \mathrm { d } } \left( \bar { z } \right)} \\ { \Delta } \end{array}  = \left( \begin{array} { c c } { K } & { 0 } \\ { 0 } & { - K } \end{array} \right) \nabla _ { ( \bar { z } , \Delta ) } \Upsilon } \end{array}\tag{195}
$$

Writing out the individual ODEs:

$$
\begin{array} { r l } & { \frac { \mathrm { d } \bar { \theta } } { \mathrm { d } t } = - \eta \bigl [ \beta ^ { 2 } \bar { m } + \bigl ( 1 - \beta ^ { 2 } \bigr ) \bar { g } \bigr ] , } \\ & { \frac { \mathrm { d } \bar { m } } { \mathrm { d } t } = ( 1 - \beta ) \bigl [ \bar { g } - \bar { m } \bigr ] , } \\ & { \frac { \mathrm { d } \delta } { \mathrm { d } t } = - 2 \delta + \eta \bigl [ \beta ^ { 2 } \gamma + \bigl ( 1 - \beta ^ { 2 } \bigr ) [ g ] \bigr ] , } \\ & { \frac { \mathrm { d } \gamma } { \mathrm { d } t } = - \bigl ( 1 + \beta \bigr ) \gamma - \bigl ( 1 - \beta \bigr ) [ g ] . } \end{array}\tag{196}
$$

For the endpoint rod flow, we have:

$$
| \begin{array} { c c } { \mathrm { d } } \\ { \mathrm { d } t } \end{array} ( { z } ) = ( \begin{array} { c c } { 0 } & { K } \\ { K } & { 0 } \end{array} ) \nabla _ { ( z , z ^ { \prime } ) } B\tag{197}
$$

Writing out the individual ODEs:

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } \theta } { \mathrm { d } t } = - \big ( \theta - \big [ \theta ^ { \prime } - \eta \big ( \beta ^ { 2 } m ^ { \prime } + ( 1 - \beta ^ { 2 } ) \nabla L ( \theta ^ { \prime } ) \big ) \big ] \big ) , } \\ { \displaystyle \frac { \mathrm { d } m } { \mathrm { d } t } = - \big ( m - \big [ \beta m ^ { \prime } + ( 1 - \beta ) \nabla L ( \theta ^ { \prime } ) \big ] \big ) , } \\ { \displaystyle \frac { \mathrm { d } \theta ^ { \prime } } { \mathrm { d } t } = - \big ( \theta ^ { \prime } - \big [ \theta - \eta \big ( \beta ^ { 2 } m + ( 1 - \beta ^ { 2 } ) \nabla L ( \theta ) \big ) \big ] \big ) , } \\ { \displaystyle \frac { \mathrm { d } m ^ { \prime } } { \mathrm { d } t } = - \big ( m ^ { \prime } - \big [ \beta m + ( 1 - \beta ) \nabla L ( \theta ) \big ] \big ) . } \end{array}\tag{198}
$$

Each endpoint regresses toward the Nesterov image of the other endpoint.

## C.4 Central Flow

We can also derive a central flow for Nesterov momentum. As with heavy ball, we first eliminate the momentum oscillation. Completing the square,

$$
\frac { 2 \beta ^ { 2 } } { 1 - \beta } \gamma \cdot \delta + \frac { \eta \beta ^ { 2 } ( 1 + 2 \beta ) } { 2 ( 1 - \beta ) } \| \gamma \| ^ { 2 } = \frac { \eta \beta ^ { 2 } ( 1 + 2 \beta ) } { 2 ( 1 - \beta ) } \left\| \gamma + \frac { 2 } { \eta ( 1 + 2 \beta ) } \delta \right\| ^ { 2 } - \frac { 2 \beta ^ { 2 } } { \eta ( 1 - \beta ) ( 1 + 2 \beta ) } \| \delta \| ^ { 2 } .\tag{199}
$$

Stationarity in $\gamma$ gives $\begin{array} { r } { \gamma = - \frac { 2 } { \eta ( 1 + 2 \beta ) } \delta } \end{array}$ . Eliminating $\gamma$ gives the reduced edge coupling

$$
\begin{array} { r } { \Upsilon ^ { \mathrm { r e d } } ( \bar { z } , \delta ) = \frac { 1 } { 2 } \big [ L ( \bar { \theta } + \delta ) + L ( \bar { \theta } - \delta ) \big ] - \frac { 1 + \beta } { \eta ( 1 - \beta ) ( 1 + 2 \beta ) } \| \delta \| ^ { 2 } - \frac { \eta \beta ^ { 2 } } { 2 ( 1 - \beta ) } \| \bar { m } \| ^ { 2 } . } \end{array}\tag{200}
$$

We can then approximate the loss about the midpoint $\bar { \theta }$ by its local quadratic approximation. Averaging over the endpoints $\bar { \theta } \pm \delta$ cancels the first-order terms, giving

$$
\begin{array} { r } { \frac 1 2 [ L ( \bar { \theta } + \delta ) + L ( \bar { \theta } - \delta ) ] \approx L ( \bar { \theta } ) + \frac 1 2 H ( \bar { \theta } ) [ \delta , \delta ] . } \end{array}\tag{201}
$$

Substituting into the reduced edge coupling and again replacing $\delta \otimes \delta$ with a general positive semidefinite matrix $\Sigma \succeq 0$ , we obtain the truncated reduced edge coupling

$$
\begin{array} { r } { \widetilde \Upsilon ^ { \mathrm { r e d } } ( \bar { z } , \Sigma ) = L ( \bar { \theta } ) - \frac { \eta \beta ^ { 2 } } { 2 ( 1 - \beta ) } \| \bar { m } \| ^ { 2 } - \frac { 1 } { 2 } \langle \Lambda _ { \mathrm { N A G } } ( \bar { \theta } ) , \Sigma \rangle , } \end{array}\tag{202}
$$

where

$$
\begin{array} { r } { \Lambda _ { \mathrm { N A G } } ( \bar { \theta } ) : = \frac { 2 ( 1 + \beta ) } { \eta ( 1 - \beta ) ( 1 + 2 \beta ) } I _ { d } - H ( \bar { \theta } ) . } \end{array}\tag{203}
$$

We can then maximize the truncated reduced edge coupling over $\Sigma \succeq 0$ . This gives the following static conditions:

$$
\Sigma \succeq 0 , \qquad \Lambda _ { \mathrm { N A G } } \succeq 0 , \qquad \langle \Lambda _ { \mathrm { N A G } } , \Sigma \rangle = 0 .\tag{204}
$$

Taking derivatives of the truncated reduced edge coupling with respect to the midpoint variables gives:

$$
\begin{array} { r } { \partial _ { \bar { \boldsymbol { \theta } } } \widetilde { \Upsilon } ^ { \mathrm { r e d } } = \nabla L ( \bar { \boldsymbol { \theta } } ) + \frac { 1 } { 2 } \nabla _ { \bar { \boldsymbol { \theta } } } \langle H ( \bar { \boldsymbol { \theta } } ) , \Sigma \rangle , } \end{array}\tag{205}
$$

$$
\begin{array} { r } { \partial _ { \bar { m } } \widetilde { \Upsilon } ^ { \mathrm { r e d } } = - \frac { \eta \beta ^ { 2 } } { 1 - \beta } \bar { m } . } \end{array}\tag{206}
$$

As with the Nesterov rod flow, we apply the matrix K to the gradient of the truncated reduced edge coupling to obtain the central flow ODE for Nesterov momentum:

$$
\frac { \mathrm { d } \bar { z } } { \mathrm { d } t } = K \partial _ { \bar { z } } \widetilde { \Upsilon } ^ { \mathrm { r e d } } ( \bar { z } , \Sigma ) .\tag{207}
$$

Writing the ODEs for <sup>¯</sup>θ and m¯ individually:

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } \bar { \theta } } { \mathrm { d } t } = - \eta \left[ \beta ^ { 2 } \bar { m } + ( 1 - \beta ^ { 2 } ) \Big ( \nabla L ( \bar { \theta } ) + \frac { 1 } { 2 } \nabla _ { \bar { \theta } } \langle H ( \bar { \theta } ) , \Sigma \rangle \Big ) \right] , } \\ { \displaystyle \frac { \mathrm { d } \bar { m } } { \mathrm { d } t } = ( 1 - \beta ) \left[ \nabla L ( \bar { \theta } ) + \frac { 1 } { 2 } \nabla _ { \bar { \theta } } \langle H ( \bar { \theta } ) , \Sigma \rangle - \bar { m } \right] . } \end{array}\tag{208}
$$

To determine Σ we invoke persistence: feasibility and complementarity must hold along the trajectory.

Let U be an orthonormal basis for the kernel of $\Lambda _ { \mathrm { N A G } }$ , and k the number of critical directions. Static complementarity requires:

$$
\Sigma = U X U ^ { \top } , \qquad X \succeq 0 .\tag{209}
$$

Computing the rate of change of the slack:

$$
\dot { \Lambda } _ { \mathrm { N A G } } = - \nabla _ { \bar { \theta } } H [ \dot { \bar { \theta } } ]\tag{210}
$$

Persistence of feasibility requires that the slack not decrease along the sharp directions:

$$
U ^ { \top } \dot { \Lambda } _ { \mathrm { N A G } } U \succeq 0 .\tag{211}
$$

Persistence of complementarity requires that there be no oscillations in directions whose slack is increasing:

$$
\langle U ^ { \top } \dot { \Lambda } _ { \mathrm { N A G } } U , X \rangle = 0 .\tag{212}
$$

These dynamic conditions, together with $X \succeq 0$ , define a semidefinite complementarity problem that determines X. Altogether, we have that the following conditions define the central flow for Nesterov momentum:

$$
\begin{array} { r l } & { \displaystyle \frac { \mathrm { d } \bar { z } } { \mathrm { d } t } = K \partial _ { \bar { z } } \widetilde { \Upsilon } ^ { \mathrm { r e d } } ( \bar { z } , \Sigma ) , \qquad \Sigma = U X U ^ { \top } , } \\ & { \quad X \succeq 0 , \qquad U ^ { \top } \dot { \Lambda } _ { \mathrm { N A G } } U \succeq 0 , \qquad \langle U ^ { \top } \dot { \Lambda } _ { \mathrm { N A G } } U , X \rangle = 0 . } \end{array}\tag{213}
$$

## D AdamW

This appendix derives the edge coupling, rod flow, and central flow for AdamW.

## D.1 Setup

AdamW [Loshchilov and Hutter, 2019] is a modification of Adam that decouples weight decay from the gradient-based update. Rather than adding an $\ell _ { 2 }$ penalty to the loss function—which would pass the decay through the momentum and the preconditioner—AdamW shrinks the weights directly. Omitting bias correction, the update equations are:

$$
\begin{array} { r l } & { m _ { t + 1 } = \beta _ { 1 } m _ { t } + ( 1 - \beta _ { 1 } ) \nabla L ( w _ { t } ) , } \\ & { \nu _ { t + 1 } = \beta _ { 2 } \nu _ { t } + ( 1 - \beta _ { 2 } ) \nabla L ( w _ { t } ) ^ { \odot 2 } , } \\ & { w _ { t + 1 } = \kappa w _ { t } - \eta P _ { t + 1 } ^ { - 1 } m _ { t + 1 } , } \end{array}\tag{214}
$$

where $\zeta \geq 0$ is the weight-decay coeficient and

$$
\kappa : = 1 - \eta \zeta\tag{215}
$$

is the factor by which the decay shrinks the weights at each step. Following the PyTorch implementation [Paszke et al., 2019], the decay is scaled by the step size, so it enters the update only through the product ηζ. At $\zeta = 0 ,$ , we have κ = 1—and AdamW reduces down to Adam.

## D.2 Edge of Stability

We derive the edge-of-stability condition for AdamW.

We will treat the preconditioner P as fixed. Consider a quadratic loss $\begin{array} { r } { L ( w ) = \frac { 1 } { 2 } w ^ { \top } H w } \end{array}$ . The AdamW updates are then

$$
m _ { t + 1 } = \beta _ { 1 } m _ { t } + ( 1 - \beta _ { 1 } ) H w _ { t } ,\tag{216}
$$

$$
w _ { t + 1 } = \kappa w _ { t } - \eta P ^ { - 1 } m _ { t + 1 } ,\tag{217}
$$

Since the update equations are linear in $( w , m )$ , we can express them as a single matrix equation:

$$
\binom { w _ { t + 1 } } { m _ { t + 1 } } = A \binom { w _ { t } } { m _ { t } } , \qquad A = \binom { \kappa I _ { d } - \eta ( 1 - \beta _ { 1 } ) P ^ { - 1 } H } { ( 1 - \beta _ { 1 } ) H } \frac { - \eta \beta _ { 1 } P ^ { - 1 } } { \beta _ { 1 } I _ { d } } \biggr ) .\tag{218}
$$

Let u be an eigenvector of $P ^ { - 1 } H$ with eigenvalue $\lambda ,$ so that $H u = \lambda P u$ . For each eigenvector $u ,$ applying A to the vectors $( u , 0 )$ and $( 0 , P u )$ gives

$$
A \binom { u } { 0 } = ( \kappa - \eta ( 1 - \beta _ { 1 } ) \lambda ) \binom { u } { 0 } + ( 1 - \beta _ { 1 } ) \lambda \binom { 0 } { P u } ,\tag{219}
$$

$$
A \left( { 0 \atop P u } \right) = - \eta \beta _ { 1 } \left( { u \atop 0 } \right) + \beta _ { 1 } \left( { 0 \atop P u } \right) .\tag{220}
$$

We have that A maps the span of these two vectors to itself. In the basis $\{ ( u , 0 ) , ( 0 , P u ) \}$ , it acts as the $2 \times 2$ matrix

$$
A _ { \lambda } = \left( \begin{array} { c c } { { \kappa - \eta ( 1 - \beta _ { 1 } ) \lambda } } & { { - \eta \beta _ { 1 } } } \\ { { ( 1 - \beta _ { 1 } ) \lambda } } & { { \beta _ { 1 } } } \end{array} \right) ,\tag{221}
$$

whose characteristic polynomial is

$$
p ( \alpha ) = \alpha ^ { 2 } - ( \kappa + \beta _ { 1 } - \eta ( 1 - \beta _ { 1 } ) \lambda ) \alpha + \kappa \beta _ { 1 } .\tag{222}
$$

The iterates are stable along u when both roots of $p$ lie strictly inside the unit circle—with instability setting in when a root crosses the unit circle. Consider the crossing at $\alpha = - 1$ , which corresponds to the iterates oscillating. The value $\alpha = - 1$ is a root of $p$ when

$$
p ( - 1 ) = ( 1 + \kappa ) ( 1 + \beta _ { 1 } ) - \eta ( 1 - \beta _ { 1 } ) \lambda = 0 .\tag{223}
$$

Rearranging to isolate $\lambda ,$ we obtain the edge of stability condition for AdamW:

$$
\boxed { \lambda _ { \mathrm { m a x } } ( P ^ { - 1 / 2 } H P ^ { - 1 / 2 } ) = \Big ( 1 - \frac { \eta \zeta } { 2 } \Big ) \frac { 2 ( 1 + \beta _ { 1 } ) } { \eta ( 1 - \beta _ { 1 } ) } . }\tag{224}
$$

## D.3 Edge Coupling

We would like to derive an edge coupling for AdamW. We will again treat the preconditioner $P$ as fixed. Written as diference equations, the update equations of AdamW are

$$
\begin{array} { r l } & { w _ { t + 1 } - w _ { t } = - \eta \zeta w _ { t } - \eta \beta _ { 1 } P ^ { - 1 } m _ { t } - \eta ( 1 - \beta _ { 1 } ) P ^ { - 1 } \nabla L ( w _ { t } ) , } \\ & { m _ { t + 1 } - m _ { t } = ( 1 - \beta _ { 1 } ) \nabla L ( w _ { t } ) - ( 1 - \beta _ { 1 } ) m _ { t } . } \end{array}\tag{225}
$$

The update can be split into two terms: a term that is linear in z and a term that is linear in the gradient of the loss function.

$$
z _ { t + 1 } - z _ { t } = ( 1 - \beta _ { 1 } ) \left( \begin{array} { c } { - \eta P ^ { - 1 } } \\ { I _ { d } } \end{array} \right) \nabla L ( w _ { t } ) + R z _ { t } , \qquad R = \left( \begin{array} { c c } { - \eta \zeta I _ { d } } & { - \eta \beta _ { 1 } P ^ { - 1 } } \\ { 0 } & { - ( 1 - \beta _ { 1 } ) I _ { d } } \end{array} \right) .\tag{226}
$$

The weight decay enters only through the $( w , w )$ block of $R .$

We will assume that the generating function has the form $\begin{array} { r } { \mathcal { L } ( z ) = L ( w ) + \frac { 1 } { 2 } z ^ { \top } Q z } \end{array}$ . We seek a symmetric structure matrix $K$ and a symmetric matrix $Q$ such that $z _ { t + 1 } - z _ { t } = K \nabla \mathcal { L } ( z _ { t } )$ . Taking the derivative of ${ \mathcal { L } } ,$ we have that:

$$
K \nabla \mathcal { L } ( z ) = K \left( \begin{array} { c } { I _ { d } } \\ { 0 } \end{array} \right) \nabla L ( w ) + K Q z .\tag{227}
$$

Setting this equal to the update yields:

$$
K \binom { I _ { d } } { 0 } \nabla L ( w ) + K Q z = ( 1 - \beta _ { 1 } ) \binom { - \eta P ^ { - 1 } } { I _ { d } } \nabla L ( w ) + R z\tag{228}
$$

Since the above equation must hold for arbitrary loss functions, the gradient terms and the linear terms must match separately. Matching the gradient terms, we have that

$$
K \binom { I _ { d } } { 0 } = ( 1 - \beta _ { 1 } ) \binom { - \eta P ^ { - 1 } } { I _ { d } } ,\tag{229}
$$

which fixes the first block column of $K { - } \mathrm { i t }$ must equal the coeficient of $\nabla L ( w )$ in the update. And because K is symmetric, this also fixes the first block row as well. The only remaining unknown of K is the $( m , m )$ block:

$$
K = \left( \begin{array} { c c } { { - \eta ( 1 - \beta _ { 1 } ) P ^ { - 1 } } } & { { ( 1 - \beta _ { 1 } ) I _ { d } } } \\ { { ( 1 - \beta _ { 1 } ) I _ { d } } } & { { K _ { m m } } } \end{array} \right) .\tag{230}
$$

Matching the linear terms, we have that

$$
K Q = R .\tag{231}
$$

Since $Q$ is symmetric, its $( m , w )$ block is the transpose of its $( w , m )$ block, so there are four unknowns: $K _ { m m } , Q _ { w w } , Q _ { w m }$ , and $Q _ { m m }$ . Writing out $K Q = R$ block by block, we obtain four equations—one for each block of R:

$$
- \eta ( 1 - \beta _ { 1 } ) P ^ { - 1 } Q _ { w w } + ( 1 - \beta _ { 1 } ) Q _ { w m } ^ { \top } = - \eta \zeta I _ { d } ,\tag{232}
$$

$$
- \eta ( 1 - \beta _ { 1 } ) P ^ { - 1 } Q _ { w m } + ( 1 - \beta _ { 1 } ) Q _ { m m } = - \eta \beta _ { 1 } P ^ { - 1 } ,\tag{233}
$$

$$
( 1 - \beta _ { 1 } ) Q _ { w w } + K _ { m m } Q _ { w m } ^ { \top } = 0 ,\tag{234}
$$

$$
( 1 - \beta _ { 1 } ) Q _ { w m } + K _ { m m } Q _ { m m } = - ( 1 - \beta _ { 1 } ) I _ { d } .\tag{235}
$$

We can solve these four equations for the four unknown sub-blocks. In order for the above system of equations to have a solution, it is crucial that every known matrix in the equations above commutes with P. Otherwise, the number of independent scalar equations would exceed the efective number of scalar unknowns. It is also crucial that $P$ be symmetric: the solutions for the diagonal blocks $K _ { m m } , Q _ { w w }$ , and $Q _ { m m }$ are expressed in terms of $P _ { \mathrm { : } }$ , and these blocks must themselves be symmetric. Solving the third equation for $Q _ { w w } .$ we obtain:

$$
\begin{array} { r } { Q _ { w w } = - \frac { 1 } { 1 - \beta _ { 1 } } K _ { m m } Q _ { w m } ^ { \top } . } \end{array}\tag{236}
$$

We can then substitute into the first equation to obtain an equation in only $Q _ { w m }$ and $K _ { m m }$

$$
\begin{array} { r } { - \eta ( 1 - \beta _ { 1 } ) P ^ { - 1 } ( - \frac { 1 } { 1 - \beta _ { 1 } } K _ { m m } Q _ { w m } ^ { \top } ) + ( 1 - \beta _ { 1 } ) Q _ { w m } ^ { \top } = - \eta \zeta I _ { d } . } \end{array}\tag{237}
$$

Similarly, using the second equation, we can solve for $Q _ { m m }$

$$
\begin{array} { r } { Q _ { m m } = \eta P ^ { - 1 } Q _ { w m } - \frac { \eta \beta _ { 1 } } { 1 - \beta _ { 1 } } P ^ { - 1 } . } \end{array}\tag{238}
$$

We can then substitute this into the fourth equation to obtain:

$$
\begin{array} { r } { ( 1 - \beta _ { 1 } ) Q _ { w m } + K _ { m m } ( \eta P ^ { - 1 } Q _ { w m } - \frac { \eta \beta _ { 1 } } { 1 - \beta _ { 1 } } P ^ { - 1 } ) = - ( 1 - \beta _ { 1 } ) I _ { d } . } \end{array}\tag{239}
$$

As we are assuming that the diagonal sub-blocks can be expressed in terms of $P _ { \mathrm { : } }$ , they must commute with P. We then have that $P ^ { - 1 } Q _ { w w }$ is symmetric—from which it follows that $Q _ { w m }$ is symmetric as well. Rearranging, we have the following system of equations in $Q _ { w m }$ and $K _ { m m }$

$$
\begin{array} { r l } & { \left[ ( 1 - \beta _ { 1 } ) I _ { d } + \eta P ^ { - 1 } K _ { m m } \right] Q _ { w m } = - \eta \zeta I _ { d } , } \\ & { \left[ ( 1 - \beta _ { 1 } ) I _ { d } + \eta P ^ { - 1 } K _ { m m } \right] Q _ { w m } = \frac { \eta \beta _ { 1 } } { 1 - \beta _ { 1 } } P ^ { - 1 } K _ { m m } - ( 1 - \beta _ { 1 } ) I _ { d } . } \end{array}\tag{240}
$$

The left-hand sides are the same. Setting the right-hand sides equal gives us $K _ { m m } \colon$

$$
\begin{array} { r } { - \eta \zeta I _ { d } = \frac { \eta \beta _ { 1 } } { 1 - \beta _ { 1 } } P ^ { - 1 } K _ { m m } - ( 1 - \beta _ { 1 } ) I _ { d } \implies K _ { m m } = \frac { ( 1 - \beta _ { 1 } ) ( \kappa - \beta _ { 1 } ) } { \eta \beta _ { 1 } } P . } \end{array}\tag{241}
$$

We then have for the structure matrix:

$$
K _ { \nu } = \left( \begin{array} { c c } { { - \eta ( 1 - \beta _ { 1 } ) P ( \nu ) ^ { - 1 } } } & { { ( 1 - \beta _ { 1 } ) I _ { d } } } \\ { { ( 1 - \beta _ { 1 } ) I _ { d } } } & { { \frac { ( 1 - \beta _ { 1 } ) ( \kappa - \beta _ { 1 } ) } { \eta \beta _ { 1 } } P ( \nu ) \right) . } } \end{array}\tag{242}
$$

The inverse of the structure matrix is

$$
K _ { \nu } ^ { - 1 } = \frac { 1 } { \kappa ( 1 - \beta _ { 1 } ) } \left( \begin{array} { c c } { - \frac { \kappa - \beta _ { 1 } } { \eta } P } & { \beta _ { 1 } I _ { d } } \\ { \beta _ { 1 } I _ { d } } & { \eta \beta _ { 1 } P ^ { - 1 } } \end{array} \right) .\tag{243}
$$

As we have found $K _ { m m } ,$ by substituting into the above equations, one can easily solve for Q:

$$
Q = \frac { 1 } { \kappa ( 1 - \beta _ { 1 } ) } \left( \begin{array} { c c } { { \zeta ( \kappa - \beta _ { 1 } ) P } } & { { - \eta \beta _ { 1 } \zeta I _ { d } } } \\ { { - \eta \beta _ { 1 } \zeta I _ { d } } } & { { - \eta \beta _ { 1 } P ^ { - 1 } } } \end{array} \right) .\tag{244}
$$

Substituting this $Q$ into our ansatz gives the generating function for AdamW:

$$
\begin{array} { r } { \mathcal { L } ( w , m ) = L ( w ) - \frac { \eta \beta _ { 1 } } { 2 ( 1 - \beta _ { 1 } ) } m ^ { \top } P ^ { - 1 } m + \frac { \zeta } { 2 ( 1 - \beta _ { 1 } ) } \Big [ \| w \| _ { P } ^ { 2 } - \frac { \beta _ { 1 } } { \kappa } \| w + \eta P ^ { - 1 } m \| _ { P } ^ { 2 } \Big ] , } \end{array}\tag{245}
$$

where $\| v \| _ { P } ^ { 2 } : = v ^ { \top } P v$ . Compared with Adam, the new terms in the AdamW generating function are all proportional to ζ, so they vanish when $\zeta = 0 - \mathrm { a s }$ expected.

As we have found L and K, we can obtain the edge coupling for AdamW:

$$
\begin{array} { r l } & { \mathcal { B } _ { \eta , \beta _ { 1 } , \nu } ^ { \mathrm { A d a m W } } ( z , z ^ { \prime } ) = \mathcal { B } _ { \eta , \beta _ { 1 } , \nu } ( z , z ^ { \prime } ) } \\ & { \qquad + \frac { \zeta } { 2 \kappa ( 1 - \beta _ { 1 } ) } \Big [ \kappa ( \| w \| _ { P } ^ { 2 } + \| w ^ { \prime } \| _ { P } ^ { 2 } ) - 2 \beta _ { 1 } ( w + \eta P ^ { - 1 } m ) ^ { \top } P ( w ^ { \prime } + \eta P ^ { - 1 } m ^ { \prime } ) \Big ] . } \end{array}\tag{246}
$$

where $B _ { \eta , \beta _ { 1 } , \nu } ( z , z ^ { \prime } )$ denotes the edge coupling for Adam.

For the centered edge coupling, we have:

$$
\begin{array} { r l } & { \Upsilon _ { \eta , \beta _ { 1 } , \nu } ^ { \mathrm { A d a m W } } ( \bar { z } , \Delta ) : = \frac { 1 } { 2 } \mathcal { B } _ { \eta , \beta _ { 1 } , \nu } ^ { \mathrm { A d a m W } } ( \bar { z } + \Delta , \bar { z } - \Delta ) } \\ & { \qquad = \Upsilon _ { \eta , \beta _ { 1 } , \nu } ( \bar { z } , \Delta ) + \frac { \zeta } { 2 \kappa ( 1 - \beta _ { 1 } ) } \Big [ \kappa \| \bar { w } \| _ { P } ^ { 2 } - \beta _ { 1 } \| \bar { w } + \eta P ^ { - 1 } \bar { m } \| _ { P } ^ { 2 } \Big ] } \\ & { \qquad + \frac { \zeta } { 2 \kappa ( 1 - \beta _ { 1 } ) } \Big [ \kappa \| \delta \| _ { P } ^ { 2 } + \beta _ { 1 } \| \delta + \eta P ^ { - 1 } \gamma \| _ { P } ^ { 2 } \Big ] . } \end{array}\tag{247}
$$

As we did in Appendix C, we will drop the superscript and subscripts for the remainder of this appendix—writing B and Υ for the edge couplings of AdamW.

## D.4 Rod Flow

As with Adam, we treat the second moment ν¯ as a smooth auxiliary variable. Applying the structure matrix to the gradient of the centered edge coupling, we obtain the rod flow for AdamW:

$$
\begin{array} { r l } & { \left[ \frac { \mathrm { d } } { \mathrm { d } t } \left( \frac { \bar { z } } { \Delta } \right) = \left( \begin{array} { c c } { K _ { \bar { \nu } } } & { 0 } \\ { 0 } & { - K _ { \bar { \nu } } } \end{array} \right) \nabla _ { ( \bar { z } , \Delta ) } \Upsilon , \right. } \\ & { \left. \frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \big [ \bar { g } ^ { \odot 2 } + \big [ [ g ] ^ { \odot 2 } - \bar { \nu } \big ] . \right. } \end{array}\tag{248}
$$

Writing out the individual ODEs:

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } \overline { { w } } } { \mathrm { d } t } = - \eta \zeta \overline { { w } } - \eta P ^ { - 1 } \big [ \beta _ { 1 } \overline { { m } } + ( 1 - \beta _ { 1 } ) \overline { { g } } \big ] , } \\ { \displaystyle \frac { \mathrm { d } \overline { { n } } } { \mathrm { d } t } = ( 1 - \beta _ { 1 } ) \big [ \bar { g } - \bar { m } \big ] , } \\ { \displaystyle \frac { \mathrm { d } \overline { { \nu } } } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \big [ \bar { g } ^ { \odot 2 } + [ g ] ^ { \odot 2 } - \bar { \nu } \big ] , } \\ { \displaystyle \frac { \mathrm { d } \overline { { \delta } } } { \mathrm { d } t } = - ( 1 + \kappa ) \delta + \eta P ^ { - 1 } \big [ \beta _ { 1 } \gamma + ( 1 - \beta _ { 1 } ) [ g ] \big ] , } \\ { \displaystyle \frac { \mathrm { d } \overline { { \gamma } } } { \mathrm { d } t } = - ( 1 + \beta _ { 1 } ) \gamma - ( 1 - \beta _ { 1 } ) [ g ] . } \end{array}\tag{249}
$$

For the endpoint rod flow, we have:

$$
\begin{array} { r l r } {  { ( \frac { \mathrm { d } } { \mathrm { d } t } ( \begin{array} { l } { z } \\ { z ^ { \prime } } \end{array} ) = ( K _ { \bar { \nu } } \begin{array} { c c } { K _ { \bar { \nu } } } \\ { K _ { \bar { \nu } } } & { 0 } \end{array} ) \nabla _ { ( z , z ^ { \prime } ) } \mathcal { B } , } } \\ & { } & { \frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \big [ \frac { 1 } { 2 } \big ( \nabla L ( w ) ^ { \odot 2 } + \nabla L ( w ^ { \prime } ) ^ { \odot 2 } \big ) - \bar { \nu } \big ] . } \end{array}\tag{250}
$$

Writing out the individual ODEs:

$$
\begin{array} { r l } & { \frac { \mathrm { d } w } { \mathrm { d } t } = - \big ( w - \big [ \kappa w ^ { \prime } - \eta P ^ { - 1 } \big ( \beta _ { 1 } m ^ { \prime } + \big ( 1 - \beta _ { 1 } ) \nabla L ( w ^ { \prime } ) \big ) \big ] \big ) . } \\ & { \frac { \mathrm { d } m } { \mathrm { d } t } = - \big ( m - \big [ \beta _ { 1 } m ^ { \prime } + \big ( 1 - \beta _ { 1 } \big ) \nabla L ( w ^ { \prime } ) \big ] \big ) , } \\ & { \frac { \mathrm { d } \hat { \nu } } { \mathrm { d } t } = \big ( 1 - \beta _ { 2 } \big ) \big [ \frac { 1 } { 2 } \big ( \nabla L ( w ) ^ { \circledast 2 } + \nabla L ( w ^ { \prime } ) ^ { \circledast 2 } \big ) - \hat { \nu } \big ] , } \\ & { \frac { \mathrm { d } w ^ { \prime } } { \mathrm { d } t } = - \big ( w ^ { \prime } - \big [ \kappa w - \eta P ^ { - 1 } \big ( \beta _ { 1 } m + \big ( 1 - \beta _ { 1 } \big ) \nabla L ( w ) \big ) \big ] \big ) , } \\ & { \frac { \mathrm { d } m ^ { \prime } } { \mathrm { d } t } = - \big ( m ^ { \prime } - \big [ \beta _ { 1 } m + \big ( 1 - \beta _ { 1 } \big ) \nabla L ( w ) \big ] \big ) . } \end{array}\tag{251}
$$

Each endpoint regresses toward the AdamW image of the other endpoint.

## D.5 Central Flow

We can also derive a central flow for AdamW. As with Adam, we first eliminate the momentum oscillation. For fixed δ, the terms involving $\gamma$ can be written as:

$$
\begin{array} { r l } & { \frac { 2 \beta _ { 1 } } { 1 - \beta _ { 1 } } \gamma \cdot \delta + \frac { \eta \beta _ { 1 } } { 2 ( 1 - \beta _ { 1 } ) } \gamma ^ { \top } P ^ { - 1 } \gamma + \frac { \zeta \beta _ { 1 } } { 2 \kappa ( 1 - \beta _ { 1 } ) } \big ( 2 \eta \gamma \cdot \delta + \eta ^ { 2 } \gamma ^ { \top } P ^ { - 1 } \gamma \big ) } \\ & { \qquad = \frac { ( 1 + \kappa ) \beta _ { 1 } } { \kappa ( 1 - \beta _ { 1 } ) } \gamma \cdot \delta + \frac { \eta \beta _ { 1 } } { 2 \kappa ( 1 - \beta _ { 1 } ) } \gamma ^ { \top } P ^ { - 1 } \gamma } \\ & { \qquad = \frac { \eta \beta _ { 1 } } { 2 \kappa ( 1 - \beta _ { 1 } ) } \left( \gamma + \frac { 1 + \kappa } { \eta } P \delta \right) ^ { \top } P ^ { - 1 } \left( \gamma + \frac { 1 + \kappa } { \eta } P \delta \right) - \frac { ( 1 + \kappa ) ^ { 2 } \beta _ { 1 } } { 2 \eta \kappa ( 1 - \beta _ { 1 } ) } \delta ^ { \top } P \delta . } \end{array}\tag{252}
$$

One can see that the minimum is attained at $\begin{array} { r } { \gamma = - \frac { 1 + \kappa } { \eta } P \delta } \end{array}$ . Eliminating $\gamma$ from the centered edge coupling gives the reduced edge coupling:

$$
\begin{array} { r l } & { \Upsilon ^ { \mathrm { r e d } } ( \bar { z } , \delta ) = \frac { 1 } { 2 } \big [ L ( \bar { w } + \delta ) + L ( \bar { w } - \delta ) \big ] - \frac { \eta \beta _ { 1 } } { 2 ( 1 - \beta _ { 1 } ) } \bar { m } ^ { \top } P ^ { - 1 } \bar { m } } \\ & { \qquad + \frac { \zeta } { 2 ( 1 - \beta _ { 1 } ) } \Big [ \| \bar { w } \| _ { P } ^ { 2 } - \frac { \beta _ { 1 } } { \kappa } \| \bar { w } + \eta P ^ { - 1 } \bar { m } \| _ { P } ^ { 2 } \Big ] - \frac { ( 1 + \kappa ) ( 1 + \beta _ { 1 } ) } { 2 \eta ( 1 - \beta _ { 1 } ) } \delta ^ { \top } P \delta . } \end{array}\tag{253}
$$

We again replace the loss with its local quadratic approximation about $\bar { w }$ and replace $\delta \otimes \delta$ with a generic positive semidefinite matrix Σ. The truncated reduced edge coupling is

$$
\begin{array} { r } { \widetilde { \Upsilon } ^ { \mathrm { r e d } } ( \bar { z } , \Sigma ) = \mathcal { L } ( \bar { z } ) - \frac { 1 } { 2 } \langle \Lambda _ { \mathrm { A d a m W } } , \Sigma \rangle , } \end{array}\tag{254}
$$

where $\mathcal { L }$ is the generating function for AdamW and the AdamW slack is given as:

$$
\begin{array} { r } { \Lambda _ { \mathrm { A d a m W } } ( \bar { w } , \bar { \nu } ) : = \frac { ( 1 + \kappa ) ( 1 + \beta _ { 1 } ) } { \eta ( 1 - \beta _ { 1 } ) } P ( \bar { \nu } ) - H ( \bar { w } ) . } \end{array}\tag{255}
$$

Compared with Adam, the weight decay lowers the threshold by the factor $1 - \eta \zeta / 2$ , in agreement with the edge of stability condition. Maximizing over $\Sigma \succeq 0$ gives the static conditions

$$
\Sigma \succeq \succeq 0 , \qquad \Lambda _ { \mathrm { A d a m W } } \succeq 0 , \qquad \langle \Lambda _ { \mathrm { A d a m W } } , \Sigma \rangle = 0 .\tag{256}
$$

Let U be an orthonormal basis for ker $\Lambda _ { \mathrm { A d a m W } }$ . We can express the covariance matrix as $\Sigma = U X U ^ { \top }$ with $X \succeq 0$

We then obtain the central flow ODEs for AdamW by taking the gradient of the truncated edge coupling with respect to the midpoint variables, treating the second moment as an auxiliary variable.

$$
\frac { \mathrm { d } \bar { w } } { \mathrm { d } t } = - \eta \zeta \bar { w } - \eta P ^ { - 1 } \left[ \beta _ { 1 } \bar { m } + ( 1 - \beta _ { 1 } ) \Big ( \nabla L ( \bar { w } ) + { \textstyle \frac { 1 } { 2 } } \nabla _ { \bar { w } } \langle H ( \bar { w } ) , \Sigma \rangle \Big ) \right] ,
$$

$$
\frac { \mathrm { d } \bar { m } } { \mathrm { d } t } = ( 1 - \beta _ { 1 } ) \left[ \nabla L ( \bar { w } ) + \textstyle { \frac { 1 } { 2 } } \nabla _ { \bar { w } } \langle H ( \bar { w } ) , \Sigma \rangle - \bar { m } \right] ,
$$

$$
\frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \left[ ( \nabla L ( \bar { w } ) ) ^ { \odot 2 } + \mathrm { d i a g } \left( H ( \bar { w } ) \Sigma H ( \bar { w } ) \right) - \bar { \nu } \right] .\tag{257}
$$

As with Adam, the second-moment equation uses the approximation

$$
\bar { g } ^ { \odot 2 } + [ [ g ] ^ { \odot 2 } \approx ( \nabla L ( \bar { w } ) ) ^ { \odot 2 } + ( H ( \bar { w } ) \delta ) ^ { \odot 2 } ,\tag{258}
$$

which follows from the local quadratic approximation of the loss about w¯ and by replacing $\delta \otimes \delta$ with Σ.

We determine the covariance matrix by imposing persistence of feasibility and complementarity. The time derivative of the slack is:

$$
\begin{array} { r } { \dot { \Lambda } _ { \mathrm { A d a m W } } = \frac { ( 1 + \kappa ) ( 1 + \beta _ { 1 } ) } { \eta ( 1 - \beta _ { 1 } ) } \nabla _ { \bar { \nu } } P ( \bar { \nu } ) [ \dot { \bar { \nu } } ] - \nabla _ { \bar { w } } H ( \bar { w } ) [ \dot { \bar { w } } ] . } \end{array}\tag{259}
$$

At fixed $( \bar { w } , \bar { \nu } )$ , both the $\dot { \bar { w } }$ and $\dot { \bar { \nu } }$ terms are afine in Σ. For our persistence conditions, we have:

$$
\begin{array} { r } { \boldsymbol { U } ^ { \top } \dot { \boldsymbol { \Lambda } } _ { \mathrm { A d a m W } } \boldsymbol { U } \succeq 0 , \qquad \left. \boldsymbol { U } ^ { \top } \dot { \boldsymbol { \Lambda } } _ { \mathrm { A d a m W } } \boldsymbol { U } , \boldsymbol { X } \right. = 0 . } \end{array}\tag{260}
$$

The above persistence conditions in combination with $X \succeq 0$ define a semidefinite complementarity problem for $X$ . Altogether, we have that the following conditions define the central flow for AdamW:

$$
\Bigg | \frac { \mathrm { d } \overline { { z } } } { \mathrm { d } t } = K _ { \bar { \nu } } \partial _ { \bar { z } } \widetilde { \Upsilon } ^ { \mathrm { r e d } } ( \bar { z } , \Sigma ) , \qquad \Sigma = U X U ^ { \top } ,
$$

$$
\frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \Big [ ( \nabla L ( \bar { w } ) ) ^ { \odot 2 } + \mathrm { d i a g } \big ( H ( \bar { w } ) \Sigma H ( \bar { w } ) \big ) - \bar { \nu } \Big ] ,\tag{261}
$$

$$
X \succeq 0 , \qquad U ^ { \top } \dot { \Lambda } _ { \mathrm { A d a m W } } U \succeq 0 , \qquad \langle U ^ { \top } \dot { \Lambda } _ { \mathrm { A d a m W } } U , X \rangle = 0 .
$$

## E Rod Flow

This appendix collects the details of rod flow that the main text defers.

## E.1 Extent Matrix Formulation

The edge-coupling-derived rod flow of Section 4 evolves the vector $\delta ,$ whereas the original rod flow [Regis and Chewi, 2026b] evolves the extent matrix Σ. To more easily compare the two formulations, it helps to express the edge-coupling-derived rod flow in terms of $\Sigma$

Let $\Sigma = \delta \otimes \delta$ . Note that

$$
\operatorname { t r } \Sigma = \| \delta \| ^ { 2 } .\tag{262}
$$

The vector $\delta$ can be identified with the principal eigenvector of Σ scaled by the square root of its principal eigenvalue. Since the centered edge coupling is even in $\delta ,$ we can write the centered edge coupling as a function of Σ:

$$
\begin{array} { r } { \Psi ( \bar { w } , \Sigma ) = \frac { 1 } { 2 } [ L ( \bar { w } + \delta ) + L ( \bar { w } - \delta ) ] - \frac { 1 } { \eta } \mathrm { t r } \Sigma . } \end{array}\tag{263}
$$

We now want to derive the dynamics of Σ. As with $\delta ,$ Σ must perform gradient ascent. However, when taking the derivative with respect to $\Sigma ,$ we need to be careful about which metric we use. While the Euclidean metric was appropriate for $\bar { w }$ and $\delta ,$ it is not the right choice for Σ. Instead, the relevant metric is the one induced on Σ by the map $\delta \mapsto \delta \otimes \delta$ . This turns out to be the Bures–Wasserstein metric [Bhatia et al., 2019], the same metric that gives the optimal transport distance between centered Gaussian distributions.

Consider a function $F ( \Sigma )$ , and let $\nabla _ { \Sigma } F$ denote the Euclidean derivative of $F .$ . The Bures– Wasserstein gradient of $F$ is obtained from the Euclidean derivative $\nabla _ { \Sigma } F$ via

$$
\operatorname { g r a d } _ { \mathrm { B W } } F ( \Sigma ) = 2 \bigl ( \nabla _ { \Sigma } F \Sigma + \Sigma \nabla _ { \Sigma } F \bigr ) .\tag{264}
$$

Consider the Euclidean derivative $\nabla _ { \Sigma } \Psi$ . By using the chain rule, we can relate the Euclidean derivatives of $\Psi$ with respect to $\delta$ and Σ:

$$
\nabla _ { \delta } \Psi = 2 ( \nabla _ { \Sigma } \Psi ) \delta\tag{265}
$$

We can then deduce:

$$
\begin{array} { r l } & { \mathrm { g r a d } _ { \mathrm { B W } } \Psi = 2 \bigl ( \nabla _ { \Sigma } \Psi \Sigma + \Sigma \nabla _ { \Sigma } \Psi \bigr ) } \\ & { \quad \quad \quad \quad \quad = 2 \bigl ( \nabla _ { \Sigma } \Psi \left[ \delta \otimes \delta \right] + \left[ \delta \otimes \delta \right] \nabla _ { \Sigma } \Psi \bigr ) } \\ & { \quad \quad \quad = \nabla _ { \delta } \Psi \otimes \delta + \delta \otimes \nabla _ { \delta } \Psi } \\ & { \quad \quad \quad = ( [ g ] ] - \frac { 2 } { \eta } \delta ) \otimes \delta + \delta \otimes ( [ g ] - \frac { 2 } { \eta } \delta ) } \\ & { \quad \quad \quad = [ g ] \otimes \delta + \delta \otimes [ g ] ] - \frac { 4 } { \eta } \Sigma } \end{array}\tag{266}
$$

Setting $\begin{array} { r } { \frac { \mathrm { d } \Sigma } { \mathrm { d } t } = \eta \mathrm { g r a d } _ { \mathrm { B W } } \Psi } \end{array}$ , we then obtain the following rod flow ODE:

$$
\frac { \mathrm { d } \Sigma } { \mathrm { d } t } = \eta ( \mathbb { I } g \mathbb { I } \otimes \delta + \delta \otimes \mathbb { I } g \mathbb { I } ) - 4 \Sigma .\tag{267}
$$

The right-hand side can be expressed in terms of Σ alone. By the fundamental theorem of calculus, the diference between the endpoint gradients can be expressed as integral with respect to the Hessian:

$$
\nabla L ( \bar { w } + \delta ) - \nabla L ( \bar { w } - \delta ) = \int _ { - 1 } ^ { 1 } \nabla ^ { 2 } L ( \bar { w } + s \delta ) \delta d s\tag{268}
$$

Let $\widetilde { H }$ denote the rod-averaged Hessian.

$$
\widetilde H ( \bar { w } , \delta ) =  { { \frac { 1 } { 2 } } } \int _ { - 1 } ^ { 1 } \nabla ^ { 2 } L ( \bar { w } + s \delta ) \mathrm { d } s\tag{269}
$$

We then have that the gradient half-diference can be expressed as:

$$
[ [ g ] ] = { \widetilde { H } } \delta\tag{270}
$$

Substituting, we then have that the rod flow ODE for Σ can be expressed as:

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } \Sigma } { \mathrm { d } t } = \eta \bigl ( \widetilde { H } \Sigma + \Sigma \widetilde { H } \bigr ) - 4 \Sigma } \\ { = \eta \Bigl [ \bigl ( \widetilde { H } - \frac { 2 } { \eta } I _ { d } \bigr ) \Sigma + \Sigma \bigl ( \widetilde { H } - \frac { 2 } { \eta } I _ { d } \bigr ) \Bigr ] . } \end{array}\tag{271}
$$

## E.2 The Noise Floor

In the stable regime, the sharp component of the gradient should decay exponentially under gradient descent. In practice, however, rounding noise prevents this decay: the sharp component of the discrete iterates instead plateaus at a small but nonzero value.

To motivate why that would be the case, consider the following model of the rounding error. Every time the optimizer stores its state, each stored component $z _ { i }$ is rounded to a nearby representable number. Let u represent the unit round-of $\left( \mathsf { u } = 2 ^ { - 2 4 } \right.$ for float32). We can model the rounding error as uniform on the interval $[ - \mathbf { u } | z _ { i } | , \mathbf { u } | z _ { i } ] ] .$ —and independent across coordinates and iterations. As the round-of noise is uniformly-distributed, it has a variance of $\mathsf { u } ^ { 2 } z _ { i } ^ { 2 } / 3$

Let $\hat { u } _ { j }$ denote an eigenvector of the Hessian. Projected onto $\hat { u } _ { j }$ , the rounding of the stored position and momentum injects per-step variances of

$$
V _ { j } = \frac { \mathrm { u } ^ { 2 } } { 3 } \sum _ { i } ( \hat { u } _ { j } ) _ { i } ^ { 2 } \bar { w } _ { i } ^ { 2 } , \qquad V _ { j } ^ { \mathrm { m o m } } = \frac { \mathrm { u } ^ { 2 } } { 3 } \sum _ { i } ( \hat { u } _ { j } ) _ { i } ^ { 2 } \bar { m } _ { i } ^ { 2 }\tag{272}
$$

into that direction.

![](images/f49f7ec8341c6287a6ec39f2c4b6394b1521f0e0a6b2c585e82420873878b844.jpg)  
Figure 12: Round-Of Sets the Floor. Gradient descent on an MLP at $\eta = 0 . 0 5$ (left) and $\eta = 0 . 0 7 5$ (right), run in single and double precision from the same initialization. Shown is the sharp component $| \langle a _ { t } , u _ { 1 } \rangle |$ of the oscillation amplitude. Before the crossing, the sharp component sits at the floor predicted by the rounding model, which is about eight orders of magnitude lower in double precision. As a result, the double-precision runs self-stabilize later than the single-precision runs, though eventually they settle on the same plateau.

Why must rounding noise be the culprit? Could the plateau of the sharp component not arise from the dynamics of gradient descent itself?

Two observations single out round-of. First, if the dynamics caused the plateau of the sharp component, we would expect the plateau to scale with the step size. Instead, the measured floor is the same for gradient descent across step sizes spanning several orders of magnitude. Second, repeating the gradient descent run in double precision from the same initialization lowers the floor by several orders of magnitude, to the level predicted for float64 (Figure 12).

The model above predicts the noise injected at each step. However, to predict the stationary covariance, we must also account for the dynamics of the optimizer.

Let ${ z } _ { t } = ( w _ { t } , m _ { t } )$ denote the phase-space point of the optimizer, and let $\xi _ { t }$ be the accumulated deviation from the trajectory that the optimizer would have followed if there were no rounding error.

Since the deviation is of the order of the rounding error, a linearization sufices: we can replace the loss by its local quadratic approximation. The update map then becomes an afine map of z. Let A denote the matrix of its linear part (Table 2).

Let $C _ { t } = \mathbb { E } [ \xi _ { t } \xi _ { t } ^ { \top } ]$ be the covariance of the deviation, taken over the rounding errors. Each step multiplies the deviation by A and adds a fresh rounding error that is independent of the deviation:

$$
\begin{array} { r } { C _ { t + 1 } = A C _ { t } A ^ { \top } + \frac { { \mathbf u } ^ { 2 } } { 3 } \mathrm { d i a g } ( \bar { z } ^ { \odot 2 } ) . } \end{array}\tag{273}
$$

Rod flow’s noise floor gives its extent the same stationary variance that round-of error induces in the discrete iterates.

Assume that the Hessian is fixed. Let $C ^ { * }$ denote the stationary covariance of the deviation. Along each eigencomponent of H, it satisfies the following equation:

$$
C _ { j } ^ { * } = A _ { j } C _ { j } ^ { * } A _ { j } ^ { \top } + \mathrm { d i a g } ( V _ { j } , V _ { j } ^ { \mathrm { m o m } } ) .\tag{274}
$$

Let $R _ { j }$ be the position sub-block of the stationary covariance:

$$
R _ { j } = ( C _ { j } ^ { * } ) _ { w w }\tag{275}
$$

![](images/0469c1c4e66f6a1474116dda5ce85321b3d1de812792af3587218c9c6c870d11.jpg)  
Figure 13: A Floor for Each Precision. Midpoint rod flow with the single- and double-precision rounding floors, compared against gradient descent run in each precision from the same initialization (MLP, $\eta = 0 . 0 5 )$ . Top: The magnitude of the oscillation along the top eigenvector $\hat { u } _ { 1 }$ over time. Gradient descent and rod flow agree closely at each precision. Bottom: The sharpness at the onset of self-stabilization. Self-stabilization lags behind the crossing of the sharpness threshold, and the lag is longer in double precision for both gradient descent and rod flow.

gradient descent, MLP, MSE: floor vs no floor  
Oscillation on the top mode (zoomed in)  
![](images/7caa31fa5668a0c3586f205ffc9e211cf142750b1b58df2c2767994cd0078d2d.jpg)

Oscillation on the top mode (zoomed out)  
![](images/74308c13a7a8bf09e8e63d88d6a0aecc712c3cd5d794cfe2d386e0d70ea04ee5.jpg)

Sharpness  
![](images/af44180d751bb9f8dab59ace2c76e0c3a6b701dd191c6d624ddc5792a2959185.jpg)

Figure 14: Floor Versus No Floor. Comparison of midpoint rod flow, endpoint rod flow, edge flow, and the mean–fluctuation model against gradient descent on an MLP (learning rate $\eta = 0 . 0 5 )$ . Each model is shown with a noise floor (solid) and without it (dotted). The floor of the mean–fluctuation model is ours (Appendix G.4). Top and middle: Oscillation along the top eigenvector $\hat { u } _ { 1 }$ , shown at two zoom levels. Without a floor, every flow’s oscillation decays by more than fifty orders of magnitude before the crossing, and endpoint rod flow’s collapses to exactly zero. Bottom: Without a floor, the flows arrest late. With it, every flow arrests at the same time as the gradient descent iterates.

Table 2: Linearized Update Maps. The linear part A of each optimizer’s update under the quadratic approximation about the midpoint. The preconditioner P is held fixed and bias correction is omitted.
<table><tr><td>Optimizer</td><td>State</td><td>A</td></tr><tr><td>Gradient descent</td><td>w</td><td> $I _ { d } - \eta H$ </td></tr><tr><td>RMSProp</td><td>w</td><td> $I _ { d } - \eta P ^ { - 1 } H$ </td></tr><tr><td>Heavy ball</td><td> $( w , m )$ </td><td> $\left( \begin{array} { c c } { { I _ { d } - \eta ( 1 - \beta ) H } } & { { - \eta \beta I _ { d } } } \\ { { ( 1 - \beta ) H } } & { { \beta I _ { d } } } \end{array} \right)$ </td></tr><tr><td>Nesterov</td><td> $( w , m )$ </td><td> $\begin{array} { r l } { \bigg ( I _ { d } - \eta ( 1 - \beta ) H } & { { } - \eta \beta \big ( I _ { d } - \eta ( 1 - \beta ) H \big ) \bigg ) } \\ { \big ( 1 - \beta ) H } & { { } \quad \beta \big ( I _ { d } - \eta ( 1 - \beta ) H \big ) } \end{array}$ </td></tr><tr><td>Adam</td><td> $( w , m )$ </td><td> $\left( \begin{array} { c c } { { I _ { d } - \eta ( 1 - \beta _ { 1 } ) P ^ { - 1 } H } } & { { - \eta \beta _ { 1 } P ^ { - 1 } } } \\ { { ( 1 - \beta _ { 1 } ) H } } & { { \beta _ { 1 } I _ { d } } } \end{array} \right)$ </td></tr><tr><td>AdamW</td><td> $( w , m )$ </td><td> $\left( \begin{array} { c c } { { \kappa I _ { d } - \eta ( 1 - \beta _ { 1 } ) P ^ { - 1 } H } } & { { - \eta \beta _ { 1 } P ^ { - 1 } } } \\ { { ( 1 - \beta _ { 1 } ) H } } & { { \beta _ { 1 } I _ { d } } } \end{array} \right)$ </td></tr></table>

$R _ { j }$ is the stationary covariance along uˆ<sub>j</sub>. $\hat { u } _ { j }$

Let $\chi _ { j } = \langle \delta , \hat { u } _ { j } \rangle$ be the component of the extent along $\hat { u } _ { j }$ . For simplicity, assume that $\gamma$ is in a fixed ratio to δ. Then $\chi _ { j }$ decays at the rate

$$
s _ { j } = \frac { \operatorname* { d e t } ( I + A _ { j } ) } { 1 + ( A _ { j } ) _ { m m } } ,\tag{276}
$$

so one forward Euler substep multiplies $\chi _ { j }$ by

$$
\rho _ { j } = 1 - s _ { j } d t .\tag{277}
$$

After every substep, the noise floor increases the magnitude of each $\chi _ { j }$ in quadrature while preserving its sign:

$$
\chi _ { j }  \mathrm { s i g n } ( \chi _ { j } ) \sqrt { \chi _ { j } ^ { 2 } + R _ { j } ( 1 - \rho _ { j } ^ { 2 } ) } .\tag{278}
$$

## E.3 Bias Correction

Adam initializes its first and second moment estimates at zero, which biases them toward zero early in training. To correct for this, Adam introduces the bias-correction factors

$$
\begin{array} { r } { \mathrm { b c } _ { 1 } = 1 - \beta _ { 1 } ^ { t + 1 } , \qquad \mathrm { b c } _ { 2 } = 1 - \beta _ { 2 } ^ { t + 1 } . } \end{array}\tag{279}
$$

These define the bias-corrected moments

$$
\hat { m } _ { t + 1 } = \frac { m _ { t + 1 } } { \mathrm { b c } _ { 1 } } ,\tag{280}
$$

$$
\hat { \nu } _ { t + 1 } = \frac { \nu _ { t + 1 } } { \mathrm { b c _ { 2 } } } .\tag{281}
$$

For both rod flow and central flow for Adam, the bias-corrected flows are obtained by replacing m and ν with mˆ and ${ \hat { \nu } } ,$ respectively.

## F Central Flow

This appendix collects the details of central flow that the main text defers.

## F.1 Convex Quadratic Programs

In the main text, we determined the covariance by defining a semidefinite complementarity problem (SDCP). Here, we show that the SDCP is equivalent to a quadratic program. This equivalence explains why the SDCP determines the covariance.

First, it helps to keep the following taxonomy in mind. We can categorize the objects in this appendix into six types, which form three dual pairs:

• Displacements are tangent vectors at w¯. Examples include $\dot { \bar { w } }$ and the columns of $U$

• Gradients are cotangent vectors at w¯ and are dual to displacements. Examples include $\nabla L$ and the momentum $\bar { m }$

• Covariances are symmetric $d \times d$ matrices built from pairs of displacements. Examples include the covariance matrix Σ and the inverse preconditioner $P ^ { - 1 }$

• Duals of covariances are symmetric $d \times d$ matrices that act on pairs of displacements. Examples include the Hessian H, the preconditioner $P _ { \mathrm { : } }$ , and the slack Λ.

• Critical covariances are symmetric $k \times k$ matrices built from pairs of displacements along the critical directions. An example is the critical covariance matrix X.

• Duals of critical covariances are symmetric $k \times k$ matrices that act on pairs of displacements along the critical directions. An example is the critical slack $U ^ { \top } \Lambda U$

For simplicity, we will consider the central flow for gradient descent. The ODE is given as

$$
\frac { d \bar { w } } { d t } = - \eta \nabla L ( \bar { w } ) - \frac { \eta } { 2 } \nabla _ { \bar { w } } \langle H ( \bar { w } ) , \Sigma \rangle
$$

It helps to introduce the notation $\tau$ to denote the sharpness gradient as a linear map:

$$
\mathcal { T } [ \Sigma ] : = \nabla _ { \bar { w } } \langle H ( \bar { w } ) , \Sigma \rangle ,\tag{282}
$$

$\tau$ is a linear operator from the space of covariance matrices to the cotangent space at $\bar { w }$ . Let v be a tangent vector at $\bar { w }$ . Now consider the inner product of v with the sharpness gradient $\tau [ \Sigma ]$ :

$$
\langle \mathcal { T } [ \Sigma ] , v \rangle .\tag{283}
$$

This inner product measures how much the curvature of $L$ in the directions of $\Sigma$ increases when we displace along v. We define the adjoint $\tau ^ { * }$ as the linear operator from the tangent space at w¯ to the space dual to covariance matrices that satisfies

$$
\langle T ^ { \ast } [ v ] , \Sigma \rangle = \langle T [ \Sigma ] , v \rangle\tag{284}
$$

for all tangent vectors v and covariance matrices Σ. The adjoint can be identified with the directional derivative of the Hessian:

$$
{ \cal T } ^ { * } [ v ] = \frac { \mathrm { d } } { \mathrm { d } s } H ( \bar { w } + s v ) \vert _ { s = 0 } .\tag{285}
$$

Now consider the rate of change of the slack. Its critical block is afine in $X$

$$
\boldsymbol { U } ^ { \top } \dot { \boldsymbol { \Lambda } } \boldsymbol { U } = \eta ( \mathsf { A } + \mathsf { B } [ \boldsymbol { X } ] ) ,\tag{286}
$$

where $\eta \mathsf { A }$ is the rate under the free flow $( X = 0 )$ , and B is a linear operator on symmetric $k \times k$ matrices such that $\eta \mathsf { B } [ X ]$ gives the rate of change of the slack due to the oscillations:

$$
{ \sf A } = U ^ { \top } T ^ { * } [ \nabla L ] U , \qquad { \sf B } [ X ] = \frac { 1 } { 2 } U ^ { \top } T ^ { * } [ \mathcal { T } [ U X U ^ { \top } ] ] U .\tag{287}
$$

Note that for any symmetric $k \times k$ matrices X and $X ^ { \prime } ,$ we have

$$
\begin{array} { r } { \langle \mathsf { B } [ \boldsymbol { X } ] , \boldsymbol { X } ^ { \prime } \rangle = \frac { 1 } { 2 } \langle \boldsymbol { U } ^ { \top } \boldsymbol { T } ^ { * } [ \mathcal { T } [ \boldsymbol { U } \boldsymbol { X } \boldsymbol { U } ^ { \top } ] ] \boldsymbol { U } , \boldsymbol { X } ^ { \prime } \rangle } \\ { = \frac { 1 } { 2 } \langle \boldsymbol { T } ^ { * } [ \mathcal { T } [ \boldsymbol { U } \boldsymbol { X } \boldsymbol { U } ^ { \top } ] ] , \boldsymbol { U } \boldsymbol { X } ^ { \prime } \boldsymbol { U } ^ { \top } \rangle } \\ { = \frac { 1 } { 2 } \langle \boldsymbol { T } [ \boldsymbol { U } \boldsymbol { X } \boldsymbol { U } ^ { \top } ] , \mathcal { T } [ \boldsymbol { U } \boldsymbol { X } ^ { \prime } \boldsymbol { U } ^ { \top } ] \rangle . } \end{array}\tag{288}
$$

The final expression is symmetric in X and $X ^ { \prime }$ , so B is self-adjoint. And setting $X ^ { \prime } = X$ gives

$$
\begin{array} { r } { \langle { \mathsf { B } } [ X ] , X \rangle = \frac { 1 } { 2 } \| \mathcal { T } [ U X U ^ { \top } ] \| ^ { 2 } \ge 0 , } \end{array}\tag{289}
$$

so B is positive semidefinite.

We let $\mathrm { S D C P } ( { \mathsf { A } } , { \mathsf { B } } )$ denote the SDCP defined in terms of A and B:

$$
X \succeq 0 , \qquad \mathsf { A } + \mathsf { B } [ X ] \succeq 0 , \qquad \langle \mathsf { A } + \mathsf { B } [ X ] , \ X \rangle = 0 .\tag{290}
$$

The above SDCP can be shown to be equivalent to a convex quadratic program. Consider the objective

$$
f ( \boldsymbol { X } ) : = \frac { 1 } { \eta ^ { 2 } } \Big \| \frac { \mathrm { d } \bar { w } } { \mathrm { d } t } \Big \| ^ { 2 } = \Big \| \nabla L ( \bar { w } ) + \frac { 1 } { 2 } \mathcal { T } [ \boldsymbol { U } \boldsymbol { X } \boldsymbol { U } ^ { \top } ] \Big \| ^ { 2 } .\tag{291}
$$

$f ( X )$ is the squared speed of the midpoint $\bar { w }$ when the critical covariance matrix is $X$ . Because the midpoint velocity is afine in $X , f$ is convex in $X$ . Taking the gradient of $f$ with respect to $X$

$$
\begin{array} { r l } & { \nabla _ { X } f = \nabla _ { X } \Big \| \nabla L ( \bar { w } ) + \frac { 1 } { 2 } \mathcal { T } [ U X U ^ { \top } ] \Big \| ^ { 2 } } \\ & { \qquad = \nabla _ { X } \langle \nabla L ( \bar { w } ) + \frac { 1 } { 2 } \mathcal { T } [ U X U ^ { \top } ] , \nabla L ( \bar { w } ) + \frac { 1 } { 2 } \mathcal { T } [ U X U ^ { \top } ] \rangle } \\ & { \qquad = \nabla _ { X } \Big ( \| \nabla L ( \bar { w } ) \| ^ { 2 } + \langle \mathcal { T } [ U X U ^ { \top } ] , \nabla L ( \bar { w } ) \rangle + \frac { 1 } { 4 } \| \mathcal { T } [ U X U ^ { \top } ] \| ^ { 2 } \Big ) } \\ & { \qquad = \nabla _ { X } \Big ( \langle U ^ { \top } \mathcal { T } ^ { * } [ \nabla L ( \bar { w } ) ] U , X \rangle + \frac { 1 } { 4 } \langle U ^ { \top } \mathcal { T } ^ { * } [ \mathcal { T } [ U X U ^ { \top } ] ] U , X \rangle \Big ) } \\ & { \qquad = U ^ { \top } T ^ { * } [ \nabla L ( \bar { w } ) + \frac { 1 } { 2 } \mathcal { T } [ U X U ^ { \top } ] ] U } \\ & { \qquad = \mathsf { A } + \mathsf { B } [ X ] , } \end{array}\tag{292}
$$

which is the rate of change of the slack in the critical block, divided by η. We can also deduce that

$$
\begin{array} { r } { f ( X ) = \| \nabla L ( \bar { w } ) \| ^ { 2 } + \langle \mathsf { A } , X \rangle + \frac { 1 } { 2 } \langle \mathsf { B } [ X ] , X \rangle . } \end{array}\tag{293}
$$

Consider the problem of minimizing f over positive semidefinite matrices $X$

$$
\operatorname* { m i n } _ { X \succeq 0 } f ( X ) .\tag{294}
$$

When there are no constraints, a convex function is minimized where its gradient is zero. However, the unconstrained minimizer might not lie in the PSD cone, so the gradient need not vanish at the constrained minimizer. Instead, the constrained minimizer must satisfy $\nabla _ { X } f ( X ) \succeq 0$ . Otherwise, one could increase the component of X along an eigenvector of the gradient with a negative eigenvalue, which would decrease the objective while keeping X in the PSD cone. Furthermore, along any eigenvector of the gradient with a positive eigenvalue, the component of X must be zero. Otherwise, one could decrease that component to decrease the objective. This means that $\nabla _ { X } f ( X )$ and X must be orthogonal.

Table 3: Inputs to the Quadratic Program. The rate of change of the critical block of the slack, $U ^ { \top } \dot { \Lambda } U = \eta ( \mathsf { A } + \mathsf { B } [ X ] )$ , for each optimizer. $\eta \mathsf { A }$ is the rate of change under the free flow. ηB[X] is the rate of change due to the oscillations. For RMSProp, Adam, and AdamW, Λ denotes that optimizer’s slack.
<table><tr><td>Optimizer</td><td>A</td><td>B[X]</td></tr><tr><td>Gradient descent</td><td> $U ^ { \top } T ^ { * } [ \nabla L ] U$ </td><td> $\scriptstyle { \frac { 1 } { 2 } } U ^ { \top } { \mathcal { T } } ^ { * } { \mathcal { T } } [ U X U ^ { \top } ] U$ </td></tr><tr><td>Heavy ball</td><td> $\begin{array} { r } { U ^ { \top } \mathcal { T } ^ { * } \big [ \beta \bar { m } + ( 1 - \beta ) \nabla L \big ] U } \end{array}$ </td><td> $\begin{array} { r } { \frac { 1 - \beta } { 2 } \boldsymbol { U } ^ { \top } \boldsymbol { \mathcal { T } } ^ { * } \boldsymbol { \mathcal { T } } [ \boldsymbol { U } \boldsymbol { X } \boldsymbol { U } ^ { \top } ] \boldsymbol { U } } \end{array}$ </td></tr><tr><td>Nesterov</td><td> $U ^ { \top } \mathcal { T } ^ { * } \big [ \beta ^ { 2 } \bar { m } + ( 1 - \beta ^ { 2 } ) \nabla L ( \bar { \theta } ) \big ] U$ </td><td> $\textstyle { \frac { 1 - \beta ^ { 2 } } { 2 } } U ^ { \top } { \mathcal { T } } ^ { * } { \mathcal { T } } [ U X U ^ { \top } ] U$ </td></tr><tr><td>RMSProp</td><td> $\boldsymbol { U } ^ { \top } \mathcal { T } ^ { * } \bigl \lceil \boldsymbol { P } ^ { - 1 } \boldsymbol { \nabla } \boldsymbol { L } \bigr \rceil \boldsymbol { U }$   $\begin{array} { r } { + \frac { 1 - \beta _ { 2 } } { \eta } U ^ { \top } \nabla _ { \bar { \nu } } \Lambda \big [ ( \nabla L ) ^ { \odot 2 } - \bar { \nu } \big ] U } \end{array}$ </td><td> ${ \scriptstyle { \frac { 1 } { 2 } } } U ^ { \top } { \mathcal { T } } ^ { * } \big [ P ^ { - 1 } { \mathcal { T } } [ U X U ^ { \top } ] \big ] U$   $\begin{array} { r l } { { } } & { { } + \frac { 1 - \beta _ { 2 } } { \eta } U ^ { \top } \nabla _ { \bar { \nu } } \Lambda \big [ \mathrm { d i a g } ( H U X U ^ { \top } H ) \big ] U } \end{array}$ </td></tr><tr><td>Adam</td><td> $\begin{array} { r } { U ^ { \top } T ^ { * } \big [ P ^ { - 1 } \big ( \beta _ { 1 } \bar { m } + ( 1 - \beta _ { 1 } ) \nabla L \big ) \big ] U } \end{array}$   $\begin{array} { r l r } { \mathrm { ~ } } & { { } } & { + \frac { 1 - \beta _ { 2 } } { \eta } U ^ { \top } \nabla _ { \bar { \nu } } \Lambda \big [ ( \nabla L ) ^ { \odot 2 } - \bar { \nu } \big ] U } \end{array}$ </td><td> $\begin{array} { r l } {  { \frac { 1 - \beta _ { 1 } } { 2 } \boldsymbol { U } ^ { \top } \boldsymbol { \mathcal { T } ^ { * } } \bigl [ P ^ { - 1 } \boldsymbol { \mathcal { T } } [ \boldsymbol { U } X \boldsymbol { U } ^ { \top } ] \bigr ] \boldsymbol { U } } } \end{array}$   $\begin{array} { r l } { { } } & { { } + \frac { 1 - \beta _ { 2 } } { \eta } U ^ { \top } \nabla _ { \bar { \nu } } \Lambda \big [ \mathrm { d i a g } ( H U X U ^ { \top } H ) \big ] U } \end{array}$ </td></tr><tr><td>AdamW</td><td> $\begin{array} { r l } {  { U ^ { \top } \mathcal { T } ^ { * } \bigl [ \zeta \bar { w } + P ^ { - 1 } \bigl ( \beta _ { 1 } \bar { m } + ( 1 - \beta _ { 1 } ) \nabla L \bigr ) \bigr ] U } \qquad } & { { } } \end{array}$   $\begin{array} { r } { + \frac { 1 - \beta _ { 2 } } { \eta } U ^ { \top } \nabla _ { \bar { \nu } } \Lambda \big [ ( \nabla L ) ^ { \odot 2 } - \bar { \nu } \big ] U } \end{array}$ </td><td> $\begin{array} { r l } {  { \frac { 1 - \beta _ { 1 } } { 2 } \boldsymbol { U } ^ { \top } \boldsymbol { \mathcal { T } ^ { * } } \bigl [ P ^ { - 1 } \boldsymbol { \mathcal { T } } [ \boldsymbol { U } X \boldsymbol { U } ^ { \top } ] \bigr ] \boldsymbol { U } } } \end{array}$   $\begin{array} { r l } { { } } & { { } + \frac { 1 - \beta _ { 2 } } { \eta } U ^ { \top } \nabla _ { \bar { \nu } } \Lambda \big [ \mathrm { d i a g } ( H U X U ^ { \top } H ) \big ] U } \end{array}$ </td></tr></table>

Since f is convex, these conditions are also suficient for optimality. We have that minimizing f subject to $X \succeq 0$ is equivalent to

$$
X \succeq 0 , \quad \nabla _ { X } f ( X ) \succeq 0 , \quad \langle \nabla _ { X } f ( X ) , X \rangle = 0 .\tag{295}
$$

Substituting our expression for $\nabla _ { X } f$ recovers $\mathrm { S D C P } ( { \mathsf { A } } , { \mathsf { B } } )$ . It’s important to note that if the map $X \mapsto \mathcal { T } [ U X U ^ { \top } ]$ is injective, then B is positive definite—from which it follows that f is a strongly convex quadratic. Because minimizing a strongly convex function over a closed convex set always yields a unique solution, X is uniquely determined.

Interestingly, because the objective f is the squared speed of the midpoint w¯, there is an alternative interpretation: among all covariance matrices whose oscillations are contained within the critical subspace, the central flow selects the one for which the midpoint moves as slowly as possible.

The same structure recurs for the other optimizers (Table 3). In each case, the rate of change of the critical block of the slack is afine in X, and the persistence conditions give rise to SDCP(A, B), with A and B specific to the optimizer. Whenever B is self-adjoint and positive semidefinite, the SDCP gives the optimality conditions of a convex quadratic program—which has a unique minimizer when $\mathsf { B } \succ 0$

In terms of computational implementation, we found that we were able to use the same SDCP solver as Cohen et al. [2025] with minimal modification.

## F.2 Central Flows as Projected Dynamical Systems

For gradient descent, central flow can be interpreted as a projected gradient flow. We extend this interpretation to the other optimizers, where it becomes a constrained flow.

Consider the central flow for gradient descent. Define the objective $\mathcal { F }$ as the supremum over $\Sigma \succeq 0$ of the truncated edge coupling:

$$
\mathcal { F } ( \bar { w } ) : = \operatorname* { s u p } _ { \Sigma \succeq 0 } \widetilde { \Psi } ( \bar { w } , \Sigma ) = L ( \bar { w } ) + \iota _ { \Omega } ( \bar { w } ) ,\tag{296}
$$

where $\Omega = \{ \bar { w } : \Lambda ( \bar { w } ) \succeq 0 \}$ is the feasible region and $\iota \Omega$ is its indicator function:

$$
\iota _ { \Omega } ( \bar { w } ) = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ } \bar { w } \in \Omega , } \\ { + \infty } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{297}
$$

The indicator $\iota _ { \Omega } ( \bar { w } )$ creates an infinite potential barrier that constrains the flow to the feasible region. The central flow is a gradient flow on $\mathcal { F }$ , though only in the weak sense of a subdiferential inclusion [Brézis, 1973]:

$$
\frac { \mathrm { d } \bar { w } } { \mathrm { d } t } \in - \eta \partial \mathcal { F } ( \bar { w } ) = - \eta \big ( \nabla L ( \bar { w } ) + N _ { \Omega } ( \bar { w } ) \big ) ,\tag{298}
$$

where $N _ { \Omega } ( \bar { w } )$ is the normal cone to $\Omega$ at $\bar { w } ,$ whose elements act as normal forces constraining the flow to the feasible region. Among the velocities that the inclusion allows, central flow selects the one of least norm (Appendix F.1). It is the slow solution of the inclusion [Henry, 1973, Cornet, 1983].

We can interpret the other central flows similarly. For each optimizer, we define an objective by taking the supremum over $\Sigma \succeq 0$ of the truncated reduced edge coupling:

$$
\mathcal { F } ( \bar { z } ) : = \operatorname* { s u p } _ { \Sigma \succeq 0 } \widetilde { \Upsilon } ^ { \mathrm { r e d } } ( \bar { z } , \Sigma ) = \mathcal { L } ( \bar { z } ) + \iota _ { \Omega } ( \bar { w } ) ,\tag{299}
$$

with $\mathcal { L }$ and $\Omega$ given in Table 4. The central flow then satisfies the diferential inclusion

$$
\frac { \mathrm { d } \bar { z } } { \mathrm { d } t } \in K \partial \mathcal { F } ( \bar { z } ) = K \left( \nabla \mathcal { L } ( \bar { z } ) + \binom { N _ { \Omega } ( \bar { z } ) } { 0 } \right) ,\tag{300}
$$

where K is the structure matrix of the optimizer.

For the preconditioned methods, the constraint depends on the joint state $( \bar { w } , \bar { \nu } )$ . There are then two ways to leave the feasible region: the sharpness can increase as w¯ moves, or ν¯ can shrink. The covariance acts to counteract both efects. As a result, each normal force in $N _ { \Omega } ( \bar { z } , \bar { \nu } )$ is a 2d-dimensional vector, with components for both w¯ and $\bar { \nu } .$

Just as we defined the w¯ component of the normal force using the subdiferential of the truncated edge coupling with respect to $\bar { w } .$ , we define the $\bar { \nu }$ component using the subdiferential with respect to ν¯. The $\bar { \nu }$ component of the normal force, which we denote $N _ { \Omega } ^ { \bar { \nu } }$ , is then

$$
\begin{array} { l } { { \displaystyle N _ { \Omega } ^ { \bar { \nu } } = \partial _ { \bar { \nu } } \tilde { \Psi } _ { \eta , \bar { \nu } } } } \\ { { \displaystyle \quad = - \partial _ { \bar { \nu } } \frac 1 2 \langle \Lambda _ { \mathrm { R M S P r o p } } , \Sigma \rangle } } \\ { { \displaystyle \quad = - \frac 1 { 2 \eta } \mathrm { D i a g } ( \bar { \nu } ) ^ { - 1 / 2 } \mathrm { d i a g } ( \Sigma ) . } } \end{array}\tag{301}
$$

The central flow ODE for $\bar { \nu }$ is

$$
\frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = \left( 1 - \beta _ { 2 } \right) \left[ ( \nabla { L } ) ^ { \odot 2 } + \mathrm { d i a g } ( H \Sigma H ) - \bar { \nu } \right] .\tag{302}
$$

Recall that the columns of $U$ lie in the critical subspace. Since these columns have preconditioned sharpness equal to $2 / \eta$ , we have $\begin{array} { r } { H U = \frac { 2 } { \eta } P U } \end{array}$ . Because $P$ is diagonal, it follows that

$$
\begin{array} { r } { \mathrm { d i a g } ( H { \Sigma } H ) = \frac { 4 } { \eta ^ { 2 } } P ^ { 2 } \mathrm { d i a g } ( { \Sigma } ) . } \end{array}\tag{303}
$$

Table 4: Central Flow Objectives. For each optimizer, we give the generating function L and the feasible region Ω of the objective. For the preconditioned methods, we give the metric $W _ { \nu }$ through which the normal cone acts on the second moment.
<table><tr><td>Optimizer</td><td> $\mathcal { L }$ </td><td>Ω</td><td> $W _ { \nu }$ </td></tr><tr><td>Gradient descent</td><td> $L ( \bar { w } )$ </td><td> $\{ \Lambda ( \bar { w } ) \succeq 0 \}$ </td><td></td></tr><tr><td>Heavy ball</td><td> $\begin{array} { r } { L ( \bar { w } ) - \frac { \eta \beta } { 2 ( 1 - \beta ) } \| \bar { m } \| ^ { 2 } } \end{array}$ </td><td> $\left\{ \Lambda _ { \mathrm { H B } } ( \bar { w } ) \succeq 0 \right\}$ </td><td></td></tr><tr><td>Nesterov</td><td> $\begin{array} { r } { L ( \bar { \theta } ) - \frac { \eta \beta ^ { 2 } } { 2 ( 1 - \beta ) } \| \bar { m } \| ^ { 2 } } \end{array}$ </td><td> $\{ \Lambda _ { \mathrm { N A G } } ( \bar { \theta } ) \succeq 0 \}$ </td><td></td></tr><tr><td>RMSProp</td><td>L(w)</td><td> $\left\{ \Lambda _ { \mathrm { R M S P r o p } } ( \bar { w } , \bar { \nu } ) \succeq 0 \right\}$ </td><td> $\frac { 8 ( 1 - \beta _ { 2 } ) } { \eta } P ^ { 2 } \operatorname { D i a g } \left( \sqrt { \bar { \nu } } \right)$ </td></tr><tr><td>Adam</td><td> $\begin{array} { r } { L ( \bar { w } ) - \frac { \eta \beta _ { 1 } } { 2 \left( 1 - \beta _ { 1 } \right) } \bar { m } ^ { \top } P ^ { - 1 } \bar { m } } \end{array}$ </td><td> $\left\{ \Lambda _ { \mathrm { A d a m } } ( \bar { w } , \bar { \nu } ) \succeq 0 \right\}$ </td><td> $\frac { 8 ( 1 - \beta _ { 2 } ) ( 1 + \beta _ { 1 } ) } { \eta ( 1 - \beta _ { 1 } ) } P ^ { 2 } \mathrm { D i a g } ( \sqrt { \bar { \nu } } )$ </td></tr><tr><td>AdamW</td><td> $\begin{array} { r } { L ( \bar { w } ) - \frac { \eta \beta _ { 1 } } { 2 \left( 1 - \beta _ { 1 } \right) } \bar { m } ^ { \top } P ^ { - 1 } \bar { m } } \end{array}$   $+ \frac { \zeta } { 2 ( 1 - \beta _ { 1 } ) } [ \| \bar { w } \| _ { P } ^ { 2 }$   $\begin{array} { r } { - \frac { \beta _ { 1 } } { \kappa } \| \bar { w } + \eta P ^ { - 1 } \bar { m } \| _ { P } ^ { 2 } ] } \end{array}$ </td><td> $\left\{ \Lambda _ { \mathrm { A d a m W } } ( \bar { w } , \bar { \nu } ) \succeq 0 \right\}$ </td><td> $\frac { 4 ( 1 - \beta _ { 2 } ) ( 1 + \kappa ) ( 1 + \beta _ { 1 } ) } { \eta ( 1 - \beta _ { 1 } ) } P ^ { 2 } \mathrm { D i a g } ( \sqrt { \bar { \nu } } )$ </td></tr></table>

Substituting into the ODE for ν¯ and comparing with (301), we deduce that

$$
\frac { \mathrm { d } \bar { \nu } } { \mathrm { d } t } = \bigl ( 1 - \beta _ { 2 } \bigr ) \bigl [ ( \nabla L ) ^ { \odot 2 } - \bar { \nu } \bigr ] - W _ { \nu } \partial _ { \bar { \nu } } \widetilde { \Psi } _ { \eta , \bar { \nu } } ,\tag{304}
$$

where the matrix

$$
\begin{array} { r } { W _ { \nu } = \frac { 8 ( 1 - \beta _ { 2 } ) } { \eta } P ^ { 2 } \mathrm { D i a g } ( \bar { \nu } ) ^ { 1 / 2 } } \end{array}\tag{305}
$$

is efectively an inverse metric tensor that converts the normal force into its efect on the second moment. The central flow of RMSProp is therefore a projected dynamical system [Dupuis and Nagurney, 1993, Nagurney and Zhang, 1996], though not a projected gradient flow, because the unconstrained dynamics of ν¯ are not a gradient flow.

## G Experimental Details

This appendix provides further details about our experimental setup. Code to reproduce our experiments is included as ancillary files with the arXiv version of this paper.

## G.1 Stable Flows

The stable flow of an optimizer is its naïve continuous-time limit, obtained by replacing the discrete diference $z _ { t + 1 } - z _ { t }$ in the update equations with the time derivative z˙. For a structured gradient method, the stable flow is $\dot { z } = K \nabla \mathcal { L } ( z )$

Below the sharpness threshold, the stable flow closely tracks the discrete iterates. However, while the iterates self-stabilize such that their sharpness hovers at the threshold, the sharpness of the stable flow keeps rising past the threshold. We therefore use the stable flow as the baseline against which to compare rod flow and central flow. The stable flows of the six optimizers are as follows.

Gradient descent.

$$
\frac { \mathrm { d } w } { \mathrm { d } t } = - \eta \nabla L ( w ) .\tag{306}
$$

RMSProp.

$$
\frac { \mathrm { d } w } { \mathrm { d } t } = - \eta P ^ { - 1 } \nabla L ( w ) ,
$$

$$
\frac { \mathrm { d } \nu } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \big [ \nabla L ( w ) ^ { \odot 2 } - \nu \big ] .\tag{307}
$$

Heavy ball.

$$
\frac { \mathrm { d } w } { \mathrm { d } t } = - \eta \bigl [ \beta m + ( 1 - \beta ) \nabla L ( w ) \bigr ] ,
$$

$$
\frac { \mathrm { d } m } { \mathrm { d } t } = ( 1 - \beta ) \big [ \nabla L ( w ) - m \big ] .\tag{308}
$$

Nesterov momentum.

$$
\frac { \mathrm { d } \theta } { \mathrm { d } t } = - \eta \bigl [ \beta ^ { 2 } m + ( 1 - \beta ^ { 2 } ) \nabla L ( \theta ) \bigr ] ,
$$

$$
\frac { \mathrm { d } m } { \mathrm { d } t } = ( 1 - \beta ) \big [ \nabla L ( \theta ) - m \big ] .\tag{309}
$$

Adam.

$$
\frac { \mathrm { d } w } { \mathrm { d } t } = - \eta \hat { P } ^ { - 1 } \big [ \beta _ { 1 } \hat { m } + ( 1 - \beta _ { 1 } ) \nabla \hat { L } ( w ) \big ] ,
$$

$$
\frac { \mathrm { d } m } { \mathrm { d } t } = ( 1 - \beta _ { 1 } ) \big [ \nabla L ( w ) - m \big ] ,\tag{310}
$$

$$
\frac { \mathrm { d } \nu } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \big [ \nabla L ( w ) ^ { \odot 2 } - \nu \big ] .
$$

AdamW.

$$
\frac { \mathrm { d } w } { \mathrm { d } t } = - \eta \zeta w - \eta \hat { P } ^ { - 1 } \big [ \beta _ { 1 } \hat { m } + \big ( 1 - \beta _ { 1 } \big ) \nabla \widehat { L } ( w ) \big ] ,
$$

$$
\frac { \mathrm { d } m } { \mathrm { d } t } = ( 1 - \beta _ { 1 } ) \big [ \nabla L ( w ) - m \big ] ,\tag{311}
$$

$$
\frac { \mathrm { d } \nu } { \mathrm { d } t } = ( 1 - \beta _ { 2 } ) \big [ \nabla L ( w ) ^ { \odot 2 } - \nu \big ] .
$$

Note that the stable flows for Adam and AdamW are bias-corrected (Appendix E.3).

## G.2 Runs

We ran each of the six optimizers on three architectures (MLP, CNN, and ViT). The step size, handof iteration, and crossing iteration for each run are given in Table 5.

Data and models. We used the first 5 000 images of the CIFAR-10 training set, normalized per channel, with no data augmentation. All runs on a given architecture start from the same initialization: PyTorch’s default, with a fixed seed.

The MLP flattens each input image into a vector of 3 072 values. It has two hidden layers of width 50, each consisting of a fully connected layer followed by a SiLU activation, and a linear output layer that maps to the ten outputs. In total, the MLP has 156 710 parameters.

The CNN operates directly on the image, represented as three channels of $3 2 \times 3 2$ pixels. It has two convolutional layers, each with 32 channels and $3 \times 3$ kernels, and each padded by one pixel so that the convolution preserves the image size. Each convolution is followed by a tanh activation and $2 \times 2$ average pooling, which halves the image size from $3 2 \times 3 2$ to $1 6 \times 1 6$ , then to $8 \times 8 .$ . A final linear layer maps the resulting $3 2 \times 8 \times 8 = 2 0 4 8$ features to the ten outputs. In total, the CNN has 30 634 parameters.

Table 5: Runs. The step size $\underline { { \boldsymbol { \eta } , } }$ handof $t _ { h } ,$ , and crossing $t _ { c }$ of each run.
<table><tr><td rowspan="2">Optimizer</td><td colspan="3">MLP</td><td colspan="3">CNN</td><td colspan="3">ViT</td></tr><tr><td>η</td><td> $t _ { h }$ </td><td> $t _ { c }$ </td><td>η</td><td> $t _ { h }$ </td><td> $t _ { c }$ </td><td>η</td><td> $t _ { h }$ </td><td> $t _ { c }$ </td></tr><tr><td>Gradient descent</td><td>0.05</td><td>8700</td><td>9700</td><td>0.05</td><td>19600</td><td>20 600</td><td>0.1</td><td>4600</td><td>5100</td></tr><tr><td>RMSProp</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>1500</td><td>2500</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>3700</td><td>4800</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>3700</td><td>4200</td></tr><tr><td>Heavy ball</td><td></td><td>0.3 3800</td><td>4800</td><td>0.7</td><td>1600</td><td>2800</td><td>0.2</td><td>25700</td><td>6200</td></tr><tr><td>Nesterov</td><td>0.2</td><td>3100</td><td>4100</td><td>0.4</td><td>1200</td><td>2200</td><td>0.2</td><td>3700</td><td>4300</td></tr><tr><td>Adam</td><td> $1 0 ^ { - 5 }$ </td><td>3900</td><td>4900</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>1000</td><td></td><td> $1 0 ^ { - 5 }$ </td><td>4000</td><td>4500</td></tr><tr><td>AdamW</td><td> $1 0 ^ { - 5 }$ </td><td>6 200</td><td>7200</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>1800</td><td>2700</td><td> $1 0 ^ { - 5 }$ </td><td>5600</td><td>6100</td></tr></table>

The ViT is a SimpleViT, as implemented in the vit-pytorch library. It splits each image into an $8 \times 8$ grid of $4 \times 4$ patches, giving 64 tokens. Each patch, a vector of 48 values, passes through a LayerNorm, a linear embedding of width 64, and a second LayerNorm. The ViT then adds fixed two-dimensional sinusoidal position embeddings. It does not use a class token. The tokens pass through three transformer blocks. Each block applies multi-head self-attention, with 8 heads of width 8 and bias-free projections, followed by an MLP of width 256 with GELU activations. Both sublayers are preceded by a LayerNorm and wrapped in a residual connection. After a final LayerNorm, the ViT mean-pools the tokens, and a linear head maps the result to the ten outputs. In total, the ViT has 153 322 parameters. To lower the sharpness at initialization, we halve the initial weights of the head.

We also replace each LayerNorm with a reimplementation in elementary $\mathrm { P y }$ Torch operations, so that the loss can be diferentiated three times, as central flow requires.

Training. We train full-batch with the MSE loss. We chose MSE because it’s the loss function which induces sustained edge of stability dynamics. We also ran experiments with cross-entropy loss on gradient descent and RMSProp (Appendix H.2).

Hyperparameters. All four momentum methods use $\beta = \beta _ { 1 } = 0 . 9$ . Adam and AdamW use $\beta _ { 2 } = 0 . 9 9 9$ . These are the standard hyperparameter settings used in practice. Adam and AdamW also use $\epsilon = 1 0 ^ { - 7 }$ and bias correction. RMSProp uses the PyTorch defaults: $\beta _ { 2 } = 0 . 9 9$ and $\epsilon = 1 0 ^ { - 8 }$ without bias correction. For AdamW, we chose a weight decay $\zeta$ large enough to noticeably delay the crossing into the edge of stability relative to Adam.

Step sizes. We chose the step sizes using probe runs of the discrete optimizer, so that the crossing occurs neither too early nor too late. In practice, this meant choosing a step size such that the edge of stability was reached within a couple thousand iterations.

Crossing and handof. The crossing $t _ { c }$ is the first iteration at which the sharpness at the discrete midpoint exceeds the threshold. Since the sharpness is computed every 100 iterations, $t _ { c }$ is a multiple of 100. We placed the handof $t _ { h }$ using the crossing in a probe run. As a result, the handof comes roughly 1 000 iterations before the crossing (500 for the ViT). On the CNN, Adam’s sharpness plateaus just below the threshold and never crosses it. For this run, the handof instead comes 1 000 iterations before the sharpness reaches its plateau.

Table 6: Efective Sharpness and Thresholds. The efective sharpness of each optimizer and its threshold S<sup>⋆</sup>.
<table><tr><td>Optimizer</td><td>Effective sharpness</td><td>Threshold  $S ^ { \star }$ </td></tr><tr><td>Gradient descent</td><td> $\lambda _ { 1 } ( H )$ </td><td>2</td></tr><tr><td rowspan="3">RMSProp</td><td></td><td>η2-m</td></tr><tr><td> $\lambda _ { 1 } ( P ^ { - 1 / 2 } H P ^ { - 1 / 2 } )$ </td><td></td></tr><tr><td></td><td></td></tr><tr><td rowspan="2">Heavy ball</td><td> $\lambda _ { 1 } ( H )$ </td><td> $2 ( 1 + \beta )$ </td></tr><tr><td></td><td> $\overline { { \eta ( 1 - \beta ) } }$   $2 ( 1 + \beta )$ </td></tr><tr><td>Nesterov</td><td> $\lambda _ { 1 } ( H )$ </td><td> $\overline { { \eta ( 1 - \beta ) ( 1 + 2 \beta ) } }$ </td></tr><tr><td>Adam</td><td> $\lambda _ { 1 } ( \hat { P } ^ { - 1 / 2 } H \hat { P } ^ { - 1 / 2 } )$ </td><td> $2 ( 1 + \beta _ { 1 } )$   $\overline { { \eta ( 1 - \beta _ { 1 } ) } }$ </td></tr><tr><td>AdamW</td><td> $\lambda _ { 1 } ( \hat { P } ^ { - 1 / 2 } H \hat { P } ^ { - 1 / 2 } )$ </td><td> $\left( 1 - \frac { \eta \zeta } { 2 } \right) \frac { 2 ( 1 + \beta _ { 1 } ) } { \eta ( 1 - \beta _ { 1 } ) }$ </td></tr></table>

Handof state. At the handof, each flow is initialized from the last two discrete iterates. Its center is set to their midpoint. For rod flow, the extent is initialized from the half-diferences of the last two iterates’ positions and momenta. These half-diferences are dominated by the drift of the center rather than by the oscillation, so the initial amplitude of the extent corresponds to the entire discrete step, not just the oscillating portion. However, this transient quickly fades—well before the edge of stability is reached.

## G.3 Measurements

Sharpness. Every 100 iterations, we compute the top six eigenvalues of the Hessian at the center of each trajectory. We use LOBPCG on Hessian-vector products, warm-started from the eigenvectors of the previous call. For RMSProp, Adam, and AdamW, we instead compute the eigenvalues of the pencil $( H , P )$ , which are those of $P ^ { - 1 / 2 } H P ^ { - 1 / 2 }$ (with $\hat { P }$ in place of $P$ for Adam and AdamW). Each trajectory uses its own preconditioner: the iterates use the discrete optimizer’s, and each flow uses the one given by its second moment ν¯. Table 6 gives the resulting efective sharpness of each optimizer, along with its threshold.

Oscillation envelope. The half-displacement $\hat { \delta } _ { t }$ of consecutive iterates contains not only the oscillation but also the drift of the midpoint. For gradient descent, writing $[ [ g ] ] _ { t }$ and $\bar { g } _ { t }$ for the half-diference and the average of the gradients at $w _ { t }$ and $w _ { t + 1 }$ , we have

$$
\begin{array} { r } { \hat { \delta } _ { t } = \frac { \eta } { 2 } [ [ g ] ] _ { t } + \frac { \eta } { 2 } \bar { g } _ { t } , \qquad \frac { \eta } { 2 } \bar { g } _ { t } = - \frac { 1 } { 2 } \big ( \bar { w } _ { t + 1 } - \bar { w } _ { t } \big ) . } \end{array}
$$

Before the crossing, the drift dominates the half-displacement, so the onset of the oscillation cannot be seen in it. We therefore remove the drift and undo the alternating sign. For every optimizer, we define the oscillation envelope of the iterates as

$$
a _ { t } : = ( - 1 ) ^ { t } \big [ \widehat { \delta } _ { t } + \textstyle \frac 1 2 ( \bar { w } _ { t + 1 } - \bar { w } _ { t } ) \big ] .\tag{312}
$$

The figures compare the norm of $a _ { t }$ with the norm of the extent $\lVert \delta \rVert$ of the rod flows.

Distance. We measure how closely a flow tracks the iterates by the Euclidean distance from its center to the discrete midpoint $\bar { w } _ { t }$ . For the momentum methods, this distance compares positions only. We likewise evaluate the loss at each center.

## G.4 Numerical Integration

Time stepping. The flows run in lockstep with the discrete optimizer: after each optimizer iteration, every flow takes four forward Euler substeps of size $\mathrm { d } t = 0 . 2 5$ , so one unit of flow time corresponds to one iteration.

Storing the extent. Midpoint rod flow does not store its extent as a vector. Without the noise floor, the extent decays by up to several hundred e-folds between the handof and the crossing, approaching the range of double precision, which is about 700 e-folds. A stored vector could then underflow to zero and lose its direction.

We instead store a unit axis vˆ and a log-amplitude $\log \lVert \Delta \rVert$ . Each substep applies forward Euler to the extent, $\Delta \gets \| \Delta \| ( \hat { v } + \mathrm { d } t f )$ , where f is the velocity of the extent per unit amplitude. The axis ${ \hat { v } } + \mathrm { d } t \ f$ is then renormalized, and the logarithm of its norm is added to the log-amplitude. By contrast, endpoint rod flow stores the endpoints themselves.

Noise floor. Both rod flows include the noise floor of Appendix E.2 along the three sharpest directions. The floor models the rounding error in the discrete optimizer’s single-precision state.

Central flow. Central flow solves its semidefinite complementarity problem at every substep. To find the critical directions, it recomputes the top eigenvectors of the (preconditioned) Hessian at each substep, warm-started from the previous substep. The eigenvectors whose eigenvalues lie within 2.5% of the threshold are passed to the problem, so that it can catch an eigenvalue arriving at the threshold. With one such eigenvector, the problem is solved in closed form. With more, it is solved using the convex solver of Cohen et al. [2025], at a relative accuracy of about $1 0 ^ { - 4 }$

Edge flow. For gradient descent, we also ran edge flow [Marion, 2026], which tracks a center $\bar { w } .$ , a unit axis $u ,$ and an amplitude x. These evolve as

$$
\begin{array} { r l } & { \dot { \bar { w } } = - \frac { \eta } { 2 } \big [ \nabla L ( \bar { w } + x u ) + \nabla L ( \bar { w } - x u ) \big ] , } \\ & { \dot { x } = \left( \eta \| H ( \bar { w } ) u \| - 2 \right) x , } \\ & { u \gets H ( \bar { w } ) u / \| H ( \bar { w } ) u \| . } \end{array}\tag{313}
$$

The center follows the negative average of the gradients at the endpoints $\bar { w } \pm x u .$ , and the amplitude grows when the sharpness estimate $\| H ( \bar { w } ) u \|$ exceeds $2 / \eta$ . The axis takes one step of power iteration at each substep. At the handof, we set u to the top eigenvector of the Hessian and $x = | \langle \hat { \delta } , u \rangle |$ Following Marion [2026], we kept the floor $x \geq | \langle \nabla L ( \bar { w } ) , u \rangle |$

Mean–fluctuation model. We implemented the mean–fluctuation model from the first version of Chodron de Courcel [2026]. The second version keeps the same equations but computes the covariance diferently. The model tracks a center $\bar { w }$ and a covariance $\Sigma ,$ which evolve as

$$
\begin{array} { r l } & { \dot { \bar { w } } = - \eta \big [ \nabla L ( \bar { w } ) + \frac { 1 } { 2 } \nabla _ { \bar { w } } \langle H ( \bar { w } ) , \Sigma \rangle \big ] , } \\ & { \dot { \Sigma } = - \eta \big ( H ( \bar { w } ) \Sigma + \Sigma H ( \bar { w } ) \big ) + \eta ^ { 2 } H ( \bar { w } ) \Sigma H ( \bar { w } ) . } \end{array}\tag{314}
$$

![](images/76ab89c102178a5b25d7e4d605aafa92bcdd20dc2cadbe7d0d3afc215e04b3d6.jpg)  
Figure 15: Double-Precision Flow State. Distance between the center of midpoint rod flow and the discrete midpoint, with the flow’s state stored in single and double precision (gradient descent, MLP, $\eta = 0 . 0 5 )$ . With single precision, the distance grows as rounding error accumulates, reaching several times that of the double-precision flow.

The center moves as in central flow, but Σ follows the linearized gradient descent map, with no threshold. As in the first version, we restricted Σ to the span of the top three Hessian eigenvectors (two for the ViT), which we recomputed every 2.5 iterations rather than transported. At the handof, we set Σ to the projection of $\hat { \delta } \otimes \hat { \delta }$ onto this span. Neither version of the model has a noise floor, so we added one: we held the standard deviation of Σ along each of these eigenvectors at $1 0 ^ { - 8 }$ or above, about the size of the rounding noise along uˆ<sub>1</sub> before the crossing (Figure 12).

Precision. The discrete optimizers run in single precision. Every flow keeps its state in double precision and casts it to single precision only to evaluate the loss and its derivatives. The flows thus compute their gradients with the same arithmetic as the iterates, but rounding error does not accumulate in their state. Endpoint rod flow needs double precision for a second reason: a single-precision pair of endpoints cannot resolve the kicks of the noise floor.

On the CNN, the discrete runs are not bitwise reproducible. We kept PyTorch’s default cuDNN settings, which allow the convolutions to use TF32 arithmetic and nondeterministic algorithms. This does not afect the comparisons, since the flows run in lockstep with the iterates.

## H Additional Experiments

In this appendix, we present the results of our experiments, including additional experiments not featured in the main text.

## H.1 Flow Comparisons

Figures 16 to 21 compare the flows with the discrete iterates. Each experiment starts with a warmup phase in which the discrete iterates progressively sharpen. About 1 000 iterations before the edge of stability (500 for the ViT), the flows are initialized from the discrete iterates. The iterates and the flows then advance in lockstep for 4 000 iterations (2 000 for the ViT), which is long enough to capture the transition from the stable regime into the edge of stability and to let the fluctuations from the onset of self-stabilization dissipate. We compare the iterates and the flows using metrics such as the loss and the sharpness.

![](images/c9730676194d1470ad703c6fb160e7c6fd7e46b725a1b3a864df69ccadd0bb69.jpg)  
Figure 16: Flow Comparison for Gradient Descent. MLP: $\eta = 0 . 0 5$ . CNN: $\eta = 0 . 0 5$ . ViT: $\eta = 0 . 1$ . The rows show, from top to bottom: the loss at each center; the sharpness at each center; the top six eigenvalues at the discrete optimizer’s midpoint; the oscillation amplitude; the alignment of each flow’s oscillation with the discrete-time oscillation; and the distance from each flow’s center to the discrete optimizer’s midpoint. Dashed vertical lines mark the handof, and dotted vertical lines mark the crossing into the edge of stability.

![](images/88f1cc965afae2847ed58bfdef0bf7de4b24231e3f40c752b836e57338df01a0.jpg)  
Figure 17: Flow Comparison for RMSProp (no bias correction). MLP: $\eta = 2 \times 1 0 ^ { - 6 }$ . CNN: $\eta = 2 \times 1 0 ^ { - 6 }$ . ViT: $\eta = 2 \times 1 0 ^ { - 6 }$ . All with $\beta _ { 2 } = 0 . 9 9$ and $\epsilon = 1 0 ^ { - 8 }$ . The rows show, from top to bottom: the loss at each center; the preconditioned sharpness at each center; the raw sharpness at each center; the top six preconditioned eigenvalues at the discrete optimizer’s midpoint; the oscillation amplitude; the alignment of each flow’s oscillation with the discrete-time oscillation; the distance from each flow’s center to the discrete optimizer’s midpoint; and the norm of the second-moment estimate. Dashed vertical lines mark the handof, and dotted vertical lines mark the crossing into the edge of stability.

![](images/dfcb9e4566fcc8cef184dfbf4f5b86afb7b3883dac4f568b9c6a28f4967bfe35.jpg)  
Figure 18: Flow Comparison for Heavy Ball Momentum. MLP: $\eta = 0 . 3 .$ . CNN: $\eta = 0 . 7 .$ ViT: $\eta = 0 . 2$ . All with $\beta = 0 . 9$ . The rows show, from top to bottom: the loss at each center; the sharpness at each center; the top six eigenvalues at the discrete optimizer’s midpoint; the oscillation amplitude; the alignment of each flow’s oscillation with the discrete-time oscillation; and the distance from each flow’s center to the discrete optimizer’s midpoint. Dashed vertical lines mark the handof, and dotted vertical lines mark the crossing into the edge of stability.

![](images/2d3277dc7ae6d42a192c5883baf243b45bae990121290da8e695cbd84833f4f2.jpg)  
Figure 19: Flow Comparison for Nesterov Momentum. MLP: $\eta = 0 . 2 .$ CNN: $\eta = 0 . 4 .$ . ViT: $\eta = 0 . 2$ . All with $\beta = 0 . 9$ . The rows show, from top to bottom: the loss at each center; the sharpness at each center; the top six eigenvalues at the discrete optimizer’s midpoint; the oscillation amplitude; the alignment of each flow’s oscillation with the discrete-time oscillation; and the distance from each flow’s center to the discrete optimizer’s midpoint. Dashed vertical lines mark the handof, and dotted vertical lines mark the crossing into the edge of stability.

![](images/5a00623f68664db956a1249969641abd15763160d771c93dbdd76baa20cf36b2.jpg)  
Figure 20: Flow Comparison for Adam (with bias correction). MLP: $\eta = 1 0 ^ { - 5 }$ CNN: $\eta = 5 \times 1 0 ^ { - 5 }$ ViT: $\eta = 1 0 ^ { - 5 }$ . All with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , and $\epsilon = 1 0 ^ { - 7 }$ . The rows show, from top to bottom: the loss at each center; the preconditioned sharpness at each center; the raw sharpness at each center; the top six preconditioned eigenvalues at the discrete optimizer’s midpoint; the oscillation amplitude; the alignment of each flow’s oscillation with the discrete-time oscillation; the distance from each flow’s center to the discrete optimizer’s midpoint; and the norm of the second-moment estimate. Dashed vertical lines mark the handof, and dotted vertical lines mark the crossing into the edge of stability (none on the CNN, which never crosses).

![](images/b2e36aa3d42613beeb8eb943473c7ccb4e32967d981a7c8331dea9bff21462b3.jpg)  
Figure 21: Flow Comparison for AdamW (with bias correction). MLP: $\eta = 1 0 ^ { - 5 } , \zeta = 1 0$ CNN: $\eta = 5 \times 1 0 ^ { - 5 }$ , ζ = 40. ViT: $\eta = 1 0 ^ { - 5 } , \zeta = 1 0 .$ All with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , and $\epsilon = 1 0 ^ { - 7 }$ The rows show, from top to bottom: the loss at each center; the preconditioned sharpness at each center; the raw sharpness at each center; the top six preconditioned eigenvalues at the discrete optimizer’s midpoint; the oscillation amplitude; the alignment of each flow’s oscillation with the discrete-time oscillation; the distance from each flow’s center to the discrete optimizer’s midpoint; and the norm of the second-moment estimate. Dashed vertical lines mark the handof, and dotted vertical lines mark the crossing into the edge of stability.

## H.2 Cross-Entropy Loss

Figures 22 and 23 repeat the flow comparisons on the MLP with cross-entropy loss. We ran only gradient descent and RMSProp. For gradient descent, the edge of stability ends about 1 000 iterations after the crossing, when the sharpness falls back below the threshold. By the end of the comparison window, rod flow is twenty times closer to the iterates than the stable flow is, and central flow about twice as close. For RMSProp, both are more than fifty times closer.

![](images/0ab0c656693a1d0ff704a2ba562e4c7efd23978712a36e2dfb460be397a7ad62.jpg)  
Figure 22: Flow Comparison for Gradient Descent Under Cross-Entropy Loss. MLP: $\eta = 0 . 0 1 5$ . The rows show, from top to bottom: the loss at each center; the sharpness at each center; the top six eigenvalues at the discrete optimizer’s midpoint; the oscillation amplitude; the alignment of each flow’s oscillation with the discrete-time oscillation; and the distance from each flow’s center to the discrete optimizer’s midpoint. Dashed vertical lines mark the handof, and dotted vertical lines mark the crossing into the edge of stability.

![](images/6d705ee4edc1b3725c69325a454f968b36cd25f0a5548ef53cb80947cfb19e85.jpg)  
Figure 23: Flow Comparison for RMSProp Under Cross-Entropy Loss (no bias correction). MLP: $\eta = 1 0 ^ { - 5 } , \beta _ { 2 } = 0 . 9 9 , \epsilon = 1 0 ^ { - 8 }$ . The rows show, from top to bottom: the loss at each center; the preconditioned sharpness at each center; the raw sharpness at each center; the top six preconditioned eigenvalues at the discrete optimizer’s midpoint; the oscillation amplitude; the alignment of each flow’s oscillation with the discrete-time oscillation; the distance from each flow’s center to the discrete optimizer’s midpoint; and the norm of the second-moment estimate. Dashed vertical lines mark the handof, and dotted vertical lines mark the crossing into the edge of stability.

## H.3 Rod Turning

Figures 24 to 29 repeat the rod-turning experiments of Figures 6 and 7 for each optimizer on the MLP. Each experiment again starts with a warmup phase, but the warmup now continues past the onset of the edge of stability, because the goal is to capture the window in which there are multiple critical eigenvalues.

At the end of the warmup, rod flow and central flow are initialized from the discrete iterates, and the iterates and the flows then advance in lockstep for a couple thousand iterations. The first 300–500 of these iterations let the flows settle. The rest form the averaging span, throughout which there are exactly two critical eigenvalues (three in the bottom half of each figure). We cut the span into consecutive windows of T iterations, and over each window we average the extent matrix $\Sigma _ { \mathrm { r o d } }$ of rod flow and its discrete counterpart $a _ { t } \otimes a _ { t }$ . We compare each average with $\Sigma _ { \mathrm { c e n t r a l } }$ at the middle of its window using the participation ratio and the normalized Bures–Wasserstein distance. We also compare the turning rates of rod flow and of the discrete oscillation with those predicted by the linear law (derived in Sections 8.3 and 8.4).

In every run, the participation ratio of the time-averaged extent matrix completes 90% of its rise within 250 iterations, and at the widest window it is within 4% of that of central flow, for both the discrete iterates and rod flow. As the window widens, the normalized distance to $\Sigma _ { \mathrm { c e n t r a l } }$ falls from 0.5–1.0 for a single step to 0.05–0.17.

In every run, the measured turning rate tracks the linear law, with a correlation of at least 0.95. On average, the two agree to within 3% for both rod flow and the discrete oscillation, except that RMSProp’s discrete oscillation turns about 10% slower than predicted.

![](images/b94f17b793e90159709a68761532d9f989311e8fdffa22c18e6ab4b46ac7b70c.jpg)  
Figure 24: Rod Turning for Gradient Descent. MLP. Top: Two critical eigenvalues, $\eta = 0 . 0 5$ Bottom: Three critical eigenvalues, $\eta = 0 . 0 7 5$ . In each half, the first row compares the timeaveraged extent matrix with $\Sigma _ { \mathrm { c e n t r a l } }$ . The second row compares the turning rate with the linear law, as in Figure 7.

![](images/8ea81da47e02b0b7440eb29851a3b8fa35824d3420d5da580c70da6e036acdf8.jpg)  
Figure 25: Rod Turning for RMSProp (no bias correction). MLP, $\eta = 2 \times 1 0 ^ { - 6 } , \beta _ { 2 } = 0 . 9 9$ . Top: Two critical eigenvalues. Bottom: Three critical eigenvalues. In each half, the first row compares the time-averaged extent matrix with $\Sigma _ { \mathrm { c e n t r a l } }$ . The second row compares the turning rate with the linear law, as in Figure 7.

![](images/e2a17363ac9cfbb9b32421111d9c57bdf2746a694d3b619a1b2bda5a2d32bb0b.jpg)  
Figure 26: Rod Turning for Heavy Ball Momentum. MLP, $\eta = 0 . 3 , \beta = 0 . 9 .$ . Top: Two critical eigenvalues. Bottom: Three critical eigenvalues. In each half, the first row compares the time-averaged extent matrix with $\Sigma _ { \mathrm { c e n t r a l } }$ . The second row compares the turning rate with the linear law, as in Figure 7.

![](images/ee1a55f535a32ccb94393a6023c5c5218977244ecb82e14a765c0eae2ffad60c.jpg)  
Figure 27: Rod Turning for Nesterov Momentum. MLP, $\eta = 0 . 2 , \beta = 0 . 9$ . Top: Two critical eigenvalues. Bottom: Three critical eigenvalues. In each half, the first row compares the time-averaged extent matrix with $\Sigma _ { \mathrm { c e n t r a l } }$ . The second row compares the turning rate with the linear law, as in Figure 7.

![](images/c96788ea698e39be0501f567414d9c26c91a1fe2b3dc75309227a27eb3fd566a.jpg)  
Figure 28: Rod Turning for Adam (with bias correction). MLP, $\eta = 1 0 ^ { - 5 } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ Top: Two critical eigenvalues. Bottom: Three critical eigenvalues. In each half, the first row compares the time-averaged extent matrix with $\Sigma _ { \mathrm { c e n t r a l } }$ . The second row compares the turning rate with the linear law, as in Figure 7.

![](images/9eba29300280c9e4c39dc080b75302cbe7833f25271310026e04f505d49eb5a0.jpg)  
Figure 29: Rod Turning for AdamW (with bias correction). MLP, $\eta = 1 0 ^ { - 5 } , \zeta = 1 0 , \beta _ { 1 } = 0 . 9$ $\beta _ { 2 } = 0 . 9 9 9$ . Top: Two critical eigenvalues. Bottom: Three critical eigenvalues. In each half, the first row compares the time-averaged extent matrix with $\Sigma _ { \mathrm { c e n t r a l } }$ . The second row compares the turning rate with the linear law, as in Figure 7.

## H.4 Position and Momentum Extents

Figures 30 and 31 repeat the experiments of Section 8.4 for each momentum optimizer. Each run has the same warmup and averaging span as the corresponding two-critical-eigenvalue run in Appendix H.3. We ran only rod flow alongside the iterates, since central flow represents only the position oscillations.

Over this span, we compare the momentum extent with the position extent for both the iterates and rod flow. We use three metrics: the amplitude ratio between the two extents, the alignment between them, and the normalized Bures–Wasserstein distance between the time averages of their outer products, rescaled as in Figure 8. We also compare the rate of change of the relative momentum $\tilde { \gamma }$ with that predicted by the linear law.

In every run, the median amplitude ratio is within 5% of $2 / \eta ,$ its value on a period-two orbit. The median alignment is above 0.998 for both the iterates and rod flow, though both quantities fluctuate in bursts for Adam and AdamW. As the averaging window widens from one iteration to 250, the distance between the time-averaged extent matrices falls ten- to thirtyfold, except for Nesterov momentum, where it is below 0.01 even without time averaging. The rate of change of $\tilde { \gamma }$ also tracks the linear law, for both the iterates and rod flow.

![](images/8f50802890d83cc4d2c4a264c2e6572ea514780583110e7132f9d294a7ab0100.jpg)  
Figure 30: Position and Momentum Extents of the Momentum Optimizers. MLP, with two critical eigenvalues. Heavy ball: $\eta = 0 . 3 , \beta = 0 . 9$ . Nesterov: $\eta = 0 . 2 , \beta = 0 . 9$ . Adam (with bias correction): $\eta = 1 0 ^ { - 5 } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ . AdamW: as Adam, with $\zeta = 1 0$ . The columns show, from left to right: the ratio of the momentum amplitude to the position amplitude, with its value on a period-two orbit (dashed); the alignment between the position and momentum extents; and the distance between their time averages, against the width of the averaging window. For Adam and AdamW, the momentum extent is preconditioned.

![](images/1d90a463e7b8d39a809d601879251870e60dba07cea8833dc5d4854d84467aeb.jpg)  
Figure 31: Rate of Change of the Relative Momentum. Runs as in Figure 30. The relative momentum γ˜ is the momentum extent divided by the position amplitude. Left: The relative change of γ˜ per iteration, for the discrete oscillation and for rod flow, with the linear law for rod flow (dashed). Right: Measured against the linear law.

## H.5 Near Two-Periodicity

Regis and Chewi [2026a] suggested that period-doubling bifurcations might explain why their momentum rod flow failed to track the iterates at large $\beta _ { 1 }$ . Their rationale was that since rod flow is intended to model nearly two-periodic behavior, there is reason to suspect that it might fail if the iterates oscillated in four-cycles.

However, at the hyperparameters used in our experiments, theory suggests nearly two-periodic behavior. Fong and Yang [2026] analyze Adam at the edge of stability under the condition $\beta _ { 2 } > \beta _ { 1 } ^ { 2 }$ They show that when $\beta _ { 2 } \leq \beta _ { 1 } ^ { 2 }$ , Adam can have orbits of period four. Their experiments agree: across fully connected networks, ResNets, ViTs, LSTMs, and GPT-2, and across values of $\beta _ { 1 }$ and $\beta _ { 2 }$ , they found that much of the motion at the edge of stability is a fast, approximately period-two oscillation.

Since our setting satisfies $\beta _ { 2 } = 0 . 9 9 9 > \beta _ { 1 } ^ { 2 } = 0 . 8 1$ by a wide margin, we also expect nearly two-periodic behavior. To test this, we measure how close the discrete oscillation is to a two-cycle on the MLP when there are two critical eigenvalues. At every iteration, we record the oscillation envelope $a _ { t }$ in the basis of the top five eigenvectors of the (preconditioned) Hessian. Since $a _ { t }$ undoes the sign alternation of a two-cycle, the oscillation itself is $o _ { t } = ( - 1 ) ^ { t } a _ { t }$

Let $S ( f )$ be the power spectrum of $o _ { t }$ . To estimate S, we use Welch’s method on Hann windows of 256 iterations and sum over the coordinates. A two-cycle puts all of its power at $\begin{array} { r } { f = \frac { 1 } { 2 } } \end{array}$ cycles per iteration. Period doubling would add power at $\begin{array} { r } { f = \frac { 1 } { 4 } } \end{array}$ for a cycle of period four, at $\frac { 3 } { 8 }$ for period eight, and at $\frac { 7 } { 1 6 }$ for period sixteen.

![](images/5fa53e6bda142e8a6ac359dd47fb9cd9d0de85b27c0d39ce9fd657d65e932527.jpg)  
Figure 32: Power Spectrum of the Discrete Oscillation. MLP with two critical eigenvalues. Runs as in the top halves of Figures 24 to 29. Each panel shows the power spectrum of the oscillation of the iterates, normalized to unit area, so that the area under the curve over any range of frequencies is the share of the power in that range. The gray band marks the frequencies within 0.03 of $\frac { 1 } { 2 }$ , and the number in each panel gives the share of the power in this band (dark area). Dashed lines mark where period doubling to periods four, eight, and sixteen would add power.

Our runs agree with Fong and Yang [2026] (Figure 32). For every optimizer, the power is concentrated near $\begin{array} { r } { f = \frac { 1 } { 2 } } \end{array}$ . What departs from a steady two-cycle is the regularity of the oscillation. Less than half of Adam’s power lies within 0.03 of $\begin{array} { r } { f = \frac { 1 } { 2 } } \end{array}$ , compared with nearly all of it for gradient descent, RMSProp, heavy ball, and Nesterov momentum. The rest forms a broad hump extending down to $f \approx 0 . 4$ . This hump peaks near $\frac { 7 } { 1 6 }$ , but it is many times wider than a spectral line, so the oscillation also varies on a time scale of ten to thirty iterations, though not periodically. AdamW behaves similarly, but slightly more regularly.

## H.6 The Ring and the Onset

Figures 33 and 34 test the self-stabilization loop (119) for gradient descent and rod flow on an MLP with a single critical eigenvalue. Each experiment starts with a warmup phase in which the discrete iterates progressively sharpen. Rod flow and the stable flow are initialized from the discrete iterates 1 000 iterations before the edge of stability. The iterates and the flows then advance in lockstep for about 2 200 iterations, long enough to capture the onset of self-stabilization and to let the ring that follows it decay. We kick the oscillation 750–800 iterations after the crossing, and follow the ring that the kick sets of for the last 450 iterations. At each center, we measure $s , x , r ,$ and c at every iteration.

The orbit-average relation holds: over each orbit, $\left. s ^ { 2 } \right.$ is within 0.3% of $2 r / ( \eta c )$ for gradient descent and within 3% for rod flow. However, the orbits themselves are not closed. They spiral inward, decaying at a rate of 0.011–0.014 per iteration, and their period is 1.4–1.6 times the period $2 \pi / \sqrt { 2 \eta r }$ of the loop. Rod flow traces the same spiral as the iterates, reaching the onset slightly ahead of them and overshooting slightly less.

Why are the $x { - } s$ loops not closed? Unlike in an idealized Lotka–Volterra system, $r$ is not constant at the edge of stability. The oscillation pushes the center with velocity $- \frac { \eta } { 2 } s ^ { 2 } \nabla \lambda _ { 1 } .$ and the center’s own descent relaxes this push along each eigenvector $\hat { u } _ { i }$ at the rate $\eta \lambda _ { i }$ , which raises $^ r .$ In the steady-state edge-of-stability regime, the r of the iterates is about ten times that of the stable flow.

With this response, c in the linearized loop is replaced by a frequency-dependent coeficient $c _ { \mathrm { e f f } } ( z )$ , and the loop’s complex frequencies z solve

$$
z ^ { 2 } + \eta ^ { 2 } \langle s ^ { 2 } \rangle c _ { \mathrm { e f f } } ( z ) = 0 , \qquad c _ { \mathrm { e f f } } ( z ) = \sum _ { i } b _ { i } ^ { 2 } \frac { z } { z + \eta \lambda _ { i } } ,
$$

where $b _ { i } = \langle \nabla \lambda _ { 1 } , \hat { u } _ { i } \rangle$ . Directions much flatter than the ring leave their contribution to c intact, as the loop assumes. Much stifer directions relax the push before it acts, softening the restoring force and lengthening the period. Directions in between respond with a lag, which damps the ring. For the kicked ring, the roots of this equation match the measured period to within 5% and the decay rate to within 10% (Figure 34).

gradient descent, MLP, MSE: the self-stabilization loop

![](images/9b193fb6b9ac6906c1efe2ac10cc8bb9f4ac12ed971a7bc92e6efb761ebf1da6.jpg)

![](images/727b9a1994e1df04f692e781281eb8e73efe7abecf3f69ce4f79f576153a12a3.jpg)

![](images/3e3e8de5b004d0a23966eae8153540067e2de85b34d672a554bbb2ce43872919.jpg)

gradient descent, MLP, MSE: the self-stabilization loop  
![](images/a93fe023252fd25056046e576809fb1b261d7755587d7da1eda324cc1f65ca70.jpg)

![](images/9b410f5c7619d0880ab25cea0ffee8a905e2820be942083ff04930272cc37f0a.jpg)

![](images/5d50441f212ae3027c51c846e8867f0e78982f276a54074e1fbff39700b647ac.jpg)  
Figure 33: The Self-Stabilization Loop. Gradient descent and rod flow on an MLP. Top: $\eta = 0 . 0 5$ . Bottom: $\eta = 0 . 0 7 5$ . In each half, the left panel shows the phase portrait of the loop in the log-amplitude and the excess sharpness. The right panels show the log-amplitude and the progressive sharpening constant over time. The dashed curve is the progressive sharpening constant of the stable flow plus the response of the center to the oscillation, with no free parameters.

![](images/0c8dad1db76b802a26cd7f1d6cbc46523c4ad38ac9305a22a033b0a29ab3fbc0.jpg)

![](images/97ffe8d429911bdcaa7d500559b37d9aff8f2aed0eb52d7e7f292e0b375b6723.jpg)

![](images/94ef37b66062d8c721e2a3e17bbcd6a007d578e9920a48f8b2c7a9cb79459534.jpg)

gradient descent, MLP, MSE: the kicked ring  
![](images/a64f8571e697510fe82aae7ce62ab44568fda8af18d4ff1c2f10b30fb4b74e89.jpg)

![](images/d95b5bab05512af5b7b1edcb6137bbe96d7649d6e69ea99b541ea804824b30aa.jpg)

![](images/fa1ef93435c1e9208905b88c98e696393764cb4387b78d39202fac185e7ce7b5.jpg)  
Figure 34: The Kicked Ring. Gradient descent and rod flow on an MLP, kicked by multiplying the amplitude of the oscillation by 1.2. Top: $\eta = 0 . 0 5$ . Bottom: $\eta = 0 . 0 7 5$ . In each half, the left panel shows the phase portrait, with dots marking the state at the kick and arrows marking the kick. The right panels show the log-amplitude and the progressive sharpening constant over time. The dashed curves are predictions. For the log-amplitude, only the frequency and the decay rate are predicted; the amplitude, phase, and trend are fitted.