# Sequential Learning-Based Physics-Informed Neural Network Modeling for Parameter Estimation on a Power Converter

Samuela Rokocakau

ABB Corporate Research Center

Vaster¨ as, Sweden˚

Email: samuela.rokocakau@abb.se.com

Giulia Tresca

Dept. of Electrical, Computer and Biomedical Engineering

University of Pavia, Pavia, Italy

Email: giulia.tresca@unipv.it

Panagiotis Kakosimos

ABB Corporate Research Center

Vaster¨ as, Sweden˚

Email: panagiotis.kakosimos@se.abb.com

Pericle Zanchetta

Dept. of Electrical, Computer and Biomedical Engineering

University of Pavia, Pavia, Italy

Email: pericle.zanchetta@unipv.it

Abstract— A staged physics-informed neural network (PINN) is proposed for identifying the inductance, capacitance and parasitic resistances of a DC buck converter from time-domain data. Gradients are released in the order of $L  C  r _ { L }  r _ { C } ,$ a sequence derived from sensitivity analysis. The five-stage schedule recovers L and C to within 5% and the series ESR to 10%, while a three-stage variant attains the same accuracy 11% faster. An 8<sup>4</sup> factorial sweep confirms that the trained PINN remains accurate for simultaneous ±8 % perturbations of all four components, achieving a waveform RMSE of $2 . 6 \times 1 0 ^ { - 2 }$ over 5 ms—an order-of-magnitude better than a simultaneous-training baseline.

Index Terms—Physics Informed Neural Network, fault diagnostics, Buck Converter, neural network, power converter, parameter estimation, parasitic resistance, condition monitoring

## I. INTRODUCTION

electric vehicles, smart grids, and mission-critical systems. In such systems, accurate knowledge of component values and their parasitic elements is essential for condition monitoring, fault detection, predictive maintenance, and high-performance control [1]–[5]. Power converters, in particular, rely on accurate modeling to minimize downtime, reduce financial losses, and improve operational safety and efficiency.

Conventional system identification methods fall broadly into parametric and non-parametric categories. Non-parametric techniques often assume steady-state operation and use frequency domain measurements, requiring complex setups and offline post-processing [6]. These methods also require intrusive excitation signals that may lead to system instability.

In contrast, parametric identification approaches — especially data-driven ones — offer flexibility by learning system dynamics from time-domain input–output data. These methods can operate during normal converter operation and avoid the need for frequency-response acquisition [7]–[9].

Recent work has emphasized the importance of identifying multiple parameters simultaneously, especially in real-world scenarios where idealized component assumptions no longer hold [10]. The rise of digital twins, state observers, and neural models illustrates a broader trend of using real-time, embedded estimators for live converter health and performance assessment.

Physics-Informed Neural Networks (PINNs) combine the data-driven learning power of neural networks with embedded physical constraints derived from differential equations. Unlike black-box models, PINNs integrate system-level knowledge to enforce physically consistent outputs. This makes them attractive for learning hidden dynamics and parameters of systems governed by ordinary differential equations, such as power converters [8], [11], [12].

Several studies [10], [13], [14] have successfully applied PINNs for estimating circuit parameters in switched-mode power supplies. These include efforts to model unknown topological durations, parasitic behaviors, and transient response—all while maintaining interpretability and structure imposed by physics.

Despite their promise, many PINN approaches in power electronics rely on complex solvers like Runge–Kutta, require full-state observability, or lack modularity for staged training. Moreover, few works isolate and study how staged parameter unfreezing affects convergence, robustness, and generalization to unseen component values.

This work presents a simplified and interpretable sequential PINN workflow tailored to a DC buck converter. It estimates four physical parameters—inductance, capacitance, inductor ESR, and capacitor parallel resistance—using staged learning. More specifically, we propose a modular training strategy for PINNs, employing both five-stage and three-stage sequential schemes that progressively activate trainable parameters to enhance convergence and interpretability. Our modeling framework incorporates parasitic resistance effects in both the inductor and capacitor branches, enriching the fidelity of the governing ODE system. To assess robustness, we conduct synthetic parameter sweeps and introduce noise-injected test cases, analyzing RMSE trends and residual compensation behavior. We further compare full and staged training strategies, highlighting trade-offs in convergence speed, accuracy, and generalizability. Finally, we present a suite of quantitative and qualitative tools for evaluating PINN performance, including RMSE heatmaps, trajectory comparisons, and physics residual analyses.

![](images/6ae74dc96dcc862f1fdd8c8de9d8088bd632f87950c48ec74037cd899dd9b671.jpg)  
Fig. 1: Buck converter with series inductor ESR $( r _ { L } )$ , switch on-resistance (r<sub>DS,on</sub>),and capacitor leakage (r<sub>C</sub>) explicitly modeled

## II. SYSTEM OVERVIEW

Figure 1 depicts a non-ideal buck converter used as the plant for identification. A high-side MOSFET switch S modulates the input source $V _ { i n }$ with duty ratio $S ( t ) \in \{ 0 , 1 \}$ , and its channel on-resistance is represented by $r _ { D S , \mathrm { o n } }$ . When $S$ is open, the free-wheel diode $D$ allows the inductor current $i _ { L }$ to return to ground. Energy is stored in the series inductor $L$ and shunt capacitor $C ,$ whose parasitic resistances are explicitly modeled: $r _ { L }$ represents copper loss in the coil, while $r _ { C }$ represents dielectric leakage across the capacitor plates. The load is a purely resistive element $( R _ { l o a d } )$ , and the output voltage $( V _ { o u t } )$ is measured between the positive rail and ground. Adding $r _ { D S , \mathrm { o n } } , r _ { L }$ , and $r _ { C }$ transforms the model from an ideal, loss-less prototype to a realistic powerstage description capable of predicting efficiency, wave-shape distortion, and damping effects observed in hardware.

The buck converter is written in continuous-time state–space form:

$$
{ \dot { \pmb x } } ( t ) = { \pmb f } ( { \pmb x } ( t ) , u ( t ) , { \pmb \theta } ) ,\tag{1a}
$$

$$
\mathbf { \boldsymbol { x } } ( t ) = \left[ \begin{array} { l l } { i _ { L } ( t ) } \\ { v _ { O } ( t ) } \end{array} \right] , \qquad \mathbf { \boldsymbol { u } } ( t ) = \boldsymbol { S } ( t )\tag{1b}
$$

where $i _ { L }$ is the inductor current, $v _ { O }$ the output-capacitor voltage and $S ( t ) \in \{ 0 , 1 \}$ the duty ratio of the power switch.

The parameter vector $\pmb \theta = \{ L , C , r _ { L } , r _ { C } , R _ { \mathrm { l o a d } } , V _ { \mathrm { i n } } \}$ collects the inductance, capacitance, series coil resistance, parallel leakage resistance, load resistance and input voltage, respectively. Applying Kirchhoff’s laws under the usual assumptions of an ideal switch and negligible switch on-resistance yields the bilinear dynamics

$$
\begin{array} { r l } & { \dot { \iota } _ { L } = \frac { S V _ { \mathrm { i n } } - v _ { O } - r _ { L } \iota _ { L } } { L } , } \\ & { \dot { v } _ { O } = \frac { \iota _ { L } - v _ { O } / R _ { \mathrm { l o a d } } - v _ { O } / r _ { C } } { C } . } \end{array}\tag{2}
$$

Equations (1)-(2) describe the hard physics included in the latter phases of the sequential physics-informed neural network (PINN). During Stage 1, the neural network approximates the state derivatives on its own, allowing it to absorb measurement noise and unmodeled effects. The PINN recovers physical parameters θ while maintaining the interpretability of a standard state-space representation by imposing (2) as a differentiable constraint in Stage 2 and beyond.

The approach utilized in this work to construct the PINN model is a sequential approach where each stage of the training process allows for better visibility and control over the process. Given in Fig.2 demonstrate the workflow for the neural network modeling In PINNs, the loss functions used is related to the residual between the neural network predictions and the true vales as defined by the mean square error. This function is defined as follows:

Stage 1 — Data–only Loss:

$$
\mathcal { L } _ { \mathrm { d } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \Bigl [ w _ { i } \bigl ( \hat { i } _ { L , s } ^ { ( n ) } - \dot { i } _ { L , s } ^ { ( n ) } \bigr ) ^ { 2 } + w _ { v } \bigl ( \hat { i } _ { O , s } ^ { ( n ) } - \dot { v } _ { O , s } ^ { ( n ) } \bigr ) ^ { 2 } \Bigr ] ,\tag{3}
$$

where hats denote network predictions, the subscript s indicates variables scaled to zero mean and unit variance, w<sub>i</sub>, w<sub>v</sub> are balancing weights, and N is the mini-batch size.

Stage 2 — Combined Data & Physics Loss:

$$
\mathcal { L } _ { \mathrm { t o t } } = \mathcal { L } _ { \mathrm { d } } + \lambda _ { \mathrm { p h y s } } \mathcal { L } _ { \mathrm { p } } ,\tag{4a}
$$

$$
\mathcal { L } _ { \mathrm { p } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \Bigl [ w _ { i } ^ { \mathrm { p } } r _ { i , s } ^ { 2 ( n ) } + w _ { v } ^ { \mathrm { p } } r _ { v , s } ^ { 2 ( n ) } \Bigr ] ,\tag{4b}
$$

$$
f _ { i } = \frac { S V _ { \mathrm { i n } } - v _ { O } - r _ { L } i _ { L } } { L } , \quad f _ { v } = \frac { i _ { L } - v _ { O } / R _ { \mathrm { l o a d } } - v _ { O } / r _ { C } } { C } .\tag{5}
$$

with scaled residuals

$$
r _ { i , s } = \frac { \hat { i } _ { L } - f _ { i } } { \sigma _ { i _ { L } } } , \qquad r _ { v , s } = \frac { \hat { v } _ { O } - f _ { v } } { \sigma _ { v _ { O } } } ,\tag{6}
$$

the empirical standard deviations used for nondimensionalization.

In (5), the symbols $f _ { i }$ and $f _ { v }$ denote the analytic righthand sides of the state-space equations for the buck converter, i.e. the deterministic expressions that give the ideal time derivatives of the inductor current and the capacitor voltage for a given state $( { \dot { \imath } } _ { L } , v _ { O } )$ , duty ratio S, and parameter set $\{ L , C , r _ { L } , r _ { C } , R _ { \mathrm { l o a d } } , V _ { \mathrm { i n } } \}$

$f _ { i }$ is the theoretical value of $i _ { L } ,$ obtained from Kirchhoff’s voltage law around the power stage.

![](images/680451028b2a3a83c8e8fbfdadf70a1a18f19b88878fa545acb5960487fc7cf7.jpg)  
Fig. 2: PINN modeling workflow.

• $f _ { v }$ is the corresponding theoretical value of $\dot { v } _ { O }$ , derived from the capacitor-current equation.

These expressions serve as physics targets in the loss function: the PINN’s derivative predictions are driven to match $\hat { i } _ { L } \approx f _ { i }$ and $\hat { \dot { v } } _ { O } \approx f _ { v }$ once the physics term is activated. The hyper-parameter $\lambda _ { \mathrm { p h y s } }$ weights the physics term against the data term. In the construction of the PINN model explored in this study, the data-driven and physics-driven components of the model are slightly adjusted, with parametric estimation for the converter used from a monitoring perspective using a sequential training process.

Table I shows the models used in this work to contextualize the technique. The models use the PINN architecture’s sequential training approach, beginning by optimizing the data-driven component of the overall to enable generalized learning system behavior. The initial stage of training focuses on the data-loss consideration of the overall loss function while determining the ODE of the overall buck converter for the specific scenario. This consideration is achieved by ”freezing” the learning gradients of the observed parameters on the PDE. It is worth mentioning that for the scenarios for 2 stage training, singular parasitic resistance estimation, respectively the results were maintained good estimation performance but remained simplistic preliminary configuration leading to the other scenarios mentioned in this work.

The input parameters for the NN model is initial values for inductor current, $i _ { L 0 } ,$ , the output voltage $v _ { L 0 }$ and the Duty Cycle, $S ( t ) \in \{ 0 , 1 \}$ . Given in Table II are the true values used during the model training. Figure 3 shows the neural-network trajectory (dashed red) overlaid on the measured inductor current and capacitor-voltage traces (solid blue) from the synthetic dataset. Because the network has only experienced the mean-squared data loss of Stage 1, the close agreement in this panel demonstrates that the learnt derivatives integrate forward in time with no substantial drift, even across the whole 5 ms window.

Figure 4 shows a complementary point-wise examination. Each scatter plot compares the network’s instantaneous derivative prediction to the analytic right-hand side based on the true component values; the $4 5 ^ { \circ }$ reference line indicates the best-fit line for perfect assimilation between the two datapoints. The strong clustering around this line implies that the Stage 1 model well reflects the local slopes of both state variables. Together, the two figures show that the solely datadriven surrogate accurately reproduces the converter’s dynamics, establishing a solid baseline before physics penalties are included in further training rounds.

Table III shows the training hyperparameter used to realize the models used in this work. As shown, the random seed for the training are kept the same for the two scenarios considered to maintain comparability. The Adam optimizer is used to optimize the loss function for the model with gradient clipping enabled to avoid spike during gradient learning at every stage considered.

## III. RESULTS AND DISCUSSION

In this section, the training schedules and the parameter sensitivity sweep of component uncertainty for the trained model are provided to demonstrate the efficacy of the stagedapproach as described earlier.

## A. Training Schedules

Tables Va and Vb describe the parameter trajectories for the five-stage and three-stage sequential-PINN schedules, respectively. During the first stage of the experiment, the network is trained using the training data to provide a universal approximation for the overall system based on the ODE. In the next stages of learning, the inductance and capacitance are retained at their inferior initial estimates ( 50% and $\approx - 8 0 \%$ 2 respectively), while both parasitic resistive parameters are similarly set as untrained estimate values. Therefore, freezing the PDE gradients from the training procedure.

In the five-stage training schedule case, the inductance estimate is reduced to 2.08 mH (+4% error) by releasing the inductance learning gradient in Stage 2 for 50 k training epochs, with no effect on the other values. In Stage 3, the capacitance learning gradient is unfrozen and exactly converges to the true value of $5 ~ \mu F$ with an error of 240 $p F ( + 0 . 0 5 \% )$ , leaving the previously estimated inductance unaltered. The parasitic resistance for inductance and capacitance learning gradient is allowed in Stages 4 and 5, respectively. In Stage 4, the series coil resistance is modified to 0.552 Ω ( +10%), and the parallel leakage resistance, $r _ { C }$ pathway is released for the last training stage. This resulted in a reduction in the $r _ { C }$ error from 67% to 20% demonstrating that the gradual release of gradients keeps earlier parameters from drifting.

TABLE I: Sequential-training stages required for each identification scenario
<table><tr><td>Scenario</td><td>Stages of learning</td></tr><tr><td>Parasitic inductor resistance  $( r _ { L }$  only)</td><td>2</td></tr><tr><td>Parasitic capacitor resistance (rc only)</td><td>2</td></tr><tr><td>Both and  $r _ { C }$ </td><td>3*5</td></tr><tr><td> $r _ { L }$ </td><td></td></tr></table>

TABLE II: Buck Converter Parameter True Values
<table><tr><td>Component Parameter</td><td>True Values</td></tr><tr><td>Inductor,  $\overline { { L } }$ </td><td>2mH</td></tr><tr><td>Capacitor,  $C$ </td><td> $5 \mu F$ </td></tr><tr><td>Parasitic Inductor Resistance,  $r _ { L }$ </td><td>0.5Ω</td></tr><tr><td>Parasitic Capacitor Resistance,  $r _ { C }$ </td><td>30kΩ</td></tr></table>

Stage 1 NN Simulation vs True Data  
![](images/16f5c29d591e45fd6fe6481be35bf20d65c5cfb26ef771e2bcfe878f334f3a35.jpg)

![](images/f7032a9cd25fcf3d5a35efc74ec79512bb82ec777d45658f64bddb4835fc8036.jpg)  
Fig. 3: Stage 1 - True Data vs NN Simulation Plots

![](images/57ccac310b6152c1ee80c551e0dfcc1ba6454fd537465abbd6aa310cb5236e44.jpg)

![](images/669fbe40c772cff38042d8a19c96e6f7ffdc8930b18a1b88b701fef3c9df45eb.jpg)  
Fig. 4: Stage 1 - True Data vs NN Simulation Plots

The three-stage training schedule compresses the technique, combining L and C into a single learning phase (stage 2) and the two resistive parameters into a second (Stage 3), maintaining the same training strategy at Stage 1 as above. During Stage 2, in 50, k epochs, the inductance is decreased to 2.08 mH and the capacitance to 5.00024 $\mu F$ by the joint release of $L$ and $C$ following a 50 k-epoch data-only pre-fit in Stage 1. This implies that fine-grained staging isn’t strictly required for convergence.

Figures 5 and 6 compare the convergence behaviour of the five-stage and three-stage training schedules. In both cases the log-scaled loss drops steeply during the data-only pre-fit, indicating that the neural network alone can approximate the measured derivatives with little optimization effort. For the five-stage run the loss then decreases in a series of smaller, step-like reductions as $L , ~ C , ~ R _ { L }$ and $R _ { C }$ are sequentially unfrozen; each parameter release introduces a brief spike, followed by rapid settling to a lower plateau.

By contrast, the three-stage curve shows only two such plateaux, because $L$ and $C$ are learned together in Stage 2, and the two parasitics are learned together in Stage 3. Although the three-stage loss reaches its final plateau about 10 % sooner in time, the five-stage trace reveals a more gradual, monotonic descent, suggesting a smoother optimization path and better interpretability of intermediate parameter values.

The residual error is essentially limited to the parallel leakage resistance in both schedules. The high impedance of $r _ { C }$ creates a weak signature in the state derivatives, resulting in a tiny gradient that is easily disguised by numerical noise. The remaining 20% uncertainty in $r _ { C }$ is unimportant for loop-shaping and transient-response investigations, while the accurate recovery of L, C, and $r _ { L }$ is enough for extensive efficiency and control analysis. As a result, a viable workflow begins with the compact three-stage schedule to obtain reliable core parameters and advances to the five-stage sequence only when a higher leakage resistance boundary is required.

![](images/56064db95822ac8d1db355012231646c2402d5a8e9028c921ca4e3724272e1d1.jpg)  
Fig. 5: Training loss per stage (five-stage schedule).

TABLE III: Common hyper-parameters used in both training schedules
<table><tr><td>Setting</td><td>Value / Rationale</td></tr><tr><td>Optimizer</td><td>Adam  $\left[ \beta _ { 1 } = 0 . 9 , \ \beta _ { 2 } = 0 . 9 9 9 \right]$ </td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 3 }$  during data  $\mathrm { p r e } { \cdot } \mathrm { f i t } ; 1 { \times } 1 0 ^ { - 4 }$  for every parameter-release stage</td></tr><tr><td>Batch size</td><td>Full record (5 001 time-samples, 5 ms,  $\Delta t = 1 \mu \mathrm { s } )$ </td></tr><tr><td>Gradient clip</td><td> $\lVert \nabla \theta \rVert _ { 2 } \leq 1 . 0$  to avoid spikes when a parameter is unfrozen</td></tr><tr><td>Input scaling</td><td>z-score on  $i _ { L }$   $v _ { O } ,$  and duty ratio  $S$  (mean 0, std 1)</td></tr><tr><td>Target scaling</td><td>Same z-score applied to  $\dot { i } _ { L }$  and  $\dot { v } _ { O } ;$  data-loss weights  $w _ { i } = w _ { v } = 1$ </td></tr><tr><td>Physics residual scale</td><td>Residuals divided by  $\begin{array} { r l } { \sigma _ { \dot { i } _ { L } } , } & { { } \sigma _ { \dot { v } _ { O } } ; } \end{array}$  physics weight  $\lambda _ { \mathrm { p h y s } } = 1$ </td></tr><tr><td>Weight initialization</td><td>He-uniform for NN layers; log- parameters for  $L , C , R _ { L } , R _ { C }$  set to</td></tr><tr><td>Random seed</td><td>nominal guesses Fixed to make three- and five-stage scenarios run directly comparable</td></tr></table>

TABLE IV: Parameter-sweep grid used for model-robustness analysis
<table><tr><td>Offset (%) L (mH)</td><td> $C _ { \mathrm { \Omega } } ( \mu \mathrm { F } )$   $r _ { L } ~ ( \Omega )$ </td><td>(kΩ)  $r _ { C }$ </td></tr><tr><td>-8 1.84</td><td>4.60 0.460</td><td>27.6</td></tr><tr><td>-6 1.88</td><td>4.70</td><td>0.470 28.2</td></tr><tr><td>-4 1.92</td><td>4.80</td><td>0.480 28.8</td></tr><tr><td>-2 1.96</td><td>4.90</td><td>0.490 29.4</td></tr><tr><td>+2 2.04</td><td>5.10 0.510</td><td>30.6</td></tr><tr><td>+4 2.08</td><td>5.20</td><td>0.520 31.2</td></tr><tr><td>+6 2.12</td><td>5.30 0.530</td><td>31.8</td></tr><tr><td>+8 2.16</td><td>5.40</td><td>0.540 32.4</td></tr></table>

![](images/fc06a1c710c238d5ebbbb7fadb17e2171b13fe96fcd7c0e0ae0ee61d41db88da.jpg)  
Fig. 6: Training loss per stage (three-stage schedule).

Table IV depicts the eight-point grid used in the sensitivity sweep. To achieve offsets of $\pm 2 \% , \ \pm 4 \% , \ \pm 6 \%$ and $\pm 8 \%$ , each component value is symmetrically perturbed about its nominal setting in uniform 2 % increments. The test points’ inductance vector spans 1.84-2.16 mH, capacitance vector spans $4 . 6 0 – 5 . 4 0 ~ \mu F _ { \cdot }$ , series coil resistance vector spans 0.460-0.540 Ω, and parallel leakage resistance vector spans 27.6-32.4 kΩ. The study assesses $\bar { 8 ^ { 4 } }$ = 4096 different parameter choices due to the factorial sweep, which is adequate to demonstrate second-order sensitivity effects without incurring high processing costs.

## B. Sensitivity of the trained PINN to component uncertainty

As show in in Fig. 7 blue, orange, and green curves correspond to the current channel $( i _ { L } – \mathbf { R M S E } )$ , the voltage channel (v -RMSE), and their sum (Total RMSE), respectively. It summarizes the $8 ^ { 4 } \cdot$ -point sensitivity experiment by combining each three-parameter sub-grid into a single root-mean-square error (RMSE) trace for the remaining parameter.

Inductance sweep. Adjusting L between 1.84 2.16 mH results in a unique bowl shape of the total RMSE. It reaches a minimum of $2 . 6 \times 1 0 ^ { - 2 }$ at the true value of $L = 2 . 0 0$ mH and quickly increases to the $\pm 8 \%$ extremes. The curve is dominated by the voltage component, and the current error is below $8 \times 1 0 ^ { - 3 }$ , indicating that the network’s voltage prediction is particularly susceptible to inductance bias.

Capacitance sweep. Varying C between $4 . 6 0 \mathrm { - } 5 . 4 0 \mu \mathrm { F }$ induces a stable, nearly parabolic trend in total RMSE. Throughout the range, the RMSE fluctuation is less than $5 \times 1 0 ^ { - 3 }$ , demonstrating remarkable stability. This confirms that increasing precision leads to declining gains in prediction fidelity after C is predicted within a few percent.

Series ESR sweep. The total error for $r _ { L }$ drops monotonically in the range of $0 . 4 6 0 - 0 . 5 4 0 \Omega ,$ , reaching a minimum at $r _ { L } ~ = ~ 0 . 5 2 \Omega$ . Overestimating significantly dampens the converter and improves the fit, whereas underestimating can inflate the RMSE by about 40%.

![](images/f22436aa33d57ee7fa238b5a29ff548b736403230d62e75e73ae9674b4cff3e8.jpg)  
Fig. 7: RMSE Plots for Parameter Sweep Grid

Leakage resistance sweep. Changing $r _ { C }$ between 27.6 and 32.4 kΩ leaves all traces essentially flat; a full 8% perturbation moves the total RMSE by less than 1 %. The leakage path carries micro-ampere currents under normal load, because its exact value has negligible impact on one-step prediction.

The suggested sequential-learning framework’s parameter release order is supported by the sensitivity data in Fig. 7. The monotonic fall in the series ESR and the prominent Ushape obtained when scanning the inductance illustrate that a tiny bias in either $L$ or $r _ { L }$ directly generates a large voltagechannel RMSE. The capacitance curve is nearly flat, while the leakage-resistance trace is nearly horizontal. This suggests that moderate faults in $C$ or $r _ { C }$ have minimal impact on one-stepahead performance.

These gradient magnitudes provide insight into the empirical behavior of the training plan. The optimizer can quickly descend without disturbing the current-channel fit formed during the data-only stage by releasing L, which removes the stiffest direction in the loss surface. Adding $r _ { L }$ improves the voltage prediction by dampening the LC pole and stabilizing the inductance estimate. The less sensitive parameters $C$ and $r _ { C }$ are only unfrozen when the dominant quantities have stabilized. Their fine-tuning produces small but measurable improvements in steady-state accuracy.

In summary, the sweep confirms that the sequential order $\{ L \ \to \ r _ { L } \ \to \ C \ \to \ r _ { C } \}$ mirrors the natural hierarchy of parameter influence, thereby reducing optimization stiffness and accelerating convergence while preserving final fidelity. The PINN reproduces the full-order response of the top three buck converter parameter sweep scenarios, as shown in Fig. 8, even when all four components are affected concurrently. After an initial 0.3 ms rise, the predicted inductor current and output voltage for the three lowest-RMSE configurations are visually indistinguishable from the reference traces, and the steady state ripple exhibits no phase lag. The results demonstrate that progressively determined parameters are generalizable to unseen operating points that differ by up to 8% from the nominal values. The loss surface is steepest along the inductance and ESR vectors, but virtually flat along the capacitance and leakage paths, according to the sensitivity research.

(a) Five-Stage Training Scenario
<table><tr><td>Stage</td><td>Epochs</td><td> $L ~ ( \mathrm { m H } )$ </td><td> $C _ { \mathrm { \Omega } } \left( \mu \mathrm { F } \right)$ </td><td> $r _ { L } ~ ( \Omega )$ </td><td> $r _ { C } ~ ( \mathrm { k } \Omega )$ </td><td>Err.  $L , C$ </td><td> $\mathbf { E r r } . \ r _ { L } , r _ { C }$ </td></tr><tr><td>1 (data)</td><td>20k</td><td>1.00</td><td>1.00</td><td>0.10</td><td>10</td><td> $- 5 0 \% , - 8 0 \%$ </td><td>−80 %, −67 %</td></tr><tr><td>2 (+L)</td><td>50k</td><td>2.08</td><td>1.00</td><td>0.10</td><td>10</td><td>+4 %, −80 %</td><td>−80 %, –67 %</td></tr><tr><td>3 (+C)</td><td>50k</td><td>2.08</td><td>5.00024</td><td>0.10</td><td>10</td><td>+4 %, +0.05 %</td><td>−80 %, −67 %</td></tr><tr><td>4 (+rL)</td><td>50k</td><td>2.08</td><td>5.00</td><td>0.552</td><td>10</td><td>+4 %, +0.05 %</td><td>+10 %, -67 %</td></tr><tr><td> $5 \ ( + r _ { C } )$ </td><td>50k</td><td>2.08</td><td>5.00</td><td>0.552</td><td>24</td><td>+4 %, +0.05 %</td><td>+10 %, −20 %</td></tr></table>

(b) Three-Stage Training Scenario
<table><tr><td>Stage</td><td>Epochs</td><td>L (mH)</td><td> $C _ { \mathrm { \Omega } } ( \mu \mathrm { F } )$ </td><td> $r _ { L } ~ ( \Omega )$ </td><td> $r _ { C } ~ ( \mathrm { k } \Omega )$ </td><td>Err.  $L , C$ </td><td> $\mathbf { E r r } . \ r _ { L } , r _ { C }$ </td></tr><tr><td>1 (data)</td><td>60k</td><td>1.00</td><td>1.00</td><td>0.10</td><td>10</td><td> $- 5 0 \% , - 8 0 \%$ </td><td> $- 8 0 \% , - 6 7 \%$ </td></tr><tr><td> $2 \ ( + L , C )$ </td><td>50k</td><td>2.08</td><td>5.00024</td><td>0.10</td><td>10</td><td>+4 %, +0.05 %</td><td>−80 %, −67 %</td></tr><tr><td> $3 \ ( + r _ { L } , r _ { C } )$ </td><td>45k</td><td>2.08</td><td>5.00</td><td>0.552</td><td>24</td><td> $+ 4 \% ,$  +0.05 %</td><td> $+ 1 0 \% , - 2 0 \%$ </td></tr></table>

![](images/30a4e8849591bd00b15f467c48a5b47777ccfaffb481718fb39acfe692685b4c.jpg)

Config 1: vO (L=0.00204, C=5.1e-06, RL=0.53, RC=32400.0)  
![](images/de664646f23724a346681c70cd3a20b12c872628290253f04a3e18474e66df0b.jpg)

Config 2: iL (L=0.00204, C=5.1e-06, RL=0.53, RC=29400.0)  
![](images/56cd8b0490741dbe2e2109b27e67f095befcf7941ac15ac900b16ab4f95f7758.jpg)  
Config 3: iL (L=0.00204, C=5.1e-06, RL=0.53, RC=30600.0)

Config 2: vO (L=0.00204, C=5.1e-06, RL=0.53, RC=29400.0)  
![](images/a594dc159d8ddf1ac516e7466916c0021b61ad986ca2d3752d91be1ed63f2101.jpg)

![](images/a7f7372cf8feed62b441e10a4f390e49618ad786040497e04b40ba839bd60a1b.jpg)

Config 3: vO (L=0.00204, C=5.1e-06, RL=0.53, RC=30600.0)  
![](images/67c10e8ebb24e252693103628f5f7d240ae62665c88341e069f1f4433e031756.jpg)  
Fig. 8: PINN (red) versus truth (blue) for the three best parameter sets.

## IV. CONCLUSION

In this work, the investigated five-stage schedule $( L \ \to$ $r _ { L }  C  r _ { C } )$ reduced the leakage-resistance bias from - 67% to -20%, while driving the neural network to sub-5% error on L and $C ,$ and to 10% on $r _ { L } .$ . A three-stage compression produced the same precision at an 11% faster rate, proving that steps can be merged when time and resources are limited. No grid search or human tuning is required because each block is trained with a single, unit data-to-physics weight, and the parameter traces resolve monotonically. As a result, the staged PINN outperforms a one-shot fit in terms of accuracy, drift-free parameter estimations, computation time, and hyperparameter management. Finally, the sensitivity analysis revealed that a PINN model could potentially exhibit a stable behavior in tracking parameter changes.

[1] F. Hosseinabadi, S. Chakraborty, S. K. Bhoi, G. Prochart, D. Hrvanovic, and O. Hegazy, “A comprehensive overview of reliability assessment strategies and testing of power electronics converters,” IEEE Open Journal of Power Electronics, vol. 5, pp. 473–512, 2024.

[2] S. S. Williamson, A. K. Rathore, and F. Musavi, “Industrial electronics for electric transportation: Current state-of-the-art and future challenges,” IEEE Transactions on Industrial Electronics, vol. 62, no. 5, pp. 3021– 3032, 2015.

[3] S. Yang, D. Xiang, A. Bryant, P. Mawby, L. Ran, and P. Tavner, “Condition monitoring for device reliability in power electronic converters: A review,” IEEE Transactions on Power Electronics, vol. 25, no. 11, pp. 2734–2752, 2010.

[4] F. Wu, K. Chen, G. Qiu, H. Ying, H. Sheng, and Y. Wang, “Detectability based data-driven fault diagnosis method for multiple device faults of converters,” IEEE Transactions on Power Electronics, vol. 40, no. 4, pp. 5983–5998, 2025.

[5] P. Wen, Z.-S. Ye, Y. Li, S. Chen, P. Xie, and S. Zhao, “Physics-informed neural networks for prognostics and health management of lithium-ion batteries,” IEEE Transactions on Intelligent Vehicles, vol. 9, no. 1, pp. 2276–2289, 2024.

[6] M. Ahmeid, M. Armstrong, S. Gadoue, M. Al-Greer, and P. Missailidis, “Real-time parameter estimation of dc–dc converters using a self-tuned kalman filter,” IEEE Transactions on Power Electronics, vol. 32, no. 7, pp. 5666–5674, 2017.

[7] Y. Peng, S. Zhao, and H. Wang, “A digital twin based estimation method for health indicators of dc–dc converters,” IEEE Transactions on Power Electronics, vol. 36, no. 2, pp. 2105–2118, 2021.

[8] Y. Gao, S. Wang, T. Dragicevic, P. Wheeler, and P. Zanchetta, “Artificial intelligence techniques for enhancing the performance of controllers in power converter-based systems—an overview,” IEEE Open Journal of Industry Applications, vol. 4, pp. 366–375, 2023.

[9] S. Zhao, F. Blaabjerg, and H. Wang, “An overview of artificial intelligence applications for power electronics,” IEEE Transactions on Power Electronics, vol. 36, no. 4, pp. 4633–4658, 2021.

[10] Z. She, Y. Liu, W. Cao, and G. Chen, “Full-parameter identification of buck converter through bp-nn fitting explicit time-domain relationships,” IEEE Transactions on Power Electronics, vol. 39, no. 6, pp. 7560–7571, 2024.

[11] M. Raissi, P. Perdikaris, and G. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” Journal of Computational Physics, vol. 378, pp. 686–707, 2019. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0021999118307125

[12] X. Yang, L. Shu, and D. Yang, “Hierarchical physics-embedding neural network framework for 3d magnetic modeling of medium-frequency transformers,” IEEE Transactions on Power Electronics, vol. 40, no. 3, pp. 4486–4497, 2025.

[13] S. Zhao, Y. Peng, Y. Zhang, and H. Wang, “Parameter estimation of power electronic converters with physics-informed machine learning,” IEEE Transactions on Power Electronics, vol. 37, no. 10, pp. 11 567– 11 578, 2022.

[14] Y. Xiang, H. Lin, and H. S.-H. Chung, “Extended physics-informed neural networks for parameter identification of switched mode power converters with undetermined topological durations,” IEEE Transactions on Power Electronics, vol. 40, no. 1, pp. 2235–2247, 2025.