# GraphCast–Mamba: Learning Non-Markovian Corrections for Weather Forecasting

Ilya Vilkoviskiy<sup>1,∗</sup>, Liang-Hong Mo<sup>1,∗</sup>, Suozhi Huang<sup>2</sup>, Dmitry A. Abanin<sup>1,3,4,5</sup>

<sup>1</sup>Department of Physics, Princeton University, Princeton, NJ 08544, USA

<sup>2</sup>Department of Electrical and Computer Engineering, Princeton University, Princeton, NJ 08544, USA

<sup>3</sup>École Polytechnique Fédérale de Lausanne (EPFL), 1015 Lausanne, Switzerland <sup>4</sup>Princeton Quantum Initiative, Princeton University, Princeton, NJ 08544, USA <sup>5</sup>Google Research, Brandschenkestrasse 150, 8002 Zürich, Switzerland

## Abstract

Unresolved atmospheric processes can introduce history dependence beyond the short input windows used by many weather forecasting models. We introduce GraphCast–Mamba, which augments a frozen, pretrained 1<sup>◦</sup> GraphCast backbone with a trainable Mamba correction branch combining graph message passing and Mamba layers. The branch retains a temporal memory state across successive six-hour forecast steps, learning corrections informed by temporal context beyond GraphCast’s two-frame input. Training is confined to the correction branch, using ERA5 data from 2015–2021. After 2,500 optimizer updates, including warm-up, the selected model achieves substantial improvements relative to the uncorrected GraphCast backbone. It reduces the GraphCast loss (normalized weighted mean-squared error; MSE) by 14.42% when aggregated over the full ten-day forecast and by 21.66% at day 10 (240 h after forecast initialization). At this final forecast lead time, root-mean-square error (RMSE) decreases by 7.09–14.60% across all 11 evaluated variable groups, spanning temperature, wind, pressure, geopotential, specific humidity, and precipitation. These results demonstrate broad improvements in ten-day weather forecasts through a short correction-training schedule while retaining the pretrained backbone.

## 1 Introduction

Learned global weather models produce skillful forecasts from one or a few recent atmospheric states (Bi et al., 2023; Lam et al., 2023; Chen et al., 2023b). GraphCast, for example, uses two states six hours apart to predict the next state. These inputs provide only a partial description of the atmosphere: they contain a limited set of variables at finite spatial resolution, while unresolved processes can still influence their evolution. The history of the resolved state may carry information about these influences that a short input window does not capture.

This is a familiar closure problem in turbulent-flow modeling. Resolving the smallest dynamically active scales is computationally costly (Yang and Grifin, 2021), so practical models must account for their efects on the resolved scales. A common approach uses approximate dissipative closures, such as eddy viscosity or hyperviscosity, to represent transfer to unresolved scales. These closures can reproduce selected aspects of the flow, but their accuracy depends on the quantity and scale of interest. For example, in forced two-dimensional turbulence, viscous, hyperviscous, and Leith closures can approximate enstrophy transfer, whereas the Smagorinsky closure dissipates enstrophy at the wrong scales (Pietarila Graham and Ringler, 2013). Resolution studies provide a related caution: mean dissipation and enstrophy can be nearly converged while higher-order moments of velocity gradients, dissipation, and enstrophy, together with distribution tails, remain substantially underestimated (Donzis et al., 2008). Reproducing bulk statistics therefore does not ensure accurate fluctuations or extreme events.

A complementary approach allows the closure to depend on the history of the resolved state. Unresolved variables have their own dynamics, so their influence need not be determined by the current resolved variables alone; past observations can contain information about those hidden degrees of freedom. The Mori–Zwanzig formalism makes this explicit: eliminating unresolved variables generally yields an instantaneous term, a history-dependent memory term, and a term associated with unresolved initial conditions, even when the full dynamics are Markovian (Lin et al., 2021; De Wit et al., 2026). Retaining the memory contribution leads to a non-Markovian closure, in which the resolved evolution depends on past states as well as the present one. This motivates persistent temporal memory as a mechanism for representing unresolved influences that an instantaneous dissipative approximation may miss.

Evidence from learned partial diferential equation (PDE) models supports this motivation. Memory Neural Operator (MemNO), which combines a neural operator with a structured state-space sequence (S4) layer, improves predictions from coarse or noisy observations (Buitrago Ruiz et al., 2025). Here, we ask whether retaining temporal context beyond GraphCast’s two-frame input improves forecast corrections when the correction architecture and parameter count are held fixed.

To investigate this question, we introduce GraphCast–Mamba, which augments a frozen pretrained 1<sup>◦</sup> GraphCast backbone with a trainable correction branch combining graph message passing and Mamba layers (Lam et al., 2023; Gu and Dao, 2023). Graph message passing exchanges spatial information, while Mamba retains temporal context across successive six-hour forecast steps. The branch adds a history-dependent correction to each backbone forecast, and the corrected forecast supplies the next input. Only the correction branch is trained.

Our principal experiment trains the correction on ERA5 data from 2015–2021, selects the model on 32 forecast initializations from the 2022 development set, and evaluates ten-day forecasts on 122 initializations in 2023 at 1<sup>◦</sup> resolution. This experiment tests the benefit of correction for one pretrained backbone at a fixed resolution. Its dependence on resolution and transfer to other forecasters remain open questions.

Our contributions are threefold.

• A temporal-memory correction model. GraphCast–Mamba combines spatial message passing with temporal Mamba layers in a separate Mamba correction branch, extending the predictor’s temporal context while keeping the pretrained GraphCast backbone frozen.

• Evidence for a contribution from temporal memory. A matched control trained and evaluated without persistent temporal memory uses the same architecture, parameter count, and optimization schedule. Retaining memory improves ten-day GraphCast loss reduction by 5.26 percentage points (pointwise paired 95% confidence interval: 4.70–5.86) for this training realization.

• Broad improvements in ten-day forecasts. After 2,500 optimizer updates, including warm-up, our model reduces the GraphCast loss by 14.42% aggregated over the ten-day rollout and by 21.66% at the final 240-hour lead. At that final lead, RMSE decreases by 7.09–14.60% across all 11 evaluated variable groups relative to the paired frozen backbone.

## 2 Related Work

Our work connects three lines of research: memory-based modeling of partially observed dynamics, temporal memory in weather forecasting, and learned corrections to existing predictors.

Memory and learned closures in reduced dynamics. When only part of a dynamical system is resolved, the evolution of the retained variables can depend on their history, as described by the Mori–Zwanzig formalism (Lin et al., 2021). Motivated by this connection, MemNO combines Fourier neural operators with S4 memory for coarse or noisy PDE observations (Buitrago Ruiz et al., 2025); SS-NO combines spatial state-space operators with temporal memory (Koren and Lanthaler, 2025). Memory also enters learned corrections to imperfect dynamics: Gupta and Lermusiaux (2021) augment reduced models with neural delay diferential equations, while Levine and Stuart (2022) develop a framework for learning both memoryless and memory-dependent model error. These studies motivate using temporal context to represent efects missing from a predictor’s resolved state. Our experiments assess the predictive value of retained context without identifying a physical memory kernel.

Temporal memory in weather forecasting. Recurrent weather models already retain information across forecast steps. SwinRNN carries hidden features through global forecasts, while SwinVRNN adds learned perturbations for ensemble prediction (Hu et al., 2023); SwinRDM couples recurrent forecasts with difusion super-resolution (Chen et al., 2023a). For radar nowcasting, GauMamba combines Mamba processing with gated recurrent unit (GRU) memory to predict evolving Gaussian representations (Wang et al., 2025). These models establish temporal memory as a forecasting mechanism. Our architecture places memory in a separate correction branch: Mamba processes time at each mesh node, graph message passing exchanges spatial information, and the pretrained GraphCast backbone remains frozen.

Learned forecast corrections. Learning corrections provides a way to improve an existing predictor while retaining its forecast model. δ-Adapter learns input adjustments and output residual corrections around frozen time-series forecasters (Liang et al., 2026). In numerical weather prediction, Chen et al. (2022) learn state-dependent corrections from analysis increments and apply them during FV3-GFS integration. Solver-in-the-Loop trains corrections through diferentiable solver rollouts, accounting for their efects on subsequent states (Um et al., 2020). These approaches establish both frozen-model adaptation and correction during model integration. For our weather predictor, the remaining question is whether a correction benefits from temporal context beyond the backbone’s two-frame input.

These precedents motivate our investigation of persistent temporal memory in a correction branch for a frozen pretrained global weather model. Our experiments compare corrections with and without retained state using the same architecture and parameter count.

## 3 Method: A hybrid GraphCast–Mamba architecture

We study whether a temporal memory state improves forecasts beyond the finite input history supplied to the frozen GraphCast backbone. GraphCast uses two successive atmospheric states as input; its forecast is Markovian with respect to this augmented state. Figure 1a shows how the Mamba correction branch corrects each GraphCast backbone forecast and carries its temporal memory state across successive six-hour steps. The GraphCast–Mamba forecast advances the meteorological

![](images/825cdf8ec3d15c2ea5daea28f4973f1fe394a95d82e29db0b14fd2d169c51700.jpg)

(a) Autoregressive rollout over three six-hour steps. Each GraphCast–Mamba forecast advances the two-frame input window; dashed orange arrows carry the temporal memory state, initialized here with m<sub>0</sub> = 0.
<table><tr><td rowspan=1 colspan=1>Geopotential</td><td rowspan=1 colspan=1>1.2</td><td rowspan=1 colspan=1>3.1</td><td rowspan=1 colspan=1>5.2</td><td rowspan=1 colspan=1>7.1</td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=1>10.3</td></tr><tr><td rowspan=1 colspan=1>Temperature</td><td rowspan=1 colspan=1>2.1</td><td rowspan=1 colspan=1>4.1</td><td rowspan=1 colspan=1>6.0</td><td rowspan=1 colspan=1>7.7</td><td rowspan=1 colspan=1>9.2</td><td rowspan=1 colspan=1>10.1</td></tr><tr><td rowspan=1 colspan=1>Specific humidity</td><td rowspan=1 colspan=1>4.3</td><td rowspan=1 colspan=1>6.2</td><td rowspan=1 colspan=1>7.9</td><td rowspan=1 colspan=1>9.4</td><td rowspan=1 colspan=1>10.8</td><td rowspan=1 colspan=1>11.7</td></tr><tr><td rowspan=1 colspan=1>Zonal wind</td><td rowspan=1 colspan=1>2.8</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>9.1</td><td rowspan=1 colspan=1>10.8</td><td rowspan=1 colspan=1>11.9</td></tr><tr><td rowspan=1 colspan=1>Meridional wind</td><td rowspan=1 colspan=1>2.8</td><td rowspan=1 colspan=1>5.1</td><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>9.6</td><td rowspan=1 colspan=1>11.8</td><td rowspan=1 colspan=1>13.2</td></tr><tr><td rowspan=1 colspan=1>Pressure vertical velocity</td><td rowspan=1 colspan=1>1.9</td><td rowspan=1 colspan=1>3.2</td><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>5.6</td><td rowspan=1 colspan=1>6.5</td><td rowspan=1 colspan=1>7.1</td></tr><tr><td rowspan=1 colspan=1>10 m zonal wind</td><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>9.9</td><td rowspan=1 colspan=1>11.9</td><td rowspan=1 colspan=1>13.3</td><td rowspan=1 colspan=1>14.1</td></tr><tr><td rowspan=1 colspan=1>10 m meridional wind</td><td rowspan=1 colspan=1>4.6</td><td rowspan=1 colspan=1>7.1</td><td rowspan=1 colspan=1>9.5</td><td rowspan=1 colspan=1>11.4</td><td rowspan=1 colspan=1>13.5</td><td rowspan=1 colspan=1>14.6</td></tr><tr><td rowspan=2 colspan=1>2 m temperatureMean sea-level pressure</td><td rowspan=1 colspan=1>2.7</td><td rowspan=1 colspan=1>4.7</td><td rowspan=1 colspan=1>6.3</td><td rowspan=1 colspan=1>7.6</td><td rowspan=1 colspan=1>8.9</td><td rowspan=1 colspan=1>9.6</td></tr><tr><td rowspan=1 colspan=1>2.8</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>7.7</td><td rowspan=1 colspan=1>9.5</td><td rowspan=1 colspan=1>10.8</td><td rowspan=1 colspan=1>11.5</td></tr><tr><td rowspan=1 colspan=1>6 h precipitation</td><td rowspan=1 colspan=1>3.0</td><td rowspan=1 colspan=1>4.3</td><td rowspan=1 colspan=1>5.6</td><td rowspan=1 colspan=1>6.5</td><td rowspan=1 colspan=1>7.4</td><td rowspan=1 colspan=1>8.4</td></tr><tr><td></td><td rowspan=1 colspan=1>5</td><td></td><td></td><td></td><td></td><td></td></tr></table>

(b) Variable-group RMSE reductions at days 5 through 10.  
![](images/c0edcee8d876aa076be00a3a5da9b70452d1b998d40e0e26117d67d120667fe4.jpg)  
(c) Channel scores: channels with lower RMSE than GraphCast at each six-hour lead.

Figure 1: GraphCast–Mamba: temporal-memory correction and forecast-skill overview. Panel (a) shows forecast feedback and memory carry. In (b), positive percentages indicate lower variable-group RMSE than the paired GraphCast baseline; (c) counts channels with lower RMSE, out of 83. Counts describe aggregate diferences without channel-wise significance tests. Section 4.1 defines the evaluation protocol and RMSE aggregation.

![](images/ecec943d352419eeb9bcfadaf5e888a35a853fef2640762cc013e6749d70a1a1.jpg)  
Figure 2: Trainable Mamba correction branch. Two graph message-passing steps each feed a two-layer Mamba module, followed by decoding and an output head initialized to zero. The frozen GraphCast backbone forecast $G _ { \theta } ( X _ { t } )$ is combined with the learned forecast correction $c _ { t }$ to produce $\hat { Y } _ { t } .$ . The Mamba correction branch has separate parameters and a temporal memory state $m _ { t }$ , comprising the SSM states and causal-convolution caches of its four Mamba layers.

input window, while the temporal memory state retains information from the preceding trajectory. Figure 2 details how graph message passing mixes information across mesh nodes and Mamba updates memory independently at each node. This introduces history dependence relative to the baseline input representation.

## 3.1 Architecture

GraphCast backbone The frozen GraphCast backbone uses a pretrained checkpoint and also serves as the standalone forecast baseline. This graph neural network operates on a latitude–longitude grid and an auxiliary mesh, with graph $\mathcal { G } ( \mathcal { V } ^ { \mathrm { G } } , \mathcal { V } ^ { \mathrm { M } } , \mathcal { E } ^ { \mathrm { G 2 M } } , \mathcal { E } ^ { \mathrm { M } } , \mathcal { E } ^ { \mathrm { M 2 G } } )$ . The vertices $\mathcal { V } ^ { \mathrm { G } }$ represent the meteorological grid and $\mathcal { V } ^ { \mathrm { M } }$ the mesh. Encoding uses grid-to-mesh edges $\mathcal { E } ^ { \mathrm { G 2 M } }$ , followed by $N = 1 6$ graph message-passing steps over $\mathcal { E } ^ { \mathrm { M } }$ and decoding over mesh-to-grid edges $\mathcal { E } ^ { \mathrm { M 2 G } }$ . Latent node features have width $W _ { \mathrm { G C } } = 5 1 2$ . The mesh provides a more compact representation for spatial processing than the meteorological grid.

Mamba correction branch: temporal memory on the mesh. The Mamba correction branch uses Mamba (Gu and Dao, 2023) to process forecast time independently at each mesh node, with spatial exchange handled by graph message passing. Each layer normalizes the width-512 mesh features, projects them into a $d _ { \mathrm { i n n e r } }$ -dimensional temporal representation, and projects its output back to the mesh width through a residual connection. We use $d _ { \mathrm { i n n e r } } = 1 6$ . Across autoregressive forecast calls, each layer retains both its state-space model (SSM) state and its three-token causal-convolution cache separately for each batch element and mesh node.

Our model uses an SSM state dimension of 16 per inner channel, a causal convolution width of 4, a rank-32 projection for the step-size parameter $\Delta _ { t }$ , and one shared group of input-dependent $B _ { t } , C _ { t }$ coeficients. Additional studies examine $d _ { \mathrm { i n n e r } } = 3 2$ . The memory-reset controls in Section 4.5 use $d _ { \mathrm { i n n e r } } = 1 6$ and one shared B, C group to test the contribution of temporal memory.

An additive correction with temporal memory. Let $X _ { t }$ denote the two-frame meteorological input (shown as $X _ { t - 1 : t }$ in the diagrams) and the associated forcings. Forecast calls are indexed from $t = 1 ; { \hat { Y } } _ { t }$ denotes the next-state forecast produced by call t. The temporal memory state $m _ { t }$ collects the SSM states and causal-convolution caches of all Mamba layers. As shown in Figure 1a, GraphCast– Mamba combines the frozen GraphCast backbone $G _ { \theta }$ with the trainable Mamba correction branch $R _ { \phi }$ , whose output $c _ { t }$ is the forecast correction:

$$
( c _ { t } , m _ { t } ) = R _ { \phi } ( X _ { t } , m _ { t - 1 } ) , \qquad \hat { Y } _ { t } = G _ { \theta } ( X _ { t } ) + c _ { t } , \qquad \theta \mathrm { ~ f i x e d } .\tag{1}
$$

The Mamba correction branch in Figure 2 has its own grid-to-mesh encoder, mesh processor, and mesh-to-grid decoder. It receives the same meteorological inputs as the GraphCast backbone and uses separate parameters throughout. Its mesh processor uses two graph message-passing steps, each followed by a two-layer Mamba module, giving four recurrent layers in total. Encoder grid features are also passed directly to the decoder. Suppressing this grid-feature connection in the notation, the mesh-processing path is

$$
L _ { t } ^ { ( 0 ) } = \operatorname { E n c o d e } _ { \phi } ( X _ { t } ) ,\tag{2}
$$

$$
\widetilde { L } _ { t } ^ { ( k ) } = \mathrm { G N N } _ { \phi } ^ { ( k ) } ( L _ { t } ^ { ( k - 1 ) } ) ,\tag{3}
$$

$$
( L _ { t } ^ { ( k ) } , m _ { t } ^ { ( k ) } ) = \mathrm { M a m b a M o d u l e } _ { \phi } ^ { ( k ) } ( \widetilde { L } _ { t } ^ { ( k ) } , m _ { t - 1 } ^ { ( k ) } ) , \qquad k = 1 , 2 ,\tag{4}
$$

$$
c _ { t } = \mathrm { D e c o d e } _ { \phi } ( L _ { t } ^ { ( 2 ) } ) .\tag{5}
$$

Each Mamba module contains two residual Mamba layers; $m _ { t } ^ { ( k ) }$ comprises their SSM states and causal-convolution caches. The output head is initialized to zero, so the initial GraphCast–Mamba forecast equals the GraphCast backbone forecast. During autoregressive forecasting, the oldest input frame is dropped, the GraphCast–Mamba forecast $\hat { Y } _ { t }$ is appended, and the known forcings are updated to form $X _ { t + 1 }$ . The temporal memory state is passed separately to the next correction-branch call. The cold-start rollout illustrated in Figure 1a begins with $m _ { 0 } = 0$ and retains the updated temporal memory state thereafter.

## 3.2 Training

Initialization and trainable parameters. The Mamba correction branch inherits pretrained spatial weights and uses a zero-initialized output head, so the initial GraphCast–Mamba forecast matches the GraphCast backbone forecast. We train all correction-branch parameters $\phi$ on ERA5 2015–2021 while keeping the separate GraphCast backbone parameters θ frozen.

Training segments and memory carry. We divide the training sequence into contiguous segments of 96 six-hour steps (24 days), each processed as four consecutive 24-step windows (Figure 3). The temporal memory state, comprising both the SSM states and causal-convolution caches, starts at zero for each segment and is carried through all 96 steps. Each window begins with a four-step teacher-forced prefix (ERA5 inputs), followed by 20 feedback-conditioned forecast steps that use the GraphCast–Mamba forecast as feedback. Meteorological inputs therefore return to ERA5 at the start of each window, while the correction branch retains the preceding window’s temporal memory state. Gradients propagate through the temporal memory state within a window; at its end, the state is carried forward with its gradient detached. Gradients are also stopped through the meteorological feedback.

$$
\begin{array} { r } { m _ { \mathrm { i n i t } } = 0 \longrightarrow \underbrace { \left[ \underline { { 1 \cdot 2 4 } } \right] \stackrel { \mathrm { s g } } { \longrightarrow } \left[ \underline { { 2 5 \cdot 4 8 } } \right] \stackrel { \mathrm { s g } } { \longrightarrow } \left[ \underline { { 4 9 \cdot 7 2 } } \right] \stackrel { \mathrm { s g } } { \longrightarrow } \left[ 7 3 \cdot 9 6 \right] } _ { \mathrm { o n e ~ s e g m e n t : ~ m e m o r y ~ c a r r i e d ~ t h r o u g h ~ 9 6 ~ s t e p s } } \xrightarrow { \mathrm { r e s e t } } \ m _ { \mathrm { i n i t } } ^ { \prime } = 0 } \end{array}
$$

Figure 3: Memory carry during training. Boxes show the four 24-step windows of one segment. The arrows marked sg carry the SSM states and causal-convolution caches unchanged across window boundaries while stopping their gradients. Both are reset for the next segment.

Training loss. At each step, the forecast correction is supervised against the residual target $r _ { t } = Y _ { t } - G _ { \theta } ( X _ { t } )$ , where $Y _ { t }$ is ERA5 truth and $X _ { t }$ is the current input. Since $c _ { t } - r _ { t } = \hat { Y } _ { t } - Y _ { t }$ , this minimizes the GraphCast loss $ { \ell _ { \mathrm { G C } } }$ of the GraphCast–Mamba forecast. We average the loss over all H = 24 steps, including the teacher-forced prefix, and the B batch elements:

$$
\small \mathcal { I } ( \phi ) = \frac { 1 } { B H } \sum _ { b = 1 } ^ { B } \sum _ { k = 1 } ^ { H } \ell _ { \mathrm { G C } } ( \hat { Y } _ { b , k } , Y _ { b , k } ) .\tag{6}
$$

Appendix A.2 gives the full loss definition and weights.

Training schedule. We train with AdamW for 2,000 post-warm-up updates after a 500-update warm-up, or 2,500 optimizer updates in total. The evaluated model uses stochastic weight averaging (SWA); Appendix A provides the complete training configuration, model-selection procedure, and computational costs.

## 4 Results

We train the GraphCast–Mamba correction branch on seven years of ERA5 data (2015–2021), keeping the GraphCast backbone frozen. We then evaluate whether the model improves forecast skill, how broadly the gains extend across the atmospheric state, and how strongly performance depends on temporal memory. To assess the latter, we compare against a matched reset-trained control and a memory-reset evaluation of the selected checkpoint.

## 4.1 Evaluation protocol and metrics

The principal experiments use GraphCast–Mamba with the original pretrained 1<sup>◦</sup> GraphCast backbone and the Mamba correction branch trained in Section 3.2. The same checkpoint supplies the lead-time, variable-group, and channel-coverage results. The principal results use 122 forecast initializations at 00 UTC every third day from 1 January to 30 December 2023, hereafter the evaluation initializations, on the native $1 ^ { \circ }$ grid at 40 six-hour forecast steps from 6 to 240 h. The checkpoint was selected among the six learning-rate arms on 32 forecast initializations from the 2022 development set (Appendix A.4) and fixed for this 2023 evaluation.

Each forecast takes two ERA5 states as input. GraphCast–Mamba initializes the temporal memory state to zero, and advances for ten days using its own predictions as feedback. GraphCast– Mamba feeds back its full prediction, while GraphCast follows an independent uncorrected trajectory. Stateful models retain their updated temporal memory state between steps; the controls without persistent temporal memory reset it at every step.

We first examine aggregate forecast performance using the GraphCast loss, with diferencestandard-deviation normalization, latitude weights, pressure-level weights, and variable weights. For forecast initializations indexed by s, let $\mathcal { L } _ { s , k }$ denote this loss at six-hour lead step k. The ten-day rollout loss reduction is

$$
\mathcal { T } _ { 1 : 4 0 } = 1 0 0 \left( 1 - \frac { \sum _ { s } \sum _ { k = 1 } ^ { 4 0 } \mathcal { L } _ { s , k } ^ { \mathrm { m o d e l } } } { \sum _ { s } \sum _ { k = 1 } ^ { 4 0 } \mathcal { L } _ { s , k } ^ { \mathrm { G C } } } \right) .\tag{7}
$$

Day-10 loss corresponds to k = 40 only. Losses are aggregated before taking the ratio; the rollout score is not an average of per-lead percentages.

For each variable and lead, RMSE pools squared physical-unit errors across forecast initializations using normalized cosine-latitude weights before taking the square root. Upper-air variable-group

![](images/7c5c9897ca7d77d169043f787195f3120014ab4b40eb9785eb051f15061550d5.jpg)

![](images/79e7077848595a283d880cd3d1eafb98a36578097a57c27e4bf04deffc2d78b4.jpg)

![](images/a1b1a3e556ffb7df0e332201bf1c50631d4e44c89a0e373ae87ff4c90a33c4f2.jpg)

(d) 2 m temperature  
![](images/2863d237fe7d960b9f1fec1a0256d51c43c4b9264f01fab833a858b66069a4b0.jpg)  
Figure 4: GraphCast–Mamba forecast skill on the 122 evaluation initializations in 2023 (Section 4.1). Top: GraphCast loss and its per-lead reduction. Bottom: 2 m temperature RMSE and RMSE reduction on the same forecast initializations. The Mamba correction branch uses $d _ { \mathrm { i n n e r } } = 1 6$ a threefold Mamba learning rate (LR), and SWA over post-warm-up updates 500–2,000. Every reduction uses the paired GraphCast baseline shown in the absolute-error panels.

RMSE additionally pools squared errors equally over 13 pressure levels; channel RMSE scores each surface variable and each upper-air pressure level separately. We report $( 1 - \mathrm { R M S E _ { m o d e l } / R M S E _ { G C } } ) \times$ 100%, with positive values denoting lower error.

## 4.2 Forecast skill improves at longer forecast lead times

Figure 4 shows that the benefit of the correction is concentrated at longer forecast lead times. Relative to the GraphCast baseline, the GraphCast loss decreases by 14.42% over the full ten-day rollout and by 21.66% at the final 240-hour lead.

Physical-unit errors show a similar pattern. For 2 m temperature, RMSE changes remain small through day 3, before the reduction grows at longer forecast lead times. At day 10, RMSE falls from 2.67 K to 2.41 K, a 9.59% reduction.

## 4.3 Gains extend across the predicted atmospheric state

The gains at longer forecast lead times extend across the predicted atmospheric state. All 11 variable groups have lower RMSE at every evaluated six-hour lead from day 5 through day 10, with day-10 reductions of 7.09–14.60% (Figure 1b). These improvements span circulation, thermodynamic, moisture, and surface quantities. Table 3 in Appendix A.7 reports the reductions by variable at selected leads, together with absolute day-10 RMSEs.

Appendix D complements these aggregate results with individual forecast trajectories at Paris and New York City, including both improvements and larger local errors in a historical January 2022 forecast.

## 4.4 Comparison with higher-resolution GraphCast

Figure 5 compares the selected 1<sup>◦</sup> GraphCast–Mamba model with its frozen GraphCast\_smal backbone and the public 0.25<sup>◦</sup> GraphCast checkpoint (Lam et al., 2023). Forecasts are initialized throughout 2023 and verified against identical ERA5 fields on a common 1.5<sup>◦</sup> grid, with RMSE expressed relative to the matched ECMWF ensemble mean (ENS). At each diagnostic and lead, every predictor uses the same available dates. GraphCast–Mamba has lower RMSE than the 0.25<sup>◦</sup> checkpoint for all 12 diagnostics at every evaluated lead from day 3.5 through day 10 (Appendix C). At day 10, reductions are 7.70% for $7 0 0 \mathrm { { h P a } }$ specific humidity, 10.58% for 2 m temperature, and 9.83% for 850 hPa zonal wind, spanning 7.70–11.42% across the 12 diagnostics.

Despite operating at 1<sup>◦</sup> resolution, GraphCast–Mamba achieves better long-lead accuracy than the more computationally demanding 0.25<sup>◦</sup> GraphCast model on the common verification grid. The reference uses approximately 16 times as many horizontal grid points, four times as many mesh vertices, and 37 rather than 13 pressure levels, increasing spatial computation and activation-memory requirements. This comparison concerns the specified fixed models on the common verification grid; it does not isolate the efect of resolution or measure skill at native 0.25<sup>◦</sup> scales. ENS retains lower RMSE on all 12 diagnostics at day 10.

![](images/790386a2f45237ad62b41cb7559dd16e3321e6cd87a6f7d7beff81521444d41a.jpg)  
Figure 5: Forecast RMSE relative to the matched ECMWF ensemble mean (ENS) on a common 1.5<sup>◦</sup> grid. From left to right: 700 hPa specific humidity (Q700), 2 m temperature (T2m), and 850 hPa zonal wind (U850). Blue shows the selected GraphCast–Mamba model M1 (1<sup>◦</sup>), dashed gray its public GraphCast\_small backbone (1<sup>◦</sup>), and green the public GraphCast checkpoint (0.25<sup>◦</sup>). From 122 candidate 00 UTC initializations every third day in 2023, each diagnostic and lead uses the identical ENS-available dates for all four forecasts (92–105 starts per point in these panels). Squared errors are pooled before taking square roots. The horizontal line marks ENS at 100%; lower values indicate lower RMSE. Appendix C specifies the protocol.

## 4.5 Temporal memory contributes to forecast skill

The preceding results show that GraphCast–Mamba improves forecasts at longer lead times across the predicted atmospheric state. The correction branch introduces both additional trainable capacity and temporal memory. To examine the contribution of temporal memory, we compare the selected model with a matched control trained and evaluated with memory reset at every step. We also reset the selected model’s memory only during forecasting to measure its dependence on retained state.

Table 1 summarizes three policies obtained from two training runs. Policy names specify memory handling during training first and forecasting second. Carry passes the updated SSM state and causal-convolution cache to the next step; reset zeros both before every step. Resetting leaves all learned weights and the two-frame meteorological input in place: the full correction branch still computes a correction, but cannot retain its temporal memory state between calls. Every training segment and independent forecast begins with a zero temporal memory state under all three policies.

<table><tr><td rowspan="2">Policy Training/forecasting</td><td colspan="2">Temporal memory retained</td></tr><tr><td>Training</td><td>Forecasting</td></tr><tr><td>Carry/carry</td><td>√</td><td>√</td></tr><tr><td>Carry/reset</td><td>√</td><td>×</td></tr><tr><td>Reset/reset</td><td>×</td><td>X</td></tr></table>

Table 1: Memory retained between successive steps: ✓ denotes retention; × denotes resetting both the SSM state and causal-convolution cache before every step. Carry/carry and carry/reset share the same trained weights. Reset/reset is trained separately with the same architecture and parameter count.

Carry/carry and carry/reset use the selected checkpoint. Reset/reset is trained independently with memory reset in both its 500-update warm-up and 2,000-update main stage. Both training configurations use seven years of data, a K = 20 feedback-conditioned training tail, inner dimension $d _ { \mathrm { i n n e r } } = 1 6$ , and the same batch size, optimizer schedule, training seed, and SWA checkpoint indices (Appendix A.1). We evaluate all three policies on the same 122 forecast initializations and 40 lead times of Section 4.1, using a common GraphCast baseline. Appendices A.2 and A.3 describe the loss calculation and paired uncertainty estimates.

Over the full ten-day rollout, carry/carry achieves a 14.42% GraphCast loss reduction, compared with 9.16% for matched reset/reset. The paired advantage is 5.26 percentage points (pointwise 95% confidence interval: 4.70–5.86). At day 10, the reductions are 21.66% and 13.64%, respectively, a diference of 8.02 percentage points (pointwise 95% confidence interval: 7.20–9.05). Figure 6 shows all three policies.

Resetting the selected model’s memory only during forecasting (carry/reset) yields 4.57% rollout and 9.62% day-10 loss reduction. Its rollout reduction is 9.85 percentage points below carry/carry. This shows that the model uses its temporal memory state, although the drop may partly reflect the mismatch between training and forecasting. The matched training-policy comparison supports a benefit from persistent memory for this configuration and training realization. The intervals quantify forecast-sample uncertainty for fixed checkpoints; variation across training seeds is not measured.

## 5 Conclusion

Our results show that a small correction branch with persistent temporal memory can improve forecasts from a frozen pretrained weather model at longer forecast lead times. The matched reset-trained control supports a benefit from retaining temporal memory for the tested configuration and training realization, while inference-time reset demonstrates the selected model’s dependence on retained history. This suggests that compact temporal extensions ofer a promising way to improve existing forecasters with limited additional training. How this benefit depends on spatial resolution and the choice of backbone remains an open question.

![](images/97c3182655516fecf984564dd5c90a18f03fa35e0368145b0b810bc592c5a3b5.jpg)  
Figure 6: Comparison of carry/carry, carry/reset, and matched reset/reset memory policies defined in Table 1. Curves show exact GraphCast loss reduction relative to a common GraphCast baseline on the 122 initializations in 2023. Reset/reset uses the matched training schedule described in the text.

Limitations. Our results show that non-Markovian memory can improve forecasts from a frozen Markovian backbone. This suggests the possibility of jointly learning the forecasting dynamics and temporal memory within a native model. However, autoregressive training introduces sequential dependencies between forecast steps, limiting temporal parallelism and opportunities for batching and data reuse. Backpropagation through long rollouts also requires storing or recomputing intermediate activations, making eficient computation and memory access central challenges for joint training of the full model. To keep training tractable, our study restricts learning to a correction branch attached to a frozen pretrained forecaster.

A separate direction concerns high spatial frequencies. At fixed resolution, dynamically important unresolved scales may increase the value of temporal history, as suggested by the PDE studies discussed in the introduction. Regimes with substantial high-frequency content could therefore provide a more sensitive probe of non-Markovian memory. Our aggregate loss and RMSE metrics do not establish whether memory improves fine-scale predictions or reduces excessive smoothing; this requires evaluating scale-dependent forecast errors and spatial-gradient statistics.

## AI use statement

We used a large language model for minor editorial assistance, including correcting grammar and typo and refining formatting. It did not contribute to the development of research ideas, the analysis, or the claims.

## Ethics Statement

All datasets, models in this work are publicly available. The usage of these datasets, models, code strictly follows the terms and conditions set by the providers.

## Reproducibility Statement

To ensure reproducibility of our results, we provide detailed documentation of our experimental setup and methodology in the main paper and appendix. The code for data, evaluation, and training will be open-sourced upon publication.

## Acknowledgments

Ilya Vilkoviskiy and Dmitry Abanin acknowledge support through a Brown Investigator Award and grant 864597 from the European Research Council (ERC), funded by the European Union’s Horizon 2020 research and innovation programme. Ilya Vilkoviskiy also acknowledges support from OpenAI through its ChatGPT for Academic Researchers program (OpenAI, 2026).

## References

Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. Accurate mediumrange global weather forecasting with 3d neural networks. Nature, 619:533–538, 2023. doi: 10.1038/s41586-023-06185-3. URL https://doi.org/10.1038/s41586-023-06185-3.

Ricardo Buitrago Ruiz, Tanya Marwah, Albert Gu, and Andrej Risteski. On the benefits of memory for modeling time-dependent PDEs. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 89379d5fc6eb34ff98488202fb52b9d0-Abstract-Conference.html.

Lei Chen, Fei Du, Yuan Hu, Zhibin Wang, and Fan Wang. SwinRDM: Integrate SwinRNN with difusion model towards high-resolution and high-quality weather forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 322–330, 2023a. doi: 10.1609/ aaai.v37i1.25105. URL https://ojs.aaai.org/index.php/AAAI/article/view/25105.

Lei Chen, Xiaohui Zhong, Feng Zhang, Yuan Cheng, Yinghui Xu, Yuan Qi, and Hao Li. FuXi: a cascade machine learning forecasting system for 15-day global weather forecast. npj Climate and Atmospheric Science, 6:190, 2023b. doi: 10.1038/s41612-023-00512-1. URL https://doi.org/10. 1038/s41612-023-00512-1.

Tse-Chun Chen, Stephen G. Penny, Jefrey S. Whitaker, Sergey Frolov, Robert Pincus, and Stefan Tulich. Correcting systematic and state-dependent errors in the NOAA FV3-GFS using neural

networks. Journal of Advances in Modeling Earth Systems, 14:e2022MS003309, 2022. doi: 10.1029/2022MS003309. URL https://doi.org/10.1029/2022MS003309.

Xander M De Wit, Alessandro Gabbana, Michael Woodward, Yen Ting Lin, Federico Toschi, and Daniel Livescu. Data-driven mori–zwanzig modeling of lagrangian particle dynamics in turbulent flows. Proceedings of the National Academy of Sciences, 123(13):e2525390123, 2026.

D. A. Donzis, P. K. Yeung, and K. R. Sreenivasan. Dissipation and enstrophy in isotropic turbulence: Resolution efects and scaling in direct numerical simulations. Physics of Fluids, 20(4):045108, 04 2008. ISSN 1070-6631. doi: 10.1063/1.2907227. URL https://doi.org/10.1063/1.2907227.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Abhinav Gupta and Pierre F. J. Lermusiaux. Neural closure models for dynamical systems. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 477(2252):20201004, 2021. doi: 10.1098/rspa.2020.1004. URL https://doi.org/10.1098/rspa.2020.1004.

Hans Hersbach, Bill Bell, Paul Berrisford, et al. The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730):1999–2049, 2020. doi: 10.1002/qj.3803. URL https://doi.org/10.1002/qj.3803.

Yuan Hu, Lei Chen, Zhibin Wang, and Hao Li. SwinVRNN: A data-driven ensemble forecasting model via learned distribution perturbation. Journal of Advances in Modeling Earth Systems, 15:e2022MS003211, 2023. doi: 10.1029/2022MS003211. URL https://doi.org/10.1029/ 2022MS003211.

Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization, 2019. URL https://arxiv. org/abs/1803.05407.

Nodens F. Koren and Samuel Lanthaler. Merging memory and space: A state space neural operator. arXiv preprint arXiv:2507.23428, 2025. URL https://arxiv.org/abs/2507.23428v5. Version 5, revised March 2026.

Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, Alexander Merose, Stephan Hoyer, George Holland, Oriol Vinyals, Jacklynn Stott, Alexander Pritzel, Shakir Mohamed, and Peter Battaglia. Learning skillful medium-range global weather forecasting. Science, 382(6677):1416– 1421, 2023. doi: 10.1126/science.adi2336. URL https://doi.org/10.1126/science.adi2336.

Matthew E. Levine and Andrew M. Stuart. A framework for machine learning of model error in dynamical systems. Communications of the American Mathematical Society, 2:283–344, 2022. doi: 10.1090/cams/10. URL https://doi.org/10.1090/cams/10.

Daojun Liang, Qi Li, Yinglong Wang, Jing Chen, Hu Zhang, Xiaoxiao Cui, Qizheng Wang, and Shuo Li. The forecast after the forecast: A post-processing shift in time series. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_ files/paper/2026/hash/6f7d90b1198fec96defd80b5ebd5bc81-Abstract-Conference.html.

Yen Ting Lin, Yifeng Tian, Daniel Livescu, and Marian Anghel. Data-driven learning for the mori–zwanzig formalism: A generalization of the koopman learning framework. SIAM Journal

on Applied Dynamical Systems, 20(4):2558–2601, 2021. doi: 10.1137/21M1401759. URL https: //doi.org/10.1137/21M1401759.

OpenAI. Accelerating scientific discovery with ChatGPT for academic researchers. https://openai. com/index/chatgpt-for-academic-researchers/, July 2026. Published July 29, 2026; accessed October 14, 2026.

Jonathan Pietarila Graham and Todd Ringler. A framework for the evaluation of turbulence closures used in mesoscale ocean large-eddy simulations. Ocean Modelling, 65:25–39, 2013. doi: 10.1016/j.ocemod.2013.01.004. URL https://doi.org/10.1016/j.ocemod.2013.01.004.

Kiwon Um, Robert Brand, Yun (Raymond) Fei, Philipp Holl, and Nils Thuerey. Solver-in-the-loop: Learning from diferentiable physics to interact with iterative PDE-solvers. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https://proceedings.neurips.cc/ paper/2020/hash/43e4e6a6f341e00671e123714de019a8-Abstract.html.

Ziye Wang, Yiran Qin, Lin Zeng, and Ruimao Zhang. High-dynamic radar sequence prediction for weather nowcasting using spatiotemporal coherent Gaussian representation. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_ files/paper/2025/hash/4d1344317478ad99ff5f4e414aeab689-Abstract-Conference.html.

Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs V: Tuning large neural networks via zero-shot hyperparameter transfer. arXiv preprint arXiv:2203.03466, 2022. URL https://arxiv.org/abs/2203.03466.

Xiang IA Yang and Kevin P Grifin. Grid-point and time-step requirements for direct numerical simulation and large-eddy simulation. Physics of Fluids, 33(1), 2021.

## A Training details and model selection

## A.1 Training configuration

The configuration below supplies the principal GraphCast loss and RMSE results in Section 4.

Backbone, data, and normalization. We use the public GraphCast\_small checkpoint trained on ERA5 1979–2015, with 1<sup>◦</sup> resolution, 13 pressure levels, a mesh refined through level 5, and precipitation as both an input and an output. Its filename is GraphCast\_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz.

Training examples are drawn from ERA5 2015–2021, with 2022 reserved for development. These data come from the six-hourly WeatherBench2 ERA5 archive,<sup>1</sup> subsampled by selecting every fourth latitude and longitude of the $0 . 2 5 ^ { \circ }$ grid to obtain a $1 8 1 \times 3 6 0$ grid at 1<sup>◦</sup>. We retain pressure levels 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, and 1000 hPa. We reuse the GraphCast normalization files mean\_by\_level.nc, stddev\_by\_level.nc, and diffs\_stddev\_ by\_level.nc, distributed under gs://dm\_graphcast/graphcast/stats/, throughout correction training and evaluation.

Initialization and batching. We initialize the Mamba correction branch from the frozen Graph-Cast backbone checkpoint: its grid-to-mesh encoder (G2M), mesh-to-grid decoder (M2G), and two graph message-passing steps copy the corresponding pretrained weights, using the first two of GraphCast’s 16 graph message-passing steps. We insert newly initialized Mamba modules after these graph message-passing steps and append a linear output head with weights and biases initialized to zero. Consequently, the initial forecast correction is zero and the GraphCast–Mamba forecast equals the pretrained GraphCast backbone forecast. The copied spatial parameters, Mamba modules, and output head form the trainable parameters $\phi ;$ the separate GraphCast backbone parameters θ remain frozen throughout training on ERA5 2015–2021. Each optimizer update processes one window from each of four independent segments, giving batch size B = 4 under the memory-carry procedure in Section 3.2.

Optimizer and checkpoint settings. The reported model uses two optimization stages. The first stage, called the 500-update warm-up in the main text, linearly increases the learning rate from zero to $1 0 ^ { - 4 }$ over its first 50 updates and then holds it constant. Spatial and Mamba parameters use the same learning rate during this stage. The second stage transfers the learned correction weights and starts a fresh optimizer for 2,000 further updates, resetting the random-number generator, temporal memory, and data cursor. Its spatial learning rate ramps from zero to $1 0 ^ { - 4 }$ over 50 updates and then follows cosine decay toward $1 0 ^ { - 5 }$ over the remaining 1,950 updates. The Mamba learning rate is three times the spatial learning rate throughout this second stage.

Both stages use AdamW with moments (0.9, 0.98), weight decay $1 0 ^ { - 4 }$ applied to all trainable correction parameters, global gradient-norm clipping at 1, and random seed 22. Model computation uses bfloat16, with float32 meteorological fields and temporal-state boundaries. The configuration uses spatial width 512, inner dimension 16, and one B, C group. The evaluated model averages the parameters saved at second-stage updates 500, 1,000, 1,500, and 2,000 using SWA; these are the post-warm-up checkpoint indices used elsewhere in the paper.

Matched reset-training control. The reset/reset control uses the same backbone, training data, correction architecture, parameter count, batch size four, random seed 22, precision, optimizer schedules, and SWA indices as the selected carry-trained model. Both its 500-update warm-up and 2,000-update main stage reset the SSM states and causal-convolution caches before every correction prediction; the main stage starts from its own reset-trained warm-up weights. The 24-step training windows, $K = 2 0$ feedback-conditioned tail, full correction feedback, and loss objective are retained. Hyperparameters are fixed to the selected configuration without further model selection. The control uses the recovered archived training and model code, with previously unarchived dependencies supplied from the workspace and frozen for the new run. The historical environment was not fully archived, so matching refers to the configured training protocol and does not imply bitwise reproduction of the earlier run.

## A.2 Complete loss definition

The per-step loss in Equation 6 is the GraphCast loss:

$$
\ell _ { \mathrm { G C } } ( \hat { Y } , Y ) = \sum _ { v } \frac { \lambda _ { v } } { N _ { \mathrm { g } } L _ { v } } \sum _ { i = 1 } ^ { N _ { \mathrm { g } } } \sum _ { \ell = 1 } ^ { L _ { v } } a _ { i } q _ { v , \ell } \left( \frac { \hat { Y } _ { v , \ell , i } - Y _ { v , \ell , i } } { \sigma _ { v , \ell } } \right) ^ { 2 } .\tag{8}
$$

Here $N _ { \mathrm { g } }$ is the number of grid points and $L _ { v }$ is the number of levels for variable v. The scales $\sigma _ { v , \ell }$ are GraphCast’s standard deviations of six-hour diferences. The latitude weights $a _ { i }$ have unit mean and include the pole-cell correction. For upper-air variables, $q _ { v , \ell } = p _ { \ell } / \overline { { p } }$ weights levels in proportion to pressure; surface variables have $L _ { v } = 1$ and $q _ { v , 1 } = 1$ . The variable weights $\lambda _ { v }$ are 1 for upper-air variables and 2 m temperature, and 0.1 for the two 10 m wind components, mean sea-level pressure, and six-hour accumulated precipitation.

Evaluation variants. We use GraphCast loss as the common label for the weighted MSE diagnostics. The principal results and updated 2023 memory comparisons evaluate Equation 8 directly, using a common GraphCast baseline. The closed-loop training-tail sweep reconstructs it from channel RMSEs using per-level normalization and pressure weights, without the pole-cell correction. The archived open-loop sweep instead uses variable-group RMSEs and level-averaged normalization scales. These reconstructions therefore approximate the loss above.

## A.3 Memory-control evaluation and uncertainty

The matched reset/reset evaluation reuses the original frozen evaluator, initialization manifest, input data, and normalization statistics. Carry/carry and carry/reset reuse their original forecast files. All three policies start with zero temporal memory, feed back their full corrected forecasts, and are scored on the same 122 initializations and 40 six-hour leads; late-December forecasts use verifying ERA5 states through 9 January 2024. All loss reductions use the original carry/carry evaluation’s uncorrected GraphCast trajectory as their common denominator, pooling losses before computing ratios.

The carry/carry versus matched reset/reset rollout contrast is a separate planned analysis added after the original selected-model evaluation; it is not part of that evaluation’s original confirmatory endpoint family. Day 10 is a secondary endpoint. We resample complete paired forecasts using 10,000 non-wrapping block-bootstrap draws with seed 20260916 and blocks of five consecutive initialization positions (12 elapsed days). Intervals for reductions, contrasts, and leadwise curves are pointwise 95% intervals, conditional on the fixed checkpoints and the 2023 sample. They do not measure training-seed or between-year variation, or establish simultaneous significance across leads.

The positive rollout contrast persists with three-start and seven-start blocks, resampling within calendar quarters, and exclusion of the three previously exposed pilot initializations present in this sample (15 February, 15 July, and 15 November 2023, all at 00 UTC). The latter retains 119 starts and gives 5.27 percentage points (95% interval: 4.70–5.90); excluded dates are masked after resampling on the original regular grid. This sensitivity does not remove prior test-year exposure or overlap between forecast windows. Non-wrapping blocks underweight boundary dates, and quarter-wise resampling breaks dependence across quarter boundaries.

Repeated uncorrected GraphCast evaluations difer numerically. For reset/reset, the pooled baseline loss difers by 0.0872% from the common baseline, with a largest initialization/lead diference of 4.39%. Using each policy’s own repeated baseline instead gives a carry/carry versus reset/reset rollout contrast of 5.34 percentage points (95% interval: 4.78–5.95), preserving the conclusion. The reported main estimates consistently use the common baseline.

## A.4 Mamba capacity and learning-rate selection

Learning-rate choice depends on network width and parameterization. Yang et al. (2022) derive width-dependent scaling rules under maximal update parameterization $( \mu \mathrm { P } )$ , including an inversewidth scaling for Adam learning rates on hidden weight matrices. In our correction branch, the newly initialized Mamba modules contain approximately 0.11M and 0.22M parameters at inner dimensions 16 and 32, respectively, compared with 10.24M parameters in the spatial components. Motivated by these scaling results and the diference in module size, we test whether the compact Mamba modules benefit from a larger learning rate than the spatial components initialized from pretrained GraphCast weights. We treat this as an empirical hypothesis; the $\mu \mathrm { P }$ scaling rule does not directly prescribe a learning-rate multiplier for our architecture.

We compare Mamba learning-rate multipliers of 0.3×, 1×, and 3× at each inner dimension, holding the spatial peak learning rate fixed at $1 0 ^ { - 4 }$ . A multiplier of 1× gives equal learning rates to both parameter groups. All six configurations use spatial width 512, one B, C group, batch size four, and AdamW moments (0.9, 0.98). Each run has its own 500-update warm-up followed by 2,000 post-warm-up updates, with SWA over post-warm-up updates 500, 1,000, 1,500, and 2,000. The GraphCast backbone remains frozen throughout. Warm-up parameter values difer across runs despite the common training seed, so the comparisons assess complete training runs.

Table 2: Mamba capacity and learning-rate comparison at $1 ^ { \circ }$ . Entries are GraphCast loss reductions (%) on the 32 forecast initializations from the 2022 development set; bold marks the largest value in each metric column. The multiplier applies only to Mamba parameters, with the spatial learning rate fixed. All six configurations use the same training schedule and SWA window.
<table><tr><td> $d _ { \mathrm { i n n e r } }$ </td><td>Mamba LR multiplier</td><td>Ten-day aggregate</td><td>Day 10</td></tr><tr><td>16</td><td>0.3×</td><td>14.70</td><td>21.98</td></tr><tr><td>16</td><td>1×</td><td>15.67</td><td>22.44</td></tr><tr><td>16</td><td>3×</td><td>16.34</td><td>23.68</td></tr><tr><td>32</td><td>0.3×</td><td>15.82</td><td>22.96</td></tr><tr><td>32</td><td>1×</td><td>16.29</td><td>23.32</td></tr><tr><td>32</td><td>3×</td><td>15.73</td><td>22.75</td></tr></table>

Table 2 shows that, for inner dimension 16, increasing the Mamba learning-rate multiplier from 1× to 3× raises the ten-day aggregate loss reduction from 15.67% to 16.34% and the day-10 reduction from 22.44% to 23.68%. For inner dimension 32, the best tested multiplier is 1×; increasing it to 3× lowers the aggregate reduction from 16.29% to 15.73%. Dimension 32 performs better at 0.3× and 1×, whereas dimension 16 performs better at 3×. Thus, the smaller GraphCast-Mamba head benefits from a larger learning rate in these runs, supporting joint selection of Mamba capacity and learning rate. The results do not establish a monotonic benefit from increasing either quantity.

Across all six tested combinations of Mamba inner dimension and learning-rate multiplier, ten-day aggregate GraphCast loss reductions on the 2022 development set range from 14.70% to 16.34%, with day-10 reductions of 21.98–23.68%. These consistently substantial gains indicate that the overall correction benefit is robust across the tested choices of Mamba capacity and learning rate.

## A.5 Optimizer configuration study

These optimizer comparisons use the 2022 development set. The optimizer study varies peak learning rate $( 1 0 ^ { - 4 } \mathrm { ~ o r ~ } 5 \times 1 0 ^ { - 5 } )$ , Adam moments, Mamba initialization scheme (legacy or Mamba1), and the paired inner-dimension/group settings (16/1 or 64/4). The latter changes two factors simultaneously and does not isolate inner dimension. Runs use 10,000 updates, seed 22, a 200-update warm-up, cosine decay to one tenth of the peak learning rate, and weight decay $1 0 ^ { - 4 }$ . The correction branch is trained with closed-loop feedback, 24-step backpropagation windows, a training tail of 20 feedbackconditioned steps, and loss across all 24 steps. The forecast correction contribution to the feedback is stop-gradient, while temporal memory state gradients propagate within the backpropagation window.

Day-10 reductions for the 22 completed selected SWA configurations range from 13.47% to 23.21%. The configuration at the upper end uses Mamba1 initialization, inner dimension 16, one B, C group, learning rate $1 0 ^ { - 4 }$ , and AdamW moments (0.9, 0.98); its aggregate loss reduction is 16.28%.

## A.6 Checkpoint averaging and model selection

Stochastic weight averaging. SWA averages correction-branch parameter values from saved checkpoints to reduce dependence on a single training iterate (Izmailov et al., 2019). Every configuration in Table 2 averages four checkpoints at post-warm-up updates 500, 1,000, 1,500, and 2,000. The dimension-16 Mamba1-initialized optimizer-study configuration described above averages seven checkpoints, saved every 1k updates from 2k through 8k. These are parameter averages, not ensembles of independent forecast predictions. Evaluation initializes the temporal memory state to zero, including for SWA models.

Training methods and model selection. Table 2 reports all six SWA configurations from the Mamba capacity and learning-rate comparison. Figure 4 uses the dimension-16, 3× Mamba learningrate SWA configuration, which has the largest aggregate and day-10 GraphCast loss reductions on the 2022 development sample among the completed SWA configurations in the reported optimizer and learning-rate studies. This checkpoint is fixed for the 2023 evaluation.

In the optimizer study, each run screens periodic checkpoints and three SWA windows on eight forecast initializations. The selected checkpoint and SWA are then evaluated on the 32 forecast initializations from the 2022 development set.

## A.7 Variable-group RMSE

Table 3 provides the numerical results underlying Figure 1b at selected forecast leads. It uses the principal seven-year correction and its paired GraphCast baseline on the 122 forecast initializations in 2023, evaluated on the native $1 ^ { \circ }$ grid as described in Section 4.1.
<table><tr><td></td><td colspan="2">Day-10 RMSE</td><td colspan="3">RMSE reduction (%)</td></tr><tr><td>Variable [unit]</td><td>GraphCast</td><td>GraphCast- Mamba</td><td>Day 5 Day 7 Day 10</td><td></td><td></td></tr><tr><td colspan="6">Upper-air variables (13 pressure levels)</td></tr><tr><td>Geopotential  $[ \mathrm { m ^ { 2 } s ^ { - 2 } } ]$ </td><td>784.41</td><td>703.68</td><td>1.24</td><td>5.22</td><td>10.29</td></tr><tr><td>Temperature [K]</td><td>3.10</td><td>2.79</td><td>2.11</td><td>5.98</td><td>10.07</td></tr><tr><td> $\mathrm { S p e c i f i c ~ h u m i d i t y ~ [ g \ k g ^ { - 1 } ] }$ </td><td>1.10</td><td>0.97</td><td>4.33</td><td>7.88</td><td>11.68</td></tr><tr><td>Zonal wind  $\bigl [ \mathrm { m } \mathrm { s } ^ { - 1 } \bigr ]$ </td><td>8.67</td><td>7.63</td><td>2.76</td><td>7.19</td><td>11.94</td></tr><tr><td>Meridional wind  $[ \mathrm { \bar { m } s ^ { - 1 } } ]$ </td><td>9.05</td><td>7.85</td><td>2.78</td><td>7.22</td><td>13.23</td></tr><tr><td>Pressure vertical velocity</td><td>0.177</td><td>0.164</td><td>1.89</td><td>4.52</td><td>7.09</td></tr><tr><td colspan="6"> $\mathrm { [ P a s ^ { - 1 } ] }$ </td></tr><tr><td>Surface variables 10 m zonal wind  $[ \mathrm { m } \mathrm { s } ^ { - 1 } ]$ </td><td>4.28</td><td>3.68</td><td>4.51</td><td>9.90</td><td>14.13</td></tr><tr><td>10 m meridional wind  $[ \mathrm { m } \mathrm { s } ^ { - 1 } ]$ </td><td>4.50</td><td>3.84</td><td>4.58</td><td>9.50</td><td>14.60</td></tr><tr><td>2 m temperature [K]</td><td>2.67</td><td>2.41</td><td>2.66</td><td>6.26</td><td>9.59</td></tr><tr><td>Mean sea-level pressure [hPa]</td><td>7.20</td><td>6.37</td><td>2.78</td><td>7.67</td><td>11.55</td></tr><tr><td>6 h accumulated precipitation [mm]</td><td>2.69</td><td>2.47</td><td>2.96</td><td>5.57</td><td>8.36</td></tr></table>

Table 3: Physical-unit RMSE and relative RMSE reduction for the principal GraphCast–Mamba model. Days 5, 7, and 10 denote leads of 120, 168, and 240 h. RMSE pools cosine-latitude-weighted squared errors across forecast initializations and, for upper-air groups, equally across 13 pressure levels before taking the square root. Each reduction is $1 0 0 ( 1 - \mathrm { R M S E } _ { \mathrm { m o d e l } } / \mathrm { R M S E } _ { \mathrm { G C } } )$ , computed before rounding. Positive values indicate lower RMSE; these percentages are distinct from GraphCast loss reductions.

## A.8 Computational resources

Table 4 reports the measured resources for the selected model. The elapsed time covers the complete GPU allocation for the 500-update initialization stage, the 2,000-update main stage, compilation, data loading, checkpointing, and SWA construction. Data preparation, hyperparameter search, and forecast evaluation are excluded. Forecast evaluation ran separately, and neither training stage performed inline validation. Host memory is the peak resident memory reported by Slurm; GPU memory is the largest value observed in five-second telemetry samples.

<table><tr><td>Quantity</td><td>Selected model</td></tr><tr><td>Accelerator</td><td>1 NVIDIA A100 80GB PCIe</td></tr><tr><td>Allocated CPU cores</td><td>8</td></tr><tr><td>Elapsed allocation time</td><td>22 h 40 min 45 s</td></tr><tr><td>GPU allocation cost</td><td>22.68 GPU-hours</td></tr><tr><td>Peak host memory</td><td>97.1 GiB</td></tr><tr><td>Sampled peak GPU memory</td><td>23.15 GiB</td></tr><tr><td>Frozen backbone parameters</td><td>35,979,347</td></tr><tr><td>Trainable correction parameters</td><td>10,348,879</td></tr><tr><td>Spatial components and output head</td><td>10,238,863</td></tr><tr><td>Mamba modules</td><td>110,016</td></tr></table>

Table 4: Training resources and parameter counts for the selected correction model. GPU-hours equal elapsed allocation hours because the run uses one GPU. The GPU-memory value is a sampled peak.

## B Training tail length

These archived training-tail sweeps use 2022 forecasts and separate training configurations.

We examine how forecast skill depends on the training tail length K under two input-feedback policies. Each 24-step training window begins with a teacher-forced prefix: the model receives ERA5 states as meteorological inputs while updating and retaining its temporal memory. For $K > 1$ , the final K steps instead use forecast feedback, leaving $2 4 - K$ teacher-forced steps. The label $K = 1$ denotes fully teacher-forced training, with ERA5 inputs at every step. We evaluate the resulting models over 40 six-hour forecast steps (ten days) and report GraphCast loss reduction relative to GraphCast; positive values indicate improved forecast skill.

## B.1 Closed-loop training

In closed-loop training, the corrected GraphCast–Mamba forecast supplies the next meteorological input during the training tail. The learned correction therefore afects both the current prediction and the inputs encountered at subsequent steps. Figure 7 compares models trained with diferent K, showing how the duration of this feedback changes forecast skill.

![](images/22c53cabfa2fefed28a23191e83ffa71b92b6463673a860b2759e240a865e409.jpg)  
Figure 7: Forecast skill after closed-loop training with diferent training tail lengths K. During the tail, corrected GraphCast–Mamba predictions supply subsequent inputs. Each curve shows GraphCast loss reduction relative to GraphCast over ten forecast days; positive values indicate improvement. The vertical dashed line marks $2 0 \times 6 = 1 2 0 \mathrm { { h } }$ (5 days), the training tail duration for $K = 2 0$ . Gains for this model persist beyond its training tail to day 10, twice that duration.

Short training tails primarily improve early forecasts. Models trained with $K = 1$ and $K = 4$ reduce error during the first one to two days but increase it thereafter, whereas longer training tails yield substantial gains at longer lead times. At day 10, the error reduction relative to GraphCast increases from 5.2% for $K = 1 0$ to 24.3% for $K = 2 0$ . These curves reveal a crossover: shorter tails favor early forecast skill, while longer tails become more efective as the forecast advances. To examine this crossover in a more controlled setting, we remove correction feedback and train the Mamba correction branch on precomputed forecast trajectories from the frozen GraphCast backbone. In this open-loop setting, corrections do not afect subsequent backbone predictions. As shown in the next section, the crossover becomes more pronounced under this training scheme.

## B.2 Open-loop training and a sharper crossover

In open-loop training, the teacher-forced prefix is followed by a tail driven by the uncorrected GraphCast forecasts. The correction branch learns along this trajectory, but its predictions do not alter subsequent meteorological inputs. Thus, open loop refers to the absence of correction feedback, while teacher forcing refers to the use of ERA5 inputs in the prefix.

In our experiments, training with this teacher-forced prefix and a tail independent of the learned correction was more stable but yielded smaller improvements at longer lead times. Figure 8 shows that short-tail models remain close to GraphCast at long lead times, avoiding the substantial degradation seen in the closed-loop comparison. The crossover is more pronounced here: long-lead gains change markedly between $K = 1 6$ and $K = 1 8$ and increase further for $K = 2 0$ and $K = 2 2$

![](images/ba41aa96d1bb769a34baea32aa71427bfa2de6267dc874fc221ffd74d0cff290.jpg)  
Figure 8: Forecast skill after open-loop training with a teacher-forced prefix and diferent training tail lengths K. During the tail, uncorrected GraphCast forecasts supply subsequent inputs. GraphCast loss reduction exhibits a pronounced crossover, with substantially larger long-lead gains for $K \ge 1 8$ The vertical dashed line marks $2 0 \times 6 = 1 2 0 \mathrm { { h } }$ (5 days), the training tail duration for $K = 2 0 ;$ this model retains positive gains beyond that duration, up to day 10. This figure and Figure 7 use the same vertical scale but compare separate sweeps with diferent experimental settings and loss reconstructions (Appendix A.2).

This crossover raises an interesting question about how temporal memory, training tail length, and feedback interact. Its mechanism, including why it is sharper under open-loop training, remains unclear. We regard it as an empirical finding that motivates further investigation, rather than evidence for a universal critical value of K.

![](images/883654b3d84f26f62be56328c268b5a3d0e6b88c6f52a70294d5a299068582bc.jpg)  
Figure 9: The comparison in Figure 5 extended to 12 diagnostics. Blue shows the selected GraphCast– Mamba model M1 (1<sup>◦</sup>), dashed gray its public GraphCast\_small backbone (1<sup>◦</sup>), and green the public GraphCast checkpoint (0.25<sup>◦</sup>). Z denotes geopotential, T temperature, Q specific humidity, U zonal wind, and V meridional wind; a three-digit sufix gives pressure in hPa. T2m denotes 2 m temperature, U10 and V10 the 10 m wind components, and MSL mean sea-level pressure. All four forecasts use identical dates and ERA5 truth at each diagnostic and lead on the 1.5<sup>◦</sup> grid; the available date set varies across points (91–108 starts). Curves show pooled RMSE as a percentage of the matched ENS-mean RMSE, with ENS at 100%.

## C Comparison on a common verification grid

Models and forecast protocol. All three GraphCast-based forecasts are generated locally from fixed checkpoints. M1 is the same selected correction used in the principal evaluation, trained on 2015–2021 and selected on 2022 development data. Its frozen backbone is the public DeepMind GraphCast\_small checkpoint trained on ERA5 1979–2015, with a 1<sup>◦</sup> grid, 13 pressure levels and mesh refinement level 5. The higher-resolution reference is the public GraphCast checkpoint trained on ERA5 1979–2017, with a 0.25<sup>◦</sup> grid, 37 levels and mesh refinement level 6. Both public checkpoints include precipitation as an input and output and use the supplied GraphCast normalization statistics. Their grids contain 181 × 360 and 721 × 1440 points, respectively. Their meshes contain 10,242 and 40,962 vertices, respectively. The higher-resolution model therefore operates on much larger spatial graphs; the computational comparison concerns FLOPs and activation memory. Exact checkpoint filenames, checksums and configuration are recorded in the figure-source manifest.

Each forecast receives observed ERA5 states at −6 and 0 h, then advances through 40 sixhour leads from 6 to 240 h using its own forecast feedback. Time-varying 0.25<sup>◦</sup> ERA5 inputs are subsampled to 1<sup>◦</sup> for GraphCast\_small and M1 and restricted to their 13 pressure levels; these models retain static fields from their existing training and evaluation store. M1 begins each forecast with zero temporal memory and feeds back its full corrected prediction. Network inference uses bfloat16, with physical fields and state boundaries in float32.

Dates and ENS reference. The candidate sample consists of 122 initializations at 00 UTC every third day from 1 January through 30 December 2023, with verification extending through 9 January 2024. The evaluation protocol records earlier numerical-pilot exposure, including three candidate starts (15 February, 15 July and 15 November); no checkpoint selection used this comparison. ENS denotes the ECMWF IFS ensemble-mean forecast distributed by WeatherBench 2 on its conservative $1 2 1 \times 2 4 0 ~ \mathrm { g r i d }$ . We use available 2023 forecasts from the extended 2016–2024 store.<sup>2</sup> Missing metadata and coordinate arrays were reconstructed from the complete 2018–2022 sibling store, with decoding checked against 71 overlapping forecast chunks. Missing forecast chunks remain unavailable.

At each diagnostic and lead, we retain exactly the same ENS-available initializations for all four forecasts. This gives 91–108 starts per point across the 12 diagnostics (92–105 in Figure 5), with 46,930 initialization–lead–diagnostic combinations per model. The matched date set can vary between diagnostics and leads; there is no single complete subset underlying every point. Selection depends only on forecast availability, not errors. Availability masks and the full candidate manifest accompany the figures.

Verification and metric. Every forecast is verified against identical ERA5 fields (Hersbach et al., 2020) from ARCO ERA5, with source-data repairs validated against the independently curated NSF NCAR ERA5 archive. Native model predictions and 0.25<sup>◦</sup> verification fields are remapped to the common 1.5<sup>◦</sup> grid using first-order conservative spherical-cell overlap with periodic longitude; the archived ENS fields already occupy this grid. For each diagnostic and lead, we compute spatial mean squared errors with normalized cosine-latitude weights, average equally over the matched starts, and take the square root. The plotted value is $1 0 0 \mathrm { R M S E } _ { \mathrm { m o d e l } } / \mathrm { R M S E } _ { \mathrm { E N S } }$ , with identical dates in numerator and denominator. These physical-unit RMSE diagnostics are separate from the normalized GraphCast training loss.

Interpretation. The figures show point estimates for the specified fixed checkpoints on the available matched samples. Ten-day verification windows overlap, so the starts are not independent weather realizations. Checkpoints also difer in architecture, pressure levels and training coverage; their comparison does not isolate resolution or attribute the full advantage to memory. The matched memory controls in Section 4.5 address the latter question. Lower RMSE after common-grid remapping establishes neither finer-scale skill at 0.25<sup>◦</sup> nor a measured inference speedup. At day 10, M1 has lower RMSE than the public 0.25<sup>◦</sup> checkpoint on all 12 diagnostics, but higher RMSE than ENS.

## D Forecast examples at city locations

To complement the aggregate seven-year results in Section 4, Figure 10 illustrates local forecast behavior for temperature, wind, and precipitation at Paris and New York City. These examples use a separate correction trained on 15 years of ERA5 data (2007–2021), with the configuration specified in the figure caption. Over 120–240 h, the correction reduces mean absolute error (MAE) in four of the six displayed city–variable pairs. Errors increase for Paris temperature and New York precipitation, illustrating that aggregate skill gains can coexist with deterioration in individual forecasts.

![](images/9cbfd5ada41289cd634688aa7f0c42380d513685ca474c0e05f8eb8410086cac.jpg)  
Figure 10: Ten-day forecasts from 00 UTC on 15 January 2022 at the 1<sup>◦</sup> grid points nearest Paris $( 4 9 ^ { \circ } \mathrm { N } , 2 ^ { \circ } \mathrm { E } ;$ left) and New York City $( 4 1 ^ { \circ } \mathrm { N } , 7 4 ^ { \circ } \mathrm { W } ; \mathrm { r i g h t } )$ . From top to bottom, rows show 2 m temperature (T2m), 850 hPa zonal wind (U850), and 6 h accumulated precipitation in physical units. ERA5 is shown in black, the frozen GraphCast forecast in dashed orange, and the GraphCast–Mamba forecast in blue. The correction was trained separately on 15 years of ERA5 data (2007–2021), with a training tail of $K = 2 0$ feedback-conditioned steps, Mamba inner dimension 128, spatial width 512, and two graph message-passing steps; SWA averages checkpoints from updates 6k–11k. Both forecasts use the ERA5 states at −6 and 0 h (before and on the dotted line) as inputs, then feed back their own predicted fields at each step; the GraphCast–Mamba model begins with a zero temporal memory state. Precipitation is shown as predicted and may take small negative values.