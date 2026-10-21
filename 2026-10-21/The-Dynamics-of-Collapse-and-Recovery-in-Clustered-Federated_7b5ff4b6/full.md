# The Dynamics of Collapse and Recovery in Clustered Federated Learning

Priyanka Sinha

Snigdhansu Chatterjee

University of Maryland, Baltimore County

psinha1@umbc.edu

snigchat@umbc.edu

## Abstract

Clustered federated learning (CFL) can suffer from cluster collapse. In particular, under unfavorable initialization, early client assignments concentrate heterogeneous groups on one cluster model, which is then trained toward a pooled compromise, while another model receives no updates. Since the current models determine client routing and routing determines subsequent model updates, we formulate clustered federated learning as control of an unknown dynamical system, with cluster models as the evolving state, client routing patterns as the control input, and routing interventions as state dependent controllers. In a two group linear model, we identify an initialization region in which greedy routing makes collapse self reinforcing. We then derive finite horizon error dynamics showing how routing interventions can reverse collapse by repeatedly training the starved model, primarily with updates from its compatible group. Motivated by this condition, each client uses blocked EXP3 to choose between greedy routing and interventions that redirect updates toward unused models. Experiments show that with simpler predictors and faster specialist formation, autonomous controller selection recovers usable specialists from unfavorable starts that trap standard CFL baselines. In transformed FEMNIST, the synchronized-entry configuration with persistent commitment recovers usable specialists in settings where the autonomous controller configuration often does not. Code and scripts for reproducing the experiments and figures are available at https://github.com/prsinha1/FedRecover.

## 1 Introduction

Clustered federated learning (CFL) addresses client heterogeneity by maintaining several shared models and assigning each client to the model that best fits its local data [3, 7]. However, under an unfavorable initialization, CFL is widely known to be prone to cluster collapse, where clients from several latent groups may initially prefer the same model, while other models are starved of training. Under strong heterogeneity, such a surviving model is trained across conflicting client distributions and becomes a compromised pooled solution, leading to optimization instability [6]. Initialization sensitivity is therefore a consequence of a deeper dynamical failure, where early routing decisions determine which models improve, and those improvements determine later routing decisions.

Existing CFL methods either assume or seek favorable initialization, or modify the clustering architecture to mitigate collapse [3, 4, 6, 8]. We instead ask whether a fixed collection of cluster models can recover after unfavorable initialization has already induced collapsed routing.

To answer this question, we make the feedback loop between client side routing and server side model updates explicit by viewing CFL as a controlled dynamical system. The cluster models $W ^ { t }$ form the state, the routing pattern $M ^ { t }$ is the control input, and the server update defines the state evolution, $W ^ { t + 1 } =$ $\mathcal { F } ( W ^ { t } , M ^ { t } ; \mathcal { E } )$ , where $\mathcal { E }$ represents the unknown client objectives and latent group structure. We introduce $\Phi ^ { t }$ as a feedback controller that maps observable client information to routing decisions inducing $M ^ { t }$ . Preventing collapse then becomes a control problem over an unknown and partially observed dynamical system, paralleling the control theoretic treatment of interactive personalized recommendation systems [9].

Our contributions. First, in a two group linear model, we characterize an unfavorable initialization basin in which greedy routing permanently starves one cluster model while the active model converges to a pooled solution. Second, for arbitrary realized routing sequences, we derive a finite horizon recovery condition requiring sufficient updates to the disadvantaged model while controlling incompatible client contributions, which we call persistent selective exposure. Third, motivated by this condition, we design complementary routing controllers and let each client learn among them online. Experiments show that decentralized control recovers distinct usable models when specialization forms quickly. For transformed FEMNIST, we compare autonomous controller selection with synchronized entry under the specified controller settings.

## 2 Initialization induced failure and recovery

We consider n clients each coming from one of G heterogeneous latent groups $I _ { 1 } , \ldots , I _ { G }$ . A client $i \in$ $I _ { g }$ has local distribution $\mathcal { D } _ { i }$ and shares the group population objective $L _ { g } ( w ) = \mathbb { E } _ { z \sim D _ { i } } [ \ell ( w ; z ) ]$ . Group membership is hidden from both clients and server. At round t, the server maintains cluster models $W ^ { t } =$ $( w _ { 1 } ^ { t } , \dots , w _ { K } ^ { t } )$ . Greedy CFL assigns the client to a lowest loss model and updates each cluster using only the clients assigned to it.

Favorable and unfavorable initialization basins: To demonstrate how the assignment and update feedback determines the training trajectory, consider two latent groups, relabeled 0 and 1, and two cluster models. For group $g \in \{ 0 , 1 \}$ , samples $( x , y )$ satisfy $y = x ^ { \top } \theta _ { g } + \epsilon .$ where $\theta _ { g } \in \mathbb { R } ^ { d }$ is the group specific regression parameter. We assume $\mathbb { E } [ x ] = 0 , \mathbb { E } [ x x ^ { \top } ] = I$ , and independent zero mean observation noise. Under squared loss, the population objective is, up to an additive noise constant, $\begin{array} { r } { L _ { g } ( w ) = \frac { 1 } { 2 } \| w - \theta _ { g } \| _ { 2 } ^ { 2 } } \end{array}$ with $\nabla L _ { g } ( w ) = w - \theta _ { g }$ . Thus $\theta _ { g }$ is the unique population minimizer for group g. Let $\Delta = \lVert \theta _ { 1 } - \theta _ { 0 } \rVert _ { 2 }$ , and let the group proportions be α and $1 - \alpha$

Initial preference geometry and success criteria: An initialization has common preference when both groups initially prefer the same cluster. Formally, define

$$
\mathcal { C } _ { \mathrm { c o m m o n } } : = \bigcup _ { k \in \{ 0 , 1 \} } \left\{ W ^ { 0 } : L _ { g } ( w _ { k } ^ { 0 } ) < L _ { g } ( w _ { 1 - k } ^ { 0 } ) , \forall g \in \{ 0 , 1 \} \right\} .\tag{1}
$$

Otherwise, apart from ties, the two groups initially prefer different clusters. These labels describe only the first greedy assignment and do not determine whether later updates preserve it. Define the permutation invariant recovery error

$$
R _ { T } : = \operatorname* { m i n } _ { m \in \mathfrak { S } _ { 2 } } \operatorname* { m a x } _ { g \in \{ 0 , 1 \} } \left. w _ { m ( g ) } ^ { T } - \theta _ { g } \right. _ { 2 } .\tag{2}
$$

The system achieves r recovery at round $T$ when $R _ { T } \leq r$ . When $r < \Delta / 2$ , each group strictly prefers its matched model under terminal greedy routing. Thus recovery gives each group a distinct usable specialist rather than a single pooled compromise. In the balanced experimental setting, each matched model is also better for its group than the pooled parameter.

Define the certified specialization basin

$$
\mathcal { B } _ { \mathrm { s p e c i a l i z e } } : = \bigcup _ { m \in \mathfrak { S } _ { 2 } } \left\{ W ^ { 0 } = ( w _ { 0 } ^ { 0 } , w _ { 1 } ^ { 0 } ) \in \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } : \sum _ { g = 0 } ^ { 1 } \left\| w _ { m ( g ) } ^ { 0 } - \theta _ { g } \right\| _ { 2 } < \Delta \right\} .\tag{3}
$$

Theorem 1 (Greedy specialization basin). Suppose $W ^ { 0 } \in B _ { \mathrm { s p e c i a l i z e } }$ and $0 < \eta \leq 1$ . Let $m \in \mathfrak { S } _ { 2 }$ witness membership in (3). Then group g greedily selects cluster $m ( g )$ at every round, and

$$
w _ { m ( g ) } ^ { t + 1 } = w _ { m ( g ) } ^ { t } - \eta ( w _ { m ( g ) } ^ { t } - \theta _ { g } ) , \qquad w _ { m ( g ) } ^ { t } \longrightarrow \theta _ { g } , \qquad g \in \{ 0 , 1 \} .
$$

Proof: See Appendix A.1.

The basin condition certifies a favorable initialization in which the groups initially prefer different models and the induced updates preserve that separation. We next identify an initialization region with the opposite behavior. Let $\bar { \theta } _ { \alpha } = \alpha \theta _ { 0 } + ( 1 - \alpha ) \theta _ { 1 } . { \mathrm { F o r ~ } } j \in \{ 0 , 1 \}$ , define the basin in which cluster $j$ is starved as

$$
\begin{array} { r } { B _ { \mathrm { s t a r v e } , j } : = \left\{ W ^ { 0 } = ( w _ { 0 } ^ { 0 } , w _ { 1 } ^ { 0 } ) \in \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } : L _ { g } ( w _ { j } ^ { 0 } ) > \operatorname* { m a x } \left\{ L _ { g } ( w _ { 1 - j } ^ { 0 } ) , L _ { g } ( \bar { \theta } _ { \alpha } ) \right\} , \forall g \in \lbrace 0 , 1 \rbrace \right\} , } \end{array}\tag{4}
$$

$$
B _ { \mathrm { s t a r v e } } : = B _ { \mathrm { s t a r v e , 0 } } \cup B _ { \mathrm { s t a r v e , 1 } } .
$$

Within either component, both groups select the same active cluster, the resulting updates preserve that common preference, and the other cluster remains permanently starved.

Theorem 2 (Greedy starvation basin). Let $0 < \eta \leq 1 . \ I f W ^ { 0 } \in { \mathcal { B } } _ { \mathrm { s t a r v e } , j } ,$ then both groups route to cluster $1 - j$ at every round, $w _ { j } ^ { t } = w _ { j } ^ { 0 } ,$ and $w _ { 1 - j } ^ { t + 1 } = w _ { 1 - j } ^ { t } - \eta ( w _ { 1 - j } ^ { t } - \bar { \theta } _ { \alpha } )$ . Consequently, $w _ { 1 - j } ^ { t } \to \bar { \theta } _ { \alpha }$ while cluster $j$ remains permanently starved.

Proof: See Appendix A.2.

![](images/3ef6d89450ec58f1c1acf18f76ba1aace7e534f22e7d39770c422fd7a0c489d1.jpg)  
Figure 1: Empirical frequency of initial common preference, membership in the certified starvation basin $B _ { \mathrm { s t a r v e } } ,$ persistent greedy collapse, and terminal nonrecovery as functions of the initialization spread $\rho .$

To compare the theorem with the realized dynamics, we sample 20,000 initialization pairs per value of $\rho$ from $w _ { k } ^ { 0 } = \bar { \theta } + \rho z _ { k }$ , where $z _ { k } \sim \mathcal { N } ( 0 , I )$ and $\bar { \theta } = ( \theta _ { 0 } + \theta _ { 1 } ) / 2$ . In Figure 1, initial common preference is the event $W ^ { 0 } \in \mathcal { C } _ { \mathrm { c o m m o n } }$ from (1). It records only whether the first assignments coincide. The certified starvation curve is the frequency of $W ^ { 0 } \in \mathcal { B } _ { \mathrm { s t a r v e } }$ and therefore counts initializations for which Theorem 2 guarantees permanent starvation. Persistent greedy collapse means that greedy routing uses only one cluster throughout the 200 simulated rounds. Terminal nonrecovery is the event $R _ { 2 0 0 } \geq \Delta / 4$

All four frequencies increase with $\rho$ and exceed 80 percent at the largest tested spread. Common preference is more frequent than the other events at smaller $\rho ,$ showing that a shared first choice alone does not certify failure. Every sampled initialization in $B _ { \mathrm { s t a r v e } }$ remains persistently collapsed, while some common preference initializations outside the certified basin escape as training changes the loss ordering. Persistent collapse and terminal nonrecovery closely follow the certified basin frequency. Thus Figure 1 shows both that the sufficient basin has substantial probability under pooled centered Gaussian initialization and that its deterministic conclusion accurately predicts the simulated greedy trajectories.

Routing controlled dynamics: The starvation theorem describes the trajectory induced by greedy routing. To characterize what an intervention must change, let $a _ { i } ^ { t } \in \{ 1 , \ldots , K \}$ be client i’s sampled cluster and let $q _ { i } ^ { t } = e _ { a _ { i } ^ { t } }$ be its one hot executed route. Write $\begin{array} { r } { N _ { g } = \sum _ { i \in I _ { a } } n _ { i } , N = \sum _ { i } n _ { i } } \end{array}$ , and $\alpha _ { g } = N _ { g } / N$ . Define

$$
M _ { g k } ^ { t } = \frac { 1 } { N _ { g } } \sum _ { i \in I _ { g } } n _ { i } q _ { i k } ^ { t } = \frac { 1 } { N _ { g } } \sum _ { i \in I _ { g } } n _ { i } \mathbf { 1 } \{ a _ { i } ^ { t } = k \} ,\tag{5}
$$

$$
\begin{array} { r l } & { s _ { k } ^ { t } = \mathbf { 1 } \Bigg \{ \displaystyle \sum _ { h } \alpha _ { h } M _ { h k } ^ { t } > 0 \Bigg \} = \mathbf { 1 } \Bigg \{ \displaystyle \sum _ { i } n _ { i } \mathbf { 1 } \{ a _ { i } ^ { t } = k \} > 0 \Bigg \} , } \\ & { \beta _ { g k } ^ { t } = \frac { \alpha _ { g } M _ { g k } ^ { t } } { \displaystyle \sum _ { h } \alpha _ { h } M _ { h k } ^ { t } } = \frac { \sum _ { i \in I _ { g } } n _ { i } \mathbf { 1 } \{ a _ { i } ^ { t } = k \} } { \sum _ { i } n _ { i } \mathbf { 1 } \{ a _ { i } ^ { t } = k \} } . } \end{array}\tag{6}
$$

The second equality in each line makes the hard routing and sample count weighting explicit. Here $M _ { g k } ^ { t }$ is the fraction of group $g ^ { \ast } \mathbf { s }$ sample mass routed to cluster $k , s _ { k } ^ { t }$ indicates whether cluster k receives any client update, and $\beta _ { g k } ^ { t }$ is the sample weighted contribution of group $g$ to that cluster's update when $s _ { k } ^ { t } = 1$ . The routing induced server update remains

$$
\begin{array} { r } { w _ { k } ^ { t + 1 } = \left\{ \begin{array} { l l } { w _ { k } ^ { t } - \eta \sum _ { g } \beta _ { g k } ^ { t } \nabla L _ { g } ( w _ { k } ^ { t } ) , } & { s _ { k } ^ { t } = 1 , } \\ { w _ { k } ^ { t } , } & { s _ { k } ^ { t } = 0 . } \end{array} \right. } \end{array}\tag{7}
$$

If no client samples cluster $k ,$ then $M _ { g k } ^ { t } = 0$ for every group, $s _ { k } ^ { t } = 0$ , and the cluster remains frozen. A routing intervention can alter this trajectory by giving an otherwise unused cluster positive sampling probability and thereby producing executed updates for it.

Matched error dynamics: Fix a matching m between groups and clusters, and define the matched error $e _ { g } ^ { t } = w _ { m ( g ) } ^ { t } - \theta _ { g }$ , activity $a _ { g } ^ { t } = s _ { m ( g ) } ^ { t }$ , and incompatible update contribution $\begin{array} { r } { c _ { g } ^ { t } = \sum _ { h \neq g } \beta _ { h , m ( g ) } ^ { t } ( \theta _ { h } - \theta _ { g } ) } \end{array}$ Substituting these in Equation (7) gives

$$
e _ { g } ^ { t + 1 } = \left\{ \begin{array} { l l } { ( 1 - \eta ) e _ { g } ^ { t } + \eta c _ { g } ^ { t } , } & { a _ { g } ^ { t } = 1 , } \\ { e _ { g } ^ { t } , } & { a _ { g } ^ { t } = 0 . } \end{array} \right.\tag{8}
$$

Persistent selective exposure: When the matched cluster is inactive, its error cannot contract. When it is active, its error contracts toward the group specialist, while incompatible updates displace it through $c _ { g } ^ { t } .$ Recovery therefore requires repeated training of the matched cluster while its update composition remains dominated by the group it is intended to serve. We call this combination persistent selective exposure. For $\begin{array} { r } { E _ { g } ( t , T ) : = \sum _ { \tau = t } ^ { T - \bar { 1 } } a _ { g } ^ { \tau } } \end{array}$ , iterating Equation (8) gives the following condition.

Theorem 3 (Finite horizon recovery under selective exposure). Fix a matching m, let $0 \textless \eta \leq 1$ , and consider any realized routing sequence through round T. If, for each $g \in \{ 0 , 1 \}$

$$
( 1 - \eta ) ^ { E _ { g } ( 0 , T ) } \| e _ { g } ^ { 0 } \| _ { 2 } + \eta \sum _ { t = 0 } ^ { T - 1 } a _ { g } ^ { t } ( 1 - \eta ) ^ { E _ { g } ( t + 1 , T ) } \| c _ { g } ^ { t } \| _ { 2 } \leq r ,\tag{9}
$$

then $\| w _ { m ( q ) } ^ { T } - \theta _ { g } \| _ { 2 } \leq r . \ I f r < \Delta / 2$ , terminal greedy routing strictly separates the two groups. The proof and excess risk conclusion appear in Appendix A.3.

## 3 Adaptive routing control

Theorem 3 identifies three properties that routing must produce. Exposure activates a matched cluster, making $a _ { g } ^ { t } \ = \ 1$ so its error can contract. Persistence accumulates enough active rounds in $E _ { g } ( 0 , T )$ to reduce the initial error term in (9). Selectivity keeps $\| c _ { g } ^ { t } \| _ { 2 }$ small by directing the cluster's updates primarily from its compatible group. At the controller level, exploration creates exposure, exploitation of a useful lowest loss model promotes selectivity, and hysteresis makes an intervention persistent.

A routing controller $\phi _ { r }$ maps client ¿’s observable history $H _ { i } ^ { t }$ to a distribution over cluster routes:

$$
\pi _ { i } ^ { t } = \phi _ { r } ( H _ { i } ^ { t } ) \in \Delta _ { K } .\tag{10}
$$

Let $k _ { i } ^ { ( 1 ) } ( t )$ and $k _ { i } ^ { ( 2 ) } ( t )$ denote client i’s lowest and second lowest loss clusters. The Greedy controller is

$$
\phi _ { \mathrm { g r } } ( H _ { i } ^ { t } ) = e _ { k _ { i } ^ { ( 1 ) } ( t ) } .\tag{11}
$$

It exploits the current loss ordering by routing only to the lowest loss model. Once distinct useful models have formed, this promotes selective group specific updates. Under collapsed initialization, however, it gives the unused model no exposure and leaves its activity equal to zero.

The Rank Hedge controller is

$$
\phi _ { \mathrm { r h } } ( H _ { i } ^ { t } ) = 0 . 8 e _ { k _ { i } ^ { ( 1 ) } ( t ) } + 0 . 2 e _ { k _ { i } ^ { ( 2 ) } ( t ) } .\tag{12}
$$

Its second ranked weight explores an alternative cluster and can activate a model that Greedy starves. This exposure does not by itself ensure selectivity: if incompatible clients are redirected to the same model, their contributions increase $\| c _ { g } ^ { t } \| _ { 2 }$ in Theorem 3 and can oppose recovery.

The controller distribution is not used as a soft aggregation weight. Client i samples one cluster and sends a one hot executed route,

$$
a _ { i } ^ { t } \sim \pi _ { i } ^ { t } , \qquad q _ { i } ^ { t } = e _ { a _ { i } ^ { t } } ,\tag{13}
$$

then trains only $\boldsymbol { w _ { a _ { i } ^ { t } } ^ { t } }$ . The feedback assigned to the selected controller arm is the validation loss expected over this hard cluster draw,

$$
z _ { i } ^ { t } = \sum _ { k = 1 } ^ { K } \pi _ { i k } ^ { t } \widehat { L } _ { i } ( w _ { k } ^ { t } ) .\tag{14}
$$

Thus controller feedback averages the losses of possible hard routes, whereas the server sees only the sampled route $q _ { i } ^ { t }$ and the corresponding local model update.

For $S _ { t , k } = \{ i : a _ { i } ^ { t } = k \}$ and $\begin{array} { r } { N _ { t , k } = \sum _ { i \in S _ { t , k } } n _ { i } } \end{array}$ , the server performs hard sample weighted aggregation:

$$
w _ { k } ^ { t + 1 } = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { N _ { t , k } } \sum _ { i \in S _ { t , k } } n _ { i } \widetilde { w } _ { i } ^ { t } , } & { N _ { t , k } > 0 , } \\ { \displaystyle w _ { k } ^ { t } , } & { N _ { t , k } = 0 . } \end{array} \right.\tag{15}
$$

This is the empirical counterpart of $( 7 ) { \mathrm { : } }$ the sampled one hot routes determine $M _ { g k } ^ { t } , s _ { k } ^ { t }$ , and $\beta _ { g k } ^ { t }$ , while $\pi _ { i } ^ { t }$ never enters the server average.

Because the useful balance between exposure and selectivity depends on the unknown routing geometry and changes with the cluster models, each client independently runs EXP3 over a small controller menu [2]. Outside FEMNIST, $\Pi = \{ \phi _ { \mathrm { g r } } , \phi _ { \mathrm { r h } } \}$ . At block b, client i forms $p _ { i } ^ { b }$ , samples $R _ { i } ^ { b } \sim p _ { i } ^ { b }$ , holds that controller

throughout the block, sets $\begin{array} { r } { \ell _ { i } ^ { b } = | \mathcal { T } _ { b } | ^ { - 1 } \sum _ { t \in \mathcal { T } _ { b } } z _ { i } ^ { t } } \end{array}$ , and updates

$$
\begin{array} { r l r } & { } & { p _ { i r } ^ { b } = ( 1 - \gamma ) \frac { \omega _ { i r } ^ { b } } { \sum _ { s } \omega _ { i s } ^ { b } } + \frac { \gamma } { | \Pi | } , } \\ & { } & { \omega _ { i r } ^ { b + 1 } = \omega _ { i r } ^ { b } \exp \left( - \eta _ { \mathrm { p o l } } \frac { \mathbf { 1 } \left\{ R _ { i } ^ { b } = r \right\} \ell _ { i } ^ { b } } { p _ { i r } ^ { b } } \right) . } \end{array}\tag{16}
$$

Blocking gives a selected intervention time to alter the shared models before its feedback is evaluated. The exploratory component of (16) maintains opportunities for exposure, while learned controller weights can shift probability toward Greedy when the recovered models make selective exploitation effective. Since controller choices alter future model states and losses, ordinary EXP3 regret does not guarantee recovery [1]. Theorem 3 instead evaluates whether the realized routing sequence supplies sufficient exposure and persistence while controlling incompatible forcing.

Algorithm 1 gives the complete training order. At the start of each block, every client samples one controller arm and retains it throughout the block. In each communication round, the server broadcasts the current cluster models, each client samples one hard cluster route and trains only that model, and the server aggregates the returned models by sampled cluster using client sample counts. At the end of the block, each client updates EXP3 using its average expected hard route feedback.

Server executes: initialize $W ^ { 0 }$ and $\omega _ { i r } ^ { 0 }  1$ for every client i and controller $r \in \Pi ;$   
for blocks $b = 0 , 1 , \ldots$ do   
for each client i in parallel do   
compute $p _ { i } ^ { b }$ from $( 1 6 ) \div$ sample $R _ { i } ^ { b } \sim p _ { i } ^ { b } ;$   
for $t \in \mathcal T _ { b }$ do   
broadcast $W ^ { t } ;$   
for each client i in parallel do   
$\begin{array} { r } { ( a _ { i } ^ { t } , q _ { i } ^ { t } , n _ { i } , \widetilde { w } _ { i } ^ { t } , z _ { i } ^ { t } ) \gets \mathrm { C L I E N T U P D A T E } ( i , W ^ { t } , R _ { i } ^ { b } , H _ { i } ^ { t } ) ; } \end{array}$   
for each cluster k do   
$S _ { t , k } \gets \{ i : a _ { i } ^ { t } = k \}$ and $\begin{array} { r } { N _ { t , k } \gets \sum _ { i \in S _ { t , k } } n _ { i } ; } \end{array}$ update $w _ { k } ^ { t + 1 }$ by (15);   
for each client i in parallel do   
$\begin{array} { r } { \ell _ { i } ^ { b } \gets | \mathcal { T } _ { b } | ^ { - 1 } \sum _ { t \in \mathcal { T } _ { b } } \dot { z } _ { i } ^ { t } ; } \end{array}$ update $\{ \omega _ { i r } ^ { b + 1 } : r \in \Pi \}$ by (16);   
CLIENTUPDATE $( i , \tilde { W } ^ { t } , R _ { i } ^ { b } , H _ { i } ^ { t } ) \colon \pi _ { i } ^ { t } \gets \phi _ { R _ { i } ^ { b } } ( H _ { i } ^ { t } )$ ; sample $a _ { i } ^ { t } \sim \pi _ { i } ^ { t }$ and set $q _ { i } ^ { t }  e _ { a _ { i } ^ { t } } ; \widetilde { w } _ { i } ^ { t }  \mathrm { L o c a l U p d a t e } _ { i } ( w _ { a _ { i } ^ { t } } ^ { t } ) ;$   
$\begin{array} { r } { z _ { i } ^ { t } \gets \sum _ { k = 1 } ^ { K } \pi _ { i k } ^ { t } \widehat { L } _ { i } ( w _ { k } ^ { t } ) ; \mathbf { r e t u r n } ( a _ { i } ^ { t } , q _ { i } ^ { t } , n _ { i } , \widetilde { w } _ { i } ^ { t } , z _ { i } ^ { t } ) } \end{array}$

## Algorithm 1: Blocked decentralized routing control.

With a slowly trained neural model, isolated exploration may create activity without providing enough persistent compatible updates for the model to become competitive. The selective hysteresis controller therefore treats an entered target $c _ { i }$ as a temporary commitment. Let $d _ { i } ^ { t }$ count client $i \ ' s$ routed updates to $c _ { i }$ and define the target's excess loss

$$
G _ { i } ^ { t } : = \widehat { L } _ { i } ( w _ { c _ { i } } ^ { t } ) - \operatorname* { m i n } _ { j } \widehat { L } _ { i } ( w _ { j } ^ { t } ) .\tag{17}
$$

The controller is

$$
\phi _ { \mathrm { h y s } } ( H _ { i } ^ { t } ) = \left\{ \begin{array} { l l } { e _ { c _ { i } } , } & { d _ { i } ^ { t } < H _ { \mathrm { m i n } } , } \\ { e _ { c _ { i } } , } & { H _ { \mathrm { m i n } } \leq d _ { i } ^ { t } < H _ { \mathrm { m a x } } \mathrm { a n d } G _ { i } ^ { t } > \epsilon _ { \mathrm { e x i t } } , } \\ { e _ { k _ { i } ^ { ( 1 ) } ( t ) } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{18}
$$

Thus an entered client remains on the target for at least $H _ { \mathrm { m i n } }$ updates and subsequently continues while the target's excess loss exceeds $\epsilon _ { \mathrm { e x i t } }$ , subject to the episode-specific cap $H _ { \mathrm { m a x } }$ [5]. The counter records completed target updates before the current routing decision. When the competitiveness or cap condition releases an episode, the client executes its Greedy route in that same round and becomes eligible for a new controller decision in the following round. The cap applies separately to every hysteresis episode, including episodes entered after EXP3 selection. Hysteresis supplies persistence, but does not itself ensure compatible update composition.

We address that entry problem in FEMNIST with one synchronized round. Each client first reports a provisional Greedy route $\widetilde { a } _ { i } ^ { 0 }$ , which is used only to measure the sample weighted provisional mass

$$
\widetilde \mu _ { k } ^ { 0 } : = \frac { 1 } { N } \sum _ { i } n _ { i } \mathbf { 1 } \{ \widetilde a _ { i } ^ { 0 } = k \} .\tag{19}
$$

After the server broadcasts these masses, client i tests a candidate underused cluster k by making a temporary local update $U _ { i } ( w _ { k } ^ { 0 } )$ and computing

$$
B _ { i k } ^ { 0 } : = \operatorname* { m i n } _ { j } \widehat { L } _ { i } ( w _ { j } ^ { 0 } ) - \widehat { L } _ { i } \big ( U _ { i } ( w _ { k } ^ { 0 } ) \big ) .\tag{20}
$$

The provisional routes are never aggregated. A client executes the candidate route only when $\widetilde \mu _ { k } ^ { 0 } < \mu _ { \mathrm { r e v } }$ and $B _ { i k } ^ { 0 } > \tau$ . Qualifying clients therefore enter the weak model in the same round, creating coordinated exposure from clients whose local tests indicate compatibility.

Algorithm 2 mirrors Algorithm 1, but inserts one synchronized entry round. Provisional Greedy reports determine only the broadcast masses; ENTRYUPDATE either executes a qualifying underused cluster route and enters forced hysteresis or executes the provisional Greedy route. The server aggregates these executed round zero updates using the same hard update as Algorithm 1. Forced entrants make no EXP3 weight updates during their entry episode. Upon release they execute Greedy, then become eligible for independent selection over $\Pi _ { \mathrm { F E M } } = \left\{ \phi _ { \mathrm { g r } } , \phi _ { \mathrm { h y s } } \right\}$ in the following round. FEMNIST uses client-specific actions rather than shared fixed blocks: a Greedy action lasts one round, whereas an entered hysteresis action persists until release. EXP3 receives the mean normalized training-loss feedback of each completed selected action, as specified in Appendix B.3. Thus entry supplies exposure and initial selectivity, while hysteresis supplies persistence. Further settings and interpretation appear in Appendix B.3.

Server executes synchronized entry:   
for each client i in parallel do   
report $\widetilde { a } _ { i } ^ { 0 } \in$ arg minj $\widehat { L } _ { i } ( w _ { j } ^ { 0 } ) ;$   
compute (19) and broadcast $\widetilde { \mu } ^ { 0 } ;$ these reports do not update any model   
for each client i in parallel do   
$( a _ { i } ^ { 0 } , q _ { i } ^ { 0 } , n _ { i } , \tilde { w } _ { i } ^ { 0 } , \dot { F _ { i } } , c _ { i } , d _ { i } ) \gets \mathrm { E N T R Y U P D A T E } ( i , W ^ { 0 } , \tilde { a } _ { i } ^ { 0 } , \tilde { \mu } ^ { 0 } ) ;$   
for each cluster k do   
$S _ { 0 , k } \gets \{ i : a _ { i } ^ { 0 } = k \}$ and $N _ { 0 , k } \gets \sum _ { i \in S _ { 0 , k } }$ ni; update wk by (15);   
ENTRYUPDATE $( i , W ^ { 0 } , \widetilde { a } _ { i } ^ { 0 } , \widetilde { \mu } ^ { 0 } )$   
$\mathcal { C } _ { i } ^ { 0 }  \{ k : \widetilde { \mu } _ { k } ^ { 0 } < \mu _ { \mathrm { r e v } } \} ;$   
if ${ \mathcal { C } } _ { i } ^ { 0 } \neq { \mathcal { D } } ,$ compute (20) and choose $c _ { i } ^ { 0 } \in \mathrm { a r g } \operatorname* { m a x } _ { k \in \mathcal { C } _ { i } ^ { 0 } } B _ { i k } ^ { 0 } ;$   
if $\mathcal { C } _ { i } ^ { 0 } \neq$ Ø and $B _ { i , c _ { i } ^ { 0 } } ^ { 0 } > \tau$ then   
$a _ { i } ^ { 0 }  c _ { i } ^ { 0 } ; \widetilde { w } _ { i } ^ { 0 }  U _ { i } ^ { ' } ( w _ { c _ { i } ^ { 0 } } ^ { 0 } ) ; ( F _ { i } , c _ { i } , d _ { i } )  ( 1 , c _ { i } ^ { 0 } , 1 ) ;$   
else   
$a _ { i } ^ { 0 }  \widetilde { a } _ { i } ^ { 0 } ; \widetilde { w } _ { i } ^ { 0 } $ LocalUpdate $( w _ { a _ { i } ^ { 0 } } ^ { 0 } ) ; F _ { i } \gets 0 ;$   
$q _ { i } ^ { 0 } \gets e _ { a _ { i } ^ { 0 } } ;$ return $( a _ { i } ^ { 0 } , q _ { i } ^ { 0 } , n _ { i } , \widetilde { w } _ { i } ^ { 0 } , F _ { i } , c _ { i } , d _ { i } ) ;$   
Server executes subsequent rounds:   
for rounds $t = 1 , \dots , T - 1$ do   
broadcast $W ^ { t } ;$   
for each client i in parallel do   
if an episode is active, apply (18); on release, execute Greedy in this round, close the episode, and set $F _ { i } \gets 0 ;$   
otherwise, sample a new arm from IIFEM using (16), with a client-specific action index;   
execute the selected Greedy route, hysteresis fallback, or committed route, and set q $\gets e _ { a _ { i } ^ { t } } ;$   
$\widetilde { w } _ { i } ^ { t } \gets \mathrm { L o c a l U p d a t e } _ { i } ( w _ { a _ { i } ^ { t } } ^ { t } )$ and compute $z _ { i , \mathrm { F E M } } ^ { t }$ by (22);   
accumulate feedback only for rounds belonging to the selected action; the Greedy exit round is excluded from the released   
episode;   
when an EXP3-selected action completes, update its weights by (16) using its mean accumulated cost and its sampling probabil  
ity; do not update EXP3 for the forced entry episode;   
an episode released this round is eligible for a new controller decision only at round $t + 1 ;$   
for each cluster k do   
$S _ { t , k } \gets \{ i : a _ { i } ^ { t } = k \}$ and $N _ { t , k } \gets \sum _ { i \in S _ { t , k } }$ ni; update $w _ { k } ^ { t + 1 }$ by (15);  
Algorithm 2: Synchronized entry followed by decentralized routing control.

## 4 Experiments

We evaluate decentralized online controller selection against Greedy IFCA and FedCAM on Linear regression and transformed FEMNIST; multiclass GMM and California Housing appear in Appendices B.4 and B.5. Because these predictors have different parameter spaces, their raw initialization scales are not comparable. We instead summarize the induced routing geometry by the round zero attraction ${ \bf A } _ { g k } ^ { 0 } : =$ $\begin{array} { r } { | I _ { g } | ^ { - 1 } \sum _ { i \in I _ { q } } \mathbf { 1 } \{ k = \arg \operatorname* { m i n } _ { j } \widehat { L } _ { i } ( w _ { j } ^ { 0 } ) \} } \end{array}$ . A collapsed entrance sends several groups to the same cluster and leaves another unused; we test whether adaptive routing can transform such an entrance into models that remain separated under controller free greedy evaluation.

Code and reproducibility. The complete code, tests, compact evidence, plotting scripts, and experiment runners used to reproduce the results and figures are available at https: //github.com/prsinha1/ FedRecover. The experiments reported in this paper correspond to release v1 . 0 .1. The complete reproducibility package is the asset FedRecover\_reproducibility\_v1 . 0 .1 . zip attached to release v1 . 0 . 1. GitHub's automatically generated “Source code" archives should not be used.

Linear Regression: We use 80 clients balanced across the two linear groups of Section 2 and initialize $w _ { k } ^ { 0 } = \bar { \theta } + \rho z _ { k }$ , with $z _ { k } \sim \mathcal { N } ( 0 , I )$ , so ρ controls spread around the pooled parameter. The audited common preference frequency increases from 0.10 to 1.00 across the tested $\rho$ sweep. Using the recovery error $R _ { T }$ from (2), Figure 2(a) reports the recovery rate, $\widehat { \mathrm { P r } } ( R _ { T } < 0 . 5 )$ . Greedy IFCA and FedCAM become less reliable as $\rho$ increases, while EXP3 recovers every evaluated Greedy failure cell; the dashed curve is the certificate based upper bound on Greedy recovery.

![](images/a19d55b0ebf8211980b6b2c966e5d588588c60976c9f3c9d52c88e43424a27f6.jpg)  
(a) Recovery sweep

![](images/463326d19db740992355cee18163c9c4738b928ea9fc7bcb9edb5f650ffb6a78.jpg)  
(b) Routed test loss

![](images/410ea0b7e104061dab7af1f6f10f70494b3045889a27bb751faeae5cd777dc99.jpg)  
(c) Weak cluster activity

![](images/ad04a3112cc959422d2eda32da9513ce8530ebbde34ef351acf87e0589391aef.jpg)  
(d) Update composition

![](images/9040e20fedd8c9c182f957fa7114a24ea9488b6f5003b64310b746bfb6ec3074.jpg)  
(e) Usability across α

![](images/5ef99c77091ccedb72d96a39b09126a88ae92325d9274a011989e1784036764e.jpg)  
(f) Executed activity

![](images/83d6a0c27ab5849a217ee48e4cf9f3225719d124836c4bca96b0b00e59ed0bc2.jpg)  
(g) Model usability

![](images/5689aaefe29711f6169799b93c588f908e454bffdf1dfd498b766a9325f9637e.jpg)  
(h) Specialization  
Figure 2: Recovery from collapsed initialization in linear (a:d) and FEMNIST (e:h) settings.

On a representative failed start, panel (b) plots routed test loss $\begin{array} { r } { L _ { t } ^ { \mathrm { r o u t e } } : = n ^ { - 1 } \sum _ { i } L _ { i , \mathrm { t e s t } } ( w _ { r _ { i } ^ { t } } ^ { t } ) } \end{array}$ . Panel (c) shows that only EXP3 restores weak cluster mass $\begin{array} { r } { \mu _ { \mathrm { w e a k } } ^ { t } : = \sum _ { g } \alpha _ { g } M _ { g , \mathrm { w e a k } } ^ { t } . } \end{array}$ and hence weak cluster activity $s _ { \mathrm { w e a k } } ^ { t } = \mathbf { 1 } \{ \mu _ { \mathrm { w e a k } } ^ { t } > 0 \}$ . Panel (d) shows that its updates become dominated by the compatible group, with $\beta _ { g ^ { \star } , \mathrm { w e a k } } ^ { t }  1$ and $\begin{array} { r } { \sum _ { h \ne g ^ { \star } } \beta _ { h , \mathrm { w e a k } } ^ { t }  0 } \end{array}$ . Thus EXP3 creates the activity and compatible update composition identified by Equation (8).

FEMNIST Classification: We use 10 clean clients and 10 clients with digits 6 and 9 rotated by $1 8 0 ^ { \circ }$ without relabeling. Each model is a width 64 ReLU network with one hidden layer. We set $w _ { 0 } ^ { 0 } = w _ { \mathrm { m i x } }$ and $w _ { 1 } ^ { 0 } = ( 1 - \alpha ) \xi + \alpha w _ { \mathrm { t r a n s } }$ , where $w _ { \mathrm { m i x } }$ and $w _ { \mathrm { t r a n s } }$ are pretrained on pooled and transformed clients and $\xi$ is a weak base model. Increasing α moves the unused model toward $w _ { \mathrm { t r a n s } } ,$ yet $\mathbf { A } ^ { 0 } = { \bigl ( } { } _ { 1 } ^ { 1 } \mathbf { \Lambda } _ { 0 } ^ { 0 } { \bigr ) }$ for every tested α. All four methods run for 800 communication rounds. The synchronized-entry configuration uses $H _ { \mathrm { m a x } } = 8 0 0$ , while autonomous EXP3 uses $H _ { \mathrm { m a x } } = 1 6 0 ;$ full controller settings appear in Appendix B.3.

$\mathbf { A } \mathbf { t } \ W ^ { 8 0 0 }$ , each client selects its lowest held-out-loss model, $r _ { i } ^ { 8 0 0 , \mathrm { e v a l } } \ \in \ \mathrm { a r g } \operatorname* { m i n } _ { k } \widehat { L } _ { i , \mathrm { t e } } ( w _ { k } ^ { 8 0 0 } )$ , and $K _ { 8 0 0 } ^ { \mathrm { e v a l } } : = | \{ r _ { i } ^ { 8 0 0 , \mathrm { e v a l } } : i \in [ n ] \} |$ counts the selected models. Panel (e) reports terminal two-model usability: Greedy IFCA and FedCAM remain at zero, autonomous EXP3 improves with $\alpha ,$ and synchronized entry reaches a recovery fraction of one from $\alpha = 0 . 5 5$ onward. $\mathbf { A } \mathbf { t } \ W ^ { 8 0 0 }$ , 42 of the 54 synchronized-entry configurations have two usable models on both training and held-out data. Panels (f), (g), and (h) show the representative run at $\alpha = 0 . 6 0$ , data seed 1 and selector seed 0. Panel (f) reports $K _ { t } ^ { \mathrm { r o u t e } }$ : synchronized entry routes updates to both models throughout the 800 rounds, autonomous EXP3 does so in 798 rounds, and the baselines use one model. Panel (g) reports training-loss-based, controller-free model usability: autonomous EXP3 remains at one usable model, whereas synchronized entry reaches two at $W ^ { 6 1 2 }$ and retains both through the plotted horizon. Panel (h) reports permutation-invariant assignment accuracy, $\begin{array} { r } { A _ { t } ^ { \mathrm { e v a l } } : = \operatorname* { m a x } _ { m \in \mathfrak { S } _ { 2 } } \frac { 1 } { 2 } \sum _ { g } | I _ { g } | ^ { - 1 } \sum _ { i \in I _ { q } } \mathbf { 1 } \{ r _ { i } ^ { t , \mathrm { e v a l } } = m ( g ) \} } \end{array}$ . Synchronized entry raises this score from $1 / 2$ to 0.95 at the final plotted state $W ^ { 7 9 9 }$ , while autonomous EXP3 remains at $1 / 2 .$ Thus executed activity, two-model usability, and exact group specialization are distinct: synchronized entry with persistent commitment produces two usable models in this run, but does not achieve perfect group specialization within the evaluated horizon.

## References

[1] Raman Arora, Ofer Dekel, and Ambuj Tewari. Online bandit learning against an adaptive adversary: From regret to policy regret, 2012. arXiv:1206.6400.

[2] Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32:48–77, 2002.

[3] Avishek Ghosh, Jichan Chung, Dong Yin, and Kannan Ramchandran. An efficient framework for clustered federated learning, 2021. arXiv:2006.04088.

[4] Harshvardhan, Avishek Ghosh, and Arya Mazumdar. An improved algorithm for clustered federated learning, 2022. arXiv:2210.11538.

[5] João P. Hespanha, Daniel Liberzon, and A. Stephen Morse. Hysteresis-based switching algorithms for supervisory control of uncertain systems. Automatica, 39:263–272, 2003.

[6] Jie Ma, Tianyi Zhou, Guodong Long, Jing Jiang, and Chengqi Zhang. Structured federated learning through clustered additive modeling. In Advances in Neural Information Processing Systems, volume 36, pages 43097–43107, 2023.

[7] Yishay Mansour, Mehryar Mohri, Jae Ro, and Ananda Theertha Suresh. Three approaches for personalization with applications to federated learning, 2020. arXiv:2002.10619.

[8] John Nguyen, Jianyu Wang, Kshitiz Malik, Maziar Sanjabi, and Michael Rabbat. Where to begin? on the impact of pre-training and initialization in federated learning, 2022. arXiv:2210.08090.

[9] Giulia De Pasquale, Sarah Dean, and Paolo Frasca. Recommender systems as control systems, 2026. arXiv:2605.01503.

## A Theory Supplement

This appendix proves the theoretical results stated in the main paper.

## A.1 Favorable greedy specialization

Proof of Theorem 1. Choose a matching m $\in { \mathfrak { S } } _ { 2 }$ witnessing $W ^ { 0 } \in B _ { \mathrm { s p e c i a l i z e } }$ and relabel the clusters so that $m ( g ) = g$ Let $r _ { g } = \| w _ { g } ^ { 0 } - \theta _ { g } \| _ { 2 }$ . Then $r _ { 0 } + r _ { 1 } < \Delta$ . For group 0, the reverse triangle inequality gives

$$
\begin{array} { r } { \| w _ { 1 } ^ { 0 } - \theta _ { 0 } \| _ { 2 } \geq \Delta - r _ { 1 } > r _ { 0 } = \| w _ { 0 } ^ { 0 } - \theta _ { 0 } \| _ { 2 } . } \end{array}
$$

The analogous inequality holds for group 1, so the initial greedy routing is separated. If routing remains separated through round $t ,$ cluster g receives only the group g gradient, and therefore

$$
\boldsymbol { w } _ { g } ^ { t + 1 } - \boldsymbol { \theta } _ { g } = ( 1 - \eta ) ( \boldsymbol { w } _ { g } ^ { t } - \boldsymbol { \theta } _ { g } ) , \qquad \| \boldsymbol { w } _ { g } ^ { t } - \boldsymbol { \theta } _ { g } \| _ { 2 } = ( 1 - \eta ) ^ { t } \boldsymbol { r } _ { g } .
$$

At every round,

$$
\| w _ { 1 - g } ^ { t } - \theta _ { g } \| _ { 2 } \ge \Delta - \| w _ { 1 - g } ^ { t } - \theta _ { 1 - g } \| _ { 2 } \ge \Delta - r _ { 1 - g } > r _ { g } \ge \| w _ { g } ^ { t } - \theta _ { g } \| _ { 2 } .
$$

Thus the separated preference is preserved by induction. Since $( 1 - \eta ) ^ { t } \to 0$ , each cluster converges to its matched specialist. □

## A.2 Proof of greedy starvation

Proof of Theorem 2. It suffices to prove the result for $j = 1 ;$ the case $j = 0$ follows by exchanging the cluster labels. Membership in $ { \beta _ { \mathrm { s t a r v e } , 1 } }$ implies $L _ { g } ( w _ { 0 } ^ { 0 } ) < L _ { g } ( w _ { 1 } ^ { 0 } )$ for both groups. Hence both groups initially select cluster 0, while cluster 1 receives no update.

Suppose this common preference has persisted through round t. Then $w _ { 1 } ^ { t } = w _ { 1 } ^ { 0 }$ , while the active cluster follows

$$
w _ { 0 } ^ { t + 1 } = w _ { 0 } ^ { t } - \eta ( w _ { 0 } ^ { t } - \bar { \theta } _ { \alpha } ) , \qquad w _ { 0 } ^ { t } = \bar { \theta } _ { \alpha } + ( 1 - \eta ) ^ { t } ( w _ { 0 } ^ { 0 } - \bar { \theta } _ { \alpha } ) .
$$

Thus $w _ { 0 } ^ { t }$ lies on the segment joining $w _ { 0 } ^ { 0 }$ and $\theta _ { \alpha }$ . Convexity of $L _ { g }$ gives

$$
L _ { g } ( w _ { 0 } ^ { t } ) \leq \operatorname* { m a x } \{ L _ { g } ( w _ { 0 } ^ { 0 } ) , L _ { g } ( \bar { \theta } _ { \alpha } ) \} < L _ { g } ( w _ { 1 } ^ { 0 } ) = L _ { g } ( w _ { 1 } ^ { t } ) .
$$

Both groups therefore continue to select cluster 0. Induction proves that $w _ { 1 } ^ { t } = w _ { 1 } ^ { 0 }$ for every t, and the explicit recursion gives $w _ { 0 } ^ { t } \to \bar { \theta } _ { \alpha }$ □

## A.3 Finite horizon recovery under arbitrary routing

Fix a matching m and recall that $e _ { g } ^ { t } = w _ { m ( g ) } ^ { t } - \theta _ { g } , a _ { g } ^ { t } = s _ { m ( g ) } ^ { t }$ , and $\begin{array} { r } { c _ { g } ^ { t } = \sum _ { h \neq g } \beta _ { h , m ( g ) } ^ { t } ( \theta _ { h } - \theta _ { g } ) } \end{array}$ . Recall $\begin{array} { r } { E _ { g } ( t , T ) = \sum _ { \tau = t } ^ { T - 1 } a _ { g } ^ { \tau } } \end{array}$ , the number of active updates received by the matched cluster. Repeated substitution in Equation (8) gives the exact response

$$
e _ { g } ^ { T } = ( 1 - \eta ) ^ { E _ { g } ( 0 , T ) } e _ { g } ^ { 0 } + \eta \sum _ { t = 0 } ^ { T - 1 } a _ { g } ^ { t } ( 1 - \eta ) ^ { E _ { g } ( t + 1 , T ) } c _ { g } ^ { t } .\tag{21}
$$

Proof of Theorem 3. Equation (21) follows by repeatedly applying Equation (8). Each active round multiplies the existing matched error by 1 — η and adds $\eta c _ { g } ^ { t } .$ , while each inactive round leaves the error unchanged.

Taking norms in Equation (21) and applying the triangle inequality gives

$$
\| e _ { g } ^ { T } \| _ { 2 } \le ( 1 - \eta ) ^ { E _ { g } ( 0 , T ) } \| e _ { g } ^ { 0 } \| _ { 2 } + \eta \sum _ { t = 0 } ^ { T - 1 } a _ { g } ^ { t } ( 1 - \eta ) ^ { E _ { g } ( t + 1 , T ) } \| c _ { g } ^ { t } \| _ { 2 } .
$$

Condition (9) therefore implies $\| w _ { m ( g ) } ^ { T } - \theta _ { g } \| _ { 2 } = \| e _ { g } ^ { T } \| _ { 2 } \leq r$ The excess risk conclusion follows from

$$
L _ { g } ( w _ { m ( g ) } ^ { T } ) - L _ { g } ( \theta _ { g } ) = \frac { 1 } { 2 } \| w _ { m ( g ) } ^ { T } - \theta _ { g } \| _ { 2 } ^ { 2 } .
$$

Finally, if $r < \Delta / 2$ , then

$$
\| w _ { m ( 1 - g ) } ^ { T } - \theta _ { g } \| _ { 2 } \ge \Delta - \| w _ { m ( 1 - g ) } ^ { T } - \theta _ { 1 - g } \| _ { 2 } \ge \Delta - r > r \ge \| w _ { m ( g ) } ^ { T } - \theta _ { g } \| _ { 2 } .
$$

Thus each group strictly prefers its matched model under terminal greedy routing.

Equation (9) separates the two routing requirements. The first term contracts only when the matched cluster receives repeated updates. The second term measures incompatible contributions, discounted by later active updates. Their combination is the persistent selective exposure condition used in the main paper.

## B Experimental Supplement

## B.1 Shared evaluation and decentralized controller selection

Within each experimental cell, all methods use the same clients, corresponding initial cluster models, and local optimization budget. The FEMNIST horizon and evaluation-state conventions are specified below. We record executed routes during training and controller free greedy routes after freezing the evaluated models. Linear and GMM use $\{ \phi _ { \mathrm { g r } } , \phi _ { \mathrm { r h } } \}$ , Housing uses $\{ \phi _ { \mathrm { g r } } , \phi _ { \mathrm { c o m } } , \phi _ { \mathrm { r e v } } \}$ , and autonomous FEMNIST uses $\{ \phi _ { \mathrm { g r } } , \phi _ { \mathrm { h y s } } \}$ . In the fixed-block experiments, a selected arm remains fixed within each block ${ \mathcal { T } } _ { b } .$ with feedback averaged over that block. FEMNIST instead uses client-specific action durations and normalized training-loss feedback. Algorithm 1 specifies fixed-block client controller selection, hard route sampling, local training, sample weighted server aggregation, and the block level EXP3 update; Algorithm 2 specifies the FEMNIST action timing.

## B.2 Linear supplementary evidence

Under the pooled centered initialization of Section 4, the audited fraction of initializations for which both groups prefer the same cluster increases from 0.10 to 1.00 across the tested $\rho$ sweep. Figure B.1 asks whether the online selector maintains a fixed mixture of Greedy and Rank Hedge or adapts controller use as training evolves. For each controller $r ,$ it reports the client averaged EXP3 probability $\begin{array} { r } { \bar { p } _ { r } ^ { b } = n ^ { - 1 } \sum _ { i } p _ { i r } ^ { b } } \end{array}$ and the executed selection fraction $n ^ { - 1 } \textstyle \sum _ { i } \mathbf { 1 } \{ R _ { i } ^ { b } = r \}$ , smoothed over five blocks and averaged across 147 Greedy failure cells. EXP3 begins with approximately equal probability on both controllers, so Rank Hedge initially supplies alternative routing. The median cell recovers by block 1. After recovery, the mean Greedy probability rises steadily from approximately 0.5 to 0.9, while the Rank Hedge probability falls to approximately 0.1. The executed selection fractions closely track these learned probabilities. Thus the online learner does not maintain a fixed exploration rate: once the recovered models make lowest loss routing effective, it increasingly selects Greedy and suppresses unnecessary redirection. Because most of this adaptation occurs after the median recovery block, the figure demonstrates adjustment after recovery rather than showing that EXP3 learned the escaping controller before recovery.

![](images/1557c9e083e5a5ba8a7a098e25275e94b055223364b75d6f5e89d8bd1ab53904.jpg)  
Figure B.1: Linear controller adaptation over Greedy failure cells.

## B.3 FEMNIST synchronized-entry specialist construction

The round zero audit covers every $\alpha \in \{ . 4 5 , . 5 0 , . 5 5 , . 6 0 , . 6 5 , . 7 0 \}$ and data seed in {0, 1, 2}. In all 18 cells, both groups prefer cluster 0, so $\mathbf { A } ^ { 0 } = { \bigl ( } { } _ { 1 } ^ { 1 } \mathbf { \mathbf { \Phi } } _ { 0 } ^ { 0 } { \bigr ) } , K _ { 0 } = 1$ , and the common preference frequency is 1.00.

All four FEMNIST methods are evaluated for $T = 8 0 0$ rounds, indexed $t = 0 , \ldots , 7 9 9$ , producing terminal models $W ^ { 8 0 0 }$ . The initialization construction is given in Section 4. Model parameters use float32 arithmetic, and each executed client update uses one local SGD step with learning rate 0.003. We evaluate six initialization values, $\alpha \in \{ 0 . 4 5 , 0 . 5 0 , 0 . 5 5 , 0 . 6 0 , 0 . 6 5 , 0 . 7 0 \}$ , and three data seeds. Each adaptive method additionally uses three selector seeds, giving 54 runs per adaptive method and 18 runs per baseline, or 144 runs overall. The nine adaptive runs at each α comprise three data contexts with three selector seeds each, rather than nine independent datasets.

Both adaptive methods use $H _ { \mathrm { m i n } } = 5 , \epsilon _ { \mathrm { e x i t } } = 0 . 0 0 5 , \mu _ { \mathrm { r e v } } = 0 . 2 0 , \tau = - 0 . 7$ , and EXP3 parameters $\eta _ { \mathrm { E X P 3 } } = 2 . 0$ and $\gamma = 0 . 0 0 1$ . The synchronized-entry method uses $H _ { \mathrm { m a x } } = 8 0 0$ , whereas autonomous EXP3 uses $H _ { \mathrm { m a x } } = 1 6 0$ . Consequently, this comparison evaluates these controller configurations and does not isolate synchronization under matched commitment caps.

Selective hysteresis acts only after client i has entered an episode with target $c _ { i } ;$ it does not choose the target. Equations (17) and (18) give the routing rule. The entry update initializes the target-update counter to one. At each subsequent routing decision the counter records completed target updates. After the minimum commitment period, an episode ends when $G _ { i } ^ { t } \leq \epsilon _ { \mathrm { e x i t } }$ or its own counter reaches $H _ { \mathrm { m a x } }$ . The client executes Greedy in that same exit round and becomes eligible for a new controller decision in the following round. These rules apply both to the synchronized-entry episode and to later EXP3-selected episodes.

Since training executes rounds 0, . . . , 799, an episode beginning at round zero cannot reach an age-800 cap-triggered exit within the observed horizon. Competitiveness-triggered exits can occur earlier. Therefore, successful cap-800 recovery does not by itself demonstrate successful recovery after a forced cap-triggered handoff to EXP3.

Algorithm 2 coordinates only the initial entry. Provisional Greedy choices measure model use but never train a model. The server broadcasts their sample weighted masses, revealing underused candidates. Each client temporarily updates a candidate and evaluates the benefit in (20). Qualifying clients execute the candidate route together in round 0 and initialize $( c _ { i } , d _ { i } )$ ; the others execute their provisional Greedy routes. During the forced hysteresis episode, the client accumulates target updates but does not update EXP3. Upon release it executes Greedy, then becomes eligible for decentralized controller selection in the following round.

A Greedy action lasts one round. Selecting hysteresis either produces a one-round Greedy fallback or begins an episode that persists until its exit condition is met. EXP3 receives feedback only when its selected action completes; unfinished episodes are right-censored at the evaluation horizon. The synchronized entry episode supplies no EXP3 weight update. Define

$$
\begin{array} { r } { \displaystyle s _ { i } = \operatorname* { m a x } \left\{ \operatorname* { m i n } _ { k } \widehat { L } _ { i , \mathrm { t r } } ( w _ { k } ^ { 0 } ) , 1 0 ^ { - 1 2 } \right\} , } \\ { \displaystyle z _ { i , \mathrm { F E M } } ^ { t } = \frac { 1 } { 1 0 } \exp \left( \frac { \widehat { L } _ { i , \mathrm { t r } } ( w _ { a _ { i } ^ { t } } ^ { t } ) } { s _ { i } } , 0 , 1 0 \right) . } \end{array}\tag{22}
$$

The completed action's feedback is the mean of these costs over its executed rounds, excluding the Greedy exit round of a released hysteresis episode. The EXP3 update uses the probability with which that action was sampled. Held-out losses are used for evaluation, not controller selection or local optimization.

Panel (e) reports the fraction of runs with two models selected by controller-free, lowest-held-out-loss evaluation at $W ^ { 8 0 0 }$ . Panels (f)–(h) show the pre-update trajectory $W ^ { 0 } , \ldots , W ^ { 7 9 9 }$ : executed routing activity in panel (f), and training-loss-based controller-free usability and assignment accuracy in panels (g) and (h). FedCAM evaluation uses its combined global-plus-cluster predictor.

Training recovery means $K _ { \mathrm { t r } } ( W ^ { 8 0 0 } ) = 2 .$ Joint training and held-out recovery requires $K _ { \mathrm { t r } } ( W ^ { 8 0 0 } ) =$ $K _ { \mathrm { t e } } ( W ^ { 8 0 0 } ) = 2$ . Sustained recovery requires this joint condition at every state $W ^ { 7 8 1 } , \ldots , W ^ { 8 0 0 }$ . Exact specialization additionally requires permutation-invariant assignment accuracy equal to one on both splits, with the maximizing permutation chosen separately for each split. These assignment scores describe recovery of the client groups and are distinct from digit-classification accuracy. Held-out model preferences are evaluated offline and do not determine training routes.

Figure B.2 reports the last logged executed-routing test loss,

$$
L _ { 7 9 9 } ^ { \mathrm { r o u t e } } = n ^ { - 1 } \sum _ { i } L _ { i , \mathrm { t e s t } } \Big ( w _ { r _ { i } ^ { 7 9 9 } } ^ { 7 9 9 } \Big ) .\tag{23}
$$

Curves show means across runs, and shaded regions show one standard deviation. This quantity evaluates the routes actually executed during training, rather than selecting a model by held-out loss. Synchronized entry incurs higher predictive cost than the baselines at smaller α, while its mean routed test loss falls below both baselines at $\alpha = 0 . 6 5$ and 0.70.

![](images/55b70edb87d8a962d75d1e11c765a5409d2ffe5bdc471ea6998851280552bff7.jpg)  
Figure B.2: FEMNIST predictive cost under executed routing.

## B.4 GMM collapse and recovery

We use a multiclass Gaussian mixture model with $G = 3$ latent groups, $C = 5$ classes, and feature dimension $d = 1 0$ . Each group $g \in \{ 1 , 2 , 3 \}$ has a feature center $\mu _ { g } \in \mathbb { R } ^ { d }$ , and its samples are generated as $\begin{array} { r } { x = \mu _ { g } + \sigma \xi , \qquad \xi \sim \mathcal { N } ( 0 , I _ { d } ) } \end{array}$ . Clients are pure: every client i belongs to one latent group $g _ { i }$ , and all of its samples follow the corresponding group distribution. We use ten clients per group.

Labels follow a mismatch model containing shared and group specific components. For class $c \in$ $\{ 1 , \ldots , C \}$ and group g, the generating logit and label are

$$
r _ { g , c } ( x ) = \alpha u _ { c } ^ { \top } x + \lambda v _ { g , c } ^ { \top } x + b _ { g , c } , \qquad y = \arg \operatorname* { m a x } _ { c \in \{ 1 , \ldots , C \} } r _ { g , c } ( x ) .
$$

Here $U \in \mathbb { R } ^ { C \times d }$ contains the shared class directions $u _ { c } , V _ { g } \in \mathbb { R } ^ { C \times d }$ contains the group specific directions $v _ { g , c } ,$ and $b _ { g } \in \mathbb { R } ^ { C }$ is the group specific class bias. Thus clients share part of the classification rule while retaining group dependent decision boundaries. The server trains three linear softmax models for 800 rounds. The far initialization $W _ { k } ^ { 0 } = { R } _ { \mathrm { f a r } } D _ { k }$ scales fixed, class centered directions $D _ { k }$ of unit Frobenius norm chosen independently of the group specific classifiers.

Figure B.3 reports the endpoint collapse frequency $\widehat { \mathrm { P r } } ( K _ { T } ^ { \mathrm { e v a l } } < 3 )$ , permutation invariant terminal assignment $A _ { T }$ , and final routed test loss. Greedy IFCA and FedCAM remain vulnerable across the radius sweep, while EXP3 usually restores three group specialization and substantially reduces routed loss on the Greedy collapse cells. The larger variation at $R _ { \mathrm { f a r } } ~ = ~ 1 2$ marks a harder recovery regime rather than a monotone initialization severity scale.

![](images/0f398693265b014703c140e7e5379a7772d39c3702f1a14b416cff28b272e427.jpg)

![](images/e16e25da14b24fa80d3db07bdc5900327a37154a35119812f29d84019ddd8740.jpg)

![](images/a09798b87b6a2fb92f8da83aeca6ee3007007599803ef0b0e6fa04cd5f906141.jpg)  
Figure B.3: GMM terminal structure and loss across the initialization radius.

## B.5 Housing collapse and recovery

We use California Housing as a real tabular regression benchmark. Each census block group has eight housing and demographic features, with median house value as the target. Let $q _ { 0 . 7 5 } ^ { \mathrm { t r } }$ be the global-training 0.75 quantile of average occupancy per household, AveOccup. We assign a block group to group 0 if AveOccup $< q _ { 0 . 7 5 } ^ { \mathrm { t r } }$ and to group 1 otherwise. The groups share the feature space and target but represent different occupancy regimes.

Each cluster model is a linear regressor with an intercept. We standardize the eight features using global-training statistics and append a constant coordinate, giving $x \in \mathbb { R } ^ { 9 }$ and $f _ { w } ( x ) = x ^ { \top } w$ . For client i, the empirical objective is $\begin{array} { r } { L _ { i } ( w ) = n _ { i } ^ { - 1 } \sum _ { i = 1 } ^ { n _ { i } } ( x _ { i j } ^ { \top } w - y _ { i j } ) ^ { 2 } } \end{array}$ . For group oracle parameters $\beta _ { 0 } ^ { \star }$ and $\beta _ { 1 } ^ { \star }$ , let $\Delta _ { \beta } = \| \beta _ { 0 } ^ { \star } - \beta _ { 1 } ^ { \star } \| _ { 2 }$ . Along fixed unit directions $d _ { g } ,$ define $u _ { g } ( R ) = \beta _ { g } ^ { \star } + R \Delta _ { \beta } d _ { g }$ and center the initial cluster models near $\begin{array} { r } { \bar { u } ( R ) = \frac { 1 } { 2 } \big ( u _ { 0 } ( R ) + u _ { 1 } ( R ) \big ) } \end{array}$

Figure B.4 reports $\tilde { \mathrm { P r } } ( K _ { T } ^ { \mathrm { e v a l } } = 2 )$ , the fraction of runs in which both frozen terminal models receive at least one controller free greedy client. EXP3 retains two usable models throughout the tested radius sweep, while Greedy IFCA and FedCAM increasingly collapse as the initialization radius R grows.

Housing uses $\Pi _ { \mathrm { h o u s e } } = \{ \phi _ { \mathrm { g r } } , \phi _ { \mathrm { r e v } } , \phi _ { \mathrm { c o m } } \}$ . For low mass revival, fix $\mu _ { \mathrm { r e v } } \in ( 0 , 1 )$ and define ${ \mathcal K } _ { \mathrm { w e a k } } ^ { t } = \{ k \in [ K ] : \mu _ { k } ^ { t - 1 } <$ $\mu _ { \mathrm { r e v } } \}$ . Let $k _ { \operatorname* { m i n } } ( t ) \in$ arg min $\cdot k \in [ K ] \ \mu _ { k } ^ { t - 1 }$ be a least used cluster. The revival controller is

![](images/103effea743803ade6912f184911be61e3ff31a8d4a4dfd0f91ef6f388594a7f.jpg)

$$
\phi _ { \mathrm { r e v } } ( H _ { i } ^ { t } ) = \left\{ \begin{array} { l l } { e _ { k _ { \mathrm { m i n } } ( t ) } , } & { \ K _ { \mathrm { w e a k } } ^ { t } \neq \emptyset , } \\ { e _ { k _ { i } ^ { ( 1 ) } ( t ) } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Figure B.4: Housing model usability.

Thus revival routes the client to the least used cluster whenever some cluster has mass below $\mu _ { \mathrm { r e v } } ;$ otherwise it uses Greedy.

A single nongreedy update may not make a weak model competitive, so the commitment controller lets client i retain target $c _ { i } ^ { t }$ for several rounds. Let $\boldsymbol { \tau } _ { i } ^ { t }$ be the remaining duration and $\kappa _ { \mathrm { c o m } } \geq 1$ the accepted loss tolerance. The controller is

$$
\phi _ { \mathrm { c o m } } ( H _ { i } ^ { t } ) = \left\{ \begin{array} { l l } { e _ { c _ { i } ^ { t } } , } & { \tau _ { i } ^ { t } > 0 \mathrm { a n d } \widehat L _ { i } ( w _ { c _ { i } ^ { t } } ^ { t } ) \leq \kappa _ { \mathrm { c o m } } \widehat L _ { i } \bigg ( w _ { k _ { i } ^ { ( 1 ) } ( t ) } ^ { t } \bigg ) , } \\ { e _ { k _ { i } ^ { ( 1 ) } ( t ) } , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

For a committed nongreedy route, the client stores its target and remaining duration. Several updates can then accumulate before reevaluation, while the loss tolerance prevents continued commitment to an excessively poor target.