# PAIRAUDIT: GUIDING HUMAN REVIEW WITH GRAPHTOKENS UNDER DISTRIBUTION SHIFT

Jiran Tao Binyan Jiang

The Hong Kong Polytechnic University

22117758r@connect.polyu.hk by.jiang@polyu.edu.hk

## ABSTRACT

Intrusion detectors can confidently misclassify attacks that were not seen during training. Human review can correct these errors, but only a limited number of cases can be checked. Uncertainty-based review may overlook confident errors, while anomaly scores alone do not show whether changing the review plan will correct more errors. We introduce PAIRAUDIT to find overlooked errors and improve review under a fixed budget. Its graph tokens capture prediction patterns across connected nodes. Rather than building another predictor through feature aggregation, PAIRAUDIT uses unusual relational patterns to uncover potential errors in existing predictions. Human feedback then helps decide whether these findings justify changing review priorities. Experiments across security tasks show that PAIRAUDIT corrects more errors on average than uncertainty-based review, including more errors on unseen attacks. These gains account for all review costs and do not require retraining the detector.

## 1 INTRODUCTION

A detector can work well on familiar cases and still make confident mistakes when new patterns appear. This is a practical problem under out-of-distribution (OOD) shift. In security, an unseen attack subtype may be mistaken for benign activity even though its correct label still belongs to a known category. Human review can correct these mistakes while the detector stays in use. The difficulty is that reviewers can check only a small part of the data, so errors outside their review queue may remain unnoticed.

A natural approach is to review the cases the detector is least certain about. This makes good use of its predictions, but can miss errors made with high confidence. OOD and error-detection scores offer other ways to order cases (Hendrycks & Gimpel, 2017; Liu et al., 2020; Jiang et al., 2018; Wu et al., 2023; Ma et al., 2024). Yet a different ranking is not automatically a better review plan. Bringing one case into a fixed budget means leaving another out. A useful method must find errors the current queue overlooks and help decide whether the proposed change is worth making.

Graph structure offers a way to find such cases. A node may look ordinary on its own while its prediction forms an unusual pattern with those of its neighbors. But disagreement alone is not enough: some kinds of nodes routinely differ from their neighbors. For example, a legitimate shared server may connect to many devices flagged as malicious. Its own benign prediction can still be correct, and such disagreement may be familiar for servers with that role. Treating these familiar patterns as suspicious can waste review effort. We therefore ask whether the observed relationship is expected for a node with those features, rather than treating every unusual-looking relationship in the same way.

We introduce PAIRAUDIT to connect this relational evidence with decisions about human review. Its graph tokens describe the predictions of a node and its neighborhood. A density model learns several familiar patterns within each known class, and node features determine how much weight each pattern receives. A strong disagreement can be unsurprising when it fits a familiar pattern for those features; a different relationship can still be flagged. The score is fitted and selected using known training and validation labels, without requiring OOD labels.

Uncertainty review can remain useful even when it misses confident errors. PAIRAUDIT therefore retains part of the original queue (Raw) and proposes replacements from outside its budget. Reviewers provide ordinary class labels, not OOD status or new subtype names. A sequential pair audit uses these labels to decide whether to adopt the remaining exchange. Retention, auditing and subsequent selection form one review policy. Both audited endpoints count toward the same budget. The objective is to correct more errors than the original queue after paying for the decision.

Our contributions are:

• Relational evidence for overlooked errors. We use graph tokens and feature-conditioned pattern densities to guide review of a frozen detector. This accounts for different familiar neighborhood patterns within a class and uses them to assess new cases, without retraining the detector or changing the review labels.

• An adaptive review policy with explicit costs. We combine Raw retention and sequential pair audits with exact budget and net gain identities. These identities account for both useful audit labels and the reviews they displace. We also characterize uniform pair sampling without assuming independent graph nodes, while keeping these guarantees distinct from the working posterior used for decisions.

• Consistent benefits across the tested attack shifts. All four primary tasks show positive mean overall and unseen-attack correction gains with high authorization rates. Decisions use about 5% of the budget on average. Additional tests show gains at low and high OOD prevalence and in multiclass review. The low-OOD test also corrects more errors on known cases. Controls assess robustness to pairing and exchange size and show early rejection of a harmful candidate.

## 2 RELATED WORK

OOD scores and review decisions. Maximum softmax probability and Energy provide uncertainty and OOD scores (Hendrycks & Gimpel, 2017; Liu et al., 2020). GNNSafe propagates energy scores over a graph, and GRASP studies when score propagation helps and augments it accordingly (Wu et al., 2023; Ma et al., 2024). Trust Score assesses predicted-class support through reference distances and explicitly motivates human review prioritization (Jiang et al., 2018). These methods can induce review rankings. Their scoring procedures, however, do not supply a sequential decision based on current analyst labels about whether to replace part of an existing queue under a shared budget. Their scores alone do not establish a positive correction gain on a new workload. Our policy retains part of Raw and uses paid feedback to guide replacements. Table 4 compares complete review policies at equal capacity. Table 4 documents how their correction benefits vary across the tested tasks.

Conditional anomaly and graph models. Conditional anomaly detection distinguishes unusual behavior from unusual context (Song et al., 2007). Conditional mixture densities and graph mixture density networks provide foundations for multimodal modeling (Bishop, 1994; Errica et al., 2021); graph neighborhood reconstruction exploits relational evidence (Roy et al., 2024). We do not claim these as new primitives. Our graph tokens and feature-conditioned densities model the relational behavior of a frozen classifier and supply proposals whose correction value is assessed through paid review.

Human feedback and allocation. Active testing allocates labels for model evaluation (Kossen et al., 2021); human feedback can improve OOD decisions (Vishwakarma et al., 2024; Bai et al., 2025); conservative policy improvement considers retaining a baseline with limited evidence (Laroche et al., 2019). Our feedback is the analyst’s ordinary class judgment. Its acquisition both corrects an object and consumes capacity that could have corrected another. Consequently, a favorable candidate score or a favorable remaining exchange does not by itself imply a beneficial complete workflow.

## 3 REVIEW OBJECTIVE AND AVAILABLE INFORMATION

The target pool $\mathcal { U } = \{ 1 , \ldots , N \}$ contains N nodes and K known review classes (e.g., benign and attack for $K = 2 ) . \operatorname { A }$ frozen classifier returns $p _ { i } = ( p _ { i 1 } , \ldots , p _ { i K } ) \colon p _ { i k }$ is its probability for class $k ,$ with $\textstyle \sum _ { k } p _ { i k } = 1$ . Its predicted label is $\hat { y } _ { i } = \arg \operatorname* { m a x } _ { k } p _ { i k }$ . Reviewers provide $y _ { i } \in \{ 1 , \ldots , K \}$

Assuming equal review cost per node, let $H < N$ be the maximum number of distinct nodes that the available human budget can cover. It limits how many samples reviewers can see and label. Define the Raw uncertainty score and baseline set by

$$
s _ { R } ( i ) = 1 - \operatorname* { m a x } _ { k } p _ { i k } , \qquad R _ { H } = \{ R ( 1 ) , \ldots , R ( H ) \} .
$$

Here $R = ( R ( 1 ) , \ldots , R ( N ) )$ orders all target nodes by decreasing $s _ { R } .$ , with node ID breaking ties. Under Raw, reviewers start with the least confident predictions and stop after H nodes: $R _ { H }$ is the set they can reach, while the remaining $N - H$ nodes stay unreviewed. PAIRAUDIT changes which nodes enter this human review window, not its size $H$

Let J rank all N nodes using the candidate score (Section B.1). Removing nodes already in $R _ { H }$ leaves the eligible pool $\boldsymbol { \mathcal { U } } \setminus \boldsymbol { R _ { H } }$ . For a preselected exchange fraction $\tau \in ( 0 , 1 )$ , the candidate set C contains the first $\bar { L ^ { } } = \operatorname* { m i n } ( \dot { N } - H , | \tau \bar { H } | )$ ) eligible nodes in J order. Thus $| C | = L$ and $C \cap R _ { H } = \emptyset$ Only these L candidates enter the paired comparison.

The reviewed set $S \subseteq \mathcal { U } , | S | = H$ , contains all nodes actually reviewed by PAIRAUDIT: paid audit nodes, retained Raw nodes, and the continuation selected by the audit. A corrected prediction earns one unit:

$$
u _ { i } = { \bf 1 } \{ y _ { i } \neq \hat { y } _ { i } \} , \qquad U ( S ) = \sum _ { i \in S } u _ { i } , \qquad G = U ( S ) - U ( R _ { H } ) .
$$

We compare complete review policies, whether they follow fixed rankings or adapt to paid labels. The objective is $\bar { G } > 0 ;$ more errors corrected with the same budget. Missed attacks and false alarms have equal reward. Paid audit nodes remain in $S$ even if replacement is declined.

Let $v _ { i } = 1$ for a subtype absent from training and validation, and 0 otherwise. Its contribution to net gain is

$$
G _ { \mathrm { O O D } } = \sum _ { i \in S } u _ { i } v _ { i } - \sum _ { i \in R _ { H } } u _ { i } v _ { i } .
$$

This measures corrected OOD errors, not OOD counts or new types.

Known training labels fit the classifier and density; known validation labels select density settings. Scores and configuration are frozen before review, which reveals only requested class labels. Unqueried labels and OOD flags $v _ { i }$ are reserved for offline evaluation. Unlabelled graph covariates may be transductive (Appendix D).

## 4 THE PAIRAUDIT MODEL

## 4.1 GRAPH TOKENS AND FEATURE-CONDITIONED DENSITIES

We construct a graph token from a node’s prediction and the average prediction of its available onehop neighbors. Let ${ \bar { p } } _ { i }$ be the mean class-probability vector of node i’s available one-hop neighbors, using $p _ { i }$ for an isolated node. Apply the isometric log-ratio transform (ilr) (Egozcue et al., 2003) to each vector, mapping its $K$ probabilities to $K - 1$ nonredundant coordinates:

$$
t _ { i } = [ \mathrm { i } \mathrm { l r } ( p _ { i } ) , \mathrm { i } \mathrm { l r } ( \bar { p } _ { i } ) ] , \qquad z _ { i } = ( t _ { i } - \mu _ { \mathrm { t r a i n } } ) / \operatorname* { m a x } ( s _ { \mathrm { t r a i n } } , 1 0 ^ { - 6 } ) .
$$

The graph token $t _ { i }$ has $d = 2 ( K - 1 )$ coordinates. Training means $\mu _ { \mathrm { t r a i n } }$ and standard deviations $s _ { \mathrm { t r a i n } }$ standardize it to $z _ { i } \in \mathbb { R } ^ { d }$ ; division and the floor are coordinate-wise.

To model familiar patterns in these tokens, we fit a Gaussian mixture within each known class. For known class $k ,$ let $T _ { k }$ and $V _ { k }$ contain its training and validation nodes. A Gaussian mixture model (GMM) uses $M _ { k }$ components indexed by $b = 1 , \bar { \dots } , M _ { k } . \ M _ { k }$ is the number ofGaussian components, not classes or attack subtypes; components approximate different patterns of node and neighborhood predictions within a class.

For each candidate value of $M _ { k }$ , hold that count fixed and fit the GMM on $T _ { k }$ . Component b has class-wide weight $\pi _ { k b } > 0$ , mean $\mu _ { k b } \in \mathbb { R } ^ { d }$ and positive definite covariance $\Sigma _ { k b } .$ , with $\begin{array} { r } { \sum _ { b } \pi _ { k b } = 1 } \end{array}$

The component density and training log likelihood to maximize are

$$
\phi _ { d } ( z ; \mu , \Sigma ) = \frac { \exp [ - \frac { 1 } { 2 } ( z - \mu ) ^ { \top } \Sigma ^ { - 1 } ( z - \mu ) ] } { ( 2 \pi ) ^ { d / 2 } | \Sigma | ^ { 1 / 2 } } ,
$$

$$
\ell _ { k } = \sum _ { i \in T _ { k } } \log \sum _ { b = 1 } ^ { M _ { k } } \pi _ { k b } \phi _ { d } ( z _ { i } ; \mu _ { k b } , \Sigma _ { k b } ) .
$$

Full covariance captures dependencies between token coordinates. After K-means initialization, expectation-maximization (EM) alternates two steps. The E step computes responsibility $\gamma _ { i b } .$ , the soft assignment of $z _ { i }$ to component b:

$$
\gamma _ { i b } = \frac { \pi _ { k b } \phi _ { d } ( z _ { i } ; \mu _ { k b } , \Sigma _ { k b } ) } { \sum _ { c = 1 } ^ { M _ { k } } \pi _ { k c } \phi _ { d } ( z _ { i } ; \mu _ { k c } , \Sigma _ { k c } ) } , \qquad i \in T _ { k } .
$$

Let $\begin{array} { r } { N _ { b } = \sum _ { i \in T _ { k } } } \end{array}$ γ<sub>ib</sub> be its effective membership. The M step updates

$$
\pi _ { k b } ^ { \mathrm { n e w } } = N _ { b } / | T _ { k } | , \qquad \mu _ { k b } ^ { \mathrm { n e w } } = N _ { b } ^ { - 1 } \sum _ { i \in T _ { k } } \gamma _ { i b } z _ { i } ,
$$

$$
\Sigma _ { k b } ^ { \mathrm { n e w } } = N _ { b } ^ { - 1 } \sum _ { i \in T _ { k } } \gamma _ { i b } \big ( z _ { i } - \mu _ { k b } ^ { \mathrm { n e w } } \big ) \big ( z _ { i } - \mu _ { k b } ^ { \mathrm { n e w } } \big ) ^ { \top } + \epsilon I ,
$$

where I is the d-dimensional identity and $\epsilon = 1 0 ^ { - 4 }$ prevents singular covariances. EM yields a local fit. After fitting each admissible count, select the fitted GMM with the highest mean log likelihood on $V _ { k } .$ . Thus training estimates parametersfor a fixed count; validation selects the count and its fitted parameters. The selected $M _ { k } , \pi _ { k b } , \mu _ { k b } , \Sigma _ { k b }$ are then held fixed.

The fitted mixture describes familiar patterns within each class. Its class-wide weights, however, may not suit every node. The weights $\pi _ { k b }$ describe how common each pattern is in class $k ,$ but may not suit every node. For example, disagreement with neighbors may be rare among benign nodes overall yet familiar for benign shared servers. A component describing this server pattern can have a small class-wide weight, making an ordinary server look surprising.

We use a feature gate $g _ { k } ( x ) = ( g _ { k 1 } ( x ) , \ldots , g _ { k M _ { k } } ( x ) )$ to adapt the weights to a node’s input features x. Its outputs are nonnegative and sum to one. For server-like features, it can assign more weight to the familiar server pattern. This changes the weighted sum of the fixed component densities, not the Gaussian means or covariances. The token must still fit that pattern to receive higher density.

Table 1: Notation. R, J rank all N nodes; C is the L-node candidate set, and $S$ is the H-node reviewed set.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\mathcal { U } , N , K$   $p _ { i } , p _ { i k } , \hat { y } _ { i } , y _ { i }$ </td><td>Target node pool, its size, and number of known review classes. Class probabilities, class-k probability, predicted label and human label.</td></tr><tr><td> $H , s _ { R } ( i ) , R , R _ { H }$ </td><td>Reviewable node limit, Raw uncertainty, full Raw ordering, and its first H nodes.</td></tr><tr><td> $S , u _ { i } , U ( S )$   $v _ { i } , G , G _ { \mathrm { O O D } }$ </td><td>Reviewed set  $( | S | = H ) ,$  correction indicator, and correction count. Unseen-subtype flag; overall and OOD net corrections relative to Raw.</td></tr><tr><td> $\bar { p } _ { i } , x _ { i } , z _ { i }$   $M _ { k } , \gamma _ { i b } , w _ { k b } ( x )$ </td><td>Mean neighbor posterior, input features and standardized graph token. Number of class-k Gaussian components, responsibility, and node-specific</td></tr><tr><td> $\pi _ { k b } , \mu _ { k b } , \Sigma _ { k b }$ </td><td>weight. Class-wide weight, mean and covariance of Gaussian component b in class k.</td></tr><tr><td> $g _ { k } ( x ) , \eta _ { k }$ </td><td>Feature gate and validation-selected conditioning strength.</td></tr><tr><td> $f _ { k } ( z \mid x ) , s _ { J } ( i )$ </td><td>Conditional graph token density and relational surprise score.</td></tr><tr><td> $q _ { i } , J$   $\tau , L$ </td><td>Priority-adjusted review score and its descending candidate ordering.</td></tr><tr><td> $C , D$ </td><td>Exchange fraction and block size  $L = \operatorname* { m i n } ( N - H , \lfloor \tau H \rfloor )$ </td></tr><tr><td> $B$ </td><td>Candidate set outside  $R _ { H }$  and proposed removals within  $R _ { H }$  , each of size  $L .$ </td></tr><tr><td></td><td>Shared Raw set  $R _ { H } \setminus D ;$  retain its first  $H - L - m$  nodes after m audited pairs.</td></tr><tr><td> $m , I _ { m } , \delta _ { j } , p _ { m }$ </td><td>Audited pair count and indices; pair reward difference; remaining-gain proba-</td></tr><tr><td> $A$ </td><td>bility.</td></tr><tr><td></td><td>Authorization indicator: 1 for adopting the remaining exchange, 0 otherwise.</td></tr></table>

Learn $g _ { k }$ from known training pairs $( x _ { i } , z _ { i } )$ by maximizing conditional log likelihood:

$$
\operatorname* { m a x } _ { g _ { k } } \sum _ { i \in T _ { k } } \log \sum _ { b = 1 } ^ { M _ { k } } \frac { g _ { k b } ( x _ { i } ) + \lambda \pi _ { k b } } { 1 + \lambda } \phi _ { d } ( z _ { i } ; \mu _ { k b } , \Sigma _ { k b } ) , \quad \lambda = 1 0 ^ { - 4 } .\tag{1}
$$

This objective rewards higher density for relationships observed with those features. A feature-tree gate is initialized from $\gamma _ { i b } .$ , then optimized for this likelihood (Appendix A).

Blend the learned weights with the class-wide weights:

$$
w _ { k b } ( x ) = ( 1 - \eta _ { k } ) \pi _ { k b } + \eta _ { k } \frac { g _ { k b } ( x ) + \lambda \pi _ { k b } } { 1 + \lambda } .
$$

Here $\lambda$ keeps weights positive, and $\eta _ { k }$ controls how strongly features adjust them. Although $\eta _ { k }$ is shared within class $k , w _ { k b } ( x )$ varies by node. The conditional density is

$$
f _ { k } ( z \mid x ) = \sum _ { b = 1 } ^ { M _ { k } } w _ { k b } ( x ) \phi _ { d } ( z ; \mu _ { k b } , \Sigma _ { k b } ) .
$$

Select the gate checkpoint and $\eta _ { k } \in \{ 0 , . 2 5 , . 5 , . 7 5 , 1 \}$ by mean conditional log likelihood on $V _ { k }$ . At $\eta _ { k } = 0$ , the class-wide mixture is recovered; with one component, weights cannot vary. Neither stage uses OOD labels or simulated audit gains.

The gate uses only $x ;$ each Gaussian still checks the actual relationship z. Familiar features cannot excuse a mismatching token. For fixed $\begin{array} { r } { x , \int f _ { k } ( z \mid x ) d z = \sum _ { b } w _ { k b } \dot { ( x ) } = 1 } \end{array}$ . Training rewards aggregate likelihood, not necessarily higher density for every node; high density does not certify a correct prediction.

We rank target nodes for review using the fitted conditional density, evaluating each token under its predicted class density since the true class is unknown. For a target node, use its predicted class and define surprise $a _ { i } = - \log f _ { \hat { y } _ { i } } ( z _ { i } \mid x _ { i } )$ . Higher surprise means lower density. Known validation nodes $v \in V _ { k }$ provide reference values $a _ { v } = -$ log $f _ { k } ( \bar { z } _ { v } \mid x _ { v } )$ . The ranking score is

$$
s _ { J } ( i ) = \frac { \sum _ { v \in V _ { k } } { \bf 1 } \{ a _ { v } \leq a _ { i } \} + \frac 1 2 + \frac { 1 } { 2 \pi } \arctan ( ( a _ { i } - m _ { k } ) / \sigma _ { k } ) } { | V _ { k } | + 1 } , \quad k = \hat { y } _ { i } ,
$$

where $m _ { k }$ and $\sigma _ { k }$ are the reference median and standard deviation, with $\sigma _ { k }$ floored at $1 0 ^ { - 8 }$ . The reference aligns class scales. When the validation comparison gives two nodes the same score, the small arctan term ranks the node with higher surprise first. Larger $s _ { J }$ means greater relational surprise. Appendix A specifies numerical tie handling.

## 4.2 ORDINARY LABELS AND SEQUENTIAL CONTINUATION

From the fixed candidate ranking J, take $C$ as the first L nodes outside $R _ { H }$ , and $D$ as the $L$ nodes inside $R _ { H }$ ranked lowest by J. The remainder $B = R _ { H } \backslash D$ follows Raw order; these disjoint sets are fixed before querying labels. Appendix B.1 defines priorities for different applications and evaluates giving priority to nodes predicted as attacks when false alarms are costly.

Sort $C$ and D separately by increasing predicted class ID, increasing $s _ { R } ( i )$ , then node ID. Match corresponding positions to obtain $( c _ { j } , \bar { d _ { j } } ) , j = 1 , \dots , L$ , with $c _ { j } \in C$ and $\dot { d } _ { j } \in D$ . This fixes pair membership, not review order. Draw one uniform random permutation of the pairs and review it without replacement. Each pair costs two ordinary class labels and gives

$$
\delta _ { j } = u _ { c _ { j } } - u _ { d _ { j } } \in \{ - 1 , 0 , 1 \} .
$$

A +1 means the frozen detector misclassified only $c _ { j } ; - 1$ means only $d _ { j } ; 0$ means both or neither.   
Pairing never uses human labels. Appendix E.5 tests alternative pairings.

After m pairs, 2m distinct nodes have been reviewed. Let $n = ( n _ { + } , n _ { - } , n _ { 0 } )$ count the observed signs, with $n _ { + } + n _ { - } + n _ { 0 } = m$ . The working model assigns unknown probabilities $\theta = ( \theta _ { + } , \theta _ { - } , \theta _ { 0 } )$ to a candidate win, Raw win and tie. A Dirichlet $( 1 / 2 , 1 / \bar { 2 } , 1 / 2 )$ Jeffreys prior gives

$$
\theta \mid n \sim \mathrm { D i r i c h l e t } ( \alpha _ { + } , \alpha _ { - } , \alpha _ { 0 } ) , \qquad \alpha _ { t } = n _ { t } + { \textstyle { \frac { 1 } { 2 } } } , \quad t \in \{ + , - , 0 \} .
$$

![](images/6aa6dd7fd416c96d7ec83a382abd5bf043d349f56f6fd0cb40d198cab0cec0d3.jpg)  
Figure 1: Raw (upper row) and candidate replacements (lower row). Check marks indicate retention; crossed nodes overlap Raw’s first H. The die denotes uniform pair sampling; p/l gives candidate gain +1 and Raw gain 0. Feedback updates $p _ { m }$ (Equation (3)); authorization also requires three non-ties. Both paid endpoints count within H (inset; Equation (4)). Dashed lines connect queue segments. Node counts are illustrative.

For $r = L - m$ unreviewed pairs, let $X _ { + } , X _ { - } , X _ { 0 }$ count their unknown signs. Choosing candidate rather than Raw endpoints adds $X _ { + } - X _ { - }$ corrections, giving predictive probability $p _ { m } \stackrel { \cdot } { = } \operatorname* { P r } ( X _ { + } >$ $X _ { - } \mid n )$ . Set $V = \bar { X } _ { + } + X _ { - }$ , the remaining non-tie count. Writing $\mathrm { B B } ( r , a , b )$ for a beta-binomial count with r trials and beta parameters a, b,

$$
\begin{array} { c } { { V \mid n \sim \mathrm { B B } ( r , \alpha _ { + } + \alpha _ { - } , \alpha _ { 0 } ) , } } \\ { { X _ { + } \mid V = v , n \sim \mathrm { B B } ( v , \alpha _ { + } , \alpha _ { - } ) . } } \end{array}\tag{2}
$$

For $V = v$ , positive gain requires $X _ { + } > v / 2$ . Averaging over possible non-tie counts gives

$$
p _ { m } = \sum _ { v = 0 } ^ { r } \operatorname* { P r } ( V = v \mid n ) \left[ 1 - F _ { \mathrm { B B } ( v , \alpha _ { + } , \alpha _ { - } ) } ( \lfloor v / 2 \rfloor ) \right] ,\tag{3}
$$

where $F$ is the indicated distribution’s cumulative probability. The $v = 0$ term is zero: all ties give no gain.

We use $p _ { m }$ to decide whether to adopt the remaining exchange. Decisions occur after label batches, and each is called a look. Nominal cumulative costs are $5 , 1 0 , \mathrm { \bar { 2 } 0 } , 3 0 , 5 0 \%$ of $H ;$ actual cost is $2 m / H$ (rounding and caps: Appendix B). Authorize $\mathrm { i f } p _ { m } \geq . 9 5$ and $n _ { + } + n _ { - } \ge 3 .$ , requiring three observed non-ties. Reject if $p _ { m } < . 0 5$ . Otherwise continue, rejecting if the last look remains unresolved. This working-model evidence concerns the remaining exchange, not total workflow gain.

Stopping ends pair sampling, not the remaining reviews. Let $I _ { m }$ contain reviewed pair indices and $\overline { { I } } _ { m } = \{ \bar { 1 } , \dots , \bar { L } \} \setminus I _ { m }$ . Write $C _ { I _ { m } } = \{ c _ { j } : j \in I _ { m } \}$ , analogously for D and unreviewed indices. Let $B _ { 1 : h }$ be the first h nodes of B in Raw order (empty for $h \stackrel { - } { = } 0 \mathrm { . }$ ). Keep all paid endpoints and form

$$
S = C _ { I _ { m } } \cup D _ { I _ { m } } \cup \left\{ \begin{array} { l l } { C _ { \overline { { I } } _ { m } } , } & { \mathrm { a u t h o r i z e d , } } \\ { D _ { \overline { { I } } _ { m } } , } & { \mathrm { o t h e r w i s e , } } \end{array} \right. \cup B _ { 1 : H - L - m } .\tag{4}
$$

Thus 2m audited nodes, $L - m$ continuation nodes and $H - L - m$ retained nodes sum to $H .$ Keeping both pair endpoints costs one extra slot per pair, displacing m nodes from the tail of B. Feasible looks require $m < L$ and $m \leq H - L ;$ if $L < 2$ , use $R _ { H }$ without audit queries.

## 5 WHAT THE AUDIT ESTABLISHES

Proposition 1 (Budget and gain identity). Equation (4) gives exactly H unique reviews at each feasible look. Let $A \in \{ 0 , 1 \}$ indicate authorization and $B _ { \mathrm { l o s t } } = B \setminus B _ { 1 : H - L - m }$ . Then

$$
G = U ( C _ { I _ { m } } ) - U ( B _ { \mathrm { l o s t } } ) + A \sum _ { j \notin I _ { m } } \delta _ { j } .
$$

The three disjoint blocks total $H ;$ subtracting $U ( B ) { + } U ( D )$ gives the identity. It separates corrections found during audit, corrections displaced by its cost, and the remaining exchange gain. No node independence is required.

A favorable exchange may not cover audit costs. We distinguish harmful authorization, which yields fewer subsequent corrections than Raw after the same paid queries, from total workflow loss, $\dot { G } < 0$ Appendix C defines the comparison and derives the posterior calculation.

We next consider the sampling distribution for the fixed pairs. For frozen pairs with sign totals $K = ( K _ { + } , K _ { - } , K _ { 0 } )$ , uniform sampling of m pairs gives

$$
\operatorname* { P r } ( n \mid K ) = { \frac { \prod _ { t \in \{ + , - , 0 \} } { \binom { K _ { t } } { n _ { t } } } } { \binom { L } { m } } } , \qquad \sum _ { t } n _ { t } = m .
$$

This law holds even with clustered graph errors (Cochran, 1977). It does not calibrate the Dirichlet working posterior, which supplies model-based evidence rather than a distribution-free guarantee.

## 6 EXPERIMENTS

## 6.1 PROTOCOL AND QUESTIONS

We evaluate correction gains, relational evidence and harmful candidate rejection on four primary security tasks, prevalence and multiclass extensions, and a Physics control. All use $H = \lceil . 1 N \rceil$ and default $\tau = 1 / 3$ This preset can be adjusted for different datasets, but we do not use each dataset’s best tested value for the main results. PairAudit still has room for gains exceeding those presented in the main experiments, as shown in Appendix E.4. A fit is a frozen backbone and source configuration qualified on known training and validation labels. We average 600 audit randomizations per workload, then weight fits equally; these are not independent attack events. Review reveals ordinary dataset labels at equal cost without added noise; subtype flags are evaluator-only. Tasks were selected retrospectively.

Appendices D and F provide data, backbones and seeds. Component count, exchange size and pairing tests appear in Appendices E.3, E.4 and E.5; Figure 3 shows the exchange size benefit–cost tradeoff.

## 6.2 WHICH ERRORS ENTER THE PAID REVIEW BUDGET?

Table 2: Equal-budget corrections and net gains over Raw. $G _ { \mathrm { O O D } }$ counts only unseen-subtype errors.
<table><tr><td>Scenario</td><td>Fits</td><td>Raw</td><td>PairAudit</td><td>G</td><td> $G _ { \mathrm { O O D } }$ </td><td>Auth. (%)</td></tr><tr><td>UNSW / Fuzzers</td><td>20</td><td>224.95</td><td>327.90</td><td>+102.95</td><td>+105.72</td><td>100.000</td></tr><tr><td>CICIDS / Hulk</td><td>10</td><td>1097.60</td><td>1887.39</td><td>+789.79</td><td>+830.80</td><td>100.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>3</td><td>440.33</td><td>780.63</td><td>+340.30</td><td>+461.00</td><td>100.000</td></tr><tr><td>TON /XSS</td><td>3</td><td>383.33</td><td>463.35</td><td>+80.01</td><td>+102.02</td><td>100.000</td></tr></table>

All paid reviews are included. Auth. is the percentage of runs authorizing the candidate.

All four primary scenarios show positive mean G and $G _ { \mathrm { O O D } }$ after all pair costs, and both fit-level means are positive in all 36 configurations. UNSW adds 102.95 net corrections to Raw’s 224.95, while CICIDS adds 789.79 to 1,097.60. The effect is meaningful because these are net errors Raw would not recover within the same capacity. In X-IIoTID / RDOS, $G _ { \mathrm { O O D } } = 4 6 1 . 0 0$ exceeds $G = 3 4 0 . 3 0 \colon$

Table 3: Target population and review budget used before deciding.
<table><tr><td>Scenario</td><td>N</td><td>H</td><td>H/N (%) Labels</td><td>2m/H (%)</td><td>On auth. (%)</td></tr><tr><td>UNSW / Fuzzers</td><td>25,305</td><td>2,531</td><td>10.00</td><td>144.4 5.71</td><td>5.71</td></tr><tr><td>CICIDS / Hulk</td><td>27,114</td><td>2,712</td><td>10.00</td><td>136.0 5.01</td><td>5.01</td></tr><tr><td>X-IIoTID / RDOS</td><td>14,187</td><td>1,419</td><td>10.00</td><td>72.0 5.07</td><td>5.07</td></tr><tr><td>TON / XSS</td><td>11,407</td><td>1,141</td><td>10.00</td><td>78.9 6.92</td><td>6.92</td></tr></table>

Labels and $2 m / H$ average all runs; the last column averages authorized runs only. Final review cost is always H.

![](images/c2ca0969d44bcc670e7ef5eb3ae716ff711ca458b7165fcedcb8e1e52bcbca22.jpg)  
Figure 2: (a) New authorizations at each nominal look, as percentages of all runs per task; the inset magnifies rare late decisions. (b) Early rejection on Coauthor Physics, with $\bar { H _ { \mathrm { ~ } } } = 1 { , } 2 8 2$ reviews. Adopting the same candidate exchange $\bar { B } \cup \bar { C }$ without audit corrects 863.33 errors; PairAudit corrects 1,097.37, including audit costs. Raw corrects 1,120.33. Trust Score is a separate ranking reference.

120.70 known-object corrections are lost, but the unseen-type benefit more than compensates. This is a tradeoff, not a count of additional unknown objects.

Decisions generally leave most of the budget available for continuation (Table 3). CICIDS and X-IIoTID / RDOS authorize at the first feasible look in all runs; TON requires 6.92% of H on average. All 21,600 primary runs authorize. Authorization is a fraction of complete audit runs, not a fraction of labels judged beneficial. Figure 2 shows newly occurring decisions by look. The primary tasks show neither harmful authorized continuation nor negative total gain in the evaluated runs.

On Physics, PairAudit rejects its candidate at the first look in all 1,800 runs after 66 labels (5.15% of H). Compared with adopting the same exchange without audit, it preserves 234.03 corrections at the same budget (Figure 2b). Its mean loss relative to Raw is 22.97, versus 204.33 for Trust Score; the latter is a separate scorer comparison.

## 6.3 COMPARISON WITH INDEPENDENT REVIEW RANKINGS

Table 4 compares complete policies, not isolated scores. Raw retention and paid audits are parts of PAIRAUDIT’s policy. Trust Score is strongest in CICIDS and TON on total corrections, so PAIRAUDIT is not uniformly superior. On CICIDS, however, PAIRAUDIT corrects 13.60 more unseen-attack errors while giving up 96.81 total corrections. On X-IIoTID / RDOS, Trust Score loses 41.00 total corrections and adds only 0.67 OOD corrections, whereas PAIRAUDIT improves both. These differences show why overall and unseen-attack corrections should be assessed together. Appendix E.1 gives public settings and input limitations.

Table 4: Independent review strategies at equal budget. Entries are $G / G _ { \mathrm { O O D } }$
<table><tr><td>Method</td><td>UNSW / Fuzzers</td><td>CICIDS / Hulk</td><td>X-IIoTID / RDOS</td><td>TON / XSS</td></tr><tr><td>Raw</td><td>0.00 / 0.00</td><td>0.00 / 0.00</td><td>0.00 / 0.00</td><td>0.00 / 0.00</td></tr><tr><td>Trust Score</td><td>-23.90 / -29.90</td><td>+886.60 /+817.20</td><td>-41.00 / +0.67</td><td> $+ 1 6 6 . 6 7 / + 1 3 5 . 6 7$ </td></tr><tr><td>Energy</td><td>-37.15 / -36.05</td><td>-270.40 / -263.80</td><td>-12.67 / +0.00</td><td> $+ 2 6 . 3 3 / + 2 9 . 3 3$ </td></tr><tr><td>GNNSafe</td><td>-61.40 / -50.60</td><td>-546.60 / -549.70</td><td>-98.67 / +0.00</td><td>-13.33 / +26.67</td></tr><tr><td>GRASP</td><td>-11.20 /+3.90</td><td>+224.10/+425.70</td><td>-155.67 / +0.00</td><td>-138.33 / -87.00</td></tr><tr><td>PairAudit</td><td> $\mathbf { + 1 0 2 . 9 5 } / \mathbf { + 1 0 5 . 7 2 }$ </td><td> $\mathbf { + 7 8 9 . 7 9 } \ / + 8 3 0 . 8 0$ </td><td> $\mathbf { + 3 4 0 . 3 0 } / \mathbf { + 4 6 1 . 0 0 }$ </td><td> $\mathbf { + 8 0 . 0 1 } / + \mathbf { 1 0 2 . 0 2 }$ </td></tr></table>

External methods use their own first H nodes, without our audit. All methods in the CICIDS column share ten checkpoint reruns; Appendix E.1 gives details.

## 6.4 WHAT DOES RELATIONAL CONDITIONING CONTRIBUTE?

Appendix E.2 reports the full controls in Table 9. Feature-conditioned weights improve mean gain over fixed weights on all four primary tasks. TON provides the clearest evidence for relational information: the graph token model yields $\dot { G } = + 8 0 . 0 1$ , versus −39.86 using only node coordinates and +65.36 after matched neighborhood shuffling. The benefit is task dependent: removing neighborhood coordinates improves X-IIoTID / RDOS, and matched shuffling does not reduce gain on CICIDS.

## 6.5 PREVALENCE STRESS TESTS AND MULTICLASS REVIEW

Table 5: Stress tests and multiclass review, relative to the equal-budget Raw baseline for each pool.
<table><tr><td>Task / pool</td><td>N</td><td>H</td><td>OOD (%)</td><td> $\rho _ { \mathrm { e r r } } \left( \% \right)$ </td><td>G</td><td> $G _ { \mathrm { O O D } }$ </td><td>Auth. (%)</td><td>Cost (%)</td></tr><tr><td>IoT23 / ARP (1%)</td><td>13,693</td><td>1,370</td><td>1.00</td><td></td><td>0.514+58.59</td><td>+20.03</td><td>99.96</td><td>9.83</td></tr><tr><td>IoMT / ARP (full)</td><td>6,554</td><td>656</td><td>20.90</td><td>12.084+44.19</td><td></td><td>+44.51</td><td>100.00</td><td>8.61</td></tr><tr><td>IoMT / ARP (95%)</td><td>1,442</td><td>145</td><td>95.01</td><td> $5 4 . 9 2 4 ~ + 4 4 . 3 3 ~$ </td><td></td><td>+44.33</td><td>100.00</td><td>5.53</td></tr><tr><td>BoT / DDoS-TCP</td><td>7,482</td><td>749</td><td>66.65</td><td></td><td></td><td>65.143 +37.81 +101.60</td><td>99.50</td><td>13.05</td></tr></table>

$\rho _ { \mathrm { { e r r } } } \mathbf { \dot { \mathbf { \cdot } } }$ OOD errors divided by $N ;$ cost: $2 m / H .$ . ARP tasks are binary; BoT uses three classes without priority. Pool construction and configurations are in Appendices D and F.

CIC IoT 2023 / MITM-ArpSpoofing yields 58.59 net corrections, including 20.03 on unseen attacks, with only 1% OOD objects: benefit need not require high OOD prevalence. IoMT / ARP also improves both its full and 95% OOD pools, mainly through unseen-attack corrections (Table 5). This comparison is descriptive because datasets and graph contexts differ. In multiclass BoT-IoT, PairAudit increases corrections from 407.00 to 444.81 with 99.50% authorization while retaining the DDoS/DoS/reconnaissance review task. Its 101.60 additional OOD corrections outweigh 63.79 fewer known-subtype corrections. Appendices D and F give construction and run-level losses.

## 7 LIMITATIONS AND CONCLUSION

PAIRAUDIT turns graph-based evidence into actionable review decisions under distribution shift. It finds potentially overlooked errors and uses ordinary class labels to decide whether a candidate queue deserves the remaining review budget, without retraining the detector or changing the review task. Across the four primary security tasks, it delivers positive mean overall and unseen-attack correction gains with high authorization rates and decisions that use only a small share of the budget. The rejection control further shows how early feedback can limit losses from an unfavorable proposal. Together, these results support combining Raw retention, relational evidence and paid feedback within a single review policy.

The benefit of graph tokens varies across tasks, and our current formulation assumes equal review costs and gives each corrected error the same value. The evaluation also does not explicitly model reviewer mistakes. Extending the framework to different review times, error costs, and uncertain labels would require changes to the review objective and audit rule. These limitations motivate further work on cost sensitive review, noisy human feedback, and stronger statistical guarantees.

## AI USE STATEMENT

AI tools assisted with manuscript preparation. Reported quantities are computed from saved predictions and review ledgers. AI outputs are not experimental observations or sources of authority.

## REPRODUCIBILITY STATEMENT

The supplement specifies the Gaussian and gate estimation, audit computation, data roles and all configurations. Code implements the frozen scoring and paid review procedures; evidence files record frozen inputs, audit seeds, decisions, review-set checksums and aggregate metrics. Appendix F describes verification and source provenance. No paid human-subject study is claimed.

## REFERENCES

Muna Al-Hawawreh, Elena Sitnikova, and Neda Aboutorab. X-IIoTID: A connectivity-agnostic and device-agnostic intrusion data set for industrial internet of things. IEEE Internet of Things Journal, 9(5):3962–3977, 2022. doi: 10.1109/JIOT.2021.3102056.

Haoyue Bai, Xuefeng Du, Katie Rainey, Shibin Parameswaran, and Yixuan Li. Out-of-distribution learning with human feedback. Transactions on Machine Learning Research, 2025.

Christopher M. Bishop. Mixture density networks. Technical Report NCRG/94/004, Aston University, 1994.

Canadian Institute for Cybersecurity. CIC IoT Dataset 2023. University of New Brunswick, 2023. URL https://www.unb.ca/cic/datasets/iotdataset-2023.html.

Canadian Institute for Cybersecurity. CIC IoMT Dataset 2024. University of New Brunswick, 2024. URL https://www.unb.ca/cic/datasets/iomt-dataset-2024.html.

William G. Cochran. Sampling Techniques. John Wiley & Sons, 3 edition, 1977.

Juan Jose Egozcue, Vera Pawlowsky-Glahn, Gl´ oria Mateu-Figueras, and Carles Barcel\` o-Vidal.´ Isometric logratio transformations for compositional data analysis. Mathematical Geology, 35(3): 279–300, 2003. doi: 10.1023/A:1023818214614.

Federico Errica, Davide Bacciu, and Alessio Micheli. Graph mixture density networks. In International Conference on Machine Learning, pp. 3025–3035, 2021.

Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In International Conference on Learning Representations, 2017.

Heinrich Jiang, Been Kim, Melody Guan, and Maya Gupta. To trust or not to trust a classifier. In Advances in Neural Information Processing Systems, 2018.

Nickolaos Koroniotis, Nour Moustafa, Elena Sitnikova, and Benjamin Turnbull. Towards the development of realistic botnet dataset in the internet of things for network forensic analytics: Bot-iot dataset. Future Generation Computer Systems, 100:779–796, 2019.

Jannik Kossen, Sebastian Farquhar, Yarin Gal, and Tom Rainforth. Active testing: Sample-efficient model evaluation. In International Conference on Machine Learning, pp. 5753–5763, 2021.

Romain Laroche, Paul Trichelair, and Remi Tachet Des Combes. Safe policy improvement with baseline bootstrapping. In International Conference on Machine Learning, pp. 3652–3661, 2019.

Weitang Liu, Xiaoyun Wang, John D. Owens, and Yixuan Li. Energy-based out-of-distribution detection. In Advances in Neural Information Processing Systems, 2020.

Longfei Ma, Yiyou Sun, Kaize Ding, Zemin Liu, and Fei Wu. Revisiting score propagation in graph out-of-distribution detection. In Advances in Neural Information Processing Systems, 2024.

Nour Moustafa. A new distributed architecture for evaluating AI-based security systems at the edge: Network TON IoT datasets. Sustainable Cities and Society, 72:102994, 2021. doi: 10.1016/j.scs. 2021.102994.

Nour Moustafa and Jill Slay. UNSW-NB15: A comprehensive data set for network intrusion detection systems. In Military Communications and Information Systems Conference, pp. 1–6, 2015. doi: 10.1109/MilCIS.2015.7348942.

Amit Roy, Juan Shu, Jia Li, Carl Yang, Olivier Elshocht, Jeroen Smeets, and Pan Li. GAD-NR: Graph anomaly detection via neighborhood reconstruction. In ACM International Conference on Web Search and Data Mining, pp. 576–585, 2024. doi: 10.1145/3616855.3635767.

Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani. Toward generating a new intrusion detection dataset and intrusion traffic characterization. In International Conference on Information Systems Security and Privacy, pp. 108–116, 2018. doi: 10.5220/0006639801080116.

Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, and Stephan Gunnemann. Pitfalls¨ of graph neural network evaluation. arXiv preprint arXiv:1811.05868, 2018.

Xiuyao Song, Mingxi Wu, Christopher Jermaine, and Sanjay Ranka. Conditional anomaly detection. IEEE Transactions on Knowledge and Data Engineering, 19(5):631–645, 2007. doi: 10.1109/ TKDE.2007.1009.

Keith Stouffer, Michael Pease, CheeYee Tang, Timothy Zimmerman, Victoria Pillitteri, Suzanne Lightman, Adam Hahn, Stephanie Saravia, Aslam Sherule, and Michael Thompson. Guide to operational technology (OT) security. Technical Report SP 800-82 Rev. 3, National Institute of Standards and Technology, 2023.

Harit Vishwakarma, Heguang Lin, and Ramya Korlakai Vinayak. Taming false positives in out-ofdistribution detection with human feedback. In International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings ofMachine Learning Research, pp. 1486–1494, 2024.

Washington University in St. Louis. WUSTL-IIoT-2021 dataset for IIoT cybersecurity research, 2021. URL https://www.cs.wustl.edu/<sub>˜</sub>jain/iiot2/index.html.

Qitian Wu, Yiting Chen, Chenxiao Yang, and Junchi Yan. Energy-based out-of-distribution detection for graph neural networks. In International Conference on Learning Representations, 2023.

## A GAUSSIAN MIXTURES AND FEATURE-CONDITIONED WEIGHTS

## A.1 COORDINATE CONVENTIONS

Clip probabilities below $1 0 ^ { - 8 }$ and renormalize. $\mathbf { A } \left( K - 1 \right) \times K$ Helmert matrix A with $A A ^ { \top } = I$ and $A \mathbf { 1 } = 0$ implements $\operatorname { i l r } ( p ) = A \log p$

## A.2 EM ESTIMATION AND COMPONENT SELECTION

The GMM implementation uses scikit-learn GaussianMixture: full covariance, two initializations, at most 300 iterations, tolerance $1 0 ^ { - 3 }$ and seed 9126101.

Cap fitting at 8,000 training nodes per class, subsampling uniformly with seed $9 1 2 6 1 0 0 + k$ . Declare the candidate counts before target evaluation. A count M requires M max(10, 2d) fit samples, EM convergence, finite validation likelihood and expected membership of at least two per component. Validation ties within $1 0 ^ { - 1 2 }$ favor smaller $M ;$ no admissible count means fitting fails.

## A.3 CONDITIONAL-LIKELIHOOD GATE ESTIMATION

Initialize an ExtraTreesRegressor from training responsibilities: 128 trees, minimum leaf size 32, feature fraction .7, seed 9127201. Freeze its partitions. For leaf ℓ among $T = 1 2 8$ trees, set $A _ { i \ell } =$ $1 / T$ if node i reaches it, else zero. With leaf probabilities $\vartheta _ { \ell }$ summing to one, $\begin{array} { r } { g _ { k b } ( x _ { i } ) = \sum _ { \ell } A _ { i \ell } \vartheta _ { \ell b } } \end{array}$

Write $\phi _ { i b } = \phi _ { d } ( z _ { i } ; \mu _ { k b } , \Sigma _ { k b } )$ ) and $\begin{array} { r } { F _ { i } = \sum _ { b } ( g _ { k b } ( x _ { i } ) + \lambda \pi _ { k b } ) \phi _ { i b } / ( 1 + \lambda ) } \end{array}$ . The E step assigns trainable responsibilities

$$
\xi _ { i \ell b } = \frac { A _ { i \ell } \vartheta _ { \ell b } \phi _ { i b } } { ( 1 + \lambda ) F _ { i } } , \qquad \vartheta _ { \ell b } ^ { \mathrm { n e w } } = \frac { \sum _ { i } \xi _ { i \ell b } } { \sum _ { i , c } \xi _ { i \ell c } } .
$$

The fixed $\lambda \pi _ { k }$ term receives no trainable responsibility; zero-membership leaves keep their weights. In exact arithmetic, these EM updates do not decrease Equation (1), which is concave in the leaf weights for fixed partitions and Gaussian components.

Initialize leaf means from $\gamma _ { i } ,$ , floor at $1 0 ^ { - 1 0 }$ and renormalize. Select among updates $\{ 0 , 1 , 2 , 4 , 8 , 1 6 , 3 2 , 6 4 \}$ and the main-text $\eta _ { k }$ grid by known-validation likelihood. Update zero is initialization; ties within $1 0 ^ { - 1 2 }$ favor the earlier update, then smaller $\eta _ { k }$

## A.4 NUMERICAL SCORING

Use log-sum-exp, fixed per-row Cholesky arithmetic, eight-decimal log-density rounding and singlethreaded gate prediction to keep scores stable across batch sizes.

## B EXECUTION, LABEL ACCESS AND NUMERICAL AUDIT

## B.1 CANDIDATE RANKING

Let $\mathrm { r a n k } _ { s _ { J } } ( i )$ order nodes by decreasing $s J ,$ , starting at 1, with node ID breaking ties. For a priority class $c _ { \mathrm { p r i o r i t y } }$ chosen before review, define

$$
q _ { i } = 2 { \bf 1 } \{ \hat { y } _ { i } = c _ { \mathrm { p r i o r i t y } } \} + 1 - \frac { \mathrm { r a n k } _ { s _ { J } } ( i ) - 1 } { N + 1 } .
$$

The rank term lies in (0, 1], so adding 2 puts that predicted class first while preserving its score order. Sorting $q _ { i }$ from highest to lowest gives $J ;$ without a priority class, omit the indicator. Construct $C , D , { \bar { B } }$ as in Section 4.2. This changes candidate and replacement selection, not the uncertaintybased Raw reference. Each corrected error still earns one unit.

The UNSW, CICIDS, X-IIoTID and TON experiments represent settings where missed attacks are especially costly. We therefore set $c _ { \mathrm { p r i o r i t y } } = \mathrm { b e n i g n }$ to seek attacks hidden among benign predictions. In industrial applications, falsely flagging normal activity can instead prompt costly blocking or isolation; automatic responses must account for their effect on normal operation (Stouffer et al., 2023). When avoiding these disruptions is the review priority, set $c _ { \mathrm { p r i o r i t y } } = \mathrm { a t t a c k }$ , with the

same density model, budget and audit rule. These cost preferences describe intended applications;   
the datasets do not measure an economic cost ratio.

Table 6 evaluates this choice on X-IIoTID / CoAP and WUSTL-IIoT-2021 / UDP. The held-out normal service or protocol is absent from training and validation but retains its benign label, so OOD corrections here recover false alarms on unfamiliar normal activity. Three fits per task each average 600 audit runs; Appendix D.3 gives construction and selection details.

Table 6: Predicted-attack priority on unseen normal activity, relative to equal-budget Raw.
<table><tr><td>Task</td><td>H</td><td>OOD (%)</td><td>G</td><td> $G _ { \mathrm { O O D } }$ </td><td>Auth. (%)</td><td>Cost (%)</td></tr><tr><td>X-IIoTID / CoAP</td><td>1,419</td><td>35.24</td><td>+282.47</td><td>+447.33</td><td>100.00</td><td>5.07</td></tr><tr><td>WUSTL-IIoT-2021 / UDP</td><td>710</td><td>70.51</td><td>+118.64</td><td>+124.03</td><td>98.39</td><td>9.12</td></tr></table>

G: net change in all corrected errors; $G _ { \mathrm { O O D } } \mathrm { { : } }$ the same change restricted to held-out normal activity. Both compare final paid reviews with Raw’s first H nodes. Cost is $2 m / H ;$ all decision labels count toward $H .$

Both gains are positive in every fit mean, CoAP notably has no negative gain run or harmful authorization in 1,800 runs.

Below is the comparison with benign priority Raw. This baseline places predicted-benign nodes first, orders each predicted class by Raw uncertainty, and directly reviews the first H nodes without an audit. Table 7 compares it with PAIRAUDIT on the same frozen fits and budgets. PAIRAUDIT improves both mean gains on these two tasks; the comparison evaluates the complete policies, not the graph score alone.

Table 7: Benign-priority comparison. Entries are $G / G _ { \mathrm { O O D } }$ , relative to the original Raw ranking.
<table><tr><td>Task</td><td>Benign priority + Raw</td><td>PAIRAUDIT</td></tr><tr><td>X-IIoTID / RDOS</td><td> $- 1 0 2 . 6 7 / 0 . 0 0$ </td><td> $+ 3 4 0 . 3 0 / + 4 6 1 . 0 0$ </td></tr><tr><td>UNSW / Fuzzers</td><td> $+ 9 1 . 4 5 / + 1 0 4 . 8 5$ </td><td> $+ 1 0 2 . 9 5 / + 1 0 5 . 7 2$ </td></tr></table>

Means over three X-IIoTID and 20 UNSW fits. PairAudit averages 600 audit runs per fit; all paid labels count toward H.

## B.2 OBSERVATION SCHEDULE AND IMPLEMENTATION

Follow Section 4.2. The pair permutation uses the recorded audit seed plus 1,000,000. For nominal cost fraction $f \in \{ . 0 5 , . 1 , . 2 , . 3 , . 5 \}$ , use

$$
m _ { f } = \operatorname* { m i n } ( L - 1 , H - L , \operatorname* { m a x } ( 1 , \lceil f H / 2 \rceil ) ) .
$$

Deduplicate and sort these counts. If $L < 2 ,$ review $R _ { H }$ without pair queries. Query only previously unseen endpoints and reuse their labels.

For $H = 1 , 0 0 0 , \tau = 1 / 3$ and $L = 3 3 3$ , a first look at 25 pairs uses 50 labels. Completion uses 50 audited nodes, 308 nodes from the chosen side and the first 642 nodes of B, totaling H.

## C AUDIT CALCULATIONS AND DIAGNOSTICS

## C.1 HARM DIAGNOSTICS AND PAIRING

For the harmful-authorization diagnostic, let $O = C _ { I _ { m } } \cup D _ { I _ { m } }$ contain all paid pair endpoints, and let $R ^ { O }$ contain the first $H - | O$ unreviewed nodes in Raw order. An authorization is harmful when $U ( S \backslash O ) < U ( R ^ { O } )$ . Total workflow loss instead means $G < 0$ , including investigation costs. Literal Raw continuation $\mathring { R } ^ { O }$ need not coincide with the local fallback in Equation (4); these comparison targets are therefore reported separately.

Every bijective pairing preserves $\begin{array} { r } { \sum _ { i } \delta _ { j } = U ( C ) - U ( D ) } \end{array}$ , but can change ties, sampling variance and stopping time. A common permutation of fixed pairs changes only review order.

## C.2 EXACT POSTERIOR EVALUATION

Under the working model, $X \quad \mid \quad \theta , n \quad \sim$ Multinomia $\displaystyle \lvert ( r , \theta )$ Integrating $\theta \quad \mid \quad n \quad \sim$ Dirichlet $( \alpha _ { + } , \alpha _ { - } , \alpha _ { 0 } )$ gives, with $\begin{array} { r } { \alpha . = \sum _ { t } \alpha _ { t } , } \end{array}$

$$
\operatorname* { P r } ( X = x \mid n ) = { \frac { r ! \Gamma ( \alpha . ) } { \Gamma ( \alpha . + r ) } } \prod _ { t \in \{ + , - , 0 \} } { \frac { \Gamma ( \alpha _ { t } + x _ { t } ) } { x _ { t } ! \Gamma ( \alpha _ { t } ) } } , \qquad \sum _ { t } x _ { t } = r .
$$

Dirichlet aggregation makes $\theta _ { + } + \theta _ { - } \sim \mathrm { B e t a } ( \alpha _ { + } + \alpha _ { - } , \alpha _ { 0 } )$ independent of $\rho = \theta _ { + } / ( \theta _ { + } + \theta _ { - } )$ ∼ $\mathrm { B e t a } ( \alpha _ { + } , \alpha _ { - } )$ , conditional on n. Integrating the binomial counts yields Equation (2); averaging over V yields Equation (3). A cached Polya-urn recursion evaluates the same probability deterministically and is checked against the beta-binomial sum.

## D DATASETS, GRAPHS AND BACKBONE QUALIFICATION

Table 8: Known-source qualification. Ranges span the fitted configurations, not independent events.
<table><tr><td>Task</td><td>Fits</td><td>Train acc.</td><td>Train F1</td><td>Val. acc.</td><td>Val. F1</td></tr><tr><td>UNSW / Fuzzers</td><td>20</td><td>0.996–0.997</td><td>0.993-0.994</td><td>0.995-0.997</td><td>0.992–0.995</td></tr><tr><td>CICIDS / Hulk</td><td>10</td><td>0.979–0.981</td><td>0.962–0.965</td><td>0.979–0.981</td><td>0.962-0.966</td></tr><tr><td>X-IIoTID / RDOS</td><td>3</td><td>0.952–0.953</td><td>0.951–0.952</td><td>0.939–0.945</td><td>0.937–0.944</td></tr><tr><td>TON / XSS</td><td>3</td><td>0.985-0.986</td><td>0.982–0.984</td><td>0.986-0.986</td><td>0.982–0.984</td></tr><tr><td>IoT23 / ARP</td><td>3</td><td>0.959–0.960</td><td>0.866–0.868</td><td>0.958–0.959</td><td>0.860–0.865</td></tr><tr><td>IoMT / ARP</td><td>3</td><td>0.968–0.974</td><td>0.951–0.961</td><td>0.964–0.971</td><td>0.945–0.956</td></tr><tr><td>BoT / DDoS-TCP</td><td>3</td><td>0.932–0.950</td><td>0.929–0.946</td><td>0.926–0.951</td><td>0.923-0.948</td></tr></table>

F1 is macro averaged.

All reported detectors satisfy known training and validation macro-F1 $\ge . 8 .$ Table 8 covers the main-text tasks; Appendix D.3 reports the industrial extensions. Binary tasks use attack/benign labels: main-text tasks prioritize predicted-benign nodes, while Table 6 prioritizes predicted-attack nodes. Multiclass tasks and Physics have no priority.

## D.1 PRIMARY SECURITY TASKS

UNSW-NB15 and CICIDS2017. Fuzzers (UNSW-NB15) and DoS Hulk (CICIDS2017) are held out from training and validation (Moustafa & Slay, 2015; Sharafaldin et al., 2018). Target pools combine known test nodes and the held-out subtype. We use 20 GraphSAGE and 10 APPNP fits, respectively. Relations use available endpoint, protocol, service/session or temporal fields, never labels. UNSW fits the detector on source-induced edges, then computes posteriors on the unlabelled full graph; source tokens may contain target covariate context.

X-IIoTID / RDOS. Hold out RDOS from X-IIoTID training and validation, retaining attack/benign review labels (Al-Hawawreh et al., 2022). Use the author’s release and the 55 numeric predictors described in Appendix D.3, excluding labels, addresses, ports, timestamps, service and auxiliary detector alerts. Remove conflicting binary-label duplicates and deduplicate predictor vectors. Retain up to 5,000 rows per group at preprocessing, then 5,000 for benign and RDOS groups and 2,000 for each other group. Split known groups 60/20/20 within subtype; all 5,000 RDOS nodes enter the target only. Each fit has 27,541 training, 9,181 validation and 14,187 target nodes.

Fit imputation, signed-log transformation and standardization on training data, then clip to $[ - 1 0 , 1 0 ]$ Build separate symmetrized ten-nearest-neighbor source and target feature graphs. Three GraphSAGE fits (seeds 9269301–9269303) use hidden size 64, dropout .3, Adam learning rate .01, weight decay .0005 and inverse-square-root class weights. Known-validation macro-F1 selects checkpoints within 300 epochs, with patience 45 after epoch 100. Audit seeds are 9270000–9270599.

TON-IoT. XSS is source-excluded; three GraphSAGE fits retain attack/benign labels (Moustafa, 2021). Separate source and target graphs use source/destination addresses, directed address pairs and protocol/port groups. Features include duration, byte and packet counts, missed bytes, protocol, service and connection state. Subtypes never enter scoring or audit.

## D.2 STRESS POOLS AND MULTICLASS REVIEW

IoMT / ARP. Hold out ARP Spoofing in CIC IoMT 2024 (Canadian Institute for Cybersecurity, 2024). Labels are capture-derived.

Remove labels/identifiers and conflicting predictor duplicates. Cap preprocessing at 5,000 nodes per subtype, then retain up to 5,000 benign/held-out nodes and 2,000 per other subtype. Known subtypes use seeded 60/20/20 training/validation/test splits without predictor overlap; held-out nodes are target-only. Fit medians, signed-log transformation and standardization on training data, clip features to [−10, 10], and build source and target feature-neighbor graphs separately.

Known validation selects among GCN, GraphSAGE, APPNP and ResidualSAGE, subject to qualification. ResidualSAGE wins all three fits. It joins 128-dimensional two-layer feature and GraphSAGE pathways, with dropout .1, Adam learning rate .003, weight decay .0005 and inverse-square-root class weights. Validation early stopping selects within 600 epochs; other backbones use learning rate .01 and at most 300 epochs.

CIC IoT 2023 / MITM-ArpSpoofing. Hold out MITM-ArpSpoofing in CIC IoT 2023 (Canadian Institute for Cybersecurity, 2023). Deduplication, caps, 60/20/20 known splits and transforms follow IoMT. Three GraphSAGE fits (seeds 9137101–9137103) use hidden size 64, dropout .3, learning rate .01, weight decay .0005 and inverse-square-root class weights. Each has 40,668 training and 13,556 validation nodes. Known-validation macro-F1 selects checkpoints within 300 epochs, with patience 45 after epoch 100.

Each low-OOD cohort contains all 13,556 known test nodes and 137 uniformly sampled held-out attacks. Seeds 9136000–9136002 give three cohorts per fit. Rebuild target edges using only cohort nodes before frozen inference; omitted attacks provide no context. Source and target graphs remain separate.

Stress construction and denominators. For IoMT high-OOD pools, set $\begin{array} { r l } { N ^ { \prime } } & { { } = } \end{array}$ min $( N , \lfloor N _ { \mathrm { O O D } } / a \rfloor , \lfloor N _ { \mathrm { k n o w n } } / ( 1 - a ) \rfloor )$ with $a ~ = ~ . 9 5$ , sample round(aN<sup>′</sup>) OOD nodes and permute the pool. This retains 1,370 ARP nodes and samples 72 known nodes; three seeds give three cohorts per fit. Unlike IoT23, graph context and scores remain from full-target inference. Each pool uses $\bar { H _ { \mathrm { ~ } } } \bar { = } \left\lceil . 1 N ^ { \prime } \right\rceil$ . In Table $\begin{array} { r } { { 5 } , \bar { \rho } _ { \mathrm { e r r } } = N ^ { \prime - 1 } \sum _ { i } u _ { i } v _ { i } } \end{array}$ counts OOD errors per target node; prevalence also includes correctly predicted OOD nodes.

BoT-IoT multiclass review. Use the public 5% BoT-IoT subset and native category/subcategory labels (Koroniotis et al., 2019). Review assigns DDoS, DoS or reconnaissance. Normal and Theft are outside this task because of insufficient representation. Hold out DDoS/TCP, retaining DDoS/UDP and DDoS/HTTP as known. Use 18 measurements of protocol/flags/state, counts, durations and rates; exclude IPs, ports, absolute timestamps, sequence IDs, labels and capture aggregates. Deduplication, conflict removal, caps and training-only transforms follow IoMT. Known 60/20/20 splits yield 7,483 training and 2,495 validation nodes. The target has 7,482 nodes, including 4,987 DDoS/TCP nodes; H = 749. Source and target use separate symmetrized ten-nearest-neighbor graphs.

Three GCN fits (seeds 9139101–9139103) use hidden size 64, dropout .3, Adam learning rate .01, weight decay .0005 and inverse-square-root class weights. Known-validation macro-F1 selects checkpoints within 300 epochs, with patience 45 after epoch 100. Validation class recall exceeds .87. All classification errors have unit utility; no class-priority offset is applied.

The three fits share one capture collection and held-out subtype.

## D.3 INDUSTRIAL TASKS WITH PREDICTED-ATTACK PRIORITY

X-IIoTID contains industrial control and service activity (Al-Hawawreh et al., 2022); WUSTL-IIoT-2021 comes from an industrial testbed (Washington University in St. Louis, 2021). Use the author’s

X-IIoTID Kaggle release and the public WUSTL mirror<sup>1</sup>. Hold out normal CoAP rows identified by Service in X-IIoTID and normal UDP rows (Target=0, Proto=17) in WUSTL. This is a constructed business-type shift, not an official temporal split.

Remove labels, addresses, ports and timestamps; also remove X-IIoTID’s service field and auxiliary detector alerts, and WUSTL’s IP IDs and epoch-like IdleTime. Retain 55 and 38 numeric features, respectively. Drop conflicting binary-label duplicates and deduplicate predictor vectors. Deterministic caps retain at most 5,000 rows per group, then 2,000 per known group and 5,000 for the held-out group; discard groups below 100 rows. Known groups use seeded 60/20/20 splits, while the held-out group enters only the target. Imputation, signed-log transformation, standardization and clipping follow IoMT; source and target ten-nearest-neighbor feature graphs are separate.

Fix GraphSAGE (hidden size 64, dropout .3), Adam learning rate .01, weight decay .0005 and inverse-square-root class weights. Known-validation macro-F1 selects checkpoints within 300 epochs, with patience 45 after epoch 100. Confirmation seeds 9268201–9268203 yield mean known training/validation macro-F1 of .947/.939 for CoAP and .996/.997 for UDP. Their target sizes are 14,187 and 7,091, each with 5,000 held-out normal nodes. Use $H = \lceil . 1 N \rceil , \tau = 1 / 3$ , unchanged .95/.05 thresholds, and audit seeds 9269000–9269599.

## D.4 REJECTION CONTROL AND SOURCE LIMITATIONS

Coauthor Physics. Use native coauthorship edges, word features and field labels (Shchur et al., 2018). Field 4 is absent from output classes 0–3. Known classes use 50% training, 20% validation and the remainder for testing; all 3,519 field-4 nodes enter the 12,813-node target. Three knownvalidation-selected GraphSAGE fits use no priority. Every field-4 label is automatically a detector error.

## E ADDITIONAL EXPERIMENTS

The following controls evaluate fixed settings; they do not select deployment parameters from target gains.

## E.1 REVIEW POLICY IMPLEMENTATIONS

Each baseline reviews the first H nodes in its own ranking. PAIRAUDIT uses its complete policy, including candidate priorities, Raw retention and paid audits. Combining another ranking with our audit would define a hybrid policy.

Trust Score uses the public default without filtering (Jiang et al., 2018). Its anomaly score is $- d _ { \mathrm { o t h e r } } / ( d _ { \mathrm { p r e d } } + 1 0 ^ { - \bar { 1 } 2 } )$ , using the second nearest known-training neighbor per class in detector input space. Exact-distance acceleration is checked against the original KDTree.

Energy is − log $\begin{array} { r } { \sum _ { k } \exp ( \ell _ { i k } ) } \end{array}$ at temperature one. GNNSafe propagates the logit-derived energy over two steps with coefficient .5. GRASP uses its public MSP configuration: eight steps, $\alpha = 0 , \tau _ { 1 } = 5$ $\tau _ { 2 } = 5 0$ and $\delta = 1 . 0 0 1$ , receiving unlabelled target indices rather than OOD membership.

CICIDS originally lacked saved weights and logits. Recover all ten APPNP fits with the same splits, seeds and 50-epoch recipe, then use one shared probability/logit bank for all methods, including a known-only refitted PairAudit density. Target class predictions match the original banks; scores and individual runs need not be identical. Energy uses actual logits. Other tasks retain their saved checkpoints.

Recovered checkpoint predictions preserve Raw’s first-H set; the maximum Raw-score error is $2 . 3 9 \times 1 0 ^ { - 7 }$ . An original-KDTree check reproduces the Trust Score comparison.

Table 9: Complete component and relational control metrics. All settings retain the same budget and ordinary correction utility.
<table><tr><td>Scenario</td><td>Condition</td><td>G</td><td> $G _ { \mathrm { O O D } }$ </td><td>Auth. (%)</td><td> $2 m / H$  (%)</td><td>Harm (%)</td></tr><tr><td>UNSW / Fuzzers</td><td>PairAudit</td><td>+102.95</td><td>+105.72</td><td>100.000</td><td>5.71</td><td>0.000</td></tr><tr><td>UNSW / Fuzzers</td><td>Fixed weights</td><td>+95.35</td><td>+98.28</td><td>100.000</td><td>5.75</td><td>0.000</td></tr><tr><td>UNSW / Fuzzers</td><td>Node marginal</td><td>+97.95</td><td>+100.35</td><td>100.000</td><td>5.51</td><td>0.000</td></tr><tr><td>UNSW / Fuzzers</td><td>Matched shuffle</td><td>+95.32</td><td>+97.81</td><td>100.000</td><td>5.72</td><td>0.000</td></tr><tr><td>UNSW / Fuzzers</td><td>Global shuffle</td><td>+57.98</td><td>+60.60</td><td>98.833</td><td>7.88</td><td>0.017</td></tr><tr><td>CICIDS / Hulk</td><td>PairAudit</td><td>+789.79</td><td>+830.80</td><td>100.000</td><td>5.01</td><td>0.000</td></tr><tr><td>CICIDS / Hulk</td><td>Fixed weights</td><td>+788.53</td><td>+831.60</td><td>100.000</td><td>5.01</td><td>0.000</td></tr><tr><td>CICIDS / Hulk</td><td>Node marginal</td><td>+778.19</td><td>+818.20</td><td>100.000</td><td>5.01</td><td>0.000</td></tr><tr><td>CICIDS / Hulk</td><td>Matched shuffle</td><td>+790.77</td><td>+829.60</td><td>100.000</td><td>5.01</td><td>0.000</td></tr><tr><td>CICIDS / Hulk</td><td>Global shuffle</td><td>+697.10</td><td>+720.40</td><td>100.000</td><td>5.01</td><td>0.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>PairAudit</td><td>+340.30</td><td>+461.00</td><td>100.000</td><td>5.07</td><td>0.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>Fixed weights</td><td>+338.62</td><td>+457.33</td><td>100.000</td><td>5.07</td><td>0.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>Node marginal</td><td>+354.38</td><td>+466.67</td><td>100.000</td><td>5.07</td><td>0.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>Matched shuffle</td><td>+217.96</td><td>+341.33</td><td>100.000</td><td>5.21</td><td>0.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>Global shuffle</td><td>+333.01</td><td>+470.67</td><td>100.000</td><td>5.07</td><td>0.000</td></tr><tr><td>TON / XSS</td><td>PairAudit</td><td>+80.01</td><td>+102.02</td><td>100.000</td><td>6.92</td><td>0.000</td></tr><tr><td>TON / XSS</td><td>Fixed weights</td><td>+78.16</td><td>+97.66</td><td>100.000</td><td>7.02</td><td>0.000</td></tr><tr><td>TON / XSS</td><td>Node marginal</td><td>-39.86</td><td>-16.69</td><td>51.556</td><td>30.89</td><td>20.333</td></tr><tr><td>TON / XSS</td><td>Matched shuffle</td><td>+65.36</td><td>+93.78</td><td>100.000</td><td>8.00</td><td>0.000</td></tr><tr><td>TON / XSS</td><td>Global shuffle</td><td>+46.17</td><td>+62.71</td><td>99.778</td><td>9.48</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.667</td></tr></table>

## E.2 COMPONENT AND RELATIONAL CONTROLS

Controls use the first 300 audit seeds per fit, matched to PairAudit’s 600-run evaluation. Fixed weights sets $\eta _ { k } = 0$ with the same Gaussian components and rebuilt reference ranks. Node marginal scores only node coordinates of the fitted density while retaining the gate. It removes explicit neighbor coordinates, but the frozen GNN and training responsibilities still contain graph information.

Matched shuffle permutes neighbor-token halves within data role, predicted class, confidence quartile and feature cluster. Quartiles use known validation; 32 MiniBatchKMeans clusters use known training features. Refit the density and gate after shuffling. Global shuffle permutes within data role only. Each uses one token permutation and 300 audit runs per fit, not a permutation test. TON’s node-marginal control has 20.33% harmful authorized continuations versus 0.78% harmful remaining pair sums; the distinction is defined in Appendix C.

## E.3 COMPONENT-COUNT SENSITIVITY

Table 10 fixes $M _ { k } = M$ and selects gate settings on known validation labels. Each count uses the same 48 fits, 63 workloads and 600 audit seeds per workload. These are fixed-count comparisons, not searches up to M.

Sensitivity varies by task. At $M = 1$ , net gain is negative on TON and full-pool IoMT; BoT is negative for $M = 1 , 2 , 3 .$ . Larger counts can help, but are not uniformly better. Deployment counts must be selected from validation density fit, not these test gains.

Table 10: Component-count sensitivity: net corrections G at the same total review budget.
<table><tr><td>Scenario</td><td>Reported</td><td> $M = 1$ </td><td>2</td><td>3</td><td>4</td><td>8</td><td>16</td></tr><tr><td>UNSW / Fuzzers</td><td>102.95</td><td>103.99</td><td>72.32</td><td>81.27</td><td>102.95</td><td>103.83</td><td>103.62</td></tr><tr><td>CICIDS / Hulk</td><td>789.79</td><td>720.16</td><td>792.47</td><td>792.94</td><td>789.60</td><td>800.41</td><td>789.80</td></tr><tr><td>X-IIoTID / RDOS</td><td>340.30</td><td>334.34</td><td>335.71</td><td>337.40</td><td>340.30</td><td>344.53</td><td>347.21</td></tr><tr><td>TON /XSS</td><td>80.01</td><td>-22.07</td><td>65.20</td><td>73.25</td><td>80.01</td><td>72.63</td><td>67.73</td></tr><tr><td>IoMT / ARP, full</td><td>44.19</td><td>-6.47</td><td>30.03</td><td>46.11</td><td>44.38</td><td>39.16</td><td>48.35</td></tr><tr><td>IoMT / ARP, 95% OOD</td><td>44.33</td><td>41.37</td><td>42.65</td><td>43.66</td><td>44.22</td><td>42.87</td><td>43.32</td></tr><tr><td>IoT23 / ARP, 1% OOD</td><td>58.59</td><td>19.78</td><td>51.58</td><td>62.58</td><td>58.59</td><td>60.72</td><td>58.51</td></tr><tr><td>BoT / DDoS-TCP</td><td>37.81</td><td>-18.59</td><td>-33.47</td><td>-35.44</td><td>37.81</td><td>41.97</td><td>92.42</td></tr><tr><td>Coauthor Physics</td><td>-22.97</td><td>-20.28</td><td>-21.64</td><td>-23.47</td><td>-22.97</td><td>-19.75</td><td>-22.00</td></tr></table>

Reported uses known-validation density selection; other columns fix $M _ { k } = M$ for every class.

## E.4 EXCHANGE-FRACTION SENSITIVITY

$$
{ \begin{array} { l r l r l } { - \emptyset - } & { { \mathrm { U N S W } } / { \mathsf { F u z z e r s } } } & { - \emptyset - } & { { \mathrm { C l C l D S } } / { \mathsf { H u l k } } } & { - { \mathrm { a - } } } & { { \mathsf { X - l l o T 1 D } } / { \mathsf { R D O S } } } & { - \emptyset - } & { { \mathrm { T O N } } / { \mathsf { X S S } } } \end{array} }
$$

![](images/894439e4bcbc53ac6472ac59ad8db4a49142855fac6f2ab8aa943953ec7a3471.jpg)

![](images/ab94eb0d0feae288f4a3b4c704d520218b1fe6555dc77177a9bb1de5b2b62ee2.jpg)  
Figure 3: Exchange-fraction sensitivity at fixed H. (a) Net gain relative to mean Raw corrections. (b) Decision labels as a fraction of H. Points average the same primary fits and 300 audit runs per fit; the dotted line marks $\tau = 1 / 3$ . Table 11 includes exact gains, authorization and harm rates.

Test $\tau \in \{ . 1 , . 2 , 1 / 3 , . 4 , . 5 \}$ on 36 primary and three Physics fits, with 300 audit runs each (58,500 total). Hold scores and H fixed.

Larger exchanges improve CICIDS, X-IIoTID / RDOS and TON mean gains, while increasing TON’s decision cost. The default $1 / 3$ remains positive on all primary tasks but is not universally optimal. At $\tau = . 5 ,$ one of 900 TON runs has negative total gain despite no harmful continuation. Physics always rejects and remains negative relative to Raw.

Table 11: Sensitivity to the exchange fraction τ. Each setting has 300 audit runs per fit.
<table><tr><td>Scenario</td><td>Setting</td><td>G</td><td> $G _ { \mathrm { O O D } }$ </td><td>Auth. (%)</td><td>Cost (%)</td><td>Harm (%)</td></tr><tr><td>UNSW / Fuzzers</td><td>0.1</td><td>+88.52</td><td>+88.80</td><td>100.000</td><td>5.06</td><td>0.000</td></tr><tr><td>UNSW / Fuzzers</td><td>0.2</td><td>+99.78</td><td>+101.16</td><td>100.000</td><td>5.15</td><td>0.000</td></tr><tr><td>UNSW / Fuzzers</td><td>1/3</td><td>+102.95</td><td>+105.74</td><td>100.000</td><td>5.65</td><td>0.000</td></tr><tr><td>UNSW / Fuzzers</td><td>0.4</td><td>+102.30</td><td>+105.66</td><td>100.000</td><td>6.04</td><td>0.000</td></tr><tr><td>UNSW / Fuzzers</td><td>0.5</td><td>+99.16</td><td>+104.02</td><td>100.000</td><td>6.97</td><td>0.000</td></tr><tr><td>CICIDS / Hulk</td><td>0.1</td><td>+213.15</td><td>+223.90</td><td>100.000</td><td>5.01</td><td>0.000</td></tr><tr><td>CICIDS / Hulk</td><td>0.2</td><td>+457.60</td><td>+482.80</td><td>100.000</td><td>5.01</td><td>0.000</td></tr><tr><td>CICIDS / Hulk</td><td>1/3</td><td>+789.81</td><td>+830.80</td><td>100.000</td><td>5.01</td><td>0.000</td></tr><tr><td>CICIDS / Hulk</td><td>0.4</td><td>+956.85</td><td>+1000.30</td><td>100.000</td><td>5.01</td><td>0.000</td></tr><tr><td>CICIDS / Hulk</td><td>0.5</td><td>+1202.68</td><td>+1248.10</td><td>100.000</td><td>5.01</td><td>0.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>0.1</td><td>+101.41</td><td>+129.00</td><td>100.000</td><td>5.07</td><td>0.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>0.2</td><td>+207.19</td><td>+271.00</td><td>100.000</td><td>5.07</td><td>0.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>1/3</td><td>+340.34</td><td>+461.00</td><td>100.000</td><td>5.07</td><td>0.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>0.4</td><td>+401.48</td><td>+555.00</td><td>100.000</td><td>5.07</td><td>0.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>0.5</td><td>+499.59</td><td>+696.33</td><td>100.000</td><td>5.07</td><td>0.000</td></tr><tr><td>TON / XSS</td><td>0.1</td><td>+61.89</td><td>+70.67</td><td>100.000</td><td>5.08</td><td>0.000</td></tr><tr><td>TON / XSS</td><td>0.2</td><td>+64.63</td><td>+83.83</td><td>100.000</td><td>5.87</td><td>0.000</td></tr><tr><td>TON / XSS</td><td>1/3</td><td>+80.18</td><td>+102.24</td><td>100.000</td><td>6.83</td><td>0.000</td></tr><tr><td>TON / XSS</td><td>0.4</td><td>+88.96</td><td>+114.25</td><td>100.000</td><td>7.33</td><td>0.000</td></tr><tr><td>TON / XSS</td><td>0.5</td><td>+90.08</td><td>+118.79</td><td>100.000</td><td>8.27</td><td>0.000</td></tr><tr><td>Coauthor Physics</td><td>0.1</td><td>-23.35</td><td>-23.27</td><td>0.000</td><td>5.15</td><td>0.000</td></tr><tr><td>Coauthor Physics</td><td>0.2</td><td>-23.11</td><td>-24.12</td><td>0.000</td><td>5.16</td><td>0.000</td></tr><tr><td>Coauthor Physics</td><td>1/3</td><td>-22.94</td><td>-23.17</td><td>0.000</td><td>5.15</td><td>0.000</td></tr><tr><td>Coauthor Physics</td><td>0.4</td><td>-22.46</td><td>-21.87</td><td>0.000</td><td>5.15</td><td>0.000</td></tr><tr><td>Coauthor Physics</td><td>0.5</td><td>-19.63</td><td>-18.50</td><td>0.000</td><td>5.15</td><td>0.000</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## E.5 PAIR-FORMATION ROBUSTNESS

Table 12: Sensitivity to pair formation. Each setting has 300 audit runs per fit.
<table><tr><td>Scenario</td><td>Setting</td><td>G</td><td> $G _ { \mathrm { O O D } }$ </td><td>Auth. (%)</td><td>Cost (%)</td><td>Harm (%)</td></tr><tr><td>UNSW / Fuzzers</td><td>Ordered</td><td>+102.95</td><td>+105.74</td><td>100.000</td><td>5.65</td><td>0.000</td></tr><tr><td>UNSW / Fuzzers</td><td>Shuffle C</td><td>+102.96</td><td>+105.74</td><td>100.000</td><td>5.64</td><td>0.000</td></tr><tr><td>UNSW / Fuzzers</td><td>Shuffle D</td><td>+102.97</td><td>+105.74</td><td>100.000</td><td>5.62</td><td>0.000</td></tr><tr><td>UNSW / Fuzzers</td><td>Shuffle both</td><td>+102.95</td><td>+105.72</td><td>100.000</td><td>5.69</td><td>0.000</td></tr><tr><td>CICIDS / Hulk</td><td>Ordered</td><td>+789.81</td><td>+830.80</td><td>100.000</td><td>5.01</td><td>0.000</td></tr><tr><td>CICIDS / Hulk</td><td>Shuffle C</td><td>+789.81</td><td>+830.80</td><td>100.000</td><td>5.01</td><td>0.000</td></tr><tr><td>CICIDS / Hulk</td><td>Shuffle D</td><td>+789.79</td><td>+830.80</td><td>100.000</td><td>5.01</td><td>0.000</td></tr><tr><td>CICIDS / Hulk</td><td>Shuffle both</td><td>+789.79</td><td>+830.80</td><td>100.000</td><td>5.01</td><td>0.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>Ordered</td><td>+340.34</td><td>+461.00</td><td>100.000</td><td>5.07</td><td>0.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>Shuffle C</td><td>+340.34</td><td>+461.00</td><td>100.000</td><td>5.07</td><td>0.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>Shuffle D</td><td>+340.29</td><td>+461.00</td><td>100.000</td><td>5.07</td><td>0.000</td></tr><tr><td>X-IIoTID / RDOS</td><td>Shuffle both</td><td>+340.29</td><td>+461.00</td><td>100.000</td><td>5.07</td><td>0.000</td></tr><tr><td>TON / XSS</td><td>Ordered</td><td>+80.18</td><td>+102.24</td><td>100.000</td><td>6.83</td><td>0.000</td></tr><tr><td>TON / XSS</td><td>Shuffle C</td><td>+79.96</td><td>+102.04</td><td>100.000</td><td>7.01</td><td>0.000</td></tr><tr><td>TON / XSS</td><td>Shuffle D</td><td>+79.67</td><td>+101.70</td><td>100.000</td><td>7.10</td><td>0.000</td></tr><tr><td>TON / XSS</td><td>Shuffle both</td><td>+79.75</td><td>+101.77</td><td>100.000</td><td>7.03</td><td>0.000</td></tr><tr><td>Coauthor Physics</td><td>Ordered</td><td>-22.94</td><td>-23.17</td><td>0.000</td><td>5.15</td><td></td></tr><tr><td>Coauthor Physics</td><td>Shuffle C</td><td>-22.91</td><td>-23.16</td><td>0.000</td><td>5.15</td><td>0.000 0.000</td></tr><tr><td>Coauthor Physics</td><td>Shuffle D</td><td>-22.94</td><td>-23.17</td><td>0.000</td><td>5.15</td><td>0.000</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Coauthor Physics</td><td>Shuffle both</td><td>-22.91</td><td>-23.16</td><td>0.000</td><td>5.15</td><td>0.000</td></tr></table>

Compare the fixed pairing, shuffled C, shuffled D, and independently shuffled sides on 39 fits, with 300 audit runs per setting (46,800 total). Keep sets, scores and H fixed. Side permutations depend only on configuration and run IDs, before outcomes are accessed.

Primary-task mean gains change by at most .52 corrections. All matched primary runs authorize with no harmful continuation or negative total gain; shuffling slightly raises TON’s decision cost. Physics always rejects at the first look.

## F WORKLOAD RESULTS AND REPRODUCIBILITY

## F.1 FITS AND AUDIT RUNS

Table 13 lists 63 workloads from 48 frozen fits: 36 primary, three Physics, three BoT, three IoMT full, nine IoMT high-OOD and nine IoT23 low-OOD workloads. Each row averages 600 audit runs; cohort means weight nine workloads equally. IoT23 L1–L3 denote source seeds 9137101–9137103; BoT M1–M3 denote 9139101–9139103. Audit seeds randomize pair order, not events. Saved ledgers verify exact H, retention of paid nodes and net/OOD gains.

## F.2 RUN-LEVEL LOSSES

The 21,600 primary runs have no negative gain or harmful authorization. IoT23 has one harmful authorization and eight negative-gain runs out of 5,400. Full-pool IoMT has 42/1,800 negative-gain runs and BoT has 21/1,800, with no harmful authorization in either. High-OOD IoMT has neither outcome in 5,400 runs. Physics rejects all 1,800 times but loses 22.97 corrections on average through investigation costs.

Table 13: Main-text PairAudit configurations and workloads. Each row contains 600 audit runs; fit IDs identify frozen configurations in the supplement.
<table><tr><td rowspan=1 colspan=13>Task               Fit ID   Mix/cohort     N    H       G   $G _ { \mathrm { O O D } }$  Auth. (%) Cost (%)</td></tr><tr><td rowspan=1 colspan=1>UNSW / F</td><td></td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=10>21       Full        25,3052,531+104.96+104.97    100.00     5.11</td></tr><tr><td rowspan=1 colspan=1>UNSW / F</td><td></td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=6>22       Full        25,3052,531+166.39</td><td rowspan=1 colspan=3>+171.00    100.00</td><td rowspan=1 colspan=1>5.07</td></tr><tr><td rowspan=1 colspan=1>UNSW / F</td><td></td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=6>23       Full        25,3052,531+153.07</td><td rowspan=1 colspan=3>+153.00    100.00</td><td rowspan=1 colspan=1>5.06</td></tr><tr><td rowspan=1 colspan=1>UNSW / F</td><td></td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=4>24       Full        25,305</td><td rowspan=1 colspan=2>2,531  +44.11</td><td rowspan=1 colspan=3>+49.32    100.00</td><td rowspan=1 colspan=1>9.27</td></tr><tr><td rowspan=1 colspan=1>UNSW / F</td><td></td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=4>25       Full        25,305</td><td rowspan=1 colspan=2>2,531 +96.33</td><td rowspan=1 colspan=3>+98.95    100.00</td><td rowspan=1 colspan=1>5.32</td></tr><tr><td rowspan=1 colspan=1>UNSW / F</td><td></td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=4>26       Full        25,305</td><td rowspan=1 colspan=2>2,531  +96.35</td><td rowspan=1 colspan=3>+101.84    100.00</td><td rowspan=1 colspan=1>5.48</td></tr><tr><td rowspan=1 colspan=2>UNSW / F</td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=4>27       Full        25,305</td><td rowspan=1 colspan=2>2,531  +61.83</td><td rowspan=1 colspan=3>+69.37    100.00</td><td rowspan=1 colspan=1>7.49</td></tr><tr><td rowspan=1 colspan=2>UNSW / F</td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=3>28       Full</td><td rowspan=1 colspan=1>25,305</td><td rowspan=1 colspan=2>2,531+111.07</td><td rowspan=1 colspan=1>+110.99</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.07</td></tr><tr><td rowspan=1 colspan=2>UNSW / F</td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=3>29       Full</td><td rowspan=1 colspan=1>25,305</td><td rowspan=1 colspan=2>2,531+105.33</td><td rowspan=1 colspan=1>+109.95</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.32</td></tr><tr><td rowspan=1 colspan=2>UNSW / F</td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=3>30       Full</td><td rowspan=1 colspan=1>25,305</td><td rowspan=1 colspan=2>2,531  +79.99</td><td rowspan=1 colspan=1>+80.91</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.28</td></tr><tr><td rowspan=1 colspan=2>UNSW / F</td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=3>31       Full</td><td rowspan=1 colspan=1>25,305</td><td rowspan=1 colspan=2>2,531  +61.66</td><td rowspan=1 colspan=1>+62.58</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.78</td></tr><tr><td rowspan=1 colspan=2>UNSW / F</td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=3>32       Full</td><td rowspan=1 colspan=1>25,305</td><td rowspan=1 colspan=2>2,531 +62.25</td><td rowspan=1 colspan=1>+64.96</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>6.35</td></tr><tr><td rowspan=1 colspan=2>UNSW / F</td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=3>33       Full</td><td rowspan=1 colspan=1>25,305</td><td rowspan=1 colspan=2>2,531+210.23</td><td rowspan=1 colspan=1>+213.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.06</td></tr><tr><td rowspan=1 colspan=2>UNSW / F</td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=3>34       Full</td><td rowspan=1 colspan=1>25,305</td><td rowspan=1 colspan=2>2,531  +63.86</td><td rowspan=1 colspan=1>+65.68</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.99</td></tr><tr><td rowspan=1 colspan=2>UNSW / F</td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=3>35       Full</td><td rowspan=1 colspan=1>25,305</td><td rowspan=1 colspan=2>2,531+109.99</td><td rowspan=1 colspan=1>+107.99</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.07</td></tr><tr><td rowspan=1 colspan=2>UNSW / F</td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=3>36       Full</td><td rowspan=1 colspan=1>25,305</td><td rowspan=1 colspan=1>2,531</td><td rowspan=1 colspan=1>+66.85</td><td rowspan=1 colspan=1>+71.48</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>6.40</td></tr><tr><td rowspan=1 colspan=2>UNSW / F</td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=3>37       Full</td><td rowspan=1 colspan=1>25,305</td><td rowspan=1 colspan=1>2,531</td><td rowspan=1 colspan=1>+160.24</td><td rowspan=1 colspan=1>+162.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.06</td></tr><tr><td rowspan=1 colspan=2>UNSW / F</td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=3>38       Full</td><td rowspan=1 colspan=1>25,305</td><td rowspan=1 colspan=1>2,531</td><td rowspan=1 colspan=1>+108.06</td><td rowspan=1 colspan=1>+111.75</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.37</td></tr><tr><td rowspan=1 colspan=2>UNSW / F</td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=3>39       Full</td><td rowspan=1 colspan=1>25,305</td><td rowspan=1 colspan=1>2,531</td><td rowspan=1 colspan=1>+97.26</td><td rowspan=1 colspan=1>+102.80</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.39</td></tr><tr><td rowspan=1 colspan=2>UNSW / F</td><td rowspan=1 colspan=1>uzzers</td><td rowspan=1 colspan=3>40       Full</td><td rowspan=1 colspan=1>25,305</td><td rowspan=1 colspan=1>2,531</td><td rowspan=1 colspan=1>+99.10</td><td rowspan=1 colspan=1>+101.97</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.20</td></tr><tr><td rowspan=1 colspan=2>CICIDS</td><td rowspan=1 colspan=1>/ Hulk</td><td rowspan=1 colspan=3>51       Full</td><td rowspan=1 colspan=1>27,114</td><td rowspan=1 colspan=1>2,712</td><td rowspan=1 colspan=1>+797.03</td><td rowspan=1 colspan=1>+836.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.01</td></tr><tr><td rowspan=1 colspan=2>CICIDS</td><td rowspan=1 colspan=1>/ Hulk</td><td rowspan=1 colspan=3>52       Full</td><td rowspan=1 colspan=1>27,114</td><td rowspan=1 colspan=1>2,712</td><td rowspan=1 colspan=1>+766.96</td><td rowspan=1 colspan=1>+803.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.01</td></tr><tr><td rowspan=1 colspan=2>CICIDS</td><td rowspan=1 colspan=1>/ Hulk</td><td rowspan=1 colspan=3>53       Full</td><td rowspan=1 colspan=1>27,114</td><td rowspan=1 colspan=1>2,712</td><td rowspan=1 colspan=1>+793.51</td><td rowspan=1 colspan=1>+841.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.01</td></tr><tr><td rowspan=1 colspan=2>CICIDS</td><td rowspan=1 colspan=1>/ Hulk</td><td rowspan=1 colspan=3>54       Full</td><td rowspan=1 colspan=1>27,114</td><td rowspan=1 colspan=1>2,712</td><td rowspan=1 colspan=1>+792.09</td><td rowspan=1 colspan=1>+840.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.01</td></tr><tr><td rowspan=1 colspan=2>CICIDS</td><td rowspan=1 colspan=1>/ Hulk</td><td rowspan=1 colspan=3>55       Full</td><td rowspan=1 colspan=1>27,114</td><td rowspan=1 colspan=1>2,712</td><td rowspan=1 colspan=1>+813.44</td><td rowspan=1 colspan=1>+853.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.01</td></tr><tr><td rowspan=1 colspan=2>CICIDS</td><td rowspan=1 colspan=1>/ Hulk</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=2>Full</td><td rowspan=1 colspan=1>27,114</td><td rowspan=1 colspan=1>2,712</td><td rowspan=1 colspan=1>+751.11</td><td rowspan=1 colspan=1>+788.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.01</td></tr><tr><td rowspan=1 colspan=2>CICIDS</td><td rowspan=1 colspan=1>/ Hulk</td><td rowspan=1 colspan=1>57</td><td rowspan=1 colspan=2>Full</td><td rowspan=1 colspan=1>27,114</td><td rowspan=1 colspan=1>2,712</td><td rowspan=1 colspan=1>+765.54</td><td rowspan=1 colspan=1>+819.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.01</td></tr><tr><td rowspan=1 colspan=2>CICIDS</td><td rowspan=1 colspan=1>/ Hulk</td><td rowspan=1 colspan=1>58</td><td rowspan=1 colspan=2>Full</td><td rowspan=1 colspan=1>27,114</td><td rowspan=1 colspan=1>2,712</td><td rowspan=1 colspan=1>+799.15</td><td rowspan=1 colspan=1>+827.00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>5.01</td></tr><tr><td rowspan=1 colspan=3>CICIDS / Hulk</td><td rowspan=1 colspan=1>59</td><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>27,114</td><td rowspan=1 colspan=1>2,712</td><td rowspan=1 colspan=1>+800.79</td><td rowspan=1 colspan=1>+852.00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>5.01</td></tr><tr><td rowspan=1 colspan=3>CICIDS / Hulk</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=2>Full</td><td rowspan=1 colspan=1>27,114</td><td rowspan=1 colspan=1>2,712</td><td rowspan=1 colspan=1>+818.29</td><td rowspan=1 colspan=1>+849.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.01</td></tr><tr><td rowspan=1 colspan=3>X-IIoTID / RDOS</td><td rowspan=1 colspan=1>9269301</td><td rowspan=1 colspan=2>Full</td><td rowspan=1 colspan=1>14,187</td><td rowspan=1 colspan=1>1,419</td><td rowspan=1 colspan=1>+364.03</td><td rowspan=1 colspan=1>+454.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.07</td></tr><tr><td rowspan=1 colspan=3>X-IIoTID / RDOS</td><td rowspan=1 colspan=1>9269302</td><td rowspan=1 colspan=2>Full</td><td rowspan=1 colspan=1>14,187</td><td rowspan=1 colspan=1>1,419</td><td rowspan=1 colspan=1>+317.96</td><td rowspan=1 colspan=1>+461.00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>5.07</td></tr><tr><td rowspan=1 colspan=3>X-IIoTID / RDOS</td><td rowspan=1 colspan=1>9269303</td><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>14,187</td><td rowspan=1 colspan=1>1,419</td><td rowspan=1 colspan=1>+338.91</td><td rowspan=1 colspan=1>+468.00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>5.07</td></tr><tr><td rowspan=1 colspan=3>TON /XSS</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>11,407</td><td rowspan=1 colspan=1>1,141</td><td rowspan=1 colspan=1>+91.12</td><td rowspan=1 colspan=1>+109.59</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>6.19</td></tr><tr><td rowspan=1 colspan=3>TON /XSS</td><td rowspan=1 colspan=1>71</td><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>11,407</td><td rowspan=1 colspan=1>1,141</td><td rowspan=1 colspan=1>+72.64</td><td rowspan=1 colspan=1>+96.65</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>7.86</td></tr><tr><td rowspan=1 colspan=3>TON /XSS</td><td rowspan=1 colspan=1>74</td><td rowspan=1 colspan=2>Full</td><td rowspan=1 colspan=1>11,407</td><td rowspan=1 colspan=1>1,141</td><td rowspan=1 colspan=1>+76.28</td><td rowspan=1 colspan=1>+99.84</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>6.70</td></tr><tr><td rowspan=1 colspan=3>IoT23 / ARP</td><td rowspan=1 colspan=1>L1</td><td rowspan=1 colspan=2>1%/0</td><td rowspan=1 colspan=1>13,693</td><td rowspan=1 colspan=1>1,370</td><td rowspan=1 colspan=1>+60.69</td><td rowspan=1 colspan=1>+21.97</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>9.59</td></tr><tr><td rowspan=1 colspan=3>IoT23 / ARP</td><td rowspan=1 colspan=1>L1</td><td rowspan=1 colspan=2>1%/1</td><td rowspan=1 colspan=1>13,693</td><td rowspan=1 colspan=1>1,370</td><td rowspan=1 colspan=1>+41.90</td><td rowspan=1 colspan=1>+14.92</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>99.83</td><td rowspan=1 colspan=1>11.62</td></tr><tr><td rowspan=1 colspan=3>IoT23 / ARP</td><td rowspan=1 colspan=1>L1</td><td rowspan=1 colspan=2>1%/2</td><td rowspan=1 colspan=1>13,693</td><td rowspan=1 colspan=1>1,370</td><td rowspan=1 colspan=1>+51.65</td><td rowspan=1 colspan=1>+12.29</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>10.21</td></tr><tr><td rowspan=1 colspan=3>IoT23 / ARP</td><td rowspan=1 colspan=1>L2</td><td rowspan=1 colspan=2>1%/0</td><td rowspan=1 colspan=1>13,693</td><td rowspan=1 colspan=1>1,370</td><td rowspan=1 colspan=1>+61.71</td><td rowspan=1 colspan=1>+17.68</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>8.87</td></tr><tr><td rowspan=1 colspan=3>IoT23 / ARP</td><td rowspan=1 colspan=1>L2</td><td rowspan=1 colspan=2>1%/1</td><td rowspan=1 colspan=1>13,693</td><td rowspan=1 colspan=1>1,370</td><td rowspan=1 colspan=1>+69.18</td><td rowspan=1 colspan=1>+26.90</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>99.83</td><td rowspan=1 colspan=1>8.99</td></tr><tr><td rowspan=1 colspan=3>IoT23 / ARP</td><td rowspan=1 colspan=1>L2</td><td rowspan=1 colspan=2>1%/2</td><td rowspan=1 colspan=1>13,693</td><td rowspan=1 colspan=1>1,370</td><td rowspan=1 colspan=1>+71.12</td><td rowspan=1 colspan=1>+23.23</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>9.26</td></tr><tr><td rowspan=1 colspan=3>IoT23 / ARP</td><td rowspan=1 colspan=1>L3</td><td rowspan=1 colspan=2>1%/0</td><td rowspan=1 colspan=1>13,693</td><td rowspan=1 colspan=1>1,370</td><td rowspan=1 colspan=1>+55.93</td><td rowspan=1 colspan=1>+16.45</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>10.13</td></tr><tr><td rowspan=1 colspan=3>IoT23 / ARP</td><td rowspan=1 colspan=1>L3</td><td rowspan=1 colspan=3>1%/1       13,693</td><td rowspan=1 colspan=1>1,370</td><td rowspan=1 colspan=1>+61.23</td><td rowspan=1 colspan=1>+24.99</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>9.55</td></tr><tr><td rowspan=1 colspan=3>IoT23 / ARP</td><td rowspan=1 colspan=1>L3</td><td rowspan=1 colspan=3>1%/2       13,693</td><td rowspan=1 colspan=1>1,370</td><td rowspan=1 colspan=1>+53.87</td><td rowspan=1 colspan=1>+21.86</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>10.27</td></tr><tr><td rowspan=1 colspan=1>IoMT /</td><td rowspan=1 colspan=2>ARP</td><td rowspan=1 colspan=1>9124101</td><td rowspan=1 colspan=3>Full         6,554</td><td rowspan=1 colspan=1>656</td><td rowspan=1 colspan=1>+46.47</td><td rowspan=1 colspan=1>+42.62</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>7.45</td></tr><tr><td rowspan=1 colspan=1>IoMT /</td><td rowspan=1 colspan=2>ARP</td><td rowspan=1 colspan=1>9124101</td><td rowspan=1 colspan=3>95%/0       1,442</td><td rowspan=1 colspan=1>145</td><td rowspan=1 colspan=1>+45.00</td><td rowspan=1 colspan=1>+45.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.52</td></tr><tr><td rowspan=1 colspan=1>IoMT /</td><td rowspan=1 colspan=2>ARP</td><td rowspan=1 colspan=1>9124101</td><td rowspan=1 colspan=3>95%/1       1,442</td><td rowspan=1 colspan=1>145</td><td rowspan=1 colspan=1>+44.00</td><td rowspan=1 colspan=1>+44.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.52</td></tr><tr><td rowspan=1 colspan=1>IoMT /</td><td rowspan=1 colspan=2>ARP</td><td rowspan=1 colspan=1>9124101</td><td rowspan=1 colspan=3>95%/2       1,442</td><td rowspan=1 colspan=1>145</td><td rowspan=1 colspan=1>+41.97</td><td rowspan=1 colspan=1>+41.97</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.55</td></tr><tr><td rowspan=1 colspan=1>IoMT /</td><td rowspan=1 colspan=2>ARP</td><td rowspan=1 colspan=1>9124102</td><td rowspan=1 colspan=3>Full         6,554</td><td rowspan=1 colspan=1>656</td><td rowspan=1 colspan=1>+65.72</td><td rowspan=1 colspan=1>+64.32</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>6.44</td></tr><tr><td rowspan=1 colspan=1>IoMT /</td><td rowspan=1 colspan=2>ARP</td><td rowspan=1 colspan=1>9124102</td><td rowspan=1 colspan=3>95%/0       1,442</td><td rowspan=1 colspan=1>145</td><td rowspan=1 colspan=1>+44.99</td><td rowspan=1 colspan=1>+44.99</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.54</td></tr><tr><td rowspan=1 colspan=1>IoMT /</td><td rowspan=1 colspan=2>ARP</td><td rowspan=1 colspan=1>9124102</td><td rowspan=1 colspan=3>95%/1       1,442</td><td rowspan=1 colspan=1>145</td><td rowspan=1 colspan=1>+46.00</td><td rowspan=1 colspan=1>+46.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.52</td></tr><tr><td rowspan=1 colspan=1>IoMT /</td><td rowspan=1 colspan=2>ARP</td><td rowspan=1 colspan=1>9124102</td><td rowspan=1 colspan=4>95%/2       1,442  145</td><td rowspan=1 colspan=1>+46.00</td><td rowspan=1 colspan=1>+46.00</td><td rowspan=1 colspan=2>100.00</td><td rowspan=1 colspan=1>5.52</td></tr><tr><td rowspan=1 colspan=1>IoMT /</td><td rowspan=1 colspan=3>ARP       9124103</td><td rowspan=1 colspan=4>Full         6,554  656</td><td rowspan=1 colspan=1>+20.39</td><td rowspan=1 colspan=3>+26.60    100.00</td><td rowspan=1 colspan=1>11.95</td></tr><tr><td rowspan=1 colspan=1>IoMT /</td><td rowspan=1 colspan=3>ARP      9124103</td><td rowspan=1 colspan=4>95%/0       1,442  145</td><td rowspan=1 colspan=1>+42.99</td><td rowspan=1 colspan=3>+42.99    100.00</td><td rowspan=1 colspan=1>5.58</td></tr></table>

Continued on next page

Table 13 continued
<table><tr><td>Task</td><td>Fit ID</td><td>Mix/cohort</td><td>N</td><td>H</td><td>G</td><td>GOOD</td><td>Auth. (%)</td><td>Cost (%)</td></tr><tr><td>IoMT / ARP</td><td>9124103</td><td>95%/1</td><td>1,442</td><td>145</td><td>+44.00</td><td>+44.00</td><td>100.00</td><td>5.52</td></tr><tr><td>IoMT / ARP</td><td>9124103</td><td>95%/2</td><td>1,442</td><td>145</td><td>+44.00</td><td>+44.00</td><td>100.00</td><td>5.53</td></tr><tr><td>BoT / DDoS-TCP</td><td>M1</td><td>Full</td><td>7,482</td><td>749</td><td>+29.00</td><td>+98.06</td><td>98.50</td><td>15.46</td></tr><tr><td>BoT / DDoS-TCP</td><td>M2</td><td>Full</td><td>7,482</td><td>749</td><td>+42.82</td><td>+103.42</td><td>100.00</td><td>11.66</td></tr><tr><td>BoT / DDoS-TCP</td><td>M3</td><td>Full</td><td>7,482</td><td>749</td><td>+41.62</td><td>+103.31</td><td>100.00</td><td>12.04</td></tr><tr><td>Coauthor Physics</td><td>9094101</td><td>Full</td><td>12,813</td><td>1,282</td><td>-22.10</td><td>-23.00</td><td>0.00</td><td>5.15</td></tr><tr><td>Coauthor Physics</td><td>9094102</td><td>Full</td><td>12,813</td><td>1,282</td><td>-24.36</td><td>-24.50</td><td>0.00</td><td>5.15</td></tr><tr><td>Coauthor Physics</td><td>9094103 Full</td><td></td><td>12,813</td><td>1,282</td><td>-22.44</td><td>-22.13</td><td>0.00</td><td>5.15</td></tr></table>