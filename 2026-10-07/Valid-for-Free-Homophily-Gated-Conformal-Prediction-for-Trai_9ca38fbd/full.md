# Valid for Free: Homophily-Gated Conformal Prediction for Training-Free Node Classification with Tabular Foundation Models

Nguyen Duy Long<sup>1</sup>, Phung Minh Hien<sup>1</sup>, Nguyen Trong Viet<sup>2</sup>, and Nguyen Thai Anh<sup>3,\*</sup>

<sup>1</sup>Faculty of Information Technology, Posts and Telecommunications Institute of Technology, Hanoi, Vietnam

<sup>2</sup>Faculty of Engineering, School of Computer Science, The University of Sydney, Sydney, Australia

<sup>3</sup>Faculty of Information Technology, Van Lang School of Technology, Van Lang University, Ho Chi Minh City, Vietnam Email: longnd.b24cn364@stu.ptit.edu.vn, hienpm.b24cn199@stu.ptit.edu.vn, trng0882@uni.sydney.edu.au, anh.nt@vlu.edu.vn <sup>\*</sup>Corresponding author: Nguyen Thai Anh

Abstract—Tabular foundation models (TFMs) can classify the nodes of a graph without training on it, by reading node and neighborhood features as table rows next to labeled context rows. Work in this line reports predictive performance, not conformal coverage or prediction-set size. To our knowledge, we give the first reliability study of the setting, with TabICL as the TFM and half of each graph as labeled context. As for any predictor fixed before calibration, a frozen in-context predictor makes split conformal prediction exactly valid in finite samples, with no training, validation fold, or tuning on the target graph. An audit across ten graphs then shows that the training-free TabICL posterior has lower expected calibration error (ECE) than GCN with temperature scaling (GCN+TS) on nine of them. Its mean ECE over the ten graphs is 0.019, about 35 percent below the 0.029 of GCN+TS. We also introduce HG-DAPS, a training-free diffusion score whose homophily gate reads only the in-context[ labels, so the guarantee still holds. Relative to adaptive prediction sets (APS), it reduces mean set size by 5.8 to 17.1 percent on six homophilous graphs and changes it by under 1 percent on four heterophilous ones. On two binary, class-imbalanced graphs, a pre-registered trap case shows that gating on raw rather than adjusted homophily lowers coverage among low-homophily nodes by 0.27 and 0.12. Marginal coverage stays at the nominal 0.90 and masks this drop.

Index Terms—foundation models, conformal prediction, node classification, uncertainty quantification, trustworthy machine learning

## I. INTRODUCTION

Tabular foundation models (TFMs) such as TabPFN [1] and TabICL [2] are pretrained on synthetic tabular tasks and classify a new table by conditioning on its labeled rows, with no gradient updates. Recent work applies them to node classification without any training on the target graph [3], [4].<sup>1</sup> Each node becomes a row of its own and its neighborhood features, and a frozen TFM reads the labeled context rows and predicts the remaining nodes. On standard benchmarks this recipe is competitive with trained graph neural networks (GNNs), but reported evaluations focus on accuracy.

Accuracy alone is not sufficient for deployment. A fraudflagging system must know when to refer a node to a human, and that decision rests on the reliability of the model’s confidence. Conformal prediction has been used to measure this reliability for vision foundation models [5], but not for TFMs on graphs. The gap is sharper for a training-free model, since a reliability step that needs training or a validation fold would forfeit its main appeal.

Two aspects make the graph question non-routine. First, split conformal prediction needs calibration and test nodes to be exchangeable. For a trained GNN this holds when training, early stopping, and tuning never depend on the calibration split [6], [7], a condition that has to be checked for each protocol. Second, diffusing conformity scores along edges helps when neighbors tend to share labels (homophily), but on a heterophilous graph it mixes evidence across unlike classes. Deciding whether to diffuse therefore needs a homophily estimate that does not depend on the calibration split. The intuitive statistic, raw edge homophily, is inflated whenever the chance same-label rate is high, as with few or imbalanced classes [8]. A gate built on it can be wrong exactly where caution matters.

Our contributions are as follows (pipeline in Fig. 1).

• Valid for free (Lemma 1). We observe that split conformal prediction is exactly valid for a frozen in-context predictor by the transductive exchangeability argument of [6], [7], with no training protocol to audit and no validation fold.

• A pre-registered audit. On nine of ten graphs, trainingfree TabICL has lower expected calibration error (ECE) than temperature-scaled GCN, with no post-hoc step.

• HG-DAPS. We gate DAPS-style score diffusion [6] by the adjusted homophily of the context labels alone, with no tuning. Relative to APS, it shrinks mean set size on the six homophilous graphs and changes it by under 1% on the four heterophilous ones.

• A trap case. On two binary, class-imbalanced graphs, gating on raw edge homophily lowers coverage among low-homophily nodes, a drop that marginal coverage hides.

## II. RELATED WORK

Tabular foundation models on graphs. Besides node-table methods [3], [4], GraphPFN pretrains a graph-specific priorfitted network [9]. These approaches reach competitive accuracy without task-specific training, but none reports conformal coverage or prediction-set quality. Calibration appears only in a concurrent two-dataset ablation [10].

![](images/0ca665f36a620f8c786bd5cff7d518363d7ca9c52562c00d05e3c1b08aa94005.jpg)  
Fig. 1. Overview. (1) A featurizer ϕ maps each node to a table row (reduced own features, one- and two-hop neighborhood means, one-hop extrema, log-degree), and a frozen TabICL reads the labeled context rows to return ${ \hat { p } } ( \cdot \mid v ) . ( 2 )$ HG-DAPS mixes the score matrix S<sup>full</sup> (APS rows on $U ,$ one-hot LAC rows on $C ,$ written S in the panel) with its row-normalized neighbor average as in (4). The weight δ comes from a homophily gate that reads context labels only. On CiteSeer, $\hat { h } _ { \mathrm { a d j } } = 0$ .64 gives $\delta = 0 . 2 9$ , and the true-label score of the example node falls from 0.72 to 0.55. (3) Only then is U split at random into $\mathcal { D } _ { \mathrm { c a l } }$ and $\mathcal { D } _ { \mathrm { t e s t } } .$ , and $\check { C } ( v ) = \{ k : \tilde { S } _ { v , k } \leq \hat { q } \}$ keeps three labels where APS alone keeps four. Stages 1 and 2 do not depend on this split, so Lemma 1 gives $\mathrm { P r } ( y _ { v ^ { \ast } } \in \hat { C } ( v ^ { \ast } ) ) \geq 1 - \alpha$ for a random test node $v ^ { * }$ . Values are from one run (CiteSeer, context seed 0, $\alpha = 0 . 1 0 ,$ , classes indexed from 0).

Conformal prediction on graphs. Split conformal prediction gives distribution-free marginal coverage under exchangeability [11], with adaptive prediction sets (APS) [12], regularized APS (RAPS) [13], and the least ambiguous setvalued classifier (LAC) [14] as standard scores. In the inductive setting, weighted conformal scores account for network dependence [15]. In the transductive setting, exchangeability holds when the model uses no calibration or test labels and the split is uniform [6], [7], and score diffusion preserves it [6]. DAPS (diffusion APS) [6] diffuses nonconformity scores over edges, SNAPS [16] adds feature similarity and neighborhood information, and CF-GNN [7] learns a graph-based correction network. HeAD-CP [17] derives node- and edge-wise diffusion coefficients from GNN softmax outputs, which a frozen TFM posterior could also supply. HG-DAPS instead sets one graphlevel diffusion weight from the adjusted homophily of the observed context labels, with no tuning.

Calibration of graph models. Temperature scaling on held-out data is a standard remedy for miscalibrated neural networks [18]. CaGCN learns node-dependent calibration functions from neighboring confidence patterns [19], and GATS estimates node-wise temperatures with graph attention [20]. Both need held-out labels and target trained GNNs, whereas we audit a frozen TabICL posterior with no post-hoc step.

## III. SETTING AND EXACT FINITE-SAMPLE VALIDITY

## A. Setting

Let $G = ( V , E )$ be an undirected graph with node features $X \in \mathbb { R } ^ { N \times d }$ and labels $y _ { v } \in [ K ] = \{ 1 , . . . , K \}$ . A random subset $C \subset V$ (the context, drawn with a class-balance check, Section V) has observed labels $y _ { C }$ , and the remaining nodes

$U = V \backslash C$ form the evaluation pool. Each node is mapped to a table row

$$
\begin{array} { r l } & { \phi ( v ) = \big [ \mathrm { S V D } _ { 6 4 } ( x _ { v } ) , \mathrm { ~ S V D } _ { 6 4 } ( \bar { x } _ { v } ^ { ( 1 ) } ) , \mathrm { ~ S V D } _ { 6 4 } ( \bar { x } _ { v } ^ { ( 2 ) } ) , } \\ & { \qquad \mathrm { a g g r } , \mathrm { ~ l o g } ( 1 + \mathrm { d e g } v ) \big ] , } \end{array}
$$

where $\bar { x } _ { v } ^ { ( r ) }$ is the mean of features over the r-hop neighborhood, aggr holds the one-hop maximum and minimum of the reduced features, and every singular value decomposition (SVD) is fit on all nodes using no labels. Afrozen TFM f reads a prompt of at most 5,000 context rows $( \phi ( c ) , y _ { c } )$ and returns by in-context learning, at softmax temperature 1 and with no parameter update, a posterior $\hat { p } ( \cdot | \ v ) \in \Delta ^ { K - 1 }$ for every $v \in U$

As nonconformity score we use APS [12]. With classes sorted by decreasing $\hat { p } ( k \mid v )$ , the score of label k is the probability mass ranked before k plus a randomized share of its own mass. One uniform draw per row is made once and cached, so the score matrix $S \in [ 0 , 1 ] ^ { | U | \times K }$ of the evaluation pool is fixed before any split. Given a calibration set $\mathcal { D } _ { \mathrm { c a l } } \subset U$ of size $n _ { \mathrm { c a l } }$ and a level $\alpha ,$ split conformal prediction takes the threshold

$$
\hat { q } \ = \ s _ { ( j ) } , \qquad j = \lceil ( n _ { \mathrm { c a l } } + 1 ) ( 1 - \alpha ) \rceil ,\tag{1}
$$

the j-th smallest calibration score $s _ { i } = S _ { i , y _ { i } } ~ ( \hat { q } = \infty { \mathrm { ~ i f ~ } } j >$ $n _ { \mathrm { c a l } } )$ , and predicts $\hat { C } ( v ) = \{ k : S _ { v , k } \leq \hat { q } \}$ for every test node.

## B. Valid for free

Every component that produces the score matrix is fixed before any calibration data is chosen: the graph, the features, the featurization, the context and its prompt, the frozen model with its inference seeds, and the APS draws. Validity therefore needs no assumption about the data distribution, the model, or the graph, only a uniformly random calibration/test split of U. Validity is thus free of training, of a validation fold, and of hyperparameter selection, though not free of labels, since the context and the calibration set are labeled.

Lemma 1 (Exact finite-sample validity). Condition on G, X, all labels, the context $C$ with its prompt, the frozen predictor $f$ with its inference seeds, and the APS draws, so that the pairs $\{ ( S _ { v , \cdot } , y _ { v } ) \} _ { v \in U }$ form a fixed finite population. Let $\mathcal { D } _ { \mathrm { c a l } }$ be a uniformly random subset of U of size $n _ { \mathrm { c a l } } $ , and let the test node $v ^ { * }$ be drawn uniformly from $U \backslash { \mathcal { D } } _ { \mathrm { c a l } }$ . Then, with $\hat { q }$ as in (1),

$$
\operatorname* { P r } \left( y _ { v ^ { * } } \in { \hat { C } } ( v ^ { * } ) \right) \geq 1 - \alpha ,
$$

where the probability is over the random split alone.

Proof. The $n _ { \mathrm { c a l } } + 1$ true-label scores $( s _ { 1 } , \ldots , s _ { n _ { \mathrm { c a l } } } , s _ { v ^ { * } } )$ are a uniform sample without replacement from the fixed population $\{ S _ { v , y _ { v } } \} _ { v \in U }$ , so their joint law is invariant under permutation and they are exchangeable. Break ties uniformly at random and let R be the rank of $s _ { v ^ { * } }$ among the $n _ { \mathrm { c a l } } + 1$ scores. By exchangeability R is uniform on $\{ 1 , \ldots , n _ { \mathrm { c a l } } + 1 \}$ . If $R \leq j .$ at most $j - 1$ calibration scores lie strictly below $s _ { v ^ { * } }$ , so $s _ { v ^ { * } } \leq$ $s _ { ( j ) } = \hat { q }$ and $y _ { v ^ { * } } \in \hat { C } ( v ^ { * } )$ . Hence the coverage probability is at least $\mathrm { P r } ( R \leq j ) = j / ( n _ { \mathrm { c a l } } + 1 ) \geq 1 - \alpha$ □

Lemma 1 is the finite-population form of Proposition 1 of [6] and of the transductive exchangeability argument of [7]. Because it conditions on the frozen predictor, it needs no permutation-equivariance assumption on the model.

Remark 1 (No upper bound under ties). The familiar upper bound $\begin{array} { r } { 1 - \alpha + \frac { 1 } { n _ { \mathrm { c a l } } + 1 } } \end{array}$ requires pairwise distinct true-label scores in U. When the posterior takes few distinct values, the non-randomized LAC score $1 - \hat { p } ( y \mid v )$ has heavy ties. This happens for the SVD-only ablation of TabICL on Minesweeper, whose raw features are one-hot counts of neighboring mines. There LAC coverage reaches 0.96 at $\alpha = 0 . 1 0$ . This paper claims only the lower bound.

Remark 2 (Trained GNNs). Lemma 1 applies to any predictor that depends only on $( G , X , C , y _ { C } )$ and on randomness independent of the calibration/test split, including a GNN trained and early-stopped on the context, as our baselines are. For a trained model this premise has to be verified by auditing the training protocol, and a validation fold consumes context labels. The training-free pipeline satisfies it with no training protocol to audit.

## IV. HG-DAPS: HOMOPHILY-GATED DIFFUSION SCORES

Diffusing nonconformity scores over the graph sharpens prediction sets when neighbors tend to share labels. On heterophilous graphs the same smoothing mixes scores of unlike classes and can enlarge the sets [17]. DAPS selects its diffusion weight by tuning on labeled nodes [6], and HeAD-CP adapts node-wise coefficients from model predictions [17]. HG-DAPS decides whether to diffuse from the observed context labels, so the decision does not depend on the calibration split.

## A. A context-only homophily gate

Let $E _ { C } \subseteq E$ be the set of intra-context edges, whose endpoints both lie in $C ,$ , and let $m = \vert E _ { C } \vert$ . All quantities below use only $E _ { C }$ and $y _ { C }$ . The raw edge homophily $h _ { \mathrm { e d g e } }$

is the fraction of edges in $E _ { C }$ whose endpoints share a label. We use the adjusted homophily of Platonov et al. [8],

$$
\hat { h } _ { \mathrm { a d j } } = \frac { h _ { \mathrm { e d g e } } - P } { 1 - P } , \qquad P = \sum _ { k = 1 } ^ { K } p _ { k } ^ { 2 } ,\tag{2}
$$

where $p _ { k }$ is the fraction of the 2m endpoints of $E _ { C }$ that carry label k. Here P is the same-label rate expected when edge endpoints are matched at random with their labels. So $\hat { h } _ { \mathrm { a d j } }$ is about 0 when edges ignore labels, whatever the class balance, and it equals 1 only when $h _ { \mathrm { e d g e } } = 1$ . Since $\hat { h } _ { \mathrm { a d j } }$ is a ratio estimate from m edges, we shrink it toward 0 when the context has few intra-context edges:

$$
\delta = \delta _ { \mathrm { m a x } } \mathrm { c l i p } _ { [ 0 , 1 ] } \Big ( \hat { h } _ { \mathrm { a d j } } \cdot \frac { m } { m + m _ { 0 } } \Big ) ,\tag{3}
$$

with $\delta _ { \mathrm { m a x } } = 0 . 5$ and $m _ { 0 } = 1 0 0$ set a priori in the locked configuration. We set $\delta = 0$ when $m = 0$ or $P = 1$ , and then, as whenever $\hat { h } _ { \mathrm { a d j } } \leq 0$ , HG-DAPS reduces exactly to APS. Two ablations use the same form: the raw-homophily gate replaces $\hat { h } _ { \mathrm { a d j } }$ by $h _ { \mathrm { e d g e } }$ in (3), and ungated DAPS fixes $\delta = \delta _ { \mathrm { m a x } }$

## B. Gated diffusion of the score matrix

Extend the score matrix to all of V as $S ^ { \mathrm { f u l l } } \in [ 0 , 1 ] ^ { | V | \times K }$ A context row is the one-hot LAC score of its known label, $S _ { c , k } ^ { \mathrm { f u l l } } = \mathbf { 1 } [ k \neq y _ { c } ]$ , so known labels enter the diffusion as certain evidence, and evaluation rows keep their cached APS scores. With adjacency matrix A and degree matrix D, one gated diffusion step gives

$$
\tilde { S } = ( 1 - \delta ) S ^ { \mathrm { f u l l } } + \delta D ^ { - 1 } A S ^ { \mathrm { f u l l } } ,\tag{4}
$$

and HG-DAPS calibrates the threshold (1) on the true-label entries of S<sup>˜</sup> in $U$ and forms prediction sets from the same matrix. Nodes of degree zero keep their own row. The sets remain nested in $\hat { q }$ but need not be prefixes of the TabICL ranking. The cost is one sparse multiply, $O ( | E | K )$ , on top of one cached forward pass of the frozen model.

Corollary 1 (Validity is preserved). The map $S ^ { \mathrm { f u l l } } \mapsto \tilde { S }$ is a fixed function of $( A , \delta ) , S ^ { \mathrm { f u l l } }$ depends only on $( S , y _ { C } )$ , and δ in (3) depends only on $( E _ { C } , y _ { C } )$ . All are determined before the calibration set is drawn, so $\{ ( \tilde { S } _ { v , \cdot } , y _ { v } ) \} _ { v \in U }$ is again a fixed population and Lemma 1 applies verbatim to HG-DAPS. Coverage of at least $1 - \alpha$ holds for every graph and every gate value, including a misestimated one.

Corollary 1 mirrors Proposition 2 of [6] for a gated weight. Only efficiency (set size) and conditional coverage across node groups depend on the gate. This motivates the pre-registered trap-case hypothesis H4. On Minesweeper and Tolokers, labels are binary and imbalanced, so $h _ { \mathrm { e d g e } }$ is high while the mean $\hat { h } _ { \mathrm { a d j } }$ is at most 0.10. A gate built on $h _ { \mathrm { e d g e } }$ turns diffusion on and is expected to lower coverage for low-homophily nodes, whereas the adjusted gate (3) stays near zero. Marginal coverage stays at least $1 - \alpha$ in both cases, so an audit of marginal coverage alone would miss the failure.

TABLE I  
TRAINING-FREE TABICL NODE CLASSIFICATION AT $\alpha = 0 . 1 0 \mathrm { { ; } }$ MEAN±SD OVER 3 CONTEXT SEEDS. ACC IS TABICL ACCURACY. COVERAGE AND SET SIZES ARE AVERAGED OVER 20 CALIBRATION/TEST RESPLITS PER SEED. ECE IS COMPARED WITH GCN+TS, THE PRE-REGISTERED COMPARATOR. SET SIZES OF APS AND HG-DAPS SHARE THE SAME 0.90 COVERAGE GUARANTEE. DATASETS ARE SORTED BY THE CONTEXT ESTIMATE $\hat { h } _ { \mathrm { a d j } }$
<table><tr><td>Dataset</td><td> $\hat { h } _ { \mathrm { a d j } }$ </td><td>Acc</td><td>ECE (TabICL)</td><td> $\mathrm { E C E } \ ( \mathrm { G C N + T S } )$ </td><td>Cov (APS)</td><td>Size (APS)</td><td>Size (HG-DAPS)</td></tr><tr><td>Photo</td><td>0.79</td><td>0.960±0.004</td><td> $0 . 0 0 7 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 0 1 7 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 9 0 1 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td>1.047±0.009</td><td>0.956±0.004</td></tr><tr><td>Cora</td><td>0.79</td><td>0.868±0.005</td><td>0.028±0.004</td><td>0.034±0.004</td><td>0.901±0.004</td><td>1.362±0.022</td><td>1.151±0.018</td></tr><tr><td>Coauthor-CS</td><td>0.78</td><td>0.934±0.001</td><td>0.008±0.004</td><td>0.007±0.002</td><td>0.900±0.001</td><td>1.101±0.014</td><td>0.988±0.004</td></tr><tr><td>PubMed</td><td>0.68</td><td>0.877±0.005</td><td>0.010±0.002</td><td>0.014±0.002</td><td>0.900±0.001</td><td>1.198±0.009</td><td>1.128±0.017</td></tr><tr><td>CiteSeer</td><td>0.64</td><td>0.757±0.011</td><td>0.041±0.003</td><td>0.054±0.014</td><td>0.902±0.003</td><td>1.921±0.060</td><td>1.719±0.063</td></tr><tr><td>WikiCS</td><td>0.59</td><td>0.860±0.003</td><td>0.015±0.000</td><td>0.040±0.004</td><td>0.900±0.002</td><td>1.482±0.022</td><td>1.229±0.002</td></tr><tr><td>Amazon-Ratings</td><td>0.14</td><td>0.485±0.006</td><td>0.021±0.002</td><td>0.023±0.003</td><td>0.899±0.003</td><td>2.875±0.021</td><td>2.856±0.010</td></tr><tr><td>Tolokers</td><td>0.10</td><td>0.827±0.003</td><td>0.016±0.003</td><td>0.046±0.003</td><td>0.900±0.003</td><td>1.298±0.012</td><td>1.288±0.010</td></tr><tr><td>Minesweeper</td><td>0.01</td><td>0.850±0.005</td><td>0.010±0.002</td><td>0.020±0.004</td><td>0.899±0.001</td><td>1.235±0.003</td><td>1.234±0.003</td></tr><tr><td>Actor</td><td>0.00</td><td>0.362±0.007</td><td>0.031±0.004</td><td>0.036±0.008</td><td>0.901±0.003</td><td>3.406±0.021</td><td>3.406±0.021</td></tr></table>

## V. EXPERIMENTAL SETUP

Datasets. We use ten standard node classification benchmarks, with context estimates $\hat { h } _ { \mathrm { a d j } }$ from 0.00 to 0.79. Cora, CiteSeer, and PubMed [23], Photo and Coauthor-CS [24], and WikiCS [25] are homophilous $( \hat { h } _ { \mathrm { a d j } } > 0 . 5 )$ . Actor [26], Amazon-Ratings, Minesweeper, and Tolokers [27] are heterophilous $( \hat { h } _ { \mathrm { a d j } } < 0 . 2 )$

Protocol. For each dataset we draw three 50% context splits C, and the remaining nodes form U. Each context receives 20 calibration/test resplits of U, giving 60 resplits per dataset, with $n _ { \mathrm { c a l } } = \operatorname* { m i n } ( 1 0 0 0 , \lfloor \lvert U \lvert / 2 \rfloor )$ and $n _ { \mathrm { t e s t } } = | U | - n _ { \mathrm { c a l } }$ . All reported conformal results use $\alpha = 0 . 1 0$ . The low-homophily stratum holds the evaluation nodes with fewer than 20% samelabel neighbors, a label-based quantity used for evaluation only.

Models. The primary model is TabICL on the feature table ϕ. On Amazon-Ratings, Coauthor-CS, PubMed, Tolokers, and WikiCS, the context holds more than 5,000 nodes, so the TabICL prompt uses only part of the context labels. A featureonly ablation (SVD-only) runs the same frozen TabICL on the SVD block of the raw node features alone. The trained baselines are a graph convolutional network (GCN) [28], a graph attention network (GAT) [29], and GCN with temperature scaling (GCN+TS), tuned on a validation fold carved from the context. A multilayer perceptron (MLP) and logistic regression use the same feature table. Label propagation [30] completes the eight models.

Safeguards. Every random draw is seeded deterministically from the dataset name, the context seed, and the purpose of the draw. The context split is redrawn when its class frequencies deviate by more than 0.05 from those of the full graph. The validation fold and the prompt are checked in the same way against the context. This check selects C using labels outside it, but Lemma 1 conditions on C, so validity is unaffected. The calibration/test resplits are pure uniform draws, as the lemma requires. The four hypotheses (H1 audit, H2 validity, H3 method, H4 trap case), their thresholds, and the decision rule were fixed in a locked configuration before the full run. Inference runs once per context split on one 8 GB consumer GPU, and all later CPU steps take seconds.

## VI. RESULTS

Audit (H1). In the pre-registered comparison, TabICL has lower ECE than GCN+TS on 9 of the 10 datasets (Table I, Fig. 2(a)). The exception is Coauthor-CS, the only graph with more than ten classes, where GCN+TS reaches 0.007 against 0.008. On Actor, Amazon-Ratings, and CiteSeer, the ECE advantage of TabICL is smaller than the standard deviation of GCN+TS over context seeds. In a secondary comparison with all trained baselines, TabICL has the lowest ECE on 8 of the 10 datasets. Its mean ECE over the ten datasets is 0.019, about 35% below the 0.029 of GCN+TS and of MLP, with no task-specific training and no post-hoc calibration. It also attains the highest mean accuracy, 0.778 against 0.752 for GCN. On the binary Minesweeper and Tolokers, its accuracy is about 0.05 above the majority-class rate. Its SVD-only ablation has lower ECE on three datasets, at lower mean accuracy (0.733). Label propagation is poorly calibrated, with mean ECE 0.120.

Validity (H2). The pre-registered check covers 180 cells of the TabICL posterior: 10 datasets, 3 context seeds, and 6 scores (LAC, APS, RAPS, ungated DAPS, HG-DAPS, and the raw-homophily gate). In every cell, the mean coverage over the 20 resplits of that seed lies inside the exact two-sided Clopper-Pearson 95% interval at 0.90 for $n _ { \mathrm { t e s t } }$ trials. The lowest cell mean is 0.894 (Minesweeper, LAC). The band treats one test set as binomial and ignores calibration variability, so it is too strict for a single resplit and loose for a mean over 20 resplits. It is therefore only a coarse sanity check, and its outcome is consistent with Lemma 1 (pooled means in Fig. 2(b)). Outside the pre-registered cells, LAC on the SVD-only ablation exceeds the upper edge on Minesweeper, the tie-induced overcoverage of Remark 1.

Method (H3). The pre-registered H3 contrast covers five homophilous graphs. On all five, HG-DAPS reduces mean set size relative to APS by 5.8% (PubMed) to 17.1% (WikiCS), and it does so by 10.3% on Coauthor-CS (Table I, Fig. 3(a)). On these six graphs its mean coverage over 60 resplits stays within 0.2 percentage points of that of APS. On Photo and Coauthor-CS, accuracy above 0.90 forces APS, and a mean size below 1 forces HG-DAPS, to output some empty sets, which count as size 0 and may explain part of the reduction. On the four heterophilous graphs the adjusted gate gives $\delta \leq 0 . 0 7$ and mean set size changes by less than 1%, while ungated diffusion enlarges Actor sets by 9.9% (Fig. 4(a)). On the six homophilous graphs, ungated DAPS shrinks the sets by a similar average amount (11.5% against 11.3%), so there the gate mainly decides whether to diffuse.

(a) Set size at the same 0.90 coverage guarantee  
![](images/faf8c25de4994cb7adc6950014ae7c6ceb8bec14943ab3a1cdd2226cdc14f29c.jpg)

![](images/66753c2064b6a6f23e9e8625c1511ec2dc8a0c3a6f8e0b8b3cff531966ff50cc.jpg)

Fig. 2. Reliability audit at $\alpha = 0 . 1 0$ . Datasets are ordered by the context estimate $\hat { h } _ { \mathrm { a d j } }$ (in parentheses), and the dashed rule separates homophilous $( \hat { h } _ { \mathrm { a d j } } > 0 . 5 )$ from heterophilous $( \hat { h } _ { \mathrm { a d j } } < 0 . 2 )$ graphs. (a) ECE with 15 equal-mass bins [21] for TabICL, GCN+TS, and MLP, with the TabICL and GCN+TS values in Table I. (b) Marginal coverage of LAC, APS, and RAPS on the TabICL posterior, averaged over 60 calibration/test resplits, as deviation from the nominal 0.90. Error bars are exact 95% Clopper-Pearson intervals [22] at each mean for $n _ { \mathrm { t e s t } }$ trials. They are sized for one test set, not for the 60-resplit mean, and every error bar covers the nominal level.  
![](images/554815e6a05f41dde627cdaab4978564b4e39a7f6e98df2637a3ee6e22e511e9.jpg)

![](images/e17a28a26587867f555296caf184fca125ac7b288a68a493e0239119893ab5ef.jpg)

Fig. 3. HG-DAPS at $\alpha = 0 . 1 0 ,$ ordered as in Fig. 2. (a) Mean set size of APS and HG-DAPS at the same coverage guarantee, with the relative change. (b) Gate statistics per context split, colored by set-size reduction. ${ \mathrm { B y ~ } } ( 2 ) .$ , the gap below the diagonal $\hat { h } _ { \mathrm { a d j } } = h _ { \mathrm { e d g e } } \mathrm { i s } P ( 1 - h _ { \mathrm { e d g e } } ) / ( 1 - P )$ , which grows with class imbalance at fixed $h _ { \mathrm { e d g e } } .$ The shaded trap corner $( h _ { \mathrm { e d g e } } > 0 . 5 , \ : \hat { h } _ { \mathrm { a d j } } < 0 . 2 )$ holds both trap datasets.  
![](images/1d04ea2ba6ff6b0bc85cf265c92e1cc73d5af804185dde6428b6416a801826af.jpg)

![](images/cfd4c363e2f2fccd7723466c8f56975f8223e89fd1cc1ba380d9816745ef4f32.jpg)  
Fig. 4. Gate ablation at $\alpha = 0 . 1 0 ,$ ordered as in Fig. 2. Ungated DAPS and the raw-homophily gate are defined after (3). (a) Change in mean set size relative to APS. (b) Coverage in the low-homophily stratum (fewer than 20% same-label neighbors) on the two trap datasets. Their mean marginal coverage over 60 resplits is 0.897 to 0.901 for all four methods, which hides the drop.

Trap case (H4). On Minesweeper and Tolokers the mean $h _ { \mathrm { e d g e } }$ is 0.69 and 0.59 (Fig. 3(b)), so the raw gate diffuses with $\delta = 0 . 3 4$ and 0.30, whereas the adjusted gate gives $\delta = 0 . 0 0 3$ and 0.05. Marginal coverage stays at the nominal 0.90, but coverage in the low-homophily stratum falls by 0.27 and 0.12 relative to the adjusted gate (Fig. 4(b)). The raw gate also shrinks mean set size relative to APS by 4.4% and 3.3%, against 0.1% and 0.8% for HG-DAPS, so an efficiency audit would favor it. All methods under-cover this stratum, with APS at 0.82 and 0.75, and the adjusted gate stays within 0.02 of APS, so it limits the harm without restoring coverage.

Scope and limitations. Lemma 1 and Corollary 1 hold for any frozen predictor, but the ECE and set-size results are specific to TabICL. We did not evaluate TabPFN v2, GraphPFN [9], or HeAD-CP [17] on the same posterior. The guarantee averages over calibration draws, so one fixed calibration set can realize coverage below $1 - \alpha$ . Set sizes are compared with APS only, not with LAC, RAPS, or DAPS with a context-tuned weight, and empty-set rates are not reported. Stratum coverage is reported only on the trap graphs, although HG-DAPS diffuses on homophilous graphs with weights such as $\delta = 0 . 2 9$ on CiteSeer (Fig. 1), close to the raw-gate weights of H4. We did not vary $\delta _ { \mathrm { m a x } } , m _ { 0 }$ , or the 50% context fraction, and no graph has $\hat { h } _ { \mathrm { a d j } }$ between 0.2 and 0.5. We also did not evaluate class-specific calibration [14], multi-hop diffusion, or a variant that separates the one-hot context rows, which DAPS does not use [6], from score smoothing.

## VII. CONCLUSION

A frozen in-context predictor makes split conformal prediction exactly valid in finite samples with no training protocol to audit (Lemma 1). On ten graphs, training-free TabICL has lower ECE than GCN+TS on nine, and HG-DAPS reduces mean set size relative to APS on the six homophilous ones. Its adjusted gate avoids the extra loss of 0.12 to 0.27 in lowhomophily coverage that raw-homophily gating causes on the two trap graphs. Whether HG-DAPS lowers that coverage inside homophilous graphs is the main open question.

## REFERENCES

[1] N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer, S. B. Hoo, R. T. Schirrmeister, and F. Hutter, “Accurate predictions on small data with a tabular foundation model,” Nature, vol. 637, pp. 319–326, 2025.

[2] J. Qu, D. Holzmüller, G. Varoquaux, and M. Le Morvan, “TabICL: A tabular foundation model for in-context learning on large data,” in Proc. Int. Conf. Machine Learning (ICML), vol. 267, 2025, pp. 50 817–50 847.

[3] A. Hayler, X. Huang, <sup>˙</sup>I. <sup>˙</sup>I. Ceylan, M. Bronstein, and B. Finkelshtein, “Bringing graphs to the table: Zero-shot node classification via tabular foundation models,” 2025, arXiv:2509.07143.

[4] D. Eremeev, G. Bazhenov, O. Platonov, A. Babenko, and L. Prokhorenkova, “Turning tabular foundation models into graph foundation models,” 2025, NeurIPS 2025 Workshop on New Perspectives in Graph Machine Learning; arXiv:2508.20906.

[5] L. Fillioux, J. Silva-Rodríguez, I. Ben Ayed, P.-H. Cournède, M. Vakalopoulou, S. Christodoulidis, and J. Dolz, “Are foundation models for computer vision good conformal predictors?” Transactions on Machine Learning Research, 2026, arXiv:2412.06082.

[6] S. H. Zargarbashi, S. Antonelli, and A. Bojchevski, “Conformal prediction sets for graph neural networks,” in Proc. Int. Conf. Machine Learning (ICML), 2023.

[7] K. Huang, Y. Jin, E. Candès, and J. Leskovec, “Uncertainty quantification over graph with conformalized graph neural networks,” in Advances in Neural Information Processing Systems (NeurIPS), 2023.

[8] O. Platonov, D. Kuznedelev, A. Babenko, and L. Prokhorenkova, “Characterizing graph datasets for node classification: Homophily-heterophily dichotomy and beyond,” in Advances in Neural Information Processing Systems (NeurIPS), 2023.

[9] D. Eremeev, O. Platonov, G. Bazhenov, A. Babenko, and L. Prokhorenkova, “GraphPFN: A prior-data fitted graph foundation model,” in Proc. Int. Conf. Machine Learning (ICML), 2026, arXiv:2509.21489.

[10] M. Yang, Z. Guo, J. Yang, and W. Zuo, “LoGIC: Budgeted context construction for node-level graph in-context learning with tabular foundation models,” 2026, arXiv:2609.05955.

[11] V. Vovk, A. Gammerman, and G. Shafer, Algorithmic Learning in a Random World. Springer, 2005.

[12] Y. Romano, M. Sesia, and E. J. Candès, “Classification with valid and adaptive coverage,” in Advances in Neural Information Processing Systems (NeurIPS), 2020.

[13] A. N. Angelopoulos, S. Bates, J. Malik, and M. I. Jordan, “Uncertainty sets for image classifiers using conformal prediction,” in Proc. Int. Conf. Learning Representations (ICLR), 2021.

[14] M. Sadinle, J. Lei, and L. Wasserman, “Least ambiguous set-valued classifiers with bounded error levels,” Journal of the American Statistical Association, vol. 114, no. 525, pp. 223–234, 2019.

[15] J. Clarkson, “Distribution free prediction sets for node classification,” in Proc. Int. Conf. Machine Learning (ICML), vol. 202, 2023, pp. 6268– 6278.

[16] J. Song, J. Huang, W. Jiang, B. Zhang, S. Li, and C. Wang, “Similaritynavigated conformal prediction for graph neural networks,” in Advances in Neural Information Processing Systems (NeurIPS), 2024.

[17] P. B. N. Lam and N. T. Anh, “HeAD-CP: Heterophily-aware diffused conformal prediction sets for graph neural networks,” in Proc. Int. Conf. Multimedia Analysis and Pattern Recognition (MAPR), 2026, arXiv:2607.25273.

[18] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. Int. Conf. Machine Learning (ICML), 2017.

[19] X. Wang, H. Liu, C. Shi, and C. Yang, “Be confident! Towards trustworthy graph neural networks via confidence calibration,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 34, 2021, pp. 23 768– 23 779.

[20] H. H.-H. Hsu, Y. Shen, C. Tomani, and D. Cremers, “What makes graph neural networks miscalibrated?” in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, 2022, pp. 13 775–13 786.

[21] J. Nixon, M. W. Dusenberry, L. Zhang, G. Jerfel, and D. Tran, “Measuring calibration in deep learning,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW), 2019.

[22] C. J. Clopper and E. S. Pearson, “The use of confidence or fiducial limits illustrated in the case of the binomial,” Biometrika, vol. 26, no. 4, pp. 404–413, 1934.

[23] P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Gallagher, and T. Eliassi-Rad, “Collective classification in network data,” AI Magazine, vol. 29, no. 3, pp. 93–106, 2008.

[24] O. Shchur, M. Mumme, A. Bojchevski, and S. Günnemann, “Pitfalls of graph neural network evaluation,” in NeurIPS Workshop on Relational Representation Learning (R2L), 2018.

[25] P. Mernyei and C. Cangea, “Wiki-CS: A Wikipedia-based benchmark for graph neural networks,” in ICML Workshop on Graph Representation Learning and Beyond (GRL+), 2020.

[26] H. Pei, B. Wei, K. C.-C. Chang, Y. Lei, and B. Yang, “Geom-GCN: Geometric graph convolutional networks,” in Proc. Int. Conf. Learning Representations (ICLR), 2020.

[27] O. Platonov, D. Kuznedelev, M. Diskin, A. Babenko, and L. Prokhorenkova, “A critical look at the evaluation of GNNs under heterophily: Are we really making progress?” in Proc. Int. Conf. Learning Representations (ICLR), 2023.

[28] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. Int. Conf. Learning Representations (ICLR), 2017.

[29] P. Velickoviˇ c, G. Cucurull, A. Casanova, A. Romero, P. Liò, and´ Y. Bengio, “Graph attention networks,” in Proc. Int. Conf. Learning Representations (ICLR), 2018.

[30] X. Zhu and Z. Ghahramani, “Learning from labeled and unlabeled data with label propagation,” Carnegie Mellon University, Tech. Rep. CMU-CALD-02-107, 2002.