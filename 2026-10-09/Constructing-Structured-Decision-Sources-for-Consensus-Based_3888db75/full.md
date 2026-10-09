# Constructing Structured Decision Sources for Consensus-Based Pseudo-Label Learning

Long Wang alueas@126.com ORCID: 0009-0006-8371-1562

## Abstract

Consensus can make pseudo-label learning more reliable, but only when its predictors contribute genuinely diferent evidence. Multiple models that repeat the same boundary provide additional votes without additional information. We address this problem by constructing decision sources through controlled changes to within-class structure. Starting from a shared graph representation, we vary center granularity and neighborhood mixing, reproduce each resulting source to test its stability, and select a complementary subset using node-pair co-assignment. Unanimous predictions from the selected sources are then ranked for student training. On the public fixed splits of Cora, CiteSeer, and PubMed, evaluated with five random seeds, the constructed sources improve fixed-budget training pseudo-label precision by 1.19–4.39 percentage points over three conventionally initialized GCN sources. Under matched structural filters, three-source consensus is more precise than each constituent source in all 45 dataset–slot–seed comparisons. The gains are strongest in pseudo-label quality: downstream accuracy remains competitive but does not lead on every dataset. These results identify source construction—rather than model count alone—as an important design problem for consensus-based pseudo-label learning.

Keywords: graph semi-supervised learning; decision-source construction; multi-source consensus; near-center filtering; pseudo-label reliability

## 1 Introduction

Graph neural networks can exploit both node attributes and topology, yet their performance still depends on scarce labeled data. Pseudo-labeling extends supervision to unlabeled nodes, but an early error can be reinforced when the model later trains on its own prediction; message passing may then spread that error beyond the original node. Reliable pseudo-label admission is therefore a central problem in graph self-training [1].

Most work controls pseudo-label noise after predictions have been produced, using confidence schedules, training dynamics, prototypes, or agreement among models [2, 3, 4, 5, 6, 7]. Agreement is useful only if the participating predictors fail in diferent ways. Random initializations may instead recover nearly identical boundaries or introduce unstable variation. The unresolved question is therefore upstream of voting: how can a single graph yield decision sources that are reproducible enough to trust and diferent enough to cross-check one another?

DiPat demonstrates that one graph can support multiple decision patterns and that their agreement can guide pseudo-label selection [8]. We study the preceding design problem: constructing, verifying, and selecting the sources that enter consensus. This distinction matters because apparent diversity can arise from transient optimization noise rather than a stable change in structural criterion.

Our framework varies the number of within-class centers k and the mixing coeficient β between node and neighborhood representations. Each intervention produces a candidate source from a shared warm start. Reproduction training tests whether its induced partition persists, and node-pair co-assignment measures whether two sources impose meaningfully diferent partitions without relying on permutable center identifiers. We select three stable, complementary sources from four candidates and rank their unanimous pseudo-labels by the weakest supporting confidence. A training-calibrated near-center gate provides a separate precision–coverage diagnostic; student training uses the ranked consensus labels directly.

Experiments on Cora, CiteSeer, and PubMed compare the constructed sources with a supervised GCN, a graph-only DiPat adaptation, three conventionally initialized GCN sources, and matched single-source controls. The constructed sources improve fixed-budget pseudo-label precision on all three datasets, and consensus improves precision over every matched constituent source. Downstream Accuracy and Macro-F1 remain competitive but are not uniformly superior, separating the quality of the admitted evidence from the final task outcome.

Our contributions are as follows:

• We formulate source construction as part of the consensus-learning problem and instantiate it through controlled within-class granularity and neighborhood mixing.

• We connect source reproducibility and complementarity to consensus utility through coassignment probes, matched single-source controls, and random-source replacement.

• Across three citation graphs, the resulting sources consistently improve pseudo-label evidence while revealing the accompanying coverage and downstream-performance tradeofs.

## 2 Related Work

## 2.1 Pseudo-label admission, correction, and reliability

Pseudo-label admission methods determine when and how strongly model predictions should enter training. Fixed confidence rules [9], curricula and dynamic thresholds [2, 10], class-adaptive thresholds [3, 11], and soft weighting [12] all manage the quantity–quality tradeof after a predictor has produced its scores. Our question precedes admission: which predictors should supply those scores to a consensus rule?

Pseudo-labels can also be improved through teacher feedback, complementary classifiers, prototype consistency, or temporal disagreement [13, 14, 5, 4]. These approaches refine supervision using information already present in the training process. We instead treat the generation and validation of diferentiated decision signals as an explicit design problem.

## 2.2 Structure and noise control in graph semi-supervised learning

Graph semi-supervised methods exploit structure through clustering and self-training [15], auxiliary manifold mixing [16], stochastic propagation and consistency [17, 18], or explicit correction and smoothing [19]. They show that graph structure can regularize predictions, but generally use structural variation to strengthen one learner rather than to construct a set of comparable sources.

Robust graph learning further addresses noisy labels and topology through graph reconstruction, correction, and topology-aware sample selection [20, 21, 22, 23, 24]. Our intervention has a diferent role: center granularity and neighborhood mixing deliberately create candidate sources, and a shared co-assignment probe distinguishes reproducible structural variation from incidental optimization noise.

## 2.3 Multi-source consensus, diversity, and structured decision sources

Agreement models and diverse ensembles use cross-model consistency to control unlabeled supervision [25, 7, 6, 26]. Their gains depend on exploitable complementarity, not the number of members alone. This observation motivates treating diversity as something to construct and verify before voting.

DiPat is the closest comparison because it derives complementary decision patterns from one graph [8]. It begins with predefined patterns and studies their use in consensus; we study how candidate sources can be generated, distinguished, and accepted before fusion. Structural interventions create the candidates, co-assignment makes their partitions comparable, and reproduction tests whether those partitions persist.

## 3 Method

Figure 1 summarizes the pipeline from a shared warm start through source construction and reproduction to consensus use. Controlled structural intervention precedes consensus learning, and the q90 quality diagnostic is kept separate from the student-training branch.

![](images/ec84a9795187614682202da155809047d1b2721f0c0eb3352833facd81ea4d3d.jpg)  
Figure 1: Overview of structured decision-source construction. Structural interventions $( k , \beta )$ generate four candidates from a shared warm-start representation. Reproduction stability and between-source co-assignment diferences select three sources. Their consensus candidates enter either the ${ \mathrm { q } } 9 0$ quality diagnostic or minimum-confidence ranking for student training; the q90 gate is not used in student training.

## 3.1 Problem definition and notation

Given a graph $G = ( V , E , X )$ , a public training-node set $T \subset V$ , and class set Y, the candidate domain is $U = V \setminus T$ This is a transductive setting: U includes validation, test, and other unlabeled nodes, but their ground-truth labels are not used in source construction or filtering. Let $\Omega = \{ \omega _ { 1 } , \dots , \omega _ { M } \}$ be the intervention set, where $\omega _ { m } = ( k _ { m } , \beta _ { m } )$ specifies the number of within-class centers and the neighborhood mixing coeficient. Training on the same data under diferent interventions yields

$$
f _ { m } ( G ) = \{ p _ { m } ( y \mid i ; \omega _ { m } ) : i \in V \} , \qquad m = 1 , \ldots , M .\tag{1}
$$

Definition 1 (Sample co-assignment probe). Group nodes in U by the class predicted by the shared warm-start model, and sample a set of within-group node pairs P under a fixed random seed. Let $a _ { m } ( i )$ denote the within-class center assigned to node i by source m. For $p = ( i , j ) \in \mathcal { P }$ define

$$
C _ { m } ( \boldsymbol { p } ) = \mathbb { I } [ a _ { m } ( i ) = a _ { m } ( j ) ] .\tag{2}
$$

At most $5 \times 1 0 ^ { 4 }$ pairs are sampled per seed, with quotas allocated according to the number of available pairs in each predicted class. This Boolean representation records only whether two nodes are co-assigned and is invariant to center-label permutations.

The structural diference between sources m and $m ^ { \prime }$ is

$$
d _ { \mathrm { c o } } ( m , m ^ { \prime } ) = \frac { 1 } { | \mathcal { P } | } \sum _ { p \in \mathcal { P } } | C _ { m } ( p ) - C _ { m ^ { \prime } } ( p ) | .\tag{3}
$$

Structure-guided curriculum reproduction of source m produces ${ \widetilde { f } } _ { m }$ and ${ \widetilde { C } } _ { m }$ . Its reproduction stability is

$$
s _ { \mathrm { r e g } } ( m ) = 1 - \frac { 1 } { | \mathcal { P } | } \sum _ { p \in \mathcal { P } } | C _ { m } ( p ) - \widetilde { C } _ { m } ( p ) | .\tag{4}
$$

The two quantities describe between-source diferences and within-source reproducibility on the same fixed node pairs; they are not direct semantic measurements of internal network representations.

## 3.2 Why complementary support can help

Let H be the event that a candidate pseudo-label is correct, with prior $P ( H ) = \pi$ . Let $E _ { m }$ be the event that source m supports that label.

For intuition, suppose that $E _ { 1 } , \dots , E _ { K }$ are conditionally independent given H or $\neg H$ , and that every retained source provides informative support:

$$
{ \frac { P ( E _ { m } \mid H ) } { P ( E _ { m } \mid \neg H ) } } \geq \lambda > 1 .\tag{5}
$$

The posterior odds after unanimous support then satisfy

$$
{ \frac { P ( H \mid E _ { 1 : K } ) } { P ( \neg H \mid E _ { 1 : K } ) } } = { \frac { \pi } { 1 - \pi } } \prod _ { m = 1 } ^ { K } { \frac { P ( E _ { m } \mid H ) } { P ( E _ { m } \mid \neg H ) } }\tag{6}
$$

$$
\geq { \frac { \pi } { 1 - \pi } } \lambda ^ { K } .\tag{7}
$$

Equivalently,

$$
P ( H \mid E _ { 1 } , \dots , E _ { K } ) \geq { \frac { \pi \lambda ^ { K } } { 1 - \pi + \pi \lambda ^ { K } } } .\tag{8}
$$

This calculation supplies intuition, not a model of our trained sources: they share a graph and training process, and the experiments do not establish conditional independence or a common likelihood ratio. Our empirical criterion is consequently functional. A construction is useful when, under matched voting and filtering, its sources improve the precision–coverage tradeof and the quality of the labels used for training.

## 3.3 Structured construction, reproduction, and consensus

We first warm-start a GCN on labeled training nodes to obtain hidden representations H. For each $\omega = ( k , \beta )$ , we mix H with symmetrically normalized neighborhood aggregation $\widehat { A } H$ bWithin each class, k centers are fit using labeled nodes of that class together with non-training nodes predicted as that class with confidence at least 0.8. Non-training nodes use only predicted classes during center fitting and assignment. For every within-class center, contributing nontraining nodes are ordered by distance to the center. Prefixes containing 50%, 75%, and 100% of those nodes are successively combined with all labeled training nodes to train the source. The structure is then refit, and the same curriculum trains a reproduction model; diferences between the initial and reproduced co-assignment vectors determine $s _ { \mathrm { r e g } } .$ . Figure 2 separates structural partitioning, co-assignment probing, and stable nonredundant selection. The default method imposes no hard minimum occupancy; Section 4.5 reports the corresponding ablation.

Figure 2. Structural intervention, co-assignment probing, and reproduction-based selection.  
![](images/8dfeaa47188600191f113ce48007b7a8fbc8e209171e941765d5580d8eab62e2.jpg)  
Figure 2: Structural intervention, co-assignment probing, and reproduction-based selection. Varying k and $\beta$ creates diferent within-class partitions. Co-assignment matrices remove centerlabel permutations, and within-source stability $s _ { \mathrm { r e g } }$ and between-source diference $d _ { \mathrm { c o } }$ select three stable, nonduplicate sources. The illustrated nodes and matrices do not represent experimental samples.

Co-assignment relations reveal redundancy among candidates. Rather than tune a post hoc hard threshold, we retain a fixed K = 3 sources. Stability is min–max normalized over the four candidates; pairwise distances are min–max normalized over the six of-diagonal candidate pairs. A zero-range quantity is mapped to zero, and ties follow lexicographic order of $( k , \beta )$ . The most reproducible source is selected first; further sources are greedily added using normalized stability and their minimum co-assignment distance from the selected set:

$$
g ( m \mid \mathcal { M } ^ { \star } ) = \frac { 1 } { 2 } \widetilde { s } _ { \mathrm { r e g } } ( m ) + \frac { 1 } { 2 } \operatorname* { m i n } _ { j \in \mathcal { M } ^ { \star } } \widetilde { d } _ { \mathrm { c o } } ( m , j ) .\tag{9}
$$

For $i \in U$ , each retained source predicts $\hat { y } _ { m , i }$ with confidence $q _ { m , i } = \operatorname* { m a x } _ { y } p _ { m } ( y \mid i )$ . The unanimous three-source candidate set is

$$
\mathcal { V } _ { 3 / 3 } = \{ i \in U : \exists c \in \mathcal { V } , \ \hat { y } _ { m , i } = c \ \forall m \in \mathcal { M } ^ { \star } \} .\tag{10}
$$

Let $\hat { y } _ { i }$ be the common label. In the pseudo-label quality experiment, all nodes in $\nu _ { 3 / 3 }$ are evaluated before confidence ranking. For retained source m and class $c ,$ let $d _ { m , c } ( i )$ be the distance from a node to its nearest center of class c. Using labeled training nodes only, define

$$
t _ { m , c } = Q _ { 0 , 9 0 } \big ( \{ d _ { m , c } ( i ) : i \in T , \ y _ { i } = c \} \big ) , \qquad S _ { q 9 0 } = \{ i \in \mathcal { V } _ { 3 / 3 } : d _ { m , \hat { y } _ { i } } ( i ) \leq t _ { m , \hat { y } _ { i } } , \ \forall m \in \mathcal { M } ^ { \star } \} .\tag{11}
$$

Quality predictions come from source models after six ranking rounds, whereas centers and their training-node q90 thresholds are fixed at the pre-ranking reproduction stage. For DiPat and the joint six-source setting, their voting sets replace $\nu _ { 3 / 3 }$ while the same three selected structuralsource centers are used as an additional distance filter evaluated at the candidate’s voted class. This gate supplies distance-filtering information, not another class vote, and is not a native DiPat score. A maximum normalized distance can be used ofline to compute error-detection AUC, but it does not define node-level admission.

In the separate downstream task experiment, the frozen source configuration undergoes six rounds of dynamic ranking. Consensus candidates are finally ordered by conservative confidence $\begin{array} { r } { q _ { \operatorname* { m i n } } ( i ) = \operatorname* { m i n } _ { m \in \mathcal { M } ^ { \star } } q _ { m , i } } \end{array}$ , and at most ⌈0.675|U|⌉ nodes are used to train a student from scratch. The q90 gate is not used in this branch. The supervised and pseudo-label losses are

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { s u p } } + \mathcal { L } _ { \mathrm { p l } } , \quad \mathcal { L } _ { \mathrm { p l } } = - \frac { 1 } { \vert \mathcal { S } _ { u } \vert } \sum _ { ( x , \hat { y } ) \in \mathcal { S } _ { u } } \log p ( \hat { y } \mid x ) .\tag{12}
$$

```latex
Algorithm 1 Structured source construction and separated quality and task evaluation
Require: Graph, public training mask, intervention set Ω, retained-source count $\overline { { K = 3 } }$
Ensure: Voting and ${ \mathrm { q } } 9 0$ quality statistics; independent student evaluation
1: for $\omega _ { m } \in \Omega$ do
2: Train structured source $f _ { m }$ and compute node-pair co-assignment vector $C _ { m }$
3: Build a curriculum from within-class center assignments and distances; retrain $\widetilde { f } _ { m }$
4: Compute reproduction stability $s _ { \mathrm { r e g } } ( m )$
5: end for
6: Select the source with highest $s _ { \mathrm { r e g } } ,$ then greedily add sources by $g ( m \mid \mathcal { M } ^ { \star } )$ until K are
retained
7: Evaluate all non-training nodes admitted by voting and the subset retained by the training
calibrated q90 gate
8: Independently run source ranking, fix the final budget, and train a student from scratch
9: return Separated quality and task metrics
```

If training one decision source costs $T _ { f } ,$ explicitly constructing co-assignment matrices over n nodes costs $O ( M n ^ { 2 } )$ . With a fixed sample of $P$ pairs, relation probing costs $O ( M P )$ . We use four candidates and retain three throughout; increasing the number of sources is not treated as a cost-free performance guarantee.

## 4 Experimental Design

## 4.1 Data, splits, and frozen configuration

We use the Planetoid public fixed splits from PyTorch Geometric [27, 28] and run seeds 0–4 on Cora, CiteSeer, and PubMed. Table 1 reports dataset sizes verified against the data manifest. Each public split contains 20 training nodes per class. Validation and test nodes remain in the non-training candidate domain U, whose sizes are 2,568, 3,207, and 19,657. All datasets use only row-normalized original node features $X$ ; neither cora\_2025.pt nor external text or multimodal features is used. Seeds alter model initialization and training randomness but not the public masks.

Table 1: Statistics and fixed masks for the three Planetoid public datasets. Edges are directed PyG edge entries.
<table><tr><td>Dataset</td><td>Nodes</td><td>Edge entries</td><td>Features</td><td>Classes</td><td>Train/val/test</td><td> $| U |$ </td></tr><tr><td>Cora</td><td>2708</td><td>10556</td><td>1433</td><td>7</td><td>140/500/1000</td><td>2568</td></tr><tr><td>CiteSeer</td><td>3327</td><td>9104</td><td>3703</td><td>6</td><td>120/500/1000</td><td>3207</td></tr><tr><td>PubMed</td><td>19717</td><td>88648</td><td>500</td><td>3</td><td>60/500/1000</td><td>19657</td></tr></table>

The source configuration was selected during Cora development and frozen for all three datasets without tuning on CiteSeer or PubMed test results. Four structural candidates are constructed by

$$
H _ { \beta } = ( 1 - \beta ) H + \beta \widehat { A } H , \qquad ( k , \beta ) \in \{ 2 , 3 \} \times \{ 0 , 0 . 5 \} .\tag{13}
$$

Three are selected independently for each dataset and seed using reproduction stability and coassignment distance, without labels beyond the public training set. Source curricula use 50%, 75%, and 100% prefixes. Retained sources then undergo six ranking iterations of 10 epochs each; the admission ratio starts at 47.5% and rises by four percentage points per round to 67.5%. Source networks use hidden dimension 86 and dropout 0.39; students use hidden dimension 32 and dropout 0.5. The learning rate is 0.01 and weight decay is $5 \times 1 0 ^ { - 4 }$ . Final task budgets $\left\lceil 0 . 6 7 5 | U | \right\rceil$ are 1,734, 2,165, and 13,269 nodes. The 0.8 prediction-confidence condition used during source construction is distinct from both the q90 quality threshold and the student’s ranking cutof.

## 4.2 Comparisons, metrics, and use of ground truth

The main downstream controls are a supervised two-layer GCN with hidden dimension 64 and dropout 0.5, and a graph-only DiPat adaptation. The latter uses one shared marker input and a network with 3C outputs; it is not three independent models and does not include the original method’s auxiliary cat\_x input [8]. The source-replacement ablation uses three ordinary supervised GCNs with hidden dimension 86, dropout 0.39, and diferent initialization seeds. Voting, final ranking budget, and student initialization are aligned with our method, although total source-training computation is not matched to our guided, reproduction, and ranking stages. The joint six-source experiment performs only late voting after our sources and DiPat have been trained separately.

Pseudo-label precision $\mathrm { P r e c } = | \{ i \in S : \hat { y } _ { i } = y _ { i } \} | / | S |$ and coverage $\mathrm { C o v } \ : = \ : | S | / | U |$ are computed per seed and averaged over five seeds. We record counts of correct and incorrect nodes before and after q90 filtering, including the correct and incorrect pseudo-labels removed by the gate. This quality branch performs no confidence ranking and does not truncate to the task-training budget. The downstream branch reports only Test Accuracy and Macro-F1 of students trained from scratch. Training labels supervise models and calibrate q90; validation labels select checkpoints for both source models and students, even though validation nodes may also enter training with pseudo-labels; and test labels are used only for ofline evaluation after filtering and training. This overlap in validation use is a limitation of the protocol. Raw files, code versions, node-level masks, and probabilities are retained in the experiment artifacts. All three datasets use the same implementation under Python 3.11.6, PyTorch 2.12.0+cu130, PyG 2.8.0.post1, and CUDA.

## 5 Results

## 5.1 All voting candidates and the q90 structural gate

Table 2 shows the central precision–coverage tradeof. The q90 gate removes many candidates but raises conditional precision on every dataset, and stricter joint voting raises it further. Counts, precision, and coverage are arithmetic means of per-seed metrics; all filtered sets exclude labeled training nodes. For the DiPat and joint rows, q90 contributes structural information from our selected sources, so these rows evaluate external candidates under the same gate rather than native DiPat alone. In a DiPat view, an argmax outside that view’s C-class output block abstains, and a joint label must receive the stated $4 / 6 , 5 / 6 ,$ or $6 / 6$ support.

For structured $3 / 3$ sources, q90 raises conditional precision by 8.26, 12.46, and 4.99 percentage points on Cora, CiteSeer, and PubMed, while coverage falls from 96.32%, 96.08%, and 98.10% to 56.56%, 41.22%, and 71.75%. Across seeds, post-gate precision has standard deviations of 0.74, 1.35, and 0.90 percentage points. Strict $6 / 6$ consensus followed by the same gate reaches 94.19%, 89.34%, and 85.62%, the highest precision in each dataset, with lower coverage of 54.39%, 38.13%, and 68.12%. Figure 3 displays the common trend: stricter voting increases precision while reducing coverage. After the structural gate, DiPat candidates have higher precision than structured $3 / 3$ candidates on all three datasets, showing that the gate can filter other candidate sources; conversely, the joint $6 / 6$ result cannot be attributed to our three structural sources alone.

![](images/74ad6738e752cf8f50fa53b9daa47cd36a2c0f9e89b47a65af713160113b88c9.jpg)

Table 2: Pseudo-label quality before and after q90 filtering (five-seed means). Precision and coverage are percentages. Removed correct/incorrect gives the numbers removed by ${ \mathrm { q } } 9 0$ . Within each dataset, bold marks the largest post-gate precision and the largest post-gate coverage.
<table><tr><td>Dataset</td><td>Voting rule</td><td>Before N</td><td>Before prec.</td><td>After N</td><td></td><td></td><td>After prec. After cov. Removed correct/incorrect</td></tr><tr><td>Cora</td><td>Structured sources  $\mathbf { 3 / 3 }$ </td><td>2473.6</td><td>84.27</td><td>1452.4</td><td>92.53</td><td>56.56</td><td>741.0/280.2</td></tr><tr><td></td><td>DiPat 3/3 + structural gate</td><td>2437.4</td><td>83.23</td><td>1416.4</td><td>93.47</td><td>55.16</td><td>705.0/316.0</td></tr><tr><td></td><td>Joint  $4 / 6$ </td><td>2388.4</td><td>85.36</td><td>1431.8</td><td>93.38</td><td>55.76</td><td>702.2/254.4</td></tr><tr><td></td><td>Joint  $5 / 6$ </td><td>2298.8</td><td>86.87</td><td>1415.4</td><td>93.86</td><td>55.12</td><td>668.8/214.6</td></tr><tr><td></td><td>Joint  $6 / 6$ </td><td>2208.8</td><td>88.21</td><td>1396.8</td><td>94.19</td><td>54.39</td><td>632.8/179.2</td></tr><tr><td></td><td>CiteSeer Structured sources  $\mathbf { 3 / 3 }$ </td><td>3081.2</td><td>73.40</td><td>1322.0</td><td>85.86</td><td>41.22</td><td>1126.8/632.4</td></tr><tr><td></td><td>DiPat  $3 / 3$  + structural gate</td><td>2981.0</td><td>73.16</td><td>1232.8</td><td>88.87</td><td>38.44</td><td>1086.0/662.2</td></tr><tr><td></td><td>Joint  $4 / 6$ </td><td>2855.0</td><td>75.99</td><td>1247.6</td><td>88.71</td><td>38.90</td><td>1065.0/542.4</td></tr><tr><td></td><td>Joint  $5 / 6$ </td><td>2738.8</td><td>77.34</td><td>1236.2</td><td>89.02</td><td>38.55</td><td>1019.2/483.4</td></tr><tr><td></td><td>Joint  $6 / 6$ </td><td>2594.8</td><td>78.82</td><td>1222.8</td><td>89.34</td><td>38.13</td><td>952.4/419.6</td></tr><tr><td>PubMed</td><td>Structured sources  $\mathbf { 3 / 3 }$ </td><td>19283.2</td><td>79.36</td><td>14103.6</td><td>84.35</td><td>71.75</td><td>3410.8/1768.8</td></tr><tr><td></td><td>DiPat  $3 / 3$  + structural gate</td><td>18746.8</td><td>79.68</td><td>13683.2</td><td>84.85</td><td>69.61</td><td>3329.0/1734.6</td></tr><tr><td></td><td>Joint  $4 / 6$ </td><td>18535.0</td><td>80.61</td><td>13842.6</td><td>84.88</td><td>70.42</td><td>3194.6/1497.8</td></tr><tr><td></td><td>Joint  $5 / 6$ </td><td>17972.2</td><td>81.42</td><td>13637.8</td><td>85.22</td><td>69.38</td><td>3013.8/1320.6</td></tr><tr><td>Joint</td><td> $6 / 6$ </td><td>17378.0</td><td>82.25</td><td>13390.4</td><td>85.62</td><td>68.12</td><td>2830.2/1157.4</td></tr></table>

Figure 3: Precision–coverage tradeof after the common q90 structural gate (five-seed means). Lines connect adjacent voting strictness levels only to show direction and are not continuous threshold curves.

These results indicate that the training-calibrated near-center criterion reduces the risk of accepting incorrect labels. It should not be interpreted as evidence that output probabilities are calibrated. Error-detection AUC before gating is an auxiliary ranking diagnostic over continuous distance risk; node admission uses only the fixed q90 thresholds. Group counts and nodelevel votes, distances, thresholds, and pre-/post-gate masks are retained with the results for verification.

## 5.2 Students trained from scratch

The structured-source and graph-only DiPat methods in Table 3 each select at most ⌈0.675|U|⌉ pseudo-labels under the task protocol; the supervised GCN uses none. No method uses the q90 gate from Table 2 to train the student. Test Accuracy and Macro-F1 are five-seed means and should not be explained directly from post-gate precision.

Table 3: Downstream results under the shared local protocol (Accuracy/F1 in percent; five-seed mean ± sample standard deviation). Students do not use the q90 gate. Within each dataset and metric, bold marks the largest mean; ties are included.
<table><tr><td>Dataset</td><td colspan="2">Method Training pseudo-labels</td><td>Test Accuracy</td><td>Test Macro-F1</td></tr><tr><td rowspan="3">Cora</td><td>Supervised GCN</td><td>0</td><td> $8 1 . 6 6 \pm 0 . 8 2$ </td><td> $8 0 . 5 1 \pm 0 . 8 2$ </td></tr><tr><td>Graph-only DiPat adaptation</td><td>1734</td><td> $8 2 . 4 8 \pm 0 . 3 3$ </td><td> $8 1 . 3 1 \pm 0 . 4 7$ </td></tr><tr><td>Structured sources</td><td>1734</td><td> $\mathbf { 8 4 . 0 4 \pm 1 . 1 2 }$ </td><td> ${ \bf 8 3 . 0 3 \pm 0 . 8 9 }$ </td></tr><tr><td rowspan="3">CiteSeer</td><td>Supervised GCN</td><td>0</td><td> $7 1 . 4 6 \pm 0 . 6 6$ </td><td> ${ \bf 6 8 . 2 9 \pm 0 . 6 2 }$ </td></tr><tr><td>Graph-only DiPat adaptation</td><td>2165</td><td> $7 3 . 5 4 \pm 0 . 6 6$ </td><td> $6 7 . 2 6 \pm 0 . 6 1$ </td></tr><tr><td>Structured sources</td><td>2165</td><td> ${ \bf 7 3 . 7 4 \pm 0 . 6 3 }$ </td><td> ${ \bf 6 8 . 2 9 \pm 1 . 1 6 }$ </td></tr><tr><td rowspan="3">PubMed</td><td>Supervised GCN</td><td>0</td><td> $7 9 . 2 6 \pm 0 . 3 2$ </td><td> $7 8 . 9 0 \pm 0 . 2 6$ </td></tr><tr><td>Graph-only DiPat adaptation</td><td>13269</td><td> ${ \bf 8 0 . 2 6 \pm 0 . 6 7 }$ </td><td> ${ \bf 7 9 . 7 3 \pm 0 . 6 0 }$ </td></tr><tr><td>Structured sources</td><td>13269</td><td> $7 9 . 9 4 \pm 0 . 5 4$ </td><td> $7 9 . 5 3 \pm 0 . 5 3$ </td></tr></table>

Structured sources achieve the highest mean Test Accuracy and Macro-F1 on Cora. On CiteSeer, Accuracy is 0.20 points above graph-only DiPat, while supervised GCN and structured sources have the same rounded F1. On PubMed, graph-only DiPat is slightly higher on both metrics. The experiment therefore does not support universal downstream superiority across graph datasets. The graph-only DiPat implementation is an adaptation under shared inputs rather than a full reproduction of the original method, which also uses auxiliary descriptions and its own feature and training protocol [8].

## 5.3 Source-construction ablations and single-source controls

Table 4 replaces the complete construction pipeline with three ordinary supervised GCN sources of the same architecture and diferent initializations. The groups use the same 3/3 rule, q90 centers from the selected structural sources, final ranking quota, and student initialization. Ordinary GCN sources do not have within-class structural centers, so their post-gate values represent ordinary-source candidates plus our structural gate. The first two quality columns evaluate all voting candidates, whereas training pseudo-label precision is computed on the separately ranked, fixed task budget.

Table 4: Replacement ablation against three conventionally initialized GCN sources (five-seed means, percent). Both post-gate rows use the same additional q90 structural gate; task training does not use the gate. Within each dataset and metric, bold marks the larger value; q90 coverage is not ranked because it trades of against precision.
<table><tr><td>Dataset</td><td>Decision sources</td><td></td><td></td><td>All-vote prec. q90 prec./cov. Training PL prec. Test Acc.</td><td></td><td>Test Macro-F1</td></tr><tr><td rowspan="2">Cora</td><td>Structured sources</td><td>84.27</td><td>92.53/56.56</td><td>93.04</td><td>84.04</td><td>83.03</td></tr><tr><td>Three random-seed GCNs</td><td>84.80</td><td>91.94/57.06</td><td>91.30</td><td>83.30</td><td>81.95</td></tr><tr><td rowspan="2">CiteSeer</td><td>Structured sources</td><td>73.40</td><td>85.86/41.22</td><td>83.71</td><td>73.74</td><td>68.29</td></tr><tr><td>Three random-seed GCNs</td><td>71.55</td><td>84.63/42.03</td><td>79.33</td><td>73.08</td><td>69.21</td></tr><tr><td rowspan="2">PubMed</td><td>Structured sources</td><td>79.36</td><td>84.35/71.75</td><td>88.18</td><td>79.94</td><td>79.53</td></tr><tr><td>Three random-seed GCNs</td><td>79.96</td><td>84.06/71.10</td><td>87.00</td><td>79.82</td><td>79.35</td></tr></table>

Under the same q90 gate, structured-source conditional precision exceeds the random-source control by 0.59, 1.23, and 0.29 percentage points on Cora, CiteSeer, and PubMed. Computed from unrounded per-seed aggregates, fixed-budget training pseudo-label precision improves by 1.74, 4.39, and 1.19 points; subtracting the displayed two-decimal means can difer by 0.01. As

![](images/23e497ffa448b8970248c408376b9f8a476f00c71b7e5c78e544ab132f419195.jpg)

![](images/f3dbce80ebbc20734204f9e0bc71bef7219b827f67a79fab502ccfc1d6ccacc2.jpg)  
Figure 4: Pseudo-label quality for structured sources and three random-seed GCNs (five-seed means). Left: conditional precision after the shared q90 structural gate. Right: fixed-budget training pseudo-label precision without q90 filtering. The panels evaluate diferent branches and should not be compared numerically across panels.

Figure 4 shows, this training-label improvement is positive for all five seeds on every dataset. Before gating, however, the random sources are slightly more precise on Cora and PubMed. Downstream gains are also mixed: the random sources have higher CiteSeer Macro-F1, and PubMed gains are only 0.12/0.18 points in Accuracy/F1. The comparison therefore supports a consistent pseudo-label-quality gain, while the unmatched multi-stage computation prevents attributing the full diference uniquely to k or $\beta$

A direct test of voting compares predictions from one retained source under its q90 gate with unanimous 3/3 labels under that same source’s gate. Each cell in Table 5 reports the latter minus the former in post-gate precision points. Source slots indicate selection order within a seed, not fixed $( k , \beta )$ settings. All five seeds in every slot and dataset yield positive gains of 0.25–0.80 points. Three-source voting removes some nodes relative to one source. In a separate comparison, moving from one matched gate to the full three-gate configuration raises conditional precision further at an additional coverage cost.

Table 5: Post-gate precision gain of unanimous 3/3 voting over one source under the same single structural gate (percentage points; five-seed mean). Parentheses show seeds with positive gain.
<table><tr><td>Dataset</td><td>Selected slot 0</td><td>Selected slot 1</td><td>Selected slot 2</td></tr><tr><td>Cora</td><td>+0.61 (5/5)</td><td>+0.80 (5/5)</td><td>+0.42 (5/5)</td></tr><tr><td>CiteSeer</td><td>+0.66 (5/5)</td><td>+0.40 (5/5)</td><td>+0.38 (5/5)</td></tr><tr><td>PubMed</td><td>+0.34 (5/5)</td><td>+0.25 (5/5)</td><td>+0.32 (5/5)</td></tr></table>

The matched-gate test uses predictions from post-ranking source models and centers from the pre-ranking reproduction stage, consistent with the historical q90 diagnostic in Table 2. It is a selective filtering comparison rather than an equal-coverage threshold test.

## 5.4 Structure as a diference probe

Co-assignment is a statistical probe of whether selected sources impose diferent substructure criteria on the same node pairs. All datasets use the probe in Definition 1: for each seed, at most $5 \times 1 0 ^ { 4 }$ shared pairs are sampled from non-training nodes within warm-start predicted classes, and Boolean center co-assignment vectors are compared at the reproduction stage. Table 6 summarizes 15 reproduction-stability values and 15 pairwise co-assignment distances per dataset.

CiteSeer and PubMed replay only the construction stage; selected source identifiers and nodelevel nearest-center distances match the original experiment cache, and ranking iterations and students are not rerun. Cora uses the existing records under the same definition.

Table 6: Shared-pair structural probes for selected sources (five seeds; each statistic summarizes 15 values). Distances and stability lie in [0, 1].
<table><tr><td>Dataset</td><td>Mean reproduction stability</td><td>Mean co-assignment distance</td><td>Distance range</td></tr><tr><td>Cora</td><td>0.833</td><td>0.222</td><td>0.140-0.303</td></tr><tr><td>CiteSeer</td><td>0.814</td><td>0.263</td><td>0.128-0.412</td></tr><tr><td>PubMed</td><td>0.918</td><td>0.234</td><td>0.088-0.329</td></tr></table>

All three datasets show nonzero co-assignment diferences, and selected source identifiers vary across seeds. A separate distance-risk probe gives mean pairwise correlations of normalized nearcenter distance ratios of 0.799 and 0.811 for CiteSeer and PubMed, with q90-gate disagreement rates of 0.188 and 0.140 among structured 3/3 candidates. This is a diferent statistic from co-assignment distance. Because the sampled node-pair populations difer by dataset, distance magnitudes are not a direct ranking of source-construction quality; utility is assessed by the filtering and task controls in Tables 2–5.

## 5.5 Negative and boundary ablations

Adding a minimum center-occupancy anti-collapse constraint to the frozen Cora configuration reduces five-seed Test Accuracy from 84.04% to 83.44% and Macro-F1 from 83.03% to 82.61%. This constraint changes the construction trajectory but does not improve mean task performance under the tested form and strength. Its pseudo-label metrics use the student’s final ranking budget and are not mixed with the all-vote/q90 quantities in Table 2.

A compound diagnostic shufles center assignments and replaces curriculum distances with random values. Its student Accuracy is 83.38%, 73.84%, and 80.06% on Cora, CiteSeer, and PubMed, with post-q90 precision of 92.46%, 85.51%, and 84.46%. Because the diagnostic changes assignment and curriculum order together while retaining the original centers for gating, it does not isolate either component. The absence of a consistent downstream loss cautions against assigning a separate causal efect to every stage of the pipeline.

## 6 Discussion and Limitations

The most direct evidence for source construction comes from the matched functional controls. Consensus improves precision over every constituent source under the same gate, and the fixedbudget labels used for training are more precise than those from three random sources on all three datasets. Co-assignment distance explains how the selected sources difer, but the filtering comparisons establish whether that diference is useful. The benefit is concentrated in pseudolabel quality: coverage is lower, source-training computation is unmatched, and downstream results remain mixed.

The q90 diagnostic exposes the cost of selective reliability. It raises conditional precision but removes many correct labels, most sharply on CiteSeer, where coverage falls to 41.22%. Its efect on DiPat and six-source candidates also shows that the learned geometry can filter labels proposed elsewhere. We therefore interpret q90 as a selective structural diagnostic, not as probability calibration or part of student training.

The theoretical bound assumes conditional independence and a likelihood-ratio lower bound, neither of which is verified for sources sharing a graph and training process. The configuration was developed on Cora, where test results were viewed during development. CiteSeer and

PubMed therefore provide the more informative frozen-parameter checks, though they add only two citation graphs of the same type. Five seeds share the same public split and do not replace multi-split or out-of-domain validation. Validation nodes may be pseudo-labeled for training and are also used for checkpoint selection; test labels are used only for final accounting. Reported Accuracy and F1 values are descriptive means under the fixed protocol rather than significance claims.

## 7 Conclusion

We presented a pipeline that constructs consensus sources through controlled within-class interventions and selects them by reproducibility and complementarity. Across three citation graphs, the resulting sources provide more precise fixed-budget training labels than conventional random-seed sources, and their consensus is consistently more precise than matched constituent predictions. The downstream gains vary by dataset, so the evidence supports structured source construction as a way to improve pseudo-label admission—not as a universal route to the highest test accuracy.

## References

[1] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017.

[2] Paola Cascante-Bonilla, Fuwen Tan, Yanjun Qi, and Vicente Ordonez. Curriculum labeling: Revisiting pseudo-labeling for semi-supervised learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 6912–6920, 2021.

[3] Bowen Zhang, Yidong Wang, Wenxin Hou, Hao Wu, Jindong Wang, Manabu Okumura, and Takahiro Shinozaki. Flexmatch: Boosting semi-supervised learning with curriculum pseudo labeling. In Advances in Neural Information Processing Systems, volume 34, 2021.

[4] Hongbin Pei, Yuheng Xiong, Pinghui Wang, Jing Tao, Jialun Liu, Huiqi Deng, Jie Ma, and Xiaohong Guan. Memory disagreement: A pseudo-labeling measure from training dynamics for semi-supervised graph learning. In Proceedings of the ACM Web Conference 2024, pages 434–445, 2024.

[5] Islam Nassar, Munawar Hayat, Ehsan Abbasnejad, Hamid Rezatofighi, and Gholamreza Hafari. Protocon: Pseudo-label refinement via online clustering and prototypical consistency for eficient semi-supervised learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11641–11650, 2023.

[6] Ambroise Odonnat, Vasilii Feofanov, and Ievgen Redko. Leveraging ensemble diversity for robust self-training in the presence of sample selection bias. In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 595–603, 2024.

[7] Nguyen Nhat Minh To, Paul F. R. Wilson, Viet Nguyen, Mohamed Harmanani, Michael Cooper, Fahimeh Fooladgar, Purang Abolmaesumi, Parvin Mousavi, and Rahul Krishnan. Diverse prototypical ensembles improve robustness to subpopulation shift. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 59761–59783, 2025.

[8] Kai Liu and Long Wang. Diferentiated information mining: Semi-supervised graph learning with independent patterns. Mathematics, 14(2):279, 2026.

[9] Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A. Rafel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li. Fixmatch: Simplifying semisupervised learning with consistency and confidence. In Advances in Neural Information Processing Systems, volume 33, 2020.

[10] Yi Xu, Lei Shang, Jinxing Ye, Qi Qian, Yu-Feng Li, Baigui Sun, Hao Li, and Rong Jin. Dash: Semi-supervised learning with dynamic thresholding. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 11525–11536, 2021.

[11] Yidong Wang, Hao Chen, Qiang Heng, Wenxin Hou, Yue Fan, Zhen Wu, Jindong Wang, Marios Savvides, Takahiro Shinozaki, Bhiksha Raj, Bernt Schiele, and Xing Xie. Freematch: Self-adaptive thresholding for semi-supervised learning. In International Conference on Learning Representations, 2023.

[12] Hao Chen, Ran Tao, Yue Fan, Yidong Wang, Jindong Wang, Bernt Schiele, Xing Xie, Bhiksha Raj, and Marios Savvides. Softmatch: Addressing the quantity-quality trade-of in semi-supervised learning. In International Conference on Learning Representations, 2023.

[13] Hieu Pham, Zihang Dai, Qizhe Xie, and Quoc V. Le. Meta pseudo labels. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11557– 11568, 2021.

[14] Islam Nassar, Samitha Herath, Ehsan Abbasnejad, Wray Buntine, and Gholamreza Hafari. All labels are not created equal: Enhancing semi-supervision via label grouping and cotraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7241–7250, 2021.

[15] Ke Sun, Zhouchen Lin, and Zhanxing Zhu. Multi-stage self-supervised learning for graph convolutional networks on graphs with few labeled nodes. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 5892–5899, 2020.

[16] Vikas Verma, Meng Qu, Kenji Kawaguchi, Alex Lamb, Yoshua Bengio, Juho Kannala, and Jian Tang. Graphmix: Improved training of GNNs for semi-supervised learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 10024– 10032, 2021.

[17] Wenzheng Feng, Jie Zhang, Yuxiao Dong, Yu Han, Huanbo Luan, Qian Xu, Qiang Yang, Evgeny Kharlamov, and Jie Tang. Graph random neural networks for semi-supervised learning on graphs. In Advances in Neural Information Processing Systems, volume 33, 2020.

[18] Haibo Wang, Chuan Zhou, Xin Chen, Jia Wu, Shirui Pan, and Jilong Wang. Graph stochastic neural networks for semi-supervised learning. In Advances in Neural Information Processing Systems, volume 33, 2020.

[19] Qian Huang, Horace He, Abhay Singh, Ser-Nam Lim, and Austin R. Benson. Combining label propagation and simple models out-performs graph neural networks. In International Conference on Learning Representations, 2021.

[20] Enyan Dai, Charu Aggarwal, and Suhang Wang. NRGNN: Learning a label noise resistant graph neural network on sparsely and noisily labeled graphs. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 227–236, 2021.

[21] Enyan Dai, Wei Jin, Hui Liu, and Suhang Wang. Towards robust graph neural networks for noisy graphs with sparse labels. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining, pages 181–191, 2022.

[22] Yuwen Li, Miao Xiong, and Bryan Hooi. Graphcleaner: Detecting mislabelled samples in popular graph learning benchmarks. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 20195–20209, 2023.

[23] Yuhao Wu, Jiangchao Yao, Xiaobo Xia, Jun Yu, Ruxin Wang, Bo Han, and Tongliang Liu. Mitigating label noise on graphs via topological sample selection. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 53944–53972, 2024.

[24] Abdullah Alchihabi and Yuhong Guo. Learning robust graph neural networks with limited supervision. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 8723–8733, 2023.

[25] Otilia Stretcu, Krishnamurthy Viswanathan, Dana Movshovitz-Attias, Emmanouil Antonios Platanios, Sujith Ravi, and Andrew Tomkins. Graph agreement models for semisupervised learning. In Advances in Neural Information Processing Systems, volume 32, 2019.

[26] Tianyi Zhou, Shengjie Wang, and Jef A. Bilmes. Diverse ensemble evolution: Curriculum data-model marriage. In Advances in Neural Information Processing Systems, volume 31, 2018.

[27] Zhilin Yang, William W. Cohen, and Ruslan Salakhutdinov. Revisiting semi-supervised learning with graph embeddings. In Proceedings of the 33rd International Conference on Machine Learning, 2016.

[28] Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. arXiv preprint arXiv:1903.02428, 2019.