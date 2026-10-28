# WHEN SHOULD WE INTERVENE AT TEST TIME?

Jiahao Zhang<sup>∗</sup> Zhengzhou University of Aeronautics marchematics@gmail.com

Liangliang Li Xi’an University of Technology 1240311009@stu.xaut.edu.cn

Xinling Wen Zhengzhou University of Aeronautics wenxinling@zua.edu.cn

Yu Chen<sup>∗</sup> Zhengzhou University of Aeronautics chenyu@zua.edu.cn

<sup>∗</sup>Corresponding authors

## ABSTRACT

Test-time adaptation helps a deployed model respond to changing inputs, but the same intervention can rescue one batch and damage the next. Updating normalization statistics, for example, can improve corrupted images yet fail when a batch contains mostly one class. Average accuracy hides these failures and cannot tell a model when to use an intervention. Target labels can guide this decision, but they are scarce and some choices are already clear. We introduce PORTAL, a controller that chooses when to deploy an intervention, which inputs to label, and when to stop querying. Each available prediction rule is a candidate, and its predictions are fixed before current feedback arrives. Every acquired label therefore evaluates all candidates on the same input. PORTAL combines these shared comparisons and directs queries toward candidates that can still overtake the current leader. It stops when the remaining labels cannot change the choice or when the possible accuracy gain no longer justifies another query. Our analysis links the required feedback to accuracy gaps and shared mistakes, explaining why some choices need much less evidence. Experiments show that this control turns an intervention that loses accuracy overall into a useful choice on selected batches. It prevents severe failures caused by changes in arrival order and preserves gains across complete corruption streams. When a fixed rule already works well, PORTAL uses little feedback to retain its performance. Periodic label checks reveal when past evidence needs refreshing and help the controller recover as conditions change. The acquired labels also train stronger candidates for future batches, and selection brings nearly all of their accuracy gains into deployment. The same feedback therefore improves both the choice made now and the predictions available to later batches.

## 1 INTRODUCTION

A deployed model encounters changes in both image appearance and class composition. Fog may replace clear weather, or one object class may dominate a run of inputs. Test-time adaptation responds by changing the model as data arrive. Yet an update that rescues one batch can break predictions on the next. Its average gain across a test set does not tell us when to use it. Consider batch normalization (BN), which normalizes activations using feature statistics. Recomputing these statistics on incoming images often helps under corruption. Averaging predictions from an image and its horizontally flipped view offers another intervention. We call each available prediction rule a candidate. Existing adaptation methods provide many such candidates (Wang et al., 2021; Niu et al., 2022; 2023). A fixed rule chooses one for the whole stream. Local selection chooses for each batch, retaining the unchanged model when an intervention is unhelpful.

The need for selection becomes clear on PACS, which contains Photo, Art, Cartoon and Sketch domains. Combining BN updates with original and flipped views gives BN+flip. This candidate works well when domains arrive in blocks or are mixed together. Grouping inputs by class, called class-block arrivals, changes the batch statistics and reduces its accuracy to 22.58%. We introduce PORTAL, a controller that uses sparse target labels to choose among the available predictions.<sup>1</sup> It reaches 51.28% under the same arrivals. Choosing well requires evidence, but labels are scarce. Active test-time adaptation acquires labels to guide updates and selection (Gui et al., 2024; Li et al., 2024; Wang & Ding, 2025); adaptive labeling also supports evaluation under shift (Mittal et al., 2024). A close comparison may cost more feedback than its accuracy gain is worth. The controller must decide which candidate to deploy, which input to label, and when to stop. Appendix X places these decisions in the broader literature.

PORTAL uses one label to evaluate every candidate on the same input. Agreement gives candidates the same outcome; disagreement reveals which predictions are correct. We group inputs by the classes that candidates predict and estimate class probabilities within each group. Every label updates all comparisons, including the overlap between errors. Queries target rivals that can overtake the leader within the remaining budget. Selection stops when no remaining labels can change the winner or when the possible accuracy gain cannot justify another query. Our analysis relates the feedback requirement to accuracy gaps and shared errors. It also shows why biased queries can make a wrong choice more certain.

Earlier labels reduce query costs while their class relationships remain stable; periodic checks reveal when comparisons need refreshing. After choosing among fixed predictions, we use the new labels to train stronger candidates for later batches. Our experiments follow this progression from intervention failure to future learning. Holding a candidate fixed shows that targeted feedback turns its aggregate accuracy loss into a gain. A more harmful transfer shows how selection prevents most of an intervention’s damage. Complete ImageNet-C corruption streams test control at scale. CIFAR comparisons show the label savings when a strong fixed rule already works well. Finally, learned candidates raise PACS deployment accuracy from 59.68% to 72.64%. The same labels improve current deployment decisions and train stronger candidates for later batches.

## 2 SHARED FEEDBACK FOR INTERVENTION SELECTION

## 2.1 THE DEPLOYMENT DECISION

At batch t, the candidate set $\boldsymbol { A } _ { t }$ includes the unchanged rule I, called the incumbent. A candidate compared against it is a challenger. Other candidates update normalization, average image views, or use another classifier. A classifier produces class scores, or logits. Averaging these scores gives an ensemble. All candidate predictions are fixed before querying.

Split the batch into a query set $Q _ { t }$ and a separate deployment set $E _ { t }$ . Each input keeps its assigned role throughout the stream. The selector labels inputs from $Q _ { t }$ and chooses one candidate for all of $E _ { t }$ . Deployment labels are opened only for evaluation. For input x with label $y ,$ let $R _ { a } ( x , y ) =$ $\mathbf { 1 } \{ a ( x ) = y \}$ record whether candidate a is correct. We seek the highest expected accuracy on $E _ { t }$ given the available evidence. For comparison, a batch oracle uses every deployment label to choose the most accurate candidate.

## 2.2 ONE LABEL UPDATES EVERY COMPARISON

Merge candidates that agree on every current input. Each input’s prediction pattern lists the remaining candidates’ predicted classes. Inputs with the same pattern form a stratum. In stratum s, candidate a predicts $p _ { s a } ;$ the stratum’s deployment share is $w _ { s } = | E _ { t } \cap s | / | E _ { t } |$

For C classes, let $\theta _ { s } = ( \theta _ { s 1 } , \ldots , \theta _ { s C } )$ give the class probabilities in stratum s, and collect them as $\theta = ( \theta _ { s } ) _ { s }$ . We assume query and deployment inputs within a stratum share these probabilities. This is the query-to-deployment transport assumption. Candidate a then has expected accuracy $\begin{array} { r } { A _ { a } ( \theta ) = \sum _ { s } w _ { s } \theta _ { s , p _ { s a } } } \end{array}$

A Dirichlet prior describes uncertainty about each stratum’s class probabilities. The priors are independent across strata. After observing labels, let $\alpha _ { s c }$ be the class counts including the prior. Write $\begin{array} { r } { n _ { s } = \sum _ { c } \alpha _ { s c } } \end{array}$ for their total and $\pi _ { s c } = \alpha _ { s c } / n _ { s }$ for the estimated class probabilities. Given the current evidence $\mathcal { F } _ { t }$ , candidate accuracy means and covariances are

$$
\mu _ { a } = \sum _ { s } w _ { s } \pi _ { s , p _ { s a } } ,\tag{1}
$$

$$
\Sigma _ { a b } = \sum _ { s } \frac { w _ { s } ^ { 2 } } { n _ { s } + 1 } \left[ 1 \{ p _ { s a } = p _ { s b } \} \pi _ { s , p _ { s a } } - \pi _ { s , p _ { s a } } \pi _ { s , p _ { s b } } \right] .\tag{2}
$$

The covariance records which candidates succeed or fail together. The weights $w _ { s }$ account for the current mix of prediction patterns. A label therefore updates all comparisons without treating shared errors as independent. Reusing past counts also requires stable class probabilities within each stratum. Section 4.6 tests what happens when this stability or representative sampling breaks.

## 2.3 QUERY ONLY WHILE THE DECISION CAN IMPROVE

Let $j$ be the candidate with the highest estimated accuracy $\mu _ { a }$ . A designated fixed candidate breaks ties. A rival is reachable if some sequence of affordable labels can put it ahead of j. To test this, give every queried disagreement the rival’s predicted class. After $k _ { s }$ such labels in stratum s, its best possible advantage is

$$
F _ { a } ( \pmb { k } ) = \sum _ { s : p _ { s a } \neq p _ { s j } } w _ { s } \frac { \alpha _ { s , p _ { s a } } - \alpha _ { s , p _ { s j } } + k _ { s } } { n _ { s } + k _ { s } } .\tag{3}
$$

Maximize $F _ { a }$ over available query counts and remaining budget. Decreasing marginal increments make greedy integer allocation exact. If every rival is unreachable, further queries have zero expected accuracy value for this batch.

For a reachable rival, let $\mu _ { a } ^ { ( s , c ) }$ be the mean after observing class c in stratum s. A query’s expected decision value is

$$
V _ { s } = \sum _ { c } \pi _ { s c } \operatorname* { m a x } _ { a } \mu _ { a } ^ { ( s , c ) } - \operatorname* { m a x } _ { a } \mu _ { a } .\tag{4}
$$

Query the eligible stratum with the greatest $V _ { s }$ . A useful switch may need several labels before a rival overtakes the leader. If every $V _ { s }$ is zero, use $T _ { s }$ , the largest variance reduction across reachable leader–rival comparisons (Appendix S). In Figure 1, $s ^ { * }$ is the selected stratum and b is the remaining label budget. F and BF denote horizontal flip and BN+flip.

Bounding the value of further information. The expected value ofperfect information (EVPI) is the gain from learning all candidate accuracies exactly. Bounding this gain bounds the benefit of further querying. Define the leader’s gap $d _ { a } = \mu _ { j } - \mu _ { a }$ and paired posterior variance $s _ { a } ^ { 2 } =$ $\Sigma _ { a a } + \Sigma _ { j j } - \Sigma \bar { \Sigma } _ { a j }$

Theorem 1 (Posterior value and local stopping). Assume independent positive Dirichlet stratum priors, the categorical label likelihood, andfixed target weights. For a posterior-mean maximizer $j ,$ the expected accuracy improvement from any subsequent label policy is at most

$$
U = \operatorname* { m i n } \left\{ 1 - \mu _ { j } , \sum _ { a \neq j } \frac { \sqrt { d _ { a } ^ { 2 } + s _ { a } ^ { 2 } } - d _ { a } } { 2 } \right\} .\tag{5}
$$

Let $c$ be the cost of another label in expected batch-accuracy units. $H c \geq U ,$ , continuing cannot improve expected net utility. For a fixed remaining budget and available stratum counts, suppose Equation (3) is nonpositive for every feasible allocation and every rival. Those remaining labels then have zero expected accuracy value.

The bound adds the possible gains from rivals while accounting for their shared errors. Its proof appears in Appendix D. It governs the current batch under the transport assumption. Reusing past counts also requires temporal stability.

Learning after selection. A label can help beyond the current decision. After comparing the fixed predictions, we use it to train the next batch’s classifier. Uniform queries fill any unused training allowance, and each label is counted once. Paired experiments retain or undo an update to measure its effect on later predictions (Appendix F).

![](images/348b8b2659181f83e2b9d05037413d51617283ed64397c0ba9cfc3fa98485054.jpg)  
Figure 1: PORTAL compares fixed candidates through shared labels. It deploys the leader when no rival can overtake it or when the value bound falls below the label price. Otherwise, it queries to improve the decision. Snowflakes mark fixed predictions; flames mark updates to the classprobability estimates.

## 3 THE EVIDENCE NEEDED TO CHOOSE AN INTERVENTION

A larger accuracy gap makes a choice easier. Shared errors also help: labels need only distinguish the candidates where their predictions differ. We first quantify these effects for independent queries within strata, then examine a fixed query pool.

## 3.1 ACCURACY GAPS AND PAIRED DIFFERENCES

Fix the prediction map, class probabilities and positive weights $w _ { s }$ . Let ∗ be the unique best candidate after merging duplicates. For rival a, define $\Delta _ { a } = A _ { * } - A _ { a }$ and $v _ { s , a } = \mathrm { V a r } ( R _ { * } - R _ { a } \mid s )$ . An allocation q assigns query fractions $q _ { s } \geq 0$ with $\textstyle \sum _ { s } q _ { s } = 1$ . The paired variance is

$$
V _ { a } ( q ) = \sum _ { s } \frac { w _ { s } ^ { 2 } v _ { s , a } } { q _ { s } } , \qquad v _ { s , a } = 1 \{ p _ { s * } \neq p _ { s a } \} [ \theta _ { s , p _ { s * } } + \theta _ { s , p _ { s a } } - ( \theta _ { s , p _ { s * } } - \theta _ { s , p _ { s a } } ) ^ { 2 } ] .\tag{6}
$$

Only disagreements contribute to this variance. Adding a duplicate candidate creates no new comparison. A close rival, by contrast, can require many labels to distinguish.

A uniformly δ-correct procedure selects the best candidate with error probability at most $\delta \in \mathsf { \Gamma }$ $( 0 , 1 / 2 )$ for every allowed distribution. Let τ count its labels. Categorical Kullback–Leibler divergence KL and binary divergence kl measure how well observations distinguish distributions.

Theorem 2 (Shared-label information and local allocation). Let $\operatorname { A l t } ( \theta )$ be the set ofalternative class distributions λ whose best candidate differs from that under θ. Every uniformly δ-correct strategy that stops almost surely at each allowed distribution with a unique best candidate satisfies

$$
\mathbb { E } _ { \theta } \tau \geq \frac { \mathrm { k l } ( 1 - \delta , \delta ) } { \operatorname* { s u p } _ { q } \operatorname* { i n f } _ { \lambda \in \mathrm { A l t } ( \theta ) } \sum _ { s } q _ { s } \mathrm { K L } ( \theta _ { s } \| \lambda _ { s } ) } .\tag{7}
$$

For small gaps, with class probabilities and queryfractions bounded awayfrom zero, the local opti mal allocation minimizes max<sub>a</sub> $V _ { a } ( q ) / \Delta _ { a } ^ { 2 } .$ A pilot-then-confirm procedure approaches the leading label coefficient as error probability tends to zero before the gap. Appendix C gives the expansion, conditions and construction.

The proof adapts best-action identification (Garivier & Kaufmann, 2016) to shared class labels. The ratio $\dot { V _ { a } } ( q ) / \Delta _ { a } ^ { 2 }$ gives a direct reading of the result. Small gaps increase the label requirement; shared errors reduce the variance that labels must resolve.

## 3.2 FINITE BUDGETS AND BIASED QUERIES

For a fixed pair, each query favors one candidate or ties their errors. Uniform sampling without replacement gives exact selection probabilities, including ties, and hence expected deployment accuracy. Appendix E derives the sampling law; Section 4.4 tests it on real predictions.

Let $\Delta \ : > \ : 0$ be the better candidate’s deployment advantage. After m queried labels, let $\widehat { d } _ { m } \sim$ $\mathcal { N } ( \Delta + b , \sigma ^ { 2 } / m )$ estimate this advantage. Here b is query bias and $\sigma ^ { 2 } / m$ is the variance, with $\sigma > 0$ . Write Φ for the standard normal cumulative distribution. Selecting by its sign gives expected regret, or extra deployment error,

$$
\mathcal { R } _ { m } = \Delta \Phi \left( - \frac { \sqrt { m } ( \Delta + b ) } { \sigma } \right) ,\tag{8}
$$

More labels reduce regret when $\Delta + b > 0$ . When bias reverses that sign, more labels increase regret: the evidence becomes more certain about the wrong choice. Appendix E extends this calculation to multiple candidates.

With four candidates, correlated errors strengthen the query ranking by canceling shared noise (Appendix E, Figure 7). This helps representative queries and can reinforce biased choices.

## 4 EXPERIMENTS

We first expose intervention failures, then hold the candidates fixed to measure the effect of control. Full corruption streams test the scale of the effect. We next examine stopping and changing evidence, before using the same feedback to train stronger candidates. Accuracy differences are in percentage points (pp).

The main comparisons use ResNet-18 on CIFAR-10-C/100-C and a Photo-trained ResNet-50 on PACS. ImageNet-C uses a ResNet-50 updated by EATTA, an active test-time adaptation method (Wang & Ding, 2025). Natural-domain tests use contrastive language–image pretraining (CLIP), with a large vision transformer and 14-pixel patches (ViT-L/14). Its readout maps features to class scores. CIFAR-10.1 supplies newly collected images (Recht et al., 2018).

Fixed rules use the unchanged model, BN, horizontal flip, or BN+flip. We also compare their meanlogit ensemble. Empirical risk minimization (ERM) selects the candidate with the fewest queried errors, using a fixed tie rule. Its queries are uniform, prioritize class disagreements, or target symmetric KL divergence between prediction probabilities (Top-KL). Shared-feedback comparisons keep query and evaluation inputs separate. Historical paired-admission tests use queried-error comparisons. Their protocols are described below; the shared-posterior experiments use Section 2.

## 4.1 ARRIVAL ORDER CAN BREAK A STRONG INTERVENTION

On PACS, BN+flip works well when domains arrive in blocks or are mixed together. Grouping images by class changes its behavior sharply. Accuracy falls to 22.58%, below the unchanged model’s 50.88%. PORTAL chooses among the same predictions and reaches 51.28%. Local selection prevents the fixed intervention from turning a change in arrival order into a large accuracy loss.

Figure 2 also measures net batch harm: positive batch error increases over the unchanged model, summed and divided by evaluation size. New errors count previously correct predictions made wrong. These measures distinguish the damage an intervention causes from its net accuracy.

Adding candidates can help or hurt. A separate controlled experiment expands the candidate set, or portfolio. Expansion raises error by 14.32 points under a HILTTA-style controller (Li et al., 2024). Under PORTAL, it lowers error by 1.24 points. This yields a 15.56-point controller–portfolio interaction across ten seeds (Appendix R). We next fix the predictions to measure the effect of control directly.

## 4.2 CONTROL MAKES A HARMFUL CANDIDATE USEFUL

We use paired admission: deploy a challenger only when it makes fewer queried errors than the incumbent. Ties keep the incumbent. The historical PACS replay holds the incumbent, horizontalview challenger and 816 batches fixed. Its PORTAL rule targets three symmetric-KL queries per batch; admission then equals ERM on those labels.

![](images/f10946ba2e7387b5faa8da48e15541c71e9a579eeaa82f30b15b4405f954a5cb.jpg)

![](images/8a2ec4ec3ad2a3c6f9229be7928f07754c8071c4bb2e54b3948043a00304afac.jpg)

![](images/f07481e2ac12f4e1515fce2016738e6614770729a3692f510645ff4c5d59bc66.jpg)  
Figure 2: A strong fixed intervention fails when classes arrive together. On class-block PACS, BN+flip reaches 22.58% accuracy and PORTAL reaches 51.28%. Net batch harm falls from 32.41 to 0.34 percentage points. Domain, Mixed and Class denote domain blocks, mixed domains and class blocks. Results use a Photo source and batch size 64. Bars average three orders; accuracy error bars show order standard deviation (SD). The inset enlarges the selectors’ harm values.

Always deploying the challenger loses 5.80 points. Targeted feedback turns this into a 2.26-point gain (Figure 3). Random admission still loses accuracy, and uniform queries recover less of the gain. Excluding current queried inputs preserves the ordering: targeted queries gain 1.79 points and uniform queries gain 0.94. The useful intervention is unchanged; the evidence determines when to use it.

VLCS tests a more harmful transfer. Always deploying its challenger loses 11.26 points; paired admission reduces the loss to 0.35. This avoids 96.9% of the fixed intervention’s net accuracy loss (Table 1).

![](images/7045e26d06e61c697dce688fd2eb491953054bf7fbbe588ba8961d2c1aa93f44.jpg)

Control on a full corruption stream. In a sharedprefix variant, EATTA reuses the sequence of selection labels for its committed update. On 750,000 ImageNet-C images, error falls from 50.14% to 49.55%, improving all 15 corruptions. Of admitted batches, 77.3% benefit. Accepted choices stop after a mean prefix of 2.87 labels. Across all batches, the mean prefix is 3.999 labels. Trial updates access additional labels outside this count (Appendix J).

Figure 3: Accuracy gains over the incumbent on the recorded PACS trajectory, including queried inputs. PORTAL uses symmetric-KL queries and paired ERM. Random gate matches its admission rate; random queries use three labels. Oracle uses full-batch labels. Appendix R.2 gives the full protocol.  
Table 1: Gain over the incumbent (pp). Selection uses paired admission on PACS/VLCS and sequential admission on ImageNet-C.
<table><tr><td>Stream</td><td>Always</td><td>Selected</td></tr><tr><td>VLCS</td><td>-11.258</td><td>-0.353</td></tr><tr><td>PACS</td><td>-5.797</td><td>+2.258</td></tr><tr><td>ImageNet-C</td><td>+1.409</td><td>+0.590</td></tr></table>

With separate acquisition, three selection labels in addition to three adaptation labels reduce error by 1.008 points. This retains 73.37% of the fixed challenger’s gain. An independent full stream gives

0.982 points. Appendix J reports both protocols and six benchmarks’ fixed-intervention and readout comparisons.

## 4.3 STRONG FIXED RULES NEED LESS FEEDBACK

A strong fixed rule leaves less for selection to gain. On CIFAR corruptions, BN supplies most of the improvement and flipped views add to it. Fixed BN+flip reaches 79.65% on CIFAR-10-C and 54.32% on CIFAR-100-C. PORTAL reaches 79.26% and 53.87%. Compared with Top-KL selection, it keeps similar accuracy while using 91.3% and 94.6% fewer labels. Table 2 gives every corruption result.

Table 2: Severity-five CIFAR accuracy (%) on shared held-out images, averaged over three source models. Summary: mean ± standard deviation (SD) and labels per suite. Bold marks the best deployable rule; the oracle is an evaluation reference.
<table><tr><td rowspan="2">Model / rule</td><td colspan="2">Noise</td><td colspan="4"></td><td colspan="4">Weather</td><td colspan="4">Digital</td><td colspan="2">Summary</td></tr><tr><td>Gau.</td><td>Shot</td><td>Imp.</td><td>Def.</td><td>Gla.</td><td>Mot.</td><td>Zoom Snow</td><td>Frost</td><td>Fog</td><td>Bri.</td><td>Con.</td><td>Ela.</td><td>Pix.</td><td>JPEG</td><td>Avg.</td><td>Labels</td></tr><tr><td colspan="2">CIFAR-10-C / ResNet-18</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Source</td><td>25.8</td><td>32.8</td><td>25.0</td><td>56.6 51.5</td><td>67.6</td><td>65.0</td><td>77.3</td><td>66.4</td><td>73.5</td><td>91.2</td><td>33.9</td><td>76.8</td><td>45.6</td><td>73.1</td><td> $5 7 . 4 8 { \scriptstyle \pm 1 . 1 2 }$ </td><td>0</td></tr><tr><td>BN</td><td>66.2</td><td>68.5</td><td>59.5</td><td>87.2</td><td>65.0</td><td>86.1</td><td>87.3 81.2</td><td>80.1</td><td>84.7</td><td>90.6</td><td>86.9</td><td>76.6</td><td>79.0</td><td>71.4</td><td> $7 8 . 0 2 { \scriptstyle \pm 0 . 5 0 }$ </td><td>0</td></tr><tr><td>Flip</td><td>26.3</td><td>33.2</td><td>25.2</td><td>57.1</td><td>52.5</td><td>68.3</td><td>65.9</td><td>78.2 67.2</td><td>74.0</td><td>92.1</td><td>33.9</td><td>78.1</td><td>45.7</td><td>74.4</td><td> $5 8 . 1 4 \pm 1 . 0 0$ </td><td>0</td></tr><tr><td>BN+flip</td><td>68.7</td><td>71.2</td><td>62.7</td><td>88.6</td><td>67.2</td><td>86.9</td><td>88.4</td><td>83.0</td><td>81.6 85.8</td><td>91.1</td><td>87.5</td><td>78.4</td><td>80.1</td><td>73.4</td><td> $\mathbf { 7 9 . 6 5 _ { \pm 0 . 5 6 } }$ </td><td>0</td></tr><tr><td>Mean logits</td><td>58.6</td><td>64.0</td><td>54.5</td><td>83.9</td><td>64.2</td><td>84.4</td><td>85.3</td><td>83.4 80.4</td><td>84.9</td><td>93.2</td><td>83.7</td><td>80.2</td><td>72.0</td><td>75.7</td><td> $7 6 . 5 5 { \scriptstyle \pm 1 . 2 9 }$ </td><td>0</td></tr><tr><td>Uniform ERM</td><td>66.1</td><td>68.9</td><td>60.2</td><td>87.1</td><td>64.7</td><td>85.7</td><td>87.3</td><td>81.8 79.7</td><td>84.6</td><td>91.3</td><td>86.7</td><td>78.2</td><td>78.6</td><td>73.3</td><td>78.28±0.58</td><td>7,065</td></tr><tr><td>Disagree. ERM</td><td>66.6</td><td>70.0</td><td>60.5</td><td>87.7</td><td>64.5</td><td>85.9</td><td>87.5</td><td>80.9</td><td>79.6 84.0</td><td>91.4</td><td>87.1</td><td>77.8</td><td>78.8</td><td>73.1</td><td>78.36±0.59</td><td>7,065</td></tr><tr><td>Top-KL ERM</td><td>68.7</td><td>71.0</td><td>62.3</td><td>88.5</td><td>65.8</td><td>86.8</td><td>88.3</td><td>81.2 80.8</td><td>84.9</td><td>91.3</td><td>87.5</td><td>77.7</td><td>80.1</td><td>73.0</td><td> $7 9 . 2 0 { \scriptstyle \pm 0 . 5 0 }$ </td><td>7,065</td></tr><tr><td>PORTAL</td><td>68.5</td><td>70.9</td><td>62.4</td><td>88.3</td><td>66.4</td><td>86.6</td><td>87.9</td><td>81.8 81.1</td><td>84.9</td><td>91.1</td><td>87.4</td><td>78.1</td><td>80.1</td><td>73.5</td><td> $7 9 . 2 6 _ { \pm 0 . 5 1 }$ </td><td>612.3</td></tr><tr><td>Batch oracle</td><td>69.7</td><td>72.1</td><td>63.7</td><td>89.2</td><td>68.4</td><td>87.9</td><td>89.3</td><td>84.7 82.7</td><td></td><td>86.8 94.1</td><td>88.2</td><td>81.6 81.2</td><td></td><td>77.4</td><td> $8 1 . 1 4 { \scriptstyle \pm 0 . 4 5 }$ </td><td></td></tr><tr><td colspan="2">CIFAR-100-C / ResNet-18</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Source</td><td>11.0</td><td>12.4</td><td>7.7</td><td>34.2</td><td>19.8</td><td>43.2</td><td>41.3</td><td>46.5 34.2</td><td>40.1</td><td>67.2</td><td>18.0</td><td>50.1</td><td>24.3</td><td>44.7</td><td>32.98±0.41</td><td>0</td></tr><tr><td>BN</td><td>39.2</td><td>40.5</td><td>34.2</td><td>65.2</td><td>41.7</td><td>62.5</td><td>64.4</td><td>53.3 53.4</td><td>57.8</td><td>67.2</td><td>64.5</td><td>53.5</td><td>56.1</td><td>45.1</td><td>53.25±0.11</td><td>0</td></tr><tr><td>Flip</td><td>10.9</td><td>12.6</td><td>7.7</td><td>34.3</td><td>19.9</td><td>44.1</td><td>41.8</td><td>47.3 34.8</td><td>40.7</td><td>67.8</td><td>18.1</td><td>51.5</td><td>24.4</td><td>45.4</td><td>33.43±0.41</td><td>0</td></tr><tr><td>BN+flip</td><td>40.6</td><td>42.1</td><td>35.7</td><td>65.9</td><td>42.7</td><td>63.2</td><td>65.3</td><td>54.6 54.4</td><td>58.9</td><td>68.2</td><td>65.2</td><td>54.6</td><td>57.2</td><td>46.1</td><td>54.32±0.05</td><td>0</td></tr><tr><td>Mean logits</td><td>32.5</td><td>34.8</td><td>30.2</td><td>62.4</td><td>39.3</td><td>61.6</td><td>63.6</td><td>56.8 55.0</td><td>58.2</td><td>72.0</td><td>58.9</td><td>56.8</td><td>53.7</td><td>49.5</td><td>52.36±0.21</td><td>0</td></tr><tr><td>Uniform ERM</td><td>39.3</td><td>39.9</td><td>34.5</td><td>64.1</td><td>40.8</td><td>61.3</td><td>62.8</td><td>53.3</td><td>51.9 56.8</td><td>67.9</td><td>64.1</td><td>53.9</td><td>55.6</td><td>45.7</td><td>52.79±0.08</td><td>7,065</td></tr><tr><td>Disagree. ERM</td><td>39.4</td><td>39.8</td><td>34.6</td><td>64.7</td><td>40.9</td><td>61.4</td><td>63.0</td><td>53.1</td><td>52.6 56.6</td><td>67.7</td><td>63.8</td><td>53.3</td><td>55.8</td><td>45.5</td><td> $5 2 . 8 0 { \scriptstyle \pm 0 . 2 3 }$ </td><td>7,065</td></tr><tr><td>Top-KL ERM</td><td>40.1</td><td>41.5</td><td>35.2</td><td>65.9</td><td>42.0</td><td>62.8</td><td>64.8</td><td>53.6 53.8</td><td>58.1</td><td>67.7</td><td>65.2</td><td>53.3</td><td>56.8</td><td>45.4</td><td> $5 3 . 7 6 { \scriptstyle \pm 0 . 1 2 }$ </td><td>7,065</td></tr><tr><td>PORTAL</td><td>40.5</td><td>41.9</td><td>35.1</td><td>65.5</td><td>42.0</td><td>62.6</td><td>64.6</td><td>54.3 53.5</td><td>58.1</td><td>68.1</td><td>65.2</td><td>54.6</td><td>55.9</td><td>46.1</td><td> $5 3 . 8 7 \pm 0 . 2 3$ </td><td>381.7</td></tr><tr><td>Batch oracle</td><td>41.6</td><td>43.3</td><td>36.9</td><td>67.0</td><td>44.0</td><td>64.4</td><td>66.4</td><td>56.4</td><td>55.7 60.1</td><td>71.6</td><td>66.1</td><td>57.4</td><td>58.3</td><td>50.0</td><td> $5 5 . 9 5 _ { \pm 0 . 1 5 }$ </td><td></td></tr></table>

CLIP gives a clearer case for stopping early. On PACS and OfficeHome, PORTAL matches the eightview output with 1.4 and 7.4 labels per stream. The four readouts use an original image, prompt averaging, two views or eight views. For VLCS, we add StatA, a text-initialized Gaussian classifier fitted to unlabeled features (Zanella et al., 2025).

Adding StatA raises selected VLCS accuracy from 83.92% to 84.58%. The fixed row shows the gain available from this candidate (Table 18). Appendix O gives its settings.

An ensemble can also be a candidate. On PACS, selection reaches 62.07% against the mean-logit ensemble’s 61.63%, improving four of five orders. Across ten arrival conditions, cumulative-history selection beats the best fixed rule in four. Discounting old counts beats every fixed rule under class blocks (Table 66). CIFAR-10.1 needs only 1.9 queries per stream: strong fixed outputs leave little gain to resolve (Table 64).

## 4.4 SMALL LABEL BUDGETS CAN RESOLVE REAL ADVANTAGES

We compare BN+flip and mean logits on streams with disjoint image sets per corruption segment. BN+flip wins on noise; mean logits wins on brightness and JPEG. We sample uniformly from each segment’s full query pool. Selection favors fewer queried errors, with BN+flip breaking ties (Appendix W.5).

Even 75 labels leave a 24–39% probability of choosing the worse output (Figure 4). Exact probabilities agree with independent subset draws. Knowing that a better candidate exists does not make it easy to identify.

![](images/8ee3ef2ade1dfa868e03a61d6733f56a74e1147c87db335b0523d9a96352933d.jpg)

![](images/c299280097a06bcd15f40c8de3cc7443b6aa1c4c0a9ea702e11adb17b8de3d7f.jpg)  
Figure 4: Exact selection-error probabilities for brightness (left) and JPEG (right). Curves use uniform sampling without replacement; markers use 2,000 independent subset draws per setting. Means cover three checkpoints. Schedules use distinct image blocks.

## 4.5 STOPPING BEFORE HIGH CONFIDENCE IS REACHED

Uncertainty need not justify another query. A rival can remain plausible even when the remaining budget cannot make it win. On four PACS candidates, we compare PORTAL with stopping based on value, reachability, confidence or the accuracy gap. Each rule carries its own history. Unused label credits accumulate up to four times the per-batch allowance (Appendix W.2).

At three credits per batch, PORTAL uses 140.0 labels at 59.35% accuracy. Confidence stopping uses 392.8 at 59.39%. Reachability drives this saving: the remaining labels cannot change the winner. Removing the value trigger leaves the queries and actions unchanged at the primary label price.

Giving policies the same starting evidence each batch removes history differences. PORTAL and confidence stopping then match accuracy with 85.6 and 392.8 labels. Figure 5 instead follows their separate histories.

Table 3 isolates acquisition and stopping. Chang- ord inci ing only acquisition, PORTAL uses 85.6 labels versus uniform sampling’s 132.8 at similar accuracy. Five queries after a reachability stop yield no mean accuracy gai

![](images/066bacc2a75bbf93f514f045f12c31845ca4c6343fc339dc7346c80fe564f7c0.jpg)  
Figure 5: Five PACS orders. Error bars show er SD; numbers give credits per batch. Codent curves share a legend entry.

Table 3: Four-candidate PACS controls. A shares the posterior, history, budget and stopping rule. B adds labels after stopping. Harm averages positive batch accuracy losses relative to the unchanged model over evaluation inputs; pp denotes percentage points.
<table><tr><td colspan="4">A. Common-posterior acquisition</td></tr><tr><td>Acquisition</td><td>Accuracy (%)</td><td>Labels</td><td>Harm (pp)</td></tr><tr><td>Uniform</td><td> $5 9 . 3 5 \pm 0 . 1 3$ </td><td>132.8</td><td>1.78</td></tr><tr><td>Disagreement</td><td> $5 9 . 3 1 \pm 0 . 2 3$ </td><td>96.8</td><td>1.86</td></tr><tr><td>Maximum-pair KL</td><td> $5 9 . 3 3 \pm 0 . 2 6$ </td><td>118.8</td><td>1.85</td></tr><tr><td>PORTAL value</td><td> $5 9 . 3 2 \pm 0 . 3 4$ </td><td>85.6</td><td>1.86</td></tr><tr><td colspan="4">B. Labels purchased after stopping</td></tr><tr><td>Stop setting</td><td>Extra labels</td><td>Choice change</td><td>∆ accuracy (pp)</td></tr><tr><td>Primary unreachable</td><td>5.0</td><td>0.31%</td><td> $+ 0 . 0 0 0 \pm 0 . 0 5 1$ </td></tr><tr><td>EVPI stop, price 0.05</td><td>1.0</td><td>1.20%</td><td> $- 0 . 0 7 5 \pm 0 . 0 5 2$ </td></tr><tr><td>EVPI stop, price 0.05</td><td>3.0</td><td>2.21%</td><td> $- 0 . 0 1 9 \pm 0 . 1 0 0$ </td></tr><tr><td>EVPI stop, price 0.05</td><td>5.0</td><td>2.81%</td><td> $- 0 . 0 1 2 \pm 0 . 1 4 1$ </td></tr></table>

## 4.6 REFRESHING PAST EVIDENCE AS CONDITIONS CHANGE

Past labels help only while they describe the present. Figure 6 changes class probabilities within prediction patterns. Predictions and pattern frequencies stay fixed, and query and deployment labels share the new probabilities. After an abrupt reversal, cumulative history buys no labels and keeps the old choice. Late accuracy falls from 72.21% to 22.54%. Discounting old counts recovers 67.66%.

The weights $w _ { s }$ track how often a pattern occurs. They cannot detect a change in which labels that pattern

![](images/d12ea87f64dc7d4c35bfb381dc5833ee1e85f3ffc3700c034f8651ca32abe600.jpg)  
Figure 6: A change in the label law with fixed candidate predictions and stratum frequencies. Vertical lines mark the transition. Decay multiplies past counts by 0.9. The original five-seed experiment and its costs appear in Table 70.

contains. Risk monitoring and domain-shift detection address related failures (Schirmer et al., 2025;   
Park et al., 2025a). Here the candidate predictions themselves stay fixed.

We therefore reserve periodic labels to check whether old evidence still holds. These guarded selectors buy extra labels after surprising observations. We test a stream with two fixed patterns and another with 300 hidden groups, called modes. Each batch in the latter contains four modes. Prioritized queries oversample atypical examples; known query probabilities correct this sampling bias.

The comparator uses exponentially weighted (EW) class counts. Methods share candidates and label ceilings. Recovery begins at the first of five consecutive population-best choices. The target is delay at most ten batches in at least 90% of streams at each change (Appendix W.7).

Table 4: Post-change gains over our reconstructed EW selector at equal label ceilings; brackets give paired intervals. Recovery is the fraction with delay ≤ 10. Sparse rows give changes 25/50/75 under prioritized queries. The sparse 160-label variant changes both budget and settings.
<table><tr><td>Stream</td><td>Label ceiling</td><td>Post-change gain (pp)</td><td>Recovery, delay ≤ 10</td></tr><tr><td>Two-pattern abrupt</td><td>120</td><td>+9.52 [8.83, 10.21]</td><td>0.92</td></tr><tr><td>Two-pattern gradual</td><td>120</td><td>+4.87 [4.16, 5.58]</td><td>0.80</td></tr><tr><td>Two-pattern gradual, larger budget</td><td>160</td><td>+6.44 [5.92, 6.96]</td><td>0.94</td></tr><tr><td>300-mode recurrent, priority audit</td><td>100</td><td>+3.28 [0.61, 5.95]</td><td>0.685 / 0.798 / 0.642</td></tr><tr><td>300-mode recurrent, larger-budget variant</td><td>160</td><td>+8.55 [7.68, 9.41]</td><td>0.907 / 0.933 / 0.893</td></tr></table>

At the original synthetic ceiling, stable-period accuracy stays close to the strongest fixed rule: −0.122 pp with paired interval [−0.150, −0.094]. Abrupt recovery is fast; more labels improve gradual recovery (Table 4). Sparse post-change accuracy exceeds EW under both query schedules. With 100 labels, recovery is slow at the first and third recurrent changes. The 160-label variant approaches the target, still missing it at the last change.

A harmful choice selects a candidate worse than the prespecified strong fixed rule. Across sparse scenarios and schedules, the mean harmful-batch rate per seed is 3.73%, with upper confidence bound 3.91%. Recurrent changes raise the mean to 7.46%. The pooled bound describes the crossscenario average; repeated changes remain harder.

## 4.7 THE SAME LABELS CREATE STRONGER FUTURE CANDIDATES

Feedback can improve the choices available to later batches. A residual readout learns a correction to fixed class scores. Earlier labels determine its current predictions. Labels queried for the current choice are decision labels. They compare these predictions, then train the next readout. Uniform queries fill the remaining training allowance.

Deployment accuracy rises from fixed BN+flip’s 59.68% to 72.64%. The learned candidate reaches 73.03%. PORTAL therefore retains all but 0.39 points of its 13.35- point gain (Table 5).

All learning policies use 393 labels per stream. PORTAL uses 70.8 for current decisions; those labels also enter later training. At the same total budget, disagreement ERM deploys at 70.29%, while its learned candidate reaches 72.98%. Learning supplies a

Table 5: PACS accuracy (%). Selected is deployed accuracy, given as mean ± SD over five stream orders. Learned always uses each policy’s learned candidate. Decision labels form part of the 393 total labels per stream. The fixed rule uses zero labels.
<table><tr><td>Rule</td><td></td><td></td><td>Selected Learned Decision</td></tr><tr><td>Fixed BN+flip</td><td>59.68 ±0.28</td><td></td><td>0</td></tr><tr><td>Uniform ERM</td><td>65.91 ±0.49</td><td>73.03</td><td>393.0</td></tr><tr><td>Disagreement ERM</td><td>70.29 ±0.76</td><td>72.98</td><td>393.0</td></tr><tr><td>PORTAL</td><td>72.64±1.05</td><td>73.03</td><td>70.8</td></tr></table>

large gain, and selection determines how much reaches deployment. Appendix K gives all trajectories and settings.

## 4.8 RETAINING AN UPDATE CHANGES FUTURE PREDICTIONS

A useful current choice need not improve the next batch. We compare two model copies with identical future inputs, labels and updates. One retains the current update; the other undoes it. Retaining the update gains 0.23 points now but changes next-batch accuracy by −0.20 points. Undoing it leaves future predictions unchanged. Appendix F relates these effects to the update dynamics and bounds the later loss change. Learning a candidate and deciding when to deploy it remain distinct decisions.

## 5 CONCLUSION

An intervention’s value depends on the batch it receives. A rule that helps under ordinary arrivals can fail when classes arrive together. PORTAL uses shared feedback to choose when to deploy each candidate and stops buying labels when they cannot improve that choice. Holding the challenger fixed shows how targeted feedback turns an aggregate loss into useful local gains. Periodic checks keep the evidence current. The same labels train stronger future candidates, and selection brings their gains to deployment.

## AI USE STATEMENT

ChatGPT was used only for literature search. The authors are responsible for the final content.

## REPRODUCIBILITY STATEMENT

The appendix contains the model settings, query/evaluation splits, update timing, query budgets, and statistical procedures. It also provides complete corruption and domain results, comparisons of method components, and proofs. The changing-label experiments include their selector definitions, reconstructed comparison methods, and independent sampling units. The manuscript, implementation, released result data, and available frozen prediction banks are provided in our anonymous repository: https://anonymous.4open.science/r/PORTAL-8322/. Its data inventory lists the external datasets and checkpoints needed for further reproduction.

## REFERENCES

Yarin Bar, Shalev Shaer, and Yaniv Romano. Protected test-time adaptation via online entropy matching: A betting approach. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper\_files/paper/2024/hash/ 9b35a0a20d617dc68ae98a7a57df2f51-Abstract-Conference.html.

Chentao Cao, Zhun Zhong, Zhanke Zhou, Tongliang Liu, Yang Liu, Kun Zhang, and Bo Han. Noisy test-time adaptation in vision-language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=iylpeTI0Ql.

Hyeongheon Cha, Dong Min Kim, Hye Won Chung, Taesik Gong, and Sung-Ju Lee. SNAP: low-latency test-time adaptation with sparse updates. In Danielle Belgrave, Cheng Zhang, Laura N. Montoya, Hsuan-Tien Lin, Razvan Pascanu, Piotr Koniusz, Marzyeh Ghassemi, Nancy Chen, Ivan Vladimir Meza Ru ´ ´ız, and Arturo Loaiza-Bonilla (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025. URL http://papers.nips.cc/paper\_files/paper/2025/hash/ 820e42f39773c6cbbd875553db45658f-Abstract-Conference.html.

Mingcai Chen, Baoming Zhang, Zongbo Han, Wenyu Jiang, Yanmeng Wang, Shuai Feng, Yuntao Du, and Bingkun Bao. Test-time selective adaptation for uni-modal distribution shift in multi-modal data. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025. URL https://proceedings.mlr.press/v267/chen25ch.html.

Xiao Chen, Jiazhen Huang, Zhiming Liu, Qinting Jiang, Fanding Huang, Jingyan Jiang, and Zhi Wang. Test-time distillation for continual model adaptation. arXiv preprint arXiv:2506.02671, 2026. URL https://arxiv.org/abs/2506.02671.

Yaofo Chen, Shuaicheng Niu, Yaowei Wang, Shoukai Xu, Hengjie Song, and Mingkui Tan. Towards robust and efficient cloud-edge elastic model adaptation via selective entropy distillation. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id= vePdNU3u6n.

Qiyuan Dai and Sibei Yang. Free on the fly: Enhancing flexibility in test-time adaptation with online EM. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp. 9538–9548. Computer Vision Foundation / IEEE,

2025. doi: 10.1109/CVPR52734.2025.00891. URL https://openaccess.thecvf.com/ content/CVPR2025/html/Dai\_Free\_on\_the\_Fly\_Enhancing\_Flexibility\_ in\_Test-Time\_Adaptation\_with\_CVPR\_2025\_paper.html.

Dexin Duan, Rui Xu, Peilin Liu, and Fei Wen. Lifelong test-time adaptation via online learning in tracked low-dimensional subspace. In Danielle Belgrave, Cheng Zhang, Laura N. Montoya, Hsuan-Tien Lin, Razvan Pascanu, Piotr Koniusz, Marzyeh Ghassemi, Nancy Chen, Ivan´ Vladimir Meza Ru´ız, and Arturo Loaiza-Bonilla (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 /Mexico City, Mexico, November 30 - December 5, 2025, 2025. URL http://papers.nips.cc/paper\_files/paper/2025/hash/ 1b6f528fe522a956dce80d5b98be8cd5-Abstract-Conference.html.

Xinqi Fan, Xueli Chen, Luoxiao Yang, Chuin Hong Yap, Rizwan Qureshi, Qi Dou, Moi Hoon Yap, and Mubarak Shah. Test-time retrieval-augmented adaptation for vision-language models. In IEEE/CVF International Conference on Computer Vision, ICCV 2025, Honolulu, HI, USA, October 19-25, 2025, pp. 8810–8819. IEEE, 2025. doi: 10.1109/ICCV51701.2025.00824. URL https://doi.org/10.1109/ICCV51701.2025.00824.

Matteo Farina, Gianni Franchi, Giovanni Iacca, Massimiliano Mancini, and Elisa Ricci. Frustratingly easy test-time adaptation of vision-language models. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper\_files/paper/2024/hash/ e92cb6f981a2cacb2a710ecaa0d7b141-Abstract-Conference.html.

Yiwei Fu, Hui Wan, Xiao Luo, and Minghua Deng. STAR: test-time adaptation can enhance universal prompt learning for vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2026, Denver, CO, USA, June 3-7, 2026, pp. 31482–31492. Computer Vision Foundation, 2026. URL https://openaccess.thecvf.com/content/CVPR2026/html/Fu\_STAR\_ Test-Time\_Adaptation\_Can\_Enhance\_Universal\_Prompt\_Learning\_for\_ Vision-Language\_CVPR\_2026\_paper.html.

Zhengqing Gao, Xu-Yao Zhang, and Cheng-Lin Liu. Unified entropy optimization for openset test-time adaptation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 23975–23984. IEEE, 2024. doi: 10.1109/CVPR52733.2024.02263. URL https://doi.org/10.1109/CVPR52733. 2024.02263.

Aurelien Garivier and Emilie Kaufmann. Optimal best arm identification with fixed confidence.´ In Proceedings of the 29th Annual Conference on Learning Theory, volume 49 of Proceedings of Machine Learning Research, pp. 998–1027, 2016. URL https://proceedings.mlr. press/v49/garivier16a.html.

Shurui Gui, Xiner Li, and Shuiwang Ji. Active test-time adaptation: Theoretical analyses and an algorithm. In International Conference on Learning Representations, 2024. URL https:// arxiv.org/abs/2404.05094.

Jiayi Guo, Junhao Zhao, Chaoqun Du, Yulin Wang, Chunjiang Ge, Zanlin Ni, Shiji Song, Humphrey Shi, and Gao Huang. Everything to the synthetic: Diffusion-driven test-time adaptation via synthetic-domain alignment. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp. 30503– 30513. Computer Vision Foundation / IEEE, 2025. doi: 10.1109/CVPR52734.2025.02840. URL https://openaccess.thecvf.com/content/CVPR2025/html/ Guo\_Everything\_to\_the\_Synthetic\_Diffusion-driven\_Test-time\_ Adaptation\_via\_Synthetic-Domain\_Alignment\_CVPR\_2025\_paper.html.

Jisu Han, Jaemin Na, and Wonjun Hwang. Ranked entropy minimization for continual testtime adaptation. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Forty-second International

Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025. URL https://proceedings.mlr.press/v267/han25e.html.

Fanding Huang, Jingyan Jiang, Qinting Jiang, Hebei Li, Faisal Nadeem Khan, and Zhi Wang. COSMIC: clique-oriented semantic multi-space integration for robust CLIP testtime adaptation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp. 9772–9781. Computer Vision Foundation / IEEE, 2025a. doi: 10.1109/CVPR52734.2025.00913. URL https://openaccess.thecvf.com/content/CVPR2025/html/Huang\_ COSMIC\_Clique-Oriented\_Semantic\_Multi-space\_Integration\_for\_ Robust\_CLIP\_Test-Time\_Adaptation\_CVPR\_2025\_paper.html.

Zhaohong Huang, Yuxin Zhang, Jingjing Xie, Fei Chao, and Rongrong Ji. Gs-bias: Global-spatial bias learner for single-image test-time adaptation of vision-language models. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025b. URL https://proceedings.mlr. press/v267/huang25aw.html.

Zhaohong Huang, Yuxin Zhang, Wenjing Liu, Fei Chao, and Rongrong Ji. Prototype-based test-time adaptation of vision-language models. arXiv preprint arXiv:2604.21360, 2026.

Hyeonseo Jang, Jaebyeong Jeon, Joong-Won Hwang, and Kibok Lee. Improving calibration in testtime prompt tuning for vision-language models via data-free flatness-aware prompt pretraining. CoRR, abs/2604.27715, 2026. doi: 10.48550/ARXIV.2604.27715. URL https://doi.org/ 10.48550/arXiv.2604.27715.

Juwon Kang, Nayeong Kim, Jungseul Ok, and Suha Kwak. Membn: Robust test-time adaptation via batch norm with statistics memory. In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gul Varol (eds.),¨ Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XXVIII, volume 15086 of Lecture Notes in Computer Science, pp. 467–483. Springer, 2024. doi: 10.1007/ 978-3-031-73390-1\ 27. URL https://doi.org/10.1007/978-3-031-73390-1\_ 27.

Adilbek Karmanov, Dayan Guan, Shijian Lu, Abdulmotaleb El Saddik, and Eric Xing. Efficient test-time adaptation of vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14162–14171, 2024. doi: 10.1109/CVPR52733.2024.01343. URL https://openaccess.thecvf.com/ content/CVPR2024/html/Karmanov\_Efficient\_Test-Time\_Adaptation\_ of\_Vision-Language\_Models\_CVPR\_2024\_paper.html.

Eungyeup Kim, Mingjie Sun, Christina Baek, Aditi Raghunathan, and J. Zico Kolter. Test-time adaptation induces stronger accuracy and agreement-on-the-line. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper\_files/paper/2024/hash/ d96fcc07d623a9eba68616629911143a-Abstract-Conference.html.

Hyeongyu Kim, Geonhui Han, and Dosik Hwang. Actta: Rethinking test-time adaptation via dynamic activation. CoRR, abs/2603.26096, 2026. doi: 10.48550/ARXIV.2603.26096. URL https://doi.org/10.48550/arXiv.2603.26096.

Sunoh Kim and Daeho Um. Ss-tpt: Stability and suitability-guided test-time prompt tuning for adversarially robust vision-language models. arXiv preprint arXiv:2606.06943, 2026.

Marc Lafon, Gustavo Vargas Hakim, Clement Rambour, Christian Desrosiers, and Nicolas Thome.´ Cliptta: Robust contrastive vision-language test-time adaptation. In Advances in Neural Information Processing Systems, volume 38, pp. 71849–71883, 2025. doi: 10.52202/085713-2412.

Daeun Lee, Jaehong Yoon, and Sung Ju Hwang. Becotta: Input-dependent online blending of experts for continual test-time adaptation. In Ruslan Salakhutdinov, Zico Kolter, Katherine A. Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, volume 235 of Proceedings ofMachine Learning Research, pp. 27072–27093. PMLR / OpenReview.net, 2024a. URL https://proceedings.mlr.press/v235/lee24ab.html.

Jonghyun Lee, Dahuin Jung, Saehyung Lee, Junsung Park, Juhyeon Shin, Uiwon Hwang, and Sungroh Yoon. Entropy is not enough for test-time adaptation: From the perspective of disentangled factors. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024b. URL https://openreview.net/ forum?id=9w3iw8wDuE.

Taeckyung Lee, Sorn Chottananurak, Taesik Gong, and Sung-Ju Lee. AETTA: label-free accuracy estimation for test-time adaptation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 28643–28652. IEEE, 2024c. doi: 10.1109/CVPR52733.2024.02706. URL https://doi.org/10.1109/CVPR52733. 2024.02706.

Taeckyung Lee, Sorn Chottananurak, Junsu Kim, Jinwoo Shin, Taesik Gong, and Sung-Ju Lee. Testtime adaptation with binary feedback. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13- 19, 2025, volume 267 of Proceedings ofMachine Learning Research. PMLR / OpenReview.net, 2025a. URL https://proceedings.mlr.press/v267/lee25g.html.

Youngjun Lee, Doyoung Kim, Junhyeok Kang, Jihwan Bang, Hwanjun Song, and Jae-Gil Lee. RA-TTA: retrieval-augmented test-time adaptation for vision-language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025b. URL https://openreview.net/forum?id=V3zobHnS61.

Aodi Li, Liansheng Zhuang, Xiao Long, Houqiang Li, and Shafei Wang. Flatness guided test-time adaptation for vision-language models. arXiv preprint arXiv:2501.18864, 2025a.

Haobin Li, Peng Hu, Qianjun Zhang, Xi Peng, XitingLiu, and Mouxing Yang. Test-time adaptation for cross-modal retrieval with query shift. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025b. URL https://openreview.net/forum?id=BmG88rONaU.

Yushu Li, Yongyi Su, Xulei Yang, Kui Jia, and Xun Xu. Exploring human-in-the-loop test-time adaptation by synergizing active learning and model selection. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id=P09rAv8UH7.

Yushu Li, Yongyi Su, Adam Goodge, Kui Jia, and Xun Xu. Efficient and context-aware label propagation for zero-/few-shot training-free adaptation of vision-language model. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025c. URL https://openreview.net/forum?id=D10yarGQNk.

Xuefeng Liu, Fangfang Xia, Rick L. Stevens, and Yuxin Chen. Contextual active model selection. In Advances in Neural Information Processing Systems, volume 37, pp. 49914–49956, 2024. doi: 10.52202/079017-1581. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 5966e7c5c017d8449d215faeb1db4173-Paper-Conference.pdf.

Yihong Luo, Wenwu He, Dong Liang, Yihang Zhou, and Zhuo-Xu Cui. D <sup>2</sup> o: A dual debiasing operator for training-free test-time adaptation of vision–language models. In Forty-third International Conference on Machine Learning.

Jing Ma. Improved self-training for test-time adaptation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 23701– 23710. IEEE, 2024. doi: 10.1109/CVPR52733.2024.02237. URL https://doi.org/10. 1109/CVPR52733.2024.02237.

Ke Ma, Jiaqi Tang, Bin Guo, Fan Dang, Sicong Liu, Zhui Zhu, Lei Wu, Cheng Fang, Ying-Cong Chen, Zhiwen Yu, and Yunhao Liu. SURGEON: memory-adaptive fully test-time adaptation via dynamic activation sparsity. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp. 30514– 30523. Computer Vision Foundation / IEEE, 2025. doi: 10.1109/CVPR52734.2025.02841. URL https://openaccess.thecvf.com/content/CVPR2025/html/Ma\_ SURGEON\_Memory-Adaptive\_Fully\_Test-Time\_Adaptation\_via\_Dynamic\_ Activation\_Sparsity\_CVPR\_2025\_paper.html.

Xiao Ma, Young D Kwon, Pan Zhou, and Dong Ma. Architecture-agnostic test-time adaptation via backprop-free embedding alignment. In International Conference on Learning Representations, volume 2026, pp. 127222–127244, 2026.

Sarthak Kumar Maharana, Baoming Zhang, Leonid Karlinsky, Rogerio Feris, and Yunhui Guo. ´ BATCLIP: bimodal online test-time adaptation for CLIP. In IEEE/CVF International Conference on Computer Vision, ICCV 2025, Honolulu, HI, USA, October 19-25, 2025, pp. 1569–1579. IEEE, 2025. doi: 10.1109/ICCV51701.2025.00154. URL https://doi.org/10.1109/ ICCV51701.2025.00154.

Yu Mitsuzumi, Akisato Kimura, Yasuhiro Fujiwara, and Hisashi Kashima. Cross-device collaborative test-time adaptation with zeroth-order optimization and model merging. CoRR, abs/2607.02988, 2026. doi: 10.48550/ARXIV.2607.02988. URL https://doi.org/10. 48550/arXiv.2607.02988.

Daksh Mittal, Yuanzhe Ma, Shalmali Joshi, and Hongseok Namkoong. Adaptive labeling for efficient out-of-distribution model evaluation. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper\_files/paper/2024/hash/ 830c927e4e9c9a457d476c145514fb56-Abstract-Conference.html.

Chenggong Ni, Fan Lyu, Jiayao Tan, Fuyuan Hu, Rui Yao, and Tao Zhou. Maintaining consistent inter-class topology in continual test-time adaptation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp. 15319–15328. Computer Vision Foundation / IEEE, 2025. doi: 10.1109/CVPR52734.2025.01427. URL https://openaccess.thecvf.com/ content/CVPR2025/html/Ni\_Maintaining\_Consistent\_Inter-Class\_ Topology\_in\_Continual\_Test-Time\_Adaptation\_CVPR\_2025\_paper.html.

Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, and Mingkui Tan. Efficient test-time model adaptation without forgetting. In Proceedings of the 39th International Conference on Machine Learning, volume 162, pp. 16888–16905, 2022. URL https://proceedings.mlr.press/v162/niu22a.html.

Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, and Mingkui Tan. Towards stable test-time adaptation in dynamic wild world. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= g2YraF75Tj.

Shuaicheng Niu, Chunyan Miao, Guohao Chen, Pengcheng Wu, and Peilin Zhao. Test-time model adaptation with only forward passes. In Ruslan Salakhutdinov, Zico Kolter, Katherine A. Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, volume 235 of Proceedings ofMachine Learning Research, pp. 38298–38315. PMLR / OpenReview.net, 2024. URL https://proceedings.mlr.press/v235/niu24a.html.

Shuaicheng Niu, Guohao Chen, Deyu Chen, Yifan Zhang, Jiaxiang Wu, Zhiquan Wen, Yaofo Chen, Peilin Zhao, Chunyan Miao, and Mingkui Tan. Adapt in the wild: Test-time entropy minimization with sharpness and feature regularization. CoRR, abs/2509.04977, 2025. doi: 10.48550/ARXIV. 2509.04977. URL https://doi.org/10.48550/arXiv.2509.04977.

Yeongtak Oh, Jonghyun Lee, Jooyoung Choi, Dahuin Jung, Uiwon Hwang, and Sungroh Yoon. Efficient diffusion-driven corruption editor for test-time adaptation. In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gul Varol (eds.),¨ Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LII, volume 15110 of Lecture Notes in Computer Science, pp. 184–201. Springer, 2024. doi: 10.1007/978-3-031-72943-0\ 11. URL https://doi.org/10.1007/ 978-3-031-72943-0\_11.

David Osowiechi, Mehrdad Noori, Gustavo A. Vargas Hakim, Moslem Yazdanpanah, Ali Bahri, Milad Cheraghalikhani, Sahar Dastani, Farzad Beizaee, Ismail Ben Ayed, and Christian Desrosiers. Watt: Weight average test-time adaptation of clip. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/ 079017-1522. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/55d16334943f8728073e17139e5baa3d-Paper-Conference.pdf.

Hyejin Park, Jeongyeon Hwang, Sunung Mun, Sangdon Park, and Jungseul Ok. Medbn: Robust test-time adaptation against malicious test samples. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 5997–6007. IEEE, 2024. doi: 10.1109/CVPR52733.2024.00573. URL https://doi.org/10.1109/ CVPR52733.2024.00573.

Hyewon Park, Hyejin Park, Jueun Ko, and Dongbo Min. Hybrid-tta: Continual test-time adaptation via dynamic domain shift detection. In IEEE/CVF International Conference on Computer Vision, ICCV 2025, Honolulu, HI, USA, October 19-25, 2025, pp. 2877–2886. IEEE, 2025a. doi: 10. 1109/ICCV51701.2025.00276. URL https://doi.org/10.1109/ICCV51701.2025. 00276.

Mincheol Park, Heeji Won, Won Woo Ro, and Suhyun Kim. Rethinking entropy in test-time adaptation: The missing piece from energy duality. In Danielle Belgrave, Cheng Zhang, Laura N. Montoya, Hsuan-Tien Lin, Razvan Pascanu, Piotr Koniusz, Marzyeh Ghassemi, Nancy Chen, Ivan´ Vladimir Meza Ru´ız, and Arturo Loaiza-Bonilla (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 /Mexico City, Mexico, November 30 - December 5, 2025, 2025b. URL http://papers.nips.cc/paper\_files/paper/2025/hash/ 435422305988b73c6cc00bcb29ba2531-Abstract-Conference.html.

Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do CIFAR-10 classifiers generalize to CIFAR-10? arXiv preprint arXiv:1806.00451, 2018. URL https: //arxiv.org/abs/1806.00451.

Mona Schirmer, Metod Jazbec, Christian Andersson Naesseth, and Eric T. Nalisnick. Monitoring risks in test-time adaptation. In Danielle Belgrave, Cheng Zhang, Laura N. Montoya, Hsuan-Tien Lin, Razvan Pascanu, Piotr Koniusz, Marzyeh Ghassemi, Nancy Chen, Ivan´ Vladimir Meza Ru´ız, and Arturo Loaiza-Bonilla (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 /Mexico City, Mexico, November 30 - December 5, 2025, 2025. URL http://papers.nips.cc/paper\_files/paper/2025/hash/ 746960ad49ddb47248970a0e1404230c-Abstract-Conference.html.

Manli Shu, Weili Nie, De-An Huang, Zhiding Yu, Tom Goldstein, Anima Anandkumar, and Chaowei Xiao. Test-time prompt tuning for zero-shot generalization in vision-language models. In Advances in Neural Information Processing Systems, volume 35, pp. 14274–14289, 2022. doi: 10.52202/068431-1038. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/ 5bf2b802e24106064dc547ae9283bb0c-Paper-Conference.pdf.

Sudarshan Sreeram, Young D Kwon, and Cecilia Mascolo. Tempora: Characterising the timecontingent utility of online test-time adaptation. arXiv preprint arXiv:2602.06136, 2026.

Jingwei Sun, Jianing Zhu, Jiangchao Yao, Gang Niu, Masashi Sugiyama, and Bo Han. Bilateral information-aware test-time adaptation for vision-language models. In The Fourteenth International Conference on Learning Representations, 2026.

Mingkui Tan, Guohao Chen, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Peilin Zhao, and Shuaicheng Niu. Uncertainty-calibrated test-time model adaptation without forgetting. IEEE Trans. Pattern Anal. Mach. Intell., 47(8):6274–6289, 2025. doi: 10.1109/TPAMI.2025.3560696. URL https: //doi.org/10.1109/TPAMI.2025.3560696.

Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=uXl3bZLkr3.

Guowei Wang and Changxing Ding. Effortless active labeling for long-term test-time adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 25633–25642, 2025. doi: 10.1109/CVPR52734.2025.02387. URL https://openaccess.thecvf.com/content/CVPR2025/html/Wang\_ Effortless\_Active\_Labeling\_for\_Long-Term\_Test-Time\_Adaptation\_ CVPR\_2025\_paper.html.

Xingyu Wang and Tao Wang. FOZO: forward-only zeroth-order prompt optimization for test-time adaptation. CoRR, abs/2603.04733, 2026. doi: 10.48550/ARXIV.2603.04733. URL https: //doi.org/10.48550/arXiv.2603.04733.

Xucong Wang, Zhe Zhao, Zekun Wang, Xiaofeng Cao, Xu Wang, Di Wu, Pengkun Wang, and Yang Wang. Long-tailed test-time adaptation for vision-language models. In International Conference on Learning Representations, volume 2026, pp. 46492–46530, 2026.

Zixin Wang, Dong Gong, Sen Wang, Zi Huang, and Yadan Luo. Is less more? exploring token condensation as training-free test-time adaptation. In IEEE/CVF International Conference on Com puter Vision, ICCV 2025, Honolulu, HI, USA, October 19-25, 2025, pp. 144–154. IEEE, 2025. doi: 10.1109/ICCV51701.2025.00021. URL https://doi.org/10.1109/ICCV51701. 2025.00021.

Haoyan Wu, Yahao Liu, Yinjie Lei, Lixin Duan, and Wen Li. Dynamic logits adjustment and exploration for test-time adaptation in vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3143–3153, 2026a.

Xiangyu Wu, Feng Yu, Yang Yang, Qing-Guo Chen, and Jianfeng Lu. Multi-label test-time adaptation with bound entropy minimization. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=75PhjtbBdr.

Xiangyu Wu, Dongming Jiang, Feng Yu, Yueying Tian, Jiaqi Tang, Qing-Guo Chen, Yang Yang, and Jianfeng Lu. Adaptive debiasing tsallis entropy for test-time adaptation. CoRR, abs/2602.11743, 2026b. doi: 10.48550/ARXIV.2602.11743. URL https://doi.org/10.48550/arXiv. 2602.11743.

Zehao Xiao, Shilin Yan, Jack Hong, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiayi Shen, Qi Wang, and Cees G. M. Snoek. Dynaprompt: Dynamic test-time prompt tuning. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=EFZEdHB3Mp.

Yingkai Yang, Chaoqi Chen, and Hui Huang. Back to source: Open-set continual test-time adaptation via domain compensation. CoRR, abs/2604.21772, 2026. doi: 10.48550/ARXIV.2604.21772. URL https://doi.org/10.48550/arXiv.2604.21772.

Yige Yuan, Bingbing Xu, Liang Hou, Fei Sun, Huawei Shen, and Xueqi Cheng. TEA: testtime energy adaptation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 23901–23911. IEEE, 2024. doi: 10.1109/CVPR52733.2024.02256. URL https://doi.org/10.1109/CVPR52733. 2024.02256.

Maxime Zanella and Ismail Ben Ayed. On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning? In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 23783–23793. IEEE, 2024. doi: 10.1109/CVPR52733.2024.02245. URL https://doi.org/10.1109/ CVPR52733.2024.02245.

Maxime Zanella, Clement Fuchs, Christophe De Vleeschouwer, and Ismail Ben Ayed. Realistic´ test-time adaptation of vision-language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. doi: 10.1109/CVPR52734.2025.02337.

Chenyu Zhang, Kunlun Xu, Zichen Liu, Yuxin Peng, and Jiahuan Zhou. SCAP: transductive test-time adaptation via supportive clique-based attribute prompting. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp. 30032–30041. Computer Vision Foundation / IEEE, 2025a. doi: 10.1109/CVPR52734. 2025.02795. URL https://openaccess.thecvf.com/content/CVPR2025/html/ Zhang\_SCAP\_Transductive\_Test-Time\_Adaptation\_via\_Supportive\_ Clique-based\_Attribute\_Prompting\_CVPR\_2025\_paper.html.

Qiang Zhang, Mengsheng Zhao, Jiawei Liu, Fanrui Zhang, Yongchao Xu, and Zheng-Jun Zha. Hierarchical knowledge prompt tuning for multi-task test-time adaptation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp. 30524–30533. Computer Vision Foundation / IEEE, 2025b. doi: 10.1109/CVPR52734.2025.02842. URL https://openaccess.thecvf. com/content/CVPR2025/html/Zhang\_Hierarchical\_Knowledge\_Prompt\_ Tuning\_for\_Multi-task\_Test-Time\_Adaptation\_CVPR\_2025\_paper.html.

Taolin Zhang, Jinpeng Wang, Hang Guo, Tao Dai, Bin Chen, and Shu-Tao Xia. Boostadapter: Improving vision-language test-time adaptation via regional bootstrapping. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024a. URL http://papers.nips.cc/paper\_files/paper/2024/ hash/7d60bfd8458b67acbbaf18b892338d00-Abstract-Conference.html.

Yuchao Zhang, Hao Wang, Fan Zhang, Qirui Mi, Mengyue Yang, Yisen Wang, Jun Wang, Haoxuan Li, and Zhouchen Lin. Learning to memorize with attributive and associative memory for online test-time adaptation of vision-language models. In Forty-third International Conference on Machine Learning.

Yunbei Zhang, Akshay Mehra, Shuaicheng Niu, and Jihun Hamm. Dpcore: Dynamic prompt coreset for continual test-time adaptation. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13- 19, 2025, volume 267 of Proceedings ofMachine Learning Research. PMLR / OpenReview.net, 2025c. URL https://proceedings.mlr.press/v267/zhang25bf.html.

Zhen-Yu Zhang, Zhiyu Xie, Huaxiu Yao, and Masashi Sugiyama. Test-time adaptation in nonstationary environments via adaptive representation alignment. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024b. URL http://papers.nips.cc/paper\_files/paper/2024/ hash/abe31a12e83111fdf2cfd54deed5a2ce-Abstract-Conference.html.

Lihua Zhou, Mao Ye, Shuaifeng Li, Nianxin Li, Xiatian Zhu, Lei Deng, Hongbin Liu, and Zhen Lei. Bayesian test-time adaptation for vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp. 29999– 30009. Computer Vision Foundation / IEEE, 2025. doi: 10.1109/CVPR52734.2025.02792. URL https://openaccess.thecvf.com/content/CVPR2025/html/Zhou\_ Bayesian\_Test-Time\_Adaptation\_for\_Vision-Language\_Models\_CVPR\_ 2025\_paper.html.

Zhi Zhou, Ming Yang, Shi-Yu Tian, Kun-Yang Yu, Lan-Zhe Guo, and Yu-Feng Li. On the learnability of test-time adaptation: A recovery complexity perspective. arXiv preprint arXiv:2605.28057, 2026.

Xingyu Zhu, Huanshen Wu, Shuo Wang, Beier Zhu, Jiannan Ge, Jiaheng Zhang, and Long Chen. Robustifying vision-language models via test-time prompt adaptation. arXiv preprint arXiv:2607.09450, 2026.

Zhilin Zhu, Xiaopeng Hong, Zhiheng Ma, Weijun Zhuang, Yaohui Ma, Yong Dai, and Yaowei Wang. Reshaping the online data buffering and organizing mechanism for continual test-time adaptation. In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gul Varol¨ (eds.), Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29- October 4, 2024, Proceedings, Part LXXXII, volume 15140 of Lecture Notes in Computer Science, pp. 415–433. Springer, 2024. doi: 10.1007/978-3-031-73007-8\ 24. URL https://doi. org/10.1007/978-3-031-73007-8\_24.

Christos Ziakas and Alessandra Russo. Vita: Zero-shot value functions via test-time adaptation of vision–language models. In International Conference on Learning Representations, volume 2026, pp. 523–543, 2026.

## A NOTATION AND ORGANIZATION OF THE ANALYSIS

The analyses ask three questions. How much can shared labels tell us about candidate accuracy? How often does a finite query budget select the better output? How does retaining an update affect future predictions? Tables 6 and 7 summarize the models and results. Appendix W adds stopping comparisons, calculations on real prediction pairs, and tests with changing class probabilities.

Table 6: Statistical objects used throughout the paper.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\boldsymbol { \mathcal { A } } _ { t } , \boldsymbol { I }$ </td><td>Available batch interventions and incumbent at time t.</td></tr><tr><td> $Q _ { t } , E _ { t }$ </td><td>Permanent eligible query and deployment/evaluation inputs.</td></tr><tr><td> $R _ { a } , D _ { a }$ </td><td>Candidate correctness and its difference from incumbent correctness.</td></tr><tr><td> $s , w _ { s } , p _ { s a }$ </td><td>Joint prediction stratum, target frequency and candidate class.</td></tr><tr><td> $\theta _ { s } , A _ { a } ( \theta )$ </td><td>Conditional label law and expected target accuracy.</td></tr><tr><td> $\Delta _ { a } , v _ { s , a }$ </td><td>Best-candidate gap and within-stratum paired reward variance.</td></tr><tr><td> $q _ { s } , N _ { s } , \tau$ </td><td>Query allocation, acquired stratum count and stopping time.</td></tr><tr><td> $\textstyle \alpha _ { s } , \mu _ { a } , \Sigma _ { a b }$ </td><td>Dirichlet parameters, posterior accuracy mean and covariance.</td></tr><tr><td> $V _ { s } , \bar { U }$ </td><td>One-label decision value and perfect-information upper bound.</td></tr><tr><td> $r _ { a } , b _ { a } , \mathcal { R } _ { m }$ </td><td>Deployment risk, Gaussian evidence bias and selection regret.</td></tr><tr><td> $n _ { + } , n _ { - } , n _ { 0 } , S _ { m }$ </td><td>Finite-pool rescue/harm/neutral counts and queried signed score.</td></tr><tr><td> $\Delta \Theta , \Delta m , d$ </td><td>Residual-head difference, cache difference and induced logit change.</td></tr></table>

Table 7: Result dependencies and the quantity each result predicts.
<table><tr><td>Analysis</td><td>Required observation structure</td><td>Result and validation</td></tr><tr><td>Shared-label complex- ity</td><td>Independent categorical labels within known, queryable strata; fixed predic- tion signatures</td><td>Information lower bound, paired- variance upper bound and local allo- cation coefficient.</td></tr><tr><td>Posterior querying</td><td>Positive independent Dirichlet priors and shared stratum label laws</td><td>Joint accuracy covariance, one-label value and sufficient local stopping conditions.</td></tr><tr><td>Gaussian finite budget</td><td>Common-factor joint Gaussian feed- back with fixed risks and biases</td><td>Exact winner probabilities, budget and correlation responses; indepen- dent score simulation.</td></tr><tr><td>Frozen neural pools</td><td>Uniform subsets of a specified finite query pool; frozen predictions and evaluation outcomes</td><td>Exact discrete selection and deploy- ment loss; independent subset simula- tion.</td></tr><tr><td>Persistent state</td><td>Shared future inputs, query labels and affine head/cache updates</td><td>Exact state recurrence, directional loss interval and margin bound; explicit branch replay.</td></tr></table>

The batch oracle in empirical tables chooses the best candidate after seeing that batch’s evalua tion labels. The categorical population oracle maximizes $A _ { a } ( \theta )$ . The posterior analysis gives the covariance adjustment required to analyze a realized finite evaluation block. Mean-logit and learnedreadout outputs expand the prediction functions themselves and are listed separately when they fall outside an oracle’s discrete candidate set.

## B SHARED-LABEL INFORMATION AND FINITE-CANDIDATE SELECTION

This appendix establishes the information bound and local allocation result in Theorem 2, then derives the posterior calculations and stopping rule in Theorem 1. The observation is a class label. Its value comes from the joint vector of candidate outcomes that it reveals.

## B.1 STATISTICAL EXPERIMENT AND ASSUMPTIONS

There are S strata, C classes and K candidate rules. A candidate a predicts class $p _ { s a }$ throughout stratum s. These prediction signatures are fixed during the decision. The known target weights satisfy

$w _ { s } > 0$ and $\textstyle \sum _ { s } w _ { s } = 1$ . A query in stratum s returns an independent label $Y \sim { \mathrm { C a t e g o r i c a l } } ( \theta _ { s } )$ with $\theta _ { s } \in \Delta _ { C }$ , where $\Delta _ { C }$ contains nonnegative class probabilities summing to one. Query and target labels have this same conditional law. The population accuracy of candidate a is

$$
A _ { a } ( \theta ) = \sum _ { s = 1 } ^ { S } w _ { s } \theta _ { s , p _ { s a } } .\tag{9}
$$

A stratum may be queried repeatedly in this experiment. A fixed finite pool sampled without replacement has a different sampling law, treated in the finite-budget appendix. Conditional agreement of query and target label laws is the transport assumption: changing the frequency of strata is corrected by $w _ { s } ,$ , while a changed label law within a stratum requires a richer model.

Rules with the same prediction signature on every positive-weight stratum are represented once. For fixed-confidence identification, the remaining set has a unique best rule ∗ and gaps $\Delta _ { a } = A _ { * } - A _ { a } >$ 0 for $a \neq *$ . The case $K = 1$ requires no queries. A uniformly δ-correct strategy, $0 < \delta < 1 / 2$ stops almost surely and returns the best rule with probability at least $1 - \delta$ at every admissible unique-best parameter. Randomized stratum choices and stopping times may depend on previously observed labels.

The Bayesian calculations add independent priors $\theta _ { s } \sim \mathrm { D i r i c h l e t } ( \alpha _ { s } ^ { 0 } )$ , with every coordinate strictly positive. They describe posterior expected population accuracy under this model. The finiteevaluation oracle, which sees the realized labels of a target block, is distinguished in Section D.4.

## B.2 PAIRED REWARD GEOMETRY

One observed label produces all rewards $R _ { a } = { \bf 1 } \{ p _ { s a } = Y \}$ . Write $D _ { s , a b } = R _ { a } - R _ { b } , d _ { s , a b } =$ $\mathbb { E } D _ { s , a b }$ , and $v _ { s , a b } = \mathrm { V a r } ( D _ { s , a b } )$ . On agreement strata $D _ { s , a b } = 0$ . On disagreement strata its values $1 , - 1 , 0$ have probabilities $\theta _ { s , p _ { s a } } , \theta _ { s , p _ { s b } } , 1 - \theta _ { s , p _ { s a } } - \theta _ { s , p _ { s b } } .$ Consequently,

$$
d _ { s , a b } = \theta _ { s , p _ { s a } } - \theta _ { s , p _ { s b } } ,\tag{10}
$$

$$
v _ { s , a b } = { \bf 1 } \{ p _ { s a } \neq p _ { s b } \} \left[ \theta _ { s , p _ { s a } } + \theta _ { s , p _ { s b } } - d _ { s , a b } ^ { 2 } \right] ,\tag{11}
$$

$$
\operatorname { C o v } ( R _ { a } , R _ { b } \mid s ) = \mathbf { 1 } \{ p _ { s a } = p _ { s b } \} \theta _ { s , p _ { s a } } - \theta _ { s , p _ { s a } } \theta _ { s , p _ { s b } } .\tag{12}
$$

These expressions retain dependence between candidates. A label is counted once, including when many candidates predict its class. Let ${ \mathcal { T } } _ { a b } = \{ s : p _ { s a } \neq p _ { s b } \}$ . Both the pair’s mean difference and its estimation variance depend exclusively on these disagreement strata.

## B.3 ADAPTIVE INFORMATION LOWER BOUND

Let $N _ { s } ( \tau )$ be the number of labels from stratum s before stopping. Define

$$
{ \mathrm { A l t } } ( \theta ) = \{ \lambda : \lambda { \mathrm { h a s ~ a ~ u n i q u e ~ b e s t ~ c a n d i d a t e ~ d i f f e r e n t ~ f r o m ~ * } } \} ,\tag{13}
$$

$$
\Gamma ^ { * } ( \theta ) = \operatorname* { s u p } _ { q \in \Delta _ { S } } \operatorname* { i n f } _ { \lambda \in \operatorname { A l t } ( \theta ) } \sum _ { s } q _ { s } \operatorname { K L } ( \theta _ { s } \| \lambda _ { s } ) , \qquad T ^ { * } ( \theta ) = \Gamma ^ { * } ( \theta ) ^ { - 1 } .\tag{14}
$$

The divergence is categorical KL, $\begin{array} { r } { \mathrm { K L } ( \theta _ { s } \| \boldsymbol { \lambda _ { s } } ) = \sum _ { c } \theta _ { s c } \log ( \theta _ { s c } / \lambda _ { s c } ) } \end{array}$

Lemma 1 (Stopped shared-label likelihood). For every uniformly δ-correct strategy and every admissible alternative λ,

$$
\sum _ { s } \mathbb { E } _ { \theta } N _ { s } ( \tau ) \mathrm { K L } ( \theta _ { s } \| \lambda _ { s } ) \geq \mathrm { k l } ( 1 - \delta , \delta ) .\tag{15}
$$

Here kl is binary relative entropy. Interior parameters suffice; boundary parameters use the usual absolute-continuity and extended-divergence conventions.

Proof. The transcript contains randomization, chosen strata, labels, the stopping decision and the final recommendation. Given its observed history, the strategy uses the same action kernel under θ and λ. These kernels cancel in the likelihood ratio. Thus

$$
\log { \frac { d P _ { \theta } } { d P _ { \lambda } } } = \sum _ { t = 1 } ^ { \tau } \log { \frac { \theta _ { S _ { t } , Y _ { t } } } { \lambda _ { S _ { t } , Y _ { t } } } } .
$$

For interior finite categorical laws, each increment is bounded in absolute value. If $\mathbb { E } _ { \theta } \tau < \infty$ absolute summability justifies exchanging expectation and the stopped sum. Conditioning on the preceding history gives the left-hand side of (15). If expected time is infinite, the expected-time lower bound is already satisfied.

Apply data processing to $B = \{ \widehat { a } = * \}$ . Correctness implies $P _ { \theta } ( B ) \geq 1 - \delta$ and $P _ { \lambda } ( B ) \leq \delta$ Binary KL is increasing in its first argument and decreasing in its second when the first exceeds the second: its derivatives are $\log [ p ( 1 - q ) / ( q ( 1 - p ) ) ]$ and $( q - p ) / ( q ( 1 - q ) )$ . Transcript KL is therefore at least $\operatorname { k l } ( 1 - \delta , \delta )$ □

Normalize the expected counts by $\mathbb { E } _ { \theta } \tau$ to obtain an allocation in $\Delta _ { S } .$ . Taking the infimum over alternatives and then the supremum over allocations proves Equation (7). An empty alternative set is a trivial identification problem; zero characteristic information yields an infinite lower bound. The proof is the change-of-measure principle of Garivier $\&$ Kaufmann (2016) applied to a shared categorical observation.

For a fixed $q ,$ the inner optimization is computable through $K - 1$ convex programs. Its alternative closure is the union of constraints $A _ { a } ( \lambda ) \geq A _ { * } ( \lambda ) , a \neq *$ , over product simplices. To verify the closure, concentrating each stratum on candidate $\therefore { } a \mathbf { \hat { s } }$ predicted class makes a uniquely best among distinct signatures. A boundary point where a and ∗ share the maximum can be perturbed toward that law. If another rule already exceeds $^ { * , }$ the strict inequality persists locally. A further arbitrarily small interior perturbation breaks maximal ties: distinct linear accuracy functions have equality hyperplanes with empty interior. The infimum over unique-best alternatives consequently agrees with that over these weak boundary constraints. This argument uses the full product-simplex parameter space.

## B.4 A FINITE-SAMPLE CONFIDENCE PROCEDURE

Choose $q _ { s } > 0$ before collecting certification labels. At stage $h \geq 0$ , put $t _ { h } = 2 ^ { h }$ and gather the first $n _ { s } ( h ) = \lceil t _ { h } q _ { s } \rceil + 2$ labels from each independent stratum stream, reusing previous prefixes. Let $M = K ( K - 1 ) / 2$ and

$$
H _ { h } = \log \frac { 4 S M ( h + 1 ) ( h + 2 ) } { \delta } .\tag{16}
$$

All unique candidates remain in the certification set at every stage. The empirical maximizer is returned when its weighted empirical reward difference exceeds the corresponding radius against every other candidate.

Lemma 2 (Paired variance and mean confidence). For $n _ { s } \geq 2 ,$ , set $k _ { s } = \lfloor n _ { s } / 2 \rfloor$ andform disjoint observation pairs

$$
\begin{array} { r } { W _ { i , s , a b } = \frac { 1 } { 2 } ( D _ { 2 i - 1 , s , a b } - D _ { 2 i , s , a b } ) ^ { 2 } , \qquad i = 1 , \ldots , k _ { s } . } \end{array}
$$

Define $U _ { s , a b } = 0$ on agreement strata and otherwise

$$
U _ { s , a b } = \operatorname* { m i n } \{ 1 , 2 { \overline { { W } } } _ { s , a b } + 8 H _ { h } / k _ { s } \} .\tag{17}
$$

With probability at least $1 - \delta ,$ , simultaneously for every stage and pair,

$$
v _ { s , a b } \leq U _ { s , a b } \leq 4 v _ { s , a b } + 1 3 H _ { h } / k _ { s } ,\tag{18}
$$

$$
| \widehat { d } _ { a b } - ( A _ { a } - A _ { b } ) | \leq \rho _ { a b } ( h ) ,\tag{19}
$$

$$
\rho _ { a b } ( h ) = \sqrt { 2 H _ { h } \sum _ { s } \frac { w _ { s } ^ { 2 } U _ { s , a b } } { n _ { s } ( h ) } } + \frac { 4 H _ { h } } { 3 } \operatorname* { m a x } _ { s \in \mathcal { T } _ { a b } } \frac { w _ { s } } { n _ { s } ( h ) } .\tag{20}
$$

The maximum over an empty disagreement set equals zero.

Proof. For independent centered $Z _ { i } , | Z _ { i } | \le B$ , and $\textstyle \sum _ { i } \mathbb { E } Z _ { i } ^ { 2 } \leq V$ , the exponential-moment expansion gives

$$
\log \mathbb { E } e ^ { t Z _ { i } } \leq \frac { t ^ { 2 } \mathbb { E } Z _ { i } ^ { 2 } } { 2 ( 1 - t B / 3 ) } , \qquad 0 \leq t < 3 / B .
$$

Indeed, $\mathbb { E } | Z _ { i } | ^ { k } \le B ^ { k - 2 } \mathbb { E } Z _ { i } ^ { 2 }$ and $k ! \geq 2 \cdot 3 ^ { k - 2 }$ . Multiplication, exponential Markov and optimization yield the two-sided Bernstein bound

$$
P \left( \left| \sum _ { i } Z _ { i } \right| > { \sqrt { 2 V x } } + 2 B x / 3 \right) \leq 2 e ^ { - x } .
$$

Independence within each disjoint pair gives $\mathbb { E } W = v ,$ , while $0 \leq W \leq 2$ and $\mathrm { V a r } ( W ) \leq 2 v$ . For $a = H _ { h } / k _ { i }$ <sub>s</sub> the preceding bound implies $| \overline { { W } } - v | \leq 2 \sqrt { v a } + 4 a / 3$ . Using $2 \sqrt { v a } \leq v / 2 + 2 a$ gives $v \leq 2 \overline { { W } } + 2 0 a / 3$ . Using $2 { \sqrt { v a } } \leq v + a$ gives $2 \overline { { W } } + 8 a \leq 4 v + 3 8 a / 3$ . Since $v \leq 1$ , clipping preserves the lower bound in (17).

For a weighted empirical pair difference, the centered summands have total variance $\textstyle \sum _ { s } w _ { s } ^ { 2 } v _ { s } / n _ { s }$ and absolute bound $2 \operatorname* { m a x } _ { s \in \mathbb { Z } _ { a b } } w _ { s } / n _ { s }$ . Bernstein gives (20) whenever the variance upper bound hold. At stage $h ,$ union bounds over pairs and stratum-pair variance estimates cost at most $2 M e ^ { - H _ { h } } + \bar { 2 S } M e ^ { - H _ { h } }$ . Summing stages uses $\begin{array} { r } { \sum _ { h > 0 } [ ( h \dot { + } 1 ) ( h + 2 ) ] ^ { - 1 } = 1 } \end{array}$ , so total failure probability is at most δ. The paired events need not be independent. □

Define

$$
V _ { a b } ( q ) = \sum _ { s } \frac { w _ { s } ^ { 2 } v _ { s , a b } } { q _ { s } } , \qquad W _ { a b } ( q ) = \left( \sum _ { s \in \mathcal { T } _ { a b } } \frac { w _ { s } ^ { 2 } } { q _ { s } ^ { 2 } } \right) ^ { 1 / 2 } .\tag{21}
$$

Because $k _ { s } \ge n _ { s } / 3$ , the confidence event implies

$$
\rho _ { a b } ( h ) \leq 3 \sqrt { H _ { h } V _ { a b } ( q ) / t _ { h } } + 1 1 H _ { h } W _ { a b } ( q ) / t _ { h } .\tag{22}
$$

For the constants, split the square root into $\sqrt { 8 H _ { h } \sum _ { s } w _ { s } ^ { 2 } v _ { s } / n _ { s } }$ <sub>s</sub> and $\textstyle \sqrt { 7 8 } H _ { h } ( \sum _ { s \in \mathcal { T } } w _ { s } ^ { 2 } / n _ { s } ^ { 2 } ) ^ { 1 / 2 }$ ; the range term adds $4 / 3$ times the latter norm. Use $\sqrt { 8 } < 3$ and $\sqrt { 7 8 } + 4 / 3 < 1 1$

Theorem 3 (Fixed-allocation identification). The full-set certification procedure is uniformly $\delta \mathfrak { - }$ correct and stops almost surely at every unique-best law. At any stage satisfying

$$
t _ { h } \ge \operatorname* { m a x } _ { a \ne * } \operatorname* { m a x } \left\{ \frac { 5 7 6 H _ { h } V _ { * , a } ( q ) } { \Delta _ { a } ^ { 2 } } , \frac { 8 8 H _ { h } W _ { * , a } ( q ) } { \Delta _ { a } } \right\} ,\tag{23}
$$

it has stopped after at most $t _ { h } + 3 S$ labels with probability at least $1 - \delta .$

Proof. On the simultaneous event, an incorrect certified winner j would satisfy $A _ { j } - A _ { * } \geq \widehat { d } _ { j * } -$ $\rho _ { j * } > 0 $ , a contradiction. Condition (23) makes each term of (22) at most $\Delta _ { a } / 8 .$ . Thus $\stackrel { \cdot } { d _ { * , a } } \geq$ $3 \bar { \Delta } _ { a } / 4 > \rho _ { * , a }$ for all rivals: the true best is the empirical maximizer and is certified. Counts sum to at most $t _ { h } + 3 S$

For almost-sure termination, apply the strong law to every bounded independent stratum stream. All empirical accuracies converge, every radius tends to zero because $U \overset { \cdot } { \leq } 1$ , and the finite set has a positive minimum best-versus-rival gap. Eventually the best is the empirical maximizer and all its empirical gaps exceed their radii. This argument applies on the entire probability-one convergence event, including paths outside the simultaneous confidence event. Keeping the full candidate set in certification is essential to this argument when inferior population accuracies tie. □

One query informs every pair. The leading variance term is therefore a maximum of pair difficulties, $\operatorname* { m a x } _ { a \neq * } \dot { V } _ { * , a } ( q ) / { \Delta _ { a } ^ { 2 } }$ , with explicit range and logarithmic terms. For a single pair, Cauchy–Schwarz gives

$$
\operatorname* { i n f } _ { q \in \Delta _ { S } } V _ { a b } ( q ) = \left( \sum _ { s } w _ { s } \sqrt { v _ { s , a b } } \right) ^ { 2 } , \qquad q _ { s } \propto w _ { s } \sqrt { v _ { s , a b } } .\tag{24}
$$

Zero-variance strata can receive zero mass in this variance-only optimization; the preceding finitesample procedure retains positive allocations and its range term.

For outcome-dependent counts, correctness can instead be based on a simultaneous bound over all prefix vectors. Replacing the mean threshold by $\log ( 4 M / \delta ) + \textstyle \sum _ { s } \log [ n _ { s } ( n _ { s } - 1 ) ]$ and the variance threshold at prefix-pair count k by $\log [ 4 S M \dot { k } ( k + 1 ) / \delta ]$ gives the same confidence conclusion through summable union bounds. The price is an $S \log n$ term. This supplies an adaptive certificate, while a stopping rate additionally requires the policy to sample every unresolved distinguishing stratum.

## B.5 FIXED-BUDGET REGRET

For deterministic stratum counts $n _ { s } \geq 1$ , let $\widehat { a }$ maximize weighted empirical reward. Define

$$
V _ { a } ( n ) = \sum _ { s } w _ { s } ^ { 2 } { v } _ { s , * , a } / n _ { s } , \qquad B _ { a } ( n ) = \operatorname* { m a x } _ { s \in \mathcal { L } _ { * , a } } w _ { s } / n _ { s } .
$$

Then

$$
{ \mathbb E } [ A _ { * } - A _ { \hat { a } } ] \leq \operatorname* { m i n } \left\{ 1 , \sum _ { a \neq * } \Delta _ { a } \exp \left[ - \frac { \Delta _ { a } ^ { 2 } } { 2 V _ { a } ( n ) + ( 4 / 3 ) B _ { a } ( n ) \Delta _ { a } } \right] \right\} .\tag{25}
$$

Selecting a requires $\widehat { d } _ { * , a } \leq 0$ . The one-sided Bernstein exponent for centered bound $2 B _ { a } ( n )$ gives its probability; multiplying by $\Delta _ { a }$ and summing possible winners proves the result. A deterministic positive-gap pair has zero error probability and may contribute zero directly. Independent pilotselected counts followed by fresh samples satisfy the same conditional argument. Arbitrary stopped counts require a simultaneous certificate.

If candidate 0 is the incumbent, the deployment relation is

$$
\mathbb { E } [ A _ { \widehat { a } } - A _ { 0 } ] = ( A _ { * } - A _ { 0 } ) - \mathbb { E } [ A _ { * } - A _ { \widehat { a } } ] .\tag{26}
$$

Thus a stronger portfolio improves available accuracy while the feedback mechanism determines how much of that improvement is realized. Under conditional query/target exchangeability, an independent target block with the specified stratum weights estimates the left side.

## C LOCAL INFORMATION GEOMETRY AND AN ATTAINABLE ALLOCATION

The finite-sample bound retains conservative constants and a range term. We next identify a regime in which paired variance gives the leading information complexity and construct a procedure attaining that coefficient.

## C.1 UNIFORM EXPANSION OF THE EXACT CATEGORICAL PROBLEM

For a best-versus-rival pair, define $\begin{array} { r } { g _ { s c } = \mathbf { 1 } \{ p _ { s * } = c \} - \mathbf { 1 } \{ p _ { s a } = c \} , d _ { s } = \sum _ { c } \theta _ { s c } g _ { s c } } \end{array}$ , and $\Delta =$ $\begin{array} { r } { \sum _ { s } w _ { s } d _ { s } > 0 . } \end{array}$ . Let

$$
I _ { a } ( \theta , q ) = \operatorname* { i n f } _ { \substack { \lambda : \sum _ { s , c } w _ { s } g _ { s c } \lambda _ { s c } \leq 0 } } \sum _ { s } q _ { s } \mathrm { K L } ( \theta _ { s } \| \lambda _ { s } ) .
$$

Fix $b _ { 0 } > 0$ and $\eta > 0$ , with $C b _ { 0 } < 1$ and $S \eta < 1$ , and restrict to $\theta _ { s c } \geq b _ { 0 }$ and $Q _ { \eta } = \{ q \in \Delta _ { S }$ $q _ { s } \geq \eta \}$

Lemma 3 (Uniform pairwise information expansion). For afixedfinite prediction map and positive target weights,

$$
I _ { a } ( \theta , q ) = \frac { \Delta ^ { 2 } } { 2 V _ { a } ( \theta , q ) } + O ( \Delta ^ { 3 } )\tag{27}
$$

uniformly on the preceding domain as $\Delta \downarrow 0 .$ . The remainder constant depends on $b _ { 0 }$ , $\eta ,$ the weights and the prediction map.

Proof. The pair is distinct, so its variance functional is bounded above and away from zero on this compact domain. Define the tangent displacement

$$
u _ { s c } = - \frac { \Delta } { V _ { a } } \frac { w _ { s } } { q _ { s } } \theta _ { s c } ( g _ { s c } - d _ { s } ) .\tag{28}
$$

Its rows sum to zero and $\begin{array} { r } { \sum _ { s , c } w _ { s } g _ { s c } u _ { s c } = - \Delta } \end{array}$ . It is uniformly $O ( \Delta )$ , making $\theta + u$ a feasible interior boundary law for sufficiently small $\Delta$ . Taylor expansion has no linear term on row-sumzero tangents, and uniformly bounded third derivatives give

$$
\sum _ { s } q _ { s } \mathrm { K L } ( \theta _ { s } \| \theta _ { s } + u _ { s } ) = \frac { 1 } { 2 } \sum _ { s , c } \frac { q _ { s } u _ { s c } ^ { 2 } } { \theta _ { s c } } + O ( \| u \| ^ { 3 } ) .\tag{29}
$$

Substitution gives the required upper bound.

An exact minimizer exists by lower semicontinuity on the compact product simplex; the objective diverges at a zero coordinate because $q _ { s } \theta _ { s c } > 0$ Its Hessian in λ is diagonal with entries $q _ { s } \theta _ { s c } / \lambda _ { s c } ^ { 2 } \geq b _ { 0 } \eta$ . Since the gradient at θ vanishes on simplex tangents, the objective is at least $b _ { 0 } \eta \| \dot { \lambda } - \mathbf { \nabla } \theta \| _ { 2 } ^ { 2 } / 2$ The feasible $\mathrm { \bar { \it O } } ( \Delta ^ { 2 } )$ cost forces any optimizer into an $O ( \Delta )$ neighborhood uniformly. Its constraint is active: a point strictly beyond the boundary can be moved toward θ until it meets the boundary, strictly reducing the positive convex objective.

For any boundary tangent, centering g by $d _ { s }$ and applying Cauchy–Schwarz gives

$$
\Delta ^ { 2 } \leq \left( \sum _ { s , c } \frac { q _ { s } u _ { s c } ^ { 2 } } { \theta _ { s c } } \right) \left( \sum _ { s , c } \frac { w _ { s } ^ { 2 } \theta _ { s c } ( g _ { s c } - d _ { s } ) ^ { 2 } } { q _ { s } } \right) .
$$

The second factor is $V _ { a } ;$ the displacement (28) attains equality. Apply (29) to the exact local optimizer to obtain the matching lower bound. This proves the expansion of the actual constrained KL minimum. □

## C.2 CANDIDATE FAMILIES AND ALLOCATION OPTIMIZATION

Consider $\theta ^ { \varepsilon }  \theta ^ { 0 }$ with every categorical probability at least $b _ { 0 }$ , and with the same uniquely best candidate ∗ for $\varepsilon > 0 .$ . A nonempty set J of rivals ties with ∗ at the limit. Assume

$$
\Delta _ { a } ^ { \varepsilon } = \kappa _ { a } \varepsilon + O ( \varepsilon ^ { 2 } ) , \quad \kappa _ { a } > 0 ( a \in J ) , \qquad \operatorname* { l i m } _ { \varepsilon \downarrow 0 } \Delta _ { a } ^ { \varepsilon } > 0 ( a \not \in J ) .\tag{30}
$$

Set

$$
H _ { 0 , \eta } = \operatorname* { m i n } _ { q \in Q _ { \eta } } \operatorname* { m a x } _ { a \in J } \frac { V _ { a } ( \theta ^ { 0 } , q ) } { \kappa _ { a } ^ { 2 } } .\tag{31}
$$

Uniformity of Lemma 3 permits taking the finite minimum over rivals and the supremum over compact $Q _ { \eta } ,$ , yielding

$$
\operatorname* { s u p } _ { q \in Q _ { \eta } } \operatorname* { m i n } _ { a \neq \ast } I _ { a } ( \theta ^ { \varepsilon } , q ) = \frac { \varepsilon ^ { 2 } } { 2 H _ { 0 , \eta } } + o ( \varepsilon ^ { 2 } ) .\tag{32}
$$

Nonlocal rivals have uniformly positive boundary information by the Hessian lower bound, so they do not determine this minimum for small $\varepsilon$ . The design is convex: introduce a scalar upper bound on every $\begin{array} { r } { \sum _ { s } w _ { s } ^ { 2 } v _ { s , a } / ( \kappa _ { a } ^ { 2 } q _ { s } ) } \end{array}$ , together with the simplex and floor constraints.

When the allocation floor is inactive. Suppose every stratum separates ∗ from at least one rival in J. The unrestricted variance objective diverges when any $q _ { s }  0 ,$ , because an interior categorical law gives that pair positive variance there. Its minimizers consequently lie in the simplex interior. Exact information-maximizing allocations are also uniformly interior in the local limit. A fixed interior allocation attains at least $c \varepsilon ^ { 2 }$ . For each stratum, changing only its categorical law by $O ( \varepsilon )$ closes the gap of an associated local rival, at unweighted KL cost at most $C _ { s } \varepsilon ^ { 2 }$ . The information under any q is therefore at most $C _ { s } q _ { s } \varepsilon ^ { 2 }$ . Competing with the fixed interior allocation requires $q _ { s } \geq c / C _ { s }$ . Thus, for a sufficiently small fixed floor containing a local minimizer,

$$
\Gamma ^ { * } ( \theta ^ { \varepsilon } ) = \frac { \varepsilon ^ { 2 } } { 2 H _ { 0 } } + o ( \varepsilon ^ { 2 } ) , \qquad T ^ { * } ( \theta ^ { \varepsilon } ) = \frac { 2 H _ { 0 } } { \varepsilon ^ { 2 } } ( 1 + o ( 1 ) ) ,\tag{33}
$$

where $H _ { 0 } = H _ { 0 , \eta }$ . If this coverage condition fails, Equation (32) still describes fixed positive-floor designs; boundary allocations require a separate analysis.

## C.3 PILOT DESIGN WITH FRESH-SAMPLE CERTIFICATION

Let $L = \log ( 1 / \delta )$ and fix $0 < \eta < 1 / S$ . The following procedure uses only queried labels and known prediction signatures.

1. Query each stratum $n _ { 0 } = \lceil \operatorname* { m a x } \{ 1 6 , \sqrt { L } \} \rceil$ times and form smoothed categorical frequencies $\widehat { \theta }$ using fixed positive pseudocounts.

2. Compute the proxy best $\widehat { * }$ and choose

$$
\widehat { q } \in \arg \operatorname* { m i n } _ { q \in Q _ { \eta } } \operatorname* { m a x } _ { a \neq \widehat { \ast } } \frac { V _ { \widehat { \ast } , a } ( \widehat { \theta } , q ) } { \operatorname* { m a x } \{ \widehat { \Delta } _ { a } , ( n _ { 0 } + 1 ) ^ { - 1 } \} ^ { 2 } } .\tag{34}
$$

3. At checkpoints $t _ { 0 } = 1 , t _ { h + 1 } = \lceil ( 1 + \gamma ) t _ { h } \rceil$ , collect $\lceil t _ { h } \widehat { q _ { s } } \rceil + 2$ fresh labels in each stratum. Pilot observations do not enter the certification statistics.

4. Return the empirical maximizer when its mean difference is certified positive against every other unique candidate, keeping the full set in every certification.

Conditioned on the pilot, the subsequent prefix counts are deterministic and their labels are independent of the design. An inaccurate pilot changes efficiency, while the fresh-sample certificate remains valid.

To obtain the local leading coefficient, use the tighter variance upper bound

$$
U _ { s } = \operatorname* { m i n } \left\{ 1 , \left( \sqrt { a _ { s } } + \sqrt { \overline { { W } } _ { s } + 7 a _ { s } / 3 } \right) ^ { 2 } \right\} , \quad \quad a _ { s } = H _ { h } / \lfloor n _ { s } / 2 \rfloor ,\tag{35}
$$

with zero on agreement strata. The Bernstein event from the preceding appendix gives

$$
v _ { s } \leq U _ { s } \leq v _ { s } + 6 \sqrt { v _ { s } a _ { s } } + 9 a _ { s } .\tag{36}
$$

For the lower inequality solve $v - \overline { { W } } \leq 2 \sqrt { v a } + 4 a / 3$ as a quadratic in ${ \sqrt { v } } .$ . For the upper inequality, $\overline { { W } } + 7 a / 3 \leq v + 2 \sqrt { v a } + 1 1 a / 3 \leq ( \sqrt { v } + 2 \sqrt { a } ) ^ { 2 }$ . Add $\sqrt { a }$ and square. The stage allocation (16) again gives simultaneous confidence with total failure probability at most δ.

Use (20) with (35). On this event,

$$
\rho _ { * , a } ( h ) \leq \sqrt { 2 H _ { h } V _ { a } ( \widehat { q } ) / t _ { h } } + 9 H _ { h } W _ { a } ( \widehat { q } ) / t _ { h } .\tag{37}
$$

For completeness, write $\begin{array} { r } { V _ { n } = \sum _ { s } w _ { s } ^ { 2 } v _ { s } / n _ { s } } \end{array}$ and $\begin{array} { r } { J _ { n } = \sum _ { s \in \mathcal { T } } w _ { s } ^ { 2 } / n _ { s } ^ { 2 } } \end{array}$ . Since $k _ { s } \ge n _ { s } / 3$ , Cauchy– Schwarz gives

$$
\sum _ { s } \frac { w _ { s } ^ { 2 } U _ { s } } { n _ { s } } \leq V _ { n } + 6 \sqrt { 3 } \sqrt { H _ { h } V _ { n } J _ { n } } + 2 7 H _ { h } J _ { n } = ( \sqrt { V _ { n } } + 3 \sqrt { 3 } \sqrt { H _ { h } J _ { n } } ) ^ { 2 } .
$$

The resulting range coefficient is $3 { \sqrt { 6 } } + 4 / 3 < 9$ . In particular, if

$$
t _ { h } \ge \operatorname* { m a x } _ { a \ne * } \operatorname* { m a x } \left\{ 1 2 8 H _ { h } V _ { a } ( \widehat { q } ) / \Delta _ { a } ^ { 2 } , 7 2 H _ { h } W _ { a } ( \widehat { q } ) / \Delta _ { a } \right\} ,\tag{38}
$$

then the total labels are at most $S { n } _ { 0 } + { t } _ { h } + 3 S$ with probability at least $1 - \delta$ . The full-set argument of Theorem 3 also proves almost-sure stopping.

Theorem 4 (Ordered local efficiency). Let the interior local family satisfy (30), and define

$$
H _ { \eta } ( \theta ) = \operatorname* { m i n } _ { q \in Q _ { \eta } } \operatorname* { m a x } _ { a \neq * } \frac { V _ { a } ( \theta , q ) } { \Delta _ { a } ( \theta ) ^ { 2 } } .
$$

For the uncapped two-stage procedure with exact convex minimization, or relative optimization error tending to zero as $\delta \downarrow 0 ,$

$$
\operatorname* { l i m } _ { \varepsilon \downarrow 0 } \operatorname* { s u p } _ { \delta \downarrow 0 } \operatorname* { s u p } _ { 2 H _ { \eta } ( \theta ^ { \varepsilon } ) \log ( 1 / \delta ) } \leq 1 + \gamma .\tag{39}
$$

With $\gamma = \gamma _ { \delta } = L ^ { - 1 / 4 }$ , the right-hand side is one. Under the active-coverage and inactive-floor conditions above, this coefficient equals the local information lower bound.

Proof. First fix an interior unique-best θ. As δ ↓ 0, the pilot size diverges and is $o ( L )$ . The proxy best is correct with probability tending to one. All true gaps are positive at this fixed parameter, so the gap floor disappears asymptotically, and the pilot objective converges uniformly on compact $Q _ { \eta } .$ Its achieved true objective therefore converges in probability to $H _ { \eta } ( \theta )$ . No uniqueness of the allocation minimizer is needed.

Now consider $\theta = \theta ^ { \varepsilon }$ and the scale $t = 2 ( 1 + \zeta ) H _ { \eta } ( \theta ^ { \varepsilon } ) L$ , for fixed $\zeta > 0$ . At each fixed positive $\varepsilon ,$ fresh means and disjoint-pair variance estimates converge uniformly in probability over $q \in Q _ { \eta } .$ every count is at least ηt. The geometric stage count is ${ \cal O } _ { \gamma } ( \log L )$ , hence $\bar { H _ { h } } / L \to 1$ . Uniform local convergence also gives $\varepsilon ^ { 2 } H _ { \eta } ( \theta ^ { \varepsilon } )  H _ { 0 , \eta }$

At this scale $H _ { h } / k _ { s } = { \cal O } ( \varepsilon ^ { 2 } )$ . Substituting the consistent pair variances into (35) gives $U _ { s } ~ =$ $v _ { s } + O ( \varepsilon )$ on disagreement strata. Their true variances are bounded away from zero by the categorica interior assumption. For $a \in J$ , the limiting radius-to-gap ratio is at most

$$
\frac { 1 + O ( \varepsilon ) } { \sqrt { 1 + \zeta } } + O ( \varepsilon ) .
$$

The range term is $O ( \varepsilon ^ { 2 } )$ and the local gap is of order $\varepsilon .$ Nonlocal rivals retain positive gaps while their radii vanish with ε. For sufficiently small $\varepsilon ,$ all limiting radii are strictly below the true gaps. Empirical errors vanish as $\delta \downarrow 0$ at this fixed $\varepsilon ,$ so certification occurs by this scale with probability tending to one. Moving to the next checkpoint costs a factor at most $1 + \gamma + o ( 1 )$

To pass to expectation, use $U ~ \leq ~ 1$ and the allocation floor. Uniformly over pilot outcomes, $\rho _ { * , a } ( h ) \leq C _ { \eta } \sqrt { H _ { h } / t _ { h } } + C _ { \eta } H _ { h } / t _ { h }$ . At a fixed unique-best $\theta ,$ a constant $C _ { \theta }$ makes every radius smaller than $\Delta _ { \mathrm { m i n } } / 4$ for all $\dot { t _ { h } } \geq C _ { \theta } L$ and sufficiently large $L .$ . Here $H _ { h } \leq L + C + 2 \log \log ( t _ { h } + e )$ Failure to stop at such a checkpoint requires a best-versus-rival empirical difference below half its true gap. Conditional on any pilot, bounded independent fresh streams and $n _ { s } \geq \eta t _ { h }$ give

$$
P _ { \theta } ( \tau > S n _ { 0 } + t _ { h } + 3 S \mathrm { ~ } | \mathrm { ~ p i l o t } ) \le ( K - 1 ) e ^ { - c _ { \theta } t _ { h } }\tag{40}
$$

for $c _ { \theta } > 0$ independent of the pilot allocation. Summing this tail over geometric intervals proves uniform integrability of $\tau / L$ . The probability bound consequently passes to expectations; pilot cost is $S n _ { 0 } = o ( \bar { L } )$ . Taking the ordered limits and then $\zeta \downarrow 0$ proves (39).

For $\gamma _ { \delta } = L ^ { - 1 / 4 }$ , checkpoint overshoot tends to one and the logarithm of the stage count remains $o ( L )$ . The same tail argument, with at most polynomially many additional checkpoints at the relevant scale, proves the unit coefficient. Finally (33) and the adaptive lower bound identify the unrestricted coefficient when the floor contains a local oracle. □

This result takes confidence error to zero at each fixed positive gap before taking the local gap limit. A simultaneous shrinking-gap regime would also have to control the pilot’s probability of identifying the relevant competitors. The posterior acquisition rule uses the same paired structure, while its finite-batch decisions are evaluated directly in the experiments.

## D POSTERIOR JOINT INFORMATION AND STOPPING

## D.1 POSTERIOR MOMENTS AND ONE-LABEL UPDATES

After observing stratum class counts ${ { c } _ { s c } } ,$ let $\begin{array} { r } { \alpha _ { s c } = \alpha _ { s c } ^ { 0 } + c _ { s c } , a _ { s } = \sum _ { c } \alpha _ { s c } } \end{array}$ and $m _ { s c } = \alpha _ { s c } / a _ { s }$ Adaptive stratum selection is ignorable in this likelihood when queried inputs are conditionally exchangeable with deployment inputs within a stratum: its action kernels depend on observed history, while the remaining parameter factors are the categorical label probabilities. Selection favoring a subgroup with a different label law is not ignorable. A single class label updates its stratum count once.

Dirichlet integration gives

$$
\mathbb { E } \theta _ { s c } = \frac { \alpha _ { s c } } { a _ { s } } , \qquad \mathbb { E } [ \theta _ { s c } \theta _ { s d } ] = \frac { \alpha _ { s c } ( \alpha _ { s d } + 1 \{ c = d \} ) } { a _ { s } ( a _ { s } + 1 ) } .
$$

These identities follow by adding one or two to the corresponding exponents in the normalized Dirichlet density. Independence between strata and the linear accuracy definition then give Equations (1)–(2):

$$
\mu _ { a } = \sum _ { s } w _ { s } m _ { s , p _ { s a } } ,\tag{41}
$$

$$
\Sigma _ { a b } = \sum _ { s } \frac { w _ { s } ^ { 2 } } { a _ { s } + 1 } \left[ { \bf 1 } \{ p _ { s a } = p _ { s b } \} m _ { s , p _ { s a } } - m _ { s , p _ { s a } } m _ { s , p _ { s b } } \right] .\tag{42}
$$

Thus $\operatorname { V a r } ( A _ { a } - A _ { b } \mid { \mathcal { F } } ) = \Sigma _ { a a } + \Sigma _ { b b } - 2 \Sigma _ { a b }$ . Exact prediction copies have zero difference variance. If the next label from stratum s is $y ,$ each candidate mean changes by

$$
\mu _ { a } ^ { \prime } = \mu _ { a } + \frac { w _ { s } } { a _ { s } + 1 } \left[ { \bf 1 } \{ p _ { s a } = y \} - m _ { s , p _ { s a } } \right] .\tag{43}
$$

Weighting the new maximum by $P ( Y = y ~ \vert ~ \mathcal { F } ) = m _ { s y }$ yields the one-step value in Equation (4). The law of total variance further gives the expected reduction in posterior pair variance:

$$
\operatorname { V a r } ( A _ { a } - A _ { b } \mid { \mathcal { F } } ) - \operatorname { \mathbb { E } } [ \operatorname { V a r } ( A _ { a } - A _ { b } \mid { \mathcal { F } } , Y _ { s } ) \mid { \mathcal { F } } ] = { \frac { w _ { s } ^ { 2 } \operatorname { V a r } _ { m _ { s } } ( R _ { a } - R _ { b } ) } { ( a _ { s } + 1 ) ^ { 2 } } } .\tag{44}
$$

Indeed, the right side is the variance of the conditional-mean change in (43). It supplies the paired information score used when one label cannot yet change the posterior winner.

## D.2 PERFECT-INFORMATION UPPER BOUND

Let j maximize the current posterior means, with deterministic tie handling, and set $d _ { a } = \mu _ { j } - \mu _ { a } \geq$ 0 and $v _ { a } = \mathrm { V a r } ( A _ { a } - A _ { j } \ \bar { | } \ \mathcal { F } )$ . For $X _ { a } = A _ { a } - A _ { j }$ , the identity $x _ { + } = ( | x | + x ) / 2$ and Cauchy– Schwarz imply

$$
\mathbb { E } [ ( X _ { a } ) _ { + } \mid { \mathcal { F } } ] \leq { \frac { { \sqrt { d _ { a } ^ { 2 } + v _ { a } } } - d _ { a } } { 2 } } .\tag{45}
$$

Since $X _ { j } = 0 _ { : }$ , ma $\begin{array} { r } { \mathrm { { x } } _ { a } \ X _ { a } \leq \sum _ { a \neq j } ( X _ { a } ) _ { + } } \end{array}$ . Also max $_ a A _ { a } \leq 1$ . These inequalities prove

$$
\mathbb { E } [ \operatorname* { m a x } _ { a } A _ { a } - A _ { j } \mid \mathcal { F } ] \leq \operatorname* { m i n } \left\{ 1 - \mu _ { j } , \sum _ { a \neq j } \frac { \sqrt { d _ { a } ^ { 2 } + v _ { a } } - d _ { a } } { 2 } \right\} = U ,\tag{46}
$$

as stated in Theorem 1.

For any future query policy and eventual recommendation J, the pointwise inequality $A _ { J } \leq$ $\operatorname* { m a x } _ { a } A _ { a }$ bounds its conditional expected improvement by U. If its first further label costs $c \geq U$ in the same expected-accuracy units and later costs are nonnegative, its net expected improvement is nonpositive. If one decision serves $L _ { d }$ equally weighted future deployments and label cost is in aggregate utility units, the corresponding condition is $L _ { d } U \le c .$ This is a sufficient cost-based stopping condition.

The sum in (46) includes all unique candidates. Removing exact copies preserves the maximum; repeatedly including a nonleading copy would only loosen the sum. A smaller competitive set may be used for acquisition calculations after establishing reachability, while the displayed perfectinformation bound is evaluated over the full unique set.

## D.3 FINITE-BUDGET REACHABLE WINNERS

Suppose at most B additional labels are allowed, and at most $b _ { s }$ remain queryable in stratum s. Take $b _ { s } = \infty$ for independent unlimited streams. For a rival a and feasible integer allocation n, its largest possible future posterior mean advantage over current leader j is

$$
F _ { a } ( n ) = \sum _ { s \in \mathbb { Z } _ { a j } } w _ { s } \frac { \alpha _ { s , p _ { s a } } - \alpha _ { s , p _ { s j } } + n _ { s } } { a _ { s } + n _ { s } } , \quad 0 \leq n _ { s } \leq b _ { s } , \quad \sum _ { s } n _ { s } \leq B .\tag{47}
$$

Every new label in a disagreement stratum is assigned to the rival’s predicted class to maximize the numerator. Agreement strata always contribute zero. Positivity of the Dirichlet parameters makes every such finite label history have positive posterior predictive probability.

Lemma 4 (Exact finite-horizon no-switch criterion). All feasible further query policies have zero expected accuracy improvement if and only if max $, F _ { a } ( n ) \leq 0 f o \iota$ r every rival a.

Proof. If all these maxima are nonpositive, j remains a posterior-mean maximizer after every feasible history. Its conditional expected accuracy satisfies the tower property, so revealing additional labels cannot raise the expected posterior optimum.

Conversely, if a feasible allocation makes $F _ { a } ( n ) > 0$ , use that fixed query schedule. The favorable finite history has positive probability and gives a rival strictly greater posterior mean than j. Write $\mu _ { a } ^ { \prime } = \mathbb { E } [ A _ { a } ] ^ { \prime } ] ^ { T ^ { \prime } ] }$ for the posterior after this schedule. Then

$$
\mathbb { E } [ \operatorname* { m a x } _ { a } \mu _ { a } ^ { \prime } \mid \mathcal { F } ] - \mu _ { j } = \mathbb { E } [ \operatorname* { m a x } _ { a } \mu _ { a } ^ { \prime } - \mu _ { j } ^ { \prime } \mid \mathcal { F } ] > 0 .
$$

The integrand is nonnegative on every history and strictly positive on the favorable event. This proves positive information value when labels have zero cost; a positive information value need not exceed their acquisition cost. □

An exact integer solution is available by dynamic programming. Let $f _ { s } ( n )$ be the stratum contribution in (47), $D _ { 0 } ( 0 ) = 0$ and $D _ { 0 } ( b ) = - \infty$ for $b > 0$ . Then

$$
D _ { s } ( b ) = \operatorname* { m a x } _ { 0 \leq n \leq \operatorname* { m i n } \{ b , b _ { s } \} } \{ D _ { s - 1 } ( b - n ) + f _ { s } ( n ) \} .
$$

Maximizing $D _ { S } ( b )$ over $b \leq B$ costs $O ( S B ^ { 2 } )$ per rival. A simpler equivalent implementation exploits decreasing marginal returns. If $d _ { s } = m _ { s , p _ { s a } } - m _ { s , p _ { s j } }$ , the gain from the nth favorable label is

$$
f _ { s } ( n ) - f _ { s } ( n - 1 ) = \frac { w _ { s } ( 1 - d _ { s } ) a _ { s } } { ( a _ { s } + n ) ( a _ { s } + n - 1 ) } .\tag{48}
$$

These gains are nonnegative and nonincreasing. The B largest gains across the capped stratum sequences therefore solve the allocation: each selected gain has predecessors at least as large, so all prefix constraints can be satisfied. This is the competitive-set computation associated with Equation (3).

A rival can require several favorable labels before overtaking j. Thus a zero one-step value does not imply zero remaining multi-label value. Reachability separates these cases; Equation (44) then measures information gained before the winner changes. In numerical implementations, a tolerance ϵ on the maximal reachable advantage gives an ϵ bound on remaining expected accuracy value in exact arithmetic. Floating-point roundoff is an additional numerical approximation.

## D.4 POPULATION AND REALIZED-BLOCK UNCERTAINTY

Suppose a target block has $M _ { s } \geq 1$ unseen conditionally independent labels in stratum s. Its realized candidate accuracy is $\begin{array} { r } { \widetilde { A } _ { a } = \sum _ { s } w _ { s } M _ { s } ^ { - 1 } \sum _ { i = 1 } ^ { M _ { s } } R _ { s , i , a } } \end{array}$ . The mean remains $\mu _ { a } ,$ , while total covariance gives

$$
\operatorname { C o v } ( { \tilde { A } } _ { a } , { \tilde { A } } _ { b } \mid { \mathcal { F } } ) = \sum _ { s } w _ { s } ^ { 2 } { \frac { a _ { s } + M _ { s } } { M _ { s } ( a _ { s } + 1 ) } } \left[ \mathbf { 1 } \{ p _ { s a } = p _ { s b } \} m _ { s , p _ { s a } } - m _ { s , p _ { s a } } m _ { s , p _ { s b } } \right] .\tag{49}
$$

The expected conditional categorical covariance equals $a _ { s } / ( a _ { s } + 1 )$ times the bracket; averaging $M _ { s }$ labels divides it by $M _ { s }$ . The posterior covariance of the conditional mean adds $1 / ( a _ { s } + 1 )$ times the bracket, yielding the displayed factor. Applying (45) with this covariance bounds the hindsight finite-block oracle instead of the population oracle.

For example, two opposite binary predictions under a known balanced label law both have population accuracy $1 / 2 ,$ , giving zero population perfect-information value. On one realized target item, one prediction is always correct, so a hindsight selector has expected advantage $1 / 2$ . The two oracle values answer different questions. The experiment tables identify their reference unit, and the posterior stopping rule uses conditional expected target accuracy.

## E FINITE-BUDGET SELECTION MODELS

A finite budget determines how often an available intervention is selected, even when its deployment risk is fixed. This section derives exact selection probabilities for a joint Gaussian experiment and a finite categorical query pool. The first model isolates gaps, correlation and evidence bias. The second conditions on actual neural predictions and keeps the discrete sampling law and tie convention. Both calculate expected deployment loss by combining selection probabilities with the corresponding candidate risks.

## E.1 BINARY RISK, EVIDENCE ALIGNMENT AND CONCENTRATION

Let action 0 have deployment risk $r _ { 0 } ,$ and let action 1 have risk $r _ { 1 } = r _ { 0 } + \Delta$ , where $\Delta \geq 0$ . Suppose the estimated risk difference after m labels follows

$$
\widehat { d } _ { m } \sim { \mathcal N } ( \Delta + b , \sigma ^ { 2 } / m ) .\tag{50}
$$

The bias difference $b$ concerns the feedback experiment; the deployment risk gap remains $\Delta .$ . For $\sigma > 0$ , selection of action 1 has probability $\Phi [ - \bar { \sqrt { m } } ( \Delta + b ) / \sigma ]$ . A tie has probability zero. Selection regret is zero for action 0 and $\Delta$ for action 1, giving Equation (8).

Proposition 1 (Direction of the budget response). For $\sigma > 0$ and $\Delta > 0 ,$ , Gaussian selection regret decreases strictly with m when $\Delta + b > 0 ,$ , increases strictly when $\Delta + b < 0$ , and equals $\Delta / 2$ for every m when $\Delta + b = 0$

Proof. Let $\phi = \Phi ^ { \prime }$ denote the standard normal density. Extend m to positive real values and differentiate:

$$
\frac { \partial \mathcal { R } _ { m } } { \partial m } = - \frac { \Delta ( \Delta + b ) } { 2 \sigma \sqrt { m } } \phi \left( \frac { \sqrt { m } ( \Delta + b ) } { \sigma } \right) ,\tag{51}
$$

$$
\frac { \partial \mathcal { R } _ { m } } { \partial b } = - \frac { \Delta \sqrt { m } } { \sigma } \phi \biggl ( \frac { \sqrt { m } ( \Delta + b ) } { \sigma } \biggr ) .\tag{52}
$$

The normal density is positive at every finite argument, establishing the sign. Restricting back to integer budgets preserves the monotonicity. At a zero mean, symmetry gives probability $1 / 2$ □

When $\sigma = 0$ , the evidence difference is deterministic. Regret is $\Delta$ if $\Delta + b < 0$ and zero if $\Delta + b > 0$ . At equality, it is $\Delta$ multiplied by the probability assigned to action 1 by the specified tie rule. If $\Delta = 0$ , all decisions have zero regret. These cases matter for duplicated candidates and for noise whose common component cancels completely.

The result separates the amount of feedback from its ordering. Increasing m concentrates the estimator on its mean. The resulting deployment value depends on the sign of that mean relative to the true risk gap. This same mechanism will appear below when a realized query pool and evaluation pool favor different candidates.

## E.2 A SOLVABLE JOINT CANDIDATE EXPERIMENT

For candidates $j = 0 , \ldots , K - 1$ , let $r _ { j }$ be fixed deployment risks and $b _ { j }$ be fixed feedback biases. One joint observation is

$$
X _ { t j } = r _ { j } + b _ { j } + s \left( { \sqrt { \rho } } Z _ { t 0 } + { \sqrt { 1 - \rho } } Z _ { t , g ( j ) } \right) , \qquad t = 1 , \ldots , m ,\tag{53}
$$

where $s \geq 0 , 0 \leq \rho \leq 1$ , and all common and group-specific standard normals are independent across distinct indices and times. Candidates in the same group are exact copies with equal $r _ { j } , b _ { j }$ and the same residual draw. The common-factor index is distinct from every group index. We minimize the empirical risk score and break ties by the first candidate index.

Writing $\nu _ { j } = r _ { j } + b _ { j }$ and $\tau = s \sqrt { ( 1 - \rho ) / m }$ , the empirical scores have the joint law

$$
\overline { { { X } } } _ { j } = \nu _ { j } + \frac { s \sqrt { \rho } } { \sqrt { m } } Z _ { 0 } + \tau Z _ { g ( j ) } .\tag{54}
$$

The common term cancels from all comparisons. Let $\mathcal { G }$ contain the smallest-index representative of each exact-copy group.

Proposition 2 (Joint winner probabilities). For $\tau > 0 ,$ , representative $j \in \mathcal G$ wins with probability

$$
P _ { j } = \int _ { - \infty } ^ { \infty } \phi ( z ) \prod _ { \ell \in \mathcal { G } \backslash \{ j \} } \Phi \left( \frac { \nu _ { \ell } - \nu _ { j } } { \tau } - z \right) d z .\tag{55}
$$

Nonrepresentative copies have probability zero under first-index ties. Expected deployed risk is $\sum _ { j } \bar { P _ { j } } r _ { j }$ and regret is $\begin{array} { r } { \sum _ { j } P _ { j } ( r _ { j } - \operatorname* { m i n } _ { \ell } \dot { r _ { \ell } } ) } \end{array}$

Proof. Condition on the representative residual $Z _ { g ( j ) } = z$ . Candidate $j$ wins if, for every other group $\ell , Z _ { g ( \ell ) } > z + ( \nu _ { j } - \nu _ { \ell } ) / \tau$ . The residuals are conditionally independent and continuous. Their joint conditional probability is the product in Equation (55); integration over z proves the result. Exact copies always tie, assigning their common win to the first representative. The risk formulas follow by total expectation. □

For $\tau \ = \ 0 .$ , the winner is the first index minimizing $\nu _ { j }$ . For two distinct groups, the singleobservation difference variance is $\sigma ^ { 2 } = 2 s ^ { 2 } ( 1 - \rho )$ , recovering the binary expression. A correlation increase reduces this difference variance at fixed marginal variance. It improves concentration around a correct ordering and strengthens concentration around an incorrect ordering.

Redundancy and complementarity. Appending exact copies changes neither the group competition nor the deployed risk. Complementarity can increase both oracle capacity and selection loss. Consider two equally likely regimes, an incumbent with risk (0.3, 0.3), and a specialist with risk (0.05, 0.3). A four-candidate redundant portfolio contains three copies of that specialist. A complementary four-candidate portfolio contains the incumbent, the original specialist, a second specialist with risks (0.3, 0.05) and a generalist with risks (0.2, 0.2). With $s = 0 . 3 , \rho = 0 . 5$ and zero bias, the oracle risks are 0.175 and 0.05. At m = 1, the expected deployed risks are 0.20029 and 0.14021. The complementary portfolio has better deployed risk and larger selection regret, 0.09021 against 0.02529. $\mathrm { A t } \ m = 1 6 $ , deployed risks approach 0.17505 and 0.05355. Reporting both quantities shows how additional useful capacity can create a larger evidence requirement while still improving the system.

![](images/40c495e6595808f5d5c139b62ab3dc772f1cf0b17bb0dbb40bba6590acdc5f6b.jpg)  
Figure 7: Four-candidate selection under varying accuracy gaps (columns), query bias (rows) and error correlation $\rho$ (curves). Lines integrate Gaussian winner probabilities. Markers and 95% error bars use 100,000 independent score vectors per condition. Canceling shared noise strengthens the query ranking, including when bias makes that ranking wrong. Appendix E gives parameters and neural-network validation.

## E.3 NUMERICAL SOLUTION AND INDEPENDENT SIMULATION

The deterministic solver evaluates Equation (55) in standard-normal coordinates. It computes products of cumulative distribution functions (CDFs) by summing their logarithms. Truncation to [−12, 12] discards at most $2 \Phi ( - 1 2 ) < 3 . 6 \times 1 0 ^ { - \dot { 3 } 3 }$ probability per integral, because the integrand is bounded by ϕ(z). Adaptive quadrature splits the interval at zero and at clipped transition points between candidate means. The winner probabilities are checked for summing to one without renormalization.

The simulation generates raw joint Gaussian score vectors and applies the actual first-index argmin. It neither reads winner probabilities nor samples winners from the theoretical distribution. Gaussian closure permits direct generation of the empirical mean vector for a budget m. The common input to the two programs is the declared parameter grid.

The main factorial contains three risk gaps, four correlations, three bias settings and four budgets, giving 144 conditions. Additional binary, fixed-count portfolio, duplicate and deterministic cases bring the total to 197. Every condition uses 100,000 independent score vectors. Across 740 winnercount comparisons, exact binomial tests use a Bonferroni threshold of $0 . 0 1 / 7 4 0$ . No comparison rejects. The maximum absolute probability discrepancy is 0.00412, and the largest regret discrepancy is 2.58 Monte Carlo standard errors. The binary integral matches its closed form within $5 . 6 \times \bar { 1 0 } ^ { - 1 \bar { 7 } }$ These checks concern the solved common-factor experiment, with separate theoretical curves and sampled decisions.

## E.4 EXACT FINITE-POOL SELECTION PROBABILITIES

For a frozen pair of neural predictors, each eligible query label reveals a signed accuracy difference $D \in \{ - 1 , 0 , 1 \}$ . Let $n _ { + } , n _ { - } , n _ { 0 }$ denote the counts in a fixed pool of size N. A uniform subset of $m \leq N$ examples has counts $\dot { K } _ { + } , K _ { - } , K _ { 0 }$ with multivariate hypergeometric probability in Equation (56). Terms outside $0 ~ \le ~ K _ { c } ~ \le ~ n _ { c }$ or $K _ { + } + K _ { - } \le m$ have probability zero. The score $S _ { m } = K _ { + } - K _ { - }$ determines the empirical rule.

$$
\operatorname* { P r } ( K _ { + } = u , K _ { - } = v ) = { \frac { { \binom { n _ { + } } { u } } { \binom { n _ { - } } { v } } { \binom { n _ { 0 } } { m - u - v } } } { { \binom { n _ { + } + n _ { - } + n _ { 0 } } { m } } } } .\tag{56}
$$

For strict incumbent-default selection, the candidate is chosen if $S _ { m } \ > \ 0 ;$ candidate-default ties change this to $S _ { m } \geq 0$ . For a threshold $h ,$ the strict rule requires $S _ { m } \geq \lfloor m h \rfloor + 1$ . Let $p _ { m }$ be the corresponding summed hypergeometric probability, and let $e _ { I } , e _ { C }$ be the fixed evaluation error counts. Then

$$
\mathbb { E } [ e _ { \mathrm { s e l e c t e d } } ] = ( 1 - p _ { m } ) e _ { I } + p _ { m } e _ { C } ,\tag{57}
$$

$$
\mathbb { E } [ \mathrm { g a i n } ] = \frac { 1 0 0 p _ { m } ( e _ { I } - e _ { C } ) } { | E | } ,\tag{58}
$$

$$
\mathbb { E } [ \mathrm { o r a c l e ~ r e g r e t } ] = \frac { 1 0 0 [ ( 1 - p _ { m } ) e _ { I } + p _ { m } e _ { C } - \operatorname* { m i n } ( e _ { I } , e _ { C } ) ] } { | E | } .\tag{59}
$$

Aggregating error counts and evaluation sizes across batches yields the stream metrics. This exact calculation conditions on the entire frozen pool and is used by the evaluator. The running selector sees acquired labels alone.

Finite-population Gaussian approximation. The pool mean and variance are

$$
\bar { d } = \frac { n _ { + } - n _ { - } } { N } , \qquad v = \frac { n _ { + } + n _ { - } } { N } - \bar { d } ^ { 2 } .\tag{60}
$$

For $N > 1$ , uniform sampling without replacement gives

$$
\mathbb { E } S _ { m } = m \bar { d } , \qquad \mathrm { V a r } ( S _ { m } ) = m \frac { N - m } { N - 1 } v .\tag{61}
$$

A normal tail approximates the probability of crossing the signed-count threshold. The calculation uses the midpoint between adjacent attainable support values as its continuity correction; zero variance and thresholds outside support are handled exactly. The support can have spacing two when only rescue and harm outcomes are present, so a universal half-unit correction is inappropriate. The exact hypergeometric law remains the reference at small budgets.

Query-to-evaluation alignment. For a query pool $P \subseteq Q$ , write

$$
\bar { d } _ { P } - \bar { d } _ { E } = ( \bar { d } _ { P } - \bar { d } _ { Q } ) + ( \bar { d } _ { Q } - \bar { d } _ { E } ) .\tag{62}
$$

The first term describes the realized effect of restricting the acquisition pool; the second records the realized difference between the permanent query and evaluation pools. Both terms can affect a batch decision. Even uniform querying has a finite realized $Q / E$ discrepancy. The sign groups below are evaluator-defined conditional diagnostics, not strata available from unlabeled inputs.

![](images/dab97f74eccc70ec43159782568ab0f7c44bcb21f3997bfb84e2391675252d2f.jpg)

![](images/77686704a0ecd9b24c77d436d5cb48c657a70ba6f8a86cea4a4765e08436541b.jpg)  
† Actual labels = min(cap, pool size); 7 KL pools use 7–9 labels at cap 10.

![](images/b08a56ccf5ffcf1f23e7b2837a8b1adc7ed418e99670b8731dcdd34e08c58f0f.jpg)  
Uniform Q Exact finite population Subset MC (95% MC CI)  
Top-half KL pool Gaussian approximation Fixed BF (panel a)

Figure 8: Exact finite-budget predictions on all 655 PACS batches with incumbent-default ties. Solid lines use the hypergeometric law; dashed lines use its finite-population Gaussian approximation; points and 95% error bars use independent subset simulation. The left panel evaluates the same full population for both pools. The middle and right panels condition on agreement or opposition between realized pool and evaluation advantage signs. Uniform and top-half-KL pools have 404/397 aligned and 153/154 opposed batches, respectively. Seven top-half pools use fewer than ten labels at cap ten. Error bars measure simulation error.

## E.5 NEURAL PREDICTION VALIDATION

We retain all 655 PACS batches from five source-checkpoint stream orders. Two pools are evaluated: the complete eligible query pool, and its label-free top half ranked by symmetric KL. Both sample uniformly without replacement within the specified pool. The grid includes label caps 1, 3, 5, 10 and both tie defaults. Actual usage is min(m, |P|): seven top-half pools contain fewer than ten items and remain in the evaluation.

The exact solver covers 10,480 conditions. The independent simulator samples actual subsets, sums their signed outcomes and applies the rule, giving 52.4 million subset draws. All 10,480 winnerfrequency tests pass a simultaneous 1% Bonferroni test family. The largest aggregate difference is 1.99 Monte Carlo standard errors. Exhaustive enumeration independently verifies another 2,310 small-population configurations. The Gaussian approximation has mean absolute probability error 0.01039 and maximum error 0.17174.

For incumbent-default ties, uniform-query regret on aligned pools falls from 11.35 to 2.62 percentage points as the cap grows from one to ten. On opposed pools it rises from 4.39 to 6.45 points. The corresponding top-half-KL curves change from 9.01 to 1.26 and from 4.50 to 7.02 points. With candidate-default ties, the strong fixed branch already supplies a high starting accuracy; intermediate-budget curves need not be monotone. Fixed BN+flip remains above the aggregate selected gain for every displayed budget. The opposite conditional trends describe how the same concentration mechanism operates within the observed deployment stream.

## F PERSISTENT STATE AND FUTURE DEPLOYMENT LOSS

Retaining an update changes future predictions. Undoing it after the current prediction restores the previous state. This section tracks the difference between these two choices under identical future data and updates. We derive exact recurrences for the residual classifier and feature cache, then bound their effects on future losses.

![](images/74f543cec402ae43b199a037a798a7df7724187cb03e676bc182cbf029e3693b.jpg)

![](images/6137bdc7fddbe7f961f0d8898ba649e6aefe44985578e38f2be4bbc2297b1449.jpg)  
Q: 404; KL: 397 batches  
† Actual labels = min(cap, pool size); 7 KL pools use 7–9 labels at cap 10.

![](images/e09d2e5682a901fe8928c17844ff4dfe0c58af01918b191594ec4530e3d4679c.jpg)  
Q: 153; KL: 154 batches  
Top-half KL pool Gaussian approximation Fixed BF (panel a)

Figure 9: The same finite-budget experiment with BN+flip selected on ties. All-batch gain uses all 20,775 evaluation occurrences across five orders; conditional panels retain their pool-specific aligned and opposed groups. The strong fixed default changes the small-budget response while leaving the exact sampling calculation unchanged. Curves and uncertainty follow the preceding figure.

## F.1 A GENERAL COMMON-UPDATE BOUND

Let two branches leave a decision with states $S ^ { a }$ and $S ^ { I }$ , whose distance is $e = \| S ^ { a } - S ^ { I } \|$ . Both subsequently use the same update maps $U _ { j }$ . Assume $\| U _ { j } ( s ) - U _ { j } ( s ^ { \prime } ) \| \le L _ { j } \| s - s ^ { \prime } \|$ and that the scored loss at future step k is $L _ { \ell , j }$ <sub>k</sub>-Lipschitz in state. Repeated application gives

$$
\| S _ { k } ^ { a } - S _ { k } ^ { I } \| \leq e \prod _ { j < k } L _ { j } , \qquad | \ell _ { k } ( S _ { k } ^ { a } ) - \ell _ { k } ( S _ { k } ^ { I } ) | \leq L _ { \ell , k } e \prod _ { j < k } L _ { j } .\tag{63}
$$

Let $\begin{array} { r } { \xi ( H ) = \sum _ { k = 0 } ^ { H - 1 } [ \ell _ { k } ( S _ { k } ^ { a } ) - \ell _ { k } ( S _ { k } ^ { I } ) ] } \end{array}$ be the cumulative loss difference over H future steps. Summing the preceding bound gives

$$
| \xi ( H ) | \leq e \sum _ { k = 0 } ^ { H - 1 } L _ { \ell , k } \prod _ { j < k } L _ { j } .\tag{64}
$$

For constants $L _ { j } = L _ { U }$ and $L _ { \ell , k } = L _ { \ell } ,$ , this becomes $\textstyle L _ { \ell } e \sum _ { k = 0 } ^ { H - 1 } L _ { U } ^ { k }$ . The common update maps include their future data and labels; changing those inputs between branches changes the comparison. The experiments implement shared inputs explicitly, so the branch-state difference is the manipulated variable.

## F.2 RESIDUAL-HEAD DYNAMICS

Let d be the feature dimension and C the number of classes. The rows of $\ b X _ { s } ~ \in ~ \mathbb R ^ { q \times d }$ are the unit-normalized features of $q = 3$ queried examples. Their targets $R _ { s }$ subtract the fixed source probabilities from the one-hot class labels. The residual classifier has weights $\Theta \in \mathbb { R } ^ { d \times C }$ . With learning rate η and regularization strength λ, a squared-loss update is

$$
\begin{array} { r } { U _ { s } ( \Theta ) = M _ { s } \Theta + B _ { s } , } \end{array}\tag{65}
$$

$$
M _ { s } = ( 1 - \eta \lambda ) I - \eta X _ { s } ^ { \top } X _ { s } / q , \qquad B _ { s } = \eta X _ { s } ^ { \top } R _ { s } / q .\tag{66}
$$

The implemented values are $\eta \ : = \ : 0 . 5$ and $\lambda \ : = \ : 0 . 1$ . Fixed-source targets make the affine term identical on both branches. Subtraction therefore gives the exact recurrence

$$
\Delta \Theta _ { s + 1 } = M _ { s } \Delta \Theta _ { s } .\tag{67}
$$

Proposition 3 (Head contraction and projected output). For unit-norm rows of $X _ { s }$ and the specified parameters, $\| \Delta \Theta _ { s + 1 } \| _ { F } \le 0 . 9 5 \| \Delta \Theta _ { s } \| _ { F }$ . For any unit-norm future evaluation feature $x ,$ the logit difference after k common updates is

$$
d _ { k } = x M _ { t + k - 1 } \cdot \cdot \cdot M _ { t } \Delta \Theta _ { t } , \qquad \| d _ { k } \| _ { 2 } \leq 0 . 9 5 ^ { k } \| \Delta \Theta _ { t } \| _ { F } .\tag{68}
$$

Proof. The matrix $X _ { s } ^ { \top } X _ { s } / q$ is positive semidefinite and has trace one. Its eigenvalues belong to $[ 0 , 1 ]$ . Equation (66) gives eigenvalues of $M _ { s }$ in [0.45, 0.95], so its spectral norm is at most 0.95. Submultiplicativity proves contraction and the norm envelope. Iterating the exact difference recurrence and applying the linear readout gives the displayed $d _ { k }$ □

The product tracks the direction of the update difference. A large parameter difference can have a small future output effect when future features have little projection onto that direction. The implementation applies the rank-q update to the difference without constructing a dense d×d matrix. Independently updating both explicit head states verifies the recurrence.

## F.3 FEATURE-CENTERING CACHE

Let $m _ { s }$ be the stored feature mean, $\overline { { f } } _ { Q , s }$ the mean of the selected query features, and $\rho$ the weight retained from the previous mean. The cache update is

$$
m _ { s + 1 } = \rho m _ { s } + ( 1 - \rho ) \overline { { f } } _ { Q , s } , \qquad \rho = 0 . 9 ,\tag{69}
$$

The classifier subtracts αm $\mathbf { \Sigma } _ { s } W ^ { \top }$ from its logits, where $W$ is the fixed source classifier and $\alpha =$ 0.05 controls the correction strength. The two model copies use the same query features. Their differences obey

$$
\Delta m _ { s + 1 } = \rho \Delta m _ { s } , \qquad d _ { k } = - \alpha \rho ^ { k } \Delta m _ { t } W ^ { \top } .\tag{70}
$$

Both identities follow by subtracting the shared affine updates. The continuation scale is α∥ $\lvert \Delta m _ { t } W ^ { \top } \rvert \rvert _ { 2 }$ , the classifier-visible state difference. A component in the null space of $W$ has zero prediction effect at every horizon, despite contributing to the feature-state distance.

The cache operates at the classifier input. Its state is an actual persistent feature mean, distinct from the internal running statistics of backbone batch-normalization layers. Both this cache and the residual head change future logits while the backbone stays fixed.

## F.4 DIRECTIONAL CROSS-ENTROPY BOUNDS

For $\begin{array} { r } { \ell _ { y } ( z ) = \log \sum _ { c } e ^ { z _ { c } } - z _ { y } } \end{array}$ , the gradient is $p - e _ { y }$ and the Hessian is dia $\boldsymbol { \mathrm { g } } ( \boldsymbol { p } ) - \boldsymbol { p } \boldsymbol { p } ^ { \intercal }$ . For any v, its quadratic form is the variance of the coordinates $v _ { c }$ under $c \sim p$ . The range bound for a bounded variable gives

$$
v ^ { \top } \nabla ^ { 2 } \ell _ { y } ( z ) v \leq \frac { ( \operatorname* { m a x } _ { c } v _ { c } - \operatorname* { m i n } _ { c } v _ { c } ) ^ { 2 } } { 4 } \leq \frac { \| v \| _ { 2 } ^ { 2 } } { 2 } .\tag{71}
$$

The last step uses $( v _ { i } - v _ { j } ) ^ { 2 } \leq 2 ( v _ { i } ^ { 2 } + v _ { j } ^ { 2 } )$ . The Hessian is positive semidefinite and has operator norm at most $1 / 2$

Taylor’s theorem with integral remainder now gives, for every $z , d ,$ y,

$$
\ell _ { y } ( z + d ) - \ell _ { y } ( z ) = ( p - e _ { y } ) ^ { \top } d + \int _ { 0 } ^ { 1 } ( 1 - u ) d ^ { \top } \nabla ^ { 2 } \ell _ { y } ( z + u d ) d u ,\tag{72}
$$

$$
\begin{array} { r } { 0 \leq \ell _ { y } ( z + d ) - \ell _ { y } ( z ) - ( p - e _ { y } ) ^ { \top } d \leq \frac { 1 } { 4 } \| d \| _ { 2 } ^ { 2 } . } \end{array}\tag{73}
$$

Summing over future evaluation examples yields the interval tested in the state experiments. The propagated d comes from the known update operators and initial state difference. Evaluation labels determine the directional gradient, making this a retrospective mechanism calculation. It can distinguish positive and negative loss effects that a magnitude bound treats identically.

A label-free envelope follows from $\| p - e _ { y } \| _ { 2 } \leq { \sqrt { 2 } } .$ . For one mean-loss score per future batch, head continuation satisfies

$$
\sum _ { k = 0 } ^ { H - 1 } | \Delta \bar { \ell } _ { k } | \leq \sqrt { 2 } \| \Delta \Theta _ { t } \| _ { F } \sum _ { k = 0 } ^ { H - 1 } 0 . 9 5 ^ { k } .\tag{74}
$$

The cache envelope replaces the initial norm by α∥ $\Delta m _ { t } W ^ { \top } \lVert _ { 2 }$ and 0.95 by $\rho .$ For image-sum rather than mean-batch losses, each term is multiplied by its evaluation batch size.

## F.5 PREDICTION CHANGES AND ZERO-ONE LOSS

Let $m ( z )$ be the difference between the largest and second-largest baseline logits. Each logit moves by at most $\| d \| _ { \infty }$ , so the winning class cannot change if $m ( z ) { \bar { \mathbf { \alpha } } } > 2 \| d \| _ { \infty }$ . For a deterministic shared tie rule,

$$
| \mathbf { 1 } \{ \arg \operatorname* { m a x } ( z + d ) \neq y \} - \mathbf { 1 } \{ \arg \operatorname* { m a x } z \neq y \} | \leq \mathbf { 1 } \{ m ( z ) \leq 2 \| d \| _ { \infty } \} .\tag{75}
$$

The bound concerns the set where a prediction could change. Corrected errors, newly introduced errors and neutral prediction changes are measured separately. Contraction of state or cross-entropy does not fix the sign of an accuracy change, because the decision boundary depends on individual margins and perturbation directions.

## F.6 PAIRED UPDATE DIAGNOSTIC

An update can change both the current choice and future predictions. We separate these effects with two model copies. One keeps the current update; the other undoes it. Both then receive the same inputs, labels and updates. We test a learned linear classifier and a cache that adjusts scores using a running feature mean.

The parameter difference contracts by at most 0.95 per classifier step and 0.9 per cache step, predicting future logit differences d. For baseline logits z, let $p = \operatorname { s o f t m a x } ( z ) , e _ { y }$ be the true-class unit vector, and $\begin{array} { r } { \ell _ { y } ( z ) = \log \sum _ { c } e ^ { z _ { c } } - z _ { y } } \end{array}$ be cross-entropy. Then

$$
\begin{array} { r } { ( p - e _ { y } ) ^ { \top } d \le \ell _ { y } ( z + d ) - \ell _ { y } ( z ) \le ( p - e _ { y } ) ^ { \top } d + \frac 1 4 \| d \| _ { 2 } ^ { 2 } . } \end{array}\tag{76}
$$

Define the loss change ∆CE, lower bound $L = ( p - e _ { y } ) ^ { \top } d$ and width $\begin{array} { r } { R = \frac { 1 } { 4 } \| d \| _ { 2 } ^ { 2 } } \end{array}$ . For $R > 0$ $( \Delta \mathrm { C E } - L ) / R \in [ \bar { 0 , 1 } ]$ (Figure 10).

Retaining an update improves current accuracy by 0.23 points and changes next-batch accuracy by −0.20 points. Undoing it leaves future predictions unchanged. Immediate gains and lasting effects need separate measurement.

![](images/c0a4fe4e432bf22b3a784de8fad21c486a4ca803c7782219c48134154a9a9a80.jpg)

![](images/3708f1beb3a36638fefa3e125579f18517af815e3aa3916a3c138555b2b3a2fd.jpg)

![](images/61ff47432146584b174219a4ff078e6446e91e56eb0358f827ab1f4940733054.jpg)

![](images/8b655062caf3d8ed721a0e6427c663d9709146849e6eff62c7ed9202c61ce7a4.jpg)  
Figure 10: Retain versus undo an update. (a,b) Immediate and future accuracy changes with order SD. (c,d) Normalized loss changes $( \mathrm { { \bar { \Delta } C E { - } } } L ) / R \colon$ quartiles, medians and full-range whiskers within the predicted interval [0, 1].

## F.7 UPDATE TIMING AND EXPERIMENTAL VERIFICATION

At a chosen batch t, we score an intervention and either retain or undo its update. We call this the branch point. Both copies then receive the same future query features, residual targets, and update order. Each future batch is scored before its update. Thus H future scored batches contain $H - 1$ common updates before the final score. The baseline copy skips only the update at t. The reference stream continues its prescribed classifier and cache updates, resetting at domain boundaries.

Each of five complete PACS orders acquires 393 distinct query labels. The model copies use these same labels. Horizons $H = 1 , 4 .$ , 16 retain 128,119,83 valid branch points per order. Their future windows overlap. The means summarize paired update effects, and the stream-order standard deviations describe sensitivity to arrival order. Immediate means vary with the horizon because the eligible branch points differ.

Source logits reconstructed from the released classifier agree with the stored predictions within 1.1× $1 0 ^ { - 6 }$ . Explicit branch updates and propagated state differences agree within $\phantom { + } 1 . 3 \times 1 0 ^ { - 1 5 }$ . All crossentropy observations satisfy Equation (73), and all hard-prediction changes satisfy Equation (75). Changing evaluation labels leaves learned state unchanged. Output rollback produces exactly zero future logit and loss difference.

The residual-head intervention improves immediate accuracy by 0.23 percentage points at one-batch branch points, while next-batch accuracy changes by −0.20 points and next-batch cross-entropy by −0.00059. Immediate and future cross-entropy changes have opposite signs in 20.16% of one-batch comparisons. The feature cache produces smaller persistent changes. Together these observation establish that a local intervention’s immediate benefit, future accuracy and future cross-entropy are distinct quantities, connected by the update operator and the direction of the parameter change.

## G EXPERIMENTAL POPULATIONS, FEEDBACK AND OUTPUTS

The experiments measure fixed intervention capacity, its realization through output selection, and its expansion through learned states. Each comparison retains its model, evaluation population, output timing and annotation budget.

Corruption populations. CIFAR10-C and CIFAR100-C use all 15 severity-5 corruptions and three source ResNet18 checkpoints. Each corruption contains 10,000 images. Batches contain 64 images, including a final batch of 16. The total is 150,000 images per dataset and checkpoint. A permanent split assigns 5,000 base-image indices to the query population Q and 5,000 to the evaluation population E. The role of an underlying image is unchanged across corruptions and checkpoints. Each corruption is a separate deployment: selection statistics and annotation history reset at its boundary. An annotation purchased in one deployment is charged again if queried in another.

Natural domain shifts. The CLIP ViT-L/14 experiments include all four domains of PACS, OfficeHome and VLCS, with 9,991, 15,588 and 7,510 image instances, respectively. Their permanent query/evaluation counts are 5,070/4,921, 7,801/7,787 and 3,751/3,759. The split is assigned to encoded-content groups, so repeated occurrences remain on the same side. All dataset instances and their original annotations are retained. OfficeHome has 15,176 content groups, including 87 groups whose 178 occurrences have conflicting directory labels. VLCS contains 7,506 groups and no conflicting labels. PACS has no repeated content in this enumeration. Annotation costs count queried instances. PACS and VLCS identities are recorded during inference. OfficeHome identities follow the deterministic class/file order of its saved four-readout stream and are checked against its stored labels.

Timing and aggregation. Fixed whole-dataset scores include every example and use zero target labels. Controlled comparisons score the common E population after acquiring current-batch labels from Q. These labels choose among predictions already formed. In the online-classifier experiment, they train the classifier after current predictions have been scored. That update affects later batches. CIFAR scores pool equal-size corruption populations. Vision-language model (VLM) results report both equal-domain and image-pooled accuracy, with the equal-domain mean as primary. Standard deviations cover three CIFAR checkpoints or five natural-domain stream orders, as indicated in each table.

Query rules and defaults. Empirical selection minimizes queried classification errors and keeps the designated default on ties. Uniform querying samples three available Q examples. Disagreement-first querying prioritizes inputs with different candidate class predictions, in random order. Symmetric-KL querying ranks disagreement between predicted class probabilities. The complete CIFAR experiment sums the six pairwise scores; the four-readout CLIP experiment uses their mean. Both give the same ranking for a fixed candidate set. The StatA and convex-readout experiments use the largest pairwise score.

The fixed default is BN+flip on CIFAR and the eight-view readout on CLIP. PORTAL uses total prior mass 1: it assigns 0.75 to candidate-predicted classes and spreads the remaining 0.25 over all classes (Equation (86)). The label price is 0.002, with at most 12 queries per batch. Three query credits arrive per batch. CIFAR caps the stored balance at 12; the VLM experiments carry all unused credits. Tables report actual label use.

## H COMPLETE CORRUPTION RESULTS

## H.1 CAPACITY OF FIXED NORMALIZATION AND VIEW INTERVENTIONS

The four CIFAR candidates share one source network. I uses source normalization statistics; B uses current-batch statistics. F averages the original and horizontally flipped source logits. BF averages these two views using batch statistics. B and BF leave the source normalization buffers unchanged. The four candidates require four forward passes, with each view shared by the relevant readouts. A fixed I or B output requires one forward pass, and F or BF requires two.

Tables 8 and 9 retain every corruption and checkpoint. BF gives 79.9218% and 54.7544% on CIFAR10-C and CIFAR100-C, respectively. Its fixed increments over I are 22.2178 and 21.5864 percentage points. These gains precede label acquisition. The B and F columns separate normalization from views, while local variation leaves room for different batch decisions.

Short names denote the standard noise, blur, elastic-transformation and JPEG-compression corruptions. All scores include the final 16 images.  
Table 8: Complete CIFAR10-C fixed-intervention accuracy (%). Each checkpoint evaluates all 10,000 images in every severity-5 corruption; all columns use zero feedback.
<table><tr><td></td><td colspan="3">Checkpoint 0</td><td colspan="4">Checkpoint 1</td><td colspan="4">Checkpoint 2</td><td colspan="2"></td></tr><tr><td>Corruption</td><td>I</td><td>B</td><td>F</td><td>BF</td><td>I</td><td>B</td><td></td><td>F BF</td><td></td><td>I</td><td>B</td><td>F BF</td></tr><tr><td>Gaussian</td><td>22.1366.36 22.0368.41 27.73 67.01 28.4869.52 28.0667.3728.30 69.90</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Shot</td><td>29.2567.61 29.2670.4334.2669.00 34.8171.78 36.0669.52 36.3272.13</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Impulse</td><td>22.5358.86 22.9762.09 22.02 60.92 22.04 63.55 29.5660.8530.3863.89</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Defocus</td><td>54.5487.3455.3188.6155.8887.7356.4888.9359.7487.7060.1988.99</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Glass</td><td>55.10 65.17 56.0867.3250.14 64.8851.6367.0650.8365.94 51.9967.76</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Motion</td><td>67.4586.04 68.18 87.02 66.11 86.09 67.04 87.15 68.5886.5569.30 87.61</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Zoom</td><td>64.3686.6465.6388.1862.5087.6463.8388.6867.5487.9768.1488.54</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Snow</td><td>78.2480.8879.0982.1977.5281.5678.2283.2878.1481.8678.9283.68</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Frost</td><td>69.93 79.57 71.02 80.98 65.3180.72 66.5981.68 65.7680.9566.51 82.49</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Fog</td><td>74.2685.3275.2486.1673.2684.8473.9086.1274.9286.0975.8387.09</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Brightness 91.29 90.75 92.36 91.23 91.43 90.60 92.17 91.11 91.79 91.15 92.45 91.80</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Contrast</td><td>35.4587.0535.4287.5432.0187.2032.2987.8434.2587.0234.4487.69</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Elastic</td><td>77.2076.4978.5678.2675.8975.8277.5277.7576.7677.1477.9978.93</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pixelate</td><td>46.5579.4146.4780.7545.7479.1146.1180.3246.8979.3047.0580.58</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>JPEG</td><td>73.3971.3174.8373.3672.4671.3473.4673.3873.8772.8175.4174.72</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mean</td><td>57.44 77.92 58.16 79.50 56.82 78.30 57.64 79.8858.85 78.81 59.5580.39</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9: Complete CIFAR100-C fixed-intervention accuracy (%). Each checkpoint evaluates all 10,000 images in every severity-5 corruption; all columns use zero feedback.
<table><tr><td colspan="4">Checkpoint 0</td><td colspan="4">Checkpoint 1</td><td colspan="4">Checkpoint 2</td><td colspan="2"></td></tr><tr><td>Corruption</td><td>I</td><td>B</td><td>F</td><td>BF</td><td>I</td><td>B</td><td></td><td>F BF</td><td></td><td>I</td><td>B</td><td>F BF</td></tr><tr><td>Gaussian</td><td></td><td>10.2139.7610.1941.13</td><td></td><td></td><td></td><td>39.53 39.58</td><td></td><td>9.8341.0912.13 38.3112.0140.48</td><td></td><td></td><td></td><td></td></tr><tr><td>Shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>11.9941.11 12.2142.8511.0340.1911.3842.0513.41 40.2013.5642.22</td><td></td><td></td><td></td><td></td></tr><tr><td>Impulse</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>6.64 34.72 6.77 36.32 6.63 33.61 6.62 35.27 8.23 33.57 8.23 35.83</td><td></td><td></td><td></td><td></td></tr><tr><td>Defocus</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>34.80 66.1835.00 67.2334.49 66.0834.8566.98 35.22 65.67 35.47 66.23</td><td></td><td></td><td></td><td></td></tr><tr><td>Glass</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>20.1942.6619.9343.5619.0642.6919.4943.6619.4841.91 19.5643.46</td><td></td><td></td><td></td><td></td></tr><tr><td>Motion</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>43.8963.3344.8164.2144.7863.8145.5364.6243.6062.9944.5263.97</td><td></td><td></td><td></td><td></td></tr><tr><td>Zoom</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>41.6564.9342.1565.6442.4464.7642.7965.7041.7064.6642.6365.41</td><td></td><td></td><td></td><td></td></tr><tr><td>Snow</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>46.3853.8746.9255.1346.0553.6547.0654.9046.5553.4147.1054.40</td><td></td><td></td><td></td><td></td></tr><tr><td>Frost</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>34.0553.5934.4554.6333.9353.6334.3654.4334.6153.7635.2954.80</td><td></td><td></td><td></td><td></td></tr><tr><td>Fog</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>40.0157.8540.6358.8941.4458.2242.2259.1941.7157.9342.3459.28</td><td></td><td></td><td></td><td></td></tr><tr><td>Brightness</td><td>67.44 68.04 68.37 68.69 67.32 68.07 68.30 69.04 67.15 68.17 67.88 69.23</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Contrast</td><td>17.79 64.2417.87 64.7518.61 64.62 18.66 65.3319.00 64.0819.00 65.03</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Elastic</td><td>50.73 54.26 51.6355.41 50.11 53.5551.3554.7750.1354.4951.2555.68</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="4">Checkpoint 0</td><td colspan="4">Checkpoint 1</td><td colspan="4">Checkpoint 2</td><td></td></tr><tr><td>Corruption</td><td>I</td><td>B</td><td></td><td>F</td><td>BF</td><td>I</td><td>B</td><td>F</td><td>BF</td><td>I</td><td>B</td><td>F</td><td>BF</td></tr><tr><td>Pixelate</td><td></td><td></td><td></td><td></td><td>22.5957.24 22.8657.8724.69</td><td></td><td></td><td></td><td>56.6724.7157.8527.4856.4627.7457.48</td><td></td><td></td><td></td><td></td></tr><tr><td>JPEG</td><td></td><td></td><td></td><td></td><td>45.2845.6545.8746.4444.4144.7044.9046.1044.0045.4644.9146.72</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mean</td><td></td><td></td><td></td><td></td><td>32.91 53.83 33.31 54.85 32.9753.5933.47 54.73 33.63 53.4034.10 54.68</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## H.2 SELECTION ON A COMMON EVALUATION POPULATION

Table 10 evaluates the entire portfolio on the same held-aside image roles. The fixed mean-logit rule averages the four candidate logits and needs no annotations. The batch oracle chooses its action from all current E labels and provides an evaluation-only upper reference. Each empirical rule spends 471 labels per corruption, including three on the final 16-image batch, for 7,065 labels per suite.

PORTAL attains 79.2587% on CIFAR10-C with 612.3 labels per suite and 53.8680% on CIFAR100- C with 381.7 labels, on average. Symmetric-KL empirical selection gives 79.2022% and 53.7604%, using 7,065 labels in either dataset. PORTAL achieves similar accuracy with far fewer labels. Fixed BF has the highest deployable accuracy here: 79.6462% and 54.3213% on the same E populations.

The six checkpoint tables retain every common-E corruption outcome, including individual fixed actions, the batch oracle, and actual joint-query counts.

Table 10: Complete-suite comparison on the common evaluation population. Accuracy is averaged over three checkpoints; the accompanying value is the sample SD. Label counts are per 15- corruption suite.

<table><tr><td>Dataset</td><td>Method</td><td>Accuracy (%) ∆ BF (pp)</td><td></td><td>Labels</td></tr><tr><td>CIFAR10-C</td><td>Fixed BF</td><td> $7 9 . 6 4 6 \pm 0 . 5 6 1$ </td><td>0.000</td><td>0.0</td></tr><tr><td>CIFAR10-C</td><td></td><td>Mean logits 76.548 ± 1.295</td><td>-3.099</td><td>0.0</td></tr><tr><td>CIFAR10-C</td><td></td><td>Uniform ERM 78.275 ± 0.583</td><td>-1.371</td><td>7065.0</td></tr><tr><td>CIFAR10-C</td><td>Disagreement ERM 78.361 ± 0.591</td><td></td><td>-1.285</td><td>7065.0</td></tr><tr><td>CIFAR10-C</td><td>Symmetric-KL ERM 79.202 ± 0.499</td><td></td><td>-0.444</td><td>7065.0</td></tr><tr><td>CIFAR10-C</td><td></td><td>PORTAL 79.259 ± 0.506</td><td>-0.388</td><td>612.3</td></tr><tr><td>CIFAR10-C</td><td></td><td>Batch oracle 81.135 ± 0.449</td><td></td><td>1.48975000.0</td></tr><tr><td>CIFAR100-C</td><td></td><td>Fixed BF 54.321 ± 0.052</td><td>0.000</td><td>0.0</td></tr><tr><td>CIFAR100-C</td><td></td><td>Mean logits 52.356 ± 0.206</td><td>-1.966</td><td>0.0</td></tr><tr><td>CIFAR100-C</td><td></td><td>Uniform ERM 52.788 ± 0.082</td><td>-1.533</td><td>7065.0</td></tr><tr><td></td><td>CIFAR100-C Disagreement ERM 52.802 ± 0.232</td><td></td><td>-1.519</td><td>7065.0</td></tr><tr><td></td><td>CIFAR100-C Symmetric-KL ERM</td><td> $5 3 . 7 6 0 \pm 0 . 1 1 9$ </td><td>-0.561</td><td>7065.0</td></tr><tr><td>CIFAR100-C</td><td>PORTAL</td><td> $5 3 . 8 6 8 \pm 0 . 2 3 1$ </td><td>-0.453</td><td>381.7</td></tr><tr><td>CIFAR100-C</td><td>Batch oracle</td><td> $5 5 . 9 5 2 \pm 0 . 1 4 9$ </td><td></td><td>1.63075000.0</td></tr></table>

Table 11: CIFAR10-C, checkpoint 0: accuracy (%) on the same 5,000 evaluation images per corruption. U, D and K are uniform, disagreement-first and symmetric-KL empirical selection; J is joint selection; O is the batch oracle. The last column counts J labels, with a suite total in the Mean row.
<table><tr><td>Corruption</td><td>I</td><td>B</td><td>F</td><td>BF Mean</td><td>U</td><td>D</td><td>K</td><td>J</td><td>0 QJ</td></tr><tr><td>Gaussian</td><td></td><td></td><td></td><td></td><td></td><td></td><td>21.78 65.92 21.86 67.56 56.70 65.0865.5867.5867.56 68.76</td><td></td><td>53</td></tr><tr><td>Shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td>29.0468.1229.14 70.4863.0468.6268.8470.2470.2471.42</td><td></td><td>50</td></tr><tr><td>Impulse</td><td></td><td></td><td></td><td></td><td></td><td></td><td>23.0657.7423.0461.3855.3457.9658.0461.2861.3862.24</td><td></td><td>54</td></tr><tr><td>Defocus</td><td></td><td></td><td></td><td></td><td></td><td></td><td>54.62 87.1454.90 88.2683.2086.8087.3488.2687.9689.06</td><td></td><td>44</td></tr><tr><td>Glass</td><td></td><td></td><td></td><td></td><td></td><td></td><td>54.6065.0455.5867.4465.8665.0864.7466.2465.72 68.72</td><td></td><td>72</td></tr><tr><td>Motion</td><td></td><td></td><td></td><td></td><td></td><td></td><td>67.5086.2068.22 86.6484.72 84.9485.6486.4286.3487.72</td><td></td><td>42</td></tr><tr><td>Zoom</td><td></td><td></td><td></td><td></td><td></td><td></td><td>64.42 86.5865.32 88.18 85.10 87.04 86.3888.0887.96 88.78</td><td></td><td>39</td></tr><tr><td>Snow</td><td></td><td></td><td></td><td></td><td></td><td></td><td>77.5680.7078.44 81.6282.8880.8480.18 79.9481.0884.14</td><td></td><td>27</td></tr><tr><td>Frost</td><td></td><td></td><td></td><td></td><td></td><td></td><td>69.32 79.02 69.92 80.8081.34 79.38 78.14 80.00 79.74 82.02</td><td></td><td>44</td></tr><tr><td>Fog</td><td></td><td></td><td></td><td></td><td></td><td></td><td>73.7084.2673.8885.2484.9084.2483.2484.1884.0486.40</td><td></td><td>60</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Brightness 91.10 90.50 92.06 91.06 93.10 90.98 91.18 91.12 90.96 93.98</td><td></td><td>10</td></tr></table>

Table 11: CIFAR10-C, checkpoint 0: accuracy (%) on the same 5,000 evaluation images per corruption (continued).
<table><tr><td>Corruption</td><td>I</td><td>B</td><td>F</td><td>BF Mean</td><td>U</td><td>D</td><td>K</td><td>J</td><td>0</td><td> $Q _ { J }$ </td></tr><tr><td>Contrast</td><td></td><td></td><td></td><td>35.2086.84 35.2087.3284.16 87.32 87.0887.3287.2688.00</td><td></td><td></td><td></td><td></td><td></td><td>59</td></tr><tr><td>Elastic</td><td></td><td></td><td></td><td>77.5076.5478.52 78.2080.4078.3078.04 77.7878.1881.90</td><td></td><td></td><td></td><td></td><td></td><td>34</td></tr><tr><td>Pixelate</td><td></td><td></td><td></td><td>45.9479.5645.7080.5672.3879.2479.4880.5280.5681.64</td><td></td><td></td><td></td><td></td><td></td><td>30</td></tr><tr><td>JPEG</td><td></td><td></td><td></td><td>73.4671.0474.7672.6675.7472.6272.6673.0473.0677.38</td><td></td><td></td><td></td><td></td><td></td><td>31</td></tr><tr><td>Mean</td><td></td><td></td><td></td><td>57.25 77.6857.77 79.1676.5977.9077.77 78.8078.8080.81 649</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 12: CIFAR10-C, checkpoint 1: accuracy (%) on the same 5,000 evaluation images per corruption. U, D and K are uniform, disagreement-first and symmetric-KL empirical selection; J is joint selection; O is the batch oracle. The last column counts J labels, with a suite total in the Mean row.

<table><tr><td>Corruption</td><td>I B</td><td></td><td>F</td><td>BF Mean</td><td>U</td><td>D</td><td>K</td><td>J</td><td>O QJ</td></tr><tr><td>Gaussian</td><td></td><td></td><td></td><td></td><td></td><td></td><td>27.50 66.08 28.64 69.2856.76 66.60 67.16 69.24 68.80 70.14</td><td></td><td>58</td></tr><tr><td>Shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td>33.60 68.32 34.50 71.3061.78 67.86 70.12 71.20 70.94 72.00</td><td></td><td>55</td></tr><tr><td>Impulse</td><td></td><td></td><td></td><td></td><td></td><td></td><td>22.04 60.30 22.04 62.82 50.3660.20 60.86 62.48 62.74 63.92</td><td></td><td>62</td></tr><tr><td>Defocus</td><td></td><td></td><td></td><td></td><td></td><td></td><td>55.72 87.28 56.48 88.6682.80 87.38 87.94 88.64 88.66 89.26</td><td></td><td>34</td></tr><tr><td>Glass</td><td></td><td></td><td></td><td></td><td></td><td></td><td>49.28 64.1850.62 66.74 62.2863.88 64.1864.86 66.34 67.72</td><td></td><td>34</td></tr><tr><td>Motion</td><td></td><td></td><td></td><td></td><td></td><td></td><td>66.32 85.60 66.86 86.44 83.02 85.74 85.52 86.32 86.34 87.46</td><td></td><td>28</td></tr><tr><td>Zoom</td><td></td><td></td><td></td><td></td><td></td><td></td><td>62.70 87.5864.1088.5884.3887.3888.1088.4287.64 89.44</td><td></td><td>33</td></tr><tr><td>Snow</td><td></td><td></td><td></td><td></td><td></td><td></td><td>76.8681.1877.7483.3683.1082.0081.1881.4081.9084.62</td><td></td><td>39</td></tr><tr><td>Frost</td><td></td><td></td><td></td><td></td><td></td><td></td><td>64.54 80.42 65.94 81.5879.60 79.38 80.2080.78 81.32 82.84</td><td></td><td>36</td></tr><tr><td>Fog</td><td></td><td></td><td></td><td></td><td></td><td></td><td>72.80 84.1873.46 85.5084.24 84.0884.2885.02 85.30 86.42</td><td></td><td>29</td></tr><tr><td>Brightness 91.06 90.24 91.90 90.58 92.90 90.90 91.10 90.92 90.64 93.84</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>16</td></tr><tr><td>Contrast</td><td></td><td></td><td></td><td></td><td></td><td></td><td>32.3087.10 32.40 87.7881.8686.22 87.0087.7887.5288.56</td><td></td><td>46</td></tr><tr><td>Elastic</td><td></td><td></td><td></td><td></td><td></td><td></td><td>75.9676.12 77.7477.9679.66 77.72 77.3277.1077.3881.26</td><td></td><td>25</td></tr><tr><td>Pixelate</td><td></td><td></td><td></td><td></td><td></td><td></td><td>44.60 78.02 45.04 79.28 70.76 77.68 78.02 79.26 79.10 80.32</td><td></td><td>27</td></tr><tr><td>JPEG</td><td></td><td></td><td></td><td></td><td></td><td></td><td>72.10 70.3073.06 72.92 74.98 72.72 72.42 72.28 72.94 76.40 34</td><td></td><td></td></tr><tr><td>Mean</td><td></td><td></td><td></td><td></td><td></td><td></td><td>56.49 77.79 57.37 79.52 75.23 77.98 78.36 79.05 79.17 80.95 556</td><td></td><td></td></tr></table>

Table 13: CIFAR10-C, checkpoint 2: accuracy (%) on the same 5,000 evaluation images per corruption. U, D and K are uniform, disagreement-first and symmetric-KL empirical selection; J is joint selection; O is the batch oracle. The last column counts J labels, with a suite total in the Mean row.
<table><tr><td>Corruption</td><td>I</td><td>B</td><td>F</td><td>BF Mean</td><td></td><td>U</td><td>D</td><td>K</td><td>J</td><td>O</td><td>QJ</td></tr><tr><td>Gaussian</td><td></td><td></td><td></td><td></td><td>28.14 66.7028.4069.3462.2066.70 67.0469.24 69.1670.28</td><td></td><td></td><td></td><td></td><td></td><td>56</td></tr><tr><td>Shot</td><td></td><td></td><td></td><td></td><td>35.8869.1035.84 71.8067.14 70.26 70.9271.64 71.3872.74</td><td></td><td></td><td></td><td></td><td></td><td>66</td></tr><tr><td>Impulse</td><td></td><td></td><td></td><td></td><td>29.9860.4830.4464.0457.8062.3262.5063.2463.1464.96</td><td></td><td></td><td></td><td></td><td></td><td>64</td></tr><tr><td>Defocus</td><td>59.5887.24 60.00 88.74 85.7687.1687.8088.6688.34 89.38</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>43</td></tr><tr><td>Glass</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>50.66 65.82 51.42 67.44 64.54 65.18 64.60 66.42 67.02 68.76</td><td></td><td></td><td>47</td></tr><tr><td>Motion</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>68.98 86.62 69.86 87.68 85.48 86.28 86.52 87.66 87.14 88.40</td><td></td><td></td><td>38</td></tr><tr><td>Zoom</td><td>67.84 87.8068.16 88.52 86.2887.5288.10 88.5088.20 89.56</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>40</td></tr><tr><td>Snow</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>77.42 81.5878.42 83.94 84.26 82.64 81.4082.18 82.34 85.26</td><td></td><td></td><td>38</td></tr><tr><td>Frost</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>65.22 80.7665.76 82.5080.34 80.32 80.5281.54 82.22 83.30</td><td></td><td></td><td>37</td></tr><tr><td>Fog</td><td></td><td></td><td></td><td></td><td>74.0885.5674.76 86.6285.4885.4884.5885.3885.34 87.72</td><td></td><td></td><td></td><td></td><td></td><td>55</td></tr><tr><td>Brightness 91.32 91.0892.20 91.74 93.60 91.96 92.02 91.88 91.56 94.56</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>13</td></tr><tr><td>Contrast</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>34.22 86.82 34.12 87.4284.94 86.5687.1087.42 87.42 88.08</td><td>47</td></tr><tr><td>Elastic</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>77.06 77.02 77.96 78.94 80.4078.56 77.94 78.3078.76 81.56</td><td>29</td></tr><tr><td>Pixelate</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>46.38 79.5046.32 80.60 72.7878.82 79.04 80.5680.62 81.76</td><td></td><td>33</td></tr><tr><td>JPEG</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>73.66 72.8875.38 74.5876.30 74.44 74.20 73.78 74.4078.40</td><td>26</td></tr><tr><td>Mean</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>58.69 78.60 59.27 80.26 77.82 78.95 78.95 79.76 79.80 81.65 632</td><td></td><td></td></tr></table>

Table 14: CIFAR100-C, checkpoint 0: accuracy (%) on the same 5,000 evaluation images per corruption. U, D and K are uniform, disagreement-first and symmetric-KL empirical selection; J is joint selection; O is the batch oracle. The last column counts J labels, with a suite total in the Mean row.
<table><tr><td>Corruption</td><td>I</td><td>B</td><td>F</td><td>BF Mean</td><td>U</td><td>D</td><td>K</td><td>J</td><td>OQJ</td></tr><tr><td>Gaussian</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>10.7039.6610.54 40.84 32.32 39.46 39.8440.3440.8441.88</td><td></td></tr><tr><td>Shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>12.2641.0812.5242.6834.5841.0640.7241.9442.3644.00</td><td>52 63</td></tr><tr><td>Impulse</td><td></td><td></td><td></td><td></td><td></td><td></td><td>7.1635.04 7.22 36.34 29.1835.34 35.32 36.1634.74 37.74</td><td></td><td>46</td></tr><tr><td>Defocus</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>34.1865.3034.0066.2262.4464.2464.8466.22 65.5067.24</td><td>27</td></tr><tr><td>Glass</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>20.3242.1219.92 42.8639.44 40.3040.4642.3042.5644.30</td><td>36</td></tr><tr><td>Motion</td><td>43.2862.5844.28 63.2861.76 61.4660.86 62.64 63.08 64.48</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>12</td></tr><tr><td>Zoom</td><td></td><td></td><td></td><td></td><td></td><td></td><td>41.04 64.1041.40 64.9863.70 62.0662.22 64.6463.64 66.14</td><td></td><td>25</td></tr><tr><td>Snow</td><td></td><td></td><td></td><td></td><td></td><td></td><td>46.3253.1047.00 54.4656.74 52.84 52.4853.7053.9856.08</td><td></td><td>8</td></tr><tr><td>Frost</td><td></td><td></td><td></td><td></td><td></td><td></td><td>33.84 52.98 34.42 54.14 54.78 51.48 52.14 53.62 52.38 55.26</td><td></td><td>25</td></tr><tr><td>Fog</td><td></td><td></td><td></td><td></td><td></td><td></td><td>39.04 57.50 39.76 58.76 57.92 56.58 56.38 58.20 58.76 59.88</td><td></td><td>21</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Brightness 67.52 66.62 68.00 67.5871.62 67.36 67.52 67.44 67.38 71.56</td><td></td><td>4</td></tr><tr><td>Contrast</td><td></td><td></td><td></td><td></td><td></td><td></td><td>17.42 64.62 17.38 65.0658.66 64.16 63.18 65.04 65.04 66.14</td><td></td><td>34</td></tr><tr><td>Elastic</td><td></td><td></td><td></td><td></td><td></td><td></td><td>50.2053.6251.54 54.2656.9053.54 53.04 53.1854.2857.38</td><td></td><td>10</td></tr><tr><td>Pixelate</td><td></td><td></td><td></td><td></td><td></td><td></td><td>22.0656.42 22.34 57.10 53.26 55.98 55.54 56.92 53.8458.40</td><td></td><td>38</td></tr><tr><td>JPEG</td><td></td><td></td><td></td><td></td><td></td><td></td><td>45.7045.6446.5446.1850.1046.1645.7445.4246.2050.68</td><td></td><td>8</td></tr><tr><td>Mean</td><td></td><td></td><td></td><td></td><td></td><td></td><td>32.74 53.36 33.12 54.32 52.23 52.80 52.69 53.85 53.64 56.08409</td><td></td><td></td></tr></table>

Table 15: CIFAR100-C, checkpoint 1: accuracy (%) on the same 5,000 evaluation images per corruption. U, D and K are uniform, disagreement-first and symmetric-KL empirical selection; J is joint selection; O is the batch oracle. The last column counts J labels, with a suite total in the Mean row.

<table><tr><td>Corruption</td><td>I</td><td>B</td><td>F</td><td>BF Mean</td><td>U</td><td>D</td><td>K</td><td>J</td><td>0 QJ</td></tr><tr><td>Gaussian</td><td></td><td>9.9239.7010.2040.9431.3039.64 39.6040.72 40.9442.08</td><td></td><td></td><td></td><td></td><td></td><td></td><td>58</td></tr><tr><td>Shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>11.12 40.3011.5641.7833.6038.4038.6841.4841.7842.82</td><td>57</td></tr><tr><td>Impulse</td><td></td><td></td><td></td><td>7.36 34.02 7.36 35.3030.76 34.56 34.90 34.96 35.10 36.50</td><td></td><td></td><td></td><td></td><td>41</td></tr><tr><td>Defocus</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>33.86 64.76 33.88 65.88 62.60 63.38 64.46 65.76 65.48 66.84</td><td>25</td></tr><tr><td>Glass</td><td>19.1042.2419.7843.1639.1441.7642.1442.5442.3244.28</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>34</td></tr><tr><td>Motion</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>43.52 62.80 44.54 63.64 62.12 61.80 62.38 63.46 62.42 64.96</td><td>11</td></tr><tr><td>Zoom</td><td>41.86 64.2042.18 65.4463.70 62.44 62.8065.02 65.44 66.58</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>22</td></tr><tr><td>Snow</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>46.82 53.52 47.70 54.94 57.42 54.02 53.38 53.58 54.72 56.82</td><td>9</td></tr><tr><td>Frost</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>34.0653.4434.60 54.1854.44 51.90 52.5053.34 53.6855.56</td><td>19</td></tr><tr><td>Fog</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>40.20 58.0440.96 58.9857.90 56.5856.80 57.8858.20 60.18</td><td>19</td></tr><tr><td>Brightness67.2467.4668.1668.6071.9068.44 68.2468.02 68.4871.88</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>8</td></tr><tr><td>Contrast</td><td></td><td>18.06 65.02 18.26 65.66 59.34 64.54 64.70 65.62 65.66 66.60</td><td></td><td></td><td></td><td></td><td></td><td></td><td>46</td></tr><tr><td>Elastic</td><td></td><td>49.5853.0651.14 54.4456.3453.66 53.14 52.5654.44 57.20</td><td></td><td></td><td></td><td></td><td></td><td></td><td>10</td></tr><tr><td>Pixelate</td><td></td><td></td><td></td><td></td><td></td><td></td><td>24.04 55.82 23.90 57.16 54.12 56.2056.5456.88 57.16 58.16</td><td></td><td>29</td></tr><tr><td>JPEG</td><td></td><td>44.5444.2044.7645.5449.0245.62 45.7845.2645.6849.40</td><td></td><td></td><td></td><td></td><td></td><td></td><td>5</td></tr><tr><td>Mean</td><td></td><td></td><td></td><td></td><td></td><td></td><td>32.7553.24 33.2754.38 52.25 52.86 53.07 53.81 54.10 55.99393</td><td></td><td></td></tr></table>

Table 16: CIFAR100-C, checkpoint 2: accuracy (%) on the same 5,000 evaluation images per corruption. U, D and K are uniform, disagreement-first and symmetric-KL empirical selection; J is joint selection; O is the batch oracle. The last column counts J labels, with a suite total in the Mean row.

<table><tr><td>Corruption</td><td>I</td><td>B</td><td>F</td><td>BF Mean</td><td></td><td>U</td><td>D</td><td>K</td><td>J</td><td>0 QJ</td></tr><tr><td>Gaussian</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>12.32 38.1812.0639.9034.02 38.8038.66 39.3839.66 40.76</td><td>53</td></tr><tr><td>Shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td>13.6840.1613.62 41.86 36.10 40.10 39.9841.20 41.56 43.06</td><td></td><td></td><td>36</td></tr><tr><td>Impulse</td><td></td><td></td><td></td><td></td><td></td><td></td><td>8.64 33.44 8.66 35.58 30.74 33.60 33.52 34.60 35.5836.60</td><td></td><td></td><td>32</td></tr><tr><td>Defocus</td><td></td><td></td><td></td><td></td><td></td><td></td><td>34.64 65.4634.90 65.72 62.14 64.62 64.76 65.70 65.62 67.06</td><td></td><td></td><td>29</td></tr><tr><td>Glass</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>19.9040.7419.8842.2239.2840.3040.2241.24 41.12 43.30</td><td></td><td>27</td></tr><tr><td>Motion</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>42.6862.22 43.58 62.74 60.9660.7860.90 62.42 62.2863.72</td><td></td><td>14</td></tr><tr><td>Zoom</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>41.10 64.9841.9665.4063.4663.8063.8464.7664.8066.54</td><td></td><td>16</td></tr><tr><td>Snow</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>46.50 53.22 47.08 54.3656.38 52.9853.40 53.46 54.0856.16</td><td>10</td></tr></table>

Table 16: CIFAR100-C, checkpoint 2: accuracy (%) on the same 5,000 evaluation images per corruption (continued).
<table><tr><td>Corruption</td><td>I</td><td>B</td><td>F</td><td></td><td>BF Mean</td><td>U</td><td>D</td><td>K</td><td>J</td><td></td><td>O QJ</td></tr><tr><td>Frost</td><td></td><td></td><td></td><td></td><td>34.70 53.8635.4654.9655.6452.32 53.0854.4854.40 56.22</td><td></td><td></td><td></td><td></td><td></td><td>23</td></tr><tr><td>Fog</td><td></td><td></td><td></td><td></td><td>40.98 57.9841.40 59.0858.70 57.32 56.6058.30 57.3860.30</td><td></td><td></td><td></td><td></td><td></td><td>14</td></tr><tr><td>Brightness66.74 67.52 67.3468.4472.4067.8867.3867.5868.36 71.42</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>8</td></tr><tr><td>Contrast</td><td></td><td></td><td></td><td></td><td>18.62 63.90 18.78 65.00 58.62 63.66 63.54 65.02 65.00 65.70</td><td></td><td></td><td></td><td></td><td></td><td>36</td></tr><tr><td>Elastic</td><td></td><td></td><td></td><td></td><td>50.56 53.96 51.78 55.1057.2654.4653.5854.06 55.10 57.74</td><td></td><td></td><td></td><td></td><td></td><td>2</td></tr><tr><td>Pixelate</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>26.9056.12 27.1057.2453.7054.6855.2056.74 56.6058.38</td><td></td><td>31</td></tr><tr><td>JPEG</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>43.9245.3844.8646.4849.5045.2045.1245.4446.4449.84</td><td>12</td></tr><tr><td>Mean</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>33.46 53.14 33.90 54.27 52.5952.70 52.6553.63 53.87 55.79343</td><td></td></tr></table>

## I VISUAL-LANGUAGE READOUTS ACROSS NATURAL DOMAINS

## I.1 WHOLE-DATASET FIXED PERFORMANCE

The CLIP candidates combine text prompts and image views. S uses one text template and T averages eight templates, both on the original image. V2 averages the eight-template log probabilities over the original image and its horizontal flip. V8 adds three resized crops and their flips, for eight views in total. Later tables call V2 and V8 “U2V” and “U8V” to indicate the shared template set. The crop scale is [0.8, 1.0] and the aspect-ratio interval is [3/4, 4/3]. All experiments use these same transformations. Five repeats vary only the stream and query order.

Table 17 reports every domain and both aggregation rules. The V8 readout gives 98.5171% equaldomain accuracy on PACS and 83.7927% on VLCS without target labels. The gains are distributed unevenly across domains. On VLCS, the SUN domain rises from 66.3909% under S to 77.7536% under V8, while LABELME decreases from 71.6514% to 71.3825%. This domain variation motivates retaining the complete domain table alongside the aggregate.

The OfficeHome saved prediction stream used for current control has V8 accuracy 88.8263%. The separate full-stream fixed endpoint used for the reported-reference comparison is 88.8435%. Table 24 retains that endpoint. The control comparisons below always use their own fixed predictions as the paired reference.

Table 17: Whole-dataset CLIP ViT-L/14 readouts with zero feedback. S uses one template and one view; T uses eight templates on the original view; V2 and V8 average universal-template log probabilities over two and eight views. OfficeHome uses the saved prediction stream evaluated by its controller.

<table><tr><td>Dataset</td><td>Domain</td><td>n</td><td>S</td><td>T</td><td>V2</td><td>V8</td></tr><tr><td>PACS</td><td>Art Painting</td><td>2048</td><td>98.68</td><td>99.02</td><td>99.17</td><td>99.17</td></tr><tr><td>PACS</td><td>Cartoon</td><td>2344</td><td>99.45</td><td>99.57</td><td>99.57</td><td>99.62</td></tr><tr><td>PACS</td><td>Photo</td><td>1670</td><td>99.88</td><td>99.94</td><td>99.94</td><td>99.94</td></tr><tr><td>PACS</td><td>Sketch</td><td>3929</td><td>95.47</td><td>95.04</td><td>95.19</td><td>95.34</td></tr><tr><td>PACS</td><td>Equal-domain</td><td>9991</td><td>98.369</td><td>98.393</td><td>98.468</td><td>98.517</td></tr><tr><td>PACS</td><td>Pooled</td><td></td><td>9991 97.798</td><td>97.738</td><td>97.828</td><td>97.898</td></tr><tr><td>OfficeHome Art</td><td></td><td>2427</td><td>86.98</td><td>88.01</td><td>88.26</td><td>88.71</td></tr><tr><td>OfficeHome Clipart</td><td></td><td>4365</td><td>73.08</td><td>76.01</td><td>76.33</td><td>78.49</td></tr><tr><td>OfficeHome Product</td><td></td><td>4439</td><td>92.90</td><td>93.74</td><td>93.67</td><td>94.12</td></tr><tr><td>OfficeHome Real World</td><td></td><td>4357</td><td>93.05</td><td>93.94</td><td>93.83</td><td>93.99</td></tr><tr><td></td><td>OfficeHome Equal-domain 1558886.503</td><td></td><td></td><td>87.925</td><td></td><td>88.022 88.826</td></tr><tr><td>OfficeHome Pooled</td><td></td><td></td><td></td><td></td><td>1558886.47087.93988.016 88.863</td><td></td></tr><tr><td>VLCS</td><td>Caltech</td><td></td><td></td><td></td><td>991 100.00 100.00 100.00 100.00</td><td></td></tr><tr><td>VLCS</td><td>Labelme</td><td>1859</td><td>71.65</td><td>71.87</td><td>71.60</td><td>71.38</td></tr><tr><td>VLCS</td><td>Pascal</td><td>2363</td><td>84.85</td><td>85.78</td><td>86.16</td><td>86.03</td></tr><tr><td>VLCS</td><td>Sun</td><td>2297</td><td>66.39</td><td>74.75</td><td>75.88</td><td>77.75</td></tr><tr><td>VLCS</td><td>Equal-domain</td><td></td><td>7510 80.723</td><td>83.09983.41083.793</td><td></td><td></td></tr><tr><td>VLCS</td><td>Pooled</td><td></td><td>751077.93680.83981.23881.718</td><td></td><td></td><td></td></tr></table>

Table 18: CLIP accuracy (%) / labels on shared evaluation images. Accuracy averages domains; labels average five orders. Mean-pair KL ranks inputs by average pairwise divergence. A dash marks an unmeasured entry. Appendix I gives full readouts.
<table><tr><td>Rule</td><td>PACS OfficeHome</td><td></td><td>VLCS</td></tr><tr><td>Fixed eight views</td><td>98.44/0</td><td>89.01/0</td><td>84.10/0</td></tr><tr><td>Uniform ERM</td><td>98.43/473.6</td><td>88.90/738.0</td><td>84.07/354.8</td></tr><tr><td>Mean-pair KL</td><td>98.44/473.6</td><td>88.63/738.0</td><td>83.91/354.8</td></tr><tr><td>PORTAL, four</td><td>98.44/1.4</td><td>89.01/7.4</td><td>83.92/89.4</td></tr><tr><td>Fixed StatA</td><td>98.40/0</td><td></td><td>85.11/0</td></tr><tr><td>PORTAL + StatA</td><td>98.44/2.4</td><td></td><td>84.58/96.0</td></tr></table>

## I.2 FIXED CAPACITY AND ACTUAL USE OF FEEDBACK

Table 19 places all fixed readouts and query rules on their common E populations. Joint selection uses 1.4 labels per PACS stream and changes the equal-domain score by +0.0050 percentage points relative to V8. On OfficeHome, it uses 7.4 labels and changes accuracy by +0.0021 points. These near-default outcomes retain the high accuracy of the fixed readout with little annotation. On VLCS, joint selection gives 83.9188%, below fixed V8 at 84.0966% and uniform empirical selection at 84.0657%. The actual effect of feedback therefore depends on both the available readouts and the dataset.

Table 20 resolves these effects by domain. Fixed readouts have zero stream-order variance because their predictions and evaluation membership remain unchanged.

Table 19: VLM evaluation on a permanent common E population. Accuracy is an equal-domain mean over five stream orders; SD measures stream/query variation on one checkpoint and set of image transformations. Pooled accuracy weights images. Labels are actual per-stream annotations.
<table><tr><td>Dataset</td><td>Method</td><td>Equal-domain (%)</td><td>Pooled</td><td>∆ V8 Labels</td><td></td></tr><tr><td>PACS</td><td>Fixed S</td><td>98.1906 ± 0.0000 97.5412 -0.2462</td><td></td><td></td><td>0.0</td></tr><tr><td>PACS</td><td>Fixed T</td><td>98.2791 ± 0.0000 97.5615-0.1578</td><td></td><td></td><td>0.0</td></tr><tr><td>PACS</td><td>Fixed V2</td><td>98.4249 ± 0.0000 97.7037 -0.0120</td><td></td><td></td><td>0.0</td></tr><tr><td>PACS</td><td>Fixed V8</td><td>98.4369 ± 0.0000 97.7444 0.0000</td><td></td><td></td><td>0.0</td></tr><tr><td>PACS</td><td>Uniform ERM</td><td>98.4343 ± 0.005697.7403 -0.0025</td><td></td><td></td><td>473.6</td></tr><tr><td>PACS</td><td>Disagreement ERM</td><td>98.4343 ± 0.0314 97.7403-0.0025</td><td></td><td></td><td>473.6</td></tr><tr><td>PACS</td><td></td><td>Symmetric-KL ERM 98.4394 ± 0.0287 97.7484 0.0025</td><td></td><td></td><td>473.6</td></tr><tr><td>PACS</td><td>PORTAL</td><td>98.4419 ± 0.0069 97.7525 0.0050</td><td></td><td></td><td>1.4</td></tr><tr><td>OfficeHome Fixed S</td><td></td><td>86.6214 ± 0.000086.4133-2.3866</td><td></td><td></td><td>0.0</td></tr><tr><td>OfficeHome Fixed T</td><td></td><td>88.1175 ± 0.0000 88.0057 -0.8905</td><td></td><td></td><td>0.0</td></tr><tr><td>OfficeHome Fixed V2</td><td></td><td>88.2257 ± 0.000088.1341-0.7823</td><td></td><td></td><td>0.0</td></tr><tr><td>OfficeHome Fixed V8</td><td></td><td>89.0080 ± 0.0000 88.9174 0.0000</td><td></td><td></td><td>0.0</td></tr><tr><td>OfficeHome Uniform ERM</td><td></td><td>88.8990 ± 0.093088.7967 -0.1091</td><td></td><td></td><td>738.0</td></tr><tr><td></td><td>OfficeHome Disagreement ERM</td><td>88.5772 ± 0.1174 88.4911 -0.4308</td><td></td><td></td><td>738.0</td></tr><tr><td></td><td>OfficeHome Symmetric-KL ERM 88.6338 ± 0.127888.5579 -0.3742</td><td></td><td></td><td></td><td>738.0</td></tr><tr><td>OfficeHome PORTAL</td><td></td><td>89.0101 ± 0.012488.9200 0.0021</td><td></td><td></td><td>7.4</td></tr><tr><td>VLCS</td><td>Fixed S</td><td>80.8449 ± 0.0000 78.0261 -3.2517</td><td></td><td></td><td>0.0</td></tr><tr><td>VLCS</td><td>Fixed T</td><td>83.5229 ± 0.000081.2982-0.5737</td><td></td><td></td><td>0.0</td></tr><tr><td>VLCS</td><td>Fixed V2</td><td>83.8614 ± 0.000081.7239-0.2352</td><td></td><td></td><td>0.0</td></tr><tr><td>VLCS</td><td>Fixed V8</td><td>84.0966 ± 0.0000 82.0165 0.0000</td><td></td><td></td><td>0.0</td></tr><tr><td>VLCS</td><td>Uniform ERM</td><td>84.0657 ± 0.174381.9633-0.0309</td><td></td><td></td><td>354.8</td></tr><tr><td>VLCS</td><td>Disagreement ERM</td><td>83.8731 ± 0.150281.7398-0.2235</td><td></td><td></td><td>354.8</td></tr><tr><td>VLCS</td><td></td><td>Symmetric-KL ERM 83.9097 ± 0.123981.7930-0.1869</td><td></td><td></td><td>354.8</td></tr><tr><td>VLCS</td><td>PORTAL</td><td>83.9188 ± 0.170381.7930-0.1778</td><td></td><td></td><td>89.4</td></tr></table>

Table 20: VLM common-E domain-level accuracy (%) averaged over five stream orders. U, D and K denote uniform, disagreement-first and symmetric-KL empirical selection. J denotes PORTAL, and $Q _ { J }$ counts its labels. V8 is the fixed default.
<table><tr><td>Dataset</td><td>Domain</td><td>V8</td><td>U</td><td>D</td><td>K</td><td>J</td><td> $Q _ { J }$ </td></tr><tr><td>PACS</td><td>Art Painting</td><td>99.069</td><td>99.069</td><td>99.069</td><td>99.069</td><td>99.069</td><td>0.0</td></tr><tr><td>PACS</td><td>Cartoon</td><td>99.570</td><td>99.570</td><td>99.570</td><td>99.570</td><td>99.570</td><td>0.0</td></tr><tr><td>PACS</td><td>Photo</td><td>100.000100.000</td><td></td><td>100.000</td><td>100.000</td><td>100.000</td><td>0.0</td></tr><tr><td>PACS</td><td>Sketch</td><td>95.108</td><td>95.098</td><td>95.098</td><td>95.119</td><td>95.129</td><td>1.4</td></tr><tr><td>OfficeHome Art</td><td></td><td>88.898</td><td>88.865</td><td>88.405</td><td>88.388</td><td>88.898</td><td>0.0</td></tr><tr><td>OfficeHome Clipart</td><td></td><td>78.016</td><td>77.668</td><td>77.239</td><td>77.426</td><td>78.034</td><td>6.0</td></tr><tr><td>OfficeHome Product</td><td></td><td>94.177</td><td>94.131</td><td>93.949</td><td>93.985</td><td>94.177</td><td>0.6</td></tr><tr><td>OfficeHome Real World</td><td></td><td>94.941</td><td>94.932</td><td>94.717</td><td>94.735</td><td>94.932</td><td>0.8</td></tr><tr><td>VLCS</td><td>Caltech</td><td>100.000</td><td>100.000</td><td>100.000</td><td>100.000</td><td>100.000</td><td>0.0</td></tr><tr><td>VLCS</td><td>Labelme</td><td>71.354</td><td>71.647</td><td>71.396</td><td>71.228</td><td>71.438 61.2</td><td></td></tr><tr><td>VLCS</td><td>Pascal</td><td>86.356</td><td>86.304</td><td>85.872</td><td>85.717</td><td>85.66514.6</td><td></td></tr><tr><td>VLCS</td><td>Sun</td><td>78.677</td><td>78.312</td><td>78.225</td><td>78.695</td><td>78.57313.6</td><td></td></tr></table>

## I.3 OUTPUT TIMING AT A HIGHER ANNOTATION DENSITY

The OfficeHome experiment with batches of eight measures the effect of decision timing. Four readouts are formed first, then three queried labels select one of them. We apply that choice either to the current batch or to the next batch. Table 21 shows a 1.3689-point gain when all current examples are scored. Excluding the queried examples reduces the gain to 0.0639 points; delaying application gives −0.0887 points. Most of the current-batch gain comes from the examples used to make the choice. The queried-excluded score and the permanent-E experiment use different evaluation populations.

Table 21: OfficeHome with three labels per eight-image microbatch. Each timing rule is compared with the fixed output on the same scored inputs. Fixed accuracy is 88.8263% on whole or next batches and 92.8201% on unqueried inputs.

<table><tr><td colspan="3">Estimator Accuracy (%) ∆ fixed (pp)</td></tr><tr><td>Unqueried fixed V8</td><td>92.8201</td><td>0.0000</td></tr><tr><td>Unqueried selected output</td><td>92.8839</td><td>0.0639</td></tr><tr><td>Whole current batch</td><td>90.1952</td><td>1.3689</td></tr><tr><td>Next microbatch</td><td>88.7376</td><td>-0.0887</td></tr></table>

## J IMAGENET INTERVENTIONS AND REPORTED REFERENCE RESULTS

## J.1 FIXED RESTORATION AND SOURCE-STATE COMPOSITIONS

The ImageNet-C experiment evaluates all 15 severity-5 corruptions with 50,000 images each. EATTA, the active-labeling adaptation method of Wang & Ding (2025), supplies the base model update and acquires three labels per batch. Restoration and source-reset candidates use those same query positions. After comparing their outputs, we restore the updated EATTA state. Restoration uses a $3 \times 3$ median filter with reflection padding or an unsharp transform with Gaussian scale 1 and amount 1. The source-reset candidate starts from the source model with an empty sample buffer. The U readout uses batch statistics and averages original and horizontal-flip logits.

Write the five outputs as $z _ { E } , z _ { U } , z _ { M } , z _ { D } , z _ { S } .$ EATTA, two-view, median, unsharp, and source reset. Their restoration blend is

$$
z _ { \mathrm { b l e n d } } = { \textstyle { \frac { 1 } { 2 } } } z _ { U } + { \textstyle { \frac { 1 } { 6 } } } ( z _ { M } + z _ { D } + z _ { S } ) .\tag{77}
$$

The pairwise mixtures UM, UD and US use equal weights. UMD uses weights $( 1 / 2 , 1 / 6 , 1 / 3 )$ Avg weights all five readouts equally. $U _ { M . 2 5 }$ and $U _ { D . 2 5 }$ use a quarter restoration weight and three quarters U. These fixed compositions expose which operations contribute across different corruption families.

Table 22 gives all composition errors. The blend reaches 49.3395% error, improving the locally evaluated EATTA at 50.8808% by 1.5413 points. Its margin against the reported EATTA value of 50.90% is 1.5605 points. The local and literature-reference differences use their respective EATTA results.

Table 22: ImageNet-C error (%) for fixed logit compositions on all 15 severity-5 corruptions, 50,000 images each. E is the EATTA readout; U is the two-view batch-statistic readout; M, D and S are median restoration, unsharp restoration and a source-reset branch. Weights are defined in the text.
<table><tr><td>Corruption</td><td>E</td><td>U</td><td>UM</td><td>UD</td><td>US</td><td>UMD</td><td>Blend</td><td>Avg</td><td>UM.25</td><td>UD.25</td></tr><tr><td>Gaussian</td><td>62.61</td><td>60.33</td><td>60.23</td><td>61.86</td><td>64.26</td><td>60.71</td><td>60.54</td><td>61.36</td><td>59.77</td><td>60.73</td></tr><tr><td>Shot</td><td>55.25</td><td>53.35</td><td>54.04</td><td>54.60</td><td>57.33</td><td>54.01</td><td>53.71</td><td>54.48</td><td>53.34</td><td>53.69</td></tr><tr><td>Impulse</td><td>54.88</td><td>53.12</td><td>50.12</td><td>53.86</td><td>57.03</td><td>52.21</td><td>51.93</td><td>52.19</td><td>51.31</td><td>53.29</td></tr><tr><td>Defocus</td><td>64.08</td><td>62.50</td><td>63.02</td><td>62.79</td><td>65.30</td><td>62.75</td><td>62.61</td><td>63.08</td><td>62.66</td><td>62.54</td></tr><tr><td>Glass</td><td>61.54</td><td>60.28</td><td>60.54</td><td>60.58</td><td>63.43</td><td>60.53</td><td>60.26</td><td>60.61</td><td>60.34</td><td>60.32</td></tr><tr><td>Motion</td><td>53.52</td><td>53.51</td><td>53.61</td><td>53.01</td><td>54.85</td><td>53.11</td><td>52.60</td><td>52.61</td><td>53.38</td><td>53.14</td></tr><tr><td>Zoom</td><td>49.57</td><td>48.16</td><td>49.22</td><td>48.44</td><td>47.80</td><td>48.61</td><td>47.53</td><td>47.86</td><td>48.61</td><td>48.21</td></tr><tr><td>Snow</td><td>50.62</td><td>49.30</td><td>51.05</td><td>49.79</td><td>50.45</td><td>49.88</td><td>49.26</td><td>49.67</td><td>49.80</td><td>49.39</td></tr><tr><td>Frost</td><td>54.32</td><td>53.01</td><td>53.25</td><td>53.58</td><td>54.07</td><td>53.31</td><td>52.47</td><td>52.89</td><td>52.88</td><td>53.19</td></tr><tr><td>Fog</td><td>42.29</td><td>41.29</td><td>42.28</td><td>41.53</td><td>41.23</td><td>41.47</td><td>40.95</td><td>41.20</td><td>41.58</td><td>41.29</td></tr><tr><td>Brightness</td><td>34.70</td><td>34.16</td><td>34.89</td><td>34.36</td><td>32.21</td><td>34.43</td><td>33.32</td><td>33.44</td><td>34.37</td><td>34.20</td></tr><tr><td>Contrast</td><td>51.89</td><td>50.22</td><td>51.31</td><td>50.89</td><td>55.77</td><td>50.68</td><td>50.95</td><td>51.68</td><td>50.45</td><td>50.44</td></tr><tr><td>Elastic</td><td>44.19</td><td>43.08</td><td>44.14</td><td>43.17</td><td>43.29</td><td>43.29</td><td>42.61</td><td>42.95</td><td>43.38</td><td>43.01</td></tr><tr><td>Pixelate</td><td>40.08</td><td>39.17</td><td>39.46</td><td>39.62</td><td>39.61</td><td>39.44</td><td>38.77</td><td>39.01</td><td>39.15</td><td>39.32</td></tr><tr><td>JPEG</td><td>43.67</td><td>42.59</td><td>42.98</td><td>43.13</td><td>44.19</td><td>42.92</td><td>42.56</td><td>42.89</td><td>42.66</td><td>42.77</td></tr><tr><td>Mean</td><td>50.880849.604450.0092 50.080851.387149.823649.339549.727949.5789</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>949.7016</td></tr></table>

## J.2 SEPARATE AND SHARED LABEL ACQUISITION

Table 23 compares two selection procedures on complete 750,000-image streams. In the first, EATTA updates the model before the horizontal-view challenger predicts the current batch. Three separately acquired symmetric-divergence queries choose between their outputs. The error reduction is 1.0083 points, and an independent full stream gives 0.9824 points. Both streams improve all 15 corruptions. In the first stream, always using the challenger improves error by 1.3741 points. Admission therefore retains 73.37% of that gain. A paired moving-block bootstrap gives a 95% interval of [0.9681, 1.0478] points for admission gain and [71.06, 75.97]% for the retained fraction. It resamples ten-batch blocks within each corruption, using 10,000 replicates. Selection requests three labels in addition to EATTA’s three. The recorded counts identify each request set; their overlap was not recorded.

The shared-prefix procedure passes the selector’s ordered labels to EATTA’s committed update. The recorded prefix contains 46,908 labels over 11,730 batches, or 3.99898 labels per batch. Error falls from 50.1383% to 49.5487%, a gain of 0.5896 points. Every corruption improves. Always using the challenger on the same model trajectory gives 48.7289%, a fixed gain of 1.4093 points.

Admission occurs on 20.32% of batches. Among admitted batches, 77.31% improve accuracy, 8.98% worsen it and 13.72% tie, up to rounding. Mean recorded prefixes contain 2.87 labels for admission, 2.94 for evidence rejection and 4.54 when the budget is exhausted. The evidence floor is the configured threshold for admitting a challenger. All 2,384 recorded admissions cross this threshold. These are decision-trace statistics.

Provisional branches run supervised EATTA updates before the prefix stops. These trial updates access at least 56,299 current-batch positions. The 46,908 committed-prefix labels therefore omit some supervised access. Predictions are missing for 9,391 available but unused positions. The saved trace cannot reconstruct evaluation on all never-accessed inputs. Removing only committed-prefix inputs gives a 0.2899-point gain; this is a prefix-excluded score.

The provisional model changes are rolled back, but admission affects the label-credit bank carried into later decisions. Holding other recorded values fixed, the opposite admission choice changes the next bank in 7,686 of 11,730 batches. A shared model successor therefore does not imply the same future query process. Table 23 preserves the measured errors and identifies q as the committed-prefix count.

Table 23: ImageNet-C current-batch error (%) under two feedback protocols. The first two streams use three adaptation labels plus three separate selection labels per batch. The shared-prefix stream reuses selection labels for the committed update; q counts its committed prefix, excluding additional provisional supervised reads. P, F and A denote EATTA, fixed challenger and selected output.
<table><tr><td colspan="4">Additional-query stream</td><td colspan="3">Independent repeat</td><td colspan="4">Shared-prefix stream</td></tr><tr><td>Corruption</td><td>P</td><td>F</td><td>A</td><td>P</td><td>F</td><td>A</td><td>P</td><td>F</td><td>A</td><td>q</td></tr><tr><td>Gaussian</td><td>62.82</td><td>60.32</td><td>61.26</td><td>62.98</td><td>60.36</td><td>61.14</td><td>61.77</td><td>59.90</td><td>61.10</td><td>3.92</td></tr><tr><td>Shot</td><td>55.32</td><td>53.33</td><td>53.98</td><td>55.38</td><td>53.58</td><td>54.22</td><td>54.69</td><td>53.37</td><td>54.06</td><td>3.99</td></tr><tr><td>Impulse</td><td>54.90</td><td>53.02</td><td>53.66</td><td>54.88</td><td>53.08</td><td>53.63</td><td>54.24</td><td>51.38</td><td>53.23</td><td>3.49</td></tr><tr><td>Defocus</td><td>64.11</td><td>62.44</td><td>63.05</td><td>64.26</td><td>62.56</td><td>63.08</td><td>63.12</td><td>61.85</td><td>62.66</td><td>4.19</td></tr><tr><td>Glass</td><td>61.39</td><td>60.00</td><td>60.51</td><td>61.46</td><td>60.26</td><td>60.72</td><td>60.47</td><td>59.37</td><td>59.99</td><td>4.30</td></tr><tr><td>Motion</td><td>52.99</td><td>52.88</td><td>52.58</td><td>53.73</td><td>53.48</td><td>53.18</td><td>52.39</td><td>51.54</td><td>51.94</td><td>4.28</td></tr><tr><td>Zoom</td><td>49.47</td><td>48.06</td><td>48.43</td><td>49.85</td><td>48.51</td><td>48.91</td><td>49.00</td><td>47.11</td><td>48.32</td><td>3.53</td></tr><tr><td>Snow</td><td>50.49</td><td>49.15</td><td>49.45</td><td>50.55</td><td>49.12</td><td>49.63</td><td>49.90</td><td>48.52</td><td>49.33</td><td>3.97</td></tr><tr><td>Frost</td><td>54.45</td><td>53.03</td><td>53.42</td><td>54.13</td><td>52.84</td><td>53.23</td><td>53.32</td><td>51.82</td><td>52.68</td><td>3.90</td></tr><tr><td>Fog</td><td>42.08</td><td>40.97</td><td>41.11</td><td>42.22</td><td>41.12</td><td>41.37</td><td>41.86</td><td>40.53</td><td>41.32</td><td>4.00</td></tr><tr><td>Brightness</td><td>34.78</td><td>34.10</td><td>34.14</td><td>34.96</td><td>34.23</td><td>34.30</td><td>34.39</td><td>32.99</td><td>33.81</td><td>3.83</td></tr><tr><td>Contrast</td><td>52.00</td><td>50.17</td><td>50.75</td><td>51.64</td><td>50.08</td><td>50.55</td><td>50.84</td><td>50.08</td><td>50.31</td><td>4.09</td></tr><tr><td>Elastic</td><td>44.23</td><td>42.95</td><td>43.21</td><td>44.04</td><td>42.89</td><td>43.12</td><td>43.67</td><td>42.28</td><td>43.11</td><td>4.16</td></tr><tr><td>Pixelate</td><td>39.79</td><td>38.81</td><td>38.97</td><td>39.94</td><td>39.03</td><td>39.20</td><td>39.43</td><td>38.20</td><td>38.84</td><td>4.16</td></tr><tr><td>JPEG</td><td>43.52</td><td>42.49</td><td>42.69</td><td>43.56</td><td>42.36</td><td>42.58</td><td>42.98</td><td>41.99</td><td>42.52</td><td>4.17</td></tr><tr><td>Mean</td><td>50.8223</td><td></td><td></td><td></td><td>49.448149.814050.905749.567549.923350.1383</td><td></td><td></td><td>348.728949.5487 3.9990</td><td></td><td></td></tr></table>

## J.3 SYSTEM ENDPOINTS AND EXTERNAL COMPARISONS

Table 24 compares complete-system scores with published results. These include active labeling with EATTA on ImageNet-C/K (Wang & Ding, 2025) and TDA, a training-free adaptation method for CLIP, on ImageNet-A (Karmanov et al., 2024). For natural-domain CLIP results, the references use weight-average test-time adaptation (WATT), test-time prompt tuning (TPT), and entropy minimization (TENT) (Osowiechi et al., 2024; Shu et al., 2022; Wang et al., 2021). CIFAR also includes active-feedback SimATTA (Gui et al., 2024). Our local selection comparisons use paired predictions on the same evaluation population.

The label-free OfficeHome endpoint also lies 4.6435 percentage points above the reported 84.20% CLIPTTA ViT-B/16 result (Lafon et al., 2025). This cross-backbone comparison provides context for the full system’s score. The ViT-L/14 reference in the table is 86.72%, giving a within-backbone margin of 2.1235 points.

Table 24: Direct fixed-intervention endpoints and reported literature values. ImageNet-C/K use EATTA with three labels per batch; the visual-language rows use zero target labels. Margins orient improvement positively. The ImageNet-A endpoint averages two declared runs. CIFAR fixed endpoints use zero labels, while their SimATTA references use active feedback.

<table><tr><td>Benchmark</td><td>Backbone</td><td></td><td>Reference Metric Reported</td><td></td><td>Fixed</td><td>∆</td></tr><tr><td>ImageNet-C</td><td>ResNet-50 EATTA</td><td></td><td>Error</td><td></td><td>50.9049.3395</td><td>1.5605</td></tr><tr><td>ImageNet-K</td><td>ResNet-50 EATTA</td><td></td><td>Error</td><td></td><td>61.4061.1252</td><td>0.2748</td></tr><tr><td>ImageNet-A</td><td>ViT-B/16</td><td>TDA</td><td>Acc.</td><td></td><td>60.11 60.2000 0.0900</td><td></td></tr><tr><td>PACS</td><td>ViT-L/14</td><td>WATT</td><td>Acc.</td><td></td><td>97.51 98.5171</td><td>1.0071</td></tr><tr><td>OfficeHome</td><td>ViT-L/14</td><td>TPT</td><td>Acc.</td><td></td><td>86.7288.8435</td><td>2.1235</td></tr><tr><td>VLCS</td><td>ViT-L/14</td><td>TENT</td><td>Acc.</td><td></td><td>80.20 83.7927 3.5927</td><td></td></tr><tr><td>CIFAR10-C</td><td>ResNet18</td><td>SimATTA Acc.</td><td></td><td></td><td>83.1379.9218-3.2082</td><td></td></tr><tr><td>CIFAR100-C ResNet18</td><td></td><td>SimATTA Acc.</td><td></td><td></td><td>55.8654.7544-1.1056</td><td></td></tr></table>

## K LEARNING A STRONGER CANDIDATE FROM PAST FEEDBACK

The online-head experiment expands the PACS portfolio with a candidate that changes as labels accumulate. The Photo-source ResNet50 from test-time distillation (Chen et al., 2026) supplies the fixed BF default and its original classifier-input features. The three target domains contain 8,321

images, of which 4,166 belong to $Q$ and 4,155 to $E .$ Each of five complete orders contains 131 batches and purchases 393 distinct $Q$ labels. Head state resets at a domain boundary.

Normalize each feature as $x = f / \| f \| _ { 2 }$ . Let X contain past queried features and $Y$ their one-hot class labels. The residual targets are $\dot { R } = Y - \mathrm { s o f t m a x } ( z _ { \mathrm { B F } } )$ . With regularization strength λ and residual scale s, the learned output is

$$
\begin{array} { r } { z _ { \mathrm { l e a r n e d } } ( x ) = z _ { \mathrm { B F } } ( x ) + s x X ^ { \top } ( X X ^ { \top } + \lambda I ) ^ { - 1 } R . } \end{array}\tag{78}
$$

The primary setting uses $\lambda = 1$ and $s \ = \ 4 .$ . Current labels choose between predictions already formed from past feedback, then enter the residual fit for subsequent batches.

Always using the learned candidate gives approximately 73% accuracy across query policies, compared with 59.6775% for BF (Table 25). Uniform three-label selection retains 6.2335 points of this improvement, disagreement-based selection retains 10.6089 points, and PORTAL retains 12.9627 points. PORTAL stays 0.3851 points below always using its learned candidate. Learning supplies the main gain; selection determines how much reaches deployment.

PORTAL uses an average of 70.8 labels for current decisions. Uniform queries fill the remaining training allowance, giving the same total of 393 labels as the simple policies. Each policy uses its own acquired labels and learned state. Table 26 reports the primary setting and every prespecified regularization and scale comparison.

Table 25: Causal PACS residual-head learning with ridge coefficient 1 and logit scale 4. Each query policy generates its own full learning trajectory over five orders. Decision labels are part of the total 393 labels, and unused joint-decision slots train the next-batch head. Fixed BF alone uses zero target labels.

<table><tr><td>Query</td><td>Output</td><td>Accuracy (%)</td><td>∆ BF</td><td>CE</td><td>Decision Q</td><td>Total Q</td></tr><tr><td>Uniform</td><td>Fixed BF</td><td> $5 9 . 6 7 7 \pm 0 . 2 7 8$ </td><td>0.000</td><td>1.2205</td><td>0.0</td><td>0</td></tr><tr><td>Uniform</td><td>Always learned</td><td> $7 3 . 0 3 0 \pm 0 . 6 9 5$ </td><td></td><td>13.3530.9430</td><td>393.0</td><td>393</td></tr><tr><td>Uniform</td><td>Selected output</td><td> $6 5 . 9 1 1 \pm 0 . 4 9 3$ </td><td></td><td>6.2331.0918</td><td>393.0</td><td>393</td></tr><tr><td>Disagreement Fixed BF</td><td></td><td> $5 9 . 6 7 7 \pm 0 . 2 7 8$ </td><td></td><td>0.0001.2205</td><td>0.0</td><td>0</td></tr><tr><td></td><td>Disagreement Always learned</td><td> $7 2 . 9 8 2 \pm 0 . 7 7 9$ </td><td></td><td>13.3040.9424</td><td>393.0</td><td>393</td></tr><tr><td></td><td>Disagreement Selected output</td><td> $7 0 . 2 8 6 \pm 0 . 7 5 9$ </td><td>10.609 0.9999</td><td></td><td>393.0</td><td>393</td></tr><tr><td>PORTAL</td><td>Fixed BF</td><td> $5 9 . 6 7 7 \pm 0 . 2 7 8$ </td><td></td><td>0.0001.2205</td><td>0.0</td><td>0</td></tr><tr><td>PORTAL</td><td>Always learned</td><td> $7 3 . 0 2 5 \pm 0 . 9 7 2$ </td><td></td><td>13.3480.9390</td><td>70.8</td><td>393</td></tr><tr><td>PORTAL</td><td>Selected output</td><td> $7 2 . 6 4 0 \pm 1 . 0 4 8$ </td><td></td><td>12.9630.9553</td><td>70.8</td><td>393</td></tr></table>

Table 26: All residual-head sensitivity settings fixed before evaluation on PACS. Every row uses five complete orders and 393 total labels per order. The primary setting remains lambda=1 and scale=4.
<table><tr><td>λ Scale Query</td><td></td><td>Always learned Selected</td><td></td><td>∆ learned</td></tr><tr><td>1</td><td>4 Uniform</td><td>73.030</td><td>65.911</td><td>-7.119</td></tr><tr><td>1</td><td>4 Disagreement</td><td>72.982</td><td>70.286</td><td>-2.696</td></tr><tr><td>1</td><td>4 PORTAL</td><td>73.025</td><td>72.640</td><td>-0.385</td></tr><tr><td>0.1</td><td>4 Uniform</td><td>75.321</td><td>68.125</td><td>-7.196</td></tr><tr><td>0.1</td><td>4 Disagreement</td><td>75.576</td><td>71.981</td><td>-3.596</td></tr><tr><td>0.1</td><td>4 PORTAL</td><td>75.302</td><td>74.729</td><td>-0.573</td></tr><tr><td>10</td><td>4 Uniform</td><td>67.475</td><td>62.373</td><td>-5.102</td></tr><tr><td>10</td><td>4 Disagreement</td><td>67.466</td><td>65.083</td><td>-2.383</td></tr><tr><td>10</td><td>4 PORTAL</td><td>67.114</td><td>66.503</td><td>-0.611</td></tr><tr><td>1</td><td>1 Uniform</td><td>65.483</td><td>60.934</td><td>-4.549</td></tr><tr><td>1</td><td>1 Disagreement</td><td>65.584</td><td>64.438</td><td>-1.146</td></tr><tr><td>1</td><td>1 PORTAL</td><td>65.526</td><td>64.414</td><td>-1.112</td></tr><tr><td>1</td><td>8 Uniform</td><td>72.606</td><td>66.416</td><td>-6.190</td></tr><tr><td>1</td><td>8 Disagreement</td><td>71.957</td><td>69.049</td><td>-2.907</td></tr><tr><td>1</td><td>8 PORTAL</td><td>72.226</td><td>72.125</td><td>-0.101</td></tr></table>

## L IMMEDIATE OUTPUT AND PERSISTENT-STATE CONTINUATION

Persistent-state effects are measured with two explicit update operators on the Photo-source features. One state is a residual linear classifier Θ trained by a regularized squared-loss step. The other is a feature-mean cache $\mu$ that changes logits through the released classifier W. The output is $z = z _ { \mathrm { s o u r c e } } + x \Theta - \alpha \mu \dot { W } ^ { \top }$ , with $\alpha = 0 . 0 5$ . The cache averages raw features, while the residual head uses unit-normalized features.

At each batch, we copy the pre-update state to test three actions: use an update only for the current output, retain a classifier update, or retain a cache update. The first action undoes its update after prediction. All copies then receive identical future queries and updates, with each batch scored before its update. The reference trajectory applies both updates after the current batch. States reset at domain boundaries.

The head step uses learning rate $\eta = 0 . 5$ and regularization $\lambda = 0 . 1$ . If $X _ { t }$ contains its three queried unit feature rows, the difference between committed and rollback head states evolves as

$$
\Delta \Theta _ { t + 1 } = A _ { t } \Delta \Theta _ { t } , \qquad A _ { t } = ( 1 - \eta \lambda ) I - \frac { \eta } { 3 } X _ { t } ^ { \top } X _ { t } .\tag{79}
$$

The eigenvalues of $A _ { t }$ lie in [0.45, 0.95]. The cache uses $\mu _ { t + 1 } = 0 . 9 \mu _ { t } + 0 . 1 \bar { f } _ { Q , t }$ , so its branch difference contracts exactly by 0.9. These operators determine the direction and scale of later logit changes.

Table 27 measures horizons of 1, 4 and 16 future batches. There are respectively 128, 119 and 83 valid within-domain branch points per order. At one-step branch points, the head improves immediate accuracy by 0.2296 points but changes next-batch accuracy by −0.1972 points. Next-batch cross-entropy nevertheless decreases. The same immediate action can therefore have a different future accuracy effect, and cross-entropy and zero-one loss capture different aspects of that continuation.

For cross-entropy, the directional expansion satisfies

$$
\begin{array} { r } { L \leq \ell _ { y } ( z + \Delta z ) - \ell _ { y } ( z ) \leq L + \frac { 1 } { 4 } \| \Delta z \| _ { 2 } ^ { 2 } , \quad L = ( \mathrm { s o f t m a x } ( z ) - e _ { y } ) ^ { \top } \Delta z . } \end{array}\tag{80}
$$

The linear term uses evaluation labels to measure the loss change retrospectively. The exact state recurrence supplies the logit difference. Undoing the current update gives zero future effect. A hard prediction can change only if the baseline top-two logit margin is at most $2 \| \Delta z \| _ { \infty }$ . We measure accuracy changes directly. Future windows overlap, and the paired averages describe these shared trajectories.

Table 27: Persistent-state continuation on PACS. Accuracy changes are in percentage points. Future cross-entropy (CE) change, directional linear term L and remainder ceiling R are multiplied by 1,000. The actual CE change lies in [L,L+R] before averaging. Changes counts hard-prediction differences over all valid overlapping comparisons per order.

<table><tr><td>State</td><td>H ∆ now</td><td>∆ future</td><td>ΔCE</td><td>L</td><td>R Changes</td></tr><tr><td>Head commit</td><td>1 0.2296</td><td></td><td></td><td>-0.1972-0.5907-2.65308.1582</td><td>174.2</td></tr><tr><td>Head commit</td><td>40.2038</td><td></td><td></td><td>-0.0578-0.7994-1.56343.0299</td><td>333.6</td></tr><tr><td>Head commit</td><td>160.2313</td><td></td><td></td><td>-0.0169-0.4486-0.65740.8191</td><td>302.8</td></tr><tr><td>Cache commit</td><td>1 0.0293</td><td></td><td></td><td>0.0148-0.2187-0.2212 0.0115</td><td>7.6</td></tr><tr><td>Cache commit 4 0.0366</td><td></td><td></td><td></td><td>0.0118-0.1917-0.1935 0.0086</td><td>23.4</td></tr><tr><td>Cache commit 16 0.0372</td><td></td><td></td><td></td><td>0.0023-0.1503-0.1510 0.0036</td><td>40.4</td></tr></table>

## M IMPLEMENTATION AND RESOURCE ACCOUNTING

The CIFAR experiment retains four float32 logits and four corresponding 512-dimensional feature vectors per image. The linear classifier maps averaged original/flip features to the matching averaged logits. CLIP inference processes eight views sequentially at batch size 16. Its saved outputs retain the original features, text prototypes and individual view log probabilities. Selection operates on batches of 64. The online ridge head operates in double precision on normalized 2,048-dimensional features and stores at most 186 queried supports in a target domain.

Table 28: Measured OfficeHome CLIP ViT-L/14 component costs, batch size 8. Forward columns are ms/image; decision is ms/batch after action logits exist. Each domain contributes 64 measured images. Memory is peak allocation in MiB.
<table><tr><td>Domain</td><td>Single Eight-view</td><td></td><td>Decision Eight-view MiB</td></tr><tr><td>Art</td><td>2.354</td><td>16.669</td><td>0.507 2616.9</td></tr><tr><td>Clipart</td><td>2.084</td><td>16.732</td><td>0.546 2617.6</td></tr><tr><td>Product</td><td>2.093</td><td>16.784</td><td>0.633 2617.6</td></tr><tr><td>Real World</td><td>2.100</td><td>16.809</td><td>0.451 2617.6</td></tr><tr><td>Mean</td><td>2.158</td><td>16.749</td><td>0.534 2617.4</td></tr></table>

Table 28 separates visual-intervention cost from decision cost on OfficeHome. The eight-view forward costs 16.7485 ms/image, and the decision arithmetic adds 0.5343 ms per eight-image batch after logits are available. This corresponds to 16.8153 ms/image for the combined path, compared with 2.1577 ms/image for one view. Annotation-response latency is excluded, and feedback is reported separately as three labels per microbatch.

The five complete online-head orders require approximately 0.62 seconds per order for the simple query rules and 0.73 seconds for joint selection on the saved features. Their peak stored support/head arrays occupy 2.93 MiB, with temporary kernel-solver memory additional. The complete CIFAR collector peaks at 134.58 MiB of allocated CUDA memory, while the sequential-view CLIP collection peaks at approximately 1,051 MiB. These are measurements of the implemented schedules: the CLIP component benchmark above retains a different allocation pattern. Backbone evaluation, feature storage, controller arithmetic and target annotations remain separate resource components.

## N FIXED-OUTPUT MECHANISM AND COST CONTROLS

These PACS controls hold the Photo-source ResNet50 and permanent image roles fixed. The targets are Art, Cartoon and Sketch, with 1,007, 1,175 and 1,973 evaluation images. Five stream orders recompute the original, normalized, flipped and normalized-plus-flipped predictions. Each comparison uses the running-statistic original prediction I as its default and one challenger at a time. This identifies the effect of the candidate component and acquisition rule before introducing the learned head.

For an evaluation image, let $d _ { i }$ be the default classification error minus the selected output’s error. With N evaluation images, gain is $G \ = \ 1 0 0 \textstyle \sum _ { i } d _ { i } / N$ , the new-error percentage is $S \ =$ $\textstyle 1 0 0 \sum _ { i } { \mathbf { 1 } } \{ d _ { i } = - 1 \} / N$ , and negative batch contribution is $\begin{array} { r } { H _ { B } = 1 0 0 \sum _ { b } [ - \sum _ { i \in b } d _ { i } ] _ { + } / N } \end{array}$ . These quantities distinguish individual errors from a batch’s net effect. Gains and costs always refer to the same fixed outputs and evaluation population.

## N.1 CANDIDATE COMPONENTS AND CONFIDENCE SCALE

Tables 29–31 give the three-label decision rules for normalization, horizontal views, and their combination. The count k is the minimum queried signed advantage needed to select the challenger. Ties retain I. The K-Margin query rule scales source features by $\bar { 1 } - ( p _ { ( 1 ) } - p _ { ( 2 ) } )$ , where $p _ { ( 1 ) }$ and $p _ { ( 2 ) }$ are the largest two class probabilities. It greedily selects distant scaled features. Here we compare its queries on the same fixed candidate predictions.

Table 29: Normalization on the frozen Photo-source PACS model. Each acquisition rule uses three query labels and the distinct minimum signed count k. The default is running-statistic I. G and $H _ { B }$ are percentage points; S is the percentage of newly introduced errors.

<table><tr><td>Acquisition k</td><td>G  $H _ { B }$  S</td></tr><tr><td>Always candidate</td><td>- 7.492.471 11.85</td></tr><tr><td>Uniform</td><td>14.540.84</td></tr><tr><td></td><td>4.33</td></tr><tr><td>Uniform</td><td>2 1.58 0.26 1.17</td></tr><tr><td>Uniform</td><td>3 0.30 0.00 0.05</td></tr><tr><td>Disagreement</td><td>1 5.771.25 5.81</td></tr></table>

Table 29: Normalization on the frozen Photo-source PACS model (continued).
<table><tr><td>Acquisition</td><td>k G</td><td> $H _ { B }$ </td><td> $S$ </td></tr><tr><td>Disagreement</td><td>2 3.50 0.53</td><td></td><td rowspan="6">2.47 0.51 5.68 2.38</td></tr><tr><td>Disagreement</td><td>31.19 0.09</td></tr><tr><td>Symmetric KL</td><td>1 6.131.23</td></tr><tr><td>Symmetric KL</td><td>2 4.48 0.55</td></tr><tr><td>Symmetric KL</td><td>3 2.07 0.12</td></tr><tr><td></td><td>0.78 6.36</td></tr><tr><td></td><td></td></tr><tr><td>K-Margin</td><td>1 5.15 1.28</td></tr><tr><td>K-Margin</td><td>2 2.600.26 1.94</td></tr><tr><td>K-Margin</td><td>3 0.33 0.08 0.34</td></tr></table>

Table 30: Horizontal views on the frozen Photo-source PACS model. Each acquisition rule uses three query labels and the distinct minimum signed count k. The default is running-statistic I. G and $H _ { B }$ are percentage points; S is the percentage of newly introduced errors.

<table><tr><td>Acquisition</td><td>k</td><td>G  $H _ { B }$ </td></tr><tr><td>Always candidate</td><td></td><td>0.02 1.19 2.07</td></tr><tr><td>Uniform</td><td>1</td><td>0.01 0.05 0.12</td></tr><tr><td>Uniform</td><td>2</td><td>0.00 0.00 0.00</td></tr><tr><td>Uniform</td><td></td><td>3 0.00 0.00 0.00</td></tr><tr><td>Disagreement</td><td></td><td>1-0.04 0.43 0.68 2 0.03 0.07 0.12</td></tr><tr><td>Disagreement</td><td></td><td></td></tr><tr><td>Disagreement</td><td></td><td>3 0.01 0.01 0.02</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Symmetric KL</td><td></td><td>1-0.00 0.280.51</td></tr><tr><td>Symmetric KL</td><td></td><td>2 0.00 0.03 0.05</td></tr><tr><td>Symmetric KL</td><td></td><td>3 0.00 0.00 0.00</td></tr><tr><td>K-Margin</td><td></td><td>1 0.03 0.17 0.31</td></tr><tr><td>K-Margin</td><td></td><td>2 -0.02 0.02 0.03</td></tr><tr><td>K-Margin</td><td></td><td>3 0.00 0.00 0.00</td></tr></table>

Table 31: Normalization and views on the frozen Photo-source PACS model. Each acquisition rule uses three query labels and the distinct minimum signed count k. The default is running-statistic I. G and $H _ { B }$ are percentage points; S is the percentage of newly introduced errors.

<table><tr><td>Acquisition</td><td>k</td><td>G  $H _ { B }$ </td><td> $S$ </td></tr><tr><td>Always candidate</td><td></td><td>8.80 2.00</td><td>11.18</td></tr><tr><td>Uniform</td><td></td><td>1 5.15 0.72</td><td>4.30</td></tr><tr><td>Uniform</td><td></td><td>2 1.86 0.19</td><td>1.18</td></tr><tr><td>Uniform</td><td></td><td>3 0.33 0.00</td><td>0.10</td></tr><tr><td>Disagreement</td><td></td><td>1 7.03 0.69</td><td>5.16</td></tr><tr><td>Disagreement</td><td></td><td>24.260.24</td><td>2.31</td></tr><tr><td>Disagreement</td><td></td><td>31.53 0.03</td><td>0.62</td></tr><tr><td>Symmetric KL</td><td></td><td>1 7.25 1.15</td><td>6.15</td></tr><tr><td>Symmetric KL</td><td></td><td>2 5.200.51</td><td>2.66</td></tr><tr><td>Symmetric KL</td><td></td><td>32.78 0.12</td><td>0.90</td></tr><tr><td>K-Margin</td><td></td><td>1 6.081.03</td><td>6.08</td></tr><tr><td>K-Margin</td><td></td><td>2 3.10 0.24</td><td>1.83</td></tr><tr><td>K-Margin</td><td></td><td>30.480.04</td><td>0.31</td></tr></table>

Positive temperature scaling preserves every hard prediction but changes the probability discrepancies used for acquisition. Table 32 compares all five tested temperatures. At temperature 0.25, the selected query set has 68.40% mean overlap with temperature one, and 19.54% of batch decisions change. Since both candidates’ zero-one losses are fixed, the resulting differences arise through which labels are acquired.

Table 32: Positive-temperature acquisition control with fixed hard predictions. The candidate is BF, the budget is three, and k=1. Overlap and decision changes use temperature one as the reference.
<table><tr><td colspan="2"> $T$  G  $H _ { B }$  S Overlap (%) Changed (%)</td></tr><tr><td>0.257.501.407.54</td><td>68.40 19.54</td></tr><tr><td>0.57.371.34 7.22</td><td>77.51 13.74</td></tr><tr><td>1 7.251.15 6.15</td><td>100.00 0.00</td></tr><tr><td>2 7.10 1.01 5.78</td><td>82.49 9.31</td></tr><tr><td>4 6.85 1.05 5.90</td><td>76.18 13.59</td></tr></table>

The score-bin comparison in Table 33 uses the mean score of the three selected query points to form quintiles. One construction pools domains; the other forms bins within each domain and then combines corresponding bins. Query advantage and evaluation advantage are calculated after bin assignment. Their separation shows how a query score can organize observed disagreement without directly determining the evaluation benefit of deploying the challenger.

Table 33: Symmetric-KL score quintiles for BF, from low to high. Query and evaluation gains use their own image populations. Negative is the percentage of batches where BF increases evaluation errors. Bin membership depends on the label-free query score.

<table><tr><td>Binning</td><td>Bin Batches</td><td> $n _ { E }$ </td><td>Query gain E gain Negative (%)</td><td></td><td></td></tr><tr><td>All domains</td><td>1</td><td></td><td>27849.4</td><td>14.07</td><td>6.10 17.04</td></tr><tr><td>All domains</td><td>2</td><td>26 829.2</td><td></td><td>45.64 17.16</td><td>10.00</td></tr><tr><td>All domains</td><td>3</td><td>26 829.2</td><td></td><td>33.08 10.60</td><td>26.92</td></tr><tr><td>All domains</td><td>4</td><td>26 842.4</td><td></td><td>21.79</td><td>4.75 38.46</td></tr><tr><td>All domains</td><td>5</td><td>26 804.8</td><td></td><td>33.85</td><td>5.34 33.08</td></tr><tr><td>Within domain</td><td>1</td><td>28 900.4</td><td></td><td>25.71</td><td>9.23 22.86</td></tr><tr><td>Within domain</td><td>2</td><td>28 903.4</td><td></td><td>24.29</td><td>8.88 27.14</td></tr><tr><td>Within domain</td><td>3</td><td>25800.0</td><td></td><td>26.40 8.61</td><td>24.80</td></tr><tr><td>Within domain</td><td>4</td><td>25 775.0</td><td></td><td>35.47</td><td>7.57 30.40</td></tr><tr><td>Within domain</td><td>5</td><td>25 776.2</td><td></td><td>37.07</td><td>9.64 20.00</td></tr></table>

## N.2 DISTINCT BUDGETS AND EMPIRICAL RULE MIXTURES

Table 34 retains all distinct measured count rules at budgets 1, 3, 5 and 10. Rules with equivalent thresholds appear once. A budget-m rule spends 131m labels per complete order. The hard-capthree library contains its fixed outputs, all distinct rules with budgets at most three, and the measured three-label K-Margin rules. The expected-budget library additionally admits the measured five- and ten-label rules.

Table 34: All distinct measured frozen-PACS budget/count rules. Each acquisition block reports gain G, negative batch contribution $H _ { B }$ , and sample-new-error percentage S. Thresholds that induce the same signed count are represented once.

<table><tr><td>m k</td><td>Uniform G  $H _ { B }$ </td><td>Disagreement Symmetric KL G  $H _ { B }$  S G</td></tr><tr><td>S</td><td></td><td> $H _ { B }$  S 1 1 2.42 0.40 2.22 5.05 0.51 3.89 5.85 1.00 5.36</td></tr><tr><td></td><td></td><td>31 5.15 0.72 4.30 7.03 0.69 5.16 7.251.15 6.15</td></tr><tr><td></td><td></td><td>3 2 1.86 0.19 1.18 4.26 0.24 2.31 5.20 0.51 2.66 33 0.33 0.00 0.10 1.53 0.03 0.62 2.78 0.12 0.90</td></tr><tr><td></td><td></td><td>51 6.51 0.74 4.87 7.56 1.01 6.26 7.63 1.20 6.80 5 2 3.71 0.38 2.23 6.04 0.47 3.63 6.72 0.53 3.53</td></tr></table>

A mixture draws its rule before acquiring labels. Its expected gain, cost and query count are the corresponding convex combinations of measured operating points. Table 35 gives two fitted examples. Mixing the disagreement rule with fixed BF yields gain 7.6495 points at the same negative batch cost as the 7.2491-point symmetric-KL rule, using 1.9522 expected labels per batch. The mixture introduces more individual errors, so this comparison changes when S is the constrained cost. The second mixture uses two distinct disagreement counts. These fitted values characterize the observed rule library.

Table 35: Expected outcomes of two fitted rule mixtures and their symmetric-KL targets at matched negative batch cost. The mixtures use high-endpoint probabilities 0.3492647 and 0.5957447, respectively.
<table><tr><td>Mixture family</td><td>Output</td><td>G</td><td> $H _ { B }$ </td><td>S Labels/batch</td></tr><tr><td>Disagreement + fixed BF Mixture</td><td></td><td></td><td>7.64951.14567.2663</td><td>1.9522</td></tr><tr><td>Disagreement + fixed BF Symmetric KL 7.2491 1.1456 6.1468</td><td></td><td></td><td></td><td>3.0000</td></tr><tr><td>Two disagreement counts Mixture</td><td></td><td></td><td>5.9117 0.50544.0110</td><td>3.0000</td></tr><tr><td>Two disagreement counts Symmetric KL 5.1986 0.5054 2.6619</td><td></td><td></td><td></td><td>3.0000</td></tr></table>

For Table 36, the alternative library excludes symmetric-KL rules and retains K-Margin. Each of the $5 ^ { 5 } = 3$ , 125 ordered stream resamples recomputes gains and costs, then refits the frontier at the resampled symmetric-KL cost. The resulting percentile ranges include variation of the fitted probabilities. They condition on the common images, source checkpoint and measured rules. The fixed-I endpoint keeps the complete frontier feasible when an interpolation between two particular rules cannot attain the target cost.

Table 36: Symmetric-KL gain minus the empirical frontier without symmetric-KL rules, including K-Margin in the alternative library. Every one of the 3,125 ordered stream resamples refits the frontier. Bounds are the 2.5th and 97.5th descriptive percentiles, in percentage points.

<table><tr><td>Cost k Point gap</td><td>2.5% 97.5%</td></tr><tr><td>Batch net cost 1</td><td>-0.4004-0.5453-0.2555</td></tr><tr><td>Batch net cost 2</td><td>-0.7131-1.5901 0.7030</td></tr><tr><td>Batch net cost 3</td><td>0.1090-2.5777 1.2591</td></tr><tr><td>New errors 1</td><td>-0.0717-0.2560 0.1362</td></tr><tr><td>New errors 2 New errors 3 0.7927 0.0611</td><td>0.5973-0.1205 1.1252</td></tr></table>

## N.3 SCORING PARTITIONS AND NATURAL GROUPS

Table 37 changes the unit at which negative contributions are accumulated. Every image keeps its prediction and every batch keeps its decision. Coarsening the scoring partition allows positive and negative contributions to cancel, reducing $H _ { B }$ while leaving G and S unchanged. This control isolates the scoring definition from a change in the network’s input batch size.

Table 37: Negative contribution when the same selected outputs are scored at different grouping scales. Only the scoring partition changes. Individual-image groups yield S, and original-batch groups yield $H _ { B }$
<table><tr><td>Output</td><td>Image</td><td></td><td>Batch 2 batches</td><td>4 batches</td><td>8 batches</td><td>Domain Stream</td><td></td></tr><tr><td>Always BF</td><td>11.1817</td><td>1.9976</td><td>1.2130</td><td>0.7220</td><td>0.3995</td><td>0.0000</td><td>0.0000</td></tr><tr><td>Disagreement</td><td></td><td>5.16490.6883</td><td>0.5247</td><td>0.3418</td><td>0.1877</td><td></td><td>0.0000 0.0000</td></tr><tr><td>Symmetric KL</td><td>6.14681.1456</td><td></td><td>0.8857</td><td>0.6691</td><td>0.4043</td><td></td><td>0.0193 0.0000</td></tr></table>

Table 38 retains all 21 domain-by-class groups. The same frozen BF intervention that improves overall accuracy can worsen a particular group’s accuracy. Matching identities across the five orders also identifies repeated errors: fixed BF harms 330 evaluation images in all five orders, compared with 35 under symmetric-KL selection and 15 under the reference disagreement rule. These counts describe recurring outcomes on the shared evaluation images.

Table 38: All domain-by-class gains on the permanent PACS evaluation population. Each group denominator is fixed across the five orders. The two query rules use three labels and k=1.
<table><tr><td>Domain Class</td><td></td><td> $n _ { E }$ </td><td>Always BF I</td><td>Disagreement Symmetric  $\mathrm { K L }$ </td></tr><tr><td>Art</td><td>dog</td><td>183</td><td>-6.23</td><td>-3.72 -2.40</td></tr><tr><td>Art</td><td>elephant 123</td><td></td><td>9.43</td><td>5.20 4.88</td></tr><tr><td>Art</td><td>giraffe</td><td>131</td><td>-10.08</td><td>-5.19 -3.66</td></tr><tr><td>Art</td><td>guitar</td><td>91</td><td>-4.84</td><td>-2.86 -2.42</td></tr><tr><td>Art</td><td>horse</td><td>92</td><td>-3.26</td><td>-2.17 -2.17</td></tr><tr><td>Art</td><td>house</td><td>149</td><td>1.48</td><td>0.94 0.81</td></tr><tr><td>Art</td><td>person</td><td>238</td><td>29.83</td><td>17.56 13.61</td></tr><tr><td>Cartoon dog</td><td></td><td>218</td><td>8.62</td><td>6.88 8.35</td></tr><tr><td></td><td>Cartoon elephant 222</td><td></td><td>31.17</td><td>25.68 30.09</td></tr><tr><td>Cartoon giraffe</td><td></td><td>172</td><td>28.84</td><td>22.67 27.21</td></tr><tr><td>Cartoon guitar</td><td></td><td>65</td><td>-3.38</td><td>-3.38 -3.38</td></tr><tr><td>Cartoon horse</td><td></td><td>168</td><td>10.36</td><td>8.33 9.40</td></tr><tr><td>Cartoon house</td><td></td><td>137</td><td>5.84</td><td>4.53 5.69</td></tr><tr><td>Cartoon person</td><td></td><td>193</td><td>60.41</td><td>50.88 56.06</td></tr><tr><td>Sketch</td><td>dog</td><td>381</td><td>14.17</td><td>5.98 6.61</td></tr><tr><td>Sketch</td><td>elephant 371</td><td></td><td>43.56</td><td>19.57 22.59</td></tr><tr><td>Sketch</td><td>giräffe</td><td>392</td><td>-34.23</td><td>-12.14 -16.73</td></tr><tr><td>Sketch</td><td>guitar</td><td>313</td><td>-20.45</td><td>-8.31 -11.69</td></tr><tr><td>Sketch</td><td>horse</td><td>393</td><td>-8.85</td><td>-2.85 -4.99</td></tr><tr><td>Sketch</td><td>house</td><td>35</td><td>78.29</td><td>34.29 36.00</td></tr><tr><td>Sketch</td><td>person</td><td>88</td><td>28.86</td><td>12.50 15.68</td></tr></table>

## O GAUSSIAN READOUT ADAPTATION AND PORTFOLIO EXPANSION

## O.1 EPISODIC STATA ON SHARED CLIP REPRESENTATIONS

StatA (Zanella et al., 2025) adapts a Gaussian classifier initialized from text prototypes. It combines class assignments, a feature-neighbor graph, and updates to class means and covariances. We use the authors’ solver with original-image CLIP ViT-L/14 features and the same eight-template prototypes as the fixed readouts. Each batch starts from these prototypes. The data include all 9,991 PACS and 7,510 VLCS instances, in five stream orders with batches of 64. StatA sees the full batch’s unlabeled features and uses zero target labels. We score both the full data and the permanent evaluation set E.

StatA’s anchor strength $\alpha ,$ initial-assignment weight $\lambda _ { \hat { y } } ,$ , and graph regularization weigh $\lambda _ { \mathrm { L a p l a c i a n } }$ are all 1. Its mixing coefficient β balances batch-estimated statistics against the text anchor and is updated from hard class counts. We run ten mean/covariance iterations with three neighbors. The final three-image LABELME batch uses two neighbors. Table 39 gives full-data and common-E performance; Tables 40 and 41 give every domain.

On VLCS, StatA raises full-data equal-domain accuracy from the fixed U8V readout’s 83.793% to 84.754%. On common $E ,$ the corresponding scores are 84.097% and 85.106%. The LABELME improvement contributes strongly to this result. PACS exhibits a different ordering: StatA attains 98.445% on the full data and 98.404% on E, compared with $\mathrm { U } 8 \mathrm { V } { \mathrm { : } }$ 98.517% and 98.437%. This variation motivates retaining both readouts in the portfolio.

Table 39: Episodic StatA and fixed CLIP readouts. Accuracy is equal-domain mean ± stream-order SD. Fixed readouts have no order variation; all rows use zero target labels.
<table><tr><td>Dataset</td><td>Readout</td><td>Full data (%)</td><td>Common E (%)</td><td>Labels</td></tr><tr><td>PACS</td><td>Prompt average</td><td> $9 8 . 3 9 3 \pm 0 . 0 0 0$ </td><td> $9 8 . 2 7 9 \pm 0 . 0 0 0$ </td><td>0</td></tr><tr><td>PACS</td><td>Eight views</td><td> $9 8 . 5 1 7 \pm 0 . 0 0 0$ </td><td> $9 8 . 4 3 7 \pm 0 . 0 0 0$ </td><td>0</td></tr><tr><td>PACS</td><td>Mean logits</td><td> $9 8 . 5 0 2 \pm 0 . 0 0 0$ </td><td> $9 8 . 3 9 0 \pm 0 . 0 0 0$ </td><td>0</td></tr><tr><td>PACS</td><td>StatA</td><td> $9 8 . 4 4 5 \pm 0 . 0 5 1$ </td><td> $9 8 . 4 0 4 \pm 0 . 0 6 3$ </td><td>0</td></tr><tr><td>VLCS</td><td>Prompt average</td><td> $8 3 . 0 9 9 \pm 0 . 0 0 0$ </td><td> $8 3 . 5 2 3 \pm 0 . 0 0 0$ </td><td>0</td></tr><tr><td>VLCS</td><td>Eight views</td><td> $8 3 . 7 9 3 \pm 0 . 0 0 0$ </td><td> $8 4 . 0 9 7 \pm 0 . 0 0 0$ </td><td>0</td></tr><tr><td>VLCS</td><td>Mean logits</td><td> $8 3 . 1 7 6 \pm 0 . 0 0 0$ </td><td> $8 3 . 6 5 0 \pm 0 . 0 0 0$ </td><td>0</td></tr><tr><td>VLCS</td><td>StatA</td><td> $8 4 . 7 5 4 \pm 0 . 0 8 1$ </td><td> $8 5 . 1 0 6 \pm 0 . 1 3 7$ </td><td>0</td></tr></table>

Table 40: Domain-level StatA comparison on full data. Values are accuracy means ± stream-order SD over five orders.
<table><tr><td>Dataset</td><td>Domain</td><td>Readout</td><td>Accuracy (%)</td></tr><tr><td>PACS</td><td>Art</td><td>Eight views</td><td> $9 9 . 1 7 0 \pm 0 . 0 0 0$ </td></tr><tr><td>PACS</td><td>Art</td><td>StatA</td><td> $9 9 . 2 1 9 \pm 0 . 0 3 5$ </td></tr><tr><td>PACS</td><td>Cartoon</td><td>Eight views</td><td> $9 9 . 6 1 6 \pm 0 . 0 0 0$ </td></tr><tr><td>PACS</td><td>Cartoon</td><td>StatA</td><td> $9 9 . 7 2 7 \pm 0 . 0 3 8$ </td></tr><tr><td>PACS</td><td>Photo</td><td>Eight views</td><td> $9 9 . 9 4 0 \pm 0 . 0 0 0$ </td></tr><tr><td>PACS</td><td>Photo</td><td>StatA</td><td> $1 0 0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>PACS</td><td>Sketch</td><td>Eight views</td><td> $9 5 . 3 4 2 \pm 0 . 0 0 0$ </td></tr><tr><td>PACS</td><td>Sketch</td><td>StatA</td><td> $9 4 . 8 3 3 \pm 0 . 1 7 2$ </td></tr><tr><td>VLCS</td><td>Caltech</td><td>Eight views</td><td> $1 0 0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>VLCS</td><td>Caltech</td><td>StatA</td><td> $1 0 0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>VLCS</td><td>LabelMe</td><td>Eight views</td><td> $7 1 . 3 8 2 \pm 0 . 0 0 0$ </td></tr><tr><td>VLCS</td><td>LabelMe</td><td>StatA</td><td> $7 5 . 1 4 8 \pm 0 . 1 0 1$ </td></tr><tr><td>VLCS</td><td>Pascal</td><td>Eight views</td><td> $8 6 . 0 3 5 \pm 0 . 0 0 0$ </td></tr><tr><td>VLCS</td><td>Pascal</td><td>StatA</td><td> $8 5 . 7 2 2 \pm 0 . 2 3 1$ </td></tr><tr><td>VLCS</td><td>SUN</td><td>Eight views</td><td> $7 7 . 7 5 4 \pm 0 . 0 0 0$ </td></tr><tr><td>VLCS</td><td>SUN</td><td>StatA</td><td> $7 8 . 1 4 5 \pm 0 . 3 9 2$ </td></tr></table>

The resource measurements in Table 42 come from a dedicated rerun launched without another GPU process. They cover synchronized solver calls, including feature transfers and final CPU outputs; CLIP feature extraction is outside the timing. PACS and VLCS take $7 . 9 5 \pm 0 . 2 4$ and $5 . 9 2 \pm 0 . 0 3$ seconds per complete four-domain stream. The outputs and probabilities exactly match the original full run. The reported anchor difference is the maximum absolute gap between the solver’s initial probabilities and the cached universal-template probabilities. StatA consumes one original-image embedding per instance, whereas U8V combines eight image views. The Gaussian readout is reset after each batch, so its cost and output depend on batch composition without creating a carried adaptation state.

Table 41: Domain-level StatA comparison on common E. Values are accuracy means $\pm$ streamorder SD over five orders.
<table><tr><td>Dataset</td><td>Domain</td><td>Readout</td><td>Accuracy (%)</td></tr><tr><td>PACS</td><td>Art</td><td>Eight views</td><td> $9 9 . 0 6 9 \pm 0 . 0 0 0$ </td></tr><tr><td>PACS</td><td>Art</td><td>StatA</td><td> $9 9 . 2 5 5 \pm 0 . 0 4 6$ </td></tr><tr><td>PACS</td><td>Cartoon</td><td>Eight views</td><td> $9 9 . 5 7 0 \pm 0 . 0 0 0$ </td></tr><tr><td>PACS</td><td>Cartoon</td><td>StatA</td><td> $9 9 . 7 2 5 \pm 0 . 0 9 4$ </td></tr><tr><td>PACS</td><td>Photo</td><td>Eight views</td><td> $1 0 0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>PACS</td><td>Photo</td><td>StatA</td><td> $1 0 0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>PACS</td><td>Sketch</td><td>Eight views</td><td> $9 5 . 1 0 8 \pm 0 . 0 0 0$ </td></tr><tr><td>PACS</td><td>Sketch</td><td>StatA</td><td> $9 4 . 6 3 4 \pm 0 . 1 7 0$ </td></tr><tr><td>VLCS</td><td>Caltech</td><td>Eight views</td><td> $1 0 0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>VLCS</td><td>Caltech</td><td>StatA</td><td> $1 0 0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>VLCS</td><td>LabelMe</td><td>Eight views</td><td> $7 1 . 3 5 4 \pm 0 . 0 0 0$ </td></tr><tr><td>VLCS</td><td>LabelMe</td><td>StatA</td><td> $7 5 . 8 0 3 \pm 0 . 4 1 0$ </td></tr><tr><td>VLCS</td><td>Pascal</td><td>Eight views</td><td> $8 6 . 3 5 6 \pm 0 . 0 0 0$ </td></tr><tr><td>VLCS</td><td>Pascal</td><td>StatA</td><td> $8 5 . 6 3 0 \pm 0 . 3 8 4$ </td></tr><tr><td>VLCS</td><td>SUN</td><td>Eight views</td><td> $7 8 . 6 7 7 \pm 0 . 0 0 0$ </td></tr><tr><td>VLCS</td><td>SUN</td><td>StatA</td><td> $7 8 . 9 9 0 \pm 0 . 4 7 5$ </td></tr></table>

Table 42: StatA resource measurements from an isolated rerun. Time is seconds per four-domain stream; memory is peak allocated GPU memory. The final three-image batch uses two neighbors. Initialization $| \Delta p |$ is the largest difference between the solver’s initial probabilities and the stored prompt-average probabilities.
<table><tr><td>Dataset</td><td>Time (s)</td><td>Peak MiB</td><td>Final-batch adjustments</td><td>Initialization  $| \Delta p |$ </td></tr><tr><td>PACS</td><td> $7 . 9 5 \pm 0 . 2 4$ </td><td>24.51</td><td>0</td><td> $1 . 7 5 \mathrm { e } { - } 0 6 $ </td></tr><tr><td>VLCS</td><td> $5 . 9 2 \pm 0 . 0 3$ </td><td>20.33</td><td>5</td><td> $1 . 8 3 \mathrm { e } { - } 0 6$ </td></tr></table>

## O.2 ADDING THE GAUSSIAN READOUT AS A FIFTH CANDIDATE

We append StatA to the original single-template, universal-template, U2V, and U8V readouts. Each StatA prediction comes from the exact stream order and batch in which the five-candidate comparison is evaluated. The original four logits are normalized with log-softmax. StatA probabilities are floored at $1 0 ^ { - 1 2 }$ and converted to normalized log probabilities; their hard predictions are preserved. Mean-logit prediction averages these normalized score vectors. U8V remains the declared default, including empirical-error ties, in both portfolios.

Uniform, disagreement-first, and maximum pairwise symmetric-KL selection each use up to three current-batch Q labels. Uniform uses identical query sets for the four- and five-candidate comparisons. The other rules can change their queries when StatA changes the disagreement pattern. PORTAL keeps prior mass 1, predicted-class mass 0.75, and label price 0.002. Three query credits arrive per batch; unused credits carry forward, with at most twelve queries in one batch. Each candidate set has its own history. The batch oracle uses current E labels to choose one candidate for the whole batch. We rerun the empirical query rules in paired four- and five-candidate comparisons. Fixed outputs and PORTAL reproduce their original scores.

Tables 43 and 44 show every fixed readout, the zero-label mean, and the query-based rules. On VLCS, adding StatA raises batch-oracle accuracy from 85.637% to 86.769%. PORTAL rises from 83.919% to 84.576%, with mean query counts increasing from 89.4 to 96.0. The candidate adds 1.132 percentage points of attainable batch accuracy, of which this selection rule realizes 0.657 points. The new mean-logit rule reaches 84.732%, and fixed StatA reaches 85.106%, both with zero labels. These comparisons expose the remaining selection loss alongside the additional capability.

On PACS, the batch oracle rises from 98.646% to 98.774%, while PORTAL stays at 98.442%. Meanlogit accuracy improves from 98.390% to 98.472%. StatA helps through averaging even when selection changes little. Its additional computation is measured separately. Its predictions are recomputed for each order and batch, then stored for the selection experiment.

Table 43: PACS portfolio expansion on common E. Accuracy is equal-domain mean $\pm$ streamorder SD. The four-action portfolio contains the original, prompt-average, two-view and eight-view readouts; the fifth action is episodic StatA. Empirical-query rows are paired reruns for this expansion.
<table><tr><td rowspan="2">Rule</td><td colspan="2">Four readouts</td><td colspan="2">Add StatA</td></tr><tr><td>Accuracy (%)</td><td>Labels</td><td>Accuracy (%)</td><td>Labels</td></tr><tr><td>Original view</td><td> $9 8 . 1 9 1 \pm 0 . 0 0 0$ </td><td>0.0</td><td> $9 8 . 1 9 1 \pm 0 . 0 0 0$ </td><td>0.0</td></tr><tr><td>Prompt average</td><td> $9 8 . 2 7 9 \pm 0 . 0 0 0$ </td><td>0.0</td><td> $9 8 . 2 7 9 \pm 0 . 0 0 0$ </td><td>0.0</td></tr><tr><td>Two views</td><td> $9 8 . 4 2 5 \pm 0 . 0 0 0$ </td><td>0.0</td><td> $9 8 . 4 2 5 \pm 0 . 0 0 0$ </td><td>0.0</td></tr><tr><td>Eight views</td><td> $9 8 . 4 3 7 \pm 0 . 0 0 0$ </td><td>0.0</td><td> $9 8 . 4 3 7 \pm 0 . 0 0 0$ </td><td>0.0</td></tr><tr><td>StatA</td><td></td><td></td><td>98.404 ± 0.063</td><td>0.0</td></tr><tr><td>Mean logits</td><td> $9 8 . 3 9 0 \pm 0 . 0 0 0$ </td><td>0.0</td><td> $9 8 . 4 7 2 \pm 0 . 0 5 5$ </td><td>0.0</td></tr><tr><td>Uniform ERM</td><td> $9 8 . 4 2 9 \pm 0 . 0 0 7$ </td><td>473.6</td><td> $9 8 . 4 2 4 \pm 0 . 0 0 9$ </td><td>473.6</td></tr><tr><td>Disagreement ERM</td><td> $9 8 . 4 3 4 \pm 0 . 0 3 1$ </td><td>473.6</td><td> $9 8 . 4 1 9 \pm 0 . 0 3 4$ </td><td>473.6</td></tr><tr><td>Maximum-KL ERM</td><td> $9 8 . 4 3 9 \pm 0 . 0 2 9$ </td><td>473.6</td><td> $9 8 . 4 2 2 \pm 0 . 0 3 3$ </td><td>473.6</td></tr><tr><td>PORTAL</td><td> $9 8 . 4 4 2 \pm 0 . 0 0 7$ </td><td>1.4</td><td> $9 8 . 4 4 2 \pm 0 . 0 1 7$ </td><td>2.4</td></tr><tr><td>Batch oracle</td><td> $9 8 . 6 4 6 \pm 0 . 0 1 6$ </td><td></td><td> $9 8 . 7 7 4 \pm 0 . 0 3 5$ </td><td></td></tr></table>

Table 44: VLCS portfolio expansion on common $E .$ Accuracy is equal-domain mean $\pm$ streamorder SD. The four-action portfolio contains the original, prompt-average, two-view and eight-view readouts; the fifth action is episodic StatA. Empirical-query rows are paired reruns for this expansion.
<table><tr><td rowspan="2">Rule</td><td colspan="2">Four readouts</td><td colspan="2">Add StatA</td></tr><tr><td>Accuracy (%)</td><td>Labels</td><td>Accuracy (%)</td><td>Labels</td></tr><tr><td>Original view</td><td> $8 0 . 8 4 5 \pm 0 . 0 0 0$ </td><td>0.0</td><td> $8 0 . 8 4 5 \pm 0 . 0 0 0$ </td><td>0.0</td></tr><tr><td>Prompt average</td><td> $8 3 . 5 2 3 \pm 0 . 0 0 0$ </td><td>0.0</td><td> $8 3 . 5 2 3 \pm 0 . 0 0 0$ </td><td>0.0</td></tr><tr><td>Two views</td><td> $8 3 . 8 6 1 \pm 0 . 0 0 0$ </td><td>0.0</td><td> $8 3 . 8 6 1 \pm 0 . 0 0 0$ </td><td>0.0</td></tr><tr><td>Eight views</td><td> $8 4 . 0 9 7 \pm 0 . 0 0 0$ </td><td>0.0</td><td> $8 4 . 0 9 7 \pm 0 . 0 0 0$ </td><td>0.0</td></tr><tr><td>StatA</td><td></td><td></td><td> $8 5 . 1 0 6 \pm 0 . 1 3 7$ </td><td>0.0</td></tr><tr><td>Mean logits</td><td> $8 3 . 6 5 0 \pm 0 . 0 0 0$ </td><td>0.0</td><td>84.732 ± 0.140</td><td>0.0</td></tr><tr><td>Uniform ERM</td><td> $8 3 . 9 0 7 \pm 0 . 2 0 2$ </td><td>354.8</td><td> $8 3 . 9 6 2 \pm 0 . 2 2 4$ </td><td>354.8</td></tr><tr><td>Disagreement ERM</td><td> $8 3 . 8 6 9 \pm 0 . 1 2 6$ </td><td>354.8</td><td> $8 4 . 1 0 9 \pm 0 . 0 9 5$ </td><td>354.8</td></tr><tr><td>Maximum-KL ERM</td><td> $8 3 . 9 1 8 \pm 0 . 1 5 8$ </td><td>354.8</td><td> $8 4 . 2 5 8 \pm 0 . 1 6 0$ </td><td>354.8</td></tr><tr><td>PORTAL</td><td> $8 3 . 9 1 9 \pm 0 . 1 7 0$ </td><td>89.4</td><td> $8 4 . 5 7 6 \pm 0 . 1 7 4$ </td><td>96.0</td></tr><tr><td>Batch oracle</td><td> $8 5 . 6 3 7 \pm 0 . 1 4 8$ </td><td></td><td> $8 6 . 7 6 9 \pm 0 . 0 7 3$ </td><td></td></tr></table>

## P SUPERVISED CONVEX COMBINATIONS OF READOUT LOGITS

## P.1 A LEARNABLE READOUT AND TARGET-AWARE ACQUISITION

We also fit a weighted average of the original four readouts on PACS, OfficeHome, and VLCS. Let K be the candidate count and C the class count. Matrix $Z _ { x } \in \mathbb { R } ^ { K \times C }$ contains their logits for input x, with class column $z _ { x , c } .$ The weights $w \in \Delta _ { K }$ are nonnegative and sum to one, giving combined logits $Z _ { x } ^ { \top } w .$ . Let $Q _ { \mathrm { o b s } }$ be the acquired inputs and $y _ { q }$ their labels. We regularize toward uniform weights u with strength $\lambda > 0$ by minimizing

$$
\sum _ { \boldsymbol { q } \in Q _ { \mathrm { o b s } } } \left[ \log \sum _ { c } \exp ( \boldsymbol { w } ^ { \top } \boldsymbol { z } _ { \boldsymbol { q } , c } ) - \boldsymbol { w } ^ { \top } \boldsymbol { z } _ { \boldsymbol { q } , \boldsymbol { y } _ { \boldsymbol { q } } } \right] + \frac { \lambda } { 2 } \| \boldsymbol { w } - \boldsymbol { u } \| _ { 2 } ^ { 2 } , \qquad \boldsymbol { u } = \frac { 1 } { K } \mathbf { 1 } .\tag{81}
$$

The regularizer favors uniform averaging when evidence is limited. We subtract each candidate’s mean class logit, which leaves the mixture probabilities unchanged. Sequential least-squares programming (SLSQP) uses analytic gradients, tolerance $1 0 ^ { - 9 }$ , and at most 200 iterations. Weights reset to u each batch and are refitted after each query, up to three labels. The primary regularization strength is $\lambda = 1 ;$ ; sensitivity comparisons use 0.1 and 10.

Let $\ell _ { x } ( w , y )$ be the cross-entropy loss of the mixture for label $y ,$ and let $p _ { x } = \operatorname { s o f t m a x } ( Z _ { x } ^ { \top } w )$ . Its gradient and Hessian are

$$
\nabla _ { w } \ell _ { x } ( w , y ) = Z _ { x } ( p _ { x } - e _ { y } ) ,\tag{82}
$$

$$
H _ { x } = \nabla _ { w } ^ { 2 } \ell _ { x } ( w , y ) = Z _ { x } \left[ \mathrm { d i a g } ( p _ { x } ) - p _ { x } p _ { x } ^ { \top } \right] Z _ { x } ^ { \top } .\tag{83}
$$

The quadratic form of $H _ { x }$ is a predictive variance, so $H _ { x }$ is positive semidefinite. The positive regularizer makes the objective strictly convex on the simplex. Under the current categorical model $Y \sim p _ { x }$ , the gradient covariance also equals $H _ { x }$ , connecting curvature to paired candidate-gradient uncertainty.

We rank queries by a local interior curvature approximation in the sum-zero simplex subspace. Let B have orthonormal columns spanning that subspace, $\bar { H } _ { x } = B ^ { \top } H _ { x } B$ , and

$$
A = \lambda I + \sum _ { q \in Q _ { \mathrm { o b s } } } \bar { H } _ { q } , \qquad \bar { H } _ { E } = \frac { 1 } { | E | } \sum _ { e \in E } \bar { H } _ { e } .\tag{84}
$$

We call the following acquisition score target-Fisher, because it uses the curvature of the target prediction loss:

$$
s ( x ) = \frac { 1 } { 2 } \operatorname { t r } \left[ \bar { H } _ { E } \left( A ^ { - 1 } - ( A + \bar { H } _ { x } ) ^ { - 1 } \right) \right] .\tag{85}
$$

This score measures prospective covariance reduction weighted by target curvature. Its E dependence uses unlabeled prediction vectors. After each query, the fitted weights and curvature are updated before ranking the remaining $Q$ examples.

## P.2 READOUT FITTING AND ACQUISITION OUTCOMES

Table 45 compares uniform, disagreement-first, maximum symmetric-KL, and target-Fisher queries at $\lambda = 1$ . Each fitted mixture is paired with ERM on exactly the same queries, using U8V on ties. Fixed U8V and mean logits provide zero-label references. PORTAL uses its accumulated evidence and actual query budget. All OfficeHome instances remain included. The convex experiment skips batches without evaluation inputs, using 737.4 labels on average; the discrete-selector experiment uses 738.0.

Fitting improves on uniform averaging for OfficeHome and VLCS. At the primary regularizer, target-Fisher reaches 88.735% and 84.030%, respectively, compared with mean logits at 88.237% and 83.650%. Maximum symmetric-KL reaches 88.757% and 84.115%, so the curvature score does not improve this acquisition comparison. Fixed U8V reaches 89.008% and 84.097%. On PACS, the fitted target-Fisher mixture gives 98.387%, close to mean logits at 98.390% and below U8V at 98.437%. Cross-entropy and Brier score measure probability quality; the Brier score is the squared distance to the one-hot label. These metrics and all 36 regularizer/acquisition settings are retained in Tables 46, 47, and 48. The readout fit supplies an additional learnable candidate, while its advantages over the strongest fixed readout remain setting-dependent.

Table 45: Primary convex-readout results (λ = 1) on common E. Accuracy is equal-domain mean ± stream-order SD; labels average five orders. Discrete ERM rows use the same queries as the corresponding mixture.
<table><tr><td></td><td colspan="2">PACS</td><td colspan="2">OfficeHome</td><td colspan="2">VLCS</td></tr><tr><td>Rule</td><td>Acc.</td><td>Labels</td><td>Acc.</td><td>Labels</td><td>Acc.</td><td>Labels</td></tr><tr><td>Eight views</td><td> $9 8 . 4 3 7 \pm 0 . 0 0 0$ </td><td>0.0</td><td> $8 9 . 0 0 8 \pm 0 . 0 0 0$ </td><td>0.0</td><td> $8 4 . 0 9 7 \pm 0 . 0 0 0$ </td><td>0.0</td></tr><tr><td>Mean logits</td><td> $9 8 . 3 9 0 \pm 0 . 0 0 0$ </td><td>0.0</td><td> $8 8 . 2 3 7 \pm 0 . 0 0 0$ </td><td>0.0</td><td> $8 3 . 6 5 0 \pm 0 . 0 0 0$ </td><td>0.0</td></tr><tr><td>PORTAL</td><td> $9 8 . 4 4 2 \pm 0 . 0 0 7$ </td><td>1.4</td><td> $8 9 . 0 1 0 \pm 0 . 0 1 2$ </td><td>7.4</td><td> $8 3 . 9 1 9 \pm 0 . 1 7 0$ </td><td>89.4</td></tr><tr><td>Uniform / mixture</td><td> $9 8 . 3 9 8 \pm 0 . 0 1 4$ </td><td>473.6</td><td> $8 8 . 4 1 5 \pm 0 . 0 9 8$ </td><td>737.4</td><td> $8 3 . 8 6 8 \pm 0 . 2 2 8$ </td><td>354.8</td></tr><tr><td>Disagreement / mixture</td><td> $9 8 . 4 1 5 \pm 0 . 0 0 9$ </td><td>473.6</td><td> $8 8 . 6 3 6 \pm 0 . 0 6 8$ </td><td>737.4</td><td> $8 4 . 0 0 5 \pm 0 . 1 3 9$ </td><td>354.8</td></tr><tr><td>Maximum KL / mixture</td><td> $9 8 . 3 9 3 \pm 0 . 0 1 4$ </td><td>473.6</td><td> $8 8 . 7 5 7 \pm 0 . 0 8 7$ </td><td>737.4</td><td> $8 4 . 1 1 5 \pm 0 . 0 8 3$ </td><td>354.8</td></tr><tr><td>Target Fisher / mixture</td><td> $9 8 . 3 8 7 \pm 0 . 0 1 4$ </td><td>473.6</td><td> $8 8 . 7 3 5 \pm 0 . 1 0 9$ </td><td>737.4</td><td> $8 4 . 0 3 0 \pm 0 . 1 3 7$ </td><td>354.8</td></tr><tr><td>Uniform / discrete ERM</td><td> $9 8 . 4 2 9 \pm 0 . 0 0 7$ </td><td>473.6</td><td> $8 8 . 8 6 3 \pm 0 . 1 2 1$ </td><td>737.4</td><td> $8 3 . 9 0 7 \pm 0 . 2 0 2$ </td><td>354.8</td></tr><tr><td>Disagreement / discrete ERM</td><td> $9 8 . 4 3 4 \pm 0 . 0 3 1$ </td><td>473.6</td><td> $8 8 . 6 3 7 \pm 0 . 1 1 8$ </td><td>737.4</td><td> $8 3 . 8 6 9 \pm 0 . 1 2 6$ </td><td>354.8</td></tr><tr><td>Maximum KL / discrete ERM</td><td> $9 8 . 4 3 9 \pm 0 . 0 2 9$ </td><td>473.6</td><td> $8 8 . 6 8 8 \pm 0 . 1 2 2$ </td><td>737.4</td><td> $8 3 . 9 1 8 \pm 0 . 1 5 8$ </td><td>354.8</td></tr><tr><td>Target Fisher / discrete ERM</td><td> $9 8 . 4 3 9 \pm 0 . 0 2 9$ </td><td>473.6</td><td> $8 8 . 6 8 5 \pm 0 . 1 2 7$ </td><td>737.4</td><td> $8 3 . 8 5 0 \pm 0 . 0 5 2$ </td><td>354.8</td></tr></table>

Table 46: Complete convex-readout grid on PACS common E. Every acquisition uses up to three current-batch labels; means and SDs cover five stream orders. Accuracy, cross-entropy and Brier columns pair discrete ERM (D) with the fitted mixture (M). Runtime is CPU mixture-fitting/replay time per stream.
<table><tr><td colspan="4"></td><td rowspan="2">Accuracy (%)</td><td colspan="2"></td><td colspan="2">CE</td><td colspan="2">Brier</td></tr><tr><td></td><td>λ Acquisition</td><td>Labels</td><td></td><td>D</td><td>M</td><td>D</td><td>M</td><td>D</td><td>M Fit (s)</td></tr><tr><td>0.1</td><td>Uniform</td><td>473.6</td><td> $9 8 . 4 2 9 \pm 0 . 0 0 7$ </td><td></td><td> $9 8 . 3 8 2 \pm 0 . 0 2 1$ </td><td>0.0580</td><td>0.0604</td><td>0.0252</td><td>0.0264</td><td>0.821</td></tr><tr><td>0.1</td><td>Disagreement</td><td>473.6</td><td> $9 8 . 4 3 4 \pm 0 . 0 3 1$ </td><td></td><td> $9 8 . 3 9 3 \pm 0 . 0 3 3$ </td><td>0.0584</td><td>0.0603</td><td>0.0253</td><td>0.0262</td><td>0.825</td></tr><tr><td>0.1</td><td>Maximum KL</td><td>473.6</td><td> $9 8 . 4 3 9 \pm 0 . 0 2 9$ </td><td></td><td> $9 8 . 4 0 7 \pm 0 . 0 4 9$ </td><td>0.0584</td><td>0.0597</td><td>0.0253</td><td>0.0259</td><td>1.028</td></tr><tr><td>0.1</td><td>Target Fisher</td><td>473.6</td><td> $9 8 . 4 3 4 \pm 0 . 0 3 4$ </td><td></td><td> $9 8 . 4 1 7 \pm 0 . 0 6 9$ </td><td>0.0583</td><td>0.0595</td><td>0.0253</td><td>0.0258</td><td>1.167</td></tr><tr><td>1</td><td>Uniform</td><td>473.6</td><td> $9 8 . 4 2 9 \pm 0 . 0 0 7$ </td><td></td><td> $9 8 . 3 9 8 \pm 0 . 0 1 4$ </td><td>0.0580</td><td>0.0604</td><td>0.0252</td><td>0.0264</td><td>0.503</td></tr><tr><td>1</td><td>Disagreement</td><td>473.6</td><td> $9 8 . 4 3 4 \pm 0 . 0 3 1$ </td><td></td><td> $9 8 . 4 1 5 \pm 0 . 0 0 9$ </td><td>0.0584</td><td>0.0603</td><td>0.0253</td><td>0.0263</td><td>0.560</td></tr><tr><td>1</td><td>Maximum KL</td><td>473.6</td><td> $9 8 . 4 3 9 \pm 0 . 0 2 9$ </td><td></td><td> $9 8 . 3 9 3 \pm 0 . 0 1 4$ </td><td>0.0584</td><td>0.0597</td><td>0.0253</td><td>0.0260</td><td>0.706</td></tr><tr><td>1</td><td>Target Fisher</td><td>473.6</td><td> $9 8 . 4 3 9 \pm 0 . 0 2 9$ </td><td></td><td> $9 8 . 3 8 7 \pm 0 . 0 1 4$ </td><td>0.0584</td><td>0.0597</td><td>0.0253</td><td>0.0261</td><td>0.825</td></tr><tr><td>10</td><td>Uniform</td><td>473.6</td><td> $9 8 . 4 2 9 \pm 0 . 0 0 7$ </td><td></td><td> $9 8 . 3 8 8 \pm 0 . 0 1 1$ </td><td>0.0580</td><td>0.0604</td><td>0.0252</td><td>0.0265</td><td>0.597</td></tr><tr><td>10</td><td>Disagreement</td><td>473.6</td><td> $9 8 . 4 3 4 \pm 0 . 0 3 1$ </td><td></td><td> $9 8 . 3 9 3 \pm 0 . 0 0 6$ </td><td>0.0584</td><td>0.0604</td><td>0.0253</td><td>0.0264</td><td>0.705</td></tr><tr><td>10</td><td>Maximum KL</td><td>473.6</td><td> $9 8 . 4 3 9 \pm 0 . 0 2 9$ </td><td></td><td> $9 8 . 3 8 8 \pm 0 . 0 1 1$ </td><td>0.0584</td><td>0.0603</td><td>0.0253</td><td>0.0264</td><td>0.987</td></tr><tr><td>10</td><td>Target Fisher</td><td>473.6</td><td> $9 8 . 4 3 9 \pm 0 . 0 2 9$ </td><td></td><td> $9 8 . 3 8 8 \pm 0 . 0 1 1$ </td><td>0.0584</td><td>0.0603</td><td>0.0253</td><td>0.0264</td><td>1.101</td></tr></table>

Table 47: Complete convex-readout grid on OFFICEHOME common E. Every acquisition uses up to three current-batch labels; means and SDs cover five stream orders. Accuracy, cross-entropy and Brier columns pair discrete ERM (D) with the fitted mixture (M). Runtime is CPU mixturefitting/replay time per stream.
<table><tr><td></td><td></td><td></td><td colspan="2">Accuracy (%)</td><td colspan="2">M</td><td colspan="2">CE</td><td colspan="2">Brier</td></tr><tr><td></td><td>λ Acquisition</td><td>Labels</td><td></td><td>D</td><td></td><td>D</td><td>M</td><td>D</td><td></td><td>M Fit (s)</td></tr><tr><td>0.1</td><td>Uniform</td><td>737.4</td><td> $8 8 . 8 6 3 \pm 0 . 1 2 1$ </td><td></td><td> $8 8 . 5 2 3 \pm 0 . 1 4 4$ </td><td>0.3952</td><td>0.4105</td><td>0.1580</td><td>0.1623</td><td>1.464</td></tr><tr><td>0.1</td><td>Disagreement</td><td>737.4</td><td> $8 8 . 6 3 7 \pm 0 . 1 1 8$ </td><td></td><td> $8 8 . 5 7 9 \pm 0 . 0 9 4$ </td><td>0.4080</td><td>0.4060</td><td>0.1617</td><td>0.1609</td><td>1.283</td></tr><tr><td>0.1</td><td>Maximum KL</td><td>737.4</td><td> $8 8 . 6 8 8 \pm 0 . 1 2 2$ </td><td></td><td> $8 8 . 7 2 6 \pm 0 . 1 1 2$ </td><td>0.4018</td><td>0.4000</td><td>0.1601</td><td>0.1592</td><td>1.401</td></tr><tr><td>0.1</td><td>Target Fisher</td><td>737.4</td><td> $8 8 . 6 8 7 \pm 0 . 0 8 8$ </td><td></td><td> $8 8 . 7 2 4 \pm 0 . 0 7 7$ </td><td>0.4029</td><td>0.4006</td><td>0.1604</td><td>0.1595</td><td>1.570</td></tr><tr><td>1</td><td>Uniform</td><td>737.4</td><td> $8 8 . 8 6 3 \pm 0 . 1 2 1$ </td><td></td><td> $8 8 . 4 1 5 \pm 0 . 0 9 8$ </td><td>0.3952</td><td>0.4084</td><td>0.1580</td><td>0.1619</td><td>0.938</td></tr><tr><td>1</td><td>Disagreement</td><td>737.4</td><td> $8 8 . 6 3 7 \pm 0 . 1 1 8$ </td><td></td><td> $8 8 . 6 3 6 \pm 0 . 0 6 8$ </td><td>0.4080</td><td>0.4030</td><td>0.1617</td><td>0.1599</td><td>1.151</td></tr><tr><td>1</td><td>Maximum KL</td><td>737.4</td><td> $8 8 . 6 8 8 \pm 0 . 1 2 2$ </td><td></td><td> $8 8 . 7 5 7 \pm 0 . 0 8 7$ </td><td>0.4018</td><td>0.3982</td><td>0.1601</td><td>0.1586</td><td>1.377</td></tr><tr><td>1</td><td>Target Fisher</td><td>737.4</td><td> $8 8 . 6 8 5 \pm 0 . 1 2 7$ </td><td></td><td> $8 8 . 7 3 5 \pm 0 . 1 0 9$ </td><td>0.4020</td><td>0.3984</td><td>0.1603</td><td>0.1587</td><td>1.544</td></tr><tr><td>10</td><td>Uniform</td><td>737.4</td><td> $8 8 . 8 6 3 \pm 0 . 1 2 1$ </td><td></td><td> $8 8 . 2 9 8 \pm 0 . 0 4 0$ </td><td>0.3952</td><td>0.4107</td><td>0.1580</td><td>0.1627</td><td>1.272</td></tr><tr><td>10</td><td>Disagreement</td><td>737.4</td><td> $8 8 . 6 3 7 \pm 0 . 1 1 8$ </td><td></td><td>88.433 ± 0.076</td><td>0.4080</td><td>0.4071</td><td>0.1617</td><td>0.1614</td><td>1.684</td></tr><tr><td>10</td><td>Maximum KL</td><td>737.4</td><td> $8 8 . 6 8 8 \pm 0 . 1 2 2$ </td><td></td><td>88.555 ± 0.031</td><td>0.4018</td><td>0.4039</td><td>0.1601</td><td>0.1605</td><td>1.971</td></tr><tr><td>10</td><td>Target Fisher</td><td>737.4</td><td> $8 8 . 6 3 7 \pm 0 . 1 1 2$ </td><td></td><td>88.561 ± 0.027</td><td>0.4027</td><td>0.4036</td><td>0.1605</td><td>0.1604</td><td>2.135</td></tr></table>

Table 48: Complete convex-readout grid on VLCS common E. Every acquisition uses up to three current-batch labels; means and SDs cover five stream orders. Accuracy, cross-entropy and Brier columns pair discrete ERM (D) with the fitted mixture (M). Runtime is CPU mixture-fitting/replay time per stream.
<table><tr><td></td><td></td><td></td><td></td><td colspan="2">Accuracy (%)</td><td colspan="2">CE</td><td colspan="2">Brier</td><td></td></tr><tr><td>λ</td><td>Acquisition</td><td>Labels</td><td></td><td>D</td><td>M</td><td>D</td><td>M</td><td>D</td><td></td><td>M Fit (s)</td></tr><tr><td>0.1</td><td>Uniform</td><td>354.8</td><td>83.907 ±</td><td>0.202</td><td>83.819 ± 0.248</td><td>0.4629</td><td>0.4648</td><td>0.2402</td><td>0.2392</td><td>0.681</td></tr><tr><td>0.1</td><td>Disagreement</td><td>354.8</td><td>83.869 ±</td><td>0.126</td><td>83.956 ± 0.164</td><td>0.4651</td><td>0.4625</td><td>0.2404</td><td>0.2390</td><td>0.630</td></tr><tr><td>0.1</td><td>Maximum KL</td><td>354.8</td><td>83.918 ±</td><td>0.158</td><td>84.068 ± 0.151</td><td>0.4661</td><td>0.4603</td><td>0.2408</td><td>0.2371</td><td>0.666</td></tr><tr><td>0.1</td><td>Target Fisher</td><td>354.8</td><td>83.887 ±</td><td>0.181</td><td>83.901 ± 0.241</td><td>0.4659</td><td>0.4613</td><td>0.2409</td><td>0.2381</td><td>0.775</td></tr><tr><td>1</td><td>Uniform</td><td>354.8</td><td>83.907 ±</td><td>0.202</td><td>83.868 土 0.228</td><td>0.4629</td><td>0.4615</td><td>0.2402</td><td>0.2382</td><td>0.462</td></tr><tr><td>1</td><td>Disagreement</td><td>354.8</td><td>83.869 ±</td><td>0.126</td><td>84.005 ± 0.139</td><td>0.4651</td><td>0.4610</td><td>0.2404</td><td>0.2386</td><td>0.517</td></tr><tr><td>1</td><td>Maximum KL</td><td>354.8</td><td>83.918 ±</td><td>0.158</td><td>84.115 ± 0.083</td><td>0.4661</td><td>0.4584</td><td>0.2408</td><td>0.2366</td><td>0.562</td></tr><tr><td>1</td><td>Target Fisher</td><td>354.8</td><td>83.850 ± 0.052</td><td></td><td>84.030 ± 0.137</td><td>0.4661</td><td>0.4585</td><td>0.2407</td><td>0.2370</td><td>0.651</td></tr><tr><td>10</td><td>Uniform</td><td>354.8</td><td>83.907 ± 0.202</td><td></td><td>83.757 ± 0.068</td><td>0.4629</td><td>0.4641</td><td>0.2402</td><td>0.2408</td><td>0.653</td></tr><tr><td>10</td><td>Disagreement</td><td>354.8</td><td>83.869 ± 0.126</td><td></td><td>83.850 ± 0.144</td><td>0.4651</td><td>0.4630</td><td>0.2404</td><td>0.2404</td><td>0.755</td></tr><tr><td>10</td><td>Maximum KL</td><td>354.8</td><td>83.918 ± 0.158</td><td></td><td>83.842 ± 0.125</td><td>0.4661</td><td>0.4609</td><td>0.2408</td><td>0.2390</td><td>0.828</td></tr><tr><td></td><td>10 Target Fisher</td><td>354.8</td><td>83.876 ± 0.080</td><td></td><td>83.851 ± 0.095</td><td>0.4658</td><td>0.4607</td><td>0.2406</td><td>0.2390</td><td>0.926</td></tr></table>

## Q DYNAMIC CONDITIONS, SOURCE CHANGES AND CLASS-LABEL NOISE

Batch composition changes normalization-based predictions. We regenerate all four ResNet outputs for each condition and keep every target image once per stream. Three arrival orders are tested with four source checkpoints. For the Photo source, batch sizes are 16,32,64,128. At batch size 64, we test mixed, reversed, gradually overlapping, and class-block orders. Art, Cartoon, and Sketch source models are each evaluated on the remaining three domains. This gives 33 complete runs. The original query/evaluation roles remain fixed. Photo images, used as targets for the other source models, receive a content-based split before evaluation.

Arrival construction. Domain-block streams shuffle images within each domain. Reverse streams reverse the domain order while preserving their within-domain permutations. Mixed streams uniformly permute all target images. For gradual transitions, the within-domain normalized rank receives Gaussian jitter of standard deviation 0.45 before sorting across domains; overlapping regions retain every image once. Class-block streams group the benchmark’s labels within each domain, constructing temporally imbalanced arrivals. Labels define this offline evaluation arrangement and remain hidden from querying and deployment. Batches ending a block retain their original smaller size.

Every model uses the same released architecture and preprocessing as the primary PACS experiment. Source and batch-statistic branches are frozen before querying; batch-statistic normalization uses the actual current inputs and preserves running buffers. Selection history carries across the stream, and a new selector begins each condition/order. Three-label empirical rules and joint selection see the same candidate predictions and evaluation images. Small query pools reduce the attainable fixedthree budget, so tables report actual label counts.

Batch and domain effects. For Photo source, smaller batches reduce the fixed normalization-andflip accuracy: the batch-16 mean is 58.49%, compared with 59.87% at batch 128. The corresponding batch oracle changes its action granularity, so its accuracy is interpreted within each condition. Uniform mean logits remain a strong reference. Joint selection at batch 16 reaches 58.87% using 114.3 labels per stream, compared with 58.12% for maximum-KL selection using 1,561.7 labels. Across mixed-domain arrivals, joint selection reaches 60.16% with 114.7 labels and maximum-KL selection reaches 58.34% with 390.3. Fixed BN+flip reaches 60.56% in that same condition.

Temporally concentrated classes. Class-block arrivals materially alter batch statistics. The fixed BN+flip branch reaches 22.58%, while source accuracy is about 50.88%. Joint selection reaches 51.28% with 187.7 labels; maximum-KL selection reaches 54.11% with 393. Their batch-net harm values are 0.34 and 0.86 percentage points, respectively, versus 32.41 for fixed BN+flip. The result exposes a different evidence–accuracy tradeoff from mixed arrivals. It also demonstrates that a candidate strong under ordinary domain-block evaluation can require frequent local rejection under a changed temporal class distribution.

![](images/d4a6b525d7cd7d6e3c2b234af891b1eaf85331286eb1cac80dc517fef94bbbd3.jpg)  
Figure 11: Actual batch/source changes and label-channel stress tests. Panels (a)–(d) use regenerated predictions under three stream orders per condition; (e)–(f) use the primary five frozen-prediction orders with multiclass annotation replacement. Accuracy error bars are order standard deviations; label curves show mean actual acquisitions. Fixed output and mean-logit references retain zero feedback cost. Class-block arrivals change the normalization predictions, while label corruption leaves every candidate prediction fixed.

Source-model variation. The Art, Cartoon and Sketch checkpoints alter both the learned source representation and target domains. Joint accuracy is 77.91%,81.02%,63.00%, with 69.0,64.0,54.3 labels per target stream. The corresponding maximum-KL accuracies are 77.45%,80.97%,62.71%. Fixed BN+flip reaches 78.39%,81.41%,63.02%. Each released source checkpoint is repeated over three target stream orders.

## Q.1 MULTICLASS ANNOTATION NOISE

To test the annotation channel, every requested class label is replaced by a uniformly chosen different class with probability $\eta \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 4 \}$ . A corrupted class is shared by every candidate’s correctness evaluation. This can move rescue, harm and neutral outcomes in either direction, as occurs in multiclass annotation. Candidate logits and evaluation labels remain unchanged. Imagekeyed randomization couples noise across methods and noise levels; each acquired instance still costs one label.

The complete primary four-action PACS banks and all five stream orders are retained. Joint selection uses the same prior, label price, default, carryover rule and query cap as the uncorrupted experiment. Every zero-noise result reproduces its primary counterpart. The independently checked traces reconstruct acquired-label counts, actual class replacements and all evaluation errors from the prediction banks.

Joint accuracy varies from 59.31% at zero corruption to 59.25% at corruption probability 0.4, while mean queried labels increase from 130.8 to 221.2. Three-label maximum-KL selection varies from 58.73% to 58.43% at its constant 393 labels. Fixed BN+flip remains at 59.68% with zero labels throughout. The experiment shows how the implemented posterior rule responds to corrupted clas evidence; Table 51 retains all tested rates and simple selection references.

Table 49: PACS under changes in source checkpoint, batch size and arrival order. Every entry averages three regenerated stream orders.  
Accuracy (%).
<table><tr><td></td><td colspan="3">Source checkpoint</td><td colspan="4">Photo source: batch size</td><td colspan="4">Photo source: arrival order</td></tr><tr><td>Rule</td><td>Art</td><td>Cart.</td><td>Sketch</td><td>16</td><td>32</td><td>64</td><td>128</td><td>Rev.</td><td>Mixed</td><td>Grad.</td><td>Class</td></tr><tr><td>Source</td><td>68.42</td><td>76.91</td><td>38.32</td><td>50.85</td><td>50.85</td><td>50.88</td><td>50.88</td><td>50.88</td><td>50.88</td><td>50.88</td><td>50.88</td></tr><tr><td>BN</td><td>77.33</td><td>80.35</td><td>62.45</td><td>57.18</td><td>58.07</td><td>58.17</td><td>58.80</td><td>58.17</td><td>59.67</td><td>59.08</td><td>22.65</td></tr><tr><td>Flip</td><td>68.94</td><td>77.51</td><td>38.02</td><td>50.90</td><td>50.90</td><td>50.90</td><td>50.90</td><td>50.90</td><td>50.90</td><td>50.90</td><td>50.90</td></tr><tr><td>BN+flip</td><td>78.39</td><td>81.41</td><td>63.02</td><td>58.49</td><td>59.39</td><td>59.53</td><td>59.87</td><td>59.53</td><td>60.56</td><td>60.63</td><td>22.58</td></tr><tr><td>Mean logits</td><td>78.89</td><td>82.20</td><td>55.22</td><td>60.79</td><td>61.54</td><td>61.34</td><td>61.47</td><td>61.34</td><td>59.15</td><td>60.92</td><td>45.25</td></tr><tr><td>Uniform</td><td>76.25</td><td>80.81</td><td>61.34</td><td>58.19</td><td>58.80</td><td>59.17</td><td>59.16</td><td>58.38</td><td>58.92</td><td>59.43</td><td>50.39</td></tr><tr><td>Disagreement</td><td>76.47</td><td>80.82</td><td>61.72</td><td>58.22</td><td>58.85</td><td>58.21</td><td>59.34</td><td>59.09</td><td>58.41</td><td>59.13</td><td>52.92</td></tr><tr><td>Max. KL</td><td>77.45</td><td>80.97</td><td>62.71</td><td>58.12</td><td>59.25</td><td>58.91</td><td>59.29</td><td>58.91</td><td>58.34</td><td>59.40</td><td>54.11</td></tr><tr><td>PORTAL</td><td>77.91</td><td>81.02</td><td>63.00</td><td>58.87</td><td>59.15</td><td>59.01</td><td>59.16</td><td>58.96</td><td>60.16</td><td>60.02</td><td>51.28</td></tr><tr><td>Oracle</td><td>79.94</td><td>83.66</td><td>64.00</td><td>66.05</td><td>64.28</td><td>62.81</td><td>61.93</td><td>62.81</td><td>62.58</td><td>62.87</td><td>56.52</td></tr></table>

Order standard deviation (percentage points).
<table><tr><td></td><td colspan="3">Source checkpoint</td><td colspan="4">Photo source: batch size</td><td colspan="4">Photo source: arrival order</td></tr><tr><td>Rule</td><td>Art</td><td>Cart.</td><td>Sketch</td><td>16</td><td>32</td><td>64</td><td>128</td><td>Rev.</td><td>Mixed</td><td>Grad.</td><td>Class</td></tr><tr><td>Source</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>BN</td><td>0.08</td><td>0.17</td><td>0.15</td><td>0.10</td><td>0.28</td><td>0.09</td><td>0.18</td><td>0.09</td><td>0.16</td><td>0.16</td><td>0.35</td></tr><tr><td>Flip</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>BN+flip</td><td>0.25</td><td>0.04</td><td>0.13</td><td>0.21</td><td>0.09</td><td>0.18</td><td>0.22</td><td>0.18</td><td>0.41</td><td>0.10</td><td>0.17</td></tr><tr><td>Mean logits</td><td>0.40</td><td>0.16</td><td>0.43</td><td>0.07</td><td>0.07</td><td>0.23</td><td>0.09</td><td>0.23</td><td>0.22</td><td>0.18</td><td>0.05</td></tr><tr><td>Uniform</td><td>0.46</td><td>0.34</td><td>0.23</td><td>0.28</td><td>0.23</td><td>0.29</td><td>0.15</td><td>0.47</td><td>0.26</td><td>0.24</td><td>0.86</td></tr><tr><td>Disagreement</td><td>0.70</td><td>0.50</td><td>0.81</td><td>0.61</td><td>0.18</td><td>1.02</td><td>0.20</td><td>0.29</td><td>0.41</td><td>0.54</td><td>0.13</td></tr><tr><td>Max. KL</td><td>0.12</td><td>0.12</td><td>0.21</td><td>0.35</td><td>0.38</td><td>0.26</td><td>0.09</td><td>0.26</td><td>0.61</td><td>0.58</td><td>0.31</td></tr><tr><td>PORTAL</td><td>0.07</td><td>0.54</td><td>0.16</td><td>0.40</td><td>0.50</td><td>0.38</td><td>0.60</td><td>0.30</td><td>0.53</td><td>0.72</td><td>0.51</td></tr><tr><td>Oracle</td><td>0.18</td><td>0.20</td><td>0.06</td><td>0.28</td><td>0.25</td><td>0.33</td><td>0.08</td><td>0.33</td><td>0.15</td><td>0.20</td><td>0.07</td></tr></table>

Acquired labels per stream.
<table><tr><td></td><td colspan="3">Source checkpoint</td><td colspan="4">Photo source: batch size</td><td colspan="4">Photo source: arrival order</td></tr><tr><td>Rule</td><td>Art</td><td>Cart.</td><td>Sketch</td><td>16</td><td>32</td><td>64</td><td>128</td><td>Rev.</td><td>Mixed</td><td>Grad.</td><td>Class</td></tr><tr><td>Source</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>BN</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Flip</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>BN+flip</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Mean logits</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Uniform</td><td>377.3</td><td>362.7</td><td>287.7</td><td>1,561.7</td><td>783</td><td>393</td><td>198</td><td>393</td><td>390.3</td><td>390</td><td>393</td></tr><tr><td>Disagreement</td><td>377.3</td><td>362.7</td><td>287.7</td><td>1,561.7</td><td>783</td><td>393</td><td>198</td><td>393</td><td>390.3</td><td>390</td><td>393</td></tr><tr><td>Max. KL</td><td>377.3</td><td>362.7</td><td>287.7</td><td>1,561.7</td><td>783</td><td>393</td><td>198</td><td>393</td><td>390.3</td><td>390</td><td>393</td></tr><tr><td>PORTAL</td><td>69</td><td>64</td><td>54.3</td><td>114.3</td><td>125.3</td><td>164</td><td>143</td><td>189.3</td><td>114.7</td><td>110</td><td>187.7</td></tr><tr><td>Oracle</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/cb3df4399af4f4155122b9d8211db11ba546367b131d92d77cb461189166f2c4.jpg)  
Figure 12: Outcome decomposition under dynamic arrival conditions. Color identifies the selected rule and texture identifies whether a sample is rescued, newly harmed, correct under both outputs, or incorrect under both. The components are a linear decomposition of the reported source accuracy, selected accuracy and sample-level new errors on the common evaluation population.

Table 50: Net batch harm and sample-level new errors under the same PACS conditions as Table 49. Net batch harm (percentage points).
<table><tr><td></td><td colspan="3">Source checkpoint</td><td colspan="4">Photo source: batch size</td><td colspan="4">Photo source: arrival order</td></tr><tr><td>Rule</td><td>Art</td><td>Cart.</td><td>Sketch</td><td>16</td><td>32</td><td>64</td><td>128</td><td>Rev.</td><td>Mixed</td><td>Grad.</td><td>Class</td></tr><tr><td>Source</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>BN</td><td>0.59</td><td>1.73</td><td>0.01</td><td>5.97</td><td>3.83</td><td>2.58</td><td>1.48</td><td>2.58</td><td>0.87</td><td>1.53</td><td>32.41</td></tr><tr><td>Flip</td><td>1.00</td><td>1.13</td><td>1.46</td><td>1.75</td><td>1.52</td><td>1.21</td><td>0.91</td><td>1.21</td><td>1.28</td><td>1.29</td><td>1.12</td></tr><tr><td>BN+flip</td><td>0.55</td><td>1.21</td><td>0.00</td><td>5.21</td><td>3.28</td><td>1.93</td><td>1.12</td><td>1.93</td><td>0.70</td><td>1.08</td><td>32.41</td></tr><tr><td>Mean logits</td><td>0.12</td><td>0.37</td><td>0.01</td><td>1.14</td><td>0.58</td><td>0.19</td><td>0.04</td><td>0.19</td><td>0.14</td><td>0.13</td><td>8.10</td></tr><tr><td>Uniform</td><td>0.51</td><td>1.11</td><td>0.01</td><td>3.73</td><td>2.35</td><td>1.38</td><td>0.71</td><td>1.40</td><td>0.63</td><td>0.88</td><td>4.62</td></tr><tr><td>Disagreement</td><td>0.42</td><td>0.99</td><td>0.00</td><td>3.62</td><td>2.35</td><td>1.45</td><td>0.64</td><td>1.29</td><td>0.61</td><td>0.84</td><td>1.94</td></tr><tr><td>Max. KL</td><td>0.46</td><td>0.84</td><td>0.00</td><td>3.67</td><td>2.10</td><td>1.37</td><td>0.86</td><td>1.37</td><td>0.52</td><td>0.91</td><td>0.86</td></tr><tr><td>PORTAL</td><td>0.52</td><td>0.80</td><td>0.00</td><td>3.56</td><td>2.89</td><td>1.72</td><td>1.16</td><td>1.57</td><td>0.60</td><td>1.15</td><td>0.34</td></tr><tr><td>Oracle</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td></td><td colspan="9">New errors (%).</td><td></td><td></td></tr><tr><td></td><td colspan="3">Source checkpoint</td><td colspan="4">Photo source: batch size</td><td colspan="4">Photo source: arrival order</td></tr><tr><td>Rule</td><td>Art</td><td>Cart.</td><td>Sketch</td><td>16</td><td>32</td><td>64</td><td>128</td><td>Rev.</td><td>Mixed</td><td>Grad.</td><td>Class</td></tr><tr><td>Source</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>BN</td><td>7.13</td><td>7.16</td><td>3.24</td><td>12.96</td><td>12.27</td><td>12.07</td><td>11.62</td><td>12.07</td><td>9.48</td><td>10.42</td><td>34.79</td></tr><tr><td>Flip</td><td>1.78</td><td>2.04</td><td>2.25</td><td>2.07</td><td>2.07</td><td>2.07</td><td>2.07</td><td>2.07</td><td>2.07</td><td>2.07</td><td>2.07</td></tr><tr><td>BN+flip</td><td>6.74</td><td>6.36</td><td>3.25</td><td>12.24</td><td>11.58</td><td>11.30</td><td>11.05</td><td>11.30</td><td>8.85</td><td>9.68</td><td>34.88</td></tr><tr><td>Mean logits</td><td>1.40</td><td>1.90</td><td>1.31</td><td>2.51</td><td>2.34</td><td>2.49</td><td>2.47</td><td>2.49</td><td>2.57</td><td>2.49</td><td>10.24</td></tr><tr><td>Uniform</td><td>5.53</td><td>5.53</td><td>3.13</td><td>8.95</td><td>8.31</td><td>8.42</td><td>7.61</td><td>8.22</td><td>7.47</td><td>8.05</td><td>6.38</td></tr><tr><td>Disagreement</td><td>5.39</td><td>5.09</td><td>3.10</td><td>8.49</td><td>8.50</td><td>7.90</td><td>7.93</td><td>7.91</td><td>6.99</td><td>7.82</td><td>3.35</td></tr><tr><td>Max. KL</td><td>6.28</td><td>4.57</td><td>3.20</td><td>8.58</td><td>8.18</td><td>8.20</td><td>8.33</td><td>8.20</td><td>6.58</td><td>8.15</td><td>1.96</td></tr><tr><td>PORTAL</td><td>6.58</td><td>5.17</td><td>3.22</td><td>9.10</td><td>10.35</td><td>10.01</td><td>10.49</td><td>9.63</td><td>8.34</td><td>9.34</td><td>0.66</td></tr><tr><td>Oracle</td><td>4.56</td><td>2.78</td><td>2.86</td><td>2.73</td><td>4.20</td><td>5.17</td><td>6.27</td><td>5.17</td><td>5.86</td><td>5.70</td><td>0.71</td></tr></table>

Table 51: Class-label corruption on the complete frozen PACS portfolio, five stream orders. A queried class is replaced uniformly by a different class with probability η. Accuracy is mean ± order SD; labels and newly introduced errors are means. Predictions and evaluation labels are unchanged.
<table><tr><td>η</td><td>Rule</td><td>Accuracy (%)</td><td>Labels</td><td>New errors (pp)</td></tr><tr><td>0</td><td>Uniform</td><td> $5 8 . 9 7 \pm 0 . 7 2$ </td><td>393.0</td><td>8.53</td></tr><tr><td>0</td><td>Disagreement</td><td> $5 8 . 9 3 \pm 0 . 4 7$ </td><td>393.0</td><td>8.66</td></tr><tr><td>0</td><td>Maximum KL</td><td> $5 8 . 7 3 \pm 0 . 2 3$ </td><td>393.0</td><td>8.55</td></tr><tr><td>0</td><td>PORTAL</td><td> $5 9 . 3 1 \pm 0 . 3 6$ </td><td>130.8</td><td>10.53</td></tr><tr><td>0.1</td><td>Uniform</td><td> $5 9 . 0 0 \pm 0 . 6 5$ </td><td>393.0</td><td>8.52</td></tr><tr><td>0.1</td><td>Disagreement</td><td> $5 8 . 8 1 \pm 0 . 4 7$ </td><td>393.0</td><td>8.45</td></tr><tr><td>0.1</td><td>Maximum KL</td><td> $5 8 . 6 8 \pm 0 . 3 2$ </td><td>393.0</td><td>8.53</td></tr><tr><td>0.1</td><td>PORTAL</td><td> $5 9 . 2 9 \pm 0 . 3 2$ </td><td>159.2</td><td>10.33</td></tr><tr><td>0.2</td><td>Uniform</td><td> $5 8 . 9 3 \pm 0 . 4 9$ </td><td>393.0</td><td>8.48</td></tr><tr><td>0.2</td><td>Disagreement</td><td> $5 8 . 9 6 \pm 0 . 4 7$ </td><td>393.0</td><td>8.49</td></tr><tr><td>0.2</td><td>Maximum KL</td><td> $5 8 . 8 6 \pm 0 . 3 6$ </td><td>393.0</td><td>8.34</td></tr><tr><td>0.2</td><td>PORTAL</td><td> $5 9 . 3 9 \pm 0 . 3 2$ </td><td>140.2</td><td>10.48</td></tr><tr><td>0.4</td><td>Uniform</td><td> $5 8 . 6 5 \pm 0 . 4 6$ </td><td>393.0</td><td>8.54</td></tr><tr><td>0.4</td><td>Disagreement</td><td> $5 8 . 4 4 \pm 0 . 6 8$ </td><td>393.0</td><td>8.22</td></tr><tr><td>0.4</td><td>Maximum KL</td><td> $5 8 . 4 3 \pm 0 . 4 6$ </td><td>393.0</td><td>8.29</td></tr><tr><td>0.4</td><td>PORTAL</td><td> $5 9 . 2 5 \pm 0 . 3 0$ </td><td>221.2</td><td>9.97</td></tr></table>

## R HISTORICAL CONTROLLER COMPARISONS

## R.1 RESPONSE TO CANDIDATE-SET EXPANSION

This experiment asks how the benefit of adding candidates changes with the controller. The locked confirmation uses one RobustBench CIFAR-10 Standard checkpoint, CIFAR-10/CIFAR-10-C and three corruption families. Ten new seeds (10–19) produce 120 episodes and 192,000 scored examples. The HILTTA-style controller follows a harmonized selection rule based on Li et al. (2024). In this experiment, PORTAL evaluates candidates reset to the pretrained model through queried errors. This configuration isolates the controller–portfolio interaction. The main experiments additionally evaluate the shared-posterior formulation in Section 2.

Define the expansion effect as expanded-set error minus narrow-set error. It is +14.3203 points for HILTTA-style and −1.2422 points for PORTAL. Their difference is the 15.5625-point controller– portfolio interaction. This comparison follows each controller’s own trajectory. It measures how expansion behaves under the two policies, rather than isolating selection on one common prediction pool.

Table 52: Locked controller comparison across ten confirmation seeds. Errors include every scored episode input. Positive expansion effects mean that adding candidates increases error. A dash indicates that no expansion contrast is reported for that comparator.
<table><tr><td>Controller</td><td>Expanded-set error (%)</td><td>Expansion effect (pp)</td></tr><tr><td>HILTTA-style</td><td>36.1698</td><td>+14.3203</td></tr><tr><td>PORTAL</td><td>18.9375</td><td>-1.2422</td></tr><tr><td>Harmonized SimATTA-style</td><td>15.2719</td><td></td></tr><tr><td>Committee</td><td>18.3438</td><td></td></tr></table>

PORTAL reduces expanded-set error by 17.2323 points relative to HILTTA-style and wins in all ten seeds. The exact two-sided sign-test probability is 0.001953. Table 52 also reports the harmonized SimATTA-style and Committee comparators. SimATTA-style has the lowest error in this experiment. The interaction answers the question about expansion; these complete scores show the accompanying controller ranking.

All expanded-set controllers use eight labels per episode. The narrow-portfolio PORTAL cell uses at most eight: 748 of 960 available labels, with all eight used in 70 of 120 episodes. Primary error includes queried inputs. Their labels do not enter current candidate generation or gradients. These details distinguish this experiment from the permanent query/deployment split used for the main shared-feedback comparisons.

## R.2 FIXED-CHALLENGER ADMISSION ON PACS AND VLCS

The PACS replay fixes a SimATTA incumbent trajectory, a horizontal-view challenger called U02, and 816 batches. U02 averages original and horizontally flipped predictions. Every controller in Table 53 uses the same recorded outputs. Positive gains mean higher accuracy than the incumbent. Paired admission selects U02 only when it makes strictly fewer queried errors; ties keep the incumbent. The PORTAL row queries the three largest symmetric-KL disagreements. Its admission decisions coincide with queried ERM in all 816 batches.

Always deployment uses U02 throughout. The random gate admits with probability 29.41%, matching the targeted controller’s rate. Uniform admission samples three query positions and applies the same error comparison. The oracle selects from full-batch evaluation labels. It is an evaluation reference. Keeping these predictions fixed isolates the effect of acquiring and using evidence on the recorded trajectory.

Table 53: PACS gains in percentage points over the matched incumbent. All inputs include current queried examples. Query-excluded removes current query positions, not historical labels or sourcedomain reuse. Brackets give 95% random-policy replay intervals where available.
<table><tr><td>Controller</td><td>All inputs</td><td>Current-query-excluded</td></tr><tr><td>Always challenger</td><td>-5.7965</td><td>-5.4337</td></tr><tr><td>Random gate</td><td>-1.6997 [-2.2946, -1.1256]</td><td>-1.5937 [-2.1556, -1.0389]</td></tr><tr><td>Uniform three-query admission</td><td>+1.1508 [0.8320, 1.4689]</td><td>+0.9441 [0.6104, 1.2663]</td></tr><tr><td>PORTAL paired admission</td><td>+2.2583</td><td>+1.7863</td></tr><tr><td>Full-batch oracle</td><td>+3.3280</td><td>+3.1247</td></tr></table>

The 79,928 PACS observations are eight readouts of 9,991 images, across 32 phases. They include the Photo source domain and images used in earlier adaptation. Current-query exclusion therefore does not produce a never-labeled population. A later replay first separates each batch into a common query pool and evaluation pool. Across five split/query seeds, symmetric-KL admission gains 1.8250±0.1222 points, uniform admission $0 . 9 4 2 8 { \pm } 0 . 0 3 8 2$ , always deployment $- 5 . 7 8 6 1 { \scriptstyle \pm 0 . 0 9 7 4 }$ and the evaluation-pool oracle $3 . 6 7 8 9 \pm 0 . 1 4 7 0$ . These repeats vary the split and queries on one recorded trajectory.

The VLCS transfer uses 872 batches and 85,832 observations over 32 phases. Always using U02 loses 11.2580 points; three-query symmetric-KL admission loses 0.3530. The avoided fraction of net accuracy loss is $1 - 0 . 3 5 3 0 / \dot { 1 } 1 . \dot { 2 } 5 8 0 = 9 6 .$ .86%. A ten-batch moving-block bootstrap gives an admission-gain interval of $[ - 0 . \dot { 5 } 6 3 2 , - 0 . 1 7 4 6 ]$ points. This measures the suppression of a harmful intervention; the remaining loss is reported in full.

## S JOINT SELECTION ALGORITHM AND IMPLEMENTATION

The selector receives candidate logits, query and deployment indices, image identifiers, and the remaining label allowance. It can request query labels; the evaluator alone holds deployment labels. This section specifies the finite-batch procedure. Appendix B gives the separate fixed-confidence procedure and its sampling rules.

## S.1 CANDIDATE REPRESENTATION AND PRIOR

Candidates with identical hard outputs on every current input are merged. If a group contains the designated fixed default, it is retained as the representative; otherwise the first action is retained. A joint prediction pattern $p _ { s } = ( p _ { s 1 } , \dots , p _ { s K } )$ groups current inputs into strata. Patterns occurring only in the query pool have target weight zero. For C classes, let $\mathcal { C } _ { s }$ contain the distinct predicted classes and let ν be the prior mass assigned to them. The initial parameters are

$$
\alpha _ { s c } ^ { 0 } = \frac { 1 - \nu } { C } + \frac { \nu } { | \mathcal { C } _ { s } | } \mathbf { 1 } \{ c \in \mathcal { C } _ { s } \} , \qquad \nu = 0 . 7 5 .\tag{86}
$$

The total prior mass is one. Each acquired label adds one class count. We store historical counts by candidate identity and prediction pattern, and count each acquired image once per deployment. When the same pattern reappears, its accumulated counts initialize the posterior. This reuse assumes that the pattern’s class probabilities stay stable. The no-history comparison starts from Equation (86) at every batch.

Target weights are empirical frequencies of unlabeled deployment inputs. The selector recomputes posterior means and covariances after every acquired class label using Equations (1)–(2). The leader is the default if its posterior mean is within $1 0 ^ { - 1 2 }$ of the maximum; otherwise the first maximizer is used. Class labels supply complete candidate correctness vectors, so the annotation count grows by one regardless of the number of candidates.

## S.2 REACHABILITY AND QUERY SCORES

For each rival, Equation (3) bounds its best possible posterior advantage after the remaining queries. Write $d _ { s a } = \pi _ { s , p _ { s a } } - \pi _ { s , p _ { s j } }$ . The improvement from assigning the kth favorable label to a disagreement stratum is

$$
w _ { s } ( 1 - d _ { s a } ) \frac { n _ { s } } { ( n _ { s } + k ) ( n _ { s } + k - 1 ) } .\tag{87}
$$

This quantity is nonnegative and decreases with k. Collecting the largest available increments, up to the remaining label count, solves the capped integer allocation. The rival remains competitive if its optimistic advantage exceeds $1 0 ^ { - 1 2 }$ . The leader is always retained. The full posterior perfectinformation bound still includes every distinct action.

For each eligible target stratum, Equation (4) computes one-label value on the competitive set. If all values are at most $1 0 ^ { - 1 4 }$ , the selector uses paired posterior variance reduction. For rival a and leader j that predict different classes in stratum s, the expected reduction is

$$
B _ { s , a j } = \frac { w _ { s } ^ { 2 } } { ( n _ { s } + 1 ) ^ { 2 } } \left[ \pi _ { s , p _ { s a } } + \pi _ { s , p _ { s j } } - ( \pi _ { s , p _ { s a } } - \pi _ { s , p _ { s j } } ) ^ { 2 } \right] .\tag{88}
$$

The fallback stratum score is

$$
T _ { s } = \operatorname* { m a x } _ { \substack { a \neq j \mathrm { c o m p e t i t i v e } } } \frac { B _ { s , a j } } { ( \mu _ { j } - \mu _ { a } ) ^ { 2 } + \Sigma _ { a a } + \Sigma _ { j j } - 2 \Sigma _ { a j } + 1 0 ^ { - 1 2 } } .\tag{89}
$$

The numerator is the expected squared change in the pair’s estimated accuracy difference after one label. The denominator gives priority to uncertain comparisons with small estimated gaps. This score can gather evidence when several labels are needed before the winner changes. Appendix B analyzes a separate frequentist allocation procedure.

The highest-scoring eligible stratum is selected, with deterministic pattern-order ties. Its first un queried input in stream order is labeled. No score uses evaluation labels. If both one-label value and the fallback vanish, or no target stratum contains an available query, the current batch decision stops. The mandatory-spending ablation instead fills its remaining budget from available query strata, including strata absent from the target batch when necessary.

## S.3 EXECUTION SEQUENCE

Table 54: Finite-batch joint selection. Predictions, current target composition and available query locations are inputs; acquired class labels are the feedback.

<table><tr><td>Step</td><td>Operation</td></tr><tr><td>1</td><td>Freeze all candidate logits and merge exact hard-output copies. Form joint prediction strata and target frequencies.</td></tr><tr><td>2</td><td>Initialize each stratum from Equation (86), adding eligible historical acquired-label counts.</td></tr><tr><td>3 4</td><td>Compute the posterior means/covariance, default-tied leader and all-candidate bound U. Stop if the local query allowance is exhausted or  $U \leq 0 . 0 0 2 .$  Stop if no available target-stratum</td></tr><tr><td>5</td><td>query remains. Compute capped rival reachability. Stop if every rival is unreachable within the remaining local</td></tr><tr><td>6</td><td>allowance. Compute one-label decision values. On a zero-value plateau, use Equation (89). Stop if no available</td></tr><tr><td>7</td><td>query has positive score. Acquire one class label in the chosen stratum. Update its posterior and historical count once, record</td></tr><tr><td>8</td><td>its identifier and decrement the annotation allowance. Return to step 3.</td></tr><tr><td></td><td>Deploy the posterior leader&#x27;s frozen predictions on the current target inputs. Apply the experiment&#x27;s prescribed subsequent model update.</td></tr></table>

Budget handling. The fixed-prediction PACS and VLM experiments accrue three labels per batch, carry unspent credits, and permit at most twelve acquisitions within a batch. The complete CIFAR frozen-selection experiment caps the stored balance at twelve as well as the within-batch count. Fixed-three baselines acquire three eligible labels, including on agreement inputs when needed. Their budget is reduced only when the eligible pool has fewer than three examples. The learnedreadout experiments use three total labels in each batch: the selector can spend part of this budget on output evidence and uniform sampling supplies the remainder for the subsequent learned state. That total includes all labels used for either purpose.

Computational structure. Posterior means require O(SK) operations and the full covariance $O ( S { \bar { K } } ^ { 2 } )$ ). The finite-horizon reachability calculation processes at most SKB marginal increments for local allowance B, with sorting in the current implementation. One-label value evaluates C possible outcomes per eligible stratum and competitive action. The matrices have candidate dimension, independent of the feature encoder’s parameter count. Candidate generation remains a separate cost: four ResNet actions require original/flip forwards under source and batch statistics; eight-view CLIP requires all specified image encodings. Resource tables report these operations separately from selection arithmetic.

## S.4 CONVEX-READOUT IMPLEMENTATION

Section P defines the supervised simplex objective, its Hessian and target-Fisher query score. The implementation centers every candidate’s class logits, initializes the weights uniformly, and solves the strictly convex objective with analytic-gradient SLSQP at tolerance $1 0 ^ { - 9 }$ . Weights reset at every batch and are refitted after each of at most three acquired labels. The primary regularizer is $\lambda = 1 ;$ the complete grid also retains 0.1 and 10.

The comparison uses the same queried examples for each fitted mixture and its discrete ERM control. Uniform, disagreement and maximum-pair-KL acquisition use their declared score, while target Fisher recomputes Equation (85) after every label. Evaluation inputs contribute unlabeled logits to the target curvature and supply labels only to the scorer. Tables 46–48 contain every regularizer, acquisition, output rule and probability metric.

## T ACQUISITION AND STOPPING CONTROLS

## T.1 CHANGING ONLY THE CURRENT QUERY LOCATIONS

The query comparison uses four fixed PACS candidates and five arrival orders. Each policy starts every batch from the same history, acquired labels, prior, deployment weights, and stopping rules. It can buy at most three labels, for a stream ceiling of 393. Only the choice of query inputs differs.

Uniform acquisition samples an eligible input. Disagreement acquisition prioritizes inputs where the active candidates predict different classes. The KL rule maximizes pairwise symmetric divergence. PORTAL uses one-label decision value, with paired variance reduction when that value is zero. Each rule uses a fixed random seed for ties. After the comparison, PORTAL supplies the next batch’s shared history. This gives all query rules the same starting evidence.

Table 55: Acquisition with a common posterior, history, available budget and stopping rule. Only the current query locations change. Means ± SD cover five PACS stream orders.
<table><tr><td>Acquisition</td><td>Accuracy (%)</td><td>Labels</td><td>Batch harm (pp)</td></tr><tr><td>Uniform</td><td> $5 9 . 3 5 \pm 0 . 1 3$ </td><td>132.8</td><td>1.78</td></tr><tr><td>Disagreement</td><td> $5 9 . 3 1 \pm 0 . 2 3$ </td><td>96.8</td><td>1.86</td></tr><tr><td>Maximum-pair KL</td><td> $5 9 . 3 3 \pm 0 . 2 6$ </td><td>118.8</td><td>1.85</td></tr><tr><td>PORTAL value</td><td> $5 9 . 3 2 \pm 0 . 3 4$ </td><td>85.6</td><td>1.86</td></tr></table>

Table 55 shows that the four accuracy means lie within 0.09 percentage points. PORTAL uses the fewest labels, 85.6 per stream. Disagreement is the closest alternative at 96.8 labels and 59.31% accuracy. Uniform and maximum-pair KL spend 132.8 and 118.8 labels. The result attributes most of the saving in this controlled replay to the shared posterior and stopping logic; decision-value acquisition further reduces cost relative to the other query locations.

## T.2 PURCHASING LABELS AFTER AN OBSERVED STOP

For every PORTAL stop, we copy the posterior and remaining eligible queries. The counterfactual continuation ranks those inputs by maximum-pair symmetric KL without labels, then reveals prefixes of one, three and five query labels. These labels affect only the copied current decision and never enter future stream history. We record the stop reason, action change and held-out accuracy change.

The model-side check computes the exact one-step posterior value of the first continuation query and verifies that it does not exceed the all-candidate bound U. For a reachability stop, the original candidate remains a posterior maximizer after every tested prefix within the original local budget. A tie can change the deterministic output without contradicting this property.

Table 56: Counterfactual labels after stopping. Extra labels follow a fixed maximum-pair-KL order and never enter future history. Accuracy change is positive when continuation reduces held-out errors.
<table><tr><td>Setting</td><td>Stop reason</td><td>Stops</td><td>Extra</td><td>Choice change</td><td>∆ accuracy (pp)</td></tr><tr><td>Primary</td><td>Unreachable</td><td>653</td><td>1.0</td><td>0.00%</td><td> $+ 0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>Primary</td><td>Unreachable</td><td>653</td><td>3.0</td><td>0.00%</td><td> $+ 0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>Primary</td><td>Unreachable</td><td>653</td><td>5.0</td><td>0.31%</td><td> $+ 0 . 0 0 0 \pm 0 . 0 5 1$ </td></tr><tr><td>Price 0.05</td><td>EVPI bound</td><td>498</td><td>1.0</td><td>1.20%</td><td> $- 0 . 0 7 5 \pm 0 . 0 5 2$ </td></tr><tr><td>Price 0.05</td><td>EVPI bound</td><td>498</td><td>3.0</td><td>2.21%</td><td> $- 0 . 0 1 9 \pm 0 . 1 0 0$ </td></tr><tr><td>Price 0.05</td><td>EVPI bound</td><td>498</td><td>5.0</td><td>2.81%</td><td> $- 0 . 0 1 2 \pm 0 . 1 4 1$ </td></tr></table>

At the primary price, 653 of 655 batches stop because no rival can overtake the leader within the remaining budget. The other two exhaust the budget. Five extra queries change the tie-broken action in 0.31% of stopped states and leave mean accuracy unchanged. At price 0.05, 498 batches stop because $U \leq 0 . 0 5$ . Their mean U is 0.0377, and the exact value of the first extra query is 0.00011. One, three, and five extra queries change accuracy by $- 0 . 0 7 5 , - 0 . 0 1 9$ , and −0.012 points. These are observed averages; the theorem bounds expected gain under the label model.

## T.3 PRIOR AND HISTORICAL EVIDENCE

Table 57: Prior, stopping-price and history sensitivity on five PACS stream orders. All settings were fixed before this grid was evaluated.
<table><tr><td>Variant</td><td>Accuracy (%)</td><td>Labels</td><td>Batch harm</td><td>New errors</td></tr><tr><td>Primary</td><td> $5 9 . 3 1 \pm 0 . 3 6$ </td><td>130.8</td><td>1.92</td><td>10.53</td></tr><tr><td>Label price 0.01</td><td> $5 9 . 3 1 \pm 0 . 3 6$ </td><td>129.8</td><td>1.92</td><td>10.53</td></tr><tr><td>Label price 0.025</td><td> $5 9 . 3 2 \pm 0 . 3 9$ </td><td>123.8</td><td>1.91</td><td>10.46</td></tr><tr><td>Label price 0.05</td><td> $5 9 . 3 7 \pm 0 . 3 4$ </td><td>108.0</td><td>1.91</td><td>10.52</td></tr><tr><td>Prior mass 0.25</td><td> $5 9 . 3 1 \pm 0 . 3 4$ </td><td>137.0</td><td>1.89</td><td>10.42</td></tr><tr><td>Prior mass 4</td><td> $5 9 . 3 4 \pm 0 . 2 4$ </td><td>156.4</td><td>1.88</td><td>10.36</td></tr><tr><td>Vote mass 0</td><td> $5 9 . 3 5 \pm 0 . 3 5$ </td><td>136.0</td><td>1.90</td><td>10.51</td></tr><tr><td>Vote mass 0.5</td><td> $5 9 . 3 3 \pm 0 . 3 3$ </td><td>141.0</td><td>1.88</td><td>10.43</td></tr><tr><td>Vote mass 0.9</td><td> $5 9 . 3 2 \pm 0 . 3 5$ </td><td>135.4</td><td>1.91</td><td>10.48</td></tr><tr><td>No history</td><td> $5 8 . 9 8 \pm 0 . 2 6$ </td><td>390.6</td><td>1.15</td><td>6.98</td></tr><tr><td>History decay 0.9</td><td> $5 9 . 2 8 \pm 0 . 5 7$ </td><td>368.0</td><td>1.52</td><td>8.97</td></tr><tr><td>History decay 0.5</td><td> $5 8 . 9 0 \pm 0 . 3 9$ </td><td>390.6</td><td>1.15</td><td>7.17</td></tr><tr><td>One audit label</td><td> $5 9 . 4 4 \pm 0 . 2 1$ </td><td>250.0</td><td>1.91</td><td>10.54</td></tr><tr><td>Audit + surprise reset</td><td> $5 9 . 4 4 \pm 0 . 1 6$ </td><td>274.8</td><td>1.91</td><td>10.56</td></tr></table>

Prior mass and vote mass have small effects on mean accuracy in this grid, while label use ranges from 135.4 to 156.4. Historical evidence has a larger effect on annotation cost. Removing history or multiplying a matched pattern’s counts by 0.5 whenever it reappears spends 390.6 labels, compared with 130.8 for cumulative history. Revisit decay 0.9 spends 368.0. One audit label from the highestmass available target stratum uses 250.0 labels, and resetting that stratum when its observed class has posterior probability below 0.05 uses 274.8. These variants receive the same three-credit stream budget. They show that stable reusable evidence is a major source of efficiency in this PACS setting and make the cost of forgetting and auditing explicit.

## T.4 OBSERVED STRATUM SCALE AND HISTORICAL REUSE

Table 58: Observed prediction patterns. $C$ counts classes and $K$ counts candidates. Stratum counts are per batch; P95 is their 95th percentile. History mass is the fraction of deployment inputs whose pattern has acquired labels from earlier batches. RN50 denotes ResNet-50. Query rate is labels per batch.
<table><tr><td>Setting</td><td>C</td><td>K Strata mean</td><td>P95</td><td>History mass</td><td>Queries/batch</td></tr><tr><td>PACS RN50</td><td>7</td><td>4</td><td>27.17 37.00</td><td>34.30%</td><td>1.00</td></tr><tr><td>CIFAR-10-C</td><td>10</td><td>4</td><td>30.62 42.00</td><td>20.20%</td><td>0.26</td></tr><tr><td>CIFAR-100-C</td><td>100</td><td>4</td><td>60.43 64.00</td><td>3.51%</td><td>0.16</td></tr><tr><td>PACS CLIP</td><td>7</td><td>4</td><td>7.53 9.00</td><td>0.01%</td><td>0.01</td></tr><tr><td>OfficeHome CLIP</td><td>65</td><td>4</td><td>42.28 48.55</td><td>0.31%</td><td>0.03</td></tr><tr><td>VLCS CLIP</td><td>5</td><td>4</td><td>8.34 12.00</td><td>8.74%</td><td>0.75</td></tr><tr><td>VLCS CLIP + StatA</td><td>5</td><td>5</td><td>10.17 15.00</td><td>8.72%</td><td>0.81</td></tr></table>

The finite candidate count does not determine the number of observed joint prediction patterns. CIFAR-100-C has 60.43 strata per batch on average with $C = 1 0 0$ , close to its batch size, while PACS CLIP has 7.53 because its readouts usually agree. Adding StatA to VLCS raises the mean from 8.34 to 10.17 and the 95th percentile from 12 to 15. Historical evidence covers 34.30% of target mass in the PACS RN50 stream, 20.20% on CIFAR-10-C, and only 0.31% on OfficeHome CLIP. These measurements connect class count, candidate diversity and reusable evidence to the actual selector workload.

## U CAUSAL RESIDUAL-READOUT LEARNING ON CORRUPTION STREAMS

The learned intervention adds a residual classifier to the fixed BN+flip output. For an input image, write $z _ { B }$ for its BN+flip logits and x for its unit-normalized classifier-input feature. All training labels available at time t come from earlier batches. Matrix $X _ { < t }$ contains their normalized features, and $R _ { < t }$ contains targets $e _ { y } - \mathrm { s o f t m a x } ( z _ { B } )$ . With residual scale s and regularization strength $\lambda ,$ the candidate is

$$
z _ { L } ( x ) = z _ { B } ( x ) + s x W _ { t - 1 } , \qquad W _ { t - 1 } = ( X _ { < t } ^ { \top } X _ { < t } + \lambda I ) ^ { - 1 } X _ { < t } ^ { \top } R _ { < t } .\tag{90}
$$

The transferred primary parameters are $\lambda = 1$ and $s = 4$ . They are shared across both CIFAR datasets, all corruptions and all checkpoints. The CIFAR prediction banks contain the exact averaged features underlying the normalization-and-flip logits. The earlier PACS experiment uses its available original classifier-input features; both settings apply the same normalized-feature ridge objective.

Incremental computation. The ridge solution is updated without reconstructing its complete kernel matrix. Write ${ \bf \dot { P } } = ( X ^ { \top } X + \lambda I ) ^ { - 1 }$ and let $X _ { q } , R _ { q }$ contain the current queried examples. The Woodbury identity gives

$$
K _ { q } = P X _ { q } ^ { \top } ( I + X _ { q } P X _ { q } ^ { \top } ) ^ { - 1 } ,\tag{91}
$$

$$
P ^ { \prime } = P - K _ { q } X _ { q } P ,\tag{92}
$$

$$
W ^ { \prime } = W + K _ { q } ( R _ { q } - X _ { q } W ) .\tag{93}
$$

For the last identity, substitute $W = P X ^ { \top } R$ into $W ^ { \prime } = P ^ { \prime } ( X ^ { \top } R + X _ { q } ^ { \top } R _ { q } )$ and use $P ^ { \prime } X _ { q } ^ { \top } = K _ { q } .$ Thus the incremental update solves the same ridge objective as Equation (90). Its state consists of $\mathrm { ~ a ~ 5 1 2 ~ } \times \mathrm { ~ 5 1 2 ~ }$ inverse covariance and a $5 1 2 \times C$ readout. Symmetrizing $P ^ { \prime }$ removes floating-point asymmetry. The residual learner changes the readout; the backbone remains frozen.

Prediction and feedback order. Each current batch first receives frozen $z _ { B }$ and $z _ { L }$ predictions from the preceding state. Uniform querying, disagreement-first uniform querying and posterior querying each produce their own learning trajectory. The two empirical selectors compare errors on three current query labels, choosing $z _ { B }$ on ties. Posterior selection may stop before using three comparison labels; the remaining slots are filled uniformly for future ridge training after its current output is fixed. The current labels are appended to the ridge objective only after the frozen current predictions are selected and scored. The update is performed for every batch, regardless of the selected output.

Table 59: Causal residual-readout transfer on complete severity-five corruption suites. Accuracy is mean ± SD over three source checkpoints. Each learned trajectory uses 7,065 labels; decision labels are the subset used for current output comparison. All methods score the same 75,000 evaluation occurrences per checkpoint suite.
<table><tr><td>Query policy</td><td>Output</td><td>Accuracy (%)</td><td>∆BF</td><td>Total labels</td><td>Decision labels</td></tr><tr><td colspan="6">CIFAR-10-C</td></tr><tr><td></td><td>Fixed BN+flip</td><td> $7 9 . 6 5 \pm 0 . 5 6$ </td><td>0.00</td><td>0</td><td>0</td></tr><tr><td>Uniform</td><td>Learned</td><td> $7 9 . 7 4 \pm 0 . 5 4$ </td><td>+0.09</td><td>7065</td><td>7065.0</td></tr><tr><td>Uniform</td><td>Selected</td><td> $7 9 . 6 4 \pm 0 . 5 6$ </td><td>0.00</td><td>7065</td><td>7065.0</td></tr><tr><td>Disagreement</td><td>Learned</td><td> $7 9 . 7 4 \pm 0 . 5 2$ </td><td>+0.10</td><td>7065</td><td>7065.0</td></tr><tr><td>Disagreement</td><td>Selected</td><td> $7 9 . 6 5 \pm 0 . 5 8$ </td><td>0.00</td><td>7065</td><td>7065.0</td></tr><tr><td>PORTAL</td><td>Learned</td><td> $7 9 . 7 4 \pm 0 . 5 4$ </td><td>+0.09</td><td>7065</td><td>50.3</td></tr><tr><td>PORTAL</td><td>Selected</td><td> $7 9 . 6 6 \pm 0 . 5 6$ </td><td>+0.01</td><td>7065</td><td>50.3</td></tr><tr><td colspan="6">CIFAR-100-C</td></tr><tr><td></td><td>Fixed BN+flip</td><td> $5 4 . 3 2 \pm 0 . 0 5$ </td><td>0.00</td><td>0</td><td>0</td></tr><tr><td>Uniform</td><td>Learned</td><td> $5 4 . 3 1 \pm 0 . 0 9$ </td><td>-0.01</td><td>7065</td><td>7065.0</td></tr><tr><td>Uniform</td><td>Selected</td><td> $5 4 . 3 2 \pm 0 . 0 5$ </td><td>+0.00</td><td>7065</td><td>7065.0</td></tr><tr><td>Disagreement</td><td>Learned</td><td> $5 4 . 3 4 \pm 0 . 1 2$ </td><td>+0.02</td><td>7065</td><td>7065.0</td></tr><tr><td>Disagreement</td><td>Selected</td><td> $5 4 . 3 4 \pm 0 . 0 7$ </td><td>+0.01</td><td>7065</td><td>7065.0</td></tr><tr><td>PORTAL</td><td>Learned</td><td> $5 4 . 3 1 \pm 0 . 0 8$ </td><td>-0.01</td><td>7065</td><td>14.7</td></tr><tr><td>PORTAL</td><td>Selected</td><td> $5 4 . 3 2 \pm 0 . 0 5$ </td><td>0.00</td><td>7065</td><td>14.7</td></tr></table>

Coverage and accounting. Each corruption is an independent deployment episode with reset ridge and posterior states. Every episode includes all 10,000 inputs, its fixed 5,000 query and 5,000 evaluation roles, and all 157 original batches, including the final 16 inputs. Every batch purchases three distinct query labels. A complete fifteen-corruption suite therefore uses 7,065 labels per learned trajectory. Fixed output alone uses zero labels. Every learned trajectory counts all acquired labels, including batches that select the fixed output. The two datasets and three sourcetraining checkpoints yield 90 episodes for each query policy. Evaluation labels enter the scorer after acquisition and are excluded from the learner and selector.

Results report fixed output, always-learned output and selected output on each policy’s actual candidate trajectory. Accuracy, cross-entropy, sample-level newly introduced errors and batch-net harmful contributions are scored on the same evaluation examples. Across-checkpoint standard deviations describe the three training initializations. CPU readout timings exclude the previously completed backbone and candidate forwards.

## V ACTIVE MODEL SELECTION, SHARED LEARNING AND SELECTOR ABLATIONS

## V.1 CONTEXTUAL ACTIVE MODEL SELECTION WITH PERMANENT QUERY ROLES

Contextual Active Model Selection (CAMS) (Liu et al., 2024) chooses a model for each input. We apply its stochastic procedure to the four fixed PACS candidates: identity, BN, flip, and BN+flip. Each of four advice policies recommends one candidate. CAMS combines these policies through exponential weights and smoothed advice, then samples a model. It chooses that model before deciding whether to request the label. Label requests follow an entropy-dependent Bernoulli probability, and policy losses use inverse-probability weights. Only permanent query inputs can be labeled.

CAMS restarts at domain boundaries. The cumulative caps 131,393,655 are allocated to domains in proportion to their batch counts, corresponding to nominal budgets one,three,five. The algorithm can stop below its cap because its native requests remain stochastic. Five stream orders couple their own selector randomization to the fixed prediction banks. BN predictions are generated from the full unlabeled batch, so the per-example action timing applies to model selection while feature computation remains batch based.

With four single-candidate advice policies, CAMS smoothing assigns $7 / 1 6$ probability to the recommended model and $3 / 1 6$ to each other model. Model choice remains random even when one policy dominates. Our implementation retains this distribution. Twelve comparisons with the authors’ routine reproduce label requests, counts, weights, and sampled prediction losses, including zero and exhausted budgets. Changing withheld evaluation labels leaves model choices and updates unchanged.

Table 60: Full stochastic CAMS with four single-candidate advice policies, permanent query eligibility and per-example predictions. Mean ± SD over five PACS stream orders. Cap allocation is proportional to domain batch counts; realized requests follow CAMS.
<table><tr><td>Nominal labels/batch</td><td>Total cap</td><td>Actual labels</td><td>Accuracy (%)</td></tr><tr><td>1</td><td>131</td><td>112.8</td><td> $5 5 . 2 8 \pm 0 . 4 5$ </td></tr><tr><td>35</td><td>393</td><td>128.6</td><td> $5 5 . 2 8 \pm 0 . 4 4$ </td></tr><tr><td></td><td>655</td><td>128.6</td><td> $5 5 . 2 8 \pm 0 . 4 4$ </td></tr></table>

On the same evaluation images, fixed identity, BN, flip, and BN+flip reach 50.88%,58.37%,50.90%,59.68%. CAMS makes per-example choices before the corresponding label request (Table 60). PORTAL can use current query labels before choosing an output for the held-out batch. These experiments compare the methods at their respective decision times. The CAMS result uses the four single-candidate advice policies defined above.

## V.2 HUMAN-IN-THE-LOOP ADAPTATION WITH SHARED LABELS

Human-in-the-loop test-time adaptation (HILTTA) (Li et al., 2024) combines active querying with model updates. We use its released TENT-based implementation, which adapts normalization parameters by entropy minimization. The K-Margin rule scales features by uncertainty $1 - \left( p _ { ( 1 ) } - p _ { ( 2 ) } \right)$ and selects three query inputs by greedy Euclidean farthest-point selection, starting from the zero vector. Remaining inputs support unlabeled adaptation.

HILTTA takes one entropy step at each of seven learning rates in sequence:

$$
5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 2 . 5 \times 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 2 . 5 \times 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 3 } .\tag{94}
$$

Each step continues the preceding state and saves model and optimizer parameters. Adam updates normalization affine parameters with $( \beta _ { 1 } , \beta _ { 2 } ) \ : = \ : ( 0 . 9 , 0 . 9 9 9 )$ and zero weight decay. The initial state lies outside the seven scored choices.

Each saved state receives a queried cross-entropy score and an anchor score, the mean squared difference between adapted and fixed-source softmax outputs on unqueried inputs. Each sevenscore vector is shifted by its minimum and divided by its resulting maximum. A zero denominator produces an all-ones vector, following the implementation. The normalized loss vectors are added. The score history initializes on the first batch, then averages past and current scores with equal weight. The lowest score selects the saved state, using its first occurrence on ties.

The chosen state receives a supervised Adam update at rate $1 0 ^ { - 5 }$ . This step enables model gradients, computes acquired-label cross-entropy through a full-batch forward, and restores the TENT gradient configuration afterward. Model and score history continue across target domains; every new stream order restarts from the source checkpoint. The source anchor remains fixed.

We store the incoming and BN+flip predictions before acquiring current labels. A paired selector reuses HILTTA’s labels and chooses BN+flip when it has strictly higher queried accuracy. The choice leaves HILTTA’s learning updates and future queries unchanged. A further comparison uses predictions after the supervised update. Every output uses the same 393 acquired labels per order, each counted once.

The shared-label selector reaches 79.98%, the incoming output 79.81%, the fixed BN+flip output 80.46% and the post-update output 80.05%. These scores describe distinct available outputs of the same acquired-label learning trajectories. A frozen model uses running source statistics and receives a BN+flip intervention; the adaptive incoming model already uses batch statistics and continuously changes parameters. Their difference includes model state, candidate increment and acquisition, while the fixed-bank K-Margin control isolates acquisition on common predictions.

Table 61: Shared-label HILTTA on the same 4,155 permanent evaluation images per order. Every row uses 393 total labels. Incoming and BN+flip predictions are fixed before current labels; postupdate predictions are available later. Initial-only uses BN+flip on the first batch and incoming output thereafter.
<table><tr><td>Order</td><td>Incoming</td><td>BN+flip</td><td>Shared gate</td><td>Post-update</td><td>Initial-only</td></tr><tr><td>201</td><td>79.06</td><td>79.95</td><td>79.21</td><td>79.28</td><td>79.04</td></tr><tr><td>202</td><td>78.87</td><td>79.37</td><td>78.92</td><td>79.16</td><td>78.87</td></tr><tr><td>203</td><td>79.78</td><td>80.31</td><td>80.05</td><td>79.88</td><td>79.78</td></tr><tr><td>204</td><td>81.35</td><td>81.85</td><td>81.47</td><td>81.76</td><td>81.35</td></tr><tr><td>205</td><td>80.00</td><td>80.82</td><td>80.26</td><td>80.17</td><td>80.05</td></tr><tr><td> $\mathbf { M e a n } \pm \mathbf { S D }$ </td><td> $7 9 . 8 1 \pm 0 . 9 8$ </td><td> $8 0 . 4 6 \pm 0 . 9 4$ </td><td> $7 9 . 9 8 \pm 1 . 0 0$ </td><td> $8 0 . 0 5 \pm 1 . 0 4$ </td><td> $7 9 . 8 2 \pm 0 . 9 9$ </td></tr></table>

Table 62: Fixed-bank portfolio and selector ablations on PACS, five complete stream orders. Default-tied ERM uses BN+flip on empirical ties; first-tied ERM uses the first action. Accuracy is $\mathrm { m e a n } \pm \mathrm { S D } .$ , and labels are actual per-stream means. Mean logits is an additional output function.
<table><tr><td rowspan="2">Rule</td><td colspan="2">Two candidates</td><td colspan="2">Four candidates</td></tr><tr><td>Accuracy</td><td>Labels</td><td>Accuracy</td><td>Labels</td></tr><tr><td>Identity</td><td> $5 0 . 8 8 \pm 0 . 0 0$ </td><td>0.0</td><td> $5 0 . 8 8 \pm 0 . 0 0$ </td><td>0.0</td></tr><tr><td>BN</td><td></td><td></td><td> $5 8 . 3 7 \pm 0 . 1 4$ </td><td>0.0</td></tr><tr><td>Flip</td><td></td><td></td><td> $5 0 . 9 0 \pm 0 . 0 0$ </td><td>0.0</td></tr><tr><td>BN+flip</td><td> $5 9 . 6 8 \pm 0 . 2 8$ </td><td>0.0</td><td> $5 9 . 6 8 \pm 0 . 2 8$ </td><td>0.0</td></tr><tr><td>Mean logits</td><td> $6 0 . 9 6 \pm 0 . 1 2$ </td><td>0.0</td><td> $6 1 . 6 3 \pm 0 . 1 6$ </td><td>0.0</td></tr><tr><td>Uniform, default ties</td><td> $5 9 . 2 7 \pm 0 . 7 0$ </td><td>393.0</td><td> $5 8 . 9 7 \pm 0 . 7 2$ </td><td>393.0</td></tr><tr><td>Disagreement, default ties</td><td> $5 9 . 2 1 \pm 0 . 2 0$ </td><td>393.0</td><td> $5 8 . 9 3 \pm 0 . 4 7$ </td><td>393.0</td></tr><tr><td>Maximum KL, default ties</td><td> $5 9 . 3 1 \pm 0 . 3 1$ </td><td>393.0</td><td>58.73 ± 0.23</td><td>393.0</td></tr><tr><td>Uniform, first ties</td><td> $5 5 . 5 1 \pm 1 . 1 3$ </td><td>393.0</td><td> $5 5 . 4 5 \pm 0 . 6 6$ </td><td>393.0</td></tr><tr><td>Disagreement, first ties</td><td> $5 7 . 8 3 \pm 0 . 4 1$ </td><td>393.0</td><td>57.36 ± 0.82</td><td>393.0</td></tr><tr><td>Maximum KL, first ties</td><td> $5 8 . 1 3 \pm 0 . 3 5$ </td><td>393.0</td><td>57.36 ± 0.38</td><td>393.0</td></tr><tr><td>PORTAL</td><td> $5 9 . 3 1 \pm 0 . 2 5$ </td><td>112.2</td><td> $5 9 . 3 1 \pm 0 . 3 6$ </td><td>130.8</td></tr><tr><td>PORTAL, constant three labels</td><td> $5 9 . 2 9 \pm 0 . 4 4$ </td><td>393.0</td><td> $5 9 . 3 9 \pm 0 . 1 9$ </td><td>393.0</td></tr><tr><td>PORTAL, no history</td><td> $5 9 . 2 1 \pm 0 . 5 4$ </td><td>392.4</td><td> $5 8 . 9 8 \pm 0 . 2 6$ </td><td>390.6</td></tr><tr><td>PORTAL, no history/three labels</td><td> $5 9 . 0 0 \pm 0 . 5 7$ </td><td>393.0</td><td> $5 8 . 9 0 \pm 0 . 2 8 $ </td><td>393.0</td></tr><tr><td>PORTAL, no competitive pruning</td><td> $5 9 . 5 2 \pm 0 . 2 0$ </td><td>391.8</td><td> $5 9 . 2 1 \pm 0 . 2 3$ </td><td>392.4</td></tr><tr><td>Batch oracle</td><td> $6 1 . 6 8 \pm 0 . 4 0$ </td><td></td><td> $6 2 . 8 7 \pm 0 . 2 8$ </td><td></td></tr></table>

## V.3 FULL POSTERIOR AND EMPIRICAL-RULE ABLATIONS

Table 62 retains every distinct fixed prediction, the mean-logit output, both empirical tie conventions and each posterior ablation. All rows use identical evaluation images within a stream. The twoaction portfolio contains identity and BN+flip; the four-action portfolio adds BN and flip separately. Adding exact copies to make four and eight nominal candidates, respectively, leaves the joint output and query sequence unchanged. This is checked on every batch of all five streams.

Constant-three spending turns off stopping and fills the available query budget even on agreement inputs. Removing historical evidence reinitializes the prior on each batch. Removing competitive pruning retains all candidates when evaluating one-label value. These changes isolate the implemented sources of budget savings. The posterior perfect-information calculation always contains all distinct candidates, including in the competitive-pruning variant.

## W STOPPING, CHANGING EVIDENCE, AND EXPANDED CANDIDATE SETS

This section connects the additional experiments to the decision studied in the main text. The stopping comparison holds the original PACS candidates fixed. The expanded portfolio makes the meanlogit ensemble available to the selector. The finite-pool calculation measures how difficult a real candidate difference is to identify. The conditional-label experiment changes that difference without changing the prediction patterns.

## W.1 PROTOCOLS AND LABEL ACCOUNTING

Three stopping experiments use the same four PACS candidates but different budget and history rules. The original experiment carries all unused credits, allows twelve queries per batch, and uses 130.8 labels on average. The independent-history experiment caps stored credits at four times the per-batch allowance. With three new credits per batch, it allows at most twelve queries and uses 140.0 labels. The shared-history experiment gives each copied state three current queries and uses 85.6 labels. Its decisions all start from the reference policy’s history.

The five-candidate PACS, dynamic, CIFAR-10.1, and corruption-switch studies use three credits per batch with a stored balance capped at twelve. Mean logits is the default on ties. Each selector has its own history and label record. All labels, including audit labels, count toward cost. Fixed outputs use no labels. Every comparison retains the same permanent query and evaluation roles.

## W.2 STOPPING WITH INDEPENDENT AND MATCHED HISTORIES

The stopping study uses five PACS stream orders and the same four frozen candidate predictions as the original acquisition controls. Full PORTAL uses both the value bound and reachable-winner test. Reachability-only disables the value-bound trigger. Value-bound-only disables the reachability trigger. Every variant retains the competitive candidate set for query scoring. If no positive query score remains, the latter variants select by target-stratum mass while waiting for their own stopping condition or budget exhaustion.

For the confidence comparator, approximate each posterior difference by a normal distribution with mean gap $d _ { a }$ and variance $s _ { a } ^ { 2 }$ from Theorem 1. The comparator stops when the sum of the normal lower-tail probabilities over rivals is at most 0.05. This is a Gaussian confidence proxy. The meangap rule instead stops when the leading posterior mean exceeds the runner-up by at least 0.05. These are the fixed thresholds tested here.

For per-batch credit $b \in \{ 1 , 3 , 5 \}$ , stored credit is capped at 4b. The implementation sets the perbatch query cap to max(12, 4b); available credit therefore limits a batch to at most 4b queries. Independent-stream variants maintain their own histories. In the matched-current-batch experiment, every policy starts from a copy of the same history and record of acquired labels. Only the full-policy branch supplies the history for the next batch. Table 63 reports both experiments.

With three credits, full PORTAL and reachability-only match in all 655 batches. Of these, 654 stop because the posterior winner is unreachable by a rival; one exhausts the budget. The value-bound trigger does not fire at price 0.002. Full and reachability-only also coincide at the other two credits and in the matched-current-batch comparison. Relative to value-bound-only, full PORTAL saves 252.8 labels at three credits. The paired accuracy difference is −0.043 percentage points, with a 95% stream-order t interval of [−0.315, 0.228]. This interval describes order variation on one checkpoint and the same target images.

Table 63: Stopping controls on five PACS stream orders. Accuracy is mean ± order SD in percent. Labels count actual purchases. Harm is the positive batch-level accuracy loss relative to the unchanged model, averaged over all evaluation inputs, in percentage points.
<table><tr><td>History</td><td>Credit</td><td>Stop rule</td><td>Accuracy</td><td>Labels</td><td>Harm</td></tr><tr><td>Independent</td><td>1</td><td>PORTAL</td><td> $5 9 . 3 4 \pm 0 . 1 1$ </td><td>90.4</td><td>1.74</td></tr><tr><td>Independent</td><td>1</td><td>Reachability only</td><td> $5 9 . 3 4 \pm 0 . 1 1$ </td><td>90.4</td><td>1.74</td></tr><tr><td>Independent</td><td>1</td><td>Value bound only</td><td> $5 9 . 0 6 \pm 0 . 2 6$ </td><td>131.0</td><td>1.69</td></tr><tr><td>Independent</td><td>1</td><td>Confidence proxy</td><td> $5 9 . 0 6 \pm 0 . 2 6$ </td><td>131.0</td><td>1.69</td></tr><tr><td>Independent</td><td>1</td><td>Mean-gap threshold</td><td> $5 9 . 0 6 \pm 0 . 2 6$ </td><td>131.0</td><td>1.69</td></tr><tr><td>Independent</td><td>3</td><td>PORTAL</td><td> $5 9 . 3 5 \pm 0 . 3 4$ </td><td>140.0</td><td>1.88</td></tr><tr><td>Independent</td><td>3</td><td>Reachability only</td><td> $5 9 . 3 5 \pm 0 . 3 4$ </td><td>140.0</td><td>1.88</td></tr><tr><td>Independent</td><td>3</td><td>Value bound only</td><td> $5 9 . 3 9 \pm 0 . 1 9$ </td><td>392.8</td><td>1.82</td></tr><tr><td>Independent</td><td>3</td><td>Confidence proxy</td><td> $5 9 . 3 9 \pm 0 . 1 9$ </td><td>392.8</td><td>1.82</td></tr><tr><td>Independent</td><td>3</td><td>Mean-gap threshold</td><td> $5 9 . 2 9 \pm 0 . 2 6$ </td><td>392.2</td><td>1.88</td></tr><tr><td>Independent</td><td>5</td><td>PORTAL</td><td> $5 9 . 2 8 \pm 0 . 3 3$ </td><td>130.0</td><td>1.95</td></tr><tr><td>Independent</td><td>5</td><td>Reachability only</td><td> $5 9 . 2 8 \pm 0 . 3 3$ </td><td>130.0</td><td>1.95</td></tr><tr><td>Independent</td><td>5</td><td>Value bound only</td><td> $5 9 . 4 0 \pm 0 . 3 6$ </td><td>654.0</td><td>1.81</td></tr><tr><td>Independent</td><td>5</td><td>Confidence proxy</td><td> $5 9 . 4 0 \pm 0 . 3 6$ </td><td>654.0</td><td>1.81</td></tr><tr><td>Independent</td><td>5</td><td>Mean-gap threshold</td><td> $5 9 . 3 5 \pm 0 . 3 7$ </td><td>652.4</td><td>1.81</td></tr><tr><td>Matched</td><td>3</td><td>PORTAL</td><td> $5 9 . 3 2 \pm 0 . 3 4$ </td><td>85.6</td><td>1.86</td></tr><tr><td>Matched</td><td>3</td><td>Reachability only</td><td> $5 9 . 3 2 \pm 0 . 3 4$ </td><td>85.6</td><td>1.86</td></tr><tr><td>Matched</td><td>3</td><td>Value bound only</td><td> $5 9 . 3 2 \pm 0 . 3 4$ </td><td>392.8</td><td>1.86</td></tr><tr><td>Matched</td><td>3</td><td>Confidence proxy</td><td> $5 9 . 3 2 \pm 0 . 3 4$ </td><td>392.8</td><td>1.86</td></tr><tr><td>Matched</td><td>3</td><td>Mean-gap threshold</td><td> $5 9 . 2 8 \pm 0 . 3 1$ </td><td>391.0</td><td>1.86</td></tr></table>

## W.3 AN ENSEMBLE INSIDE THE CANDIDATE SET

Table 64: Fixed accuracy and selector accuracy (%) / labels on common evaluation inputs. K counts selectable candidates. Labels are actual totals per suite or stream.
<table><tr><td>Data (K)</td><td>Fixed</td><td>PORTAL Top-KL</td></tr><tr><td>CIFAR-10-C (4)</td><td>79.65</td><td>79.26/612.3 79.20/7065</td></tr><tr><td>CIFAR-100-C (4)</td><td>54.32</td><td>53.87/381.7 53.76/7065</td></tr><tr><td>PACS (5)</td><td>61.63</td><td>62.07/144.2 60.57/393</td></tr><tr><td>CIFAR-10.1 (5)</td><td>89.87</td><td>89.59/1.9 89.06/96</td></tr></table>

Fixed outputs: BN+flip on CIFAR-C, mean logits on PACS, and flip on CIFAR-10.1. They use no labels.

The five-candidate portfolio contains identity, BN, flip, BN+flip, and the mean of their logits. All empirical and posterior selectors can select the ensemble. The primary PACS replay uses the original Photo-source checkpoint and five target orders. Table 65 reports cumulative history, decay, and ordinary queried-error selection. In a decay variant, a pattern’s historical counts are multiplied by 0.9 each time that pattern reappears.

Table 65: Five-candidate PACS replay, five stream orders. The fixed ensemble is available to every selector and resolves ties. Harm is measured against that ensemble in percentage points. The audited detector is a separate exploratory variant.
<table><tr><td>Rule</td><td>Accuracy (%)</td><td>Labels</td><td>Harm</td></tr><tr><td>Fixed BN+flip</td><td> $5 9 . 6 8 \pm 0 . 2 8$ </td><td>0.0</td><td>5.05</td></tr><tr><td>Fixed mean logits</td><td> $6 1 . 6 3 \pm 0 . 1 6$ </td><td>0.0</td><td>0.00</td></tr><tr><td>Uniform empirical selection</td><td> $6 0 . 4 5 \pm 0 . 7 2$ </td><td>393.0</td><td>2.08</td></tr><tr><td>Top-KL empirical selection</td><td> $6 0 . 5 7 \pm 0 . 4 3$ </td><td>393.0</td><td>2.24</td></tr><tr><td>PORTAL, cumulative</td><td> $6 2 . 0 7 \pm 0 . 7 1$ </td><td>144.2</td><td>1.55</td></tr><tr><td>PORTAL, decay 0.9</td><td> $6 2 . 1 1 \pm 0 . 6 7$ </td><td>250.4</td><td>1.62</td></tr><tr><td>Audited change detector</td><td> $6 1 . 8 6 \pm 0 . 5 1$ </td><td>263.2</td><td>1.74</td></tr></table>

Cumulative PORTAL improves on fixed mean logits in four of five orders. The mean paired gain is 0.448 percentage points, with a 95% stream-order t interval of [−0.345, 1.240]. The audited detector was developed on the synthetic change experiment. It uses an eight-observation window, threshold 2, and one paid audit per batch. It reaches 61.86% with 263.2 labels on this replay, compared with 62.07% and 144.2 for cumulative history. Its additional monitoring cost does not improve this real-stream result.

Regenerated arrival conditions. The dynamic extension tests ten conditions with three regenerated orders each. Conditions change the source checkpoint, batch size, or arrival order. Every comparison holds the four original predictions fixed and adds their mean logits as a fifth candidate. Table 66 gives all conditions, including the original four-candidate Top-KL result and its five-candidate extension.

Table 66: All five-candidate PACS dynamic conditions, averaged over three orders. Condition lists source, arrival order, and batch size. Paired entries are accuracy (%)/labels. Best fixed is the hindsight best single fixed output for that condition. KL-4 is the original four-candidate Top-KL accuracy from Table 49; KL-5 uses the expanded portfolio.
<table><tr><td>Condition</td><td>Best fixed</td><td>Ensemble</td><td>Cumulative</td><td>Decay</td><td>KL-5</td><td>KL-4</td></tr><tr><td>Art, domain, 64</td><td>78.89</td><td>78.89</td><td>78.01/79.0</td><td>77.90/138.7</td><td>77.89/377.3</td><td>77.45</td></tr><tr><td>Cartoon, domain, 64</td><td>82.20</td><td>82.20</td><td>81.79/51.3</td><td>81.65/72.3</td><td>81.72/362.7</td><td>80.97</td></tr><tr><td>Photo, domain, 16</td><td>60.79</td><td>60.79</td><td>60.50/98.7</td><td>60.66/156.3</td><td>60.20/1561.7</td><td>58.12</td></tr><tr><td>Photo, domain, 32</td><td>61.54</td><td>61.54</td><td>61.30/111.0</td><td>61.39/202.3</td><td>60.52/783.0</td><td>59.25</td></tr><tr><td>Photo, domain, 128</td><td>61.47</td><td>61.47</td><td>61.94/156.7</td><td>61.76/182.3</td><td>60.71/198.0</td><td>59.29</td></tr><tr><td>Photo, class, 64</td><td>50.90</td><td>45.25</td><td>51.74/211.3</td><td>53.60/330.0</td><td>49.58/393.0</td><td>54.11</td></tr><tr><td>Photo, gradual, 64</td><td>60.92</td><td>60.92</td><td>60.99/138.0</td><td>60.67/236.0</td><td>59.98/390.0</td><td>59.40</td></tr><tr><td>Photo, mixed, 64</td><td>60.56</td><td>59.15</td><td>60.02/90.3</td><td>59.97/197.7</td><td>59.05/390.3</td><td>58.34</td></tr><tr><td>Photo, reverse, 64</td><td>61.34</td><td>61.34</td><td>61.84/143.7</td><td>61.68/238.7</td><td>60.19/393.0</td><td>58.91</td></tr><tr><td>Sketch, domain, 64</td><td>63.02</td><td>55.22</td><td>62.03/66.0</td><td>61.83/137.7</td><td>59.64/287.7</td><td>62.71</td></tr></table>

Under class-block arrival, flip is the best fixed output at 50.90%. Cumulative history reaches 51.74% with 211.3 labels, and decay reaches 53.60% with 330.0. Four-candidate Top-KL reaches 54.11% with 393 labels. Interpolating between fixed flip and Top-KL gives 53.597% expected accuracy at 330 labels. This mixture is fitted from the measured endpoints after evaluation.

## W.4 CIFAR-10.1 AND CORRUPTION-SWITCH STREAMS

Version 6 of the CIFAR-10.1 benchmark contains 2,000 images (Recht et al., 2018). A fixed hash of image identity assigns half to querying and half to evaluation. We regenerate predictions for three source checkpoints and three image orders, using batches of 64 and the original CIFAR-C preprocessing. Model weights stay fixed. The ensemble averages the four original logits, and every posterior selector uses it on ties. Table 67 averages orders within each checkpoint before reporting the mean and checkpoint SD.

Table 67: CIFAR-10.1 with three checkpoints and three orders per checkpoint. Each order has 1,000 evaluation images. Accuracy is mean ± checkpoint SD after averaging orders within a checkpoint. Audit labels are included in cost.
<table><tr><td>Rule</td><td>Accuracy (%) Labels</td></tr><tr><td>Fixed identity</td><td> $8 9 . 2 3 \pm 0 . 9 7$  0.0</td></tr><tr><td>Fixed BN</td><td> $8 5 . 8 4 \pm 0 . 6 8$  0.0</td></tr><tr><td>Fixed flip</td><td> $8 9 . 8 7 \pm 0 . 8 1$  0.0</td></tr><tr><td>Fixed B+flip</td><td> $8 6 . 2 2 \pm 0 . 4 2$  0.0</td></tr><tr><td>Fixed mean logits</td><td> $8 9 . 6 9 \pm 0 . 7 0$  0.0</td></tr><tr><td>Uniform empirical selection</td><td> $8 9 . 5 6 \pm 0 . 7 2$  96.0</td></tr><tr><td>Top-KL empirical selection</td><td> $8 9 . 0 6 \pm 0 . 8 0$  96.0</td></tr><tr><td>PORTAL, cumulative</td><td> $8 9 . 5 9 \pm 0 . 6 9$  1.9</td></tr><tr><td>PORTAL, decay 0.9</td><td> $8 9 . 5 9 \pm 0 . 6 9$  1.9</td></tr><tr><td>Audited change detector</td><td> $8 9 . 5 9 \pm 0 . 6 9$  33.9</td></tr></table>

The corruption-switch replay uses saved severity-five CIFAR-10-C predictions. Each segment contains 1,600 consecutive base-image indices aligned with the saved 64-image normalization batches.

A stream uses 6,400 distinct base images, so an image never reappears under another corruption within that stream. The original query/evaluation roles are preserved. The three schedules are Gaussian noise–brightness–shot noise–JPEG, shot noise–JPEG–impulse noise–brightness, and brightness–Gaussian noise–JPEG–shot noise. The same three source checkpoints are used for each schedule. Every selector starts with empty history at the beginning of a stream and carries it across segments.

Table 68: Sequential corruption-switch selection. Accuracy is mean ± checkpoint SD on the common evaluation images. Each stream has 100 batches and disjoint base images across segments. The fixed BN+flip output is the strongest fixed candidate in every complete schedule.
<table><tr><td>Schedule</td><td>Rule</td><td>Accuracy (%)</td><td>Labels</td></tr><tr><td>Noise first</td><td>Fixed BN+flip</td><td> $7 6 . 6 7 \pm 0 . 6 3$ </td><td>0.0</td></tr><tr><td>Noise first</td><td>Fixed mean logits</td><td>72.52 ± 1.40</td><td>0.0</td></tr><tr><td>Noise first</td><td>Uniform empirical selection</td><td>73.55 ± 0.65</td><td>300.0</td></tr><tr><td>Noise first</td><td>Top-KL empirical selection</td><td>74.07 ± 0.76</td><td>300.0</td></tr><tr><td>Noise first</td><td>PORTAL, cumulative</td><td>74.96 ± 1.00</td><td>57.3</td></tr><tr><td>Noise first</td><td>PORTAL, decay 0.9</td><td>74.98 ± 1.05</td><td>80.7</td></tr><tr><td>Noise first</td><td>Audited change detector</td><td>75.15 ± 0.90</td><td>144.0</td></tr><tr><td>Alternating</td><td>Fixed BN+flip</td><td>75.54 ± 0.76</td><td>0.0</td></tr><tr><td>Alternating</td><td>Fixed mean logits</td><td>72.08 ± 1.53</td><td>0.0</td></tr><tr><td>Alternating</td><td>Uniform empirical selection</td><td>72.84 ± 0.23</td><td>300.0</td></tr><tr><td>Alternating</td><td>Top-KL empirical selection</td><td>73.03 ± 0.77</td><td>300.0</td></tr><tr><td>Alternating</td><td>PORTAL, cumulative</td><td>74.18 ± 0.63</td><td>57.3</td></tr><tr><td>Alternating</td><td>PORTAL, decay 0.9</td><td>74.39 ± 0.75</td><td>85.7</td></tr><tr><td>Alternating</td><td>Audited change detector</td><td> $7 4 . 4 0 \pm 0 . 6 7$ </td><td>144.7</td></tr><tr><td>Brightness first</td><td>Fixed BN+flip</td><td> $7 6 . 0 2 \pm 0 . 6 9$ </td><td>0.0</td></tr><tr><td>Brightness first</td><td>Fixed mean logits</td><td> $7 3 . 3 5 \pm 1 . 7 6$ </td><td>0.0</td></tr><tr><td>Brightness first</td><td>Uniform empirical selection</td><td> $7 4 . 1 0 \pm 0 . 9 8$ </td><td>300.0</td></tr><tr><td>Brightness first</td><td>Top-KL empirical selection</td><td> $7 4 . 2 5 \pm 1 . 2 2$ </td><td>300.0</td></tr><tr><td>Brightness first</td><td>PORTAL, cumulative</td><td> $7 5 . 3 9 \pm 0 . 8 0$ </td><td>52.0</td></tr><tr><td>Brightness first</td><td>PORTAL, decay 0.9</td><td> $7 5 . 3 0 \pm 0 . 9 5$ </td><td>87.0</td></tr><tr><td>Brightness first</td><td>Audited change detector</td><td> $7 5 . 4 6 \pm 0 . 7 5$ </td><td>138.3</td></tr></table>

## W.5 EXACT PAIRED-LABEL CALCULATIONS ON REAL PREDICTIONS

Within each corruption segment, we compare BN+flip with mean logits. The two outputs are correct or incorrect on the same underlying images. Their paired correctness difference takes values 1, −1, or 0. Its query-pool mean and variance determine the finite-pool outcome counts in Equation (56). We compute the probability that a uniform sample selects BN+flip, including ties, and use the separate evaluation pool to identify the better output. We then draw 2,000 independent query subsets per checkpoint, segment, and budget to check the exact result.

The pool contains about 800 eligible inputs per segment. Budgets are 0, 3, 15, 75, and 300 pooled labels. These calculations select one output for a complete segment; the sequential selectors in Table 68 instead select once per batch. Table 69 gives the exact probability of choosing the worse output. The zero-label rule always selects BN+flip. It is correct in the noise segments and wrong in the brightness and JPEG segments. Small nonzero budgets can disturb a correct default before enough evidence has accumulated to stabilize the empirical choice.

Table 69: Exact probability (%) of selecting the worse of BN+flip and mean logits. Means cover three checkpoints. Columns give the number of labels sampled uniformly without replacement from one corruption segment. Tie decisions use BN+flip. Differences between schedules also reflect their distinct image-index blocks.
<table><tr><td>Schedule</td><td>Corruption</td><td>3 labels</td><td>15 labels</td><td>75 labels</td><td>300 labels</td></tr><tr><td>Noise first</td><td>Brightness</td><td>90.82</td><td>64.90</td><td>24.05</td><td>2.35</td></tr><tr><td>Noise first</td><td>Gaussian noise</td><td>11.53</td><td>11.59</td><td>1.73</td><td>0.00</td></tr><tr><td>Noise first</td><td>Jpeg compression</td><td>86.02</td><td>59.90</td><td>35.03</td><td>14.52</td></tr><tr><td>Noise first</td><td>Shot noise</td><td>11.66</td><td>16.63</td><td>7.61</td><td>1.06</td></tr><tr><td>Alternating</td><td>Brightness</td><td>91.41</td><td>67.66</td><td>30.35</td><td>4.71</td></tr><tr><td>Alternating</td><td>Impulse noise</td><td>12.40</td><td>15.79</td><td>5.53</td><td>0.06</td></tr><tr><td>Alternating</td><td>Jpeg compression</td><td>85.00</td><td>57.94</td><td>31.59</td><td>8.00</td></tr><tr><td>Alternating</td><td>Shot noise</td><td>14.73</td><td>22.30</td><td>14.01</td><td>2.98</td></tr><tr><td>Brightness first</td><td>Brightness</td><td>91.20</td><td>68.50</td><td>37.15</td><td>10.74</td></tr><tr><td>Brightness first</td><td>Gaussian noise</td><td>10.64</td><td>11.26</td><td>1.84</td><td>0.00</td></tr><tr><td>Brightness first</td><td>Jpeg compression</td><td>86.71</td><td>61.90</td><td>38.47</td><td>15.51</td></tr><tr><td>Brightness first</td><td>Shot noise</td><td>14.04</td><td>18.81</td><td>7.87</td><td>0.35</td></tr></table>

## W.6 HISTORICAL EVIDENCE UNDER A CONTROLLED LABEL CHANGE

The synthetic experiment uses four candidates, three classes, and two fixed prediction patterns. Each batch has 64 inputs: 16 query and 16 evaluation inputs in each pattern. Every batch therefore has the same stratum weights. The class probabilities are shared by query and deployment inputs. Before a change, the probabilities of classes 0, 1, and 2 are 0.725, 0.225, and 0.05. An abrupt change exchanges the first two probabilities at batch 50. A gradual change interpolates between them over batches 50–70. The stable stream keeps the initial law. Five independent seeds generate the potential labels shared by all policies.

Cumulative history retains acquired counts. No-history selection resets the prior each batch. Decay multiplies a pattern’s historical counts by 0.9 when it reappears. An audit variant buys one label from the largest available stratum before ordinary selection. Audit/reset also clears that stratum’s counts when the observed class had a posterior probability below its threshold. The strata have equal deployment weights and query counts here. The deterministic tie rule repeatedly sends the first audit to the same stratum.

Table 70: Conditional-label change with fixed predictions and stratum weights. Accuracy is in percent; labels count purchases within the displayed period, including audits. Stable and abrupt columns cover the final 50 batches. Gradual columns cover the final 30, after the transition. Means cover five synthetic seeds.
<table><tr><td rowspan="2">History</td><td colspan="2">Stable</td><td colspan="2">Abrupt</td><td colspan="2">Gradual</td></tr><tr><td>Acc.</td><td>Labels</td><td>Acc. Labels</td><td></td><td>Acc. Labels</td><td></td></tr><tr><td>Cumulative</td><td>72.21</td><td>0.0</td><td>22.54</td><td>0.0</td><td>22.92</td><td>0.0</td></tr><tr><td>No history</td><td>62.60</td><td>150.4</td><td>64.10</td><td>150.2</td><td>63.42</td><td>90.0</td></tr><tr><td>Decay 0.9</td><td>72.21</td><td>150.4</td><td>67.66</td><td>149.8</td><td>70.88</td><td>85.4</td></tr><tr><td>One audit per batch</td><td>72.21</td><td>50.0</td><td>27.24</td><td>72.8</td><td>22.92</td><td>30.0</td></tr><tr><td>Audit/reset 0.05</td><td>72.21</td><td>54.2</td><td>38.34</td><td>74.4</td><td>33.71</td><td>38.2</td></tr><tr><td>Audit/reset 0.30</td><td>71.78</td><td>132.4</td><td>55.36</td><td>125.4</td><td>52.73</td><td>71.4</td></tr></table>

After the abrupt change, cumulative history never selects the expected-optimal action for five consecutive batches in any seed. Decay reaches this criterion after 7.0 batches on average. After the gradual change crosses the decision boundary, its mean delay is 10.4 batches. The reset threshold of 0.30 improves recovery relative to 0.05, but also causes 19.4 resets on average in the stable stream. These observations separate acquiring a diagnostic label from replacing enough old evidence to change the decision.

A window-based change detector improves recovery on the synthetic stream. On PACS, it uses more labels and reaches lower accuracy than cumulative history (Table 65). This motivates the budgetmatched checks below.

## W.7 BUDGETED AUDITS UNDER FEEDBACK BIAS AND LABEL DRIFT

We next compare guarded selection with two explicitly defined estimates of class probabilities. Exponentially weighted (EW) counts discount past labels by 0.95 per batch. Direct inverse-propensity weighting with EW (Direct IPW+EW) also corrects for unequal query probabilities. For query probability $\pi _ { j }$ and eligible pool size $| Q | ,$ its label weight is min $\{ 4 , \bar { 1 } / ( \bar { \pi } _ { j } \bar { | Q | } ) \}$ . The cap is inactive in the reported runs. Each comparator chooses the candidate with the highest estimated deployment accuracy. Both share the guarded selector’s predictions and total label ceiling. These are our reconstructions: the original implementations named in the experiment specification were unavailable.

The two-pattern stream keeps predictions fixed and shares class probabilities between query and deployment inputs within each batch. Those probabilities stay stable, reverse abruptly, or change gradually. The guarded selector reserves periodic labels, buys a short burst after surprising evidence, and tests a switch before retaining it.

At a 120-label ceiling, its stable-period difference from the strongest fixed rule is −0.122 pp, with paired interval [−0.150, −0.094] over 960 fresh streams. On 360 paired changing streams, postchange gains over same-budget EW are $+ 9 . 5 2 \mathrm { p p }$ for abrupt change and $+ 4 . 8 7 \mathrm { p p }$ for gradual change. Recovery starts within ten batches in 0.92 and 0.80 of streams, respectively. Raising only the ceiling to 160 improves gradual recovery to 0.94. The stable-period difference becomes $- 0 . 1 0 0 \mathrm { p p }$ , with interval [−0.128, −0.071]. We report each budget separately.

The sparse protocol draws four of 300 hidden modes per batch. A mode is a subgroup with its own class probabilities; several modes share a prediction stratum. A tenth of each mode’s examples have an atypical label distribution. Prioritized audits oversample these examples, creating feedback bias even before the distribution changes.

The sparse selector corrects acquired labels using their known query probabilities. Every input type must have positive query probability. A run-length posterior estimates how long the current label distribution has persisted. Its change hazard, the prior probability of change per batch, is 0.02. Under the 100-label ceiling, the selector starts with eight labels and reserves periodic audits. Surprising evidence triggers a burst capped at five labels per batch, which may continue for three batches after the trigger. Uniform and prioritized schedules use the same candidates and budget. Table 71 reports 28 independently generated blocks of 30 seed streams each. Gradual-change estimates exceed EW with intervals spanning zero; recurrent-change gains have positive intervals under both schedules.

Table 71: Post-change accuracy difference from reconstructed EW in percentage points on the 300- mode sparse protocol. Priority oversamples atypical examples within a mode; uniform draws do not. Each cell contains 28 blocks and 840 seed streams, with the same scored examples and 100-label ceiling for both methods.
<table><tr><td rowspan=1 colspan=4>Change                       Prioritized audit                       Uniform audit</td></tr><tr><td rowspan=2 colspan=3>Abrupt                     +3.94 [0.59, 7.28]</td><td rowspan=2 colspan=1> $+ 4 . 7 5 \ : [ 1 . 3 6 , 8 . 1 4 ]$ </td></tr><tr><td rowspan=1 colspan=1>Gradual</td><td rowspan=1 colspan=1>+1.69 [-3.24, 6.62</td><td rowspan=2 colspan=1>+3.84 [1.24, 6.43]</td></tr><tr><td rowspan=1 colspan=3>Recurrent                  +3.28 [0.61, 5.95]</td></tr></table>

Under 100 labels and prioritized audits, recovery within ten batches is 0.685, 0.798, and 0.642 at recurrent changes 25, 50, and 75. The target is 0.90 at each change. A separate 160-label variant changes both the budget and selector settings. It reaches 0.907, 0.933, and 0.893 across five fresh blocks. Its pooled recovery is 0.911, with bootstrap interval [0.884, 0.936]. The final change remains below the target. Post-change accuracy exceeds same-budget EW by 8.55 pp, with paired interval [7.68, 9.41] over 150 streams. Mean harmful-batch rates are 3.15% under prioritized abrupt change and 4.62% under gradual change. The gradual-change 95% upper bound is 5.07%, above the 5% threshold.

A harmful choice has lower accuracy than the prespecified strong fixed candidate on the same de ployment inputs. We count the fraction of harmful batches in each stream. At 100 labels, each seed contributes an average over six combinations of change scenario and query schedule. Across 840 seeds, the pooled mean is 3.73%, with a one-sided 95% t upper bound of 3.91%. Recurrent streams have a higher mean of 7.46% across the two schedules. Individual six-scenario means exceed 5% for 193 seeds, and eleven of 28 block intervals cross 5%. These measurements locate the low pooled rate and the higher harm under repeated changes.

On real PACS predictions, the guarded four-candidate selector reaches 59.610% with 131 labels. The strongest fixed candidate reaches 59.677%, and reconstructed EW reaches 56.987%. The stabledomain difference from the strongest fixed candidate is −0.020 pp. Five arrival orders share one checkpoint, so this comparison measures arrival-order variation.

On CIFAR-10.1, a separate fixed probability ensemble uses zero labels and scores 89.733%, against 89.867% for the strongest fixed candidate. The paired difference is −0.133 pp, with interval [−1.216, +0.949] over three checkpoints. The noninferiority criterion allows a loss of at most 0.20 pp. The point estimate meets it; the interval is too wide to establish it. Repeated orders are averaged within checkpoints. This experiment evaluates an additional fixed candidate.

Numerical and implementation checks. The stored CIFAR-10.1 predictions reproduce batch errors and stream totals. Fixed outputs in the dynamic experiment match the original predictions. Label counts include all audit requests. Replacing withheld evaluation labels leaves query choices unchanged. An independent finite-pool calculation samples the disagreement count first and then its favorable outcomes; it agrees with the exact probabilities. These checks validate the calculations. Statistical uncertainty uses the checkpoints and orders specified for each experiment.

## X RELATED WORK

The literature offers many ways to change a deployed model. It also offers ways to judge whether a change helps. PORTAL brings these two questions together through a shared label budget.

## X.1 FEEDBACK AND THE RISK OF A WRONG CHOICE

Adaptive labeling and label-free accuracy estimation help evaluate models after distribution shift (Mittal et al., 2024; Lee et al., 2024c). Accuracy–agreement relations provide another signal from unlabeled data (Kim et al., 2024). Binary feedback reduces the amount of information requested from an annotator (Lee et al., 2025a). In our setting, a full class label judges every candidate on the same input.

Feedback can lose its value as the stream changes. Risk monitoring, protected entropy matching, and domain-shift detection track different signs of trouble (Schirmer et al., 2025; Bar et al., 2024; Park et al., 2025a). Recovery time and prediction latency add two costs beyond current accuracy (Zhou et al., 2026; Sreeram et al., 2026). Our drift experiments measure how quickly fresh labels can replace stale evidence in the selection decision.

The deployment population may also change in ways that a prediction pattern hides. Open-set inputs, malicious test samples, and noisy inputs challenge adaptation in different ways (Gao et al., 2024; Park et al., 2024; Cao et al., 2025). Domain compensation addresses open-set streams, while selective adaptation isolates a shifted modality (Yang et al., 2026; Chen et al., 2025). Long-tailed classes add another source of unequal feedback (Wang et al., 2026). These settings make the link between queried and deployed inputs a substantive part of the decision.

## X.2 HOW CANDIDATE PREDICTIONS CHANGE

An intervention can update a model through self-training or prediction energy (Ma, 2024; Yuan et al., 2024). Disentangled objectives show that prediction entropy alone does not describe every useful update (Lee et al., 2024b). Ranked entropy and entropy–energy duality refine the signal used for adaptation (Han et al., 2025; Park et al., 2025b). Uncertainty calibration, Tsallis entropy, and sharpness-aware regularization offer further choices (Tan et al., 2025; Wu et al., 2026b; Niu et al., 2025).

Bayesian updates and online expectation–maximization change how a candidate uses incoming examples (Zhou et al., 2025; Dai & Yang, 2025). Inter-class topology and adaptive representation alignment address structure across a changing stream (Ni et al., 2025; Zhang et al., 2024b). Each method creates predictions that can be judged against a fixed rule on the next deployment batch.

An update’s state can live in batch-normalization memory, an organized example buffer, or a tracked subspace (Kang et al., 2024; Zhu et al., 2024; Duan et al., 2025). Expert blending chooses an input-dependent mixture (Lee et al., 2024a). Diffusion-based transformations change corruptions or align an input with a synthetic domain (Oh et al., 2024; Guo et al., 2025). Our retain-versus-undo experiment measures the later effect of keeping one such state change.

The cost of constructing a candidate also matters. Forward-only adaptation, backpropagation-free alignment, and activation-function adaptation alter the required computation (Niu et al., 2024; Ma et al., 2026; Kim et al., 2026). Sparse updates and activation sparsity reduce work or memory (Cha et al., 2025; Ma et al., 2025). Cloud–edge distillation, cross-device merging, and zeroth-order prompt optimization offer other deployment choices (Chen et al., 2024; Mitsuzumi et al., 2026; Wang & Wang, 2026). Label cost and computation cost remain distinct when choosing a complete system.

## X.3 VISION–LANGUAGE PREDICTION RULES

Simple adaptation rules and analyses of zero-shot generalization provide strong starting points for vision–language models (Farina et al., 2024; Zanella & Ayed, 2024). Regional bootstrapping, attribute prompts, and hierarchical knowledge change what the model sees or asks of a class (Zhang et al., 2024a; 2025a;b). Semantic integration and dynamic prompts offer further candidate rules (Huang et al., 2025a; Xiao et al., 2025). A prompt coreset carries selected context across the stream (Zhang et al., 2025c).

Candidates can also use spatial bias, both modalities, or fewer visual tokens (Huang et al., 2025b; Maharana et al., 2025; Wang et al., 2025). Retrieval and label propagation use neighboring examples to change predictions (Lee et al., 2025b; Fan et al., 2025; Li et al., 2025c). Prototypes and attributive–associative memory retain different summaries of past inputs (Huang et al., 2026; Zhang et al.). Dual debiasing and dynamic logit adjustment modify the final scores (Luo et al.; Wu et al., 2026a). These designs enlarge the set of rules that a selector may compare.

Flatness-guided and robust prompt methods target prediction stability (Li et al., 2025a; Zhu et al., 2026; Kim & Um, 2026). Flatness-aware pretraining addresses prompt calibration (Jang et al., 2026). Bilateral information use and universal prompt learning explore different sources of testtime signal (Sun et al., 2026; Fu et al., 2026). Test-time adaptation also supports zero-shot value functions (Ziakas & Russo, 2026). Beyond single-label classification, cross-modal retrieval and multi-label prediction need different outputs (Li et al., 2025b; Wu et al., 2025). Our experiments use a shared class label so that one query directly compares all candidate rules.