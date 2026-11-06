# SIMPLE META-LEARNING FOR HPO AND CASH USING PERFORMANCE REPRESENTATIONS

Emiel Witting, Cheng Yan & Tom Julian Viering

Delft University of Technology

Delft, The Netherlands

emiel.witting@gmail.com, {C.Yan-1,t.j.viering}@tudelft.nl

## ABSTRACT

Bayesian optimization for Hyperparameter Optimization (HPO) and Combined Algorithm Selection and Hyperparameter Optimization (CASH) often relies on explicit hyperparameter encodings as surrogate inputs. In high-dimensional search spaces, however, such representations can make it difficult for standard surrogates to learn useful performance models. We instead study an alternative meta-learning representation that describes each configuration directly as its historical performance across prior tasks. This representation can be paired directly with standard surrogate models. Our scope is the iterative selection component of AutoML, deciding which configuration to evaluate next from a fixed candidate set, rather than a complete framework with features such as runtime estimation, cost-aware budgeting and ensembling. We further apply this representation to maximal marginal relevance (MMR), a diversity-based re-ranking criterion from recommender systems, as a posterior-free acquisition rule. When selecting and optimising across multiple models on the TabArena benchmark, the performancebased representation paired with a linear Bayesian ridge regression model outperforms dedicated meta-learning baselines, achieving the same tuned performance in roughly half the number of trials. For single-model HPO, it remains a competitive second. Across both benchmarks this method closely matches or improves on regret at a fraction of the complexity and cost, requiring no pre-training or tuning and scaling independently of search space dimensionality. Code for reproducing our results is available at https://github.com/EWitting/simplemetalearning.

## 1 INTRODUCTION

Automated machine learning (AutoML) aims to remove the trial-and-error of selecting an algorithm and tuning its hyperparameters for a given task. The dominant paradigm casts this as hyperparameter optimization (HPO), or, when the algorithm itself is a decision variable, as Combined Algorithm Selection and Hyperparameter Optimization (CASH) (Thornton et al., 2013). Both are typically solved with Bayesian Optimization (BO), in which a surrogate model predicts the performance of a candidate configuration and an acquisition function selects what to evaluate next (Brochu et al., 2010). When historical results on related tasks are available, meta-learning can transfer this information to improve optimization on a new task (Brazdil et al., 2022). Such transfer has been realised through multi-task and transfer surrogates that share information across related optimization problems (Swersky et al., 2013; Wistuba et al., 2018; Feurer et al., 2018; Wistuba & Grabocka, 2021; Jomaa et al., 2021; Wang et al., 2024), or more recent meta-trained or in-context models that learn from collections of prior tasks end-to-end (Wei et al., 2019; Ruhkopf et al., 2023; Pineda Arango et al., 2024).

Despite their differences, most of these methods share the same configuration representation: an encoding of the hyperparameter vector, mapped to performance prediction. In CASH, this representation often concatenates the hyperparameter spaces of every candidate algorithm together with an algorithm identifier, leading to a high-dimensional conditional space in which most input dimensions are inactive for any given configuration (Thornton et al., 2013; Feurer et al., 2015; Lindauer et al., 2022). Thus, learning useful interactions with such an input space is challenging.

![](images/d874afcb6ef4f5ba77f96137638328d74a8de1f5edaab1fc50702f653a5e1a74.jpg)  
Figure 1: Hyperparameter-based vs. performance-based configuration representations. To estimate performance on a target task, standard BO represents a configuration by its hyperparameter vector λ (top) and possibly uses results on prior tasks for meta-learning, whereas our approach uses the historical performance vector P<sub>λ</sub> across prior tasks directly as input representation (bottom). Our approach can be interpreted as learning to describe the target task in terms of prior tasks, for example as weighted sum of prior tasks in the case of a linear estimator.

An alternative line of work, inspired by collaborative filtering (Koren et al., 2009), represents configurations by their historical performance in prior tasks to tackle the AutoML problem (Fusi et al., 2018; Yang et al., 2019). This view aligns naturally with recommender systems, where tasks play the role of users, configurations of items, and validation scores of ratings. Prior matrix-based methods typically exploit this structure through compressed or latent representations derived from the task–configuration performance matrix (Fusi et al., 2018; Yang et al., 2019).

We instead ask whether the raw historical performance vector itself can serve directly as a configuration representation for standard surrogate models. Figure 1 illustrates this contrast between the standard hyperparameter-based view and our performance-based representation. Additionally, we find that historical performance vectors also support a useful measure for representing diversity of configurations. In combination with maximal marginal relevance (MMR), a classical re-ranking criterion from information retrieval and recommender systems (Carbonell & Goldstein, 1998), and using historical mean performance as the relevance, this yields MeanMMR, a model-free acquisition rule that can stand in for uncertainty-driven exploration normally seen in Bayesian optimisation.

We benchmark these methods on TabArena (Erickson et al., 2025), a recent benchmark that covers diverse algorithms with randomly sampled hyperparameter configurations each across carefully curated realistic tabular classification and regression tasks, yielding a high-dimensional CASH search space. The performance-based representation paired with Bayesian ridge regression outperforms dedicated meta-learning baselines on CASH and remains a competitive second on HPO, at a fraction of their complexity and cost. Surprisingly, our simple model-free MeanMMR baseline also obtains lower average regret than these baselines on CASH.

Scope. We study the selection loop of AutoML, not a complete AutoML framework. Systems such as AutoGluon (Erickson et al., 2020) and auto-sklearn (Feurer et al., 2022) add features such as runtime estimation, cost-aware budget allocation and ensembling, which would confound a comparison that isolates the configuration representation, so we do not compare against them. Our representation is a component such frameworks could adopt for their selection step.

To summarize, our contributions are:

• We propose to use the raw historical performance vectors directly as configuration representations in BO, offering a simple route to meta-learning that allows standard surrogate models to meta-learn from previous tasks without specialized meta-learning procedures.

• We introduce MMR as a diversity-based, posterior-free acquisition function, and show that combined with historical mean performance it forms MeanMMR, a strong model-free meta-learning baseline.

• We benchmark popular surrogate models crossed with both representations on the TabArena benchmark, isolating the effect of the representation, and find that the performance representation makes a simple Bayesian ridge surrogate competitive with or better than dedicated meta-learning baselines at a fraction of their cost.

## 2 RELATED WORK

Table 1 provides an overview of related work and the positioning of our work.

Hyperparameter Optimization and CASH. HPO aims to find the hyperparameter configuration of a learning algorithm that maximizes a chosen performance metric on a target dataset (Hutter et al., 2019). Thornton et al. (2013) extend HPO to the case where the algorithm itself is also a decision variable, Combined Algorithm Selection and Hyperparameter Optimization (CASH). As the number of hyperparameters grows, exhaustive enumeration becomes infeasible, and random search becomes a simple and widely used baseline (Bergstra & Bengio, 2012; Hutter et al., 2019).

To handle large search spaces, practical HPO and CASH methods treat configuration search as a sequential allocation problem, deciding which candidate to evaluate next under a limited evaluation budget. Hutter et al. (2011) introduce Sequential Model-Based Optimization (SMBO), which fits a surrogate model to observed configuration performances and uses an acquisition function to select the next configuration. Bayesian Optimization (BO) (Snoek et al., 2012) is a common probabilistic instance of this framework, using the surrogate’s predictive distribution to trade off exploiting promising configurations and exploring uncertain regions (Brochu et al., 2010). Modern AutoML systems typically operate on explicit hyperparameter-space encodings of candidate configurations (Lindauer et al., 2022; Falkner et al., 2018; Thornton et al., 2013; Kotthoff et al., 2017; Feurer et al., 2015; 2022).

The CASH search space is structurally more challenging than ordinary HPO, concatenating hyperparameters across every candidate algorithm into a high-dimensional, heavily conditional space (Feurer et al., 2015; Lindauer et al., 2022). Prior work also finds that only a small subset of hyperparameters tends to dominate performance variation (Hutter et al., 2014; Van Rijn & Hutter, 2018; Probst et al., 2019), so a surrogate using an explicit hyperparameter encoding must infer the relevant dimensions and interactions from few target-task evaluations.

Algorithm Selection. A related problem setting is algorithm selection (AS) from a finite portfolio of algorithms, without explicitly considering any hyperparameter values (Brazdil et al., 2022). The simplest strategy ranks candidates by their average performance across prior tasks. Active Testing (Leite et al., 2012) makes this adaptive: it selects which candidate to evaluate next on the target task by estimating, from cross-task performance similarity, which evaluation is most informative for identifying the best candidate. These methods rely entirely on historical performance rather than a hyperparameter-space surrogate, similar to our setting, but use fixed or heuristic tournamentstyle ordering without Bayesian optimisation’s uncertainty-driven exploration. We instead reframe HPO and CASH as the algorithm selection problem setting, while retaining Bayesian optimisation techniques.

Meta-Learning and Transfer for Bayesian Optimization. When historical results on related tasks are available, Bayesian optimization can transfer information to a new task (Brazdil et al., 2022). Swersky et al. (2013) introduce multi-task BO, which couples related problems through a joint task kernel. Ranking-weighted ensembles such as RGPE (Feurer et al., 2018) and two-stage transfer surrogates such as TST-R (Wistuba et al., 2018) combine per-task Gaussian process models using cross-task ranking agreement. ABLR (Perrone et al., 2018) meta-trains a shared hyperparameter feature embedding network with a per-task Bayesian linear head. Further related work spans deep-kernel transfer, pre-trained Gaussian processes, and in-context meta-models (Wistuba & Grabocka, 2021; Jomaa et al., 2021; Wang et al., 2024; Wei et al., 2019; Pineda Arango et al., 2024; Ruhkopf et al., 2023). These methods share two traits: a hyperparameter-encoded configuration input, and a need for specialized meta-learning machinery. We instead move the meta-learning into the input representation, so that a standard surrogate suffices.

Collaborative Filtering and Matrix-Based AutoML. A parallel line of work, inspired by collaborative filtering, observes that a task-configuration performance matrix mimics the user-item ratings matrix used in recommender systems. Tasks act as users, configurations as items, and validation scores as ratings (Koren et al., 2009; Yang et al., 2019; Fusi et al., 2018). Existing matrix-based methods typically compress this matrix into latent representations. Yang et al. (2019) introduce OBOE, which constructs compact configuration representations using low-rank matrix factorization and PCA-based embeddings. Fusi et al. (2018) introduce PMF, which learns latent configuration representations by gradient-based optimization and naturally handles sparse performance matrices. Two-Tower models add deep learning to factorization, generating separate task and configuration embeddings via multilayer perceptrons before scoring their compatibility (Huang et al., 2013; Covington et al., 2016; Yi et al., 2019; Raza et al., 2026). In contrast to all of these, we use the observed historical performance vector directly as a surrogate input, without an additional compression or factorization step. Our linear model is related to OBOE, and we compare Two-Tower variants embedding either hyperparameters or performance directly (see Appendix B.10).

Table 1: Positioning of our approach against representative methods from four lines of related work. We combine the performance-matrix representation of algorithm selection with the active surrogateand-EI loop of Bayesian optimization, using the raw performance vector and any standard surrogate, without bespoke meta-training or latent compression. Abbreviations: BO, Bayesian optimization; AS, algorithm selection; CF, collaborative filtering.
<table><tr><td>Method</td><td>Line of work</td><td>Config. representation</td><td>Surrogate</td><td>Selection</td></tr><tr><td>SMAC3 (Lindauer et al., 2022)</td><td>BO</td><td>Hyperparameter</td><td>Random forest</td><td>EI</td></tr><tr><td>GP-BO (Snoek et al., 2012)</td><td>BO</td><td>Hyperparameter</td><td>Gaussian process</td><td>EI</td></tr><tr><td>Average ranking (Brazdil et al.,</td><td>AS</td><td>Performance</td><td>None (ranking)</td><td>Static</td></tr><tr><td>2022) Active Testing (Leite et al., 2012)</td><td>AS</td><td>Performance</td><td>None (matrix)</td><td>Active</td></tr><tr><td>Multi-task GP (Swersky et al., 2013)</td><td>Meta-learning BO</td><td>Hyperparameter</td><td>Multi-task GP</td><td>EI</td></tr><tr><td>RGPE (Feurer et al., 2018)</td><td>Meta-learning BO</td><td>Hyperparameter</td><td>GP ensemble</td><td>EI</td></tr><tr><td>TST-R (Wistuba et al., 2018)</td><td>Meta-learning BO</td><td>Hyperparameter</td><td>GP ensemble</td><td>EI</td></tr><tr><td>ABLR (Perrone et al., 2018)</td><td>Meta-learning BO</td><td>Hyperparameter</td><td>NN + Bayesian linear</td><td>EI</td></tr><tr><td>OBOE (Yang et al., 2019)</td><td>CF / matrix</td><td>Performance (PCA)</td><td>Linear</td><td>Custom</td></tr><tr><td>PMF (Fusi et al., 2018)</td><td>CF / matrix</td><td>Performance (latent)</td><td>Gaussian process</td><td>EI</td></tr><tr><td>Two-Tower (Huang et al., 2013)</td><td>CF / matrix</td><td>Hyper. / perf. (latent)</td><td>Two-tower network</td><td>Greedy / MMR</td></tr><tr><td>Ours</td><td>AS + BO</td><td>Performance</td><td>Any regression model</td><td>EI / MMR</td></tr></table>

## 2.1 DIVERSITY-BASED ACQUISITION AND MMR

Beyond the surrogate model, our work also revisits the acquisition function used to select the next configuration. A common BO acquisition function is Expected Improvement (EI) (Jones et al., 1998), which uses the surrogate’s predictive posterior to balance promising predicted performance against uncertainty (see Appendix B.5 for a short primer). When the surrogate does not provide a calibrated predictive posterior, standard uncertainty-driven rules such as EI do not directly apply. Prior matrix-based AutoML methods face the same issue: OBOE’s bilinear low-rank surrogate (Yang et al., 2019) is not posterior-providing, so it replaces EI with a custom procedure, motivating more general posterior-free alternatives.

Maximal marginal relevance (MMR) was originally introduced for document retrieval and summa rization and has since been used as a diversification strategy in recommender and retrieval-oriented settings (Carbonell & Goldstein, 1998; Vargas & Castells, 2011; Di Noia et al., 2017; Chen et al., 2021; Do et al., 2025; Korikov & Sanner, 2026). Related diversity-aware ideas have also appeared in ensemble-oriented CASH and diversified batch Bayesian optimization (Shen et al., 2022; Nava et al., 2022). However, to the best of our knowledge, we are the first to formulate MMR as acquisition rule for diversity-aware exploration in standard HPO and CASH, providing a simple posterior-free alternative to uncertainty-driven acquisition.

## 3 PERFORMANCE-BASED REPRESENTATIONS FOR SIMPLE META-LEARNING

## 3.1 PROBLEM SETUP

We study HPO and CASH as sequential configuration selection over a finite candidate set Λ. Each $\lambda \in \Lambda$ denotes a fully specified candidate that can be evaluated on a learning task $t ,$ yielding a performance score $\mathcal { P } ( t , \bar { \lambda } )$ where larger values indicate better performance. In the HPO setting, Λ contains configurations of a single algorithm; in the CASH setting, Λ spans a joint conditional configuration space over all candidate algorithms and their hyperparameters, represented through a one-hot algorithm indicator together with the full hyperparameter encoding. For a target task $t ,$ an optimization method iteratively selects an unevaluated $\lambda \in \Lambda$ , observes $\mathcal { P } ( t , \lambda )$ , and updates a surrogate estimator $\hat { \mathcal { P } }$ for unobserved configurations. We restrict all methods to the same Λ and instantiate each surrogate model family with two alternative inputs $x _ { \lambda } \colon$ the standard hyperparameterbased representation and a performance-based representation built from the historical performance across prior tasks, allowing simple meta-learning.

## 3.2 PERFORMANCE-BASED REPRESENTATION FOR META-LEARNING

We propose a performance-based representation that describes a configuration by how it performed on previous tasks, instead of describing a configuration in terms of hyperparameter values. Let $T _ { \mathrm { t r a i n } }$ be the set of prior tasks available for meta-learning. We collect the corresponding scores in a historical performance matrix $P \in \mathbb { R } ^ { | T _ { \mathrm { t r a i n } } | \times | \Lambda | }$ , where each entry is $P _ { t , \lambda } = \mathcal { P } ( t , \lambda )$ for $t \in T _ { \mathrm { t r a i n } }$ and $\lambda \in \Lambda$ . Each column of $P$ corresponds to one configuration and each row corresponds to one prior task. The column $P _ { \lambda } : = P _ { : , \lambda }$ is therefore the historical performance vector of configuration $\lambda ,$ and we use this column as the estimator input, $x _ { \lambda } ^ { \mathrm { p e r f } } = P _ { \lambda }$ . Configurations are thus compared by their empirical behaviour across previous tasks rather than by their hyperparameter values. This moves meta-learning into the input representation itself, allowing any standard surrogate that accepts vector inputs to exploit previous tasks by replacing $x _ { \lambda } ^ { \mathrm { h y p e r } }$ with $x _ { \lambda } ^ { \mathrm { p e r f } }$ , without changing the surrogate architecture, kernel, or training objective.

## 3.3 SURROGATE MODELS FOR PERFORMANCE BASED REPRESENTATIONS

We focus on the surrogate central to our contribution, Bayesian ridge regression, which to the best of our knowledge we are the first to study for HPO and CASH with the proposed performance-based representation. We additionally study a Two-Tower collaborative-filtering network as a deep metalearning comparison; as it is not a central contribution, its description and results are deferred to Appendix B.10.

Linear (Bayesian ridge regression). Since the performance-based representation already encodes how each configuration behaves across prior tasks, we use Bayesian ridge regression (MacKay, 1992; Tipping, 2001) as a low-capacity surrogate that remains stable with few target-task observations and provides the predictive posterior needed for EI (Jones et al., 1998). Specifically, we use a linear-Gaussian model with a zero-mean isotropic Gaussian prior on the weights and Gamma hyperpriors on the noise and weight precisions. Training uses iterative empirical Bayes/evidencemaximization updates of these precisions rather than variational inference; conditional on the precisions, the weight posterior is Gaussian in closed form. When the input is $x _ { \lambda } ^ { \mathrm { p e r f } }$ , the coefficients can be interpreted as weights over prior tasks in $T _ { \mathrm { t r a i n } }$ , so target-task adaptation learns which previous tasks are informative for the current one. Implementation details are given in Appendix C.1, and its relation to OBOE is discussed in Appendices B.4 and C.2.

## 3.4 THE MAXIMAL MARGINAL RELEVANCE (MMR) ACQUISITION AND MEANMMR BASELINE

To support surrogates without a calibrated predictive posterior, we adapt MMR (Carbonell & Goldstein, 1998) as a diversity-based acquisition function that selects configurations with high predicted relevance while penalizing similarity to configurations already evaluated on the target task. Let $\Lambda _ { \mathrm { o b s } }$ be the set of configurations already evaluated on the target task. MMR selects the next configuration

![](images/0d827783ac8ac5ad7ba5e490d32ce5856bd13c89ce8147d59b759341602798fc.jpg)  
Random Mean MeanMMR Linear (Perf) Linear (Hyper) RF (Perf) RF (Hyper) GP (Perf) GP (Hyper) ABLR Active Testing  
Figure 2: Test normalized regret per model family and overall rank vs. number of evaluations on the CASH benchmark (TabArena). Methods introduced in this work (the Perf variants and MeanMMR) are marked with an upward-pointing triangle (▲). Shaded bands are 95% bootstrap CIs across tasks.

by maximizing:

$$
\mathrm { M M R } ( \lambda ) = \alpha \cdot \hat { \mathcal { P } } ( t , \lambda ) - ( 1 - \alpha ) \cdot \operatorname* { m a x } _ { \lambda ^ { \prime } \in \Lambda _ { \mathrm { o b s } } } \cos ( z _ { \lambda } , z _ { \lambda ^ { \prime } } )\tag{1}
$$

where $\hat { \mathcal { P } } ( t , \lambda )$ is the relevance score, $z _ { \lambda }$ is the representation used to compute similarity, and α controls the relevance-diversity tradeoff (larger α places more emphasis on relevance, while smaller α encourages greater novelty relative to previously evaluated configurations).

Because MMR only requires a relevance score and a similarity, we can drop the surrogate altogether. MeanMMR replaces the relevance score $\hat { \mathcal { P } } ( t , \lambda )$ in Equation 1 with the historical mean performance of λ across T<sub>train</sub>, and computes similarity directly on the performance vectors $P _ { \lambda }$ (full expression in Appendix B.5). It turned out to be a surprisingly strong baseline.

## 4 EXPERIMENTAL SETUP

Baselines. We compare results against dedicated meta-learning methods: the hyperparameterbased MTGP (Swersky et al., 2013), RGPE (Feurer et al., 2018) and ABLR (Perrone et al., 2018), as well as Active Testing (Leite et al., 2012) designed for algorithm selection. These dedicated baselines are included where our compute budget allowed, as the GP-based methods (MTGP, RGPE) do not stay computationally feasible on the high-dimensional CASH benchmark, and MTGP required subsampling modification to apply to HPO (see Appendix B.9 for computational cost analysis).

In addition, we include three more reference baselines. Random search (Bergstra & Bengio, 2012) uses no prior information and suggests configurations in random order. Mean ranking orders configurations by their average performance in P, one of the simplest forms of meta-learning and a well-studied baseline for both HPO and CASH (Brazdil et al., 2022). MeanMMR is the model-free acquisition baseline defined in Section 3.4.

Our models. We compare directly the hyperparameter-based and performance-based representa tions introduced in Section 3 on three surrogate models: Bayesian ridge regression, the ubiquitous Gaussian process regressor (Balandat et al., 2020), with automatic relevance determination (ARD) enabled to assign an independent lengthscale per input dimension (implementation details in $\mathsf { A p - }$ pendix B.6), and a Random Forest inspired by SMAC (Lindauer et al., 2022). Throughout the experiments, we use the suffixes Hyper and Perf to denote variants using hyperparameter-based and performance-based representations, respectively. Within each surrogate family, the two variants differ only in whether the surrogate receives $x _ { \lambda } ^ { \mathrm { h y p e r } } \ \mathrm { o r } \ x _ { \lambda } ^ { \mathrm { p e r f } }$ . We use explicit hyperparameter encodings for the hyperparameter-based variants and the full historical performance vector $P _ { \lambda }$ for the performance-based variants; implementation details are given in Appendix B.4.

Dataset. All experiments use the TabArena benchmark (Erickson et al., 2025) (see Appendix E), which suits our study by combining curated tabular classification and regression tasks with leaderboard results for many modern algorithms and numerous sampled configurations per algorithm, allowing us to construct both CASH and HPO benchmarks from a shared task-configuration performance matrix. Specifically, we construct two finite search spaces: a CASH benchmark over a multi-algorithm configuration set (see Appendix E.1) and an HPO benchmark over an XGBoostonly subset (see Appendix E.2). The candidates in each benchmark define Λ and, on prior tasks, the performance matrix P used by the performance-based methods. Detailed benchmark construction is described in Appendix B.1.

Table 2: Average normalized regret on the test split (×100) and average rank across tasks, over 200 evaluations amongst the discrete set of configurations for the CASH TabArena benchmark and the HPO TabArena XGBoost-only benchmark. Bold: best per metric per benchmark. Underline: best per model family per metric. <sup>∗</sup>: significantly better than others within the family (Wilcoxon signedrank, $p < 0 . 0 5 )$ . Time/task: single-threaded per-task time spent fitting the surrogate and evaluating the acquisition function over the 200 evaluations. <sup>‡</sup>: excludes the required one-time pretraining phase shared across tasks (ABLR 3.4m on CASH, 27s on HPO; RGPE 2.0m on HPO).
<table><tr><td colspan="2"></td><td colspan="3">CASH</td><td colspan="3">HPO</td></tr><tr><td>Model</td><td>Input</td><td>Regret↓</td><td>Rank↓</td><td>Time/task</td><td>Regret↓</td><td>Rank↓</td><td>Time/task</td></tr><tr><td rowspan="2">Linear</td><td>Perf.</td><td>12.19*</td><td>4.34*</td><td>2.7s</td><td>9.84*</td><td>6.14*</td><td>1.1s</td></tr><tr><td>Hyper.</td><td>14.59</td><td>5.95</td><td>24s</td><td>12.79</td><td>7.04</td><td>0.7s</td></tr><tr><td rowspan="2">Random Forest</td><td>Perf.</td><td>18.53*</td><td>5.83*</td><td>2.0m</td><td>11.29</td><td>6.51</td><td>1.8m</td></tr><tr><td>Hyper.</td><td>28.59</td><td>8.06</td><td>2.7m</td><td>13.46</td><td>6.59</td><td>53s</td></tr><tr><td rowspan="2">Gaussian Process</td><td>Perf.</td><td>23.73</td><td>6.99</td><td>7.3m</td><td>13.40</td><td>6.87</td><td>1.5m</td></tr><tr><td>Hyper.</td><td>20.25</td><td>5.93</td><td>1.8h</td><td>13.61</td><td>6.68</td><td>7.9m</td></tr><tr><td rowspan="3">ABLR Active-Testing MTGP</td><td>Hyper.</td><td>15.47</td><td>5.22</td><td>1.1s‡</td><td>9.02</td><td>5.97</td><td>0.6s</td></tr><tr><td>Perf.</td><td>16.54</td><td>5.37</td><td>0.9s</td><td>16.63</td><td>7.56</td><td>0.7s</td></tr><tr><td>Hyper.</td><td></td><td></td><td></td><td>13.52</td><td>6.98</td><td>10s</td></tr><tr><td>RGPE</td><td>Hyper.</td><td></td><td></td><td></td><td>10.57</td><td>6.18</td><td>9.3m‡</td></tr><tr><td>Random</td><td></td><td>19.45</td><td>6.61</td><td>0.5s</td><td>16.86</td><td>7.63</td><td>0.3s</td></tr><tr><td>Mean</td><td>Perf.</td><td>17.56</td><td>5.93</td><td>0.7s</td><td>25.03</td><td>9.37</td><td>0.3s</td></tr><tr><td>MeanMMR</td><td>Perf.</td><td>14.60</td><td>5.77</td><td>6.6s</td><td>14.92</td><td>7.47</td><td>0.5s</td></tr></table>

Metrics and Validation Scheme. We use five-fold cross-validation at the task level. Meta-training tasks define the performance matrix and any prior-task training, while held-out tasks are optimized independently as target HPO or CASH problems. Search uses validation-fold feedback only, and final regret is computed on separate test folds for the validation-selected configuration to avoid skewed results due to adaptive overfitting (Schroder et al., 2025). Further details are given in Appendix B.2.¨

## 5 RESULTS

On the high-dimensional CASH benchmark, the performance-based representation paired with a simple Bayesian ridge surrogate (Linear-Perf) attains the best average regret and rank overall, both in the regret curves (Figure 2) and at the full budget of 200 evaluations (Table 2). Concretely, ABLR needs 149 evaluations to reach the regret Linear-Perf reaches in 73, close to a 2x reduction in trials needed. Linear-Perf is also the fastest of the surrogate models to fit, averaging under three seconds per task against roughly two hours for GP-Hyper on the same benchmark. On the lowerdimensional HPO benchmark, the ABLR method that meta-learns a hyperparameter embedding network performs best, with Linear-Perf following at a close second.

Comparing the performance representation against the usual hyperparameter representation shows statistically significant improvements for the linear and random forest surrogates in the CASH benchmark, but no statistically significant or consistent improvement for the Gaussian Process: the benefit of historical performance as representation is not observed as strongly across all surrogate types.

Rankings of the static baselines that do not fit a surrogate model differ between benchmarks. On the XGBoost HPO setting, all model-based methods consistently achieve better regret than random search after the first few iterations, with mean ranking the worst strategy overall (25.03 versus 16.86 for random search). In the CASH setting, the hyperparameter-based random forest does not clearly improve upon random search, and the mean ranking gives a very strong initial selection of models until being surpassed after ten evaluations, and still beats random search (regret 17.56 versus 19.45). Active testing benefits from the strong initialization on CASH, and is surpassed by Linear-Perf after roughly twenty evaluations. The MeanMMR baseline improves upon mean ranking, such that it beats random search on HPO, and is competitive with non-metalearning methods for roughly the first fifty evaluations, although surpassed earlier by strong meta learners. Surprisingly, MeanMMR outperforms the dedicated meta-learning methods on CASH in terms of average regret.

![](images/90b00c9d22e681ec28472f6667d4480424989709a5dc6fd7a82c1f986ae82455.jpg)  
Figure 3: Average normalized regret over 200 configuration evaluations on the CASH benchmark vs. number of training tasks. Methods introduced in this work (the Perf variants and MeanMMR) are marked with an upward-pointing triangle (▲). Dashed horizontal lines show methods that do not make use of results on previous tasks. Shaded bands are 95% bootstrap CIs across tasks.

We measure how the methods depend on the number of tasks for which historical performance is known (Figure 3); as expected, average regret generally improves as the number of prior tasks increases. For CASH, Linear-Hyper is the strongest non-metalearning method whose average regret is not surpassed until at least 16 training tasks are used by Linear-Perf. In the HPO setting, ABLR is immediately competitive with Linear-Hyper and offers the best regret once results for at least one prior training task are available. Methods using mean ranking, including active testing, appear to benefit initially but stop improving after more than 16 tasks are added.

## 6 DISCUSSION & LIMITATIONS

The performance-based representation implements meta-learning well specifically in combination with a linear surrogate model, indicating that a configuration’s performance on the target task fits well to a linear combination of its performances on prior tasks: Linear-Perf effectively learns to represent the target task as a weighted average of prior tasks. That aligns with ideas behind related work such as the weighted ensembling over task models of RGPE. We performed an ablation of RGPE that replaces the base-GP predictions with the true performance columns of the prior tasks (RGPE-Oracle), which is still strong on HPO but not as competitive with Linear-Perf on CASH (see Appendix B.9). Freely determining weights per prior task via regression may be more effective and expressive than RGPE’s non-negative heuristic ranking agreement; a fixed-precision ablation confirms this is not simply due to adaptive precision estimation (see Appendix C.1). Consistent with this, the weights learned by Bayesian ridge regression tend to be both positive and negative, with roughly half negative on CASH (see Appendix C.3), likely necessary for a good fit on the high-dimensional problem.

Rankings of models differ between the CASH and HPO settings. In the lower-dimensional HPO case, hyperparameter methods all perform better than random search by a larger margin, and the hyperparameter embedding-based ABLR method is best overall. This can be explained by the manageable dimensionality, with only 11 hyperparameters whose effects to learn, versus 475 values after encoding for the high-dimensional CASH problem. This reflects the curse of dimensionality: more data points are needed to obtain the same predictive performance, and since only a subset of hyperparameters is active per selected algorithm, relevant data points per hyperparameter are further reduced. Architectures that exploit this structure efficiently, e.g., separate encoders per algorithm or surrogates designed for conditional, tree-structured search spaces (Jenatton et al., 2017), might have performed better; not pursuing that direction is a limitation of our baselines, but consistent with our motivation that large-scale CASH invites unwanted engineering complexity. One might argue this large-scale task is an unfair comparison, but it also reflects a realistic problem: the choices data scientists face with the full breadth of state-of-the-art tabular models on benchmark leaderboards such as TabArena (Erickson et al., 2025).

The model-free baseline results also support that meta-learning helps more on CASH than on HPO. Within one algorithm’s hyperparameter space the choice of configuration might depend more on the specific tasks, while across multiple algorithm types some might be clearly stronger or weaker. That might explain why historical performance gives larger advantages on CASH.

The strong initialisation of the mean best configuration on CASH, as used by active testing, could easily be applied to other methods as well, further strengthening for example Linear-Perf. This was not done for the base model set to keep the experiment controlled by only varying the input representation, and the other meta-learning baselines did not explicitly include this initialisation step in their algorithm.

MeanMMR benefits from the strong mean signal, while not suffering from the low performance that mean ranking has on HPO. The exploration term based on historical performance diversity bridges part of the gap between static ranking methods and exploration-driven Bayesian optimisation. The low performance of mean ranking on HPO might therefore also be caused by it suggesting lowdiversity configurations from a local optimum, suggesting portfolio methods might benefit from similar diversity measures. MMR also extends to surrogate models without a predictive posterior, such as the Two-Tower recommender model; we explored this variant but omit it from the main comparison as its results were competitive but not state-of-the-art (see Appendix B.10).

The biggest limitation of the performance-based methods is requiring a dense matrix with historical performance of every candidate configuration on every prior task. The cost of collecting the matrix scales with both dimensions, number of configurations and number of tasks. Given this initial cost, the method is most applicable to popular search spaces where a centralized matrix can be maintained, shared and reused many times for different applications. For niche domains and architectures where a search space is only used a handful of times, benefits of faster search likely do not outweigh the initial cost of collecting the matrix. Furthermore, unlike factorization methods such as PMF (Fusi et al., 2018), our method would require imputation to support sparse matrices.

Multi-task benchmarks give a more optimistic perspective on this data requirement. In our scalability experiment on CASH, roughly 16 prior tasks were needed before performance representation models outperformed alternatives, while the full TabArena leaderboard (Erickson et al., 2025) already contains results for 51 tasks. Matrices built from benchmark leaderboards are also likely to stay up to date, since authors of new methods contribute the required data whenever they report results on the benchmark.

Scaling of possible performance with respect to the number of configurations is clear: A performance-based method can at best suggest the optimal configuration included in the matrix, when the truly optimal configuration may in reality lie somewhere else in the continuous hyperparameter search space. Future work should examine the continuous case.

Because it requires no hyperparameter encoding at all, our representation also avoids having to define a single unified search space across heterogeneous algorithms, which is especially useful when aggregating results across many different model types, for example from platforms such as OpenML (Bischl et al., 2025).

## 7 CONCLUSION

In summary, using historical performance as a configuration representation is an effective and simple approach to meta-learning. Paired with a linear Bayesian ridge regression model, it outperforms dedicated meta-learning baselines on the high-dimensional CASH benchmark, reaching the same tuned performance in roughly half the number of trials, and remains a competitive second on singlemodel HPO. Simplicity is key: the well-performing Bayesian ridge regression only models linear relationships between tasks and requires no pre-training, hyperparameter preprocessing, or tuning. Using performance representations avoids complexity in combining search space encodings across models with different hyperparameters. For a completely model-free alternative, ranking by the mean and diversity of the performance vectors (MeanMMR) is stronger than pure average ranking and competitive with limited target-task observations.

## AI USE STATEMENT

We have not used generative AI tools for any of the following tasks: generating synthetic data sets, developing theoretical models or conceptual frameworks, formulating mathematical claims, providing critical ingredients for proving mathematical claims, assisting in the writing of proofs, proposing or refining hypotheses, designing or providing feedback on research methodology or experiments, implementing methods, assisting with translation, cleaning and reformatting data sets, supporting qualitative and thematic data analysis, and interpreting results. We did use generative AI tools for sourcing and summarising existing literature, for editing software code, limited to refactoring, visualisation scripts, and scripts that aggregate experiment outputs into tables and run the reported significance tests, for creating and modifying figures, for suggesting a structure for this paper, for LAT<sub>E</sub>X formatting support, and for proofreading to improve readability. We have reviewed all AI-assisted work: all reported results were produced and verified by the authors’ own code, AI-suggested references were checked against the primary sources, and all AI-assisted text was read and edited by the authors. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

All reported experiments are simulated by lookups into the public TabArena leaderboard results (Erickson et al., 2025), so no base model has to be retrained to reproduce them. Appendix B.1 describes how the CASH and HPO search spaces are constructed from that data, Appendix E lists the exact base model set and the XGBoost search space, and Appendices B.2 and B.3 define the task-level cross-validation protocol, the validation-test split, and the regret normalization. Implementation details for every surrogate and baseline are given in Appendices B.4–B.10, with the final hyperparameters and random seeds collected in Table 3, and the hardware, software versions, and total compute are reported in Appendix B.11. The source code, including the configuration files that define each reported run, is available at https://github.com/EWitting/simple-metalearning.

## REFERENCES

Maximilian Balandat, Brian Karrer, Daniel R. Jiang, Samuel Daulton, Benjamin Letham, Andrew Gordon Wilson, and Eytan Bakshy. BoTorch: A framework for efficient Monte-Carlo Bayesian optimization. In Advances in Neural Information Processing Systems, volume 33, 2020.

James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(10):281–305, 2012.

Bernd Bischl, Giuseppe Casalicchio, Taniya Das, Matthias Feurer, Sebastian Fischer, Pieter Gijsbers, Subhaditya Mukherjee, Andreas C. Muller, L¨ aszl´ o N´ emeth, Luis Oala, Lennart Purucker,´ Sahithya Ravi, Jan N. van Rijn, Prabhant Singh, Joaquin Vanschoren, Jos van der Velde, and Marcel Wever. OpenML: Insights from 10 years and more than a thousand papers. Patterns, 6(7): 101317, 2025.

Pavel Brazdil, Jan N. van Rijn, Carlos Soares, and Joaquin Vanschoren. Metalearning: Applications to Automated Machine Learning and Data Mining. Cognitive Technologies. Springer, 2022. doi: 10.1007/978-3-030-67024-5.

Eric Brochu, Vlad M. Cora, and Nando de Freitas. A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning. arXiv preprint arXiv:1012.2599, 2010.

Jaime Carbonell and Jade Goldstein. The use of MMR, diversity-based reranking for reordering documents and producing summaries. In Proceedings of the 21st annual international ACM SIGIR conference on Research and development in information retrieval, pp. 335–336, 1998.

Wanyu Chen, Pengjie Ren, Fei Cai, Fei Sun, and Maarten De Rijke. Multi-interest diversification for end-to-end sequential recommendation. ACM Transactions on Information Systems (TOIS), 40(1):1–30, 2021.

Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for YouTube recommendations. In Proceedings of the 10th ACM conference on recommender systems, pp. 191–198, 2016.

Tommaso Di Noia, Jessica Rosati, Paolo Tomeo, and Eugenio Di Sciascio. Adaptive multi-attribute diversity for recommender systems. Information Sciences, 382:234–253, 2017.

Juseon Do, Jaesung Hwang, Jingun Kwon, Hidetaka Kamigaito, and Manabu Okumura. Considering length diversity in retrieval-augmented summarization. In Findings of the Association for Computational Linguistics: NAACL 2025, pp. 2489–2500, 2025.

Nick Erickson, Jonas Mueller, Alexander Shirkov, Hang Zhang, Pedro Larroy, Mu Li, and Alexander Smola. AutoGluon-Tabular: Robust and accurate AutoML for structured data. arXiv preprint arXiv:2003.06505, 2020.

Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmuller, Prateek Desai, David Sali-¨ nas, and Frank Hutter. TabArena: A living benchmark for machine learning on tabular data. Advances in Neural Information Processing Systems, 38, 2025.

Stefan Falkner, Aaron Klein, and Frank Hutter. BOHB: Robust and efficient hyperparameter optimization at scale. In International conference on machine learning, pp. 1437–1446. PMLR, 2018.

Matthias Feurer, Aaron Klein, Katharina Eggensperger, Jost Tobias Springenberg, Manuel Blum, and Frank Hutter. Efficient and robust automated machine learning. In Advances in Neural Information Processing Systems, volume 28, 2015.

Matthias Feurer, Benjamin Letham, Frank Hutter, and Eytan Bakshy. Practical transfer learning for Bayesian optimization. arXiv preprint arXiv:1802.02219, 2018.

Matthias Feurer, Katharina Eggensperger, Stefan Falkner, Marius Lindauer, and Frank Hutter. Autosklearn 2.0: Hands-free AutoML via meta-learning. Journal of Machine Learning Research, 23 (261):1–61, 2022.

Nicolo Fusi, Rishit Sheth, and Melih Elibol. Probabilistic matrix factorization for automated machine learning. Advances in neural information processing systems, 31, 2018.

Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management, pp. 2333–2338, 2013.

Frank Hutter, Holger H Hoos, and Kevin Leyton-Brown. Sequential model-based optimization for general algorithm configuration. In International conference on learning and intelligent optimiza tion, pp. 507–523. Springer, 2011.

Frank Hutter, Holger Hoos, and Kevin Leyton-Brown. An efficient approach for assessing hyperparameter importance. In International conference on machine learning, pp. 754–762. PMLR, 2014.

Frank Hutter, Lars Kotthoff, and Joaquin Vanschoren. Automated machine learning: methods, systems, challenges. Springer, 2019.

Rodolphe Jenatton, Cedric Archambeau, Javier Gonz´ alez, and Matthias Seeger. Bayesian optimiza-´ tion with tree-structured dependencies. In Proceedings of the 34th International Conference on Machine Learning (ICML), volume 70 of PMLR, pp. 1655–1664, 2017.

Hadi Samer Jomaa, Sebastian Pineda Arango, Lars Schmidt-Thieme, and Josif Grabocka. Transfer learning for Bayesian HPO with end-to-end landmark meta-features. In Fifth Workshop on Meta-Learning at the Conference on Neural Information Processing Systems, 2021.

Donald R Jones, Matthias Schonlau, and William J Welch. Efficient global optimization of expensive black-box functions. Journal ofGlobal optimization, 13(4):455–492, 1998.

Yehuda Koren, Robert Bell, and Chris Volinsky. Matrix factorization techniques for recommender systems. Computer, 42(8):30–37, 2009.

Anton Korikov and Scott Sanner. ReBOL: Retrieval via Bayesian optimization with batched LLM relevance observations and query reformulation. arXiv preprint arXiv:2603.20513, 2026.

Lars Kotthoff, Chris Thornton, Holger H Hoos, Frank Hutter, and Kevin Leyton-Brown. Auto-WEKA 2.0: Automatic model selection and hyperparameter optimization in WEKA. Journal of Machine Learning Research, 18(25):1–5, 2017.

Rui Leite, Pavel Brazdil, and Joaquin Vanschoren. Selecting classification algorithms with active testing. In Machine Learning and Data Mining in Pattern Recognition (MLDM), pp. 117–131. Springer, 2012.

Marius Lindauer, Katharina Eggensperger, Matthias Feurer, Andre Biedenkapp, Difan Deng, Car-´ olin Benjamins, Tim Ruhkopf, Rene Sass, and Frank Hutter. SMAC3: A versatile Bayesian´ optimization package for hyperparameter optimization. Journal of Machine Learning Research, 23(54):1–9, 2022.

Marco Loog, Tom Viering, Alexander Mey, Jesse H Krijthe, and David MJ Tax. A brief prehistory of double descent. Proceedings ofthe National Academy ofSciences, 117(20):10625–10626, 2020.

David JC MacKay. Bayesian interpolation. Neural computation, 4(3):415–447, 1992.

Elvis Nava, Mojmir Mutny, and Andreas Krause. Diversified sampling for batched Bayesian optimization with determinantal point processes. In International Conference on Artificial Intelligence and Statistics, pp. 7031–7054. PMLR, 2022.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal ofMachine Learning Research, 12:2825–2830, 2011.

Valerio Perrone, Rodolphe Jenatton, Matthias W. Seeger, and Cedric Archambeau. Scalable hyper-´ parameter transfer learning. In Advances in Neural Information Processing Systems, volume 31, 2018.

Sebastian Pineda Arango, Fabio Ferreira, Arlind Kadra, Frank Hutter, and Josif Grabocka. Quicktune: Quickly learning which pretrained model to finetune and how. In International Conference on Learning Representations, 2024.

Philipp Probst, Anne-Laure Boulesteix, and Bernd Bischl. Tunability: Importance of hyperparameters of machine learning algorithms. Journal of Machine Learning Research, 20(53):1–32, 2019.

Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian Processesfor Machine Learning. MIT Press, 2006.

Shaina Raza, Mizanur Rahman, Safiullah Kamawal, Armin Toroghi, Ananya Raval, Farshad Navah, and Amirmohammad Kazemeini. A comprehensive review of recommender systems: Transitioning from theory to practice. Computer Science Review, 59:100849, 2026.

Tim Ruhkopf, Aditya Mohan, Difan Deng, Alexander Tornede, Frank Hutter, and Marius Lindauer. MASIF: Meta-learned algorithm selection using implicit fidelity information. Transactions on Machine Learning Research, 2023.

Sietse Schroder, Mitra Baratchi, and Jan N. van Rijn. Overfitting in combined algorithm selection¨ and hyperparameter optimization. In Advances in Intelligent Data Analysis XXIII (IDA), pp. 181– 194. Springer, 2025. doi: 10.1007/978-3-031-91398-3 14.

Yu Shen, Yupeng Lu, Yang Li, Yaofeng Tu, Wentao Zhang, and Bin Cui. DivBO: Diversity-aware CASH for ensemble learning. Advances in Neural Information Processing Systems, 35:2958– 2971, 2022.

Jasper Snoek, Hugo Larochelle, and Ryan P. Adams. Practical Bayesian optimization of machine learning algorithms. In Advances in Neural Information Processing Systems, volume 25, 2012.

Kevin Swersky, Jasper Snoek, and Ryan P Adams. Multi-task Bayesian optimization. Advances in neural information processing systems, 26, 2013.

Chris Thornton, Frank Hutter, Holger H Hoos, and Kevin Leyton-Brown. Auto-WEKA: Combined selection and hyperparameter optimization of classification algorithms. In Proceedings ofthe 19th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 847–855, 2013.

Michael E. Tipping. Sparse Bayesian learning and the relevance vector machine. Journal ofMachine Learning Research, 1:211–244, 2001.

Jan N Van Rijn and Frank Hutter. Hyperparameter importance across datasets. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 2367–2376, 2018.

Saul Vargas and Pablo Castells. Rank and relevance in novelty and diversity metrics for recom-´ mender systems. In Proceedings of the fifth ACM conference on Recommender systems, pp. 109– 116, 2011.

Tom Viering and Marco Loog. The shape of learning curves: a review. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(6):7799–7819, 2022.

Zi Wang, George E Dahl, Kevin Swersky, Chansoo Lee, Zachary Nado, Justin Gilmer, Jasper Snoek, and Zoubin Ghahramani. Pre-trained Gaussian processes for Bayesian optimization. Journal of Machine Learning Research, 25(212):1–83, 2024.

Ying Wei, Peilin Zhao, Huaxiu Yao, and Junzhou Huang. Transferable neural processes for hyperparameter optimization. arXiv preprint arXiv:1909.03209, 2019.

Martin Wistuba and Josif Grabocka. Few-shot Bayesian optimization with deep kernel surrogates. In International Conference on Learning Representations, 2021.

Martin Wistuba, Nicolas Schilling, and Lars Schmidt-Thieme. Scalable Gaussian process-based transfer surrogates for hyperparameter optimization. Machine Learning, 107(1):43–78, 2018.

Chengrun Yang, Yuji Akimoto, Dae Won Kim, and Madeleine Udell. OBOE: Collaborative filtering for AutoML model selection. In Proceedings ofthe 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 1173–1183, 2019.

Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, and Ed Chi. Sampling-bias-corrected neural modeling for large corpus item recommendations. In Proceedings of the 13th ACM conference on recommender systems, pp. 269–277, 2019.

## APPENDIX

This appendix provides the implementation and evaluation-protocol details, additional figures, and extended discussion referenced from the main paper.

## A HPO BENCHMARK FIGURES

Plots showing the test normalized regret against number of evaluations or number of observed prior tasks on the HPO benchmark are shown in Figure 4 and Figure 5.

![](images/8e605a1a0802b88f31c6be12cea1e6139571cb8fc65b7fd5a842238212f9e42b.jpg)

![](images/1c7c96639c399dabd2b1aba3a241842ea7356a2118d77802a6c3e8b592d52f70.jpg)

![](images/ed30f6962d5cda392af40ede45f57fcf8c2618047b9d221f3dbcddda4d9900aa.jpg)

![](images/081799a0b90ecc77ea5a313be301aaf0ab1340320464b64882d721a5640beb28.jpg)

![](images/86cb0155c50575dac21089e17b6a4f00d570773b2137b01aa9cda54a57a7a794.jpg)  
Random Mean MeanMMR Linear (Perf) Linear (Hyper) RF (Perf) RF (Hyper) GP (Perf) GP (Hyper) ABLR Active Testing MTGP RGPE

Figure 4: Test normalized regret per model family and overall rank vs. number of evaluations on the HPO benchmark (TabArena XGBoost-only). Methods introduced in this work (the Perf variants and MeanMMR) are marked with an upward-pointing triangle (▲). Shaded bands are 95% bootstrap CIs across tasks.  
![](images/cc601c92e5165116d88ee2fc4161474120e875458ac547f2600b0bd0362939a5.jpg)  
Figure 5: Average normalized regret over 200 model evaluations on the HPO benchmark vs. number of previous tasks observed, grouped by model family. Methods introduced in this work (the Perf variants and MeanMMR) are marked with an upward-pointing triangle (▲). Dashed horizontal lines show methods that do not make use of results on previous tasks. Shaded bands are 95% bootstrap CIs across tasks.

## B EXPERIMENTAL SETUP DETAILS

This appendix provides the implementation details behind the experimental protocol, including benchmark construction, encoding choices, acquisition rules, and model settings.

## B.1 BENCHMARK CONSTRUCTION

The public TabArena leaderboard dataset contains cross-validated results for evaluating each listed algorithm on each task. For most algorithms this includes not only the default hyperparameter setting, but also 200 randomly selected hyperparameter configurations. From these data we construct the performance matrix P used in our meta-learning models and representations. Using all base algorithms and tasks, the matrix contains 3623 configurations across 23 algorithms and 51 tasks (see Table 6). The algorithm set does not include large ensembles like AutoGluon, as they could have a disproportionate impact through their high performance, while our methods and evaluation framework do not penalize them for disproportionately high training cost. For the HPO benchmark, we use the 201 XGBoost configurations, consisting of one default configuration and 200 random configurations (see Table 7). The choice of XGBoost is arbitrary; any competitive method with a sufficiently large list of configurations could have been used.

These 3623 configurations for CASH and 201 configurations for HPO fully define the search space Λ in each benchmark. Such a finite set is required for defining the matrix used by performance-based methods. The hyperparameter-based methods do not have that restriction in principle and could suggest configurations from the underlying continuous search space. However, using the same finite search space for all variants isolates the effect of changing the representation. Additionally, staying within this set makes large-scale experimentation possible by simulating configuration evaluations with lookups in the precomputed benchmark results.

## B.2 EVALUATION PROTOCOL

We employ five-fold cross-validation at the task level, each fold resulting in a meta-training set of approximately 40 training tasks and 10 test tasks. The meta-training tasks are used for the performance matrix and representations, and for training the Two-Tower model, though not all methods make use of them. Within the meta-test set, HPO or CASH is run on each test task independently. When a model is updated on one target task, it is reverted to its state after training on the meta-training tasks before continuing with the next target task. Metrics are aggregated using the mean across all out-of-fold tasks, and confidence intervals are computed using bootstrap resampling across tasks.

The same procedure is used when plotting performance with respect to training tasks, with the addition of randomly subsampling tasks within each training fold. There might be strategies for selecting an optimal diverse subset of tasks, we choose randomly for simplicity. The random subsample is without replacement, and seeded such that it is independent across folds but consistent amongst methods.

## B.3 REGRET AND NORMALIZATION

As HPO and CASH are iterative processes, we measure regret as a curve over the number of configurations that have been evaluated. As a summary statistic, regret is averaged over this curve. Often, regret is reported as the difference between the optimal value known for an objective function and the optimal value among the configurations selected so far by a search strategy. However, this does not take adaptive overfitting into account, where the selection of the best sofar may not translate to other test sets (Schroder et al., 2025). To prevent adaptive overfitting from positively skewing the¨ results, we split validation and test folds. The TabArena leaderboard data reports scores for individ ual folds per task; we set apart half of the folds to form the validation split and the other half as the test split. The validation split defines $\mathcal { P } _ { \mathrm { v a l } }$ , which provides the matrix $P ,$ all target-task observations seen during search, and the selection of the best so far configuration. The held-out split defines $\mathcal { P } _ { \mathrm { t e s t } }$ , which is used only for final reporting. The final regret calculation therefore evaluates the test performance of the configuration selected by validation performance:

$$
r _ { n } ( t ) = \operatorname* { m a x } _ { \lambda \in \Lambda } \mathcal { P } _ { \mathrm { t e s t } } ( t , \lambda ) - \mathcal { P } _ { \mathrm { t e s t } } \left( t , \operatorname { a r g m a x } _ { \lambda \in \Lambda _ { \mathrm { o b s } } ^ { ( n ) } } \mathcal { P } _ { \mathrm { v a l } } ( t , \lambda ) \right)\tag{2}
$$

where n indexes the number of evaluated models, $\mathcal { P } _ { \mathrm { t e s t } }$ is performance on the test folds and $\mathcal { P } _ { \mathrm { v a l } }$ on the validation folds. In practice, we did observe some diverging of validation and test regret, shown in Section D).

Besides adaptive overfitting, we aim to eliminate another type of possible bias observed in related work: normalization leakage. As some tasks may have consistently higher or lower performance across configurations, it is common to report normalized regret. In the case of TabArena this is scaled such that the best score for a task is 1 and the median is 0. This is justified when comparing AutoML methods from an outside perspective, but normalized values should not be available during the search process, as they contain information about the optimal achievable score.

To prevent leakage but keep performance on similar scales across tasks, we use two types of normalization. During search, the performance matrix P and observed target-task values are scaled by the dummy score. The dummy score is the error achieved when constantly predicting the mean for regression tasks or the majority class for classification, which we consider a zero-cost value in practice that all methods would have access to. When reporting normalized regret in graphs and tables, we use the aforementioned scaling that maps the best performance overall for a task to 1 and the median to 0.

Similarly, while the TabArena leaderboard imputes the few combinations where an algorithm is not capable of a specific task type with the score of a random forest, we score these combinations using the near zero-cost dummy score instead.

## B.4 REPRESENTATION ENCODING

For the HPO benchmark, $\phi ( \lambda )$ contains the hyperparameters of the single algorithm under consideration. For the CASH benchmark, where $\lambda = ( A , h )$ , it concatenates the hyperparameter lists of all algorithms with a one-hot algorithm indicator. Categorical choices are one-hot encoded and inactive or non-applicable hyperparameters are encoded consistently across all candidates in Λ. For the vector-based performance variants, namely Linear-Perf, RF-Perf, GP-Perf, and MeanMMR, we use the full historical performance vector $P _ { \lambda }$ directly and do not apply dimensionality reduction. This differs from OBOE (Yang et al., 2019), which uses low-rank PCA-based representations; in early experiments, our method improved as the representation approached the full vector.

## B.5 ACQUISITION RULES

Expected Improvement is used for surrogate models that provide a predictive mean and uncertainty estimate, while MMR-based acquisition is defined in the main paper. For random forests, the predictive mean and standard deviation used by EI are estimated from the tree ensemble (Lindauer et al., 2022). Given predictive mean $\mu ( \lambda )$ ), standard deviation $\sigma ( \lambda )$ , and best observed performance ${ \mathcal { P } } ^ { * }$ EI is:

$$
\operatorname { E I } ( \lambda ) = \left( \mu ( \lambda ) - { \mathcal { P } } ^ { * } \right) \Phi ( Z ) + \sigma ( \lambda ) \varphi ( Z ) ,\tag{3}
$$

$$
Z = { \frac { \mu ( \lambda ) - { \mathcal { P } } ^ { * } } { \sigma ( \lambda ) } } ,\tag{4}
$$

where $\Phi$ and $\varphi$ are the standard normal CDF and PDF, respectively.

For MeanMMR, the general MMR rule from the main paper is instantiated with the historical mean as the relevance score and cosine similarity on raw performance vectors as the diversity term:

$$
\mathrm { M e a n M M R } ( \lambda ) = \alpha \cdot \frac { \displaystyle \sum _ { s \in T _ { \mathrm { t r a i n } } } P _ { s , \lambda } } { | T _ { \mathrm { t r a i n } } | } - ( 1 - \alpha ) \cdot \operatorname* { m a x } _ { \lambda ^ { \prime } \in \Lambda _ { \mathrm { o b s } } } \cos ( P _ { \lambda } , P _ { \lambda ^ { \prime } } )\tag{5}
$$

Because the first term does not depend on target-task observations, the historical mean ranking is fixed, and only the diversity term updates as configurations are evaluated on the target task.

## B.6 GAUSSIAN PROCESS IMPLEMENTATION

Gaussian processes (GP’s) (Rasmussen & Williams, 2006) are the classical probabilistic surrogate in BO and provide a predictive posterior for EI. We include a GP surrogate with both representations. The Gaussian process surrogate is implemented using BoTorch’s SingleTaskGP (Balandat et al., 2020) with a Matern-´ $\cdot \frac { 5 } { 2 }$ kernel (Rasmussen & Williams, 2006). Automatic Relevance Determination (ARD) is enabled, assigning an independent lengthscale to each input dimension so the model can identify which hyperparameters are most predictive. Gamma priors are placed on the lengthscales (Gamma(3.0, 6.0)) and output scale (Gamma(2.0, 0.15)) to regularize the fit when few observations are available. Inputs are min-max normalized to [0, 1] and outputs are standardised per task. Quantile scaling was attempted, which should automatically redistribute hyperparameter values that were originally sampled by TabArena authors at log-scale, but did not lead to an improvement. Kernel hyperparameters are optimized by maximising the marginal log-likelihood via L-BFGS-B at each BO step, warm-started from the previous iteration’s solution.

BoTorch’s implementation is used across experiments and variants for consistency. Its performance relative to scikit-learn’s GaussianProcessRegressor (Pedregosa et al., 2011) with the default RBF kernel varies by benchmark: on the high-dimensional CASH benchmark, the simpler

ARD-free scikit-learn kernel achieves lower regret for both representations (see Table 2 in the main paper), whereas BoTorch performs better on the lower-dimensional HPO benchmark. Neither variant surpasses Linear-Perf.

## B.7 RANDOM FOREST IMPLEMENTATION

Both RF-Hyper and RF-Perf use scikit-learn’s RandomForestRegressor (Pedregosa et al., 2011) with n estimators=100 and random state=42. RF-Hyper additionally sets min samples leaf=3, following SMAC’s convention (Lindauer et al., 2022) of preventing overly specific single-observation splits early in the search, when few configurations have been evaluated. RF-Perf uses scikit-learn’s default min samples leaf=1 and, like GP-Perf and Linear-Perf (Section B.4), operates on the full performance vector without dimensionality reduction. Predictive mean and standard deviation for EI are estimated from the tree ensemble as described in Section B.5.

## B.8 SUMMARY OF FINAL HYPERPARAMETERS

For reproducibility, Table 3 lists the final hyperparameter settings used for every surrogate model in the main experiments. Settings not listed follow scikit-learn or BoTorch defaults. Two-Tower settings were tuned per benchmark and are listed separately in Table 4. Every model with a stochastic fitting procedure (Random Forest, GP restarts, Two-Tower initialization/training, and the metalearning baselines) is seeded with random state=42; Bayesian ridge regression has no internal source of randomness.

Table 3: Final hyperparameters used for every model in the main experiments. Exact configuration files are included in the code release.
<table><tr><td>Model</td><td>Config key</td><td>Final hyperparameters</td></tr><tr><td>Linear-Hyper /</td><td>ridge_ei/</td><td>BayesianRidge, scikit-learn defaults  $( \alpha _ { 1 } = \alpha _ { 2 } = \lambda _ { 1 } = \lambda _ { 2 } = 1 0 ^ { - 6 } ,$  max_iter=300, tol=1e-3); Linear-Perf fits without intercept</td></tr><tr><td>Linear-Perf</td><td>perf_vector_ridge_ei</td><td>(Section C.1)</td></tr><tr><td>RF-Hyper</td><td>rf_ei</td><td>n_estimators=100,min_samples_leaf=3,random_state=42 n_estimators=100,min_samples_leaf=1 (default),</td></tr><tr><td>RF-Perf</td><td>perf_vector_rf_ei</td><td>random_state=42 BoTorch Sing1eTaskGP, Matérn-5/2 ARD kernel, lengthscale prior</td></tr><tr><td>GP-Hyper/GP-Perf botorch-gp-ei/</td><td>botorch_gp-perf-{cash,hpo}</td><td>Gamma(3.0, 6.0), outputscale prior Gamma(2.0, 0.15), L-BFGS-B MLL optimization (Section B.6)</td></tr><tr><td>TwoTower-Hyper / TwoTower-Perf</td><td>two_tower_hyper_*/ two_tower-perf_*</td><td>random_st ate=42; remaining settings tuned per benchmark by random search, see Table 4</td></tr><tr><td>ABLR</td><td>ablr</td><td>feature dim 50, hidden layers [50, 50], 300 epochs, lr=0.01, random_state=42</td></tr><tr><td>Active-Testing RGPE</td><td>active_testing</td><td>similarity sharpness=1.0 256 bootstrap samples, 5 target CV folds, base-variance weighting enabled,</td></tr><tr><td></td><td>rgpe</td><td>weight-dilution prevention enabled, random_state=42</td></tr><tr><td>RGPE-Oracle TST-R</td><td>rgpe_oracle</td><td>as RGPE, with base-GP predictions replaced by true performance columns bandwidth=0.5, base-variance weighting enabled, random_state=42</td></tr><tr><td>MTGP</td><td>tstr_discrete</td><td>6 auxiliary tasks, 40 (HPO) / 48 (CASH) context flows, refit factor=3.0,</td></tr><tr><td></td><td>mtgp-{hpo, cash}</td><td>random_state=42</td></tr></table>

## B.9 COST OF GP-BASED TRANSFER AND THE RGPE-ORACLE ABLATION

Results of the GP-based meta-learning baselines MTGP (Swersky et al., 2013) and RGPE (Feurer et al., 2018) are reported on the HPO benchmark only. Both fit exact Gaussian processes, whose cost grows cubically with the number of observations n. RGPE maintains one base GP for every prior task, scaling with $t n ^ { 3 }$ for t tasks and n configurations per task. MTGP includes all observations in one Gaussian process, thus scaling even faster following $( t n ) ^ { 3 }$ . In the experimental setup for HPO $( n = 2 0 1 )$ this is affordable, on CASH $( n = 3 6 2 3 )$ it is not. For both cases, $t = 4 0 \mathrm { o r } t = 4 1$ depending on the fold.

To quantify this, we timed BoTorch’s SingleTaskGP fit on the real CASH hyperparameter features. Single-run fit times were roughly 48, 249, and 1029 seconds at $n = 6 0 0 , 9 0 0 .$ , and 1200 respectively, that roughly matches the $\dot { O ( n ^ { 3 } ) }$ cost and confirms that BoTorch does not fall back to a sub-cubic solver in this regime. Extrapolating to the full CASH set, a single base GP fit at $n = 3 6 2 3$ is on the order of six hours, so RGPE’s roughly forty base GPs per fold amount to several days of pretraining per fold, before any target-task search. These are single measurements and intended only as orders of magnitude, but the gap to ABLR’s 3.4-minute pretraining, or Linear-Perf’s none, is large enough that we exclude the GP-based methods from CASH rather than report downsampled results.

MTGP remains feasible on HPO only after subsampling: each fit conditions on a random subset of 40 context configurations drawn from 6 auxiliary tasks, without which the joint multi-task GP over context and target observations does not fit the HPO time budget. On CASH the fit would scale even more extremely than the several days of pretraining, given that it includes all tasks in one model.

To separate the cost of the GP fitting from the value of the ranking-transfer idea, we also evaluate RGPE-Oracle, a GP-free ablation that replaces each base-GP’s predictions with the true performance column of the corresponding prior task. This removes the cubic cost and lets the method run on CASH. On HPO, RGPE-Oracle reaches an average normalized regret of 10.00, competitive with the strongest methods and slightly ahead of the full GP-based RGPE (10.57), which indicates that the GP machinery adds little there. On CASH, where it can run but the full RGPE cannot, it reaches only 17.58, well behind Linear-Perf (12.19). This supports the interpretation in the main paper’s Discussion section that the non-negative ranking-agreement weighting over prior tasks is more restrictive than the signed linear combination fit by Linear-Perf.

## B.10 TWO-TOWER IMPLEMENTATION AND TUNING

We use a Two-Tower surrogate as a collaborative-filtering-style meta-learning model. Let $z _ { t }$ denote the embedding of task t and $z _ { \lambda }$ the embedding of configuration λ. The predicted performance of configuration λ on task t is given by their dot product, $\hat { \mathcal { P } } ( t , \lambda ) = z _ { \lambda } ^ { \top } z _ { t }$ (Huang et al., 2013; Raza et al., 2026). It can learn across tasks when trained on the meta-training tasks, even when its configuration tower receives $x _ { \lambda } ^ { \mathrm { h y p e r } }$ . The two representation choices differ in how $z _ { \lambda }$ is obtained: Two-Tower-Hyper maps $x _ { \lambda } ^ { \mathrm { h y p e r } } \ \mathrm { t o } \ z _ { \lambda }$ through a learnable encoder, whereas Two-Tower-Perf uses no encoder and instead learns one free configuration embedding per column of the historical performance matrix by fitting $P _ { t , \lambda } \approx z _ { t } ^ { \top } z _ { \lambda }$ . This makes Two-Tower-Perf closely related to matrix factorization. Since the Two-Tower model does not produce a predictive posterior, EI is unavailable, and we instead use the MMR acquisition defined in the main paper with relevance score $\hat { \mathcal { P } } ( t , \lambda ) = z _ { \lambda } ^ { \top } z _ { t }$ and $z _ { \lambda }$ the learned configuration embedding. The Two-Tower model offers a more nuanced comparison, being able to meta-learn using both hyperparameter or performance representations.

For Two-Tower-Hyper, the configuration tower is a multilayer perceptron that maps $x _ { \lambda } ^ { \mathrm { h y p e r } } \mathrm { t o } z _ { \lambda }$ We do not concatenate separate learned configuration latents to this input, so the variant remains purely hyperparameter-based. The task tower is kept as an identity transformation from learned task latents $z _ { t } .$ For Two-Tower-Perf, both towers are identity transformations over learned latents $z _ { \lambda }$ and $z _ { t } ,$ making it equivalent to a matrix-factorization-style model trained with the same procedure as Two-Tower-Hyper. During target-task adaptation, only the task-side parameters are updated. MMR is used as acquisition rule instead of expected improvement due to the lack of posterior distribution over the output of the architecture.

For a fair comparison, it is valuable to also tune the top-level methods themselves. We expected this to be most essential for the Two-Tower model, as a deep learning system is sensitive to settings such as learning rate and the dimensions of the multilayer perceptron. Both Two-Tower variants were optimized through fifty iterations of random search on the first of the five folds, optimizing average normalized test regret. This introduces a risk of overfitting and positively skewing the results for that model. We consider this acceptable for the present comparison because the main intention is to compare variants within the same model family, both variants are treated the same, and the tuning fold accounts for only a fifth of the final score. Alternatives such as nested cross-validation or a clean split with held-out tuning tasks would have required downscaling all experiments or accepting higher variance due to the smaller number of remaining tasks after holdout.

The shared search space covers three hyperparameters: embedding dimension sampled from {8, 16, 32}, learning rate sampled log-uniformly from [0.005, 0.1], and batch size sampled from {256, 512, 1024, 2048}. Two-Tower-Hyper additionally searches over the hyperparameter-tower MLP architecture, with hidden-layer options {[16], [32], [64], [128], [16,16], [32,32], [64,64], [128,128]}, while Two-Tower-Perf has no MLP in the configuration tower (embeddings feed directly into the dot product). The optimal hyperparameters found per benchmark are listed in Table 4.

Table 4: Best hyperparameters found by random search for each Two-Tower variant and benchmark.
<table><tr><td>Variant</td><td>Benchmark</td><td>Emb. dim</td><td>LR</td><td>Batch size</td><td>Hyper. tower</td></tr><tr><td>Two-Tower-Hyper</td><td>CASH</td><td>8</td><td>0.007</td><td>1024</td><td>[64, 64]</td></tr><tr><td>Two-Tower-Hyper</td><td>HPO</td><td>8</td><td>0.007</td><td>256</td><td>[16]</td></tr><tr><td>Two-Tower-Perf</td><td>CASH</td><td>8</td><td>0.005</td><td>1024</td><td></td></tr><tr><td>Two-Tower-Perf</td><td>HPO</td><td>8</td><td>0.020</td><td>512</td><td></td></tr></table>

Figure 6 shows the test regret of both Two-Tower variants against MeanMMR and Linear-Perf on each benchmark. Performance on the HPO benchmark after tuning was almost the exact same between Two-Tower-Perf and Two-Tower-Hyper. This could suggest that training on the entire meta-dataset is sufficient for learning to map hyperparameter vectors to algorithm performance, providing evidence against claims that that mapping would be too complex of a task. However, it is also possible that Two-Tower-Perf does not see an improvement because of an unrelated bottleneck in the architecture that affects both variants. Neither variant clearly improves upon the baseline of MeanMMR, which should be reproducible in theory by a Two-Tower-Perf model with embedding size one where the configuration embedding value is the mean of the configuration. What that suggests is that while the model may be expressive enough, theoretical performance is lost as the training process does not find the optimal solution in practice. Increasing the number of epochs from 500 to 5000 did not yield a clear improvement. The Two-Tower variants were also observed to be highly sensitive to the training procedure parameters, unoptimized variants found during the tuning process sometimes performed worse than random search. So, while the Two-Tower model offers a good meta-learning solution for CASH in high-dimensional hyperparameter search space, the complexity of gradient-descent model training likely is the reason that it does not profit as much as other methods do when performance-based representations can be used directly.

![](images/483ac99b7f79f66645f9afc0968ec98dd6eb977b937714ff2f2795ec15b28628.jpg)

![](images/8cd62b8c9478b4f7964df6becda50db00c83db7ae9b6e34b7009e93bbbd89d16.jpg)  
Figure 6: Test normalized regret of the Two-Tower variants on the CASH (left) and HPO (right) benchmarks, with MeanMMR and Linear-Perf as references. The two representations behave almost identically on HPO and similarly on CASH; neither clearly improves on MeanMMR, and both trail Linear-Perf. Shaded bands are 95% bootstrap CIs across tasks.

## B.11 COMPUTE

All experiments were run on a single machine using an AMD Ryzen 5 2600 Six-Core CPU, 16 GB of Memory and a GTX 1080 GPU, where only the Two-Tower model uses the GPU, running Windows with Python 3.11. Key library versions are scikit-learn 1.3.2, PyTorch 2.5.1 (CUDA 11.8), BoTorch 0.16.1, GPyTorch 1.15.2, and Hydra 1.3.2; exact pinned versions for all dependencies are given in the code release’s pyproject.toml and uv.lock. Table 5 reports total wall-clock time per model family across all tasks for both benchmarks including the scalability experiments, for 27h 20m in total.

Table 5: Total wall-clock compute time per model family.
<table><tr><td>Model family</td><td>Wall-clock time</td></tr><tr><td>Two-Tower</td><td>9h 25m</td></tr><tr><td>Gaussian Process</td><td>7h43m</td></tr><tr><td>Random Forest</td><td>5h40m</td></tr><tr><td>Meta-learning baselines Model-free</td><td>2h57m</td></tr><tr><td></td><td>0h54m</td></tr><tr><td>Linear</td><td>0h41m</td></tr><tr><td>Total</td><td>27h 20m</td></tr></table>

## C LINEAR SURROGATE MODELS

## C.1 BAYESIAN RIDGE FORMULATION

We use the Bayesian ridge regression formulation of MacKay (1992), as implemented in scikitlearn’s BayesianRidge estimator (Pedregosa et al., 2011). Both Linear-Hyper and Linear-Perf use this estimator; for Linear-Perf, we disable the intercept to preserve the weighted-combination interpretation over prior tasks, which also gave a small empirical improvement. For target-task observations $\mathcal { D } _ { t } ^ { ( n ) } = \{ ( x _ { \lambda _ { i } } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , the surrogate assumes a linear Gaussian likelihood,

$$
y _ { i } = x _ { \lambda _ { i } } ^ { \top } w + \epsilon _ { i } , \qquad \epsilon _ { i } \sim { \mathcal N } ( 0 , \alpha ^ { - 1 } ) ,\tag{6}
$$

with an isotropic Gaussian prior over the regression weights,

$$
p ( w \mid \eta ) = \mathcal { N } ( w \mid 0 , \eta ^ { - 1 } I ) .\tag{7}
$$

Here α denotes the noise precision and η the weight precision, corresponding to scikit-learn’s coefficient-precision parameter lambda ; we use $\eta$ to avoid overloading the configuration notation λ. Both precisions are themselves assigned Gamma hyperpriors,

$$
\alpha \sim \mathrm { G a m m a } ( \alpha _ { 1 } , \alpha _ { 2 } ) , \qquad \eta \sim \mathrm { G a m m a } ( \lambda _ { 1 } , \lambda _ { 2 } ) ,\tag{8}
$$

using the shape-rate parameterization, where we use the scikit-learn defaults $\alpha _ { 1 } = \alpha _ { 2 } = \lambda _ { 1 } =$ $\lambda _ { 2 } \doteq 1 0 ^ { - 6 }$ . Training is therefore not variational inference. For fixed α and $\eta ,$ the posterior over w is available in closed form as a Gaussian. The precision parameters are then updated iteratively by empirical Bayes, also known as type-II maximum likelihood or evidence maximization, using the MacKay-style update rules described by MacKay (1992) and Tipping (2001). We keep the default stopping settings, max $. { \dot { \mathtt { 1 } } } \mathrm { t e r } { = } 3 0 0$ and $\mathtt { t o l } \mathtt { = } 1 \mathtt { e } \mathtt { - } 3$ . Predictions use the resulting Gaussian posterior over w to return both a predictive mean and standard deviation, which are passed to Expected Improvement.

## C.2 CHOICE OF THE LINEAR SURROGATE

A series of early experiments was done that resulted in the choice of Bayesian ridge regression. Initially, ordinary least squares optimization without regularization was attempted, referencing the existing work of OBOE (Yang et al., 2019). However, tuning the dimensionality reduction factor with PCA resulted in two findings. Firstly, predictive performance was most optimal when the rank equals the number of tasks (the full matrix rank). Secondly, there was a strong peak in prediction error whenever the number of observed target samples was equal to the rank of the dimensionality reduction. The peaking effect was most obvious when measuring prediction error per step, and not as easily seen when viewing regret curves over the entire process, possibly hiding this from previous authors. This peaking behavior is closely related to the classical peaking phenomenon and its modern discussion under double descent (Loog et al., 2020; Viering & Loog, 2022). The instability of the unregularized model depending on the dimensionality motivated the use of Ridge regression without dimensionality reduction. This consistently provided better performance estimates without peaking behaviour.

Ridge regression was adapted to have an acquisition function either using non-probabilistic ridge regression with MMR, or Bayesian ridge regression using EI. When tuned well, both methods performed comparably. That also confirms the feasibility of using MMR as a replacement for EI.

However, Ridge regression with MMR requires tuning the ridge regularization strength and MMR greediness, and Bayesian ridge regression with EI did not require any tuning, beyond always disabling the intercept. Hence Bayesian ridge regression was proposed and used to represent the linear model family.

Bayesian ridge regression re-estimates the noise and weight-prior precisions by marginal-likelihood maximisation on every per-task fit. To test whether this per-fit adaptation is the reason for any significant differences between the Linear-Perf model with respect to ABLR, we compare against a fixed-precision Bayesian linear regression head identical to the one used by ABLR (Perrone et al., 2018). On both benchmarks the two are statistically indistinguishable (average test normalized regret, CASH 12.19 vs. 12.08, p = 0.76; HPO 9.84 vs. 9.96, p = 0.20; paired Wilcoxon signedrank over tasks). The adaptive prior precision estimation is therefore not the source of the linear model’s performance: the same fixed-precision head used by ABLR performs equivalently when set once to a good value, supporting the use of the simpler Bayesian ridge surrogate that does not require setting the precision.

## C.3 SIGN OF THE LEARNED TASK WEIGHTS

When Linear-Perf uses the historical performance vector as input, each regression coefficient weights performance on one specific prior task, so the fitted weights reveal how the target task is expressed as a combination of prior tasks. Figure 7 shows these weights for a representative CASH target task. As target-task observations accumulate, the fraction of negative weights grows to roughly half, far from the non-negative weights that a simple averaging of prior tasks would produce. The Bayesian ridge surrogate therefore does not merely average similar prior tasks; it also subtracts dissimilar ones. This signed combination is strictly more expressive than the non-negative ranking-agreement weights used by RGPE, consistent with RGPE-Oracle trailing Linear-Perf on CASH (see Section B.9).

![](images/be712edfbaa8f8ad9aa1db22461b5e46f4720ee41c228e8dde8b3daf71ed33aa.jpg)  
Figure 7: Sign of the learned Bayesian-ridge weights over prior tasks for a representative CASH target task using the performance representation: the fraction of weights that are positive versus negative as the number of observed target-task configurations grows.

## D ADAPTIVE OVERFITTING

As described in Section B.3, we separate validation and test regret, and report only test regret in all main figures and tables. Figure 8 and Figure 9 compare the two metrics per method for the CASH and HPO benchmarks. Validation regret is consistently somewhat better than test regret, although this pattern is shared across methods and would not have significantly changed the main conclusions. The exception is a slightly larger gap for active testing. None of the methods appear to be overfitting to the extent that early stopping would have been necessary. We retain the validation-test split and reported test regret as it reflects more accurately how well a configuration might work when applied to real-world data after deployment that is not from the split it has been tuned on.

Across methods, validation and test regret track each other closely and decline together, with only minor gaps at higher evaluation counts (for example Active Testing on CASH). We therefore observe little adaptive overfitting in these results. On HPO the two curves stop improving simultaneously, which points to the limited search space, rather than overfitting to the validation split, as the facto that bounds further progress. We nonetheless retain the validation-test split as a conservative safeguard, since selecting the best-so-far configuration on the same data used to guide the search can in principle reward overfitting (Schroder et al., 2025).¨

![](images/30e6233a32c79d6475a9aa2b3e908a0addd92cad3acae049b3fd5692493df1ca.jpg)  
Figure 8: Validation and test regret for all methods on the CASH benchmark, with a horizontal line highlighting the best obtained test regret across methods within the 200 iterations.

## E TABARENA LEADERBOARD DATASET

## E.1 BASE MODEL SET

All models are taken from the TabArena benchmark leaderboard (Erickson et al., 2025). For most models, the dataset contains results of one default configuration plus 200 randomly sampled configurations from the model’s search space (201 total). Foundation models that were only evaluated at their default setting have a single configuration.

## E.2 XGBOOST SEARCH SPACE

The XGBoost search space used by the TabArena authors to generate the 200 random configurations is shown below in Table 7.

![](images/22453d3fadcad7daae9f76657c4b961014e96b77fe838fdf6c217fb8fe5ea141.jpg)  
Figure 9: Validation and test regret for all methods on the HPO benchmark, with a horizontal line highlighting the best obtained test regret across methods within the 200 iterations.

Table 6: Base models used in the TabArena meta-dataset and the number of configurations evaluated per model.
<table><tr><td>Model</td><td>Category</td><td>Configurations</td></tr><tr><td>XGBoost</td><td>Gradient boosting</td><td>201</td></tr><tr><td>LightGBM</td><td>Gradient boosting</td><td>201</td></tr><tr><td>LightGBM (AIO)</td><td>Gradient boosting</td><td>201</td></tr><tr><td>CatBoost</td><td>Gradient boosting</td><td>201</td></tr><tr><td>Extra-Trees</td><td>Tree ensemble</td><td>201</td></tr><tr><td>Random Forest</td><td>Tree ensemble</td><td>201</td></tr><tr><td>k-NN</td><td>Classical</td><td>201</td></tr><tr><td>Linear Model</td><td>Classical</td><td>201</td></tr><tr><td>MLP (Torch)</td><td>Neural network</td><td>201</td></tr><tr><td>MLP (FastAI)</td><td>Neural network</td><td>201</td></tr><tr><td>RealMLP</td><td>Neural network</td><td>201</td></tr><tr><td>ModernNCA</td><td>Neural network</td><td>201</td></tr><tr><td>TabM</td><td>Neural network</td><td>201</td></tr><tr><td>xRFM</td><td>Neural network</td><td>201</td></tr><tr><td>EBM</td><td>Interpretable</td><td>201</td></tr><tr><td>TabPFNv2</td><td>Foundation model</td><td>201</td></tr><tr><td>RealTabPFN v2.5</td><td>Foundation model</td><td>201</td></tr><tr><td>TabDPT</td><td>Foundation model</td><td>201</td></tr><tr><td>TabICL</td><td>Foundation model</td><td>1</td></tr><tr><td>LimiX</td><td>Foundation model</td><td>1</td></tr><tr><td>Mitra</td><td>Foundation model</td><td>1</td></tr><tr><td>BetaTabPFN</td><td>Foundation model</td><td>1</td></tr><tr><td>TabFlex</td><td>Foundation model</td><td>1</td></tr><tr><td>Total</td><td></td><td>3623</td></tr></table>

Table 7: XGBoost hyperparameter search space. Log scale means the parameter is sampled uniformly in log space.
<table><tr><td>Hyperparameter</td><td>Type</td><td>Range</td><td>Scale</td></tr><tr><td>learning_rate</td><td>Float</td><td> $[ 5 \times 1 0 ^ { - 3 } , 1 0 ^ { - 1 } ]$ </td><td>Log</td></tr><tr><td>max_depth</td><td>Integer</td><td> $[ 4 , 1 0 ]$ </td><td>Log</td></tr><tr><td>min_child_weight</td><td>Float</td><td> $[ 1 0 ^ { - 3 } , 5 . 0 ]$ </td><td>Log</td></tr><tr><td>subsample</td><td>Float</td><td>[0.6, 1.0]</td><td>Linear</td></tr><tr><td>colsample_bylevel</td><td>Float</td><td>[0.6, 1.0]</td><td>Linear</td></tr><tr><td>colsample_bynode Float</td><td></td><td>[0.6, 1.0]</td><td>Linear</td></tr><tr><td>reg-alpha</td><td>Float</td><td> $\mathrm { | 1 0 ^ { - 4 } , 5 . 0 | }$ </td><td>Linear</td></tr><tr><td>reg_lambda</td><td>Float</td><td> $[ 1 0 ^ { - 4 } , 5 . 0 ]$ </td><td>Linear</td></tr><tr><td>grow-policy</td><td></td><td>Categorical {depthwise, lossguide}</td><td></td></tr><tr><td>max_cat_to_onehot Integer</td><td></td><td>[8, 100]</td><td>Log</td></tr><tr><td>max_leaves</td><td>Integer</td><td>[8, 1024]</td><td>Log</td></tr></table>