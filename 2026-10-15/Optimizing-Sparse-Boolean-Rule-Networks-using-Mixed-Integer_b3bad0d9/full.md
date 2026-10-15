# Optimizing Sparse Boolean Rule Networks using Mixed Integer Linear Programming

Thomas Pammer<sup>1[0009−0009−1140−1890]</sup>, Johannes Fürnkranz<sup>1,2[0000−0002−1207−0159]</sup>, and Markus Sinnl<sup>3[0000−0003−1439−8702]</sup>

<sup>1</sup> Institute for Application-Oriented Knowledge Processing (FAW) <sup>2</sup> LIT AI Lab

3 Institute of Business Analytics & Technology Transformation/JKU Business School Johannes Kepler University Linz, Austria

Abstract. We study the problem of learning accurate and cognitively simple binary classifiers that are directly interpretable as Boolean rules. We propose mixed integer linear programming (MILP) formulations for training sparse disjunctive normal form (DNF) networks, viewed as small networks with a conjunctive hidden layer and a disjunctive output layer. Contrary to previous work that uses MILP for inductive rule learning, we start with a fixed network structure, for which we then find optimal weights. A regularized objective explicitly trades of misclassification costs against sparsity, yielding globally optimized rule sets. Our main technical contribution is an enhanced formulation that substantially reduces the number of integer decision variables and eliminates redundant constraints, while preserving exact Boolean semantics for all feasible solutions. Furthermore, we develop a combined approach based on initialization and dropout to break symmetries in the weights between rules, mitigating redundancy, and encouraging more diverse and stable rule representations. Experiments on synthetic benchmarks and UCI datasets show that the advanced formulation improves computational eficiency and achieves competitive or superior accuracy compared to established interpretable baselines, including SAT-based DNF learning and well-known rule- and tree-based learners. Overall, our approach provides a principled optimization framework for learning compact Boolean rule sets with explicit control over accuracy and sparsity trade-ofs.

Keywords: DNF Networks · Mixed Integer Linear Programming · Rule Learning · Classification.

## 1 Introduction

The importance of interpretable classifiers is gaining attention in machine learning, with strong links to logic, optimization, and knowledge representation. Although neural networks achieve remarkable predictive performance, their blackbox nature often limits their practical applicability in domains where transparency is essential. A promising direction toward interpretable models is to represent classifiers as Boolean circuits, where outputs arise from logical combinations of input features. Such circuits are structurally simple, semantically clear, and align naturally with rule-based reasoning. Among various representations, disjunctive normal form (DNF) structures, which essentially represent sets of rules, are particularly appealing for their interpretability and compatibility with eficient reasoning. However, exact learning of Boolean circuits from data remains dificult because the space of possible configurations grows exponentially with the number of features and constraints.

Mixed integer linear programming (MILP) provides a principled way to combine logical consistency with optimization. By embedding Boolean relationships directly into linear constraints, MILP enables the learning of accurate and compact classifiers with explicit control of sparsity and model complexity. Our approach employs a parallel disjunctive structure where multiple conjunctive nodes act simultaneously, and the classifier predicts a positive outcome whenever any learned rule is satisfied.

We introduce two complementary MILP formulations for learning such Boolean circuit classifiers and we additionally compare two slightly diferent variants of the more advanced formulation. They can be viewed as small rule networks, where input features feed into hidden conjunctive nodes connected by a disjunction to the output. Each hidden node defines a rule of the form “if the active features are present, then predict positive”, where the activation of a feature depends on its weight. In the basic formulation, hidden nodes are restricted to take binary values, whereas in the advanced formulations they are either relaxed to continuous variables or handled in aggregated constraints. Using reformulated constraints, parts of the decision variables can be removed, further improving the computational eficiency of both advanced formulations. Empirical results on synthetic and benchmark datasets show that we achieve competitive accuracy.

Our contributions are summarized as follows:

– A basic MILP formulation for learning interpretable classification rules.

– An advanced formulation that improves eficiency while maintaining interpretability.

– A reformulation of the advanced MILP formulation using aggregated constraints to improve computational eficiency.

– A comparison with rule-, decision tree- and SAT-based methods demonstrating competitive accuracy and exact optimization.

– A combined approach using informed initialization and variable reduction for our advanced formulation to tackle larger problems.

The remainder of this paper is structured as follows. Section 2 shows an overview of related work. Section 3 formally introduces sparse Boolean rule networks, while Section 4 presents the MILP formulations developed in this paper. Section 5 reports the computational results obtained with the proposed MILP models and compares them with well-established classification algorithms. Section 6 introduces a novel combined approach for eficiently solving larger problem instances. Finally, Section 7 summarizes the main findings of this work and outlines promising directions for future research.

## 2 Related Work

Learning interpretable classifiers based on Boolean logic has a long tradition in machine learning, including circuit learning, inductive logic programming, rule induction, and associative classification. DNF classifiers have been extensively studied from a theoretical perspective [15,23] and are implicitly produced by many practical rule learning systems. However, most existing approaches do not explicitly optimize for cognitive simplicity under global optimality guarantees. In particular, inductive logic programming [8,18], associative classification [6,29], classical rule induction methods [7,12] and modern rule learning algorithms [10,14,16,25] typically rely on greedy or sequential covering strategies, which often yield large rule sets without explicit global control of sparsity or misclassification.

More recently, optimization-based approaches have been proposed to explicitly balance predictive accuracy and model simplicity. MILP has emerged as a powerful framework in this context, as it enables the direct encoding of logical constraints in combination with global optimization objectives. Several works represent trained neural networks as MILPs in order to model piecewise-linear activations and analyze network behavior [11]. Related approaches formulate the training of sparse neural networks directly as MILP problems by encoding binary activations and sparsity constraints while optimizing continuous or quantized weights [20]. Furthermore, rule-based models have been optimized via column generation techniques to handle large candidate spaces [9,27].

Only a limited number of optimization-based methods explicitly target sparse DNF classifiers as structured rule networks. Existing approaches often rely on pre-mined candidate rules, explicit pattern enumeration, or decomposed optimization strategies to manage combinatorial complexity [22,24]. In contrast, our current formulations adopt a predefined DNF network structure that maps binary inputs directly to conjunctions and optimizes all rules simultaneously within a single MILP. This provides explicit control over sparsity and misclassification while maintaining exact Boolean semantics and a clear rule-network interpretation.

Recent work has employed SAT-based methods for learning Boolean networks with layered structures in negation normal form (NNF) [21], which may be viewed as a generalization of DNF. This approach encodes network behavior as propositional satisfiability problems and leverages modern SAT solvers to obtain logically consistent solutions. However, while SAT solvers provide strong guarantees of logical consistency, they do not naturally support optimization objectives such as accuracy–sparsity trade-ofs, which are directly addressed in our MILP formulations. Greedy optimization of NNF network structures has also previously been explored in [1,26], but without any guarantees for optimality.

MILP has also been applied to learning optimal decision trees [4,13], where splits, feature usage, and depth are optimized jointly. Although such methods provide strong interpretability guarantees, they rely on hierarchical if–then–else structures with ordered and mutually exclusive decision paths.

## 3 Sparse Boolean Rule Networks

To ensure the interpretability of the learned model, we formulate our network using a DNF representation. A DNF is a logical formula composed of a disjunction of conjunctions of literals. For example, the DNF expression

$$
\left( f _ { 1 } \wedge f _ { 2 } \wedge f _ { 6 } \right) \vee \left( f _ { 3 } \wedge f _ { 4 } \right) \vee \left( f _ { 5 } \right)\tag{1}
$$

where $f _ { 1 } , f _ { 2 } , f _ { 3 } , f _ { 4 } , f _ { 5 }$ and $f _ { 6 }$ are Boolean input variables, evaluates to true if at least one of its conjunctions evaluates to true. From an interpretability perspective, each conjunctive node corresponds to a human-readable rule that defines a suficient condition for the positive class. Thus, the overall DNF structure can be viewed as a set of rules where satisfying any one of them yields a positive classification.

Note that a DNF formula can also be interpreted as a two-layer network in which the first layer consists of conjunctive nodes (logical AND gates), and the second layer, which is a single output node, performs a disjunction (logical OR) over the outputs of the first layer. For example, the above DNF expression corresponds to the network shown below in Figure 1c.

## 3.1 Problem Definition

Given a training dataset consisting of an n × m binary feature matrix X, where n is the number of samples and m the number of features, and a binary label vector y of length n, our goal is to learn a DNF classifier that not only minimizes misclassification on the training data but also the complexity of the network for overfitting avoidance and increased interpretability. Analogous to neural networks, we assume a fixed network architecture with a predefined number of conjunctive nodes $\ell ,$ and then search for the optimal weights for this fixed architecture. More precisely, the proposed DNF network consists of:

– an input layer with m original and negated features,

– a hidden layer of ℓ conjunctive nodes, each representing a rule, and

– a single output node performing a disjunction over the conjunction outputs.

In principle, the network is fully connected: each input literal is connected to every conjunctive node, and each conjunctive node is connected to the output node. However, each connection is parameterized by a binary weight indicating whether the corresponding literal is active for the following conjunction or not. To promote interpretability and prevent overfitting, we aim to learn a sparse DNF network by also minimizing the number of active weights in the model. If the optimal classifier requires fewer than ℓ hidden nodes, the model automatically replicates the simplest rule — i.e., the rule with the fewest active weights — thereby allowing duplicate rules to be removed post hoc while preserving a sparse network. After training, a sample is classified as positive if at least one conjunctive node evaluates to true given the input features and the learned binary weights. Consequently, when predicting labels for unseen data, only connections with active (i.e. positive) weights should be considered.

<table><tr><td></td><td>Wi1 Wi2 Wi3</td></tr><tr><td>f1</td><td>1 0 0</td></tr><tr><td>f2</td><td>1 0 0</td></tr><tr><td>f3</td><td>0 1 0</td></tr><tr><td>f4</td><td>0 1 0</td></tr><tr><td>f5</td><td>0 0 1</td></tr><tr><td>f6</td><td>1 0 0</td></tr></table>

![](images/911d0628faa6e44dd9e86f9870fede5f8bd023dca46ea5051f638f2eae15e934.jpg)  
Fig. 1: Illustration of a DNF formula (a), its conversion to a weight matrix W (b), and the resulting network structure (c), consisting of an input layer X into which the k-th training example $X _ { k }$ is fed, a hidden layer H which is connected to the inputs with edges corresponding to the weight matrix, and a predictive node $\hat { y } _ { k }$ . Solid lines show active, and dashed lines show inactive weights. Bold lines highlight the computation path for the first conjunctive node.

Example 1. The highlighted connections in Figure 1c represent the weights of the first conjunctive node, corresponding to the logical rule $( f _ { 1 } \land f _ { 2 } \land f _ { 6 } )$ , which evaluates to true only if all three input features $f _ { 1 } , \ f _ { 2 }$ , and $f _ { 6 }$ are equal to 1. Using examples $X _ { 1 } = [ 1 , 0 , 1 , 0 , 0 , 1 ]$ and $X _ { 2 } = [ 1 , 1 , 0 , 1 , 0 , 1 ]$ we can observe that the first sample does not satisfy any of these rules, while the second sample fulfills the first. Consequently, due to the disjunctive nature of the DNF structure, the second sample is classified as positive.

Since the overall goal of this work is to learn optimal weights for a sparse DNF network, given an objective function that minimizes both misclassification and the number of active weights for a fixed network structure, the next section presents how this learning problem can be formulated and optimized as a MILP.

## 4 Optimizing Sparse Boolean Rule Networks via MILP

To optimize the weights of the sparse DNF network, we propose two MILP formulations. The first formulation, referred to as the basic MILP approach, is formulated directly based on the logical structure shown in Section 3, while the second formulation, called the advanced MILP approach, introduces several improvements to enhance computational eficiency, while ensuring the same level of interpretability.

Pammer et al.

## 4.1 Basic MILP Approach

The basic idea of the MILP formulation is to explicitly model the value of each hidden node for every training sample, independent of its label, using binary decision variables and linear constraints that capture the logical relationships between all inputs, weights, and hidden nodes. Therefore, we introduce the following binary decision variables:

$- \ w _ { i j } \in \{ 0 , 1 \}$ encodes whether input i is included in conjunctive node j $- ~ h _ { k j } \in \{ 0 , 1 \}$ represents the output of conjunctive node j for sample k $- ~ \hat { y } _ { k } \in \{ 0 , 1 \}$ represents the predicted class for sample k $- ~ e _ { k } \in \{ 0 , 1 \}$ indicates whether sample k is misclassified.

Using these variables, the MILP problem can be formulated as follows:

Objective:

$$
\operatorname* { m i n } \sum _ { k = 1 } ^ { n } e _ { k } + \left( \alpha \frac { \sqrt { n } } { \sqrt { m } } \right) \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { \ell } w _ { i j }\tag{2a}
$$

Subject to:

$$
\sum _ { i = 1 } ^ { m } ( 1 - x _ { k i } ) w _ { i j } \geq 1 - h _ { k j }
$$

$$
\forall \quad 1 \leq k \leq n , 1 \leq j \leq \ell\tag{2b}
$$

$$
\sum _ { i = 1 } ^ { m } ( 1 - x _ { k i } ) w _ { i j } \leq m ( 1 - h _ { k j } ) \qquad \forall \quad 1 \leq k \leq n , 1 \leq j \leq \ell\tag{2c}
$$

$$
\hat { y } _ { k } \ge h _ { k j }
$$

$$
\forall \quad 1 \leq k \leq n , 1 \leq j \leq \ell\tag{2d}
$$

$$
\hat { y } _ { k } \le \sum _ { j = 1 } ^ { \ell } h _ { k j }
$$

$$
\forall \quad 1 \leq k \leq n\tag{2e}
$$

$$
e _ { k } \geq \hat { y } _ { k } - y _ { k }
$$

$$
\forall \quad 1 \leq k \leq n\tag{2f}
$$

$$
e _ { k } \geq y _ { k } - \hat { y } _ { k }
$$

$$
\forall \quad 1 \leq k \leq n\tag{2g}
$$

$$
w _ { i j } , h _ { k j } , \hat { y } _ { k } , e _ { k } \in \{ 0 , 1 \}
$$

$$
\forall \quad 1 \leq k \leq n , 1 \leq i \leq m , 1 \leq j \leq \ell\tag{2h}
$$

where the left part of the objective function (2a) minimizes misclassifications on the training data, while the right part minimizes the number of active weights in the network to ensure sparsity, which handles overfitting and further ensures interpretability. The regularization parameter α controls the trade-of between accuracy on the training data and sparsity. Constraint (2b) ensures that rule j predicts true for sample $k \ ( \mathrm { i . e . , } \ h _ { k j } = 1 )$ if all input literals included in the conjunction (those for which $w _ { i j } = 1 )$ are also true. Conversely, constraint (2c) guarantees that if at least one input literal $x _ { k i }$ included in the conjunction $( w _ { i j } = 1 )$ is false for sample $k ,$ then rule j is not satisfied for sample k. Constraint (2d) ensures that the predicted label $\hat { y } _ { k }$ is true if at least one rule $j$ fires for sample k $( h _ { k j } = 1 )$ , whereas constraint $\mathrm { ( 2 e ) }$ guarantees that if no rule j is fulfilled for sample k $( \forall j : h _ { k j } = 0 )$ , then the predicted class $\hat { y } _ { k }$ must be false. Constraints (2f) and $\left( \mathrm { { 2 g } } \right)$ define the misclassification variable $e _ { k }$ based on the predicted class yˆ<sub>k</sub> and the true label $y _ { k }$ for each sample $k .$ Finally, constraint (2h) enforces the binary nature of all decision variables in the formulation.

![](images/4b2fb3d69e89fbe72cba1380c952a766226c96d5160549ddaaf9e6ab887b978a.jpg)  
(a) Basic approach

![](images/086ccf93fd930765fdd40e0d620c30b29e2a587a16595cbb559c815fa2dfafa5.jpg)  
(b) Advanced approach  
Fig. 2: DNF network structure for basic and advanced constraint formulation.

## 4.2 Advanced MILP Approach

As illustrated in Figure 2a, the basic MILP approach requires adding constraints for every combination of input feature, conjunctive node, and training sample, regardless of the input feature values, resulting in a fully connected network. Moreover, constraints must be added to restrict the hidden node values from both sides, since samples with positive $( y _ { k } = 1 )$ and negative class $( y _ { k } = 0 )$ labels, hereafter referred to as positive and negative samples, are treated identically. More precisely, for the decision variables representing hidden node activations, whose values depend on both the input features and the weights, two constraints are always introduced: one ensuring that the variable is set to one whenever the combination of input features and weights activates the node, and another ensuring that it is set to zero otherwise. Consequently, the number of constraints grows rapidly with the size of the instance, leading to substantial scalability challenges. The resulting large number of binary decision variables may lead to large branch-and-bound trees during optimization, making this basic formulation computationally expensive, in particular for larger datasets. To be more precise, using this formulation we need to optimize over $m \times \ell + n \times ( \ell + 2 )$ binary decision variables.

To address these limitations of the basic MILP approach, we propose an advanced MILP approach, which introduces the following improvements:

– Removal of decision variables for predicted class of all samples and hidden nodes of negative samples

– Replacement of binary decision variables for the remaining hidden nodes and misclassification indicators with continuous variables

– Reformulation of constraints to ensure that the continuous decision variables attain binary values in all feasible solutions

– Formulation of only the constraints that are necessary based on the label and feature values of each sample

Using these improvements, the advanced MILP model is formulated as follows:

Objective:

$$
\operatorname* { m i n } \sum _ { k = 1 } ^ { n } e _ { k } + \left( \alpha \frac { \sqrt { n } } { \sqrt { m } } \right) \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { \ell } w _ { i j }\tag{3a}
$$

Subject to:

$$
w _ { i j } \le 1 - h _ { k j } \qquad \forall 1 \le k \le n , 1 \le i \le m , 1 \le j \le \ell : \ y _ { k } = 1 , \ x _ { k i } = 0\tag{3b}
$$

$$
e _ { k } \geq 1 - \sum _ { j = 1 } ^ { \ell } h _ { k j } \qquad \forall 1 \leq k \leq n : \ y _ { k } = 1\tag{3c}
$$

$$
e _ { k } \geq 1 - \sum _ { i = 1 : x _ { k i } = 0 } ^ { m } w _ { i j } \quad \forall 1 \leq k \leq n , 1 \leq j \leq \ell : \ y _ { k } = 0\tag{3d}
$$

$$
w _ { i j } \in \{ 0 , 1 \} \qquad \forall 1 \leq i \leq m , 1 \leq j \leq \ell\tag{3e}
$$

$$
h _ { k j } , e _ { k } \geq 0 \qquad \forall 1 \leq k \leq n , 1 \leq j \leq \ell\tag{3f}
$$

While the objective function (3a) remains unchanged compared to the basic formulation, the constraints have been significantly modified to improve computational eficiency. When constructing the constraints, we distinguish between samples based on their class labels. Constraint (3b) ensures that, for positive samples, rule j is not fulfilled $( h _ { k j } = 0 )$ if sample k contains a false input literal $( x _ { k i } = 0 )$ that is included in the conjunctive node $( w _ { i j } = 1 )$

As the objective function automatically pushes all $h _ { k j }$ variables for positive samples to be fulfilled in order to minimize misclassifications, we only need to restrict the value of $h _ { k j }$ from above. Furthermore, we observe that only weight connections to negative input features must be considered, as positive input features cannot force hidden nodes to be unfulfilled, regardless of the weight values. By exploiting this separation, we avoid introducing this constraint for every combination of hidden nodes and decision variables for negative samples, as well as constraints involving weight connections to positive features for positive samples. Constraint (3c) ensures that, for positive samples $( y _ { k } = 1 )$ , the misclassification variable $e _ { k }$ is set to true if no rule j is fulfilled by sample k $( \forall j : h _ { k j } = 0 )$ Again, we only need to restrict the value of the misclassification decision variable from below, because the objective function automatically pushes it to be zero. For negative samples we can reformulate the constraints so that we only need the weight variables to define whether a sample is misclassified, thereby removing all decision variables for the hidden nodes of negative samples. This is possible because the values of the remaining decision variables for negative samples depend only on the values of the weights. More precisely, the reformulated constraint (3d) directly ensures that a negative sample is misclassified, if there exists at least one hypothetical node with no active weight connection from a negative input feature. Interestingly, no additional decision variables for the hidden nodes are required, as the misclassification can be mapped directly through the weights. Furthermore, we again only need to restrict the misclassification error from one side, because the objective function automatically pushes it to be zero. With these reformulations, we not only eliminate all decision variables for the hidden nodes of the negative samples, but also only need to consider connections from negative input features. Finally, constraints (3e) and (3f) define the binary nature of the weight variables and the continuous nature of the hidden node and misclassification variables, respectively. With these reformulations, all continuous decision variables are forced to binary values in any feasible solution.

Figure 2b illustrates that the advanced MILP approach only requires constraints for negative input features and distinguishes between positive and negative samples when constructing the constraints for the hidden nodes and misclassification variables. This substantially reduces the number of constraints in the model. In addition, the reformulation using continuous decision variables, which reduces the number of binary decision variables to $m \times \ell$ and the total number of decision variables to $m \times \ell + n ^ { \oplus } \times \ell + n$ , where $n ^ { \oplus }$ represents the number of positive samples, leads to smaller branch-and-bound trees during optimization, thereby making the advanced formulation computationally more eficient, particularly for larger datasets.

## 4.3 Aggregated Constraints

An aggregated constraint combines multiple individual constraints into one more compact aggregated formulation, thereby reducing the overall number of constraints. Although linear programming (LP) theory generally suggests that aggregated constraints lead to weaker LP relaxations at branch and bound nodes, partial aggregation can still be beneficial in practice, as it reduces the total number of constraints required to ensure model correctness [28]. More specifically, we aim to investigate the trade-of between a more compact model formulation with fewer constraints and the potentially weaker relaxation induced by the aggregation. This trade-of is particularly relevant in combination with the heuristic and branching techniques employed by modern MILP solvers such as Gurobi,<sup>4</sup> which is used as the optimization solver in our computational evaluation.

More precisely, constraints (3b) can be replaced by the following aggregated formulation:

$$
\sum _ { i = 1 : x _ { k i } = 0 } ^ { m } \frac { w _ { i j } } { n _ { k } ^ { \ominus } } + h _ { k j } \leq 1 \quad \forall \ 1 \leq k \leq n , 1 \leq j \leq \ell : \ y _ { k } = 1 ,\tag{4}
$$

where $n _ { k } ^ { \ominus }$ represents the number of negative input features of sample k. When using constraints (4), the hidden-node decision variables must be explicitly restricted to binary values in order to ensure model correctness, since the aggregated formulation no longer guarantees integrality implicitly through the constraint structure. Nevertheless, because Gurobi employs advanced branching, cutting-plane, and primal heuristic strategies, the aggregated formulation may still provide competitive computational performance despite the weaker LP relaxation.

## 5 Experimental Evaluation

In this section, we first compare the basic and advanced MILP approaches regarding their computational eficiency and solution quality on artificial datasets, to analyze the eficiency gain of the advanced formulation using diferent time limits for the solver. Secondly, we evaluate the performance of the advanced MILP approach against state-of-the-art interpretable classification methods on real-world UCI datasets. In particular, we compare the accuracy of our approach against the SAT-based method of [21], where the authors also use a DNF structure for classification, the Lord method [14], which learns locally optimal rules and outperforms other rule-based methods, as well as the well-known rule learning algorithm Ripper [7] and the decision tree learner Cart [5]. We computed the accuracy of our MILP approaches and all compared algorithms using 10-fold cross-validation.

## 5.1 Experimental Setup

We implemented both MILP formulations in Python, using the Gurobi 12.0.0 MILP solver to optimize the MILP models shown in Section 4. We let our experiments run on a single core of an AMD EPYC 7543 processor running at 3.9 GHz. To compare and evaluate the proposed MILP approaches against state-of-the-art interpretable classification methods, we selected the following datasets.

– Artificial instance set: This instance set consists of 20 datasets with 10 binary features and 1024 samples each. These datasets were artificially generated by [2] and have also been used in [1,3,21]. We use the same number of hidden nodes as used in [2] and [21], which is 20 hidden nodes, to ensure a fair comparison. We identify these datasets by ordering them based on the random seed used during their generation and referring to them as d01 to d20. We selected these datasets to compare our approaches under diferent time limits, as all instances have the same size and can therefore be evaluated using identical time limits across all experiments.

UCI Datasets: To evaluate our advanced approach on real-world data, we selected nine widely used benchmark datasets from the UCI repository <sup>5</sup>. The number of binarized features ranges from 17 to 116, while the number of samples ranges from 435 to 8124. All datasets represent binary classification problems, except for the car-evaluation dataset, which we transformed into a binary task by predicting whether a sample belongs to the most frequent class. For our approach, all input attributes were binarized by first discretizing them with Fusinter [30] and then using a one-hot encoding, as described in [14]. The resulting binary datasets were used in all experiments. Additional experiments on the original datasets with rule- and tree-based learning algorithms yielded similar results, so we used the binarized datasets for all methods to ensure a fair comparison.

## 5.2 Parameter Tuning

As we use a fixed network architecture with a predefined number of hidden nodes, we only need to tune the sparsity parameter α for both MILP approaches. The reason for using a fixed network architecture is that the long-term goal of this work is to apply the optimization approach to individual components of a decomposed deep rule network, where the structure is predefined. We performed the parameter tuning on a training dataset where we included five artificial datasets and one UCI dataset. We used the automatic parameter tuning tool Irace [17], which uses an iterated racing procedure to identify the best value for α. We set the experimental budget to 1000 algorithm runs while using a time limit of 1200 seconds for solving each MILP, where we used the advanced approach. We defined the candidate values for α to be in the range of [0.001, 0.5]. The best identified value for α was 0.06, which we used for all MILP approaches in all experiments.

## 5.3 Comparison of Basic and Advanced MILP Approaches

To show the improvements found with the advanced MILP approach and to analyze the impact of the aggregation of constraints, we first tested our approaches on the artificial datasets using the same time limits and parameter configuration for each version. We therefore let both variants run using 10-fold cross-validation as described above and reported the results in Table 1.

In Table 1, the first column lists the dataset identifiers. The second, third, and fourth columns report the accuracies achieved by the basic MILP approach, the advanced MILP approach, and the advanced approach with aggregated constraints, respectively, under a time limit of 120 seconds. The last three columns present the corresponding results for a time limit of 360 seconds.

As can be observed, both variants of the advanced approach consistently outperform the basic MILP formulation under both time limits. When comparing the average accuracies, the performance gap between the advanced and the basic formulations increases as the time limit decreases. This observation further supports the claim that the advanced MILP formulation is computationally more eficient than the basic model.

Table 1: Comparing accuracies of basic and advanced MILP approaches on artificial datasets. The best accuracy per instance is marked in bold, while the best accuracy for the shorter time limit is in italic.
<table><tr><td rowspan="2">Dataset</td><td colspan="4">120 s</td><td colspan="2">360 s</td></tr><tr><td>basic</td><td>advanced</td><td>aggregated</td><td>basic</td><td>advanced</td><td>aggregated</td></tr><tr><td>d01</td><td>96.00</td><td>96.49</td><td>98.05</td><td>99.90</td><td>100.00</td><td>100.00</td></tr><tr><td>d02</td><td>84.66</td><td>97.85</td><td>98.05</td><td>97.55</td><td>99.12</td><td>99.22</td></tr><tr><td>d03</td><td>96.49</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td>d04</td><td>84.76</td><td>89.94</td><td>94.44</td><td>99.02</td><td>99.71</td><td>99.71</td></tr><tr><td>d05</td><td>88.18</td><td>91.21</td><td>94.44</td><td>94.54</td><td>98.64</td><td>98.83</td></tr><tr><td>d06</td><td>83.50</td><td>91.79</td><td>97.94</td><td>94.94</td><td>99.71</td><td>99.71</td></tr><tr><td>d07</td><td>96.68</td><td>98.05</td><td>98.34</td><td>99.22</td><td>100.00</td><td>99.90</td></tr><tr><td>d08</td><td>94.64</td><td>99.61</td><td>99.90</td><td>99.61</td><td>99.90</td><td>99.90</td></tr><tr><td>d09</td><td>80.96</td><td>93.46</td><td>97.66</td><td>96.49</td><td>99.41</td><td>99.51</td></tr><tr><td>d10</td><td>96.31</td><td>100.00</td><td>99.80</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td>d11</td><td>91.12</td><td>88.50</td><td>97.46</td><td>98.54</td><td>99.12</td><td>99.51</td></tr><tr><td>d12</td><td>89.25</td><td>96.49</td><td>98.53</td><td>97.16</td><td>98.73</td><td>98.93</td></tr><tr><td>d13</td><td>90.14</td><td>89.16</td><td>96.88</td><td>96.68</td><td>99.32</td><td>99.51</td></tr><tr><td>d14</td><td>89.54</td><td>88.39</td><td>94.44</td><td>98.44</td><td>97.95</td><td>98.83</td></tr><tr><td>d15</td><td>98.44</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td>d16</td><td>97.66</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td>d17</td><td>85.56</td><td>87.91</td><td>97.65</td><td>99.32</td><td>99.61</td><td>99.71</td></tr><tr><td>d18</td><td>89.95</td><td>92.18</td><td>96.09</td><td>97.56</td><td>99.22</td><td>99.61</td></tr><tr><td>d19</td><td>88.29</td><td>96.19</td><td>96.09</td><td>96.39</td><td>98.44</td><td>99.22</td></tr><tr><td>d20</td><td>89.68</td><td>93.56</td><td>98.54</td><td>99.61</td><td>99.32</td><td>99.61</td></tr><tr><td>Ø rank</td><td>5.85</td><td>4.25</td><td>3.30</td><td>2.85</td><td>1.75</td><td>1.05</td></tr><tr><td>∅ acc</td><td>90.59</td><td>94.54</td><td>97.72</td><td>98.25</td><td>99.41</td><td>99.59</td></tr></table>

Moreover, the results indicate that, particularly under tight time limits, the trade-of of using a model with fewer constraints outweighs the benefits of the stronger formulations obtained from individual constraints. Overall, the best results are achieved by the advanced approach using aggregated constraints with a time limit of 360 seconds. In summary, the advanced approach clearly outperforms the basic one for both considered time limits, while the variant with aggregated constraints even further improves eficiency as it reduces the number of constraints. Although the performance diference between the aggregated and non-aggregated versions decreases as the time limit increases, which is consistent with our expectations, we decided to use the aggregated-constraint formulation for the comparison with well-known algorithms on real-world datasets. The reason is that, especially for larger datasets, computational eficiency becomes increasingly important.

Table 2: Comparing accuracies of the advanced MILP approach with aggregated constraints to existing algorithms on UCI datasets. The best accuracy per instance is marked in bold.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Time limit</td><td colspan="5">Approaches</td></tr><tr><td>MILP</td><td>SAT</td><td>LORD</td><td>RIPPER</td><td>CART</td></tr><tr><td>breast</td><td>300</td><td>95.99</td><td>94.56</td><td>94.71</td><td>94.28</td><td>93.42</td></tr><tr><td>car-evaluation</td><td>1200</td><td>99.36</td><td>98.67</td><td>98.20</td><td>96.70</td><td>98.03</td></tr><tr><td>kr-vs-kp</td><td>7200</td><td>99.66</td><td>99.50</td><td>99.50</td><td>99.37</td><td>99.44</td></tr><tr><td>monk-1</td><td>30</td><td>100.00</td><td>100.00</td><td>100.00</td><td>93.70</td><td>100.00</td></tr><tr><td>monk-2</td><td>180</td><td>100.00</td><td>98.67</td><td>87.20</td><td>73.54</td><td>97.84</td></tr><tr><td>monk-3</td><td>1800</td><td>98.92</td><td>96.20</td><td>97.65</td><td>98.56</td><td>98.92</td></tr><tr><td>mushroom</td><td>600</td><td>100.00</td><td>100.00</td><td>100.00</td><td>99.99</td><td>99.28</td></tr><tr><td>tic-tac-toe</td><td>600</td><td>100.00</td><td>99.27</td><td>98.22</td><td>97.91</td><td>93.22</td></tr><tr><td>vote</td><td>200</td><td>96.56</td><td>91.03</td><td>95.41</td><td>94.94</td><td>95.63</td></tr><tr><td>Ø rank</td><td></td><td>1.00</td><td>2.56</td><td>2.67</td><td>4.33</td><td>3.44</td></tr><tr><td>∅ acc</td><td></td><td>98.97</td><td>97.54</td><td>96.77</td><td>94.33</td><td>97.31</td></tr></table>

## 5.4 Comparing Advanced MILP Approach to Existing Approaches

As previously mentioned, we compare the accuracy of our method to the SATbased method for learning DNF structures from [21], the Lord rule discoverer by [14] and the classic algorithms Ripper [7] and Cart [5]. These algorithms have achieved highly competitive results on the tested datasets and can therefore serve as well-established reference points for our approach.

Table 2 compares the performance of the advanced MILP approach to its competitors on the real-world UCI datasets. As indicated by the bold values, our model matches or surpasses the predictive accuracy of all benchmark algorithms on all tested datasets. This further strengthens the argument for the robustness of our method, even on real-world instances. More precisely, the advanced MILP approach achieves a mean accuracy of nearly 99 % on the UCI datasets, outperforming all compared algorithms by more than 1 % on average. Since our method is based on an exact optimization framework, it naturally cannot compete with the solution times of heuristic approaches such as rule learners and decision tree learners which, however, was not the goal of this work.

## 6 Informed Initialization and Variable Reduction

The main objective of this combined approach is to eficiently tackle larger problems. This section is divided into two parts. First, we investigate the impact of warm-starting the MILP optimization with an initial solution that has been obtained by a quick greedy optimization, in our case the solution obtained by Ripper. Second, we analyze the efect of variable reduction by dropping unused weight connections from the Ripper solution. To evaluate the efectiveness of the proposed approach, we test the advanced MILP formulation with aggregated constraints, both with and without the proposed enhancements, on the UCI dataset adult, which consists of more than 48,000 samples and 130 binarized features. All experiments are conducted using 10-fold cross-validation.

## 6.1 Ripper-Based Initialization

Table 3 shows the impact of warm-starting the MILP optimization with the Ripper solution. As we can see, using the initialization leads to a significant improvement on the test accuracy, with an even larger diference at shorter time limits. While the solution obtained from random initializations slowly improves over time and can be expected to reach the same level at higher time-outs, an informed initialization helps to jump-start the optimization algorithm at a much higher level. However, the warm-started MILP optimization also constantly improves the initial Ripper solution, already by more than 0.3 % using a time limit of 300 seconds, highlighting the eficiency of the advanced MILP formulation in refining existing solutions.

Table 3: Accuracies on adult dataset with and without Ripper-Initialization.
<table><tr><td rowspan="2">RIPPER Initialization</td><td colspan="6">Time Limit</td></tr><tr><td>300</td><td>600</td><td>900</td><td>1800</td><td>3600</td><td>7200</td></tr><tr><td>No</td><td>76.07</td><td>76.07</td><td>76.07</td><td>76.63</td><td>78.63</td><td>80.28</td></tr><tr><td>Yes</td><td>84.75</td><td>84.77</td><td>84.87</td><td>84.87</td><td>84.96</td><td>85.17</td></tr></table>

## 6.2 Ripper-Based Variable Reduction

Figure 3 visualizes the impact of diferent dropout rates for the unused weight connections from the Ripper solution, on the advanced MILP formulation using aggregated constraints and the Ripper-Initialization. As expected, for very small time limits, configurations with higher dropout rates perform better because they reduce the solution space more aggressively. Interestingly, even for a higher time limit of 7200 seconds, the best result is achieved using a dropout rate of 70 %, thereby improving the original Ripper solution by more than 1 %, achieving a final test accuracy of 85.52 %. Similar results on dropout rates of 60 and 80 % strengthen the argument that, for our problem, it is useful to break symmetries in the weight matrix, since, without such variable reduction, each rule may be represented by any hidden node.

Overall, these findings indicate that it is not only beneficial to warm-start the MILP solver with a good initial solution, but that it is also useful to use variable reduction in order to decrease the solution space. Consequently, similar decomposition and variable reduction strategies represent a promising direction for future research.

![](images/df058cf7f6b268bd25e239fde24a2b5276d5504ce28c9ab12c4fc4838da2b9ee.jpg)  
Fig. 3: Accuracy on adult dataset by time limit and dropout rate.

## 7 Conclusion and Future Work

In summary, we demonstrated that the advanced MILP approach clearly outperforms the basic formulation. The main reason for this improvement lies in the reformulation of the constraints and the removal of unnecessary decision variables. Furthermore, we showed that the use of aggregated constraints can further improve the computational eficiency of the model, thereby enhancing the overall performance of the approach. Additionally, our advanced MILP approach consistently outperforms existing algorithms in terms of accuracy on real-world UCI datasets. As discussed in Section 3, the major advantage of MILP-based models is that sparsity can be directly incorporated into the objective function. This not only mitigates overfitting but also enhances interpretability by encouraging simpler, shorter rules within the optimization process. Through the reformulations introduced in the advanced approach, we have also improved the model’s computational eficiency for modern MILP solvers such as Gurobi. In addition, we showed that, particularly under tight time limits and for larger datasets, initializing the advanced MILP formulation with a greedily obtained solution and pruning part of the unused weight connections improves eficiency by mitigating network symmetries.

Even though these techniques already allow us to scale up to large datasets such as adult, scalability is still the main limitation of our current approach. A particularly promising direction is the application of Benders decomposition, which would allow us to divide the overall problem into a master problem and a set of subproblems (one per sample). We intend to explore this idea further in future work, with the goal of extending it towards deeper Boolean rule network architectures, with several consecutive AND and OR layers [2]. To this end, we aim to iteratively optimize selected parts of the network using decomposition strategies similar to those developed in previous work by the main author [19].

## Acknowledgments

This research was funded by the Austrian Science Fund (FWF) 10.55776/COE12.

## References

1. Beck, F., Fürnkranz, J.: Beyond DNF: First steps towards deep rule learning. In: Conference on Information Technologies, Application and Theory. pp. 61–68 (2021)

2. Beck, F., Fürnkranz, J.: An empirical investigation into deep and shallow rule learning. Frontiers in Artificial Intelligence 4, 689398 (2021)

3. Beck, F., Fürnkranz, J., Huynh, V.Q.P.: Structuring rule sets using binary decision diagrams. In: International Joint Conference on Rules and Reasoning. pp. 48–61. Springer (2021)

4. Bertsimas, D., Dunn, J.: Optimal classification trees. Machine Learning 106(7), 1039–1082 (2017)

5. Breiman, L., Friedman, J., Olshen, R.A., Stone, C.J.: Classification and regression trees. Wadsworth & Brooks (1984)

6. Cheng, H., Yan, X., Han, J., Hsu, C.W.: Discriminative frequent pattern analysis for efective classification. In: 2007 IEEE 23rd International Conference on Data Engineering. pp. 716–725. IEEE (2006)

7. Cohen, W.W., et al.: Fast efective rule induction. In: Proceedings of the twelfth International Conference on Machine Learning. pp. 115–123 (1995)

8. Cropper, A., Dumancic, S., Evans, R., Muggleton, S.H.: Inductive logic programming at 30. Machine Learning 111(1), 147–172 (2022)

9. Dash, S., Gunluk, O., Wei, D.: Boolean decision rules via column generation. Advances in Neural Information Processing Systems 31 (2018)

10. Ding, Y., Zhu, H., Chen, R., Li, R.: An eficient adaboost algorithm with the multiple thresholds classification. Applied Sciences 12(12), 5872 (2022)

11. Fischetti, M., Jo, J.: Deep neural networks and mixed integer linear optimization. Constraints 23(3), 296–309 (2018)

12. Fürnkranz, J., Gamberger, D., Lavrač, N.: Foundations of Rule Learning. Springer Science & Business Media (2012)

13. Günlük, O., Kalagnanam, J., Li, M., Menickelly, M., Scheinberg, K.: Optimal decision trees for categorical data via integer programming. Journal of Global Optimization 81(1), 233–260 (2021)

14. Huynh, V.Q.P., Fürnkranz, J., Beck, F.: Eficient learning of large sets of locally optimal classification rules. Machine Learning 112(2), 571–610 (2023)

15. Kearns, M.J., Vazirani, U.: An introduction to computational learning theory. MIT Press (1994)

16. Lakkaraju, H., Bach, S.H., Leskovec, J.: Interpretable decision sets: A joint framework for description and prediction. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp. 1675– 1684. ACM (2016)

17. López-Ibáñez, M., Dubois-Lacoste, J., Cáceres, L.P., Birattari, M., Stützle, T.: The irace package: Iterated racing for automatic algorithm configuration. Operations Research Perspectives 3, 43–58 (2016)

18. Muggleton, S., De Raedt, L.: Inductive logic programming: Theory and methods. The Journal of Logic Programming 19, 629–679 (1994)

19. Pammer, T., Sinnl, M.: A matheuristic for solving the single row facility layout problem. Computers & Operations Research p. 107397 (2026)

20. Rosenhahn, B.: Optimization of sparsity-constrained neural networks as a mixed integer linear program: NN2MILP. Journal of Optimization Theory and Applications 199(3), 931–954 (2023)

21. Seip, P., Beck, F., Fürnkranz, J., Hofstadler, C., Pfeifer, P., Seidl, M., Peharz, R., Szeider, S.: Towards SAT-based learning of NNF networks. In: 2025 27th International Symposium on Symbolic and Numeric Algorithms for Scientific Computing (SYNASC). pp. 1–5. IEEE (2025)

22. Su, G., Wei, D., Varshney, K.R., Malioutov, D.M.: Learning sparse two-level Boolean rules. In: 2016 IEEE 26th International Workshop on Machine Learning for Signal Processing (MLSP). pp. 1–6. IEEE (2016)

23. Valiant, L.G.: A theory of the learnable. Communications of the ACM 27(11), 1134–1142 (1984)

24. Wang, T., Rudin, C.: Learning optimized Or’s of And’s. arXiv preprint arXiv:1511.02210 (2015)

25. Wang, T., Rudin, C., Doshi-Velez, F., Liu, Y., Klampfl, E., MacNeille, P.: A bayesian framework for learning rule sets for interpretable classification. Journal of Machine Learning Research 18, 70:1–70:37 (2017)

26. Wang, Z., Zhang, W., Liu, N., Wang, J.: Transparent classification with multilayer logical perceptrons and random binarization. In: Proceedings 34th AAAI Conference on Artificial Intelligence. pp. 6331–6339. AAAI Press (2020)

27. Wei, D., Dash, S., Gao, T., Gunluk, O.: Generalized linear rule models. In: International Conference on Machine Learning. pp. 6687–6696. PMLR (2019)

28. Wolsey, L.A.: Integer programming. John Wiley & Sons (2020)

29. Yin, X., Han, J.: CPAR: Classification based on predictive association rules. In: Proceedings of the 2003 SIAM International Conference on Data Mining. pp. 331– 335. SIAM (2003)

30. Zighed, D.A., Rabaséda, S., Rakotomalala, R.: FUSINTER: A method for discretization of continuous attributes. International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems 6(03), 307–326 (1998)