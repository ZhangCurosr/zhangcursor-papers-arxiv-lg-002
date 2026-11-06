Authors are encouraged to submit new papers to INFORMS journals by means of a style file template, which includes the journal title. However, use of a template does not certify that the paper has been accepted for publication in the named journal. INFORMS journal templates are for the exclusive purpose of submitting to an INFORMS journal and should not be used to distribute the papers in print or online or to submit the papers to another publication.

# Understanding Data Influence in Transportation Networks: A Physics-Constrained Influence-Function Framework

Hamidreza Attari

Department of Civil and Environmental Engineering, University of California, Davis, CA, USA, hattari@ucdavis.edu

Yueyue Fan

Department of Civil and Environmental Engineering, University of California, Davis, CA, USA, yyfan@ucdavis.edu Corresponding author

Johannes O. Royset

Daniel J. Epstein Dept. of Industrial and Systems Engineering, University of Southern California, CA, USA, royset@usc.edu

Abstract: Understanding how individual training observations influence model predictions is important for developing reliable and interpretable data-driven transportation systems. Existing influence-function methods, however, are primarily formulated for unconstrained learning and do not explicitly account for physical constraints that couple diferent components of a transportation network. Under such constraints, perturbing a training observation can afect model parameters and predictions only through physically feasible directions, potentially changing how its influence propagates across the network. This paper develops a hard physics-informed influence-function framework for constrained learning, in which training-data influence is characterized within the physically feasible space defined by network constraints. The analytical results reveal how physical constraints alter the magnitude, direction, and spatial distribution of training-sample influence relative to conventional influence measures. Using trafic prediction under flow-conservation constraints as a case study, we show that influential observations identified by the proposed framework difer systematically from those identified by classical influence functions. Through orthogonal decomposition, we quantitatively characterize how much of a perturbation in the training data is propagated through

the network via the flow-conservation constraints and how much is removed by the constraint projection. The proposed influence framework provides a physically consistent approach for evaluating training-data influence in constrained transportation learning problems, with potential applications to data diagnostics, sensor evaluation, and targeted data collection.

Keywords: Influence functions; Physics-constrained learning; Constraint projection; Transportation network flow conservation; Value of information

## 1. Introduction

The rapid growth of AI and machine learning is creating new opportunities to develop data-driven models for understanding, predicting, and managing complex transportation systems. The accuracy and robustness of these models heavily depend on the quality and informativeness of the observations used for training. In transportation systems, data are often incomplete, noisy, and unevenly distributed across the network. Therefore, understanding how training data influence model learning and performance becomes increasingly important as such understanding can help identify where additional data collection is needed or matters most.

In the machine learning literature, Influence Functions (IFs), originating from robust statistics, have been adapted as a principled framework for quantifying the influence of training data. IFs provide a useful means of understanding model behavior by characterizing model sensitivity to perturbations in training data and identifying influential observations; such influence measures can further provide a basis for assessing the value of training information (Koh and Liang 2017). However, classical IF formulations primarily consider unconstrained learning, where observations influence model parameters through the empirical loss (Schioppa et al. 2022, Basu et al. 2021). This limits the direct applicability of classical IF methods to transportation network problems, where learning is often subject to physical constraints that capture the spatial and temporal interdependencies among network components. Such constraints fundamentally alter how model parameters respond to perturbations in training data, requiring influence analysis that explicitly accounts for the constrained learning process.

In this paper, we establish a new framework for evaluating data influence in transportation networks by approaching the problem as a physics-constrained sensitivity problem. While the proposed influence-function framework is motivated by the broader class of physics-constrained learning problems, network-level physical constraints are particularly fundamental to transportation systems. Many transportation problems are defined over networks in which observations and system states across diferent components are coupled through physical and operational constraints such as flow conservation, network connectivity, and capacity restrictions. We therefore use short-term transportation network flow prediction as a motivating use case for developing the proposed framework.

In this setting, flow-conservation constraints provide a concrete representation of network-level coupling among link flows, allowing us to demonstrate why such structural dependencies should be explicitly incorporated into influence-function analysis.

Physics-informed machine learning has received growing attention in both the broader AI community and transportation research, reflecting increasing recognition that incorporating physical knowledge and system structure into learning can improve model accuracy, robustness, and consistency with underlying system behavior. For example, in the transportation community, early eforts in short-term trafic prediction progressed from classical statistical and machine-learning methods to increasingly sophisticated deep learning architectures that capture temporal and spatial dependencies in trafic data (Williams and Hoel 2003, Ma et al. 2015, Fu et al. 2016). More recent graph-based approaches, such as the difusion convolutional recurrent neural network (DCRNN) (Li et al. 2018) and the spatio-temporal graph convolutional network (STGCN) (Yu et al. 2018), incorporate roadnetwork topology into the learning architecture, allowing dependencies among connected network components to be captured. However, representing network connectivity does not necessarily enforce the physical relationships governing flows over the network. Despite their strong predictive performance, these models generally do not explicitly impose network-level physical constraints, such as flow conservation, on model learning and prediction. Predictions that are physically implausible can misrepresent the actual system state, trigger inefective or counterproductive downstream actions, and ultimately impair the performance and reliability of transportation networks.

At the methodological level, physics-informed and constrained learning approaches have increasingly shifted from soft constraint enforcement toward hard feasibility guarantees. Early approaches typically incorporated physical or domain knowledge through penalty terms in the training loss, including penalty weights or Lagrange multipliers, thereby encouraging—but not guaranteeing—constraint satisfaction (Dener et al. 2020, Márquez-Neila et al. 2017). More recent approaches seek to enforce constraints directly within neural networks through feasible-set parameterizations, diferentiable projections, repair mechanisms, KKT-based formulations, or variable transformations (Min and Azizan 2025, Frerix et al. 2020, Huang et al. 2021, Wang et al. 2023a, Tordesillas et al. 2023, Konstantinov and Utkin 2023, Chen et al. 2024). Collectively, these developments demonstrate growing recognition that physical constraints should be explicitly accounted for in how AI models learn and generate predictions. Yet, while physical constraints have increasingly been incorporated into how models learn, their implications for how the influence of training data is measured remain largely unexplored.

Existing sensitivity analysis and interpretability methods, including perturbation-based approaches and IFs, have primarily been developed for unconstrained learning settings, where prediction sensitivity reflects how changes in input data propagate through the learned mapping from observations to outputs. However, when hard physical constraints are embedded into the learning architecture, the prediction response is no longer determined solely by learned data-driven relationships. Instead, it is shaped by the interaction between learned representations and the feasible space defined by physical constraints. As a result, perturbations in input data may influence predictions through both the learned model and the constraint-enforcing mechanism, creating new challenges for identifying the true sources of predictive information. Understanding which sensor observations contribute most to physically constrained predictions is critical for improving data quality management and guiding sensor maintenance, calibration, and investment decisions in transportation networks.

In summary, existing literature shows that spatiotemporal deep learning can deliver strong forecasting results. Hard constraints can ensure physical consistency by design. IFs can connect predictions to individual training observations. However, these three areas have not been linked. As shown in table 1, despite growing interest in IFs, limited attention has been given to integrating them with learning architectures that impose hard physical constraints, tracing training-data influence through constraint projection, or using the resulting influence scores to identify sensors that disproportionately afect predictions from physically constrained models.

To address those gaps, this paper introduces a hard physics-informed influence-function framework for learning under physical constraints. The proposed framework quantifies training-data influence within the physically feasible space through constraint projection, explicitly capturing how network-level physical constraints alter the magnitude, distribution, and value of information. We first develop a physics-informed spatiotemporal neural network for short-term trafic flow forecasting. The model combines data-driven learning with a Hard Constraint Projection layer. This layer projects raw neural network outputs onto a set defined by junction-level flow conservation. We then analyze how local perturbations in trafic observations afect the learned prediction. To do this, we create a hard physics-informed influence-function framework that improves on traditional influence analysis for constrained trafic learning. Unlike standard influence functions, which quantify training-data influence on the unconstrained model output, our framework evaluates influence on the physically feasible output obtained through constraint projection. As a result, the influence scores show how physical constraints change, reduce, or redistribute the impact of training samples across the network. Practically, finding these influence scores can help identify critical infrastructure in transportation networks.

In the rest of the paper, Section 2 explains our physics constrained influence analysis method, Section 3 describes the setup of numerical experiments in short-term trafic prediction context, Section 4 presents results, and Section 5 summarizes key conclusions and discusses possible future extensions.

Table 1 Related work organized by how physics is handled and how sensitivity is analyzed.
<table><tr><td rowspan="2">Physics treatment</td><td colspan="3">Sensitivity / data perturbation</td></tr><tr><td>No sensitivity consideration</td><td>Training behavior</td><td>Post-hoc attribution</td></tr><tr><td>No constraint</td><td>Liu et al. (2026) Ma et al. (2015) Fu et al. (2016) Li et al. (2018) Yu et al. (2018) Guo et al. (2019) Wang and Wang (2022)</td><td></td><td>Koh and Liang (2017)</td></tr><tr><td>Soft constraint</td><td>Márquez-Neila et al. (2017) Dener et al. (2020)</td><td>Wang et al. (2022) De Ryck et al. (2024) Daw et al. (2023) Wang et al. (2023b) Lu et al. (2021) Jagtap et al. (2020)</td><td>Krasowski et al. (2026)</td></tr><tr><td>Hard constraint</td><td>Frerix et al. (2020) Huang et al. (2021) Wang et al. (2023a) Tordesillas et al. (2023) Konstantinov and Utkin (2023) Chen et al. (2024) Min and Azizan (2025)</td><td></td><td>This work</td></tr></table>

## 2. Methodology

## 2.1. A Hard-Constrained Neural Network Formulation

Consider a hard-constrained neural network architecture, in which an unconstrained neural-network prediction is passed through a diferentiable enforcement layer before being used as the final model output. Let $\mathcal { X } \subseteq \mathbb { R } ^ { n _ { x } }$ denote the input space and let $f _ { \theta } : \mathcal { X }  \mathbb { R } ^ { n _ { y } }$ be a neural network parameterized by θ, where $n _ { x }$ is the input dimension and $n _ { y }$ is the output dimension. For an input $x \in \mathcal { X }$ , the raw neural-network prediction is denoted by $\hat { y } _ { \boldsymbol { \theta } } ( { \boldsymbol { x } } ) = f _ { \boldsymbol { \theta } } ( { \boldsymbol { x } } ) \in \mathbb { R } ^ { n _ { \boldsymbol { y } } }$ . In the hard-constrained setting, this raw prediction is not directly used as the final output because it may violate known constraints. Instead, the final prediction is obtained by applying a projection operator P to the raw output, yielding

$$
y _ { \theta } ^ { H } ( x ) = P ( f _ { \theta } ) ( x ) ,\tag{1}
$$

where $y _ { \theta } ^ { H } ( x )$ represents the hard-constrained prediction. The goal is to create an enforcement mapping $P ( f _ { \theta } ) : \mathcal { X }  \mathbb { R } ^ { n _ { \mathrm { y } } }$ that meets the specified constraints for every $x \in \mathcal { X }$ . Although the projection does not need to be globally smooth, being diferentiable almost everywhere allows gradients to propagate through the projection layer, thereby enabling end-to-end gradient-based training (Min and Azizan 2025). By construction, this hard-constrained model ensures that final output satisfies the constraints, unlike many physics-informed learning models that impose constraints softly through adding a penalty term to the training loss.

We consider the HardNet-Af formulation for enforcing input-dependent afine constraints (Min and Azizan 2025). Specifically, suppose the admissible output must satisfy

$$
b ^ { \ell } ( x ) \leq A ( x ) y \leq b ^ { u } ( x ) , \quad \forall x \in \mathcal { X } ,\tag{2}
$$

where $A ( x ) \in \mathbb { R } ^ { n _ { c } \times n _ { y } }$ is the constraint matrix, $b ^ { \ell } ( x ) , b ^ { u } ( x ) \in \mathbb { R } ^ { n _ { c } }$ are the lower and upper constraint limits, and $n _ { c }$ is the number of afine constraint rows. In transportation networks, this class of constraints includes, for example, flow-conservation constraints at a node $i ,$

$$
\sum _ { a \in \delta ^ { - } ( i ) } y _ { a } - \sum _ { a \in \delta ^ { + } ( i ) } y _ { a } = d _ { i } ,\tag{3}
$$

where $\delta ^ { - } ( i )$ and $\delta ^ { + } ( i )$ denote the sets of incoming and outgoing links, respectively, and $d _ { i }$ represents the net flow supply or demand at node i. Such an equality can be represented in the two-sided inequality form by setting the corresponding lower and upper bounds equal, $b _ { i } ^ { \ell } ( x ) = b _ { i } ^ { u } ( x ) = d _ { i }$ Similarly, a link-capacity constraint for link a can be written as $0 \leq y _ { a } \leq c _ { a } ( x )$ . Where $c _ { a } ( x )$ denotes the available capacity of link a. Thus, both network flow-conservation and capacity constraints can be represented within the general afine constraint form considered here. Inequalities are understood component-wise. The feasible set associated with input x is therefore:

$$
\begin{array} { r } { \mathcal { C } ( \boldsymbol { x } ) = \left\{ \boldsymbol { y } \in \mathbb { R } ^ { n _ { \boldsymbol { y } } } : b ^ { \ell } ( \boldsymbol { x } ) \leq A ( \boldsymbol { x } ) \boldsymbol { y } \leq b ^ { u } ( \boldsymbol { x } ) \right\} . } \end{array}\tag{4}
$$

The HardNet-Af projection maps the raw neural-network output $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ to a feasible point in $\mathcal C ( x )$ through the closed-form correction

$$
P ( f _ { \theta } ) ( x ) = f _ { \theta } ( x ) + A ( x ) ^ { + } \bigl [ \mathrm { R e L U } \bigl ( b ^ { \ell } ( x ) - A ( x ) f _ { \theta } ( x ) \bigr ) - \mathrm { R e L U } ( A ( x ) f _ { \theta } ( x ) - b ^ { u } ( x ) ) \bigr ] ,\tag{5}
$$

where $\mathrm { R e L U } ( r ) = \operatorname* { m a x } \{ 0 , r \}$ is applied component-wise and

$$
A ( x ) ^ { + } = A ( x ) ^ { \top } \big ( A ( x ) A ( x ) ^ { \top } \big ) ^ { - 1 }\tag{6}
$$

is the right pseudoinverse of $A ( x )$ . This expression assumes that $A ( x )$ has full row rank for each $x \in \mathcal { X }$ , so that $A ( x ) A ( x ) ^ { \top }$ is invertible, and that $\mathcal C ( x )$ is nonempty for every input.

A useful special case arises when the constraints are equalities of the form

$$
A ( x ) y = b ( x ) .\tag{7}
$$

Unlike (5), whose ReLU-based correction introduces non-diferentiability, the following equality projection in (8) is afine in $f _ { \theta } ( x )$ for fixed x and therefore here smoothness is preserved. In this case, (5) simplifies to

$$
P ( f _ { \theta } ) ( x ) = f _ { \theta } ( x ) - A ( x ) ^ { + } ( A ( x ) f _ { \theta } ( x ) - b ( x ) ) ,\tag{8}
$$

![](images/40d5a2024d926c4e5baa56e543eb9b325816fd49a710459452b407bfa3c63f14.jpg)  
Figure 1 Schematic illustration of the HardNet-Af framework, adapted from Min and Azizan Min and Azizan (2025). A standard neural network first maps the input x to an unconstrained prediction $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ . The prediction is then passed through a diferentiable constraint-enforcement layer, where the afine constraint set $\mathcal C ( x )$ is defined for the given input and the projection operator $\mathcal { P }$ maps the raw output to a feasible point. The final projected output satisfies $\mathcal { P } ( f _ { \theta } ) ( x ) \in \mathcal { C } ( x )$ by construction, while gradients can still be backpropagated through the projection layer during end-to-end training.

and under the full-row-rank assumption, $A ( x ) A ( x ) ^ { + } = I _ { n _ { c } } ,$ so that the projected output satisfies $A ( x ) y _ { \theta } ^ { H } ( x ) = b ( x )$ exactly for every $x \in \mathcal { X }$ . Even though this study does not deal with general nonlinear constraints, our formulation does extend the enforcement of hard constraints to cases involving input-dependent afine constraints with exact feasibility. Figure 1 explains how the projection will be applied in our framework.

## 2.2. Hard Physics-Informed Influence Functions

Influence functions are used to quantify how each training sample afects the learned model and its physically feasible predictions. Let $D = \{ z _ { i } = ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ be the training dataset, where $x _ { i } \in \mathcal X$ and $y _ { i } \in \mathbb { R } ^ { m }$ . Since the final model output is $y _ { \boldsymbol { \theta } } ^ { H } ( x )$ , the sample-wise hard physics-informed loss is defined as

$$
\ell ^ { H } ( z _ { i } ; \theta ) = \ell \bigl ( y _ { \theta } ^ { H } ( x _ { i } ) , y _ { i } \bigr ) ,\tag{9}
$$

where $\ell ( \cdot , \cdot )$ is the prediction loss function. The empirical hard-constrained risk and the trained parameters are

$$
L ^ { H } ( \theta ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell ^ { H } ( z _ { i } ; \theta )\tag{10}
$$

To measure the contribution of training sample $z _ { i } .$ , we consider an infinitesimal upweighting of its loss term and define the perturbed objective as

$$
L _ { \epsilon , i } ^ { H } ( \theta ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \ell ^ { H } ( z _ { j } ; \theta ) + \epsilon \ell ^ { H } ( z _ { i } ; \theta ) .\tag{11}
$$

Henceforth, we assume that the relevant local minimizer is unique in a neighborhood of ${ \widehat { \theta } } ,$ and denote the corresponding perturbed solution by $\widehat { \theta } _ { \epsilon , i }$ . The parameter-level influence is the first-order change $d \hat { \theta } _ { \epsilon , i } / d \epsilon | _ { \epsilon = 0 } $ , approximated by

$$
\boldsymbol { \mathcal { Z } } _ { \mathrm { p a r a m } } ^ { H } ( \boldsymbol { z } _ { i } ) = - H _ { H } ^ { - 1 } \nabla _ { \boldsymbol { \theta } } \ell ^ { H } ( \boldsymbol { z } _ { i } ; \boldsymbol { \hat { \boldsymbol { \theta } } } ) , \qquad \boldsymbol { H } _ { H } = \nabla _ { \boldsymbol { \theta } } ^ { 2 } L ^ { H } ( \boldsymbol { \theta } ) \big | _ { \boldsymbol { \theta = \hat { \boldsymbol { \theta } } } } ,\tag{12}
$$

where $H _ { H }$ is the Hessian of the empirical hard-constrained risk evaluated at ${ \hat { \theta } } .$ This first-order approximation relies on the standard assumptions of the influence function that $L ^ { H } ( \theta )$ is twice continuously diferentiable in a neighborhood of $ { \hat { \theta } } , H _ { H }$ is non-singular, and the perturbation ϵ is infinitesimal; see, e.g., Royset and Wets (2021) for general subdiferential calculus rules.

For a test sample ${ \boldsymbol { z } } _ { t } = ( x _ { t } , y _ { t } )$ , the hard physics-informed influence of training sample $z _ { i }$ on the test loss at $z _ { t }$ is

$$
\begin{array} { r } { \mathcal { T } ^ { H } ( { z } _ { i } , { z } _ { t } ) = - \nabla _ { \boldsymbol { \theta } } \ell ^ { H } ( { z } _ { t } ; \hat { \boldsymbol { \theta } } ) ^ { \top } H _ { H } ^ { - 1 } \nabla _ { \boldsymbol { \theta } } \ell ^ { H } ( { z } _ { i } ; \hat { \boldsymbol { \theta } } ) . } \end{array}\tag{13}
$$

Defining $g _ { i } ^ { H } = \nabla _ { \boldsymbol { \theta } } \ell ^ { H } ( z _ { i } ; \boldsymbol { \hat { \theta } } )$ and $g _ { t } ^ { H } = \nabla _ { \boldsymbol { \theta } } \ell ^ { H } \big ( z _ { t } ; \hat { \boldsymbol { \theta } } \big )$ , this becomes $\begin{array} { r } { \mathcal { T } ^ { H } ( z _ { i } , z _ { t } ) = - ( g _ { t } ^ { H } ) ^ { \top } H _ { H } ^ { - 1 } g _ { i } ^ { H } } \end{array}$ . A positive value indicates that upweighting $z _ { i }$ increases the hard-constrained test loss at $z _ { t } ;$ a negative value indicates it decreases it.

The main distinction from the classical influence function (Koh and Liang 2017) is that all derivatives are computed through the hard-constrained prediction map. By the chain rule,

$$
\nabla _ { \boldsymbol { \theta } } \ell ^ { H } ( z _ { i } ; \boldsymbol { \theta } ) = \frac { \partial \ell } { \partial y _ { \boldsymbol { \theta } } ^ { H } } \cdot \frac { \partial y _ { \boldsymbol { \theta } } ^ { H } } { \partial \hat { y } _ { \boldsymbol { \theta } } } \cdot \frac { \partial \hat { y } _ { \boldsymbol { \theta } } } { \partial \boldsymbol { \theta } } ,\tag{14}
$$

so the Jacobian $\partial y _ { \boldsymbol { \theta } } ^ { H } / \partial \hat { y } _ { \boldsymbol { \theta } }$ includes the impact of the projection layer in all calculations of influence. Consequently, $\mathcal { T } ^ { H } ( z _ { i } , z _ { t } )$ measures the sensitivity of the physically feasible prediction, not the unconstrained neural-network output.

More generally, the same framework evaluates the influence of a training sample on any diferentiable scalar quantity $\psi ^ { H } ( x _ { t } ; \theta )$ derived from the hard-constrained prediction:

$$
\boldsymbol { \mathcal { T } } _ { \boldsymbol { \psi } } ^ { H } ( \boldsymbol { z } _ { i } , \boldsymbol { x } _ { t } ) = - \nabla _ { \boldsymbol { \theta } } \boldsymbol { \psi } ^ { H } ( \boldsymbol { x } _ { t } ; \hat { \boldsymbol { \theta } } ) ^ { \top } \boldsymbol { H } _ { H } ^ { - 1 } \nabla _ { \boldsymbol { \theta } } \boldsymbol { \ell } ^ { H } ( \boldsymbol { z } _ { i } ; \hat { \boldsymbol { \theta } } ) .\tag{15}
$$

For a group of training samples $G \subseteq \{ 1 , \ldots , N \}$ , the group influence on $z _ { t }$ is

$$
{ \mathcal { T } } ^ { H } ( G , z _ { t } ) = - \left( g _ { t } ^ { H } \right) ^ { \top } H _ { H } ^ { - 1 } \sum _ { i \in G } g _ { i } ^ { H } .\tag{16}
$$

This group formulation supports sensitivity analysis over subsets of the training data such as spatial regions, time periods, operating conditions, or physical regimes.

## 2.3. Null-Space Decomposition and the Physical Influence Purity Indicator

For the equality constraints in (7), the projection in (8) specializes to

$$
\begin{array} { r } { { y } _ { \theta } ^ { H } ( X _ { t } ) = \hat { y } _ { \theta } ( X _ { t } ) - A _ { c } ^ { + } ( A _ { c } \hat { y } _ { \theta } ( X _ { t } ) - b _ { c } ) . } \end{array}\tag{17}
$$

Assuming that $A _ { c } \in \mathbb { R } ^ { r \times m }$ has full row rank, where m is the number of monitored links and $r$ the number of conservation equations, its right pseudoinverse is $A _ { c } ^ { + } = A _ { c } ^ { top } ( A _ { c } A _ { c } ^ { \top } ) ^ { - 1 }$ . The corresponding projection Jacobian, obtained from (14), is

$$
\frac { \partial y _ { \theta } ^ { H } } { \partial \hat { y } _ { \theta } } = I _ { m } - A _ { c } ^ { + } A _ { c } = I _ { m } - A _ { c } ^ { \top } \left( A _ { c } A _ { c } ^ { \top } \right) ^ { - 1 } A _ { c } = : \Pi _ { c } .\tag{18}
$$

Here, $\Pi _ { c } \in \mathbb { R } ^ { m \times m }$ is the orthogonal projector onto $\mathrm { \ n u l l } ( A _ { c } )$ , representing directions that do not afect the satisfaction of conservation constraints. Meanwhile, $I _ { m } - \Pi _ { c } = A _ { c } ^ { + } A _ { c }$ projects onto the constraint-related directions. This represents the component that is adjusted to enforce conservation constraints. The projector $\Pi _ { c }$ is symmetric, idempotent, and depends only on the constraint matrix $A _ { c } ;$ once the trafic network and conservation constraints are specified, $\Pi _ { c }$ remains fixed during training.

Substituting Equation (18) into Equation (14), the gradient of the hard-constrained loss with respect to the raw prediction is

$$
\nabla _ { \boldsymbol { \hat { y } } } \ell ^ { H } ( \boldsymbol { z } _ { i } ; \boldsymbol { \hat { \theta } } ) = \Pi _ { c } \delta _ { i } , \qquad \delta _ { i } : = \nabla _ { \boldsymbol { y } ^ { H } } \ell \big ( \boldsymbol { y } _ { \boldsymbol { \hat { \theta } } } ^ { H } ( \boldsymbol { X } _ { i } ) , \boldsymbol { y } _ { i } \big ) \in \mathbb { R } ^ { m } ,\tag{19}
$$

where $z _ { i } = ( X _ { i } , y _ { i } )$ and $\delta _ { i }$ is the gradient of the prediction loss with respect to the hard-constrained output at training sample $z _ { i }$ . Since $\Pi _ { c }$ is an orthogonal projector, any $\delta _ { i } \in \mathbb { R } ^ { m }$ admits the unique orthogonal decomposition

$$
\delta _ { i } = \underbrace { \Pi _ { c } \delta _ { i } } _ { \delta _ { i } ^ { \mathrm { p h y s } } } + \underbrace { ( I _ { m } - \Pi _ { c } ) \delta _ { i } } _ { \delta _ { i } ^ { \mathrm { v i o l } } } ,\tag{20}
$$

where $\delta _ { i } ^ { \mathrm { p h y s } }$ is the conservation-consistent component and $\delta _ { i } ^ { \mathrm { v i o l } }$ is the conservation-violating component. This interpretation follows directly from the geometry of the feasible set. If y is feasible to satisfy $A _ { c } y = b _ { c } ,$ for any $v \in \mathrm { n u l l } ( A _ { c } )$ we have $A _ { c } ( y + \eta v ) = b _ { c }$ for all $\eta \in \mathbb { R }$ . Thus, moving in a null-space direction does not afect conservation. For a nonzero $v \in \operatorname { r o w } ( A _ { c } )$ , we can write $v = A _ { c } ^ { \top } w$ Since $A _ { c }$ has full row rank, $A _ { c } A _ { c } ^ { \top }$ is nonsingular, and therefore $A _ { c } v \neq 0$ . Hence, for $\eta \neq 0$ , moving in this direction changes the constraint residual. The decomposition in (20) therefore separates the gradient into directions that are compatible with the conservation constraints and directions that are removed by the projection. The orthogonality of these two components yields

$$
\lVert \delta _ { i } \rVert _ { 2 } ^ { 2 } = \lVert \delta _ { i } ^ { \mathrm { p h y s } } \rVert _ { 2 } ^ { 2 } + \lVert \delta _ { i } ^ { \mathrm { v i o l } } \rVert _ { 2 } ^ { 2 } .\tag{21}
$$

Equation (19) shows that only $\delta _ { i } ^ { \mathrm { p h y s } }$ contributes to the gradient with respect to the raw prediction. The component $\delta _ { i } ^ { \mathrm { v i o l } }$ is in the row space of $A _ { c }$ and is eliminated by the projection Jacobian $\Pi _ { c }$ Therefore, under the projected loss, this component does not propagate to the model parameters through the hard-constrained prediction map.

The decomposition also shows how much of a sample gradient is retained after the constraint projection. To compare this efect across samples, the measure should reflect the gradient’s alignment with null $\left( A _ { c } \right)$ rather than its magnitude; otherwise, samples with larger residuals would dominate the comparison. A common range is also useful when comparing diferent time periods or network locations. Since $\Pi _ { c }$ acts on all m output components at once, the measure should be based on the full gradient vector.

The orthogonal decomposition in (21) gives this quantity directly. So the retained component can be normalized by the total gradient norm. The resulting ratio is independent of gradient magnitude, lies in [0, 1], and measures the share of the squared gradient norm that is aligned with conservationconsistent directions.

Definition 1 (Physical Influence Purity) The Physical Influence Purity (PIP) of training sample $z _ { i }$ is defined as

$$
\mathrm { P I P } ( z _ { i } ) = \frac { | | \delta _ { i } ^ { \mathrm { p h y s } } | | _ { 2 } ^ { 2 } } { | | \delta _ { i } | | _ { 2 } ^ { 2 } + \varepsilon } = \frac { | | \Pi _ { c } \delta _ { i } | | _ { 2 } ^ { 2 } } { | | \delta _ { i } | | _ { 2 } ^ { 2 } + \varepsilon } ,\tag{22}
$$

where $\varepsilon > 0$ is a small constant used to avoid division by zero. When $\Vert \delta _ { i } \Vert _ { 2 } > 0$ $\operatorname { P I P } ( z _ { i } ) \in [ 0 , 1 ]$

PIP is bounded in [0, 1] directly from the orthogonal decomposition in (21). In particular, both $\| \delta _ { i } ^ { \mathrm { p h y s } } \| _ { 2 } ^ { 2 }$ and $\| \delta _ { i } ^ { \mathrm { v i o l } } \| _ { 2 } ^ { 2 }$ are non-negative. Therefore, $\| \delta _ { i } ^ { \mathrm { p h y s } } \| _ { 2 } ^ { 2 }$ cannot exceed $\lVert \delta _ { i } \rVert _ { 2 } ^ { 2 }$ , and the ratio in (22) lies between zero and one. Squared norms are used because (21) gives an additive decomposition of the squared gradient magnitude. Thus, PIP represents the fraction of the total squared gradient magnitude that remains in conservation-consistent directions after projection.

A value close to $\mathrm { P I P } ( z _ { i } ) = 1$ means that the loss-gradient direction is nearly aligned with null $\left( A _ { c } \right)$ Thus, the change indicated by the gradient is largely compatible with the conservation constraints. In contrast, a value close to $\mathrm { P I P } ( z _ { i } ) = 0$ means that the gradient lies mainly in row $\left( A _ { c } \right)$ . This component is removed when the gradient is multiplied by the projection matrix $\Pi _ { c }$ and therefore does not contribute to updates through the prediction.

## 2.4. Practical Computation of Influence Functions

The influence score in Equation (13) requires computing $H _ { H } ^ { - 1 } \mathbf { g } _ { t } ^ { H }$ , where $H _ { H } = \nabla _ { \boldsymbol { \theta } } ^ { 2 } L ^ { H } ( \hat { \boldsymbol { \theta } } ) \in \mathbb { R } ^ { p \times p }$ is the Hessian of the hard-constrained empirical loss. With a high number of trainable parameters, forming $H _ { H }$ is computationally infeasible, and direct inversion costs $O ( p ^ { 3 } )$ operations. The key observation is that we never need $H _ { H } ^ { - 1 }$ as an explicit matrix. For a fixed test window $z _ { t } ,$ every influence score $\begin{array} { r } { \mathbb { Z } ^ { H } \left( z _ { i } , z _ { t } \right) = - s _ { t } ^ { \top } \mathbf { g } _ { i } ^ { H } } \end{array}$ reduces to a dot product once the inverse Hessian-vector product (iHVP)

$$
s _ { t } = H _ { H } ^ { - 1 } \mathbf { g } _ { t } ^ { H }\tag{23}
$$

is available. The computational task therefore reduces to solving the linear system $H _ { H } s _ { t } = \mathbf { g } _ { t } ^ { H }$ without ever forming $H _ { H }$

The iterative solver we adopt requires only Hessian-vector products (HVPs), that is, products of the form $H _ { H } v$ for arbitrary $v \in \mathbb { R } ^ { p }$ . By the Pearlmutter identity (Pearlmutter 1994), this product is equal to a gradient:

$$
H _ { H } \boldsymbol { v } = \nabla _ { \boldsymbol { \theta } } \Big ( \nabla _ { \boldsymbol { \theta } } L ^ { H } ( \hat { \boldsymbol { \theta } } ) ^ { \top } \boldsymbol { v } \Big ) .\tag{24}
$$

The right-hand side is the gradient of the scalar $\nabla _ { \theta } L ^ { H } \cdot \boldsymbol { v }$ with respect to $\theta .$ This costs $O ( p )$ time and memory with no need to store $H _ { H }$ . A significant structural outcome is that every Hessian-vector product is automatically physics-informed. By the chain rule, the gradient of the hard-constrained loss satisfies:

$$
\mathbf { g } ^ { H } = \nabla _ { \boldsymbol { \theta } } \ell ^ { H } = \frac { \partial \ell } { \partial y ^ { H } } \cdot \boldsymbol { \Pi _ { c } } \cdot \frac { \partial \hat { y } } { \partial \boldsymbol { \theta } } ,\tag{25}
$$

so the conservation projection $\Pi _ { c }$ acts on every gradient before it enters the HVP. Diferentiating (25) again, the Hessian separates into a Gauss-Newton term of the form $J ^ { \top } \Pi _ { c } \bigl ( \nabla _ { y ^ { H } } ^ { 2 } \ell \bigr ) \Pi _ { c } J$ . Here, $J =$ $\partial \hat { y } / \partial \theta$ , and there is also a term that involves the second derivative of the network output that is weighted by $\Pi _ { c } \delta .$ In the Gauss-Newton term, any change in parameters that would push the raw prediction in a direction that violates conservation is canceled by $\Pi _ { c }$ and therefore does not contribute to the curvature term.

To approximate $s _ { t }$ iteratively, we rely on the Neumann series identity. For any step size α satisfying $0 < \alpha < 2 / \rho ( H _ { H } )$ , the inverse admits the convergent series:

$$
H _ { H } ^ { - 1 } = \alpha \sum _ { k = 0 } ^ { \infty } ( I - \alpha H _ { H } ) ^ { k } ,\tag{26}
$$

which holds because all eigenvalues of $\left( I - \alpha H _ { H } \right)$ are strictly inside the unit circle under the stated condition on $\alpha .$ Multiplying both sides of (26) by $\mathbf { g } _ { t } ^ { H }$ and identifying the partial sums with an iterative update leads to the LiSSA recursion (Agarwal et al. 2017):

$$
h _ { 0 } = \alpha { \bf g } _ { t } ^ { H } , \qquad h _ { k + 1 } = \alpha { \bf g } _ { t } ^ { H } + h _ { k } - \alpha H _ { H } h _ { k } ,\tag{27}
$$

where $h _ { K }  s _ { t }$ exactly as $K  \infty$ . Each step requires exactly one HVP evaluation.

For our model, however, the naïve recursion (27) converges very slowly in practice. A large spectral radius $\rho ( H _ { H } )$ , restricts the admissible step size $\alpha ,$ while near-zero Hessian eigenvalues lead to poor

conditioning. The solution for this problem is to normalize $H _ { H }$ by its spectral radius before applying the Neumann series. Define the scaled and damped Hessian:

$$
\widetilde H _ { d } = \frac { H _ { H } } { \rho ( H _ { H } ) } + d I ,\tag{28}
$$

where $d > 0$ is a small damping constant. Dividing by $\rho ( H _ { H } )$ maps all eigenvalues of $H _ { H }$ into $[ 0 , 1 ]$ , so $\widetilde { H } _ { d }$ has eigenvalues in $[ d , 1 + d ]$ and is positive definite. The Neumann series for $\widetilde { H } _ { d } ^ { - 1 }$ then converges with the step size $\alpha = 1$ , which yields the update rule:

$$
h _ { 0 } = { \bf g } _ { t } ^ { H } , \qquad h _ { k + 1 } = { \bf g } _ { t } ^ { H } + \left( 1 - d \right) h _ { k } - \frac { 1 } { \rho ( H _ { H } ) } H _ { H } h _ { k } .\tag{29}
$$

If $\lambda _ { i }$ denotes the i-th eigenvalue of $H _ { H }$ , the corresponding eigenvalue of the iteration matrix $( I - \widetilde { H } _ { d } )$ is:

$$
\mu _ { i } = 1 - \frac { \lambda _ { i } } { \rho ( H _ { H } ) } - d \in [ - d , 1 - d ] ,\tag{30}
$$

where we assume $H _ { H } \succeq 0$ , so that $\lambda _ { i } \in \left[ 0 , \rho ( H _ { H } ) \right]$ and hence $\lambda _ { i } / \rho ( H _ { H } ) \in [ 0 , 1 ]$ ]. The spectral radius of the iteration matrix is therefore:

$$
\rho ( I - \widetilde { H } _ { d } ) = \operatorname* { m a x } _ { i } \vert \mu _ { i } \vert = \operatorname* { m a x } ( d , 1 - d ) = 1 - d .\tag{31}
$$

Theorem 1 (Convergence of Scaled LiSSA) Let $h ^ { * } = \widetilde { H } _ { d } ^ { - 1 } \mathbf { g } _ { t } ^ { H }$ denote the unique solution to the scaled linear system $\widetilde { \cal H } _ { d } h = { \bf g } _ { t } ^ { H }$ . Starting from $h _ { 0 } = \mathbf { g } _ { t } ^ { H }$ , the iterate $h _ { K }$ produced by (29) satisfies:

$$
\begin{array} { r } { \| h _ { K } - h ^ { * } \| \leq ( 1 - d ) ^ { K } \| h _ { 0 } - h ^ { * } \| . } \end{array}\tag{32}
$$

The rate $( 1 - d ) ^ { K }$ depends only on d and K, not on $\rho ( H _ { H } )$ or $\kappa ( H _ { H } )$

Proof. Step 1: Identify the fixed point. The update rule (29) is a linear map in $h _ { k }$ . A fixed point $h ^ { * }$ satisfies $h _ { k + 1 } = h _ { k } = h ^ { * }$ , so substituting into (29):

$$
h ^ { \ast } = \mathbf { g } _ { t } ^ { H } + \left( 1 - d \right) h ^ { \ast } - \frac { 1 } { \rho ( H _ { H } ) } H _ { H } h ^ { \ast } .\tag{33}
$$

Collecting all terms involving $h ^ { * }$ on the left:

$$
h ^ { \ast } - \left( 1 - d \right) h ^ { \ast } + \frac { 1 } { \rho ( H _ { H } ) } H _ { H } h ^ { \ast } = { \bf g } _ { t } ^ { H } .\tag{34}
$$

Simplifying the left side:

$$
d h ^ { * } + \frac { H _ { H } } { \rho ( H _ { H } ) } h ^ { * } = { \bf g } _ { t } ^ { H } .\tag{35}
$$

Factoring out $h ^ { * }$ and recognizing the definition (28):

$$
\underbrace { \left( \frac { H _ { H } } { \rho ( H _ { H } ) } + d I \right) } _ { \widetilde { H } _ { d } } h ^ { * } = \mathbf { g } _ { t } ^ { H } .\tag{36}
$$

Since $\widetilde { H } _ { d }$ is positive definite it is invertible, and the unique fixed point is:

$$
h ^ { * } = \widetilde { H } _ { d } ^ { - 1 } \mathbf { g } _ { t } ^ { H } .\tag{37}
$$

Step 2: Derive the error recursion. The update rule (29) can be written compactly as:

$$
h _ { k + 1 } = { \bf g } _ { t } ^ { H } + ( I - \widetilde { H } _ { d } ) h _ { k } .\tag{38}
$$

The fixed point equation (36) equivalently reads:

$$
h ^ { * } = { \mathbf { g } } _ { t } ^ { H } + \left( I - \widetilde { H } _ { d } \right) h ^ { * } .\tag{39}
$$

Define the error $e _ { k } = h _ { k } - h ^ { * }$ . Subtracting (39) from (38):

$$
\begin{array} { r l } & { e _ { k + 1 } = h _ { k + 1 } - h ^ { * } } \\ & { \qquad = \left[ \mathbf { g } _ { t } ^ { H } + \left( I - \widetilde { H } _ { d } \right) h _ { k } \right] - \left[ \mathbf { g } _ { t } ^ { H } + \left( I - \widetilde { H } _ { d } \right) h ^ { * } \right] } \\ & { \qquad = \left( I - \widetilde { H } _ { d } \right) \left( h _ { k } - h ^ { * } \right) } \\ & { \qquad = \left( I - \widetilde { H } _ { d } \right) e _ { k } . } \end{array}\tag{40}
$$

Step 3: Apply the recursion K times. Equation (40) shows that the error is multiplied by $( I - \widetilde { H } _ { d } )$ at every step. Applying it repeatedly:

$$
e _ { K } = ( I - \widetilde { H } _ { d } ) ^ { K } e _ { 0 } .\tag{41}
$$

Step 4: Bound the norm. Since $H _ { H }$ is symmetric and I is symmetric, the iteration matrix $I - \widetilde { H } _ { d }$ is symmetric, hence normal, so its spectral norm equals its spectral radius: $\| I - \widetilde { H } _ { d } \| _ { 2 } =$ $\rho ( I - \widetilde { H } _ { d } )$ . Taking the Euclidean norm of both sides of (41) and applying this fact together with $\rho ( A ^ { K } ) = \rho ( A ) ^ { K }$ :

$$
\lvert | e _ { K } \rvert | = \lvert | ( I - \widetilde { H } _ { d } ) ^ { K } e _ { 0 } \rvert | \leq \rho \Big ( ( I - \widetilde { H } _ { d } ) ^ { K } \Big ) \lvert | e _ { 0 } \rvert | = \rho ( I - \widetilde { H } _ { d } ) ^ { K } \lvert | e _ { 0 } \rvert | .\tag{42}
$$

Substituting (31):

$$
\begin{array} { r } { \| h _ { K } - h ^ { * } \| \leq ( 1 - d ) ^ { K } \| h _ { 0 } - h ^ { * } \| . } \end{array}
$$

It is worth being precise about what the scaled LiSSA computes relative to the exact iHVP. From equation (36), the fixed point $h ^ { * }$ solves $\widetilde { \cal H } _ { d } h ^ { * } = \mathbf { g } _ { t } ^ { H }$ . Substituting the definition (28) and multiplying both sides by $\rho ( H _ { H } )$ :

$$
\left( { \frac { H _ { H } } { \rho ( H _ { H } ) } } + d I \right) h ^ { * } = \mathbf { g } _ { t } ^ { H } \implies \left( H _ { H } + \lambda I \right) h ^ { * } = \rho ( H _ { H } ) \mathbf { g } _ { t } ^ { H } ,\tag{43}
$$

where $\lambda = d \cdot \rho ( H _ { H } )$ is the efective regularization. Solving for $h ^ { * }$ gives $h ^ { \ast } = \rho ( H _ { H } ) ( H _ { H } + \lambda I ) ^ { - 1 } \mathbf { g } _ { t } ^ { H }$ Substituting into the influence score:

$$
\mathcal { Z } _ { \lambda } ^ { H } ( z _ { i } , z _ { t } ) = - ( \mathbf { g } _ { t } ^ { H } ) ^ { \top } ( H _ { H } + \lambda I ) ^ { - 1 } \mathbf { g } _ { i } ^ { H } = - \frac { ( h ^ { \ast } ) ^ { \top } \mathbf { g } _ { i } ^ { H } } { \rho ( H _ { H } ) } .\tag{44}
$$

This is the regularized influence function, which replaces the exact $H _ { H } ^ { - 1 }$ with $( H _ { H } + \lambda I ) ^ { - 1 }$ . Convergence is true for any $d > 0$ according to Theorem 1. The regularization $\lambda = d \cdot \rho ( H _ { H } )$ reflects the cost of a contraction rate that does not depend on the conditioning of $H _ { H }$ . This regularization can be made tiny, but this will require more iterations. Since $\rho ( H _ { H } )$ is a global constant shared across all pairs $( z _ { i } , z _ { t } )$ , it cancels upon normalization and does not afect the relative ranking of training samples.

## 2.5. Station Sensitivity Score

The influence score $\mathcal { T } ^ { H } ( z _ { i } , z _ { t } )$ ranks training samples. To measure how sensitive the forecast at each detector station is, we define a score for each station that uses the $s _ { t }$ from (23). For a test window $z _ { t }$ with input $X _ { t } ,$ let

$$
\hat { s } _ { t } = \frac { s _ { t } } { \left\| s _ { t } \right\| }\tag{45}
$$

be the unit direction of the iHVP, and let $y _ { e } ^ { H } ( X _ { t } ; \hat { \theta } )$ be the physics-feasible forecast at station $e \in \{ 1 , \ldots , N \}$ . The Station Sensitivity Score (SSS) of station e is

$$
\mathrm { S S S } _ { e } = \frac { 1 } { | T | } \sum _ { z _ { t } \in \mathcal { T } } \left| \boldsymbol { \hat { s } } _ { t } ^ { \top } \nabla _ { \theta } \boldsymbol y _ { e } ^ { H } ( \boldsymbol X _ { t } ; \hat { \theta } ) \right| ,\tag{46}
$$

where $\tau$ is the set of test windows. The term inside the absolute value is the rate of change of the forecast at station e when the parameters move along $\hat { s } _ { t } . \mathrm { ~ A ~ }$ large SSS means that the forecast at station e responds strongly to the parameter changes that matter most for the test windows. For comparison across stations, the scores are scaled.

## 3. Numerical Experiments

## 3.1. Trafic Flow Prediction Problem Setup

We formulate trafic flow forecasting on a directed trafic network $\mathcal { G } = ( \nu , \mathcal { E } )$ , where V is the set of junctions and ${ \mathcal { E } } = \{ 1 , \ldots , m \}$ is the set of monitored directed links. At time t, the feature vector of link $e \in { \mathcal { E } }$ is $s _ { e } ( t ) = [ q _ { e } ( t ) , v _ { e } ( t ) , o _ { e } ( t ) ] ^ { \top }$ , where $q _ { e } ( t ) , ~ v _ { e } ( t )$ , and $o _ { e } ( t )$ denote flow, speed, and occupancy, respectively. Stacking all link-level features gives the network trafic-state matrix $S ( t ) \in$ $\mathbb { R } ^ { m \times 3 }$

To capture temporal dependence, we use a rolling time window of length w. The input sample at time t is $X _ { t } = \{ S ( t - w + 1 ) , \ldots , S ( t ) \} \in \mathbb { R } ^ { w \times m \times 3 }$ , and the prediction target is the next-hour flow vector $y _ { t } = [ q _ { 1 } ( t + 1 ) , \ldots , q _ { m } ( t + 1 ) ] ^ { \intercal } \in \mathbb { R } ^ { m }$ . The raw neural-network prediction is $\hat { y } _ { \theta } ( X _ { t } ) = f _ { \theta } ( X _ { t } )$

where $f _ { \theta } : \mathbb { R } ^ { w \times m \times 3 }  \mathbb { R } ^ { m }$ is a spatiotemporal forecasting model, and the final prediction is the hard-constrained output $y _ { \theta } ^ { H } ( X _ { t } )$

The physical constraint considered in this work is junction-level flow conservation. Let $\mathcal { V } _ { c } \subseteq \mathcal { V }$ denote the set of junctions where conservation is imposed. For each $u \in \mathcal { V } _ { c }$ , let $\mathcal { T } ( u ) \subseteq \mathcal { E }$ and $\mathcal { O } ( u ) \subseteq \mathcal { E }$ be the sets of incoming and outgoing monitored links. The predicted flows are required to satisfy

$$
\sum _ { e \in \mathcal { T } ( u ) } y _ { e } - \sum _ { e \in \mathcal { O } ( u ) } y _ { e } = b _ { c , u } ,\tag{47}
$$

where $b _ { c , u }$ is the net imbalance at junction u. For example, if the time window is suficient to clear the trafic, b could be zero. Stacking over all constrained junctions gives the equality constraint

$$
\begin{array} { r } { A _ { c } y = b _ { c } , } \end{array}\tag{48}
$$

where $A _ { c } \in \mathbb { R } ^ { | \mathcal { V } _ { c } | \times m }$ is the junction–link conservation matrix with entries

$$
( A _ { c } ) _ { u , e } = \left\{ \begin{array} { r l } { { 1 , } } & { { \mathrm { i f ~ l i n k ~ } e \mathrm { ~ e n t e r s ~ j u n c t i o n ~ } u , } } \\ { { - 1 , } } & { { \mathrm { i f ~ l i n k ~ } e \mathrm { ~ l e a v e s ~ j u n c t i o n ~ } u , } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e , } } } \end{array} \right.\tag{49}
$$

and $b _ { c } \in \mathbb { R } ^ { | \nu _ { c } | }$ is the vector of net junction imbalances. Using the HardNet-Af projection from Section 2.1, the final forecast satisfies $A _ { c } y _ { \theta } ^ { H } ( X _ { t } ) = b _ { c }$ for every input window $X _ { t }$

Given the training dataset $D = \{ ( X _ { t } , y _ { t } ) \} _ { t = w } ^ { T - 1 }$ , the model is trained by minimizing

$$
\operatorname* { m i n } _ { \theta } \frac { 1 } { T - w } \sum _ { t = w } ^ { T - 1 } \ell \big ( y _ { \theta } ^ { H } ( X _ { t } ) , y _ { t } \big ) ,\tag{50}
$$

where $\ell ( \cdot , \cdot )$ is the forecasting loss function.

## 3.2. Experimental Setup

We use hourly PeMS District 3 data covering 318 monitored freeway links across the Sacramento metropolitan network from January to May 2025, yielding $T = 3 , 6 2 3$ hourly observations. A map of the study area is provided in the Results section (Figure 3). Each link carries three features - flow, speed, and occupancy - and the prediction target is the next-hour flow vector $y _ { t } \in \mathbb { R } ^ { 3 1 8 }$ . Using a 24-hour input window, the dataset produces 3,599 input-target pairs, split chronologically into 70% training, 15% validation, and 15% test. Features are standardized using training-set statistics; targets are retained in original units (vehicles per hour).

We construct a node-link graph from corridor topology, generating 312 candidate junction flow conservation equations. Each equation is validated empirically: a constraint is retained only if its mean relative residual does not exceed 10% and its 95th-percentile relative residual does not exceed 30% of total involved flow, with all constituent links observed. Of the 312 candidates, 62 pass both criteria and form the constraint matrix $A _ { c } \in \mathbb { R } ^ { 6 2 \times 3 1 8 }$ , which has full row rank. Every retained constraint relates exactly one consecutive station pair (+1 incoming, −1 outgoing). The constraint vector $b _ { c }$ is estimated as the mean of $A _ { c } y _ { \mathrm { t r a i n } } ( t )$ over the training period, accounting for systematic junction imbalances due to residual unmonitored flows.

As shown in Figure 2, HardNetAf-STGNN consists of a backbone of the spatial-temporal graph neural network followed by a hard equality projection layer. With approximately $1 0 . 5 \times 1 0 ^ { 6 }$ trainable parameters, The backbone includes two graph convolutional layers with 64 hidden units, a singlelayer LSTM with 128 hidden units, and a prediction head composed of a fully connected hidden layer with 64 units and a final linear output layer producing predictions for 318 stations. ReLU activation and dropout with a rate of 0.1 are used for nonlinearity and regularization. The raw station-leve predictions are then passed through the hard afine equality projection layer to enforce trafic-flow conservation constraints exactly. The model is trained end to end with the Adam optimizer using a learning rate of $1 0 ^ { - 3 }$ , weight decay of $1 0 ^ { - 5 }$ , and a batch size of 32, with gradient clipping at a maximum norm of 5.0.

![](images/aa0f60650356aa2be2c6df7e912d7cce7d0eb46e271549ee4dca80744fe5685a.jpg)  
Figure 2 HardNetAf-STGNN architecture for trafic-flow prediction. The STGNN backbone learns spatial and temporal trafic patterns from loop detector data, while the hard equality projection layer enforces trafic-flow conservation to produce physics-feasible station-level predictions (figure generated in ChatGPT).

The spectral radius $\rho ( H _ { H } )$ is estimated by stochastic power iteration. For the main $\mathrm { i H V P }$ computation we use damping d = 0.05, K = 400 iterations, and J = 3 independent chains, yielding a deterministic convergence error of $( 0 . 9 5 ) ^ { 4 0 0 } \approx 1 . 2 \times 1 0 ^ { - 9 }$ per Theorem 1. The batch size is set to

$B = 2 5 6$ , which reduces the stochastic noise floor by a factor of $2 5 6 / 3 2 = 8$ compared to the commonly used default of $B = 3 2$ , bringing run-to-run agreement below the 8% threshold. Convergence of each iHVP estimate is verified by the run-agreement criterion:

$$
\frac { \| h ^ { ( 1 ) } - h ^ { ( 2 ) } \| } { \frac { 1 } { 2 } \| h ^ { ( 1 ) } + h ^ { ( 2 ) } \| } < 0 . 0 8 ,\tag{51}
$$

where $h ^ { ( 1 ) }$ and $h ^ { ( 2 ) }$ are the two independent chains initialized from the same $\mathbf { g } _ { t } ^ { H }$ but with diferent random batch sequences. Table 2 summarizes all hyperparameters.

Table 2 Scaled LiSSA hyperparameters used in experiments.
<table><tr><td>Parameter</td><td>Value Justification</td><td></td></tr><tr><td>Damping d</td><td>0.05</td><td>Convergence error  $( 0 . 9 5 ) ^ { 4 0 0 } \approx 1 . 2 \times 1 0 ^ { - 9 }$  (Theorem 1)</td></tr><tr><td>Iterations K</td><td>400</td><td>Error bound</td></tr><tr><td>Runs J</td><td>3</td><td>Variance reduction; agreement verified by (51)</td></tr><tr><td>Batch size B</td><td>256</td><td>Noise floor  $\propto 1 / B ; 8 \times$  reduction vs.  $B { = } 3 2 { : }$  agreement  $\approx 3 \ – 4 \%$ </td></tr><tr><td>Power steps  $T _ { \rho }$ </td><td>40</td><td>Stable  $\hat { \rho }$  estimate</td></tr><tr><td>Power batches M Agreement thresh-</td><td>12</td><td>Accurate HVP per power step 8% Accounts for irreducible stochastic noise at</td></tr><tr><td>old</td><td></td><td> $B { = } 2 5 6$ </td></tr></table>

All experiments were conducted in Google Colab using high-RAM runtime with a single NVIDIA T4 GPU.

## 4. Results

Table 3 shows test set performance of constrained and unconstrained models, measured by data alignment and constraint alignment. Data alignment is measured by

$$
\mathrm { M A E } = \frac { 1 } { T _ { \mathrm { t e s t } } m } \sum _ { t , e } \left| \hat { y } _ { t , e } - y _ { t , e } \right| , \qquad \mathrm { R M S E } = \sqrt { \frac { 1 } { T _ { \mathrm { t e s t } } m } \sum _ { t , e } \left( \hat { y } _ { t , e } - y _ { t , e } \right) ^ { 2 } } ,\tag{52}
$$

where $y _ { t , e }$ and $\hat { y } _ { t , e }$ are the observed and predicted flow at link e in test window t, both in vehicles per hour on the original scale. The HardNetAf-STGNN and unconstrained model achieve a similar level of data alignment. The key diference is in the alignment with physical constraints. The unconstrained output breaks junction flow conservation by an average of 121.95 veh/hr, reaching as high as 536.143 veh/hr in some cases. This is equivalent to the hourly capacity of a full freeway lane. The hard-constrained output reduces this to $9 . 9 \times 1 0 ^ { - 5 }$ veh/hr on average. This confirms near-exact enforcement of all 62 constraints at once throughout the test period.

The small accuracy gain shows a common feature of hard-constrained architectures. The projection ensures physical feasibility by design instead of improving the raw predictive signal. Therefore, accuracy gains are less important than meeting constraint satisfaction guarantees.

Table 3 Test set results on 318-link Sacramento PeMS network (540 test windows). Flow units: veh/hr.
<table><tr><td rowspan="2">Model</td><td colspan="3">Data alignment</td><td colspan="2">Physical constraint alignment</td></tr><tr><td>MAE</td><td>RMSE</td><td> $R ^ { 2 }$ </td><td> $\overline { { | A y - b | } }$ </td><td> $\operatorname* { m a x } | A y - b |$ </td></tr><tr><td>Unconstrained</td><td>267.209</td><td>443.807</td><td>0.9147</td><td>121.95</td><td>536.143</td></tr><tr><td>HardNetAff-STGNN</td><td>263.659</td><td>439.798</td><td>0.9162</td><td> $\mathbf { 9 . 9 \times 1 0 ^ { - 5 } }$ </td><td> $\mathbf { 5 . 6 6 \times 1 0 ^ { - 4 } }$ </td></tr></table>

## 4.1. Physics-Informed Purity Analysis

Table 4 shows the PIP statistics for the HardNetAf-STGNN across the training sets. These values represent the fraction of loss-gradient energy that aligns with the conservation-consistent subspace null $\left( A _ { c } \right)$ . The HardNetAf-STGNN has a mean PIP of 0.923 on the training set. The close match between training and test PIP indicates that the model’s gradient structure for conservation performs well with new observations. A mean PIP of 0.923 means that about 92.3% of the loss-gradient energy at the hard-constrained output is in the conservation-consistent subspace null(A<sub>c</sub>). The remaining 7.7% is in the conservation-violating direction row $\left( A _ { c } \right)$ , and the projection Jacobian $\Pi _ { c }$ removes it before it afects the model parameters.

Table 4 PIP statistics for the HardNetAf-STGNN.
<table><tr><td></td><td>Training set</td></tr><tr><td>Mean PIP Median PIP</td><td>0.923 0.931</td></tr><tr><td>AM peak (07:00-09:00)</td><td>0.948</td></tr><tr><td>PM peak (16:00-18:00)</td><td>0.938</td></tr><tr><td>Night (22:00-05:00)</td><td>0.891</td></tr></table>

PIP changes during diferent trafic periods. The AM peak period has the highest PIP at \$0.948. This indicates that morning commute patterns create loss gradients that closely match the conservation-consistent subspace. The PM peak has a PIP of \$0.938. The night period, from 22:00 to 05:00, shows the lowest PIP at \$0.891, which is \$0.057 lower than the AM peak. During overnight hours, low and highly variable trafic flow leads to larger prediction errors. This makes the physical constraints more influential.

## 4.2. Station Sensitivity Score and Spatial Analysis

Figure 3 show how SSS values are spread across the 318 monitored freeway stations. The map highlights a clear star-shaped concentration around the downtown Sacramento interchange (lat ≈ $3 8 . 6 ^ { \circ }$ , lon $\approx - 1 2 1 . 4 ^ { \circ } )$ , where Interstates 80 and 5, US Highway 50, and State Route 99 meet. SSS declines steadily from this core along each corridor, dropping to nearly zero at outer stations more than about 50 km from the interchange. This pattern reflects the layout of the conservation constraint matrix $A _ { c } \mathrm { : }$ stations near the most constrained junctions show the highest alignment with the network-level loss gradient $\hat { g } _ { t }$ . This finding indicates that the model’s predictions for all 318 links depend most critically on the flows at these key nodes. The northeast arm, which corresponds to the inbound I-80 corridor from Roseville, remains elevated over the largest distance. This is consistent with this route carrying the highest inbound trafic volume and giving the most valuable gradient signal during training.

![](images/5eeee1bd1520ab5729021fceef9cf63e286381ea54d7af668cca3d4d75790ba9.jpg)  
Figure 3 Spatial distribution of Station Sensitivity Scores (SSS) across the Sacramento PeMS network. Yellow indicates high sensitivity; dark blue indicates near-zero sensitivity. The star-shaped pattern centers on the downtown interchange where I-80, I-5, US-50, and CA-99 meet.

## 4.3. Impact of Physics Constraints on Influence Assessment

We compare three frameworks that difer in their models and influence methods. In the first framework, we apply the classic influence function to the unconstrained model, so no part of the analysis uses physics. In the second, we apply the classic influence function to the physics-informed model. The model is trained with our HardNet-Af model, but the influence analysis ignores the projection layer and uses only the network’s raw prediction. In the third, we apply our physics-informed influence function framework to the physics-informed model, so both the model and the influence analysis include the conservation constraints. Diference between the first and second frameworks shows the efect of considering physics in the model. Diference between the second and third shows the efect of considering physics in the influence analysis.

![](images/5943e7fa21df954156ecfe7bf969a0fce414075bda5770af7162a07a49616a72.jpg)  
Figure 4 Top 30 stations by importance score under three frameworks: classic influence function on the unconstrained model (left), classic influence function on the physics-informed model (middle), and physics-informed influence function on the physics-informed model (right). Each box is a station at its rank, and each line follows the same station from one framework to the next. Colors show the freeway route. Thick flat lines mark stations with the same rank in both frameworks. Dashed lines mark stations that enter or leave the top 30; their rank outside the top 30 is shown at the bottom.

Figure 4 compares the top 30 most important stations based on the SSS score calculated under the three frameworks. Each column corresponds lists the top 30 stations in the order of importance under one framework. The lines flow between the columns show the varying order of importance across diferent frameworks. The top two stations, 312010 on Route 50 and 314491 on Route 80, are consistently identified as the most important across all three methods. However, incorporating physics constraints into the model and influence analysis leads to diferent influence estimates, substantially altering the relative rankings of most sensors in the top 30.

## 4.4. Comparison with Trafic-based and Topological Importance Measures

We now examine whether the physics-informed influence score (SSS) provide new information beyond simpler importance measures such as trafic volume, node degree centrality, and betweenness centrality. The node degree is the number of road links that converge at a station. Betweenness centrality measures how frequently the station lies on shortest paths of demand; the higher the value, the more central the station is.

Figure 5 shows the top 30 stations under each measure. The two network-centrality measures point to very diferent stations compared with SSS. Betweenness selects a continuous stretch, where many shortest paths pass, but only 7 of these stations are in the SSS top 30. Several of them rank below 200 by SSS. Degree performs no better, with 8 shared stations. The 11 stations with degree 3 or 4 are all at freeway-to-freeway interchanges. Only two of them are in the SSS top 30, and the rest rank between 83 and 290. Degree cannot separate the remaining stations, since most of the stations have exactly two road links. Trafic volume is closer to SSS; 26 out of the 30 stations appear in the top 30 list from both methods. This is expected, because busier stations contribute more to network-wide prediction error. However, the ranking of the 40 sensors difer substantially. The two lists also difer at the top-ranked stations. Stations 318934 and 3002042 carry high volume (ranks 11 and 14), but SSS places them outside its top 30 (ranks 39 and 36). The diferences suggest that physics-informed influence provides information that is not readily captured by conventional trafic-based or topological metrics.

## 4.5. Assessing the Learning Impact of Influential Sensors

To evaluate whether SSS accurately identifies stations that are important for model learning, we conduct a sensor holdout experiment using the unconstrained baseline model and HardNetAf-STGNN model. Two separate experiments were performed: withholding data from the ten highestranked stations and withholding data from the ten lowest-ranked stations during training. Each model was retrained under both conditions and evaluated across all 318 stations using MAE, RMSE, and $R ^ { 2 }$ . The Percentage changes were calculated relative to the corresponding full model trained using all stations.

In each trial, the input features for the removed stations are masked in the training set. The model is retrained from scratch and tested on all 318 stations using the full unmasked test set.

![](images/b6b2004526c62f8093d9b538ec38e46c1b8cab6b4efe587321a712ee514a5f54.jpg)  
(a) SSS (physics-informed influence)

![](images/68ab10f091681586b01d09a0037211972eba9fc32c9bf1fefcb6a67ec2162ac6.jpg)  
(b) Trafic volume (26 of 30 shared with SSS)

![](images/df84717eb47d68645b7c739e785572575c5647c73ab3d354231fbbcf21ebb624.jpg)  
(c) Degree (8 of 30 shared with SSS)

![](images/52235025d54c30c2e45c82c88c4e2167675e1f6c457a534a304addcd02735170.jpg)  
(d) Betweenness (7 of 30 shared with SSS)  
Figure 5 Top 30 stations by (a) the physics-informed influence score (SSS), (b) mean hourly trafic volume, (c) degree, and (d) betweenness centrality. Red markers show the top 30 stations of each measure; blue rings mark stations that are also in the SSS top 30. The table beside each map lists the ranking and the SSS rank of each station.

As shown in table 5,Withholding the top-10 SSS stations substantially degrades both models. For the unconstrained model, MAE and RMSE increase by approximately 52.6% and 119%, respectively, while $R ^ { 2 }$ decreases by 33.2%. The physics-informed model shows increases of 44.2% in MAE and 114.19% in RMSE, together with a 32% reduction in $R ^ { 2 }$ . In contrast, withholding the bottom-10 stations increases MAE and RMSE by approximately 4% and 2.1% for the unconstrained model, while reducing these errors by 1.87% and 0.76% for the physics-informed model. The corresponding Changes in $R ^ { 2 }$ are near zero for both models. These results explain the usefulness of SSS for identifying stations whose data strongly influence the prediction accuracy. Although the physicsinformed model shows smaller relative increases in error following top-10 removal, both models remain sensitive to the loss of these stations.

Table 5 Impact of holding out the top-10 and bottom-10 SSS stations on the unconstrained and physics-informed models. All metrics are evaluated across all stations. ∆ (%) denotes the percentage change relative to the corresponding full model.
<table><tr><td></td><td colspan="4">Unconstrained</td><td colspan="4">Physics-informed</td></tr><tr><td></td><td colspan="2">Holdout value</td><td colspan="2">∆(%)</td><td colspan="2">Holdout value</td><td colspan="2">∆(%)</td></tr><tr><td>Metric</td><td></td><td>Top-10 Bottom-10</td><td>Top-10</td><td>Bottom-10</td><td></td><td>Top-10 Bottom-10</td><td>Top-10</td><td>Bottom-10</td></tr><tr><td>MAE (veh/hr)</td><td>383.99</td><td>263.84</td><td>+52.6</td><td>+4</td><td>389.83</td><td>265.23</td><td>+44.2</td><td>-1.87</td></tr><tr><td>RMSE (veh/hr)</td><td>945.92</td><td>439.26</td><td>+119</td><td>+2.1</td><td>947.53</td><td>438.90</td><td>+114.19</td><td>-0.76</td></tr><tr><td>R2</td><td>0.613</td><td>0.917</td><td>-33.2</td><td>≈0</td><td>0.6113</td><td>0.9166</td><td>-32</td><td>≈0</td></tr></table>

The diferences between the two removal experiments confirm that SSS is a valuable sensitivity indicator. The ten highest-SSS stations, focused on the downtown Sacramento interchange and along the inbound I-80 corridor, greatly enhance the model’s ability to learn flow patterns across the network. Removing them cuts of the model’s access to the most limited point in the network, making predictions in downstream areas unreliable. In contrast, the ten lowest-SSS stations are located at the edges and have little impact on the gradient directions learned by the model, so their removal hardly afects its quality.

## 5. Conclusions and Discussion

This paper presents a new framework for evaluating influence of data on transportation network problems subject to linear set of physical constraints like flow conservation. Through sensitivity analysis and orthogonal decomposition, we provide analytical insights how physical constraints alter the magnitude, direction, and spatial distribution of training-sample influence. Numerically, we demonstrate the diference between our influence result compared with classical methods, and how such diference translates to model prediction performance. Practically, Station Sensitivity Score (SSS) based on the influence calculation of each station data can further support downstream applications to data diagnostics, sensor evaluation, and targeted data collection.

For future research, despite the computational improvements achieved by the proposed framework, extending influence analysis to larger transportation networks remains challenging, particularly as the number of model parameters, monitoring stations, and physical constraints increases. An important direction for future research is to exploit the sparse and localized structure of trafic network interactions to further reduce the computational cost of influence estimation, rather than treating all model parameters and physical constraints as globally coupled. From a modeling perspective, future studies can relax the assumption of a fixed conservation structure by introducing time-changing constraints that respond to incidents, control actions, or changing demand patterns. More broadly, the proposed framework lays the groundwork for adaptive, physics-aware learning in transportation networks. By incorporating SSS into online sensing and model updating, future systems could continuously identify the most informative data sources and adapt sensing strategies in response to changing trafic conditions. This perspective may extend the proposed framework beyond static prediction toward closed-loop, self-improving transportation intelligence.

## Acknowledgments

This work is supported by the National Science Foundation under grant CMMI-2432336/CMMI-2432337.

## References

Naman Agarwal, Brian Bullins, and Elad Hazan. Second-order stochastic optimization for machine learning in linear time. Journal of Machine Learning Research, 18(116):1–40, 2017.

Samyadeep Basu, Phillip Pope, and Soheil Feizi. Influence functions in deep learning are fragile. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id= xHKVVHGDOEk.

Hao Chen, Gonzalo E. Constante-Flores, and Can Li. Physics-informed neural networks with hard linear equality constraints. Computers & Chemical Engineering, 189:108764, 2024. doi: 10.1016/j. compchemeng.2024.108764.

Arka Daw, Jie Bu, Sifan Wang, Paris Perdikaris, and Anuj Karpatne. Mitigating propagation failures in physics-informed neural networks using retain-resample-release (R3) sampling. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 7264–7302. PMLR, 2023.

Tim De Ryck, Florent Bonnet, Siddhartha Mishra, and Emmanuel de Bézenac. An operator preconditioning perspective on training in physics-informed machine learning. In International Conference on Learning Representations, 2024.

Alp Dener, Marco Andres Miller, Randy Michael Churchill, Todd S. Munson, and Choong-Seock Chang. Training neural networks under physical constraints using a stochastic augmented lagrangian approach, 2020. URL https://arxiv.org/abs/2009.07330.

Thomas Frerix, Matthias Niessner, and Daniel Cremers. Homogeneous linear inequality constraints for neural network activations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 3229–3234. IEEE Computer Society, 2020. doi: 10.1109/CVPRW50498. 2020.00382.

Rui Fu, Zuo Zhang, and Li Li. Using LSTM and GRU neural network methods for trafic flow prediction. In 2016 31st Youth Academic Annual Conference of Chinese Association of Automation (YAC), pages 324–328, 2016. doi: 10.1109/YAC.2016.7804912.

Shengnan Guo, Youfang Lin, Ning Feng, Chao Song, and Huaiyu Wan. Attention based spatial-temporal graph convolutional networks for trafic flow forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 922–929, 2019. doi: 10.1609/aaai.v33i01.3301922.

Dou Huang, Haoran Zhang, Xuan Song, and Ryosuke Shibasaki. Diferentiable projection for constrained deep learning, 2021. URL https://arxiv.org/abs/2111.10785.

Ameya D. Jagtap, Kenji Kawaguchi, and George Em Karniadakis. Adaptive activation functions accelerate convergence in deep and physics-informed neural networks. Journal of Computational Physics, 404: 109136, 2020. doi: 10.1016/j.jcp.2019.109136.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1885–1894. PMLR, 2017. URL https://proceedings.mlr.press/v70/koh17a. html.

Andrei V. Konstantinov and Lev V. Utkin. A new computationally simple approach for implementing neural networks with output hard constraints. Doklady Mathematics, 108(Suppl. 2):S233–S241, 2023. doi: 10.1134/S1064562423701077.

Aleksander Krasowski, Jonas R. Naujoks, Moritz Weckbecker, Galip Ü. Yolcu, Thomas Wiegand, Sebastian Lapuschkin, Wojciech Samek, and René P. Klausen. PINNfluence: Interpreting PINNs through influence functions. In Proceedings of the 43rd International Conference on Machine Learning. PMLR, 2026.

Yaguang Li, Rose Yu, Cyrus Shahabi, and Yan Liu. Difusion convolutional recurrent neural network: Data-driven trafic forecasting. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=SJiHXGWAZ.

Xin Liu, Lanqi Qin, Meng Xu, Yicheng Zhou, Bo Wang, Weiren Yu, and Wenxin Xiong. A comprehensive review of trafic flow forecasting based on deep learning. Neurocomputing, 668:132269, 2026. doi: 10.1016/j.neucom.2025.132269.

Lu Lu, Xuhui Meng, Zhiping Mao, and George Em Karniadakis. DeepXDE: A deep learning library for solving diferential equations. SIAM Review, 63(1):208–228, 2021. doi: 10.1137/19M1274067.

Xiaolei Ma, Zhimin Tao, Yinhai Wang, Haiyang Yu, and Yunpeng Wang. Long short-term memory neural network for trafic speed prediction using remote microwave sensor data. Transportation Research Part C: Emerging Technologies, 54:187–197, 2015. doi: 10.1016/j.trc.2015.03.014.

Pablo Márquez-Neila, Mathieu Salzmann, and Pascal Fua. Imposing hard constraints on deep networks: Promises and limitations. In CVPR Workshop on Negative Results in Computer Vision, 2017. URL https://arxiv.org/abs/1706.02025.

Youngjae Min and Navid Azizan. HardNet: Hard-constrained neural networks with universal approximation guarantees. In Neural Information Processing Systems (NeurIPS) Constrained Machine Learning Workshop, 2025. URL https://arxiv.org/abs/2410.10807.

Barak A. Pearlmutter. Fast exact multiplication by the hessian. Neural Computation, 6(1):147–160, 1994. doi: 10.1162/neco.1994.6.1.147.

Johannes O. Royset and Roger J-B Wets. An Optimization Primer. Springer Series in Operations Research and Financial Engineering. Springer, Cham, 2021. doi: 10.1007/978-3-030-76275-9.

Andrea Schioppa, Polina Zablotskaia, David Vilar, and Artem Sokolov. Scaling up influence functions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 8179–8186, 2022. doi: 10.1609/aaai.v36i8.20791.

Jesus Tordesillas, Victor Klemm, Jonathan P. How, and Marco Hutter. RAYEN: Imposition of hard convex constraints on neural networks, 2023. URL https://arxiv.org/abs/2307.08336.

Bowen Wang and Jingsheng Wang. ST-MGAT: Spatio-temporal multi-head graph attention network for trafic prediction. Physica A: Statistical Mechanics and its Applications, 603:127762, 2022. doi: 10. 1016/j.physa.2022.127762.

Runzhong Wang, Yunhao Zhang, Ziao Guo, Tianyi Chen, Xiaokang Yang, and Junchi Yan. LinSATNet: The positive linear satisfiability neural networks. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 36605–36625. PMLR, 2023a. URL https://proceedings.mlr.press/v202/wang23at.html.

Sifan Wang, Xinling Yu, and Paris Perdikaris. When and why PINNs fail to train: A neural tangent kerne perspective. Journal of Computational Physics, 449:110768, 2022. doi: 10.1016/j.jcp.2021.110768.

Sifan Wang, Shyam Sankaran, Hanwen Wang, and Paris Perdikaris. An expert’s guide to training physicsinformed neural networks, 2023b. URL https://arxiv.org/abs/2308.08468.

Billy M. Williams and Lester A. Hoel. Modeling and forecasting vehicular trafic flow as a seasonal ARIMA process: Theoretical basis and empirical results. Journal of Transportation Engineering, 129(6):664–672, 2003. doi: 10.1061/(ASCE)0733-947X(2003)129:6(664).

Bing Yu, Haoteng Yin, and Zhanxing Zhu. Spatio-temporal graph convolutional networks: A deep learning framework for trafic forecasting. In Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, pages 3634–3640. International Joint Conferences on Artificial Intelligence Organization, 2018. doi: 10.24963/ijcai.2018/505.