# Topological Sheaves of Irreducibility: A Theoretical Framework for Multi-Agent Task Obstructions

Steven Motta<sup>1</sup>, Davide Mambelli<sup>1</sup>, Alessandro Rozza<sup>1</sup>

distributed cognition

Multi-agent systems can difer under identical agents and communication budgets when they support diferent forms of joint interaction. We introduce Topological Sheaves of Irreducibility (TSI), a sheaftheoretic framework separating task-level compatibility obstructions from system capability. A task is represented by an interaction complex � and a capacity sheaf F; nonzero $H ^ { n } ( K ; { \mathcal { F } } )$ defines a structural signature, while registered interventions test capability. For cocycle data, cochain realization is solvable exactly when the represented cohomology class vanishes. Examples show that the signature is sheafrelative, and the degree-1 reduction yields cycle-space witnesses and observation requirements. We evaluate two finite one-round models. Across 15,625 degree-1 inputs, an all-agent additive primitive solves decision and repair exactly; the optimal pairwise system reaches 88.80% on decision, one input above the 88.79% majority-class baseline, and 26.63% on repair. On a degree-2 tetrahedral-boundary task over 531,441 inputs, the same-2-skeleton system retains all four three-agent primitives yet matches the pairwise-only optimum throughout the registered separation window, whereas adding the classfilling four-agent primitive makes both tasks exact. Status-preserving robustness checks retain exact performance but are not cost-matched to the class-restoring ablation. Conditional on the registered primitives and cost models, TSI provides task-relative, falsifiable tests of higher-order interaction.

## 1. Introduction

Multi-agent systems are often compared by changing the agents, the amount of communication, or the learning algorithm. However, two systems can use the same agents and the same communication budget and still difer because they support diferent forms of interaction. Pairwise communication is not always equivalent to a genuinely joint operation under the same task and communication constraints. This distinction becomes important when a task requires locally produced information to be combined into one globally consistent solution.

A simple example illustrates the issue. Suppose three agents can exchange messages pairwise around a cycle, but cannot perform one operation whose inputs and outcome are jointly available to all three. An ordinary or weighted graph records only pairwise relations. Edge weights can express the cost, strength, or bandwidth of each pairwise channel, but they cannot represent an atomic operation available only to the three agents together. A simplicial complex distinguishes the two cases: three pairwise channels give three edges, while a genuine three-way operation adds the filled 2-simplex they bound. Cellular sheaves then specify what information each interaction carries and how local pieces must agree.

This leads to the central idea of TSI. We separate the structure of the task from the capability of the system. The task is represented by a pair (�, F), where � describes the task-relevant compatibility structure and F describes the corresponding data and restriction maps. A nonzero cohomology group $H ^ { n } ( K ; { \mathcal { F } } )$ identifies a sheaf-relative obstruction at cochain degree �. We call this a structural TSI signature. It is an obstruction carried by the task model, not a score assigned to the system. For the cochain-realization family studied in Section 5, Proposition 5.1 gives the obstruction an exact realizability meaning in every cochain degree; Sections 9.1 and 9.2 instantiate degrees 1 and 2. System capability is evaluated separately by asking whether a registered interaction primitive removes the relevant obstruction and whether performance degrades when that primitive is ablated. Section 3.4 formalises this separation.

This viewpoint difers from several established frameworks for multi-agent synergy and irreducibility. Cooperative game theory assigns value to coalitions [31]; partial information decomposition measures information that is available only jointly [35]; and integratedinformation approaches study irreducibility under a diferent set of assumptions [23, 33]. These frameworks are useful comparison points, but they do not directly encode the combination of interaction geometry and task-specific compatibility constraints that TSI targets.

The cohomological criterion underlying TSI is the standard equivalence between a vanishing class and membership in the image of the coboundary map. We use this criterion to specify distributed tasks and then study their communication requirements. The contribution is the task–system separation, the class-targeted intervention protocol, and the finite-model comparisons under explicit primitive and cost assumptions. The paper develops these points as follows:

1. A clear separation between signature and capability. We define the structural signature using sheaf cohomology, and we define operational support using behavioural, class-targeted ablations (Definitions 3.4 and 3.6).

2. A task formulation of the cohomological criterion. For cochain-realization and consistency-decision tasks, Proposition 5.1 records that a cocycle datum is realizable exactly when its cohomology class vanishes. For distributed degree-1 diference constraints, the corresponding cycle-space certificates give Fredholm/Farkas-type infeasibility witnesses (Proposition 5.2).

3. An observation requirement for the registered task family. For decision rules that factor through per-simplex observations, Proposition 5.4 identifies which �-cell data an exact cochain-realizability rule must observe, with the triangle argument as the degree-1 special case.

4. Examples showing that topology alone is not enough. A twisted sheaf can remove a degree-1 class from a cycle, while a representational sheaf can create one on a contractible complex (Sections 4.2 and 4.3).

5. A practical modelling protocol. Section 8.2 gives three recipes for constructing capacity sheaves and requires the task model to be fixed before behavioural evaluation.

6. Fixed-task benchmarks in degrees 1 and 2. We hold each task model fixed and vary only the system interaction structure. The degree-1 experiment gives a same-1- skeleton separation under exhaustive enumeration; the degree-2 tetrahedral-boundary experiment gives the analogous same-2-skeleton separation and adds status-preserving stress tests (Section 9).

The intended claim is therefore limited but testable. A nonzero $H ^ { n } ( K ; { \mathcal { F } } )$ identifies a class of task-level compatibility demands. The empirical question is whether systems equipped with the registered interaction structure perform better specifically on tasks carrying the predicted obstruction, and whether that advantage disappears under the corresponding ablation.

## 2. Background

Interaction topology. A graph cannot distinguish three pairwise links from a genuine three-way interaction. A simplicial or cellular complex can: a �-simplex records a (�+1)-way interaction, while an unfilled cycle records a hole. Applied topology uses these objects to encode higher-order relations [13, 19]. Related work covers higher-order networks [4, 5, 7, 34], empirical motif and simplex structure [6], and Hodge-Laplacian methods on simplicial complexes [3, 30]. Recent studies further show that the choice of higher-order representation can itself change collective dynamics [37], and that homological structure can be used to analyse empirically reconstructed high-order interaction networks [12]. Related work combines simplicial lifting, generalized Hodge operators, and distributed decision optimisation for multilayer networks [36]. These results reinforce the need to distinguish the chosen interaction representation from the task-specific semantics that TSI places in the capacity sheaf.

Cellular sheaves. A cellular sheaf attaches vector spaces and compatible linear maps to the cells of a complex. Cellular descriptions of sheaf data on stratified spaces trace back to Shepard’s thesis [32], with modern computational formulations developed in subsequent work [9, 10]. Existing applications include local measurement systems, sensor consistency, network coding, and structured graph learning [8, 14, 17, 26, 27]. Discourse sheaves use restriction maps to model how agents translate shared topics through private frames [18], which is close to the capacity semantics used here. More broadly, recent topological deeplearning work has emphasised learning and message passing on simplicial, cellular, and other non-graph domains [24, 38]. Sheaf cohomology depends on the stalks and restriction maps, so it is not a homotopy invariant of the underlying complex. That dependence lets the model encode task semantics as well as interaction shape.

Sheaf-theoretic local-to-global obstructions also play a central role in contextuality. Abramsky and Brandenburger formulate contextuality and non-locality in terms of compatible local measurement data that fail to admit a global section [1]. This is an important conceptual antecedent for TSI. The present framework difers in using cellular capacity sheaves on registered multi-agent interaction complexes, degree-specific sheaf cohomology as a task-relative structural signature, and separate behavioural interventions to assess system capability.

Abramsky, Mansfield and Soares Barbosa subsequently made the obstruction explicitly cohomological, using Čech cohomology of an abelian presheaf derived from local sections over measurement contexts to construct cohomological witnesses for contextuality [2]. This is the closest antecedent to the present construction: both use a non-vanishing cohomological obstruction to witness a failure of local-to-global assembly. In the contextuality setting this witness is suficient but not necessary, whereas Proposition 5.1 gives an exact criterion for the linear cochain-realization family considered here. The settings also difer operationally: there the presheaf is built over measurement contexts, whereas TSI places cellular capacity sheaves on registered multi-agent interaction complexes and couples the structural class to system capability through separately registered interventions (Definition 3.6).

Multi-agent capability and irreducibility. Partial information decomposition studies information available only from variables taken jointly [35]. Cooperative game theory studies coalition value and superadditivity [31], and integrated-information theory uses its own axioms for irreducibility [23, 33]. TSI instead models the interaction complex and task-specific compatibility constraints. Section 7 treats the other quantities as comparison targets.

## 3. Formal Framework

## 3.1. Interaction Complexes

Let $A = \{ a _ { 1 } , \ldots , a _ { m } \}$ be a finite set of agents. We model task-relevant interactions by a finite abstract simplicial complex � on �.

Definition 3.1 (Operational simplex criterion). Let Π be the registered communication model with its budget. A simplex � with $| \sigma | \geq 2$ is admissible in � only if its agents can perform, under Π, a joint operation that cannot be simulated within the same budget by relaying over the proper faces of �. Such an operation may be a simultaneous observation, an atomic joint decision, or a shared channel with common-knowledge semantics. Message trafic among the agents of � does not by itself make � admissible. A missing simplex means that no such joint operation is registered as available under Π.

Definition 3.2 (Interaction complex). An interaction complex is an abstract simplicial complex � whose vertices are agents and whose �-simplices represent registered $( k + 1 )$ -way interactions relevant to a task. Thus $\{ a _ { i } , a _ { j } \} \in K$ records a pairwise interaction, whereas $\{ a _ { i } , a _ { j } , a _ { \ell } \} \in K$ records a registered three-way interaction that is not reducible, under the specified communication model and budget, to relaying over its proper faces.

The construction of � is part of the empirical model and must satisfy the operational simplex criterion of Definition 3.1. In particular, the presence of a simplex records the availability of a registered interaction primitive under the stated model, rather than merely the observation of message trafic among its agents.

## 3.2. Capacity Sheaves

Definition 3.3 (Capacity sheaf). Let � be a finite simplicial complex. A capacity sheaf F of finite-dimensional real vector spaces assigns:

1. a stalk $\mathcal { F } ( \sigma )$ to every simplex $\sigma \in K ;$ and

2. a linear restriction map $\rho _ { \sigma \tau } : \mathcal { F } ( \sigma )  \mathcal { F } ( \tau )$ for every face relation $\sigma \subseteq \tau ,$

subject to $\rho _ { \sigma \sigma } = \mathrm { i d }$ and $\rho _ { \tau v } \circ \rho _ { \sigma \tau } = \rho _ { \sigma v }$ whenever $\sigma \subseteq \tau \subseteq v .$

Here $\mathcal { F } ( \nu )$ is the space of states, plans, estimates, or representations available to agent �. For $| \sigma | \geq 2 , \mathcal { F } ( \sigma )$ contains the quantities on which the agents in � must agree or that they must produce jointly. The map $\rho _ { \sigma \tau }$ expresses a subgroup’s data in the coordinates of the larger interaction. The model specifies what its vectors mean and why the maps preserve task structure. Section 8.2 gives three constructions. The constant sheaf ℝ, with stalk ℝ and identity maps, is the topology-only baseline.

## 3.3. Cochains and Cohomology

After choosing an orientation for each simplex, define

$$
C ^ { k } ( K ; \mathcal { F } ) = \bigoplus _ { \sigma \in K _ { k } } \mathcal { F } ( \sigma ) ,\tag{1}
$$

where $K _ { k }$ is the set of oriented �-simplices. The coboundary maps $\delta ^ { k } : C ^ { k } ( K ; { \mathcal { F } } ) \to C ^ { k + 1 } ( K ; { \mathcal { F } } )$ are induced by the restriction maps and signed incidences: for an oriented (�+1)-simplex $\tau = [ \nu _ { 0 } , \dots , \nu _ { k + 1 } ]$ 2

$$
( \delta ^ { k } f ) ( \tau ) = \sum _ { i = 0 } ^ { k + 1 } ( - 1 ) ^ { i } \rho _ { d _ { i } \tau , \tau } \bigl ( f ( d _ { i } \tau ) \bigr ) , \qquad d _ { i } \tau = [ \nu _ { 0 } , \ldots , \widehat { \nu _ { i } } , \ldots , \nu _ { k + 1 } ] .\tag{2}
$$

The functoriality of $\mathcal { F }$ gives $\delta ^ { k + 1 } \circ \delta ^ { k } = 0$ , yielding the cochain complex

$$
0 \longrightarrow C ^ { 0 } ( K ; \mathcal { F } ) \xrightarrow { \delta ^ { 0 } } C ^ { 1 } ( K ; \mathcal { F } ) \xrightarrow { \delta ^ { 1 } } C ^ { 2 } ( K ; \mathcal { F } ) \xrightarrow { \delta ^ { 2 } } \cdots ,\tag{3}
$$

whose cohomology is

$$
H ^ { n } ( K ; { \mathcal { F } } ) = \ker ( \delta ^ { n } ) / \operatorname { i m } ( \delta ^ { n - 1 } ) .\tag{4}
$$

A class in $H ^ { n } ( K ; { \mathcal { F } } )$ is an �-dimensional compatible cochain that is not the coboundary of lower-dimensional data.

Definition 3.4 (Structural TSI signature). A task model $( K , { \mathcal { F } } )$ has a structural TSI signature of degree � when

$$
H ^ { n } ( K ; \mathcal { F } ) \neq 0 .\tag{5}
$$

The dimension $\beta ^ { n } ( K ; \mathcal { F } ) = \dim H ^ { n } ( K ; \mathcal { F } )$ is its multiplicity.

Definition 3.5 (Task-relative irreducibility). A cocycle demand $c \in \ker ( \delta ^ { n } ) \subset C ^ { n } ( K ; { \mathcal F } )$ is reducible in $( K , { \mathcal { F } } ) i f c \in \mathrm { i m } ( \delta ^ { n - 1 } )$ , and irreducible otherwise. Equivalently,

$$
c i s i r r e d u c i b l e \quad \Longleftrightarrow \quad [ c ] \neq 0 i n H ^ { n } ( K ; \mathcal { F } ) .
$$

A structural TSI signature of degree � therefore exists precisely when the task model admits at least one irreducible cocycle demand at that degree. Irreducibility in this sense is a property of the pair $( K , { \mathcal { F } } )$ and the cochain degree, not of the agents or their intelligence: it states that the cocycle demand cannot be assembled from (�−1)-dimensional data over the registered interaction structure. It is distinct from irreducibility in cooperative game theory or integrated information theory; Section 7 treats those notions as comparison targets rather than equivalents.

Remark 3.1 (What the signature is not). The signature is a property of the pair $( K , { \mathcal { F } } )$ , not of the complex alone (Sections 4.2 and 4.3), and not of the system’s intelligence. Its degree identifies an obstruction type, not a ranking; $\beta ^ { n }$ should not be compared across tasks without normalisation and a behavioural interpretation.

## 3.4. Operational Support for TSI Signatures

Operational support couples the structural signature to system performance and controlled interventions.

Definition 3.6 (Operational support). Let T be a task family with a registered model $( K , { \mathcal { F } } )$ carrying a structural signature of degree � with class $[ \omega ] \neq 0 .$ . An admissible intervention has a registered action on the interaction complex and one of the following comparison maps.

1. Removal. Removing simplices gives a subcomplex $\iota : K ^ { - } \hookrightarrow K$ and the restriction map

$$
\iota ^ { * } : H ^ { n } ( K ; { \mathcal { F } } ) \longrightarrow H ^ { n } ( K ^ { \cdot } ; { \mathcal { F } } | _ { K ^ { - } } ) .
$$

2. Sheaf alteration. A sheaf morphism $\varphi : { \mathcal { F } }  { \mathcal { F } } ^ { \prime }$ over � gives

$$
\varphi _ { * } : H ^ { n } ( K ; { \mathcal { F } } ) \longrightarrow H ^ { n } ( K ; { \mathcal { F } } ^ { \prime } ) .
$$

3. Addition. Adding simplices gives an inclusion $\jmath : K \hookrightarrow K ^ { + }$ together with a registered extension $\widetilde { \mathcal F }$ on $K ^ { + }$ satisfying $\jmath ^ { - 1 } \widetilde { \mathcal { F } } \cong \mathcal { F }$ . The induced restriction map is

$$
\begin{array} { r } { \boldsymbol { \jmath } ^ { * } : H ^ { n } ( K ^ { + } ; \widetilde { \mathcal { F } } ) \longrightarrow H ^ { n } ( K ; \mathcal { F } ) . } \end{array}
$$

For removal and sheaf alteration, the corresponding map records whether [�] survives $o r$ is sent to zero. For an addition, no natural map sends [�] forward. We therefore call the addition class-filling when

$$
[ \omega ] \not \in \mathrm { i m } ( \jmath ^ { * } ) ,
$$

namely when the task class does not extend to the enlarged complex; it is extension-preserving when $[ \omega ] \in \mathrm { i m } ( \ j ^ { * } )$

A joint primitive on a participant set $\sigma$ is interpreted relative to a registered base complex $K _ { 0 }$ through its primitive closure

$$
\mathsf { P } _ { \sigma } ( K _ { 0 } ) = \{ \tau \subseteq \sigma : | \tau | \geq 2 , \tau \notin K _ { 0 } \} .
$$

Ablating the primitive removes its entire closure $\mathsf { P } _ { \sigma } ( K _ { 0 } )$ and retains the simplices already present in $K _ { 0 } .$ . Removing only the maximal simplex is not an admissible primitive ablation. The ablation is class-restoring if [�] extends again after removal. Controls are matched by registered primitive cost, including external input/output width, per-agent congestion, parameter count, and retained base complex. The raw number of removed simplices is not the matching criterion.

A multi-agent system � provides operational support for the degree-� TSI signature on $\mathcal { T }$ $i f \colon$

1. (Performance) � solves $\mathcal { T }$ above a pre-specified threshold under a fixed communication budget;

2. (Class-restoring sensitivity) for a registered class-filling mechanism used by �, primitiveclosure ablations that restore [�] degrade performance on $\mathcal { T }$ by more than a pre-specified margin; and

3. (Status-preserving insensitivity) matched interventions of equal registered cost that leave the extension status of [�] unchanged do not produce a comparable degradation.

All three conditions are behavioural and falsifiable. Condition (1) without (2) and (3) is an ordinary benchmark result; (2) without (3) cannot distinguish a topology-specific mechanism from generic capacity loss. Operational support is the conjunction.

Remark 3.2 (Why the comparison maps are needed). An intervention changes $( K , { \mathcal { F } } )$ and therefore changes the group $H ^ { n }$ . Comparing a class before and after the intervention requires a canonical map. For additions, this map also requires an explicit sheaf extension $\widetilde { \mathcal F }$ on $K ^ { + }$ ; the expression $H ^ { n } ( K ^ { + } ; { \mathcal { F } } )$ is otherwise undefined. For example, under $\jmath : \partial \Delta ^ { 2 } \hookrightarrow \Delta ^ { 2 }$ with constant coeficients,

$$
H ^ { 1 } ( \Delta ^ { 2 } ; \underline { { \mathbb { R } } } ) = 0 \longrightarrow H ^ { 1 } ( \partial \Delta ^ { 2 } ; \underline { { \mathbb { R } } } ) \cong \mathbb { R } ,
$$

so the boundary generator is not in the image of $\jmath ^ { * }$ : a 1-cochain with nonzero circulation ceases to be a cocycle once the 2-simplex is present. In relative cellular sheaf cohomology, the long exact sequence of $( K ^ { + } , K )$ contains

$$
H ^ { n } ( K ^ { + } ; \widetilde { \mathcal { F } } ) \stackrel { { \mathcal { I } } ^ { * } } { \longrightarrow } H ^ { n } ( K ; \mathcal { F } ) \longrightarrow H ^ { n + 1 } ( K ^ { + } , K ; \widetilde { \mathcal { F } } ) ,
$$

so $[ \omega ]$ extends exactly when its relative obstruction vanishes. A primitive-level ablation must therefore remove the full closure generated by the primitive. Removing only its maximal simplex would leave some of the primitive’s lower faces available.

Remark 3.3 (Signature–capability separation). Definition 3.6 keeps the two roles separate. The class [�] describes the task’s compatibility demand, as made explicit by Proposition 5.1. System capability is the observed ability to meet that demand, either through interaction structure that fills the obstruction or through an aggregation strategy subject to the observation requirements that Proposition 5.4 imposes on rules factoring through per-simplex observations.

## 4. Worked Examples

## 4.1. The Three-Agent Cycle: Topology Creates a Class

Consider agents 0, 1, 2 with pairwise channels but no three-way channel: $K = \partial \Delta ^ { 2 }$ , with oriented edges [0, 1], [0, 2], [1, 2] and no 2-simplex. Equip � with the constant sheaf ℝ. Then $C ^ { 0 } \cong \mathbb { R } ^ { 3 } , C ^ { 1 } \cong \mathbb { R } ^ { 3 } , C ^ { 2 } = 0$ , and

$$
\delta ^ { 0 } = \left( { \begin{array} { c c c } { - 1 } & { 1 } & { 0 } \\ { - 1 } & { 0 } & { 1 } \\ { 0 } & { - 1 } & { 1 } \end{array} } \right) .\tag{6}
$$

For $c = ( c _ { 0 1 } , c _ { 0 2 } , c _ { 1 2 } ) \in C ^ { 1 }$ , define $\ell ( c ) = c _ { 0 1 } - c _ { 0 2 } + c _ { 1 2 }$ . A direct check gives $\ell \circ \delta ^ { 0 } = 0$ , so im $( \delta ^ { 0 } ) \subseteq \ker ( \ell )$ ; both spaces have dimension two, hence they are equal, and since ${ \cal C } ^ { 2 } = 0$ 2

$$
H ^ { 1 } ( K ; \mathbb { R } ) \cong \mathbb { R } ^ { 3 } / \operatorname { i m } ( \delta ^ { 0 } ) \cong \mathbb { R } .\tag{7}
$$

Figure 1 summarizes the class-filling mechanism in degrees 1 and 2 for the constant sheaf.

![](images/b539ea49203de4aed4888e97184439e00bbaabf4e75f140f58787cdab7d0111e.jpg)  
Figure 1 | Class-filling in degrees 1 and 2, with the constant sheaf ℝ. (a) Three agents with pairwise channels only span the boundary of a triangle and carry a degree-1 class; (b) the three-agent joint operation fills the 2-simplex and the class vanishes. (c) The tetrahedral boundary retains all four three-agent operations, yet carries a degree-2 class; (d) only the four-agent operation fills the 3-simplex and removes it. Dashed segments are the edges hidden behind the body. In each row the communication graph is identical on both sides: the diference is the top-dimensional joint operation, not the number of pairwise channels. Panel (c) is the configuration of the system $S _ { \mathrm { f a c e } }$ in Section 9.2, whose exhaustive optimum coincides with pairwise-only communication throughout the registered separation window.

The model has a degree-1 structural signature. Adding the 2-simplex [0, 1, 2] yields $\delta ^ { 1 } ( c ) = c _ { 0 1 } - c _ { 0 2 } + c _ { 1 2 } = \ell ( c )$ , so $\ker ( \delta ^ { 1 } ) = \mathrm { i m } ( \delta ^ { 0 } )$ and $H ^ { 1 } ( \Delta ^ { 2 } ; \mathbb { R } ) = 0$ : the joint channel fills the obstruction. Section 5 gives this example its exact operational meaning.

## 4.2. A Twisted Sheaf: Sheaf Structure Destroys the Class

Keep $K = \partial \Delta ^ { 2 }$ but replace the constant sheaf with a rank-one monodromy sheaf (local system) $\mathcal { F } _ { \lambda }$ : all stalks ℝ, all restriction maps the identity except $\rho _ { \nu _ { 0 } , [ 0 , 1 ] } ( x ) = \lambda x$ for a fixed $\lambda \ \in \ \mathbb { R } \ \backslash \ \{ 0 , 1 \}$ . (Invertibility of the restriction maps, $\lambda \neq 0$ , is what licenses the term “monodromy”; the cohomology computation below in fact only uses $\lambda \neq 1 . )$ Then

$$
\delta ^ { 0 } = \left( { \begin{array} { c c c } { { - \lambda } } & { { 1 } } & { { 0 } } \\ { { - 1 } } & { { 0 } } & { { 1 } } \\ { { 0 } } & { { - 1 } } & { { 1 } } \end{array} } \right) , \qquad \operatorname* { d e t } ( \delta ^ { 0 } ) = 1 - \lambda \neq 0 ,\tag{8}
$$

so $\delta ^ { 0 }$ is an isomorphism, giving $H ^ { 0 } ( K ; \mathcal { F } _ { \lambda } ) = 0$ and $H ^ { 1 } ( K ; \mathcal { F } _ { \lambda } ) = 0$ . The cycle is unchanged, but its signature vanishes. Here $H ^ { 0 } = 0$ means that the twisted sheaf has no nonzero global sections. Since $\delta ^ { 0 }$ is invertible, every edge datum � has a unique realizing vertex assignment. One interpretation is a systematic rescaling by agent 0 on the [0, 1] channel, such as a calibration or unit conversion. This changes the consistency problem: realization is uniquely solvable for every datum. Thus monodromy around the loop can remove the compatibility obstruction.

## 4.3. A Representational Sheaf: Obstruction Without Topology

Now take � to be a single edge $\tau = [ 0 , 1 ]$ , a contractible complex with trivial ordinary cohomology. Define $\mathcal { F }$ by $\mathcal { F } ( \nu _ { 0 } ) = \mathcal { F } ( \nu _ { 1 } ) = \mathbb { R } , \mathcal { F } ( \tau ) = \mathbb { R } ^ { 2 }$ , and

$$
\rho _ { \nu _ { 0 } \tau } ( x ) = ( x , 0 ) , \qquad \rho _ { \nu _ { 1 } \tau } ( y ) = ( y , 0 ) .\tag{9}
$$

Then $\delta ^ { 0 } ( f _ { 0 } , f _ { 1 } ) = \rho _ { \nu _ { 1 } \tau } ( f _ { 1 } ) - \rho _ { \nu _ { 0 } \tau } ( f _ { 0 } ) = ( f _ { 1 } - f _ { 0 } , 0 ) , s \mathrm { o } \mathrm { i m } ( \delta ^ { 0 } ) = \mathbb { R } \times \{ 0 \}$ and

$$
\begin{array} { r } { \boldsymbol { H } ^ { 1 } ( \boldsymbol { K } ; \mathcal { F } ) \cong \mathbb { R } ^ { 2 } / \left( \mathbb { R } \times \{ \boldsymbol { 0 } \} \right) \cong \mathbb { R } \neq \boldsymbol { 0 } . } \end{array}\tag{10}
$$

The complex is contractible, yet it has a degree-1 class. The second component of $\mathcal { F } ( \tau )$ can represent a jointly produced quantity, such as a cross-check or negotiated parameter, that neither agent can supply from its individual representation. This obstruction comes from the sheaf rather than the topology of �.

Remark 4.1. Together, the two examples show that a structural TSI signature is not simply a Betti number. Changing the sheaf can remove a class from a cycle or create one on a contractible complex. Empirical work must therefore specify and fix the pair $( K , { \mathcal { F } } )$ before evaluation (Section 8.2).

## 5. The Obstruction–Task Correspondence

To make the signature testable, we relate it to a task family for which the cohomology class determines solvability.

## 5.1. Realization and Consistency-Decision Tasks

Fix a model $( K , { \mathcal { F } } )$ . Each edge $e = \left[ i , j \right]$ carries a datum $c _ { e } \in \mathcal { F } ( e )$ , such as an observed ofset or required relation. The initial allocation of these data among agents is part of the registered communication model, not part of the cohomological solvability statement. The correspondence below therefore makes no assumption about which agents initially observe each $c _ { e } ;$ Section 9 specifies concrete allocations for the one-round benchmarks.

• Realization task ${ \sf R } ( c )$ : the agents must output local states $f _ { \nu } \in \mathcal { F } ( \nu )$ such that $\rho _ { \nu _ { j } e } ( f _ { j } ) -$ $\rho _ { \nu _ { i } e } ( f _ { i } ) = c _ { e }$ for every oriented edge $e = [ \nu _ { i } , \nu _ { j } ] ; \mathrm { i . e . } \delta ^ { 0 } f = c .$

• Consistency-decision task $\mathsf { D } ( c )$ : the agents must output the single bit $\mathbb { I } c \in \mathrm { i m } ( \delta ^ { 0 } )$ ]].

These degree-1 tasks cover familiar distributed problems, including clock synchronisation from pairwise ofsets, sensor calibration from pairwise discrepancies, consistency of exchange rates in log coordinates, and feasibility checking for distributed constraints.

More generally, for $n \geq 1$ and $c \in C ^ { n } ( K ; \mathcal { F } )$ , define the cochain-realization task $\mathsf { R } ^ { n } ( c )$ as finding $x \in C ^ { n - 1 } ( K ; \mathcal { F } )$ such that $\delta ^ { n - 1 } x = c ,$ and define $\mathsf { D } ^ { n } ( { \boldsymbol { c } } )$ as deciding membership in $\mathrm { i m } ( \delta ^ { n - 1 } )$ . Thus $\mathsf { R } ^ { 1 } = \mathsf { R }$ and $\mathsf { D } ^ { 1 } = \mathsf { D }$ . The task-specific interpretation of the $( n - 1 )$ -cochain � must be supplied by the application; Section 9.2 gives one degree-2 construction.

Proposition 5.1 (Cochain-realization correspondence). Let $( K , { \mathcal { F } } )$ be arbitrary, let $n \geq 1 .$ and let $c \in C ^ { n } ( K ; \mathcal { F } )$ . Every realizable datum $c \in \mathrm { i m } ( \delta ^ { n - 1 } )$ is a cocycle. Moreover, for every $c \in \ker ( \delta ^ { n } )$ , the realization task $\mathsf { R } ^ { n } ( c )$ is solvable if and only if the cohomology class [�] vanishes in $H ^ { n } ( K ; { \mathcal { F } } )$ . If � is �-dimensional, $C ^ { n + 1 } = 0$ makes every �-cochain a cocycle. Consequently: (i) $i f H ^ { n } ( K ; { \mathcal { F } } ) = 0 ,$ , every cocycle datum is realizable; (ii) if $\dot { \cdot } H ^ { n } ( K ; \mathcal { F } ) \neq 0$ then,for any norm $\| \cdot \|$ on the finite-dimensional space $C ^ { n } ( K ; { \mathcal { F } } )$ and any $\epsilon > 0 ,$ , every realizable datum has a non-realizable cocycle within distance $\epsilon ;$ and on cocycles ${ \sf D } ^ { n } ( c ) = \left[ { \sf I } \left[ c \right] \right] = 0 ] \quad$

Proof. $\mathsf { R } ^ { n } ( c )$ is solvable if and only if $c \in \mathrm { i m } ( \delta ^ { n - 1 } )$ . Since $\delta ^ { n } \circ \delta ^ { n - 1 } = 0$ , every realizable datum is a cocycle. For $c \in \ker ( \delta ^ { n } )$ , membership in im $( \delta ^ { n - 1 } )$ is equivalent to $[ c ] = 0$ in $\ker ( \delta ^ { n } ) / \mathrm { i m } ( \delta ^ { n - 1 } ) = H ^ { n } ( K ; { \mathcal F } )$ . If � is $n { \mathrm { - d i m e n s i o n a l } } _ { ; }$ , then $C ^ { n + 1 } = 0$ . For (ii), fix any norm $\| \cdot \|$ on $C ^ { n } ( K ; { \mathcal { F } } )$ ; since this space is finite-dimensional all norms on it are equivalent, so the statement does not depend on the choice. Choose a cocycle � with $[ \omega ] \neq 0$ . For any realizable � and any $t \neq 0$ , the cochain $c + t \omega$ is a cocycle with $[ c + t \omega ] = t [ \omega ] \neq 0$ , hence not realizable, and $\| ( c + t \omega ) - c \| = | t | \| \omega \| < \epsilon$ once $| t | < \epsilon / \| \omega \|$ □

The proposition gives the same algebraic realizability criterion in every cochain degree. For cocycle data, a non-realizable demand is therefore irreducible in the sense of Definition 3.5. In degree 1, the unknowns are agent states and the criterion has the direct distributed interpretation used below; $\beta ^ { 1 } ( K ; \mathcal { F } )$ counts independent directions of unrealizable edge demand. Section 9.2 supplies a degree-2 instance in which the unknowns live on edges and the prescribed defects live on triangles.

## 5.2. A Worked Reduction: Distributed Diference-Constraint Feasibility

Consider a distributed diference-constraint system. Agents hold variables $x _ { \nu } \in \mathbb { R }$ , and each edge $e = \left[ i , j \right]$ of a constraint graph � carries the equation $x _ { j } - x _ { i } = c _ { e } .$ . The same form appears in clock synchronisation, sensor calibration, no-arbitrage checks on log exchange rates, and equality constraints in scheduling. For the constant sheaf, feasibility is the realization task $\delta ^ { 0 } x = c$ . Related sheaf-based distributed-optimization formulations impose more general homological constraints through sheaf Laplacians [16]; here we focus instead on exact feasibility of diference constraints and the corresponding dual infeasibility witnesses. The reduction also identifies the meaning of the dual objects.

Proposition 5.2 (Certificate space and Fredholm/Farkas-type witnesses). Let � be a connected graph, ℝ the constant sheaf, and define $\mathcal { A } = ( \mathrm { i m } \delta ^ { 0 } ) ^ { \perp } \subseteq C ^ { 1 } ( K ; \mathbb { R } ) ^ { * }$ . Then:

1. the system $\delta ^ { 0 } x = c$ is feasible $i f f \eta ( c ) = 0$ for every $\eta \in \mathcal { A } ;$

2. ${ \mathcal { A } } = \ker \left( ( \delta ^ { 0 } ) ^ { \top } \right)$ is identified with the cycle space under the standard Euclidean pairing of $K ,$ of dimension $\beta ^ { 1 } ( K ; \mathbb { R } ) = | E | - | V | + 1$ ; for a fixed datum $c ,$ the Fredholm infeasibility witnesses are precisely the functionals $\eta \in \mathcal { A }$ satisfying $\eta ( c ) \neq 0$ . These are the equalityconstraint analogue of Farkas certificates;

3. if the system is infeasible, every support-minimal witness is a nonzero scalar multiple of the signed indicator of a simple cycle � with nonzero signed discrepancy $\textstyle \sum _ { e \in \gamma } \pm c _ { e } \ [ 2 9 ]$ Conversely, every such cycle provides a certificate, and its constraints form a minimally infeasible subsystem.

Proof. (1) is Proposition 5.1 restated through the annihilator, since im $( \delta ^ { 0 } )$ is the common kernel of A. (2): identifying $C ^ { 1 }$ with its dual by the standard basis, ${ \mathcal { A } } = ( \operatorname { i m } \delta ^ { 0 } ) ^ { \perp } = \ker ( ( \delta ^ { 0 } ) ^ { \top } )$ and $( \delta ^ { 0 } ) ^ { \top }$ is the boundary map sending an edge vector to the diference of its endpoint indicators; its kernel is the cycle space, whose dimension is $\left| E \right| - \left| V \right| + 1$ for connected �. The finite-dimensional Fredholm alternative states that $A x = b$ is infeasible if there is $y$ with $y ^ { \top } A = 0$ and $y ^ { \top } b \neq 0 ;$ ; with $A = \delta ^ { 0 }$ , these are exactly the elements $\eta \in \mathcal { A }$ with $\eta ( c ) \neq 0 . ( 3 )$ recall that the elementary vectors of a subspace of $\mathbb { R } ^ { E }$ are its nonzero vectors of inclusionminimal support, and that every vector $\eta$ of the subspace admits a conformal decomposition $\begin{array} { r } { \eta = \sum _ { r } \eta _ { r } } \end{array}$ <sub>�</sub> into elementary vectors of the same subspace with supp $( \eta _ { r } ) \subseteq \operatorname { s u p p } ( \eta )$ and no sign cancellation on any coordinate; moreover, if $\eta$ is not itself a scalar multiple of an elementary vector, then each $\eta _ { r }$ has support strictly contained in supp(�) [29]. For the cycle space of a graph the elementary vectors are exactly the signed indicators of simple cycles.

Now let $\eta$ be a witness of inclusion-minimal support, that is, $\eta \in \mathcal { A }$ with $\eta ( c ) \neq 0$ and no $\eta ^ { \prime } \in \mathcal { A }$ satisfying $\eta ^ { \prime } ( c ) \neq 0$ has support strictly contained in supp(�). Decompose $\begin{array} { r } { \eta = \sum _ { r } \eta _ { r } } \end{array}$ conformally. From $\begin{array} { r } { \sum _ { r } \eta _ { r } ( c ) = \eta ( c ) \neq 0 } \end{array}$ some summand satisfies $\eta _ { r } ( c ) \neq 0$ , so that $\eta _ { r }$ is itself a witness with supp $( \eta _ { r } ) \subseteq \operatorname { s u p p } ( \eta )$ . Minimality of $\eta$ forbids strict containment, so $\eta$ is a scalar multiple of an elementary vector, i.e. of the signed indicator of a simple cycle $\gamma _ { ; }$ , and $\eta ( c ) \neq 0$ says exactly that $\textstyle \sum _ { e \in \gamma } \pm c _ { e } \neq 0$ . Conversely, the signed indicator of any simple cycle lies in the cycle space, so any simple cycle with nonzero signed discrepancy is a certificate. Finally, deleting any single edge constraint from such a cycle leaves a forest system, which is always feasible; hence the constraints indexed by $\gamma$ form a minimally infeasible subsystem. □

For diference constraints, the cohomological obstruction space is the certificate space, and $\beta ^ { 1 }$ counts its independent directions. For a fixed datum, the active Fredholm/Farkas-type witnesses are the elements that pair nontrivially with it. Harmonic representatives can describe how a certificate class is distributed across cells once inner products are chosen, but they need not be sparse or local. Any localisation claim requires an additional sparsity criterion. Other reductions should provide the same three ingredients: a task family, a registered sheaf, and a dictionary between dual cohomological data and task-specific witnesses.

## 5.3. What Local Observation Cannot Decide

Proposition 5.3 (Triangle case). Let $K = \partial \Delta ^ { 2 }$ with the constant sheaf, and let $D : C ^ { 1 } ( K ; \underline { { \mathbb { R } } } ) \to$ $\{ 0 , 1 \}$ decide membership in im $( \delta ^ { 0 } )$ . No decision rule that observes only a proper subset of the three edge coordinates computes � on all inputs.

Proof. Let $S \subsetneq \{ 0 1 , 0 2 , 1 2 \}$ and let $e \not \in S$ be unobserved. Let ${ \bf 1 } _ { e } \in C ^ { 1 }$ be the indicator cochain of � (value 1 on $e , 0$ elsewhere). The inputs 0 and ${ \bf 1 } _ { e }$ agree on all coordinates in $S ,$ yet $0 \in \mathrm { i m } ( \delta ^ { 0 } )$ while $\ell ( { \bf 1 } _ { e } ) = \pm 1 \neq 0$ , so 1<sub>�</sub> ∉ im $( \delta ^ { 0 } )$ . Any rule depending only on � answers identically on inputs with diferent labels. □

Proposition 5.4 (General support lower bound). Let $( K , { \mathcal { F } } )$ be arbitrary and fix $n \geq 1$ Let $\mathcal { A } _ { n } \ : = \ : ( \mathrm { i m } \delta ^ { n - 1 } ) ^ { \perp } \ : \subseteq \ : C ^ { n } ( K ; \mathcal { F } ) ^ { * }$ be the annihilator of the coboundary image. For an �- simplex � write $\iota _ { \sigma } : { \mathcal { F } } ( \sigma ) \hookrightarrow C ^ { n } ( K ; { \mathcal { F } } )$ for the inclusion of the corresponding summand, put �|<sub>�</sub> $: = \eta \circ \iota _ { \sigma } \in \mathcal { F } ( \sigma ) ^ { * }$ , and set

$$
\begin{array} { r l } & { W _ { \sigma } = \operatorname { s p a n } \{ \eta | _ { \sigma } : \eta \in \mathcal { A } _ { n } \} \subseteq \mathcal { F } ( \sigma ) ^ { * } , } \\ & { E _ { n } ^ { * } = \bigcup _ { \eta \in \mathcal { A } _ { n } \backslash \{ 0 \} } \operatorname { s u p p } ( \eta ) } \\ & { \qquad = \{ \sigma \in K _ { n } : W _ { \sigma } \neq 0 \} . } \end{array}\tag{11}
$$

Suppose an exact decision rule for membership in im $( \delta ^ { n - 1 } )$ depends on the datum only through per-simplex observations $c _ { \sigma } \mapsto O _ { \sigma } ( c _ { \sigma } )$ , and that each $O _ { \sigma }$ is constant on the cosets of a subspace $V _ { \sigma } \subseteq { \mathcal { F } } ( \sigma )$ of ignored directions. Then $V _ { \sigma } \subseteq W _ { \sigma } ^ { \circ }$ for every �, where $W _ { \sigma } ^ { \circ } = \{ y \in \mathcal { F } ( \sigma ) : \phi ( y ) =$ 0 for all $\phi \in W _ { \sigma } \}$ . Thus every globally ignored linear direction must lie in $W _ { \sigma } ^ { \circ } ,$ ; in particular no simplex of $E _ { n } ^ { * }$ may be ignored entirely.

Proof. Since im $( \delta ^ { n - 1 } )$ is a subspace, $c \in \mathrm { i m } ( \delta ^ { n - 1 } )$ if and only if $\eta ( c ) = 0$ for all $\eta \in \mathcal { A } _ { n }$ Suppose $z \in V _ { \sigma } \ : \backslash \ : W _ { \sigma } ^ { \circ }$ for some $\sigma .$ . Then some $\phi \in W _ { \sigma }$ has $\phi ( z ) \neq 0$ , hence some $\eta \in \mathcal { A } _ { n }$ has $\eta ( \iota _ { \sigma } z ) \neq 0$ . Compare the two data 0 and $\iota _ { \sigma } \mathcal { Z }$ . They agree outside $\sigma ,$ and on $\sigma$ they difer by $z \in V _ { \sigma } ,$ , along which $O _ { \sigma }$ is constant; the rule therefore returns the same answer on both. Yet $0 \in \mathrm { i m } ( \delta ^ { n - 1 } )$ while $\eta ( \iota _ { \sigma } z ) \neq 0$ gives $\iota _ { \sigma } z \not \in \mathrm { i m } ( \delta ^ { n - 1 } )$ , so the rule errs on one of them. Hence $V _ { \sigma } \subseteq W _ { \sigma } ^ { \circ }$ . In particular, any linear direction that the observation discards must be annihilated by every restricted witness in $W _ { \sigma }$ . Choosing a basis of $\mathcal { F } ( \sigma )$ adapted to a decomposition $\mathcal { F } ( \sigma ) = W _ { \sigma } ^ { \circ }$ ⊕ � gives the coordinatewise reading: no nonzero direction in the complementary subspace � can be ignored by an exact rule. □

For the �-cycle with constant sheaf and $n = 1 , \mathcal { A } _ { 1 }$ is spanned by the full-support functional $\begin{array} { r } { \ell _ { m } ( c ) = \sum _ { i } \pm c _ { i , i + 1 } } \end{array}$ . Thus $E _ { 1 } ^ { * }$ contains every edge, and any exact decision procedure must depend on each edge datum. This is an observation lower bound: it specifies which inputs the decision may ignore. If edge data are initially local, information from every edge must reach the decision rule; a joint evaluation can aggregate those data in one operation. A round or bit lower bound is a diferent statement and needs additional assumptions about synchrony, output location, precision, randomisation, and message size; a protocol may aggregate or compress the raw cell values while still depending on all of them. We do not derive such a bound at this level of generality. Sections 9.1 and 9.2 prove one-round bottlenecks for two specific registered coordinator-output models, and the combinatorial topology of distributed computing [20] provides tools for the general analysis.

## 6. Structural Properties

## 6.1. Invariance Is Sheaf-Relative

For the constant sheaf, $H ^ { n } ( K ; \underline { { \mathbb { R } } } )$ is ordinary cohomology and is invariant under homotopy equivalence. The boundaries of a triangle and a square therefore carry the same degree-1 signature. For a general capacity sheaf, the shape of � does not determine cohomology (Sections 4.2 and 4.3); the relevant object is the pair $( K , { \mathcal { F } } )$ up to sheaf-preserving equivalence. Empirical reports must specify both.

## 6.2. Degree Is an Obstruction Type, Not a Ranking

For the constant sheaf, degree has a geometric interpretation: degree 1 corresponds to cyclelevel incompatibility, degree 2 to a two-dimensional cavity, and so on. This interpretation fails for a general capacity sheaf. Section 4.3, for example, gives a degree-1 class on a contractible complex. In that setting, degree identifies the cochain level at which a compatibility pattern fails to arise from lower-dimensional data. Multiplicities $\beta ^ { n }$ also require task-specific normalisation and behavioural interpretation. Proposition 5.1 gives a generic algebraic cochain-realization interpretation in every degree, while domain-specific semantics remain task dependent; Section 9.2 gives one explicit degree-2 instance.

## 7. Relation to Other Frameworks

This section positions TSI with respect to neighbouring approaches to multi-agent interaction, irreducibility, and distributed computation. Table 1 highlights the objects each framework studies and identifies empirical comparisons that distinguish their predictions from those of TSI.

Table 1 | Conceptual relations. Each row is an empirical comparison target, not an identification.
<table><tr><td>Framework</td><td>Primary object</td><td>Relation to TSI</td></tr><tr><td>Partial information de- composition [35]</td><td>Joint information unavailable from marginals</td><td>TSI represents interaction geometry and compatibility constraints; it assumes no information-theoretic decomposition. A direct empirical comparison is whether PID synergy and  $\beta ^ { n }$  dissociate on tasks with matched information but different interaction structure.</td></tr><tr><td>Cooperative game theory [31]</td><td>Coalition value function</td><td>TSI requires no transferable utility and compares no coalition values; it identifies sheaf-relative obstructions. A direct empirical comparison is whether coalition superadditivity and TSI signatures dissociate across matched task families.</td></tr><tr><td>Integrated information [23, 33]</td><td>Axiomatic integration measure</td><td>Both address irreducibility with different primitives; no equivalence is asserted. Shared benchmark tasks could support direct comparison where the corresponding integration measure is well-defined.</td></tr><tr><td>Sheaf-theoretic contextuality [1, 2]</td><td>Compatibility of local measurement sections with a global section</td><td>Shares the local-to-global obstruction pattern, but TSI uses cellular interaction complexes and task-specific capacity sheaves and evaluates system capability separately through registered interventions.</td></tr><tr><td>Sheaf neural networks [8, 17]</td><td>Learnable sheaves and sheaf Laplacians</td><td>One way to learn capacity sheaves from data (Section 8.2, Recipe B).</td></tr><tr><td>Combinatorial topology of distributed computing [20]</td><td>Task solvability via subdivisions</td><td>Tools for turning Proposition 5.4 into model-specific round and bit lower bounds.</td></tr></table>

## 8. Measurement and Evaluation Protocol

## 8.1. Procedure

1. Specify the task and interaction semantics. Define outputs, success criteria, and the communication budget. Apply the operational simplex criterion of Definition 3.1: distinguish pairwise relay from genuine joint interaction, with the inclusion rule written down before data collection.

2. Construct or learn the capacity sheaf using a recipe from Section 8.2, and freeze it (pre-register the construction pipeline) before any behavioural evaluation.

3. Compute the signature. Assemble the coboundary matrices and compute

$$
\beta ^ { n } ( K ; \mathcal { F } ) = \dim \ker ( \delta ^ { n } ) - \operatorname { r a n k } ( \delta ^ { n - 1 } ) .\tag{12}
$$

With inner products on cochains, the sheaf Hodge Laplacian

$$
L _ { \mathcal { F } } ^ { n } = ( \delta ^ { n } ) ^ { * } \delta ^ { n } + \delta ^ { n - 1 } ( \delta ^ { n - 1 } ) ^ { * }\tag{13}
$$

satisfies ker $L _ { \mathcal { F } } ^ { n } \cong H ^ { n } ( K ; \mathcal { F } ) \ [ 1 7 ]$ . Once inner products are chosen, the harmonic representative describes how the obstruction is distributed across cells. It need not be sparse or local; localisation requires a separate criterion or regulariser. These computations reduce to ranks and null spaces of the cochain matrices and are polynomial in matrix size.

4. Test class-indexed predictions. Compare systems with matched agents, parameters, and communication budgets but diferent interaction complexes. Use task families from Proposition 5.1 and matched controls whose classes vanish. Any performance diference should be specific to the class-carrying tasks.

5. Run causal ablations per Definition 3.6. Class-targeted interventions versus classpreserving controls of matched magnitude.

6. Report null results. A nonzero signature with no advantage on its own correspondence tasks is evidence against the capacity interpretation of that sheaf, even when the algebra is correct.

## 8.2. Three Recipes for Capacity Sheaves

If $\mathcal { F }$ is fitted after the results are known, it can be chosen to produce a desired signature.   
The following recipes constrain its construction.

Recipe A: consistency sheaves from shared variables. For each agent $\nu ,$ let $\mathcal { F } ( \nu )$ be its state or plan space. For each simplex $\sigma ,$ let $\mathcal { F } ( \sigma )$ contain the variables on which the agents of � must agree, such as shared observables, interface variables, or common resources. The map $\rho _ { \upsilon \sigma }$ extracts agent �’s report of those variables. This follows established sensor-integration constructions [17, 26, 27]: $H ^ { 0 }$ contains globally consistent joint states, while nonzero $H ^ { 1 }$ can encode compatibility obstructions for cocycle-level inconsistency data. For noisy or approximate assignments, Robinson’s consistency radius and consistency filtration provide complementary quantitative measures of disagreement and its localisation [28]. The system’s interface specification fixes the sheaf before evaluation.

Recipe B: learned sheaves via sheaf difusion. Parameterise the restriction maps and fit them by minimising sheaf Dirichlet energy on logged traces of successful coordination episodes, as in neural sheaf difusion [8]. Freeze the maps before computing cohomology and evaluating held-out task instances. Fitting the sheaf on evaluation episodes invalidates the TSI claim.

Recipe C: belief-fusion sheaves. This recipe is the one that makes the sheaf-alteration intervention of Definition 3.6 (2) physically realisable, since its controls act on � rather than on �. The full Gaussian information form $( \Lambda , \eta )$ cannot itself be a stalk because positivedefinite precision matrices form an open cone rather than a vector space. One compatible construction fixes $\Lambda _ { \nu } ~ \succ ~ 0$ as registered model parameters and uses $\mathcal { F } ( \nu ) = \mathbb { R } ^ { d }$ for the information vectors $\eta _ { \nu } ,$ with joint-observation spaces on the simplices and linear observation maps as restrictions. A second uses the ambient vector space $\mathrm { S y m } _ { d } ( \mathbb { R } ) \times \mathbb { R } ^ { d }$ as the stalk and imposes $\Lambda \succ 0$ separately as a feasibility constraint. In either case, changing a restriction map corresponds to a physical intervention such as sensor recalibration, so the controls in Definition 3.6 can be implemented in hardware or simulation.

## 9. Fixed-Task Benchmarks and Exhaustive Simulations

We evaluate the framework on two finite experiments that can be enumerated exactly. In each experiment, every system receives the same task model; only the system interaction structure and registered primitives change. If the task model also changed between systems, the comparison would mix a new problem, a new cohomology group, and a new solver. Section 9.1 studies a degree-1 cycle task, and Section 9.2 gives the degree-2 tetrahedralboundary analogue.

Fixed task model. The task model is $\left( K _ { T } , \mathcal { F } _ { T } \right) \ = \ \left( C _ { m } , \underline { { \mathbb { R } } } \right)$ for all systems. Write the cyclically oriented edges as $e _ { i } = [ i , i + 1 ( { \bmod { m } } ) ]$ , fix an integer bound $B \in \mathbb N$ with $B \geq 1$ , and let $c _ { i } \in \{ - B , \ldots , B \} \subset \mathbb { Z }$ be the datum on $e _ { i }$ . With this orientation,

$$
\ell _ { m } ( c ) = \sum _ { i = 0 } ^ { m - 1 } c _ { i } .
$$

Hence $\beta ^ { 1 } ( K _ { T } ; \mathcal { F } _ { T } ) = 1$ , the critical support is every edge, and realizability is $\ell _ { m } ( c ) = 0$ . The experiment evaluates every point in $\{ - B , \ldots , B \} ^ { m }$ under the uniform distribution. Since $K _ { T }$ is a graph, every 1-cochain is a cocycle; $\ell _ { m } ( c ) = 0$ is the realizability condition. A real-valued variant could instead use the promise problem $| \ell _ { m } ( c ) | \leq \varepsilon$ versus $| \ell _ { m } ( c ) | \geq 2 \varepsilon$

Input allocation and outputs. Agent � initially owns $c _ { i } ,$ , and agent 0 is the designated coordinator for the primary outputs. The benchmark tasks are:

• ${ \sf T } _ { 1 }$ (decision): agent 0 outputs whether $[ c ] = 0$

${ \sf T } _ { 3 }$ (designated-edge, minimum-support repair): when $[ c ] \neq 0$ , agent 0 outputs the correction $\textit { d } = - \ell _ { m } ( c )$ to the designated edge $e _ { 0 }$ . The correction has the smallest possible nonzero support, namely one edge. It need not minimize norm, magnitude, or application cost.

The realization task ${ \sf T } _ { 2 }$ from Proposition 5.1 remains part of the formal family. We omit it from the one-round comparison because an all-agent realization generally requires Θ(� log(��)) output bits or a richer personalised-output primitive.

Communication model Π. A system is a pair $( K _ { S } , \Pi )$ over the same � agents, distinct from the fixed task model $K _ { T }$ . Communication is synchronous. Let

$$
h _ { B } = \left\lceil \log _ { 2 } ( 2 B + 1 ) \right\rceil , \qquad q _ { m , B } = \left\lceil \log _ { 2 } ( 2 m B + 1 ) \right\rceil .
$$

The registered primitives are:

• Pairwise channel $\{ u , \upsilon \}$ : in one round, each endpoint may send an arbitrary message of at most � bits to the other endpoint.

• Joint channel on $\sigma \subseteq A { : }$ an atomic aggregation primitive whose function is fixed before evaluation. In one round every member of � receives the same $q _ { m , B } .$ -bit value $g _ { \sigma }$ of the members’ current inputs. The value is common knowledge within � and counts against each recipient’s congestion budget. We register $\begin{array} { r } { g _ { A } ( c ) = \sum _ { i = 0 } ^ { m - 1 } c _ { i } = \ell _ { m } ( c ) } \end{array}$ in advance as an additive reduction. This function, rather than the simplex alone, is what gives the joint channel its operational meaning under Definition 3.2.

• Congestion budget: every agent may send and receive at most $\beta$ bits in total per round across all available communication primitives; the received value of a joint channel is included in this total. Registered instances satisfy

$$
q _ { m , B } \leq \beta < ( m - 1 ) h _ { B } ,
$$

so the joint reduction is admissible while an exact one-round pairwise collection at the coordinator is congested. Such conditions can be found in applications like over-the-air computation [15].

Every recipient is charged for each invoked joint primitive’s $q _ { m , B } { \mathrm { - b i t } }$ output. For every deterministic protocol, the subset of available primitives that it invokes is fixed as part of the registered protocol before the input is revealed; invocation decisions are not input-dependent. Only invoked outputs are charged. The joint primitive is an atomic aggregation service: access to its participants’ current inputs is part of its registered semantics. The benchmark evaluates the pre-registered reduction $g _ { A }$ under the stated output-width and congestion cost.

The communication model is CONGEST-inspired: it combines synchronous rounds and bounded pairwise messages with an aggregate per-agent bandwidth budget [25]. Proposition 9.1 is a communication-complexity argument [21]. TSI identifies the task class and the registered primitive predicted to remove the bottleneck; the lower bound itself comes from the communication model.

Proposition 9.1 (One-round pairwise bottleneck). Fix $m \geq 3 .$ . In the coordinator-output model above, no deterministic one-round pairwise protocol can solve ${ \sf T } _ { 1 }$ whenever $\beta < ( m - 1 ) h _ { B }$ . This remains true when every non-coordinator has a direct channel to agent 0.

Proof. Consider any non-coordinator agent $i \in \{ 1 , \ldots , m - 1 \}$ . If its one-round message to agent 0 were the same for two distinct values $x , x ^ { \prime } \in \{ - B , \ldots , B \}$ , choose an index $j \in \left\{ 1 , \dots , m - 1 \right\} \setminus \left\{ i \right\}$ and set $c _ { j } = - x _ { ; }$ , all remaining coordinates (including $c _ { 0 } )$ to zero. The two inputs obtained by setting $c _ { i } = x$ and $c _ { i } = x ^ { \prime }$ give identical messages to the coordinator, because every other local input is unchanged, yet the first has $\ell _ { m } ( c ) = 0$ and the second has $\ell _ { m } ( c ) = x ^ { \prime } - x \neq 0$ . Hence agent �’s message must distinguish all $2 B + 1$ possible values and therefore contains at least $h _ { B }$ bits. This holds for each of the $m - 1$ non-coordinator agents. In one synchronous round their information can reach agent 0 only through messages incident to agent 0, which require at least $( m - 1 ) h _ { B }$ received bits in total. The congestion bound rules this out. □

Systems. All systems use the same agents, input allocation, and external per-agent congestion budget. For the primary comparison, let

$$
K _ { 0 } = { \mathrm { s k e l } _ { 1 } } ( \Delta ^ { m - 1 } )
$$

be the complete pairwise 1-skeleton on the � agents.

$S _ { \mathrm { p a i r } } \mathrm { : }$ pairwise channels on $K _ { 0 }$ only, with no higher-dimensional simplex. The outercycle task class extends to this graph: take a representative on $C _ { m }$ and set it to zero on the added chords. Because $K _ { 0 }$ is one-dimensional, this remains a cocycle whose restriction is the original nonzero class.

$S _ { \mathrm { f i l l } } \mathrm { : }$ : the same complete pairwise channels plus one joint channel on all � agents. Its induced interaction complex is the full simplex $\Delta ^ { m - 1 }$ , equipped with the canonical constant-sheaf extension. The restriction map

$$
\begin{array} { r } { \boldsymbol { H } ^ { 1 } ( \Delta ^ { m - 1 } ; \underline { { \mathbb { R } } } ) = \boldsymbol { 0 } \longrightarrow \boldsymbol { H } ^ { 1 } ( C _ { m } ; \underline { { \mathbb { R } } } ) } \end{array}
$$

has zero image, so the task class does not extend and the joint primitive is class-filling. The registered reduction $g _ { A } = \ell _ { m }$ solves ${ \sf T } _ { 1 }$ and ${ \sf T } _ { 3 }$ in one round because the tasks ask for its zero-test and designated-edge repair.

$S _ { \mathrm { r i n g } } { \mathrm { : } }$ : pairwise ring channels only, with induced complex $C _ { m }$ . This is a base-topology sanity check, not the primary ablation control.

Same-1-skeleton primitive-closure ablation. Relative to the registered system base $K _ { 0 }$ the all-agent joint primitive in $S _ { \mathrm { f i l l } }$ generates

$$
\mathsf { P } _ { A } ( K _ { 0 } ) = \{ \tau \subseteq A : | \tau | \geq 3 \} .
$$

The class-targeted ablation removes the joint primitive and every higher-order face in this closure, leaving the complete pairwise 1-skeleton unchanged. The resulting system is � <sub>i</sub> , and the task class again extends to its one-dimensional complex. This is the full primitiveclosure removal required by Definition 3.6; pairwise communication is identical on both sides of the comparison.

## 9.1. Degree-1 Exhaustive Simulation

We instantiate $m = 6 , B = 2$ , so $h _ { B } = 3 , q _ { m , B } = 5$ , and $\beta ~ = ~ 5 ~ \in ~ [ q _ { m , B } , ( m ~ - ~ 1 ) h _ { B } )$ . We set $b = \beta = 5$ and enumerate all $5 ^ { 6 } = 1 5 { , } 6 2 5$ inputs; 1,751 (11.21%) are realizable and $^ { 1 3 , 8 7 4 }$ are unrealizable. The deterministic simulation and exhaustive optimiser are provided in benchmark\_sim.py in the versioned code repository [11]. Execution instructions and expected results are provided in the repository README.

The filled system computes $g _ { A }$ within budget and solves ${ \sf T } _ { 1 }$ and ${ \sf T } _ { 3 }$ exactly. In $S _ { \mathrm { p a i r } } ,$ only messages received by agent 0 can afect its output after one synchronous round. Edges among non-coordinators do not change that view. A deterministic $b _ { i }$ -bit message from agent � partitions $\{ - 2 , - 1 , 0 , 1 , 2 \}$ into at most $2 ^ { b _ { i } }$ blocks. For any fixed tuple of partitions, the optimal coordinator chooses the majority label for ${ \sf T } _ { 1 }$ and the modal correction for ${ \sf T } _ { 3 }$ in each induced view. Refinement cannot reduce either optimum. We therefore search maximal partitions for the undominated receive-bit allocations summing to $\beta = 5 \colon ( 1 , 1 , 1 , 1 , 1 ) , ( 2 , 1 , 1 , 1 ) , ( 2 , 2 , 1 )$ $( 3 , 1 , 1 )$ , and (3, 2). Allocations granting any single sender more than $h _ { B } \ = \ 3$ bits, such as (4, 1) and (5), are dominated and need not be searched: three bits already distinguish all $2 B + 1 = 5$ values of that sender’s input, so additional bits cannot refine its message partition and can instead be allocated to another sender, as in (3, 2). This search covers every deterministic one-round complete-pairwise protocol under the registered receive budget. Table 2 reports the exact optima.

Table 2 | Exhaustive evaluation for $m = 6 , B = 2 .$ , and $\beta = 5 . ~ { \mathsf { T } } _ { 1 }$ is evaluated on all 15,625 inputs; ${ \sf T } _ { 3 }$ on the 13,874 unrealizable inputs.
<table><tr><td>System</td><td> ${ \boldsymbol { \mathsf { T } } } _ { 1 }$ </td><td> ${ \boldsymbol { \mathsf { T } } } _ { 3 }$ </td><td>Control</td></tr><tr><td> $S _ { \mathrm { f i l l } }$  (atomic  $g _ { A } )$ </td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td> $S _ { \mathrm { p a i r } }$  (same 1-skeleton)</td><td>0.8880</td><td>0.2663</td><td>1.0000</td></tr><tr><td> $S _ { \mathrm { r i n g } }$  (sanity check)</td><td>0.8879</td><td>0.1687</td><td>1.0000</td></tr><tr><td>Majority-class baseline</td><td>0.8879</td><td></td><td></td></tr></table>

The complete-pairwise optimum decides 13,875 of 15,625 inputs correctly, a single input more than the always-unrealizable majority-class baseline and therefore essentially at that baseline at this budget, and returns the exact repair on 3,694 of 13,874 unrealizable inputs. The ring matches the baseline on ${ \sf T } _ { 1 }$ and returns 2,340 exact repairs. Removing the higherorder primitive while preserving the full 1-skeleton therefore loses 1,750 correct decisions and 10,180 exact repairs. The control task is $c _ { 0 } + c _ { 1 } = 0$ , whose registered model has no relevant cycle class. One $h _ { B } = 3$ bit message over the edge sufices, so every system solves it exactly. The observed advantage is specific to the class-carrying tasks in this experiment.

Benchmark results. Proposition 9.1 and the one-round solution of $S _ { \mathrm { f i l l } }$ follow analytically from Π and the registered additive reduction. The entries in Table 2 are exact finite computations: the script enumerates the input domain and all deterministic protocols in the stated search. The degree-1 experiment establishes the performance component, a class-restoring primitive-closure ablation, and task specificity through the trivial-cohomology control. Section 9.2 adds a degree-2 same-2-skeleton benchmark and status-preserving stress tests; an exactly cost-matched status-preserving intervention remains part of Hypothesis 9.3.

## Pre-registered hypotheses for learned agents.

Hypothesis 9.1 (Learned-agent scaling). For bounded learned agents, the success–round trade-of of $S _ { \mathrm { r i n g } }$ and $S _ { \mathrm { p a i r } }$ worsens with $m ,$ whereas $S _ { \mathrm { f i l l } }$ approaches the registered one-round optimum.

Hypothesis 9.2 (Specificity). $S _ { \mathrm { f i l l } } \acute { s }$ advantage over $S _ { \mathrm { p a i r } }$ appears on ${ \sf T } _ { 1 }$ and ${ \mathsf { T } } _ { 3 } ,$ whose fixed model carries the class, and does not appear on matched control tasks over the same data whose registered models carry no relevant class.

Hypothesis 9.3 (Ablation symmetry). Removing the full primitive closure $\mathsf { P } _ { A } ( K _ { 0 } )$ from $S _ { \mathrm { f i l l } }$ returns the system to the complete pairwise 1-skeleton and restores extension of the task class; for learned agents, it degrades ${ \sf T } _ { 1 }$ and ${ \sf T } _ { 3 }$ more than a matched extension-status-preserving intervention of equal registered cost.

Falsification conditions. The learned-agent interpretation fails under the registered budgets if $S _ { \mathrm { f i l l } }$ has no advantage on ${ \sf T } _ { 1 }$ and ${ \sf T } _ { 3 }$ , if extension-status-preserving controls produce the same advantage, or if the advantage appears equally on tasks without the class. The finite protocol claim has a simpler test: a deterministic one-round pairwise protocol that matches $S _ { \mathrm { f i l l } }$ within the stated congestion budget would contradict Proposition 9.1 or expose an error in the simulation assumptions.

## 9.2. A Degree-2 Benchmark with Status-Preserving Stress Tests

The experiment of Section 9.1 fixes a degree-1 task model. We now repeat the design one cochain degree higher. The degree-2 benchmark is implemented in benchmark\_deg2.py; the accelerated three-compressor routine for the full separation window is in fastopt3.py [11]. This has two purposes: it instantiates Proposition 5.1 in degree 2, and it tests whether the class-filling separation persists when all three-agent interactions are retained. We also report status-preserving stress tests relevant to condition (3) of Definition 3.6. These stress tests are deliberately described as directional controls rather than as a proof of full operational support, because their registered costs are not exactly matched to the class-restoring ablation in every cost dimension.

Fixed task model. Let $A = \{ 0 , 1 , 2 , 3 \}$ and let the task model be $\left( K _ { T } , \mathcal { F } _ { T } \right) = \left( \partial \Delta ^ { 3 } , \underline { { \mathbb { R } } } \right)$ , the boundary of the tetrahedron with the constant sheaf: all four vertices, all six edges, all four triangles, and no 3-simplex. Then

$$
C ^ { 0 } \cong \mathbb { R } ^ { 4 } , \qquad C ^ { 1 } \cong \mathbb { R } ^ { 6 } , \qquad C ^ { 2 } \cong \mathbb { R } ^ { 4 } , \qquad C ^ { 3 } = 0 , \qquad { \mathrm { r a n k } } \delta ^ { 0 } = { \mathrm { r a n k } } \delta ^ { 1 } = 3 , \qquad 
$$

so $H ^ { 0 } \cong \mathbb { R } , H ^ { 1 } ( K _ { T } ; \mathbb { R } ) = 0$ , and

$$
H ^ { 2 } ( \partial \Delta ^ { 3 } ; \mathbb { R } ) \cong \mathbb { R } ^ { 4 } / \mathrm { i m } \delta ^ { 1 } \cong \mathbb { R } , \qquad \beta ^ { 2 } ( K _ { T } ; \mathcal { F } _ { T } ) = 1 .
$$

Orient the triangles as the faces of [0, 1, 2, 3]. Under the standard Euclidean pairing, the annihilator $\mathcal { A } _ { 2 } = ( \mathrm { i m } \delta ^ { 1 } ) ^ { \perp }$ is spanned by the total-flux functional

$$
\ell _ { 3 } ( c ) = c _ { 1 2 3 } - c _ { 0 2 3 } + c _ { 0 1 3 } - c _ { 0 1 2 } .\tag{14}
$$

Equivalently, if the missing 3-simplex is added, then evaluating $\delta ^ { 2 } c$ on [0, 1, 2, 3] gives $\ell _ { 3 } ( c )$ Hence $\ell _ { 3 } \circ \delta ^ { 1 } = 0$ . Since $K _ { T }$ is 2-dimensional, every 2-cochain is a cocycle, and Proposition 5.1 gives

$$
\mathrm { R } ^ { 2 } ( c ) { \mathrm { ~ s o l v a b l e } } \quad \iff \quad [ c ] = 0 \quad \iff \quad \ell _ { 3 } ( c ) = 0 .
$$

By Proposition 5.4, the critical support $E _ { 2 } ^ { * }$ is all four 2-simplices.

This is the degree-2 counterpart of Section 4.1: there, the task class lies on the boundary of a triangle; here it lies on the boundary of a tetrahedron. Adding [0, 1, 2, 3] gives $\delta ^ { 2 } ( c ) ( [ 0 , 1 , 2 , 3 ] ) = \ell _ { 3 } ( c )$ , so ker $\delta ^ { 2 } = \mathrm { i m } \delta ^ { 1 }$ and $H ^ { 2 } ( \Delta ^ { 3 } ; \mathbb { R } ) = 0$ . The corresponding four-agent system primitive is therefore class-filling for this task class.

Task family. The unknowns now live on the edges. Each edge $e =  { \left\{ i , j \right\} }$ carries a bilateral parameter $x _ { i j } \in \mathcal { F } _ { T } ( e )$ , and each triangle � carries a prescribed closure defect $c _ { \sigma }$ . In the additive constant-sheaf model, $( \delta ^ { 1 } x ) ( \sigma )$ is the oriented sum of the three edge parameters around �. The realization task $\mathsf { R } ^ { 2 } ( c )$ asks for edge parameters satisfying $\delta ^ { 1 } x = c ,$ and $\mathsf { D } ^ { 2 } ( c )$ asks whether the prescribed triangle-defect profile is realizable.

Algebraically, realizability requires the oriented sum of the four face defects to vanish, $\ell _ { 3 } ( c ) = 0$ . This is the condition used by the benchmark. For the constant sheaf it also has a standard discrete-exterior-calculus interpretation: the unknown 1-cochain � is a discrete vector potential on the edges, the prescribed 2-cochain � assigns a flux to each face, and $\delta ^ { 1 } x = c$ is the discrete analogue of a curl equation on a triangulated surface. In this analogy, $\ell _ { 3 } ( c ) = 0$ is the corresponding discrete Stokes compatibility condition: the oriented total flux through the closed surface $\partial \Delta ^ { 3 } \simeq S ^ { 2 }$ vanishes. The critical support is every face because $\ell _ { 3 }$ depends on all four face defects, and the registered four-agent primitive evaluates this sum in one step. The degree-1 family of Section 5.2 is the analogous algebraic construction one degree lower, with unknowns on vertices and prescribed data on edges.

Beyond this discrete-calculus reading, application-specific interpretations of the edge variables and triangle defects require a domain-specific sheaf and are not assumed by the benchmark.

Input allocation and outputs. Fix $B \in \mathbb N$ . Each agent � holds a private share $s _ { \upsilon , \sigma } \in$ $\{ - B , \ldots , B \}$ for every triangle $\sigma \geqslant \nu _ { : }$ , and

$$
c _ { \sigma } = \sum _ { \nu \in \sigma } s _ { \nu , \sigma } , \qquad \ell _ { 3 } ( c ) = \sum _ { \nu \in A } a _ { \nu } , \qquad a _ { \nu } : = \sum _ { \sigma \ni \nu } \varepsilon _ { \sigma } s _ { \nu , \sigma } ,
$$

where $\varepsilon _ { \sigma } \in \{ \pm 1 \}$ is the sign of $\sigma$ in (14). The shares make the triangle data genuinely distributed. Whether the three shares of a triangle can be aggregated atomically is a property of the system interaction model $K _ { S }$ , not of the fixed task complex $K _ { T }$ . Agent 0 is the coordinator. We use the degree-2 analogues

$\mathsf { T } _ { 1 } ^ { ( 2 ) }$ (decision): agent 0 outputs whether $[ c ] = 0 ;$

$\mathsf { T } _ { 3 } ^ { ( 2 ) }$ (designated-simplex, minimum-support repair): when $[ c ] \neq 0 .$ , agent 0 outputs the correction to a fixed designated 2-simplex � . If $\varepsilon _ { \sigma _ { 0 } } \in \{ \pm 1 \}$ is its coeficient in $\ell _ { 3 } ,$ the required correction is $d = - \varepsilon _ { \sigma _ { 0 } } \ell _ { 3 } ( c )$ . Thus a correction supported on one 2-simplex always exists, and computing it is equivalent to recovering $\ell _ { 3 } ( c )$ exactly.

Communication model. The model follows Section 9.1: one synchronous round, deterministic protocols, coordinator output, and a per-agent congestion budget. As in the degree-1 instantiation, we set the pairwise per-channel cap to $b = \beta$ . Let

$$
\begin{array} { r l } & { h _ { B } = \left\lceil \log _ { 2 } ( 2 B + 1 ) \right\rceil , \qquad w _ { B } = \left\lceil \log _ { 2 } ( 6 B + 1 ) \right\rceil , } \\ & { q _ { \sigma } = \left\lceil \log _ { 2 } ( 6 B + 1 ) \right\rceil , \qquad q _ { A } = \left\lceil \log _ { 2 } ( 2 4 B + 1 ) \right\rceil . } \end{array}
$$

The registered primitives are pairwise channels, three-agent primitives $g _ { \sigma } = c _ { \sigma }$ of output width $q _ { \sigma } \mathrm {             . }$ , and an all-agent primitive $g _ { A } = \ell _ { 3 } ( c )$ of output width $q _ { A }$ . For every deterministic protocol, the subset of available primitives that it invokes is fixed as part of the registered protocol before the input is revealed; invocation decisions are not input-dependent. Only invoked outputs are charged. In one round every message and every primitive output is a simultaneous function of the initial shares; an agent therefore cannot receive a primitive output and forward it within the same round.

Systems. All systems have the same agents, input allocation, and per-agent budget $\beta$ . For the primary class-restoring ablation the registered base is $K _ { 0 } = \partial \Delta ^ { 3 }$

$S _ { \mathrm { f i l l } } \colon$ complete pairwise channels, the four three-agent primitives, and the all-agent primitive. Its induced complex is $\Delta ^ { 3 }$ with the canonical constant-sheaf extension, and

$$
\boldsymbol { \jmath } ^ { * } : \boldsymbol { H } ^ { 2 } ( \Delta ^ { 3 } ; \underline { { \mathbb { R } } } ) = 0 \longrightarrow \boldsymbol { H } ^ { 2 } ( K _ { T } ; \underline { { \mathbb { R } } } ) \cong \mathbb { R }
$$

has zero image, so the all-agent primitive is class-filling.

$S _ { \mathrm { f a c e } } { \mathrm { : } }$ : the same system with the all-agent primitive removed, i.e. the full primitiveclosure ablation

$$
\mathsf { P } _ { A } ( K _ { 0 } ) = \{ \tau \subseteq A : | \tau | \geq 4 \} = \{ [ 0 , 1 , 2 , 3 ] \} .
$$

Its induced complex is $\partial \Delta ^ { 3 } = K _ { T ; }$ , so the class extends and the ablation is class-restoring. Panel (c) of Figure 1 depicts this configuration. This is a same-2-skeleton ablation: all four three-agent primitives are retained.

$S _ { \mathrm { p a i r } } \mathrm { : }$ pairwise channels only, with induced complex $\mathsf { s k e l } _ { 1 } ( \Delta ^ { 3 } )$ . This is a base-topology sanity check rather than the primary ablation control.

Proposition 9.2 (One-round bottleneck in degree 2). In the coordinator-output model above, no deterministic one-round protocol of $S _ { \mathrm { p a i r } }$ solves $\mathsf { T } _ { 1 } ^ { ( 2 ) }$ when $\beta < 3 w _ { B }$ . The same bound holds $f o r \ S _ { \mathrm { f a c e } } ,$ despite its four additional registered three-agent primitives.

Proof. For $S _ { \mathrm { p a i r . } }$ , the argument of Proposition 9.1 applies with $a _ { \nu }$ in place of $c _ { i }$ . If agent $i \neq 0$ sent the same message for two local inputs with $a _ { i } \neq a _ { i } ^ { \prime } ,$ the remaining independent shares can be completed so that the first instance has $\ell _ { 3 } = 0 ;$ the second then has $\ell _ { 3 } = a _ { i } ^ { \prime } - a _ { i } \neq 0$ while the coordinator’s view is unchanged. Hence each of the three non-coordinators must distinguish all $6 B + 1$ possible values of $a _ { i }$ and therefore send at least $w _ { B }$ bits. Exactness requires at least $3 w _ { B }$ received bits at agent 0.

For $S _ { \mathrm { f a c e } }$ , only the three triangle primitives containing agent 0 can afect the coordinator in one round. Let $T$ be the subset of those primitives that are invoked, and let $R = K _ { T , 2 } \ \backslash \ T$

The coordinator knows the signed triangle contributions for $\sigma \in T$ exactly. For each $\nu \neq 0$ the residual statistic is

$$
b _ { \nu } = \sum _ { \sigma \in R \atop \nu \in \sigma } \varepsilon _ { \sigma } s _ { \nu , \sigma } ,
$$

a signed sum of $k _ { \nu } = | \{ \sigma \in R : \nu \in \sigma \} |$ independent shares. The same fooling argument forces at least $\lceil \log _ { 2 } ( 2 B k _ { \nu } + 1 ) \rceil$ bits from agent �. Thus every exact protocol has receive cost at least

$$
q _ { \sigma } | T | + \sum _ { \nu = 1 } ^ { 3 } \left[ \log _ { 2 } ( 2 B k _ { \nu } + 1 ) \right] .
$$

It remains to minimise this over the eight subsets $T \subseteq S { \mathfrak { t } } ( 0 )$ . Each $\nu \neq 0$ lies in three triangles, two of which contain agent 0, so $k _ { \nu } = 3 - | T \cap \mathop { S } \mathfrak { t } ( \nu ) |$ . Writing $u _ { B } = \lceil \log _ { 2 } ( 4 B + 1 ) \rceil$ and using $q _ { \sigma } = w _ { B }$ , the four cases up to symmetry give

$$
\begin{array} { r l } & { | T | = 0 : k = ( 3 , 3 , 3 ) , \cosh = 3 w _ { B } , } \\ & { | T | = 1 : k = ( 2 , 2 , 3 ) , \cosh = 2 w _ { B } + 2 u _ { B } , } \\ & { | T | = 2 : k = ( 1 , 2 , 2 ) , \cosh = 2 w _ { B } + h _ { B } + 2 u _ { B } , } \\ & { | T | = 3 : k = ( 1 , 1 , 1 ) , \cosh = 3 w _ { B } + 3 h _ { B } . } \end{array}
$$

Since $6 B + 1 < 2 ( 4 B + 1 )$ we have $w _ { B } \leq u _ { B } + 1$ , and $u _ { B } \geq 3$ for every $B \geq 1$ ; hence the $\left| T \right| = 1$ cost exceeds the $\left| T \right| = 0$ cost by $2 u _ { B } - w _ { B } \geq u _ { B } - 1 \geq 2 > 0$ . The $\left| T \right| = 2$ cost exceeds the $| T | = 1$ cost by $h _ { B } > 0$ , and the $\left| T \right| = 3$ cost exceeds the $\left| T \right| = 0$ cost by $3 h _ { B } > 0$ . The minimum is therefore attained at $T = \emptyset$ and equals $3 w _ { B } \colon$ invoking any of the three-agent outputs available to the coordinator strictly increases the exact-protocol lower bound. □

Since $S _ { \mathrm { f i l l } }$ needs only the $q _ { A }$ -bit output of $g _ { A ; }$ , the registered separation window is

$$
q _ { A } \leq \beta < 3 w _ { B } .
$$

Exhaustive evaluation. We instantiate $B = 1$ , so $h _ { B } = 2 , w _ { B } = q _ { \sigma } = 3$ , and $q _ { A } = 5 ;$ the registered separation window is $5 ~ \leq ~ \beta ~ < ~ 9$ . The input domain is $\{ - 1 , 0 , 1 \} ^ { 1 2 } \mathrm { ~ }$ , i.e. $3 ^ { 1 2 } = \bar { 5 } 3 1 , 4 4 1$ inputs, of which 73,789 (13.88%) are realizable and 457,652 are unrealizable. We exhaustively optimise all four integer budgets $\beta \in \{ 5 , 6 , 7 , 8 \}$ in this window.

The search is exhaustive over all deterministic one-round coordinator-output protocols by the following reduction. Write � for the statistic agent 0 knows exactly after its chosen primitive invocations, so that

$$
\ell _ { 3 } \ = \ U + \sum _ { \nu \ne 0 } { b _ { \nu } } ,
$$

with $b _ { \nu }$ as in the proof of Proposition 9.2. The cochains � and $b _ { 1 } , b _ { 2 } , b _ { 3 }$ are signed sums over pairwise disjoint sets of shares; since the shares are drawn independently and uniformly, these four statistics are mutually independent. Both tasks depend on agent $\nu \mathbf { \bar { s } }$ input only through ${ { b } _ { \nu } } ,$ so a message that separates two inputs with the same $b _ { \nu }$ refines the coordinator’s view without changing the conditional distribution of $\ell _ { 3 } ;$ an optimal deterministic message is therefore, without loss of generality, a function of $b _ { \nu }$ alone, i.e. a partition of the attainable values of $b _ { \nu }$ into at most $2 ^ { b }$ blocks for a �-bit message. For any fixed tuple of partitions the optimal coordinator takes the majority label for $\mathsf { T } _ { 1 } ^ { ( 2 ) }$ and the modal value for $\mathsf { T } _ { 3 } ^ { ( 2 ) }$ in each induced view, and refining a partition cannot lower either optimum; granting a sender more bits likewise cannot lower it. It therefore sufices to search the maximal partitions of each $b _ { \nu } { \cdot } s { \mathrm { u p p o r t } }$ over the undominated receive-bit allocations, exactly as in Section 9.1.

At $\beta = 5$ and $\beta = 6$ , some undominated allocations leave three senders simultaneously compressed. To make these cases tractable without changing the protocol class, the optimiser exploits the additivity of the objective over message-block triples: the contribution of each distinct block triple is computed once and then gathered over all maximal partition triples. This is an implementation-level acceleration of the same exhaustive partition search. It reproduces the direct optimiser’s previously reported optima at $\beta = 7 , 8$ and completes the two remaining budgets. Randomised protocols are outside this search, as they are in degree 1. Table 3 reports the resulting exact optima; $\mathsf { T } _ { 1 } ^ { ( 2 ) }$ and $\mathsf { T } _ { 3 } ^ { ( 2 ) }$ are optimised separately.

Table 3 | Exhaustive evaluation across the complete registered separation window for the degree-2 task model $( \partial \Delta ^ { 3 } , \mathbb { R } )$ with $B = 1$ . Each system entry reports $( \mathsf { T } _ { 1 } ^ { ( 2 ) } , \mathsf { T } _ { 3 } ^ { ( 2 ) } ) ; \mathsf { T } _ { 1 } ^ { ( 2 ) }$ is evaluated on all 531,441 inputs and $\mathsf { T } _ { 3 } ^ { ( 2 ) }$ on the 457,652 unrealizable inputs. The trivialcohomology control is solved exactly by every system at every reported budget.
<table><tr><td> $\beta$ </td><td> $s _ { \mathrm { f i l } }$ </td><td> $S _ { \mathbf { f a c e } }$ </td><td> $s _ { \mathbf { p a i r } }$ </td><td>Majority baseline  $\mathsf { T } _ { 1 } ^ { ( 2 ) }$ </td></tr><tr><td>5</td><td>1.0000/1.0000</td><td>0.8884/0.4666</td><td>0.8884/0.4666</td><td>0.8612</td></tr><tr><td>6</td><td>1.0000/1.0000</td><td>0.9295/0.6299</td><td>0.9295/0.6299</td><td>0.8612</td></tr><tr><td>7</td><td>1.0000/1.0000</td><td>0.9453/0.7193</td><td>0.9453/0.7193</td><td>0.8612</td></tr><tr><td>8</td><td>1.0000/1.0000</td><td>0.9681/0.8401</td><td>0.9681/0.8401</td><td>0.8612</td></tr></table>

For $\beta = 5 , 6 , 7 , 8$ , respectively, the optimum of $S _ { \mathrm { f a c e } }$ decides 472,149, 493,965, 502,361, and 514,509 of 531,441 inputs correctly, and returns the exact repair on 213,563, 288,257, 329,211, and 384,488 of 457,652 unrealizable inputs. At every budget the optimum invokes none of the three three-agent primitives whose outputs are visible to the coordinator. The fourth face primitive, on {1, 2, 3}, cannot afect the coordinator’s output within the same simultaneous round. Forcing one coordinator-visible face primitive yields (0.8612, 0.2790), (0.8672, 0.3841), (0.8763, 0.4581), and (0.8944, 0.5362) across $\beta ~ = ~ 5 , 6 , 7 , 8$ , respectively; forcing two is infeasible at $\beta = 5$ and yields (0.8612, 0.2386), (0.8612, 0.2748), and (0.8612, 0.3383) at $\beta = 6 , 7 , 8$

Thus, throughout the complete registered separation window, the exhaustive optimum of $S _ { \mathrm { f a c e } }$ is attained without using any coordinator-visible three-agent primitive and is identical to the optimum of $S _ { \mathrm { p a i r } }$ . This equality is not a rounding artefact: every $S _ { \mathrm { p a i r } }$ protocol is available to $S _ { \mathrm { f a c e } } ,$ so the latter can never score lower. Proposition 9.2 predicts the failure of exact protocols without the class-filling primitive, and the exhaustive search shows that the additional local three-agent services also fail to improve the optimal approximate performance anywhere in the registered window.

Relative to $S _ { \mathrm { p a i r } } ,$ the system $S _ { \mathrm { f a c e } }$ adds a larger catalogue of local joint operations: four three-agent primitives, each of width $q _ { \sigma } = 3$ , for a total registered output width of $4 q _ { \sigma } = 1 2$ bits across the four services. This total is not a fully matched capacity measure, because the primitives have diferent recipient sets and only invoked outputs consume congestion. Nevertheless, throughout the registered separation window the exhaustive optimum of $S _ { \mathrm { f a c e } }$ is identical to that of $S _ { \mathrm { p a i r } }$ . By contrast, adding the single $q _ { A } = 5 – \mathrm { b i t }$ all-agent primitive makes both tasks exact. Thus, within this registered model, merely adding the larger catalogue of local three-agent primitives does not improve these tasks; the successful primitive is specifically the one whose induced 3-simplex is class-filling. This is evidence about the semantics and placement of the registered primitives and the extension status of the class, not a general claim that the quantity of joint capacity is irrelevant.

Control task. The trivial-cohomology control uses the full 2-simplex $\{ 0 , 1 , 2 \}$ with the task “is $c _ { 0 1 2 } = 0 ? "$ . For the constant sheaf on this simplex, $\delta ^ { 1 }$ is surjective and $H ^ { 2 } = 0$ . With the same share semantics, two $h _ { B } = 2 \mathrm { - } \mathsf { b i t }$ messages from agents 1 and 2 to agent 0 sufice, so every system solves the control exactly for $\beta \geq 4$

Status-preserving robustness checks. Condition (3) of Definition 3.6 asks for a statuspreserving intervention matched to the class-restoring intervention in registered cost. The present finite experiment includes two useful robustness checks, but they are not matched controls.

C1 (service removal). Disable two standalone three-agent service primitives while retaining the all-agent primitive $g _ { A }$ . Under the primitive-closure convention of Definition 3.6, retaining $g _ { A }$ keeps the induced interaction complex equal to $\Delta ^ { 3 }$ , so the extension status is unchanged. The exact solver uses only $g _ { A }$ , and performance remains (1.0000, 1.0000).

C2 (budget reduction). Reduce the per-agent congestion budget from $\beta = 8$ to $\beta = 5$ The interaction complex and extension status are unchanged, and $g _ { A }$ still fits in budget, so performance remains (1.0000, 1.0000).

Both interventions remove registered capacity without changing extension status and leave the exact solution intact. However, C1 and C2 alter diferent cost coordinates from the $S _ { \mathrm { f i l l } }  S _ { \mathrm { f a c e } }$ ablation. They therefore provide status-preserving robustness evidence, not the exactly cost-matched control required to conclude condition (3).

Scope of the evidence. Proposition 9.2 and the one-round solution of $S _ { \mathrm { f i l l } }$ follow analytically from the registered communication model and additive reduction; the entries of Table 3 are exact finite computations over the full input domain for all four integer budgets in the registered separation window. Together with Section 9.1, the degree-2 experiment establishes a second class-specific finite-model separation, now under a same-2-skeleton class-restoring ablation, and provides status-preserving robustness evidence rather than a matched-control result. It does not by itself establish full operational support under Definition 3.6, because an exactly cost-matched status-preserving intervention is still absent. As in degree 1, the finite computations do not measure learned-agent behaviour.

The general pattern. For every $n \geq 1$ , the constant-sheaf task model $( \partial \Delta ^ { n + 1 }$ , ℝ) has $\beta ^ { n } = 1$ . Its top-degree cohomology is detected by the oriented facet-sum functional

$$
\ell ^ { ( n ) } ( c ) = \sum _ { i = 0 } ^ { n + 1 } ( - 1 ) ^ { i } c ( d _ { i } [ 0 , \ldots , n + 1 ] ) ,
$$

whose support is all �-simplices. Adding the missing $( n + 1 )$ -simplex kills this top-degree class. Section 4.1 is the � = 1 boundary-simplex instance of this pattern; the cycle benchmark of Section 9.1 is a broader degree-1 analogue, while the present section gives the $n = 2$ boundary-of-a-tetrahedron instance.

A further degree-2 observation is specific to invertible rank-one restrictions. On $\partial \Delta ^ { 3 } \simeq S ^ { 2 }$ any rank-one local system over ℝ is trivial because $\pi _ { 1 } ( S ^ { 2 } ) = 0$ Thus the monodromy mechanism of Section 4.2, which can remove the degree-1 class on a cycle, has no nontrivial rank-one local-system analogue on the tetrahedral boundary. This statement does not cover general capacity sheaves with noninvertible or higher-rank restriction maps, whose cohomology can change for other reasons.

## 10. Limitations and Research Agenda

The present analysis has six limits. (1) Linearity. Linear cellular sheaves approximate systems that may be nonlinear, stochastic, adaptive, and history-dependent. Possible extensions include sheaves of nonlinear spaces and persistence summaries over parameterised linearisations. (2) Model identification. The interaction complex is inferred rather than observed directly. Treating message trafic as evidence of higher-order interaction can create spurious simplices and hence spurious signatures; the operational simplex criterion in Definition 3.1 reduces this risk but cannot remove it. (3) Primitive semantics. In the benchmarks, classfilling means that a registered atomic reduction is available at a stated cost. Changing the function or cost model, or charging the primitive’s internal input collection under a diferent communication model, may change the prediction. (4) Descriptive scope. Proposition 5.1 gives an exact algebraic criterion for the linear cochain-realization family in every degree, and Section 9.2 instantiates degree 2. It does not imply that an arbitrary higher-order multi-agent task has the same semantics; domain-specific reductions and nonlinear task families remain open. (5) Operational-support control. Section 9.2 adds status-preserving stress tests, but their registered costs are not exactly matched to the class-restoring ablation in every cost dimension. Full operational support under condition (3) of Definition 3.6 therefore remains a hypothesis rather than a conclusion of the finite benchmarks. (6) Static, undirected structure. Directed, asynchronous, or changing interactions require a diferent construction.

The reduction library should cover multi-sensor fusion, cooperative planning, and general linear constraints, with task-specific meanings for the dual cohomological witnesses. Capacity sheaves should also be learned from traces and then tested on held-out tasks. On the communication side, Propositions 9.1 and 9.2 should be extended to multi-round and bitcomplexity bounds in explicit models using tools from [20]. Other open problems include domain-specific higher-degree correspondence tasks beyond the linear boundary-simplex construction and exactly cost-matched status-preserving controls. The benchmarks also permit direct comparisons with PID, coalition values, and integration measures.

Multiscale composition. Nested organisations suggest a multiscale model in which a base complex encodes inter-team interaction and fibres encode intra-team interaction. We do not develop this here for two reasons: we have no registered construction that produces a fibration-like map $p : E  B$ from interaction complexes satisfying the standard hypotheses, and the Leray–Serre spectral sequence computes ordinary cohomology of the total space, so a capacity-sheaf version would require a Leray-type spectral sequence for cellular sheaf cohomology over the base. We record only what the constant-coeficient case would say if such a construction were available. Under the standard hypotheses [22],

$$
E _ { 2 } ^ { p , q } = H ^ { p } \big ( B ; \mathcal { H } ^ { q } ( X ; \mathbb { R } ) \big ) \Longrightarrow H ^ { p + q } ( E ; \mathbb { R } ) .
$$

A nonzero $E _ { 2 } ^ { 1 , 1 }$ , recording degree-1 structure at both the team and organisation scales, need not give a nonzero class in $H ^ { 2 } ( E ; \mathbb { R } )$ . In this first-quadrant spectral sequence, the only diferential that can afect position (1, 1) is �<sub>2</sub> : $E _ { 2 } ^ { 1 , 1 } \to E _ { 2 } ^ { \dot { 3 } , 0 }$ : for $r \geq 3$ the outgoing diferential lands in negative vertical degree and the incoming one starts in negative horizontal degree. Hence $E _ { 3 } ^ { 1 , 1 } = E _ { \infty } ^ { 1 , 1 }$ , and the possible failure of cross-scale composition is concentrated in $d _ { 2 }$ Identifying registered constructions for which $d _ { 2 }$ vanishes, and a sheaf-level analogue of the statement, are open problems.

## 11. Conclusions

TSI provides a way to turn a cohomological obstruction into a testable hypothesis about multi-agent interaction. The central distinction is simple: the task model carries the structural signature, while the system provides behavioural evidence of capability. In general, a nonzero $H ^ { n } ( K ; { \mathcal { F } } )$ identifies a sheaf-relative obstruction at cochain degree � under the registered interaction structure and compatibility maps.

For the linear cochain-realization tasks studied here, Proposition 5.1 shows in every degree that a cocycle datum is realizable precisely when its cohomology class vanishes; equivalently, nonzero classes are exactly the task-relative irreducible cocycle demands of Definition 3.5. Degree 1 has the direct agent-state interpretation developed through distributed diference constraints; Section 9.2 provides a degree-2 realization on the tetrahedral boundary. The worked examples also show why the sheaf is essential: changing $\mathcal { F }$ can remove a class from a cycle or create one even when the underlying complex is contractible. The signature must therefore be defined for the full pair $( K , { \mathcal { F } } )$ , not for topology alone.

The fixed-task benchmarks illustrate how the structural claim can be tested operationally. In degree 1, the registered all-agent additive primitive solves both class-carrying tasks exactly while optimal pairwise systems remain limited under the same-1-skeleton ablation. In degree 2, the analogous four-agent primitive solves the tetrahedral-boundary tasks exactly while the same-2-skeleton system remains limited throughout the registered separation window $5 \leq \beta < 9$ . The degree-2 experiment sharpens the comparison in one respect: the ablated system retains every three-agent primitive, yet throughout that entire window its exhaustive optimum coincides with that of pairwise-only communication. Those local services form a larger catalogue of registered joint operations, but their aggregate output width is not a fully matched capacity measure because recipients difer and only invoked outputs are charged. The result therefore does not show that the quantity of joint capacity is generally irrelevant; it shows that these status-preserving local primitives do not improve the registered tasks, whereas the class-filling four-agent primitive does. Both benchmarks include trivialcohomology controls, and the degree-2 experiment adds status-preserving robustness checks. These are class-specific finite-model separations under registered one-round communication models; they do not yet establish full operational support under an exactly cost-matched status-preserving intervention.

The main contribution of TSI is a disciplined way to state such questions: define the task and its capacity sheaf first, identify the relevant cohomological obstruction, register the interaction primitive and its cost, and then test whether performance changes specifically when that obstruction is filled or restored.

## Statements and Declarations

## Funding

This research received no external funding.

## Competing interests

The authors declare no conflict of interest.

## Author contributions

Steven Motta and Alessandro Rozza contributed to conceptualization, methodology, and formal analysis. Steven Motta, Davide Mambelli, and Alessandro Rozza contributed to writing the original draft and to reviewing and editing the manuscript. All authors have read and approved the manuscript.

## Data availability

All benchmark inputs are synthetic and are generated by exhaustive enumeration of the finite domains specified in Sections 9.1 and 9.2; no external dataset is required. The benchmark inputs and result counts can be regenerated using the publicly available code identified in the Code availability statement.

## Code availability

The benchmark implementation is publicly available in the TSI simulation repository [11]. The version used for this submission is commit 0dfdc90fcf69e585a5192e7b4e9a5b91ed45 94d2.

## References

[1] Abramsky, S.; Brandenburger, A. The sheaf-theoretic structure of non-locality and contextuality. New Journal ofPhysics 2011, 13, 113036. https://doi.org/10.1088/1367 -2630/13/11/113036.

[2] Abramsky, S.; Mansfield, S.; Soares Barbosa, R. The Cohomology of Non-Locality and Contextuality. In Proceedings of the 8th International Workshop on Quantum Physics and Logic (QPL 2011); Electronic Proceedings in Theoretical Computer Science 2012, 95, 1–14. https://doi.org/10.4204/EPTCS.95.1.

[3] Barbarossa, S.; Sardellitti, S. Topological signal processing over simplicial complexes. IEEE Transactions on Signal Processing 2020, 68, 2992–3007. https://doi.org/10.110 9/TSP.2020.2981920.

[4] Battiston, F.; Cencetti, G.; Iacopini, I.; Latora, V.; Lucas, M.; Patania, A.; Young, J.-G.; Petri, G. Networks beyond pairwise interactions: Structure and dynamics. Physics Reports 2020, 874, 1–92. https://doi.org/10.1016/j.physrep.2020.05.004.

[5] Battiston, F.; Amico, E.; Barrat, A.; Bianconi, G.; Ferraz de Arruda, G.; Franceschiello, B.; Iacopini, I.; Kéfi, S.; Latora, V.; Moreno, Y.; et al. The physics of higher-order interactions in complex systems. Nature Physics 2021, 17, 1093–1098. https://doi.org/ 10.1038/s41567-021-01371-4.

[6] Benson, A.R.; Gleich, D.F.; Leskovec, J. Higher-order organization of complex networks. Science 2016, 353(6295), 163–166. https://doi.org/10.1126/science.aad9029.

[7] Bick, C.; Gross, E.; Harrington, H.A.; Schaub, M.T. What Are Higher-Order Networks? SIAM Review 2023, 65(3), 686–731. https://doi.org/10.1137/21M1414024.

[8] Bodnar, C.; Di Giovanni, F.; Chamberlain, B.P.; Liò, P.; Bronstein, M.M. Neural sheaf difusion: A topological perspective on heterophily and oversmoothing in GNNs. In Advances in Neural Information Processing Systems, 2022. https://proceedings.neurips. cc/paper\_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-C onference.html.

[9] Curry, J. Sheaves, Cosheaves and Applications. PhD thesis, University of Pennsylvania, Philadelphia, PA, 2014.

[10] Curry, J.; Ghrist, R.; Nanda, V. Discrete Morse theory for computing cellular sheaf cohomology. Foundations of Computational Mathematics 2016, 16(4), 875–897. https: //doi.org/10.1007/s10208-015-9266-8.

[11] Distributed Cognition. Topological Sheaves of Irreducibility (TSI) Simulations. GitHub, software, 2026; version 0dfdc90. https://github.com/distributedcog/tsi-simulation/tr ee/0dfdc90fcf69e585a5192e7b4e9a5b91ed4594d2.

[12] Feng, L.; Gong, H.; Zhang, S.; Liu, X.; Wang, Y.; Che, J.; Dong, A.; Grifin, C.H.; Gragnoli, C.; Wu, J.; et al. Hypernetwork modeling and topology of high-order interactions for complex systems. Proceedings of the National Academy of Sciences of the United States of America 2024, 121, e2412220121. https://doi.org/10.1073/pnas.2412220121.

[13] Ghrist, R. Elementary Applied Topology; CreateSpace, 2014.

[14] Ghrist, R.; Hiraoka, Y. Applications of sheaf cohomology and exact sequences to network coding. RIMS Kôkyûroku 2011, 1752, 31–40.

[15] Goldenbaum, M.; Boche, H.; Stańczak, S. Harnessing Interference for Analog Function Computation in Wireless Sensor Networks. IEEE Transactions on Signal Processing 2013, 61, 4893–4906. https://doi.org/10.1109/TSP.2013.2272921.

[16] Hansen, J.; Ghrist, R. Distributed optimization with sheaf homological constraints. In Proceedings of the 2019 57th Annual Allerton Conference on Communication, Control, and Computing (Allerton); IEEE, 2019; pp. 565–571. https://doi.org/10.1109/ALLERT ON.2019.8919796.

[17] Hansen, J.; Ghrist, R. Toward a spectral theory of cellular sheaves. Journal ofApplied and Computational Topology 2019, 3, 315–358. https://doi.org/10.1007/s41468-019 -00038-7.

[18] Hansen, J.; Ghrist, R. Opinion dynamics on discourse sheaves. SIAM Journal on Applied Mathematics 2021, 81(5), 2033–2060. https://doi.org/10.1137/20M1341088.

[19] Hatcher, A. Algebraic Topology; Cambridge University Press, Cambridge, 2002.

[20] Herlihy, M.; Kozlov, D.; Rajsbaum, S. Distributed Computing Through Combinatorial Topology; Elsevier, 2013. https://www.sciencedirect.com/book/9780124045781/distr ibuted-computing-through-combinatorial-topology.

[21] Kushilevitz, E.; Nisan, N. Communication Complexity; Cambridge University Press, Cambridge, 1997.

[22] McCleary, J. A User’s Guide to Spectral Sequences, 2nd ed.; Cambridge University Press, Cambridge, 2001. https://doi.org/10.1017/CBO9780511626289.

[23] Oizumi, M.; Albantakis, L.; Tononi, G. From the phenomenology to the mechanisms of consciousness: Integrated Information Theory 3.0. PLoS Computational Biology 2014, 10(5), e1003588. https://doi.org/10.1371/journal.pcbi.1003588.

[24] Papamarkou, T.; Birdal, T.; Bronstein, M.M.; Carlsson, G.E.; Curry, J.; Gao, Y.; Hajij, M.; Kwitt, R.; Liò, P.; Di Lorenzo, P.; et al. Position: Topological Deep Learning is the New Frontier for Relational Learning. In Proceedings of the 41st International Conference on Machine Learning; PMLR, 2024; Vol. 235, pp. 39529–39555. https: //proceedings.mlr.press/v235/papamarkou24a.html.

[25] Peleg, D. Distributed Computing: A Locality-Sensitive Approach; SIAM, Philadelphia, 2000. https://doi.org/10.1137/1.9780898719772.

[26] Robinson, M. Topological Signal Processing; Springer, Berlin, Heidelberg, 2014. https: //doi.org/10.1007/978-3-642-36104-3.

[27] Robinson, M. Sheaves are the canonical data structure for sensor integration. Information Fusion 2017, 36, 208–224. https://doi.org/10.1016/j.infus.2016.12.002.

[28] Robinson, M. Assignments to sheaves of pseudometric spaces. Compositionality 2020, 2, 2. https://doi.org/10.32408/compositionality-2-2.

[29] Rockafellar, R.T. The elementary vectors of a subspace of ℝ<sup>�</sup>. In Combinatorial Mathematics and its Applications; University of North Carolina Press, Chapel Hill, 1969; pp. 104–127.

[30] Schaub, M.T.; Benson, A.R.; Horn, P.; Lippner, G.; Jadbabaie, A. Random walks on simplicial complexes and the normalized Hodge 1-Laplacian. SIAM Review 2020, 62(2), 353–391. https://doi.org/10.1137/18M1201019.

[31] Shapley, L.S. A value for �-person games. In Contributions to the Theory of Games II; Princeton University Press, Princeton, 1953; pp. 307–317. https://doi.org/10.1515/97 81400881970-018.

[32] Shepard, A.D. A Cellular Description of the Derived Category of a Stratified Space; Ph.D. Thesis, Brown University, Providence, RI, USA, 1985.

[33] Tononi, G. An information integration theory of consciousness. BMC Neuroscience 2004, 5, 42. https://doi.org/10.1186/1471-2202-5-42.

[34] Torres, L.; Blevins, A.S.; Bassett, D.; Eliassi-Rad, T. The why, how, and when of representations for complex systems. SIAM Review 2021, 63(3), 435–485. https: //doi.org/10.1137/20M1355896.

[35] Williams, P.L.; Beer, R.D. Nonnegative decomposition of multivariate information. Unpublished arXiv preprint 2010, arXiv:1004.2515. https://arxiv.org/abs/1004.2515.

[36] Yuan, Y. Algebraic Topology Modeling and Game Decision Optimization for Multilayer Complex Network Dynamics. Mathematics 2026, 14, 1817. https://doi.org/10.3390/ math14111817.

[37] Zhang, Y.; Lucas, M.; Battiston, F. Higher-order interactions shape collective dynamics diferently in hypergraphs and simplicial complexes. Nature Communications 2023, 14, 1605. https://doi.org/10.1038/s41467-023-37190-9.

[38] Zia, A.; Khamis, A.; Nichols, J.; Tayab, U.B.; Hayder, Z.; Rolland, V.; Stone, E.; Petersson, L.; et al. Topological deep learning: A review of an emerging paradigm. Artificial Intelligence Review 2024, 57, 77. https://doi.org/10.1007/s10462-024-10710-9.