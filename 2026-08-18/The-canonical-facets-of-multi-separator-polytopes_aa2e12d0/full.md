# The canonical facets of multi-separator polytopes

Bjoern Andres<sup>a,b,1</sup>, Silvia Di Gregorio<sup>a,c</sup>, Jannik Irmai<sup>a</sup>,

Lucas Fabian Naumann<sup>a</sup> and Shengxian Zhao<sup>a,b</sup>

<sup>a</sup>Faculty of Computer Science, TU Dresden, Germany

<sup>b</sup>Center for Scalable Data Analytics and AI (ScaDS.AI) Dresden/Leipzig, Germany <sup>c</sup>Universit´e Sorbonne Paris Nord, CNRS, Laboratoire d’Informatique de Paris Nord, LIPN, F-93430 Villetaneuse, France

## Abstract

We initiate a polyhedral study of the graph multi-separator problem proposed by Irmai et al. (2024) as an alternative to the lifted multicut problem for application to the task of image segmentation. Starting with an integer linear program (ILP) formulation and the multi-separator polytope spanned by its feasible solutions, we characterize in terms of eficiently-decidable, graph-theoretic conditions all facets induced by inequalities of the ILP. We proceed by strengthening these inequalities and describing additional facets of some multi-separator polytopes induced by the stronger inequalities. Specifically, we obtain a totally dual integral description of the multi-separator polytope for paths in the case where separation is considered for all vertex pairs. Finally, we relate the multi-separator polytope to the boolean quadric polytope, showing that facets induced by odd-cycle inequalities do not transfer generally, and to the lifted multicut polytope, showing that either polytope is a projection of a face of the other.

## Contents

1 Introduction 2   
2 Related work 3   
3 Preliminaries 4   
4 Multi-separator polytopes 6   
4.1 Definition and outer approximation 6   
4.2 Dimension . . 7   
4.3 Facets induced by box inequalities 7   
4.4 Facets induced by connector inequalities 8   
4.5 Facets induced by separator inequalities 9   
4.6 Facets induced by path inequalities . 12   
4.7 Facets induced by intersection inequalities . 14   
5 Related polytopes 15   
5.1 Boolean quadric polytope 15   
5.2 Lifted multicut polytope . 17   
6 Conclusion 19   
A Separation algorithms 21   
B Construction of feasible vectors 22   
C Dimension and pseudoforests 24   
D Proofs 24

![](images/8bb3bae717d9a2f078c7782f98f9edaa8fbf2904e6536e6ad43a352b31b40d01.jpg)  
Figure 1: Depicted above are a graph G (solid black lines) and a set F of vertex pairs called interactions (dashed green lines) that includes the edges $\{ s _ { 1 } , w _ { 1 } \}$ $\left\{ s _ { 2 } , s _ { 3 } \right\}$ , $\{ w _ { 2 } , w _ { 3 } \}$ and the non-edges $\{ u _ { 1 } , u _ { 3 } \} , \{ u _ { 2 } , s _ { 2 } \} , \{ s _ { 1 } , s _ { 2 } \} , \{ u _ { 3 } , v _ { 3 } \}$ , {v<sub>2</sub>, w<sub>2</sub>}. Depicted in addition are a vertex subset $S = \{ s _ { 1 } , s _ { 2 } , s _ { 3 } , s _ { 4 } \}$ (solid black vertices) and the connected components of the graph $G [ V \backslash S ]$ induced by the complement of S (green areas). All interactions except $\{ u _ { 1 } , u _ { 3 } \}$ and $\{ w _ { 2 } , w _ { 3 } \}$ are separated by S in $G ,$ i.e., they belong to $F ( S )$ . In the multi-separator problem, vertex separation is considered for all interactions, whether these are edges or non-edges.

## 1 Introduction

The fundamental mathematical objects in this article are connectors and separators of a $\mathrm { \ g r a p h ^ { 2 } }$ For any two vertices $^ { a , }$ b of a graph G, we say that a and b are connected in G if and only if there exists a path in G from a to b. A connected component of G is a maximal subgraph of G in which any two vertices are connected. We call a subset $U \subseteq V$ an $\{ a , b \}$ -connector of $G ,$ and say that a and b are connected in G by $U _ { : }$ if and only if a and b are contained and connected in the subgraph $G [ U ]$ of G induced by U. Dually, we call a subset $S \subseteq V$ an $\{ a , b \}$ -separator of $G ,$ and say that a and b are separated in G by S, if and only if $V \backslash S$ is not an $\{ a , b \}$ -connector of $G ,$ i.e., either $\{ a , b \} \cap S \neq \emptyset$ or a, b are in diferent connected components of $G [ V \setminus S ]$ . This definition adopted from G¨oring (2000) is slightly uncommon in that an $\{ a , b \}$ -separator may contain a or b.

The multi-separator problem is a combinatorial optimization problem whose feasible solutions are the vertex subsets S of a graph G. In addition to the edges of $G ,$ , the model includes long-range interactions between prescribed vertex pairs, which reward or penalize whether these vertices remain connected after the removal of S.

Given costs for vertices and vertex pairs, the objective is to minimize the sum of the costs of the vertices in S and those vertex pairs that are separated by S in G:

Definition 1 (Irmai et al. (2024)). For any graph $G = ( V , E )$ , any set $F \subseteq ( \Sigma )$ of vertex pairs called interactions on $G ,$ and any $c \in \mathbb { R } ^ { V \cup F }$ , the instance of the (minimum cost) multi-separator problem has the form (MS) written below in which $F ( S )$ denotes the set of those $f \in F$ that are separated by S in $G .$

$$
\operatorname* { m i n } _ { S \subseteq V } \sum _ { v \in S } c _ { v } + \sum _ { f \in F ( S ) } c _ { f }\tag{MS}
$$

Interactions can be edges, i.e. vertex pairs in $F \cap E ,$ or non-edges, i.e. vertex pairs in $F \setminus E$ Costs assigned to edges (respectively, non-edges) penalize or reward the separation of neighboring (respectively, non-neighboring) vertices. Notice that E and F play diferent roles here: While the separation of any vertex pair is determined by E and S alone, costs are assigned to the separation of vertex pairs in $F .$ See also Figure 1.

The multi-separator problem generalizes the quadratic unconstrained binary optimization problem (Padberg, 1989), the node-weighted Steiner tree problem (Moss and Rabani, 2007) and the multi-terminal vertex separator problem (Cornaz et al., 2019b), according to Irmai et al. (2024, Thm. 3–5). Irmai et al. (2024) introduce it as an alternative to the lifted multicut problem (Andres et al., 2023) for application to the tasks of image segmentation (Keuper et al., 2015; Beier et al., 2017; Keuper, 2017; Keuper et al., 2020). While the two problems are reducible to one another in linear time (Irmai et al., 2024, Thm. 1 and 2), the multi-separator problem is less complex than the lifted multicut problem for certain cost patterns (Irmai et al., 2024, Thm. 8 and 9).

In this article, we state the multi-separator problem in the form of an integer linear program (ILP) and characterize in terms of eficiently decidable, graph-theoretic conditions the canonical facets of the associated multi-separator polytopes, i.e., those facets of multi-separator polytopes that are induced by inequalities of the ILP. Next, we strengthen inequalities of the ILP and describe additional facets of some multi-separator polytopes induced by the stronger inequalities. In particular, we obtain a totally dual integral description of multi-separator polytopes of paths in the case where separation is considered for all vertex pairs. Finally, we relate the multi-separator polytope to the boolean quadric polytope, showing that facets induced by odd-cycle inequalities do not transfer generally, and to the lifted multicut polytope, showing that either polytope is a projection of a face of the other.

The article is organized as follows. Section 2 reviews related work on vertex separators, with emphasis on polyhedral aspects. Section 3 recalls some basic facts about connectors and separators and their generalizations to pairs of vertex subsets, which are used in our polyhedral analysis of the multi-separator polytope. Section 4.1 presents an ILP formulation of the multi-separator problem, from which the multi-separator polytope is defined. In Section 4.2, we begin our study of the facial structure of multi-separator polytopes by determining their dimension. Section 4.3 is devoted to the study of box inequalities. Section 4.4 and Section 4.5 give complete characterizations of the facets of multi-separator polytopes defined by connector inequalities and separator inequalities, respectively. Sections 4.6 and 4.7 present two additional classes of facets of multi-separator polytopes, which are induced by inequalities that are not part of the ILP formulation. Section 5 discusses the relationships of the multi-separator polytope with two noticeable relatives: the boolean quadric polytope and the lifted multicut polytope. Section 6 concludes the article. Eficient separation algorithms for inequalities of the ILP formulation are given in Section A. A technical lemma that we use for constructing basis vectors of faces of multi-separator polytopes is put into Section B, together with its proof. A technical lemma that we use to establish the suficiency of conditions for an inequality to be facet-defining is discussed in Section C, together with its proof. All other proofs can be found in Section D.

## 2 Related work

Many combinatorial optimization problems related to vertex separators have been proposed in the literature, capturing diferent aspects of tasks we encounter in the real world, and many researchers have contributed to the analysis of the associated polytopes: Notably, Balas and Souza (2005) initiate a polyhedral study of the problem of finding a partition of the vertex set of a connected graph G into three disjoint subsets A, B and C, with A and B non-empty and bounded by an integer depending only on the order of G, such that C is an {A, B}-separator and its weight is minimized. After stating the problem in the form of a mixed integer linear program (MILP), they define the vertex separator polytope and investigate its dimension and facial structure. Based on the MILP and the valid inequalities described there, a branch-and-cut algorithm for this problem is developed in a companion paper (Souza and Balas, 2005). Another class of valid inequalities for the vertex separator polytope is proposed by Didi Biha and Meurs (2011).

A generalization of this problem has also been studied: Given positive constants β and κ, the generalized vertex separator problem asks for a vertex subset such that the graph obtained by deleting it has at most β parts, each of cardinality at most κ. A polyhedral investigation of this problem is conducted by Bornd¨orfer et al. (1998) from the perspective of matrix decomposition.

Given a vertex weighted graph G and a positive integer, the k-separator problem consists in finding a minimum-weight vertex subset whose removal leads to a graph where the size of each connected component is less than or equal to k. This problem is studied by Oosten et al. (2007) and Ben-Ameur et al. (2015) from a polyhedral point of view, based on diferent ILP formulations. The ILP proposed by Oosten et al. (2007) is similar to ours but does not include what we call separator inequalities (see Lemma 1 below), which are an essential part of the defining inequalities of the multi-separator problem. Related is the vertex k-cut problem that asks for a vertex k-cut of minimum weight. Here, a vertex k-cut is a vertex subset whose deletion disconnects the given graph into at least k connected components. Several ILP formulations are introduced by Cornaz et al. (2019a) and Furini et al. (2020), along with families of valid inequalities.

Another variant is the so-called multi-terminal vertex separator problem, in which a set of terminal vertices are prescribed, the task is to determine a minimum weight vertex subset whose deletion leads to a disconnected graph whose each connected component contains precisely one terminal vertex. The polyhedral structure related to an ILP formulation is studied by Cornaz et al. (2019b) who develop from this a branch-and-cut algorithm. The multi-separator problem we study here assigns costs not only to vertices but also to vertex pairs, which incorporates additional non-local information. Moreover, we do not impose any constraints on the number of connected components, their size or the size of separators; instead, these are properties of feasible solutions.

The multi-separator problem is closely related also to other combinatorial optimization problems regarding polyhedral aspects: The boolean quadric polytope studied by Padberg (1989) is a special case of the multi-separator polytope where only adjacent vertex pairs are taken into account. We discuss this relation in Section 5.1. The lifted multicut polytope studied by Horˇn´akov´a et al. (2017) and Andres et al. (2023) is associated with the lifted multicut problem, introduced by Keuper et al. (2015) as a generalization of the multicut problem (Chopra and Rao, 1993), that can be seen as an edge version of the multi-separator problem. We discuss this relation in Section 5.2. The cut inequalities that occur in an ILP formulation of the lifted multicut problem (Andres et al., 2023) resemble the separator inequalities we introduce for the multi-separator problem. Interestingly, these two classes of seemingly similar inequalities have diferent properties: The problem of deciding whether a cut inequality defines a facet of the lifted multicut polytope is NP-hard (Naumann et al., 2024). In contrast, we show in Section 4.5 that the problem of deciding whether a separator inequality defines a facet of the multi-separator polytope can be solved in polynomial time.

The Steiner cut inequalities provide an ILP formulation of the Steiner tree problem (Aneja, 1980; Chopra and Rao, 1994) and have a similar form as the separator inequalities. Despite this similarity, the conditions for separator inequalities to be facet-defining are completely diferent from those for the Steiner cut inequalities, which are always facet-defining (Chopra and Rao, 1994).

There are some works (Alvarez-Miranda et al., 2013; Wang et al., 2017; Rehfeldt et al., 2022;<sup>´</sup> Fischetti et al., 2017; Carvajal et al., 2013) on separator inequalities in the context of studying connectivity of graphs. Typically, such a separator inequality has the form $\begin{array} { r } { x _ { a } + x _ { b } \le 1 + \sum _ { s \in S } x _ { s } } \end{array}$ where a, b are two vertices in a graph G and S is an ab-separator of G with $S \cap \{ a , b \} = \emptyset$ . Note that though this class of separator inequalities looks similar to our separator inequalities introduced in Lemma 1, their corresponding facial structures difer substantially. In particular, Wang et al. (2017) studied the connected subgraph polytope of a graph and showed that the above separator inequality is facet-defining if and only if S is a minimal ab-separator. This is in contrast to the facet-defining separator inequalities for the multi-separator problem, whose structure is much more complicated (see Section 4.5).

Toward structural properties of vertex separators, Menger’s theorem (Menger, 1927) determines the minimum size of vertex separators in terms of the maximum number of disjoint paths between any pair of vertex subsets. Moreover, the celebrated planar separator theorem (Lipton and Tarjan, 1979) establishes an asymptotic bound on the size of a separator whose removal splits the given graph into smaller pieces, and has been generalized to other classes of graphs (Gilbert et al., 1984; Kawarabayashi and Reed, 2010). Furthermore, Escalante has discovered a natural partial order on the set of all minimal separators with respect to any two fixed non-adjacent vertices, with which this set becomes a lattice. See Escalante (1972); Escalante and Gallai (1974) for the original papers and Halin (1993) for a survey.

## 3 Preliminaries

In this section, we fix elementary terms and notation. Let $G = ( V , E )$ be a graph. For any $U \subseteq V$ we define the open neighborhood $N _ { G } ( U ) = \{ v \in V \backslash U \mid \exists u \in U : \{ u , v \} \in E \}$ of U in G as the set of all vertices not in U but adjacent to some vertex of U. We define the closed neighborhood $N _ { G } [ U ]$ of U in G as the union of U and $N _ { G } ( U )$ . For any distinct a, $b \in V$ , we use the shorthand notation ab for the unordered pair $\{ a , b \}$ . Since we interpret an edge or interaction as a two-element set of vertices, for any $f \in \left( \begin{array} { l } { V } \\ { 2 } \end{array} \right)$ with endvertices a and $b , \operatorname { i . e . , } f = \{ a , b \}$ , an f-connector (respectively, f-separator) of $G$ means an $\{ a , b \}$ -connector (respectively, $\{ a , b \}$ -separator) of G.

![](images/85f04149efdf97f569b42695cf2b245a2e9cc77d822f3cd7c7f8cdac01b814ce.jpg)  
Figure 2: Depicted above is a graph $G ,$ along with two vertex subsets, $A = \{ a \}$ and $B = \{ b _ { 1 } , b _ { 2 } , b _ { 3 } \}$ (in green). There are three minimal $\{ A , B \}$ -connectors of $G \colon \{ a , u _ { 1 } , b _ { 1 } \}$ , {a, u<sub>2</sub>, b<sub>2</sub>} and $\{ a , u _ { 2 } , u _ { 3 } , b _ { 3 } \}$ . All the minimal $\{ A , B \}$ -separators of G are given by {a}, $\{ u _ { 1 } , u _ { 2 } \} , \ \{ b _ { 1 } , b _ { 2 } , b _ { 3 } \} , \ \{ u _ { 2 } , b _ { 1 } \} , \ \{ u _ { 1 } , u _ { 3 } , b _ { 2 } \} , \ \{ b _ { 1 } , b _ { 2 } , u _ { 3 } \}$ and $\{ u _ { 1 } , b _ { 2 } , b _ { 3 } \}$ . Notice that $\{ u _ { 1 } , u _ { 2 } \}$ is the only minimal $\{ A , B \}$ -separator of G that is disjoint from A and B.

We adopt a generalization of connectors and separators from G¨oring (2000) who uses it for a short proof of Menger’s theorem (Menger, 1927). Let $A , B \subseteq V$ . We call a subset $U \subseteq V$ an $\{ A , B \}$ -connector of $G$ if and only if there exist $a \in A$ and $b \in B$ such that U is an $\{ a , b \}$ -connector of G. Dually, we call a subset $S \subseteq V$ an $\{ A , B \}$ -separator of G if and only if for all $a \in A$ and all $b \in B , S$ is an $\{ a , b \}$ -separator of G. Clearly, S is an $\{ A , B \}$ -separator of G if and only if $V \backslash S$ is not an $\{ A , B \}$ -connector of $G ,$ i.e., every $\{ A , B \}$ -connector of G intersects S. For any pair a, b of vertices of $G , \{ \{ a \} , \{ b \} \}$ -connectors/separators are exactly {a, b}-connectors/separators. So, by abuse of notation, we may safely identify them whenever no confusion arises.

Of particular interest are connectors and separators that are minimal. A minimal $\{ A , B \}$ connector of G is an $\{ A , B \}$ -connector U of G such that U is minimal among all $\{ A , B \}$ -connectors of G with respect to set-theoretic inclusion. Similarly, a minimal {A, B}-separator of G is an $\{ A , B \}$ -separator $S$ of G that is minimal with respect to set-theoretic inclusion. See Figure 2 for an illustration. The following proposition generalizes characterizations of minimal connectors and minimal separators that are well-known for vertex pairs.

Proposition 1. Let A, B and U be vertex subsets of a graph G.

(i) U is a minimal $\{ A , B \}$ -connector of G if and only $i f G [ U ]$ is an induced path $u _ { 0 } , u _ { 1 } , \ldots , u _ { n }$ for some $n \geq 0 ,$ , such that $U \cap A = \{ u _ { 0 } \}$ and $U \cap B = \{ u _ { n } \}$

(ii) U is a minimal $\{ A , B \}$ -separator of G if and only if every minimal $\{ A , B \}$ -connector of G intersects $U$ , and for every $u \in U$ , there exists a minimal $\{ A , B \}$ -connector C of G such that $C \cap U = \{ u \}$

Proof. See Section D.1.

Remark 1. Let A and B as in Proposition 1. Then, any singleton contained in $A \cap B$ is a minimal $\{ A , B \}$ -connector of G. Consequently, A ∩ B is a subset of any minimal $\{ A , B \}$ -separator of G.

Corollary 1. Let a and b be two vertices of a graph G and let U be a vertex subset of G.

(i) U is a minimal $\{ a , b \}$ -connector of G if and only if G[U] is an $\{ a , b \}$ -path.

(ii) U is a minimal $\{ a , b \}$ -separator of G if and only if every $\{ a , b \} { - p a t h ~ P }$ in G intersects U and for each $u \in U$ there exists an $\{ a , b \} { - } p a t h$ in G intersecting U at precisely u.

Given two vertices a and b in a graph G, a minimal $\{ a , b \} { \mathrm { - s e p a r a t o r } }$ of G is either a singleton whose unique element is one of the two endvertices a and $b ,$ or a vertex subset (possibly empty) whose complement in G induces a disconnected subgraph with a and b in diferent connected components.

## 4 Multi-separator polytopes

## 4.1 Definition and outer approximation

In this section, we reformulate the multi-separator problem in the form of an ILP and define the associated multi-separator polytope.

Definition 2. Let $G = ( V , E )$ be a graph and let $F \subseteq ( \Sigma )$ . Given a set U of vertices, let $x ^ { U }$ be the characteristic vector of our objects of interest, vertices and interactions, that are separated by $V \backslash U$ in G. Formally, $x ^ { U }$ is the vector in $\{ 0 , 1 \} ^ { V \cup ^ { \cdot } }$ defined such that the following conditions hold:

$$
\begin{array} { r l } & { \forall v \in V : \quad x _ { v } ^ { U } = 0 \Leftrightarrow v \in U , } \\ & { \forall f \in F : \quad x _ { f } ^ { U } = 0 \Leftrightarrow U \mathrm { ~ i s ~ a n ~ } f \mathrm { - c o n n e c t o r ~ o f ~ } G . } \end{array}
$$

We call $X _ { G F } : = \left\{ x ^ { U } \in \{ 0 , 1 \} ^ { V \cup F } \mid U \subseteq V \right\}$ the set of feasible vectors. Moreover, we call the convex hull $\Xi _ { G F } : =$ conv $X _ { G F }$ of $X _ { G F }$ in the afine space $\mathbb { R } ^ { V \cup F }$ the multi-separator polytope with respect to G and F.

The map $2 ^ { V }  X _ { G F } \colon S \mapsto x ^ { V \setminus S }$ is a bijection from the feasible set of the multi-separator problem (MS) to the set $X _ { G F }$ of feasible vectors. For any feasible solution $S \subseteq V$ to the multiseparator problem, its objective value is

$$
\sum _ { v \in S } c _ { v } + \sum _ { f \in F ( S ) } c _ { f } = \sum _ { v \in V } c _ { v } x _ { v } ^ { V \setminus S } + \sum _ { f \in F } c _ { f } x _ { f } ^ { V \setminus S } = \langle c , x ^ { V \setminus S } \rangle \ .
$$

Thus, $S \subseteq V$ is a solution to the multi-separator problem if and only if $x ^ { V \setminus S }$ is a solution to the problem min $\{ \langle c , x \rangle \mid x \in X _ { G F } \}$

Lemma 1. For any graph $G = ( V , E )$ and any $F \subseteq \left( { V } \right)$ , the set $X _ { G F }$ contains precisely those $x \in \mathbb { Z } ^ { V \cup F }$ that satisfy the linear system

$$
\forall f \in F \forall C \in f \mathrm { - c o n n e c t o r s } ( G ) : \qquad &  x _ { f } \leq \sum _ { v \in C } x _ { v } ,\tag{1}
$$

$$
\forall f \in F \forall S \in f \mathrm { - s e p a r a t o r s } ( G ) : \quad 1 - x _ { f } \leq \sum _ { v \in S } ( 1 - x _ { v } ) ,\tag{2}
$$

$$
\forall g \in V \cup F \colon \qquad 0 \leq x _ { g } \leq 1 ,\tag{3}
$$

where f-connectors(G) and f-separators(G) denote the set of all f-connectors and all f-separators of G, respectively. This still holds if we enforce (1) only for minimal connectors and enforce (2) only for minimal separators.

Proof. See Section D.2.

We call an inequality of the form (1) a connector inequality, one of the form (2) a separator inequality, and one of the form (3) a box inequality. Precisely these inequalities are said to be canonical. Intuitively, the connector (respectively, separator) inequalities encode the fact that if all the vertices of an f-connector (respectively, an f-separator) of G are labeled zero (respectively, one) by a feasible vector, then so is the interaction f itself.

Corollary 2. The polytope $\Xi _ { G F } ^ { \prime } = \{ x \in \mathbb { R } ^ { V \cup F }$ | (1) and (2) and (3)} is an outer polytope of the multi-separator polytope, $i . e . , \Xi _ { G F } \subseteq \Xi _ { G F } ^ { \prime }$ , with the same integer points, i.e., $X _ { G F } =$ $\mathcal { \bar { \Xi } } _ { G F } \cap \{ 0 , 1 \} ^ { V \cup \bar { F } } = \Xi _ { G F } ^ { \prime } \cap \{ 0 , 1 \} ^ { V \cup F }$

Corollary 2 asserts that min $\{ \langle c , x \rangle \mid x \in \mathbb { R } ^ { V \cup F }$ and (1) and (2) and (3)} is a linear relaxation of the multi-separator problem (MS) with the same integer feasible solutions. The separation problems for the canonical inequalities can be solved in polynomial time, and hence the linear relaxation. For the box inequalities, eficient separation is trivial. For the connector and separator inequalities, eficient separation algorithms are given in Section A.

![](images/b53f6f51732d1c82cec3b1c2a16294cf11e0f8b72a39658ec7c25aa49be3963a.jpg)  
Figure 3: Depicted above are graphs (black) together with interactions (green, dashed) where some condition of Theorem 1 or Theorem 2 is violated for a vertex or interaction. a) Since u is a neighbor of v and $\{ u , v \} \in F _ { \mathrm { \Omega } }$ , Condition (i) of Theorem 1 is violated, and thus, the lower box inequality $x _ { v } \geq 0$ is not facet-defining. b) Since u is a $\{ v , w \}$ -cut-vertex, Condition (ii) of Theorem 1 is violated, and thus, the lower box inequality $x _ { v } \ge 0$ is not facet-defining. c) Since both v and w are $\{ u , u ^ { \prime } \} \mathrm { - c u t - v e r t i c e s } .$ the condition in Theorem 2 is violated, and thus, the upper box inequality $x _ { \{ v , w \} } \leq 1$ is not facet-defining.

## 4.2 Dimension

In this section, we show that multi-separator polytopes of connected graphs with arbitrary interaction sets are full-dimensional.

Proposition 2. For any connected graph $G = ( V , E )$ and any $F \subseteq \left( { V } \right)$ , the multi-separator polytope $\Xi _ { G F }$ is full-dimensional, i.e., dim $\Xi _ { G F } = | V \cup F |$

Proof. See Section D.3.

Proposition 2 implies that each facet of $\Xi _ { G F }$ is defined by a unique linear inequality (up to scaling by a positive constant). In addition, Proposition 2 implies the following characterization of facet-defining canonical inequalities:

Proposition 3. Let $G = ( V , E )$ be a connected graph, let $F \subseteq \left( { V } \right)$ , and let $a x \leq \alpha$ be a facetdefining inequality for $\Xi _ { G F }$ with $a \in \mathbb { Z } ^ { V \cup F }$ and $\alpha \in \mathbb { Z }$ . Then $a x \leq \alpha$ is a canonical inequality up to positive scaling if and only if there is at most one $f \in F$ such that $\boldsymbol { a } _ { f } \neq 0$

Proof. See Section D.4.

We conclude this section by discussing how the multi-separator polytope on a graph can be constructed from the multi-separator polytopes on the connected components of that graph.

Proposition 4. Let $G = ( V , E )$ be any graph, let F be any set of interactions on $G ,$ and let $\{ G _ { i } \} _ { i \in I }$ be the collection of all connected components of G. For any $i \in I$ , let $F _ { i } \subseteq F$ contain those interactions whose both endvertices belong to $G _ { i }$ . Then, up to translation, $\Xi _ { G F }$ is the product of the multi-separator polytopes on its connected components, i.e.:

$$
\Xi _ { G F } \cong \mathbb { 1 } _ { F ( \mathcal { O } ) } + \prod _ { i \in I } \Xi _ { G _ { i } F _ { i } } .
$$

In particular, suppose an inequality ax $\leq \alpha$ with $a \in \mathbb { R } ^ { V \cup F }$ and $\alpha \in \mathbb { R }$ is facet-defining for $\Xi _ { G F }$ then there is a unique $i \in I$ such that $\begin{array} { r } { \lambda \big | _ { V _ { i } \cup F _ { i } } x \le \alpha - \sum _ { f \in F ( \varpi ) } a _ { f } } \end{array}$ is facet-defining for $\Xi _ { G _ { i } F _ { i } }$ Conversely, any facet-defining inequality $o f \Xi _ { G _ { i } F _ { i } } \ f o r$ some $\dot { i } \in \dot { I }$ is also $f a c e t - d e f i n i n g ~ f o r ~ \Xi _ { G F } .$

Proof. See Section D.5.

In view of Proposition 4, there is no loss of generality in considering only connected graphs for a polyhedral study of the multi-separator problem. This justifies our restriction to connected graphs in some of the previous and forthcoming statements.

## 4.3 Facets induced by box inequalities

In this section, we characterize those box inequalities (3) that define a facet of a multi-separator polytope. We begin by noting that some box inequalities may be implied (dominated) by other, stronger, valid linear inequalities, as demonstrated by the proposition below.

Proposition 5. For any graph $G = ( V , E )$ , any $F \subseteq ( \Sigma )$ and any $x \in \mathbb { R } ^ { V \cup F }$ satisfying inequalities (1), (2) and $x _ { f } \le 1$ for all $f \in F _ { \cdot }$ , the following two statements hold:

(i) $I f x _ { v } \geq 0$ for all $v \in V$ , then $x _ { f } \geq 0$ for all $f \in F$

(ii) Suppose that G does not contain isolated vertices and $E \subseteq F$ , then $0 \leq x _ { v } \leq 1$ for all $v \in V$

Proof. See Section D.6.

As a corollary of Proposition 5, the lower box inequalities associated with interactions never define facets of the multi-separator polytope:

Corollary 3. For any connected graph $G = ( V , E )$ , any $F \subseteq \left( { V } \right)$ and any $f \in F$ , the lower box inequality $x _ { f } \geq 0$ associated with f does not define a facet $o f \Xi _ { G F }$

In the following theorems, we state under which conditions the remaining box inequalities are facet-defining. Examples depicted in Figure 3 illustrate how the conditions of Theorem 1 or Theorem 2 can be violated for a vertex or interaction, and thus, the corresponding inequality does not define a facet.

Theorem 1. For any connected graph $G = ( V , E )$ , any $F \subseteq ( \Sigma )$ and any $v \in V$ , the lower box inequality $x _ { v } \geq 0$ defines a facet $o f \Xi _ { G F } i f$ and only if the following two conditions hold:

(i) For any $u \in N _ { G } ( v ) , \{ u , v \} \notin F$

(ii) For any $u \in N _ { G } ( v )$ and any w $\in V \setminus \{ u , v \}$ with $\{ u , w \} , \{ v , w \} \in F$ , u is not $a \ \{ v , w \} - c u t -$ $v e r t e x ^ { 3 } o f G$

Proof. See Section D.7.

Theorem 2. For any connected graph $G = ( V , E )$ , any $F \subseteq ( \Sigma )$ and any $f \in F ,$ the upper box inequality $x _ { f } \le 1$ defines a facet of $\Xi _ { G F }$ if and only if there is no interaction $f ^ { \prime } \in F \setminus \{ f \}$ such that f is a pair of f<sup>′</sup>-cut-vertices of G.

Proof. See Section D.8.

Theorem 3. For any connected graph $G = ( V , E )$ , any $F \subseteq \left( { V } \right)$ and any $v \in V$ , the upper box inequality $x _ { v } \le 1$ defines a facet $o f \Xi _ { G F }$ if and only if there is no interaction $f \in F$ such that v is an f-cut-vertex of G.

Proof. See Section D.9.

## 4.4 Facets induced by connector inequalities

In this section, we establish the exact condition under which a connector inequality (1) defines a facet of a multi-separator polytope. For this purpose, we introduce the notion of a bypass vertex of a connector. As the connector inequality associated with a non-minimal connector cannot be facet-defining (by Lemma 1), we introduce bypass vertices only for minimal connectors, avoiding complications.

Definition 3. Let $G = ( V , E )$ be a connected graph, let $F \subseteq \left( { V } \right)$ , let $f \in F$ and let $C \subseteq V$ be a minimal f-connector of G. For any $v \in V$ and any $c \in C .$ , we call v a c-bypass of C if and only if $v \not \in C$ and $( C \setminus \{ c \} ) \cup \{ v \}$ is an $f .$ -connector of $G _ { \ast }$ . For any $v \in V$ , we call v a bypass of C if and only if v is a c-bypass of C for some $c \in C$

Examples of bypasses are depicted in Figure $4 { \mathrm { ~ c } } ) - { \mathrm { e } } )$ . Notice that the neighborhood of any bypass of $C$ contains at least two vertices of C. With the definition of a bypass, we are now ready to state the conditions that characterize facet-defining connector inequalities:

![](images/f985e50de10ebe091d8d957db5d9d15e3b13415754bfa5f50501f19d54979c66.jpg)  
Figure 4: Depicted above are graphs $G = ( V , E )$ (black, solid) and interactions F (green, dashed) with $\{ p , q \} \in F$ . In each of these graphs, there exists a $\{ p , q \}$ -connector C that violates one condition of Theorem 4. Thus, the connector inequality with respect to $\{ p , q \}$ and C is not facet-defining for $\Xi _ { G F }$ Written below each of the depicted graphs is an inequality satisfied by all feasible vectors for this graph. These inequalities are examples of the inequalities constructed in the proof of Theorem 4 to establish necessity of the respective conditions. a) Condition (i) is violated for $C = \{ p , c _ { 1 } , q \}$ as $\{ p , q \} \subset C$ is a $\{ p , q \}$ }-connector. b) Condition (ii) is violated for $C = \{ p , c _ { 1 } , q \}$ as $\{ p , c _ { 1 } \} \in F . { \mathrm { ~ c } } )$ Condition (iii) is violated for $C = \{ p , c _ { 1 } , q \}$ as u is a $c _ { \mathrm { 1 } } { \mathrm { - } } \mathrm { b y }$ pass and $\{ u , c _ { 1 } \} \in F . \ \mathrm { d } )$ Condition (iv) is violated for $C = \{ p , c _ { 1 } , q \}$ as $\{ u , w \} , \{ u , c _ { 1 } \} \in F$ and w is both a c -bypass and a $\{ \{ u \} , C \}$ -cut-vertex. e) Condition (v) is violated for $C = \{ p , c _ { 1 } , c _ { 2 } , c _ { 3 } , c _ { 4 } , q \}$ as $\{ u , c _ { 2 } \} , \{ u , c _ { 3 } \} \in F$ and the set $\{ v , v ^ { \prime } \}$ of c - and c -bypasses is a $\{ \{ u \} , C \}$ separator.

Theorem 4. For any connected graph $G = ( V , E )$ , any $F \subseteq \left( { V } \right)$ , any $f \in F$ and any f-connector C of G, the associated connector inequality

$$
x _ { f } \leq \sum _ { v \in C } x _ { v }\tag{4}
$$

defines a facet of the multi-separator polytope $\Xi _ { G F }$ if and only if all the following conditions hold:

(i) C is a minimal f-connector of G.

(ii) There is no interaction $f ^ { \prime } \in F \setminus \{ f \}$ such that $f ^ { \prime } \subseteq C$

(iii) There is no interaction uw $) \in F$ such that u is a w-bypass of C.

(iv) There are no two distinct interactions uw, $u w ^ { \prime } \in F$ such that w is both a w<sup>′</sup>-bypass of C and a $\{ \{ u \} , C \}$ -cut-vertex<sup>4</sup> $o f G$

(v) There are no two distinct interactions $\{ u , w \} , \{ u , w ^ { \prime } \} \in F$ with u $\notin N _ { G } [ C ]$ and w, $w ^ { \prime } \in C$ such that $\{ v \in N _ { G } ( C ) \mid v$ is both a w-bypass and a $w ^ { \prime } .$ -bypass of C} is a $\{ \{ u \} , C \}$ -separator of G.

Proof. See Section D.10.

Non-trivial examples of connectors whose associated inequalities fail to define facets of $\Xi _ { G F }$ are depicted in Figure 4. Moreover, Condition (ii) of Theorem 4 implies the following corollary.

Corollary 4. Consider the special case in which every edge of G is an interaction in F, i.e. $E \subseteq F$ Now, for any interaction $f \in F$ and any minimal f-connector C of G containing more than two vertices, the corresponding connector inequality is not facet-defining.

## 4.5 Facets induced by separator inequalities

In this section, we characterize by graph properties those separator inequalities (2) that define a facet of a multi-separator polytope. In order to capture important features of the set of those feasible vectors that satisfy any given separator inequality at equality, we first introduce some additional concepts and notation. Since the separator inequality associated with a non-minimal separator cannot be facet-defining (by Lemma 1), we present the following definitions only for minimal separators, once again avoiding complications.

![](images/59a50cee0709cae0077386b9e52eb1f51661510861d7ca3a004883a89d81b9f1.jpg)  
Figure 5: Depicted above are a graph G (solid black), a designated interaction $f = \{ p , q \}$ (dashed red), and other interactions vs<sub>1</sub>, vp, vq, us<sub>2</sub>, uw, s<sub>1</sub>s<sub>2</sub>, qw (dashed green). The set $S = \{ s _ { 1 } , s _ { 2 } \}$ is a minimal f-separator of G. All interactions except qw are in $F ( S )$ . The subgraph $G ( S , s _ { 1 } )$ is induced by $\{ p , u , s _ { 1 } , v , w , q \}$ . The subgraph $G ( S , s _ { 2 } )$ is induced by $\{ p , u , s _ { 2 } , w , q \}$ . Moreover, $S ( f ) = S ( u w ) = S , S ( v s _ { 1 } ) = S ( v p ) = S ( v q ) =$ $\{ s _ { 1 } \} , S ( u s _ { 2 } ) = \{ s _ { 2 } \}$ , and $S ( s _ { 1 } s _ { 2 } ) = \emptyset$

Definition 4. Let $G = ( V , E )$ be a connected graph, let $F \subseteq ( \Sigma )$ , let $f \in F$ and let $S \subseteq V$ be a minimal f-separator of $G .$

(i) For any vertex $s \in S$ , let $G ( S , s )$ denote the connected component of the subgraph of $G$ induced by $V \setminus \left( S \setminus \{ s \} \right)$ that contains s.

(ii) For any interaction $f ^ { \prime } \in F ( S )$ , let $S ( f ^ { \prime } )$ denote the set of those vertices s in S for which $f ^ { \prime }$ is connected in $G ( S , s )$

(iii) Call a vertex $v \in V$ degenerate (with respect to $G , F , f$ and S) if $v \in S$ or each $s \in S$ has at least one of the following three properties: 1. $v \not \in G ( S , s ) ; 2 . \ v \in G ( S , s )$ and $\{ v , s \} \in F$ $3 . \ v \in G ( S , s )$ and v is an f-cut-vertex of $G ( S , s )$ . Otherwise, call v non-degenerate (with respect to $G , F , f$ and S).

(iv) We define $\mathcal { H } ( f , S )$ as the graph $( V \setminus S , F ^ { \prime } )$ , where $F ^ { \prime }$ is the set of those interactions $f ^ { \prime } \in { \dot { F } } ( S ) \setminus \{ f \}$ such that $f ^ { \prime } \cap S = \emptyset$ and for each $s \in S ( f ^ { \prime } )$ , the interaction $f ^ { \prime }$ contains at least one f-cut-vertex of $G ( S , s )$

Remark 2. For a fixed minimal f-separator S of G as in Definition 4, we can list all the elements of $G ( S , s ) , S ( f ^ { \prime } )$ and $\mathcal { H } ( f , S )$ in polynomial time. Indeed, such a problem boils down to checking whether two vertices are connected in some induced subgraph of $G$ . As a consequence, deciding whether a vertex is degenerate can also be done in polynomial time.

Examples of $G ( S , s )$ and $S ( f ^ { \prime } )$ are shown in Figure 5. Examples illustrating degenerate vertices and $\mathcal { H } ( f , S )$ are given in Figure 7. Note that an interaction that contains an f-cut-vertex of $G ( S , s )$ for each $s \in S$ is not necessarily a minimal f-separator of G.

Remark 3. The main motivation for introducing $G ( S , s )$ and $S ( f ^ { \prime } )$ comes from the following fact: For any vertex subset U of a graph G, the feasible vector $x ^ { U }$ is in the face Σ defined by the separator inequality associated with an interaction $f$ and an f-separator S of G if and only if either $U \cap S$ is empty, or $U \cap S$ is a singleton and U is an f-connector of G. For an informal discussion, we call a vertex whose variable assumes the value 0 (respectively 1) unblocked (respectively blocked): For every feasible solution in $\Sigma ,$ , the f-separator S is either blocked, or unblocked at exactly one vertex such that $f$ is connected by an unblocked path. Thus, in order to stay in $\Sigma ,$ when a vertex s in S is unblocked, the vertices of $G ( S , s )$ cannot be blocked or unblocked freely. Instead, we need to maintain the existence of an unblocked path connecting $f .$ On the one hand, this fact prevents us from constructing a basis of the associated linear space of Σ in a straightforward way, like in the proof of Proposition 2. On the other hand, interactions connected in components of $G [ V \setminus ( S \setminus \{ s \} ) ]$ other than $G ( S , s )$ are necessarily not in $F ( S )$ and can thus be handled analogously to the proof of Proposition 2, as will be shown in detail below.

Remark 4. As the f-separator S is assumed to be minimal in Definition 4, every $s \in S$ is such that f is connected in $G ( S , s )$ , by (ii) of Corollary 1.

Remark 5. With the notation of Definition 4, every f-cut-vertex of G is degenerate.

Theorem 5. Let $G = ( V , E )$ be a connected graph, let $F \subseteq \left( { V } \right)$ , let $f \in F$ and let $S$ be an f-separator of G. The associated separator inequality

$$
1 - x _ { f } \leq \sum _ { s \in S } ( 1 - x _ { s } )\tag{5}
$$

is facet-defining for the multi-separator polytope $\Xi _ { G F }$ if and only if all the following conditions are satisfied:

(i) S is a minimal f-separator of G.

(ii) For every interaction $f ^ { \prime } \in F ( S ) \setminus \{ f \}$ , there exists an $s \in S ( f ^ { \prime } )$ such that $f ^ { \prime }$ is not a pair of f-cut-vertices of $G ( S , s )$

(iii) If v is a non-isolated vertex of $\mathcal { H } ( f , S )$ such that S is an $\{ f , \{ v \} \}$ -separator of $G ,$ , then v is a non-degenerate $l e a f ^ { 5 }$ of $\mathcal { H } ( f , S )$

$( \operatorname { i v } ) \ \mathcal { H } ( f , S )$ is a forest in which every connected component contains at most one degenerate vertex.

Proof. See Section D.11.

The following result says that one can eficiently certify in polynomial time that a given separator inequality is facet-defining. In contrast, for a cut inequality of the closely related lifted multicut problem, such a decision problem is NP-hard (Naumann et al., 2024).

Lemma 2. Conditions ${ \bf ( i ) } \mathrm { ~ - ~ } ( \mathrm { i v } )$ in Theorem 5 can be checked in polynomial time. In particular, we can eficiently decide whether a separator inequality defines a facet.

Proof. See Section D.12.

As f-cut-vertices are a specific type of minimal f-separators, we can obtain the following result from the proof of Theorem 5.

Corollary 5. Let $G = ( V , E )$ be a connected graph, let $F \subseteq ( \Sigma )$ , let $f \in F ,$ , and let $g \in V \cup F$ be an f-cut-vertex or a pair of f-cut-vertices of G. Then the inequality $x _ { g } \le x _ { f }$ is valid.

Moreover, it is facet-defining $f o r \Xi _ { G F }$ if and only if the following two conditions are satisfied:

(i) There is no interaction $f ^ { \prime } \in F \setminus \{ f , g \}$ such that $f ^ { \prime }$ is a pair of f-cut-vertices of $G ,$ , and $g$ is an f<sup>′</sup>-cut-vertex or a pair of f<sup>′</sup>-cut-vertices of G.

(ii) There do not exist two adjacent interactions $f ^ { \prime } , f ^ { \prime \prime } \in F$ such that all of the following conditions are satisfied:

(a) The vertex in $f ^ { \prime } \cap f ^ { \prime \prime }$ is not an f-cut-vertex of $G ;$

(b) $f ^ { \prime } \triangle f ^ { \prime \prime }$ is a pair of f-cut-vertices of $G ;$

(c) g is a cut-vertex or a pair of cut-vertices of both $f ^ { \prime }$ and $f ^ { \prime \prime } \ o f G$

See Figure 6 for examples illustrating the conditions of Corollary 5.

For a better understanding of the conditions of Theorem $5 ,$ and complementary to the proof, we make some remarks and give several examples (see Figure 7) illustrating how they can be violated.

Remark 6. In Theorem 5, Conditions (iii) and (iv) are not independent: If there is an edge $\{ u , v \}$ of $\mathcal { H } ( f , S )$ whose two endvertices are degenerate and v is separated from $f$ by $S$ in $G ,$ , then both conditions are violated. See Figure $\mathrm { 7 \ a ) }$ for an example.

Remark 7. In the context of Theorem 5, assume Condition (i). If Condition (ii) does not hold for some $f ^ { \prime } \in F ( S ) \setminus \{ f \}$ , then $S ( f ^ { \prime } )$ can only be

(A) the empty set, in which case $S$ contains two disjoint $f ^ { \prime } .$ -separators of $G ;$

(B) a singleton {s} with $s \in f ^ { \prime } \cap S ,$ , in which case the endvertex of $f ^ { \prime }$ that is not in S must be an f-cut-vertex of $G ( S , s ) ;$

(C) the whole separator $S ,$ , in which case both endvertices of $f ^ { \prime }$ are f-cut-vertices of $G .$

![](images/e005786bfd37b25557fed1964ae32e71c595ca8f2638a6ee1c473793fef7fc52.jpg)  
Figure 6: Examples illustrating the conditions of Corollary 5. Each subfigure depicts a graph $G$ (solid black) together with the interactions (dashed) defining a valid inequality $x _ { g } \le x _ { f }$ . Neither inequality is facet-defining: In $\mathrm { a ) }$ , condition (i) is violated by $f ^ { \prime }$ (highlighted in red), while in b), condition (ii) is violated by $f ^ { \prime }$ and $f ^ { \prime \prime }$ (highlighted in red). For each subfigure, we additionally list an equality satisfied by all feasible vectors in the face induced by the corresponding inequality.

E.g., in Figure 5, the interactions $s _ { 1 } s _ { 2 }$ , us<sub>2</sub> and uw violate Condition (ii), since they satisfy the cases (A), (B) and (C) above, respectively. To see why only these three cases are possible, suppose $S ( f ^ { \prime } )$ is neither a singleton consisting of an endvertex of $f$ nor the empty set. In other words, $f ^ { \prime }$ is disjoint from S and it is a pair of f-cut-vertices of $G ( S , s )$ for any vertex s in the non-empty set $S ( f ^ { \prime } )$ . Because of Remark 4, this implies that $f ^ { \prime }$ is connected in $G ( S , s )$ for every $s \in S$ . Hence, $S ( f ^ { \prime } ) = S$

We also give an alternative characterization of facet-defining separator inequalities for an interesting special case.

Lemma 3. Under conditions (i), (ii) and (iii) of Theorem 5, if there exist at least three internally vertex-disjoint f-paths in G such that each of them intersects the separator $S$ in one vertex, then condition (iv) is satisfied if and only if there is no interaction $\{ p , v \} \in F ( S )$ with $p \in f$ such that the following two properties hold: 1. v is a vertex not separated from f by S in G and 2. $\{ v , s \} \in F$ for every $s \in S$

Proof. See Section D.13.

Remark 8. As an application of the facet-defining conditions we obtained, we can construct valid inequalities stronger than the ordinary separator inequalities (2). Here, as a simple illustration, we observe that Lemma 3 suggests to add the term $\begin{array} { r } { \sum _ { s \in S } ( 1 - \dot { x } _ { v s } ) - ( 1 - x _ { v p } ) } \end{array}$ to the left hand side of the separator inequality $\begin{array} { r } { ( 1 - x _ { f } ) \le \sum _ { s \in S } ( 1 - x _ { s } ) } \end{array}$ , where v is an arbitrary vertex separated from $q \in f = \{ p , q \}$ by S in G and $\{ v , p \} , \{ v , { \bar { s } } \} \in F$ for all $s \in S$ . This procedure results in the following inequality

$$
( 1 - x _ { f } ) \leq \sum _ { s \in S } ( x _ { v s } - x _ { s } ) + ( 1 - x _ { v p } ) ,\tag{6}
$$

which can indeed be verified to be valid for $\Xi _ { G F }$ . Of course, since $\begin{array} { r } { \sum _ { s \in S } ( 1 - x _ { v s } ) - ( 1 - x _ { v p } ) \geq 0 } \end{array}$ (6) is stronger than the separator inequality from which it is derived.

## 4.6 Facets induced by path inequalities

In this section, we introduce a class of valid inequalities, called path inequalities, which strengthen the connector inequalities. We then characterize when these inequalities define facets of the multi-separator polytope. Two illustrative examples are shown in Figure 8.

Definition 5. Let $G = ( V , E )$ be a connected graph, let $F \subseteq ( _ { 2 } ^ { V } )$ , let $f \in F ,$ and let $P = ( V _ { P } , E _ { P } )$ be an f-path in the graph (V, F). We call the inequality written below the path inequality associated with f and P.

$$
x _ { f } \ \le \ \sum _ { e \in E _ { P } } x _ { e } \ - \ \sum _ { v \in V _ { P } \backslash f } x _ { v }\tag{7}
$$

![](images/32b89ebbb7049b825d9ab93e45d4619e16f3a42b4eba2a6d4513d2a73187b07f.jpg)  
x<sub>f</sub> + x<sub>f2</sub> = x<sub>f1</sub> + x<sub>f3</sub>  
f) x<sub>f1</sub> + x<sub>f3</sub> = x<sub>f2</sub> + x<sub>f4</sub>  
g) x<sub>f1</sub> + x<sub>f3</sub> = x<sub>f2</sub> + x<sub>f4</sub>  
h) x<sub>f1</sub> + x<sub>f3</sub> = x<sub>f2</sub> + x<sub>f4</sub>

Figure 7: Each subfigure depicts a graph G (solid black), interactions (dashed edges), a designated interaction $f = p q$ (dashed, red), and a minimal f-separator S (filled black vertices). In each case, the separator inequality associated with $f$ and $S$ does not define a facet because either Condition (iii) or (iv) is violated, though Conditions (i) and (ii) are satisfied. This fact can also be confirmed by the additional equalities listed below the subfigures. a) All the vertices are degenerate and $f _ { 1 }$ is the only edge of $\mathcal { H } ( f , \{ s \} )$ Condition (iii) is violated by the degenerate vertex $v ,$ though it is a leaf of $\mathcal { H } ( f , \{ s \} )$ . As both endvertices of $f _ { 1 }$ are degenerate, Condition $( \mathrm { i v } )$ also does not hold. b) The only degenerate vertices are $s _ { 1 } , s _ { 2 } , p , q$ and the edge set of $\mathcal { H } ( f , S )$ consists of $f _ { 1 }$ and $f _ { 2 }$ . The unique non-isolated vertex v of $\mathcal { H } ( f , S )$ separated from $f$ by $S = \{ s _ { 1 } , s _ { 2 } \}$ in G is $v ,$ it is non-degenerate but not a leaf of $\mathcal { H } ( f , S )$ , so Condition (iii) of Theorem 5 is violated by v. Thus the corresponding separator inequality is not facet-defining. However, $\mathcal { H } ( f , S )$ satisfies Condition (iv). c) The only degenerate vertices are v, $s _ { 1 } , s _ { 2 } , p , q$ and the only edge of $\mathcal { H } ( f , \{ s _ { 1 } , s _ { 2 } \} )$ is $f _ { 3 }$ It is direct to see that Condition (iii) is satisfied as no vertices are separated from f by $S = \{ s _ { 1 } , s _ { 2 } \}$ in G. The odd path $\mathcal { H } ( f , \{ s _ { 1 } , s _ { 2 } \} )$ has two degenerate endvertices and hence Condition (iv) is false. d) All the vertices are degenerate and $f _ { 2 }$ is the only edge of $\mathcal { H } ( f , \{ s _ { 1 } , s _ { 2 } \} )$ ). As no vertices of $\mathcal { H } ( f , \{ s _ { 1 } , s _ { 2 } \} )$ are separated from $f$ by $S$ in $G$ , Condition (iii) is vacuously true. However, Condition (iv) is not satisfied as the two leaves, i.e. the endvertices of $f _ { 2 }$ , of the odd path $\mathcal { H } ( f , \{ s _ { 1 } , s _ { 2 } \} )$ are degenerate. e) In this example, $s _ { 1 } , s _ { 2 } , p , q$ are all the degenerate vertices and the edge set of $\mathcal { H } ( f , \{ s _ { 1 } , s _ { 2 } \} )$ is $\{ f _ { 1 } , f _ { 2 } , f _ { 3 } \}$ . Condition (iii) is vacuously true due to the same reason as in the previous case. $\mathcal { H } ( f , \{ s _ { 1 } , s _ { 2 } \} )$ is an odd path whose two endvertices are degenerate and hence violates Condition (iv). f) The only degenerate vertices are $s _ { 1 } , s _ { 2 } , p , q$ and the edge set of $\mathcal { H } ( f , \{ s _ { 1 } , s _ { 2 } \} )$ consists of $f _ { 1 } , f _ { 2 } , f _ { 3 }$ and $f _ { 4 }$ . As the previous case, Condition (iii) is vacuously true. On the other hand, $\mathcal { H } ( f , \{ s _ { 1 } , s _ { 2 } \} )$ is an even cycle and hence Condition (iv) is violated. g) The degenerate vertices in this case include v and $v ^ { \prime }$ besides f-cut-vertices $p , q$ and the vertices $s _ { 1 } , s _ { 2 }$ in the separator $S .$ The edge set of $\mathcal { H } ( f , S )$ consists of $f _ { 1 }$ and $f _ { 2 }$ . As no vertices are separated from $f$ by $S$ in $G ,$ Condition (iii) is trivially satisfied. Condition (iv) is violated because $\mathcal { H } ( f , S )$ is an even path with two degenerate endvertices $v , v ^ { \prime } .$ . h) The only degenerate vertices are f-cut-vertices $p ,$ q and $v ,$ besides those in $S _ { ☉ }$ . The edges of $\mathcal { H } ( f , S )$ are given by $f _ { 1 } , f _ { 2 } , f _ { 3 }$ and $f _ { 4 }$ . Again, as no vertices are separated from $f$ by $S$ in $G ,$ Condition (iii) is trivially satisfied. $\mathcal { H } ( f , S )$ is an even path with two endvertices p and v being f-cut-vertices of $G ,$ so Condition (iv) is violated.

![](images/f22cbb1883e1d224c3a4d20b35b05f2138f84a4d89151bd2e343fc4d16bea4db.jpg)  
Figure 8: Each subfigure shows a graph G (solid black) with interactions (dashed green) and a path P on vertices $\{ s , v , t \}$ . In a), the st-connector inequality $x _ { s t } \leq x _ { s } + x _ { v } + x _ { t }$ is not facet-defining by Theorem 4 (ii), whereas the path inequality $x _ { s t } \leq x _ { s v } + x _ { v t } - x _ { v }$ is facet-defining by Theorem 6. In b), the connector inequality $x _ { s t } \leq x _ { s } + x _ { v } + x _ { t }$ is not facet-defining by Theorem $4 \ ( \mathrm { i } )$ , whereas the path inequality $x _ { s t } \leq x _ { s v } - x _ { v } + x _ { v t }$ is facet-defining by Theorem 6.

The inequality (7) can be interpreted as follows: Separating the pair f requires separating at least one edge along the path P, while accounting for overlaps at internal vertices. In this sense, path inequalities refine connector inequalities by exploiting the structure of a specific path between the endpoints of f.

Lemma 4. Let $G = ( V , E )$ be a connected graph and let $F \subseteq \left( { V } \right)$ . For any $f \in F$ and any f-path P in $( V , F )$ , the corresponding path inequality (7) is valid $f o r \Xi _ { G F }$

Proof. See Section D.14.

The class of path inequalities is large and structurally rich. In the remainder of this section, we restrict attention to path inequalities where $P$ is a path in G. For this subclass, we obtain a complete characterization of when such inequalities are facet-defining.

Theorem 6. Let $G = ( V , E )$ be a connected graph, let $F \subseteq \left( { V } \right)$ , let $f \in F$ , and let $P = ( V _ { P } , E _ { P } )$ be an f-path in G. Suppose that $E _ { P } \subseteq F \setminus \{ f \}$ . Then the path inequality (7) defines a facet of $\Xi _ { G F } \ i f$ and only if P is chordless in the graph $( V , F \setminus \{ f \} )$ .

Proof. See Section D.15.

As a consequence of Theorem 6, we obtain the following characterization regarding a class of inequalities defined on cycles in G:

Corollary 6. Let $G = ( V , E )$ be a connected graph, let $F \subseteq ( \Sigma )$ , let $C = ( V _ { C } , E _ { C } )$ be a cycle in G with $E _ { C } \subseteq F$ , and let e be an edge of C. Then the inequality written below is facet-defining for $\Xi _ { G F }$ if and only $i f C$ is chordless in $( V , F )$

$$
\sum _ { v \in V _ { C } \backslash e } x _ { v } + x _ { e } - \sum _ { e ^ { \prime } \in E _ { C } \backslash \{ e \} } x _ { e ^ { \prime } } \leq 0
$$

## 4.7 Facets induced by intersection inequalities

In this section, we introduce a class of inequalities, called intersection inequalities, arising from interactions between vertex pairs, and characterize precisely when such inequalities define facets of the multi-separator polytope.

Definition 6. Let $G = ( V , E )$ be a connected graph and let $F \subseteq \left( { V } \right)$ . Suppose that for some pairwise distinct vertices $v _ { 1 } , v _ { 2 } , w _ { 1 } , w _ { 2 } \in V$ , the four pairs

$$
\{ v _ { 1 } , w _ { 1 } \} , \ \{ v _ { 2 } , w _ { 2 } \} , \ \{ v _ { 1 } , w _ { 2 } \} , \ \{ v _ { 2 } , w _ { 1 } \}
$$

are all contained in F. If, in addition, $\{ v _ { 1 } , v _ { 2 } \} , \{ w _ { 1 } , w _ { 2 } \} \in E$ , then the inequality

$$
x _ { v _ { 1 } w _ { 2 } } + x _ { v _ { 2 } w _ { 1 } } \ \leq \ x _ { v _ { 1 } w _ { 1 } } + x _ { v _ { 2 } w _ { 2 } }\tag{8}
$$

is called an intersection inequality.

![](images/35e20f35a5135e05de7352df45f6447c7a3eaac1903c426818e5d19697f97e04.jpg)  
Figure 9: An illustration of the intersection inequality (8) with G depicted in solid black: the crossing pairs $( v _ { 1 } , w _ { 1 } )$ and (v<sub>2</sub>, w<sub>2</sub>) (in red) versus the parallel pairs $( v _ { 1 } , w _ { 2 } )$ and $( v _ { 2 } , w _ { 1 } )$ (in green).

The inequality (8) compares the two parallel pairs $\{ v _ { 1 } , w _ { 2 } \}$ and $\{ v _ { 2 } , w _ { 1 } \}$ with the two crossing pairs $\{ v _ { 1 } , w _ { 1 } \}$ and $\{ v _ { 2 } , w _ { 2 } \}$ . Intuitively, it asserts that separating the parallel pairs is at least as restrictive as separating the crossing pairs, provided the endpoints are linked by the edges $\{ v _ { 1 } , v _ { 2 } \}$ and $\{ w _ { 1 } , w _ { 2 } \}$ . For an illustration, see Figure 9.

Lemma 5. Under the assumptions of Definition 6, the intersection inequality (8) is valid for the multi-separator polytope $\Xi _ { G F }$ if and only if $\{ v _ { 1 } , w _ { 1 } \}$ is a $\{ v _ { 2 } , w _ { 2 } \}$ -separator of G and $\{ v _ { 2 } , w _ { 2 } \}$ is a $\{ v _ { 1 } , w _ { 1 } \}$ -separator of G.

Proof. See Section D.16.

The next result shows that no additional conditions are required for these inequalities to be facet-defining: validity already captures the full strength of the construction.

Theorem 7. Under the assumptions of Definition 6, the intersection inequality (8) is facetdefining for the multi-separator polytope $\Xi _ { G F } \ i f$ and only $i f \left\{ v _ { 1 } , w _ { 1 } \right\}$ is a $\{ v _ { 2 } , w _ { 2 } \}$ -separator of G and $\{ v _ { 2 } , w _ { 2 } \}$ is a $\{ v _ { 1 } , w _ { 1 } \}$ -separator of G.

Proof. See Section D.17.

As we will see in Section 5.2, intersection inequalities, together with those discussed before, sufice to completely describe the multi-separator polytope for path graphs with complete pairwise vertex interactions.

## 5 Related polytopes

In this section, we discuss relations between the multi-separator polytope and two closely related polytopes, namely the boolean quadric polytope and the lifted multicut polytope.

## 5.1 Boolean quadric polytope

The boolean quadric polytope $\mathrm { B Q P } _ { G }$ associated with a graph $G = ( V , E )$ as defined by Padberg (1989) is isomorphic to the multi-separator polytope associated with the graph G and interactions $F = E$ , more specifically, it is the image of $\Xi _ { G E }$ under the automorphism of $\mathbb { R } ^ { V \cup E }$ given by $x \mapsto \mathbb { 1 } _ { V \cup E } - x$ . In this section, we consider any interaction set $F$ on $G$ and show how to transfer valid inequalities from the boolean quadric polytope associated with the graph $( V , F )$ to the multi-separator polytope associated with G and F. Afterward, we give some examples illustrating that the conditions under which an inequality that is valid for $\mathrm { B Q P } _ { G }$ becomes facet-defining for $\mathrm { B Q P } _ { G }$ are generally quite diferent from the conditions under which the corresponding derived inequality that is valid for $\Xi _ { G F }$ becomes facet-defining for $\Xi _ { G F }$

Lemma 6. Let $G = ( V , E )$ be a connected graph, let F be an interaction set on G, and let ax ≤ α be an inequality in $\mathbb { R } ^ { V \cup F }$ with $a \in \mathbb { R } ^ { V \cup F }$ and $\alpha \in \mathbb { R }$

(i) If ax $\leq \alpha$ is a valid inequality for the boolean quadric polytope $\mathrm { B Q P } _ { ( V , F ) }$ such that for every $f \in F$ we have $f \in E$ or $a _ { f } \geq 0$ , then the flipped inequality $a ( 1 - x ) \leq \alpha$ is valid for the multi-separator polytope $\Xi _ { G F }$

(ii) $I f a ( 1 - x ) \leq \alpha$ is valid $f o r \Xi _ { G F }$ such that for every $f \in F$ we have $f \in E$ or $a _ { f } \leq 0$ , then ax $\leq \alpha$ is valid $f o r \mathrm { B Q P } _ { ( V , F ) }$

(iii) Under the assumption that $\{ f \in F \mid a _ { f } \neq 0 \} \subseteq E \subseteq F , i f a ( 1 - x ) \leq \alpha$ is facet-defining $f o r \Xi _ { G F }$ , then so is $a | _ { V \cup E ^ { x } } \leq \alpha$ for $\mathrm { B Q P } _ { G }$

Proof. See Section D.18.

The last statement (iii) of Lemma 6 tells us that, under the given assumption, if a condition is necessary for $a x \leq \alpha \mathrm { t o }$ be facet-defining for $\mathrm { B Q P } _ { G } ,$ then so it is for $a ( 1 - x ) \leq \alpha$ to be facet-defining for $\Xi _ { G F }$ . We remark that the validity of an inequality ax $\leq \alpha$ for the boolean quadric polytope $\mathrm { B Q P } _ { ( V , F ) }$ does not necessarily imply that $a ( 1 - x ) \leq \alpha$ is valid for the multi-separator polytope $\Xi _ { G F } ;$ even if this is the case, the property of being facet-defining usually does not transfer from $\mathrm { B Q P } _ { ( V , F ) }$ to $\Xi _ { G F }$ . Illustrative examples can be given by the odd-cycle inequalities described by Padberg (1989):

Example 1. Let $G = ( V , E )$ be a connected graph, let F be an interaction set $F$ on $G ,$ let $C = ( V _ { C } , E _ { C } )$ be a cycle in G with $E _ { C } \subseteq F$ and let $M \subseteq E _ { C }$ be an edge subset of odd cardinality. We denote by

$$
\begin{array} { l } { { S _ { 0 } = \{ v \in V _ { C } \mid \exists e \neq e ^ { \prime } \in M : \quad e \cap e ^ { \prime } = \{ v \} \} } } \\ { { S _ { 2 } = \{ v \in V _ { C } \mid \exists e \neq e ^ { \prime } \in E _ { C } \setminus M : \quad e \cap e ^ { \prime } = \{ v \} \} } } \end{array}
$$

the set of all vertices on the cycle C with both incident edges in M and the set of all vertices on the cycle C with no incident edges in M, respectively, and

$$
S _ { 1 } = V _ { C } \setminus \left( S _ { 0 } \cup S _ { 2 } \right) .
$$

Then the odd-cycle inequality

$$
\sum _ { v \in S _ { 0 } } x _ { v } - \sum _ { v ^ { \prime } \in S _ { 2 } } x _ { v ^ { \prime } } + \sum _ { e ^ { \prime } \in E _ { C } \backslash M } x _ { e ^ { \prime } } - \sum _ { e \in M } x _ { e } \leq \lfloor | M | / 2 \rfloor\tag{9}
$$

is valid for $\mathrm { B Q P } _ { G }$ (Padberg, 1989). Since $| E _ { C } | = 2 | M | - | S _ { 0 } | + | S _ { 2 } |$ , the inequality obtained from (9) by flipping variables is

$$
\sum _ { v ^ { \prime } \in S _ { 2 } } x _ { v ^ { \prime } } - \sum _ { v \in S _ { 0 } } x _ { v } + \sum _ { e \in M } x _ { e } - \sum _ { e ^ { \prime } \in E _ { C } \backslash M } x _ { e ^ { \prime } } \leq \lfloor M \rfloor / 2 \rfloor \ ,\tag{10}
$$

which, by Lemma 6, is still valid for $\Xi _ { G F }$ . However, if the cycle $C$ is in the graph (V, F) instead of $G ,$ the associated inequality (10) may not be a valid inequality for $\Xi _ { G F }$ anymore, unless M is an edge subset of G, by Lemma 6. Such an example is given in Figure $1 0 \mathrm { ~ a ~ } )$

By Theorem 9 of Padberg (1989), an odd-cycle inequality (9) is facet-defining for $\mathrm { B Q P } _ { G }$ if and only if C is a chordless cycle of G, which is also a necessary condition for (10) to be facet-defining for $\Xi _ { G F }$ with $E \subseteq F$ , by Lemma 6. However, it can be checked that the flipped odd-cycle inequality (10) defines a facet of $\Xi _ { C F _ { C } }$ if and only if $F _ { C } = E _ { C }$ , i.e., the cycle $C$ is chordless in the graph $( V _ { C } , F _ { C } )$ , where $F _ { C } = F \cap ( { } _ { 2 } ^ { V _ { C } } )$ . Therefore, if $E _ { C }$ is a strict subset of $F _ { C }$ , then none of the facet-defining odd-cycle inequalities for the boolean quadric polytope $\mathrm { B Q P } _ { C }$ defines a facet of the multi-separator polytope $\Xi _ { C F _ { C } }$ . See Figure 10 b), c) and d) for an illustration. These examples tell us that the converse of the last statement of Lemma 6 is not true. Figure 10 e) gives a cycle in G that is chordless in $( V , F )$ , whose associated odd-cycle inequality is valid for both $\mathrm { B Q P } _ { ( V , F ) }$ and $\Xi _ { G F }$ but is facet-defining only for $\mathrm { B Q P } _ { ( V , F ) }$ . In Figure 10 f), the cycle is not chordless in $( V , F )$ and hence the associated odd-cycle inequality is not facet-defining for $\mathrm { B Q P } _ { ( V , F ) } ;$ however, it is facet-defining for $\Xi _ { G F }$

While odd-cycle inequalities (9) that are facet-defining for the boolean quadric polytope admit a simple characterization, the examples given in Figure 10 suggest that the conditions under which the odd-cycle inequalities (10) define facets of the multi-separator polytope are more complicated.

![](images/14d46e868fd45bd6c24eb74a5ed1926ec5d51944102b634f155631c7003fa93e.jpg)  
Figure 10: Depicted above are graphs G (solid) and interactions F (both solid and dashed). Vertices and interactions with coeficients +1 and −1 in the odd-cycle inequalities (10) are depicted in red and green, respectively. The odd-cycle inequality associated with the cycle in a) is not valid for the corresponding multi-separator polytope $\Xi _ { G F }$ , which can be seen by looking at the feasible vector $x ^ { \{ v _ { 0 } , v _ { 1 } , v _ { 3 } \} }$ . Each of the odd-cycle inequalities associated with the colored cycles in b), c), d) and e) is valid but not facet-defining for $\Xi _ { G F }$ due to the existence of extra interactions. More specifically, in b) we have the additional equality $x _ { v _ { 5 } } + x _ { v _ { 0 } v _ { 4 } } = x _ { v _ { 0 } v _ { 5 } } + x _ { v _ { 4 } v _ { 5 } }$ containing the face defined by the corresponding odd-cycle inequality. Similarly, the additional inequalities $x _ { v _ { 2 } v _ { 5 } } = x _ { v _ { 3 } v _ { 4 } } , x _ { v _ { 0 } v _ { 3 } } = 1$ , and $x _ { u v _ { 1 } } = x _ { u v _ { 2 } }$ are satisfied in c), d), and $\mathrm { e ) }$ respectively. The odd-cycle inequality associated with the colored cycle in f) is valid and facet-defining for $\Xi _ { G F }$

## 5.2 Lifted multicut polytope

In this section, we show that every lifted multicut polytope (see Definition 7 below) is the projection of a face of some multi-separator polytope, and vice versa. Our proofs are inspired by the linear-time reductions between the multi-separator and lifted multicut problem by Irmai et al. (2024).

Definition 7. (Horˇn´akov´a et al., 2017) For any connected graph $G = ( V , E )$ and any graph $G ^ { \prime } = ( V , F )$ with $E \subseteq F$ , let $Y _ { G G ^ { \prime } }$ denote the set of all $y \in \{ 0 , 1 \} ^ { F }$ that satisfy the following inequalities:

$$
\forall ( V _ { C } , E _ { C } ) \in \operatorname { c y c l e s } ( G ) \ \forall e \in C : \qquad \quad y _ { e } \leq \sum _ { e ^ { \prime } \in E _ { C } \setminus \{ e \} } y _ { e ^ { \prime } } ,\tag{11}
$$

$$
\forall f \in F \setminus E \ \forall ( V _ { P } , E _ { P } ) \in f \mathrm { - p a t h s } ( G ) : \qquad \quad y _ { f } \leq \sum _ { e \in E _ { P } } y _ { e } ,\tag{12}
$$

$$
\forall f \in F \setminus E \forall C \in f \mathrm { - c u t s } ( G ) : \qquad 1 - y _ { f } \leq \sum _ { e \in C } ( 1 - y _ { e } ) .\tag{13}
$$

The convex hull $\Gamma _ { G G ^ { \prime } } : = \mathrm { c o n v } Y _ { G G ^ { \prime } }$ is called the lifted multicut polytope with respect to $G$ and $G ^ { \prime }$

Remark 9. As for the multi-separator problem, replacing the cycles in (11) by chordless cycles, the paths in (12) by chordless paths, and the cuts in (13) by minimal cuts, results in equivalent definitions of the same $Y _ { G G ^ { \prime } }$ and $\Gamma _ { G G ^ { \prime } }$

Recall that the subdivision $\widehat { G }$ of a graph $G = ( V , E )$ is the graph obtained from subdividing all edges of G, i.e., every edge $e = \{ u , w \} \in E$ is replaced by two edges $\{ u , v _ { e } \}$ and $\{ v _ { e } , w \}$ via inserting a new vertex $v _ { e }$ . Therefore, $\widehat { G } = ( \widehat { V } , \widehat { E } )$ , where ${ \widehat { V } } = V \cup \{ v _ { e } \mid e \in E \}$ and $\widehat { E } = \{ \{ v , v _ { e } \} \mid v \in e \in E \}$

Proposition 6. For any connected graph $G = ( V , E )$ , any graph $G ^ { \prime } = ( V , F )$ with $E \subseteq F _ { \mathrm { : } }$ , the lifted multicut polytope $\Gamma _ { G G ^ { \prime } }$ is the projection of a face of the multi-separator polytope $\Xi _ { \widehat { G } F }$ onto $\mathbb { R } ^ { F }$ , where $\widehat { G }$ denotes the graph resulting from the subdivision of all edges of G.

Proof. See Section D.19.

Proposition 7. For any connected graph $G = ( V , E )$ with $F \subseteq \left( { V } \right)$ , the multi-separator polytope $\Xi _ { G F }$ is the projection of a face of the lifted multicut polytope $\Gamma _ { \bar { G } \bar { G } ^ { \prime } . }$ , where $\bar { G }$ is the graph obtained from the subdivision Gb of G by adding a new vertex v¯ and a new edge $\{ v , { \bar { v } } \}$ for each $v \in V$ , and $\bar { G } ^ { \prime }$ is obtained from G<sup>¯</sup> by adding new edges $\{ \bar { v } , v _ { e } \}$ for all $v \in e \in E$ and all edges in $F$

Proof. See Section D.20.

Although these two polytopes are closely related, we do not see a direct way to transform facets from one to the other in general. A special case where such a relation can be identified is when the underlying graph is a path. In the following, we show that the multi-separator polytope for paths is equivalent to the lifted multicut polytope for paths. Thanks to this equivalence and a result from Lange and Andres (2021), we are able to obtain a complete totally dual integral description of the multi-separator polytope in the case where we consider as interactions all the edges that are incident on the vertices of the underlying path.

In this part, we stick to the following notation for simplicity: For any $n \geq 1$ , let $P _ { n } = ( V _ { n } , E _ { n } )$ be the path of length n with $V _ { n } = \{ 0 , \ldots , n \}$ , $E _ { n } = \{ \{ i , i + 1 \} \mid i \in \{ 0 , \ldots , n - 1 \} \}$ , and let $F \subseteq F _ { n } = { \binom { V _ { n } } { 2 } }$ be a superset of $E _ { n }$ . Note that for $P _ { n }$ the connector inequalities (1) and separator inequalities (2) reduce to

$$
\forall i j \in F \colon i < j \colon \quad x _ { i j } \leq \sum _ { k = i } ^ { j } x _ { k }\tag{14}
$$

$$
\forall i j \in F : i < j \quad \forall k \in V : i \leq k \leq j : \quad x _ { k } \leq x _ { i j }\tag{15}
$$

Let $P _ { n + 1 } ^ { \prime } = ( V _ { n + 1 } , E _ { n + 1 } ^ { \prime } )$ be another path, where

$$
E _ { n + 1 } ^ { \prime } = \{ e ^ { \prime } = \{ i , j \} \subseteq V _ { n + 1 } \mid i < j , \{ i , j - 1 \} \in V _ { n } \cup F \} .
$$

The following proposition establishes a correspondence between the multi-separator polytope $\Xi _ { P _ { n } F }$ and the lifted multicut polytope $\Gamma _ { P _ { n + 1 } P _ { n + 1 } ^ { \prime } }$

Proposition 8. For any $n \in \mathbb { N }$ and any superset $F \subseteq { \binom { V _ { n } } { 2 } } o f E _ { n }$ , we have $\Xi _ { P _ { n } F } = \Gamma _ { P _ { n + 1 } P _ { n + 1 } ^ { \prime } } .$ Proof. See Section D.21. □

Lange and Andres (2021) have established a complete description of the lifted multicut polytope for paths which, by the above observation, yields a complete description of the lifted multi-separator polytope for paths. More specifically, let

$$
x _ { 0 n } \leq 1\tag{16}
$$

$$
\forall i \in \{ 0 , \ldots , n - 1 \} : \qquad \quad x _ { 0 i } \leq x _ { 0 , i + 1 }\tag{17}
$$

$$
\forall i \in \{ 1 , \ldots , n \} : \qquad x _ { i n } \leq x _ { i - 1 , n }\tag{18}
$$

$$
\forall i \in \{ 0 , \ldots , n - 1 \} : \qquad x _ { i , i + 1 } \leq x _ { i i } + x _ { i + 1 , i + 1 }\tag{19}
$$

$$
\forall i , j \in \{ 1 , \ldots , n - 1 \} \ \forall i \leq j : \quad x _ { i j } + x _ { i - 1 , j + 1 } \leq x _ { i - 1 , j } + x _ { i , j + 1 }\tag{20}
$$

where $x _ { i i }$ is the variable corresponding to vertex i for $i \in \{ 0 , \ldots , n \}$ . Notice that (16) is a box inequality, (19) are connector inequalities, (17) and (18) are generalized separator inequalities (See Corollary 5), and (20) are intersection inequalities introduced in Section 4.7. System $( 1 6 ) \textrm { -- } ( 2 0 )$ corresponds to system (11) – (15) of Lange and Andres (2021). By Theorem 1 of their work, this latter system provides a complete totally dual integral description of the lifted multicut polytope for paths. As a consequence, we obtain the following result immediately.

Proposition 9. For any $n \in \mathbb { N }$ , we have

$$
\Xi _ { P _ { n } F _ { n } } = \left\{ x \in \mathbb { R } ^ { V _ { n } \cup F _ { n } } \mid x { \mathrm { ~ s a t i s f i e s ~ } } ( 1 6 ) - ( 2 0 ) \right\} .
$$

Furthermore, the system $( 1 6 ) \textrm { -- } ( 2 0 )$ is totally dual integral.

Proof. See Section D.22.

## 6 Conclusion

We describe in terms of eficiently-decidable, graph-theoretic conditions the connector inequalities, separator inequalities and box inequalities that define a facet of the multi-separator polytope. To obtain a tighter polyhedral relaxation, we introduce path and intersection inequalities. For path graphs where every vertex pair forms an interaction, this yields a totally dual integral description of the multi-separator polytope. We relate the multi-separator polytope to the boolean quadric polytope, showing that facets induced by odd-cycle inequalities do not transfer generally, and to the lifted multicut polytope, showing that either polytope is a projection of a face of the other. Our finding that it can be decided eficiently whether a separator inequality defines a facet of the multi-separator polytope is in contrast to the result of Naumann et al. (2024) that deciding whether a cut inequality defines a facet of the lifted multicut-polytope is NP-hard.

## References

Eduardo Alvarez-Miranda, Ivana Ljubi´c, and Petra Mutzel. The maximum weight connected <sup>´</sup> subgraph problem. Facets of combinatorial optimization: Festschrift for Martin Gr¨otschel, pages 245–270, 2013.

Bjoern Andres, Silvia Di Gregorio, Jannik Irmai, and Jan-Hendrik Lange. A polyhedral study of lifted multicuts. Discrete Optimization, 47:100757, 2023. doi: 10.1016/j.disopt.2022.100757.

Yash P. Aneja. An integer linear programming approach to the steiner problem in graphs. Networks, 10(2):167–178, 1980. doi: 10.1002/net.3230100207.

Egon Balas and Cid C. de Souza. The vertex separator problem: a polyhedral investigation. Mathematical Programming, 103(3):583–608, 2005. doi: 10.1007/s10107-005-0574-7.

Thorsten Beier, Constantin Pape, Nasim Rahaman, Timo Prange, Stuart Berg, Davi D Bock, Albert Cardona, Graham W Knott, Stephen M Plaza, Louis K Schefer, Ullrich Koethe, Anna Kreshuk, and Fred A Hamprecht. Multicut brings automated neurite segmentation closer to human performance. Nature Methods, 14:101–102, 2017. doi: 10.1038/nmeth.4151.

Walid Ben-Ameur, Mohamed-Ahmed Mohamed-Sidi, and Jos´e Neto. The k-separator problem: polyhedra, complexity and approximation results. Journal of Combinatorial Optimization, 29: 276–307, 2015. doi: 10.1007/s10878-014-9753-x.

Ralf Bornd¨orfer, Carlos E. Ferreira, and Alexander Martin. Decomposing matrices into blocks. SIAM Journal on Optimization, 9(1):236–269, 1998. doi: 10.1137/S1052623497318682.

Rodolfo Carvajal, Miguel Constantino, Marcos Goycoolea, Juan Pablo Vielma, and Andr´es Weintraub. Imposing connectivity constraints in forest planning models. Operations Research, 61(4):824–836, 2013.

Sunil Chopra and M.R. Rao. The partition problem. Mathematical programming, 59(1-3):87–115, 1993. doi: 10.1007/BF01581239.

Sunil Chopra and M.R. Rao. The Steiner tree problem I: Formulations, compositions and extension of facets. Mathematical Programming, 64:209–229, 1994. doi: 10.1007/BF01582573.

Denis Cornaz, Fabio Furini, Mathieu Lacroix, Enrico Malaguti, A. Ridha Mahjoub, and S´ebastien Martin. The vertex k-cut problem. Discrete Optimization, 31:8–28, 2019a. doi: 10.1016/j.disopt. 2018.07.003.

Denis Cornaz, Youcef Magnouche, Ali Ridha Mahjoub, and S´ebastien Martin. The multi-terminal vertex separator problem: Polyhedral analysis and branch-and-cut. Discrete Applied Mathematics, 256:11–37, 2019b. doi: 10.1016/j.dam.2018.10.005.

George Bernard Dantzig and Delbert Ray Fulkerson. On the Max-Flow Min-Cut Theorem of Networks, pages 215–222. Princeton University Press, 1957. doi: 10.1515/9781400881987-013.

Mohamed Didi Biha and Marie-Jean Meurs. An exact algorithm for solving the vertex separator problem. Journal of Global Optimization, 49:425–434, 2011. doi: 10.1007/s10898-010-9568-y.

R.J. Dufin. The extremal length of a network. Journal of Mathematical Analysis and Applications, 5(2):200–215, 1962. doi: 10.1016/S0022-247X(62)80004-3.

Fernando Escalante. Schnittverb¨ande in Graphen. In Abhandlungen aus dem Mathematischen Seminar der Universit¨at Hamburg, volume 38, pages 199–220. Springer, 1972. doi: 10.1007/ BF02996932.

Fernando Escalante and Tibor Gallai. Note ¨uber Kantenschnittverb¨ande in Graphen. Acta Mathematica Hungarica, 25(1-2):93–98, 1974. doi: 10.1007/BF01901751.

Matteo Fischetti, Markus Leitner, Ivana Ljubi´c, Martin Luipersbeck, Michele Monaci, Max Resch, Domenico Salvagnin, and Markus Sinnl. Thinning out steiner trees: a node-based model for uniform edge costs. Mathematical Programming Computation, 9(2):203–229, 2017.

Fabio Furini, Ivana Ljubi´c, Enrico Malaguti, and Paolo Paronuzzi. On integer and bilevel formulations for the k-vertex cut problem. Mathematical Programming Computation, 12:133–164, 2020. doi: 10.1007/s12532-019-00167-1.

John R. Gilbert, Joan P. Hutchinson, and Robert Endre Tarjan. A separator theorem for graphs of bounded genus. Journal of Algorithms, 5(3):391–407, 1984. doi: 10.1016/0196-6774(84)90019-1.

Frank G¨oring. Short proof of Menger’s Theorem. Discrete Mathematics, 219(1):295–296, 2000. doi: 10.1016/S0012-365X(00)00088-1.

R. Halin. Lattices Related to Separation in Graphs. In Finite and Infinite Combinatorics in Sets and Logic, pages 153–167. Springer, 1993. doi: 10.1007/978-94-011-2080-7 10.

Andrea Horˇn´akov´a, Jan-Hendrik Lange, and Bjoern Andres. Analysis and optimization of graph decompositions by lifted multicuts. In International Conference on Machine Learning (ICML), 2017. doi: 10.5555/3305381.3305540.

Jannik Irmai, Shengxian Zhao, Mark Sch¨one, Jannik Presberger, and Bjoern Andres. A graph multi-separator problem for image segmentation. Journal of Mathematical Imaging and Vision, 66(5):839–872, 2024. doi: 10.1007/s10851-024-01201-1.

Ken-ichi Kawarabayashi and Bruce Reed. A Separator Theorem in Minor-Closed Classes. In Symposium on Foundations of Computer Science (FOCS), pages 153–162, 2010. doi: 10.1109/ FOCS.2010.22.

Margret Keuper. Higher-order minimum cost lifted multicuts for motion segmentation. In International Conference on Computer Vision (ICCV), 2017. doi: 10.1109/ICCV.2017.455.

Margret Keuper, Evgeny Levinkov, Nicolas Bonneel, Guillaume Lavou´e, Thomas Brox, and Bjoern Andres. Eficient decomposition of image and mesh graphs by lifted multicuts. In International Conference on Computer Vision (ICCV), 2015. doi: 10.1109/ICCV.2015.204.

Margret Keuper, Siyu Tang, Bjoern Andres, Thomas Brox, and Bernt Schiele. Motion segmentation and multiple object tracking by correlation co-clustering. Transactions on Pattern Analysis and Machine Intelligence, 42(1):140–153, 2020. doi: 10.1109/TPAMI.2018.2876253.

Jan-Hendrik Lange and Bj¨orn Andres. On the lifted multicut polytope for trees. In Pattern Recognition: 42nd DAGM German Conference, DAGM GCPR 2020, T¨ubingen, Germany, September 28–October 1, 2020, Proceedings 42, pages 360–372. Springer, 2021.

Richard J. Lipton and Robert Endre Tarjan. A Separator Theorem for Planar Graphs. SIAM Journal on Applied Mathematics, 36(2):177–189, 1979. doi: 10.1137/0136016.

Karl Menger. Zur allgemeinen Kurventheorie. Fundamenta Mathematicae, 10(1):96–115, 1927.

Anna Moss and Yuval Rabani. Approximation algorithms for constrained node weighted Steiner tree problems. SIAM Journal on Computing, 37(2):460–481, 2007. doi: 10.1137/S0097539702420474.

Lucas Fabian Naumann, Jannik Irmai, Shengxian Zhao, and Bjoern Andres. Box Facets and Cut Facets of Lifted Multicut Polytopes. In International Conference on Machine Learning (ICML), 2024. URL https://proceedings.mlr.press/v235/naumann24a.html.

Maarten Oosten, Jeroen H. G. C. Rutten, and Frits C. R. Spieksma. Disconnecting graphs by removing vertices: a polyhedral approach. Statistica Neerlandica, 61(1):35–60, 2007. doi: 10.1111/j.1467-9574.2007.00350.x.

Manfred Padberg. The boolean quadric polytope: Some characteristics, facets and relatives. Mathematical programming, 45:139–172, 1989. doi: 10.1007/BF01589101.

Daniel Rehfeldt, Henriette Franz, and Thorsten Koch. Optimal connected subgraphs: Integer programming formulations and polyhedra. Networks, 80(3):314–332, 2022.

Alexander Schrijver. Combinatorial Optimization: Polyhedra and Eficiency, volume 24. Springer-Verlag Berlin, 2003.

Boris Shapiro and Arkady Vaintrob. On algebras and matroids associated to undirected graphs. arXiv preprint arXiv:2005.06160, 2020.

Cid de Souza and Egon Balas. The vertex separator problem: algorithms and computations. Mathematical Programming, 103(3):609–631, 2005. doi: 10.1007/s10107-005-0573-8.

Yiming Wang, Austin Buchanan, and Sergiy Butenko. On imposing connectivity constraints in integer programs. Mathematical Programming, 166:241–271, 2017.

## A Separation algorithms

In this section we describe how to eficiently separate connector and separator inequalities. In particular, we show that the connector inequalities (1) can be separated by searching for shortest paths and that the separator inequalities (2) can be separated by solving a max-flow problem in an auxiliary graph.

Proposition 10. Let $G = ( V , E )$ be a connected graph, let $F \subseteq \left( { V } \right)$ be a set of interactions and let $x \in \mathsf { \bar { [ 0 , 1 ] } } ^ { V \cup F }$ . Then one can decide in time $\mathcal { O } ( | E | | V | + | V | ^ { 2 } l o g | \dot { V } | )$ whether there exists a violated connector inequality and return such an inequality.

Proof. We define the non-negative edge weights $\begin{array} { r } { w _ { u v } = \frac { 1 } { 2 } ( x _ { u } + x _ { v } ) \ \mathrm { f o r } \ \{ u , v \} \in E . } \end{array}$ . By this definition, for every {s, t}-path $P = ( V _ { P } , E _ { P } )$ in G it holds that

$$
w ( P ) : = \sum _ { e \in E _ { P } } w _ { e } = \sum _ { u \in V _ { P } } x _ { u } - \frac { 1 } { 2 } ( x _ { s } + x _ { t } ) .\tag{21}
$$

Now let $f = \{ s , t \} \in F$ and let $P = ( V _ { P } , E _ { P } )$ be a minimum weighted f-path in G with weights w. Then x violates a connector inequality associated with f if and only if $w ( P ) + \textstyle { \frac { 1 } { 2 } } ( x _ { s } + x _ { t } ) < x _ { f }$ by the following arguments.

Suppose it holds that $w ( P ) + \textstyle { \frac { 1 } { 2 } } ( x _ { s } + x _ { t } ) < x _ { f }$ . Clearly, $V _ { P }$ is an f-connector of G and (21) yields that x violates the connector inequality associated with the interaction f and the f-connector $V _ { P }$

Conversely, suppose that there exists an f-connector C such that x violates the connector inequality associated with f and C. By Lemma 1, every connector inequality associated with a non-minimal connector is dominated by a connector inequality associated with a minimal connector. Let $C ^ { \prime }$ be a minimal $f -$ connector of $G$ with $C ^ { \prime } \subseteq C$ . Since $C ^ { \prime }$ is an f-minimal it induces an $f \mathrm { - p a t h }$ $P ^ { \prime } = \left( V _ { P ^ { \prime } } , E _ { P ^ { \prime } } \right)$ with $V _ { P ^ { \prime } } = C ^ { \prime }$ . By the assumption that $P$ is a minimum-weight $f \mathrm { - p a t h }$ , together with (21) and the violated connector inequality, we obtain

$$
w ( P ) + { \textstyle \frac { 1 } { 2 } } ( x _ { s } + x _ { t } ) \le w ( P ^ { \prime } ) + { \textstyle \frac { 1 } { 2 } } ( x _ { s } + x _ { t } ) = \sum _ { u \in C ^ { \prime } } x _ { u } \le \sum _ { u \in C } x _ { u } < x _ { f } .
$$

Using Dijkstra’s algorithm, shortest f-paths in G for all $f \in F$ can be computed in time $\mathcal { O } ( | E | | V | + | V | ^ { 2 } \mathrm { l o g } | V | )$ □

Proposition 11. Let $G = ( V , E )$ be a connected graph, let $F \subseteq ( \Sigma )$ be a set of interactions and let $\bar { x ^ { } } \in [ 0 , 1 ] ^ { V \cup F }$ . Then for any $f \in F _ { \mathrm { : } }$ one can find an f-separator S of G such that x violates the separator inequality associated with $f$ and S or decide that such a separator does not exist in time $\mathcal { O } ( | V | ^ { 2 } | E | )$ .

Proof. Suppose $f = \{ u , v \}$ . Firstly, we define the auxiliary directed graph $G ^ { \prime } = ( V ^ { \prime } , E ^ { \prime } )$ with

$$
\begin{array} { r l } & { V ^ { \prime } = \lbrace s ^ { \prime } , s ^ { \prime \prime } \mid s \in V \rbrace , } \\ & { E ^ { \prime } = \lbrace ( s ^ { \prime } , s ^ { \prime \prime } ) \mid s \in V \rbrace } \\ & { \qquad \cup \lbrace ( s ^ { \prime \prime } , t ^ { \prime } ) , ( t ^ { \prime \prime } , s ^ { \prime } ) \mid s t \in E , s , t \notin \lbrace u , v \rbrace \rbrace } \\ & { \qquad \cup \lbrace ( u ^ { \prime \prime } , s ^ { \prime } ) \mid s \in V \setminus \lbrace u \rbrace \mathrm { ~ w i t h ~ } u s \in E \rbrace } \\ & { \qquad \cup \lbrace ( s ^ { \prime \prime } , v ^ { \prime } ) \mid s \in V \setminus \lbrace v \rbrace \mathrm { ~ w i t h ~ } v s \in E \rbrace . } \end{array}
$$

Further, we define the non-negative edge weights $w _ { ( s ^ { \prime } , s ^ { \prime \prime } ) } = 1 - x _ { s }$ for $s \in V$ and $w _ { a } = 2$ for all other arcs $a \in E ^ { \prime } \setminus \{ ( s ^ { \prime } , s ^ { \prime \prime } ) ~ | ~ s \in V \}$ . We show that there exists an f-separator S such that x violates the separator inequality associated with f and S if and only if the value of a minimum $( u ^ { \prime } , v ^ { \prime \prime } )$ -cut of $G ^ { \prime }$ is strictly smaller than $1 - x _ { f }$

Since $( u ^ { \prime } , u ^ { \prime \prime } )$ is the only arc in $E ^ { \prime }$ that is adjacent to $u ^ { \prime } ,$ , we observe that the set $\{ ( u ^ { \prime } , u ^ { \prime \prime } ) \}$ is a $( u ^ { \prime } , v ^ { \prime \prime } )$ -cut of value $w _ { ( u ^ { \prime } , u ^ { \prime \prime } ) } = ( 1 - x _ { u } ) \leq 1$ . Let $S ^ { \prime }$ be a minimum $( u ^ { \prime } , v ^ { \prime \prime } )$ -cut of $G ^ { \prime }$ with weight $\textstyle w ( S ^ { \prime } ) : = \sum _ { a \in S ^ { \prime } } w _ { a }$ . By the above observation and the minimality of $S ^ { \prime }$ , it holds that $w ( S ^ { \prime } ) \leq 1$ . It follows from the definition of w that $S ^ { \prime } \subseteq \{ ( s ^ { \prime } , s ^ { \prime \prime } ) \mid s \in V \}$ . Let $S \subseteq V$ such that $S ^ { \prime } = \{ ( s ^ { \prime } , s ^ { \prime \prime } ) \mid s \in S \}$ . By our definition of $G ^ { \prime }$ , the vertex set S is an f-separator of $G .$ . Conversely, every f-separator $T \subseteq V$ induces a $( u ^ { \prime } , v ^ { \prime \prime } )$ -cut $T ^ { \prime } = \{ ( s ^ { \prime } , s ^ { \prime \prime } ) \ | \ s \in T \}$ of $G ^ { \prime }$ . By the definition of the weights, it holds that $\begin{array} { r } { \sum _ { s \in S } ( 1 - x _ { s } ) = w ( S ^ { \prime } ) } \end{array}$ . Therefore, there exists a violated separator inequality associated with the interaction f if and only if the minimum $( u ^ { \prime } , v ^ { \prime \prime } )$ -cut $S ^ { \prime }$ has value $w ( S ^ { \prime } ) < ( 1 - x _ { f } )$

Using Dinic’s algorithm a minimum $( u ^ { \prime } , v ^ { \prime \prime } ) – \mathrm { c u t }$ of $G ^ { \prime }$ can be computed in time $\mathcal { O } ( | V ^ { \prime } | ^ { 2 } | E ^ { \prime } | )$ The result follows from the observation $| V ^ { \prime } | = 2 | V |$ and $| E ^ { \prime } | = | V | + 2 | E |$ □

## B Construction of feasible vectors

In this appendix, we introduce a basic technique for constructing from a set $X ^ { \prime }$ of feasible vectors of the multi-separator problem vectors in the the linear space

$$
\operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) = \left\{ \sum _ { x \in X ^ { \prime } } \alpha _ { x } x \ \middle \vert \ \alpha \in \mathbb { R } ^ { X ^ { \prime } } , \ \sum _ { x \in X ^ { \prime } } \alpha _ { x } = 0 \right\} \ .
$$

Lemma 7. Let G be a graph, let $F$ be a set of interactions on $G ,$ , and let $X ^ { \prime } \subseteq X _ { G F }$ . Suppose that A and B are two vertex subsets of G such that the feasible vectors $x ^ { A } , x ^ { B } , x ^ { A \cap B }$ and $\bar { x } ^ { A \cup B }$ are all in $X ^ { \prime }$ , then

$$
\mathbb { 1 } _ { Q _ { A B } } \in \mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } ) ,\tag{22}
$$

$$
\mathbb { 1 } _ { F _ { A B } } - \mathbb { 1 } _ { H _ { A B } } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) .\tag{23}
$$

![](images/01bc6a19b422a0155dd299d584daf23acf4080c7ebff7fe75f711ec491f1cbec.jpg)  
Figure 11: Examples illustrating the constructions in Lemma 7. Depicted are graphs G (solid, black) and interactions F (dashed, green).

where<sup>6</sup>

$Q _ { A B } : = \{ f \in F \mid A \cup B$ is an f-connector of G, but A ∩ B is not} $\cup \ ( A \triangle B )$ 2

$F _ { A B } : = \{ f \in F \mid A \cup B$ is an f-connector of G, but A and B are not} ,

$H _ { A B } : = \{ f \in F \mid A$ and B are f-connectors of G but $A \cap B$ is not} .

Proof. For the given two vertex subsets A and B, we define

$$
\begin{array} { l } { { y ^ { A , B } : = x ^ { A \cap B } - x ^ { A \cup B } , } } \\ { { z ^ { A , B } : = x ^ { A } + x ^ { B } - x ^ { A \cap B } - x ^ { A \cup B } . } } \end{array}
$$

Both $y ^ { A , B }$ and $z ^ { A , B }$ z<sup>A,B</sup> are in the linear space lin $( X ^ { \prime } - X ^ { \prime } )$ because $x ^ { A } , x ^ { B } , x ^ { A \cap B }$ and $x ^ { A \cup B }$ are in $X ^ { \prime }$ , by the assumption, and the sum of coeficients in the forms that define $y ^ { A , B }$ and $z ^ { A , B }$ equals zero. Thus, it is suficient to show

$$
y ^ { A , B } = \mathbb { 1 } _ { Q _ { A B } } ,\tag{24}
$$

$$
z ^ { A , B } = \mathbb { I } _ { F _ { A B } } - \mathbb { I } _ { H _ { A B } } \mathrm { ~ . ~ }\tag{25}
$$

Equation (24) is a straightforward consequence of the definition of $y ^ { A , B }$ . To establish (25), we define

$$
A ^ { \prime } : = A \cup \{ f \in F \mid f { \mathrm { ~ i s ~ c o n n e c t e d ~ i n ~ } } G [ A ] \} ,
$$

$$
B ^ { \prime } : = B \cup \{ f \in F \mid f { \mathrm { ~ i s ~ c o n n e c t e d ~ i n ~ } } G [ B ] \} ,
$$

and observe that the following equations hold:

$$
\begin{array} { r l } & { \mathbb { 1 } _ { A ^ { \prime } } = \mathbb { 1 } _ { V \cup F } - x ^ { A } , } \\ & { \mathbb { 1 } _ { B ^ { \prime } } = \mathbb { 1 } _ { V \cup F } - x ^ { B } , } \\ & { \mathbb { 1 } _ { A ^ { \prime } \cup B ^ { \prime } } = \mathbb { 1 } _ { V \cup F } - x ^ { A \cup B } - \mathbb { 1 } _ { F _ { A B } } , } \\ & { \mathbb { 1 } _ { A ^ { \prime } \cap B ^ { \prime } } = \mathbb { 1 } _ { V \cup F } - x ^ { A \cap B } + \mathbb { 1 } _ { H _ { A B } } . } \end{array}
$$

By the inclusion-exclusion principle, the characteristic vectors defined by $A ^ { \prime }$ and $B ^ { \prime }$ are related by the identity

$$
\mathbb { 1 } _ { A ^ { \prime } } + \mathbb { 1 } _ { B ^ { \prime } } - \mathbb { 1 } _ { A ^ { \prime } \cap B ^ { \prime } } - \mathbb { 1 } _ { A ^ { \prime } \cup B ^ { \prime } } = 0 ,
$$

which, together with the above equalities, implies (25).

The example below illustrates how the constructions $Q _ { A B } , F _ { A B }$ and $H _ { A B }$ introduced in Lemma 7 are computed in a special, but typical, case, i.e., when $A \backslash B$ and $B \setminus A$ contain at most one vertex.

Example 2. Let G and F be as in Lemma 7. If $A \subseteq B$ and $B \setminus A = \{ b \}$ , then

$$
Q _ { A B } = \{ b \} \cup \{ f \in F \mid b { \mathrm { ~ i s ~ a n ~ } } f { \mathrm { - c u t - v e r t e x ~ o f ~ } } G [ B ] \} \ ,
$$

and $F _ { A B } = H _ { A B } = \varnothing$ . If A and B satisfy $A \setminus B = \{ a \}$ and $B \setminus A = \{ b \}$ , then

$F _ { A B } = \{ f \in F \mid f \subseteq A \cup B$ , both a and b are f-cut-vertices of $G [ A \cup B ] \}$

$H _ { A B } = \{ f \in F \mid f \subseteq A \cap B$ , {a, b} is a minimal f-separator of $G [ A \cup B ] \}$

We illustrate these cases using Figure 11.

(A) In Figure 11 a), b) and c), let B be the vertex set of G and let $A = B \setminus \{ u \}$ . Then $Q _ { A B } = \{ u , u v , u w \}$ in Figure 11 a) and $Q _ { A B } = \{ u , u w \}$ in Figure 11 b), c). (B) In Figure 11 a), let $A = \{ u , v \}$ and $\boldsymbol { B } = \{ \boldsymbol { v } , \boldsymbol { w } \}$ . Clearly, $A \setminus B = \{ u \}$ and $B \setminus A = \{ w \}$ Thus, $F _ { A B } = \{ u w \}$ and $H _ { A B } = \mathcal { O }$

(C) In Figure 11 b) and c), let $A = \{ u , v , v ^ { \prime } \}$ and $B = \{ v , v ^ { \prime } , w \}$ . Clearly, $A \setminus B = \{ u \}$ and $B \setminus A = \{ w \}$ . In Figure 11 b), we have $F _ { A B } = \{ u w \}$ and $H _ { A B } = \mathcal { O }$ , since $\{ u , w \}$ is not a minimal $\{ v , v ^ { \prime } \}$ -separator of G. In contrast, $\{ u , w \}$ is a minimal $\{ v , v ^ { \prime } \}$ -separator of G in Figure 11 c), and thus $H _ { A B } = \{ v v ^ { \prime } \}$ , though $F _ { A B } = \{ u w \}$ as before.

## C Dimension and pseudoforests

In this section, we give an algebraic lemma that we apply in order to establish the suficiency of conditions for an inequality to induce a facet of the multi-separator polytope.

To begin with, we recall additional, basic notions from graph theory: A loop graph is an ordered pair of disjoint sets $( V , E )$ with $E \subseteq { \binom { V } { 1 } } \cup { \binom { V } { 2 } }$ . The incidence matrix of a loop graph $G = ( V , E )$ is defined as the binary vertex-edge matrix $\mathbf { \bar { \boldsymbol { M } } } \in \{ 0 , 1 \} ^ { V \times E }$ such that $M ( v , e ) = 1$ if and only if $v \in e , \mathrm { ~ A ~ }$ pseudoforest is a loop graph such that each of its connected components contains at most one cycle (including loops). It is immediate from the definition that a loop graph $G = ( V , E )$ is a pseudoforest if and only if for any vertex subset $U \subseteq V$ , the number of edges in the induced subgraph $G [ U ]$ is at most the number of vertices in U.

The result below relates pseudoforests to loop graphs whose incidence matrices have full column rank.

Lemma 8. The incidence matrix of a loop graph has full column rank $i f$ and only if the loop graph is a pseudoforest without even cycles.

Proof. See Shapiro and Vaintrob (2020).

## D Proofs

## D.1 Proof of Proposition 1

For necessity of Part (i): Since U is an $A , B .$ -connector, $G [ U ]$ contains an ab-path for some $a \in A$ $b \in B$ . A minimal such path gives a connector contained in U, hence by minimality it uses all vertices of U. Thus $G [ U ]$ is an induced path. Moreover, if $U \cap A$ contained a vertex diferent from the first endpoint, or $U \cap B$ contained a vertex diferent from the last endpoint, then a proper subpath would still be an A, B-connector, contradicting minimality. Hence $U \cap A = \{ u _ { 0 } \}$ and $U \cap B = \{ u _ { n } \}$

For suficiency of Part (i): If $G [ U ]$ is such a path, then any A, B-connector contained in U must contain the unique A-vertex u and the unique B-vertex u . Since $G [ U ]$ is a path, the only u u -path in $G [ U ]$ uses all vertices of U. Therefore no proper subset of U is an $A , B .$ -connector, so U is minimal.

For suficiency of Part (ii): We first observe that U is indeed an $\{ A , B \}$ -separator of $G ,$ since every $\{ A , B \}$ -connector of G contains a minimal $\{ A , B \}$ -connector of G. If U is a non-minimal {A, B}-separator of G, then we can find a vertex $u \in U$ such that $U \setminus \{ u \}$ is also an $\{ A , B \}$ separator of G. This means that $C \cap U \neq \{ u \}$ holds for any minimal {A, B}-connector C of G, a contradiction.

For necessity of Part (ii): Suppose U is a minimal {A, B}-separator of G. By the definition of separator, every minimal $\{ A , B \}$ -connector of G intersects U. If there exists a vertex $u \in U$ such that the intersection of U with any minimal $\{ A , B \}$ -connector C of G is not $\{ u \}$ , then it follows from the observation in the proof of suficiency that $U \backslash \{ u \}$ is also an $\{ A , B \}$ -separator of G. This contradicts the minimality of $U$ □

## D.2 Proof of Lemma 1

For the first part of the lemma, we verify that

$$
X _ { G F } = \{ x \in \mathbb { Z } ^ { V \cup F } \mid x { \mathrm { ~ s a t i s f i e s ~ } } ( 1 ) , ( 2 ) { \mathrm { ~ a n d ~ } } ( 3 ) \} .
$$

Clearly, every feasible vector $x \in X _ { G F }$ satisfies all the inequalities $( 1 ) , ( 2 )$ and (3). To see the converse, we just need to show that for any $f \in F$ and any $\mathbf { \bar { \Phi } } _ { x } \in \{ 0 , 1 \} ^ { \dot { V } \cup \dot { F } }$ satisfying (1) and (2), the following statements are equivalent:

(A) $x _ { f } = 0 .$

(B) Every f-separator of G contains a vertex at which x takes zero value.

(C) There exists an f-connector of G whose vertices are labeled zero by x.

It is immediate from the separator inequalities (2) that (A) implies (B). If (B) holds but every f-connector C of G contains at least one vertex at which x takes 1, let S be the set of all these vertices labeled 1 by x. Then S is an f-separator of G, a contradiction. Thus (C) follows from (B). By the connector inequalities (1), part (C) implies (A).

For the second part of the lemma, we show that for a given interaction $f \in F$ , connector inequalities associated with non-minimal f-connectors of G and separator inequalities associated with non-minimal f-separators of G are implied by stronger ones and hence redundant in the linear relaxation. If C is an arbitrary f-connector of $G ,$ then for every minimal $f .$ -connector $C ^ { \prime }$ of $G$ contained in $C _ { i }$ we have

$$
x _ { f } \leq \sum _ { v \in C ^ { \prime } } x _ { v } \leq \sum _ { v \in C } x _ { v } .
$$

That ${ \mathrm { i s } } ,$ the connector inequality for $C ^ { \prime }$ implies the connector inequality for $C .$ Similarly, let S be any f-separator of G and let $S ^ { \prime }$ be a minimal f-separator of G that is contained in S, then

$$
1 - x _ { f } \leq \sum _ { v \in S ^ { \prime } } ( 1 - x _ { v } ) \leq \sum _ { v \in S } ( 1 - x _ { v } ) .
$$

That is, the separator inequality for $S ^ { \prime }$ implies the separator inequality for S. Therefore, it sufices to enforce connector inequalities only for minimal f-connectors and separator inequalities only for minimal f-separators. This completes the proof. □

## D.3 Proof of Proposition 2

Since lin $( X _ { G F } - X _ { G F } )$ is the linear space associated with the afine hull of the multi-separator polytope, it sufices to show that ${ \mathbb { 1 } } _ { \{ g \} } \in \mathrm { l i n } ( X _ { G F } - X _ { G F } )$ for every $g \in V \cup F$

First let $v \in V$ . Set $A = \{ v \}$ and $B = \varnothing$ . Then $A \triangle B = \{ v \}$ . Moreover, since every interaction in $F \subseteq ( \Sigma )$ has two distinct endvertices, $A \cup B = \{ v \}$ is not an f-connector for any $f \in F$ . Hence $Q _ { A B } = \bar { \{ v \} }$ . By (22) of Lemma 7, we obtain $\mathbb { 1 } _ { \{ v \} } \in \operatorname* { l i n } ( X _ { G F } - X _ { G F } )$

Now let $f = \{ u , w \} \in F$ . Since G is connected, there exists a minimal f-connector U of G. By Corollary 1, G[U] is an induced path. Set $A = U \setminus \{ u \}$ and $B = U \setminus \{ w \}$ . Then $A \cup B = U$ and $A \cap B = U \setminus \{ u , w \}$ . Since U is a minimal f-connector, neither A nor B is an f-connector, while $A \cup B$ is. Hence $f \in F _ { A B }$ . Moreover, if $g \in F _ { A B }$ , then g must have one endvertex in $U \setminus A = \{ u \}$ and the other in $U \setminus B = \{ w \}$ , and therefore $g = f .$ Thus $F _ { A B } = \{ f \}$ . Finally, any interaction $g \in F$ with $g \subseteq A$ ∩ B is connected in $G [ A \cap B ]$ , so $H _ { A B } = \mathcal { O }$ . Therefore, by (23) of Lemma $^ { 7 , }$ we have $\mathbb { 1 } _ { \{ f \} } \in \operatorname* { l i n } ( X _ { G F } - X _ { G F } )$ , and the proof is complete. □

## D.4 Proof of Proposition 3

Necessity being trivial, we show suficiency. Let us denote $F ( a ) = \{ f \in F \mid a _ { f } \neq 0 \}$ and $V ( a ) = V ( a ) ^ { + } \cup V ( a ) ^ { - }$ , where $V ( a ) ^ { + } = \{ v \in V \mid a _ { v } > 0 \}$ and $V ( a ) ^ { - } = \{ v \in V \mid a _ { v } < 0 \}$ . The suficiency part asserts that if $| F ( a ) | \le 1$ , then the given facet-defining inequality

$$
a x \leq \alpha\tag{26}
$$

must be canonical. In the following, we consider two possible cases $F ( a ) = \emptyset$ and $| F ( a ) | = 1$ separately.

Suppose $F ( a ) = \emptyset$ . Then inequality (26) has the form

$$
\sum _ { v \in V ( a ) ^ { + } } a _ { v } x _ { v } \leq \sum _ { v \in V ( a ) ^ { - } } ( - a _ { v } ) x _ { v } + \alpha .\tag{27}
$$

As $x ^ { V ( a ) ^ { - } }$ is feasible, it holds that $\begin{array} { r } { \sum _ { v \in V ( a ) ^ { + } } a _ { v } \le \alpha . \ \mathrm { I f } \sum _ { v \in V ( a ) ^ { + } } a _ { v } < \alpha } \end{array}$ , then inequality (27) has to be strict for all feasible vectors, which contradicts the assumption that (26) is facet-defining. Thus we have $\begin{array} { r } { \alpha = \sum _ { v \in V ( a ) ^ { + } } a _ { v } , } \end{array}$ , in which case (26) becomes

$$
\sum _ { v \in V ( a ) ^ { + } } a _ { v } ( 1 - x _ { v } ) + \sum _ { v \in V ( a ) ^ { - } } ( - a _ { v } ) x _ { v } \geq 0 .
$$

This inequality is a linear combination of the box inequalities $x _ { v } \ge 0$ and $1 - x _ { v } \ge 0$ for $v \in V$ with nonnegative coeficients. Therefore, we must have either $| V ( a ) ^ { + } | = 1$ and $V ( a ) ^ { - } = \emptyset$ , or $V ( a ) ^ { + } = \emptyset$ and $| V ( a ) ^ { - } | = 1$ , which proves the first case $F ( a ) = \emptyset$

We continue with the second case $| F ( a ) | = 1$ . Suppose $F ( a ) = \{ f \}$ for some interaction $f \in F$ If $V ( a )$ is empty, then (26) becomes a box inequality associated with $f .$ From now on, we assume that $V ( a )$ is non-empty. For convenience of notation we rewrite inequality (26) as

$$
a _ { f } x _ { f } + \sum _ { v \in V ( a ) ^ { + } } a _ { v } x _ { v } \leq \sum _ { v \in V ( a ) ^ { - } } b _ { v } x _ { v } + \alpha ,\tag{28}
$$

where $b \in \mathbb { Z } _ { > 0 } ^ { V }$ is defined by $b _ { v } = - a _ { v }$ if $v \in V ( a ) ^ { - }$ , and 0 otherwise.

Firstly, we consider the case $a _ { f } > 0$ . As the feasible vector $x ^ { V \setminus V ( a ) ^ { + } }$ satisfies (28), it follows that α $\begin{array} { r } { \sum _ { v \in V ( a ) ^ { + } } a _ { v } } \end{array}$ . Furthermore, this inequality cannot be strict, i.e., we must have $\alpha =$ $\textstyle \sum _ { v \in V ( a ) ^ { + } } a _ { v } ,$ , otherwise the equality $x _ { f } = 1$ would hold for any feasible vector x that satisfies $( 2 8 )$ at equality, contradicting the assumptions that (28) is facet-defining and $V ( a ) \neq \emptyset$ . Now, substituting $\begin{array} { r } { \alpha = \sum _ { v \in V ( a ) ^ { + } } a _ { v } } \end{array}$ in (28) gives

$$
a _ { f } x _ { f } \le \sum _ { v \in V ( a ) ^ { + } } a _ { v } ( 1 - x _ { v } ) + \sum _ { v \in V ( a ) ^ { - } } b _ { v } x _ { v } .\tag{29}
$$

If $V ( a ) ^ { - }$ is not an f-connector of $G ,$ then the feasible vector $x ^ { V ( a ) }$ satisfies $x _ { f } ^ { V ( a ) ^ { - } } = 1$ , which implies $a _ { f } \leq 0$ by (29). This is a contradiction. Therefore, $V ( a ) ^ { - }$ must be an f-connector of $G .$ For every f-separator S of $G [ V ( a ) ^ { - } ]$ , it holds that

$$
a _ { f } \le \sum _ { v \in S } b _ { v } ,\tag{30}
$$

since $x ^ { V ( a ) ^ { - } \setminus S } \in X _ { G F }$ . This implies that $V ( a ) ^ { + } = \emptyset$ . Otherwise, since $x _ { f } = 1$ for any $x = x ^ { V \setminus S }$ ， $\mathrm { i . e . , } f$ is separated by S in G, implies that $S \cap V ( a ) ^ { - }$ is an f-separator of ${ \dot { G } } [ V ( a ) ^ { - } ]$ , it follows from (30) that the equality $x _ { v } = 1$ holds for every $v \in V ( a ) ^ { + }$ and every feasible vector x that satisfies (29) at equality. Hence, (29) reduces to

$$
a _ { f } x _ { f } \leq \sum _ { v \in V ( a ) ^ { - } } b _ { v } x _ { v } .\tag{31}
$$

Let us give each vertex $v \in V$ the capacity $b _ { v }$ and each edge an infinite capacity. By (30) and the fact that inequality (31) is facet-defining, we see that the minimal capacity of an f-separator is precisely $a _ { f }$ . Then, by the Max-Flow Min-Cut Theorem (Dantzig and Fulkerson, 1957; Schrijver, 2003), there exist nonnegative integers $\kappa _ { C }$ for all $f .$ -connectors C of $G [ V ( a ) ^ { - } ]$ such that

$$
\sum _ { C \in f \mathrm { - c o n n e c t o r s } ( G [ V ( a ) ^ { - } ] ) } \kappa _ { C } = a _ { f }
$$

and

$$
\sum _ { C \in f \mathrm { - } \mathrm { c o n n e c t o r s } ( G [ V ( a ) ^ { - } ] ) } \kappa _ { C } \mathbb { 1 } _ { C } \leq b .
$$

Here inequalities between vectors are meant to be componentwise. From this, we see that

$$
\begin{array} { r l r } {  { - a _ { f } x _ { f } + \sum _ { v \in V ( a ) - \atop v \in V ( a ) - 1 } b _ { v } x _ { v } } } \\ & { = } & { \sum _ { C \in f \mathrm { - c o n n e c t o r s } ( G [ V ( a ) - ] ) } \kappa _ { C } ( - x _ { f } + \sum _ { v \in C } x _ { v } ) + ( b - \sum _ { C \in f \mathrm { - c o n n e c t o r s } ( G [ V ( a ) - ] ) } \kappa _ { C } 1 _ { C } ) \cdot x | _ { V } . } \end{array}
$$

That is, (31) can be written as a nonnegative linear combination of connector inequalities associated with the interaction $f$ and f-connectors of $G ,$ and box inequalities $x _ { v } \geq 0$ associated with vertices $v \in V ( a ) ^ { - }$ . Thus, we conclude that (29) must be a connector inequality under the specified assumption.

Next, we consider the other case $a _ { f } ~ < ~ 0 .$ . As before, since $x ^ { V ( a ) ^ { - } } \in X _ { G F }$ , we get $a _ { f } +$ $\begin{array} { r } { \sum _ { v \in V ( a ) ^ { + } } a _ { v } \ \leq \ \alpha } \end{array}$ from (28). Again, the number α is determined by the vector $^ { a , }$ that is, $\begin{array} { r } { a _ { f } + \sum _ { v \in V ( a ) ^ { + } } a _ { v } = \alpha _ { : } } \end{array}$ , otherwise we would $\mathrm { g e t }$ the equality $x _ { f } = 0$ that holds for all feasible vectors x that satisfy (28) at equality, contradicting the assumptions that (28) is facet-defining and $V ( a ) \neq \emptyset$ . Hence, inequality (28) can be written as

$$
- a _ { f } ( 1 - x _ { f } ) \leq \sum _ { v \in V ( a ) ^ { + } } a _ { v } ( 1 - x _ { v } ) + \sum _ { v \in V ( a ) ^ { - } } b _ { v } x _ { v } ,\tag{32}
$$

If $V ( a ) ^ { + }$ is not an f-separator of $G ,$ then the feasible vector $x ^ { V \setminus V ( a ) ^ { + } }$ satisfies $x _ { f } ^ { V \setminus V ( a ) ^ { + } } = 0$ which implies $a _ { f } \geq 0 ,$ , a contradiction. Therefore, $V ( a ) ^ { + }$ must be an f-separator of G. For each f-connector C of G, we have

$$
- a _ { f } \leq \sum _ { v \in V ( a ) ^ { + } \cap C } a _ { v } ,\tag{33}
$$

since $x ^ { C \cup ( V \setminus V ( a ) ^ { + } ) } \in X _ { G F }$ . This implies that $V ( a ) ^ { - } = \emptyset$ . Otherwise, by (33), we would get an equality $x _ { v } = 0$ for every $v \in V ( a ) ^ { - }$ that holds for all feasible vectors x that satisfy (32) at equality. Now, inequality (32) takes the form

$$
- a _ { f } ( 1 - x _ { f } ) \leq \sum _ { v \in V ( a ) ^ { + } } a _ { v } ( 1 - x _ { v } ) .\tag{34}
$$

As in the previous case, we assign a weight $a _ { v }$ to each vertex $v \in V$ . Note that the minimum weight of an $f -$ connector of G is precisely $- \boldsymbol { a } _ { f }$ by (33) and the fact that inequality (34) is facetdefining. Then, by the Max-Potential Min-Work Theorem (Dufin, 1962; Schrijver, 2003), there exist nonnegative integers $\lambda _ { S }$ for all f-separators $S$ of $G$ such that

$$
\sum _ { S \in f \mathrm { - s e p a r a t o r s } ( G ) } \lambda _ { S } = - a _ { f }
$$

and

$$
\sum _ { S \in f \mathrm { - s e p a r a t o r s } ( G ) } \lambda _ { S } \mathbb { 1 } _ { S } \leq a | _ { V } .
$$

Again, inequalities between vectors are componentwise. Therefore, we have

$$
\begin{array} { l } { \displaystyle { a _ { f } \big ( 1 - x _ { f } \big ) + \sum _ { v \in V ( a ) ^ { + } } a _ { v } ( 1 - x _ { v } ) } } \\ { = \displaystyle { \sum _ { S \in f \mathrm { \mathrm { \mathrm { - s e p a r a t o r s } } } ( G ) } \lambda _ { S } \left( - 1 + x _ { f } + \sum _ { v \in S } ( 1 - x _ { v } ) \right) + \left( a \big | _ { V } - \sum _ { S \in f \mathrm { \mathrm { - s e p a r a t o r s } } ( G ) } \lambda _ { S } \mathbb { 1 } _ { S } \right) \cdot \big ( \mathbb { 1 } _ { V } - x \big | _ { V } \big ) } \ . } \end{array}
$$

That is, (34) is a nonnegative linear combination of separator inequalities associated with the interaction $f$ and f-separators of $G ,$ and box inequalities $x _ { v } \le 1$ associated with vertices $v \in V$ Thus, we arrive at the conclusion that (32) must be a separator inequality under the given assumption. This completes the proof. □

## D.5 Proof of Proposition 4

The decomposition of $\Xi _ { G F }$ follows immediately from the fact that interactions in $F ( \varnothing )$ are always labeled 1 by feasible vectors and the label of any interaction not in $F ( \varnothing )$ is completely determined by the connected component containing its endvertices. The remaining statement is just a basic property of polytope product. □

## D.6 Proof of Proposition 5

To see the first statement, let $f = \{ u , v \} \in F$ , then $\{ v \}$ is an f-separator of G and the associated separator inequality reads $x _ { v } \le x _ { f }$ , which implies $x _ { f } \geq 0$ under the assumption that $x _ { v } \geq 0$ . As for the second statement, let $\{ u , v \}$ be an edge of $G$ incident with $v ,$ then $x _ { v } \geq 0$ follows from the connector inequality $x _ { u v } \le x _ { u } + x _ { v }$ and the separator inequality $x _ { u } \le x _ { u v }$ , and $x _ { v } \le 1$ follows from the separator inequality $x _ { v } \le x _ { u v }$ and our assumption that $x _ { u v } \le 1$ □

## D.7 Proof of Theorem 1

Let $X ^ { \prime } = \{ x \in X _ { G F } \mid x _ { v } = 0 \}$ denote the set of all feasible vectors satisfying the given lower box inequality $x _ { v } \geq 0$ at equality and let $\mathcal { V } = \{ U \subseteq V \mid v \in U \}$ denote the set of all vertex subsets of G containing v. Clearly, every feasible vector $x \in X ^ { \prime }$ can be written as $x = x ^ { U }$ for some $U \in \mathcal { V }$ and vice versa.

Necessity. In order to prove necessity, we consider the violation of condition (i) and condition (ii) in turn and, for either case, construct a linear equality in $\mathbb { R } ^ { V \cup F }$ that is independent of $x _ { v } = 0$ but satisfied by all $x \in X ^ { \prime }$ . Since $\Xi _ { G F }$ is full-dimensional by Proposition 2, this implies that $x _ { v } \geq 0$ is not facet-defining in the case under consideration.

If condition (i) is violated, then there exists an adjacent vertex u of v in G such that $\{ u , v \} \in F$ In this case every $x \in X ^ { \prime }$ satisfies the equality $x _ { \{ u , v \} } = x _ { u }$ . To see this, we first observe that the inequality $x _ { u } \le x _ { \{ u , v \} }$ is valid, since it is just the separator inequality associated with the interaction $\{ u , v \} \in F$ and the uv-cut-vertex u of G. The other direction follows from the condition $x _ { v } = 0$ and the connector inequality $x _ { \{ u , v \} } \leq x _ { u } + x _ { \iota }$ associated with the interaction $\{ u , v \} \in F$ and the uv-connector $\{ u , v \}$ of G, since $\{ u , v \} \in E \cap F$

If condition (ii) is violated, then there exist two vertices u and w in V such that $\{ u , v \} \in E$ and $\{ v , w \} , \{ u , w \} \in F$ and u is a vw-cut-vertex of G. In this case we claim that the equality $x _ { v w } = x _ { u w }$ holds for all $x \in X ^ { \prime }$ by the following argument. Let x be an arbitrary feasible vector in X<sup>′</sup> of the form $x = x ^ { U }$ for some vertex subset $U \in \mathcal { V } . \ \mathrm { I f } \ x _ { u w } = 0 , \mathrm { i . e . , } \ U$ is a uw-connector of $G ,$ then U is also a vw-connector of G by the assumption that $\{ u , v \} \in E$ and the fact that $v \in U$ This means that $x _ { v w } = 0$ , and hence establishes the inequality $x _ { v w } \leq x _ { u w }$ for any $x \in X ^ { \prime }$ . For the other direction, we suppose that $x _ { v w } = 0 , \mathrm { i . e . , } U$ is a vw-connector of G. Then, since u is assumed to be a vw-cut-vertex of $G ,$ we see that U is also a uw-connector of $G \mathrm { , ~ i . e . , } x _ { u w } = 0$ . This yields the inequality $x _ { u w } \leq x _ { v w }$ for any $x \in X ^ { \prime }$ , as desired.

Suficiency. Assume conditions (i) and (ii) are satisfied. In the following, we show that for every vertex except v, and for every interaction, the corresponding standard unit vector is in the linear space lin $( \bar { X ^ { \prime } } - X ^ { \prime } )$ associated with the face under consideration.

For any vertex $v ^ { \prime }$ that is not v, set $A = \{ v \}$ and $B = \{ v , v ^ { \prime } \}$ . Then, by condition (i), we have $Q _ { A B } = \left\{ v ^ { \prime } \right\}$ . As both $\{ v \}$ and $\{ v , v ^ { \prime } \}$ belong to $\nu _ { \mathrm { : } }$ , it follows from (22) of Lemma 7 that

$$
\mathbb { 1 } _ { \{ v ^ { \prime } \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) .
$$

That is, all the standard unit vectors associated with vertices diferent from v are in lin $( X ^ { \prime } - X ^ { \prime } )$ For any interaction $f = \{ u , w \} \in F$ , we consider the following possible cases depending on the relation between v and minimal f-connectors of G.

(A) Suppose that there exists an f-path $P = ( V _ { P } , E _ { P } )$ in G such that v is an internal vertex of $P .$ . In this case, we set $A = V _ { P } \setminus \{ w \}$ and $B = V _ { P } \setminus \{ u \}$ . Then, $A \setminus B = \{ u \} , B \setminus A = \{ w \}$

![](images/1c95f882b86f4b985526d22c98485de56dff6667d3db1323a9c99a645902a9e5.jpg)  
Figure 12: Depicted above are graphs G (black, solid), interactions (green, dashed) and a specified vertex v, which defines a lower box inequality $x _ { v } \ \geq 0$ . Both conditions (i) and (ii) are satisfied. Case (B) in Section D.7 applies to a), where the required minimal f-connector can be chosen as $C = \{ u , r , w \}$ . Case (C) in Section D.7 applies to b) and c), where the required minimal f-connectors can be chosen as $C = \{ u , r , w \}$ and $C = \{ u , u ^ { \prime } , w \}$ , respectively. Note that $u ^ { \prime }$ is the unique vertex of C that is adjacent to v in G in both b) and c). The set $F _ { A B }$ constructed in case (C) is {f} for b) and $\{ f , f ^ { \prime } \}$ for c).

$A \cup B = V _ { P }$ and A ∩ $B = V _ { P } \setminus \{ u , w \}$ , from which it follows easily that $F _ { A B } = \{ f \}$ and $H _ { A B } = \mathcal { O }$ Moreover, all A, $B , \ A \cap B$ , and $A \cup B$ belong to V. Therefore, by (23) of Lemma 7, we get $\mathbb { 1 } _ { \{ f \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$

(B) Suppose that case (A) does not hold, i.e., every f-path in G does not contain v as an internal vertex, but we can find a minimal f-connector C of G such that $v \not \in C$ and $G [ C \cup \{ v \} ]$ is not a path, i.e., such that v belongs to neither $N _ { G } ( u ) \setminus C$ nor $N _ { G } ( w ) \setminus C$ . An example is depicted in Figure 12 a). In this case we observe that $G [ C \cup \{ v \} ]$ cannot be a chordless cycle, otherwise $G [ f \cup \{ v \} ]$ would be an $f \mathrm { - p a t h }$ with v as its internal vertex, a contradiction. Set $A = ( C \cup \{ v \} ) \setminus \{ w \}$ and $B = ( C \cup \{ v \} ) \backslash \{ u \}$ . Then, $A \setminus B = \{ u \} , B \setminus A = \{ w \} , A \cup B = C \cup \{ v \}$ and $A \cap B = ( C \cup \{ v \} ) \setminus \{ u , w \}$ . Taking into account the preceding observation, we see that $F _ { A B } = \{ f \}$ and $H _ { A B } = \mathcal { O }$ . Moreover, all $A , B , A \cap B$ and A ∪ B belong to V. Therefore, it follows from (23) of Lemma 7 that $\mathbb { 1 } _ { \{ f \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$

(C) Suppose that neither case (A) nor case (B) is true. This implies that, for any minimal f-connector C of G, the induced subgraph $G [ C \cup \{ v \} ]$ is a path and v $\notin C \setminus f$ . In particular, v lies in the closed neighborhood of f in G. Let $u ^ { \prime }$ denote the unique vertex of C that is adjacent to v in G. Without loss of generality, we assume that w is the endvertex of $G [ C \cup \{ v \} ]$ that is diferent from v. $\mathrm { S o }$ , the other endvertex u of $f$ could be equal to either v or $u ^ { \prime } .$ , depending on whether v is in $C$ or not. See Figure 12 b) and c) for illustration. Clearly, $u ^ { \prime } \neq$ w when $u ^ { \prime } = u ,$ . On the other hand, if $u ^ { \prime } \neq u$ , then $v = u$ and $u ^ { \prime }$ is adjacent to v in $G [ C ]$ , and hence by condition (i) and the fact that $\{ u , w \} \in F$ , we again have $u ^ { \prime } \ne w . \mathrm { S o } .$ , no matter which is the case, $u ^ { \prime }$ and w are always distinct vertices. Let us define $A = ( C \cup \{ v \} ) \setminus \{ w \}$ and $B = ( C \cup \{ v \} ) \cup \{ u ^ { \prime } \}$ . Then $A \setminus B = \{ u ^ { \prime } \}$ $B \setminus A = \{ w \} , A \cup B = C \cup \{ v \}$ and $A \cap B = ( C \cup \{ v \} ) \setminus \{ u ^ { \prime } , w \}$ . It is direct to check that $F _ { A B } = \{ u ^ { \prime } w _  \mathrm  $ , vw} ∩ F and $H _ { A B } = \mathcal { O }$ . Moreover, all the subsets A, $B , A \cap B$ and $A \cup B$ belong to V. Then, by (23) of Lemma $^ { 7 , }$ we obtain $\mathbb { I } _ { F _ { A B } } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$ . Now we observe that, if $F _ { A B } \setminus \{ f \}$ is non-empty, then it is a singleton and the unique interaction in it falls into case (A), from which we see that the standard unit vector corresponding to this unique interaction belongs to lin $( X ^ { \prime } - X ^ { \prime } )$ already by the previous discussion. In fact, let $f ^ { \prime }$ be the unique interaction in $\{ u ^ { \prime } w , v w \}$ that is diferent from $f .$ Since there exists a minimal vw-connector of G that does not contain u<sup>′</sup> by condition (ii), it follows from the assumption of case (C) that this can only happen when $u = v ,$ in which case $f ^ { \prime }$ coincides with $u ^ { \prime } w$ and it can be connected by a path in G that includes v as an internal vertex, which is exactly case (A). So, either $F _ { A B } \setminus \{ f \}$ is empty or it contains precisely one interaction satisfying case (A), as illustrated by b) and c) of Figure $1 2 ,$ respectively. In any case, it always holds that $\mathbb { 1 } _ { F _ { A B } \backslash \{ f \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$ . We thus get $\mathbb { 1 } _ { \{ f \} } = \mathbb { 1 } _ { F _ { A B } } - \mathbb { 1 } _ { F _ { A B } \backslash \{ f \} } \in \mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } )$ as $f \in F _ { A B }$

In summary, for every interaction in F we have constructed its corresponding standard unit vector in lin $( X ^ { \prime } - X ^ { \prime } )$

By all the results above, we see that the linear subspace lin $( X ^ { \prime } - X ^ { \prime } )$ has a basis $\{ \mathbb { 1 } _ { \{ v ^ { \prime } \} } \mid v ^ { \prime } \in$ $V \setminus \{ v \} \cup \{ \mathbb { 1 } _ { \{ f \} } | f \in F \}$ and hence is of codimension 1 in $\mathbb { R } ^ { V \cup \dot { F } }$ , and so is the face defined by $X ^ { \prime }$ Therefore, in view of Proposition $2 ,$ we conclude that the lower box inequality $x _ { v } \geq 0$ defines a facet of the multi-separator polytope $\Xi _ { G F }$ under the given conditions, and the proof is complete.

## D.8 Proof of Theorem 2

Let $X ^ { \prime } = \{ x \in X _ { G F } \mid x _ { f } = 1 \}$ denote the set of all feasible vectors satisfying the upper box inequality $x _ { f } \le 1$ at equality and let $\mathcal { V } = \{ U \subseteq V \mid V \setminus U$ is an f-separator of $G \}$ denote the set of all vertex subsets of G that are not f-connectors of G. Clearly, every vector x in $X ^ { \prime }$ has the form $x = x ^ { U }$ for some vertex subset U in V and vice versa.

Necessity. Suppose to the contrary that there is an interaction $f ^ { \prime } \in F \setminus \{ f \}$ such that $f$ is a pair of f<sup>′</sup>-cut-vertices of G. This implies that every f<sup>′</sup>-connector of G is also an f-connector of $G ,$ and thus the inequality $x _ { f } \leq x _ { f ^ { \prime } }$ is valid for $\Xi _ { G F }$ . As $x _ { f ^ { \prime } } \leq 1$ is also valid, we immediately see that the equality $x _ { f ^ { \prime } } = 1$ holds for all $x \in X ^ { \prime }$ . Therefore, the face defined by $X ^ { \prime }$ is not a facet of $\Xi _ { G F }$ by Proposition 2.

Suficiency. Assume that f is not a pair of f<sup>′</sup>-cut-vertices of G for all other interactions $f ^ { \prime }$ . We show that the given inequality defines a facet by constructing a basis of cardinality $\vert V \vert + \vert F \vert - 1$ of the associated linear space lin $( X ^ { \prime } - X ^ { \prime } )$

Firstly, for any vertex $v \in V .$ , it is clear that $\emptyset , \{ v \} \in \mathcal { V }$ and $Q _ { \mathcal { O } \{ v \} } = \{ v \}$ . Then it follows straightforwardly from (22) of Lemma 7 that $\mathbb { 1 } _ { \{ v \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$

Secondly, we consider any interaction $f ^ { \prime } = \dot { \left\{ u , w \right\} } \in F \setminus \left\{ f \right\}$ . By assumption, we can find a minimal $f ^ { \prime } .$ -connector C of G that contains at most one of the endvertices of $f .$ Set $A = C \setminus \{ w \}$ and $B = C \setminus \{ u \}$ . Then, $F _ { A B } = \{ f ^ { \prime } \}$ and $H _ { A B } = \mathcal { O }$ . It is clear that all A, $B , A \cap B$ and A ∪ B belong to V. Thus, it follows from (23) of Lemma 7 that $\mathbb { 1 } _ { \{ f ^ { \prime } \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$

Therefore, lin $( X ^ { \prime } - X ^ { \prime } )$ has codimension 1 in $\mathbb { R } ^ { V \cup F }$ and the face defined by $X ^ { \prime }$ is a facet of $\Xi _ { G F }$ by Proposition 2. □

## D.9 Proof of Theorem 3

Let $X ^ { \prime } = \{ x \in X _ { G F } \mid x _ { v } = 1 \}$ denote the set of all feasible vectors satisfying the upper box inequality $x _ { v } \leq 1$ at equality and let $\mathcal { V } = \{ U \subseteq V \mid v \not \in U \}$ denote the set of all vertex subsets of G not containing v. Clearly, the elements in V are exactly those vertex subsets U of G satisfying $x ^ { U } \in X ^ { \prime }$

Necessity. Suppose there exists an interaction $f \in F$ such that v is an $f .$ -cut-vertex of $G .$ Then, by the separator inequality $x _ { v } \le x _ { f }$ and the box inequality $x _ { f } \le 1$ , we see that the equality $x _ { f } = 1$ holds for all $x \in X ^ { \prime }$ . This implies that the face defined by $X ^ { \prime }$ is not a facet of $\Xi _ { G F }$ by Proposition 2.

Suficiency. Assume that there is no interaction with v as its cut-vertex. To establish the dimension of the face defined by $X ^ { \prime }$ , we apply the technique from Lemma 7 to construct a basis for the associated linear space lin $( X ^ { \prime } - X ^ { \prime } )$ as follows.

Firstly, for any vertex $u \in V \setminus \{ v \}$ , it is clear that $\emptyset , \{ u \} \in \mathcal { V }$ and $Q _ { \mathcal { O } \{ u \} } = \{ u \}$ . Thus $\mathbb { 1 } _ { \{ u \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$ by (22) of Lemma 7.

Secondly, let us consider an arbitrary interaction $f = \{ u , w \} \in F$ . By assumption, we may choose a minimal f-connector C of $G$ such that it does not pass through v. Set $A = C \setminus \{ u \}$ and $B = C \setminus \{ w \}$ . Then $F _ { A B } = \{ f \}$ and $H _ { A B } = \mathcal { O }$ . Since all A, B, A ∩ B and $A \cup B$ belong to $\nu ,$ it follows from (23) immediately that $\mathbb { 1 } _ { \{ f \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$

Therefore, we have shown that lin $( X ^ { \prime } - X ^ { \prime } )$ has codimension 1 in $\mathbb { R } ^ { V \cup F }$ . Since $\Xi _ { G F }$ is full-dimensional by Proposition 2, the proof follows immediately. □

## D.10 Proof of Theorem 4

Let

$$
X ^ { \prime } = \left\{ x \in X _ { G F } \middle | x _ { f } = \sum _ { v \in C } x _ { v } \right\}
$$

denote the set of all feasible vectors that satisfy connector inequality (4) associated with f and C at equality. To make the structure of $X ^ { \prime }$ more explicit, we define

$\mathcal { V } = \{ U \subseteq V \mid$ either $C \subseteq U ,$ or U is not an f-connector of G and $C \setminus U$ is a singleton}

as the set of all vertex subsets of G whose complement in V is either disjoint from C, or an f-separator of G intersecting $C$ at a single vertex. Then, a feasible vector x is in $X ^ { \prime }$ if and

only if there is an element $U$ of V such that $x = x ^ { U }$ . For the convenience of studying connector inequality (4), given any neighbor v of $C$ , we denote $\mathrm { B P } _ { v }$ as the set consisting of v and all vertices $c \in C$ such that v is a c-bypass of $C$

Necessity. Since the multi-separator polytope $\Xi _ { G F }$ is full-dimensional by Proposition 2 and non-minimal connector inequalities are implied by other canonical inequalities by Lemma 1, it follows that condition (i) is necessary for the face defined by $X ^ { \prime }$ to be a facet of $\Xi _ { G F }$

${ \mathrm { S o } } ,$ without loss of generality, in the remainder of the necessity proof we assume that condition (i) always holds, $\mathrm { i . e . , } C$ is a minimal f-connector of G.

Suppose that condition (ii) is not satisfied. Then, there is an interaction f<sup>′</sup> that is diferent from f and contained in C. Let $C ^ { \prime }$ be the unique minimal $f ^ { \prime } .$ -connector of $G [ C ]$ . As the union of an $f ^ { \prime } .$ -connector of G and $C \setminus C ^ { \prime }$ is an f-connector of $G ,$ we see that the inequality

$$
x _ { f } \leq x _ { f ^ { \prime } } + \sum _ { v \in C \setminus C ^ { \prime } } x _ { v }\tag{35}
$$

is valid for $\Xi _ { G F }$ . Thus, the connector inequality (4) can be written as a sum of two valid inequalities, $\mathrm { i . e . }$ , inequality (35) and the connector inequality associated with $f ^ { \prime }$ and $C ^ { \prime }$ . This contradicts the assumption that the convex hull of $X ^ { \prime }$ is a facet of $\Xi _ { G F }$ and the fact that $\Xi _ { G F }$ is full-dimensional by Proposition 2.

Suppose that condition (iii) does not hold; that is, there is an interaction uw $\in \ F _ { }$ with $u \in N _ { G } ( C )$ and $w \in C$ such that u is a w-bypass of C. Observe that the inequality $x _ { u } \le x _ { u w }$ is valid as it is just a separator inequality. We show that every $x \in X ^ { \prime }$ with $x _ { u } = 0$ also satisfies $x _ { u w } = 0$ , from which it follows that the equality $x _ { u } ~ = ~ x _ { u w }$ holds for all $x \in X ^ { \prime }$ , and thus the connector inequality cannot be facet-defining by Proposition $2 . \ \mathrm { \textrm { S u } }$ ppose $x _ { u } = 0$ for some $x = x ^ { U } \in X ^ { \prime }$ with $U \in \mathcal V$ . Then we argue that $x _ { v ^ { \prime } } = 0$ holds for all $v ^ { \prime } \in \mathrm { B P } _ { u }$ . Since $\mathrm { B P } _ { u }$ together with $N _ { G } ( u ) \cap C \cap U$ , which is nonempty by definitions of u and $U$ , forms a uw-connector of G, this yields $x _ { u w } = 0$ . In fact, if $x _ { v ^ { \prime } } = 1$ for some vertex $v ^ { \prime } \in \mathrm { B P } _ { u } .$ , as $x \in X ^ { \prime }$ , it follows that $x _ { f } = 1$ and $x _ { v ^ { \prime \prime } } = 0$ for all $v ^ { \prime \prime } \in C \setminus \{ v ^ { \prime } \}$ , which implies that $( C \cup \{ u \} ) \setminus \{ v ^ { \prime } \}$ cannot be an f-connector of $G ,$ contradicting the fact that u is a $ { v ^ { \prime } }  { - }  { \mathrm { b y p a s s } }$ of $C$

Suppose that condition (iv) does not hold; that is, there are two interactions uw, $u w ^ { \prime } \in F$ with u $\notin N _ { G } [ C ] .$ , w $\in { \cal N } _ { G } ( C )$ and $w ^ { \prime } \in C$ such that w is both a {{u}, C}-cut-vertex of G and a $w ^ { \prime } \mathrm { - b y }$ pass of C. If $x _ { u w ^ { \prime } } = 0$ for some x $\in { \cal { X } } _ { G F }$ , then it is also true that $x _ { u w } = 0$ as w is a $\{ \{ u \} , C \}$ -cut-vertex of G. This implies that the inequality $x _ { u w } \leq x _ { u w ^ { \prime } }$ is valid. $\mathrm { ~ I f ~ } x _ { u w } = 0$ for some $x \in X ^ { \prime }$ , just as in the previous paragraph, then we have $x _ { v ^ { \prime } } = 0$ for all $v ^ { \prime } \in \mathrm { B P } _ { w }$ , which yields $x _ { u w ^ { \prime } } = 0$ , and hence the inequality $x _ { u w } \geq x _ { u w ^ { \prime } }$ . Therefore, the equality $x _ { u w } = x _ { u w ^ { \prime } }$ holds for all $x \in X ^ { \prime }$ , which again leads to a contradiction by the same argument as above.

Suppose that condition (v) does not hold; that is, there are two distinct interactions uw, $u w ^ { \prime } \in F$ with u $, \notin N _ { G } [ C ]$ and w, $w ^ { \prime } \in C$ such that the set

$$
W = \{ v \in N _ { G } ( C ) \mid v { \mathrm { ~ i s ~ b o t h ~ a ~ } } w { \mathrm { - b y p a s s ~ a n d ~ a ~ } } w ^ { \prime } { \mathrm { - b y p a s s ~ o f ~ } } C \}
$$

is a $\{ \{ u \} , C \}$ -separator of G. If $x _ { u w } = 0$ for some $x \in X ^ { \prime }$ , then there exists a minimal uw-connector $C ^ { \prime }$ of G such that $x _ { r } = 0$ for all $r \in C ^ { \prime }$ . By definition, $C ^ { \prime }$ is also a $\{ \{ u \} , C \}$ -connector of G and therefore must intersect W, which implies that there exists a w- and w<sup>′</sup>-bypass $v \in N _ { G } ( C )$ of $C$ satisfying $x _ { v } = 0$ . As in the previous paragraph, by considering $x _ { v } = 0$ , we get $x _ { v ^ { \prime } } = 0$ for all $\boldsymbol { v } ^ { \prime } \in \mathrm { B P } _ { v }$ . This implies that $x _ { u w ^ { \prime } } = 0$ , since $C ^ { \prime }$ together with $\mathrm { B P } _ { v }$ forms a uw<sup>′</sup>-connector of $G$ and hence the inequality $x _ { u w } \geq x _ { u w ^ { \prime } }$ . A symmetric argument yields that, if $x _ { u w ^ { \prime } } = 0$ for some $x \in X ^ { \prime }$ , then it holds that $x _ { u w } = 0$ , and hence the inequality $x _ { u w } \leq x _ { u w ^ { \prime } }$ . Therefore, the equality $x _ { u w } = x _ { u w ^ { \prime } }$ holds for all $x \in X ^ { \prime }$ , which again leads to a contradiction as before.

Suficiency. In order to show that connector inequality (4) defines a facet, we construct a basis of cardinality $\vert V \vert + \vert F \vert - 1$ of its associated linear space lin $( X ^ { \prime } - X ^ { \prime } )$ by making use of the following results.

Claim 1. Assume that C is a minimal f-connector of $G _ { i }$ , let $v \in N _ { G } ( C )$ be a neighbor of C.

(a) It holds that

$$
\mathbb { 1 } _ { \{ v \} \cup \{ v w \in F | w \in \mathrm { B P } _ { v } \} } \in \mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } ) ,\tag{36}
$$

and, for any interaction vw $\in F$ with $w \in C \setminus \mathrm { B P } _ { v }$ ，

$$
\mathbb { 1 } _ { \{ v w \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) .\tag{37}
$$

(b) For any vertex u that is not in $N _ { G } [ C ]$ , and such that there exists a minimal $\{ \{ u \} , N _ { G } [ C ] \}$ connector of $G$ containing $v ,$ it holds that

$$
\mathbb { 1 } _ { \{ u w \in F | w \in \mathrm { B P } _ { v } \} } \in \mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } ) ,\tag{38}
$$

and, for any interaction uw $\in F$ with $w \in C \setminus \mathrm { B P } _ { v }$

$$
\mathbb { 1 } _ { \{ u w \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) .\tag{39}
$$

Proof of claim. As we shall see below, the constructions of the characteristic vectors in case (a) and case (b) can be done completely analogously. To this end, we assume that u is a vertex that is not in $C$ and v is a neighbor of $C$ that is on a minimal $\{ \{ u \} , N _ { G } [ C ] \}$ -connector of G. So, if u is a neighbor of $C ,$ then $u = v ,$ which is precisely case (a). Conversely, if u is not a neighbor of $C _ { i }$ , then u $\neq v ,$ , which is precisely case (b).

The remainder of this proof is structured as follows. Initially, we construct some auxiliary vectors in lin $( X ^ { \prime } - X ^ { \prime } )$ which are characteristic vectors of specific sets of interactions that are incident with u plus possibly u itself. Subsequently, we show that the characteristic vectors in the statement of the claim can be obtained as linear combinations of these auxiliary vectors and thus also belong to lin $( X ^ { \prime } - X ^ { \prime } )$

For our purposes here, let $\tilde { C }$ denote an arbitrary minimal uv-connector of G with ${ \tilde { C } } \cap N _ { G } ( C ) =$ $\{ v \}$ , which indeed exists by definition of $v ,$ and write $U = C \cup { \tilde { C } }$ . We define

$$
\begin{array} { c } { { A = U \setminus \left\{ u \right\} , } } \\ { { } } \\ { { B = \left( U \setminus \left\{ v \right\} \right) \cup \left\{ u \right\} , } } \end{array}
$$

and

$$
B ( t ) = U \setminus \{ t \}
$$

for every $t \in C \setminus \mathrm { B P } _ { v } .$ i.e., v is not a t-bypass of $C .$ Then, by construction,

$$
\begin{array} { r } { A \setminus B = \{ v \} \setminus \{ u \} , \quad B \setminus A = \{ u \} , \quad A \cap B = U \setminus \{ u , v \} , \quad A \cup B = U , } \end{array}
$$

and

$$
A \setminus B ( t ) = \{ t \} , \quad B ( t ) \setminus A = \{ u \} , \quad A \cap B ( t ) = U \setminus \{ u , t \} , \quad A \cup B ( t ) = U .
$$

Looking at the definition of a bypass vertex, we see that for all $t \in C \setminus \mathrm { B P } _ { v } ,$ neither $B ( t )$ nor $A \cap B ( t )$ is an f-connector of $G .$ . Moreover, $C \setminus B ( t ) = C \setminus ( A \cap B ( t ) ) = \{ t \}$ . Also note that $A , B ,$ $A \cap B , A \cup B$ and $A \cup B ( t )$ all contain the f-connector C. In view of the definition of $\nu ,$ we arrive at the following observation:

$$
A , B , B ( t ) , A \cap B , A \cup B , A \cap B ( t ) , A \cup B ( t ) \in \mathcal { V } .\tag{40}
$$

It follows directly from the definitions of $Q _ { A B }$ and $F _ { A B }$ in Lemma $7$ that

$$
Q _ { A B } = \left( B \setminus A \right) \cup \left\{ f ^ { \prime } \in F \mid f ^ { \prime } \subseteq U , \ u { \mathrm { i s ~ a n ~ } } f ^ { \prime } { \mathrm { c u t - v e r t e x ~ o f ~ } } G [ U ] \right\} = \left\{ v \right\} \cup \left\{ v v ^ { \prime } \in F \mid v ^ { \prime } \in C \right\}
$$

if $u = v$ (note that in this case we have $A = C$ and $B = C \cup \{ v \} )$ , and

$$
{ \cal F } _ { A B } = \{ f ^ { \prime } \in { \cal F } \mid f ^ { \prime } \subseteq { \cal U } , \ u \mathrm { ~ a n d ~ } v \mathrm { ~ a r e ~ } f ^ { \prime } \mathrm { - c u t - v e r t i c e s ~ o f ~ } G [ { \cal U } ] \} = \{ u v ^ { \prime } \in { \cal F } \mid v ^ { \prime } \in { \cal C } \cup \{ v \} \}
$$

otherwise. Similarly, for every $t \in C \setminus \mathrm { B P } _ { v }$ , we have

$F _ { A B \left( t \right) } = \left\{ f ^ { \prime } \in F \mid f ^ { \prime } \subseteq U , \right.$ u and t are f<sup>′</sup>-cut-vertices of $G [ U ] \} = \{ u v ^ { \prime } \in F \mid v ^ { \prime } \in C ( v , t ) \}$

where $C ( v , t )$ is the set of all vertices $v ^ { \prime }$ in $C$ such that t is a vv<sup>′</sup>-cut-vertex of $G [ C \cup \{ v \} ]$ . By the definition of $H _ { A B }$ in Lemma $^ { 7 , }$ we have

$$
H _ { A B } = \mathcal { O }
$$

if $u = v ,$ and

$$
H _ { A B } = \{ f ^ { \prime } \in F \mid f ^ { \prime } \subseteq U \setminus \{ u , v \} , \{ u , v \} { \mathrm { ~ i s ~ a ~ m i n i m a l ~ } } f ^ { \prime } { \mathrm { - s e p a r a t o r ~ o f ~ } } G [ U ] \} = \emptyset
$$

otherwise, upon noting that $\{ u , v \}$ is a minimal f<sup>′</sup>-separator of $G [ U ]$ if and only if $U \backslash \{ u \}$ and $U \backslash \{ v \}$ are $f ^ { \prime } { } .$ connectors of G but not $U \setminus \{ u , v \}$ . Similarly, for every $t \in C \setminus \mathrm { B P } _ { v } ,$ we have

$$
H _ { A B ( t ) } = \left\{ f ^ { \prime } \in F \mid f ^ { \prime } \subseteq U \setminus \{ u , t \} , \{ u , t \} { \mathrm { ~ i s ~ a ~ m i n i m a l ~ } } f ^ { \prime } { \mathrm { - s e p a r a t o r ~ o f ~ } } G [ U ] \right\} = \emptyset .
$$

Thus, it follows from (40) and (22) of Lemma 7 that

$$
\mathbb { 1 } _ { Q _ { A B } } = \mathbb { 1 } _ { \{ v \} } + \mathbb { 1 } _ { \{ v v ^ { \prime } \in F | v ^ { \prime } \in C \} } \in \mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } )\tag{41}
$$

when $u = v$ . In addition, by (40) and (23) of Lemma $^ { 7 , }$ we have

$$
\mathbb { 1 } _ { F _ { A B } } - \mathbb { 1 } _ { H _ { A B } } = \mathbb { 1 } _ { \{ u v ^ { \prime } \in F | v ^ { \prime } \in C \cup \{ v \} \} } \in \ln ( X ^ { \prime } - X ^ { \prime } )\tag{42}
$$

when $u \ne v$ , and

$$
\mathbb { 1 } _ { F _ { A B ( t ) } } - \mathbb { 1 } _ { H _ { A B ( t ) } } = \mathbb { 1 } _ { \{ u v ^ { \prime } \in F | v ^ { \prime } \in C ( v , t ) \} } \in \mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } )\tag{43}
$$

for all $t \in C \setminus \mathrm { B P } _ { v } .$

To understand $C ( v , t )$ more specifically, let us fix a natural linear order $c _ { 0 } < \cdots < c _ { \alpha } \leq$ $\dots \leq c _ { \beta } < \dots < c _ { n }$ on $C , \mathrm { ~ i . e . , ~ } c _ { i - 1 } c _ { i } \in E$ for $i = 1 , \ldots , n$ , where α (respectively, $\beta )$ is the minimal (respectively, maximal) index such that $c _ { \alpha }$ (respectively, $c _ { \beta } )$ is adjacent to v in G. Then, $\mathrm { B P } _ { v } \cap C = \{ c _ { \alpha + 1 } , \dotsc , c _ { \beta - 1 } \}$ . Note that α and β could coincide, which can only happen in the case that v is not a bypass of C. Let $t = c _ { h }$ be a vertex of C that is not in $\mathrm { B P } _ { v } , \mathrm { i . e . , } h \le \alpha$ or $h \geq \beta$ . If $\alpha = \beta = h$ , then $C ( v , t ) = C$ . Otherwise, $C ( v , t ) = \{ c _ { k } \in C \mid k \le h \}$ if $h \leq \alpha .$ , and $C ( v , t ) = \{ c _ { k } \in C \mid k \geq h \} { \mathrm { ~ i f ~ } } h \geq \beta .$

If we generalize our notation a little bit and define uv to be v whenever $u = v ,$ then (41), (42) and (43) together imply that

$$
\mathbb { 1 } _ { \{ u w \in N _ { G } ( C ) \cup F | w \in \mathrm { B P } _ { v } \} } = \mathbb { 1 } _ { \{ u v ^ { \prime } \in N _ { G } ( C ) \cup F | v ^ { \prime } \in C \cup \{ v \} \} } - \sum _ { t \in \{ c _ { \alpha } , c _ { \beta } \} } \mathbb { 1 } _ { F _ { A B ( t ) } } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) ,\tag{44}
$$

which yields (36) if $u = v ,$ , and (38) if $u \ne v$ . For the case u $\neq v ,$ examples of the construction above are depicted in Figure 13 and Figure 14 for $\alpha \neq \beta$ and $\alpha = \beta$ , respectively. (As for the case $u = v$ , in each of Figure 13 and Figure 14, the only diference would be that the path between u and v should be contracted to a single vertex, namely $u = v$ , and then the interaction uv reduces to the vertex $v . )$

It remains to establish (37) and (39). Suppose uw is an interaction with $w \in C \setminus \mathrm { B P } _ { v }$ . Let h be the index of $w , { \mathrm { i . e . , } } w = c _ { h }$ so that $h \leq \alpha$ or $h \geq \beta$ . We distinguish two possible cases.

(A) w is an endvertex of the path $G [ C ]$ , i.e., either $h = 0$ or $h = n$ . Let $w ^ { \prime } \in C$ be the unique adjacent vertex of w on C. More explicitly, $w ^ { \prime } = c _ { 1 }$ if $w = c _ { 0 }$ , and $w ^ { \prime } = c _ { n - 1 }$ if $w = c _ { n }$ . By (43), we obtain

$$
\mathbb { 1 } _ { \{ u w \} } = \mathbb { 1 } _ { F _ { A B ( w ) } } - \mathbb { 1 } _ { F _ { A B ( w ^ { \prime } ) } } \in \mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } )\tag{45}
$$

$$
{ \mathrm { i f ~ } } \alpha = \beta = h , { \mathrm { a n d } }
$$

$$
\mathbb { 1 } _ { \{ u w \} } = \mathbb { 1 } _ { F _ { A B ( w ) } } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )
$$

otherwise.

(46)

![](images/86a9fc98758815f25383356916fd9473d381e0f9b680cec89e56607c05166305.jpg)  
Figure 13: This figure illustrates the construction of the characteristic vector in (44) for α ̸= β. Consider the graph $G = ( V , E )$ depicted above in solid black and the f-connector $C = \{ c _ { 0 } , \ldots , c _ { 5 } \}$ for $f = c _ { 0 } c _ { 5 }$ . Here it holds that $\alpha = 1$ and $\beta = 4$ . Suppose that the set of interactions is $F = { \binom { V } { 2 } }$ . Depicted in a), b), and c) in dashed green are the characteristic vectors $\mathbb { 1 } _ { F _ { A B } } , \mathbb { 1 } _ { F _ { A B ( c _ { \alpha } ) } }$ , and $\mathbb { 1 } _ { F _ { A B ( c _ { \beta } ) } : }$ , respectively. Depicted in d) in dashed green is the characteristic vector of the interaction set $\{ u w \in F \mid w \in \mathrm { B P } _ { v } \}$ that is constructed in (44).

![](images/c0b9a96475c73475ff89331ded3fadcb6a93f28eaeebd96172ff6e55ad17b360.jpg)  
a)  
b)  
c)  
Figure 14: This figure illustrates the construction of the characteristic vector in (44) for $\alpha = \beta .$ . Consider the graph $G = ( V , E )$ depicted above in solid black and the f-connector $C = \{ c _ { 0 } , c _ { 1 } , c _ { 2 } \}$ for $f = c _ { 0 } c _ { 2 }$ . Here it holds that $\alpha = \beta = 1$ . Suppose that the set of interactions is $F = { \binom { V } { 2 } }$ . Depicted in a) and b) in dashed green are the characteristic vectors $\mathbb { 1 } _ { F _ { A B } }$ and $\mathbb { 1 } _ { F _ { A B ( c _ { \alpha } ) } }$ , respectively. Depicted in c) in dashed green is the characteristic vector of the interaction set $\{ u w \in \beth \} \stackrel { -  } { w } \in \mathrm { B P } _ { v } \}$ that is constructed in (44). Notice that we have $\mathrm { B P } _ { v } = \{ v \}$ as v is not a C-bypass.

(B) w is an interior vertex of the path $G [ C ]$ , i.e., $h \in \{ 1 , \ldots , n - 1 \} \setminus \{ \alpha + 1 , \ldots , \beta - 1 \}$ . If $\alpha = \beta = h$ , by (43), we get

$$
\mathbb { 1 } _ { \{ u w \} } = \mathbb { 1 } _ { F _ { A B ( w ) } } - \mathbb { 1 } _ { F _ { A B ( c _ { h - 1 } ) } } - \mathbb { 1 } _ { F _ { A B ( c _ { h + 1 } ) } } \in \mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } ) .\tag{47}
$$

Otherwise, let $w ^ { \prime } \in C \setminus \mathrm { B P } _ { v }$ denote the unique vertex that is adjacent to w, and such that w is a vw<sup>′</sup>-cut-vertex of $G [ C \cup \{ v \} ]$ . In other words, $w ^ { \prime } = c _ { h - 1 }$ if $1 \leq h \leq \alpha$ , and $w ^ { \prime } = c _ { h + 1 }$ if $\beta \leq h \leq n - 1$ . Hence, $\bar { C ( v , w ) \setminus C ( v , w ^ { \prime } ) } = \{ w \}$ . Then, by (43), we obtain

$$
\mathbb { 1 } _ { \{ u w \} } = \mathbb { 1 } _ { F _ { A B ( w ) } } - \mathbb { 1 } _ { F _ { A B ( w ^ { \prime } ) } } \in \mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } ) .\tag{48}
$$

An example is depicted in Figure 15.

In both case (A) and case (B), it holds that $\mathbb { 1 } _ { \{ u w \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$ , which yields (37) if $u = v$ , and (39) if u ̸= v. This completes the proof of the claim. Q.E.D.

We are now ready to delve into the construction of a basis of $\operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$ under conditions $( \mathrm { i } ) \textrm { - }$ (v) of Theorem 4.

First of all, by condition (i), C is a minimal f-connector of G. So Claim 1 applies.

Let $v \in V$ . If v is not a vertex in the closed neighborhood of C, let $A = C$ and $B = C \cup \{ v \}$ then $Q _ { A B } = \{ v \}$ . Thus, by (22) of Lemma 7, we get

$$
\mathbb { 1 } _ { \{ v \} } \in \mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } ) ,
$$

since both C and $C \cup \{ v \}$ are in V. If v is a vertex in C, we set $A = C$ and $B = C \setminus \{ v \}$ . Then $Q _ { A B } = \{ v , f \}$ . Indeed, $A \triangle B = \{ v \}$ , and by condition (ii), the only interaction in F contained in

![](images/b7a5d84b909c11c8e92c30dea23a80ad2accd9cae90ab3645351abc86e4be5c1.jpg)  
Figure 15: This figure illustrates the construction of the characteristic vector in (48) of case (B). Consider the graph depicted above in solid black and the $f \cdot$ -connector $C = \{ c _ { 0 } , \ldots , c _ { 5 } \}$ for $f = c _ { 0 } c _ { 5 }$ . Let $w = c _ { 2 } \notin \mathrm { B P } _ { v }$ . Then the unique vertex $w ^ { \prime }$ defined in the proof of Claim 1 is $w ^ { \prime } = c _ { 1 }$ . Suppose that the set of interactions is $F = { \binom { V } { 2 } }$ . Depicted in a) and b) in dashed green are the characteristic vectors $\mathbb { 1 } _ { F _ { A B ( w ) } }$ and $\mathbb { 1 } _ { F _ { A B ( w ^ { \prime } ) } }$ , respectively. Depicted in c) in dashed green is the characteristic vector $\mathbb { 1 } _ { \{ u w \} }$ that is constructed in (48).

C whose connectivity is destroyed by deleting v is $f .$ Therefore, it follows from (22) of Lemma 7 that

$$
\mathbb { 1 } _ { \{ v , f \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) ,
$$

since both C and $C \setminus \{ v \}$ are in V. For any vertex v in the open neighborhood of $C ,$ condition (iii) implies $\{ v w \in F \mid w \in \mathrm { B P } _ { v } \} = \emptyset$ . Then, by (36) of Claim 1, we get

$$
\mathbb { 1 } _ { \{ v \} } \in \mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } ) .
$$

${ \mathrm { S o } } ,$ for every vertex $v \in V \setminus C$ , its corresponding standard unit vector is in lin $( X ^ { \prime } - X ^ { \prime } )$

Now, we are left with constructing standard unit vectors associated with interactions that are diferent from f under the given conditions in the theorem. To achieve this, we partition the interactions $f ^ { \prime } = u w$ in $F \setminus \{ f \}$ into the following classes according to the relative position between C and their endvertices:

(A) $u , w \in C .$

(B) $u \in N _ { G } ( C )$ and $w \in C$

(C) $u , w \in N _ { G } ( C )$

(D) $u , w \in V \setminus N _ { G } [ C ] .$

(E) u $\sharp \ : N _ { G } [ C ] , w \in N _ { G } ( C )$ and w is not a $\{ \{ u \} , C \}$ -cut-vertex of G.

(F) u $\not \in N _ { G } [ C ] , w \in N _ { G } ( C )$ , and w is a $\{ \{ u \} , C \} _ { \mathrm { - c u t - v e r t e x ~ o f ~ } G }$

(G) $u \not \in N _ { G } [ C ] , w \in C .$

Note that these classes are exhaustive and disjoint, i.e., every interaction in $F \setminus \{ f \}$ lies in exactly one of these classes.

By condition (ii), there are no interactions in $F \setminus \{ f \}$ that satisfy (A).

Let $f ^ { \prime } = u w$ satisfy (B). By condition (iii), u cannot be a w-bypass of $C _ { i }$ i.e. $w \in C \setminus \mathrm { B P } _ { u }$ Then, it follows from (37) of Claim 1 that

$$
\mathbb { 1 } _ { \{ f ^ { \prime } \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) .
$$

Let $f ^ { \prime } = u w$ be an interaction that lies in $( \mathrm { C } ) , \mathrm { o r ~ ( D ) }$ , or (E). We consider two possible cases for $f ^ { \prime }$ . If there exists a minimal $f ^ { \prime } .$ -connector $C ^ { \prime }$ of G that is disjoint from $N _ { G } [ C ]$ , then we set

$$
A = ( C \cup C ^ { \prime } ) \setminus \{ u \} , \qquad B = ( C \cup C ^ { \prime } ) \setminus \{ w \} .
$$

Otherwise, we can choose a minimal $\{ \{ u \} , N _ { G } [ C ] \}$ -connector $C _ { 1 }$ of $G$ and a minimal $\{ \{ w \} , N _ { G } [ C ] \} .$ connector $C _ { 2 }$ of G such that w $\notin C _ { 1 }$ and $u \not \in C _ { 2 }$ . Set

$$
A = \left( C \cup C _ { 1 } \cup C _ { 2 } \right) \setminus \left\{ u \right\} , \qquad B = \left( C \cup C _ { 1 } \cup C _ { 2 } \right) \setminus \left\{ w \right\} .
$$

In each case, we see that

$$
{ \cal F } _ { A B } = \left\{ f ^ { \prime \prime } \in { \cal F } \mid f ^ { \prime \prime } \subseteq { \cal A } \cup { \cal B } , \ u \mathrm { ~ a n d ~ } w \mathrm { ~ a r e ~ } f ^ { \prime \prime } { \mathrm { - c u t - v e r t i c e s ~ o f ~ } } G [ { \cal A } \cup { \cal B } ] \right\} = \left\{ f ^ { \prime } \right\} ,
$$

and

$H _ { A B } = \{ f ^ { \prime \prime } \in F \mid f ^ { \prime \prime } \subseteq A \cap B , \{ u , w \}$ is a minimal $f ^ { \prime \prime } .$ -separator of $G [ A \cup B ] \} = \emptyset$

Then, by (23) of Lemma 7, we obtain

$$
\mathbb { 1 } _ { \{ f ^ { \prime } \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) ,
$$

since $A , B , A \cap B$ and $A \cup B$ are in V.

Let $f ^ { \prime } = u w$ be in (F). Looking at condition (iv), we see that $\{ u w ^ { \prime } \in F \mid w ^ { \prime } \in \mathrm { B P } _ { w } \} = \{ u w \}$ Then, by (38) of Claim 1, it holds that

$$
\mathbb { 1 } _ { \{ u w \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) .
$$

It remains to consider interactions that belong to (G). Let $f ^ { \prime } = u w$ be such an interaction, i.e., u $\notin N _ { G } [ C ]$ and $w \in C$ , then we consider two possible cases:

(G.1) There exists a vertex v in the intersection of $N _ { G } ( C )$ and a minimal $\{ \{ u \} , N _ { G } [ C ] \}$ connector of G such that w $\in C \setminus \mathrm { B P } _ { v }$ . Then, from (39) of Claim 1, it follows that

$$
\mathbb { 1 } _ { \{ f ^ { \prime } \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) .
$$

(G.2) For every $v \in N _ { G } ( C )$ that is on some minimal $\{ \{ u \} , N _ { G } [ C ] \}$ -connector of $G ,$ we have $w \in C \cap \mathrm { B P } _ { v }$ . Let us fix such a vertex v. By (38), we know that

$$
\mathbb { 1 } _ { \{ u w ^ { \prime } \in F | w ^ { \prime } \in \mathrm { B P } _ { v } \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) .
$$

Furthermore, in the following we argue that $\mathbb { 1 } _ { \{ u w ^ { \prime } \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$ holds for all $u w ^ { \prime } \in F$ with $w ^ { \prime } \in \mathrm { B P } _ { v } \ \backslash \ \{ w \}$ . Taking into account these two facts, we obtain

$$
\mathbb { 1 } _ { \{ f ^ { \prime } \} } = \mathbb { 1 } _ { \{ u w ^ { \prime } \in F | w ^ { \prime } \in \mathrm { B P } _ { v } \} } - \sum _ { u w ^ { \prime } \in F : \ w ^ { \prime } \in \mathrm { B P } _ { v } \setminus \{ w \} } \mathbb { 1 } _ { \{ u w ^ { \prime } \} } \in \mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } ) .
$$

In order to prove that $\mathbb { 1 } _ { \{ u w ^ { \prime } \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$ holds for all $u w ^ { \prime } \in F$ with $w ^ { \prime } \in \mathrm { B P } _ { v } \ \backslash \ \{ w \}$ , by case (G.1) and the previous discussion concerning (E), it sufices to show the following: For any other interaction $u w ^ { \prime } \in F$ with $w ^ { \prime } \in \mathrm { B P } _ { v } \ \backslash \ \{ w \}$ , there exists a vertex $v ^ { \prime }$ in the intersection of $N _ { G } ( C )$ and a minimal $\{ \{ u \} , N _ { G } [ C ] \}$ }-connector of G such that $w ^ { \prime } \notin \mathrm { B P } _ { v ^ { \prime } }$ . Suppose to the contrary that there exists an interaction $u w ^ { \prime } \in F$ with $w ^ { \prime } \in \mathrm { B P } _ { v } \ \backslash \ \{ w \}$ such that for any vertex $v ^ { \prime }$ in the intersection of $N _ { G } ( C )$ and a minimal $\{ \{ u \} , N _ { G } [ C ] \}$ -connector of G it holds that $w ^ { \prime } \in \mathrm { B P } _ { v ^ { \prime } }$ . If $w ^ { \prime } \in N _ { G } ( C )$ , then v coincides with $w ^ { \prime }$ and it is a $\{ \{ u \} , C \}$ -cut-vertex of G. This is a contradiction to condition (iv) and hence such an interaction uw<sup>′</sup> cannot exist. If $w ^ { \prime } \in C$ , then the set of all neighbors of $C$ that are on minimal $\{ \{ u \} , N _ { G } [ C ] \}$ -connectors of $G ,$ which is clearly a $\{ \{ u \} , C \}$ -separator of G, is a subset of

$\{ v \in N _ { G } ( C ) \mid$ v is both a w-bypass and a w<sup>′</sup>-bypass of C} .

This leads to a contradiction to condition (v).

Combining case (G.1) and case (G.2), we see that $\mathbb { 1 } _ { \{ f ^ { \prime } \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$ holds for every interaction $f ^ { \prime }$ in (G).

Looking at all the results obtained above, we arrive at the statement that the linear space lin $( X ^ { \prime } - X ^ { \prime } )$ has a basis

$$
\begin{array} { r l } & { \quad \left\{ \mathbb { 1 } _ { \{ v , f \} } \mid v \in C \right\} } \\ & { \cup \left\{ \mathbb { 1 } _ { \{ v \} } \mid v \in V \setminus C \right\} } \\ & { \cup \left\{ \mathbb { 1 } _ { \{ f ^ { \prime } \} } \mid f ^ { \prime } \in F \setminus \{ f \} \right\} . } \end{array}
$$

Since the polytope $\Xi _ { G F }$ is full-dimensional by Proposition $2 ,$ we conclude that the connector inequality (4) indeed defines a facet of $\Xi _ { G F }$ under the given conditions. □

## D.11 Proof of Theorem 5

In this section, we give a proof of Theorem 5, i.e., showing that conditions ${ \mathrm { ( i ) - ( i v ) } }$ are suficient and necessary for separator inequality (5) to be facet-defining. To determine the dimension of the corresponding face, we shall develop some general properties of the feasible vectors in this face. More specifically, let

$$
X ^ { \prime } = \left\{ x \in X _ { G F } \Bigg | 1 - x _ { f } = \sum _ { s \in S } ( 1 - x _ { s } ) \right\}
$$

denote the set of all feasible vectors of the multi-separator problem (MS) that satisfy separator inequality (5) associated with $f$ and $S$ at equality. Since $x ^ { \alpha } = \mathbb { 1 } _ { V \cup F } \in X ^ { \prime }$ , the linear space lin $( X ^ { \prime } - X ^ { \prime } )$ associated with the afine hull of $X ^ { \prime }$ is

$$
\begin{array} { r } { \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) = \operatorname { s p a n } \left\{ \mathbb { 1 } _ { V \cup F } - x \mid x \in X ^ { \prime } \right\} . } \end{array}
$$

So inequality (2) is facet-defining if and only if $\operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$ has codimension 1 in $\mathbb { R } ^ { V \cup F }$ . Following the strategy described in Section C, we first analyze lin $( \dot { X } ^ { \prime } - X ^ { \prime } )$ by decomposing it into several nicely behaved pieces, so that the structure of the feasible vectors belonging to each piece can be easily understood. By combining structural information from diferent pieces, we are able to obtain a spanning set of lin $( X ^ { \prime } - X ^ { \prime } )$ . Then an auxiliary loop graph is constructed from these spanning vectors so that our desired characterization follows as a consequence of Lemma 8 in Section C. For this purpose, we begin with the introduction of a few useful notions and the development of their properties, only after which are we ready to show the suficiency and necessity of the stated conditions. As the first step, let us note that, for any $s \in S$ , if we denote by $X _ { s } ^ { \prime }$ the set of all feasible vectors $x \in X ^ { \prime }$ satisfying $x ^ { - 1 } ( 0 ) \cap S \subseteq \{ s \}$ , and by $X _ { * } ^ { \prime }$ the set of all feasible vectors $x \in X ^ { \prime }$ satisfying $x ^ { - 1 } ( 0 ) \cap S = \emptyset$ , then $X ^ { \prime }$ admits the decomposition $X ^ { \prime } = \cup _ { s \in S \cup \{ * \} } X _ { s } ^ { \prime }$ . This in turn implies

$$
\mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } ) = \sum _ { s \in S \cup \{ * \} } \mathrm { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } ) .\tag{49}
$$

Before proceeding further, let us consider vertices not in $S$ and interactions not in $F ( S )$ . Because their corresponding standard unit vectors, which will later be completed to a basis of lin $\left( X ^ { \prime } - X ^ { \prime } \right)$ can be easily constructed as follows. For any vertex $v \in V \setminus S ,$ we have $Q _ { \mathcal { O } \{ v \} } = \{ v \}$ . Since both $x ^ { \alpha }$ and $x ^ { \{ v \} }$ belong to $X _ { * } ^ { \prime }$ , the corresponding standard unit vector $\mathbb { 1 } _ { \{ v \} }$ is in lin $\left( X _ { * } ^ { \prime } - X _ { * } ^ { \prime } \right)$ by $( 2 2 )$ of Lemma 7. For any interaction $f ^ { \prime } = u w \in F$ that is not in $F ( \bar { S } )$ , there exists a minimal $f ^ { \prime } .$ -connector $C$ of G such that $C \cap S = \emptyset$ . Set $A = C \backslash \{ w \}$ and $B = C \backslash \{ u \}$ , then all $x ^ { A } , x ^ { B } , x ^ { A \cap B }$ and $x ^ { A \cup B }$ belong to $X _ { * } ^ { \prime }$ . Furthermore, $F _ { A B } = \{ f ^ { \prime } \}$ and $H _ { A B } = \mathcal { O }$ . Again, by (23) of Lemma $^ { 7 , }$ the standard unit vector $\mathbb { 1 } _ { \left\{ f ^ { \prime } \right\} }$ is in lin $\left( X _ { * } ^ { \prime } - X _ { * } ^ { \prime } \right)$ . Thus, we have constructed all the standard unit vectors of vertices in $V \setminus S$ and interactions in $F \setminus F ( S )$ . Since the summand lin $\left( X _ { * } ^ { \prime } - X _ { * } ^ { \prime } \right)$ in (49) is a subspace of $\mathbb { R } ^ { V \backslash S } \oplus \dot { \mathbb { R } } ^ { F \setminus F ( S ) }$ , the result we just obtained can be summarized as

$$
\operatorname { l i n } ( X ^ { \prime } - X ^ { \prime } ) = \Bigl ( \mathbb { R } ^ { V \setminus S } \oplus \mathbb { R } ^ { F \setminus F ( S ) } \Bigr ) + \sum _ { s \in S } \operatorname { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } ) .\tag{50}
$$

This in particular indicates that in our further development we only need to concentrate on those elements in $S \cup F ( S )$ . On the other hand, constructing the remaining basis vectors is not straightforward. To achieve this, we utilize the properties of ${ \bar { X ^ { \prime } } }$ developed (till Claim 2), to further refine the decomposition (50).

To begin our development of properties of $X ^ { \prime } { \mathrm { . } }$ , let us assume that condition (i) of Theorem 5 holds, i.e., $S$ is a minimal f-separator of $G ,$ so that all the notions introduced in Definition 4 are valid. Note that this assumption is justified, because, by Lemma 1, we already know that the minimality of $S$ is a necessary condition for the associated separator inequality to be facet-defining. For notational simplicity, given any $s \in S$ , we write $C ( s )$ for the set of all f-cut-vertices of $G ( S , s )$

In the following we partition interactions that are connected in $G ( S , s )$ into several classes for each fixed $s \in S$ . Such a partition is motivated by a simple fact about $X ^ { \prime }$ as explained below. For any fixed $s \in S$ and any $x \in X _ { s } ^ { \prime } \setminus X _ { * } ^ { \prime }$ , it holds that $x _ { f } = 0 , x _ { v } = 0$ for all $v \in C ( s )$ and that $x _ { s ^ { \prime } } = 1$ for all $s ^ { \prime } \in S \setminus \{ s \}$ . Since every two vertices of $C ( s )$ are connected in $G [ V \cap x ^ { - 1 } ( 0 )$ ], it follows that $x _ { u v } = x _ { u v ^ { \prime } }$ for all $u v , u v ^ { \prime } \in F ( S )$ with $v , v ^ { \prime } \in C ( s )$ ; that is, two such related interactions $u v , u v ^ { \prime }$ are always assigned the same value by all the feasible vectors $x \in X _ { s } ^ { \prime }$ . This suggests that, if we restrict ourselves to $X _ { s } ^ { \prime } ,$ then it is natural to regard any two adjacent interactions as equivalent which are connected in $G ( S , s )$ and whose non-common endvertices belong to $C ( s )$ . More importantly, it is desirable to have the property that the characteristic vectors of equivalence classes belong to lin $( X ^ { \prime } - X ^ { \prime } )$ , from which we can proceed further to construct basis vectors of lin $( X ^ { \prime } - X ^ { \prime } )$ . The above consideration motivates us to introduce the following definition, which plays a fundamental role in our approach.

With the given minimal f-separator S of $G ,$ we define the set $[ f ^ { \prime } ] _ { s } \subseteq S \cup F ( S )$ for any $f ^ { \prime } = u w \in F ( S )$ and any $s \in S$ according to the following four possible cases:

(i) If $s \not \in S ( f ^ { \prime } )$ , then $[ f ^ { \prime } ] _ { s } = \emptyset .$

(ii) $\operatorname { I f } s \in S ( f ^ { \prime } )$ and $u , w \not \in C ( s )$ , then $[ f ^ { \prime } ] _ { s } = \{ f ^ { \prime } \}$

(iii) If $s \in S ( f ^ { \prime } )$ and there is precisely one endvertex, say w, of $f ^ { \prime }$ that is in $C ( s )$ , then $[ f ^ { \prime } ] _ { s } = \{ u w ^ { \prime } \in F ( S ) \mid w ^ { \prime } \in C ( s ) \}$

(iv) If $s \in S ( f ^ { \prime } )$ and $u , w \in C ( s )$ , then $[ f ^ { \prime } ] _ { s } = \{ s \} \cup \{ f ^ { \prime \prime } \in F ( S ) \mid f ^ { \prime \prime } \subseteq C ( s ) \}$

Note that for each fixed $s \in S$ , the collection of all the sets $[ f ^ { \prime } ] _ { s }$ for $f ^ { \prime } \in F ( S )$ with $s \in S ( f ^ { \prime } )$ gives a partition of the set consisting of s and all interactions $f ^ { \prime } \in \dot { F } ( \bar { S } )$ with $s \in S ( f ^ { \prime } )$ . For example, in Figure $7 \ \mathrm { ( e ) }$ , the vertex $s _ { 1 }$ induces the partition $\{ \{ s _ { 1 } , f , f _ { 1 } \} , \{ f _ { 2 } , f _ { 3 } \} \}$ } and $s _ { 2 }$ induces the partition $\{ \{ s _ { 2 } , f , f _ { 3 } \} , \{ f _ { 1 } , f _ { 2 } \} \}$ ; in Figure $7 \ : ( \mathrm { f } )$ , the vertex s induces the partition $\left\{ \{ s _ { 1 } , f \} , \{ f _ { 1 } , f _ { 2 } \} , \{ f _ { 3 } , f _ { 4 } \} \right\}$ and s induces the partition $\{ \{ s _ { 2 } , f \} , \{ f _ { 1 } , f _ { 4 } \} , \{ f _ { 2 } , f _ { 3 } \} \}$

The following result, which is employed to demonstrate both the suficiency and the necessity of Theorem 5 afterward, reinforces our original idea to introduce $[ f ^ { \prime } ] _ { s }$

Claim 2. Let $f ^ { \prime } \in F ( S )$ and $s \in S ( f ^ { \prime } )$ , then for any $x \in X _ { s } ^ { \prime }$ it holds that

$$
\forall g \in [ f ^ { \prime } ] _ { s } \colon x _ { g } = x _ { f ^ { \prime } } .\tag{51}
$$

Moreover, we have

$$
\mathbb { I } _ { [ f ^ { \prime } ] _ { s } } \in \operatorname* { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } ) .\tag{52}
$$

Proof of claim. To show the first statement, i.e. (51), let $x \in X _ { s } ^ { \prime }$ and let g be an arbitrary element in $[ f ^ { \prime } ] _ { s }$ . As $[ f ^ { \prime } ] _ { s } \subseteq S \cup F ( S )$ , the statement is clear when $x \in X _ { * } ^ { \prime } . \mathrm { ~ I f ~ } x \in X _ { s } ^ { \prime } \ \backslash \ X _ { * } ^ { \prime } .$ , then $x _ { s } = 0$ $x _ { f } = 0$ , and $x _ { s ^ { \prime } } = 1$ for all $s ^ { \prime } \in S \setminus \{ s \}$ . So there is a minimal f-connector C of $G ( S , s )$ with $x | _ { C } = 0$ . Suppose $x _ { f ^ { \prime } } = 0$ . Then we can find a minimal f<sup>′</sup>-connector $C ^ { \prime }$ of $G ( S , s )$ with $x | _ { C ^ { \prime } } = 0$ From the definition of $[ f ^ { \prime } ] _ { s } ,$ , we see that g is connected in $G [ C \cup C ^ { \prime } ]$ , and hence $x _ { g } = 0$ . As $f ^ { \prime } \in [ g ] _ { s } ,$ a similar argument yields that $x _ { g } = 0$ implies $x _ { f ^ { \prime } } = 0$ . Therefore the desired equality $x _ { f ^ { \prime } } = x _ { g }$ follows.

For the second statement, i.e. (52), we prove it by applying Lemma 7 to construct the characteristic vector $\mathbb { 1 } _ { [ f ^ { \prime } ] }$ as a linear combination of vectors in lin $\left( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } \right)$ . Let us start with introducing a few vertex subsets that will be used in the subsequent construction. Write $f = p q$ and $f ^ { \prime } = u w .$ Let $C _ { 1 }$ (respectively, $C _ { 2 } )$ be a minimal ps-connector (respectively, qs-connector) of $G ( S , s )$ . If u (respectively, w) is not in $C ( s )$ , then it is clear that we can choose $C _ { 1 } , C _ { 2 }$ such that $u \not \in C _ { 1 } \cup C _ { 2 }$ (respectively, w $\ , \ d \ C _ { 1 } \cup C _ { 2 } )$ . Furthermore, if both u and w are not in $C ( s )$ , then $C _ { 1 }$ and $C _ { 2 }$ can be chosen such that u, w $\notin C _ { 1 } \cup C _ { 2 }$ . Note that $C _ { 1 } \cup C _ { 2 }$ is a minimal f-connector of G. In any case, as $s \in S ( f ^ { \prime } )$ , we can pick a minimal $\{ \{ u \} , N _ { G } [ C _ { 1 } \cup C _ { 2 } ] \}$ -connector $C _ { 1 } ^ { \prime }$ of $G ( S , s )$ and a minimal $\{ \{ w \} , N _ { G } [ \dot { C } _ { 1 } \cup C _ { 2 } ] \}$ -connector $C _ { 2 } ^ { \prime }$ of $G ( S , s )$ , respectively. Write $U = C _ { 1 } \cup C _ { 2 } \cup C _ { 1 } ^ { \prime } \cup C _ { 2 } ^ { \prime }$ . Examples of the sets constructed here can be found in Figure 16. Below we show that $\mathbb { I } _ { [ f ^ { \prime } ] _ { s } } \in \operatorname* { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } )$ by distinguishing three possible cases as in the definition of $[ f ^ { \prime } ] _ { s }$

(A) If both u and w are not f-cut-vertices of $G ( S , s )$ , then we set $A = U \backslash \{ u \}$ and $B = U \backslash \{ w \}$ Thus, all $x ^ { A } , x ^ { B } , x ^ { A \cap B }$ and $x ^ { A \cup B }$ belong to $X _ { s } ^ { \prime }$ . Furthermore, $F _ { A B } = \{ f ^ { \prime } \}$ and $H _ { A B } = \delta . \mathrm { B y }$ (23) of Lemma 7, we get $\mathbb { 1 } _ { [ f ^ { \prime } ] _ { s } } = \mathbb { 1 } _ { \{ f ^ { \prime } \} } \in \mathrm { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } )$

(B) If u is not an f-cut-vertex of $G ( S , s )$ but w is an f-cut-vertex of $G ( S , s )$ , then we set $A = U \setminus \{ u \}$ and $B = U \setminus \{ s \}$ . It follows by definition that $F _ { A B } = \{ u w ^ { \prime } \in F \mid w ^ { \prime } \in C _ { 1 } \cup$ $C _ { 2 }$ and s is a uw<sup>′</sup>-cut-vertex of $G [ U ] \}$ and $H _ { A B } = \mathcal { O }$ . Thus, (23) of Lemma 7 immediately gives $\mathbb { I } _ { F _ { A B } } \in \operatorname* { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } )$ , upon noting that $x ^ { A } , x ^ { B } , x ^ { A \cap B }$ and $x ^ { A \cup B }$ belong to $X _ { s } ^ { \prime }$ . On the other hand, we have the identity

![](images/ecd44f53af852305578802106fb145399ab2e575e4ee8b52044c30244aff0b6e.jpg)  
Figure 16: Depicted above are examples illustrating the constructions in the proof of Claim 2. Each figure consists of a graph G (solid black), a designated interaction $f = p q$ (dashed red), a minimal f-separator $S = \{ s \}$ of G, and other interactions (dashed green). In a), if $C _ { 1 } = \{ p , s \} , C _ { 2 } = \{ s , w , w _ { 1 } , q \} , C _ { 1 } ^ { \prime } = \{ u \}$ and $C _ { 2 } ^ { \prime } = \{ w \}$ , then $U = \{ u , p , s , w , w _ { 1 } , q \} , A = U \setminus \{ u \}$ and $B = U \setminus \{ s \}$ . Thus, $F _ { A B } = \{ u p , u s , u w , u w _ { 1 } \}$ and $[ u w ] _ { s } = \{ u s , u w \}$ . In $\mathrm { b ) , ~ i f } \ C _ { 1 } = \{ p , u _ { 1 } , u , s \} , C _ { 2 } = \{ s , w , w _ { 2 } , q \} , C _ { 1 } ^ { \prime } = \{ u \}$ and $C _ { 2 } ^ { \prime } = \{ w \}$ , then $U =$ $\{ p , u _ { 1 } , u , s , w , w _ { 2 } , q \} , A = U \setminus \{ s \}$ and $B = U$ . Thus, $Q _ { A B } = \{ u w , u _ { 1 } w , u w _ { 2 } , s w _ { 2 } , s , f \} , [ u w ] _ { s } = \{ s , f , u w \}$ $[ u _ { 1 } w ] _ { s } = \{ u _ { 1 } w \}$ , and $[ u w _ { 2 } ] _ { s } = \{ u w _ { 2 } , s w _ { 2 } \}$

$$
\mathbb { 1 } _ { [ f ^ { \prime } ] _ { s } } = \mathbb { 1 } _ { F _ { A B } } - \sum _ { u w ^ { \prime } \in F _ { A B } \setminus F ( S ) } \mathbb { 1 } _ { \{ u w ^ { \prime } \} } - \sum _ { u w ^ { \prime } \in F _ { A B } \cap F ( S ) : \ w ^ { \prime } \notin C ( s ) } \mathbb { 1 } _ { \{ u w ^ { \prime } \} } .
$$

Therefore, $\mathbb { I } _ { [ f ^ { \prime } ] _ { s } } \in \operatorname* { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } )$ , since $\mathbb { I } _ { F _ { A B } } \in \operatorname* { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } )$ as we just saw, and $\mathbb { 1 } _ { \{ u w ^ { \prime } \} } \in \operatorname* { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } )$ for every interaction $u w ^ { \prime } \notin F ( S )$ by the discussion preceding this claim, and $\bar { \mathbb { 1 } } \dot { \left\{ u w ^ { \prime } \right\} } \in \operatorname* { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } )$ for every interaction $u w ^ { \prime } \in F ( S )$ with $w ^ { \prime } \in ( C _ { 1 } \cup C _ { 2 } ) \setminus C ( s )$ by the previous case. See Figure 16 a) for an example. Similarly, if u is an f-cut-vertex of $G ( S , s )$ but w is not an f-cut-vertex of $G ( S , s )$ then we can also obtain $\mathbb { I } _ { [ f ^ { \prime } ] _ { s } } \in \operatorname* { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } )$

(C) Now, suppose both u and w are f-cut-vertices of $G ( S , s )$ . Set $A = U \setminus \{ s \}$ and $B = U$ $\mathrm { T h e n } , x ^ { A } , x ^ { B } \in \bar { X _ { s } ^ { \prime } }$ and $Q _ { A B } = \{ s \} \cup \{ u ^ { \prime } w ^ { \prime } \in F ( S ) \mid u ^ { \prime } \in C _ { 1 }$ and $w ^ { \prime } \in C _ { 2 } \}$ . By (22) of Lemma 7, we get $\mathbb { 1 } _ { Q _ { A B } } \in \operatorname* { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } )$ . Moreover, it holds that

$$
\mathbb { 1 } _ { [ f ^ { \prime } ] _ { s } } = \mathbb { 1 } _ { Q _ { A B } } - \sum _ { \substack { [ u ^ { \prime } w ^ { \prime } ] _ { s } : u ^ { \prime } \in C _ { 1 } \backslash C ( s ) \mathrm { ~ o r ~ } w ^ { \prime } \in C _ { 2 } \backslash C ( s ) } } \mathbb { 1 } _ { [ u ^ { \prime } w ^ { \prime } ] _ { s } } .
$$

Notice that here the sum is taken over all distinct sets $[ u ^ { \prime } w ^ { \prime } ] _ { \varepsilon }$ with $u ^ { \prime } \in C _ { 1 } \setminus C ( s )$ or $w ^ { \prime } \in C _ { 2 } \setminus C ( s )$ It is clear that $x ^ { A } , x ^ { B } , x ^ { A \cap B }$ and $x ^ { A \cup B }$ are all in $X _ { s } ^ { \prime } .$ . By the results above, we have already seen that $\mathbb { 1 } _ { [ u ^ { \prime } w ^ { \prime } ] _ { s } } \in \operatorname* { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } )$ for every interaction $u ^ { \prime } w ^ { \prime } \in F ( S )$ with $u ^ { \prime } \in C _ { 1 } \backslash C ( s )$ or $w ^ { \prime } \in C _ { 2 } \backslash C ( s )$ Thus once again $\mathbb { I } _ { [ f ^ { \prime } ] _ { s } } \in$ lin $\left( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } \right)$ . See Figure 16 b) for an example. The above discussion has established (52) in all possible cases, and hence completes the proof.

Q.E.D.

As an immediate consequence of Claim $^ { 2 , }$ we have

$$
\mathrm { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } ) = \mathbb { R } ^ { V \setminus S } \oplus \mathbb { R } ^ { F \setminus F ( S ) } \oplus \sum _ { f ^ { \prime } \in F ( S ) } \mathrm { s p a n } \mathbb { 1 } _ { [ f ^ { \prime } ] _ { s } }
$$

for every $s \in S$ . Combining with (50) yields

$$
\mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } ) = \mathbb { R } ^ { V \setminus S } \oplus \mathbb { R } ^ { F \setminus F ( S ) } \oplus \sum _ { s \in S , f ^ { \prime } \in F ( S ) } \mathrm { s p a n } \mathbb { 1 } _ { [ f ^ { \prime } ] _ { s } } .\tag{53}
$$

Let us take a closer look at the last direct summand of the above decomposition. Firstly, we observe that if $f ^ { \prime } \in F ( S )$ satisfies $f ^ { \prime } \cap C ( s ) = \emptyset$ for some $s \in S ( f ^ { \prime } )$ , then $\mathbb { 1 } _ { \{ f ^ { \prime } \} } \in \operatorname* { l i n } ( X _ { s } ^ { \prime } - X _ { s } ^ { \prime } )$ by (52) of Claim 2, since $[ f ^ { \prime } ] _ { s } = \{ f ^ { \prime } \}$ in this case. Secondly, for each vertex $s \in S$ there is exactly one set, i.e., $[ f ] _ { s }$ , that contains s among all $[ f ^ { \prime } ] _ { s ^ { \prime } }$ with $s ^ { \prime } \in S$ and $f ^ { \prime } \in F ( S )$ . Similarly, each $s v \in F ( S )$ with $s \in S$ and $v \in G ( S , s ) \setminus C ( s )$ is contained in $[ s v ] _ { s }$ only. Thirdly, for every $u w \in F ( S )$ such that u is non-degenerate, u $\notin C ( s )$ for some $s \in S ( u w )$ , and $w \in C ( s )$ for any $s \in S ( u w )$ , there is one and only one set, i.e., [uw]<sub>s</sub> for some $s \in S ( u w )$ with us $\notin F$ , that contains uw among all $[ f ^ { \prime } ] _ { s ^ { \prime } }$ with $s ^ { \prime } \in S$ and $f ^ { \prime } \in F ( { \bar { S } } )$ . Taking these observations into account, the last term of (53) can be rewritten as

![](images/180051edc492148aa0f127a9377054a418e6f3b7d140ae61da5deb9012847c36.jpg)  
a) ⟨v⟩ = [f<sup>′</sup>]<sub>s1</sub> = [f<sup>′</sup>]<sub>s2</sub>

![](images/f0af8e0571fc7e31dad084a157901014ea0449593eab9df06ceda2dbc41e65ce.jpg)  
b) ⟨v⟩ = [f<sup>′</sup>]<sub>s1</sub> ̸= [f<sup>′</sup>]<sub>s2</sub>  
Figure 17: Depicted above are examples illustrating the notion ⟨v⟩. Each figure consists of a graph $G$ (solid black), a designated interaction $f = p q$ (dashed red), a minimal f-separator $S = \{ s _ { 1 } , s _ { 2 } \}$ of $G ,$ and other interactions (dashed green). In a), $[ f ^ { \prime } ] _ { s _ { 1 } } = \{ f ^ { \prime } , f ^ { \prime \prime } \} = [ f ^ { \prime } ] _ { s _ { 2 } }$ , hence $\langle v \rangle = [ f ^ { \prime } ] _ { s _ { 1 } } = [ f ^ { \prime } ] _ { s _ { 2 } }$ . As a comparison, we have $[ f ^ { \prime } ] _ { s _ { 1 } } = \{ f ^ { \prime } , f ^ { \prime \prime } \}$ and $[ f ^ { \prime } ] _ { s _ { 2 } } = \{ f ^ { \prime } , f ^ { \prime \prime } , v s _ { 2 } \}$ in b), hence $\langle v \rangle = [ f ^ { \prime } ] _ { s _ { 1 } } \neq [ f ^ { \prime } ] _ { s _ { 2 } }$

$$
\sum _ { \substack { s \in S , f ^ { \prime } \in F ( S ) } } \operatorname { s p a n } \mathbb { 1 } _ { [ f ^ { \prime } ] _ { s } } \cong \mathbb { R } ^ { S } \oplus \mathbb { R } ^ { \{ f ^ { \prime } \in F ( S ) \mid \exists s \in S ( f ^ { \prime } ) : f ^ { \prime } \cap C ( s ) = \emptyset \} } \oplus \mathbb { R } ^ { \{ s v \in F ( S ) \mid s \in S , v \in G ( S , s ) \backslash C ( s ) \} }
$$

$$
\oplus \sum _ { \substack { s \in S , f ^ { \prime } \in F _ { \parallel } } } \operatorname { s p a n } \mathbb { 1 } _ { [ f ^ { \prime } ] _ { s } } \oplus \sum _ { \substack { s \in S , f ^ { \prime } \in F ( S , s ) } } \operatorname { s p a n } \mathbb { 1 } _ { [ f ^ { \prime } ] _ { s } } ,\tag{54}
$$

where

$$
F _ { \parallel } = \left\{ u w \in F ( S ) \left| \begin{array} { l } { S \in \{ \{ u \} , f \} \mathrm { - s e p a r a t o r s } ( G ) , } \\ { u \mathrm { i s ~ n o n - d e g e n e r a t e } , \mathrm { a n d } } \\ { \forall s \in S ( u w ) \colon w \in C ( s ) } \end{array} \right. \right\}
$$

and

$$
\begin{array}{c} \begin{array} { r } { F ( S , s ) = \left\{ \left. f ^ { \prime } \in F ( S ) \setminus \binom { C ( s ) } { 2 } \right| \left. \forall \in S ( f ^ { \prime } ) , \forall f ^ { \prime \prime } \in [ f ^ { \prime } ] _ { s } \colon s \notin f ^ { \prime \prime } , \right. \right\} \\ { \left. \forall { s ^ { \prime } \in S ( f ^ { \prime } ) } \colon f ^ { \prime } \cap C ( s ^ { \prime } ) \neq \emptyset , \right. \qquad } \\ { \left. \forall v \in f ^ { \prime } \colon S \notin \{ \{ v \} , f \} \mathrm { . s e p a r a t o r s } ( G ) \right) } \end{array} }  \end{array}
$$

for every $s \in S$ . Note that if $f ^ { \prime } \in F ( S , s )$ for some $s \in S$ , then neither of the endvertices of $f ^ { \prime }$ is separated from f by $S$ in G.

Combining (53) and (54), we obtain

$$
\operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) \cong \mathbb { R } ^ { V } \oplus \mathbb { R } ^ { ( F \setminus F _ { \mathcal { G } } ) \setminus ( \cup _ { s \in S } F ( S , s ) \cup F _ { \mathbb { I } } ) } \oplus \sum _ { s \in S , f ^ { \prime } \in F _ { \mathbb { I } } } \operatorname { s p a n } \mathbb { 1 } _ { [ f ^ { \prime } ] _ { s } } \oplus \sum _ { s \in S , f ^ { \prime } \in F ( S , s ) } \operatorname { s p a n } \mathbb { 1 } _ { [ f ^ { \prime } ] _ { s } } ,\tag{55}
$$

where

$$
F _ { \mathcal { O } } = \left\{ f ^ { \prime } \in F ( S ) \left| \begin{array} { l l } { \forall s \in S ( f ^ { \prime } ) \colon \mathrm { e i t h e r } f ^ { \prime } \subseteq C ( s ) , \mathrm { ~ o r } } \\ { f ^ { \prime } \setminus ( C ( s ) \setminus S ) = \{ v \} \mathrm { ~ a n d ~ } v s \in F } \end{array} \right. \right\} .
$$

Now it is clear that in order to determine the dimension of $\operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$ , we just need to understand the last direct summand of decomposition (55). To this end, in the next step we construct an auxiliary loop graph so that its (vertex-edge) incidence matrix has full column rank if and only if so does the last direct summand of (55). Then we can utilize Lemma 8 to decide when lin $( X ^ { \prime } - X ^ { \prime } )$ has codimension 1. More specifically, we require that, for all vertices of the auxiliary loop graph that are not incident with loops, the corresponding characteristic vectors of incident edges are given exactly by $\mathbb { 1 } _ { [ f ^ { \prime } ] _ { s } }$ with $s \in S$ and $f ^ { \prime } \in F ( S , s )$ . Notice that we exclude vertices incident with loops in the above requirement because in the last direct summand of (55) it is possible that some $f ^ { \prime }$ is contained in only one equivalence class $[ f ^ { \prime } ] _ { s }$ . The exact construction will be given below after making one more vital observation.

To achieve our goal of interpreting the last direct summand of (55) as the incidence matrix of some loop graph, let us observe that for any $v \in V \setminus S$ that is not an f-cut-vertex of G and any $f ^ { \prime } \in \cup _ { s \in S } F ( S , s )$ with $v \in f ^ { \prime } ,$ it holds that $[ f ^ { \prime } ] _ { s _ { 1 } } = [ f ^ { \prime } ] _ { s _ { 2 } }$ for all $s _ { 1 } , s _ { 2 } \in S ( f ^ { \prime } )$ satisfying $v \not \in C ( s _ { 1 } ) \cup C ( s _ { 2 } )$ and $v s _ { 1 } , v s _ { 2 } \notin F$ . See Figure 17 for illustration. This observation motivates us to introduce $\langle v \rangle$ to represent the common set $[ f ^ { \prime } ] _ { s _ { 1 } } = [ f ^ { \prime } ] _ { s _ { 2 } }$ and hence the set of edges incident with v. More formally, for any $v \in V$ , we define ⟨v⟩ as the set of all those interactions $f ^ { \prime } \in F ( S )$ with $v \in f ^ { \prime }$ that satisfy the following two conditions:

(a) $f ^ { \prime } \cap C ( s ) \neq \emptyset$ for any $s \in S ( f ^ { \prime } )$

(b) There exists $s \in S ( f ^ { \prime } )$ such that v $\notin C ( s )$ and vs $\notin F$

Notice that if $v \in V$ is degenerate, then $\langle v \rangle$ is empty. Moreover, the union of all $\langle v \rangle$ with v not separated from f by S in G is exactly $\cup _ { s \in S } F ( S , s )$ . We construct an auxiliary loop graph $G _ { \infty } = ( V _ { \infty } , E _ { \infty } )$ as follows:

$V _ { \infty }$ is the set of all vertices of G that are not separated from f by S in $G .$

$E _ { \infty }$ consists of all loops {v} incident with degenerate vertices v in $V _ { \infty }$ and all interactions in $\cup _ { v \in V _ { \infty } } \langle v \rangle$

Note that the loopless part of $G _ { \infty }$ is bipartite, whose two parts are given by the two vertex subsets $\{ v \in V \setminus S \mid S \not \in$ uv-separators(G)} for $u \in f$ . This means that $G _ { \infty }$ has no odd cycles except loops. It follows from the definition of ⟨v⟩ that

$$
\sum _ { \substack { s \in S , f ^ { \prime } \in F _ { \parallel } } } \operatorname { s p a n } \mathbb { 1 } _ { [ f ^ { \prime } ] _ { s } } = \bigoplus _ { \substack { \langle v \rangle \colon v \in V \backslash V _ { \infty } } } \operatorname { s p a n } \mathbb { 1 } _ { \langle v \rangle }
$$

and

$$
\sum _ { \substack { s \in S , f ^ { \prime } \in F ( S , s ) } } \mathrm { s p a n } \mathbb { 1 } _ { [ f ^ { \prime } ] _ { s } } = \sum _ { v \in V _ { \infty } } \mathrm { s p a n } \mathbb { 1 } _ { \langle v \rangle } .
$$

Putting it all together, we have

$$
\operatorname { l i n } ( X ^ { \prime } - X ^ { \prime } ) \cong \mathbb { R } ^ { V } \oplus \mathbb { R } ^ { ( F \setminus F _ { \mathcal { B } } ) \setminus \cup _ { v \in V } \langle v \rangle } \oplus \bigoplus _ { \langle v \rangle : v \in V \setminus V _ { \infty } } \operatorname { s p a n } \mathbb { 1 } _ { \langle v \rangle } \oplus \sum _ { v \in V _ { \infty } } \operatorname { s p a n } \mathbb { 1 } _ { \langle v \rangle } .\tag{56}
$$

We further observe that the row space of the incidence matrix of $G _ { \infty }$ is

$$
\bigoplus _ { \{ v \in V _ { \infty } : \{ v \} \in E _ { \infty } \} } \operatorname { s p a n } \mathbb { 1 } _ { \{ e \in E _ { \infty } | v \in e \} } \oplus \sum _ { v \in V _ { \infty } } \operatorname { s p a n } \mathbb { 1 } _ { \langle v \rangle } .
$$

Taking the preceding observation into account and looking at the decomposition (56), we see that lin $( X ^ { \prime } - X ^ { \prime } )$ has codimension 1 in $\mathbb { R } ^ { V \cup F }$ if and only if

(ii<sup>′</sup>) $F _ { \cal { \emptyset } } = \{ f \}$

(iii<sup>′</sup>) $| \langle v \rangle | \leq 1$ for every $v \in V$ such that S is a $\{ \{ v \} , f \}$ -separator of $G$

(iv') The incidence matrix of (iv<sup>′</sup>) The incidence matrix of $G _ { \infty }$ has full column rank. has full column rank.

Moreover, by Lemma 8 and the fact that the loopless part of $G _ { \infty }$ is bipartite, condition $( \mathrm { i v ^ { \prime } } )$ is equivalent to the statement that $G _ { \infty }$ is a pseudoforest without cycles of length ≥ 2.

Finally, we claim that, assuming S is a minimal f-separator of $G ,$ conditions (ii), (iii) and (iv) in Theorem 5 are equivalent to conditions (ii<sup>′</sup>), (iii<sup>′</sup>) and $( \mathrm { i v } ^ { \prime } )$ above. This is easy to see: conditions (ii) and (iii) together are equivalent to conditions (ii<sup>′</sup>) and $( \mathrm { i i i ^ { \prime } } ) ;$ under conditions $( \mathrm { i } ) - ( \mathrm { i } \mathrm { i } )$ , condition (iv) is equivalent to condition (iv<sup>′</sup>). This concludes the proof of Theorem 5. □

## D.12 Proof of Lemma 2

Condition (i) can be verified by checking whether $S \setminus \{ s \}$ is still an f-separator of G for every $s \in S$ , which can be done clearly in polynomial time.

In view of Remark 2, we see that the decision problems for the remaining conditions of Theorem 5 can be solved in polynomial time.

## D.13 Proof of Lemma 3

Necessity is clear, since both p and v are degenerate.

To see suficiency, we first note that $\mathcal { H } ( f , S )$ is acyclic due to the existence of three such internally vertex-disjoint f-paths in $G ,$ since otherwise any cycle of $\mathcal { H } ( f , S )$ , whose vertices must not be separated from $f$ by S in G by condition (iii), has an edge $f ^ { \prime }$ whose two endvertices are not f-cut-vertices of $G ( S , s )$ for some $s \in S ( f ^ { \prime } ) = S$ , contradicting the definition of $\mathcal { H } ( f , S )$ . Let $P = ( { V _ { P } , E _ { P } } )$ be a path in $\mathcal { H } ( f , S )$ with $E _ { P } \neq \emptyset$ whose only degenerate vertices are its two endvertices. For the desired result, we just need to show that $P$ has only one edge, which shares one common vertex with $f$ and whose other endvertex is not separated from $f$ by S in $G$ and is connected to every vertex of $S$ by an interaction in $F .$ Indeed, it follows from condition (iii) that none of the vertices of $P$ is separated from $f$ by S in $G .$ Then, by the definition of $\mathcal { H } ( f , S )$ for any edge e of $P$ and any f-path $P ^ { \prime } = ( V _ { P ^ { \prime } } , E _ { P ^ { \prime } } )$ in $G$ that intersects $S$ in one vertex, we have $e \cap V _ { P ^ { \prime } } \neq \emptyset$ . However, as it is assumed that there exist at least three such f-paths that are internally vertex-disjoint, this implies that $e \cap f \neq \emptyset , { \mathrm { i . e . } }$ , e shares a common vertex with $f .$ Because the endvertices of $f$ are degenerate and the internal vertices of $P$ are assumed to be non-degenerate, $P$ has exactly one edge e. Condition (ii) requires that e neither intersects $S$ nor is a pair of f-cut-vertices of $G ,$ , and hence forces the other degenerate endvertex of $e ,$ the one that is not incident with $f ,$ to be connected to every vertex in the separator $S$ by an interaction.

## D.14 Proof of Lemma 4

Let $P = ( V _ { P } , E _ { P } )$ . By multiplying (7) by a factor 2 and rearranging, it can be written as

$$
( 2 x _ { f } - x _ { p } - x _ { q } ) \le \sum _ { u w \in E _ { P } } ( 2 x _ { u w } - x _ { u } - x _ { w } ) ,\tag{57}
$$

where $p$ and q are the two endvertices of $f .$ Note that, for any $x \in X _ { G F }$ , both $2 x _ { f } - x _ { p } - x _ { q }$ and $2 x _ { u w } - x _ { u } - x _ { w }$ for every $u w \in E _ { P }$ can only take values $0 , 1 , 2$ . To prove validity, we verify each possible case below.

I $2 x _ { f } - x _ { p } - x _ { q } = 0$ , then (57) is obviously satisfied. If $2 x _ { f } - x _ { p } - x _ { q } = 1$ , then it can only happen that $x _ { f } = 1$ and $x _ { p } \neq x _ { q }$ . In this case, the path $P$ contains an edge $u w \in F$ whose endvertices u and w take diferent values, hence $2 x _ { u w } - x _ { u } - x _ { w } = 1$ and (57) holds. If $2 x _ { f } - x _ { p } - x _ { q } = 2$ then the only possibility is that $x _ { f } = 1$ and $x _ { p } = x _ { q } = 0$ . In this case, the edges of $P$ cannot all take the value 0 by the definition of $X _ { G F }$ . From this we see that the path $P$ either possesses an edge uw $\in \ F$ with $x _ { u w } = 1 , x _ { u } = x _ { w } = 0$ , or at least two distinct edges $u _ { 1 } w _ { 1 } , u _ { 2 } w _ { 2 } \in F$ with $x _ { u _ { 1 } w _ { 1 } } = x _ { u _ { 2 } w _ { 2 } } = 1 , x _ { u _ { 1 } } \neq x _ { w _ { 1 } } , x _ { u _ { 2 } } \neq x _ { w _ { 2 } }$ , which again implies that (57) holds. Therefore, we have shown that inequality (57) is satisfied by all $x \in X _ { G F }$ , and hence (7) is valid.

## D.15 Proof of Theorem 6

For convenience, we fix a linear order on $P ,$ i.e. let

$$
V _ { P } = \{ v _ { 0 } , v _ { 1 } , \ldots , v _ { n } \}
$$

and

$$
E _ { P } = \left\{ e _ { 1 } = v _ { 0 } v _ { 1 } , \ldots , e _ { n } = v _ { n - 1 } v _ { n } \right\} .
$$

Hence $f = v _ { 0 } v _ { n }$ . We denote

$$
X ^ { \prime } = \left\{ x \in X _ { G F } \mid x _ { f } = \sum _ { e \in E { P } } x _ { e } - \sum _ { v \in V _ { P } \backslash f } x _ { v } \right\}
$$

as the set of all feasible vectors that satisfy (7) with equality.

(Necessity) Suppose that P has a chord $f ^ { \prime } = v _ { i } v _ { j }$ in $F \backslash \{ f \}$ , where $0 \leq i < j \leq n$ and $j \geq i + 2$ Then, we consider a second f-path $P ^ { \prime } = ( V _ { P ^ { \prime } } , E _ { P ^ { \prime } } )$ in $( V , F )$ obtained by replacing the part between

$v _ { i }$ and $v _ { j }$ with $f ^ { \prime } ,$ that is, $V _ { P ^ { \prime } } = \{ v _ { 0 } , \ldots , v _ { i } , v _ { j } , \ldots , v _ { n } \}$ and $E _ { P ^ { \prime } } = \{ e _ { 1 } , \dots , e _ { i } , e _ { j + 1 } , \dots , e _ { n } \} \cup \{ f ^ { \prime } \}$ By Lemma $^ { 4 , }$ it follows that the inequality

$$
x _ { f } \le x _ { f ^ { \prime } } + \sum _ { \substack { k \in \{ 1 , \dots , n \} \backslash \{ i + 1 , \dots , j \} } } x _ { e _ { k } } - \sum _ { \substack { l \in \{ 1 , \dots , n - 1 \} \backslash \{ i + 1 , \dots , j - 1 \} } } x _ { v _ { l } }\tag{58}
$$

is valid for $\Xi _ { G F }$ . Now we notice that (7) is simply a sum of (58) and the path inequality

$$
x _ { f ^ { \prime } } \leq \sum _ { k = i + 1 } ^ { j } x _ { e _ { k } } - \sum _ { l = i + 1 } ^ { j - 1 } x _ { v _ { l } } .
$$

Consequently, inequality (7) does not define a facet of $\Xi _ { G F }$ under the given condition.

(Suficiency) Assume that the path $P$ is chordless in the graph $( V , F \setminus \{ f \} )$ . In other words, there is no interaction $f ^ { \prime } \in F \setminus ( E _ { P } \cup \{ f \} )$ such that $f ^ { \prime } \subseteq V _ { P }$

Looking at this inequality, we observe that, for any vertex subset U whose intersection with $V _ { P }$ is either empty or induces a path in P that also contains an endvertex of $P _ { \mathrm { : } }$ the induced feasible vector $x ^ { U }$ satisfies this inequality tightly and hence belongs to $X ^ { \prime }$ . To see this more explicitly, as the case when either $U \cap V _ { P } = \emptyset { \mathrm { ~ o r ~ } } V _ { P } \subseteq U$ is obvious, suppose without loss of generality that $U \cap V _ { P } = \left\{ v _ { 0 } , . . . , v _ { h } \right\}$ for some $h \in \{ 0 , \ldots , n - 1 \}$ . Then, $\begin{array} { r } { \begin{array} { r } { x _ { f } ^ { U } = 1 , \sum _ { e \in E _ { P } } x _ { e } ^ { U } = \bar { n } - h } \end{array} } \end{array}$ and $\begin{array} { r } { \sum _ { v \in V _ { P } \backslash f } x _ { v } ^ { U } = n - h - 1 } \end{array}$ , and hence the observation follows.

In the following, we construct the associated standard unit vector for each vertex that is not in the interior of $P$ and each interaction that is not in $E _ { P } \cup \{ f \}$

Firstly, let v be a vertex that is not in the interior of P. Since the all-one vector $x ^ { \alpha } = \mathbb { 1 } _ { V \cup F }$ and $x ^ { \{ v \} }$ belong to $X ^ { \prime }$ by the observation above, we have

$$
\mathbb { 1 } _ { \{ v \} } = \mathbb { 1 } _ { V \cup F } - x ^ { \{ v \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) .
$$

Secondly, let $f ^ { \prime } = s t \in F$ be an interaction such that neither of the endvertices of $f ^ { \prime }$ is in $V _ { P }$ If $V _ { P }$ is not an f<sup>′</sup>-separator of $G$ , we can choose a chordless $f ^ { \prime } .$ -path $P ^ { \prime }$ in $G$ such that the two vertex sets of $P$ and $P ^ { \prime }$ are disjoint. Set

$$
A = V _ { P ^ { \prime } } \backslash \{ t \} , \quad B = V _ { P ^ { \prime } } \backslash \{ s \} .
$$

Then $F _ { A B } = \{ f ^ { \prime } \}$ and $H _ { A B } = \mathcal { O }$ . By (23) of Lemma $^ { 7 , }$ we get

$$
\mathbb { 1 } _ { \{ f ^ { \prime } \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) ,
$$

since $x ^ { A } , x ^ { B } , x ^ { A \cap B }$ and $x ^ { A \cup B }$ are in $X ^ { \prime }$ by the observation above. Otherwise, $V _ { P }$ is an $f ^ { \prime } .$ -separator of $G .$ . Then, we take a chordless $\{ s , V _ { P } \}$ -path $P ^ { \prime }$ and a chordless $\{ t , V _ { P } \}$ -path $P ^ { \prime \prime }$ in $G$ and set

$$
U = V _ { P } \cup V _ { P ^ { \prime } } \cup V _ { P ^ { \prime \prime } } , \quad A = U \setminus \left\{ s \right\} , \quad B = U \setminus \left\{ t \right\} .
$$

As $V _ { P }$ is an $f ^ { \prime } .$ -separator of $G ,$ none of $P ^ { \prime }$ and $P ^ { \prime \prime }$ contains an interior vertex of another. Thus, $F _ { A B } = \{ f ^ { \prime } \}$ and $H _ { A B } = \mathcal { O }$ . From (23) of Lemma 7 and the fact that $x ^ { A } , x ^ { B } , x ^ { A \cap B }$ and $x ^ { A \cup B }$ are in $X ^ { \prime }$ by the above observation, it follows that

$$
\mathbb { 1 } _ { \{ f ^ { \prime } \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) .
$$

An example is depicted in Figure 18 a).

Thirdly, we will consider all interactions with one endvertex in $V _ { P }$ . To that end, let $s \in V \setminus V _ { P }$ be an arbitrary but fixed vertex and let $P ^ { \prime }$ be a chordless {s, V<sub>P</sub>}-path in G. Let u be the unique vertex in $V _ { P ^ { \prime } } \cap N _ { G } ( V _ { P } )$ and let $n ^ { - }$ (respectively, $n ^ { + } )$ denote the minimal (respectively, maximal) index i such that $v _ { i }$ is adjacent to u in $G .$ For any $f ^ { \prime } = s v _ { m } \in F$ with $m < n ^ { + }$ , we set

$$
U = ( V _ { P ^ { \prime } } \setminus V _ { P } ) \cup \{ v _ { l } \in V _ { P } \mid l \geq m \} , \quad A = U \setminus \{ s \} , \quad B = U \setminus \{ v _ { m } \} .
$$

![](images/131645dd3f60981660700bf8ad4f01f0ddd647a3f2a75b35de98271cac0d196e.jpg)  
Figure 18: Examples of the constructions in the proof of Theorem 6. Each subfigure consists of a graph G (solid black) , a path $P$ on the vertex set $\{ v _ { 0 } , v _ { 1 } , v _ { 2 } , v _ { 3 } , v _ { 4 } \}$ , and interactions (dashed green). In $\mathrm { a ) }$ , the paths induced by the edges $s v _ { 1 }$ and $t v _ { 3 }$ can be chosen as $P ^ { \prime }$ and $P ^ { \prime \prime }$ respectively; in b), the path induced by any of the edges sv<sub>0</sub>, sv<sub>1</sub>, sv<sub>3</sub> can be chosen as $P ^ { \prime }$

It can be checked that $F _ { A B } = \{ f ^ { \prime } \}$ and $H _ { A B } = \mathcal { O }$ . By (23) of Lemma 7, we obtain

$$
\mathbb { 1 } _ { \{ f ^ { \prime } \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) ,
$$

since $x ^ { A } , x ^ { B } , x ^ { A \cap B }$ and $x ^ { A \cup B }$ are in $X ^ { \prime }$ by the observation above. By a symmetric argument, it can be obtained that the associated standard unit vector $\mathbb { 1 } _ { \left\{ f ^ { \prime } \right\} }$ also lies in lin $( X ^ { \prime } - X ^ { \prime } )$ for the case $f ^ { \prime } = s v _ { m } \in F$ with $m > n ^ { - }$ . An example is shown in Figure 18 b). Now, for the remaining case $f ^ { \prime } = s v _ { m } \in F$ with $m = n ^ { - } = n ^ { + }$ , we set

$$
A = ( V _ { P } \cup V _ { P ^ { \prime } } ) \backslash \{ s \} , \quad B = V _ { P } \cup V _ { P ^ { \prime } } .
$$

Then, $Q _ { A B } = \{ s t ^ { \prime } \in F \mid t ^ { \prime } \in A \} \cup \{ s \}$ . From the above observation, $x ^ { A } , x ^ { B } , x ^ { A \cap B }$ and $x ^ { A \cup B }$ are all in $X ^ { \prime }$ . It then follows from (22) of Lemma 7 that $\mathbb { 1 } _ { Q _ { A B } } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$ . Moreover, the previous discussion shows that the standard unit vector associated with every element in $Q _ { A B }$ lies in lin $( X ^ { \prime } - X ^ { \prime } )$ except for $f ^ { \prime } .$ . Therefore, it holds that

$$
\mathbb { 1 } _ { \{ f ^ { \prime } \} } = \mathbb { 1 } _ { Q _ { A B } } - \sum _ { g \in Q _ { A B } \backslash \{ f ^ { \prime } \} } \mathbb { 1 } _ { \{ g \} } \in \mathrm { l i n } ( X ^ { \prime } - X ^ { \prime } ) .
$$

Since the vertex s has been chosen arbitrarily we have $\mathbb { 1 } _ { \{ f ^ { \prime } \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$ for all $f ^ { \prime } \in F$ with exactly one endvertex in $V _ { P }$

By the previous discussion and our assumption that P is chordless in the graph $( V , F \setminus \{ f \} )$ we immediately see that the standard unit vector associated with every interaction that is not in $E _ { P } \cup \{ f \}$ is contained in lin $( X ^ { \prime } - X ^ { \prime } )$ .

Next, for every interior vertex v of $P$ and every edge e of $P _ { \mathrm { : } }$ we show that both the vectors $\mathbb { 1 } _ { \{ e \} } + \mathbb { 1 } _ { \{ f \} }$ and $\mathbb { 1 } _ { \{ v \} } - \mathbb { 1 } _ { \{ f \} }$ are elements of lin $( X ^ { \prime } - X ^ { \prime } )$ . In view of the preceding paragraph, for our purpose, it is suficient to consider the special case $F = E _ { P } \cup \{ f \}$ . Let us first deal with an edge e of $P$ . Suppose $e = e _ { k }$ for some $k \in \{ 1 , \ldots , n \}$ . Note that both $\ddot { x } ^ { \{ v _ { l } \in V _ { P } \vert l < k \} }$ and $x ^ { \{ v _ { l } \in V _ { P } | l \geq k \} }$ are members of $X ^ { \prime }$ by the observation above. Then, we have

$$
\mathbb { 1 } _ { \{ e _ { k } \} } + \mathbb { 1 } _ { \{ f \} } = x ^ { \{ v _ { l } \in V _ { P } | l < k \} } + x ^ { \{ v _ { l } \in V _ { P } | l \geq k \} } - \mathbb { 1 } _ { V \cup F } - \mathbb { 1 } _ { V \setminus V _ { P } } .
$$

It lies in lin $( X ^ { \prime } - X ^ { \prime } )$ since both 1 $V \cup F$ and $\mathbb { 1 } _ { V \backslash V _ { P } }$ belong to $X ^ { \prime }$ by the observation above. Next, let $v = v _ { m }$ be an interior vertex of $P$ for some $\dot { m } \in \{ 1 , \dots , n - 1 \}$ . As the vectors $\mathbb { 1 } _ { \{ e \} } + \mathbb { 1 } _ { \{ f \} }$ are known to be in lin $( X ^ { \prime } - X ^ { \prime } )$ for all $e \in E _ { P }$ , we can iteratively calculate $\mathbb { 1 } _ { \{ v \} } - \mathbb { 1 } _ { \{ f \} }$ from $m = 1$ to $m = n - 1$ by

$$
\mathbb { 1 } _ { \{ v _ { m } \} } - \mathbb { 1 } _ { \{ f \} } = \mathbb { 1 } _ { V \cup F } - x ^ { \{ v _ { l } \in V _ { F } | l \leq m \} } - \mathbb { 1 } _ { \{ v _ { 0 } \} } - \sum _ { 1 \leq l < m } \left( \mathbb { 1 } _ { \{ v _ { l } \} } - \mathbb { 1 } _ { \{ f \} } \right) - \sum _ { 1 \leq l \leq m } \left( \mathbb { 1 } _ { \{ e _ { l } \} } + \mathbb { 1 } _ { \{ f \} } \right) ,
$$

which also lies in lin $( X ^ { \prime } - X ^ { \prime } )$ for every $m \in \{ 1 , \ldots , n - 1 \}$

Thus, we have shown that the linear space lin $( X ^ { \prime } - X ^ { \prime } )$ has a basis

$$
\begin{array} { r l } & { \quad \left\{ \mathbb { 1 } _ { \{ v \} } \ : | \ : v \in \left( V \setminus V _ { P } \right) \cup f \right\} } \\ & { \cup \left\{ \mathbb { 1 } _ { \{ f ^ { \prime } \} } \ : | \ : f ^ { \prime } \in F \setminus \left( E _ { P } \cup \{ f \} \right) \right\} } \\ & { \cup \left\{ \mathbb { 1 } _ { \{ v \} } - \mathbb { 1 } _ { \{ f \} } \ : | \ : v \in V _ { P } \setminus f \right\} } \\ & { \cup \left\{ \mathbb { 1 } _ { \{ e \} } + \mathbb { 1 } _ { \{ f \} } \ : | \ : e \in E _ { P } \right\} , } \end{array}
$$

which implies that it has codimension 1, and hence so does the face $\operatorname { a f f } ( X ^ { \prime } )$ . Since the polytope $\Xi _ { G F }$ is full-dimensional by Proposition 2, we conclude that the path inequality (7) is facet-defining for $\Xi _ { G F }$ under the specified assumption. □

## D.16 Proof of Lemma 5

Necessity. Assume that the intersection inequality is valid for $\Xi _ { G F }$ . Without loss of generality, we may assume that $\{ v _ { 2 } , w _ { 2 } \}$ is not a $\{ v _ { 1 } , w _ { 1 } \}$ -separator of $G .$ Then, there exists a vertex subset $U$ such that $v _ { 1 } , w _ { 1 } \in U$ and v , w $\notin U$ . Thus, we have $x _ { v _ { 1 } w _ { 2 } } ^ { U } = 1$ and $x _ { v _ { 2 } w _ { 1 } } ^ { U } = 1$ . Since $x _ { v _ { 1 } w _ { 1 } } ^ { U } = 0$ the vector $x ^ { U }$ violates the intersection inequality, which contradicts our assumption.

Suficiency. Assume that $\{ v _ { 1 } , w _ { 1 } \}$ is a $\{ v _ { 2 } , w _ { 2 } \}$ -separator of G and $\{ v _ { 2 } , w _ { 2 } \}$ is a $\{ v _ { 1 } , w _ { 1 } \} .$ separator of G. Let $x \in X _ { G F }$ be an arbitrary feasible vector. We need to show that the intersection inequality is satisfied by x. Let us first consider the following two cases.

1. Case $( \mathrm { a } ) \colon x _ { v _ { 1 } w _ { 1 } } = x _ { v _ { 2 } w _ { 2 } } = 0$ . This implies the existence of a vertex subset U such that $x | _ { U } = 0$ and U is both a $\{ v _ { 1 } , w _ { 1 } \} .$ - and a $\{ v _ { 2 } , w _ { 2 } \}$ -connector of G. Since $\{ w _ { 1 } , w _ { 2 } \} \in E$ , the set U is also a $\{ v _ { 1 } , w _ { 2 } \} .$ - and a $\{ v _ { 2 } , w _ { 1 } \}$ -connector of $G .$ Consequently, $x _ { v _ { 1 } w _ { 2 } } = 0$ and $x _ { v _ { 2 } w _ { 1 } } = 0$ , implying that x satisfies the intersection inequality.

2. Case (b): $x _ { v _ { 1 } w _ { 1 } } = 0$ and $x _ { v _ { 2 } w _ { 2 } } = 1$ . In this case, there exists a vertex subset U such that $x | _ { U } = 0$ and U is a $\{ v _ { 1 } , w _ { 1 } \}$ -connector of G. Since both $\{ v _ { 1 } , v _ { 2 } \}$ and $\{ v _ { 2 } , w _ { 2 } \}$ are in $E _ { : }$ the set $U \cup \{ v _ { 2 } , w _ { 2 } \}$ is also a {v<sub>2</sub>, w<sub>2</sub>}-connector of G. Thus, if $x _ { v _ { 2 } } = x _ { w _ { 2 } } = 0$ , then $x _ { v _ { 2 } w _ { 2 } } = 0$ contradicting our assumption. Hence, at least one of $x _ { v _ { 2 } }$ and $x _ { w _ { 2 } }$ takes the value 1. Note that, $x _ { v _ { 2 } }$ and $x _ { w _ { 2 } }$ cannot both be 1. Otherwise, since $\{ v _ { 2 } , w _ { 2 } \}$ is a $\{ v _ { 1 } , w _ { 1 } \}$ -separator of $G _ { \ l }$ , we would have $x _ { v _ { 1 } w _ { 1 } } = 1$ , contradicting our assumption. Therefore, exactly one of $x _ { v _ { 2 } }$ and $x _ { w _ { 2 } }$ takes the value 1, which implies that either $x _ { v _ { 1 } w _ { 2 } } = 0 \ \mathrm { o r } \ x _ { v _ { 2 } w _ { 1 } } = 0$ , and hence the intersection inequality is satisfied by x.

Note that the case $x _ { v _ { 1 } w _ { 1 } } = 1$ and $x _ { v _ { 2 } w _ { 2 } } = 0$ can be handled by a symmetric argument, and the case $x _ { v _ { 1 } w _ { 1 } } = x _ { v _ { 2 } w _ { 2 } } = 1$ is trivial. Thus, all possible cases have been considered, and we have shown that the intersection inequality is valid for $\Xi _ { G F }$ under the given condition. □

## D.17 Proof of Theorem 7

Necessity. This follows immediately from Lemma 5.

Suficiency. Assume that $\{ v _ { 1 } , w _ { 1 } \}$ is a $\{ v _ { 2 } , w _ { 2 } \}$ }-separator of G and $\{ v _ { 2 } , w _ { 2 } \}$ is a $\{ v _ { 1 } , w _ { 1 } \}$ separator of G. Under these assumptions, we construct a basis of the linear space lin( $X ^ { \prime } - X ^ { \prime } )$ where $X ^ { \prime }$ is the set of all feasible vectors that satisfy the intersection inequality with equality.

Let us consider an arbitrary vertex v. Set $A = \emptyset$ and $B = \{ v \}$ . Then $x ^ { \bar { A } } , x ^ { B } , x ^ { A \cap B } , x ^ { \bar { A } \cup B } \in X ^ { \prime }$ because no singleton vertex set connects any of the four interactions appearing in the intersection inequality. Moreover, $Q _ { A B } = Q _ { \varnothing , \{ v \} } = \{ v \}$ . Thus, by (22) of Lemma 7, we have $\mathbb { 1 } _ { \{ v \} } \in \operatorname* { l i n } ( X ^ { \prime } { - } X ^ { \prime } )$

We first show that the following two conditions cannot both hold:

(a) $\{ v _ { 1 } , w _ { 2 } \}$ is an $\{ v _ { 2 } , w _ { 1 } \}$ -separator of G.

(b) $\left\{ v _ { 2 } , w _ { 1 } \right\}$ is an $\{ v _ { 1 } , w _ { 2 } \}$ }-separator of G.

Indeed, if both conditions hold, our initial separator assumptions together with the fact that $\{ v _ { 1 } , v _ { 2 } \} , \{ w _ { 1 } , w _ { 2 } \} \in E$ imply that any minimal $\{ v _ { 1 } , w _ { 2 } \}$ -connector $C _ { 1 }$ of G must contain $\{ v _ { 2 } , w _ { 1 } \}$ and symmetrically, any minimal $\{ v _ { 2 } , w _ { 1 } \}$ -connector $C _ { 2 }$ of $G$ must contain $\{ v _ { 1 } , w _ { 2 } \}$ . Consequently, the vertex subset $C _ { 1 } \setminus \{ v _ { 1 } , w _ { 2 } \}$ is a minimal $\{ v _ { 2 } , w _ { 1 } \}$ -connector of G disjoint from $\{ v _ { 1 } , w _ { 2 } \}$ , and $C _ { 2 } \backslash \{ v _ { 2 } , w _ { 1 } \}$ is a minimal $\{ v _ { 1 } , w _ { 2 } \}$ -connector of G disjoint from $\{ v _ { 2 } , w _ { 1 } \}$ . This yields a contradiction. By symmetry, we may assume henceforth that $\{ v _ { 1 } , w _ { 2 } \}$ is not $\mathrm { ~ a ~ } \{ v _ { 2 } , w _ { 1 } \}$ -separator of G. Let $C ^ { \prime }$ denote the union of $\{ v _ { 1 } , w _ { 2 } \}$ and a minimal $\{ v _ { 2 } , w _ { 1 } \}$ -connector of G disjoint from $\{ v _ { 1 } , w _ { 2 } \}$ . By our initial assumption and the fact that $\{ v _ { 1 } , v _ { 2 } \} , \{ w _ { 1 } , w _ { 2 } \} \in E$ , the induced subgraph $G [ C ^ { \prime } ]$ is either a chordless path, ${ \mathrm { i f ~ } } v _ { 1 } w _ { 2 } \not \in E ,$ or a chordless cycle, if $v _ { 1 } w _ { 2 } \in E$

(1)  
![](images/4bde4fcefa097bc7a328e9a4fc067a20991dcb4da70a121f8aa8bfa399eac7d1.jpg)

(2)  
![](images/8c1329d66b6b0344e3ca714c24ea8baa488ba32731677e4624c30624732ba3b4.jpg)  
(4)

(3)  
![](images/e3e0ba2f77392bb9ccc8dbf829eb528da12f1eca5c5889c338737d9a88b95f25.jpg)

![](images/6486a7cea776befa7e0c26ea3d471eb02a73142f9c76e7120401bf6dcdbb0f7e.jpg)  
Figure 19: Depicted are graph (solid black) and interactions (dashed). In each subfigure, the green interactions define an intersection equality. For each red interaction, its corresponding standard unit vector is constructed in the proof of Theorem 7.

For any interaction $f = \{ s , t \} \in F$ that is not separated by $\{ v _ { 1 } , v _ { 2 } \}$ (resp. $\{ w _ { 1 } , w _ { 2 } \} )$ in $G ,$ we choose a minimal f-connector $C$ of G such that $C \cap \left\{ v _ { 1 } , v _ { 2 } \right\} = \emptyset$ (resp. $C \cap \left\{ w _ { 1 } , w _ { 2 } \right\} = \emptyset )$ . Set $A = C \setminus \{ s \}$ and $B = C \backslash \{ t \}$ , then $F _ { A B } = \{ f \}$ and $H _ { A B } = \mathcal { O }$ . Moreover, $x ^ { A } , x ^ { \tilde { B } } , x ^ { A \cap \tilde { B } }$ and $x ^ { A \cup B }$ are all in $X ^ { \prime } .$ For example, if $C \cap \left\{ v _ { 1 } , v _ { 2 } \right\} = \emptyset$ , then every set among A, B, A ∩ B and A ∪ B is also disjoint from $\{ v _ { 1 } , v _ { 2 } \}$ . Hence none of the four special interactions can be connected, because each of them has one endpoint in $\{ v _ { 1 } , v _ { 2 } \}$ . Therefore all four special interaction variables are equal to 1, and the intersection inequality holds at equality: $1 + 1 = 1 + 1$ . By (23) of Lemma 7, we have

$$
\mathbb { 1 } _ { \{ f \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) .
$$

See Figure 19-(1) for an example.

Let $f = \{ s , t \} \in F$ be an interaction that is separated by both $\{ v _ { 1 } , v _ { 2 } \}$ and $\{ w _ { 1 } , w _ { 2 } \}$ in $G ,$ and satisfies $f \not \subseteq \{ v _ { 1 } , v _ { 2 } , w _ { 1 } , w _ { 2 } \}$ . We distinguish two possible cases depending on whether the two sets $f$ and $\{ v _ { 1 } , v _ { 2 } , w _ { 1 } , w _ { 2 } \}$ intersect.

First, suppose f does not intersect $\{ v _ { 1 } , v _ { 2 } , w _ { 1 } , w _ { 2 } \}$ ; then $f$ is also disjoint from $C ^ { \prime }$ . Hence, we can find a minimal $\{ \{ s \} , C ^ { \prime } \}$ -connector $U _ { 1 }$ of G and a minimal $\{ \{ t \} , C ^ { \prime } \}$ -connector $U _ { 2 }$ of G. Set $A = ( U _ { 1 } \cup U _ { 2 } \cup C ^ { \prime } ) \setminus \{ s \}$ and $B = ( U _ { 1 } \cup U _ { 2 } \cup C ^ { \prime } ) \backslash \{ t \}$ . By construction, $F _ { A B } = \{ f \}$ and $H _ { A B } = \mathcal { O }$ Indeed, the set $A \cup B$ contains an st-connector, whereas A misses s and B misses t. Thus $f \in F _ { A B }$ No other interaction belongs to $F _ { A B }$ , because every other interaction contained in $A \cup B$ remains connected in at least one of A or $B .$ Moreover, deleting s and t simultaneously does not destroy any connection among endpoints lying in $A \cap B ,$ , so $H _ { A B } = \mathcal { O }$ . By (23) of Lemma 7, we obtain

$$
\mathbb { 1 } _ { \{ f \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) ,
$$

since $x ^ { A } , x ^ { B } , x ^ { A \cap B }$ and $x ^ { A \cup B }$ are in $X ^ { \prime }$ . See Figure 19-(2) for an illustration.

Next, we consider the case where $f$ intersects $\{ v _ { 1 } , v _ { 2 } , w _ { 1 } , w _ { 2 } \}$ . Suppose that t is the only element in the intersection of $f$ and $\{ v _ { 1 } , v _ { 2 } , w _ { 1 } , w _ { 2 } \}$ , and let U be a minimal $\{ s , C ^ { \prime } \}$ -connector of G. If $t \in \{ v _ { 1 } , w _ { 2 } \}$ , we define $A = ( U \cup C ^ { \prime } ) \setminus \{ s \}$ and $B = ( U \cup C ^ { \prime } ) \setminus \{ t \}$ . A straightforward verification like above shows that $F _ { A B } = \{ f \}$ and $H _ { A B } = \mathcal { O }$ . By (23) of Lemma $^ { 7 , }$ we obtain

$$
\mathbb { 1 } _ { \{ f \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) ,
$$

since $x ^ { A } , x ^ { B } , x ^ { A \cap B }$ and $x ^ { A \cup B }$ are in $X ^ { \prime }$ . See Figure 19-(3) for an illustration. Symmetrically, if $t \in \{ v _ { 2 } , w _ { 1 } \} \ ( \mathrm { s a y } , t = v _ { 2 } )$ , we set $A = ( U \cup C ^ { \prime } ) \setminus \{ s , v _ { 1 } \}$ and $B = ( U \cup C ^ { \prime } ) \setminus \{ t , v _ { 1 } \}$ . It follows that

$F _ { A B } = \{ f \}$ and $H _ { A B } = \mathcal { O }$ . Applying (23) of Lemma 7 yields

$$
\mathbb { 1 } _ { \{ f \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } ) ,
$$

since $x ^ { A } , x ^ { B } , x ^ { A \cap B } , x ^ { A \cup B }$ are in $X ^ { \prime }$ . This case is illustrated in Figure $1 9 \ – ( 4 )$

Finally, we consider the remaining elements $\{ v _ { 1 } , w _ { 2 } \} , \{ v _ { 2 } , w _ { 1 } \} , \{ v _ { 1 } , w _ { 1 } \}$ , and $\{ v _ { 2 } , w _ { 2 } \}$ . One can readily verify that the following equalities hold:

$$
\mathbb { 1 } _ { \{ v _ { 1 } w _ { 1 } , v _ { 1 } w _ { 2 } \} } = x ^ { C ^ { \prime } \setminus \{ v _ { 1 } \} } - x ^ { C ^ { \prime } } - \mathbb { 1 } _ { \{ v _ { 1 } \} } - \sum _ { \substack { v _ { 1 } v \in F : v \in C ^ { \prime } \setminus \{ w _ { 1 } , w _ { 2 } \} } } \mathbb { 1 } _ { \{ v _ { 1 } v \} } ,
$$

$$
\mathbb { 1 } _ { \{ v _ { 2 } w _ { 2 } , v _ { 1 } w _ { 2 } \} } = x ^ { C ^ { \prime } \backslash \{ w _ { 2 } \} } - x ^ { C ^ { \prime } } - \mathbb { 1 } _ { \{ w _ { 2 } \} } - \sum _ { \substack { w _ { 2 } w \in F : w \in C ^ { \prime } \backslash \{ v _ { 1 } , v _ { 2 } \} } } \mathbb { 1 } _ { \{ w _ { 2 } w \} } ,
$$

and

$$
\mathbb { 1 } _ { \{ v _ { 2 } w _ { 2 } , v _ { 2 } w _ { 1 } \} } = x ^ { C ^ { \prime } \setminus \{ v _ { 1 } , v _ { 2 } \} } - x ^ { C ^ { \prime } \setminus \{ v _ { 1 } \} } - \mathbb { 1 } _ { \{ v _ { 2 } \} } - \sum _ { \substack { v _ { 2 } v \in F : v \in C ^ { \prime } \setminus \{ w _ { 1 } , w _ { 2 } \} } } \mathbb { 1 } _ { \{ v _ { 2 } v \} } .
$$

These three vectors all belong to lin $( X ^ { \prime } - X ^ { \prime } )$ , since the zero vector $x ^ { V }$ and the vectors on the right-hand sides of the equalities above are in $X ^ { \prime }$

By all the results above, we see that the linear space lin $( X ^ { \prime } - X ^ { \prime } )$ has a basis

$$
\{ \mathbb { 1 } _ { \{ g \} } | g \in \left( V \cup F \right) \setminus \left\{ v _ { 1 } w _ { 2 } , v _ { 2 } w _ { 1 } , v _ { 1 } w _ { 1 } , v _ { 2 } w _ { 2 } \right\} \} \cup \left\{ \mathbb { 1 } _ { \{ v _ { 1 } w _ { 1 } , v _ { 1 } w _ { 2 } \} } , \mathbb { 1 } _ { \{ v _ { 2 } w _ { 2 } , v _ { 1 } w _ { 2 } \} } , \mathbb { 1 } _ { \{ v _ { 2 } w _ { 2 } , v _ { 2 } w _ { 1 } \} } \right\} .
$$

Since $\Xi _ { G F }$ is full-dimensional by Proposition 2, we conclude that the intersection inequality (8) defines a facet of $\Xi _ { G F }$ under the given conditions. □

## D.18 Proof of Lemma 6

We first prove (i). Let $x \in X _ { G F }$ , and define $z \in \{ 0 , 1 \} ^ { V \cup F }$ by $z _ { v } = 1 - x _ { v }$ for all $v \in V$ and $z _ { u v } = z _ { u } z _ { v }$ for all $u v \in F$ . Then z is a vertex of $\mathrm { B Q P } _ { ( V , F ) }$

For every interaction uv $\in F$ , if $x _ { u v } = 0$ , then u and v are connected in $G [ x ^ { - 1 } ( 0 ) \cap V ]$ , and in particular $x _ { u } = x _ { v } = 0$ . Hence $z _ { u } = z _ { v } = 1$ , so $z _ { u v } = 1$ . Therefore $1 - x _ { u v } \le z _ { u v }$ for all $u v \in F$ Moreover, if uv $\in E .$ then u and v are connected in $G [ x ^ { - 1 } ( 0 ) \cap V ]$ if and only if $x _ { u } = x _ { v } = 0$ Thus $1 - x _ { u v } = z _ { u v }$ for all $u v \in E$ . Since $a _ { u v } \ge 0$ for all uv $\in F \setminus E$ , we obtain $a ( 1 - x ) \leq a z$ . By validity of $a z \leq \alpha$ for $\mathrm { B Q P } _ { ( V , F ) }$ , we have $a z \leq \alpha$ . Hence $a ( 1 - x ) \leq \alpha$ . This proves (i).

The proof of part (ii) is analogous. Let z be a vertex of $\mathrm { B Q P } _ { ( V , F ) }$ , and define $U = \{ v \in V \mid$ $z _ { v } = 1 \}$ . Let $x = x ^ { U } \in X _ { G F }$ . Then $1 - x _ { u v } \le z _ { u v }$ for every $u v \in F ,$ , with equality for $u v \in E$ Since $a _ { u v } \leq 0$ for all uv $\in F \setminus E$ , it follows that $a z \leq a ( 1 - x )$ . By validity of $a ( 1 - x ) \leq \alpha$ for $\Xi _ { G F }$ , we obtain $a z \leq \alpha$ . Thus $a z \leq \alpha$ is valid for $\mathrm { B Q P } _ { ( V , F ) }$

It remains to prove part (iii). Assume that $\{ f \in F \mid a _ { f } ^ { \prime } \neq 0 \} \subseteq E \subseteq F$ and that $a ( 1 - x ) \leq \alpha$ is facet-defining for $\Xi _ { G F }$ . Note that the inequality $a | _ { V \cup E } z \leq \alpha$ is valid for $\mathrm { B Q P } _ { G }$ . Thus we only need to show that the face defined by $a | _ { V \cup E } z \leq \alpha$ has dimension $| V | + | E | - 1$

Let $X ^ { \prime } = \{ x \in X _ { G F } \mid a ( 1 - x ) = \alpha \}$ be the set of feasible multi-separator vectors in the corresponding face. Since $\Xi _ { G F }$ is full-dimensional and $a ( 1 - x ) \leq$ α defines a facet, we have dim lin $( X ^ { \prime } - X ^ { \prime } ) = | V | + | F | - 1$ . Moreover, $\boldsymbol a _ { f } = 0$ for every $f \in F \setminus E$ . Hence $\mathbb { 1 } _ { \{ f \} } \in \operatorname* { l i n } ( X ^ { \prime } - X ^ { \prime } )$ for $f \in F \setminus E$

Let $Z ^ { \prime } = \{ z \in \mathrm { v e r t } ( \mathrm { B Q P } _ { G } ) \mid a | _ { V \cup E } z = \alpha \}$ . Since $E \subseteq F$ , the afine map defined by $z _ { v } = 1 - x _ { v }$ for $v \in V$ and $z _ { e } = 1 - x _ { e }$ for $e \in E$ identifies the projection of $X ^ { \prime }$ onto the subsapce $\mathbb { R } ^ { V \cup E }$ with $Z ^ { \prime }$ . Therefore, l $\mathrm { i n } ( X ^ { \prime } - X ^ { \prime } ) \cong \mathrm { l i n } ( Z ^ { \prime } - Z ^ { \prime } ) \oplus \mathbb { R } ^ { F \setminus \bar { E } }$ , and hence

$$
\operatorname { d i m } \operatorname { c o n v } Z ^ { \prime } = \dim \operatorname { l i m } ( Z ^ { \prime } - Z ^ { \prime } ) = \dim \operatorname { l i m } ( X ^ { \prime } - X ^ { \prime } ) - | F \setminus E | = | V | + | E | - 1 ,
$$

Since $\mathrm { B Q P } _ { G }$ is full-dimensional in $\mathbb { R } ^ { V \cup E }$ , the inequality $a | _ { V \cup E } z \leq \alpha$ defines a facet of $\mathrm { B Q P } _ { G }$ .

## D.19 Proof of Proposition 6

Let

$$
X _ { \widehat { G } F } ^ { \prime } = \{ x \in X _ { \widehat { G } F } \ | \ \forall v \in V \colon x _ { v } = 0 \land \forall e \in E \colon x _ { e } = x _ { v _ { e } } \}
$$

denote the set of all feasible vectors in $X _ { \widehat { G } F }$ that vanish on V and whose e- and v -components take the same value for each $e \in E$ . Since all inequalities $x _ { v } \geq 0$ with $v \in V$ are valid for $x \in X _ { \widehat { G } F }$ and $x _ { e } \le x _ { v _ { e } }$ are valid for $x \in X _ { \widehat { G } F }$ with $x | _ { V } = 0$ , the convex hull $\Xi _ { \widehat { G } F } ^ { \prime } : = \mathrm { c o n v } X _ { \widehat { G } F } ^ { \prime }$ is a face of the multi-separator polytope $\Xi _ { \widehat { G } F } .$

We claim that the lifted multicut polytope with respect to $G$ and $G ^ { \prime }$ is the image of the face $\Xi _ { \widehat { G } F } ^ { \prime }$ under the projection proj ${ \bf \Psi } _ { F } \colon \mathbb { R } ^ { \widehat { V } \cup F }  \mathbb { R } ^ { F }$ which maps $( x _ { 1 } , x _ { 2 } ) \in \mathbb { R } ^ { \widehat { V } } \times \mathbb { R } ^ { F }$ to $x _ { 2 } ,$ , i.e.,

$$
\Gamma _ { G G ^ { \prime } } = \mathrm { p r o j } _ { F } \Xi _ { \widehat { G } F } ^ { \prime } .
$$

For this purpose, it is enough to show that $Y _ { G G ^ { \prime } } = \mathrm { p r o j } _ { F } X _ { \widehat { G } F } ^ { \prime } ,$ since the operations of constructing the convex hull and taking the projection commute with each other.

To establish $\operatorname { p r o j } _ { F } X _ { \widehat { G } F } ^ { \prime } \subseteq Y _ { G G ^ { \prime } }$ , let $x \in X _ { \widehat { G } F } ^ { \prime }$ and let $f \in F$ , we show that those of the inequalities $( 1 1 ) \textrm { -- } ( 1 3 )$ associated with $f$ for $Y _ { G G ^ { \prime } }$ can be obtained from the connector and separator inequalities associated with $f$ for $X _ { \widehat { G } F } .$ . Suppose H is a cycle containing $f$ in $G { \mathrm { ~ i f ~ } } f \in E$ or an f-path in G if $f \in F \setminus E$ . By Remark 9, H can be assumed to be chordless in G. No matter which is the case, the vertex set of $\widehat { H }$ is always a minimal f-connector of $\widehat { G }$ by (i) of Corollary 1. In this way, each chordless cycle or path inequality associated with $f$ for $Y _ { G G ^ { \prime } }$ corresponds to a minimal connector inequality associated with $f$ for $X _ { \widehat { G } F }$ . Using this fact and the definition of $X _ { \widehat { \mathcal { O } } _ { E } } ^ { \prime } ,$ GFb it is straightforward to see that x satisfies the cycle inequalities (11) and the path inequalities (12) for $Y _ { G G ^ { \prime } }$ . Furthermore, since every minimal f-connector of ${ \widehat { G } } ,$ except $f \cup \{ v _ { f } \}$ when $f \in E .$ , arises from such a chordless cycle or path in $G ,$ , it follows from the separator inequalities for $X _ { \widehat { G } F }$ that y also satisfies the cut inequalities (13) and hence its image under $\mathrm { p r o j } _ { F }$ belongs to $Y _ { G G ^ { \prime } }$ . To see the other direction, i.e. $Y _ { G G ^ { \prime } } \subseteq \mathrm { p r o j } _ { F } X _ { \widehat { G } F } ^ { \prime }$ , for any $y \in Y _ { G G ^ { \prime } }$ we construct another vector $x \in \{ 0 , 1 \} ^ { \widehat { V } \cup F }$ , whose projection to $\mathbb { R } ^ { F }$ is $y _ { ; }$ , as follows:

$$
\forall v \in V \colon x _ { v } = 0 \land \forall e \in E \colon x _ { v _ { e } } = y _ { e } \land \forall f \in F \colon x _ { f } = y _ { f } .
$$

It is easy to verify that x is feasible and hence $x \in X _ { \widehat { G } F } ^ { \prime } .$ . This completes the proof.

## D.20 Proof of Proposition 7

Let

$$
Y _ { \bar { G } \bar { G } ^ { \prime } } ^ { \prime } = \{ y \in Y _ { \bar { G } \bar { G } ^ { \prime } } \mid \forall v \in e \in E : y _ { \{ \bar { v } , v _ { e } \} } = 1 \mathrm { ~ a n d ~ } y _ { \{ v , \bar { v } \} } + y _ { \{ v , v _ { e } \} } = 1 \}
$$

denote the set of all those feasible vectors in $Y _ { \bar { G } \bar { G } ^ { \prime } }$ whose induced lifted multicut M satisfy the following condition: All edges $\{ \bar { v } , v _ { e } \}$ are in the lifted multicut M for $v \in e \in E .$ , and either $\{ v , { \bar { v } } \} \in M$ and $\{ v , v _ { e } \} \not \in M$ for all $e \in E$ incident with $v ,$ or $\{ v , { \bar { v } } \} \not \in M$ and $\{ v , v _ { e } \} \in M$ for all $e \in E$ incident with v. The convex hull of $Y _ { \bar { G } \bar { G } } ^ { \prime } ,$ defines a face of $\Gamma _ { \bar { G } \bar { G } ^ { \prime } }$ , since all inequalities $y _ { e ^ { \prime } } \leq 1$ with $e ^ { \prime } \in E ^ { \prime }$ are valid for $y \in Y _ { \bar { G } \bar { G } ^ { \prime } }$ <sub>′</sub> and $y _ { \{ v , \bar { v } \} } + y _ { \{ v , v _ { e } \} } \geq 1$ are valid for those $y \in Y _ { \bar { G } \bar { G } } ,$ satisfying $y | _ { E ^ { \prime } } = 1$ , where $E ^ { \prime }$ denotes the set of all edges $\{ \bar { v } , v _ { e } \}$ with $v \in e \in E$ . Below we show that $\Xi _ { G F }$ can be obtained as a projection of the face conv $Y _ { \bar { G } \bar { G } ^ { \prime } } ^ { \prime }$ . Since the convex hull operator and the projection operator commute, it sufices to show that $X _ { G F }$ is a projection of $Y _ { \bar { G } \bar { G } ^ { \prime } } ^ { \prime }$

Let $\tilde { E }$ be an edge subset of the subdivision $\widehat { G }$ such that there is exactly one edge in $\tilde { E }$ incident with each vertex of G. So there is a bijection $\lambda \colon V \cup F \to \tilde { E } \cup F$ mapping $v \in V$ to its unique adjacent edge in $\tilde { E }$ and leaving $f \in F$ unchanged. This gives rise to a linear isomorphism $\lambda ^ { * } \colon \mathbb { R } ^ { \tilde { E } \cup F }  \mathbb { R } ^ { V \cup F }$ sending $x \colon { \tilde { E } } \cup F \to \mathbb { R }$ to the composite function x ◦ λ. We claim that the image of $Y _ { \bar { G } \bar { G } ^ { \prime } } ^ { \prime }$ under $\lambda ^ { * } \circ \mathrm { p r o j } _ { \tilde { E } \cup F }$ is exactly $X _ { G F }$ . Similar to the proof of Proposition $6 ,$ for any $f \in F$ , every minimal f-connector of G is the set of all the vertices in $V$ of some chordless path in $\bar { G }$ and vice versa. Thus $\lambda ^ { * } ( \mathrm { p r o j } _ { \tilde { E } \cup F } Y _ { \bar { G } \bar { G } ^ { \prime } } ^ { \prime } )$ is contained in $X _ { G F }$ . The other direction follows from the fact that any $x \in X _ { G F }$ is the image of $y$ under $\lambda ^ { * } \circ \mathrm { p r o j } _ { \tilde { E } \cup F }$ , where y is the vector in $Y _ { \bar { G } \bar { G } ^ { \prime } } ^ { \prime }$ defined by

$$
\forall v \in e \in E : y _ { \{ v , \bar { v } \} } = 1 - x _ { v } \mathrm { ~ a n d ~ } y _ { \{ v , v _ { e } \} } = x _ { v } .
$$

Altogether, we have

$$
X _ { G F } = \lambda ^ { * } \left( \mathrm { p r o j } _ { \tilde { E } \cup F } Y _ { \bar { G } \bar { G } ^ { \prime } } ^ { \prime } \right) .
$$

Therefore, up to relabeling, $\Xi _ { G F }$ is the projection of the face conv $Y _ { \bar { G } \bar { G } ^ { \prime } } ^ { \prime }$ of $\Gamma _ { \bar { G } \bar { G } ^ { \prime } }$ , as desired.

## D.21 Proof of Proposition 8

We can identity $V _ { n } \cup F$ with $E _ { n + 1 } ^ { \prime }$ by the bijection $i \mapsto \{ i , i + 1 \}$ for $i \in V _ { n }$ and $\{ i , j \} \mapsto \{ i , j + 1 \}$ for $\{ i , j \} \in F$ with $i < j . \mathrm { \ B y }$ renaming the indices of x in inequalities (14) and (15) according to this bijection we obtain precisely the path and cut inequalities for the lifted multicut polytope for paths from Lange and Andres (2021). Therefore there is a one-to-one correspondence between the multi-separator polytope for a path of length n the lifted multicut polytope for a path of length $n + 1$

## D.22 Proof of Proposition 9

Because of the one-to-one correspondence between system $( 1 6 ) \textrm { -- } ( 2 0 )$ and system (11) – (15) from Lange and Andres (2021), the desired result follows immediately from Theorem 1 of Lange and Andres (2021), which says that system (11) – (15) therein is totally dual integral.