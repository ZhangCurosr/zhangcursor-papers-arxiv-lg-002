# PRICING SYMBOLIC RULES BY WHAT GNNS LEAVE UNLEARNED ∗

Satabarto Sarkar

Department of Computer Science and Engineering University of Engineering and Management Kolkata 700091, India Satabarto.Sarkar2023@iem.edu.in

## Prithwineel Paul

Department of Computer Science and Engineering University of Engineering and Management Kolkata 700091, India prithwineel.paul@iem.edu.in

## ABSTRACT

Symbolic knowledge (motif counts, ring counts, label constraints) is routinely added to graph neural networks (GNNs) and justified by their limited expressivity. We replace this yardstick with an exact one. The largest log-loss reduction a rule S can give a model family is $\operatorname { I } ( Y ; S \mid \sigma )$ given what the family computes; every realised gain splits exactly into this term plus repaid neural debt; and for a trained network with representation T the value is ${ \dot { \operatorname { I } } } ( Y ; S \mid T )$ , the label information training failed to extract. Rule design thus becomes rule discovery: we score candidates on the network’s out-of-fold residual against calibrated permutation nulls with family-wise error control. On ZINC, discovery recovers the hidden ring term max $( 0 , \ell _ { \mathrm { m a x } } - 6 )$ of the target in three independent runs $( p _ { \mathrm { a d j } } = 0 . 0 0 1 )$ and beats expert cycle counts; frozen and applied to five unseen datasets it is never significantly worse than the plain network or the expert rule, and on BBBP it significantly beats both.

Keywords graph neural networks · neuro-symbolic learning · Weisfeiler–Leman · usable information · learnability · rule discovery

## 1 Introduction

Neuro-symbolic graph learning augments GNNs with cycle and motif counts (Bouritsas et al., 2023; Bodnar et al., 2021), probabilistic label models (Qu et al., 2019; Huang et al., 2021) and logic (Qu and Tang, 2019; Zhang et al., 2020). The standard argument is expressivity: message passing is at most as strong as 1-WL (Xu et al., 2019; Morris et al., 2019), which cannot count many substructures (Chen et al., 2020), so the missing structure should be supplied symbolically. This says why rules can help, not how much a rule is worth to a given network on given data. We prove an absorption identity (Theorem 1), a gain ledger (Theorem 2) and a learnability decomposition (Theorem 3) with a Fano certificate, and turn them into calibrated out-of-fold rule discovery (Section 4). Unlike rule extraction, which reads out what a network has learned (Geng et al., 2025; Morris et al., 2024), discovery targets what it has not.

## 2 Background

A graph G with label Y is drawn from $P ;$ predictors are scored by log-loss $L ( f ) = \mathbb { E } [ - \log f ( Y \mid G ) ]$ (nats). A family F has computable information $\sigma _ { \mathcal { F } } = \sigma \{ \bar { h } ( G ) : h \in \mathcal { H } _ { \mathcal { F } } \}$ , where every $f \in { \mathcal { F } } { \mathrm { i s ~ } } f ( y \mid G ) = g ( y , \bar { h ( G ) } )$ . Following Xu et al. (2020), $\textstyle \mathrm { H } _ { { \mathcal { F } } } ( Y \mid G ) = \operatorname* { i n f } _ { f \in { \mathcal { F } } } { \dot { L ( f ) } }$ . The rule-augmented family ${ \mathcal { F } } \oplus S$ adds a branch $\phi ( S )$ to the logits, with $\phi \equiv 0$ admissible, and the rule’s value is

$$
\Delta _ { { \mathcal F } } ( S ) = { \mathrm { H } } _ { { \mathcal F } } ( Y \mid G ) - { \mathrm { H } } _ { { \mathcal F } \oplus S } ( Y \mid G , S ) \geq 0 .\tag{1}
$$

A t-layer GIN computes a function of the 1-WL colouring $\mathrm { W L } _ { t } ( G )$ and can be injective on the $\mathrm { W L } _ { t }$ classes of any finite sample, so $\sigma _ { \mathcal { F } } = \sigma ( \mathrm { W L } _ { t } )$ there (Xu et al., 2019). The ambiguity of a rule on a dataset $\{ G _ { i } \} _ { i = 1 } ^ { N }$ is

$$
\begin{array} { r } { \alpha _ { t } ( s ) = \frac { 1 } { N } \big | \{ i : \exists j , \mathrm { W L } _ { t } ( G _ { i } ) = \mathrm { W L } _ { t } ( G _ { j } ) , s ( G _ { i } ) \neq s ( G _ { j } ) \} \big | , } \end{array}\tag{2}
$$

and $\alpha _ { t } ( s ) = 0$ means s is $\mathrm { W L } _ { t }$ -measurable on the data. For a trained network, $T = { \hat { h } } ( G )$ is the input of its readout, and $\sigma ( T ) \subseteq \sigma _ { \mathcal { F } }$

Throughout, Y and every rule S take values in countable sets with finite entropy. σ-algebras are identified with the random variables that generate them, and $\operatorname { H } ( Y \mid \sigma )$ is the conditional entropy given σ. All information quantities are in nats. The following lemma is used repeatedly.

Lemma 1 (Gibbs). Let A be a σ-algebra and $q ( \cdot \mid A )$ any A-measurable conditional distribution on $\mathcal { V } .$ . Then $\mathbb { E } [ - \log q ( Y \mid A ) ] \geq \mathrm { H } ( Y \mid A )$ , with equality $i f f q = P _ { Y | \mathcal { A } }$ almost surely.

Proof. $\begin{array} { r } { \mathbb { E } [ - \log q ( Y \mid A ) ] - \mathrm { H } ( Y \mid A ) = \mathbb { E } \big [ \sum _ { y } P ( y \mid A ) \log \frac { P ( y \mid A ) } { q ( y \mid A ) } \big ] = \mathbb { E } \mathrm { K L } ( P _ { Y \mid A } \parallel q ) \ge 0 } \end{array}$ , with equality iff the KL divergence vanishes almost surely.

## 3 Theory

## 3.1 Representational value: the absorption identity

Theorem 1 (Absorption identity). Let $\mathcal { F } ^ { \star }$ contain all σ-measurable predictors and ${ \mathcal { F } } ^ { \star } \oplus S$ all $\sigma \lor \sigma ( S )$ -measurable ones. Then

$$
\Delta ^ { \star } ( S ) = \operatorname { I } ( Y ; S \mid \sigma ) = \operatorname { I } ( Y ; S ) - \operatorname { I } ( S ; \sigma ) + \operatorname { I } ( S ; \sigma \mid Y ) ,\tag{3}
$$

$$
\begin{array} { r } { \big [ \mathrm { I } ( Y ; S ) - \mathrm { I } ( S ; \sigma ) \big ] _ { + } \leq \Delta ^ { \star } ( S ) \leq \operatorname* { m i n } \{ \mathrm { H } ( S \mid \sigma ) , \mathrm { H } ( Y \mid \sigma ) \} , } \end{array}\tag{4}
$$

and the upper bound $\operatorname { H } ( S \mid \sigma )$ exceeds the unclipped lower bound by exactly $\mathrm { H } ( S \mid Y )$

Proof. $\mathcal { F } ^ { \star }$ contains every σ-measurable conditional distribution, so by Lemma $1 \mathrm { H } _ { { \mathcal { F } } ^ { \star } } ( Y \mid G ) = \mathrm { H } ( Y \mid \sigma )$ , attained at $P _ { Y \mid \sigma }$ The augmented family contains every $( \sigma \lor \sigma ( S ) )$ )-measurable conditional distribution whose logits are additive in a σ-measurable and an S-measurable term. When the branch ϕ ranges over all functions of $( S , \sigma )$ (the Bayes family), the same lemma gives $\operatorname { H } ( Y \mid \sigma , S )$ . In general it gives at least this value, with equality whenever log $P ( y \mid \sigma , \dot { S } ) - \log P ( y \mid \sigma )$ is a function of $( y , S )$ alone. Hence $\Delta ^ { \star } = \operatorname { H } ( Y \mid \sigma ) - \operatorname { H } ( Y \mid \sigma , \hat { S _ { ) } } = \bar { \operatorname { I } } ( Y ; S \mid \sigma )$ which is the first equality in (3). The chain rule applied to $\operatorname { I } ( S ; Y , \sigma )$ in both orders gives

$$
\operatorname { I } ( S ; \sigma ) + \operatorname { I } ( S ; Y \mid \sigma ) = \operatorname { I } ( S ; Y ) + \operatorname { I } ( S ; \sigma \mid Y ) ,
$$

which rearranges to the second equality in (3). Since $\operatorname { I } ( S ; \sigma \mid Y ) \ge 0$ and $\operatorname { I } ( Y ; S \mid \sigma ) \ge 0$ , the lower bound in (4) holds. For discrete variables, $\operatorname { I } ( Y ; S \mid \sigma ) = \dot { \operatorname { H } } ( S \mid \sigma ) - \operatorname { H } ( S \mid Y , \sigma ) \leq \operatorname { H } ( S \mid \sigma )$ and symmetrically $\leq \mathrm { H } ( Y \mid \sigma )$ , which is the upper bound. Finally,

$$
\operatorname { H } ( S \mid \sigma ) - \left[ \operatorname { I } ( Y ; S ) - \operatorname { I } ( S ; \sigma ) \right] = \operatorname { H } ( S \mid \sigma ) + \operatorname { I } ( S ; \sigma ) - \operatorname { I } ( Y ; S ) = \operatorname { H } ( S ) - \operatorname { I } ( Y ; S ) = \operatorname { H } ( S \mid Y ) ,
$$

so the gap between the bounds is $\mathrm { H } ( S \mid Y )$ , which is zero iff S is a function of $Y$

Additive branches. The equality case above explains a design choice of rule branches. An additive branch reaches the Bayes gain when the log-likelihood ratio contributed by S does not interact with σ. Otherwise the realised gain is lower, and the shortfall enters the ledger of Theorem 2 as $\varepsilon _ { \mathcal { F } \oplus S }$

Corollary 1 (Expressible knowledge is worthless). $I f S$ is σ-measurable, then $\operatorname { H } ( S \mid \sigma ) = 0$ and $\Delta ^ { \star } ( S ) = 0 .$ . For a t-layer GIN, every rule with $\alpha _ { t } ( s ) = 0$ on the data distribution has zero representational value, including rules that 1-WL cannot compute on general graphs.

Proof. If S is σ-measurable, $\operatorname { H } ( S \mid \sigma ) = 0$ , and the upper bound in (4) gives $\Delta ^ { \star } = 0$ . For a t-layer GIN with injective aggregation, $\sigma _ { \mathcal { F } } = \sigma ( \mathrm { W L } _ { t } )$ on every finite sample (Xu et al., 2019; Morris et al., 2019). $\alpha _ { t } ( s ) = 0$ means that s is constant on every $\mathrm { W L } _ { t }$ class present, so s is $\mathrm { W L } _ { t } ^ { - }$ -measurable on the empirical distribution and its plug-in conditional entropy ${ \hat { \mathrm { H } } } ( S \mid \operatorname { W L } _ { t } )$ is zero. □

Hence a rule with $\alpha _ { t } ( s ) = 0$ is worth zero to a Bayes-optimal t-layer GIN, even if 1-WL cannot compute it on general graphs.

## 3.2 Realised value: the gain ledger

Real networks are not Bayes-optimal; their neural debt is $\varepsilon _ { \mathcal { F } } = \mathrm { H } _ { \mathcal { F } } ( Y \mid G ) - \mathrm { H } ( Y \mid \sigma _ { \mathcal { F } } ) \ge 0$

Theorem 2 (Gain ledger). For everyfamily F and rule S,

$$
\begin{array} { r } { \Delta _ { \mathcal { F } } ( S ) = \underbrace { \mathrm { I } ( Y ; S \mid \sigma _ { \mathcal { F } } ) } _ { r e p r e s e n t a t i o n a l } + \underbrace { \varepsilon _ { \mathcal { F } } - \varepsilon _ { \mathcal { F } \oplus S } } _ { d e b t r e p a i d } , \qquad 0 \le \Delta _ { \mathcal { F } } ( S ) \le \mathrm { I } ( Y ; S \mid \sigma _ { \mathcal { F } } ) + \varepsilon _ { \mathcal { F } } . } \end{array}\tag{5}
$$

Hence a σ<sub>F</sub>-measurable rule can only repay debt, $\Delta _ { \mathcal { F } } ( S ) \leq \varepsilon _ { \mathcal { F } }$ , and $i f { \mathcal { F } } \subseteq { \mathcal { F } } ^ { \prime }$ with $\sigma _ { \mathcal { F } ^ { \prime } } = \sigma _ { \mathcal { F } }$ , every nat ofdebt the larger family repays removes one nat from the ceiling of any such rule.

Proof. By the definitions of the debts, $\mathrm { H } _ { \mathcal { F } } ( Y \mid G ) = \mathrm { H } ( Y \mid \sigma _ { \mathcal { F } } ) + \varepsilon _ { \mathcal { F } }$ and $\mathrm { H } _ { \mathcal { F } \oplus S } ( Y \mid G , S ) = \mathrm { H } ( Y \mid \sigma _ { \mathcal { F } } , S ) + \varepsilon _ { \mathcal { F } \oplus S }$ Subtracting the second from the first and using $\mathrm { H } ( Y \mid \sigma _ { \mathcal { F } } ) { \dot { - } } \mathrm { H } ( Y \mid \sigma _ { \mathcal { F } } , S ) = \mathrm { I } ( \bar { Y } ; S \mid \sigma _ { \mathcal { F } } )$ yields (5). Every $f \in { \mathcal { F } } \oplus S$ is a conditional distribution measurable with respect to $\sigma _ { \mathcal { F } } \lor \sigma ( S )$ , so Lemma 1 gives $\varepsilon _ { \mathcal { F } \oplus S } \ge 0$ and therefore the upper bound. Choosing $\phi \equiv 0$ shows ${ \mathcal { F } } \subseteq { \mathcal { F } } \oplus { \bar { S } }$ , so $\mathrm { H } _ { \mathcal { F } \oplus S } \le \mathrm { H } _ { \mathcal { F } }$ and $\Delta \mathcal { F } \geq 0$

For the consequences: if S is σ -measurable, $\operatorname { I } ( Y ; S \mid \sigma _ { \mathcal { F } } ) = 0 \mathrm { i n } ( 5 )$ , so $\Delta _ { \mathcal { F } } ( S ) = \varepsilon _ { \mathcal { F } } - \varepsilon _ { \mathcal { F } \oplus S } \leq \varepsilon _ { \mathcal { F } }$ . In particular a test-loss reduction g by such a rule certifies $\varepsilon { \mathcal { F } } \geq g$ up to estimation error. If ${ \mathcal { F } } \subseteq { \mathcal { F } } ^ { \prime }$ , then H $_ { \cdot , F ^ { \prime } } \leq \mathrm { H } _ { \mathcal { F } } ;$ ; with $\sigma _ { \mathcal { F } ^ { \prime } } = \sigma _ { \mathcal { F } }$ the Bayes term $\operatorname { H } ( Y \mid \sigma _ { \mathcal { F } } )$ is shared, so $\varepsilon _ { \mathcal { F } ^ { \prime } } = \varepsilon _ { \mathcal { F } } - \left( \mathrm { H } _ { \mathcal { F } } - \mathrm { H } _ { \mathcal { F } ^ { \prime } } \right)$ , and the ceiling ε of every σ -measurable rule falls by exactly the debt the larger family repays. If $\varepsilon _ { \mathcal { F } } = 0 , ( 5 )$ reduces to Theorem 1. □

## 3.3 Trained networks: the learnability decomposition

Theorem 3 (Learnability decomposition). $I f \sigma ( T ) \subseteq \sigma _ { \mathcal { F } } $ , the best post-hoc gain of adding S to T is $\mathrm { H } ( Y \mid T ) - \mathrm { H } ( Y \mid$ $T , S ) = \operatorname { I } ( Y ; S \mid T )$ , and

$$
\begin{array} { r l } & { \mathrm { I } ( Y ; S \mid T ) = \mathrm { I } ( Y ; S \mid \sigma _ { \mathcal { F } } ) + \underbrace { \mathrm { I } ( Y ; \sigma _ { \mathcal { F } } \mid T ) - \mathrm { I } ( Y ; \sigma _ { \mathcal { F } } \mid T , S ) } _ { \Lambda _ { T } ( S ) : l o s t i n t r a i n i n g , r e s t o r e d b y ~ S } } \\ & { \qquad \le \mathrm { I } ( Y ; S \mid \sigma _ { \mathcal { F } } ) + \left[ \mathrm { H } ( Y \mid T ) - \mathrm { H } ( Y \mid \sigma _ { \mathcal { F } } ) \right] . } \end{array}\tag{6}
$$

Proof. The post-hoc family of all conditional distributions measurable in $( T , S )$ attains $\mathrm { H } ( Y \mid T , S )$ by Lemma 1, and the family measurable in T attains $\mathrm { H } ( Y \mid T )$ , so $\Delta _ { T } ( S ) = \operatorname { I } ( Y ; S \mid T )$ . By the chain rule,

$$
\operatorname { I } ( Y ; S , \sigma _ { \mathcal { F } } \mid T ) = \operatorname { I } ( Y ; S \mid T ) + \operatorname { I } ( Y ; \sigma _ { \mathcal { F } } \mid T , S ) = \operatorname { I } ( Y ; \sigma _ { \mathcal { F } } \mid T ) + \operatorname { I } ( Y ; S \mid \sigma _ { \mathcal { F } } , T ) .
$$

Because $\sigma ( T ) \subseteq \sigma _ { \mathcal { F } }$ , conditioning on $( \sigma _ { \mathcal { F } } , T )$ is conditioning on $\sigma { } _ { \mathcal { F } }$ , so $\operatorname { I } ( Y ; S \mid \sigma _ { \mathcal { F } } , T ) = \operatorname { I } ( Y ; S \mid \sigma _ { \mathcal { F } } )$ . Rearranging gives (6). Since $\tilde { \mathbf { I } } ( \tilde { Y } ; \sigma _ { \mathcal { F } } \mid T , S ) \stackrel { \sim } { \geq } 0 , \boldsymbol { \dot { \Lambda } } _ { T } ( S ) \stackrel { \sim } { \leq } \mathbf { I } ( Y ; \sigma _ { \mathcal { F } } \mid T ) ^ { \sim } \mathbf { H } ( \tilde { Y } \mid T ) - \mathbf { H } ( \tilde { Y } \mid \sigma _ { \mathcal { F } } , T ) \stackrel { \sim } { = } \mathbf { H } ( Y \mid \tilde { T } ) ^ { \sim } - \mathbf { H } ( \tilde { Y } \mid \tilde { \sigma } _ { \mathcal { F } } )$ If S is $\sigma { } _ { \mathcal { F } }$ -measurable, the first term of (6) vanishes.

Expressivity is thus neither necessary nor sufficient: a rule T determines is worthless however far beyond 1-WL it lies, and an unlearned σ<sub>F</sub>-measurable rule can be worth $\Lambda _ { T } ( S ) > 0$ . Theorems 2 and 3 are two views of one identity. Let $\mathcal { F } _ { T }$ be the family of predictors that read $T$ only. Its debt relative to $\sigma { } _ { \mathcal { F } }$ is $\mathrm { H } ( Y \mid T ) - \mathrm { H } ( Y \mid \sigma _ { \mathcal { F } } )$ , exactly the bound in Theorem 3. The learnability decomposition is therefore the ledger identity for the family “trained network plus arbitrary post-hoc head”, with the debt made explicit as lost label information. A probe checks which case holds:

Proposition 1 (Fano certificate). For discrete rule columns $S _ { c }$ with alphabets $ { \boldsymbol { S } } _ { c }$ and any probe $\hat { S } = \varphi ( T )$ with error rates $P _ { e , c } ,$

$$
\begin{array} { r } { \operatorname { I } ( Y ; S \mid T ) \le \operatorname { H } ( S \mid T ) \le \operatorname { H } ( S \mid \hat { S } ) \le \sum _ { c } \bigl [ h _ { b } ( P _ { e , c } ) + P _ { e , c } \log ( | S _ { c } | - 1 ) \bigr ] . } \end{array}\tag{7}
$$

Proof. $\operatorname { I } ( Y ; S \mid T ) = \operatorname { H } ( S \mid T ) - \operatorname { H } ( S \mid T , Y ) \leq \operatorname { H } ( S \mid T )$ . Since $\hat { S } \ = \ \varphi ( T )$ , conditioning on $T$ refines conditioning on ${ \hat { S } } ,$ so $\mathrm { H } ( S \mid T ) \le \mathrm { H } ( S \mid { \hat { S } } )$ . By the chain rule and because conditioning reduces entropy, $\operatorname { H } ( S \mid { \hat { S } } ) =$ $\begin{array} { r } { \sum _ { c } \operatorname { H } ( S _ { c } \mid S _ { < c } , \hat { S } ) \leq \sum _ { c } \operatorname { H } ( S _ { c } \mid \hat { S } _ { c } ) } \end{array}$ . Fano’s inequality (Cover and Thomas, 2006, Thm. 2.10.1) bounds each term by $\overline { { h _ { b } } } \big ( P _ { e , c } \big ) + P _ { e , c } \log ( | S _ { c } | - 1 )$ . If $P _ { e , c } = 0$ for all $c ,$ the bound is zero. □

## 4 Methodology: out-of-fold rule discovery

Theorem 3 names the rule worth adding: the candidate maximising $\operatorname { I } ( Y ; S \mid T )$ . Discovery therefore selects rules from a candidate library: it decides which named, interpretable statistic carries the label information a trained network failed to extract. It does not synthesise new functional forms, and every library used is listed in Section 5. Let $r = Y - \mathbb { E } [ Y \mid T ]$

be the residual (one-hot minus probabilities for classification) and $\tilde { C } _ { B } = C _ { B } - \mathbb { E } [ C _ { B } \mid T ]$ the part of a block $C _ { B }$ of standardised candidate columns that $T$ does not predict, with orthonormal basis $Q _ { B }$ . The block energy

$$
\begin{array} { r } { E _ { B } = \frac { 1 } { n } \| Q _ { B } ^ { \top } r \| _ { F } ^ { 2 } } \end{array}\tag{8}
$$

is a local surrogate of value (Proposition 2).

Proposition 2 (Local value of a block). (a) ${ \cal I } f ( r , { \tilde { C } } _ { B } )$ are jointly Gaussian given $T ,$ then $\begin{array} { r } { \operatorname { I } ( Y ; C _ { B } \mid T ) = - \frac { 1 } { 2 } \log \left( 1 - \frac { } { } \right. } \end{array}$ $E _ { B } ^ { \mathrm { p o p } } / \mathrm { V a r } ( r ) )$ , where $E _ { B } ^ { \mathrm { p o p } }$ is the population variance of the projection of r on ${ \tilde { C } } _ { B } ,$ , and $E _ { B } \to E _ { B } ^ { \mathrm { p o p } }$ . (b) Adding $\beta ^ { \top } \tilde { C } _ { B }$ to the logit ofa binary predictor $p ( T )$ changes the expected log-loss $b y - \textstyle { \frac { 1 } { 2 } } g ^ { \top } J ^ { - 1 } g + o ( \| g \| ^ { 2 } )$ at its optimum, with $g = \mathbb { E } [ ( Y - p ) \tilde { C } _ { B } ]$ and $J = \mathbb { E } [ p ( 1 - p ) \tilde { C } _ { B } \tilde { C } _ { B } ^ { \top } ]$ ]. With whitened ${ \tilde { C } } _ { B }$ and $p ( 1 - p )$ ≈ w, the reduction is $E _ { B } / ( 2 w )$ to first order. (c) For K classes the exact local reduction is ${ \begin{array} { r l } { { \frac { 1 } { 2 } } \mathrm { v e c } ( G ) ^ { \top } J ^ { - 1 } \mathrm { v e c } ( G ) } \end{array} }$ with $G = \mathbb { E } [ \tilde { C } _ { B } r ^ { \top } ]$ and $J = \mathbb { E } [ ( \mathrm { d i a g } p - p p ^ { \top } ) \otimes \tilde { C } _ { B } \tilde { C } _ { B } ^ { \top } ] ; E _ { B } = \| G \| _ { F } ^ { 2 }$ replaces J by a multiple of the identity, which is exact when the curvature is uncorrelated with $\tilde { C } _ { B } \tilde { C } _ { B } ^ { \top }$

Proof. (a) Let $r = Y - \mathbb { E } [ Y \mid T ]$ and $\tilde { C } = C _ { B } - \mathbb { E } [ C _ { B } \ | \ T ]$ be jointly Gaussian given T, with $\Sigma _ { r r } = \mathrm { V a r } ( r )$ $\Sigma _ { r c } = \mathrm { C o v } ( r , \tilde { C } )$ and $\Sigma _ { c c } = \mathrm { V a r } ( \tilde { C } )$ ). Then $\begin{array} { r } { \operatorname { I } ( Y ; C _ { B } \mid T ) = \operatorname { I } ( r ; { \tilde { C } } ) = - { \frac { 1 } { 2 } } \log \left( 1 - \Sigma _ { r c } \Sigma _ { c c } ^ { - 1 } \Sigma _ { c r } / \Sigma _ { r r } \right) } \end{array}$ . The quantity $\Sigma _ { r c } \Sigma _ { c c } ^ { - 1 } \Sigma _ { c r }$ is the variance of the projection of r onto ${ \tilde { C } } ,$ whose sample analogue with an orthonormal basis $Q _ { B }$ of the centred columns is $\| Q _ { B } ^ { \top } r \| ^ { 2 } / n = E _ { B }$ . Consistency follows from the law of large numbers for second moments.

(b) Let $\eta = \log \mathrm { i t } p ( T )$ and $L ( \beta ) = \mathbb { E } \bigl [ - Y ( \eta + \beta ^ { \top } \tilde { C } ) + \log ( 1 + e ^ { \eta + \beta ^ { \top } \tilde { C } } ) \bigr ]$ . Then $\nabla L ( 0 ) = - \mathbb { E } [ ( Y - p ) \tilde { C } ] = - g$ and $\nabla ^ { 2 } L ( 0 ) = \mathbb { E } [ p ( 1 - p ) \tilde { C } \tilde { C } ^ { \top } ] = J .$ A second-order Taylor expansion gives $\begin{array} { r } { L ( \beta ) = L ( 0 ) - \beta ^ { \top } g + \frac { 1 } { 2 } \beta ^ { \top } J \beta + O ( \| \beta \| ^ { 3 } ) } \end{array}$ which is minimised at $\beta ^ { \star } = J ^ { - 1 } g + O ( \| g \| ^ { 2 } )$ with $\begin{array} { r } { L ( 0 ) - L ( \beta ^ { \star } ) = \frac 1 2 g ^ { \top } J ^ { - 1 } g + o ( \Vert g \Vert ^ { 2 } ) } \end{array}$ ). If E $\mathbb { E } [ \tilde { C } \tilde { C } ^ { \top } ] = I$ and $p ( 1 - p ) \approx w$ , then J ≈ wI and the reduction is $\| g \| ^ { 2 } / ( 2 w )$ . With whitened sample rows $z _ { i } = \sqrt { n } Q _ { B , i } .$ <sub>·</sub>, the plug-in score is $\begin{array} { r } { \hat { g } = \frac { 1 } { n } \sum _ { i } r _ { i } z _ { i } = n ^ { - 1 / 2 } Q _ { B } ^ { \top } r , \mathbf { s o } \| \hat { g } \| ^ { 2 } = E _ { B } } \end{array}$

(c) For K classes, r is the one-hot label minus the predicted probabilities, and the same argument applied to each logit gives the Frobenius form in (8).

For Gaussian residuals the energy is therefore an exact monotone transform of value, and for a classifier it is the first-order log-loss reduction. Otherwise $E _ { B }$ only ranks blocks, so significance nominates and validation decides.

Out-of-fold residuals. In-sample residuals carry a memorisation term that cancels all evidence when $\hat { f }$ interpolates (Proposition 3). We take r and $T$ from networks trained on the other half of the data (cross-fitting, Chernozhukov et al., 2018).

Proposition 3 (Out-of-fold versus in-sample evidence). Let ${ \hat { f } } ^ { ( - k ) }$ be trained on $D \setminus D _ { k }$ . For $i \in D _ { k } , \mathbb { E } [ ( Y _ { i } -$ ${ \hat { f } } ^ { ( - k ) } ( G _ { i } ) ) \ c ( G _ { i } ) \ | \ { \hat { f } } ^ { ( - k ) } ] = \operatorname { C o v } _ { P } \left( Y - { \hat { f } } ^ { ( - k ) } ( G ) , c ( G ) \right) + \operatorname { \mathbb { E } } [ Y - { \hat { f } } ^ { ( - k ) } ] \operatorname { \mathbb { E } } [ c ]$ . For a network $\hat { f }$ trained on all ofD and ${ \hat { f } } ^ { ( - i ) }$ trained without graph i,

$$
\begin{array} { r } { \mathbb { E } \left[ \left( Y _ { i } - \hat { f } ( G _ { i } ) \right) c ( G _ { i } ) \right] = \mathbb { E } \left[ \left( Y _ { i } - \hat { f } ^ { ( - i ) } ( G _ { i } ) \right) c ( G _ { i } ) \right] - \underbrace { \mathbb { E } \left[ \left( \hat { f } ( G _ { i } ) - \hat { f } ^ { ( - i ) } ( G _ { i } ) \right) c ( G _ { i } ) \right] } _ { \mathrm { ~ . ~ } } . } \end{array}\tag{9}
$$

$I f { \hat { f } }$ interpolates its training labels, the in-sample residual vanishes and the evidence for every candidate collapses.

Proof. For $i \in D _ { k } , ( G _ { i } , Y _ { i } )$ is independent of ${ \hat { f } } ^ { ( - k ) }$ , which is a function of $D \setminus D _ { k }$ and of the training randomness only. Conditioning on ${ \hat { f } } ^ { ( - k ) }$ and writing $e ( G , Y ) = Y - { \hat { f } } ^ { ( - k ) } ( G )$ gives $\mathbb { E } [ e ( G _ { i } , Y _ { i } ) c ( G _ { i } ) \mid \hat { f } ^ { ( - k ) } ] = \mathbb { E } _ { P } [ e c ] =$ $\mathrm { C o v } _ { P } ( e , c ) + \mathbb { E } _ { P } [ e ] \mathbb { E } _ { P } [ c ]$ . For the in-sample residual, add and subtract ${ \hat { f } } ^ { ( - i ) } ( G _ { i } )$

$$
Y _ { i } - \hat { f } ( G _ { i } ) = \big ( Y _ { i } - \hat { f } ^ { ( - i ) } ( G _ { i } ) \big ) - \big ( \hat { f } ( G _ { i } ) - \hat { f } ^ { ( - i ) } ( G _ { i } ) \big ) ,
$$

then multiply by $c ( G _ { i } )$ and take expectations to obtain (9). If <sup>ˆ</sup>f interpolates the training labels, $Y _ { i } - { \hat { f } } ( G _ { i } ) = 0$ for every i, so all in-sample covariances vanish. The leave-one-out residual is unaffected, so the memorisation term ω exactly cancels the true residual covariance. □

Remark 1 (Statistical debt). The out-of-fold residual measures the debt of a network trained on $n / 2$ graphs, part of which disappears at n graphs. A block can therefore be significant for the half-data network and worth little to the full-data network, which is why significance nominates rules and validation loss decides how many to keep. In particular, the half-data networks are used only to compute residuals: every gain we report is measured after retraining on the full training set with the rule added, so a rule that a fully trained network already learns shows no gain there (Section 6.6).

![](images/aa791da8096b1cf74fa30ea3d3894eb673abcb9ef4460387758efbe76e77af3e.jpg)  
ZINC: the first block is max(0, $\ell _ { \mathrm { m a x } } - 6 )$ , the hidden ring term of penalised logP $( p _ { \mathrm { a d j } } = 0 . 0 0 1$ in three independent runs)

Figure 1: Out-of-fold rule discovery. Networks trained on half of the training graphs expose what the architecture fails to generalise; candidate rules are scored by how much of that residual their unlearned part explains, each block against its own permutation null; the winning named rule is added to the network.

Calibration and family-wise error. Blocks differ in size and skew, so we permute the residual rows Π = 1000 times, fit the null energies by a moment-matched scaled $\chi ^ { 2 } , a _ { B } \chi _ { \nu _ { B } } ^ { 2 }$ with $a _ { B } = \mathrm { V a r } / 2 \mu$ and $\nu _ { B } = 2 \mu ^ { 2 } / \mathrm { V a I }$ (Satterthwaite, 1946; Box, 1954), and score each block by its surprise $s _ { B } = - \log \mathrm { P r } ( a _ { B } \chi _ { \nu _ { B } } ^ { 2 } \geq E _ { B } )$ . The pick $\hat { B } = \arg \operatorname* { m a x } _ { B } s _ { B }$ receives the Westfall–Young max-T p-value (Westfall and Young, 1993)

$$
p _ { \mathrm { a d j } } = \frac { 1 + \# \{ \pi : \operatorname* { m a x } _ { B } s _ { B } ^ { ( \pi ) } \geq s _ { \hat { B } } \} } { 1 + \Pi } ,\tag{10}
$$

which controls the family-wise error rate over all blocks. The residual is projected off the picked span and the search repeats (group OMP, Pati et al., 1993); validation loss picks how many blocks to keep (Algorithm 1, Figure 1).

Scope of the guarantee. The family-wise error control of (10) is a guarantee for the first round: it bounds the probability that the first pick is declared significant when no block carries signal. The p-values of later greedy rounds are conditional on the earlier picks and are reported descriptively; the number of blocks kept is decided by validation loss, not by these p-values.

Validity of the calibrated max-T test. Under the complete null that the residual rows are exchangeable with respect to the candidate rows, every permutation π of the residual leaves the joint distribution of $( r ^ { \pi } , \tilde { C } )$ unchanged. The surprises $s _ { B }$ are fixed monotone transforms of the energies $E _ { B }$ , with transforms estimated from the same permutations. The max $- T p .$ -value (10) is therefore a valid permutation $p \mathrm { - }$ value for the family-wise null, and it controls the family-wise error rate at any level α (Westfall and Young, 1993). Strong control holds under subset pivotality. Because the transform is fitted on the permutation null itself, it only rescales the blocks to a common scale and does not affect validity.

Moment-matched scaled chi-square. If n ${ \mathrm { \Delta } } E _ { B } / { \mathrm { V a r } } ( r )$ were exactly $\chi _ { d _ { B } } ^ { 2 }$ (Gaussian residual, fixed basis), its mean and variance would be $d _ { B }$ and $2 d _ { B }$ . For non-Gaussian residuals and skewed candidate columns, the null of $E _ { B }$ has mean $\mu _ { B }$ and variance $v _ { B }$ that deviate from this. Matching $a \chi _ { \nu } ^ { 2 }$ to them gives $a \nu = \mu _ { B }$ and $2 a ^ { 2 } \nu = v _ { B }$ , hence $a = v _ { B } / ( 2 \mu _ { B } )$ and $\nu = 2 \mu _ { B } ^ { 2 } / v _ { B }$ (Satterthwaite, 1946; Box, 1954). This calibration removes the bias of raw z-scores towards single-column blocks, whose nulls are heavy-tailed. We confirmed this on planted signals: both a single planted column and a weak planted ten-column group were recovered, and the pure null gave $p _ { \mathrm { a d j } } \approx 0 . 5$

Cost. A discovery round costs O(Π n D K) for n graphs, total block dimension D and K residual columns, on top of training four half-data networks. This took seconds to minutes on one GPU for a few hundred candidates.

## 5 Experimental setup

Data. Data are real only: TU benchmarks (Morris et al., 2020), ZINC-12k (Dwivedi et al., 2023), scaffoldsplit OGB molecules (Hu et al., 2020a) and eleven node-classification graphs (Table 1). Sources: TU benchmarks, https://chrsmrrs.github.io/datasets/; OGB, https://ogb.stanford.edu/docs/graphprop/; ZINC-12k, https://github.com/graphdeeplearning/benchmarking-gnns; node-level graphs, https: //github.com/kimiyoung/planetoid, https://github.com/shchur/gnn-benchmark, https://github.

```latex
Algorithm 1: Out-of-fold rule discovery
Input: training graphs $D ,$ candidate library $C \in \mathbb { R } ^ { n \times p }$ with blocks B (every column and every family), rounds $K ,$
permutations Π
Split D into $D _ { 1 } , D _ { 2 } ;$ for seeds $s = 1 , 2$ train $\hat { f } _ { s } ^ { ( - k ) }$ on $D \setminus D _ { k }$ and keep its best-validation outputs;
for $i \in D _ { k }$ do
$\begin{array} { r } { \underline { { \hat { y } } } _ { i } \gets \frac { 1 } { 2 } \sum _ { s } \hat { f } _ { s } ^ { ( - k ) } ( G _ { i } ) , \ T _ { i } \gets \hat { h } ^ { ( - k ) } ( G _ { i } ) , \ r _ { i } \gets y _ { i } - \hat { y } _ { i } ; } \end{array}$
for $k = 1 , 2$ do
1 $\tilde { C } [ D _ { k } ] \gets C [ D _ { k } ] - \widehat { C } ( T ) [ D _ { k } ]$ // cross-fitted MLP probe within fold
$\mathcal { P }  \emptyset , \ r  r - \bar { r } ;$
for $j = 1 , \ldots , K$ do
for $B \in \mathcal { B } { \bf d o } Q _ { B } \gets \mathrm { o r t h } ( \tilde { C } _ { B } ) , \ E _ { B } \gets \| Q _ { B } ^ { \top } r \| _ { F } ^ { 2 } / n ;$
permute rows of r Π times, compute null energies, fit $a _ { B } \chi _ { \nu _ { B } } ^ { 2 } ,$ , surprises $s _ { B } ;$
$\begin{array} { r } { \hat { B } \gets \arg \operatorname* { m a x } _ { B \notin \mathcal { P } } s _ { B } , \ p _ { \mathrm { a d j } } \ \mathrm { b y } \left( 1 0 \right) , \ \mathcal { P } \gets \mathcal { P } \cup \{ \hat { B } \} ; } \end{array}$
$r  r - \Pi _ { \mathcal { P } } r$ ; // project off the picked span
Train the network with rule branches for the nested sets $\mathcal { P } _ { 1 } \subset \cdots \subset \mathcal { P } _ { K } ;$ keep the set with the best validation loss;
```

Table 1: Datasets. Splits: CV = stratified 10-fold cross-validation (WL, absorption and ledger experiments); 80/10/10 = fixed stratified split (discovery experiments); scaffold = OGB scaffold split; ZINC = the standard 10k/1k/1k split.
<table><tr><td>Dataset</td><td>task</td><td>graphs</td><td>classes</td><td>splits used</td></tr><tr><td>MUTAG, PTC</td><td>binary classification</td><td>188,344</td><td>2</td><td>CV</td></tr><tr><td>PROTEINS</td><td>binary classification</td><td>1113</td><td>2</td><td>CV, 80/10/10, random/size</td></tr><tr><td>NCI1</td><td>binary classification</td><td>4110</td><td>2</td><td>CV, random/size</td></tr><tr><td>IMDB-BINARY</td><td>binary classification</td><td>1000</td><td>2</td><td>CV, 80/10/10, random/size</td></tr><tr><td>IMDB-MULTI</td><td>3-class classification</td><td>1500</td><td>3</td><td>CV</td></tr><tr><td>COLLAB</td><td>3-class classification</td><td>5000</td><td>3</td><td>CV, 80/10/10</td></tr><tr><td>REDDIT-BINARY</td><td>binary classification</td><td>2000</td><td>2</td><td>80/10/10</td></tr><tr><td>ZINC-12k</td><td>regression (MAE)</td><td>12000</td><td>一</td><td>standard</td></tr><tr><td>ogbg-molhiv</td><td>binary (ROC-AUC)</td><td>41127</td><td>2</td><td>scaffold, random</td></tr><tr><td>ogbg-molbace</td><td>binary (ROC-AUC)</td><td>1513</td><td>2</td><td>scaffold, random</td></tr><tr><td>ogbg-molbbbp</td><td>binary (ROC-AUC)</td><td>2039</td><td>2</td><td>scaffold, random</td></tr></table>

Node level: Cora, CiteSeer, PubMed, Amazon-Photo, Amazon-Computers, Actor, Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions; label rates 5% and 50%; 5 seeds.

com/graphdml-uiuc-jlu/geom-gcn and https://github.com/yandex-research/heterophilous-graphs.   
ZINC, molhiv and COLLAB served for development; five datasets were held out.

Rule libraries. The classical library has eleven rules: graph size, degree histogram, node-type histogram, closed-walk counts, cyclomatic number, triangle counts, cycle counts of length 3–8, ring-atom counts, Laplacian spectral moments, bond-type histogram and a Gaussian-noise placebo. The discovery library adds, for molecules, chordless-ring counts by size (3–12), ring summaries (number of rings, $\ell _ { \mathrm { m a x } } , \mathrm { m a x } ( 0 , \ell _ { \mathrm { m a x } } - 6 )$ , rings with more than six atoms), fusion and spiro counts and the 150 most frequent ring compositions; for social graphs, 26 statistics of size, density, degree shape, triangles and clustering, cores and cliques, components, spectrum and assortativity. The classical library is included as distractors. The same discovery library is used for every molecular dataset; the term ma $\tau ( 0 , \ell _ { \mathrm { m a x } } - 6 )$ is one of its ring summaries, among about 300 ring statistics.

Networks. Models are four-layer GIN (Xu et al., 2019) or GINE with all OGB features (Hu et al., 2020b); rules enter as an MLP branch on the logits. GIN updates node states as $\begin{array} { r } { h _ { v } ^ { \prime } = \mathrm { M L P } \big ( ( 1 + \epsilon ) h _ { v } + \sum _ { u \in N ( v ) } h _ { u } \big ) } \end{array}$ with a two-layer MLP and batch normalisation. GINE adds a learned embedding of the bond features to each message before the ReLU. On OGB data, the nine atom and three bond features each have their own embedding table, and the embeddings are summed, as in the OGB AtomEncoder and BondEncoder. The representation T is the concatenation of the sum-pooled node states of all layers, including the input layer, so dim $T \stackrel { \cdot } { = } 5 { \times } \mathrm { h i d d e n }$ . The readout is batch normalisation followed by a two-layer MLP. A rule branch maps the standardised rule columns through a two-layer MLP to the output dimension, and its output is added to the logits. For efficiency, all members of one experiment (seeds × rule sets × fold networks) are trained jointly as one batched network. Each member has its own weights and batch-normalisation statistics, all members share mini-batches, and each keeps the outputs of its own best validation epoch. Table 2 lists the hyperparameters.

Table 2: Hyperparameters. All models use four layers and Adam.
<table><tr><td>Experiment</td><td>model</td><td>hidden</td><td>lr</td><td>schedule</td><td>dropout</td><td>epochs</td><td>batch</td></tr><tr><td>WL/absorption/ledger (TU, 10-fold)</td><td>GIN, DeepSets</td><td>64</td><td> $1 0 ^ { - 2 }$ </td><td>patience 100</td><td>0.5</td><td>350</td><td></td></tr><tr><td>Classical rules on ZINC/NCI1/PROTEINS</td><td>GIN, GINE</td><td>128</td><td> $1 0 ^ { - 3 }$ </td><td>cosine</td><td>0</td><td>200</td><td>128</td></tr><tr><td>Discovery, ZINC</td><td>GINE</td><td>128</td><td> $1 0 ^ { - 3 }$ </td><td>cosine</td><td>0</td><td>200</td><td>128</td></tr><tr><td>Discovery, molhiv</td><td>GINE (OGB feat.)</td><td>256</td><td> $1 0 ^ { - 3 }$ </td><td>cosine</td><td>0.5</td><td>80</td><td>128</td></tr><tr><td>Discovery, molbace/molbbbp</td><td>GINE (OGB feat.)</td><td>256</td><td> $1 0 ^ { - 3 }$ </td><td>cosine</td><td>0.5</td><td>100</td><td>32</td></tr><tr><td>Discovery, COLLAB/TU</td><td>GIN (degree tags)</td><td>64</td><td> $1 0 ^ { - 3 }$ </td><td>cosine</td><td>0</td><td>150</td><td>32</td></tr><tr><td>Random-split absorption, OGB</td><td>GINE (OGB feat.)</td><td>256 (128 molhiv)</td><td> $1 0 ^ { - 3 }$ </td><td>cosine</td><td>0.5</td><td>100 (60)</td><td>32 (128)</td></tr><tr><td>Random-split absorption, TU</td><td>GIN</td><td>64</td><td> $1 0 ^ { - 3 }$ </td><td>cosine</td><td>0</td><td>150</td><td>32</td></tr></table>

Table 3: Exact 1-WL ambiguity $\alpha _ { t }$ (%, maximum over six beyond-1-WL rules) and test-accuracy gain of adding those rules (10-fold CV).
<table><tr><td></td><td>MUTAG</td><td>PTC</td><td>PROTEINS</td><td>NCI1</td><td>IMDB-B</td><td>IMDB-M</td><td>COLLAB</td></tr><tr><td>α0</td><td>13.3</td><td>3.2</td><td>46.4</td><td>33.5</td><td>0</td><td>0</td><td>0</td></tr><tr><td> $\alpha _ { 1 } / \alpha _ { \geq 2 }$ </td><td>0/0</td><td>0/0</td><td>0/0</td><td>0.12 / 0</td><td>0/0</td><td>0/0</td><td>0/0</td></tr><tr><td>gain, DeepSets</td><td>+11.3</td><td>+2.0</td><td>+4.6</td><td>+3.1</td><td>+1.6</td><td>-0.5</td><td>+1.9</td></tr><tr><td>gain, GIN</td><td>-1.6</td><td>-3.8</td><td>+2.4</td><td>+0.8</td><td>-0.3</td><td>+0.3</td><td>+2.1</td></tr></table>

Probes and $R _ { T } ^ { 2 }$ . The probe that predicts a rule (or a candidate) from T is a two-layer MLP with 64 hidden units, trained with Adam (learning rate $3 \times 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 } )$ for 300 steps with early stopping on a held-out fifth. It is cross-fitted over two folds, so every prediction is out-of-sample. $R _ { T } ^ { 2 } = 1 - \mathbb { E } \| S - \hat { S } ( T ) \| ^ { 2 } / \mathrm { V a r } ( S )$ is averaged over the rule’s columns.

Discovery. Out-of-fold residuals come from two seeds × two folds of networks trained on half of the training graphs. For every training graph, the residual uses the average prediction of the two networks that did not see it, and the representation comes from the first of them. Candidate columns are signed-log transformed, standardised on training graphs and clipped to [−6, 6]. Ring patterns enter the pool if they occur in at least 20 training graphs (at most 150 patterns), and chordless rings are enumerated up to length 12. Blocks are every single column plus every family. The families are: each classical rule, ring sizes, ring summary, ring fusion, ring atom patterns, ring bond patterns, size, degree shape, triangles and clustering, cores and cliques, components and spectrum. We use Π = 1000 residual permutations per round and three rounds. Rule sets blk ${ \mathrm { : ~ } } \subset { \mathrm { ~ b l k _ { 2 } ~ } } \subset { \mathrm { ~ b l k _ { 3 } ~ } }$ are retrained on the full training set with the same seeds as the plain network, and the set with the lowest mean validation loss is reported. Controls are the expert rule, a size-matched random set of non-picked candidate columns, and the whole pool.

Statistics. Every comparison is paired by seed. Intervals are 95% percentile bootstrap intervals of the mean paired difference (4000 resamples). Classification log-losses are per graph in nats. OGB atom types are atomic number minus one, so in ring patterns 5 is C, 6 is N and 7 is O; bond types are 0 single, 1 double, 2 triple and 3 aromatic.

## 6 Results

## 6.1 Expressible knowledge is absorbed

Computing $\alpha _ { t }$ exactly on every graph of seven TU benchmarks (Table 3, Figure 2), up to 46% of graphs are ambiguous before message passing (PROTEINS), at most 0.12% after one round (NCI1) and none after two. By Theorem 1, six beyond-1-WL rules are worth zero to a GIN of depth two; their gain to a graph-blind DeepSets model (up to +11.3 points) largely vanishes under GIN. GNNs absorb a median 99% of a belief-propagation layer’s value on eleven node-classification graphs, and where GIN still gains, an MLP readout repays 80–83% of it (COLLAB, NCI1), as the ledger predicts.

![](images/22bb8915e3600825e316870de969c035084e9d34403205dcd4efe1160a71378d.jpg)  
Figure 2: Exact 1-WL ambiguity of beyond-1-WL rules, maximum over six rules (Table 3). One round of colour refinement removes the ambiguity almost entirely, and two rounds remove it completely.

Table 4: Gain ledger for 1-WL-computable rules on GIN (10-fold CV, test log-loss gain in nats, 95% CI). The readout and coordinate parts are the reductions in rule gain when the linear readout is replaced by an MLP readout and when log-size coordinates are added.
<table><tr><td>Dataset</td><td>observed gain</td><td>readout part</td><td>coordinate part</td><td>residual</td></tr><tr><td>COLLAB</td><td> $+ 0 . 0 5 2 \ [ + 0 . 0 3 8 , + 0 . 0 6 7 ]$ </td><td>+0.042</td><td>-0.003</td><td>+0.013[+0.003, +0.023]</td></tr><tr><td>NCI1</td><td> $+ 0 . 0 2 6 \ [ + 0 . 0 1 1 , + 0 . 0 4 1 ]$ </td><td>+0.022</td><td>-0.007</td><td>+0.011[-0.021,+0.043]</td></tr><tr><td>PROTEINS</td><td> $+ 0 . 0 2 6 \ [ + 0 . 0 0 7 , + 0 . 0 4 4 ]$ </td><td>+0.007</td><td>-0.019</td><td>+0.038 [-0.001, +0.076]</td></tr></table>

## 6.2 The gain ledger on graph-level benchmarks

For 1-WL-computable rules on GIN, the ledger of Theorem 2 splits the observed gain into the part repaid by an MLP readout, the part repaid by log-size coordinates and a residual (Table 4, Figure 3). Most of the gain on COLLAB and NCI1 is debt that a better readout also repays (Table 5). For beyond-1-WL rules, the ledger gives an observed gain of +0.020 nats on COLLAB, with a residual of +0.012 [+0.003, +0.021] after the readout and coordinate parts. On PROTEINS the observed gain is +0.023, with a residual $\mathrm { o f + 0 . 0 3 2 \ [ - 0 . 0 0 2 , + 0 . 0 6 6 ] }$ . On every other dataset the observed gain is not significant. The shuffled-rule placebo is significantly negative on COLLAB (−0.052), IMDB-B (−0.047) and IMDB-M (−0.031) and never significantly positive, because an extra branch that carries no information can only overfit.

## 6.3 Learnability, not expressivity, decides

On ZINC (Figure 4a, Table 6) bond-type counts are invisible to GIN, yet it decodes them from atoms $( R _ { T } ^ { 2 } = 0 . 8 6 )$ and they are worth nothing. Ring counts are the least learned rules $( \dot { R } _ { T } ^ { 2 } = 0 . 4 9 / 0 . 5 9 )$ and the only ones that pay, cutting MAE by 27% (GIN) and 32% (GINE). Triangles are unlearned $( \dot { R } _ { T } ^ { 2 } = 0 . 4 \dot { 0 } )$ but worthless: large $\mathrm { H } ( S \mid T )$ is necessary, not sufficient. On NCI1 and PROTEINS (GIN, same protocol), the largest single-rule gains in test log-loss are +0.021 (cyclomatic number on NCI1, $R _ { T } ^ { 2 } = 0 . 9 \dot { 5 } )$ and +0.037 (type histogram on PROTEINS, $R _ { T } ^ { 2 } = 0 . 7 1 $ . Both are learnable rules that repay debt in the sense of Theorem 2. The Fano bound of Proposition 1 held for all 14 rules for which it is defined, at 0.26–32 nats.

## 6.4 Discovery recovers the hidden term of the target

ZINC’s penalised logP subtracts the ring penalty ma $\tau ( 0 , \ell _ { \mathrm { m a x } } - 6 )$ from Crippen logP (Gómez-Bombarelli et al., 2018; Jin et al., 2018); atom-additive logP is easy for message passing, the longest-cycle term is not. Among 300 ring statistics plus the classical library, the first pick is exactly max $\cdot ( 0 , \ell _ { \mathrm { m a x } } - 6 )$ for GIN and GINE in three independent runs $( p _ { \mathrm { a d j } } = 0 . 0 0 1$ , the minimum for $\Pi = \hat { 1 } 0 0 0 )$ . The method selects this term from the library; what the result shows is that, among hundreds of plausible ring statistics and distractors, the out-of-fold residual singles out the one term of the target that message passing leaves unlearned. The validation-selected set reaches MAE 0.1196 against 0.1306 for expert cycle counts (+0.011 [0.009, 0.012]). On molhiv it adds +0.021 [+0.010, +0.031] ROC-AUC (Tables 7 and 8).

Gain ledger of 1-WL-computable rules on GIN (95% CI)  
![](images/3f6dfc2c894ac2f142d3e2a0e186d4f988839df5920dc8014b074b9fc89bd0ef.jpg)  
Figure 3: Gain ledger of 1-WL-computable rules on GIN (Table 4). The observed gain decomposes into the part repaid by an MLP readout, the part repaid by log-size coordinates, and the residual. Error bars are the 95% intervals of the observed gain and the residual.

Table 5: Plain GIN with three readouts (10-fold CV): test log-loss (nats) and accuracy. The MLP readout repays most of the debt that rules otherwise repay (Table 4).
<table><tr><td></td><td colspan="3">log-loss</td><td colspan="3">accuracy</td></tr><tr><td>Dataset</td><td>linear</td><td>MLP</td><td>log-MLP</td><td>linear</td><td>MLP</td><td>log-MLP</td></tr><tr><td>COLLAB</td><td>0.518</td><td>0.468</td><td>0.470</td><td>0.751</td><td>0.790</td><td>0.792</td></tr><tr><td>NCI1</td><td>0.484</td><td>0.479</td><td>0.474</td><td>0.783</td><td>0.788</td><td>0.783</td></tr><tr><td>PROTEINS</td><td>0.574</td><td>0.550</td><td>0.575</td><td>0.735</td><td>0.731</td><td>0.704</td></tr><tr><td>IMDB-B</td><td>0.516</td><td>0.533</td><td>0.529</td><td>0.726</td><td>0.724</td><td>0.721</td></tr></table>

On ZINC the first pick dominates every later pick by a factor of about five, and it lies far above the 95th percentile of its 1000-permutation null (Figure 5). Earlier discovery runs on ZINC confirm the result. The first discovery run used single columns, in-sample residuals and $k \in \{ 1 , 3 , 8 \}$ picks chosen by validation. On GIN it reached MAE 0.1829 (plain 0.2673, cycles 0.1951, whole pool 0.1653, random 8 columns 0.2057). On GINE it reached 0.1186 (plain 0.1938, cycles 0.1333, whole pool 0.1277, random 0.1118). The fresh-seed certification run reached 0.1779 on GIN (plain 0.2722, cycles 0.1965, whole pool 0.1659) and 0.1173 on GINE (plain 0.1885, cycles 0.1322, whole pool 0.1269, random 0.1147). Discovery beat the expert cycle counts in all four cases, by between 0.012 and 0.019 MAE with intervals above zero. Earlier molhiv runs fed the network only atom and bond type and are superseded by the full-feature protocol of Table 7.

## 6.5 Held-out datasets

On five held-out datasets (10 seeds; Figure 4b, Tables 7 and 9), the discovered set is never significantly worse than the plain network or the expert rule. On molbbbp it significantly beats both (+0.035 [+0.025, +0.045] and +0.016 [+0.002, +0.030] ROC-AUC). On the other four, point estimates lie in [−0.012, −0.002] with every interval covering zero. The molbace interval against the expert rule only just covers zero (upper limit +0.0003).

## 6.6 Out-of-fold versus in-sample evidence

For the first pick, log p (scaled-χ<sup>2</sup>) out-of-fold vs. in-sample was −9.1 vs. −3.9 on molhiv. On the held-out datasets it was −3.3 vs. −2.1 (molbbbp), −12.3 vs. −4.0 (REDDIT-B), −1.2 vs. −0.9 (IMDB-B), −4.1 vs. −4.6 (molbace) and −2.6 vs. −2.9 (PROTEINS). The networks trained on half the data are weaker than those trained on all of it (ZINC MAE 0.279 ys. 0.191, molhiv ROC-AUC 0.737 ys. 0.759. COLLAB accuracy 0.758 ys. 0.800), which is the statistical debt of Remark 1.

This weakness affects which rules are nominated, not the gains we report. Every gain in Tables 7–9 is measured on networks retrained on the full training set with the rule branch added, against the full-data plain network with the same seeds. A rule that only the half-data network fails to learn, and that a fully trained network absorbs, would therefore show no gain in these tables. On ZINC the ring term alone (blk ) still cuts the full-data MAE from 0.1911 to 0.1181, so it is information that the fully trained network also leaves unlearned. Where a nominated block is worth little at full data (REDDIT-B: $p _ { \mathrm { a d j } } = 0 . 0 0 1$ , no gain), validation loss and the paired comparison expose it, which is why significance nominates and validation decides.

(a) A rule pays only for what is not learned  
![](images/0d23cba202df3366af353cb18d9e61f121e3b1859cf5a25426c38353debdaf2f.jpg)

(b) Discovered rules vs. the expert rule  
![](images/83db50ccc121a26ba37d7ced05b2d7d37cfec2f5d8e7dae98ec3e1425c50f1a6.jpg)  
Figure 4: (a) ZINC: a rule’s gain tracks its unlearned share $1 - R _ { T } ^ { 2 }$ , not 1-WL expressivity. (b) Discovered (blue) and expert (orange) rules vs. the plain network, 95% intervals; rows below the dashed line were held out.

Table 6: Classical rules added to GIN and GINE on ZINC (3 seeds). Gain: reduction in test MAE (plain GIN 0.2742, GINE 0.1918) with 95% interval. $R _ { T } ^ { 2 }$ : share of the rule decoded from the plain network’s representation. Probe pick: the two rules with the largest residual score, chosen before any rule model was trained.
<table><tr><td rowspan="2">Rule</td><td colspan="2">GIN</td><td colspan="2">GINE</td></tr><tr><td> $R _ { T } ^ { 2 }$ </td><td>gain [95% CI]</td><td> $R _ { T } ^ { 2 }$ </td><td>gain [95% CI]</td></tr><tr><td>cycles (3–8)</td><td>0.49</td><td>+0.0751 [+0.0643, +0.0900]</td><td>0.59</td><td>+0.0617[+0.0565, +0.0667]</td></tr><tr><td>triangles</td><td>0.40</td><td>+0.0019 [-0.0206, +0.0168]</td><td>0.64</td><td>-0.0008 [-0.0085, +0.0052]</td></tr><tr><td>spectrum</td><td>0.85</td><td>-0.0070 [-0.0286, +0.0083]</td><td>0.86</td><td>+0.0032[-0.0044, +0.0114]</td></tr><tr><td>bond histogram</td><td>0.86</td><td>+0.0054[-0.0069, +0.0226]</td><td>0.98</td><td>-0.0014[-0.0041, +0.0039]</td></tr><tr><td>ring atoms</td><td>0.94</td><td>-0.0034[-0.0196, +0.0055]</td><td>0.95</td><td>-0.0063 [-0.0165, -0.0010]</td></tr><tr><td>degree histogram</td><td>0.98</td><td>+0.0004[-0.0179, +0.0163]</td><td>0.98</td><td>-0.0093 [-0.0106, -0.0070]</td></tr><tr><td>cyclomatic number</td><td>0.98</td><td>+0.0068 [-0.0094, +0.0198]</td><td>0.98</td><td>+0.0068 [-0.0022, +0.0167]</td></tr><tr><td>type histogram</td><td>0.99</td><td>-0.0046 [-0.0123, +0.0086]</td><td>0.99</td><td>+0.0040 [-0.0012, +0.0092]</td></tr><tr><td>closed walks</td><td>0.99</td><td>+0.0067 [-0.0094, +0.0163]</td><td>0.99</td><td>+0.0011[-0.0044, +0.0041]</td></tr><tr><td>size</td><td>0.99</td><td>-0.0032 [-0.0150, +0.0050]</td><td>0.99</td><td>+0.0042 [-0.0010, +0.0071]</td></tr><tr><td>noise (placebo)</td><td></td><td>-0.0003 [-0.0018, +0.0027]</td><td></td><td>-0.0018[-0.0083, +0.0103]</td></tr><tr><td>probe pick (top 2)</td><td></td><td></td><td></td><td>+0.0608 [+0.0536, +0.0648]</td></tr><tr><td>all rules</td><td></td><td> $\begin{array} { r } { + 0 . 0 7 5 5 \left[ + 0 . 0 6 2 7 , + 0 . 0 8 2 0 \right] } \\ { + 0 . 0 7 4 5 \left[ + 0 . 0 6 1 4 , + 0 . 0 8 7 2 \right] } \end{array}$ </td><td></td><td>+0.0523 [+0.0500, +0.0553]</td></tr></table>

## 6.7 In-distribution and shifted splits

Under train-small/test-large splits, the plain networks are badly overconfident on large graphs: their test log-loss is 2.85 nats (NCI1) and 4.64 nats (PROTEINS), against 0.46 and 0.50 under random splits. Rule gains under size shift therefore mix information and calibration effects, which is why we report them without interpretation (Table 10). Under random splits every classical rule set lies within ±0.02 nats of its noise-input twin (Figure 6).

![](images/6cc502c559cd0f139321d3c77002098732142f01d1d45d266d6b1165810836f4.jpg)

![](images/c9f1bdcd22df8d7c126611841f0fbd8ef02f60b8707090dc4aa990f4d2ffec45.jpg)  
Figure 5: (a) Greedy residual scores of the first discovery run (log scale). On ZINC the first pick, the ring term of the target, dominates every later pick by a factor of about five. (b) Fresh-seed run: the first pick against the 95th percentile of its 1000-permutation null.

Table 7: Out-of-fold discovery with the frozen method. Expert: cycle counts (molecules, PROTEINS) or graph size (social graphs). Gains are paired by seed with 95% bootstrap intervals; for MAE a positive gain means lower error. First block: the first pick and its Westfall–Young $p _ { \mathrm { a d j } }$
<table><tr><td>Dataset (metric)</td><td>seeds</td><td>plain</td><td>expert</td><td>discovered</td><td>gain vs. plain</td><td>gain vs. expert</td><td>first block  $( p _ { \mathrm { a d j } } )$ </td></tr><tr><td colspan="8">development</td></tr><tr><td>ZINC (MAE↓)</td><td>3</td><td>0.1911</td><td>0.1306</td><td>0.1196</td><td>+0.072 [.065, .077]</td><td>+0.011 [.009, .012]</td><td>ring term (.001)</td></tr><tr><td>molhiv (AUC)</td><td>8</td><td>0.7588</td><td>0.7871</td><td>0.7796</td><td>+0.021 [.010, .031]</td><td>-0.008 [−.023, .010]</td><td>12-ring, arom. (.001)</td></tr><tr><td>COLLAB (acc)</td><td>5</td><td>0.7996</td><td>0.8072</td><td>0.8064</td><td>+0.007 [.000, .016]</td><td>-0.001 [−.008, .006]</td><td>clustering (.001)</td></tr><tr><td colspan="8">held out, frozen method</td></tr><tr><td>molbbbp (AUC)</td><td>10</td><td>0.6421</td><td>0.6610</td><td>0.6768</td><td>+0.035 [.025, .045]</td><td>+0.016 [.002, .030]</td><td> $\mathrm { C } _ { 5 } \mathrm { O }$  6-ring (.046)</td></tr><tr><td>molbace (AUC)</td><td>10</td><td>0.8084</td><td>0.8074</td><td>0.7959</td><td>-0.012 [−.029, .008]</td><td>-0.011[−.024, .000]</td><td>walks (.006)</td></tr><tr><td>PROTEINS (acc)</td><td>10</td><td>0.7786</td><td>0.7696</td><td>0.7759</td><td>-0.003 [−.015, .008]</td><td>+0.006 [−.005, .017]</td><td>assortativity (.105)</td></tr><tr><td>IMDB-B (acc)</td><td>10</td><td>0.7590</td><td>0.7560</td><td>0.7520</td><td>-0.007 [−.023, .010]</td><td>-0.004 [−.017, .008]</td><td>type hist. (.995)</td></tr><tr><td>REDDIT-B (acc)</td><td>10</td><td>0.9175</td><td>0.9170</td><td>0.9155</td><td>-0.002 [−.014, .012]</td><td>−0.002 [−.015, .011]</td><td>mean degree (.001)</td></tr></table>

## 7 Scope

Large gains concentrate where a structural term is unlearned (ring-dependent molecular targets); on social and protein graphs trained GINs leave little usable debt. Selection buys parsimony more than accuracy: the whole candidate pool sometimes matches the selected set (molbbbp +0.047 vs. +0.035) but can also hurt (molhiv −0.039). Significance nominates but does not measure size (REDDIT-B: $p _ { \mathrm { a d j } } = 0 . 0 0 1$ , no gain; Remark 1). Under scaffold or size shift classical rules gain more on four of six datasets, but no in-distribution quantity predicted the shifted value. The Fano certificate is valid but loose (0.26–32 nats); the usable-information bound $\mathbb { E } [ - \log q ( S \mid T ) ] \geq \mathrm { H } ( S \mid T )$ (Xu et al., 2020) is a tighter alternative. A discovered rule is a residual correlate of one trained network, not a causal mechanism; named chemotypes are hypotheses for domain experts.

## 8 Conclusion

A rule is worth what a GNN fails to learn, $\operatorname { I } ( Y ; S \mid T )$ , of which expressivity controls only the first term. Out-of-fold residuals turn this identity into a calibrated search over named candidate rules, which recovered the hidden ring term of ZINC’s target and transferred, frozen, to unseen datasets without significant loss and with a significant gain on BBBP.

## References

Cristian Bodnar, Fabrizio Frasca, Nina Otter, Yuguang Wang, Pietro Liò, Guido F. Montúfar, and Michael Bronstein. Weisfeiler and Lehman go cellular: CW networks. In Advances in Neural Information Processing Systems, volume 34,

Table 8: Development datasets with the frozen out-of-fold method: every configuration, paired gain over the plain network with 95% interval (MAE: positive = lower error). Bold: validation-selected. Blocks in order of discovery. $\mathrm { Z I N C } { : \operatorname* { m a x } ( 0 , \ell _ { \operatorname* { m a x } } - 6 ) } ;$ 10-ring count; closed walks. molhiv: 12-ring with seven aromatic bonds; 7-ring with two aromatic bonds; type histogram. COLLAB: triangles/clustering family; 90th percentile of degree.
<table><tr><td></td><td colspan="2">ZINC (MAE, 3 seeds)</td><td colspan="2">molhiv (AUC, 8 seeds)</td><td colspan="2">COLLAB (acc, 5 seeds)</td></tr><tr><td>Config</td><td>score</td><td>gain</td><td>score</td><td>gain</td><td>score</td><td>gain</td></tr><tr><td>plain</td><td>0.1911</td><td></td><td>0.7588</td><td></td><td>0.7996</td><td></td></tr><tr><td>expert rule</td><td>0.1306</td><td>+0.061 [.056, .065]</td><td>0.7871</td><td>+0.028 [.013, .042]</td><td>0.8072</td><td>+0.008 [.001, .014]</td></tr><tr><td>blk1</td><td>0.1181</td><td>+0.073 [.066, .079]</td><td>0.7741</td><td>+0.015 [.007, .025]</td><td>0.8064</td><td>+0.007 [.000, .016]</td></tr><tr><td>blk2</td><td>0.1196</td><td>+0.072 [.065, .077]</td><td>0.7796</td><td>+0.021 [.010, .031]</td><td>0.8092</td><td>+0.010 [.002, .024]</td></tr><tr><td>blk3</td><td>0.1215</td><td>+0.070 [.066, .076]</td><td>0.7783</td><td>+0.020 [.005, .033]</td><td>0.8052</td><td>+0.006 [−.003, .015]</td></tr><tr><td>random (size-matched)</td><td>0.1814</td><td>+0.010 [.005, .013]</td><td>0.7742</td><td>+0.015 [.008, .023]</td><td>0.7968</td><td>-0.003 [−.016, .014]</td></tr><tr><td>whole pool</td><td>0.1301</td><td>+0.061 [.054, .070]</td><td>0.7202</td><td>-0.039 [−.046, −.031]</td><td>0.8168</td><td>+0.017 [.010, .026]</td></tr></table>

Table 9: Held-out datasets (frozen method, 10 seeds): every configuration, paired gain over the plain network with 95% interval. Bold: validation-selected.
<table><tr><td>Config</td><td>molbace (AUC)</td><td>molbbbp (AUC)</td><td>PROTEINS (acc)</td><td>IMDB-B (acc)</td><td>REDDIT-B (acc)</td></tr><tr><td>plain</td><td>0.8084</td><td>0.6421</td><td>0.7786</td><td>0.7590</td><td>0.9175</td></tr><tr><td>expert rule</td><td>-0.001 [−.019, .018]</td><td>+0.019 [.011, .027]</td><td>-0.009 [−.023, .005]</td><td>-0.003 [−.016, .009]</td><td>-0.001[−.011, .011]</td></tr><tr><td>blk1</td><td>-0.017 [−.039, .002]</td><td>+0.019 [.010, .031]</td><td>-0.003 [−.015, .008]</td><td>-0.019 [−.034, −.005]</td><td>+0.008 [−.006, .022]</td></tr><tr><td>blk2</td><td>-0.007[-.027, .014]</td><td>+0.032 [.016, .046]</td><td>-0.013 [-.030, .001]</td><td>-0.022 [−.041, −.002]</td><td>-0.002 [−.014, .012]</td></tr><tr><td>blk3</td><td>-0.012 [−.029, .008]</td><td>+0.035 [.025, .044]</td><td>-0.003 [−.016, .011]</td><td>-0.007 [−.023, .010]</td><td>-0.006 [−.017, .006]</td></tr><tr><td>random (size-matched)</td><td>-0.024 [-.047, .001]</td><td>+0.022 [.011, .033]</td><td>-0.001 [-.015, .013]</td><td>+0.001 [−.012, .015]</td><td>+0.002 [−.009, .013]</td></tr><tr><td>whole pool</td><td>-0.005 [−.026, .015]</td><td>+0.047 [.037, .058]</td><td>+0.004 [−.006, .016]</td><td>+0.015 [.002, .027]</td><td>-0.003 [−.013, .008]</td></tr></table>

pages 2625–2640, 2021. doi: 10.48550/arXiv.2106.12575.

Giorgos Bouritsas, Fabrizio Frasca, Stefanos Zafeiriou, and Michael M. Bronstein. Improving graph neural network expressivity via subgraph isomorphism counting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(1):657–668, 2023. doi: 10.1109/TPAMI.2022.3154319.

G. E. P. Box. Some theorems on quadratic forms applied in the study of analysis of variance problems, I. effect of inequality of variance in the one-way classification. The Annals ofMathematical Statistics, 25(2):290–302, 1954. doi: 10.1214/aoms/1177728786.

Zhengdao Chen, Lei Chen, Soledad Villar, and Joan Bruna. Can graph neural networks count substructures? In Advances in Neural Information Processing Systems, volume 33, pages 10383–10395, 2020. doi: 10.48550/arXiv.2002.04025.

Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21 (1):C1–C68, 2018. doi: 10.1111/ectj.12097.

Thomas M. Cover and Joy A. Thomas. Elements ofInformation Theory. John Wiley & Sons, Hoboken, NJ, 2nd edition, 2006. doi: 10.1002/047174882X.

Vijay Prakash Dwivedi, Chaitanya K. Joshi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. Journal of Machine Learning Research, 24(43):1–48, 2023. doi: 10.48550/ arXiv.2003.00982.

Chuqin Geng, Ziyu Zhao, Zhaoyue Wang, Haolin Ye, Yuhe Jiang, and Xujie Si. LogicXGNN: Grounded logical rules for explaining graph neural networks, 2025. arXiv:2503.19476; first version titled “Extracting Interpretable Logic Rules from Graph Neural Networks”.

Rafael Gómez-Bombarelli, Jennifer N. Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D. Hirzel, Ryan P. Adams, and Alán Aspuru-Guzik. Automatic chemical design using a data-driven continuous representation of molecules. ACS Central Science, 4(2): 268–276, 2018. doi: 10.1021/acscentsci.7b00572.

Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. In Advances in Neural Information Processing Systems, volume 33, pages 22118–22133, 2020a. doi: 10.48550/arXiv.2005.00687.

Table 10: Mean test log-loss gain of the classical rules over the noise-input twin (3 seeds), random split vs. shifted split (scaffold for molecules, train-small/test-large for TU). Plain: ROC-AUC (molecules) or accuracy (TU) of the plain network.
<table><tr><td>Dataset</td><td>plain (random)</td><td>plain (shifted)</td><td>mean rule gain, random</td><td>mean rule gain, shifted</td></tr><tr><td>ogbg-molhiv</td><td>0.828</td><td>0.756</td><td>+0.0021</td><td>+0.0016</td></tr><tr><td>ogbg-molbace</td><td>0.862</td><td>0.810</td><td>-0.0195</td><td>+0.0627</td></tr><tr><td>ogbg-molbbbp</td><td>0.954</td><td>0.664</td><td>-0.0062</td><td>+0.0105</td></tr><tr><td>NCI1</td><td>0.813</td><td>0.564</td><td>+0.0063</td><td>+0.2112</td></tr><tr><td>PROTEINS</td><td>0.771</td><td>0.593</td><td>-0.0157</td><td>+0.1702</td></tr><tr><td>IMDB-B</td><td>0.716</td><td>0.744</td><td>-0.0062</td><td>-1.2528</td></tr></table>

![](images/19754d3c5ee2aacb7e63cacb8bdb022dd4404e366029abd99f77ddd7574ac264.jpg)  
Figure 6: Mean test log-loss gain of the classical rules over the noise-input twin under random splits. Every dataset lies within ±0.02 nats (shaded band).

Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. Strategies for pre-training graph neural networks. In International Conference on Learning Representations, 2020b. doi: 10.48550/arXiv.1905.12265.

Qian Huang, Horace He, Abhay Singh, Ser-Nam Lim, and Austin R. Benson. Combining label propagation and simple models out-performs graph neural networks. In International Conference on Learning Representations, 2021. doi: 10.48550/arXiv.2010.13993.

Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. In Proceedings ofthe 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 2323–2332. PMLR, 2018. doi: 10.48550/arXiv.1802.04364.

Christopher Morris, Martin Ritzert, Matthias Fey, William L. Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and Leman go neural: Higher-order graph neural networks. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 33, pages 4602–4609, 2019. doi: 10.1609/aaai.v33i01.33014602.

Christopher Morris, Nils M. Kriege, Franka Bause, Kristian Kersting, Petra Mutzel, and Marion Neumann. TUDataset: A collection of benchmark datasets for learning with graphs. In ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+), 2020. doi: 10.48550/arXiv.2007.08663.

Matthew Morris, David J. Tena Cucala, Bernardo Cuenca Grau, and Ian Horrocks. Relational graph convolutional networks do not learn sound rules. In Proceedings ofthe 21st International Conference on Principles ofKnowledge Representation and Reasoning (KR 2024), pages 897–908. IJCAI Organization, 2024. doi: 10.24963/kr.2024/84.

Yagyensh C. Pati, Ramin Rezaiifar, and Perinkulam S. Krishnaprasad. Orthogonal matching pursuit: Recursive function approximation with applications to wavelet decomposition. In Proceedings of27th Asilomar Conference on Signals, Systems and Computers, pages 40–44, 1993. doi: 10.1109/ACSSC.1993.342465.

Meng Qu and Jian Tang. Probabilistic logic neural networks for reasoning. In Advances in Neural Information Processing Systems, volume 32, 2019. doi: 10.48550/arXiv.1906.08495.

Meng Qu, Yoshua Bengio, and Jian Tang. GMNN: Graph Markov neural networks. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pages 5241–5250. PMLR, 2019. doi: 10.48550/arXiv.1905.06214.

F. E. Satterthwaite. An approximate distribution of estimates of variance components. Biometrics Bulletin, 2(6): 110–114, 1946. doi: 10.2307/3002019.

Peter H. Westfall and S. Stanley Young. Resampling-Based Multiple Testing: Examples and Methods for p-Value Adjustment. Wiley Series in Probability and Mathematical Statistics. Wiley, New York, 1993. ISBN 0-471-55761-7.

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019. doi: 10.48550/arXiv.1810.00826.

Yilun Xu, Shengjia Zhao, Jiaming Song, Russell Stewart, and Stefano Ermon. A theory of usable information under computational constraints. In International Conference on Learning Representations, 2020. doi: 10.48550/arXiv. 2002.10689.

Yuyu Zhang, Xinshi Chen, Yuan Yang, Arun Ramamurthy, Bo Li, Yuan Qi, and Le Song. Efficient probabilistic logic reasoning with graph neural networks. In International Conference on Learning Representations, 2020. doi: 10.48550/arXiv.2001.11850.