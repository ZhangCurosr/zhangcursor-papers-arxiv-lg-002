# Self-Directed Apple Tasting

Tanmay Devale <sup>\*</sup> Reed Phillips <sup>†</sup> Arvind Ramaswami<sup>‡</sup>

November 5, 2026

## Abstract

We study the classical apple-tasting setting, a partial feedback model in which the learner observes the true label only when it predicts 1. A combinatorial characterization for the rates of this problem in the online adversarial setting has been studied by Raman et al. [2024]. We study the self-directed setting, in which the learner can adaptively choose the order of the points it predicts on. In the realizable setting, we establish a tetrachotomy on the expected minimax number of mistakes made by any learner. Depending on the VC dimension, the ∅-centered star number, and the self-directed dimension, we show that the rates could possibly be $\Theta ( 1 ) , O ( \log n ) \cap \omega ( 1 ) , \widetilde { \Theta } ( \sqrt { n } )$ or $\Theta ( n )$ . In the agnostic setting, the minimax regret is $\widetilde { \Theta } ( \sqrt { n } )$ or $\Theta ( n )$ , depending on the VC dimension.

## 1 Introduction

Imagine visiting a fruit vendor to buy apples: you pick one, taste it, and learn whether it is good; you then pick another, taste it, and so on. We call this self-directed online classification under apple-tasting feedback, an analogue of the model introduced by Helmbold et al. [2000]. In this model, the learner observes the correct label only when it predicts 1 (i.e. eats the apple), in contrast to standard online learning, where the correct label is revealed after every prediction.

We believe that this self-directed formulation is arguably the natural way to model literal apple tasting: the defining feature of the problem is that the learner controls which point it exposes itself to risk on, and self-direction simply takes this control to its logical conclusion by letting the learner choose the point itself, rather than having it imposed by an adversary. Prior work on apple tasting [Raman et al., 2024, Chase and Mehalel, 2025] instead studies the setting in which the point in each round is chosen by the environment or adversary, and the learner only controls its prediction. While this adversarial formulation is a natural starting point for worst-case analysis, it discards exactly the element of agency that makes apple tasting a distinctive feedback model to begin with — after all, no one is forced to buy a particular apple at the market.

A more practical instance of this problem can be seen in the hiring process at a company. They first review applications and then decide which candidates to interview. As interviewing is more informative but more resource-intensive than simply reading applications, the company might want to adaptively decide which candidates to interview in order to learn what features in applications are most desirable.

Formally, self-directed online classification with apple-tasting feedback is a repeated game between an adversary and a learner played over n rounds. At the start of the game, the adversary provides a set of n points. In round i, the learner selects one of these points $x _ { i }$ and makes a prediction $\widehat { y } _ { i } \in \{ 0 , 1 \}$ . The adversary then assigns a label $y _ { i } \in \{ 0 , 1 \}$ , unknown to the learner, and the learner incurs loss $\mathbb { 1 } \left[ \widehat { y } _ { i } \neq y _ { i } \right]$ If $\widehat { y } _ { i } = 1$ , the adversary reveals the true label $y _ { i } ;$ otherwise, the learner receives no feedback. We allow both the learner and the adversary to use randomized strategies. In most cases, however, we show that deterministic strategies would give essentially the same results.

We study both the realizable and the agnostic settings in our paper. We adopt the perspective of Raman et al. [2024] and identify the combinatorial dimensions that govern the minimax number of mistakes and regret<sup>1</sup> respectively. Our main contributions are as follows:

• The Vapnik-Chervonenkis (VC) Dimension [Vapnik and Chervonenkis, 1971] characterizes the minimax expected regret in the agnostic setting as $\widetilde { \Theta } \Big ( \sqrt { n \cdot \mathrm { V C } ( \mathcal { H } ) } \Big )$

• On the other hand, VC dimension alone is not enough to characterize the realizable setting. There are easier regimes characterized by the ∅-centered star number of H [Hanneke, 2024], denoted ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } )$ , and the self-directed dimension of [Devulapalli and Hanneke, 2024] denoted by SDdim(H).

• Consequently, in the realizable setting, we obtain a tetrachotomy on the mistake bound. $( \mathrm { i } ) \Theta _ { \mathcal { H } } ( 1 )$ when $\mathfrak { s } _ { \emptyset } ( \mathcal { H } ) < \infty , \mathrm { S D d i m } ( \mathcal { H } ) < \infty ; \mathrm { ( i i ) } O _ { \mathcal { H } } ( \log n ) \cap \omega _ { \mathcal { H } } ( 1 )$ when ${ \mathfrak { s } } _ { \varnothing } ( { \mathcal { H } } ) < \infty$ but $\mathrm { S D d i m } ( \mathscr { H } ) =$ $\infty ; ( \mathrm { i i i } ) \bar { \Theta } _ { \mathcal { H } } ( \sqrt { n } )$ when $\mathrm { V C } ( \mathcal { H } ) < \infty$ but ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } ) = \infty ; ( { \mathrm { i v } } ) \ \Theta ( n )$ when $\mathrm { V C } ( \mathcal { H } ) = \infty$

We additionally establish several auxiliary results.

• We show that for any $d \in \mathbb { N } .$ , there exists a concept class $\mathcal { H } _ { d }$ such that $\mathrm { V C } ( \mathcal { H } _ { d } ) = d$ and the optimal number of mistakes made by any algorithm is $\Omega ( { \sqrt { n d } } )$

• In the realizable setting, if the adversary may choose labels adaptively, our upper and lower bounds <sub>on</sub> <sub>the</sub> <sub>mistake</sub> <sub>bound</sub> <sub>differ</sub> <sub>by</sub> <sub>a</sub> <sub>factor</sub> <sub>of</sub> √<sub>log</sub> <sub>n.</sub> <sub>This</sub> <sub>gap</sub> <sub>vanishes</sub> <sub>when</sub> <sub>the</sub> <sub>adversary</sub> <sub>must</sub> instead commit to an unknown true concept before the game begins.

• It is straightforward to find concept classes that fit in cases (i), (iii), and (iv) of our tetrachotomy. We show that case (ii), where ${ \mathfrak { s } } _ { \varnothing } ( { \mathcal { H } } ) < \infty$ and $\mathrm { S D d i m } ( { \mathcal { H } } ) = \infty$ , is also nonempty by adapting a construction of Kuhlmann $[ 1 9 9 9 ]$

• Finally, we prove a purely combinatorial identity that may be of independent interest. For any concept class $\mathcal { H } .$ letting $\mathcal { U } ( \mathcal { H } )$ denote its union closure, we have ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } ) = { \mathfrak { s } } _ { \emptyset } ( { \mathcal { U } } ( { \mathcal { H } } ) ) = \operatorname { V C } ( { \mathcal { U } } ( { \mathcal { H } } ) )$

## 2 Literature Review

Apple Tasting: Apple tasting is a partial feedback model. It is a special case of partial monitoring (Bartók et al. [2014]) and of sequential prediction with graph feedback (Alon et al. [2015]). Online learning with apple-tasting feedback was initially studied in Helmbold et al. [2000], via randomized algorithms and an analysis of expected regret. Raman et al. [2024] provide a combinatorial characterization for online apple tasting, via the Littlestone dimension LDim(H) in the agnostic case, and show an upper bound of $O ( \sqrt { \mathrm { L D i m } ( \mathcal { H } ) T \log T } )$ on the expected regret. They provide a tighter combinatorial characterization using the effective width in the realizable case where they show that the minimax mistake bound exhibits a trichotomy: Θ(1), $\Theta ( { \sqrt { T } } )$ , or $\Theta ( T )$ . Our tetrachotomy (Theorem 4.1) is a self-directed counterpart of this result, where the rates are instead governed by the VC dimension, ∅-centered star number, and the self-directed dimension.

Chase and Mehalel [2025] study deterministic algorithms and show a $\sqrt { \log T }$ gap between randomized and deterministic learners in the realizable case. This gap could also appear in our setting; all of our upper bounds in Section 4.1 are achieved by deterministic learners, while our lower bounds in Section 4.2 hold against randomized ones.

Self-Directed Learning: Self-directed learning was studied for special classes in Goldman and Sloan [1994]. In more generality, Devulapalli and Hanneke [2024] provide a dimension SDdim to characterize the mistake bound for self-directed learning. Our work considers self-directed learning with apple-tasting rather than full-information feedback. Ben-David et al. [1995] and Kuhlmann [1999] provide separations between the number of mistakes in self-directed learning and offline learning. We use the intersection-closed property of Kuhlmann [1999]’s construction in Theorem 7.1.

The Star Number: Hanneke and Yang [2015] introduced the star number s(H) to characterize the minimax label complexity in active learning under low noise regimes under the noise model of Tsybakov [2004]. It has since appeared in several other roles: Wiener et al. [2015] provide a version space compression scheme of size ${ \mathfrak { s } } ( { \mathcal { H } } )$ , and Hanneke [2016] analyze its role in the sample complexity of empirical risk minimization (ERM). Bousquet et al. [2020] study a related variant, the hollow star number, in connection with the Helly number and an optimal SVM bound. Hanneke [2024] collects elementary observations about the star number, such as equivalence with the VC dimension of the version space class.

Our work uses the ∅-centered star number, introduced by Hanneke [2024], as one of the parameters to analyze mistake bounds for self-directed apple tasting. Intuitively, star number measures the worstcase deviation from a particular central concept. Apple tasting’s asymmetry makes the zero concept especially important. Using it as the center yields the ∅-centered star number.

Prediction with Expert Advice: Prediction with expert advice [Cesa-Bianchi et al., 2006, Vovk, 1990, 1995] is a sequential framework for making predictions. A learner receives predictions from N experts in each of T rounds and must make few mistakes relative to the best-performing expert. Multiplicative weights gives a regret bound of $\sqrt { T \log N }$ . Tighter first- and second-order bounds were studied, such as the second-order bounds in Cesa-Bianchi et al. [2007]. We specifically use the expert advice algorithm with an apple-tasting variant EXP4.AT of Raman et al. [2024].

Other tools: Two results are used directly. We use the one-inclusion graph predictor of Haussler et al. [1994], whose leave-one-out guarantee bounds the expected error of predicting an unseen point by $\mathrm { V C } ( \mathscr { H } ) / | R |$ given $| R |$ training points. We also use progress on the union-closed set conjecture, which guarantees that some point must be labeled 1 by a constant fraction of a union-closed concept class. See Proposition 1 and Bruhn and Schaudt [2013] for a survey of the conjecture.

## 3 Formal Setup and Definitions

In this section, we formally present the setup and definitions used throughout the paper.

Let X denote an arbitrary non-empty set, referred to as the instance space, and $\mathcal { V } = \{ 0 , 1 \}$ the label space. Let H denote an arbitrary non-empty set of concepts $h : \mathcal { X } \to \mathcal { Y }$ , called the concept class.

Our framework, self-directed online learning under apple-tasting feedback, combines two existing models. As in the self-directed online learning model of Ben-David et al. [1995], the learner chooses which point to query in each round. As in the apple-tasting model of Helmbold et al. [2000], feedback is partial: the learner observes the true label when it predicts 1, and receives no feedback when it predicts 0.

Unlike in full-information settings, the adversary may benefit by presenting the learner with multiple copies of the same point. Conversely, the learner may benefit from labeling different copies of the same point differently. We therefore allow the learner to choose points from a multiset $S$ and use the notation $S \subseteq { \mathcal { X } }$ to mean that every $x \in S$ is also in $\mathcal { X } .$

## 3.1 Self-directed online learning under apple-tasting feedback setting

We describe the learning protocol. Given a non-empty concept class H and a finite multiset $S \subseteq { \mathcal { X } }$ , one round of self-directed online learning under apple-tasting feedback proceeds as follows.

1. The learner chooses some $x \in S$

2. The learner predicts a label $\widehat { y } \in \mathcal { V }$ for x.

3. If $\widehat { y } = 1$ , the learner receives the true label $y \in \mathcal { V }$ from the environment. If $\widehat { y } = 0$ , the learner receives no feedback.

Rounds proceed in this fashion until the learner has labeled every point in $S .$ For brevity we say the learner eats a point x if they predict 1 on it.

## 3.2 Definitions

Because feedback is partial under apple tasting, the histories available to the learner and to the adversary differ. When the learner predicts 0, no true label is revealed, so the learner records the symbol ⋆ in its history in place of a label.

Definition 3.1. Given afinite multiset $S \subseteq { \mathcal { X } } ,$ , we formalize an adaptive learner as a pair of functions

$$
\begin{array} { r } { \mathsf { L r n } _ { x } : ( S \times ( \mathcal { V } \cup \{ \star \} ) ) ^ { * } \to \Pi ( S ) , \quad \mathsf { L r n } _ { y } : ( S \times ( \mathcal { V } \cup \{ \star \} ) ) ^ { * } \times S \to \Pi ( \mathcal { V } ) , } \end{array}
$$

Where $\Pi ( S )$ and $\Pi ( \mathcal { V } )$ are the sets of distributions over $S$ and ${ \mathcal { V } } ,$ , respectively. The adversary is a function

$$
\mathsf { A d v } : ( S \times \mathcal { Y } ) ^ { * } \times S \to \Pi ( \mathcal { Y } )
$$

In round $i , \mathsf { L } \mathsf { r } \mathsf { n } _ { x }$ is used in the first step to choose a distribution from which the queried instance $x _ { i }$ is sampled. $\mathsf { L r n } _ { y }$ is used in the second step to determine the prediction $\widehat { y } _ { i }$ of the learner. Adv is used in the third step to determine the correct label $y _ { i } \in \mathcal { V }$

Definition 3.2. We say a labeled sequence $S _ { n } = ( ( x _ { 1 } , y _ { 1 } ) , \dots , ( x _ { n } , y _ { n } ) )$ is realizable under a concept class H if

$$
\operatorname* { m i n } _ { h \in \mathcal { H } } \sum _ { i = 1 } ^ { n } \mathbb { 1 } \left[ h ( x _ { i } ) \neq y _ { i } \right] = 0 .\tag{1}
$$

Let $F _ { \mathsf { L r n } } \in ( S \times ( \mathcal { V } \cup \{ \star \} ) ) ^ { * }$ be the learner’s history. Let $F _ { \mathsf { A d v } } \in ( S \times \mathcal { V } ) ^ { * }$ be adversary’s history. We say an adversary is consistent with H if every history $F _ { \mathsf { A d v } }$ that could arise under their strategy is realizable under H.

We now define the optimal mistake bound against consistent adversaries, commonly known as the realizable setting.

Definition 3.3. For a learner Lrn and an adversary Adv, let $F _ { \mathsf { L } \mathsf { r } \mathsf { n } } ^ { ( i ) }$ and $F _ { \mathsf { A d v } } ^ { ( i ) }$ denote the respective histories at the start ofround i and let

$$
\begin{array} { r } { x _ { i } \sim \mathsf { L r n } _ { x } \Big ( F _ { \mathsf { L r n } } ^ { ( i ) } \Big ) \qquad y _ { i } \sim \mathsf { A d v } \Big ( F _ { \mathsf { A d v } } ^ { ( i ) } , x _ { i } \Big ) \qquad \widehat { y } _ { i } \sim \mathsf { L r n } _ { y } \Big ( F _ { \mathsf { L r n } } ^ { ( i ) } , x _ { i } \Big ) } \end{array}
$$

denote, the instance selected by the learner, the label chosen by the adversary, and the label predicted by the learner in round i respectively.

Let a finite multiset $S \subseteq { \mathcal { X } }$ of size n be given to the learner. Let the game last for m rounds. We define the expected number ofmistakes the learner makes against the adversary by

$$
M ( S , \mathsf { L r n } , \mathsf { A d v } ) = \mathbb { E } \left[ \sum _ { i = 1 } ^ { m } \mathbb { 1 } [ \widehat { y } _ { i } \neq y _ { i } ] \right]\tag{2}
$$

The learner’s history evolves as

$$
\begin{array} { r } { F _ { \mathsf { L } \mathsf { r } \mathsf { n } } ^ { ( i + 1 ) } = \left\{ \begin{array} { l l } { F _ { \mathsf { L } \mathsf { r } \mathsf { n } } ^ { ( i ) } \circ ( x _ { i } , y _ { i } ) } & { i f \widehat { y } _ { i } = 1 , } \\ { F _ { \mathsf { L } \mathsf { r } \mathsf { n } } ^ { ( i ) } \circ ( x _ { i } , \star ) } & { i f \widehat { y } _ { i } = 0 } \end{array} \right. } \end{array}
$$

and the adversary’s history evolves as

$$
F _ { \mathsf { A d v } } ^ { ( i + 1 ) } = F _ { \mathsf { A d v } } ^ { ( i ) } \circ ( x _ { i } , y _ { i } )
$$

where $F _ { \mathsf { L } \mathsf { r } \mathsf { n } } ^ { ( 0 ) } = F _ { \mathsf { A d } \mathsf { v } } ^ { ( 0 ) } = \emptyset$

We define the optimal mistake bound of H against an adversary providing apple-tasting feedback on S by

$$
M ( \mathcal { H } , S ) = \operatorname* { i n f } _ { \mathsf { L r n } } \operatorname* { s u p } _ { \mathsf { A d v } } M ( S , \mathsf { L r n } , \mathsf { A d v } )\tag{3}
$$

where the supremum is taken over all adversaries which are consistent with H.

Finally, we define the optimal mistake bound over over all finite submultisets of X of size at most n by

$$
M _ { \mathrm { o p t i m a l } } ( { \mathcal { H } } , n ) = \operatorname* { s u p } _ { S \subseteq { \mathcal { X } } , | S | \leq n } M ( { \mathcal { H } } , S )\tag{4}
$$

Definition 3.4. We say that a concept class H is learnable under the realizable setting of self-directed online learning with apple-tastingfeedback (or learnable, in short) $i f M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in o ( n ) ^ { 2 }$ , and not learnable $i f M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \Omega ( n )$

If the adversary is not consistent with the concept class then we try to do as well as the best concept in the concept class. This setting is commonly known as a the agnostic setting. As a result, we define expected regret as the number of mistakes made by an algorithm in comparison to the best concept in the concept class.

Definition 3.5. The learner Lrn, the adversary Adv, the history of the learner $F _ { \mathsf { L } \mathsf { r } \mathsf { n } } ^ { ( i ) }$ , the history of the adversary $F _ { \mathsf { A d v } } ^ { ( i ) }$ , the instance selected by the learner $x _ { i } ,$ the label chosen by the adversary $y _ { i }$ and the label predicted by the learner $\widehat { y } _ { i }$ in round i are exactly the same as in Definition 3.3.

For a given finite multiset $S \subseteq { \mathcal { X } }$ ofsize n, let the game go onfor m rounds. We define the expected regret of the learner against the adversary as

$$
R ( S , \mathsf { L r n } , \mathsf { A d v } ) = \mathbb { E } \left[ \sum _ { i = 1 } ^ { m } \mathbb { 1 } [ \widehat { y } _ { i } \neq y _ { i } ] - \operatorname* { i n f } _ { h \in \mathcal { H } } \sum _ { i = 1 } ^ { m } \mathbb { 1 } \left[ h ( x _ { i } ) \neq y _ { i } \right] \right]\tag{5}
$$

We define the optimal expected regret ofH against an adversary providing apple-tastingfeedback on S by

$$
R ( \mathcal { H } , S ) = \operatorname* { i n f } _ { \mathsf { L } \mathsf { r } \mathsf { n } } \operatorname* { s u p } _ { \mathsf { A d v } } R ( S , \mathsf { L } \mathsf { r } \mathsf { n } , \mathsf { A d v } )\tag{6}
$$

where the supremum is taken over all adversaries.

Finally, we define the optimal mistake bound over all finite submultisets of X of size at most n by

$$
R _ { \mathrm { o p t i m a l } } ( { \mathcal { H } } , n ) = \operatorname* { s u p } _ { S \subseteq \mathcal { X } , | S | \leq n } R ( { \mathcal { H } } , S )\tag{7}
$$

Definition 3.6. We say that a concept class H is learnable under self-directed online learning with apple-tasting feedback in the agnostic setting (or agnostic learnable, in short) $i f R _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in o ( n )$ and not agnostic learnable if $R _ { \mathrm { o p t i m a l } } ( { \mathcal { H } } , n ) \in \Omega ( n ) .$ 3

For the sake of completeness, we restate the definition of VC dimension, ∅-centered star number, and self-directed dimension.

Definition 3.7 (VC dimension Vapnik and Chervonenkis, 1971). A set S is VC-shattered by a concept class H if the projection of H onto $S , i . e . , \mathcal { H } | _ { S } ,$ has size $2 ^ { | S | }$ . The VC dimension ofH, denoted $\mathrm { V C } ( \mathscr { H } )$ is the size of the largest set that H VC-shatters, or ∞ if no such largest set exists.

Definition 3.8 (∅-centered star number Hanneke and Yang, 2015, Hanneke, 2024). A set $S$ is ∅- centered star-shattered by a concept class H if:

(1) There exists a concept $h \in \mathcal H$ such that $h ( x ) = 0 f o r a l l x \in S ;$ and

(2) For every $y \in S _ { i }$ , there exists a concept $h \in H$ such that $h ( y ) = 1$ and $h ( x ) = 0$ for all $x \in S \setminus \{ y \}$

The ∅-centered star number of H, denoted ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } )$ , is the size of the largest set that H ∅-centered starshatters, $o r \infty i f$ no such largest set exists.

For an ∅-centered star-shattered set, we say that the concept realizing (1) is the center and the concepts realizing (2) are the witnesses.

The self-directed dimension is the dimension that characterizes the self-directed online learning learning with full-information setting.

Definition 3.9 (Self-directed dimension Devulapalli and Hanneke, 2024). For a concept class H on a finite multiset S, let $\nu \subseteq \varkappa$ . We define $M ( \nu , S )$ by $M _ { \mathrm { f u l l } } ( \nu , S ) = 0$ whenever $| \nu | _ { S } | \leq 1$ , and otherwise

$$
M _ { \mathrm { f u l l } } ( \mathcal { V } , S ) = \operatorname* { m i n } _ { x \in S } \operatorname* { m i n } _ { \hat { y } \in \{ 0 , 1 \} } \operatorname* { m a x } _ { y \in \{ 0 , 1 \} } [ \mathbb { 1 } [ \widehat { y } \neq y ] + M _ { \mathrm { f u l l } } ( \mathcal { V } _ { x  y } , S \setminus \{ x \} ) ] .
$$

The recursion terminates since |S| strictly decreases. The dimension of self-directed learning $o f \mathcal { H }$ is SDdim $\begin{array} { r } { ( \mathcal { H } ) = \operatorname* { s u p } _ { S } M _ { \mathrm { f u l l } } ( \mathcal { H } | _ { S } , S ) } \end{array}$ , the supremum overfinite multisets S.

When SDdim $( \mathcal { H } ) < \infty ,$ , Devulapalli and Hanneke [2024] provide a deterministic algorithm that makes at most SDdim(H) mistakes on any input. We call this the SDSolver algorithm and use it as a black box.

Definition 3.10 (Union-closure). Let H be a nonempty concept class. For a subset $\tau \subseteq \mathcal { H } ,$ , let h<sub>T</sub> denote the conceptfor which $h \tau ( x ) = 1$ ifand only $i f h ( x ) = 1$ for some $h \in \mathcal T$ . Then the union-closure $\mathcal { U } ( \mathcal { H } )$ of $\mathcal { H } \ i s \mathcal { U } ( \mathcal { H } ) = \{ h _ { T } : T \subseteq \mathcal { H } , T \neq \emptyset \}$

Unless stated otherwise, $O , \Omega , \Theta$ hide universal constants which do not depend on any parameter of the problem. Notation such as $\Theta \varkappa$ indicates that the implied constants depend on the concept class H.

## 4 Tetrachotomy in the realizable setting

In this section, we focus on the realizable setting. The following theorem is the main result of this work.   
It expresses a fundamental tetrachotomy: there are exactly four possibilities for optimal learning rates.

Theorem 4.1. For every concept class H exactly one ofthefollowing holds:

• H is not learnable, i.e. ${ M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \Omega ( n ) }$

• H is learnable with $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \widetilde { \Theta } _ { \mathcal { H } } ( \sqrt { n } )$

• H is learnable with $M _ { \mathrm { o p t i m a l } } ( { \mathcal { H } } , n ) \in O _ { { \mathcal { H } } } ( \log n ) \cap \omega _ { { \mathcal { H } } } ( 1 )$

• H is learnable with $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \Theta _ { \mathcal { H } } ( 1 )$

A second main result of this work provides a detailed description of which of these four cases any given concept class H belongs to, by specifying complexity measures to distinguish the cases.

We use the well-known Vapnik Chervonenkis dimension denoted $\mathrm { V C } ( \mathscr { H } )$ as defined in Vapnik and Chervonenkis [1971], the ∅-centered star number, denoted ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } )$ as defined in Hanneke and Yang [2015], and the self-directed dimension SDdim(H) as defined in Devulapalli and Hanneke [2024] to characterize the various cases.

Theorem 4.2. Every concept class H falls into one ofthefollowingfour cases:

$I f \mathrm { V C } ( \mathcal { H } ) = \infty$ then H is not learnable, i.e. $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \Omega ( n )$

$I f \mathrm { V C } ( \mathcal { H } ) < \infty$ and ${ \mathfrak { s } } _ { \varnothing } ( { \mathcal { H } } ) = \infty$ then H is learnable with $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \widetilde { \Theta } _ { \mathcal { H } } ( \sqrt { n } )$

${ \cal I } f \mathfrak { s } _ { \emptyset } ( \mathcal { H } ) < \infty$ and SDdim(H) = ∞ then H is learnable with $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in O _ { \mathcal { H } } ( \log n ) \cap$ $\omega \varkappa ( 1 )$

${ \cal I } f \mathfrak { s } _ { \emptyset } ( \mathcal { H } ) < \infty$ and $\mathrm { S D d i m } ( { \mathcal { H } } ) < \infty$ then H is learnable with $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \Theta _ { \mathcal { H } } ( 1 )$

Since Theorem 4.1 follows immediately from Theorem 4.2, the focus of this work will be to prove Theorem 4.2. The proof of this theorem, and many related results, are presented in the remainder of this paper.

As a warm-up we consider the simplest case.

Theorem 4.3. $I f \mathrm { V C } ( \mathcal { H } ) = \infty$ then H is not learnable, i.e. $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \Omega ( n )$

Proof. Let $n \in \mathbb N$ and let S be a shattered set of size n. The adversary then gives S to the learner and chooses its labels uniformly at random. Since S is shattered by H, these labels are realizable. The learner then averages half a mistake per round, so $M _ { \mathrm { o p t i m a l } } ( { \mathcal { H } } , n ) \geq n / 2 \in \Omega ( n )$ □

## 4.1 Upper Bounds

We now consider the interesting regimes of $\mathrm { V C } ( \mathcal { H } ) < \infty , \mathfrak { s } _ { \emptyset } ( \mathcal { H } ) < \infty$ , or $\mathrm { S D d i m } ( \mathcal { H } ) < \infty . \mathrm { ~ } ^ { 4 }$ We first show the upper bounds.

## 4.1.1 $\mathrm { V C } ( \mathcal { H } ) < \infty$

```latex
Algorithm 1 PRUNE-AND-EAT(H, S)
1: $\overline { { \gamma  \mathcal { H } | _ { S } , n  | S | } }$ ▷ version space: projection of H onto S
2: while S is not empty:
3: $\begin{array} { r } { \mathcal { I }  \{ h \in \mathcal { V } : \sum _ { x \in S } h ( x ) \leq \sqrt { \mathrm { V C } ( \mathcal { H } ) n \log n } \} } \end{array}$ ▷ sparse concepts
4: $\mathcal { V }  \mathcal { V } \setminus \mathcal { I }$
5: if $\nu = \emptyset$ then break
6: $a \gets \arg \operatorname* { m a x } _ { x \in S } \sum _ { h \in \mathcal { V } } h ( x )$ ▷ best apple
7: Predict $\widehat { y } = 1$ on a and receive the true label y
8: $S \gets S \backslash \{ a \}$
9: $\mathcal { V }  \{ h \in \mathcal { V } : h ( a ) = y \} | _ { S }$
10: if $| S | > 0$ then predict $\widehat { y } = 0$ on every remaining $x \in S .$ ▷ no further feedback
```

We first consider the regime in which $\mathrm { V C } ( \mathcal { H } ) < \infty$ . We design an algorithm that helps the learner select relatively safe points to query while progressively eliminating concepts that are either inconsistent with the true concept or which can be safely ignored.

At a high level, PRUNE-AND-EAT alternates between two steps. First, it discards any concepts in the current version space that label too few of the remaining points as 1. This is safe because if the true concept labels almost all remaining points as 0, the learner can afford to predict 0 on all of them and still make few mistakes. Second, among the surviving hypotheses, the algorithm greedily selects the point on which the largest fraction of the concepts predicts 1, and queries its label by predicting it label to be 1. Pruning guarantees that this point is labeled 1 by a t-fraction of the version space for $t = \sqrt { \mathrm { V C } ( \mathscr { H } ) }$ log $\overline { { n / n } }$ . Hence, whenever eating it results in a mistake, that entire t−fraction of concepts is inconsistent with the revealed label and is eliminated. We now state and prove this formally.

Theorem 4.4. Let H be a concept class with $\mathrm { V C } ( \mathcal { H } ) < \infty$ . Then H is learnable by algorithm PRUNE-AND-EAT. Furthermore, we obtain a bound on the optimal number ofmistakes

$$
M _ { \mathrm { o p t i m a l } } ( { \mathcal { H } } , n ) \leq 2 { \sqrt { \mathrm { V C } ( { \mathcal { H } } ) n \log n } } + 1 .
$$

Proof. Let $S = ( x _ { 1 } , \ldots , x _ { n } ) $ be the points chosen by the adversary. Let $h ^ { \ast } \in \mathcal { H }$ denote the (unknown) true labeling concept. Let $\mathcal { V } _ { 0 } = \mathcal { H } | _ { S }$ be the initial version space, and for $i \geq 1$ let $\nu _ { i }$ denote the version space at the start of round i and $\mathcal { V } _ { i } ^ { \mathrm { p r u n e d } }$ its state immediately after the pruning step of that round. Let $S _ { i }$ denote the data set in the $i ^ { \mathrm { t h } }$ round. Every mistake the algorithm makes is of exactly one of two types, according to its prediction:

$M _ { 1 }$ : mistakes on points the algorithm eats, i.e. $\widehat { y } = 1$ while $h ^ { * } ( a ) = 0 ;$

$M _ { 2 } { \mathrm { : } }$ mistakes on points the algorithm defaults to 0 on after exiting the loop i.e. $\widehat { y } = 0$ while $h ^ { * } ( x ) = 1$

Since the algorithm predicts 1 exactly on the points it eats inside the loop and 0 on every point remaining at exit, these two classes partition the mistake rounds, so the total number of mistakes is $M _ { 1 } + M _ { 2 }$

Key observation. Fix a round i and consider the version space $\nu _ { i }$ before pruning. We remove the concepts that are sparse: i.e., if the number of points in $S$ a concept labels as 1 is less than $\sqrt { \mathrm { V C } ( \mathcal { H } ) n \log n } .$ then we remove that concept to obtain $\mathcal { V } _ { i } ^ { \mathrm { p r u n e d } }$ . We note that by construction every $h \in \mathcal { V } _ { i } ^ { \mathrm { p r u n e d } }$ satisfies $\begin{array} { r } { \sum _ { x \in S } h ( x ) > \sqrt { \mathrm { V C } ( \mathcal { H } ) n \log n } } \end{array}$ . Summing over all $h \in \mathcal { V } _ { i } ^ { \mathrm { p r u n e d } }$ and exchanging the order of summation,

$$
\sum _ { x \in S _ { i } } \sum _ { h \in \mathcal { V } _ { i } ^ { \mathrm { p r u n e d } } } h ( x ) = \sum _ { h \in \mathcal { V } _ { i } ^ { \mathrm { p r u n e d } } } \sum _ { x \in S _ { i } } h ( x ) > \left| \mathcal { V } _ { i } ^ { \mathrm { p r u n e d } } \right| \cdot \sqrt { \mathrm { V C } ( \mathcal { H } ) n \log n } .
$$

Since $| S _ { i } | \le n$ , averaging over the points of $S _ { i }$ show that some $x \in S _ { i }$ satisfies

$$
\sum _ { h \in \mathcal { V } _ { i } ^ { \mathrm { p r u n e d } } } h ( x ) > \frac { \left| \mathcal { V } _ { i } ^ { \mathrm { p r u n e d } } \right| } { n } \cdot \sqrt { \mathrm { V C } ( \mathcal { H } ) n \log n } = \left| \mathcal { V } _ { i } ^ { \mathrm { p r u n e d } } \right| \cdot \sqrt { \frac { \mathrm { V C } ( \mathcal { H } ) \log n } { n } } .
$$

Therefore after pruning, some point in $S$ is labeled 1 by at least a $t = { \sqrt { \frac { \operatorname { V C } ( { \mathcal { H } } ) \log n } { n } } }$ fraction of the surviving hypotheses. In particular, the point a chosen by the algorithm, which maximizes this count, satisfies this bound as well. Note that we can assume $t < 1 ; \mathrm { i f } t \geq 1$ , then $n \leq \sqrt { \mathrm { V C } ( \mathcal { H } ) }$ n log n, so the claimed mistake bound is trivial.

Bounding $M _ { 1 }$ . Suppose the algorithm eats a in round i and makes a mistake, so the revealed label is $y = 0$ . By the key observation, more than a t-fraction of $\mathcal { V } _ { i } ^ { \mathrm { p r u n e d } }$ labels $a$ as 1; every such concept is inconsistent with y and is removed. Hence

$$
| \mathcal { V } _ { i + 1 } | \leq ( 1 - t ) \Big | \mathcal { V } _ { i } ^ { \mathrm { p r u n e d } } \Big | \leq ( 1 - t ) | \mathcal { V } _ { i } | ,
$$

using $\mathcal { V } _ { i } ^ { \mathrm { p r u n e d } } \subseteq \mathcal { V } _ { i }$ . Version spaces are non-increasing across all other rounds, so we can chain this inequality over the $M _ { 1 }$ rounds on which an eating mistake occurs. Let $i ^ { * }$ denote the last round completed before the loop terminates. The loop breaks only upon detecting that $\nu$ is empty, so the version space at the start of that round i.e. $\nu _ { i ^ { * } }$ is nonempty, giving $| \nu _ { i ^ { * } } | \geq 1$ . Hence,

$$
1 \leq \vert \mathcal { V } _ { i ^ { \ast } } \vert \leq ( 1 - t ) ^ { M _ { 1 } - 1 } \vert \mathcal { V } _ { 0 } \vert .
$$

We now use Sauer’s Lemma, giving $| \mathcal { V } _ { 0 } | \le n ^ { \mathrm { V C } ( \mathcal { H } ) }$ , and take logarithms to obtain

$$
\mathrm { V C } ( \mathcal { H } ) \log n + ( M _ { 1 } - 1 ) \log ( 1 - t ) \geq 0 .
$$

Using log(1 − t) ≤ −t for $t \in ( 0 , 1 )$

$$
\begin{array} { r } { \operatorname { V C } ( \mathcal { H } ) \log n \geq ( M _ { 1 } - 1 ) t = ( M _ { 1 } - 1 ) \sqrt { \frac { \operatorname { V C } ( \mathcal { H } ) \log n } { n } } \implies M _ { 1 } \leq \sqrt { \operatorname { V C } ( \mathcal { H } ) n \log n } + 1 . } \end{array}
$$

Bounding $M _ { 2 } .$ . If the loop terminates with $S = \emptyset$ then every point was eaten, no point receives the default prediction, and $M _ { 2 } \ = \ 0 .$ So suppose the loop terminates with $\nu ~ = ~ \emptyset$ in some round $j .$ Since $h ^ { * }$ is consistent with every revealed label, it is stays in the version space; the only step that can remove it is pruning. Hence if $M _ { 2 } ~ > ~ 0 .$ , then $h ^ { \ast } ~ \in ~ \mathcal { T }$ at some round $k \ \leq \ j$ , which means $\begin{array} { r } { \sum _ { x \in S _ { k } } h ^ { * } ( x ) \le \sqrt { \mathrm { V C } ( \mathcal { H } ) n \log n } } \end{array}$ . The points receiving the default prediction 0 are exactly those remaining at termination, and $S _ { j } \subseteq S _ { k }$ since S only shrinks. Each such mistake occurs at a point x with $h ^ { * } ( x ) = 1$ , so

$$
M _ { 2 } = \sum _ { x \in S _ { j } } h ^ { * } ( x ) \leq \sum _ { x \in S _ { k } } h ^ { * } ( x ) \leq \sqrt { \mathrm { V C } ( \mathcal { H } ) n \log n } .
$$

Total mistakes. Combining the two bounds,

$$
M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \leq M _ { 1 } + M _ { 2 } \leq 2 \sqrt { \mathrm { V C } ( \mathcal { H } ) n \log n } + 1 .
$$

We now consider the case where ${ \mathfrak { s } } _ { \varnothing } ( { \mathcal { H } } ) < \infty$ . The following theorem will help us resolve the $O ( \log n )$ and $O ( 1 )$ regimes at the same time.

Theorem 4.5. For a concept class H,

$$
M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \leq \operatorname* { m i n } \biggl \{ \mathfrak { s } _ { \emptyset } ( \mathcal { H } ) \cdot \mathrm { S D d i m } ( \mathcal { H } ) , \frac { 1 } { c } \mathfrak { s } _ { \emptyset } ( \mathcal { H } ) \log n \biggr \}
$$

We will prove Theorem 4.5 in two cases: SDdim(H) = ∞ and SDdim(H) < ∞.

## 4.1.2 SDdim(H) = ∞, s<sub>∅</sub>(H) < ∞

Algorithm 2 UNION-AND-EAT(H, S)   
1: $\overline { { \mathcal { V }  \mathcal { U } ( \mathcal { H } ) \vert _ { S } } }$ ▷ Projection of the union closure U(H) as in Def. 3.10 onto S   
2: while there exists $x \in S , h \in \mathcal { V }$ with $h ( x ) = 1 \colon$   
3: $a \gets \arg \operatorname* { m a x } _ { x \in S } \sum _ { h \in \mathcal { V } } h ( x )$ ▷ Best apple   
4: Predict $\widehat { y } = 1$ on a and receive the true label y   
5: $S \gets S \backslash \{ a \}$   
6: $\mathcal { V }  \{ h \in \mathcal { V } : h ( a ) = y \} | _ { S }$   
7: $\mathbf { i f } \left| S \right| > 0$ then predict $\widehat { y } = 0$ on every remaining $x \in S$   
In this case, rather than working with V directly, we pass to its union closure $\mathcal { U } ( \mathcal { V } )$ , obtained by   
taking all pointwise-OR combinations of subsets of V. Learning H is no harder than learning $\mathcal { U } ( \mathcal { H } )$ since   
$\mathcal { H } \subseteq \mathcal { U } ( \mathcal { H } )$ , so any mistake bound for $\mathcal { U } ( \mathcal { H } )$ upper bounds the mistake bound for H. The advantage   
of passing to $\mathcal { U } ( \mathcal { H } )$ is that it is union-closed by construction, and union-closed families enjoy a useful   
averaging property: some point must be labeled 1 by a constant fraction of the family. Consequently,   
greedily querying this point guarantees constant multiplicative progress on every mistake, in contrast to   
Algorithm 1, where progress was only guaranteed after a pruning step. We now prove a few lemmas   
which will be helpful to prove an upper bound based on ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } )$

Lemma 4.6. For any concept class H, ${ \mathfrak { s } } _ { \emptyset } ( \mathcal { H } ) = { \mathfrak { s } } _ { \emptyset } ( \mathcal { U } ( \mathcal { H } ) )$

Proof. (≥) Let $s = \mathfrak { s } _ { \mathfrak { g } } ( \mathcal { U } ( \mathcal { H } ) )$ , and let $P = \{ p _ { 1 } , . . . , p _ { s } \}$ be ∅-centered star-shattered in $\mathcal { U } ( \mathcal { H } )$ . Let $g _ { 0 } \in \mathcal { U } ( \mathcal { H } )$ be the center: i.e. $g _ { 0 } ( p _ { j } ) = 0$ for all $j \in [ s ]$ . Let $g _ { 1 } , \dotsc , g _ { s } \in \mathcal { U } ( \mathcal { H } )$ be the witnesses: i.e. for all $i \in [ s ]$ , we have $g _ { i } ( p _ { i } ) = 1$ and $g _ { i } ( p _ { j } ) = 0$ for all $j \neq i$

We claim that H also ∅-centered star-shatters $P .$ . For the center, write $g _ { 0 } = h _ { 0 , 1 } \vee \cdots \vee h _ { 0 , m _ { 0 } }$ for some $h _ { 0 , 1 } , \hdots , h _ { 0 , m _ { 0 } } \in \mathcal { H }$ . Because $g _ { 0 } ( p _ { j } ) = 0$ for every $j \in [ s ]$ , and a union of concepts is 0 at a point only if every concept in the union is 0 there, each $h _ { 0 , k _ { i } }$ satisfies $h _ { 0 , k _ { i } } ( p _ { j } ) = 0$ for all $j \in [ s ]$

We can similarly extract the witnesses from the other $g _ { i }$ . Fix $i \in [ s ]$ and write $g _ { i } = h _ { i , 1 } \vee \cdots \vee h _ { i , m _ { i } }$ for some $h _ { i , 1 } , \ldots , h _ { i , m _ { i } } \in \mathcal { H }$ . Since $g _ { i } ( p _ { j } ) = 0$ for every $j \neq i ,$ each $h _ { i , k }$ satisfies $h _ { i , k } ( p _ { j } ) = 0$ for all $j \neq i .$ . Since $g _ { i } ( p _ { i } ) = 1$ , at least one $h _ { i , k _ { i } } \in \mathcal { H }$ satisfies $h _ { i , k _ { i } } ( p _ { i } ) = 1$ . Combining these two facts, $h _ { i , k _ { i } } \in \mathcal { H }$ agrees with g<sub>0</sub> everywhere on $P$ except at $p _ { i }$ , where it equals 1. Repeating this for every $i \in [ s ]$ produces concepts $h _ { 1 , k _ { 1 } } , \ldots , h _ { s , k _ { s } } \in$ H that, together with $h _ { 0 , 1 } \in { \mathcal { H } }$ , ∅-centered star-shatter $P .$ Hence ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } ) \geq s = { \mathfrak { s } } _ { \emptyset } ( { \mathcal { U } } ( { \mathcal { H } } ) )$

(≤) Since $\mathcal { U } ( \mathcal { H } )$ contains the singleton union of each concept in $\mathcal { H } .$ we have $\mathcal { H } \subseteq \mathcal { U } ( \mathcal { H } )$ . Hence any set ∅-centered star-shattered by H is also shattered by $\mathcal { U } ( \mathcal { H } )$ , so ${ \mathfrak { s } } _ { \emptyset } ( \mathcal { U } ( \mathcal { H } ) ) \geq { \mathfrak { s } } _ { \emptyset } ( \mathcal { H } )$ □

## Lemma 4.7. IfH is a union-closed concept class then $\mathrm { V C } ( \mathcal { H } ) = \mathfrak { s } _ { \varnothing } ( \mathcal { H } )$

Proof. (≥) Let $s = \mathfrak { s } _ { \mathfrak { d } } ( \mathcal { H } )$ , and let $P = \{ p _ { 1 } , . . . , p _ { s } \}$ be ∅-centered star-shattered by center $h _ { 0 }$ and witnesses $h _ { 1 } , \ldots , h _ { s } \in \mathcal { U } \colon h _ { 0 }$ labels every point of $P$ as 0, and $h _ { i }$ agrees with $h _ { 0 }$ except that $h _ { i } ( p _ { i } ) = 1$

We claim that H shatters P. The concept $h _ { 0 }$ realizes the all-zero labeling of $P .$ For any nonzero target labeling $y = ( y _ { 1 } , \dots , y _ { s } ) \in \{ 0 , 1 \} ^ { s }$ of $P _ { \cdot }$ , let ${ \mathcal { W } } = \{ h _ { i } : y _ { i } = 1 \} \subseteq \{ h _ { 1 } , \ldots , h _ { s } \}$ . Let $\boldsymbol { h _ { \mathcal { W } } } = \boldsymbol { \vee } _ { h \in \mathcal { W } } \boldsymbol { h }$ . Because H is union-closed and W is nonempty, we have $h _ { \mathcal { W } } \in \mathcal { H }$ . Since each $h _ { i }$ labels only $p _ { i }$ as 1 among $P , h _ { \mathcal { W } } ( p _ { i } ) = 1$ if and only if $h _ { i } \in \mathcal W$ which is if and only if $\boldsymbol { y } _ { i } = 1$ . Hence H realizes every labeling of P, so $\mathrm { V C } ( \mathscr { H } ) \geq s$

(≤) Let $v = \mathrm { V C } ( \mathscr { H } )$ , and let $S ^ { \prime } = \{ q _ { 1 } , \ldots , q _ { v } \}$ be shattered by H. By shattering, for each $i \in [ v ]$ there is a concept $g _ { i } \in \mathcal { H }$ realizing the labeling that assigns $q _ { i } \mapsto 1$ and $q _ { j } \mapsto 0$ for $j \neq i$ . There is also a concept $g _ { 0 } \in \mathcal { H }$ realizing the all-0 labeling. Then $S ^ { \prime }$ is ∅-centered star-shattered by center $g _ { 0 }$ and witnesses $g _ { 1 } , \ldots , g _ { v } , \mathsf { s o } \mathfrak { s } _ { \varnothing } ( \mathcal { H } ) \ge v$ □

Proposition 1. There is a universal constant $c > 0$ such that for every finite union-closed class H of concepts (other than the trivial class), some point x satisfies

$$
| \{ h \in { \mathcal { H } } : h ( x ) = 1 \} | \geq c \cdot | { \mathcal { H } } | .
$$

We use c as a placeholder for the best constant available. At the time of writing Gilmer [2022] establish $c = 0 . 0 1$ , Chase and Lovett [2022], Pebody [2022], Sawin [2023], Alweiss et al. [2024] establish $\begin{array} { r } { c = \frac { 3 - \sqrt { 5 } } { 2 } \approx 0 . 3 8 1 9 7 . } \end{array}$ , and finally Liu [2023] establish $c = 0 . 3 8 2 7 1$ . Please refer to Bruhn and Schaudt [2013] for the history of union-closed set conjecture (also known as Frankl’s conjecture) which suggests $\begin{array} { r } { c = \frac { 1 } { 2 } } \end{array}$ . The exact value is not relevant for our purposes as long as $c > 0$

Theorem 4.8. Let H be a concept class with ${ \mathfrak { s } } _ { \varnothing } ( \mathcal { H } ) < \infty$ . Then H is learnable by algorithm UNION-AND-EAT. Furthermore, we obtain a bound on the optimal number ofmistakes

$$
M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \leq \frac { 1 } { c } \mathfrak { s } _ { \emptyset } ( \mathcal { H } ) \log n ,
$$

where $c \in ( 0 , 1 / 2 ]$ is the universal constantfrom the union-closed set conjecture (Proposition 1).

Proof. For $i \geq 1$ , let $\nu _ { i }$ denote the version space at the start of the ith iteration of the while loop. Since we only ever remove concepts for being inconsistent with the true concept, there is always a concept consistent with the true concept in $\nu _ { i }$ . As we only exit the while loop once the remaining concepts label all remaining points as 0, we make no mistakes when labeling the remaining points 0 in step 7.

That leaves the false positive mistakes made when predicting 1 in step 4. We use Proposition 1 to bound these mistakes, so we need to make sure that the version spaces $\nu _ { i }$ are all union-closed. We show this by induction on i. As a base case, the initial version space $\mathcal { V } _ { 1 }$ is the projection of the union-closed concept class $\mathcal { U } ( \mathcal { H } )$ onto $S ,$ so it is union-closed. Now assume $\nu _ { i }$ is union-closed for some fixed i. The next version space $\nu _ { i + 1 }$ is constructed in two steps:

• First, it is restricted to the concepts which map a to the true label $y .$ Let $h _ { 1 } , h _ { 2 }$ be two such concepts. Since $\nu _ { i }$ is union-closed, we have $h _ { 1 } \vee h _ { 2 } \in \mathcal { V } _ { i }$ . Moreover, since $( h _ { 1 } \lor h _ { 2 } ) ( a ) =$ $h _ { 1 } ( a ) \vee h _ { 2 } ( a ) = y \vee y = y$ , it remains in the version space.

• Second, it is projected onto $S \setminus \{ a \}$ . As in the base case, this projection preserves union-closure.

Therefore we can guarantee that the point a selected in each round of UNION-AND-EAT satisfies $| \{ h \in$ $\mathcal { V } _ { i } : h ( a ) = 1 \} | \geq c \cdot | \mathcal { V } _ { i } |$ , where c is the constant in Proposition 1. Whenever the algorithm makes a mistake at a its true label is 0, so every $h \in \mathcal { V } _ { i }$ with $h ( a ) = 1$ is inconsistent and is discarded from the next round’s version space. Let M be the number of mistakes made in total. Since no $\nu _ { i }$ is empty, we have:

$$
( 1 - c ) ^ { M } \cdot | \mathcal { V } _ { 1 } | \geq 1 ,
$$

By Lemmas 4.6 and 4.7, we have $\mathrm { V C } ( \mathcal { V } _ { 1 } ) = \mathfrak { s } _ { \varnothing } ( \mathcal { V } _ { 1 } ) = \mathfrak { s } _ { \varnothing } ( \mathcal { H } | _ { S } ) \le \mathfrak { s } _ { \varnothing } ( \mathcal { H } )$ , so by Sauer’s Lemma:

$$
\vert \mathcal { V } _ { 1 } \vert \leq n ^ { \mathrm { V C } ( \mathcal { V } _ { 1 } ) } \leq n ^ { \mathfrak { s } _ { \varnothing } ( \mathcal { H } ) } .
$$

Taking logarithms and using log $( 1 - c ) \leq - c .$ , we obtain

$$
0 \leq \mathfrak { s } _ { \emptyset } ( \mathcal { H } ) \log n - c M \implies M \leq \frac { 1 } { c } \mathfrak { s } _ { \emptyset } ( \mathcal { H } ) \log n .
$$

So algorithm UNION-AND-EAT makes at most ${ \underline { { \boldsymbol { \mathbf { 1 } } } } } _ { \pmb { \mathfrak { s } } _ { \emptyset } } ( { \mathcal { H } } )$ log n mistakes as desired.

## 4.1.3 SDdim(H) < ∞, s<sub>∅</sub>(H) < ∞

Algorithm 3 FIRST-MISTAKE $( \mathcal { V } , ( x _ { 1 } , x _ { 2 } , \ldots , x _ { b } ) , x _ { b + 1 } )$   
1: mis ← ∞   
2: if $x _ { b + 1 } \neq \perp$ then   
3: Predict $\widehat { y } = 1$ for $x _ { b + 1 }$ and obtain the true label $y .$   
4: if $y = 1$ then return ⊥.   
5: $\mathcal { V } \left. \mathcal { V } _ { x _ { b + 1 } \right. 0 } ,$ mis $ b + 1$   
6: Let $f ( h ) = \operatorname* { m i n } \{ j \in [ b ] : h ( x _ { j } ) = 1 \}$ for all $h \in \mathcal { V } ,$ with min $\varnothing = \infty .$   
7: while there is some $h \in \mathcal V$ such that $f ( h ) <$ mis do   
8: Let $j =$ max $\{ f ( h ) : h \in \mathcal { V } , f ( h ) <$ mis}   
9: Predict $\widehat { y } = 1$ at $x _ { j }$ and obtain the true label $y .$   
10: $\mathcal { V } \left. \mathcal { V } _ { x _ { j } \right. y }$   
11: if $y = 1$ then mis $ j$   
12: if mis = ∞ then return $\perp$ else return $x _ { \mathrm { m i s } }$

$j ^ { * } = \operatorname* { m i n } \{ j \in [ b ] : h ^ { * } ( x _ { j } ) = 1 \}$ or ∞ if no such point exists

Algorithm 4 SIMULATE-AND-EAT(H, S, SDSolver)   
1: $\mathcal { V }  \mathcal { H } | _ { S }$   
2: while V contains concepts that disagree at a point in $S$ do:   
3: Let $n  \vert S \vert , S ^ { 0 }  \varnothing , x _ { \mathrm { m i s } }  \perp$   
4: Simulate a run of SDSolver $( \nu , S )$ assuming all predictions are correct.   
5: Let the order and the predictions in the previous step be σ, $\big ( \widehat { y } _ { \sigma ( 1 ) } , \dots , \widehat { y } _ { \sigma ( n ) } \big )$ resp.   
6: Let $x _ { \sigma ( n + 1 ) } = \bot , \widehat { y } _ { \sigma ( n + 1 ) } = 1$   
7: for i goes from 1 to n + 1 do   
8: if $\widehat { y } _ { \sigma ( i ) } = 0$ then $S ^ { 0 } \gets S ^ { 0 } \circ \left\{ x _ { \sigma ( i ) } \right\}$   
9: if $\widehat { y } _ { \sigma ( i ) } = 1$ then   
10: x ← FIRST-MISTAKE $\left( \gamma , S ^ { 0 } , x _ { \sigma ( i ) } \right)$   
11: if $x _ { \mathrm { m i s } } \neq \perp$ then   
12: Let R be the set of points before $x _ { \mathrm { m i s } }$ in $S$ w.r.t. order $\sigma .$   
13: Label all the points in R according to the labels given by SDSolver.   
14: $S \gets S \setminus ( R \cup \{ x _ { \mathrm { m i s } } \} )$   
15: $\mathcal { V }  \{ h \in \mathcal { V } : h ( x _ { i } ) = \widehat { y } _ { i }$ for all $x _ { i } \in R$ and $h ( x _ { \mathrm { m i s } } ) = 1 - \widehat { y } _ { \mathrm { m i s } } \}$   
16: break   
17: if $x _ { \mathrm { m i s } } = \perp$ then   
18: Label all the points in S according to the labels given by SDSolver and break   
19: Predict all remaining points according to the concepts in V

In this case, the idea is a kind of reduction from apple-tasting feedback to the full-information case. We make ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } )$ mistakes to recognize the first position at which the full-information learner makes a mistake. We know that in full-information, we can obtain the true concept in SDdim(H) mistakes. Therefore, we obtain the true concept in at most $\mathfrak { s } _ { \emptyset } ( \mathcal { H } ) \times \mathrm { S D d i m } ( \mathcal { H } )$ mistakes with apple-tasting feedback.

The following lemma shows that we can obtain the first position where the full-information (selfdirected) learner makes a mistake using apple-tasting feedback with at most ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } )$ mistakes.

Lemma 4.9. Let $( x _ { 1 } , \ldots , x _ { b } ) , x _ { b + 1 }$ be points provided to FIRST-MISTAKE. Let V be a version space such that $h ^ { \ast } \in \mathcal { V }$ and there exists $h ^ { b } \in \mathcal { V }$ such that $h ^ { b } ( x _ { i } ) = 0 f o r a l l i \in [ b ]$ and, $i f x _ { b + 1 } \neq \perp$ , then $h ^ { b } ( x _ { b + 1 } ) = 1$ . Let

Then one of these three cases occurs:

$H h ^ { * } ( x _ { b + 1 } ) = 1$ then FIRST-MISTAKE returns ⊥ and makes no mistakes.

$I f h ^ { \ast } ( x _ { b + 1 } ) = 0$ then FIRST-MISTAKE returns $x _ { \operatorname* { m i n } \{ j ^ { * } , b + 1 \} }$ and makes at most ${ \mathfrak { s } } _ { \emptyset } ( \nu )$ mistakes.

$I f x _ { b + 1 } = \perp$ then FIRST-MISTAKE returns $x _ { j ^ { * } } i f j ^ { * } < \infty$ and ⊥ otherwise. It also makes at most ${ \mathfrak { s } } _ { \emptyset } ( \nu )$ mistakes.

Proof. Throughout, every update is of the form $\mathcal { V } \left. \mathcal { V } _ { x \right. y }$ where y is the true label revealed at $x ;$ hence $h ^ { \ast } \in \mathcal { V }$ at all times, and V only shrinks. Note also that $f ( h ) = \operatorname* { m i n } \{ j \in [ b ] : h ( x _ { j } ) = 1 \}$ depends only on $h ,$ , so it is unaffected by these updates.

Case 1: $h ^ { * } ( x _ { b + 1 } ) = 1$ . In this case, the lemma statement is immediate from the algorithm: line 3 predicts $\widehat { y } = 1$ at $x _ { b + 1 }$ and the revealed label is $h ^ { * } ( x _ { b + 1 } ) = 1$ , so line 4 returns ⊥ immediately and we make no mistakes.

In the other two cases, the algorithm does non-trivial work. We first make two observations about this work by linking the variable mis to $j ^ { * }$

1. mis $\geq j ^ { * }$ when $j ^ { * } < \infty ;$ mis is lowered only at line 11, upon a probe at $x _ { j }$ revealing $y = 1$ , and then $h ^ { * } ( x _ { j } ) = 1$ forces $j \geq j ^ { * }$ . In case 2 the initial value mis $= b + 1$ satisfies this since $j ^ { * } \in [ b ]$ and in case 3 the initial value is ∞.

2. The loop cannot terminate while mis $> j ^ { * } \colon \mathrm { i f ~ } j ^ { * } < \infty$ then $h ^ { \ast } \in \mathcal { V }$ satisfies $f ( h ^ { * } ) = j ^ { * } <$ mis, so condition for the loop at line 7 holds.

Combining the two, the loop terminates with mis $= j ^ { * }$ whenever $j ^ { * } < \infty$

Correctness in cases 2 and 3.

Case 2: $h ^ { \ast } ( x _ { b + 1 } ) = 0$ . Line 3 incurs one mistake, line 5 sets mis $ b + 1$ , and the loop runs. If $j ^ { * } \leq b$ the observations give mis $= j ^ { * }$ on termination, and line 12 returns $x _ { j ^ { * } } . \operatorname { I f } j ^ { * } = \infty .$ , no probe can reveal $y = 1$ (that would force $j \geq j ^ { * } = \infty )$ , so mis remains $b + 1$ and line 12 returns $x _ { b + 1 }$ . Therefore, the returned point is $x _ { \operatorname* { m i n } \{ j ^ { * } , b + 1 \} }$

Case 3: $x _ { b + 1 } = \perp$ . Here mis $= \infty$ on entry to the loop. If $j ^ { * } < \infty$ , then mis $= j ^ { * }$ on termination and line 12 returns $x _ { j ^ { * } } . \operatorname { I f } j ^ { * } = \infty$ , then no probe reveals $y = 1$ , so mis remains ∞ and line 12 returns ⊥.

## Mistake bound for cases 2 and 3.

A probe is a mistake exactly when the revealed label is 0. Let $j _ { 1 } > j _ { 2 } > \cdots > j _ { k }$ be the indices in [b] of the loop probes revealing 0, and in Case 2 put additionally $j _ { 0 } : = b + 1$ , the index of the failed initial probe. Set $P = \{ x _ { j _ { i } } \}$ , over $i \in \{ 0 , 1 , \ldots , k \}$ in Case 2 and $i \in \{ 1 , \ldots , k \}$ in Case 3. The number of mistakes is exactly $| P |$ , so it suffices to show that the initial V ∅-centered star-shatters $P$

The center. Each $j _ { i }$ was a probe revealing the true label 0, so $h ^ { * } ( x _ { j _ { i } } ) = 0$ for every i, and $h ^ { \ast } \in \mathcal { V }$ by hypothesis. Hence $h ^ { * }$ labels every point of $P { \mathrm { ~ a s ~ 0 ~ } }$

The witnesses. For $i = 0 \ : ( \mathrm { c a s e 2 o n l y ) }$ take $h ^ { b }$ , which by hypothesis lies in V, satisfies $h ^ { b } ( x _ { b + 1 } ) = 1$ and satisfies $h ^ { b } ( x _ { j } ) = 0$ for every $j \in [ b ]$ , hence in particular at $x _ { j _ { 1 } } , \ldots , x _ { j _ { k } }$

For $i \geq 1$ , let $\nu _ { i } \subseteq \nu$ denote the version space immediately before the i-th failed loop probe. By the selection rule at line 8 there is $h _ { i } \in \mathcal V _ { i }$ with $f ( h _ { i } ) = j _ { i }$ . Then

$h _ { i } ( x _ { j _ { i } } ) = 1$ , directly from $f ( h _ { i } ) = j _ { i }$

$h _ { i } ( x _ { j _ { i ^ { \prime } } } ) = 0$ for every $i ^ { \prime } > i ,$ , since $j _ { i ^ { \prime } } < j _ { i } = f ( h _ { i } )$ and $f ( h _ { i } )$ is the least index of [b] at which $h _ { i }$ takes the value 1.

$h _ { i } ( x _ { j _ { i ^ { \prime } } } ) = 0$ for every $i ^ { \prime } < i ,$ including $i ^ { \prime } = 0$ in case 2, since $h _ { i } \in \mathcal V _ { i }$ and each of those earlier probes revealed the label 0, after which line 10 (respectively line 5) discarded every concept taking the value 1 there.

So $h _ { i }$ is 1 at $x _ { j _ { i } }$ and 0 on $P \setminus \{ x _ { j _ { i } } \}$ , and $h _ { i } \in \mathcal V$ because $\nu _ { i } \subseteq \nu _ { \cdot }$

Together with the center $h ^ { * }$ , these witnesses ∅-centered star-shatter P, hence $| P | \leq { \mathfrak { s } } _ { \emptyset } ( \nu )$ and the algorithm makes at most ${ \mathfrak { s } } _ { \emptyset } ( \nu )$ mistakes. □

Theorem 4.10. Let H be a concept class with ${ \mathfrak { s } } _ { \varnothing } ( \mathcal { H } ) < \infty$ and $\mathrm { S D d i m } ( { \mathcal { H } } ) < \infty$ . Then H is learnable by algorithm SIMULATE-AND-EAT. Furthermore, we also obtain a bound on the optimal number of mistakes

$$
M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \leq \mathfrak { s } _ { \varnothing } ( \mathcal { H } ) \cdot \mathrm { S D d i m } ( \mathcal { H } )
$$

Proof. We use induction on SDdim(H). As a base case, if $\mathrm { S D d i m } ( \mathscr { H } ) = 0 .$ , then H contains only one concept, so the main loop terminates immediately and predicts all points correctly.

Now assume that, for some $d \geq 0$ , SIMULATE-AND-EAT is correct and achieves the desired mistake bound on all concept classes H with $\mathrm { S D d i m } ( \mathscr { H } ) \leq d .$ . Let $\mathcal { H ^ { \prime } }$ be an arbitrary concept class with SDdim $( \mathcal { H ^ { \prime } } ) = d + 1$ , let S be a multiset, and let $h ^ { \ast } \in \mathcal { H } ^ { \prime }$ be the true labeling concept.

We now run SIMULATE-AND-EAT on H<sup>′</sup> and S. Let σ be the order in which SDSolver probes the points in S and let $\big ( \big ( x _ { \sigma ( 1 ) } , \widehat { y } _ { \sigma ( 1 ) } \big ) \big ) , \dots , \big ( \big ( x _ { \sigma ( n ) } , \widehat { y } _ { \sigma ( n ) } \big ) \big )$ denote the labels provided by SDSolver assuming that all the predictions it made are correct. We will use Lemma 4.9 to characterize the behavior of the calls to FIRST-MISTAKE, so we need to show that its conditions hold:

• We always have $h ^ { \ast } \in \mathcal { V }$ since $h ^ { \ast } \in \mathcal { H } ^ { \prime }$

• Since SDSolver is consistent, the concept h it follows satisfies $h ( x ) = 0$ for all $x \in S ^ { 0 }$ and $h ( x _ { \sigma ( i ) } ) = 1$ for all i such that $\widehat y _ { \sigma ( i ) } = 1$ . This concept satisfies the condition of $h ^ { b }$ in Lemma 4.9.

We now briefly handle the case where all labels provided by SDSolver are correct. Here, the only call to FIRST-MISTAKE which can make a mistake is the final one with $x _ { \sigma ( n + 1 ) } = \perp$ . By case 3 of Lemma 4.9, this call will make at most ${ \mathfrak { s } } _ { \emptyset } ( \mathcal { H } ^ { \prime } )$ mistakes and return ⊥. Line 17 then correctly labels all points as SDSolver did. Since $d + 1 \geq 1$ , we make at most ${ \mathfrak { s } } _ { \emptyset } ( \mathcal { H } ^ { \prime } )$ · SDdim(H<sup>′</sup>) mistakes as desired.

Now that at least one $\widehat { y } _ { \sigma ( i ) }$ is incorrect, let $i ^ { * }$ be the first index for which $h ^ { \ast } ( x _ { \sigma ( i ^ { \ast } ) } ) \neq \widehat { y } _ { \sigma ( i ^ { \ast } ) }$ . We claim that $x _ { \mathrm { m i s } }$ will be set to $x _ { \sigma ( i ^ { * } ) }$ , so that the updates on lines 14 and 15 correctly assign labels to $x _ { \sigma ( 1 ) } , \ldots , x _ { \sigma ( i ^ { * } ) }$ . Let i be the first index for which the call FIRST-MISTAKE $( \gamma , S ^ { 0 } , x _ { \sigma ( i ) } )$ does not fall into case 1 of Lemma 4.9; such an index exists because $x _ { \sigma ( n + 1 ) } = \perp$ . We have $i \geq i ^ { * }$ because ${ \mathrm { i f ~ } } i \leq n$ then $h ^ { \ast } ( x _ { \sigma ( i ) } ) = 0 \neq \widehat { y } _ { \sigma ( i ) }$ . We then have two cases based on $\widehat { y } _ { \sigma ( i ^ { * } ) } .$

• If $\widehat { y } _ { \sigma ( i ^ { * } ) } = 1$ , then we have $i \leq i ^ { * }$ , so $i = i ^ { * }$ . The call to FIRST-MISTAKE therefore falls under case 2 of Lemma 4.9. At this time, the true labels of all points in $S ^ { 0 }$ are 0, so FIRST-MISTAKE returns $x _ { \sigma ( i ^ { * } ) }$ as desired and makes at most ${ \mathfrak { s } } _ { \emptyset } ( \mathcal { H } ^ { \prime } )$ mistakes.

• If $\widehat { y } _ { \sigma ( i ^ { * } ) } = 0$ , then $x _ { \sigma ( i ^ { * } ) }$ is the first incorrectly labeled point in $S ^ { 0 }$ . We may be in either case 2 or 3 of Lemma $4 . 9$ , but in either case this guarantees that $x _ { \sigma ( i ^ { * } ) }$ is returned and at most ${ \mathfrak { s } } _ { \emptyset } ( \mathcal { H } ^ { \prime } )$ mistakes are made.

Therefore, lines 14 and 15 correctly label the prefix $x _ { \sigma ( 1 ) } , \ldots x _ { \sigma ( i ^ { * } ) }$ and update the version space accordingly. Let H be this new version space. By the construction of SDSolver, we have SDdim $( { \mathcal { H } } ) \leq$ $\mathrm { S D d i m } ( \mathcal { H } ^ { \prime } - 1 ) = d .$ The next iteration of the loop is indistinguishable from running SIMULATE-AND-EAT with H as the starting concept class, so by the inductive hypothesis it makes at most ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } )$ SDdim $( \mathcal { H } ) \le \mathfrak { s } _ { \varnothing } ( \mathcal { H } ^ { \prime } )$ · d more mistakes. Adding in the ${ \mathfrak { s } } _ { \emptyset } ( \mathcal { H } ^ { \prime } )$ mistakes from the first iteration we get at most ${ \mathfrak { s } } _ { \emptyset } ( \mathcal { H } ^ { \prime } ) \cdot \operatorname { S D d i m } ( \mathcal { H } ^ { \prime } )$ mistakes, completing the proof. □

ProofofTheorem 4.5. We can combine Theorem 4.8 and Theorem 4.10 to obtain $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \leq$ min $\left\{ \mathfrak { s } _ { \emptyset } ( \mathcal { H } ) \mathrm { S D d i m } ( \mathcal { H } ) , \frac { 1 } { c } \mathfrak { s } _ { \emptyset } ( \mathcal { H } ) \log n \right\}$ □

## 4.2 Lower Bounds

We now take a look at the lower bounds. Most of our lower bounds are based on an extension of the singletons concept class which we call block singletons. Suppose a concept class H satisfies ${ \mathfrak { s } } _ { \emptyset } ( \mathcal { H } ) \geq$ d for some d $\in \mathbb { N }$ . We then make a multiset of instances S by including d copies of each of the d elements of an ∅-centered star-shattered set. This set is given by the adversary to the learner at the beginning of the game. By definition, for each of these blocks of d apples there exists a concept in H that labels only that block as 1. Intuitively, any self-directed learner with apple-tasting feedback should make roughly d mistakes on this multiset of instances: either they eat apples from few blocks (and miss the d that are labeled 1), or they eat apples from many blocks (including about d that are labeled 0).

Theorem 4.11. Let $d \in \mathbb { N }$ and let H be a concept class. $I f { \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } ) \geq d ,$ then there exists a multiset $S o f$ size $d ^ { 2 }$ on which any randomized self-directed learner with apple-tasting feedback makes at least $\frac { 2 d ^ { 2 } } { 3 d + 1 }$ mistakes in expectation.

Proof. Let $\{ x _ { 1 } , \ldots , x _ { d } \}$ be a set that H ∅-centered star-shatters. Let S be a multiset consisting of d copies of each of $x _ { 1 } , \ldots , x _ { d }$ . This multiset is given by the adversary to the learner at the beginning of the game. By definition, there exist concepts $h _ { 1 } , \ldots , h _ { d } \in { \mathcal { H } }$ such that, for each $i \in \{ 1 , \ldots , d \}$ , we have $h _ { i } ( x _ { i } ) = 1$ and $h _ { i } ( x _ { j } ) = 0$ for $j \neq i$ . Each of these concepts has one good block labeled 1 and $d - 1$ bad blocks labeled 0. There also exists a concept $h _ { 0 }$ for which every block is bad. We have the adversary choose the following distribution over these concepts at the beginning of the game:

• With probability $\frac { 2 d } { 3 d + 1 }$ , choose one of $h _ { 1 } , \ldots , h _ { d }$ uniformly at random.

• With probability $\textstyle { \frac { d + 1 } { 3 d + 1 } }$ , choose $h _ { 0 }$

We claim that under this distribution no randomized learner can make fewer than $\frac { 2 d ^ { 2 } } { 3 d + 1 }$ mistakes on average.

Let A be an arbitrary randomized learning algorithm. We make a few observations which will aid our analysis.

• We can assume without loss of generality that A samples one point from each of the blocks in an arbitrary order and predicts 1 on them until it either finds a good block or decides to stop; any other strategy makes unnecessary mistakes.

• If the algorithm stops because it found a good block, then it will predict 1 for all the apples in that block and then predict 0 for all other apples. This corresponds to the learner’s discovery that the true concept is in $\{ h _ { 1 } , \ldots , h _ { d } \}$

• If the algorithm stops because it decided to terminate, then it will predict 0 for all other apples. This corresponds to the learner’s belief that the true concept is $h _ { 0 }$

• Since the only feedback A gets before stopping is a series of bad blocks, we can further assume that before making any samples it chooses some (possibly random) number k of bad blocks to sample before stopping.

We now compute the expected number of mistakes A makes for each fixed choice of $k \in \{ 0 , \ldots , d \}$ which denotes the number of bad blocks the learner sampled from. There are three possibilities, whose probabilities and mistake counts depend on k:

• In the first case the algorithm finds a good block at some $i \leq k$ . This indicates the true concept is in $\{ h _ { 1 } , \ldots , h _ { d } \}$ . In this case, the algorithm makes i − 1 mistakes.

• In the second case the algorithm does not find a good block but the adversary had chosen the true concept from $\{ h _ { 1 } , \ldots , h _ { d } \}$ . In this case, the algorithm makes $d + k$ mistakes.

• In the third case the algorithm does not find a good block and the adversary chose $h _ { 0 }$ as the true concept. In this case, the algorithm makes k mistakes.

We now combine these three cases to compute the expected number of mistakes made by the algorithm.

$$
\begin{array} { r }  \mathbb { E } [ \mathcal { A } ^ { \prime } \mathrm { s ~ m i s t a k e s ~ } | \ k ] = \underbrace { \frac { 2 d } { 3 d + 1 } } _ { \begin{array} { l } { \mathrm { f r s t ~ c a s e ~ } } \\ { \mathrm { ~ } } \\ { \mathrm { ~ } } \\ { \mathrm { ~ } } \\ { \mathrm { ~ } } \end{array} } + \underbrace { \frac { 1 } { d } \cdot \sum _ { i = 1 } ^ { k } ( i - 1 ) } _ { \begin{array} { l } { \mathrm { f i r s t ~ c a s e ~ } } \\ { \mathrm { ~ } } \\ { \mathrm { ~ } } \end{array} } + \underbrace { \frac { 2 d } { 3 d + 1 } \cdot \frac { d - k } { d \ } \cdot ( d + k ) } _ { \begin{array} { l } { \mathrm { s c o n d ~ c a s e ~ } } \\ { \mathrm { ~ } } \\ { \mathrm { ~ } } \\ { \mathrm { ~ } } \end{array} } + \underbrace { \frac { d + 1 } { 3 d + 1 } \cdot k } _ { \begin{array} { l } { \mathrm { f i r s t ~ c a s e ~ } } \\ { \mathrm { f i r i d ~ c a s e ~ } } \\ { \mathrm { ~ } } \end{array} } \end{array}
$$

Since $0 \leq k \leq d , k ( d - k ) \geq 0$ , so we have:

$$
\mathbb { E } [ { \mathcal { A } } ^ { * } { \mathrm { s } } { \mathrm { m i s t a k e s } } \mid k ] \geq { \frac { 2 d ^ { 2 } } { 3 d + 1 } }
$$

Since this inequality holds for all k, it holds for any distribution over k, and therefore A makes at least $\frac { 2 d ^ { 2 } } { 3 d + 1 }$ mistakes in expectation. □

Note that the adversary in Lemma 4.11 commits to a distribution over the true concepts before beginning the game with the learner. This adversary is weaker than the general case, since our model allows choosing the labels adaptively. We call this an oblivious adversary. We also refer readers to section 6 where we obtain a tight upper bound for a game against an oblivious adversary when $\mathrm { V C } ( \mathscr { H } ) <$ ∞.

Since this block singleton structure exists in any concept class with ${ \mathfrak { s } } _ { \varnothing } ( \mathcal { H } ) ~ \geq ~ d .$ , it is especially powerful when ${ \mathfrak { s } } _ { \varnothing } ( { \mathcal { H } } ) = \infty$ . We use this idea to show the following theorem.

Theorem 4.12. For any concept class H, $i f \mathfrak { s } _ { \varnothing } ( \mathcal { H } ) = \infty t h e n M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \Omega ( \sqrt { n } ) .$

Proof. Let H be a concept class with ${ \mathfrak { s } } _ { \varnothing } ( { \mathcal { H } } ) = \infty$ . For any fixed $n \in \mathbb N$ , let $m = \lfloor { \sqrt { n } } \rfloor$ . Then ${ \mathfrak { s } } _ { \emptyset } ( \mathcal { H } ) \geq$ m, so by Lemma 4.11 there exists a sample set of size at most $m ^ { 2 } \leq n$ on which any randomized self-directed learner with apple-tasting feedback makes at least $\frac { 2 m ^ { 2 } } { 3 m + 1 } \in \Omega ( \sqrt { n } )$ mistakes. □

When ${ \mathfrak { s } } _ { \varnothing } ( { \mathcal { H } } ) < \infty$ , Lemma 4.11 yields only a constant lower bound. We supplement it with a bound based on $\operatorname { S D d i m } ( \mathcal { H } )$ . Because apple-tasting feedback is less useful than full-information feedback, the only difficulty is in showing that $\Omega ( \mathrm { S D } \mathrm { d i m } ( { \mathcal { H } } ) )$ also lower-bounds the performance of randomized algorithms. A full proof would involve retreading much of the ground of Devulapalli and Hanneke [2024], so we refer the reader to that paper for a more complete description of the trees involved.

Theorem 4.13. Let H be a concept class such that ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } )$ and SDdim(H) are both finite. Then for all sufficiently large n we have $\begin{array} { r } { M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \geq \frac { 1 } { 2 } } \end{array}$ max $\{ \mathfrak { s } _ { \emptyset } ( \mathcal { H } )$ , SDdim(H)}. Moreover, $i f \mathrm { S D } \mathrm { d i m } ( \mathcal { H } ) =$ $\infty ,$ , then $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \omega ( 1 )$

Informal proof Apple-tasting feedback is strictly weaker than full-information: a full-information learner can simulate any apple-tasting learner by discarding the labels it would not have observed. Any lower bound for the full-information setting therefore transfers to the apple-tasting setting, so we consider the performance of a full-information learner. Devulapalli and Hanneke [2024] characterize selfdirected learning through the dimension SDdim(H), which they show equals the optimal deterministic mistake bound $M _ { s d } ( \mathcal { H } )$ . Combinatorially, SDdim(H) can be constructed as the largest k for which there is a set $S$ and a realizable $k { - } t r e e$ on $( { \mathcal { H } } , S )$ : a tree whose nodes are subsets $V \subseteq S$ , in which every $x \in V$ carries two outgoing edges labeled with distinct labels $y _ { x } \neq y _ { x } ^ { \prime }$ , every root-to-leaf path has length at least $k ,$ and every branch is realized by some concept of $\mathcal { H } .$

In an analogous way to how [Filmus et al., 2025] show bounds on randomized adversarial online learning using the Littlestone tree, we use a maximum k-tree to bound randomized self-directed learning. Here the adversary observes the learner’s choice of point and distribution over predicted labels, but only sees the realized prediction after committing to the true label.

If SDdim $( \mathcal { H } ) < \infty$ , fix a realizable k-tree for $k = { \mathrm { S D d i m } } ( \mathcal { H } )$ and let the adversary play down it. Whenever the learner selects a point x of the current node $V$ , two labels $y _ { x } \neq y _ { x } ^ { \prime }$ are available, and at least one of them receives probability at most $1 / 2$ under the learner’s distribution; the adversary answers with that label, forcing a mistake with probability at least $1 / 2$ , and the play descends one level. Since the learner must eventually predict on every point of $S ,$ and the points outside the current node are already determined by the tree, at least k such descents occur. Hence any randomized learner suffers at least $k / 2$ mistakes in expectation, giving $\begin{array} { r } { M _ { \mathrm { o p t i m a l } } \geq \frac { 1 } { 2 } \mathrm { S D d i m } ( \mathcal { H } ) } \end{array}$ . If SDdim $( \mathcal { H } ) = \infty$ then we can take k arbitrarily large for sufficiently large n, yielding the $\omega ( 1 )$ bound.

For the remainder of the lemma statement, Lemma 4.11 gives a lower bound of $\begin{array} { r } { \frac { 2 \mathfrak { s } _ { \emptyset } ( \mathcal { H } ) ^ { 2 } } { 3 \mathfrak { s } _ { \emptyset } ( \mathcal { H } ) + 1 } \geq \frac { 1 } { 2 } \mathfrak { s } _ { \emptyset } ( \mathcal { H } ) } \end{array}$ Combing these two results give us the desired bound.

## 4.3 Concluding the tetrachotomy result

We can now prove our previously stated theorem on dimension to characterize the rates.

ProofofTheorem 4.2. Let a concept class H be given. We note that,

• By Theorem 4.3, if $\mathrm { V C } ( \mathcal { H } ) = \infty$ then $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \Omega ( n )$

• By Theorems 4.4 and 4.12, if $\mathrm { V C } ( \mathcal { H } ) < \infty$ and ${ \mathfrak { s } } _ { \varnothing } ( { \mathcal { H } } ) = \infty$ then $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \widetilde { \Theta } _ { \mathcal { H } } ( \sqrt { n } )$

• By Theorems 4.5 and 4.13, if ${ \mathfrak { s } } _ { \emptyset } ( \mathcal { H } ) <$ ∞ and $\mathrm { S D d i m } ( \mathcal { H } ) = \infty$ then $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in O _ { \mathcal { H } } ( \log n ) \cap$ $\omega _ { \mathcal { H } } ( 1 )$

• By Theorem 4.10, if ${ \mathfrak { s } } _ { \varnothing } ( { \mathcal { H } } ) < \infty$ and SDdim(H) < ∞ then $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \Theta _ { \mathcal { H } } ( 1 )$

ProofofTheorem 4.1. Let a concept class H be given. By Definitions 3.7, 3.8, and 3.9 we have $\mathrm { V C } ( \mathscr { H } ) \leq$ ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } )$ and $\mathrm { V C } ( \mathcal { H } ) \leq \mathrm { S D d i m } ( \mathcal { H } )$ , so exactly one of the following holds: $\mathrm { V C } ( \mathcal { H } ) = \infty ; \mathrm { o r } \mathrm { V C } ( \mathcal { H } ) < \infty$ and $\mathfrak { s } _ { \varnothing } ( \mathcal { H } ) = \infty ; \mathrm { o r } \mathfrak { s } _ { \varnothing } ( \mathcal { H } ) < \infty$ and SDdim $( \mathcal { H } ) = \infty ; \thinspace \thinspace \mathrm { o r } \thinspace \mathfrak { s } _ { \emptyset } ( \mathcal { H } ) < \infty$ and $\mathrm { S D d i m } ( { \mathcal { H } } ) < \infty$ . Theorem $4 . 2$ assigns to each of these cases the rates $\Omega ( n ) , \widetilde { \Theta } _ { \mathcal { H } } ( \sqrt { n } ) , O _ { \mathcal { H } } ( \log n ) \cap \omega _ { \mathcal { H } } ( 1 )$ and $\Theta _ { \mathcal { H } } ( 1 )$ respectively. Hence every concept class attains one of the four rates, and no class attains two, since the cases are mutually exclusive. □

Remark 1. Self-directed online learning with apple-tasting feedback can be shown to allow learnability of more concept classes than its adversarial counterpart. For instance, the combinatorial characterization given by Raman et al. [2024] shows that $\Omega ( n )$ mistakes can be forced for thresholds in the adversarial online setting, but the VC−dimension and ∅-centered star number ${ \mathfrak { s } } _ { \emptyset }$ arefinitefor this class.

Remark 2. We note that SDdim(H) and ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } )$ could be finite or infinite, independent of each other. Singletons on $\mathbb { R }$ have infinite ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } )$ butfinite SDdim(H). In section $7$ we show that there also exists a class withfinite ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } )$ but infinite SDdim(H).

## 5 Agnostic Learning

In the agnostic case, we obtain the same bounds that Devulapalli and Hanneke [2024] did for fullfeedback self-directed learning. Our lower bound is implied by theirs. Our upper bound uses the same strategy, but with the implementation of online learning with expert advice from Raman et al. [2024] that works with apple-tasting feedback. We state both of these prior results here.

Proposition 2 (Theorem 17 from Devulapalli and Hanneke [2024]). Let H be a concept class with finite VC dimension. Then for all $n \in \mathbb N ,$ , there exists an instance space of size n on which any self-directed learning algorithm, with full-information feedback, has expected regret $\Omega ( { \sqrt { n \mathrm { V C } ( \mathcal { H } ) } } )$ .

Proposition 3 (Theorem 16 from Raman et al. [2024]). Let $N , T \in \mathbb { N }$ and let $\mathcal { E } ^ { 1 } , \dots , \mathcal { E } ^ { N }$ be experts. For expert $\mathcal { E } ^ { i } { } _ { : }$ , let $\mathcal { E } _ { 1 } ^ { i } , \ldots , \mathcal { E } _ { T } ^ { i }$ be their predictions in rounds $1 , \ldots , T$ . Let $y _ { 1 } , \ldots , y _ { T }$ be the sequence of true labels. Then there exists an online learning algorithm that receives only apple-tastingfeedback and outputs predictions $\widehat { y } _ { 1 } , \ldots , \widehat { y } _ { T }$ that satisfy:

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ { \widehat { y } } _ { t } \neq y _ { t } \right\} \right] \leq \left( \operatorname* { i n f } _ { j \in [ N ] } \sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ { \mathcal E } _ { t } ^ { j } \neq y _ { t } \right\} \right) + { \sqrt { 3 T \ln N } }
$$

Theorem 5.1. Let H be a concept class with finite VC dimension. The optimal expected regret of learning H in the self-directed, apple-tasting, agnostic setting satisfies:

$$
R _ { \mathrm { o p t i m a l } } ( { \mathcal { H } } , n ) \in \Omega ( { \sqrt { n \operatorname { V C } ( { \mathcal { H } } ) } } ) \qquad { \mathrm { ~ } } a n d \qquad R _ { \mathrm { o p t i m a l } } ( { \mathcal { H } } , n ) \in O ( { \sqrt { n \operatorname { V C } ( { \mathcal { H } } ) \log n } } )
$$

Proof. The lower bound is implied by Proposition 2, since apple-tasting feedback is weaker than fullinformation feedback.

For the upper bound, we use the algorithm of Proposition 3. After receiving the initial set of points $S ,$ the learner instantiates one expert for each projection of the concept class onto $S .$ . The performance of the best expert is then equal to the performance of the best concept, and by Sauer’s lemma we have log $N \in O ( \mathrm { V C } ( \mathcal { H } ) \log n )$ . The learner chooses an arbitrary order of the points in $S$ and commits to predicting on them in that order. The regret is therefore $O ( { \sqrt { n \log N } } ) = O ( { \sqrt { n { \mathrm { V C } } ( { \mathcal { H } } ) \log n } } )$ □

## 6 Oblivious adversaries

Throughout our paper, we have been considering the most powerful adversary possible in the selfdirected setting. Not only do they choose which points the learner can see, they can adaptively choose true labels in response to the learner’s queries. In this section, we consider a weaker variant of the realizable setting in which the adversary decides the true labels at the start of the game. This setting is colloquially called the oblivious adversary setting.

For non-oblivious adversaries in the regime where $\mathrm { V C } ( \mathcal { H } ) < \infty$ , Theorem 4.4 gives the deterministic algorithm PRUNE-AND-EAT which achieves $O ( \sqrt { \mathrm { V C } ( \mathscr { H } ) n \log n } )$ mistakes. In this section we show that the $\sqrt { \log n }$ factor can be removed for oblivious adversaries and that the resulting bound is tight.

Since the true concept $h ^ { * }$ is now chosen at the start of the game, the most natural predictor is the one-inclusion graph of Haussler et al. [1994]. We take a sample of size $\sqrt { \mathrm { V C } ( \mathscr { H } ) n }$ from the provided set of points S and treat it as training data to construct the one-inclusion graph. Using this predictor, we generalize to the other points in S.

Algorithm 5 SAMPLE-AND-EAT(H, S)   
1: Draw $R \subseteq S$ uniformly at random among all subsets of $S$ of size $\left\lceil \sqrt { \mathrm { V C } ( \mathcal { H } ) n } \right\rceil$   
2: $S  S \setminus R$   
3: Query the points in R in any order and predict $\widehat { y } = 1$ for all these points   
4: Obtain $h ^ { * } | _ { R }$ ▷ Projection of the true concept onto R   
5: Construct the one-inclusion graph $G _ { \mathcal { H } } ( R , h ^ { * } | _ { R } )$   
6: For each $x \in S { \mathrm { : } }$   
7: Predict $\widehat { y } = G _ { \mathcal { H } } ( R , h ^ { * } | _ { R } ; x )$ ▷ the one-inclusion graph prediction   
8: Ignore any feedback   
9: $S \gets S \setminus \{ x \}$

Theorem 6.1. Let H be a concept class with $\mathrm { V C } ( \mathcal { H } ) < \infty$ and assume the adversary is oblivious. Then H is learnable by algorithm SAMPLE-AND-EAT with at most $2 \sqrt { \mathrm { V C } ( \mathscr { H } ) n } + 1$ mistakes in expectation.

Proof. The algorithm makes mistakes in two phases, which we bound separately:

$M _ { 1 } \colon$ Mistakes made while predicting 1 for all points in R. We can naively bound $M _ { 1 } \leq | R | \leq$ $\sqrt { \mathrm { V C } ( \mathcal { H } ) n } + 1$

$M _ { 2 } { \mathrm { : } }$ Mistakes made by the one-inclusion graph predictor. We know the one-inclusion graph makes a mistake with probability at most $\frac { \mathrm { { V } \bar { C } ( \hat { \mathcal { H } } ) } } { | R | }$ on an unseen point. Therefore, $\mathbb { E } [ M _ { 2 } ] \le | S \ \rangle$ $\begin{array} { r } { R | \cdot \frac { \mathrm { V C } ( \mathcal { H } ) } { | R | } \leq \sqrt { \mathrm { V C } ( \mathcal { H } ) n } } \end{array}$

Combining the two bounds, we get $\mathbb { E } [ M _ { 1 } + M _ { 2 } ] \leq 2 \sqrt { \mathrm { V C } ( \mathcal { H } ) n } + 1$ as desired.

The upper bound of Theorem 6.1 still has a $\sqrt { \mathrm { V C } ( \mathscr { H } ) }$ factor. We now show that in general, this factor cannot be removed. We use several independent copies of the structure from Lemma 4.11.

Theorem 6.2. For all $d \in \mathbb { N } ,$ , there exists a concept class $\mathcal { H } _ { d }$ such that $\mathrm { V C } ( \mathcal { H } _ { d } ) = d$ and $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } _ { d } , n ) \in \Omega ( \sqrt { n d } )$ , even against an oblivious adversary.

Proof. Let $\mathcal { H } _ { 1 }$ be the infinite singleton concept class. That ${ \mathrm { i s } } ,$ let $\mathcal { X } = \mathbb { N }$ be its instance space and include one concept $h _ { k }$ for each $k \in \mathcal { X }$ , where $h _ { k } ( k ) = 1$ and $h _ { k } ( j ) = 0$ for all $j \neq k$ . We then let $\mathcal { H } _ { d }$ consists of d independent copies of $\mathcal { H } _ { 1 }$ . Since $\mathrm { V C } ( \mathcal { H } _ { 1 } ) = 1$ , we have $\mathrm { V C } ( \mathcal { H } _ { d } ) = d$ as desired.

We now show that $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } _ { d } , n ) \in \Omega ( \sqrt { n d } )$ . Since ${ \mathfrak { s } } _ { \varnothing } ( { \mathcal { H } } _ { 1 } ) = \infty$ , by Lemma 4.11 for any k we can construct instance spaces of size $k ^ { 2 }$ on which an oblivious adversary can force $\Omega ( k )$ mistakes in expectation. Let $n \in \mathbb { N }$ be arbitrary. We partition n into d blocks of size $n / d$ each, then fill each block with the construction of Lemma 4.11 using the d copies of $\mathcal { H } _ { 1 }$ in $\mathcal { H } _ { d }$ . By playing independently on each block, the adversary can therefore force $\Omega ( \sqrt { n / d } )$ mistakes on all d blocks, for a total of $\Omega ( { \sqrt { n d } } )$ □

## 7 A class in the gap

Three of the four regimes that Theorem 4.2 classifies include common concept classes:

• The class containing all concepts has infinite VC dimension;

• The infinite singletons concept class has finite VC dimension but infinite ${ \mathfrak { s } } _ { \emptyset } \colon$ and

• Finite concept classes have finite ${ \mathfrak { s } } _ { \emptyset }$ and SDdim.

The remaining regime has finite ${ \mathfrak { s } } _ { \emptyset }$ but infinite SDdim. In this section, we show that it too is nonempty. We use a combination of Lemma 4.7 and a construction of Kuhlmann [1999], whose relevant details are summarized by the following proposition.

Proposition 4 (Sub-claims of Theorem 2 from Kuhlmann [1999]). Let H be an intersection-closed concept class on an instance space X. Suppose $2 \leq \mathrm { V C } ( \mathcal { H } ) < \infty$ . Then there exists an intersectionclosed concept class $\mathcal { H ^ { \prime } }$ on an instance space $\mathcal { X } ^ { \prime }$ such that:

$\operatorname { V C } ( \mathcal { H } ^ { \prime } ) = \operatorname { V C } ( \mathcal { H } )$ , and

• For all $x \in \mathcal { X } ^ { \prime }$ and $y \in \{ 0 , 1 \}$ }, the version space $\{ h \in \mathcal { H } ^ { \prime } \mid h ( x ) = y \}$ contains a copy $o f \mathcal { H }$

Furthermore, ifX isfinite, then one can take $| \mathcal { X } ^ { \prime } | = ( | \mathcal { X } | + 1 ) \cdot | \mathcal { X } |$

Theorem 7.1. There exists a concept class H for which ${ \mathfrak { s } } _ { \varnothing } ( { \mathcal { H } } ) = 2 , { \mathrm { S D d i m } } ( { \mathcal { H } } ) = \infty a n d$ $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \Omega ( \log \log n )$

Proof. We use the construction of Proposition 4 recursively. It produces intersection-closed concept classes, but by reversing the labels we can produce union-closed concept classes instead. Then Lemma 4.7 allows us to relate their VC-dimension to their ∅-centered star number.

Specifically, let $\mathcal { X } _ { 0 }$ be a set of size 2 and let $\mathcal { H } _ { 0 }$ be the set of all concepts on $\mathcal { X } _ { 0 }$ . We then have $\mathrm { V C } ( \mathcal { H } _ { 0 } ) = 2$ and that it is union-closed. Recursively apply the construction of Proposition 4 but with inverted labels to produce instance spaces $\mathcal { X } _ { 1 } , \mathcal { X } _ { 2 } , . . .$ . and union-closed concept classes $\mathcal { H } _ { 1 } , \mathcal { H } _ { 2 } , . . .$ . that still satisfy $\mathrm { V C } ( \mathcal { H } _ { i } ) = 2$ for all i. Each $\mathcal { H } _ { i }$ contains copies of $\mathcal { H } _ { i - 1 }$ , so by identifying $\mathcal { X } _ { i - 1 }$ with one of these copies we can turn this infinite sequence of concept classes into a single infinite concept class. We take this limiting concept class to be H and the limiting instance space to be $\mathcal { X }$

We first show that ${ \mathfrak { s } } _ { \emptyset } ( \mathcal { H } ) = 2$ . Because H is union-closed, by Lemma 4.7 we have ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } ) = { \mathrm { V C } } ( { \mathcal { H } } )$ Since H contains a copy of $\mathcal { H } _ { 0 }$ we have $\mathrm { V C } ( \mathscr { H } ) \geq 2$ , and since any shattered set of size 3 would be contained in some finite $\mathcal { X } _ { i }$ we have $\mathrm { V C } ( \mathscr { H } ) \leq 2$ . Therefore, $\mathfrak { s } _ { \varnothing } ( \mathcal { H } ) = \mathrm { V C } ( \mathcal { H } ) = 2$

We now show that $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n ) \in \Omega ( \log \log n )$ . As a stepping stone, we show by induction that $M ( \mathcal { H } _ { i } , \mathcal { X } _ { i } ) \geq i / 2$ , recalling that $M ( \mathcal { H } _ { i } , \mathcal { X } _ { i } )$ denotes the expected number of mistakes an adversary can force when presenting the instance space $\mathcal { X } _ { i }$ to the learner. The base case $M ( \mathcal { H } _ { 0 } , \mathcal { X } _ { 0 } ) \ge 0$ is trivial. For the inductive step, suppose $M ( \mathcal { H } _ { i } , \mathcal { X } _ { i } ) \geq i / 2$ for some $i \geq 0$ . When the learner is presented with $\mathcal { X } _ { i + 1 }$ the adversary can choose the true label for the learner’s first point uniformly at random. This forces half a mistake on average. By construction, there then exists a copy of $\mathcal { H } _ { i }$ that agrees with this random label, so by the inductive hypothesis the adversary can force $i / 2$ more mistakes on average. They can therefore force $( i + 1 ) / 2$ in total, completing the inductive step.

For $M _ { \mathrm { o p t i m a l } } ( \mathcal { H } , n )$ , the adversary chooses the largest $\mathcal { X } _ { i }$ such that $| { \mathcal { X } } _ { i } | \leq n$ and uses their optimal strategy for $\mathcal { H } _ { i }$ . The learner then incurs $i / 2$ mistakes on average. Since $| \mathcal { X } _ { j + 1 } | = ( | \mathcal { X } _ { j } | + 1 ) \cdot | \mathcal { X } _ { j } |$ for all $j ,$ , in terms of n we have $i / 2 \in \Omega ( \log \log n )$ . Since the adversary’s strategy works even for full-information feedback, we have SDdim $( { \mathcal { H } } ) = \infty$ as well. □

## 8 AI Disclosure

We used GPT 5.6 Pro, GPT 6 Astra, and Opus 5.5 to produce some of our arguments. Our initial hypothesis was that there was a trichotomy defined by ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } )$ and $\mathrm { V C } ( \mathscr { H } )$ , and we spent a long time without AI assistance trying to show a constant upper bound for classes with finite ${ \mathfrak { s } } _ { \emptyset } ( { \mathcal { H } } )$ . This, of course, turned out to be incorrect, and prompting GPT 5.6 Pro directed us to the construction of Kuhlmann [1999] as the basis for a counterexample. Prompting with GPT 5.6 Pro and GPT 6 Astra provided the upper bounds of Theorem 4.5. A more general prompting with Opus 5.5 produced the oblivious-adversary upper bound of Theorem 6.1, which was not a setting we were considering. All writing in this paper is our own.

## 9 Conclusion and Future Work

We introduced self-directed online classification under apple-tasting feedback and determined its minimax rates in both the realizable and the agnostic settings. In the realizable setting, we obtain a tetrachotomy where the expected number of minimax mistakes can be $\Theta ( 1 ) , \omega ( 1 )$ and ${ \bar { O } } ( \log n ) , { \widetilde { \Theta } } ( { \sqrt { n } } )$ , or $\Theta ( n )$ . We suspect, however, that the log factors in the $\omega ( 1 ) \cap O ( \log n )$ and $\bar { \Theta } ( { \sqrt { n } } )$ regimes might hide more fundamental characterizations.

In the $\widetilde { \Theta } ( \sqrt { n } )$ regime, if we assume the adversary is oblivious then Theorem 6.1 attains a mistake bound of $O ( \sqrt { \mathrm { V C } ( \mathscr { H } ) n } )$ . We also obtain a matching lower bound attained by a specific class. Is the extra $\sqrt { \log n }$ factor in Theorem 4.4 an artifact of the adaptive adversary, the deterministic algorithm, or both? There are also classes with large VC dimension that do not have proportionally larger mistake bounds. For instance, a concept class consisting of the infinite singletons concept class plus a shattered set of size $d - 1$ has VC dimension d but behaves almost identically to the infinite singletons concept class. Is there a combinatorial dimension that captures this nuance?

## References

Noga Alon, Nicolò Cesa-Bianchi, Ofer Dekel, and Tomer Koren. Online learning with feedback graphs: Beyond bandits. In Proceedings of The 28th Conference on Learning Theory, volume 40, pages 23–35. PMLR, 2015. URL https://proceedings.mlr.press/v40/Alon15.html.

Ryan Alweiss, Brice Huang, and Mark Sellke. Improved lower bound for frankl’s union-closed sets conjecture, 2024. URL https://arxiv.org/abs/2211.11731.

Gábor Bartók, Dean P. Foster, Dávid Pál, Alexander Rakhlin, and Csaba Szepesvári. Partial monitoring—classification, regret bounds, and algorithms. Mathematics of Operations Research, 39 (4):967–997, 2014. doi: 10.1287/moor.2014.0663.

Shai Ben-David, Nadav Eiron, and Eyal Kushilevitz. On self-directed learning. In Proceedings of the Eighth Annual Conference on Computational Learning Theory, COLT ’95, page 136–143, New York, NY, USA, 1995. Association for Computing Machinery. ISBN 0897917235. doi: 10.1145/225298. 225314. URL https://doi.org/10.1145/225298.225314.

Olivier Bousquet, Steve Hanneke, Shay Moran, and Nikita Zhivotovskiy. Proper learning, helly number, and an optimal svm bound. In Conference on Learning Theory, pages 582–609. PMLR, 2020.

Henning Bruhn and Oliver Schaudt. The journey of the union-closed sets conjecture, 2013. URL https://arxiv.org/abs/1309.3297.

Nicolo Cesa-Bianchi, Gábor Lugosi, et al. Prediction, learning, and games, volume 1. Cambridge university press Cambridge, 2006.

Nicolo Cesa-Bianchi, Yishay Mansour, and Gilles Stoltz. Improved second-order bounds for prediction with expert advice. Machine Learning, 66(2):321–352, 2007.

Zachary Chase and Shachar Lovett. Approximate union closed conjecture, 2022. URL https:// arxiv.org/abs/2211.11689.

Zachary Chase and Idan Mehalel. Deterministic apple tasting, 2025. URL https://arxiv.org/ abs/2410.10404.

Pramith Devulapalli and Steve Hanneke. The dimension of self-directed learning. In Proceedings of the 35th International Conference on Algorithmic Learning Theory, volume 237, pages 544–573, 2024.

Yuval Filmus, Steve Hanneke, Idan Mehalel, and Shay Moran. Optimal prediction using expert advice and randomized littlestone dimension, 2025. URL https://arxiv.org/abs/2302.13849.

Justin Gilmer. A constant lower bound for the union-closed sets conjecture, 2022. URL https: //arxiv.org/abs/2211.09055.

Sally A Goldman and Robert H Sloan. The power of self-directed learning. Machine Learning, 14(3): 271–294, 1994.

Steve Hanneke. Refined error bounds for several learning algorithms. Journal of Machine Learning Research, 17(135):1–55, 2016.

Steve Hanneke. The star number and eluder dimension: Elementary observations about the dimensions of disagreement. In Proceedings of the Thirty Seventh Conference on Learning Theory, volume 247, pages 2308–2359, 2024.

Steve Hanneke and Liu Yang. Minimax analysis of active learning. Journal of Machine Learning Research, 16(109):3487–3602, 2015. URL http://jmlr.org/papers/v16/hanneke15a. html.

D. Haussler, N. Littlestone, and M.K. Warmuth. Predicting 0, 1-functions on randomly drawn points. Information and Computation, 115(2):248–292, December 1994. ISSN 0890-5401. doi: 10.1006/ inco.1994.1097. URL http://dx.doi.org/10.1006/inco.1994.1097.

David P. Helmbold, Nicholas Littlestone, and Philip M. Long. Apple tasting. Information and Computation, 161(2):85–139, 2000. ISSN 0890-5401. doi: 10.1006/inco.2000.2870. URL http: //dx.doi.org/10.1006/inco.2000.2870.

Christian Kuhlmann. On teaching and learning intersection-closed concept classes. In Paul Fischer and Hans Ulrich Simon, editors, Computational Learning Theory (EuroCOLT 1999), pages 168–182. Springer Berlin Heidelberg, 1999.

Jingbo Liu. Improving the lower bound for the union-closed sets conjecture via conditionally iid coupling, 2023. URL https://arxiv.org/abs/2306.08824.

Luke Pebody. Extension of a method of gilmer, 2022. URL https://arxiv.org/abs/2211. 13139.

Vinod Raman, Unique Subedi, Ananth Raman, and Ambuj Tewari. Apple tasting: Combinatorial dimensions and minimax rates. In Proceedings of the Thirty Seventh Conference on Learning Theory, volume 247, pages 4358–4380, 2024.

Will Sawin. An improved lower bound for the union-closed set conjecture, 2023. URL https:// arxiv.org/abs/2211.11504.

Alexander B Tsybakov. Optimal aggregation of classifiers in statistical learning. The Annals of Statistics, 32(1):135–166, 2004.

V. N. Vapnik and A. Ya. Chervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. Theory of Probability & Its Applications, 16(2):264–280, January 1971. ISSN 1095-7219. doi: 10.1137/1116025. URL http://dx.doi.org/10.1137/1116025.

Vladimir G Vovk. A game of prediction with expert advice. In Proceedings of the eighth annual conference on Computational learning theory, pages 51–60, 1995.

Volodimir G Vovk. Aggregating strategies. In Proceedings of the third annual workshop on Computational learning theory, pages 371–386, 1990.

Yair Wiener, Steve Hanneke, and Ran El-Yaniv. A compression technique for analyzing disagreementbased active learning. The Journal of Machine Learning Research, 16(1):713–745, 2015.