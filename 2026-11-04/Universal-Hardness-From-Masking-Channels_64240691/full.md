# Universal Hardness From Masking Channels

Parikshit Gopalan Apple

Manoj Prabhakaran<sup>∗</sup> Pranay Tankala<sup>†</sup> IIT Bombay Harvard

November 2, 2026

## Abstract

We prove a broad learning-theoretic generalization of the celebrated hardcore lemma concerning the average-case complexity of computing a Boolean function (Impagliazzo, 1995). Our starting point is the observation that the classic notion of a hardcore measure, which reweights a function’s input distribution, is equivalent to an input-dependent masking channel that injects randomness into the function’s output distribution until it appears fully pseudorandom. The analogue of density of the measure is the fidelity of the channel; a high-fidelity channel injects very little noise. From this viewpoint, we develop a general framework for certifying hardness that allows for non-Boolean functions f and a wide range of notions of hardness besides the classic 0-1 loss, by choosing the right notion of fidelity via convex duality. We show the dificulty of learning f using a class C to minimize a convex loss is tightly characterized by the maximum fidelity possible for a masking channel, which measures the least amount of noise the channel must add to hide the structure of f and make it C-indistinguishable from random noise.

We derive the following concrete applications from this general framework:

• We instantiate our framework to derive a number of multiclass generalizations of the hardcore lemma, and reveal some unexpected subtleties in going to larger alphabets. We tightly characterize the hardness of learning with various losses in the binary setting.

• We prove the existence of a universal masking channel that applies to many natural notions of hardness and fidelity simultaneously. Our results show that such channels are dual to omnipredictors [GKR<sup>+</sup>22], which optimize multiple losses at once.

• We observe a symmetry between masking channels and generalizations of the hardcore lemma using pseudoentropy. We present a tight characterization of pseudoentropy using the MaxEnt algorithm from machine learning.

A long line of research has shown that learning and hardness are two sides of the same coin; our framework gives a crisp and general formulation of this principle via convex duality.

## Contents

1 Introduction   
1.1 Hardness Equals Best Masking Channel Fidelity 3   
1.2 Multiclass Hardcore Lemmas 6   
1.3 Universal Masking Channels From Multicalibration . 9   
1.4 Characterizing Pseudoentropy via MaxEnt . 12   
1.5 Related Work . 14   
1.6 AI Methodology 15   
2 Preliminaries 15   
2.1 Convex Analysis 16   
2.2 Hinge Loss Rounding . 18   
2.3 Multicalibration . 18   
3 A General Framework 19   
4 Hardness of Learning and Masking Channel Fidelity 22   
5 Multiclass Hardcore Lemmas 23   
5.1 Bounds for Surrogate Losses . 26   
6 Universal Masking Channels From Multicalibration 28   
6.1 Deriving the Multiclass Result 28   
6.2 Recovering the Binary Result 33   
7 Characterizing Pseudoentropy via MaxEnt 35   
A Additional Calculations 39

## 1 Introduction

Given a function class C, which might comprise machine learning models of a given architecture or Boolean circuits of a given complexity, why might a Boolean function $f : \{ \pm 1 \} ^ { n }  \{ \pm 1 \}$ be hard to approximate by functions in the class? In other words, what properties of f might explain why every $c \in { \mathcal { C } }$ errs on at least a δ-fraction of inputs to $f ?$

One possible explanation is that the function f has a hardcore set $S \subseteq \{ \pm 1 \} ^ { n }$ of density at least 2δ on which every circuit in C errs with probability close to $1 / 2$ , the same error as achieved by naive random guessing. The classic hardcore lemma in complexity theory states that this is, in fact, the only possible explanation. The lemma can be stated as follows, where we set $\mathcal { X } = \{ \pm 1 \} ^ { n }$ and $y = f ( x )$

Theorem 1.1 (Hardcore Lemma [Imp95, Hol05]). For all distributions over $\textbf { \em x } \in { \mathcal { X } } _ { i }$ , functions $f : \mathcal { X } \to \{ \pm 1 \}$ , classes ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \{ \pm 1 \} \}$ , and $\delta , \varepsilon \in ( 0 , 1 )$ , at least one of the following $h o l d s { : } ^ { 1 }$

• (Easy Case) For some $c : \mathcal { X }  \mathbb { R }$ that is the average of $O ( \log ( 1 / \delta ) / \varepsilon ^ { 2 } )$ functions in ${ \mathcal { C } } _ { i }$

$$
\operatorname* { P r } [ \mathrm { s i g n } ( c ( \pmb { x } ) - t ) = f ( \pmb { x } ) ] \geqslant 1 - \delta ,
$$

where $t \in \mathbb { R }$ is some threshold.

• (Hard Case) There exists a measure $\mu : \mathcal { X } \to [ 0 , 1 ]$ with the following properties:<sup>2</sup>

– (Density) E[µ(x)] ⩾ 2δ,

$$
- \ ( H a r d c o r e ) \ \mathbb { E } [ \mu ( { \pmb x } ) { \mathbf 1 } [ c ( { \pmb x } ) = f ( { \pmb x } ) ] ] \leqslant ( { \mathbf 1 } / 2 + \varepsilon ) \mathbb { E } [ \mu ( { \pmb x } ) ] \ f o r \ a l l \ c \in \mathcal { C } .
$$

This lemma was first proved by Impagliazzo in [Imp95] with density δ. Holenstein [Hol05] improved the density to the tight bound of 2δ.

From Measures to Noisy Channels. In this paper, we propose a new generalization of the hardcore lemma approach to certifying hardness beyond the setting described in Theorem 1.1, which will naturally extend to settings where $f$ is not necessarily Boolean-valued and there are a variety of loss functions of interest beyond 0-1 loss. To begin, let us consider a natural relaxation of the property that $f$ has a hardcore measure: that the output of f can be made computationally indistinguishable from pure random noise by passing it through a masking channel which flips the bit $f ( x )$ with probability $\eta ( x )$ , where

$$
\mathbb { E } [ \eta ( { \pmb x } ) ] \leqslant \frac { 1 } { 2 } - \delta .
$$

Note that the noise rate $\eta ( x )$ is allowed to depend on $x ,$ which allows an adversary to concentrate noise on the easy parts of the domain. This relaxed property is directly implied by the hardcore property, and it directly implies that $f$ is δ-hard against $\mathcal { C } . { } ^ { 3 }$

Given a hardcore set $S \subseteq { \mathcal { X } }$ , we can construct such a masking channel by replacing the function outside S by random bits, as shown in Figure 1. For a distinguisher from C to tell the resulting labels apart from random noise, they would need to have an advantage over the set S itself, which is not possible by the hardcore property. The larger the hardcore set, the less the noise needed in expectation. In contrast, for an easy function (say if $f \in { \mathcal { C } } )$ , the channel would need to randomize every point in the domain $x ,$ or, more generally, corrupt half the labels on average. More generally, given a 2δ-dense hardcore measure $\mu ,$ it is easy to see that one such masking channel arises from the channel given by $\eta ( x ) = ( 1 - \mu ( x ) ) / 2$ , or equivalently from a noisy label $\tilde { \pmb { y } }$ with $\mathbb { E } [ \tilde { \pmb { y } } | \pmb { x } ] = \mu ( \pmb { x } ) f ( \pmb { x } )$

![](images/4a623b535ed7497bf0c07e94ffc7952aee17e7c055a0e87cb6a76b5ffead86c4.jpg)  
Figure 1: Masking f by randomizing outside the hardcore set.

Further, we claim that any such masking channel certifies the hardness of $f$ . The quantitative bound is better when the amount of noise added is small. To see this, recall that two functions $a , b : \mathcal { X } \to \{ \pm 1 \}$ are said to be $( \mathcal { C } , \varepsilon )$ -indistinguishable [TTV09] if $\begin{array} { r } { \operatorname* { m a x } _ { c \in \mathcal { C } } | \mathbb { E } [ c ( \pmb { x } ) ( a ( \pmb { x } ) - b ( \pmb { x } ) ) | \leqslant \varepsilon . } \end{array}$ and that this definition extends easily to pairs of non-deterministic conditional distributions ${ \pmb a } | { \pmb x }$ and ${ \pmb b } | { \pmb x } . ^ { 4 }$ If we let ${ \pmb y } , \tilde { { \pmb y } } , { \pmb b } \in \{ \pm 1 \}$ denote $f ( { \pmb x } )$ , the noisy label, and an independent uniform bit, respectively, then for all $c \in { \mathcal { C } }$

$$
\operatorname* { P r } [ c ( x ) \neq y ] \geqslant \operatorname* { P r } [ c ( x ) \neq \tilde { y } ] - \mathbb { E } [ \eta ( { \pmb x } ) ] \approx _ { \varepsilon } \underbrace { \operatorname* { P r } [ c ( { \pmb x } ) \neq b ] } _ { = 1 / 2 } - \underbrace { \mathbb { E } [ \eta ( { \pmb x } ) ] } _ { \leqslant 1 / 2 - \delta } \geqslant \delta .
$$

Our framework reinterprets the role of the measure $\mu .$ While $\mu$ was originally conceived as a reweighting of the distribution over features $x \in \mathcal { X }$ , it recasts $\mu$ as a noisy channel that leaves the marginal distribution over features fixed and instead modifies the conditional distribution over labels. This change in perspective was crucially used in distribution-specific agnostic boosting algorithms [Fel10, KK09]. We will see that it is extremely fruitful for certifying hardness.

Our Contributions. In this work, we design a general framework, based on masking channels, for certifying the hardness of a function, under various notions of what it means for functions to be easy, or have structure that is identifiable by ${ \mathcal { C } } ,$ , that substantially generalizes and extends the hardcore lemma for Boolean functions and 0-1 loss. From a machine learning point of view, this generalization gives tight, quantitative characterizations of the best loss one might hope to achieve when fitting a distribution over labeled data points $( { \pmb x } , { \pmb y } )$ using models from a prescribed class. From a computational complexity point of view, we give characterizations of the averagecase complexity of a function, with respect to a wide range of notions of hardness for functions $f : \mathcal { X }  \mathcal { V }$ , where $| { \mathcal { V } } | = k$ is finite. In particular:

(1) (Section 1.1) Our main conceptual contribution is a shift in focus from hardcore measures $\mu : \mathcal { X } \to [ 0 , 1 ]$ to hardness certificates based on input-dependent masking channels

$$
W : \mathcal { X }  \mathbb { R } ^ { k \times k } .
$$

Here, $W ( x )$ is a column-stochastic matrix mapping each possible label $y \in \mathcal { V }$ to a distribution $W ( x ) y \in \Delta _ { k }$ over noisy labels $\tilde { y } \in \mathcal { V }$ , all represented as k-dimensional column vectors.

Suppose the noisy label distribution $W ( \pmb { x } ) \pmb { y } \in \Delta _ { k }$ is C-indistinguishable from the uniform distribution $\vec { 1 } / k$ when labeled data points $( { \pmb x } , { \pmb y } )$ are drawn from a given distribution D. We call such a channel a masking channel for C under the distribution $\mathcal { D }$ and say that the output of the masking channel is C-pseudorandom.

(2) (Section 1.1 and Section 4) We show that the average-case hardness of approximating the labels using C is equal to the highest $\it { f i d e l i t y }$ masking channel $W$ with C-pseudorandom outputs, provided that the chosen notions of “hardness” and “fidelity” are convex-concave conjugates. In other words, hardness is related to the least amount of noise that must be added to the function’s output to mask the signal and make it $\mathcal { C } \mathrm { - p s e u d o r a n d o m }$ . Our theorem, which crucially relates pseudorandomness constraints to $\ell _ { 1 }$ regularization via convex duality, can be viewed as sharpening a complexity-theoretic result via a learning-theoretic perspective.

(3) (Section 1.2 and Section 5) We show that our hardness-fidelity theorem implies the existence of several distinct and interesting hardness certificates in the multiclass setting. In particular, by combining our framework with new insights on surrogate losses for the 0-1 loss, we uncover a landscape of novel “hardcore lemmas” in the multiclass setting.

(4) (Section 1.3 and Section 6) Using multicalibration and related ideas, we construct a universal masking channel, which serves as a nearly tight certificate of the hardness of a function $f$ with respect to a wide range of diferent notions of hardness simultaneously. A priori, the existence of such a universal channel is very surprising.

(5) (Section 1.4 and Section 7) Finally, we observe an elegant symmetry between our results and existing results on generalized pseudoentropy in complexity theory. We show that MaxEnt algorithm in machine learning gives a learning-theoretic exact characterization of pseudoentropy.

## 1.1 Hardness Equals Best Masking Channel Fidelity

Before setting up our framework of masking channels and channel fidelity, we give a simple example of what the framework has to ofer. To begin, let us consider the optimal $( \mathcal { C } , \varepsilon )$ -hardcore density $d ^ { \star }$ in the binary setting which can be expressed with the following optimization problem:<sup>5</sup>

$$
d ^ { \star } = \operatorname* { m a x } _ { \mu : \mathcal { X } \times \{ \pm 1 \} \to [ 0 , 1 ] } \mathbb { E } [ \mu ( \pmb { x } , \pmb { y } ) ] \quad \mathrm { s . t . } \quad \operatorname* { m a x } _ { c \in \mathcal { C } } \big | \mathbb { E } [ \mu ( \pmb { x } , \pmb { y } ) c ( \pmb { x } ) \pmb { y } ] \big | \leqslant \varepsilon .\tag{1}
$$

The optimization problem above is well-defined for any fixed distribution over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \{ \pm 1 \}$ In the special case that $\begin{array} { r } { \pmb { y } = \pmb { f } ( \pmb { x } ) } \end{array}$ for a deterministic function $f _ { i }$ , we may assume without loss of generality that $\mu ( x , y )$ ignores its second input and is a function $\mu ( x )$ of x alone.

Our framework will connect $d ^ { \star }$ to a diferent optimization problem, which at first seems entirely unrelated. This alternate problem attempts to minimize $\ell _ { 1 } { \mathrm { - r e g u l a r i z e d } }$ hinge loss attainable by the slightly enlarged class span(C) under the same distribution over $( { \pmb x } , { \pmb y } )$ pairs. Let us use the abbreviation $\begin{array} { r } { c _ { \lambda } ( x ) = \sum _ { i } \lambda _ { i } c _ { i } ( x ) } \end{array}$ for the linear combination of functions in ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } \}$ with coeficient vector λ. Define the hinge loss for a label $y \in \{ \pm 1 \}$ and action $a \in$ R by the standard formula $\ell _ { \mathrm { h i n g e } } ( y , a ) = ( 1 - a y ) _ { + }$ , where $t _ { + } = \operatorname* { m a x } ( t , 0 )$ . Then, the value we seek is:

$$
p ^ { \star } = \operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { \mathcal { C } } } \mathbb { E } [ ( 1 - c _ { \lambda } ( \pmb { x } ) \pmb { y } ) _ { + } ] + \varepsilon \| \lambda \| _ { 1 } .\tag{2}
$$

Soon, we will see that $p ^ { \star } = d ^ { \star }$ , and we claim that this equality immediately implies the optimal 2δ-density version of the classic binary hardcore lemma. Indeed, it is well known in machine learning that the hinge loss of an action $a \in \mathbb { R }$ for a label $y \in \{ \pm 1 \}$ exceeds its 0-1 loss by at least a factor of 2 after appropriate rounding ([KKMS08], also see Section 1.2 and Remark 2.13). Therefore, the inequality $p ^ { \star } \leqslant d ^ { \star }$ yields the following chain of implications:

$$
\begin{array} { r l } & { \mathrm { ~ \it ~ f ~ i s ~ \delta \delta \delta \mathrm { - h a r d ~ u n d e r ~ \it ~ \ell _ { 0 - 1 } ~ f o r ~ t h r ( s p a n ( \mathcal { C } ) ) } ~ } } \\ & { \mathrm { ~ \it ~ \ / ~ { \tau ~ } ~ } } \\ & { \mathrm { ~ \it ~ f ~ i s ~ 2 \delta \delta \mathrm { - h a r d ~ u n d e r ~ \it ~ \ell _ { \mathrm { h i n g e } } ~ f o r ~ s p a n ( \mathcal { C } ) ~ } ~ } } \\ & { \mathrm { ~ \it ~ \ / ~ { \tau ~ } ~ } } \\ & { \mathrm { ~ \it ~ f ~ h a s ~ a ~ 2 \delta \delta \mathrm { - d e n s e ~ h a r d c o r e ~ w i t h ~ r e s p e c t ~ t o ~ \mathcal { C } . ~ } ~ } } \end{array}
$$

Remarkably, we see that the elusive factor of 2, which was absent in the original version of the theorem [Imp95] and only appeared much later [Hol05], arises precisely because of the factor of $2$ relating 0-1 loss to hinge loss.<sup>6</sup> The factor-2 bound on the gap between the hinge loss and 0-1 loss was shown in the work of [KKMS08] on agnostic learning, there it was used to give an agnostic learning algorithm with optimal error, rather than twice the optimal error.

In this paper, as we explore multiclass analogues of the hardcore lemma, this surrogate loss point of view will turn out to be quite fruitful.

Masking Channels. In the multiclass case, the label $y$ is an element of a size-k set Y, which we view interchangeably as either $[ k ] = \{ 1 , \dots , k \} \subseteq \mathbb { N }$ or the standard basis vectors $e _ { 1 } , \ldots , e _ { k } \in \mathbb { R } ^ { k }$ (a.k.a. “one-hot encodings”). We are given a distribution D over labeled data points $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ Our goal is to either find a structured explanation for how the labels $\textbf {  { y } }$ are generated from ${ \mathbf { } } x ,$ or certify that ${ \pmb y } | { \pmb x }$ is pseudorandom. In some cases, we view $y = f ( x )$ as a deterministic function of the input $x ,$ and in other cases, we consider an arbitrary joint distribution over input-output pairs, which we shall often refer to as feature-label pairs.

Instead of measures $\mu : \mathcal { X } \to [ 0 , 1 ]$ , we now study input-dependent noisy channels W, which map a given label y to an x-dependent distribution $W ( x ) y \in \Delta _ { k }$ over noisy labels.

Definition 1.2 (Noisy Channel [CT06]). We let $\Delta _ { k } \subseteq \mathbb { R } ^ { k }$ denote the probability simplex of vectors with nonnegative coordinates summing to 1. We similarly let $\Delta _ { k } ^ { k } \subseteq \mathbb { R } ^ { k \times k }$ denote the set of k-by-k column-stochastic matrices with columns in $\Delta _ { k }$ . Given finite sets $\mathcal { X }$ and $\mathcal { V } = [ k ]$ , a noisy channel is simply a function $W : \mathcal { X } \to \Delta _ { k } ^ { k }$

Given a deterministic function $f : \mathcal { X }  \mathcal { V }$ , a noisy channel W can be used to define a corrupted version of $f ,$ where $y = f ( x )$ (one-hot encoded) is replaced with $W ( x ) y$ (a probability distribution). To study the hardness of a function f w.r.t. a function class ${ \mathcal { C } } ,$ we now introduce the notion of a masking channel for $f ,$ which corrupts $f$ into a randomized function that appears totally random to $\mathcal { C } .$ . Below we define a masking channel not for a deterministic function $f ,$ but for a distribution D over $\mathcal { X } \times \mathcal { V }$ (allowing for randomized functions, and incorporating a distribution on the inputs).

Definition 1.3 (Masking Channel). Given a distribution $\mathcal { D }$ over labeled data points $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ a finite collection of functions ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , a noisy channel $W : \mathcal { X } \to \Delta _ { k } ^ { k }$ and $\varepsilon \geqslant 0$ , we say that $W$ is a $( \mathcal { C } , \varepsilon )$ -masking channel for the distribution $\mathcal { D } .$ , if for all $c \in { \mathcal { C } }$

$$
\begin{array} { r } { \left| \mathbb { E } [ \langle c ( \pmb { x } ) , W ( \pmb { x } ) \pmb { y } - \vec { 1 } / k \rangle ] \right| \leqslant \varepsilon . } \end{array}
$$

We say that a random variable z that is distributed jointly with x is $( \mathcal { C } , \varepsilon )$ -pseudorandom if for every $c \in { \mathcal { C } }$ , we have $\big | \mathbb { E } [ \langle c ( { \pmb x } ) , z - \vec { 1 } / k \rangle ] \leqslant \varepsilon$ . Thus the definition of a masking channel for D requires the induced label distribution $W ( { \pmb x } ) { \pmb y }$ is $( \mathcal { C } , \varepsilon )$ -pseudorandom. If the distribution $\mathcal { D }$ is clear from the context, we will suppress D and refer to W as a $( \mathcal { C } , \varepsilon )$ -masking channel.

Example 1.4. For a simple example of such a masking channel, suppose that the label y is independent of the features x and has the following distribution:

$$
\begin{array} { r } { \mathbb { E } [ { \pmb { y } } \mid { \pmb { x } } ] = \left( \begin{array} { c } { 1 / 2 } \\ { 1 / 2 } \\ { 0 } \\ { \vdots } \\ { 0 } \end{array} \right) \in \Delta _ { k } . } \end{array}
$$

Then, one possibility for the channel W is to redistribute some of the mass from the labels 1 and 2 to each of the remaining $k - 2$ labels, until the marginal distribution is uniform:

$$
W ( x ) = \left( \begin{array} { l l l l l l } { 2 / k } & { 0 } & { 0 } & { 0 } & { \cdots } & { 0 } \\ { 0 } & { 2 / k } & { 0 } & { 0 } & { \cdots } & { 0 } \\ { 1 / k } & { 1 / k } & { 1 } & { 0 } & { \cdots } & { 0 } \\ { 1 / k } & { 1 / k } & { 0 } & { 1 } & { \cdots } & { 0 } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { 1 / k } & { 1 / k } & { 0 } & { 0 } & { \cdots } & { 1 } \end{array} \right) \in \Delta _ { k } ^ { k } .
$$

Since $\mathbb { E } [ W ( { \pmb x } ) { \pmb y } \mid { \pmb x } ] = \vec { 1 } / k$ , the pseudorandomness condition is clearly satisfied for any C and $\varepsilon \geqslant 0$ $\mathrm { A s }$ this example illustrates, if a label $i \in [ k ]$ occurs with probability 0, then the corresponding column of W can be defined arbitrarily; in particular, if $y = f ( x )$ for a deterministic function $f : \mathcal { X }  [ k ]$ , then there is only a single relevant column of $W ( x )$ for each $x \in \mathcal { X }$

Channel Fidelity. In the binary setting, we sought the hardcore measure µ with the highest possible density. By analogy, in the multiclass setting, we shall seek the highest possible fidelity among masking channels. Intuitively, a channel has high fidelity if it adds little noise to the original label. In various scenarios, there are many reasonable ways to quantify a channel’s fidelity, and we would like a definition that can capture them all. In the following definition, we use the abbreviation $\overline { { \mathbb { R } } } = \mathbb { R } \cup \{ \pm \infty \}$ for the set of extended real numbers.

Definition 1.5 (Fidelity Function). Given a finite label set $\ y = [ k ]$ , a fidelity function is any function $h : \mathcal { V } \times \Delta _ { k } \to$ R that is closed, proper, and concave in its second input (Definition 2.2).

Example 1.6. The simplest example of a fidelity function, which we will use repeatedly, is

$$
h ( y , z ) = \langle y , z \rangle = 1 - d _ { \mathrm { T V } } ( y , z ) .
$$

This function takes as input the original label $y = e _ { i }$ and a noisy label distribution $z \in \Delta _ { k }$ , and outputs the scalar $z _ { i } ,$ which is the probability that the label is not corrupted. Note that this fidelity function decreases as the total variation distance $d _ { \mathrm { T V } }$ between z and $y$ (a point mass) increases. Eventually, we will plug in $z = W ( x ) y$ , where $W ( x )$ is a column-stochastic channel matrix.

Example 1.7. Another example of a fidelity function is the negated cross-entropy or log-loss

$$
h ( e _ { i } , z ) = \log z _ { i } .
$$

Note that this fidelity function outputs $- \infty$ when $z _ { i } = 0$ , which is a valid output in ${ \overline { { \mathbb { R } } } } .$

Example 1.8. For yet another example of a fidelity function, consider

$$
h ( y , z ) = { \left\{ \begin{array} { l l } { \mu } & { { \mathrm { i f ~ } } z = \mu \cdot y + ( 1 - \mu ) \cdot ( { \vec { 1 } } / k ) { \mathrm { ~ f o r ~ s o m e ~ } } \mu \in [ 0 , 1 ] } \\ { - \infty } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

This fidelity function equals the one in the first example when the noisy label distribution z has a specific form: it mixes the original label y with the uniform distribution, with some mixing parameter $\mu \in [ 0 , 1 ] . ^ { 7 }$ If the noisy label distribution does not have this form, then this fidelity function penalizes it by assigning a fidelity value of $- \infty$

Fidelity of Masking Channel Certifies Hardness of Learning. Our first main result states:

The highest possible fidelity of a masking channel for a distribution exactly characterizes how hard it is to learn the distribution.

More precisely, given a fidelity function $h ,$ there is a convex conjugate loss function (see Section 2.1) $g : \mathcal { Y } \times \mathbb { R } ^ { k } \to \overline { { \mathbb { R } } }$ , such that the highest fidelity of a $( \mathcal { C } , \varepsilon )$ -masking channel for D equals the smallest loss incurred in representing D by a linear combination of functions from C (with an ℓ<sub>1</sub>-regularization penalty weighted by ε).

In what follows, and in the rest of this paper, the abbreviation $\pi ( a ) = a - \langle a , { \vec { 1 } } / k \rangle$ · <sup>⃗</sup>1 denotes the “centered” mean-0 version of an action vector $a \in \mathbb { R } ^ { k }$ . Also, recall from the discussion of equation (2) that $c _ { \lambda }$ is the function in span(C) with coeficients λ. We will prove the following theorem in Section 4.

Theorem 1.9 (Hardness vs Fidelity). For all finite $x , y = [ k ]$ , distributions D over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ 2 finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , convex-concave conjugates $g : \mathcal { V } \times \mathbb { R } ^ { k } \to \mathbb { R } , h : \mathcal { V } \times \Delta _ { k } \to \overline { { \mathbb { R } } }$ , and $\varepsilon \geqslant 0 , { } ^ { 8 }$

$$
\operatorname* { i n f } _ { \lambda \in \mathbb { R } ^ { c } } \mathbb { E } [ g ( \pmb { y } , \pi ( c _ { \lambda } ( \pmb { x } ) ) ) ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { W : \mathscr { X }  \Delta _ { k } ^ { k } } \mathbb { E } [ h ( \pmb { y } , W ( \pmb { x } ) \pmb { y } ) ] \quad s . t . \quad W \mathrm { ~ } i s \mathrm { ~ } ( \mathscr { C } , \varepsilon ) - m a s k i n g .
$$

A unifying insight that runs through several works on the hardcore lemma is that learning and hardness are intimately related. Theorem 1.9 gives a particularly general and crisp formulation of this insight through the lens of convex duality.

## 1.2 Multiclass Hardcore Lemmas

We now consider concrete instantiations of Theorem 1.9. Doing so will lead to a rich landscape of distinct hardcore-like objects in the multiclass setting, each natural and interesting in its own right, that capture hardness with respect to diferent multiclass loss functions. We then connect these various hardness certificates back to the classic (multiclass) 0-1 loss via its convex surrogate losses. All the results described in this section will be proved in Section 5.

Binary Setting. As a warm-up, we state several instantiations of Theorem 1.9 in the binary setting, with specific choices of convex-concave conjugate pairs. Theorem 1.10 below includes three versions, captured by equations (3), (4), and (5).

Theorem 1.10 (Hinge, $\ell _ { 1 }$ , and $\ell _ { p }$ Hardcore Lemmas). Fix a finite set $\mathcal { X }$ , a distribution over pairs $( \pmb { x } , \pmb { y } ) \in \mathcal { X } \times \{ \pmb { \pmb { \bot } } 1 \}$ , a finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } \}$ , and $\varepsilon > 0$ . Then, using the abbreviation ${ \pmb \mu } = \mu ( { \pmb x } , { \pmb y } )$

$$
\operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { c } } \mathbb { E } [ ( 1 - c _ { \lambda } ( \boldsymbol { x } ) \boldsymbol { y } ) _ { + } ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { \boldsymbol { \mu } : \boldsymbol { \mathcal { X } } \times \boldsymbol { \mathcal { Y } }  [ 0 , 1 ] } \mathbb { E } [ \boldsymbol { \mu } ] \quad s . t . \quad \forall c \in \mathcal { C } , \ \big | \mathbb { E } [ \mu c ( \boldsymbol { x } ) \boldsymbol { y } ] \big | \leqslant \varepsilon ,\tag{3}
$$

$$
\operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { \mathcal { C } } } \mathbb { E } [ | y - c _ { \lambda } ( \boldsymbol { x } ) | ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { \substack { \mu : \mathcal { X } \times \mathcal { Y }  [ - 1 , + 1 ] } } \mathbb { E } [ \mu ] \quad s . t . \quad \forall c \in \mathcal { C } , \ | \mathbb { E } [ \mu c ( \boldsymbol { x } ) y ] | \leqslant \varepsilon ,\tag{4}
$$

$$
\operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { c } } \mathbb { E } \left[ \frac { 1 } { p } | { y - c } _ { \lambda } ( { x } ) | ^ { p } \right] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { \mu : \mathcal { X } \times \mathcal { Y } \to \mathbb { R } } \mathbb { E } \left[ \mu - \frac { 1 } { q } | \mu | ^ { q } \right] \quad s . t . \quad \forall c \in \mathcal { C } , \ \left| \mathbb { E } [ \mu c ( { x } ) y ] \right| \leqslant \varepsilon ,\tag{5}
$$

where $p , q > 1$ are any real numbers satisfying $1 / p + 1 / q = 1$

We defer the proof to Section 5, but discuss their interpretation now.

• First, equation (3) is the relationship between hinge loss and hardcore density discussed in the previous section.

• Next, equation (4) relates the minimum $\ell _ { 1 }$ loss to the maximum expected value of a $[ - 1 , + 1 ] .$ valued signed measure. $\ell _ { 1 }$ loss minimization followed by randomized rounding is the algorithm used in [KKMS08] to agnostically learn functions that are well approximated by linear combinations of C. Its dual, which allows $\mu$ to take values in the interval $[ - 1 , + 1 ]$ is quite natural; for example, multiplying y by a factor of −1 corresponds to deterministically flipping the label.

• Finally, equation (5) characterizes hardness with respect to all $\ell _ { p }$ losses for $p > 1$ . In this case, $\mu$ is both a signed and unbounded measure. Although it is no longer forced to output values in $[ - 1 , + 1 ]$ , it is nevertheless strongly encouraged to do so by a penalty term of $- ( 1 / q ) | \mu | ^ { q }$ in the objective for $q = p / ( p - 1 )$ ), which is negligible when $| \mu | \ll 1$ and very large when $| \mu | \gg 1$

Multiclass Setting. Let us now discuss three natural ways to certify the hardness of a conditional label distribution ${ \pmb y } | { \pmb x }$ in the k-class setting, all of which generalize the familiar notion of a hardcore measure in the binary case $\left( k = 2 \right)$ . Using our framework, we show that each of these three distinct hardness certificates (i.e. “hardcore-like” objects) tightly characterizes hardness with respect to a distinct multiclass loss function.

First, one might seek the densest measure $\mu$ on which the labels are C-indistinguishable from the uniform distribution $\vec { 1 } / k ,$ , or equivalently, such that the noisy channel outputting the mixture $\mu ( { \pmb x } , { \pmb y } ) \cdot { \pmb y } + ( 1 - \mu ( { \pmb x } , { \pmb y } ) ) \cdot \vec { 1 } / k$ is in fact a C-masking channel. Theorem 1.11 below shows that this density captures hardness with respect to what we call the weak hinge loss, defined for $i \in [ k ]$ and mean-0 vectors $a \in \mathbb { R } ^ { k }$ by the formula $\ell _ { \mathrm { w e a k } } ( e _ { i } , a ) = ( 1 - a _ { i } ) _ { + }$ . (If a is not mean-0, we center it first.) We call this loss “weak” because it is relatively easy to minimize: if even a single label $i \in [ k ]$ can be eliminated with certainty, then $a = \vec { 1 } - k e _ { i }$ achieves 0 loss.

Theorem 1.11 (Weak Hinge Loss vs Uniform Noise). For all finite X and $\mathcal { V } = [ k ]$ , distributions over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , and $\varepsilon > 0$ ，

$$
\operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { c } } \mathbb { E } [ \ell _ { \mathrm { w e a k } } ( \pmb { y } , c _ { \lambda } ( \pmb { x } ) ) ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { \mu : \pmb { \chi } \times \pmb { y } \to [ 0 , 1 ] } \mathbb { E } [ \mu ( \pmb { x } , \pmb { y } ) ]
$$

$$
s . t . \forall c \in { \mathcal { C } } , \ \left| { \mathbb { E } } [ \mu ( { \pmb x } , { \pmb y } ) \langle c ( { \pmb x } ) , { \pmb y } - \vec { \bf l } / k \rangle ] \right| \leqslant \varepsilon .
$$

Second, one might seek the least amount of (not necessarily uniform!) noise that must be added to the labels to achieve global indistinguishability from uniform over [k]. Theorem 1.12 below shows that among such masking channels $W$ , the maximum possible fidelity $\mathbb { E } [ \langle { \pmb y } , W ( { \pmb x } ) { \pmb y } \rangle ]$ , which can also be written $1 - \mathbb { E } [ d _ { \mathrm { T V } } ( \pmb { y } , W ( \pmb { x } ) \pmb { y } ) ]$ , captures hardness with respect to what we call the strong hinge loss, defined for $i \in [ k ]$ and mean-0 vectors $a \in \mathbb { R } ^ { k }$ by the formula $\begin{array} { r } { \ell _ { \mathrm { s t r o n g } } ( e _ { i } , a ) = \operatorname* { m a x } _ { j \in [ k ] } ( e _ { i } - a ) _ { j } } \end{array}$ (If a is not mean-0, center it first.) We call this loss “strong” simply because it is a pointwise upper bound on $\ell _ { \mathrm { w e a k } }$

Theorem 1.12 (Strong Hinge Loss vs Arbitrary Noise). For all finite X and $\mathcal { V } = [ k ]$ , distributions over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , and $\varepsilon > 0$

$$
\begin{array} { l } { \displaystyle \displaystyle \operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { \mathcal { E } } } \mathbb { E } \big [ \ell _ { \mathrm { s t r o n g } } ( y , \pi ( c _ { \lambda } ( x ) ) ) \big ] + \varepsilon \| \lambda \| _ { 1 } = \displaystyle \operatorname* { m a x } _ { W : \mathcal { X } \to \Delta _ { k } ^ { k } } \mathbb { E } \big [ \langle y , W ( x ) y \rangle \big ] } \\ { \displaystyle \qquad \quad s . t . \forall c \in \mathcal { C } , ~ \big | \mathbb { E } \big [ \langle c ( x ) , W ( x ) y - \vec { 1 } / k \rangle \big ] \big | \leqslant \varepsilon . } \end{array}
$$

Third, one might seek modified labels that are as statistically far as possible from the original labels while remaining computationally indistinguishable from them.<sup>9</sup> Theorem 1.13 below shows that the maximum attainable statistical distance captures hardness with respect to the wellestablished Crammer-Singer hinge loss [CS01], defined for $i \in [ k ]$ and $a \in \mathbb { R } ^ { k }$ by the formula $\ell _ { \mathrm { { C S } } } ( e _ { i } , a ) = ( 1 - a _ { i } + \operatorname* { m a x } _ { j \neq i } a _ { j } ) _ { \mathbb { + } }$ , which depends on the margin by which the score assigned to the correct label i exceeds the maximum score assigned to any wrong label $j \neq i . ^ { 1 0 }$ We emphasize that this statistical distance is measured from point mass distribution $\pmb { y } \in \{ e _ { 1 } , \ldots , e _ { k } \}$ , which depends on the coupling between original and noisy labels, as opposed to being measured from $\mathbb { E } [ \pmb { y } | \pmb { x } ] \in \Delta _ { k }$

Theorem 1.13 (Crammer-Singer Hinge Loss vs Maximum TV Distance). For all finite X and $\mathcal { V } = [ k ]$ , distributions over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y } .$ , finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , and $\varepsilon > 0$

$$
\begin{array}{c} \operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { C } } \mathbb { E } [ \ell _ { \mathrm { C S } } ( y , c _ { \lambda } ( x ) ) ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { p : \mathcal { X } \times \mathcal { Y } \to \Delta _ { k } } \mathbb { E } [ d _ { \mathrm { T V } } ( y , p ( x , y ) ) ]  \\ { s . t . \forall c \in \mathcal { C } , \big | \mathbb { E } [ \langle c ( x ) , y - p ( x , y ) \rangle ] \big | \leqslant \varepsilon . } \end{array}
$$

Surrogate Losses. At this point, we have shown that δ-hardness under $\ell _ { \mathrm { w e a k } }$ implies a δ-density hardcore measure, that δ-hardness under $\ell _ { \mathrm { s t r o n g } }$ implies a hardcore channel with total variation distance at most $1 - \delta ,$ and that δ-hardness for $\ell _ { \mathrm { C S } }$ implies an indistinguishable model at total variation distance at least δ. But each of these seems to be a diferent hardness assumption. In analogy to the binary hardcore lemma, one could ask: What certificates of hardness can we construct in the multiclass setting assuming only δ-hardness under the 0-1 loss $\ell _ { 0 - 1 } ( y , \tilde { y } ) = \mathbf { 1 } [ y \neq \tilde { y } ] ?$ Although this loss is not convex (for starters, because its action space is discrete), the previous consequences can be derived assuming ℓ<sub>0-1</sub>-hardness if we could show that $\ell _ { \mathrm { w e a k } } , \ell _ { \mathrm { s t r o n g } } , \ell _ { \mathrm { C S } }$ were surrogate losses for $\ell _ { 0 - 1 }$ in the following sense:

Definition 1.14 (β-surrogate Loss). Given a label set $\boldsymbol { \ y } = [ k ]$ and action sets A, B, and $\beta \geqslant 0$ , we say that a loss function $\ell _ { \mathcal { A } } : \mathcal { V } \times \mathcal { A } \to \overline { { \mathbb { R } } }$ is a β-surrogate for a loss function $\ell _ { B } : \mathcal { V } \times B \to \overline { { \mathbb { R } } }$ if B is finite and there exists a rounding function $R : { \mathcal { A } } \to \Delta ( B )$ such that for all $y \in \mathcal { V }$ and $a \in { \mathcal { A } }$

$$
\ell _ { \cal A } ( y , a ) \geqslant \beta \cdot { \mathbb { E } } _ { b \sim R ( a ) } [ \ell _ { \cal B } ( y , b ) ] .
$$

We show that whether or not a loss function, such as $\ell _ { \mathrm { w e a k } } , \ell _ { \mathrm { s t r o n g } } , \ell _ { \mathrm { C S } }$ , is a surrogate for the 0-1 loss can be completely determined from its associated entropy function:

Definition 1.15 (Entropy). Given a label set Y, an action set ${ \mathcal { A } } ,$ and a loss function $\ell : \mathcal { V } \times \mathcal { A }  \overline { { \mathbb { R } } }$ the entropy of ℓ is the concave function $h : \Delta ( \mathcal { V } ) \to \overline { { \mathbb { R } } }$ given by the formula

$$
h ( p ) = \operatorname* { i n f } _ { a \in { \mathcal { A } } } \mathbb { E } _ { { \pmb y } \sim p } [ \ell ( { \pmb y } , a ) ] .
$$

Theorem 1.16 (Entropy Surrogates). Given sets $\mathcal { A } , B , \mathcal { y }$ and $\beta \geqslant 0$ and finite losses $\ell _ { \mathcal { A } } : \mathcal { V } \times \mathcal { A } $ R and $\ell _ { B } : \mathcal { V } \times \mathcal { B } \to \mathbb { R }$ , let $h _ { \mathcal { A } } , h _ { \mathcal { B } } : \Delta ( \mathcal { V } )  \overline { { \mathbb { R } } }$ denote their entropy functions. $I f \mathcal { V } , B$ are finite, then $\ell _ { A }$ is a β-surrogate loss for $\ell _ { B }$ if and only if $h _ { \mathcal { A } } ( p ) \geqslant \beta h _ { \mathcal { B } } ( p )$ for all $p \in \Delta ( \mathcal { V } )$

An additive term can easily be incorporated in the Definition 1.14 and Theorem 1.16 (i.e. $h _ { \mathcal { A } } ( p ) \geqslant \alpha + \beta h _ { \mathcal { B } } ( p ) )$ . This result is proved using the MinMax theorem. For the special case when $\ell _ { B } = \ell _ { 0 - 1 }$ , we present an alternate proof with an explicit rounding. We will also show that simple projection onto $\Delta _ { k }$ is an optimal rounding rule for $ { \ell _ { \mathrm { { C S } } } }$

With this characterization, by calculating the entropy functions and coeficients $\beta$ for loss functions, we can compare our results for those losses for the binary and multiclass case, as follows:

$\ell _ { \mathrm { w e a k } }$ is not a surrogate loss for $\ell _ { 0 - 1 }$ unless $k = 2$ , in which case it matches the hinge loss. Thus, for $k > 2$ , one cannot conclude the hardness of minimizing $\ell _ { \mathrm { w e a k } }$ from hardness for $\ell _ { 0 - 1 }$ Indeed, for a deterministic labeling function, the learner which is able to eliminate just 1 in k labels but then guesses at random will have 0 loss for $\ell _ { \mathrm { w e a k } }$ , but loss $( k - 2 ) / ( k - 1 )$ for $\ell _ { 0 - 1 }$

$\ell _ { \mathrm { s t r o n g } }$ is a surrogate for $\ell _ { 0 - 1 }$ with coeficient $\beta _ { \mathrm { s t r o n g } } = { 4 / k }$ . In particular, δ-hardness under $\ell _ { 0 - 1 }$ implies $4 \delta / k { \mathrm { - h a r d n e s s } }$ under $\ell _ { \mathrm { s t r o n g } } .$ , and hence a hardcore channel whose noisy labels are at total variation distance at most $1 - 4 \delta / k$ . When $k = 2 .$ , in fact $h _ { \mathrm { s t r o n g } } ( p ) \geqslant 1 / 2 + h _ { 0 - 1 } ( p )$ from which δ-hardness under $\ell _ { 0 - 1 }$ implies $( 1 / 2 + \delta )$ -hardness under $\ell _ { \mathrm { s t r o n g } } .$ , and hence a channel whose noisy labels are at distance $1 / 2 - \delta .$ , as expected (and shown in Figure 1).

$\ell _ { \mathrm { C S } }$ is a surrogate loss with $\beta _ { \mathrm { C S } } ~ = ~ k / ( k - 1 )$ . Hence δ-hardness under $\ell _ { 0 - 1 }$ implies a $\mathcal { C } _ { - }$ indistinguishable model at TV distance at least $k \delta / ( k - 1 )$ . This is 2δ-distance when $k = 2 .$ where for a Boolean function, such a model can be obtained from a 2δ-dense hardcore set S by complementing labels on S.

Thus we see that the familiar 2δ for the binary setting generalizes diferently to larger k for difering loss functions.

We remark that these are not ${ } ^ { 6 6 } \mathrm { { \dot { 1 } } \mathrm { { f } } }$ and only $\mathrm { i f } ^ { \dag }$ statements, as additional scaling factors are incurred by converting back to $\ell _ { 0 - 1 }$ hardness. For example, a δ-distance indistinguishable model only implies $\delta / 2 \cdot$ -hardness for $\ell _ { 0 - 1 }$ , not $( 1 - 1 / k ) \delta \cdot$ -hardness for $\ell _ { 0 - 1 }$ , since all the mass at a label $y \in \mathcal { V }$ may have simply been moved to a diferent $\tilde { y } \in \mathcal { V }$ , rather than spread out evenly among all labels.

## 1.3 Universal Masking Channels From Multicalibration

Although Theorem 1.9 gives an exact quantitative relationship between average-case hardness and masking channel fidelity, for each hardness notion g and each associated fidelity notion h, the optimal channel W may difer depending on the choice of h. What if we want a single masking channel that witnesses the hardness of f against C for all fidelity functions h simultaneously?

This question of the existence of a universal channel brings to mind recent results in learning theory regarding the existence of omnipredictors, which are predictive models that outperform a benchmark class C with respect to a wide range of loss functions $g$ simultaneously, rather than being tailored to one specific such $g .$ Perhaps surprisingly, it is known that such omnipredictors indeed exist for arbitrary families of loss functions satisfying mild conditions $[ \mathrm { G K R ^ { + } 2 2 } , \mathrm { G H K ^ { + } 2 3 } ]$ In light of the duality between learning and masking channels, the question posed above seeks the “dual” channel corresponding to an omnipredictor.

We show that even for an entire collection of loss-fidelity pairs $g , h ,$ there exists a single universal model $p : \mathcal { X }  \Delta _ { k }$ of the conditional label distribution that is not only an omnipredictor for the losses $g$ in the class, but also yields approximately optimal-fidelity masking channels for all h in the family, provided that we slightly enlarge C. Put diferently, the action used to minimize g and the channel used to maximize $h$ will only depend on $x$ through the model’s prediction $p ( x )$ . To state this formally, we will express the chosen action as $\hat { c } _ { g } ( p ( x ) )$ ), where $\hat { c } _ { g }$ is the lowest-loss response to $p ( x )$ with respect to $^ { g , }$ and we will express the chosen channel as $\widehat { W } _ { h } ( p ( x ) )$ , where $\widehat { W } _ { h } ( p ( x ) )$ is the highest-fidelity response to $p ( x )$ with respect to h. We define these best responses as follows:

Definition 1.17 (Lowest-Loss Response). Given a function $g : \mathcal { V } \times \mathbb { R } ^ { k }  \overline { { \mathbb { R } } }$ , a distribution $z \in \Delta _ { k }$ and a label $\mathbf { \nabla } _ { \mathbf { y } } \sim \mathbf { \nabla } _ { z }$ , a lowest-loss response to z with respect to $g$ is any

$$
\hat { c } _ { { g } } ( \boldsymbol { z } ) \in \operatorname * { a r g m i n } _ { \boldsymbol { c } ( \boldsymbol { z } ) \in \mathbb { R } ^ { k } } \mathbb { E } [ \boldsymbol { g } ( \boldsymbol { y } , \boldsymbol { c } ( \boldsymbol { z } ) ) ] \quad \mathrm { s . t . } \quad \langle \boldsymbol { c } ( \boldsymbol { z } ) , \vec { 1 } \rangle = 0 ,
$$

assuming any exist. We instead say $\hat { c } _ { g }$ is γ-optimal if it achieves loss within $\gamma$ of the infimum.

This lowest-loss response function plays a prominent role in the literature on omniprediction $[ \mathrm { G K R ^ { + } 2 2 } , \mathrm { G H K ^ { + } 2 3 } ]$ , where it is used to post-process predictions for a specific loss.

Definition 1.18 (Highest-Fidelity Response). Given a fidelity function $h : \mathcal { V } \times \Delta _ { k } \to \overline { { \mathbb { R } } }$ , a distribution $z \in \Delta _ { k }$ , and a label $\mathbf { \Omega } _ { \mathbf { \mathcal { Y } } } \sim \mathbf { \mathcal { Z } }$ , a best-fidelity response to z with respect to $h$ is any

$$
\widehat { W } _ { h } ( z ) \in \operatorname { a r g m a x } _ { W ( z ) \in \Delta _ { k } ^ { k } } \mathbb { E } [ h ( \pmb { y } , W ( z ) \pmb { y } ) ] \quad \mathrm { s . t . } \quad W ( z ) z = \frac { \vec { 1 } } { k } ,
$$

assuming any exist. We instead say $\widehat { W } _ { h }$ is γ-optimal if it achieves fidelity within γ of the maximum.

The space of channels we optimize over in this definition maps z to the uniform distribution, as in Example 1.4. Thus high-fidelity responses generalize the re-balancing functions on the level sets of a predictor, which are used in the proof of the hardcore lemma in [TTV09, CDV24, CGKR25].

Better yet, if each fidelity function $h ( y , z )$ in the collection is a nondecreasing function of $\langle y , z \rangle$ as the ones in Examples 1.6 and 1.7, then there not only exists a universal model $p ,$ but there also exists a universal highest-fidelity post-processing $\widehat { W } = \widehat { W } _ { h }$ of $p$ that works for all fidelity functions h in the collection. We call such fidelity functions monotone. Besides monotone fidelity functions, we also consider the natural class of monotone mixing fidelity functions, such as the one in Example 1.8, which constrain the noisy label distribution $z$ to be a mixture of the original label y and the uniform distribution:

Definition 1.19 (Monotone Fidelity). Let $\mathcal { V } = \{ e _ { 1 } , \ldots , e _ { k } \}$ . We say a fidelity function $h : \mathcal { V } \times$ $\Delta _ { k }  \overline { { \mathbb { R } } }$ is monotone if there is a nondecreasing function $h _ { 0 } : \mathbb { R }  \overline { { \mathbb { R } } }$ such that

$$
h ( y , z ) = { \left\{ \begin{array} { l l } { h _ { 0 } ( \langle y , z \rangle ) } & { { \mathrm { i f ~ } } z \in \Delta _ { k } , } \\ { - \infty } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

We instead say that h is monotone mixing if for some such $h _ { 0 }$

$$
h ( y , z ) = \left\{ \begin{array} { l l } { h _ { 0 } ( \mu ) } & { \mathrm { i f ~ } z = \mu y + ( 1 - \mu ) ( \vec { 1 } / k ) \in \Delta _ { k } \mathrm { ~ f o r ~ s o m e ~ } \mu \in [ 0 , 1 ] , } \\ { - \infty } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Although monotone mixing fidelity functions might appear to be a subset of monotone fidelity functions, this is not the case because of their difering efective domains.

In the following theorem statement, there are three parts. First, part (a) asserts that the channels are indeed “valid” or “feasible”, in the sense that they lower bound the loss of the class C for $f .$ Next, part (b) asserts that the bound from part (a) is nearly tight, in the sense that some circuit of “complexity” not much higher than that of the complexity class C indeed achieves similar loss. Finally, part (c) asserts that a single masking channel works for all monotone fidelity functions, or for all monotone mixing fidelity functions.

Theorem 1.20 (Universal Masking Channels, Informal Version of Theorems 6.4, 6.5, and 6.6). For all finite X and $\mathcal { V } = [ k ]$ , distributions over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , and $\varepsilon > 0$ 2 and a family $\mathcal { F }$ of tuples $( g , h , \widehat { c } _ { g } , \widehat { W } _ { h } )$ , where $g , h$ are convex-concave conjugates and ${ \widehat { c } } _ { g } , { \widehat { W } } _ { h }$ are best responses, there exists a model $p : \mathcal { X }  \Delta _ { k }$ of “complexity” $O ( \log ( k ) / \varepsilon ^ { 2 } )$ times that of the functions in C and $\mathcal { F } , ^ { 1 1 }$ such that the following properties hold for all $( g , h , \widehat { c } _ { g } , \widehat { W } _ { h } ) \in \mathcal { F }$ :

(a) (Hardness $\geqslant$ Average Channel Fidelity) The channel $W _ { p } = \widehat { W } _ { h } \circ p$ obtained from postprocessing the model p certifies hardness against ${ \mathcal { C } } ,$ in the sense that for all $c \in { \mathcal { C } } _ { i }$

$$
\begin{array} { r } { \mathbb { E } [ g ( \pmb { y } , \pi ( c ( \pmb { x } ) ) ) ] \geqslant \mathbb { E } [ h ( \pmb { y } , W _ { p } ( \pmb { x } ) \pmb { y } ) ] - O ( \varepsilon ) . } \end{array}
$$

(b) (Hardness $\leqslant$ Average Channel Fidelity) The action $c _ { p } = \hat { c } _ { g } \circ p$ derived from postprocessing the model p is nearly tight for the bound from part (a):

$$
\mathbb { E } [ g ( { \pmb y } , c _ { p } ( { \pmb x } ) ) ] \leqslant \mathbb { E } [ h ( { \pmb y } , W _ { p } ( { \pmb x } ) { \pmb y } ) ] + O ( \varepsilon ) .
$$

Note that even $i f c _ { p } \notin { \mathcal { C } } _ { : }$ , its “complexity” is at most $O ( \log ( k ) / \varepsilon ^ { 2 } )$ times that of C.

(c) (Universal Masking Channel) A single $\widehat { W } = \widehat { W } _ { h }$ works for all monotone fidelity functions. A (diferent) single $\widehat { W } = \widehat { W } _ { h }$ works for all monotone mixing fidelity functions.

The key technical tool underlying the proof of Theorem 1.20, which we present in Section $6 ,$ is multicalibration [HKRR18]. Multicalibration is an enhanced notion of computational indistinguishability that has been used both in omniprediction [GKR<sup>+</sup>22, $\mathrm { G H K ^ { + } 2 3 } ]$ and in recent new proofs of the traditional hardcore lemma [CDV24, CGKR25]. A model $p : \mathcal { X }  \Delta _ { k }$ of $_ { y | \pmb { x } }$ is said to be $( \mathcal { C } , \varepsilon )$ -multicalibrated if the level sets of p partition X into regions where the real label y is indistinguishable from a simulated label distributed according to $p ( { \pmb x } )$ . More formally,

$$
\mathbb { E } \Big [ \operatorname* { m a x } _ { c \in \mathcal { C } } \big | \mathbb { E } [ \langle c ( { \pmb x } ) , { \pmb y } - p ( { \pmb x } ) \rangle \mid p ( { \pmb x } ) ] \big | \Big ] \leqslant \varepsilon .
$$

It is known that regardless of the complexity of the conditional distribution ${ \pmb y } | { \pmb x }$ , multicalibrated models always exist with complexity comparable to that of C [HKRR18], and the complexity gap can be shrunk considerably in the cases that we only require indistinguishability on certain weighted combinations of the level sets of $p ,$ as opposed to all level sets [GKSZ22].

![](images/f9573672aa9c7ecdf45c0352a6e323e8891846084652f9bdca3efbf07d38a8ac.jpg)  
(a) Move mass from above $1 / k$ to below $1 / k$

![](images/925cbf22ad4070c06ab0bd735de20893a4539330923720ee93df01f83bf53e7e.jpg)  
(b) Uniformly redistribute mass above min $\textit { i } ^ { z _ { i } }$  
Figure 2: Multicalibration-based constructions of the universal noisy channel (k = 4).

Given a multicalibrated model, it is not too hard to see how to construct an optimal universal masking channel for monotone or monotone mixing fidelity functions. Indeed, given a label distribution $z \in \Delta _ { k }$ , consider the z-level set of the model $p : \mathcal { X }  \Delta _ { k }$ , which we denote by $p ^ { - 1 } ( z ) \subseteq \mathcal { X }$ On this level set, we know that each label $\textbf {  { y } }$ is indistinguishable from a fresh sample z. Therefore, at least intuitively, any noisy channel that redistributes the mass of z into the uniform distribution should also map the label y to a noisy label $\tilde { \mathbf { \pmb { y } } }$ that is indistinguishable from uniform. Among such channels, it is clear that the highest fidelity (least noisy) choice is the channel that redistributes mass from the label set $S _ { z } ^ { \mathrm { h i g h } } = \{ i \in [ k ] \mid z _ { i } > 1 / k \}$ to the label set $S _ { z } ^ { \mathrm { l o w } } = \{ i \in [ k ] \ | \ z _ { i } < 1 / k \}$ This construction of the universal channel is illustrated in Figure 2a.

For monotone mixing fidelity functions, which only allow the channel to redistribute mass uniformly, the construction instead uniformly redistributes excess mass from the label set $S _ { z } ^ { \mathrm { \neq m i n } } =$ $\left\{ i \in [ k ] \ \vert \ z _ { i } > \ \operatorname* { m i n } _ { j } z _ { j } \right\}$ . This alternate construction is illustrated in Figure 2b. In fact, both constructions can be seen as multiclass generalizations of a previously known multicalibrationbased construction of a hardcore measure in the binary 0-1 loss setting [CDV24, CGKR25]. We show in Section 6.2 how to recover this result from our new one.

Remark 1.21. The formal version of Theorem 1.20 in Section 6 is stronger than the informal version sketched here. While our informal discussion uses multicalibration, the full version will only require calibration [Daw82] and weighted multiaccuracy [TTV09, HKRR18]. These conditions are more eficiently achieved than multicalibration, improving the complexity of $p$ relative to C.

## 1.4 Characterizing Pseudoentropy via MaxEnt

In this section, we discuss the relationship between our results on masking channels and related work on pseudoentropy [VZ12, Zhe14, HV25]. We show that a tight learning-theoretic characterization of pseudoentropy can be derived using the MaxEnt algorithm [BPP96, LDPDP97, DPS07, AS06].

Much like our high fidelity masking channels, the property of having high pseudoentropy [HILL99] generalized in [VZ12, Zhe14, HV25], is a hardness-certifying relaxation of the hardcore property. This property states that $f : \mathcal { X } \to \{ \pm 1 \}$ itself is C-indistinguishable from a model $p : \mathcal { X }  [ - 1 , + 1 ]$ that is highly random. In this most basic setting, “highly random” means being $( 1 / 2 - \delta )$ -close in statistical distance to pure random noise. Equivalently, the model $p$ has high $\ell _ { 0 - 1 }$ entropy: $\mathbb { E } [ h _ { 0 - 1 } ( p ( { \pmb x } ) ) ] \geqslant \delta$ , where $h _ { 0 - 1 } ( v ) = ( 1 - | v | ) / 2$ (see Definition 1.15). As with the masking channels property, it is easy to check that high pseudoentropy certifies hardness. Also, randomizing labels on the hardcore measure yields the high-entropy model $p ( x ) = ( 1 - \mu ( x ) ) f ( x )$ . There is a symmetry between these two approaches to characterizing average-case hardness, which Figure 3 depicts in the most basic setting.

![](images/2fdcbdc4fcffee6e98f66653636926b4384e9e801ebe7de566a4857ac8b680e1.jpg)  
Figure 3: Symmetry between Theorem 1.9 (Noisy Channels) and Theorem 1.22 (MaxEnt) for binary 0-1 loss. The polytopes are C-indistinguishability regions, one centered at the original $f$ and another at pure noise 0. If $f \ ( \{ \pm 1 \} \mathrm { - v a l u e d } )$ is δ-hard for C and if µ ([0, 1]-valued) is a 2δ-dense hardcore measure, the high-entropy model is $( 1 - \mu ) f \approx c f$ and the noisy channel output is $\mu f \approx _ { \mathscr { C } } 0$

As shown in [VZ12, Zhe14], having high pseudoentropy, much like our novel noisy channel property, extends naturally to the multiclass setting and to essentially arbitrary concave “entropy” functions $h : \Delta _ { k }  \mathbb { R }$ , such as the 0-1 entropy function $1 - \operatorname* { m a x } _ { i } p _ { i }$ , the Shannon entropy $\textstyle \sum _ { i } p _ { i } \log ( 1 / p _ { i } )$ and many more. They show that for any reasonably well-behaved entropy function h and associated GLM loss function $\ell ,$ the best loss attainable by circuits in a class $\mathcal { C } ^ { \prime }$ related to $\mathcal { C }$ is approximately equal to the maximum entropy of a C-indistinguishable model.<sup>12</sup>

The elegant recent work of Hu and Vadhan [HV25] shows that in fact, even for an entire collection of entropy functions h and associated GLM loss functions, there exists a single universal model p with complexity comparable to that of C, whose h-entropy certifies the hardness of $f$ for the class C, and whose own loss on f is close to the certified lower bound.

Our new Theorem 1.20 on universal masking channels is inspired by their result and this symmetry between masking channels and pseudoentropy. But this raises the question of whether there is a parallel to Theorem 1.9. In other words, is there a learning problem that tightly characterizes pseudoentropy?

We show that the answer comes from a line of work in the natural language processing on the MaxEnt algorithm [BPP96, LDPDP97, DPS07, AS06]. The MaxEnt algorithm attempts to learn a probability distribution of the highest possible entropy, subject to a collection of linear constraints imposed by observed data points. In our setting, we wish to learn $p ( x ) = \mathbb { E } [ { \pmb y } | { \pmb x } = x ]$ under the linear constraints of C-indistinguishability. The objective is $\mathbb { E } [ h ( p ( { \pmb x } ) ) ]$ , the model’s average h-entropy. The convex dual is known to be the problem of minimizing a generalized linear model (GLM) loss over span(C), with $\ell _ { 1 }$ regularization; yielding a learning problem that exactly characterizes pseudoentropy.

Theorem 1.22 (Characterizing pseudoentropy using MaxEnt [BPP96, LDPDP97, DPS07, AS06]). For all finite X and $\mathcal { V } = [ k ]$ , distributions D over pairs $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ and convex-concave conjugate pairs $g : \mathbb { R } ^ { k }  \mathbb { R }$ and $h : \Delta _ { k } \to \overline { { \mathbb { R } } }$ , let $\ell ( y , \eta ) = g ( - \eta ) - \langle y , \eta \rangle$ (GLM

loss). Then, for all $\varepsilon \geqslant 0 , ^ { 1 3 }$

$$
\operatorname* { i n f } _ { \lambda \in \mathbb R ^ { C } } \mathbb { E } [ \ell ( y , c _ { \lambda } ( x ) ) ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { p : \mathcal X \to \Delta _ { k } } \mathbb { E } [ h ( p ( x ) ) ] { \quad s . t . \quad p ( x ) } \ i s \ ( \mathcal { C } , \varepsilon ) - i n d i s t i n g u i s h a b l e \ f r o m \ y .
$$

In Section 7, we will show that Theorem 1.22, just like our new Theorems 1.9 and 1.20, can be derived from the same underlying general framework based on convex duality.

## 1.5 Related Work

Impagliazzo’s Hardcore Lemma [Imp95, Hol05] is a cornerstone result in our understanding of the average-case complexity of Boolean functions. It has applications to hardness amplification [Imp95, Tre03, O’D04], the derandomization of BPP [IW97], cryptography [Hol05] and the design of better boosting algorithms [KS03]. There has been an extensive body of work devoted to optimizing, generalizing and extending this result. While a comprehensive survey of this work is beyond the scope of this section, we identify three broad approaches to proving the hardcore lemma that are relevant to our work.

Boosting/Learning. As observed by Klivans and Servedio [KS03], Impagliazzo’s original constructive proof is naturally viewed as a boosting procedure. We assume that the function $f$ is not too hard for dense subsets; that there is a hypothesis $c _ { S }$ that does well for every dense subset S. We use boosting to combine these hypotheses into one that does well on average. This approach is further developed in the works of [Hol05, BHK09, Fel10]. Better boosting algorithms yield improvements in various parameters of the hardcore lemma, and more recently in works such as [BCHM20, BDMM23, BBC<sup>+</sup>25].

MinMax. Impagliazzo presents an alternate proof of the hardcore lemma by Nisan using the MinMax theorem [Imp95]. We consider a two-player zero-sum game between a learner who makes predictions using hypotheses $c \in { \mathcal { C } }$ and an adversary who uses dense subsets S, the payof is how wel c can predict $f$ on S. Depending on the value of the game, either the function is easy on average, or there is a dense hardcore measure. This approach is used by [Hol05, TTV09, VZ12, VZ13].

Regularity. This approach was initiated in the seminal work of Trevisan, Tulsiani and Vadhan [TTV09]. We use a randomized function $p : \mathcal { X }  [ - 1 , 1 ]$ which is a regular model of the Boolean function $f : \mathcal { X } \to \{ \pm 1 \}$ , meaning it is indistinguishable from f to $\mathcal { C } . ^ { 1 4 }$ We then explicitly reweight the input x using the pair $( p ( x ) , f ( x ) )$ to get a hardcore measure. The notion of regularity in [TTV09] corresponds to multiaccuracy [HKRR18]. More recent works [CDV24, CGKR25, HV25] derive stronger results using stronger regularity notions, such as multicalibration or calibrated multiaccuracy which originated in the multigroup fairness literature [HKRR18, GKR<sup>+</sup>22, GHK<sup>+</sup>23]. We discuss these in more detail shortly.

Our work builds on key ideas from each of these lines of work, and generalizes them substantially. The use of Fenchel duality considerably generalizes the use of the MinMax Theorem for 2 player games, which is implied by it [Ber15]. While learning and hardness have long been understood to be two sides of the same coin, our formulation reveals them to be convex duals. Strong notions of regularity are crucial in our results on universal masking channels, which simultaneously certify hardness for a number of learning tasks, which we view as dual objects to omnipredictors.

Having discussed the techniques used to prove the hardcore lemma, we now discuss extensions of the original hardcore lemma which are particularly relevant to our work.

Beyond 0-1 Loss. Much of the initial work on the hardcore lemma focuses on 0-1 loss and min-entropy. The work of Vadhan and Zheng considered cross-entropy loss and Shannon entropy, motivated by simpler constructions of pseudorandom generators from one-way functions [VZ12, VZ13]. Zheng’s thesis considers more general entropy functions and Bregman divergences [Zhe14]. These works view the hardcore lemma as explaining hardness through pseudoentropy; given a distribution $( { \pmb x } , f ( { \pmb x } ) )$ , the aim is to construct a C-indistinguishable distribution $( { \boldsymbol { x } } , { \boldsymbol { y } } ^ { \prime } )$ that has high conditional entropy $\mathbb { E } [ H ( { \pmb y } ^ { \prime } | { \pmb x } ) ]$ ]. They characterize pseudoentropy by the optimal Bregman divergence achievable when sampling y|x using an eficient sampler. By standard characterizations of proper losses [Sav71], this is equivalent to minimizing a proper loss on the distribution (x, y).

Multiple Losses at Once. The work of [GKR<sup>+</sup>22] introduced the notion of an $( { \mathcal { L } } , { \mathcal { C } } )$ -omnipredictor, which is a single universal predictor that can perform competitively with the best hypothesis from a class ${ \mathcal { C } } ,$ for any loss in a family L of loss functions. The best hypothesis in C can be specific to the loss function, but the omnipredictor is loss-oblivious. The elegant recent work of Hu and Vadhan [HV25] gives an analogous statement from the hardness perspective: they show that a suitable predictor p yields a joint distribution $( { \pmb x } , { \pmb y } \sim p ( { \pmb x } ) )$ ) that has optimal pseudoentropy for a large family of entropy functions. The conditions on $p$ (calibrated multiaccuracy) and on the entropy/loss functions (proper loss) exactly match those needed for loss outcome indistinguishability (loss OI) for proper loss functions [GHK<sup>+</sup>23], which is a strong form of omniprediction. Our construction of universal masking channels is inspired by their work.

Surrogate Losses. Surrogate losses are used in machine learning as a way to come up with tractable convex substitutes for non-convex losses like 0-1 loss [BJM06, TB07]. Indeed, many commonly used convex losses like hinge loss, $\ell _ { 1 }$ loss and logistic loss are known to give upper bounds on the 0-1 loss [BJM06]. The tight factor of 2 for $\ell _ { 1 }$ and hinge loss is crucial to optimality in the learning algorithm of [KKMS08]. Other works that give bounds for surrogate losses for 0-1 loss in the multiclass setting include [vdH20, vdHFCB21]. Theorem 1.16 which gives a tight characterization of the optimal rounding factor for any pair of losses via entropy appears to be new.

## 1.6 AI Methodology

ChatGPT 5.6 and 6 were used for several brainstorming discussions, to speed up the calculation of numerous convex conjugates, dual programs, and entropy functions, to determine the rounding schemes presented in Theorem 1.16 and Lemma 5.2 and the associated entropy functions and surrogate constants, to determine the precise technical conditions needed for strong duality and the attainment of optimizers in our application of convex duality, to search for and understand relevant related work, and to generate figures from screenshots of hand-drawn sketches. All other theorems were formulated and proved without substantial AI assistance. None of the writing in this paper (exposition, definitions, theorems, proofs, etc.) was done by an AI model.

## 2 Preliminaries

In this section, we review the relevant background material on convex analysis (Section 2.1) and multicalibration (Section 2.3) needed to prove our main results. We especially encourage readers

who have not previously seen Fenchel duality to read Theorem 2.9 and the subsequent remarks, as there are some diferences from the more familiar notion of linear programming duality.

## 2.1 Convex Analysis

In this section, we briefly describe the elements of convex analysis that we require to prove our general theorem on average-case hardness. For simplicity, we restrict attention to the finite-dimensional case, which is the only case that we will require in this paper. We assume the reader is already familiar with the definitions of convex sets and convex/concave functions. The following definitions are all more or less standard (see, for example, [Roc70]).

Definition 2.1 (Extended Real Numbers). Let $\overline { { \mathbb { R } } } = \mathbb { R } \cup \{ \pm \infty \}$ denote the extended real numbers. The efective domain of an extended-real-valued function $f : \mathbb { R } ^ { n }  { \overline { { \mathbb { R } } } }$ is the set

$$
\operatorname { d o m } f = \{ x \in \mathbb { R } ^ { n } \mid f ( x ) \in \mathbb { R } \} .
$$

Definition 2.2 (Closed Proper Convex/Concave Functions). We say that $g : \mathbb { R } ^ { n }  { \overline { { \mathbb { R } } } }$ is a closed proper convex function if it is lower semicontinuous (i.e. has closed sublevel sets $g ^ { - 1 } ( ( - \infty , v ] )$ for each $v \in \mathbb { R } )$ , if it is proper (i.e. never outputs −∞, does not always output +∞), and if it is convex over its efective domain, which must be a convex set. If $g : S  { \overline { { \mathbb { R } } } }$ is defined only on a subset $S \subseteq \mathbb { R } ^ { n }$ , then whether or not g is a closed proper convex function is judged from its extension

$$
{ \bar { g } } ( x ) = { \left\{ \begin{array} { l l } { g ( x ) } & { { \mathrm { i f ~ } } x \in S , } \\ { + \infty } & { { \mathrm { i f ~ } } x \not \in S . } \end{array} \right. }
$$

We say that $h : S  { \overline { { \mathbb { R } } } }$ is a closed proper concave function if −h is a closed proper convex function.

Example 2.3 (Convex Indicator). If $S \subseteq \mathbb { R } ^ { k }$ is a closed nonempty convex set, then its convex indicator $I _ { S } : \mathbb { R } ^ { k }  \overline { { \mathbb { R } } }$ , defined as follows, is a closed proper convex function:

$$
I _ { S } ( x ) = { \left\{ \begin{array} { l l } { 0 } & { { \mathrm { i f ~ } } x \in S , } \\ { + \infty } & { { \mathrm { i f ~ } } x \not \in S . } \end{array} \right. }
$$

Definition 2.4 (Subgradients). We say that a vector $v \in \mathbb { R } ^ { n }$ is a subgradient of a function $f$ : $\mathbb { R } ^ { n } \to { \overline { { \mathbb { R } } } }$ at $a \in \mathbb { R } ^ { n }$ if for all $b \in \mathbb { R } ^ { n }$ 2

$$
f ( b ) \geqslant f ( a ) + \langle v , b - a \rangle .
$$

In general, subgradients need not exist or be unique. The set of all subgradients of $f$ at a is called the subdiferential at $^ { a , }$ and it is denoted by $\partial f ( a )$ . (It is known that if $f$ is convex, then $\partial f ( a ) = \{ \nabla f ( a ) \}$ at each point a where f is diferentiable.)

Definition 2.5 (Convex Conjugates). We say that two functions $f , g : \mathbb { R } ^ { n }  \overline { { \mathbb { R } } }$ are convex conjugates if they are both closed proper convex functions and

$$
\begin{array} { r } { f ( a ) = \underset { b } { \operatorname* { s u p } } \langle a , b \rangle - g ( b ) , } \\ { g ( b ) = \underset { a } { \operatorname* { s u p } } \langle a , b \rangle - f ( a ) . } \end{array}
$$

(It is known that every closed proper convex function f has a unique convex conjugate g, and that applying this transformation twice recovers the original $f . )$

Definition 2.6 (Convex-Concave Conjugates). We say that $g , h : \mathbb { R } ^ { n }  \overline { { \mathbb { R } } }$ are convex-concave conjugates if g is a closed proper convex function, h is a closed proper concave function, and the functions $a \mapsto g ( - a )$ and $b \mapsto - h ( b )$ are convex conjugates as per Definition 2.5. Equivalently,

$$
\begin{array} { l } { { g ( a ) = \displaystyle \operatorname* { s u p } _ { b } h ( b ) - \langle a , b \rangle , } } \\ { { \displaystyle h ( b ) = \operatorname* { i n f } _ { a } g ( a ) + \langle a , b \rangle . } } \end{array}
$$

Definition 2.7 (Convex-Concave Conjugates With Two Inputs). Given finite $\ y = [ k ]$ , we say that the two-input functions $g , h : \mathcal { V } \times \mathbb { R } ^ { n } \to \overline { { \mathbb { R } } }$ are convex-concave conjugates if the single-input functions $g ( y , \cdot )$ and $h ( y , \cdot )$ are convex-concave conjugates as per Definition 2.6 for all $y \in \mathcal { V }$

Definition 2.8 (Duality Map). Given convex conjugates $f , g : \mathbb { R } ^ { n }  \overline { { \mathbb { R } } }$ , we say that points $a , b \in \mathbb { R } ^ { n }$ are dual to each other if they make equality hold in the following inequality, called the Fenchel-Young inequality, which holds for all $a , b \in \mathbb { R } ^ { n }$

$$
f ( a ) + g ( b ) \geqslant \langle a , b \rangle .
$$

(When f and g are diferentiable, it is known that $a , b$ are dual to each other if $a = \nabla g ( b )$ if $b = \nabla f ( a )$ . In general, $a , b$ are dual to each other if $a \in \partial g ( b )$ if $b \in \partial f ( a ) .$ ) Similarly, given a convex-concave conjugate pair $g , h ,$ , we say that $a , b$ are dual to each other if the following inequality holds with equality:

$$
g ( a ) \geqslant h ( b ) - \langle a , b \rangle .
$$

Theorem 2.9 (Fenchel’s Duality Theorem, Special Case [Roc70]). Let $g _ { 1 } , h _ { 1 } : \mathbb { R } ^ { n }  \overline { { \mathbb { R } } }$ and $g _ { 2 } , h _ { 2 }$ $\mathbb { R } ^ { n } \to { \overline { { \mathbb { R } } } }$ be two convex-concave conjugate pairs. If at least one of these four functions is finite everywhere (i.e. never outputs $\pm \infty )$ , then

$$
\operatorname* { i n f } _ { a } g _ { 1 } ( a ) + g _ { 2 } ( a ) = \operatorname* { s u p } _ { b } h _ { 1 } ( b ) + h _ { 2 } ( - b ) .\tag{6}
$$

If the finite function among the four was one of $g _ { 1 } , g _ { 2 } ,$ , then the supremum on the right-hand side is achieved by some particular $b ^ { * } \in \mathbb { R } ^ { n }$ . If the finite function among the four was one of $h _ { 1 } , h _ { 2 }$ , then the infimum on the left-hand side is achieved by some particular $a ^ { \ast } \in \mathbb { R } ^ { n }$

Remark 2.10. To build intuition for Theorem 2.9, here is a neat “proof”:

$$
\begin{array} { l } { \displaystyle \operatorname* { i n f } _ { a } \big ( g _ { 1 } ( a ) + g _ { 2 } ( a ) \big ) = \operatorname* { i n f } _ { a _ { 1 } , a _ { 2 } } \operatorname* { s u p } _ { b } \big ( g _ { 1 } ( a _ { 1 } ) + g _ { 2 } ( a _ { 2 } ) + \langle a _ { 1 } - a _ { 2 } , b \rangle \big ) } \\ { \displaystyle \quad \overset { ? } { = } \operatorname* { s u p } _ { b } \operatorname* { i n f } _ { a _ { 1 } , a _ { 2 } } \big ( g _ { 1 } ( a _ { 1 } ) + g _ { 2 } ( a _ { 2 } ) + \langle a _ { 1 } - a _ { 2 } , b \rangle \big ) } \\ { \displaystyle \quad = \operatorname* { s u p } _ { b } \big ( \operatorname* { i n f } _ { a _ { 1 } } g _ { 1 } ( a _ { 1 } ) + \langle a _ { 1 } , b \rangle \big ) + \big ( \operatorname* { i n f } _ { a _ { 2 } } g _ { 2 } ( a _ { 2 } ) - \langle a _ { 2 } , b \rangle \big ) } \\ { \displaystyle \quad = \operatorname* { s u p } _ { b } h _ { 1 } ( b ) + h _ { 2 } ( - b ) . } \end{array}
$$

The preceding calculation is not entirely rigorous, since it swaps the inf and sup without proper justification. Nevertheless, it conveys the main idea. There also exist simple and fully rigorous proofs of Theorem 2.9 using the separating hyperplane theorem.

Remark 2.11. It is clear that $a , b$ achieve equality in (6) (i.e. are a “saddle $\mathrm { p o i n t } ^ { \prime \prime } )$ if they are dual for $g _ { 1 } , h _ { 1 }$ and $a , - b$ are dual for $g _ { 2 } , h _ { 2 }$ . Indeed, by the Fenchel-Young inequality, we have

$$
g _ { 1 } ( a ) + g _ { 2 } ( a ) \geqslant \left( h _ { 1 } ( b ) - \langle a , b \rangle \right) + \left( h _ { 2 } ( - b ) + \langle a , b \rangle \right) = h _ { 1 } ( b ) + h _ { 2 } ( - b ) ,
$$

with equality if the two duality conditions hold (see Definition 2.8).

Remark 2.12. The finiteness assumption in our statement of Theorem 2.9 is actually far stronger than necessary, not only for equating the primal value $p = \operatorname* { i n f } _ { a } g _ { 1 } ( a ) + g _ { 2 } ( a )$ with dual value $d = \operatorname* { s u p } _ { b } h _ { 1 } ( b ) + h _ { 2 } ( - b )$ , but also for proving that these values are attained when finite. In fact, to prove that there exists $b ^ { * } \in \mathbb { R } ^ { n }$ with $h _ { 1 } ( b ^ { * } ) + h _ { 2 } ( - b ^ { * } ) = p = d .$ , we only require there to be a non-vertical hyperplane that weakly separates the disjoint convex sets $\{ ( a , t ) \mid t > g _ { 1 } ( a ) \}$ and $\{ ( a , t ) \mid t < p - g _ { 2 } ( a ) \}$ . A weakly separating hyperplane between two disjoint convex sets always exists by the separating hyperplane theorem, and assuming that $g _ { 1 }$ or $g _ { 2 }$ is finite everywhere is just one of many convenient ways to force this hyperplane to be non-vertical.

## 2.2 Hinge Loss Rounding

We now discuss the known relationship between 0-1 loss, the binary hinge loss $\ell ( y , a ) = ( 1 - a y ) _ { + }$ and the Crammer-Singer multiclass hinge loss [CS01]

$$
\ell _ { \mathrm { C S } } ( y , a ) = ( 1 - a _ { y } + \underset { j \neq y } { \operatorname* { m a x } } a _ { j } ) _ { + } .
$$

Remark 2.13 (Hinge Loss Rounding [KKMS08, Section 3.2]). Given $a \in \mathbb { R }$ , let $\tilde { a } = \mathrm { p r o j } _ { [ - 1 , + 1 ] } ( a )$ be its projection onto the interval $[ - 1 , + 1 ]$ . If we sample a random threshold $\mathbf { \Delta } t \sim [ - 1 , + 1 ]$ , then

$$
\ell _ { \mathrm { h i n g e } } ( y , a ) \geqslant \ell _ { \mathrm { h i n g e } } ( y , \tilde { a } ) = | y - \tilde { a } | = 2 \cdot \mathbb { E } _ { t } [ \ell _ { 0 - 1 } ( y , \mathrm { s i g n } ( a - t ) ) ] .
$$

Thus, a 2δ-upper bound on hinge loss implies a δ-upper bound on 0-1 loss.

Remark 2.14 (Crammer-Singer Rounding [CS01]). Given $a \in \mathbb { R } ^ { k }$ , let $j ^ { \star } \in$ argmax<sub>j</sub> a<sub>j</sub>. Then,

$$
\mathbf { 1 } [ y \neq j ^ { \star } ] \leqslant \ell _ { \mathrm { C S } } ( y , a ) .
$$

Thus, a δ-upper bound on Crammer-Singer hinge loss implies a δ-upper bound on 0-1 loss.

## 2.3 Multicalibration

Here, we state the precise forms of computational indistinguishability that we require for the proof of our universal multiclass hardcore lemma. There are two definitions that we need: calibration [Daw82] and multiaccuracy [HKRR18, KNRW18].

We begin with the definition of weight-restricted calibration. This condition requires the error of the model $p : \mathcal { X }  \Delta _ { k }$ to be approximately uncorrelated with the model’s predictions themselves, even after post-processing by some appropriately bounded weight functions in a prespecified collection W. We remark that the notational similarity of W (the weight function class) and W (our noisy channel) is not intentional but nevertheless convenient since the former will be largely determined by the latter.

Definition 2.15 (Weighted Calibration [Daw82, GKSZ22]). Given finite X and $\mathcal { V } = [ k ]$ , a distribution over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , and ${ \mathcal { W } } \subseteq \{ \Delta _ { k } \to \mathbb { R } ^ { k } \}$ , we say that $p : \mathcal { X }  \Delta _ { k }$ is $( \mathcal { W } , \varepsilon )$ -calibrated if for all $w \in \mathcal W$ 2

$$
\begin{array} { r } { \big | \mathbb { E } [ \langle w ( p ( { \pmb x } ) ) , { \pmb y } - p ( { \pmb x } ) \rangle ] \big | \leqslant \varepsilon . } \end{array}
$$

Next, we define multiaccuracy and its weighted variant. Intuitively, multiaccuracy requires the error of $p$ to be approximately uncorrelated with each function in a prespecified collection $\mathcal { C } .$ . The weighted version also includes a factor depending on the model’s predictions themselves, as in the definition of a calibration weight function.

Definition 2.16 (Multiaccuracy [HKRR18, KNRW18]). Given finite X and $\mathcal { V } = [ k ]$ , a distribution over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , and a collection ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , we say that $p : \mathcal { X }  \Delta _ { k }$ is $( \mathcal { C } , \varepsilon )$ -multiaccurate if for all $c \in { \mathcal { C } }$

$$
\begin{array} { r } { \left| \mathbb { E } [ \langle c ( { \pmb x } ) , { \pmb y } - p ( { \pmb x } ) \rangle ] \right| \leqslant \varepsilon . } \end{array}
$$

Definition 2.17 (Weighted Multiaccuracy [GKSZ22, CGKR25]). In the setting of Definition 2.16, given a weight function $w : \Delta _ { k }  \mathbb { R } ^ { k \times k }$ , we say that p is $( \mathcal { C } , \varepsilon )$ -multiaccurate with respect to w if for all $c \in { \mathcal { C } }$

$$
\begin{array} { r } { \big | \mathbb { E } [ \langle w ( p ( \pmb { \mathscr { x } } ) ) c ( \pmb { \mathscr { x } } ) , \pmb { \mathscr { y } } - p ( \pmb { \mathscr { x } } ) \rangle ] \big | \leqslant \varepsilon . } \end{array}
$$

Note that standard $( \mathcal { C } , \varepsilon )$ -multiaccuracy corresponds to the case that $w ( \cdot ) = I _ { k \times k }$

It is by now well known that for any class $\mathcal { C } \mathrm { ~ ( e . g }$ . a Boolean circuit complexity class), one can construct a multiaccurate predictor $p$ of “complexity” not much greater than that of C itself. By this, we mean that $p ( x )$ can be evaluated by performing simple arithmetic operations on the values $c _ { 1 } ( x ) , \ldots , c _ { m } ( x )$ , where $m = O ( \log ( k ) / \varepsilon ^ { 2 } )$ and the functions $c _ { 1 } , \ldots , c _ { m }$ all belong to the class C. With $k = 2$ , this has been known since the work of [TTV09], and the fact that the same bound holds even if one incorporates weight functions depending on the model itself in complex ways has been known since [HKRR18, DKR<sup>+</sup>21].

Theorem 2.18 (Outcome Indistinguishability via Multiplicative Updates [DLLT23, Example $5 . 6 ] )$ For all finite X and $\ y = [ k ]$ distributions over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , collections of functions $\mathcal { F } \subseteq$ $\{ \Delta _ { k } \times \mathcal { X }  [ - 1 , + 1 ] ^ { k } \}$ , and $\varepsilon > 0$ , there exists $p : \mathcal { X }  \Delta _ { k }$ with the following properties:

• (Indistinguishability) For all functions $f \in { \mathcal { F } }$

$$
\left| \mathbb { E } [ \langle f ( p ( \pmb { x } ) , \pmb { x } ) , \pmb { y } - p ( \pmb { x } ) \rangle ] \right| \leqslant O ( \varepsilon ) .
$$

• (Complexity Bound) The model p has the form $p _ { m }$ , where we define $p _ { 0 } ( x ) = \vec { 1 } / k$ and

$$
p _ { j } ( x ) \propto \exp \Bigl ( \varepsilon \sum _ { i = 1 } ^ { j } f _ { i } ( p _ { i - 1 } ( x ) , x ) \Bigr ) ,
$$

normalized so that $\begin{array} { r } { \sum _ { y } p _ { j } ( x ) _ { y } = 1 } \end{array}$ , for some $m = O ( \log ( k ) / \varepsilon ^ { 2 } )$ functions $f _ { 1 } , \ldots , f _ { m } \in \pm \mathcal { F }$

## 3 A General Framework

In this section, we present a very general framework, captured by Theorem 3.3, which we will instantiate in later sections to prove our main results. We will also see in later sections that this framework can be used to derive several of the prior results on pseudoentropy and the MaxEnt algorithm that we discussed in the introduction.

Before stating the result, let us recall from Section 2.1 the precise statement of our convexconcave conjugacy requirement on the loss function g and fidelity function h:

$$
\begin{array} { r l } & { g ( y , a ) = \underset { b \in \mathbb { R } ^ { k } } { \operatorname* { s u p } } h ( y , b ) - \langle a , b \rangle , } \\ & { h ( y , b ) = \underset { a \in \mathbb { R } ^ { k } } { \operatorname* { i n f } } g ( y , a ) + \langle a , b \rangle . } \end{array}\tag{7}
$$

Since a supremum of linear functions is convex, and since an infimum of linear functions is concave, these conditions imply that $g ( y , \cdot )$ and $h ( y , \cdot )$ are convex and concave, respectively. Later, we will also require the efective domain of h in its second input to be contained in $\Delta _ { k }$ , but we do not yet need to impose that requirement.

Our general result will be stated not only in terms of $g , h ,$ , but also in terms of a pair of regularizer or constraint functionals R, S. These take as input a function from $\mathcal { X } \times \mathcal { Y } \mathrm { ~ t o ~ } \mathbb { R } ^ { k }$ , and they output a value in R. Intuitively, we will use these functions to impose penalties like the $\varepsilon \lVert \lambda \rVert _ { 1 }$ penalty appearing in (1) and Theorem 1.9, as well as the indistinguishability constraints appearing in (2) and Theorem 1.9. We require the functionals R and S to satisfy a conjugacy relationship similar to the one satisfied by g and h:

$$
\begin{array} { r l } & { R ( a ) = \underset { b : \mathcal { X } \times \mathcal { Y } \to \mathbb { R } ^ { k } } { \operatorname* { s u p } } S ( b ) - \mathbb { E } \big [ \langle a ( \pmb { x } , \pmb { y } ) , b ( \pmb { x } , \pmb { y } ) \rangle \big ] , } \\ & { S ( b ) = \underset { a : \mathcal { X } \times \mathcal { Y } \to \mathbb { R } ^ { k } } { \operatorname* { i n f } } R ( a ) + \mathbb { E } \big [ \langle a ( \pmb { x } , \pmb { y } ) , b ( \pmb { x } , \pmb { y } ) \rangle \big ] . } \end{array}\tag{8}
$$

There are a couple key diferences between the conjugacy relationships in (7) and (8). The first diference is that (7) optimizes over k-dimensional vectors $a , b \in \mathbb { R } ^ { k }$ , but (8) optimizes over entire functions $a , b : \mathcal { X } \times \mathcal { Y }  \mathbb { R } ^ { k }$ , which can be viewed as |X||Y|k-dimensional vectors. The second diference is that (7) uses the standard inner product in $\mathbb { R } ^ { k }$ , but (8) uses an inner product that crucially depends on the given distribution over $( { \pmb x } , { \pmb y } )$ pairs. We now give two examples of $R , S$ pairs of interest. The first is the one that we saw implicitly in the introduction.

Definition 3.1 (Indistinguishability Constraint Functionals). Given finite X and Y, a distribution over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ a finite collection ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , and $\varepsilon \geqslant 0$ , the indistinguishability constraint functionals R, S are defined as follows. For a function $b : \mathcal { X } \times \mathcal { Y }  \mathbb { R } ^ { k }$ , the concave functional S(b) enforces the constraint that b is approximately uncorrelated with each $c \in { \mathcal { C } }$

$$
S ( b ) = { \left\{ \begin{array} { l l } { 0 } & { { \mathrm { i f ~ } } | \mathbb { E } [ \langle c ( { \pmb x } ) , b ( { \pmb x } , { \pmb y } ) \rangle ] | \leqslant \varepsilon { \mathrm { ~ f o r ~ a l l ~ } } c \in { \mathcal C } , } \\ { - \infty } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

Next, for $a : \mathcal { X } \times \mathcal { Y }  \mathbb { R } ^ { k }$ , the convex functional $R ( a )$ checks that $a \in \operatorname { s p a n } ( \mathcal { C } )$ (in particular, a must ignore its y-input), and then applies an $\ell _ { 1 }$ penalty to the coeficients λ used to form a:

$$
R ( a ) = \operatorname* { i n f } _ { \lambda \in \mathbb { R } ^ { c } } \varepsilon \| \lambda \| _ { 1 } \ \mathrm { s u c h } \ \mathrm { t h a t } \ \operatorname* { P r } \Big [ a ( { \pmb x } , { \pmb y } ) = \sum _ { i } \lambda _ { i } c _ { i } ( { \pmb x } ) \Big ] = 1 .
$$

Note that $R ( a ) = + \infty$ if a $\not \in \operatorname { s p a n } ( \mathcal { C } )$ . The types of these functionals are $R , S : \{ \mathcal { X } \times \mathcal { Y }  \mathbb { R } ^ { k } \}  \overline { { \mathbb { R } } }$

Definition 3.2 (Indistinguishability Penalty Functionals). Given finite X and Y, a distribution over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ and a convex compact collection $\overline { { \mathcal { C } } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ ,<sup>15</sup> the indistinguishability penalty functionals R, S are defined as follows. For a function $b : \mathcal { X } \times \mathcal { Y } \ \to \ \mathbb { R } ^ { k }$ , the concave functional S(−b) penalizes b for being too positively correlated with any $c \in { \overline { { \mathcal { C } } } } .$

$$
S ( - b ) = - \operatorname* { m a x } _ { c \in \overline { { \mathcal { C } } } } \mathbb { E } [ \langle c ( \pmb { x } ) , b ( \pmb { x } , \pmb { y } ) \rangle ] .
$$

Note that $S ( - b )$ is roughly 0 when b is uncorrelated with C but takes on a large negative value when some b is positively correlated with some $c \in \overline { { \mathcal { C } } } . ^ { 1 6 }$ We also write “max” instead of “sup” since

$\overline { { \mathcal { C } } }$ is assumed to be compact. Next, for $a : \mathcal { X } \times \mathcal { Y }  \mathbb { R } ^ { k }$ , the convex functional R simply checks that $a \in \operatorname { s p a n } ( \mathcal { C } )$ (in particular, a must ignore its y-input):

$$
R ( a ) = { \left\{ \begin{array} { l l } { 0 } & { { \mathrm { i f ~ } } \operatorname* { P r } [ a ( { \boldsymbol { x } } , { \boldsymbol { y } } ) = c ( { \boldsymbol { x } } ) ] = 1 { \mathrm { ~ f o r ~ s o m e ~ } } c \in { \overline { { \mathcal { C } } } } , } \\ { + \infty } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

This is closely related to the convex indicator $I ( \cdot )$ function defined in Example 2.3. Finally, as in Definition 3.1, the types of these functionals are $R , S : \{ \mathcal { X } \times \mathcal { Y }  \mathbb { R } ^ { k } \}  \overline { { \mathbb { R } } } .$

Our general theorem, which we state and prove below, asserts that for any suitable $g , h , R , S ,$ minimizing g in expectation over functions a that have been penalized (or constrained) by $R ( a )$ is equivalent to maximizing h in expectation over functions b that have been penalized (or constrained) by $S ( - b )$ . Despite the seemingly abstract form of the theorem, involving six functions $a , b , g , h , R , S .$ the proof is actually a surprisingly straightforward application of Fenchel duality (Theorem 2.9). The theorem statement will inherit some mild technical assumptions from our stated version of Fenchel duality, which, as we explain in Remark 2.12, can easily be relaxed.

In subsequent sections, we will see even more examples of pairs $g , h$ and pairs $R , S$ that convey the significance of Theorem 3.3. For example, in Section 4, we apply the general result to prove our main result on channel fidelity, Theorem 1.9, and its soft penalty variant. In Section 6, we use it to prove our universal hardcore lemma, Theorem 1.20. In Section 7, we use it to recover MaxEnt duality, Theorem 1.22, as a special case. Finally, in Section 5, we use it to prove several additional interesting and concrete multiclass generalizations of the hardcore lemma.

Theorem 3.3 (Six Functions). Fix finite sets X, Y, a distribution D over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , functions $g , h : \mathcal { V } \times \mathbb { R } ^ { k } \to \overline { { \mathbb { R } } }$ , and functionals $R , S : ( \mathcal { X } \times \mathcal { Y }  \mathbb { R } ^ { k } )  \overline { { \mathbb { R } } }$ . Then, under the three technical conditions stated below, strong duality holds:

$$
\operatorname* { i n f } _ { a \colon \mathcal { X } \times \mathcal { Y } \to \mathbb { R } ^ { k } } \mathbb { E } [ g ( \pmb { y } , a ( \pmb { x } , \pmb { y } ) ) ] + R ( a ) = \operatorname* { s u p } _ { b \colon \mathcal { X } \times \mathcal { Y } \to \mathbb { R } ^ { k } } \mathbb { E } [ h ( \pmb { y } , b ( \pmb { x } , \pmb { y } ) ) ] + S ( - b ) .\tag{9}
$$

The three technical conditions $a r e ! ^ { . 1 7 }$

(a) g and h are two-input convex-concave conjugates as in Definition 2.7.

(b) R and S are one-input convex-concave conjugates as in Definition 2.6 under

$$
\begin{array} { r } { \langle a , b \rangle _ { \mathcal { D } } = \mathbb { E } _ { \mathcal { D } } [ \langle a ( \pmb { x } , \pmb { y } ) , b ( \pmb { x } , \pmb { y } ) \rangle ] . } \end{array}
$$

(c) At least one of the functions g, h, R, or S is finite everywhere $( i . e .$ never outputs ±∞).

Also, if the finite function among the four was one of $g , R$ , then the supremum on the right-hand side of equation (9) is attained if finite. Similarly, if the finite function among the four was one of $h , S$ , then the infimum on the left-hand side of equation (9) is attained if finite.

Proof. In order to apply Fenchel duality (Theorem 2.9), we must define suitable convex-concave conjugate pairs $g _ { 1 } , h _ { 1 }$ and $g _ { 2 } , h _ { 2 }$ on the space $\{ \mathcal { X } \times \mathcal { Y } \xrightarrow { } \mathbb { R } ^ { k } \} . ^ { 1 8 }$ We henceforth refer to this vector space as V, which is finite-dimensional. Indeed, its dimension is precisely dim $\nu = k n$ , where the

integer $n \leqslant | \mathcal { X } | | \mathcal { Y } |$ is the number of pairs $( x , y ) \in \mathcal { X } \times \mathcal { Y }$ that occur with strictly positive probability under D. With this in mind, define

$$
\begin{array} { r } { g _ { 1 } ( a ) = \mathbb { E } [ g ( \pmb { y } , a ( \pmb { x } , \pmb { y } ) ) ] , } \\ { h _ { 1 } ( b ) = \mathbb { E } [ h ( \pmb { y } , b ( \pmb { x } , \pmb { y } ) ) ] . } \end{array}
$$

It is clear that these are closed proper convex and closed proper concave functions on $\nu ,$ respectively. It is also easy to check that these are conjugates, as per Definition 2.6, because for $a , b \in \mathcal { V } ,$

$$
\begin{array} { r l } & { \underset { a \in \mathcal { V } } { \operatorname* { i n f } } \ g _ { 1 } ( a ) + \langle a , b \rangle _ { \mathcal { D } } = \underset { a \in \mathcal { V } } { \operatorname* { i n f } } \ \mathbb { E } [ g ( { \pmb y } , a ( { \pmb x } , { \pmb y } ) ) + \langle a ( { \pmb x } , { \pmb y } ) , b ( { \pmb x } , { \pmb y } ) \rangle ] } \\ & { \quad \quad \quad = \mathbb { E } [ \underset { t \in \mathbb { R } ^ { k } } { \operatorname* { i n f } } \ g ( { \pmb y } , t ) + \langle t , b ( { \pmb x } , { \pmb y } ) \rangle ] } \\ & { \quad \quad \quad = \mathbb { E } [ h ( { \pmb y } , b ( { \pmb x } , { \pmb y } ) ) ] . } \end{array}
$$

Next, we define $g _ { 2 } = R$ and $h _ { 2 } = S$ . By assumption, these are also a convex-concave conjugate pair of R-valued functions on V. Finally, we conclude the proof by applying Theorem 2.9 to the pairs $g _ { 1 } , h _ { 1 }$ and $g _ { 2 } , h _ { 2 }$ defined above. □

## 4 Hardness of Learning and Masking Channel Fidelity

In this section, we prove Theorem 1.9, restated below for convenience, which is our precise characterization of average-case hardness in terms of the fidelity of masking channels. Our proof strategy will be to carefully instantiate our general framework (Theorem 3.3) with an appropriate choice of the functions R and S. The proof relies on various helper lemmas that calculate convex-concave conjugate pairs; we generally defer such lemmas to Appendix A.

Theorem 1.9 (Hardness vs Fidelity). For all finite $x , y = [ k ]$ , distributions D over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , convex-concave conjugates $g : \mathcal { V } \times \mathbb { R } ^ { k } \to \mathbb { R } , h : \mathcal { V } \times \Delta _ { k } \to \overline { { \mathbb { R } } }$ , and $\varepsilon \geqslant 0 , ^ { 1 9 }$

$$
\operatorname* { i n f } _ { \lambda \in \mathbb { R } ^ { C } } \mathbb { E } [ g ( \pmb { y } , \pi ( c _ { \lambda } ( \pmb { x } ) ) ) ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { W : \mathscr { X }  \Delta _ { k } ^ { k } } \mathbb { E } [ h ( \pmb { y } , W ( \pmb { x } ) \pmb { y } ) ] \quad s . t . \quad W \mathrm { ~ } i s \mathrm { ~ } ( \mathscr { C } , \varepsilon ) - m a s k i n g .
$$

Proof. Given the functions $g , h$ from the theorem statement, define

$$
g _ { 1 } ( y , a ) = g ( y , \pi ( a ) )
$$

and

$$
h _ { 1 } ( y , b ) = { \left\{ \begin{array} { l l } { h ( y , z ) } & { { \mathrm { i f ~ } } b = z - { \vec { 1 } } / k { \mathrm { ~ f o r ~ s o m e ~ } } z \in \Delta _ { k } , } \\ { - \infty } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

These are convex-concave conjugates by Lemma A.1. Next, let R, S be the indistinguishability constraint functionals (Definition 3.1), which are convex-concave conjugates by Lemma A.2. Since h has a bounded efective domain, its conjugate $^ { g , }$ which is defined as a supremum over this domain, is always finite. Hence, so is $g _ { 1 }$ . Therefore, applying Theorem 3.3 to $g _ { 1 } , h _ { 1 } , R , S$ yields

$$
\begin{array}{c} \operatorname* { i n f } _ { \boldsymbol { x } \in \mathbb { R } ^ { c } } \mathbb { E } [ g ( \boldsymbol { y } , \pi ( c _ { \lambda } ( \boldsymbol { x } ) ) ) ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { \boldsymbol { z } : \mathcal { X } \times \mathcal { Y }  \Delta _ { k } } \mathbb { E } [ h ( \boldsymbol { y } , \boldsymbol { z } ( \boldsymbol { x } , \boldsymbol { y } ) ) ]  \\ { \mathrm { s . t . } \forall c \in \mathcal { C } , \mathrm { ~ } \big | \mathbb { E } [ \langle c ( \boldsymbol { x } ) , \boldsymbol { z } ( \boldsymbol { x } , \boldsymbol { y } ) - \vec { 1 } / k \rangle ] \big | \leqslant \varepsilon . \qquad } \end{array}
$$

Next, we make a simple notational change. Specifically, we define the noisy channel $W : \mathcal { X }  \Delta _ { k } ^ { k }$ such that the $j ^ { \mathrm { t h } }$ column of $W ( x )$ is $z ( x , e _ { j } )$ . Then, the optimization problem on the right-hand side of the above equation can be rewritten as

$$
\operatorname* { m a x } _ { W : \mathcal { X } \to \Delta _ { k } ^ { k } } \mathbb { E } [ h ( \pmb { y } , W ( \pmb { x } ) \pmb { y } ) ] \quad \mathrm { s . t . } \quad \forall c \in \mathcal { C } , \ \big | \mathbb { E } [ \langle c ( \pmb { x } ) , W ( \pmb { x } ) \pmb { y } - \vec { 1 } / k \rangle ] \big | \leqslant \varepsilon .
$$

Of course, the constraint on $W$ is precisely the requirement that $W ( { \pmb x } ) { \pmb y }$ is $( \mathcal { C } , \varepsilon )$ -pseudorandom, as per Definition 1.3. This proves the main part of Theorem 1.9.

For the second part, suppose $g$ is bounded below by the value $v \in \mathbb { R }$ on centered actions. If $\varepsilon > 0$ , then it is never beneficial to use $\lambda$ whose $\ell _ { 1 }$ norm exceeds

$$
r = { \frac { \mathbb { E } [ g ( { \pmb y } , 0 ) ] - v } { \varepsilon } } .
$$

Indeed, any such λ is strictly outperformed by $\lambda = 0 .$ . Thus, we can restrict the search space to the $\ell _ { 1 }$ ball of radius $r ,$ which is compact since $\mathcal { C }$ is finite. Since the left-hand side minimizes a lower semicontinuous function over a compact set, it attains this minimum. □

Next, we prove a slightly diferent version of the preceding theorem which uses a soft hardcore penalty instead of the hardcore constraint. This version is appealing for its more flexible treatment of the class C and fewer hyperparameters.

Theorem 4.1 (Hardness vs Fidelity, Soft Penalty). For all finite X and $\mathcal { V } = [ k ]$ , distributions over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , compact convex ${ \overline { { \mathcal { C } } } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , and convex-concave conjugates $g : \mathcal { V } \times \mathbb { R } ^ { k } $ R and $h : \mathcal { V } \times \Delta _ { k } \to \overline { { \mathbb { R } } }$

$$
\operatorname* { m i n } _ { c \in \mathcal { \overline { { C } } } } \mathbb { E } [ g ( \pmb { y } , \pi ( c ( \pmb { x } ) ) ) ] = \operatorname* { m a x } _ { W : \mathcal { X }  \Delta _ { k } ^ { k } } \Big ( \mathbb { E } [ h ( \pmb { y } , W ( \pmb { x } ) \pmb { y } ) ] - \operatorname* { m a x } _ { c \in \mathcal { \overline { { C } } } } \mathbb { E } [ \langle c ( \pmb { x } ) , W ( \pmb { x } ) \pmb { y } - \vec { 1 } / k \rangle ] \Big ) .
$$

Proof. Define $g _ { 1 } , h _ { 1 }$ in terms of $g , h$ as in the preceding proof of Theorem 1.9. Let $R , S$ be the hardcore penalty functionals (Definition 3.2), which are convex-concave conjugates by Lemma $\mathrm { A . 3 }$ Finally, we note that the function $g _ { 1 }$ is always finite by assumption, which lets us apply our general duality framework (Theorem 3.3) to $g _ { 1 } , h _ { 1 } , R , S$ . This concludes the proof. □

## 5 Multiclass Hardcore Lemmas

In this section, we state and prove our various instantiations of Theorem 1.9 and investigate multiclass hardcore certificates via surrogate losses for $\ell _ { 0 - 1 }$ , as discussed in Section 1.2. We begin with our result in the binary setting, whose statement we recall for convenience.

Theorem 1.10 (Hinge, $\ell _ { 1 }$ , and $\ell _ { p }$ Hardcore Lemmas). Fix a finite set X , a distribution over pairs $( \pmb { x } , \pmb { y } ) \in \mathcal { X } \times \{ \pmb { \pmb { \bot } } 1 \}$ , a finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } \}$ , and $\varepsilon > 0$ . Then, using the abbreviation ${ \pmb \mu } = \mu ( { \pmb x } , { \pmb y } )$

$$
\operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { c } } \mathbb { E } [ ( 1 - c _ { \lambda } ( \boldsymbol { x } ) \boldsymbol { y } ) _ { + } ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { \boldsymbol { \mu } : \mathcal { X } \times \mathcal { Y }  [ 0 , 1 ] } \mathbb { E } [ \boldsymbol { \mu } ] \quad s . t . \quad \forall c \in \mathcal { C } , ~ \big | \mathbb { E } [ \mu c ( \boldsymbol { x } ) \boldsymbol { y } ] \big | \leqslant \varepsilon ,\tag{3}
$$

$$
\operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { \mathcal { C } } } \mathbb { E } [ | y - c _ { \lambda } ( \boldsymbol { x } ) | ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { \substack { \mu : \mathcal { X } \times \mathcal { Y }  [ - 1 , + 1 ] } } \mathbb { E } [ \mu ] \quad s . t . \quad \forall c \in \mathcal { C } , \ | \mathbb { E } [ \mu c ( \boldsymbol { x } ) y ] | \leqslant \varepsilon ,\tag{4}
$$

$$
\operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { c } } \mathbb { E } \left[ \frac { 1 } { p } | y - c _ { \lambda } ( x ) | ^ { p } \right] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { \mu : \mathcal { X } \times \mathcal { Y } \to \mathbb { R } } \mathbb { E } \left[ \mu - \frac { 1 } { q } | \mu | ^ { q } \right] \quad s . t . \quad \forall c \in \mathcal { C } , \ \left| \mathbb { E } [ \mu c ( x ) y ] \right| \leqslant \varepsilon ,\tag{5}
$$

where $p , q > 1$ are any real numbers satisfying $1 / p + 1 / q = 1$

Proof. Given Theorems 1.9 and 3.3, the current proof is mostly a matter of defining appropriate loss-fidelity pairs $g , h$ and verifying that they are indeed convex-concave conjugates:

• First, consider the fidelity function

$$
h ( y , z ) = { \left\{ \begin{array} { l l } { \mu } & { { \mathrm { i f ~ } } z = \mu y { \mathrm { ~ f o r ~ s o m e ~ } } \mu \in [ 0 , 1 ] , } \\ { - \infty } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

By Definition 2.6, its conjugate is

$$
\operatorname* { s u p } _ { b \in \mathbb { R } } h ( y , b ) - a b = \operatorname* { s u p } _ { \mu \in [ 0 , 1 ] } \mu - a \cdot ( \mu y ) = \operatorname* { m a x } ( 0 , 1 - a y ) ,
$$

where we have used the fact that an afine function of $\mu \in [ 0 , 1 ]$ is maximized at either $\mu = 0$ or $\mu = 1$ . This is precisely hinge loss.

• Next, consider the fidelity function

$$
h ( y , z ) = { \left\{ \begin{array} { l l } { \mu } & { { \mathrm { i f ~ } } z = \mu y { \mathrm { ~ f o r ~ s o m e ~ } } \mu \in [ - 1 , + 1 ] , } \\ { - \infty } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

By Definition 2.6, its conjugate is

$$
\operatorname* { s u p } _ { b \in \mathbb { R } } h ( y , b ) - a b = \operatorname* { s u p } _ { \mu \in [ - 1 , + 1 ] } \mu - a \cdot ( \mu y ) = | 1 - a y | .
$$

Since $y \in \{ \pm 1 \}$ , this is precisely $\ell _ { 1 }$ loss.

• Finally, consider the fidelity function

$$
\begin{array} { r } { h ( y , z ) = \bigg \{ \mu - \frac { 1 } { q } | \mu | ^ { q } \quad \mathrm { i f } \ z = \mu y \ \mathrm { f o r \ s o m e } \ \mu \in \mathbb { R } . } \end{array}
$$

We show in Lemma A.4 that the conjugate of h is precisely $\ell _ { p }$ loss.

To conclude the proof, we invoke Theorem 1.9 with $k = 2$ . (Technically, we must map distributions in $\Delta _ { 2 }$ , which is a line segment, to the interval $[ - 1 , + 1 ]$ , but we omit the details of this very simple change of variables, which is essentially just a change of notation. Also, for the third example, where $\mu$ is not bounded in the interval [−1, +1], we instead apply Theorem 3.3 with $k = 2$ and the indistinguishability constraint functionals from Definition 3.1.) □

Next, we prove our results regarding the multiclass hardness certificates associated with multiclass hinge losses $\ell _ { \mathrm { w e a k } } , \ell _ { \mathrm { s t r o n g } } .$ , and $\ell _ { \mathrm { C S } }$ . We recall the definitions of these loss functions and the statements of the relevant theorems below. For the meaning of these losses and the interpretations of their corresponding hardness certificates, see Section 1.2.

Definition 5.1 (Multiclass Hinge Losses). Given $a \in \mathbb { R } ^ { k }$ with mean 0, let

$$
\begin{array} { c } { \ell _ { \mathrm { w e a k } } ( e _ { i } , a ) = ( 1 - a _ { i } ) _ { + } , } \\ { \ell _ { \mathrm { s t r o n g } } ( e _ { i } , a ) = \displaystyle \operatorname* { m a x } _ { j \in [ k ] } ( e _ { i } - a ) _ { j } , } \\ { \ell _ { \mathrm { C S } } ( e _ { i } , a ) = ( 1 - a _ { i } + \displaystyle \operatorname* { m a x } _ { j \neq i } a _ { j } ) _ { + } . } \end{array}
$$

We extend these formulas to all $a \in \mathbb { R } ^ { k }$ by first centering a to have mean 0. The third of these hinge-like loss functions, the Crammer-Singer loss, was introduced by [CS01].

Theorem 1.11 (Weak Hinge Loss vs Uniform Noise). For all finite X and $\mathcal { V } = [ k ]$ , distributions over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , and $\varepsilon > 0$

$$
\begin{array}{c} \operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { c } } \mathbb { E } [ \ell _ { \mathrm { w e a k } } ( y , c _ { \lambda } ( x ) ) ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { \mu : \mathcal { X } \times \mathcal { Y } \to [ 0 , 1 ] } \mathbb { E } [ \mu ( x , y ) ]  \\ { s . t . \forall c \in \mathcal { C } , \ \big | \mathbb { E } [ \mu ( x , y ) \langle c ( x ) , y - \vec { 1 } / k \rangle ] \big | \leqslant \varepsilon . } \end{array}
$$

Proof. We apply Theorem 1.9 with the fidelity function from Example 1.8, which constrains the masking channel to mix the labels with uniform noise:

$$
h ( y , z ) = { \left\{ \begin{array} { l l } { \mu } & { { \mathrm { i f ~ } } z = \mu \cdot y + ( 1 - \mu ) \cdot { \vec { 1 } } / k { \mathrm { ~ f o r ~ s o m e ~ } } \mu \in [ 0 , 1 ] , } \\ { - \infty } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

Its conjugate is the loss function

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { z } h ( y , z ) - \langle a , z \rangle = \operatorname* { s u p } _ { \mu \in [ 0 , 1 ] } \mu - \langle a , \mu \cdot y + ( 1 - \mu ) \cdot \vec { 1 } / k \rangle } \\ { \displaystyle \quad = ( 1 - \langle a , y - \vec { 1 } / k \rangle ) _ { + } - \langle a , \vec { 1 } / k \rangle , } \end{array}
$$

which agrees with $\ell _ { \mathrm { w e a k } } ( y , a )$ for all mean-0 inputs $a \in \mathbb { R } ^ { k }$

Theorem 1.12 (Strong Hinge Loss vs Arbitrary Noise). For all finite X and $\mathcal { V } = [ k ]$ , distributions over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , and $\varepsilon > 0$ ,

$$
\begin{array}{c} \operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { \mathcal { C } } } \mathbb { E } [ \ell _ { \mathrm { s t r o n g } } ( y , \pi ( c _ { \lambda } ( x ) ) ) ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { W : \mathcal { X } \to \Delta _ { k } ^ { k } } \mathbb { E } [ \langle y , W ( x ) y \rangle ]  \\ { s . t . \forall c \in \mathcal { C } , \ \big | \mathbb { E } [ \langle c ( x ) , W ( x ) y - \vec { 1 } / k \rangle ] \big | \leqslant \varepsilon . } \end{array}
$$

Proof. We apply Theorem 1.9 with the fidelity function from Example 1.6:

$$
h ( y , z ) = { \left\{ \begin{array} { l l } { \langle y , z \rangle } & { { \mathrm { i f ~ } } z \in \Delta _ { k } , } \\ { - \infty } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

Its conjugate is the loss function

$$
\begin{array} { c } { \displaystyle \operatorname* { s u p } _ { z } h ( y , z ) - \langle a , z \rangle = \displaystyle \operatorname* { s u p } _ { z \in \Delta _ { k } } \langle y , z \rangle - \langle a , z \rangle } \\ { \displaystyle = \operatorname* { m a x } _ { j \in [ k ] } ( y - a ) _ { j } , } \end{array}
$$

which agrees with $\ell _ { \mathrm { s t r o n g } } ( y , a )$ for mean-0 actions a.

Theorem 1.13 (Crammer-Singer Hinge Loss vs Maximum TV Distance). For all finite X and $\mathcal { V } = [ k ]$ , distributions over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , and $\varepsilon > 0$

$$
\begin{array}{c} \operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { \mathcal { C } } } \mathbb { E } [ \ell _ { \mathrm { C S } } ( y , c _ { \lambda } ( x ) ) ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { p : \mathcal { X } \times \mathcal { Y }  \Delta _ { k } } \mathbb { E } [ d _ { \mathrm { T V } } ( y , p ( x , y ) ) ]  \\ { s . t . \forall c \in \mathcal { C } , \big | \mathbb { E } [ \langle c ( x ) , y - p ( x , y ) \rangle ] \big | \leqslant \varepsilon . } \end{array}
$$

Proof. We apply Theorem 3.3 with the indistinguishability constraints in Definition 3.1, which are conjugates by Lemma A.2, as well as the function

$$
h ( y , b ) = \left\{ \begin{array} { l l } { { 1 - \langle y , z \rangle } } & { { \mathrm { i f ~ } b = y - z \ \mathrm { f o r \ s o m e } \ z \in \Delta _ { k } , } } \\ { { - \infty } } & { { \mathrm { o t h e r w i s e } . } } \end{array} \right.
$$

Its conjugate $g ( y , a )$ is the function

$$
\begin{array} { r l } & { \underset { b } { \operatorname* { s u p } } h ( y , b ) - \langle a , b \rangle = \underset { z \in \Delta _ { k } } { \operatorname* { s u p } } 1 - \langle y , z \rangle - \langle a , y - z \rangle } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \quad \quad \quad  \end{array}
$$

which is precisely $\ell _ { \mathrm { C S } } ( y , a )$

## 5.1 Bounds for Surrogate Losses

We now shift gears to our study of surrogate losses for $\ell _ { 0 - 1 }$ , with the goal of understanding which hardness certificates are guaranteed to exist under the mild assumption of δ-hardness under 0-1 loss for a given class $\mathcal { C }$ after appropriate rounding to $\Delta _ { k }$ . As a warm-up before proving our main result showing that surrogate losses can be determined from a loss function’s associated entropy, we prove a result specific to the Crammer-Singer loss $\ell _ { \mathrm { C S } }$ . The result states that 0-1 loss is at most $( k - 1 ) / k$ times the Crammer-Singer hinge loss, provided that we first project $a \in \mathbb { R } ^ { k }$ onto the probability simplex $\Delta _ { k }$ . While similar bounds appear in the literature [vdHFCB21], we provide a proof for completeness.

Lemma 5.2 (Crammer-Singer Rounding). Given $y \in [ k ]$ and $a \in \mathbb { R } ^ { k }$ , let $\tilde { a } = \mathrm { p r o j } _ { \Delta _ { k } } ( a )$ . Equivalently, let $\tilde { \boldsymbol { a } } _ { i } = ( a _ { i } - \tau ) _ { + }$ for the unique threshold $\tau \in \mathbb { R }$ such that $\textstyle \sum _ { i } { \tilde { a } } _ { i } = 1$ . Then,

$$
\ell _ { \mathrm { C S } } ( y , a ) \geqslant \frac { k } { k - 1 } \mathbb { E } _ { { \tilde { y } } \sim { \tilde { a } } } [ \ell _ { 0 - 1 } ( y , \tilde { y } ) ] .
$$

Proof. Let $y = e _ { i }$ . When $\tilde { a } _ { i } = 1$ , there’s nothing to prove. Otherwise, we have $1 - \tilde { a } _ { i } > 0$ , so there must be some $j ^ { \star } \neq i$ with $\tilde { a } _ { j ^ { \star } } \geqslant ( 1 - \tilde { a } _ { i } ) / ( k - 1 ) > 0$ . Since $\tilde { a } _ { j ^ { \star } } = a _ { j ^ { \star } } - \tau$ and $\tilde { a } _ { i } \geqslant a _ { i } - \tau$

$$
\ell _ { \mathrm { C S } } ( e _ { i } , a ) \geqslant 1 - a _ { i } + a _ { j ^ { \star } } \geqslant 1 - \tilde { a } _ { i } + \tilde { a } _ { j ^ { \star } } \geqslant \frac { k } { k - 1 } ( 1 - \tilde { a } _ { i } ) = \frac { k } { k - 1 } \mathbb { E } _ { \tilde { y } \sim \tilde { a } } [ \ell _ { 0 - 1 } ( e _ { i } , \tilde { y } ) ] .
$$

The preceding result implies, in particular, that $\ell _ { \mathrm { C S } }$ is a $k / ( k - 1 )$ -surrogate loss for $\ell _ { 0 - 1 }$ , as formulated in Definition 1.14. Next, we prove Theorem 1.16, which shows that β-surrogate losses can be completely determined by inspecting the relevant losses’ associated entropy functions, which are given by the formula in Definition 1.15.

Theorem 1.16 (Entropy Surrogates). Given sets $\mathcal { A } , B , \mathcal { y }$ and $\beta \geqslant 0$ and finite losses $\ell _ { \mathcal { A } } : \mathcal { V } \times \mathcal { A } $ R and $\ell _ { B } : \mathcal { V } \times \mathcal { B } \to \mathbb { R }$ , let $h _ { \mathcal { A } } , h _ { \mathcal { B } } : \Delta ( \mathcal { V } )  \overline { { \mathbb { R } } }$ denote their entropy functions. $I f \mathcal { V } , B$ are finite, then $\ell _ { A }$ is a β-surrogate loss for $\ell _ { B }$ if and only if $h _ { \lrcorner } ( p ) \geqslant \beta h _ { \mathnormal { B } } ( p )$ for all $p \in \Delta ( \mathcal { V } )$

Proof. Suppose first that $\ell _ { A }$ is $\mathrm { ~ a ~ } \beta \cdot$ -surrogate loss for $\ell _ { B }$ with rounding rule R. Then,

$$
h _ { \mathcal { A } } ( p ) = \operatorname* { i n f } _ { a \in \mathcal { A } } \mathbb { E } _ { y \sim p } [ \ell _ { A } ( y , a ) ] \geqslant \beta \operatorname* { i n f } _ { a \in \mathcal { A } } \mathbb { E } _ { \textbf { \delta } y \sim p } [ \ell _ { B } ( y , b ) ] \geqslant \beta \operatorname* { i n f } _ { b \in \mathcal { B } } \mathbb { E } _ { y \sim p } [ \ell _ { B } ( y , b ) ] = \beta h _ { B } ( p ) .
$$

Conversely, suppose that $h _ { \mathcal { A } } ( p ) \geqslant \beta h _ { \mathcal { B } } ( p )$ . Then, by the minimax theorem, for all $a \in { \mathcal { A } }$

$$
\begin{array} { r l } { \displaystyle \underset { R ( a ) \in \Delta ( \mathcal B ) } { \operatorname* { m a x } } \operatorname* { m i n } _ { y \in \mathcal { Y } } \mathbb { E } _ { b \sim R ( a ) } [ \ell _ { \mathcal { A } } ( y , a ) - \beta \cdot \ell _ { \mathcal { B } } ( y , b ) ] = \underset { p \in \Delta ( \mathcal { Y } ) } { \operatorname* { m i n } } \underset { b \in \mathcal B } { \operatorname* { m a x } } \mathbb { E } _ { y \sim p } [ \ell _ { \mathcal { A } } ( y , a ) - \beta \cdot \ell _ { \mathcal { B } } ( y , b ) ] } & \\ { = \underset { p \in \Delta ( \mathcal { Y } ) } { \operatorname* { m i n } } \mathbb { E } _ { y \sim p } [ \ell _ { \mathcal { A } } ( y , a ) ] - \beta \cdot h _ { \mathcal { B } } ( p ) } & \\ { \geqslant \underset { p \in \Delta ( \mathcal { Y } ) } { \operatorname* { m i n } } h _ { \mathcal { A } } ( p ) - \beta \cdot h _ { \mathcal { B } } ( p ) } & \\ { \geqslant 0 . } \end{array}
$$

Note that switching the $\mathrm { ^ { 6 } m i n } ^ { \prime }$ and $\mathrm { \ddot { m a x } \vec { \Omega } }$ is certainly valid because $\mathcal { V }$ and $\boldsymbol { B }$ are finite. Therefore, any maximizer $R ( a ) \in \Delta ( B )$ for the left-hand side is a rounding rule proving β-surrogacy. □

In Appendix A, we give an alternate proof of Theorem 1.16 in the special case of surrogates for the 0-1 loss. This proof is quite a bit more intuitive than the previous one, and it also clarifies the structure of the optimal rounding rule. Next, we calculate the entropy functions for $\ell _ { 0 - 1 } , \ell _ { \mathrm { w e a k } } .$ $\ell _ { \mathrm { s t r o n g } }$ , and $\ell _ { \mathrm { C S } }$ and plug these into Theorem 1.16 to relate each of the losses $\ell _ { \mathrm { w e a k } } , \ell _ { \mathrm { s t r o n g } } ,$ and $\ell _ { \mathrm { C S } }$ to 0-1 loss. This implies, in particular, that δ-hardness under $\ell _ { 0 - 1 }$ implies the existence of the hardness certificates corresponding to $\ell _ { \mathrm { s t r o n g } }$ and $ { \ell _ { \mathrm { { C S } } } }$ with the parameters discussed in Section 1.2.

Lemma 5.3 (Entropy Calculations). For any $k \geqslant 2$ , we have

$$
h _ { 0 - 1 } ( p ) = 1 - \operatorname* { m a x } _ { j } p _ { j } ,
$$

$$
h _ { \mathrm { w e a k } } ( p ) = k \operatorname* { m i n } _ { j } p _ { j } ,
$$

$$
\begin{array} { r l } & { h _ { \mathrm { s t r o n g } } ( p ) = 1 - d _ { \mathrm { T V } } ( p , 1 \vec { / } k ) , } \\ & { ~ h _ { \mathrm { C S } } ( p ) = \operatorname* { m i n } \{ 1 , 2 ( 1 - \operatorname* { m a x } _ { j } p _ { j } ) \} . } \end{array}
$$

Corollary 5.4. $I f k \geqslant 3$ , the loss functions $\ell _ { \mathrm { w e a k } } , \ell _ { \mathrm { s t r o n g } } ,$ , and $\ell _ { \mathrm { C S } }$ are surrogate losses for $\ell _ { 0 - 1 }$ with parameters $\beta _ { \mathrm { w e a k } } = 0 , \beta _ { \mathrm { s t r o n g } } = 4 / k , \beta _ { \mathrm { C S } } = k / ( k - 1 )$ $I f k = 2$ , then all three $\beta$ parameters are equal to 2 instead. Each of these $\beta$ parameters is optimal.

Proof. We combine Theorem 1.16 with the formulas from Lemma 5.3.

• For weak hinge loss, take any $p \in \Delta _ { k }$ with some coordinate equal to 0 and no coordinate equal to 1 (needs $k \geqslant 3 )$ . By Lemma 5.3, we have $h _ { \mathrm { w e a k } } = 0$ but $h _ { 0 - 1 } > 0$ , so $\beta _ { \mathrm { w e a k } } = 0$

• For strong hinge loss, let $m = \operatorname* { m a x } _ { j } p _ { j }$ and note that $p _ { i } \leqslant m$ and m $\geqslant 1 / k$ . Therefore,

$$
\frac { h _ { \mathrm { s t r o n g } } ( p ) } { h _ { 0 - 1 } ( p ) } = \frac { \sum _ { i } \operatorname* { m i n } \{ p _ { i } , 1 / k \} } { 1 - m } \geqslant \frac { \sum _ { i } p _ { i } / k m } { 1 - m } = \frac { 1 } { k m ( 1 - m ) } \geqslant \frac { 4 } { k } ,
$$

with equality at $p = ( 1 / 2 , 1 / 2 , 0 , \dots , 0 )$

• For Crammer-Singer hinge loss, the ratio between $h _ { \mathrm { C S } } ( p )$ and $h _ { 0 - 1 } ( p )$ is precisely min $\{ 1 / ( 1 -$ $\operatorname* { m a x } _ { j } p _ { j } ) , 2 \}$ , which is minimized at the uniform distribution with $\beta _ { \mathrm { C S } } = k / ( k - 1 )$

The case $k = 2$ is easily checked separately.

Although we have focused our discussion in the multiclass case on various generalizations of hinge loss that are closely related to 0-1 loss, there are of course many other multiclass convex losses, such as $\ell _ { p }$ losses and more. Each of these gives rise to its own quantitative notion of hardness and, via our results, its own notion of a “hardness certificate”. We conclude this section with just one such example, pertaining to multiclass $\ell _ { 1 }$ loss. Here, the hardness certificate is neither a measure nor a masking channel, but a function $\sigma : \mathcal { X } \times \mathcal { Y }  [ - 1 , + 1 ] ^ { k }$ , whose outputs need not comprise a probability distribution over $[ k ]$ (in fact, some coordinates may even be negative).

Theorem 5.5 (Multiclass $\ell _ { 1 }$ Hardcore). For all finite X and $\mathcal { V } = [ k ]$ , distributions over $( { \pmb x } , { \pmb y } ) \in$ $\mathcal { X } \times \mathcal { V } _ { . }$ , finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , and $\varepsilon > 0$

$$
\begin{array}{c} \operatorname* { m i n } _ { \lambda \in \mathbb R ^ { C } } \mathbb E [ \| y - c _ { \lambda } ( \pmb x ) \| _ { 1 } ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { \substack { \sigma : \mathcal X \times \mathcal { Y } \to [ - 1 , + 1 ] ^ { k } } } \mathbb E [ \langle y , \sigma ( \pmb x , y ) \rangle ]  \\ { \textnormal { } s . t . \forall c \in \mathcal C , ~ \big | \mathbb E [ \langle c ( \pmb x ) , \sigma ( \pmb x , y ) \rangle ] \big | \leqslant \varepsilon . } \end{array}
$$

Proof. As usual, we apply Theorem 3.3 with the indistinguishability constraints R, S from Definition 3.1. We use the functions $g ( y , a ) = \| y - a \|$ <sub>1</sub> and

$$
h ( y , b ) = { \left\{ \begin{array} { l l } { \langle y , b \rangle } & { { \mathrm { i f ~ } } b \in [ - 1 , + 1 ] ^ { k } , } \\ { - \infty } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

These are conjugates because

$$
\operatorname* { s u p } _ { b \in [ - 1 , + 1 ] ^ { k } } h ( y , b ) - \langle a , b \rangle = \operatorname* { s u p } _ { b \in [ - 1 , + 1 ] ^ { k } } \langle y , b \rangle - \langle a , b \rangle = \| y - a \| _ { 1 } = g ( y , a ) .
$$

Invoking Theorem 3.3 as usual completes the proof.

## 6 Universal Masking Channels From Multicalibration

In this section, we first prove our theorem on universal hardness certificates via masking channels (Theorem 1.20). We then show how this result can be instantiated to recover a previous multicalibration-based construction of hardcore measures in the setting of binary 0-1 loss [CDV24, CGKR25]. We state and prove our new result in Section 6.1 and compare it to the prior one in Section 6.2.

## 6.1 Deriving the Multiclass Result

To begin, we need a certain helper lemma. This lemma establishes the key relationship between the lowest-loss response $\hat { c } _ { g }$ to a loss function g and the highest-fidelity response $\widehat { W } _ { h }$ to a fidelity function h. To begin, we recall the definitions of these post-processing functions:

Definition 1.17 (Lowest-Loss Response). Given a function $g : \mathcal { V } \times \mathbb { R } ^ { k }  \overline { { \mathbb { R } } }$ , a distribution $z \in \Delta _ { k }$ and a label $\mathbf { \nabla } _ { \mathbf { y } } \sim \mathbf { \nabla } _ { z } .$ , a lowest-loss response to z with respect to $g$ is any

$$
\hat { c } _ { g } ( z ) \in \operatorname * { a r g m i n } _ { c ( z ) \in \mathbb { R } ^ { k } } \mathbb { E } [ g ( \pmb { y } , c ( z ) ) ] \quad \mathrm { s . t . } \quad \langle c ( z ) , \vec { 1 } \rangle = 0 ,
$$

assuming any exist. We instead say $\hat { c } _ { g }$ is γ-optimal if it achieves loss within $\gamma$ of the infimum.

Definition 1.18 (Highest-Fidelity Response). Given a fidelity function $h : \mathcal { V } \times \Delta _ { k } \to \overline { { \mathbb { R } } }$ , a distribution $z \in \Delta _ { k }$ , and a label $\mathbf { \boldsymbol { y } } \sim \mathbf { \boldsymbol { z } } ,$ a best-fidelity response to z with respect to h is any

$$
\widehat { W } _ { h } ( z ) \in \operatorname { a r g m a x } _ { W ( z ) \in \Delta _ { k } ^ { k } } \mathbb { E } [ h ( \pmb { y } , W ( z ) \pmb { y } ) ] \quad \mathrm { s . t . } \quad W ( z ) z = \frac { \vec { 1 } } { k } ,
$$

assuming any exist. We instead say $\widehat { W } _ { h }$ is γ-optimal if it achieves fidelity within γ of the maximum.

Having reviewed the definitions of $\hat { c } _ { g }$ and ${ \widehat { W } } _ { h } ,$ we now state and prove our key helper lemma, which states that these two functions actually achieve the same value for their respective goals:

Lemma 6.1. Given a label distribution $z \in \Delta _ { k }$ , let $y _ { z } \sim z$ . Then, for all convex-concave conjugates $g : \mathcal { V } \times \mathbb { R } ^ { k } \to \overline { { \mathbb { R } } }$ and $h : \mathcal { V } \times \Delta _ { k } \to \overline { { \mathbb { R } } }$ , all α-optimal $\hat { c } _ { g } ( z )$ , and all β-optimal $\widehat { W } _ { h } ( z )$ 2

$$
0 \leqslant \mathbb { E } [ g ( \pmb { y } _ { z } , \hat { c } _ { g } ( z ) ) ] - \mathbb { E } [ h ( \pmb { y } _ { z } , \widehat { W } _ { h } ( z ) \pmb { y } _ { z } ) ] \leqslant \alpha + \beta .
$$

Proof. First, define

$$
p ^ { \star } = \operatorname* { i n f } _ { c ( z ) \in \mathbb { R } ^ { k } } \mathbb { E } [ g ( \pmb { y } _ { z } , c ( z ) ) ] \quad \mathrm { s . t . } \quad \langle c ( z ) , \vec { 1 } \rangle = 0 .
$$

Next, define

$$
d ^ { \star } = \operatorname* { m a x } _ { W ( z ) \in \Delta _ { k } ^ { k } } \mathbb { E } [ h ( { \pmb y } _ { z } , W ( z ) { \pmb y } _ { z } ) ] \quad \mathrm { s . t . } \quad W ( z ) z = \frac { \vec { 1 } } { k } .
$$

By Definitions 1.17 and 1.18, the function $\hat { c } _ { g }$ is α-optimal for the optimization problem defining $p ^ { \star }$ and $\widehat { W } _ { h }$ is β-optimal for the optimization problem defining $d ^ { \star }$ . Therefore, if we could show that $p ^ { \star } = d ^ { \star }$ , we would be done. For this, we just apply Theorem 1.9 with $\mathcal { C }$ containing just the k functions $c _ { j } ( x ) = e _ { j }$ , so that $c _ { \lambda } ( x ) = \lambda$ for all $\boldsymbol { \lambda } \in \mathbb { R } ^ { k }$ , and error parameter $\varepsilon = 0$ We let $c ( z ) = \pi ( \lambda )$ . The constraint $W$ is (C, 0)-masking for $y _ { z } \sim z$ translates to $W ( z ) z = \vec { 1 } / k$ □

The preceding lemma corresponds to the important special case of Theorem 1.9 where the label $\mathbf { \Delta } _ { \mathbf { \mathcal { Y } } _ { z } } \in \mathcal { Y }$ is a truly random sample from a distribution $z \in \Delta _ { k }$ , entirely independent of the “context”, “feature vector”, or “input” $x \in { \mathcal { X } } .$ . Soon, we will apply this helper lemma to each z-level set of a multicalibrated model, within which the true label is computationally indistinguishable from precisely this scenario.

We now state and prove the formal version of Theorem 1.20, which is the main result of this section. For clarity, we shall split the formal theorem into three parts:

(1) The first part, Theorem 6.4, shows that if there exists a model $p : \mathcal { X }  \Delta _ { k }$ of the conditional distribution $_ { y | \pmb { x } }$ with certain indistinguishability properties depending on the loss function $^ { g , }$ the fidelity function $h ,$ and the class ${ \mathcal { C } } ,$ then $p$ yields a near-optimal noisy channel certifying the hardness of approximating ${ \pmb y } | { \pmb x }$ by members of C.

Implicit in our “near-optimal” claim is the assumption that the complexity of $p$ is comparable to that of C. The required complexity bound on p is established in the next part.

(2) The second part, Theorem 6.5, shows that even for a large family $\mathcal { F }$ of loss functions g and fidelity functions h, there exists a single universal model $p : \mathcal { X }  \Delta _ { k }$ satisfying the relevant indistinguishability properties for all $g , h \in { \mathcal { F } }$ simultaneously. Moreover, if the functions $c \in { \mathcal { C } }$ and $g , h \in \mathcal { F }$ and ${ \widehat { c } } _ { g } , { \widehat { W } } _ { h }$ can all be computed eficiently, then so can $p .$

The proof of this part will be very simple. It involves a direct appeal to well-known constructions of low-complexity models satisfying calibration and multiaccuracy, which we reviewed in Section 2.3.

(3) The third part, Theorem 6.6, highlights special cases where the noisy channel $\widehat { W } _ { h } ( p ( x ) )$ itself, not just the model $p ( x )$ , can be made universal. Specifically, we show that the same postprocessing $\widehat { W } = \widehat { W } _ { h }$ works if all fidelity functions h in the class are monotone, or if they are all monotone mixing. For the definition of such loss functions, recall Definition 1.19.

Before stating the first part of our result, Theorem 6.4, we must define the specific indistinguishability conditions that we will require of the model p. The first condition is a type of weighted multiaccuracy (i.e. a special case of Definition 2.17). We shall refer to this condition as channelweighted multiaccuracy:

Definition 6.2 (Channel-Weighted Multiaccuracy). Given finite X and $\mathcal { V } = [ k ]$ , a distribution over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , and a fidelity function response $\widehat { W } _ { h } : \Delta _ { k } \to \Delta _ { k } ^ { k }$ we say that a model $p : \mathcal { X }  \Delta _ { k }$ satisfies W-weighted $( \mathcal { C } , \varepsilon )$ -multiaccuracy if for all $c \in { \mathcal { C } }$

$$
\begin{array} { r } { \Big | \mathbb { E } \big [ \big \langle \widehat { W } _ { h } ( p ( \pmb { x } ) ) ^ { \top } c ( \pmb { x } ) , \pmb { y } - p ( \pmb { x } ) \big \rangle \big ] \Big | \leqslant \varepsilon . } \end{array}
$$

Intuition for this condition comes from rewriting the left-hand side as

$$
\begin{array} { r } { \mathbb { E } \big [ \big < c ( \pmb { x } ) , \widehat W _ { h } ( p ( \pmb { x } ) ) \pmb { y } - \widehat W _ { h } ( p ( \pmb { x } ) ) p ( \pmb { x } ) \big > \big ] = \mathbb { E } \big [ \big < c ( \pmb { x } ) , \widehat W _ { h } ( p ( \pmb { x } ) ) \pmb { y } - \vec { 1 } / k \big > \big ] . } \end{array}
$$

and interpreting this as saying that $W ( { \pmb x } ) = \widehat { W } _ { h } ( p ( { \pmb x } ) )$ is C-masking.

The second indistinguishability condition that we will require of the model p is a type of weighted calibration (i.e. a special case of Definition 2.15). We shall refer to this condition as duality gap calibration since the weight functions will be derived from the duality gap in Lemma 6.1.

Definition 6.3 (Duality Gap Calibration). Given finite X and $\mathcal { V } = [ k ]$ , a distribution over $( { \pmb x } , { \pmb y } ) \in$ $\mathcal { X } \times \mathcal { V }$ , finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , a loss function response $\hat { c } _ { g } : \Delta _ { k } \to \mathbb { R } ^ { k }$ and a fidelity function response $\widehat { W } _ { h } : \Delta _ { k } \to \Delta _ { k } ^ { k }$ we say that a model $p : \mathcal { X }  \Delta _ { k }$ satisfies ε-duality gap calibration if

$$
\mathbb { E } \big [ \big \langle \mathrm { G a p } ( p ( { \pmb x } ) ) , { \pmb y } - p ( { \pmb x } ) \big \rangle \big ] \leqslant \varepsilon ,
$$

where we define $\mathrm { G a p } ( z ) \in \overline { { \mathbb { R } } } ^ { k }$ by the formula

$$
\operatorname { G a p } ( z ) _ { i } = g ( e _ { i } , { \widehat { c } } _ { g } ( z ) ) - h ( e _ { i } , { \widehat { W } } _ { h } ( z ) e _ { i } ) .
$$

For applications, we will typically assume that Gap(z) is bounded in a box $[ - B , B ] ^ { k }$

Intuition for this condition comes from the loss OI technique $[ \mathrm { G H K ^ { + } 2 3 } , \mathrm { D K R ^ { + } 2 1 } ]$ . Lemma 6.1 gives us a guarantee that holds for labels $y _ { p } | { \pmb x } \sim p ( { \pmb x } )$ , the simulated distribution of the predictor $p .$ We want this guarantee to hold for the true labels y. Since the tests we wish to fool only depend on the predictions, this can be framed as a calibration condition.

At this point, having defined channel-weighted multiaccuracy and duality gap calibration, we are ready to state and prove the three main results of this section. First, we show that these properties of a model $p : \mathcal { X }  \Delta _ { k }$ are suficient for constructing masking channels that certify hardness:

Theorem 6.4 (Universal Masking Channels – Properties). Fix finite X and $\mathcal { V } = [ k ]$ , a distribution over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ , a model $p : \mathcal { X }  \Delta _ { k }$ , and convex-concave conjugates $g : \mathcal { V } \times \mathbb { R } ^ { k } \to \mathbb { R }$ and $h : \mathcal { V } \times \Delta _ { k } \to \overline { { \mathbb { R } } }$ . Let $\hat { c } _ { g } : \Delta _ { k } \to \mathbb { R } ^ { k }$ be an ε<sub>1</sub>-optimal lowest-loss response, and let $\widehat { W } _ { h } : \Delta _ { k } \to \Delta _ { k } ^ { k }$ be an ε<sub>2</sub>-optimal highest-fidelity response. Then,

(a) (Hardness $\geqslant$ Fidelity) If the model p is $\widehat { W } _ { h ^ { - } w e i g h t e d } \left( \mathcal { C } , \varepsilon _ { 3 } \right)$ -multiaccurate, then the masking channel $W _ { p } = \widehat { W } _ { h } \circ p$ derived from p certifies hardness, in the sense that for all $c \in { \mathcal { C } }$ ，

$$
\begin{array} { r } { \mathbb { E } [ g ( \pmb { y } , \pi ( c ( \pmb { x } ) ) ) ] \geqslant \mathbb { E } [ h ( \pmb { y } , W _ { p } ( \pmb { x } ) \pmb { y } ) ] - \varepsilon _ { 3 } . } \end{array}
$$

(b) (Hardness $\leqslant$ Fidelity) If the model p is $\varepsilon _ { 4 } - d u a l i t y$ gap calibrated, then the bound from part (a) is nearly tight, in the sense that the action $c _ { p } = \hat { c } _ { g } \circ p$ achieves loss

$$
\begin{array} { r } { \mathbb { E } \big [ g ( \pmb { y } , c _ { p } ( \pmb { x } ) ) \big ] \leqslant \mathbb { E } \big [ h ( \pmb { y } , W _ { p } ( \pmb { x } ) \pmb { y } ) \big ] + ( \varepsilon _ { 1 } + \varepsilon _ { 2 } + \varepsilon _ { 4 } ) . } \end{array}
$$

(Note: although $c _ { p } \notin { \mathcal { C } }$ in general, we show in Theorem 6.5 that $c _ { p }$ “nearly” belongs to $\mathcal { C } . \big )$

Proof. We prove each part separately.

(a) Convex-concave conjugacy of $g , h$ (Definition 2.7) says that for all $a , b \in \mathbb { R } ^ { k }$

$$
g ( y , a ) \geqslant h ( y , b ) - \langle a , b \rangle .
$$

For each $( x , y ) \in \mathcal { X } \times \mathcal { Y }$ and $c \in { \mathcal { C } }$ , we can set $a = \pi ( c ( x ) )$ and $b = \widehat { W } _ { h } ( p ( x ) ) y$ , which gives

$$
g ( y , \pi ( c ( x ) ) ) \geqslant h \bigl ( y , \widehat W _ { h } ( p ( x ) ) y \bigr ) - \bigl \langle \pi ( c ( x ) ) , \widehat W _ { h } ( p ( x ) ) y \bigr \rangle .
$$

Our goal is to show that the inner product term on the right is small in expectation. To begin, let us rewrite it slightly. Since the centering function $\pi ( \cdot )$ is equivalent to subtracting of the mean of $c ( x )$ , we can rewrite this inner product as

$$
\big \langle \pi ( c ( x ) ) , \widehat W _ { h } ( p ( x ) ) y \big \rangle = \big \langle c ( x ) , \widehat W _ { h } ( p ( x ) ) y - \vec { 1 } / k \big \rangle .
$$

Next, we make the substitution $\overrightarrow { 1 } / k = \widehat { W } _ { h } ( z ) z$ , where we define $z = p ( x )$ . This is valid by the very definition of $\widehat { W } _ { h }$ (Definition 1.18). Making this substitution yields the expression

$$
\langle c ( x ) , \widehat { W } _ { h } ( p ( x ) ) ( y - p ( x ) ) \rangle .
$$

Next, we move the factor of $\widehat { W } _ { h } ( p ( x ) )$ to the other side of the inner product:

$$
\langle \widehat { W } _ { h } ( p ( x ) ) ^ { \top } c ( x ) , y - p ( x ) \rangle .
$$

By the $\widehat { W } _ { h }$ -weighted $( \mathcal { C } , \varepsilon _ { 3 } )$ -multiaccuracy assumption, in expectation, this quantity is

$$
\mathbb { E } \big [ \langle \widehat { W } _ { h } ( p ( \pmb { x } ) ) ^ { \top } c ( \pmb { x } ) , \pmb { y } - p ( \pmb { x } ) \rangle \big ] \leqslant \varepsilon _ { 3 } .
$$

(b) Our goal is to show that

$$
\begin{array} { r } { \mathbb { E } [ g ( \pmb { y } , \hat { c } _ { g } ( p ( \pmb { x } ) ) ) ] - \mathbb { E } [ h ( \pmb { y } , \widehat { W } _ { h } ( p ( \pmb { x } ) ) \pmb { y } ) ] \leqslant \varepsilon _ { 1 } + \varepsilon _ { 2 } + \varepsilon _ { 4 } . } \end{array}
$$

Alternatively, phrased in terms of the function Gap from Definition 6.3, we want

$$
\mathbb { E } [ \langle \mathrm { G a p } ( p ( \pmb { x } ) ) , \pmb { y } \rangle ] \leqslant \varepsilon _ { 1 } + \varepsilon _ { 2 } + \varepsilon _ { 4 } .
$$

By the $\varepsilon _ { 4 } \cdot$ -duality gap calibration assumption, it sufices to show that

$$
\mathbb { E } [ \langle \mathrm { G a p } ( p ( \pmb { x } ) ) , p ( \pmb { x } ) \rangle ] \leqslant \varepsilon _ { 1 } + \varepsilon _ { 2 } .
$$

But we have

$$
\begin{array} { r l } & { \mathbb { E } [ \langle \mathrm { G a p } ( p ( \pmb { x } ) ) , p ( \pmb { x } ) \rangle ] = \mathbb { E } _ { \pmb { x } } \Big [ \displaystyle \sum _ { i \in [ k ] } p ( \pmb { x } ) _ { i } \mathrm { G a p } ( p ( \pmb { x } ) ) _ { i } \Big ] } \\ & { \qquad = \mathbb { E } _ { \pmb { x } } \Big [ \displaystyle \sum _ { i \in [ k ] } p ( \pmb { x } ) _ { i } ( g ( e _ { i } , \hat { c } _ { g } ( p ( \pmb { x } ) ) ) - h ( e _ { i } , \widehat { W } _ { h } ( p ( \pmb { x } ) ) e _ { i } ) ) \Big ] } \\ & { \qquad = \mathbb { E } _ { \pmb { x } , y _ { p } \mid \pmb { x } \sim p ( \pmb { x } ) } \big [ g ( \pmb { y } _ { p } , \hat { c } _ { g } ( p ( \pmb { x } ) ) ) - h ( \pmb { y } _ { p } , \widehat { W } _ { h } ( p ( \pmb { x } ) ) \pmb { y } _ { p } ) \big ] . } \end{array}
$$

So the desired bound follows from Lemma 6.1.

At this point, we have shown that if the model $p : \mathcal { X }  \Delta _ { k }$ satisfies weighted calibration and weighted multiaccuracy, it yields our desired universal masking channel. The next theorem asserts that such models indeed exist for a large family of loss functions $g$ and fidelity functions h simultaneously. Perhaps the best way to interpret the following statement, which expresses p as a combination of various functions $g , h , \hat { c } _ { g } , \hat { W _ { h } }$ , is as follows: if the $g , h , \hat { c } _ { g } , \widehat { W } _ { h }$ are eficiently computable, which they generally should be, then the predicted label distribution $p ( x ) \in \Delta _ { k }$ is computable by combining the outputs of $O ( \log ( k ) / \varepsilon ^ { 2 } )$ functions from the class $\mathcal { C }$ on the input x. In other words, the complexity of $p$ exceeds that of C by only an $O ( \log ( k ) / \varepsilon ^ { 2 } )$ factor.

Theorem 6.5 (Universal Masking Channels – Existence). In the setting of Theorem $6 . 4 ,$ consider a family of tuples $( g , h , \widehat { c } _ { g } , \widehat { W } _ { h } )$ , where $g , h$ are convex-concave loss-fidelity pairs and ${ \widehat { c } } _ { g } , { \widehat { W } } _ { h }$ are (possibly suboptimal) responses. Let $\mathcal { F }$ be the collection of all functions $f$ with one of two possible forms:

$f ( z , x ) = \widehat { W } _ { h } ( z ) ^ { \top } c ( x )$ for some $c \in { \mathcal { C } }$ and some $\widehat { W } _ { h }$ in the family (see Definition 6.2),

$f ( z , x ) = \mathrm { G a p } ( z )$ for some $g , h , \widehat { c } _ { g } , \widehat { W } _ { h }$ in the family (see Definition $6 . 3 )$

Suppose that the range of each $f \in { \mathcal { F } }$ is contained in the box $[ - B , + B ] ^ { k }$ . Then, there exists a single model $p : \mathcal { X }  \Delta _ { k }$ satisfying all of the assumptions in Theorem $6 . 4 ~ f o r$ all loss-fidelity pairs $^ { g , }$ h in the family. Moreover, p has the form $p = p _ { m }$ , where we define $p _ { 0 } ( x ) = \vec { 1 } / k$ and

$$
p _ { j } ( x ) \propto \exp \Bigl ( \frac { \varepsilon } { B ^ { 2 } } \sum _ { i = 1 } ^ { j } f _ { i } ( p _ { i - 1 } ( x ) , x ) \Bigr ) ,
$$

normalized so that $\begin{array} { r } { \sum _ { y } p _ { j } ( x ) _ { y } = 1 } \end{array}$ , for some $m = { \cal O } ( B ^ { 2 } \log ( k ) / \varepsilon ^ { 2 } )$ functions $f _ { 1 } , \ldots , f _ { m } \in \pm \mathcal { F }$ , and for some common error parameter $\varepsilon = \Omega ( \operatorname* { m i n } \{ \varepsilon _ { 1 } , \dots , \varepsilon _ { 4 } \} )$ , where $\varepsilon _ { i }$ is defined in Theorem $6 . 4$

Proof. Apply Theorem 2.18 to $\mathcal { F }$ scaled down by a factor of $B ,$ as well as $\varepsilon / B$

Theorem 6.6 (Universal Masking Channels – Common Post-Processing). There exists a common highest-fidelity response function $\widehat { W } = \widehat { W } _ { h }$ for all monotone fidelity functions. There exists a (diferent) common highest-fidelity response function $\widehat { W } = \widehat { W } _ { h }$ for all monotone mixing fidelity functions.

Proof. The proof is essentially exactly as we sketched in the introduction and depicted in Figure 2. Indeed, let $\widehat { W } _ { 1 } ( z ) \in \Delta _ { k } ^ { k }$ be any noisy channel satisfying the following properties:

• (Feasible) $\widehat { W } _ { 1 } ( z ) z = \vec { 1 } / k$ . That is, $\widehat { W } _ { 1 }$ adjusts the probability mass of z to be uniform.

• (Minimal Movement) For all $i , j \in [ k ]$ , write the probability mass moved from i to j as

$$
m _ { i j } = z _ { i } \cdot \langle \widehat { W } _ { 1 } ( z ) e _ { i } , e _ { j } \rangle .
$$

For all $i ,$ we require $\begin{array} { r } { \sum _ { j \neq i } m _ { i j } = ( z _ { i } - 1 / k ) _ { + } } \end{array}$ , and for all $j ,$ , we require $\begin{array} { r } { \sum _ { i \neq j } m _ { i j } = ( 1 / k - z _ { j } ) _ { + } } \end{array}$ It is clear that such a function $\widehat { W } _ { 1 }$ , visualized in Figure $2 \mathrm { a } .$ , maximizes $\mathbb { E } [ h ( { \pmb y } , \widehat { W } _ { 1 } ( z ) { \pmb y } ) ]$ over $\mathbf { \Omega } _ { \mathbf { \mathcal { Y } } } \sim \mathbf { \mathcal { Z } }$ subject to the feasibility constraint for any monotone fidelity function $h ,$ as defined in Definition 1.19. Similarly, for monotone mixing fidelity functions, also defined in Definition 1.19, let $z _ { \mathrm { m i n } } = \operatorname* { m i n } _ { j } z _ { j }$ and consider the matrix $\widehat { W } _ { 2 } ( z )$ whose $i ^ { \mathrm { t h } }$ column is

$$
\frac { z _ { \mathrm { m i n } } } { z _ { i } } e _ { i } + \left( 1 - \frac { z _ { \mathrm { m i n } } } { z _ { i } } \right) \frac { \vec { 1 } } { k } .
$$

Intuitively, this channel takes all probability mass initially on label i above the minimum and redistributes it uniformly among the k labels. Clearly, $\widehat { W } _ { 2 }$ is feasible and maximizes $\mathbb { E } [ h ( { \pmb y } , \widehat { W } _ { 2 } ( z ) { \pmb y } ) ]$ for any monotone mixing fidelity function h. This concludes the proof. □

## 6.2 Recovering the Binary Result

In this section, we show how Theorems 6.4, 6.5, and 6.6 can be used to recover a recent result of [CDV24, CGKR25], which shows how to construct a hardcore measure in the traditional setting of binary 0-1 loss given a multicalibrated (or calibrated multiaccurate) predictor $p .$ In fact, our new Theorem 6.6 shows that the construction of this recent result actually works for all monotone fidelity functions in the binary setting.

Theorem 6.7 ([CDV24, CGKR25]). For all finite $x ,$ distributions over $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \{ \pm 1 \}$ , finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } \}$ , and models $p : \mathcal { X }  [ - 1 , + 1 ]$ , define $\mu : \left\{ \pm 1 \right\} \times \left[ - 1 , + 1 \right] \to \left[ 0 , 1 \right] \ b y$

$$
\mu ( y , z ) = { \frac { 1 - y z } { 1 + | z | } } .
$$

Then, the following statements hold:

(a) (Hardness $\geqslant$ Density) If p satisfies the weighted multiaccuracy condition

$$
\forall c \in { \mathcal { C } } , \qquad { \mathbb { E } } \Big [ \frac { c ( { \pmb x } ) } { 1 + | { \boldsymbol p } ( { \pmb x } ) | } \big ( { \pmb y } - { \boldsymbol p } ( { \pmb x } ) \big ) \Big ] \leqslant \varepsilon _ { 2 } ,
$$

then the measure $\mu$ derived from p certifies hardness against $\mathcal { C } .$

$$
\forall c \in \mathcal { C } , \qquad \mathbb { E } [ ( 1 - c ( \pmb { x } ) \pmb { y } ) _ { + } ] \geqslant \mathbb { E } [ \mu ( \pmb { y } , p ( \pmb { x } ) ) ] - \varepsilon _ { 2 } .
$$

(b) (Hardness $\leqslant$ Density) If p satisfies the calibration condition

$$
\mathbb { E } \Big [ \frac { \mathrm { s i g n } ( p ( { \pmb x } ) ) } { 1 + | p ( { \pmb x } ) | } \big ( { \pmb y } - p ( { \pmb x } ) \big ) \Big ] \geqslant - \varepsilon _ { 3 } ,
$$

then the bound from part $( a )$ is nearly tight:

$$
\begin{array} { r } { \mathbb { E } [ ( 1 - \mathrm { s i g n } ( p ( \pmb { x } ) ) \pmb { y } ) _ { + } ] \leqslant \mathbb { E } [ \mu ( \pmb { y } , p ( \pmb { x } ) ) ] + \varepsilon _ { 3 } . } \end{array}
$$

Proof. In the statements of Theorems 6.4, 6.5, and 6.6, let us substitute $k = 2$ and map the set $\Delta _ { 2 }$ , which is a line segment, to the interval $[ - 1 , + 1 ]$ in the natural way. In particular, the highestfidelity response $\widehat { W } ( z ) \in \Delta _ { 2 } ^ { 2 }$ , which was previously a 2-by-2 matrix with columns in $\Delta _ { 2 }$ , is now just a pair of scalars $\mu ( y , z ) \in [ - 1 , + 1 ]$ for $y \in \{ \pm 1 \}$ . We must verify three properties:

(i) The highest-fidelity response simplifies to

$$
\mu ( y , z ) = { \frac { 1 - y z } { 1 + | z | } } .
$$

(ii) The channel-weighted multiaccuracy distinguishers simplify to

$$
{ \frac { c ( x ) } { 1 + | p ( x ) | } } .
$$

(iii) The duality gap calibration distinguishers simplify to

$$
- { \frac { \mathrm { s i g n } ( p ( x ) ) } { 1 + | p ( x ) | } } .
$$

To begin, consider the fidelity function from Example 1.8, specialized to $k = 2$ and $\pm 1$ notation:

$$
h ( y , z ) = { \left\{ \begin{array} { l l } { \mu } & { { \mathrm { i f ~ } } z = \mu y { \mathrm { ~ f o r ~ s o m e ~ } } \mu \in [ 0 , 1 ] , } \\ { - \infty } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

Its convex conjugate $g$ is hinge loss, whose lowest-loss response is $\hat { c } _ { g } ( z ) = \mathrm { s i g n } ( z )$ . We now check the three properties listed earlier:

(i) First, to determine the highest-fidelity response, which we write as $\mu ( \boldsymbol { y } , z )$ , we could appeal to Theorem 6.6 by monotonicity, but for clarity, let us calculate it directly using Definition 1.18. In $\pm 1$ notation, we seek the function $\mu : \{ \pm 1 \} \times [ - 1 , + 1 ] \to [ 0 , 1 ]$ that maximizes

$$
\mathbb { E } _ { \pmb { y } \sim z } [ \mu ( \pmb { y } , z ) ]
$$

subject to the constraint

$$
\mathbb { E } _ { { y } \sim z } [ \mu ( { y } , z ) { y } ] = \frac { 1 + z } { 2 } \mu ( + 1 , z ) - \frac { 1 - z } { 2 } \mu ( - 1 , z ) = 0 .
$$

From the constraint, we immediately see that $\mu$ must have the form

$$
\mu ( y , z ) = ( 1 - y z ) \nu ( z )
$$

for some function $\nu : [ - 1 , + 1 ] \to [ 0 , + \infty )$ . In terms of $\nu ,$ our objective is to maximize

$$
\begin{array} { r } { \mathbb { E } _ { { \pmb { y } } \sim z } [ ( 1 - { \pmb { y } } z ) \nu ( z ) ] = ( 1 - z ^ { 2 } ) \nu ( z ) . } \end{array}
$$

Thus, we should take $\nu$ to be as large as possible. However, we also need the range of $\mu$ to remain bounded in the interval [0, 1]. Thus, we choose $\nu = 1 / ( 1 + | z | )$ , which leads to

$$
\mu ( y , z ) = { \frac { 1 - y z } { 1 + | z | } } .
$$

(ii) The channel-weighted multiaccuracy test

$$
\begin{array} { r } { \langle \widehat { W } _ { h } ( p ( x ) ) ^ { \top } c ( x ) , y - p ( x ) \rangle , } \end{array}
$$

which can be rewritten as

$$
\langle c ( x ) , \widehat { W } _ { h } ( p ( x ) ) y - \vec { 1 } / 2 \rangle ,
$$

simplifies substantially in ±1 notation to

$$
c ( x ) \cdot \mu ( y , p ( x ) ) y = { \frac { c ( x ) } { 1 + | p ( x ) | } } ( y - p ( x ) ) ,
$$

as desired.

(iii) The duality gap calibration test

$$
\langle \mathrm { G a p } ( p ( x ) ) , y - p ( x ) \rangle ,
$$

which can be rewritten as

$$
g ( y , \hat { c } _ { g } ( p ( x ) ) ) - h ( y , \mu ( y , p ( x ) ) y ) ,
$$

simplifies substantially in ±1 notation to

$$
( 1 - \mathrm { s i g n } ( p ( x ) ) y ) _ { + } - \mu ( y , p ( x ) ) = - \frac { \mathrm { s i g n } ( p ( x ) ) } { 1 + | p ( x ) | } ( y - p ( x ) ) ,
$$

as desired.

## 7 Characterizing Pseudoentropy via MaxEnt

In this section, we show that an exact characterization of pseudoentropy can be obtained using the MaxEnt algorithm. As discussed in the introduction, this result brings together two quite distinct lines of work: one in complexity theory [VZ12, Zhe14, HV25] and another in machine learning [BPP96, LDPDP97, DPS07, AS06]. We also show that it can be derived from our general Theorem 3.3.

To begin, let us revisit the starting point of our investigation: the classic hardcore lemma for binary 0-1 loss (Theorem 1.1). Recall that the first step we took, even before moving to the multiclass case, was to take the 2δ-hardcore measure $\mu : \mathcal { X } \to [ 0 , 1 ]$ and use it to decompose the function $f : \mathcal { X } \to \{ \pm 1 \}$ into

$$
f ( x ) = \underbrace { \mu ( x ) f ( x ) } _ { \mathrm { n o i s y ~ l a b e l s } } + \underbrace { ( 1 - \mu ( x ) ) f ( x ) } _ { \mathrm { h i g h - e n t r o p y ~ m o d e l } } .
$$

The first term, $\mu ( x ) f ( x )$ , which randomizes f outside the hardcore set, was the inspiration for our new noisy channel property. However, the other term, $( 1 - \mu ( x ) ) f ( x )$ , which randomized f inside the hardcore, is the starting point for the investigation of generalized pseudoentropy, pioneered by [VZ12, VZ13]. The relation between these two views can be visualized using Figure 3.

We now state a result that exactly characterizes pseudoentropy via (the hardness of) minimizing the associated GLM loss. This provides an analogue of our Theorem 1.9 in the setting of pseudoentropy.

Theorem 1.22 (Characterizing pseudoentropy using MaxEnt [BPP96, LDPDP97, DPS07, AS06]). For all finite X and $\mathcal { V } = [ k ]$ , distributions D over pairs $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$ , finite ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } ^ { k } \}$ and convex-concave conjugate pairs $g : \mathbb { R } ^ { k }  \mathbb { R }$ and $h : \Delta _ { k } \to \overline { { \mathbb { R } } }$ , let $\ell ( y , \eta ) = g ( - \eta ) - \langle y , \eta \rangle$ (GLM loss). Then, for all $\varepsilon \geqslant 0 , ^ { 2 0 }$

$$
\operatorname* { i n f } _ { \lambda \in \mathbb R ^ { C } } \mathbb { E } [ \ell ( y , c _ { \lambda } ( x ) ) ] + \varepsilon \| \lambda \| _ { 1 } = \operatorname* { m a x } _ { p : \mathcal X \to \Delta _ { k } } \mathbb { E } [ h ( p ( x ) ) ] { \quad s . t . \quad p ( x ) } \ i s \ ( \mathcal { C } , \varepsilon ) - i n d i s t i n g u i s h a b l e \ f r o m \ y .
$$

Proof. Given $g , h , \ell$ from the theorem statement, let $g _ { 1 } = \ell$ and

$$
h _ { 1 } ( y , b ) = { \left\{ \begin{array} { l l } { h ( z ) } & { { \mathrm { i f ~ } } b = y - z { \mathrm { ~ f o r ~ s o m e ~ } } z \in \Delta _ { k } , } \\ { - \infty } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

By Lemma A.5, the functions $g _ { 1 }$ and $h _ { 1 }$ are conjugates. Next, let R, S be the indistinguishability constraint functions from Definition 3.1, which are also conjugates by Lemma $\mathrm { A . 2 }$ . Since h has a bounded efective domain, its conjugate $g ,$ which is defined as a supremum over this domain, is always finite. Hence, so is $g _ { 1 }$ . Therefore, applying Theorem 3.3 to $g _ { 1 } , h _ { 1 } , R , S$ yields

$$
\begin{array} { r l } { \displaystyle \operatorname* { i n f } _ { \boldsymbol { \lambda } \in \mathbb R ^ { C } } \mathbb { E } [ \ell ( \boldsymbol { y } , c _ { \boldsymbol { \lambda } } ( \boldsymbol { x } ) ) ] + \varepsilon \| \boldsymbol { \lambda } \| _ { 1 } = \displaystyle \operatorname* { m a x } _ { \boldsymbol { p } : \boldsymbol { \chi } \times \boldsymbol { \mathcal { V } } \to \Delta _ { k } } \mathbb { E } [ h ( \boldsymbol { p } ( \boldsymbol { x } , \boldsymbol { y } ) ) ] } & { } \\ { \mathrm { s . t . } \forall c \in \mathcal { C } , \ : \left| \mathbb { E } [ \langle c ( \boldsymbol { x } ) , \boldsymbol { y } - \boldsymbol { p } ( \boldsymbol { x } , \boldsymbol { y } ) \rangle ] \right| \leqslant \varepsilon . } \end{array}
$$

Next, we show that without loss of generality, we may assume that the function $p ( x , y )$ on the right-hand side ignores its second input. To this end, define

$$
\bar { p } ( x _ { 0 } ) = \mathbb { E } [ p ( { \pmb x } , { \pmb y } ) \ | \ { \pmb x } = x _ { 0 } ] \in \Delta _ { k } .
$$

Clearly, if $p$ is $( \mathcal { C } , \varepsilon )$ -multiaccurate (i.e. feasible for the maximization problem) then so is ${ \bar { p } } .$ Moreover, by concavity of h and Jensen’s inequality, we have

$$
\mathbb { E } [ h ( p ( { \pmb x } , { \pmb y } ) ) ] \leqslant \mathbb { E } [ h ( { \bar { p } } ( { \pmb x } ) ) ] .
$$

Thus, replacing $p$ with ${ \bar { p } } ,$ we see that y-dependence in the maximizer is unnecessary.

The proof of the final claim is almost exactly the same as in the proof of Theorem 1.9, and we include it for the sake of completeness: Suppose ℓ is bounded below by the value $v \in \mathbb { R } . \mathrm { ~ H ~ } \varepsilon > 0$ then it is never beneficial to use λ whose norm exceeds

$$
r = \frac { \mathbb { E } [ \ell ( { \pmb y } , 0 ) ] - v } { \varepsilon } .
$$

Indeed, any such λ is strictly outperformed by $\lambda = 0$ . Thus, we can restrict the search space to the $\ell _ { 1 }$ ball of radius $r ,$ which is compact since C is finite. Since the left-hand side minimizes a lower semicontinuous function over a compact set, it attains this minimum. □

## References

[AS06] Yasemin Altun and Alex Smola. Unifying divergence minimization and statistical inference via convex duality. In G´abor Lugosi and Hans Ulrich Simon, editors, Learning Theory, pages 139–153, Berlin, Heidelberg, 2006. Springer Berlin Heidelberg.

[BBC<sup>+</sup>25] Marco Bressan, Nataly Brukhim, Nicol\`o Cesa-Bianchi, Emmanuel Esposito, Yishay Mansour, Shay Moran, and Maximilian Thiessen. Of dice and games: A theory of generalized boosting. In Nika Haghtalab and Ankur Moitra, editors, The Thirty Eighth Annual Conference on Learning Theory, June 30 - July 4, 2025, Lyon, France, volume 291 of Proceedings of Machine Learning Research, pages 596–640. PMLR, 2025.

[BCHM20] Nataly Brukhim, Xinyi Chen, Elad Hazan, and Shay Moran. Online agnostic boosting via regret minimization. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

[BDMM23] Nataly Brukhim, Amit Daniely, Yishay Mansour, and Shay Moran. Multiclass boosting: Simple and intuitive weak learning criteria. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

[Ber15] Dimitri P. Bertsekas. Convex Optimization Algorithms. Athena Scientific, 2015.

[BHK09] Boaz Barak, Moritz Hardt, and Satyen Kale. The uniform hardcore lemma via approximate Bregman projections. In Proceedings of the Twentieth Annual ACM-SIAM Symposium on Discrete Algorithms, SODA ’09, pages 1193–1200. Society for Industrial and Applied Mathematics, 2009.

[BJM06] Peter L. Bartlett, Michael I. Jordan, and Jon D. McAulife. Convexity, classification, and risk bounds. Journal of the American Statistical Association, 101(473):138–156, 2006.

[BPP96] Adam L. Berger, Vincent J. Della Pietra, and Stephen A. Della Pietra. A maximum entropy approach to natural language processing. Comput. Linguist., 22(1):39–71, March 1996.

[CDV24] S´ılvia Casacuberta, Cynthia Dwork, and Salil P. Vadhan. Complexity-theoretic implications of multicalibration. In ACM Symposium on Theory of Computing (STOC), 2024.

[CGKR25] S´ılvia Casacuberta, Parikshit Gopalan, Varun Kanade, and Omer Reingold. How global calibration strengthens multiaccuracy. In IEEE Symposium on Foundations of Computer Science (FOCS), 2025.

[CS01] Koby Crammer and Yoram Singer. On the algorithmic implementation of multiclass kernel-based vector machines. J. Mach. Learn. Res., 2:265–292, 2001.

[CT06] Thomas M. Cover and Joy A. Thomas. Elements of information theory. Wiley, 2006.

[Daw82] A. Philip Dawid. The well-calibrated bayesian. Journal of the American Statistical Association, 77:605–610, 1982.

[DKR<sup>+</sup>21] Cynthia Dwork, Michael P. Kim, Omer Reingold, Guy N. Rothblum, and Gal Yona. Outcome indistinguishability. In ACM Symposium on Theory of Computing (STOC), 2021.

[DLLT23] Cynthia Dwork, Daniel Lee, Huijia Lin, and Pranay Tankala. From pseudorandomness to multi-group fairness and back. In Conference on Learning Theory (COLT), 2023.

[DPS07] Miroslav Dud´ık, Steven J. Phillips, and Robert E. Schapire. Maximum entropy density estimation with generalized regularization and an application to species distribution modeling. Journal of Machine Learning Research, 8(44):1217–1260, 2007.

[Fel10] Vitaly Feldman. Distribution-specific agnostic boosting. In Andrew Chi-Chih Yao, editor, Innovations in Computer Science - ICS 2010, Tsinghua University, Beijing, China, January 5-7, 2010. Proceedings, pages 241–250. Tsinghua University Press, 2010.

[FK96] Alan M. Frieze and Ravi Kannan. The regularity lemma and approximation schemes for dense problems. In IEEE Symposium on Foundations of Computer Science (FOCS), 1996.

[GHK<sup>+</sup>23] Parikshit Gopalan, Lunjia Hu, Michael P. Kim, Omer Reingold, and Udi Wieder. Loss minimization through the lens of outcome indistinguishability. In Innovations in Theoretical Computer Science Conference (ITCS), 2023.

[GKR<sup>+</sup>22] Parikshit Gopalan, Adam Tauman Kalai, Omer Reingold, Vatsal Sharan, and Udi Wieder. Omnipredictors. In Innovations in Theoretical Computer Science Conference (ITCS), 2022.

[GKSZ22] Parikshit Gopalan, Michael P. Kim, Mihir Singhal, and Shengjia Zhao. Low-degree multicalibration. In Conference on Learning Theory (COLT), 2022.

[HILL99] Johan H˚astad, Russell Impagliazzo, Leonid A. Levin, and Michael Luby. A pseudorandom generator from any one-way function. SIAM Journal on Computing, 28(4):1364– 1396, 1999.

[HKRR18] Ursula H´ebert-Johnson, Michael P. Kim, Omer Reingold, and Guy N. Rothblum. <sup>´</sup> Multicalibration: Calibration for the (computationally-identifiable) masses. In International Conference on Machine Learning (ICML), 2018.

[Hol05] Thomas Holenstein. Key agreement from weak bit agreement. In ACM Symposium on Theory of Computing (STOC), 2005.

[HV25] Lunjia Hu and Salil Vadhan. Generalized and unified equivalences between hardness and pseudoentropy. In Theory of Cryptography Conference (TCC), 2025.

[Imp95] Russell Impagliazzo. Hard-core distributions for somewhat hard problems. In IEEE Symposium on Foundations of Computer Science (FOCS), 1995.

[IW97] Russell Impagliazzo and Avi Wigderson. P = BPP if E requires exponential circuits: Derandomizing the XOR lemma. In Proceedings of the 29th Annual ACM Symposium on Theory of Computing (STOC), pages 220–229. ACM, 1997.

[KK09] Varun Kanade and Adam Kalai. Potential-based agnostic boosting. In Y. Bengio, D. Schuurmans, J. Laferty, C. Williams, and A. Culotta, editors, Advances in Neural Information Processing Systems, volume 22. Curran Associates, Inc., 2009.

[KKMS08] Adam Tauman Kalai, Adam R. Klivans, Yishay Mansour, and Rocco A. Servedio. Agnostically learning halfspaces. SIAM J. Comput., 37(6):1777–1805, 2008.

[KNRW18] Michael J. Kearns, Seth Neel, Aaron Roth, and Zhiwei Steven Wu. Preventing fairness gerrymandering: Auditing and learning for subgroup fairness. In International Conference on Machine Learning (ICML), 2018.

[KS03] Adam R. Klivans and Rocco A. Servedio. Boosting and hard-core set construction. Machine Learning, 51(3):217–238, 2003.

[LDPDP97] John Laferty, Stephen Della Pietra, and Vincent Della Pietra. Statistical learning algorithms based on Bregman distances. In Proceedings of the 1997 Canadian Workshop on Information Theory, pages 77–80, Fields Institute, Toronto, Canada, 1997.

[O’D04] Ryan O’Donnell. Hardness amplification within NP. Journal of Computer and System Sciences, 69(1):68–94, 2004.

[Roc70] R. Tyrrell Rockafellar. Convex Analysis. Princeton University Press, 1970.

[Sav71] Leonard J Savage. Elicitation of personal probabilities and expectations. Journal of the American Statistical Association, 66(336):783–801, 1971.

[Sze75] Endre Szemer´edi. Regular partitions of graphs. Technical report, Stanford University, Stanford, CA, USA, 1975.

[TB07] Ambuj Tewari and Peter L. Bartlett. On the consistency of multiclass classification methods. Journal of Machine Learning Research, 8:1007–1025, 2007.

[Tre03] Luca Trevisan. List-decoding using the XOR lemma. In Proceedings of the $\not 4 \not 4 t h$ Annual IEEE Symposium on Foundations of Computer Science (FOCS), pages 126– 135. IEEE Computer Society, 2003.

[TTV09] Luca Trevisan, Madhur Tulsiani, and Salil P. Vadhan. Regularity, boosting, and eficiently simulating every high-entropy distribution. In IEEE Conference on Computational Complexity (CCC), 2009.

[vdH20] Dirk van der Hoeven. Exploiting the surrogate gap in online multiclass classification. In Advances in Neural Information Processing Systems, volume 33, 2020.

[vdHFCB21] Dirk van der Hoeven, Federico Fusco, and Nicol\`o Cesa-Bianchi. Beyond bandit feedback in online multiclass classification. In Advances in Neural Information Processing Systems, volume 34, pages 13280–13291, 2021.

[VZ12] Salil P. Vadhan and Colin Jia Zheng. Characterizing pseudoentropy and simplifying pseudorandom generator constructions. In ACM Symposium on Theory of Computing (STOC), 2012.

[VZ13] Salil P. Vadhan and Colin Jia Zheng. A uniform min-max theorem with applications in cryptography. In CRYPTO: Annual International Cryptology Conference, 2013.

[Zhe14] Colin Jia Zheng. A Uniform Min-Max Theorem and Characterizations of Computational Randomness. PhD thesis, Harvard University, 2014.

## A Additional Calculations

Lemma A.1. Given convex-concave conjugates $g : \mathcal { V } \times \mathbb { R } ^ { k } \to \mathbb { R }$ and $h : \mathcal { V } \times \Delta _ { k } \to \mathbb { R }$ , define

$$
g _ { 1 } ( y , a ) = g ( y , \pi ( a ) )
$$

and

$$
h _ { 1 } ( y , b ) = \left\{ \begin{array} { l l } { { h ( y , z ) } } & { { i f b = z - \vec { 1 } / k ~ f o r ~ s o m e ~ z \in \Delta _ { k } , } } \\ { { - \infty } } & { { o t h e r w i s e . } } \end{array} \right.
$$

Then $g _ { 1 } , h _ { 1 }$ are also convex-concave conjugates.

Proof. We calculate the conjugate of $h _ { 1 }$ as follows:

$$
\begin{array} { r l } & { \underset { b \in \mathbb { R } ^ { k } } { \operatorname* { s u p } } h _ { 1 } ( y , b ) - \langle a , b \rangle = \underset { z \in \Delta _ { k } } { \operatorname* { s u p } } h ( y , z ) - \langle a , z - \vec { 1 } / k \rangle } \\ & { \qquad = \underset { z \in \Delta _ { k } } { \operatorname* { s u p } } h ( y , z ) - \langle \pi ( a ) , z \rangle } \\ & { \qquad = g ( y , \pi ( a ) ) . } \end{array}
$$

Lemma A.2. The indistinguishability constraint functionals (Definition 3.1) are conjugates.

Proof. We first recall the definitions of R and S:

$$
\begin{array} { r l } & { R ( a ) = \underset { \lambda \in \mathbb { R } ^ { \mathcal { C } } } { \operatorname* { i n f } } \varepsilon \Vert \lambda \Vert _ { 1 } \mathrm { ~ s u c h ~ t h a t ~ P r } \Big [ a ( \pmb { x } , \pmb { y } ) = \sum _ { i } \lambda _ { i } c _ { i } ( \pmb { x } ) \Big ] = 1 } \\ & { S ( b ) = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ } \vert \mathbb { E } [ \langle c ( \pmb { x } ) , b ( \pmb { x } , \pmb { y } ) \rangle ] \vert \leqslant \varepsilon \mathrm { ~ f o r ~ a l l ~ } c \in \mathcal { C } , } \\ { - \infty } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}
$$

Note that $R ( a ) = + \infty$ if a $\notin$ span(C). To verify that R and S are conjugates, we first write down the expression for the conjugate of $R ,$ as described in Definition 2.6:

$$
\operatorname* { i n f } _ { \lambda } \varepsilon \| \lambda \| _ { 1 } + \mathbb { E } [ \langle c _ { \lambda } ( \pmb { x } ) , b ( \pmb { x } , \pmb { y } ) \rangle ] .
$$

This quantity is clearly −∞ if there exists any $c ^ { * } \in \mathcal { C }$ with $| \mathbb { E } [ \langle c ^ { * } ( { \pmb x } ) , b ( { \pmb x } , { \pmb y } ) \rangle ] | > \varepsilon$ . Indeed, just let the coeficient vector λ assign increasingly large positive (or negative) values to the coordinate corresponding to $c ^ { * }$ , while assigning 0 to all other coordinates. Otherwise, if $| \mathbb { E } [ \langle c ( { \pmb x } ) , b ( { \pmb x } , { \pmb y } ) \rangle ] | \leqslant \varepsilon$ for every $c \in { \mathcal { C } }$ , the overall quantity is 0, which is attained by the optimizer $\lambda = 0$ . This is precisely the function S that we just defined. □

Lemma A.3. The indistinguishability penalty functionals (Definition 3.2) are conjugates.

Proof. We first recall the definitions of R and $S { \mathrm { : } }$

$$
\begin{array} { r l } & { R ( a ) = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ } \operatorname* { P r } [ a ( { \pmb x } , { \pmb y } ) = c ( { \pmb x } ) ] = 1 \mathrm { ~ f o r ~ s o m e ~ } c \in \overline { { \mathscr C } } , } \\ { + \infty } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \\ & { S ( - b ) = - \underset { c \in \overline { { \mathscr C } } } { \operatorname* { m a x } } \mathbb { E } [ \langle c ( { \pmb x } ) , b ( { \pmb x } , { \pmb y } ) \rangle ] . } \end{array}
$$

To verify that R and S are conjugates, we simply write down the expression for the conjugate of $R ,$ as described in Definition 2.6. Doing so gives min $\mathsf { i } _ { c \in \overline { { \mathcal { C } } } } \mathbb { E } [ \langle c ( \pmb { x } ) , b ( \pmb { x } , \pmb { y } ) \rangle ]$ , which is just $S ( b )$ for the function $S$ that we just defined. □

Lemma A.4. Let $p , q > 1$ satisfy $1 / p + 1 / q = 1$ . Then $\ell _ { p } ( y , a ) = ( 1 / p ) | y - a | ^ { p }$ and

$$
\begin{array} { r } { h ( y , z ) = \bigg \{ \mu - \frac 1 q | \mu | ^ { q } \quad i f z = \mu y \ f o r \ s o m e \ \mu \in \mathbb { R } } \end{array}
$$

are convex-concave conjugates, assuming $y \in \{ \pm 1 \}$

Proof. To see that these are conjugates, let us calculate

$$
\operatorname* { s u p } _ { b \in \mathbb { R } } h ( y , b ) - a b = \operatorname* { s u p } _ { \mu \in \mathbb { R } } \mu - { \frac { 1 } { q } } | \mu | ^ { q } - a \cdot ( \mu y )
$$

Taking the derivative of the right-hand side objective, we see that it is maximized at the point $\mu ^ { \star } = \mathrm { s i g n } ( 1 - a y ) | 1 - a y | ^ { 1 / ( q - 1 ) }$ . Completing the calculation with $\mu ^ { \star }$ , we see that

$$
\begin{array} { r l } & { \underset { i \in \mathbb { R } } { \operatorname* { s u p } } h ( y , b ) - a b = \mu ^ { \star } - \frac { 1 } { q } | \mu ^ { \star } | ^ { q } - a \cdot ( \mu ^ { \star } y ) , } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad  \end{array}
$$

where, in the last step, we used the fact that $y \in \{ \pm 1 \}$

Next, we give an alternate proof of Theorem 1.16 in the special case of surrogates for the 0-1 loss. This proof is quite a bit more intuitive than the one we gave in Section 5, and it also clarifies the structure of the optimal rounding rule. As in our first proof, the non-trivial direction is to show the existence of a rounding scheme that matches the entropy gap. Before starting the proof, we make some simplifying observations.

• We have not assumed that the losses are non-negative, but when $\ell _ { B } = \ell _ { 0 - 1 }$ , it is an implicit requirement for either inequality to hold with $\beta > 0$ , since both $h _ { 0 - 1 }$ and $\ell _ { 0 - 1 }$ are non-negative. For instance, if $\ell ( y , a ) < 0$ , then clearly $h ( y ) \leq \ell ( y , a ) < 0$ , so ℓ cannot be a β-surrogate for $\ell _ { 0 - 1 }$ for any $\beta > 0$ . This is not a strong restriction since natural losses and entropies are non-negative.

• Scaling the loss $\ell = \ell _ { A }$ by a constant positive factor also scales the entropy h. So after scaling both by $1 / \beta$ , we may in fact assume $\beta = 1$

• The non-trivial direction is to come up with a rounding scheme that matches the entropy gap. With the rescaling $\beta = 1$ , the guarantee $\ell ( y , a ) \geq \mathbb { E } _ { { \tilde { y } } \sim R ( a ) } [ \ell _ { 0 - 1 } ( y , { \tilde { y } } ) ]$ becomes trivial if $\ell ( y , a ) \geq 1$ . This suggests that the rounding $R ( a )$ focus on labels where $\ell ( i , a ) \in [ 0 , 1 ]$

• Since the probability $R ( a ) _ { i }$ should decrease with loss, this naturally suggests the quantities $( 1 - \ell ( i , a ) ) _ { + }$ . Indeed, the proof will show that they can be used as probabilities. The rest of the mass can be distributed arbitrarily.

Alternate Proof of Theorem 1.16 when $\ell _ { B } = \ell _ { 0 - 1 }$ . As in our previous proof of Theorem 1.16, the surrogate loss relation implies the entropy relation by monotonicity. In the other direction, assume $\ell = \ell _ { A }$ and $\ell _ { 0 - 1 }$ satisfy $h ( p ) \geqslant \beta \cdot h _ { 0 - 1 } ( p )$ for $\beta = 1$ . For each $i \in [ k ]$ , define the quantity

$$
q _ { i } = ( 1 - \ell ( i , a ) ) _ { + }
$$

and let $Q = \textstyle \sum _ { i } q _ { i }$ . We first show that $Q \leq 1$ . Let $S = \{ i : \ell ( e _ { i } , a ) < 1 \}$ , and let $m = | S |$ . We may assume S is non-empty, else $\ell ( i , a ) \geq 1$ for all i and the claimed bound holds trivially. We consider $p = u _ { S }$ , the uniform distribution over S. Then,

$$
1 - \frac { 1 } { m } = h _ { 0 - 1 } ( \boldsymbol { u } _ { S } ) \leq h ( \boldsymbol { u } _ { S } ) \leq \mathbb { E } _ { \boldsymbol { y } \sim \boldsymbol { u } _ { S } } [ \ell ( \boldsymbol { y } , \boldsymbol { a } ) ] = \frac { 1 } { m } \sum _ { i \in S } \ell ( \boldsymbol { i } , \boldsymbol { a } ) .
$$

Hence

$$
\sum _ { i \in S } \ell ( i , a ) \geq m - 1 .
$$

We use this to bound

$$
Q = \sum _ { i } ( 1 - \ell ( i , a ) ) _ { + } = \sum _ { i \in S } ( 1 - \ell ( i , a ) ) = m - \sum _ { i \in S } \ell ( i , a ) \leq 1 .
$$

Our rounding $R ( a )$ will distribute $Q$ probability mass according to $q _ { i }$ and the remaining $1 - Q$ mass arbitrarily, so that $R ( a ) _ { i } \geq q _ { i }$ for every $i \in [ k ]$ . So we can bound the 0-1 loss by

$$
\ell _ { 0 - 1 } ( i , R ( a ) ) = 1 - R ( a ) _ { i } \leq 1 - q _ { i } \leq \ell ( i , a ) .
$$

where the last inequality holds since

$$
\begin{array} { r } { 1 - q _ { i } = 1 - ( 1 - \ell ( i , a ) ) _ { + } \leq \ell ( i , a ) . } \end{array}
$$

Proof of Lemma 5.3.

• For 0-1 loss, it’s clear the best action is $\operatorname { a r g m a x } _ { j } p _ { j }$ , which incurs loss $1 - \operatorname* { m a x } _ { j } p _ { j }$

• For weak hinge loss, let $j ^ { \star } = \mathrm { a r g m i n } _ { j } p _ { j }$ be the least likely label. The action $a = \vec { 1 } - k e _ { j } ,$ incurs loss k min<sub>j</sub> $p _ { j }$ . To see that this is optimal, consider any $a \in \mathbb { R } ^ { k }$ with mean 0. Then,

$$
\sum _ { i = 1 } ^ { k } p _ { i } ( 1 - a _ { i } ) _ { + } \geqslant \operatorname* { m i n } _ { j } p _ { j } \cdot \sum _ { i = 1 } ^ { k } ( 1 - a _ { i } ) _ { + } \geqslant \operatorname* { m i n } _ { j } p _ { j } \sum _ { i = 1 } ^ { k } ( 1 - a _ { i } ) = k \operatorname* { m i n } _ { j } p _ { j } .
$$

• For strong hinge loss, let $S = \{ i \in [ k ] \mid p _ { i } > 1 / k \}$ and consider the action $a = \mathbf { 1 } _ { S } \in \{ 0 , 1 \} ^ { k }$ After subtracting $| S | / k$ from each coordinate, it has mean $0$ and incurs loss

$$
\sum _ { i = 1 } ^ { k } p _ { i } \operatorname* { m a x } _ { j } { ( e _ { i } - a ) _ { j } } + \frac { | S | } { k } = \sum _ { i \notin S } p _ { i } + \frac { | S | } { k } .
$$

This is just $\textstyle \sum _ { i }$ min $\{ p _ { i } , u _ { i } \}$ where u is the uniform distribution, or equivalently $1 - d _ { \mathrm { T V } } ( p , u )$ To see that this is tight, for any $a \in \mathbb { R } ^ { k }$ , define the margins $t _ { i } = a _ { i } - \operatorname* { m i n } _ { j } a _ { j }$ . It is clear that $a , t \in \mathbb { R } ^ { k }$ incur the same loss. Using the fact that $t _ { i } \geqslant 0$ , this loss is

$$
\sum _ { i = 1 } ^ { k } p _ { i } \operatorname* { m a x } _ { j } ( e _ { i } - t ) _ { j } + { \frac { 1 } { k } } \sum _ { i = 1 } ^ { k } t _ { i } = \sum _ { i = 1 } ^ { k } \Bigl ( p _ { i } ( 1 - t _ { i } ) _ { + } + { \frac { t _ { i } } { k } } \Bigr ) .
$$

Each summand is minimized at either $t _ { i } = 0$ or $t _ { i } = 1$ , and picking the better of the two yields $\sum _ { i } \operatorname* { m i n } \{ p _ { i } , u _ { i } \}$ , as claimed.

• For Crammer-Singer hinge loss, consider any $a \in \mathbb { R } ^ { k }$ and let $j ^ { \star } \in \mathrm { a r g m a x } _ { j } a _ { j }$ and let $m =$ $a _ { j ^ { \star } } - \operatorname* { m a x } _ { j \neq j ^ { \star } } a _ { j }$ be the margin between the top two coordinates of $a .$ . If the label is $j ^ { \star }$ , then a incurs loss $( 1 - m ) _ { + }$ , and otherwise a incurs loss $1 + m$ or more. Therefore,

$$
\sum _ { i = 1 } ^ { k } p _ { i } \ell _ { \mathrm { C S } } ( e _ { i } , a ) \geqslant p _ { j ^ { \star } } ( 1 - m ) _ { + } + ( 1 - p _ { j ^ { \star } } ) ( 1 + m ) .
$$

As a function of $m \geqslant 0$ , this is minimized at either $m = 0 \ \mathrm { o r } \ m = 1$ , which yield loss values 1 and $2 ( 1 - p _ { j ^ { \star } } )$ , respectively, attained by $a = 0$ and $a = e _ { j ^ { \star } }$ after centering. □

Lemma A.5. $I f g , h : \mathbb { R } ^ { k }  \overline { { \mathbb { R } } }$ are convex-concave conjugates, then so are

$$
g _ { 1 } ( y , a ) = g ( - a ) - \langle y , a \rangle
$$

and

$$
h _ { 1 } ( y , b ) = h ( y - b ) .
$$

Proof. We calculate the conjugate of $h _ { 1 }$ as follows:

$$
\begin{array} { r l } { } & { \underset { b \in \mathbb { R } ^ { k } } { \operatorname* { s u p } } h _ { 1 } ( y , b ) - \langle a , b \rangle = \underset { z \in \mathbb { R } ^ { k } } { \operatorname* { s u p } } h ( z ) - \langle a , y - z \rangle } \\ & { \qquad = \underset { z \in \mathbb { R } ^ { k } } { \operatorname* { s u p } } \big ( h ( z ) + \langle a , z \rangle \big ) - \langle y , a \rangle } \\ & { \qquad = g ( - a ) - \langle y , a \rangle . } \end{array}
$$

By definition, this is precisely $g _ { 1 }$ .