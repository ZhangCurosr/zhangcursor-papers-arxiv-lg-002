# Improved regret bounds in tabular RL under local differential privacy

Hugo Richard

Joint team Fairplay, Criteo AI Labs, ENSAE CREST, INRIA

France

research.hugo.richard@gmail.com

## Abstract

We study regret minimization for time-homogeneous episodic tabular reinforcement learning with S states and A actions under non-interactive ϵ-local differential privacy (LDP). In each episode $k \in [ K ]$ , the decision-maker sends a policy to a user, the user executes the policy for H steps, and sends an ϵ-locally differentially private message back to the decision-maker. When H = 1, the problem reduces to an LDP contextual bandit with S contexts, A arms and horizon K. We prove that the minimax regret is min $\begin{array} { r l } {  { ( K , \frac { \sqrt { \operatorname* { m a x } ( S , \exp ( \epsilon ) + 1 ) S A K ( \exp ( \epsilon ) + 1 ) } } { \exp ( \epsilon ) - 1 } ) } } & { { } } \end{array}$ , up to constants and a $\sqrt { \log ( A ) }$ factor. The upper bound is achieved by an adaptation of EXP3 and a unary encoding mechanism. When $H > 1$ , and for small $\epsilon ,$ we show a lower bound in $\Omega ( H S \frac { \sqrt { A K } } { \epsilon } )$ and an upper bound in $\tilde { O } ( H ^ { 2 } S ^ { 2 } \frac { \sqrt { A K } } { \epsilon } )$ . The upper bound is achieved by an algorithm based on UCBVI and a unary encoding mechanism. For small $\epsilon ,$ these bounds improve the previously known lower bound by a factor $\sqrt { S }$ and the previously known upper bound by a factor of $H { \sqrt { A } }$

## 1 Introduction

Reinforcement learning (RL) is the problem of learning to make decisions through interaction. Typical applications of RL include recommender systems, online advertising, healthcare, routing or robotics Kober et al. [2013], Afsar et al. [2023], Cai et al. [2017], Yu et al. [2023], Nazari et al. [2018]. The entity running the RL algorithm is called the decision-maker, which could be a dating application, an ad displaying platform or a scientist running a medical experiment. In the classical RL framework, an interaction proceeds as follows. The decision-maker observes the current state which could be the profile of a user, the context of an ad or the health status of the subject and decides on an action to take based on the observed state which could be a partner recommendation, the display of an ad or the choice of treatment to give to a subject. Conditionally on the state and action, the decision-maker receives a random numerical reward which could be a like of the recommended person, a click on the displayed ad or a numerical measure of the subject’s health. The next state is a random function of the current state and action. The goal of reinforcement learning is to optimize the policy used by the decision-maker to choose actions from previous observations so that the sum of rewards gathered over time is maximized.

In classical RL, the focus is on the decision-maker and the constraints of the user, which could be a client of a dating application, a viewer of the ad, or a subject of the medical experiment are not taken into account. A primary concern of users in the above applications is privacy. A user may not want to publicly disclose their activity on a dating application, their browsing history, or their health condition. While the decision-maker may be trustworthy, it can be hacked or leak data by accident. For instance, the dating application Ashley Madison was hacked in 2015 exposing the data of 32 million users Jones [2016]. Another example is the UK Biobank data containing health and genetic information of 500 000 subjects that was partially uploaded on Github by accident Rocher and Morley [2026].

While the UK Biobank data is de-identified, meaning the personal information such as the name or the address of the subjects is removed, journalists from the Guardian were able to re-identify some subjects Devlin and Burgis [2026]. The fact that removing personal information is not sufficient to protect the privacy of users is well-documented Sweeney [2000], Narayanan and Shmatikov [2008], Rocher et al. [2019] and justifies the use of more rigorous privacy frameworks such as local differentially privacy (LDP) Duchi et al. [2013].

The above considerations motivate the study of reinforcement learning under LDP. We consider a tabular RL problem with S states and A possible actions, where the decision-maker learns an optimal policy by interacting with K different users sequentially. The transition kernel—the probabilities of moving from one state to another when an action is taken—and reward distributions are fixed (but unknown) and the same across all users. Each user interaction corresponds to one episode, where the initial state varies from users to users. When interacting with a user the decision-maker sends a policy to the user who executes it for H steps and applies a randomized mechanism to privatize its data before sending the privatized data back to the decision-maker. The mechanism is chosen by the decision-maker but must satisfy ϵ-LDP constraints which guarantee a limit on the amount of information that the decision-maker - or hackers - can learn about the user.

Our contributions We first study the case H = 1. We prove that the minimax regret rate is given by

$$
\operatorname* { m i n } \bigg ( K , \frac { \sqrt { \operatorname* { m a x } ( S , \exp ( \epsilon ) + 1 ) S A K ( \exp ( \epsilon ) + 1 ) } } { \exp ( \epsilon ) - 1 } \bigg )
$$

up to constants and a $\sqrt { \log ( A ) }$ factor. This result is a consequence of Theorem 1 showing the lower bound and Theorem 2 providing an algorithm and a mechanism whose regret matches the lower bound up to constants and a $\sqrt { \log ( A ) }$ factor.

Second, we study the case $H > 1$ . A reduction from the bandit case implies a new lower bound shown in Corollary 1 improving upon previous work Garcelon et al. [2021] uniformly across all parameters. The improvement is particularly significant when $\epsilon \leq 1$ , gaining, in this case, a factor $\sqrt { S }$ over the previously known lower bound. Then, we present Theorem 3 which provides an algorithm and a mechanism whose regret, for $\epsilon \leq 1$ , is in $\tilde { O } ( \frac { \overline { { H ^ { 2 } S ^ { 2 } \sqrt { A K } } } } { \epsilon } )$ improving, in this high privacy regime, over the previous upper bound [Garcelon et al., 2021, Theorem 5] by a factor of $H { \sqrt { A } }$ . We emphasize that, for $\epsilon \leq 1$ , a multiplicative gap of order HS remains between this upper bound and the lower bound of Corollary 1, which is in $\Omega ( \frac { H S \sqrt { A K } } { \epsilon } )$ ). Closing this gap is left as an open question (see Section 6).

## 2 Related Work

Regret minimization for non-private time-homogeneous episodic tabular RL is well studied. A lower bound for the minimax regret of order $\Omega ( H { \sqrt { S A K } } )$ is obtained by Azar et al. [2017] thanks to a reduction to contextual bandits. Many works Azar et al. [2017], Dann et al. [2019], Zanette and Brunskill [2019], Zhang et al. [2021] study regret minimization in RL and various algorithms have been proposed. The best upper bound up to date is given by a model-based algorithm that achieves a regret in $\tilde { O } ( H \sqrt { S A K } + H S ^ { 2 } A )$ up to log factors Zhang et al. [2021]. This result shows that tabular RL is essentially as hard as contextual bandits. Whether a model-free algorithm can achieve a leading rate of $H { \sqrt { S A K } }$ is an active area of research Jin et al. [2018], Zhang et al. [2020].

The combination of online decision-making and local differential privacy is well studied in the bandits case. In classical multi-armed bandits, the optimal regret under ϵ-LDP is in $\Theta ( \operatorname* { m a x } ( \frac { \sqrt { A K } } { \epsilon } , \sqrt { A K } ) )$ Ren et al. [2020]. We recover this rate up to constants and log factor by setting $S \stackrel { \triangledown } { = } 1$ in Theorem 1 and Theorem 2.

The works most related to ours are those studying RL under LDP constraints Garcelon et al. [2021], Qiao and Wang [2023], Chowdhury and Zhou [2022b]. The work of Garcelon et al. [2021] studies the same time-homogeneous setting as us, while Qiao and Wang [2023] and Chowdhury and Zhou [2022b] study the time-varying case where rewards and transition matrices change over time. Since their bounds are for this more general setting, they are loose if applied directly. Our main point of comparison is therefore Garcelon et al. [2021]. We improve the lower bound obtained in Garcelon et al. [2021] and give an upper bound improving that of Garcelon et al. [2021] in the high privacy regime $\epsilon \leq H \sqrt { A }$

Although tabular contextual bandit under LDP has never been studied specifically, some generalizations were considered in the literature. Contextual linear bandits under LDP have been studied in Zheng et al. [2020]. After embedding the reward using one-hot features and rescaling so that the unit-norm assumption on the parameter is satisfied, the guarantee of Zheng et al. gives in the tabular setting a rate in $\overset { \cdot } { \tilde { O } } \big ( \frac { ( S A ) ^ { 5 / 4 } K ^ { 3 \widehat { / } 4 } } { \epsilon } \big )$ , which is suboptimal. If contexts are chosen i.i.d., then, under some assumptions, an $O ( K ^ { 1 / 2 } )$ rate can be recovered Han et al. [2021], Pavlovic et al. [2025]. The setting we study differs in that there are no restrictions on how contexts are chosen. Another generalization is contextual nonparametric bandits under LDP, which have been studied in Ma et al. [2025], where the authors obtain sublinear regret under regularity assumptions (Assumptions 2.2 and 2.3) and a margin assumption (Assumption 2.4). These assumptions do not necessarily hold in the setting we study. Lastly, a related problem is LDP contextual-bandit pricing Zhao et al. [2026], Chen et al. $[ 2 0 2 3 ] .$ , where the price is the action and the revenue is the reward. However, none of the results of Zhao et al. [2026] or Chen et al. [2023] apply in the setting we consider because of incompatible assumptions. Notably, in Zhao et al., the reward has a specific parametric form derived from a GLM assumption, and in Chen et al., the reward is assumed to be strongly concave. In contrast, none of these restrictions are imposed in the setting we study.

Besides LDP, other notions of privacy have been studied in bandits and RL. In global differential privacy, the decision-maker is trusted but the sequence of action it takes must not reveal too much information about the users’ reward Azize and Basu [2022], Azize et al. [2026]. This setting is only relevant for multi-armed bandits since in contextual bandits and RL, a linear regret is unavoidable Shariff and Sheffet [2018]. This motivates the notion ofjoint differential privacy Shariff and Sheffet [2018], Vietri et al. [2020] which is weaker. It guarantees that the actions taken by the decision-maker at time t do not reveal too much information about the users’ information up to time $t ^ { \prime } < t .$ . Intermediate notions based on shuffling have also been studied for linear contextual bandits Chowdhury and Zhou [2022a], Garcelon et al. [2022].

## 3 Episodic Finite-Horizon Tabular RL under LDP

Notations For an integer $^ { n , }$ we use $[ n ]$ to denote the set $\left\{ 1 , 2 , \ldots , n \right\}$ . For a finite set $x ,$ , we use $\Delta ( \mathcal { X } )$ to denote the set of functions $\nu : \mathcal { \bar { X } } \to [ 0 , 1 ]$ such that $\begin{array} { r } { \sum _ { x \in \mathcal { X } } \nu ( \overline { { x } } ) = 1 } \end{array}$ and denote by $\nu ( \cdot )$ the corresponding distribution over $\mathcal { X }$ . We use $f ( x ) = \Omega ( g ( x ) )$ to denote that there exists a constant $C > 0$ such that $f ( x ) \geq C g ( x ) , f ( x ) = O ( { \dot { g } } ( x ) )$ ) to denote that there exists a constant $C > 0$ such that $f ( x ) \leq C g ( x )$ and $f ( x ) = \Theta ( g ( x ) ) { \mathrm { ~ i f ~ } } f ( x ) = \Omega ( g ( x ) )$ and $f ( x ) = O ( g ( x ) )$ . We use ${ \tilde { O } } ( \cdot )$ $\tilde { \Omega } ( \cdot )$ and $\bar { \Theta } ( \cdot )$ to hide constants and logarithmic factors. We denote $B ( p )$ the Bernoulli distribution with parameter $p .$ $\begin{array} { r } { D _ { \mathrm { K L } } ( \nu , \nu ^ { \prime } ) = \mathbb { E } _ { \nu } \left[ \log \frac { d \nu } { d \nu ^ { \prime } } \right] } \end{array}$ and $D _ { \mathrm { T V } } ( \nu , \nu ^ { \prime } ) = \operatorname* { s u p } _ { A } | \nu ( A ) - \nu ^ { \prime } ( A ) |$ denote the Kullback-Leibler divergence and total variation distance between distributions ν and $\nu ^ { \prime }$

Non-private interaction, value functions and regret A Markov Decision Process (MDP) is defined by the tuple $I = ( \mathcal { S } = [ S ] , \mathcal { A } = [ A ] , P , r , ( \rho ^ { \check { k } } ) _ { k = 1 } ^ { K } )$ , where $s$ is the set of S states, $A = [ A ]$ the set of A actions, $P : \mathcal { S } \times \mathcal { A }  \Delta ( \mathcal { S } )$ is the transition kernel, $r : S \times A  [ 0 , 1 ]$ is the expected rewardfunction and $\rho ^ { k } \in \Delta ( \mathcal { S } )$ is the distribution of the starting state at episode k for each $k \in [ K ]$ We call ${ \mathfrak { M } } ( S , A , K )$ , or simply M when the context is clear, the set of MDP with $S$ states, A actions and K episodes. The above description corresponds to the tabular case where the number of states, actions and episodes are finite.

In the non-private setting, the decision-maker interacts directly with the MDP. The interaction consists of $K$ episodes each consisting of H steps. At episode $k ,$ the decision maker selects a policy $\pi ^ { k } =$ $( \pi _ { h } ^ { k } : \dot { S }  \Delta ( \mathcal { A } ) ) _ { h \in [ H ] }$ and samples a starting state $S _ { 1 } ^ { k }$ from $\rho ^ { k }$ . Then, at step $h \in [ H ]$ , the decision maker observes $S _ { h } ^ { k } \in \mathcal { S }$ , samples an action $A _ { h } ^ { k } \in \mathcal { A }$ from $\pi _ { h } ^ { k } ( S _ { h } ^ { k } , \cdot )$ , obtains a reward $R _ { h } ^ { k } \in [ 0 , 1 ]$ and transitions to $\bar { S } _ { h + 1 } ^ { k } \in \mathcal { S }$ . The reward $R _ { h } ^ { k }$ and the next state $S _ { h + 1 } ^ { k }$ are sampled independently conditionally on $( S _ { h } ^ { k } , A _ { h } ^ { k } )$ , with $\mathbb { E } [ R _ { h } ^ { k } \ \vert \ S _ { h } ^ { k } , A _ { h } ^ { k } ] = r ( S _ { h } ^ { k } , A _ { h } ^ { k } )$ and $\begin{array} { r } { \dot { S } _ { h + 1 } ^ { k } \ \vert \ S _ { h } ^ { k } , A _ { h } ^ { k } \sim P ( S _ { h } ^ { k } , A _ { h } ^ { k } , \cdot ) } \end{array}$ The trajectory of episode k is denoted by $\tau ^ { k } = ( S _ { 1 } ^ { k } , A _ { 1 } ^ { k } , R _ { 1 } ^ { k } , \dots , S _ { H } ^ { k } , A _ { H } ^ { k } , R _ { H } ^ { k } , S _ { H + 1 } ^ { k } ) \in \mathbb { T }$ where $\mathbb { T } = ( \mathcal { S } \times \mathcal { A } \times [ 0 , 1 ] ) ^ { H } \times \mathcal { S }$ is the set of possible trajectories. The value of a policy π at step h on MDP I at state s is defined as

$$
V _ { h } ^ { I , \pi } ( s ) = \mathbb { E } \left[ \sum _ { i = h } ^ { H } R _ { i } ^ { k } \mid S _ { h } ^ { k } = s \right] .
$$

where the expectation is taken over the trajectory $\tau ^ { k }$ . We often omit the superscript I when it is clear from the context. The optimal value function is defined as $V _ { h } ^ { * } ( s ) = \operatorname* { m a x } _ { \pi } V _ { h } ^ { I , \pi } ( s )$ . Lastly, note that the policies we use depend only on the current state and step. It can be shown that using history-dependent policies instead does not increase the optimal value function [Puterman, 2014, Theorem 4.4.2]. We call the process by which the decision maker updates policies from episode to episode the (non-private) algorithm $\mathfrak { i } \bar { \mathfrak { i } } = ( \mathfrak { A } _ { k } : \mathbb { T } ^ { k - 1 } \to ( S \to \Delta ( \bar { \mathcal { A } } ) ) ^ { H } ) _ { k \in [ K ] } ^ { }$ and call $\mathbb { \hat { A } } ^ { \mathrm { p u b l i c } }$ the set of non-private algorithms. where ${ \mathfrak { A } } _ { k }$ maps the past $k - 1$ trajectories $( \tau ^ { 1 } , \dot { \dots } , \tau ^ { k - 1 } )$ to the policy $\pi ^ { k }$

The performance of an algorithm A on an MDP I is computed by the regret defined as

$$
\mathcal { R } ^ { \mathrm { p u b l i c } } ( \mathfrak { A } , I ) = \mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \left( V _ { 1 } ^ { I , * } ( S _ { 1 } ^ { k } ) - V _ { 1 } ^ { I , \pi ^ { k } } ( S _ { 1 } ^ { k } ) \right) \right] , \quad \pi ^ { k } \sim \mathfrak { A } _ { k } ( \tau ^ { 1 } , \dots , \tau ^ { k - 1 } ) ,
$$

and the difficulty of the problem is quantified by the minimax regret defined as

$$
{ \mathfrak { R } } ^ { \mathrm { p u b l i c } } = \operatorname* { i n f } _ { \mathfrak { A } \in { \mathbb { A } } ^ { \mathrm { p u b l i c } } } \operatorname* { s u p } _ { I \in { \mathfrak { M } } } { \mathcal { R } } ( \mathfrak { A } , I ) .
$$

Private interaction and local differential privacy In the setting we study, the user does not allow the decision maker to access its true data. The algorithm will therefore not have access to the trajectories but only to a version privatized by a mechanism $\mathcal { M } : \mathbb { T } \to \Delta ( \mathcal { Y } )$ satisfying the following ϵ-LDP guarantee: for any two trajectories $\tau , \tau ^ { \prime }$ and any measurable subset $Y \subset \mathcal { Y }$ , we must have

$$
\mathbb { P } ( \mathcal { M } ( \tau ) \in Y ) \le e ^ { \epsilon } \mathbb { P } ( \mathcal { M } ( \tau ^ { \prime } ) \in Y ) .
$$

We call $\mathbb { M } _ { \epsilon }$ the set of such $\mathrm { \epsilon - L D P }$ mechanisms.

The interaction then proceeds as follows. The decision maker first selects a privacy mechanism $\mathcal { M } .$ . Then, at episode k, the decision maker selects a policy $\pi ^ { k }$ and sends it to user $k ,$ who executes the policy $\pi ^ { k }$ for H steps and observes the resulting trajectory $\tau ^ { k }$ which is then privatized by the mechanism M to produce a privatized statistic $M ^ { k } = \dot { \mathcal { M } } ( \tau ^ { k } )$ sent back to the decision maker.

The process by which the decision maker updates policies from episode to episode is now characterized by a private algorithm $\mathfrak { A } = ( \mathfrak { A } _ { k } : \dot { \mathfrak { V } } ^ { k - 1 } \overset { \cdot } {  } ( \mathcal { S }  \Delta ( \dot { \mathfrak { A } } ) ) ^ { H } ) _ { k \in [ K ] }$ and the set of private algorithms is denoted by $\mathbb { A } ^ { \mathrm { p r i v a t e } }$ . Here $\mathcal { V } ^ { k - 1 }$ is the Cartesian-product space containing all possible sequences of $k - 1$ private messages: at episode $k ,$ , the realized input to the algorithm is the history $( \bar { M ^ { 1 } } , \dotsc , M ^ { k - 1 } ) \in ^ { \dot { \gamma } k - 1 }$

Furthermore, the performance of the decision maker is now determined by the choice of mechanism $\mathcal { M }$ and the private algorithm A. We still measure the performance by the regret given by

$$
\mathcal { R } ( \mathfrak { A } , I , \mathcal { M } ) = \mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \left( V _ { 1 } ^ { I , * } ( S _ { 1 } ^ { k } ) - V _ { 1 } ^ { I , \pi ^ { k } } ( S _ { 1 } ^ { k } ) \right) \right] , \quad M _ { 1 } , \dots , M _ { K } \sim \mathcal { M } ( \tau ^ { 1 } ) , \dots , \mathcal { M } ( \tau ^ { K } )
$$

and the minimax regret is defined as

$$
\mathfrak { R } = \operatorname* { i n f } _ { \mathfrak { A } \in \mathbb { A } ^ { \mathrm { p r i v a t e } } } \operatorname* { s u p } _ { I \in \mathfrak { A } } \mathcal { R } ( \mathfrak { A } , I , \mathcal { M } ) .
$$

In the following sections, we present upper and lower bounds on $\Re$

## 4 Private contextual bandits

The current lower bounds available in the literature in either the non-private Azar et al. [2017] or private Garcelon et al. [2021] case use a reduction to contextual bandits. It is therefore natural to first study this simpler case. We therefore set in this section $H = 1$ . In this case, the transition kernel P plays no role and since A and S are fixed, an instance of contextual bandits is characterized by the reward function r and the initial distributions $\{ \rho ^ { k } \} _ { k = 1 } ^ { K }$

## 4.1 Lower bound

In Theorem 1, we show a lower bound on R.

Theorem 1. Assume $A \geq 2 ,$ , for any ϵ-LDP mechanism M and any algorithm A that the decision maker can choose, there exists an instance I characterized by a reward function r and initial distributions $\{ \rho ^ { k } \} _ { k = 1 } ^ { K }$ such that the regret is lower bounded as

$$
\mathcal { R } ( \mathfrak { A } , \mathcal { M } , I ) \geq \frac { 1 } { 3 2 } \operatorname* { m i n } \left( K , \frac { \sqrt { \operatorname* { m a x } ( S , \exp ( \epsilon ) + 1 ) S A K ( \exp ( \epsilon ) + 1 ) } } { \exp ( \epsilon ) - 1 } \right)
$$

Proof sketch (see Section A for the full proof). Fix an algorithm A, a mechanism M and a gap $\Delta >$ 0. For each $\boldsymbol { \phi } \in \mathcal { A } ^ { s }$ and $\sigma \in { S } ^ { K }$ , define the hard instance $I _ { \phi , \sigma }$ by

$$
r _ { \phi , \sigma } ( s , a ) = \frac { 1 } { 2 } + \Delta \mathbb { 1 } \{ a = \phi ( s ) \} , \qquad \rho _ { \phi , \sigma } ^ { k } ( s ) = \mathbb { 1 } \{ s = \sigma _ { k } \} .
$$

Let $I _ { \alpha , c }$ be sampled uniformly from this family and let $I _ { s } ^ { 0 }$ be the same random instance with all rewards at state s replaced by $1 / 2$ . A standard change-of-measure argument lower bounds the expected regret by the gap times the probability of not identifying the best action, and upper bound this identification probability through $D _ { \mathrm { K L } } ( ( \dot { M } _ { 1 } , \dots , M _ { K } ) _ { s , 0 } \dot { \lVert ( M _ { 1 } , \dots , M _ { K } ) _ { s , j } ) }$

The improvement comes from controlling this KL term using both LDP contractions of [Asoodeh and Zhang, 2024, Theorems 1 and 2], namely

$$
D _ { \mathrm { K L } } ( M _ { 1 } \| M _ { 2 } ) \leq \frac { ( \exp ( \epsilon ) - 1 ) ^ { 2 } } { ( \exp ( \epsilon ) + 1 ) ^ { 2 } } D _ { \mathrm { K L } } ( \tau _ { 1 } \| \tau _ { 2 } ) ,
$$

$$
D _ { \mathrm { K L } } ( M _ { 1 } \Vert M _ { 2 } ) \leq \frac { ( \exp ( \epsilon ) - 1 ) ^ { 2 } } { \exp ( \epsilon ) } D _ { \mathrm { T V } } ( \tau _ { 1 } , \tau _ { 2 } ) ^ { 2 } .
$$

Combining these inequalities with the chain rule yields the two regimes in the bound. Optimizing $\Delta$ and using the trivial upper bound $\mathcal { R } \leq K$ concludes. □

When $\epsilon \leq 1$ , the lower bound of Theorem 1 is in $\Omega ( \frac { S \sqrt { A K } } { \epsilon } )$ . Compared to the non-private rate of $\tilde { \Theta } ( \sqrt { S A K } )$ obtained by Azar et al. [2017], there is a multiplicative cost of $\frac { \sqrt { S } } { \epsilon }$ due to privacy. When $\epsilon \leq 1$ , the rate of private bandits with SA arms is in $\Theta ( \sqrt { S A K } / \epsilon )$ , the results of Theorem 1 shows that private contextual bandit with S context and A arms is strictly harder as a factor of order $\sqrt { S }$ is gained.

Intuitively the difficulty in private bandits with S context and A arms is that since context must remain private, when a privatized reward is received it is not clear from which bandit it came. The computation of the TV distance in the proof of Theorem 1 in Section A shows that a factor $\textstyle { \frac { 1 } { S } }$ appears become of the randomness of the context. This ultimately leads to the lower bound in $\frac { S \sqrt { A K } } { \epsilon }$ in the high privacy regime $\epsilon \leq 1$

Remark 1 (Sequentially interactive mechanisms). In the setting ofSection 3, the same mechanism M privatizes every trajectory. Theorem 1 holds more generally for sequentially interactive ϵ-LDP mechanisms (Definition 1 in Section A),for which the ϵ-LDP channel used at episode k may depend on the previous messages $M ^ { 1 } , \ldots , M ^ { \tilde { k - 1 } }$ . The proof only uses that, given the past messages, the same ϵ-LDP channel is applied under the two instances being compared, whether or not this channel depends on the history. The proofin Section A is writtenfor this more general class.

## 4.2 Upper bound

In this subsection, we show that the lower bound of Theorem 1 is essentially tight by showing that there exists an algorithm and a mechanism achieving a regret upper bound matching the lower bound up to constants and a multiplicative $\sqrt { \log ( A ) }$ factor.

The algorithm we use is based on Exp3 with forced exploration Auer et al. [2002] and the mechanism we use is based on the optimized unary encoding mechanism Wang et al. [2017]. We name the procedure Exp3-OUE and give an implementation in Algorithm 1.

We first describe the optimized unary encoding mechanism (OUE). Given $v \in \{ 0 , 1 \} ^ { d }$ with $\Vert \boldsymbol { v } \Vert _ { 1 } \leq$ 1, the OUE mechanism outputs a vector $\mathsf { O U E } ( v ) \in \{ 0 , 1 \} ^ { d }$ where each coordinate is generated independently as follows:

$$
\begin{array} { r l } & { \mathrm { O U E : ~ } v \in \{ w \in \{ 0 , 1 \} ^ { d } : \| w \| _ { 1 } \leq 1 \}  \mathrm { O U E } ( v ) \in \{ 0 , 1 \} ^ { d } \quad \mathrm { s u c h ~ t h a t } } \\ & { \forall i \in [ d ] , \mathrm { O U E } ( v ) _ { i } \mid v _ { i } \sim \{ \mathcal { B } \Big ( \frac { 1 } { 2 } \Big ) , \mathrm { i f ~ } v _ { i } = 1  } \\ & { \forall i \in [ d ] , \mathrm { O U E } ( v ) _ { i } \mid v _ { i } \sim \{ \mathcal { B } \Big ( \frac { 1 } { \exp ( \epsilon ) + 1 } \Big ) , \mathrm { i f ~ } v _ { i } = 0  \quad } \end{array} .\tag{1}
$$

As shown in Wang et al. [2017], OUE is ϵ-LDP. This mechanism is optimal for histogram estimation under LDP Canonne and Gentle [2024]. It is therefore a natural candidate for our problem.

We now describe Exp3-OUE informally. At episode k, the user receives a deterministic policy $\pi ^ { k }$ from the decision-maker and executes it, which yields a trajectory $\tau ^ { k } = ( S ^ { k } , A ^ { k } , R ^ { k } )$ . Then, the user binarizes the reward by sampling $B ^ { k } \sim B ( R ^ { k } )$ and constructs a vector $V ^ { k }$ of dimension $S$ where $V ^ { k } ( s ) = \mathbb { 1 } \{ S ^ { k } = s , { \dot { B } } ^ { k } = { \bar { 1 } } \}$ This vector is then privatized by OUE yielding a vector $M ^ { k } = \operatorname { O U E } ( V ^ { k } )$ sent back to the decision-maker. The decision maker then builds an importanceweighted estimator $\hat { V } ^ { k } ( s , a )$ of the reward for each state $s \in S$ and action $a \in { \mathcal { A } }$ and uses it to update the weights of Exp3 as $w _ { k + 1 } = w _ { k } \exp ( \eta \hat { V } ^ { k } )$ where η is a learning rate parameter and $w _ { k }$ is known from previous episodes (we set $w _ { 1 } = ( 1 , \ldots , 1 ) )$ . Following the Exp3 algorithm, these weights are used to compute the new policy as in Equation ${ \overset { \cdot } { ( 2 ) } } .$ . For every state $s \in S ,$ , we sample $\alpha _ { s }$ from $\Pi ^ { k + 1 } ( \cdot , s )$ and define the deterministic policy $\dot { \pi } ^ { k + 1 } \mathfrak { b y } \bar { \pi } ^ { k + 1 } ( s ) = \overline { { \alpha _ { s } } }$ . The policy $\pi ^ { k + 1 }$ is then sent to the user of episode $k + 1$ and the process continues.

Algorithm 1: Exp3 with optimized unary encoding for private contextual bandits (Exp3-OUE)   
Input: $\epsilon , K , \mathcal { A } , S , \eta , \gamma$   
▷ Decision maker (public operations)   
Initialize $w _ { 1 } ( s , a ) = 1$ for all $s \in { \bar { S } }$ and $a \in { \mathcal { A } }$   
Define the distributions $\Pi ^ { 1 } ( \cdot , s )$ as $\begin{array} { r } { \Pi ^ { 1 } ( a , s ) = ( 1 - \gamma ) \frac { w _ { 1 } ( s , a ) } { \sum _ { a ^ { \prime } \in \mathcal { A } } w _ { 1 } ( s , a ^ { \prime } ) } + \frac { \gamma } { A } } \end{array}$ for each $s , a \in \mathcal S \times \mathcal A$   
Set $\pi ^ { 1 } ( s ) = \alpha _ { s }$ where $\alpha _ { s } \sim \Pi ^ { 1 } ( \cdot , s )$ and send $\pi ^ { 1 }$ to user 1.   
for $k = \dot { 1 }$ to $K$ do   
▷ User $k$ (private operations)   
Observe $S ^ { k }$ , select $A ^ { k } = \pi ^ { k } ( S ^ { k } )$ , observe $R ^ { k }$ , and sample $B ^ { k } \sim B ( R ^ { k } )$   
Construct $V ^ { k } ( s ) = \mathbb { 1 } \left\{ S ^ { k } = { \dot { s } } \right\} B ^ { k }$ for all $s \in S$   
Construct $M ^ { k } \in \{ 0 , 1 \} ^ { \mathcal { S } } = \mathrm { O } \mathbf { \bar { U } } \mathrm { E } ( V ^ { k } )$ via the OUE mechanism Equation (1)   
Send $M ^ { k }$ to the decision maker   
▷ Decision maker (public operations)   
For each $s \in S$ and $a \in { \mathcal { A } } ,$ construct the estimator   
$\hat { V } ^ { k } ( s , a ) = M ^ { k } ( s ) \frac { \mathbb { 1 } \left\{ a = \pi ^ { k } ( s ) \right\} } { \Pi ^ { k } ( a , s ) }$   
Update the weights as for each $s , a \in \mathcal { S } \times \mathcal { A } ,$   
$w _ { k + 1 } ( s , a ) = w _ { k } ( s , a ) \exp ( \eta \hat { V } ^ { k } ( s , a ) )$   
Define $\Pi ^ { k + 1 } ( \cdot , s )$ as for each $s , a \in \mathcal S \ \times \mathcal A ,$   
$\Pi ^ { k + 1 } ( a , s ) = ( 1 - \gamma ) \frac { w _ { k + 1 } ( s , a ) } { \sum _ { a ^ { \prime } \in \mathcal { A } } w _ { k + 1 } ( s , a ^ { \prime } ) } + \frac { \gamma } { A } .$ (2)   
Set $\pi ^ { k + 1 } ( s ) = \alpha _ { s }$ where $\alpha _ { s } \sim \Pi ^ { k + 1 } ( \cdot , s )$ for each $s \in S$ and send $\pi ^ { k + 1 }$ to user $k + 1 .$

In essence, Exp3-OUE is Exp3 applied to privatized input, but two implementation choices make a significant difference compared to a direct Laplace-based LDP-OBI Garcelon et al. [2021] adapted to the bandit case. First, in Exp3-OUE, the decision-maker sends a deterministic realization $\pi ^ { k }$ of $\Pi ^ { k }$ rather than $\Pi ^ { k }$ directly, so the user privatizes a vector indexed by states rather than by state-action pairs, improving the dependence on A. Second, the user binarizes the reward and applies OUE, which improves the dependence on ϵ. The estimator $\hat { V } ^ { k } ( s , a )$ also gives the right bias-variance trade-off; debiasing it would increase the regret.

In Theorem 2, we precisely characterize the regret of Exp3-OUE and show that it matches the lower bound of Theorem 1 up to constants and a multiplicative $\sqrt { \log ( A ) }$ factor.

Theorem 2. Assume $A \geq 2 ,$ , for any instance I characterized by a rewardfunction r and initial distributions $\{ \rho ^ { k } \} _ { k = 1 } ^ { K }$ , the regret ofAlgorithm 1 run with $\gamma = A \eta$ and

$$
\eta = \operatorname* { m i n } ( \frac { 1 } { 2 A } , \sqrt { \frac { S \ln ( A ) } { 2 ( e - 1 ) K A \big ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } + \frac { S } { \exp ( \epsilon ) + 1 } \big ) } } )
$$

is upper bounded as

$$
\mathcal { R } ( \mathfrak { A } , \mathcal { M } , I ) \leq 1 6 \operatorname* { m i n } \left( K , \frac { \sqrt { \operatorname* { m a x } ( S , \exp ( \epsilon ) + 1 ) S A K ( \exp ( \epsilon ) + 1 ) \log ( A ) } } { \exp ( \epsilon ) - 1 } \right)
$$

Proof sketch (see Section B for the full proof). The proof follows the standard Exp3 potential argument [Auer et al., 2002, Theorem 3.1]. Let

$$
\Phi _ { k } = \sum _ { s \in \mathcal { S } } \log \Big ( \sum _ { a \in \mathcal { A } } w _ { k } ( s , a ) \Big ) .
$$

Since $\hat { V } ^ { k } ( s , a ) \ \in \ [ 0 , A / \gamma ]$ and $\gamma = A \eta .$ , we have $\eta \hat { V } ^ { k } ( s , a ) \in [ 0 , 1 ]$ . Expanding $\Phi _ { k + 1 } - \Phi _ { k }$ summing over $k ,$ using $\Phi _ { 1 } = S \log ( A )$ and comparing with the best fixed action $a ^ { * } ( s )$ gives, after taking expectations,

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \Big [ \sum _ { k = 1 } ^ { K } r ( S ^ { k } , a ^ { * } ( S ^ { k } ) ) - \sum _ { k = 1 } ^ { K } r ( S ^ { k } , A ^ { k } ) \Big ] \leq \frac { S \ln ( A ) } { \eta ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } ) } } & { } \\ { + \frac { 2 ( e - 1 ) K A \eta ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } + \frac { S } { \exp ( \epsilon ) + 1 } ) } { \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } } . } & { } \end{array}
$$

Plugging in the chosen learning rate and using that regret is at most K yields the theorem. □

As a consequence of Theorem 1 and Theorem 2, we have that the lower bound in Theorem 1 is tight up to log-factor. Therefore, the minimax regret R for private contextual bandits satisfies

$$
\mathfrak { R } = \tilde { \Theta } \Bigg ( \operatorname* { m i n } \Bigg ( K , \frac { \sqrt { \operatorname* { m a x } ( S , \exp ( \epsilon ) + 1 ) S A K ( \exp ( \epsilon ) + 1 ) } } { \exp ( \epsilon ) - 1 } \Bigg ) \Bigg ) .
$$

## 5 Private Tabular Reinforcement Learning

In this section, we attempt to extend the lower bound of Theorem 1 and the upper bound of Theorem 2 to the case of finite-horizon MDPs. We first focus on lower bound and see in Corollary 1 that the usual reduction from reinforcement learning to contextual bandits applies in the private setting as well. This yield a lower bound for private tabular reinforcement learning that improves over previous work Garcelon et al. [2021]. We then establish an upper bound in Theorem 3, adapting the ideas used to prove Theorem 2 to the case of finite-horizon MDPs. In the high privacy regime $\epsilon \leq 1$ , this upper bound improves over known upper bounds for private tabular reinforcement learning Garcelon et al. [2021], Qiao and Wang [2023], Chowdhury and Zhou [2022b].

## 5.1 Lower bound

We first show the following corollary of Theorem 1.

Corollary 1. Assume $S \geq 4$ and $A \geq 2$ and $H \geq 2 .$ . For any ϵ-LDP mechanism M and any algorithm A that the decision maker can choose, there exists an instance I offinite-horizon MDP such that the regret is lower bounded as

$$
\mathcal { R } ( \mathfrak { A } , \mathcal { M } , I ) \geq \frac { H } { 1 2 8 } \operatorname* { m i n } \left( K , \frac { \sqrt { \operatorname* { m a x } ( S , \exp ( \epsilon ) + 1 ) S A K ( \exp ( \epsilon ) + 1 ) } } { \exp ( \epsilon ) - 1 } \right)
$$

Proofsketch (see Section Cfor thefull proof). Consider a finite-horizon MDP with $\begin{array} { r } { \boldsymbol { S } = [ \boldsymbol { S } ] , \boldsymbol { A } = } \end{array}$ $[ A ] . \rho ^ { k }$ has support over $[ S - 2 ]$ ]. The other 2 states are labelled "good" and "bad". Both good and bad states are absorbing and yield a reward of 1 and 0 respectively. For any $s \in [ S - 2 ]$ , any actions lead to the good state with some probability and to the bad state with the remaining probability with the reward being 0 in both cases. This MDP is identical to a contextual bandit problem with $S - 2$ contexts and A actions except that rewards are scaled by $H - 1$ . Then, choose $P$ and $\rho$ such that the hard instance of Theorem 1 is obtained and the lower bound is proved. □

The lower bound in Garcelon et al. [2021] is given (up to constants) by $\begin{array} { r } { H \frac { \sqrt { S A K } } { \operatorname* { m i n } ( \exp ( \epsilon ) - 1 , 1 ) } } \end{array}$ a rate that is lower than that of Corollary 1 for any value of $H , S , A , K , \epsilon$ . More precisely, the two bounds coincide when ϵ is on the order of log S or larger, and when $\epsilon \leq \log S$ the improvement is a multiplicative factor of order $\sqrt { S / \exp ( \epsilon ) }$ .

The reason why the lower bound in Theorem 1 dominates is because in Garcelon et al. [2021], authors construct a tree in which one starts at the root and moves deterministically depending on the actions played, the number of leaves being of order $S ,$ and in which one special leaf-action pair is slightly more rewarding than the others. Their construction therefore reduces the problem to a private bandit instance with $\bar { S } A$ arms. In contrast, our lower bound reduces the problem to a private contextual bandit with S context and A arms which is a more difficult problem.

## 5.2 Upper bound

In this sub-section, we show an upper bound for the minimax regret improving over known upper bounds in the high privacy regime.It is obtained using a variant of UCBVI as the algorithm and the optimized unary encoding as the ϵ-LDP mechanism.

We name the procedure UCBVI-OUE, which we now present informally. A precise implementation is available in Algorithm 2. At episode $k ,$ the user receives a deterministic policy $\pi ^ { k }$ and an index $J ^ { k }$ from the decision-maker. The user executes the policy $\pi ^ { k }$ , which yields a trajectory $\tau ^ { k } =$ $( S _ { 1 } ^ { k } , A _ { 1 } ^ { k } , R _ { 1 } ^ { k } , \ldots , S _ { H } ^ { k } , A _ { H } ^ { k } , R _ { H } ^ { k } )$ . Then, the user binarizes the reward at step $J ^ { k }$ by $B _ { J ^ { k } } ^ { \check { k } } \sim \check { B } ( R _ { J ^ { k } } ^ { k } )$ The user then constructs a vector $I ^ { k } = \big ( \mathbb { 1 } \Big \{ S _ { J ^ { k } } ^ { k } = s , B _ { J ^ { k } } ^ { k } = b , S _ { J ^ { k } + 1 } ^ { k } = s ^ { \prime } \Big \} \big ) _ { s \in \mathcal { S } , b \in \{ 0 , 1 \} , s ^ { \prime } \in \mathcal { S } }$ of dimension $2 S ^ { 2 }$ and privatizes it using the OUE mechanism to get $M ^ { k } = \mathrm { { { O U E } } } ( I ^ { k } ) . ~ M ^ { k }$ is then sent back to the decision-maker who uses it to build for each $s \in \mathcal { S }$ and $a \in A .$ , confidence sets $\mathcal { C } ^ { k } ( s , a )$ which contain $P ( s , a , \cdot )$ and $r ( s , a )$ with high probability. Then, the decision maker computes optimistic Q-value estimates $\tilde { Q } ^ { k , ( + ) }$ using the confidence sets by backward induction as in Equation (6). Then the new policy is computed as $\pi _ { h } ^ { k + 1 } ( s ) = \operatorname { a r g m a x } _ { a \in \mathcal { A } } \tilde { Q } _ { h } ^ { k , ( + ) } ( s , a )$ for each $s \in S$ and $h \in [ H ]$ and $J ^ { k + 1 }$ is sampled uniformly at random from $[ H ] . \ \pi ^ { k + 1 }$ and $J ^ { k + 1 }$ are then sent to the user of episode $k + 1$ and the process continues.

In addition to the two ideas already used in the case of contextual bandits, which are sending a deterministic policy instead of a randomized one and binarizing the reward to use the OUE mechanism, UCBVI-OUE also uses the idea of sending the indicator of only a triplet $S _ { J ^ { k } } ^ { k } , B _ { J ^ { k } } ^ { k } , S _ { J ^ { k } + 1 } ^ { k }$ from the trajectory chosen uniformly at random as opposed to sending the indicator of the whole trajectory. This reduces the amount of noise that needs to be added to ensure $\mathrm { L D P }$ and ultimately leads to an estimator that has lower variance.

We now present Theorem 3, which gives an upper bound on the regret of Algorithm 2.

Theorem 3. Assume $S \geq 4 , A \geq 2$ and $H \geq 2 .$ . For any instance I offinite-horizon MDP, the regret of Algorithm 2 is upper bounded up to constants and log-factors (see the exact expression in Section D) as

$$
\mathcal { R } ( \mathfrak { A } , M , I ) \le \tilde { O } \bigg ( H ^ { 2 } S ^ { 2 } \sqrt { \Upsilon _ { \epsilon } ^ { 2 } A K } + H ^ { 2 } S ^ { 2 } A \Upsilon _ { \epsilon } \bigg ) , \quad \Upsilon _ { \epsilon } = \frac { \exp ( \epsilon ) + 1 } { \exp ( \epsilon ) - 1 }
$$

Proof sketch (see Section D for the full proof). Let

$$
N ^ { k } ( s , a , b , s ^ { \prime } ) = \sum _ { i \in [ k ] } \sum _ { h = 1 } ^ { H } \mathbb { 1 } \left\{ S _ { h } ^ { i } = s , B _ { h } ^ { i } = b , S _ { h + 1 } ^ { i } = s ^ { \prime } , \pi _ { h } ^ { i } ( s ) = a \right\}
$$

Algorithm 2: UCBVI with optimized unary encoding for private tabular RL (UCBVI-OUE)   
Input: $\begin{array} { r } { \epsilon , K , \mathcal { A } , \mathcal { S } , H , \Upsilon _ { \epsilon } = \frac { \exp ( \epsilon ) + 1 } { \exp ( \epsilon ) - 1 } , q _ { \epsilon } = \frac { 1 } { \exp ( \epsilon ) + 1 } . } \end{array}$   
▷ Decision maker (public operations)   
Initialize $\tilde { Q } _ { h } ^ { 0 } ( s , a ) = H$ for all $s , a , h \in \mathcal S \times \mathcal A \times [ H ]$   
Send $\pi _ { h } ^ { 1 } ( s ) = \mathrm { a r g m a x } _ { a \in \mathcal { A } } \tilde { Q } _ { h } ^ { 0 } ( s , a )$ to user 1 for each $s \in S$ and $h \in [ H ]$   
Sample $\scriptstyle { \mathcal { J } } ^ { 1 }$ uniformly at random from [H] and send $J ^ { 1 }$ to user 1.   
for $\dot { k } = 1$ to $K$ do   
▷ User $k$ (private operations)   
Play according to policy $\pi ^ { k }$ and observe $( S _ { h } ^ { k } , A _ { h } ^ { k } , R _ { h } ^ { k } , S _ { h + 1 } ^ { k } ) _ { h \in [ H ] }$   
Sample $B _ { h } ^ { k } \sim B ( R _ { h } ^ { k } )$ for each $h \in [ H ]$   
Construct $I ^ { k } ( s , b , s ^ { \prime } ) = \mathbb { 1 } \Big \{ S _ { J ^ { k } } ^ { k } = s , B _ { J ^ { k } } ^ { k } = b , S _ { J ^ { k } + 1 } ^ { k } = s ^ { \prime } \Big \}$ for all $s , s ^ { \prime } \in S , b \in \{ 0 , 1 \}$   
Construct $M ^ { k } \in \{ 0 , 1 \} ^ { S } \overset { \setminus } { = } \mathbf { O } \mathbf { U } \mathbf { E } ( I ^ { k } )$ via the OUE mechanism Equation (1)   
Send $M ^ { k }$ to the decision maker   
▷ Decision maker (public operations)   
For each $s , b , s ^ { \prime } \in \mathcal { S } \times \{ 0 , 1 \} \times \mathcal { S } ,$ , define $\hat { I } ^ { k } ( s , b , s ^ { \prime } ) = 2 \Upsilon _ { \epsilon } ( M ^ { k } ( s , b , s ^ { \prime } ) - q _ { \epsilon } )$   
For each $s , a , b , s ^ { \prime } \in \tilde { S } \times \tilde { A } \times \{ 0 , 1 \} \times \tilde { S } ,$ compute $\beta ( s , a )$ as in Equation (7) and define   
$\hat { N } ^ { k } ( s , a , b , s ^ { \prime } ) = H \sum _ { i \in [ k ] } \hat { I } ^ { i } ( s , b , s ^ { \prime } ) \mathbb { 1 } \left\{ \pi _ { J ^ { i } } ^ { i } ( s ) = a \right\}$ (3)   
$\tilde { N } ^ { k , ( - ) } ( s , a ) = \mathrm { m a x } ( 1 , \sum \mathrm { ~ \ r ~ { ~ ( ~ \hat { N } ^ { k } ( s , a , } } b , s ^ { \prime } ) - \beta _ { k } ( s , a ) ) )$ (4)   
$b \in \{ 0 , 1 \} , s ^ { \prime } \in S$   
Then, build confidence sets   
$\mathcal { C } ^ { k } ( s , a ) = \left\{ \nu \in \Delta ( S ) , \rho \in [ 0 , 1 ] : | \tilde { N } ^ { k , ( - ) } ( s , a ) \rho - \sum _ { s ^ { \prime } \in S } \hat { N } ^ { k } ( s , a , 1 , s ^ { \prime } ) | \leq 7 S \beta _ { k } ( s , a ) \right\} .$   
$\| \tilde { N } ^ { k , ( - ) } ( s , a ) \nu ( \cdot ) - \sum _ { \iota , \textrm { c f o } 1 \rangle } \hat { N } ^ { k } ( s , a , b , \cdot ) \| _ { 1 } \leq 8 S \beta _ { k } ( s , a ) \bigg \rbrace$ (5)   
b∈{0,1}   
Set $\tilde { Q } _ { H + 1 } ^ { k , ( + ) } ( s , a ) = 0$ for each $s , a \in \mathcal S \times \mathcal A$ and then compute   
$\tilde { Q } _ { h } ^ { k , ( + ) } ( s , a ) = \operatorname* { m a x } _ { \nu , \rho \in \mathcal { C } ^ { k } ( s , a ) } \left( \rho + \sum _ { s ^ { \prime } \in S } \nu ( s ^ { \prime } ) \operatorname* { m a x } _ { a ^ { \prime } \in A } \tilde { Q } _ { h + 1 } ^ { k , ( + ) } ( s ^ { \prime } , a ^ { \prime } ) \right)$ (6)   
Send $\pi _ { h } ^ { k + 1 } ( s ) = \operatorname { a r g m a x } _ { a \in \mathcal { A } } \tilde { Q } _ { h } ^ { k , ( + ) } ( s , a )$ to user $k + 1$ for each $s \in S$ and $h \in [ H ]$   
Sample $\scriptstyle { J ^ { k + 1 } }$ uniformly at random from $[ H ]$ and send $J ^ { k + 1 }$ to user $k + 1 .$

and let $\hat { N } ^ { k }$ be the private estimator in Equation (4). Freedman’s inequality Hsu et al. [2015] implies (Lemma 1) that, with probability at least $1 - \delta$ , for all $s , a , b , s ^ { \prime } , k .$

$$
\begin{array} { r l } & { | \hat { N } ^ { k } ( s , a , b , s ^ { \prime } ) - N ^ { k } ( s , a , b , s ^ { \prime } ) | \leq \beta _ { k } ( s , a ) , } \\ & { \beta _ { k } ( s , a ) = \sqrt { 2 0 H \Upsilon _ { \epsilon } ^ { 2 } \displaystyle \sum _ { h \in [ H ] , i \in [ k ] } ^ { 2 } { 1 \left\{ a = \pi _ { h } ^ { i } ( s ) \right\} \log \frac { 4 8 K ^ { 2 } S ^ { 2 } A } { \delta } } } + 4 H \Upsilon _ { \epsilon } \log \frac { 4 8 K ^ { 2 } S ^ { 2 } A } { \delta } . } \end{array}\tag{7}
$$

Furthermore, Lemma 3 shows $\mathcal { C } ^ { k } ( s , a )$ Equation (5) contains with high probability the true reward and transition probabilities for all $s , a , k$ . By Lemma 2, under these events, the optimistic Q-values then satisfy the Bellman-type bounds

$$
\begin{array} { l } { 0 \le \tilde { Q } _ { h } ^ { k , ( + ) } ( s , a ) - \left( r ( s , a ) + \displaystyle \sum _ { s ^ { \prime } \in S } P ( s ^ { \prime } \mid s , a ) \displaystyle \operatorname* { m a x } _ { a ^ { \prime } \in A } \tilde { Q } _ { h + 1 } ^ { k , ( + ) } ( s ^ { \prime } , a ^ { \prime } ) \right) \le b _ { k } ( s , a , h ) } \\ { \displaystyle \tilde { Q } _ { h } ^ { k , + } ( s , a ) \le H - h + 1 } \end{array}
$$

where $b _ { k } ( s , a , h ) = \operatorname* { m i n } ( H - h + 1 , 1 8 S \beta _ { K } ( s , a ) H / \operatorname* { m a x } ( 1 , N ^ { k } ( s , a ) - 4 S \beta _ { K } ( s , a ) ) )$ . As shown by Lemma 4, optimism reduces regret to bounding $\begin{array} { r l } { ~ } & { { } \sum _ { k , h } b _ { k - 1 } \big ( S _ { h } ^ { k } , A _ { h } ^ { k } \big ) } \end{array}$ ; the counting argument in Section D gives the stated rate. □

The upper bound shown in Theorem 3 is clearly sub-optimal in the low privacy regime. $\mathbf { A s } \epsilon \to \infty$ the upper bound becomes ${ \tilde { O } } ( H ^ { 2 } S ^ { 2 } { \sqrt { A K } } )$ which is much worse than the optimal non-private regret rate in $\Theta ( H \sqrt { S A K } )$ . Furthermore, the private term in the upper bound of Garcelon et al. [2021] is given by $H ^ { 3 } S ^ { 2 } A \sqrt { K } / \epsilon$ ϵ which is better than the bound of Theorem 3 as soon as $\varepsilon \geq H \sqrt { A }$ . However, when $\epsilon \le H \sqrt { A } - 2$ , the bound of Theorem 3 dominates and when $\epsilon \leq 1$ , the gain is of order $H { \sqrt { A } }$ The gain of the $\sqrt { A }$ factor is linked to the fact that we send a deterministic version of the policy and that we privatize a randomly chosen triplet from the trajectory. Indeed, privatizing a quantity such as $\begin{array} { r } { \sum _ { h \in [ H ] } \dot { \mathbf { 1 } } \left\{ S _ { h } ^ { k } = s , A _ { h } ^ { k } = \dot { a } , S _ { h + 1 } ^ { k } = \dot { s ^ { \prime } } \right\} } \end{array}$ by adding Laplace noise as done in Garcelon et al. [2021] gives an estimator of variance $H ^ { 2 } / \epsilon ^ { 2 }$ . Instead, privatizing 1 $\left\{ S _ { J ^ { k } } ^ { k } = s , A _ { J ^ { k } } ^ { k } = a , S _ { J ^ { k } + 1 } ^ { k } = s ^ { \prime } \right\}$ can be done with a noise of scale $1 / \epsilon ^ { 2 }$ and the variance of the resulting estimator is on the order of $\begin{array} { r } { H / \epsilon ^ { 2 } \sum _ { h = 1 } ^ { H } \mathbb { 1 } \left\{ \pi _ { h } ^ { k } ( s ) = a \right\} } \end{array}$ which can be much smaller for actions rarely selected. The gain of the H factor is due to a tighter analysis. In Garcelon et al. [2021], the summation of the bonuses introduces an additional factor on top of the one due to the Bellman updates and the one due to the privacy-noise variance (see their Equation (50)). In contrast, our careful handling of the counts $\begin{array} { r } { \sum _ { k , h } \dot { b } _ { k - 1 } \big ( \check { S } _ { h } ^ { k } , A _ { h } ^ { k } \big ) } \end{array}$ in the proof of Theorem 3 allows us not to pay this additional factor.

## 6 Conclusion and future work

We improved regret bounds for tabular RL under local differential privacy. For contextual bandits, the minimax regret is characterized up to constants and a $\sqrt { \log ( A ) }$ factor Theorems 1 and 2. For finite-horizon RL, our reduction Corollary 1 gives a lower bound that is H times the contextual bandit lower bound. The improvement over previous results increases as ϵ decreases, yielding a $\sqrt { S }$ improvement when $\epsilon \leq 1$ . Our UCBVI-OUE upper bound Theorem 3 improves over previous results when $\epsilon \leq \operatorname* { m a x } ( H \sqrt { A } - 2 , 1 )$ ), which includes the high privacy regime $\epsilon \leq 1$ , where a $H { \sqrt { A } }$ improvement is obtained. We note that when $\epsilon  \infty$ the upper bound of Theorem 3 becomes ${ \tilde { O } } ( H ^ { 2 } S ^ { 2 } { \sqrt { A K } } )$ which is much worse than the optimal non-private regret rate in $\Theta ( H \sqrt { S A K } )$ ). In fact, when $\epsilon \geq H \sqrt { A }$ , the approach of Garcelon et al. [2021] is preferable.

The main open question is whether the lower bound of Corollary 1 is tight. In non-private problems, contextual bandits and tabular RL differ by a factor H, and it is natural to conjecture that this remains true under LDP. How to close the remaining HS gap in the high-privacy regime is then the most pressing question. We conjecture that closing the S-factor gap requires model-free approaches, as model-based methods struggle with the high dimensionality of the transition kernel. However, existing analyses of model-free algorithms produce suboptimal rates even without privacy constraints. These challenging questions remain open for future investigation.

## References

M. M. Afsar, T. Crump, and B. Far. Reinforcement Learning based Recommender Systems: A Survey. ACM Computing Surveys, 55(7):1–38, July 2023. ISSN 0360-0300, 1557-7341. doi: 10.1145/3543846.

S. Asoodeh and H. Zhang. Contraction of Locally Differentially Private Mechanisms. IEEE Journal on Selected Areas in Information Theory, 5:385–395, 2024. ISSN 2641-8770. doi: 10.1109/JSAIT.2024.3397305.

P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire. The Nonstochastic Multiarmed Bandit Problem. SIAM Journal on Computing, 32(1):48–77, Jan. 2002. ISSN 0097-5397, 1095-7111. doi: 10.1137/S0097539701398375.

M. G. Azar, I. Osband, and R. Munos. Minimax regret bounds for reinforcement learning. In International Conference on Machine Learning, pages 263–272. PMLR, 2017.

A. Azize and D. Basu. When privacy meets partial information: A refined analysis of differentially private bandits. Advances in Neural Information Processing Systems, 35:32199–32210, 2022.

A. Azize, M. Jourdan, A. A. Marjani, and D. Basu. Differentially Private Best-Arm Identification, Apr. 2026.

H. Cai, K. Ren, W. Zhang, K. Malialis, J. Wang, Y. Yu, and D. Guo. Real-Time Bidding by Reinforcement Learning in Display Advertising. In Proceedings ofthe Tenth ACM International Conference on Web Search and Data Mining, pages 661–670, Cambridge United Kingdom, Feb. 2017. ACM. ISBN 978-1-4503-4675-7. doi: 10.1145/3018661.3018702.

C. L. Canonne and A. Gentle. Locally Private Histograms in All Privacy Regimes, Sept. 2024.

X. Chen, S. Miao, and Y. Wang. Differential privacy in personalized pricing with nonparametric demand models. Operations Research, 71(2):581–602, 2023.

S. R. Chowdhury and X. Zhou. Shuffle private linear contextual bandits. arXiv preprint arXiv:2202.05567, 2022a.

S. R. Chowdhury and X. Zhou. Differentially private regret minimization in episodic markov decision processes. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 6375–6383, 2022b.

C. Dann, L. Li, W. Wei, and E. Brunskill. Policy certificates: Towards accountable reinforcement learning. In International Conference on Machine Learning, pages 1507–1516. PMLR, 2019.

H. Devlin and T. Burgis. Confidential Health Records from {UK Biobank} Project Exposed Online. 2026.

J. C. Duchi, M. I. Jordan, and M. J. Wainwright. Local privacy and statistical minimax rates. In 2013 IEEE 54th Annual Symposium on Foundations ofComputer Science, pages 429–438. IEEE, 2013.

E. Garcelon, V. Perchet, C. Pike-Burke, and M. Pirotta. Local differential privacy for regret minimization in reinforcement learning. Advances in Neural Information Processing Systems, 34: 10561–10573, 2021.

E. Garcelon, K. Chaudhuri, V. Perchet, and M. Pirotta. Privacy amplification via shuffling for linear contextual bandits. In International Conference on Algorithmic Learning Theory, pages 381–407. PMLR, 2022.

Y. Han, Z. Liang, Y. Wang, and J. Zhang. Generalized linear bandits with local differential privacy. Advances in neural information processing systems, 34:26511–26522, 2021.

D. J. Hsu, A. Kontorovich, and C. Szepesvari. Mixing Time Estimation in Reversible Markov Chains from a Single Sample Path. In Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015.

C. Jin, Z. Allen-Zhu, S. Bubeck, and M. I. Jordan. Is Q-learning Provably Efficient?, July 2018.

S. Jones. Having an affair may shorten your life: The ashley Madison suicides. Ga. St. UL Rev., 33: 455, 2016.

J. Kober, J. A. Bagnell, and J. Peters. Reinforcement learning in robotics: A survey. The International Journal of Robotics Research, 32(11):1238–1274, Sept. 2013. ISSN 0278-3649, 1741-3176. doi: 10.1177/0278364913495721.

T. Lattimore and C. Szepesvári. Bandit Algorithms. Cambridge University Press, 2020.

Y. Ma, F. Jiang, Z. Zhao, H. Yang, and Y. Yu. Locally private nonparametric contextual multi-armed bandits. arXiv preprint arXiv:2503.08098, 2025.

A. Narayanan and V. Shmatikov. Robust de-anonymization of large sparse datasets. In 2008 IEEE Symposium on Security and Privacy (Sp 2008), pages 111–125. IEEE, 2008.

M. Nazari, A. Oroojlooy, L. Snyder, and M. Takác. Reinforcement learning for solving the vehicle routing problem. Advances in neural information processing systems, 31, 2018.

N. Pavlovic, S. Salgia, and Q. Zhao. Differential privacy in kernelized contextual bandits via random projections. arXiv preprint arXiv:2507.13639, 2025.

M. L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming. John Wiley & Sons, 2014.

D. Qiao and Y.-X. Wang. Near-optimal differentially private reinforcement learning. In International Conference on Artificial Intelligence and Statistics, pages 9914–9940. PMLR, 2023.

W. Ren, X. Zhou, J. Liu, and N. B. Shroff. Multi-Armed Bandits with Local Differential Privacy, July 2020.

L. Rocher and J. Morley. Protecting health data at UK Biobank, 2026.

L. Rocher, J. M. Hendrickx, and Y.-A. De Montjoye. Estimating the success of re-identifications in incomplete datasets using generative models. Nature communications, 10(1):3069, 2019.

R. Shariff and O. Sheffet. Differentially private contextual linear bandits. Advances in Neural Information Processing Systems, 31, 2018.

L. Sweeney. Simple demographics often identify people uniquely. Technical Report Data Privacy Working Paper No. 3, Carnegie Mellon University, Pittsburgh, PA, 2000. URL https: //dataprivacylab.org.

G. Vietri, B. Balle, A. Krishnamurthy, and S. Wu. Private reinforcement learning with pac and regret guarantees. In International Conference on Machine Learning, pages 9754–9764. PMLR, 2020.

T. Wang, J. Blocki, N. Li, and S. Jha. Locally differentially private protocols for frequency estimation. In 26th USENIX Security Symposium (USENIX Security 17), pages 729–745, 2017.

A. C.-C. Yao. Probabilistic computations: Toward a unified measure of complexity. In 18th Annual Symposium on Foundations of Computer Science (Sfcs 1977), pages 222–227. IEEE, 1977.

C. Yu, J. Liu, S. Nemati, and G. Yin. Reinforcement Learning in Healthcare: A Survey. ACM Computing Surveys, 55(1):1–36, Jan. 2023. ISSN 0360-0300, 1557-7341. doi: 10.1145/3477600.

A. Zanette and E. Brunskill. Tighter problem-dependent regret bounds in reinforcement learning without domain knowledge using value function bounds. In International Conference on Machine Learning, pages 7304–7312. PMLR, 2019.

Z. Zhang, Y. Zhou, and X. Ji. Almost Optimal Model-Free Reinforcement Learning via Reference-Advantage Decomposition. https://arxiv.org/abs/2004.10019v2, Apr. 2020.

Z. Zhang, X. Ji, and S. Du. Is reinforcement learning more difficult than bandits? a near-optimal algorithm escaping the curse of horizon. In Conference on Learning Theory, pages 4528–4531. PMLR, 2021.

Z. Zhao, F. Jiang, and Y. Yu. Contextual dynamic pricing: Algorithms, optimality, and local differential privacy constraints. Journal of the American Statistical Association, pages 1–16, 2026.

K. Zheng, T. Cai, W. Huang, Z. Li, and L. Wang. Locally differentially private (contextual) bandits learning. Advances in Neural Information Processing Systems, 33:12300–12310, 2020.

## A Proof of Theorem 1

We prove Theorem 1 for the following class of mechanisms, which contains the mechanisms of Section 3.

Definition 1 (Sequentially interactive ϵ-LDP mechanism). Following Duchi et al. [2013], a sequentially interactive mechanism is a sequence ofchannels ${ \mathcal { M } } = ( Q ^ { k } ) _ { k \in [ K ] }$ where,for each trajectory $\tau \in \mathbb { T }$ and each sequence ofpast messages $y \in \mathcal { V } ^ { k - 1 } , Q ^ { k } ( \cdot \mid \tau , y )$ is a probability distribution over Y. At episode k, user k sends $M ^ { k } \sim Q ^ { \bar { k } } ( \cdot \mid \tau ^ { k } , ( M ^ { 1 } , \dot { \dots } ; M ^ { k - 1 } ) )$ , drawn independently of everything else given $\tau ^ { k }$ and $( M ^ { 1 } , \ldots , M ^ { k - 1 } )$ . The mechanism is ϵ-LDP if,for every $k \in [ K ]$ and every $y \in \mathcal { D } ^ { k - 1 }$ , the channel $Q ^ { k } ( \cdot \mid \cdot , y )$ is ϵ-LDP: for any two trajectories $\tau , \tau ^ { \prime }$ and any measurable subset $Y \subset \mathcal { P }$

$$
Q ^ { k } ( Y \mid \tau , y ) \leq e ^ { \epsilon } Q ^ { k } ( Y \mid \tau ^ { \prime } , y ) .
$$

The (non-interactive) mechanisms of Section 3 are the special case where $Q ^ { k } ( \cdot \mid \tau , y )$ is the distribution of $\mathcal { M } ( \tau )$ for every k and y. Since the policy $\pi ^ { k }$ of a deterministic algorithm is a function of $( M ^ { 1 } , \ldots , \dot { M } ^ { k - 1 } )$ , the channel used at episode k may in particular depend on $\pi ^ { k }$ . The regret is defined as in Section 3, with $M ^ { k }$ drawn as above.

Proof. Fix $\Delta > 0$ , we will choose its value later. Given $\phi \in \mathcal { A } ^ { s }$ and $\sigma \in S ^ { K }$ , we define an instance $I _ { \phi , \sigma } = ( r _ { \phi , \sigma } , \rho _ { \phi , \sigma } )$ by

$$
r _ { \phi , \sigma } ( s , a ) = \frac { 1 } { 2 } + \Delta \Im \{ a = \phi ( s ) \} , \quad \rho _ { \phi , \sigma } ^ { k } ( s ) = \mathbb { 1 } \{ s = \sigma _ { k } \}
$$

Then consider the random instance $I = ( r _ { I } , \rho _ { I } ) = I _ { \alpha , c }$ where $\alpha ( s )$ is chosen uniformly at random from A for each $s \in S$ and $c _ { k }$ is chosen uniformly at random from S for each $k \in [ \check { K } ]$ . In other words, I is uniformly distributed over the set of instances $\{ I _ { \phi , \sigma } : \phi \in \mathcal { A } ^ { S } , \sigma \in \mathcal { S } ^ { K } \}$ . Lastly, consider instance $I _ { s } ^ { 0 } = ( r _ { s } ^ { 0 } , \dot { \rho } _ { s } ^ { 0 } )$ ) defined as I except that $\begin{array} { r } { r _ { s } ^ { 0 } ( s , a ) = \frac { 1 } { 2 } } \end{array}$ for each $a \in { \mathcal { A } }$ and $r _ { s } ^ { 0 } ( s ^ { \prime } , a ) = r _ { I } ( s ^ { \prime } , a )$ for each $s ^ { \prime } \neq$ s and $a \in { \mathcal { A } }$

Fix a deterministic algorithm A and a sequentially interactive ϵ-LDP mechanism ${ \mathcal { M } } = ( Q ^ { k } ) _ { k \in [ K ] }$ (Definition 1). We will show a lower bound on the expected regret of $( { \mathfrak { A } } , { \mathcal { M } } )$ on the random instance I and then conclude the proof by application of Yao’s minimax principle Yao [1977].

The expected regret can be expressed as

$$
\begin{array} { r l } & { \| \widehat { \mathbf { g } } ( \mathbf { x } , \mathbf { g } , \mathbf { x } , \theta ) \| } \\ & { = \widehat { \mathbf { E } } \underset { \mathrm { s e } } { \overset {  } { \sum } } ( \underset { i = 1 } { \overset { N } { \sum } } ( V ^ { * } \otimes V ^ { * } - V ^ { * } \otimes V ^ { * } ) ) \| } \\ & { - \sum _ { s = 1 } ^ { N } \underset { i = 1 } { \overset { N } { \sum } } ( E ^ { * } - s ) ( \ 6 V ^ { * } \otimes V ^ { * } \otimes V ^ { * } ) \| } \\ & { - \sum _ { s = 1 } ^ { N } \underset { i = 1 } { \overset { N } { \sum } } ( \underset { s = 1 } { \overset { N } { \sum } } ( V ^ { * } \otimes V ^ { * } - V ^ { * } \otimes V ^ { * } ) ) \| } \\ & { - \underset { s = 1 } { \overset { N } { \sum } } ( \underset { s = 1 } { \overset { N } { \sum } } ( \underset { s = 1 } { \overset { N } { \sum } } ( \underset { s = 1 } { \overset { N } { \sum } } ( \lambda ^ { s } - a ( \delta ) ) ) ) } \\ & { - \delta \alpha ^ { 3 } ( V ^ { * } - V ^ { * } \otimes V ^ { * } ) \underset { i = 1 } { \overset { N } { \sum } } ( \underset { s = 1 } { \overset { N } { \sum } } ( \underset { s = 1 } { \overset { N } { \sum } } ( \lambda ^ { s } - a ( \delta ) ) ) ) } \\ & { = \Delta \Big ( \underset { s = 1 } { \overset { N } { \sum } } \underset { s = 1 } { \overset { N } { \sum } } \sum _ { s = 1 } ^ { N } \underset { i = 1 } { \overset { N } { \sum } } ( \underset { s = 1 } { \overset { N } { \sum } } ( \underset { s = 1 } { \overset { N } { \sum } } ( \lambda ^ { s } - a ( \delta ) ) ^ { 2 } ) ) } \\ &  \end{array}\tag{8}
$$

Call $\nu _ { s , j }$ and $\nu _ { s , 0 }$ the distributions of the sequence of privatized messages $( M ^ { i } ) _ { i \in [ K ] }$ on instance I conditioned on $\alpha ( s ) = j$ and on instance $I _ { s } ^ { 0 }$ respectively, and denote $\mathbb { E } _ { s , j }$ and $\mathbb { E } _ { s , 0 }$ the corresponding expectations.

Since $N _ { s , j }$ is bounded by K and is a function of the sequence of privatized messages $( M ^ { i } ) _ { i \in [ K ] }$ from Auer et al. [2002] (see also [Lattimore and Szepesvári, 2020, Exercise 15.2]), we have

$$
\mathbb { E } _ { s , j } \bigg [ N _ { s , j } \bigg ] \leq \mathbb { E } _ { s , 0 } \bigg [ N _ { s , j } \bigg ] + K \sqrt { \frac { 1 } { 2 } D _ { \mathrm { K L } } ( \nu _ { s , 0 } \| \nu _ { s , j } ) } .\tag{9}
$$

Let us now detail the KL divergence $D _ { \mathrm { K L } } ( \nu _ { s , 0 } \Vert \nu _ { s , j } )$ . To simplify notation, we write KL between random variables to mean the KL between the distributions they are sampled from. Furthermore, for a random variable X that depends on the instance we consider, we write $D _ { \mathrm { K L } } ( X \mid \alpha ( s ) = 0 \| X$ $\alpha ( s ) = j )$ to mean the KL divergence between the distribution of $X$ when the instance is sampled from $I _ { s } ^ { 0 }$ and the distribution of $\breve { X }$ when the instance is sampled from $I \mid \alpha ( s ) = j$

For any $\ell \in [ K ]$ , let $H ^ { \ell } = ( M ^ { i } ) _ { i = 1 } ^ { \ell }$ , we have

$$
\begin{array} { l } { { \displaystyle D _ { \mathrm { K L } } ( ( M ^ { i } ) _ { i \in [ K ] } \mid \alpha ( s ) = 0 \| ( M ^ { i } ) _ { i \in [ K ] } \mid \alpha ( s ) = j ) } } \\ { { \displaystyle \quad = \sum _ { \ell = 0 } ^ { K - 1 } \mathbb { E } _ { s , 0 } \biggl [ D _ { \mathrm { K L } } ( M ^ { \ell + 1 } \mid \alpha ( s ) = 0 , H ^ { \ell } \| M ^ { \ell + 1 } \mid \alpha ( s ) = j , H ^ { \ell } ) \biggr ] . } } \end{array}\tag{10}
$$

where the last equality is a direct consequence of the chain rule for KL divergence. We now detail the KL divergence term $D _ { \mathrm { K L } } ( M ^ { \ell + 1 } \ | \ \alpha ( s ) = 0 , H ^ { \ell } \Vert M ^ { \ell + 1 } \ | \ \alpha ( s ) = j , H ^ { \ell } )$ . Let $P _ { 0 } ^ { \ell + 1 }$ and $P _ { i } ^ { \ell + 1 }$ denote the distributions of $( S ^ { \ell + 1 } , A ^ { \ell + 1 } , R ^ { \ell + 1 } ) \mid \alpha ( s ) = 0 , H ^ { \ell }$ and of $( S ^ { \ell + 1 } , A ^ { \ell + 1 } , R ^ { \ell + 1 } )$ | $\dot { \alpha ( s ) } = j , H ^ { \ell }$ . Conditionally on $H ^ { \ell }$ , the message $M ^ { \ell + 1 }$ is obtained by passing $( S ^ { \ell + 1 } , A ^ { \ell + 1 } , R ^ { \ell + 1 } )$ through the channel $Q ^ { \ell + 1 } ( \cdot \mid \cdot , H ^ { \ell } )$ . Hence, the distributions of $\bar { M } ^ { \bar { \ell } + 1 } \mid \bar { \alpha } ( \dot { s } ) = 0 , H ^ { \ell }$ and of $M ^ { \ell + 1 } \mid \alpha ( s ) = j , H ^ { \ell }$ are $P _ { 0 } ^ { \cdot \ell + 1 } Q ^ { \ell + 1 }$ and $P _ { j } ^ { \ell + 1 } Q ^ { \ell + 1 }$ , where $P Q ^ { \ell + 1 }$ denotes the distribution of the output of $Q ^ { \ell + 1 } ( \cdot \mid \cdot , H ^ { \ell } )$ when its input is drawn from $P .$ The same channel appears in both distributions, and it is ϵ-LDP by Definition 1. Therefore, [Asoodeh and Zhang, 2024, Theorem 2]

gives

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } ( M ^ { \ell + 1 } \mid \alpha ( s ) = 0 , H ^ { \ell } \Vert M ^ { \ell + 1 } \mid \alpha ( s ) = j , H ^ { \ell } ) } \\ & { \leq 4 \frac { ( \exp ( \epsilon ) - 1 ) ^ { 2 } } { \exp ( \epsilon ) } D _ { \mathrm { T V } } ( S ^ { \ell + 1 } , A ^ { \ell + 1 } , R ^ { \ell + 1 } \mid \alpha ( s ) = 0 , H ^ { \ell } \Vert S ^ { \ell + 1 } , A ^ { \ell + 1 } , R ^ { \ell + 1 } \mid \alpha ( s ) = j , H ^ { \ell } ) ^ { 2 } } \end{array}\tag{11}
$$

We now upper bound the TV divergence term. Notice that, the distribution of $S ^ { \ell + 1 }$ is independent of α and $H ^ { \ell }$ , and the distribution of $\mathbf { \bar { A } } ^ { \ell + 1 }$ given $S ^ { \ell + 1 }$ and $H ^ { \ell }$ is also independent of α since $\pi ^ { \ell + 1 }$ is a function of $H ^ { \ell }$ . Furthermore, given $S ^ { \ell + 1 }$ and $A ^ { \ell + 1 }$ , the reward $R ^ { \ell + 1 }$ is independent of $H ^ { \ell }$ . Since both instances share the optimal actions $\phi ( s ^ { \prime } )$ at the states $s ^ { \prime } \neq s$ , its distribution is the same under both instances unless $S ^ { \ell + 1 } = s$ and $A ^ { \ell + 1 } = j$ , in which case it is $B ( \scriptstyle { \frac { 1 } { 2 } } )$ under $I _ { s } ^ { 0 }$ and $B ( \scriptstyle { \frac { 1 } { 2 } } + \Delta )$ under $I \mid \alpha ( s ) = j$ . Therefore, we have

$$
\begin{array} { r l } & { D _ { \nabla } ( S ^ { t + 1 } , A ^ { t + 1 } , B ^ { t + 1 } , \mathbf { \phi } ) } \\ & { = \frac { 1 } { 2 } \underset { \mathbf { c } \otimes \mathbf { c } \otimes \mathbf { c } \otimes \mathbf { c } \otimes \mathbf { c } } { \sum } | \Phi ( S ^ { t + 1 } = \boldsymbol { \alpha } , A ^ { t + 1 } = \boldsymbol { \alpha } , A ^ { t + 1 } = \boldsymbol { \alpha } ^ { t + 1 } , C ^ { t + 1 }  | \otimes ( \boldsymbol { s } ) = \boldsymbol { 0 } , H ^ { t } ) } \\ & { = \frac { 1 } { 2 } \underset { \mathbf { c } \otimes \mathbf { c } \otimes \mathbf { c } \otimes \mathbf { c } \otimes \mathbf { c } } { \sum } | \Phi ( S ^ { t + 1 } = \boldsymbol { \alpha } , A ^ { t + 1 } = \boldsymbol { \alpha } , R ^ { t + 1 } = \mathbf { c } )  } \\ & {  - \nabla \{ S ^ { t + 1 } = \mathbf { c } , A ^ { t + 1 } = \boldsymbol { \alpha } , R ^ { t + 1 } = \boldsymbol { \alpha } , R ^ { t + 1 } = \boldsymbol { \alpha } \}  } \\ & { = \frac { 1 } { 2 } \underset { \mathbf { c } \otimes \mathbf { c } \otimes \mathbf { c } \otimes \mathbf { c } } { \sum } \mathbf { \alpha } \underset { \mathbf { c } \otimes \mathbf { c } \otimes \mathbf { c } } { \sum } [ \Phi ( S ^ { t + 1 } = \boldsymbol { \alpha } ) \mathbf { \mathbb { c } }   } \\ & {       \mathbb { \frac { 1 } { 2 } }    } \\ & {      \mathbb { \frac { 1 } { 2 } }     \Phi ( R ^ { t + 1 } = \boldsymbol { \alpha } , A ^ { t } ) = \boldsymbol { \alpha } , \boldsymbol { \alpha } ( \boldsymbol { s } ) = \boldsymbol { 0 } , R ^ { t } )   } \\ &   - \nabla ( R ^ { t + 1 } = \mathbf { r } | S ^ { t + 1 } = \boldsymbol { \alpha } , A ^  t  \end{array}\tag{12}
$$

Plugin the above upper bound into (11) we get

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } ( M ^ { \ell + 1 } \mid \alpha ( s ) = 0 , H ^ { \ell } \Vert M ^ { \ell + 1 } \mid \alpha ( s ) = j , H ^ { \ell } ) } \\ & { \le \cfrac { 4 ( \exp ( \epsilon ) - 1 ) ^ { 2 } } { \exp ( \epsilon ) } \Delta ^ { 2 } \cfrac { 1 } { S ^ { 2 } } \mathbb { P } ( A ^ { \ell + 1 } = j \mid S ^ { \ell + 1 } = s , H ^ { \ell } ) ^ { 2 } } \end{array}\tag{13}
$$

Then, plugin the above upper bound into the chain rule decomposition of KL divergence Equation (10) we get

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } ( ( M ^ { i } ) _ { i \in [ K ] } \mid \alpha ( s ) = 0 \| ( M ^ { i } ) _ { i \in [ K ] } \mid \alpha ( s ) = j ) } \\ & { \le 4 \frac { ( \exp ( \epsilon ) - 1 ) ^ { 2 } } { \exp ( \epsilon ) } \Delta ^ { 2 } \frac { 1 } { S ^ { 2 } } \displaystyle \sum _ { \ell = 0 } ^ { K - 1 } \mathbb { E } _ { s , 0 } \bigg [ \mathbb { P } ( A ^ { \ell + 1 } = j \mid S ^ { \ell + 1 } = s , H ^ { \ell } ) ^ { 2 } \bigg ] . } \end{array}
$$

Lastly, plugin the above upper bound into the KL inequality (9) we get

$$
\begin{array} { r l } & { \mathbb { E } _ { s , j } \bigg [ N _ { s , j } \bigg ] \leq \mathbb { E } _ { s , 0 } \bigg [ N _ { s , j } \bigg ] } \\ & { \quad \quad \quad \quad + K \sqrt { 2 \frac { ( \exp ( \epsilon ) - 1 ) ^ { 2 } } { \exp ( \epsilon ) } \Delta ^ { 2 } \frac { 1 } { S ^ { 2 } } \sum _ { \ell = 0 } ^ { K - 1 } \mathbb { E } _ { s , 0 } \bigg [ \mathbb { P } \big ( A ^ { \ell + 1 } = j \mid S ^ { \ell + 1 } = s , H ^ { \ell } \big ) ^ { 2 } \bigg ] } . } \end{array}
$$

Summing over $j \in { \mathcal { A } } ,$ , using Cauchy-Schwarz inequality and the fact that $\begin{array} { r } { \sum _ { j \in \mathcal { A } } \mathbb { P } ( A ^ { \ell + 1 } = j } \end{array}$ $S ^ { \ell + 1 } = s , H ^ { \ell } ) ^ { 2 } \leq 1$ , we get

$$
\begin{array} { l } { \displaystyle \sum _ { j \in \mathcal { A } } \mathbb { E } _ { s , j } \bigg [ N _ { s , j } \bigg ] \leq \sum _ { j \in \mathcal { A } } \mathbb { E } _ { s , 0 } \bigg [ N _ { s , j } \bigg ] + K \sqrt { 2 A \sum _ { \ell = 0 } ^ { K - 1 } \frac { ( \exp ( \epsilon ) - 1 ) ^ { 2 } } { \exp ( \epsilon ) } \Delta ^ { 2 } \frac { 1 } { S ^ { 2 } } } } \\ { = K + K \sqrt { 2 A K \frac { ( \exp ( \epsilon ) - 1 ) ^ { 2 } } { \exp ( \epsilon ) } \Delta ^ { 2 } \frac { 1 } { S ^ { 2 } } } . } \end{array}
$$

Then, we set $\begin{array} { r } { \Delta = \operatorname* { m i n } ( \frac { 1 } { 2 } , \frac { \sqrt { A } } { 4 \sqrt { 2 K \frac { ( \exp ( \epsilon ) - 1 ) ^ { 2 } } { \exp ( \epsilon ) } } \frac { 1 } { S ^ { 2 } } } ) } \end{array}$ , the above bound becomes

$$
\sum _ { j \in \mathcal { A } } \mathbb { E } _ { s , j } \bigg [ N _ { s , j } \bigg ] \leq K + \frac { 1 } { 4 } A K
$$

Finally, plugin this upper bound into the regret decomposition (8) we get

$$
\begin{array} { r l } { \mathbb { E } \langle \mathcal { R } ( \mathfrak { A } , M , I ) | \geq \Delta \bigg ( K - \displaystyle \sum _ { k \in S , S \neq k } \bigg [ N _ { \bullet , j } \bigg ] \frac { 1 } { S A _ { j } } \bigg ) } & { } \\ & { \geq \Delta \bigg ( K - S \frac { K - \frac { 1 } { S } \Delta K } { S A } \bigg ) } \\ & { \geq \Delta \bigg ( \frac { 3 } { 4 } K - K / 4 \bigg ) } \\ & { \geq \Delta \bigg ( \frac { 3 } { 4 } K - K / 2 \bigg ) } \\ & { \geq \Delta \bigg ( \frac { 1 } { 4 } K \bigg ) } \\ & { \geq \Delta \bigg ( \frac { 1 } { 4 } K \bigg ) } \\ & { \geq \operatorname* { m i n } ( K \frac { K } { S } , \frac { S \sqrt { \lambda K } \exp ( \bar { \psi } ) } { \sqrt { 6 } \sqrt { 2 ( \exp ( \bar { \psi } ) - 1 ) } } ) } \\ & { \geq \frac { 1 } { 3 } \operatorname* { m i n } ( K , \frac { K } { S } \frac { \sqrt { \lambda K } \exp ( \bar { \psi } ) } { \sqrt { 6 } \sqrt { 6 } \sqrt { 6 } \sqrt { 6 } \sqrt { 6 } - 1 } ) . } \end{array}
$$

Then we re-start the proof again and develop the same argument but replacing the bound used in Equation (11) by the KL contraction of [Asoodeh and Zhang, 2024, Theorem 1], applied to the same ϵ-LDP channel $\mathbf { \dot { Q } } ^ { \ell + 1 } ( \cdot \mid \cdot , H ^ { \ell } )$

$$
\begin{array} { r l } & { D _ { \mathrm { R L } } ( M ^ { \ell + 1 } \mid \alpha ( s ) = 0 , H ^ { \ell } \| M ^ { \ell + 1 } \mid \alpha ( s ) = j , H ^ { \ell } ) } \\ & { \le \frac { ( \exp ( { \epsilon } ) - 1 ) ^ { 2 } } { ( \exp ( { \epsilon } ) + 1 ) ^ { 2 } } D _ { \mathrm { K L } } ( S ^ { \ell + 1 } , A ^ { \ell + 1 } , R ^ { \ell + 1 } \mid \alpha ( s ) = 0 , H ^ { \ell } \| S ^ { \ell + 1 } , A ^ { \ell + 1 } , R ^ { \ell + 1 } \mid \alpha ( s ) = j , H ^ { \ell } ) } \\ & { = \frac { ( \exp ( { \epsilon } ) - 1 ) ^ { 2 } } { ( \exp ( { \epsilon } ) + 1 ) ^ { 2 } } } \\ & { \mathbb { E } [ D _ { \mathrm { R L } } ( R ^ { \ell + 1 } \mid \alpha ( s ) = 0 , H ^ { \ell } , S ^ { \ell + 1 } , A ^ { \ell + 1 } \| R ^ { \ell + 1 } \mid \alpha ( s ) = j , H ^ { \ell } , S ^ { \ell + 1 } , A ^ { \ell + 1 } ) \mid \alpha ( s ) = 0 , H ^ { \ell } ] } \\ & { = \frac { ( \exp ( { \epsilon } ) - 1 ) ^ { 2 } } { ( \exp ( { \epsilon } ) + 1 ) ^ { 2 } } \mathbb { P } ( S ^ { \ell + 1 } = s , A ^ { \ell + 1 } = j \mid H ^ { \ell } , \alpha ( s ) = 0 ) D _ { \mathrm { R L } } ( \mathcal { B } ( \frac { 1 } { 2 } ) \| \mathcal { B } ( \frac { 1 } { 2 } + \Delta ) ) } \\ & { \le 4 \frac { ( \exp ( { \epsilon } ) - 1 ) ^ { 2 } } { ( \exp ( { \epsilon } ) + 1 ) ^ { 2 } } \Delta ^ { 2 } \frac { 1 } { S } \mathbb { P } ( A ^ { \ell + 1 } = j \mid H ^ { \ell } , S ^ { \ell + 1 } = s ) } \end{array}
$$

where the last inequality is a consequence of the fact that $\begin{array} { r } { D _ { \mathrm { K L } } ( \mathcal { B } ( \frac { 1 } { 2 } ) \| \mathcal { B } ( \frac { 1 } { 2 } + \Delta ) ) \leq 4 \Delta ^ { 2 } } \end{array}$ for $\Delta \le { \frac { 1 } { 4 } }$ The bound we obtain is analogous to the one given in Equation (13).

Proceeding as before by plugin the above upper bound in the chain rule decomposition of KL divergence Equation (10), then into the KL inequality Equation (9), summing over j using the identity

$\begin{array} { r } { \sum _ { j \in \mathcal { A } } \mathbb { P } ( A ^ { \ell + 1 } = j \ | \ S ^ { \ell + 1 } = s , H ^ { \ell } ) = 1 } \end{array}$ and applying Cauchy-Schwarz, we get

$$
\begin{array} { r l } { \displaystyle \sum _ { j \in \mathcal { A } } \mathbb { E } _ { s , j } \bigg [ N _ { s , j } \bigg ] \leq \sum _ { j \in \mathcal { A } } \mathbb { E } _ { s , 0 } \bigg [ N _ { s , j } \bigg ] + K \sqrt { 2 \mathcal { A } \sum _ { \ell = 0 } ^ { K - 1 } \frac { ( \exp ( \epsilon ) - 1 ) ^ { 2 } } { ( \exp ( \epsilon ) + 1 ) ^ { 2 } } } \Delta ^ { 2 } \frac { 1 } { S } } & { } \\ { = K + K \sqrt { 2 A K \frac { ( \exp ( \epsilon ) - 1 ) ^ { 2 } } { ( \exp ( \epsilon ) + 1 ) ^ { 2 } } } \Delta ^ { 2 } \frac { 1 } { S } . } & { } \end{array}
$$

Taking $\begin{array} { r } { \Delta = \operatorname* { m i n } ( \frac { 1 } { 4 } , \frac { \sqrt { A } } { 4 \sqrt { 2 K \frac { ( \exp { ( \epsilon ) } - 1 ) ^ { 2 } } { ( \exp { ( \epsilon ) } + 1 ) ^ { 2 } } } ^ { \frac { 1 } { 3 } } } ) } \end{array}$ , we get

$$
\sum _ { j \in \mathcal { A } } \mathbb { E } _ { s , j } \bigg [ N _ { s , j } \bigg ] \leq K + \frac { 1 } { 4 } A K
$$

Then the proof follows the same steps and yield the bound

$$
\begin{array} { r } { \mathbb { E } [ \mathcal { R } ( \mathfrak { A } , \mathcal { M } , I ) ] \geq \operatorname* { m i n } ( \frac { K } { 1 6 } , \frac { \sqrt { S A K } ( \exp ( \epsilon ) + 1 ) } { 1 6 \sqrt { 2 } ( \exp ( \epsilon ) - 1 ) } ) } \\ { \geq \frac { 1 } { 3 2 } \operatorname* { m i n } ( K , \frac { \sqrt { S A K } ( \exp ( \epsilon ) + 1 ) } { ( \exp ( \epsilon ) - 1 ) } ) } \end{array}
$$

Putting the two bounds together we get

$$
\mathbb { E } [ \mathcal { R } ( \mathfrak { A } , \mathcal { M } , I ) ] \geq \frac { 1 } { 3 2 } \operatorname* { m i n } ( K , \frac { \sqrt { \operatorname* { m a x } ( S , \exp ( \epsilon ) + 1 ) S A K ( \exp ( \epsilon ) + 1 ) } } { \exp ( \epsilon ) - 1 } )
$$

## B Proof of Theorem 2

Proof. The proof closely follows the standard analysis of Exp3 [Auer et al., 2002, Theorem 3.1]. Form the potential

$$
\Phi _ { k } = \sum _ { s \in \cal S } \ln \left( \sum _ { a \in \cal A } w _ { k } ( s , a ) \right) .
$$

We have

$$
\begin{array} { r l } { \Phi _ { k + 1 } - \Phi _ { k } = \displaystyle \sum _ { s \in \mathcal { S } } \ln \left( \frac { \sum _ { a \in \mathcal { A } } w _ { k + 1 } ( s , a ) } { \sum _ { a \in \mathcal { A } } w _ { k } ( s , a ) } \right) } & { { } } \\ { \displaystyle } & { { } = \displaystyle \sum _ { s \in \mathcal { S } } \ln \left( \sum _ { a \in \mathcal { A } } \frac { w _ { k } ( s , a ) } { \sum _ { a ^ { \prime } \in \mathcal { A } } w _ { k } ( s , a ^ { \prime } ) } \exp ( \eta \hat { V } ^ { k } ( s , a ) ) \right) } \end{array}
$$

Then notice that $\hat { V } ^ { k } ( s , a ) \in \left[ 0 , \frac { A } { \gamma } \right]$ . We choose η such that $\begin{array} { r } { \gamma = A \eta \le \frac { 1 } { 2 } } \end{array}$ so that $\eta \hat { V } ^ { k } ( s , a ) \in [ 0 , 1 ]$ We can then can apply successively the inequalities e $\operatorname { x p } ( x ) \leq 1 + x + ( e - 2 ) x ^ { 2 }$ for $x \in [ 0 , 1 ]$ and ln $( 1 + x ) \leq x$ for $x \in \mathbb { R }$ to get

$$
\begin{array} { r l } & { \Phi _ { k + 1 } - \Phi _ { k } } \\ & { \le \displaystyle \sum _ { s \in \mathcal { S } } \ln \left( 1 + \sum _ { a \in \mathcal { A } } \frac { w _ { k } ( s , a ) } { \sum _ { a ^ { \prime } \in \mathcal { A } } w _ { k } ( s , a ^ { \prime } ) } \eta \hat { V } ^ { k } ( s , a ) + \displaystyle \sum _ { a \in \mathcal { A } } \frac { w _ { k } ( s , a ) } { \sum _ { a ^ { \prime } \in \mathcal { A } } w _ { k } ( s , a ^ { \prime } ) } ( e - 2 ) \eta ^ { 2 } \hat { V } ^ { k } ( s , a ) ^ { 2 } \right) } \\ & { \le \displaystyle \sum _ { s \in \mathcal { S } } \displaystyle \sum _ { a \in \mathcal { A } } \frac { w _ { k } ( s , a ) } { \sum _ { a ^ { \prime } \in \mathcal { A } } w _ { k } ( s , a ^ { \prime } ) } \eta \hat { V } ^ { k } ( s , a ) + \displaystyle \sum _ { s \in \mathcal { S } } \displaystyle \sum _ { a \in \mathcal { A } } \frac { w _ { k } ( s , a ) } { \sum _ { a ^ { \prime } \in \mathcal { A } } w _ { k } ( s , a ^ { \prime } ) } ( e - 2 ) \eta ^ { 2 } \hat { V } ^ { k } ( s , a ) ^ { 2 } } \\ & { \le \displaystyle \sum _ { s \in \mathcal { S } } \displaystyle \sum _ { a \in \mathcal { A } } \frac { \Pi ^ { k } ( s , a ) } { 1 - \gamma } \eta \hat { V } ^ { k } ( s , a ) + \displaystyle \sum _ { s \in \mathcal { S } } \displaystyle \sum _ { a \in \mathcal { A } } \frac { \Pi ^ { k } ( s , a ) } { 1 - \gamma } ( e - 2 ) \eta ^ { 2 } \hat { V } ^ { k } ( s , a ) ^ { 2 } } \end{array}
$$

Then, summing over k, we get

$$
\begin{array} { r l } & { \Phi _ { K + 1 } - \Phi _ { 1 } } \\ & { \le \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s \in { \mathcal { S } } } \sum _ { a \in { \mathcal { A } } } \frac { \Pi ^ { k } ( s , a ) } { 1 - \gamma } \eta \hat { V } ^ { k } ( s , a ) + \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s \in { \mathcal { S } } } \sum _ { a \in { \mathcal { A } } } \frac { \Pi ^ { k } ( s , a ) } { 1 - \gamma } ( e - 2 ) \eta ^ { 2 } \hat { V } ^ { k } ( s , a ) ^ { 2 } } \end{array}
$$

We have $\begin{array} { r } { \Phi _ { 1 } = \sum _ { s \in \mathcal { S } } \ln \left( \sum _ { a \in \mathcal { A } } w _ { 1 } ( s , a ) \right) = S \ln ( A ) } \end{array}$ and

$$
\begin{array} { r l } & { \Phi _ { K + 1 } = \displaystyle \sum _ { s \in \mathcal { S } } \ln \left( \displaystyle \sum _ { a \in \mathcal { A } } w _ { K + 1 } ( s , a ) \right) } \\ & { \quad \quad = \displaystyle \sum _ { s \in \mathcal { S } } \ln \left( \displaystyle \sum _ { a \in \mathcal { A } } \exp \left( \eta \displaystyle \sum _ { k = 1 } ^ { K } { \hat { V } } ^ { k } ( s , a ) \right) \right) } \\ & { \quad \quad \geq \displaystyle \sum _ { s \in \mathcal { S } } \ln \left( \exp \left( \eta \displaystyle \sum _ { k = 1 } ^ { K } { \hat { V } } ^ { k } ( s , a ^ { * } ( s ) ) \right) \right) } \\ & { \quad \quad = \displaystyle \sum _ { s \in \mathcal { S } } \eta \displaystyle \sum _ { k = 1 } ^ { K } { \hat { V } } ^ { k } ( s , a ^ { * } ( s ) ) } \end{array}
$$

where $a ^ { \ast } ( s ) = \operatorname { a r g m a x } _ { a \in \mathcal { A } } r ( s , a )$

We therefore get the following identity

$$
\begin{array} { r l } & { \displaystyle \sum _ { s \in S } \displaystyle \sum _ { k = 1 } ^ { K } \hat { V } ^ { k } ( s , a ^ { * } ( s ) ) } \\ & { \displaystyle \leq \frac { S \ln ( A ) } { \eta } + \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s \in S } \displaystyle \sum _ { a \in A } \frac { \Pi ^ { k } ( s , a ) } { 1 - \gamma } \hat { V } ^ { k } ( s , a ) + \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s \in S } \displaystyle \sum _ { a \in A } \frac { \Pi ^ { k } ( s , a ) } { 1 - \gamma } ( \epsilon - 2 ) \eta \hat { V } ^ { k } ( s , a ) ^ { 2 } } \\ & { \displaystyle = \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { s \in S } \sum _ { a \in A } \Pi ^ { k } ( s , a ) \hat { V } ^ { k } ( s , a ) } \\ & { \displaystyle + \frac { S \ln ( A ) } { \eta } + \frac { \gamma } { 1 - \gamma } \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s \in S } \displaystyle \sum _ { a \in A } \Pi ^ { k } ( s , a ) \hat { V } ^ { k } ( s , a ) + \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s \in S } \displaystyle \sum _ { a \in A } \frac { \Pi ^ { k } ( s , a ) } { 1 - \gamma } ( \epsilon - 2 ) \eta \hat { V } ^ { k } ( s , a ) ^ { 2 } } \end{array}\tag{14}
$$

The term $\textstyle \sum _ { a \in { \mathcal { A } } } \Pi ^ { k } ( s , a ) { \hat { V } } ^ { k } ( s , a ) ^ { 2 }$ can be upper bounded as

$$
\begin{array} { r l } & { \displaystyle \sum _ { a \in \mathcal { A } } \Pi ^ { k } ( s , a ) \hat { V } ^ { k } ( s , a ) ^ { 2 } } \\ & { \displaystyle = \sum _ { a \in \mathcal { A } } \Pi ^ { k } ( s , a ) M ^ { k } ( s ) \frac { \mathbb { 1 } \big \{ a = \pi ^ { k } ( s ) \big \} } { \Pi ^ { k } ( a , s ) } \hat { V } ^ { k } ( s , a ) } \\ & { \displaystyle \leq \sum _ { a \in \mathcal { A } } \hat { V } ^ { k } ( s , a ) } \end{array}
$$

Then, we compute $\mathbb { E } [ \hat { V } ^ { k } ( s , a ) ]$

$$
\begin{array} { r l r } {  { \mathbb { E } [ \hat { V } ^ { k } ( s , a ) ] = \mathbb { E } [ M ^ { k } ( s ) \frac { \mathbb { 1 } \big \{ a = \pi ^ { k } ( s ) \big \} } { \Pi ^ { k } ( a , s ) } ] } } \\ & { } & { = \mathbb { E } [ \mathbb { E } [ M ^ { k } ( s ) \mid V ^ { k } ( s ) ] \frac { \mathbb { 1 } \big \{ a = \pi ^ { k } ( s ) \big \} } { \Pi ^ { k } ( a , s ) } ] } \end{array}
$$

The term $\mathbb { E } [ M ^ { k } ( s ) \mid V ^ { k } ( s ) ]$ is given by

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ M ^ { k } ( s ) ~ \vert ~ V ^ { k } ( s ) ] = V ^ { k } ( s ) \frac { 1 } { 2 } + ( 1 - V ^ { k } ( s ) ) \frac { 1 } { \exp ( \epsilon ) + 1 } } \\ { = V ^ { k } ( s ) ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } ) + \frac { 1 } { \exp ( \epsilon ) + 1 } } \\ { = \mathbb { 1 } \left\{ S ^ { k } = s \right\} B ^ { k } ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } ) + \frac { 1 } { \exp ( \epsilon ) + 1 } } \end{array}
$$

which yield

$$
\begin{array} { r l } & { \mathbb { E } [ \hat { V } ^ { k } ( s , a ) ] } \\ & { = \mathbb { E } [ \frac { \mathbb { 1 } \{ S ^ { k } = s , A ^ { k } = a \} } { \Pi ^ { k } ( a , s ) } B ^ { k } ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } ) ] + \frac { 1 } { \exp ( \epsilon ) + 1 } } \\ & { = \mathbb { E } [ \mathbb { 1 } \{ S ^ { k } = s \} ] r ( s , a ) ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } ) + \frac { 1 } { \exp ( \epsilon ) + 1 } } \end{array}
$$

Taking expectations in Equation (14), we get

$$
\begin{array} { r l } & { \big ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } \big ) \mathbb { E } [ \frac { K } { k - 1 } r ( S ^ { k } , a ^ { * } ( S ^ { k } ) ) - \underset { k = 1 } { K } r ( S ^ { k } , A ^ { k } ) ] } \\ & { \leq \frac { S \ln ( A ) } { \eta } + \frac { \gamma } { 1 - \gamma } \underset { k = 1 } { K } \big ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } + \frac { S } { \exp ( \epsilon ) + 1 } \big ) } \\ & { + \underset { k = 1 } { K } \frac { A } { 1 - \gamma } ( e - 2 ) \eta \big ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } + \frac { S } { \exp ( \epsilon ) + 1 } \big ) } \\ & { \leq \frac { S \ln ( A ) } { \eta } + 2 ( e - 1 ) K A \eta \big ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } + \frac { S } { \exp ( \epsilon ) + 1 } \big ) } \end{array}
$$

where we used $\gamma = A \eta$ and $\begin{array} { r } { \gamma \le \frac { 1 } { 2 } } \end{array}$

So that

$$
\begin{array} { r l } {  { \mathbb { E } [ \sum _ { k = 1 } ^ { K } r ( S ^ { k } , a ^ { * } ( S ^ { k } ) ) - \sum _ { k = 1 } ^ { K } r ( S ^ { k } , A ^ { k } ) ] } } \\ & { \leq \frac { S \ln ( A ) } { \eta ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } ) } + \frac { 2 ( e - 1 ) K A \eta ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } + \frac { S } { \exp ( \epsilon ) + 1 } ) } { ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } ) } } \end{array}
$$

$$
\begin{array} { r l } & { \mathrm { { C h o o s i n g } } \ \eta = \operatorname* { m i n } ( \frac { 1 } { 2 4 } , \sqrt { \frac { S \ln ( A ) } { 2 ( \varepsilon - 1 ) K ^ { 4 } } \big ( \frac { S \ln ( A ) } { 2 - \exp ( \varepsilon ) + 1 } + \frac { \varepsilon } { \exp ( \varepsilon ) + 1 } \big ) } ) \} \mathrm { y i d d s ~ i n ~ c a s e } \ \eta < \frac { 1 } { 2 4 } , } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & & { \quad \quad \quad = 2 \frac { \sqrt { 2 ( \varepsilon - 1 ) K ^ { 4 } \ A S \ln ( A ) \big ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \varepsilon ) + 1 } + \frac { S } { \exp ( \varepsilon ) + 1 } \big ) } } { \big ( \frac { 1 } { 2 } - \frac { 1 } { \exp ( \varepsilon ) + 1 } \big ) } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & & { = 2 \frac { \sqrt { 2 ( \varepsilon - 1 ) K A S \ln ( A ) \big ( 2 ( \exp ( \varepsilon ) + 1 ) ^ { 2 } + 4 ( S - 1 ) ( \exp ( \varepsilon ) + 1 ) \big ) } } { \exp ( \varepsilon ) - 1 } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times 4 \ln ( A ) } \end{array}\tag{15}
$$

where we used $\begin{array} { r } { \frac { 1 } { 2 } - \frac { 1 } { \exp ( \epsilon ) + 1 } = \frac { \exp ( \epsilon ) - 1 } { 2 ( \exp ( \epsilon ) + 1 ) } } \end{array}$

If $\begin{array} { r } { \eta = \frac { 1 } { 2 A } } \end{array}$ , then we have

$$
\begin{array} { r l } & { \frac { 1 } { 2 A } \leq \sqrt { \frac { S \ln ( A ) } { 2 ( e - 1 ) K A \left( \frac { 1 } { 2 } - \frac { 1 } { \exp ( e ) + 1 } + \frac { S } { \exp ( e ) + 1 } \right) } } } \\ & { \implies K \leq \frac { 2 A S \ln ( A ) } { ( e - 1 ) \left( \frac { 1 } { 2 } - \frac { 1 } { \exp ( e ) + 1 } + \frac { S } { \exp ( e ) + 1 } \right) } } \\ & { \implies K \leq \frac { 4 A S \ln ( A ) ( \exp ( e ) + 1 ) } { ( e - 1 ) \left( \exp ( e ) + 2 ( S - 1 ) \right) } } \\ & { \implies K \leq \frac { 4 A S \ln ( A ) ( \exp ( e ) + 1 ) } { ( \exp ( e ) + 1 ) } } \end{array}
$$

Finally, notice that the regret is always upper bounded by K and we can then write

$$
\begin{array} { r l } & { K = \sqrt { K } \sqrt { K } } \\ & { \quad \le \sqrt { K } \sqrt { \frac { 4 A S \ln ( A ) ( \exp ( \epsilon ) + 1 ) } { \left( \exp ( \epsilon ) - 1 \right) } } } \\ & { \quad \le 8 \frac { \exp ( \epsilon ) + 1 } { \exp ( \epsilon ) - 1 } \sqrt { S K A \ln ( A ) } } \end{array}
$$

therefore, the bound Equation (15) holds in all cases.

Finally, using the identity $x + y \le 2 \operatorname* { m a x } ( x , y )$ , we get

$$
\begin{array} { r l } & { \mathbb { E } [ \displaystyle \sum _ { k = 1 } ^ { K } r ( S ^ { k } , a ^ { * } ( S ^ { k } ) ) - \displaystyle \sum _ { k = 1 } ^ { K } r ( S ^ { k } , A ^ { k } ) ] } \\ & { \leq 1 6 \operatorname* { m a x } \left( \displaystyle \frac { \exp ( \epsilon ) + 1 } { \exp ( \epsilon ) - 1 } \sqrt { S K A \ln ( A ) } , \displaystyle \frac { \sqrt { \exp ( \epsilon ) + 1 } } { \exp ( \epsilon ) - 1 } S \sqrt { K A \ln ( A ) } \right) } \\ & { \leq 1 6 \operatorname* { m i n } ( K , \displaystyle \frac { \sqrt { \operatorname* { m a x } ( S , \exp ( \epsilon ) + 1 ) S A K ( \exp ( \epsilon ) + 1 ) \ln ( A ) } } { \exp ( \epsilon ) - 1 } ) } \end{array}
$$

## C Proof of Corollary 1

Proof. Fix an algorithm A and an $\mathrm { \epsilon - L D P }$ mechanism M and consider the instance $\tilde { I } \quad =$ $( \tilde { S } , \tilde { \mathcal { A } } , \tilde { r } , \{ \tilde { \rho } ^ { k } \} _ { k = 1 } ^ { K } )$ of Theorem 1 that achieves high regret in the private contextual bandit setting with $S - 2$ states and A arms. In instance ${ \tilde { I } } ,$ , rewards are sampled from a Bernoulli distribution with mean $\tilde { r } ( s , a )$ for each $s \in \tilde { S }$ and $a \in { \tilde { \cal A } }$

We construct an instance $I ^ { \prime }$ with S states and A actions of finite-horizon MDP $( \mathcal { S } , \mathcal { A } , P , r , \{ \rho ^ { k } \} _ { k = 1 } ^ { K } )$ as follows. The action space is $\mathcal { A } = \tilde { \mathcal { A } }$ . The state space is $s = \tilde { s } \cup$ {good, bad} having two additional states. All states have reward zero, except the good state which has reward one for every action. The good state and the bad state are absorbing. The initial distributions are defined as $\rho ^ { k } ( \mathrm { g o o d } ) = \rho ^ { k } ( \mathrm { b a d } ) = 0$ and $\rho ^ { k } ( s ) = \tilde { \rho } ^ { k } ( s )$ for each $s \in \tilde { S } .$ . Then, we choose the transition probabilities as

$$
P ( \operatorname { g o o d } \mid s , a ) = \tilde { r } ( s , a ) , \quad P ( \operatorname { b a d } \mid s , a ) = 1 - \tilde { r } ( s , a )
$$

for each $s \in \tilde { S }$ and $a \in { \mathcal { A } }$ . Fix $s \in \mathcal S \ \backslash$ {good, bad}, $a \in \mathcal { A } , k \in [ K ]$ , fix any policy π and sample a trajectory $( S _ { h } ^ { k } , A _ { h } ^ { k } ) _ { h = 1 } ^ { H }$ such that $S _ { 1 } ^ { k } = s$ and $A _ { 1 } ^ { k } = a$ . We have

$$
\begin{array} { l } { { \displaystyle \sum _ { h \in [ H ] } r ^ { \prime } ( S _ { h } ^ { k } , A _ { h } ^ { k } ) = r ^ { \prime } ( s , a ) + \sum _ { h = 2 } ^ { H } r ^ { \prime } ( S _ { h } ^ { k } , A _ { h } ^ { k } ) } } \\ { { \mathrm { ~ } } } \\ { { \mathrm { ~ } = r ^ { \prime } ( s , a ) + ( H - 1 ) \mathbb 1 \big \{ S _ { 2 } ^ { k } = \operatorname* { g o o d } \big \} } } \\ { { \mathrm { ~ } = ( H - 1 ) \mathbb 1 \big \{ S _ { 2 } ^ { k } = \operatorname { g o o d } \big \} . } } \end{array}
$$

Notice that $\mathbb { 1 } \{ S _ { 2 } ^ { k } = \mathrm { g o o d } \}$ is a Bernoulli random variable with mean $\tilde { r } ( s , a )$ . Therefore, instance I is equivalent to that of <sup>˜</sup>I but with rewards scaled by $H - 1$ . Furthermore, the bandit mechanism induced by M is $\mathrm { \epsilon - L D P }$ private since each bandit observation is mapped to a trajectory and then privatized. Therefore, the lower bound of Theorem 1 applies with a multiplicative factor of $H - 1$ and we have

$$
\mathcal { R } ( \mathfrak { A } , \mathcal { M } , I ) \geq \frac { H - 1 } { 3 2 } \operatorname* { m i n } \Bigg ( K , \frac { \sqrt { \operatorname* { m a x } \bigl ( S - 2 , \exp ( \epsilon ) + 1 \bigr ) \bigl ( S - 2 \bigr ) A K ( \exp ( \epsilon ) + 1 ) } } { \exp ( \epsilon ) - 1 } \Bigg )
$$

Then since $H \geq 2$ and $S \geq 4 .$ , we have $\begin{array} { r } { H - 1 \ge \frac { H } { 2 } } \end{array}$ and $\begin{array} { r } { S - 2 \ge \frac { S } { 2 } } \end{array}$ which concludes the proof.

## D Proof of Theorem 3

Proof. Define for each s, $, a \in \mathcal { S } \times \mathcal { A } , b \in \{ 0 , 1 \}$ and $s ^ { \prime } \in { \cal S } .$

$$
N ^ { k } ( s , a , b , s ^ { \prime } ) = \sum _ { i \in [ k ] } \sum _ { h \in [ H ] } { \mathbb 1 } \big \{ S _ { h } ^ { i } = s , A _ { h } ^ { i } = a , B _ { h } ^ { i } = b , S _ { h + 1 } ^ { i } = s ^ { \prime } \big \}
$$

The following Lemma 1 quantifies the deviation of $\hat { N } ^ { k } ( s , a , b , s ^ { \prime } )$ from $N ^ { k } ( s , a , b , s ^ { \prime } )$

Lemma 1. Let $\mathcal { E } _ { 1 }$ be the event thatfor each $s \in \mathcal S , a \in \mathcal A , b \in \{ 0 , 1 \} , s ^ { \prime } \in \mathcal S , k \in [ K ]$ and $\delta \leq { \frac { 1 } { e } }$ the following holds

$$
\begin{array} { r l } & { | \hat { N } ^ { k } ( s , a , b , s ^ { \prime } ) - N ^ { k } ( s , a , b , s ^ { \prime } ) | } \\ & { \leq \sqrt { 2 0 H \Upsilon _ { \epsilon } ^ { 2 } \displaystyle \sum _ { h \in [ H ] , i \in [ k ] } { 1 \{ a = \pi _ { h } ^ { i } ( s ) \} \log ( \frac { 8 K S ^ { 2 } A \log ( 1 6 K ) } { \delta } ) } } + 4 H \Upsilon _ { \epsilon } \log ( \frac { 8 K S ^ { 2 } A \log ( 1 6 K ) } { \delta } ) } \\ & { \leq \underbrace { \sqrt { 2 0 H \Upsilon _ { \epsilon } ^ { 2 } \displaystyle \sum _ { h \in [ H ] , i \in [ k ] } { 1 \{ a = \pi _ { h } ^ { i } ( s ) \} \log ( \frac { 4 8 K ^ { 2 } S ^ { 2 } A } { \delta } ) } } + 4 H \Upsilon _ { \epsilon } \log ( \frac { 4 8 K ^ { 2 } S ^ { 2 } A } { \delta } ) } _ { : = \partial _ { k } ( s , a ) } } \end{array}
$$

Then $\mathbb { P } ( \mathcal { E } _ { 1 } ) \ge 1 - \delta .$

The proof of Lemma 1 relies on Freedman’s inequality for martingales. We use the following version from Hsu et al. [2015][Corollary 1, with $\begin{array} { r } { c = 2 , \stackrel { - } { x } = \mathrm { i o g } ( \frac { 1 } { \delta } ) ] } \end{array}$

Theorem 4 (Freedman’s inequality Hsu et al. [2015]). Let $( X _ { i } ) _ { i \in [ K ] }$ adapted to a filtration $( { \mathcal { F } } _ { i } ) _ { i \in [ K ] }$ such that $\mathbb { E } [ X _ { i } \mid { \mathcal { F } } _ { i - 1 } ] = 0$ and $| X _ { i } | \le b f o r$ each $i \in [ K ]$ and $b \in \mathbb { R } _ { + }$

Define $\begin{array} { r } { S _ { k } = \sum _ { i = 1 } ^ { k } X _ { i } } \end{array}$ and $\begin{array} { r } { V _ { k } = \sum _ { i = 1 } ^ { k } { \mathbb { E } [ X _ { i } ^ { 2 } \mid \mathcal { F } _ { i - 1 } ] } } \end{array}$ for each $k \in [ K ]$ . Then,

$$
\mathbb { P } \left( \exists k \in [ K ] : S _ { k } \ge \sqrt { 4 V _ { k } \log ( \frac { 1 } { \delta } ) } + \frac 4 3 b \log ( \frac 1 \delta ) \right) \le ( 1 + \operatorname* { m a x } ( 1 , \lceil \log ( 2 K / \log ( \frac 1 \delta ) ) \rceil ) ) \delta
$$

from which we derive the following corollary.

Corollary 2. Under the same assumptions as Theorem 4, $i f { \frac { 1 } { \delta } } \geq e ,$ we have,

$$
\mathbb { P } \left( \exists k \in [ K ] : S _ { k } \geq \sqrt { 4 V _ { k } \log ( \frac { \log ( 1 6 K ) } { \delta } ) } + \frac { 4 } { 3 } b \log ( \frac { \log ( 1 6 K ) } { \delta } ) \right) \leq \delta
$$

Proof of Lemma 1. Define the martingale difference sequence $( X _ { k } ) _ { k \in [ K ] }$ where

$$
X _ { k } = H \hat { I } ^ { k } ( s , b , s ^ { \prime } ) \mathbb { 1 } \left\{ \pi _ { J ^ { k } } ^ { k } ( s ) = a \right\} - \sum _ { h \in [ H ] } \mathbb { 1 } \left\{ S _ { h } ^ { k } = s , A _ { h } ^ { k } = a , B _ { h } ^ { k } = b , S _ { h + 1 } ^ { k } = s ^ { \prime } \right\} .
$$

Define a filtration $( \mathcal { F } _ { k } ) _ { k \in [ K ] }$ adapted to $( X _ { k } ) _ { k \in [ K ] }$ by

$$
\mathcal { F } _ { k } = \sigma \big ( ( J ^ { i } , S _ { j } ^ { i } , A _ { j } ^ { i } , B _ { j } ^ { i } , S _ { j + 1 } ^ { i } , M ^ { i } ) _ { j \in [ H ] , i \in [ k ] } \big ) .
$$

It holds that $\mathbb { E } [ X _ { k } \mid { \mathcal { F } } _ { k - 1 } ] = 0$ for each $k \in [ K ]$ and since $| \hat { I } ^ { k } | \leq 2 \Upsilon .$ <sub>ϵ</sub> for each $k \in [ K ]$ , we have that $| X _ { k } | \le 3 \dot { H } \Upsilon ,$ <sub>ϵ</sub> for each $k \in [ K ]$ . Moreover, we have

$$
\begin{array} { r l } & { \mathbb { E } [ X _ { k } ^ { 2 } \mid \mathcal { F } _ { k - 1 } ] } \\ & { = \mathbb { E } \bigg [ \bigg ( H \hat { I } ^ { k } ( s , b , s ^ { \prime } ) \mathbb { 1 } \big \lbrace \pi _ { J ^ { k } } ^ { k } ( s ) = a \big \rbrace - \displaystyle \sum _ { h \in [ H ] } \mathbb { 1 } \big \lbrace S _ { h } ^ { k } = s , A _ { h } ^ { k } = a , B _ { h } ^ { k } = b , S _ { h + 1 } ^ { k } = s ^ { \prime } \big \rbrace \bigg ) ^ { 2 } \mid \mathcal { F } _ { k - 1 } \bigg ] } \\ & { \leq 5 H \Upsilon _ { \epsilon } ^ { 2 } \displaystyle \sum _ { h \in [ H ] } \mathbb { 1 } \big \lbrace a = \pi _ { h } ^ { k } ( s ) \big \rbrace } \end{array}
$$

Then Freedman’s inequality applied to $( X _ { k } ) _ { k }$ and $( - X _ { k } ) _ { k }$ along with a union bound over $s , a , b , s ^ { \prime } \in$ $\mathcal { S } \times \mathcal { A } \times \{ 0 , 1 \} \times \mathcal { S } \times \hat { \lvert K \rvert }$ and over the two resulting inequalities gives the proof. □

Then define $Q _ { h } ^ { * }$ for each $( s , a , h ) \in \mathcal { S } \times \mathcal { A } \times [ H ] \mathfrak { b y } Q _ { H + 1 } ^ { * } ( s , a ) = 0$ and

$$
Q _ { h } ^ { * } ( s , a ) = r ( s , a ) + \sum _ { s ^ { \prime } \in S } P ( s ^ { \prime } \mid s , a ) \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } } Q _ { h + 1 } ^ { * } ( s ^ { \prime } , a ^ { \prime } )
$$

Then, we show in Lemma 2 the following optimism property for $\tilde { Q } ^ { k , ( + ) }$

Lemma 2. With probability at least $1 - 3 \delta ,$ for each $k \in [ K ] , s \in \mathcal { S } , h \in [ H ]$ and $a \in { \mathcal { A } } ,$ , we have

$$
0 \leq \tilde { Q } _ { h } ^ { k , ( + ) } ( s , a ) - \left( r ( s , a ) + \sum _ { s ^ { \prime } \in S } P ( s ^ { \prime } \mid s , a ) \operatorname* { m a x } _ { a ^ { \prime } \in A } \tilde { Q } _ { h + 1 } ^ { k , ( + ) } ( s ^ { \prime } , a ^ { \prime } ) \right) \leq b _ { k } ( s , a , h )\tag{16}
$$

$$
\tilde { Q } _ { h } ^ { k , + } ( s , a ) \leq H - h + 1\tag{17}
$$

where

$$
b _ { k } ( s , a , h ) = \operatorname* { m i n } ( H - h + 1 , \frac { 1 8 S \beta _ { K } ( s , a ) H } { \operatorname* { m a x } ( 1 , N ^ { k } ( s , a ) - 4 S \beta _ { K } ( s , a ) ) } ) .
$$

ProofofLemma 2. Equation (17) follows easily by induction. The case $H + 1$ is satisfied since $\tilde { Q } _ { H + 1 } ^ { k , ( + ) } ( s , a ) = 0$ then assuming $\tilde { Q } _ { h + 1 } ^ { k , ( + ) } ( s , a ) \leq H - h$ , we have since $\rho \leq 1$ and $\begin{array} { r } { \sum _ { s ^ { \prime } \in \mathcal { S } } \nu ( s ^ { \prime } ) = 1 } \end{array}$ that $\tilde { Q } _ { h } ^ { k , ( + ) } ( s , a ) \leq H - h + 1$

For Equation (16), we first need to show the following Lemma 3 stating that P and r belong to the confidence sets.

Lemma 3. Call ${ \mathcal { E } } _ { 2 }$ the event thatfor each $s \in \mathcal S , a \in \mathcal A , k \in [ K ]$ , the following holds

$$
P ( s , a , \cdot ) , r ( s , a ) \in \mathcal { C } ^ { k } ( s , a )
$$

Then $\mathbb { P } ( \mathcal { E } _ { 2 } \cap \mathcal { E } _ { 1 } ) \geq 1 - 3 \delta$

ProofofLemma 3. First if $\tilde { N } ^ { k , ( - ) } ( s , a ) = 1$ and ${ \mathcal { E } } _ { 1 }$ holds, we have for each $s , a , k \in \mathcal S \times \mathcal A \times [ K ]$ that

$$
\begin{array} { r l } {  { \sum _ { s ^ { \prime } \in S , b \in \{ 0 , 1 \} } N ^ { k } ( s , a , b , s ^ { \prime } ) \leq \sum _ { s ^ { \prime } \in S , b \in \{ 0 , 1 \} } ( \hat { N } ^ { k } ( s , a , b , s ^ { \prime } ) + \beta _ { k } ( s , a ) ) } } \\ & { \leq \sum _ { s ^ { \prime } \in S , b \in \{ 0 , 1 \} } ( \hat { N } ^ { k } ( s , a , b , s ^ { \prime } ) + \beta _ { k } ( s , a ) ) } \\ & { = \sum _ { s ^ { \prime } \in S , b \in \{ 0 , 1 \} } \hat { N } ^ { k } ( s , a , b , s ^ { \prime } ) + 2 S \beta _ { k } ( s , a ) } \\ & { \leq \hat { N } ^ { k , ( - ) } ( s , a ) + 4 S \beta _ { k } ( s , a ) } \\ & { = 1 + 4 S \beta _ { k } ( s , a ) } \end{array}
$$

and thus

$$
\begin{array} { r l r } {  { \big | \widehat { N } ^ { k , ( - ) } ( s , \mathfrak { a } ) r ( s , \mathfrak { a } ) - \sum _ { s ^ { \prime } \in S } \widehat { N } ^ { k } ( s , \mathfrak { a } , \mathfrak { a } , 1 , s ^ { \prime } ) \big | } } \\ & { = | r ( s , \mathfrak { a } ) - \sum _ { s ^ { \prime } \in S } \widehat { N } ^ { k } ( s , \mathfrak { a } , 1 , s ^ { \prime } ) | } \\ & { \leq 1 + \big | \sum _ { s ^ { \prime } \in S } \widehat { N } ^ { k } ( s , \mathfrak { a } , 1 , s ^ { \prime } ) \big | } \\ & { \leq 1 + \sum _ { s ^ { \prime } \in S } \widehat { N } ^ { k } ( s , \mathfrak { a } , 1 , s ^ { \prime } ) \big | } \\ & { \leq 1 + \sum _ { s ^ { \prime } \in S } \widehat { N } ^ { k } ( s , \mathfrak { a } , 1 , s ^ { \prime } ) \big | } \\ & { \leq 1 + \sum _ { s ^ { \prime } \in S } N ^ { k } ( s , \mathfrak { a } , 1 , s ^ { \prime } ) + \beta _ { k } ( s , \mathfrak { a } ) \big ) } & { \qquad ( \mathbb { B } \mathfrak { H } _ { 1 } \operatorname { a n d } \beta _ { k } ( s , \mathfrak { a } ) > 0 ) } \\ & { \leq 2 + 5 \beta \beta _ { k } ( s , \mathfrak { a } ) } \\ & { \leq 7 \delta _ { k } ( s , \mathfrak { a } ) } \end{array}
$$

and

$$
\begin{array} { r l } & { \| \hat { N } ^ { k , ( - ) } ( s , a ) P ( s , a , \cdot ) - \displaystyle \sum _ { b \in \{ 0 , 1 \} } \hat { N } ^ { k } ( s , a , b , \cdot ) \| _ { 1 } } \\ & { = \| P ( s , a , \cdot ) - \displaystyle \sum _ { b \in \{ 0 , 1 \} } \hat { N } ^ { k } ( s , a , b , \cdot ) \| _ { 1 } } \\ & { \leq 1 + \displaystyle \sum _ { b \in \{ 0 , 1 \} , s ^ { \prime } \in S } \Big | \hat { N } ^ { k } ( s , a , b , s ^ { \prime } ) \Big | } \\ & { \leq 1 + \displaystyle \sum _ { b \in \{ 0 , 1 \} , s ^ { \prime } \in S } \big ( N ^ { k } ( s , a , b , s ^ { \prime } ) + \beta _ { k } ( s , a ) \big ) } \\ & { \leq 1 + \displaystyle \sum _ { b \in \{ 0 , 1 \} , s ^ { \prime } \in S } \big ( N ^ { k } ( s , a , b , s ^ { \prime } ) + \beta _ { k } ( s , a ) \big ) } \\ & { \leq 2 + 6 S \beta _ { k } ( s , a ) } \\ & { \leq 8 S \beta _ { k } ( s , a ) } \end{array}
$$

which implies ${ \mathcal { E } } _ { 2 }$

We now study the case $\tilde { N } ^ { k , ( - ) } ( s , a ) > 1$ . Define the martingale difference sequence $( X _ { k } ) _ { k \in [ K ] }$ where

$$
X _ { k } = H \hat { I } ^ { k } ( s , b , s ^ { \prime } ) \mathbb { 1 } \left\{ \pi _ { J ^ { k } } ^ { k } ( s ) = a \right\} - \sum _ { h \in [ H ] } \mathbb { 1 } \left\{ S _ { h } ^ { k } = s , A _ { h } ^ { k } = a , B _ { h } ^ { k } = b \right\} P ( s ^ { \prime } \mid s , a ) .
$$

Then, denoting

$$
N ^ { k } ( s , a , b ) = \sum _ { i \in [ k ] } \sum _ { h \in [ H ] } \mathbb { 1 } \left\{ S _ { h } ^ { i } = s , A _ { h } ^ { i } = a , B _ { h } ^ { i } = b \right\}
$$

and applying the same steps as in the proof of Lemma 1, we show that with probability at least $1 - \delta ,$ for each $\mathbf { \bar { \sigma } } _ { s } \in S , a \in A , b \in \{ 0 , 1 \} , s ^ { \prime } \in S , k \in [ K ]$ , it holds that

$$
| N ^ { k } ( s , a , b ) P ( s , a , s ^ { \prime } ) - \hat { N } ^ { k } ( s , a , b , s ^ { \prime } ) | \leq \beta _ { k } ( s , a ) .
$$

Equivalently, if we denote $\mathcal { E } _ { 3 }$ the event that for each $s , a , b , s ^ { \prime } , k \in \mathcal { S } \times \mathcal { A } \times \{ 0 , 1 \} \times \mathcal { S } \times [ K ]$

$$
| N ^ { k } ( s , a , b ) P ( s , a , s ^ { \prime } ) - \hat { N } ^ { k } ( s , a , b , s ^ { \prime } ) | \leq \beta _ { k } ( s , a ) ,
$$

then $\mathbb { P } ( \mathcal { E } _ { 3 } ) \ge 1 - \delta$

Consider $\mathcal { E } _ { 1 }$ and $\mathcal { E } _ { 3 }$ holds and $\tilde { N } ^ { k , ( - ) } ( s , a ) > 1$ meaning

$$
\tilde { N } ^ { k , ( - ) } ( s , a ) = \sum _ { b \in \{ 0 , 1 \} , s ^ { \prime } \in S } ( \hat { N } ^ { k } ( s , a , b , s ^ { \prime } ) - \beta _ { k } ( s , a ) ) .
$$

Then, we have that for each $s , a , s ^ { \prime } , k \in \mathcal { S } \times \mathcal { A } \times \mathcal { S } \times [ K ]$

$$
\begin{array} { r l r } & { \displaystyle \sum _ { b \in \{ 0 , 1 \} } \hat { N } ^ { k } ( s , a , b , s ^ { \prime } ) } \\ & { \ge \displaystyle \sum _ { b \in \{ 0 , 1 \} } \big ( N ^ { k } ( s , a , b ) P ( s , a , s ^ { \prime } ) - \beta _ { k } ( s , a ) \big ) } & { \mathrm { ( B y ~ \mathcal { E } _ 3 ) } } \\ & { \ge \displaystyle \sum _ { b \in \{ 0 , 1 \} } \bigg ( \displaystyle \sum _ { y \in \mathcal { S } } \big ( \hat { N } ^ { k } ( s , a , b , y ) - \beta _ { k } ( s , a ) \big ) P ( s , a , s ^ { \prime } ) - \beta _ { k } ( s , a ) \bigg ) } & { \mathrm { ( B y ~ \mathcal { E } _ 1 ) } } \\ & { = \displaystyle \tilde { N } ^ { k , ( - ) } ( s , a ) P ( s , a , s ^ { \prime } ) - 2 \beta _ { k } ( s , a ) } & { \mathrm { ( B y ~ \tilde { N } ^ { k , ( - ) } ( s , a ) > 1 ) } } \end{array}
$$

Similarly, we can write for each $s , a , s ^ { \prime } , k \in \mathcal { S } \times \mathcal { A } \times \mathcal { S } \times [ K ]$

$$
\begin{array} { r l r } {  { \sum _ { b \in \{ 0 , 1 \} } \overbrace { N ^ { k } } ( s , a , b , s ^ { \prime } ) } } \\ & { \le \sum _ { b \in \{ 0 , 1 \} } \big ( N ^ { k } ( s , a , b ) P ( s , a , s ^ { \prime } ) + \beta _ { k } ( s , a ) \big ) } & { \mathrm { ( B y ~  { \mathcal { E } _ 3 } ) } } \\ & { \le \sum _ { b \in \{ 0 , 1 \} } \bigg ( \sum _ { y \in \pm } \big ( \overbrace { N } ^ { k } ( s , a , b , y ) + \beta _ { k } ( s , a ) \big ) P ( s , a , s ^ { \prime } ) + \beta _ { k } ( s , a ) \bigg ) } & { \mathrm { ( B y ~  { \mathcal { E } _ 1 } ) } } \\ & { = \big ( \overbrace { N } ^ { k , ( - ) } ( s , a ) + 2 \sum _ { b \in \{ 0 , 1 \} , y \in S } \beta _ { k } ( s , a ) \big ) P ( s , a , s ^ { \prime } ) + 2 \beta _ { k } ( s , a ) } & { \mathrm { ( B y ~  { \tilde { N } } ^ { k , ( - ) } ( s , a ) > 1 ) } } \\ & { \le \overbrace { N } ^ { k , ( - ) } ( s , a ) P ( s , a , s ^ { \prime } ) + 4 S \beta _ { k } ( s , a ) P ( s , a , s ^ { \prime } ) + 2 \beta _ { k } ( s , a ) } & \end{array}
$$

so we get

$$
| \tilde { N } ^ { k , ( - ) } ( s , a ) P ( s , a , s ^ { \prime } ) - \sum _ { b \in \{ 0 , 1 \} } \hat { N } ^ { k } ( s , a , b , s ^ { \prime } ) | \leq 4 S \beta _ { k } ( s , a ) P ( s , a , s ^ { \prime } ) + 2 \beta _ { k } ( s , a ) P ( s , a , b , s ^ { \prime } ) .
$$

which implies

$$
\| \tilde { N } ^ { k , ( - ) } ( s , a ) P ( s , a , \cdot ) - \sum _ { b \in \{ 0 , 1 \} } \hat { N } ^ { k } ( s , a , b , \cdot ) \| _ { 1 } \leq 6 S \beta _ { k } ( s , a )
$$

meaning $P ( s , a , \cdot )$ satisfies the corresponding constraint of the confidence set $\mathcal { C } ^ { k } ( s , a )$

For the other constraint, we consider the martingale difference sequence $( X _ { k } ) _ { k \in [ K ] }$ where

$$
X _ { k } = H \hat { I } ^ { k } ( s , 1 , s ^ { \prime } ) \mathbb { 1 } \left\{ \pi _ { J ^ { k } } ^ { k } ( s ) = a \right\} - \sum _ { h \in [ H ] } \mathbb { 1 } \left\{ S _ { h } ^ { k } = s , A _ { h } ^ { k } = a , S _ { h + 1 } ^ { k } = s ^ { \prime } \right\} r ( s , a ) .
$$

and apply the same steps as in the proof of Lemma 1 to show that with probability at least $1 - \delta .$ , for each $s \in \mathcal { S } , a \in \mathcal { A } , k \in [ K ] , s ^ { \prime } \in \overset { \cdot } { \mathcal { S } }$ , it holds that

$$
\begin{array} { r } { | N ^ { k } ( s , a , s ^ { \prime } ) r ( s , a ) - \hat { N } ^ { k } ( s , a , 1 , s ^ { \prime } ) | \leq \beta _ { k } ( s , a ) . } \end{array}
$$

Equivalently, if we denote ${ \mathcal { E } } _ { 4 }$ the event that for each $s , a , s ^ { \prime } , k \in \mathcal { S } \times \mathcal { A } \times \mathcal { S } \times [ K ] .$

$$
\begin{array} { r } { | N ^ { k } ( s , a , s ^ { \prime } ) r ( s , a ) - \hat { N } ^ { k } ( s , a , 1 , s ^ { \prime } ) | \leq \beta _ { k } ( s , a ) , } \end{array}
$$

then $\mathbb { P } ( \mathcal { E } _ { 4 } ) \geq 1 - \delta .$ . Consider ${ \mathcal { E } } _ { 1 }$ and ${ \mathcal { E } } _ { 4 }$ hold and $\tilde { N } ^ { k , ( - ) } ( s , a ) \ > \ 1$ . We have that for each $s , a , s ^ { \prime } , k \in \mathcal { S } \times \mathcal { A } \times \mathcal { S } \times [ K ]$

$$
\begin{array} { r l r } {  { \sum _ { s ^ { \prime } \in S } \hat { N } ^ { k } ( s , a , 1 , s ^ { \prime } ) } } \\ & { \geq \sum _ { s ^ { \prime } \in S } \big ( N ^ { k } ( s , a , s ^ { \prime } ) r ( s , a ) - \beta _ { k } ( s , a ) \big ) } & { \mathrm { ( B y ~ \mathcal { E } _ 4 ) } } \\ & { \geq \displaystyle \sum _ { s ^ { \prime } \in S } ( \big ( \sum _ { b \in \{ 0 , 1 \} } \hat { N } ^ { k } ( s , a , b , s ^ { \prime } ) - 2 \beta _ { k } ( s , a ) \big ) r ( s , a ) - \beta _ { k } ( s , a ) ) } & { \mathrm { ( B y ~ \mathcal { E } _ 1 ) } } \\ & { \geq \displaystyle \tilde { N } ^ { k , ( - 1 ) } ( s , a ) r ( s , a ) - S \beta _ { k } ( s , a ) } & { \mathrm { ( B y ~ \tilde { N } ^ { k , ( - ) } ( s , a ) > 1 ) } } \end{array}
$$

and

$$
\sum _ { s ^ { \prime } \in \mathcal { S } } \hat { N } ^ { k } ( s , a , 1 , s ^ { \prime } )
$$

$$
\leq \sum _ { s ^ { \prime } \in S } \left( N ^ { k } ( s , a , s ^ { \prime } ) r ( s , a ) + \beta _ { k } ( s , a ) \right)
$$

(By $\mathcal { E } _ { 4 } )$

$$
\leq \sum _ { s ^ { \prime } \in S } \Bigg ( \sum _ { b \in \{ 0 , 1 \} } \big ( \hat { N } ^ { k } ( s , a , b , s ^ { \prime } ) + \beta _ { k } ( s , a ) \big ) r ( s , a ) + \beta _ { k } ( s , a ) \Bigg )
$$

(By $\mathcal { E } _ { 1 } )$

$$
\begin{array} { r l } & { \le ( \tilde { N } ^ { k , ( - ) } ( s , a ) + 4 S \beta _ { k } ( s , a ) ) r ( s , a ) + S \beta _ { k } ( s , a ) } \\ & { \le \tilde { N } ^ { k , ( - ) } ( s , a ) r ( s , a ) + 5 S \beta _ { k } ( s , a ) } \end{array}
$$

(By $\tilde { N } ^ { k , ( - ) } ( s , a ) > 1 )$

So it holds that $\mathcal { E } _ { 2 } \supseteq \mathcal { E } _ { 1 } \cap \mathcal { E } _ { 3 } \cap \mathcal { E } _ { 4 }$ and therefore $\mathbb { P } ( \mathcal { E } _ { 2 } \cap \mathcal { E } _ { 1 } ) \geq 1 - 3 \delta$

In the rest of the proof, we assume $\mathcal { E } _ { 2 } \cap \mathcal { E } _ { 1 }$ holds. First, we show the left-hand side of Equation (16). We have that for each $s , a \in \mathcal S \times \mathcal A$ and $h \in [ H ]$

$$
\begin{array} { r l } & { \tilde { Q } _ { h } ^ { k , ( + ) } ( s , a ) - \bigl ( r ( s , a ) + \sum _ { s ^ { \prime } \in S } P ( s ^ { \prime } \mid s , a ) \underset { a ^ { \prime } \in \mathcal { A } } { \operatorname* { m a x } } \tilde { Q } _ { h + 1 } ^ { k , ( + ) } ( s ^ { \prime } , a ^ { \prime } ) \bigr ) } \\ & { = \underset { \nu , \rho \in \mathcal { C } ^ { k } ( s , a ) } { \operatorname* { m a x } } \left( \rho + \sum _ { s ^ { \prime } \in S } \nu ( s ^ { \prime } ) \underset { a ^ { \prime } \in \mathcal { A } } { \operatorname* { m a x } } \tilde { Q } _ { h + 1 } ^ { k , ( + ) } ( s ^ { \prime } , a ^ { \prime } ) \right) } \\ & { - \left( r ( s , a ) + \sum _ { s ^ { \prime } \in S } P ( s ^ { \prime } \mid s , a ) \underset { a ^ { \prime } \in \mathcal { A } } { \operatorname* { m a x } } \tilde { Q } _ { h + 1 } ^ { k , ( + ) } ( s ^ { \prime } , a ^ { \prime } ) \right) } \\ & { > 0 } \end{array}\tag{By E<sub>2</sub>}
$$

Then, we show the right-hand side of Equation (16). By ${ \mathcal { E } } _ { 2 }$ and the definition of $\mathcal { C } ^ { k } ( s , a )$ , it holds that for any $\nu , \rho \in \mathcal { C } ^ { k } ( s , a )$

$$
\begin{array} { r l } & { | \tilde { N } ^ { k , ( - ) } ( s , a ) ( \rho - r ( s , a ) ) | } \\ & { \leq | \tilde { N } ^ { k , ( - ) } ( s , a ) \rho - \displaystyle \sum _ { s ^ { \prime } \in \cal S } \hat { N } ^ { k } ( s , a , 1 , s ^ { \prime } ) | + | \sum _ { s ^ { \prime } \in \cal S } \hat { N } ^ { k } ( s , a , 1 , s ^ { \prime } ) - \tilde { N } ^ { k , ( - ) } ( s , a ) r ( s , a ) | } \\ & { \leq 1 4 \cal S \beta _ { K } ( s , a ) } \end{array}
$$

Due to $\mathcal { E } _ { 1 } , \tilde { N } ^ { k , ( - ) } ( s , a ) \geq \operatorname* { m a x } ( 1 , N ^ { k } ( s , a ) - 4 S \beta _ { K } ( s , a ) )$ so that we have

$$
\vert \rho - r ( s , a ) \vert \leq \frac { 1 4 S \beta _ { K } ( s , a ) } { \operatorname* { m a x } ( 1 , N ^ { k } ( s , a ) - 4 S \beta _ { K } ( s , a ) ) }
$$

Similarly for ν, we have

$$
\begin{array} { r l } & { \| \tilde { N } ^ { k , ( - ) } ( s , a ) ( \nu - P ( s , a , \cdot ) ) \| _ { 1 } } \\ & { \leq \| \tilde { N } ^ { k , ( - ) } ( s , a ) \nu - \displaystyle \sum _ { b \in \{ 0 , 1 \} } \hat { N } ^ { k } ( s , a , b , \cdot ) \| _ { 1 } + \| \displaystyle \sum _ { b \in \{ 0 , 1 \} } \hat { N } ^ { k } ( s , a , b , \cdot ) - \tilde { N } ^ { k , ( - ) } ( s , a ) P ( s , a , \cdot ) \| _ { 1 } } \\ & { \leq 1 6 S \beta _ { K } ( s , a ) } \end{array}
$$

and due to $\mathcal { E } _ { 1 } , \tilde { N } ^ { k , ( - ) } ( s , a ) \geq \operatorname* { m a x } ( 1 , N ^ { k } ( s , a ) - 4 S \beta _ { K } ( s , a ) )$ ) so that we have

$$
\| \nu - P ( s , a , \cdot ) \| _ { 1 } \leq \frac { 1 6 S \beta _ { K } ( s , a ) } { \operatorname* { m a x } ( 1 , N ^ { k } ( s , a ) - 4 S \beta _ { K } ( s , a ) ) }
$$

Then, we conclude the proof by writing for each $s , a \in \mathcal S \times \mathcal A$ and $h \in [ H ]$

$$
\begin{array} { r l } & { \mathbb { E } _ { \xi \sim \frac { 1 } { \tau } } ^ { \xi \sim \frac { 1 } { \tau } \nu _ { 1 } } ( \xi , s ) = \frac { 1 } { \tau } \sum _ { j = 1 } ^ { \infty } \frac { \nu _ { j } } { \mu _ { j } } \sum _ { \ell = 1 } ^ { \infty } \exp { \frac { \xi \log \left( \frac { \eta _ { j } ^ { \ell } } { \mu _ { j } } \right) } { \mu _ { j } } } \leq \frac { \mu _ { j } } { \mu _ { j } } \sum _ { \ell = 1 } ^ { \infty } \frac { \nu _ { j } } { \mu _ { j } } } \\ & { = - \frac { 1 } { \sqrt { \pi } \sqrt { \mu _ { j } \pi } } \exp \left( \frac { \pi \sqrt { \mu _ { j } } } { \sqrt { \pi } \xi } \right) \frac { \sqrt { \pi } } { \mu _ { j } } \exp \left( \frac { \sqrt { \mu _ { j } } \xi } { \sqrt { \pi } \sqrt { \mu _ { j } \pi } } \right) ^ { 2 } , } \\ & { = \left( \frac { 1 } { \sqrt { \pi } \sqrt { \mu _ { j } \pi } } \right) \frac { \sqrt { \pi } } { \mu _ { j } } \exp \left( \frac { \sqrt { \pi } \sqrt { \mu _ { j } \pi } } { \sqrt { \pi } \sqrt { \mu _ { j } \pi } } \right) \frac { \sqrt { \pi } } { \mu _ { j } ^ { 2 } \sqrt { \pi } } \exp \left( \frac { \pi \sqrt { \mu _ { j } \pi } } { \sqrt { \mu _ { j } \pi } } \right) } \\ & { \quad \times \frac { 1 } { \sqrt { \pi } \sqrt { \mu _ { j } \pi } } \exp \left( \frac { \pi \sqrt { \mu _ { j } \pi } } { \sqrt { \pi } \sqrt { \mu _ { j } \pi } } \right) \frac { \sqrt { \pi } } { \mu _ { j } } \exp \left( \frac { \pi \sqrt { \mu _ { j } \pi } } { \sqrt { \pi } \sqrt { \mu _ { j } \pi } } \right) } \\ &  \quad \times \frac { 1 }  \sqrt  \pi \end{array}
$$

Furthermore, since $\tilde { Q } _ { h } ^ { k , ( + ) } ( s , a ) \leq H - h + 1$ and

$$
\tilde { Q } _ { h } ^ { k , ( + ) } ( s , a ) - \left( r ( s , a ) + \sum _ { s ^ { \prime } \in S } P ( s ^ { \prime } \mid s , a ) \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal A } \tilde { Q } _ { h + 1 } ^ { k , ( + ) } ( s ^ { \prime } , a ^ { \prime } ) \right) \geq 0 ,
$$

we have

$$
\tilde { Q } _ { h } ^ { k , ( + ) } ( s , a ) - \left( r ( s , a ) + \sum _ { s ^ { \prime } \in S } P ( s ^ { \prime } \mid s , a ) \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } } \tilde { Q } _ { h + 1 } ^ { k , ( + ) } ( s ^ { \prime } , a ^ { \prime } ) \right) \le H - h + 1
$$

thus concluding the proof.

Then, we show the optimism of $\begin{array} { r } { \tilde { V } ^ { k , ( + ) } ( s ) = \operatorname* { m a x } _ { a \in \mathcal { A } } \tilde { Q } _ { h } ^ { k , ( + ) } ( s , a ) } \end{array}$ for each $s , h , k \in \mathcal { S } \times [ H ] \times$ $[ K ] \cup \{ 0 \}$ . Define $\begin{array} { r } { \hat { V } _ { h } ^ { * } ( s ) = \operatorname* { m a x } _ { a \in \mathcal { A } } \hat { Q } _ { h } ^ { * } ( s , a ) } \end{array}$ for each $s \in S$ and $h \in [ H ]$ where $Q _ { h } ^ { * } ( s , a ) =$ $\begin{array} { r } { \bar { r } ( s , a ) + \sum _ { s ^ { \prime } \in S } P ( s ^ { \prime } \mid s , a ) } \end{array}$ max $\cdot a ^ { \prime } \in \mathcal { A } \stackrel { \cdot \cdot } { Q } _ { h + 1 } ^ { * } ( s ^ { \prime } , a ^ { \prime } )$ for each $s \in \mathcal S , a \in \mathcal A$ and $h \in [ H ]$ with the convention that $Q _ { H + 1 } ^ { * } ( s , a ) = 0$ for each $s \in S , a \in A$ . We show the following result.

Lemma 4. Under $\mathcal { E } _ { 2 } \cap \mathcal { E } _ { 1 } ,$ , for each $s \in \mathcal S , h \in [ H ] , k \in [ K ] \cup \{ 0 \}$ , we have

$$
\tilde { V } _ { h } ^ { k , ( + ) } ( s ) \geq V _ { h } ^ { * } ( s ) .
$$

ProofofLemma 4. The result for $k = 0$ comes from $V _ { h } ^ { * } \leq H$ . For $k > 0$ , assume $\mathcal { E } _ { 2 } \cap \mathcal { E } _ { 1 }$ holds. We show the result by induction. The case $H + 1$ is satisfied since $\tilde { V } _ { H + 1 } ^ { k , ( + ) } ( s ) = 0$ and $V _ { H + 1 } ^ { * } ( s ) = 0$ for

each $s \in S .$ . Then, assuming $\tilde { V } _ { h + 1 } ^ { k , ( + ) } ( s ) \geq V _ { h + 1 } ^ { * } ( s )$ for each $s \in S .$ , we have that for each $s \in S ,$

$$
\begin{array} { r l } & { \hat { V } _ { h } ^ { k , ( i + ) } ( s ) - V _ { h } ^ { k } ( s ) } \\ & { = \operatorname* { m a x } _ { \alpha \in A } \hat { Q } _ { h } ^ { k , ( i + ) } ( s , a ) - \operatorname* { m a x } _ { \alpha \in A } Q _ { h } ^ { k } ( s , a ) } \\ & { \geq \operatorname* { m a x } _ { \alpha \in A } \gamma ( s , a ) + \sum _ { \sigma \in A } P ( s ^ { \prime } \mid s , a ) \hat { V } _ { h + 1 } ^ { k , ( i + ) } ( s ^ { \prime } ) } \\ & { - \operatorname* { m a x } _ { \alpha \in A } ( r ( s , a ) + \sum _ { \sigma \in A } P ( s ^ { \prime } \mid s , a ) V _ { h + 1 } ^ { k } ( s ^ { \prime } ) ) \quad \quad \mathrm { ( B y ~ \mathcal { E } _ 2 ~ a n d ~ \mathcal { E } _ 1 ~ a n d ~ L e m m a 2 ) } } \\ & { \geq \operatorname* { m a x } _ { \alpha \in A } r ( s , a ) + \sum _ { \sigma \in B } P ( s ^ { \prime } \mid s , a ) V _ { h + 1 } ^ { \star } ( s ^ { \prime } ) } \\ & { - \operatorname* { m a x } _ { \alpha \in A } \big ( r ( s , a ) + \sum _ { s ^ { \prime } \in B } P ( s ^ { \prime } \mid s , a ) V _ { h + 1 } ^ { \star } ( s ^ { \prime } ) \big ) \quad \quad \mathrm { ( B y ~ \mathcal { E } _ 2 ~ a n d ~ \mathcal { E } _ i ~ a n d ~ i n d u a r i o n ~ h y p o t h e s i s ) } } \\ & { \geq 0 } \end{array}
$$

Finally, we show the regret bound. Assume $\mathcal { E } _ { 2 } \cap \mathcal { E } _ { 1 }$ holds. We will first show by induction that for $k \in [ \check { K } ]$ and $h \in [ H ]$

$$
\begin{array} { r l } & { \mathbb { E } [ \tilde { V } _ { h } ^ { k - 1 , ( + ) } ( S _ { h } ^ { k } ) - V _ { h } ^ { \pi ^ { k } } ( S _ { h } ^ { k } ) \mid \mathcal { F } _ { k - 1 } , S _ { h } ^ { k } ] } \\ & { \leq \mathbb { E } [ \displaystyle \sum _ { i = h } ^ { H } b _ { k - 1 } ( S _ { i } ^ { k } , \pi _ { i } ^ { k } ( S _ { i } ^ { k } ) ) \mid \mathcal { F } _ { k - 1 } , S _ { h } ^ { k } ] } \end{array}\tag{18}
$$

The above result is true for $h = H + 1$ since $\tilde { V } _ { H + 1 } ^ { k - 1 , ( + ) } ( s ) = 0$ and $V _ { H + 1 } ^ { \pi ^ { k } } ( s ) = 0$ for each $s \in S$ Then, assuming the result holds for $h + 1$ , we have that for each $k \in [ K ]$ that

$$
\begin{array} { r l r } & { \mathbb { E } [ \tilde { V } _ { h } ^ { k - 1 , ( + ) } ( S _ { h } ^ { k } ) - V _ { h } ^ { n ^ { k } } ( S _ { h } ^ { k } ) \ \vert \mathcal { F } _ { k - 1 } , S _ { h } ^ { k } ] } & \\ & { = \mathbb { E } [ \underset { 0 \leq k \leq i } { \operatorname* { m a x } } \tilde { Q } _ { h } ^ { k - 1 , ( + ) } ( S _ { h } ^ { k } , a ) - Q _ { h } ^ { k } ( S _ { h } ^ { k } , \pi _ { h } ^ { k } ( S _ { h } ^ { k } ) ) \ \vert \mathcal { F } _ { k - 1 } , S _ { h } ^ { k } ] } & \\ & { = \mathbb { E } [ \tilde { Q } _ { h } ^ { k - 1 , ( + ) } ( S _ { h } ^ { k } , \pi _ { h } ^ { k } ( S _ { h } ^ { k } ) ) - Q _ { h } ^ { n ^ { k } } ( S _ { h } ^ { k } , \pi _ { h } ^ { k } ( S _ { h } ^ { k } ) ) \ \vert \mathcal { F } _ { k - 1 } , S _ { h } ^ { k } ] } & { \mathrm { ( B y ~ t h e ~ d e f i n i t i o n ~ o f ~ } \pi ^ { k } ) } \\ & { \leq \mathbb { E } [ \tilde { V } _ { h + 1 } ^ { k - 1 , ( + ) } ( S _ { h + 1 } ^ { k } ) + b _ { k - 1 } ( S _ { h } ^ { k } , \pi _ { h } ^ { k } ( S _ { h } ^ { k } ) ) - V _ { h + 1 } ^ { n ^ { k } } ( S _ { h + 1 } ^ { k } ) \ \vert \mathcal { F } _ { k - 1 } , S _ { h } ^ { k } ] } & { \mathrm { ( B y ~ \mathcal { E } _ 2 ~ a n d ~ \mathcal { E } _ 1 ~ a n d ~ L e m m a ~ 2 ) } } \\ &  \leq \mathbb { E } [ b _ { k - 1 } ( S _ { h } ^ { k } , \pi _ { h } ^ { k } ( S _ { h } ^ { k } ) ) + \mathbb { E } [ \displaystyle \sum _ { i = h + 1 } ^ { H } b _ { k - 1 } ( S _  \end{array}
$$

Then, the regret bound can be derived as follows. Given $s \in S$ , we have

$$
\begin{array} { r l r } & { \displaystyle \sum _ { k = 1 } ^ { K } \left( V _ { 1 } ^ { * } ( s ) - V _ { 1 } ^ { \pi ^ { k } } ( s ) \right) } \\ & { \le \displaystyle \sum _ { k = 1 } ^ { K } \left( \tilde { V } _ { 1 } ^ { k - 1 , ( + ) } ( s ) - V _ { 1 } ^ { \pi ^ { k } } ( s ) \right) } & { \mathrm { ( B y ~ L e m m a ~ 4 ) ~ } } \\ & { \le \displaystyle \sum _ { k = 1 } ^ { K } \mathbb { E } [ \displaystyle \sum _ { h = 1 } ^ { H } b _ { k - 1 } ( S _ { h } ^ { k } , A _ { h } ^ { k } ) \mathbb { 1 } \{ \mathcal { E } _ { 1 } \cap \mathcal { E } _ { 2 } \} \mid S _ { 1 } ^ { k } = s ] + 3 H ^ { 2 } K \delta } & { \mathrm { ( B y ~ E q u a t i o n ~ ( 1 8 ) ) ~ } } \end{array}
$$

We set $\begin{array} { r } { \delta = \frac { 1 } { 3 K } } \end{array}$ and then focus on $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \sum _ { h = 1 } ^ { H } b _ { k - 1 } ( S _ { h } ^ { k } , A _ { h } ^ { k } ) } \end{array}$

$$
\begin{array} { l } { \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { h = 1 } ^ { H } b _ { k - 1 } ( S _ { h } ^ { k } , A _ { h } ^ { k } ) } \\ { = \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { h = 1 } ^ { H } \operatorname* { m i n } \big ( H - h + 1 , \frac { 1 8 S \beta _ { K } ( S _ { h } ^ { k } , \pi _ { h } ^ { k } ( S _ { h } ^ { k } ) ) H } { \operatorname* { m a x } ( 1 , N ^ { k - 1 } ( S _ { h } ^ { k } , \pi _ { h } ^ { k } ( S _ { h } ^ { k } ) ) - 4 S \beta _ { K } ( S _ { h } ^ { k } , \pi _ { h } ^ { k } ( S _ { h } ^ { k } ) ) ) } \big ) } \end{array}
$$

Define

$$
N _ { h } ^ { k } ( s , a ) = N ^ { k - 1 } ( s , a ) + \sum _ { i = 1 } ^ { h - 1 } \mathbb { 1 } \big \{ S _ { i } ^ { k } = s , A _ { i } ^ { k } = a \big \}
$$

for each $s \in \mathcal S , a \in \mathcal A , h \in [ H ]$ and $k \in [ K ]$ . It holds that $N ^ { k - 1 } ( s , a ) \geq N _ { h } ^ { k } ( s , a ) - H$ so we can write

$$
\begin{array} { r l } & { \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { h = 1 } ^ { H } b _ { k - 1 } ( S _ { h } ^ { k } , A _ { h } ^ { k } ) } \\ & { \leq \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { h = 1 } ^ { H } \operatorname* { m i n } \big ( H - h + 1 , \frac { 1 8 \mathcal { S } \beta _ { K } ( S _ { h } ^ { k } , \pi _ { h } ^ { k } ( S _ { h } ^ { k } ) ) H } { \operatorname* { m a x } ( 1 , N _ { h } ^ { k } ( S _ { h } ^ { k } , \pi _ { h } ^ { k } ( S _ { h } ^ { k } ) ) - 4 \mathcal { S } \beta _ { K } ( S _ { h } ^ { k } , \pi _ { h } ^ { k } ( S _ { h } ^ { k } ) ) - H ) } \big ) } \\ & { = \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { h = 1 } ^ { H } \sum _ { s \in S } \displaystyle \sum _ { a \in A } \sum _ { n \in [ K H ] } \mathbf { 1 } \big \{ S _ { h } ^ { k } = s , A _ { h } ^ { k } = a , N _ { h } ^ { k } ( s , a ) = n \big \} } \\ & { \operatorname* { m i n } \big ( H , \frac { 1 8 \mathcal { S } \beta _ { K } ( s , a ) H } { \operatorname* { m a x } ( 1 , n - 4 \mathcal { S } \beta _ { K } ( s , a ) - H ) } \big ) } \end{array}\tag{19}
$$

Then, notice that given $\begin{array} { r } { s , a , n \in \mathcal S \times A \times [ K H ] , \mathbb { 1 } \big \{ S _ { h } ^ { k } = s , A _ { h } ^ { k } = a , N _ { h } ^ { k } ( s , a ) = n \big \} } \end{array}$ is zero except for maybe a single couple k, h we call $k ( s , a , n ) , h ( s , a , n )$ , so we have

$$
\begin{array} { r l } & { \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { h \in - 1 } ^ { H } \delta _ { h _ { k } - 1 } ( S _ { h _ { k } } ^ { h } , \lambda _ { h _ { k } } ^ { h } ) } \\ & { \le \alpha \operatorname* { l i m s i n } _ { s \le t \le N \le 1 } \mathbb { 1 } \left\{ S _ { h ( s , \alpha , n ) } ^ { h ( s , a , n ) } = s , A _ { h ( s , n ) } ^ { k ( s , a , n ) } = \alpha , N _ { h ( s , \alpha , n ) } ^ { k ( s , a , n ) } ( s , \alpha ) = n \right\} } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \displaystyle \operatorname* { m i n } ( H , \underbrace { 1 \times 3 \delta \beta _ { k } ( s , a ) H } _ { \le \alpha \times 1 } ) \frac { 1 } { s } \beta \alpha ( s , a ) - H ) } \\ & { \le \alpha \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \operatorname* { m i n } ( H , \underbrace { 1 \times 3 \beta _ { k } ( s , a ) H } _ { \mathrm { m a x } ( 1 , n , n - 1 } ) ) } \\ & { \quad \le \alpha , \displaystyle \operatorname* { m a x } _ { \delta \le h \le \alpha } \left\{ R H \right\} } \\ & { \le 4 H S \quad \displaystyle \sum _ { s \le h \le \alpha } \delta \mu _ { k } ( s , a ) + H ^ { 2 } S A + 1 8 S \int _ { s \le N \le s } \beta _ { k } ( s , a ) H \log ( K H ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \le 2 H S \log ( 3 K H ) \bigcup \displaystyle \sum _ { s \in \delta \le s \le 4 } \beta _  k \end{array}
$$

and then we can bound $\textstyle \sum _ { s , a \in { \mathcal { S } } \times { \mathcal { A } } } \beta _ { K } ( s , a )$ as follows.

$$
\begin{array} { r l } & { \underset { s \in \cal S , a \in \cal A } { \sum } \beta _ { K } ( s , a ) } \\ & { \le \underset { s \in \cal S , a \in \cal A } { \sum } \sqrt { 2 0 H \Upsilon _ { \epsilon } ^ { 2 } \sum _ { h \in [ H ] , i \in [ K ] } 1 \{ a = \pi _ { h } ^ { i } ( s ) \} \log ( 1 4 4 K ^ { 3 } S ^ { 2 } A ) } } \\ & { + 4 H S A \Upsilon _ { \epsilon } \log ( 1 4 8 K ^ { 3 } S ^ { 2 } A ) } \\ & { \le S \sqrt { 2 0 \Upsilon _ { \epsilon } ^ { 2 } A H ^ { 2 } K \log ( 1 4 8 K ^ { 3 } S ^ { 2 } A ) } + 4 H S A \Upsilon _ { \epsilon } \log ( 1 4 8 K ^ { 3 } S ^ { 2 } A ) } \end{array}
$$

The final regret bound is therefore given by

$$
\begin{array} { r l } & { ~ \displaystyle \sum _ { k \in [ K ] } \left( V _ { 1 } ^ { * } ( s ) - V _ { 1 } ^ { \pi ^ { k } } ( s ) \right) } \\ & { \le 2 2 H ^ { 2 } S ^ { 2 } \sqrt { 2 0 \Upsilon _ { \epsilon } ^ { 2 } A K \log ( 1 4 8 K ^ { 3 } S ^ { 2 } A ) } \log ( 3 K H ) + 8 8 H ^ { 2 } S ^ { 2 } A \Upsilon _ { \epsilon } \log ( 1 4 8 K ^ { 3 } S ^ { 2 } A ) \log ( 3 K H ) } \\ & { + H ^ { 2 } S A + H ^ { 2 } } \\ & { \le 2 2 H ^ { 2 } S ^ { 2 } \sqrt { 2 0 \Upsilon _ { \epsilon } ^ { 2 } A K \log ( 1 4 8 K ^ { 3 } S ^ { 2 } A ) } \log ( 3 K H ) + 9 2 H ^ { 2 } S ^ { 2 } A \Upsilon _ { \epsilon } \log ( 1 4 8 K ^ { 3 } S ^ { 2 } A ) \log ( 3 K H ) } \end{array}
$$

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state the minimax regret bounds claimed in the paper. All claims are supported by formal theorems (Theorem 1, Theorem 2, Corollary 1, Theorem 3) with complete proofs, and the scope (time-homogeneous episodic tabular RL under non-interactive ϵ-LDP) is precisely defined.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The conclusion section discusses the remaining gap of HS between lower and upper bounds for the high-privacy regime $( \epsilon \leq 1 )$ , the sub-optimality of the upper bound in the low-privacy regime, and the conjecture that the $S ^ { 2 }$ dependency is unimprovable for model-based algorithms.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: All theorems and corollaries are numbered, their assumptions are explicitly stated, and complete proofs are provided in the appendix (Sections A, B and D). Proof sketches in the main body provide intuition for the full appendix proofs.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [N/A]

Justification: The paper is purely theoretical and contains no experiments.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.

In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [N/A]

Justification: The paper is purely theoretical and contains no experiments requiring code or data.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [N/A]

Justification: The paper is purely theoretical and contains no experiments.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [N/A]

Justification: The paper is purely theoretical and contains no experiments.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [N/A]

Justification: The paper is purely theoretical and contains no experiments.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification:

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [N/A]

Justification: This is foundational theoretical research on privacy-preserving reinforcement learning. The work contributes to understanding fundamental limits of private learning, with no direct path to negative applications.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper releases no models, datasets, or other assets that pose a risk of misuse.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [N/A]

Justification: The paper does not use any external code, datasets, or models. All prior works are properly cited.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: The paper introduces no new datasets, models, or code assets.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing or research with human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing or research with human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: ChatGPT 5.5 was used to brainstorm proof strategies, verifying proposed algorithms and proofs, proofreading and polishing the manuscript. The authors verified the correctness and originality of all content.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.