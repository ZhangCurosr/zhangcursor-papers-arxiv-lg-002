# USING TRANSFORMERS FOR TEACHING AND LEARNING VIA SYMBOLIC DISTILLATION OF NEURAL A-LIFE AGENTS

## A PREPRINT

Shuwei Wang<sup>1</sup>, Vadim Bulitko<sup>1</sup>, Matthew R. G. Brown<sup>1</sup>, Michael Youngblood<sup>2</sup>, Ramon Lawrence<sup>3</sup>, Roscoe Smith<sup>1</sup>, Shinichi Nakagawa<sup>4</sup>, and William Yeoh<sup>5</sup>

<sup>1</sup>Department of Computing Science, University of Alberta, 116 Street and 85 Avenue, Edmonton, AB, T6G 2R3, Canada

<sup>2</sup>Filuta AI, 1606 Headway Circle, Austin, TX 78754, USA

<sup>3</sup>Department of Computer Science, Data Science, University of British Columbia, 6200 University Blvd, Vancouver, BC Canada V6T 1Z4, Canada

<sup>4</sup>Department of Biology, University of Alberta, 116 Street and 85 Avenue, Edmonton, AB, T6G 2R3, Canada <sup>5</sup>McKelvey School of Engineering, Washington University in St. Louis, 1 Brookings Drive, St. Louis, MO 63130-4899, USA

{shuwei4,bulitko}@ualberta.ca, mrbrown23@gmail.com, gmyoungblood@gmail.com, ramon.lawrence@ubc.ca, roscoe1@ualberta.ca, snakagaw@ualberta.ca, william.yeoh@gmail.com

November 6, 2026

## ABSTRACT

Neural networks controlling behavior of A-life agents can evolve over multiple generations. While effective in making the agents survive, such networks are generally opaque to a human observer. Furthermore, the neural weights cannot be transferred between agents with different neural architectures. Distilling such neural networks into compact symbolic policies can help humans understand agent behavior in A-life. Additionally, it allows the agents to share their knowledge in a compact fashion, improving population-level learning. Distilling a neural policy into a symbolic policy is typically performed by searching the space of symbolic policies to find one that approximates the neural policy. However, even with a domain-specific language the space of policies can be large, necessitating incomplete stochastic search such as genetic algorithms. Yet the distillation remains computationally expensive. We propose to instead use a transformer pre-trained on readily available random percept-action data. Doing so speeds up distillation substantially. We derive the transformer architecture via a neural architecture search. Empirical results compare different distillation methods and the influence of the symbolic space on A-life population dynamics through self-reflection and teaching and learning.

Keywords Artificial life · Symbolic distillation · Transformers · Neurosymbolic AI · Teaching and learning

## 1 Introduction

While neural networks have enabled breakthrough performance in many applications over the last decade [Mnih et al., 2015] their learned knowledge is encoded non-transparently via a large number of trained parameters (weights). The training process itself can be unstable and the learned knowledge cannot be compactly transferred between different neural architectures. Neurosymbolic Artificial Intelligence (AI) [Chaudhuri et al., 2021, Kautz, 2022, Hitzler et al., 2022, Garcez and Lamb, 2023] seeks to address these three shortcomings by complementing neurally represented knowledge with symbolically represented knowledge. One way of doing so is distillation of neural control policies: finding a compact symbolic policy of similar behavior [Verma et al., 2018].

The distillation methods used in this process often come from the field of symbolic regression [Gulwani et al., 2017] and involve searching a space of programs [Odena et al., 2020]. Because the space of programs is usually large, incomplete stochastic methods are utilized. In particular, stochastic search in the form of simulated annealing [Medeiros et al., 2022] and genetic algorithms (GA) [Koza, 1992, Cranmer et al., 2020] is common in program synthesis [Bulitko et al., 2022, Wang et al., 2023] and neural distillation [Deproost et al., 2024]. While it has been successfully used for neurosymbolic AI [Udrescu and Tegmark, 2020, Acharya et al., 2023] it can be slow and unreliable.

In this paper we adopt the idea of symbolic regression via pre-trained transformers [Biggio et al., 2021, Vaswani et al., 2017] and synthesize compact symbolic policies of neural agents evolved within A-life. The transformer receives a set of an agent’s percepts and the actions taken by the agent and produces a series of tokens describing a symbolic policy that approximately matches the percept-action pairs. We use a neural architecture search [Chen et al., 2021] to automatically find an effective neural architecture for the transformer. Once trained the transformer can be enhanced with bagging [Kamienny et al., 2022] and beam search [Vaswani et al., 2017]. We show that not only can it surpass the accuracy of symbolic policies produced by genetic algorithms but its distillation is also substantially faster.

The distillation allows a population of A-life agents to exchange their knowledge in a compact, symbolic fashion. Such knowledge sharing enables population-level learning. The compact nature of the symbolic policies also makes them human-readable which may lead to theoretical verifiability [Zhu et al., 2019, Bastani et al., 2018, Stevens et al., 2023] and human insight [Wang et al., 2024]. The symbolic policies can also be used to stabilize the neural training in an iterative manner [Verma et al., 2018, 2019] and to communicate knowledge between agents with different neural architectures [Stanley and Miikkulainen, 2002].

The paper makes the following contributions. We propose a method for automatically designing and training a transformer for distilling neural policies of A-life agents. Once trained the transformer surpasses the distillation accuracy of a genetic algorithm on average while being substantially faster. The short per-policy distillation time makes teaching and learning within A-life more practical. In particular, we consider the impact of the space of symbolic policies used for the distillation on teaching and learning in the population. Finally, we discuss human readability of symbolic policies. Our code and data are available to other researchers upon request.

## 2 Problem Formulation

A neural control policy π of an A-life agent maps the agent’s percept p to an action $\pi ( p ) = a$ . The policy π is not directly available to the distiller which instead receives a set of input (agent percept) and output (agent action) pairs: $\{ ( p _ { i } , \bar { \pi } ( p _ { i } ) ) \}$ }. Then given a space M of symbolic policies the distillation task for agent policy π is to find a policy $m ^ { * } \in M$ with the maximum accuracy on the percept-action set $( p _ { i } , a _ { i } ) \colon m ^ { * } = \operatorname { a r g m a x } \bar { \mathcal { A } } ( m , \{ ( p _ { i } , \pi ( p _ { i } ) ) \} )$ where m∈M

A is the percentage of percepts $p _ { i }$ on which the symbolic policy gives the same action as the agent’s neural policy: $m ( p _ { i } ) = \pi ( p _ { i } )$

We evaluate our distillers on their ability to accurately distill neural agents evolved in A-life. The corresponding performance measure is the distillation accuracy. We generalize this definition to percept-action pairs whose actions come from other sources (e.g., an unknown symbolic policy). For instance, given a set of symbolic policies one can create a set of percept-action pairs for each of them by running each policy on random percepts. Then the distillation accuracy of a distiller is the proportion of actions produced by a symbolic policy that match the actions of the original policy.

We prefer our distiller to be constructed automatically. We also prefer that it produces a compact symbolic policy to help humans with readability and theoretical verifiability. Within A-life compact symbolic policies are faster to transmit and interpret. Thus ideally the symbolic policies produced by a distiller are more compact than the agent’s neural policies that they approximate.

Finally, we consider how symbolic policies produced by a distiller and used for teaching and learning within a population of autonomous agents evolving in A-life affect the population dynamics (e.g., agent life expectancy).

## 3 Related Work

Discovering a symbolic approximation of an unknown function is related to program synthesis which seeks to find a symbolic program for a given problem specification [Alur et al., 2013]. Search-based methods such as random sampling, simulated annealing, genetic algorithms and inductive logic programming have been used [Bulitko et al., 2022, Wang et al., 2023, 2024, Stevens et al., 2023]. Additionally synthesized programs can be human-explainable and allow for theoretical verifiability [Wang et al., 2024, Stevens et al., 2023]. Such synthesis methods can also be applied iteratively with synthesized programs added to the domain-specific language as the new building blocks [Hernandez and Bulitko, 2021]. However, when searching the space of programs program-synthesis methods commonly evaluate each candidate program by running it on a data set. This is computationally expensive. The stochastic nature of the search can also make it unreliable. The latter can be remedied by running a portfolio of such searches and taking the best of several runs [Botea and Bulitko, 2025] but the computation costs increase even further.

A line of work proposes to aid the combinatorial symbolic search by neural networks [Menon et al., 2013, Ellis et al., 2018, Valkov et al., 2018, Shah et al., 2020, Ellis et al., 2021]. The neural networks are trained on problem specification and program pairs and, in general, a search in the symbolic space is still required which is something we wish to eliminate in solving our problem.

Recent research has applied pre-trained transformers for symbolic regression tasks demonstrating that they can efficiently synthesize symbolic expressions, perform sequence-to-sequence program induction and generalize across a variety of rule-based problems [Biggio et al., 2021, Valipour et al., 2021, Kamienny et al., 2022, Shojaee et al., 2023, Lalande et al., 2023, Vastl et al., 2024, Wang et al., 2026]. Analysis reveals that a transformer’s internal representations can capture recursive rules and variable binding properties [Brinkmann et al., 2024, Wu et al., 2025]. Neurosymbolic paradigms have emerged that combine transformer-based symbolic regressors with evolutionary or preference-based algorithms to enhance robustness when dealing with limited or noisy data [Jha et al., 2024, Lalande et al., 2023]. These methods often outperform traditional search-based symbolic program synthesis in terms of efficiency and accuracy. To the best of our knowledge none of these studies has been applied to A-life-style multi-agent systems considered in this paper.

Large language models (LLMs) and transformers have been used in agent-based modeling to control the agents [Jimenez-Romero et al., 2025] and to capture the environment dynamics and agent trajectories in a neural format [Berkovich and Buehler, 2025, Berkovich et al., 2025, Yuan et al., 2021]. None of these tackles our problem of distilling an agent neural policy into a symbolic approximation.

Work on rule inference trained transformers to produce cellular automata rules given a target phenomenon [Berkovich et al., 2025]. While our problem is also to produce symbolic policies we seek to have the policies approximate evolved neural policies of individual A-life agents as opposed to global dynamics of the environment. Work on modeling agents from observed trajectories used transformers to capture influences of other agents’ policies which can then be used to condition the controlled agent’s policy and improve its reinforcement learning [Wallace et al., 2025]. However, our problem is to generate compact symbolic policies of individual neural agents in A-life.

Finally, our effort in this paper to enable agents to teach and learn from each other is related to the large body of work on explainable AI where AI agents seek to explain their predictions [Chakraborty et al., 2017], recommendations [Zhang and Chen, 2020], schedules [Vasileiou et al., 2023] or plans [Fox et al., 2017, Kambhampati, 2019]. When an AI agent is using machine learning (ML) it may be employing ML representations that range from ones that can be analyzed easily (e.g., logistic regression and decision trees) to those that are more complex and can highlight key regions in the input data [Springenberg et al., 2015] or features [Chen et al., 2014, Guidotti et al., 2018, Chakraborty et al., 2017] relevant to the solution. Within the context of symbolic planning and sequential decision making the problem is cast as explainable planning [Fox et al., 2017, Kambhampati, 2019]. In this subarea a popular approach is one that reconciles the models of the human and AI agents such that they both understand the plan being explained [Chakraborti et al., 2017]. However, existing work [Chakraborti et al., 2017, Sreedharan et al., 2018, 2021, Son et al., 2021, Vasileiou et al., 2021] makes simplifying assumptions such as that the AI agent has a complete and accurate mental model of the user and that the explanations are one-shot interactions. Researchers have started to address some of these limitations by generalizing the framework to learn the mental model of users over time [Tang et al., 2025] as well as to allow for conversational multi-shot interactions [Vasileiou et al., 2024]. Our problem can be seen as a combination of both interpretable ML and explainable planning where the neural policy of one agent is distilled into a symbolic representation communicated to another agent to update its own neural policy. Yet the work cited above has not been applied in A-life.

## 4 Proposed Approach

We adopt transformer-based symbolic regression [Biggio et al., 2021, Valipour et al., 2021, Kamienny et al., 2022, Shojaee et al., 2023, Lalande et al., 2023, Vastl et al., 2024] and apply it to the problem of distilling neural control policies of A-life agents. While some previous work used manually designed transformer architectures [Biggio et al., 2021, Valipour et al., 2021, Kamienny et al., 2022, Shojaee et al., 2023, Lalande et al., 2023, Vastl et al., 2024, Wang et al., 2026] we apply automated neural architecture search (NAS) in a space of transformer architectures. To do so efficiently we combine Chen et al. [2021]’s autoformers and Bulitko et al. [2022]’s triage.

## 4.1 Neural Architecture Search

In line with work by Chen et al. [2021] we first define a space of transformer architectures and initialize the supernet as the largest architecture in that space. Then on each training iteration we randomly select a subnet from the supernet and update its weights. We continue to minimize the training loss of sampled subnets until the validation loss no longer improves over a number of epochs. Once the supernet is trained we run a GA over the space of transformer architectures for a subnet to maximize a fitness function — the distillation accuracy regularized by the subnet size.

## 4.2 Training the Transformer

Our transformers receive a set of an agent’s percept-action pairs and produce a sequence of tokens representing a symbolic policy. For neural A-life agents percept-action pairs are readily available. However, a symbolic policy for each such neural agent is generally unavailable since the evolution within A-life is entirely neural and does not involve symbolic representations. Thus we do not use data from actual A-life agents to train our transformer. Instead we draw a random symbolic policy m from the given space M of policies and execute m on random agent percepts $p _ { i } ,$ generating a set of percept-action pair $\{ ( p _ { i } , m ( p _ { i } ) ) \}$ }. The set of pairs forms the transformer input whereas the symbolic policy m, expressed as a sequence of tokens in a domain-specific language, constitutes the target output for the transformer. We collate such data from many different policies m to train the transformer.

This method allows us to train the transformer on a large and diverse set of symbolic policies. This is conceptually similar to training large language models (LLM) on text and code from web pages and code repositories and then applying the trained LLM to queries from a specific user.

To keep our symbolic policies compact and increase their human-readability we define our own A-life-specific language to represent such policies. Doing so renders open-source transformers [Kamienny et al., 2022, Shojaee et al., 2023] not directly applicable.

## 5 Empirical Evaluation

We first present the A-life environment with evolving neural agents. We then define the space of symbolic policies. Finally, we describe the construction and training of the transformer. All parameters in our empirical evaluation were chosen via preliminary experiments to balance the scope of the experiments and the computational cost.

## 5.1 A-life

We evaluate our transformer-based distillation in an A-life environment — a variant of the common predator-prey environment [Ackley and Littman, 1991, Wilensky, 1997, Bulitko et al., 2017, 2018a,b, Soares et al., 2018].

![](images/bd2527af8f5b9f0a34fdf2fb07c772235164f840846cbf3e00971c52a66958de.jpg)  
Figure 1: The A-life environment.

Our A-life agents exist on a rectangular grid map (Figure 1). Each grid cell is either closed (black squares in the figure) or open. Open cells contain at most one agent (circles in the figure). Agents have five actions available to them: the four moves in the cardinal directions and the action of staying at the current location. Each agent has an energy level. Existing, thinking (i.e., executing the neural control policy) and taking actions all decrease an agent’s energy which is replenished by eating grass (shades of green in the figure). When the energy reaches 0 the agent dies and is removed from the map. Grass re-grows in each cell up to the maximum value of 1. The re-growth rate decays over time for a guaranteed eventual population extinction.

The A-life simulation proceeds in discrete time steps. On each time step the agents (sorted in a random order) perceive the environment around them and run their neural control policies. A policy is a neural network which maps eleven inputs that collectively form the agent’s percept to one of the five actions. The eleven inputs are: the grass levels in the four cardinal directions up to a sight radius, inversely scaled by their distance to the agent (inputs 1-4); grass level in the agent’s position (input 5); presence of other agents in the four cardinal directions up to a sight radius, inversely scaled by their distance to the agent (inputs 6-9); the agent’s energy (input 10) and the agent’s normalized age (input 11). All eleven inputs are in the range of [0, 1].

While our testbed allows for the neural network architecture to evolve over time, in this paper all agents share the same architecture where the eleven input neurons are connected to eight hidden layers of five neurons each and then connected to the five output neurons. The output neuron with the highest activation indicates the action to take. The network thus has 300 learnable weights (connection and bias weights). The weights are specified in the agent’s genome at birth but can be updated within the agent’s lifetime through self-reflection and/or teaching and learning.

Reproduction is asexual. A parental agent has an offspring when it is old enough, has enough energy and the current number of agents does not exceed a pre-specified maximum. The parent gives a part of its energy to the newborn. The offspring’s genes are created by copying the weights of the parent and then perturbing them with random values (mutation). Note that similarly to other A-life simulations and unlike the standard GA, our simulation is asynchronous and parents and their children can co-exist. Furthermore, there is no explicit fitness function to sort the population by (as in GA). Finally, the population size is not fixed and can grow up to the maximum size allowed by the specifications. An A-life run ends when all agents die.

Table 1: Environment specifications.
<table><tr><td>parameter</td><td>value</td></tr><tr><td>grass growth</td><td> $^ { 1 / 3 3 . 7 5 }$ </td></tr><tr><td>grass growth decay</td><td> $1 - { } ^ { 1 } / 6 6 6 . 6 7$ </td></tr><tr><td>move energy</td><td> $^ 1 / 1 0$ </td></tr><tr><td>live energy</td><td> $^ { 1 / 8 . 3 6 }$ </td></tr><tr><td>power coefficient</td><td>1</td></tr><tr><td>max grass bite</td><td> $\overline { { 4 . 7 8 \times 1 0 ^ { 4 } } }$   $^ { 1 / 1 . 3 3 }$ </td></tr><tr><td>min reproduction energy</td><td> $^ { 1 / 2 . 7 5 }$ </td></tr><tr><td>min reproduction age</td><td>8</td></tr><tr><td>initial agents</td><td>75</td></tr><tr><td>max agents</td><td>61</td></tr><tr><td>mutation rate</td><td>50</td></tr><tr><td>map size</td><td> $6 8 \times 6 8$ </td></tr><tr><td>sight radius</td><td>7</td></tr><tr><td>ANN topology</td><td>11-5-5-5-5-5-5-5-5-5 (300 weights)</td></tr></table>

The environment is specified by a set of parameters (Table 1) as well as the grid map (Figure 1). The parameters and the map were found automatically by stochastically searching the space of parameters and the space of maps via a separate GA. The fitness of a set of environment specifications for this GA was the sum of cubed lifetimes of all agents during an evolution run. Maximizing this fitness function yields environments with numerous yet longer living agents. Since each A-life run is stochastic for a single set of environment specifications we ran 32 independent A-life runs and took the median value of the number of agents born.

## 5.2 Symbolic Policies

We use an A-life-specific language to define the space M of symbolic policies. Each symbolic policy is a formula defined by the following context-free grammar:

$$
\begin{array} { r l } { S } & {  \xrightarrow { T } | \begin{array} { l } { O _ { 1 } } \end{array} | O _ { 2 } | O _ { 3 } | O _ { 4 } | O _ { 5 }  } \\ { T } & {  \begin{array} { l } { g _ { * } | \begin{array} { l } { | g _ { * } | } \end{array} | g _ { * } | \begin{array} { l } { | } { | g _ { * } | } \end{array} | a _ { * } | a _ { \uparrow } | a _ { * } | a _ { * } | e  | a _ { \downarrow } | e  } \end{array} } \\ { O _ { 1 } } & {  \begin{array} { l } { S + + | \begin{array} { l } { S - | } { | S | } \end{array} | \mathrm { r o u n d } ( S ) } \end{array} } \\ { O _ { 2 } } & {  \begin{array} { l } { S + S | \begin{array} { l } { S - S | } { \big | } \end{array} | \begin{array} { l } { S } \\ { \mathrm { m i n } ( S , S ) } \end{array} | \mathrm { a r g m i n } ( S , S ) | \begin{array} { l } { \mathrm { a r g m a x } ( S , S ) } \end{array} } \\ { \vdots } \\ { O _ { 3 } } \end{array} } \\ { \begin{array} { l } { O _ { 4 } | O _ { 5 }  } \end{array} } \\ { O _ { 5 } } \\ { O _ { 5 } } &   \begin{array} { l } { \mathrm { m i n } ( S , S , S , S ) | \begin{array} { l } { \mathrm { m a x } ( S , S , S , S ) | \begin{array} { l } { \mathrm { a r g m i n } ( S , S , S ) | \begin{array} { l } { | \mathrm { a r g m a x } ( S , S ) } \end{array} | } } \\ { \vdots } \\ { \mathrm { m i n } ( S , S , S , S , S ) | \begin{array} { l } { \mathrm { m a x } ( S , S , S , S , S ) | } \end{array} } \end{array} } \end{array} \end{array} \end{array}
$$

The tokens $g _ { \left. } , g _ { \uparrow } , g _ { \right. } , g _ { \downarrow } , g _ { + } , a _ { \left. } , a _ { \uparrow } , a _ { \right. } , a _ { \downarrow } , e ,$ a form the percept of an A-life agent. They represent the grass levels in the four cardinal directions around the agent $( g _ { \left. } , g _ { \uparrow } , g _ { \right. } , g _ { \downarrow } )$ , the grass at the agent’s position $( g _ { + } )$ , the presence of other agents $( a _ { \left. } , a _ { \uparrow } , a _ { \right. } , a _ { \downarrow } )$ , the agent’s energy e and the agent’s age a. Both the grass and the presence of other agents are inversely scaled by the distance.

The production T lists the terminal tokens: the agent’s percepts above and the numeric constants: $C = \{ 0 , 0 . 1 , \ldots , 1 \}$ The production $O _ { 1 }$ lists unary operators (e.g., the absolute value |S|). The production $O _ { 2 }$ lists binary operators and so on until $O _ { 5 }$ , which lists five-argument operators such as argmax(S, S, S, S, S).

In line with recent work on transformers trained for symbolic regression [Kamienny et al., 2022, Shojaee et al., 2023, Lalande et al., 2023, Valipour et al., 2021, Vastl et al., 2024, Biggio et al., 2021] we use the natural sequence-of-tokens notation for our symbolic policies. For instance, the sequence of six tokens argmax $g _ { \left. } \ g _ { \uparrow } \ g _ { \right. } \ g _ { \downarrow } \ g _ { + }$ represents the formula argmax $( g _ { \left. } , g _ { \uparrow } , g _ { \right. } , g _ { \downarrow } , g _ { + } )$ which returns the index of the grid cell around the agent with the highest grass level. We index the agent’s actions left, up, right, down, stay by $1 , \ldots , 5$ so the formula above is the symbolic always-move-to-the-highest-grass-neighbor policy.

If on some inputs a formula produces an output outside of the set $\{ 1 , \ldots , 5 \}$ of action indices the symbolic policy outputs the action index closest to the output.

## 5.3 Training Data for the Transformer

To form the training set we generated random sequences of tokens from the grammar. For each sequence, tokens not used in computing the output were removed. If the resulting sequence was empty or exceeded 30 tokens, it was discarded.

We then formed a set of 1,024 percepts with each of the eleven inputs uniformly randomly drawn from [0, 1]. If on any of the percepts the formula did not return a numeric value or returned a value outside of the action index range [1, 5] the formula was discarded. We additionally discarded formulae whose output was the same for over 99% of the percepts.

Removing duplicate formulae, this process yielded a set of 450,100 unique formulae to train the transformer on. For each formula the transformer input was a 64 × 12 matrix of the percept-action pairs. In each row of the matrix the first eleven columns describe a percept and the twelfth column lists the action produced by the formula on the percept. The transformer output was the sequence of tokens representing the formula. We used 5 random percept-action sets of size $6 4 \times 1 2$ each for training.

## 5.4 Defining the Transformer Architecture

We used the neural architecture search to find an appropriate architecture for the transformer. The space of possible architectures is defined in Table 2, inspired by existing work [Chen et al., 2021]. The encoder and decoder had the same components with the same space of dimensions.

Percept-action pairs in our training data are permutation invariant. Thus we removed the positional embeddings from the encoder [Kamienny et al., 2022]. To blend the column features we instead added a standard feedforward layer before the encoder [Lalande et al., 2023].

In addition to the previously listed tokens we added three more tokens for transformer training: a beginning-of-sequence token $\left. b o s \right.$ , an end-of-sequence token $\langle e o s \rangle$ and the padding token $\langle p a d \rangle$ . During training we added $\left. b o s \right.$ at the beginning of each formula that served as input for the decoder along with the percept-action pairs for the encoder and we added a ⟨eos⟩ token at the end of each of the target formulae so that the transformer can predict the next token based on the percept-action pairs and previous tokens as context.

Table 2: Transformer architecture space.
<table><tr><td rowspan=1 colspan=2>component</td><td rowspan=1 colspan=1>minimum</td><td rowspan=1 colspan=1>maximum</td></tr><tr><td rowspan=6 colspan=2>number of layersnumber of headsembedding dimensionQ-K dimensionV dimensionMLP dimension</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>240</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>256</td></tr><tr><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>256</td></tr><tr><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>512</td></tr><tr><td rowspan=1 colspan=2>total number of learnables</td><td rowspan=1 colspan=1>73,581</td><td rowspan=1 colspan=1>17,296,205</td></tr></table>

We used the Adam optimizer [Kingma and Ba, 2015] with a learning rate of $1 0 ^ { - 4 }$ , first and second moment decay constants of 0.9 and 0.999 respectively and ϵ of 10<sup>−8</sup>. We stopped training when the validation loss did not decrease for over five epochs. We used a mini batch size of 1,024. We randomly partitioned the data into three subsets: 80% for training, 10% for validation during training (to compute the validation loss for early termination) and 10% for testing.

We brought all formulae to the length of 30 by adding padding tokens $\langle p a d \rangle$ at the end of shorter formulae. This allowed us to package the formulae into a tensor. When computing gradients during training the padding tokens do not affect gradient computations as they are ignored.

The transformer architecture search started with training the largest transformer (supernet) in our architecture space with 17.3 million learnable parameters (Table 2). During the training we did not train the supernet as a whole but rather trained a randomly selected subnet and updated its weights on each iteration [Chen et al., 2021]. We continued the training for 90 epochs which took 14.5 hours on a single NVIDIA H100 GPU with 80GB VRAM.

Once the supernet was trained we ran a GA with the population size of 20 to find a subnet with a high distillation fitness regularized by the number of neurons. The fitness computation was triaged [Bulitko et al., 2022]: a candidate subnet was first evaluated on 2,048 percept-action pairs. If the candidate was fitter than the previous champion it was subsequently evaluated on 10% of the percept-action pairs. Otherwise it was discarded. When the champion fitness did not improve over 20 generations the GA terminated.

The GA ran for about 1.5 minutes and found a subnet (Table 3) of 5.4 million learnable parameters (about 32% of the supernet). The subnet then inherited the weights from the supernet and was trained on the training data set for another 69 epochs with validation patience of 5 which took 13.5 hours. We found that the accuracy of the evolved subnet is higher when training starts with inherited weights rather than from scratch.

Table 3: The NAS-found transformer architecture.
<table><tr><td>component</td><td>maximum size</td></tr><tr><td>encoder depth encoder heads encoder embedding dimension encoder Q-K dimension encoder V dimension encoder MLP dimension</td><td>5 (4, 1, 4, 2, 1) 150 (139, 207, 85, 177, 138) (223, 99, 97, 256, 183) (330, 125, 459, 277, 147)</td></tr><tr><td>decoder depth decoder heads decoder embedding dimension decoder Q-K dimension decoder V dimension decoder MLP dimension</td><td>5 (4, 4, 1, 1, 2) 205 (209, 124, 169, 169, 193) (200, 156, 148, 146, 171) (333, 265, 348, 240, 497)</td></tr></table>

## 5.5 Distillation of Neural Agents

We evaluated our trained transformer on distilling A-life’s neural policies into symbolic policies and compared the performance to GA [Bulitko et al., 2022, Wang et al., 2023] and decision trees.

To form the data set we conducted 1,024 A-life runs. A total of 5,613,559 agents lived during those runs. Within each A-life run early agents have random neural policies. Thus to give evolution time to work we discarded all agents who died before time step 105. To focus on more successful agents we additionally discarded all agents who lived fewer than 105 time steps or made fewer than 75% energy-gain-maximizing action choices or whose neural policy’s most common action on random percepts accounted for at least 90% of actions taken. The filtering process left us with 243 agents.

For each of the 243 agents we ran its neural policy on 1,024 random percepts and recorded the action given by the neural policy. We then drew from the 1,024 percept-action pairs 512 for distillation and another 512 for testing the resulting symbolic policies. We then compared four distillation methods on this data set as follows.

Decision Trees. We fitted a classification and regression tree (CART) [Breiman et al., 1984] to the percept-action pairs using MATLAB’s built-in fitctree function. The tree partitions the percept space by recursively selecting the split that minimizes the Gini impurity and classifies each leaf region with the plurality action. We evaluated three tree sizes by limiting the maximum number of splits to 5, 10 and 20.

GA. We ran genetic algorithms with a population of 384 candidate formulae. On each generation the formulae were evaluated for their distillation accuracy on the percept-action data set. The fitness of each formula was its accuracy regularized by the number of tokens in the formula. The fittest 128 formulae were selected as parents and produced 128 offspring formulae via crossover and mutation. Each offspring was added to the population twice: as is and with tokens not used in calculation of the formula value removed. A running fitness champion was maintained. If the fitness did not improve over 100 generations the population was reinitialized. The process terminated when the time limit was reached and the fitness champion was returned as the result of the distillation. We used six different time limits: 5, 10, 30 seconds and 1, 3, 45 minutes.

Portfolio GA. Given the stochastic nature of the GA we also ran a portfolio GA with a portfolio of size four. Each member of the portfolio was the previously described GA but run with a different random number generator seed. When all four GA runs finished we selected the formula with the highest distillation fitness.

Transformer. When using the transformer for distillation besides the percept-action pairs we gave it a starting token ⟨bos⟩ as the context. When given a context sequence of any length the transformer returned a sequence of the same length with the last token being the newly predicted token. Each new token was appended to the previous context sequence for the transformer to predict the next token. The process proceeded until either the transformer predicted an ⟨eos⟩ token or the length of the sequence reached its maximum length. During distillation we randomly sampled 64 from the given 512 percept-action pairs as the input for the transformer to predict a formula, similar to the bagging trick described in existing work [Kamienny et al., 2022, Shojaee et al., 2023, Wang et al., 2026]. We applied beam search with width 1 and 5 [Vaswani et al., 2017] to investigate the effect of beam width on accuracy.

![](images/19b8d7d12e96220172e35f887e0747d86dda5632ecd6ceae7da49f10b602c350.jpg)  
Figure 2: Distillation of percept-action pairs of A-life agents.

Observations. The transformer with beam search with beam widths of 1 and 5 had the average distillation times of 0.63 and 4.08 seconds and their average distillation accuracy on the test set surpassed any GA including that with the 45-minute time limit (Figure 2).

Human readability of symbolic policies. Both the GA and the transformer tend to produce symbolic policies of similar length. We will now discuss their human readability on a specific example from a preliminary experiment. Consider an evolved A-life agent that lived for 434 time steps and died at time step 890. Its actions on random percepts were diverse with the most common action constituting 27% of all actions.

When run on a set of random percepts and the corresponding actions of the agent’s neural policy, the transformer output the symbolic policy argmax $( g _ { \left. } , g _ { \uparrow } , g _ { \right. } , a )$ which is readily understood as go left/up/right when that’s the direction with the most grass and the agent’s normalized age is below that grass amount. Go down when the agent’s normalized age becomes higher than the three grass amounts. This symbolic policy has the accuracy of about 99% on the set of 512 random percepts. In fact only three of the agent’s 512 actions are mislabeled by the symbolic policy: two right actions are mislabeled as left and one right is mislabeled as up.

While a decision tree can be human-readable [Rudin, 2019] even a shallow tree with 5 splits produces a branching structure that must be traced path by path to understand the policy. In practice this makes decision trees more difficult for a human to parse than compact formula-based symbolic policies produced by the GA or the transformer.

## 5.6 Policy Distillation within A-life

When distillation of agent neural policies takes place during an agent’s lifetime within A-life it can affect both individual agents and the population as a whole. In this section we consider two ways of using distillation within A-life.

Self-reflection. Every 25 time steps of its life each A-life agent ran the distillation process if its energy was at least 75% of the maximum. To do so, the agent ran its neural policy on 256 random percepts and recorded the actions returned by the policy. It then invoked the transformer to produce a symbolic policy approximating the percept-action set and, therefore, its neural policy. The resulting symbolic policy was then used to generate 32,768 percept-action pairs. These were in turn used to update the weights of its own neural policy in an act of self-reflection. While perfectly accurate distillation would likely result in minimal changes to the neural weights, a symbolic bias in the distillation can affect the neural weights severely.

Teaching and learning. An agent then engaged in teaching by distilling its neural policy and sending the resulting formula to a random agent in the population. The receiving agent updated its own neural policy in the manner described above. This mechanism allows the agents to share their control knowledge in a compact fashion as transmitting a short formula representing the symbolic policy is faster and easier than transmitting percept-action pairs or neural weights.

Symbolic subspaces. Self-reflection, teaching and learning pass an agent’s neural knowledge through the symbolic space. In our A-life environment this is the only mechanism by which the agents learn within their lifetime. Therefore the learning bias imposed by the symbolic space can in principle affect agents individually and their population as a whole.

In a preliminary experiment with a smaller A-life environment we first considered a trivial space of symbolic policies containing only a single formula argmax $( g _ { \left. } , g _ { \uparrow } , g _ { \right. } , g _ { \downarrow } , g _ { + } )$ which always takes the action in the direction of the most grass. Having only a single symbolic policy imposes a strong learning bias on self-reflection and teaching and learning. The results from 32 A-life runs suggest that such biased self-reflection and teaching and learning make the agents live longer and die later (Figure 3).

![](images/6a0a07142b3f9be278449d0cc8dd5f1fda65011f06f89823664d3ad2ff35d007.jpg)

![](images/8b810151e94cffe01b9a9af0aad3fec2a08210bf616ed0e573f65f153a273aba.jpg)

![](images/efb59bb54034dd1829524e667f875adb792e9633c1456c1e43d5f0c96367d637.jpg)  
Figure 3: Influence of self-reflection (SR) and teaching and learning (TL) on A-life population dynamics with a strong symbolic bias.

![](images/6b90409a0af73c92614e67947f9147d2ed797427b62fe41644843bc64ec7c9a4.jpg)  
Figure 5: Influence of self-reflection (SR) and teaching and learning (TL) on A-life population dynamics with the transformer and the full symbolic space.

![](images/26854098eecf6e40465d6d157113a82b64aba359c7e865ca260eec19b4687639.jpg)  
Figure 4: Different subspaces of symbolic policies. Both self-reflection and teaching and learning are performed.

We then used the larger A-life environment described earlier in the paper. We incrementally weakened the learning bias by allowing the transformer to use progressively larger subspaces of the full space of symbolic policies. The subspaces were: argmax<sup>5</sup> which is the space of all formulae starting with a 5-argument argmax; argmax<sup>5</sup> ∨ argmin<sup>5</sup> which is the space of all formulae starting with a 5-argument argmax or argmin; and the full space of all formulae in our grammar. These subspaces are strict subsets of each other: argmax<sup>5</sup> ⊂ argmax<sup>5</sup> ∨ argmin<sup>5</sup> ⊂ full space. All formulae were restricted to having at most 30 tokens.

With the smaller and simpler A-life environment and the high symbolic bias, combining teaching and learning with self-reflection affects the medians for the number of agents lived and the time of death (Figure 3). However, removing the symbolic bias and using a larger, more complex A-life environment eliminates the effects (Figure 5).

## 6 Shortcomings & Future Work

Currently the effects of self-reflection and teaching and learning are specific to the A-life environment with only the smaller, simpler A-life environment and stronger symbolic bias showing significant effects. Future work will investigate why the same trends are not observed with the larger, more complex A-life environment and larger symbolic policy spaces.

Despite the orders-of-magnitude reduction in run time relative to the GA, running the transformer with 5.4 million weights is still notably more expensive than running the agent’s own 300-weight neural policy. Had the distillation not been free the A-life agents would run out of energy and die after a single distillation. Thus, it will be of interest to evolve distillers together with the agent control policy within A-life evolution and see if smaller, more practical distillers emerge similar to previous work in reinforcement learning [Landajuela et al., 2021]. Furthermore, our agents currently self-reflect and teach involuntarily. Future work will explore environments which are conducive to emergence of self-reflection and teaching and learning as a voluntary action taken by agents.

Some of our preliminary results showed that the space of symbolic policies can indeed affect population dynamics. Thus, it will be of interest to evolve symbolic policy spaces in A-life alongside the agents and their distillers. An interesting question is whether, for any agent, there necessarily exists a task that the agent can perform but cannot symbolically articulate (distill) due to the computational overhead of such distillation. In particular if an articulation of a policy is a computational shortcut to the original policy then computational irreducibility may restrict the type of policies that can be articulated [Wolfram, 2002, 2024, Berkovich and Buehler, 2025]. Consequently, learning agents in A-life may have to use their own life experience to complement the symbolic policy they receive from a teaching agent.

## 7 Conclusions

Neural control policies of A-life agents are generally opaque to an external observer. Approximating them with compact symbolic policies can help human understanding. We proposed to use a transformer to distill neural policies of autonomous agents in A-life. Once trained, the transformer surpasses the distillation accuracy of genetic algorithms while being substantially faster. The fast distillation afforded by the transformer makes it more practical to deploy it on the agents themselves which can enable practical teaching and learning within a multi-agent system.

## Acknowledgments

We thank the Natural Sciences and Engineering Research Council of Canada (NSERC) for the funding and the Digital Research Alliance of Canada (DRAC) for the computational resources.

## References

Kamal Acharya, Waleed Raza, Carlos Dourado, Alvaro Velasquez, and Houbing Herbert Song. Neurosymbolic reinforcement learning and planning: A survey. IEEE Transactions on Artificial Intelligence, 5(5):1939–1953, 2023.

D. Ackley and M. Littman. Interactions between learning and evolution. Artificial Life II, 1991.

Rajeev Alur, Rastislav Bodik, Garvit Juniwal, Milo MK Martin, Mukund Raghothaman, Sanjit A Seshia, Rishabh Singh, Armando Solar-Lezama, Emina Torlak, and Abhishek Udupa. Syntax-guided synthesis. In Proceedings ofthe Conference on Formal Methods in Computer-Aided Design (FMCAD), 2013.

Osbert Bastani, Yewen Pu, and Armando Solar-Lezama. Verifiable reinforcement learning via policy extraction. Advances in Neural Information Processing Systems, 2018.

Jaime A Berkovich and Markus J Buehler. LifeGPT: Topology-agnostic generative pretrained transformer model for cellular automata. npj Artificial Intelligence, page 23, 2025.

Jaime A Berkovich, Noah S David, and Markus J Buehler. AutomataGPT: Forecasting and ruleset inference for two-dimensional cellular automata. arXiv preprint arXiv:2506.17333, 2025.

Luca Biggio, Tommaso Bendinelli, Alexander Neitz, Aurelien Lucchi, and Giambattista Parascandolo. Neural symbolic regression that scales. In Proceedings ofthe International Conference on Machine Learning (ICML), 2021.

Adi Botea and Vadim Bulitko. Generating high-score crosswords puzzles with bi-objective, stochastic and systematic search. ICGA Journal, 2025.

L Breiman, JH Friedman, R Olshen, and CJ Stone. Classification and Regression Trees. Wadsworth, 1984.

Jannik Brinkmann, Abhay Sheshadri, Victor Levoso, Paul Swoboda, and Christian Bartelt. A mechanistic analysis of a transformer trained on a symbolic multi-step reasoning task. In Findings of the Association for Computational Linguistics: ACL 2024, pages 4082–4102, 2024.

Vadim Bulitko, Shelby Carleton, Delia Cormier, Devon Sigurdson, and John Simpson. Towards positively surprising non-player characters in video games. In Proceedings ofthe Experimental AI in Games (EXAG) Workshop at the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment (AIIDE), 2017.

Vadim Bulitko, Kacy Doucet, John Simpson, and Aaryn Flynn. A-life for non-playable characters in video games. In Event Proceedings for Late-Breaking Abstracts at the A-Life Conference, 2018a.

Vadim Bulitko, Mac Walters, Morgan Cselinacz, and Matthew Brown. Evolving NPC behaviours in A-Life with player proxies. In Proceedings of the Experimental AI in Games (EXAG) Workshop at the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment (AIIDE), 2018b.

Vadim Bulitko, Shuwei Wang, Justin Stevens, and Levi HS Lelis. Portability and explainability of synthesized formula-based heuristics. In Proceedings of the International Symposium on Combinatorial Search (SoCS), 2022.

Tathagata Chakraborti, Sarath Sreedharan, Yu Zhang, and Subbarao Kambhampati. Plan explanations as model reconciliation: Moving beyond explanation as soliloquy. In Proceedings ofthe International Joint Conference on Artificial Intelligence (IJCAI), pages 156–163, 2017.

Supriyo Chakraborty, Richard Tomsett, Ramya Raghavendra, Daniel Harborne, Moustafa Alzantot, Federico Cerutti, Mani B. Srivastava, Alun D. Preece, Simon Julier, Raghuveer M. Rao, Troy D. Kelley, Dave Braines, Murat Sensoy, Christopher J. Willis, and Prudhvi Gurram. Interpretability of deep learning models: A survey of results. In Proceedings ofthe IEEE SmartWorld/SCALCOM/UIC/ATC/CBDCom/IOP/SCI Conference, pages 1–6, 2017.

S. Chaudhuri, K. Ellis, O. Polozov, R. Singh, A. Solar-Lezama, Y. Yue, et al. Neurosymbolic programming. FTPL, 2021.

Minghao Chen, Houwen Peng, Jianlong Fu, and Haibin Ling. Autoformer: Searching transformers for visual recognition. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

Wenlin Chen, Yixin Chen, and Kilian Q. Weinberger. Fast flux discriminant for large-scale sparse nonlinear classification. In Proceedings ofthe ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), pages 621–630, 2014.

Miles Cranmer, Alvaro Sanchez Gonzalez, Peter Battaglia, Rui Xu, Kyle Cranmer, David Spergel, and Shirley Ho. Discovering symbolic models from deep learning with inductive biases. Advances in Neural Information Processing Systems, 33:17429–17442, 2020.

Senne Deproost, Denis Steckelmacher, and Ann Nowe. Human-readable programs as the actor of a reinforcement learning agent using critic-moderated evolution. In Proceedings ofthe Joint International Scientific Conferences on AI and Machine Learning (BNAIC/BeNeLearn), pages 1–18, 2024.

Kevin Ellis, Lucas Morales, Mathias Sablé-Meyer, Armando Solar-Lezama, and Josh Tenenbaum. Learning libraries of subroutines for neurally-guided Bayesian program induction. Advances in Neural Information Processing Systems, 31, 2018.

Kevin Ellis, Catherine Wong, Maxwell Nye, Mathias Sablé-Meyer, Lucas Morales, Luke Hewitt, Luc Cary, Armando Solar-Lezama, and Joshua B Tenenbaum. DreamCoder: Bootstrapping inductive program synthesis with wakesleep library learning. In Proceedings ofthe ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI), pages 835–850, 2021.

Maria Fox, Derek Long, and Daniele Magazzeni. Explainable planning. arXiv preprint arXiv:1709.10256, 2017.

A. Garcez and L. Lamb. Neurosymbolic AI: The 3rd wave. AI Review, (11), 2023.

Riccardo Guidotti, Anna Monreale, Salvatore Ruggieri, Franco Turini, Fosca Giannotti, and Dino Pedreschi. A survey of methods for explaining black box models. ACM Computing Surveys, pages 1–42, 2018.

S. Gulwani, O. Polozov, R. Singh, et al. Program synthesis. FTPL, (1-2), 2017.

Sergio Poo Hernandez and Vadim Bulitko. Speeding up heuristic function synthesis via extending the formula grammar. In Proceedings ofthe International Symposium on Combinatorial Search (SoCS), pages 233–235, 2021.

P. Hitzler, A. Eberhart, M. Ebrahimi, M. Sarker, and L. Zhou. Neuro-symbolic approaches in artificial intelligence. National Science Review, 2022.

Samyak Jha, Sergei Gleyzer, E Reinhardt, Victor Baules, Nobuchika Okada, and Francois Charton. Evolutionary and transformer-based methods for symbolic regression. In Proceedings of the NeurIPS Workshop on Machine Learning and the Physical Sciences, 2024.

Cristian Jimenez-Romero, Alper Yegenoglu, and Christian Blum. Multi-agent systems powered by large language models: Applications in swarm intelligence. Frontiers in Artificial Intelligence, page 1593017, 2025.

Subbarao Kambhampati. Synthesizing explainable behavior for human-AI collaboration. In Proceedings of the International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS), pages 1–2, 2019.

Pierre-Alexandre Kamienny, Stéphane d’Ascoli, Guillaume Lample, and François Charton. End-to-end symbolic regression with transformers. Advances in Neural Information Processing Systems, pages 10269–10281, 2022.

H. Kautz. The third AI summer: AAAI Robert S. Engelmore memorial lecture. Artificial Intelligence, 2022.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the International Conference on Learning Representations (ICLR), 2015.

J. Koza. Genetic Programming: On the Programming of Computers by Means of Natural Selection. MIT Press, 1992.

Florian Lalande, Yoshitomo Matsubara, Naoya Chiba, Tatsunori Taniai, Ryo Igarashi, and Yoshitaka Ushiku. A transformer model for symbolic regression towards scientific discovery. In Proceedings ofthe NeurIPS Workshop on AIfor Science, 2023.

Mikel Landajuela, Brenden K Petersen, Sookyung Kim, Claudio P Santiago, Ruben Glatt, Nathan Mundhenk, Jacob F Pettit, and Daniel Faissol. Discovering symbolic policies with deep reinforcement learning. In Proceedings ofthe International Conference on Machine Learning (ICML), pages 5979–5989, 2021.

L. Medeiros, D. Aleixo, and L. Lelis. What can we learn even from the weakest? learning sketches for programmatic strategies. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2022.

Aditya Menon, Omer Tamuz, Sumit Gulwani, Butler Lampson, and Adam Kalai. A machine learning framework for programming by example. In Proceedings of the International Conference on Machine Learning (ICML), pages 187–195, 2013.

V. Mnih, K. Kavukcuoglu, D. Silver, A. Rusu, J. Veness, M. Bellemare, A. Graves, M. Riedmiller, A. Fidjeland, G. Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 2015.

Augustus Odena, Kensen Shi, David Bieber, Rishabh Singh, Charles Sutton, and Hanjun Dai. BUSTLE: Bottom-up program synthesis through learning-guided exploration. In Proceedings of the International Conference on Learning Representations (ICLR), 2020

Cynthia Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence, 1(5):206–215, 2019.

Ameesh Shah, Eric Zhan, Jennifer Sun, Abhinav Verma, Yisong Yue, and Swarat Chaudhuri. Learning differentiable programs with admissible neural heuristics. Advances in Neural Information Processing Systems, pages 4940–4952, 2020.

Parshin Shojaee, Kazem Meidani, Amir Barati Farimani, and Chandan Reddy. Transformer-based planning for symbolic regression. Advances in Neural Information Processing Systems, pages 45907–45919, 2023.

S. E. Soares, V. Bulitko, K. Doucet, M. Cselinacz, T. Soule, S. Heck, and L. Wright. Learning to recognize A-Life behaviours. In ACS, 2018.

Tran Cao Son, Van Nguyen, Stylianos Loukas Vasileiou, and William Yeoh. Model reconciliation in logic programs. In Proceedings ofthe European Conference on Logics in Artificial Intelligence (JELIA), pages 393–406, 2021.

Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin A. Riedmiller. Striving for simplicity: The all convolutional net. In Proceedings of the International Conference on Learning Representations (ICLR) Workshop Track, 2015.

Sarath Sreedharan, Tathagata Chakraborti, and Subbarao Kambhampati. Handling model uncertainty and multiplicity in explanations via model reconciliation. In Proceedings ofthe International Conference on Automated Planning and Scheduling (ICAPS), pages 518–526, 2018.

Sarath Sreedharan, Siddharth Srivastava, and Subbarao Kambhampati. Using state abstractions to compute personalized contrastive explanations for AI agent behavior. Artificial Intelligence, page 103570, 2021.

Kenneth O. Stanley and Risto Miikkulainen. Evolving neural networks through augmenting topologies. Evolutionary Computation, pages 99–127, 2002.

Justin Stevens, Vadim Bulitko, and David Thue. Solving witness-type triangle puzzles faster with an automatically learned human-explainable predicate. arXiv preprint arXiv:2308.02666, 2023.

Yinxu Tang, Stylianos Loukas Vasileiou, and William Yeoh. Does your AI agent get you? A personalizable framework for approximating human models from argumentation-based dialogue traces. In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI), pages 14405–14413, 2025.

Silviu-Marian Udrescu and Max Tegmark. AI Feynman: A physics-inspired method for symbolic regression. Science Advances, 2020.

Mojtaba Valipour, Bowen You, Maysum Panju, and Ali Ghodsi. SymbolicGPT: A generative transformer model for symbolic regression. CoRR, 2021.

Lazar Valkov, Dipak Chaudhari, Akash Srivastava, Charles Sutton, and Swarat Chaudhuri. Houdini: Lifelong learning as program synthesis. Advances in Neural Information Processing Systems, 2018.

Stylianos Loukas Vasileiou, Alessandro Previti, and William Yeoh. On exploiting hitting sets for model reconciliation. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), pages 6514–6521, 2021.

Stylianos Loukas Vasileiou, Borong Xu, and William Yeoh. A logic-based framework for explainable agent scheduling problems. In Proceedings ofthe European Conference on Artificial Intelligence (ECAI), 2023.

Stylianos Loukas Vasileiou, Ashwin Kumar, William Yeoh, Tran Cao Son, and Francesca Toni. Dialectical reconciliation via structured argumentative dialogues. In Proceedings of the International Conference on Principles of Knowledge Representation and Reasoning (KR), 2024.

Martin Vastl, Jonáš Kulhánek, Jiˇrí Kubalík, Erik Derner, and Robert Babuška. Symformer: End-to-end symbolic regression using transformer-based architecture. IEEE Access, 2024.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.

A. Verma, V. Murali, R. Singh, P. Kohli, and S. Chaudhuri. Programmatically interpretable reinforcement learning. In Proceedings ofthe International Conference on Machine Learning (ICML), 2018.

A. Verma, H. Le, Y. Yue, and S. Chaudhuri. Imitation-projected programmatic reinforcement learning. In Proceedings ofthe Conference on Neural Information Processing Systems (NeurIPS), 2019.

Conor Wallace, Umer Siddique, and Yongcan Cao. TransAM: Transformer-based agent modeling for multi-agent systems via local trajectory encoding. volume 6, pages 2327–2341, 2025.

Shuwei Wang, Vadim Bulitko, Taoan Huang, Sven Koenig, and Roni Stern. Synthesizing priority planning formulae for multi-agent pathfinding. In Proceedings ofthe AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment (AIIDE), pages 360–369, 2023.

Shuwei Wang, Vadim Bulitko, and William Yeoh. Explaining synthesized pathfinding heuristics via iterative visualization and modification. In Proceedings ofthe IEEE Conference on Games (CoG), pages 1–4, 2024.

Shuwei Wang, Vadim Bulitko, Michael Youngblood, Ramon Lawrence, William Yeoh, Shinichi Nakagawa, Matthew RG Brown, and Yu Wang. Synthesizing physics formulae with transformers. arXiv preprint arXiv:2610.03947, 2026.

U Wilensky. NetLogo wolf sheep predation model. http://ccl.northwestern.edu/netlogo/models/WolfSheepPredation, 1997. URL http://ccl.northwestern.edu/netlogo/models/WolfSheepPredation.

Stephen Wolfram. A New Kind ofScience. Wolfram Media, 2002.

Stephen Wolfram. Can AI solve science?, 2024. URL http://writings.stephenwolfram.com/2024/03 can-ai-solve-science.

Yiwei Wu, Atticus Geiger, and Raphaël Millière. How do transformers learn variable binding in symbolic programs? In Proceedings ofthe International Conference on Machine Learning (ICML), 2025.

Ye Yuan, Xinshuo Weng, Yanglan Ou, and Kris M Kitani. AgentFormer: Agent-aware transformers for socio-temporal multi-agent forecasting. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9813–9823, 2021.

Yongfeng Zhang and Xu Chen. Explainable recommendation: A survey and new perspectives. Foundations and trends® in information retrieval, pages 1–101, 2020.

He Zhu, Zikang Xiong, Stephen Magill, and Suresh Jagannathan. An inductive synthesis framework for verifiable reinforcement learning. In Proceedings of the ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI), pages 686–701, 2019.