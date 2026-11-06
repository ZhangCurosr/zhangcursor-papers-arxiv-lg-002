# The Blindfolded Controller: Decision Model-Guided Training for Reinforcement Learning

Muhammad Ibrahim Khan Coventry University, Coventry, UK

Aniket Dixit Coventry University, Coventry, UK

Faizan Ahmed Coventry University, Coventry, UK

James Brusey Coventry University, Coventry, UK

khanm442@coventry.ac.uk

dixita4@uni.coventry.ac.uk

ahmedf84@uni.coventry.ac.uk

aa3172@coventry.ac.uk

## Abstract

Reinforcement learning (RL) provides a framework for continuous improvement through interaction with an environment. However, RL agents struggle against tasks that ofer exploration challenges and sparse rewards, because learning eficient policies from scratch is slow and sample-ineficient. In contrast, decision models (DMs), given a description of the environment and task, can perform well in sparse-reward tasks, but, as they do not learn from reward, they are not able to improve towards the optimal solution. In this paper, we present Decision Model-Guided Training (DMGT), a technique that uses a DM with a description of the environment and task to initialise an RL policy, combining the DM’s ability to act without training with RL’s ability to improve from reward. Instead of exploring from scratch, the agent reaches the goal and receives reward from its first episodes, and RL then improves its policy beyond what the DM achieves. We demonstrate through experiments on seven tasks that our approach outperforms RL in all seven and DM in five. In addition, we provide comprehensive guidelines, derived from ablation studies, for describing an environment and task to a DM.

## 1 Introduction

Reinforcement learning learns through interactions with the environment through trial and error. But for tasks with dificult exploration and sparse rewards, an agent starting from scratch can take a lot of timesteps without any reward, so the learning process is slow and data ineficient. For example, in Lunar Lander (Towers et al., 2025), during early stages of training PPO crashes because the agent must learn to fire a lander’s engine at the beginning of the episode to slow down the falling lander. PPO with default hyperparameters was not able to solve the environment even after a million steps.

A common way to address RL’s speed concern is to give the agent a head start. The agent can first imitate demonstrations recorded from human experts (Silver et al., 2016; Hester et al., 2018; Rajeswaran et al., 2018), learn from a teacher policy that already performs the task (Schmitt et al., 2018; Uchendu et al., 2023), or build on a hand-designed controller (Johannink et al., 2019; Silver et al., 2018). These approaches show that a good starting policy can substantially reduce the interaction that RL needs. In each case, however, the head start is behaviour that must first be recorded, trained or engineered for the task before RL begins.

A decision model (DM) ofers a diferent perspective, it is a pretrained model that, given a description of the environment, the goal and the possible actions, picks an action without any training on the environment. Hence it is zero-shot: it needs no demonstrations, no reward and no training on the environment, only a description of it. DM can perform well in sparse-reward and dificult tasks where RL struggles, because a DM’s behaviour comes from the description of the environment in natural language and not from trial and error. However a DM does not learn from rewards, therefore it cannot improve from experience, its behaviour is fixed by the model and the description. Given a description of the lander, the pad’s location, available actions and the goal, a DM lands the lander on its first attempt, without any training. However, it was only successful for 18 of 20 episodes, and does not have the ability to learn from the unsuccessfu episodes.

![](images/2e8658b635dabcc0a1b7e9647e07042b59df95a4c277234dfcf0f0131a6a4f6e.jpg)  
Figure 1: Decision Model-Guided Training (DMGT): combining a decision model (DM) with reinforcement learning (RL). A DM reads a text description of the current situation and returns a probability for each of a fixed set of answers. It needs no training, but the reward is never part of what it reads: it is blindfolded to the reward. Top: three agents in the same world, with the reward at the goal. RL alone explores from scratch and rarely finds the reward. The DM alone reaches the goal from its description but cannot learn from the reward. DMGT (ours) starts from the DM’s behaviour (the faded path) and then learns from the reward. Bottom: how DMGT works. (1) Each state is described in words, following our guidelines on what a DM needs to be told: the environment, the goal, how to read the values, the state, recent history, a priority (what is most urgent now), a strategy hint and the answer options. (2) The DM’s most likely answer is the action. (3) A policy network imitates these decisions. (4) RL (here PPO) improves that network from the reward. (5) Schematic learning curves: DMGT starts at the DM’s level, RL alone from scratch.

Therefore RL and DMs have complementary strengths: RL improves through interactions with the environment but starts from scratch, whereas a DM starts with useful behaviour but it cannot improve. Initialised from a single landing by the DM, PPO succeeded in all 20 test episodes, which PPO from scratch did not, and ends up landing more reliably than the DM itself.

A DM acts based on the description of the task in natural language and the description determines if it succeeds. For example, in Lunar Lander, a DM given only a concise description of the environment lands just as often as with the full description, even though it is told less than a third of the information. What matters is therefore not how much the DM is told, but what information it receives. We study this through an extensive ablation study on seven tasks in six environments, where we remove parts of the description one at a time and then search for the concise and smallest description that still succeeds, and derive guidelines for describing an environment and task to a DM.

We evaluated DMGT on six environments, comparing it with RL alone an with DM alone. In Lunar Lander, PPO initialised from a single landing by the DM reached an average return of 200, after 216k steps, a level that PPO alone was not able to achieve in 1M steps, and its final policy lands in 20 of 20 test episodes, more often than both the DM and PPO. Across all seven tasks, DMGT outperforms PPO alone and converges faster, and it outperforms the DM alone in five of them. Across the six environments, our ablation study shows that the DM cannot act on the raw observations that PPO receives, and an accurate description of each action is essential, and that surface details, such as order of text and number of decimals make no measurable diference.

Our contributions are the following:

• We introduce Decision-Model Guided Training (DMGT), a technique that uses a DM, given a description of the environment and task in natural language, to initialise an RL policy through imitation learning from the DM’s decisions, RL then further improves the policy.

• We study how the description and content given to a DM afect its ability to solve an environment, through ablation studies across seven task in six environments, we derived a guidelines for describing the environment and the task to a DM

• We compare RL alone, the DM alone and DMGT on seven tasks in six environments, and show that DMGT outperforms RL alone in all seven tasks and the DM alone in five tasks.

## 2 Related Work

Exploration from scratch. When rewards are sparse, an RL agent that starts from scratch may act for a long time before it receives any reward. Exploration bonuses based on pseudo-counts (Bellemare et al., 2016) or on the error of a learned predictor (Pathak et al., 2017; Burda et al., 2019) reward the agent for reaching novel states, and Go-Explore returns to promising states before exploring further (Ecofet et al., 2021). Reward shaping instead adds hand-designed terms that guide the agent towards the goal, and potentialbased shaping does so without changing the optimal policy (Ng et al., 1999). These methods change how the agent explores or what it is rewarded for, but the agent still starts from a random policy; we start it from the behaviour of a DM instead.

Demonstrations as a head start. A common alternative is to begin from demonstrations. AlphaGo first imitated human expert moves and then improved by self-play RL (Silver et al., 2016). DQfD and DDPGfD add demonstrations to of-policy RL (Hester et al., 2018; Vecerik et al., 2017), Nair et al. (2018) add a behaviour cloning loss, and DAPG pretrains on demonstrations and adds them to the policy gradient (Rajeswaran et al., 2018). PIRLNav and VPT pretrain a policy by behaviour cloning on human demonstrations or on videos with inferred actions, then fine-tune it with RL (Ramrakhya et al., 2023; Baker et al., 2022). The demonstrations need not be optimal: POfD surpasses the imperfect demonstrations it learns from (Kang et al., 2018), and RLPD uses previously collected, possibly suboptimal, data in online RL (Bal et al., 2023). In all of these, the demonstrations or data are recorded on the task beforehand; in our work, they come from a DM given only a description of the environment and task.

Teachers and controllers. A head start can also come from a policy that already performs the task. Kickstarting trains a student to match a trained teacher while it learns from reward (Schmitt et al., 2018), Jump-Start RL lets a guide policy act for the first part of each episode (Uchendu et al., 2023), and reincarnating RL reuses an earlier agent’s policy or data (Agarwal et al., 2022). Teachers trained with privileged information have been distilled into students that act from their own observations (Chen et al., 2019; Lee et al., 2020), and residual RL learns a correction on top of a hand-designed controller (Johannink et al., 2019; Silver et al., 2018). Each of these needs a teacher trained on the task or a controller engineered for it.

Imitation learning before RL. Behaviour cloning (BC) trains a policy by supervised learning on an expert’s states and actions (Pomerleau, 1991), but its errors compound once the policy drifts from the expert’s states, so the bound on its cost grows quadratically with the horizon (Ross & Bagnell, 2010).

DAgger reduces this to linear growth by having the expert label the states that the learner itself visits, but it needs an expert that can be queried at any state (Ross et al., 2011), which is costly with a human and straightforward with a DM. LOKI similarly runs a few iterations of online imitation before switching to policy-gradient RL (Cheng et al., 2018). RL fine-tuning can erase what a pretrained policy learned on states that are not visited early in fine-tuning (Wołczyk et al., 2024), which VPT and PIRLNav counter with a divergence penalty and a critic warm-up respectively. We use BC and DAgger only to initialise the policy, and fine-tune it with standard PPO.

Descriptions of the task in RL. Knowledge of the task can also be given in other forms. Advice-taking learners insert instructions, written in a simple programming language, into the agent’s Q-network, which RL then refines (Maclin & Shavlik, 1996). Natural language has helped agents through game manuals (Branavan et al., 2012), descriptions of entities (Narasimhan et al., 2018), documents describing new dynamics (Zhong et al., 2020) and instructions turned into shaping rewards (Goyal et al., 2019); Luketina et al. (2019) survey this line of work. In these methods, the agent learns to use the text by training on the task or on related ones, or the reward model needs annotated examples. A DM, in contrast, acts on a description without any training on the environment. In our work, the DM is used only to initialise the policy through BC or DAgger, after which PPO improves the policy from the environment reward alone, without querying the DM. We also study systematically which parts of a description a DM needs.

Language models as teachers. Large language models (LLMs) have also been used as teachers for RL. LLM4Teach distils an LLM’s choice among pretrained skills, estimated by sampling its answer several times for the same prompt, into a student RL agent that also learns from reward (Zhou et al., 2024). Yan et al. (2025) turn several free-form LLM answers per state into actions with a hand-written rule and use them as a prior over actions, while instructRL asks an LLM, one action at a time, whether to take it under a human instruction and regularises RL towards the resulting prior (Hu & Sadigh, 2023). A DM is a diferent kind of model altogether: it is built to decide rather than to write, and works like a classifier whose classes are the options given in its query, returning a probability for each in a single pass. Its answer is therefore always one of those options, there is no output to parse and nothing that can fail to parse, and one call yields the whole distribution. This makes a DM far faster and cheaper to run than a teacher that must generate text, be sampled repeatedly or be asked once per action.

## 3 Background

## 3.1 Reinforcement Learning

In reinforcement learning (RL), an agent learns to act in an environment by interacting with the environment and receiving rewards (Sutton & Barto, 2018). We model the environment as a Markov decision process (MDP) $( S , \mathcal { A } , P , r , \mu , \gamma )$ . An episode starts in a state $s _ { 0 } \sim \mu _ { ; }$ , and at each step t, the agent receives an observation $o _ { t }$ of the current state $s _ { t } \in S$ and takes an action $a _ { t } \sim \pi ( \cdot \mid o _ { t } )$ from a discrete or continuous action space ${ \mathcal { A } } .$ . It then receives a reward $r _ { t } = r ( s _ { t } , a _ { t } )$ , and the environment moves to a new state $s _ { t + 1 } \sim$ $\textstyle P ( \cdot \mid s _ { t } , a _ { t } )$ . RL seeks a policy π that maximises the expected return $\begin{array} { r } { J ( \pi ) = \mathbb { E } _ { \pi } \big [ \sum _ { t = 0 } ^ { H - 1 } \gamma ^ { t } r _ { t } \big ] } \end{array}$ over an episode of H steps, where the discount factor $\gamma \in [ 0 , 1 )$ weights near rewards more than distant ones.

We use Proximal Policy Optimization (PPO) (Schulman et al., 2017), an on-policy actor-critic method with a policy (the actor) π<sub>θ</sub> and a value function (the critic) $V _ { \psi } ( o )$ that estimates the expected return from observation o. PPO alternates between collecting a batch of steps with the current policy and updating θ for several epochs on that batch to maximise

$$
L ^ { \mathrm { C L I P } } ( \theta ) = \mathbb { E } _ { t } \Big [ \operatorname* { m i n } \big ( \rho _ { t } ( \theta ) \hat { A } _ { t } , ~ \mathrm { c l i p } \left( \rho _ { t } ( \theta ) , 1 - \epsilon , 1 + \epsilon \right) \hat { A } _ { t } \big ) \Big ] , \qquad \rho _ { t } ( \theta ) = \frac { \pi _ { \theta } \big ( a _ { t } \big | ~ o _ { t } \big ) } { \pi _ { \theta _ { \mathrm { c l d } } } \big ( a _ { t } \big | ~ o _ { t } \big ) } ,\tag{1}
$$

where $\theta _ { \mathrm { o l d } }$ are the parameters that collected the batch and ϵ limits how far one update can move the policy. The advantages $\hat { A } _ { t }$ are computed from the rewards and $V _ { \psi }$ by generalized advantage estimation (Schulman et al., 2016), and $V _ { \psi }$ is trained alongside $\pi _ { \theta }$ to predict the returns.

```latex
Algorithm 1 Decision Model-Guided Training. ϕ turns the history $h _ { t }$ (observations and actions so far) into
a query: the description of the environment and task, the current state, and one answer option per action.
$a _ { t } ^ { M } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } M ( a \mid \phi ( h _ { t } ) )$ is the decision model’s action, and $\begin{array} { r } { \operatorname { F I T } ( \mathcal { D } ) = \arg \operatorname* { m i n } _ { \boldsymbol { \pi } } \sum _ { \left( o , a \right) \in \mathcal { D } } - \log \pi ( a \mid o ) } \end{array}$
trains a new student on ${ \underline { { \mathcal { D } } } } .$
Require: environment E with actions $\mathcal { A } ;$ decision model $M ;$ description function $\phi ;$ numbers of demonstra
tion episodes N, DAgger iterations $K ,$ episodes per iteration $E$ and PPO steps $T$
1: Run M in E for N episodes, taking $a _ { t } ^ { M }$ ; store $( o _ { t } , a _ { t } ^ { M } )$ for every step in D
2: $\pi _ { 0 }  \operatorname { F I T } ( \mathcal { D } )$ ▷ behaviour cloning
3: for $k = 1 , \ldots , K$ do ▷ DAgger; K = 0 skips it
4: Run $\pi _ { k - 1 }$ greedily in $\mathcal { E }$ for E episodes; add $( o _ { t } , a _ { t } ^ { M } )$ for every step to $\mathcal { D }$
5: $\pi _ { k } \gets \operatorname { F I T } ( \mathcal { D } )$
6: end for
7: πˆ ← the student among $\pi _ { 0 } , \ldots , \pi _ { K }$ that performs best on validation episodes
8: Initialise the PPO policy $\pi _ { \boldsymbol { \theta } } ~ \hat { \pi }$ (value function untrained); train it with PPO for T steps in $\mathcal { E }$
9: return $\pi _ { \theta }$
```

## 3.2 Decision Models

A decision model (DM) is a pretrained model that chooses among given options. Examples include Jev (TypeSafe AI, 2026), the Decision 3.0 models d3-flash and d3-turbo (vLLM Semantic Router Team, 2026a;b), Clef (Cloudflare, 2026), Perplexity’s Decider (Perplexity, 2026) and Laya (Convai Innovations, 2026). To ask a decision model to make a decision, we send it a query x written in natural language. The query has three parts: a description of the situation, a question saying what is to be decided, and a finite set of answer options Y, each with a very concise description of what it means. The DM returns a probability $M ( y \mid x )$ for each option $y \in \mathcal { V } ,$ , and its choice is always the most probable option. We only query DMs, they are never trained or fine-tuned on the environment therefore they are zero-shot.

A DM can therefore perform well in an environment it has not been trained on. At every step, the query describes the environment, the task and the current state, and each action. Applying the chosen option’s action at every step defines the DM policy $\pi _ { M }$ , whose action at step t we write $\bar { a _ { t } ^ { M } }$ . Unlike an RL policy, a DM policy does not improve with experience.

## 3.3 Imitation Learning

Imitation learning trains a learner policy to act like an expert. Behaviour cloning (BC) (Pomerleau, 1991) records the expert’s observations and actions in a dataset D and trains the learner by supervised learning, typically by minimising $\textstyle - \sum _ { ( o , a ) \in { \mathcal { D } } } \log \pi ( a \mid o )$ . Because BC trains only on states the expert visits, a learner that makes a mistake can reach states the expert never visited, and its errors can compound over the episode (Ross & Bagnell, 2010). DAgger (Ross et al., 2011) reduces this by also training on the states the learner reaches: it repeatedly runs the current learner, has the expert label the observations it visited, adds them to D and trains a new learner on all of D. This requires an expert that can be asked for its action in any state, not only one that can record demonstrations.

## 4 Decision Model-Guided Training

In Decision Model-Guided Training (DMGT), an RL agent do not start from scratch, it begins with the behaviour of a DM and then improves on it. DMGT has three stages, as illustrated in the lower half of Figure 1. First, we describe the environment and the task in natural language to the DM, which then decides an action at every step without seeing the environment or the reward (Section 4.1). Second, we transfer this behaviour to the RL policy by imitation learning. The DM’s choices become labels for a network that sees only the environment’s observations (Section 4.2). Third, PPO then improves this policy by repeated interactions with the environment and the reward. The DM is used only before RL begins: PPO, and the policy it returns, act without it.

## 4.1 Describing the Task

The description function $\phi$ is the only part of DMGT that is written for a particular environment, and it is only written once. At every step, it builds a query of the form described in Section 3.2. In natural language, the description concisely explains the environment, what the agent can control, what the goal is and the state’s current values, and where useful, the last few steps with the action taken at each step. It can also include a short strategy hint: one or two sentences of general knowledge about the task, the kind a person would tell a new player. The query has the same layout at every step and only the state changes. The reward is never part of the query, the DM only acts based on the goal provided. The action of the option it chooses is $a _ { t } ^ { M }$ , and whenever the DM acts, this action is applied in the environment.

## 4.2 Transferring by Imitation

The DM cannot learn from repeated interactions with the environment and the rewards, and PPO cannot act from a query; it needs a policy network that maps the observation $o _ { t }$ to an action. We therefore train such a network, the student, to imitate the DM. The student only sees the $o _ { t }$ and it has the same architecture as PPO’s policy, so that it can directly initialise the PPO policy. Behaviour cloning fits the student to the pairs $( o _ { t } , a _ { t } ^ { M } )$ from N episodes in which the DM acts. Because a cloned student can drift into states that the DM never visited, DAgger then lets the student act and asks the DM to label every state it reaches, building each query from the student’s own history. A DM suits DAgger algorithm well, since DAgger needs an expert that can be asked for its action in any state, and a DM can provide this by answering one more query on demand, without a human in the loop.

## 4.3 Improving from Reward

PPO’s policy is initialised with the weights of the selected student πˆ. Imitation teaches actions but not values, so the value function is still untrained, same as in PPO from scratch. PPO is then trained with default hyperparameters, to ensure consistency between DMGT and PPO from scratch, with the only diference being the policy’s starting weights. PPO from scratch begins with random behaviour, randomly exploring the environment and it must find a good behaviour by this random exploration. Whereas, with DMGT the behaviour learned from the DM guides RL model to explore potentially higher value states from the start. We can therefore expect it to reach higher returns in fewer environment steps, which Section 6 tests. This is also where the blindfold comes of: the agent learns from the environment reward for the first time and can improve beyond the DM. DMGT does not rely on PPO in particular, the student can initialise the policy of any RL algorithm that uses a policy network, although we evaluate only PPO.

## 5 Experimental Setup

We evaluate Decision Model-Guided Training (DMGT) on seven tasks in six environments and compare it with PPO from scratch and also with the Decision Model (DM). Along with the final performance, we also measure the number of environment steps PPO and DMGT need to reach a given level. Additionally, an ablation study of the query breaks down the parts of the description that the DM needs, and a guideline is formed (Section 5.5).

## 5.1 Environments

We use six environments from Gymnasium (Towers et al., 2025), highway-env (Leurent, 2018), MiniGrid (Chevalier-Boisvert et al., 2023) and the Arcade Learning Environment (Bellemare et al., 2013; Machado et al., 2018), with MultiRoom in two sizes (N4-S5 and N6), and together they cover diferent types of action space, observation space and rewards. We chose them because RL alone struggles in them: the reward is sparse or delayed, or reaching it takes a long and precise sequence of actions, so PPO alone learns slowly or not at all. The exception is Highway-fast, where PPO learns quickly; it shows how DMGT does when RL alone already works. Every environment is used as gym.make returns it, with its own reward.

![](images/ae075e3346c06f26755a8f6c51e19da076049a4003c7ea024220f413bd0e4578.jpg)  
Figure 2: Rows are the four systems compared and columns are their stages. DM alone: the decision model (DM) reads a text query describing the environment and the current state, and chooses every action, with no network and no training. PPO from scratch is reinforcement learning (RL) alone, starting from random weights. The two Decision Model-Guided Training (DMGT) systems first train the policy network to imitate the DM, from the environment’s observation alone. In DMGT (BC) (behaviour cloning), the DM plays demonstration (demo) episodes and the network learns the DM’s choices. In DMGT (DAgger), this BC student then plays, the DM labels the states it reaches, and the network learns from the demos and labels, over several rounds. All three PPO runs are otherwise identical, so only the starting policy difers. All four systems are tested on the same held-out episodes, each taking its most likely action. The bars show the DM calls each system needs. The DM alone needs one at every step for as long as it is used, while DMGT needs them only before PPO.

![](images/ce063a14d4a8eadd98d8b0f3a938f5b8ab020923e72b9f4cba599d09d64f8263.jpg)  
Figure 3: The six environments, each shown at one step of an episode played by the decision model (DM). MultiRoom is used in two sizes, N4-S5 (shown) and N6. Section B describes each one.

## 5.2 Decision Models

We use d3-turbo (vLLM Semantic Router Team, 2026b), a pretrained DM that reads text and returns a probability for every answer option. It is never trained or fine-tuned on our environments and never sees their reward or a demonstration, so every decision it makes is zero-shot, from the query alone. In each environment, d3-turbo is both the DM alone (baseline) and the DM that DMGT learns from.

## 5.3 Methods Compared

We compare four methods (Figure 2): First, PPO from scratch starts from a randomly initialised policy and learns from the environment reward alone. Second, the DM alone chooses every action from the query described in Section 4.1; it never sees the reward and does not learn. Third, DMGT (BC) runs Algorithm 1 without DAgger (K = 0): the policy is initialised by behaviour cloning on the DM’s demonstration episodes and then trained with PPO. Lastly, DMGT (DAgger) adds DAgger iterations (K > 0) before PPO. The two DMGT variants and PPO from scratch use the same network, hyperparameters and number of environment steps, and difer only in the policy’s starting weights. PPO from scratch thus shows what the reward alone achieves, and the DM alone what the description alone achieves.

## 5.4 Training and Evaluation Protocol

Training. All PPO runs use the Stable-Baselines3 implementation (Rafin et al., 2021) with its default hyperparameters and network. Within an environment, every PPO run trains for the same number of steps: 200k in Mountain Car, 500k in Highway-fast, 1M in Lunar Lander and Parking, 2M in MultiRoom N4-S5 and MultiRoom N6, and 5M in Skiing. Each PPO run uses five training seeds.

Evaluation. All four methods are tested on the same fixed episodes. Their seeds are not used for demonstrations, DAgger or validation, and trained policies take their most likely action. The main measure is the share of episodes that land, park or reach the goal in Lunar Lander, Parking and MultiRoom N4-S5, the return in Mountain Car and Highway-fast, and the final time on the game clock in Skiing, with 5 s added for every missed gate. For PPO from scratch and both DMGT methods, each of the five trained policies plays all the test episodes, and we report the mean and standard deviation over all these episodes; the DM alone plays each test episode once. We also report learning curves of the return and of success during training, such as landing, parking or reaching the goal. To measure learning speed, we report the number of environment steps PPO needs to reach a set return, such as 200 in Lunar Lander.

## 5.5 Query Ablations

To find which parts of a query a DM needs, we take the full query of each environment apart, using the DM alone, without RL. Each configuration changes the full query, and the DM plays the same 3 to 20 episodes with it as with the full query. Because d3-turbo gives the same answer to the same query, one run per episode is enough, and any diference from the full query comes from the change. We first remove each part of the query on its own, such as a sentence of the description, a state value or the recent history. We also remove groups of parts together and change how some parts are written, for example giving a value as a number instead of in words. We then search for the smallest query: parts are removed, and each removal is kept only if the DM still does as well as with the full query, within a tolerance set for each environment. Because the result of this search depends on the order in which parts are removed, in most environments we check it by removing each remaining part, and pairs of parts, once more. We then test the smallest query against the full query on more episodes. From these results, and from the versions of each query that came before the ablations, we derive a detailed guideline on how to query to a DM(Section 6.3).

## 6 Results

## 6.1 DMGT against PPO and DM

Table 1 gives the return of the four methods on the test episodes. DMGT has the highest return in five out of seven tasks, both DMGT methods are ahead of PPO from scratch and of the DM alone in Lunar Lander, Mountain Car and Skiing, DMGT (DAgger) on Highway-fast and DMGT (BC) in Parking. In Lunar Lander, the return of DMGT also varies far less from one test episode to the next than that of either. In every task, at least one DMGT method consistently outperforms PPO from scratch, and only in the two MultiRoom environments does the DM alone keep the highest return.

Table 1: Mean return on the test episodes, ± the standard deviation over these episodes, pooled over the five training seeds for PPO from scratch and both DMGT methods. The return is the sum of the rewards that the environment gives over one episode, so higher is better in every row. The best mean in each row is in bold.
<table><tr><td>Environment</td><td>PPO from scratch</td><td>DM alone</td><td>DMGT (BC)</td><td>DMGT (DAgger)</td></tr><tr><td> $\mathtt { L u n a r L a n d e r - v 3 }$ </td><td> $1 8 5 . 3 \pm 8 6 . 1$ </td><td> $2 0 9 . 0 \pm 1 2 5 . 8$ </td><td> $2 6 3 . 5 \pm 2 6 . 6$ </td><td> $\mathbf { 2 6 9 . 3 \ : \pm { \ : 1 8 . 1 } }$ </td></tr><tr><td> $\mathtt { M o u n t a i n C a r C o n t i n u o u s - v O }$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $8 8 . 0 \pm 1 . 0 $ </td><td> ${ \bf 9 3 . 3 \ : \pm { \ : 0 . 7 } }$ </td><td> $9 3 . 2 \pm 0 . 6$ </td></tr><tr><td> $\mathtt { h i g h w a y - f a s t - v o }$ </td><td> $2 1 . 7 \pm 0 . 4$ </td><td> $2 2 . 7 \pm 2 . 2$ </td><td> $2 1 . 5 \pm 4 . 2$ </td><td> ${ \bf 2 2 . 9 \pm 0 . 5 }$ </td></tr><tr><td>parking-v0</td><td> $- 1 6 . 9 \pm 0 . 9$ </td><td> $- 1 9 . 4 \pm 9 . 3$ </td><td> $- \mathbf { 1 5 . 6 } \pm \mathbf { 5 . 7 }$ </td><td> $- 2 6 . 2 \pm 4 0 . 7$ </td></tr><tr><td> $\mathtt { M i n i G r i d - M u l t i R o o m - N 4 - S 5 - v 1 }$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $\mathbf { 0 . 7 6 3 \ : \pm { \ : 0 . 0 8 4 } }$ </td><td> $0 . 6 7 9 \pm 0 . 2 5 3$ </td><td> $0 . 7 2 7 \pm 0 . 1 8 6$ </td></tr><tr><td> $\mathtt { M i n i G r i d - M u l t i R o o m - N 6 - v 0 }$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $\mathbf { 0 . 6 0 4 \ : \pm { \ : 0 . 1 2 7 } }$ </td><td> $0 . 4 7 1 \pm 0 . 2 9 1$ </td><td> $0 . 3 6 0 \pm 0 . 3 2 8$ </td></tr><tr><td> $\mathtt { A L E } / \mathtt { S k i i n g - v 5 }$ </td><td> $- 3 , 8 9 9 \pm 7 0 1$ </td><td> $- 3 , 6 7 7 \pm 1 6$ </td><td> $\mathbf { - 3 , 3 4 8 } \pm \mathbf { 6 0 }$ </td><td> $^ { - 3 , 3 5 1 \pm 4 8 }$ </td></tr></table>

PPO from scratch fails in four of the tasks (Figure 5): it never reaches the flag in Mountain Car, never parks in Parking and never reaches the goal in either MultiRoom environment. Instead, it settles on a behaviour that avoids the task: in Mountain Car it learns to stand still, so that its return rises to 0 and stays there (Figure 4), and in Parking it hits a wall in every episode. DMGT succeeds in all four from its first checkpoint and keeps succeeding throughout training: it reaches the flag in every evaluation episode, the goal in about 98% of them in MultiRoom N4-S5 and in about three quarters in MultiRoom N6, and parks in about three quarters. In Skiing, DMGT misses almost no gates during training, whereas PPO from scratch still misses about one per episode at the end.

## 6.2 Learning Speed

DMGT starts from the DM’s behaviour, so it begins training at a level that PPO from scratch reaches late or never (Figure 4). DMGT (DAgger) stays above PPO from scratch throughout training in every environment, and DMGT (BC) does too, except between 21k and 59k steps in Lunar Lander and in brief dips on Highwayfast. In Mountain Car and both MultiRoom environments, the return of DMGT is close to its final level from the first training episodes. On Highway-fast, the first 100 training episodes of DMGT already average a higher return than PPO from scratch reaches at the end of training (22.5 with BC and 23.1 with DAgger, against 21.3), while PPO from scratch needs 26.5k steps to come within 5% of its own final level. In Skiing, PPO from scratch starts far below DMGT and stays below it for the whole 5M steps. In Lunar Lander, the average return of DMGT (DAgger) reaches 200 after 83k steps and that of DMGT (BC) after 216k, a level that PPO from scratch does not reach in 1M steps. The landings on the evaluation episodes follow the same order (Figure 5): DMGT (DAgger) first lands in all of them after 175k steps, DMGT (BC) after 325k and PPO from scratch after 625k, and over the whole of training they land in 92.5%, 77% and 42.5% of them.

## 6.3 Guidelines on how to Query a Decision Model

Figure 6 the results of the query ablations as guidelines for describing an environment and task to a DM. They follow from one view of the DM, it is like a blindfolded capable player, but the player has never played the game before and forgets everything between turns, so its only context is what the query tells it. A query therefore has two parts, a briefing, which is the same at every step, and a description of the current state, which keeps its form while its numbers change from one step to the next. A well-designed query lets the DM perform well. In MultiRoom N4-S5, with a query that describes the world, along with the observational data and the observational data’s explanation, and some historic data, the agent reached the goal in all test episodes.

The query is what lets the DM act. Given only the raw observations that PPO receives in place of the description, the DM does not reach the goal in any of the test episodes of MultiRoom N4-S5 and parks in none of the test episodes of Parking.

![](images/550d40586a0637642796d7130eb8b0d08921bd03460a43b1f5d1c35e86a7d73d.jpg)  
Figure 4: Return during training for PPO from scratch, DMGT (BC) and DMGT (DAgger), one panel per task. The x-axis counts PPO’s environment steps. Each line is the mean return over the last 100 training episodes, or over the episodes in the last 5% of the steps shown when these are fewer. The dark band is the standard error and the light band one standard deviation either side.

An accurate description of each action is essential. In Mountain Car, with only the names of the actions, the DM pushes right on every step and never reaches the flag, and when nothing says what the actions do, it never parks in Parking. The facts the DM cannot guess matter together, not one by one: on Highway-fast, removing any one of them barely changes the return, but removing all of them causes a collision. Rewording the goal, on the other hand, changes almost nothing.

In the description of the current state, positions are best given from the agent’s point of view: in Parking, a shorter query parks in some test episodes from the driver’s seat, but in none in the car park’s coordinates. What is directly in front of the agent also has to be described, and each value needs to be given only once, as a number or in words. Some history helps, but more does not, and it keeps each query diferent from the last, which matters because the DM gives the same answer to the same query: in many of the loops in which the DM got stuck in MultiRoom N4-S5, the query did not change from one step to the next.

The surface of the query, such as key names, the order of options and fields and the number of decimals, makes no measurable diference, and a shorter query can work as well: in Mountain Car, less than half of the tokens still reach the flag in every episode. Because any change can send an episode down a diferent path, a query has to be tested on several episodes, and a shortened one on more episodes than it was found on.

These results give the guidelines in Figure 6; the ablation results behind each one are in Section D.

## 7 Conclusion

In this paper, we introduced Decision Model-Guided Training (DMGT), which combines the complementary strengths of a decision model and reinforcement learning. A DM acts well from a description of the environment and the task in natural language, without any training, but it cannot learn from reward, while RL improves from reward but starts from scratch. DMGT uses the DM’s decisions to initialise an RL policy through imitation learning, and RL then improves this policy from the reward. Across seven tasks in six environments, DMGT outperforms RL alone in all seven tasks and the DM alone in five, and it starts training at a level that RL alone reaches late or never. Where RL alone settles on a behaviour that avoids the task, such as standing still in Mountain Car or hitting a wall in Parking, DMGT succeeds from the start of training. Once trained, the policy acts without the DM, so the DM is only needed before training.

![](images/be6e4c49fab888139cd612e06e144d24c4dfbb22661de2d5c62190878d122ab2.jpg)  
Figure 5: Success during training for PPO from scratch, DMGT (BC) and DMGT (DAgger), in the environments of Figure 4 and in Parking. At checkpoints during training (every 10k steps; 20k in MultiRoom, 25k in Lunar Lander and 250k in Skiing), the policy plays fixed evaluation episodes, other than the test episodes, with its most likely action: 10 per checkpoint, 5 in Mountain Car and 50 in MultiRoom. Each panel gives the share of these episodes in which the agent lands, reaches the flag, crashes, reaches the goal, parks or hits a wall, or, in Skiing, the gates missed per episode that reaches the finish; lower is better for crashed, gates missed and hit a wall. The band is the standard error over the five training seeds.

Through query ablations, we also showed that what a DM is told decides whether it succeeds. The DM cannot act on the raw observations that RL receives, an accurate description of each action is essential, and surface details of the query make no measurable diference. We collected these results as guidelines for describing an environment and task to a DM, which treat it as a blindfolded player who knows only what the query tells it. Together, these results show that a DM can give RL a head start without behaviour that must first be recorded or trained for the task: it only needs a good description of the environment and the task.

<table><tr><td>The environment and the task, the same at every step</td></tr><tr><td>1. The goal. State the goal as the environment scores it, and nothing else. Rewording the goal will not steer behaviour.</td></tr><tr><td>2. The options. Make each option a decision that matters, and leave low-level control, such as keeping a steady speed, to code. Then check that the task can still be done when a choice is made only as often as the DM is asked.</td></tr><tr><td>3. The actions. Describe each action accurately; this is essential. Give each option a name that says what</td></tr><tr><td>it does and a description of its measured effect, including when it does nothing. 4. One place for each action. Explain what each action does in one place only: its option. Do not describe</td></tr><tr><td>the actions again elsewhere in the query. 5. The facts. Give the facts the DM cannot guess, measured in the environment and with units, together in one place: how long each choice lasts, what ends the episode and how it is scored, what the agent can</td></tr><tr><td>see, and anything surprising. The moment, whose numbers change at every step</td></tr><tr><td>6. The point of view. Describe the moment from the agent&#x27;s point of view; map coordinates also work with context. Say what is directly in front of the agent, define left and right once, state what each sign</td></tr><tr><td>means, and use metres, seconds and degrees, never pixels. 7. No repetition. Do not repeat information, especially in a different format. Repetition can be harmful,</td></tr><tr><td>and removing it helps slightly. 8. Some history. Give the step number, the steps left, and the last few choices with what each did, but not</td></tr><tr><td>much more. History also keeps each query different: given an identical query, a DM may repeat a move that changed nothing, such as walking into a wall.</td></tr><tr><td>9. Words, not pictures. Describe; never send a picture or the raw readings. Before relying on a query</td></tr></table>

Figure 6: Guidelines for querying a decision model, drawn from the query ablations.

## References

Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, and Marc G. Bellemare. Reincarnating Reinforcement Learning: Reusing Prior Computation to Accelerate Progress. In Advances in Neural Information Processing Systems, volume 35, pp. 28955–28971, 2022.

Bowen Baker, Ilge Akkaya, Peter Zhokhov, Joost Huizinga, Jie Tang, Adrien Ecofet, Brandon Houghton, Raul Sampedro, and Jef Clune. Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos. In Advances in Neural Information Processing Systems, volume 35, pp. 24639–24654, 2022.

Philip J. Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine. Eficient Online Reinforcement Learning with Ofline Data. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, 2023.

Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The Arcade Learning Environment: An Evaluation Platform for General Agents. Journal of Artificial Intelligence Research, 47:253–279, June 2013. doi: 10.1613/jair.3912.

Marc G. Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Rémi Munos. Unifying Count-Based Exploration and Intrinsic Motivation. In Advances in Neural Information Processing Systems, volume 29, 2016.

S. R. K. Branavan, David Silver, and Regina Barzilay. Learning to Win by Reading Manuals in a Monte-Carlo Framework. Journal of Artificial Intelligence Research, 43:661–704, 2012. doi: 10.1613/jair.3484.

Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by Random Network Distillation. In International Conference on Learning Representations, 2019.

Dian Chen, Brady Zhou, Vladlen Koltun, and Philipp Krähenbühl. Learning by Cheating. In Proceedings of the Conference on Robot Learning, volume 100 of Proceedings of Machine Learning Research, 2019.

Ching-An Cheng, Xinyan Yan, Nolan Wagener, and Byron Boots. Fast Policy Learning through Imitation and Reinforcement. In Proceedings of the Thirty-Fourth Conference on Uncertainty in Artificial Intelligence, pp. 844–854, 2018.

Maxime Chevalier-Boisvert, Bolun Dai, Mark Towers, Rodrigo Perez-Vicente, Lucas Willems, Salem Lahlou, Suman Pal, Pablo Samuel Castro, and Jordan Terry. Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks. In Advances in Neural Information Processing Systems, volume 36, pp. 73383–73394, 2023.

Cloudflare. Clef. https://huggingface.co/Cloudflare/clef, 2026.

Convai Innovations. Laya. https://huggingface.co/convaiinnovations/laya, 2026.

Adrien Ecofet, Joost Huizinga, Joel Lehman, Kenneth O. Stanley, and Jef Clune. First return, then explore. Nature, 590(7847):580–586, 2021. doi: 10.1038/s41586-020-03157-9.

Prasoon Goyal, Scott Niekum, and Raymond J. Mooney. Using Natural Language for Reward Shaping in Reinforcement Learning. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, pp. 2385–2391, 2019. doi: 10.24963/ijcai.2019/331.

Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Ian Osband, Gabriel Dulac-Arnold, John Agapiou, Joel Leibo, and Audrunas Gruslys. Deep Q-learning From Demonstrations. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, April 2018. doi: 10.1609/aaai.v32i1.11757.

Hengyuan Hu and Dorsa Sadigh. Language instructed reinforcement learning for human-AI coordination. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 13584–13598, 2023.

Tobias Johannink, Shikhar Bahl, Ashvin Nair, Jianlan Luo, Avinash Kumar, Matthias Loskyll, Juan Aparicio Ojea, Eugen Solowjow, and Sergey Levine. Residual Reinforcement Learning for Robot Control. In 2019 International Conference on Robotics and Automation (ICRA), pp. 6023–6029, May 2019. doi: 10.1109/ICRA.2019.8794127.

Bingyi Kang, Zequn Jie, and Jiashi Feng. Policy Optimization with Demonstrations. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 2469–2478, 2018.

Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter. Learning quadrupedal locomotion over challenging terrain. Science Robotics, 5(47):eabc5986, October 2020. doi: 10.1126/ scirobotics.abc5986.

Edouard Leurent. An Environment for Autonomous Driving Decision-Making. https://github.com/ Farama-Foundation/HighwayEnv, 2018.

Jelena Luketina, Nantas Nardelli, Gregory Farquhar, Jakob Foerster, Jacob Andreas, Edward Grefenstette, Shimon Whiteson, and Tim Rocktäschel. A Survey of Reinforcement Learning Informed by Natural Language. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19), pp. 6309–6317, 2019. doi: 10.24963/ijcai.2019/880.

Marlos C. Machado, Marc G. Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, and Michael Bowling. Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents. Journal of Artificial Intelligence Research, 61:523–562, 2018. doi: 10.1613/jair.5699.

Richard Maclin and Jude W. Shavlik. Creating Advice-Taking Reinforcement Learners. Machine Learning, 22(1–3):251–281, 1996. doi: 10.1007/BF00114730.

Ashvin Nair, Bob McGrew, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. Overcoming Exploration in Reinforcement Learning with Demonstrations. In 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 6292–6299, 2018. doi: 10.1109/ICRA.2018.8463162.

Karthik Narasimhan, Regina Barzilay, and Tommi Jaakkola. Grounding Language for Transfer in Deep Reinforcement Learning. Journal of Artificial Intelligence Research, 63:849–874, 2018. doi: 10.1613/jair. 1.11263.

Andrew Y. Ng, Daishi Harada, and Stuart J. Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Proceedings of the Sixteenth International Conference on Machine Learning, pp. 278–287, San Francisco, CA, USA, 1999. Morgan Kaufmann Publishers Inc. ISBN 1-55860-612-2.

Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, and Trevor Darrell. Curiosity-driven Exploration by Self-supervised Prediction. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 2778–2787, 2017.

Perplexity. pplx-decider-v1.2. https://huggingface.co/perplexity-ai/pplx-decider-v1.2, 2026.

Dean A. Pomerleau. Eficient Training of Artificial Neural Networks for Autonomous Navigation. Neural Computation, 3(1):88–97, 1991. doi: 10.1162/neco.1991.3.1.88.

Antonin Rafin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stable-Baselines3: Reliable Reinforcement Learning Implementations. Journal of Machine Learning Research, 22(268):1–8, 2021.

Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine. Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations. In Robotics: Science and Systems XIV, June 2018. doi: 10.15607/RSS.2018.XIV.049.

Ram Ramrakhya, Dhruv Batra, Erik Wijmans, and Abhishek Das. PIRLNav: Pretraining with Imitation and RL Finetuning for ObjectNav. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17896–17906, 2023. doi: 10.1109/CVPR52729.2023.01716.

Stéphane Ross and J. Andrew Bagnell. Eficient Reductions for Imitation Learning. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pp. 661–668, 2010.

Stéphane Ross, Geofrey J. Gordon, and J. Andrew Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pp. 627– 635, 2011.

Simon Schmitt, Jonathan J. Hudson, Augustin Zidek, Simon Osindero, Carl Doersch, Wojciech M. Czarnecki, Joel Z. Leibo, Heinrich Kuttler, Andrew Zisserman, Karen Simonyan, and S. M. Ali Eslami. Kickstarting Deep Reinforcement Learning, March 2018.

John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. High-Dimensional Continuous Control Using Generalized Advantage Estimation. In International Conference on Learning Representations, 2016.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal Policy Optimization Algorithms, 2017.

David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of Go with deep neural networks and tree search. Nature, 529(7587):484–489, January 2016. ISSN 0028-0836, 1476-4687. doi: 10.1038/nature16961.

Tom Silver, Kelsey Allen, Josh Tenenbaum, and Leslie Kaelbling. Residual Policy Learning, December 2018.

Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. MIT Press, Cambridge, MA, 2 edition, 2018. ISBN 978-0-262-03924-6.

Mark Towers, Ariel Kwiatkowski, John U. Balis, Gianluca De Cola, Tristan Deleu, Manuel Goulão, Andreas Kallinteris, Markus Krimmel, Arjun KG, Rodrigo Perez-Vicente, Jordan Terry, Andrea Pierré, Sander Schulhof, Jun Jet Tai, Hannah Tan, and Omar G. Younis. Gymnasium: A Standard Interface for Reinforcement Learning Environments. In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track), 2025.

TypeSafe AI. Jev: Models. https://docs.typesafe.ai/models, 2026.

Ikechukwu Uchendu, Ted Xiao, Yao Lu, Banghua Zhu, Mengyuan Yan, Joséphine Simon, Matthew Bennice, Chuyuan Fu, Cong Ma, Jiantao Jiao, Sergey Levine, and Karol Hausman. Jump-Start Reinforcement Learning. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, 2023.

Mel Vecerik, Todd Hester, Jonathan Scholz, Fumin Wang, Olivier Pietquin, Bilal Piot, Nicolas Heess, Thomas Rothörl, Thomas Lampe, and Martin Riedmiller. Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards, 2017.

vLLM Semantic Router Team. d3-flash: A multimodal foundation decision model. https://huggingface. co/vllm-sr/d3-flash, 2026a.

vLLM Semantic Router Team. d3-turbo: A multimodal decision model that reads once. https: //huggingface.co/vllm-sr/d3-turbo, 2026b.

Maciej Wołczyk, Bartłomiej Cupiał, Mateusz Ostaszewski, Michał Bortkiewicz, Michał Zając, Razvan Pascanu, Łukasz Kuciński, and Piotr Miłoś. Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 53039–53078, 2024.

Xue Yan, Yan Song, Xidong Feng, Mengyue Yang, Haifeng Zhang, Haitham Bou Ammar, and Jun Wang. Eficient reinforcement learning with large language model priors. In International Conference on Learning Representations, 2025.

Victor Zhong, Tim Rocktäschel, and Edward Grefenstette. RTFM: Generalising to New Environment Dynamics via Reading. In International Conference on Learning Representations, 2020.

Zihao Zhou, Bin Hu, Chenyang Zhao, Pu Zhang, and Bin Liu. Large language model as a policy teacher for training reinforcement learning agents. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, pp. 5671–5679, 2024. doi: 10.24963/ijcai.2024/627.

## A Decision Model Queries

In every environment, the DM receives one query per decision, with the three parts of a query: a description of the situation, written as a JSON object; a question, which asks what the agent does next; and the options, which are the actions, each with a name and a description of what it does. The description opens with a briefing, the same at every step, that describes the world, the scoring, the measured facts the DM cannot guess, a short strategy hint and how to read the values. The rest of the description, the moment, describes the current state and changes at every step. Below, for each environment, are the goal, the options, what the moment contains, and the size and rate of the queries. The full queries are in the code (Section F).

Lunar Lander. Goal: “land the lander. Bring it down to the ground so that it comes to rest on its legs, without its body touching the ground.” Options: do\_nothing, fire\_left\_engine, fire\_main\_engine and fire\_right\_engine, each described by what the engine does to the lander. The moment: the lander’s position and height above the pad, its sideways and vertical speed, its tilt and turning, whether each leg touches the ground, where the pad is, and the last 10 steps. About 2230 tokens per query, one query per environment step (50 per second), 348 per test episode.

Mountain Car. Goal: “get the car to the flag.” Options: push\_left, no\_push and push\_right: full force to the left, no force, or full force to the right. The moment: the car’s position, its height above the valley floor and its speed, where the flag is, the last 10 steps, and the episode so far (the step, the steps left and the furthest positions reached). About 1290 tokens per query, one query per step, 120 per test episode.

Highway-fast. Goal: “score as much as possible over the 30-second drive: drive as fast as you can without colliding with another car, in the right lane when you can.” Options: change\_lane\_left, keep\_going, change\_lane\_right, speed\_up and slow\_down, each with its measured efect and when it does nothing. The moment: the time; the car’s lane and speed; for its own lane and each lane beside it, the car ahead, its gap and speed, the closing speed and the seconds until the car would reach it; the cars in view; and the last 5 choices. About 2250 tokens per query, one query per second, 29 per test episode.

Parking. Goal: “park the car in your space, as quickly as you can, without hitting a wall.” Options: forward\_left, forward\_straight, forward\_right, back\_up\_left, back\_up\_straight, back\_up\_right and stop: drive forward or backward with the wheels turned fully left, straight or fully right, or stop. The moment: the time, the car’s speed, where the parking space is from the driver’s seat and which way it points, how far the walls are ahead and behind, and the last 5 choices. About 1800 tokens per query, five queries per second, 54 per test episode.

MultiRoom N4-S5 and N6. Goal: “reach the green goal square in as few steps as possible”, followed by how this is scored, with 80 steps in MultiRoom N4-S5 and 120 in MultiRoom N6. Options: MiniGrid’s seven actions (turn\_left, turn\_right, move\_forward, pick\_up, drop, toggle and done), each described with what it does here, including when it does nothing. The moment: the time; the agent’s view as a text map; what is right in front of it; the doors and the goal it has seen, where they are and how many steps away; the doors it has opened; and the last 5 choices with what each did. About 1730 tokens per query in N4-S5 and 1820 in N6, one query per step, 21 and 52 per test episode.

Skiing. Goal: “ski through all 20 gates, between the two flags of each, and reach the finish as fast as you can.” Options: turn\_left, keep\_direction and turn\_right: turn the skis one notch towards the left or the right of the screen, or keep them as they are. The moment: the time; the direction of the skis and the skier’s downhill and sideways speed; the gates reached, passed and missed; where the next gate is and how soon the skier reaches it; up to three trees ahead; and the last 5 choices. About 1820 tokens per query, 15 queries per second, about 550 per test episode.

## B Environment Details

Table 2 describes each environment as PPO sees it and as the DM acts in it.

The test episodes use the environment seeds from 10000 (10000–10019; 10000–10009 in Mountain Car; 10000– 10099 in MultiRoom), the DAgger episodes the seeds from 20000, the evaluation episodes during training those from 30000, and the validation episodes those from 40000. The DM’s demonstrations use the seeds 0–99, 0–19 in Skiing, 0–7 in Mountain Car and 0 in Lunar Lander.

Table 2: The environments: what PPO observes, the actions of PPO and of the DM, the reward and the length of an episode.
<table><tr><td>Environment</td><td>Observation</td><td>Actions</td><td>Reward</td><td>Episode</td></tr><tr><td>Lunar Lander</td><td>8 values: position, ve- locity, tilt, turning rate and the contact of each leg</td><td>4: no engine, or the left, main or right engine; the DM chooses among the same four</td><td>shaped towards coming to rest on the pad: +100 for coming to rest, -100 for a crash, +10 for each leg on the ground, small costs for fir- ing the engines</td><td>1000 steps, 50 per sec- ond</td></tr><tr><td>Mountain Car</td><td>2 values: position and velocity</td><td>a force in [−1, 1]; the DM&#x27;s three options apply -1, 0 or +1</td><td>+100 at the flag, −0.1 a2 per step for the force a</td><td>999 steps</td></tr><tr><td>Highway-fast</td><td>5 × 5: presence, posi- tion and velocity of the car and of the 4 near- est cars</td><td>5: change lane left, keep going, change lane right, faster, slower; the DM chooses among the same five</td><td>up to 1 per second for speed be- tween 20 and 30 m/s and for the right lane; a collision ends the drive</td><td>30 deci- sions, one per second</td></tr><tr><td>Parking</td><td>18 values: position, ve- locity and heading of the car and of its space</td><td>acceleration and steering in [−1, 1]; the DM&#x27;s seven op- tions (forward or backward with the wheels left, straight or right, or stop) set these controls</td><td>minus the square root of the weighted distance to the space at every step, and -5 for hitting a wall; parked when the reward is above -0.12</td><td>100 s, five decisions per second</td></tr><tr><td>MultiRoom N4-S5 and N6</td><td>the 7 × 7 × 3 image of the agent&#x27;s view</td><td>MiniGrid&#x27;s 7 actions; the DM chooses among the same seven</td><td>ter t steps, 0 otherwise</td><td>1 − 0.9 t/T on reaching the goal af- T = 80 or 120 steps</td></tr><tr><td>Skiing</td><td>object features read off the screen</td><td>3: left, none, right; the DM chooses among the same three</td><td>minus the time on the game clock, with 5s added for each of the 20 gates missed</td><td>until the finish, or 5 minutes on</td></tr></table>

## C Training Details and Hyperparameters

PPO uses Stable-Baselines3 2.9.0; Table 3 lists its default hyperparameters and networks. DMGT starts PPO from the student’s policy network, and in Mountain Car and Parking also from its action spread.

Table 4 gives the imitation in each environment. Behaviour cloning minimises the cross-entropy of the DM’s choice, or, for the continuous actions of Mountain Car and Parking, the negative log-likelihood of the action that the choice applies. It uses Adam with a learning rate of $1 0 ^ { - 3 }$ and batches of 64, for 200 epochs in Lunar Lander and Mountain Car and for at most 15000 updates elsewhere (60000 in Parking). In Mountain Car, the inputs are scaled by the demonstrations’ mean and standard deviation, and this scaling is folded into the first layer, so that PPO receives the raw observation. Each DAgger iteration has the student play its episodes, labels every state it reaches with the DM’s choice, and trains the student again on all the data.

Table 3: PPO hyperparameters: the defaults of Stable-Baselines3 2.9.0.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Policy and value networks</td><td>separate, two hidden layers of 64 units, tanh</td></tr><tr><td>Environments × steps per update</td><td>1 × 2048 64</td></tr><tr><td>Minibatch size</td><td></td></tr><tr><td>Epochs per update</td><td>10 3 × 10−4</td></tr><tr><td>Learning rate</td><td></td></tr><tr><td>Discount factor γ</td><td>0.99</td></tr><tr><td>GAEλ</td><td>0.95</td></tr><tr><td>Clip range</td><td>0.2</td></tr><tr><td>Entropy coefficient</td><td>0</td></tr><tr><td>Value-loss coefficient</td><td>0.5</td></tr><tr><td>Maximum gradient norm</td><td>0.5</td></tr><tr><td>Target KL divergence</td><td>none</td></tr></table>

Table 4: Imitation in each environment: the DM’s demonstration episodes and their decisions, and the DAgger iterations and episodes per iteration with the DM’s labels.
<table><tr><td>Environment</td><td>Demonstrations</td><td>DAgger</td></tr><tr><td>Lunar Lander</td><td>1 (357 decisions)</td><td>3 × 4 (2909 labels)</td></tr><tr><td>Mountain Car</td><td>8 (903)</td><td>2 × 2 (454–457)</td></tr><tr><td>Highway-fast</td><td>100 (2828)</td><td>5 × 10 (1390)</td></tr><tr><td>Parking</td><td>100 (4559)</td><td>5 × 10 (3140)</td></tr><tr><td>MultiRoom N4-S5</td><td>100 (2075)</td><td>5 × 50 (5360)</td></tr><tr><td>MultiRoom N6</td><td>100 (5292)</td><td>5 × 50 (19802)</td></tr><tr><td>Skiing</td><td>20 (10880)</td><td>5 × 4 (11103–11115)</td></tr></table>

## D Additional Results

Table 5 gives, for each guideline in Figure 6, the ablation results it is based on. Each result compares a changed query with the full query on the same episodes, with the DM alone and no RL. The number of episodes is given with each result.

Table 5: The ablation results behind each guideline of Section 6.3.
<table><tr><td>Guideline</td><td>Evidence</td></tr><tr><td>1. The goal</td><td>Highway-fast: three rewordings of the goal change the return by at most 0.1.</td></tr><tr><td>2. Accurate action descrip- tions</td><td>Mountain Car: names without descriptions, no run reaches the flag (0 of 2), pushing right on all 300 steps. Neutral names without descriptions: no episode parks in Parking (0 of 6), and MultiRoom N4-S5 needs 23.8 more steps per episode.</td></tr><tr><td>3. Actions explained once</td><td>Mountain Car: the description of the environment also said that the car can be pushed left, pushed right or not pushed. Without that sentence, so that only the options explain the actions, the car still reaches the flag (in 112 steps instead of 110), and the shortest query leaves the sentence out.</td></tr><tr><td>4. Facts the DM cannot guess</td><td>Highway-fast (return out of 30): the facts matter as a group. Removed one at a time, each of the six lowers the return by at most 1.3, with no collision; removed all together, they lower it by 2.8, and the car collides in 1 of 3 episodes. A false fact, that the car brakes on its own, lowers the return by 1.1.</td></tr><tr><td>5. The agent&#x27;s point of view</td><td>MultiRoom N4-S5: with a shorter query, without the view and what is in front, 9 instead of 19 of 20 episodes reach the goal.</td></tr><tr><td>6. No repetition</td><td>Highway-fast: every value only as a number or only in words changes the return by at most 0.4. Without both copies of the other cars, a shortened query collides in 2 of 3 episodes.</td></tr><tr><td>7. Some history</td><td>Highway-fast: without the recent choices, 2.8 lower and a collision; with 1 to 20 of them, at most 0.1 different. Mountain Car: the shortest query fails without its one recent step. MultiRoom N4-S5: in 306 of 731 loops, the query repeats exactly.</td></tr><tr><td>8. No pictures or raw read- ings</td><td>PPO&#x27;s raw observation in place of the description: 0 of 100 test episodes in MultiRoom N4-S5, 0 of 6 in Parking, and 10.0 lower with collisions in 2 of 3 on Highway-fast; adding the description of the environment and the task does not help. Lunar Lander: with a picture at every step,</td></tr><tr><td>9. The surface</td><td>the lander flies off the screen. Highway-fast: the description in the question, 1.1 lower; key names, option order, field order and decimals, at most 0.1. Mountain Car: 489 tokens instead of 1,294 reach the flag in 8 of 8</td></tr><tr><td>10. Testing</td><td>episodes. Lunar Lander: on 5 episodes, 52 of 69 changes to the query landed fewer, or different, episodes than the full query. Mountain Car: the shortest query, found on 1 episode, held on 8.</td></tr></table>

## E Compute and Cost

Training ran on a laptop (Intel Core i7-12700H) and on two servers, each with two Intel Xeon Gold 5217 CPUs and two NVIDIA Quadro RTX 8000 GPUs. The DM was queried through its hosted free API. Table 6 gives the time of one PPO run, the DM calls that the DM alone makes per test episode, the DM calls that

DMGT makes once, for training, and the time of one call. Once trained, DMGT makes no calls. The calls include retries after failed answers.

Table 6: Compute and DM calls. PPO time: one training run, the range over the methods and seeds. DMGT calls: the demonstrations for DMGT (BC), and the demonstrations and DAgger labels together for DMGT (DAgger).
<table><tr><td>Environment</td><td>PPO time per run</td><td>DM calls per test episode</td><td>DMGT calls, BC; DAgger</td><td>Time per DM call</td></tr><tr><td>Lunar Lander</td><td>18 min</td><td>348</td><td>480; 5143</td><td>0.38 s</td></tr><tr><td>Mountain Car</td><td>2-4 min</td><td>120</td><td>903; 1357–1360</td><td></td></tr><tr><td>Highway-fast</td><td>4.1-4.2 h</td><td>29</td><td>2866; 4256</td><td>0.61 s</td></tr><tr><td>Parking</td><td>2.3–2.4 h</td><td>54</td><td>5076; 8475</td><td>0.54s</td></tr><tr><td>MultiRoom N4-S5</td><td>43-44 min</td><td>21</td><td>2075; 4172</td><td>1.50 s</td></tr><tr><td>MultiRoom N6</td><td>44–56 min</td><td>52</td><td>5292; 25094</td><td>1.71 s</td></tr><tr><td>Skiing</td><td>1.7-1.9 h</td><td>about 550</td><td>16737; about 27850</td><td>0.48 s</td></tr></table>

## F Reproducibility

Our implementation is one Python 3.12 workspace with one folder per environment and a shared package of DM clients. Every DM call is logged with its query, the probability of each option and the model that answered. The experiments use Gymnasium 1.4.0, highway-env 1.12.1, MiniGrid 3.2.0, ALE 0.12.1, Stable-Baselines3 2.9.0, PyTorch 2.14.1 and NumPy 2.5.4. The environment seeds are given in Section B.