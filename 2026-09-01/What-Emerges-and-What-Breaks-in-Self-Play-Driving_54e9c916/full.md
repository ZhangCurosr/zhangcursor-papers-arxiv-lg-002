# What Emerges and What Breaks in Self-Play Driving

Laur Sisask , Ardi Tampuu , and Tambet Matiisen

University of Tartu, Tartu, Estonia {laur.sisask,ardi.tampuu,tambet.matiisen}@ut.ee

Abstract. Training autonomous driving policies through pure self-play has recently shown promising results. Following Gigaflow and Pufer-Drive, we train driving policies in a similar self-play fashion, but extend the models from MLPs to Transformers and train on the high-definition map of a real city, where we ultimately aim to deploy them. On the CARLA and Waymax benchmarks, our policies fall short of Gigaflow, and we trace the gap to specific failure modes, including reward hacking at trafic lights and a missing incentive to stop at stop signs. We further analyze which trafic rules emerge from self-play and how closely they match human driving, and we confirm that reward conditioning yields the intended diversity of driving behaviors. A demonstration of a trained policy is available at https://laursisask-ut.github.io/eccvdemo.

Keywords: Autonomous Driving · Self-play · Reinforcement Learning

## 1 Introduction

Learning to drive through reinforcement learning requires far more experience than can safely be collected on real roads. Driving simulators are a natural alternative to real-world data collection. Photorealistic simulators such as CARLA model the driving environment in detail but have relatively low throughput, making them less suitable for large-scale reinforcement learning. Over the last few years, a number of driving simulators have emerged that meet the need for collecting large amounts of data by simulating trafic at a high level of abstraction, representing the world as objects such as vehicles and lanes instead of raw sensor data. Simulators such as Gigaflow [6], PuferDrive [3], and Waymax [9] make it possible to train driving policies at large scale, with or without human driving examples, and prior work has shown that the resulting policies can become reasonably good drivers [5, 6]. Most notably, Gigaflow reported more than 3 million simulated km between collision or of-road incidents in a long-form self-play evaluation [6]. Motivated by these results, we set out to build a similar system with the end goal of deploying it in the real world, in Tartu, Estonia.

In this article, we work towards reproducing the results of Gigaflow using the PuferDrive software library. Rather than attempting to reproduce Gigaflow’s benchmark scores, we investigate whether its broad self-play approach can support our longer-term goal of deployment in Tartu. We make the following major changes to the training process. First, we train on the high-definition map of Tartu instead of synthetic maps, which brings training closer to our intended deployment domain. Second, we explore Transformer architectures in addition to the MLPs used in prior work. Although our models do not reach the performance of Gigaflow, we analyze in detail where and why they fail, study which trafic rules the policies learn and where their behavior diverges from human driving, and verify that the reward conditioning produces the intended diversity of behaviors.

## 2 Simulation Environment

We train our models with PuferDrive [3], a high-throughput autonomous driving simulator and training loop that combines self-play with the proximal policy optimization algorithm [12]. We modify the simulator in several ways, described in the sections that follow and for the most part inspired by Gigaflow [6].

The environment runs at a timestep of 0.1 seconds, with episodes lasting 20, 30, or 40 seconds. Although training is episodic, we treat the task as continuous by bootstrapping the return from the final state with the learned value function.

## 2.1 The Map

We train our models in a world built from a high-definition map of Tartu, Estonia, a mid-sized European city. Since our ultimate goal is to deploy the models on a real autonomous vehicle in Tartu, training on the same map keeps the training environment close to the deployment environment. This contrasts with Gigaflow [6], which used synthetic CARLA maps [7], and with Cornelisse et al. [5], which used real map data from the United States.

Our map consists of 436 km of drivable lanes and presents diverse trafic scenarios, from small residential roads with right-hand-rule intersections to roundabouts and regulated intersections with up to 16 incoming lanes in total. The drivable area is captured by a set of lines indicating the edges of the road, and lanes are represented by their centerlines. Regulatory features such as trafic lights, yield signs, stop signs, and crosswalks are all represented by special lines on the map, each marking the position where a vehicle must stop when required. One limitation is that our map represents yield- and stop-sign lines as a single entity, which we call a stop line. Because yield signs are much more common in Tartu, we treat all stop lines as yield lines. This means that we do not desire that an autonomous vehicle stops at these lines unless it needs to yield to another vehicle.

The lanes form a graph in which one lane is connected to another if a vehicle is legally allowed to drive from the first to the second. This graph is used to plan a global route between any two points on the map. The map is shown in Fig. 1.

![](images/687098a08a09162ddfd17308ff981ce6f917d5dea12a792409c4a45bf936ba57.jpg)  
Fig. 1: HD map of Tartu used in the training

## 2.2 Initialization

At the start of each training episode, we sample a small region of our map: a $2 0 0 \times 2 0 0$ m area in which we randomly spawn 32 agents, requiring at least 15 meters between any two vehicles. Each vehicle receives randomly sampled dimensions, ranging from a bicycle to a bus (more details are in supplementary material). 90% of the agents spawn aligned with the lane direction, driving at a low randomly sampled initial speed, while the remaining 10% start as parked vehicles on the side of the road. Finally, we use the lane graph to generate up to 8 waypoints per agent, spaced on average 5 seconds apart when driving at the applicable speed limit.

After the spawn locations have been determined, we additionally place up to 48 stationary vehicles near the side of the road, as well as pedestrians that cross the road by following precomputed trajectories, both at crosswalks and elsewhere. For trafic lights, we use trafic-light randomization inspired by Gigaflow but with a diferent distribution: all trafic lights are disabled in 50% of the episodes, and otherwise each light is initialized with a random state (of, red, yellow, or green). During the episode, the lights change state at random times, with each state lasting between 0.5 and 10 seconds. Figure 2 shows two examples of initial episode states.

## 2.3 Reward Function and Conditioning

Following Gigaflow [6], our agents receive the following reward terms, most of which have independently sampled coeficients supplied to the model as inputs:

– a reward for reaching a waypoint, given when the agent’s distance from the waypoint is less than D m, always $R = 0 . 2 5 , D \sim U ( 1 , 3 )$

– a penalty for colliding with another vehicle, given as $R = - C _ { \mathrm { c o l l i s i o n } } ( 1 + 0 . 1 v )$ where v is speed of the vehicle in $\mathrm { n / s } , C _ { \mathrm { c o l l i s i o n } } \sim U ( 0 , 1 )$

– a penalty for colliding with the road border, given as $R = - C _ { \mathrm { o f f r o a d } } ( 1 + 0 . 1 v )$ where v is speed of the vehicle in $\mathrm { m / s } , C _ { \mathrm { o f f r o a d } } \sim U ( 0 , 1 )$

![](images/e3eb8fcaf358f5ab1437f246323f29723c13fec23871a2d4d60e83c489b3f5e1.jpg)  
Fig. 2: Examples of initial vehicle positions. Green vehicles are moving vehicles controlled by the policy, and grey vehicles are stationary. The small square boxes are pedestrians following fixed trajectories.

a penalty for crossing a trafic light line while it is red, given as $R = - C _ { \mathrm { r e d } }$ $C _ { \mathrm { r e d } } \sim U ( 0 , 1 )$

– a penalty at every timestep where longitudinal or lateral acceleration exceeds $\mathrm { 3 \ m / s ^ { 2 } }$ , given as $R = - C _ { \mathrm { c o m f o r t } } , C _ { \mathrm { c o m f o r t } } \sim U ( 0 , 0 . 0 1 )$

– a reward for driving close to center of road, given at every timestep when angle between vehicle and road centerline is less than 60 degrees, $R \ =$ $\begin{array} { r } { C _ { \mathrm { c e n t e r } } ( - | B - s | + \frac { 0 . 0 5 } { \exp ( | B - s | - 0 . 5 ) } ) } \end{array}$ where B is bias and s distance in meters from road center, $C _ { \mathrm { c e n t e r } } \sim \dot { U } ( 2 . 5 { \cdot } 1 0 ^ { - 5 } , 2 . 5 { \cdot } 1 0 ^ { - 4 } )$ with probability 50%, and $C _ { \mathrm { c e n t e r } } \sim U ( 2 . 5 \cdot 1 0 ^ { - 4 } , 2 . 5 \cdot 1 0 ^ { - 3 } )$ with probability 50%, $B \sim U ( - 0 . 3 , 0 . 3 )$

– a reward at every timestep based on the angle between the agent and the road, given as $R = C _ { \mathrm { a l i g n } } ( \operatorname* { m i n } ( \cos \alpha , 0 ) + 0 . 0 0 2 5 \cdot ( 1 - \left| \alpha \right| / \frac { \pi } { 2 } ) )$ where α is angle between the agent and centerline in radians, $C _ { \mathrm { a l i g n } } \sim U ( \bar { 2 } . 5 \cdot 1 0 ^ { - 5 } , 2 . 5 \cdot 1 0 ^ { - 3 } )$ with probability 50%, and $C _ { \mathrm { a l i g n } } \sim U ( 2 . 5 \cdot 1 0 ^ { - 3 } , 2 . 5 \cdot 1 0 ^ { - 2 } )$ with probability 50%

– a reward at every timestep where vehicle is moving at least $2 . 5 \mathrm { m } / \mathrm { s } ,$ given as $R = 2 . 5 \cdot 1 0 ^ { - 4 } \cdot \operatorname* { m a x } ( \cos \alpha , 0 )$ where α is the angle between the vehicle and road centerline.

A vehicle collision terminates an agent’s trajectory when $C _ { \mathrm { c o l l i s i o n } } \geq 0 . 5$ , and a road-border collision does so when $C _ { \mathrm { o f f r o a d } } \geq 0 . 5$ . This mirrors the real world, where a drive ends after a collision and no further rewards can be collected.

We do not penalize agents for running over stop lines. This should work identically to Gigaflow, but we note that the Gigaflow article makes some conflicting claims regarding reward at stop lines, so we are not fully sure about this. Regardless, our main reason to not penalize at stop lines is due to our treatment of them as yield signs, not stop signs — we do not want our vehicle to stop at a stop line unless it is to yield to another vehicle.

The purpose of this reward randomization is to populate the environment with a diverse set of agents. For example, some drivers avoid collisions at all costs, while others largely ignore them and focus on keeping to the center of their lane.

## 2.4 Action Space

We use the discrete action space of PuferDrive with two action dimensions one for the steering angle and one for acceleration. For acceleration, we use the PuferDrive defaults with values $\pm \{ 0 , 1 . 3 3 , 2 . 6 7 , 4 \} \mathrm { m / s } ^ { 2 }$ . For steering, we diverge from the defaults: early experiments showed that the agents struggled to keep the car aligned with the road because the default action space contained only sharp turning angles. We therefore replace some of the sharpest angles with smaller ones, giving the policy more granular control over the vehicle. The resulting steering space is ±{0, 2, 5, 10, 19, 29, 38} degrees.

## 2.5 Observation Space

We start with the default PuferDrive observation space, which includes the positions and speeds of the closest 63 objects and the positions of the closest 200 road elements, both within 100 meters. Since our perception stack in the real world does not distinguish between classes of objects like pedestrians and vehicles, we do not include the class in the observation space. If needed, the model can infer the class of an object based on its properties like dimensions or speed.

Each road element is a straight line segment on the map, observed through its middle point and direction relative to the ego vehicle. For trafic light lines, the ego additionally observes its current state — of, red, yellow, or green. In addition, the ego vehicle observes its own dimensions, speed, current and next waypoint, the number of remaining waypoints, its distance from the lane center, the angle between the lane and the vehicle, and the reward conditioning coeficients. The observation space is illustrated in Fig. 3.

## 2.6 Physics

In our early experiments, agents drove through sharp curves at speeds that would be impossible in the real world or in more realistic driving simulators. We therefore extend the PuferDrive simulator to limit the friction available to the vehicles. For each vehicle, we sample a coeficient of friction $\mu \sim U ( 0 . 7 , 1 . 3 )$ Whenever the agent’s desired net acceleration exceeds $\mu g .$ , where $g = 9 . 8 1 \mathrm { m / s } ^ { 2 }$ we constrain the lateral and longitudinal acceleration applied to the vehicle. As a result, a vehicle entering a turn too fast understeers and follows a trajectory with a larger radius than the policy requested. Unlike the other observed variables, the friction coeficient is hidden from the ego vehicle.

In addition, we introduce two further hidden variables: a steering gain and an acceleration gain, each sampled from U(0.9, 1.1). Every action selected by the policy is multiplied by the corresponding gain. Gigaflow used a similar form of conditioning, giving diferent vehicles diferent steering and acceleration capabilities, but in our case the gains are hidden from the model, whereas they included them in the agent observation.

![](images/6c3b0297518af232ee45a9c79081cfe150d3255f53cb1fd7f9c8c5d6da7c929f.jpg)  
Fig. 3: Observation space of the ego vehicle.

Road lines. Road edges and lane centerlines are represented by segments, each of which is a straight line. The ego has access to the position of the segment’s center point in egocentric coordinates (dx, dy), the angle between the line and the ego’s direction (θ), and the segment’s length.

Other vehicles. The ego vehicle has access to the position of each vehicle in egocentric coordinates (dx, dy), the angle between the ego’s direction and the vehicle’s direction (θ), as well as the vehicle’s length, width, and velocity (v).

Ego state. The ego can access its own width, length, velocity (v), the angle between the lane centerline and the vehicle (α), its lateral distance from the centerline (s), and positions of the next two waypoints in egocentric coordinates (dx, dy).

These two variables proved crucial for running our models in environments such as CARLA, where acceleration and steering are not controlled directly and follow more complex dynamics. Without this variation, our models failed to make any meaningful progress in CARLA due to excessive or insuficient steering.

## 3 Models

We train two models: a model from PuferDrive based on the Deep Sets architecture [15], and a Transformer [14] inspired by Wayformer [10]. Both models use separate networks to encode observation data from the diferent modalities into sets of vectors of the same size. Both architectures are illustrated in Fig. 4.

The Deep Sets model collapses each set into a fixed-size, order-invariant vector by taking an element-wise maximum over the set. This is done for each modality separately, and the resulting vectors are fed into an MLP followed by an LSTM, with linear heads producing the actor and critic outputs. The architecture shares Gigaflow’s modality-specific encoding and max pooling, but ours adds an LSTM. Additionally, Gigaflow uses separate feed-forward actor and critic networks.

![](images/ae72eec06c031cf0a12bf43957b750759f0b3fd862e603ebce7581a7e682306a.jpg)  
Fig. 4: Model architectures

The Wayformer-based model likewise converts each data point in the observation into a fixed-size vector. These vectors are modulated by the reward conditioning coeficients through a FiLM layer [11] and then become input tokens to a Transformer encoder, where self-attention is allowed between all tokens, including those from diferent modalities. A Transformer decoder with a single learned query embedding and cross-attention over the encoder outputs produces the actor and critic outputs. Both Transformer encoder and decoder have layer normalization at the start of every sub-layer and use ReLU as activation function. Unlike the original Wayformer, our Transformer only accesses the current state, a simplification that lowers the computational cost of running the model.

## 4 Experiments

The Deep Sets model uses the default configuration from PuferDrive: all inputs are projected to vectors of dimensionality 64 and the LSTM has a hidden size of 256, giving 596K parameters in total. The Transformer uses embeddings of dimensionality 128, an encoder with two Transformer blocks, a decoder with one block, and four attention heads, giving 638K parameters. Both models are thus around 10× smaller than the Gigaflow model, which has around 6 million parameters. We experimented with larger models but settled on these small ones:

![](images/70a4acaefae31cbcb5e56e9d4f374cbc76a3b5c8062b2391e8da38b9ba739f0a.jpg)  
Fig. 5: Examples of failures on CARLA routes. Yellow star denotes the position of the waypoint. Green shade shows the trajectory taken by the policy. The NPC vehicles are shown in blue.

(a) Deviation from the route

(b) Reward hacking at red light

(c) Collision with a vehicle

in preliminary experiments, model size made little diference in capability, and smaller models train with higher throughput.

We use the default training loop from PuferDrive with the PPO-clip algorithm [12]. Each model trains for 8–10 days on a single NVIDIA B200 GPU and 32 CPU cores, with 16,384 parallel environments for the Deep Sets model and 4,096 for the Transformer. Other hyperparameters and learning curves are given in supplementary material. Over the course of training, the Deep Sets model observes 630 years’ worth of driving data and the Transformer 57 years’ worth. The comparison is therefore matched by hardware and training time rather than by environment transitions.

## 5 Results

We evaluate our models on the existing CARLA [7] and Waymax [9] benchmarks, as well as on our own custom scenarios. While the vehicles train with diverse reward conditioning, evaluation uses fixed coeficients. For all reward terms, we set all reward coeficients to the maximal values used during training. The bias coeficient for lane center we set to 0 and goal radius to 3 m.

## 5.1 CARLA

Following Gigaflow, we use privileged trafic-participant states and detailed map geometry. Since CARLA route outcomes vary with the random seed, we run each route three times. Table 1 summarizes the results on CARLA. Overall, our models score worse than Gigaflow. We attribute this to several factors, which we analyze below.

Completion Rate. There are two reasons why our policies do not reach a 100% completion rate — deviating from the route (occurs in 45% of the scenarios) and becoming blocked (in 17% of the scenarios). Deviations from the route occur when the policy drives in the wrong direction at an intersection. After reviewing some scenarios, one case where this frequently happens seems to be when the path to the waypoint is blocked by stopped vehicles, as shown in Fig. 5a where the vehicle started turning to the right towards a waypoint, but diverted back to the main road.

Table 1: Results on diferent CARLA benchmarks. DS stands for Driving Score, RC for Route Completion, IP for Infraction Penalty, Ped for collisions with pedestrians, Veh for collisions with vehicles, Lay for collisions with layout, Red for red light violations, Stop for stop sign violations, Of for driving of- road, Dev for deviating from the route, TO for timeout, and Block for the vehicle getting blocked.
<table><tr><td>Model</td><td>DS</td><td>RC</td><td>IP</td><td>Ped Veh Lay Red Stop</td><td></td><td></td><td></td><td></td><td></td><td>Off Dev TO Block</td></tr><tr><td colspan="11">CARLA Leaderboard 1.0 Testing Routes [7]</td></tr><tr><td>Deep Sets</td><td>30±1</td><td></td><td></td><td>56±5 0.60±0.05 0.03 0.16 0.26 1.18 0.55 3.28 0.20 0.03 0.60</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Transformer</td><td>29±2</td><td></td><td></td><td>84±4 0.41±0.05 0.00 0.02 0.11 2.200.730.72 0.11 0.01</td><td></td><td></td><td></td><td></td><td></td><td>0.14</td></tr><tr><td>Gigaflow</td><td>93±1</td><td></td><td></td><td>97±2 0.95±0.01 0.00 0.07 0.00 0.01 0.00 0.64 0.01 0.07</td><td></td><td></td><td></td><td></td><td></td><td>0.00</td></tr><tr><td colspan="2">LAV benchmark [1]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Deep Sets</td><td>27±3</td><td></td><td></td><td>52±9 0.62±0.04 0.00 0.10 0.00 1.12 1.91 2.61 0.26 0.00</td><td></td><td></td><td></td><td></td><td></td><td>0.73</td></tr><tr><td>Transformer</td><td>17±2</td><td></td><td></td><td>80±1 0.34±0.00 0.00 0.04 0.00 3.66 2.13 0.04 0.09 0.00</td><td></td><td></td><td></td><td></td><td></td><td>0.30</td></tr><tr><td>Gigaflow</td><td>99±1</td><td></td><td></td><td>99±1 1.00±0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02</td><td></td><td></td><td></td><td></td><td></td><td>0.00</td></tr><tr><td colspan="2">Longest6 benchmark [2]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Deep Sets</td><td></td><td></td><td></td><td>18±1 36±1 0.62±0.02 0.06 0.38 0.17 0.18 0.46 3.99 0.42 0.02 1.36</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>Transformer 23±0.3 67±1 0.44±0.00 0.01 0.10 0.03 2.82 0.64 0.86 0.18 0.03</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.40</td></tr><tr><td>Gigaflow</td><td></td><td></td><td>92±2 99±1 0.93±0.01 0.02 0.08 0.00 0.04 0.03 0.24 0.03 0.05</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.00</td></tr></table>

The other reason, becoming blocked, occurs in two situations. The first is when our policy attempts to pass a vehicle in front of it while another vehicle is driving towards it: both vehicles brake before colliding, and neither our policy nor the NPC takes action to resolve the situation. The second case is tight turns where the policy stops in the middle of the turn even if there is space to complete it. We hypothesize that this happens due to such turns being out-of-distribution for our models, as unlike Gigaflow, they were not trained on CARLA maps.

Stop Signs. Stop-sign violations are common. We hypothesize this happens because the policy lacks a reward signal for stopping at stop signs (for reasons explained in Sec. 2.3). We watched recordings of 117 stop sign violations and found that our policy does stop at stop signs, but only when another vehicle is approaching the intersection. More specifically, we found that in 62% of violations, there were either no other vehicles at an intersection or the vehicle did not need to interact with them (e.g. turning right when vehicle approaches from right). In 24% of the cases, the policy did stop to yield to another vehicle, but did so either before or after the stop sign. In the remaining 14% of cases, the policy did not stop and created dangerous situations.

Red Lights. After manually inspecting 110 trafic light violations of our Transformer model, we estimate that around 84% happened during trafic light transitions from green to yellow to red after the vehicle had already crossed the red light stop line. We hypothesize that these violations occur because of a discrepancy between our training environment and CARLA — we penalize the agent if its bounding box intersects with a red stop line, while CARLA penalizes the agent if it has not fully driven into the intersection when light turns red, even if the agent has crossed the stop line.

In the remaining 16% of violations, our model engaged in reward hacking. In our environment, trafic lights are modeled as colored stop lines, and agents are penalized for crossing them while they are red. At intersections, these stop lines only span the lane entering the intersection, which allows the vehicle to reward hack by using the oncoming lane to bypass the red stop line without incurring a penalty. An example of this is shown in Fig. 5b. Interestingly, Gigaflow incurred very few red light violations, which is interesting because, to the best of our knowledge, they modeled the trafic light behavior and reward very similar to us.

Collisions. When it comes to collisions with other vehicles and pedestrians, our Transformer model achieves performance broadly similar to Gigaflow across the three benchmarks. In total, the Transformer-based policy participates in 15 collisions with other vehicles and one with a pedestrian. Two of these collisions happen due to the policy not yielding at a four-way stop, one of which is shown in Fig. 5c. Three collisions happen while vehicle drives in oncoming lane while passing another vehicle, five are side-swipes, in three cases the vehicle is rear-ended by an NPC, one is a head-on collision caused by NPC driving into oncoming trafic, and in one scenario the route starts with vehicle already in collision. The sole collision with pedestrian happened when a pedestrian spawned on the road, close to the vehicle, leaving very little time to react. As for collisions with the layout, we found it dificult to model some objects such as light poles, as the CARLA simulator only exposes bounding boxes, which for light poles extend over the lane. We did not include these objects in the observations, which led to occasional collisions with them.

## 5.2 Waymax

Similar to Gigaflow, we run our evaluation on scenarios from the validation split of the Waymo Open Motion Dataset (WOMD) version 1.2.0 [8]. The ego vehicle is controlled by our policy, and the other vehicles are controlled by the IDM algorithm [13]. Table 2 summarizes the results on Waymax.

On Waymax, our models perform worse than Gigaflow on the collision and of-road rate metrics. We looked at 56 failures and all these failures can be attributed to one of the following causes:

– Policy drives too close to the road edge and hits it (18%)

Table 2: Results on the Waymax benchmark on the WOMD validation set. One vehicle is controlled by our model and the others by IDM [13]. Score is Gigaflow’s custom aggregate, not an oficial Waymax metric.
<table><tr><td>Model</td><td>Off-road Collision Kinematic rate</td><td>rate</td><td>infeasibility</td><td>Log ADE</td><td>Route</td><td>Total progress driven ratio</td><td>Score</td></tr><tr><td>Expert demo</td><td>0.32%</td><td>0.61%</td><td>4.33%</td><td>0.00 m</td><td>100%</td><td>100%</td><td>≤ 99.07%</td></tr><tr><td>Deep Sets</td><td>2.36%</td><td>4.32%</td><td>0.10%</td><td>11.38 m</td><td>207%</td><td>137%</td><td>93.61%</td></tr><tr><td>Transformer</td><td>1.70%</td><td>2.20%</td><td>0.10%</td><td>8.34 m</td><td>162%</td><td>120%</td><td>96.35%</td></tr><tr><td>Gigaflow</td><td>0.43%</td><td>0.43%</td><td>0.14%</td><td>5.87 m</td><td>146%</td><td>107%</td><td>99.16%</td></tr></table>

![](images/3d19ffde5d041f9dfe32d39ba612c23b8434b4b759d05b0ee1ce18ee1122e0b9.jpg)

![](images/63aa1776e4ad02d67ea7c0bea52a3cd4049b1d2639f13fc53076daffb360adfd.jpg)  
Fig. 6: Examples of synthetic scenarios and the trajectories taken by our Transformer policy when turning left and right. Yellow star denotes position of the waypoint.

– While driving close to the road edge, the ego vehicle turns away from the edge too sharply and its rear end hits the road edge (14%)

– The ego vehicle does not leave enough room to the side when passing another vehicle (13%)

– The vehicle in front or coming from the side brakes, and the ego policy hits it (14%)

– A vehicle spawns in front or inside of the ego vehicle, and the policy collides with it (20%)

– The ego vehicle invades the lane of another vehicle, and that vehicle hits the ego from the side or from behind (11%)

– Other failures (10%)

## 5.3 Adherence to Trafic Rules

Next, we study whether the learned policy follows the trafic rules of the real world. To do this, we construct scenarios on the map of Tartu, let the policy drive them, and manually assess its behavior. Because of the manual efort involved, we restrict this evaluation to the Transformer model.

(a) Agent navigates to the right lane to safely (b) Agent navigates back to left lane after pass the pedestrian the crosswalk  
![](images/417c1ba9bb853ddc4dd993549ff846200cc7b9066db41f705fd05480456a5b4b.jpg)  
(a) Blue vehicle slows down to yield to (b) After green vehicle has passed, blue the green vehicle on the main road vehicle continues driving

Fig. 7: Example of a synthetic scenario containing a stop line and the trajectories taken by agents. Yellow star shows the position of the waypoint for both agents.  
![](images/1aa985ac3dc922d465157d2682f0119086c4eae8985af0087e8d1540df850a52.jpg)  
Fig. 8: Example of a synthetic scenario where pedestrian (in yellow) follows fixed trajectory along a crosswalk. Yellow star shows position of the waypoint for the agent.

Turning from the Correct Lane. In real trafic, there are often dedicated lanes for making left and right turns. We construct 128 scenarios where a vehicle has to make a turn but starts in the wrong lane and must change lanes before turning. Among these 128 scenarios, we observe a single failure in which the vehicle starts the turn from the wrong lane. However, for left turns the policy changes lanes only 1–3 seconds before the turn itself. Interestingly, this is not the case for right turns: in around 90% of cases, the policy changes lanes almost immediately and more gradually. Figures 6a and 6b show example scenarios and the trajectories taken by the policy.

Stop Lines. As discussed in the CARLA evaluation, our policy lacks an explicit reward for yielding at stop lines. To measure their impact, we craft 63 scenarios where two vehicles approach an intersection or roundabout, one on the main road and one on a side road with a stop line in front of it. We set the initial positions so that the vehicles would collide if both continued at constant speed. In the evaluation runs, there are no collisions. In 59% of cases, the vehicle on the main road yields to the vehicle on the side road, and in 41% the vehicle on the side road yields to the vehicle on the main road. Figure 7 shows an example scenario and the trajectories selected by the vehicles. We then repeat the evaluation with the stop lines hidden from the agent. The yielding behavior changes in only one scenario, in which the vehicle behind the stop line previously did not yield to the vehicle on the main road but now does. In all other scenarios, the yielding behavior remains exactly the same. These tests therefore provide no evidence that the policy uses the stop-line feature.

Crosswalks. First, we construct 64 scenarios where a pedestrian is waiting on the side of the road to cross at a crosswalk. In these scenarios, our policy stops in 39% of cases to let the pedestrian cross the road. Then, we take the same scenarios and make the pedestrian walk over the crosswalk along a fixed trajectory. In those scenarios, our policy stops in front of the crosswalk in 14% of cases to let the pedestrian pass. In 15% of cases, the policy does not yield to the pedestrian and drives over the crosswalk in front of them. In the remaining 71% of cases, the policy yields to the pedestrian by driving over the crosswalk behind them. Figure 8 shows an example of such a case. Despite receiving no pedestrianinteraction reward, the policy yields in 39% of waiting-pedestrian scenarios and 85% of moving-pedestrian scenarios, but its behavior remains inconsistent.

## 5.4 Behavior Diversity

In Sec. 2.3, we introduced randomized reward coeficients to encourage diverse agent behaviors, with the end goal of making the policy more robust. However, introducing such rewards does not by itself guarantee that the policy actually learns diverse behaviors. To verify this, we recorded certain events during training and analyzed how their frequencies vary with the reward conditioning coefficients. All results below are based on the last 10% of the Transformer training run.

Figure 9a visualizes the distribution of events where a vehicle goes of-road (collides with the road border) as a function of the of-road reward coeficient. Vehicles with a low coeficient clearly go of-road more frequently, but among vehicles with a coeficient larger than 0.25, the distribution becomes uniform and vehicles avoid the road border to the same extent. We hypothesize this is because beyond very low coeficient values, there is little benefit to colliding with the road border.

Figure 9b shows the number of collisions as a function of the coeficients of the two vehicles involved in the crash. The heatmap shows two trends. First, in the region where both vehicles have a low collision coeficient, collisions become much more likely. For example, 8% of collisions are between vehicles whose coeficients are both less than 0.1, despite such vehicles making up only 1% of vehicle pairs on the road.

Second, two vehicles are more likely to collide as long as just one of them has a low collision coeficient, and the value of the other coeficient makes little diference. More concretely, in 49% of collisions, one of the involved vehicles has a collision coeficient of less than 0.1. This could be because avoiding collisions is a collaborative efort, and there is only so much one driver can do.

![](images/510848a06279af3e73bad57c711f8177d1b6c0cd2bf9986dae86f63a2742237f.jpg)  
(a) Of-road events vs of-road reward coeficient

![](images/cb26cc487e53995ff36815c6cb0a6457260ca28575813b320b199da85e33e0cc.jpg)  
(b) Collision events vs collision reward coeficient

![](images/cd6219aa24b5c6841d44af84b81d0e85184875e3c003baf50573f7d6068cb793.jpg)  
(c) Lateral ofset vs lane center bias  
Fig. 9: Diversity of agent behavior due to the conditioning coeficients. The data comes from the last 10% of the Transformer training run.

Finally, Fig. 9c shows the observed lateral ofset values (the vehicle’s distance from the lane center) as a function of the lane center bias. The lane center bias represents the ofset from the actual lane center that is treated as the lane center for that particular vehicle — when the vehicle drives at this ofset, it receives the highest lane center reward. As one would expect, the observed lateral ofsets are closely clustered around the bias. For very high biases near 0.3 m, the lateral ofsets deviate from the trend, likely because positive biases are to the right of the lane center and there may not be enough room to drive that far right due to road edges.

## 6 Limitations and Future Work

The ultimate goal of this line of research is to deploy the learned policy in real trafic in Tartu, Estonia. Our results highlight several obstacles on this path. The collision and of-road rates remain too high for real-world use, and the policies obey the trafic rules inconsistently by not yielding at stop lines or crosswalks and by reward hacking at trafic lights. We see two complementary approaches to closing this gap. The first is scale: training longer and training larger models. Both of our models have around 600K parameters, roughly a tenth of the size of the Gigaflow model and tiny by the standards of modern machine learning.

The second approach is to align the policy with the trafic code explicitly rather than relying on alignment to emerge with scale. As our results show, some behaviors desired in real trafic did not emerge naturally and may need more specific guidance to be learned, either via additional rewards or imitation learning. More sophisticated reward terms could be introduced into the selfplay, such as a penalty for passing pedestrians who are waiting to cross the road. The trafic light reward in particular requires more careful modeling, e.g., an additional penalty for driving through an intersection while the light is red, so that crossing through the oncoming lane no longer avoids the penalty.

On the other hand, for some behaviors, specifying the reward may be nontrivial. For example, we have yet to come up with a general algorithm for determining whether a vehicle yielded at a stop line or not. To learn such behaviors, self-play could be combined with learning from human data, as demonstrated by Cornelisse et al. [4].

A more fundamental limitation of our current work is that it assumes perfect observation of the world. Since our deployment is limited to Tartu, assuming a perfect HD map is realistic. Perfect detection of other vehicles and objects, however, is not: some objects are hidden behind physical barriers such as walls, and detections inevitably contain errors. To work in practice, our models must therefore be trained under partial observability, with occlusions and detection noise included in the simulation.

## 7 Conclusion

Building on ideas from Gigaflow and using the PuferDrive training simulator, we trained Deep Sets and Transformer policies for autonomous driving through self-play without human driving data, with the end goal of deploying them in Tartu, Estonia. Although the policies were trained solely on a map of Tartu and without human driving data, we evaluated them on the CARLA and Waymax benchmarks, where they fell short of the performance of Gigaflow. Our analysis identified likely contributors this gap to several failure modes, including reward hacking at trafic light stop lines, disregarding stop signs, and relying too heavily on the cooperation of other vehicles. At the same time, the policies almost always turned from the correct lane in our constructed scenarios, negotiated intersections without collisions, and exhibited the diverse driving behaviors targeted by the reward conditioning. We see these results as a step towards deploying a selfplay policy in real trafic, with scale, reward design, and partial observability as the main challenges ahead.

## References

1. Chen, D., Krähenbühl, P.: Learning from all vehicles. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 17222–17231 (2022)

2. Chitta, K., Prakash, A., Jaeger, B., Yu, Z., Renz, K., Geiger, A.: TransFuser: Imitation with transformer-based sensor fusion for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 12878–12895 (2023). https://doi.org/10.1109/tpami.2022.3200245

3. Cornelisse, D., Cheng, S., Mandavilli, P., Hunt, J., Joseph, K., Doulazmi, W., Charraut, V., Gupta, A., Suarez, J., Vinitsky, E.: PuferDrive: A fast and friendly driving simulator for training and evaluating RL agents (2025), https://github. com/Emerge-Lab/PufferDrive

4. Cornelisse, D., Hunt, J., Zhang, Z., Doulazmi, W., Joseph, K., Fernández Fisac, J., Vinitsky, E.: Human-like autonomy emerges from self-play and a pinch of human data (2026), https://arxiv.org/abs/2606.19370

5. Cornelisse, D., Pandya, A., Joseph, K., Suárez, J., Vinitsky, E.: Building reliable sim driving agents by scaling self-play (2025). https://doi.org/10.48550/arXiv. 2502.14706, https://arxiv.org/abs/2502.14706

6. Cusumano-Towner, M., Hafner, D., Hertzberg, A., Huval, B., Petrenko, A., Vinitsky, E., Wijmans, E., Killian, T., Bowers, S., Sener, O., Krähenbühl, P., Koltun, V.: Robust autonomy emerges from self-play (2025), https://arxiv.org/abs/2502. 03349

7. Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., Koltun, V.: CARLA: An open urban driving simulator. In: Proceedings of the 1st Annual Conference on Robot Learning. pp. 1–16 (2017)

8. Ettinger, S., Cheng, S., Caine, B., Liu, C., Zhao, H., Pradhan, S., Chai, Y., Sapp, B., Qi, C.R., Zhou, Y., Yang, Z., Chouard, A., Sun, P., Ngiam, J., Vasudevan, V., McCauley, A., Shlens, J., Anguelov, D.: Large scale interactive motion forecasting for autonomous driving: The Waymo open motion dataset. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9710–9719 (Oct 2021)

9. Gulino, C., Fu, J., Luo, W., Tucker, G., Bronstein, E., Lu, Y., Harb, J., Pan, X., Wang, Y., Chen, X., Co-Reyes, J.D., Agarwal, R., Roelofs, R., Lu, Y., Montali, N., Mougin, P., Yang, Z., White, B., Faust, A., McAllister, R., Anguelov, D., Sapp, B.: Waymax: An accelerated, datadriven simulator for large-scale autonomous driving research. In: Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets and Benchmarks Track (2023), https://proceedings.neurips.cc/paper\_files/ paper / 2023 / hash / 1838feeb71c4b4ea524d0df2f7074245 - Abstract - Datasets \_ and\_Benchmarks.html

10. Nayakanti, N., Al-Rfou, R., Zhou, A., Goel, K., Refaat, K.S., Sapp, B.: Wayformer: Motion forecasting via simple & eficient attention networks. In: 2023 IEEE International Conference on Robotics and Automation (ICRA). pp. 2980–2987 (2023). https://doi.org/10.1109/ICRA48891.2023.10160609

11. Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A.: FiLM: Visual reasoning with a general conditioning layer. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 32 (2018). https://doi.org/10.1609/aaai.v32i1. 11671

12. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms (2017), https://arxiv.org/abs/1707.06347

13. Treiber, M., Hennecke, A., Helbing, D.: Congested trafic states in empirical observations and microscopic simulations. Physical Review E 62, 1805–1824 (2000). https://doi.org/10.1103/physreve.62.1805

14. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. In: Advances in Neural Information Processing Systems. vol. 30, pp. 5998–6008. Curran Associates, Inc. (2017)

15. Zaheer, M., Kottur, S., Ravanbakhsh, S., Póczos, B., Salakhutdinov, R., Smola, A.J.: Deep sets. In: Advances in Neural Information Processing Systems. vol. 30, pp. 3391–3401. Curran Associates, Inc. (2017)