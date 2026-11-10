# VLA Adaptation with Real and Simulated Data: A Case Study on Robotic Chess

Matej Kripnerˇ

Milan Straka

Charles University, Faculty of Mathematics and Physics

Prague, Czech Republic

kripner@ufal.mff.cuni.cz

Abstract: Vision-language-action (VLA) models are trained as general-purpose foundation models for robot learning, and require fine-tuning for each downstream task and robot. This raises practical questions such as how much data and training updates are needed, how checkpoints should be selected, and whether simulation can be leveraged to alleviate the cost of real data collection. We study these questions on a task that requires both precise manipulation and spatial understanding of the scene. In this task, an SO-101 arm moves a chess king between two squares that are given only by their names, such as e4 and a5. We fine-tune $\pi _ { 0 . 5 }$ with various dataset sizes and update budgets, using teleoperated demonstrations as well as simulated demonstrations from a randomized scripted expert that imitates the variability of human demonstrations. In simulation, the success rate scales with both data and updates. Importantly, we find that a short learning-rate cool-down adds up to 31 points and is necessary for comparing checkpoints, and that validation loss and action error are unreliable for model selection. On the real arm, we fine-tune on up to 256 teleoperated demonstrations and evaluate ten policies in 25 trials each. Partial results indicate that pretraining on the simulated demonstrations helps substantially, most clearly when real data is scarce. We also present a negative result, in which training on the policy’s own rollouts with filtered behavior cloning, RECAP-style advantage conditioning, corrections from an automatic coach, or residual reinforcement learning did not yield a measurable improvement over plain behavior cloning in our setting. We release the simulator, the expert and the real demonstrations.

## 1 Introduction

A common way to teach a robot a new manipulation task is to fine-tune a pretrained vision-languageaction (VLA) model [1, 2, 3] on human-collected demonstrations of that task. This raises practical questions, such as how many demonstrations to collect, how long to train, which checkpoint to deploy, and whether pretraining in simulation is worthwhile. Each of these choices carries a substantial cost in data, compute or robot time.

A central challenge is that closed-loop task success, the quantity of interest, is expensive to measure. On a real robot, each trial takes minutes and requires a person to reset the scene. It is therefore tempting to rely on offline proxies such as validation loss or action error, or on a few dozen trials per setting. Our study complements the fine-tuning experiments reported with the VLA models themselves [1, 2, 3] and data-scaling studies of imitation learning [4] with a detailed look at a single task, where a simulator lets us evaluate every model on many trials.

In this paper, we study the task of moving a chess king on a board with a low-cost robot arm. Given an instruction such as “Move the king from e4 to a5,” the policy must locate the named squares on the board it sees and then grasp, carry and place the king (Fig. 1). The task therefore requires both precise manipulation and spatial understanding of the scene. We fine-tune $\pi _ { 0 . 5 }$ [2] on teleoperated demonstrations and on simulated demonstrations from a randomized scripted expert. The simulator provides demonstrations in any quantity and lets us evaluate every model on the same 1,000 graded trials. On the real arm, we fine-tune on up to 256 teleoperated demonstrations, starting either from the base $\pi _ { 0 . 5 }$ model or from a policy pretrained on the simulated demonstrations. We also test whether the policy improves when it is trained on its own rollouts. Beyond king moves, we also built a system on the same arm that plays full games with all pieces, but only as a demonstration, without evaluation or ablations.

![](images/947fae4e8f0ef939d61c5630f94e0795a469aa91b1817753ee20f4374d55c391.jpg)  
Figure 1: The five stages of a move on the real arm (top) and in simulation (bottom), labeled with the stage tokens the policy predicts.

In summary, our contributions are:

• In a simulated version of the task, we present quantitative experiments indicating that the success rate scales with both data and updates, that checkpoints must be compared after a learning-rate cool-down or stochastic weight averaging (SWA) [5], and that validation loss and action error are unreliable for model selection (Section 3.1).

• On the real arm, partial results indicate that pretraining on simulated demonstrations helps when real data is scarce (Section 3.3).

• As a negative result, training on the policy’s own rollouts with filtered behavior cloning, RECAP-style advantage conditioning [6], corrections from an automatic coach or residual reinforcement learning did not yield a measurable improvement over behavior cloning in our setting (Section 3.2).

• We release the simulator with its randomized scripted expert and the teleoperated demonstrations, and contribute the recordings of the real evaluation trials to the Oopsie Data collection of real-robot policy rollouts<sup>1</sup> (Section 2.2, Appendix G).

## 2 Setup

The task is to pick up the king from one square of an otherwise empty chess board, place it on another square and return to the ready pose (Fig. 1). The source and target squares are given only in the instruction. We study the task on a real robot arm and in its simulated counterpart.

## 2.1 Real Arm

The real setup consists of a low-cost 3D-printed arm with five joints and a gripper, the SO-101 [7, 8], a second arm of the same type as the leader for teleoperation, and two cameras (Appendix A). One camera is mounted on the wrist and looks along the gripper. The other is fixed above the board on a 3D-printed overhead mount (Fig. 3). Both have wide-angle fisheye lenses, and the policy sees their images rectified to the simulator’s pinhole model. The board is wooden with 31 mm squares. The king is a wooden piece, 64 mm tall on a 20 mm base.

![](images/b7417012261353c7f7245b3510f60376335f13242e87e16a66223e9f3e9ecd79.jpg)

![](images/cc5865607625709ef43bce81cda35cf8adea27a8cfbc1096573c244a03804a06.jpg)

![](images/3297212071290d03e1480a7efb52302d53297d3b976286c10ade7c812555e1b0.jpg)

![](images/3cfb863e3a54b03c93752374546176129e0d91f2f48e36de62205cc52382d27f.jpg)  
Figure 2: Gripper paths seen from above, from the randomized scripted expert, teleoperated demonstrations, trials of the simulation policy, and the first trial of eight of the real policies of Section 3.3. Dashed paths indicate unsuccessful trials.

![](images/1ad73a799a5c31de2d2fcacf67128e32a5f71c7913e4bdba57c3fdf73d6a3a2f.jpg)  
Figure 3: Inputs and outputs of the policy at one frame of a real trial. The policy observes the two rectified camera images, the joint positions, the instruction and the current stage. It outputs a chunk of 50 joint targets and predicts whether to transition to the next stage.

The task is difficult for this arm for four reasons. First, the policy has to find each square from its name alone. Second, the arm’s posture varies considerably across the board, from steeply folded near the base to almost fully extended at the far ranks. Third, the gripper has one fixed finger and one moving jaw, and a grasp that is a few millimeters off the king’s axis topples the piece. Fourth, the printed links flex under load by up to 2 cm, so the arm cannot be positioned precisely without visual feedback.

## 2.2 Simulator and Scripted Expert

The simulator is a MuJoCo [9] twin of the real setup. It reproduces the arm’s kinematics and servo dynamics, the measured pose of the board, the scanned king, and the poses and fields of view of the two cameras.

A motion planner would execute every move identically, which neither resembles teleoperation nor gives enough variation for imitation learning. Our scripted expert therefore imitates the variability of a human operator. It reaches for a target with a perception error, observes the remaining error after a latency and corrects, and repeats this until the error is below a tolerance. Each episode samples its own tempo, errors, tolerances and grasp parameters, so paths and speeds differ between episodes (Fig. 2). Grasp errors arise from the same mechanism, and the expert detects a missed grasp, reopens and retries. It succeeds in 93% of its episodes.

The paths of the human operator on the real arm show a similar spread, although the operator’s carries are straighter. The fine-tuned policies vary less than both the expert and the operator, in simulation and on the real arm.

![](images/a3c3e783babdc1719c6a2532a9239296927057a5d2cab6b03b53991b95a5c1f2.jpg)  
Figure 4: Success rate of the cooled-down models over updates, one curve per dataset size, with Wilson 95% intervals. Dashed lines mark the end of each size’s first epoch.

## 2.3 Policy

We start from the $\pi _ { 0 . 5 }$ model, which combines a vision-language model with a flow-matching action expert that attends to it. The model was not pretrained on the SO-101 arm. The policy receives the two camera views, the six joint positions, the instruction and the current stage. It outputs chunks of 50 joint targets at 15 Hz and executes 10 of them before planning again (Fig. 3). We fine-tune the action expert and the language model, 2.7 of the model’s 4.1 billion parameters, and keep the vision encoder frozen. The language model also learns to predict the current stage as one token (Fig. 1), and the action expert attends to that token (Appendix C). At test time the policy re-scores the current and the next stage once per action chunk and advances when the next one wins twice in a row.

## 3 Experiments

Our experiments address three questions, namely how behavior cloning scales with data and training (Section 3.1), whether the policy can improve from its own rollouts (Section 3.2), and how much simulation pretraining helps on the real arm (Section 3.3). Appendix B lists all simulation results.

## 3.1 Behavior Cloning in Simulation

This section studies behavior cloning on the simulated demonstrations, where every model can be evaluated on many trials.

Data. The simulated demonstration set holds about 11.4k expert episodes with the piece offset by up to 8 mm inside its square, of which 93% are successes. Training sets are nested subsets of the successes with n = 100, 316, 1,000, 3.2k and 10k episodes, and 30 further successes are held out for validation.

Training. Every run uses a constant learning rate of $3 \cdot 1 0 ^ { - 5 }$ after 500 warm-up steps and a global batch of 32 frames. The update budgets form a geometric ladder of eight steps from 1,000 to 165k (one epoch of the largest set), extended to 237k for the two largest sets. For each dataset size, one constant-rate trunk keeps a checkpoint at 90% of every update budget, and a branch from each checkpoint cools the rate to zero over the remaining updates [10, 11]. Each configuration is trained once.

Evaluation. Each model runs 1,000 closed-loop trials on a fixed schedule of moves, with the piece offset by up to 8 mm as in the data, and with real-time chunking [12] at deployment settings. A trial succeeds if the king ends standing with its center inside the target square and the arm is back at rest. Strict success also requires the king’s whole base to be inside the square. We report success rate with Wilson 95% intervals, which are about ±3 points at 30–60%.

![](images/007fad94de1326fe90ba47c75fc2dd840b695709a02ab19243696819a6457267.jpg)  
Figure 5: Constant-rate checkpoints against the cooled-down branches started from them.

![](images/8869a0d3c4149c2cefadd2a0ff405c5b4ae9cf661e0322d8d025cdf744dd79a6.jpg)

![](images/4ef150642e121c519de10ce5963a1566bd8791457226ed4e3638bab2f5984d17.jpg)

![](images/f6bec5a6c620b6f53475863752b3bd1487a358d3485d25efb9edbf495531d425.jpg)  
Updates (log scale)  
Figure 6: Validation loss (left) and open-loop action error (middle) of the constant-rate trunks, against the success rate of the cooled-down models (right). One curve per dataset size.

Success increases with data and with updates. Below about 4k updates no model succeeds, independent of the dataset size (Fig. 4). Above that, success increases along both axes. With 10k demonstrations it is still increasing at the largest budgets, from 60% at 165k updates to 64% at 237k. At small update budgets, additional data has little effect (10% and 11% with 1,000 and 10k demonstrations at 9k updates). In our grid, a smaller dataset performs as well as a larger one until its first epoch is complete, and performs worse afterwards. Training for many epochs still helps, and in no case does it lower success beyond the evaluation noise. In a partial ablation at 10k demonstra tions, a model without $\pi _ { 0 . 5 } \mathrm { ^ { \circ } s }$ robot pretraining reaches the same success rate (Appendix F).

A learning-rate cool-down is necessary for comparing checkpoints. A branch continues its trunk checkpoint for the last 10% of the budget while the learning rate falls to zero (Fig. 5). With 10k demonstrations, the branch ending at 165k updates reaches 60%, the constant-rate checkpoint it started from reaches 29%, and the trunk’s own checkpoint at 165k updates reaches 49%. The trunk’s success rate is not monotonic in the number of updates, and early stopping on its checkpoints would select a model 18 points below what the run reaches with a cool-down. Initial results suggest that SWA over the same updates is nearly as effective, with 57% against 60% for the cool-down at 10k demonstrations (Appendix B).

Offline metrics are unreliable for model selection. With 100 demonstrations the validation loss is lowest at about 2k updates and more than doubles by the end of training, while success increases from zero to 10% (Fig. 6). Selection by validation loss would therefore return a model with zero success. The open-loop action error does fall monotonically, but it does not discriminate between models. At about 79k updates the error is 22 to 24 mm for all five dataset sizes, while the succes rate of the cooled-down models ranges from 8% to 48%.

Where trials fail. The milestones are acquired in order during training (Fig. 7). Touching the piece comes first, gripping and lifting follow together, and placing comes last and is still improving at the largest budget. Most failures after a grip are toppled pieces, in about a third of all trials. The cool-down mostly improves the grasp and the placement.

![](images/f058a7f1f7f891e2d25ea7a27827619cbd6c381a870bfd83cc790d8dfd6515c0.jpg)  
Figure 7: Share of the 1,000 trials that reach each milestone (blue, strict success dashed) and that topple the piece or time out (orange, timeouts dashed), at 10k demonstrations. Left: cooled-down models. Right: constant-rate checkpoints.

![](images/4c61e1cb8cb47a7586fc44b172057b07029cdb618294a043788adde813486d5f.jpg)  
Figure 8: The value function of the 1K+1K round. Left: predicted value (solid) and actual return (dashed) on held-out episodes. Right: distribution of the advantage over the training frames, on a symmetric-log axis. Rollout frames right of the dashed threshold are labeled positive; demonstration frames are labeled positive regardless. One tick is one control step, 1/15 s.

## 3.2 Learning from the Policy’s Own Rollouts

In this section, we evaluate whether the policy improves when it is trained on its own rollouts in addition to the demonstrations.

Setup. We take the best behavior-cloning model of a dataset size (165k updates), run it on new moves from the collection distribution and record every trial with its graded outcome. A new policy is trained from the base model on the demonstrations plus the rollouts, on the same update ladder. There are two rounds. 1K+1K uses 1,000 demonstrations and 1,000 rollouts of the 1,000- demonstration model (32% success). 10K+10K uses 10k demonstrations and 10k rollouts of the 10k-demonstration model (60% success).

We compare four variants. Filtered BC trains on the demonstrations plus the successful rollouts. RECAP-style advantage conditioning [6] labels every frame with a learned value function. A crossfitted variant of it labels each half of the episodes with a value function that never saw it. Outcome labels mark every frame of a success as positive and every frame of a failure as negative. We sample advantage-conditioned policies with the positive label, optionally with classifier-free guidance [13]. The baseline is the model trained on the demonstrations alone for the same number of updates. A large value function trained for several epochs memorized its training episodes in our setting, which makes its labels uninformative. We therefore use a smaller value function, train it for one epoch and monitor it on held-out rollouts (Fig. 8). Appendix E gives the details and all numbers.

![](images/46e81ea80d0a88e7e44e6f8017202d10352bae216a46c9d8e31a085ecd38b42e.jpg)  
Figure 9: Partial results in the real setting, 25 trials per model. Left: policies initialized from the simulation policy. Right: policies initialized from the base $\pi _ { 0 . 5 }$ model. Top: success rate (%) over demonstrations and updates, where a dash marks a model without a full evaluation. Bottom: share of trials that reach each milestone, per evaluated model.

Results. No variant improves over behavior cloning beyond the evaluation noise (Appendix E, Table 1). The three advantage-conditioned variants match behavior cloning. The policy does use the label, since success drops from 30% with the positive label to 24% with the negative label. Without a label it reaches 26%, guidance at $\beta = 1 . 5$ gives 31%, and guidance at $\beta = 2 . 5$ lowers success to 23% (Appendix E, Fig. 14). Filtered behavior cloning is about three points ahead of behavior cloning at two intermediate update budgets, but not at the largest one. The failed rollouts, which only the labeled variants see, provide no measurable benefit.

Corrections and residual reinforcement learning. In the original RECAP, a person corrects the policy during its rollouts. We reproduce this in simulation with an automatic coach. A monitor on the simulator’s state hands control to the scripted expert when a failure is imminent, and the expert regrasps or places the king and returns control to the policy. Training on about 1,000 coached rollouts gives no consistent gain. The best coached variant is above behavior cloning on one evaluation schedule and below it on the other. Residual reinforcement learning, in which a small network trained with PPO adds joint offsets to the frozen policy’s actions, did not improve success in any of four runs.

## 3.3 Experiments on the Real Arm

On the real arm of Section 2.1, we measure how much real data the policy needs and whether pretraining in simulation helps.

Data. One of the authors teleoperated 261 king moves with the leader arm, each starting and ending at the ready pose. The source and target squares were drawn at random, and the king was placed up to 8 mm off the center of its square, as in the simulated data. 259 of the moves are successes, and these form the real dataset, about half an hour of motion in total. The stage labels are detected from the recorded trace with the simulator’s own definitions.

Training. We fine-tune $\pi _ { 0 . 5 }$ on nested random subsets of 32, 128 and 256 demonstrations, from two initializations. Base is the released $\pi _ { 0 . 5 }$ model. Pretrained in simulation is our $\pi _ { 0 . 5 }$ simulation model trained on 10k demonstrations for 237k updates and cooled down (64% success in simulation). We call the policies fine-tuned from it sim-initialized and those fine-tuned from the base $\pi _ { 0 . 5 }$ model base-initialized. The recipe is the same as in simulation (Section 3.1). A constant-rate trunk runs to 45k updates and keeps checkpoints at 90% of the budgets 2k, 15k and 45k updates, and a branch cools down from each checkpoint. The trunks are also extended to 75k updates, with one more cooled-down model per trunk.

Evaluation. Every model runs the same 25 trials, the first 25 moves of the simulator’s easy schedule, with the king centered on its square. The policy runs each trial autonomously, at one third of the demonstrated tempo because of the inference latency in our deployment. A person resets the king and grades the highest milestone reached, with the same definitions of success and strict success as in simulation (Appendix H). With 25 trials per model, we only interpret large differences.

Results. The results are partial, with ten models evaluated in full (Fig. 9, Appendix B, Table 2). In all five evaluated pairs the sim-initialized policy is better, and with little real data the difference is large. With 128 demonstrations, the sim-initialized policy succeeds in 18 of 25 trials and the baseinitialized policy succeeds in 2. With 32 demonstrations, the sim-initialized policy succeeds in 7 trials, while the base-initialized policy never grips the king. With 256 demonstrations the difference is smaller. Depending on the update budget, the sim-initialized policy succeeds in 14 to 17 trials and the base-initialized policy in 5 to 10 trials.

## 4 Related Work

VLAs and their adaptation. π [1] introduced the pairing of a vision-language model with a flow-matching action expert, and π<sub>0.5</sub> [2] extended it with co-training on heterogeneous data for open-world generalization. SmolVLA [3] showed that a small open model trained on community data is competitive. Lin et al. [4] measured data scaling in imitation learning for policies trained from scratch.

Simulation as pretraining. Simulated demonstrations are a cheap complement to real ones. Cotraining a policy on both, with the simulator roughly matched to the real scene, raises real success well beyond what the real data alone gives [14], and large simulated corpora are part of the pretraining mix of recent generalist models [15].

Learning-rate schedules. Warmup-stable-decay schedules [10] train at a constant rate and cool down briefly at the end. Hagele et al. [¨ 11] showed that cool-downs branched off one constant-rate run match full cosine runs at every length, so one run per dataset size is enough for a whole update axis. They also suggested SWA [5] as an alternative to the cool-down.

Learning from experience. RECAP [6] trains a value function on the policy’s own rollouts and corrections, labels each frame by its advantage, conditions the policy on a text indicator of that label, and steers with classifier-free guidance [13] at test time. It improved throughput and robustness on long real-world tasks. Filtered behavior cloning, which keeps only the successful rollouts, is a simple baseline for learning from rollouts.

## 5 Conclusion

We studied the fine-tuning of a VLA for a chess-move task with real and simulated demonstrations. In our setting, fine-tuning choices could only be judged by closed-loop evaluation on many trials, which the simulator made affordable. The same simulator also provided useful pretraining data for the real arm.

Limitations. Our results come from one task and with a single training run per setting. The scripted expert is randomized but its similarity to human demonstrations is not rigorously measured. Our RECAP variants omit the original’s advantage-conditioned pretraining and human corrections. The statistical evidence from the real arm is limited, with 25 trials per model.

## Acknowledgments

This project was supported by the Charles University Grant Agency (GAUK), project no. 458326. This research was partially supported by SVV project number 260 821. We acknowledge VSB – Technical University of Ostrava, IT4Innovations National Supercomputing Center, Czech Republic, for awarding this project access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium through the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (grant ID: 90254).

## References

[1] K. Black, N. Brown, D. Driess, A. Esmail, M. R. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, L. X. Shi, L. Smith, J. Tanner, Q. Vuong, A. Walling, H. Wang, and U. Zhilinsky. π<sub>0</sub>: A vision-language-action flow model for general robot control. In Robotics: Science and Systems (RSS), 2025.

[2] K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. R. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke, A. Walling, H. Wang, L. Yu, and U. Zhilinsky. π<sub>0.5</sub>: a vision-language-action model with open-world generalization. In Conference on Robot Learning (CoRL), volume 305 of Proceedings ofMachine Learning Research, pages 17–40, 2025.

[3] M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti, S. Alibert, M. Cord, T. Wolf, and R. Cadene. SmolVLA: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.

[4] F. Lin, Y. Hu, P. Sheng, C. Wen, J. You, and Y. Gao. Data scaling laws in imitation learning for robotic manipulation. In International Conference on Learning Representations (ICLR), 2025.

[5] P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, and A. G. Wilson. Averaging weights leads to wider optima and better generalization. In Conference on Uncertainty in Artificial Intelligence (UAI), 2018.

[6] Physical Intelligence, A. Amin, R. Aniceto, A. Balakrishna, K. Black, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, J. DiCarlo, et al. π<sup>∗</sup> : a VLA that learns from experience. arXiv preprint arXiv:2511.14759, 2025.

[7] R. Knight, P. Kooijmans, R. Cadene, S. Alibert, M. Aractingi, D. Aubakirova, A. Zouitine, M. Russi, S. Palma, C. Pascal, and T. Wolf. Standard Open SO-100 & SO-101 Arms. https: //github.com/TheRobotStudio/SO-ARM100, 2024.

[8] R. Cadene, S. Alibert, A. Soare, Q. Gallouedec, A. Zouitine, S. Palma, P. Kooijmans, M. Aractingi, M. Shukor, D. Aubakirova, M. Russi, F. Capuano, C. Pascal, J. Choghari, K. Meftah, M. Ellerbach, J. Moss, and T. Wolf. LeRobot: State-of-the-art machine learning for real-world robotics in Pytorch. https://github.com/huggingface/lerobot, 2024.

[9] E. Todorov, T. Erez, and Y. Tassa. MuJoCo: A physics engine for model-based control. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 5026– 5033, 2012.

[10] S. Hu, Y. Tu, X. Han, C. He, G. Cui, X. Long, Z. Zheng, Y. Fang, Y. Huang, W. Zhao, et al. MiniCPM: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

![](images/efa09a8067033bdc4bfa8b3c23000fdddad207881a31805ab49c515b685aac50.jpg)  
Figure 10: Both camera frames as recorded (raw) and rectified to the simulator’s pinhole model, as the policy sees them.

[11] A. Hagele, E. Bakouch, A. Kosson, L. Ben Allal, L. Von Werra, and M. Jaggi. Scaling laws and¨ compute-optimal training beyond fixed training durations. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[12] K. Black, M. Y. Galliker, and S. Levine. Real-time execution of action chunking flow policies. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[13] J. Ho and T. Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[14] A. Maddukuri, Z. Jiang, L. Y. Chen, S. Nasiriany, Y. Xie, Y. Fang, W. Huang, Z. Wang, Z. Xu, N. Chernyadev, S. Reed, K. Goldberg, A. Mandlekar, L. Fan, and Y. Zhu. Sim-and-real cotraining: A simple recipe for vision-based robotic manipulation. In Robotics: Science and Systems (RSS), 2025.

[15] NVIDIA, J. Bjorck, F. Castaneda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, ˜ F. Hu, S. Huang, et al. GR00T N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

## A Setup of the Real Arm

Hardware. The arm is the SO-101 of The Robot Studio and LeRobot [7, 8], with a follower and a leader arm. The gripper consists of a fixed finger and a moving jaw with flat pads. The two cameras are identical USB modules with a 130<sup>◦</sup> diagonal field of view, recorded at 1280 × 720. The follower and both cameras connect to a laptop that runs the control loop, and the policy is served from a remote GPU. The simulator uses the arm’s URDF and servo dynamics, the measured board pose, the king’s scanned profile, and the calibrated camera poses. It does not model the compliance of the printed links.

Sim-to-real differences. The real cameras have fisheye lenses and the simulator’s cameras are pinhole cameras. We record the real frames raw and rectify them to the simulator’s intrinsics in training and deployment (Fig. 10).

The joint zeros of the real calibration differ from the simulator’s by a constant offset per joint, which normalization with the real dataset’s statistics absorbs. Two differences are not handled. The real policy runs at a third of the demonstrated tempo, and the simulator applies the 15 Hz joint targets as steps where the real controller interpolates them.

Collection. The operator sees the next move on a screen, places the king, and moves it with the leader arm, starting and ending at the ready pose. The recorded frame rate varied between sessions (15 to 23 frames per second, stored as 30), so the policies were trained on a mix of apparent tempos.

## B Result Tables

This appendix lists the numbers behind the figures of Section 3.

Table 1: Simulation results: success rate (% of 1,000 trials). Top: behavior cloning, with models cooled down unless noted. Bottom: learning from 1,000 of the policy’s own rollouts.
<table><tr><td>Model</td><td>Demos</td><td>Updates</td><td>Success</td></tr><tr><td> $\pi _ { 0 . 5 } .$  behavior cloning</td><td>100</td><td>165k</td><td>9.9</td></tr><tr><td>π0.5, behavior cloning</td><td>316</td><td>165k</td><td>22.0</td></tr><tr><td>π0.5, behavior cloning</td><td>1,000</td><td>165k</td><td>31.7</td></tr><tr><td>π0.5, behavior cloning</td><td>3.2k</td><td>165k</td><td>39.7</td></tr><tr><td>π0.5, behavior cloning</td><td>10k</td><td>165k</td><td>60.2</td></tr><tr><td> $\pi _ { 0 . 5 } ,$  behavior cloning</td><td>10k</td><td>237k</td><td>64.1</td></tr><tr><td>constant rate, no cool-down</td><td>10k</td><td>165k</td><td>48.9</td></tr><tr><td>SWA, no cool-down</td><td>10k</td><td>165k</td><td>57.0</td></tr><tr><td>no robot pretraining</td><td>10k</td><td>165k</td><td>59.5</td></tr><tr><td> $\pi _ { 0 } ,$  stock recipe</td><td>10k</td><td>165k</td><td>77.7</td></tr><tr><td>SmolVLA, stock recipe</td><td>10k</td><td>165k</td><td>54.5</td></tr><tr><td>Behavior cloning, demonstrations only</td><td>1,000</td><td>79k</td><td>29.9</td></tr><tr><td>Filtered BC (+ successful rollouts)</td><td>1,000</td><td>79k</td><td>32.9</td></tr><tr><td>RECAP, value function</td><td>1,000</td><td>79k</td><td>30.0</td></tr><tr><td>RECAP, cross-fitted value function</td><td>1,000</td><td>79k</td><td>31.3</td></tr><tr><td>RECAP, outcome labels</td><td>1,000</td><td>79k</td><td>29.8</td></tr><tr><td>RECAP, outcome labels, guidance  $\beta = 1 . 5$ </td><td>1,000</td><td>79k</td><td>31.4</td></tr><tr><td>RECAP, coached rollouts</td><td>1,000</td><td>79k</td><td>34.7</td></tr><tr><td>Filtered BC, coached rollouts</td><td>1,000</td><td>79k</td><td>28.9</td></tr></table>

Table 2: Evaluation on the real arm, as the number of the 25 trials that reached each milestone.
<table><tr><td></td><td></td><td colspan="3">Pretrained in simulation</td><td colspan="3">Base</td></tr><tr><td>Demos</td><td>Updates</td><td>Touched</td><td>Gripped</td><td>Success</td><td>Touched</td><td>Gripped</td><td>Success</td></tr><tr><td>32</td><td>2k</td><td>25</td><td>12</td><td>7</td><td>6</td><td>0</td><td>0</td></tr><tr><td>128</td><td>45k</td><td>25</td><td>24</td><td>18</td><td>15</td><td>8</td><td>2</td></tr><tr><td>256</td><td>15k</td><td>25</td><td>23</td><td>14</td><td>23</td><td>13</td><td>5</td></tr><tr><td>256</td><td>45k</td><td>25</td><td>22</td><td>14</td><td>23</td><td>17</td><td>10</td></tr><tr><td>256</td><td>75k</td><td>25</td><td>23</td><td>17</td><td>24</td><td>15</td><td>10</td></tr></table>

## C Stage Prediction

![](images/4d1ccc39dde0e2e278a99a8312b3cacdf9ff3ca77d5d597bbde25ec10ab2d89b.jpg)

![](images/e3bbce2df0be4689e8d551bc43e919a987f92713b7b2a0303cec22dc843c59ec.jpg)  
Figure 11: Accuracy (left) and cross-entropy (right) of the predicted stage on held-out demonstrations.

The stage accuracy is close to its final value after about 4k updates for every dataset size, long before the success rate rises (Fig. 11).

## D Value Function Training

![](images/ca856c4a3f7c7f0f632c827b7fb5cdd8368bba6fff2fc0c5fe125abbc0e4dfd7.jpg)  
Figure 12: Error of the 1K+1K value function during its single training epoch, on held-out and on training episodes. One tick is 1/15 s.

At the end of the epoch, the error is 205 ticks on held-out episodes and 135 ticks on training episodes (Fig. 12). Failed episodes account for most of it, with 375 ticks held-out against 106 ticks on successful episodes.

## E Details on Learning from Rollouts

![](images/e30573352260ad2ce866fb0d6910ac383b6c52f1e59324ac03f89ea03ff339e5.jpg)

![](images/7b45d87c9d2cd0dc700685e4545966c3fd7470a2887af7f123269ddaa6ddd530.jpg)  
Figure 13: Learning from the policy’s own rollouts, cooled-down models, each sampled with the positive label.  
Figure 13 shows the variants of Section 3.2 at every update budget they were trained for.

![](images/302a1e753d844ac6449df9c36cc2c413fa549fc9ed22db5e9c07b795f585fcd2.jpg)  
Guidance strength β  
Figure 14: Success of the outcome-label RECAP policy at 79k updates, sampled with different labels and guidance strengths $\beta .$ Shaded: the Wilson 95% interval of behavior cloning.

Variants. All variants train on the demonstrations plus the policy’s rollouts and are compared with behavior cloning on the demonstrations alone at the same number of updates.

• Filtered BC adds only the successful rollouts.

• RECAP, value function follows RECAP [6] with a 500M value function trained for one epoch. A frame’s advantage is the progress made over the next second minus the progress the value function expects. The threshold is the 30th percentile of the advantages of all training frames, demonstrations included. Rollout frames above it are labeled positive, and every demonstration frame is labeled positive. This labels 72% of the frames of successful rollouts and 48% of those of failed rollouts positive.

• RECAP, cross-fitted labels each half of the episodes with a value function that never saw it.

• RECAP, outcome labels marks every frame of a success as positive and every frame of a failure as negative.

Advantage-conditioned policies are sampled with the positive label, and with classifier-free guidance [13] at $\beta \in \{ 1 . 5 , 2 . 5 \}$

Value function. A value function with the policy’s 3B backbone, trained for two epochs, memorized its training episodes and was at chance on unseen ones. The smaller value function used here is trained for one epoch and monitored on held-out rollouts (Fig. 8, Appendix D). On held-out episodes it separates successes from failures with an AUC of 0.85 four seconds into the episode, which is before the grasp.

Results. At 79k updates in the 1K+1K round, behavior cloning reaches 30%, the three advantageconditioned variants reach 30 to 31%, and filtered BC reaches 33% (Table 1). At 165k updates, filtered BC reaches 34% against 32% for behavior cloning, and the advantage-conditioned variants reach 29 to 32%. Sampled with the negative label, without a label and with the positive label, the outcome-label model reaches 24%, 26% and 30% (Fig. 14). In the 10K+10K round, outcome labels reach 46% at 79k updates against 48% for behavior cloning.

Corrections from an automatic coach. A monitor on the simulator’s state hands control to the scripted expert when the king tilts or slides, when a grasp misses, or when a forward simulation of the queued actions predicts a topple. We train on about 1,000 coached rollouts with RECAP labels and as filtered BC on the coached successes. At 79k updates coached RECAP reaches 35% on the hard schedule, above behavior cloning (30%, paired McNemar test, $p \ < \ 0 . 0 2 )$ , but on the easy schedule it is below behavior cloning (46% against 50%).

Residual reinforcement learning. A small network trained with PPO from privileged simulator state adds a joint offset of at most $3 ^ { \circ }$ to the frozen policy’s actions. In four runs with different rewards and exploration noise, the success rate stayed within evaluation noise of its initial value.

## F Pretraining Ablation

![](images/207d79fc3ff679cb851b9db6b5702d56441ec5a2cd15c4e6a6508d1d06e01b3f.jpg)  
Figure 15: Partial results of the pretraining ablation, for cooled-down models at 10k demonstrations. Left: $\pi _ { 0 . 5 }$ with and without robot pretraining, trained with the same recipe. Right: other model families with their stock recipes.

We initialize the vision-language model from PaliGemma, on which $\pi _ { 0 . 5 }$ is built, and initialize the action expert and the projections randomly. Trained with the same recipe on 10k demonstrations, this model follows the same curve as $\pi _ { 0 . 5 }$ and also reaches 60% at 165k updates (Fig. 15). This result is partial, since the ablation covers only the largest dataset size, and robot pretraining may matter with smaller datasets. Other model families, fine-tuned with their stock recipes, perform in the same range or better. At 165k updates, $\pi _ { 0 }$ [1] reaches 78% and SmolVLA [3] reaches 55%. Their recipes differ from ours, so these numbers compare model and recipe together.

## G Released Dataset and Environment

The real demonstrations are in the LeRobot format (Table 3). The simulated demonstrations can be regenerated with the released environment and scripted expert. The code is at https://github.com/kripner/vla-chess and the dataset at https://huggingface.co/datasets/Kripi/so101-chess-king-moves-real.

Table 3: Released dataset.
<table><tr><td>Dataset</td><td>Source</td><td>Episodes</td><td>Contents</td></tr><tr><td>king moves, real</td><td>teleoperation</td><td>261</td><td>the demonstrations of Section  $3 . 3$ </td></tr></table>

## H Evaluation Protocol on the Real Arm

Every model is evaluated on the same 25 trials, the first 25 moves of the simulated easy schedule, in which each trial’s target square is the next trial’s source and the king starts centered on its square. A trial starts with the arm at its ready pose and ends when the policy switches to its final stage or afte 60 s of control time, which counts as a failure. The human evaluator only resets the king and grades the highest milestone reached (Table 4). A failed trial is never discarded, and a trial is repeated only for a setup fault, such as the king placed on the wrong square. Every trial is recorded, whether it succeeds or not.

Table 4: Milestones graded by the evaluator. Success is milestone 5 or 6, strict success milestone 6.
<table><tr><td></td><td>Milestone</td><td>Observed</td></tr><tr><td>0</td><td>none</td><td>the jaws never touched the king</td></tr><tr><td>1</td><td>touched</td><td>a jaw touched the king</td></tr><tr><td>2</td><td>gripped</td><td>the king was between both jaws, touching both</td></tr><tr><td>3</td><td>lifted</td><td>gripped and lifted clear of the board (about 1.5 cm)</td></tr><tr><td>4</td><td>over target</td><td>carried until its center was above the target square</td></tr><tr><td>5</td><td>success</td><td>at the end the king stands, released, its center inside the target square</td></tr><tr><td>6</td><td>strict success</td><td>as  $5 ,$  and its base lies entirely inside the square</td></tr></table>