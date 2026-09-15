# Task-Distribution-Aware Counterweight Synthesis and Constrained Co-Design for Serial Manipulators

Mohammad Abbadi

College of Engineering and Information Technology, University of Dubai Dubai, United Arab Emirates mabbadi@ud.ac.ae

## Abstract

9 September 2026 Preprint – not peer reviewed

Passive counterweights are mechanically simple gravity compensators, but a counterweight selected from a single pose is not generally optimal for the configurations and tasks a manipulator actually executes. This paper develops a task-distribution-aware synthesis framework in which the operating measure � q enters the design explicitly. For a counterweight moment $p ~ = ~ m _ { c } r _ { c }$ whose gravity torque is $- g p \phi ( \mathbf { q } )$ , the weighted mean-square residual gravity torque has the closed-form minimizer $p ^ { * } = \mathbb { E } _ { \rho } [ \tau _ { g } \phi ] / ( g \mathbb { E } _ { \rho } [ \phi ^ { 2 } ] )$ . If payload gravity torque is afine in payload mass, the optimum is likewise afine, $p ^ { * } ( m _ { p } , \rho ) = p _ { 0 } ^ { * } ( \rho ) + m _ { p } K _ { p } ( \rho )$ . The formulation also makes a second design fact explicit: for fixed static moment, added counterweight inertia is $I _ { c } = p r _ { c }$ while counterweight mass is $m _ { c } = p / r _ { c } ,$ so mass–radius selection is underdetermined unless packaging, mass, structural, or actuator constraints are supplied. A recovered three-link physical manipulator is used as a transparent case study. At $r _ { c } = 0 . 2 0 \mathrm { m }$ equivalent zero-payload optima are 0.672 kg for uniform joint-space operation, 0.683 kg for approximately uniform task-space operation, 0.713 kg for a representative pick-and-place family, and 0.952 kg for a declared high-gravity-biased distribution. Thus the selected mass changes by more than 40% solely through operating-distribution choice. Full nondominated fronts show that geometric knees move with declared engineering bounds, directly exposing the need for physical constraints. A rated-torque-referenced all-joint screen increases zero-payload feasible task-space coverage from 78.1% without compensation to 93.7% for the uniform-distribution design. A lumped point-mass trajectory study identifies a provisional crossover from no counterweight at very aggressive motion to stronger compensation as motion slows. These actuator and dynamic results are engineering consequence studies rather than physical validation; identified rigid-body inertias, friction, and paired experiments remain necessary for hardware-level claims.

Keywords: gravity compensation; counterweight synthesis; task distribution; static balancing; serial manipulator; constrained multi-objective design.

## 1 Introduction

Passive gravity compensation reduces the actuator efort required to support links and payloads without continuous external power. Counterweights, springs, cable–pulley systems, compliant mechanisms, magnetic devices, and gear–spring modules have all been used for this purpose [1–3]. Counterweights are particularly transparent mechanically, but the mass added to generate a balancing moment also adds inertia andjoint reaction. That trade-of becomes important in lightweight manipulators where the balancing mass is comparable with the moving mass [4, 5].

Recent mechanism research has moved beyond the binary question of whether gravity can be balanced. Work has addressed four-bar and gear–spring synthesis [6–9], spatial and multi-DoF compensation [10–13], variable payloads [14], permanent-magnet compensation [15], and dynamic degradation under velocity, acceleration, and unbalanced loads [16]. Current reviews therefore identify payload variability, multi-DoF coupling, non-ideal components, and multi-objective trade-ofs as active challenges [2].

For a pure counterweight, however, two design ambiguities remain easy to conceal. Static compensation is governed by the first mass moment

$$
p = m _ { c } r _ { c } ,\tag{1}
$$

whereas the point-mass inertia added about the compensated joint is

$$
I _ { c } = m _ { c } r _ { c } ^ { 2 } = p r _ { c } .\tag{2}
$$

Infinitely many mass–radius pairs therefore generate the same static moment but have diferent inertia, total mass, packaging demand, and structural consequence. Equally importantly, the moment that should be compensated is not unique unless the manipulator’s intended operating distribution is declared.

The central question of this paper is: for a declared operating distribution and payload model, what passive counterweight moment minimizes residual gravity torque, and what physical constraints are required to turn that moment into a unique mass–radius design?

The paper makes four contributions. First, it derives a general operating-distribution-aware closed-form synthesis law and a payload-afine extension. Second, it formalizes why counterweight mass–radius realization is underdetermined without physical constraints. Third, it quantifies task-distribution sensitivity and replaces an arbitrary weighted-knee heuristic with complete nondominated fronts and a weight-free geometric diagnostic under explicitly labeled engineering scenarios. Fourth, it demonstrates system consequences on a recovered three-link prototype using all-joint rated-torque reference classes, quasi-static task-space feasibility, and a provisional dynamic-regime map. Historical hardware is used for provenance, not as experimental validation of the new designs.

## 2 Related work and positioning

Classical static balancing has a mature theoretical foundation for serial and parallel mechanisms [1, 17–20]. Modern balancing taxonomies distinguish gravity balancing from inertial and force balancing, an important distinction because a statically attractive mechanism can remain dynamically unfavorable [5]. Martini et al. combined counterweights and springs and ranked variants using task-level indicators [4]; later work introduced compact spring, cable-driven, coupled-spring, and permanent-magnet compensators [8, 10, 11, 13, 15].

Dynamic-load analysis has shown that the benefits of static balancing degrade with velocity, acceleration, and unbalanced payload [16]. Variable-payload and multi-objective formulations are likewise established [14, 21, 22]. The contribution here is therefore not the existence of counterweights, variable-payload compensation, or generic multi-objective optimization. It is the explicit introduction of the operating measure into closed-form counterweight synthesis, the resulting payload law, and the mechanism-design consequence that a static moment does not uniquely determine a physically preferred mass–radius pair.

## 3 Task-distribution-aware counterweight synthesis

## 3.1 Weighted gravity-compensation objective

Consider a serial manipulator with configuration ${ \bf q } \in \mathcal { Q }$ and a counterweight acting about one selected joint. Let the uncompensated gravity torque about that joint be $\tau _ { g } ( \mathbf { q } , m _ { p } )$ , where $m _ { p }$ is payload mass. A counterweight with moment $p = m _ { c } r _ { c }$ contributes

$$
\tau _ { c } ( \mathbf { q } , p ) = - g p \phi ( \mathbf { q } ) ,\tag{3}
$$

where $\phi$ is the gravity projection determined by the mounting geometry. For the shoulder-mounted case study, $\phi ( \mathbf { q } ) = \cos q _ { 1 }$ . Let $\rho ( \mathbf { q } ) \geq 0$ be a normalized operating density. The weighted mean-square residual gravity torque is

$$
J ( p ) = \int _ { Q } \rho ( { \bf q } ) \left[ \tau _ { g } ( { \bf q } , m _ { p } ) - g p \phi ( { \bf q } ) \right] ^ { 2 } d { \bf q } .\tag{4}
$$

Proposition 1 (weighted RMS-optimal passive moment). If $\mathbb { E } _ { \rho } [ \phi ^ { 2 } ] > 0 , J ( p )$ is strictly convex and has the unique unconstrained minimizer

$$
\boxed { p ^ { * } ( m _ { p } , \rho ) = \frac { \mathbb { E } _ { \rho } [ \tau _ { g } ( \mathbf { q } , m _ { p } ) \phi ( \mathbf { q } ) ] } { g \mathbb { E } _ { \rho } [ \phi ( \mathbf { q } ) ^ { 2 } ] } } .\tag{5}
$$

The result follows directly from $d J / d p = - 2 g \mathbb { E } _ { \rho } [ ( \tau _ { g } - g p \phi ) \phi ]$ and $d ^ { 2 } J / d p ^ { 2 } = 2 g ^ { 2 } \mathbb { E } _ { \rho } [ \phi ^ { 2 } ] > 0$ . Manipulator geometry and mass distribution enter through $\tau _ { g } \mathrm { . }$ ; intended operation enters through $\rho .$

## 3.2 Payload-afine synthesis

For rigid-link gravity loading, payload contribution is linear in payload mass. Write

$$
\tau _ { g } ( { \bf q } , m _ { p } ) = S _ { 0 } ( { \bf q } ) + m _ { p } S _ { p } ( { \bf q } ) .\tag{6}
$$

Substitution into Eq. (5) gives

$$
\Big | p ^ { * } ( m _ { p } , \rho ) = p _ { 0 } ^ { * } ( \rho ) + m _ { p } K _ { p } ( \rho ) \Big | ,\tag{7}
$$

with

$$
p _ { 0 } ^ { \ast } = \frac { \mathbb { E } _ { \rho } [ S _ { 0 } \phi ] } { g \mathbb { E } _ { \rho } [ \phi ^ { 2 } ] } , \qquad K _ { p } = \frac { \mathbb { E } _ { \rho } [ S _ { p } \phi ] } { g \mathbb { E } _ { \rho } [ \phi ^ { 2 } ] } .\tag{8}
$$

Thus the optimum is afine in payload under the stated gravity model, but both intercept and slope depend on the declared task distribution.

## 3.3 Mass–radius underdetermination

For a prescribed static moment $p , m _ { c } = p / r _ { c }$ and $I _ { c } = p r _ { c }$ . Reducing $r _ { c }$ reduces point inertia while increasing mass. Hence neither static torque nor inertia alone yields a unique physical pair $( m _ { c } , r _ { c } )$ . Packaging, mass, structural, clearance, or actuator constraints are required to make the co-design well posed. This observation is used below to interpret, rather than hide, boundary-active Pareto solutions.

## 4 Case-study model and verification

## 4.1 Recovered physical parameters

The case study is a planar three-link manipulator reconstructed from original project records. The moving-link lengths are $L _ { 1 } = 0 . 2 3$ m, $L _ { 2 } = 0 . 2 0$ m, and $L _ { 3 } = 0 . 1 4 ~ \mathrm { m }$ . The audited lumped masses are ${ \tt W } 2 { = } 0 . 2 1 4$ kg, ${ \mathrm { W } } 3 { = } 0 . 0 6 0$ kg, W4=0.060 kg, ${ \tt W S } { = } 0 . 0 5 6$ kg, $\scriptstyle \mathrm { W } 6 = 0 . 1 2 6$ kg, $\scriptstyle \mathbf { W } 7 = 0 . 0 5 6 \mathbf { k g }$ , and ${ \mathrm { W } } 8 { = } 0 . 0 7 8$ kg. The source counterweight is 0.887 kg at 0.20 m. Historical records identify a Power HD HD1235MG base servo and 1501MG joint servos, but their published stall ratings are not treated as continuous-duty actuator limits.

At the horizontal pose, the audited model gives a shoulder gravity torque of 2.17384 N m. The historical source counterweight reduces it to 0.43414 N m, an 80.03% pose-specific reduction. This value is retained as a provenance check, not as the principal contribution.

## 4.2 Gravity-vector verification

For point mass �, let $\mathbf { r } _ { i } ( \mathbf { q } )$ be its planar position and $J _ { i }$ its translational Jacobian. The reconstructed gravity vector is

$$
\mathbf { G } ( \mathbf { q } ) = \sum _ { i } m _ { i } g J _ { i , y } ( \mathbf { q } ) ^ { T } .\tag{9}
$$

The counterweight changes only the shoulder component by $- g p$ cos $q _ { 1 }$ . The implementation is checked against an independently evaluated potential-energy gradient and grid-convergence tests in the accompanying audit package. These are numerical verification tests; they are not physical validation of the robot.

![](images/21fbefb1def0d1d244e6e56d401a79f77ff215545f6a7224f237b4b8d3927e77.jpg)  
Figure 1: Planar three-link case-study model. The counterweight moment $p = m _ { c } r _ { c }$ and its inertia $I _ { c } = p r _ { c }$ separate static compensation from mass–radius realization.

Table 1: Zero-payload task-distribution-aware synthesis at $r _ { c } = 0 . 2 0 ~ \mathrm { n }$ .
<table><tr><td>Distribution</td><td> $p ^ { * } \left( \mathrm { k g } \mathrm { m } \right)$ </td><td> $m _ { c } \ ( \bf k g )$ </td><td>RMS (N m)</td></tr><tr><td>Uniform joint space</td><td>0.13446</td><td>0.672</td><td>0.592</td></tr><tr><td>Approx. uniform task space</td><td>0.13656</td><td>0.683</td><td>0.623</td></tr><tr><td>Pick-and-place family</td><td>0.14259</td><td>0.713</td><td>0.554</td></tr><tr><td>High-gravity-biased</td><td>0.19045</td><td>0.952</td><td>0.295</td></tr><tr><td>Folded-biased</td><td>0.14401</td><td>0.720</td><td>0.698</td></tr></table>

## 5 Operating-distribution and payload results

Five declared operating measures are examined: uniform joint space; an approximately uniform task-space occupancy measure using a 60 60 Cartesian occupancy grid; a representative family of three quintic point-topoint joint trajectories; a Gaussian high-gravity bias near extended configurations; and a folded-biased Gaussian sensitivity scenario. The latter two are declared scenarios rather than measured duty-cycle distributions.

The same robot therefore admits materially diferent optimal moments solely because $\rho$ changes. Relative to uniform joint-space operation, the high-gravity-biased equivalent mass is more than 40% larger. Accordingly, the phrase “optimal counterweight” is incomplete unless the objective measure is stated.

The payload-afine law is numerically satisfied to machine precision over 0–0.5 kg for every tested distribution. For uniform joint-space operation at $r _ { c } = 0 . 2 0 \mathrm { m }$

$$
m _ { c } ^ { * } ( m _ { p } ) \approx 0 . 6 7 2 3 + 1 . 3 0 8 9 m _ { p } \quad \mathrm { ( k g ) } .\tag{10}
$$

The slope is distribution dependent, as predicted analytically.

## 6 Constrained mass–radius co-design

The previous weighted scalar knee is replaced by full nondominated fronts in residual RMS gravity torque and added counterweight inertia, with mass and radius reported explicitly. A geometric knee, when shown, is selected by maximum deviation from the chord joining normalized Pareto endpoints; no subjective mass weight enters the construction.

The historical archive establishes only one directly documented radius, 0.20 m, and does not establish structural hard limits. Accordingly, the study separates a historical fixed-radius analysis from three explicitly labeled engineering scenarios (compact, moderate, and extended). These are scenario bounds for sensitivity, not hardware-certified constraints.

![](images/00b2c72836e2a8d211868f49a4ceda28931730aee1fb54bfae03f168b5f9f7a4.jpg)  
Figure 2: Continuous sensitivity of the equivalent optimum to a mixture between uniform joint-space and high-gravity-biased operating measures.

The selected geometric knee changes with the admissible design box and can become mass- or radiusbound. This is not interpreted as a failure of optimization: it numerically confirms the analytical result that a counterweight moment does not uniquely determine a physical mass–radius realization.

Parameter uncertainty is treated conservatively as sensitivity rather than reliability. Independent 2%, 5%, and 10% perturbation scenarios are applied to reconstructed mass/lever contributions. Because these are not measurement-derived probability distributions, no probabilistic reliability claim is made.

## 7 Actuator-reference and workspace implications

The recovered historical hobby servos publish stall rather than defensible continuous-duty torque. For an engineering threshold study only, current CubeMars frameless motors are used as reference classes: RO60, RO80, and RO100 publish rated torques of approximately 0.8, 1.3, and 4 N m, respectively [23–25]. They are not claimed as drop-in replacements.

Under nominal static peak torque, the uncompensated shoulder requires 2.174 N m at the horizontal pose, the source design leaves about 1.307 N m peak over the sampled envelope, and the uniform-� design reduces the sampled peak to approximately 1.001 N m. This threshold crossing is an actuator-class feasibility indication, not sustained downsizing evidence; dynamic peaks, thermal duty, mechanical integration, and safety margins remain to be checked.

A quasi-static task-space cell is considered feasible if at least one sampled configuration reaches it while all three gravity torques satisfy reference limits of 1.3, 0.8, and 0.8 N m for joints 1–3. A zero-thickness link-1/link-3 self-intersection test removes obvious planar self-crossings. Finite link thickness and counterweight swept volume are unavailable.

At zero payload, feasible coverage increases from 78.1% without compensation to 93.7% for the uniformdistribution design. At 0.25 kg payload, the corresponding fractions are 25.8% and 40.8%. Compensation therefore enlarges the torque-feasible region, but payload exposes the unaltered requirements of joints 2 and 3.

![](images/10e45d1919147d6e2fc1762d3566c0cfe1c9749d794071107ef4e6268b6c5135.jpg)  
Figure 3: Payload-afine equivalent optimum for three declared operating distributions.

## 8 Provisional dynamic design regimes

The static synthesis theory does not require a dynamic model, but motion aggressiveness determines whether added inertia can outweigh gravity benefit. The source archive lacks identified rigid-body rotational inertias, reflected motor/gear inertia, friction, and drivetrain eficiency. A point-mass inverse-dynamics surrogate is therefore retained only to identify qualitative regime hypotheses.

Four fixed designs are compared on the same quintic point-to-point family: no counterweight (D0), the historical source counterweight (D1), the uniform-joint static-RMS design (D2), and a declared scenario reference of 1.0 kg at 0.10 m (D3). Figure 6 reports the design with minimum surrogate shoulder RMS torque over payload and motion duration. Extremely aggressive motion favors no counterweight in part of the tested domain, while slower motion favors stronger compensation. The crossover is a hypothesis for identified multibody and experimental validation, not a measured motor-energy result.

## 9 Generalization, limitations, and discussion

The reusable result is Eq. (5), not a prototype-specific mass. It separates a robot geometry/mass term from an operating-distribution term, while the relation $I _ { c } = p r _ { c }$ exposes the additional physical constraints required to realize a moment. The three-link arm demonstrates the magnitude of task dependence and the way actuator thresholds, payload, and dynamic aggressiveness can alter the preferred design.

![](images/0c034c1646567284bf38202448784acfc8804b1817003dde7c76ee24679e692d.jpg)  
Figure 4: Weight-free RMS-torque versus added-inertia nondominated fronts under declared engineeringbound scenarios. Boundary movement is itself evidence that physical packaging and structural constraints are required.

Three evidence layers are deliberately distinguished. First, the analytical implementation is numerically verified by independent gravity/potential-energy calculations and convergence studies. Second, a recovered historical SimMechanics model establishes project provenance but is too simplified for high-fidelity validation. Third, the new counterweight designs have not been validated experimentally. The archive contains no repeated D0–D3 current, voltage, joint-torque, or measured tracking data.

The dominant remaining limitations are physical rather than numerical: identified rigid-body inertias and COMs, gearbox/motor inertia, friction/backlash and eficiency, actual mass/radius packaging bounds, bracket stress and stifness, finite-thickness collision geometry, continuous-duty actuator integration, and paired measurements. These limitations bound the engineering consequence studies but do not alter the closed-form static synthesis result.

## 10 Conclusions

A passive counterweight cannot be called “optimal” independently of how a manipulator is expected to operate. Introducing the operating measure $\rho ( \mathbf { q } )$ into the weighted residual-gravity objective yields a closed-form synthesis law and an exact payload-afine extension under the stated model. The same formulation exposes why static counterweight moment does not uniquely determine physical mass and radius: reducing radius lowers point inertia but increases mass, so packaging, structural, or actuator constraints are necessary to make the design well posed.

On the recovered three-link case study, the equivalent zero-payload design at 0.20 m shifts from 0.672 kg under uniform joint-space operation to 0.683 kg under approximately uniform task-space weighting, 0.713 kg for the representative pick-and-place family, and 0.952 kg under the high-gravity-biased scenario. Full nondominated fronts further show that representative knees shift with declared design bounds. The all-joint rated-torque-referenced screen also indicates that compensation can expand quasi-static feasible task-space, while the lumped dynamic surrogate suggests that suficiently aggressive motion can reverse the preference. The general theory and numerical verification are reproducible; high-fidelity multibody reconstruction and paired hardware measurements remain the next step for physical validation.

![](images/e0a385bcf1a971eea4b0b6549ce7ee3b9395b756c2736703e948c197a32ef2d8.jpg)  
Figure 5: All-joint rated-torque-referenced quasi-static task-space feasibility versus payload. The motor classes are engineering references rather than validated replacements.

## CRediT authorship contribution statement

Mohammad Abbadi: Conceptualization, Methodology, Software, Formal analysis, Investigation of archived project records, Visualization, Writing – original draft, Writing – review and editing.

## Funding

This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

![](images/ca9997cddf992b51ce786467429d3828cc90c74dfb6db8fb7b4767129d6df845.jpg)  
Figure 6: Provisional payload–duration design-regime map from the lumped point-mass surrogate. D0: no counterweight; D1: historical source; D2: uniform-� design; D3: declared scenario reference.

## Data and code availability

The analysis code and derived numerical data supporting the findings are available from the corresponding author and are being prepared for public archival release. The submission package contains all source files required to reproduce the manuscript figures and numerical tables; no physical measurements absent from the historical archive are represented as new experimental data.

## Declaration of competing interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## References

[1] Vigen Arakelian. Gravity compensation in robotics. Advanced Robotics, 30(2):79–96, 2016. doi: 10.1080/01691864.2015.1090334.

[2] Vu Linh Nguyen. Passive gravity compensation in mechanisms and robots: Methods, advances, and challenges. Mechanism and Machine Theory, 220:106345, 2026. doi: 10.1016/j.mechmachtheory.2025. 106345.

[3] Yash J. Vyas, Volkert van der Wijk, and Silvio Cocuzza. A review of mechanical design approaches for balanced robotic manipulation. Robotics, 14(11):151, 2025. doi: 10.3390/robotics14110151.

[4] Alberto Martini, Marco Troncossi, and Alessandro Rivola. Algorithm for the static balancing of serial and parallel mechanisms combining counterweights and springs: Generation, assessment and ranking of efective design variants. Mechanism and Machine Theory, 137:336–354, 2019. doi: 10.1016/j.mechmachtheory.2019.03.031.

[5] Hubert Schneegans, Jan J. De Jong, Florent Cosandier, and Simon Henein. Mechanism balancing taxonomy. Mechanism and Machine Theory, 191:105518, 2024. doi: 10.1016/j.mechmachtheory.2023. 105518.

[6] Vu Linh Nguyen. A design approach for gravity compensators using planar four-bar mechanisms and a linear spring. Mechanism and Machine Theory, 172:104770, 2022. doi: 10.1016/j.mechmachtheory.202 2.104770.

[7] Chin-Hsing Kuo and Yi-Xin Wu. Perfect static balancing using cardan-gear spring mechanisms. Mechanism and Machine Theory, 181:105229, 2023. doi: 10.1016/j.mechmachtheory.2023.105229.

[8] Cheng-Hsuan Hsu, Chi-Shiun Jhuang, and Dar-Zen Chen. A novel slider-crank spring gravity balance module for 1-dof rotary link and its application to serial manipulators. Mechanism and Machine Theory, 209:105991, 2025. doi: 10.1016/j.mechmachtheory.2025.105991.

[9] Chin-Hsing Kuo. Complete gravity balancing of the general four-bar linkage using linear springs. Mechanism and Machine Theory, 214:106140, 2025. doi: 10.1016/j.mechmachtheory.2025.106140.

[10] Ke Shi, Jun Yang, Yao Tong, Zhimin Hou, and Haoyong Yu. Design and analysis of a cable-driven gravity compensation mechanism for spatial multi-dof robotic systems. Mechanism and Machine Theory, 190:105452, 2023. doi: 10.1016/j.mechmachtheory.2023.105452.

[11] Chia-Wei Juang, Chi-Shiun Jhuang, and Dar-Zen Chen. A novel spring gravity-balance method for spatial articulated manipulators without auxiliary links. Mechanism and Machine Theory, 191:105497, 2024. doi: 10.1016/j.mechmachtheory.2023.105497.

[12] Yijia Peng, Jinrong Deng, Jian Song, and Chaoqun Xiang. Design of a passive spatial 2-dof singularityconfigurable gravity compensator for forearm and shank with roll-pitch motion. Mechanism and Machine Theory, 217:106255, 2025. doi: 10.1016/j.mechmachtheory.2025.106255.

[13] Yiwei Wang, Peiji Chen, Shunta Togo, Hiroshi Yokoi, and Yinlai Jiang. A novel gravity compensation mechanism for orthogonal dofs with coupled springs. Mechanism and Machine Theory, 216:106220, 2025. doi: 10.1016/j.mechmachtheory.2025.106220.

[14] Vu Linh Nguyen. Nonlinear gear-spring design for gravity balancing of robotic manipulators with variable payloads: Methods and comparison. Journal ofMechanisms and Robotics, 17(11):111003, 2025. doi: 10.1115/1.4068878.

[15] Xiangxian Zeng, Chin-Hsing Kuo, and Emre Sariyildiz. Design optimization and validation of a permanent-magnet array for gravity compensation in long-stroke linear motion. Mechanism and Machine Theory, 209:105990, 2025. doi: 10.1016/j.mechmachtheory.2025.105990.

[16] Vu Linh Nguyen, Chin-Hsing Kuo, and Po Ting Lin. Performance analysis of gravity-balanced serial robotic manipulators under dynamic loads. Mechanism and Machine Theory, 191:105519, 2024. doi: 10.1016/j.mechmachtheory.2023.105519.

[17] J. Wang and C. M. Gosselin. Static balancing of spatial three-degree-of-freedom parallel mechanisms. Mechanism and Machine Theory, 34:437–452, 1999. doi: 10.1016/S0094-114X(98)00031-7.

[18] J. Wang and C. M. Gosselin. Static balancing of spatial four-degree-of-freedom parallel mechanisms. Mechanism and Machine Theory, 35:563–592, 2000. doi: 10.1016/S0094-114X(99)00029-4.

[19] Sunil K. Agrawal and Abbas Fattah. Gravity-balancing of spatial robotic manipulators. Mechanism and Machine Theory, 39(12):1331–1344, 2004. doi: 10.1016/j.mechmachtheory.2004.05.019.

[20] Andrea Russo, Rosario Sinatra, and Fengfeng Xi. Static balancing of parallel robots. Mechanism and Machine Theory, 40(2):191–202, 2005. doi: 10.1016/j.mechmachtheory.2004.06.011.

[21] Vu Linh Nguyen. A multi-objective optimal design method for gravity compensators with consideration of minimizing joint reaction forces. Journal ofMechanisms and Robotics, 16(8):084501, 2024. doi: 10.1115/1.4064236.

[22] Vu Linh Nguyen, Chin-Hsing Kuo, and Po Ting Lin. Reliability-based analysis and optimization of the gravity balancing performance of spring-articulated serial robots with uncertainties. Journal of Mechanisms and Robotics, 14(3):031016, 2022. doi: 10.1115/1.4053048.

[23] CubeMars. Ro60 kv115 frameless outrunner torque motor specifications. Manufacturer product data, 2026. URL https://www.cubemars.com/product/ro60-kv115-standard-with-hall-frame less-torque-motor.html. Accessed 9 September 2026.

[24] CubeMars. Ro80 kv105 frameless outrunner torque motor specifications. Manufacturer product data, 2026. URL https://www.cubemars.com/categorys/product. Accessed 9 September 2026.

[25] CubeMars. Ro100 kv55 frameless outrunner torque motor specifications. Manufacturer product data, 2026. URL https://store.cubemars.com/products/ro100. Accessed 9 September 2026.