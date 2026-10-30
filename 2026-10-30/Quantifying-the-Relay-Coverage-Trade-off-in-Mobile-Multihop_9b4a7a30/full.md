# Quantifying the Relay-Coverage Trade-off in Mobile Multihop Flooding

Johannes F. Loevenich<sup>∗†</sup>, Thies Mohlenhof ¨ <sup>∗</sup>, Roberto Rigolin F. Lopes<sup>∗</sup>

<sup>∗</sup>Thales Deutschland, cortAIx Labs, Germany

<sup>†</sup>Institute of Computer Science, University of Osnabruck, Germany¨

{johannes.loevenich, thies.moehlenhof, roberto.rigolin}@thalesgroup.com

Abstract—Network-wide flooding in mobile multihop networks selects relays from locally maintained topology information. Mobility makes this information outdated, leaving two basic responses: add redundant relays or refresh the topology and re-select relays more frequently. How these mechanisms trade reliability for overhead, however, is not well quantified. We quantify this trade-off on a public trace of 157 vehicles using Optimized Link State Routing version 2 (OLSRv2) and exact minimum-cost relay sets. Twofold coverage increases median relay-cover lifetime by only 2.3×, rather than about an order of magnitude as predicted by an independent-hazard model. The limitation is structural: 43.6 % of two-hop neighbours have only one possible relay, and newly appearing neighbours invalidate existing covers. Moreover, the delivery gain of a second relay falls from 0.061 to 0.016 when relay selection uses instantaneous link state, showing that redundancy mainly compensates for outdated topology information. Relay coverage and topology refresh therefore form a measurable redundancy–freshness trade-off rather than independent robustness mechanisms. In our trace, twofold coverage has lower overhead below about 100 bit/s of application flooding per node, whereas fresher topology information has lower overhead above it. This makes relay redundancy a load-dependent design choice rather than a fixed heuristic robustness setting.

Index Terms—mobile multihop networks, network-wide broadcast, relay selection, topology freshness, control overhead, mobility, redundancy

## I. INTRODUCTION

Network-wide broadcast supports route discovery and control dissemination in infrastructureless wireless networks, but naive flooding wastes capacity through redundant transmissions and collisions [1]. Relay-based schemes instead select one-hop neighbours that collectively reach the two-hop neighbourhood, as in Optimized Link State Routing (OLSR)/OLSR version 2 (OLSRv2) [2]–[4], Simplified Multicast Forwarding (SMF) [5], and dominating-set broadcast [6]. Mobility creates a common failure mode: the topology changes while the local information used to choose relays ages.

A protocol can tolerate this mismatch in two ways. It can add relay redundancy, spending forwarding capacity so that outdated choices remain useful, or maintain a fresher topology view, spending control capacity so obsolete choices are replaced sooner. OLSR exposes the former through MPR\_COVERAGE and explicitly suggests increasing coverage under mobility [3, §16]. Yet the exchange between these mechanisms, how much robustness a second relay buys, why, and when that cost is preferable to fresher topology information, has not been quantified on a realistic mobility trace (Section II).

The answer need not follow independent-backup intuition. Some two-hop neighbours have only one possible relay, new neighbours can invalidate an otherwise intact cover, and a router can continue to believe in a link after it breaks. Redundancy may therefore be limited by local topology and may act less as protection against instantaneous relay loss than as insurance against outdated information.

We measure this exchange on a public trace of 157 vehicles, using OLSRv2/Neighborhood Discovery Protocol (NHDP) as the primary instance because its neighbourhood state and forwarding semantics are explicit. An exact minimum-cost selector provides a yardstick against heuristic selector quality; SMF and collision arms test whether the mechanism survives a different relay rule and intra-flood contention.

The measurements show that twofold coverage extends median cover lifetime by only 2.3×, not by an order of magnitude. More importantly, most of its delivery gain disappears when relay selection receives current link state. Redundancy and topology freshness are therefore alternative ways of spending overhead against the same topology uncertainty, and application flooding load determines which is cheaper.

This paper makes three contributions:

• Trade-off. We quantify the reliability–overhead exchange between relay redundancy and topology freshness under mobility.

• Mechanism. We show that structural coverability limits redundancy and that its delivery gain mainly compensates for outdated topology belief.

• Design rule. We identify the application-load crossover at which fresher topology becomes cheaper than twofold relay coverage; the underlying mechanism also appears with SMF.

Together, these results turn relay redundancy from a fixed robustness heuristic into a measurable, load-dependent design choice for mobile multihop flooding.

## II. RELATED WORK

Relay flooding. Relay-based broadcast reduces the broadcast storm by restricting forwarding to a backbone. Multipoint relaying lets each node select forwarders from its one-hop neighbourhood [2], [4]; SMF’s Essential Connected Dominating Set (E-CDS) instead lets nodes self-elect from their two-hop views [5]; connected dominating sets and gossiping provide structural or probabilistic alternatives [6], [7]. These schemes differ in how relays are chosen, but all face the same mobility problem: forwarding decisions are based on a local view that can age before the next flood.

Redundant coverage. OLSR’s MPR\_COVERAGE= m asks that every two-hop neighbour be reached through m multipoint relays (MPRs) ${ } ^ { * } i f$ possible” and gives poorly covered nodes all available coverers [3]; OLSRv2 no longer exposes this parameter. Double coverage has been studied as protection against lossy links [8], while range buffers protect against outdated neighbour views [9]. Nilsson and Sterner compare additional MPRs with fading margins or hysteresis and find that the preferred robustness mechanism depends on topology density [10], [11]. This establishes that redundancy can help, but not how much temporal slack a second relay actually buys or when its recurring forwarding cost is preferable to a fresher view.

Stable selection and topology freshness. A complementary line tries to make a chosen backbone last longer by preferring stable links, predicting link lifetime, using mobility models, or retaining MPRs across runs [12]–[15]. These approaches choose better relays from the information currently available. Separately, HELLO intervals and hold times have been tuned to make that information more current [16]–[18]. The two lines therefore act on different sides of the same mismatch: one tolerates or avoids topology change, the other reduces uncertainty about it.

Gap. Prior work studies redundant coverage, longer-lived relay selection, and neighbourhood refresh largely as separate mechanisms. We instead measure their common exchange: how much outdated-topology error can be tolerated by additional relay coverage, how structural coverability limits that protection, and at what application load refreshing the topology becomes cheaper than forwarding through a redundant backbone.

## III. SYSTEM MODEL

The key distinction is between the true mobile topology and the local topology belief on which relay decisions are made. At time $t ,$ let $G _ { t } = ( V , E _ { t } )$ be the true graph, and let $N _ { 1 } ^ { t } ( v )$ and $N _ { 2 } ^ { t } ( v )$ denote node v’s one-hop and strict two-hop neighbours; members of $N _ { 2 } ^ { t } ( v )$ are terminals. Node v selects relays $R _ { t } ( v ) \subseteq N _ { 1 } ^ { t } ( v )$

Definition 1 (r-cover). $R _ { t } ( v )$ is an r-cover if every terminal $u \in N _ { 2 } ^ { t } ( v )$ is adjacent to at least r relays in $R _ { t } ( v )$ . Standard coverage has $r = 1$

Let $\widehat { G } _ { t } ( v )$ be v’s believed two-hop topology. Under perfect knowledge it equals the true local topology; under NHDP [19], periodic HELLOs update it and advertised entries persist for a hold time $T _ { \mathrm { h o l d } }$ . Relay sets are selected on $\widehat { G } _ { t } ( v )$ but used on $G _ { t }$ . Under OLSRv2 forwarding [4], a relay can act for a selector only after that selection has been advertised, so outdated links, outdated membership, and relay-flag lag can all affect delivery.

The two control levers spend different resources. Increasing r can increase the number of nodes that forward each application flood; decreasing the HELLO interval H or $T _ { \mathrm { h o l d } }$ increases control traffic but makes $\widehat { G } _ { t } ( v )$ track $G _ { t }$ more closely. We quantify the reliability–overhead exchange between redundancy and topology freshness and identify when each mechanism is the lower-cost choice.

## IV. THE REDUNDANCY–LIFETIME EXCHANGE

Before introducing topology-belief error, we derive the lifetime benefit expected if covering links fail independently and redundant cover is uniformly available. This gives a reference model for the measurements.

Proposition 1 (Coverage lifetime). Let covering links fail independently with hazard $\lambda ,$ and let a relay set reach $m _ { a }$ of the m terminals exactly a times. To first order in $\lambda L \ll 1$ , the time L by which some terminal is uncovered with probability ε satisfies

$$
\sum _ { a } m _ { a } ( \lambda L ) ^ { a } \approx \varepsilon .
$$

For a uniform r-cover relative to a 1-cover,

$$
\frac { L _ { r } } { L _ { 1 } } = \left( \frac { m } { \varepsilon } \right) ^ { 1 - 1 / r } .
$$

Proof. Fix the relay set at time $0 ;$ a covering link of terminal y joins $y$ to one of its $a _ { y }$ relays. Each covering link has failed by time L with probability $\begin{array} { r } { \dot { p } = 1 - e ^ { - \lambda L } } \end{array}$ , independently of all others. Terminal $y$ is uncovered by time L if and only if all its covering links have failed, an event $A _ { y }$ of probability $p ^ { a _ { y } }$ . Distinct terminals share no covering link, so the $A _ { y }$ are independent and some terminal is uncovered by time L with probability $\begin{array} { r } { F ( L ) = 1 - \prod _ { u } ( 1 - p ^ { a _ { y } } ) } \end{array}$ . Let $\begin{array} { r } { S = \sum _ { u } p ^ { a _ { y } } = } \end{array}$ $\textstyle \sum _ { a } m _ { a } p ^ { a }$ . The union bound gives $F \leq S ;$ the Bonferroni inequality and independence give $\begin{array} { r } { F \ge S - \sum _ { u < u ^ { \prime } } p ^ { a _ { y } } p ^ { a _ { y ^ { \prime } } } \ge } \end{array}$ $S \mathrm { ~ - ~ } S ^ { 2 } / 2$ . As $x - x ^ { 2 } / 2 \ \leq \ 1 - e ^ { - x } \ \leq \ x ^ { \sigma }$ for $x \ge 0 .$ $\lambda L ( 1 - \lambda L / 2 ) \leq p \leq \lambda L$ . Let L solve $\begin{array} { r } { \sum _ { a } m _ { a } ( \lambda L ) ^ { a } \ : = \varepsilon } \end{array}$ and a¯ = max $\{ a : m _ { a } > 0 \}$ . Then $\varepsilon ( 1 - \lambda { \bar { L } } / 2 ) ^ { \bar { a } } \leq S \leq \varepsilon .$ and since $S - S ^ { 2 } / 2$ increases on $[ 0 , 1 ] ,$

$$
\varepsilon \Big ( 1 - \frac { \lambda L } { 2 } \Big ) ^ { \bar { a } } \Big ( 1 - \frac { \varepsilon } { 2 } \Big ) \leq F ( L ) \leq \varepsilon .
$$

Hence $F ( L ) = \varepsilon { \bigl ( } 1 - O ( { \bar { a } } \lambda L + \varepsilon ) { \bigr ) }$ : to first order, $L$ is the time by which some terminal is uncovered with probability $\varepsilon ,$ and it never overstates that time. The upper bound needs no independence across terminals. For a uniform r-cover, $m _ { r } =$ m gives $\lambda L _ { r } = ( \varepsilon / m ) ^ { 1 / r } ; \mathtt { a }$ 1-cover reaching every terminal once gives $\lambda L _ { 1 } = \varepsilon / m$ . Dividing, $L _ { r } / L _ { 1 } \stackrel {  } { = } ( \varepsilon / \stackrel { \cdot } { m } ) ^ { 1 / r - 1 } =$ $( m / \varepsilon ) ^ { 1 - 1 / r }$ □

Uniform redundancy need not exist. For terminal $y ,$ define its number of possible coverers

$$
\kappa ( y ) = | \{ w \in N _ { 1 } ( v ) : y \in N _ { 1 } ( w ) \} | ,
$$

and cap achievable coverage at $r _ { y } = \operatorname* { m i n } ( r , \kappa ( y ) )$ . A terminal with $\begin{array} { l l l } { \kappa ( y ) } & { = } & { 1 } \end{array}$ is pinned: no relay budget can give it redundancy.

Corollary 1 (Structural cap). For a capped r-cover with $r \geq$ 2, let $m _ { \pi } > 0$ be the number of pinned terminals. The firstorder lifetime gain over a 1-cover is bounded by $m / m _ { \pi } ;$ additional coverage elsewhere cannot remove this cap while pinned terminals remain.

The reference law is deliberately optimistic: if redundancy were uniformly available and covering-link losses independent, a second relay should buy roughly an order of magnitude of lifetime at the neighbourhood sizes observed here. The capped model predicts why this need not happen: a single pinned terminal restores a first-order failure mode that extra relays elsewhere cannot remove. The size of the gap between these two cases is therefore itself diagnostic. E1 tests whether mobility behaves closer to the uniform backup model or to the structural cap; E2 then asks whether the remaining delivery value of redundancy comes from tolerating an outdated topology belief rather than extending the physical lifetime of the relay set.

## V. EXPERIMENTAL DESIGN

The experiments isolate three questions: E1 measures the intrinsic lifetime bought by redundancy on the true topology; E2 identifies what redundancy protects against under NHDP belief; E3 prices redundancy against fresher topology at matched delivery.

a) Trace and baseline: We use the public Anglova Vignette-2 trace [20], [21]: 157 vehicles over 7800 s at 1 Hz, with staging, road-march, and assembly phases over the same terrain. Links use a 120 dB pathloss threshold; the largest component contains 94.7 % of nodes on average (mean degree 31). Unless stated otherwise, results use the road march, where mobility is highest. The first 10 s are discarded. Baseline NHDP sends HELLOs every 2 s at staggered phases and expires entries after 6 s; floods run on the true graph with Request for Comments (RFC) 7181 forwarding [4].

b) Relay rules: All selectors use the same two-hop view and capped coverage min(r, κ). The greedy selector applies weighted-set-cover cost effectiveness [22]; the exact selector solves the minimum-cost capped cover with constraint programming–satisfiability (CP-SAT). A robustness arm uses the RFC 7181 Appendix-B ordering with the RFC 3626 coverage extension. A node re-evaluates every T seconds and otherwise holds its incumbent; at evaluation it keeps a stillvalid set or re-selects a broken one. T = 1 s is the conformant point on the 1-Hz trace; larger T deliberately measures slack. Blind flooding (all reached nodes forward) is a transmission reference.

c) E1: cover lifetime: E1 isolates the intrinsic temporal value of redundancy from topology-belief error. Every 30 s, a fresh capped r-cover is constructed on the true topology and then frozen while the topology evolves. The outcome is time until some current terminal loses all relays; departing terminals and lifetimes beyond 600 s are censored. Varying only $r \in \{ 1 , 2 , 3 \}$ therefore asks: if the decision were initially correct, how much longer would an additional relay keep it valid under mobility?

d) E2: belief and delivery: E2 reintroduces the information mismatch. Selectors run for $r ~ \in ~ \{ 1 , 2 , 3 \}$ and $T = 1 \dots 3 0 0 { \mathrm { s } }$ under NHDP belief and perfect knowledge; every node originates a flood every 10 s, and delivery is the fraction of the source’s true component reached. Link and membership oracles selectively correct outdated belief while retaining relay-flag delay. The distinguishing prediction is simple: if redundancy mainly protects against physical relay loss, its gain should remain when the topology view is corrected; if it mainly protects against outdated information, that gain should collapse. HELLO loss, link hysteresis, and a denser 130 dB graph test sensitivity. SMF/E-CDS [5, App. A] with RFC 5614 backup relays [23] tests a self-elected relay rule. A collision arm adds up to 0.5 s forwarding jitter and loses a reception when another neighbour transmits within one packet airtime, with or without carrier sense; floods are simulated in isolation.

e) E3: price: E3 treats the two mechanisms exposed by E2 as substitutes for a common reliability target. Redundancy pays additional forwarding cost on every application flood; fresher topology pays mainly additional periodic control cost. We therefore compare configurations at matched delivery and vary application flood rate to identify where the loweroverhead choice changes. Costs are transmitted bytes per node-minute: periodic HELLO and Topology Control (TC) traffic plus application-flood transmissions. The sensitivity analysis spans 48 accounting variants (message layout, Internet Protocol (IP) headers, routing-MPR choice, and relaychange signalling); responsive signalling is charged but not credited with faster propagation. Selectors are deterministic; 95 % confidence intervals (CIs) use moving-block bootstrap over the timestep series (300-s blocks, paired indices; 500– 2000 resamples), with cover-lifetime intervals bootstrapped over anchors [24].

The experiments thus form a progression rather than three benchmarks: E1 establishes the effect, E2 identifies its mechanism, and E3 converts that mechanism into a network-design decision at matched reliability.

## VI. RESULTS: THE REDUNDANCY–FRESHNESS EXCHANGE

The results answer the three questions in order: how much time redundancy buys, what its delivery gain protects against, and when redundancy is cheaper than fresher topology.

## A. E1: How much time does redundancy buy?

a) Independence is optimistic: For terminals with $\kappa \geq$ 3, a second relay follows Proposition 1 within 13 % for small failure probabilities, but losses are correlated: at 60 s the independence prediction overstates survival by 0.17 for both two and three relays (Fig. 1b).

b) Neighbourhood lifetime gains only 2.3×: On the road march, 43.6 % of terminals are pinned and a uniform 2-cover exists in only 6.5 % of neighbourhoods. Median cover lifetime rises from 2.6 to 6.0 s, a factor 2.3 (95 % CI 2.2–2.5), versus roughly 11× from the uniform reference model (Fig. 1a).

a = 3 independence  
(a)  
![](images/1ea8b100613acc2fd448b568f9bd6ebe49fae361febadd0f494f093f6c2b102a.jpg)

(b)  
![](images/ba7a81553fd8e78fc1886ec13100fbd1ec483182469478a94eb089ab427eef5b.jpg)

(c)  
![](images/1995e45a235fe19c1c5f4ff35502f936a44981fd9683a1e90515af1c35474a81.jpg)

(d)  
![](images/3b5eb27e87740868a0103a9cb42e7978adae71c0c9beef4103311883449e594d.jpg)  
NHDP r= 1 perfect r= 1  
Fig. 1. Lifetime and delivery exchange on the road march. (a) Survival of fresh capped covers. (b) Per-terminal survival versus independent-cover prediction. (c) Delivery versus relay-flag churn under NHDP and perfect knowledge; rings mark T = 1 s. (d) Configurations reaching delivery ≥ 0.95; blind means all reached nodes forward. Bars/bands: 95 % CIs.

Pinning and newly arriving terminals jointly explain much of the gap: arrivals cause 40–53 % of first failures, and censoring both causes raises the factor to 6.7 (95 % CI 3.7–18.7). A third relay adds only 1.4× over the 2-cover, unresolved by the CIs. Thus redundancy buys time, but structural coverability and topology arrivals prevent the order-of-magnitude gain of uniform independent backup.

c) The structural cap is not just link flicker: The coverability distribution in Fig. 3b reinforces this interpretation. The staging phase is the densest of the three mission phases yet has the largest pinned share, so simply adding neighbours does not guarantee additional relay choices. Among pinned terminal pairs that remain two hops apart one second later, 92.5 % remain pinned. Pinning is therefore mostly a persistent geometric constraint at this timescale rather than a one-snapshot threshold effect. Consistently, the measured lifetime factor falls from 3.9× in neighbourhoods without a pinned terminal to 1.2× where most terminals are pinned, although those strata also differ in neighbourhood size. The evidence supports the structural mechanism without claiming that pinning alone explains every failure.

## B. E2: What does redundancy protect against?

a) Single-cover re-selection reaches a belief-limited ceiling: With perfect knowledge, a greedy cover re-evaluated each second delivers every flood (Fig. 1c). Under NHDP belief, the conformant T = 1 s single cover reaches only 0.893 (95 % CI 0.845–0.928); the exact single cover reaches 0.865. A capped second cover adds +0.061 (paired 95 % CI 0.041–0.088) to greedy delivery and +0.078 to exact. Reselecting faster therefore cannot remove the ceiling created by the information on which selection operates.

b) The second relay mostly insures outdated links: On the road march, 14.5 % of believed covering links have already broken. Correcting the belief removes most of both the single-cover deficit and the redundancy gain (Fig. 2a): with current links, single-cover delivery rises to 0.960 and the second-relay gain falls from +0.061 to +0.016. Boundary misses show that the second relay mainly removes terminals believed covered over broken links, not terminals absent from the belief (Fig. 2b). Across hold-time and hysteresis interventions, the gain increases with the share of outdated links (Fig. 2c). The residual +0.016 may reflect other delays; relayflag lag is one candidate. The central comparison is therefore redundancy versus topology freshness, not redundancy versus re-selection.

(a)  
![](images/7ba4e6e7b2f291adbd726efd421b8ba5a5c4cab4d2e5d1b9604c4419e330912f.jpg)

(b)  
![](images/8c85510a98952c3f8fe495431ea7bde28cac1ea2ff251a89d2b0a9266bc15bc4.jpg)

![](images/bfb9e5c28df57d19cdca0af61a0aa36032bb00adcd3f9eed568b105f354b4b6f.jpg)

(d)  
![](images/d1fa1e8e09d8d69358bd1eaed83b7442f46e3f0f61bbdf88aeebee2d0fe06acf.jpg)  
Fig. 2. Redundancy mostly insures against outdated topology belief (road march, greedy, $T = 1 { \mathrm { s } } ) .$ (a) Delivery as belief is corrected. (b) Boundary misses by cause. (c) Second-relay gain versus outdated covering links. (d) Single-cover delivery versus hold time; band: default exact 2-cover.

The oracle sequence also distinguishes outdated links from missing topology. Correcting current links removes most of the benefit, while terminals absent from the belief remain unreachable by either one or two relays because no selector can cover a terminal it does not know exists. The membership oracle behaves similarly because it also removes part of the outdated-link error. Thus the second relay is not creating new topology knowledge; it is increasing the chance that at least one path assumed by the existing belief still exists when the flood occurs.

(a)  
![](images/87e3bee2d28df1084bc6ddfd264bd1d5509698e41b7de4431fc44d8e8601d546.jpg)

(b)  
![](images/f662b74015b4e8403063d256fe10af2ec7f5d2cf1b2afda816718af9024b3d89.jpg)

![](images/74e794951354e900ea5c4c620896da1e59f9f88450efcdb1e940b102ebd4ae66.jpg)

(d)  
![](images/e5a755aa358efbacc7ce8889d096fbb776b29197aa823b1febdd4b493b5da2d6.jpg)  
Fig. 3. Robustness and price. (a) Second-relay gain across selector, belief, E-CDS, and collision variants. (b) Terminal coverability κ. (c) Delivery versus control bytes. (d) Twofold/single-cover byte ratio at matched delivery under 48 accounting variants and three application loads.

## C. E3: When is redundancy cheaper than freshness?

Reducing H HOLD TIME partially substitutes for redundancy (Fig. 2d), but the timer cannot be shortened arbitrarily when HELLOs may be lost: RFC 6130 recommends a hold time significantly larger than the refresh interval [19]. With $2 \mathrm { { - s } \ H E L L O s , }$ a 3–4-s hold recovers less than half of the second relay’s gain; the control-byte increase is only a few percent without HELLO loss but grows to roughly onequarter to one-half under 10 % loss. With 1-s HELLOs and a 3-s hold, preserving the default 3:1 hold/HELLO ratio, a single cover matches the default exact 2-cover’s delivery at 1.32× its control bytes. Faster HELLOs alone do not help because outdated entries persist until their hold timers expire. Freshness is therefore a joint timer choice rather than simply “send HELLOs faster.

a) The mechanism survives the tested variants: The second-relay gain remains positive across all tested belief variants and shrinks as the share of outdated links falls (Fig. 3a). With ±3 dB hysteresis, outdated believed links fall from 14.5 to 5.9 % and the gain from +0.061 to +0.037. The mechanism is not specific to OLSRv2 in these experiments: E-CDS delivers 0.895 under NHDP belief and its backup layer adds +0.083 (95 % CI 0.057–0.120), but no gain on the true graph. Under our isolated-flood collision model, the gain is essentially unchanged at 160 kbit/s and above; without carrier sense it falls to +0.024 at 100 ms airtime and reverses at 250 ms.

b) Application load sets the crossover: At $T \ = \ 1 { \mathrm { s } }$ twofold coverage roughly doubles flood transmissions: exact uses 16.2 versus 8.3 transmissions per flood (greedy: 20.3 versus 10.6). This creates different scaling laws for the two remedies. The redundant backbone pays almost nothing when no application flood is sent, but pays its extra forwarders again for every subsequent flood. A fresher topology configuration instead raises periodic control traffic whether or not the application floods, so its marginal cost per additional flood is small. Application load should therefore create a crossover even when mobility and the target delivery are fixed.

That crossover appears across the accounting variants in Fig. 3d. At the conformant comparison, the default exact 2- cover crosses a single cover with 1-s HELLOs and a 3-s hold at 3.7 floods of 200 B per node-minute (95 % CI 2.9–4.8), about 100 bit/s per node. Below this load, twofold coverage has lower total overhead at matched delivery because its forwarding premium is rarely paid; above it, the fixed cost of fresher topology is amortized across enough floods to become cheaper. The numerical crossover belongs to this trace and accounting, but the opposing fixed-versus-per-flood cost structure explains why a load-dependent decision arises. The accounting sensitivity shows that the decision is not an artifact of one packet layout. With no application floods, reselection/freshness is typically cheaper when relay changes ride on periodic HELLOs, while more expensive responsive signalling can make the held redundant backbone competitive. Once application traffic is included, the repeated forwarding cost dominates: at one 200-B flood per node every 10 s, the held 2-cover is more expensive in 89 of the 96 matcheddelivery accounting cells, and at one every 2 s it is more expensive in all of them. The precise boundary moves, but the direction of the load effect is stable.

Together, E1–E3 establish the exchange: structural topology caps the lifetime benefit, outdated topology belief explains most of the delivery gain, and application load determines whether tolerating outdated belief with redundancy or correcting it with fresher updates costs less network capacity.

c) Implication for OLSRv2: The measured exchange suggests a simple implementation rule rather than a new message format. Under light application flooding, request capped twofold coverage, i.e., cover each terminal y through min(2, κ(y)) neighbours; the existing neighbour lists already reveal κ(y), and additional relay flags can ride in the periodic HELLO. Under heavier flooding, keep single coverage and spend the budget on a fresher neighbourhood view, represented here by the 1-s HELLO/3-s hold configuration. The rule is deliberately load-dependent: it does not claim that twofold coverage or faster refresh is universally preferable, only that the cheaper way to reach the same reliability changes with how often the forwarding backbone is exercised.

d) Transfer beyond the measured instance: The OL-SRv2 timer and coverage settings are protocol-specific, but the quantities exposed by the experiments provide a more general way to analyse local-view relay schemes. A protocol designer can ask three questions in the same order: how many alternative relays each destination structurally admits, how outdated the local information is when forwarding decisions are exercised, and whether robustness is paid periodically in control traffic or repeatedly per application flood. E-CDS reproduces the belief-insurance mechanism without OLSRv2’s selector/flag semantics, suggesting that the decomposition is not tied to one forwarding rule. We do not claim that another protocol shares the same crossover; rather, the result identifies the measurements needed to locate its own redundancy– freshness operating point.

e) Scope and limitations: The collision arm models contention only among transmissions of the same flood; concurrent floods and interference from HELLO or TC traffic are outside the model. E-CDS and the belief interventions support the proposed mechanism beyond one selector, but do not establish it for every flooding protocol or mobility regime. What the experiments support more broadly is the decomposition itself: redundancy is constrained by local coverability, can mask outdated topology information, and incurs a perflood cost that competes with the largely periodic cost of maintaining a fresher view.

## VII. CONCLUSION

Relays chosen on HELLO-derived belief expire in two ways: the topology moves, and the belief lags it. Re-selection repairs the first; only redundancy or a fresher belief repairs the second. On this trace, a second relay extends median cover lifetime by 2.3×, not by an order of magnitude: 43.6 % of two-hop neighbours admit no second relay, and newly appearing two-hop neighbours break covers too. A single cover re-selected as fast as the trace allows tops out at delivery 0.893, while capped twofold coverage adds +0.061 for about twice the forwarding transmissions. With current link state, that gain falls to +0.016, showing that redundancy mainly insures against outdated topology belief.

Two parts of this explanation are structural: redundancy is capped wherever a two-hop neighbour has no alternative relay, and a neighbourhood belief retained beyond its observation interval can become outdated under mobility. In our experiments the same redundancy effect also appears with SMF’s E-CDS, so it is not specific to OLSRv2 forwarding. It also survives intra-flood collisions at 160 kbit/s and above under the collision model evaluated here. The numerical gains and crossover remain specific to the measured trace and protocol configuration.

For OLSRv2, the resulting rule is load-dependent. Where an application requires more delivery than single coverage provides, capped twofold coverage and fresher topology information offer two conformant ways to obtain it. Under light application flooding, below about 3.7 floods of 200 B per nodeminute $( \approx 1 0 0 \mathrm { b i t / s }$ per node), request $\operatorname* { m i n } ( 2 , \kappa ( y ) )$ flooding relays for each two-hop neighbour y. This corresponds to OL-SRv1’s MPR\_COVERAGE = 2 semantics, requires no change to the OLSRv2 message format, and incurs its additional cost primarily when floods are forwarded. Above that load, use single coverage with a 1 s HELLO interval and 3 s hold time instead: the additional control cost is then cheaper than paying the redundant forwarding cost on every flood. Relay coverage should therefore be selected jointly with topology freshness and application load rather than fixed as a static robustness parameter.

## REFERENCES

[1] S.-Y. Ni, Y.-C. Tseng, Y.-S. Chen, and J.-P. Sheu, “The broadcast storm problem in a mobile ad hoc network,” in Proc. ACM/IEEE MobiCom, 1999, pp. 151–162.

[2] A. Qayyum, L. Viennot, and A. Laouiti, “Multipoint relaying for flooding broadcast messages in mobile wireless networks,” in Proc HICSS, 2002, pp. 3866–3875.

[3] T. Clausen and P. Jacquet, “Optimized link state routing protocol (OLSR),” IETF, RFC 3626, Oct. 2003.

[4] T. Clausen, C. Dearlove, P. Jacquet, and U. Herberg, “The optimized link state routing protocol version 2,” IETF, RFC 7181, Apr. 2014.

[5] J. Macker, “Simplified multicast forwarding,” IETF, RFC 6621, May 2012.

[6] F. Dai and J. Wu, “On constructing k-connected k-dominating set in wireless ad hoc and sensor networks,” J. Parallel Distrib. Comput., vol. 66, no. 7, pp. 947–958, 2006.

[7] Z. J. Haas, J. Y. Halpern, and L. Li, “Gossip-based ad hoc routing,” IEEE/ACM Trans. Netw., vol. 14, no. 3, pp. 479–491, 2006.

[8] W. Lou and J. Wu, “Toward broadcast reliability in mobile ad hoc networks with double coverage,” IEEE Trans. Mobile Comput., vol. 6, no. 2, pp. 148–163, 2007.

[9] J. Wu and F. Dai, “Efficient broadcasting with guaranteed coverage in mobile ad hoc networks,” IEEE Trans. Mobile Comput., vol. 4, no. 3, pp. 259–270, 2005.

[10] J. Nilsson and U. Sterner, “Robust MPR-based flooding in mobile ad-hoc networks,” in Proc. IEEE MILCOM, 2012, pp. 1–6, doi: 10.1109/MILCOM.2012.6415631.

[11] ——, “Performance of MPR-based broadcast methods in mobile ad hoc networks for rural terrains,” in Proc. ICMCIS, 2015, pp. 1–7.

[12] S. Obilisetty, A. Jasti, and R. Pendse, “Link stability based enhancements to OLSR (LS-OLSR),” in Proc. IEEE VTC-Fall, 2005, pp. 306– 310.

[13] A. Ouacha, N. Lakki, J. El Abbadi, A. Habbani, B. Bouamoud, and M. Elkoutbi, “Reliable MPR selection based on link lifetime-prediction method,” in Proc. IEEE ICNSC, 2013, pp. 11–16.

[14] Z. Li and Y. Wu, “Smooth mobility and link reliability-based optimized link state routing scheme for MANETs,” IEEE Commun. Lett., vol. 21, no. 7, pp. 1529–1532, 2017.

[15] L. Maccari and R. Lo Cigno, “How to reduce and stabilize MPR sets in OLSR networks,” in Proc. IEEE WiMob, 2012, pp. 373–380.

[16] Y. Huang, S. N. Bhatti, and D. Parker, “Tuning OLSR,” in Proc. IEEE PIMRC, 2006, pp. 1–5.

[17] R. Ernst and P. Martini, “Adaptive HELLO for the neighborhood discovery protocol,” in Proc. IEEE LCN, 2012, pp. 470–478.

[18] F. Jovel, J. McCartney, P. J. Teller, E. Ruiz, and M. P. McGarry, “Neighbor discovery message hold times for MANETs,” Comput. Commun., vol. 112, pp. 38–46, 2017.

[19] T. Clausen, C. Dearlove, and J. Dean, “Mobile ad hoc network (MANET) neighborhood discovery protocol (NHDP),” IETF, RFC 6130, Apr. 2011.

[20] N. Suri et al., “A realistic military scenario and emulation environment for experimenting with tactical communications and heterogeneous networks,” in Proc. ICMCIS, 2016, pp. 1–8.

[21] ——, “The Anglova tactical military scenario and experimentation environment,” in 2018 International Conference on Military Communications and Information Systems (ICMCIS), 2018, pp. 1–8.

[22] V. Chvatal, “A greedy heuristic for the set-covering problem,” ´ Math Oper. Res., vol. 4, no. 3, pp. 233–235, 1979.

[23] R. Ogier and P. Spagnolo, “Mobile ad hoc network (MANET) extension of OSPF using connected dominating set (CDS) flooding,” IETF, RFC 5614, Aug. 2009.

[24] H. R. Kunsch, “The jackknife and the bootstrap for general stationary¨ observations,” Ann. Statist., vol. 17, no. 3, pp. 1217–1241, 1989.