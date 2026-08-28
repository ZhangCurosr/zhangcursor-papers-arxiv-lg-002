# Ultra Low-Power, Lightweight, Probabilistic RSS-Based Path Reconstruction: A System for Landscape-Scale Bee Tracking

Christopher J. Noroozi, Joseph L. Woodgate, Michael Mangan and Michael T. Smith

Abstract—Applications in fields such as movement ecology, Internet of Things or robotics share the need for systems that localize devices that are too small and power constrained to implement GNSS (Global Navigation Satellite Systems). Alternative low-power localization methods often rely on only measurements of RSS (Received Signal Strength) to infer the AoA (Angle of Arrival) of a transmitted radio frequency signal, but are limited by range and the power demand of the large number of RSS measurements required to infer an accurate AoA. In this paper we address these issues with a novel RSS-based method for tracking ultra lightweight and low-power moving receivers across a complex landscape, achieved by using a minimal number of RSS measurements from simple rotating high-gain transmitters with a range of 300m, and applying probabilistic modelling to infer their AoA. The receiver’s movement path is then modelled using a Gaussian process and reconstructed using doubly stochastic variational inference, resulting in ∼15m accuracy tracking of receivers weighing 38mg (including power source) over a scalable landscape range while consuming <180µW, increased to ∼10m accuracy at <600µW by taking more RSS measurements. We anticipate that this method will support fields such as the behavioural study of flying insect species, which we demonstrate by applying the system to track Bombus terrestris nest return flights.

Index Terms—Direction of arrival estimation, RSSI, Biotelemetry, Gaussian processes, Localization, Bayes methods

## I. INTRODUCTION

N several fields, the ability to continuously measure the position of very lightweight devices over large areas is of great importance. In movement ecology, automated tracking of an animal’s position over time has been key in advancing knowledge of fundamental aspects of animal behaviour [1] and informing conservation efforts [2]. Tagged tracking methods are often required for landscape-scale tracking, and also show promise in the implementation of biologging sensors to infer behaviour along with position [3]. To enable the study of very small animals such as insects, these tags must be extremely lightweight and low-power. The ubiquity of personal mobile devices has driven widespread research in the use of lowenergy, lightweight Internet of Things (IoT) sensor networks [4] requiring device localization to deliver services to users, including navigation and medical services [5]. Emerging fields such as smart dust or swarm robotics also demand novel localization of very small devices [6], [7]. All of these applications share a common instrumentation problem: measuring position continuously and accurately under constraints of minimal mass and power consumption.

The tracking of such devices is most commonly performed using GNSS (Global Navigation Satellite Systems). This however requires substantial hardware and battery capacity. Grenier et al. [8] concludes that GNSS is often the most power consuming sensor in embedded systems, with the smallest GNSS units commercially available such as the U-Blox UBX-M10150-CC [9] requiring 8mW to function. In highly hardware constrained tracking applications where minimal power consumption is key, the high power demand and added weight of a sufficient enough battery for GNSS to function precludes tracking of smaller subjects such as the majority of flying insect species [10]. Many applications also require tracking in situations where GNSS is not available such as indoor tracking of mobile devices [11], or underground localization of UAVs which faces both an absence of GNSS availability and strict power constraints [12].

Existing methods for automated, landscape-scale tracking of flying insects have relied on either Very High Frequency transceivers [13] or harmonic radar [14], [15]. The former requires a relatively heavy tag (>100mg), causing energetic costs on the animal [16] and precluding the tracking of widely studied species such as Bombus terrestris which Stelzer et al. [17] show carry up to 70mg of weight per foraging bout. The latter is prohibitively expensive and struggles in complex environments. Dressler et al. [18] note that since most uses of sensor technology for wildlife monitoring are battery powered, energy consumption is a key consideration. A system that is suitable for tracking such subjects therefore must consume minimal power and use tags weighing < 70mg.

Received signal strength (RSS) based localization methods are an ideal alternative for miniaturised systems due to the minimal signal processing and hardware overhead required to take RSS measurements [19], [20]. Distance-free RSS methods exploit the anisotropy of a directional antenna’s radiation pattern (ARP) to infer the angle of arrival (AoA) of a transmitted signal. Several bearing observations from spatially separated transceivers can then be used to triangulate a device or reconstruct the path it took. However, as reviewed in Section II, existing RSS-based methods either require a prohibitively large number of RSS measurements per inferred AoA - incurring energy costs incompatible with ultra low-power receivers - or are limited in range, preventing deployment at landscape scale.

To address these measurement limitations, we present a novel RSS-based, distance-free method of tracking a moving, low-power, lightweight receiver across a complex, unknown landscape. By using a sparse duty cycle to receive short bursts of Bluetooth Low-Energy (BLE) packets from rotating highgain transmitters placed across the landscape, the receiver requires far less power than previous RSS-based AoA methods. Rather than seeking the peak or null of the ARP over a full rotation, which requires dense RSS measurement, we instead present and evaluate three methods to probabilistically model and infer the AoA of the received packets based on their change in RSS over time. We then perform a form of ‘probabilistic triangulation’ from these non-concurrent, uncertain AoA predictions to reconstruct the receiver’s full movement path. We do this by modelling the receiver’s path using a Gaussian process (GP), and perform inference via doubly stochastic variational inference, providing an expression of the path’s uncertainty over time.

The result is a 38mg (including power source) archival receiver, whose path at a landscape scale can be reconstructed using packets received from simple rotating transmitters each with a range of 300m, with a positional accuracy of ∼15m mean absolute error while consuming <180µW, improving to ∼10m at $< 6 0 0 \mu \mathrm { W } .$

We validate the ability of the system to track the receiver by applying it in several ground truthed localization tasks, evaluating AoA & path reconstruction accuracy and the effect of varying the number of RSS measurements per AoA inferred. We also demonstrate a use case of the system in movement ecology by deploying it in a field trial to track the return flight of a foraging Bombus terrestris (buff-tailed bumblebee).

This paper is organized as follows. In Section II the state of the art in the field of RSS-based localization is explored. In Section III the novel hardware for our proposed ultra lowpower, lightweight tracking system is described. Section IV presents three approaches for probabilistically inferring an AoA based on received RSS measurements, and Section V describes the path reconstruction algorithm that uses these AoAs to infer the receiver’s path. Section VI outlines the experimental set-up for validating the performance of the system, the results of which are presented in Section VII. In Section VIII the conclusions are drawn.

## II. RELATED WORK

To explore alternatives to GNSS, Hayward et al. [21] define four groups of localization methods: acoustic, inertial, visual and radio frequency (RF) based (although outside this taxonomy, other methods do exist such as using magnetic fields [22]). Of these, RF based approaches are the most commonly used.

Localization via RF has been achieved using: the phase difference between signals received by multiple antenna to obtain the Angle of Arrival (AoA) of a signal [23], the difference in time of arrival (ToA) between multiple received signals [24], measurements of received signal strength (RSS) or a hybrid of these methods [25]–[27]. ToA requires precise timing & synchronization, and accessing the phase difference of received signals requires additional & complex hardware beyond a single antenna [20]. Instead, purely RSS based localization is ideal for miniaturized low-power systems due to the relatively low complexity of signal processing and calibration required [19].

Distance-based methods of localization using RSS attempt to infer the relation between RSS and the transmitter-receiver distance using a path loss model [28], and therefore find the distance from several transceivers and trilaterate the position of a device [29]. However a key challenge of using RSS measurements to infer distance is the effect of other factors such as antenna orientation or the significant noise induced by cluttered environments and multipath fading [30]. These severely impact the reliability of distance estimates. Fingerprinting techniques have improved distance based localization by first learning the RSS at a number of known locations from several static beacons [31]. RSS measurements at the device to be localized, either a single vector of measurements or a series of measurements over time [32], are then compared with these “fingerprints” via clustering [33], deep learning [34], filtering [35] or other methods to infer the device’s location. This method has resulted in high-accuracy localization, but learning the required fingerprints leads to a poor ability to generalize and scale for unknown and changing environments without laborious recalibration, which much research aims to streamline [36].

Distance-free methods of RSS-based localization exploit the anisotropy of a non-isotropic antenna’s RF radiation pattern (ARP) coupled with a device’s movement to infer the relative AoA, sometimes also known as Direction of Arrival, of a transmitted signal [37]. Multiple AoAs, essentially bearing observations, from different transceivers can then be used to either triangulate a device [38] or reconstruct a full trajectory via filtering and smoothing frameworks, such as pseudo-linear or Kalman filter-based approaches [39]–[41].

These methods often seek the null [42] or peak [43] of an ARP facing a known angle to infer the AoA. As RSS varies with AoA and distance, this requires very frequent measuring at the device to capture enough of the ARP to find the RSS minima or maxima respectively, incurring significant energy cost that is not feasible for a low-power device.

To infer AoA with fewer RSS samples, other methods have used a form of monopulse in which RSS measurements are taken from multiple antenna with differently oriented ARPs at the same location [44]. The difference between measurements of RSS from these antenna is then compared with the known difference in gain across their ARPs to infer a relative AoA [45]. Many such methods [46] have seen success in lowpower, lightweight device localization applications, especially in animal tracking, such as the triangulation of bats in the wild [47]. The system was able to perform high resolution tracking in a relatively small area of a few hectares. Extensions to such approaches have lead to inference of distance alongside AoA [48], effectively allowing for localisation from a single node consisting of only three antenna, or improvements in AoA accuracy to within 1<sup>◦</sup> [49]. Common across all these methods is low gain of the antenna used, leading to a poor ability to scale across large areas due to the prohibitively large number of antennas needed to cover a large portion of the landscape and the reduced angular precision from their wide half-power beamwidth (HPBW). The use of high-gain yagi antenna have resulted in greater transmitter range in a study tracking monarch butterflies [50]. However, whilst highgain antennas extend range, their narrow HPBW means that achieving full $3 6 0 ^ { \circ }$ coverage from a single node would require a prohibitively large number of antenna elements. Fisher et al. [50] mentions that 9 receiving stations - each using 4 antennas resulting in 36 antennas total - would be required to cover an area of 500m × 500m using their method, limiting the practicality of landscape-scale deployment.

To overcome this Varotto et al. [51] identifies the need for motorized directional antennas, used in fingerprinting approaches [52] and robotics tracking applications in which the orientation of the ARP indicates the AoA the tracked target lies upon. These methods once again rely on capturing a large number of RSS measurements to find the peak RSS from an ARP’s main lobe [53] (200 RSS measurements per antenna rotation) or reconstruct a portion of the antenna’s ARP [54] (RSS measurement every $1 1 ^ { \circ } - 1 6 ^ { \circ }$ of antenna rotation).

All of the above RSS based tracking methods either perform localization using too many RSS measurements to be feasible for an ultra low-power receiver, or struggle to scale across a larger landscape scale due to low-range or cumbersome antenna.

The system we present in this paper addresses these issues by performing AoA inference with a burst of as few as 3 RSS measurements, received from simple rotating high-gain antenna with a range of 300m, sufficient to cover a landscape scale. This is accomplished by probabilistically modelling the change in measured RSS across the burst. We then model the receiver’s movement path from several of these AoA predictions using a Gaussian process, and perform inference using doubly stochastic variational inference, resulting in full path reconstruction.

## III. HARDWARE

Our system reconstructs the movement path of a 38mg archival receiver mounted on the subject that we want to track. The receiver logs packets and their RSS from any number of rotating transmitters placed in known, stationary positions across the landscape, which are later used in Section V for AoA inference.

The receiver shown in Fig. 1 was designed to be as low profile and lightweight as possible. It consists of a flexible polyimide PCB upon which the DA14531 System on Chip [55], one of the smallest and lowest power commercially available 2.4GHz BLE enabled chips, is surface mounted along with a 32MHz crystal oscillator and two 22µF multi-layered ceramic capacitors (a buffer for the high demand receiving duty cycle). The device uses a 3cm long, 0.2mm width copper PCB trace as a monopole antenna. For the power source we use a 24mg, 11mF Seiko CPH3225A supercapacitor [56] which results in a $1 6 \mathrm { { m m } ^ { 2 } }$ (excluding antenna surface area) package weighing only 38mg. Other power sources are feasible such as a button battery, thin film solar cell or microbatteries.

The receiver scans for and stores 2.4GHz BLE packets sent from the transmitters along with their RSSI (Received Signal Strength Indicator), using a short and therefore low-power

![](images/77e3b10fa7e8285bbbb99553bdd9f187cb255bf8668d1f017707cb820807b49b.jpg)

![](images/a2b1cedb0f046a3dcd2658a9944000dda588d06eb12aa737796983370da74246.jpg)

![](images/622a373a8c9bf8f8ad9a7733f90e87d11c20cda6701df5690cdbb78610b90d3f.jpg)  
Fig. 1. The 38mg supercapacitor powered receiver that the system will track. (a) The receiver, affixed to a green queen bee marking tag. Ruler provided for scale. (b) One side of the receiver showing the supercapacitor and a pad for charging, the white outline bounds the receiver on the PCB, the tracks and pads outside this region are used for programming the DA14531 and then removed. (c) Other side of the receiver, showing the DA14531, crystal oscillator, multi-layer ceramic capacitors, ground pad for charging and a pad to trigger an interrupt for reading data off the receiver.

7ms receiving period (as this is the minimum window of time needed to receive the majority of packets) to capture a packet from all transmitters in range and then enter a hibernation mode until the duty cycle repeats. The RSSI measured by the DA14531 is a chipset-specific indicator proportional to the RSS of the packets. At a receiving power expenditure of 5mA, the 11mF supercapacitor powered receiver can perform over 500 receiving duty cycles over its powered lifespan. Bursts of these receiving cycles are used to sample sets of k packets which are used for AoA inference once the receiver has been retrieved and the data read from its storage.

The total duration of the burst and the intervals within this duration that the receiver turns on can be varied to change k, and therefore balance power consumption & device lifespan with AoA inference accuracy and resulting path reconstruction accuracy. We investigate the effects of varying these parameters through the experiments described in Section VI.

The receiver is charged prior to tracking using a pair of terminals on its corner. The data the receiver has stored can be read using an interrupt pad to change it to a transmitting mode, meaning the receiver is archival.

The rotating transmitters shown in Fig. 2, mounted on tripods at known locations across the landscape, use a 14 dBi high-gain Yagi-Uda antenna to broadcast BLE directed advertising packets to localize the receiver. This type of advertising packet contains only the source (transmitter) and destination (receiver) Bluetooth device addresses, and was chosen as it is specifically addressed to the receiver, allowing for the receiver to ignore all other BLE traffic, reducing the power it consumes when processing received packets. Further, directed advertising packets can be updated quickly and have a high chance of being received at any given receiving duty cycle due to their short, aggressive advertising interval. Directed advertising packets have no data payload; to allow for transmission of data to the receiver, we encode within the 6 byte BLE advertising address the angle γ of the transmitter’s rotation (relative to North), the time of transmission and the id of the transmitter (from which the known location of the transmitter can be found). The packets are constantly transmitted, and updated every 10ms (approximately every 1.8<sup>◦</sup> of rotation at 30rpm). These packets can be received by our BLE receiver up to 300m away from the transmitter, and ideally the receiver should be within range of at least two transmitters for localization.

![](images/98a311a7d1f2b2f770df3c8f5a3cc6f71c481457f9384062fde5dd4962412b3d.jpg)  
Fig. 2. Our simple 30rpm rotating high-gain transmitter with a 14dBi Yagi Uda antenna mounted on a 1.5 meter tall tripod in the field, transmitting packets used to track the receiver.

The transmitters, not including the tripod they are mounted on, weigh 3 kg and measure 20 cm × 20 cm × 20 cm, with the Yagi antenna extending an additional 35 cm forward. A 30 rpm Pololu 150:1 Metal Gearmotor [57] drives the transmitter’s rotation while a Fanstel BT832XE Bluetooth Module [58] reads the motor’s encoder for the angle of rotation and constructs the BLE packets transmitted. A magnet is placed due North on the tripod, such that a Hall effect sensor on the rotating portion of the transmitter can align the angle of rotation with North. A 12V 8Ah LiFePo4 within the transmitter can power it for over 10 hours. The simple design of the transmitters allows several to be easily deployed across a landscape, scaling the trackable area and providing overlapping coverage in the presence of clutter.

## IV. ANGLE OF ARRIVAL INFERENCE

We must infer the AoA of at least two transmitters from the receiver’s location for use in path reconstruction, using the packets and their RSS received by the limited number of duty cycle bursts performed by the receiver. We wish to infer, as shown in Fig. 3(a), the unknown angle θ relative to North that the receiver lies upon when receiving a burst of k RSS measurements, y, taken at known angles $\gamma$ of the transmitter ARP’s (shown in Fig. 3(b)) rotation relative to North (encoded in the received packets) i.e. $p ( \theta | \boldsymbol { y } , \gamma )$ .

Ideally we should minimize the value of k required to infer an accurate AoA, as this will minimize the power consumption of the receiver. We present here three different approaches for inferring an AoA from a limited number of

![](images/e03b98c32e3c9138a6c7390463eb02df9c2f1845ca682eb8464c3a94343cfae7.jpg)  
Fig. 3. The problem of inferring receiver angle from packets received. (a) Definitions of angles. We seek θ, the angle the receiver lies upon from the transmitter relative to North. γ is the angle relative to North that the transmitter and therefore its ARP is oriented towards, this is always known as it is encoded in the packets received. (b) Polar plot of the ARP of our transmitter’s 14 dBi Yagi Uda antenna oriented $4 5 ^ { \circ }$ from North, to show how the antenna’s gain varies by angle. The ARP was measured empirically as described in Section VI-A, and averaged at each angle to remove noise as suggested by Kim et al. [59].

RSS measurements, and evaluate their performance in field experiments described in Section VI-B.

## A. Full Probability Distribution Method

Similar to the approaches applied by Ghosh et al. [53] and Varotto et al. [37] - in which patterns in RSS measurements are matched with the peaks/nulls of a known high-gain antenna’s ARP - we note that the anisotropy of our high-gain antenna generates unique changes in y across its rotation irrespective of distance, notably around its ${ \sim } 4 0 ^ { \circ }$ H-plane HPBW. These changes in y will be offset by the known $\gamma$ and unknown θ. Therefore we can infer θ from observing y and using the known values of $\gamma .$ . We model this using a Bayesian approach,

$$
p ( \theta | \mathbf { \boldsymbol { y } } , \gamma ) = \frac { p ( \mathbf { \boldsymbol { y } } | \theta , \gamma ) p ( \theta ) } { p ( \mathbf { \boldsymbol { y } } | \gamma ) } .\tag{1}
$$

The true signal strengths t, transmitted from the rotating high-gain antenna in the direction of the receiver, depend only on the direction of the receiver, $\theta ,$ offset by the known angles the transmitter is facing, $\gamma ,$

$$
\pmb { t } = \left[ t ( \theta - \gamma _ { 1 } ) , t ( \theta - \gamma _ { 2 } ) , . . . , t ( \theta - \gamma _ { k } ) \right] ^ { \top } .\tag{2}
$$

They are then affected by some unknown attenuation $^ { a , }$ which is primarily due to transmitter-receiver distance, but can also result from obstructions or multipath interference. An important side note is that, we assume that the k measurements of RSS within the burst are taken close enough together in time that they are affected by the same constant value of a. The most informative part of the pattern is around the narrow main lobe, and as the $4 0 ^ { \circ }$ HPBW passes across the receiver in only 220ms, it is likely that the attenuation is almost constant. Finally the actual observations of RSS made by the receiver are assumed to be corrupted by some uncorrelated, zero-mean Gaussian noise with variance $\sigma ^ { 2 }$ . This model is described with plate notation in Fig. 4.

To use this approach we must first empirically profile the ARP of our transmitter by taking several RSS measurements from a range of angles of rotation $\gamma \sim \mathrm { U n i f o r m } ( 0 ^ { \circ } , 3 6 0 ^ { \circ } )$ at a known $\theta = 0$ . We can interpolate between the average signal strengths at each sampled angle to allow us to estimate $t ( \theta - \gamma )$ - i.e. what the transmitter’s signal strength will be when the tag is at bearing θ from the transmitter, and the transmitter is pointing at angle $\gamma .$ . Note that these empirical transmitter RSS values are themselves attenuated, but because the receiver was stationary during this data collection, this is a constant offset; and, as we’re only interested in the relative change in signal strength, doesn’t affect inference.

![](images/c2ca484a5af2f64fd13a3c147da439cd6aa83bc6d73a7921d613984dd3a51ab4.jpg)  
Fig. 4. The simple graphical model relating the angle θ the receiver lies upon relative to North and the observed RSS measurements $\textbf { \textit { y } } - \textbf { \textit { a } }$ set of k RSS observations taken at angles $\gamma$ of transmitter rotation relative to North. We assume that the $\mathrm { \hbar } \mathrm { \hbar } \mathrm { \hbar } \mathrm { \Omega } ^ { \ast }$ unattenuated signal strength, $t _ { i } ,$ depends on just the angle $\theta - \gamma _ { i }$ . We assume in this model that all the RSS measurements were taken close enough together in time that they all were generated by angles close to θ and experienced the same attenuation a to result in signal strengths $f .$ Finally the RSS measurements actually observed by the receiver, $^ { \mathbf { \psi } _ { \mathbf { \psi } } }$ are modelled by assuming some Gaussian noise corrupts this attenuated signal.

So a single observation of the RSS, $y _ { i } .$ , is modelled as,

$$
p ( y _ { i } | \theta , \gamma _ { i } , a ) = N ( y _ { i } | t ( \theta - \gamma _ { i } ) - a , \sigma ^ { 2 } ) .\tag{3}
$$

We have several observations $( y _ { 1 } . . . y _ { k } )$ for a given value of θ. a is unknown but, by putting an improper flat prior on a, we can integrate it out in closed form (see Supplementary Material SI for detailed derivation). Defining $d = | | ( { \pmb t } - { \bar { \pmb t } } ) - ( { \pmb y } - { \bar { \pmb y } } ) | | _ { 2 }$ we can write,

$$
p ( \pmb { y } | \theta , \gamma ) = N ( d | 0 , \sigma ^ { 2 } ) .\tag{4}
$$

So to compute $p ( \theta | \mathbf { y } , \gamma ) = p ( \mathbf { \pmb { y } } | \theta , \gamma ) p ( \theta ) / p ( \mathbf { \pmb { y } } | \gamma )$ , we note that we have assumed a uniform prior on θ, and that the denominator is constant w.r.t. θ, which means we can compute $p ( \pmb { y } | \theta , \gamma )$ over a uniform grid and normalise.

To summarise the calculation: We compute for a uniform grid of angles from $\theta \sim U n i f o r m ( 0 ^ { \circ } , 3 6 0 ^ { \circ } )$ , the predicted transmitter signal strength $t ( \theta \mathrm { ~ - ~ } \gamma _ { i } )$ associated with each observation’s transmitter angle $\gamma _ { i }$ . We then mean centre y<sub>i</sub> and $t _ { i }$ and compute the probability of our $k$ observations as,

$$
p ( \pmb { y } | \theta , \gamma ) = C \times \exp \Big [ - \sum _ { i = 1 } ^ { k } \frac { ( ( t _ { i } - \bar { t } ) - ( y _ { i } - \bar { y } ) ) ^ { 2 } } { \sigma ^ { 2 } } \Big ] .\tag{5}
$$

![](images/7c359fc03e4c8931b02deaf7571ead796f957b0d9c293ea2136b38beeafb95f1.jpg)  
Fig. 5. Example of AoA prediction returned by full probability distribution approach. (a) An example of a burst of RSS measurements (red), taken at a true angle of 4.6 radians (green), of packets received from a single rotating high gain transmitter. This burst was 2000ms in duration, and receiving duty cycles were performed at an interval of 120ms, resulting in a total of $k = 1 5$ samples taken. (b) The calculated probability distribution of the predicted angle $p ( \boldsymbol { \theta } | \mathbf { \boldsymbol { y } } , \gamma )$ from the full probability distribution method given the RSS measurements taken (blue). Note that the probability density is highest around the true angle, showing that our $p ( \theta | \mathbf { \boldsymbol { y } } , \bar { \boldsymbol { \gamma } } )$ closely represents the true angle θ that generated the RSS observations y at angles of transmitter rotation γ.

Finally we normalise these probabilities, across θ, to give the distribution of $p ( \theta | \mathbf { \boldsymbol { y } } , \gamma ) \cdot \mathrm { o u r }$ probability distribution of the angle the tag lies upon from the transmitter relative to North. An example of a calculated distribution of $p ( \boldsymbol { \theta } | \boldsymbol { y } , \gamma )$ given an RSS measurement burst is shown in Fig. 5.

We can use the full probability distribution of $p ( \boldsymbol { \theta } | \boldsymbol { y } , \gamma )$ as an AoA observation for path reconstruction. This captures not only the AoA we predict the receiver lies upon from the transmitter, but also the uncertainty in our prediction, which is useful as even highly uncertain predictions may still contribute information to the path of movement the receiver took through the landscape.

## B. Rejection Sampling Method

$p ( \boldsymbol { \theta } | \boldsymbol { y } , \gamma )$ can also be approximated using a training set of signal strengths to perform rejection sampling.

To do this we generate a table R in which each row is a set of RSS measurements (associated with a known θ and $\gamma )$ that were received at a set of angles whose relative differences match those in our observations $\gamma .$ . We then select all the rows of R in which the training set’s mean-centred signal strengths are almost equal to our mean-centred signal strength observations. For these selected rows, their associated values of $\theta$ are taken and used as empirical sets of samples from $p ( \boldsymbol { \theta } | \boldsymbol { y } , \gamma )$ . These are plotted in Fig. 6(a) in which we estimate $p ( \boldsymbol { \theta } | \boldsymbol { y } , \gamma )$ at each observation time by assuming we only have access to five packets spaced uniformly at times [-1s, -0.5s, 0s, +0.5s, +1s] (enough time for the transmitter to rotate one radian on either side of the prediction time).

(a)  
(b)  
![](images/b45dfbfc82396be0f4a727faa99a0b05df2d155f788f50f722c074227b1be248.jpg)  
Fig. 6. Example of AoA predictions returned by rejection sampling approach. (b) A plot of RSS measurements made when the receiver was positioned π radians relative to North from the transmitter and captured packets (red) every 50ms for 10 seconds. (a) The data in (b) is subsampled, such that, at each time point just 5 packets spaced evenly 1 second to either side are used with the rejection sampling approach: The angles associated in the training set with similar relative signal strengths are plotted in black (approximating $p ( \boldsymbol { \theta } | \boldsymbol { \mathbf { y } } , \gamma ) )$ . One can see that, from RSS measurements taken around the main lobe of the transmitter’s antenna (where the RSS peaks), the samples from $p ( \boldsymbol { \theta } | \mathbf { \boldsymbol { y } } , \gamma )$ are confidently centred around the true angle $\theta = \pi$ (blue line). Samples taken away from the main lobe are instead highly uncertain and distributed across many values of θ.

As demonstrated by Fig. 6a, RSS measurements shown in Fig. 6(b) taken around the HPBW of the antenna (where the RSS peaks) generate samples from R that are densely centred around the true θ, but away from this unique part of the ARP there are many likely values of θ. This demonstrates a future direction for this work; using an active learning approach to target the most informative periods.

If we wish to avoid using the full distribution of $p ( \boldsymbol { \theta } | \mathbf { \boldsymbol { y } } , \gamma )$ for later path inference, and instead use a single estimate of $\theta ,$ we can select only confident predictions from multiple bursts by only including those with a small standard deviation, and use the circular mean of these as a single angle observation for later path reconstruction. We found rejection sampling highly sensitive to parameters such as the number of signal strengths per burst (therefore the size of each row in R) and the threshold of how similar our observations y must be to those in a row for its associated θ to be sampled.

## C. Peak RSS Method

A simple heuristic can also be used for AoA inference by noting that, if we were to receive a very large number of packets, $\theta = \gamma$ at the angle $\gamma$ that causes the transmitter to face the receiver and therefore produces the peak RSS across a full rotation of the transmitter. However, we do not run the receiving duty cycle often enough for one of the packets to be received exactly when $\theta = \gamma$

To approximately find the this peak, we smooth y across its corresponding values of $\gamma$ using a Savitzky–Golay filter, and use the peak of the smoothed curve to be our estimate of the AoA θ. This method is simple to implement but highly susceptible to noise and the properties of the duty cycle burst we use to measure y. A burst that is too short or does not have enough measurements k to have a high probability of measuring a value of y around the HPBW will not produce a good estimate of θ.

## V. PATH RECONSTRUCTION

## A. Choice of Model

Triangulation from the AoA predictions over time inferred in Section IV is not feasible as predictions from multiple transmitters are not necessarily concurrent, and the predictions made are often highly uncertain due to RSS noise. We wish to infer $p ( F | Y )$ , the path $F$ the receiver took over time that generated the AoA observations Y. This is a Bayesian inference problem,

$$
p ( F | Y ) = \frac { p ( Y | F ) ~ p ( F ) } { p ( Y ) } .\tag{6}
$$

We model the receiver’s path as a Gaussian process (GP) $\mathrm { ~ \textbf ~ { ~ - ~ } ~ a ~ }$ Bayesian model that can be used for regression, defined by a mean function and covariance/kernel function $K ( x , x ^ { \prime } )$ We place GP priors on the $\mathrm { ~ x ~ } \& \ \mathrm { y }$ coordinates of the receiver as functions of time. Our kernels are constructed with the assumption that the axes are independent a priori; however we model these with variational inference using a Gaussian endowed with a full rank covariance matrix, such that in the posterior the two axes can interact. A GP is a useful choice of model as it allows a priori knowledge of the tracked subject’s plausible movement to be included via the choice of $K$ and provides a convenient expression of uncertainty [60].

We identify two suitable kernel functions, parametrized by length scale l and scale factor $s ^ { 2 }$ . The first is the commonly used EQ kernel,

$$
K _ { E Q } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = s ^ { 2 } \exp \Big ( - \frac { | \mathbf { x } - \mathbf { x } ^ { \prime } | ^ { 2 } } { 2 l ^ { 2 } } \Big ) .\tag{7}
$$

The second is the integral of the EQ kernel, which places the EQ prior on the velocity. An important difference is that while the posterior of the EQ kernel will tend towards the mean and a limited variance in the absence of observations, the integrated EQ kernel will have a mean that depends mainly on the last location observed and a variance that grows over time. This is clearly preferable in tracking scenarios where the subject does not reliably return to a nominal position when not observed. As we seek a GP over positions rather than velocities, we integrate the EQ kernel w.r.t. each of its inputs [61],

$$
K _ { f } ( x , x ^ { \prime } ) = \int _ { 0 } ^ { x } \int _ { 0 } ^ { x ^ { \prime } } K ( u , v ) d v d u\tag{8}
$$

$$
= \int _ { 0 } ^ { x } \int _ { 0 } ^ { x ^ { \prime } } s ^ { 2 } \exp \Big ( - \frac { ( u - v ) ^ { 2 } } { 2 l ^ { 2 } } \Big ) d v d u\tag{9}
$$

$$
= s ^ { 2 } \frac { l ^ { 2 } } { 2 } \Bigg [ g \Big ( \frac { x } { l } \Big ) - g \Big ( \frac { x - x ^ { \prime } } { l } \Big ) + g \Big ( \frac { x ^ { \prime } } { l } \Big ) - g ( 0 ) \Bigg ] ,\tag{10}
$$

where $g ( z ) = z { \sqrt { \pi } } \operatorname { e r f } ( z ) + e ^ { - z ^ { 2 } }$

Note that the start index 0 is arbitrary - we would recommend placing it sufficiently far back in time from the observations such that the model can explain the observed locations through integrating a small velocity in a way compatible with the kernel’s hyperparameters (i.e. that doesn’t require excessive speeds).

## B. Variational Inference

We use variational inference [62] to seek a surrogate posterior $q ( F )$ to approximate the unknown true posterior $p ( F | Y )$ with variational parameters we know. Variational inference has the benefit of being computationally inexpensive when compared to methods such as Markov Chain Monte Carlo [63] or Expectation Propagation [64], allowing for inference to be performed in the field upon retrieving the receiver.

The surrogate posterior $q ( F )$ is optimized to most closely represent the true posterior by minimizing the Kullback-Liebler (KL) Divergence between them,

$$
{ \mathcal { D } } _ { K L } [ q ( F ) | | p ( F | Y ) ] = - \int q ( F ) \log { \frac { p ( F | Y ) } { q ( F ) } } d F\tag{11}
$$

$$
= - \int q ( { \boldsymbol { F } } ) ~ \log ~ \frac { p ( { \boldsymbol { F } } , { \boldsymbol { Y } } ) } { q ( { \boldsymbol { F } } ) p ( { \boldsymbol { Y } } ) } d { \boldsymbol { F } }\tag{12}
$$

$$
= - \int q ( F ) \log \frac { p ( F , Y ) } { q ( F ) } d F + \int q ( F ) \log p ( Y ) d F\tag{13}
$$

$$
= - \underbrace { \int q ( \boldsymbol { F } ) \log \frac { p ( \boldsymbol { F } , \boldsymbol { Y } ) } { q ( \boldsymbol { F } ) } } _ { \mathrm { ~ } q ( \boldsymbol { F } ) } d \boldsymbol { F } + \log p ( \boldsymbol { Y } )\tag{14}
$$

The marginal likelihood $p ( Y )$ is unknown, but can be regarded as constant w.r.t. the surrogate distribution. Therefore maximising the Evidence Lower Bound (ELBO) is equivalent to minimizing the KL divergence bounded by this constant [62].

We use a vector of inducing variables u [65] which describe $F ,$ selected at inducing inputs $Z$ evenly spaced in time across the duration of the path. We still assume that $Y$ depends only on $F .$ . The posterior is therefore calculated as $p ( F , u | Y ) \propto p ( Y | F ) p ( F | u ) p ( u )$ . We assume that u is able to sufficiently determine F such that $p ( F | u , Y ) = p ( F | u )$ [66], and using this assumption we should augment the ELBO to use the inducing variables,

$$
\mathcal { L } = \int \int q ( \boldsymbol { F } , \boldsymbol { u } ) \log \frac { p ( \boldsymbol { F } , \boldsymbol { u } , Y ) } { q ( \boldsymbol { F } , \boldsymbol { u } ) } d \boldsymbol { F } d \boldsymbol { u }\tag{15}
$$

$$
= \int \int q ( { \cal F } , u ) ~ \mathrm { l o g } ~ \frac { p ( { \cal Y } | { \cal F } ) p ( { \cal F } | u ) p ( u ) } { p ( { \cal F } | u ) q ( u ) } d { \cal F } d u\tag{16}
$$

$$
= \int \int q ( F , u ) \log p ( Y | F ) d F d u + \int q ( u ) \log { \frac { p ( u ) } { q ( u ) } } d u\tag{17}
$$

$$
= \mathbb { E } _ { q ( F , u ) } [ \log { p ( Y | F ) } ] - { \mathcal { D } } _ { K L } [ q ( u ) | | p ( u ) ] .\tag{18}
$$

One can see that $\mathcal { L }$ now consists of a data fitting term and a prior term.

L is now iteratively maximised w.r.t. the variational parameters (mean m and lower diagonal matrix $R ,$ to give the positive semi-definite covariance $R R ^ { \top } )$ ) using doubly stochastic variational inference [67]: at each iteration multiple samples are drawn from $F _ { i } \sim q ( F )$ by sampling from $q ( u )$ and computing $F _ { i }$ from the resulting $u _ { i }$ . These are used to calculate the log likelihood of the observations log $p ( Y | F _ { i } )$ and averaged to approximate the expectation of the first term of ${ \mathcal { L } } ,$ which is,

$$
\mathbb { E } _ { q ( F , u ) } [ \log p ( Y | F ) ] \approx \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log p ( Y | F _ { i } ) .\tag{19}
$$

As the prior $p ( u )$ and surrogate posterior $q ( u )$ are both Gaussian distributed, their KL divergence in the second term of Equation (18) has a closed form solution.

JAX [68], an automatic differentiation framework, is then used to compute the gradient of $\mathcal { L }$ wrt to m and $R ,$ and update them accordingly. When $\mathcal { L }$ reaches its global maximum, the surrogate distribution most closely matches the true posterior. The resulting surrogate distribution $q ( F )$ can be used to compute the location distribution at any given time point.

## C. Likelihood

The variational inference approach requires that we write an autodifferentiable function to compute log $p ( Y | F _ { i } )$ ), the log likelihood of an observation, given a sample $F _ { i }$

AoA inference via the full probability distribution method (described in section IV-A) produces AoA observations consisting of a distribution over a finite set of angles. The rejection sampling method (section IV-B) returns an empirical set of samples of θ which can be used to return a single estimate of the AoA. The peak RSS finding method (section IV-C) only returns a single AoA. We present two likelihood functions to be used in the calculation of the ELBO, one for the full distribution and one for a single AoA.

AoA Probability Distribution Likelihood: The probability distribution we compute in Section IV-A for the $\mathrm { A o A } p ( \boldsymbol { \theta } | \boldsymbol { y } , \gamma )$ is already nearly the likelihood function we need - we simply need to compute, for the sample $F _ { i }$ , what the angle to the receiver $\phi$ would have been and, in principle, simply report the value of log $p ( \phi | \mathbf { \boldsymbol { y } } , \gamma )$

We first note that, to find ϕ simply requires trigonometry: the point p on the sampled path $F _ { i }$ at the time of this observation has the transmitter’s location l subtracted from it. Then the angle relative to North is computed,

$$
\phi = \arctan 2 ( p _ { N o r t h } - l _ { N o r t h } , p _ { e a s t } - l _ { e a s t } )\tag{20}
$$

Our final step is to simply report log $p ( \phi | \boldsymbol { y } , \gamma )$ for this value of $\phi .$ However we computed $p ( \boldsymbol { \theta } | \mathbf { \boldsymbol { y } } , \gamma )$ previously by evaluating $p ( \pmb { y } | \gamma , \theta )$ over a grid of $\theta \sim U n i f o r m ( 0 ^ { \circ } , 3 6 0 ^ { \circ } )$ and normalizing. We need to now report log $p ( \phi | \boldsymbol { y } , \gamma )$ for an arbitrary $\phi$ such that it is still normalized, it is fast and it is possible to autodifferentiate.

To do this we linearly interpolate between the precomputed values of log $p ( \boldsymbol { \theta } | \mathbf { \mathcal { y } } , \boldsymbol { \gamma } ) \ \mathrm { e . g . }$ . if $\phi = 3 3 . 2 ^ { \circ }$ , the result will be log $p ( \theta = 3 3 | y , \gamma ) \times 0 . 8 ~ +$ log $p ( \theta = 3 4 | y , \gamma ) \times 0 . 2$ . This allows the autodifferentiation tool in $\mathrm { J A X }$ to compute meaningful gradients. We assume that the precomputed values are close enough together that the gradient is well approximated by this interpolation.

Single AoA Likelihood: In Sections IV-B and IV-C we described approaches for selecting a single angle θ rather than a distribution for each observation. For observations in this form (of a single AoA) we need an alternative likelihood function. One might just place a Gaussian over the angular difference between the observed angle and the proposed receiver location, this however penalises those predictions close to a transmitter where a small change in location will lead to a large change in angle.

We convert the single angle observation into a two dimensional line L (using cos θ and sin θ) described by a unit vector v extending from a transmitter’s location l. The likelihood of $L$ given a predicted receiver location $\pmb { p }$ on a sampled path $F _ { i }$ should depend on $\pmb { p } \mathbf { \check { s } }$ distance $d ,$ from $L ,$ as the further away a predicted location is from this line, the less likely we assume it to be. This is given as,

$$
d = | \pmb { v } \times ( \pmb { p } - \pmb { l } ) |\tag{21}
$$

A Gaussian distribution $p ( \theta | f ) = N ( d , 0 , \sigma ^ { 2 } )$ is then used to compute the likelihood of observations. For this to be a valid likelihood $\int p ( y | f ) d y$ must equal a constant. This isn’t quite true for this likelihood function, but can be shown to be approximately true for our problem (see Supplementary Material SII for detailed derivation).

## VI. EXPERIMENTAL DESIGN

To demonstrate the ability of our system to track the receiver using the methods discussed, we have designed several experiments described here, the results of which are presented in Section VII.

## A. Empirical Measurement of Transmitter ARP

For all experiments conducted, training data (used to empirically measure the transmitters’ ARP for use in the full probability distribution and rejection sampling AoA inference methods) was collected from 10 minutes of continuous scanning of packets along with their RSS from a single rotating transmitter at The Ponderosa Park, Sheffield, UK. The receiver was positioned 30m due North from the transmitter with clear line-of-sight, and was powered by an external power supply to allow for high frequency (10ms scan interval) capturing of packets, ensuring that sufficient training data was collected. This data is then used when inferring AoAs from new RSS observations as described in Sections IV-A and IV-B.

All experiments described below are conducted in different locations to that which the training data was collected, ensuring we assess the ability of the system to generalise for unknown landscapes.

## B. AoA Inference Experiment

The first experiment aims to assess how accurately the three different AoA inference approaches can infer an AoA from a burst of k received packets. It also investigates how varying k affects the accuracy of AoA inference across the approaches, to explore to what extent k can be minimized to save power. Data was collected at Norfolk Heritage Park, Sheffield, UK in which we placed the receiver at a known angle θ from North, 100m away from the transmitter with clear line-of-sight, and used the receiver to capture 100 bursts, 2000ms in length, with 200 packets per burst (the highest sampling rate possible on the DA14531).

From this test data we randomly choose one of the bursts, subsample k packets from the burst at uniformly spaced intervals, and apply the three approaches for AoA inference as described in section IV. We calculate the angular difference between the inferred AoA and the known true value of $\theta ,$ and repeat 1000 times for all three AoA inference methods to calculate the MAE (mean absolute error) and standard deviation of the MAE in degrees for each. As the number $k$ packets per burst is a key consideration in the power consumption of the tag, we do this for several values of k to present how the different approaches for AoA inference vary with the granularity of RSS measurements across a burst.

In these tests, the assumed Gaussian noise standard deviation $\sigma$ in the RSS values is chosen to be 6dB based on empirical observations. To allow us to also assess the accuracy of the full probability distribution method, we use the mode (the θ with the highest probability) to be the angle prediction we calculate the error for so that the methods are comparable - though this does discount some information present in the full probability distribution that may be beneficial to path reconstruction later.

## C. Path Reconstruction Experiment

The next experiment is conducted to assess the ability of the system to reconstruct the path of a receiver using a GNSS unit for ground truth, and compare how AoA observations inferred from our multiple AoA inference approaches and varying values of k affect the resulting path reconstruction. This allows us to determine how few RSS measurements we can use to minimize power consumption while successfully reconstructing a path.

Data was collected at the Norfolk Heritage Park, Sheffield, UK - a park with complex topology differing by 10 meters between the highest and lowest point of the area we conducted our experiment in, some tree cover and buildings around the edges. This experiment consists of 5 complex paths and 1 straight line path, each around 3 minutes in length within a 200m × 200m area. The paths conducted were each in different directions and of different shapes. Four transmitters were set up ∼125m apart from each other. A researcher moved the receiver at a fast walking pace along the paths while also recording their movements with a mobile GNSS device. The tag was set to perform burst scans 2000ms in length with a scanning interval of 10ms, meaning up to 200 packets per 2 second burst could be received (the maximum sampling rate possible with the DA14531). We assume the bursts of samples are continuous i.e. every 2000ms there is a 2000ms burst.

The receiver was powered by an external power supply during data collection to allow for the dense scanning duty cycle of up to 200 packets per burst to be performed over the several minute span of each path. We can then, for each path, subsample k packets from each burst at uniformly spaced intervals to evaluate performance between both the limited packets received at low values of k that the supercapacitor can handle and higher values of k. By subsampling from the same underlying measurements we ensure that the exact same movement paths and attenuation generated the RSS measurements evaluated across different values of k

We then reconstruct a path using the variational inference approach discussed in section V, from AoA predictions inferred using the AoA inference methods presented in Section IV.

The doubly stochastic variational inference algorithm optimises 60 inducing points per axis, evenly spaced between the start and end time of the data collected for each path. The integral of the EQ kernel is used, with a length scale of 0.5 seconds and a scale factor of $0 . 5 \mathrm { m s ^ { - 1 } }$ , chosen to reflect smooth continuous movement and a velocity that is consistent with walking.

Once the ELBO converges, the paths are inferred at evenly spaced time points at which each of the GNSS data points were recorded, such that we can calculate point-for-point the error (in meters) between the GNSS path and the inferred path. These errors are used to calculate the MAE and the standard deviation of the MAE across all paths assessed for each value of k and each AoA inference method.

## D. Duty Cycle Parameter Investigation

The limited capacity of a power source small enough to fit on a subject such as a flying insect necessitates careful use of the few scanning duty cycles available across the powered lifespan of the receiver. So far the experiments described have only explored the effect of changing the number of samples in each burst, but in reality more samples per burst means we must take fewer bursts (for the same energy budget).

This experiment tests if it is more beneficial to take less frequent, accurate AoA observations or a greater number of uncertain AoA observations. Being able to customise the trade-off between power usage and measurement accuracy is a design consideration also explored by other localization systems in which power is a key factor [12]. We first define a synthetic path comprised of a sine wave shape (arbitrarily selected as a complex path to test) across a minute of time, and two transmitters positioned North and West of the path. We use a synthetic path here so that we know the true AoAs that the path generated and therefore can plot AoA inference accuracy against path reconstruction accuracy.

We generate n AoA observations, inferred from a burst scan of k RSS measurements, at linearly spaced intervals in time along the path. We do this by finding the angle from a transmitter at each time and taking k samples from an ARP profiled at Weston Park, Sheffield, UK (collected in the same method as the training data was generated in Section VI-A) oriented to the matching angle. This ARP profile is only used to generate the RSS observations from our synthetic path; the training data for the AoA inference models is still the data that was collected at The Ponderosa, Sheffield as described in Section VI-A. We use a different ARP here to generate the synthetic path so that our synthetic RSS observations are not generated from the same training data used to infer an AoA, avoiding overfitting. The samples are taken equally spaced across a 2000ms duration, so as to simulate a 2000ms burst scan.

We simulate a budget of 100 RSS measurements (though the real receiver’s budget is considerably larger) taken across the path. We then consider a number of values of k such that our total set of observations varies between a few bursts containing many RSS measurements (resulting in confident AoA predictions) or many bursts containing very few RSS measurements (resulting in uncertain AoA predictions). For each value of k, we sample 100 sets of RSS measurements from our synthetic path, and for each set we perform AoA inference on each burst, reconstruct a path and calculate the total MAE of AoA inference and path reconstruction across the sets. This allows us to assess how trading off quality of AoA predictions for quantity of predictions affects our final path reconstruction.

## E. Application in Flying Insect Tracking

To demonstrate the practical application of the system, we deploy it in a flying insect tracking field trial. The goal of the trial is to observe the ability of foraging Bombus terrestris to find their way back to their nest [69].

An artificial Bombus terrestris nest [70] was placed in a copse of trees at Common Lane Open Space, Sheffield, UK. The nest was given around a week to begin foraging in the locale to allow workers to become familiar with the local environment, and following this on the 19th of August 2025, four transmitters were set up in the 250m × 250m area surrounding the nest site. This area contained a second copse of trees and highly complex topology with a difference of 16m of elevation between the highest and lowest points.

A release site was designated ∼100m away from the nest. Foraging Bombus terrestris returning to the nest were caught with a net and placed in a queen marking pot, then moved to the release site. The receiver was glued to a queen bee marking tag to increase the surface area with which the tag could contact the bee, and the tag was affixed to the caught bee’s thorax using cyanoacrylate superglue (shown in Fig. 7).

The receiver was set to perform burst scans 1500ms in length consisting of k = 10 RSS measurements. The burst scans were performed 10 seconds apart from each other. From the retrieved data, we inferred AoA observations using the full probability distribution method, and used the integral EQ kernel with the same hyperparameter choices as those in Section VII-B in path reconstruction.

Finally the bee was released. Upon re-observation of a tagged bee at the nest the tag was removed, data was downloaded from the tag via BLE and the path the bee took from the release site to the nest was reconstructed.

## VII. RESULTS

## A. AoA Inference Accuracy

We first assess the ability of the system to infer the AoA of a burst scan of packets received from a single transmitter as described in Section VI-B. The results are presented in Fig. 8.

![](images/f7579bb7caaf09cbc16865c20135e9986bea196e63abcfa74b321e83fb25b6f4.jpg)  
Fig. 7. Image of Bombus terrestris tagged with a receiver described in Section III (glued to a green queen marking tag) sitting on the plunger of a queen marking pot following release during the experiment described in Section VI-E.

(a)  
![](images/d4ad226447a6b93287c81d8fbb1f2338606dded2a1de42e2d1f3b3b855c67804.jpg)

![](images/3a2d5c7be1d00eefa65d1728a103a149b20bbfbde83d40ae6b45100468a90273.jpg)

![](images/dea6d8edfacbc435d44b7873dccc2def32ea58053c007820c3d68c2cf347c1db.jpg)  
Fig. 8. Plots showing the MAE in degrees of each AoA inference method across a range of k number of RSS measurements used in the burst scan from which the angles were inferred. One standard deviation above and below the mean result is represented by the shaded areas. (a) MAE of rejection sampling method. (b) MAE of peak RSS method. (c) MAE of full probability distribution method.

These results show that all three methods result in a low mean angular error when a large number (k) of RSS measurements are taken in a burst. The full probability distribution and peak RSS methods produce an MAE of under $2 ^ { \circ }$ with a standard deviation of approximately 1<sup>◦</sup>. Rejection sampling is notably inconsistent at these high values of k, resulting in an MAE of $5 ^ { \circ }$ with a large standard deviation of around $4 0 ^ { \circ }$ , where the other methods produce confidently correct predictions of $\theta$ as expected. This is because, at high values of $k ,$ there are very few training points which match all observed RSS values in the rejection sampling training set. While a simple solution could be collecting more training data, exponentially more data is required as k increases and this would significantly increase the already high computational complexity of constructing the rejection record and sampling from it.

We are, however, aiming to use the minimum value of k possible to save power on the tag, and all methods begin to notably decay in accuracy below 15 RSS measurements. The full probability distribution and rejection sampling approach produce an MAE of ${ \sim } 5 0 ^ { \circ }$ at $k = 5$ packets, while the peak finding approach nears a random estimate of θ, and all methods result in highly uncertain AoA predictions with a standard deviation of over ${ \sim } 5 0 ^ { \circ }$

This likely occurs as the limited number of samples y taken at low values of k are unlikely to fall in or close to the HPBW of the high-gain antenna - the ${ \sim } 4 0 ^ { \circ }$ frontal lobe of the ARP constitutes only 11% of the ARP and we scan over a 2000ms burst (one rotation of the 30 rpm transmitter), so as k gets smaller than 10 the probability of missing the HPBW increases. Peak finding fails as the peak is largely absent from the data. Rejection sampling simply returns a high number of records constructed from regions of the ARP away from the main lobe. Both of these approaches at values of k lower than 10 are too uncertain to be useful for later path reconstruction.

The full distribution method also suffers from highly uncertain AoA predictions at these low values of k, but has the added benefit of returning the full distribution of $p ( \boldsymbol { \theta } | \boldsymbol { y } , \gamma )$ . We perform another test to show that estimates of $p ( \boldsymbol { \theta } | \boldsymbol { y } , \gamma )$ from the full probability distribution method that have a poor MAE due to using low values of k are not confidently wrong and do capture the uncertainty of our prediction so that we can still use these uncertain estimates as input observations for path reconstruction. Using the method described in Section VI-B, we calculate the mean negative log predictive density (NLPD) of $p ( \boldsymbol { \theta } | \boldsymbol { y } , \gamma )$ inferred from several subsampled sets of y across a number of values of $k$ as a measure of AoA inference effectiveness. We plot this against the mean entropy of these inferred $p ( \boldsymbol { \theta } | \boldsymbol { y } , \gamma )$ distributions in Fig. 9 to show that estimates with poor predictive density have high entropy and therefore capture the uncertainty in predictions of θ.

## B. Path Reconstruction Accuracy

We now assess the ability of the system to reconstruct the path of a receiver using a GNSS unit for ground truth through the experiment described in Section VI-C. We have chosen not to assess the rejection sampling method (Section IV-B) further as it was shown in Section VII-A to have significantly poorer AoA inference accuracy than the other AoA inference methods (with substantially higher computational cost too).

An example of one of the paths reconstructed in this experiment and placement of the transmitters in the landscape is shown in Fig. 10, and all paths assessed are shown in Supplementary Material SIII.

Figure 11 presents the MAE computed by combining the six paths using either the full probability distribution method or peak finding method at a number of choices of k to show how the degradation in AoA inference from taking very few RSS measurements observed in Section VII-A propagates to the accuracy of the inferred path.

![](images/ed28600deca12794db0cacbd1ad31507a15a6e4e31fd517e62f76a689a4e3ecb.jpg)  
Fig. 9. Plot of mean NLPD against mean entropy for our predicted angle distribution $p ( \boldsymbol { \theta } | \mathbf { \boldsymbol { y } } , \gamma )$ calculated for 1000 y subsampled from the test data across several values of k, showing that, as the predictive ability of $p ( \boldsymbol { \theta } | \mathbf { \boldsymbol { y } } , \gamma )$ decays with reducing values of $\bar { k , }$ its entropy increases therefore capturing the uncertainty of our reduced accuracy predictions.

![](images/7f95d9170711e72fe0907e8e1bf7820070a4f593267e4e40204399724afe4815.jpg)

(b)  
![](images/2d6d3aa3fcf59d572bddd0cf649e7465225313814c9bb774867de1b62f8eb166.jpg)

![](images/817d6e8529f5b2331e6af14a6f327d96c276ea374a35f42e044be1a6d3ea6e32.jpg)  
Fig. 10. Example of one of the six paths used for assessing path reconstruction accuracy. The four transmitters are labelled, the blue path is the GNSS ground truth (3m accuracy), the orange path is the mean of the reconstructed path and the ellipses are $^ 2$ standard deviations of the covariance matrix at time points every 10 seconds along the inferred path. $k = 1 0$ samples were used per burst in these examples, and a path reconstructed from AoAs inferred via the (a) full probability distribution method and (b) peak finding method are presented. (c) A satellite view of Norfolk Heritage Park, Sheffield, UK at which the paths took place is also presented, retrieved from Google Maps [71]. Note that the inferred paths’ uncertainty grows at the beginning and end of the tracked paths, as before/after these points in time respectively there are no observations.

Our results show that path inference can be successfully carried out to within 10m MAE using both assessed AoA inference methods when using a high enough number of packets $k$ per burst. However, we are aiming to use a minimal amount of power by using the smallest value of k possible. The full probability distribution method is by far the most robust at low values of $k ,$ being able to reconstruct a path of movement within an accuracy of ∼15m in extremely power constrained scenarios where as few as 3 RSS samples per burst are taken.

![](images/1ca640f3384b8102ec375c08c24986e1d1301dc3ebd48dac0dac6d01461b57f7.jpg)  
Fig. 11. Plot of the MAE (m) of paths reconstructed using variational inference, from k AoA observations inferred via two angle inference methods using parameter choices as presented in Section VI-C. The coloured shaded areas represent one standard deviation above and below the MAE. The gray shaded indicates the estimated lower bound on the MAE achievable using this evaluation method. This bound arises from the positional uncertainty of the GNSS (≈3m) used to collect the ground truthed path and the positions of the four transmitters. Assuming these errors are independent, their root-sum-ofsquares yields a minimum error of ∼4m.

Below 10 RSS measurements per burst, path reconstruction MAE from AoA observations inferred via peak finding begins to increase sharply as many bursts consisting of so few samples will not contain a measurement within the HPBW of the transmitters’ antenna. This further demonstrates that using the full probability distribution of our AoA predictions at low k values does contribute positively to reconstructing the receiver’s path.

Across all values of k the standard deviation of the MAE is approximately the same, suggesting that the error introduced by increasingly limited RSS sampling affects paths of different shapes equally.

To investigate the distribution of errors in the inferred paths, Fig. 12 presents the cumulative probability distribution of the pointwise error in each inferred path at four selected values of k.

Consistent with the results presented in Fig. 11, both methods infer paths that have 80% of their predictions within around 10m of error at high values of $k .$ The full probability distribution method is largely insensitive to lower values of $k ,$ as at worst 80% of predictions lie within around 16m of error when $k = 3$ . The peak finding method degrades much more severely as k lowers. Both methods have a very small proportion of predictions that exceed 30m of error, explainable by the beginning and end of each reconstructed path where the absence of observations before or after the tracked period causes the posterior to revert towards the prior. The inferred paths such as the example presented in Fig. 10 demonstrate that this is captured by a growth in uncertainty at the start and end of the reconstructed paths.

![](images/9ef94bccdb08e1c4f97777cf3415a13871ef53f5c626e83516417271c0788cc6.jpg)  
Fig. 12. Empirical cumulative distribution plot of point-wise error in meters between all six reconstructed paths and their corresponding GNSS ground truth, from AoA observations inferred using (a) the peak finding method and (b) the full probability distribution method, at four selected values of k. The horizontal axis is truncated at 50m due to the poor performance of the peak finding method at k = 3, approaching a random estimate of the ground truth path, and therefore having a large number of predictions exceed 50m of error.

Based on the performance of the system at different values of k we can also infer the lifespan of the 11mF supercapacitor powered receiver. The DA14531 chip used for the receiver draws 5mA while receiving, and has an operating voltage of between 1.8V and 3.3V in bypass power mode meaning that there is 16.5mC of usable charge. Since a 7ms duty cycle is used to sample every RSS measurement within a 2000ms burst, each RSS measurement contributes approximately 60µW to the average power consumption of the receiver in the case that we are constantly performing 2000ms bursts such as in this experiment. At the minimal k = 3 samples per burst, with an accuracy of around 15m, the system will consume 180µW, powering the receiver for approximately 5 minutes of continuous tracking. At k = 10 samples per burst, where the accuracy is around 10m, the system consumes 600µW and therefore the receiver lifespan reduces to approximately 90 seconds. This highlights that - while the power demand of the system is far lower than GNSS and existing RSS-based tracking methods which require many more samples - varying the duty cycle parameters to perform less frequent burst scans is critical to extend the powered lifespan of the tag.

## C. Trade-offs in Duty Cycle Parameters

We investigate if it is beneficial to take less frequent, more accurate AoA observations (consisting of a higher k number of packets per burst) or a greater number of uncertain AoA observations (lower k) - keeping the total energy used constant, using the experiment presented in Section VI-D. The results on a synthetic path, used for comparison of AoA inference accuracy and path reconstruction accuracy, are shown in Fig. 13.

The results shows that the system is able to reconstruct a path accurate to within ∼10m, aligning with the results in Section VII-B, when given both numerous uncertain AoA observations and a smaller set of more accurate observations. This is important to note, as the powered lifespan of the receiver is limited, and the ability to control the duty cycle parameters is critical for customising the trade-off between the length of the tracked path and its reconstruction accuracy. It is only at the extremes of both cases that path reconstruction accuracy is degraded, though generally it appears to be better to have frequent uncertain AoA observations than too few AoA observations, shown by the sharp increase in error as the number of bursts lowers but the k RSS measurements per burst increases, compared to the still relatively accurate ∼15m MAE at the inverse. This is likely because the extremely small number of AoAs inferred from high values of k within the allowance cannot represent the path of movement that generated them.

![](images/dc12a822de0e2ddf29f7e4166b8e574ab866a8c20332194fdb47eb48ea4b1b59.jpg)  
Fig. 13. Plot of the MAE of AoA inference (blue) and resulting path reconstruction (green) from varying an allowance of 100 RSS measurements across values of k (number of burst scans is 100/k) using synthetic data. The green shaded area is one standard deviation above and below the MAE of the path reconstruction.

A further variable of the duty cycle we can control is the length of the burst (the time over which our k measurements are taken). We repeat the test with four selections of burst length in Fig. 14, which shows that a 2000ms burst window results in the best path and angle reconstruction at low values of k. This is because the transmitters rotate every two seconds, so bursts substantially shorter than this will sometimes not include samples from the period when the highly informative HPBW portion of the transmitter’s ARP is facing towards the receiver. These results are applicable in the case of our transmitter design which rotates at 30 rpm, though different transmitter rotation speeds would necessitate an adjusted burst length.

## D. Bombus Terrestris Flight Tracking Trial

We demonstrate a major use-case of our system, tracking insect flight paths, by tagging bumblebee workers as they returned to their nest and displacing them in order to track their homing flight (see Section VI-E).

Three tagged bees were re-observed within varying windows of time after release, ranging between a few minutes and multiple hours. Fig. 15 presents one of the paths reconstructed from a bee re-observed around 10 minutes post release, showing a clear path the tagged bee took from the release site to the nest. Notably, researchers at the release site observed the tagged bee fly around the copse of trees immediately on release which matches the reconstructed path. The path also shows the tagged subject following the row of trees on the North side of the field, consistent with the well studied behaviour of Bombus terrestris following linear landscape features [72]. Finally the reconstructed path shows the bee return to the nest as observed by fieldworkers when the tagged bee was retrieved at the nest site.

![](images/7e8c98c5e08e8897e3b9742fbc36d82b7397bb904b5a9db38499e2ecc9bca0a1.jpg)  
Fig. 14. Plot of the MAE of AoA inference (blue) and resulting path reconstruction (green) from varying an allowance of 100 RSS measurements across values of k, for several burst scan lengths: (a) 2000ms, (b) 1500ms, (c) 1000ms, (d) 500ms. The green shaded area is one standard deviation above and below the MAE of the path reconstruction.

## VIII. CONCLUSION

We have presented a novel RSS-based method for probabilistically reconstructing the movement path of an ultra lowpower, lightweight receiver across a complex, unknown landscape using simple rotating high-gain transmitters. Through ground-truthed experiments we show ∼15m path reconstruction accuracy is achieved using $< 1 8 0 \mu \mathrm { W }$ with as few as 3 RSS measurements per AoA observation, improving to ∼10m at $< 6 0 0 ~ \mu \mathrm { W }$ with increased RSS measurements per observation.

This is achieved through two contributions, the first being a novel method of AoA inference by modelling RSS measurements from an anisotropic, rotating antenna with a Bayesian approach, allowing for probabilistic predictions of AoA to be made even with very few, noisy RSS measurements. The second is a probabilistic path reconstruction algorithm, able to reconstruct the receiver’s movement path from uncertain, nonconcurrent AoA predictions by modelling the path as a GP and performing inference using doubly stochastic variational inference [67].

We demonstrate a use case of the system by tracking the return flights of displaced Bombus terrestris foragers in a complex environment. The ability to track bee movement and behaviour over a landscape scale is critical in advancing knowledge of behavioural ecology [73] and informing environmental conservation efforts [74], but previous tagged tracking methods [14], [50] have limitations in viability in complex landscapes and tag weight. Our 38mg receiver was light enough to be carried by the tagged forager, was sufficiently powered by its supercapacitor for long enough to capture the bee’s full return flight, and the reconstructed path is consistent with field observations. The receiver is archival, meaning re-observation of the tracked subject is required to read off the received data, making the system particularly well suited to landscape-scale tracking of flying insect central place foragers in complex environments previously inaccessible to researchers.

![](images/cb5ec5ff3ec9dbba8442c5c2c69a15cb7347c6d71b7a77416a1abd3a78b21207.jpg)  
Fig. 15. Reconstructed path (using full distribution AoA method and variational inference as described in Sections IV-A and V) of a tagged Bombus terrestris worker, caught at the Nest Box and displaced & released at the Release Site. The bee after release was re-observed at the Nest Box and the path reconstructed from its retrieved tag successfully shows the bee flying from the Release Site to the Nest Box. The mean of the reconstructed path is represented by the blue line and the shaded blue region is the uncertainty encoded by the reconstructed path’s covariance matrix (<sup>+</sup>2 standard deviations). The three transmitters b,e and f are also shown, overlain onto a satellite image of Common Lane Open Space, Sheffield, UK retrieved from Google Maps [71].

## ACKNOWLEDGMENTS

Special thanks to fieldworkers Euan Emery and Lauren Thomas.

## ETHICS STATEMENT

Under the UK Animals (Scientific Procedures) Act 1986, protected animals are defined as living vertebrates other than man, together with cephalopods. Invertebrates such as Bombus terrestris used in this research fall outside this definition and their use in research does not require a Home Office licence or associated ethical review. The University of Sheffield’s Animal Welfare and Ethical Review Body operates in compliance with this $\operatorname { A c t } ,$ , and accordingly no institutional ethical review was required for this work. Notwithstanding this, we applied the principles of the three-Rs (Replacement, Reduction and Refinement) in our approach to experiments by using only one artificial nest box, handling individuals with care, removing tags from animals following recapture and noting that tagged bees were observed to return to flying and navigation after release.

## REFERENCES

[1] R. Kays, M. C. Crofoot, W. Jetz, and M. Wikelski, “Terrestrial animal tracking as an eye on life and planet,” Science, vol. 348, no. 6240, p. aaa2478, 2015.

[2] T. E. Katzner and R. Arlettaz, “Evaluating contributions of recent tracking-based animal movement ecology to conservation management,” Frontiers in Ecology and Evolution, vol. 7, p. 519, 2020.

[3] H. J. Williams, L. A. Taylor, S. Benhamou, A. I. Bijleveld, T. A. Clay, S. de Grissac, U. Demsar, H. M. English, N. Franconi,ˇ A. Gomez-Laich, R. C. Griffiths, W. P. Kay, J. M. Morales,´ J. R. Potts, K. F. Rogerson, C. Rutz, A. Spelt, A. M. Trevail, R. P. Wilson, and L. Borger, “Optimizing the use of biologgers¨ for movement ecology research,” Journal of Animal Ecology, vol. 89, no. 1, pp. 186–206, 2020. [Online]. Available: https: //besjournals.onlinelibrary.wiley.com/doi/abs/10.1111/1365-2656.13094

[4] M. E. E. Alahi, A. Sukkuea, F. W. Tina, A. Nag, W. Kurdthongmee, K. Suwannarat, and S. C. Mukhopadhyay, “Integration of iot-enabled technologies and artificial intelligence (ai) for smart city scenario: Recent advancements and future trends,” Sensors, vol. 23, no. 11, 2023. [Online]. Available: https://www.mdpi.com/1424-8220/23/11/5206

[5] H. Esmaeili Gorjan and V. P. Gil Jimenez, “Improving indoor wifi´ localization by using machine learning techniques,” Sensors, vol. 24, no. 19, 2024. [Online]. Available: https://www.mdpi.com/1424-8220/ 24/19/6293

[6] K. Romer, “The lighthouse location system for smart dust,” in¨ Proceedings of the 1st international conference on Mobile systems, applications and services, 2003, pp. 15–30.

[7] S. Chen, D. Yin, and Y. Niu, “A survey of robot swarms’ relative localization method,” Sensors, vol. 22, no. 12, p. 4424, 2022.

[8] A. Grenier, E. S. Lohan, A. Ometov, and J. Nurmi, “A survey on lowpower gnss,” IEEE Communications Surveys & Tutorials, vol. 25, no. 3, pp. 1482–1509, 2023.

[9] U-Blox AG, “Ubx-m10150-cc: Ultra-low power gnss receiver for wearables,” https://www.u-blox.com/en/product/ubx-m10150cc-chip, 2026, accessed: 2026-03-28.

[10] R. P. Wilson, K. A. Rose, R. Gunner, M. D. Holton, N. J. Marks, N. C. Bennett, S. H. Bell, J. P. Twining, J. Hesketh, C. M. Duarte, N. Bezodis, M. Jezek, M. Painter, V. Silovsky, M. C. Crofoot, R. Harel, J. P. Y. Arnould, B. M. Allan, D. A. Whisson, A. Alagaili, and D. M. Scantlebury, “Animal lifestyle affects acceptable mass limits for attached tags,” Proceedings of the Royal Society B: Biological Sciences, vol. 288, no. 1961, p. 20212005, 2021. [Online]. Available: https://royalsocietypublishing.org/doi/abs/10.1098/rspb.2021.2005

[11] N. M. Tiglao, M. Alipio, R. D. Cruz, F. Bokhari, S. Rauf, and S. A. Khan, “Smartphone-based indoor localization techniques: State-of-theart and classification,” Measurement, vol. 179, p. 109349, 2021.

[12] K. Zhang, P. Chen, T. Ma, and S. Gao, “On-demand precise tracking for energy-constrained uavs in underground coal mines,” IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1–14, 2022.

[13] M. Wikelski, J. Moxley, A. Eaton-Mordas, M. M. Lopez-Uribe, R. Holland, D. Moskowitz, D. W. Roubik, and R. Kays, “Large-range movements of neotropical orchid bees observed via radio telemetry,” PLoS one, vol. 5, no. 5, p. e10738, 2010.

[14] J. Riley, A. Smith, D. Reynolds, A. Edwards, J. Osborne, I. Williams, N. Carreck, and G. Poppy, “Tracking bees with harmonic radar,” Nature, vol. 379, no. 6560, pp. 29–30, 1996.

[15] J. L. Woodgate, J. C. Makinson, K. S. Lim, A. M. Reynolds, and L. Chittka, “Life-long radar tracking of bumblebees,” PloS one, vol. 11, no. 8, p. e0160333, 2016.

[16] M. Hagen, M. Wikelski, and W. D. Kissling, “Space use of bumblebees (bombus spp.) revealed by radio-tracking,” PloS one, vol. 6, no. 5, p. e19997, 2011.

[17] R. J. Stelzer, L. Chittka, M. Carlton, and T. C. Ings, “Winter active bumblebees (bombus terrestris) achieve high foraging rates in urban britain,” PLoS One, vol. 5, no. 3, p. e9559, 2010.

[18] F. Dressler, M. Mutschlechner, M. Nabeel, and J. Blobel, “Ultra lowpower sensor networks for next generation wildlife monitoring,” 01 2019, pp. 44–48.

[19] T. Nowak, M. Hartmann, L. Patino-Studencki, and J. Thielecke, “Fundamental limits in rssi-based direction-of-arrival estimation,” in 2016 13th Workshop on Positioning, Navigation and Communications (WPNC), 2016, pp. 1–6.

[20] A. Coluccia and F. Ricciato, “On ml estimation for automatic rss-based indoor localization,” in IEEE 5th International Symposium on Wireless Pervasive Computing 2010. IEEE, 2010, pp. 495–502.

[21] S. Hayward, K. van Lopik, C. Hinde, and A. West, “A survey of indoor location technologies, techniques and applications in industry,” Internet of Things, vol. 20, p. 100608, 2022. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2542660522000907

[22] A. Sheinker, B. Ginzburg, N. Salomonski, and A. Engel, “Localization of a mobile platform equipped with a rotating magnetic dipole source,” IEEE Transactions on Instrumentation and Measurement, vol. 68, no. 1, pp. 116–128, 2018.

[23] M. A. G. Al-Sadoon, R. Asif, Y. I. A. Al-Yasir, R. A. Abd-Alhameed, and P. S. Excell, “Aoa localization for vehicle-tracking systems using a dual-band sensor array,” IEEE Transactions on Antennas and Propagation, vol. 68, no. 8, pp. 6330–6345, 2020.

[24] Y. Zou and H. Liu, “A simple and efficient iterative method for toa localization,” 05 2020, pp. 4881–4884.

[25] A. Pourkabirian, F. Kooshki, M. H. Anisi, and A. Jindal, “An accurate rss/aoa-based localization method for internet of underwater things,” Ad Hoc Networks, vol. 145, p. 103177, 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1570870523000975

[26] S. Liu, L. Tang, and Z. Wang, “Fingerprint localization based on hybrid toa, aoa, and rss measurements,” Discover Applied Sciences, vol. 7, 01 2025.

[27] I. O. Nyantakyi, Q. Wan, L. Ni, E. O. Mensah, and O. Bamisile, “Acga: Adaptive conjugate gradient algorithm for non-line-of-sight hybrid tdoaaoa localization,” Measurement, vol. 224, p. 113820, 2024.

[28] F. Bandiera, A. Coluccia, and G. Ricci, “A cognitive algorithm for received signal strength based localization,” IEEE Transactions on signal processing, vol. 63, no. 7, pp. 1726–1736, 2015.

[29] J. Luomala and I. Hakala, “Adaptive range-based localization algorithm based on trilateration and reference node selection for outdoor wireless sensor networks,” Computer Networks, vol. 210, p. 108865, 2022. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S1389128622000743

[30] A. Zanella, “Best practice in rss measurements and ranging,” IEEE Communications Surveys & Tutorials, vol. 18, no. 4, pp. 2662–2686, 2016.

[31] P. Kriz, F. Maly, and T. Kozel, “Improving indoor localization using bluetooth low energy beacons,” Mobile information systems, vol. 2016, no. 1, p. 2083094, 2016.

[32] B. Shin, J. H. Lee, C. Yu, H. Kyung, and T. Lee, “Received signal strength-based robust positioning system in corridor environment,” IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1–15, 2022.

[33] S. Yiu, M. Dashti, H. Claussen, and F. Perez-Cruz, “Wireless rssi fingerprinting localization,” Signal Processing, vol. 131, pp. 235– 244, 2017. [Online]. Available: https://www.sciencedirect.com/science/ article/pii/S0165168416301566

[34] S. Subedi and J.-Y. Pyun, “Practical fingerprinting localization for indoor positioning system by using beacons,” Journal of Sensors, vol. 2017, no. 1, p. 9742170, 2017.

[35] X. Peng, R. Chen, K. Yu, G. Guo, F. Ye, and W. Xue, “A new wifi dynamic selection of nearest neighbor localization algorithm based on rss characteristic value extraction by hybrid filtering,” Measurement Science and Technology, vol. 32, no. 3, p. 034003, 2021.

[36] A. Nagy, T. Bigler, A. Treytl, R. Stenzl, S. Wilker, T. Sauter, and T. Wien, “Rss-based localization for directional antennas,” in 2020 25th IEEE International Conference on Emerging Technologies and Factory Automation (ETFA), vol. 1, 2020, pp. 774–781.

[37] L. Varotto and A. Cenedese, “Transmitter discovery through radio-visual probabilistic active sensing,” in 2021 25th International Conference on Methods and Models in Automation and Robotics (MMAR). IEEE, 2021, pp. 191–196.

[38] N. Honma, R. Tazawa, A. Miura, Y. Sugawara, and H. Minamizawa, “Rss-based doa/dod estimation using bluetooth signal and its application for indoor tracking,” in 2018 International Conference on Indoor Positioning and Indoor Navigation (IPIN). IEEE, 2018, pp. 1–7.

[39] Z. Cai, S. Xing, W. Meng, J. Wang, X. Su, and S. Quan, “Fusion unbiased pseudo-linear kalman filter-based bearings-only target tracking,” Remote Sensing, vol. 16, no. 23, 2024. [Online]. Available: https://www.mdpi.com/2072-4292/16/23/4536

[40] Z. Huang, S. Chen, C. Hao, and D. Orlando, “Bearings-only target tracking with an unbiased pseudo-linear kalman filter,” Remote Sensing, vol. 13, no. 15, 2021. [Online]. Available: https: //www.mdpi.com/2072-4292/13/15/2915

[41] M. Qian, W. Chen, and R. Sun, “A maneuvering target tracking algorithm based on cooperative localization of multi-uavs with bearing-only measurements,” IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1–11, 2024.

[42] J. Tamura and H. Arai, “Simple and accurate received signal strengthbased localization using compact null-steering antennas,” IEEE Antennas and Wireless Propagation Letters, vol. 22, no. 2, pp. 417–421, 2023.

[43] B. N. Hood and P. Barooah, “Estimating doa from radio-frequency rssi measurements using an actuated reflector,” IEEE Sensors Journal, vol. 11, no. 2, pp. 413–417, 2010.

[44] J.-R. Jiang, C.-M. Lin, F.-Y. Lin, and S.-T. Huang, “Alrd: Aoa localization with rssi differences of directional antennas for wireless sensor networks,” International Journal of Distributed Sensor Networks, vol. 9, no. 3, p. 529489, 2013. [Online]. Available: https://doi.org/10. 1155/2013/529489

[45] M. Malajner, D. Gleich, and P. Planinsiˇ c, “Angle of arrival measurementˇ using multiple static monopole antennas,” IEEE Sensors Journal, vol. 15, no. 6, pp. 3328–3337, 2015.

[46] M. Jais, P. Ehkan, R. B. Ahmad, I. Ismail, T. Sabapathy, and M. Jusoh, “Review of angle of arrival (aoa) estimations through received signal strength indication (rssi) for wireless sensors network (wsn),” in 2015 International Conference on Computer, Communications, and Control Technology (I4CT). IEEE, 2015, pp. 354–359.

[47] N. Duda, T. Nowak, M. Hartmann, M. Schadhauser, B. Cassens, P. Wagemann, M. Nabeel, S. Ripperger, S. Herbst, K. Meyer-Wegener¨ et al., “Bats: Adaptive ultra low power sensor network for animal tracking,” Sensors, vol. 18, no. 10, p. 3343, 2018.

[48] Y. Li, K. Yan, Z. He, Y. Li, Z. Gao, L. Pei, R. Chen, and N. El-Sheimy, “Cost-effective localization using rss from single wireless access point,” IEEE Transactions on Instrumentation and Measurement, vol. 69, no. 5, pp. 1860–1870, 2019.

[49] Q. Zheng, L. Luo, H. Song, G. Sheng, and X. Jiang, “A rssi-aoa-based uhf partial discharge localization method using music algorithm,” IEEE Transactions on Instrumentation and Measurement, vol. 70, pp. 1–9, 2021.

[50] K. E. Fisher, P. M. Dixon, G. Han, J. S. Adelman, and S. P. Bradbury, “Locating large insects using automated vhf radio telemetry with a multiantennae array,” Methods in Ecology and Evolution, vol. 12, no. 3, pp. 494–506, 2021.

[51] L. Varotto, A. Cenedese, and A. Cavallaro, “Active sensing for search and tracking: A review,” 2021. [Online]. Available: https://arxiv.org/abs/2112.02381

[52] T. Burton and K. Rasmussen, “Orientation estimation using wireless device radiation patterns,” arXiv preprint arXiv:2203.10052, 2022.

[53] P. Ghosh, J. A. Tran, and B. Krishnamachari, “Arrest: A rssi based approach for mobile sensing and tracking of a moving object,” IEEE Transactions on Mobile Computing, vol. 19, no. 6, pp. 1260–1273, 2019.

[54] J. Graefenstein, A. Albert, P. Biber, and A. Schilling, “Wireless node localization based on rssi using a rotating antenna on a mobile robot,” in 2009 6Th Workshop on positioning, navigation and communication. IEEE, 2009, pp. 253–259.

[55] Renesas Electronics, “Smartbond tiny™ da14531: World’s smallest and lowest power bluetooth 5.1 soc,” https://www.renesas.com/en/products/ da14531, 2026, accessed: 2026-03-31.

[56] CPH3225A Chip Capacitor (Electric Double Layer Capacitor) Datasheet, https://www.sii.co.jp/en/me/datasheets/chip-capacitor/ cph3225a/, Seiko Instruments Inc., 2025, accessed: 2025-11-28.

[57] Pololu Corporation, “150:1 metal gearmotor 37dx73l mm 12v with 64 cpr encoder (helical pinion),” https://www.pololu.com/product/2828, 2026, item no. 2828, Accessed: 2026-03-28.

[58] Fanstel Corp., “Bt832xe: The longest range bluetooth 5 module with u.fl connector,” https://www.fanstel.com/ bt832xe-the-longest-range-bluetooth-5-module-external, 2026, based on Nordic nRF52832; Accessed: 2026-03-28.

[59] Y. Kim, H. Shin, Y. Chon, and H. Cha, “Smartphone-based wi-fi tracking system exploiting the rss peak to overcome the rss variance problem,” Pervasive and Mobile Computing, vol. 9, no. 3, pp. 406–420, 2013.

[60] C. E. Rasmussen, “Evaluation of gaussian processes and other methods for non-linear regression,” Ph.D. dissertation, University of Toronto Toronto, Canada, 1997.

[61] M. T. Smith, M. A. Alvarez, and N. D. Lawrence, “Gaussian process regression for binned data,” arXiv preprint arXiv:1809.02010, 2018.

[62] D. M. Blei, A. Kucukelbir, and J. D. McAuliffe, “Variational inference: A review for statisticians,” Journal of the American statistical Association, vol. 112, no. 518, pp. 859–877, 2017.

[63] T. Salimans, D. Kingma, and M. Welling, “Markov chain monte carlo and variational inference: Bridging the gap,” in International conference on machine learning. PMLR, 2015, pp. 1218–1226.

[64] T. P. Minka, “Expectation propagation for approximate bayesian inference,” arXiv preprint arXiv:1301.2294, 2013.

[65] J. Quinonero-Candela and C. E. Rasmussen, “A unifying view of sparse approximate gaussian process regression,” The Journal of Machine Learning Research, vol. 6, pp. 1939–1959, 2005.

[66] A. K. Uhrenholt, V. Charvet, and B. S. Jensen, “Probabilistic selection of inducing points in sparse gaussian processes,” in Uncertainty in Artificial Intelligence. PMLR, 2021, pp. 1035–1044.

[67] M. Titsias and M. Lazaro-Gredilla, “Doubly stochastic variational´ bayes for non-conjugate inference,” in Proceedings of the 31st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, E. P. Xing and T. Jebara, Eds., vol. 32, no. 2. Bejing, China: PMLR, 22–24 Jun 2014, pp. 1971–1979. [Online]. Available: https://proceedings.mlr.press/v32/titsias14.html

[68] R. Frostig, M. Johnson, and C. Leary, “Jax: composable transformations of python+numpy programs,” GitHub repository, 2018. [Online]. Available: https://github.com/google/jax

[69] D. Goulson and J. C. Stout, “Homing ability of the bumblebee bombus terrestris (hymenoptera: Apidae),” Apidologie, vol. 32, no. 1, pp. 105– 111, 2001.

[70] Biobest Group NV, User Guide for Biobest Bumblebee Hives: Premium, Standard, and Medium, Biobest Group NV, Westerlo, Belgium, 2023, technical Manual for Bombus terrestris Commercial Hives. [Online]. Available: https://www.biobestgroup.com/

[71] Google, “Google Maps,” https://maps.app.goo.gl/, 2026.

[72] J. S. Brebner, J. C. Makinson, O. K. Bates, N. Rossi, K. S. Lim, T. Dubois, T. Gomez-Moracho, M. Lihoreau, L. Chittka, and J. L.´ Woodgate, “Bumble bees strategically use ground level linear features in navigation,” Animal Behaviour, vol. 179, pp. 147–160, 2021.

[73] R. Trappes, “How tracking technology is transforming animal ecology: epistemic values, interdisciplinarity, and technology-driven scientific change,” Synthese, vol. 201, no. 4, p. 128, 2023.

[74] S. Chowdhury, V. K. Dubey, S. Choudhury, A. Das, D. Jeengar, B. Sujatha, A. Kumar, N. Kumar, A. Semwal, and V. Kumar, “Insects as bioindicator: A hidden gem for environmental monitoring,” Frontiers in Environmental Science, vol. 11, p. 1146052, 2023.

## SUPPLEMENTARY MATERIAL

## SI. MARGINALISING OUT AN IMPROPER RANDOMVARIABLE MEAN OF A GAUSSIAN THAT LIES ON ASTRAIGHT INFINITE LINE

Let y be a random variable, normally distributed with isotropic variance $\sigma ^ { 2 }$ and mean ${ \mathbf { } } ^ { \mathbf { } } \mathbf { { x } } ,$

$$
p ( { \pmb y } | { \pmb x } ) = N ( { \pmb y } | { \pmb x } , \sigma ^ { 2 } I ) .\tag{S22}
$$

We wish to marginalise out x which is a random variable defined using an improper prior such that it must lie upon a line that passes through the origin. To extend this to a line passing through any point, p, simply repeat the derivation with a new RV equal to $\mathbf { \nabla } _ { \boldsymbol { y } } - \mathbf { \nabla } _ { \boldsymbol { p } }$

So $\mathbf { \boldsymbol { x } } = \beta \hat { \mathbf { { x } } }$ where xˆ is a unit vector and $\beta$ is a random variable $\beta \sim U n i f o r m ( - \infty , \infty )$ . We define d as the shortest distance between y and the line defined by $\beta \hat { \mathbfit { x } }$ , shown in Fig. S1.

We wish to show that the normalised marginal is,

$$
p ( { \pmb y } ) = N ( d | 0 , \sigma ^ { 2 } ) .\tag{S23}
$$

This is equivalent to,

$$
\int _ { - \infty } ^ { \infty } p ( { \pmb y } | \beta { \hat { \pmb x } } , \sigma ^ { 2 } I ) p ( \beta ) d \beta = N ( d | 0 , \sigma ^ { 2 } ) .\tag{S24}
$$

![](images/e2f5dad398b16c64240f321dfc96d713e9e025b3e5a0f4b400f83f77c548ddb7.jpg)  
Fig. S1. An illustration of the distance d between the random variable y and the line defined by xˆ.

We first write down $\pmb { y } = \alpha \hat { \pmb { x } } + \pmb { v }$ where αxˆ is the point on the line βxˆ that is closest to y as shown in Fig. S2. Therefore $\hat { \pmb x } ^ { \top } \pmb v = 0$ by definition.

![](images/1f51d3341142d6472a71c04d4a5123b14936a81b66d6af13b511b81a7b448ed3.jpg)  
Fig. S2. An illustration of the the point αxˆ on the line defined by xˆ that is closest to y.

Our probability density $p ( \pmb { y } | \pmb { x } )$ (which we will write $p ( \pmb { y } | \beta )$ to explicitly emphasise the dependence on the location along the line) is now,

$$
p ( \pmb { y } | \beta ) \propto \mathrm { e x p } \left( \frac { - ( \alpha \hat { \pmb { x } } + \pmb { v } - \beta \hat { \pmb { x } } ) ^ { T } ( \alpha \hat { \pmb { x } } + \pmb { v } - \beta \hat { \pmb { x } } ) } { 2 \sigma ^ { 2 } } \right) .\tag{S25}
$$

By also noting that $\hat { \pmb x } ^ { T } \hat { \pmb x } = 1$ and expanding out the product

we obtain,

$$
p ( \pmb { y } | \beta ) \propto \frac { - \big ( \alpha ^ { 2 } - 2 \alpha \beta + \beta ^ { 2 } + | \pmb { v } | _ { 2 } ^ { 2 } \big ) } { 2 \sigma ^ { 2 } }\tag{S26}
$$

So the probability density is,

$$
p ( \pmb { y } | \beta ) \propto \exp \left( \frac { - ( \alpha - \beta ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) \ \exp \left( \frac { - | \pmb { v } | _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \right) .\tag{S27}
$$

We can now marginalise (integrate out) $\beta .$ We note that it has an improper (constant prior),

$$
p ( \pmb { y } ) \propto \int _ { - \infty } ^ { \infty } e ^ { \frac { - ( \alpha - \beta ) ^ { 2 } } { 2 \sigma ^ { 2 } } } d \beta \times e ^ { \frac { - | \pmb { v } | _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } } .\tag{S28}
$$

We moved the second exponential of Equation (S27) outside the integral as it did not contain $\beta .$

This is invariate w.r.t. α as this is an integral over infinity (to see this, consider a change of variables, let $\epsilon = \beta - \alpha$ , so $d \epsilon / d \beta = 1$ , the integral becomes $\begin{array} { r } { \int _ { - \infty } ^ { \infty } e ^ { - \epsilon ^ { 2 } / 2 \sigma ^ { 2 } } d \epsilon = \mathrm { c o n s t . } ) } \end{array}$ We note that $| \pmb { v } | _ { 2 } ^ { 2 } ~ \mathrm { i s } ~ d ^ { 2 }$ so this is $\propto N ( d | 0 , \sigma ^ { 2 } )$

## A. Applying this result to integrating out attenuation

Let d be the distance between our vector of observed signal strengths y and a line L in direction of ${ \bf 1 } _ { k }$ (a vector of ones, of length k) passing through the transmitter’s predicted (unattenuated) signal strengths t for a given value of θ. This line represents all possible expected signal strengths obtained by shifting t by the unknown attenuation a. d measures how far observations $\textbf {  { y } }$ lie from the attenuated version of t that lies closest to y.

To compute $d ,$ we define the difference vector $\pmb { v } = \pmb { t } - \pmb { y }$ and remove its component along L. This is done by projecting v onto L and subtracting it from y, leaving a vector perpendicular to the line. The distance is therefore,

$$
d = \left\| v - { \frac { v ^ { \top } \mathbf { 1 } _ { k } } { k } } \mathbf { 1 } _ { k } \right\| .\tag{S29}
$$

To summarise: We wish to integrate out a from $p ( \pmb { y } | \pmb { t } \ +$ $a \mathbf { 1 } _ { k } , \sigma ^ { 2 } )$ . The shortest distance between the line defined by $\mathbf { \Delta } t + a \mathbf { 1 } _ { k }$ is d. As shown in Supplementary Material SII this means that, if $a \sim U n i f o r m ( - \infty , \infty )$ , then $p ( \pmb { y } | \pmb { t } +$ $a \mathbf { 1 } _ { k } , \sigma ^ { 2 } ) = p ( d | 0 , \sigma ^ { 2 } )$

Finally, to compute d in a more convenient form we note that the second term of Equation (S29) is the mean average of v as the product ${ \pmb v } ^ { \top } { \bf 1 } _ { k }$ simply sums v, and we divide by its length k. Substituting the definition of v in means that d is simply,

$$
d = \left\| \begin{array} { c } { { \left[ \left( t _ { 1 } - \bar { t } \right) - \left( y _ { 1 } - \bar { y } \right) \right] } } \\ { { \left( t _ { 2 } - \bar { t } \right) - \left( y _ { 2 } - \bar { y } \right) } } \\ { { \vdots } } \\ { { \left( t _ { k } - \bar { t } \right) - \left( y _ { k } - \bar { y } \right) } } \end{array} \right\| .\tag{S30}
$$

Hence, in Section IV-A, we define $d = | | ( { \pmb t } - { \bar { \pmb t } } ) - ( { \pmb y } - { \bar { \pmb y } } ) | | _ { 2 }$

## SII. ASSUMPTIONS NEEDED TO ENSURE THE LOG LIKELIHOOD OF VECTOR OBSERVATIONS IS APPROXIMATELY VALID

Ideally (with f fixed), $\begin{array} { r } { \int p ( \pmb { y } | \pmb { f } ) , d \pmb { y } = \ 1 } \end{array}$ . However, the integral over y is over a unit circle while the likelihood is a distribution over the length of the cross product with $f ,$ so clearly it doesn’t integrate to one. One could perform a change-of-variable operation although this is complicated by the awkward 1-to-2 variable mapping (from distance to vector) via the 2D cross product. Instead we show that the likelihood function does approximate a conditional distribution under some reasonable assumptions.

If one assumes w.l.o.g. that f lies along the y-axis, and the standard deviation (σ) of our distribution over the error distance’ (d) is much less than the (predicted) distance of the receiver from the transmitter $( f _ { y } )$ , then one can approximate the vector from the predicted location $f { - } c$ to the nearest point in the y-direction as a 1-D scalar $a _ { x }$ whose magnitude equals d. There is one angle to specify the direction of ${ \mathbf { } } ^ { \mathbf { \Gamma } } \mathbf { ^ { \mathbf { 3 } } } \mathbf { \Gamma } ^ { \mathrm { ~ \mathrm { ~ } ~ } }$ sin $\theta =$ $a _ { x } / f _ { y }$ . One can use the small-angle approximation to write that $f _ { y } \theta \approx a _ { x }$ . It is required that $\overline { { C \int _ { - \infty } ^ { \infty } e ^ { - a _ { x } ^ { 2 } / ( 2 \sigma ^ { 2 } ) } d a _ { x } = 1 } }$ (for some normalising constant C). If one were to assume that this equality is approximately true over a small domain $( - r \ < \ a _ { x } \ < \ r )$ , then one can immediately see that if $a _ { x }$ were replaced with $f _ { y } \theta$ one will still have a Gaussian, albeit with a different normalising constant (equal to $C f _ { y } )$ . This would appear to be a problem (the conditional distribution of y given f should integrate to one). However, the computation of the ELBO uses the expectation of the $l o g$ likelihood, and so the normalising term becomes an additive constant. Problematically though this constant’ is now proportional to $f _ { y } ,$ which compromises the assumptions around the Monte Carlo approximation to the expectation. It is expected that the range of values for $f _ { y }$ for a given computation of the expectation should be narrow (otherwise it would suggest there is no capability for estimating the receiver’s location), so I propose that one can approximate that term with a single constant.

## SIII. PATH RECONSTRUCTION SAMPLE RESULTS

Presented in the following Figures in this section are the three 4 minute long paths that were used in assessing the path reconstruction algorithm in Section VI-C The GNSS paths have been presented alongside a path reconstructed from AoA observations produced from the peak finding and full probability distribution methods, via the variational inference method and parameters used in Section VI-C, across a selection of values of k RSS measurements per burst.

The first path is presented in Fig. S3, the second path is presented in Fig. S4 and the third path is presented in Fig. S5, the fourth path is presented in Fig. S6, the fifth path is presented in Fig. S7 and the sixth path is presented in Fig. S8,

![](images/6ffa3b5535890939cac928e12c8e9e68034a9cbd396c9d0404676e3cc393ca32.jpg)

![](images/682c4a7b13a3431ef7e8825c017cc5e1eb4e3fc9580cdc4b3b80ef0c75cca1ff.jpg)

![](images/e5422d7657f7dbd00b921a1a5d0ad4d7d961dd8c16265bd72d293efb8007de86.jpg)

![](images/9e9bd068ff03a3493dababcdf1fb4a2d3d42cf44d5f55d2121243b2749172dc3.jpg)

![](images/0a93201ed2f26d0cade4042959fd919d58e10ebe99e6c8959f6faa4b414c3a7d.jpg)

![](images/8e8bb4f77b1c6e3ab692d5dc919b4a73e4293e5e6263934c25bddd4febf569ab.jpg)

![](images/c804d1d052b59fb275ec2cf2b75ecdbb561621a9376e4300199f57e63d8990e3.jpg)

![](images/c82674e0f46b9bf9d24bfacead3e0dc02180689facab46e9b005baec8db3d333.jpg)  
Fig. S3. Plot of path one’s GNSS ground truth (blue) and a reconstructed path’s mean (orange) and standard deviation of its covariance matrix at 20 evenl spaced time points across the path (black ellipses). The transmitters a,b,c,e are plotted and labelled. Path reconstructions using k = 3, 5, 10 & 25 packets per burst were used alongside the peak finding (left) and full probability distribution (right) AoA inference methods

![](images/454e8ef8c3ca67d202ab14a10c973076f4da8ab22f1d09f880c9702ab5fedbf0.jpg)

![](images/7aa645d2575d9193edaca0e6f6fcfb091f716f15e8a4c4a28d62a80e028f3c5f.jpg)

![](images/cb4f0968be9f2a4eca16cb25ba938da3a70db79ae3920d9d1c5b2e07da9fe1df.jpg)

![](images/50ec55d9ee0dc0cfa035bd14f024dad497c0f9651a8463a34f6d5013590c2d18.jpg)

![](images/ac0a3864c95259110b2328b99c78aa0e5f9626d60c401965a08cc660b0e17bca.jpg)

![](images/2f6e0668160ce2d72baaa482334046c79ad8f44f3da856563c4397178171a098.jpg)

![](images/276076f6ad96e55fcfec68556643975ab9af592fb78553b76dfe71aec611815b.jpg)

![](images/4adaa1e7dbffd2d2730040686c0365688e2d81ffaf72765a10602eb991c5d55f.jpg)  
Fig. S4. Plot of path two’s GNSS ground truth (blue) and a reconstructed path’s mean (orange) and standard deviation of its covariance matrix at 20 evenl spaced time points across the path (black ellipses). The transmitters a,b,c,e are plotted and labelled. Path reconstructions using k = 3, 5, 10 & 25 packets per burst were used alongside the peak finding (left) and full probability distribution (right) AoA inference methods

![](images/ad8769d18c5df460ffd4128d54c4bc0835141699a14c94213f209e9c9583a4da.jpg)

![](images/9eba18a3f42568fef62884186f913a392529d78b37eb60aaebb1e31b654daf4f.jpg)

![](images/40d6c649ba898d941df75e26d98821860ebe70fc2a97b9364b3abe1da514d340.jpg)

![](images/219c696d7531a7f779619ede11fc07bdc2972dd158d5484f6d7062b60b821606.jpg)

![](images/60f21ddd3b437d3073bcbb694f3530e0d962ee976f9c24a725e0273b2c9b13cd.jpg)

![](images/078252329d390ae17fa2f4a7a5cbdd1c312c6ef5a890aa73e1f0e6148c5eb3f7.jpg)

![](images/8b1d17cfafdb160de0bd32fbd8435e41343a84fa41406847be75ec2985e2f55a.jpg)

![](images/de53b63a05665c1ac96e6fa802b8bdfa0c5a0520b92d8766a23739d42386c227.jpg)  
Fig. S5. Plot of path three’s GNSS ground truth (blue) and a reconstructed path’s mean (orange) and standard deviation of its covariance matrix at 20 evenl spaced time points across the path (black ellipses). The transmitters a,b,c,e are plotted and labelled. Path reconstructions using k = 3, 5, 10 & 25 packets per burst were used alongside the peak finding (left) and full probability distribution (right) AoA inference methods

![](images/305701c65631348a64da9c52299e9999af56dc01ab0abff61d765d605673a83f.jpg)

![](images/e55b7a78d23d3b37f666c30de08b784368a44368195eb7192031d0294b5db28e.jpg)

![](images/f0477ec1020d79711dda201224a37ae134abf74d147686609ab7fb16a5f0b79c.jpg)

![](images/0b3095b55ffe103786d4ef831ec78f03212bb8882212cb033414bb3f33698912.jpg)

![](images/215ff6d96f52c7f94684e779c15bd219c170ee5304619ba9a9cbc52341b51af8.jpg)

![](images/39d1c2e8d600fe22afc19ca15b50e6647f54213ea200b4ecc2195c42f9fcabd5.jpg)

![](images/290927f1f85ac5d98ccfe59c6594d37f08f8bc6295f569aa2c35cdf206b339b0.jpg)

![](images/2ae712174bd5635c7af4f4a2fb42d599d20f0b1818709141a79db69b06abaaf8.jpg)  
Fig. S6. Plot of path four’s GNSS ground truth (blue) and a reconstructed path’s mean (orange) and standard deviation of its covariance matrix at 20 evenly spaced time points across the path (black ellipses). The transmitters a,b,c,e are plotted and labelled. Path reconstructions using k = 3, 5, 10 & 25 packets per burst were used alongside the peak finding (left) and full probability distribution (right) AoA inference methods

![](images/44b1fa6c29567f2d1dbd1fdddc405a9d93b4f7557c6d32c12cefb98b09846699.jpg)

![](images/86160031d3c9838fe3578abbf1478633f9363d808a34a88b1bb5c7f04af6ccb7.jpg)

![](images/aa8079fc3eb93167b3909bb20f536d8dc33ce5055ab3330563af4071c6b5bfe9.jpg)

![](images/d7591e3960755b2dfb50d7212d4d1c0ae1ef5c6eeb36d4eb21cf882195642528.jpg)

![](images/8d594ccd53633101c61f0dc0fafedab2f9c4cd1d38fa04a40283cf28cfe92b03.jpg)

![](images/de2b5bc990936559296b81f4963ed3c7c7f4d22bf2e98d33fde941a5c75974ea.jpg)

![](images/2bffb42ddf61802ca4d89ed895b12ceac71c48747150676ae3f9ea33b76b70f5.jpg)

![](images/1fbcd299cf7cdb807af07623abcc5e1faa39ec606917a313781747ca476ed837.jpg)  
Fig. S7. Plot of path five’s GNSS ground truth (blue) and a reconstructed path’s mean (orange) and standard deviation of its covariance matrix at 20 evenly spaced time points across the path (black ellipses). The transmitters a,b,c,e are plotted and labelled. Path reconstructions using k = 3, 5, 10 & 25 packets per burst were used alongside the peak finding (left) and full probability distribution (right) AoA inference methods

![](images/c15d721795b8b6f43d69e536f50d89bef75870b85c96b941991f0d854e44daa7.jpg)

![](images/f55556ea3256079b548a452e02033511d092e13121e35b1b95b8483319f924bd.jpg)

![](images/09eef0136154dd1380884765fd54df46dde5ef725735fff78a82747b776888ea.jpg)

![](images/8eee3d070dea7f3a885a9b070e53952b41512bf57e7aacf2f4381b266eaa579f.jpg)

![](images/43c5f657b52ffc921a01a319ddf3752db59cb3e836f850a140b2780f545f58f1.jpg)

![](images/b9d77ce28195897709ce54a2d3d9e89c62f34d40b3f9cd59aa1bdcb58361bf29.jpg)

![](images/4e870211bbc326f7cafdd91b34786118f4d74ef658d9df1b212976380b639dc1.jpg)

![](images/edabf467b794ca73c829a54e64ef13298984eda06d7c2cbb34d7e256086270bd.jpg)  
Fig. S8. Plot of path six’s GNSS ground truth (blue) and a reconstructed path’s mean (orange) and standard deviation of its covariance matrix at 20 evenly spaced time points across the path (black ellipses). The transmitters a,b,c,e are plotted and labelled. Path reconstructions using k = 3, 5, 10 & 25 packets per burst were used alongside the peak finding (left) and full probability distribution (right) AoA inference methods