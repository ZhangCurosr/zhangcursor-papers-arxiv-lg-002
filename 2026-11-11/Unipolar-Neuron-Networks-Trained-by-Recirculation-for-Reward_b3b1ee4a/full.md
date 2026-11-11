# Unipolar Neuron Networks Trained by Recirculation for Reward-Based Continual Learning

Florin Leon

florin.leon@academic.tuiasi.ro

Faculty of Automatic Control and Computer Engineering

“Gheorghe Asachi” Technical University of Ias<sub>,</sub>i

Bd. Mangeron 27, 700050 Iasi, Romania

## Abstract

Unipolar neurons are the predominant type of nerve cell in the octopus and in most other invertebrates. A single neurite (a fiber extending from the cell body) branches into a tree that carries both input and output synapses. The branches send graded local outputs to nearby cells, and one site on the main neurite sends a global output to distant targets. This paper proposes a network of such neurons and a learning rule for it. Each neuron is modeled as a tree of passive segments, whose activations are computed exactly as the steady state of a linear system. The network learns from reward alone, with a rule called recirculation. After each forward pass, a second pass sends requested changes of the outputs back through the same connections. Each synapse then adjusts its weight using only quantities available at its own location. Because the system of each neuron is symmetric, this update closely follows the gradient of the error in the predicted reward. Two mechanisms limit forgetting when tasks are learned in sequence: a task code inhibits a diferent subset of branches in each task, while consolidation lowers the learning rate of synapses that changed strongly in the recent past. The model is evaluated on a sequence of MNIST and Iris tasks. The network maintains high accuracy with low forgetting, whereas gradient-based training of the same network forgets earlier tasks almost completely.

Keywords: unipolar neurons, octopus neurons, dendritic computation, recirculation, reward-based learning, continual learning, shunting inhibition, metaplasticity

## 1 Introduction

Most artificial neural networks model a neuron as a single point. The weighted inputs are summed and passed through a nonlinearity, and every target receives the same output. Biological neurons receive their inputs at diferent points of a dendritic tree. The position of a synapse afects how its input interacts with other inputs. In unipolar neurons, the predominant type of nerve cell in the octopus and in most other invertebrates, a single neurite (a fiber extending from the cell body) branches into a tree. Input and output synapses lie side by side on the same branches. Action potentials start at one site of the main neurite and travel along the axon to distant targets. A unipolar neuron therefore has many graded local outputs, each of which mainly reflects the inputs to its own branch, and one global output that combines the contributions of all branches.

This paper applies the unipolar organization to continual learning, in which a network learns a sequence of tasks without access to the data of previous tasks. Networks trained by gradient descent tend to lose earlier tasks when they learn new ones, a problem known as catastrophic forgetting. Most methods that reduce forgetting add a separate component to the learning algorithm, such as a penalty on weight changes, a mask on the gradients or a bufer of stored examples. The network considered here learns without labels. At each step, it chooses an action and receives a reward that indicates whether the action was correct.

The proposed model is a network of unipolar neurons. Each neuron is a tree of passive segments, whose activations are the steady state of a linear system. The network learns with a two-pass rule called recirculation. The first pass computes the outputs and selects an action. The second pass sends requested changes of the outputs back through the same connections. Each synapse then changes its weight according to the quantities available at its own location. Because the system of each neuron is symmetric, this rule closely follows the gradient of the error in the predicted reward. Forgetting is limited by two mechanisms. A task code inhibits a diferent subset of branches at their base in each task. Consolidation lowers the learning rate of each synapse according to its recent changes. The inhibition of a branch has two efects. It reduces the contribution of the branch to the global output, so that each task uses partly distinct branches. It also attenuates the requested changes that reach the branch in the second pass, so that the weights learned in other tasks are largely preserved. Much of the protection thus follows from the structure of the neuron.

The main contributions of the paper are the following:

• A neuron model based on the organization of unipolar neurons, with graded local outputs, one global output and shunting or divisive inhibition, which scales down the efect of excitation at the same location. A single neuron of this type can compute the XOR function of two inputs, which a perceptron cannot;

• Recirculation, a learning rule that uses only the reward and the quantities available at each synapse, and an analysis of its relation to the gradient;

• Two mechanisms for continual learning, namely a task code that acts through dendritic inhibition and a consolidation rule that each synapse computes from its own history. Neither mechanism requires stored data or copies of earlier weights;

• An experimental evaluation on a sequence of MNIST tasks and on the Iris dataset, including a comparison with gradient-based training of the same network, an analysis of the balance between learning new tasks and retaining earlier ones, and ablations of the main elements of the model.

The remainder of the paper is organized as follows. Section 2 reviews related work on unipolar neurons, biologically plausible learning rules and methods for continual learning. Section 3 describes the neuron model, the network structure, the learning rule, the mechanisms for continual learning, and some variants used for comparison. Section 4 presents the experimental results on MNIST and Iris, followed by a discussion of the findings. Section 5 concludes the paper and outlines directions for future work.

The implementation is available at: https://github.com/florinleon/OctopusNeuron.

## 2 Related Work

## 2.1 Unipolar Neurons

To the best of our knowledge, unipolar neurons have not been used so far as units of machine learning models. In the neural network literature, the term unipolar usually describes units whose activation lies between 0 and 1. Neuroscience studies have, however, built compartmental models of unipolar neurons in insects. A compartmental model divides a neuron into small connected segments and computes the voltage (activation) of each segment from its inputs and its neighbors, as the model of Section 3 does. These studies examine whether a neuron behaves as one unit or as several partly independent units, and how inhibition and the location of the inputs afect its output.

Meier and Borst (2019) studied a neuron of the fruit fly (Drosophila melanogaster) that receives visual input from the entire eye. The neuron has a separate terminal for each small region of the visual field, and each terminal both receives input from that region and sends output back to it. Recordings of activity showed that each terminal responds to stimuli in its own region, independently of the neighboring terminals. A compartmental model indicated that the thin connections between the terminals make this independence possible. A single neuron can therefore act as many local units, each with its own inputs and outputs.

Amin et al. (2020) studied a single inhibitory neuron in a brain region of the fruit fly that is involved in learning odors. This neuron receives input from a large population of neurons and inhibits the same population in return. It communicates through continuous changes of its voltage, without spikes. Activity evoked at one location decayed with distance along its branches, and the inhibition was strongest near that location. A model of this decay predicted that each neuron of the population is inhibited more by its own activity than by that of any other single neuron. One inhibitory neuron thus acts as many local feedback loops.

Liu et al. (2022) also built compartmental models of fruit fly neurons that receive olfactory information. Their shapes and synapse positions came from the connectome, i.e., a map of synaptic connections reconstructed from electron microscopy images. Models with many compartments matched the measured responses better than models that treat each neuron as a point. Some of these neurons have two groups of branches joined by a long, thin cable. Because the cable attenuates signals, each group weights its inputs with its own gain while sharing part of its activity with the other. The authors suggested that this partial separation allows computations within a single neuron.

Hafez et al. (2023) built a compartmental model of an output neuron in the same learning region of the fruit fly, again based on the connectome. Voltages difered strongly between the branches of the neuron, but every synapse had nearly the same efect on its output. The neuron therefore sums its inputs regardless of where they arrive. The authors argued that this property makes the readout of stored memories robust to the random wiring of the inputs. Together with the previous study, this result indicates that the shape of the tree determines whether a neuron combines its inputs locally or globally.

Groschner et al. (2022) explained how a motion-sensitive neuron of the fruit fly computes a product of its inputs, as classic models of motion detection require. Its excitatory and shunting inhibitory inputs arrive at diferent locations of its dendrite. Shunting inhibition opens channels in the membrane with little change of its voltage, so it divides the efect of excitation. When a moving stimulus removes this inhibition as excitation arrives, the response exceeds the sum of the two separate efects. A passive model reproduced this multiplication without any threshold. The preference for one direction of motion follows from the diferent arrival times of the spatially separated inputs.

Dewell et al. (2022) studied a large neuron of the grasshopper that detects approaching objects and produces its spikes at one site of its main branch. Inputs that signal darkening and inputs that signal brightening arrive on two separate fields of its dendritic tree, one far from this site and one close to it. A biophysical model showed that the inputs of the closer field are attenuated less on their way to the site, so fewer synapses are needed to produce spikes. The location of a group of inputs within the tree is thus part of the computation of the neuron.

Lappalainen et al. (2024) built a network model of the motion-detecting part of the fruit fly visual system, with connections taken from the connectome. Each neuron is a single compartment that produces graded outputs and no spikes. The parameters of the neurons and the strengths of the synapses were trained by backpropagation, so that the network estimates the motion in videos. The trained network predicted the responses of many types of neurons, and these predictions agreed with experimental recordings. The study shows that connectivity taken from biology, combined with optimization for a task, can reproduce the function of real neurons.

The present paper difers from these studies, which either analyze individual neurons without learning or train networks of single-compartment neurons. This paper proposes a model of such a neuron as the unit of a network that learns a sequence of tasks from reward.

## 2.2 Continual Learning and Reinforcement Learning

Khetarpal et al. (2022) reviewed continual reinforcement learning, which they framed as learning from reward in environments that change over time. They classified the settings by the scope of the change, for example whether it afects the rewards or the dynamics of the environment, and by the factors that drive it. They also surveyed the main families of methods, the benchmarks and the evaluation metrics. The review discusses open problems, such as the balance between retaining earlier knowledge and adapting to new conditions. It also relates continual learning in artificial agents to findings on learning and memory in animals.

Kaplanis et al. (2018) replaced the scalar weights of learning agents with a model of complex synapses. In this model, each synapse has several hidden variables that interact and change on diferent timescales, so its efective plasticity depends on its history. The authors applied the synaptic model to tabular and deep reinforcement learning agents. It reduced forgetting when two simple tasks were learned in sequence, without any information about task boundaries. Within a single task, it also reduced the need for an experience replay bufer, a memory of past transitions that is used for repeated training.

Elsayed and Mahmood (2024) addressed catastrophic forgetting and the loss of plasticity, the gradual decline in the ability of a network to learn. Their method, utility-based perturbed gradient descent, estimates the utility of each weight as the increase in the loss that its removal would cause. Useful weights receive smaller updates, which protects them, while noise added to less useful weights keeps them plastic. The method runs online without task boundaries or replay. It performed comparably to or better than elastic weight consolidation (Kirkpatrick et al., 2017) and synaptic intelligence (Zenke et al., 2017). In a policy-gradient agent, it also avoided the late decline of performance observed with the Adam optimizer.

Ben-Iwhiwhu et al. (2023) applied modulating masks to lifelong reinforcement learning. The network keeps its random initial weights, and a separate mask over them is learned for each task. Because these shared parameters never change, a new task cannot overwrite the earlier ones. The masks of previous tasks can be combined linearly to form the mask of a new task, which speeds up learning and is beneficial when rewards are sparse. The method retained earlier tasks better than online elastic weight consolidation and experience replay. It requires the identity of the task during training and testing, and it stores one mask per task.

Iyer et al. (2022) added active dendritic segments to the units of a deep network. Each segment receives a context signal, such as a code of the current task, and the strongest dendritic response modulates the output of the unit. A k-winner-take-all step, which keeps only the k most active units of a layer, then selects a sparse subnetwork for each context, and only its units are updated. The model was applied to multi-task reinforcement learning, in which a robot learns several manipulation tasks with one network, and to continual supervised learning. Diferent contexts selected overlapping but distinct subnetworks, which reduced interference between tasks.

The present paper difers from these approaches in two respects: the task code gates branches through shunting inhibition (inhibition that divides the efect of excitation at the same location instead of subtracting from it), which also attenuates the weight changes in the gated branches, and the network learns from reward with a local rule in which each synapse computes its own consolidation.

## 3 Model Description

## 3.1 Unipolar Neurons

A perceptron (Rosenblatt, 1958) models a neuron as a single point. All inputs are summed at this point, and one output is sent to every target (Figure 1a):

$$
y = \varphi \Big ( \sum _ { i } w _ { i } x _ { i } - \theta \Big ) .\tag{1}
$$

Here $x _ { i }$ are the inputs, $w _ { i }$ their weights, θ a threshold, and $\varphi$ a nondecreasing nonlinearity such as max(0, ·). Each weight sets how strongly its input contributes to the output, and every target receives the same value y.

The neuron used in this paper is a tree of connected segments (Figure 1b). Each segment has an activation, a number between 0 and 1 that depends on the synaptic inputs to the segment and on the activations of its neighbors (Section 3.2). Input synapses sit on the segments. The branch segments also carry output synapses, which send a graded local output $y _ { k }$ that depends on the activation of their segment. Local outputs reach only nearby neurons. One segment of the trunk, the trigger s, produces the global output z of the neuron. Only the global output reaches distant neurons and the readout of the network. A single neuron therefore has many inputs, several local outputs and one global output.

Each synapse has a weight and a fixed location in the tree. The efect of an input is largest on the segment of its synapse and decreases with every step along the tree. A local output therefore reflects mainly the inputs to its own branch. The trigger receives a reduced contribution from every branch and combines the contributions of all branches. All outputs of a neuron depend on the same weighted inputs, but each output reads them at a diferent location. In this sense, a unipolar neuron behaves like a small layer of units that share their input weights. The weight of a synapse sets the strength of its input, and its location sets how this strength is divided among the outputs. The locations are drawn at random when the network is built and remain fixed; only the weights are learned.

![](images/4100817fb761b603b9cc42ec857803afac618051a727325999e35509a23e55d7.jpg)  
Figure 1: A perceptron (a) and a unipolar neuron (b). (a) All inputs $x _ { i }$ are summed at one point, and every target receives the same output y. (b) Input synapses and output synapses lie side by side along the branches; gaps in the branches mark the boundaries between segments. Each local output (y<sub>1</sub> to $y _ { 4 } )$ depends on the activation of its own segment and reaches nearby neurons. The trigger s sends the global output z along the axon to distant neurons. The cell body is attached to the trunk by a thin stalk, receives no synapses and is not part of the model.

The shape of the tree determines how independent the outputs are. When neighboring segments are weakly coupled, each segment depends mainly on its own inputs, and the neuron behaves like several separate units. When they are strongly coupled, all segments have similar activations, and the neuron behaves like a point neuron. Inputs can also be inhibitory. An input that excites one segment and inhibits another can raise one output of the neuron and lower another. With such wiring, a single neuron with three segments computes the exclusive-or (XOR) of two binary inputs, which no perceptron can do (Section 3.2).

Unipolar neurons are the typical neuron type of the octopus and of most other invertebrates (Bullock and Horridge, 1965; Young, 1971). A single neurite leaves the cell body and branches into a tree. Input and output synapses lie side by side on the same branches. In many invertebrate neurons, output synapses release transmitter in a graded way that depends on the local membrane potential, without action potentials (Burrows and Siegler, 1978). Action potentials start at one site of the main process and travel along the axon to distant targets. The cell body is attached to the main neurite by a thin stalk and receives no synapses.

The proposed model uses this organization and omits the cell body. It describes the neuron in mathematical terms suited to machine learning and does not reproduce the electrical dynamics of real neurons.

## 3.2 The Neuron Model

A neuron is a tree of K segments. One segment of the trunk, the trigger segment $s ,$ produces the global output. In the default neuron, a single trunk segment contains the trigger, and four branches of two segments each are attached to it, so $K = 9$ (Figure 2). Each segment k has an activation $v _ { k } .$ . Neighboring segments $j$ and k are linked by a coupling $c _ { j k } > 0$ , and each segment has a leak $\ell _ { k } > 0$ . Couplings and leaks describe the shape of the neuron (e.g., leak 2 for the trunk, 0.5 for a branch segment, and coupling 2). They are fixed and not learned.

The inputs of the neuron arrive through synapses. Synapse i is located on segment $k ( i )$ receives an input $x _ { i } \geq 0$ and has a learned weight $w _ { i }$ between 0 and an upper bound (e.g., 10). Its weighted input is $g _ { i } = w _ { i } x _ { i }$ . Each synapse is either excitatory or inhibitory. The type is encoded by a fixed target value $E _ { i }$ , which is 1 for an excitatory synapse and 0 for an inhibitory synapse. Every input source of a neuron (a pixel or an output of another neuron) forms one excitatory and one inhibitory synapse on it, each on a segment drawn at random, including the trunk. A single input can therefore excite one part of the tree and inhibit another. The model also allows several synapses of each type per source, or synapses on branch segments only.

The activation of a segment is a weighted average of three kinds of terms: the target values of its synapses, the activations of its neighbors, and 0. The weights of the average are the weighted inputs $g _ { i }$ , the couplings $c _ { j k }$ and the leak $\ell _ { k } \mathbf { \cdot }$

$$
v _ { k } = \frac { \sum _ { i \in k } g _ { i } E _ { i } + \sum _ { j \sim k } c _ { j k } v _ { j } + \ell _ { k } \cdot 0 } { \sum _ { i \in k } g _ { i } + \sum _ { j \sim k } c _ { j k } + \ell _ { k } } .\tag{2}
$$

Here $i \in k$ runs over the synapses on segment $k ,$ and $j \sim k$ over its neighbors. Excitatory inputs move the activation toward 1. Inhibitory inputs and the leak move it toward 0.

Eq. (2) defines each activation through the activations of its neighbors. Multiplying it by its denominator gives one linear equation per segment. Together, these equations form the system:

$$
\mathbf { M } \mathbf { v } = \mathbf { b } .\tag{3}
$$

Here v is the vector of all activations. The diagonal entries of M are the denominators of Eq. (2), $\begin{array} { r } { M _ { k k } = \ell _ { k } + \sum _ { i \in k } g _ { i } + \sum _ { j \sim k } c _ { j k } } \end{array}$ . Two neighboring segments have $M _ { j k } = - c _ { j k } $ and all other entries are zero. The right-hand side $\begin{array} { r } { b _ { k } = \sum _ { i \in k } g _ { i } E _ { i } } \end{array}$ is the excitatory weighted input of segment $k ,$ because inhibitory synapses have $E _ { i } = 0$ . The matrix M is symmetric. In each row, the diagonal entry exceeds the sum of the magnitudes of the other entries by at least the leak, so M is positive definite and the system has a unique solution. All activations lie between 0 and 1: the largest activation is an average of numbers that are at most 1 or at most itself, with a positive weight on 0, so it cannot exceed 1, and the same argument applied to the smallest activation shows that it cannot be negative. Because the tree has no cycles, Gaussian elimination from the leaves toward the trigger, followed by substitution back toward the leaves, solves the system exactly in a number of operations proportional to K (Hines, 1984). The model solves it for every input.

![](images/fcf92fa4fe9aabdeb067e2abdfe86735d69a14463944a2a62b0aee3d793d3552.jpg)  
Figure 2: Structure of the default neuron. A single trunk segment contains the trigger $s ,$ and four branches are attached to it, each made of an entry segment next to the trunk and a distal segment. Each segment k has an activation $v _ { k }$ and a leak $\ell _ { k } .$ and neighboring segments are linked by couplings $c _ { j k }$ . Excitatory and inhibitory synapses can sit on any segment, including the trunk. Each of the eight branch segments sends a local output $y _ { k }$ to nearby neurons, and the trigger sends the global output z to distant neurons and the readout. The back-boost (dashed) adds $\beta z \rho _ { k }$ to the activation of every segment k $\left( \operatorname { E q . 9 } \right)$ ; its lighter shade on the distal segments indicates that $\rho _ { k }$ decreases with distance from the trigger.

The activations are a nonlinear function of the inputs, because each weighted input appears both in the numerator and in the denominator of Eq. (2), that is, both in b and in M. Three properties follow from this form.

The first two properties are easiest to see in a single segment without neighbors. Let this segment have leak ℓ, total excitatory weighted input $g _ { \mathrm { e x c } }$ (the sum of $g _ { i }$ over its excitatory synapses) and total inhibitory weighted input $g _ { \mathrm { i n h } }$ . Eq. (2) then reduces to:

$$
v = \frac { g _ { \mathrm { e x c } } } { g _ { \mathrm { e x c } } + g _ { \mathrm { i n h } } + \ell } .\tag{4}
$$

First, the activation saturates. For $\ell = 1$ , total excitatory inputs of 1, 2 and 10 give activations of 0.50, 0.67 and 0.91 (Figure 3). The activation approaches 1 but never exceeds it. Second, inhibition acts divisively. The inhibitory input appears only in the denominator, and Eq. (4) can be written as:

$$
v = { \frac { g _ { \mathrm { e x c } } } { g _ { \mathrm { e x c } } + \ell } } \cdot { \frac { g _ { \mathrm { e x c } } + \ell } { g _ { \mathrm { e x c } } + g _ { \mathrm { i n h } } + \ell } } .\tag{5}
$$

![](images/52cd6b5925ff9f1148a5730bdcf56329695e19a7dbb10f830abf358ccc921e20.jpg)  
Figure 3: Activation of a single segment without neighbors as a function of its total excitatory weighted input $g _ { \mathrm { e x c } } ,$ , for leak $\ell = 1 \ ( \mathrm { E q . \ 4 } )$ . Without inhibitory input (blue), the activation saturates toward 1. An inhibitory input $g _ { \mathrm { i n h } } = 6$ (black) scales the activation down. The markers show the activations for $g _ { \mathrm { e x c } } = 2 \colon 0 . 6 7$ without inhibition and 0.22 with inhibition.

The first factor is the activation without inhibition. The second factor lies between 0 and 1 and scales it down. For ℓ = 1 and $g _ { \mathrm { e x c } } = 2 ,$ , an inhibitory input of 6 multiplies the activation by $3 / 9 ,$ , from 0.67 to 0.22. Without excitatory input, the same inhibitory input has no efect, because the activation is already 0. Excitation and inhibition at the same location therefore interact multiplicatively.

The third property concerns segments without synapses. By Eq. (2), the activation of such a segment is a weighted average of the activations of its neighbors and 0, so it is smaller than the activation of its most active neighbor. Activation therefore decreases with every step away from the synapses that produce it, and it decreases faster when the leaks are large compared with the couplings. The trigger receives an attenuated contribution from every branch and combines the contributions of all branches. The location of inhibition matters in the same way. Inhibition on the trunk enters only the equation of the trigger segment, so it lowers mainly the trigger activation and changes the branch activations little. Inhibition on a branch lowers the activations of that branch and, with them, its contribution to the trigger.

Each branch segment k produces a local output, a rectified linear function (ReLU) of its activation:

$$
y _ { k } = \operatorname* { m a x } \left( 0 , \ v _ { k } - t _ { k } \right) .\tag{6}
$$

Here $t _ { k }$ is the threshold of output k. In the default neuron, all eight branch segments have local outputs; they can also be restricted to the branch tips. The trigger segment produces the global output:

$$
z = \sigma \left( \frac { v _ { s } - \theta } { T } \right) ,\tag{7}
$$

Table 1: Activations and outputs in Example 1 (output thresholds 0.3, $\theta = 0 . 1 4$ $T = 0 . 0 0 2 )$
<table><tr><td>Inputs</td><td> $( x _ { 1 } , x _ { 2 } )$   $v _ { 1 }$ </td><td> $v _ { 2 }$ </td><td> $v _ { 3 }$ </td><td>y1</td><td>y2</td><td>z</td></tr><tr><td>(0,0)</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.00</td></tr><tr><td>(1,0)</td><td>0.524</td><td>0.048</td><td>0.095</td><td>0.224</td><td>0</td><td>0.00</td></tr><tr><td>(0, 1)</td><td>0.048</td><td>0.524</td><td>0.095</td><td>0</td><td>0.224</td><td>0.00</td></tr><tr><td>(1, 1)</td><td>0.545</td><td>0.545</td><td>0.182</td><td>0.245</td><td>0.245</td><td>1.00</td></tr></table>

where $\sigma$ is the logistic function, θ is the threshold of the neuron, and $T$ is a temperature that sets the slope. The global output lies between 0 and 1 and equals 0.5 when $v _ { s } = \theta .$ It is a real number and a deterministic function of the input; no spikes are sampled. The thresholds and the temperature are set before training by calibration (Section 3.3) and then remain fixed.

Example 1: AND. Both this example and the next use a neuron with 3 segments (Figure 4). Segments 1 and 2 are branches with leak 1. Segment 3 is the trunk with leak 4 and contains the trigger. Each branch is linked to the trunk with coupling 1. The neuron has two binary inputs, $x _ { 1 }$ and $x _ { 2 }$ . The examples omit the back-boost defined below, so that all values can be computed more easily. In Example 1, $x _ { 1 }$ has an excitatory synapse on segment 1 and $x _ { 2 }$ has one on segment 2. Both synapses have weight 2, so an active input contributes a weighted input of 2.

The trigger segment has no synapses. By Eq. (2), its activation is the weighted average of the two branch activations and 0, with weights 1, 1 and 4:

$$
v _ { 3 } = { \frac { 1 \cdot v _ { 1 } + 1 \cdot v _ { 2 } } { 1 + 1 + 4 } } = { \frac { v _ { 1 } + v _ { 2 } } { 6 } } .
$$

The trigger activation is therefore one sixth of the sum of the two branch activations.

Table 1 lists all activations and outputs. When only $x _ { 1 }$ is active, the activation of segment 1 is 0.524. The coupling passes activation on to the trunk, where it is 0.095, and from there to segment 2, where it is 0.048. When both inputs are active, each branch has activation 0.545, and the trigger activation is 0.182.

The thresholds determine which function each output computes. With output thresholds of 0.3, y<sub>1</sub> is positive exactly when $x _ { 1 }$ is active, and $y _ { 2 }$ exactly when $x _ { 2 }$ is active. With $\theta = 0 . 1 4$ and $T = 0 . 0 0 2$ , the global output z is 1.00 when both inputs are active and 0.00 otherwise. The global output therefore computes $x _ { 1 } \ \mathrm { A N D } \ x _ { 2 }$ , while each local output passes on one input.

Example 2: XOR. The XOR of two binary inputs is 1 when exactly one of them is active. A neuron with only excitatory synapses cannot compute it, because adding an excitatory input increases the activation of every segment, so the trigger activation is always highest when both inputs are active.

Example 2 changes the wiring in one respect (Figure 5): each input also has an inhibitory synapse with weight 6 on the other branch. Input $x _ { 1 }$ thus excites segment 1 and inhibits segment 2, and $x _ { 2 }$ excites segment 2 and inhibits segment 1. Because every input forms one excitatory and one inhibitory synapse on randomly chosen segments of each neuron that receives it, this kind of wiring arises in the network.

![](images/26f8934497cb04dafc486d5970b4d907c04b31a5847ce51e6f2c55e036369476.jpg)

![](images/01f8baaaa959d3e9889273d7a512931bb39cdba5d9a959bbf7ff3808c7fcb911.jpg)  
Figure 4: Example 1: AND. Each input of the three-segment neuron has an excitatory synapse with weight 2 on its own branch. The bars show the trigger activation v<sub>3</sub> for each input combination. The right column shows the global output z for θ = 0.14 and T = 0.002.

With one active input, for example $x _ { 1 } = 1$ and $x _ { 2 } = 0$ , segment 1 receives an excitatory input of 2 and no inhibitory input. By Eq. (2), its activation and the trigger activation are:

$$
\begin{array} { l } { v _ { 1 } = \displaystyle \frac { 2 \cdot 1 + 1 \cdot v _ { 3 } } { 2 + 1 + 1 } = 0 . 5 2 2 } \\ { v _ { 3 } = \displaystyle \frac { v _ { 1 } + v _ { 2 } } { 6 } = 0 . 0 8 9 . } \end{array}
$$

Segment 2 receives an inhibitory input of 6 but no excitatory input. Its activation comes only from the trunk and is small (0.011), so the inhibition has little efect. The trigger activation, 0.089, is close to its value in Example 1 (0.095).

With both inputs active, each branch receives an excitatory input of 2 and an inhibitory input of 6. The inhibitory input adds 6 to the denominator of Eq. (2):

$$
\begin{array} { l } { v _ { 1 } = \frac { 2 \cdot 1 + 1 \cdot v _ { 3 } } { 2 + 6 + 1 + 1 } = 0 . 2 0 7 } \\ { v _ { 3 } = \frac { 0 . 2 0 7 + 0 . 2 0 7 } { 6 } = 0 . 0 6 9 . } \end{array}
$$

Each branch activation falls from 0.522 to 0.207. The sum of the two branch activations, 0.414, is now smaller than the sum with a single active input, 0.533. The trigger activation therefore decreases from 0.089 to 0.069.

The trigger activations for the input combinations (0, 0), (1, 0), (0, 1) and (1, 1) are 0, 0.089, 0.089 and 0.069. Any threshold between 0.069 and 0.089 separates the two combinations with exactly one active input from the other two. With $\theta \ : = \ : 0 . 0 8$ and

![](images/88ad9a504e7e1cb18677cdbea62d3b7de868bf5bc42174cda25334338bee1e3e.jpg)

![](images/c4e1044c33112de4bf37ddcaea646e527b3ef9eec0b21a233a6caa354084ef61.jpg)  
Figure 5: Example 2: XOR. Each input of the three-segment neuron has an excitatory synapse with weight 2 on its own branch and an inhibitory synapse with weight 6 on the other branch. The bars show the trigger activation $v _ { 3 }$ for each input combination. The right column shows the global output z for $\theta \ : = \ : 0 . 0 8$ and $T = 0 . 0 0 2$

Table 2: Activations and outputs in Example 2 (output thresholds 0.3, $\theta = 0 . 0 8 .$ $T = 0 . 0 0 2 )$ .
<table><tr><td>Inputs  $( x _ { 1 } , x _ { 2 } )$ </td><td> $v _ { 1 }$ </td><td> $v _ { 2 }$ </td><td>v3</td><td>y1</td><td>y2</td><td> $z$ </td></tr><tr><td>(0,0)</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.00</td></tr><tr><td>(1,0)</td><td>0.522</td><td>0.011</td><td>0.089</td><td>0.222</td><td>0</td><td>0.99</td></tr><tr><td>(0,1)</td><td>0.011</td><td>0.522</td><td>0.089</td><td>0</td><td>0.222</td><td>0.99</td></tr><tr><td>(1, 1)</td><td>0.207</td><td>0.207</td><td>0.069</td><td>0</td><td>0</td><td>0.00</td></tr></table>

$T = 0 . 0 0 2$ , the global output z is 0.00, 0.99, 0.99 and 0.00, which is the XOR of the inputs (Table 2).

The local outputs compute two further functions. With output thresholds of 0.3, $y _ { 1 }$ is positive only for the combination (1, 0). When both inputs are active, the inhibition from x<sub>2</sub> keeps the activation of segment 1 below the threshold. The local output $y _ { 1 }$ therefore computes $x _ { 1 }$ AND NOT $x _ { 2 } .$ , and $y _ { 2 }$ computes x<sub>2</sub> AND NOT $x _ { 1 }$ . A single neuron with four synapses thus computes three diferent functions of its two inputs.

In the full model, the global output also acts on the branches, a process called back-boost. This efect is described with the spread vector:

$$
\pmb { \rho } = \mathbf { M } ^ { - 1 } \mathbf { e } _ { s } ,\tag{8}
$$

where $\mathbf { e } _ { s }$ is the unit vector of the trigger segment. If the right-hand side of the trigger segment in Eq. (3) grows by a small amount $\epsilon ,$ the solution changes by $\epsilon \rho .$ . The entry $\rho _ { k }$ is therefore the change of the activation of segment k per unit of additive input at the trigger. Because M is symmetric, $\rho _ { k } = ( \mathbf { M } ^ { - 1 } ) _ { k s } = ( \mathbf { M } ^ { - 1 } ) _ { s k }$ is also the change of the trigger activation per unit of additive input at segment k. The spread vector thus measures how strongly each segment and the trigger influence each other. It depends on the input, because the weighted inputs enter M. A strongly inhibited segment has a large diagonal entry, and both this segment and the segments beyond it have small entries $\rho _ { k }$

The back-boost adds to every segment the change $\beta z \rho _ { k }$ that an additive input $\beta z$ at the trigger would produce, where $\beta$ is the back-boost strength (e.g., 2). The local outputs read the boosted activations:

$$
y _ { k } = \operatorname* { m a x } \left( 0 , \ v _ { k } + \beta z \rho _ { k } - t _ { k } \right) .\tag{9}
$$

In the full model, Eq. (9) replaces Eq. (6). The global output is computed from the activation of the trigger before the boost, so the boost does not change z. Through the back-boost, each local output combines the inputs to its own branch with the global output of the neuron. When the global output is high, all local outputs rise, most in branches that are strongly connected to the trigger and least in inhibited branches. The spread vector appears again in learning, because requested changes of the global output enter at the trigger and reach the segments in proportion to $\rho$ (Section 3.4).

Eq. (2) is the steady state of a passive compartmental model of a neuron (Koch, 1999). Each segment is a compartment, and $v _ { k }$ is its membrane potential relative to rest, scaled so that the reversal potential of excitatory synapses is 1. The weighted inputs $g _ { i }$ are synaptic conductances, the target values $E _ { i }$ are reversal potentials, and the couplings and leaks are the axial and leak conductances. Eq. (2) then states that the currents into each compartment sum to zero. Inhibition whose reversal potential equals the resting potential divides the efect of excitation, as in Eq. (4), and is known as shunting inhibition. The weights are nonnegative because a conductance cannot be negative. The global output can be read as the firing probability of the trigger, and the back-boost as a steady-state counterpart of the action potential, which spreads from the trigger back into the branches.

## 3.3 The Network Model

The network is defined independently of the task. It needs only the number of inputs and their positions, the number of tasks and the number of actions. Neurons are arranged in layers. Each layer places groups of neurons on a regular grid of positions in the unit square (e.g., 8 neurons at each of $7 \times 7$ positions). The local outputs of a neuron take the position of their neuron.

A neuron receives one or more groups of inputs, each drawn from one population of sources: the pixels in an image, the local outputs of an earlier layer, or the global outputs of an earlier layer. From each group, the neuron samples a fixed number of sources once, when the network is built. Inputs and local outputs are sampled within a square window around the position of the neuron, and global outputs are sampled from anywhere. Each sampled source then forms one excitatory and one inhibitory synapse on the neuron (Section 3.2). This wiring gives the network a receptive-field structure similar to that of a convolutional network, but without weight sharing: every neuron learns its own weights.

The default network has two layers (Figure 6). If the input is an image, like in MNIST, a neuron of the first layer samples pixels near its position (e.g., 64 pixels from a window of about 8 × 8 pixels). A neuron of the second layer samples local outputs of nearby first-layer neurons (e.g., 96) and global outputs of first-layer neurons from anywhere (e.g., 32). Local outputs therefore pass detailed, spatially restricted information forward, and global outputs provide a compact summary of the whole image. Networks with more layers or other combinations of sources are built in the same way, and the learning rule applies to them without modification (Section 3.4).

![](images/195e18faab6b54b3ef414717a279cefe6f806b01aad059d54883a2a1a7edb0f1.jpg)  
Figure 6: Default network for MNIST. Each layer is a grid of 7 × 7 positions with 8 neurons at each. A neuron of layer 1 samples 64 pixels from a window around its position. A neuron of layer 2 samples 96 local outputs of layer-1 neurons from a window around its position, and 32 global outputs of layer-1 neurons from anywhere. The global outputs of layer 2, and optionally those of layer 1 (dashed), reach the readout, which predicts the reward of each of the 10 actions. The task code inhibits a random subset of branch entries in every neuron of both layers.

When the identity of the current task is available, it reaches the network as a one-hot task code, a vector with one unit per task in which only the unit of the current task is active. This code is given during both training and testing. Each code unit acts on every neuron through fixed inhibitory synapses at the entry segments of its branches. For every combination of neuron, code unit and branch, such a synapse exists with a fixed probability (e.g., 0.5), drawn once when the network is built. Its weight is fixed (e.g., 6), not learned. An active code unit therefore adds a large inhibitory weighted input to the entry segments it reaches, in the same way as any inhibitory synapse in Eq. (2).

Strong inhibition at the entry segment separates a branch from the trunk. Inputs to the gated branch then contribute little to the trigger activation and thus to the global output. Under a given task code, about half of the branches of each neuron are gated. Diferent codes gate diferent branches, so the same input drives partly distinct branches and neurons under diferent tasks. The scheme resembles context-dependent gating of hidden units (Masse et al., 2018; Leon, 2026a), applied here to the branches within each neuron. Optionally, the code can also inhibit the trigger segment of a random subset of neurons, which silences their global output while their branches continue to work. Section 3.5 shows how gating also protects the weights of gated branches during learning.

In biophysical terms, each gate is a shunting inhibitory synapse at the base of a branch, which isolates the branch from the trunk.

The readout is linear in the global outputs of the last layer. It assigns a value to each action a:

$$
Q _ { a } = \sum _ { c } u _ { a c } z _ { c } .\tag{10}
$$

The sum runs over the neurons c whose global outputs reach the readout, and $u _ { a c }$ are the readout weights, which start at zero. Optionally, the global outputs of all layers reach the readout. Local outputs do not. For one-step episodes, as in the experiments, $Q _ { a }$ predicts the reward of action a for the current input. The readout has no bias term; a bias is shared by all inputs, so rewarding the classes of a new task would raise their values for the inputs of earlier tasks as well.

During training, the network chooses its action from the policy:

$$
\pi _ { a } = \left( 1 - \varepsilon \right) \frac { \exp ( Q _ { a } / \tau ) } { \sum _ { b } \exp ( Q _ { b } / \tau ) } + \frac { \varepsilon } { A } ,\tag{11}
$$

a softmax over the values with temperature $\tau \ ( \mathrm { e . g . , 0 . 1 } )$ , mixed with a small share $\varepsilon \ { \mathrm { ( e . g . } }$ 0.1) of uniformly random actions. Here A is the number of actions (e.g., 10 for MNIST). Because the forward pass is deterministic, the action choice is the only source of exploration. The uniform share keeps every action in use, including actions that a policy shaped by earlier tasks would almost never choose, such as the classes of a new task. At test time, the network chooses the action with the largest value.

## 3.4 Learning by Recirculation

The network learns online and from reward alone, with two passes through the network at every step (Figure 7). The first pass is the forward pass of Sections 3.2 and 3.3: the network computes all activations and outputs, chooses an action and receives a reward. The second pass starts at the readout, which compares its prediction with the outcome and signals each neuron that it reads to change its global output in proportion to the error. These requested changes travel back through the same connections that carried activity forward, from the last layer to the first. Each neuron responds to the requests it receives with a modification of its segment activations. Each synapse then changes according to its input, its type and the modification of its own segment. The rule and the signals it uses are the same in every layer, so additional layers do not require any adjustments.

The name of the method refers to the recirculation procedure of Hinton and McClelland (1988), in which each weight changes in proportion to its presynaptic activity and to the diference between the postsynaptic activities in two passes. O’Reilly (1996) generalized this procedure to recurrent networks that learn arbitrary input–output mappings from local activation diferences. The second pass is also related to the second phase of equilibrium propagation (Scellier and Bengio, 2017), in which the outputs are nudged toward a better value and the resulting change of the steady state determines the weight updates.

(a) Two passes through the network  
![](images/37044e5d3a5f323c040348e6f48b7214957006d91009de5a401cfccd64bb3711.jpg)

(b) Second pass inside one neuron of layer 2  
![](images/a6878046672ef683a0219e234efd123095c435411a4d8ab0a56498bed6f2f6d0.jpg)  
Figure 7: One learning step. In the first pass, the image passes through both layers to the readout, which predicts the reward of every action; the network chooses an action and receives a reward. In the second pass, the readout error $\delta$ of the chosen action returns as requested changes of the global outputs. Each neuron shifts its segment activations by $\Delta \mathbf { v }$ in response, each synapse changes in proportion to its input, its driving term $E _ { i } - v _ { k }$ and the shift of its segment, and the synapses pass requests back to the outputs of the earlier layer.

After the network chooses an action a, the environment returns a reward r. The readout compares its prediction with the outcome:

$$
\delta = r + \gamma V ^ { \prime } - Q _ { a } .\tag{12}
$$

Here $\begin{array} { r } { V ^ { \prime } = \sum _ { b } \pi _ { b } ^ { \prime } Q _ { b } ^ { \prime } } \end{array}$ is the value of the next input under the policy of Eq. (11), γ is the discount factor $( \mathrm { e . g . , 0 . 9 } )$ , and $V ^ { \prime }$ is 0 when the episode ends. As in temporal-diference learning, the term $r + \gamma V ^ { \prime }$ is treated as a fixed target. In the experiments, every episode has a single step, so the error is simply $\delta = r - Q _ { a }$ , the diference between the reward and its prediction. Only the chosen action has an error.

To move $Q _ { a }$ toward the outcome, the readout demands each neuron c that it reads to change its global output by:

$$
d _ { c } = \delta u _ { a c } .\tag{13}
$$

When the prediction is too low $\left( \delta > 0 \right)$ , neurons with a positive weight to the chosen action are demanded to become more active, and neurons with a negative weight less active. The readout thus sends the error back through each of its connections, with a strength equal to the forward weight of that connection. When the global outputs of all layers reach the readout, the neurons of earlier layers receive such requests directly, in addition to the requests that come from later layers.

A neuron can receive a request d for its global output, from the readout or from neurons of later layers, and requests $d _ { k }$ for its local outputs, from nearby neurons of the next layer. It turns all requests into additive terms $\Delta \mathbf { b }$ on the right-hand side of its activation system (3). At each output segment $k \colon$

$$
\Delta b _ { k } = \left\{ \begin{array} { l l } { d _ { k } } & { \mathrm { i f ~ } y _ { k } > 0 , } \\ { 0 } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{14}
$$

An active local output follows the activation of its segment one to one $\left( \operatorname { E q . 9 } \right)$ , so the request enters at that segment unchanged. An inactive output lies on the flat part of its rectifier function and passes nothing back. At the trigger segment:

$$
\Delta b _ { s } = \frac { z ( 1 - z ) } { T } \Big ( d + \beta \sum _ { k } \rho _ { k } \Delta b _ { k } \Big ) .\tag{15}
$$

The factor $z ( 1 - z ) / T$ is the slope of the global output in Eq. (7), that is, how much z changes per unit of trigger activation. The sum accounts for the back-boost. Every local output rises by $\beta \rho _ { k }$ per unit of global output (Eq. 9), so requests for the local outputs also signal the global output to change.

The neuron responds with the change:

$$
\Delta \mathbf { v } = \mathbf { M } ^ { - 1 } \Delta \mathbf { b } ,\tag{16}
$$

which is the change of the solution of Eq. (3) when ∆b is added to its right-hand side. The weighted inputs do not change between the passes, so M is the matrix of the forward pass and the shift is exactly linear in the requests. The overall size of the requests therefore only scales all weight changes and is absorbed into the learning rate. The part of $\Delta \mathbf { b }$ that enters at the trigger spreads into the tree along the spread vector $\rho$ of Eq. (8), and each part that enters at an output segment spreads from that segment. Strongly inhibited branches have small entries in $\rho ,$ so they shift little in response to requests for the global output (Section 3.5).

Each synapse compares the activation of its own segment in the two passes. Synapse i on segment $k = k ( i )$ changes its weight by:

$$
\Delta w _ { i } = \eta x _ { i } \left( E _ { i } - v _ { k } \right) \Delta v _ { k } ,\tag{17}
$$

where $\eta$ is the learning rate. The three factors have a direct interpretation. The input $x _ { i }$ restricts learning to synapses that were active. The driving term $E _ { i } - v _ { k }$ is the direction in which a stronger synapse would move its segment: upward for an excitatory synapse, since $v _ { k } < 1$ , and downward for an inhibitory synapse, since $v _ { k } > 0$ . The shift $\Delta v _ { k }$ is the direction in which the segment is asked to move. A synapse whose driving term agrees with the requested shift is strengthened, and a synapse whose driving term opposes it is weakened. When a segment is asked to rise, its active excitatory synapses grow stronger and its active inhibitory synapses grow weaker; when it is asked to fall, the reverse happens. All three factors are available at the synapse itself. Weights are kept between 0 and their upper bound, and all updates are averaged over a batch of episodes that run in parallel (e.g., 32). Section 3.5 adds a factor that slows down synapses that have changed strongly in the past.

The synapses also pass requests back to their sources. Raising the activity of source $j$ by a small amount raises the weighted input of each of its synapses in proportion to the weight $w _ { i } .$ , and each of these synapses moves its segment in the direction $E _ { i } - v _ { k }$ . The change requested of source $j$ is therefore:

$$
d _ { j } = \sum _ { i \in S _ { j } } w _ { i } \left( E _ { i } - v _ { k ( i ) } \right) \Delta v _ { k ( i ) } ,\tag{18}
$$

where $S _ { j }$ is the set of synapses of source $j$ on all the neurons that sample it. Each contact multiplies the factor $( E _ { i } - v _ { k } ) \Delta v _ { k }$ that drives its own weight change by its forward weight instead of its input, so its return strength equals its forward weight. When the source is a local output of an earlier neuron, $d _ { j }$ reaches that neuron as a request for this local output. When the source is a global output, $d _ { j }$ reaches it as a request for the global output. The earlier layer then responds in the same way, by Eqs. (14)–(18), so the requests travel back layer by layer, until the first layer that sends nothing back.

The second pass computes the gradient of the squared readout error $L = \delta ^ { 2 } / 2$ , with one omission. Consider a synapse on segment k. Raising its weighted input $g _ { i }$ by a small amount ϵ adds ϵ to the diagonal entry $M _ { k k }$ and $\epsilon E _ { i }$ to $\begin{array} { r } { b _ { k } , } \end{array}$ , so to first order the activations change by $\boldsymbol { \epsilon } ( E _ { i } - v _ { k } ) \mathbf { M } ^ { - 1 } \mathbf { e } _ { k }$ . The derivative of $L$ with respect to $g _ { i }$ is therefore:

$$
\left( E _ { i } - v _ { k } \right) \left( \mathbf { M } ^ { - 1 } \partial L / \partial \mathbf { v } \right) _ { k } ,\tag{19}
$$

where $\partial L / \partial \mathbf { v }$ contains the direct derivatives of L with respect to the activations through the outputs of the neuron. By the chain rule, these direct derivatives are exactly −∆b of Eqs. (14) and (15), because each request is the negative derivative of $L$ with respect to the output it addresses. Since M is symmetric:

$$
\mathbf { M } ^ { - 1 } \partial L / \partial \mathbf { v } = - \Delta \mathbf { v } ,\tag{20}
$$

and:

$$
- { \frac { \partial L } { \partial g _ { i } } } = ( E _ { i } - v _ { k } ) \Delta v _ { k } .\tag{21}
$$

With $g _ { i } = w _ { i } x _ { i }$ , Eq. (17) is a step of gradient descent on $L ,$ and Eq. (18) gives the negative derivative of L with respect to each source. The symmetry of M is what allows the error to spread through a neuron by the same rule as activity, so that no separate backward computation is needed inside the neuron.

The omission concerns the back-boost. The weighted inputs also change the spread vector $\rho ,$ and with it the boost $\beta z \rho _ { k }$ that each local output receives. The second pass leaves this dependence out. It has no efect in the last layer, whose local outputs receive no requests, so there the update follows the gradient exactly. In earlier layers, the update approximates the gradient. Apart from this term, the update departs from gradient descent only through the consolidation factor of Section 3.5 and the bounds on the weights. Section 3.6 describes training the same network with the exact gradient.

After the second pass, the readout moves the prediction of the chosen action toward the outcome by a normalized least-mean-squares step:

$$
\Delta u _ { a c } = \alpha \frac { \delta z _ { c } } { 1 + \sum _ { c ^ { \prime } } z _ { c ^ { \prime } } ^ { 2 } } ,\tag{22}
$$

where α is the step size (e.g., 0.5). The weights of the other actions do not change. The update follows the second pass because the requests of Eq. (13) must use the weights that made the prediction. The normalization makes the change of $Q _ { a }$ almost independent of how many neurons are active: after the step, $Q _ { a }$ has moved toward the outcome by about the fraction α of the error. A confident wrong action is therefore corrected quickly, and the classes of a new task can enter a policy that earlier tasks have made confident. The readout weights start at zero, so the neurons receive no requests in the first step and begin to learn as soon as the readout makes predictions.

From a biophysical perspective, the requests are small currents injected at the active output synapses and at the trigger, and $\Delta \mathbf { v }$ is the change of the membrane potentials once the neuron has reached its new steady state. The driving term $E _ { i } - v _ { k }$ is the driving force of the synapse. Eq. (17) thus combines three quantities that are present at the synapse: presynaptic activity, driving force and the local change of membrane potential. The only assumption that goes beyond the single synapse is that each contact returns requests with a strength equal to its forward weight.

## 3.5 Continual Learning

The network learns its tasks one after another and does not encounter the data of the earlier tasks again. The learning rule itself does not use task boundaries: every step is processed in the same way, whichever task it belongs to. Two mechanisms limit the forgetting of earlier tasks. Consolidation slows down synapses that have changed strongly in the recent past, and the task code protects the weights of the branches and neurons that it gates.

Consolidation lowers the learning rate of each synapse according to its own history. With consolidation, the weight change of Eq. (17) becomes:

$$
\Delta w _ { i } = \frac { \eta } { 1 + \kappa \Omega _ { i } } x _ { i } \left( E _ { i } - v _ { k } \right) \Delta v _ { k } ,\tag{23}
$$

where $\kappa$ is the consolidation strength $( \mathrm { e . g . , 1 0 ^ { 4 } } )$ and $\Omega _ { i }$ is the importance of the synapse. The importance is a slowly decaying sum of the recent weight changes of the synapse, updated after every step:

$$
\Omega _ { i } \gets \left( 1 - \nu \right) \Omega _ { i } + \left| \Delta w _ { i } \right| .\tag{24}
$$

The decay $\nu$ is small (e.g., 0.0002), so the importance reflects approximately the last $1 / \nu$ steps, which is longer than the training on a single task in the experiments. A synapse that has changed little keeps the full learning rate η. A synapse whose recent changes add up to 1/κ learns at half that rate, and its rate keeps falling as it continues to change. When a synapse stops changing, its importance decays and its learning rate slowly recovers. The strength κ sets the balance between keeping earlier tasks, which favors a large κ, and learning new tasks quickly, which favors a small one. As the synapses used by earlier tasks slow down, later tasks are learned mainly by synapses that have changed little before.

Like elastic weight consolidation (Kirkpatrick et al., 2017) and synaptic intelligence (Zenke et al., 2017), consolidation slows down learning on the weights that earlier tasks have relied on. It difers from both methods in three respects. The importance measures how much a synapse has recently changed, without any estimate of how much the loss depends on it. Consolidation acts on the learning rate, instead of adding a penalty on the deviation of the weights from the values they had at the end of earlier tasks. Finally, it needs neither task boundaries nor stored copies of earlier weights, and every synapse computes it only from its own changes.

In biological terms, consolidation resembles metaplasticity, in which the history of a synapse changes its capacity for later plasticity (Abraham and Bear, 1996).

Section 3.3 introduced the task code as a way to route diferent tasks through partly diferent branches and neurons. The task code also limits how much the weights of a gated branch change. In the second pass, requests for the global output of a neuron enter at its trigger and reach the segments in proportion to the spread vector ρ (Eqs. 15 and 16). The strong inhibition at the entry of a gated branch adds a large term to the diagonal of M at that segment, so little of the spread passes into the branch. Its segments therefore shift little (Figure 7b), its synapses change little by Eq. (23), and the weights it learned under other tasks are largely kept.

The same protection applies to whole neurons. A neuron that the current task code silences has a global output close to 0, where the slope $z ( 1 - z ) / T$ in Eq. (15) is almost 0, so requests for its global output have almost no efect on it. The protection is partial; requests for local outputs enter directly at their segments (Eq. 14) and are not blocked by the gate, so a gated branch whose local outputs are active can still learn.

The two mechanisms complement each other. The task code gives each task partly distinct branches and neurons, so that fewer synapses are shared between tasks. Consolidation slows down the synapses that a task has used, including those that later tasks share. Masse et al. (2018) found the same complementarity in conventional networks, where context-dependent gating combined with synaptic stabilization was more efective than either method alone. Several design choices described earlier also limit interference between tasks: the readout has no bias and keeps a share of random actions (Section 3.3), the thresholds remain fixed after calibration (Section 3.3), and the readout updates only the prediction of the chosen action (Section 3.4).

## 3.6 Variants

The model of Sections 3.2–3.5 is the one used in the main experiments. This subsection describes the variants used for comparison and in the ablations, and a local learning rule of an earlier version of the model. They are not part of the model described above.

Gradient-based learning. Because the global output is a firing probability, the forward pass is a diferentiable function of all weights and thresholds. The same network can therefore

also be trained by backpropagation. Inside a neuron, the exact gradient follows from two linear systems with the matrix of the forward pass:

$$
\lambda = \mathbf { M } ^ { - 1 } \frac { \partial L } { \partial \mathbf { v } }\tag{25}
$$

$$
\pmb { \mu } = \mathbf { M } ^ { - 1 } \frac { \partial L } { \partial \pmb { \rho } } .\tag{26}
$$

Here $\partial L / \partial \mathbf { v }$ contains the derivatives of L through the outputs of the neuron, as in Section 3.4. The vector $\partial L / \partial \rho$ contains the derivatives through the back-boost: its entry for an active local output k is $\beta z \partial L / \partial y _ { k } ~ ( \mathrm { E q . ~ 9 ) }$ , and its other entries are 0. The derivative with respect to the weighted input of synapse i on segment k is:

$$
{ \frac { \partial L } { \partial g _ { i } } } = \left( E _ { i } - v _ { k } \right) \lambda _ { k } - \rho _ { k } \mu _ { k } .\tag{27}
$$

The first term has the form that recirculation computes, with $\pmb { \lambda } = - \Delta \mathbf { v }$ (Eqs. 16 and 17). The second term describes how the weighted inputs change the spread vector $\rho ,$ and with it the back-boost. It is the term that the second pass of recirculation leaves out.

Three objectives are available. The supervised objective is the cross-entropy between the softmax of the values $Q _ { a }$ and the label, and the network chooses the action with the largest value. It uses the label, so it serves as a reference with full information. The action-value objective uses the reward only and is the objective of recirculation learning: the value $Q _ { a }$ of the chosen action is regressed on $r + \gamma V ^ { \prime }$ , with the target held fixed. Gradients then reach every synapse by backpropagation instead of the second pass, so the comparison with recirculation keeps the reward signal the same and changes the way in which the weights are updated. The actor-critic objective also uses the reward only. The outputs $Q _ { a }$ of the readout then serve as the preferences of the softmax policy, and a second linear readout, the critic, predicts the value $V$ of the current input. The error $r + \gamma V ^ { \prime } - V$ trains the critic and weights the policy gradient of the softmax policy, and a small entropy bonus keeps the policy from becoming deterministic too early (Sutton and Barto, 2018).

All three objectives use the Adam optimizer (Kingma and Ba, 2015) and add a small penalty on the mean global output of every layer, which keeps long-distance activity sparse. After each step, the weights are projected back to the interval between 0 and their upper bound. The thresholds can be learned as well. With the action-value objective, gradientbased learning therefore difers from recirculation in five respects: the exact gradient including the second term of Eq. (27), the optimizer, the sparsity penalty, the learned thresholds, and the absence of consolidation.

Local learning rule. An earlier version of the model learned with a local rule, which recirculation later replaced. In that version, the trigger produces a sampled spike during training: z is 1 with the probability p given by Eq. (7), and 0 otherwise. The randomness of the spikes provides both the exploration and the signal from which each synapse estimates its contribution to the reward. There is no second pass; instead, a single error is broadcast to every synapse of the network, the temporal-diference error of the value of the current input:

$$
\bar { \delta } = r + \gamma V ^ { \prime } - V ,\tag{28}
$$

where $\begin{array} { r } { V = \sum _ { a } \pi _ { a } Q _ { a } } \end{array}$ . Unlike the readout error of Eq. (12), $\bar { \delta }$ does not depend on the action that was taken. Each synapse combines this error with quantities that are available at the synapse, as in three-factor learning rules (Fr´emaux and Gerstner, 2016). The weight change has two parts, a reward part and a branch part.

The reward part credits each synapse for its efect on the spike of its neuron:

$$
\Delta w _ { i } ^ { \mathrm { r e w a r d } } = \eta _ { \mathrm { r } } \bar { \delta } \frac { z - p } { T } x _ { i } \left( E _ { i } - v _ { k } \right) \rho _ { k } ,\tag{29}
$$

with learning rate $\eta _ { \mathrm { r } }$ . The product $x _ { i } ( E _ { i } - v _ { k } ) \rho _ { k }$ measures how strongly the weight moves the trigger activation, because $\rho _ { k }$ is the efect of segment k on the trigger (Section 3.2). The synapse has access to $\rho _ { k }$ because the back-boost adds $\beta z \rho _ { k }$ to its segment whenever the neuron fires. The factor $( z - p ) / T$ is positive whenever the neuron fired and negative whenever it stayed silent, and its magnitude is largest when the outcome was unlikely. Together, these factors form the derivative of the log-probability of the spike with respect to the weight, so for each neuron Eq. (29) is the REINFORCE estimate of the policy gradient (Williams, 1992), with $\bar { \delta }$ as the reinforcement. In multi-step episodes, the factors after $\bar { \delta }$ are accumulated in a decaying eligibility trace (Sutton and Barto, 2018).

The reward part credits only what a neuron does through its spike. The branch part trains the local outputs. Each local output taken before the back-boost, $\hat { y } _ { k } = \operatorname* { m a x } ( 0 , v _ { k } - t _ { k } )$ learns to predict the spike of its own neuron. The spike reaches the branch through the back-boost, so the target of this prediction is available at the segment. The prediction error $\delta _ { k } = z + \gamma \hat { y } _ { k } ^ { \prime } - \hat { y } _ { k }$ changes the synapses on segment $k ,$ while the output is active, by

$$
\Delta { w } _ { i } ^ { \mathrm { b r a n c h } } = \eta _ { \mathrm { b } } \delta _ { k } x _ { i } \left( E _ { i } - v _ { k } \right) ,\tag{30}
$$

with learning rate $\eta _ { \mathrm { b } }$ . This part needs no reward. Near the trigger, $\rho _ { k }$ is large and the reward part dominates; on distal segments, $\rho _ { k }$ is small and the branch part dominates. The two parts are added and multiplied by the consolidation factor of Eq. (23). The readout learns as in Eq. (22), and the task code protects gated branches as in Section 3.5: a gated branch receives little of the back-boost and therefore little reward credit. In addition, each trigger threshold was adapted during training so that its neuron fired on a target fraction of the steps. This homeostatic mechanism (Leon, 2026a) kept activity sparse and reactivated silent neurons, but it also changed the representations of earlier tasks.

All credit in this rule comes from the correlation between one broadcast error and the random spike of each neuron. The random spikes of all other neurons add noise to this correlation, so each synapse receives a weak signal, and the signal becomes weaker as the network grows. In the experiments, the local rule had little efect on the results. Recirculation replaces the random spikes by a deterministic second pass, in which every neuron receives its own requested change.

Options of the model. Table 3 lists the options that change the default model. Each option changes one element, and the default model is set when all of them are of.

Table 3: Options of the model used in the experiments. Each row describes the change from the default model.
<table><tr><td>Option</td><td>Change from the default model</td></tr><tr><td>No task code</td><td>The network receives no task code, so no branch is gated, and all tasks use the same branches and</td></tr><tr><td>Trunk gating</td><td>neurons. The task code also inhibits the trigger segment of a random subset of neurons, which silences their global</td></tr><tr><td>Readout from all layers</td><td>output under that task. The global outputs of all layers reach the readout,</td></tr><tr><td>Readout bias</td><td>not only those of the last layer (Eq. 10). Each action has a bias for the readout, shared by all inputs.</td></tr><tr><td>Outputs at branch tips</td><td>Synapses on branches only No synapse is placed on the trunk segment. Only the distal segment of each branch has a local</td></tr><tr><td>No back-boost</td><td>output, 4 per neuron instead of 8. The back-boost strength is β = 0, so local outputs</td></tr><tr><td>No consolidation</td><td>do not depend on the global output.</td></tr><tr><td></td><td>The consolidation strength is κ = 0, so every synapse keeps the full learning rate.</td></tr><tr><td>More synapses per source</td><td>Each sampled source forms several excitatory or in- hibitory synapses on the neuron.</td></tr><tr><td>Other neuron shapes</td><td>The neuron is an arbitrary tree, with its own seg- ments, leaks, couplings, outputs and gated segments.</td></tr></table>

## 4 Experimental Results

## 4.1 Setup

In these experiments, the same network is trained with diferent learning methods on two problems, MNIST and Iris, in which tasks are learned one after another. The scenarios compare how well each method handles the current task and how well it remembers the earlier ones.

MNIST. The 10 digits are divided into 5 tasks of 2 digits each, learned in the order 0–1, 2–3, 4–5, 6–7 and 8–9. Each training step presents a batch of 32 images of the current task. Each image is a one-step episode: the network chooses one of the ten classes and receives a reward of 1 if the class is correct and 0 otherwise. All 10 actions remain available in every task, so the network can also choose a class of an earlier or a later task. Recirculation is trained for 500 or 1500 steps per task, and the other learners for 1500 steps, which corresponds to about 4 passes over the roughly 12,000 training images of a task. The images of earlier tasks do not reappear. After each task, the network is tested on the test images of every task. At test time, the global outputs produce firing probabilities for every learner, and the network chooses the action with the largest value.

In the default setting, the network receives the task code presented in Section 3.3 during training and testing. Because the code identifies the task at test time, this setting is task-incremental in the terminology of van de Ven et al. (2022), although the network has a single readout and always chooses among all 10 classes. Without the task code, the network must recognize the digit without being told which task it belongs to.

Iris. The Iris data set (Fisher, 1936) contains 150 flowers, 50 of each of 3 species: setosa, versicolor and virginica. Each flower is described by 4 measurements: the length and width of its sepals and of its petals. Setosa is linearly separable from the other two species, whereas versicolor and virginica slightly overlap. Each measurement is scaled to [0, 1] with fixed ranges that enclose all flowers, and it is encoded by 8 Gaussian tuning curves with evenly spaced centers (Leon, 2026b) and a width equal to their spacing, so the network receives 32 inputs. Flowers of diferent sizes therefore excite largely diferent inputs. 15 instances of each class are held out for testing, with a new split in every run.

The sequence has 3 tasks: setosa and versicolor, then versicolor and virginica, and finally all three species. Setosa is absent from the second task, so the accuracy on the first task after the second one measures forgetting. All 3 actions are available in every task, and no task code is used. The network has a single layer of 64 neurons. The inputs have no positions, so each neuron samples 12 of the 32 inputs from all of them, and the readout reads the global outputs of the layer. Training lasts 300 steps per task, with batches of 16 flowers.

Learning methods. Four learning methods are compared on the same network, built and calibrated in the same way:

• Recirculation, the model of Sections 3.4 and 3.5, with learning rate $\eta = 2 0$ in Eq. (23), consolidation strength $\kappa = 1 0 ^ { 4 }$ and decay $\nu = 0 . 0 0 0 2$

• Gradient-based learning (Section 3.6), with the supervised and the action-value objectives. Adam updates all weights, including the readout, with learning rate $1 0 ^ { - 3 }$ $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9$ . The weight of the sparsity penalty is $1 0 ^ { - 3 }$ , the thresholds are learned, and there is no consolidation;

• The local rule (Section 3.6), with learning rates $\eta _ { r } = 0 . 0 0 2$ and $\eta _ { b } = 0 . 0 0 0 5$ and consolidation strength $\kappa = 1 0 ^ { 3 }$ . The adaptation of the trigger thresholds is switched of, because it shifted the representations of earlier tasks;

• A fixed network: the local rule with both learning rates set to zero. The neurons keep their calibrated weights, and only the readout learns. This learner shows how much the structure of the network contributes by itself.

Performance measures. Let $R _ { t , k }$ be the test accuracy on task k after training on task $t ,$ for $t , k = 1 , \ldots , N$ , where N is the number of tasks. The results are summarized by four measures:

• The accuracy after training, $\begin{array} { r } { \frac { 1 } { N } \sum _ { k } R _ { k , k } } \end{array}$ , the mean accuracy on each task right after it is learned. It measures how well a learning method acquires new tasks;

• The final accuracy, $\begin{array} { r } { \frac { 1 } { N } \sum _ { k } R _ { N , k } , } \end{array}$ the mean accuracy on all tasks at the end of training;

• The forgetting, $\begin{array} { r } { \frac { 1 } { N - 1 } \sum _ { k < N } \bigl ( \operatorname* { m a x } _ { k \le t < N } R _ { t , k } - R _ { N , k } \bigr ) } \end{array}$ , the mean loss of accuracy of each earlier task between its best value and the end of training (Chaudhry et al., 2018);

Table 4: Gradient-based learning and recirculation on MNIST with the task code. After training: mean accuracy on each task right after it is learned. Final: mean accuracy on all tasks at the end of training. Forgetting: mean loss of accuracy of the earlier tasks. Early reward: mean training reward in the first 300 steps of each task. Steps: training steps per task. With the supervised objective, the network does not explore, so its training reward is not comparable with that of the reward-based learners.
<table><tr><td>Learner</td><td>Steps</td><td>After training</td><td>Final</td><td>Forgetting</td><td>Early reward</td></tr><tr><td>Gradient, supervised</td><td>1500</td><td>0.993</td><td>0.197</td><td>0.994</td><td></td></tr><tr><td>Gradient, action values</td><td>1500</td><td>0.994</td><td>0.198</td><td>0.995</td><td>0.846</td></tr><tr><td>Recirculation</td><td>500</td><td>0.944</td><td>0.939</td><td>0.009</td><td>0.632</td></tr><tr><td>Recirculation</td><td>1500</td><td>0.966</td><td>0.868</td><td>0.123</td><td>0.611</td></tr></table>

• The early training reward, the mean reward over the first 300 training steps of each task (25 steps for Iris), averaged over the tasks. It includes exploration and measures how quickly new tasks are learned.

On Iris, the tasks share classes, so the averages over tasks count some classes more than once. The Iris results are therefore reported as individual entries of $R _ { t , k }$

The values in the tables are means over 10 runs of each configuration, and the figures show example runs.

## 4.2 Gradient-Based Learning and Recirculation

Table 4 compares gradient-based learning and recirculation on MNIST with the task code, and Figure 8 shows an example run of each.

Gradient-based learning acquires every task almost perfectly and quickly. With either objective, its accuracy after training is 0.99. With the action-value objective, the training reward rises to about 0.9 within roughly a hundred steps of each new task (Figure 8a, right), close to the limit of 0.91 set by the share $\varepsilon = 0 . 1$ of random actions. However, each new task erases all earlier ones. After each task, the accuracy on all earlier tasks drops to zero (Figure 8a, left), and the mean accuracy over the tasks seen so far falls as $1 / t$ (middle). The final accuracy, 0.197 with the supervised objective and 0.198 with the action-value objective, is the accuracy on the last task divided by five. This occurs both with the label and with the reward alone, even though the task code gates a diferent subset of branches in each task.

Recirculation retains the tasks it has learned. With 500 steps per task, its forgetting is 0.009, more than a hundred times lower than that of gradient-based learning, and its final accuracy of 0.939 is almost equal to its accuracy right after training, 0.944. All earlier tasks remain close to their initial accuracy until the end of the sequence (Figure 8b, left), and the mean accuracy over the tasks seen so far stays flat (middle). Recirculation achieves this only from the reward, with quantities available at each synapse, without task boundaries in the learning rule, and without stored copies of earlier weights.

![](images/8b16e0b8faeadc1979f37335c44ebc9f6ef6d00f1dfb881746433d885e8610d0.jpg)

(a) Gradient-based learning, action-value objective, 1500 steps per task  
![](images/35d10b0c38c86ea1dd4533aee8a09dee47bd29a104bb539a39534d9a85657447.jpg)  
(b) Recirculation, 500 steps per task  
Figure 8: Example runs on MNIST with the task code. Left: test accuracy on each task (columns) after training on each task (rows); the tasks are numbered from 0 in the order of training. Middle: mean accuracy over the tasks seen so far. Right: training reward, with the task boundaries marked.

Longer training raises the accuracy of recirculation on each task. With 1500 steps per task, the accuracy after training is 0.966, close to the 0.99 of gradient-based learning, while the forgetting, 0.123, remains about eight times lower than that of gradient-based learning. The final accuracy is 0.868, more than four times the 0.198 of gradient-based learning with the same reward signal. The length of training and the strength of consolidation set the balance between learning new tasks and keeping earlier ones (Section 4.3).

The action-value objective uses the same reward signal as recirculation, and with it, gradient-based learning reaches the same accuracy and the same forgetting as with the label. The diference in forgetting between the two learning methods therefore comes from the way in which recirculation changes the weights, and not from the feedback received.

## 4.3 Stability–Plasticity Trade-of

Three parameters of recirculation determine how far a synapse moves while a task is learned and how much of this change later tasks can undo: the learning rate η, the consolidation strength κ and the decay ν of the importance (Eqs. 23 and 24). The number of training steps per task acts in a similar way. Table 5 and Figure 9 show how these settings place recirculation between learning each new task as well as possible and keeping all earlier tasks.

Table 5: Recirculation on MNIST with the task code, for diferent consolidation strengths $\kappa ,$ importance decays $\nu ,$ learning rates η and numbers of training steps per task. Gradient-based learning with the action-value objective is shown for comparison. The measures are those of Table 4.
<table><tr><td>κ</td><td>ν</td><td>η</td><td>Steps</td><td>After training</td><td>Final</td><td>Forgetting</td></tr><tr><td> $\overline { { 1 0 ^ { 4 } } }$ </td><td>0.0002</td><td>20</td><td>500</td><td>0.944</td><td>0.939</td><td>0.009</td></tr><tr><td> $1 0 ^ { 4 }$ </td><td>0.0002</td><td>20</td><td>1500</td><td>0.966</td><td>0.868</td><td>0.123</td></tr><tr><td> $1 0 ^ { 3 }$ </td><td>0.0002</td><td>5</td><td>1500</td><td>0.966</td><td>0.866</td><td>0.126</td></tr><tr><td> $1 0 ^ { 3 }$ </td><td>0</td><td>20</td><td>1500</td><td>0.971</td><td>0.786</td><td>0.230</td></tr><tr><td> $1 0 ^ { 3 }$ </td><td>0.0002</td><td>20</td><td>1500</td><td>0.972</td><td>0.733</td><td>0.298</td></tr><tr><td colspan="2">Gradient, action values</td><td></td><td>1500</td><td>0.994</td><td>0.198</td><td>0.995</td></tr></table>

![](images/59dcc850247d5a4afd6489d5da33aa29607ac7a200fd460adf89976140e2bd31.jpg)  
Figure 9: Accuracy after training and forgetting on MNIST with the task code, for recirculation with the settings of Table 5 and for gradient-based learning with both objectives. Unless marked otherwise, $\nu = 0 . 0 0 0 2$ , $\eta = 2 0$ and training lasts 1500 steps per task. The settings $\kappa = 1 0 ^ { 4 }$ and $\kappa = 1 0 ^ { 3 }$ with $\eta = 5$ give almost the same result.

Stronger consolidation reduces forgetting at almost no cost in accuracy on the current task. With 1500 steps per task, raising κ from $1 0 ^ { 3 }$ to $1 0 ^ { 4 }$ lowers the forgetting from 0.298 to 0.123 and raises the final accuracy from 0.733 to 0.868, while the accuracy after training changes only from 0.972 to 0.966. Consolidation slows down mainly the synapses that have changed strongly in the recent past, so the synapses that a new task needs still learn at almost the full rate.

Table 6: Ablations of recirculation on MNIST. Each group compares the default model with changes of one element, at the consolidation strength κ and the number of training steps per task given in the row. No task code: the network receives no task code in training and testing. Readout from all layers: the global outputs of both layers reach the readout. Single layer: the network has only the first layer, whose global outputs reach the readout. 80% of branches gated: each code unit gates the entry segment of a branch with probability 0.8 instead of 0.5. The measures are those of Table 4.
<table><tr><td>Configuration</td><td>κ</td><td>Steps</td><td>After training</td><td>Final</td><td>Forgetting</td></tr><tr><td>Default</td><td> $\overline { { 1 0 ^ { 4 } } }$ </td><td>500</td><td>0.944</td><td>0.939</td><td>0.009</td></tr><tr><td>No task code</td><td> $1 0 ^ { 4 }$ </td><td>500</td><td>0.895</td><td>0.195</td><td>0.875</td></tr><tr><td>Readout from all layers</td><td>10⁴4</td><td>500</td><td>0.969</td><td>0.297</td><td>0.840</td></tr><tr><td>Default</td><td>104</td><td>1500</td><td>0.966</td><td>0.868</td><td>0.123</td></tr><tr><td>Single layer</td><td>104</td><td>1500</td><td>0.980</td><td>0.203</td><td>0.971</td></tr><tr><td>Default</td><td>103</td><td>1500</td><td>0.972</td><td>0.733</td><td>0.298</td></tr><tr><td>80% of branches gated</td><td> $1 0 ^ { 3 }$ </td><td>1500</td><td>0.972</td><td>0.602</td><td>0.463</td></tr></table>

The other parameters have similar efects. An importance that never decays $( \nu = 0 )$ lowers the forgetting at $\kappa = 1 0 ^ { 3 }$ from 0.298 to 0.230. A learning rate of $\eta = 5$ instead of 20 has almost the same efect as the stronger consolidation, with a forgetting of 0.126 and an accuracy after training of 0.966. Shorter training gives the best final accuracy: with $\kappa = 1 0 ^ { 4 }$ and 500 steps per task, the forgetting is 0.009 and the final accuracy 0.939.

Across all settings, the accuracy after training stays between 0.944 and 0.972, within 0.05 of gradient-based learning, while the forgetting ranges from 0.009 to 0.298, less than a third of the 0.995 of gradient-based learning. The final accuracy ranges from 0.733 to 0.939, compared with 0.198. Recirculation thus ofers a continuous control of the balance between learning new tasks and keeping earlier ones, through parameters that each synapse applies to its own changes. Gradient-based learning without consolidation lies at one extreme of this trade-of, where each new task is learned best and all earlier tasks are lost (Figure 9, upper left).

## 4.4 Ablations

The ablations presented in this section change one element of the model at a time. Table 6 lists the ablations of recirculation and Table 7 those of the local rule, and Figure 10 shows an example run of each learning method.

The task code is essential for remembering earlier tasks. Without it, recirculation still learns each task, with an accuracy after training of 0.895, but it forgets 0.875 of it, and the final accuracy falls to 0.195. The accuracy matrix then becomes almost diagonal (Figure 10a), as that of gradient-based learning (Figure 8a). The local rule behaves in the same way, with a forgetting of 0.889 without the code. With the code, each task drives its own subset of branches and neurons, and the gating limits how much the weights of the other subsets change (Section 3.5). Gradient-based learning forgets earlier tasks even with the task code (Section 4.2), and so does the local rule without consolidation (forgetting 0.777, Table 7), so it is the combination of the task code and consolidation that helps remember earlier tasks.

Table 7: Ablations of the local rule on MNIST with the task code, with 1500 steps per task and consolidation strength $\kappa = 1 0 ^ { 3 }$ . No reward part: $\eta _ { \mathrm { r } } = 0$ in Eq. (29). No branch part: $\eta _ { \mathrm { b } } = 0$ in Eq. (30). Readout only: both parts are removed, which gives the fixed network of Section 4.1. No consolidation: $\kappa = 0$ . The measures are those of Table 4.
<table><tr><td>Configuration</td><td>After training</td><td>Final</td><td>Forgetting</td><td>Early reward</td></tr><tr><td>Full rule</td><td>0.939</td><td>0.934</td><td>0.008</td><td>0.520</td></tr><tr><td>No reward part</td><td>0.939</td><td>0.933</td><td>0.009</td><td>0.521</td></tr><tr><td>No branch part</td><td>0.936</td><td>0.931</td><td>0.008</td><td>0.510</td></tr><tr><td>Readout only</td><td>0.936</td><td>0.931</td><td>0.008</td><td>0.510</td></tr><tr><td>No consolidation</td><td>0.947</td><td>0.326</td><td>0.777</td><td>0.572</td></tr><tr><td>No task code</td><td>0.934</td><td>0.223</td><td>0.889</td><td>0.421</td></tr></table>

![](images/6648c350691ad5d7803ddcea4cf1185a6925bb8c1982779d598c01994a5bc377.jpg)

![](images/7f02edfda0d1551aded7686763d70b7796dfef46acc37d65eb7226e752649587.jpg)

![](images/58281699f75374fadd537cf3148734ccb65a0dc882ad289d5f029cc351caf92c.jpg)  
(a) Recirculation without the task code, 500 steps per task

![](images/a25a6db016432f1e9e6462bd7fe6df372f4a383a1d0d62c8eb0fbd6a2aa0a2d1.jpg)

![](images/3d01e965285ae35c966231dd85ba66d7099f18728b71bfb9f7050469a284c5e6.jpg)

![](images/399c31882c6366c7ab794103617534c2b2fc0a45b3f0c1285f00812feec0ee6a.jpg)  
(b) Local rule with the task code, 1500 steps per task  
Figure 10: Example runs of ablations on MNIST. The panels are those of Figure 8.

The readout usually receives only the global outputs of the last layer. When the outputs of the first layer also reach the readout, recirculation acquires each task better, with an accuracy after training of 0.969, but the forgetting rises to 0.840. A network with a single layer forgets 0.971. A likely reason is that the global outputs of the second layer depend on the task code twice, through the gating of their own branches and through the gated outputs of the first layer that they receive, so that they are more specific to each task than those of the first layer.

Table 8: Test accuracy $R _ { t , k }$ on Iris. Rows: after training on task t. Columns: tested task k. The entries above the diagonal are measured before task k is trained.
<table><tr><td>After training on</td><td>Setosa, versicolor</td><td>Versicolor, virginica</td><td>All three species</td></tr><tr><td>Setosa, versicolor</td><td>1.000</td><td>0.500</td><td>0.667</td></tr><tr><td>Versicolor, virginica</td><td>0.980</td><td>0.947</td><td>0.964</td></tr><tr><td>All three species</td><td>0.980</td><td>0.947</td><td>0.964</td></tr></table>

Gating half of the branches is more efective than gating most of them. With 80% of the branches gated, the accuracy after training is unchanged, but the forgetting rises from 0.298 to 0.463. The default fraction of 0.5 is therefore used in all other experiments.

With the local rule, the structure of the network protects the earlier tasks by itself. Even when only the readout learns, the forgetting is 0.008, and the final accuracy is 0.931 (Table 7). The two parts of the local rule change these results little: removing the reward part, the branch part or both changes every accuracy by at most 0.003. The credit signal of the local rule is weak, because each synapse correlates one broadcast error with the random spike of its own neuron, and the random spikes of all other neurons add noise to this correlation (Section 3.6). Consolidation is what retains the earlier tasks when the rule does change the weights: without it, the accuracy after training rises to 0.947, but the forgetting rises to 0.777.

Recirculation keeps the protection given by the structure of the network while it trains the neurons. With 500 steps per task, its forgetting is as low as that of the fixed network (0.009 compared with 0.008), and its final accuracy is higher (0.939 compared with 0.931). With 1500 steps per task, its accuracy after training rises to 0.966, compared with 0.936 for the fixed network. In recirculation, every neuron receives its own requested change through the second pass, instead of a single broadcast error, so the training of the neurons increases the accuracy on each task.

## 4.5 Iris

On Iris, the network has a single layer of 64 neurons and receives no task code. Table 8 gives the test accuracy on each task after training on each task, and Figure 11 shows an example run.

The network learns the first task, setosa and versicolor, perfectly. The training reward rises to about 0.9 within roughly 25 steps (Figure 11, right), close to the limit set by the random actions. Before the second task is trained, the accuracy on it is already 0.500, because the versicolor flowers of the second task are recognized from the first task.

![](images/eeb3f0b5a6427801cbeec5d87d0db1e652985c6ce293060599cb44e9eb9f403e.jpg)  
Figure 11: Example run of recirculation on Iris. Task 0: setosa and versicolor. Task 1: versicolor and virginica. Task 2: all three species. The panels are those of Figure 8.

The second task, versicolor and virginica, contains the two species that overlap. The network learns it to an accuracy of 0.947, and it keeps the first task, although no setosa flower is shown during the second task and no task code indicates which task is being trained: the accuracy on setosa and versicolor decreases only from 1.000 to 0.980. When the second task starts, the training reward drops to about 0.5, because virginica flowers are first classified as versicolor, and it recovers within a few tens of steps.

After the second task, the network already classifies all three species with an accuracy of 0.964, although it has never seen setosa and virginica together. The two tasks, learned one after the other, combine into a three-way classifier without any joint training. The third task, which shows all three species together, leaves every test accuracy unchanged, and its training reward stays high from its first step.

Recirculation also retains earlier tasks when they are presented in a diferent order. When the network learns versicolor and virginica first, then setosa and versicolor, and finally setosa and virginica, the accuracy on the last task is already 0.900 before it is trained, the final accuracy is 0.933, and the forgetting is 0.083.

## 4.6 Discussion

The results point to three features of the model:

• A large part of the protection of earlier tasks follows from the structure of the neurons rather than from an added mechanism;

• The neurons can be trained with signals available at each synapse without losing this protection;

• The balance between learning new tasks and keeping earlier ones is set by parameters that each synapse applies to its own changes.

In conventional networks, protection against forgetting is usually added to the learning algorithm, as a penalty on weight changes, a mask on the gradients or a bufer of stored examples. In the proposed model, much of it is a consequence of the neuron model. The task code acts through shunting inhibition at the entry of a branch, and because inhibition is divisive and acts where it is placed (Eq. 5), a gated branch is separated from the trunk in both passes. In the first pass, it contributes little to the global output. In the second pass, it receives little of the requested change, because a strongly inhibited segment has small entries in the spread vector ρ. The same inhibition that routes a task through a subset of branches thus shields the other subsets from learning, and no mask has to be stored or applied to the weight changes. The readout adds to this protection, because it has no bias and changes only the value of the chosen action, in proportion to the global outputs that are active. With these elements alone, a network whose neurons do not learn keeps its earlier tasks almost completely, with a forgetting below 0.01.

The same structure does not protect earlier tasks under gradient-based learning, whose forgetting is almost complete even with the task code. Inside each neuron, the exact gradient is attenuated in gated branches in the same way as the second pass of recirculation, because both are computed with the same matrix. Some properties of the gradient-based learner work against this attenuation. Adam normalizes the update of each weight by the size of its recent gradients, so small gradients in gated branches still produce steps of normal size. The thresholds are learned and act as biases, which shift with every task. Finally, nothing slows down the weights on which earlier tasks rely. In recirculation, the update of each weight is proportional to the shift of its own segment, so gated branches change little, the thresholds remain fixed, and consolidation slows down the weights that have already changed. Before consolidation, recirculation follows the direction of the gradient, except for the efect of the weights on the back-boost, but it keeps the attenuation that the structure imposes. The neurons can therefore be trained, which raises the accuracy on each task above that of the fixed network, while most of the protection is kept. The local rule marks the other extreme: its broadcast error gives each synapse too weak a signal to improve on the fixed network.

All quantities used by recirculation are available at the synapse: the presynaptic activity, the driving force and the change of the activation of its own segment between the two passes. Because the matrix of each neuron is symmetric, requested changes spread through the tree by the same equations as activity, and no separate backward computation is needed inside a neuron. Consolidation is computed by each synapse from its own history. It needs no task boundaries, no stored copies of earlier weights and no estimate of how much the loss depends on each weight. Because it acts on the learning rate and not on the weights, it never freezes a synapse: a synapse that stops changing slowly regains its plasticity as its importance decays. The cost of a neuron is dominated by its synapses, as for a point neuron, and the exact solution of its tree adds a term linear in the number of segments; the second pass costs about as much as the first.

The model combines in a single neuron mechanisms that are usually treated separately. The task code is a form of context-dependent gating (Masse et al., 2018), applied to the branches within each neuron by dendritic inhibition instead of to whole units. Consolidation follows the idea of elastic weight consolidation (Kirkpatrick et al., 2017) and synaptic intelligence (Zenke et al., 2017), but as a metaplastic change of the learning rate (Abraham and Bear, 1996) that each synapse computes from its own changes. The learning rule belongs to the family of recirculation (Hinton and McClelland, 1988; O’Reilly, 1996) and equilibrium propagation (Scellier and Bengio, 2017), in which weights change according to the diference

between two states of the network. Here, the two states are steady states of passive dendritic trees, and the network learns from reward instead of from target outputs. The masking of gated branches and the locality of the updates, which other approaches add as separate components, follow from the biophysics of the neuron.

In terms of limitations, on MNIST the protection of earlier tasks relies on the task code. Without it, the network forgets almost as much as with gradient-based learning (0.875 compared with 0.995), so on such data the model in its present form addresses the task-incremental setting. Deriving the code from the input, or from the global outputs of the network itself, would extend the protection to the class-incremental setting. The model also trades some plasticity for retention: each task is learned more slowly and to a slightly lower accuracy than with gradient-based learning. The protection also requires that the readout read only the last layer, because the global outputs of the first layer are not protected as well. The thresholds are fixed after calibration on images of the first task, so later tasks are represented with thresholds that do not reflect their statistics. Finally, each contact returns requested changes with a strength equal to its forward weight. This is the only assumption of the learning rule that goes beyond the single synapse, and it plays the role that weight symmetry plays in backpropagation. Natural extensions of the model are a task code generated by the network, tasks with multi-step episodes, for which the learning rule is already defined, and deeper networks in which every layer contributes to the protection.

## 5 Conclusions

This paper introduced a network of unipolar neurons that learns a sequence of tasks only from reward. Each neuron is modeled as a tree of passive segments with graded local outputs and one global output. The network learns with recirculation, a two-pass rule in which each synapse uses only quantities available at its own location. A task code inhibits a diferent subset of branches in each task, and consolidation slows down the synapses that have changed strongly in the recent past.

On the sequence of MNIST tasks with the task code, recirculation learned each new task to an accuracy close to that of gradient-based training and remembered the earlier tasks very well. Gradient-based training of the same network forgot the earlier tasks almost completely. The consolidation strength, the learning rate and the length of training determine the balance between learning new tasks and retaining earlier ones. The ablations showed that the task code is necessary for retention and that, in general, the readout should receive only the global outputs of the last layer. On Iris, the network learned tasks with overlapping classes in sequence without a task code, and the tasks combined into a classifier of all three species without joint training.

The most important extension is to infer the task code instead of providing it. This would make the model applicable to the class-incremental setting, in which the identity of the task is unknown at test time. One possibility is a separate group of neurons whose global outputs identify the context of the input. Another is to assign a new code when the error of the reward prediction increases sharply, since such an increase can indicate a change of task.

A second direction is learning in tasks with several steps and delayed reward. The learning rule includes a discount factor, but all experiments used single-step episodes. The control of a simulated flexible arm, similar to the arms of an octopus, could be a suitable test case. Local outputs could coordinate neighboring parts of the arm, and global outputs could convey commands across the whole network.

Several fixed parts of the model could also be learned. Currently, the locations of the synapses and the shapes of the trees are random. Structural plasticity, i.e., the formation and removal of synapses (Leon, 2026a), could place inputs on the branches where they contribute most to the task.

The requested changes could be returned through separate feedback weights instead of the forward weights. This would show how much of the accuracy and of the protection depends on the symmetry between the two directions. A theoretical analysis could also relate the interference between tasks to the strength of the inhibition from the task code and to the fraction of branches it reaches.

Finally, the model should be evaluated with deeper networks, longer task sequences and more complex data.

## References

Wicklife C. Abraham and Mark F. Bear. Metaplasticity: The plasticity of synaptic plasticity. Trends in Neurosciences, 19(4):126–130, 1996. doi: 10.1016/s0166-2236(96)80018-x.

Hoger Amin, Anthi A. Apostolopoulou, Raquel Su´arez-Grimalt, Eleftheria Vrontou, and Andrew C. Lin. Localized inhibition in the Drosophila mushroom body. eLife, 9:e56954, 2020. doi: 10.7554/eLife.56954.

Eseoghene Ben-Iwhiwhu, Saptarshi Nath, Praveen K. Pilly, Soheil Kolouri, and Andrea Soltoggio. Lifelong reinforcement learning with modulating masks. Transactions on Machine Learning Research, 2023. URL https://openreview.net/forum?id=V7tahqGrOq.

Theodore H. Bullock and G. Adrian Horridge. Structure and Function in the Nervous Systems of Invertebrates. W. H. Freeman, San Francisco, 1965.

Malcolm Burrows and Melody V. S. Siegler. Graded synaptic transmission between local interneurones and motor neurones in the metathoracic ganglion of the locust. The Journal of Physiology, 285:231–255, 1978. doi: 10.1113/jphysiol.1978.sp012569.

Arslan Chaudhry, Puneet K. Dokania, Thalaiyasingam Ajanthan, and Philip H. S. Torr. Riemannian walk for incremental learning: Understanding forgetting and intransigence. In Proceedings of the European Conference on Computer Vision (ECCV), pages 532–547, 2018.

Richard B. Dewell, Ying Zhu, Margaret Eisenbrandt, Richard Morse, and Fabrizio Gabbiani. Contrast polarity-specific mapping improves eficiency of neuronal computation for collision detection. eLife, 11:e79772, 2022. doi: 10.7554/eLife.79772.

Mohamed Elsayed and A. Rupam Mahmood. Addressing loss of plasticity and catastrophic forgetting in continual learning. In International Conference on Learning Representations (ICLR), 2024. URL https://openreview.net/forum?id=sKPzAXoylB.

Ronald A. Fisher. The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2):179–188, 1936.

Nicolas Fr´emaux and Wulfram Gerstner. Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules. Frontiers in Neural Circuits, 9:85, 2016. doi: 10.3389/fncir.2015.00085.

Lukas N. Groschner, Jonatan G. Malis, Birte Zuidinga, and Alexander Borst. A biophysical account of multiplication by a single neuron. Nature, 603(7899):119–123, 2022. doi: 10.1038/s41586-022-04428-3.

Omar A. Hafez, Benjamin Escribano, Rouven L. Ziegler, Jan J. Hirtz, Ernst Niebur, and Jan Pielage. The cellular architecture of memory modules in Drosophila supports stochastic input integration. eLife, 12:e77578, 2023. doi: 10.7554/eLife.77578.

Michael Hines. Eficient computation of branched nerve equations. International Journal of Bio-Medical Computing, 15(1):69–76, 1984. doi: 10.1016/0020-7101(84)90008-4.

Geofrey E. Hinton and James L. McClelland. Learning representations by recirculation. In Dana Z. Anderson, editor, Neural Information Processing Systems, pages 358–366, New York, 1988. American Institute of Physics.

Abhiram Iyer, Karan Grewal, Akash Velu, Lucas Oliveira Souza, Jeremy Forest, and Subutai Ahmad. Avoiding catastrophe: Active dendrites enable multi-task learning in dynamic environments. Frontiers in Neurorobotics, 16:846219, 2022. doi: 10.3389/fnbot.2022. 846219.

Christos Kaplanis, Murray Shanahan, and Claudia Clopath. Continual reinforcement learning with complex synapses. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 2497–2506. PMLR, 2018. URL https://proceedings.mlr. press/v80/kaplanis18a.html.

Khimya Khetarpal, Matthew Riemer, Irina Rish, and Doina Precup. Towards continual reinforcement learning: A review and perspectives. Journal of Artificial Intelligence Research, 75:1401–1476, 2022. doi: 10.1613/jair.1.13673.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521–3526, 2017. doi: 10.1073/pnas.1611835114.

Christof Koch. Biophysics of Computation: Information Processing in Single Neurons. Oxford University Press, New York, 1999.

Philipp Kr¨ahenb¨uhl, Carl Doersch, Jef Donahue, and Trevor Darrell. Data-dependent initializations of convolutional neural networks. In International Conference on Learning Representations (ICLR), 2016.

Janne K. Lappalainen, Fabian D. Tschopp, Sridhama Prakhya, Mason McGill, Aljoscha Nern, Kazunori Shinomiya, Shin-ya Takemura, Eyal Gruntman, Jakob H. Macke, and Srinivas C. Turaga. Connectome-constrained networks predict neural activity across the fly visual system. Nature, 634(8036):1132–1140, 2024. doi: 10.1038/s41586-024-07939-3.

Florin Leon. Biologically inspired mechanisms for facilitating grokking in multilayer perceptrons. arXiv preprint arXiv:2608.28184, 2026a. doi: 10.48550/arxiv.2608.28184.

Florin Leon. Sparse neural code representations for reinforcement learning with linear action-value function approximation. In Proceedings of the 18th International Conference on Computational Collective Intelligence (ICCCI 2026), Heraklion, Greece, volume 16908 of Lecture Notes in Computer Science, pages 230–243, Cham, 2026b. Springer. doi: 10.1007/978-3-032-36868-3 16.

Tony X. Liu, Pasha A. Davoudian, Kristyn M. Lizbinski, and James M. Jeanne. Connectomic features underlying diverse synaptic connection strengths and subcellular computation. Current Biology, 32(3):559–569.e5, 2022. doi: 10.1016/j.cub.2021.11.056.

Nicolas Y. Masse, Gregory D. Grant, and David J. Freedman. Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization. Proceedings of the National Academy of Sciences, 115(44):E10467–E10475, 2018. doi: 10.1073/pnas. 1803839115.

Matthias Meier and Alexander Borst. Extreme compartmentalization in a Drosophila amacrine cell. Current Biology, 29(9):1545–1550.e2, 2019. doi: 10.1016/j.cub.2019.03.070.

Randall C. O’Reilly. Biologically plausible error-driven learning using local activation diferences: The generalized recirculation algorithm. Neural Computation, 8(5):895–938, 1996. doi: 10.1162/neco.1996.8.5.895.

Frank Rosenblatt. The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65(6):386–408, 1958. doi: 10.1037/ h0042519.

Benjamin Scellier and Yoshua Bengio. Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. Frontiers in Computational Neuroscience, 11: 24, 2017. doi: 10.3389/fncom.2017.00024.

Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. MIT Press, Cambridge, MA, 2nd edition, 2018.

Gido M. van de Ven, Tinne Tuytelaars, and Andreas S. Tolias. Three types of incremental learning. Nature Machine Intelligence, 4(12):1185–1197, 2022. doi: 10.1038/ s42256-022-00568-3.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3–4):229–256, 1992. doi: 10.1007/bf00992696.

John Z. Young. The Anatomy of the Nervous System of Octopus vulgaris. Clarendon Press, Oxford, 1971.

Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 3987–3995. PMLR, 2017.