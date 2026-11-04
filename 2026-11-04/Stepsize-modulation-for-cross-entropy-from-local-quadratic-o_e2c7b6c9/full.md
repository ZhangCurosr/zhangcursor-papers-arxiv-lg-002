# Stepsize modulation for cross entropy from local quadratic optimization

Bruno Bauwens<sup>∗</sup>, Viacheslav Krivodub<sup>†</sup>, and Bingyu Zhang

November 4, 2026

## Abstract

Two methods are presented that perturb a given learning rate schedule in stochastic gradient descent with cross entropy loss. Both methods use an extra forward pass to estimate various derivatives of the minibatch loss.

The first method selects a value η by linearising the softmax values and minimizing the squared error $\begin{array} { r } { \frac { 1 } { 2 } | | y - q - \eta q ^ { \prime } | | _ { F } ^ { 2 } } \end{array}$ , where q is the matrix representing the softmax values of the minibatch and y is the matrix with one-hot encodings of the labels. The second method considers a quadratic Taylor approximation of the minibatch loss under the outer approximation of the Hessian. Thus, it minimizes $\begin{array} { r } { L ( 0 ) + \eta L ^ { \prime } ( 0 ) + \frac { 1 } { 2 } \eta ^ { 2 } L ^ { \prime \prime } ( 0 ) } \end{array}$ . Here, the derivatives $L ^ { \prime }$ and $L ^ { \prime \prime }$ are computed from the derivatives of the logits (using the outer approximation of the Hessian of the loss). We observe that both methods give very similar values of $\eta .$

Finally, these estimated η are used to slightly increase or decrease the learning rate of an existing schedule. The method is evaluated on Cifar10 and Cifar100 datasets for training Resnets, where they decrease the loss faster than the schedule that is perturbed and increase monotonicity. For Cifar10 it also leads to a small improvement of test accuracy, but in Cifar100 test accuracy remains the same.<sup>1</sup>

## 1 Introduction

The selection of the learning rate is a very consequential decision when training neural nets. Despite much research in automatically selecting step sizes, typical training with convolutional nets is done with some fixed schedule for which maximum learning rate and weight decays are tuned. In this paper, we ask whether one can modulate the standard cosine schedule using batch dependant stepsizes to improve training. More precisely, can we inspect certain properties of a minibatch and use it to decide whether a bigger or smaller stepsize should be used compared to the scheduled value of the epoch. Can this speed up training and increase test accuracy?

## The Polyak stepsize

Consider a loss function L(θ). We aim to find a value of θ that makes L as small as possible. The gradient descent method starts from an arbitrary $\theta _ { 0 }$ and iteratively computes

$$
\theta _ { t + 1 } = \theta _ { t } - \eta _ { t } \nabla L ( \theta _ { t } ) .
$$

Already in 1969, Polyak proposed to automatically select the stepsize using a linear approximation of the loss function [24]. Consider the linear approximation

$$
L ( \theta _ { t } - w ) \approx L ( \theta _ { t } ) - \nabla L ( \theta _ { t } ) \cdot w .
$$

![](images/9327ae86e13513b065bf1a0d232ad67fc9c8b04a011e22ac0970f54c6e02868d.jpg)  
Figure 1: The Polyak stepsize over estimates on convex strictly positive functions

For $w = \eta _ { t } \nabla L ( \theta _ { t } )$ , this simplifies to

$$
L ( \theta _ { t + 1 } ) \approx L ( \theta _ { t } ) - \eta _ { t } \vert \vert \nabla L ( \theta _ { t } ) \vert \vert ^ { 2 } .
$$

Solving for $L ( \theta _ { t + 1 } ) \approx 0$ , gives

$$
\eta _ { t } = \frac { L ( \theta ) } { | | \nabla L ( \theta _ { t } ) | | ^ { 2 } } .
$$

It was directly clear that if L is smooth and strongly convex, then gradient descent with this stepsize converges exponentially fast in t [24]. A bit later, Polyak proved that for all convex functions (possibly non-smooth) this method has $O ( 1 / { \sqrt { t } } )$ convergence: lim inf $\sqrt { t } ( L ( \theta _ { t } ) - L ( \theta ^ { * } ) ) = 0$

In machine learning, the above gradients are estimated on small random selections of the training data, called minibatches. This means that for each step, a random approximation $L _ { t } ( \theta )$ of the loss is generated. The Polyak stepsize under this function is called the stochastic Polyak stepsize, for which $O ( 1 / t )$ convergence was proven under star-convexity of the loss [9, theorem 4.4], and this is locally satisfied in deep neural nets [15, 33].

Unfortunately, the Stochastic Polyak stepsize is very unstable and can spike. This often happens in the beginning of training when $L ( \theta _ { t } )$ is large and the gradients are small (especially if the softmax is used). In the case of cross entropy loss, the curve does not reach zero and is locally convex for most minibatches. Hence, the Polyak stepsize overestimates, as illustrated in figures 1 and 2.

For this reason, many variants for practical use have been proposed [3, 19, 14, 20, 23, 21]. However, they lead often lead to unstable training, especially when combined with momentum. Moreover, they are not able to replace manual learning rate selection because of sensitive dependencies on newly introduced hyper parameters. (Also for LLMs there are such reports [22].) Perhaps, an exception is the momo scheduler from [26], which is designed to optimize the loss aggregated from the last few minibatches. It works well with momentum and weight decay. Moreover, it has has only a single hyper parameter that is easy to tune, but it has not been tested systematically against competitors.

## Second order methods

Approaches with test points. To select the learning rate, one may consider a quadratic approximation of the loss,

$$
\begin{array} { r } { \mathrm { L } ( \theta _ { t } - w ) \approx \mathrm { L } ( \theta _ { t } ) - w \cdot \nabla \mathrm { L } ( \theta _ { t } ) + \frac { 1 } { 2 } w ^ { \mathsf { T } } \mathrm { H } _ { \mathrm { L } } w . } \end{array}
$$

Each optimization method in a typical machinelearning optimizer defines a direction, for example, for basic SGD the direction is $w = - \eta \nabla L ( \theta _ { t } )$ or with momentum, it is $w = - \eta \nabla L ( \theta _ { t } ) - \mu v _ { t - 1 }$ . After substituting in $L _ { : }$ , one obtains an equation of the form $f ( \eta ) = a \eta ^ { 2 } + $ bη $+ c$ from which an optimal stepsize $\eta$ can be computed. In [5, 34], the coeficients $a , b , c$ are obtained from 3 evaluations of $f .$ Since gradient descent already provides 1 evaluation, this method makes 2 extra forward evaluations, and hence, is %66 slower (assuming a backwards pass takes twice the time of a forward pass). These methods reduce the training loss very fast, but at the expense of the test loss [34]. A similar technique is applied in [5] in which the computed learning rate is smoothed using a moving average and improvements on some fine tuning tasks are reported.<sup>2</sup>

In [8] it is noted that $w \cdot \nabla L ( \theta _ { t } )$ is fast to compute, thus to find the coeficients $a , b , c ,$ only 1 extra forward evaluation is needed. The paper presents an algorithm that is essentially parameter free and provably converges to a stationary point for Lipschitz functions. Unfortunately, they do not report test performance.

Barzilai-Borwein stepsizes. Another approach is to calculate the solution using the available gradient information, thus one needs

$$
\eta _ { t } = \frac { v \cdot \nabla L ( \theta _ { t } ) } { v ^ { \mathsf { T } } \mathrm { H } _ { L } v } .\tag{1}
$$

In the Barzilai-Borwein method [2] one assumes $v = \nabla L ( \theta _ { t } )$ and estimates the above value by

$$
\eta _ { t } = \frac { | | \theta _ { t } - \theta _ { t - 1 } | | ^ { 2 } } { \left( \theta _ { t } - \theta _ { t - 1 } \right) \cdot \left( \nabla L ( \theta _ { t } ) - \nabla L ( \theta _ { t - 1 } ) \right) } .
$$

This inspired a number of automated learning rate methods [6, 28, 30]. Again, the method sufers from instability, but [30] results come close to standard methods such as SGD with cosine annealing.

1st order estimates of the 2nd order. In [1], the stepsize (1) is estimated directly with the already computed gradient. The denominator involves the Hessian, which is expensive to compute. However, typical networks of the last decade have wide hidden layers, and this allows for a good approximation with gradients. This is because preactivations of the output neurons are locally linear functions, as explained in many papers about the neural tangent kernels, see for example [12, 17, 25].

For this, consider the loss as a function of output preactivations

$$
L ( \theta ) = { \mathcal { L } } ( F ( \theta ) ) .
$$

The Hessians of L can be written using the Hessians of L and the Hessians of the output preactivations $F _ { a }$ Let $\begin{array} { r } { \mathrm { J } _ { a \mu } = \frac { \partial F _ { a } } { \partial \theta _ { \mu } } } \end{array}$ be the Jacobian of F.

$$
\mathrm { H } _ { L } = \mathrm { J } \mathrm { H } _ { \mathscr { L } } \mathrm { J } + \sum _ { a } \frac { \partial \mathscr { L } } { \partial F _ { a } } \mathrm { H } _ { F _ { a } } .
$$

For large neural nets, the spectral norm of the Hessians $F _ { a }$ are proportional to O(depth/width), see [25]. In the outer product approximation of $\mathrm { H } _ { L }$ all these Hessian are neglected and only the 1st term remains, see for example [4, p 243, sect.8.1.6]. In [1] this first term is computed using the available gradient, and (1) is obtained with negligible computational overhead. They show in a few experiments that this reduces the train error better than SGD, but do not report test errors.

Below, we propose a method that uses a single test point to estimate this 2nd order stepsize in the same way.

## Other methods

In the paper [18], a wide range of optimizers is modified in a generic way: in each minibatch, the existing learning rate schedule is randomly increased or decreased by a few percentages. Despite its simplicity, this method is reported to consistently increase test error, including on Cifar10 and Resnet18 experiments.

This easy method is similar to other regularization methods like dropout [27] or neftune [13] that increase randomness during training. The motivation is that solutions with wide minima have better generalization and it is argued that these perturbations can help to escape narrow local minima. See [31] for this theory.

Finally, the paper [7] uses a complex algorithm from the field of online convex optimization to select the learning rate. [todo: test error?]

## 2 Two proposed stepsizes

We aim to minimize to optimize parameters θ to train a neural nets. For this, a standard run of stochastic gradient descend is applied, which means that we start from a randomly generated $\theta _ { 0 }$ and at each step, a

minibatch loss $\mathrm { L } _ { t } ( \theta )$ is used to update $\theta _ { t }$ using

$$
\begin{array} { r l } & { v _ { t } = \mu v _ { t - 1 } + \eta _ { t } \nabla \operatorname { L } _ { t } ( \theta _ { t - 1 } + \mu v _ { t - 1 } ) + \lambda \theta _ { t - 1 } , } \\ & { \theta _ { t } = \theta _ { t - 1 } - v _ { t } , } \end{array}\tag{2}
$$

where $\mu$ and $\lambda$ are the parameters for momentum and weight decay.

Notation. Consider a neural net with $n _ { \mathrm { o u t } }$ output neurons and a train set whose examples are denoted by $( x _ { \alpha } , y _ { \alpha } )$ with Greek indices. For each training example, the label $y _ { \alpha }$ is given as a one-hot encoding of the correct label, thus $y _ { \alpha c } = 1 { \mathrm { ~ i f ~ } } x _ { \alpha }$ belongs to the c-th class. The preactivations of the output neurons are denoted by the vector $f ( x _ { \alpha } , \theta )$ , where θ is the list of all trainable weights of the net.

We now define the minibatch loss $\mathrm { L } _ { t }$ . Let

$$
\begin{array} { l } { { z _ { \alpha i } = f _ { i } ( x _ { \alpha } , \theta ) } } \\ { { q _ { \alpha i } = \displaystyle \frac { \exp { z _ { \alpha i } } } { \sum _ { j \le n _ { \mathrm { o u t } } } \exp { z _ { \alpha j } } } } } \end{array}
$$

For a list of train examples $S _ { t } .$ , the t-th minibatch loss is

$$
\mathrm { L } _ { t } ( \theta ) = - \sum _ { \alpha \in S _ { t } } \sum _ { i \leq n _ { \mathrm { o u t } } } y _ { \alpha i } \ln q _ { \alpha i } .
$$

Notation. We write $\textstyle \sum _ { \alpha }$ to denote $\textstyle \sum _ { \alpha \in S _ { t } }$ and sums over indices $i , j , k$ range from 1 to $n _ { \mathrm { o u t } }$

The second order outer approximated stepsize (so-dz)

Fix t and let $L ( \tilde { \eta } ) = \mathrm { L } _ { t } ( \theta _ { t } - \tilde { \eta } v _ { t } )$ . We select the stepsize $\eta _ { \mathrm { s o } }$ to minimize the quadratic approximation of $\mathrm { L } _ { t } ( \tilde { \eta } )$ . As explained in subsection 1, we may assume that $z _ { \alpha i }$ is linear in $\tilde { \eta }$ and we can obtain the derivative from only the gradient. Still, to increase stability, we use a test point.

Let us explain this plan in more details. We have

$$
L _ { t } ( \eta ) \approx L _ { t } ( 0 ) - \eta L ^ { \prime } ( 0 ) + \frac { \eta ^ { 2 } } { 2 } L ^ { \prime \prime } ( 0 ) .
$$

To compute $L ^ { \prime } ( 0 )$ we view L as a function of the output preactivations $z _ { \alpha i }$ and apply the chain rule. The 2nd derivative $L ^ { \prime \prime } ( 0 )$ is computed similarly, while the 2nd derivatives of $z _ { \alpha i }$ are neglected, see appendix A for details. Let $\delta _ { i j }$ be the Kronecker function, which equals 1 if $i = j$ and 0 otherwise.

$$
\eta _ { \mathrm { s o } } ^ { * } = \frac { L ^ { \prime } ( 0 ) } { L ^ { \prime \prime } ( 0 ) } = \frac { \sum _ { \alpha , i } ( y _ { \alpha i } - q _ { \alpha i } ) \frac { \mathrm { d } z _ { \alpha i } } { \mathrm { d } \eta } } { \sum _ { \alpha , i , j } ( \delta _ { i j } - q _ { \alpha i } ) q _ { \alpha j } \frac { \mathrm { d } z _ { \alpha i } } { \mathrm { d } \eta } \frac { \mathrm { d } z _ { \alpha j } } { \mathrm { d } \eta } } .
$$

Finally, in this expression, $\frac { \mathrm { d } z _ { \alpha j } } { \mathrm { d } \eta }$ is estimated by evaluating the output preactivations z with a test update of η using a fixed learning rate $\dot { \eta } ^ { \mathrm { t e s t } }$ . More precisely, let

$$
\begin{array} { r } { \theta _ { t } ^ { \mathrm { t e s t } } = \theta _ { t } - \mu v _ { t } - \eta ^ { \mathrm { t e s t } } \nabla \mathrm { L } ( \theta _ { t } ) - \lambda \theta _ { t } . } \end{array}
$$

Let $q _ { \alpha } ^ { \mathrm { t e s t } }$ and $q _ { \alpha }$ be the softmax values evaluated with parameters $\theta _ { t } ^ { \mathrm { t e s t } }$ and $\theta _ { t }$ . Then, estimate $\frac { \mathrm { d } z _ { \alpha } } { \mathrm { d } \eta }$ by $( z _ { \alpha } ^ { \mathrm { t e s t } } - z _ { \alpha } ) / \eta ^ { \mathrm { t e s t } }$ . Thus,

$$
\hat { \eta } _ { \mathrm { s o } } = \eta ^ { \mathrm { t e s t } } \frac { \sum _ { \alpha , i } ( y _ { \alpha i } - q _ { \alpha i } ) ( z _ { \alpha i } ^ { \mathrm { t e s t } } - z _ { \alpha i } ) } { \sum _ { \alpha , i , j } ( \delta _ { i j } - q _ { \alpha i } ) q _ { \alpha j } ( z _ { \alpha i } ^ { \mathrm { t e s t } } - z _ { \alpha i } ) ( z _ { \alpha j } ^ { \mathrm { t e s t } } - z _ { \alpha j } ) } .\tag{3}
$$

This finishes the construction. The pseudocode is given in algorithm 1. In the experiments, we fixed $\eta ^ { \mathrm { t e s t } } = 5 \cdot 1 0 ^ { - 5 }$

```latex
Algorithm 1 Stepsize modulation algorithms with $\hat { \eta } _ { \mathrm { s o } }$ and $\hat { \eta } _ { \mathrm { c s } }$
Initialize: $\theta _ { 0 }$ randomly, $v _ { 0 } = 0$
for epoch $= 1 , \ldots , T$ do
sample a fixed-size mini-batch S and use data $\{ ( x _ { \alpha } , y _ { \alpha } ) \} _ { \alpha \in S }$
η<sub>epoch</sub> ← schedulerValue(epoch)
$\begin{array} { r l r l } {  { \vartheta _ { t } } } & { { } } & {  \theta _ { t } - ( \mu v _ { t } + \lambda \theta _ { t } ) } \end{array}$
$z _ { \alpha }$ $\gets f ( x _ { \alpha } , \vartheta _ { t } )$
$q _ { \alpha }$ ← softmax $\left( z _ { \alpha } \right)$
$\mathcal { L } _ { t }$ ← crossEntropy $( z _ { S } , y _ { S } )$
θ<sup>test</sup> $ \vartheta _ { t } - \eta ^ { \mathrm { t e s t } } \nabla _ { \theta } \mathcal { L } _ { t }$
$\begin{array} { r l } { z _ { \alpha } ^ { \mathrm { t e s t } } } & { { }  f ( x _ { \alpha } , \overbar { { \theta } ^ { \mathrm { t e s t } } } ) } \end{array}$
$q _ { \alpha } ^ { \mathrm { t e s t } }$ ← softmax $\left( z _ { \alpha } ^ { \mathrm { t e s t } } \right)$
$\hat { \eta }$ ← eq (3) or (4) using $z _ { S } , z _ { S } ^ { \mathrm { t e s t } } , q _ { S } , q _ { S } ^ { \mathrm { t e s t } }$
$\eta _ { t + 1 }$ ← modulation(ˆη, η<sub>epoch</sub>)
$\begin{array} { r l } { \theta _ { t + 1 } } & { { }  \theta ^ { \mathrm { t e s t } } - ( \eta _ { t + 1 } - \eta ^ { \mathrm { t e s t } } ) \nabla _ { \theta } \mathcal { L } _ { t } } \end{array}$
$\begin{array} { r l } { v _ { t + 1 } } & { { }  \mu v _ { t } + \lambda \theta _ { t } + \eta _ { t + 1 } \nabla _ { \theta } \mathcal { L } _ { t } } \end{array}$
return $\theta _ { T + 1 }$
```

## The closest softmax stepsize (cs-dq)

A linear approximation of the softmax values in $\eta$ is used to select a value that minimizes the average Euclidean distance to the one-hot encodings $y _ { \alpha }$ . Thus we minimize the sum of their 2-norms

$$
\frac { 1 } { 2 } \sum _ { \alpha } \bigg \| y _ { \alpha } - q _ { \alpha } - \eta \frac { \mathrm { d } q _ { \alpha } } { \mathrm { d } \eta } \bigg \| ^ { 2 } .
$$

Recall that sums over α range over the t-th minibatch $S _ { t }$ . Solving the derivative to η gives

$$
\eta _ { \mathrm { c s } } ^ { * } = \frac { \sum _ { \alpha } \left( y _ { \alpha } - q _ { \alpha } \right) \cdot \frac { \mathrm { d } q _ { \alpha } } { \mathrm { d } \eta } } { \sum _ { \alpha } \left. \frac { \mathrm { d } q _ { \alpha } } { \mathrm { d } \eta } \right. ^ { 2 } } .
$$

Finally, we use the same probe for $\eta ^ { \mathrm { t e s t } }$ as before to estimate $\frac { \mathrm { d } q _ { \alpha } } { \mathrm { d } \eta }$

$$
\hat { \eta } _ { \mathrm { s o } } = \eta ^ { \mathrm { t e s t } } \frac { \sum _ { \alpha } ( y _ { \alpha } - q _ { \alpha } ) \cdot ( q _ { \alpha } ^ { \mathrm { t e s t } } - q _ { \alpha } ) } { \sum _ { \alpha } \left\| q ^ { \mathrm { t e s t } } - q _ { \alpha } \right\| ^ { 2 } } .\tag{4}
$$

## Both stepsizes are close

Despite being derived by diferent philosophies, the averages of these learning rates are close to each other. Indeed, we trained Resnet18 on Cifar10 with standard SGD (max learning rate 0.02, momentum $\mu = 0 . 9$ weight decay $\lambda = 0 . 0 0 5 )$ and cosine learning rate schedule. During this run we computed the average values of the Polyak stepsize and the 2 above stepsizes. In figure 2, we compare these stepsizes. We conclude that after halving the Polyak stepsize, these values are close to each other during the middle part of the training.

[To do: discuss fluctuations and compare values during minibatches.]

## 3 Practical refinements

## Outlier removals in stepsize estimates

If for some training example α of a minibatch, the softmax values do not seem to improve, then we drop this input in the estimation of $\hat { \eta } _ { \mathrm { s o } }$ and $\hat { \eta } _ { \mathrm { c s } }$ . More precisely, we drop α from minibatch $S _ { t }$ if cos $\phi _ { \alpha } < 1 / 4$ where

$$
\cos \phi _ { \alpha } = \frac { ( y _ { \alpha } - q _ { \alpha } ) \cdot ( q _ { \alpha } ^ { \mathrm { t e s t } } - q _ { \alpha } ) } { | | y _ { \alpha } - q _ { \alpha } | | | q _ { \alpha } ^ { \mathrm { t e s t } } - q _ { \alpha } | | } .
$$

![](images/33eaadcc79d78b040f46eac35a9ede68d4b08bfeaccd96d8a8abe31c551b954c.jpg)

Figure 2: comparison between stepsizes: so-dz (red), cs-dq (green), Polyak (orange), cosine decay (blue).  
![](images/27780613a36d268cb542851ab86346486921ed84c1c74de333fe8986256c367d.jpg)  
Figure 3: Fraction of removed outliers in the stepsize calculation

The efect is small, because only few points satisfy the condition $\cos ( \phi _ { \alpha } ) < 1 / 4$ , as shown in figure 3. But it seems to help to stabilize the estimates [can we say something more convincing or precise]?

## Subsampling and computational overhead

The estimation of the stepsize requires an extra forward pass. This increases the runtime typically by 33%, because the time for a backward pass is usually twice the time of a forward pass.

In our 2nd series of experiments, we took 20 random inputs from the remaining minibatch $S _ { t } ,$ (which had 128 elements, before the above pruning), and estimated the learning rate from them. This reduced the overhead from 33% to about 21% (or better on GPU’s with few cores and fast memory).

## Modulate an existing learning rate schedule

The efect of changing the learning rate for a given step also has an efect on future steps, because of momentum. Several papers report that this is a challenge to design automated learning rate schemes, for example in [29, footnote 1 on $\mathrm { p } 7 ]$ . Moreover, the several theoretical studies suggest that during initial learning it is beneficial to keep the stepsize larger, in order to avoid parameters $\theta _ { t }$ approaching sharp local minima. For these reasons, we focussed on perturbing existing schedules, rather than reconstructing them from scratch.

• Let $\eta _ { \mathrm { r e f } , e }$ be a reference learning schedule evaluated in epoch e. In experiments we will use cosine decay.

• Let $\eta _ { t }$ be the stepsize from 1 of the above methods, thus $\hat { \eta } _ { \mathrm { s o } }$ or $\hat { \eta } _ { \mathrm { c s } }$ at batch t.

• Let $\bar { \eta } _ { t }$ be the average of $\eta _ { t }$ over last $N = 5 0$ batches.

• Let $\beta = 0 . 0 5$ be fixed. [Todo: it turns out that we have only lookahead during experiments of our paper.]

Rather than clipping, a smoothly perturbed version of $\eta _ { \mathrm { r e f } , e }$ is obtained by multiplyin it with a the following factor in $[ 1 - \beta \pi / 2 , 1 + \beta \pi / 2 ]$ ，

$$
\eta _ { \mathrm { f i n a l } , t } = \eta _ { \mathrm { r e f } , e } \Big ( 1 + \beta \arctan \frac { \eta _ { t } - \bar { \eta } _ { t } } { \beta \bar { \eta } _ { t } } \Big ) .
$$

The arctan function was used because the derivative in the origin equals 1.

## Lookahead

In the 1st series of experiments, the lookahead technique from [32] was applied. The idea is to stabilize an existing optimizer Opt (like SGD, AdamW, etc) by letting it first make k updates and then make a single update of the aggregated “gradients”. The optimizer has 2 parameters k and α. We use the default values k = 5 and $\alpha = 0 . 8$

More precisely, the optimizer Opt is viewed as a fast optimizer and is used to create a slow optimizer as follows. Let $\theta _ { t }$ denote the current parameters of the slow optimizer at the start of its slow step. The fast optimizer is initialized with $\phi _ { t , 0 } = \theta _ { t }$ and computes k updates $\phi _ { t , 1 } , \ldots , \phi _ { t , k }$ . Finally, the slow optimizers sets

$$
\theta _ { t + 1 } = ( 1 - \alpha ) \theta _ { t } + \alpha \phi _ { t , k } .
$$

## 4 Experiments

We use the well known Resnet18 and Resnet34 models and trained them from scratch on Cifar10 and Cifar100. Data augmentation and the modification of the last hidden layer was applied as is now standard, see for example [10].

The 2 baseline experiments are as follows.

• Standard SGD with cosine learning rate decay (without warmup). The max-learning rate is 0.02, the momentum parameter is $\mu = 0 . 9$ , the weight decay is 5e − 3 and the minibatch size is 128.

• Lookahead with the above SGD as fast optimizer. The standard values k = 5 and $\alpha = 0 . 8$ are used.

We have a default experiment which uses Resnet18 on Cifar10 with 50 epochs. All pictures where generated with this default experiment. The max-learning rate and weight decay parameters reported above where tuned for this experiment. Our 2 stepsizes were designed using this same experiment. However, below we report these baselines for diferent settings with 100 and 200 epochs, and we also report the experiments with Resnet34 and Cifar100. In all experiments the hyper parameters remain the same and are fixed.

![](images/de59cf512b54baac6a5032f4468f94f2a6e61c8ec2f48fe01b577d0a87a0b085.jpg)

![](images/9df47ed83e7ff8b49a4fb8ccffe8b0fb641e389abd982e840aaf533304e960b4.jpg)  
Figure 4: Loss and test error per epoch. Experiment with CIFAR100 and ResNet18, 50 epochs, lookahead.

<table><tr><td></td><td></td><td colspan="2">Cifar10</td><td colspan="2">Cifar100</td></tr><tr><td>Net</td><td>optimiser</td><td>100 epochs</td><td> $\overline { { 2 0 0 \ \mathrm { e p o c h s } } }$ </td><td>100 epochs</td><td>200 epochs</td></tr><tr><td rowspan="4">Resnet18</td><td>SGD</td><td> $9 5 . 2 5 6 \pm 0 . 1 5 2$ </td><td> $9 5 . 4 7 6 \pm 0 . 0 9 9$ </td><td> $7 8 . 0 6 3 \pm 0 . 2 3 9$ </td><td> $\overline { { 7 8 . 6 4 9 { \scriptstyle \pm 0 . 2 5 9 } } }$ </td></tr><tr><td>la</td><td> $9 5 . 2 8 7 \pm 0 . 1 2$ </td><td> $9 5 . 6 0 3 \pm 0 . 1 4 9$ </td><td> $7 8 . 1 2 3 \pm 0 . 2 9 6$ </td><td> $7 8 . 5 4 3 \pm 0 . 2 5 4$ </td></tr><tr><td>la, so-dz</td><td> $\mathbf { 9 5 . 3 9 9 \pm 0 . 1 2 7 }$ </td><td> $9 5 . 6 0 8 \pm 0 . 1 0 2$ </td><td> $\mathbf { 7 8 . 1 5 7 \pm 0 . 2 2 9 }$ </td><td> $7 8 . 5 9 3 \pm 0 . 2 4 1$ </td></tr><tr><td>la, cs-dq</td><td> $9 5 . 3 7 9 \pm 0 . 0 9 7$ </td><td> $\mathbf { 9 5 . 6 7 3 { \scriptstyle \pm 0 . 1 6 7 } }$ </td><td> $7 8 . 1 1 4 \pm 0 . 2 7 7$ </td><td> $7 8 . 5 5 4 \pm 0 . 2 0 4$ </td></tr><tr><td rowspan="4">Resnet34</td><td>SGD</td><td> $\overline { { 9 5 . 3 4 2 \pm 0 . 1 0 6 } }$ </td><td> $9 5 . 5 3 6 \pm 0 . 1 2 7$ </td><td> $\overline { { 7 8 . 8 9 1 \pm 0 . 2 4 9 } }$ </td><td> $\overline { { 7 9 . 3 5 7 \pm 0 . 2 5 9 } }$ </td></tr><tr><td>la</td><td> $9 5 . 4 7 1 \pm 0 . 1 3 4$ </td><td> $9 5 . 6 2 9 \pm 0 . 1 5 3$ </td><td> $\mathbf { 7 8 . 9 6 8 { \scriptstyle \pm 0 . 2 6 5 } }$ </td><td> $\mathbf { 7 9 . 5 3 7 \pm 0 . 1 9 7 }$ </td></tr><tr><td>la, so-dz</td><td> $9 5 . 5 5 0 \pm 0 . 1 0 9$ </td><td> $\mathbf { 9 5 . 7 1 1 \bot 0 . 0 9 7 }$ </td><td> $7 8 . 9 1 9 \pm 0 . 2 2 5$ </td><td> $7 9 . 4 6 1 \pm 0 . 1 7 4$ </td></tr><tr><td>la, cs-dq</td><td> $\mathbf { 9 5 . 5 7 4 { \scriptstyle \pm 0 . 1 0 2 } }$ </td><td> $9 5 . 6 6 3 \pm 0 . 0 8 7$ </td><td> $7 8 . 9 5 9 \pm 0 . 2 0 4$ </td><td> $7 9 . 5 1 9 \pm 0 . 2 5 6$ </td></tr></table>

Table 1: Test accuracy, averaged over 16 repetitions
<table><tr><td></td><td></td><td colspan="2">Cifar10</td><td colspan="2">Cifar100</td></tr><tr><td>Net</td><td>Optimiser</td><td>100 epochs</td><td>200 epochs</td><td>100 epochs</td><td>200 epochs</td></tr><tr><td rowspan="4">Resnet18</td><td>SGD</td><td> $9 5 . 2 6 6 \pm 0 . 1 6 4$ </td><td> $\overline { { 9 5 . 5 5 2 \pm 0 . 1 3 0 } }$ </td><td> $7 7 . 8 9 6 \pm 0 . 2 6 2$ </td><td> $\overline { { 7 8 . 6 1 0 \pm 0 . 4 0 4 } }$ </td></tr><tr><td>lookahead</td><td> $9 5 . 2 9 4 \pm 0 . 1 4 8$ </td><td> $9 5 . 5 2 0 \pm 0 . 1 5 4$ </td><td> $\mathbf { 7 8 . 2 2 5 { \pm } 0 . 2 6 3 }$ </td><td> $7 8 . 7 2 3 { \scriptstyle \pm 0 . 1 7 9 }$ </td></tr><tr><td>la, so-dz</td><td> $9 5 . 3 6 3 \pm 0 . 0 7 1$ </td><td> $9 5 . 5 5 8 \pm 0 . 1 4 7$ </td><td> $7 8 . 1 7 6 \pm 0 . 1 5 3$ </td><td> $7 8 . 5 9 5 \pm 0 . 1 8 0$ </td></tr><tr><td>SGD, cs-dq</td><td> $\mathbf { 9 5 . 4 2 5 { \pm } } 0 . 1 3 4$ </td><td> $\mathbf { 9 5 . 6 5 3 { \scriptstyle \pm 0 . 1 3 5 } }$ </td><td> $7 8 . 0 5 1 \pm 0 . 2 1 2$ </td><td> $7 8 . 3 5 3 \pm 0 . 3 0 0$ </td></tr><tr><td rowspan="4">Resnet34</td><td>SGD</td><td> $\overline { { 9 5 . 4 3 6 \pm 0 . 1 3 2 } }$ </td><td> $9 5 . 4 2 8 \pm 0 . 1 9 8$ </td><td> $\overline { { 7 8 . 8 3 3 \pm 0 . 2 4 9 } }$ </td><td> $\overline { { 7 9 . 3 4 7 \pm 0 . 1 9 0 } }$ </td></tr><tr><td>lookahead</td><td> $9 5 . 4 3 8 \pm 0 . 1 6 5$ </td><td> $9 5 . 6 6 7 \pm 0 . 1 3 9$ </td><td> $7 8 . 8 6 3 \pm 0 . 2 5 7$ </td><td> $\mathbf { 7 9 . 5 9 2 } { \scriptstyle \pm 0 . 2 5 7 }$ </td></tr><tr><td>SGD, so-dz</td><td> $\mathbf { 9 5 . 5 7 4 { \scriptstyle \pm 0 . 1 5 3 } }$ </td><td> $9 5 . 6 2 5 \pm 0 . 1 5 1$ </td><td> $\mathbf { 7 8 . 9 4 1 } { \scriptstyle \pm 0 . 2 3 6 }$ </td><td> $7 9 . 4 1 5 \pm 0 . 1 4 1$ </td></tr><tr><td>SGD, cs-dq</td><td> $9 5 . 4 3 3 \pm 0 . 1 0 7$ </td><td> $\mathbf { 9 5 . 7 0 3 { \scriptstyle \pm 0 . 1 5 8 } }$ </td><td> $7 8 . 8 7 6 \pm 0 . 3 0 2$ </td><td> $7 9 . 4 2 7 \pm 0 . 2 8 9$ </td></tr></table>

Table 2: Test accuracy averaged over 6 repetitions, subsampling

In all experiments, the hyper parameters of the stepsize $\hat { \eta } _ { \mathrm { s o } }$ and $\hat { \eta } _ { \mathrm { c s } }$ are $\eta ^ { \mathrm { t e s t } } = 5 \cdot 1 0 ^ { - 5 }$ , the perturbation $\beta = 0 . 0 5$ and the average stepsize is computed from 50 minibatches. Experiments were performed on V100 GPU’s on the cluster of the Higher School of Economics in Moscow [16]. The code is available here https: //github.com/Brbauwens/adaptive\_stepsize\_modular.

For both the so-dz and cs-dq stepsizes, the initial test error decreases more rapidly and the curve is more monotone. However, recall that this method requires 33% more compute (and 21% more compute with subsampling of the test probe). Especially, this is true when combined with lookahead, see figure 4 for the default experiment. However, near the end, improvement becomes slower.

Our methods often improve error on the test set. In a 1st series of experiments, lookahead was applied and no subsampling was done. All 4 methods were applied to the same minibatches. The best test set accuracies are given in tables 1. For the Cifar10 training set, both the so-dz and cs-dq stepsizes exceed both baselines. Unfortunately, for Cifar100 the results look random.

In a 2nd series of experiments, subsampling was applied and lookahead was switched of. The runs of the baselines were repeated with diferent random seeds. The results in table 2 suggest that this does not decrease performance. Again for Cifar10, the best results is either for so-dz or cs-dq, but in the case of Resnet34, the gains look random. Also, for Cifar100, the methods do not provide reliable improvements.

## 5 Conclusion

Two methods were proposed to automatically generate a stepsize for each minibatch. In $\hat { \eta } _ { \mathrm { s o } }$ stepsize optimizes a 2nd order approximation of the minibatch loss, and the $\hat { \eta } _ { \mathrm { c s } }$ optimizes the softmax values. We argued theoretical and experimentally that both methods give closely related stepsizes.

These stepsizes were used to improve the cosine decay schedule. Experiments show that they speed up convergence beyond the 20%-33% especially in the middle of the training. Moreover, the decrease is more monotone. They give essentially the same accuracy, and in a few experiments, their best accuracy is improved compared to standard SGD and its combination with lookahead.

## References

[1] Dario Balboni and Davide Bacciu. ADLER – an eficient Hessian-based strategy for adaptive learning rate. In Proceedings of the 32nd European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), 2024.

[2] Jonathan Barzilai and Jonathan M Borwein. Two-point step size gradient methods. IMA journal of numerical analysis, 8(1):141–148, 1988.

[3] Leonard Berrada, Andrew Zisserman, and M Pawan Kumar. Training neural networks for and by interpolation. In International conference on machine learning, pages 799–809. PMLR, 2020.

[4] Christopher M Bishop and Hugh Bishop. Deep learning: Foundations and concepts, volume 1. Springer, Cham, Switzerland, 2024.

[5] Zhiqi Bu and Shiyun Xu. Gradient descent with generalized Newton’s method. In International Conference on Learning Representations, 2025.

[6] Camille Castera, J´erˆome Bolte, C´edric F´evotte, and Edouard Pauwels. Second-order step-size tuning of sgd for non-convex optimization. Neural Processing Letters, 54(3):1727–1752, 2022.

[7] Ashok Cutkosky, Aaron Defazio, and Harsh Mehta. Mechanic: A learning rate tuner. In Advances in Neural Information Processing Systems, volume 36, pages 47828–47848, 2023.

[8] Minghan Fu and Fang-Xiang Wu. Qlabgrad: A hyperparameter-free and convergence-guaranteed scheme for deep learning. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 12072–12081, 2024.

[9] Robert Gower, Othmane Sebbouh, and Nicolas Loizou. Sgd for structured nonconvex functions: Learning rates, minibatching and interpolation. In International Conference on Artificial Intelligence and Statistics, pages 1315–1323. PMLR, 2021.

[10] Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, and Mu Li. Bag of tricks for image classification with convolutional neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 558–567, 2019.

[11] Nikhil Iyer, V. Thejas, Nipun Kwatra, Ramachandran Ramjee, and Muthian Sivathanu. LRTuner: A learning rate tuner for deep neural networks. In 8th ICML Workshop on Automated Machine Learning, 2021.

[12] Arthur Jacot, Gabriel Franck, and Cl´ement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems, 31, 2018.

[13] Neel Jain, Ping-yeh Chiang, Yuxin Wen, John Kirchenbauer, Hong-Min Chu, Gowthami Somepalli, Brian Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Aniruddha Saha, et al. Neftune: Noisy embeddings improve instruction finetuning. In International Conference on Learning Representations, volume 2024, pages 17566–17591, 2024.

[14] Daragh King. Using the polyak step size in training convolutional neural networks. In The Second Tiny Papers Track at ICLR, 2024.

[15] Bobby Kleinberg, Yuanzhi Li, and Yang Yuan. An alternative view: When does sgd escape local minima? In International conference on machine learning, pages 2698–2707. PMLR, 2018.

[16] P. Kostenetskiy, V. Kozyrev, R. Chulkevich, M. Salikova, and R. Mishenin. High-performance computing at hse university. In Parallel Computational Technologies, volume 2891 of Communications in Computer and Information Science, Cham, 2026. Springer.

[17] Chaoyue Liu, Libin Zhu, and Mikhail Belkin. Loss landscapes and optimization in over-parameterized non-linear systems and neural networks. Applied and Computational Harmonic Analysis, 59:85–116, 2022.

[18] Hengyu Liu, Qiang Fu, Lun Du, Tiancheng Zhang, Ge Yu, Shi Han, and Dongmei Zhang. Learning rate perturbation: A generic plugin of learning rate schedule towards flatter local minima. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, CIKM ’22, pages 4234–4238. Association for Computing Machinery, 2022.

[19] Nicolas Loizou, Sharan Vaswani, Issam Hadj Laradji, and Simon Lacoste-Julien. Stochastic polyak step size for sgd: An adaptive learning rate for fast convergence. In International Conference on Artificial Intelligence and Statistics, pages 1306–1314. PMLR, 2021.

[20] Dimitris Oikonomou and Nicolas Loizou. Stochastic polyak step-sizes and momentum: Convergence guarantees and practical performance. arXiv preprint arXiv:2406.04142, 2024.

[21] Francesco Orabona and Ryan D’Orazio. New perspectives on the polyak stepsize: Surrogate functions and negative results. arXiv preprint arXiv:2505.20219, 2025.

[22] Francesco Orabona and Ryan D’Orazio. New perspectives on the polyak stepsize: Surrogate functions and negative results. Advances in Neural Information Processing Systems, 38:155395–155422, 2026.

[23] Antonio Orvieto and Lin Xiao. An adaptive stochastic gradient method with non-negative Gauss-Newton stepsizes. arXiv preprint arXiv:2407.04358, 2024.

[24] Boris Teodorovich Polyak. Minimization of unsmooth functionals. USSR Computational Mathematics and Mathematical Physics, 9(3):14–29, 1969.

[25] Daniel A Roberts, Sho Yaida, and Boris Hanin. The principles of deep learning theory, volume 46. Cambridge University Press Cambridge, MA, USA, 2022.

[26] Fabian Schaipp, Ruben Ohana, Michael Eickenberg, Aaron Defazio, and Robert M. Gower. MoMo: Momentum models for adaptive learning rates. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 43542–43570. PMLR, 2024.

[27] Nitish Srivastava, Geofrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929–1958, 2014.

[28] Conghui Tan, Shiqian Ma, Yu-Hong Dai, and Yuqiu Qian. Barzilai–borwein step size for stochastic gradient descent. In Advances in Neural Information Processing Systems (NeurIPS), 2016. Also available as arXiv:1605.04131.

[29] Sharan Vaswani, Aaron Mishkin, Issam Laradji, Mark Schmidt, Gauthier Gidel, and Simon Lacoste-Julien. Painless stochastic gradient: Interpolation, line-search, and convergence rates. Advances in neural information processing systems, 32, 2019.

[30] Zhi-Jun Wang, He-Bei Gao, Xiang-Hong Wang, Shuai-Ye Zhao, Hong Li, and Xiao-Qin Zhang. Adaptive learning rate optimization algorithms with dynamic bound based on barzilai-borwein method. Information Sciences, 634:42–54, 2023.

[31] Zeke Xie, Issei Sato, and Masashi Sugiyama. A difusion theory for deep learning dynamics: Stochastic gradient descent exponentially favors flat minima. arXiv preprint arXiv:2002.03495, 2020.

[32] Michael Zhang, James Lucas, Jimmy Ba, and Geofrey E Hinton. Lookahead optimizer: k steps forward, 1 step back. Advances in neural information processing systems, 32, 2019.

[33] Yi Zhou, Junjie Yang, Huishuai Zhang, Yingbin Liang, and Vahid Tarokh. Sgd converges to global minimum in deep learning via star-convex path. arXiv preprint arXiv:1901.00451, 2019.

[34] Yingqiu Zhu, Danyang Huang, Yuan Gao, Rui Wu, Yu Chen, Bo Zhang, and Hansheng Wang. Automatic, dynamic, and nearly optimal learning rate specification via local quadratic approximation. Neural Networks, 141:11–29, 2021.

## A Derivation of so-dz stepsize

Fix the minibatch is fixed and we drop subscripts t. Below, the quantities q and z are considered for the updated parameter $\theta _ { t + 1 }$ , which depends on the updated learning rate through (2), thus $q = q ( \theta _ { t + 1 } ( \eta ) )$ and $z = z ( \theta _ { t + 1 } ( \eta ) )$ .

$$
\begin{array} { r } { q _ { \alpha i } = \cfrac { \exp { z _ { \alpha i } } } { \sum _ { j } \exp { z _ { \alpha j } } } } \\ { L ( \tilde { \eta } ) = \operatorname { L } ( \theta - \tilde { \eta } w ) . } \end{array}
$$

We need to evaluate $L ^ { \prime } ( 0 )$ and $L ^ { \prime \prime } ( 0 )$ and write it as a function of $q$ and ${ \frac { \mathrm { d } { \boldsymbol { z } } } { \mathrm { d } { \boldsymbol { \eta } } } } .$ . Recall that $y _ { \alpha j }$ is the one hot encoding of the label of training input $x _ { \alpha }$ . One calculates that

$$
\begin{array} { c } { { \displaystyle \frac { \partial q _ { i \alpha } } { \partial z _ { j \alpha } } = ( \delta _ { i j } - q _ { \alpha i } ) q _ { \alpha j } } } \\ { { \displaystyle \frac { \partial L } { \partial z _ { i \alpha } } = q _ { \alpha i } - y _ { \alpha i } } } \\ { { \displaystyle \frac { \partial ^ { 2 } L } { \partial z _ { i \alpha } \partial z _ { j \alpha } } = ( \delta _ { i j } - q _ { \alpha i } ) q _ { \alpha j } . } } \end{array}\tag{5}
$$

We have

$$
L ^ { \prime } ( 0 ) = \sum _ { \alpha , i } { \frac { \partial L } { \partial z _ { \alpha i } } } { \frac { \mathrm { d } z _ { \alpha i } } { \mathrm { d } \eta } } = \sum _ { \alpha , i } ( q _ { \alpha i } - y _ { \alpha i } ) { \frac { \mathrm { d } z _ { \alpha i } } { \mathrm { d } \eta } } .
$$

$$
\begin{array} { r } { L ^ { \prime \prime } ( 0 ) = \displaystyle \sum _ { \alpha , j } \frac { \partial L ^ { \prime } } { \partial z _ { \alpha j } } \frac { \mathrm { d } z _ { \alpha j } } { \mathrm { d } \eta } = \displaystyle \sum _ { \alpha , i , j } \frac { \partial ^ { 2 } L } { \partial z _ { j \alpha } \partial z _ { i \alpha } } \frac { \mathrm { d } z _ { \alpha i } } { \mathrm { d } \eta } \frac { \mathrm { d } z _ { \alpha j } } { \mathrm { d } \eta } } \\ { = \displaystyle \sum _ { \alpha , i , j } ( \delta _ { i j } - q _ { \alpha i } ) q _ { \alpha j } \frac { \mathrm { d } z _ { \alpha i } } { \mathrm { d } \eta } \frac { \mathrm { d } z _ { \alpha j } } { \mathrm { d } \eta } . } \end{array}
$$

To calculate $L ^ { \prime \prime } .$ , we neglect the 2nd derivatives of the output preactivation z. Thus,