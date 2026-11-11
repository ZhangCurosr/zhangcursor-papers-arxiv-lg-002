# TorchCruves: Diferentiable Parametric Curves

Alex Shtof

alex.shtf@gmail.com

## Abstract

We present TorchCurves, a PyTorch library that treats vector-valued parametric curves as learnable primitives. Its central operation maps batches of scalar inputs to points on curves represented in B-spline or Legendre polynomial bases, with diferentiation through both inputs and coeficients. Curve coeficients can be optimized jointly with the surrounding model or produced by another network. The library provides trainable modules, reusable basis modules, and a functional interface, supporting applications ranging from continuous feature embeddings to Kolmogorov–Arnold networks and functional regression. Its implementation combines classical evaluation algorithms with vectorized PyTorch operations, emphasizing readability and maintainability without custom hardware kernels. Timing experiments compare curve evaluation against NumPy and SciPy and KAN training against EficientKAN across a range of problem sizes. The results illustrate where this generality remains computationally practical and where overhead is substantial. Library is available on GitHub at https://github.com/alexshtf/torchcurves.

Keywords: spline, kolmogorov-arnold-network, kan, machine-learning

## 1 Introduction

![](images/ee4d34c1066e0fe967d15c1403cb18a3a9d3fd4eaea90aac32375425d3ab003b.jpg)  
Figure 1: The basic operation behind TorchCurves. Input: a mini-batch of numerical features of size. Output: a mini-batch of points on multi-dimensional curves.

The use of parametric curves spans a wide variety of scientific and engineing disciplines, both in machine learning and outside of it. Examples include spline regression, continuous features embeddings (Shtof et al., 2024), Kolmogorov-Anrnold networks (Liu et al., 2025), computer aided design (B´ezier, 1968) and others. Yet, there seems to be no software library providing a common, composable interface for curves as learnable primitives, but rather every re-implements its own high-level object directly. TorchCurves puts the learnable parametric curve as a primitive in its own right, and facilitates its use in a variety of model achitectures. Its genericity allows PyTorch (Ansel et al., 2024) users to build models that include learnable parametric curves as a component.

The essence of TorchCurves is one operation - transforming a mini-batch of input features to a mini-batch of points on curves in $\mathbb { R } ^ { d }$ , as depicted in Figure 1. At this stage we support B-Spline and Legendre polynomials. Our implementation is vectorized and there are no loops on the feature, batch and curve dimensions. But we do not have custom kernels for specialized hardware - we chose to implement everything in pure PyTorch for readability and maintainability reasons. This basic operation is diferentiable in both its parameters, the input features and the curve parameters, allowing both learning a curve and back-propagating to previous network layers.

The next sections are structures as follows: in Section 3 we present the main components of the library and present examples of applications: continuous feature embeddings, Kolmogorov-Arnold networks, and functional regression. In Section 4 we discuss some implementation and testing details.

## 2 Related work

Polynomial and spline representations are well established in numerical software. NumPy (Harris et al., 2020) and SciPy (Virtanen et al., 2020) provide routines for representing, evaluating, and diferentiating such functions. Diferentiable spline packages also exist within the PyTorch ecosystem: torchcubicspline (Kidger, n.d.) constructs natural cubic interpolants from observations, while torch-bspline (Melchers and Cochran, 2025) provides diferentiable B-spline evaluation.

In machine learning, mappings from scalar values to learned vectors appear in numerical feature embeddings (Gorishniy et al., 2022; Shtof et al., 2024). Learnable univariate functions also constitute the edges of Kolmogorov-Arnold networks (Liu et al., 2025), with specialized implementations such as EficientKAN (Cao, 2024). These applications illustrate several roles that parametric curves can play within a larger model.

TorchCurves takes the curve itself as the reusable abstraction: a parameterized mapping from a scalar input to a vector output, diferentiable with respect to both its input and its coeficients. It provides a common interface for B-spline and Legendre polynomial representations, whose coeficients can be learned directly or supplied by another model. Its contribution is to make these curves convenient to compose with other learning components, without requiring an interpolation problem or prescribing a particular network architecture.

## 3 API and applications

We first present the basic API, and then show use-case examples where it can serve.

## 3.1 The API

The library provides two basic components, BSplineCurve and LegendreCurves, which are one-dimensional curves embedded in a high-dimensional space. Each curve is represented as

$$
\pmb { c } ( \boldsymbol { x } ) = \sum _ { i = 1 } ^ { n } { \pmb { v } } _ { i } B _ { i } ( \boldsymbol { x } ) ,
$$

where $B _ { i }$ are the appropriate basis functions, B-Spline or Legendre, and $\pmb { v } _ { 1 } , \ldots , \pmb { v } _ { n }$ are its (possibly) learned coeficients. Each component can evaluate a bulk of curves $\pmb { c } _ { 1 } , \ldots , \pmb { c } _ { m }$ simultaneously, as shown in Figure 1. Below is an example:

```haskell
import torch
import torchcurves as tc
u = torch.rand(4, 3) # 4 samples with 3 features.
curve = tc.BSplineCurve(
num_curves=3, dim=2 # 3 curves in 2 dimensions
)
y = curve(u) # (4 samples, 3 curves, 2 dimensions)
print(u.shape, "->", y.shape) # torch.Size([4, 3]) -> torch.Size([4, 5, 2])
```

Of course, mathemaically inclined readers will say this example is incomplete. And justifiably so - both B-Spline and Legendre polynomial bases require configuration, such as the polynomial degree, the number of knots in case of B-Splines, and others. Moreover, splines and polynomial approximations work in a compact interval, which is nowhere to be seen in the above example.

Indeed, we provide the modules with defaults, but they are are fully configurable. Both B-Splines and Legendre polynomials are assumed to approximate inputs on the interval $\mathcal { U } = [ - 1 , 1 ]$ , and can be configured with a mapping for transforming their input to that interval. For example:

spl\_curve = tc.BSplineCurve(   
num\_curves=3, dim=2, # dimensions   
input\_map='real.arctan', # maps \`x\` to \`2 arctan(x) pi\`   
num\_knots=20, # 20 uniformly-spaced knots, with not-a-knot endpoints   
degree=5 # polynomial splines of degree 5   
)   
lgd\_curve = tc.LegendreCurve(   
num\_curves=3, dim=5 # dimensions   
input\_map='arctan', # same as above   
degree=50 # degree-50 Legendre polynomials   
)

Mapping functions include the above-mentioned arctangent function, which is also the default, but also $x \to x / \sqrt { 1 + x ^ { 2 } }$ called real.rational and is inspired by by its use in the diferential equations literature on unbounded domains (Guo and Wang, 2004; Boyd, 1987; Wang and Guo, 2004), and a simple clamping to U called real.clamp. The prefix real means these mappings are designed for arbitrary real numbers. The library also has nonneg.rational and nonneg.arctan for non-negative numbers. The users can also provide their own, in the form of a Python callable.

Trainable modules are useful, but they do not cover all use-cases the library aims to support. Thus, the library provides three API levels which we shall explain below: (a) trainable modules, as we just seen; (b) basis modules, which cache tensors to avoid their recomputation but have no trainable parameters; (c) a fully functional API, akin to torch.nn.functional.

The fully functional API, as its name suggests, consists of pure functions that have no state. Below is an example:

```python
import torchcurves.functional as tcf
bspl_func = tcf.bspline_curves(
inputs, # (B, C) - a batch of input points for each curve
coefs, # (C, P, D) coefficients of C B-Spline functions having P coefficients in D dimensions
knots, # B-Spline knots. unif. spaced in [-1, 1] with not-a-knot conditions, if unspecified.
degree=5 # the piecewise-polynomial degree. 3 if unspecified.
)
leg_func = tcf.legendre_curves(
inputs, # (B, C) - a batch of input points for each curve
coefs, # (C, P, D) coefficients of C legendre polynomials having P coefficinets in D dimensions
degree=50 # the polynomial degree. We always have P=degree-1.
```

These are useful if the coeficients are not learned directly, but come as an output of another module. Note, that in case of B-Spline functions, the connection between the number of knots and the number of coeficients P must be P = |knots| − degree + 1. If the knot vector is not specified, the number of uniformly-spaced knots is automatically determined from the degree and the number of coeficients P.

Finally, in between learnable modules and fully functional API we have the basis modules. A basis module is similar to the functional API, but stores immutable bufers, such as the knot vector, for convenience. That way, if a model is moved from CPU to an accelerator, such as a GPU, the bufer automatically moves with it by PyTorch. Below is an example:

```python
import torchcurves as tc
# quadratic splines with 50 B-Spline functions and a rational mapping to [-1, 1]
bspl_basis = tc.BSplineBasis(degree=2, knot_config=50, input_map="real.rational")
# evaluate a B-Spline function. Note - 50 coefficients for our 50 B-Spline functions.
bspl_func = bspl_basis(
input, # (B, C) - a batch of input points for each curve
coefs # (C, 50, D) - the coefficients of C B-Spline functions in D dimensions.
)
```

## 3.2 Applications

One of the first applications that comes to mind is producing embeddings for continuous features, such as height, weight, BMI, the time since the last visit of the user on our website, and others. Imagine we have a model whose inputs are embedding vectors, such as a Factorization Machine, or a Transformer. For discrete, or categorical features - we can just map features to embeddings using a standard embedding table. But for a continuous value, we need a mapping of the value to a vector in the embedding space - and a curve is such a mapping. Here is an example that uses Legendre polynomial basis for such curves:

```python
from torch import nn, cat
import torchcurves as tc
class Net(nn.Module):
"""Composes PyTorch `nn.Embedding` and Legendre curves with another model"""
def __init__(self, n_discrete, n_continuous, degree, dim):
super().__init__()
self.discrete = nn.Embedding(n_discrete, dim)
self.cont = tc.LegendreCurve(n_continuous, dim, degree)
self.inner_model = my_super_duper_transformer()
def forward(self, cat, cont):
# concatenate all embeddings into one tensor
embeddings = cat((self.discrete(cat), self.cont(cont)), dim=-2)
# invoke our desired model on the embeddings
return self.inner_model(embeddings)
```

Kolmogorov-Arnold networks are also an immediate application - each layer computes many univariate functions of the input, and sums-up their results. Recalling that the output of our curve layers has the shape (sample, curve, dimension), a KAN layer just applies our curve layer, followed by a sum over the curve dimension. For convenience, we include the Sum module in the library as well. Here is an example:

```python
from torch import nn
import torchcurves as tc
kan = nn.Sequential([
# layer 1
tc.BSplineCurve(input_dim, hidden_dim),
tc.Sum(dim=-2),
# layer 2
tc.BSplineCurve(hidden_dim, hidden_dim),
tc.Sum(dim=-2),
# layer 3
tc.BSplineCurve(hidden_dim, output_dim),
tc.Sum(dim=-2),
])
```

A less trivial application is that of functional regression - a model whose output is itself a function. In theory, we can think of any model f(x) as ”functional regression” - just partition ${ \pmb x } = ( { \pmb x } _ { 1 } , { \pmb x } _ { 2 } )$ and define the model as the function $f ( { \pmb x } _ { 1 } , \cdot )$ . The point is not in the model’s architecture, but as the study of the model, including its performance, as such a function. Since the authors come from the online advertising field, one prominent example is fitting the probability of winning an ad auction given auction features and a non-negative bid. See Pan et al. (2020); Zhou et al. (2021).

Note that this is a CDF - it must be nondecreasing. A natural choice is using the property of B-Spline functions - if the coeficient vector is non-decreasing, so is the function. So below is such an example - we encode the auction into a non-decreasing vector of coeficients, which in turn serves as the coeficient vector of a B-Spline function of the bid. So below is a model for learning such a probability:

```python
from torch import nn
```

```python
import torch.nn.functional as F
import torchcurves as tc
class AuctionWinModel(nn.Module):
def __init__(self):
super().__init__()
self.encoder = MySuperBestAuctionEncoder()
self.spline = tc.BSplineBasis(input_map="nonneg.arctan")
def forward(self, auction_features, bids):
# encode auction to a latent vector
raw = self.encoder(auction_features)
# convert the latent vector to non-decreasing coefficients:
# use first component as is, add cumulative sum of softplus(rest)
increments = F.softplus(raw[:, 1:])
coefficients = raw[:, :1] + F.pad(increments.cumsum(-1), (1, 0))
# non-decreasing spline function of bids to produce P(win) logits.
return self.spline(bids[None], coefficients[..., None]).flatten()
```

## 4 Algorithms

Beyond being vectorized, the implementations we use should also be fairly eficient to be useful for machine learning research. Our eficiency goals are modest - we aim for the best we can achieve using pure PyTorch code, without custom kernels. Thus, we implement vectorized versions of well-known algorithms, both for B-Spline and Legendre polynomial evaluation.

The main feature of the B-Spline basis is its sparsity - at any given point, only degree+2 basis functions are non-zero. An eficient algorithm for exploiting the sparsity is the wellknown Cox-deBoor method (COX, 1972; De Boor, 1972). The algorithm provides both an eficient sparsity-exploiting way to compute the value and the derivative. These, in turn, are used in a custom PyTorch autograd function to facilitate eficient back-propagation through the inputs and the coeficients. The implementation is done in a vectorized manner without iterating on any of the input dimension, but does require iterating on the degree. Thus, a spline of degree five does require six Python iterations to compute.

For Legendre polynomials we have the well-known Clenshaw recursion (Clenshaw, 1955). Here, as well, we do not iterate on any input tensor dimensions, but we do iterate on the degree. Since typically good approximation is achieved with high-degree polynomials, higher than B-Splines, this may be less eficient in some situations. For example, a polynomial of degree 50 will require 51 Python for-loop iterations.

In our implementation we do take special care to avoid memory allocation of tensors. We do not claim optimality, but merely a best-efort. Again, our eficiency claim is modest - to be fast enough to be useful.

The library is thoroughly tested, and in particular, the results are compared to SciPy (Virtanen et al., 2020) splines and NumPy (Harris et al., 2020) Legendre polynomials. Derivatives are validated using internal gradient-checking mechanisms of PyTorch, that compare numerically approximated derivatives to those of back-propagation. Comparisons were done for both 32 and 64 bit floating point numbers.

## 5 Experiments

Our objective here is giving the reader a qualitative understanding of what we mean by ”our eficiency claim is modest”. To that end, we perform two comparisons: (a) speed of inference-only computation of Legendre polynomials and Splines on CPU, compared to NumPy and SciPy; and (b) training speed of a KAN, compared to EficientKAN (Cao, 2024), both on CPU and GPU, each implemented in its idiomatic form. Our aim is not to outperform specialized implementations, but to show that the flexibility of treating curves as general-purpose trainable primitives remains computationally practical, even when composing them into higher-level architectures such as KANs.

All measurements were done on a Lenovo P14s Gen-7 laptop, with an Intel Core Ultra 7 356H CPU, and a NVIDIA RTX PRO 500 Blackwell GPU. The experiments were running on Ubuntu 26.04.01, with Python 3.12, NumPy 2.5.4, SciPy 1.18.1, and PyTorch 2.15 (CUDA 13.2). CPU afinity is limited to the first cores 1-4, which are the performance cores of that CPU, to both allow PyTorch exploit parallelism and obtain consistent speed measurements.

We note that both NumPy and SciPy do not provide an autograd mechanism, thus they are by definition “inference-only” libraries. Therefore, our comparison in (a) are inferenceonly in nature - we perform no backpropagation, but only compute the value of the spline / Legendre polynomial functions using each library, and compare the number of milliseconds per call for various coeficient counts, curve dimensions, curve counts, and batch sizes. The results are in Figures 2 and 3. We can see that torchcurves is slower for small number of curves or small batch sizes, and in the ballpark or faster for larger scales. It is our conjecture that at smaller scales Python overhead wins, whereas for larger scales vectorized PyTorch code lets it exploit multiple cores, while NumPy and SciPy remain serial. In all cases, we perform

We now turn to KAN training speed comparison. We first note that there could be a different variant - implement a layer that is identical, mathematically, to that of EficientKAN but implemented as a custom layer using torchcurves functional API. We, on purpose, compare the recommended way to build a KAN in both libraries. KANs have two hidden layers, and have the architecture [16, W, W, 1], meaning, 16 input features, two hidden layers of width W, and a scalar output. We vary the width W, the number of coeficients K, and the training batch size B. Perform 50 warmup steps without measuring time, and then repeat five times training of 100 steps that include clearing gradiens, forward sweep, backward sweep and optimizer step. We divide total train time by 100 to obtain the time per step in each of the five attempts, and then take the median among the attempts to obtain the median time. Results are in Figure 4. We can see that torchcurves is generally in the ballpark of EficientKAN. Faster for larger batch sizes and higher dimensions, while slower otherwise.

## References

Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, Brian Hirsh, Sherlock Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos, Mario Lezcano, Yanbo Liang, Jason Liang, Yinghai Lu, CK Luk, Bert Maher, Yunjie Pan, Christian Puhrsch, Matthias Reso, Mark Saroufim, Marcos Yukio Siraichi, Helen Suk, Michael Suo, Phil Tillet, Eikan Wang, Xiaodong Wang, William Wen, Shunting Zhang, Xu Zhao, Keren Zhou, Richard Zou, Ajit Mathews, Gregory Chanan, Peng Wu, and Soumith Chintala. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. In 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS ’24). ACM, April 2024. doi: 10.1145/3620665.3640366. URL https://docs.pytorch.org/assets/pytorch2-2.pdf.

![](images/9ad413a80665d214e50c9c5c3a7efb3f74b897735891ea4c18edf631bb3a60ba.jpg)  
Figure 2: CPU evaluation of independent vector-valued B-Spline curves. Panel rows show curve count C; panel columns show output dimension D. Within each panel, rows show batch size B and columns show coeficients per scalar output K. Each cell is median TorchCurves time divided by median SciPy time. Values below 1 mean TorchCurves is faster. Medians use five timing measurements.

![](images/99cbaa75fdaad8d0c27a032124cfb1c3c4b3f325d8686074edbfdb99fd651b8c.jpg)  
Figure 3: CPU evaluation of independent vector-valued Legendre-polynomial curves. Panel rows show curve count C; panel columns show output dimension D. Within each panel, rows show batch size B and columns show coeficients per scalar output K. Each cell is median TorchCurves time divided by median reference time. Values below 1 mean TorchCurves is faster. Medians use five timing measurements.

KAN training steps · TorchCurves vs EficientKAN  
![](images/a990c6cb7d1e382277f80cfdb35bc39b1508fe69305ba6d547da7ec9e1cd36e5.jpg)  
Figure 4: Training-step time ratios for idiomatic [32, W, W, 1] cubic KANs. Panel rows show CPU and GPU; panel columns show coeficient count K. Within each panel, rows show hidden width W and columns show batch size B. Each cell is median TorchCurves time divided by median EficientKAN time. EficientKAN uses its KAN model with SiLU branches, trainable spline scaling, extended knots, and no added clamp. Each library initializes its own parameters. Architectures and parameter counts difer. Both use the same generated data, MSE, and Adam with fixed grids. Float32 eager execution.

Pierre E B´ezier. How renault uses numerical control for car body design and tooling. In 1968 Automotive Engineering Congress and Exposition 90221, 1968.

John P. Boyd. Spectral methods using rational basis functions on an infinite interval. Journal of Computational Physics, 69(1):112–142, 1987. doi: 10.1016/0021-9991(87)90158-6.

Huanqi Cao. An eficient implementation of Kolmogorov-Arnold network. https://github.com/Blealtan/efficient-kan, 2024. Software. Commit 7b6ce1c87f18c8bc90c208f6b494042344216b11.

Charles W Clenshaw. A note on the summation of chebyshev series. Mathematics of computation, 9(51):118–120, 1955.

M. G. COX. The numerical evaluation of b-splines. IMA Journal of Applied Mathematics, 10(2):134–149, 10 1972. ISSN 0272-4960. doi: 10.1093/imamat/10.2.134. URL https: //doi.org/10.1093/imamat/10.2.134.

Carl De Boor. On calculating with b-splines. Journal of Approximation theory, 6(1):50–62, 1972.

Yury Gorishniy, Ivan Rubachev, and Artem Babenko. On embeddings for numerical features in tabular deep learning. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://papers.neurips.cc/paper\_files/paper/2022/hash/ 9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html.

Benyu Guo and Zhongqing Wang. Legendre rational approximation on the whole line. Science in China Series A: Mathematics, 47(Suppl.):155–164, 2004. doi: 10.1360/04za0014.

Charles R. Harris, K. Jarrod Millman, St´efan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fern´andez del R´ıo, Mark Wiebe, Pearu Peterson, Pierre G´erard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. Array programming with NumPy. Nature, 585(7825):357–362, September 2020. doi: 10.1038/s41586-020-2649-2. URL https: //doi.org/10.1038/s41586-020-2649-2.

Patrick Kidger. torchcubicspline. https://github.com/patrick-kidger/ torchcubicspline, n.d. Software. Accessed 9 November 2026.

Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljacic, Thomas Hou, and Max Tegmark. Kan: Kolmogorov–arnold networks. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Learning Representations, volume 2025, pages 70367– 70413, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ afaed89642ea100935e39d39a4da602c-Paper-Conference.pdf.

Hugo Melchers and Jefrey Cochran. torch-bspline. https://github.com/ jeffrey-cochran/torch\_bspline, 2025. Software. Version 0.1.3.

Shengjun Pan, Brendan Kitts, Tian Zhou, Hao He, Bharatbhushan Shetty, Aaron Flores, Djordje Gligorijevic, Junwei Pan, Tingyu Mao, San Gultekin, et al. Bid shading by win-rate estimation and surplus maximization. arXiv preprint arXiv:2009.09259, 2020.

Alex Shtof, Elie Abboud, Rotem Stram, and Oren Somekh. Function basis encoding of numerical features in factorization machines. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=M4222IBHsh.

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, St´efan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, <sup>˙</sup>Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antˆonio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17:261– 272, 2020. doi: 10.1038/s41592-019-0686-2.

Zhongqing Wang and Benyu Guo. Modified Legendre rational spectral method for the whole line. Journal of Computational Mathematics, 22(3):457–474, 2004. doi: 10.4208/jcm.v22. n3.p457.

Tian Zhou, Hao He, Shengjun Pan, Niklas Karlsson, Bharatbhushan Shetty, Brendan Kitts, Djordje Gligorijevic, San Gultekin, Tingyu Mao, Junwei Pan, Jianlong Zhang, and Aaron Flores. An eficient deep distribution network for bid shading in first-price auctions. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, KDD ’21, page 3996–4004, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450383325. doi: 10.1145/3447548.3467167. URL https://doi. org/10.1145/3447548.3467167.