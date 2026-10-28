# Property-Based Testing for Robust Softmax and Cross-Entropy

Muhammad Maaz

## Abstract

A recent paper of Barron et al. (2026) identified two numerical issues with the softmax and cross-entropy functions in virtually all standard machine learning libraries (e.g., PyTorch and JAX). We show that these issues can be recovered through simple property-based tests in Hypothesis. However, our tests identify an additional edge case where their implementation fails. We then propose an even more robust softmax implementation which passes these tests and also supports use in causal attention masking. We compare our implementation with theirs and discuss some of the differences. Overall, this work shows that property-based testing can be a powerful tool for discovering and fixing numerical issues in machine learning libraries. Our implementations and property-based tests can be found at https://github.com/mmaaz-git/softmax-fix.

## 1 Introduction

Suppose we have logits $z \in \mathbb { R } ^ { K }$ . It is common to compute softma $: ( z ) ~ = ~ e ^ { z } / \sum _ { i = 1 } ^ { K } e ^ { z _ { i } }$ by subtracting the maximum logit (Higham, 2002), which is more numerically stable. More precisely, we compute the log-softmax as follows:

$$
\operatorname { l o g s o f t m a x } ( z ) = ( z - \operatorname* { m a x } _ { i } z _ { i } ) - \log \sum _ { i = 1 } ^ { K } e ^ { z _ { i } - \operatorname* { m a x } _ { i } z _ { i } }
$$

Then, we can compute the softmax as softma $\mathbf { \Psi } _ { : } ( z ) = e ^ { \mathrm { l o g s o f t m a x } ( z ) }$ , or directly by applying the max-shift trick:

$$
\operatorname { s o f t m a x } ( z ) = e ^ { z - \operatorname* { m a x } _ { i } z _ { i } } / \sum _ { i = 1 } ^ { K } e ^ { z _ { i } - \operatorname* { m a x } _ { i } z _ { i } }
$$

The max-shift trick is widely used in practice, e.g., in PyTorch and JAX.

Given logits z and a target probability vector $p \in \Delta ^ { K - 1 }$ , the soft-target cross-entropy is $\operatorname { C E } ( z , p ) =$ $- \textstyle \sum _ { i = 1 } ^ { K } p _ { i }$ [logsoftmax(z)] . When $\mathbf { \boldsymbol { p } } = \mathbf { \boldsymbol { e } } _ { t }$ is a one-hot vector corresponding to target class t, this simplifies to the hard-target cross-entropy: $\mathrm { C E } ( z , e _ { t } ) = - [ \mathrm { l o g s o f t m a x } ( z ) ] _ { i }$ <sub>t</sub>.

Recently, Barron et al. (2026) observed that even with the max-shift trick, two possible numerical issues can arise: (1) softmax collapse, where, as the model becomes more confident in the correct class, the gradient of the hard-target cross-entropy with respect to the target logit incorrectly rounds to 0, and (2) softmax cancellation, where the $L ^ { 2 }$ norm of the gradient of the soft-target cross-entropy incorrectly plateaus at a non-zero value as the student logits approach the teacher logits. They claim that these issues were found and fixed using Gemini Flash 3.7 (Barron, 2026). The problem of softmax collapse in grokking was previously studied by Prieto et al. (2025).

However, we show that these issues can also be found via property-based testing. Property-based testing is a type of software testing where an abstract property of the system under test is specified, and then random inputs are generated and checked against this property. Properties can pertain to single functions (e.g., that for all inputs, the output should be non-negative), or, more powerfully, to multiple functions (e.g., that encoding and decoding yields the original input). Property-based testing began with QuickCheck in Haskell (Claessen and Hughes, 2000), and has since been adopted by many other programming languages, including in Python, in the Hypothesis library (MacIver et al., 2019). Both of these issues expose failures of key mathematical properties of the softmax and cross-entropy functions, which take no more than a few lines of code to test. Property-based testing is increasingly being adopted in the test suites of popular software libraries, like NumPy, and in fact has been used to discover bugs in NumPy and other libraries (Maaz et al., 2025).

However, we will show that Barron et al. (2026)’s fix still fails. We propose a better fix which is even more robust, and show that this allows the new softmax to be used in the setting of causal attention masking. We release our implementations and tests at the codebase at https://github. com/mmaaz-git/softmax-fix.

## 2 Rediscovering the Bugs with Property-Based Testing

For a logit vector z, let $\pmb q = \mathrm { s o f t m a x } ( z )$ and denote its Jacobian by $J ( z )$ , with entries $J _ { i j } = \partial q _ { i } / \partial z _ { j }$ Fix an output index $i \in \{ 1 , \ldots , K \}$ . It is well-known that:

$$
\begin{array} { l } { \displaystyle \frac { \partial q _ { i } } { \partial z _ { i } } = q _ { i } ( 1 - q _ { i } ) , } \\ { \displaystyle \frac { \partial q _ { i } } { \partial z _ { j } } = - q _ { i } q _ { j } , \quad j \neq i . } \end{array}
$$

Thus, the entries of the gradient of each output probability sum to zero:

$$
\sum _ { j = 1 } ^ { K } \frac { \partial q _ { i } } { \partial z _ { j } } = q _ { i } ( 1 - q _ { i } ) - \sum _ { j \ne i } q _ { i } q _ { j } = q _ { i } ( 1 - q _ { i } ) - q _ { i } ( 1 - q _ { i } ) = 0 .
$$

Hence, every row of the Jacobian $J ( z )$ sums to zero in exact arithmetic. Failure of this identity is related to the softmax collapse phenomenon as described by Barron et al. (2026).

To see why, suppose the target logit $z _ { t }$ dominates, and define the margin m by

$$
m = z _ { t } - \log \sum _ { j \neq t } e ^ { z _ { j } } , \qquad q _ { t } = \frac { 1 } { 1 + e ^ { - m } } .
$$

For sufficiently large $m , 1 + e ^ { - m }$ rounds to 1. Using hats to denote computed quantities, we then have $\widehat { q } _ { t } = 1$ and $\textstyle \sum _ { j \neq t } { \widehat { q } } _ { j } > 0$ . The standard softmax backward formula therefore gives

$$
\begin{array} { c } { { \widehat { J } _ { t t } = \widehat { q } _ { t } ( 1 - \widehat { q } _ { t } ) = 0 , } } \\ { { \widehat { J } _ { t j } = - \widehat { q } _ { t } \widehat { q } _ { j } = - \widehat { q } _ { j } , \quad j \neq t , } } \\ { { \displaystyle \sum _ { j = 1 } ^ { K } \widehat { J } _ { t j } = - \sum _ { j \neq t } \widehat { q } _ { j } < 0 . } } \end{array}
$$

This also explains the corresponding hard-target cross-entropy failure. For $L = \mathrm { C E } ( z , e _ { t } ) = - \log q _ { t }$ the chain rule gives

$$
g _ { j } = { \frac { \partial L } { \partial z _ { j } } } = - { \frac { 1 } { q _ { t } } } J _ { t j } .
$$

Propagating the erroneous Jacobian row through this expression yields $\widehat g _ { t } = 0$ and $\widehat { g } _ { j } = \widehat { q } _ { j }$ for $\begin{array} { r } { j \neq t , } \end{array}$ so $\dot { \sum _ { j } { \widehat { g } _ { j } } } > 0$ . The standard cross-entropy backward formula $\widehat { g } _ { t } = \widehat { q } _ { t } - 1$ exhibits the same failure. In exact arithmetic, however, the target gradient is

$$
g _ { t } = - \frac { e ^ { - m } } { 1 + e ^ { - m } } = - \sum _ { j \neq t } q _ { j } ,
$$

which balances the non-target gradients.

In fact, it is possible to compute when this softmax collapse occurs, based on the precision of the numerical representation. Recall that the collapse arises due to floating point numbers near 1. Recall that $q _ { t } = 1 / ( 1 + e ^ { - m } )$ . If the format has b significant bits, then the next representable number above 1 is $1 + 2 ^ { 1 - b } .$ Using a round-to-nearest mode, an addition smaller than half that gap will disappear, so that if $e ^ { - m } < 2 ^ { - b }$ , then the floating point representation of $1 + e ^ { - m }$ will be 1. Thus, under this rounding model, the collapse occurs when $m > b \ln { 2 }$ . For bf $6 , b = 8 ,$ , so the threshold is 5.545; for float32, b = 24, the threshold is 16.636; for float64, $b = 5 3 .$ , the threshold is 36.737. These predicted thresholds line up with the numerical experiments of Barron et al. (2026).

Next, let z and w denote the student and teacher logits, respectively, with student probabilities $\pmb q = \mathrm { s o f t m a x } ( z )$ and target probabilities $\pmb { p } = \mathrm { s o f t m a x } ( \pmb { w } )$ . Holding the teacher logits fixed, in exact arithmetic we have $\nabla _ { z } \operatorname { C E } ( { \dot { z } } , { \pmb { p } } ) = { \pmb q } - { \dot { \pmb p } } .$ . If the student logits approach the teacher logits, we should have that the gradient of the cross-entropy with respect to the student logits approaches 0. So, the second key property we will test is that if the student logits equal the teacher logits, this gradient is also zero. More precisely, for any fixed teacher logits w,

$$
\nabla _ { z } \operatorname { C E } ( z , \operatorname { s o f t m a x } ( \pmb { w } ) ) | _ { z = \pmb { w } } = \mathbf { 0 } .
$$

To see why this can fail, recall that the teacher probabilities are computed using softmax(w), while the backward pass of the cross-entropy can compute the student probabilities using exp(logsoftmax(z)). When the logits are identical, these expressions are mathematically equivalent, but their different sequences of floating-point operations can give different rounding errors.

We can simply test these two properties with Hypothesis, in PyTorch, as below.

## Property 1: Each row of the softmax Jacobian sums to zero.

Here, we compute the Jacobian of the softmax function and check that each row sums to zero with a relative tolerance of $1 0 ^ { - 1 2 }$ , using the sum of the absolute entries in that row as the scale. The code is as follows:

```julia
@given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
@settings(max_examples=10_000, deadline=None)
def test_softmax_gradients_sum_to_zero(softmax, xs):
x = torch.tensor(xs, dtype=torch.float64, requires_grad=True)
J = torch.autograd.functional.jacobian(softmax, x)
assert (J.sum(dim=1).abs() <= 1e-12 * J.abs().sum(dim=1)).all()
```

## Property 2: When student and teacher logits match, the cross-entropy gradient with respect to the student logits is zero, holding the teacher fixed.

Here, soft\_target\_cross\_entropy takes student and teacher logits as its two arguments, and converts the teacher logits to target probabilities internally. Its mathematical loss is therefore CE(z, softmax(w)). The test passes identical logit values for the student and teacher, with x.detach() keeping the teacher fixed when computing the gradient.

```julia
@given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
@settings(max_examples=10_000, deadline=None)
def test_soft_target_cross_entropy_has_zero_gradient_for_identical_logits(
soft_target_cross_entropy, xs):
x = torch.tensor(xs, dtype=torch.float64, requires_grad=True)
loss = soft_target_cross_entropy(x, x.detach())
grad, = torch.autograd.grad(loss, x)
assert (grad == 0).all()
```

These test functions directly test the two properties for a given function (softmax or cross-entropy) and a given tensor. The Hypothesis decorators specify the space of valid inputs, i.e., lists of floats excluding NaN and infinity, with size at least 2, and we generate at most 10,000 such random inputs. The list of floats is then converted to a tensor within the test function. In the implementation in the codebase at https $: / / { \tt g \dot { 1 } }$ thub.com/mmaaz-git/softmax-fix, we additionally add a parameter to allow the user to specify which softmax or cross-entropy function to test.

As expected, we find that using the built-in PyTorch implementations of the softmax and crossentropy functions, these properties fail. On one test run, Hypothesis finds the following failures: the vector [0.0, 37.0] breaks Property 1, and the vector [0.0, 1.0] breaks Property 2. For Property 1, the Jacobian row corresponding to the maximum logit is approximately $[ - 8 . 5 3 3 \times 1 0 ^ { - 1 7 } , 0 ] \colon$ the diagonal derivative has collapsed to zero while the other derivative remains nonzero. Its margin of 37 exceeds the predicted float64 threshold of 36.737. For Property 2, the computed gradient is approximately $[ - 5 . 5 5 \times 1 0 ^ { - 1 7 }$ , 0], rather than zero.

## 3 A Remaining Failure in the Proposed Fix

To fix softmax collapse, Barron et al. (2026) proposed a fix to logsoftmax (and compute softmax(·) as exp(logsoftmax(·))) and hard-target cross-entropy. They also introduce a new version of the soft-target cross-entropy to fix softmax cancellation. We refer the reader to their paper for the details, which we have also re-implemented in our codebase.

We then test Properties 1 and 2 on their functions. Surprisingly, we find that the test for Property 1 fails, although Property 2 passes. In particular, Hypothesis finds that the following vector produces a NaN Jacobian: [1.7976931348623157 · $1 0 ^ { 3 0 8 } , - 9 . 9 7 9 2 0 1 5 4 7 6 7 3 6 \cdot 1 0 ^ { 2 9 1 } ]$ To see where this failure arises, note that (Barron et al., 2026) do the following to compute logsoftmax: subtract the maximum logit from the logits, explicitly setting the maximum logit to 0; then, only compute the logsumexp over shifted entries except the maximum logit, and finally apply softplus to the result; lastly, subtracting this log-normalizer from each shifted logit gives logsoftmax. This fixes the issue that we identified where 1 plus a small number can round to 1.

However, in the failure identified by Hypothesis, performing max-shifting overflows the second element to −∞, so when we exclude the maximum we take the logsumexp of −∞. Although the forward value is correctly −∞, the backward value becomes NaN. Importantly, note that this error does not occur with the built-in PyTorch implementation of softmax. For example, for the vector $[ 1 0 ^ { 3 0 8 } , - 1 0 ^ { 3 0 8 } ]$ , both the standard implementation and the Barron et al. (2026) implementation correctly give [1, 0] for the forward value; but the standard implementation gives all zeros for the backward value, while the Barron et al. (2026) implementation gives all NaNs.

## 4 Our Fix

We propose a fix for computing logsoftmax (and hence softmax) which preserves the utility of Barron et al. (2026)’s fix, but also retains proper gradients in the extreme case we identified above. In our fix, to compute logsoftmax, we perform max-shifting as in Barron et al. (2026). But, then, we compute the sum of the exponentials of the shifted non-maximum logits, and then apply log1p to get the log-normalizer; then, lastly, we subtract this log-normalizer from each shifted logit as usual.

In essence, the difference is the following:

```python
# Barron et al.
log_S = logsumexp(others)
normalizer = softplus(log_S)
# Ours
S = exp(others).sum()
normalizer = log1p(S)
```

See our codebase for the full implementation. Mathematically, these are equivalent. Let a be the shifted logits excluding the selected maximum, and define $S \doteq \textstyle \sum _ { i } e ^ { a _ { i } }$ . Then, for $S > 0$ , we have:

$$
\begin{array} { r l } { \mathrm { s o f t p l u s } ( \mathrm { l o g s u m e x p } ( \pmb { a } ) ) = \mathrm { s o f t p l u s } ( \mathrm { l o g } S ) } & { } \\ & { = \mathrm { l o g } ( 1 + e ^ { \log S } ) } \\ & { = \mathrm { l o g } ( 1 + S ) } \\ & { = \mathrm { l o g 1 p } ( S ) } \end{array}
$$

But now note that for $S = 0$ , the derivative of log1p(S) is finite and equals 1, whereas automatic differentiation of logsumexp on an all-−∞ vector gives NaN. Indeed, our implementation actually does pass the tests for Properties 1 and 2. To see this concretely, consider the vector $[ 1 0 ^ { 3 0 8 } , - 1 0 ^ { 3 0 8 } ]$ Our implementation gives the forward value [1, 0], and the backward value all zeros, as seen in Table 1.

Table 1: Forward softmax probabilities and computed Jacobians for $[ 1 0 ^ { 3 0 8 } , - 1 0 ^ { 3 0 8 } ]$ , evaluated in float64 on CPU using PyTorch 2.7.1.
<table><tr><td>Implementation</td><td>Forward probabilities</td><td>Computed Jacobian</td></tr><tr><td>Standard (PyTorch)</td><td>[1,0]</td><td>All zeros</td></tr><tr><td>Barron et al. (2026) (PyTorch translation)</td><td>[1, 0]</td><td>All NaNs</td></tr><tr><td>Ours</td><td>[1, 0]</td><td>All zeros</td></tr></table>

We reuse Barron et al. (2026)’s fix for soft-target cross-entropy, only changing the logsoftmax computation. We also note that using log1p is not necessarily a new idea: Blanchard et al. (2021) also use it to compute logsumexp. However, our implementation combines this with Barron et al. (2026)’s explicit handling of the maximum logit.

The root of the Barron et al. (2026) failure is taking the logsumexp of an all-−∞ vector. While this can certainly occur with enormous finite-logit margins, this also appears practically in e.g., causal attention masking in transformers (Vaswani et al., 2017). Precisely, the first query can attend only to the first key, giving a masked logit row that looks like $[ a , - \infty , - \infty , \ldots , - \infty ]$ , where a is finite. Excluding the maximum logit gives an all-−∞ vector, who’s logsumexp is −∞, but whose backward gives NaN gradients. So, using Barron et al. (2026)’s softmax for the attention probabilities in this case would produce NaN gradients, whereas our fix gives the correct forward and backward values.

## 5 Conclusion

We reframed the numerical issues found by Barron et al. (2026) as failures of simple mathematical properties of the softmax and cross-entropy functions, and wrote property-based tests that rediscover these issues. However, we then find that their proposed softmax fails the Jacobian row-sum test by producing NaNs for some finite inputs. We then developed a new fix, using log1p, which makes it more robust to overflow after max-shifting, and also adequately handles the case of causal attention masking. More generally, we hope that property-based testing can be adopted more widely in test suites in machine learning libraries to discover such numerical issues.

## References

Jonathan T. Barron. Tweet about softmax cancellation and softmax collapse. X (formerly Twitter), October 2026. URL https://x.com/jon\_barron/status/2113707885486375022. Posted October 23, 2026. Accessed October 25, 2026.

Jonathan T. Barron, Dan Gnanapragasam, and Michael Tschannen. A fix for softmax and crossentropy, 2026. URL https://arxiv.org/abs/2610.31642.

Pierre Blanchard, Desmond J. Higham, and Nicholas J. Higham. Accurately computing the log-sumexp and softmax functions. IMA Journal ofNumerical Analysis, 41(4):2311–2330, 2021. doi: 10.1093/imanum/draa038. URL https://doi.org/10.1093/imanum/draa038.

Koen Claessen and John Hughes. QuickCheck: A lightweight tool for random testing of Haskell programs. In Proceedings ofthe Fifth ACM SIGPLAN International Conference on Functional Programming, ICFP ’00, pages 268–279. Association for Computing Machinery, 2000. doi: 10.1145/351240.351266. URL https://doi.org/10.1145/351240.351266.

Nicholas J. Higham. Accuracy and Stability ofNumerical Algorithms. Society for Industrial and Applied Mathematics, 2 edition, 2002.

Muhammad Maaz, Liam DeVoe, Zac Hatfield-Dodds, and Nicholas Carlini. Agentic property-based testing: Finding bugs across the Python ecosystem, 2025. URL https://arxiv.org/abs/2510. 09907. Presented at the 4th Deep Learning for Code Workshop, NeurIPS 2025.

David R. MacIver, Zac Hatfield-Dodds, and many other contributors. Hypothesis: A new approach to property-based testing. Journal ofOpen Source Software, 4(43):1891, 2019. doi: 10.21105/joss. 01891. URL https://joss.theoj.org/papers/10.21105/joss.01891.

Lucas Prieto, Melih Barsbey, Pedro A. M. Mediano, and Tolga Birdal. Grokking at the edge of numerical stability, 2025. URL https://arxiv.org/abs/2501.04697.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017. URL https://papers.nips.cc/paper\_files/paper/ 2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.