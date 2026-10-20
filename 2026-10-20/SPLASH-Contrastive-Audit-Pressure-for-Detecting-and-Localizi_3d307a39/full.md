# SPLASH: Contrastive Audit Pressure for Detecting and Localizing Security Hallucinations in Generated Code

MIRZA MASFIQUR RAHMAN<sup>∗†</sup>, Purdue University, USA

MD AJWAD AKIL, Purdue University, USA

IMTIAZ KARIM, University of Texas at Dallas, USA

ASHISH KUNDU, Cisco Research, USA

RAMANA KOMPELLA, Cisco Research, USA

ELISA BERTINO, Purdue University, USA

Code <sub>g</sub>enerated b<sub>y</sub> lar<sub>g</sub>e lan<sub>g</sub>ua<sub>g</sub>e models often <sub>p</sub>asses ever<sub>y</sub> functional test while remainin<sub>g</sub> insecure. These security hallucinations are common, written with high confidence, and repeated across samples, so neither the <sub>g</sub>enerator’s own uncertaint<sub>y</sub> nor static anal<sub>y</sub>zers reliabl<sub>y</sub> catch them. We introduce SPLASH, which is <sub>gu</sub>id<sub>e</sub>d b<sub>y</sub> <sub>a</sub> <sub>new</sub> <sub>measure</sub> <sub>ca</sub>ll<sub>e</sub>d <sub>au</sub>dit <sub>pressure.</sub> It <sub>con</sub>t<sub>ras</sub>t<sub>s</sub> t<sub>eac</sub>h<sub>er-</sub>f<sub>orce</sub>d l<sub>og-pro</sub>b<sub>a</sub>biliti<sub>es</sub> f<sub>or</sub> th<sub>e</sub> <sub>genera</sub>t<sub>e</sub>d <sub>p</sub>ro<sub>g</sub>ram under a securit<sub>y</sub> review and a st<sub>y</sub>le review. A<sub>gg</sub>re<sub>g</sub>ated to statements and <sub>p</sub>ro<sub>g</sub>rams<sub>,</sub> and <sub>g</sub>rounded in a <sub>p</sub>robabilistic model of com<sub>p</sub>etin<sub>g</sub> edits, audit <sub>p</sub>ressure allows SPLASH to choose the secure <sub>p</sub>ro<sub>g</sub>ram amon<sub>g</sub> <sub>a mo</sub>d<sub>e</sub>l’<sub>s can</sub>did<sub>a</sub>t<sub>e programs an</sub>d l<sub>oca</sub>t<sub>e</sub> th<sub>e vu</sub>l<sub>nera</sub>bl<sub>e s</sub>t<sub>a</sub>t<sub>emen</sub>t<sub>, w</sub>ith <sub>no mo</sub>d<sub>e</sub>l t<sub>ra</sub>i<sub>n</sub>i<sub>ng,</sub> CWE<sub>-spec</sub>ifi<sub>c</sub> <sub>ru</sub>l<sub>es, or</sub> i<sub>n</sub>t<sub>erna</sub>l <sub>access</sub> t<sub>o</sub> th<sub>e genera</sub>ti<sub>ng mo</sub>d<sub>e</sub>l<sub>.</sub> O<sub>n</sub> 63<sub>,</sub>684 <sub>programs genera</sub>t<sub>e</sub>d b<sub>y</sub> 12 <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> f<sub>or</sub> 899 t<sub>as</sub>k<sub>s</sub> SPLASH <sub>p</sub>icks secure code in 74% of cases, <sub>p</sub>laces the vulnerable statement first in 58% of insecure <sub>p</sub>ro<sub>g</sub>rams, an<sup>d</sup> c<sup>h</sup>ooses <sup>b</sup>etter t<sup>h</sup>an a <sup>f</sup>rontier LLM ju<sup>d</sup>ge.

CCS Concepts: • Security and privacy → Software and application security.

Additional Ke<sub>y</sub> Words and Phrases: Code Generation<sub>,</sub> Sec<sub>u</sub>rit<sub>y,</sub> Hall<sub>u</sub>cination<sub>,</sub> Lar<sub>g</sub>e Lan<sub>gu</sub>a<sub>g</sub>e Models

## 1 Introduction

Lar<sub>g</sub>e lan<sub>g</sub>ua<sub>g</sub>e models (LLMs) [8, 14, 36] now <sub>p</sub>roduce s<sub>y</sub>ntacticall<sub>y</sub> and functionall<sub>y</sub> correct <sub>p</sub>ro-<sub>g</sub>rams from natural-lan<sub>g</sub>ua<sub>g</sub>e s<sub>p</sub>ecifications and are widel<sub>y</sub> inte<sub>g</sub>rated into commercial develo<sub>p</sub>ment workflows [27, 41]. The standard <sub>p</sub>ractice for validatin<sub>g</sub> <sub>g</sub>enerated code is functional testin<sub>g</sub>: if the <sub>p</sub>ro<sub>g</sub>ram <sub>p</sub>asses its test suite<sub>,</sub> it is considered read<sub>y</sub> to de<sub>p</sub>lo<sub>y</sub>. However<sub>,</sub> a <sub>p</sub>ro<sub>g</sub>ram can satisf<sub>y</sub> ever<sub>y</sub> functional re<sub>q</sub>uirement while containin<sub>g</sub> a securit<sub>y</sub> vulnerabilit<sub>y</sub> that no unit test exercises.

A model asked to hash a user’s <sub>p</sub>assword ma<sub>y</sub> write sha256(pw) without a salt: the code com<sub>p</sub>iles, returns the ex<sub>p</sub>ected di<sub>g</sub>est, and <sub>p</sub>asses the unit test. The secure alternative, bcrypt with a random salt, is rarely among the model’s likely continuations. This is a security hallucination [5, 44] (Definition 2.1): a functionall<sub>y</sub> correct <sub>p</sub>ro<sub>g</sub>ram that fails its securit<sub>y</sub> test. For ca<sub>p</sub>able models, the <sub>same mo</sub>d<sub>e</sub>l th<sub>a</sub>t <sub>wr</sub>it<sub>es unsa</sub>lt<sub>e</sub>d SHA<sub>-</sub>256 <sub>can</sub> id<sub>en</sub>tif<sub>y</sub> th<sub>e</sub> fl<sub>aw w</sub>h<sub>en as</sub>k<sub>e</sub>d t<sub>o rev</sub>i<sub>ew</sub> it<sub>, s</sub>h<sub>ow</sub>i<sub>ng</sub> th<sub>a</sub>t the securit<sub>y</sub> knowled<sub>g</sub>e is <sub>p</sub>resent but remains inactive durin<sub>g g</sub>eneration. Securit<sub>y</sub> hallucinations <sub>are</sub> <sub>a</sub>l<sub>so</sub> <sub>pro</sub>d<sub>uce</sub>d <sub>w</sub>ith hi<sub>g</sub>h <sub>con</sub>fid<sub>ence</sub> <sub>an</sub>d <sub>repro</sub>d<sub>uce</sub>d <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>tl<sub>y</sub> <sub>across</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>samp</sub>l<sub>es,</sub> makin<sub>g</sub> them invisible to methods that rel<sub>y</sub> on out<sub>p</sub>ut variation.

Securit<sub>y</sub> hallucinations are common. Amon<sub>g</sub> workin<sub>g p</sub>ro<sub>g</sub>rams from o<sub>p</sub>en-wei<sub>g</sub>ht models in our evaluation, 24–60% fail their securit<sub>y</sub> tests. Frontier models are safer (Claude O<sub>p</sub>us 5: 6%, GPT-5.6 Sol: 8%), but a<sub>g</sub>entic <sub>p</sub>i<sub>p</sub>elines retain rou<sub>g</sub>hl<sub>y</sub> a third of their workin<sub>g</sub> solutions insecure.

![](images/e169ed7edc37e831c1f95475be31d6e40f676d9db6f103db61023e0811e4aeab.jpg)  
Fig. 1. A working but insecure program. DeepSeek-Coder-6.7B evaluates the user’s expression with eval (line 12; CWE-94), which passes the unit tests and fails the security test. Left: the writer’s own uncertainty peaks on the blocklist (line 5), not on the flaw; audit pressure (our method) peaks on line 12. Right: at line 12 the security reviewer expected a comment before return (91%) and (ast after eval (74%), and gave the program’s own tokens under 1%; the style reviewer gives them far more (2% and 51%).

S<sub>uc</sub>h f<sub>requen</sub>t <sub>na</sub>t<sub>ure an</sub>d <sub>con</sub>fid<sub>en</sub>t b<sub>e</sub>h<sub>av</sub>i<sub>or o</sub>f th<sub>e mo</sub>d<sub>e</sub>l<sub>s pos</sub>iti<sub>ons</sub> it <sub>as a cr</sub>iti<sub>ca</sub>l <sub>concern</sub> i<sub>n</sub> <sub>so</sub>ft<sub>ware</sub> d<sub>eve</sub>l<sub>opmen</sub>t lif<sub>ecyc</sub>l<sub>e.</sub>

Limitations of existing approaches. Hallucination detectors for natural language measure variation in the <sub>g</sub>enerator’s out<sub>p</sub>ut distribution [7, 16, 18, 23, 25], but securit<sub>y</sub> hallucinations violate their shared assum<sub>p</sub>tion as 86% of insecure workin<sub>g</sub> <sub>p</sub>ro<sub>g</sub>rams have onl<sub>y</sub> insecure siblin<sub>g</sub>s<sub>,</sub> and <sub>g</sub>eneration uncertaint<sub>y</sub> is weak (AUROC 0.61). Static anal<sub>y</sub>zers [4, 34, 38, 47] <sub>p</sub>roduce a CWEmatched findin<sub>g</sub> on onl<sub>y</sub> 15% of securit<sub>y</sub> hallucinations. Generation-time defenses [11, 12, 48] reduce vulnerabilit<sub>y</sub> rates b<sub>y</sub> 25–35% but re<sub>q</sub>uire trainin<sub>g</sub> access and do not eliminate hallucinations. Post-hoc methods such as <sub>p</sub>rocess reward models [49] and steerin<sub>g</sub> vectors [37] need labeled data an<sup>d</sup> w<sup>h</sup>ite-<sup>b</sup>ox access. LLM-as-ju<sup>d</sup>ge approac<sup>h</sup>es can name <sup>fl</sup>awe<sup>d</sup> <sup>l</sup>ines <sup>b</sup>ut give no gra<sup>d</sup>e<sup>d</sup> score to r<sub>a</sub>nk <sub>ca</sub>ndid<sub>a</sub>t<sub>es</sub> <sub>a</sub>nd <sub>cos</sub>t <sub>a</sub>n API <sub>ca</sub>ll <sub>pe</sub>r <sub>p</sub>r<sub>og</sub>r<sub>a</sub>m<sub>.</sub>

Research challenges. Three properties make security hallucinations resistant to the approaches above. (C1) The <sub>g</sub>enerator’s own token distribution carries little si<sub>g</sub>nal about securit<sub>y</sub>: when code is written with near-zero entro<sub>py,</sub> the tokens reveal ne<sub>g</sub>li<sub>g</sub>ible information about an<sub>y</sub> downstream <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> (Pro<sub>p</sub>osition 2.2). (C2) The vulnerabilit<sub>y</sub> is t<sub>yp</sub>icall<sub>y</sub> an absence rather than a <sub>p</sub>resence (missin<sub>g</sub> validation, omitted salt), so there is no token to fla<sub>g</sub>. (C3) Whether code is insecure de<sub>p</sub>ends more on the task than on the model (57% of variance vs 6%), so absolute scores do not transfer across tasks for an<sub>y</sub> jud<sub>g</sub>e. SPLASH therefore asks the two <sub>q</sub>uestions a develo<sub>p</sub>er actuall<sub>y</sub> has: which <sub>can</sub>did<sub>a</sub>t<sub>e s</sub>h<sub>ou</sub>ld I k<sub>eep, an</sub>d <sub>w</sub>hi<sub>c</sub>h li<sub>ne s</sub>h<sub>ou</sub>ld I <sub>rev</sub>i<sub>ew</sub>?

Our approach. We present SPLASH (Statement-level Pressure Localization via Audit against Security Hallucinations). At its core is audit pressure, a novel method that reads the model’s distribution under review (addressin<sub>g</sub> C1): an auditor model reads a candidate <sub>p</sub>ro<sub>g</sub>ram under two instructions, a security-review and a style-review, and the program is scored token by token via t<sub>eac</sub>h<sub>er</sub> f<sub>orc</sub>i<sub>ng</sub> <sub>w</sub>ith <sub>no</sub>thi<sub>ng</sub> <sub>samp</sub>l<sub>e</sub>d<sub>.</sub> Th<sub>e</sub> <sub>me</sub>th<sub>o</sub>d <sub>recor</sub>d<sub>s</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>au</sub>dit <sub>pressure</sub> <sub>w</sub>h<sub>erever</sub> th<sub>e</sub> <sub>secur</sub>it<sub>y</sub> <sub>rev</sub>i<sub>ewer</sub> <sub>wou</sub>ld h<sub>ave</sub> <sub>wr</sub>itt<sub>en</sub> dif<sub>eren</sub>t <sub>co</sub>d<sub>e</sub> b<sub>u</sub>t th<sub>e</sub> <sub>s</sub>t<sub>y</sub>l<sub>e</sub> <sub>rev</sub>i<sub>ewer</sub> i<sub>s</sub> f<sub>ar</sub> <sub>more</sub> <sub>w</sub>illi<sub>ng</sub> t<sub>o</sub> <sub>repro</sub>d<sub>uce</sub> th<sub>e</sub> t<sub>o</sub>k<sub>en.</sub> B<sub>ecause</sub> t<sub>eac</sub>h<sub>er-</sub>f<sub>orce</sub>d <sub>copy</sub>i<sub>ng</sub> <sub>revea</sub>l<sub>s</sub> <sub>w</sub>h<sub>ere</sub> th<sub>e</sub> <sub>rev</sub>i<sub>ewer</sub> <sub>wan</sub>t<sub>e</sub>d t<sub>o</sub> i<sub>nser</sub>t <sub>co</sub>d<sub>e,</sub> th<sub>e pressure o</sub>f <sub>a m</sub>i<sub>ss</sub>i<sub>ng c</sub>h<sub>ec</sub>k l<sub>an</sub>d<sub>s on</sub> th<sub>e</sub> li<sub>ne w</sub>h<sub>ere</sub> it <sub>wou</sub>ld h<sub>ave</sub> b<sub>een wr</sub>itt<sub>en</sub> (addressin<sub>g</sub> C2). Fi<sub>g</sub>ure 1 shows the conce<sub>p</sub>t: on eval(math\_expr), the securit<sub>y</sub> reviewer ex<sub>p</sub>ected (ast after eval, i.e. <sub>p</sub>arsin<sub>g</sub> the in<sub>p</sub>ut safel<sub>y</sub>, while st<sub>y</sub>le reviewer acce<sub>p</sub>ts it far more readil<sub>y</sub>.

The <sub>p</sub>er-token lo<sub>g</sub>-<sub>p</sub>robabilit<sub>y</sub> diference between the two instructions decom<sub>p</sub>oses exactl<sub>y</sub> into a lo<sub>g</sub> Ba<sub>y</sub>es factor b<sub>y</sub> the chain rule. Under a formal model of com<sub>p</sub>etin<sub>g</sub> edits (§4), this token-level <sub>p</sub>ressure a<sub>gg</sub>re<sub>g</sub>ates to statement scores and <sub>p</sub>ro<sub>g</sub>ram scores with a <sub>p</sub>robabilistic inter<sub>p</sub>retation. The decomposition enables two outputs from the same pass: (1) selecting the most secure candidate (by lowest program score) and (2) localizing the vulnerable statement (by highest statement score). B<sub>ecause on</sub>l<sub>y w</sub>ithi<sub>n-</sub>t<sub>as</sub>k <sub>compar</sub>i<sub>sons are nee</sub>d<sub>e</sub>d<sub>,</sub> th<sub>e</sub> t<sub>as</sub>k<sub>-</sub>l<sub>eve</sub>l <sub>s</sub>hift<sub>s</sub> th<sub>a</sub>t d<sub>e</sub>f<sub>ea</sub>t <sub>cross-</sub>t<sub>as</sub>k classification do not afect either out<sub>p</sub>ut (addressin<sub>g</sub> C3). SPLASH a<sub>pp</sub>lies to code from an<sub>y</sub> source - o<sub>p</sub>en-wei<sub>g</sub>ht models<sub>,</sub> closed-API models<sub>,</sub> and a<sub>g</sub>entic <sub>p</sub>i<sub>p</sub>elines<sub>,</sub> with no trainin<sub>g,</sub> no CWE-s<sub>p</sub>ecific <sub>ru</sub>l<sub>es, an</sub>d <sub>no</sub> l<sub>a</sub>b<sub>e</sub>l<sub>e</sub>d d<sub>a</sub>t<sub>a.</sub>

In practice. Security tests do not exist in ordinary development as they must be written per task b<sub>y someone w</sub>h<sub>o a</sub>l<sub>rea</sub>d<sub>y</sub> k<sub>nows</sub> th<sub>e vu</sub>l<sub>nera</sub>bilit<sub>y.</sub> D<sub>eve</sub>l<sub>opers</sub> h<sub>ave, a</sub>t <sub>mos</sub>t<sub>, un</sub>it t<sub>es</sub>t<sub>s.</sub> Alth<sub>oug</sub>h we use security tests in our evaluation to measure efectiveness of SPLASH], it does not rely on them as in<sub>p</sub>uts. Meanwhile, sam<sub>p</sub>lin<sub>g</sub> several candidates is alread<sub>y</sub> a common <sub>p</sub>ractice (IDE su<sub>gg</sub>estions, best-of-�, a<sub>g</sub>ent retries), but there is no established wa<sub>y</sub> to choose amon<sub>g</sub> them for securit<sub>y</sub>. Audit <sub>p</sub>ressure fills this slot as it re<sub>q</sub>uires no trainin<sub>g,</sub> no <sub>p</sub>er-CWE rules<sub>,</sub> no <sub>g</sub>enerator wei<sub>g</sub>hts<sub>,</sub> and no <sub>p</sub>aid API. It works with whatever unit tests exist and also without an<sub>y</sub> (Fi<sub>g</sub>ure 5b), and <sub>p</sub>oints the <sub>rev</sub>i<sub>ewer a</sub>t th<sub>e</sub> li<sub>ne; e.g.,</sub> th<sub>e vu</sub>l<sub>nera</sub>bl<sub>e s</sub>t<sub>a</sub>t<sub>emen</sub>t i<sub>s</sub> i<sub>n</sub> th<sub>e</sub> t<sub>op</sub> 3 i<sub>n</sub> 85% <sub>o</sub>f <sub>cases.</sub>

Findings. We evaluate SPLASH on 899 tasks with executable security tests across 12 generators:

• Selection. SPLASH picks secure code in 74% of decisive candidate pools (random: 54%, uncertaint<sub>y</sub>: 65%), closin<sub>g</sub> 46% of the <sub>g</sub>a<sub>p</sub> to a <sub>p</sub>erfect <sub>p</sub>ick. On held-out tasks, the <sub>g</sub>ain is within two <sub>p</sub>oints. On a<sub>g</sub>entic code, SPLASH <sub>p</sub>icks secure in 76% of <sub>p</sub>ools (random: 57%).

• Localization. SPLASH places the vulnerable statement first in 58% of insecure programs (random: 9%) and identifies the exact chan<sub>g</sub>ed line in 77% of benchmark reference <sub>p</sub>airs.

• Practitioner comparison. SPLASH selects secure code more often than a frontier LLM judge (77% vs. 68%) while re<sub>q</sub>uirin<sub>g</sub> onl<sub>y</sub> local inference rather than <sub>p</sub>er-<sub>q</sub>uer<sub>y</sub> API calls. Althou<sub>g</sub>h the jud<sub>g</sub>e localizes more <sub>p</sub>recisel<sub>y</sub> on its sam<sub>p</sub>le (70% vs. 54% to<sub>p</sub>-1), SPLASH <sub>p</sub>rovides both <sub>se</sub>l<sub>ec</sub>ti<sub>on an</sub>d l<sub>oca</sub>li<sub>za</sub>ti<sub>on</sub> f<sub>rom</sub> th<sub>e same pass.</sub>

Contributions. We make the following contributions in this paper:

• Problem characterization. We provide the first study showing that security hallucinations <sub>are</sub> <sub>wr</sub>itt<sub>en</sub> <sub>con</sub>fid<sub>en</sub>tl<sub>y</sub> <sub>an</sub>d <sub>repea</sub>t<sub>e</sub>d <sub>across</sub> <sub>samp</sub>l<sub>es,</sub> <sub>across</sub> 12 <sub>genera</sub>t<sub>ors</sub> <sub>an</sub>d 899 t<sub>es</sub>t<sub>e</sub>d t<sub>as</sub>k<sub>s,</sub> establishin<sub>g</sub> that <sub>g</sub>eneration uncertaint<sub>y</sub> and static anal<sub>y</sub>sis both miss most of them (§6.1).

• Method and system. We introduce audit pressure, a training-free, model-agnostic signal <sub>g</sub>rounded in a formal model of com<sub>p</sub>etin<sub>g</sub> edits, and build SPLASH, a <sub>p</sub>i<sub>p</sub>eline that takes <sub>g</sub>ener-<sub>a</sub>t<sub>e</sub>d <sub>co</sub>d<sub>e</sub> f<sub>rom</sub> <sub>any</sub> <sub>source</sub> <sub>an</sub>d <sub>pro</sub>d<sub>uces</sub> <sub>per-s</sub>t<sub>a</sub>t<sub>emen</sub>t l<sub>oca</sub>li<sub>za</sub>ti<sub>on</sub> <sub>an</sub>d <sub>per-poo</sub>l <sub>can</sub>did<sub>a</sub>t<sub>e</sub> selection (§4, §3).

• Rigorous evaluation. We show that SPLASH achieves consistent gains across benchmarks, tem<sub>p</sub>eratures<sub>,</sub> model sizes<sub>,</sub> and beats random chance on 16 of 17 CWE t<sub>yp</sub>es<sub>,</sub> com<sub>p</sub>arin<sub>g</sub> a<sub>g</sub>ainst static anal<sub>y</sub>zers, a frontier LLM jud<sub>g</sub>e, and the <sub>g</sub>enerator’s own uncertaint<sub>y</sub> (§6.2–§6.5). The code and artifacts of SPLASH are <sub>p</sub>ublicl<sub>y</sub> available at: htt<sub>p</sub>s://<sub>g</sub>ithub.com/Prometheused SPLASH

## 2 Background

I<sub>n</sub> thi<sub>s sec</sub>ti<sub>on, we</sub> di<sub>scuss re</sub>l<sub>evan</sub>t b<sub>ac</sub>k<sub>groun</sub>d <sub>on co</sub>d<sub>e genera</sub>ti<sub>on, secur</sub>it<sub>y</sub> h<sub>a</sub>ll<sub>uc</sub>i<sub>na</sub>ti<sub>on, an</sub>d code securit<sub>y</sub> anal<sub>y</sub>sis.

## 2.1 Code Generation and Generation Uncertainty

Given the <sub>p</sub>rom<sub>p</sub>t and the tokens <sub>g</sub>enerated so far<sub>,</sub> an autore<sub>g</sub>ressive code model writes a <sub>p</sub>ro<sub>g</sub>ram one token at a time, sam<sub>p</sub>lin<sub>g</sub> each token from its <sub>p</sub>redicted distribution [8, 13, 14, 36]. Var<sub>y</sub>in<sub>g</sub> the random seed or the sam<sub>p</sub>lin<sub>g</sub> tem<sub>p</sub>erature<sub>,</sub> a sin<sub>g</sub>le <sub>p</sub>rom<sub>p</sub>t <sub>y</sub>ields man<sub>y</sub> distinct candidate <sub>programs</sub> th<sub>a</sub>t <sub>s</sub>h<sub>are</sub> th<sub>e same</sub> f<sub>unc</sub>ti<sub>ona</sub>l i<sub>n</sub>t<sub>en</sub>t b<sub>u</sub>t <sub>may</sub> dif<sub>er</sub> i<sub>n correc</sub>t<sub>ness, s</sub>t<sub>y</sub>l<sub>e, an</sub>d <sub>secur</sub>it<sub>y.</sub> <sup>For</sup> <sup>a</sup> p<sup>rom</sup>p<sup>t</sup> � <sup>and</sup> <sup>a</sup> g<sup>enerated</sup> p<sup>ro</sup>g<sup>ram</sup> $P = \left( c _ { 1 } , \ldots , c _ { n } \right)$ , the generation uncertainty of model � is

$$
U ( P ) = - { \frac { 1 } { n } } \sum _ { t = 1 } ^ { n } \log p _ { M } { \bigl ( } c _ { t } \mid x , c _ { < t } { \bigr ) } ,\tag{1}
$$

which is hi<sub>g</sub>h when the model is unsure of the tokens. It serves as our main baseline (§5).

## 2.2 Security Hallucinations

A hallucination in code generation is typically defined as output that deviates from the intended behavior: the <sub>p</sub>ro<sub>g</sub>ram does not com<sub>p</sub>ile, fails its tests, or solves the wron<sub>g</sub> <sub>p</sub>roblem [21, 35, 42]. We d<sub>e</sub>fi<sub>ne a s</sub>t<sub>r</sub>i<sub>c</sub>t<sub>er var</sub>i<sub>an</sub>t<sub>.</sub>

Definition 2.1 (Security hallucination). A security hallucination is a generated program � that <sub>p</sub>asses ever<sub>y</sub> functional test and fails at least one securit<sub>y</sub> test: Func(�) = 1 and Sec(�) = 0. A program is secure and working if Func $( P ) = 1$ <sub>an</sub>d S<sub>ec</sub> $( P ) = 1$

Securit<sub>y</sub> hallucinations are dan<sub>g</sub>erous because the<sub>y</sub> a<sub>pp</sub>ear correct - the <sub>p</sub>ro<sub>g</sub>ram runs<sub>,</sub> returns th<sub>e</sub> <sub>r</sub>i<sub>g</sub>ht <sub>ou</sub>t<sub>pu</sub>t<sub>s,</sub> <sub>an</sub>d <sub>o</sub>ft<sub>en</sub> <sub>passes</sub> <sub>a</sub> d<sub>eve</sub>l<sub>oper</sub>’<sub>s</sub> <sub>spo</sub>t <sub>c</sub>h<sub>ec</sub>k<sub>s.</sub> H<sub>a</sub>ll<sub>uc</sub>i<sub>na</sub>ti<sub>on</sub> d<sub>e</sub>t<sub>ec</sub>t<sub>ors</sub> <sub>re</sub>l<sub>y</sub> <sub>on</sub> th<sub>e</sub> model hesitatin<sub>g</sub> or disa<sub>g</sub>reein<sub>g</sub> with itself across sam<sub>p</sub>les [7, 23]. However, securit<sub>y</sub> hallucinations t<sub>en</sub>d t<sub>o</sub> b<sub>e</sub> <sub>con</sub>fid<sub>en</sub>t <sub>an</sub>d <sub>repea</sub>t<sub>e</sub>d <sub>across</sub> <sub>samp</sub>l<sub>es.</sub> Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> b<sub>oun</sub>d <sub>ma</sub>k<sub>es</sub> it <sub>prec</sub>i<sub>se</sub> <sub>w</sub>h<sub>y</sub> <sub>con</sub>fid<sub>en</sub>t <sub>co</sub>d<sub>e carr</sub>i<sub>es</sub> littl<sub>e ev</sub>id<sub>ence.</sub>

Proposition 2.2 (Entropy bounds what a window can reveal). Let $r ( \cdot \mid u )$ be the distribution the generator samples from after prefix � and $H _ { r } ( u )$ its entropy. For a window $W = \left( Y _ { a } , \ldots , Y _ { b - 1 } \right)$ starting at position � and ending at stopping time $b ,$ and for every function � of the finished program (in particular $\beta = \mathsf { S e c } )$

$$
I \big ( \beta ( P ) ; W \mid u _ { 0 } \big ) \ \leq \ \mathbb { E } \left[ \sum _ { t = a } ^ { b - 1 } H _ { r } ( Y _ { < t } ) \Bigg | \ u _ { 0 } \right] .\tag{2}
$$

A statement written with near-zero entro<sub>py</sub> therefore reveals almost nothin<sub>g</sub> about Sec(�) to an<sub>y</sub> f<sub>unc</sub>ti<sub>on o</sub>f th<sub>e genera</sub>t<sub>or</sub>’<sub>s own</sub> t<sub>o</sub>k<sub>en</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub>

## 2.3 Static Analysis

Vulnerabilit<sub>y</sub> t<sub>yp</sub>es are documented b<sub>y</sub> the Common Weakness Enumeration (CWE) [24]. Static anal<sub>y</sub>zers such as Sem<sub>g</sub>re<sub>p</sub> [38], Bandit [34], C<sub>pp</sub>check [4], and Flawfinder [47] fla<sub>g</sub> vulnerable <sub>p</sub>atterns without executin<sub>g</sub> the code. The<sub>y</sub> are fast and deterministic<sub>,</sub> and the<sub>y</sub> excel at fla<sub>gg</sub>in<sub>g</sub> dan<sub>g</sub>erous code that is <sub>p</sub>resent (e.<sub>g</sub>., eval on user in<sub>p</sub>ut, SQL built b<sub>y</sub> strin<sub>g</sub> formattin<sub>g</sub>), but a flaw that consists of a missin<sub>g</sub> check (absent in<sub>p</sub>ut validation, an omitted salt, no authorization <sub>g</sub>uard) <sub>o</sub>ft<sub>en</sub> l<sub>eaves no pa</sub>tt<sub>ern</sub> f<sub>or a ru</sub>l<sub>e</sub> t<sub>o ma</sub>t<sub>c</sub>h<sub>.</sub>

## 3 Overview

W<sub>e</sub> fi<sub>rs</sub>t d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> t<sub>wo</sub> t<sub>as</sub>k<sub>s</sub> thi<sub>s</sub> <sub>paper</sub> <sub>a</sub>dd<sub>resses,</sub> th<sub>en</sub> id<sub>en</sub>tif<sub>y</sub> th<sub>e</sub> <sub>researc</sub>h <sub>c</sub>h<sub>a</sub>ll<sub>enges,</sub> <sub>an</sub>d fi<sub>na</sub>ll<sub>y</sub> <sub>ou</sub>tli<sub>ne</sub> h<sub>ow</sub> <sub>au</sub>dit <sub>pressure</sub> <sub>a</sub>dd<sub>resses</sub> <sub>eac</sub>h <sub>one.</sub>

## 3.1 Problem Statement

Let $\boldsymbol { S } = ( x , T _ { f } , T _ { s } , C )$ be a task with <sub>p</sub>rom<sub>p</sub>t <sub>�,</sub> functional and securit<sub>y</sub> test suites<sub>,</sub> and CWE classes. A <sub>genera</sub>t<sub>or</sub> <sub>pro</sub>d<sub>uces</sub> � <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>programs.</sub> W<sub>e</sub> <sub>a</sub>dd<sub>ress</sub> t<sub>wo</sub> t<sub>as</sub>k<sub>s:</sub>

Choosing. Given a pool $Q = \{ P _ { 1 } , \ldots , P _ { k } \}$ of candidate <sub>p</sub>ro<sub>g</sub>rams for one task (o<sub>p</sub>tionall<sub>y</sub> restricted to those <sub>p</sub>assin<sub>g</sub> functional tests), select the <sub>p</sub>ro<sub>g</sub>ram most likel<sub>y</sub> to be secure and workin<sub>g</sub>:

$$
\operatorname { p i c k } ( Q ) = \arg \operatorname* { m i n } _ { P \in Q } g ( P ) ,
$$

where <sub>�</sub> is a scorin<sub>g</sub> rule (hi<sub>g</sub>her = more likel<sub>y</sub> insecure).

Locating. Given a single insecure program � with statement rows $\mathcal { R } ( P ) = \{ s _ { 1 } , . . . , s _ { R } \}$ <sub>,</sub> <sub>ran</sub>k <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t<sub>s</sub> b<sub>y</sub> th<sub>e</sub>i<sub>r</sub> lik<sub>e</sub>lih<sub>oo</sub>d <sub>o</sub>f <sub>con</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> <sub>vu</sub>l<sub>nera</sub>bilit<sub>y:</sub>

$$
\operatorname { l o c a t e } ( P ) = \arg \operatorname* { m a x } _ { s \in { \mathcal { R } } ( P ) } g ( s ) .
$$

Both of these compare programs or statements within one coding task, by design.

## 3.2 Challenges

D<sub>e</sub>t<sub>ec</sub>ti<sub>ng</sub> <sub>an</sub>d l<sub>oca</sub>li<sub>z</sub>i<sub>ng</sub> <sub>secur</sub>it<sub>y</sub> fl<sub>aws</sub> i<sub>n</sub> <sub>genera</sub>t<sub>e</sub>d <sub>co</sub>d<sub>e</sub> f<sub>aces</sub> f<sub>our</sub> <sub>researc</sub>h <sub>c</sub>h<sub>a</sub>ll<sub>enges</sub> th<sub>a</sub>t <sub>ex</sub>i<sub>s</sub>ti<sub>ng</sub> h<sub>a</sub>ll<sub>uc</sub>i<sub>na</sub>ti<sub>on-</sub>d<sub>e</sub>t<sub>ec</sub>ti<sub>on an</sub>d <sub>s</sub>t<sub>a</sub>ti<sub>c-ana</sub>l<sub>ys</sub>i<sub>s me</sub>th<sub>o</sub>d<sub>s</sub> d<sub>o no</sub>t <sub>overcome.</sub>

C1: Security hallucinations are written confidently, so the generator’s own distribution carries no feedback. Hallucination detectors exploit model hesitation [7, 23, 25]: token probabilities <sub>sprea</sub>d <sub>across a</sub>lt<sub>erna</sub>ti<sub>ves an</sub>d dif<sub>eren</sub>t <sub>samp</sub>l<sub>es pro</sub>d<sub>uce</sub> dif<sub>eren</sub>t <sub>ou</sub>t<sub>pu</sub>t<sub>s.</sub> S<sub>ecur</sub>it<sub>y</sub> h<sub>a</sub>ll<sub>uc</sub>i<sub>na-</sub> tions show the opposite pattern as 86% of insecure working programs have only insecure siblings across inde<sub>p</sub>endent sam<sub>p</sub>les<sub>,</sub> and <sub>g</sub>eneration uncertaint<sub>y</sub> ranks insecurit<sub>y</sub> at AUROC 0.61 overall ( §6.1). Pro<sub>p</sub>osition 2.2 confirms that confidentl<sub>y</sub> written flaws are invisible to ever<sub>y</sub> function of the <sub>genera</sub>t<sub>or</sub>’<sub>s</sub> t<sub>o</sub>k<sub>en</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub>

C2: The dominant vulnerability pattern is an absence instead of presence. A security hallucination t<sub>yp</sub>icall<sub>y</sub> omits a necessar<sub>y</sub> securit<sub>y</sub> evaluation check rather than insertin<sub>g</sub> a dan<sub>g</sub>erous f<sub>unc</sub>ti<sub>on ca</sub>ll<sub>.</sub> B<sub>ecause</sub> th<sub>e</sub> fl<sub>aw</sub> i<sub>s</sub> i<sub>n non-genera</sub>t<sub>e</sub>d <sub>co</sub>d<sub>e, syn</sub>t<sub>ac</sub>ti<sub>c pa</sub>tt<sub>ern ma</sub>t<sub>c</sub>hi<sub>ng canno</sub>t d<sub>e</sub>t<sub>ec</sub>t it: static anal<sub>y</sub>zers find a CWE-matched issue in onl<sub>y</sub> 15% of insecure <sub>p</sub>ro<sub>g</sub>rams<sub>,</sub> and nearl<sub>y</sub> as often in secure ones (i.e., 12% false <sub>p</sub>ositives). An efective diferentiatin<sub>g</sub> si<sub>g</sub>nal must reason about what should be in the code, not only about what is already there.

C3: Task-level label shifts defeat program-level detection. Whether generated code is insecure d<sub>epen</sub>d<sub>s more on</sub> th<sub>e</sub> t<sub>as</sub>k th<sub>an</sub> th<sub>e mo</sub>d<sub>e</sub>l<sub>.</sub> T<sub>as</sub>k id<sub>en</sub>tit<sub>y a</sub>l<sub>one exp</sub>l<sub>a</sub>i<sub>ns</sub> 57% <sub>o</sub>f th<sub>e var</sub>i<sub>ance</sub> i<sub>n</sub> whether a workin<sub>g</sub> <sub>p</sub>ro<sub>g</sub>ram is insecure<sub>,</sub> whereas <sub>g</sub>enerator identit<sub>y</sub> ex<sub>p</sub>lains 6%. An<sub>y</sub> score a<sub>pp</sub>lied <sub>across</sub> t<sub>as</sub>k<sub>s</sub> i<sub>n</sub>h<sub>er</sub>it<sub>s</sub> th<sub>ese</sub> b<sub>ase-ra</sub>t<sub>e s</sub>hift<sub>s.</sub> Th<sub>e pro</sub>bl<sub>em</sub> i<sub>s wors</sub>t <sub>w</sub>h<sub>en a</sub> t<sub>as</sub>k’<sub>s secur</sub>it<sub>y</sub> t<sub>es</sub>t<sub>s en</sub>f<sub>orce</sub> requirements t<sup>h</sup>e prompt never states. On t<sup>h</sup>ese tas<sup>k</sup>s every co<sup>d</sup>e-<sup>l</sup>eve<sup>l</sup> ju<sup>d</sup>ge, inc<sup>l</sup>u<sup>d</sup>ing a yes/no verdict<sub>,</sub> <sub>p</sub>erforms <sub>p</sub>oorl<sub>y</sub>.

C4: Localization requires token-level attribution without training data. A binary verdict <sub>p</sub>er <sub>p</sub>ro<sub>g</sub>ram cannot isolate the securit<sub>y</sub> flaw. An<sub>y</sub> measure must decom<sub>p</sub>ose to individual tokens and a<sub>gg</sub>re<sub>g</sub>ate to statements. Because the missin<sub>g</sub> securit<sub>y</sub> check is not in the <sub>g</sub>enerated <sub>p</sub>ro<sub>g</sub>ram<sub>,</sub> t<sup>h</sup>e au<sup>d</sup>itor<sup>’</sup>s <sup>d</sup>isagreement appears at t<sup>h</sup>e to<sup>k</sup>ens just a<sup>f</sup>ter t<sup>h</sup>e gap: t<sup>h</sup>is is w<sup>h</sup>ere t<sup>h</sup>e au<sup>d</sup>itor wou<sup>ld</sup> h<sub>ave s</sub>t<sub>ar</sub>t<sub>e</sub>d <sub>wr</sub>iti<sub>ng</sub> th<sub>e c</sub>h<sub>ec</sub>k<sub>,</sub> b<sub>u</sub>t th<sub>e program con</sub>ti<sub>nues w</sub>ith <sub>o</sub>th<sub>er co</sub>d<sub>e</sub> i<sub>ns</sub>t<sub>ea</sub>d<sub>.</sub> Th<sub>e me</sub>th<sub>o</sub>d <sub>mus</sub>t th<sub>ere</sub>f<sub>ore</sub> <sub>a</sub>tt<sub>r</sub>ib<sub>u</sub>t<sub>e</sub> <sub>pressure</sub> t<sub>o</sub> <sub>a</sub> li<sub>ne</sub> th<sub>a</sub>t f<sub>o</sub>ll<sub>ows</sub> th<sub>e</sub> <sub>a</sub>b<sub>sen</sub>t <sub>co</sub>d<sub>e,</sub> <sub>no</sub>t t<sub>o</sub> th<sub>e</sub> <sub>a</sub>b<sub>sen</sub>t <sub>co</sub>d<sub>e</sub> it<sub>se</sub>lf<sub>.</sub>

## 3.3 Our Approach

W<sub>e</sub> <sub>now</sub> b<sub>r</sub>i<sub>e</sub>fl<sub>y</sub> <sub>a</sub>dd<sub>ress</sub> th<sub>e</sub> <sub>c</sub>h<sub>a</sub>ll<sub>enges</sub> <sub>a</sub>b<sub>ove.</sub> W<sub>e</sub> di<sub>scuss</sub> th<sub>e</sub> d<sub>e</sub>t<sub>a</sub>il<sub>e</sub>d <sub>me</sub>th<sub>o</sub>d i<sub>n</sub> $\ S 4$

Addressing C1: Read a model’s distribution under a review instruction. The securit knowl edge needed to identify a flaw is present in a capable language model but remains inactive during <sub>co</sub>d<sub>e ge</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>.</sub> F<sub>u</sub>nd<sub>a</sub>m<sub>e</sub>nt<sub>a</sub>ll<sub>y,</sub> th<sub>e</sub> m<sub>o</sub>d<sub>e</sub>l <sub>ca</sub>n n<sub>o</sub>t r<sub>ev</sub>i<sub>ew</sub> th<sub>e co</sub>d<sub>e w</sub>hil<sub>e w</sub>ritin<sub>g</sub> it<sub>.</sub> A<sub>u</sub>dit <sub>p</sub>r<sub>essu</sub>r<sub>e</sub> activates it: a se<sub>p</sub>arate audit reads the finished <sub>p</sub>ro<sub>g</sub>ram under a securit<sub>y</sub>-review <sub>p</sub>rom<sub>p</sub>t and scores b<sub>ase</sub>d <sub>on</sub> h<sub>ow</sub> <sub>w</sub>illi<sub>ng</sub> it i<sub>s</sub> t<sub>o</sub> <sub>copy</sub> <sub>eac</sub>h t<sub>o</sub>k<sub>en.</sub> Wh<sub>ere</sub> th<sub>e</sub> <sub>au</sub>dit<sub>or</sub> <sub>wou</sub>ld h<sub>ave</sub> <sub>wr</sub>itt<sub>en</sub> dif<sub>eren</sub>t code<sub>,</sub> its co<sub>py</sub> <sub>p</sub>robabilit<sub>y</sub> dro<sub>p</sub>s<sub>,</sub> revealin<sub>g</sub> the flaw. This also o<sub>p</sub>ens the <sub>p</sub>ossibilit<sub>y</sub> for the <sub>g</sub>enerator <sub>mo</sub>d<sub>e</sub>l t<sub>o</sub> <sub>au</sub>dit it<sub>se</sub>lf <sub>or</sub> f<sub>or</sub> <sub>a</sub> <sub>separa</sub>t<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l t<sub>o</sub> <sub>au</sub>dit it<sub>.</sub> Si<sub>nce</sub> th<sub>e</sub> <sub>au</sub>dit<sub>or</sub> <sub>rea</sub>d<sub>s</sub> fi<sub>n</sub>i<sub>s</sub>h<sub>e</sub>d <sub>co</sub>d<sub>e,</sub> <sub>no</sub>t th<sub>e</sub> <sub>g</sub>enerator’s writin<sub>g</sub>-time distribution, C1’s entro<sub>py</sub> bound (Pro<sub>p</sub>. 2.2) does not a<sub>pp</sub>l<sub>y</sub> to the auditor.

Addressing C2: Contrastive framing isolates security from style. A security-review prompt alone does not se<sub>p</sub>arate securit<sub>y</sub> concerns from <sub>g</sub>eneral code-<sub>q</sub>ualit<sub>y</sub> concerns. Audit <sub>p</sub>ressure subtracts the score under a style-review prompt from the score under a security-review prompt, t<sub>o</sub>k<sub>en</sub> b<sub>y</sub> t<sub>o</sub>k<sub>en,</sub> i<sub>so</sub>l<sub>a</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>secur</sub>it<sub>y-spec</sub>ifi<sub>c</sub> <sub>re</sub>l<sub>uc</sub>t<sub>ance.</sub> C<sub>ruc</sub>i<sub>a</sub>ll<sub>y,</sub> thi<sub>s</sub> d<sub>e</sub>t<sub>ec</sub>t<sub>s</sub> <sub>a</sub>b<sub>sences:</sub> <sub>w</sub>h<sub>ere</sub> th<sub>e</sub> <sub>secur</sub>it<sub>y</sub> <sub>rev</sub>i<sub>ewer</sub> <sub>wou</sub>ld h<sub>ave</sub> i<sub>nser</sub>t<sub>e</sub>d <sub>a</sub> <sub>c</sub>h<sub>ec</sub>k b<sub>e</sub>f<sub>ore</sub> <sub>a</sub> li<sub>ne,</sub> th<sub>ose</sub> t<sub>o</sub>k<sub>ens</sub> <sub>rece</sub>i<sub>ve</sub> hi<sub>g</sub>h <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> under the st<sub>y</sub>le frame and low <sub>p</sub>robabilit<sub>y</sub> under the securit<sub>y</sub> frame<sub>,</sub> <sub>p</sub>roducin<sub>g</sub> a <sub>p</sub>ositive <sub>p</sub>ressure measure even thou<sub>g</sub>h the vulnerabilit<sub>y</sub> is based on missin<sub>g</sub> artifacts in the <sub>p</sub>ro<sub>g</sub>ram (§4.4).

Addressing C3: Within-task ranking instead of cross-task detection. We frame the problem as within-task com<sub>p</sub>arisons. For choosin<sub>g</sub> a secure <sub>p</sub>ro<sub>g</sub>ram<sub>,</sub> all candidates in a <sub>p</sub>ool share the <sub>same</sub> <sub>promp</sub>t<sub>,</sub> <sub>so</sub> dif<sub>erences</sub> i<sub>n</sub> h<sub>ow</sub> <sub>o</sub>ft<sub>en</sub> <sub>a</sub> t<sub>as</sub>k <sub>pro</sub>d<sub>uces</sub> i<sub>nsecure</sub> <sub>co</sub>d<sub>e</sub> d<sub>o</sub> <sub>no</sub>t <sub>a</sub>f<sub>ec</sub>t th<sub>e</sub> <sub>ran</sub>ki<sub>ng</sub> (Lemma 4.4). For locatin<sub>g</sub> vulnerable statements, all statements belon<sub>g</sub> to the same <sub>p</sub>ro<sub>g</sub>ram, so <sub>p</sub>ro<sub>g</sub>ram-level biases cancel out. This invariance is wh<sub>y</sub> audit <sub>p</sub>ressure succeeds at choosin<sub>g</sub> and locatin<sub>g</sub> securit<sub>y</sub> hallucinations even without task s<sub>p</sub>ecific securit<sub>y</sub> tests (§6.5).

Addressing C4: Exact per-token decomposition via the chain rule. The log-probability dif<sub>erence</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> t<sub>wo</sub> <sub>rev</sub>i<sub>ew</sub> f<sub>rames</sub> d<sub>ecomposes</sub> <sub>exac</sub>tl<sub>y</sub> i<sub>n</sub>t<sub>o</sub> <sub>one</sub> t<sub>erm</sub> <sub>per</sub> t<sub>o</sub>k<sub>en</sub> <sub>w</sub>ith<sub>ou</sub>t an<sub>y</sub> a<sub>pp</sub>roximation<sub>,</sub> <sub>g</sub>radient or trainin<sub>g</sub> re<sub>q</sub>uirement. Each token’s contribution is assi<sub>g</sub>ned to the corresponding code statement. For a missing check, this places the pressure on the line after the <sub>g</sub>a<sub>p</sub>. Summin<sub>g</sub> over a statement’s tokens <sub>g</sub>ives the statement score—the hi<sub>g</sub>hest-scorin<sub>g</sub> statement is the localization out<sub>p</sub>ut. Under a com<sub>p</sub>etin<sub>g</sub>-edits model<sub>,</sub> hi<sub>g</sub>her statement scores corres<sub>p</sub>ond to hi<sub>g</sub>her <sub>p</sub>robabilities su<sub>gg</sub>estin<sub>g</sub> the auditor’s fix would be<sub>g</sub>in there (Theorem 4.1, §4.3).

## 4 Detailed Design of SPLASH

Fi<sub>g</sub>ure 2 shows the overview of SPLASH. We now describe each com<sub>p</sub>onent in detail.

## 4.1 Task Corpus, Generation, and Extraction

A t<sub>as</sub>k $S = ( x , \ell , \sigma , T _ { f } , T _ { s } , C )$ <sup>consists</sup> <sup>of</sup> <sup>a</sup> <sup>natural-lan</sup>g<sup>ua</sup>g<sup>e</sup> p<sup>rom</sup>p<sup>t</sup> �, <sup>a</sup> p<sup>ro</sup>g<sup>rammin</sup>g <sup>lan</sup>g<sup>ua</sup>g<sup>e</sup> <sup>ℓ</sup>, <sup>an</sup> o<sub>p</sub>tional function si<sub>g</sub>nature $\sigma ,$ functional and securit<sub>y</sub> test suites $T _ { f }$ <sub>an</sub>d $T _ { s }$ , and the CWE classes C the task exercises. The <sub>p</sub>rom<sub>p</sub>t never mentions a CWE or an<sub>y p</sub>articular securit<sub>y</sub> re<sub>q</sub>uirement. We curated 899 tasks with executable securit<sub>y</sub> tests that come from SecCodePLT [28], CWEval [33], and 16 tasks we authored for under-re<sub>p</sub>resented CWEs (❶); see $\ S 5$ f<sub>or</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> <sub>an</sub>d <sub>coun</sub>t<sub>s.</sub> W<sub>e</sub> b<sub>u</sub>ild <sub>our cus</sub>t<sub>om a</sub>d<sub>ap</sub>t<sub>er</sub> th<sub>a</sub>t <sub>can parse</sub> d<sub>a</sub>t<sub>a</sub> f<sub>rom any o</sub>f th<sub>e a</sub>b<sub>ove-men</sub>ti<sub>one</sub>d b<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> i<sub>n</sub>t<sub>o a</sub> <sub>un</sub>if<sub>orm</sub> t<sub>as</sub>k <sub>sc</sub>h<sub>ema.</sub>

For each task, � <sub>p</sub>ro<sub>g</sub>rams are sam<sub>p</sub>led from a <sub>g</sub>enerator � (❷). As model res<sub>p</sub>onses often contain multi<sub>p</sub>le code blocks accom<sub>p</sub>anied b<sub>y</sub> natural lan<sub>g</sub>ua<sub>g</sub>e <sub>p</sub>rose<sub>,</sub> we a<sub>pp</sub>l<sub>y</sub> a code extraction strate<sub>gy</sub> b<sub>ase</sub>d <sub>on</sub> <sub>scor</sub>i<sub>ng.</sub> S<sub>cores</sub> <sub>are</sub> <sub>ass</sub>i<sub>gne</sub>d b<sub>ase</sub>d <sub>on</sub> th<sub>e</sub>i<sub>r</sub> lik<sub>e</sub>lih<sub>oo</sub>d <sub>o</sub>f b<sub>e</sub>i<sub>ng</sub> <sub>a</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e,</sub> <sub>va</sub>lid <sub>program</sub> (e.g., for Python, blocks tagged with \`\`\`���ℎ��...\`\`\`, contain keywords like “import", “def", etc.). Th<sub>e</sub> t<sub>op-scor</sub>i<sub>ng</sub> <sub>co</sub>d<sub>e</sub> bl<sub>oc</sub>k i<sub>s</sub> <sub>ex</sub>t<sub>rac</sub>t<sub>e</sub>d f<sub>rom</sub> <sub>eac</sub>h <sub>response.</sub> Sibli<sub>ng</sub> bl<sub>oc</sub>k<sub>s</sub> <sub>are</sub> <sub>merge</sub>d <sub>on</sub>l<sub>y</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> b<sub>es</sub>t bl<sub>oc</sub>k d<sub>oes no</sub>t <sub>parse.</sub> R<sub>eason</sub>i<sub>ng</sub> t<sub>races</sub> f<sub>rom</sub> thi<sub>n</sub>ki<sub>ng mo</sub>d<sub>e</sub>l<sub>s are</sub> di<sub>scar</sub>d<sub>e</sub>d<sub>.</sub> A<sub>ny mo</sub>d<sub>e</sub>l <sub>can</sub> serve as the <sub>g</sub>enerator<sub>,</sub> includin<sub>g</sub> closed-API models and a<sub>g</sub>entic <sub>p</sub>i<sub>p</sub>elines<sub>,</sub> because audit <sub>p</sub>ressure <sub>requ</sub>i<sub>res on</sub>l<sub>y</sub> th<sub>e</sub> fi<sub>n</sub>i<sub>s</sub>h<sub>e</sub>d <sub>source co</sub>d<sub>e.</sub> Th<sub>e genera</sub>t<sub>or</sub>’<sub>s per-</sub>t<sub>o</sub>k<sub>en</sub> l<sub>og-pro</sub>b<sub>a</sub>biliti<sub>es, w</sub>h<sub>en ava</sub>il<sub>a</sub>bl<sub>e,</sub> serve as a baseline (<sub>g</sub>eneration uncertaint<sub>y</sub>) but are not used b<sub>y</sub> audit <sub>p</sub>ressure.

![](images/9608204fa3efca0ac0d77d823d043bc67a1b8b497a4d642371b2d2d444ea7228.jpg)  
Fig. 2. Overview of SPLASH. Tasks with executable security tests enter the pipeline (<sup>❶</sup>). Each program’s topscoring fenced code block is extracted (<sup>❷</sup>). Comments are blanked, and an auditor model scores every token under a security-review and a style-review prompt; the per-token diference (audit pressure) is aggregated to statements and to the whole program (<sup>❸</sup>). An independent oracle provides ground-truth labels used only for evaluation (<sup>❹</sup>), and five research questions measure efectiveness (<sup>❺</sup>).

## 4.2 Audit Pressure: Scoring Programs Under Review

Audit <sub>p</sub>ressure is the core contribution of SPLASH, com<sub>p</sub>uted in ste<sub>p</sub> ❸. The core idea is sim<sub>p</sub>le: we <sub>s</sub>h<sub>ow</sub> th<sub>e</sub> fi<sub>n</sub>i<sub>s</sub>h<sub>e</sub>d <sub>program</sub> t<sub>o an au</sub>dit<sub>or mo</sub>d<sub>e</sub>l <sub>un</sub>d<sub>er</sub> t<sub>wo rev</sub>i<sub>ew</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>ons an</sub>d <sub>measure</sub> h<sub>ow</sub> <sub>muc</sub>h l<sub>ess w</sub>illi<sub>ng</sub> th<sub>e secur</sub>it<sub>y rev</sub>i<sub>ewer</sub> i<sub>s</sub> t<sub>o repro</sub>d<sub>uce eac</sub>h t<sub>o</sub>k<sub>en.</sub>

Comment blanking. Before scoring, all comments are removed from program � to produce �<sup>˜</sup>, <sub>preserv</sub>i<sub>ng</sub> <sub>every</sub> li<sub>ne</sub> b<sub>rea</sub>k <sub>so</sub> th<sub>a</sub>t li<sub>ne</sub> <sub>num</sub>b<sub>ers</sub> <sub>an</sub>d <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t <sub>rows</sub> <sub>rema</sub>i<sub>n</sub> <sub>unc</sub>h<sub>ange</sub>d<sub>.</sub> With<sub>ou</sub>t this ste<sub>p</sub>, a comment such as # prevent path traversal would let the auditor read the develo<sub>p</sub>er’s <sub>c</sub>l<sub>a</sub>i<sub>m ra</sub>th<sub>er</sub> th<sub>an reason</sub> f<sub>rom</sub> th<sub>e co</sub>d<sub>e.</sub>

Frames. A frame � is a chat prompt $\pi _ { f } ( x , \tilde { P } )$ th<sub>a</sub>t <sub>presen</sub>t<sub>s</sub> th<sub>e</sub> t<sub>as</sub>k <sub>an</sub>d th<sub>e</sub> <sub>commen</sub>t<sub>-</sub>bl<sub>an</sub>k<sub>e</sub>d <sub>co</sub>d<sub>e,</sub> <sub>pre-</sub>filli<sub>ng</sub> th<sub>e</sub> <sub>ass</sub>i<sub>s</sub>t<sub>an</sub>t t<sub>urn</sub> <sub>w</sub>ith <sub>an</sub> <sub>open</sub>i<sub>ng</sub> <sub>co</sub>d<sub>e</sub> f<sub>ence.</sub> W<sub>e</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>ree</sub> f<sub>rames,</sub> <sub>none</sub> <sub>o</sub>f <sub>w</sub>hi<sub>c</sub>h <sub>names</sub> th<sub>e</sub> t<sub>as</sub>k’<sub>s</sub> CWE<sub>:</sub>

sec: “Review this implementation for security vulnerabilities, as an attacker would. If nothing needs fixing, output it unchanged. Output only the code.”

neu: the same with “. . . for readability and style” and “revised”.

verdict: “Does this implementation contain a security vulnerability? Answer with Yes or No only.” B<sub>ecause</sub> $\tilde { P }$ is alread<sub>y</sub> visible in the <sub>p</sub>rom<sub>p</sub>t<sub>,</sub> out<sub>p</sub>uttin<sub>g</sub> it unchan<sub>g</sub>ed is the default: the auditor co<sub>p</sub>ies <sub>un</sub>l<sub>ess</sub> it <sub>sees a reason</sub> t<sub>o e</sub>dit<sub>.</sub>

Scoring and audit pressure. Each frame is scored in a single forward pass with teacher forcing. The re<sub>p</sub>l<sub>y</sub> is $r = \tilde { P } \parallel \mathsf { \bar { \backslash } } \mathsf { n } ^ { \cdot \cdot \cdot }$ (the comment-blanked code followed b<sub>y</sub> the closin<sub>g</sub> fence), tokenized into $c _ { 1 } , \ldots , c _ { n }$ . For e<sub>v</sub>er<sub>y</sub> token <sub>p</sub>osition �:

$$
\ell _ { f } ( t ) = \log p _ { A } \big ( c _ { t } \mid \pi _ { f } ( x , \tilde { P } ) , c _ { < t } \big ) .\tag{3}
$$

All frames score the same token ids, so the log-probabilities are directly comparable. The audit pressure of token � is

$$
d _ { t } = \ell _ { \mathsf { n e u } } ( t ) - \ell _ { \mathsf { s e c } } ( t ) .\tag{4}
$$

A <sub>pos</sub>iti<sub>ve</sub> $d _ { t }$ means the securit<sub>y</sub> reviewer is less willin<sub>g</sub> to co<sub>py</sub> this token than the st<sub>y</sub>le reviewer. Th<sub>e</sub> <sub>con</sub>t<sub>ras</sub>t <sub>removes</sub> <sub>w</sub>h<sub>a</sub>t <sub>ma</sub>k<sub>es</sub> <sub>a</sub> t<sub>o</sub>k<sub>en</sub> h<sub>ar</sub>d t<sub>o</sub> <sub>copy</sub> <sub>un</sub>d<sub>er</sub> <sub>any</sub> <sub>rev</sub>i<sub>ew</sub> <sub>an</sub>d i<sub>so</sub>l<sub>a</sub>t<sub>es</sub> th<sub>e</sub> <sub>secur</sub>it<sub>y-</sub> s<sub>p</sub>ecific reluctance. Summed over the <sub>p</sub>ro<sub>g</sub>ram<sub>,</sub> audit <sub>p</sub>ressure e<sub>q</sub>uals a lo<sub>g</sub> Ba<sub>y</sub>es factor b<sub>y</sub> the <sub>c</sub>h<sub>a</sub>i<sub>n ru</sub>l<sub>e:</sub>

$$
\sum _ { t = 1 } ^ { n } d _ { t } = \log p _ { A } ( \mathbf { c } \mid \pi _ { \mathsf { n e u } } ) - \log p _ { A } ( \mathbf { c } \mid \pi _ { \mathsf { s e c } } ) .\tag{5}
$$

Two <sub>p</sub>asses <sub>p</sub>er <sub>p</sub>ro<sub>g</sub>ram sufice. One the other hand, the verdict frame <sub>p</sub>roduces a binar<sub>y</sub> readout:

$$
v ( P ) = \frac { \mathcal { P } _ { A } ( \mathrm { Y e s } \mid \pi _ { v \mathrm { e r d i c t } } ) } { \mathcal { P } _ { A } ( \mathrm { Y e s } \mid \pi _ { \mathrm { v e r d i c t } } ) + p _ { A } ( \mathrm { N o } \mid \pi _ { \mathrm { v e r d i c t } } ) } ,\tag{6}
$$

<sub>measur</sub>i<sub>ng</sub> th<sub>e au</sub>dit<sub>or</sub>’<sub>s</sub> di<sub>rec</sub>t b<sub>e</sub>li<sub>e</sub>f th<sub>a</sub>t th<sub>e program</sub> i<sub>s vu</sub>l<sub>nera</sub>bl<sub>e.</sub> B<sub>o</sub>th <sub>rea</sub>d<sub>ou</sub>t<sub>s come</sub> f<sub>rom</sub> th<sub>e</sub> <sub>same mo</sub>d<sub>e</sub>l <sub>on</sub> th<sub>e same co</sub>d<sub>e;</sub> th<sub>e</sub>i<sub>r re</sub>l<sub>a</sub>ti<sub>ons</sub>hi<sub>p</sub> i<sub>s exp</sub>l<sub>ore</sub>d i<sub>n</sub> $\ S 6 . 2$

## 4.3 The Reviewer as an Edit Process

The lo<sub>g</sub>-<sub>p</sub>robabilities in E<sub>q</sub>. (3) have a natural inter<sub>p</sub>retation as edit hazards. Let $C \sim p _ { A } ( \cdot \mid \pi _ { f } )$ b<sub>e</sub> the re<sub>p</sub>l<sub>y</sub> the auditor would actuall<sub>y</sub> write. Its diver<sub>g</sub>ence time is $\tau _ { f } = \operatorname* { m i n } \{ t : C _ { t } \neq c _ { t } \}$ <sub>,</sub> <sub>an</sub>d it<sub>s</sub> edit hazard at <sub>p</sub>osition � is $h _ { f } ( t ) = \operatorname* { P r } [ \tau _ { f } = t \mid \tau _ { f } \geq t ]$ . The securit<sub>y</sub> reviewer ma<sub>y</sub> start an edit for the same reasons as the st<sub>y</sub>le reviewer (formattin<sub>g</sub>, namin<sub>g</sub>) or because it sees a vulnerabilit<sub>y</sub>. We <sub>mo</sub>d<sub>e</sub>l th<sub>e</sub> <sub>secon</sub>d <sub>as</sub> <sub>an</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>c</sub>l<sub>oc</sub>k<sub>.</sub>

Assumption 1 (Competing edits). There exist security-edit hazards $\lambda _ { t } \in [ 0 , 1 )$ <sub>suc</sub>h th<sub>a</sub>t

$$
1 - h _ { \mathsf { s e c } } ( t ) = { \big ( } 1 - h _ { \mathsf { n e u } } ( t ) { \big ) } { \big ( } 1 - \lambda _ { t } { \big ) } , \qquad t = 1 , \ldots , n .\tag{7}
$$

Theorem 4.1 (Audit pressure is the security-edit hazard). Under Assumption 1, for every token �, every statement �, and the program $P { : }$

$$
d _ { t } = - \log ( 1 - \lambda _ { t } ) ~ \geq ~ 0 ,\tag{8}
$$

$$
\begin{array} { r } { \mathrm { A P } ( s ) = \sum _ { t \in B ( s ) } d _ { t } = - \log ( 1 - \pi _ { s } ) , } \end{array}\tag{9}
$$

$$
\begin{array} { r } { \sum _ { t = 1 } ^ { n } d _ { t } = - \log \rho ( P ) , } \end{array}\tag{10}
$$

where $\pi _ { s } = \operatorname* { P r } [ \tau ^ { \lambda } \in B ( s ) \mid \tau ^ { \lambda } \ge$ min $B ( s ) ]$ is the probability that the security fix begins in statement �, and $\rho ( P ) = \mathrm { P r } [ \tau ^ { \lambda } = \infty ]$ is the probability that nothing needs fixing.

Proof sketch<sub>.</sub> Sin<sub>ce</sub> $\left\{ C _ { < t } = c _ { < t } \right\} = \left\{ \tau _ { f } \geq t \right\}$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub> $p _ { A } ( c _ { t } \mid \pi _ { f } , c _ { < t } ) = 1 - h _ { f } ( t )$ , so $\ell _ { f } ( t ) =$ $\log ( 1 { - } h _ { f } ( t ) )$ . Combined <sub>w</sub>ith E<sub>q</sub>. $( 7 ) , d _ { t } = \log \left( 1 - h _ { \mathrm { n e u } } ( t ) \right) - \log \left[ \left( 1 - h _ { \mathrm { n e u } } ( t ) \right) \left( 1 - \lambda _ { t } \right) \right] = - \log \left( 1 - \lambda _ { t } \right)$ Summin<sub>g</sub> over a statement’s token block and over all tokens <sub>g</sub>ives (9) and (10) b<sub>y</sub> telesco<sub>p</sub>in<sub>g</sub>. □

Clipping. When $d _ { t } ~ < ~ 0$ <sub>,</sub> th<sub>e</sub> i<sub>mp</sub>li<sub>e</sub>d $\lambda _ { t }$ i<sub>s</sub> n<sub>ega</sub>ti<sub>ve,</sub> <sub>v</sub>i<sub>o</sub>l<sub>a</sub>tin<sub>g</sub> A<sub>ssu</sub>m<sub>p</sub>ti<sub>o</sub>n 1<sub>.</sub> R<sub>ep</sub>l<sub>ac</sub>in<sub>g</sub> $d _ { t }$ <sub>w</sub>ith $\operatorname* { m a x } ( d _ { t } , 0 )$ sets $\lambda _ { t }$ t<sub>o</sub> <sub>zero,</sub> th<sub>e</sub> <sub>sma</sub>ll<sub>es</sub>t h<sub>azar</sub>d th<sub>e</sub> <sub>assump</sub>ti<sub>on</sub> <sub>a</sub>ll<sub>ows,</sub> <sub>zero</sub>i<sub>ng</sub> <sub>ou</sub>t <sub>s</sub>t<sub>y</sub>l<sub>e-on</sub>l<sub>y</sub> di<sub>s-</sub> a<sub>g</sub>reements. All scores below use this <sub>p</sub>ositive <sub>p</sub>art.

## 4.4 Statement and Program Scores

Attribution. A program’s statement rows are its leaf AST statements (Python ast, $\mathrm { C } / \mathrm { C } { + + }$ <sub>v</sub>i<sub>a</sub> tree-sitter), with com<sub>p</sub>ound-statement headers s<sub>p</sub>lit at clause boundaries so that ever<sub>y</sub> source li<sub>ne</sub> b<sub>e</sub>l<sub>ongs</sub> t<sub>o exac</sub>tl<sub>y one row.</sub> T<sub>o</sub>k<sub>en</sub> � i<sub>s a</sub>tt<sub>r</sub>ib<sub>u</sub>t<sub>e</sub>d t<sub>o</sub> th<sub>e row own</sub>i<sub>ng</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>non-w</sub>hit<sub>espace</sub> <sub>c</sub>h<sub>arac</sub>t<sub>er o</sub>f $\tilde { P }$ <sub>a</sub>t <sub>or a</sub>ft<sub>er</sub> th<sub>e</sub> t<sub>o</sub>k<sub>en</sub>’<sub>s s</sub>t<sub>ar</sub>t<sub>.</sub> Thi<sub>s ru</sub>l<sub>e ma</sub>tt<sub>ers as</sub> th<sub>e pressure o</sub>f <sub>a m</sub>i<sub>ss</sub>i<sub>ng c</sub>h<sub>ec</sub>k <sub>appears on</sub> th<sub>e w</sub>hit<sub>espace or</sub> fi<sub>rs</sub>t t<sub>o</sub>k<sub>en w</sub>h<sub>ere</sub> th<sub>e c</sub>h<sub>ec</sub>k <sub>wou</sub>ld h<sub>ave</sub> b<sub>een</sub> i<sub>nser</sub>t<sub>e</sub>d<sub>, an</sub>d th<sub>e</sub> attribution carries that <sub>p</sub>ressure to the correct line.   
Scores.

Definition 4.2 (Statement and program scores).

$$
\mathrm { A P } ( s ) = \sum _ { t \in B ( s ) } \operatorname* { m a x } ( d _ { t } , 0 ) , \qquad \mathrm { A P } _ { \operatorname* { m a x } } ( P ) = \operatorname* { m a x } _ { s \in \mathcal { R } ( P ) } \mathrm { A P } ( s ) , \qquad \mathrm { A P } _ { \Sigma } ( P ) = \sum _ { t = 1 } ^ { n } \operatorname* { m a x } ( d _ { t } , 0 ) ,\tag{11}
$$

<sub>w</sub>h<sub>ere</sub> $B ( s )$ i<sub>s</sub> th<sub>e</sub> <sub>se</sub>t <sub>o</sub>f t<sub>o</sub>k<sub>en</sub> <sub>pos</sub>iti<sub>ons</sub> <sub>a</sub>tt<sub>r</sub>ib<sub>u</sub>t<sub>e</sub>d t<sub>o</sub> <sub>row</sub> <sub>�</sub> <sub>an</sub>d $\mathcal { R } ( P )$ i<sub>s</sub> th<sub>e</sub> <sub>se</sub>t <sub>o</sub>f <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t <sub>rows.</sub>

Here, AP(�) is the locatin<sub>g</sub> score: the statement with the lar<sub>g</sub>est audit <sub>p</sub>ressure is the most likel<sub>y</sub> site of a securit<sub>y</sub> fix (Theorem 4.1, E<sub>q</sub>. 9). $\mathrm { A P } _ { \operatorname* { m a x } }$ i<sub>s</sub> th<sub>e c</sub>h<sub>oos</sub>i<sub>ng score:</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e w</sub>h<sub>ose mos</sub>t <sub>con</sub>t<sub>es</sub>t<sub>e</sub>d <sub>s</sub>t<sub>a</sub>t<sub>emen</sub>t <sub>carr</sub>i<sub>es</sub> th<sub>e</sub> l<sub>eas</sub>t <sub>pressure</sub> i<sub>s</sub> th<sub>e one mos</sub>t lik<sub>e</sub>l<sub>y</sub> t<sub>o</sub> b<sub>e cop</sub>i<sub>e</sub>d <sub>unc</sub>h<sub>ange</sub>d b<sub>y</sub> th<sub>e</sub> <sub>secu</sub>rit<sub>y</sub> r<sub>ev</sub>i<sub>ewe</sub>r<sub>.</sub>

Why $\mathrm { A P } _ { \operatorname* { m a x } }$ rather than $\mathrm { A P } _ { \Sigma }$ . Under i.i.d. measurement noise $\varepsilon _ { t } \sim { \cal N } ( 0 , \sigma ^ { 2 } )$ <sub>,</sub> th<sub>e no</sub>i<sub>se</sub> fl<sub>oor o</sub>f $\mathrm { A P _ { \Sigma } }$ <sup>on a secure</sup> p<sup>ro</sup>g<sup>ram</sup> g<sup>rows as</sup> $\sigma n / { \sqrt { 2 \pi } } ,$ , <sub>p</sub>enalizin<sub>g</sub> lon<sub>g</sub>er <sub>p</sub>ro<sub>g</sub>rams (which are t<sub>yp</sub>icall<sub>y</sub> the secure ones, since the<sub>y</sub> add checks and error handlin<sub>g</sub>). $\mathrm { A P } _ { \operatorname* { m a x } } \mathrm { \tilde { \it s } }$ <sub>no</sub>i<sub>se</sub> fl<sub>oor</sub> d<sub>epen</sub>d<sub>s on one s</sub>t<sub>a</sub>t<sub>emen</sub>t’<sub>s</sub> l<sub>eng</sub>th <sub>� an</sub>d <sub>on</sub>l<sub>y</sub> $O ( { \sqrt { \log R } } )$ <sub>on</sub> th<sub>e num</sub>b<sub>er o</sub>f <sub>rows</sub> $R { : }$

Proposition 4.3 (Length bias of summed pressure). Let $\hat { d } _ { t } = d _ { t } + \varepsilon _ { t }$ with $\varepsilon _ { t } \ i . i . d . \ N ( 0 , \sigma ^ { 2 } )$ , and let � be a secure program $( d _ { t } = 0 )$ with � tokens, rows ofat most � tokens, and $| { \mathcal { R } } | = R$ rows. Then

$$
\mathbb { E } [ \widehat { \mathrm { A P } } _ { \Sigma } ] = \frac { \sigma } { \sqrt { 2 \pi } } n , \qquad \mathbb { E } [ \widehat { \mathrm { A P } } _ { \mathrm { m a x } } ] \leq \frac { \sigma } { \sqrt { 2 \pi } } m + \sigma \sqrt { 2 m \log R } .\tag{12}
$$

## 4.5 Choosing and Locating

A pool � is a set of candidate programs for one task from one generator, optionally restricted to those <sub>p</sub>assin<sub>g</sub> functional tests (❹). A rule <sub>�</sub> assi<sub>g</sub>ns each <sub>p</sub>ro<sub>g</sub>ram a score (hi<sub>g</sub>her means more likel<sub>y</sub> insecure). Its <sub>p</sub>ick is ar<sub>g</sub> min $\scriptstyle { \mathsf { i } } _ { P \in Q } g ( P )$ <sub>.</sub> Ch<sub>oos</sub>i<sub>ng</sub> d<sub>epen</sub>d<sub>s on</sub>l<sub>y on</sub> th<sub>e w</sub>ithi<sub>n-</sub>t<sub>as</sub>k <sub>or</sub>d<sub>er o</sub>f <sub>scores:</sub>

Lemma 4.4 (Choosing sees only within-task order). Let $\phi _ { x }$ be strictly increasing for every task � and define $g ^ { \prime } ( P ) = \phi _ { x ( P ) } ( g ( P ) )$ . Then pic $\mathfrak { c } _ { g ^ { \prime } } ( Q ) = \mathrm { p i c k } _ { g } ( Q )$ for every pool, so the secure-pick rate and within-task AUROC are unchanged.

This invariance is wh<sub>y</sub> SPLASH ranks within a task. Tasks whose securit<sub>y</sub> tests check an unstated <sup>l</sup>imit s<sup>h</sup>i<sup>f</sup>t every co<sup>d</sup>e-<sup>l</sup>eve<sup>l</sup> ju<sup>d</sup>ge in t<sup>h</sup>e same <sup>d</sup>irection <sup>f</sup>or a<sup>ll</sup> o<sup>f</sup> t<sup>h</sup>at tas<sup>k’</sup>s can<sup>d</sup>i<sup>d</sup>ates. Suc<sup>h</sup> s<sup>h</sup>i<sup>f</sup>ts <sub>c</sub>h<sub>ange</sub> <sub>scores</sub> <sub>across</sub> t<sub>as</sub>k<sub>s</sub> b<sub>u</sub>t <sub>canno</sub>t <sub>move</sub> <sub>a</sub> <sub>p</sub>i<sub>c</sub>k<sub>,</sub> b<sub>ecause</sub> th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>es</sub> <sub>s</sub>h<sub>are</sub> th<sub>e</sub> t<sub>as</sub>k<sub>.</sub>

Th<sub>e w</sub>ithin-t<sub>as</sub>k AUROC b<sub>ou</sub>nd<sub>s</sub> h<sub>ow o</sub>ft<sub>e</sub>n <sub>a p</sub>i<sub>c</sub>k <sub>succee</sub>d<sub>s</sub>:

Theorem 4.5 (Secure picks from within-task ranking). Fix a task and let scores within the secure and insecure groups each be i.i.d., continuous, and independent ofeach other, with within-task AUROC �. For a pool with $g \geq 1$ good and $b \geq 1$ bad candidates,

$$
A ^ { b } \ \leq \ \operatorname* { P r } [ \mathit { p i c k i s g o o d } ] \ \leq \ 1 - ( 1 - A ) ^ { g } ,\tag{13}
$$

with equality when $g = b = 1$

Locatin<sub>g</sub> ranks the statements of a sin<sub>g</sub>le insecure <sub>p</sub>ro<sub>g</sub>ram b<sub>y</sub> $\operatorname { A P } ( s )$ <sub>, w</sub>hi<sub>c</sub>h b<sub>y</sub> Th<sub>eorem</sub> 4<sub>.</sub>1 i<sub>s</sub> e<sub>q</sub>uivalent to rankin<sub>g</sub> b<sub>y</sub> $\pi _ { s } .$ <sub>,</sub> the <sub>p</sub>robabilit<sub>y</sub> that the securit<sub>y</sub> fix be<sub>g</sub>ins in statement <sub>�</sub>. We measure localization with hit@� (§5).

## 4.6 Ground Truth: Labels Independent of the Method

L<sub>a</sub>b<sub>e</sub>l<sub>s</sub> <sub>are</sub> <sub>cons</sub>t<sub>ruc</sub>t<sub>e</sub>d i<sub>n</sub> <sub>s</sub>t<sub>ep</sub> $\mathbf { \otimes } ,$ <sub>,</sub> entirel<sub>y</sub> inde<sub>p</sub>endentl<sub>y</sub> of audit <sub>p</sub>ressure.

Program-level labels. Every program runs inside a sandboxed executor (no network, read-only files<sub>y</sub>stem, memor<sub>y</sub> and time limits). A <sub>p</sub>ro<sub>g</sub>ram is a securit<sub>y</sub> hallucination if it <sub>p</sub>asses all functional t<sub>es</sub>t<sub>s</sub> <sub>an</sub>d f<sub>a</sub>il<sub>s</sub> <sub>a</sub>t l<sub>eas</sub>t <sub>one</sub> <sub>secur</sub>it<sub>y</sub> t<sub>es</sub>t<sub>.</sub> P<sub>rograms</sub> th<sub>e</sub> <sub>san</sub>db<sub>ox</sub> <sub>cou</sub>ld <sub>no</sub>t <sub>measure</sub> <sub>are</sub> <sub>exc</sub>l<sub>u</sub>d<sub>e</sub>d<sub>.</sub> Statement-level labels. On generated code, a statement is labeled vulnerable when a staticanal<sub>y</sub>zer findin<sub>g</sub> in the task’s CWE famil<sub>y</sub> sits on it and the <sub>p</sub>ro<sub>g</sub>ram failed its securit<sub>y</sub> test:

$$
y _ { \mathrm { c o n f } } ( s ) = \left\{ \begin{array} { l l } { 1 } & { \mathrm { ~ i f ~ } \mathrm { S e c } ( P ) = 0 \mathrm { ~ a n d ~ } m ( s ) = 1 , } \\ { 0 } & { \mathrm { ~ i f ~ } \mathrm { S e c } ( P ) = 1 , \mathrm { ~ o r ~ } \mathrm { S e c } ( P ) = 0 , m ( s ) = 0 , \ M ( P ) \neq \emptyset , } \\ { \perp } & { \mathrm { ~ o t h e r w i s e ~ } ( \mathrm { e x c l u d e d } ) , } \end{array} \right.\tag{14}
$$

<sub>w</sub>h<sub>ere</sub> $m ( s ) = 1$ if a anal<sub>y</sub>zer findin<sub>g</sub> whose CWE famil<sub>y</sub> intersects C is attached to �. The anal<sub>y</sub>zer <sub>says</sub> <sub>w</sub>h<sub>ere,</sub> <sub>an</sub>d th<sub>e</sub> <sub>execu</sub>t<sub>e</sub>d <sub>secur</sub>it<sub>y</sub> t<sub>es</sub>t <sub>says</sub> <sub>w</sub>h<sub>e</sub>th<sub>er.</sub> O<sub>n</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>re</sub>f<sub>erence</sub> <sub>pa</sub>i<sub>rs</sub> th<sub>a</sub>t dif<sub>er</sub> b<sub>y a</sub> k<sub>nown e</sub>dit<sub>,</sub> th<sub>e vu</sub>l<sub>nera</sub>bl<sub>e</sub> li<sub>nes are</sub> th<sub>ose</sub> th<sub>e</sub> li<sub>ne</sub> dif <sub>rep</sub>l<sub>aces or</sub> d<sub>e</sub>l<sub>e</sub>t<sub>es, requ</sub>i<sub>r</sub>i<sub>ng no ana</sub>l<sub>yzer.</sub>

Finall<sub>y</sub>, ste<sub>p</sub> ❺ evaluates SPLASH with five research <sub>q</sub>uestions. We discuss the evaluation in $\ S 6 .$ AI Usage Note. AI coding assistants (such as Claude Opus) were used for implementation and debu<sub>gg</sub>in<sub>g</sub> of the main framework of SPLASH and for <sub>p</sub>lottin<sub>g</sub> data. All AI-assisted im<sub>p</sub>lementations <sub>were rev</sub>i<sub>ewe</sub>d b<sub>y</sub> th<sub>e au</sub>th<sub>ors, an</sub>d <sub>repor</sub>t<sub>e</sub>d <sub>num</sub>b<sub>ers come</sub> f<sub>rom execu</sub>t<sub>e</sub>d <sub>runs o</sub>f th<sub>e co</sub>d<sub>e.</sub>

## 5 Experimental Setup

A <sub>secur</sub>it h<sub>a</sub>ll<sub>uc</sub>i<sub>na</sub>ti<sub>on can on</sub>l b<sub>e es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d b <sub>execu</sub>ti<sub>n a secur</sub>it t<sub>es</sub>t<sub>.</sub> S<sub>evera</sub>l <sub>w</sub>id<sub>e</sub>l <sub>use</sub>d securit<sub>y</sub> benchmarks <sub>p</sub>rovide <sub>p</sub>rom<sub>p</sub>ts without executable tests (Securit<sub>y</sub>Eval [39], LLMSecEval [45]) and some <sub>p</sub>rovide onl<sub>y</sub> functional tests (HumanEval+ [22]). On the latter, insecurit<sub>y</sub> could onl<sub>y</sub> be estimated b<sub>y</sub> static anal<sub>y</sub>sis, which misses most securit<sub>y</sub> hallucinations (§6.1). We therefore evaluate on the 899 tasks that carr<sub>y</sub> both functional and securit<sub>y</sub> tests: SecCodePLT [28] (819 P<sub>y</sub>thon tasks), CWEval [33] (64 tasks in P<sub>y</sub>thon, C, and C++), and 16 tasks we authored for CWE t<sub>ypes un</sub>d<sub>er-represen</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e pu</sub>bli<sub>c</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s.</sub>

Authored tasks. The 16 new tasks we add span six vulnerability families: cryptography (password hashing, encryption with randomized IV—CWE-327/916/759, CWE-329/1204), injection (SQL injection, XSS sanitization, comman<sup>d</sup> injection, unsa<sup>f</sup>e YAML <sup>d</sup>eseria<sup>l</sup>ization—CWE-89, CWE-79/94, CWE-78/88, CWE-502), file handlin<sub>g</sub> (u<sub>p</sub>load validation, zi<sub>p</sub>-sli<sub>p</sub> extraction—CWE-434/22, CWE-22/409), authentication (SSL certificate verification, API re<sub>q</sub>uest si<sub>g</sub>nin<sub>g</sub>, JWT verification, <sub>p</sub>assword-reset tokens—CWE-295/297, CWE-522/798, CWE-347/345, CWE-330/613), in<sub>p</sub>ut validation (email re<sub>g</sub>ex DoS—CWE-1333), and memor<sub>y</sub> safet<sub>y</sub> (C++ fixed-bufer co<sub>py</sub>—CWE-120/787). E<sub>ac</sub>h t<sub>as</sub>k <sub>w</sub>ith <sub>execu</sub>t<sub>a</sub>bl<sub>e</sub> t<sub>es</sub>t<sub>s carr</sub>i<sub>es</sub> b<sub>o</sub>th <sub>a</sub> f<sub>unc</sub>ti<sub>ona</sub>l <sub>an</sub>d <sub>a secur</sub>it<sub>y</sub> t<sub>es</sub>t<sub>, an</sub>d th<sub>e secur</sub>it<sub>y</sub> t<sub>es</sub>t <sub>c</sub>h<sub>ec</sub>k<sub>s</sub> <sub>an</sub> <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> <sub>consequence</sub> $( \mathbf { e . g . }$ , “two hashes of the same <sub>p</sub>assword must difer”) rather than a s<sub>p</sub>ecific API<sub>,</sub> so it acce<sub>p</sub>ts an<sub>y</sub> correct im<sub>p</sub>lementation.

Generators. We generate code with 11 open models spanning six families (Qwen2.5-Coder 1.5B/7B/32B, Qwen3.6-27B, DeepSeek-Coder 6.7B/33B/V2-Lite-16B, Llama-3.1-8B, Granite-3.3-8B, GLM-4.7-Flash-30B, Nemotron-3-Nano-4B). Each model <sub>g</sub>enerates at $T \in \{ 0 . 2 , 0 . 8 \}$ <sub>w</sub>ith th<sub>ree see</sub>d<sub>s,</sub> <sub>p</sub>roducin<sub>g</sub> six sam<sub>p</sub>les <sub>p</sub>er task <sub>p</sub>er model. A fast API model (GPT-4.1-mini) also falls under this settin<sub>g</sub>. Two frontier models (Claude O<sub>p</sub>us 5, GPT-5.6 Sol), and two a<sub>g</sub>ents (Qwen2.5-Coder-7B and Qwen3-8B in a 10-step tool loop) are tested over the 80 CWEval and authored tasks at $T = 0 . 8 .$

O<sub>p</sub>en models are Hu<sub>gg</sub>in<sub>g</sub>Face check<sub>p</sub>oints run locall<sub>y</sub> with vLLM [20] (bf16 u<sub>p</sub> to 8B <sub>p</sub>arameters, 4-bit AWQ above). GPT-4.1-mini and GPT-5.6 Sol, and Claude Opus 5 are accessed through their res<sub>p</sub>ective APIs [1, 30]. All local models, includin<sub>g</sub> the auditor, run on a sin<sub>g</sub>le NVIDIA RTX 3090 (24 GB) each, on a machine with 3x such GPUs, 256 GB RAM, and AMD R<sub>y</sub>zen 24-core CPU.

Testing. Every program runs in a Bubblewrap sandbox (no network, read-only filesystem, memory and time limits) [3]. Pro<sub>g</sub>rams the sandbox could not measure are excluded and never counted as f<sub>a</sub>il<sub>ures.</sub> Of 64<sub>,</sub>137 <sub>measure</sub>d <sub>programs,</sub> 38<sub>,</sub>774 <sub>wor</sub>k<sub>;</sub> 14<sub>,</sub>495 <sub>o</sub>f th<sub>ose are</sub> i<sub>nsecure.</sub>

Auditor. We use two variations of auditing: (1) self-audit and (2) fixed audit. For the rest of the <sub>p</sub>a<sub>p</sub>er, we call them as AP (self) and AP (fixed). Qwen2.5-Coder-32B-Instruct (AWQ 4-bit) is used as th<sub>e</sub> fix<sub>e</sub>d <sub>au</sub>dit<sub>o</sub>r<sub>.</sub> A<sub>u</sub>dit Pr<sub>essu</sub>r<sub>e</sub> n<sub>ee</sub>d<sub>s</sub> t<sub>wo</sub> <sub>p</sub>r<sub>o</sub>m<sub>p</sub>t<sub>s</sub> <sub>pe</sub>r <sub>p</sub>r<sub>og</sub>r<sub>a</sub>m<sub>,</sub> <sub>so</sub> <sub>a</sub> b<sub>a</sub>t<sub>c</sub>h <sub>o</sub>f 1<sub>,</sub>000 <sub>p</sub>r<sub>og</sub>r<sub>a</sub>m<sub>s</sub> takes a<sub>pp</sub>roximatel<sub>y</sub> 20–35 minutes on one RTX 3090 GPU. Ever<sub>y</sub> settin<sub>g</sub> (frame wordin<sub>g</sub>, score d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> $\left( \mathrm { A P _ { m a x } } \right)$ , auditor identit<sub>y</sub>, comment removal) was frozen on a <sub>p</sub>ilot run over 70% of tasks (selected b<sub>y</sub> a deterministic hash of the task identifier). The main run was scored once and no settin<sub>g</sub> was adjusted after seein<sub>g</sub> main-run results. The remainin<sub>g</sub> 30% (“held-out tasks”) were never examined durin<sub>g</sub> an<sub>y</sub> desi<sub>g</sub>n decision.

Baselines. We use the following baselines: (1) Random pick; lowest generation uncertainty (the <sub>g</sub>enerator’s <sub>p</sub>er-token lo<sub>g</sub>-<sub>p</sub>robabilit<sub>y</sub>, avera<sub>g</sub>ed); most check lines (if/raise/assert/try/throw count); fewest static-anal<sub>y</sub>zer findin<sub>g</sub>s in the task’s CWE famil<sub>y</sub> (Bandit, Sem<sub>g</sub>re<sub>p</sub>, c<sub>pp</sub>check, Flawfinder); AP (self) (each o<sub>p</sub>en <sub>g</sub>enerator audits its own code); 32B <sub>y</sub>es/no (the auditor’s onetoken �(Yes) verdict); GPT-5.6 Sol as an LLM jud<sub>g</sub>e that names vulnerable lines. Because the jud<sub>g</sub>e is an API model, we ran it on a seeded random sam<sub>p</sub>le of 148 tasks with 6 <sub>g</sub>enerators (5,276 <sub>p</sub>ro<sub>g</sub>rams, 398 insecure programs with confirmed vulnerable statements). All comparisons with the judge use <sub>exac</sub>tl<sub>y</sub> thi<sub>s samp</sub>l<sub>e.</sub>

Evaluation protocol. We define the following terminologies for our evaluation: (i) Pool: A set of candidates for one task from one model. We report results on pools where the choice matters, i.e., those including at least one secure and one insecure working program. (ii) Secure picks: share of pools whose pick is secure and working. (iii) Gap closed: the gain over random choice divided by (100% − random). (iv) Hit@�: the vulnerable statement is amon<sub>g</sub> the to<sub>p</sub> �.

## 6 Evaluation

We answer five research <sub>q</sub>uestions to evaluate SPLASH:

• RQ1. Why do existing signals miss security hallucinations? (§6.1)

• RQ2. Can audit pressure choose the secure program among a model’s samples? (§6.2)

• RQ3. Does audit pressure locate the vulnerable statement? (§6.3)

• RQ4. How does it compare with what practitioners would use instead? (§6.4)

• RQ5. How robust is SPLASH across various settings? (§6.5)

## 6.1 RQ1: Why Do Existing Signals Miss Security Hallucinations?

Every model writes security hallucinations. Figure 3(a) shows the insecure share of working <sub>p</sub>ro<sub>g</sub>rams across 899 tested tasks: 24–60% for the 11 o<sub>p</sub>en <sub>g</sub>enerators<sub>,</sub> 21% for GPT-4.1-mini. Frontier models are safer (Claude O<sub>p</sub>us 5: 6%, GPT-5.6 Sol: 8% on 80 tasks), but a<sub>g</sub>ents kee<sub>p</sub> rou<sub>g</sub>hl<sub>y</sub> a third of their workin<sub>g</sub> solutions insecure (31% and 34%, Fi<sub>g</sub>ure 3b), and the <sub>g</sub>enerator’s own uncertaint<sub>y</sub> over the a<sub>g</sub>ent’s edits does not fla<sub>g</sub> them (AUROC 0.40).

The task as decider. Task identity explains 57% of the variance in insecurity; the generator only 6%. The CWE alone <sub>p</sub>redicts whether a workin<sub>g</sub> <sub>p</sub>ro<sub>g</sub>ram is insecure at AUROC 0.77 (Fi<sub>g</sub>ure 3c). Security hallucinations are consistent. Of 14 495 insecure working programs, 86% have only insecure siblin<sub>g</sub>s—other sam<sub>p</sub>les from the same <sub>g</sub>enerator, task, and tem<sub>p</sub>erature (Fi<sub>g</sub>ure 4c)— defeatin<sub>g</sub> detectors that rel<sub>y</sub> on out<sub>p</sub>ut variation [7, 23, 25]. Frontier models (GPT-5.6-Sol, Claude

![](images/f3580a90262667ece4d5870b5e274c2f95430e277a1ab9e6d11daf2f7c18d082.jpg)

(a) All 899 tested tasks  
![](images/caa2e546e75defdd819f9527bffd0fd4e40c8ae1c1309f29e17caf1b5349ff42.jpg)  
Programs that work (pass unit tests)

![](images/4ef67bb8c999d7d7f1daa0c4f978167212b0edcf8734880a346fa28574ccf416.jpg)

(b) w/ closed models, agents (c) Insecure rate per CWE  
![](images/a40fd59a55a9d26cdc962e3d5556d902543c7fc3f79122b65b1cffb752410e7d.jpg)  
Fig. 3. (a) Per generator: share that work vs. share insecure (899 tasks, 95% intervals). (b) Adding closed models and agents on 80 tasks. (c) Insecure rate by CWE and generator.  
(b) Per-task AUROC, median marked (c) Repeated across samples

(a) 300 tasks, one dot each  
![](images/07f16d67af3d2fabbd07c343b0894808557d3c8afdd3b21d28ac6dd73ba13240.jpg)

![](images/5702738da45d10eec160da254d0cb022fc4e70e1e66a20553192270d982501ce.jpg)

![](images/e5876bd3ebcfe5d1254168145d498102c270b55df13db178c16623552320b585.jpg)  
AUROC per task  
Fig. 4. (a) Per-task AUROC: AP (fixed 32B) vs uncertainty (300 tasks). (b) Distribution across four methods. (c) Insecure programs whose siblings are all insecure.

O<sub>p</sub>us 5) hallucinate less, but when a candidate <sub>p</sub>ro<sub>g</sub>ram from them contains securit<sub>y</sub> hallucinations, their siblin<sub>g</sub>s also contain them <sub>p</sub>revalentl<sub>y</sub>. Fi<sub>g</sub> 4c de<sub>p</sub>icts the scenario.

Generation uncertainty fails at both levels. Uncertainty ranks insecure programs at AUROC 0.61 <sub>overa</sub>ll <sub>an</sub>d 0<sub>.</sub>47 <sub>on</sub> CWE<sub>va</sub>l<sub>, w</sub>h<sub>ere con</sub>fid<sub>en</sub>t <sub>co</sub>d<sub>e</sub> i<sub>s more o</sub>ft<sub>en</sub> i<sub>nsecure.</sub> O<sub>n</sub> 300 t<sub>as</sub>k<sub>s w</sub>ith <sub>enoug</sub>h same-model <sub>p</sub>airs, AP (fixed 32B) ranks better in 65% of tasks (Fi<sub>g</sub>ure 4a, b). At the statement level, <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y p</sub>l<sub>aces</sub> th<sub>e vu</sub>l<sub>nera</sub>bl<sub>e s</sub>t<sub>a</sub>t<sub>emen</sub>t fi<sub>rs</sub>t i<sub>n on</sub>l<sub>y</sub> 5% <sub>o</sub>f <sub>programs,</sub> b<sub>e</sub>l<sub>ow</sub> th<sub>e</sub> 9% <sub>expec</sub>t<sub>e</sub>d b<sub>y</sub> random order (Table 1). Static anal<sub>y</sub>zers cover 15% of securit<sub>y</sub> hallucinations—and 12% of secure <sub>p</sub>ro<sub>g</sub>rams<sub>,</sub> su<sub>gg</sub>estin<sub>g</sub> hi<sub>g</sub>h false <sub>p</sub>ositive rate.

Summary: Security hallucinations afect every model (24–60% open, 6–8% frontier) and are written confidentl<sub>y</sub>: 86% have onl<sub>y</sub> insecure siblin<sub>g</sub>s. Generation uncertaint<sub>y</sub> (AUROC 0.61) and static anal<sub>y</sub>zers (15% covera<sub>g</sub>e) both miss most of them.

## 6.2 RQ2: Can Audit Pressure Choose the Secure Program?

All <sub>resu</sub>lt<sub>s use</sub> th<sub>e</sub> 11 <sub>open genera</sub>t<sub>ors, un</sub>it t<sub>es</sub>t<sub>s run</sub> fi<sub>rs</sub>t<sub>, poo</sub>l<sub>s w</sub>h<sub>ere</sub> th<sub>e c</sub>h<sub>o</sub>i<sub>ce ma</sub>tt<sub>ers.</sub> A <sub>ran</sub>d<sub>om</sub> <sub>p</sub>ick selects secure code 54.0% of the time at � = 6 and 50.0% at � = 2.

![](images/fbb5b41e8598eeb41d43279d067436589ee2e4f3680563a90b8ebdd5ee1a40a3.jpg)

![](images/ffc39cbd7634f2233bee454042a2c7fddc934bc2949be22916f459b82101ee90.jpg)  
Candidate programs per task (k)

![](images/4b8b513afc19c57c89c111a6c62a6c994b1891fb755b9ad855e7753b594ac2e9.jpg)  
Fig. 5. Secure picks among � candidates (11 generators, decisive pools). (a) With unit tests. Without: (b) security of working picks and (c) share of programs that work.

![](images/b936682df7c2f87661e781408c4544430c82c31b0d9ba5cdd4dc9e3e49d8e5af.jpg)

![](images/98afa4b964bc0532f50398246c3b0c743aec1b4cb14471265624564991c3d158.jpg)  
Fig. 6. (a) Net wins in percentage points of row over column (1,305 pools). (b) AUROC per CWE (17 CWEs, ≥25 tasks); bar: tasks testing an unstated limit.

Capable models audit their own code. AP (self) beats the generator’s own uncertainty for 8 of 11 <sub>g</sub>enerators (Table 2), but small models fail at self-audit: AP (self) falls below uncertaint<sub>y</sub> for Qwen2.5-Coder-1.5B (50 vs 66%), Dee<sub>p</sub>Seek-Coder-V2-Lite (57 vs 64%), and Granite-3.3-8B (58 vs 67%), su<sub>gg</sub>estin<sub>g</sub> a fixed auditor can be more useful for smaller models.

A fixed auditor does best. At � = 6, AP (fixed 32B) reaches 74.3% secure picks, 32B yes/no 74.2%, AP (self) 67.5%, uncertaint 64.7%, static anal zers 55.7%, random 54.0% (Fi ure 5a). The advanta<sub>g</sub>e a<sub>pp</sub>ears alread<sub>y</sub> at � = 2 (73.2% vs 55.5%). Over 1,305 decisive <sub>p</sub>ools, AP (fixed 32B) <sub>g</sub>ains +21.6 <sub>p</sub>oints over random [18.8, 24.1], closin<sub>g</sub> 46% of the <sub>g</sub>a<sub>p</sub> to a <sub>p</sub>erfect <sub>p</sub>ick, and nets +15 over uncertaint<sub>y</sub> and +19 over static anal<sub>y</sub>zers head to head (Fi<sub>g</sub>ure 6a). AP (fixed 32B) is best for 7 of 11 <sub>g</sub>enerators on choosin<sub>g</sub>, 32B <sub>y</sub>es/no for 4 (Table 2). Removin<sub>g</sub> the 32B auditor’s own <sub>p</sub>ro<sub>g</sub>rams leaves the result unchan<sub>g</sub>ed (74.5% of 1,218 <sub>p</sub>ools vs 52.6% random). The bound of Theorem 4.5 holds in <sub>p</sub>ractice: in <sub>p</sub>ools with a sin<sub>g</sub>le insecure candidate<sub>,</sub> where it <sub>g</sub>uarantees at least the within-task AUROC (0.72), AP (fixed 32B) <sub>p</sub>icks the secure <sub>p</sub>ro<sub>g</sub>ram 86% of the time.

The yes/no verdict ties for choosing but cannot locate. The 32B auditor’s yes/no answer matches AP for choosin<sub>g</sub> (74.2% vs 74.3%, net wins +1), but <sub>g</sub>ives one bit <sub>p</sub>er <sub>p</sub>ro<sub>g</sub>ram. The distinction matters in RQ3, where audit pressure’s per-statement ranking enables localization.

![](images/5bc3d020dc65afd2ee2aeead76543618e5af58a6ec1841d885b8f067a8db6a88.jpg)  
False-positive rate

![](images/33f10869fda2068aa9de17b69aaccc403b740499f855ef46b1d4504fb08e312e.jpg)  
False-positive rate

![](images/165513a71167cf70680315ec53c370429f7918d72658d8e9422765c97e091268.jpg)  
Fig. 7. (a) Statement ROC (3,910 programs from the 11 open generators). (b) Judge’s sample at low FPR; circle: AP at the judge’s rate. (c) Program ROC within a task; star: static analyzers.

![](images/f5df00cba68f84b96b6588c0499caf8948b46e662de0f0cea893f1567fd67cfa.jpg)

![](images/d4b4133014351d5965e8c7a8d5724d19faa85c858c2f83d7304b0bc463d5ab51.jpg)

![](images/176115337b7e3a06209c7f288663381964b953be90917a9c8cbf2cec7e7b2068.jpg)  
Statements inspected (k) Position of the vulnerable statement in the ranking Statements in the program  
Fig. 8. (a) Hit@� (4,091 insecure programs, all generators). (b) Where vulnerable statement lands. (c) Top-1 by prog. length.

Without unit tests, a pick must both work and be secure. Among picks that work, AP (fixed 32B) chooses secure code 87% of the time and AP (self) 81%, a<sub>g</sub>ainst 79% for uncertaint<sub>y</sub> and 74% for random (Fi<sub>g</sub>ure 5b). Uncertaint<sub>y</sub>’s advanta<sub>g</sub>e lies in avoidin<sub>g</sub> <sub>p</sub>ro<sub>g</sub>rams that fail their tests (Fi<sub>g</sub>ure 5c), the job unit tests normall<sub>y</sub> do.

Summary: AP (fixed 32B) selects secure code in 74% of decisive pools at � = 6, closing 46% of the <sub>g</sub>a<sub>p</sub> to a <sub>p</sub>erfect <sub>p</sub>ick (+21.6 <sub>pp</sub> over random, +15 net wins over uncertaint<sub>y</sub>). The 32B <sub>y</sub>es/no verdict ties for choosin<sub>g</sub> but cannot locate.

## 6.3 RQ3: Does Audit Pressure Locate the Vulnerable Statement?

On 4 091 insecure generated programs, AP (fixed 32B) places the vulnerable statement first in 58% of cases (85% at to<sub>p</sub> 3, 97% at to<sub>p</sub> 10; Table 1). AP (self) reaches 51% at to<sub>p</sub> 1, uncertaint<sub>y</sub> onl<sub>y</sub> 5%—below the 9% from random order. Statement AUROC: 0.88 for AP (fixed 32B), 0.83 for AP (self), 0.58 for uncertaint<sub>y</sub> (Fi<sub>g</sub>ure 7a). AP (fixed 32B) is best for 9 of 11 <sub>g</sub>enerators on locatin<sub>g</sub>; AP (self) for 3 (Table 2). To<sub>p</sub>-1 hit rate de<sub>g</sub>rades <sub>g</sub>racefull<sub>y</sub> with <sub>p</sub>ro<sub>g</sub>ram size: 98% for 3–6 statements, 58% for 11–15, 29% for 36+ (random: 3%; Fi<sub>g</sub>ure 8c).

Reference solutions. On 808 reference pairs from SecCodePLT—where the label is the exact changed line, no analyzer involved—AP (fixed 32B) identifies it in 77% of pairs (random: 27%; within one line: 89% vs 61%).

The frontier judge locates better on its sample. On the judge’s sample (148 tasks × 6 generators), GPT-5.6 Sol <sub>p</sub>laces the vulnerable statement first in 70% of insecure <sub>p</sub>ro<sub>g</sub>rams, vs AP (fixed 32B) 54% and AP (self) 57% (Table 1). At 5% false <sub>p</sub>ositives: 82% vs 53% (Fi<sub>g</sub>ure 7b). AP ranks ever<sub>y</sub> statement of all 4 091 insecure programs from the same pass that also chooses—and as RQ4 shows, it chooses b<sub>e</sub>tt<sub>er.</sub>

Table 1. Locating the vulnerable statement: share of insecure programs whose top-ranked statement (top 3) is the vulnerable one (%). Generated code: statements where a static finding in the task’s CWE family agrees with a failed security test. Reference solutions: the lines the secure reference changes. AP (self) excludes GPT-4.1-mini, which has no open weights.
<table><tr><td>Programs (n)</td><td>Method</td><td>Top 1 Top 3</td><td></td></tr><tr><td>Generated (4,091)</td><td>Random order</td><td>9</td><td>26</td></tr><tr><td rowspan="7">Judge sample (398)</td><td>Uncertainty</td><td>5</td><td>26</td></tr><tr><td>AP (self)</td><td>51</td><td>81</td></tr><tr><td>AP (fixed 32B)</td><td>58</td><td>85</td></tr><tr><td>Random order</td><td>8</td><td>25</td></tr><tr><td>Uncertainty</td><td>5</td><td>23</td></tr><tr><td>AP (self)</td><td>57</td><td>84</td></tr><tr><td>AP (fixed 32B)</td><td>54</td><td>80</td></tr><tr><td rowspan="2">Reference, exact (808)</td><td>GPT-5.6 judge</td><td>70</td><td>95</td></tr><tr><td>AP (fixed 32B)</td><td>77</td><td>97</td></tr><tr><td rowspan="2">Reference, ±1 (819)</td><td>Random order</td><td>27</td><td>68</td></tr><tr><td>AP (fixed 32B)</td><td>89</td><td>99</td></tr><tr><td></td><td>Random order</td><td>61</td><td>93</td></tr></table>

Table 2. Every method on every open generator. Choose: secure picks (%). Locate: top-1 hit (%); yes/no cannot locate. Best per row in bold. Locate uses the 3,910 insecure programs of the 11 open generators.
<table><tr><td></td><td colspan="7">Choose: secure picks (%)</td><td colspan="7">Locate: top 1 (%)</td></tr><tr><td></td><td></td><td colspan="3">Baselines</td><td colspan="3"></td><td colspan="3"></td><td colspan="3">Baselines</td></tr><tr><td>Generator</td><td>n</td><td>Rand.</td><td>Static</td><td>Unc.</td><td>32B yes/no</td><td>Self</td><td>Fixed 32B</td><td>n</td><td>Rand.</td><td>Unc.</td><td>32B yes/no</td><td>Self</td><td>Fixed 32B</td></tr><tr><td>Qwen2.5-C-1.5B</td><td>104</td><td>56</td><td>57</td><td>66</td><td>78</td><td>50</td><td>84</td><td>549</td><td>10</td><td>2</td><td>1</td><td>41</td><td>71</td></tr><tr><td>Nemotron-3-4B</td><td>110</td><td>47</td><td>47</td><td>55</td><td>68</td><td>56</td><td>75</td><td>304</td><td>10</td><td>5</td><td>一</td><td>54</td><td>71</td></tr><tr><td>DS-Coder-6.7B</td><td>136</td><td>51</td><td>55</td><td>52</td><td>74</td><td>64</td><td>72</td><td>477</td><td>11</td><td>9</td><td>=</td><td>53</td><td>60</td></tr><tr><td>Qwen2.5-C-7B</td><td>135</td><td>53</td><td>61</td><td>62</td><td>78</td><td>76</td><td>81</td><td>362</td><td>9</td><td>7</td><td>一</td><td>54</td><td>68</td></tr><tr><td>Granite-3.3-8B</td><td>153</td><td>50</td><td>52</td><td>67</td><td>73</td><td>58</td><td>68</td><td>475</td><td>10</td><td>9</td><td>1</td><td>55</td><td>50</td></tr><tr><td>Llama-3.1-8B</td><td>132</td><td>50</td><td>50</td><td>52</td><td>73</td><td>61</td><td>77</td><td>520</td><td>7</td><td>3</td><td></td><td>60</td><td>45</td></tr><tr><td>DS-Coder-V2-Lite</td><td>84</td><td>54</td><td>60</td><td>64</td><td>69</td><td>57</td><td>68</td><td>374</td><td>10</td><td>3</td><td></td><td>47</td><td>51</td></tr><tr><td>Qwen3.6-27B</td><td>68</td><td>57</td><td>56</td><td>60</td><td>71</td><td>68</td><td>76</td><td>129</td><td>5</td><td>5</td><td>一</td><td>25</td><td>47</td></tr><tr><td>GLM-4.7-Flash</td><td>151</td><td>55</td><td>56</td><td>62</td><td>75</td><td>67</td><td>70</td><td>163</td><td>8</td><td>2</td><td></td><td>38</td><td>53</td></tr><tr><td>Qwen2.5-C-32B</td><td>87</td><td>56</td><td>58</td><td>64</td><td>68</td><td>72</td><td>72</td><td>166</td><td>10</td><td>4</td><td>=</td><td>61</td><td>61</td></tr><tr><td>DS-Coder-33B</td><td>145</td><td>54</td><td>58</td><td>54</td><td>75</td><td>74</td><td>77</td><td>391</td><td>10</td><td>10</td><td>一</td><td>55</td><td>67</td></tr><tr><td>All generators</td><td>1,305</td><td>53</td><td>55</td><td>60</td><td>73</td><td>64</td><td>74</td><td>3,910</td><td>9</td><td>6</td><td>=</td><td>51</td><td>59</td></tr><tr><td>Held-out tasks</td><td>392</td><td>52</td><td>54</td><td>62</td><td>72</td><td>61</td><td>71</td><td>1,239</td><td>11</td><td>7</td><td>=</td><td>57</td><td>63</td></tr><tr><td>Wins (of 11)</td><td></td><td>0</td><td>0</td><td>0</td><td>4</td><td>1</td><td>7</td><td></td><td>0</td><td>0</td><td>一</td><td>3</td><td>9</td></tr></table>

Summary: AP (fixed 32B) places the vulnerable statement first in 58% of insecure programs (random: 9%) and identifies the exact chan<sub>g</sub>ed line in 77% of reference <sub>p</sub>airs. Generation uncertaint<sub>y</sub> fails at localization entirel<sub>y</sub> (5% to<sub>p</sub>-1, below random).

## 6.4 RQ4: Comparison with Practitioner Alternatives

Table 3 com<sub>p</sub>ares all methods on the jud<sub>g</sub>e’s sam<sub>p</sub>le (148 tasks × 6 <sub>g</sub>enerators). With unit tests (120 decisive <sub>p</sub>ools), AP (fixed 32B) <sub>p</sub>icks secure code in 76.7% [69.2, 83.3] of <sub>p</sub>ools, the GPT-5.6 Sol jud<sub>g</sub>e 68.0% [62.7, 73.6], uncertaint<sub>y</sub> 52.5%, random 52.9%. AP <sub>p</sub>icks secure code more often, but the intervals overla<sub>p</sub> (Fi<sub>g</sub>ure 9a).

Beside static analyzers. At the analyzers’ false-alarm rate (12%, 3,506 working programs on the ju<sup>d</sup>ge<sup>’</sup>s samp<sup>l</sup>e), static ana<sup>l</sup>yzers <sup>fl</sup>ag 18% o<sup>f</sup> security <sup>h</sup>a<sup>ll</sup>ucinations, t<sup>h</sup>e GPT-5.6 So<sup>l</sup> ju<sup>d</sup>ge 21%, 32B <sub>y</sub>es/no 27%, AP (fixed 32B) 32%, and AP combined with anal<sub>y</sub>zers 37% (Fi<sub>g</sub>ure 9b).

Table 3. What practitioners could use instead, on the GPT-5.6 judge’s sample (148 tasks, 6 generators). Choose: secure picks among the model’s own samples of a task, where they include both a secure and an insecure working program. Locate: the top statement is the vulnerable one (ties broken at random; the judge names lines without ranking). A local pass is one forward pass of the auditor over the program; AP needs one per review prompt.
<table><tr><td>Method</td><td>API Locates</td><td></td><td>Cost Choose (%) Locate (%)</td><td></td></tr><tr><td>Static analyzers</td><td>no</td><td>yes</td><td>free</td><td>55</td></tr><tr><td>Lowest generation uncertainty</td><td>no</td><td>yes</td><td>logprobs of the generator</td><td>52 5</td></tr><tr><td>Most check lines (text rule)</td><td>no</td><td>no</td><td>free</td><td>54</td></tr><tr><td>32B yes/no</td><td>no</td><td>no</td><td>1 local pass</td><td>75</td></tr><tr><td>GPT-5.6 Sol judge</td><td>yes</td><td>yes</td><td>$7.3 per 1k programs</td><td>68 70</td></tr><tr><td>AP (self)</td><td>no</td><td>yes</td><td>2 local passes</td><td>70 57</td></tr><tr><td>AP (fixed 32B)</td><td>no</td><td>yes</td><td>2 local passes</td><td>77 54</td></tr></table>

Static anal<sub>y</sub>zers (Bandit, Sem<sub>g</sub>re<sub>p</sub>, c<sub>pp</sub>check, Flawfinder) choose the candidate with the fewest findin<sub>g</sub>s in the task’s CWE f<sub>am</sub>il<sub>y;</sub> th<sub>ey</sub> <sub>are</sub> <sub>no</sub>t <sub>score</sub>d <sub>on</sub> l<sub>oca</sub>ti<sub>ng,</sub> b<sub>ecause</sub> th<sub>e</sub> l<sub>oca</sub>li<sub>za</sub>ti<sub>on</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> <sub>are</sub> b<sub>u</sub>ilt f<sub>rom</sub> th<sub>e</sub>i<sub>r</sub> fi<sub>n</sub>di<sub>ngs.</sub> At th<sub>e</sub> <sub>ana</sub>l<sub>yzers</sub>’ false-alarm rate (12%), the<sub>y</sub> fla<sub>g</sub> 18% of securit<sub>y</sub> hallucinations on this sam<sub>p</sub>le, the jud<sub>g</sub>e 21%, AP (fixed 32B) 32%, and AP and the anal<sub>y</sub>zers to<sub>g</sub>ether 37% (Fi<sub>g</sub>ure 9b).

![](images/08816b4faedc7dfbe4ad3bd63b13aa50c5bfcd40743d0c298ab3b57d5dfaafca.jpg)  
(b) Security hallucinations flagged

![](images/bf9779dbdaf0da1443ff3ad049b43804b5b02538efc7bad46a8b2fd4e050d0dc.jpg)  
Fig. 9. Judge’s sample (148 tasks × 6 generators). (a) Share of the possible gain over random on five measures. (b) Hallucinations flagged at the analyzers’ 12% false-alarm rate.

Table 4. Qualitative comparison. <sup>✓</sup> = supported, = partial, ✗ = not supported. \* = open-weight generators.
<table><tr><td>Method</td><td>Selects</td><td>Localizes</td><td>Statement Rank</td><td>Any Gen.</td><td>No train.</td><td>Local Infer.</td></tr><tr><td>Static Tools [4, 34, 38, 47]</td><td>x</td><td>0</td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Gen. Uncertainty [7]</td><td>①</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>SVEN [11]</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>LLM judge [29]</td><td>①</td><td>√</td><td>x</td><td>√</td><td>√</td><td>x</td></tr><tr><td>32B yes/no [14]</td><td>√</td><td>x</td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>AP (self)</td><td>√</td><td></td><td>√</td><td>①*</td><td>√</td><td></td></tr><tr><td>AP (fixed 32B)</td><td>√</td><td></td><td>√</td><td></td><td>√</td><td>J</td></tr></table>

Cost and generality. AP requires two local forward passes per program (∼20–35 min/1k on one 24 GB GPU); the judge costs \$7.3/1k through an API. Because AP reads only the finished code, it needs no <sub>g</sub>enerator <sub>p</sub>robabilities: on GPT-4.1-mini’s 168 insecure <sub>p</sub>ro<sub>g</sub>rams it <sub>p</sub>laces the vulnerable statement first 24% of the time (random: 5%), and on a<sub>g</sub>entic code it <sub>p</sub>icks secure code in 76% of 25 <sub>p</sub>ools without tests (random: 57%).

Summary: AP selects secure code more often than a GPT-5.6 Sol judge (77% vs. 68%) at the cost of local GPU time instead of \$7.3/1k API calls. The judge localizes better on its sample (70% vs. 54% to<sub>p</sub>-1). AP (fixed 32B) is the onl<sub>y</sub> method that selects, localizes with a <sub>p</sub>er-statement rankin<sub>g</sub>, and works on code from an<sub>y</sub> <sub>g</sub>enerator without trainin<sub>g</sub> (Table 4).

![](images/3a8a0da8c69449ca440d758503725ef8a1f5f37eae3a77f3464b97901f84ba13.jpg)  
Fig. 10. Gain over random in secure picks (pp, 95% intervals) per seting. Last column: without tests, working picks only.

## 6.5 RQ5: Robustness & Generalization

Held-out tasks. On the 30% of tasks never used to choose any setting (392 pools), AP (fixed 32B) <sub>g</sub>ains +19.9 <sub>p</sub>oints over random [15.2, 24.5], within two <sub>p</sub>oints of the main result (Fi<sub>g</sub>ure 10).

Temperature and benchmarks. The gain over random is stable across temperatures (+21.3 at � = 0.2, +21.6 at � = 0.8) and benchmarks (+21.1 on SecCodePLT, +27.5 [18.8, 35.2] on CWEval and authored tasks, 101 <sub>p</sub>ools; Fi<sub>g</sub>ure 10).

Model size. AP (fixed 32B) gains +24.3 points for 1.5–8B generators and +17.6 for 16–33B, helping <sub>more w</sub>h<sub>ere</sub> th<sub>e genera</sub>t<sub>or</sub> i<sub>s wea</sub>k<sub>er.</sub>

Per CWE. Within each task, AP (fixed 32B) beats chance in 16 of 17 CWEs with at least 25 tasks (the exce<sub>p</sub>tion is CWE-915, AUROC 0.48; Fi<sub>g</sub>ure 6b). Reversals occur across tasks for CWE-347 (AP 0.18, <sub>y</sub>es/no 0.09) and CWE-352 (0.33, 0.11), and for the <sub>y</sub>es/no answer on CWE-770 (0.16), where 98% <sub>o</sub>f t<sub>as</sub>k<sub>s</sub> t<sub>es</sub>t <sub>a</sub> li<sub>m</sub>it th<sub>e</sub> <sub>promp</sub>t <sub>never</sub> <sub>s</sub>t<sub>a</sub>t<sub>es.</sub>

Summary: Results hold on held-out tasks (within 2 pp), across temperatures, benchmarks, model sizes<sub>,</sub> and 16 of 17 CWE t<sub>yp</sub>es.

## 7 Related Work

Hallucination detection. Natural-language hallucination detectors rely on the model’s own uncertaint<sub>y</sub> or self-consistenc<sub>y</sub> [6, 7, 16, 18, 23, 25]. Amon<sub>g</sub> them, semantic entro<sub>py</sub> clusters com<sub>p</sub>letions b<sub>y</sub> meanin<sub>g</sub> [7], <sub>p</sub>robes distill this into a linear classifier on hidden states [18], P(True) asks the model to rate its own answer [16], SelfCheckGPT fla<sub>g</sub>s unre<sub>p</sub>roduced statements [23], chain-of-verification has models fact-check themselves [6], and self-contradiction detectors work within sin<sub>g</sub>le res<sub>p</sub>onses [25]. Models encode correctness in hidden states without surfacin<sub>g</sub> it [31], thou<sub>g</sub>h such <sub>p</sub>robes de<sub>g</sub>rade under <sub>p</sub>erturbation [10]. Securit<sub>y</sub> hallucinations violate the shared assum<sub>p</sub>tion ( §6.1). Code-s<sub>p</sub>ecific work classifies hallucination t<sub>yp</sub>es [21, 42], a<sub>pp</sub>lies SelfCheckGPT to code without addressin<sub>g</sub> securit<sub>y</sub> [15], and measures <sub>p</sub>acka<sub>g</sub>e hallucination at 5–22% [19, 40], but none se<sub>p</sub>arates secure from insecure code amon<sub>g</sub> functionall<sub>y</sub> correct <sub>p</sub>ro<sub>g</sub>rams.

Secure code generation. Studies report 40-76% of generated programs are vulnerable [17, 32, 43]. Benchmarks with executable securit<sub>y</sub> tests include SecCodePLT [28], CWEval [33], CodeSecEval [46], and SecureA<sub>g</sub>entBench [2]. Securit<sub>y</sub>Eval [39] and LLMSecEval [45] <sub>p</sub>rovide securit<sub>y</sub> <sub>p</sub>rom<sub>p</sub>ts without executable tests, and HumanEval+ [22] tar<sub>g</sub>ets correctness. Generation-time defenses steer models via <sub>p</sub>refix vectors [11],securit<sub>y</sub> instruction tunin<sub>g</sub> [12], <sub>p</sub>reference learnin<sub>g</sub> [48], or <sub>p</sub>rom<sub>p</sub>t en<sub>g</sub>ineerin<sub>g</sub> [9, 26], but all re<sub>q</sub>uire trainin<sub>g</sub> access or <sub>p</sub>rom<sub>p</sub>t control. Post-<sub>g</sub>eneration, static anal<sub>y</sub>zers cover onl<sub>y</sub> 15% of securit<sub>y</sub> hallucinations. SecCodePRM [49] trains a <sub>p</sub>rocess reward model on ste<sub>p</sub>-level annotations<sub>;</sub> and <sub>p</sub>er-vulnerabilit<sub>y</sub> steerin<sub>g</sub> vectors reduce insecure <sub>g</sub>eneration b<sub>y</sub> u<sub>p</sub> to 74% but need white-box access to the <sub>g</sub>enerator [37]. Audit <sub>p</sub>ressure re<sub>q</sub>uires no trainin<sub>g</sub> <sub>or</sub> l<sub>a</sub>b<sub>e</sub>l<sub>e</sub>d d<sub>a</sub>t<sub>a</sub> <sub>an</sub>d <sub>pro</sub>d<sub>uces</sub> b<sub>o</sub>th <sub>a</sub> <sub>program</sub> <sub>ran</sub>ki<sub>ng</sub> <sub>an</sub>d <sub>a</sub> <sub>per-s</sub>t<sub>a</sub>t<sub>emen</sub>t l<sub>oca</sub>li<sub>za</sub>ti<sub>on</sub> f<sub>rom</sub> t<sub>wo</sub> t<sub>eac</sub>h<sub>er-</sub>f<sub>orce</sub>d <sub>passes</sub> <sub>on</sub> <sub>co</sub>d<sub>e</sub> f<sub>rom</sub> <sub>any</sub> <sub>source.</sub>

## 8 Threats to Validity

Internal validity. Frame wording, $\operatorname { A P } _ { \operatorname* { m a x } } ,$ <sub>an</sub>d <sub>commen</sub>t bl<sub>an</sub>ki<sub>ng were</sub> f<sub>rozen on a</sub> 70% <sub>p</sub>il<sub>o</sub>t <sub>sp</sub>lit <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d b<sub>y</sub> <sub>a</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c</sub> h<sub>as</sub>h <sub>o</sub>f th<sub>e</sub> t<sub>as</sub>k id<sub>en</sub>tifi<sub>er.</sub> Th<sub>e</sub> <sub>ma</sub>i<sub>n</sub> <sub>run</sub> <sub>was</sub> <sub>score</sub>d <sub>once,</sub> <sub>an</sub>d th<sub>e</sub> 30% held-out s<sub>p</sub>lit re<sub>p</sub>roduces the secure-<sub>p</sub>ick <sub>g</sub>ain over random within two <sub>p</sub>oints (§6.5). The fixed auditor is also one of the <sub>g</sub>enerators and mi<sub>g</sub>ht <sub>p</sub>refer its own code<sub>,</sub> but removin<sub>g</sub> its <sub>p</sub>ro<sub>g</sub>rams leaves the result unchan<sub>g</sub>ed (74.5% of 1 218 <sub>p</sub>ools, §6.2).

External validity. SecCodePLT, a python-based dataset, contributes 91% of programs. CWEval and authored tasks, which include C/C++, show a lar<sub>g</sub>er <sub>g</sub>ain (§6.5). Benchmark datasets are <sub>p</sub>ublic, <sub>so memor</sub>i<sub>za</sub>ti<sub>on</sub> i<sub>s poss</sub>ibl<sub>e.</sub> H<sub>owever, our au</sub>th<sub>ore</sub>d t<sub>as</sub>k<sub>s are no</sub>t <sub>pu</sub>bli<sub>c.</sub> W<sub>e use one</sub> fi<sub>xe</sub>d <sub>au</sub>dit<sub>or</sub> (Qwen2.5-Coder-32B). Self-audit across 11 <sub>g</sub>enerators from six families confirms the measure is not s<sub>p</sub>ecific to one model (§6.2), thou<sub>g</sub>h other fixed auditors ma<sub>y</sub> behave sli<sub>g</sub>htl<sub>y</sub> diferentl<sub>y</sub>. Sam<sub>p</sub>les for frontier models and a ents are small (tested on 80 tasks), so those results are directional.

Construct validity. We equate secure as a ground truth with passing the task’s security tests. A t<sub>es</sub>t <sub>can m</sub>i<sub>ss a vu</sub>l<sub>nera</sub>bilit<sub>y, or c</sub>h<sub>ec</sub>k <sub>a requ</sub>i<sub>remen</sub>t th<sub>e promp</sub>t <sub>never s</sub>t<sub>a</sub>t<sub>es.</sub> St<sub>a</sub>t<sub>emen</sub>t l<sub>a</sub>b<sub>e</sub>l<sub>s</sub> <sub>requ</sub>i<sub>re a</sub> CWE<sub>-ma</sub>t<sub>c</sub>h<sub>e</sub>d <sub>ana</sub>l<sub>yzer</sub> fi<sub>n</sub>di<sub>ng, so</sub> l<sub>oca</sub>li<sub>za</sub>ti<sub>on</sub> i<sub>s eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>on</sub>l<sub>y on</sub> fl<sub>aws</sub> th<sub>e ana</sub>l<sub>yzers</sub> <sub>ca</sub>n <sub>p</sub>l<sub>ace, u</sub>nd<sub>e</sub>r-r<sub>ep</sub>r<sub>ese</sub>ntin<sub>g</sub> mi<sub>ss</sub>in<sub>g</sub>-<sub>c</sub>h<sub>ec</sub>k <sub>vu</sub>ln<sub>e</sub>r<sub>a</sub>biliti<sub>es.</sub> A<sub>s a seco</sub>nd <sub>eva</sub>l<sub>ua</sub>ti<sub>o</sub>n fr<sub>ee o</sub>f thi<sub>s</sub> bias, we use 808 benchmark reference <sub>p</sub>airs whose label is the line the secure version chan<sub>g</sub>es (77% to<sub>p</sub>-1, §6.3). B<sub>y</sub> desi<sub>g</sub>n, SPLASH ranks within a task.

## 9 Conclusion

We <sub>p</sub>resented SPLASH, a <sub>p</sub>ost-hoc method for choosin<sub>g</sub> secure code and locatin<sub>g</sub> securit<sub>y</sub> hallucinations from LLM-<sub>g</sub>enerated code. Its core idea is audit <sub>p</sub>ressure<sub>,</sub> which measures the reluctance <sub>o</sub>f <sub>a secur</sub>it<sub>y rev</sub>i<sub>ewer</sub> t<sub>o repro</sub>d<sub>uce eac</sub>h t<sub>o</sub>k<sub>en compare</sub>d t<sub>o a s</sub>t<sub>y</sub>l<sub>e rev</sub>i<sub>ewer.</sub> T<sub>wo</sub> t<sub>eac</sub>h<sub>er-</sub>f<sub>orce</sub>d <sub>p</sub>asses <sub>p</sub>rovide both a <sub>p</sub>ro<sub>g</sub>ram rankin<sub>g</sub> and a statement rankin<sub>g,</sub> with no trainin<sub>g,</sub> no access to the <sub>g</sub>enerator, and no CWE-s<sub>p</sub>ecific rules. On 899 curated tasks across 12 <sub>g</sub>enerators, SPLASH <sub>p</sub>icks th<sub>e secure can</sub>did<sub>a</sub>t<sub>e</sub> i<sub>n</sub> 74% <sub>o</sub>f <sub>poo</sub>l<sub>s w</sub>h<sub>ere</sub> th<sub>e c</sub>h<sub>o</sub>i<sub>ce ma</sub>tt<sub>ers, w</sub>hi<sub>c</sub>h i<sub>s</sub> b<sub>e</sub>tt<sub>er</sub> th<sub>an</sub> th<sub>e genera</sub>t<sub>or</sub> model’s own uncertaint<sub>y</sub> (65%). It further <sub>p</sub>laces the vulnerable statement first in 58% of insecure <sub>p</sub>ro<sub>g</sub>rams<sub>,</sub> within the to<sub>p</sub> three in 85%<sub>,</sub> and chooses secure code more often than a frontier LLM jud<sub>g</sub>e (77% vs. 68%), all while runnin<sub>g</sub> locall<sub>y</sub>. Audit <sub>p</sub>ressure transfers efectivel<sub>y</sub> to a<sub>g</sub>entic code (76% secure <sub>p</sub>icks) and com<sub>p</sub>lements static anal<sub>y</sub>zers. Because audit <sub>p</sub>ressure shows exactl<sub>y</sub> where a securit<sub>y</sub> reviewer would chan<sub>g</sub>e the code<sub>,</sub> it is a natural startin<sub>g</sub> <sub>p</sub>oint for automated <sub>p</sub>ro<sub>g</sub>ram re<sub>p</sub>a<sup>i</sup>r.

## 10 Data Availability

We o<sub>p</sub>en-source the code and artifacts of SPLASH at htt<sub>p</sub>s://<sub>g</sub>ithub.com/Prometheused/SPLASH.

## References

[1] Anthropic. 2026. Anthropic API Platform. https://platform.claude.com/

[2] Junkai Chen, Huihui Huan<sub>g</sub>, Yunbo L<sub>y</sub>u, Junwen An, Jieke Shi, Chen<sub>g</sub>ran Yan<sub>g</sub>, Tin<sub>g</sub> Zhan<sub>g</sub>, Hao<sub>y</sub>e Tian, Yikun Li<sub>,</sub> Zh<sub>e</sub>nh<sub>ao</sub> Li<sub>,</sub> Xin Zh<sub>ou,</sub> Xin<sub>g</sub> H<sub>u, a</sub>nd D<sub>av</sub>id L<sub>o.</sub> 2026<sub>.</sub> S<sub>ecu</sub>r<sub>e</sub>Vib<sub>e</sub>B<sub>e</sub>n<sub>c</sub>h: B<sub>e</sub>n<sub>c</sub>hm<sub>a</sub>rkin<sub>g</sub> S<sub>ecu</sub>r<sub>e</sub> Vib<sub>e</sub> C<sub>o</sub>din<sub>g o</sub>f AI Agents via Reconstructing Vulnerability-Introducing Scenarios. In Proceedings ofthe 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jur<sub>g</sub>ens (Eds.). Association for Com<sub>p</sub>utational Lin<sub>g</sub>uistics, San Die<sub>g</sub>o, California, United States, 24144–24168. d<sub>o</sub>i<sub>:</sub>10<sub>.</sub>18653/<sub>v</sub>1/2026<sub>.ac</sub>l<sub>-</sub>l<sub>on .</sub>110

[3] Containers. 2026. Bubblewrap Sandboxing tool. https://github.com/containers/bubblewrap

[4] C<sub>pp</sub>check team. 2019. C<sub>pp</sub>check: A Tool for Static C/C++ Code Anal<sub>y</sub>sis. htt<sub>p</sub>s://c<sub>pp</sub>check.sourcefor<sub>g</sub>e.io/ Version 1.90.

[5] Salma Daoud. 2026. Securit<sub>y</sub>-Related Code Hallucinations in Lar<sub>g</sub>e Lan<sub>g</sub>ua<sub>g</sub>e Models: A S<sub>y</sub>stematic Literature Review. In 2026 International Symposium ofSystems, Advanced Technologies and Knowledge (ISSATK). 1–6. doi:10.1109/ ISSATK69667.2026.11566869

[6] Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. 2024. Chain-of-Verification Reduces Hallucination in Large Language Models. In Findings ofthe Association for Computational Linguistics: ACL 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computationa Lin<sub>g</sub>uistics<sub>,</sub> Ban<sub>g</sub>kok<sub>,</sub> Thailand<sub>,</sub> 3563–3578. doi:10.18653/v1/2024.findin<sub>g</sub>s-acl.212

[7] Sebastian Far<sub>q</sub>uhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detectin<sub>g</sub> hallucinations in lar<sub>g</sub>e lan<sub>g</sub>ua<sub>g</sub>e models using semantic entropy. Nature 630, 8017 (June 2024), 625–630. doi:10.1038/s41586-024-07421-0

[8] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y.K. Li, Fuli Luo<sub>,</sub> Yin<sub>g</sub>fei Xion<sub>g,</sub> and Wenfen<sub>g</sub> Lian<sub>g</sub>. 2024. Dee<sub>p</sub>Seek-Coder: When the Lar<sub>g</sub>e Lan<sub>g</sub>ua<sub>g</sub>e Model Meets Pro<sub>g</sub>rammin<sub>g</sub> – The Rise of Code Intelli<sub>g</sub>ence. htt<sub>p</sub>s://arxiv.or<sub>g</sub>/abs/2401.14196

[9] Hossein Haji our Keno Hassler Thorsten Holz Lea Schönherr and Mario Fritz. 2024. CodeLMSec Benchmark: Systematically Evaluating and Finding Security Vulnerabilities in Black-Box Code Language Models . In 2024 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). IEEE Computer Society, Los Alamitos, CA, USA, 684–709<sub>.</sub> d<sub>o</sub>i:10<sub>.</sub>1109/S<sub>a</sub>TML59370<sub>.</sub>2024<sub>.</sub>00040

[10] Patrick Haller, Mark Ibrahim, Polina Kirichenko, Levent Sa<sub>g</sub>un, and Samuel J Bell. 2025. LLM Knowled<sub>g</sub>e is Brittle: Truthfulness Representations Rely on Superficial Resemblance. arXiv preprint arXiv:2510.11905 (2025).

[11] Jin<sub>g</sub>xuan He and Martin Vechev. 2023. Lar<sub>g</sub>e Lan<sub>g</sub>ua<sub>g</sub>e Models for Code: Securit<sub>y</sub> Hardenin<sub>g</sub> and Adversarial Testing. In Proceedings ofthe 2023 ACM SIGSAC Conference on Computer and Communications Security. 1865–1879. doi:10.1145/3576915.3623175

[12] Jin<sub>g</sub>xuan He, Mark Vero, Gabriela Krasno<sub>p</sub>olska, and Martin Vechev. 2024. Instruction tunin<sub>g</sub> for secure code <sub>g</sub>eneration. In Proceedings ofthe 41st International Conference on Machine Learning (Vienna, Austria) (ICML’24). JMLR.org, Article <sup>723</sup>, <sup>20</sup> p<sup>a</sup>g<sup>es.</sup>

[13] Ari Holtzman, Jan Bu<sub>y</sub>s, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The Curious Case of Neural Text De<sub>g</sub>eneration. In International Conference on Learning Representations. https://openreview.net/forum?id=rygGQyrFvH

[14] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, Kai Dang, Yang Fan, Yichang Zhang, An Yang, Rui Men, Fei Huang, Bo Zheng, Yibo Miao, Shanghaoran Quan, Yunlon<sub>g</sub> Fen<sub>g</sub>, Xin<sub>g</sub>zhan<sub>g</sub> Ren, Xuanchen<sub>g</sub> Ren, Jin<sub>g</sub>ren Zhou, and Jun<sub>y</sub>an<sub>g</sub> Lin. 2024. Qwen2.5-Coder Technica Report. arXiv preprint arXiv:2409.12186 (2024).

[15] Waka Ito, Yui Obara, Mi<sub>y</sub>u Sato, and Kimio Kuramitsu. 2025. Hallucination Detection on Code Generation with SelfCheckGPT. Journal of Information Processing 33 (2025), 487–493. doi:10.2197/ipsjjip.33.487

[16] Saurav Kadavath, Tom Conerl<sub>y</sub>, Amanda Askell, Tom Heni<sub>g</sub>han, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. 2022. Lan<sub>g</sub>ua<sub>g</sub>e models (mostl<sub>y</sub>) know what the<sub>y</sub> know. arXiv preprint arXiv:2207.05221 (2022).

[17] Ra<sub>p</sub>haël Khour<sub>y</sub>, Anderson R. Avila, Jacob Brunelle, and Baba Mamadou Camara. 2023. How Secure is Code Generated by ChatGPT?. In 2023 IEEE International Conference on Systems, Man, and Cybernetics (SMC). 2445–2451. doi:10.1109 SMC53992.2023.10394237

[18] Jannik Kossen, Jiaton<sub>g</sub> Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, and Yarin Gal. 2024. Semantic Entro<sub>py</sub> Probes: Robust and Chea<sub>p</sub> Hallucination Detection in LLMs. arXiv:2406.15927 [cs.CL]

[19] Arjun Krishna, Erick Galinkin, Leon Dercz<sub>y</sub>nski, and Jefre<sub>y</sub> Martin. 2025. Im<sub>p</sub>ortin<sub>g</sub> Phantoms: Measurin<sub>g</sub> LLM Package Hallucination Vulnerabilities. arXiv preprint arXiv:2501.19012 (2025).

[20] Woosuk Kwon, Zhuohan Li, Si<sub>y</sub>uan Zhuan<sub>g</sub>, Yin<sub>g</sub> Shen<sub>g</sub>, Lianmin Zhen<sub>g</sub>, Cod<sub>y</sub> Hao Yu, Jose<sub>p</sub>h Gonzalez, Hao Zhan<sub>g</sub>, and Ion Stoica. 2023. Eficient Memor<sub>y</sub> Mana<sub>g</sub>ement for Lar<sub>g</sub>e Lan<sub>gu</sub>a<sub>g</sub>e Model Ser<sub>v</sub>in<sub>g w</sub>ith Pa<sub>g</sub>edAttention. In Proceedings ofthe 29th Symposium on Operating Systems Principles (Koblenz, Germany) (SOSP ’23). Association for C<sub>o</sub>m<sub>pu</sub>tin<sub>g</sub> M<sub>ac</sub>hin<sub>e</sub>r<sub>y,</sub> N<sub>ew</sub> Y<sub>o</sub>rk<sub>,</sub> NY<sub>,</sub> USA<sub>,</sub> 611–626<sub>.</sub> d<sub>o</sub>i:10<sub>.</sub>1145/3600006<sub>.</sub>3613165

[21] Fan<sub>g</sub> Liu, Yan<sub>g</sub> Liu, Lin Shi, Zhen Yan<sub>g</sub>, Li Zhan<sub>g</sub>, Xiaoli Lian, Zhon<sub>gq</sub>i Li, and Yuchi Ma. 2026. Be<sub>y</sub>ond Functional Correctness: Exploring Hallucinations in LLM-Generated Code. IEEE Transactions on Software Engineering 52, 3 (2026), 1037–1055<sub>.</sub> d<sub>o</sub>i:10<sub>.</sub>1109/TSE<sub>.</sub>2026<sub>.</sub>3657432

[22] Jiawei Liu, Chun<sub>q</sub>iu Steven Xia, Yu<sub>y</sub>ao Wan<sub>g</sub>, and Lin<sub>g</sub>min<sub>g</sub> Zhan<sub>g</sub>. 2023. Is Your Code Generated b<sub>y</sub> ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation. Advances in neural information processing systems 36 (2023), 21558–21572. doi:10.52202/075280-0943

[23] Potsawee Manakul, Adian Liusie, and Mark Gales. 2023. SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics,

Sin<sub>g</sub>a<sub>p</sub>ore<sub>,</sub> 9004–9017. doi:10.18653/v1/2023.emnl<sub>p</sub>-main.557

[24] MITRE Cor<sub>p</sub>oration. 2024. CWE: Common Weakness Enumeration. htt<sub>p</sub>s://cwe.mitre.or<sub>g</sub>

[25] Niels Mündler, Jin<sub>g</sub>xuan He, Slobodan Jenko, and Martin Vechev. 2024. Self-contradictor<sub>y</sub> Hallucinations of Lar<sub>g</sub>e Language Models: Evaluation, Detection and Mitigation. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=EmQSOi1X2

[26] Mahmoud Nazzal, Issa Khalil, Abdallah Khreishah, and NhatHai Phan. 2024. PromSec: Prom<sub>p</sub>t O<sub>p</sub>timization for Secure Generation of Functional Source Code with Large Language Models (LLMs). In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security. 2266–2280.

[27] New Market Pitch. 2026. AI Assistant Market Growth. https://newmarketpitch.com/blogs/news/ai-code-assistantcursor-vs-<sub>g</sub>ithub-co<sub>p</sub>ilot

[28] Yuzhou Nie, Zhun Wan<sub>g</sub>, Yu Yan<sub>g</sub>, Ruizhe Jian<sub>g</sub>, Yuhen<sub>g</sub> Tan<sub>g</sub>, Xander Davies, Yarin Gal, Bo Li, Wenbo Guo, and Dawn Son<sub>g</sub>. 2025. SeCodePLT: A Unified Platform for Evaluatin<sub>g</sub> the Securit<sub>y</sub> ofCode GenAI. (2025). arXiv:2410.11096 [cs.CR] htt s://arxiv.or /abs/2410.11096

[29] OpenAI. 2026. GPT-5.6 Sol: A Next Generation Model. https://openai.com/index/previewing-gpt-5-6-sol/

[30] OpenAI. 2026. OpenAI API Models. https://openai.com/api/

[31] Hadas Or<sub>g</sub>ad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Sz<sub>p</sub>ektor, Hadas Kotek, and Yonatan Belinkov. 2025. LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id=KRnsX5Em3W

[32] Hammond Pearce, Balee<sub>g</sub>h Ahmad, Benjamin Tan, Brendan Dolan-Gavitt, and Ramesh Karri. 2022. Aslee<sub>p</sub> at the Keyboard? Assessing the Security of GitHub Copilot’s Code Contributions. In 2022 IEEE Symposium on Security and Privacy (SP). 754–768. doi:10.1109/SP46214.2022.9833571

[33] Jinjun Peng, Leyi Cui, Kele Huang, Junfeng Yang, and Baishakhi Ray. 2025. CWEval: Outcome-driven Evaluation on Functionality and Security of LLM Code Generation. In 2025 IEEE/ACM International Workshop on Large Language Models for Code (LLM4Code). IEEE, 33–40. doi:10.1109/LLM4Code66737.2025.00009

[34] P<sub>y</sub>thon Code Qualit<sub>y</sub> Authorit<sub>y</sub> (P<sub>y</sub>CQA). 2026. Bandit. htt<sub>p</sub>s://<sub>g</sub>ithub.com/P<sub>y</sub>CQA/bandit Version 1.9.4.

[35] Mirza Masfi<sub>q</sub>ur Rahman and Ashish Kundu. 2024. Code Hallucination. arXiv:2407.04831 [cs.AI]

[36] Ba<sub>p</sub>tiste Roziere, Jonas Gehrin<sub>g</sub>, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiao<sub>q</sub>in<sub>g</sub> Ellen Tan, Yossi Adi, Jin<sub>gy</sub>u Liu, Romain Sauvestre, Tal Remez, et al. 2023. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950 (2023).

[37] Gustavo Sandoval, Brendan Dolan-Gavitt, and Siddharth Gar<sub>g</sub>. 2026. Sur<sub>g</sub>ical Re<sub>p</sub>air of Insecure Code Generation in LLMs. arXiv preprint arXiv:2604.16697 (2026).

[38] Sem<sub>g</sub>re<sub>p</sub>, Inc. 2024. Sem<sub>g</sub>re<sub>p</sub>. htt<sub>p</sub>s://<sub>g</sub>ithub.com/sem<sub>g</sub>re<sub>p</sub>/sem<sub>g</sub>re<sub>p</sub> Version 1.101.0.

[39] Mohammed Latif Siddi<sub>q</sub> and Joanna C. S. Santos. 2022. Securit<sub>y</sub>Eval Dataset: Minin<sub>g</sub> Vulnerabilit<sub>y</sub> Exam<sub>p</sub>les to Evaluate Machine Learning-Based Code Generation Techniques. In Proceedings of the 1st International Workshop on Mining Software Repositories Applications for Privacy and Security (MSR4P&S22). doi:10.1145/3549035.3561184

[40] Jose<sub>p</sub>h S<sub>p</sub>racklen, Raveen Wijewickrama, A H M Nazmus Sakib, Anind<sub>y</sub>a Maiti, Bimal Viswanath, and Murtuza Jadliwala. 2025. We Have a Packa<sub>g</sub>e for You! A Com<sub>p</sub>rehensive Anal<sub>y</sub>sis of Packa<sub>g</sub>e Hallucinations b<sub>y</sub> Code Generatin<sub>g</sub> LLMs. In 34th USENIX Security Symposium (USENIX Security 25). USENIX Association, Seattle, WA, 3687–3706. htt<sub>p</sub>s://www.usenix.or<sub>g</sub>/conference/usenixsecurit<sub>y</sub>25/<sub>p</sub>resentation/s<sub>p</sub>racklen

[41] Tech Xplore. 2026. AI Written Software Statistics. https://techxplore.com/news/2026-01-ai-software-code.html

[42] Yuchen Tian, Weixiang Yan, Qian Yang, Xuandong Zhao, Qian Chen, Wen Wang, Ziyang Luo, Lei Ma, and Dawn Song. 2025. CodeHalu: Investigating Code Hallucinations in LLMs via Execution-based Verification. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 39. https://ojs.aaai.org/index.php/AAAI/article/view/34717

[43] Norbert Tihan<sub>y</sub>i, Tamas Bisztra<sub>y</sub>, Ridhi Jain, Mohamed Amine Ferra<sub>g</sub>, Lucas C Cordeiro, and Vasileios Mavroeidis. 2023. The FormAI Dataset: Generative AI in Software Security through the Lens of Formal Verification. In Proceedings of the 19th international conference on predictive models and data analytics in software engineering. 33–43.

[44] Calvin Tjoa<sub>q</sub>uinn and Sani Muhamad Isa. 2026. Characterizin<sub>g p</sub>hantom securit<sub>y</sub> functions in LLM-<sub>g</sub>enerated code: A linguistic and taxonomic analysis across programming languages. Edelweiss Applied Science and Technology 10, 6 (2026), 234–246. doi:10.55214/2576-8484.v10i6.13075

[45] Catherine Ton<sub>y</sub>, Markus Mutas, Nicolás E. Díaz Ferre<sub>y</sub>ra, and Riccardo Scandariato. 2023. LLMSecEval: A Dataset of Natural Language Prompts for Security Evaluations. In 2023 IEEE/ACM 20th International Conference on Mining Software Repositories (MSR). 588–592. doi:10.1109/MSR59073.2023.00084

[46] Jiexin Wan<sub>g</sub>, Liuwen Cao, Xiton<sub>g</sub> Luo, Yan<sub>g</sub> Cao, Zhen<sub>g</sub>hao Li, Yun<sub>y</sub>i Xiao, Men<sub>g</sub>chen Zhao, Adam Jatowt, and Yi Cai. 2026<sub>.</sub> T<sub>owa</sub>rd S<sub>ecu</sub>r<sub>e</sub> C<sub>o</sub>d<sub>e</sub> G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n: Brid<sub>g</sub>in<sub>g</sub> C<sub>o</sub>rr<sub>ec</sub>tn<sub>ess a</sub>nd S<sub>ecu</sub>rit<sub>y v</sub>i<sub>a</sub> T<sub>as</sub>k-Ad<sub>ap</sub>ti<sub>ve</sub> V<sub>u</sub>ln<sub>e</sub>r<sub>a</sub>bilit<sub>y</sub> M<sub>o</sub>d<sub>e</sub>lin<sub>g</sub> and Execution-Based Benchmarking. In Proceedings ofthe 35th ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA). Oakland, CA, USA. doi:10.1145/3832159

[47] David A. Wheeler. 2019. Flawfinder. htt<sub>p</sub>s://dwheeler.com/flawfinder/ Version 2.0.10.

[48] Xian<sub>g</sub>zhe Xu, Zian Su, Jin<sub>y</sub>ao Guo, Kai<sub>y</sub>uan Zhan<sub>g</sub>, Zhentin<sub>g</sub> Wan<sub>g</sub>, and Xian<sub>gy</sub>u Zhan<sub>g</sub>. 2025. PROSEC: fortif<sub>y</sub>in<sub>g</sub> code LLMs with proactive security alignment. In Proceedings ofthe 42nd International Conference on Machine Learning (Vancouver, Canada) (ICML’25). JMLR.org, Article 2784, 16 pages.

[49] Weichen Yu, Ravi Man<sub>g</sub>al, Yin<sub>y</sub>i Luo, Kai Hu, Jin<sub>g</sub>xuan He, Corina S. Pasareanu, and Matt Fredrikson. 2026. Sec-CodePRM: A Process Reward Model for Code Security. In Proceedings ofthe 43rd International Conference on Machine Learning (ICML).