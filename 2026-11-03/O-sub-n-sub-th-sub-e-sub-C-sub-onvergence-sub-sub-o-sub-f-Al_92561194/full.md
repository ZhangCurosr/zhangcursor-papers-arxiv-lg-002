# O<sub>n</sub> th<sub>e</sub> C<sub>onvergence</sub> <sub>o</sub>f Alt<sub>erna</sub>ti<sub>ng</sub> R<sub>egre</sub>t M<sub>a</sub>t<sub>c</sub>hi<sub>ng</sub> Al<sub>gor</sub>ith<sub>ms</sub>

A<sub>n</sub>h D<sub>uc</sub> N<sub>guyen</sub><sup>1</sup><sub>,</sub> I<sub>oann</sub>i<sub>s</sub> A<sub>nagnos</sub>tid<sub>es</sub><sup>1</sup><sub>,</sub> B<sub>r</sub>i<sub>an</sub> H<sub>u</sub> Zh<sub>ang</sub><sup>2</sup><sub>, an</sub>d T<sub>uomas</sub> S<sub>an</sub>dh<sub>o</sub>l<sub>m</sub><sup>1,3</sup>

<sup>1</sup>C<sub>a</sub>rn<sub>eg</sub>i<sub>e</sub> M<sub>e</sub>ll<sub>o</sub>n Uni<sub>ve</sub>r<sub>s</sub>it<sub>y,</sub> C<sub>o</sub>m<sub>pu</sub>t<sub>e</sub>r S<sub>c</sub>i<sub>e</sub>n<sub>ce</sub> D<sub>epa</sub>rtm<sub>e</sub>nt

<sup>2</sup>M<sub>assac</sub>h<sub>use</sub>tt<sub>s</sub> I<sub>ns</sub>tit<sub>u</sub>t<sub>e o</sub>f T<sub>ec</sub>h<sub>no</sub>l<sub>ogy</sub>

<sup>3</sup>Additi<sub>ona</sub>l <sub>a</sub>fili<sub>a</sub>ti<sub>ons:</sub> St<sub>ra</sub>t<sub>egy</sub> R<sub>o</sub>b<sub>o</sub>t<sub>,</sub> I<sub>nc.,</sub> St<sub>ra</sub>t<sub>eg</sub>i<sub>c</sub> M<sub>ac</sub>hi<sub>ne,</sub> I<sub>nc.,</sub> O<sub>p</sub>ti<sub>m</sub>i<sub>ze</sub>d M<sub>ar</sub>k<sub>e</sub>t<sub>s,</sub> I<sub>nc.</sub> {anhducn,ianagnos,sandholm}@cs.cmu.edu, zhangbh@csail.mit.edu

N<sub>ovem</sub>b<sub>er</sub> 3<sub>,</sub> 2026

## Abstract

Alternating updates, that is, where each player is updated in turn, tend to typically perform better in regret matching <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> th<sub>an s</sub>i<sub>mu</sub>lt<sub>aneous up</sub>d<sub>a</sub>t<sub>es.</sub> Alt<sub>erna</sub>ti<sub>on w</sub>ithi<sub>n</sub> th<sub>e regre</sub>t <sub>ma</sub>t<sub>c</sub>hi<sub>ng</sub> f<sub>am</sub>il<sub>y</sub> h<sub>as</sub> l<sub>ong</sub> b<sub>een a cen</sub>t<sub>ra</sub>l i<sub>ngre</sub>di<sub>en</sub>t <sub>o</sub>f <sub>prac</sub>ti<sub>ca</sub>l <sub>so</sub>l<sub>vers</sub> f<sub>or</sub> <sub>zero-sum</sub> <sub>games.</sub> Y<sub>e</sub>t <sub>our</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng</sub> i<sub>s</sub> l<sub>agg</sub>i<sub>ng</sub> b<sub>e</sub>hi<sub>n</sub>d<sub>.</sub> I<sub>n</sub> thi<sub>s</sub> <sub>paper,</sub> <sub>we</sub> <sub>a</sub>dd<sub>ress</sub> thi<sub>s gap</sub> b<sub>y</sub> d<sub>eve</sub>l<sub>op</sub>i<sub>ng a sys</sub>t<sub>ema</sub>ti<sub>c way o</sub>f <sub>ana</sub>l<sub>yz</sub>i<sub>ng suc</sub>h <sub>a</sub>l<sub>gor</sub>ith<sub>ms.</sub> O<sub>ur</sub> f<sub>ramewor</sub>k <sub>cen</sub>t<sub>ers on an a</sub>lt<sub>erna</sub>ti<sub>ng ana</sub>l<sub>og</sub> <sub>o</sub>f th<sub>e</sub> RVU <sub>proper</sub>t<sub>y</sub> th<sub>a</sub>t <sub>we</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce, w</sub>hi<sub>c</sub>h <sub>we use</sub> t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e op</sub>ti<sub>ma</sub>l �<sup>−1</sup> <sub>average-</sub>it<sub>era</sub>t<sub>e convergence w</sub>ithi<sub>n</sub> this family. In particular, we establish this for two simple variants of predictive regret matching<sup>+</sup>: i) increasing-regret predictive regret matching<sup>+</sup> and ii) half-prediction regret matching<sup>+</sup>. Interestingly, we find that half prediction is the unique prediction multiplier that satisfies our alternating RVU property, giving it a strong theoretical justification.

G<sub>o</sub>i<sub>ng</sub> b<sub>eyon</sub>d <sub>average-</sub>it<sub>era</sub>t<sub>e</sub> <sub>convergence,</sub> <sub>we</sub> <sub>re</sub>fi<sub>ne</sub> <sub>our</sub> f<sub>ramewor</sub>k t<sub>o</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> �<sup>−1/2</sup> b<sub>es</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e</sub> <sub>convergence</sub> <sub>an</sub>d f<sub>ur</sub>th<sub>er</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>convergence</sub> <sub>o</sub>f th<sub>e</sub>i<sub>r</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>pro</sub>fil<sub>es</sub> t<sub>o</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>um,</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>assum</sub>i<sub>ng</sub> <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> <sub>un</sub>i<sub>queness.</sub> W<sub>e</sub> <sub>go</sub> <sub>on</sub> t<sub>o</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>separa</sub>ti<sub>ons</sub> b<sub>e</sub>t<sub>ween</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>an</sub>d <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub> <sub>up</sub>d<sub>a</sub>t<sub>es,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>a</sub> <sub>game</sub> i<sub>n</sub> <sub>w</sub>hi<sub>c</sub>h h<sub>a</sub>lf<sub>-pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>regre</sub>t <sub>ma</sub>t<sub>c</sub>hi<sub>ng</sub><sup>+</sup> <sub>converges</sub> <sub>un</sub>d<sub>er</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>up</sub>d<sub>a</sub>t<sub>es</sub> b<sub>u</sub>t f<sub>a</sub>il<sub>s</sub> t<sub>o</sub> <sub>converge</sub> i<sub>n</sub> t<sub>erms</sub> <sub>o</sub>f th<sub>e</sub> l<sub>as</sub>t it<sub>era</sub>t<sub>e un</sub>d<sub>er s</sub>i<sub>mu</sub>lt<sub>aneous up</sub>d<sub>a</sub>t<sub>es.</sub> Thi<sub>s prov</sub>id<sub>es</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>suppor</sub>t f<sub>or</sub> th<sub>e emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> th<sub>a</sub>t <sub>a</sub>lt<sub>erna</sub>ti<sub>on</sub> <sub>can s</sub>t<sub>a</sub>bili<sub>ze</sub> l<sub>earn</sub>i<sub>ng</sub> d<sub>ynam</sub>i<sub>cs.</sub> Fi<sub>na</sub>ll<sub>y, exper</sub>i<sub>men</sub>t<sub>s on</sub> 15 <sub>game</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>a</sub>t th<sub>e s</sub>t<sub>rong</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>proper</sub>ti<sub>es o</sub>f th<sub>e</sub> h<sub>a</sub>lf<sub>-pre</sub>di<sub>c</sub>ti<sub>on mec</sub>h<sub>an</sub>i<sub>sm</sub> t<sub>rans</sub>l<sub>a</sub>t<sub>e</sub> t<sub>o s</sub>t<sub>rong prac</sub>ti<sub>ca</sub>l <sub>per</sub>f<sub>ormance,</sub> b<sub>e</sub>i<sub>ng on par w</sub>ith <sub>s</sub>t<sub>a</sub>t<sub>e-o</sub>f<sub>-</sub>th<sub>e-ar</sub>t <sub>so</sub>l<sub>vers.</sub>

## Contents

Introduction 1   
1<sub>.</sub>1 O<sub>ur</sub> C<sub>on</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons .</sub> 2   
1<sub>.</sub>2 Oth<sub>er</sub> R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d R<sub>esearc</sub>h 2   
Preliminaries on Zero-Sum Games 3   
3 Regret Matching: Old and New 4   
4 Alternating vs Simultaneous Updates 5   
5 Average-Iterate Convergence 6   
5<sub>.</sub>1 Alt<sub>erna</sub>ti<sub>ng</sub> RVU <sub>an</sub>d A<sub>na</sub>l<sub>ys</sub>i<sub>s</sub> T<sub>emp</sub>l<sub>a</sub>t<sub>e .</sub> 6   
5.2 Alt-RVU with Norm Monotonicit : Alt-IR-PRM<sup>+</sup> 9   
5.3 Alt-RVU without Norm Monotonicit : Alt-HPRM 10   
5<sub>.</sub>4 I<sub>nsu</sub>fi<sub>c</sub>i<sub>enc o</sub>f St<sub>an</sub>d<sub>ar</sub>d RVU 11   
Alt-RVU Failure of Scaled-Prediction Regret Matching 12   
Best- and Last-Iterate Convergence 14   
Separation of Alternating and Simultaneous Updates 16   
Experiments 17   
9<sub>.</sub>1 N<sub>umer</sub>i<sub>ca</sub>l S<sub>epara</sub>ti<sub>ons un</sub>d<sub>er</sub> P<sub>re</sub>di<sub>c</sub>ti<sub>on</sub> S<sub>ca</sub>li<sub>ng</sub> 17   
9<sub>.</sub>2 P<sub>er</sub>f<sub>ormance on</sub> E<sub>x</sub>t<sub>ens</sub>i<sub>ve-</sub>F<sub>orm</sub> B<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> 18   
10 Conclusions 20   
11 AI Disclosure 20   
12 Acknowledgments 20   
A Omitted Proofs 26   
A<sub>.</sub>1 P<sub>roo</sub>f<sub>s</sub> f<sub>rom</sub> S<sub>u</sub>b<sub>sec</sub>ti<sub>on</sub> 5<sub>.</sub>1 <sub>.</sub> 26   
A<sub>.</sub>2 P<sub>roo</sub>f<sub>s</sub> f<sub>rom</sub> S<sub>u</sub>b<sub>sec</sub>ti<sub>on</sub> 5<sub>.</sub>2 <sub>.</sub> 29   
A<sub>.</sub>3 Th<sub>e</sub> Th<sub>eore</sub>ti<sub>ca</sub>l R<sub>o</sub>l<sub>e o</sub>f H<sub>a</sub>lf P<sub>re</sub>di<sub>c</sub>ti<sub>on</sub> 31   
A<sub>.</sub>4 P<sub>roo</sub>f<sub>s</sub> f<sub>rom</sub> S<sub>u</sub>b<sub>sec</sub>ti<sub>on</sub> 5<sub>.</sub>3 32   
A<sub>.</sub>5 P<sub>roo</sub>f<sub>s</sub> f<sub>rom</sub> S<sub>u</sub>b<sub>sec</sub>ti<sub>on</sub> 5<sub>.</sub>4 37   
A<sub>.</sub>6 P<sub>roo</sub>f<sub>s</sub> f<sub>rom</sub> S<sub>ec</sub>ti<sub>on</sub> 6 <sub>.</sub> 39   
A<sub>.</sub>7 P<sub>roo</sub>f<sub>s</sub> f<sub>rom</sub> S<sub>ec</sub>ti<sub>on</sub> 7 48   
A<sub>.</sub>8 P<sub>roo</sub>f<sub>s</sub> f<sub>rom</sub> S<sub>ec</sub>ti<sub>on</sub> 8 <sub>.</sub> 55   
B Additional Experimental Details and Results 58   
B<sub>.</sub>1 C<sub>ons</sub>t<sub>ruc</sub>ti<sub>on an</sub>d V<sub>a</sub>lid<sub>a</sub>ti<sub>on o</sub>f th<sub>e</sub> N<sub>umer</sub>i<sub>ca</sub>l S<sub>epara</sub>ti<sub>ons .</sub> 58   
B<sub>.</sub>2 Vi<sub>sua</sub>li<sub>za</sub>ti<sub>ons</sub> f<sub>or</sub> th<sub>e</sub> S<sub>epara</sub>ti<sub>on o</sub>f Alt<sub>erna</sub>ti<sub>ng an</sub>d Si<sub>mu</sub>lt<sub>aneous</sub> U<sub>p</sub>d<sub>a</sub>t<sub>es .</sub> 59   
B<sub>.</sub>3 F<sub>ur</sub>th<sub>er</sub> D<sub>e</sub>t<sub>a</sub>il<sub>s</sub> f<sub>or</sub> P<sub>er</sub>f<sub>ormance on</sub> E<sub>x</sub>t<sub>ens</sub>i<sub>ve-</sub>F<sub>orm</sub> B<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> 59   
B<sub>.</sub>4 E<sub>x</sub>t<sub>ra</sub> Pl<sub>o</sub>t<sub>s</sub> f<sub>or</sub> B<sub>enc</sub>h<sub>mar</sub>k G<sub>ames</sub> 61

## 1 Introduction

Regret matching (RM)—introduced by Hart and Mas-Colell [2000]—and its variants underlie many of the most efective <sub>me</sub>th<sub>o</sub>d<sub>s</sub> f<sub>or prac</sub>ti<sub>ca</sub>l <sub>equ</sub>ilib<sub>r</sub>i<sub>um compu</sub>t<sub>a</sub>ti<sub>on</sub> th<sub>roug</sub>h <sub>se</sub>lf<sub>-p</sub>l<sub>ay.</sub> I<sub>n</sub> thi<sub>s approac</sub>h<sub>, eac</sub>h <sub>p</sub>l<sub>ayer repea</sub>t<sub>e</sub>dl<sub>y up</sub>d<sub>a</sub>t<sub>es</sub> it<sub>s</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy us</sub>i<sub>ng a regre</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zer, w</sub>ith th<sub>e o</sub>th<sub>er p</sub>l<sub>ayer</sub>’<sub>s s</sub>t<sub>ra</sub>t<sub>egy</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>ng</sub> th<sub>e o</sub>b<sub>serve</sub>d <sub>u</sub>tilit<sub>y vec</sub>t<sub>or.</sub> P<sub>rom</sub>i<sub>nen</sub>t variants in the RM famil are regret matching<sup>+</sup> (RM<sup>+</sup>) [Tammelin, 2014], discounted regret matching (DRM) [Brown and Sandholm, 2019], and predictive regret matching<sup>+</sup> (PRM<sup>+</sup>) [Farina et al., 2021]. A central practical advantage of RM, $\mathbb { R } \mathbb { M } ^ { + }$ <sub>, an</sub>d $\mathrm { P R M ^ { + } }$ i<sub>s</sub> th<sub>a</sub>t th<sub>ey requ</sub>i<sub>re no</sub> l<sub>earn</sub>i<sub>ng-ra</sub>t<sub>e</sub> t<sub>un</sub>i<sub>ng an</sub>d <sub>are</sub> i<sub>nvar</sub>i<sub>an</sub>t t<sub>o pos</sub>iti<sub>ve resca</sub>li<sub>ng o</sub>f <sub>u</sub>tiliti<sub>es.</sub> O<sub>n</sub> th<sub>e</sub> <sub>o</sub>th<sub>er</sub> h<sub>an</sub>d<sub>, s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>s</sub>ti<sub>c m</sub>i<sub>rror-</sub>d<sub>escen</sub>t <sub>me</sub>th<sub>o</sub>d<sub>s,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng op</sub>ti<sub>m</sub>i<sub>s</sub>ti<sub>c mu</sub>lti<sub>p</sub>li<sub>ca</sub>ti<sub>ve we</sub>i<sub>g</sub>ht<sub>s,</sub> i<sub>ns</sub>t<sub>ea</sub>d d<sub>epen</sub>d <sub>on</sub> a suitable choice of learnin<sub>g</sub> rate, whose a<sub>pp</sub>ro<sub>p</sub>riate ma<sub>g</sub>nitude chan<sub>g</sub>es with the <sub>p</sub>a<sub>y</sub>of scale [Rakhlin and Sridharan, 2013, Farina et al., 2021, Zhang et al., 2025]. This distinction becomes particularly relevant within counterfactual regret minimization (CFR), which decomposes regret minimization in an extensive-form game into local regret minimizers at individual information sets [Zinkevich et al., 2007]. RM variants allow these local learners to o<sub>p</sub>erate without calibratin<sub>g</sub> learnin<sub>g</sub> rates across <sub>p</sub>otentiall<sub>y</sub> man<sub>y</sub> information sets with diferent utilit<sub>y</sub> scales [Chakrabarti et al., 2024]. Moreover, em irical com arisons show that RM-based solvers can substantiall out erform o timistic mirror-descent baselines on benchmark extensive-form <sub>g</sub>ames, des<sub>p</sub>ite the latter’s stron<sub>g</sub>er worst-case conver<sub>g</sub>ence <sub>g</sub>uarantees [Farina et al., 2019c].

Average-iterate convergence is central to equilibrium computation in two-player zero-sum games, as it provides a wa<sub>y</sub> to obtain an a<sub>pp</sub>roximate Nash e<sub>q</sub>uilibrium even when the individual strate<sub>gy p</sub>rofiles do not stabilize [Freund and Scha<sub>p</sub>ire, 1999, Zinkevich et al., 2007, S<sub>y</sub>r<sub>g</sub>kanis et al., 2015]. For uniform avera<sub>g</sub>es, the ${ \cal O } ( T ^ { - 1 } )$ <sub>ra</sub>t<sub>e</sub> i<sub>s wors</sub>t<sub>-case</sub> o<sub>p</sub>timal in its de<sub>p</sub>endence on the horizon �, even when the hidden constant ma<sub>y</sub> de<sub>p</sub>end on the <sub>g</sub>ame [Daskalakis et al., 2015, Zhan<sub>g</sub> et al., 2025]. Nonuniform avera<sub>g</sub>in<sub>g</sub> schemes that em<sub>p</sub>hasize later strate<sub>g</sub>ies reduce the influence of <sub>p</sub>oor earl<sub>y</sub> iterates and can substantiall<sub>y</sub> accelerate em<sub>p</sub>irical ex<sub>p</sub>loitabilit<sub>y</sub> reduction in CFR solvers [Brown and Sandholm, 2019, Zhan<sub>g</sub> et al., 2026]. In <sub>p</sub>ractice, we also want to understand the <sub>q</sub>ualit<sub>y</sub> of the individual <sub>p</sub>rofiles <sub>genera</sub>t<sub>e</sub>d d<sub>ur</sub>i<sub>ng</sub> <sub>se</sub>lf<sub>-p</sub>l<sub>ay,</sub> <sub>espec</sub>i<sub>a</sub>ll<sub>y</sub> b<sub>ecause</sub> th<sub>e</sub> b<sub>es</sub>t <sub>an</sub>d l<sub>as</sub>t it<sub>era</sub>t<sub>e</sub> <sub>can</sub> <sub>converge</sub> <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> f<sub>as</sub>t<sub>er</sub> th<sub>an</sub> th<sub>e</sub> <sub>average</sub> on benchmark <sub>g</sub>ames [Cai et al., 2025, Zhan<sub>g</sub> et al., 2025, Zhan<sub>g</sub> and Sandholm, 2026]. This motivates stud<sub>y</sub>in<sub>g</sub> best-iterate convergence, which quantifies how quickly self-play reaches a profile with small exploitability, together with last-iterate convergence, which ensures that all suficiently late profiles have small exploitability [Anagnostides et al., 2022, Cai et al., 2025].

Alternating updates, where players update their strategies one after the other, are widely used in practical equilibrium <sub>compu</sub>t<sub>a</sub>ti<sub>on</sub> th<sub>roug</sub>h <sub>se</sub>lf<sub>-p</sub>l<sub>ay w</sub>ith <sub>regre</sub>t <sub>ma</sub>t<sub>c</sub>hi<sub>ng me</sub>th<sub>o</sub>d<sub>s.</sub> Th<sub>ey were par</sub>t <sub>o</sub>f <sub>coun</sub>t<sub>er</sub>f<sub>ac</sub>t<sub>ua</sub>l <sub>regre</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub><sup>+</sup> (CFR<sup>+</sup>) from its introduction [Tammelin, 2014] and were subse<sub>q</sub>uentl<sub>y</sub> used in two landmark achievements in <sub>p</sub>oker AI: Cepheus near-optimally solved heads-up limit Texas hold’em [Bowling et al., 2015], while Libratus achieved su<sub>p</sub>erhuman <sub>p</sub>erformance in heads-u<sub>p</sub> no-limit Texas hold’em [Brown and Sandholm, 2018]. Subse<sub>q</sub>uent methods, includin<sub>g</sub> discounted and <sub>p</sub>redictive CFR variants, also use alternatin<sub>g</sub> u<sub>p</sub>dates to obtain stron<sub>g</sub> em<sub>p</sub>irical im<sub>p</sub>rovements [Brown and Sandholm, 2019, Farina et al., 2021]. Des<sub>p</sub>ite this wides<sub>p</sub>read use, substantial <sub>g</sub>a<sub>p</sub>s remain in <sub>our un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng o</sub>f th<sub>e</sub>i<sub>r convergence.</sub> G<sub>uaran</sub>t<sub>ees</sub> f<sub>or</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>regre</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zers or s</sub>i<sub>mu</sub>lt<sub>aneous se</sub>lf<sub>-p</sub>l<sub>ay</sub> d<sub>o</sub> not automaticall<sub>y</sub> establish conver<sub>g</sub>ence under alternatin<sub>g</sub> u<sub>p</sub>dates [Farina et al., 2019b, Burch et al., 2019]. In fact, a recurrin<sub>g p</sub>attern in the RM and CFR literature is to evaluate alternatin<sub>g</sub> im<sub>p</sub>lementations for their stron<sub>g</sub> em<sub>p</sub>irical <sub>per</sub>f<sub>ormance, w</sub>hil<sub>e</sub> th<sub>e accompany</sub>i<sub>ng</sub> th<sub>eory es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>regre</sub>t l<sub>earner</sub> b<sub>oun</sub>d<sub>s or ana</sub>l<sub>yzes equ</sub>ilib<sub>r</sub>i<sub>um</sub> conver<sub>g</sub>ence under simultaneous u<sub>p</sub>dates [Brown and Sandholm, 2019, Farina et al., 2019a, 2021, 2023, Chakrabarti et al., 2024, Xu et al., 2024, Zhan<sub>g</sub> et al., 2025, Men<sub>g</sub> et al., 2026]. These anal<sub>y</sub>ses do not, b<sub>y</sub> themselves, establish conver<sub>g</sub>ence of the alternatin<sub>g</sub> d<sub>y</sub>namics evaluated in <sub>p</sub>ractice. Des<sub>p</sub>ite <sub>g</sub>uarantees for some alternatin<sub>g</sub> variants [Burch et al., 2019, Sullins and Kash, 2026], a s<sub>y</sub>stematic understandin<sub>g</sub> of their worst-case conver<sub>g</sub>ence remains incom<sub>p</sub>lete. This <sub>g</sub>a<sub>p</sub> is <sub>p</sub>articularl<sub>y</sub> strikin<sub>g</sub> for alternatin<sub>g</sub> PRM<sup>+</sup> (Alt-PRM<sup>+</sup>), whose CFR instantiation Alt-PCFR<sup>+</sup> has achieved state-of-the-art empirical performance across the majority of benchmark games [Farina et al., 2021, Chakrabarti et al., 2024, Zhang et al., 2025, 2026] and is the equilibrium solver within Obscuro, the first superhuman AI for Fog of War chess [Zhan<sub>g</sub> and Sandholm, 2026]. Yet no <sub>g</sub>eneral <sub>g</sub>uarantee is known for its avera<sub>g</sub>e-, best-, or last-iterate conver<sub>g</sub>ence, even in finite two-<sub>p</sub>la<sub>y</sub>er zero-sum normal-form <sub>g</sub>ames [Cai et al., 2025].

Our <sub>g</sub>oals are to address this <sub>g</sub>a<sub>p</sub> and develo<sub>p</sub> a s<sub>y</sub>stematic anal<sub>y</sub>sis of alternatin<sub>g</sub> u<sub>p</sub>dates for the RM famil<sub>y</sub> that di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>es</sub> li<sub>m</sub>it<sub>a</sub>ti<sub>ons</sub> <sub>o</sub>f <sub>ex</sub>i<sub>s</sub>ti<sub>ng</sub> <sub>proo</sub>f t<sub>ec</sub>h<sub>n</sub>i<sub>ques</sub> f<sub>rom</sub> <sub>essen</sub>ti<sub>a</sub>l <sub>requ</sub>i<sub>remen</sub>t<sub>s</sub> f<sub>or</sub> <sub>convergence,</sub> <sub>an</sub>d <sub>gu</sub>id<sub>es</sub> th<sub>e</sub> d<sub>es</sub>i<sub>gn</sub> <sub>o</sub>f <sub>me</sub>th<sub>o</sub>d<sub>s</sub> th<sub>a</sub>t <sub>re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e</sub> f<sub>am</sub>il<sub>y</sub>’<sub>s</sub> <sub>prac</sub>ti<sub>ca</sub>l <sub>s</sub>t<sub>reng</sub>th<sub>s</sub> <sub>w</sub>hil<sub>e</sub> <sub>a</sub>d<sub>m</sub>itti<sub>ng</sub> <sub>s</sub>t<sub>rong</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>guaran</sub>t<sub>ees.</sub> W<sub>e</sub> <sub>rev</sub>i<sub>s</sub>it the challen<sub>g</sub>es involved in establishin<sub>g</sub> conver<sub>g</sub>ence <sub>g</sub>uarantees of PRM : i) decreases in re<sub>g</sub>ret vector norms (which runs <sub>coun</sub>t<sub>er</sub> t<sub>o</sub> th<sub>e</sub> <sub>recen</sub>t id<sub>ea</sub> <sub>o</sub>f <sub>ac</sub>hi<sub>ev</sub>i<sub>ng</sub> <sub>convergence</sub> i<sub>n</sub> <sub>cer</sub>t<sub>a</sub>i<sub>n</sub> <sub>s</sub>i<sub>mu</sub>lt<sub>aneous-up</sub>d<sub>a</sub>t<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> <sub>us</sub>i<sub>ng</sub> <sub>non</sub>d<sub>ecreas</sub>i<sub>ng</sub> re<sub>g</sub>ret vector norms [Zhan<sub>g</sub> et al., 2025]) and ii) strate<sub>gy</sub> instabilit<sub>y</sub> [Farina et al., 2023] (simultaneous u<sub>p</sub>dates can oscillate back and forth). We focus on increasing-regret predictive regret matching<sup>+</sup> (IR-PRM<sup>+</sup>) of Zhang et al. [2025], which enforces nondecreasing regret vector norms, and an algorithm we introduce, half-prediction regret matching<sup>+</sup> (HPRM<sup>+</sup>), which sim<sub>p</sub>l<sub>y</sub> halves the <sub>p</sub>rediction term of PRM<sup>+</sup>. The latter al<sub>g</sub>orithm belon<sub>g</sub>s to the famil<sub>y p</sub>ro<sub>p</sub>osed b<sub>y</sub> Men<sub>g</sub> et al. [2026]. We also investi<sub>g</sub>ate whether alternatin<sub>g</sub> u<sub>p</sub>dates merel<sub>y</sub> accelerate conver<sub>g</sub>ence or can determine whether the strate<sub>gy p</sub>rofile a<sub>pp</sub>roaches equilibrium at all (the latter is su<sub>gg</sub>ested b<sub>y</sub> the ex<sub>p</sub>eriments of simultaneou and alternatin<sub>g</sub> PRM<sup>+</sup> of Cai et al. [2025]).

## 1.1 Our Contributions

W<sub>e</sub> d<sub>eve</sub>l<sub>op a</sub> f<sub>ramewor</sub>k f<sub>or ana</sub>l<sub>yz</sub>i<sub>ng</sub> th<sub>e average-,</sub> b<sub>es</sub>t<sub>- an</sub>d l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e convergence o</sub>f <sub>a</sub>lt<sub>erna</sub>ti<sub>ng regre</sub>t <sub>ma</sub>t<sub>c</sub>hi<sub>ng</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>ms, es</sub>t<sub>a</sub>bli<sub>s</sub>hi<sub>ng</sub> th<sub>e s</sub>t<sub>rong convergence guaran</sub>t<sub>ees</sub> t<sub>oge</sub>th<sub>er w</sub>ith <sub>s</sub>t<sub>rong emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>per</sub>f<sub>ormance on</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s.</sub> I<sub>n more</sub> d<sub>e</sub>t<sub>a</sub>il<sub>, we summar</sub>i<sub>ze our ma</sub>i<sub>n con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> b<sub>e</sub>l<sub>ow.</sub>

• A systematic analysis of alternating self-play. We introduce the alternating regret bounded by variation in utilities bound (Alt-RVU) and its refinement, refined Alt-RVU, together with a common analysis template for <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>hi<sub>ng</sub> th<sub>ese</sub> b<sub>oun</sub>d<sub>s.</sub> Alt<sub>-</sub>RVU <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> ${ \cal O } ( T ^ { - 1 } )$ <sub>average-</sub>it<sub>era</sub>t<sub>e convergence, w</sub>hil<sub>e</sub> th<sub>e re</sub>fi<sub>nemen</sub>t <sub>y</sub>i<sub>e</sub>ld<sub>s an</sub> ${ \cal O } ( T ^ { - 1 / 2 } )$ best-iterate rate and a last-iterate conver<sub>g</sub>ence to the Nash set (Subsection 5.1 and Section 7). We also show that certain re<sub>g</sub>ret minimizers can fail the standard (simultaneous) RVU of [S<sub>y</sub>r<sub>g</sub>kanis et al., 2015] <sub>w</sub>hil<sub>e</sub> h<sub>av</sub>i<sub>ng</sub> th<sub>e op</sub>ti<sub>ma</sub>l ${ \cal O } ( T ^ { - 1 } )$ <sub>ave</sub>r<sub>age-</sub>it<sub>e</sub>r<sub>a</sub>t<sub>e co</sub>n<sub>ve</sub>r<sub>ge</sub>n<sub>ce.</sub> H<sub>e</sub>n<sub>ce, s</sub>t<sub>a</sub>nd<sub>a</sub>rd RVU <sub>ca</sub>nn<sub>o</sub>t r<sub>ep</sub>l<sub>ace</sub> Alt<sub>-</sub>RVU in our anal<sub>y</sub>sis (Subsection 5.4). As a first a<sub>pp</sub>lication, we establish both new Alt-RVU bounds for the increasin<sub>g</sub> re<sub>g</sub>ret <sub>p</sub>redictive re<sub>g</sub>ret matchin<sub>g</sub> al<sub>g</sub>orithm IR-PRM<sup>+</sup> of Zhan<sub>g</sub> et al. [2025], showin<sub>g</sub> how norm monotonicit<sub>y</sub> and norm-<sub>p</sub>reservin<sub>g p</sub>rediction su<sub>pp</sub>l<sub>y</sub> the required control (Subsection 5.2 and Section 7).

• Fast convergence without norm monotonicity. We show that the re<sub>g</sub>ret-vector-norm monotonicit<sub>y</sub> used to <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h th<sub>ese</sub> b<sub>oun</sub>d<sub>s</sub> f<sub>or</sub> $\mathrm { \ I R - P R M ^ { + } \ i s , }$ i<sub>n</sub> f<sub>ac</sub>t<sub>, no</sub>t <sub>necessary.</sub> U<sub>n</sub>d<sub>er a</sub>lt<sub>erna</sub>ti<sub>ng se</sub>lf<sub>-p</sub>l<sub>ay, we prove w</sub>ith <sub>our</sub> Alt<sub>-</sub> RVU framework that half-<sub>p</sub>rediction re<sub>g</sub>ret matchin<sub>g</sub> HPRM<sup>+</sup> achieves its ${ \cal O } ( T ^ { - 1 } )$ <sup>a</sup>v<sup>era</sup>g<sup>e-iterate</sup> <sup>con</sup>v<sup>er</sup>g<sup>ence</sup>, an ${ \cal O } ( T ^ { - 1 / 2 } )$ b<sub>es</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e ra</sub>t<sub>e, an</sub>d l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e convergence</sub> t<sub>o a s</sub>i<sub>ng</sub>l<sub>e</sub> N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> d<sub>esp</sub>it<sub>e a</sub>ll<sub>ow</sub>i<sub>ng</sub> it<sub>s</sub> re<sub>g</sub>ret vector norm to decrease, while existin<sub>g</sub> anal<sub>y</sub>sis for this famil<sub>y</sub> [Men<sub>g</sub> et al., 2026] does not establish an<sub>y</sub> such conver<sub>g</sub>ence <sub>g</sub>uarantees (Subsection 5.3 and Section 7). Our ke<sub>y</sub> insi<sub>g</sub>ht is to introduce and anal<sub>y</sub>ze a nondecreasing corrected squared norm that controls the cumulative efect of norm decreases without modifying th<sub>e</sub> l<sub>earner</sub>’<sub>s up</sub>d<sub>a</sub>t<sub>es.</sub> W<sub>e</sub> th<sub>ere</sub>b<sub>y re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e s</sub>t<sub>rong guaran</sub>t<sub>ees w</sub>hil<sub>e avo</sub>idi<sub>ng</sub> th<sub>e norm-preserv</sub>i<sub>ng sca</sub>l<sub>ar</sub> com<sub>p</sub>utation at ever<sub>y</sub> iteration re<sub>q</sub>uired b<sub>y</sub> the IR mechanism in IR-PRM<sup>+</sup> (Subsection 5.3). We further show th<sub>a</sub>t<sub>, among var</sub>i<sub>an</sub>t<sub>s o</sub>f $\mathrm { P R M ^ { + } }$ th<sub>a</sub>t <sub>sca</sub>l<sub>e</sub> th<sub>e pre</sub>di<sub>c</sub>ti<sub>on</sub> t<sub>erm</sub> b<sub>y a</sub> fi<sub>xe</sub>d <sub>nonnega</sub>ti<sub>ve mu</sub>lti<sub>p</sub>li<sub>er, on</sub>l<sub>y</sub> h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> satisfies Alt-RVU on ever<sub>y</sub> bounded utilit<sub>y</sub> sequence in ever<sub>y</sub> finite dimension (Section 6).

• An analytical separation between alternating and simultaneous updates. While Cai et al. [2025] rovide numerical evidence that alternatin<sub>g</sub> and simultaneous PRM<sup>+</sup> difer in last-iterate conver<sub>g</sub>ence, we establish such a distinction anal<sub>y</sub>ticall<sub>y</sub> for HPRM<sup>+</sup>. On an ex<sub>p</sub>licit matrix <sub>g</sub>ame, under the standard initialization, Alt-HPRM<sup>+</sup> conver<sub>g</sub>es to a sin<sub>g</sub>le Nash e<sub>q</sub>uilibrium, whereas Sim-HPRM<sup>+</sup> has a dualit<sub>y</sub> <sub>g</sub>a<sub>p</sub> as<sub>y</sub>m<sub>p</sub>toticall<sub>y</sub> bounded awa<sub>y</sub> f<sub>rom zero.</sub> B<sub>y</sub> h<sub>o</sub>ldi<sub>ng</sub> th<sub>e game,</sub> l<sub>earner, an</sub>d i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on</sub> fi<sub>xe</sub>d<sub>,</sub> thi<sub>s separa</sub>ti<sub>on s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>up</sub>d<sub>a</sub>t<sub>e or</sub>d<sub>er a</sub>l<sub>one</sub> <sub>can</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e s</sub>t<sub>ra</sub>t<sub>egy pro</sub>fil<sub>es approac</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>um,</sub> b<sub>eyon</sub>d i<sub>mprov</sub>i<sub>ng</sub> th<sub>e convergence ra</sub>t<sub>e o</sub>f their avera<sub>g</sub>es (Section 8).

• Practical gains from half prediction. We evaluate the CFR instantiation of $\mathsf { A l t { - } H P R M ^ { + } }$ <sub>on</sub> 15 <sub>ex</sub>t<sub>ens</sub>i<sub>ve-</sub>f<sub>orm</sub> <sub>g</sub>ame benchmarks, where the resultin<sub>g</sub> Alt-HPCFR<sup>+</sup> al<sub>g</sub>orithm remains com<sub>p</sub>etitive with leadin<sub>g</sub> solvers across diverse <sub>g</sub>ames. In <sub>p</sub>articular, it im<sub>p</sub>roves over Alt-PCFR<sup>+</sup> in most com<sub>p</sub>arisons of the iterations and learner <sub>wa</sub>ll<sub>-c</sub>l<sub>oc</sub>k ti<sub>me</sub> <sub>nee</sub>d<sub>e</sub>d t<sub>o</sub> <sub>reac</sub>h t<sub>arge</sub>t <sub>exp</sub>l<sub>o</sub>it<sub>a</sub>bilit<sub>y.</sub> It <sub>a</sub>l<sub>so</sub> <sub>ou</sub>t<sub>per</sub>f<sub>orms</sub> $\mathsf { A l t { - } I R { - } P C F R { ^ + } }$ <sub>on mos</sub>t b<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> i<sub>n</sub> b<sub>o</sub>th <sub>measures,</sub> <sub>s</sub>h<sub>ow</sub>i<sub>ng</sub> th<sub>a</sub>t it<sub>s</sub> b<sub>ene</sub>fit<sub>s</sub> <sub>ex</sub>t<sub>en</sub>d b<sub>eyon</sub>d <sub>avo</sub>idi<sub>ng</sub> th<sub>e</sub> IR <sub>sca</sub>l<sub>ar</sub> <sub>compu</sub>t<sub>a</sub>ti<sub>on</sub> t<sub>o</sub> i<sub>mprov</sub>i<sub>ng</sub> th<sub>e</sub> <sub>p</sub>ro<sub>g</sub>ress made <sub>p</sub>er iteration (Section 9).

## 1.2 Other Related Research

I<sub>n</sub> thi<sub>s sec</sub>ti<sub>on we</sub> di<sub>scuss a</sub>dditi<sub>ona</sub>l <sub>re</sub>l<sub>a</sub>t<sub>e</sub>d <sub>researc</sub>h<sub>.</sub>

Concurrent work. Sullins and Kash [2026] independently establish ${ \cal O } ( T ^ { - 1 } )$ <sub>avera e-</sub>it<sub>era</sub>t<sub>e,</sub> ${ \cal O } ( T ^ { - 1 / 2 } )$ b<sub>es</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e</sub> <sub>ra</sub>t<sub>es, an</sub>d <sub>convergence</sub> t<sub>o</sub> th<sub>e</sub> N<sub>as</sub>h <sub>se</sub>t f<sub>or</sub> $\tt T R { - } P R M ^ { + }$ under both simultaneous and alternatin<sub>g</sub> u<sub>p</sub>dates. Our IR-PRM<sup>+</sup> <sub>ra</sub>t<sub>es un</sub>d<sub>er a</sub>lt<sub>erna</sub>ti<sub>ng se</sub>lf<sub>-p</sub>l<sub>ay over</sub>l<sub>ap w</sub>ith th<sub>e</sub>i<sub>rs,</sub> b<sub>u</sub>t <sub>we a</sub>dditi<sub>ona</sub>ll<sub>y es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>convergence o</sub>f $\mathsf { A l t { - } I R { - } P R M ^ { + } }$ to a sin<sub>g</sub>le Nash equilibrium without assumin<sub>g</sub> uniqueness (Theorem 7.5). Our anal tical a<sub>pp</sub>roach difers substantiall f<sub>rom</sub> th<sub>e</sub>i<sub>rs, w</sub>hi<sub>c</sub>h <sub>re</sub>li<sub>es on con</sub>t<sub>ro</sub>lli<sub>ng</sub> th<sub>e coup</sub>l<sub>e</sub>d <sub>grow</sub>th <sub>o</sub>f th<sub>e p</sub>l<sub>ayers</sub>’ <sub>regre</sub>t <sub>vec</sub>t<sub>or norms.</sub> S<sub>pec</sub>ifi<sub>ca</sub>ll<sub>y, our</sub> Alt<sub>-</sub>RVU f<sub>ramewor</sub>k <sub>an</sub>d it<sub>s re</sub>fi<sub>nemen</sub>t i<sub>ns</sub>t<sub>ea</sub>d f<sub>ormu</sub>l<sub>a</sub>t<sub>e common con</sub>diti<sub>ons on</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>regre</sub>t l<sub>earners, separa</sub>ti<sub>ng</sub> th<sub>e convergence argumen</sub>t f<sub>or a</sub>lt<sub>erna</sub>ti<sub>ng se</sub>lf<sub>-p</sub>l<sub>ay</sub> f<sub>rom</sub> th<sub>e a</sub>l<sub>gor</sub>ith<sub>m-spec</sub>ifi<sub>c argumen</sub>t<sub>s es</sub>t<sub>a</sub>bli<sub>s</sub>hi<sub>ng</sub> th<sub>ese con</sub>diti<sub>ons.</sub> Th<sub>ese con</sub>diti<sub>ons a</sub>l<sub>so</sub> h<sub>o</sub>ld f<sub>or</sub> $\mathrm { \tt H P R M ^ { + } }$ d<sub>esp</sub>it<sub>e</sub> d<sub>ecreases</sub> i<sub>n</sub> th<sub>e regre</sub>t <sub>vec</sub>t<sub>or norm</sub> $( i . e .$ ., the lack of the increasing-regret mechanism). Thus, our anal<sub>y</sub>sis extends be<sub>y</sub>ond increasin<sub>g</sub>-re<sub>g</sub>ret methods and shows that norm monotonicit<sub>y</sub> is <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>t b<sub>u</sub>t <sub>no</sub>t <sub>necessary</sub> f<sub>or</sub> th<sub>ese convergence guaran</sub>t<sub>ees.</sub>

Regret matching family. Connections to online optimization have provided a basis for developing new regretmatchin<sub>g</sub> methods. Within Blackwell a<sub>pp</sub>roachabilit<sub>y</sub>, Farina et al. [2021] recover RM and $\mathbb { R } \mathbb { M } ^ { + }$ f<sub>rom</sub> f<sub>o</sub>ll<sub>ow-</sub>th<sub>e-</sub> <sub>regu</sub>l<sub>ar</sub>i<sub>ze</sub>d<sub>-</sub>l<sub>ea</sub>d<sub>er an</sub>d <sub>on</sub>li<sub>ne m</sub>i<sub>rror</sub> d<sub>escen</sub>t<sub>, respec</sub>ti<sub>ve</sub>l<sub>y, an</sub>d d<sub>er</sub>i<sub>ve pre</sub>di<sub>c</sub>ti<sub>ve var</sub>i<sub>an</sub>t<sub>s</sub> th<sub>roug</sub>h th<sub>e</sub>i<sub>r op</sub>ti<sub>m</sub>i<sub>s</sub>ti<sub>c</sub> counter<sub>p</sub>arts. Other work stren<sub>g</sub>thens conver<sub>g</sub>ence <sub>g</sub>uarantees throu<sub>g</sub>h stabilization and extra<sub>g</sub>radient u<sub>p</sub>dates [Farina et al., 2023, Cai et al., 2025], or throu<sub>g</sub>h the increasin<sub>g</sub>-re<sub>g</sub>ret mechanism [Zhan<sub>g</sub> et al., 2025]. Within CFR, com<sub>p</sub>lementar<sub>y</sub> develo<sub>p</sub>ments include discountin<sub>g p</sub>ast re<sub>g</sub>rets [Brown and Sandholm, 2019], combinin<sub>g p</sub>rediction with wei<sub>g</sub>hted re<sub>g</sub>ret accumulation [Xu et al., 2024], usin<sub>g</sub> as<sub>y</sub>mmetric ste<sub>p</sub> sizes for re<sub>g</sub>ret accumulation and <sub>p</sub>rediction [Men<sub>g</sub> et al., 2026], and introducin<sub>g</sub> time-var<sub>y</sub>in<sub>g</sub> schedules for discountin<sub>g</sub> and strate<sub>gy</sub> avera<sub>g</sub>in<sub>g</sub> [Zhan<sub>g</sub> et al., 2026].

Alternating updates. Alternating update order has also been studied extensively for gradient methods in uncon-<sub>s</sub>t<sub>ra</sub>i<sub>ne</sub>d bili<sub>near games, revea</sub>li<sub>ng</sub> dif<sub>erences</sub> i<sub>n s</sub>t<sub>a</sub>bilit<sub>y an</sub>d <sub>convergence</sub> b<sub>e</sub>t<sub>ween s</sub>i<sub>mu</sub>lt<sub>aneous an</sub>d <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> d<sub>y</sub>namics [Gidel et al., 2019, Baile<sub>y</sub> et al., 2020, Zhan<sub>g</sub> and Yu, 2020]. On constrained domains, anal<sub>y</sub>ses of alternating mirror descent and projected gradient descent–ascent establish improved average-iterate convergence guarantees under their res<sub>p</sub>ective assum<sub>p</sub>tions [Wibisono et al., 2022, Katona et al., 2026, Nan et al., 2026b]. From an onlinel<sub>earn</sub>i<sub>ng perspec</sub>ti<sub>ve, a</sub>lt<sub>erna</sub>ti<sub>ng regre</sub>t <sub>prov</sub>id<sub>es a way</sub> t<sub>o accoun</sub>t f<sub>or</sub> th<sub>e up</sub>d<sub>a</sub>t<sub>e sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e an</sub>d <sub>o</sub>bt<sub>a</sub>i<sub>n s</sub>t<sub>ronger a</sub>d<sub>versar</sub>i<sub>a</sub>l <sub>g</sub>uarantees in online linear and convex o<sub>p</sub>timization [Cevher et al., 2023, Hait et al., 2025, Tao and Zhen<sub>g</sub>, 2026]. These im<sub>p</sub>rovements are al<sub>g</sub>orithm-de<sub>p</sub>endent: Hait et al. [2025] also <sub>p</sub>rove $\Omega ( { \sqrt { T } } )$ <sub>a</sub>lt<sub>erna</sub>ti<sub>ng-regre</sub>t l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> f<sub>or</sub> $\mathrm { P R M ^ { + } }$ <sub>an</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>s</sub>ti<sub>c</sub> <sub>gra</sub>di<sub>en</sub>t <sub>me</sub>th<sub>o</sub>d<sub>s.</sub>

## 2 Preliminaries on Zero-Sum Games

F<sub>o</sub>r <sub>a</sub>n int<sub>ege</sub>r $d \geq 1 ,$ <sub>,</sub> l<sub>e</sub>t $[ d ] : = \{ 1 , \ldots , d \}$ <sub>an</sub>d l<sub>e</sub>t $\Delta ( d ) : = \{ \pmb { x } \in \mathbb { R } _ { > 0 } ^ { d } : \langle \pmb { 1 } , \pmb { x } \rangle = 1 \}$ d<sub>eno</sub>t<sub>e</sub> th<sub>e ro</sub>b<sub>a</sub>bilit <sub>s</sub>i<sub>m</sub> l<sub>ex.</sub> W<sub>e</sub> <sub>cons</sub>id<sub>er a</sub> t<sub>wo-p</sub>l<sub>ayer zero-sum game w</sub>ith <sub>payo</sub>f <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ <sup>and</sup> <sup>strate</sup>gy <sup>s</sup>p<sup>aces</sup> $\chi = \Delta ( m )$ <sub>an</sub>d $y = \Delta ( n )$ <sub>.</sub> Th<sub>e</sub> row <sub>p</sub><sup>l</sup>a<sub>y</sub>er c<sup>h</sup>ooses $\boldsymbol { x } \in \boldsymbol { \chi }$ t<sub>o max</sub>i<sub>m</sub>i<sub>ze</sub> $\langle { \pmb x } , { \bf A } { \bf y } \rangle$ <sub>, w</sub>hil<sub>e</sub> th<sub>e co</sub>l<sub>umn p</sub>l<sub>ayer c</sub>h<sub>ooses</sub> $y \in \mathcal { Y }$ t<sub>o m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>ze</sub> it<sub>.</sub> At <sub>a s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>pro</sub>fil<sub>e</sub> $( x , y ) \in X \times y$ <sub>,</sub> th<sub>e</sub>i<sub>r u</sub>tilit<sub>y vec</sub>t<sub>ors are</sub> $\mathbf { A } \mathbf { y }$ <sub>an</sub>d $- \mathbf { A } ^ { \top } \pmb { x }$ , res<sub>p</sub>ect<sup>i</sup>ve<sup>l</sup><sub>y</sub>.

W<sub>e</sub> <sub>measure</sub> <sub>su</sub>b<sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> i<sub>n</sub> th<sub>e</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>way</sub> <sub>as</sub> th<sub>e</sub> d<sub>ua</sub>lit<sub>y</sub> <sub>gap:</sub>

$$
\mathrm { G a p } ( \pmb { x } , \pmb { y } ) : = \operatorname* { m a x } _ { \pmb { x } ^ { \prime } \in \mathcal { X } } \langle \pmb { x } ^ { \prime } , \mathbf { A } \pmb { y } \rangle - \operatorname* { m i n } _ { \pmb { y } ^ { \prime } \in \mathcal { Y } } \langle \pmb { y } ^ { \prime } , \mathbf { A } ^ { \top } \pmb { x } \rangle = \operatorname* { m a x } _ { i \in [ m ] } ( \mathbf { A } \pmb { y } ) _ { i } - \operatorname* { m i n } _ { j \in [ n ] } ( \mathbf { A } ^ { \top } \pmb { x } ) _ { j } .\tag{1}
$$

Th<sub>e</sub> <sub>gap</sub> i<sub>s</sub> <sub>a</sub>l<sub>ways</sub> <sub>nonnega</sub>ti<sub>ve</sub> <sub>an</sub>d <sub>van</sub>i<sub>s</sub>h<sub>es</sub> <sub>exac</sub>tl<sub>y</sub> <sub>a</sub>t N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>a.</sub> W<sub>e</sub> d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> <sub>se</sub>t b<sub>y</sub> $z ^ { \ast }$ F<sub>or</sub> $T \geq 1$ <sub>, we</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e un</sub>if<sub>orm</sub>l<sub>y average</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es</sub>

$$
\bar { \pmb { x } } ^ { ( T ) } : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \pmb { x } ^ { ( t ) } , ~ \bar { \pmb { y } } ^ { ( T ) } : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \pmb { y } ^ { ( t ) } .
$$

W<sub>e</sub> <sub>measure</sub> <sub>average-,</sub> b<sub>es</sub>t<sub>-,</sub> <sub>an</sub>d l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e</sub> <sub>convergence</sub> b<sub>y</sub>

$$
\mathrm { G a p } ( \bar { \boldsymbol { x } } ^ { ( T ) } , \bar { \boldsymbol { y } } ^ { ( T ) } ) , \operatorname* { m i n } _ { 1 \leq t \leq T } \mathrm { G a p } ( \boldsymbol { x } ^ { ( t ) } , \boldsymbol { y } ^ { ( t ) } ) , \mathrm { G a p } ( \boldsymbol { x } ^ { ( T ) } , \boldsymbol { y } ^ { ( T ) } ) ,
$$

<sub>respec</sub>ti<sub>ve</sub>l<sub>y.</sub> E<sub>ac</sub>h <sub>no</sub>ti<sub>on requ</sub>i<sub>res</sub> th<sub>e correspon</sub>di<sub>ng quan</sub>tit<sub>y</sub> t<sub>o van</sub>i<sub>s</sub>h <sub>as</sub> $T \to \infty$ <sub>.</sub> Si<sub>nce</sub> $\chi _ { \times } y$ i<sub>s compac</sub>t <sub>an</sub>d th<sub>e</sub> d<sub>ua</sub>lit<sub>y gap</sub> i<sub>s con</sub>ti<sub>nuous, van</sub>i<sub>s</sub>hi<sub>ng</sub> l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e gap</sub> i<sub>s equ</sub>i<sub>va</sub>l<sub>en</sub>t t<sub>o</sub> th<sub>e</sub> di<sub>s</sub>t<sub>ance o</sub>f $( \mathbf x ^ { ( T ) } , \mathbf y ^ { ( T ) } )$ to $z ^ { \ast }$ t<sub>en</sub>di<sub>ng</sub> t<sub>o zero.</sub> Thi<sub>s</sub> d<sub>oes no</sub>t<sub>,</sub> b<sub>y</sub> it<sub>se</sub>lf<sub>,</sub> i<sub>mp</sub>l<sub>y</sub> th<sub>e s</sub>t<sub>ronger proper</sub>t<sub>y</sub> th<sub>a</sub>t th<sub>e</sub> it<sub>era</sub>t<sub>es converge</sub> t<sub>o a s</sub>i<sub>ng</sub>l<sub>e</sub> N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>um, name</sub>l<sub>y</sub> th<sub>a</sub>t th<sub>ere ex</sub>i<sub>s</sub>t<sub>s</sub> $( \pmb { x } ^ { * } , \pmb { y } ^ { * } ) \in \mathcal { Z } ^ { * }$ <sub>suc</sub>h th<sub>a</sub>t $( { \pmb x } ^ { ( T ) } , \mathbf { \hat { y } } ^ { ( \bar { T } ) } ) \xrightarrow { } ( { \pmb x } ^ { * } , { \pmb y } ^ { * } )$

## 3 Regret Matching: Old and New

I<sub>n</sub> thi<sub>s</sub> <sub>wor</sub>k<sub>,</sub> <sub>we</sub> <sub>s</sub>t<sub>u</sub>d<sub>y</sub> <sub>op</sub>ti<sub>m</sub>i<sub>s</sub>ti<sub>c</sub> <sub>regre</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zers</sub> <sub>on</sub> th<sub>e</sub> <sub>s</sub>i<sub>mp</sub>l<sub>ex</sub> $\Delta ( d )$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> � i<sub>s</sub> th<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>ac</sub>ti<sub>ons.</sub> At ti<sub>me</sub> �<sub>,</sub> <sup>a</sup> <sup>learner</sup> p<sup>la</sup>y<sup>s</sup> <sup>a</sup> <sup>strate</sup>gy ${ \pmb x } ^ { ( t ) } \in \Delta ( d )$ <sub>an</sub>d th<sub>en</sub> <sub>o</sub>b<sub>serves</sub> <sub>a</sub> <sub>u</sub>tilit<sub>y</sub> <sub>vec</sub>t<sub>or</sub> ${ \pmb u } ^ { ( t ) } \in \mathbb { R } ^ { d }$ <sub>.</sub> Th<sub>e</sub> <sub>regre</sub>t <sub>a</sub>ft<sub>er</sub> � <sub>o</sub>b<sub>serva</sub>ti<sub>ons</sub> i<sub>s</sub>

$$
\mathrm { R e g } ( T ) : = \operatorname* { m a x } _ { \pm \mathbf { x } \in \Delta ( d ) } \sum _ { t = 1 } ^ { T } \langle \pmb { u } ^ { ( t ) } , \pmb { x } - \pmb { x } ^ { ( t ) } \rangle .
$$

W<sub>e</sub> l<sub>e</sub>t $\tilde { r } ^ { ( t ) } \in \mathbb { R } _ { \geq 0 } ^ { d }$ be the regret vector after the first $t - 1$ observations. The prediction step forms the prediction regret vector $\boldsymbol { r } ^ { ( t ) }$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>may</sub> h<sub>ave</sub> <sub>nega</sub>ti<sub>ve</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es.</sub> Th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> i<sub>s</sub>

$$
\pmb { x } ^ { ( t ) } = \frac { [ \pmb { r } ^ { ( t ) } ] _ { + } } { \Vert [ \pmb { r } ^ { ( t ) } ] _ { + } \Vert _ { 1 } } ,\tag{2}
$$

where [·]<sub>+</sub> denotes the coordinatewise <sub>p</sub>ositive <sub>p</sub>art. If the denominator is zero, we set ${ \pmb x } ^ { ( t ) } = { \bf 1 } / d$

W<sub>e</sub> d<sub>eno</sub>t<sub>e</sub> th<sub>e pre</sub>di<sub>c</sub>ti<sub>on an</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>on rou</sub>ti<sub>nes</sub> b<sub>y</sub> N<sub>ext</sub>S<sub>trategy an</sub>d O<sub>bserve</sub>U<sub>tility, respec</sub>ti<sub>ve</sub>l<sub>y.</sub> Th<sub>e</sub> f<sub>ormer</sub> t<sub>a</sub>k<sub>es a u</sub>tilit<sub>y pre</sub>di<sub>c</sub>ti<sub>on</sub> $\pmb { m } ^ { ( t ) } \in \mathbb { R } ^ { d }$ an<sup>d</sup> returns t<sup>h</sup>e strate<sub>gy</sub> $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ <sub>;</sub> th<sub>e</sub> l<sub>a</sub>tt<sub>er rece</sub>i<sub>ves</sub> th<sub>e o</sub>b<sub>serve</sub>d <sub>u</sub>tilit<sub>y vec</sub>t<sub>or</sub> $\mathbf { \boldsymbol { u } } ^ { ( t ) }$ <sub>an</sub>d <sub>up</sub>d<sub>a</sub>t<sub>es</sub> th<sub>e ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> t<sub>o</sub> $\tilde { r } ^ { ( t + 1 ) }$ <sub>.</sub> U<sub>n</sub>l<sub>ess s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>o</sub>th<sub>erw</sub>i<sub>se, we use</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on</sub>

$$
\tilde { \pmb { r } } ^ { ( 1 ) } = \pmb { r } ^ { ( 1 ) } = \pmb { m } ^ { ( 1 ) } = \pmb { 0 } , \pmb { x } ^ { ( 1 ) } = \pmb { 1 } / d ,
$$

<sub>an</sub>d <sub>use</sub> th<sub>e mos</sub>t <sub>recen</sub>tl<sub>y o</sub>b<sub>serve</sub>d <sub>u</sub>tilit<sub>y as</sub> th<sub>e nex</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>on,</sub> th<sub>a</sub>t i<sub>s,</sub> $\pmb { m } ^ { ( t + 1 ) } = \pmb { u } ^ { ( t ) }$ f<sub>or</sub> $t \geq 1$

Prediction. Predictive regret matching<sup>+</sup> (PRM<sup>+</sup>) [Farina et al., 2021] updates its regret vector according to

$$
\tilde { r } ^ { ( t + 1 ) } = [ \tilde { r } ^ { ( t ) } + u ^ { ( t ) } - \langle u ^ { ( t ) } , x ^ { ( t ) } \rangle \mathbf { 1 } ] _ { + } .\tag{3}
$$

Th<sub>e</sub> <sub>vec</sub>t<sub>or</sub> ${ \pmb u } ^ { ( t ) } - \langle { { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } } \rangle 1$ is the instantaneous regret vector. With the last observed utility as the prediction, the <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> i<sub>s</sub>

$$
\pmb { r } ^ { ( t + 1 ) } = \tilde { \pmb { r } } ^ { ( t + 1 ) } + \pmb { u } ^ { ( t ) } - \langle \pmb { u } ^ { ( t ) } , \pmb { x } ^ { ( t ) } \rangle \pmb { 1 } .\tag{4}
$$

<sup>Th</sup>e next strate<sub>gy</sub> $\boldsymbol { x } ^ { ( t + 1 ) }$ i<sub>s o</sub>bt<sub>a</sub>i<sub>ne</sub>d b<sub>y app</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e norma</sub>li<sub>za</sub>ti<sub>on ru</sub>l<sub>e</sub> $( 2 ) \tan r ^ { ( t + 1 ) }$

Scaled-prediction. We consider a general family of regret minimizers which we call scaled-prediction regret matching<sup>+</sup>. The family is parameterized by a fixed scalar $\lambda \in \mathbb { R } _ { \geq 0 }$ <sub>.</sub> E<sub>ac</sub>h <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> i<sub>n</sub> thi<sub>s</sub> f<sub>am</sub>il<sub>y</sub> <sub>uses</sub> th<sub>e</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> u<sub>p</sub>date (3) and forms the scaled-<sub>p</sub>rediction re<sub>g</sub>ret vector

$$
\pmb { r } ^ { ( t + 1 ) } = \tilde { \pmb { r } } ^ { ( t + 1 ) } + \lambda ( \pmb { u } ^ { ( t ) } - \langle \pmb { u } ^ { ( t ) } , \pmb { x } ^ { ( t ) } \rangle \pmb { 1 } ) .
$$

Th<sub>e canon</sub>i<sub>ca</sub>l <sub>c</sub>h<sub>o</sub>i<sub>ces</sub> $\lambda = 0$ <sub>an</sub>d $\lambda = 1$ recover $\mathbb { R } \mathbb { M } ^ { + }$ [Tammelin, 2014] and $\mathrm { P R M ^ { + } }$ , res<sub>p</sub>ect<sup>i</sup>ve<sup>l</sup><sub>y</sub>.

F<sub>or</sub> $\lambda \in ( 0 , 1 ]$ these u dates coincide with the local as mmetric u dates of Men et al. [2026] when their arameter i<sub>s</sub> fi<sub>xe</sub>d <sub>a</sub>t $\alpha = \lambda ^ { - 1 } - 1$ <sub>.</sub> I<sub>n par</sub>ti<sub>cu</sub>l<sub>ar,</sub> $\lambda = 1 / 3$ <sub>recovers</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>regre</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zer use</sub>d b<sub>y</sub> th<sub>e</sub>i<sub>r s</sub>i<sub>mp</sub>l<sub>e asymme</sub>t<sub>r</sub>i<sub>c</sub> $\mathsf { P C F R ^ { + } \Sigma ( S A P C F R ^ { + } ) }$ <sub>, w</sub>hi<sub>c</sub>h fi<sub>xes</sub> $\alpha = 2$ <sub>.</sub> F<sub>or every</sub> fi<sub>xe</sub>d $\lambda \in \mathsf { \Gamma } ( 0 , 1 ]$ , Men<sub>g</sub> et al. [2026, Theorem 4.1] shows that $\mathrm { R e g } ( T ) = O ( { \sqrt { T } } )$ <sub>on</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>u</sub>tilit<sub>y sequences.</sub> Thi<sub>s</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>regre</sub>t l<sub>earner guaran</sub>t<sub>ee a</sub>l<sub>so app</sub>li<sub>es un</sub>d<sub>er a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>up</sub>d<sub>a</sub>t<sub>es,</sub> b<sub>u</sub>t d<sub>o</sub> <sub>no</sub>t b<sub>y</sub> th<sub>emse</sub>l<sub>ves</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>average-</sub>it<sub>era</sub>t<sub>e,</sub> b<sub>es</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e,</sub> <sub>or</sub> l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e</sub> <sub>convergence</sub> t<sub>o</sub> N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> <sub>un</sub>d<sub>er a</sub>lt<sub>erna</sub>ti<sub>ng se</sub>lf<sub>-p</sub>l<sub>ay.</sub>

Half-prediction. In this paper, we mainly focus on the choice $\lambda = 1 / 2 ,$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>g</sub>i<sub>ves</sub>

$$
\pmb { r } ^ { ( t + 1 ) } = \tilde { \pmb { r } } ^ { ( t + 1 ) } + \frac { 1 } { 2 } ( \pmb { u } ^ { ( t ) } - \langle \pmb { u } ^ { ( t ) } , \pmb { x } ^ { ( t ) } \rangle \pmb { 1 } ) .\tag{5}
$$

We call this specialization half-prediction regret matching<sup>+</sup>, abbreviated as HPRM<sup>+</sup>, as it halves the prediction while kee<sub>p</sub>in<sub>g</sub> the re<sub>g</sub>ret vector u<sub>p</sub>date (3) identical to PRM<sup>+</sup>. Subsection 5.3 and Section 6 will establish its theoretical <sub>guaran</sub>t<sub>ees an</sub>d <sub>exam</sub>i<sub>ne</sub> th<sub>e ro</sub>l<sub>e o</sub>f th<sub>e c</sub>h<sub>o</sub>i<sub>ce</sub> $\lambda = 1 / 2$ <sub>w</sub>ithi<sub>n</sub> th<sub>e</sub> <sub>sca</sub>l<sub>e</sub>d<sub>-pre</sub>di<sub>c</sub>ti<sub>on</sub> f<sub>am</sub>il<sub>y.</sub>

Increasing regret. Increasing-regret predictive regret matching $^ { + } _ { , } ( \mathtt { I R - P R M } ^ { + } )$ [Zhan<sub>g</sub> et al., 2025] forms its <sub>p</sub>rediction <sub>regre</sub>t <sub>vec</sub>t<sub>or us</sub>i<sub>ng a sca</sub>l<sub>ar s</sub>hift th<sub>a</sub>t <sub>preserves</sub> th<sub>e</sub> E<sub>uc</sub>lid<sub>ean norm a</sub>ft<sub>er</sub> t<sub>a</sub>ki<sub>ng</sub> th<sub>e pos</sub>iti<sub>ve par</sub>t<sub>.</sub> Wh<sub>en</sub> $\tilde { \mathbf { \Xi } } \tilde { r } ^ { ( \bar { t } ) } \neq \mathbf { 0 }$ , we l<sub>e</sub>t $\gamma ^ { ( t ) } \in \mathbb { R }$ b<sub>e</sub> th<sub>e un</sub>i<sub>que sca</sub>l<sub>ar sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub>

$$
\| [ \tilde { r } ^ { ( t ) } + m ^ { ( t ) } - \gamma ^ { ( t ) } \mathbf { 1 } ] _ { + } \| _ { 2 } = \| \tilde { r } ^ { ( t ) } \| _ { 2 } ,\tag{6}
$$

<sub>an</sub>d f<sub>orm</sub> th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>as</sub>

$$
\pmb { r } ^ { ( t ) } = \tilde { \pmb { r } } ^ { ( t ) } + \pmb { m } ^ { ( t ) } - \gamma ^ { ( t ) } \mathbf { 1 } .
$$

<sup>Th</sup>e strate<sub>gy</sub> $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ is then obtained b<sub>y</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the normalization rule (2). After observin<sub>g</sub> $\mathbf { \boldsymbol { u } } ^ { ( t ) }$ <sub>,</sub> th<sub>e</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> i<sub>s</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub>d <sub>accor</sub>di<sub>ng</sub> t<sub>o</sub>

$$
\tilde { \pmb { r } } ^ { ( t + 1 ) } = [ \tilde { \pmb { r } } ^ { ( t ) } + \pmb { u } ^ { ( t ) } - ( \gamma ^ { ( t ) } + \langle \pmb { u } ^ { ( t ) } - \pmb { m } ^ { ( t ) } , \pmb { x } ^ { ( t ) } \rangle ) \pmb { 1 } ] _ { + } .\tag{7}
$$

Wh<sub>en</sub> $\tilde { r } ^ { ( t ) } = \mathbf { 0 } .$ <sub>,</sub> <sub>we</sub> <sub>overr</sub>id<sub>e</sub> th<sub>e</sub> <sub>u</sub>tilit<sub>y</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> b<sub>y</sub> <sub>se</sub>tti<sub>ng</sub> $\pmb { m } ^ { ( t ) } = \mathbf { 0 }$ , set $\mathbf { \boldsymbol { r } } ^ { ( t ) } = \mathbf { \boldsymbol { 0 } }$ <sub>an</sub>d $\gamma ^ { ( t ) } = 0 .$ <sub>, an</sub>d <sub>use</sub> th<sub>e un</sub>if<sub>orm</sub> <sup>strate</sup>gy ${ \pmb x } ^ { ( t ) } = { \pmb 1 } / d$ <sub>.</sub> Aft<sub>er</sub> <sub>o</sub>b<sub>serv</sub>i<sub>ng</sub> $\mathbf { \boldsymbol { u } } ^ { ( t ) }$ , we a<sub>pp</sub>l<sub>y</sub> (7) with these values. These u<sub>p</sub>dates ensure that the Euclidean <sub>norm</sub> <sub>o</sub>f th<sub>e</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> i<sub>s</sub> <sub>non</sub>d<sub>ecreas</sub>i<sub>ng:</sub>

$$
\| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } \geq \| \tilde { r } ^ { ( t ) } \| _ { 2 } .
$$

The extra<sub>g</sub>radient s<sub>p</sub>ecialization IREG-PRM<sup>+</sup> of Zhan<sub>g</sub> et al. [2025] instead uses <sub>p</sub>redictions obtained from an additional <sub>u</sub>tilit<sub>y eva</sub>l<sub>ua</sub>ti<sub>on.</sub> W<sub>e</sub> d<sub>o no</sub>t <sub>s</sub>t<sub>u</sub>d<sub>y</sub> thi<sub>s spec</sub>i<sub>a</sub>li<sub>za</sub>ti<sub>on</sub> h<sub>ere, as we</sub> f<sub>ocus on a</sub>lt<sub>erna</sub>ti<sub>ng up</sub>d<sub>a</sub>t<sub>es w</sub>ith l<sub>as</sub>t<sub>-u</sub>tilit<sub>y</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>ons.</sub>

## 4 Alternating vs Simultaneous Updates

I<sub>n zero-sum games, se</sub>lf<sub>-p</sub>l<sub>ay up</sub>d<sub>a</sub>t<sub>es can</sub> b<sub>e con</sub>d<sub>uc</sub>t<sub>e</sub>d <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub>l<sub>y or</sub> b<sub>y a</sub>lt<sub>erna</sub>ti<sub>ng</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e agen</sub>t<sub>s.</sub> L<sub>e</sub>t $\mathcal { R } _ { \boldsymbol { X } }$ <sub>an</sub>d $\mathcal { R } _ { \boldsymbol { y } }$ b<sub>e</sub> th<sub>e</sub> l<sub>earners o</sub>f th<sub>e row p</sub>l<sub>ayer an</sub>d <sub>co</sub>l<sub>umn p</sub>l<sub>ayer, respec</sub>ti<sub>ve</sub>l<sub>y,</sub> b<sub>o</sub>th <sub>us</sub>i<sub>ng</sub> th<sub>e same regre</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zer</sub> d<sub>e</sub>fi<sub>ne</sub>d above. Al<sub>g</sub>orithms 1 and 2 s<sub>p</sub>ecif<sub>y</sub> the simultaneous and row-first alternatin<sub>g</sub> u<sub>p</sub>dates, denoted b<sub>y</sub> the <sub>p</sub>refixes Sim-<sub>an</sub>d $\mathsf { A l t - } ,$ <sub>respec</sub>ti<sub>ve</sub>l<sub>y.</sub> Si<sub>mu</sub>lt<sub>aneous up</sub>d<sub>a</sub>t<sub>es compu</sub>t<sub>e</sub> $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ <sub>an</sub>d $\mathbf { y } ^ { ( t ) }$ <sub>us</sub>i<sub>ng</sub> th<sub>e respec</sub>ti<sub>ve u</sub>tilit<sub>y pre</sub>di<sub>c</sub>ti<sub>ons</sub> $\mathbf { A } \mathbf { y } ^ { ( t - 1 ) }$ <sub>an</sub>d $- \mathbf { A } ^ { \top } \pmb { x } ^ { ( t - \bar { 1 } ) }$ <sub>,</sub> b<sub>e</sub>f<sub>ore o</sub>b<sub>serv</sub>i<sub>ng</sub> $\mathbf { A } \mathbf { y } ^ { ( t ) }$ <sub>an</sub>d $- \mathbf { A } ^ { \top } \hat { \mathbf { x } ^ { ( t ) } }$ <sub>.</sub> U<sub>n</sub>d<sub>er row-</sub>fi<sub>rs</sub>t <sub>a</sub>lt<sub>erna</sub>ti<sub>ng up</sub>d<sub>a</sub>t<sub>es,</sub> th<sub>e row p</sub>l<sub>ayer</sub> lik<sub>ew</sub>i<sub>se</sub> <sup>com</sup>pu<sup>tes</sup> $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ us<sup>i</sup>n<sub>g</sub> $\mathbf { A } \mathbf { y } ^ { ( t - 1 ) }$ <sub>,</sub> b<sub>u</sub>t th<sub>e co</sub>l<sub>umn p</sub>l<sub>ayer</sub> fi<sub>rs</sub>t <sub>o</sub>b<sub>serves</sub> $- \mathbf { A } ^ { \top } \pmb { x } ^ { ( t ) }$ <sub>w</sub>hil<sub>e</sub> <sub>p</sub>l<sub>ay</sub>i<sub>ng</sub> $\mathbf { y } ^ { ( t - 1 ) }$ , u<sub>p</sub><sup>d</sup>ates <sup>i</sup>ts re<sub>g</sub>ret <sub>vec</sub>t<sub>or,</sub> <sub>an</sub>d <sub>uses</sub> thi<sub>s</sub> <sub>same</sub> <sub>u</sub>tilit<sub>y</sub> <sub>as</sub> th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> f<sub>or</sub> <sub>compu</sub>ti<sub>ng</sub> $\mathbf { y } ^ { ( t ) }$ <sub>.</sub> Th<sub>e</sub> <sub>row</sub> <sub>p</sub>l<sub>ayer</sub> th<sub>en</sub> <sub>o</sub>b<sub>serves</sub> $\mathbf { A } \mathbf { y } ^ { ( t ) }$ <sub>w</sub>hil<sub>e</sub> p<sup>la</sup>y<sup>in</sup>g $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ <sub>an</sub>d <sub>up</sub>d<sub>a</sub>t<sub>es</sub> it<sub>s regre</sub>t <sub>vec</sub>t<sub>or.</sub> W<sub>e</sub> f<sub>ocus on row-</sub>fi<sub>rs</sub>t <sub>a</sub>lt<sub>erna</sub>ti<sub>ng up</sub>d<sub>a</sub>t<sub>es,</sub> h<sub>erea</sub>ft<sub>er re</sub>f<sub>erre</sub>d t<sub>o as a</sub>lt<sub>erna</sub>ti<sub>ng</sub> updates, since the column-first case follows readily by exchanging the players and replacing A with $- \mathbf { A } ^ { \top }$

Learner indices count local rounds: the normalization rule (2) ma<sub>p</sub>s $r _ { X } ^ { ( t ) }$ to $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ <sub>un</sub>d<sub>er</sub> b<sub>o</sub>th <sub>sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>es an</sub>d $r _ { y } ^ { ( t ) }$ to $\mathbf { y } ^ { ( t ) }$ <sub>un</sub>d<sub>er s</sub>i<sub>mu</sub>lt<sub>aneous up</sub>d<sub>a</sub>t<sub>es or</sub> $\mathbf { y } ^ { ( t - 1 ) }$ <sub>un</sub>d<sub>er a</sub>lt<sub>erna</sub>ti<sub>ng up</sub>d<sub>a</sub>t<sub>es.</sub> Th<sub>e</sub> fi<sub>na</sub>l li<sub>ne o</sub>f Al<sub>gor</sub>ith<sub>m</sub> 2 <sub>supp</sub>li<sub>es</sub> $\pmb { x } ^ { ( T + 1 ) }$ for alternatin<sub>g</sub> re<sub>g</sub>ret (Subsection 5.1) without additional feedback. Under both schedules, the avera<sub>g</sub>es use onl<sub>y</sub> the <sub>ou</sub>t<sub>pu</sub>t <sub>pro</sub>fil<sub>es</sub> $( \mathbf x ^ { ( t ) } , \mathbf y ^ { ( t ) } )$ f<sub>or</sub> $1 \leq t \leq T$

Algorithm 1: Simultaneous Updates Algorithm 2: Row-first Alternating Updates   
Input: Matrix A; horizon �; learners $\overline { { \mathcal { R } _ { X } , \mathcal { R } _ { y } } }$ Input: Matrix A; horizon �; learners $\overline { { \mathcal { R } _ { X } , \mathcal { R } _ { y } } }$   
1 $\pmb { \iota } _ { \chi } ^ { ( 0 ) }  \mathbf { 0 }$ 1 ${ \pmb u } _ { X } ^ { ( 0 ) }  { \pmb 0 }$   
2 ${ \pmb u } _ { y } ^ { ( 0 ) }  { \pmb 0 }$ 2 ${ \pmb u } _ { y } ^ { ( 0 ) }  { \pmb 0 }$   
3 for $t = 1 , 2 , \dots , T$ do 3 $\mathbf { y } ^ { ( 0 ) }  \mathcal { R } _ { y }$ <sub>.</sub>N<sub>ext</sub>S<sub>tr</sub>a<sub>tegy</sub> $( \pmb { u } _ { \pmb { y } } ^ { ( 0 ) } )$   
4 $\pmb { x } ^ { ( t ) }  \pmb { \mathcal { R } } _ { X }$ <sub>.</sub>N<sub>ext</sub>S<sub>tr</sub>a<sub>tegy</sub> $( \pmb { u } _ { \chi } ^ { ( t - 1 ) } )$ 4 for $t = 1 , 2 , \dots , T$ do   
5 $\mathbf { y } ^ { ( t ) } \gets \mathcal { R } _ { y }$ <sub>.</sub>N<sub>ext</sub>S<sub>tr</sub>a<sub>tegy</sub> $( \pmb { u } _ { \pmb { y } } ^ { ( t - 1 ) } )$ 5 $\pmb { x } ^ { ( t ) }  \pmb { \mathcal { R } } _ { X }$ <sub>.</sub>N<sub>ext</sub>S<sub>tr</sub>a<sub>tegy</sub> $( \pmb { u } _ { \chi } ^ { ( t - 1 ) } )$   
6 ${ \pmb u } _ { X } ^ { ( t ) }  { \bf A } { \bf y } ^ { ( t ) }$ 6 $\pmb { u } _ { y } ^ { ( t ) }  - \mathbf { A } ^ { \top } \pmb { x } ^ { ( t ) }$   
7 $\pmb { u } _ { y } ^ { ( t ) }  - \mathbf { A } ^ { \top } \pmb { x } ^ { ( t ) }$ 7 $\mathcal { R } _ { y . \mathrm { O B S E R V E U T I L I T Y } } ( \pmb { u } _ { y } ^ { ( t ) } )$   
8 $\mathcal { R } _ { X . } \mathrm { O B s E R V E U T I L I T Y } ( \pmb { u } _ { \chi } ^ { ( t ) } )$ 8 $\mathbf { y } ^ { ( t ) }  \mathcal { R } _ { y }$ <sub>.</sub>N<sub>ext</sub>S<sub>tr</sub>a<sub>tegy</sub> $( \pmb { u } _ { \pmb { y } } ^ { ( t ) } )$   
9 $\mathcal { R } _ { { y } . \mathrm { O B S E R V E U T I L I T Y } } ( u _ { y } ^ { ( t ) } )$ 9 ${ \pmb u } _ { \chi } ^ { ( t ) }  { \bf A } { \bf y } ^ { ( t ) }$   
10 $\widetilde { \mathcal { R } _ { X } } . \mathrm { O B s s E R V E U T I L I T Y } ( \pmb { u } _ { \chi } ^ { ( t ) } )$   
11 $\dot { \mathbf { x } ^ { ( T + 1 ) } } \gets \mathcal { R } _ { \boldsymbol { X } . \mathrm { N E X T S T R A T E G Y } } ( \mathbf { \boldsymbol { u } } _ { \boldsymbol { X } } ^ { ( T ) } )$

## 5 Average-Iterate Convergence

A<sub>verage-</sub>it<sub>era</sub>t<sub>e</sub> <sub>convergence</sub> <sub>quan</sub>tifi<sub>es</sub> h<sub>ow</sub> <sub>qu</sub>i<sub>c</sub>kl<sub>y</sub> <sub>se</sub>lf<sub>-p</sub>l<sub>ay</sub> <sub>pro</sub>d<sub>uces</sub> <sub>an</sub> <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub> N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>um,</sub> <sub>even</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>pro</sub>fil<sub>es</sub> h<sub>ave</sub> <sub>no</sub>t <sub>s</sub>t<sub>a</sub>bili<sub>ze</sub>d<sub>.</sub> Alt<sub>erna</sub>ti<sub>ng</sub> <sub>me</sub>th<sub>o</sub>d<sub>s</sub> i<sub>n</sub> th<sub>e</sub> <sub>regre</sub>t <sub>ma</sub>t<sub>c</sub>hi<sub>ng</sub> f<sub>am</sub>il<sub>y</sub> <sub>o</sub>ft<sub>en</sub> <sub>ou</sub>t<sub>per-</sub> form their simultaneous counter<sub>p</sub>arts for <sub>p</sub>ractical self-<sub>p</sub>la<sub>y</sub> [Brown and Sandholm, 2019, Farina et al., 2021], <sub>y</sub>et <sub>a sys</sub>t<sub>ema</sub>ti<sub>c un</sub>d<sub>ers</sub>t<sub>an</sub>di<sub>ng o</sub>f th<sub>e</sub>i<sub>r convergence rema</sub>i<sub>ns</sub> l<sub>arge</sub>l<sub>y</sub> i<sub>ncomp</sub>l<sub>e</sub>t<sub>e.</sub> F<sub>or</sub> $\mathsf { A l t { - } R M ^ { + } }$ <sub>, a su</sub>b<sub>op</sub>ti<sub>ma</sub>l ${ \cal O } ( T ^ { - 1 / 2 } )$ avera<sub>g</sub>e-iterate <sub>g</sub>uarantee is known [Burch et al., 2019], while for $\mathsf { A l t { - } P R M ^ { + } }$ , an avera<sub>g</sub>e-<sup>i</sup>terate conver<sub>g</sub>ence rema<sup>i</sup>ns unresolved [Cai et al., 2025]. B<sub>y</sub> com<sub>p</sub>arison, simultaneous o<sub>p</sub>timistic and extra<sub>g</sub>radient variants in this famil<sub>y</sub> admit <sub>s</sub>t<sub>ronger guaran</sub>t<sub>ees,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> th<sub>e op</sub>ti<sub>ma</sub>l ${ \cal O } ( T ^ { - 1 } )$ avera<sub>g</sub>e-iterate rates [Farina et al., 2023, Zhan<sub>g</sub> et al., 2025]. These <sub>gaps</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>eory</sub> <sub>an</sub>d <sub>prac</sub>ti<sub>ce</sub> <sub>mo</sub>ti<sub>va</sub>t<sub>e</sub> <sub>us</sub> t<sub>o</sub> d<sub>eve</sub>l<sub>op</sub> <sub>a</sub> <sub>sys</sub>t<sub>ema</sub>ti<sub>c</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>o</sub>f f<sub>as</sub>t <sub>average-</sub>it<sub>era</sub>t<sub>e</sub> <sub>convergence</sub> <sub>un</sub>d<sub>er</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng up</sub>d<sub>a</sub>t<sub>es.</sub>

W<sub>e</sub> fi<sub>rs</sub>t <sub>a</sub>d<sub>ap</sub>t <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>ana</sub>l<sub>y</sub>ti<sub>ca</sub>l t<sub>oo</sub>l<sub>s</sub> f<sub>or s</sub>i<sub>mu</sub>lt<sub>aneous up</sub>d<sub>a</sub>t<sub>es</sub> t<sub>o</sub> th<sub>e a</sub>lt<sub>erna</sub>ti<sub>ng se</sub>tti<sub>ng, prov</sub>idi<sub>ng a c</sub>l<sub>ear rou</sub>t<sub>e</sub> to ${ \cal O } ( T ^ { - 1 } )$ <sub>average-</sub>it<sub>era</sub>t<sub>e convergence.</sub> Whil<sub>e some mo</sub>difi<sub>ca</sub>ti<sub>ons may appear</sub> t<sub>o</sub> b<sub>e mere</sub> b<sub>oo</sub>kk<sub>eep</sub>i<sub>ng, our</sub> l<sub>a</sub>t<sub>er</sub> se<sub>p</sub>aration results show wh<sub>y</sub> the<sub>y</sub> are needed: the existin<sub>g</sub> (simultaneous) bounds can fail for learners whose alternatin<sub>g</sub> self-<sub>p</sub>la<sub>y</sub> nevertheless conver<sub>g</sub>es (Subsection 5.4). The estimates we develo<sub>p</sub> for avera<sub>g</sub>e-iterate conver<sub>g</sub>ence in this <sub>sec</sub>ti<sub>on</sub> <sub>a</sub>l<sub>so</sub> <sub>prov</sub>id<sub>e</sub> th<sub>e</sub> <sub>s</sub>t<sub>ar</sub>ti<sub>ng</sub> <sub>po</sub>i<sub>n</sub>t f<sub>or</sub> th<sub>e</sub> <sub>re</sub>fi<sub>ne</sub>d <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>o</sub>f b<sub>es</sub>t<sub>-</sub> <sub>an</sub>d l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e</sub> <sub>convergence</sub> i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 7<sub>.</sub>

We begin by recalling the Regret bounded by Variation in Utilities (RVU) bound [Syrgkanis et al., 2015], which <sub>un</sub>d<sub>er</sub>li<sub>es</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>ana</sub>l<sub>ys</sub>i<sub>s o</sub>f <sub>s</sub>i<sub>mu</sub>lt<sub>aneous op</sub>ti<sub>m</sub>i<sub>s</sub>ti<sub>c me</sub>th<sub>o</sub>d<sub>s.</sub> F<sub>or</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>ons, we</sub> fi<sub>x a</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>u</sub>tilit<sub>y</sub> d<sub>oma</sub>i<sub>n</sub> i<sub>n</sub> $\mathbb { R } ^ { d }$ and a <sub>p</sub>air of <sub>p</sub>rimal-dual norms (| · |, | · |<sub>∗</sub>).

Definition 5.1 (RVU). A learner satisfies the RVU bound if there exist constants $\alpha , \beta , \gamma > 0$ <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub> f<sub>or every u</sub>tilit<sub>y</sub> <sup>se</sup>qu<sup>ence</sup> $( \pmb { u } ^ { ( t ) } ) _ { t \geq 1 }$ <sub>1</sub> an<sup>d</sup> ever<sub>y</sub> $T \geq 1$ ,

$$
\mathrm { R e g } ( T ) \leq \alpha + \beta \sum _ { t = 2 } ^ { T } \Vert \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \Vert _ { * } ^ { 2 } - \gamma \sum _ { t = 2 } ^ { T } \Vert \pmb { x } ^ { ( t ) } - \pmb { x } ^ { ( t - 1 ) } \Vert ^ { 2 } .\tag{8}
$$

U<sub>n</sub>d<sub>er s</sub>i<sub>mu</sub>lt<sub>aneous u</sub> d<sub>a</sub>t<sub>es, w</sub>h<sub>en</sub> th<sub>e cons</sub>t<sub>an</sub>t<sub>s a</sub>ll<sub>ow</sub> th<sub>e ne a</sub>ti<sub>ve s</sub>t<sub>ra</sub>t<sub>e movemen</sub>t t<sub>erms</sub> t<sub>o a</sub>b<sub>sor</sub>b th<sub>e u</sub>tilit <sub>var</sub>i<sub>a</sub>ti<sub>on</sub> i<sub>n</sub>d<sub>uce</sub>d b th<sub>e o onen</sub>t<sub>s</sub>’ <sub>movemen</sub>t<sub>s, summ</sub>i<sub>n</sub> th<sub>e</sub> t<sub>wo</sub> l<sub>a ers</sub>’ RVU b<sub>oun</sub>d<sub>s</sub> i<sub>e</sub>ld<sub>s</sub> ${ \cal O } ( T ^ { - 1 } )$ <sub>avera e-</sub>it<sub>era</sub>t<sub>e</sub> <sup>con</sup>v<sup>er</sup>g<sup>ence: the</sup> p<sup>la</sup>y<sup>ers’ re</sup>g<sup>rets s</sup>u<sup>m to</sup> $T \mathrm { G a p } ( \bar { \mathbf { x } } ^ { ( T ) } , \bar { \mathbf { y } } ^ { ( T ) } ) \geq 0 \mathrm { [ S y r g }$ kanis et al., 2015].

U<sub>n</sub>d<sub>er a</sub>lt<sub>erna</sub>ti<sub>ng up</sub>d<sub>a</sub>t<sub>es,</sub> h<sub>owever,</sub>

$$
\mathrm { R e g } _ { X } ( T ) + \mathrm { R e g } _ { \boldsymbol { y } } ( T ) = T \mathrm { G a p } ( \bar { \mathbf { x } } ^ { ( T ) } , \bar { \mathbf { y } } ^ { ( T ) } ) - \sum _ { t = 1 } ^ { T } ( \mathbf { x } ^ { ( t ) } ) ^ { \top } \mathbf { A } ( \mathbf { y } ^ { ( t ) } - \mathbf { y } ^ { ( t - 1 ) } ) ,
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t t<sub>erm sum</sub> h<sub>as no</sub> fi<sub>xe</sub>d <sub>s</sub>i<sub>gn an</sub>d d<sub>oes no</sub>t <sub>genera</sub>ll<sub>y</sub> t<sub>e</sub>l<sub>escope.</sub> Thi<sub>s ex</sub>t<sub>ra</sub> t<sub>erm ar</sub>i<sub>ses</sub> b<sub>ecause</sub> th<sub>e row</sub> <sub>p</sub><sup>l</sup>a<sub>y</sub>er o<sup>b</sup>serves $\mathbf { A } \mathbf { y } ^ { ( t ) }$ at $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ <sub>,</sub> <sub>w</sub>h<sub>ereas</sub> th<sub>e</sub> <sub>co</sub>l<sub>umn</sub> <sub>p</sub>l<sub>ayer</sub> <sub>o</sub>b<sub>serves</sub> $- \mathbf { A } ^ { \top } \mathbf { x } ^ { ( t ) } ~ \mathrm { a t } ~ \mathbf { y } ^ { ( t - 1 ) }$ <sub>.</sub> A<sub>n</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l RVU b<sub>oun</sub>d <sub>can</sub> <sub>s</sub>till h<sub>o</sub>ld <sub>un</sub>d<sub>er a</sub>lt<sub>erna</sub>ti<sub>ng</sub> f<sub>ee</sub>db<sub>ac</sub>k<sub>,</sub> b<sub>u</sub>t <sub>summ</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>wo p</sub>l<sub>ayers</sub>’ b<sub>oun</sub>d<sub>s no</sub> l<sub>onger g</sub>i<sub>ves</sub> th<sub>e usua</sub>l <sub>cance</sub>ll<sub>a</sub>ti<sub>on.</sub> Th<sub>us,</sub> i<sub>n a</sub>dditi<sub>on</sub> t<sub>o es</sub>t<sub>a</sub>bli<sub>s</sub>hi<sub>ng</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>regre</sub>t b<sub>oun</sub>d<sub>s, a</sub> k<sub>ey c</sub>h<sub>a</sub>ll<sub>enge</sub> i<sub>n ana</sub>l<sub>yz</sub>i<sub>ng a</sub>lt<sub>erna</sub>ti<sub>ng up</sub>d<sub>a</sub>t<sub>es</sub> i<sub>s</sub> t<sub>o con</sub>t<sub>ro</sub>l the interaction between the players’ updates.

## 5.1 Alternating RVU and Analysis Template

O<sub>ur</sub> <sub>goa</sub>l i<sub>s</sub> t<sub>o</sub> <sub>a</sub>d<sub>ap</sub>t th<sub>e</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> t<sub>o</sub> thi<sub>s</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> <sub>or</sub>d<sub>er.</sub> O<sub>ur</sub> fi<sub>rs</sub>t <sub>s</sub>t<sub>ep</sub> i<sub>s</sub> t<sub>o</sub> <sub>cons</sub>id<sub>er</sub> th<sub>e</sub> <sub>recen</sub>t id<sub>ea</sub> <sub>o</sub>f <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> re<sub>g</sub>ret [Cevher et al., 2023, Hait et al., 2025], which is more natural to the structure of alternatin<sub>g</sub> u<sub>p</sub>dates:

$$
\operatorname { A l t } \mathrm { - } \mathrm { R e } \mathbf { g } _ { X } ( T ) = \operatorname* { m a x } _ { z \in \mathcal { X } } \sum _ { t = 1 } ^ { T } \langle \mathbf { A } y ^ { ( t ) } , 2 z - x ^ { ( t ) } - { \mathbf { x } } ^ { ( t + 1 ) } \rangle , \ \operatorname { A l t } \mathrm { - } \mathrm { R e } \mathbf { g } _ { y } ( T ) = \operatorname* { m a x } _ { z \in \mathcal { Y } } \sum _ { t = 1 } ^ { T } \langle - \mathbf { A } ^ { T } { \mathbf { x } } ^ { ( t ) } , 2 z - { \mathbf { y } } ^ { ( t - 1 ) } - { \mathbf { y } } ^ { ( t ) } \rangle .
$$

I<sub>n</sub>t<sub>u</sub>iti<sub>ve</sub>l <sub>,</sub> i<sub>n</sub> th<sub>e a</sub>lt<sub>erna</sub>ti<sub>n u</sub> d<sub>a</sub>t<sub>e sc</sub>h<sub>eme,</sub> b<sub>o</sub>th $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ <sub>an</sub>d $\boldsymbol { x } ^ { ( t + 1 ) }$ are <sub>p</sub><sup>l</sup>a<sub>y</sub>e<sup>d</sup> a<sub>g</sub>a<sup>i</sup>nst $\mathbf { y } ^ { ( t ) }$ <sub>.</sub> Alt<sub>erna</sub>ti<sub>ng regre</sub>t f<sub>or</sub> th<sub>e</sub> � <sub>p</sub>la<sub>y</sub>er takes into account both of these strate<sub>g</sub>ies, while standard (simultaneous) re<sub>g</sub>ret onl<sub>y</sub> considers $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ a<sub>g</sub>a<sup>i</sup>nst $\mathbf { y } ^ { ( t ) }$ . The same oes for the � la er, as both $\mathbf { y } ^ { ( t - 1 ) }$ <sub>an</sub>d $\mathbf { y } ^ { ( t ) }$ are <sub>p</sub><sup>l</sup>a<sub>y</sub>e<sup>d</sup> a<sub>g</sub>a<sup>i</sup>nst $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ <sub>.</sub> M<sub>oreover,</sub> th<sub>e sum o</sub>f t<sub>wo p</sub>l<sub>ayers</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng regre</sub>t<sub>s res</sub>t<sub>ores</sub> th<sub>e</sub> d<sub>es</sub>i<sub>re</sub>d <sub>cance</sub>ll<sub>a</sub>ti<sub>on</sub>

$$
\mathrm { A l t - R e g } _ { X } ( T ) + \mathrm { A l t - R e g } _ { y } ( T ) = 2 T \mathrm { G a p } ( \bar { \pmb x } ^ { ( T ) } , \bar { \pmb y } ^ { ( T ) } ) + ( \pmb x ^ { ( 1 ) } ) ^ { \top } \mathbf { A } \pmb y ^ { ( 0 ) } - ( \pmb x ^ { ( T + 1 ) } ) ^ { \top } \mathbf { A } \pmb y ^ { ( T ) } ,\tag{9}
$$

since the endpoint diference is at most range(A).

U<sub>s</sub>i<sub>ng</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>regre</sub>t<sub>,</sub> <sub>we</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> Alt<sub>-</sub>RVU b<sub>oun</sub>d<sub>.</sub> U<sub>n</sub>lik<sub>e</sub> th<sub>e</sub> <sub>un</sub>if<sub>orm</sub> <sub>requ</sub>i<sub>remen</sub>t i<sub>n</sub> th<sub>e</sub> RVU definition (5.1), its constants ma de end on the utilit se uence, but must remain valid for ever horizon �. This de<sub>p</sub>endence also arises in the zero-initialized conver<sub>g</sub>ence anal<sub>y</sub>sis of Zhan<sub>g</sub> et al. [2025, Section 5.1]: in both their <sub>an</sub>d <sub>our</sub> l<sub>a</sub>t<sub>er ana</sub>l<sub>yses,</sub> th<sub>e pos</sub>iti<sub>ve cons</sub>t<sub>an</sub>t <sub>o</sub>f th<sub>e nega</sub>ti<sub>ve s</sub>t<sub>ra</sub>t<sub>egy movemen</sub>t t<sub>erm</sub> i<sub>s o</sub>bt<sub>a</sub>i<sub>ne</sub>d f<sub>rom</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>nonzero</sub> <sup>re</sup>g<sup>ret</sup> v<sup>ector</sup>.

Definition 5.2 (Alt-RVU). A learner satisfies the Alt-RVU bound ${ \mathrm { i f } } ,$ f<sub>or</sub> <sub>every</sub> <sub>u</sub>tilit<sub>y</sub> <sub>sequence</sub> $( \pmb { u } ^ { ( t ) } ) _ { t \geq 1 }$ <sub>,</sub> th<sub>ere ex</sub>i<sub>s</sub>t constants $\alpha , \beta , \gamma > 0$ <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub> f<sub>or every</sub> $T \geq 1$

$$
\mathrm { A l t - R e g } ( T ) \leq \alpha + \beta \sum _ { t = 2 } ^ { T } \Vert \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \Vert _ { * } ^ { 2 } - \gamma \sum _ { t = 1 } ^ { T } \Vert \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } \Vert ^ { 2 } .\tag{10}
$$

Com<sub>p</sub>ared with standard RVU (Definition 5.1), Alt-RVU not onl<sub>y</sub> bounds alternatin<sub>g</sub> re<sub>g</sub>ret but also <sub>p</sub>airs utilit<sub>y</sub> <sub>var</sub>i<sub>a</sub>ti<sub>on w</sub>ith th<sub>e su</sub>b<sub>sequen</sub>t <sub>s</sub>t<sub>ra</sub>t<sub>egy movemen</sub>t<sub>, ra</sub>th<sub>er</sub> th<sub>an w</sub>ith <sub>movemen</sub>t <sub>over</sub> th<sub>e same pa</sub>i<sub>r o</sub>f<sub>consecu</sub>ti<sub>ve</sub> it<sub>era</sub>ti<sub>ons.</sub> Thi<sub>s</sub> <sub>s</sub>hift i<sub>n</sub> i<sub>n</sub>d<sub>ex</sub>i<sub>ng</sub> <sub>prec</sub>i<sub>se</sub>l<sub>y</sub> <sub>re</sub>fl<sub>ec</sub>t<sub>s</sub> th<sub>e</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> <sub>or</sub>d<sub>er.</sub> U<sub>s</sub>i<sub>ng</sub> Alt<sub>-</sub>RVU<sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>guaran</sub>t<sub>ees</sub> f<sub>or a</sub>lt<sub>erna</sub>ti<sub>ng up</sub>d<sub>a</sub>t<sub>es</sub> th<sub>roug</sub>h <sub>an argumen</sub>t <sub>ana</sub>l<sub>ogous</sub> t<sub>o</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d RVU <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> f<sub>or s</sub>i<sub>mu</sub>lt<sub>aneous up</sub>d<sub>a</sub>t<sub>es.</sub>

Proposition 5.3. Suppose that in a normal-form game with alternating update scheme, for a pair of primal-dual norms $( \| \cdot \| , \| \cdot \| _ { * } )$ , both players satisfy the above Alt-RVU bound with $( \alpha _ { \boldsymbol { X } } , \beta _ { \boldsymbol { X } } , \gamma _ { \boldsymbol { X } } ) , ( \alpha _ { \boldsymbol { y } } , \beta _ { \boldsymbol { y } } , \gamma _ { \boldsymbol { y } } ) \ \in \ \mathbb { R } _ { > 0 } ^ { 3 }$ such that $\| \mathbf { A } \| ^ { 2 } \beta _ { X } < \gamma _ { y }$ and $\| \mathbf { A } \| ^ { 2 } \beta _ { y } < \gamma _ { X }$ , where $\| \mathbf { A } \| = \operatorname* { s u p } _ { \| \mathbf { x } \| = \| \mathbf { y } \| = 1 } | \mathbf { x } ^ { \top } \mathbf { A } \mathbf { y } |$ is the induced bilinear norm. Then, both players’ squared strategy movement sum and both $p l a y e r s ^ { \prime } \ u t i l i t y$ variation sum are finite, and we obtain the average-iterate convergence

$$
\mathrm { G a p } ( \bar { x } ^ { ( T ) } , \bar { y } ^ { ( T ) } ) \leq \frac { \alpha _ { X } + \alpha _ { y } + \mathrm { r a n g e } ( \mathbf { A } ) } { 2 T } .
$$

Analysis Template. We now develop a common approach to establishing Alt-RVU for the regret-matching learners <sub>cons</sub>id<sub>ere</sub>d i<sub>n</sub> thi<sub>s wor</sub>k<sub>.</sub> Th<sub>roug</sub>h<sub>ou</sub>t th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng ana</sub>l<sub>yses, we cons</sub>id<sub>er a</sub> l<sub>earner on</sub> $\Delta ( d )$ <sub>, w</sub>h<sub>ere</sub> � i<sub>s</sub> th<sub>e num</sub>b<sub>er o</sub>f <sub>ac</sub>ti<sub>ons</sub> $( i . e . , d = m$ f<sub>or</sub> th<sub>e</sub> <sub>row</sub> <sub>p</sub>l<sub>ayer</sub> <sub>an</sub>d $d = n$ for the column <sub>p</sub>la<sub>y</sub>er), with the standard initialization (first strate<sub>g</sub>ies are uniform and re<sub>g</sub>ret vectors are zero) and last-utilit<sub>y p</sub>redictions $( i . e .$ , the <sub>p</sub>rediction is the <sub>p</sub>revious utilit<sub>y</sub>). Lastl<sub>y</sub>, <sub>we on</sub>l <sub>cons</sub>id<sub>er</sub> th<sub>e a</sub>i<sub>r o</sub>f E<sub>uc</sub>lid<sub>ean norms</sub> f<sub>or a</sub>ll th<sub>e</sub> Alt<sub>-</sub>RVU b<sub>oun</sub>d<sub>s</sub> i<sub>n</sub> thi<sub>s wor</sub>k<sub>.</sub>

F<sub>or</sub> th<sub>e</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l<sub>-</sub>l<sub>earner es</sub>ti<sub>ma</sub>t<sub>es</sub> b<sub>e</sub>l<sub>ow, we assume</sub> $\| \pmb { u } ^ { ( t ) } \| _ { 2 } \leq \| \mathbf { A } \| _ { 2 }$ <sup>f</sup>or ever<sub>y</sub> $t \geq 1$ <sub>.</sub> A<sub>par</sub>t f<sub>rom</sub> thi<sub>s</sub> b<sub>oun</sub>d<sub>,</sub> th<sub>ese es</sub>ti<sub>ma</sub>t<sub>es</sub> i<sub>mpose no res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>on on</sub> h<sub>ow</sub> th<sub>e u</sub>tilit<sub>y sequence</sub> i<sub>s genera</sub>t<sub>e</sub>d<sub>.</sub> Th<sub>e same proo</sub>f<sub>s app</sub>l<sub>y</sub> t<sub>o any</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>u</sub>tilit<sub>y</sub> <sub>sequence</sub> b<sub>y</sub> <sub>rep</sub>l<sub>ac</sub>i<sub>ng</sub> $\| \mathbf { A } \| _ { 2 }$ i<sub>n</sub> th<sub>e</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l<sub>-</sub>l<sub>earner es</sub>ti<sub>ma</sub>t<sub>es w</sub>ith $\operatorname { s u p } _ { t \geq 1 } \Vert \pmb { u } ^ { ( t ) } \Vert _ { 2 }$ <sub>.</sub> I<sub>n</sub> <sub>se</sub>lf<sub>-p</sub>l<sub>ay,</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d b<sub>oun</sub>d h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> b<sub>o</sub>th <sub>p</sub>l<sub>ayers</sub> b<sub>ecause</sub> ${ \pmb u } _ { X } ^ { ( t ) } = { \bf A } { \bf y } ^ { ( t ) }$ <sub>an</sub>d $\pmb { u } _ { y } ^ { ( t ) } = - \mathbf { A } ^ { \top } \pmb { x } ^ { ( t ) }$ <sub>,</sub> <sub>w</sub>ith $\| \mathbf { \boldsymbol { x } } ^ { ( t ) } \| _ { 2 } , \| \mathbf { \boldsymbol { y } } ^ { ( t ) } \| _ { 2 } \le 1$ <sub>.</sub> F<sub>or no</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>conven</sub>i<sub>ence, we a</sub>l<sub>so se</sub>t $\tilde { r } ^ { ( 0 ) } = \mathbf { 0 }$

T<sub>o</sub> <sub>s</sub>t<sub>ar</sub>t <sub>w</sub>ith<sub>,</sub> <sub>we</sub> h<sub>ave</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>genera</sub>l b<sub>oun</sub>d <sub>on</sub> th<sub>e</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>regre</sub>t<sub>.</sub>

Lemma 5.4. Consider a learner on $\Delta ( d )$ with $\tilde { r } ^ { ( 1 ) } = \mathbf { 0 }$ whose regret vectors satisfy

$$
\tilde { r } ^ { ( t + 1 ) } \geq \tilde { r } ^ { ( t ) } + u ^ { ( t ) } - \langle u ^ { ( t ) } , x ^ { ( t ) } \rangle { \bf 1 }\tag{11}
$$

coordinate-wise. Then,for every $T \geq 1 _ { : }$ , this learner satisfies

$$
\mathrm { A l t - R e g } ( T ) \le 2 \| \tilde { r } ^ { T + 1 } \| _ { 2 } - \sum _ { t = 1 } ^ { T } \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle .\tag{12}
$$

F<sub>or a spec</sub>ifi<sub>c no-regre</sub>t l<sub>earner w</sub>ith <sub>a</sub>lt<sub>erna</sub>ti<sub>ng up</sub>d<sub>a</sub>t<sub>es,</sub> thi<sub>s</sub> l<sub>emma</sub> id<sub>en</sub>tifi<sub>es</sub> th<sub>e</sub> t<sub>wo</sub> k<sub>ey componen</sub>t<sub>s o</sub>f <sub>our</sub> l<sub>a</sub>t<sub>er</sub> analyses: an upper bound on the terminal regret vector norm $\| \tilde { r } ^ { ( T + 1 ) } \| _ { 2 }$ and a lower bound on the cumulative utility improvement $\begin{array} { r } { \sum _ { t = 1 } ^ { \hat { T } } \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle } \end{array}$ <sub>.</sub> S<sub>pec</sub>ifi<sub>ca</sub>ll<sub>y,</sub> th<sub>ese upper an</sub>d l<sub>ower</sub> b<sub>oun</sub>d<sub>s mus</sub>t d<sub>epen</sub>d <sub>on</sub> th<sub>e u</sub>tilit<sub>y var</sub>i<sub>a</sub>ti<sub>on</sub> and strate<sub>gy</sub> movement terms in Alt-RVU (10).

W<sub>e</sub> <sub>a</sub>l<sub>so</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>nonzero</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> i<sub>n</sub>d<sub>ex</sub>

$$
t _ { 0 } : = \operatorname* { m i n } \{ t \geq 1 : \tilde { r } ^ { ( t ) } \neq 0 \} ,\tag{13}
$$

<sub>w</sub>ith $t _ { 0 } = \infty$ if thi<sub>s se</sub>t i<sub>s emp</sub>t<sub>y.</sub> If $t _ { 0 } = \infty$ <sub>,</sub> th<sub>e s</sub>t<sub>ra</sub>t<sub>egy rema</sub>i<sub>ns un</sub>if<sub>orm an</sub>d <sub>every o</sub>b<sub>serve</sub>d <sub>u</sub>tilit<sub>y vec</sub>t<sub>or</sub> h<sub>as equa</sub>l <sub>coor</sub>di<sub>na</sub>t<sub>es.</sub> Th<sub>us,</sub> Alt<sub>-</sub>RVU h<sub>o</sub>ld<sub>s</sub> t<sub>r</sub>i<sub>v</sub>i<sub>a</sub>ll<sub>y.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore assume</sub> $t _ { 0 } < \infty$ i<sub>n</sub> th<sub>e su</sub>b<sub>sequen</sub>t <sub>ana</sub>l<sub>ys</sub>i<sub>s.</sub>

$\mathbf { A } \mathrm { t } \ t = t _ { 0 } - 1$ <sub>,</sub> th<sub>e</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> i<sub>s</sub> <sub>s</sub>till <sub>un</sub>if<sub>orm,</sub> <sub>an</sub>d <sub>a</sub>ll th<sub>ree</sub> l<sub>earners</sub> <sub>we</sub> <sub>cons</sub>id<sub>er</sub> i<sub>n</sub> thi<sub>s</sub> <sub>paper</sub> h<sub>ave</sub> th<sub>e</sub> <sub>same</sub> <sub>regre</sub>t<sub>-vec</sub>t<sub>or</sub> u<sub>p</sub><sup>d</sup>ate:

$$
\tilde { r } ^ { ( t _ { 0 } ) } = \Bigg [ u ^ { ( t _ { 0 } - 1 ) } - \frac { \langle u ^ { ( t _ { 0 } - 1 ) } , \mathbf { 1 } \rangle } { d } \mathbf { 1 } \Bigg ] _ { + } .
$$

C<sub>onsequen</sub>tl<sub>y,</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>nonzero</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
0 < \| \tilde { r } ^ { ( t _ { 0 } ) } \| _ { 2 } ^ { 2 } \leq \left\| u ^ { ( t _ { 0 } - 1 ) } - \frac { \left. u ^ { ( t _ { 0 } - 1 ) } , \mathbf { 1 } \right. } { d } \mathbf { 1 } \right\| _ { 2 } ^ { 2 } = \| u ^ { ( t _ { 0 } - 1 ) } \| _ { 2 } ^ { 2 } - \frac { \langle u ^ { ( t _ { 0 } - 1 ) } , \mathbf { 1 } \rangle ^ { 2 } } { d } \leq \| \mathbf { A } \| _ { 2 } ^ { 2 } .\tag{14}
$$

Thi<sub>s</sub> <sub>prov</sub>id<sub>es</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on</sub> b<sub>oun</sub>d <sub>use</sub>d i<sub>n</sub> <sub>our</sub> <sub>su</sub>b<sub>sequen</sub>t <sub>es</sub>ti<sub>ma</sub>t<sub>es</sub> <sub>o</sub>f th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>norm.</sub>

F<sub>or</sub> th<sub>e</sub> l<sub>earners</sub> <sub>cons</sub>id<sub>ere</sub>d h<sub>ere,</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>u</sub>tilit<sub>y</sub> i<sub>mprovemen</sub>t i<sub>s</sub> <sub>pos</sub>iti<sub>ve,</sub> <sub>as</sub> <sub>we</sub> <sub>ver</sub>if<sub>y</sub> i<sub>n</sub> th<sub>e</sub>i<sub>r</sub> <sub>respec</sub>ti<sub>ve</sub> <sub>ana</sub>l<sub>yses.</sub> Si<sub>nce</sub> th<sub>e s</sub>t<sub>ra</sub>t<sub>egy</sub> d<sub>oes no</sub>t <sub>c</sub>h<sub>ange</sub> b<sub>e</sub>f<sub>ore</sub> it<sub>era</sub>ti<sub>on</sub> $t _ { 0 } - 1$ <sub>, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\sum _ { t = 1 } ^ { t _ { 0 } - 1 } \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle = \langle { \pmb u } ^ { ( t _ { 0 } - 1 ) } , { \pmb x } ^ { ( t _ { 0 } ) } - { \pmb x } ^ { ( t _ { 0 } - 1 ) } \rangle > 0 .\tag{15}
$$

F<sub>or</sub> $T \geq t _ { 0 }$ , we can therefore dro<sub>p</sub> this initial contribution when a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> (12) and focus on the utilit<sub>y</sub> im<sub>p</sub>rovements f<sub>rom</sub> $t _ { 0 }$ <sub>onwar</sub>d<sub>.</sub>

Obstacles that arise when applying the analysis template to Alt-PRM<sup>+</sup>. As discussed in the introduction, $\mathsf { A l t { - } P R M ^ { + } }$ is amon the best- erformin methods in ractice [Farina et al. 2021 Chakrabarti et al. 2024] et its avera e-iterate <sub>convergence rema</sub>i<sub>ns unreso</sub>l<sub>ve</sub>d<sub>.</sub> W<sub>e wou</sub>ld th<sub>ere</sub>f<sub>ore wan</sub>t t<sub>o app</sub>l<sub>y</sub> th<sub>e ana</sub>l<sub>ys</sub>i<sub>s</sub> t<sub>emp</sub>l<sub>a</sub>t<sub>e</sub> t<sub>o</sub> $\mathtt { A l t - P R M ^ { + } }$ <sub>, see</sub>ki<sub>ng</sub> t<sub>o</sub> <sub>exp</sub>l<sub>a</sub>i<sub>n</sub> it<sub>s</sub> <sub>prac</sub>ti<sub>ca</sub>l <sub>success</sub> th<sub>roug</sub>h Alt<sub>-</sub>RVU<sub>.</sub> H<sub>owever,</sub> th<sub>a</sub>t d<sub>oes</sub> <sub>no</sub>t <sub>wor</sub>k<sub>.</sub> Alth<sub>oug</sub>h <sub>we</sub> <sub>canno</sub>t <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h Alt<sub>-</sub>RVU f<sub>or</sub> $\mathsf { A l t { - } P R M ^ { + } }$ <sub>, we</sub> id<sub>en</sub>tif th<sub>e o</sub>b<sub>s</sub>t<sub>ac</sub>l<sub>es</sub> t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub>i<sub>n suc</sub>h <sub>a</sub> b<sub>oun</sub>d

W<sub>e</sub> b<sub>eg</sub>i<sub>n</sub> b<sub>y</sub> b<sub>oun</sub>di<sub>ng</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>norm.</sub>

Lemma 5.5. For every $T \geq 1$ , a PRM<sup>+</sup> learner satisfies

$$
\operatorname* { m a x } _ { 1 \leq s \leq T + 1 } \| \tilde { r } ^ { ( s ) } \| _ { 2 } ^ { 2 } \leq \| \mathbf { A } \| _ { 2 } ^ { 2 } + 2 ( 1 + \sqrt { d } ) ^ { 2 } \sum _ { t = 2 } ^ { T } \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + 2 d \| \mathbf { A } \| _ { 2 } ^ { 2 } \sum _ { t = 1 } ^ { T } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } .\tag{16}
$$

W<sub>e</sub> <sub>nex</sub>t b<sub>oun</sub>d th<sub>e</sub> <sub>u</sub>tilit<sub>y</sub> i<sub>mprovemen</sub>t i<sub>n</sub> t<sub>erms</sub> <sub>o</sub>f th<sub>e</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>ors.</sub>

Lemma 5.6. At time step $t \geq t _ { 0 } ,$ a PRM<sup>+</sup> learner satisfies

$$
\langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle \geq \frac { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + 2 \| [ r ^ { ( t + 1 ) } ] _ { + } - \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } .
$$

F<sub>or</sub> $T \geq t _ { 0 }$ <sub>,</sub> <sub>summ</sub>i<sub>ng</sub> th<sub>e</sub> <sub>a</sub>b<sub>ove</sub> b<sub>oun</sub>d <sub>an</sub>d <sub>separa</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>c</sub>h<sub>ange</sub> i<sub>n</sub> th<sub>e</sub> <sub>square</sub>d <sub>regre</sub>t <sub>norm</sub> f<sub>rom</sub> th<sub>e</sub> t<sub>wo</sub> <sub>square</sub>d dif<sub>erences, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\sum _ { t = t _ { 0 } } ^ { T } \langle \pmb { u } ^ { ( t ) } , \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } \rangle \ge \underbrace { \sum _ { t = t _ { 0 } } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } } _ { \Omega } + \underbrace { \sum _ { t = t _ { 0 } } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + 2 \| [ r ^ { ( t + 1 ) } ] _ { + } - \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } } _ { \Omega \mathrm { m } } .\tag{17}
$$

We can show (II) controls a wei<sub>g</sub>hted sum of squared strate<sub>gy</sub> movements, u<sub>p</sub> to a fixed initial term.

Lemma 5.7. For any $T \geq t _ { 0 } ,$ , we have

$$
\begin{array} { l } { \displaystyle \sum _ { t = t _ { 0 } } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + 2 \| [ r ^ { ( t + 1 ) } ] _ { + } - \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } \geq \frac { 1 } { 6 d } \displaystyle \sum _ { t = t _ { 0 } } ^ { T } \operatorname* { m a x } \{ \| [ r ^ { ( t ) } ] _ { + } \| _ { 1 } , \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } \} \| x ^ { ( t + 1 ) } - x ^ { ( t ) } \| _ { 2 } ^ { 2 } } \\ { \displaystyle \qquad - \frac { \| [ r ^ { ( t _ { 0 } ) } ] _ { + } - \tilde { r } ^ { ( t _ { 0 } ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t _ { 0 } ) } ] _ { + } \| _ { 1 } } . } \end{array}
$$

T<sub>wo o</sub>b<sub>s</sub>t<sub>ac</sub>l<sub>es rema</sub>i<sub>n.</sub> Th<sub>e s</sub>t<sub>ra</sub>t<sub>egy movemen</sub>t <sub>we</sub>i<sub>g</sub>ht<sub>s max</sub> $\{ \| [ \pmb { r } ^ { ( t ) } ] _ { + } \| _ { 1 } , \| [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } \}$ i<sub>n</sub> th<sub>e</sub> <sub>prece</sub>di<sub>ng</sub> b<sub>oun</sub>d d<sub>epen</sub>d <sub>on</sub> th<sub>e pre</sub>di<sub>c</sub>ti<sub>on regre</sub>t <sub>vec</sub>t<sub>ors, so</sub> thi<sub>s es</sub>ti<sub>ma</sub>t<sub>e</sub> d<sub>oes no</sub>t b<sub>y</sub> it<sub>se</sub>lf <sub>prov</sub>id<sub>e</sub> th<sub>e</sub> fi<sub>xe</sub>d <sub>pos</sub>iti<sub>ve coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>requ</sub>i<sub>re</sub>d b<sub>y</sub> Alt-RVU. More im<sub>p</sub>ortantl<sub>y</sub>, the main <sub>p</sub>roblem lies in (I), since the summand numerator does not have a fixed si<sub>g</sub>n <sub>an</sub>d <sub>canno</sub>t b<sub>e rea</sub>dil<sub>y con</sub>t<sub>ro</sub>ll<sub>e</sub>d b<sub>y</sub> th<sub>e up</sub>d<sub>a</sub>t<sub>es o</sub>f $\mathtt { A l t - P R M ^ { + } }$ <sub>.</sub> W<sub>e w</sub>ill ill<sub>us</sub>t<sub>ra</sub>t<sub>e suc</sub>h <sub>nega</sub>ti<sub>ve con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons o</sub>f th<sub>e</sub> <sub>summan</sub>d<sub>s numer</sub>i<sub>ca</sub>ll<sub>y un</sub>d<sub>er a</sub>lt<sub>erna</sub>ti<sub>ng se</sub>lf<sub>-p</sub>l<sub>ay</sub> i<sub>n a ma</sub>t<sub>r</sub>i<sub>x game an</sub>d di<sub>scuss</sub> it i<sub>n</sub> f<sub>ur</sub>th<sub>er</sub> d<sub>e</sub>t<sub>a</sub>il i<sub>n</sub> S<sub>u</sub>b<sub>sec</sub>ti<sub>on</sub> 5<sub>.</sub>3<sub>.</sub>

## 5.2 Alt-RVU with Norm Monotonicity: Alt-IR-PRM<sup>+</sup>

The obstacles we identified above motivate us to consider the learnin<sub>g</sub> al<sub>g</sub>orithm of Zhan<sub>g</sub> et al. [2025]. Inter<sub>p</sub>retin<sub>g</sub> th<sub>e regre</sub>t <sub>vec</sub>t<sub>or norm as an</sub> i<sub>nverse</sub> l<sub>earn</sub>i<sub>ng ra</sub>t<sub>e,</sub> th<sub>ey see</sub>k t<sub>o preven</sub>t i<sub>ncreases</sub> i<sub>n</sub> th<sub>e e</sub>f<sub>ec</sub>ti<sub>ve</sub> l<sub>earn</sub>i<sub>ng ra</sub>t<sub>e</sub> b<sub>y</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>i<sub>ng</sub> th<sub>e</sub> IR <sub>mec</sub>h<sub>an</sub>i<sub>sm, w</sub>hi<sub>c</sub>h <sub>ensures</sub> $\lVert \tilde { r } ^ { ( t + 1 ) } \rVert _ { 2 } ^ { 2 } \geq \lVert \tilde { r } ^ { ( t ) } \rVert _ { 2 } ^ { 2 }$ at ever<sub>y</sub> iteration �. Our decom<sub>p</sub>osition (17) <sub>p</sub>rovides com<sub>p</sub>lementar<sub>y</sub> motivations for this mechanism: norm monotonicit<sub>y p</sub>revents adverse norm chan<sub>g</sub>es in (I) and, to<sub>g</sub>ether with norm <sub>p</sub>reservation durin<sub>g p</sub>rediction, kee<sub>p</sub>s the wei<sub>g</sub>hts in (II) bounded awa<sub>y</sub> from 0 for $t \geq t _ { 0 }$ <sub>.</sub> We <sub>nex</sub>t d<sub>er</sub>i<sub>ve</sub> th<sub>e correspon</sub>di<sub>ng u</sub>tilit<sub>y-</sub>i<sub>mprovemen</sub>t b<sub>oun</sub>d<sub>s</sub> f<sub>or</sub> $\tt T R { - } P R M ^ { + }$ <sub>an</sub>d <sub>s</sub>h<sub>ow</sub> h<sub>ow</sub> th<sub>ese proper</sub>ti<sub>es y</sub>i<sub>e</sub>ld $\mathbf { A l t - R V U }$ and thus that IR-PRM<sup>+</sup> has ${ \cal O } ( T ^ { - 1 } )$ <sup>a</sup>v<sup>era</sup>g<sup>e-iterate con</sup>v<sup>er</sup>g<sup>ence</sup>.

R<sub>eca</sub>ll th<sub>a</sub>t <sub>un</sub>d<sub>er</sub> IR<sub>,</sub> th<sub>e up</sub>d<sub>a</sub>t<sub>e</sub> f<sub>or</sub> th<sub>e regre</sub>t <sub>vec</sub>t<sub>or</sub> $\tilde { r } ^ { ( t + 1 ) }$ becomes (7). We can rewrite it as

$$
\tilde { \boldsymbol { r } } ^ { ( t + 1 ) } = \left[ \tilde { \boldsymbol { r } } ^ { ( t ) } + \boldsymbol { u } ^ { ( t ) } - \langle \boldsymbol { u } ^ { ( t ) } , \boldsymbol { x } ^ { ( t ) } \rangle \mathbf { 1 } + c ^ { ( t ) } \mathbf { 1 } \right] _ { + } ,
$$

<sub>w</sub>h<sub>ere</sub> $c ^ { ( t ) } = \langle { { \pmb m } ^ { ( t ) } , { \pmb x } ^ { ( t ) } } \rangle - \gamma ^ { ( t ) }$ hi<sub>g</sub>hli<sub>g</sub>hts the new addition from the PRM<sup>+</sup> re<sub>g</sub>ret vector u<sub>p</sub>date (3). Zhan<sub>g</sub> et al. [2025, Proof of Theorem 3.2] showed that $c ^ { ( \bar { t } ) } \geq 0$ f<sub>or an</sub> �<sub>.</sub> Whil<sub>e</sub> th<sub>e</sub> did <sub>no</sub>t <sub>ex</sub> li<sub>c</sub>itl <sub>use</sub> $c ^ { ( t ) }$ i<sub>n</sub> th<sub>e ana</sub>l <sub>s</sub>i<sub>s, we w</sub>ill f<sub>u</sub>ll<sub>y u</sub>tili<sub>ze</sub> thi<sub>s sca</sub>l<sub>ar</sub> t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub> ti<sub>g</sub>ht b<sub>oun</sub>d<sub>s.</sub>

Si<sub>nce</sub> $c ^ { ( t ) } \geq 0 .$ , for IR-PRM<sup>+</sup> we obtain a similar ine<sub>q</sub>ualit<sub>y</sub> to (11):

$$
\tilde { r } ^ { ( t + 1 ) } = \left[ \tilde { r } ^ { ( t ) } + u ^ { ( t ) } - \langle u ^ { ( t ) } , x ^ { ( t ) } \rangle \mathbf { 1 } + c ^ { ( t ) } \mathbf { 1 } \right] _ { + } \geq \tilde { r } ^ { ( t ) } + u ^ { ( t ) } - \langle u ^ { ( t ) } , x ^ { ( t ) } \rangle \mathbf { 1 } .
$$

W<sub>e</sub> <sub>now</sub> b<sub>oun</sub>d th<sub>e</sub> <sub>u</sub>tilit<sub>y</sub> i<sub>mprovemen</sub>t f<sub>rom</sub> <sub>up</sub>d<sub>a</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub>

Lemma 5.8. At every time step $t \geq 1 ,$ , an IR-PRM<sup>+</sup> learner satisfies

$$
\langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle \geq \frac { 1 } { 4 d } \| \tilde { { \pmb r } } ^ { ( t + 1 ) } \| _ { 2 } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } + c ^ { ( t + 1 ) } - c ^ { ( t ) } .\tag{18}
$$

The next ste<sub>p</sub> is to bound the terminal re<sub>g</sub>ret vector norm. A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the norm estimate of Zhan<sub>g</sub> et al. [2025, Eq. (5)] over iterations $t _ { 0 } , \ldots , s - 1$ <sub>, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l r } {  { \| \tilde { r } ^ { ( s ) } \| _ { 2 } ^ { 2 } \leq \| \tilde { r } ^ { ( t _ { 0 } ) } \| _ { 2 } ^ { 2 } + \displaystyle \sum _ { t = t _ { 0 } } ^ { s - 1 } \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } - \langle { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } , { \pmb x } ^ { ( t ) } \rangle { \pmb 1 } \| _ { 2 } ^ { 2 } } } \\ & { } & { \leq \| { \bf A } \| _ { 2 } ^ { 2 } + \displaystyle \sum _ { t = t _ { 0 } } ^ { s - 1 } \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } - \langle { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } , { \pmb x } ^ { ( t ) } \rangle { \pmb 1 } \| _ { 2 } ^ { 2 } , } \end{array}
$$

where the second inequalit<sub>y</sub> comes from (14). Since $\| \pmb { x } ^ { ( t ) } \| _ { 2 } \leq 1$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\begin{array} { r } { \| { \boldsymbol u } ^ { ( t ) } - { \boldsymbol u } ^ { ( t - 1 ) } - \langle { \boldsymbol u } ^ { ( t ) } - { \boldsymbol u } ^ { ( t - 1 ) } , { \boldsymbol x } ^ { ( t ) } \rangle { \boldsymbol 1 } \| _ { 2 } \le \| { \boldsymbol u } ^ { ( t ) } - { \boldsymbol u } ^ { ( t - 1 ) } \| _ { 2 } + \sqrt { d } | \langle { \boldsymbol u } ^ { ( t ) } - { \boldsymbol u } ^ { ( t - 1 ) } , { \boldsymbol x } ^ { ( t ) } \rangle | \le ( 1 + \sqrt { d } ) \| { \boldsymbol u } ^ { ( t ) } - { \boldsymbol u } ^ { ( t - 1 ) } \| _ { 2 } . } \end{array}
$$

W<sub>e</sub> h<sub>ence</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> b<sub>oun</sub>d f<sub>or</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>norm:</sub>

$$
\operatorname* { m a x } _ { 1 \leq s \leq T + 1 } \| \tilde { r } ^ { ( s ) } \| _ { 2 } ^ { 2 } \leq \| \mathbf { A } \| _ { 2 } ^ { 2 } + ( 1 + \sqrt { d } ) ^ { 2 } \sum _ { t = t _ { 0 } } ^ { T } \| u ^ { ( t ) } - u ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } \leq \| \mathbf { A } \| _ { 2 } ^ { 2 } + ( 1 + \sqrt { d } ) ^ { 2 } \sum _ { t = 2 } ^ { T } \| u ^ { ( t ) } - u ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } .\tag{19}
$$

With that, we can now com<sub>p</sub>lete the Alt-RVU bound for IR-PRM<sup>+</sup> as follows.

Proposition 5.9 (Alt-RVU for IR-PRM<sup>+</sup>). For any $\beta > 0 ;$ , a IR-PRM<sup>+</sup> learner satisfies the Alt-RVU bound for $\alpha = 2 \| \mathbf { A } \| _ { 2 } + ( 1 + \sqrt { d } ) ^ { 2 } / \beta$ and $\gamma = \| \widetilde { r } ^ { ( t _ { 0 } ) } \| _ { 2 } / ( 4 d )$ . Therefore, in every finite two-player zero-sum normal-form game, alternating self-play with IR-PRM<sup>+</sup> has ${ \cal O } ( T ^ { - 1 } )$ average-iterate convergence.

Th<sub>e</sub> IR <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> th<sub>us prov</sub>id<sub>es a su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>way</sub> t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e con</sub>t<sub>ro</sub>l <sub>requ</sub>i<sub>re</sub>d b<sub>y</sub> Alt<sub>-</sub>RVU<sub>.</sub> W<sub>e nex</sub>t i<sub>nves-</sub> ti<sub>ga</sub>t<sub>e</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>en</sub>f<sub>orc</sub>i<sub>ng</sub> <sub>norm</sub> <sub>mono</sub>t<sub>on</sub>i<sub>c</sub>it<sub>y</sub> i<sub>s</sub> <sub>necessary</sub> <sub>or</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> Alt<sub>-</sub>RVU <sub>can</sub> b<sub>e</sub> <sub>o</sub>bt<sub>a</sub>i<sub>ne</sub>d <sub>w</sub>ith<sub>ou</sub>t <sub>en</sub>f<sub>orc</sub>i<sub>ng</sub> non<sup>d</sup>ecreas<sup>i</sup>n<sub>g</sub> re<sub>g</sub>ret vector norms.

## 5.3 Alt-RVU without Norm Monotonicity: Alt-HPRM<sup>+</sup>

The <sub>p</sub>recedin<sub>g</sub> subsection establishes Alt-RVU for IR-PRM<sup>+</sup> b<sub>y</sub> ex<sub>p</sub>loitin<sub>g</sub> the nondecreasin<sub>g</sub> re<sub>g</sub>ret vector norm enforced b<sub>y</sub> the IR mechanism. We now show that HPRM<sup>+</sup> satisfies Alt-RVU without re<sub>q</sub>uirin<sub>g</sub> this monotonicit<sub>y</sub>. The motivation for allowin<sub>g</sub> norm decreases comes from Alt-PRM<sup>+</sup> self-<sub>p</sub>la<sub>y</sub>, which exhibits fast em<sub>p</sub>irical avera<sub>g</sub>e-iterate <sub>convergence</sub> d<sub>esp</sub>it<sub>e suc</sub>h d<sub>ecreases on</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng ma</sub>t<sub>r</sub>i<sub>x game.</sub>

$$
{ \bf A } = \left( { \begin{array} { c c c } { 2 0 0 } & { - 2 0 0 } & { 0 } \\ { - 2 0 0 } & { 2 0 2 } & { - 1 } \\ { 0 } & { - 1 } & { 1 / 2 } \end{array} } \right) .\tag{20}
$$

Fi<sub>gure</sub> 1 <sub>s</sub>h<sub>ows</sub> b<sub>o</sub>th <sub>p</sub>l<sub>ayers</sub>’ <sub>regre</sub>t <sub>vec</sub>t<sub>or norms</sub> d<sub>ecreas</sub>i<sub>ng over</sub> it<sub>era</sub>ti<sub>ons</sub> $1 0 ^ { 5 }$ to $1 0 ^ { 6 }$ , <sub>p</sub>ro<sup>d</sup>uc<sup>i</sup>n<sub>g</sub> ne<sub>g</sub>at<sup>i</sup>ve contributions to term (I). Nevertheless, in the ri<sub>g</sub>htmost <sub>p</sub>lot, the avera<sub>g</sub>e-iterate <sub>g</sub>a<sub>p</sub> em<sub>p</sub>iricall<sub>y</sub> deca<sub>y</sub>s a<sub>pp</sub>roximatel<sub>y</sub> as $T ^ { - 1 }$ <sub>.</sub> Thi<sub>s</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or mo</sub>ti<sub>va</sub>t<sub>es con</sub>t<sub>ro</sub>lli<sub>ng</sub> th<sub>e cumu</sub>l<sub>a</sub>ti<sub>ve e</sub>f<sub>ec</sub>t <sub>o</sub>f <sub>norm c</sub>h<sub>anges w</sub>ith<sub>ou</sub>t <sub>requ</sub>i<sub>r</sub>i<sub>ng every c</sub>h<sub>ange</sub> t<sub>o</sub> b<sub>e nonnega</sub>ti<sub>ve.</sub> M<sub>oreover,</sub> th<sub>e</sub> IR <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> i<sub>ncurs a</sub>dditi<sub>ona</sub>l <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>over</sub>h<sub>ea</sub>d b<sub>y so</sub>l<sub>v</sub>i<sub>ng</sub> f<sub>or</sub> th<sub>e norm-</sub> <sub>preserv</sub>i<sub>ng</sub> <sub>sca</sub>l<sub>ar</sub> <sub>s</sub>hift $\gamma$ in (6) <sub>p</sub>er iteration. These considerations motivate us to seek an al<sub>g</sub>orithm that not onl<sub>y</sub> <sub>perm</sub>it<sub>s norm</sub> d<sub>ecreases</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so avo</sub>id<sub>s</sub> thi<sub>s sca</sub>l<sub>ar compu</sub>t<sub>a</sub>ti<sub>on w</sub>hil<sub>e re</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> Alt<sub>-</sub>RVU <sub>guaran</sub>t<sub>ee.</sub> I<sub>n</sub> f<sub>ac</sub>t<sub>, as we</sub> will show later in this <sub>p</sub>a<sub>p</sub>er, an al<sub>g</sub>orithm that this relaxation enables (HPRM<sup>+</sup>) si<sub>g</sub>nificantl<sub>y</sub> out<sub>p</sub>erforms $\tt T R { - } P R M ^ { + }$ i<sub>n</sub> <sub>prac</sub>ti<sub>ce an</sub>d th<sub>ey</sub> h<sub>ave</sub> th<sub>e same</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>convergence guaran</sub>t<sub>ees un</sub>d<sub>er a</sub>lt<sub>erna</sub>ti<sub>ng se</sub>lf<sub>-p</sub>l<sub>ay.</sub>

![](images/baff2d088b6e7977b87cf2e79fdfd94add32f7fa235ac4936ad56ed830d5a5ca.jpg)

![](images/ea19fbf533bf3ba17985d9d2dd770a2e39df4229ffea9d2a1104887d3d9bfb30.jpg)

![](images/1bc2142b68bea26681867a7b39c5b3763c1e941106248a967dffa7028eb044e4.jpg)  
Fi<sub>g</sub>ure 1: Alt-PRM<sup>+</sup> on the normal-form <sub>g</sub>ame (20).

W<sub>e</sub> b<sub>eg</sub>i<sub>n</sub> b<sub>y</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ng</sub> <sub>c</sub>h<sub>anges</sub> i<sub>n</sub> th<sub>e</sub> <sub>square</sub>d <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>norm</sub> t<sub>o</sub> <sub>consecu</sub>ti<sub>ve</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> i<sub>ncremen</sub>t<sub>s.</sub>

Lemma 5.10. For every $t \geq 1 ,$ , a HPRM<sup>+</sup> learner satisfies

$$
\| \tilde { { \boldsymbol r } } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { { \boldsymbol r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } \geq \frac { 1 } { 2 } \| \tilde { { \boldsymbol r } } ^ { ( t + 1 ) } - \tilde { { \boldsymbol r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } \| \tilde { { \boldsymbol r } } ^ { ( t ) } - \tilde { { \boldsymbol r } } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } .\tag{21}
$$

This bound gives us the idea of proposing the following corrected squared norm

$$
f ^ { ( t ) } = \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } ,
$$

b<sub>ecause</sub> it i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t

$$
f ^ { ( t + 1 ) } - f ^ { ( t ) } \geq 0 .
$$

Th<sub>e correc</sub>t<sub>e</sub>d <sub>square</sub>d <sub>norm</sub> $f ^ { ( t ) }$ <sub>rema</sub>i<sub>ns non</sub>d<sub>ecreas</sub>i<sub>ng even w</sub>h<sub>en</sub> th<sub>e regre</sub>t <sub>vec</sub>t<sub>or norm</sub> d<sub>ecreases.</sub> A<sub>n ana</sub>l<sub>ogous</sub> <sub>correc</sub>ti<sub>on</sub> i<sub>s non</sub>d<sub>ecreas</sub>i<sub>ng</sub> f<sub>or every</sub> $0 \leq \lambda \leq 1 / 2$ <sub>,</sub> b<sub>u</sub>t h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on a</sub>dditi<sub>ona</sub>ll<sub>y e</sub>li<sub>m</sub>i<sub>na</sub>t<sub>es a pos</sub>iti<sub>ve movemen</sub>t <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> f<sub>rom</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>regre</sub>t<sub>-norm</sub> b<sub>oun</sub>d<sub>, a</sub>ll<sub>ow</sub>i<sub>ng us</sub> t<sub>o com</sub>bi<sub>ne</sub> th<sub>e</sub> t<sub>wo es</sub>ti<sub>ma</sub>t<sub>es</sub> t<sub>o es</sub>t<sub>a</sub>bli<sub>s</sub>h Alt<sub>-</sub>RVU<sub>.</sub> A<sub>ppen</sub>di<sub>x</sub> A<sub>.</sub>3 d<sub>er</sub>i<sub>ves</sub> th<sub>e correc</sub>ti<sub>on</sub> f<sub>or genera</sub>l � <sub>an</sub>d <sub>exp</sub>l<sub>a</sub>i<sub>ns</sub> thi<sub>s</sub> di<sub>s</sub>ti<sub>nc</sub>ti<sub>on.</sub>

Th<sub>e</sub> k<sub>ey</sub> id<sub>ea</sub> i<sub>s</sub> th<sub>a</sub>t <sub>s</sub>i<sub>nce</sub> $f ^ { ( t ) }$ i<sub>s non</sub>d<sub>ecreas</sub>i<sub>ng,</sub> it <sub>prov</sub>id<sub>es us more con</sub>t<sub>ro</sub>l th<sub>an</sub> th<sub>e regre</sub>t <sub>vec</sub>t<sub>or norm</sub> $\| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 }$ O<sub>ur ana</sub>l <sub>s</sub>i<sub>s can s</sub>hift t<sub>o</sub> $f ^ { ( t ) }$ if it<sub>s</sub> i<sub>ncremen</sub>t<sub>s a</sub>l<sub>so con</sub>t<sub>ro</sub>l th<sub>e u</sub>tilit i<sub>m rovemen</sub>t<sub>.</sub> W<sub>e can,</sub> i<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> b<sub>oun</sub>d th<sub>e u</sub>tilit i<sub>mprovemen</sub>t <sub>w</sub>ith $f ^ { ( t + 1 ) } - f ^ { ( t ) }$

Lemma 5.11. At every time step $t \geq t _ { 0 } - 1 _ { : }$ , a $H P R M ^ { + }$ learner satisfies

$$
\langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle \geq \frac { \| \widetilde { { \pmb r } } ^ { ( t + 1 ) } - \widetilde { { \pmb r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { 4 \| \left[ { \pmb r } ^ { ( t + 1 ) } \right] _ { + } \| _ { 1 } } - \frac { f ^ { ( t + 1 ) } - f ^ { ( t ) } } { 2 \| \left[ { \pmb r } ^ { ( t + 1 ) } \right] _ { + } \| _ { 1 } } .\tag{22}
$$

F<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>our</sub> Alt<sub>-</sub>RVU <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> t<sub>emp</sub>l<sub>a</sub>t<sub>e,</sub> <sub>we</sub> <sub>now</sub> b<sub>oun</sub>d th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>norm.</sub> C<sub>ompare</sub>d <sub>w</sub>ith th<sub>e</sub> PRM<sup>+</sup> estimate, this bound also involves re<sub>g</sub>ret-vector increments. Their coeficient can be chosen to be arbitraril<sub>y</sub> <sub>sma</sub>ll<sub>, w</sub>hi<sub>c</sub>h <sub>w</sub>ill <sub>a</sub>ll<sub>ow us</sub> t<sub>o con</sub>t<sub>ro</sub>l th<sub>e</sub>i<sub>r con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on us</sub>i<sub>ng</sub> th<sub>e u</sub>tilit<sub>y-</sub>i<sub>mprovemen</sub>t b<sub>oun</sub>d<sub>.</sub>

Lemma 5.12. For every $\varepsilon > 0$ and $T \geq 1$ , a HPRM<sup>+</sup> learner satisfies

$$
\operatorname* { m a x } _ { 1 \leq s \leq T + 1 } \| \tilde { r } ^ { ( s ) } \| _ { 2 } ^ { 2 } \leq \| \mathbb { A } \| _ { 2 } ^ { 2 } + \left( 2 + \frac { 1 } { 2 s } \right) \left[ ( 1 + \sqrt { d } ) ^ { 2 } \sum _ { t = 2 } ^ { T } \| u ^ { ( t ) } - u ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + d \| \mathbb { A } \| _ { 2 } ^ { 2 } \sum _ { t = 1 } ^ { T } \| x ^ { ( t + 1 ) } - x ^ { ( t ) } \| _ { 2 } ^ { 2 } \right] + \varepsilon \sum _ { t = 2 } ^ { T } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } .\tag{23}
$$

W<sub>e</sub> <sub>now</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng.</sub>

Proposition 5.13 (Alt-RVU for $\mathsf { A l t { - } H P R M ^ { + } } )$ . Set $\gamma = \sqrt { f ^ { ( t _ { 0 } ) } } / ( 6 4 \sqrt { 2 } ( 1 + \sqrt { d } ) ^ { 2 } ) > 0 .$ . For every $\beta > 0 ,$ , there exists $\alpha > 0$ so that $A \mathsf { 1 } t \mathsf { - } H P R M ^ { \dagger }$ satisfies the Alt-RVU bound. Therefore, in every finite two-player zero-sum normal-form game, alternating self-play with HPRM<sup>+</sup> has ${ \cal O } ( T ^ { - 1 } )$ average-iterate convergence.

Thi<sub>s propos</sub>iti<sub>on</sub> th<sub>us g</sub>i<sub>ves</sub> $\mathsf { A l t { - } H P R M ^ { + } }$ th<sub>e same</sub> ${ \cal O } ( T ^ { - 1 } )$ <sup>a</sup>v<sup>era</sup>g<sup>e-iterate</sup> gu<sup>arantee as</sup> $\mathsf { A l t { - } I R { - } P R M ^ { + } }$ <sub>w</sub>ith<sub>ou</sub>t enforcin<sub>g</sub> nondecreasin<sub>g</sub> re<sub>g</sub>ret vector norms. Indeed, such norm decreases occur on the same <sub>g</sub>ame (20) for Alt-HPRM<sup>+</sup> <sub>se</sub>lf<sub>-p</sub>l<sub>ay, w</sub>h<sub>ere</sub> Fi<sub>gure</sub> 2 <sub>s</sub>h<sub>ows sma</sub>ll d<sub>ecreases</sub> i<sub>n</sub> b<sub>o</sub>th <sub>p</sub>l<sub>ayers</sub>’ <sub>regre</sub>t <sub>vec</sub>t<sub>or norms up</sub> t<sub>o</sub> it<sub>era</sub>ti<sub>ons</sub> $1 0 ^ { 6 }$ <sub>.</sub> B<sub>y s</sub>hifti<sub>ng</sub> th<sub>e ana</sub>l<sub>ys</sub>i<sub>s</sub> t<sub>o</sub> th<sub>e non</sub>d<sub>ecreas</sub>i<sub>ng</sub> $f ^ { ( t ) }$ <sub>, we con</sub>t<sub>ro</sub>l th<sub>e cumu</sub>l<sub>a</sub>ti<sub>ve e</sub>f<sub>ec</sub>t <sub>o</sub>f <sub>regre</sub>t <sub>vec</sub>t<sub>or norm</sub> d<sub>ecreases an</sub>d <sub>re</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e</sub> <sub>s</sub>i<sub>mp</sub>l<sub>er</sub> h<sub>a</sub>lf<sub>-pre</sub>di<sub>c</sub>ti<sub>on up</sub>d<sub>a</sub>t<sub>e, w</sub>hi<sub>c</sub>h <sub>avo</sub>id<sub>s</sub> th<sub>e a</sub>dditi<sub>ona</sub>l <sub>sca</sub>l<sub>ar compu</sub>t<sub>a</sub>ti<sub>on requ</sub>i<sub>re</sub>d b<sub>y</sub> th<sub>e</sub> IR <sub>mec</sub>h<sub>an</sub>i<sub>sm.</sub> Th<sub>e</sub> benefits extend be<sub>y</sub>ond the lower <sub>p</sub>er-iteration cost: Section 9 <sub>p</sub>rovides ex<sub>p</sub>eriments in which the CFR instantiation of Alt-HPRM<sup>+</sup> out<sub>p</sub>erforms that of $\mathsf { A l t { - } I R { - } P R M ^ { + } }$ <sub>on mos</sub>t b<sub>enc</sub>h<sub>mar</sub>k<sub>s, ac</sub>hi<sub>ev</sub>i<sub>ng</sub> f<sub>as</sub>t<sub>er exp</sub>l<sub>o</sub>it<sub>a</sub>bilit<sub>y re</sub>d<sub>uc</sub>ti<sub>on</sub> i<sub>n</sub> t<sub>erms</sub> <sub>o</sub>f b<sub>o</sub>th it<sub>era</sub>ti<sub>on coun</sub>t <sub>an</sub>d l<sub>earner wa</sub>ll<sub>-c</sub>l<sub>oc</sub>k ti<sub>me.</sub>

![](images/146a6dba50c334307097b09b092b0e199925c5158884672fb985b1f41fb7c5c7.jpg)

![](images/ed10a06adb3c0e01180d71902022b1e4b707dbaeb4707b719549e15cee81b1fe.jpg)  
Fi<sub>g</sub>ure 2: Alt-HPRM<sup>+</sup> on the normal-form <sub>g</sub>ame (20).

![](images/85ff05e493687603ad73702ff2b6fd7f01fcb10c2f903e41e320031184fe3b83.jpg)

## 5.4 Insuficiency of Standard RVU

In the above subsections, we develo<sub>p</sub>ed Alt-RVU to establish avera<sub>g</sub>e-iterate conver<sub>g</sub>ence <sub>g</sub>uarantees for Alt-IR-PRM<sup>+</sup> <sub>an</sub>d $\mathsf { A l t { - } H P R M ^ { + } }$ <sub>.</sub> A <sub>na</sub>t<sub>ura</sub>l <sub>ues</sub>ti<sub>on</sub> i<sub>s w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>ese uaran</sub>t<sub>ees</sub> f<sub>or a</sub>lt<sub>erna</sub>ti<sub>n se</sub>lf<sub>-</sub> l<sub>a cou</sub>ld i<sub>ns</sub>t<sub>ea</sub>d b<sub>e o</sub>bt<sub>a</sub>i<sub>ne</sub>d b establishin<sub>g</sub> a standard RVU bound (Definition 5.1) for each learner. We show in this subsection that standard RVU i<sub>s</sub> t<sub>oo</sub> <sub>res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>ve</sub> t<sub>o</sub> <sub>serve</sub> <sub>as</sub> <sub>a</sub> l<sub>earner-</sub>l<sub>eve</sub>l <sub>con</sub>diti<sub>on</sub> <sub>cover</sub>i<sub>ng</sub> <sub>a</sub>ll <sub>op</sub>ti<sub>m</sub>i<sub>s</sub>ti<sub>c</sub> <sub>me</sub>th<sub>o</sub>d<sub>s</sub> th<sub>a</sub>t <sub>converge</sub> <sub>un</sub>d<sub>er</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>se</sub>lf<sub>-p</sub>l<sub>ay.</sub> I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> $\mathrm { \tt H P R M ^ { + } }$ f<sub>a</sub>il<sub>s</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d RVU b<sub>oun</sub>d d<sub>esp</sub>it<sub>e sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> Alt<sub>-</sub>RVU <sub>an</sub>d <sub>ac</sub>hi<sub>ev</sub>i<sub>ng</sub> th<sub>e op</sub>ti<sub>ma</sub>l ${ \cal O } ( T ^ { - 1 } )$ <sub>average-</sub>it<sub>era</sub>t<sub>e convergence ra</sub>t<sub>e un</sub>d<sub>er a</sub>lt<sub>erna</sub>ti<sub>ng se</sub>lf<sub>-p</sub>l<sub>ay.</sub> Thi<sub>s resu</sub>lt hi<sub>g</sub>hli<sub>g</sub>ht<sub>s</sub> th<sub>e nee</sub>d t<sub>o go</sub> b<sub>eyon</sub>d <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d RVU t<sub>o encompass suc</sub>h <sub>a</sub>lt<sub>erna</sub>ti<sub>ng me</sub>th<sub>o</sub>d<sub>s, an</sub>d <sub>our</sub> Alt<sub>-</sub>RVU f<sub>ramewor</sub>k <sub>an</sub>d <sub>accompany</sub>i<sub>ng ana</sub>l<sub>ys</sub>i<sub>s prov</sub>id<sub>e a sys</sub>t<sub>ema</sub>ti<sub>c</sub> <sub>ex</sub>t<sub>ens</sub>i<sub>on</sub> th<sub>a</sub>t d<sub>oes so.</sub>

W<sub>e</sub> <sub>now</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>a</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d RVU b<sub>oun</sub>d i<sub>s</sub> i<sub>mposs</sub>ibl<sub>e</sub> f<sub>or</sub> $\mathsf { A l t { - } H P R M ^ { + } }$ <sub>,</sub> <sub>even</sub> <sub>on</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>u</sub>tilit<sub>y</sub> <sub>sequence.</sub>

Proposition 5.14. There is a utility sequence $( \pmb { u } ^ { ( t ) } ) _ { t \geq 1 } \subset [ - 1 , 1 ] ^ { 2 }$ , which can be generated by an opponent under alternating updates, on which a HPRM<sup>+</sup> learner satisfies, for every $T \geq 1$

$$
0 \leq \mathrm { R e g } ( T ) \leq 3 , \sum _ { t = 2 } ^ { \infty } \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } < \infty , \sum _ { t = 2 } ^ { \infty } \| { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } = \infty .\tag{24}
$$

Consequently, the standard RVU bound fails on this sequence for every choice of horizon-independent positive constants, even when they may depend on the entire utility sequence.

Th<sub>e</sub> di<sub>s</sub>ti<sub>nc</sub>ti<sub>on</sub> b<sub>e</sub>t<sub>ween or</sub>di<sub>nary an</sub>d <sub>a</sub>lt<sub>erna</sub>ti<sub>ng regre</sub>t i<sub>s no</sub>t<sub>ewor</sub>th<sub>y</sub> h<sub>ere.</sub> Si<sub>nce</sub> $\mathrm { \tt H P R M ^ { + } }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> Alt<sub>-</sub>RVU<sub>,</sub> th<sub>e</sub> <sub>same</sub> fi<sub>n</sub>it<sub>e</sub> <sub>u</sub>tilit<sub>y</sub> <sub>var</sub>i<sub>a</sub>ti<sub>on</sub> <sub>an</sub>d di<sub>vergen</sub>t <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>movemen</sub>t i<sub>mp</sub>l<sub>y</sub> $\mathrm { A l t - R e g } ( T )  - \infty$ <sub>on</sub> thi<sub>s sequence.</sub> F<sub>rom</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>ons,</sub> <sub>we</sub> h<sub>ave</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y</sub>

$$
\mathrm { A l t - R e g } ( T ) = 2 \mathrm { R e g } ( T ) - \sum _ { t = 1 } ^ { T } \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle .
$$

Thi<sub>s</sub> <sub>equa</sub>lit<sub>y</sub> i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t th<sub>e</sub> <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>u</sub>tilit<sub>y</sub> i<sub>mprovemen</sub>t <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\textstyle \sum _ { t = 1 } ^ { T } \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle  + \infty$ <sub>, w</sub>hil<sub>e</sub> $0 \leq$ $\mathrm { R e g } ( T ) \leq 3$ <sub>.</sub> Alt<sub>erna</sub>ti<sub>ng</sub> <sub>regre</sub>t <sub>accoun</sub>t<sub>s</sub> f<sub>or</sub> th<sub>e</sub> <sub>ga</sub>i<sub>ns</sub> f<sub>rom</sub> <sub>up</sub>d<sub>a</sub>ti<sub>ng</sub> $\mathbf { \Delta } _ { \mathbf { X } } ^ { ( t ) } \dot { \mathbf { t o } } \dot { \mathbf { x } } ^ { ( t + 1 ) }$ <sub>aga</sub>i<sub>ns</sub>t th<sub>e same u</sub>tilit<sub>y vec</sub>t<sub>or</sub> ${ \pmb u } ^ { ( t ) } ,$ <sub>w</sub>h<sub>ereas or</sub>di<sub>nary regre</sub>t <sub>eva</sub>l<sub>ua</sub>t<sub>es on</sub>l<sub>y</sub> th<sub>e s</sub>t<sub>ra</sub>t<sub>egy</sub> $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ <sub>p</sub>l<sub>aye</sub>d b<sub>e</sub>f<sub>ore</sub> th<sub>e up</sub>d<sub>a</sub>t<sub>e.</sub> O<sub>n</sub> thi<sub>s sequence,</sub> th<sub>ese accumu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>ga</sub>i<sub>ns</sub> d<sub>r</sub>i<sub>ve</sub> $\mathrm { A l t - R e g } ( T )  - \infty$ <sub>, as</sub> Alt<sub>-</sub>RVU <sub>requ</sub>i<sub>res w</sub>h<sub>en u</sub>tilit<sub>y var</sub>i<sub>a</sub>ti<sub>on</sub> i<sub>s</sub> fi<sub>n</sub>it<sub>e</sub> b<sub>u</sub>t $\begin{array} { r } { \sum _ { t = 1 } ^ { \hat { T } } \| \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } \| _ { 2 } ^ { 2 } \to \infty } \end{array}$ Si<sub>nce</sub> $\begin{array} { r } { \mathrm { R e g } ( T ) \geq 0 } \end{array}$ <sub>, no c</sub>h<sub>o</sub>i<sub>ce o</sub>f h<sub>or</sub>i<sub>zon-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t $\alpha , \beta , \gamma > 0$ <sub>can ma</sub>k<sub>e or</sub>di<sub>nar re re</sub>t <sub>sa</sub>ti<sub>s</sub>f th<sub>e corres on</sub>di<sub>n</sub> bound, whose ri<sub>g</sub>ht-hand side tends to −∞. Chan<sub>g</sub>in<sub>g</sub> the re<sub>g</sub>ret <sub>q</sub>uantit<sub>y</sub> is therefore essential here, rather than merel<sub>y</sub> adjusting the constants in standard RVU.

Remark 5.15. The Alt-RVU framework itself is not restricted to the regret matching family. Its definition involves only <sub>u</sub>tiliti<sub>es, s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es, an</sub>d <sub>a</sub>lt<sub>erna</sub>ti<sub>ng regre</sub>t<sub>, so</sub> it<sub>s convergence</sub> i<sub>mp</sub>li<sub>ca</sub>ti<sub>ons app</sub>l<sub>y</sub> t<sub>o o</sub>th<sub>er a</sub>lt<sub>erna</sub>ti<sub>ng</sub> l<sub>earners w</sub>h<sub>enever</sub> th<sub>e re u</sub>i<sub>re</sub>d b<sub>oun</sub>d<sub>s an</sub>d <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>con</sub>diti<sub>ons</sub> h<sub>o</sub>ld<sub>.</sub> W<sub>e can rove s</sub>i<sub>m</sub>il<sub>ar</sub> Alt<sub>-</sub>RVU<sub>-</sub>b<sub>ase</sub>d <sub>uaran</sub>t<sub>ees</sub> f<sub>or o</sub>th<sub>er c</sub>l<sub>asses</sub> <sub>o</sub>f <sub>a</sub>l<sub>gor</sub>ith<sub>ms,</sub> f<sub>or examp</sub>l<sub>e,</sub> ${ \cal O } ( T ^ { - 1 } )$ average-iterate convergence under alternating self-play for the Adaptive Optimistic Gradient Descent (AdOGD) algorithm of Zhang et al. [2025]. We omit the routine details and focus on regret matching <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> h<sub>ere.</sub>

## 6 Alt-RVU Failure of Scaled-Prediction Regret Matching

Th<sub>e prece</sub>di<sub>ng ana</sub>l<sub>ys</sub>i<sub>s es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> th<sub>e</sub> Alt<sub>-</sub>RVU b<sub>oun</sub>d f<sub>or</sub> th<sub>e</sub> h<sub>a</sub>lf<sub>-pre</sub>di<sub>c</sub>ti<sub>on,</sub> $i . e . , \lambda = 1 / 2$ <sub>,</sub> b<sub>u</sub>t l<sub>eaves open w</sub>h<sub>e</sub>th<sub>er</sub> <sub>o</sub>th<sub>er c</sub>h<sub>o</sub>i<sub>ces o</sub>f � <sub>can a</sub>d<sub>m</sub>it th<sub>e same</sub> b<sub>oun</sub>d<sub>.</sub> N<sub>ow, we s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>ey canno</sub>t<sub>:</sub> f<sub>or every</sub> fi<sub>xe</sub>d $\lambda \geq 0$ <sub>w</sub>ith $\lambda \neq 1 / 2$ , we <sub>cons</sub>t<sub>ruc</sub>t <sub>a</sub> b<sub>oun</sub>d<sub>e</sub>d i<sub>n</sub>fi<sub>n</sub>it<sub>e u</sub>tilit<sub>y sequence on w</sub>hi<sub>c</sub>h Alt<sub>-</sub>RVU f<sub>a</sub>il<sub>s.</sub>

Farina et al. [2023, Theorem 3.1] construct finite utilit<sub>y</sub> se<sub>q</sub>uences on which the strate<sub>g</sub>ies <sub>g</sub>enerated b<sub>y</sub> $\mathbb { R } \mathbb { M } ^ { + }$ <sub>an</sub>d $\mathrm { P R M ^ { + } }$ <sub>repea</sub>t<sub>e</sub>dl<sub>y a</sub>lt<sub>erna</sub>t<sub>e</sub> b<sub>e</sub>t<sub>ween</sub> $( 1 / 2 , 1 / 2 )$ and (0, 1). Normalization can produce such movements from small changes in the <sub>p</sub>rediction re<sub>g</sub>ret vectors when their <sub>p</sub>ositive <sub>p</sub>arts have small norms. For PRM<sup>+</sup>, this dificult<sub>y</sub> also a<sub>pp</sub>ears in our lower bound on (II), which wei<sub>g</sub>hts squared strate<sub>gy</sub> movement b<sub>y</sub> max $\{ \| [ \pmb { r } ^ { ( t ) } ] _ { + } \| _ { 1 } , \| [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } \}$ <sub>.</sub> With<sub>ou</sub>t <sub>a pos</sub>iti<sub>ve</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>on</sub> th<sub>ese we</sub>i<sub>g</sub>ht<sub>s,</sub> thi<sub>s es</sub>ti<sub>ma</sub>t<sub>e</sub> d<sub>oes no</sub>t <sub>prov</sub>id<sub>e a</sub> fi<sub>xe</sub>d <sub>pos</sub>iti<sub>ve coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>on unwe</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>movemen</sub>t<sub>.</sub> Th<sub>e</sub>i<sub>r</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> <sub>a</sub>l<sub>so</sub> <sub>ru</sub>l<sub>es</sub> <sub>ou</sub>t <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d RVU b<sub>oun</sub>d<sub>s</sub> <sub>w</sub>ith <sub>cons</sub>t<sub>an</sub>t<sub>s</sub> <sub>c</sub>h<sub>osen</sub> <sub>un</sub>if<sub>orm</sub>l<sub>y</sub> <sub>across</sub> utilit<sub>y</sub> se<sub>q</sub>uences. Rescalin<sub>g</sub> their <sub>g</sub>eometricall<sub>y</sub> <sub>g</sub>rowin<sub>g</sub> se<sub>q</sub>uences as in Farina et al. [2023, Remark B.1] <sub>g</sub>ives finite <sub>u</sub>tilit<sub>y</sub> <sub>sequences</sub> i<sub>n</sub> $[ - 1 , 1 ] ^ { 2 }$ <sub>w</sub>ith <sub>un</sub>if<sub>orm</sub>l<sub>y</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>or</sub>di<sub>nary regre</sub>t <sub>an</sub>d <sub>square</sub>d <sub>u</sub>tilit<sub>y var</sub>i<sub>a</sub>ti<sub>on, w</sub>hil<sub>e cumu</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>square</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>movemen</sub>t <sub>grows</sub> li<sub>near</sub>l<sub>y.</sub> E<sub>x</sub>t<sub>en</sub>di<sub>ng</sub> <sub>eac</sub>h <sub>sequence</sub> b<sub>y</sub> <sub>zero</sub> <sub>u</sub>tiliti<sub>es</sub> th<sub>ere</sub>f<sub>ore</sub> i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t<sub>,</sub> f<sub>or</sub> <sub>eac</sub>h <sub>o</sub>f $\mathbb { R } \mathbb { M } ^ { + }$ and PRM<sup>+</sup>,

$$
\forall \alpha , \beta , \gamma > 0 , ~ \exists ( u ^ { ( t ) } ) _ { t \geq 1 } \subset [ - 1 , 1 ] ^ { 2 } , ~ \exists T \geq 1 : \mathbf { R e g } ( T ) > \alpha + \beta \sum _ { t = 2 } ^ { T } \Vert u ^ { ( t ) } - u ^ { ( t - 1 ) } \Vert _ { 2 } ^ { 2 } - \gamma \sum _ { t = 2 } ^ { T } \Vert x ^ { ( t ) } - x ^ { ( t - 1 ) } \Vert _ { 2 } ^ { 2 } .
$$

W<sub>e see</sub>k <sub>an ana</sub>l<sub>ogous</sub> f<sub>a</sub>il<sub>ure resu</sub>lt f<sub>or</sub> Alt<sub>-</sub>RVU<sub>, w</sub>h<sub>ose</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on a</sub>ll<sub>ows</sub> th<sub>e cons</sub>t<sub>an</sub>t<sub>s</sub> t<sub>o</sub> d<sub>epen</sub>d <sub>on</sub> th<sub>e u</sub>tilit<sub>y</sub> <sub>sequence.</sub> E<sub>s</sub>t<sub>a</sub>bli<sub>s</sub>hi<sub>ng suc</sub>h <sub>a resu</sub>lt th<sub>ere</sub>f<sub>ore requ</sub>i<sub>res more</sub> th<sub>an rep</sub>l<sub>ac</sub>i<sub>ng or</sub>di<sub>nary regre</sub>t b<sub>y a</sub>lt<sub>erna</sub>ti<sub>ng regre</sub>t <sub>an</sub>d <sub>c</sub>h<sub>an</sub> i<sub>n</sub> th<sub>e</sub> i<sub>n</sub>di<sub>ces.</sub> W<sub>e mus</sub>t <sub>cons</sub>t<sub>ruc</sub>t <sub>a s</sub>i<sub>n</sub> l<sub>e</sub> b<sub>oun</sub>d<sub>e</sub>d i<sub>n</sub>fi<sub>n</sub>it<sub>e u</sub>tilit <sub>se uence on w</sub>hi<sub>c</sub>h th<sub>e</sub> Alt<sub>-</sub>RVU b<sub>oun</sub>d f<sub>a</sub>il<sub>s</sub>

f<sub>or</sub> <sub>every</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>o</sub>f h<sub>or</sub>i<sub>zon-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>pos</sub>iti<sub>ve</sub> <sub>cons</sub>t<sub>an</sub>t<sub>s,</sub> th<sub>a</sub>t i<sub>s,</sub>

$$
\exists ( { \boldsymbol { \mathfrak { a } } } ^ { ( t ) } ) _ { t \geq 1 } \subset [ - 1 , 1 ] ^ { d } , \forall { \boldsymbol { \alpha } } , { \boldsymbol { \beta } } , { \boldsymbol { \gamma } } > 0 , \ \exists T \geq 1 : { \mathrm { A l t } } \mathbf { \cdot R e g } ( T ) > { \boldsymbol { \alpha } } + { \boldsymbol { \beta } } \sum _ { t = 2 } ^ { T } \lVert { \boldsymbol { \mathfrak { u } } } ^ { ( t ) } - { \boldsymbol { \mathfrak { u } } } ^ { ( t - 1 ) } \rVert _ { 2 } ^ { 2 } - \gamma \sum _ { t = 1 } ^ { T } \lVert { \boldsymbol { \mathfrak { x } } } ^ { ( t + 1 ) } - { \boldsymbol { \mathbf { x } } } ^ { ( t ) } \rVert _ { 2 } ^ { 2 } .
$$

W<sub>e cons</sub>t<sub>ruc</sub>t <sub>one suc</sub>h <sub>u</sub>tilit<sub>y sequence</sub> f<sub>or every</sub> $\lambda \neq 1 / 2$ <sub>.</sub> F<sub>or</sub> $\lambda > 1 / 2 ,$ <sub>,</sub> <sub>we</sub> <sub>ma</sub>k<sub>e</sub> th<sub>e</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>norms</sub> d<sub>ecrease</sub> <sub>geome</sub>t<sub>r</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es</sub> <sub>repea</sub>t<sub>.</sub>

Proposition 6.1. For every fixed $\lambda > 1 / 2$ , there is a utility sequence $( \pmb { u } ^ { ( t ) } ) _ { t \geq 1 } \subset [ - 1 , 1 ] ^ { 2 }$ , on which the scaled prediction regret matching<sup>+</sup> learner satisfies

$$
\sum _ { t = 1 } ^ { \infty } \| { \pmb u } ^ { ( t ) } \| _ { 2 } < \infty , \ \operatorname* { s u p } _ { T \geq 1 } | \mathrm { A l t } \mathrm { - } \mathrm { R e g } ( T ) | < \infty , \ \sum _ { t = 1 } ^ { T } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } = \Theta ( T ) .\tag{25}
$$

Consequently, Alt-RVU fails on this sequence for every choice of horizon-independent constants.

W<sub>e v</sub>i<sub>sua</sub>li<sub>ze</sub> thi<sub>s cons</sub>t<sub>ruc</sub>ti<sub>on</sub> i<sub>n</sub> Fi <sub>ure</sub> 3<sub>.</sub> I<sub>n</sub>t<sub>u</sub>iti<sub>ve</sub>l <sub>, we cons</sub>t<sub>ruc</sub>t <sub>success</sub>i<sub>ve</sub> bl<sub>oc</sub>k<sub>s w</sub>ith <sub>s</sub>i<sub>m</sub>il<sub>ar s</sub>i<sub>zes</sub> th<sub>a</sub>t <sub>mu</sub>lti l th<sub>e re re</sub>t <sub>vec</sub>t<sub>ors</sub> $\tilde { r } ^ { ( t ) }$ <sub>an</sub>d <sub>u</sub>tiliti<sub>es</sub> $\mathbf { \boldsymbol { u } } ^ { ( t ) }$ b th<sub>e same</sub> f<sub>ac</sub>t<sub>or</sub> $\rho \in ( 0 , 1 )$ <sub>a</sub>t th<sub>e same os</sub>iti<sub>ons across</sub> dif<sub>eren</sub>t bl<sub>oc</sub>k<sub>s.</sub> N<sub>orma</sub>li<sub>za</sub>ti<sub>on</sub> <sub>removes</sub> thi<sub>s</sub> f<sub>ac</sub>t<sub>or,</sub> <sub>so</sub> th<sub>e</sub> <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es</sub> $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ c<sub>y</sub>c<sup>l</sup>e.

![](images/335f765c80adba0beaacf15e5486e71631e63d321248ddff3d3a6b74dd64516a.jpg)  
Fi<sub>gure</sub> 3<sub>:</sub> Th<sub>e cons</sub>t<sub>ruc</sub>ti<sub>on o</sub>f P<sub>ropos</sub>iti<sub>on</sub> 6<sub>.</sub>1 <sub>w</sub>ith 2 <sub>ac</sub>ti<sub>ons.</sub>

T<sub>a</sub>ki<sub>ng</sub> $\lambda = 1$ i<sub>n</sub> P<sub>ropos</sub>iti<sub>on</sub> 6<sub>.</sub>1 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t $\mathrm { P R M ^ { + } }$ f<sub>a</sub>il<sub>s</sub> Alt<sub>-</sub>RVU<sub>.</sub> Thi<sub>s cons</sub>t<sub>ruc</sub>ti<sub>on c</sub>l<sub>ar</sub>ifi<sub>es w</sub>h<sub>y con</sub>t<sub>ro</sub>lli<sub>ng</sub> th<sub>e</sub> norm decrease in (I) alone is insuficient. Both (I) and (II) remain bounded on this sequence, since their contributions over successive blocks deca<sub>y g</sub>eometricall<sub>y</sub>. Thus, the ne<sub>g</sub>ative contribution of (I) can be absorbed into a horizoni<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>cons</sub>t<sub>an</sub>t<sub>.</sub>

The remainin<sub>g</sub> dificult<sub>y</sub> is that our lower bound on (II) su<sub>pp</sub>lies a strate<sub>gy</sub>-movement coeficient <sub>p</sub>ro<sub>p</sub>ortional to th<sub>e pre</sub>di<sub>c</sub>ti<sub>on regre</sub>t <sub>vec</sub>t<sub>or norms:</sub>

$$
\frac { 1 } { 6 d } \operatorname* { m a x } \{ \| [ \pmb { r } ^ { ( t ) } ] _ { + } \| _ { 1 } , \| [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } \}  0 .
$$

Th<sub>ese coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> d<sub>ecay geome</sub>t<sub>r</sub>i<sub>ca</sub>ll<sub>y across</sub> bl<sub>oc</sub>k<sub>s, w</sub>hil<sub>e</sub> th<sub>e same nonzero s</sub>t<sub>ra</sub>t<sub>egy movemen</sub>t<sub>s repea</sub>t<sub>.</sub> C<sub>onsequen</sub>tl<sub>y,</sub> the wei<sub>g</sub>hted movement sum remains bounded even thou<sub>g</sub>h the unwei<sub>g</sub>hted sum <sub>g</sub>rows linearl<sub>y</sub>. Since (II) itself remains b<sub>oun</sub>d<sub>e</sub>d<sub>,</sub> thi<sub>s canno</sub>t b<sub>e repa</sub>i<sub>re</sub>d b<sub>y s</sub>h<sub>arpen</sub>i<sub>ng</sub> it<sub>s</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>.</sub> I<sub>n par</sub>ti<sub>cu</sub>l<sub>ar, no</sub> fi<sub>xe</sub>d $\gamma > 0$ <sub>an</sub>d $C < \infty$ <sub>can ma</sub>k<sub>e</sub>

$$
\sum _ { t = t _ { 0 } } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + 2 \| [ r ^ { ( t + 1 ) } ] _ { + } - \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } \geq \gamma \sum _ { t = t _ { 0 } } ^ { T } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } - C
$$

h<sub>o</sub>ld f<sub>or</sub> <sub>every</sub> $T \geq t _ { 0 }$ <sub>,</sub> b<sub>ecause</sub> th<sub>e</sub> l<sub>e</sub>ft<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e rema</sub>i<sub>ns</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>w</sub>hil<sub>e</sub> th<sub>e r</sub>i ht<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e rows</sub> li<sub>near</sub>l <sub>w</sub>ith �<sub>.</sub> F<sub>or</sub> $0 \leq \lambda < 1 / 2$ <sub>, we</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>cons</sub>t<sub>ruc</sub>t <sub>a</sub>lt<sub>erna</sub>ti<sub>ng regre</sub>t <sub>o</sub>f <sub>or</sub>d<sub>er</sub> $\sqrt { T }$ <sub>w</sub>ith <sub>on</sub>l<sub>y</sub> l<sub>ogar</sub>ith<sub>m</sub>i<sub>c square</sub>d <sub>u</sub>tilit<sub>y var</sub>i<sub>a</sub>ti<sub>on.</sub> Thi<sub>s ru</sub>l<sub>es ou</sub>t <sub>even</sub> th<sub>e upper</sub> b<sub>oun</sub>d <sub>o</sub>bt<sub>a</sub>i<sub>ne</sub>d b<sub>y remov</sub>i<sub>ng</sub> th<sub>e nega</sub>ti<sub>ve movemen</sub>t t<sub>erm</sub> f<sub>rom</sub> Alt<sub>-</sub>RVU<sub>.</sub>

Proposition 6.2. $F i x 0 \le \lambda < 1 / 2$ and an integer $d \ge 4$ satisfying $d > ( ( 1 + 2 \lambda ) / ( 1 - 2 \lambda ) ) ^ { 2 }$ . There is a utility sequence $( \pmb { u } ^ { ( t ) } ) _ { t \geq 1 } \subset [ - 1 , 1 ] ^ { d }$ on which the scaled-prediction regret matching<sup>+</sup> learner satisfies

$$
\mathrm { A l t - R e g } ( T ) = \Theta ( \sqrt { T } ) , \ \sum _ { t = 2 } ^ { T } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } = O ( \log T ) .\tag{26}
$$

Consequently, Alt-RVU fails on this sequence for every choice of horizon-independent constants.

T<sub>a</sub>ki<sub>ng</sub> $\lambda = 1 / 3$ in Pro<sub>p</sub>osition 6.2 shows that the local learner underl<sub>y</sub>in<sub>g</sub> the SAPCFR<sup>+</sup> al<sub>g</sub>orithm of Men<sub>g</sub> et al. [2026] fails Alt-RVU in ever<sub>y</sub> dimension $d \ge 2 6$ <sub>.</sub> T<sub>oge</sub>th<sub>er w</sub>ith th<sub>e pos</sub>iti<sub>ve resu</sub>lt i<sub>n</sub> S<sub>u</sub>b<sub>sec</sub>ti<sub>on</sub> 5<sub>.</sub>3<sub>,</sub> P<sub>ropos</sub>iti<sub>ons</sub> 6<sub>.</sub>1 <sub>an</sub>d 6<sub>.</sub>2 id<sub>en</sub>tif<sub>y</sub> $\lambda = 1 / 2$ as the unique fixed multiplier for which Alt-RVU holds on every bounded utility sequence in <sub>every</sub> fi<sub>n</sub>it<sub>e</sub> di<sub>mens</sub>i<sub>on.</sub>

Alt<sub>-</sub>RVU i<sub>s a su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>cr</sub>it<sub>er</sub>i<sub>on</sub> f<sub>or</sub> ${ \cal O } ( T ^ { - 1 } )$ <sub>average-</sub>it<sub>era</sub>t<sub>e convergence, an</sub>d it<sub>s</sub> f<sub>a</sub>il<sub>ure</sub> d<sub>oes no</sub>t <sub>ru</sub>l<sub>e ou</sub>t <sub>s</sub>l<sub>ower</sub> <sub>convergence</sub> <sub>guaran</sub>t<sub>ees</sub> f<sub>or</sub> <sub>o</sub>th<sub>er</sub> <sub>mu</sub>lti<sub>p</sub>li<sub>ers.</sub> A<sub>ny</sub> ${ \cal O } ( T ^ { - 1 } )$ <sub>guaran</sub>t<sub>ee</sub> f<sub>or ano</sub>th<sub>er</sub> fi<sub>xe</sub>d <sub>mu</sub>lti<sub>p</sub>li<sub>er across a</sub>ll fi<sub>n</sub>it<sub>e</sub> <sub>games mus</sub>t <sub>exp</sub>l<sub>o</sub>it <sub>a</sub>dditi<sub>ona</sub>l <sub>s</sub>t<sub>ruc</sub>t<sub>ure o</sub>f <sub>se</sub>lf<sub>-p</sub>l<sub>ay or use an ana</sub>l<sub>ys</sub>i<sub>s</sub> b<sub>eyon</sub>d <sub>our genera</sub>l Alt<sub>-</sub>RVU <sub>ana</sub>l<sub>ys</sub>i<sub>s.</sub> I<sub>n</sub> S<sub>u</sub>b<sub>sec</sub>ti<sub>on</sub> 9<sub>.</sub>1<sub>, we comp</sub>l<sub>emen</sub>t th<sub>ese</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>resu</sub>lt<sub>s w</sub>ith <sub>numer</sub>i<sub>ca</sub>l <sub>separa</sub>ti<sub>ons</sub> i<sub>n average- an</sub>d l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or</sub> <sub>un</sub>d<sub>er a</sub>lt<sub>erna</sub>ti<sub>ng se</sub>lf<sub>-p</sub>l<sub>ay, compar</sub>i<sub>ng</sub> h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on w</sub>ith <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>mu</sub>lti<sub>p</sub>li<sub>ers.</sub>

## 7 Best- and Last-Iterate Convergence

Th<sub>e</sub> <sub>prece</sub>di<sub>ng</sub> di<sub>scuss</sub>i<sub>on</sub> <sub>prov</sub>id<sub>e</sub>d th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>average-</sub>it<sub>era</sub>t<sub>e</sub> <sub>convergence</sub> ${ \cal O } ( T ^ { - 1 } )$ f<sub>or</sub> $\mathsf { A l t { - } I R { - } P R M ^ { + } }$ <sub>an</sub>d $\mathsf { A l t { - } H P R M ^ { + } }$ A<sub>verage-</sub>it<sub>era</sub>t<sub>e</sub> <sub>convergence,</sub> h<sub>owever,</sub> d<sub>oes</sub> <sub>no</sub>t f<sub>u</sub>ll<sub>y</sub> <sub>cap</sub>t<sub>ure</sub> th<sub>e</sub> <sub>prac</sub>ti<sub>ca</sub>l <sub>per</sub>f<sub>ormance</sub> <sub>o</sub>f <sub>se</sub>lf<sub>-p</sub>l<sub>ay,</sub> <sub>as</sub> <sub>averag</sub>i<sub>ng</sub> <sub>can</sub> retain the influence of <sub>p</sub>oor earl<sub>y</sub> strate<sub>g</sub>ies and conver<sub>g</sub>e more slowl<sub>y</sub> than the last iterate in standard benchmarks [Cai et al., 2025, Zhan<sub>g</sub> et al., 2025]. Our ex<sub>p</sub>eriments in Section 9 also exhibit this behavior, reinforcin<sub>g</sub> the im<sub>p</sub>ortance of <sub>a</sub>l<sub>so</sub> <sub>s</sub>t<sub>u</sub>d<sub>y</sub>i<sub>ng</sub> b<sub>es</sub>t<sub>-</sub> <sub>an</sub>d l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e</sub> <sub>convergence.</sub>

N<sub>ow,</sub> t<sub>o</sub> <sub>s</sub>t<sub>u</sub>d<sub>y</sub> b<sub>es</sub>t<sub>-</sub> <sub>an</sub>d l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e</sub> <sub>convergence,</sub> <sub>we</sub> <sub>mus</sub>t f<sub>ocus</sub> <sub>on</sub> th<sub>e</sub> N<sub>as</sub>h <sub>gap</sub> <sub>a</sub>t th<sub>e</sub> �<sub>-</sub>th it<sub>era</sub>ti<sub>on.</sub> Thi<sub>s</sub> N<sub>as</sub>h <sub>gap</sub> i<sub>s</sub> <sub>exac</sub>tl<sub>y</sub> th<sub>e</sub> <sub>sum</sub> <sub>o</sub>f th<sub>e</sub> <sub>p</sub>l<sub>ayers</sub>’ b<sub>es</sub>t<sub>-response</sub> <sub>errors</sub> <sub>a</sub>t th<sub>a</sub>t it<sub>era</sub>ti<sub>on:</sub>

$$
\mathrm { G a p } ( \pmb { x } ^ { ( t ) } , \pmb { y } ^ { ( t ) } ) = \operatorname* { m a x } _ { z \in X } \langle \mathbf { A } \pmb { y } ^ { ( t ) } , z - \pmb { x } ^ { ( t ) } \rangle + \operatorname* { m a x } _ { z \in \pmb { y } } \langle - \mathbf { A } ^ { \top } \pmb { x } ^ { ( t ) } , z - \pmb { y } ^ { ( t ) } \rangle .
$$

H<sub>ere,</sub> <sub>one</sub> <sub>m</sub>i<sub>g</sub>ht t<sub>ry</sub> t<sub>o</sub> b<sub>oun</sub>d th<sub>e</sub> t<sub>wo</sub> t<sub>erms</sub> <sub>on</sub> th<sub>e</sub> RHS<sub>.</sub> H<sub>owever,</sub> th<sub>ese</sub> t<sub>erms</sub> d<sub>o</sub> <sub>no</sub>t di<sub>rec</sub>tl<sub>y</sub> <sub>re</sub>fl<sub>ec</sub>t th<sub>e</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub> di<sub>scusse</sub>d i<sub>n</sub> S<sub>u</sub>b<sub>sec</sub>ti<sub>on</sub> 5<sub>.</sub>1<sub>.</sub> S<sub>pec</sub>ifi<sub>ca</sub>ll<sub>y,</sub> $\mathbf { y } ^ { ( t ) }$ i<sub>s</sub> <sub>p</sub>l<sub>aye</sub>d <sub>aga</sub>i<sub>ns</sub>t b<sub>o</sub>th $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ <sub>an</sub>d $\boldsymbol { x } ^ { ( t + 1 ) }$ <sub>,</sub> <sub>w</sub>h<sub>ereas</sub> th<sub>e</sub> fi<sub>rs</sub>t t<sub>erm</sub> i<sub>nvo</sub>l<sub>ves on</sub>l<sub>y</sub> $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ <sub>.</sub> Lik<sub>ew</sub>i<sub>se,</sub> $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ i<sub>s p</sub>l<sub>aye</sub>d <sub>aga</sub>i<sub>ns</sub>t b<sub>o</sub>th $\mathbf { y } ^ { ( t ^ { - 1 } ) }$ <sub>an</sub>d $\mathbf { y } ^ { ( t ) }$ <sub>, w</sub>h<sub>ereas</sub> th<sub>e secon</sub>d t<sub>erm</sub> i<sub>nvo</sub>l<sub>ves on</sub>l<sub>y</sub> $\mathbf { y } ^ { ( t ) }$

O<sub>ur</sub> k<sub>ey</sub> i<sub>ns</sub>i<sub>g</sub>ht i<sub>s</sub> t<sub>o</sub> b<sub>oun</sub>d th<sub>e</sub> N<sub>as</sub>h <sub>gap a</sub>t th<sub>e</sub> �<sub>-</sub>th it<sub>era</sub>ti<sub>on w</sub>ith t<sub>erms</sub> th<sub>a</sub>t <sub>are more na</sub>t<sub>ura</sub>l t<sub>o a</sub>lt<sub>erna</sub>ti<sub>ng up</sub>d<sub>a</sub>t<sub>e</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure.</sub> W<sub>e no</sub>ti<sub>ce</sub>d th<sub>a</sub>t

$$
\operatorname* { m a x } _ { z \in X } \langle \mathbf { A } \mathbf { y } ^ { ( t ) } , z - \mathbf { x } ^ { ( t + 1 ) } \rangle + \operatorname* { m a x } _ { z \in X } \langle \mathbf { A } \mathbf { y } ^ { ( t ) } , z - \mathbf { x } ^ { ( t ) } \rangle = \operatorname* { m a x } _ { z \in X } \langle \mathbf { A } \mathbf { y } ^ { ( t ) } , 2 z - \mathbf { x } ^ { ( t ) } - \mathbf { x } ^ { ( t + 1 ) } \rangle .
$$

A<sub>na</sub>l<sub>ogous</sub>l<sub>y,</sub>

$$
\operatorname* { m a x } _ { z \in \mathcal { Y } } \langle - \mathbf { A } ^ { \top } { \pmb x } ^ { ( t ) } , z - { \mathbf y } ^ { ( t - 1 ) } \rangle + \operatorname* { m a x } _ { z \in \mathcal { Y } } \langle - \mathbf { A } ^ { \top } { \pmb x } ^ { ( t ) } , z - { \mathbf y } ^ { ( t ) } \rangle = \operatorname* { m a x } _ { z \in \mathcal { Y } } \langle - \mathbf { A } ^ { \top } { \pmb x } ^ { ( t ) } , 2 z - { \mathbf y } ^ { ( t ) } - { \mathbf y } ^ { ( t - 1 ) } \rangle .
$$

With th<sub>ese</sub> id<sub>en</sub>titi<sub>es, we can</sub> b<sub>oun</sub>d th<sub>e</sub> N<sub>as</sub>h <sub>gap a</sub>t th<sub>e</sub> �<sub>-</sub>th it<sub>era</sub>ti<sub>on as</sub> f<sub>o</sub>ll<sub>ows.</sub>

$$
\begin{array} { r l } & { \mathrm { G a p } ( { \pmb x } ^ { ( t ) } , { \pmb y } ^ { ( t ) } ) \leq \mathrm { G a p } ( { \pmb x } ^ { ( t ) } , { \pmb y } ^ { ( t ) } ) + \underset { z \in { \pmb X } } { \operatorname* { m a x } } \langle { \bf A } { \pmb y } ^ { ( t ) } , z - { \pmb x } ^ { ( t + 1 ) } \rangle + \underset { z \in { \pmb Y } } { \operatorname* { m a x } } \langle - { \bf A } ^ { \top } { \pmb x } ^ { ( t ) } , z - { \pmb y } ^ { ( t - 1 ) } \rangle } \\ & { \qquad = \underset { z \in { \pmb X } } { \operatorname* { m a x } } \langle { \bf A } { \pmb y } ^ { ( t ) } , 2 z - { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t + 1 ) } \rangle + \underset { z \in { \pmb Y } } { \operatorname* { m a x } } \langle - { \bf A } ^ { \top } { \pmb x } ^ { ( t ) } , 2 z - { \pmb y } ^ { ( t ) } - { \pmb y } ^ { ( t - 1 ) } \rangle . } \end{array}
$$

Thi<sub>s</sub> b<sub>oun</sub>d id<sub>en</sub>tifi<sub>es</sub> th<sub>e</sub> <sub>quan</sub>tit<sub>y</sub> <sub>m</sub>i<sub>ss</sub>i<sub>ng</sub> f<sub>rom</sub> Alt<sub>-</sub>RVU<sub>.</sub> It<sub>s</sub> <sub>express</sub>i<sub>on</sub> i<sub>s</sub> th<sub>e</sub> <sub>same</sub> <sub>as</sub> th<sub>e</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>o</sub>f <sub>one</sub> observation to alternating regret, but the comparator � is now maximized separately at each iteration. A bound on <sub>a</sub>lt<sub>erna</sub>ti<sub>ng regre</sub>t h<sub>ence</sub> d<sub>oes no</sub>t di<sub>rec</sub>tl<sub>y con</sub>t<sub>ro</sub>l th<sub>ese</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>max</sub>i<sub>ma.</sub>

F<sub>or</sub> b<sub>es</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e convergence, we see</sub>k <sub>a</sub>t l<sub>eas</sub>t <sub>one</sub> it<sub>era</sub>ti<sub>on w</sub>ith <sub>a sma</sub>ll N<sub>as</sub>h <sub>gap.</sub> F<sub>or</sub> l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e convergence</sub> t<sub>o</sub> th<sub>e</sub> N<sub>as</sub>h <sub>se</sub>t<sub>, we nee</sub>d th<sub>e gap</sub> t<sub>o</sub> b<sub>ecome sma</sub>ll <sub>a</sub>t <sub>every su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> l<sub>a</sub>t<sub>e</sub> it<sub>era</sub>ti<sub>on.</sub> A b<sub>oun</sub>d <sub>on</sub> th<sub>e sum o</sub>f <sub>square</sub>d <sub>errors</sub> i<sub>s a</sub>bl<sub>e</sub> t<sub>o prov</sub>id<sub>e</sub> b<sub>o</sub>th<sub>.</sub> S<sub>pec</sub>ifi<sub>ca</sub>ll<sub>y, we</sub> h<sub>ave</sub>

$$
\sum _ { t = 1 } ^ { T } \mathrm { G a p } ( { \pmb x } ^ { ( t ) } , { \pmb y } ^ { ( t ) } ) ^ { 2 } \leq 2 \sum _ { t = 1 } ^ { T } \left[ \operatorname* { m a x } \langle { \bf A } { \pmb y } ^ { ( t ) } , 2 { \pmb z } - { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t + 1 ) } \rangle \right] ^ { 2 } + 2 \sum _ { t = 1 } ^ { T } \left[ \operatorname* { m a x } \langle - { \bf A } ^ { \top } { \pmb x } ^ { ( t ) } , 2 { \pmb z } - { \pmb y } ^ { ( t - 1 ) } - { \pmb y } ^ { ( t ) } \rangle \right] ^ { 2 } .\tag{27}
$$

It th<sub>ere</sub>f<sub>ore</sub> <sub>su</sub>fi<sub>ces</sub> t<sub>o</sub> b<sub>oun</sub>d <sub>eac</sub>h <sub>o</sub>f th<sub>e</sub> t<sub>wo</sub> <sub>sums</sub> <sub>on</sub> th<sub>e</sub> <sub>r</sub>i<sub>g</sub>ht h<sub>an</sub>d <sub>s</sub>id<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> <sub>o</sub>f �<sub>.</sub> Thi<sub>s</sub> <sub>mo</sub>ti<sub>va</sub>t<sub>es</sub> <sub>us</sub> t<sub>o</sub> <sub>re</sub>fi<sub>ne</sub> th<sub>e</sub> Alt<sub>-</sub>RVU b<sub>oun</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows.</sub>

Definition 7.1 (Refined Alt-RVU). <sup>1</sup> A learner satisfies the refined Alt-RVU bound on a bounded utility sequence if th<sub>ere are cons</sub>t<sub>an</sub>t<sub>s</sub> $\alpha , \beta , \gamma , \zeta > 0 .$ <sub>,</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f �<sub>,</sub> <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub> f<sub>or</sub> <sub>every</sub> $T \geq 1$

$$
\mathrm { A l t - R e g } ( T ) + \zeta \sqrt { \sum _ { t = 1 } ^ { T } \left[ \operatorname* { m a x } _ { z \in \Lambda ( d ) } \left. u ^ { ( t ) } , 2 z - x ^ { ( t ) } - x ^ { ( t + 1 ) } \right. \right] ^ { 2 } } \leq \alpha + \beta \sum _ { t = 2 } ^ { T } \| u ^ { ( t ) } - u ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } - \gamma \sum _ { t = 1 } ^ { T } \| x ^ { ( t + 1 ) } - x ^ { ( t ) } \| _ { 2 } ^ { 2 } .\tag{28}
$$

Th<sub>e</sub> <sub>a</sub>dditi<sub>ona</sub>l <sub>secon</sub>d t<sub>erm</sub> d<sub>oes</sub> <sub>no</sub>t <sub>rep</sub>l<sub>ace</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>regre</sub>t<sub>.</sub> I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> <sub>max</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> <sub>separa</sub>t<sub>e</sub>l<sub>y</sub> <sub>a</sub>t <sub>eac</sub>h it<sub>era</sub>ti<sub>on</sub> <sub>an</sub>d <sub>app</sub>l<sub>y</sub>i<sub>ng</sub> C<sub>auc</sub>h<sub>y-</sub>S<sub>c</sub>h<sub>warz, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\mathrm { A l t - R e g } ( T ) \leq \sum _ { t = 1 } ^ { T } \operatorname* { m a x } _ { z \in \Delta ( d ) } \left. u ^ { ( t ) } , 2 z - x ^ { ( t ) } - { \mathbf { x } } ^ { ( t + 1 ) } \right. \leq { \sqrt { T } } { \sqrt { \sum _ { t = 1 } ^ { T } \left[ \operatorname* { m a x } _ { z \in \Delta ( d ) } \left. u ^ { ( t ) } , 2 z - x ^ { ( t ) } - x ^ { ( t + 1 ) } \right. \right] ^ { 2 } } } .
$$

Th<sub>us,</sub> b<sub>oun</sub>di<sub>ng</sub> th<sub>e square-roo</sub>t t<sub>erm</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y o</sub>f � <sub>g</sub>i<sub>ves on</sub>l<sub>y an</sub> ${ \cal O } ( T ^ { 1 / 2 } )$ u<sub>pp</sub>er <sup>b</sup>oun<sup>d</sup> on a<sup>l</sup>ternat<sup>i</sup>n<sub>g</sub> re<sub>g</sub>ret, <sub>w</sub>hi<sub>c</sub>h <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> <sub>an</sub> ${ \cal O } ( T ^ { - \bar { 1 } / 2 } )$ <sub>average-</sub>it<sub>era</sub>t<sub>e guaran</sub>t<sub>ee ra</sub>th<sub>er</sub> th<sub>an</sub> ${ \cal O } ( T ^ { - 1 } )$ <sub>.</sub> C<sub>onverse</sub>l<sub>y,</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>regre</sub>t d<sub>oes</sub> <sub>no</sub>t <sub>con</sub>t<sub>ro</sub>l th<sub>e</sub> b<sub>es</sub>t<sub>-response errors a</sub>t i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l it<sub>era</sub>ti<sub>ons.</sub> R<sub>e</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> b<sub>o</sub>th t<sub>erms a</sub>ll<sub>ows us</sub> t<sub>o es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>a</sub>ll th<sub>e</sub> d<sub>es</sub>i<sub>re</sub>d gu<sup>arantees</sup>.

W<sub>e now s</sub>h<sub>ow</sub> h<sub>ow</sub> thi<sub>s re</sub>fi<sub>ne</sub>d Alt<sub>-</sub>RVU <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> b<sub>es</sub>t<sub>- an</sub>d l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e convergence</sub> t<sub>o</sub> th<sub>e</sub> N<sub>as</sub>h <sub>se</sub>t<sub>.</sub>

Proposition 7.2. Suppose both players in afinite two-player zero-sum normal-form game satisfy the refined Alt-RVU bound under alternating updates, with common constants �, $\beta , \gamma , \zeta > 0$ satisfying $\beta \| \mathbf { A } \| _ { 2 } ^ { 2 } \leq \gamma$ . Then, we have

$$
\sum _ { t = 1 } ^ { \infty } \mathrm { G a p } ( { \pmb x } ^ { ( t ) } , { \pmb y } ^ { ( t ) } ) ^ { 2 } \leq \left( \frac { 2 \alpha + \mathrm { r a n g e } ( \mathbf { A } ) } { \zeta } \right) ^ { 2 } .
$$

Consequently, for every $T \geq 1$ , we obtain

$$
\operatorname* { m i n } _ { 1 \leq t \leq T } \mathrm { G a p } ( { \pmb x } ^ { ( t ) } , { \pmb y } ^ { ( t ) } ) \leq \frac { 2 \alpha + \mathrm { r a n g e } ( { \bf A } ) } { \zeta \sqrt { T } } ,
$$

which implies ${ \cal O } ( T ^ { - 1 / 2 } )$ best-iterate rate, and the last iterates converge to the Nash set $z ^ { \ast }$

W<sub>e</sub> fi<sub>rs</sub>t <sub>re</sub>l<sub>a</sub>t<sub>e</sub> th<sub>ese</sub> b<sub>es</sub>t<sub>-response</sub> <sub>errors</sub> t<sub>o</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> i<sub>ncremen</sub>t<sub>s.</sub>

Lemma 7.3. IR-PRM<sup>+</sup> and HPRM<sup>+</sup> learners satisfy,for every $t \geq 1$

$$
\operatorname* { m a x } _ { z \in \Delta ( d ) } \langle { \pmb u } ^ { ( t ) } , 2 z - { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t + 1 ) } \rangle \leq \sqrt { 5 } \| \widetilde { \pmb r } ^ { ( t + 1 ) } - \widetilde { \pmb r } ^ { ( t ) } \| _ { 2 } .\tag{29}
$$

For IR-PRM<sup>+</sup>, the constant $\sqrt { 5 }$ can be improved to 2.

Th<sub>e</sub> <sub>prece</sub>di<sub>ng</sub> l<sub>emma</sub> <sub>re</sub>l<sub>a</sub>t<sub>es</sub> th<sub>e</sub> <sub>a</sub>dditi<sub>ona</sub>l b<sub>es</sub>t<sub>-response</sub> <sub>errors</sub> t<sub>o</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> i<sub>ncremen</sub>t<sub>s.</sub> W<sub>e</sub> <sub>now</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> <sub>nega</sub>ti<sub>ve</sub> t<sub>erms</sub> <sub>re</sub>t<sub>a</sub>i<sub>ne</sub>d i<sub>n</sub> th<sub>e</sub> Alt<sub>-</sub>RVU <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>a</sub>l<sub>so</sub> <sub>con</sub>t<sub>ro</sub>l th<sub>ese</sub> i<sub>ncremen</sub>t<sub>s,</sub> <sub>y</sub>i<sub>e</sub>ldi<sub>ng</sub> th<sub>e</sub> <sub>re</sub>fi<sub>ne</sub>d Alt<sub>-</sub>RVU b<sub>oun</sub>d<sub>.</sub>

Proposition 7.4. Both $\widetilde { \cal I R } { - } P R M ^ { + }$ and HPRM<sup>+</sup> satisfy refined Alt-RVU. Specifically, for every bounded utility sequence, there exist $\gamma , \zeta > 0$ such that,for every $\beta > 0 ,$ , there exists $\alpha > 0 f o r$ which (28) holdsfor every $T \geq 1$ . These constants are independent of�.

P<sub>ropos</sub>iti<sub>on</sub> 7<sub>.</sub>4 <sub>a</sub>ll<sub>ows us</sub> t<sub>o c</sub>h<sub>oose common cons</sub>t<sub>an</sub>t<sub>s</sub> f<sub>or</sub> th<sub>e</sub> t<sub>wo p</sub>l<sub>ayers sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> $\beta \| \mathbf { A } \| _ { 2 } ^ { 2 } < \gamma$ <sub>.</sub> P<sub>ropos</sub>iti<sub>on</sub> 7<sub>.</sub>2 th<sub>en y</sub>i<sub>e</sub>ld<sub>s an</sub> ${ \cal O } ( T ^ { - 1 / 2 } )$ b<sub>es</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e ra</sub>t<sub>e an</sub>d <sub>conver ence o</sub>f th<sub>e</sub> l<sub>as</sub>t it<sub>era</sub>t<sub>es</sub> t<sub>o</sub> th<sub>e</sub> N<sub>as</sub>h <sub>se</sub>t ${ \boldsymbol { \mathcal { Z } } } ^ { * }$

C<sub>onvergence</sub> t<sub>o a s</sub>i<sub>ng</sub>l<sub>e equ</sub>ilib<sub>r</sub>i<sub>um,</sub> h<sub>owever, requ</sub>i<sub>res more</sub> th<sub>an re</sub>fi<sub>ne</sub>d Alt<sub>-</sub>RVU<sub>: van</sub>i<sub>s</sub>hi<sub>ng exp</sub>l<sub>o</sub>it<sub>a</sub>bilit<sub>y an</sub>d <sub>square-summa</sub>bl<sub>e s</sub>t<sub>ra</sub>t<sub>egy movemen</sub>t<sub>s</sub> d<sub>o no</sub>t <sub>ru</sub>l<sub>e ou</sub>t <sub>mu</sub>lti<sub>p</sub>l<sub>e equ</sub>ilib<sub>r</sub>i<sub>um accumu</sub>l<sub>a</sub>ti<sub>on po</sub>i<sub>n</sub>t<sub>s.</sub> W<sub>e es</sub>t<sub>a</sub>bli<sub>s</sub>h thi<sub>s</sub> stron<sub>g</sub>er conclusion b<sub>y</sub> further ex<sub>p</sub>loitin<sub>g</sub> the structure of the IR-PRM<sup>+</sup> and HPRM<sup>+</sup> u<sub>p</sub>dates, showin<sub>g</sub> that conver<sub>g</sub>ence <sub>o</sub>f <sub>a su</sub>b<sub>sequence</sub> t<sub>o a</sub> N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> f<sub>orces</sub> th<sub>e en</sub>ti<sub>re s</sub>t<sub>ra</sub>t<sub>egy sequence</sub> t<sub>o converge</sub> t<sub>o</sub> th<sub>e same equ</sub>ilib<sub>r</sub>i<sub>um.</sub>

Theorem 7.5. In every finite two-player zero-sum normal-form game, $_ { A } I t - I R - P R M ^ { + }$ and $A \mathsf { 1 } t \mathsf { - } H P R M ^ { \dagger }$ satisfy

$$
\operatorname* { m i n } _ { 1 \leq t \leq T } { \mathrm { G a p } ( { \pmb x } ^ { ( t ) } , { \pmb y } ^ { ( t ) } ) } = O ( T ^ { - 1 / 2 } ) a n d ( { \pmb x } ^ { ( t ) } , { \pmb y } ^ { ( t ) } )  ( { \pmb x } ^ { * } , { \pmb y } ^ { * } ) ,
$$

where $( \pmb { x } ^ { * } , \pmb { y } ^ { * } ) \in \mathcal { Z } ^ { * }$ is a Nash equilibrium.

Remark 7.6. Cai et al. [2025] establish similar convergence to a single equilibrium for $\mathbb { E } \mathbb { x } \mathbb { R } \mathbb { M } ^ { + }$ and smooth SPRM<sup>+</sup> under simultaneous u<sub>p</sub>dates, but both methods re<sub>q</sub>uire <sub>p</sub>a<sub>y</sub>of-de<sub>p</sub>endent ste<sub>p</sub> sizes. For IR variants, Sullins and Kash [2026] and Li and Huan<sub>g</sub> [2026] establish conver<sub>g</sub>ence to the Nash set for IR-PRM<sup>+</sup> and IREG-PRM<sup>+</sup>, res<sub>p</sub>ectivel<sub>y</sub>, leavin<sub>g</sub> <sub>convergence</sub> t<sub>o a s</sub>i<sub>ng</sub>l<sub>e equ</sub>ilib<sub>r</sub>i<sub>um unreso</sub>l<sub>ve</sub>d <sub>w</sub>h<sub>en</sub> th<sub>e game</sub> d<sub>oes no</sub>t h<sub>ave a un</sub>i<sub>que equ</sub>ilib<sub>r</sub>i<sub>um.</sub> Th<sub>eorem</sub> 7<sub>.</sub>5 <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> thi<sub>s s</sub>t<sub>ronger conc</sub>l<sub>us</sub>i<sub>on</sub> f<sub>or</sub> th<sub>e parame</sub>t<sub>er-</sub>f<sub>ree</sub> $\mathsf { A l t { - } I R { - } P R M ^ { + } }$ <sub>an</sub>d $\mathsf { A l t { - } H P R M ^ { + } }$ <sub>a</sub>l<sub>gor</sub>ith<sub>ms.</sub>

## 8 Separation of Alternating and Simultaneous Updates

I<sub>n</sub> t<sub>wo-p</sub>l<sub>ayer zero-sum games, a</sub>lt<sub>erna</sub>ti<sub>ng m</sub>i<sub>rror</sub> d<sub>escen</sub>t <sub>a</sub>d<sub>m</sub>it<sub>s</sub> ${ \cal O } ( T ^ { - 4 / 5 } )$ <sup>a</sup>v<sup>era</sup>g<sup>e-iterate</sup> <sup>con</sup>v<sup>er</sup>g<sup>ence</sup> gu<sup>arantees</sup> u<sup>n-</sup> d<sub>er cer</sub>t<sub>a</sub>i<sub>n regu</sub>l<sub>ar</sub>it<sub>y assump</sub>ti<sub>ons,</sub> i<sub>mprov</sub>i<sub>ng on</sub> th<sub>e c</sub>l<sub>ass</sub>i<sub>ca</sub>l ${ \cal O } ( T ^ { - 1 / 2 } )$ <sub>g</sub>uarantee for simultaneous mirror descent [Katona et al., 2026]. Similarly, projected alternating gradient descent–ascent achieves an ${ \cal O } ( T ^ { - 1 } )$ <sub>average-</sub>it<sub>era</sub>t<sub>e ra</sub>t<sub>e</sub> i<sub>n</sub> fi<sub>n</sub>it<sub>e ma</sub>t<sub>r</sub>i<sub>x games, compare</sub>d <sub>w</sub>ith th<sub>e c</sub>l<sub>ass</sub>i<sub>ca</sub>l $\bar { O } ( T ^ { - 1 / 2 } )$ <sub>g</sub>uarantee for simultaneous u<sub>p</sub>dates [Nan et al., 2026b]. For re<sub>g</sub>ret-matchin<sub>g</sub> methods, Cai et al. [2025, Section 3] <sub>p</sub>rovide numerical evidence that alternation can afect con-<sub>vergence</sub> it<sub>se</sub>lf<sub>.</sub> O<sub>n a</sub> $3 \times 3$ matrix <sub>g</sub>ame, their ex<sub>p</sub>eriments su<sub>gg</sub>est last-iterate nonconver<sub>g</sub>ence of Sim- $\cdot \mathrm { P R M ^ { + } }$ <sub>,</sub> <sub>w</sub>h<sub>ereas</sub> $\mathtt { A l t - P R M ^ { + } }$ <sub>appears</sub> t<sub>o</sub> <sub>converge</sub> t<sub>o</sub> <sub>equ</sub>ilib<sub>r</sub>i<sub>um.</sub> Th<sub>ese</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>ons</sub> <sub>mo</sub>ti<sub>va</sub>t<sub>e</sub> <sub>a</sub> <sub>s</sub>t<sub>ronger</sub> <sub>separa</sub>ti<sub>on</sub> th<sub>an</sub> <sub>an</sub> i<sub>mprovemen</sub>t i<sub>n</sub> <sub>average-</sub>it<sub>era</sub>t<sub>e</sub> <sub>ra</sub>t<sub>es,</sub> b<sub>u</sub>t l<sub>eave</sub> <sub>open</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> b<sub>o</sub>th b<sub>e</sub>h<sub>av</sub>i<sub>ors</sub> <sub>can</sub> b<sub>e</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d <sub>ana</sub>l<sub>y</sub>ti<sub>ca</sub>ll<sub>y.</sub>

W<sub>e es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>suc</sub>h <sub>an ana</sub>l<sub>y</sub>ti<sub>ca</sub>l <sub>separa</sub>ti<sub>on</sub> f<sub>or</sub> $\mathrm { \tt H P R M ^ { + } }$ b<sub>y com</sub>bi<sub>n</sub>i<sub>ng</sub> it<sub>s a</sub>lt<sub>erna</sub>ti<sub>ng convergence guaran</sub>t<sub>ee w</sub>ith th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng s</sub>i<sub>mu</sub>lt<sub>aneous nonconvergence resu</sub>lt<sub>.</sub>

Theorem 8.1. Consider the normal-form game

$$
\mathbf { A } = \left( \begin{array} { c c c c } { 0 } & { - 1 } & { 1 } & { 4 } \\ { 1 } & { 0 } & { - 1 } & { 3 } \\ { - 1 } & { 1 } & { 0 } & { 3 } \\ { - 4 } & { - 3 } & { - 3 } & { 0 } \end{array} \right) .\tag{30}
$$

Under standard initialization, Sim-HPRM<sup>+</sup> satisfies

$$
\operatorname* { l i m } _ { t \to \infty } \operatorname* { i n f } _ { \mathbf { G } \mathbf { a p } ( \pmb { x } ^ { ( t ) } , \pmb { y } ^ { ( t ) } ) \geq \frac { 1 } { \sqrt { 3 } } } ,
$$

while $A \bar { \varGamma } t - \bar { H P R M } ^ { + }$ converges to the unique Nash equilibrium $\pmb { x } ^ { * } = \mathbf { y } ^ { * } = ( 1 / 3 , 1 / 3 , 1 / 3 , 0 )$ (Theorem 7.5).

We constructed the exam<sub>p</sub>le above b<sub>y</sub> startin<sub>g</sub> from three-action rock–<sub>p</sub>a<sub>p</sub>er–scissors (RPS), whose skew s<sub>y</sub>mmetr<sub>y</sub> k<sub>eeps</sub> th<sub>e</sub> t<sub>wo s</sub>i<sub>mu</sub>lt<sub>aneous</sub> l<sub>earners sync</sub>h<sub>ron</sub>i<sub>ze</sub>d<sub>, w</sub>ith $\mathbf { \boldsymbol { x } } ^ { ( t ) } = \mathbf { \boldsymbol { y } } ^ { ( t ) }$ <sub>an</sub>d <sub>zero rea</sub>li<sub>ze</sub>d <sub>u</sub>tilit<sub>y.</sub> H<sub>owever, s</sub>t<sub>an</sub>d<sub>ar</sub>d i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on wou</sub>ld <sub>p</sub>l<sub>ace</sub> th<sub>em exac</sub>tl<sub>y a</sub>t th<sub>e un</sub>if<sub>orm equ</sub>ilib<sub>r</sub>i<sub>um,</sub> l<sub>eav</sub>i<sub>ng no non</sub>t<sub>r</sub>i<sub>v</sub>i<sub>a</sub>l d<sub>ynam</sub>i<sub>cs</sub> t<sub>o ana</sub>l<sub>yze.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore a</sub>dd<sub>e</sub>d <sub>a s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> d<sub>om</sub>i<sub>na</sub>t<sub>e</sub>d f<sub>our</sub>th <sub>ac</sub>ti<sub>on an</sub>d <sub>ass</sub>i<sub>gne</sub>d <sub>unequa</sub>l <sub>payo</sub>f<sub>s</sub> 4<sub>,</sub> 3<sub>,</sub> 3 <sub>aga</sub>i<sub>ns</sub>t it<sub>, c</sub>h<sub>oos</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>our</sub>th <sub>row</sub> t<sub>o preserve s</sub>k<sub>ew symme</sub>t<sub>ry.</sub> U<sub>n</sub>d<sub>er un</sub>if<sub>orm</sub> i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on,</sub> th<sub>e</sub> RPS <sub>payo</sub>f<sub>s cance</sub>l<sub>, so</sub> th<sub>e unequa</sub>l <sub>payo</sub>f<sub>s</sub> 4<sub>,</sub> 3<sub>,</sub> 3 <sub>aga</sub>i<sub>ns</sub>t th<sub>e</sub> f<sub>our</sub>th <sub>ac</sub>ti<sub>on pro</sub>d<sub>uce</sub> $\pmb { x } ^ { ( 2 ) } = \pmb { y } ^ { ( 2 ) } = ( 2 / 5 , 3 / 1 0 , 3 / 1 0 , 0 )$ <sub>a</sub>ft<sub>er</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>s</sub>i<sub>mu</sub>lt<sub>aneous</sub> <sub>up</sub>d<sub>a</sub>t<sub>e.</sub> Th<sub>e</sub> f<sub>our</sub>th action receives negative utility throughout the subsequent simultaneous trajectory and is never played again. Its only <sub>ro</sub>l<sub>e</sub> i<sub>s</sub> h<sub>ence</sub> t<sub>o genera</sub>t<sub>e a nonun</sub>if<sub>orm s</sub>t<sub>ar</sub>ti<sub>ng po</sub>i<sub>n</sub>t f<sub>or</sub> th<sub>e</sub> RPS d<sub>ynam</sub>i<sub>cs</sub> th<sub>roug</sub>h th<sub>e prescr</sub>ib<sub>e</sub>d <sub>up</sub>d<sub>a</sub>t<sub>es</sub> th<sub>emse</sub>l<sub>ves.</sub> Si<sub>nce</sub> b<sub>o</sub>th <sub>p</sub>l<sub>ayers</sub> h<sub>ave</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>regre</sub>t <sub>vec</sub>t<sub>ors</sub> <sub>an</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es,</sub> <sub>we</sub> <sub>ana</sub>l<sub>yze</sub> <sub>one</sub> l<sub>earner</sub>’<sub>s</sub> <sub>su</sub>b<sub>sequen</sub>t <sub>up</sub>d<sub>a</sub>t<sub>es</sub> <sub>w</sub>ithi<sub>n</sub> th<sub>e</sub> th<sub>ree-ac</sub>ti<sub>on</sub> RPS <sub>su</sub>b<sub>game an</sub>d <sub>prove</sub> th<sub>a</sub>t it<sub>s s</sub>t<sub>ra</sub>t<sub>egy rema</sub>i<sub>ns</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>away</sub> f<sub>rom equ</sub>ilib<sub>r</sub>i<sub>um.</sub> A <sub>v</sub>i<sub>sua</sub>li<sub>za</sub>ti<sub>on</sub> f<sub>or</sub> thi<sub>s</sub> <sub>separa</sub>ti<sub>on</sub> i<sub>s s</sub>h<sub>own</sub> i<sub>n</sub> Fi<sub>gure</sub> 9 i<sub>n</sub> th<sub>e</sub> A<sub>ppen</sub>di<sub>x.</sub>

## 9 Experiments

I<sub>n</sub> thi<sub>s sec</sub>ti<sub>on, we</sub> fi<sub>rs</sub>t <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>ll<sub>y exam</sub>i<sub>ne</sub> h<sub>ow pre</sub>di<sub>c</sub>ti<sub>on sca</sub>li<sub>ng a</sub>f<sub>ec</sub>t<sub>s average- an</sub>d l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or</sub> i<sub>n</sub> normal-form <sub>g</sub>ames, then evaluate whether half <sub>p</sub>rediction im<sub>p</sub>roves the <sub>p</sub>ractical <sub>p</sub>erformance of CFR on standard <sub>ex</sub>t<sub>ens</sub>i<sub>ve-</sub>f<sub>orm</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>ames.</sub>

## 9.1 Numerical Separations under Prediction Scaling

Th<sub>e prece</sub>di<sub>ng resu</sub>lt<sub>s</sub> i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 6 id<sub>en</sub>tif<sub>y</sub> $\lambda = 1 / 2$ <sub>as</sub> th<sub>e</sub> <sub>un</sub>i<sub>que</sub> fi<sub>xe</sub>d <sub>nonnega</sub>ti<sub>ve</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>mu</sub>lti<sub>p</sub>li<sub>er</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> Alt<sub>-</sub>RVU <sub>on ever</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>u</sub>tilit <sub>se uence</sub> i<sub>n ever</sub> fi<sub>n</sub>it<sub>e</sub> di<sub>mens</sub>i<sub>on, an</sub>d th<sub>ere</sub>f<sub>ore ru</sub>l<sub>e ou</sub>t b<sub>o</sub>th Alt<sub>-</sub>RVU <sub>an</sub>d it<sub>s</sub> <sub>re</sub>fi<sub>nemen</sub>t f<sub>or a</sub>ll <sub>o</sub>th<sub>er</sub> fi<sub>xe</sub>d <sub>nonnega</sub>ti<sub>ve pre</sub>di<sub>c</sub>ti<sub>on mu</sub>lti<sub>p</sub>li<sub>ers.</sub> W<sub>e now exam</sub>i<sub>ne w</sub>h<sub>e</sub>th<sub>er pre</sub>di<sub>c</sub>ti<sub>on sca</sub>li<sub>ng pro</sub>d<sub>uces</sub> <sub>separa</sub>ti<sub>ons</sub> i<sub>n ac</sub>t<sub>ua</sub>l <sub>a</sub>lt<sub>erna</sub>ti<sub>ng se</sub>lf<sub>-p</sub>l<sub>ay.</sub>

A<sub>cross</sub> th<sub>e</sub> th<sub>ree games</sub> i<sub>n</sub> Fi<sub>gure</sub> 4<sub>,</sub> th<sub>e c</sub>h<sub>o</sub>i<sub>ces</sub> $\lambda = 1 0 ^ { - 4 } , \lambda = 1 0 ^ { - 2 }$ <sub>, an</sub>d $\lambda = 1 / 3$ <sub>, respec</sub>ti<sub>ve</sub>l<sub>y, ex</sub>hibit <sub>un</sub>if<sub>orm</sub> <sup>a</sup>v<sup>era</sup>g<sup>e-iterate</sup> g<sup>a</sup>p <sup>deca</sup>y <sup>a</sup>pp<sup>roximatel</sup>y p<sup>ro</sup>p<sup>ortional to</sup> $T ^ { - 1 / 2 }$ <sub>a</sub>t l<sub>a</sub>t<sub>e</sub> it<sub>era</sub>ti<sub>ons, w</sub>h<sub>ereas</sub> h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on ex</sub>hibit<sub>s</sub> $T ^ { - 1 }$ d<sub>ecay on</sub> th<sub>e same games.</sub> Th<sub>e</sub> l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e gaps revea</sub>l <sub>a</sub> f<sub>ur</sub>th<sub>er separa</sub>ti<sub>on:</sub> f<sub>or</sub> th<sub>e</sub> t<sub>es</sub>t<sub>e</sub>d <sub>non-</sub>h<sub>a</sub>lf <sub>mu</sub>lti<sub>p</sub>li<sub>ers,</sub> th<sub>ey</sub> <sub>rema</sub>i<sub>n</sub> l<sub>arge an</sub>d <sub>osc</sub>ill<sub>a</sub>t<sub>ory</sub> th<sub>roug</sub>h $1 0 ^ { 9 }$ it<sub>era</sub>ti<sub>ons, w</sub>hil<sub>e</sub> h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on re</sub>d<sub>uces</sub> th<sub>em</sub> t<sub>o</sub> th<sub>e numer</sub>i<sub>ca</sub>l <sub>prec</sub>i<sub>s</sub>i<sub>on</sub> li<sub>m</sub>it<sub>.</sub>

![](images/f98e4c2a51e2f15861b389439dd44e0aea094a9261f7cfe75efee6f0638b8f18.jpg)  
Fi<sub>gure</sub> 4<sub>:</sub> A<sub>verage-</sub>it<sub>era</sub>t<sub>e</sub> <sub>an</sub>d l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e</sub> <sub>gaps</sub> <sub>un</sub>d<sub>er</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>se</sub>lf<sub>-p</sub>l<sub>ay.</sub> H<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> i<sub>s</sub> <sub>compare</sub>d <sub>w</sub>ith $\lambda = 1 0 ^ { - 4 }$ <sub>on</sub> th<sub>e</sub> $3 \times 3$ <sub>g</sub>ame of Cai et al. [2025] (with the <sub>p</sub>a<sub>y</sub>of matrix ne<sub>g</sub>ated to match our maximizin<sub>g</sub>-row convention), $\lambda = 1 0 ^ { - 2 }$ on a 11, 002 × 11, 002 block <sub>g</sub>ame, and $\lambda = 1 / 3$ <sub>o</sub>n <sub>a</sub> 177<sub>,</sub> $7 0 2 \times 1 7 7$ <sub>,</sub> 702 bl<sub>oc</sub>k <sub>game.</sub> Th<sub>e gray</sub> li<sub>nes</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>e</sub> th<sub>e re</sub>f<sub>erence s</sub>l<sub>opes</sub> $T ^ { - 1 / 2 }$ <sub>an</sub>d $T ^ { - 1 }$

Th<sub>e compar</sub>i<sub>son w</sub>ith $\lambda = 1 / 3$ i<sub>s par</sub>ti<sub>cu</sub>l<sub>ar</sub>l<sub>y no</sub>t<sub>ewor</sub>th<sub>y</sub> b<sub>ecause</sub> thi<sub>s</sub> i<sub>s</sub> th<sub>e pre</sub>di<sub>c</sub>ti<sub>on mu</sub>lti<sub>p</sub>li<sub>er use</sub>d b<sub>y</sub> th<sub>e</sub> l<sub>oca</sub>l l<sub>earner</sub> <sub>un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> ${ \mathsf { S A P C F R } } ^ { + }$ [Men<sub>g</sub> et al., 2026]. Their anal<sub>y</sub>sis does not establish conver<sub>g</sub>ence under alternatin<sub>g</sub> self-<sub>p</sub>l<sub>ay, an</sub>d <sub>our numer</sub>i<sub>ca</sub>l <sub>resu</sub>lt<sub>s sugges</sub>t <sub>a su</sub>b<sub>op</sub>ti<sub>ma</sub>l $T ^ { - 1 / 2 }$ <sub>avera e-</sub>it<sub>era</sub>t<sub>e ra</sub>t<sub>e an</sub>d <sub>oss</sub>ibl<sub>e</sub> l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e nonconver ence</sub> f<sub>or</sub> thi<sub>s pre</sub>di<sub>c</sub>ti<sub>on c</sub>h<sub>o</sub>i<sub>ce on</sub> th<sub>e</sub> thi<sub>r</sub>d <sub>game,</sub> i<sub>n con</sub>t<sub>ras</sub>t t<sub>o our convergence guaran</sub>t<sub>ees</sub> f<sub>or</sub> h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on.</sub>

Th<sub>a</sub>t b<sub>e</sub>i<sub>ng sa</sub>id<sub>,</sub> th<sub>ese exper</sub>i<sub>men</sub>t<sub>s</sub> d<sub>o no</sub>t <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>asymp</sub>t<sub>o</sub>ti<sub>c</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s or</sub> l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e nonconvergence.</sub> O<sub>ur</sub> <sub>numer</sub>i<sub>ca</sub>l <sub>compar</sub>i<sub>son</sub> h<sub>ere para</sub>ll<sub>e</sub>l<sub>s</sub> th<sub>e</sub> $3 \times 3$ exam<sub>p</sub>le of Cai et al. [2025, Section 3], who re<sub>p</sub>ort numerical evidence <sub>o</sub>f l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e nonconver ence</sub> f<sub>or</sub> ${ \mathsf { S i m } } { - } { \mathsf { P R M } } ^ { + }$ <sub>an</sub>d <sub>convergence</sub> f<sub>or</sub> $\mathtt { A l t - P R M ^ { + } }$ <sub>on</sub> th<sub>e</sub> <sub>same</sub> <sub>game.</sub> Th<sub>e</sub> th<sub>ree</sub> <sub>games</sub> i<sub>n</sub> Fi<sub>gure</sub> 4 <sub>concern on</sub>l<sub>y</sub> th<sub>e</sub> t<sub>es</sub>t<sub>e</sub>d <sub>mu</sub>lti<sub>p</sub>li<sub>ers</sub> b<sub>e</sub>l<sub>ow</sub> $1 / 2 ;$ <sub>w</sub>h<sub>e</sub>th<sub>er ana</sub>l<sub>ogous separa</sub>ti<sub>ons occur</sub> f<sub>or</sub> $\lambda > 1 / 2$ <sub>un</sub>d<sub>er</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng se</sub>lf<sub>-p</sub>l<sub>ay rema</sub>i<sub>ns unreso</sub>l<sub>ve</sub>d<sub>,</sub> d<sub>esp</sub>it<sub>e</sub> th<sub>e</sub> f<sub>a</sub>il<sub>ure o</sub>f Alt<sub>-</sub>RVU i<sub>n</sub> th<sub>a</sub>t <sub>range.</sub> Th<sub>e</sub> t<sub>wo</sub> bl<sub>oc</sub>k <sub>games s</sub>h<sub>are a</sub> <sub>s</sub>i<sub>mp</sub>l<sub>e</sub> $3 \times 3$ i<sub>n</sub>t<sub>eger</sub> k<sub>erne</sub>l <sub>an</sub>d dif<sub>er on</sub>l<sub>y</sub> i<sub>n</sub> th<sub>e</sub>i<sub>r ac</sub>ti<sub>on mu</sub>lti<sub>p</sub>li<sub>c</sub>iti<sub>es, w</sub>hi<sub>c</sub>h <sub>we se</sub>l<sub>ec</sub>t<sub>e</sub>d th<sub>roug</sub>h h<sub>eur</sub>i<sub>s</sub>ti<sub>c numer</sub>i<sub>ca</sub>l <sub>searc</sub>h<sub>.</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>1 <sub>g</sub>i<sub>ves</sub> f<sub>ur</sub>th<sub>er</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s on</sub> th<sub>e cons</sub>t<sub>ruc</sub>ti<sub>on an</sub>d <sub>numer</sub>i<sub>ca</sub>l <sub>va</sub>lid<sub>a</sub>ti<sub>on.</sub>

## 9.2 Performance on Extensive-Form Benchmarks

W<sub>e con</sub>d<sub>uc</sub>t<sub>e</sub>d <sub>exper</sub>i<sub>men</sub>t<sub>s on s</sub>i<sub>zea</sub>bl<sub>e, s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>ex</sub>t<sub>ens</sub>i<sub>ve-</sub>f<sub>orm</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>games.</sub> O<sub>ur a</sub>l<sub>gor</sub>ith<sub>ms</sub> f<sub>or norma</sub>l<sub>-</sub>f<sub>orm</sub> <sub>g</sub>ames are incor<sub>p</sub>orated into extensive form usin<sub>g</sub> the standard counterfactual re<sub>g</sub>ret minimization (CFR) framework [Zinkevich et al., 2007, Farina et al., 2019d], so in the names of the al<sub>g</sub>orithms, we re<sub>p</sub>lace RM with CFR as i<sub>s s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>prac</sub>ti<sub>ce.</sub> W<sub>e compare</sub> th<sub>e a</sub>l<sub>gor</sub>ith<sub>ms</sub> $\mathsf { A l t { - } P C F R ^ { + } }$ $\mathsf { A l t { - } H P C F R { ^ + } }$ $\mathsf { A l t { - } S A P C F R { ^ + } }$ [Men et al., 2026], $\mathsf { A l t { - } I R { - } P C F R { + } }$ , Alt-PDCFR<sup>+</sup> [Xu et al., 2024], and $\mathsf { A l t - D C F R ^ { + } }$ [Brown and Sandholm, 2019, Xu et al., 2024]. Here, $\mathsf { A l t { - } S A P C F R { ^ + } }$ <sub>uses</sub> l<sub>oca</sub>l l<sub>earners</sub> f<sub>rom</sub> th<sub>e same sca</sub>l<sub>e</sub>d<sub>-pre</sub>di<sub>c</sub>ti<sub>on</sub> f<sub>am</sub>il<sub>y as</sub> $\mathsf { A l t { - } H P C F R { ^ + } }$ <sub>,</sub> b<sub>u</sub>t <sub>w</sub>ith $\lambda = 1 / 3$ Th<sub>e</sub> b<sub>ase</sub>li<sub>nes</sub> $\mathtt { A l t - P D C F R } ^ { + }$ <sub>an</sub>d $\mathsf { A l t - D C F R ^ { + } }$ are based on the famil<sub>y</sub> of discounted re<sub>g</sub>ret matchin<sub>g</sub> methods (with the constant � set as su<sub>gg</sub>ested in those <sub>p</sub>a<sub>p</sub>ers). Table 2 in the a<sub>pp</sub>endix lists the 15 benchmark <sub>g</sub>ames and their iteration b<sub>u</sub>d<sub>ge</sub>t<sub>s.</sub> All <sub>me</sub>th<sub>o</sub>d<sub>s use a</sub>lt<sub>erna</sub>ti<sub>ng up</sub>d<sub>a</sub>t<sub>es, w</sub>ith <sub>one</sub> it<sub>era</sub>ti<sub>on cons</sub>i<sub>s</sub>ti<sub>ng o</sub>f <sub>one up</sub>d<sub>a</sub>t<sub>e</sub> b<sub>y eac</sub>h <sub>p</sub>l<sub>ayer.</sub> W<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e resu</sub>lti<sub>ng s</sub>t<sub>ra</sub>t<sub>egy pro</sub>fil<sub>e a</sub>ft<sub>er</sub> b<sub>o</sub>th <sub>up</sub>d<sub>a</sub>t<sub>es.</sub> F<sub>ur</sub>th<sub>er</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s a</sub>b<sub>ou</sub>t th<sub>e exper</sub>i<sub>men</sub>t<sub>s are</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub>3<sub>.</sub>

I<sub>n</sub> thi<sub>s</sub> <sub>sec</sub>ti<sub>on,</sub> $\mathbf { \boldsymbol { x } } ^ { ( t ) }$ <sub>an</sub>d $\mathbf { y } ^ { ( t ) }$ d<sub>eno</sub>t<sub>e</sub> <sub>rea</sub>li<sub>za</sub>ti<sub>on</sub> <sub>p</sub>l<sub>ans.</sub> W<sub>e</sub> <sub>repor</sub>t th<sub>e</sub> l<sub>as</sub>t it<sub>era</sub>t<sub>e,</sub> th<sub>e</sub> <sub>un</sub>if<sub>orm</sub> <sub>average,</sub> <sub>an</sub>d th<sub>e</sub> HS30 wei<sub>g</sub>hted avera<sub>g</sub>e [Zhan<sub>g</sub> et al., 2026]. For a fixed bud<sub>g</sub>et � on the number of iterations, the latter is

$$
\bar { x } _ { { \mathrm { H S 3 } } 0 } ^ { ( t ) } : = \frac { \sum _ { s = 1 } ^ { t } w _ { s } ^ { ( t ) } { \pmb x } ^ { ( s ) } } { \sum _ { s = 1 } ^ { t } w _ { s } ^ { ( t ) } } , \ \bar { y } _ { { \mathrm { H S 3 } } 0 } ^ { ( t ) } : = \frac { \sum _ { s = 1 } ^ { t } w _ { s } ^ { ( t ) } { \pmb y } ^ { ( s ) } } { \sum _ { s = 1 } ^ { t } w _ { s } ^ { ( t ) } } , \ \mathrm { w h e r e } \ w _ { s } ^ { ( t ) } : = \prod _ { k = s + 1 } ^ { t } \left( \frac { k - 1 } { k } \right) ^ { 3 0 - 5 k / T } .\tag{31}
$$

All <sub>o</sub>f th<sub>e</sub> <sub>me</sub>th<sub>o</sub>d<sub>s</sub> <sub>use</sub> th<sub>e</sub> <sub>same</sub> <sub>averag</sub>i<sub>ng</sub> <sub>ru</sub>l<sub>es</sub> f<sub>or</sub> <sub>compu</sub>ti<sub>ng</sub> <sub>average</sub> <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es.</sub> Th<sub>ose</sub> <sub>ru</sub>l<sub>es</sub> d<sub>o</sub> <sub>no</sub>t <sub>a</sub>f<sub>ec</sub>t th<sub>e</sub>i<sub>r</sub> re<sub>g</sub>ret-u<sub>p</sub><sup>d</sup>at<sup>i</sup>n<sub>g</sub> ru<sup>l</sup>es.

W<sub>e</sub> <sub>measure</sub> <sub>exp</sub>l<sub>o</sub>it<sub>a</sub>bilit<sub>y</sub> <sub>as</sub> h<sub>a</sub>lf th<sub>e</sub> d<sub>ua</sub>lit<sub>y</sub> <sub>gap,</sub> <sub>w</sub>ith b<sub>es</sub>t <sub>responses</sub> t<sub>a</sub>k<sub>en</sub> <sub>over</sub> th<sub>e</sub> <sub>ex</sub>t<sub>ens</sub>i<sub>ve-</sub>f<sub>orm</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>spaces:</sub>

$$
E _ { \mathrm { l a s t } } ( t ) : = \frac { 1 } { 2 } \operatorname { G a p } ( \boldsymbol { x } ^ { ( t ) } , \boldsymbol { y } ^ { ( t ) } ) , E _ { \mathrm { a v g } } ( t ) : = \frac { 1 } { 2 } \operatorname { G a p } ( \bar { \boldsymbol { x } } ^ { ( t ) } , \bar { \boldsymbol { y } } ^ { ( t ) } ) , E _ { \mathrm { H S 3 0 } } ( t ) : = \frac { 1 } { 2 } \operatorname { G a p } ( \bar { x } _ { \mathrm { H S 3 0 } } ^ { ( t ) } , \bar { y } _ { \mathrm { H S 3 0 } } ^ { ( t ) } ) .
$$

A<sub>cross</sub> <sub>mos</sub>t b<sub>enc</sub>h<sub>mar</sub>k<sub>s,</sub> th<sub>e</sub> HS30 <sub>average</sub> <sub>c</sub>l<sub>ose</sub>l<sub>y</sub> t<sub>rac</sub>k<sub>s</sub> th<sub>e</sub> l<sub>as</sub>t it<sub>era</sub>t<sub>e</sub> <sub>an</sub>d <sub>usua</sub>ll<sub>y</sub> <sub>ac</sub>hi<sub>eves</sub> l<sub>ower</sub> <sub>exp</sub>l<sub>o</sub>it<sub>a</sub>bilit<sub>y.</sub> B<sub>o</sub>th <sub>genera</sub>ll<sub>y ou</sub>t<sub>per</sub>f<sub>orm</sub> th<sub>e un</sub>if<sub>orm average</sub> b<sub>y a su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>l <sub>marg</sub>i<sub>n.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore presen</sub>t th<sub>e</sub> l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e an</sub>d HS30 <sub>resu</sub>lt<sub>s</sub> <sub>aga</sub>i<sub>ns</sub>t it<sub>era</sub>ti<sub>on</sub> <sub>coun</sub>t i<sub>n</sub> Fi<sub>gures</sub> 6 <sub>an</sub>d 7<sub>,</sub> <sub>respec</sub>ti<sub>ve</sub>l<sub>y,</sub> <sub>an</sub>d d<sub>e</sub>f<sub>er</sub> th<sub>e</sub> <sub>average-</sub>it<sub>era</sub>t<sub>e</sub> <sub>resu</sub>lt<sub>s</sub> t<sub>o</sub> Fi<sub>gure</sub> 10 i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> B<sub>.</sub> Fi<sub>gures</sub> 11<sub>,</sub> 12<sub>, an</sub>d 13 i<sub>n</sub> th<sub>e same appen</sub>di<sub>x repor</sub>t th<sub>e</sub> l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e,</sub> HS30<sub>, average-</sub>it<sub>era</sub>t<sub>e convergence</sub> <sub>aga</sub>i<sub>ns</sub>t l<sub>earner wa</sub>ll<sub>-c</sub>l<sub>oc</sub>k ti<sub>me.</sub>

Alt-HPCFR<sup>+</sup> vs Alt-PCFR<sup>+</sup>. We com<sub>p</sub>are $\mathsf { A l t { - } H P C F R { ^ + } }$ <sub>w</sub>ith $\mathsf { A l t { - } P C F R { ^ + } }$ <sub>, w</sub>hi<sub>c</sub>h h<sub>as ac</sub>hi<sub>eve</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e-o</sub>f<sub>-</sub>th<sub>e-ar</sub>t <sub>em-</sub> pirical performance on the majority of standard non-poker benchmarks [Farina et al., 2021, Chakrabarti et al., 2024, Zhan<sub>g</sub> et al., 2026]. For each <sub>g</sub>ame, we com<sub>p</sub>are al<sub>g</sub>orithms b<sub>y</sub> the first saved iteration at which either the HS30 avera<sub>g</sub>e <sub>or</sub> l<sub>as</sub>t it<sub>era</sub>t<sub>e reac</sub>h<sub>es exp</sub>l<sub>o</sub>it<sub>a</sub>bilit<sub>y a</sub>t <sub>mos</sub>t <sub>�, w</sub>h<sub>ere</sub> $\epsilon \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 6 } , 1 0 ^ { - 9 } \}$ <sub>.</sub> W<sub>e cons</sub>id<sub>er</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>save</sub>d <sub>c</sub>h<sub>ec</sub>k<sub>po</sub>i<sub>n</sub>t th<sub>a</sub>t <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\operatorname* { m i n } \left\{ E _ { \mathrm { H S 3 0 } } ( t ) , E _ { \mathrm { l a s t } } ( t ) \right\} \leq \epsilon .\tag{32}
$$

W<sub>e re or</sub>t th<sub>e ra</sub>ti<sub>o o</sub>f th<sub>e</sub> it<sub>era</sub>ti<sub>on coun</sub>t <sub>an</sub>d <sub>se ara</sub>t<sub>e</sub>l th<sub>e recor</sub>d<sub>e</sub>d <sub>run</sub>ti<sub>me o</sub>f $\mathtt { A l t - P C F R ^ { + } }$ t<sub>o</sub> th<sub>a</sub>t <sub>o</sub>f $\mathsf { A l t { - } H P C F R { ^ + } }$ f<sub>or reac</sub>hi<sub>n eac</sub>h <sub>�</sub> th<sub>res</sub>h<sub>o</sub>ld i<sub>n</sub> Fi <sub>ure</sub> 5<sub>.</sub> A <sub>ra</sub>ti<sub>o rea</sub>t<sub>er</sub> th<sub>an one s</sub>h<sub>ows</sub> th<sub>a</sub>t $\mathsf { A l t { - } H P C F R { ^ + } }$ out<sub>p</sub>er<sup>f</sup>orms $\mathsf { A l t { - } P C F R { ^ + } }$ If <sub>ne</sub>ith<sub>er a</sub>l<sub>gor</sub>ith<sub>m reac</sub>h<sub>es</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>, we repor</sub>t NA f<sub>or no</sub>t <sub>app</sub>li<sub>ca</sub>bl<sub>e.</sub> A<sub>cross mos</sub>t <sub>games an</sub>d th<sub>res</sub>h<sub>o</sub>ld<sub>s,</sub> $\mathsf { A l t { - } H P C F R { ^ + } }$ i<sub>s compe</sub>titi<sub>ve w</sub>ith $\mathtt { A l t - P C F R ^ { + } }$ <sub>an</sub>d <sub>o</sub>ft<sub>en reac</sub>h<sub>es</sub> th<sub>e</sub> t<sub>arge</sub>t <sub>w</sub>ith <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> f<sub>ewer</sub> it<sub>era</sub>ti<sub>ons an</sub>d l<sub>ess</sub> <sub>run</sub>ti<sub>me.</sub> Th<sub>e</sub> <sub>no</sub>ti<sub>cea</sub>bl<sub>e</sub> <sub>excep</sub>ti<sub>ons</sub> <sub>are</sub> B<sub>a</sub>ttl<sub>es</sub>hi<sub>p</sub> $( 3 \times 2 , 3$ shots) and, at $\epsilon = 1 0 ^ { - 6 }$ , Leduc oker (9 ranks). Across th<sub>e</sub> 33 <sub>game-</sub>th<sub>res</sub>h<sub>o</sub>ld <sub>pa</sub>i<sub>rs</sub> f<sub>or w</sub>hi<sub>c</sub>h <sub>a</sub>t l<sub>eas</sub>t <sub>one a</sub>l<sub>gor</sub>ith<sub>m reac</sub>h<sub>es</sub> th<sub>e</sub> t<sub>arge</sub>t<sub>,</sub> $\mathsf { A l t { - } H P C F R { ^ + } }$ <sub>ac</sub>hi<sub>eves w</sub>i<sub>n–</sub>l<sub>oss–</sub>d<sub>raw</sub> <sub>coun</sub>t<sub>s o</sub>f $2 0 : 6 : 7$ i<sub>n</sub> it<sub>era</sub>ti<sub>on coun</sub>t <sub>an</sub>d $2 2 : 6 : 5$ i<sub>n recor</sub>d<sub>e</sub>d <sub>run</sub>ti<sub>me,</sub> t<sub>rea</sub>ti<sub>ng ac</sub>t<sub>ua</sub>l <sub>ra</sub>ti<sub>os</sub> th<sub>a</sub>t <sub>roun</sub>d t<sub>o</sub> 1<sub>.</sub>0 <sub>a</sub>t <sub>one</sub> d<sub>ec</sub>i<sub>ma</sub>l <sub>p</sub>l<sub>ace</sub> <sub>as</sub> d<sub>raws</sub> <sub>an</sub>d <sub>coun</sub>ti<sub>ng</sub> <sub>so</sub>l<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>a</sub>tt<sub>a</sub>i<sub>nmen</sub>t <sub>as</sub> <sub>a</sub> <sub>w</sub>i<sub>n.</sub> M<sub>oreover,</sub> <sub>among</sub> th<sub>e</sub> 7 <sub>games</sub> <sub>w</sub>h<sub>ere</sub> <sub>ne</sub>ith<sub>er</sub> al orithm reaches at least one tar et (i.e., NA), Alt-HPCFR<sup>+</sup> achieves lower ex loitabilit at the end of the run in 6 <sub>g</sub>ames under the same criterion (32). The onl<sub>y</sub> exce<sub>p</sub>tion is Leduc <sub>p</sub>oker (9 ranks).

<table><tr><td rowspan=2 colspan=1>Kuhn pokerGoofspielImp (4 cards)</td><td rowspan=1 colspan=1>1.35</td><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=1>1.12</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.25</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.88</td></tr><tr><td rowspan=1 colspan=1>1.05</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.09</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=1>Liar&#x27;s Dice (4 sides)</td><td rowspan=1 colspan=1>1.09</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.16</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.98</td></tr><tr><td rowspan=2 colspan=1>Leduc pokerGoofspielImp (5 cards)</td><td rowspan=1 colspan=1>1.58</td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.69</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.93</td></tr><tr><td rowspan=1 colspan=1>1.37</td><td rowspan=1 colspan=1>1.36</td><td rowspan=1 colspan=1>1.12</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.38</td><td rowspan=1 colspan=1>1.30</td><td rowspan=1 colspan=1>1.11</td></tr><tr><td rowspan=1 colspan=1>Liar&#x27;s Dice (5 sides)</td><td rowspan=1 colspan=1>2.07</td><td rowspan=1 colspan=1>7.07</td><td rowspan=1 colspan=1>3.80</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2.47</td><td rowspan=1 colspan=1>8.21</td><td rowspan=1 colspan=1>4.03</td></tr><tr><td rowspan=1 colspan=1>Pursuit-evasion (5 turns)</td><td rowspan=1 colspan=1>2.16</td><td rowspan=1 colspan=1>21.86</td><td rowspan=1 colspan=1>&gt;118.73</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.17</td><td rowspan=1 colspan=1>65.91</td><td rowspan=1 colspan=1>&gt;312.50</td></tr><tr><td rowspan=1 colspan=1>Liar&#x27;s Dice (6 sides)</td><td rowspan=1 colspan=1>2.64</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.33</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1>NA</td></tr><tr><td rowspan=1 colspan=1>Leduc poker (9 ranks)</td><td rowspan=1 colspan=1>1.08</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>NA</td></tr><tr><td rowspan=1 colspan=1>Battleship (3×2, 3 shots)</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>0.37</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.36</td></tr><tr><td rowspan=1 colspan=1>Leduc poker (13 ranks)</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>1.52</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.07</td><td rowspan=1 colspan=1>1.63</td><td rowspan=1 colspan=1>NA</td></tr><tr><td rowspan=1 colspan=1>Liar&#x27;s Dice (8 sides)</td><td rowspan=1 colspan=1>1.58</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.77</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1>NA</td></tr><tr><td rowspan=1 colspan=1>HUNL subgame 4</td><td rowspan=1 colspan=1>1.47</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.39</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1>NA</td></tr><tr><td rowspan=2 colspan=1>HUNL subgame 3HUNL subgame 2</td><td rowspan=1 colspan=1>1.18</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.18</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1>NA</td></tr><tr><td rowspan=1 colspan=1>1.29</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.31</td><td rowspan=1 colspan=1>NA</td><td rowspan=1 colspan=1>NA</td></tr></table>

Fi<sub>g</sub>ure 5: Runtime and iteration ratios (Alt-PCFR<sup>+</sup>/Alt-HPCFR<sup>+</sup>) to reach tar<sub>g</sub>et ex<sub>p</sub>loitabilit<sub>y</sub>. Values lar<sub>g</sub>er than 1 mean that Alt-HPCFR<sup>+</sup> out<sub>p</sub>erforms Alt-PCFR<sup>+</sup>.
<table><tr><td></td><td colspan="2">Recorded runtime</td><td colspan="2">Iteration count</td></tr><tr><td>Algorithm</td><td>Top 1</td><td>Top 3</td><td>Top 1</td><td>Top 3</td></tr><tr><td> $\mathsf { A l t { - } H P C F R { ^ + } }$ </td><td>14 (11 sole, 3 ties)</td><td>30</td><td>10 (9 sole, 1 tie)</td><td>28</td></tr><tr><td> $\mathsf { A l t { - } S A P C F R { ^ + } }$ </td><td>10 (7 sole, 3 ties)</td><td>24</td><td>10 (6 sole, 4 ties)</td><td>22</td></tr><tr><td> $\mathsf { A l t { - } P C F R ^ { + } }$ </td><td>5 (4 sole, 1 tie)</td><td>21</td><td>3 (3 sole, 0 ties)</td><td>19</td></tr><tr><td> $\mathtt { A l t - P D C F R ^ { + } }$ </td><td>3 (3 sole, 0 ties)</td><td>11</td><td>6 (6 sole, 0 ties)</td><td>19</td></tr><tr><td> $\mathsf { A l t { - } I R { - } P C F R { ^ + } }$ </td><td>0 (0 sole, 0 ties)</td><td>9</td><td>2 (0 sole, 2 ties)</td><td>17</td></tr><tr><td> $\mathsf { A l t - D C F R ^ { + } }$ </td><td>5 (5 sole, 0 ties)</td><td>8</td><td>6 (5 sole, 1 tie)</td><td>8</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 1<sub>:</sub> T<sub>op-one an</sub>d t<sub>op-</sub>th<sub>ree p</sub>l<sub>acemen</sub>t<sub>s across</sub> th<sub>e</sub> 33 <sub>game-</sub>th<sub>res</sub>h<sub>o</sub>ld <sub>compar</sub>i<sub>sons.</sub>

All 6 algorithms comparisons. We compare all six algorithms using the same criterion above (32), ranking them by <sub>recor</sub>d<sub>e</sub>d <sub>run</sub>ti<sub>me</sub> <sub>an</sub>d<sub>,</sub> <sub>separa</sub>t<sub>e</sub>l<sub>y,</sub> it<sub>era</sub>ti<sub>on</sub> <sub>coun</sub>t t<sub>o</sub> <sub>reac</sub>h th<sub>e</sub> <sub>exp</sub>l<sub>o</sub>it<sub>a</sub>bilit<sub>y</sub> $\epsilon \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 6 } , 1 0 ^ { - 9 } \}$ <sub>.</sub> W<sub>e</sub> r<sub>e</sub>t<sub>a</sub>in th<sub>e</sub> 33 <sub>game-</sub>th<sub>res</sub>h<sub>o</sub>ld <sub>pa</sub>i<sub>rs</sub> f<sub>or</sub> <sub>w</sub>hi<sub>c</sub>h <sub>a</sub>t l<sub>eas</sub>t th<sub>ree</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> <sub>reac</sub>h th<sub>e</sub> t<sub>arge</sub>t<sub>.</sub> T<sub>o</sub> h<sub>an</sub>dl<sub>e</sub> ti<sub>es,</sub> <sub>we</sub> <sub>group</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e</sub> f<sub>as</sub>t<sub>es</sub>t <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>w</sub>h<sub>enever</sub> th<sub>e</sub>i<sub>r</sub> <sub>ac</sub>t<sub>ua</sub>l <sub>cos</sub>t <sub>ra</sub>ti<sub>os</sub> <sub>roun</sub>d t<sub>o</sub> 1<sub>.</sub>0 <sub>a</sub>t <sub>one</sub> d<sub>ec</sub>i<sub>ma</sub>l <sub>p</sub>l<sub>ace,</sub> <sub>an</sub>d <sub>ass</sub>i<sub>gn</sub> <sub>ran</sub>k<sub>s</sub> <sub>suc</sub>h <sub>as</sub> 1<sub>,</sub> 1<sub>,</sub> 3<sub>.</sub> Al<sub>gor</sub>ith<sub>ms</sub> th<sub>a</sub>t d<sub>o no</sub>t <sub>reac</sub>h th<sub>e</sub> t<sub>arge</sub>t <sub>rece</sub>i<sub>ve no p</sub>l<sub>acemen</sub>t<sub>.</sub> T<sub>a</sub>bl<sub>e</sub> 1 <sub>repor</sub>t<sub>s</sub> th<sub>e num</sub>b<sub>er o</sub>f t<sub>op-one</sub> <sub>an</sub>d t<sub>op-</sub>th<sub>ree p</sub>l<sub>acemen</sub>t<sub>s.</sub> W<sub>e</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e a</sub>ll ti<sub>e</sub>d <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> i<sub>n</sub> th<sub>e</sub> t<sub>op-</sub>th<sub>ree coun</sub>t<sub>, even</sub> if <sub>more</sub> th<sub>an</sub> th<sub>ree qua</sub>lif<sub>y, an</sub>d <sub>repor</sub>t <sub>separa</sub>t<sub>e</sub>l<sub>y</sub> h<sub>ow</sub> <sub>o</sub>ft<sub>en</sub> <sub>eac</sub>h <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> fi<sub>n</sub>i<sub>s</sub>h<sub>es</sub> fi<sub>rs</sub>t <sub>a</sub>l<sub>one</sub> <sub>or</sub> ti<sub>es</sub> f<sub>or</sub> fi<sub>rs</sub>t <sub>p</sub>l<sub>ace.</sub>

Re<sub>g</sub>ardin<sub>g</sub> Alt-IR-PCFR<sup>+</sup>, its <sub>p</sub>erformance is weaker in recorded runtime than in iteration count, consistent with the additional <sub>p</sub>er-iteration com<sub>p</sub>utations re<sub>q</sub>uired b<sub>y</sub> the IR mechanism. Meanwhile, the discounted method Alt-DCFR<sup>+</sup> <sub>rema</sub>i<sub>ns</sub> th<sub>e</sub> f<sub>as</sub>t<sub>es</sub>t <sub>me</sub>th<sub>o</sub>d <sub>on</sub> th<sub>e</sub> t<sub>es</sub>t<sub>e</sub>d HUNL <sub>su</sub>b <sub>ames, cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>w</sub>ith <sub>rev</sub>i<sub>ous</sub> fi<sub>n</sub>di<sub>n s</sub> th<sub>a</sub>t <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> DCFR famil of al orithms is articularl well suited for oker ames [Farina et al., 2021, Xu et al., 2024]. However, its l<sub>owes</sub>t <sub>num</sub>b<sub>er o</sub>f t<sub>op-</sub>th<sub>ree p</sub>l<sub>acemen</sub>t<sub>s</sub> i<sub>n</sub> b<sub>o</sub>th <sub>recor</sub>d<sub>e</sub>d <sub>run</sub>ti<sub>me an</sub>d it<sub>era</sub>ti<sub>on coun</sub>t i<sub>n</sub>di<sub>ca</sub>t<sub>es</sub> l<sub>ess cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>an</sub>d <sub>ro</sub>b<sub>us</sub>t <sub>p</sub>erformance across the broader benchmark suite. The <sub>p</sub>redictive variant Alt-PDCFR<sup>+</sup> from the same discounted famil<sub>y</sub> achieves more to<sub>p</sub>-three <sub>p</sub>lacements than Alt-DCFR<sup>+</sup>, but remains less consistentl<sub>y</sub> com<sub>p</sub>etitive in recorded runtime th<sub>an</sub> th<sub>e sca</sub>l<sub>e</sub>d<sub>-pre</sub>di<sub>c</sub>ti<sub>on</sub> f<sub>am</sub>il<sub>y un</sub>d<sub>er our eva</sub>l<sub>ua</sub>ti<sub>on pro</sub>t<sub>oco</sub>l<sub>.</sub>

Withi<sub>n</sub> th<sub>e</sub> <sub>sca</sub>l<sub>e</sub>d<sub>-pre</sub>di<sub>c</sub>ti<sub>on</sub> f<sub>am</sub>il<sub>y,</sub> $\mathsf { A l t { - } P C F R { } ^ { + } }$ <sub>rema</sub>i<sub>ns</sub> <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>tl<sub>y</sub> <sub>compe</sub>titi<sub>ve,</sub> <sub>p</sub>l<sub>ac</sub>i<sub>ng</sub> i<sub>n</sub> th<sub>e</sub> t<sub>op</sub> th<sub>ree</sub> f<sub>re-</sub> <sub>quen</sub>tl<sub>y</sub> i<sub>n</sub> b<sub>o</sub>th <sub>recor</sub>d<sub>e</sub>d <sub>run</sub>ti<sub>me an</sub>d it<sub>era</sub>ti<sub>on coun</sub>t<sub>.</sub> Thi<sub>s agrees w</sub>ith it<sub>s s</sub>t<sub>rong emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>per</sub>f<sub>ormance repor</sub>t<sub>e</sub>d in the literature [Farina et al., 2021, Chakrabarti et al., 2024, Zhan<sub>g</sub> et al., 2026]. Nevertheless, Alt-HPCFR<sup>+</sup> and Alt-SAPCFR<sup>+</sup> achieve more first-<sub>p</sub>lace and to<sub>p</sub>-three finishes in both measures, showin<sub>g</sub> that scalin<sub>g</sub> the <sub>p</sub>rediction <sub>can</sub> f<sub>ur</sub>th<sub>er</sub> i<sub>mprove spee</sub>d <sub>an</sub>d <sub>cons</sub>i<sub>s</sub>t<sub>enc across</sub> th<sub>e</sub> t<sub>es</sub>t<sub>e</sub>d b<sub>enc</sub>h<sub>mar</sub>k<sub>s.</sub> Alth<sub>oug</sub>h $\mathsf { A l t { - } S A P C F R { ^ + } }$ i<sub>s compe</sub>titi<sub>ve w</sub>ith $\mathsf { A l t { - } H P C F R { ^ + } }$ , the anal<sub>y</sub>sis of Men<sub>g</sub> et al. [2026] does not establish conver<sub>g</sub>ence for its local learner under alternatin<sub>g</sub> <sub>se</sub>lf<sub>-p</sub>l<sub>ay, even</sub> i<sub>n norma</sub>l<sub>-</sub>f<sub>orm games.</sub> M<sub>oreover, our numer</sub>i<sub>ca</sub>l <sub>resu</sub>lt<sub>s</sub> i<sub>n</sub> S<sub>u</sub>b<sub>sec</sub>ti<sub>on</sub> 9<sub>.</sub>1 <sub>sugges</sub>t <sub>a su</sub>b<sub>op</sub>ti<sub>ma</sub>l $T ^ { - 1 / \overline { { 2 } } }$ <sub>average-</sub>it<sub>era</sub>t<sub>e ra</sub>t<sub>e an</sub>d <sub>poss</sub>ibl<sub>e</sub> l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e nonconvergence</sub> f<sub>or</sub> it<sub>s pre</sub>di<sub>c</sub>ti<sub>on mu</sub>lti<sub>p</sub>li<sub>er</sub> $\lambda = 1 / 3$ . Overall, Alt-HPCFR<sup>+</sup> <sub>ac</sub>hi<sub>eves</sub> th<sub>e s</sub>t<sub>ronges</sub>t <sub>aggrega</sub>t<sub>e per</sub>f<sub>ormance un</sub>d<sub>er our eva</sub>l<sub>ua</sub>ti<sub>on pro</sub>t<sub>oco</sub>l<sub>.</sub> H<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> i<sub>s a</sub>l<sub>so,</sub> t<sub>o our</sub> k<sub>now</sub>l<sub>e</sub>d<sub>ge,</sub> th<sub>e on</sub>l<sub>y c</sub>h<sub>o</sub>i<sub>ce among</sub> fi<sub>xe</sub>d <sub>pos</sub>iti<sub>ve pre</sub>di<sub>c</sub>ti<sub>on mu</sub>lti<sub>p</sub>li<sub>ers w</sub>ith <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d <sub>convergence guaran</sub>t<sub>ees</sub> f<sub>or a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>se</sub>lf<sub>-p</sub>l<sub>ay</sub> i<sub>n every</sub> fi<sub>n</sub>it<sub>e</sub> t<sub>wo-p</sub>l<sub>ayer zero-sum norma</sub>l<sub>-</sub>f<sub>orm game.</sub>

## 10 Conclusions

W<sub>e</sub> d<sub>eve</sub>l<sub>ope</sub>d Alt<sub>-</sub>RVU <sub>an</sub>d it<sub>s re</sub>fi<sub>nemen</sub>t t<sub>o connec</sub>t <sub>con</sub>diti<sub>ons on</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l l<sub>earners w</sub>ith th<sub>e convergence o</sub>f alternatin<sub>g</sub> self-<sub>p</sub>la<sub>y</sub>. For Alt-IR-PRM<sup>+</sup> and $\mathsf { A l t { - } H P R M ^ { + } }$ <sub>,</sub> thi<sub>s</sub> f<sub>ramewor</sub>k <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> ${ \cal O } ( T ^ { - 1 } )$ <sub>average-</sub>it<sub>era</sub>t<sub>e an</sub>d $\bar { O } ( T ^ { - 1 / 2 } )$ b<sub>es</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e</sub> <sub>ra</sub>t<sub>es</sub> i<sub>n</sub> <sub>every</sub> fi<sub>n</sub>it<sub>e</sub> t<sub>wo-p</sub>l<sub>ayer</sub> <sub>zero-sum</sub> <sub>norma</sub>l<sub>-</sub>f<sub>orm</sub> <sub>game.</sub> F<sub>ur</sub>th<sub>er</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>o</sub>f th<sub>e</sub>i<sub>r</sub> <sub>up</sub>d<sub>a</sub>t<sub>es</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> <sub>convergence</sub> t<sub>o a s</sub>i<sub>ng</sub>l<sub>e</sub> N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>um w</sub>ith<sub>ou</sub>t <sub>assum</sub>i<sub>ng un</sub>i<sub>queness.</sub>

O<sub>ur</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>o</sub>f h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t f<sub>as</sub>t <sub>convergence</sub> <sub>can</sub> b<sub>e</sub> <sub>o</sub>bt<sub>a</sub>i<sub>ne</sub>d b<sub>y</sub> <sub>con</sub>t<sub>ro</sub>lli<sub>ng</sub> th<sub>e</sub> <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>e</sub>f<sub>ec</sub>t <sub>o</sub>f <sub>regre</sub>t<sub>-norm</sub> d<sub>ecreases</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>preven</sub>ti<sub>ng</sub> th<sub>ose</sub> d<sub>ecreases</sub> th<sub>roug</sub>h <sub>a</sub>dditi<sub>ona</sub>l <sub>compu</sub>t<sub>a</sub>ti<sub>on.</sub> O<sub>ur</sub> i<sub>mposs</sub>ibilit<sub>y</sub> <sub>resu</sub>lt<sub>s</sub> f<sub>ur</sub>th<sub>er</sub> id<sub>en</sub>tif<sub>y</sub> h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on as</sub> th<sub>e un</sub>i<sub>que</sub> fi<sub>xe</sub>d <sub>nonnega</sub>ti<sub>ve mu</sub>lti<sub>p</sub>li<sub>er sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> Alt<sub>-</sub>RVU <sub>on every</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>u</sub>tilit<sub>y</sub> <sub>sequence</sub> i<sub>n</sub> <sub>every</sub> fi<sub>n</sub>it<sub>e</sub> di<sub>mens</sub>i<sub>on.</sub>

Our se<sub>p</sub>arations show that both the re<sub>g</sub>ret <sub>q</sub>uantit<sub>y</sub> and the u<sub>p</sub>date schedule matter. The HPRM<sup>+</sup> learner fails the <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d RVU b<sub>oun</sub>d d<sub>esp</sub>it<sub>e</sub> <sub>ac</sub>hi<sub>ev</sub>i<sub>ng</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l ${ \cal O } ( T ^ { - 1 } )$ <sub>average-</sub>it<sub>era</sub>t<sub>e convergence ra</sub>t<sub>e un</sub>d<sub>er a</sub>lt<sub>erna</sub>ti<sub>ng se</sub>lf<sub>-</sub> <sub>p</sub>l<sub>ay,</sub> d<sub>emons</sub>t<sub>ra</sub>ti<sub>ng</sub> th<sub>e</sub> i<sub>mpor</sub>t<sub>ance o</sub>f <sub>our</sub> Alt<sub>-</sub>RVU<sub>.</sub> S<sub>epara</sub>t<sub>e</sub>l<sub>y, on</sub> th<sub>e same ma</sub>t<sub>r</sub>i<sub>x game w</sub>ith th<sub>e same s</sub>t<sub>an</sub>d<sub>ar</sub>d i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on, a</sub>lt<sub>erna</sub>ti<sub>ng</sub> h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on converges</sub> t<sub>o equ</sub>ilib<sub>r</sub>i<sub>um w</sub>hil<sub>e s</sub>i<sub>mu</sub>lt<sub>aneous</sub> h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>ns a</sub> d<sub>ua</sub>lit<sub>y gap asymp</sub>t<sub>o</sub>ti<sub>ca</sub>ll<sub>y</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>away</sub> f<sub>rom zero.</sub> Th<sub>us, up</sub>d<sub>a</sub>t<sub>e or</sub>d<sub>er can</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>pro</sub>fil<sub>es</sub> <sub>approac</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> <sub>a</sub>t <sub>a</sub>ll<sub>.</sub>

Half <sub>p</sub>rediction also achieves stron<sub>g p</sub>ractical <sub>p</sub>erformance. Its CFR instantiation has the stron<sub>g</sub>est a<sub>gg</sub>re<sub>g</sub>ate <sub>per</sub>f<sub>ormance among</sub> th<sub>e s</sub>i<sub>x</sub> t<sub>es</sub>t<sub>e</sub>d <sub>so</sub>l<sub>vers across</sub> 15 <sub>ex</sub>t<sub>ens</sub>i<sub>ve-</sub>f<sub>orm</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s un</sub>d<sub>er our eva</sub>l<sub>ua</sub>ti<sub>on pro</sub>t<sub>oco</sub>l<sub>.</sub> C<sub>om-</sub> <sub>p</sub>l<sub>emen</sub>t<sub>ary numer</sub>i<sub>ca</sub>l <sub>examp</sub>l<sub>es s</sub>h<sub>ow s</sub>l<sub>ower average-</sub>it<sub>era</sub>t<sub>e</sub> d<sub>ecay an</sub>d <sub>pers</sub>i<sub>s</sub>t<sub>en</sub>t l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e osc</sub>ill<sub>a</sub>ti<sub>ons</sub> f<sub>or se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>sma</sub>ll<sub>er pre</sub>di<sub>c</sub>ti<sub>on mu</sub>lti<sub>p</sub>li<sub>ers, sugges</sub>ti<sub>ng</sub> th<sub>a</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>on sca</sub>li<sub>ng can a</sub>f<sub>ec</sub>t <sub>more</sub> th<sub>an ra</sub>t<sub>e cons</sub>t<sub>an</sub>t<sub>s.</sub>

## 11 AI Disclosure

Th<sub>e cons</sub>t<sub>ruc</sub>ti<sub>on</sub> i<sub>n</sub> P<sub>ropos</sub>iti<sub>on</sub> 6<sub>.</sub>1 <sub>was</sub> i<sub>nsp</sub>i<sub>re</sub>d b<sub>y an</sub> id<sub>ea</sub> f<sub>rom a c</sub>h<sub>a</sub>t <sub>w</sub>ith GPT 6 A<sub>s</sub>t<sub>ra</sub> i<sub>n w</sub>hi<sub>c</sub>h th<sub>e</sub> AI <sub>sugges</sub>t<sub>e</sub>d <sub>a cons</sub>t<sub>ruc</sub>ti<sub>on</sub> th<sub>a</sub>t <sub>a</sub>ll<sub>ows</sub> th<sub>e regre</sub>t <sub>vec</sub>t<sub>ors an</sub>d <sub>u</sub>tiliti<sub>es</sub> t<sub>o s</sub>h<sub>r</sub>i<sub>n</sub>k <sub>w</sub>hil<sub>e</sub> th<sub>e s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es osc</sub>ill<sub>a</sub>t<sub>e.</sub> GPT<sub>-</sub>6 A<sub>s</sub>t<sub>ra a</sub>l<sub>so</sub> <sub>ass</sub>i<sub>s</sub>t<sub>e</sub>d i<sub>n</sub> d<sub>eve</sub>l<sub>op</sub>i<sub>ng</sub> th<sub>e numer</sub>i<sub>ca</sub>l <sub>searc</sub>h <sub>proce</sub>d<sub>ure</sub> f<sub>or</sub> th<sub>e</sub> $3 \times 3$ k<sub>erne</sub>l <sub>ma</sub>t<sub>r</sub>i<sub>x use</sub>d i<sub>n</sub> th<sub>e examp</sub>l<sub>e games</sub> i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 9<sub>.</sub>1<sub>,</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> th<sub>e</sub> <sub>au</sub>th<sub>ors</sub>’ h<sub>eur</sub>i<sub>s</sub>ti<sub>c.</sub> Th<sub>e</sub> <sub>same</sub> AI <sub>was</sub> <sub>a</sub>l<sub>so</sub> <sub>use</sub>d t<sub>o</sub> <sub>op</sub>ti<sub>m</sub>i<sub>ze</sub> th<sub>e</sub> <sub>co</sub>d<sub>e</sub> f<sub>or</sub> th<sub>e</sub> <sub>exper</sub>i<sub>men</sub>t<sub>s.</sub>

## 12 Acknowledgments

Th<sub>e wo</sub>rk <sub>p</sub>r<sub>ese</sub>nt<sub>e</sub>d in thi<sub>s pape</sub>r <sub>was suppo</sub>rt<sub>e</sub>d b<sub>y</sub> V<sub>a</sub>nn<sub>eva</sub>r B<sub>us</sub>h F<sub>acu</sub>lt<sub>y</sub> F<sub>e</sub>ll<sub>ows</sub>hi<sub>p</sub> ONR N00014<sub>-</sub>23<sub>-</sub>1<sub>-</sub>2876 <sub>a</sub>nd N<sub>a</sub>ti<sub>ona</sub>l S<sub>c</sub>i<sub>ence</sub> F<sub>oun</sub>d<sub>a</sub>ti<sub>on</sub> <sub>gran</sub>t RI<sub>-</sub>2312342<sub>.</sub> A<sub>ny</sub> <sub>op</sub>i<sub>n</sub>i<sub>ons,</sub> fi<sub>n</sub>di<sub>ngs,</sub> <sub>conc</sub>l<sub>us</sub>i<sub>ons,</sub> <sub>or</sub> <sub>recommen</sub>d<sub>a</sub>ti<sub>ons</sub> <sub>expresse</sub>d in this material are those of the author(s) and do not necessaril<sub>y</sub> reflect the views of the fundin<sub>g</sub> a<sub>g</sub>encies.

![](images/ba5224aff8c8bca0b7cff756f098328e20f6f652dcd67003e9eee2343e65a56e.jpg)  
† Restricted y-range: high early values are outside the panel. All x-axes start at zero.

Fi<sub>gure</sub> 6<sub>:</sub> L<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e convergence aga</sub>i<sub>ns</sub>t it<sub>era</sub>ti<sub>on coun</sub>t<sub>.</sub>

Alt-PCFR+ (λ = 1) Alt-HPCFR+ (λ = 1/2) Alt-SAPCFR+ (λ = 1/3) Alt-IR-PCFR+ Alt-PDCFR+ Alt-DCFR

![](images/39323a31461623d30a181e23f9eccf9fb960bc32aaec65e0e6740b7227f6a077.jpg)  
† Restricted y-range: high early values are outside the panel. All x-axes start at zero.

Fi<sub>gu</sub>r<sub>e</sub> 7<sub>:</sub> HS30 <sub>ave</sub>r<sub>age</sub> <sub>co</sub>n<sub>ve</sub>r<sub>ge</sub>n<sub>ce</sub> <sub>aga</sub>in<sub>s</sub>t it<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>cou</sub>nt<sub>.</sub>

## References

I<sub>oann</sub>i<sub>s</sub> A<sub>nagnos</sub>tid<sub>es,</sub> I<sub>oann</sub>i<sub>s</sub> P<sub>anageas,</sub> G<sub>a</sub>b<sub>r</sub>i<sub>e</sub>l<sub>e</sub> F<sub>ar</sub>i<sub>na,</sub> <sub>an</sub>d T<sub>uomas</sub> S<sub>an</sub>dh<sub>o</sub>l<sub>m.</sub> O<sub>n</sub> l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e</sub> <sub>convergence</sub> b<sub>eyon</sub>d zero-sum games. In Proceedings ofthe International Conference on Machine Learning (ICML), 2022.

J<sub>ames</sub> P<sub>.</sub> B<sub>a</sub>il<sub>ey,</sub> G<sub>au</sub>thi<sub>er</sub> Gid<sub>e</sub>l<sub>,</sub> <sub>an</sub>d G<sub>eorg</sub>i<sub>os</sub> Pili<sub>ouras.</sub> Fi<sub>n</sub>it<sub>e</sub> <sub>regre</sub>t <sub>an</sub>d <sub>cyc</sub>l<sub>es</sub> <sub>w</sub>ith fi<sub>xe</sub>d <sub>s</sub>t<sub>ep-s</sub>i<sub>ze</sub> <sub>v</sub>i<sub>a</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> gradient descent-ascent. In Proceedings ofthe Conference on Learning Theory (COLT), 2020.

Mi<sub>c</sub>h<sub>ae</sub>l B<sub>ow</sub>li<sub>ng,</sub> N<sub>e</sub>il B<sub>urc</sub>h<sub>,</sub> Mi<sub>c</sub>h<sub>ae</sub>l J<sub>o</sub>h<sub>anson, an</sub>d O<sub>s</sub>k<sub>ar</sub>i T<sub>amme</sub>li<sub>n.</sub> H<sub>ea</sub>d<sub>s-up</sub> li<sub>m</sub>it h<sub>o</sub>ld’<sub>em po</sub>k<sub>er</sub> i<sub>s so</sub>l<sub>ve</sub>d<sub>.</sub> Science, 347(6218):145–149, January 2015.

N<sub>oam</sub> B<sub>rown an</sub>d T<sub>uomas</sub> S<sub>an</sub>dh<sub>o</sub>l<sub>m.</sub> S<sub>uper</sub>h<sub>uman</sub> AI f<sub>or</sub> h<sub>ea</sub>d<sub>s-up no-</sub>li<sub>m</sub>it <sub>po</sub>k<sub>er:</sub> Lib<sub>ra</sub>t<sub>us</sub> b<sub>ea</sub>t<sub>s</sub> t<sub>op pro</sub>f<sub>ess</sub>i<sub>ona</sub>l<sub>s.</sub> Science, 359(6374):418–424, 2018. Appeared online in Science in 2017.

N<sub>oam</sub> B<sub>rown an</sub>d T<sub>uomas</sub> S<sub>an</sub>dh<sub>o</sub>l<sub>m.</sub> S<sub>o</sub>l<sub>v</sub>i<sub>ng</sub> i<sub>mper</sub>f<sub>ec</sub>t<sub>-</sub>i<sub>n</sub>f<sub>orma</sub>ti<sub>on games v</sub>i<sub>a</sub> di<sub>scoun</sub>t<sub>e</sub>d <sub>regre</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on.</sub> I<sub>n</sub> Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI), 2019.

Neil Burch, Matej Moravčík, and Martin Schmid. Revisiting CFR+ and alternating updates. Journal of Artificial Intelligence Research, 64:429–443, 2019.

Y<sub>a</sub>n<sub>g</sub> C<sub>a</sub>i<sub>,</sub> G<sub>a</sub>bri<sub>e</sub>l<sub>e</sub> F<sub>a</sub>rin<sub>a,</sub> J<sub>u</sub>li<sub>e</sub>n Gr<sub>a</sub>nd<sub>-</sub>Clém<sub>e</sub>nt<sub>,</sub> Chri<sub>s</sub>ti<sub>a</sub>n Kr<sub>oe</sub>r<sub>,</sub> Ch<sub>u</sub>n<sub>g-</sub>W<sub>e</sub>i L<sub>ee,</sub> H<sub>a</sub>i<sub>pe</sub>n<sub>g</sub> L<sub>uo, a</sub>nd W<sub>e</sub>i<sub>q</sub>i<sub>a</sub>n<sub>g</sub> Zh<sub>e</sub>n<sub>g.</sub> Last-iterate convergence properties of regret-matching algorithms in games. In Proceedings of the International Conference on Learning Representations (ICLR), 2025.

V<sub>o</sub>lk<sub>an</sub> C<sub>ev</sub>h<sub>er,</sub> A<sub>s</sub>h<sub>o</sub>k C<sub>u</sub>tk<sub>os</sub>k<sub>y,</sub> Ali K<sub>av</sub>i<sub>s,</sub> G<sub>eorg</sub>i<sub>os</sub> Pili<sub>ouras,</sub> St<sub>ra</sub>ti<sub>s</sub> Sk<sub>ou</sub>l<sub>a</sub>ki<sub>s, an</sub>d L<sub>uca</sub> Vi<sub>ano.</sub> Alt<sub>erna</sub>ti<sub>on ma</sub>k<sub>es</sub> the adversary weaker in two-player games. In Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 2023.

D<sub>ars</sub>h<sub>an</sub> Ch<sub>a</sub>k<sub>ra</sub>b<sub>ar</sub>ti<sub>,</sub> J<sub>u</sub>li<sub>en</sub> G<sub>ran</sub>d<sub>-</sub>Clé<sub>men</sub>t<sub>, an</sub>d Ch<sub>r</sub>i<sub>s</sub>ti<sub>an</sub> K<sub>roer.</sub> E<sub>x</sub>t<sub>ens</sub>i<sub>ve-</sub>f<sub>orm game so</sub>l<sub>v</sub>i<sub>ng v</sub>i<sub>a</sub> Bl<sub>ac</sub>k<sub>we</sub>ll <sub>ap-</sub> proachability on treeplexes. In Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 2024<sub>.</sub>

C<sub>ons</sub>t<sub>an</sub>ti<sub>nos</sub> D<sub>as</sub>k<sub>a</sub>l<sub>a</sub>ki<sub>s,</sub> Al<sub>an</sub> D<sub>ec</sub>k<sub>e</sub>lb<sub>aum, an</sub>d A<sub>n</sub>th<sub>ony</sub> Ki<sub>m.</sub> N<sub>ear-op</sub>ti<sub>ma</sub>l <sub>no-regre</sub>t <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> f<sub>or zero-sum</sub> games. Games and Economic Behavior, 92:327–348, 2015.

G<sub>a</sub>b<sub>r</sub>i<sub>e</sub>l<sub>e</sub> F<sub>ar</sub>i<sub>na,</sub> Ch<sub>r</sub>i<sub>s</sub>ti<sub>an</sub> K<sub>roer,</sub> N<sub>oam</sub> B<sub>rown,</sub> <sub>an</sub>d T<sub>uomas</sub> S<sub>an</sub>dh<sub>o</sub>l<sub>m.</sub> St<sub>a</sub>bl<sub>e-pre</sub>di<sub>c</sub>ti<sub>ve</sub> <sub>op</sub>ti<sub>m</sub>i<sub>s</sub>ti<sub>c</sub> <sub>coun</sub>t<sub>er</sub>f<sub>ac</sub>t<sub>ua</sub>l regret minimization. In Proceedings of the International Conference on Machine Learning (ICML), 2019a.

G<sub>a</sub>b<sub>r</sub>i<sub>e</sub>l<sub>e</sub> F<sub>ar</sub>i<sub>na,</sub> Ch<sub>r</sub>i<sub>s</sub>ti<sub>an</sub> K<sub>roer, an</sub>d T<sub>uomas</sub> S<sub>an</sub>dh<sub>o</sub>l<sub>m.</sub> O<sub>n</sub>li<sub>ne convex op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> f<sub>or sequen</sub>ti<sub>a</sub>l d<sub>ec</sub>i<sub>s</sub>i<sub>on processes</sub> and extensive-form games. In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI), 2019b.

G<sub>a</sub>b<sub>r</sub>i<sub>e</sub>l<sub>e</sub> F<sub>ar</sub>i<sub>na,</sub> Ch<sub>r</sub>i<sub>s</sub>ti<sub>an</sub> K<sub>roer, an</sub>d T<sub>uomas</sub> S<sub>an</sub>dh<sub>o</sub>l<sub>m.</sub> O<sub>p</sub>ti<sub>m</sub>i<sub>s</sub>ti<sub>c regre</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> f<sub>or ex</sub>t<sub>ens</sub>i<sub>ve-</sub>f<sub>orm games v</sub>i<sub>a</sub> dilated distance-generating functions. In Proceedings ofthe Conference on Neural Information Processing Systems (NeurIPS), 2019c.

G<sub>a</sub>b<sub>r</sub>i<sub>e</sub>l<sub>e</sub> F<sub>ar</sub>i<sub>na,</sub> Ch<sub>un</sub> K<sub>a</sub>i Li<sub>ng,</sub> F<sub>e</sub>i F<sub>ang, an</sub>d T<sub>uomas</sub> S<sub>an</sub>dh<sub>o</sub>l<sub>m.</sub> Efi<sub>c</sub>i<sub>en</sub>t <sub>regre</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on a</sub>l<sub>gor</sub>ith<sub>m</sub> f<sub>or</sub> extensive-form correlated equilibrium. In Proceedings ofthe Conference on Neural Information Processing Systems (NeurIPS), 2019d.

G<sub>a</sub>b<sub>r</sub>i<sub>e</sub>l<sub>e</sub> F<sub>ar</sub>i<sub>na,</sub> Ch<sub>r</sub>i<sub>s</sub>ti<sub>an</sub> K<sub>roer, an</sub>d T<sub>uomas</sub> S<sub>an</sub>dh<sub>o</sub>l<sub>m.</sub> F<sub>as</sub>t<sub>er game so</sub>l<sub>v</sub>i<sub>ng v</sub>i<sub>a pre</sub>di<sub>c</sub>ti<sub>ve</sub> Bl<sub>ac</sub>k<sub>we</sub>ll <sub>approac</sub>h<sub>a</sub>bilit<sub>y:</sub> Connecting regret matching and mirror descent. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2021.

G<sub>a</sub>bri<sub>e</sub>l<sub>e</sub> F<sub>a</sub>rin<sub>a,</sub> J<sub>u</sub>li<sub>e</sub>n Gr<sub>a</sub>nd<sub>-</sub>Clém<sub>e</sub>nt<sub>,</sub> Chri<sub>s</sub>ti<sub>a</sub>n Kr<sub>oe</sub>r<sub>,</sub> Ch<sub>u</sub>n<sub>g-</sub>W<sub>e</sub>i L<sub>ee, a</sub>nd H<sub>a</sub>i<sub>pe</sub>n<sub>g</sub> L<sub>uo.</sub> R<sub>eg</sub>r<sub>e</sub>t m<sub>a</sub>t<sub>c</sub>hin<sub>g</sub>+<sub>:</sub> (in)stability and fast convergence in games. In Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 2023.

Yoav Freund and Robert E. Schapire. Adaptive game playing using multiplicative weights. Games and Economic Behavior, 29:79–103, 1999.

G<sub>au</sub>thi<sub>er</sub> Gid<sub>e</sub>l<sub>,</sub> R<sub>e</sub> h<sub>ane</sub> A<sub>s</sub>k<sub>ar</sub>i H<sub>emma</sub>t<sub>,</sub> M<sub>o</sub>h<sub>amma</sub>d P<sub>ezes</sub>hki<sub>,</sub> Ré<sub>m</sub>i L<sub>e</sub> P<sub>r</sub>i<sub>o</sub>l<sub>,</sub> G<sub>a</sub>b<sub>r</sub>i<sub>e</sub>l H<sub>uan ,</sub> Si<sub>mon</sub> L<sub>acos</sub>t<sub>e-</sub>J<sub>u</sub>li<sub>en,</sub> and Ioannis Mitliagkas. Negative momentum for improved game dynamics. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS), 2019.

S<sub>oum</sub>it<sub>a</sub> H<sub>a</sub>it<sub>,</sub> Pi<sub>ng</sub> Li<sub>,</sub> H<sub>a</sub>i<sub>peng</sub> L<sub>uo, an</sub>d M<sub>engx</sub>i<sub>ao</sub> Zh<sub>ang.</sub> Alt<sub>erna</sub>ti<sub>ng regre</sub>t f<sub>or on</sub>li<sub>ne convex op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on.</sub> I<sub>n</sub> Proceedings ofthe Conference on Learning Theory (COLT), 2025.

Sergiu Hart and Andreu Mas-Colell. A simple adaptive procedure leading to correlated equilibrium. Econometrica, 68(5):1127–1150, 2000.

Jonas E. Katona, Xiuyuan Wang, and Andre Wibisono. A symplectic analysis of alternating mirror descent. Journal ofMachine Learning Research, 27(44):1–61, 2026.

Boning Li and Longbo Huang. Why last-iterate scale-invariant regret matching converges linearly? arXiv preprint arXiv:2609.38162, 2026.

Linjian Meng, Tianpei Yang, Youzhi Zhang, Zhenxing Ge, and Yang Gao. Faster game solving via asymmetry of step sizes. In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI), 2026.

Tianlong Nan, Shuvomo<sub>y</sub> Das Gupta, Garud I<sub>y</sub>engar, and Christian Kroer. On the �(1/�) convergence of alternating gradient descent–ascent in bilinear games. In Proceedings of the International Conference on Learning Representations (ICLR), 2026a.

Tianlong Nan, Garud I<sub>y</sub>engar, Christian Kroer, and Shuvomo<sub>y</sub> Das Gupta. AltGDA achieves global �(1/�) ergodic convergence in matrix games. arXiv preprint arXiv:2609.32995, 2026b.

Alexander Rakhlin and Karthik Sridharan. Online learning with predictable sequences. In Proceedings of the Conference on Learning Theory (COLT), 2013.

Mi<sub>c</sub>h<sub>ae</sub>l S<sub>u</sub>lli<sub>ns an</sub>d I<sub>an</sub> A<sub>.</sub> K<sub>as</sub>h<sub>.</sub> IR<sub>-</sub>PRM+ <sub>w</sub>ith <sub>prev</sub>i<sub>ous-u</sub>tilit<sub>y pre</sub>di<sub>c</sub>ti<sub>on:</sub> C<sub>ons</sub>t<sub>an</sub>t <sub>regre</sub>t <sub>an</sub>d b<sub>es</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e conver-</sub> gence. arXiv preprint arXiv:2610.19016, 2026.

V<sub>as</sub>ili<sub>s</sub> S<sub>yrg</sub>k<sub>an</sub>i<sub>s,</sub> Al<sub>e</sub>kh A<sub>garwa</sub>l<sub>,</sub> H<sub>a</sub>i<sub>peng</sub> L<sub>uo, an</sub>d R<sub>o</sub>b<sub>er</sub>t E S<sub>c</sub>h<sub>ap</sub>i<sub>re.</sub> F<sub>as</sub>t <sub>convergence o</sub>f <sub>regu</sub>l<sub>ar</sub>i<sub>ze</sub>d l<sub>earn</sub>i<sub>ng</sub> i<sub>n</sub> games. In Proceedings ofthe Conference on Neural Information Processing Systems (NeurIPS), 2015.

Oskari Tammelin. Solving large imperfect information games using CFR+. arXiv preprint arXiv:1407.5042, 2014.

Yixin Tao and Weiqiang Zheng. Optimal alternating regret for online learning and games. arXiv preprint arXiv:2608.24731, 2026.

A<sub>n</sub>d<sub>re</sub> Wibi<sub>sono,</sub> M<sub>o</sub>l<sub>e</sub>i T<sub>ao, an</sub>d G<sub>eorg</sub>i<sub>os</sub> Pili<sub>ouras.</sub> Alt<sub>erna</sub>ti<sub>ng m</sub>i<sub>rror</sub> d<sub>escen</sub>t f<sub>or cons</sub>t<sub>ra</sub>i<sub>ne</sub>d <sub>m</sub>i<sub>n-max games.</sub> I<sub>n</sub> Proceedings ofthe Conference on Neural Information Processing Systems (NeurIPS), 2022.

Han<sub>g</sub> Xu, Kai Li, Bin<sub>gy</sub>un Liu, Haobo Fu, Qian<sub>g</sub> Fu, Junlian<sub>g</sub> Xin<sub>g</sub>, and Jian Chen<sub>g</sub>. Minimizin<sub>g</sub> wei<sub>g</sub>hted counterfactual regret with optimistic online mirror descent. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2024.

B<sub>r</sub>i<sub>an</sub> H<sub>u</sub> Zh<sub>ang an</sub>d T<sub>uomas</sub> S<sub>an</sub>dh<sub>o</sub>l<sub>m.</sub> G<sub>enera</sub>l <sub>searc</sub>h t<sub>ec</sub>h<sub>n</sub>i<sub>ques w</sub>ith<sub>ou</sub>t <sub>common</sub> k<sub>now</sub>l<sub>e</sub>d<sub>ge</sub> f<sub>or</sub> i<sub>mper</sub>f<sub>ec</sub>t<sub>-</sub> information games, and application to superhuman Fog of War chess. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2026.

B<sub>r</sub>i<sub>an</sub> H<sub>u</sub> Zh<sub>ang,</sub> I<sub>oann</sub>i<sub>s</sub> A<sub>nagnos</sub>tid<sub>es, an</sub>d T<sub>uomas</sub> S<sub>an</sub>dh<sub>o</sub>l<sub>m.</sub> S<sub>ca</sub>l<sub>e-</sub>i<sub>nvar</sub>i<sub>an</sub>t <sub>regre</sub>t <sub>ma</sub>t<sub>c</sub>hi<sub>ng an</sub>d <sub>on</sub>li<sub>ne</sub> l<sub>earn</sub>i<sub>ng</sub> with optimal convergence: Bridging theory and practice in zero-sum games. arXiv preprint arXiv:2510.04407, 2025<sub>.</sub>

Guojun Zhang and Yaoliang Yu. Convergence of gradient methods on bilinear zero-sum games. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2020.

N<sub>a</sub>if<sub>eng</sub> Zh<sub>ang,</sub> St<sub>ep</sub>h<sub>en</sub> M<sub>arcus</sub> M<sub>c</sub>Al<sub>eer,</sub> <sub>an</sub>d T<sub>uomas</sub> S<sub>an</sub>dh<sub>o</sub>l<sub>m.</sub> F<sub>as</sub>t<sub>er</sub> <sub>game</sub> <sub>so</sub>l<sub>v</sub>i<sub>ng</sub> <sub>v</sub>i<sub>a</sub> h<sub>yperparame</sub>t<sub>er</sub> <sub>sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>es.</sub> In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI), 2026.

M<sub>ar</sub>ti<sub>n</sub> Zi<sub>n</sub>k<sub>ev</sub>i<sub>c</sub>h<sub>,</sub> Mi<sub>c</sub>h<sub>ae</sub>l J<sub>o</sub>h<sub>anson,</sub> Mi<sub>c</sub>h<sub>ae</sub>l B<sub>ow</sub>li<sub>ng, an</sub>d C<sub>arme</sub>l<sub>o</sub> Pi<sub>cc</sub>i<sub>one.</sub> R<sub>egre</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> i<sub>n games w</sub>ith incomplete information. In Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 2007<sub>.</sub>

## A Omitted Proofs

## A.1 Proofs from Subsection 5.1

Proposition 5.3. Suppose that in a normal-form game with alternating update scheme, for a pair of primal-dual norms $( \| \cdot \| , \| \cdot \| _ { * } )$ , both players satisfy the above Alt-RVU bound with $( \alpha _ { \boldsymbol { X } } , \beta _ { \boldsymbol { X } } , \gamma _ { \boldsymbol { X } } ) , ( \alpha _ { \boldsymbol { y } } , \beta _ { \boldsymbol { y } } , \gamma _ { \boldsymbol { y } } ) \ \in \ \mathbb { R } _ { > 0 } ^ { 3 }$ such that $\| \mathbf { A } \| ^ { 2 } \beta _ { X } < \gamma _ { y }$ and $\| \mathbf { A } \| ^ { 2 } \beta y < \gamma _ { X }$ , where $\| \mathbf { A } \| = \operatorname* { s u p } _ { \| \mathbf { x } \| = \| \mathbf { y } \| = 1 } | \mathbf { x } ^ { \top } \mathbf { A } \mathbf { y } |$ is the induced bilinear norm. Then, both players’ squared strategy movement sum and both players’ utility variation sum are finite, and we obtain the average-iterate convergence

$$
\mathrm { G a p } ( \bar { \pmb { x } } ^ { ( T ) } , \bar { \pmb { y } } ^ { ( T ) } ) \leq \frac { \alpha _ { X } + \alpha _ { Y } + \mathrm { r a n g e } ( \mathbf { A } ) } { 2 T } .
$$

Proof. Since each player’s utility vector depends linearly on the opponent’s strategy, the operator norm bound gives

$$
\sum _ { t = 2 } ^ { T } \| \boldsymbol { u } _ { \boldsymbol { X } } ^ { ( t ) } - \boldsymbol { u } _ { \boldsymbol { X } } ^ { ( t - 1 ) } \| _ { * } ^ { 2 } \le \| \mathbf { A } \| ^ { 2 } \sum _ { t = 2 } ^ { T } \| \mathbf { y } ^ { ( t ) } - \mathbf { y } ^ { ( t - 1 ) } \| ^ { 2 } , \ \sum _ { t = 2 } ^ { T } \| \boldsymbol { u } _ { \boldsymbol { y } } ^ { ( t ) } - \boldsymbol { u } _ { \boldsymbol { y } } ^ { ( t - 1 ) } \| _ { * } ^ { 2 } \le \| \mathbf { A } \| ^ { 2 } \sum _ { t = 1 } ^ { T - 1 } \| \mathbf { x } ^ { ( t + 1 ) } - \mathbf { x } ^ { ( t ) } \| ^ { 2 } .\tag{33}
$$

We can sum the Alt-RVU bounds of the two <sub>p</sub>la<sub>y</sub>ers and use the identit<sub>y</sub> (9) to obtain

$$
2 T \mathrm { G a p } ( \bar { x } ^ { ( T ) } , \bar { y } ^ { ( T ) } ) + ( \gamma _ { X } - \| \mathbf { A } \| ^ { 2 } \beta y ) \sum _ { t = 1 } ^ { T } \| x ^ { ( t + 1 ) } - { \mathbf { x } } ^ { ( t ) } \| ^ { 2 } + ( \gamma y - \| \mathbf { A } \| ^ { 2 } \beta _ { X } ) \sum _ { t = 1 } ^ { T } \| y ^ { ( t ) } - { \mathbf { y } } ^ { ( t - 1 ) } \| ^ { 2 }
$$

$$
\leq \alpha _ { X } + \alpha _ { y } + \mathrm { r a n g e } ( \mathbf { A } ) .
$$

E<sub>very</sub> LHS t<sub>erm</sub> i<sub>s nonnega</sub>ti<sub>ve</sub> d<sub>ue</sub> th<sub>e con</sub>diti<sub>on</sub> $\| \mathbf { A } \| ^ { 2 } \beta _ { X } < \gamma _ { y }$ <sub>an</sub>d $\| \mathbf { A } \| ^ { 2 } \beta _ { y } < \gamma _ { X }$ <sub>.</sub> W<sub>e ca</sub>n dr<sub>op</sub> th<sub>e</sub> n<sub>o</sub>nn<sub>ega</sub>ti<sub>ve</sub> t<sub>erms</sub> t<sub>o</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> d<sub>es</sub>i<sub>re</sub>d <sub>average-</sub>it<sub>era</sub>t<sub>e</sub> <sub>convergence</sub>

$$
\mathrm { G a p } ( \bar { \pmb { x } } ^ { ( T ) } , \bar { \pmb { y } } ^ { ( T ) } ) \leq \frac { \alpha _ { X } + \alpha _ { Y } + \mathrm { r a n g e } ( \mathbf { A } ) } { 2 T } .
$$

W<sub>e</sub> <sub>can</sub> <sub>a</sub>l<sub>so</sub> d<sub>rop</sub> th<sub>e</sub> $2 T \mathbf { G a p } ( \bar { \mathbf { x } } ^ { ( T ) } , \bar { \mathbf { y } } ^ { ( T ) } )$ t<sub>erm</sub> t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\sum _ { t = 1 } ^ { T } \| \mathbf { x } ^ { ( t + 1 ) } - \mathbf { x } ^ { ( t ) } \| ^ { 2 } + \sum _ { t = 1 } ^ { T } \| \mathbf { y } ^ { ( t ) } - \mathbf { y } ^ { ( t - 1 ) } \| ^ { 2 } \leq \frac { \alpha _ { X } + \alpha _ { \mathcal { Y } } + \mathrm { r a n g e } ( \mathbf { A } ) } { \operatorname* { m i n } \{ \gamma _ { X } - \| \mathbf { A } \| ^ { 2 } \beta _ { \mathcal { Y } } , \gamma _ { \mathcal { Y } } - \| \mathbf { A } \| ^ { 2 } \beta _ { X } \} } ,
$$

which im<sub>p</sub>lies that the squared strate<sub>gy</sub> movement sum is finite. This also im<sub>p</sub>lies from the bound (33) that the squared <sub>u</sub>tilit<sub>y var</sub>i<sub>a</sub>ti<sub>on sum</sub> i<sub>s</sub> fi<sub>n</sub>it<sub>e.</sub> □

Lemma 5.4. Consider a learner on $\Delta ( d )$ with $\tilde { r } ^ { ( 1 ) } = \mathbf { 0 }$ whose regret vectors satisfy

$$
\tilde { r } ^ { ( t + 1 ) } \geq \tilde { r } ^ { ( t ) } + u ^ { ( t ) } - \langle u ^ { ( t ) } , x ^ { ( t ) } \rangle { \bf 1 }\tag{11}
$$

coordinate-wise. Then, for every $T \geq 1$ , this learner satisfies

$$
\mathrm { A l t - R e g } ( T ) \le 2 \| \tilde { r } ^ { T + 1 } \| _ { 2 } - \sum _ { t = 1 } ^ { T } \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle .\tag{12}
$$

Proof. Inner producting both sides of (11) with �, we obtain

$$
\langle \tilde { { \boldsymbol r } } ^ { ( t + 1 ) } , \boldsymbol z \rangle \ge \langle \tilde { { \boldsymbol r } } ^ { ( t ) } , \boldsymbol z \rangle + \langle { \boldsymbol u } ^ { ( t ) } , \boldsymbol z \rangle - \langle { \boldsymbol u } ^ { ( t ) } , { \boldsymbol x } ^ { ( t ) } \rangle \langle { \bf 1 } , \boldsymbol z \rangle = \langle \tilde { { \boldsymbol r } } ^ { ( t ) } , \boldsymbol z \rangle + \langle { \boldsymbol u } ^ { ( t ) } , \boldsymbol z \rangle - \langle { \boldsymbol u } ^ { ( t ) } , { \boldsymbol x } ^ { ( t ) } \rangle ,
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>mp</sub>li<sub>es</sub>

$$
\begin{array} { r } { \langle \tilde { \pmb { r } } ^ { ( t + 1 ) } - \tilde { \pmb { r } } ^ { ( t ) } , z \rangle \geq \langle \pmb { u } ^ { ( t ) } , z - \pmb { x } ^ { ( t ) } \rangle . } \end{array}
$$

C<sub>onsequen</sub>tl<sub>y,</sub> <sub>eac</sub>h <sub>summan</sub>d i<sub>n</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>regre</sub>t <sub>o</sub>b<sub>eys</sub>

$$
\begin{array} { r } { \langle { \pmb u } ^ { ( t ) } , 2 z - { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t + 1 ) } \rangle \leq 2 \langle \tilde { { \pmb r } } ^ { ( t + 1 ) } - \tilde { { \pmb r } } ^ { ( t ) } , z \rangle - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle . } \end{array}
$$

F<sub>or</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on</sub> $\tilde { r } ^ { ( 1 ) } = 0$ <sub>,</sub> <sub>we</sub> <sub>can</sub> <sub>sum</sub> b<sub>o</sub>th <sub>s</sub>id<sub>es</sub> f<sub>rom</sub> 1 t<sub>o</sub> � t<sub>o</sub> <sub>ge</sub>t

$$
\sum _ { t = 1 } ^ { T } \langle u ^ { ( t ) } , 2 z - x ^ { ( t ) } - x ^ { ( t + 1 ) } \rangle \leq 2 \langle \tilde { r } ^ { ( T + 1 ) } , z \rangle - \sum _ { t = 1 } ^ { T } \langle u ^ { ( t ) } , x ^ { ( t + 1 ) } - x ^ { ( t ) } \rangle \leq 2 \| \tilde { r } ^ { ( T + 1 ) } \| _ { 2 } - \sum _ { t = 1 } ^ { T } \langle u ^ { ( t ) } , x ^ { ( t + 1 ) } - x ^ { ( t ) } \rangle ,
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e</sub> f<sub>ac</sub>t th<sub>a</sub>t $\| z \| _ { 2 } \leq 1$ <sub>.</sub> M<sub>ax</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng over</sub> $z \in \Delta ( d )$ <sub>, we o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> d<sub>es</sub>i<sub>re</sub>d i<sub>nequa</sub>lit<sub>y.</sub> □

Lemma 5.6. At time step $t \geq t _ { 0 } ,$ a PRM<sup>+</sup> learner satisfies

$$
\langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle \geq \frac { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + 2 \| [ r ^ { ( t + 1 ) } ] _ { + } - \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } .
$$

Proof. From $\mathrm { P R M ^ { + } }$ u<sub>p</sub>dates (3) and (4), we have the followin<sub>g</sub> two identities

$$
\begin{array} { r l } & { \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , \tilde { r } ^ { ( t + 1 ) } \rangle = \langle \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } , \tilde { r } ^ { ( t + 1 ) } \rangle , } \\ & { \qquad { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } = ( [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } - \tilde { r } ^ { ( t + 1 ) } ) + ( { \pmb r } ^ { ( t + 1 ) } - [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } ) . } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>equa</sub>lit<sub>y</sub> h<sub>o</sub>ld<sub>s</sub> b<sub>ecause, on coor</sub>di<sub>na</sub>t<sub>es w</sub>h<sub>ere</sub> $\tilde { r } ^ { ( t + 1 ) }$ i<sub>s</sub> <sub>pos</sub>iti<sub>ve,</sub> th<sub>e</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> i<sub>s</sub> <sub>un</sub>t<sub>runca</sub>t<sub>e</sub>d<sub>;</sub> <sub>on</sub> th<sub>e rema</sub>i<sub>n</sub>i<sub>n coor</sub>di<sub>na</sub>t<sub>es,</sub> b<sub>o</sub>th i<sub>nner ro</sub>d<sub>uc</sub>t<sub>s con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e zero.</sub> F<sub>or</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>ex ress</sub>i<sub>on, we can ex an</sub>d it<sub>s</sub> RHS t<sub>o</sub> g<sup>et</sup>

$$
\langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , \tilde { r } ^ { ( t + 1 ) } \rangle = \frac { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { 2 } .\tag{34}
$$

F<sub>or</sub> th<sub>e</sub> <sub>secon</sub>d <sub>express</sub>i<sub>on,</sub> <sub>we</sub> <sub>can</sub> t<sub>a</sub>k<sub>e</sub> th<sub>e</sub> i<sub>nner</sub> <sub>pro</sub>d<sub>uc</sub>t <sub>w</sub>ith $[ \pmb { r } ^ { ( t + 1 ) } ] _ { + } - \tilde { \pmb { r } } ^ { ( t + 1 ) }$ f<sub>or</sub> b<sub>o</sub>th <sub>s</sub>id<sub>es</sub> t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } & { \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } - \tilde { { \pmb r } } ^ { ( t + 1 ) } \rangle } \\ & { = \| [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } - \tilde { { \pmb r } } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } + \langle { \pmb r } ^ { ( t + 1 ) } - [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } , [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \rangle - \langle { \pmb r } ^ { ( t + 1 ) } - [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } , \tilde { { \pmb r } } ^ { ( t + 1 ) } \rangle } \\ & { = \| [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } - \tilde { { \pmb r } } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \langle { \pmb r } ^ { ( t + 1 ) } - [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } , \tilde { { \pmb r } } ^ { ( t + 1 ) } \rangle } \\ & { \ge \| [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } - \tilde { { \pmb r } } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e</sub> f<sub>ac</sub>t th<sub>a</sub>t $r ^ { ( t + 1 ) } - \lceil r ^ { ( t + 1 ) } \rceil _ { + } \leq 0$ <sub>an</sub>d $\tilde { r } ^ { ( t + 1 ) } \geq \mathbf { 0 }$

Addi<sub>ng</sub> thi<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> t<sub>o</sub> th<sub>e a</sub>b<sub>ove</sub> id<sub>en</sub>tit<sub>y</sub> f<sub>or</sub> $\langle { \pmb u } ^ { ( t ) } - \langle { { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } } \rangle { \bf 1 } , \tilde { r } ^ { ( t + 1 ) } \rangle$ ⟩, we obtain

$$
\langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \rangle \ge \frac { \| \tilde { \pmb r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { \pmb r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \| \tilde { \pmb r } ^ { ( t + 1 ) } - \tilde { \pmb r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { 2 } + \| [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } - \tilde { \pmb r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } ,
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>mp</sub>li<sub>es</sub>

$$
\frac { \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \pmb 1 } , [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \rangle } { \| [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } \geq \frac { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + 2 \| [ r ^ { ( t + 1 ) } ] _ { + } - \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } .
$$

Si<sub>nce</sub> $\pmb { x } ^ { ( t + 1 ) } = [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } / \Vert [ \pmb { r } ^ { ( t + 1 ) } ]$ <sub>+</sub>∥<sub>1</sub>, we can rewrite the LHS as

$$
\frac { \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \rangle } { \| [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } = \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , { \pmb x } ^ { ( t + 1 ) } \rangle = \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle .
$$

H<sub>ence, we o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> d<sub>es</sub>i<sub>re</sub>d <sub>c</sub>l<sub>a</sub>i<sub>m.</sub>

Lemma 5.5. For every $T \geq 1$ , a PRM<sup>+</sup> learner satisfies

$$
\operatorname* { m a x } _ { 1 \leq s \leq T + 1 } \| \tilde { r } ^ { ( s ) } \| _ { 2 } ^ { 2 } \leq \| \mathbf { A } \| _ { 2 } ^ { 2 } + 2 ( 1 + \sqrt { d } ) ^ { 2 } \sum _ { t = 2 } ^ { T } \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + 2 d \| \mathbf { A } \| _ { 2 } ^ { 2 } \sum _ { t = 1 } ^ { T } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } .\tag{16}
$$

Proof. For $t \geq 2 .$ , we use identit<sub>y</sub> (34) established in the <sub>p</sub>roof of Lemma 5.6 to obtain

$$
\begin{array} { r } { \| \tilde { { \boldsymbol r } } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { { \boldsymbol r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } = 2 \langle { \boldsymbol u } ^ { ( t ) } - \langle { \boldsymbol u } ^ { ( t ) } , { \boldsymbol x } ^ { ( t ) } \rangle { \bf 1 } , \tilde { { \boldsymbol r } } ^ { ( t + 1 ) } - [ { \boldsymbol r } ^ { ( t ) } ] _ { + } \rangle - \| \tilde { { \boldsymbol r } } ^ { ( t + 1 ) } - \tilde { { \boldsymbol r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } . } \end{array}
$$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> $\pmb { r } ^ { ( t ) } = \tilde { \pmb { r } } ^ { ( t ) } + \pmb { u } ^ { ( t - 1 ) } - \langle \pmb { u } ^ { ( t - 1 ) } , \pmb { x } ^ { ( t - 1 ) } \rangle ]$ <sub>an</sub>d <sub>expan</sub>di<sub>ng</sub> th<sub>e</sub> <sub>square</sub>d <sub>norm</sub> $\| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 }$ , w<sup>e</sup> g<sup>et</sup>

$$
\begin{array} { r l } & { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } = 2 \langle u ^ { ( t ) } - u ^ { ( t - 1 ) } - ( \langle u ^ { ( t ) } , x ^ { ( t ) } \rangle - \langle u ^ { ( t - 1 ) } , { \bf x } ^ { ( t - 1 ) } \rangle ) { \bf 1 } , \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } \rangle } \\ & { \qquad - \| \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } - \| [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + 2 \langle r ^ { ( t ) } - [ r ^ { ( t ) } ] _ { + } , \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } \rangle } \\ & { \qquad \le 2 \langle u ^ { ( t ) } - u ^ { ( t - 1 ) } - ( \langle u ^ { ( t ) } , x ^ { ( t ) } \rangle - \langle u ^ { ( t - 1 ) } , { \bf x } ^ { ( t - 1 ) } \rangle ) { \bf 1 } , \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } \rangle } \\ & { \qquad - \| \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } - \| [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } \\ & { \qquad \le \Big \| u ^ { ( t ) } - u ^ { ( t - 1 ) } - ( \langle u ^ { ( t ) } , { \bf x } ^ { ( t ) } \rangle - \langle u ^ { ( t - 1 ) } , { \bf x } ^ { ( t - 1 ) } \rangle ) { \bf 1 } \Big \| _ { 2 } ^ { 2 } , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> fi<sub>rs</sub>t i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> $\langle \pmb { r } ^ { ( t ) } - [ \pmb { r } ^ { ( t ) } ] _ { + } , \tilde { \pmb { r } } ^ { ( t + 1 ) } - [ \pmb { r } ^ { ( t ) } ] _ { + } \rangle = \langle \pmb { r } ^ { ( t ) } - [ \pmb { r } ^ { ( t ) } ] _ { + } , \tilde { \pmb { r } } ^ { ( t + 1 ) } \rangle \leq 0$ <sub>, an</sub>d th<sub>e secon</sub>d i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> b<sub>y</sub> <sub>comp</sub>l<sub>e</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>square</sub> <sub>an</sub>d di<sub>scar</sub>di<sub>ng</sub> $- \| [ { \pmb r } ^ { ( t ) } ] _ { + } - \tilde { { \pmb r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } .$

Usin<sub>g</sub> $\| \pmb { x } ^ { ( t ) } \| _ { 2 } \leq 1$ <sub>an</sub>d $\| \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } \leq \| \mathbf { A } \| _ { 2 } ,$ <sub>, we</sub> f<sub>ur</sub>th<sub>er o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } & { \left\| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } - \left( \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle - \langle { \pmb u } ^ { ( t - 1 ) } , { \pmb x } ^ { ( t - 1 ) } \rangle \right) { \bf 1 } \right\| _ { 2 } } \\ & { \leq \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } \| _ { 2 } + \sqrt { d } \left| \langle { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } , { \pmb x } ^ { ( t ) } \rangle \right| + \sqrt { d } \left| \langle { \pmb u } ^ { ( t - 1 ) } , { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t - 1 ) } \rangle \right| } \\ & { \leq ( 1 + \sqrt { d } ) \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } \| _ { 2 } + \sqrt { d } \| { \bf A } \| _ { 2 } \| { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t - 1 ) } \| _ { 2 } . } \end{array}
$$

Thi<sub>s</sub> i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t

$$
\begin{array} { r l r } & { } & { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } \leq \Big ( ( 1 + \sqrt { d } ) \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } \| _ { 2 } + \sqrt { d } \| { \pmb A } \| _ { 2 } \| { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t - 1 ) } \| _ { 2 } \Big ) ^ { 2 } } \\ & { } & { \qquad \leq 2 ( 1 + \sqrt { d } ) ^ { 2 } \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + 2 d \| { \pmb A } \| _ { 2 } ^ { 2 } \| { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } . } \end{array}
$$

At i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on,</sub> $\tilde { r } ^ { ( 1 ) } = \mathbf { 0 }$ <sub>an</sub>d ${ \pmb x } ^ { ( 1 ) } = { \pmb 1 } / d$ , so

$$
\big \| \ b { \tilde { r } } ^ { ( 2 ) } \big \| _ { 2 } ^ { 2 } \leq \left\| \ b { u } ^ { ( 1 ) } - \frac { \langle \ b { u } ^ { ( 1 ) } , \mathbf { 1 } \rangle } { d } \mathbf { 1 } \right\| _ { 2 } ^ { 2 } = \big \| \ b { u } ^ { ( 1 ) } \big \| _ { 2 } ^ { 2 } - \frac { \langle \ b { u } ^ { ( 1 ) } , \mathbf { 1 } \rangle ^ { 2 } } { d } \leq \| \mathbf { A } \| _ { 2 } ^ { 2 } .
$$

F<sub>or eac</sub>h $2 \leq s \leq T + 1$ <sub>, we now sum</sub> th<sub>e norm</sub> i<sub>ncremen</sub>t<sub>s:</sub>

$$
\begin{array} { l } { \displaystyle \| \tilde { r } ^ { ( s ) } \| _ { 2 } ^ { 2 } = \| \tilde { r } ^ { ( 2 ) } \| _ { 2 } ^ { 2 } + \sum _ { t = 2 } ^ { s - 1 } \left( \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } \right) } \\ { \displaystyle \qquad \leq \| \mathbf { A } \| _ { 2 } ^ { 2 } + 2 ( 1 + \sqrt { d } ) ^ { 2 } \sum _ { t = 2 } ^ { s - 1 } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + 2 d \| \mathbf { A } \| _ { 2 } ^ { 2 } \sum _ { t = 1 } ^ { s - 2 } \| \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } \| _ { 2 } ^ { 2 } } \\ { \displaystyle \qquad \leq \| \mathbf { A } \| _ { 2 } ^ { 2 } + 2 ( 1 + \sqrt { d } ) ^ { 2 } \sum _ { t = 2 } ^ { T } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + 2 d \| \mathbf { A } \| _ { 2 } ^ { 2 } \sum _ { t = 1 } ^ { T } \| \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } \| _ { 2 } ^ { 2 } , } \end{array}
$$

<sub>w</sub>h<sub>ere we re</sub>i<sub>n</sub>d<sub>ex</sub> th<sub>e s</sub>t<sub>ra</sub>t<sub>egy-movemen</sub>t <sub>sum</sub> th<sub>e</sub> fi<sub>rs</sub>t i<sub>nequa</sub>lit<sub>y, an</sub>d <sub>ex</sub>t<sub>en</sub>d th<sub>e nonnega</sub>ti<sub>ve sums</sub> f<sub>or</sub> th<sub>e secon</sub>d<sub>.</sub> Th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> b<sub>oun</sub>d i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f <sub>� an</sub>d <sub>a</sub>l<sub>so</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> $s = 1$ <sub>, s</sub>i<sub>nce</sub> $\tilde { r } ^ { ( 1 ) } = \mathbf { 0 }$ <sub>.</sub> T<sub>a</sub>ki<sub>ng</sub> th<sub>e max</sub>i<sub>mum over</sub> $1 \leq s \leq T + 1$ <sub>g</sub>i<sub>ves us</sub> th<sub>e</sub> d<sub>es</sub>i<sub>re</sub>d <sub>c</sub>l<sub>a</sub>i<sub>m.</sub> □

Lemma 5.7. For any $T \geq t _ { 0 } ,$ , we have

$$
\begin{array} { l } { \displaystyle \sum _ { t = t _ { 0 } } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + 2 \| [ r ^ { ( t + 1 ) } ] _ { + } - \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } \geq \frac { 1 } { 6 d } \displaystyle \sum _ { t = t _ { 0 } } ^ { T } \operatorname* { m a x } \{ \| [ r ^ { ( t ) } ] _ { + } \| _ { 1 } , \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } \} \| x ^ { ( t + 1 ) } - x ^ { ( t ) } \| _ { 2 } ^ { 2 } } \\ { \displaystyle \qquad - \frac { \| [ r ^ { ( t _ { 0 } ) } ] _ { + } - \tilde { r } ^ { ( t _ { 0 } ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t _ { 0 } ) } ] _ { + } \| _ { 1 } } . } \end{array}
$$

Proof. We first show the normalization bound

$$
\begin{array} { r } { \operatorname* { m a x } \{ \| [ \boldsymbol { r } ^ { ( t ) } ] _ { + } \| _ { 1 } , \| [ \boldsymbol { r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } \} \| \boldsymbol { x } ^ { ( t + 1 ) } - \boldsymbol { x } ^ { ( t ) } \| _ { 2 } \leq \sqrt { d } \| [ \boldsymbol { r } ^ { ( t + 1 ) } ] _ { + } - [ \boldsymbol { r } ^ { ( t ) } ] _ { + } \| _ { 2 } . } \end{array}\tag{35}
$$

I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> if $\| [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } \geq \| [ \pmb { r } ^ { ( t ) } ] _ { + } \| _ { 1 }$ <sub>, we ca</sub>n <sub>w</sub>rit<sub>e</sub> th<sub>e</sub> LHS <sub>as</sub>

$$
\begin{array} { r l } {  { \| [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } ( \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } ) = \| [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } ( \frac { [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } } { \| [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } - \pmb { x } ^ { ( t ) } ) } } \\ & { = [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } - \| [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } \pmb { x } ^ { ( t ) } } \\ & { = ( I - \pmb { x } ^ { ( t ) } \pmb { 1 } ^ { \top } ) [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } . } \end{array}
$$

Si<sub>nce</sub> $[ { \pmb r } ^ { ( t ) } ] _ { + } = \| [ { \pmb r } ^ { ( t ) } ] _ { + } \| _ { 1 } { \pmb x } ^ { ( t ) }$ <sub>,</sub> <sub>we</sub> <sub>no</sub>t<sub>e</sub> th<sub>a</sub>t

$$
( I - \pmb { x } ^ { ( t ) } \mathbf { 1 } ^ { \top } ) [ \pmb { r } ^ { ( t ) } ] _ { + } = [ \pmb { r } ^ { ( t ) } ] _ { + } - \pmb { x } ^ { ( t ) } \Vert [ \pmb { r } ^ { ( t ) } ] _ { + } \Vert _ { 1 } = [ \pmb { r } ^ { ( t ) } ] _ { + } - [ \pmb { r } ^ { ( t ) } ] _ { + } = 0 .
$$

A<sub>s</sub> $\| I - x ^ { ( t ) } \mathbf { 1 } ^ { \top } \| _ { 2 } \leq \sqrt { d }$ <sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> <sub>a</sub>b<sub>ove</sub> <sub>norma</sub>li<sub>za</sub>ti<sub>on</sub> b<sub>oun</sub>d<sub>.</sub> Th<sub>e</sub> <sub>o</sub>th<sub>er</sub> <sub>case</sub> i<sub>s</sub> <sub>symme</sub>t<sub>r</sub>i<sub>c.</sub>

W<sub>e</sub> <sub>now</sub> d<sub>ecompose</sub> th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> dif<sub>erence</sub> <sub>as</sub>

$$
[ \boldsymbol { r } ^ { ( t + 1 ) } ] _ { + } - [ \boldsymbol { r } ^ { ( t ) } ] _ { + } = \big ( [ \boldsymbol { r } ^ { ( t + 1 ) } ] _ { + } - \tilde { \boldsymbol { r } } ^ { ( t + 1 ) } \big ) + \big ( \tilde { \boldsymbol { r } } ^ { ( t + 1 ) } - \tilde { \boldsymbol { r } } ^ { ( t ) } \big ) + \big ( \tilde { \boldsymbol { r } } ^ { ( t ) } - [ \boldsymbol { r } ^ { ( t ) } ] _ { + } \big ) .
$$

A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> <sub>norma</sub>li<sub>za</sub>ti<sub>on</sub> b<sub>oun</sub>d <sub>an</sub>d th<sub>en</sub> C<sub>auc</sub>h<sub>y–</sub>S<sub>c</sub>h<sub>warz,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } &  \frac { \operatorname* { m a x } \{ \| [ r ^ { ( t ) } ] _ { + } \| _ { 1 } , \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } \} \| { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } { \mathbf { | } } } \\ &  \qquad \leq \frac { \| \tilde { r ^ { ( t + 1 ) } - \tilde { r ^ { ( t ) } } \| _ { + } ^ { 2 } + \| [ r ^ { ( t + 1 ) } ] _ { + } - \tilde { r ^ { ( t + 1 ) } } \| _ { 2 } ^ { 2 } + \| [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { 2 \operatorname* { m a x } \{ \| [ r ^ { ( t ) } ] _ { + } \| _ { 1 } , \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } \} } } \\ &  \qquad \leq \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \| [ r ^ { ( t + 1 ) } ] _ { + } - \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } + \frac { \| [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } }  2 \| [ r ^  ( t \end{array}
$$

<sub>w</sub>h<sub>ere</sub> <sub>we</sub> <sub>rep</sub>l<sub>ace</sub> th<sub>e</sub> <sub>common</sub> d<sub>enom</sub>i<sub>na</sub>t<sub>or</sub> b<sub>y</sub> th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>sma</sub>ll<sub>er</sub> <sub>norms</sub> f<sub>or</sub> th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y.</sub>

W<sub>e can sum</sub> th<sub>e</sub> l<sub>as</sub>t t<sub>erm</sub> f<sub>or</sub> $t = t _ { 0 } , \ldots , T$ an<sup>d</sup> rearran<sub>g</sub>e

$$
\begin{array} { r l } { \displaystyle \sum _ { t = t _ { 0 } } ^ { T } \frac { \| [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t ) } ] _ { + } \| _ { 1 } } = \displaystyle \sum _ { t = t _ { 0 } } ^ { T } \frac { \| [ r ^ { ( t + 1 ) } ] _ { + } - \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } + \frac { \| [ r ^ { ( t _ { 0 } ) } ] _ { + } - \tilde { r } ^ { ( t _ { 0 } ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t _ { 0 } ) } ] _ { + } \| _ { 1 } } - \frac { \| [ r ^ { ( T + 1 ) } ] _ { + } - \tilde { r } ^ { ( T + 1 ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( T + 1 ) } ] _ { + } \| _ { 1 } } } & { } \\ { \leq \displaystyle \sum _ { t = t _ { 0 } } ^ { T } \frac { \| [ r ^ { ( t + 1 ) } ] _ { + } - \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } + \frac { \| [ r ^ { ( t _ { 0 } ) } ] _ { + } - \tilde { r } ^ { ( t _ { 0 } ) } \| _ { 2 } ^ { 2 } } { 2 \| [ r ^ { ( t _ { 0 } ) } ] _ { + } \| _ { 1 } } . } & { } \end{array}
$$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> thi<sub>s</sub> id<sub>en</sub>tit<sub>y</sub> <sub>an</sub>d di<sub>scar</sub>di<sub>ng</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>nonpos</sub>iti<sub>ve</sub> t<sub>erm,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> <sub>c</sub>l<sub>a</sub>i<sub>me</sub>d i<sub>nequa</sub>lit<sub>y.</sub>

## A.2 Proofs from Subsection 5.2

Lemma 5.8. At every time step $t \geq 1 ,$ , an $\widetilde { \cal I } R { - } P R M ^ { + }$ learner satisfies

$$
\langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle \geq \frac { 1 } { 4 d } \| \tilde { { \pmb r } } ^ { ( t + 1 ) } \| _ { 2 } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } + c ^ { ( t + 1 ) } - c ^ { ( t ) } .\tag{18}
$$

Proof. Suppose first that $\tilde { r } ^ { ( t + 1 ) } \neq \mathbf { 0 }$ <sub>.</sub> Th<sub>e</sub> IR r<sub>eg</sub>r<sub>e</sub>t <sub>vec</sub>t<sub>o</sub>r <sub>up</sub>d<sub>a</sub>t<sub>e</sub> <sub>g</sub>i<sub>ves</sub>

$$
\tilde { r } ^ { ( t + 1 ) } \geq \tilde { r } ^ { ( t ) } + u ^ { ( t ) } - \langle u ^ { ( t ) } , { \bf x } ^ { ( t ) } \rangle { \bf 1 } + c ^ { ( t ) } { \bf 1 } .
$$

T<sub>a</sub>ki<sub>ng</sub> th<sub>e</sub> i<sub>nner pro</sub>d<sub>uc</sub>t <sub>w</sub>ith $[ r ^ { ( t ) } ] _ { + } ,$ <sub>, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } & { \langle \tilde { r } ^ { ( t + 1 ) } , [ r ^ { ( t ) } ] _ { + } \rangle \ge \langle \tilde { r } ^ { ( t ) } , [ r ^ { ( t ) } ] _ { + } \rangle + \langle u ^ { ( t ) } - \langle u ^ { ( t ) } , x ^ { ( t ) } \rangle \mathbf { 1 } , [ r ^ { ( t ) } ] _ { + } \rangle + c ^ { ( t ) } \| [ r ^ { ( t ) } ] _ { + } \| _ { 1 } } \\ & { \qquad = \langle r ^ { ( t ) } , [ r ^ { ( t ) } ] _ { + } \rangle + \langle u ^ { ( t ) } - \langle u ^ { ( t ) } , x ^ { ( t ) } \rangle \mathbf { 1 } , [ r ^ { ( t ) } ] _ { + } \rangle , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> <sub>equa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>ons</sub> <sub>o</sub>f $\boldsymbol { r } ^ { ( t ) }$ <sub>an</sub>d $c ^ { ( t ) }$ <sub>.</sub> Si<sub>nce</sub> $[ { \pmb r } ^ { ( t ) } ] _ { + } = \| [ { \pmb r } ^ { ( t ) } ] _ { + } \| _ { 1 } { \pmb x } ^ { ( t ) }$ , w<sup>e</sup> <sup>can</sup> g<sup>et</sup>

$$
\begin{array} { r l } & { \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle = \| [ { \pmb r } ^ { ( t ) } ] _ { + } \| _ { 1 } ( \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle \langle { \bf 1 } , { \pmb x } ^ { ( t ) } \rangle ) } \\ & { \qquad = \| [ { \pmb r } ^ { ( t ) } ] _ { + } \| _ { 1 } ( \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle ) } \\ & { \qquad = 0 . } \end{array}\tag{36}
$$

H<sub>ence,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r } { \langle \tilde { { \pmb r } } ^ { ( t + 1 ) } , [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle \geq \langle { \pmb r } ^ { ( t ) } , [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle = \| [ { \pmb r } ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } = \| \tilde { { \pmb r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } . } \end{array}
$$

With thi<sub>s</sub> i<sub>nequa</sub>lit<sub>y,</sub> <sub>we</sub> <sub>ge</sub>t

$$
\begin{array} { r l } & { \| \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } = \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } + \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - 2 \langle \tilde { r } ^ { ( t + 1 ) } , [ r ^ { ( t ) } ] _ { + } \rangle } \\ & { \qquad \leq \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } . } \end{array}\tag{37}
$$

F<sub>rom</sub> th<sub>e regre</sub>t <sub>vec</sub>t<sub>or up</sub>d<sub>a</sub>t<sub>e, we</sub> h<sub>ave</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y</sub>

$$
\begin{array} { r } { \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } + c ^ { ( t ) } { \bf 1 } , \tilde { r } ^ { ( t + 1 ) } \rangle = \langle \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } , \tilde { r } ^ { ( t + 1 ) } \rangle . } \end{array}
$$

Th<sub>e nex</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>on sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\pmb { r } ^ { ( t + 1 ) } - \tilde { \pmb { r } } ^ { ( t + 1 ) } = \pmb { u } ^ { ( t ) } - \langle \pmb { u } ^ { ( t ) } , \pmb { x } ^ { ( t + 1 ) } \rangle \pmb { 1 } + c ^ { ( t + 1 ) } \pmb { 1 }$ <sub>.</sub> S<sub>u</sub>bt<sub>rac</sub>ti<sub>n</sub> it<sub>s</sub> i<sub>nner ro</sub>d<sub>uc</sub>t <sub>w</sub>ith $\tilde { r } ^ { ( t + 1 ) }$ f<sub>rom</sub> th<sub>e prece</sub>di<sub>ng</sub> id<sub>en</sub>tit<sub>y, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } & { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 } ( \langle u ^ { ( t ) } , \mathbf { x } ^ { ( t + 1 ) } - \mathbf { x } ^ { ( t ) } \rangle + c ^ { ( t ) } - c ^ { ( t + 1 ) } ) = \langle \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } , \tilde { r } ^ { ( t + 1 ) } \rangle + \langle \tilde { r } ^ { ( t + 1 ) } - r ^ { ( t + 1 ) } , \tilde { r } ^ { ( t + 1 ) } \rangle } \\ & { \qquad \geq \frac { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \| \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 2 } ^ { 2 } } { 2 } } \\ & { \qquad \overset { ( 3 ) } { \geq } \frac { \| \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } + \| \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 2 } ^ { 2 } } { 2 } . } \end{array}
$$

Th<sub>e</sub> fi<sub>rs</sub>t i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> $r ^ { ( t + 1 ) } \leq [ r ^ { ( t + 1 ) } ] _ { + }$ <sub>, an</sub>d $\| [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } \| _ { 2 } = \| \widetilde { \pmb { r } } ^ { ( t + 1 ) } \| _ { 2 } ;$ for the second, we use (37) bound <sub>an</sub>d di<sub>scar</sub>d $\lVert \bar { \boldsymbol { r } } ^ { ( t + \bar { 1 } ) } - \tilde { \boldsymbol { r } } ^ { ( t ) } \rVert _ { 2 } ^ { 2 }$ <sub>.</sub> W<sub>e can</sub> di<sub>v</sub>id<sub>e</sub> $\| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 }$ <sub>on</sub> b<sub>o</sub>th <sub>s</sub>id<sub>es</sub> t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle + c ^ { ( t ) } - c ^ { ( t + 1 ) } \geq \frac { \| \tilde { \pmb r } ^ { ( t + 1 ) } - [ { \pmb r } ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } + \| \tilde { \pmb r } ^ { ( t + 1 ) } - [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \| _ { 2 } ^ { 2 } } { 2 \| \tilde { \pmb r } ^ { ( t + 1 ) } \| _ { 1 } } .
$$

A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the normalization bound (35) to $\tilde { r } ^ { ( t + 1 ) }$ <sub>an</sub>d $[ r ^ { ( s ) } ] _ { + }$ <sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } & { \left\| \tilde { r } ^ { ( t + 1 ) } - \| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 } { \boldsymbol x } ^ { ( s ) } \right\| _ { 2 } = \| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 } \left\| \frac { \tilde { r } ^ { ( t + 1 ) } } { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 } } - { \boldsymbol x } ^ { ( s ) } \right\| _ { 2 } } \\ & { \qquad \leq \operatorname* { m a x } \{ \| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 } , \| [ r ^ { ( s ) } ] _ { + } \| _ { 1 } \} \left\| \frac { \tilde { r } ^ { ( t + 1 ) } } { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 } } - { \boldsymbol x } ^ { ( s ) } \right\| _ { 2 } } \\ & { \qquad \leq \sqrt { d } \| \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( s ) } ] _ { + } \| _ { 2 } . } \end{array}
$$

F<sub>or</sub> $s = t$ <sub>an</sub>d $s = t + 1$ <sub>,</sub> th<sub>ese</sub> b<sub>oun</sub>d<sub>s</sub> <sub>con</sub>t<sub>ro</sub>l th<sub>e</sub> t<sub>wo</sub> t<sub>erms</sub> i<sub>n</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y</sub>

$$
\| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 } ( { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } ) = ( \tilde { r } ^ { ( t + 1 ) } - \| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 } { \pmb x } ^ { ( t ) } ) - ( \tilde { r } ^ { ( t + 1 ) } - \| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 } { \pmb x } ^ { ( t + 1 ) } ) .
$$

T<sub>a</sub>ki<sub>ng square</sub>d <sub>norms an</sub>d <sub>app</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e square</sub>d t<sub>r</sub>i<sub>ang</sub>l<sub>e</sub> i<sub>nequa</sub>lit<sub>y, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } & { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 } ^ { 2 } \| { \boldsymbol x } ^ { ( t + 1 ) } - { \boldsymbol x } ^ { ( t ) } \| _ { 2 } ^ { 2 } \leq 2 \| \tilde { r } ^ { ( t + 1 ) } - \| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 } { \boldsymbol x } ^ { ( t ) } \| _ { 2 } ^ { 2 } + 2 \| \tilde { r } ^ { ( t + 1 ) } - \| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 } { \boldsymbol x } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq 2 d ( \| \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } + \| \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 2 } ^ { 2 } ) . } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> <sub>secon</sub>d i<sub>nequa</sub>lit<sub>y</sub> <sub>app</sub>li<sub>es</sub> th<sub>e</sub> t<sub>wo</sub> <sub>norma</sub>li<sub>za</sub>ti<sub>on</sub> b<sub>oun</sub>d<sub>s</sub> <sub>a</sub>ft<sub>er</sub> <sub>squar</sub>i<sub>ng</sub> th<sub>em.</sub> Di<sub>v</sub>idi<sub>ng</sub> b<sub>y</sub> 2� <sub>an</sub>d <sub>su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> <sub>u</sub>tilit<sub>y-</sub>i<sub>mprovemen</sub>t b<sub>oun</sub>d<sub>,</sub> <sub>we</sub> <sub>ge</sub>t

$$
\begin{array} { r } { \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle + c ^ { ( t ) } - c ^ { ( t + 1 ) } \geq \frac { \| \tilde { \pmb r } ^ { ( t + 1 ) } \| _ { 1 } ^ { 2 } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { 4 d \| \tilde { \pmb r } ^ { ( t + 1 ) } \| _ { 1 } } } \\ { = \frac { \| \tilde { \pmb r } ^ { ( t + 1 ) } \| _ { 1 } } { 4 d } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } } \\ { \geq \frac { \| \tilde { \pmb r } ^ { ( t + 1 ) } \| _ { 2 } } { 4 d } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } . } \end{array}
$$

R<sub>earrang</sub>i<sub>ng, we o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> d<sub>es</sub>i<sub>re</sub>d b<sub>oun</sub>d

$$
\langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle \geq \frac { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } } { 4 d } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } + c ^ { ( t + 1 ) } - c ^ { ( t ) } .
$$

If $\tilde { r } ^ { ( t + 1 ) } = \mathbf { 0 } .$ <sub>,</sub> th<sub>e norm-</sub>i<sub>ncrease</sub> b<sub>oun</sub>d <sub>prove</sub>d <sub>a</sub>b<sub>ove</sub> i<sub>mp</sub>li<sub>es</sub> $\tilde { \mathbf { \Delta } } \tilde { r } ^ { ( t ) } = \mathbf { 0 }$ <sub>.</sub> B<sub>o</sub>th <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es are</sub> th<sub>en</sub> th<sub>e un</sub>if<sub>orm</sub> d<sub>e</sub>f<sub>au</sub>lt <sup>strate</sup>gy <sup>and</sup> $c ^ { ( t ) } = c ^ { ( t + 1 ) } = 0 ,$ <sub>, so</sub> b<sub>o</sub>th <sub>s</sub>id<sub>es o</sub>f th<sub>e c</sub>l<sub>a</sub>i<sub>me</sub>d i<sub>nequa</sub>lit<sub>y are zero.</sub> □

Proposition 5.9 (Alt-RVU for IR-PRM<sup>+</sup>). For any $\beta > 0 ,$ , a $\scriptstyle { \underline { { I R } } } - P R M ^ { + }$ learner satisfies the Alt-RVU bound $f o r$ $\alpha = 2 \| \mathbf { A } \| _ { 2 } + ( 1 + \sqrt { d } ) ^ { 2 } / \beta$ and $\gamma = \| \widetilde { r } ^ { ( t _ { 0 } ) } \| _ { 2 } / ( 4 d )$ . Therefore, in every finite two-player zero-sum normal-form game, alternating self-play with IR-PRM<sup>+</sup> has ${ \cal O } ( T ^ { - 1 } )$ average-iterate convergence.

Proof. We have just shown that

$$
\| \tilde { \pmb { r } } ^ { ( T + 1 ) } \| _ { 2 } ^ { 2 } \leq \| \mathbf { A } \| _ { 2 } ^ { 2 } + ( 1 + \sqrt { d } ) ^ { 2 } \sum _ { t = 2 } ^ { T } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } .
$$

T<sub>a</sub>ki<sub>ng</sub> <sub>square</sub> <sub>roo</sub>t<sub>s</sub> <sub>an</sub>d <sub>app</sub>l<sub>y</sub>i<sub>ng</sub> Y<sub>oung</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
2 \| { \tilde { r } } ^ { ( T + 1 ) } \| _ { 2 } \le 2 \| \mathbf { A } \| _ { 2 } + 2 ( 1 + \sqrt { d } ) \sqrt { \sum _ { t = 2 } ^ { T } \| u ^ { ( t ) } - u ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } } \le 2 \| \mathbf { A } \| _ { 2 } + \frac { ( 1 + \sqrt { d } ) ^ { 2 } } { \beta } + \beta \sum _ { t = 2 } ^ { T } \| u ^ { ( t ) } - u ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } .
$$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e a</sub>b<sub>ove a</sub>lt<sub>erna</sub>ti<sub>ng regre</sub>t b<sub>oun</sub>d <sub>proves</sub> th<sub>e c</sub>l<sub>a</sub>i<sub>m w</sub>ith th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>� an</sub>d $\gamma$

## A.3 The Theoretical Role of Half Prediction

T<sub>o exp</sub>l<sub>a</sub>i<sub>n</sub> th<sub>e c</sub>h<sub>o</sub>i<sub>ce o</sub>f h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on, we</sub> fi<sub>rs</sub>t d<sub>er</sub>i<sub>ve a correc</sub>t<sub>e</sub>d <sub>square</sub>d <sub>norm</sub> f<sub>or a genera</sub>l <sub>pre</sub>di<sub>c</sub>ti<sub>on mu</sub>lti<sub>p</sub>li<sub>er</sub> $\lambda \geq 0$ . Re<sub>p</sub>eatin<sub>g</sub> the ar<sub>g</sub>ument underl<sub>y</sub>in<sub>g</sub> (21), the <sub>p</sub>rediction ste<sub>p</sub> <sub>g</sub>ives

$$
\| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } \geq \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - 2 \lambda \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } .
$$

A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> 2<sub>�</sub>� $\leq a ^ { 2 } + b ^ { 2 }$ <sub>an</sub>d <sub>regroup</sub>i<sub>ng,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r } { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } \geq \lambda \left( \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } \right) + ( 1 - 2 \lambda ) \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } . } \end{array}
$$

Th<sub>e</sub> fi<sub>rs</sub>t t<sub>erm on</sub> th<sub>e r</sub>i<sub>g</sub>ht i<sub>s</sub> it<sub>se</sub>lf <sub>a</sub> dif<sub>erence</sub> b<sub>e</sub>t<sub>ween consecu</sub>ti<sub>ve square</sub>d i<sub>ncremen</sub>t<sub>s.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore su</sub>bt<sub>rac</sub>t th<sub>e</sub> <sub>correspon</sub>di<sub>ng correc</sub>ti<sub>on</sub> f<sub>rom</sub> th<sub>e square</sub>d <sub>regre</sub>t <sub>vec</sub>t<sub>or norm an</sub>d d<sub>e</sub>fi<sub>ne</sub>

$$
f _ { \lambda } ^ { ( t ) } = \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \lambda \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } .
$$

Thi<sub>s a</sub>b<sub>sor</sub>b<sub>s</sub> th<sub>a</sub>t dif<sub>erence</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> l<sub>e</sub>ft<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e an</sub>d <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
f _ { \lambda } ^ { ( t + 1 ) } - f _ { \lambda } ^ { ( t ) } \geq ( 1 - 2 \lambda ) \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } .
$$

Th<sub>us,</sub> $f _ { \lambda } ^ { ( t ) }$ i<sub>s</sub> <sub>non</sub>d<sub>ecreas</sub>i<sub>ng</sub> f<sub>or</sub> <sub>every</sub> $0 \leq \lambda \leq 1 / 2$ <sub>, an</sub>d <sub>a</sub>t $\lambda = 1 / 2$ it <sub>co</sub>i<sub>nc</sub>id<sub>es w</sub>ith th<sub>e correc</sub>t<sub>e</sub>d <sub>square</sub>d <sub>norm use</sub>d i<sub>n our ana</sub>l<sub>ys</sub>i<sub>s.</sub>

Thi<sub>s exp</sub>l<sub>a</sub>i<sub>ns</sub> th<sub>e correc</sub>ti<sub>on,</sub> b<sub>u</sub>t <sub>mono</sub>t<sub>on</sub>i<sub>c</sub>it<sub>y a</sub>l<sub>one</sub> d<sub>oes no</sub>t <sub>se</sub>l<sub>ec</sub>t h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on.</sub> W<sub>e a</sub>l<sub>so nee</sub>d t<sub>o</sub> b<sub>oun</sub>d th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>regre</sub>t <sub>vec</sub>t<sub>or norm.</sub> F<sub>or</sub> $0 \leq \lambda \leq 1 / 2$ , the ar<sub>g</sub>ument underl<sub>y</sub>in<sub>g</sub> (23) leaves a coeficient $1 - 2 \lambda + \varepsilon$ on <sub>cumu</sub>l<sub>a</sub>ti<sub>ve square</sub>d <sub>regre</sub>t<sub>-vec</sub>t<sub>or</sub> i<sub>ncremen</sub>t<sub>s, w</sub>h<sub>ere</sub> $\varepsilon > 0$ i<sub>s ar</sub>bit<sub>rary.</sub> At h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on,</sub> th<sub>e</sub> fi<sub>xe</sub>d <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> 1 − 2� vanishes, allowin<sub>g</sub> us to choose the remainin<sub>g</sub> coeficient suficientl<sub>y</sub> small to absorb this contribution usin<sub>g</sub> the <sub>u</sub>tilit<sub>y-</sub>i<sub>mprovemen</sub>t b<sub>oun</sub>d<sub>.</sub> F<sub>or sma</sub>ll<sub>er mu</sub>lti<sub>p</sub>li<sub>ers, a non</sub>d<sub>ecreas</sub>i<sub>ng correc</sub>t<sub>e</sub>d <sub>square</sub>d <sub>norm rema</sub>i<sub>ns ava</sub>il<sub>a</sub>bl<sub>e,</sub> b<sub>u</sub>t thi<sub>s cance</sub>ll<sub>a</sub>ti<sub>on</sub> d<sub>oes no</sub>t <sub>occur.</sub> H<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> th<sub>ere</sub>f<sub>ore supp</sub>li<sub>es</sub> b<sub>o</sub>th f<sub>orms o</sub>f <sub>con</sub>t<sub>ro</sub>l <sub>use</sub>d t<sub>o es</sub>t<sub>a</sub>bli<sub>s</sub>h Alt<sub>-</sub>RVU<sub>.</sub>

## A.4 Proofs from Subsection 5.3

Lemma 5.10. For every $t \geq 1 ,$ , a HPRM<sup>+</sup> learner satisfies

$$
\| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } \geq \frac { 1 } { 2 } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } .\tag{21}
$$

Proof. For $t \ = \ 1$ <sub>,</sub> th<sub>e</sub> i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y</sub> f<sub>rom</sub> $\tilde { r } ^ { ( 0 ) } = \tilde { r } ^ { ( 1 ) } = 0$ <sub>.</sub> F<sub>or</sub> $t \geq 2$ <sub>,</sub> th<sub>e</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>an</sub>d half-<sub>p</sub>rediction u<sub>p</sub>dates (5) im<sub>p</sub>l<sub>y</sub> that

$$
[ \pmb { r } ^ { ( t ) } ] _ { + } = \left[ \tilde { r } ^ { ( t ) } + \frac { 1 } { 2 } ( \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } ) \right] _ { + } .
$$

I<sub>n</sub>d<sub>ee</sub>d<sub>, on coor</sub>di<sub>na</sub>t<sub>es w</sub>h<sub>ere</sub> $\tilde { r } ^ { ( t ) }$ i<sub>s pos</sub>iti<sub>ve,</sub> th<sub>e prece</sub>di<sub>ng regre</sub>t <sub>vec</sub>t<sub>or up</sub>d<sub>a</sub>t<sub>e</sub> i<sub>s un</sub>t<sub>runca</sub>t<sub>e</sub>d<sub>; on</sub> th<sub>e rema</sub>i<sub>n</sub>i<sub>ng</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es,</sub> b<sub>o</sub>th <sub>s</sub>id<sub>es are zero.</sub> Si<sub>nce</sub> t<sub>a</sub>ki<sub>ng</sub> th<sub>e pos</sub>iti<sub>ve par</sub>t i<sub>s nonexpans</sub>i<sub>ve an</sub>d $\tilde { r } ^ { ( t ) } \geq \mathbf { 0 } ,$ <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\| [ \pmb { r } ^ { ( t ) } ] _ { + } - \tilde { \pmb { r } } ^ { ( t ) } \| _ { 2 } \leq \frac { 1 } { 2 } \| \tilde { \pmb { r } } ^ { ( t ) } - \tilde { \pmb { r } } ^ { ( t - 1 ) } \| _ { 2 } .\tag{38}
$$

Th<sub>e</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> <sub>a</sub>l<sub>so</sub> <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> <sub>coor</sub>di<sub>na</sub>t<sub>ew</sub>i<sub>se</sub> i<sub>nequa</sub>lit<sub>y</sub>

$$
\tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \geq u ^ { ( t ) } - \langle u ^ { ( t ) } , x ^ { ( t ) } \rangle \mathbf { 1 } .
$$

T<sub>a</sub>ki<sub>ng</sub> it<sub>s</sub> i<sub>nner pro</sub>d<sub>uc</sub>t <sub>w</sub>ith $[ { \pmb r } ^ { ( t ) } ] _ { + } \geq { \bf 0 } .$ , w<sup>e</sup> g<sup>et</sup>

$$
\begin{array} { r } { \langle \tilde { { \pmb r } } ^ { ( t + 1 ) } - \tilde { { \pmb r } } ^ { ( t ) } , [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle \geq \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle = 0 , } \end{array}\tag{39}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e equa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> $[ { \pmb r } ^ { ( t ) } ] _ { + } = \| [ { \pmb r } ^ { ( t ) } ] _ { + } \| _ { 1 } { \pmb x } ^ { ( t ) }$ <sub>,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>w</sub>h<sub>en</sub> $[ \boldsymbol { r } ^ { ( t ) } ] _ { + } = \mathbf { 0 }$

E<sub>xpan</sub>di<sub>ng</sub> th<sub>e</sub> dif<sub>erence o</sub>f <sub>square</sub>d <sub>norms an</sub>d <sub>app</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e a</sub>b<sub>ove</sub> b<sub>oun</sub>d<sub>s, we o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> d<sub>es</sub>i<sub>re</sub>d <sub>c</sub>l<sub>a</sub>i<sub>m</sub>

$$
\begin{array} { r l } & { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } = \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - 2 \langle \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } , - \tilde { r } ^ { ( t ) } \rangle } \\ & { \qquad \overset { ( 3 9 ) } \geq \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - 2 \langle \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } , [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \rangle } \\ & { \qquad \geq \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - 2 \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } \| [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \| _ { 2 } } \\ & { \qquad \overset { ( 3 8 ) } \geq \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } } \\ & { \qquad \geq \frac { 1 } { 2 } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y uses</sub> $\| \widetilde { r } ^ { ( t + 1 ) } - \widetilde { r } ^ { ( t ) } \| _ { 2 } \| \widetilde { r } ^ { ( t ) } - \widetilde { r } ^ { ( t - 1 ) } \| _ { 2 } \le \| \widetilde { r } ^ { ( t + 1 ) } - \widetilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } / 2 + \| \widetilde { r } ^ { ( t ) } - \widetilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } / 2 .$

Lemma 5.11. At every time step $t \geq t _ { 0 } - 1 _ { : }$ , a HPRM<sup>+</sup> learner satisfies

$$
\langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle \geq \frac { \| \widetilde { { \pmb r } } ^ { ( t + 1 ) } - \widetilde { { \pmb r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { 4 \| \left[ { \pmb r } ^ { ( t + 1 ) } \right] _ { + } \| _ { 1 } } - \frac { f ^ { ( t + 1 ) } - f ^ { ( t ) } } { 2 \| \left[ { \pmb r } ^ { ( t + 1 ) } \right] _ { + } \| _ { 1 } } .\tag{22}
$$

Proof. The case $t = 1$ <sub>ar</sub>i<sub>ses</sub> <sub>on</sub>l<sub>y</sub> <sub>w</sub>h<sub>en</sub> $t _ { 0 } = 2 ,$ , so $\tilde { r } ^ { ( 2 ) } \neq \mathbf { 0 } .$ Si<sub>nce</sub> $[ { r ^ { ( 2 ) } } ] _ { + } = 3 \tilde { r } ^ { ( 2 ) } / 2$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\langle { \pmb u } ^ { ( 1 ) } , { \pmb x } ^ { ( 2 ) } - { \pmb x } ^ { ( 1 ) } \rangle = \frac { \| \tilde { \pmb r } ^ { ( 2 ) } \| _ { 2 } ^ { 2 } } { \| \tilde { \pmb r } ^ { ( 2 ) } \| _ { 1 } } > 0 .
$$

M<sub>oreover,</sub> $f ^ { ( 1 ) } = 0$ <sub>an</sub>d $f ^ { ( 2 ) } = \| \tilde { r } ^ { ( 2 ) } \| _ { > } ^ { 2 } / 2$ , so the ri<sub>g</sub>ht-hand side of (22) is zero, <sub>p</sub>rovin<sub>g</sub> the claim for $t = 1$ F<sub>or</sub> $t \ge \operatorname* { m a x } \{ 2 , t _ { 0 } - 1 \}$ <sub>,</sub> th<sub>e proo</sub>f <sub>o</sub>f L<sub>emma</sub> 5<sub>.</sub>10 <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub>

$$
\langle \widetilde { \pmb { r } } ^ { ( t + 1 ) } - \widetilde { \pmb { r } } ^ { ( t ) } , [ \pmb { r } ^ { ( t ) } ] _ { + } \rangle \ge 0 , \ \| [ \pmb { r } ^ { ( t ) } ] _ { + } - \widetilde { \pmb { r } } ^ { ( t ) } \| _ { 2 } \le \frac { 1 } { 2 } \| \widetilde { \pmb { r } } ^ { ( t ) } - \widetilde { \pmb { r } } ^ { ( t - 1 ) } \| _ { 2 } .
$$

E<sub>xpan</sub>di<sub>ng</sub> th<sub>e</sub> <sub>square</sub>d <sub>norms</sub> <sub>an</sub>d <sub>app</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>ese</sub> b<sub>oun</sub>d<sub>s,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } & { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } = \| \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } - \| [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + 2 \langle \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } , [ r ^ { ( t ) } ] _ { + } \rangle } \\ & { \qquad \geq - \| [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } \\ & { \qquad \geq - \displaystyle \frac { 1 } { 4 } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } . } \end{array}\tag{40}
$$

W<sub>e nex</sub>t <sub>app</sub>l<sub>y</sub> th<sub>e pos</sub>iti<sub>ve-par</sub>t <sub>argumen</sub>t f<sub>rom</sub> L<sub>emma</sub> 5<sub>.</sub>6 t<sub>o</sub> th<sub>e</sub> h<sub>a</sub>lf<sub>-pre</sub>di<sub>c</sub>ti<sub>on up</sub>d<sub>a</sub>t<sub>e.</sub> Si<sub>nce</sub> $r ^ { ( t + 1 ) } - \tilde { r } ^ { ( t + 1 ) } =$ $( { \pmb u } ^ { ( t ) } - \langle { { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } } \rangle \bar { \bf 1 } ) / 2$ , w<sup>e</sup> g<sup>et</sup>

$$
\begin{array} { r } { \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } - \tilde { { \pmb r } } ^ { ( t + 1 ) } \rangle \geq 2 \Vert [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } - \tilde { { \pmb r } } ^ { ( t + 1 ) } \Vert _ { 2 } ^ { 2 } \geq 0 . } \end{array}
$$

Combinin<sub>g</sub> this with the identit<sub>y</sub> (34), which also holds for HPRM<sup>+</sup> because its re<sub>g</sub>ret vector u<sub>p</sub>date is unchan<sub>g</sub>ed, we <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } & { \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \rangle \ge \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , \tilde { \pmb r } ^ { ( t + 1 ) } \rangle } \\ & { \qquad = \frac { \| \tilde { \pmb r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { \pmb r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \| \tilde { \pmb r } ^ { ( t + 1 ) } - \tilde { \pmb r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { 2 } . } \end{array}
$$

B<sub>y</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f $f ^ { ( t ) }$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub>

$$
f ^ { ( t + 1 ) } - f ^ { ( t ) } = \| \tilde { \pmb { r } } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { \pmb { r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } \| \tilde { \pmb { r } } ^ { ( t + 1 ) } - \tilde { \pmb { r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 } \| \tilde { \pmb { r } } ^ { ( t ) } - \tilde { \pmb { r } } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } .
$$

C<sub>onsequen</sub>tl<sub>y, we can rewr</sub>it<sub>e</sub> th<sub>e prece</sub>di<sub>ng</sub> l<sub>ower</sub> b<sub>oun</sub>d <sub>as</sub>

$$
\begin{array} { l } { \displaystyle \frac { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { 2 } } \\ { = \displaystyle \frac { 1 } { 4 } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } ( f ^ { ( t + 1 ) } - f ^ { ( t ) } ) + \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \frac { 1 } { 4 } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } } \\ { \geq \displaystyle \frac { 1 } { 4 } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } ( f ^ { ( t + 1 ) } - f ^ { ( t ) } ) , } \end{array}
$$

where the inequalit<sub>y</sub> follows from (40). We hence have

$$
\langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \rangle \ge \frac { 1 } { 4 } \| \widetilde { \pmb r } ^ { ( t + 1 ) } - \widetilde { \pmb r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } \big ( f ^ { ( t + 1 ) } - f ^ { ( t ) } \big ) ,
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t

$$
\frac { \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \rangle } { \| [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } \geq \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { 4 \| [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } - \frac { f ^ { ( t + 1 ) } - f ^ { ( t ) } } { 2 \| [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } .
$$

Si<sub>nce</sub> $\pmb { x } ^ { ( t + 1 ) } = [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } / \Vert [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } \Vert _ { 1 }$ <sub>1, we can o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> LHS <sub>as</sub>

$$
\frac { \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \rangle } { \| [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } = \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \bf 1 } , { \pmb x } ^ { ( t + 1 ) } \rangle = \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle ,
$$

<sub>so we o</sub>bt<sub>a</sub>i<sub>ne</sub>d th<sub>e</sub> d<sub>es</sub>i<sub>re</sub>d <sub>c</sub>l<sub>a</sub>i<sub>m.</sub>

Lemma 5.12. For every $\varepsilon > 0$ and $T \geq 1 ,$ , a HPRM<sup>+</sup> learner satisfies

$$
\operatorname* { m a x } _ { 1 \leq s \leq T + 1 } \| \bar { r } ^ { ( s ) } \| _ { 2 } ^ { 2 } \leq \| \mathbb { A } \| _ { 2 } ^ { 2 } + \left( 2 + \frac { 1 } { 2 s } \right) \left[ ( 1 + \sqrt { d } ) ^ { 2 } \sum _ { t = 2 } ^ { T } \| u ^ { ( t ) } - u ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + d \| \mathbb { A } \| _ { 2 } ^ { 2 } \sum _ { t = 1 } ^ { T } \| x ^ { ( t + 1 ) } - x ^ { ( t ) } \| _ { 2 } ^ { 2 } \right] + \varepsilon \sum _ { t = 2 } ^ { T } \| \bar { r } ^ { ( t ) } - \bar { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } .\tag{23}
$$

Proof. For $t \geq 2 ,$ <sub>,</sub> th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f L<sub>emma</sub> 5<sub>.</sub>10 <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub>

$$
[ \pmb { r } ^ { ( t ) } ] _ { + } = \left[ \tilde { \pmb { r } } ^ { ( t ) } + \frac { 1 } { 2 } \big ( \tilde { \pmb { r } } ^ { ( t ) } - \tilde { \pmb { r } } ^ { ( t - 1 ) } \big ) \right] _ { + } , \| [ \pmb { r } ^ { ( t ) } ] _ { + } - \tilde { \pmb { r } } ^ { ( t ) } \| _ { 2 } \leq \frac { 1 } { 2 } \| \tilde { \pmb { r } } ^ { ( t ) } - \tilde { \pmb { r } } ^ { ( t - 1 ) } \| _ { 2 } .
$$

R<sub>eca</sub>ll f<sub>rom</sub> th<sub>e proo</sub>f <sub>o</sub>f L<sub>emma</sub> 5<sub>.</sub>5 th<sub>a</sub>t

$$
\langle \pmb { r } ^ { ( t ) } - [ \pmb { r } ^ { ( t ) } ] _ { + } , \tilde { \pmb { r } } ^ { ( t + 1 ) } - [ \pmb { r } ^ { ( t ) } ] _ { + } \rangle = \langle \pmb { r } ^ { ( t ) } - [ \pmb { r } ^ { ( t ) } ] _ { + } , \tilde { \pmb { r } } ^ { ( t + 1 ) } \rangle \leq 0 .
$$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> $\begin{array} { r } { \pmb { r } ^ { ( t ) } = \pmb { \tilde { r } } ^ { ( t ) } + \frac { 1 } { 7 } \big ( \pmb { u } ^ { ( t - 1 ) } - \langle \pmb { u } ^ { ( t - 1 ) } , \pmb { x } ^ { ( t - 1 ) } \rangle \pmb { 1 } \big ) } \end{array}$ <sub>an</sub>d <sub>rearrang</sub>i<sub>ng,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\frac { 1 } { 2 } \langle { \pmb u } ^ { ( t - 1 ) } - \langle { \pmb u } ^ { ( t - 1 ) } , { \pmb x } ^ { ( t - 1 ) } \rangle { \pmb 1 } , \tilde { { \pmb r } } ^ { ( t + 1 ) } - [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle \leq \langle [ { \pmb r } ^ { ( t ) } ] _ { + } - \tilde { { \pmb r } } ^ { ( t ) } , \tilde { { \pmb r } } ^ { ( t + 1 ) } - [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle ,
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>mp</sub>li<sub>es</sub>

$$
\langle { \pmb u } ^ { ( t - 1 ) } - \langle { \pmb u } ^ { ( t - 1 ) } , { \pmb x } ^ { ( t - 1 ) } \rangle { \pmb 1 } , \tilde { { \pmb r } } ^ { ( t + 1 ) } - [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle \leq 2 \langle [ { \pmb r } ^ { ( t ) } ] _ { + } - \tilde { { \pmb r } } ^ { ( t ) } , \tilde { { \pmb r } } ^ { ( t + 1 ) } - [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle .\tag{41}
$$

Since the re<sub>g</sub>ret vector u<sub>p</sub>date is the same, we use identit<sub>y</sub> (34) in the <sub>p</sub>roof of Lemma 5.6 to obtain

$$
\begin{array} { r } { \| \tilde { \pmb { r } } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { \pmb { r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } = 2 \langle \pmb { u } ^ { ( t ) } - \langle \pmb { u } ^ { ( t ) } , \pmb { x } ^ { ( t ) } \rangle \mathbf { 1 } , \tilde { \pmb { r } } ^ { ( t + 1 ) } - [ \pmb { r } ^ { ( t ) } ] _ { + } \rangle - \| \tilde { \pmb { r } } ^ { ( t + 1 ) } - \tilde { \pmb { r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } . } \end{array}
$$

We first add and subtract the <sub>p</sub>revious instantaneous re<sub>g</sub>ret vector. A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the inequalit<sub>y</sub> (41), we obtain

$$
\begin{array} { r l } & { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } = 2 \langle { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } - \left( \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle - \langle { \pmb u } ^ { ( t - 1 ) } , { \pmb x } ^ { ( t - 1 ) } \rangle \right) { \bf 1 } , \tilde { r } ^ { ( t + 1 ) } - [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle } \\ & { \qquad + 2 \langle { \pmb u } ^ { ( t - 1 ) } - \langle { \pmb u } ^ { ( t - 1 ) } , { \pmb x } ^ { ( t - 1 ) } \rangle { \bf 1 } , \tilde { r } ^ { ( t + 1 ) } - [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle - \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } \\ & { \qquad \le 2 \langle { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } - \left( \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle - \langle { \pmb u } ^ { ( t - 1 ) } , { \pmb x } ^ { ( t - 1 ) } \rangle \right) { \bf 1 } , \tilde { r } ^ { ( t + 1 ) } - [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle } \\ & { \qquad + 4 \langle [ { \pmb r } ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } , \tilde { r } ^ { ( t + 1 ) } - [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle - \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } . } \end{array}
$$

T<sub>o expan</sub>d th<sub>e</sub> l<sub>as</sub>t <sub>square, we wr</sub>it<sub>e</sub>

$$
\| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } = \| \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } + \| [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + 2 \langle [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } , \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } \rangle .
$$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> thi<sub>s</sub> id<sub>en</sub>tit<sub>y,</sub> th<sub>e</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>o</sub>f th<sub>e</sub> <sub>cross</sub> t<sub>erm</sub> d<sub>ecreases</sub> f<sub>rom</sub> 4 t<sub>o</sub> 2<sub>.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } & { \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } \leq 2 \langle u ^ { ( t ) } - u ^ { ( t - 1 ) } - \left( \langle u ^ { ( t ) } , x ^ { ( t ) } \rangle - \langle u ^ { ( t - 1 ) } , x ^ { ( t - 1 ) } \rangle \right) \mathbf { 1 } + [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } , \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } \rangle } \\ & { \qquad - \| \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } - \| [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } \\ & { \qquad = \bigg \| u ^ { ( t ) } - u ^ { ( t - 1 ) } - \left( \langle u ^ { ( t ) } , x ^ { ( t ) } \rangle - \langle u ^ { ( t - 1 ) } , x ^ { ( t - 1 ) } \rangle \right) \mathbf { 1 } + [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \bigg \| _ { 2 } ^ { 2 } - \| [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } \\ & { \qquad - \left\| \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } - \left( u ^ { ( t ) } - u ^ { ( t - 1 ) } - \left( \langle u ^ { ( t ) } , x ^ { ( t ) } \rangle - \langle u ^ { ( t - 1 ) } , x ^ { ( t - 1 ) } \rangle \right) \mathbf { 1 } + [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \right) \right\| _ { 2 } ^ { 2 } . } \end{array}
$$

Di<sub>scar</sub>di<sub>ng</sub> th<sub>e</sub> l<sub>as</sub>t <sub>nonpos</sub>iti<sub>ve</sub> t<sub>erm</sub> <sub>an</sub>d <sub>expan</sub>di<sub>ng</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>square,</sub> <sub>we</sub> <sub>ge</sub>t

$$
\begin{array} { r l } & { \| \check { \boldsymbol { r } } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \check { \boldsymbol { r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } \leq \| \boldsymbol { \mathbf { u } } ^ { ( t ) } - \boldsymbol { \mathbf { u } } ^ { ( t - 1 ) } - ( \langle \boldsymbol { u } ^ { ( t ) } , \boldsymbol { x } ^ { ( t ) } \rangle - \langle \boldsymbol { u } ^ { ( t - 1 ) } , \boldsymbol { x } ^ { ( t - 1 ) } \rangle ) \mathbf { 1 } \| _ { 2 } ^ { 2 } } \\ & { \qquad + 2 \langle \boldsymbol { u } ^ { ( t ) } - \boldsymbol { u } ^ { ( t - 1 ) } - ( \langle \boldsymbol { u } ^ { ( t ) } , \boldsymbol { x } ^ { ( t ) } \rangle - \langle \boldsymbol { u } ^ { ( t - 1 ) } , \boldsymbol { x } ^ { ( t - 1 ) } \rangle ) \mathbf { 1 } , [ \boldsymbol { r } ^ { ( t ) } ] _ { + } - \bar { \boldsymbol { r } } ^ { ( t ) } \rangle } \\ & { \qquad \leq \| \boldsymbol { u } ^ { ( t ) } - \boldsymbol { u } ^ { ( t - 1 ) } - ( \langle \boldsymbol { u } ^ { ( t ) } , \boldsymbol { x } ^ { ( t ) } \rangle - \langle \boldsymbol { u } ^ { ( t - 1 ) } , \boldsymbol { x } ^ { ( t - 1 ) } \rangle ) \mathbf { 1 } \| _ { 2 } ^ { 2 } } \\ & { \qquad + 2 \| \boldsymbol { u } ^ { ( t ) } - \boldsymbol { u } ^ { ( t - 1 ) } - ( \langle \boldsymbol { u } ^ { ( t ) } , \boldsymbol { x } ^ { ( t ) } \rangle - \langle \boldsymbol { u } ^ { ( t - 1 ) } , \boldsymbol { x } ^ { ( t - 1 ) } \rangle ) \mathbf { 1 } \| _ { 2 } \| [ \boldsymbol { r } ^ { ( t ) } ] _ { + } - \tilde { \boldsymbol { r } } ^ { ( t ) } \| _ { 2 } } \\ &  \qquad \leq \| \boldsymbol { u } ^ { ( t ) } - \boldsymbol { u } ^ { ( t - 1 ) } - ( \langle \boldsymbol { u } ^ { ( t ) } , \ \end{array}\tag{42}
$$

<sub>w</sub>h<sub>ere</sub> <sub>we</sub> <sub>use</sub> C<sub>auc</sub>h<sub>y–</sub>S<sub>c</sub>h<sub>warz</sub> i<sub>n</sub> th<sub>e</sub> <sub>secon</sub>d i<sub>nequa</sub>lit<sub>y</sub> <sub>an</sub>d th<sub>e</sub> b<sub>oun</sub>d $\begin{array} { r } { \| [ \pmb { r } ^ { ( t ) } ] _ { + } - \tilde { \pmb { r } } ^ { ( t ) } \| _ { 2 } \leq \frac { 1 } { 2 } \| \tilde { \pmb { r } } ^ { ( t ) } - \tilde { \pmb { r } } ^ { ( t - 1 ) } \| _ { 2 } } \end{array}$ i<sub>n</sub> th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y.</sub>

Th<sub>e</sub> <sub>u</sub>tilit<sub>y-var</sub>i<sub>a</sub>ti<sub>on</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> <sub>a</sub>l<sub>rea</sub>d<sub>y</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f L<sub>emma</sub> 5<sub>.</sub>5 <sub>g</sub>i<sub>ves</sub>

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 2 } ^ { T } \left\| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } - \big ( \langle \pmb { u } ^ { ( t ) } , \pmb { x } ^ { ( t ) } \rangle - \langle \pmb { u } ^ { ( t - 1 ) } , \pmb { x } ^ { ( t - 1 ) } \rangle \big ) \pmb { 1 } \right\| _ { 2 } ^ { 2 } } \\ & { \leq 2 ( 1 + \sqrt { d } ) ^ { 2 } \displaystyle \sum _ { t = 2 } ^ { T } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + 2 d \| \pmb { \mathrm { A } } \| _ { 2 } ^ { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \| \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } \| _ { 2 } ^ { 2 } . } \end{array}
$$

Fi<sub>x</sub> $\varepsilon > 0$ . A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Youn<sub>g</sub>’s inequalit<sub>y</sub> to the <sub>p</sub>roduct in (42), and usin<sub>g</sub> the initialization bound, we obtain, for <sup>ever</sup>y $2 \leq s \leq T + 1$

$$
\begin{array} { r l } & { \| \tilde { r } ^ { ( s ) } \| _ { 2 } ^ { 2 } = \| \tilde { r } ^ { ( 2 ) } \| _ { 2 } ^ { 2 } + \displaystyle \sum _ { t = 2 } ^ { s - 1 } \left( \| \tilde { r } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } \right) } \\ & { \qquad \leq \| { \mathbf { A } } \| _ { 2 } ^ { 2 } + \left( 1 + \frac { 1 } { 4 \delta } \right) \displaystyle \sum _ { i = 2 } ^ { s - 1 } \left\| { \mathbf { u } } ^ { ( t ) } - { \mathbf { u } } ^ { ( t - 1 ) } - \left( \left. { \mathbf { u } } ^ { ( t ) } , { \mathbf { x } } ^ { ( t ) } \right. - \left. { \mathbf { u } } ^ { ( t - 1 ) } , { \mathbf { x } } ^ { ( t - 1 ) } \right. \right) { \mathbf { 1 } } \right\| _ { 2 } ^ { 2 } + \varepsilon \displaystyle \sum _ { t = 2 } ^ { s - 1 } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq \| { \mathbf { A } } \| _ { 2 } ^ { 2 } + \left( 2 + \frac { 1 } { 2 \delta } \right) \left[ \left( 1 + \sqrt { d } \right) ^ { 2 } \displaystyle \sum _ { t = 2 } ^ { T } \| { \mathbf { u } } ^ { ( t ) } - { \mathbf { u } } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + d \| { \mathbf { A } } \| _ { 2 } ^ { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \| { \mathbf { x } } ^ { ( t + 1 ) } - { \mathbf { x } } ^ { ( t ) } \| _ { 2 } ^ { 2 } \right] + \varepsilon \displaystyle \sum _ { t = 2 } ^ { T } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } . } \end{array}
$$

Th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> b<sub>y</sub> <sub>ex</sub>t<sub>en</sub>di<sub>ng</sub> th<sub>e</sub> <sub>nonnega</sub>ti<sub>ve</sub> <sub>sums</sub> t<sub>o</sub> � <sub>an</sub>d <sub>app</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> <sub>prece</sub>di<sub>ng</sub> <sub>u</sub>tilit<sub>y-var</sub>i<sub>a</sub>ti<sub>on</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e.</sub> Si<sub>nce</sub> $\tilde { r } ^ { ( 1 ) } = \mathbf { 0 }$ <sub>,</sub> t<sub>a</sub>ki<sub>ng</sub> th<sub>e</sub> <sub>max</sub>i<sub>mum</sub> <sub>over</sub> $1 \leq s \leq T + 1$ <sub>proves</sub> th<sub>e</sub> <sub>c</sub>l<sub>a</sub>i<sub>m.</sub> □

Proposition 5.13 (Alt-RVU for Alt-HPRM<sup>+</sup>). Set $\gamma = \sqrt { f ^ { ( t _ { 0 } ) } } / ( 6 4 \sqrt { 2 } ( 1 + \sqrt { d } ) ^ { 2 } ) > 0 .$ . For every $\beta > 0 ,$ , there exists $\alpha > 0$ so that Alt-HPRM<sup>+</sup> satisfies the $A l t { - } R V U$ bound. Therefore, in every finite two-player zero-sum normal-form game, alternating self-play with $H P R M ^ { + }$ has ${ \cal O } ( T ^ { - 1 } )$ average-iterate convergence.

Proof. $\mathrm { ~ I f ~ } t _ { 0 }$ d<sub>oes no</sub>t <sub>ex</sub>i<sub>s</sub>t<sub>,</sub> th<sub>e</sub> i<sub>ns</sub>t<sub>an</sub>t<sub>aneous regre</sub>t <sub>vec</sub>t<sub>ors an</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>egy movemen</sub>t<sub>s are zero, so</sub> th<sub>e conc</sub>l<sub>us</sub>i<sub>on</sub> h<sub>o</sub>ld<sub>s</sub> <sub>w</sub>ith <sub>any pos</sub>iti<sub>ve cons</sub>t<sub>an</sub>t<sub>s.</sub> Oth<sub>erw</sub>i<sub>se,</sub> $f ^ { ( t _ { 0 } ) } = \lVert \tilde { r } ^ { ( t _ { 0 } ) } \rVert _ { > } ^ { 2 } / 2 > 0$ <sub>.</sub> It <sub>su</sub>fi<sub>ces</sub> t<sub>o cons</sub>id<sub>er</sub> $T \geq t _ { 0 } - 1$ <sub>, s</sub>i<sub>nce a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>regre</sub>t <sub>an</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>egy movemen</sub>t <sub>are zero</sub> b<sub>e</sub>f<sub>ore</sub> th<sub>a</sub>t ti<sub>me.</sub>

We first bound the <sub>p</sub>rediction norms from below. B<sub>y</sub> (38), for ever<sub>y</sub> $t \geq t _ { 0 }$

$$
f ^ { ( t ) } = \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } \le \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - 2 \| [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } = 2 \| [ r ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } - \| \tilde { r } ^ { ( t ) } - 2 [ r ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } \le 2 \| [ r ^ { ( t ) } ] _ { + } \| _ { 1 } ^ { 2 } .
$$

Si<sub>nce</sub> $f ^ { ( t ) }$ i<sub>s non</sub>d<sub>ecreas</sub>i<sub>ng, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\| [ \boldsymbol { r } ^ { ( t ) } ] _ { + } \| _ { 1 } \ge \sqrt { \frac { f ^ { ( t ) } } { 2 } } \ge \sqrt { \frac { f ^ { ( t _ { 0 } ) } } { 2 } } > 0 .\tag{43}
$$

C<sub>onsequen</sub>tl<sub>y,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { f ^ { ( t + 1 ) } - f ^ { ( t ) } } { 2 \lVert [ r ^ { ( t + 1 ) } ] _ { + } \rVert _ { 1 } } \le \frac { 1 } { \sqrt { 2 } } \sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { f ^ { ( t + 1 ) } - f ^ { ( t ) } } { \sqrt { f ^ { ( t + 1 ) } } } \le \sqrt { 2 } \sum _ { t = t _ { 0 } - 1 } ^ { T } \left( \sqrt { f ^ { ( t + 1 ) } } - \sqrt { f ^ { ( t ) } } \right) = \sqrt { 2 f ^ { ( T + 1 ) } } \le \sqrt { 2 } \lVert \tilde { r } ^ { ( T + 1 ) } \rVert _ { 2 } .
$$

Combinin<sub>g</sub> the utilit<sub>y</sub>-im<sub>p</sub>rovement bound (22) with (12), we therefore <sub>g</sub>et

$$
\begin{array} { r } { \mathrm { A l t - R e g } ( T ) \le ( 2 + \sqrt { 2 } ) \| \tilde { r } ^ { ( T + 1 ) } \| _ { 2 } - \displaystyle \frac { 1 } { 4 } \sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } } \\ { \le 4 \operatorname* { m a x } _ { 1 \le s \le T + 1 } \| \tilde { r } ^ { ( s ) } \| _ { 2 } - \displaystyle \frac { 1 } { 4 } \sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } . } \end{array}\tag{44}
$$

W<sub>e</sub> <sub>nex</sub>t <sub>re</sub>l<sub>a</sub>t<sub>e</sub> th<sub>e</sub> <sub>nega</sub>ti<sub>ve</sub> <sub>sum</sub> t<sub>o</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>movemen</sub>t<sub>.</sub> Th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> id<sub>en</sub>tit<sub>y</sub> f<sub>rom</sub> L<sub>emma</sub> 5<sub>.</sub>10 <sub>g</sub>i<sub>ves</sub>

$$
\begin{array} { r l r } {  { \| [ \boldsymbol { r } ^ { ( t + 1 ) } ] _ { + } - [ \boldsymbol { r } ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } \leq \| \frac { 3 } { 2 } ( \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } ) - \frac { 1 } { 2 } ( \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } ) \| _ { 2 } ^ { 2 } } } \\ & { } & { \leq 3 \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } . ~ } \end{array}
$$

Usin<sub>g</sub> the normalization bound (35) and the <sub>p</sub>rediction norm bound (43), for $t \geq t _ { 0 }$ <sub>we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\sqrt { \frac { f ^ { ( t _ { 0 } ) } } { 2 } } \| \boldsymbol { x } ^ { ( t + 1 ) } - \boldsymbol { x } ^ { ( t ) } \| _ { 2 } ^ { 2 } \leq ( 1 + \sqrt { d } ) ^ { 2 } \left( \frac { 3 \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } + \frac { \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } } { \| [ r ^ { ( t ) } ] _ { + } \| _ { 1 } } \right) .
$$

S<sub>umm</sub>i<sub>ng</sub> <sub>an</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>o</sub>f $\gamma _ { : }$ , w<sup>e</sup> g<sup>et</sup>

$$
\begin{array} { r l } { \displaystyle \sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| \left[ r ^ { ( t + 1 ) } \right] _ { + } \| _ { 1 } } \geq \frac { \sqrt { f ^ { ( t _ { 0 } ) } } } { 4 \sqrt { 2 } ( 1 + \sqrt { d } ) ^ { 2 } } \displaystyle \sum _ { t = t _ { 0 } } ^ { T } \| { \boldsymbol x } ^ { ( t + 1 ) } - { \boldsymbol x } ^ { ( t ) } \| _ { 2 } ^ { 2 } \qquad } & { } \\ { \quad \quad } & { = 1 6 \gamma \displaystyle \sum _ { t = t _ { 0 } } ^ { T } \| { \boldsymbol x } ^ { ( t + 1 ) } - { \boldsymbol x } ^ { ( t ) } \| _ { 2 } ^ { 2 } } \\ { \quad \quad } & { \geq 1 6 \gamma \displaystyle \sum _ { t = 1 } ^ { T } \| { \boldsymbol x } ^ { ( t + 1 ) } - { \boldsymbol x } ^ { ( t ) } \| _ { 2 } ^ { 2 } - 3 2 \gamma , } \end{array}\tag{45}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y uses</sub> th<sub>a</sub>t <sub>movemen</sub>t<sub>s</sub> b<sub>e</sub>f<sub>ore</sub> $t _ { 0 } - 1$ <sub>van</sub>i<sub>s</sub>h <sub>an</sub>d $\| \pmb { x } ^ { ( t _ { 0 } ) } - \pmb { x } ^ { ( t _ { 0 } - 1 ) } \| _ { 2 } ^ { 2 } \leq 2$

It remains to absorb the <sub>p</sub>ositive maximum norm in (44). Since $\tilde { r } ^ { ( t ) } \ge \mathbf { 0 } ,$ <sub>,</sub> th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> id<sub>en</sub>tit<sub>y</sub> <sub>a</sub>l<sub>so</sub> <sub>g</sub>i<sub>ves</sub>

$$
[ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \le \frac { 3 } { 2 } \tilde { { \pmb r } } ^ { ( t + 1 ) } , \ \lVert [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \rVert _ { 1 } \le 2 \sqrt { d } \operatorname* { m a x } _ { 1 \le s \le T + 1 } \lVert \tilde { { \pmb r } } ^ { ( s ) } \rVert _ { 2 } .
$$

Th<sub>ere</sub>f<sub>ore,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\sum _ { t = 2 } ^ { T } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } \leq 2 \sqrt { d } \left( \operatorname* { m a x } _ { 1 \leq s \leq T + 1 } \| \tilde { r } ^ { ( s ) } \| _ { 2 } \right) \sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } .
$$

Substitutin<sub>g</sub> this bound into (23) with $\varepsilon = 1 / ( 6 4 { \sqrt { d } } )$ <sub>an</sub>d <sub>app</sub>l<sub>y</sub>i<sub>ng</sub> Y<sub>oung</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } & { \underset { 1 \leq s \leq T + 1 } { \operatorname* { m a x } } \Vert \tilde { r } ^ { ( s ) } \Vert _ { 2 } ^ { 2 } \leq \Vert \Delta \Vert _ { 2 } ^ { 2 } + ( 1 + 1 6 \sqrt { d } ) ( 2 ( 1 + \sqrt { d } ) ^ { 2 } \underset { t = 0 , - 1 } { \overset { T } { \sum } } \Vert u ^ { ( t ) } - u ^ { ( t - 1 ) } \Vert _ { 2 } ^ { 2 } + 2 d \Vert \mathbf { A } \Vert _ { 2 } ^ { 2 } \underset { t = 1 } { \overset { T } { \sum } } \Vert { \mathbf { x } } ^ { ( t + 1 ) } - { \mathbf { x } } ^ { ( t ) } \Vert _ { 2 } ^ { 2 } ) } \\ & { \qquad + \displaystyle \frac { 1 } { 3 2 } ( \underset { 1 \leq s \leq T + 1 } { \operatorname* { m a x } } \Vert \tilde { r } ^ { ( s ) } \Vert _ { 2 } ) \underset { t = \theta _ { 0 } - 1 } { \overset { T } { \sum } } \frac { \Vert \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \Vert _ { 2 } ^ { 2 } } { \Vert [ r ^ { ( t + 1 ) } ] _ { + } \Vert \Vert _ { 1 } } } \\ & { \qquad \leq \Vert \mathbf { A } \Vert _ { 2 } ^ { 2 } + ( 1 + 1 6 \sqrt { d } ) ( 2 ( 1 + \sqrt { d } ) ^ { 2 } \underset { t = 2 } { \overset { T } { \sum } } \Vert u ^ { ( t ) } - u ^ { ( t - 1 ) } \Vert _ { 2 } ^ { 2 } + 2 d \Vert \mathbf { A } \Vert _ { 2 } ^ { 2 } \underset { t = 1 } { \overset { T } { \sum } } \Vert { \mathbf { x } } ^ { ( t + 1 ) } - { \mathbf { x } } ^ { ( t ) } \Vert _ { 2 } ^ { 2 } ) } \\ &  \qquad + \displaystyle \frac { 1 } { 2 } \underset { 1 \leq s \leq T + 1 } { \operatorname* { m a x } } \Vert \tilde { r } ^ { ( s ) } \Vert _ { 2 } ^ { 2 } + \frac { 1 } { 2 0 4 8 } ( \underset { t = t - 1 } \end{array}
$$

R<sub>ea</sub>rr<sub>a</sub>n<sub>g</sub>in<sub>g</sub> <sub>a</sub>nd t<sub>a</sub>kin<sub>g</sub> <sub>squa</sub>r<sub>e</sub> r<sub>oo</sub>t<sub>s,</sub> <sub>us</sub>in<sub>g</sub> $\sqrt { 2 + 3 2 \sqrt { d } } \leq 6 d ^ { 1 / 4 }$ , w<sup>e</sup> g<sup>et</sup>

$$
\begin{array} { l } { \displaystyle \underset { 1 \leq s \leq T + 1 } { \operatorname* { m a x } } \| \tilde { r } ^ { ( s ) } \| _ { 2 } \leq 2 \| \mathbf { A } \| _ { 2 } + 6 d ^ { 1 / 4 } \left( 2 ( 1 + \sqrt { d } ) ^ { 2 } \displaystyle \sum _ { t = 2 } ^ { T } \| { \boldsymbol u } ^ { ( t ) } - { \boldsymbol u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + 2 d \| \mathbf { A } \| _ { 2 } ^ { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \| { \boldsymbol x } ^ { ( t + 1 ) } - { \boldsymbol x } ^ { ( t ) } \| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } } \\ { + \displaystyle \frac { 1 } { 3 2 } \displaystyle \sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } . } \end{array}
$$

Substitutin<sub>g</sub> into (44) and then a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> (45), we obtain

$$
\begin{array} { r l } & { \begin{array} { r l } & { \displaystyle \mathrm { A l t . R e g } ( T ) \leq 8 \| { \mathbf { A } } \| _ { 2 } + 4 \gamma + 2 4 d ^ { l / 4 } \left( 2 ( 1 + \sqrt { d } ) ^ { 2 } \displaystyle \sum _ { t = 2 } ^ { T } \| { \boldsymbol u } ^ { ( t ) } - { \boldsymbol u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + 2 d \| { \mathbf { A } } \| _ { 2 } ^ { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \| { \boldsymbol x } ^ { ( t + 1 ) } - { \boldsymbol x } ^ { ( t ) } \| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } } \\ & { \qquad - 2 \gamma \displaystyle \sum _ { t = 1 } ^ { T } \| { \boldsymbol x } ^ { ( t + 1 ) } - { \boldsymbol x } ^ { ( t ) } \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq 8 \| { \mathbf { A } } \| _ { 2 } + 4 \gamma + \displaystyle \frac { 2 8 8 \sqrt { d } ( 1 + \sqrt { d } ) ^ { 2 } } { \beta } + \frac { 2 8 8 d ^ { 3 / 2 } \| { \mathbf { A } } \| _ { 2 } ^ { 2 } } { \gamma } } \\ & { \qquad + \beta \displaystyle \sum _ { t = 2 } ^ { T } \| { \boldsymbol u } ^ { ( t ) } - { \boldsymbol u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } - \gamma \displaystyle \sum _ { t = 1 } ^ { T } \| { \boldsymbol x } ^ { ( t + 1 ) } - { \boldsymbol x } ^ { ( t ) } \| _ { 2 } ^ { 2 } , } \end{array} } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> b<sub>y sp</sub>litti<sub>ng</sub> th<sub>e square roo</sub>t <sub>an</sub>d <sub>app</sub>l<sub>y</sub>i<sub>ng</sub> Y<sub>oung</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> t<sub>o eac</sub>h t<sub>erm.</sub> Th<sub>us, one</sub> <sub>a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e � c</sub>h<sub>o</sub>i<sub>ce</sub> i<sub>s</sub>

$$
\alpha = 8 \| \mathbf { A } \| _ { 2 } + 4 \gamma + \frac { 2 8 8 \sqrt { d } ( 1 + \sqrt { d } ) ^ { 2 } } { \beta } + \frac { 2 8 8 d ^ { 3 / 2 } \| \mathbf { A } \| _ { 2 } ^ { 2 } } { \gamma } ,
$$

which is inde<sub>p</sub>endent of �. We hence obtain the desired Alt-RVU bound for HPRM<sup>+</sup>.

## A.5 Proofs from Subsection 5.4

Proposition 5.14. There is a utility sequence $( \pmb { u } ^ { ( t ) } ) _ { t \geq 1 } \subset [ - 1 , 1 ] ^ { 2 }$ , which can be generated by an opponent under alternating updates, on which a HPRM<sup>+</sup> learner satisfies,for every $T \geq 1$

$$
0 \leq \mathrm { R e g } ( T ) \leq 3 , \sum _ { t = 2 } ^ { \infty } \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } < \infty , \sum _ { t = 2 } ^ { \infty } \| { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } = \infty .\tag{24}
$$

Consequently, the standard RVU bound fails on this sequence for every choice of horizon-independent positive constants, even when they may depend on the entire utility sequence.

Proof. We construct a sequence whose standard regret and squared utility variation remain bounded, while its squared <sub>s</sub>t<sub>ra</sub>t<sub>egy movemen</sub>t di<sub>verges.</sub> W<sub>e use</sub> th<sub>e same</sub> bl<sub>oc</sub>k<sub>s as</sub> i<sub>n</sub> P<sub>ropos</sub>iti<sub>on</sub> 6<sub>.</sub>1<sub>,</sub> b<sub>u</sub>t i<sub>ncrease</sub> th<sub>e</sub>i<sub>r</sub> l<sub>eng</sub>th<sub>s an</sub>d d<sub>ecrease</sub> th<sub>e</sub>i<sub>r u</sub>tilit<sub>y coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s.</sub> Thi<sub>s w</sub>ill k<sub>eep</sub> th<sub>e accumu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>norm</sub> i<sub>ncrease</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>w</sub>hil<sub>e a</sub>ll<sub>ow</sub>i<sub>ng eac</sub>h bl<sub>oc</sub>k t<sub>o pro</sub>d<sub>uce</sub> a nonvan<sup>i</sup>s<sup>hi</sup>n<sub>g</sub> strate<sub>gy</sub> c<sup>h</sup>an<sub>g</sub>e.

U<sub>se</sub> th<sub>e same</sub> th<sub>ree</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>u</sub>tiliti<sub>es</sub> $\pmb { u } ^ { ( 1 ) } = ( 1 , - 1 ) , \pmb { u } ^ { ( 2 ) } = ( 0 , 1 )$ <sub>,</sub> <sub>an</sub>d ${ \pmb u } ^ { ( 3 ) } = { \pmb 0 } .$ <sub>,</sub> <sub>so</sub> th<sub>a</sub>t $\tilde { r } ^ { ( 4 ) } = r ^ { ( 4 ) } = ( 1 , 1 )$ <sub>.</sub> F<sub>or</sub> bl<sub>oc</sub>k $k \geq 1$ <sub>,</sub> <sub>c</sub>h<sub>oose</sub>

$$
N _ { k } = k + 2 , \varepsilon _ { k } = \frac { 1 } { 1 2 N _ { k } } .
$$

Th<sub>e</sub> bl<sub>oc</sub>k <sub>cons</sub>i<sub>s</sub>t<sub>s o</sub>f $N _ { k } - 1$ <sub>u</sub>tiliti<sub>es</sub> $\varepsilon _ { k } ( - r _ { 2 } ^ { ( t ) } , r _ { 1 } ^ { ( t ) } )$ <sub>, one zero u</sub>tilit<sub>y,</sub> $N _ { k } - 1$ <sub>u</sub>tiliti<sub>es</sub> $\varepsilon _ { k } ( r _ { 2 } ^ { ( t ) } , - r _ { 1 } ^ { ( t ) } )$ <sub>, an</sub>d <sub>one zero</sub> <sub>u</sub>tilit<sub>y.</sub>

Th<sub>e pos</sub>iti<sub>v</sub>it<sub>y argumen</sub>t <sub>an</sub>d <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e proo</sub>f <sub>o</sub>f P<sub>ropos</sub>iti<sub>on</sub> 6<sub>.</sub>1 <sub>rema</sub>i<sub>n va</sub>lid <sub>a</sub>t $\lambda = 1 / 2 \colon$ t<sup>h</sup>e<sub>y</sub> re<sub>q</sub>u<sup>i</sup>re $N _ { k } \ge 3$ <sub>an</sub>d $\varepsilon _ { k } = 1 / [ 8 N _ { k } ( 1 + \lambda ) ]$ <sub>,</sub> b<sub>u</sub>t d<sub>o no</sub>t <sub>requ</sub>i<sub>re</sub> th<sub>e s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>norm</sub> d<sub>ecrease.</sub> Th<sub>us,</sub> if bl<sub>oc</sub>k � <sub>s</sub>t<sub>ar</sub>t<sub>s a</sub>t ti<sub>me</sub> � <sub>w</sub>ith $\tilde { r } ^ { ( t ) } = r ^ { ( t ) } = R _ { k } ( 1 , 1 )$ <sub>, every coor</sub>di<sub>na</sub>t<sub>e o</sub>f b<sub>o</sub>th <sub>vec</sub>t<sub>ors rema</sub>i<sub>ns</sub> i<sub>n</sub> $[ R _ { k } / 2 , 3 R _ { k } / 2 ]$ <sub>, an</sub>d th<sub>e</sub> bl<sub>oc</sub>k <sub>en</sub>d<sub>s w</sub>ith

$$
\tilde { r } ^ { ( t + 2 N _ { k } ) } = r ^ { ( t + 2 N _ { k } ) } = R _ { k + 1 } ( 1 , 1 ) , R _ { k + 1 } = \rho _ { k } R _ { k } ,
$$

<sub>w</sub>h<sub>ere</sub> $R _ { 1 } = 1$ <sub>an</sub>d $\rho _ { k }$ i<sub>s</sub> th<sub>e</sub> f<sub>orwar</sub>d h<sub>a</sub>lf’<sub>s square</sub>d<sub>-norm ra</sub>ti<sub>o.</sub> Th<sub>roug</sub>h<sub>ou</sub>t th<sub>e</sub> bl<sub>oc</sub>k<sub>, c</sub>li<sub>pp</sub>i<sub>ng</sub> i<sub>s</sub> i<sub>nac</sub>ti<sub>ve an</sub>d

$$
\langle { \pmb u } ^ { ( s ) } , { \pmb x } ^ { ( s ) } \rangle = 0 , \tilde { r } ^ { ( s + 1 ) } = \tilde { r } ^ { ( s ) } + { \pmb u } ^ { ( s ) } , r ^ { ( s + 1 ) } = \tilde { r } ^ { ( s + 1 ) } + \frac { 1 } { 2 } { \pmb u } ^ { ( s ) } .
$$

W<sub>e</sub> fi<sub>rs</sub>t b<sub>oun</sub>d th<sub>e</sub> <sub>regre</sub>t<sub>-vec</sub>t<sub>or</sub> <sub>norms.</sub> S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> $\lambda = 1 / 2$ into (51), the utilit<sub>y</sub>-square term vanishes. Usin<sub>g</sub> the forward-half variation estimate (53), scaled b<sub>y</sub> $R _ { k } ^ { 2 } ,$ <sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
0 \le \rho _ { k } - 1 = \frac { \| \widetilde { \mathbfcal { r } } ^ { ( t + N _ { k } ) } \| _ { 2 } ^ { 2 } - 2 R _ { k } ^ { 2 } } { 2 R _ { k } ^ { 2 } } = \frac { 1 } { 4 R _ { k } ^ { 2 } } \sum _ { s = t } ^ { t + N _ { k } - 1 } \| \pmb { u } ^ { ( s ) } - \pmb { u } ^ { ( s - 1 ) } \| _ { 2 } ^ { 2 } < \frac { 7 } { 4 } \pmb { \varepsilon } _ { k } ^ { 2 } .
$$

<sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>,

$$
1 \leq R _ { k } = \prod _ { j = 1 } ^ { k - 1 } \rho _ { j } \leq \exp \left( \frac { 7 } { 4 } \sum _ { j = 1 } ^ { \infty } \varepsilon _ { j } ^ { 2 } \right) = \exp \left( \frac { 7 } { 5 7 6 } \sum _ { n = 3 } ^ { \infty } \frac { 1 } { n ^ { 2 } } \right) \leq \exp \left( \frac { 7 } { 1 1 5 2 } \right) < 2 ,
$$

<sub>w</sub>h<sub>ere</sub> <sub>we</sub> <sub>use</sub> $\begin{array} { r } { \sum _ { n = 3 } ^ { \infty } n ^ { - 2 } \leq \sum _ { n = 3 } ^ { \infty } [ n ( n - 1 ) ] ^ { - 1 } = 1 / 2 } \end{array}$ <sub>.</sub> I<sub>n par</sub>ti<sub>cu</sub>l<sub>ar, every u</sub>tilit<sub>y a</sub>ft<sub>er</sub> i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\| \pmb { u } ^ { ( s ) } \| _ { \infty } \le \frac { 3 } { 2 } R _ { k } \varepsilon _ { k } < 3 \varepsilon _ { k } \le \frac { 1 } { 1 2 } .
$$

T<sub>oge</sub>th<sub>er w</sub>ith th<sub>e</sub> th<sub>ree</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>u</sub>tiliti<sub>es,</sub> thi<sub>s proves</sub> th<sub>a</sub>t th<sub>e en</sub>ti<sub>re sequence</sub> li<sub>es</sub> i<sub>n</sub> $[ - 1 , 1 ] ^ { 2 }$

W<sub>e nex</sub>t b<sub>oun</sub>d <sub>square</sub>d <sub>u</sub>tilit<sub>y var</sub>i<sub>a</sub>ti<sub>on.</sub> E<sub>ac</sub>h bl<sub>oc</sub>k h<sub>as</sub> f<sub>our c</sub>h<sub>anges</sub> t<sub>o or</sub> f<sub>rom zero, eac</sub>h <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ng a</sub>t <sub>mos</sub>t $9 R _ { k } ^ { 2 } \varepsilon _ { k } ^ { 2 } / 2$ <sub>.</sub> F<sub>or</sub> t<sub>wo consecu</sub>ti<sub>ve nonzero u</sub>tiliti<sub>es w</sub>ithi<sub>n</sub> th<sub>e same</sub> h<sub>a</sub>lf<sub>,</sub> th<sub>e up</sub>d<sub>a</sub>t<sub>e</sub> id<sub>en</sub>tit<sub>y g</sub>i<sub>ves</sub>

$$
\| \pmb { u } ^ { ( s ) } - \pmb { u } ^ { ( s - 1 ) } \| _ { 2 } = \varepsilon _ { k } \| \pmb { r } ^ { ( s ) } - \pmb { r } ^ { ( s - 1 ) } \| _ { 2 } = \varepsilon _ { k } \left\| \frac { 3 } { 2 } \pmb { u } ^ { ( s - 1 ) } - \frac { 1 } { 2 } \pmb { u } ^ { ( s - 2 ) } \right\| _ { 2 } \le 3 \sqrt { 2 } R _ { k } \varepsilon _ { k } ^ { 2 } .
$$

Th<sub>ere</sub> <sub>are</sub> $2 ( N _ { k } - 2 )$ <sub>suc</sub>h i<sub>n</sub>t<sub>er</sub>i<sub>or c</sub>h<sub>anges.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\sum _ { s = t } ^ { t + 2 N _ { k } - 1 } \| { \boldsymbol u } ^ { ( s ) } - { \boldsymbol u } ^ { ( s - 1 ) } \| _ { 2 } ^ { 2 } \le 1 8 R _ { k } ^ { 2 } \varepsilon _ { k } ^ { 2 } + 3 6 ( N _ { k } - 2 ) R _ { k } ^ { 2 } \varepsilon _ { k } ^ { 4 } = R _ { k } ^ { 2 } \varepsilon _ { k } ^ { 2 } \left( 1 8 + \frac { N _ { k } - 2 } { 4 N _ { k } ^ { 2 } } \right) < 7 6 \varepsilon _ { k } ^ { 2 } .
$$

Thi<sub>s sum</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>es</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>c</sub>h<sub>ange</sub> f<sub>rom</sub> th<sub>e prece</sub>di<sub>ng zero u</sub>tilit<sub>y.</sub> Th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> $\| \pmb { u } ^ { ( 2 ) } - \pmb { u } ^ { ( 1 ) } \| _ { 2 } ^ { 2 } +$ $\| \pmb { u } ^ { ( 3 ) } - \pmb { u } ^ { ( 2 ) } \| _ { 2 } ^ { 2 } = 6$ , so

$$
\sum _ { s = 2 } ^ { \infty } \| \pmb { u } ^ { ( s ) } - \pmb { u } ^ { ( s - 1 ) } \| _ { 2 } ^ { 2 } \le 6 + 7 6 \sum _ { k = 1 } ^ { \infty } \varepsilon _ { k } ^ { 2 } < \infty .
$$

W<sub>e now s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>eac</sub>h f<sub>orwar</sub>d h<sub>a</sub>lf <sub>c</sub>h<sub>anges</sub> th<sub>e s</sub>t<sub>ra</sub>t<sub>egy</sub> b<sub>y an amoun</sub>t b<sub>oun</sub>d<sub>e</sub>d <sub>away</sub> f<sub>rom zero.</sub> At th<sub>e</sub> b<sub>eg</sub>i<sub>nn</sub>i<sub>ng</sub> <sub>o</sub>f bl<sub>oc</sub>k �<sub>, we</sub> h<sub>ave</sub> $\pmb { x } ^ { ( t ) } = ( 1 / 2 , 1 / 2 )$ ). Summing its $N _ { k } - 1$ <sub>nonzero</sub> f<sub>orwar</sub>d <sub>u</sub>tiliti<sub>es, we ge</sub>t

$$
\widetilde { r } _ { 2 } ^ { ( t + N _ { k } ) } - \widetilde { r } _ { 1 } ^ { ( t + N _ { k } ) } = \varepsilon _ { k } \sum _ { s = t } ^ { t + N _ { k } - 2 } \left( r _ { 1 } ^ { ( s ) } + r _ { 2 } ^ { ( s ) } \right) \ge R _ { k } ( N _ { k } - 1 ) \varepsilon _ { k } .
$$

Th<sub>e</sub> fi<sub>na</sub>l <sub>zero</sub> <sub>u</sub>tilit<sub>y</sub> i<sub>n</sub> thi<sub>s</sub> h<sub>a</sub>lf <sub>g</sub>i<sub>ves</sub> $\boldsymbol { r } ^ { ( t + N _ { k } ) } = \tilde { \boldsymbol { r } } ^ { ( t + N _ { k } ) }$ <sub>.</sub> Si<sub>nce</sub> it<sub>s</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>sum</sub> i<sub>s</sub> <sub>a</sub>t <sub>mos</sub>t $3 R _ { k }$ <sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\left| x _ { 1 } ^ { ( t + N _ { k } ) } - \frac { 1 } { 2 } \right| = \frac { \tilde { r } _ { 2 } ^ { ( t + N _ { k } ) } - \tilde { r } _ { 1 } ^ { ( t + N _ { k } ) } } { 2 \| \tilde { r } ^ { ( t + N _ { k } ) } \| _ { 1 } } \geq \frac { ( N _ { k } - 1 ) \varepsilon _ { k } } { 6 } \geq \frac { 1 } { 1 0 8 } .
$$

A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> C<sub>auc</sub>h<sub>y–</sub>S<sub>c</sub>h<sub>wa</sub>r<sub>z</sub> t<sub>o</sub> th<sub>ese</sub> $N _ { k }$ <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>c</sub>h<sub>anges,</sub> <sub>we</sub> f<sub>ur</sub>th<sub>er</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\sum _ { s = t } ^ { t + N _ { k } - 1 } \| { \pmb x } ^ { ( s + 1 ) } - { \pmb x } ^ { ( s ) } \| _ { 2 } ^ { 2 } \geq \frac { \| { \pmb x } ^ { ( t + N _ { k } ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { N _ { k } } = \frac { 2 } { N _ { k } } \left| x _ { 1 } ^ { ( t + N _ { k } ) } - \frac { 1 } { 2 } \right| ^ { 2 } \geq \frac { 1 } { 5 8 3 2 N _ { k } } .
$$

The forward halves are disjoint, and $\textstyle \sum _ { k = 1 } ^ { \infty } 1 / N _ { k } = \infty$ <sub>.</sub> H<sub>ence,</sub>

$$
\sum _ { s = 2 } ^ { \infty } \| \pmb { x } ^ { ( s ) } - \pmb { x } ^ { ( s - 1 ) } \| _ { 2 } ^ { 2 } = \infty .
$$

It <sub>rema</sub>i<sub>ns</sub> t<sub>o</sub> b<sub>oun</sub>d <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>regre</sub>t<sub>.</sub> E<sub>very</sub> <sub>p</sub>l<sub>aye</sub>d <sub>payo</sub>f i<sub>s</sub> <sub>zero,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> d<sub>ur</sub>i<sub>ng</sub> i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on.</sub> M<sub>oreover,</sub> th<sub>e</sub> <sub>regre</sub>t<sub>-vec</sub>t<sub>or</sub> <sub>up</sub>d<sub>a</sub>t<sub>es</sub> t<sub>e</sub>l<sub>escope</sub> f<sub>rom</sub> ti<sub>me</sub> 4 <sub>onwar</sub>d<sub>.</sub> Th<sub>us,</sub> f<sub>or</sub> <sub>every</sub> $T \geq 3$

$$
\sum _ { s = 1 } ^ { T } \pmb { u } ^ { ( s ) } = ( 1 , 0 ) + \sum _ { s = 4 } ^ { T } \left( \tilde { r } ^ { ( s + 1 ) } - \tilde { r } ^ { ( s ) } \right) = \tilde { r } ^ { ( T + 1 ) } - ( 0 , 1 ) ,
$$

<sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub>

$$
\mathrm { R e g } ( T ) = \operatorname * { m a x } \left\{ \tilde { r } _ { 1 } ^ { ( T + 1 ) } , \tilde { r } _ { 2 } ^ { ( T + 1 ) } - 1 \right\} .
$$

P<sub>os</sub>iti<sub>v</sub>it<sub>y</sub> <sub>an</sub>d th<sub>e</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> b<sub>oun</sub>d<sub>s</sub> i<sub>mp</sub>l<sub>y</sub> $0 \leq \mathrm { R e g } ( T ) \leq 3$ <sub>.</sub> Th<sub>e</sub> fi<sub>rs</sub>t t<sub>wo</sub> h<sub>or</sub>i<sub>zons</sub> b<sub>o</sub>th h<sub>ave</sub> <sub>regre</sub>t 1<sub>.</sub>

F<sub>or any</sub> h<sub>or</sub>i<sub>zon-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>cons</sub>t<sub>an</sub>t<sub>s</sub> $\alpha , \beta , \gamma > 0$ <sub>, we</sub> h<sub>ave</sub>

$$
\alpha + \beta \sum _ { s = 2 } ^ { T } \| \pmb { u } ^ { ( s ) } - \pmb { u } ^ { ( s - 1 ) } \| _ { 2 } ^ { 2 } - \gamma \sum _ { s = 2 } ^ { T } \| \pmb { x } ^ { ( s ) } - \pmb { x } ^ { ( s - 1 ) } \| _ { 2 } ^ { 2 } \longrightarrow - \infty ,
$$

<sub>w</sub>h<sub>ereas</sub> $\begin{array} { r } { \mathrm { R e g } ( T ) \geq 0 . } \end{array}$ <sub>.</sub> Thi<sub>s con</sub>t<sub>ra</sub>di<sub>c</sub>t<sub>s s</sub>t<sub>an</sub>d<sub>ar</sub>d RVU<sub>, even w</sub>h<sub>en</sub> th<sub>e cons</sub>t<sub>an</sub>t<sub>s</sub> d<sub>epen</sub>d <sub>on</sub> th<sub>e en</sub>ti<sub>re sequence.</sub> Th<sub>e</sub> <sub>same conc</sub>l<sub>us</sub>i<sub>on</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or any</sub> fi<sub>xe</sub>d <sub>pa</sub>i<sub>r o</sub>f <sub>pr</sub>i<sub>ma</sub>l<sub>-</sub>d<sub>ua</sub>l <sub>norms</sub> b<sub>y norm equ</sub>i<sub>va</sub>l<sub>ence</sub> i<sub>n</sub> di<sub>mens</sub>i<sub>on</sub> t<sub>wo.</sub> Fi<sub>na</sub>ll<sub>y,</sub> th<sub>e</sub> <sub>common ma</sub>t<sub>r</sub>i<sub>x-game rea</sub>li<sub>za</sub>ti<sub>on o</sub>f <sub>u</sub>tiliti<sub>es</sub> i<sub>n</sub> $[ - 1 , 1 ] ^ { 2 }$ <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t thi<sub>s sequence can</sub> b<sub>e genera</sub>t<sub>e</sub>d b<sub>y an opponen</sub>t <sub>un</sub>d<sub>er a</sub>lt<sub>erna</sub>ti<sub>ng up</sub>d<sub>a</sub>t<sub>es.</sub> □

## A.6 Proofs from Section 6

Proposition 6.1. For every fixed $\lambda > 1 / 2$ , there is a utility sequence $( \pmb { u } ^ { ( t ) } ) _ { t \geq 1 } \subset [ - 1 , 1 ] ^ { 2 } ,$ , on which the scaledprediction regret matching<sup>+</sup> learner satisfies

$$
\sum _ { t = 1 } ^ { \infty } \| { \pmb u } ^ { ( t ) } \| _ { 2 } < \infty , \ \operatorname* { s u p } _ { T \geq 1 } | \mathrm { A l t } \mathrm { - } \mathrm { R e g } ( T ) | < \infty , \ \sum _ { t = 1 } ^ { T } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } = \Theta ( T ) .\tag{25}
$$

Consequently, Alt-RVU fails on this sequence for every choice of horizon-independent constants.

Proof. We work with two actions throughout. Recall the scaled-prediction updates

$$
\tilde { r } ^ { ( t + 1 ) } = \left[ \tilde { r } ^ { ( t ) } + u ^ { ( t ) } - \langle u ^ { ( t ) } , \mathbf { x } ^ { ( t ) } \rangle \mathbf { 1 } \right] _ { + } , r ^ { ( t + 1 ) } = \tilde { r } ^ { ( t + 1 ) } + \lambda \left( u ^ { ( t ) } - \langle u ^ { ( t ) } , \mathbf { x } ^ { ( t ) } \rangle \mathbf { 1 } \right) ,
$$

<sub>w</sub>ith $\pmb { x } ^ { ( t ) } = [ \pmb { r } ^ { ( t ) } ] _ { + } / \Vert [ \pmb { r } ^ { ( t ) } ] _ { + } \Vert$ <sub>1 an</sub>d th<sub>e un</sub>if<sub>orm s</sub>t<sub>ra</sub>t<sub>egy w</sub>h<sub>en</sub> th<sub>e</sub> d<sub>enom</sub>i<sub>na</sub>t<sub>or</sub> i<sub>s zero.</sub>

Construction. Starting from the initialization $\tilde { r } ^ { ( 1 ) } = r ^ { ( 1 ) } = \mathbf { 0 }$ <sub>, we se</sub>t th<sub>e</sub> fi<sub>rs</sub>t th<sub>ree u</sub>tiliti<sub>es</sub>

$$
{ \pmb u } ^ { ( 1 ) } = ( 1 , - 1 ) , { \pmb u } ^ { ( 2 ) } = ( 0 , 1 ) , { \pmb u } ^ { ( 3 ) } = { \bf 0 } .
$$

Th<sub>e</sub> fi<sub>rs</sub>t th<sub>ree o</sub>b<sub>serva</sub>ti<sub>ons are</sub>

$$
\tilde { r } ^ { ( 2 ) } = ( 1 , 0 ) , r ^ { ( 2 ) } = ( 1 + \lambda , - \lambda ) , x ^ { ( 2 ) } = ( 1 , 0 ) ,
$$

<sub>an</sub>d

$$
\tilde { r } ^ { ( 3 ) } = ( 1 , 1 ) , \ r ^ { ( 3 ) } = ( 1 , 1 + \lambda ) , \ x ^ { ( 3 ) } = \left( \frac { 1 } { 2 + \lambda } , \frac { 1 + \lambda } { 2 + \lambda } \right) ,
$$

<sub>an</sub>d

$$
\tilde { r } ^ { ( 4 ) } = r ^ { ( 4 ) } = ( 1 , 1 ) , \ x ^ { ( 4 ) } = \left( \frac { 1 } { 2 } , \frac { 1 } { 2 } \right) .\tag{46}
$$

W<sub>e see</sub>k <sub>a</sub> bl<sub>oc</sub>k th<sub>a</sub>t <sub>re</sub>t<sub>urns</sub> b<sub>o</sub>th <sub>regre</sub>t <sub>vec</sub>t<sub>ors</sub> t<sub>o</sub> thi<sub>s</sub> di<sub>rec</sub>ti<sub>on a</sub>t <sub>a sma</sub>ll<sub>er sca</sub>l<sub>e.</sub> A <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y sma</sub>ll <sub>� w</sub>ill k<sub>eep</sub> <sub>every coor</sub>di<sub>na</sub>t<sub>e pos</sub>iti<sub>ve</sub> th<sub>roug</sub>h<sub>ou</sub>t th<sub>e</sub> bl<sub>oc</sub>k<sub>, w</sub>hil<sub>e su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y many nonzero u</sub>tiliti<sub>es w</sub>ill <sub>ma</sub>k<sub>e</sub> th<sub>e regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>norm</sub> d<sub>ecrease.</sub> W<sub>e ma</sub>k<sub>e</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> fi<sub>xe</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ces:</sub>

$$
N = 1 + \left\lceil \frac { 1 6 \lambda } { 2 \lambda - 1 } \right\rceil , \varepsilon = \frac { 1 } { 8 N ( 1 + \lambda ) } .\tag{47}
$$

W<sub>e now prescr</sub>ib<sub>e</sub> th<sub>e u</sub>tiliti<sub>es</sub>

$$
\pmb { u } ^ { ( t ) } = \left\{ \begin{array} { l l } { \varepsilon ( - r _ { 2 } ^ { ( t ) } , r _ { 1 } ^ { ( t ) } ) , } & { 4 \leq t \leq N + 2 , } \\ { \mathbf { 0 } , } & { t = N + 3 , } \\ { \varepsilon ( r _ { 2 } ^ { ( t ) } , - r _ { 1 } ^ { ( t ) } ) , } & { N + 4 \leq t \leq 2 N + 2 , } \\ { \mathbf { 0 } , } & { t = 2 N + 3 . } \end{array} \right.\tag{48}
$$

E<sub>ac</sub>h h<sub>a</sub>lf <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> $N - 1$ <sub>nonzero u</sub>tiliti<sub>es</sub> f<sub>o</sub>ll<sub>owe</sub>d b<sub>y one zero u</sub>tilit<sub>y, so</sub> th<sub>e</sub> f<sub>u</sub>ll bl<sub>oc</sub>k h<sub>as</sub> 2� <sub>o</sub>b<sub>serva</sub>ti<sub>ons.</sub>

No clipping in a block. First, we prove by induction that every coordinate of both regret vectors lies in [1/2, 3/2] th<sub>roug</sub>h<sub>ou</sub>t th<sub>e</sub> bl<sub>oc</sub>k<sub>, an</sub>d th<sub>a</sub>t th<sub>e ac</sub>t<sub>ua</sub>l <sub>regre</sub>t <sub>vec</sub>t<sub>ors up</sub>d<sub>a</sub>t<sub>es sa</sub>ti<sub>s</sub>f<sub>y</sub>

$$
\tilde { \pmb { r } } ^ { ( t + 1 ) } = \tilde { \pmb { r } } ^ { ( t ) } + \pmb { u } ^ { ( t ) } , \pmb { r } ^ { ( t + 1 ) } = \tilde { \pmb { r } } ^ { ( t + 1 ) } + \lambda \pmb { u } ^ { ( t ) } .\tag{49}
$$

At $t \ = \ 4 .$ , both vectors equal (1, 1), and there are no preceding updates within the block. Fix an iterate � where $4 \leq t \leq 2 N + 3$ . <sup>W</sup>e su<sub>pp</sub>ose t<sup>h</sup>at $\pmb { r } ^ { ( s ) } , \tilde { r } ^ { ( s ) } \in [ 1 / 2 , 3 / 2 ] ^ { \bar { 2 } }$ <sup>f</sup>or ever<sub>y</sub> $4 \leq s \leq t ,$ , and that (49) holds for $4 \leq s < t$ <sup>F</sup>or ever<sub>y</sub> $4 \leq s \leq t ,$ th<sub>e</sub> i<sub>n</sub>d<sub>uc</sub>ti<sub>on</sub> h<sub>ypo</sub>th<sub>es</sub>i<sub>s an</sub>d th<sub>e u</sub>tilit<sub>y ru</sub>l<sub>e g</sub>i<sub>ve</sub>

$$
\langle { \pmb u } ^ { ( s ) } , { \pmb x } ^ { ( s ) } \rangle = \frac { \langle { \pmb u } ^ { ( s ) } , { \pmb r } ^ { ( s ) } \rangle } { \| { \pmb r } ^ { ( s ) } \| _ { 1 } } = 0 , ~ \| { \pmb u } ^ { ( s ) } \| _ { \infty } \leq \varepsilon \operatorname* { m a x } \{ | r _ { 2 } ^ { ( s ) } | , | r _ { 1 } ^ { ( s ) } | \} \leq \frac { 3 } { 2 } \varepsilon .
$$

S<sub>umm</sub>i<sub>ng</sub> th<sub>e prev</sub>i<sub>ous up</sub>d<sub>a</sub>t<sub>e</sub> id<sub>en</sub>titi<sub>es g</sub>i<sub>ves</sub>

$$
\tilde { \pmb { r } } ^ { ( t ) } - ( 1 , 1 ) = \tilde { \pmb { r } } ^ { ( t ) } - \tilde { \pmb { r } } ^ { ( 4 ) } = \sum _ { s = 4 } ^ { t - 1 } \left( \tilde { \pmb { r } } ^ { ( s + 1 ) } - \tilde { \pmb { r } } ^ { ( s ) } \right) = \sum _ { s = 4 } ^ { t - 1 } \pmb { u } ^ { ( s ) } ,
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>mp</sub>li<sub>es</sub>

$$
\tilde { r } ^ { ( t ) } + u ^ { ( t ) } - ( 1 , 1 ) = \sum _ { s = 4 } ^ { t } { \pmb u } ^ { ( s ) } .
$$

Si<sub>nce</sub> th<sub>ere are a</sub>t <sub>mos</sub>t $2 N - 2$ <sub>nonzero u</sub>tiliti<sub>es</sub> i<sub>n</sub> th<sub>e</sub> f<sub>u</sub>ll bl<sub>oc</sub>k<sub>, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\left\| \widetilde { r } ^ { ( t ) } + u ^ { ( t ) } - ( 1 , 1 ) \right\| _ { \infty } = \left\| \sum _ { s = 4 } ^ { t } u ^ { ( s ) } \right\| _ { \infty } \leq \frac 3 2 ( 2 N - 2 ) \varepsilon \leq \frac 3 2 ( 2 N - 2 + \lambda ) \varepsilon \leq \frac 3 8 ,
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> $2 N - 2 + \lambda \leq 2 N ( 1 + \lambda )$ and (47). In <sub>p</sub>articular, $\tilde { \pmb { r } } ^ { ( t ) } + \pmb { u } ^ { ( t ) } \in [ 5 / 8 , 1 1 / 8 ] ^ { 2 }$ , so <sub>c</sub>li<sub>pp</sub>i<sub>ng</sub> i<sub>s</sub> i<sub>nac</sub>ti<sub>ve an</sub>d $\mathbf { \widetilde { r } } ^ { ( t + 1 ) } = \tilde { r } ^ { ( t ) } + \mathbf { u } ^ { ( t ) }$ <sub>.</sub> F<sub>or</sub> th<sub>e su</sub>b<sub>sequen</sub>t <sub>pre</sub>di<sub>c</sub>ti<sub>on regre</sub>t <sub>vec</sub>t<sub>or, we s</sub>i<sub>m</sub>il<sub>ar</sub>l<sub>y ge</sub>t

$$
\left\| r ^ { ( t + 1 ) } - ( 1 , 1 ) \right\| _ { \infty } = \left\| \sum _ { s = 4 } ^ { t } { \pmb u } ^ { ( s ) } + \lambda { \pmb u } ^ { ( t ) } \right\| _ { \infty } \le \frac 3 2 ( 2 N - 2 + \lambda ) \varepsilon \le \frac 3 8 .
$$

C<sub>o</sub>n<sub>seque</sub>ntl<sub>y,</sub> f<sub>o</sub>r $i = 1 , 2$ <sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\frac { 5 } { 8 } \leq \tilde { r } _ { i } ^ { ( t + 1 ) } , r _ { i } ^ { ( t + 1 ) } \leq \frac { 1 1 } { 8 } .
$$

Th<sub>us,</sub> b<sub>o</sub>th <sub>vec</sub>t<sub>ors a</sub>t ti<sub>me</sub> $t + 1$ b<sub>e</sub>l<sub>ong</sub> t<sub>o</sub> $[ 5 / 8 , 1 1 / 8 ] ^ { 2 } \subset [ 1 / 2 , 3 / 2 ] ^ { 2 }$ , and (49) holds at time �. This com<sub>p</sub>letes the i<sub>n</sub>d<sub>uc</sub>ti<sub>on</sub> th<sub>roug</sub>h ti<sub>me</sub> $2 N + 4$

I<sub>n par</sub>ti<sub>cu</sub>l<sub>ar,</sub> t<sub>runca</sub>ti<sub>on</sub> i<sub>s</sub> i<sub>nac</sub>ti<sub>ve</sub> th<sub>roug</sub>h<sub>ou</sub>t th<sub>e</sub> bl<sub>oc</sub>k<sub>.</sub> F<sub>or</sub> $4 \leq t \leq 2 N + 4$ <sub>, we a</sub>l<sub>so</sub> h<sub>ave</sub>

$$
\frac { 1 } { 2 } = 2 \left( \frac { 1 } { 2 } \right) ^ { 2 } \leq \| r ^ { ( t ) } \| _ { 2 } ^ { 2 } \leq 2 \left( \frac { 3 } { 2 } \right) ^ { 2 } = \frac { 9 } { 2 } ,
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>mp</sub>li<sub>es,</sub> <sub>a</sub>t <sub>every</sub> <sub>nonzero-u</sub>tilit<sub>y</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on,</sub>

$$
\frac { \varepsilon ^ { 2 } } { 2 } \leq \| \pmb { u } ^ { ( t ) } \| _ { 2 } ^ { 2 } \leq \frac { 9 \varepsilon ^ { 2 } } { 2 } , \mathrm { w h e r e } t \in \{ 4 , \ldots , N + 2 \} \cup \{ N + 4 , \ldots , 2 N + 2 \} .\tag{50}
$$

Th<sub>e upper</sub> b<sub>oun</sub>d <sub>a</sub>l<sub>so</sub> h<sub>o</sub>ld<sub>s a</sub>t th<sub>e</sub> t<sub>wo zero-u</sub>tilit<sub>y o</sub>b<sub>serva</sub>ti<sub>ons.</sub> M<sub>oreover,</sub> f<sub>or</sub> 4 $\leq t \leq 2 N + 3 .$ <sub>,</sub> <sub>every</sub> <sub>u</sub>tilit<sub>y</sub> i<sub>n</sub> th<sub>e</sub> bl<sub>oc</sub>k <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\| \pmb { u } ^ { ( t ) } \| _ { \infty } \leq \frac { 3 } { 2 } \varepsilon = \frac { 3 } { 1 6 N ( 1 + \lambda ) } < 1 .
$$

Regret norm decrease over the first half block. We now show that the first half block decreases the regret-vector <sub>norm.</sub> Si<sub>nce</sub> $\pmb { r } ^ { ( t ) } = \tilde { \pmb { r } } ^ { ( t ) } + \lambda \pmb { u } ^ { ( t - 1 ) }$ <sub>an</sub>d $\langle { \pmb u } ^ { ( t ) } , { \pmb r } ^ { ( t ) } \rangle = 0$ <sub>, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\| \tilde { \pmb { r } } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \tilde { \pmb { r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } = 2 \langle \pmb { u } ^ { ( t ) } , \tilde { \pmb { r } } ^ { ( t ) } \rangle + \| \pmb { u } ^ { ( t ) } \| _ { 2 } ^ { 2 } = \| \pmb { u } ^ { ( t ) } \| _ { 2 } ^ { 2 } - 2 \lambda \langle \pmb { u } ^ { ( t - 1 ) } , \pmb { u } ^ { ( t ) } \rangle .
$$

Usin<sub>g</sub> $2 \langle { \pmb u } ^ { ( t - 1 ) } , { \pmb u } ^ { ( t ) } \rangle = \| { \pmb u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + \| { \pmb u } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 }$ <sub>an</sub>d <sub>summ</sub>i<sub>ng</sub> <sub>over</sub> $t = 4 , \ldots , N + 3$ , w<sup>e</sup> g<sup>et</sup>

$$
\| \tilde { \boldsymbol { r } } ^ { ( N + 4 ) } \| _ { 2 } ^ { 2 } - \| \tilde { \boldsymbol { r } } ^ { ( 4 ) } \| _ { 2 } ^ { 2 } = - ( 2 \lambda - 1 ) \sum _ { t = 4 } ^ { N + 2 } \| \boldsymbol { u } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \lambda \sum _ { t = 4 } ^ { N + 3 } \| \boldsymbol { u } ^ { ( t ) } - \boldsymbol { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } ,\tag{51}
$$

<sub>w</sub>h<sub>ere we use</sub> $\pmb { u } ^ { ( 3 ) } = \pmb { u } ^ { ( N + 3 ) } = \pmb { 0 }$

E<sub>ac</sub>h <sub>o</sub>f th<sub>e</sub> $N - 1$ <sub>nonzero</sub> <sub>u</sub>tiliti<sub>es</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> <sub>a</sub>t l<sub>eas</sub>t $\varepsilon ^ { 2 } / 2$ to the first sum b<sub>y</sub> (50). The choice of � therefore <sub>g</sub>ives

$$
( 2 \lambda - 1 ) \sum _ { t = 4 } ^ { N + 2 } \| \pmb { u } ^ { ( t ) } \| _ { 2 } ^ { 2 } \geq \frac { ( 2 \lambda - 1 ) ( N - 1 ) } { 2 } \varepsilon ^ { 2 } \geq 8 \lambda \varepsilon ^ { 2 } .\tag{52}
$$

F<sub>or</sub> th<sub>e</sub> <sub>u</sub>tilit<sub>y</sub> <sub>var</sub>i<sub>a</sub>ti<sub>on</sub> <sub>sum,</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>an</sub>d fi<sub>na</sub>l <sub>c</sub>h<sub>anges</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e</sub> $2 \varepsilon ^ { 2 }$ <sub>an</sub>d <sub>a</sub>t <sub>mos</sub>t $9 \varepsilon ^ { 2 } / 2$ <sub>,</sub> <sub>respec</sub>ti<sub>ve</sub>l<sub>y.</sub> Thi<sub>s</sub> i<sub>s</sub> b<sub>ecause</sub>

$$
\begin{array} { c } { \displaystyle \| { \pmb u } ^ { ( 4 ) } - { \pmb u } ^ { ( 3 ) } \| _ { 2 } ^ { 2 } = \| { \pmb u } ^ { ( 4 ) } \| _ { 2 } ^ { 2 } = 2 \varepsilon ^ { 2 } , } \\ { \| { \pmb u } ^ { ( N + 3 ) } - { \pmb u } ^ { ( N + 2 ) } \| _ { 2 } ^ { 2 } = \| { \pmb u } ^ { ( N + 2 ) } \| _ { 2 } ^ { 2 } \leq \displaystyle \frac { 9 \varepsilon ^ { 2 } } { 2 } . } \end{array}
$$

T<sub>o</sub> b<sub>oun</sub>d th<sub>e</sub> i<sub>n</sub>t<sub>er</sub>i<sub>or</sub> t<sub>erms</sub> f<sub>or</sub> $5 \leq t \leq N + 2$ , (49) <sub>g</sub>ives

$$
\pmb { r } ^ { ( t ) } - \pmb { r } ^ { ( t - 1 ) } = ( 1 + \lambda ) \pmb { u } ^ { ( t - 1 ) } - \lambda \pmb { u } ^ { ( t - 2 ) } .
$$

Th<sub>e u</sub>tiliti<sub>es sa</sub>ti<sub>s</sub>f<sub>y</sub> $\| \pmb { u } ^ { ( t ) } \| _ { 2 } \leq 3 \varepsilon / \sqrt { 2 }$ from (50), so we have, for $5 \leq t \leq N + 2$

$$
\| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } = \varepsilon \| \pmb { r } ^ { ( t ) } - \pmb { r } ^ { ( t - 1 ) } \| _ { 2 } \leq \frac { 3 } { \sqrt { 2 } } ( 1 + 2 \lambda ) \varepsilon ^ { 2 } .
$$

C<sub>onsequen</sub>tl<sub>y, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\sum _ { t = 4 } ^ { N + 3 } \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } \le 2 \varepsilon ^ { 2 } + \frac { 9 \varepsilon ^ { 2 } } { 2 } + \frac { 9 } { 2 } ( N - 2 ) ( 1 + 2 \lambda ) ^ { 2 } \varepsilon ^ { 4 } \le \frac { 1 3 \varepsilon ^ { 2 } } { 2 } + \frac { 9 ( N - 2 ) \varepsilon ^ { 2 } } { 3 2 N ^ { 2 } } < 7 \varepsilon ^ { 2 } ,\tag{53}
$$

where we substitute (47) and use $1 + 2 \lambda \leq 2 ( 1 + \lambda )$ <sub>an</sub>d $N \geq 3$ . Substitutin<sub>g</sub> (52) and (53) into (51), we obtain

$$
\| \tilde { \pmb { r } } ^ { ( N + 4 ) } \| _ { 2 } ^ { 2 } - \| \tilde { \pmb { r } } ^ { ( 4 ) } \| _ { 2 } ^ { 2 } \leq - 8 \lambda \varepsilon ^ { 2 } + \lambda \sum _ { t = 4 } ^ { N + 3 } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } < - 8 \lambda \varepsilon ^ { 2 } + 7 \lambda \varepsilon ^ { 2 } = - \lambda \varepsilon ^ { 2 } < 0 .\tag{54}
$$

H<sub>ence,</sub> th<sub>e</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>norm</sub> <sub>a</sub>t th<sub>e</sub> <sub>en</sub>d <sub>o</sub>f th<sub>e</sub> f<sub>orwar</sub>d h<sub>a</sub>lf bl<sub>oc</sub>k i<sub>s</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> <sub>sma</sub>ll<sub>er</sub> th<sub>an</sub> <sub>a</sub>t it<sub>s</sub> b<sub>eg</sub>i<sub>nn</sub>i<sub>ng.</sub>

Regret vector to original direction over the second half block. It remains to prove that the second half block returns the regret vector to its original direction. We denote the skew symmetric matrix J as

$$
\mathbf { J } = \left( { \begin{array} { c c } { 0 } & { - 1 } \\ { 1 } & { 0 } \end{array} } \right) .
$$

It <sub>s</sub>t<sub>ra</sub>i<sub>g</sub>htf<sub>orwar</sub>dl<sub>y sa</sub>ti<sub>s</sub>f<sub>y</sub> $\mathbf { J } ^ { 2 } = - \mathbf { I } , \mathbf { J } ^ { \top } = - \mathbf { J }$ , where I is the identity matrix. The first update of the block can be written in terms of J as $\tilde { r } ^ { ( 5 ) } = ( \dot { I } + \varepsilon J ) \tilde { r } ^ { ( 4 ) }$ <sub>.</sub> F<sub>or</sub> $5 \leq t \leq N + 2 .$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub> $\bar { \pmb { u } } ^ { ( t - 1 ) } = \tilde { \pmb { r } } ^ { ( t ) } - \tilde { \pmb { r } } ^ { ( t - 1 ) }$ , so

$$
\tilde { r } ^ { ( t + 1 ) } = \left( \mathbf { I } + ( 1 + \lambda ) \varepsilon \mathbf { J } \right) \tilde { r } ^ { ( t ) } - \lambda \varepsilon \mathbf { J } \tilde { r } ^ { ( t - 1 ) } .\tag{55}
$$

W<sub>e</sub> <sub>can</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>rea</sub>l <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>s</sub> th<sub>a</sub>t <sub>represen</sub>t thi<sub>s</sub> <sub>recurrence:</sub>

$$
p _ { 0 } ( z ) = 1 , p _ { 1 } ( z ) = 1 + z , p _ { k + 1 } ( z ) = \left( 1 + ( 1 + \lambda ) z \right) p _ { k } ( z ) - \lambda z p _ { k - 1 } ( z ) ,
$$

f<sub>or</sub> $k \geq 1$ <sub>.</sub> W<sub>e now a</sub>i<sub>m</sub> t<sub>o prove</sub> b<sub>y</sub> i<sub>n</sub>d<sub>uc</sub>ti<sub>on on</sub> � th<sub>a</sub>t

$$
\tilde { r } ^ { ( 4 + k ) } = p _ { k } ( \varepsilon { \bf J } ) \tilde { r } ^ { ( 4 ) } ,
$$

f<sub>or</sub> $0 \leq k \leq N - 1$ <sub>.</sub> F<sub>or</sub> $k = 0$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub> $p _ { 0 } ( \varepsilon \mathbf { J } ) = \mathbf { I } .$ <sub>,</sub> <sub>so</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y</sub> h<sub>o</sub>ld<sub>s.</sub> F<sub>or</sub> $k = 1$ <sub>,</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>up</sub>d<sub>a</sub>t<sub>e</sub> <sub>g</sub>i<sub>ves</sub>

$$
\tilde { r } ^ { ( 5 ) } = ( { \bf I } + \varepsilon { \bf J } ) \tilde { r } ^ { ( 4 ) } = p _ { 1 } ( \varepsilon { \bf J } ) \tilde { r } ^ { ( 4 ) } .
$$

N<sub>ow</sub> fi<sub>x</sub> $1 \leq k \leq N - 2$ <sub>an</sub>d <sub>suppose</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> $k - 1$ <sub>an</sub>d �<sub>, name</sub>l<sub>y,</sub>

$$
\tilde { r } ^ { ( 3 + k ) } = p _ { k - 1 } ( \varepsilon { \bf J } ) \tilde { r } ^ { ( 4 ) } , \tilde { r } ^ { ( 4 + k ) } = p _ { k } ( \varepsilon { \bf J } ) \tilde { r } ^ { ( 4 ) } .
$$

Si<sub>nce</sub> $5 \leq 4 + k \leq N + 2$ , we can a<sub>pp</sub>l<sub>y</sub> (55) at time $t = 4 + k$ t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\tilde { r } ^ { ( 5 + k ) } = \left( \mathbf { I } + ( 1 + \lambda ) \varepsilon \mathbf { J } \right) \tilde { r } ^ { ( 4 + k ) } - \lambda \varepsilon \mathbf { J } \tilde { r } ^ { ( 3 + k ) } .
$$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> th<sub>e</sub> t<sub>wo</sub> i<sub>n</sub>d<sub>uc</sub>ti<sub>on</sub> h<sub>ypo</sub>th<sub>eses an</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on o</sub>f $p _ { k + 1 }$ , w<sup>e</sup> g<sup>et</sup>

$$
\tilde { r } ^ { ( 5 + k ) } = \left[ \left( \mathbf { I } + ( 1 + \lambda ) \varepsilon \mathbf { J } \right) p _ { k } ( \varepsilon \mathbf { J } ) - \lambda \varepsilon \mathbf { J } p _ { k - 1 } ( \varepsilon \mathbf { J } ) \right] \tilde { r } ^ { ( 4 ) } = p _ { k + 1 } ( \varepsilon \mathbf { J } ) \tilde { r } ^ { ( 4 ) } .
$$

Thi<sub>s</sub> <sub>proves</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y</sub> f<sub>or</sub> $k + 1$ <sub>an</sub>d <sub>comp</sub>l<sub>e</sub>t<sub>es</sub> th<sub>e</sub> i<sub>n</sub>d<sub>uc</sub>ti<sub>on.</sub>

T<sub>a</sub>ki<sub>ng</sub> $k = N - 1$ i<sub>n</sub> th<sub>e</sub> <sub>prece</sub>di<sub>ng</sub> id<sub>en</sub>tit<sub>y</sub> <sub>g</sub>i<sub>ves</sub> $\tilde { r } ^ { ( N + 3 ) } = p _ { N - 1 } ( \varepsilon \mathbf { J } ) \tilde { r } ^ { ( 4 ) }$ <sub>.</sub> Si<sub>nce</sub> $\pmb { u } ^ { ( N + 3 ) } = \pmb { 0 }$ <sub>,</sub> th<sub>e</sub> <sub>nex</sub>t <sub>up</sub>d<sub>a</sub>t<sub>e</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\tilde { r } ^ { ( N + 4 ) } = \tilde { r } ^ { ( N + 3 ) } + u ^ { ( N + 3 ) } = \tilde { r } ^ { ( N + 3 ) } , r ^ { ( N + 4 ) } = \tilde { r } ^ { ( N + 4 ) } + \lambda u ^ { ( N + 3 ) } = \tilde { r } ^ { ( N + 4 ) } .
$$

C<sub>onsequen</sub>tl<sub>y,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\tilde { r } ^ { ( N + 4 ) } = r ^ { ( N + 4 ) } = p _ { N - 1 } ( \varepsilon \mathbf { J } ) \tilde { r } ^ { ( 4 ) } .\tag{56}
$$

Th<sub>e</sub> <sub>secon</sub>d h<sub>a</sub>lf bl<sub>oc</sub>k <sub>uses</sub> $\pmb { u } ^ { ( t ) } = - \varepsilon \mathbf { J } r ^ { ( t ) }$ f<sub>or</sub> $N + 4 \leq t \leq 2 N + 2$ <sub>.</sub> Si<sub>nce</sub> $\pmb { r } ^ { ( N + 4 ) } = \tilde { \pmb { r } } ^ { ( N + 4 ) }$ <sub>,</sub> it<sub>s</sub> fi<sub>rs</sub>t <sub>up</sub>d<sub>a</sub>t<sub>e</sub> i<sub>s</sub>

$$
\tilde { r } ^ { ( N + 5 ) } = \tilde { r } ^ { ( N + 4 ) } - \varepsilon { \bf J } r ^ { ( N + 4 ) } = ( { \bf I } - \varepsilon { \bf J } ) \tilde { r } ^ { ( N + 4 ) } = p _ { 1 } ( - \varepsilon { \bf J } ) \tilde { r } ^ { ( N + 4 ) } .
$$

For the remainin<sub>g</sub> nonzero u<sub>p</sub>dates, the same derivation as in (55) <sub>g</sub>ives

$$
\tilde { r } ^ { ( t + 1 ) } = \left( \mathbf { I } - ( 1 + \lambda ) \varepsilon \mathbf { J } \right) \tilde { r } ^ { ( t ) } + \lambda \varepsilon \mathbf { J } \tilde { r } ^ { ( t - 1 ) } ,
$$

f<sub>or</sub> $N + 5 \leq t \leq 2 N + 2$ . Thus, both the initial update and the recurrence are those of the first half with �J replaced by $- \varepsilon \mathbf { J } .$ A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e s</sub>i<sub>m</sub>il<sub>ar</sub> i<sub>n</sub>d<sub>uc</sub>ti<sub>on, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\tilde { r } ^ { ( N + 4 + k ) } = p _ { k } ( - \varepsilon { \bf J } ) \tilde { r } ^ { ( N + 4 ) } ,
$$

f<sub>or</sub> $0 \leq k \leq N - 1$ <sub>.</sub> I<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar,</sub> <sub>a</sub>t $k = N - 1$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\tilde { r } ^ { ( 2 N + 3 ) } = p _ { N - 1 } ( - \varepsilon { \bf J } ) \tilde { r } ^ { ( N + 4 ) } .
$$

Fi<sub>na</sub>ll<sub>y,</sub> $\pmb { u } ^ { ( 2 N + 3 ) } = \mathbf { 0 }$ <sub>g</sub><sup>i</sup>ves

$$
\tilde { r } ^ { ( 2 N + 4 ) } = \tilde { r } ^ { ( 2 N + 3 ) } , r ^ { ( 2 N + 4 ) } = \tilde { r } ^ { ( 2 N + 4 ) } .
$$

Substitutin<sub>g</sub> the <sub>p</sub>recedin<sub>g</sub> ex<sub>p</sub>ression and (56), we obtain

$$
\tilde { r } ^ { ( 2 N + 4 ) } = r ^ { ( 2 N + 4 ) } = p _ { N - 1 } ( - \varepsilon { \bf J } ) p _ { N - 1 } ( \varepsilon { \bf J } ) \tilde { r } ^ { ( 4 ) } .\tag{57}
$$

W<sub>e</sub> fi<sub>rs</sub>t <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> <sub>mu</sub>lti<sub>p</sub>l<sub>y</sub>i<sub>ng</sub> $\tilde { r } ^ { ( 4 ) }$ in (57) is a scalar multiple of I. Replacin<sub>g</sub> � b<sub>y</sub> −� in the pol<sub>y</sub>nomial $p _ { N - 1 } ( - z ) p _ { N - 1 } ( z )$ <sub>g</sub><sup>i</sup>ves

$$
p _ { N - 1 } ( z ) p _ { N - 1 } ( - z ) = p _ { N - 1 } ( - z ) p _ { N - 1 } ( z ) .
$$

Thus, this re<sub>p</sub>lacement leaves the <sub>p</sub>ol<sub>y</sub>nomial unchan<sub>g</sub>ed. Since re<sub>p</sub>lacin<sub>g</sub> � b<sub>y</sub> −� reverses the si<sub>g</sub>n of ever<sub>y</sub> odd <sub>p</sub>ower, <sub>a</sub>ll <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> <sub>o</sub>f <sub>o</sub>dd <sub>powers</sub> <sub>mus</sub>t b<sub>e</sub> <sub>zero.</sub> Th<sub>e</sub> <sub>pro</sub>d<sub>uc</sub>t th<sub>ere</sub>f<sub>ore</sub> <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> <sub>on</sub>l<sub>y</sub> <sub>powers</sub> $z ^ { 2 j }$ <sub>.</sub> Aft<sub>er</sub> <sub>su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> $z = \varepsilon \mathbf { J }$ <sub>eac</sub>h <sub>suc</sub>h <sub>power</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
( \varepsilon { \bf J } ) ^ { 2 j } = \varepsilon ^ { 2 j } ( { \bf J } ^ { 2 } ) ^ { j } = ( - 1 ) ^ { j } \varepsilon ^ { 2 j } { \bf I } .
$$

Every term in the product is consequently a scalar multiple of I, so there is a real scalar $\rho$ <sub>suc</sub>h th<sub>a</sub>t

$$
p _ { N - 1 } ( - \varepsilon \mathbf { J } ) p _ { N - 1 } ( \varepsilon \mathbf { J } ) = \rho \mathbf { I } .
$$

W<sub>e</sub> <sub>nex</sub>t id<sub>en</sub>tif<sub>y</sub> $\rho$ <sub>us</sub>i<sub>ng</sub> th<sub>e</sub> <sub>norm</sub> <sub>a</sub>t th<sub>e</sub> <sub>en</sub>d <sub>o</sub>f th<sub>e</sub> f<sub>orwar</sub>d h<sub>a</sub>lf bl<sub>oc</sub>k<sub>.</sub> Si<sub>nce</sub> th<sub>e</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> <sub>are</sub> <sub>rea</sub>l<sub>,</sub> trans<sub>p</sub>os<sup>i</sup>n<sub>g</sub> <sup>i</sup>ts matr<sup>i</sup>x ar<sub>g</sub>ument <sub>g</sub><sup>i</sup>ves

$$
p _ { N - 1 } ( \varepsilon \mathbf { J } ) ^ { \top } = p _ { N - 1 } ( ( \varepsilon \mathbf { J } ) ^ { \top } ) = p _ { N - 1 } ( - \varepsilon \mathbf { J } ) ,
$$

<sub>w</sub>h<sub>ere</sub> <sub>we</sub> <sub>use</sub> $\mathbf { J } ^ { \top } = - \mathbf { J }$ <sub>.</sub> H<sub>ence,</sub>

$$
p _ { N - 1 } ( \varepsilon \mathbf { J } ) ^ { \top } p _ { N - 1 } ( \varepsilon \mathbf { J } ) = \rho \mathbf { I } .
$$

T<sub>a</sub>ki<sub>ng</sub> th<sub>e</sub> i<sub>nner pro</sub>d<sub>uc</sub>t <sub>w</sub>ith $\tilde { r } ^ { ( 4 ) }$ <sub>on</sub> b<sub>o</sub>th <sub>s</sub>id<sub>es, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r } { \| p _ { N - 1 } ( \pmb { \varepsilon } \mathbf { J } ) \tilde { \pmb { r } } ^ { ( 4 ) } \| _ { 2 } ^ { 2 } = \langle \tilde { \pmb { r } } ^ { ( 4 ) } , p _ { N - 1 } ( \pmb { \varepsilon } \mathbf { J } ) ^ { \top } p _ { N - 1 } ( \pmb { \varepsilon } \mathbf { J } ) \tilde { \pmb { r } } ^ { ( 4 ) } \rangle = \rho \| \tilde { \pmb { r } } ^ { ( 4 ) } \| _ { 2 } ^ { 2 } . } \end{array}
$$

B<sub>y</sub> (56), the vector inside the norm on the left is $\tilde { r } ^ { ( N + 4 ) }$ <sub>.</sub> Si<sub>nce</sub> $\tilde { r } ^ { ( 4 ) } = ( 1 , 1 )$ <sub>,</sub> <sub>we</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>ge</sub>t

$$
\rho = \frac { \| \tilde { r } ^ { ( N + 4 ) } \| _ { 2 } ^ { 2 } } { \| \tilde { r } ^ { ( 4 ) } \| _ { 2 } ^ { 2 } } = \frac { \| \tilde { r } ^ { ( N + 4 ) } \| _ { 2 } ^ { 2 } } { 2 } .
$$

Th<sub>e</sub> <sub>vec</sub>t<sub>or</sub> $\tilde { r } ^ { ( N + 4 ) }$ has <sub>p</sub>ositive coordinates, and (54) <sub>g</sub>ives $\| \tilde { r } ^ { ( N + 4 ) } \| _ { 2 } ^ { 2 } < \| \tilde { r } ^ { ( 4 ) } \| _ { 2 } ^ { 2 } = 2$ . <sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>, $0 < \rho < 1$ Substitutin<sub>g</sub> the scalar matrix into (57), we conclude that

$$
\tilde { r } ^ { ( 2 N + 4 ) } = r ^ { ( 2 N + 4 ) } = \rho \tilde { r } ^ { ( 4 ) } = \rho ( 1 , 1 ) ,\tag{58}
$$

<sub>w</sub>h<sub>ere</sub> $0 < \rho < 1$ . Thus, the full block returns both regret vectors to their original directions with strictly smaller norms.

Repeating the block to violate Alt-RVU. We now repeat the utility rule (48), shifting its indices by 2� after each block. B<sub>y</sub> (58), the second block starts from $\tilde { r } ^ { ( 2 N + 4 ) } = \hat { \bar { r } } ^ { ( 2 N + 4 ) } = \rho ( \bar { 1 } , 1 )$ <sub>.</sub> M<sub>u</sub>lti<sub>p</sub>l<sub>y</sub>i<sub>ng a pos</sub>iti<sub>ve pre</sub>di<sub>c</sub>ti<sub>on vec</sub>t<sub>or</sub> b<sub>y</sub> $\rho > 0$ l<sub>eaves</sub> it<sub>s s</sub>t<sub>ra</sub>t<sub>egy unc</sub>h<sub>ange</sub>d<sub>, s</sub>i<sub>nce</sub>

$$
\frac { \rho \pmb { r } ^ { ( t ) } } { \| \rho \pmb { r } ^ { ( t ) } \| _ { 1 } } = \frac { \pmb { r } ^ { ( t ) } } { \| \pmb { r } ^ { ( t ) } \| _ { 1 } } = \pmb { x } ^ { ( t ) } .
$$

Th<sub>e u</sub>tilit<sub>y ru</sub>l<sub>e</sub> th<sub>ere</sub>f<sub>ore mu</sub>lti<sub>p</sub>li<sub>es</sub> th<sub>e correspon</sub>di<sub>ng u</sub>tilit<sub>y</sub> b<sub>y</sub> $\rho .$ <sub>.</sub> Si<sub>nce</sub> th<sub>e payo</sub>f <sub>rema</sub>i<sub>ns zero,</sub> th<sub>e</sub> t<sub>wo regre</sub>t <sub>vec</sub>t<sub>ors</sub> u<sub>p</sub><sup>d</sup>ates <sub>g</sub><sup>i</sup>ve

$$
[ \rho \tilde { r } ^ { ( t ) } + \rho { \pmb u } ^ { ( t ) } ] _ { + } = \rho \tilde { r } ^ { ( t + 1 ) } , \rho \tilde { r } ^ { ( t + 1 ) } + \lambda \rho { \pmb u } ^ { ( t ) } = \rho r ^ { ( t + 1 ) } .
$$

Th<sub>us, every regre</sub>t <sub>vec</sub>t<sub>or an</sub>d <sub>u</sub>tilit<sub>y</sub> i<sub>n</sub> th<sub>e secon</sub>d bl<sub>oc</sub>k i<sub>s �</sub> ti<sub>mes</sub> it<sub>s coun</sub>t<sub>erpar</sub>t i<sub>n</sub> th<sub>e</sub> fi<sub>rs</sub>t bl<sub>oc</sub>k<sub>, w</sub>h<sub>ereas</sub> th<sub>e</sub> <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es are</sub> id<sub>en</sub>ti<sub>ca</sub>l<sub>.</sub> P<sub>os</sub>iti<sub>v</sub>it<sub>y</sub> i<sub>s preserve</sub>d b<sub>ecause</sub> $\rho > 0 .$ <sub>.</sub> R<sub>epea</sub>ti<sub>ng</sub> thi<sub>s argumen</sub>t <sub>over success</sub>i<sub>ve</sub> bl<sub>oc</sub>k<sub>s, we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\tilde { r } ^ { ( t + 2 N ) } = \rho \tilde { r } ^ { ( t ) } , \ r ^ { ( t + 2 N ) } = \rho r ^ { ( t ) } , \ u ^ { ( t + 2 N ) } = \rho u ^ { ( t ) } , \ x ^ { ( t + 2 N ) } = x ^ { ( t ) } ,
$$

f<sub>or</sub> $t \geq 4 .$ <sub>.</sub> Th<sub>e</sub> <sub>repea</sub>t<sub>e</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es</sub> <sub>are</sub> <sub>no</sub>t <sub>cons</sub>t<sub>an</sub>t<sub>.</sub> I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> ${ \pmb u } ^ { ( 4 ) } = \varepsilon ( - 1 , 1 )$ <sub>an</sub>d $\tilde { r } ^ { ( 4 ) } = r ^ { ( 4 ) } = ( 1 , 1 )$ <sub>g</sub><sup>i</sup>ve

$$
r ^ { ( 5 ) } = \left( 1 - ( 1 + \lambda ) \varepsilon , 1 + ( 1 + \lambda ) \varepsilon \right) , \ x ^ { ( 5 ) } = \left( \frac { 1 - ( 1 + \lambda ) \varepsilon } { 2 } , \frac { 1 + ( 1 + \lambda ) \varepsilon } { 2 } \right) .
$$

Si<sub>nce</sub> $\pmb { x } ^ { ( 4 ) } = ( 1 / 2 , 1 / 2 )$ <sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\| \pmb { x } ^ { ( 5 ) } - \pmb { x } ^ { ( 4 ) } \| _ { 2 } ^ { 2 } = \frac { ( 1 + \lambda ) ^ { 2 } \pmb { \varepsilon } ^ { 2 } } { 2 } > 0 .
$$

E<sub>very</sub> <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> bl<sub>oc</sub>k <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> thi<sub>s</sub> <sub>same</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>movemen</sub>t<sub>.</sub> Th<sub>ere</sub> <sub>are</sub> $\lfloor ( T - 3 ) / ( 2 N ) \rfloor$ <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> bl<sub>oc</sub>k<sub>s</sub> <sub>among</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>ons</sub> $4 , \ldots , T ,$ <sub>,</sub> <sub>so,</sub> f<sub>or</sub> $T \geq 3$ , w<sup>e</sup> g<sup>et</sup>

$$
\sum _ { t = 1 } ^ { T } \| \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } \| _ { 2 } ^ { 2 } \geq \left\lfloor \frac { T - 3 } { 2 N } \right\rfloor \frac { ( 1 + \lambda ) ^ { 2 } \varepsilon ^ { 2 } } { 2 } .
$$

O<sub>n</sub> th<sub>e o</sub>th<sub>er</sub> h<sub>an</sub>d<sub>,</sub> $\| \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } \| _ { 2 } ^ { 2 } \leq 2$ b<sub>ecause</sub> b<sub>o</sub>th <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es</sub> b<sub>e</sub>l<sub>ong</sub> t<sub>o</sub> $\Delta ( 2 )$ . <sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>,

$$
\sum _ { t = 1 } ^ { T } \| \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } \| _ { 2 } ^ { 2 } = \Theta ( T ) .
$$

I<sub>n con</sub>t<sub>ras</sub>t<sub>,</sub> th<sub>e u</sub>tiliti<sub>es</sub> d<sub>ecrease geome</sub>t<sub>r</sub>i<sub>ca</sub>ll<sub>y across</sub> bl<sub>oc</sub>k<sub>s.</sub> Si<sub>nce</sub> $0 < \rho < 1$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\sum _ { t = 1 } ^ { \infty } \| \pmb { u } ^ { ( t ) } \| _ { 2 } = \sum _ { t = 1 } ^ { 3 } \| \pmb { u } ^ { ( t ) } \| _ { 2 } + \sum _ { k = 0 } ^ { \infty } \rho ^ { k } \sum _ { t = 4 } ^ { 2 N + 3 } \| \pmb { u } ^ { ( t ) } \| _ { 2 } = \sum _ { t = 1 } ^ { 3 } \| \pmb { u } ^ { ( t ) } \| _ { 2 } + \frac { 1 } { 1 - \rho } \sum _ { t = 4 } ^ { 2 N + 3 } \| \pmb { u } ^ { ( t ) } \| _ { 2 } < \infty .
$$

All <sub>u</sub>tiliti<sub>es</sub> li<sub>e</sub> i<sub>n</sub> $[ - 1 , 1 ] ^ { 2 }$ <sub>,</sub> <sub>s</sub>i<sub>nce</sub> thi<sub>s</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on</sub> <sub>an</sub>d th<sub>e</sub> fi<sub>rs</sub>t bl<sub>oc</sub>k<sub>,</sub> <sub>as</sub> <sub>su</sub>b<sub>sequen</sub>t bl<sub>oc</sub>k<sub>s</sub> <sub>on</sub>l<sub>y</sub> d<sub>ecrease</sub> th<sub>e</sub>i<sub>r magn</sub>it<sub>u</sub>d<sub>es.</sub> M<sub>oreover, we no</sub>t<sub>e</sub> th<sub>a</sub>t

$$
\sum _ { t = 2 } ^ { \infty } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } \leq 2 \sum _ { t = 2 } ^ { \infty } \left( \| \pmb { u } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \| \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } \right) \leq 4 \sum _ { t = 1 } ^ { \infty } \| \pmb { u } ^ { ( t ) } \| _ { 2 } ^ { 2 } \leq 4 \left( \sum _ { t = 1 } ^ { \infty } \| \pmb { u } ^ { ( t ) } \| _ { 2 } \right) ^ { 2 } < \infty .
$$

Fi<sub>na</sub>ll<sub>y,</sub> f<sub>or</sub> <sub>any</sub> $z \in \Delta ( 2 )$ <sub>,</sub> th<sub>e</sub> t<sub>r</sub>i<sub>ang</sub>l<sub>e</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>g</sub>i<sub>ves</sub> $\| 2 z - x ^ { ( t ) } - x ^ { ( t + 1 ) } \| _ { 2 } \leq 4$ <sub>.</sub> H<sub>ence,</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>regre</sub>t <sub>rema</sub>i<sub>ns</sub> b<sub>oun</sub>d<sub>e</sub>d i<sub>n</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> <sub>va</sub>l<sub>ue:</sub>

$$
| \mathrm { A l t - R e g } ( T ) | \leq \operatorname* { m a x } _ { z \in \Delta ( 2 ) } \sum _ { t = 1 } ^ { T } \left| \langle { \pmb u } ^ { ( t ) } , 2 z - { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t + 1 ) } \rangle \right| \leq 4 \sum _ { t = 1 } ^ { T } \| { \pmb u } ^ { ( t ) } \| _ { 2 } \leq 4 \sum _ { t = 1 } ^ { \infty } \| { \pmb u } ^ { ( t ) } \| _ { 2 } < \infty .
$$

F<sub>or any</sub> h<sub>or</sub>i<sub>zon-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>cons</sub>t<sub>an</sub>t<sub>s</sub> $\alpha , \beta , \gamma > 0$ <sub>,</sub> <sub>we</sub> th<sub>ere</sub>f<sub>ore</sub> h<sub>ave</sub>

$$
\alpha + \beta \sum _ { t = 2 } ^ { T } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } - \gamma \sum _ { t = 1 } ^ { T } \| \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } \| _ { 2 } ^ { 2 } \longrightarrow - \infty ,
$$

<sub>w</sub>h<sub>ereas</sub> $\mathrm { A l t - R e g } ( T )$ i<sub>s</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>e</sub>l<sub>ow.</sub> Th<sub>us,</sub> th<sub>e</sub> Alt<sub>-</sub>RVU i<sub>nequa</sub>lit<sub>y</sub> <sub>canno</sub>t h<sub>o</sub>ld f<sub>or</sub> <sub>every</sub> h<sub>or</sub>i<sub>zon</sub> <sub>on</sub> thi<sub>s</sub> <sub>sequence,</sub> <sub>even</sub> <sub>w</sub>h<sub>en</sub> it<sub>s</sub> <sub>cons</sub>t<sub>an</sub>t<sub>s</sub> d<sub>epen</sub>d <sub>on</sub> th<sub>e</sub> <sub>en</sub>ti<sub>re</sub> <sub>sequence.</sub> B<sub>y</sub> <sub>norm</sub> <sub>equ</sub>i<sub>va</sub>l<sub>ence</sub> i<sub>n</sub> di<sub>mens</sub>i<sub>on</sub> t<sub>wo,</sub> th<sub>e</sub> <sub>same</sub> <sub>con</sub>t<sub>ra</sub>di<sub>c</sub>ti<sub>on</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or any</sub> fi<sub>xe</sub>d <sub>pa</sub>i<sub>r o</sub>f <sub>pr</sub>i<sub>ma</sub>l<sub>-</sub>d<sub>ua</sub>l <sub>norms.</sub> Si<sub>nce re</sub>fi<sub>ne</sub>d $\mathsf { A l t - R V U }$ <sub>a</sub>dd<sub>s a nonnega</sub>ti<sub>ve</sub> t<sub>erm</sub> t<sub>o</sub> th<sub>e</sub> l<sub>e</sub>ft<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e,</sub> it <sub>canno</sub>t h<sub>o</sub>ld <sub>e</sub>ith<sub>er.</sub> □

Proposition 6.2. $F i x 0 \le \lambda < 1 / 2$ and an integer $d \ge 4$ satisfying $d > ( ( 1 + 2 \lambda ) / ( 1 - 2 \lambda ) ) ^ { 2 }$ . There is a utility sequence $( \pmb { u } ^ { ( t ) } ) _ { t \geq 1 } \subset [ - 1 , 1 ] ^ { d }$ on which the scaled-prediction regret matching<sup>+</sup> learner satisfies

$$
\mathrm { A l t - R e g } ( T ) = \Theta ( \sqrt { T } ) , \ \sum _ { t = 2 } ^ { T } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } = O ( \log T ) .\tag{26}
$$

Consequently, Alt-RVU fails on this sequence for every choice of horizon-independent constants.

Proof. We adapt the forward and reverse half blocks from the proof of Proposition 6.1. For $\lambda < 1 / 2$ <sub>,</sub> th<sub>ese</sub> bl<sub>oc</sub>k<sub>s</sub> i<sub>ncrease</sub> th<sub>e regre</sub>t <sub>vec</sub>t<sub>or norm.</sub> W<sub>e</sub> l<sub>e</sub>t th<sub>e</sub>i<sub>r</sub> l<sub>eng</sub>th<sub>s</sub> i<sub>ncrease w</sub>ith thi<sub>s norm an</sub>d <sub>vary</sub> th<sub>e u</sub>tilit<sub>y coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s smoo</sub>thl<sub>y</sub> <sub>w</sub>ithi<sub>n eac</sub>h h<sub>a</sub>lf bl<sub>oc</sub>k t<sub>o con</sub>t<sub>ro</sub>l th<sub>e square</sub>d <sub>u</sub>tilit<sub>y var</sub>i<sub>a</sub>ti<sub>on.</sub>

Construction. We choose the initial direction

$$
\nu = \left( 1 , { \frac { 1 } { 1 + { \sqrt { d } } } } , \ldots , { \frac { 1 } { 1 + { \sqrt { d } } } } \right) , \ \| \nu \| _ { 1 } = { \sqrt { d } } , \ \| \nu \| _ { 2 } ^ { 2 } = { \frac { 2 { \sqrt { d } } } { 1 + { \sqrt { d } } } } .
$$

Th<sub>e</sub> <sub>assump</sub>ti<sub>on</sub> <sub>on</sub> � <sub>g</sub>i<sub>ves</sub>

$$
\frac { \| \pmb { \nu } \| _ { 2 } ^ { 2 } } { ( 1 - 2 \lambda ) \| \pmb { \nu } \| _ { 1 } } = \frac { 2 } { ( 1 - 2 \lambda ) ( 1 + \sqrt { d } ) } < 1 .
$$

W<sub>e can</sub> th<sub>ere</sub>f<sub>ore</sub> fi<sub>x</sub> $\varepsilon > 0$ <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> <sub>sma</sub>ll th<sub>a</sub>t

$$
\varepsilon < \frac { 1 } { 2 ( 1 + \sqrt { d } ) } , ( 1 - \varepsilon ) ^ { 2 } > \frac { \| \pmb { \nu } \| _ { 2 } ^ { 2 } } { ( 1 - 2 \lambda ) \| \pmb { \nu } \| _ { 1 } } .\tag{59}
$$

W<sub>e</sub> <sub>use</sub> th<sub>e</sub> <sub>s</sub>k<sub>ew-symme</sub>t<sub>r</sub>i<sub>c</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub>

$$
\mathbf { J } = { \frac { ( e _ { 2 } - e _ { 3 } ) { \boldsymbol { \nu } } ^ { \top } - { \boldsymbol { \nu } } ( e _ { 2 } - e _ { 3 } ) ^ { \top } } { \sqrt { 2 } \| { \boldsymbol { \nu } } \| _ { 2 } } } ,
$$

<sub>w</sub>h<sub>ere</sub> $e _ { i }$ i<sub>s</sub> th<sub>e</sub> �<sub>-</sub>th <sub>coor</sub>di<sub>na</sub>t<sub>e vec</sub>t<sub>or.</sub> Si<sub>nce</sub> $\langle \pmb { \nu } , \pmb { e } _ { 2 } - \pmb { e } _ { 3 } \rangle = 0$ <sub>,</sub> thi<sub>s ma</sub>t<sub>r</sub>i<sub>x sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\mathbf { J } ^ { \top } = - \mathbf { J }$ <sub>an</sub>d $\mathbf { J } ^ { 2 } = - \mathbf { I }$ on <sup>s</sup>p<sup>an</sup> $\{ \pmb { \nu } , \pmb { e } _ { 2 } - \pmb { e } _ { 3 } \}$ . In particular, J preserves the Euclidean norm on this plane.

St<sub>ar</sub>ti<sub>ng</sub> f<sub>rom</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on, we se</sub>t

$$
{ \pmb u } ^ { ( 1 ) } = \left( 1 , - \frac { 1 } { d - 1 } , \ldots , - \frac { 1 } { d - 1 } \right) , { \pmb u } ^ { ( 2 ) } = \left( 0 , \frac { 1 } { 1 + \sqrt { d } } , \ldots , \frac { 1 } { 1 + \sqrt { d } } \right) , { \pmb u } ^ { ( 3 ) } = { \pmb 0 } .
$$

Th<sub>e</sub> fi<sub>rs</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>on g</sub>i<sub>ves</sub> $\tilde { r } ^ { ( 2 ) } = e _ { 1 } \mathrm { a n d } x ^ { ( 2 ) } = e _ { 1 }$ <sub>.</sub> Th<sub>e secon</sub>d <sub>g</sub>i<sub>ves</sub> $\tilde { r } ^ { ( 3 ) } = \nu$ <sub>, an</sub>d th<sub>e</sub> thi<sub>r</sub>d <sub>rese</sub>t<sub>s</sub> th<sub>e pre</sub>di<sub>c</sub>ti<sub>on, so</sub>

$$
\tilde { r } ^ { ( 4 ) } = r ^ { ( 4 ) } = \nu .
$$

All th<sub>ree</sub> <sub>p</sub>l<sub>aye</sub>d <sub>u</sub>tiliti<sub>es</sub> <sub>are</sub> <sub>zero.</sub>

W<sub>e</sub> <sub>cons</sub>t<sub>ruc</sub>t <sub>success</sub>i<sub>ve</sub> bl<sub>oc</sub>k<sub>s.</sub> L<sub>e</sub>t $\tau _ { k }$ b<sub>e</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> <sub>o</sub>f bl<sub>oc</sub>k �<sub>,</sub> <sub>w</sub>h<sub>ere</sub> b<sub>o</sub>th <sub>regre</sub>t <sub>vec</sub>t<sub>ors</sub> <sub>equa</sub>l $s _ { k } \nu$ <sub>w</sub>ith $\tau _ { 1 } = 4$ <sub>an</sub>d $s _ { 1 } = 1$ <sub>.</sub> W<sub>e</sub> <sub>c</sub>h<sub>oose</sub>

$$
N _ { k } = \operatorname* { m a x } \{ 2 , \lceil s _ { k } \rceil \} , \tau _ { k + 1 } = \tau _ { k } + 2 N _ { k } ,
$$

<sub>an</sub>d<sub>,</sub> f<sub>or</sub> $1 \le j \le N _ { k }$ <sub>,</sub> <sub>prescr</sub>ib<sub>e</sub>

$$
\begin{array} { l } { { \displaystyle { \pmb u } ^ { ( \tau _ { k } + j - 1 ) } = \frac { \varepsilon \sin ( \pi j / N _ { k } ) } { 4 ( 1 + \lambda ) \| \nu \| _ { 2 } N _ { k } } \mathbf { J } r ^ { ( \tau _ { k } + j - 1 ) } } , } \\ { { \displaystyle { \pmb u } ^ { ( \tau _ { k } + N _ { k } + j - 1 ) } = - \frac { \varepsilon \sin ( \pi j / N _ { k } ) } { 4 ( 1 + \lambda ) \| \nu \| _ { 2 } N _ { k } } \mathbf { J } r ^ { ( \tau _ { k } + N _ { k } + j - 1 ) } } . } \end{array}\tag{60}
$$

Each half ends with a zero utility. We will show that the full block returns both regret vectors to the direction �, thereby d<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub> $s k { + 1 }$ <sub>an</sub>d <sub>a</sub>ll<sub>ow</sub>i<sub>ng</sub> th<sub>e cons</sub>t<sub>ruc</sub>ti<sub>on</sub> t<sub>o con</sub>ti<sub>nue.</sub> All i<sub>mp</sub>li<sub>c</sub>it <sub>cons</sub>t<sub>an</sub>t<sub>s</sub> b<sub>e</sub>l<sub>ow</sub> d<sub>epen</sub>d <sub>on</sub>l<sub>y on</sub> $\lambda , d ,$ <sub>an</sub>d th<sub>e</sub> fi<sub>xe</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ce o</sub>f $\varepsilon .$

No clipping in a block. We follow the induction used in the preceding proof. Suppose that the prediction regret <sub>vec</sub>t<sub>ors up</sub> t<sub>o</sub> th<sub>e curren</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>on sa</sub>ti<sub>s</sub>f<sub>y</sub> $\| r ^ { ( t ) } \| _ { 2 } \leq 2 s _ { k } \| \nu \| _ { 2 }$ <sub>an</sub>d h<sub>ave</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es.</sub> Sk<sub>ew</sub> <sub>symme</sub>t<sub>ry</sub> <sub>g</sub>i<sub>ves</sub> $\langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle = 0$ <sub>, w</sub>hil<sub>e</sub> th<sub>e u</sub>tilit<sub>y ru</sub>l<sub>e g</sub>i<sub>ves</sub>

$$
\| \pmb { u } ^ { ( t ) } \| _ { 2 } \le \frac { \varepsilon s _ { k } } { 2 ( 1 + \lambda ) N _ { k } } .
$$

S<sub>umm</sub>i<sub>ng</sub> th<sub>e</sub> <sub>prev</sub>i<sub>ous</sub> li<sub>near</sub> <sub>up</sub>d<sub>a</sub>t<sub>es,</sub> th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\| \tilde { r } ^ { ( t ) } + u ^ { ( t ) } - s _ { k } \nu \| _ { 2 } \leq 2 N _ { k } \frac { \varepsilon s _ { k } } { 2 ( 1 + \lambda ) N _ { k } } \leq \varepsilon s _ { k } .
$$

E<sub>very</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>o</sub>f $s _ { k } \nu$ i<sub>s</sub> <sub>a</sub>t l<sub>eas</sub>t $s _ { k } / ( 1 + \sqrt { d } )$ , so this candidate remains <sub>p</sub>ositive b<sub>y</sub> (59). Thus, cli<sub>pp</sub>in<sub>g</sub> is inactive and (49) holds. The next <sub>p</sub>rediction re<sub>g</sub>ret vector satisfies

$$
\| r ^ { ( t + 1 ) } - s _ { k } \pmb { \nu } \| _ { 2 } \leq ( 2 N _ { k } + \lambda ) \frac { \varepsilon s _ { k } } { 2 ( 1 + \lambda ) N _ { k } } \leq \varepsilon s _ { k } .
$$

It <sub>a</sub>l<sub>so</sub> h<sub>as pos</sub>iti<sub>ve coor</sub>di<sub>na</sub>t<sub>es an</sub>d <sub>norm a</sub>t <sub>mos</sub>t $2 s _ { k } \| \pmb { \nu } \| _ { 2 }$ <sub>, comp</sub>l<sub>e</sub>ti<sub>ng</sub> th<sub>e</sub> i<sub>n</sub>d<sub>uc</sub>ti<sub>on.</sub>

C<sub>onsequen</sub>tl<sub>y,</sub> th<sub>roug</sub>h<sub>ou</sub>t th<sub>e</sub> bl<sub>oc</sub>k<sub>,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> it<sub>s en</sub>d<sub>po</sub>i<sub>n</sub>t<sub>s,</sub>

$$
\| \tilde { r } ^ { ( t ) } - s _ { k } \nu \| _ { 2 } \leq \varepsilon s _ { k } , \ \| r ^ { ( t ) } - s _ { k } \nu \| _ { 2 } \leq \varepsilon s _ { k } .\tag{61}
$$

I<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar,</sub>

$$
\frac { s _ { k } \| \pmb { \nu } \| _ { 2 } } { 2 } \leq \| \pmb { r } ^ { ( t ) } \| _ { 2 } \leq 2 s _ { k } \| \pmb { \nu } \| _ { 2 } , \ \| \pmb { r } ^ { ( t ) } \| _ { 1 } \geq ( 1 - \varepsilon ) s _ { k } \| \pmb { \nu } \| _ { 1 } , \ \tilde { r } _ { 1 } ^ { ( t ) } \geq ( 1 - \varepsilon ) s _ { k } .\tag{62}
$$

F<sub>or</sub> th<sub>e</sub> $\ell _ { 1 }$ b<sub>oun</sub>d<sub>,</sub> <sub>we</sub> <sub>use</sub> $\| \pmb { \nu } \| _ { 1 } = \sqrt { d }$ <sub>an</sub>d $\| \pmb { r } ^ { ( t ) } - s _ { k } \pmb { \nu } \| _ { 1 } \leq \sqrt { d } \| \pmb { r } ^ { ( t ) } - s _ { k } \pmb { \nu } \| _ { 2 }$ <sub>.</sub> M<sub>oreover,</sub> <sub>s</sub>i<sub>nce</sub> $N _ { k } \ge s _ { k }$

$$
\| { \pmb u } ^ { ( t ) } \| _ { \infty } \le \| { \pmb u } ^ { ( t ) } \| _ { 2 } \le \frac { \varepsilon s _ { k } } { 2 ( 1 + \lambda ) N _ { k } } \le \frac { \varepsilon } { 2 ( 1 + \lambda ) } < 1 .
$$

Th<sub>us,</sub> <sub>every</sub> <sub>u</sub>tilit<sub>y</sub> b<sub>e</sub>l<sub>ongs</sub> t<sub>o</sub> $[ - 1 , 1 ] ^ { d }$

Utility bounds within a block. All regret vectors remain in span $\{ \pmb { \nu } , \pmb { e } _ { 2 } - \pmb { e } _ { 3 } \}$ , where J reserves their Euclidean norms. Combinin<sub>g</sub> (62) with

$$
\sum _ { j = 1 } ^ { N _ { k } } \sin ^ { 2 } ( \pi j / N _ { k } ) = \frac { N _ { k } } { 2 } ,
$$

<sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\sum _ { t = \tau _ { k } } ^ { \tau _ { k } + N _ { k } - 1 } \| \pmb { u } ^ { ( t ) } \| _ { 2 } ^ { 2 } = \Theta \left( \frac { s _ { k } ^ { 2 } } { N _ { k } } \right) .
$$

Th<sub>e same es</sub>ti<sub>ma</sub>t<sub>e</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> th<sub>e secon</sub>d h<sub>a</sub>lf<sub>.</sub>

T<sub>o</sub> b<sub>oun</sub>d <sub>u</sub>tilit<sub>y</sub> <sub>var</sub>i<sub>a</sub>ti<sub>on,</sub> <sub>we</sub> <sub>use</sub>

$$
| \sin ( \pi j / N _ { k } ) - \sin ( \pi ( j - 1 ) / N _ { k } ) | \leq \frac { \pi } { N _ { k } } .
$$

At <sub>every</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> <sub>a</sub>ft<sub>er</sub> th<sub>e</sub> fi<sub>rs</sub>t i<sub>n</sub> <sub>e</sub>ith<sub>er</sub> h<sub>a</sub>lf bl<sub>oc</sub>k<sub>,</sub> th<sub>e</sub> li<sub>near</sub> <sub>up</sub>d<sub>a</sub>t<sub>es</sub> <sub>g</sub>i<sub>ve</sub>

$$
\pmb { r } ^ { ( t ) } - \pmb { r } ^ { ( t - 1 ) } = ( 1 + \lambda ) \pmb { u } ^ { ( t - 1 ) } - \lambda \pmb { u } ^ { ( t - 2 ) } .
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
\| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } \le \frac { \pi \varepsilon \| r ^ { ( t - 1 ) } \| _ { 2 } } { 4 ( 1 + \lambda ) \| \pmb { \nu } \| _ { 2 } N _ { k } ^ { 2 } } + \frac { \varepsilon \| r ^ { ( t ) } - r ^ { ( t - 1 ) } \| _ { 2 } } { 4 ( 1 + \lambda ) \| \pmb { \nu } \| _ { 2 } N _ { k } } = O \left( \frac { s _ { k } } { N _ { k } ^ { 2 } } \right) .
$$

At th<sub>e</sub> fi<sub>rs</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>on o</sub>f <sub>e</sub>ith<sub>er</sub> h<sub>a</sub>lf<sub>,</sub> th<sub>e prece</sub>di<sub>ng u</sub>tilit<sub>y</sub> i<sub>s zero, an</sub>d th<sub>e same</sub> b<sub>oun</sub>d f<sub>o</sub>ll<sub>ows</sub> di<sub>rec</sub>tl<sub>y</sub> f<sub>rom s</sub>i<sub>n</sub> $( \pi / N _ { k } ) \le$ $\pi / N _ { k }$ <sub>.</sub> H<sub>ence,</sub>

$$
\begin{array} { r l r } {  { \sum _ { t = \tau _ { k } } ^ { \tau _ { k + 1 } - 1 } \| \pmb { u } ^ { ( t ) } \| _ { 2 } ^ { 2 } = \Theta ( \frac { s _ { k } ^ { 2 } } { N _ { k } } ) , } } \\ & { } & { \ \sum _ { t = \tau _ { k } } ^ { \tau _ { k + 1 } - 1 } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } = O ( \frac { s _ { k } ^ { 2 } } { N _ { k } ^ { 3 } } ) . } \end{array}\tag{63}
$$

Th<sub>ese es</sub>ti<sub>ma</sub>t<sub>es a</sub>l<sub>so</sub> h<sub>o</sub>ld <sub>w</sub>h<sub>en</sub> th<sub>e sums are res</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d t<sub>o e</sub>ith<sub>er</sub> h<sub>a</sub>lf bl<sub>oc</sub>k<sub>.</sub>

Return to the original direction and norm growth. The recurrence (55), with the prescribed coeficient at each observation, expresses the first half map as a real polynomial in J. The second half uses the same coeficients with J re<sub>p</sub>laced b<sub>y</sub> −J. Both halves be<sub>g</sub>in with e<sub>q</sub>ual re<sub>g</sub>ret and <sub>p</sub>rediction re<sub>g</sub>ret vectors because the <sub>p</sub>recedin<sub>g</sub> utilit<sub>y</sub> is zero. Thus, the <sub>p</sub>ol<sub>y</sub>nomial ar<sub>g</sub>ument leadin<sub>g</sub> to (57) a<sub>pp</sub>lies on s<sub>p</sub>an $\{ \pmb { \nu } , \pmb { e } _ { 2 } - \pmb { e } _ { 3 } \}$ <sub>an</sub>d <sub>g</sub>i<sub>ves</sub>

$$
\tilde { r } ^ { ( \tau _ { k + 1 } ) } = r ^ { ( \tau _ { k + 1 } ) } = s _ { k + 1 } \nu , \frac { s _ { k + 1 } } { s _ { k } } = \frac { \| \tilde { r } ^ { ( \tau _ { k } + N _ { k } ) } \| _ { 2 } ^ { 2 } } { s _ { k } ^ { 2 } \| \pmb { \nu } \| _ { 2 } ^ { 2 } } .
$$

Thi<sub>s</sub> d<sub>e</sub>fi<sub>nes</sub> th<sub>e nex</sub>t bl<sub>oc</sub>k’<sub>s sca</sub>l<sub>e.</sub>

The identit<sub>y</sub> (51) does not require a constant utilit<sub>y</sub> coeficient. A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> it to the first half block, whose <sub>p</sub>recedin<sub>g</sub> <sub>an</sub>d fi<sub>na</sub>l <sub>u</sub>tiliti<sub>es</sub> <sub>are</sub> <sub>zero,</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
s _ { k } \| \nu \| _ { 2 } ^ { 2 } ( s _ { k + 1 } - s _ { k } ) = ( 1 - 2 \lambda ) \sum _ { t = \tau _ { k } } ^ { \tau _ { k } + N _ { k } - 1 } \| \pmb { u } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \lambda \sum _ { t = \tau _ { k } } ^ { \tau _ { k } + N _ { k } - 1 } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } .
$$

Si<sub>nce</sub> $1 - 2 \lambda > 0 .$ <sub>,</sub> th<sub>e</sub> <sub>prece</sub>di<sub>ng</sub> b<sub>oun</sub>d<sub>s</sub> i<sub>mp</sub>l<sub>y</sub>

$$
s _ { k + 1 } - s _ { k } = \Theta \left( \frac { s _ { k } } { N _ { k } } \right) > 0 .
$$

St<sub>ar</sub>ti<sub>ng</sub> f<sub>rom</sub> $s _ { 1 } = 1$ <sub>, we</sub> th<sub>ere</sub>f<sub>ore</sub> h<sub>ave</sub> $s _ { k } \geq 1$ <sub>an</sub>d $s _ { k } \le N _ { k } \le 2 s _ { k }$ . <sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>,

$$
s _ { k + 1 } - s _ { k } = \Theta ( 1 ) , s _ { k } = \Theta ( k ) , N _ { k } = \Theta ( k ) , \tau _ { k } = \Theta ( k ^ { 2 } ) .\tag{64}
$$

F<sub>or a</sub> h<sub>or</sub>i<sub>zon</sub> � i<sub>n</sub> bl<sub>oc</sub>k �<sub>, we</sub> h<sub>ave</sub> $T = \Theta ( K ^ { 2 } )$ . Summin<sub>g</sub> the utilit<sub>y</sub> variation bound in (63) <sub>g</sub>ives

$$
\sum _ { t = 2 } ^ { T } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } \le O ( 1 ) + \sum _ { k = 1 } ^ { K } O \left( \frac { s _ { k } ^ { 2 } } { N _ { k } ^ { 3 } } \right) = O \left( \sum _ { k = 1 } ^ { K } \frac { 1 } { k } \right) = O ( \log T ) .
$$

Alternating regret. It remains to show that alternating regret grows as ${ \sqrt { T } } .$ <sub>.</sub> W<sub>e</sub> fi<sub>rs</sub>t b<sub>oun</sub>d <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>u</sub>tilit<sub>y</sub> i<sub>mprovemen</sub>t<sub>.</sub> Si<sub>nce</sub> $\langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle = 0$ <sub>,</sub> th<sub>e</sub> li<sub>near up</sub>d<sub>a</sub>t<sub>es g</sub>i<sub>ve</sub>

$$
\langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle = \frac { \| { \pmb u } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \lambda \langle { \pmb u } ^ { ( t ) } , { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } \rangle } { \| { \pmb r } ^ { ( t + 1 ) } \| _ { 1 } } .
$$

Usin<sub>g</sub> (62), Cauch<sub>y</sub>–Schwarz, and (63), we obtain

$$
\sum _ { t = \tau _ { k } } ^ { \tau _ { k + 1 } - 1 } \left| \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle \right| \leq \frac { \sum _ { t = \tau _ { k } } ^ { \tau _ { k + 1 } - 1 } \| { \pmb u } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { ( 1 - \varepsilon ) s _ { k } \| \nu \| _ { 1 } } + O \left( \frac { s _ { k } } { N _ { k } ^ { 2 } } \right) .
$$

A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> (51) over the full block also <sub>g</sub>ives

$$
\| \pmb { \nu } \| _ { 2 } ^ { 2 } ( s _ { k + 1 } ^ { 2 } - s _ { k } ^ { 2 } ) = ( 1 - 2 \lambda ) \sum _ { t = \tau _ { k } } ^ { \tau _ { k + 1 } - 1 } \| \pmb { u } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \lambda \sum _ { t = \tau _ { k } } ^ { \tau _ { k + 1 } - 1 } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } .
$$

Th<sub>e</sub> l<sub>as</sub>t <sub>sum</sub> i<sub>s</sub> <sub>nonnega</sub>ti<sub>ve.</sub> S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> <sub>upper</sub> b<sub>oun</sub>d <sub>on</sub> th<sub>e</sub> <sub>square</sub>d <sub>u</sub>tiliti<sub>es,</sub> <sub>an</sub>d <sub>us</sub>i<sub>ng</sub> $s _ { k + 1 } - s _ { k } = \Theta ( 1 )$ , <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\begin{array} { r } { \displaystyle \sum _ { t = \tau _ { k } } ^ { \tau _ { k + 1 } - 1 } \left| \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle \right| \leq \frac { \| \pmb \nu \| _ { 2 } ^ { 2 } } { ( 1 - 2 \lambda ) ( 1 - \varepsilon ) \| \pmb \nu \| _ { 1 } } \frac { s _ { k + 1 } ^ { 2 } - s _ { k } ^ { 2 } } { s _ { k } } + O \left( \frac { 1 } { s _ { k } } \right) } \\ { = \frac { 2 \| \pmb \nu \| _ { 2 } ^ { 2 } } { ( 1 - 2 \lambda ) ( 1 - \varepsilon ) \| \pmb \nu \| _ { 1 } } ( s _ { k + 1 } - s _ { k } ) + O \left( \frac { 1 } { s _ { k } } \right) . } \end{array}
$$

In particular, the sum of absolute utilit<sub>y</sub> improvements within an<sub>y</sub> one block is �(1). We can therefore sum over the <sub>comp</sub>l<sub>e</sub>t<sub>e</sub> bl<sub>oc</sub>k<sub>s an</sub>d i<sub>nc</sub>l<sub>u</sub>d<sub>e any</sub> i<sub>ncomp</sub>l<sub>e</sub>t<sub>e</sub> fi<sub>na</sub>l bl<sub>oc</sub>k t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\left| \sum _ { t = 1 } ^ { T } \langle \pmb { u } ^ { ( t ) } , \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } \rangle \right| \leq \frac { 2 \| \pmb { \nu } \| _ { 2 } ^ { 2 } } { ( 1 - 2 \lambda ) ( 1 - \varepsilon ) \| \pmb { \nu } \| _ { 1 } } s _ { K } + O ( \log K ) .\tag{65}
$$

E<sub>very</sub> <sub>p</sub>l<sub>aye</sub>d <sub>u</sub>tilit<sub>y</sub> i<sub>s</sub> <sub>zero.</sub> M<sub>oreover,</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on</sub> <sub>an</sub>d th<sub>e</sub> <sub>su</sub>b<sub>sequen</sub>t li<sub>near</sub> <sub>up</sub>d<sub>a</sub>t<sub>es</sub> i<sub>mp</sub>l<sub>y,</sub> f<sub>or</sub> $T \geq 3 .$

$$
\sum _ { t = 1 } ^ { T } \pmb { u } ^ { ( t ) } = \tilde { \pmb { r } } ^ { ( T + 1 ) } - \left( 0 , \frac { 1 } { d - 1 } , . . . , \frac { 1 } { d - 1 } \right) .
$$

Thus, for � in block �, (61) <sub>g</sub>ives

$$
( 1 - \varepsilon ) s _ { K } \leq \tilde { r } _ { 1 } ^ { ( T + 1 ) } \leq \mathrm { R e g } ( T ) \leq \| \tilde { r } ^ { ( T + 1 ) } \| _ { \infty } \leq ( 1 + \varepsilon ) s _ { K } .
$$

Combinin<sub>g</sub> this with (65) and the identit<sub>y</sub> between ordinar<sub>y</sub> and alternatin<sub>g</sub> re<sub>g</sub>ret, we obtain

$$
\begin{array} { r l } & { \operatorname { A l t - R e g } ( T ) = 2 \operatorname { R e g } ( T ) - \displaystyle \sum _ { t = 1 } ^ { T } \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle } \\ & { \qquad \quad \ge 2 \left( 1 - \varepsilon - \frac { \| { \pmb \nu } \| _ { 2 } ^ { 2 } } { ( 1 - 2 \lambda ) ( 1 - \varepsilon ) \| { \pmb \nu } \| _ { 1 } } \right) s _ { K } - O ( \log K ) . } \end{array}
$$

Th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>o</sub>f $s _ { K }$ is strictl<sub>y p</sub>ositive b<sub>y</sub> (59). Since $s _ { K } = \Theta ( K ) = \Theta ( \sqrt { T } )$ <sub>,</sub> thi<sub>s</sub> <sub>proves</sub> $\mathrm { A l t - R e g } ( T ) = \Omega ( { \sqrt { T } } )$ <sub>.</sub> Th<sub>e</sub> <sub>upper</sub> b<sub>oun</sub>d<sub>s</sub> <sub>on</sub> <sub>or</sub>di<sub>nary</sub> <sub>regre</sub>t <sub>an</sub>d <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>u</sub>tilit<sub>y</sub> i<sub>mprovemen</sub>t <sub>g</sub>i<sub>ve</sub> $\mathrm { A l t - R e g } ( T ) = O ( { \sqrt { T } } )$ <sub>as we</sub>ll<sub>.</sub>

W<sub>e</sub> h<sub>ave</sub> <sub>cons</sub>t<sub>ruc</sub>t<sub>e</sub>d <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> b<sub>oun</sub>d<sub>e</sub>d i<sub>n</sub>fi<sub>n</sub>it<sub>e</sub> <sub>u</sub>tilit<sub>y</sub> <sub>sequence</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub>

$$
\mathrm { A l t - R e g } ( T ) = \Theta ( \sqrt { T } ) , \ \sum _ { t = 2 } ^ { T } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } = O ( \log T ) .
$$

F<sub>or any</sub> h<sub>or</sub>i<sub>zon-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>cons</sub>t<sub>an</sub>t<sub>s �,</sub> ${ \it B } > 0 { \it }$ <sub>,</sub> Alt<sub>-</sub>R<sub>eg even</sub>t<sub>ua</sub>ll<sub>y excee</sub>d<sub>s</sub> $\begin{array} { r } { \alpha + \beta \sum _ { t = 2 } ^ { T } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { \gamma } ^ { 2 } } \end{array}$ <sub>.</sub> S<sub>u</sub>bt<sub>rac</sub>ti<sub>ng</sub> <sub>a nonnega</sub>ti<sub>ve s</sub>t<sub>ra</sub>t<sub>egy-movemen</sub>t t<sub>erm on</sub>l<sub>y</sub> d<sub>ecreases</sub> thi<sub>s upper</sub> b<sub>oun</sub>d<sub>.</sub> H<sub>ence,</sub> Alt<sub>-</sub>RVU f<sub>a</sub>il<sub>s on</sub> thi<sub>s sequence</sub> f<sub>or</sub> <sub>every c</sub>h<sub>o</sub>i<sub>ce o</sub>f h<sub>or</sub>i<sub>zon-</sub>i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>pos</sub>iti<sub>ve cons</sub>t<sub>an</sub>t<sub>s.</sub> □

## A.7 Proofs from Section 7

Proposition 7.2. Suppose both players in afinite two-player zero-sum normal-form game satisfy the refined Alt-RVU bound under alternating updates, with common constants �, $\beta , \gamma , \zeta > 0$ satisfying $\beta \| \mathbf { A } \| _ { 2 } ^ { 2 } \leq \gamma$ . Then, we have

$$
\sum _ { t = 1 } ^ { \infty } \mathrm { G a p } ( { \pmb x } ^ { ( t ) } , { \pmb y } ^ { ( t ) } ) ^ { 2 } \leq \left( \frac { 2 \alpha + \mathrm { r a n g e } ( \mathbf { A } ) } { \zeta } \right) ^ { 2 } .
$$

Consequently, for every $T \geq 1$ , we obtain

$$
\operatorname* { m i n } _ { 1 \leq t \leq T } \mathrm { G a p } ( { \pmb x } ^ { ( t ) } , { \pmb y } ^ { ( t ) } ) \leq \frac { 2 \alpha + \mathrm { r a n g e } ( { \bf A } ) } { \zeta \sqrt { T } } ,
$$

which implies ${ \cal O } ( T ^ { - 1 / 2 } )$ best-iterate rate, and the last iterates converge to the Nash set $z ^ { \ast }$

Proof. From the bound (9) for Alt-Reg, we can further obtain

$$
\begin{array} { r l } & { \mathrm { A l t - R e g } _ { \boldsymbol { \chi } } ( T ) + \mathrm { A l t - R e g } _ { \boldsymbol { y } } ( T ) = 2 T \mathrm { G a p } ( \bar { x } ^ { ( T ) } , \bar { y } ^ { ( T ) } ) + ( x ^ { ( 1 ) } ) ^ { \top } \mathbf { A } y ^ { ( 0 ) } - ( x ^ { ( T + 1 ) } ) ^ { \top } \mathbf { A } y ^ { ( T ) } } \\ & { \phantom { { \mathrm { A l t - R e g } } } \geq ( x ^ { ( 1 ) } ) ^ { \top } \mathbf { A } y ^ { ( 0 ) } - ( x ^ { ( T + 1 ) } ) ^ { \top } \mathbf { A } y ^ { ( T ) } } \\ & { \phantom { { \mathrm { A l t - R e g } } } \geq - \mathrm { r a n g e } ( \mathbf { A } ) , } \end{array}\tag{66}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> fi<sub>rs</sub>t i<sub>ne ua</sub>lit f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> $\mathrm { G a p } ( \bar { \pmb x } ^ { ( T ) } , \bar { \pmb y } ^ { ( T ) } ) \geq 0$ <sub>, an</sub>d th<sub>e secon</sub>d i<sub>nequa</sub>lit<sub>y</sub> h<sub>o</sub>ld<sub>s</sub> b<sub>ecause</sub> $( \mathbf { x } ^ { ( 1 ) } ) ^ { \top } \mathbf { A } \mathbf { y } ^ { ( 0 ) } \geq$ <sub>m</sub>i<sub>n</sub> $\therefore j A _ { i j }$ <sub>an</sub>d $( \mathbf { x } ^ { ( { \hat { T } } + 1 ) } ) ^ { \dagger } \mathbf { A } \mathbf { y } ^ { ( T ) } \leq \operatorname* { m a x } _ { i , j } { \dot { A } } _ { i j }$

Assume both � and � players satisfy the refined Alt-RVU bound, we can add the two bounds and absorb utility <sub>var</sub>i<sub>a</sub>ti<sub>on</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e nega</sub>ti<sub>ve s</sub>t<sub>ra</sub>t<sub>egy movemen</sub>t

$$
\begin{array} { r l } & { \zeta \sqrt { \displaystyle \sum _ { t = 1 } ^ { T } \left[ \operatorname* { m a x } _ { z \in X } \left. A \mathbf { y } ^ { ( t ) } , 2 z - \mathbf { x } ^ { ( t ) } - \mathbf { x } ^ { ( t + 1 ) } \right. \right] ^ { 2 } } + \zeta \sqrt { \displaystyle \sum _ { t = 1 } ^ { T } \left[ \operatorname* { m a x } _ { z \in y } \left. - A ^ { \top } \mathbf { x } ^ { ( t ) } , 2 z - \mathbf { y } ^ { ( t - 1 ) } - \mathbf { y } ^ { ( t ) } \right. \right] ^ { 2 } } } \\ & { \leq 2 \alpha - \mathrm { A l t } \mathrm { - } \mathrm { R e } _ { \mathbf { g } _ { X } } ( T ) - \mathrm { A l t } \mathrm { - } \mathrm { R e } _ { \mathbf { g } _ { Y } } ( T ) } \\ & { \overset { ( 6 6 ) } { \leq } 2 \alpha + \mathrm { r a n g e } ( \mathbf { A } ) . } \end{array}
$$

Combinin<sub>g</sub> this inequalit<sub>y</sub> with the <sub>p</sub>ointwise u<sub>pp</sub>er bound on the Nash <sub>g</sub>a<sub>p</sub> <sub>p</sub>recedin<sub>g</sub> (27) and a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the trian<sub>g</sub>le i<sub>nequa</sub>lit<sub>y,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } & { \sqrt { \displaystyle \sum _ { t = 1 } ^ { T } \mathrm { G a p } ( x ^ { ( t ) } , \mathbf { y } ^ { ( t ) } ) ^ { 2 } } } \\ &  \le \sqrt { \displaystyle \sum _ { t = 1 } ^ { T } \left[ \frac { \operatorname* { m a x } \left( \mathbf { A } \mathbf { y } ^ { ( t ) } , 2 z - x ^ { ( t ) } - x ^ { ( t + 1 ) } \right) + \operatorname* { m a x } \left\{ - \mathbf { A } ^ { \top } \mathbf { x } ^ { ( t ) } , 2 z - y ^ { ( t - 1 ) } - \mathbf { y } ^ { ( t ) } \right\}  ^ { 2 } } { z \displaystyle \sum _ { t = 1 } ^ { T } \left[ \frac { \operatorname* { m a x } \left\{ \mathbf { A } \mathbf { y } ^ { ( t ) } , 2 z - x ^ { ( t ) } - x ^ { ( t + 1 ) } \right\}  ^ { 2 } } { z \displaystyle + \sqrt { \displaystyle \sum _ { t = 1 } ^ { T } \left[ \frac { \operatorname* { m a x } \left\{ - \mathbf { A } ^ { \top } \mathbf { x } ^ { ( t ) } , 2 z - \mathbf { y } ^ { ( t - 1 ) } - \mathbf { y } ^ { ( t ) } \right\}  ^ { 2 } } } } } }\right] \\ &  \le\right] \sqrt  \right]\displaystyle \sum _ { t = 1 } ^ { T } \left[ \frac { \operatorname* { m a x } \left\{ \mathbf { A } \mathbf { y } ^ { ( t ) } , 2 z - \mathbf { x } ^ { ( t ) } - x ^ { ( t + 1 ) } \right\} \right] ^ { 2 } } + \sqrt { \displaystyle \sum _ { t = 1 } ^ { T } \left[ \frac { \operatorname* { m a x } \left\{ - \mathbf { A } ^ { \top } \mathbf { x } ^ { ( t ) } , 2 z - \mathbf { y } ^ { ( t - 1 ) } - \mathbf { y } ^ { ( t ) } \right\}  ^ { 2 } } { z \displaystyle \sum _ { t = 1 } ^ { T } \left[ \frac { \operatorname* { m a x } \left\{ - \mathbf { A } ^ { \top } \mathbf { x } ^ { ( t ) } , 2 z - \mathbf { y } ^ { ( t - 1 ) } - \mathbf { y } ^ { ( t ) } \right\}  ^ { 2 } } } } \\ \right]& \right] \end{array}
$$

C<sub>onsequen</sub>tl<sub>y,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\operatorname* { m i n } _ { 1 \leq t \leq T } \mathrm { G a p } ( { \pmb x } ^ { ( t ) } , { \pmb y } ^ { ( t ) } ) \leq \left( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathrm { G a p } ( { \pmb x } ^ { ( t ) } , { \pmb y } ^ { ( t ) } ) ^ { 2 } \right) ^ { 1 / 2 } \leq \frac { 2 \alpha + \mathrm { r a n g e } ( \mathbf { A } ) } { \zeta \sqrt { T } } ,
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>mp</sub>li<sub>es an</sub> ${ \cal O } ( T ^ { - 1 / 2 } )$ b<sub>es</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e ra</sub>t<sub>e.</sub> R<sub>e</sub>t<sub>urn</sub>i<sub>ng</sub> t<sub>o</sub> th<sub>e</sub> b<sub>oun</sub>d <sub>on</sub> $\sqrt { \textstyle \sum _ { t = 1 } ^ { T } \mathbf { G a p } ( \pmb { x } ^ { ( t ) } , \pmb { y } ^ { ( t ) } ) ^ { 2 } }$ <sub>, squar</sub>i<sub>ng</sub> b<sub>o</sub>th <sub>s</sub>id<sub>es</sub> <sub>an</sub>d l<sub>e</sub>tti<sub>ng</sub> $T \to \infty$ <sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\sum _ { t = 1 } ^ { \infty } \operatorname { G a p } ( \pmb { x } ^ { ( t ) } , \pmb { y } ^ { ( t ) } ) ^ { 2 } \leq \left( \frac { 2 \alpha + \mathrm { r a n g e } ( \mathbf { A } ) } { \zeta } \right) ^ { 2 } < \infty ,
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>mp</sub>li<sub>es</sub> $\mathrm { G a p } ( \mathbf { \boldsymbol { x } } ^ { ( t ) } , \mathbf { \boldsymbol { y } } ^ { ( t ) } ) \to 0$ <sub>.</sub> Si<sub>nce</sub> th<sub>e s</sub>t<sub>ra</sub>t<sub>egy space</sub> i<sub>s compac</sub>t <sub>an</sub>d th<sub>e</sub> N<sub>as</sub>h <sub>gap</sub> i<sub>s con</sub>ti<sub>nuous an</sub>d <sub>van</sub>i<sub>s</sub>h<sub>es</sub> <sub>exac</sub>tl<sub>y a</sub>t N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>a,</sub> th<sub>e</sub> l<sub>as</sub>t it<sub>era</sub>t<sub>es converge</sub> t<sub>o</sub> th<sub>e</sub> N<sub>as</sub>h <sub>se</sub>t<sub>.</sub> □

Lemma 7.3. IR-PRM<sup>+</sup> and HPRM<sup>+</sup> learners satisfy,for every � ≥ 1, $t \geq 1$

$$
\operatorname* { m a x } _ { z \in \Delta ( d ) } \langle { \pmb u } ^ { ( t ) } , 2 z - { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t + 1 ) } \rangle \leq \sqrt { 5 } \| \widetilde { \pmb r } ^ { ( t + 1 ) } - \widetilde { \pmb r } ^ { ( t ) } \| _ { 2 } .\tag{29}
$$

For IR-PRM<sup>+</sup>, the constant $\sqrt { 5 }$ can be improved to $2 .$

Proof. For $\tt T R { - } P R M ^ { + }$ , the re<sub>g</sub>ret vector u<sub>p</sub>date (7), rewritten usin<sub>g</sub> $c ^ { ( t ) }$ , <sub>g</sub><sup>i</sup>ves

$$
\operatorname* { m a x } _ { z \in \Delta ( d ) } \langle { \pmb u } ^ { ( t ) } , z - { \pmb x } ^ { ( t ) } \rangle + c ^ { ( t ) } \leq \operatorname* { m a x } _ { z \in \Delta ( d ) } \langle \tilde { { \pmb r } } ^ { ( t + 1 ) } - \tilde { { \pmb r } } ^ { ( t ) } , z \rangle \leq \| \tilde { { \pmb r } } ^ { ( t + 1 ) } - \tilde { { \pmb r } } ^ { ( t ) } \| _ { 2 } .
$$

Combinin<sub>g</sub> this with (18), we obtain

$$
\begin{array} { r l } & { \displaystyle \operatorname* { m a x } _ { z \in \Delta ( d ) } \langle { \pmb u } ^ { ( t ) } , 2 z - { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t + 1 ) } \rangle = 2 \ \displaystyle \operatorname* { m a x } _ { z \in \Delta ( d ) } \langle { \pmb u } ^ { ( t ) } , { z } - { \pmb x } ^ { ( t ) } \rangle - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle } \\ & { \qquad \le 2 \| \tilde { \pmb r } ^ { ( t + 1 ) } - \tilde { \pmb r } ^ { ( t ) } \| _ { 2 } - c ^ { ( t ) } - c ^ { ( t + 1 ) } } \\ & { \qquad \le 2 \| \tilde { \pmb r } ^ { ( t + 1 ) } - \tilde { \pmb r } ^ { ( t ) } \| _ { 2 } , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> <sub>uses</sub> $c ^ { ( t ) } , c ^ { ( t + 1 ) } \geq 0 .$

For HPRM<sup>+</sup>, the coordinatewise ine<sub>q</sub>ualit<sub>y</sub> (11) remains valid because its re<sub>g</sub>ret vector u<sub>p</sub>date is unchan<sub>g</sub>ed. Su<sub>pp</sub>ose fi<sub>rs</sub>t th<sub>a</sub>t $[ \boldsymbol { r } ^ { ( t + 1 ) } ] _ { + } \neq \mathbf { 0 }$ <sub>.</sub> Th<sub>e pre</sub>di<sub>c</sub>ti<sub>on</sub> id<sub>en</sub>tit<sub>y es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d i<sub>n</sub> th<sub>e proo</sub>f <sub>o</sub>f L<sub>emma</sub> 5<sub>.</sub>10 i<sub>mp</sub>li<sub>es</sub>

$$
\tilde { r } _ { i } ^ { ( t + 1 ) } = 0 \implies [ \boldsymbol { r } _ { i } ^ { ( t + 1 ) } ] _ { + } = \left[ - \frac { 1 } { 2 } \tilde { r } _ { i } ^ { ( t ) } \right] _ { + } = 0 .
$$

Th<sub>us,</sub> th<sub>e regre</sub>t <sub>vec</sub>t<sub>or up</sub>d<sub>a</sub>t<sub>e</sub> i<sub>s un</sub>t<sub>runca</sub>t<sub>e</sub>d <sub>on</sub> th<sub>e suppor</sub>t <sub>o</sub>f $\boldsymbol { x } ^ { ( t + 1 ) }$ <sub>, an</sub>d <sub>we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r } { \langle { { \pmb u } ^ { ( t ) } } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle = \langle { { \pmb u } ^ { ( t ) } } - \langle { { \pmb u } ^ { ( t ) } } , { \pmb x } ^ { ( t ) } \rangle { \pmb 1 } , { \pmb x } ^ { ( t + 1 ) } \rangle = \langle \widetilde { { \pmb r } } ^ { ( t + 1 ) } - \widetilde { { \pmb r } } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } \rangle . } \end{array}
$$

Usin<sub>g</sub> this identit<sub>y</sub> and (11), we <sub>g</sub>et

$$
\operatorname* { m a x } _ { z \in \Delta ( d ) } \langle { \boldsymbol u } ^ { ( t ) } , 2 z - { \boldsymbol x } ^ { ( t ) } - { \boldsymbol x } ^ { ( t + 1 ) } \rangle \le \operatorname* { m a x } _ { z \in \Delta ( d ) } \langle \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } , 2 z - { \boldsymbol x } ^ { ( t + 1 ) } \rangle \le \sqrt { 5 } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ,
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> C<sub>auc</sub>h<sub>y-</sub>S<sub>c</sub>h<sub>warz</sub> <sub>an</sub>d

$$
\| 2 z - { \pmb x } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } = 4 \| z \| _ { 2 } ^ { 2 } + \| { \pmb x } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - 4 \langle z , { \pmb x } ^ { ( t + 1 ) } \rangle \leq 5 .
$$

Fi<sub>na</sub>ll<sub>y,</sub> if $[ \boldsymbol { r } ^ { ( t + 1 ) } ] _ { + } = \mathbf { 0 }$ , (43) im<sub>p</sub>lies that activation has not occurred. Hence $\tilde { r } ^ { ( t ) } = \tilde { r } ^ { ( t + 1 ) } = \mathbf { 0 } ;$ <sub>un</sub>if<sub>orm</sub> <sub>p</sub>l<sub>ay</sub> <sub>an</sub>d th<sub>e</sub> re<sub>g</sub>ret vector u<sub>p</sub>date force the instantaneous re<sub>g</sub>ret vector to be zero. Both sides of (29) then vanish. □

Proposition 7.4. Both IR-PRM<sup>+</sup> and HPRM<sup>+</sup> satisfy refined Alt-RVU. Specifically, for every bounded utility sequence, there exist $\gamma , \zeta > 0$ such that,for every $\beta > 0 ,$ , there exists � $\prime > 0 f o r$ which (28) holdsfor every $T \geq 1$ . These constants are independent of�.

Proof. If $t _ { 0 } = \infty$ , the instantaneous re<sub>g</sub>ret vectors and strate<sub>gy</sub> movements are zero, so (28) holds with an<sub>y</sub> <sub>p</sub>ositive <sub>cons</sub>t<sub>an</sub>t<sub>s.</sub> Oth<sub>erw</sub>i<sub>se,</sub> it <sub>su</sub>fi<sub>ces</sub> t<sub>o cons</sub>id<sub>er</sub> $T \geq t _ { 0 } - 1$ <sub>, s</sub>i<sub>nce a</sub>lt<sub>erna</sub>ti<sub>ng regre</sub>t <sub>an</sub>d th<sub>e</sub> b<sub>es</sub>t<sub>-response errors van</sub>i<sub>s</sub>h f<sub>or</sub> <sub>ear</sub>li<sub>er</sub> h<sub>or</sub>i<sub>zons.</sub>

IR-PRM<sup>+</sup>. We first retain the terms involvin<sub>g</sub> $c ^ { ( t ) }$ i<sub>n</sub> th<sub>e a</sub>lt<sub>erna</sub>ti<sub>ng-regre</sub>t <sub>ana</sub>l<sub>ys</sub>i<sub>s.</sub> Th<sub>e coor</sub>di<sub>na</sub>t<sub>ew</sub>i<sub>se regre</sub>t <sub>vec</sub>t<sub>or</sub> u<sub>p</sub><sup>d</sup>ate <sub>g</sub><sup>i</sup>ves, <sup>f</sup>or ever<sub>y</sub> $z \in \Delta ( d )$ ,

$$
\sum _ { t = 1 } ^ { T } \langle { \pmb u } ^ { ( t ) } , z - { \pmb x } ^ { ( t ) } \rangle \leq \langle \tilde { { \pmb r } } ^ { ( T + 1 ) } , z \rangle - \sum _ { t = 1 } ^ { T } c ^ { ( t ) } .
$$

U<sub>s</sub>i<sub>ng</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y</sub> b<sub>e</sub>t<sub>ween</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>regre</sub>t <sub>an</sub>d <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>u</sub>tilit<sub>y</sub> i<sub>mprovemen</sub>t<sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\mathrm { A l t - R e g } ( T ) \le 2 \| \tilde { { \boldsymbol r } } ^ { ( T + 1 ) } \| _ { 2 } - 2 \sum _ { t = 1 } ^ { T } c ^ { ( t ) } - \sum _ { t = 1 } ^ { T } \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \rangle .
$$

S<sub>e</sub>t $\gamma = \lVert \tilde { r } ^ { ( t _ { 0 } ) } \rVert _ { 2 } / ( 4 d ) > 0$ . Summin<sub>g</sub> (18), usin<sub>g</sub> the nondecreasin<sub>g</sub> re<sub>g</sub>ret-vector norms and $c ^ { ( 1 ) } = 0 .$ , <sub>g</sub><sup>i</sup>ves

$$
\sum _ { t = 1 } ^ { T } \langle \pmb { u } ^ { ( t ) } , \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } \rangle \geq \gamma \sum _ { t = 1 } ^ { T } \| \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } \| _ { 2 } ^ { 2 } + c ^ { ( T + 1 ) } .
$$

C<sub>onsequen</sub>tl<sub>y,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\mathrm { A l t - R e g } ( T ) \le 2 \| \tilde { r } ^ { ( T + 1 ) } \| _ { 2 } - 2 \sum _ { t = 1 } ^ { T } c ^ { ( t ) } - \gamma \sum _ { t = 1 } ^ { T } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } .\tag{67}
$$

W<sub>e</sub> <sub>nex</sub>t <sub>use</sub> th<sub>e</sub> <sub>re</sub>t<sub>a</sub>i<sub>ne</sub>d <sub>sum</sub> <sub>o</sub>f $c ^ { ( t ) }$ t<sub>o</sub> <sub>con</sub>t<sub>ro</sub>l th<sub>e</sub> <sub>a</sub>dditi<sub>ona</sub>l b<sub>es</sub>t<sub>-response</sub> <sub>errors.</sub> N<sub>orm</sub> <sub>preserva</sub>ti<sub>on</sub> <sub>an</sub>d th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f $c ^ { ( t ) }$ <sup>i</sup>m<sub>p</sub><sup>l</sup><sub>y</sub>

$$
c ^ { ( t ) } \| [ \pmb { r } ^ { ( t ) } ] _ { + } \| _ { 1 } = \| \tilde { \pmb { r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \langle \tilde { \pmb { r } } ^ { ( t ) } , [ \pmb { r } ^ { ( t ) } ] _ { + } \rangle = \frac { 1 } { 2 } \| [ \pmb { r } ^ { ( t ) } ] _ { + } - \tilde { \pmb { r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } .
$$

M<sub>oreover,</sub> $[ { \pmb r } ^ { ( t ) } ] _ { + } = \| [ { \pmb r } ^ { ( t ) } ] _ { + } \| _ { 1 } { \pmb x } ^ { ( t ) }$ <sub>g</sub><sup>i</sup>ves

$$
\begin{array} { r } { \langle { \pmb u } ^ { ( t ) } - \langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle { \pmb 1 } , [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle = 0 . } \end{array}
$$

T<sub>a</sub>ki<sub>ng</sub> th<sub>e</sub> i<sub>nner</sub> <sub>pro</sub>d<sub>uc</sub>t <sub>o</sub>f th<sub>e</sub> <sub>coor</sub>di<sub>na</sub>t<sub>ew</sub>i<sub>se</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> <sub>w</sub>ith $[ r ^ { ( t ) } ] _ { - }$ <sub>+,</sub> <sub>we</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r } { \langle \tilde { { \pmb r } } ^ { ( t + 1 ) } , [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle \geq \langle \tilde { { \pmb r } } ^ { ( t ) } , [ { \pmb r } ^ { ( t ) } ] _ { + } \rangle + c ^ { ( t ) } \| [ { \pmb r } ^ { ( t ) } ] _ { + } \| _ { 1 } = \| [ { \pmb r } ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } . } \end{array}
$$

E<sub>xpa</sub>ndin<sub>g</sub> th<sub>e</sub> <sub>squa</sub>r<sub>e</sub> <sub>a</sub>nd <sub>us</sub>in<sub>g</sub> n<sub>o</sub>rm <sub>p</sub>r<sub>ese</sub>r<sub>va</sub>ti<sub>o</sub>n<sub>,</sub> <sub>we</sub> <sub>ge</sub>t

$$
\| \widetilde { \pmb { r } } ^ { ( t + 1 ) } - [ \pmb { r } ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } \leq \| \widetilde { \pmb { r } } ^ { ( t + 1 ) } \| _ { 2 } ^ { 2 } - \| \widetilde { \pmb { r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } .
$$

W<sub>e now</sub> d<sub>ecompose eac</sub>h <sub>regre</sub>t<sub>-vec</sub>t<sub>or</sub> i<sub>ncremen</sub>t <sub>aroun</sub>d $[ \pmb { r } ^ { ( t ) } ]$ <sub>+ an</sub>d <sub>sum</sub> t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\sum _ { t = 1 } ^ { T } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } \leq 2 \sum _ { t = 1 } ^ { T } \| \tilde { r } ^ { ( t + 1 ) } - [ r ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } + 2 \sum _ { t = 1 } ^ { T } \| [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } .
$$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>prece</sub>di<sub>ng</sub> b<sub>oun</sub>d<sub>s</sub> <sub>an</sub>d <sub>us</sub>i<sub>ng</sub> $\tilde { r } ^ { ( 1 ) } = \mathbf { 0 }$ <sub>, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\sum _ { t = 1 } ^ { T } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } \leq 2 \| \tilde { r } ^ { ( T + 1 ) } \| _ { 2 } ^ { 2 } + 4 \sum _ { t = 1 } ^ { T } c ^ { ( t ) } \| [ r ^ { ( t ) } ] _ { + } \| _ { 1 } \leq 2 \| \tilde { r } ^ { ( T + 1 ) } \| _ { 2 } ^ { 2 } + 4 \sqrt { d } \| \tilde { r } ^ { ( T + 1 ) } \| _ { 2 } \sum _ { t = 1 } ^ { T } c ^ { ( t ) } .
$$

Th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> $\| [ \pmb { r } ^ { ( t ) } ] _ { + } \| _ { 1 } \leq \sqrt { d } \| \widetilde { \pmb { r } } ^ { ( t ) } \| _ { 2 }$ <sub>an</sub>d th<sub>e</sub> <sub>non</sub>d<sub>ecreas</sub>i<sub>ng</sub> <sub>regre</sub>t<sub>-vec</sub>t<sub>or</sub> <sub>norms.</sub>

A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Lemma 7.3 with its constant 2 for IR-PRM<sup>+</sup>, we <sub>g</sub>et

$$
\sqrt { \sum _ { t = 1 } ^ { T } \left[ \operatorname* { m a x } _ { z \in \Delta ( d ) } \langle { \pmb u } ^ { ( t ) } , 2 z - { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t + 1 ) } \rangle \right] ^ { 2 } } \leq 2 \sqrt { 2 } \| \widetilde { r } ^ { ( T + 1 ) } \| _ { 2 } + 4 d ^ { 1 / 4 } \sqrt { \| \widetilde { r } ^ { ( T + 1 ) } \| _ { 2 } \sum _ { t = 1 } ^ { T } c ^ { ( t ) } } .
$$

Y<sub>oung</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>g</sub>i<sub>ves</sub>

$$
4 d ^ { 1 / 4 } \sqrt { \| \tilde { r } ^ { ( T + 1 ) } \| _ { 2 } \sum _ { t = 1 } ^ { T } c ^ { ( t ) } } \leq 4 \sqrt { d } \| \tilde { r } ^ { ( T + 1 ) } \| _ { 2 } + \sum _ { t = 1 } ^ { T } c ^ { ( t ) } .
$$

Addin<sub>g</sub> these bounds to (67) and discardin<sub>g</sub> the remainin<sub>g</sub> ne<sub>g</sub>ative sum of $c ^ { ( t ) }$ <sub>,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\mathrm { A l t . R e g } ( T ) + \sqrt { \sum _ { t = 1 } ^ { T } \left[ \operatorname* { m a x } _ { z \in \Lambda ( d ) } \left. u ^ { ( t ) } , 2 z - x ^ { ( t ) } - x ^ { ( t + 1 ) } \right. \right] ^ { 2 } } \leq ( 2 + 2 \sqrt { 2 } + 4 \sqrt { d } ) \| \tilde { r } ^ { ( T + 1 ) } \| _ { 2 } - \gamma \sum _ { t = 1 } ^ { T } \| x ^ { ( t + 1 ) } - x ^ { ( t ) } \| _ { 2 } ^ { 2 } .
$$

Th<sub>e</sub> <sub>norm</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f P<sub>ropos</sub>iti<sub>on</sub> 5<sub>.</sub>9 <sub>g</sub>i<sub>ves</sub>

$$
\| \tilde { \pmb { r } } ^ { ( T + 1 ) } \| _ { 2 } \leq \| \mathbf { A } \| _ { 2 } + ( 1 + \sqrt { d } ) \sqrt { \sum _ { t = 2 } ^ { T } \| \pmb { u } ^ { ( t ) } - \pmb { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } } .
$$

Substitutin<sub>g</sub> this estimate and a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> Youn<sub>g</sub>’s inequalit<sub>y p</sub>roves (28) with

$$
\zeta = 1 , \mathrm { ~ a n d ~ } \alpha = ( 2 + 2 { \sqrt { 2 } } + 4 { \sqrt { d } } ) \| \mathbf { A } \| _ { 2 } + { \frac { ( 2 + 2 { \sqrt { 2 } } + 4 { \sqrt { d } } ) ^ { 2 } ( 1 + { \sqrt { d } } ) ^ { 2 } } { 4 \beta } } .
$$

HPRM<sup>+</sup>. For this learner, we retain <sub>p</sub>art of the ne<sub>g</sub>ative wei<sub>g</sub>hted re<sub>g</sub>ret-vector increment sum in (44). The <sub>p</sub>rediction b<sub>oun</sub>d f<sub>rom</sub> th<sub>e</sub> <sub>prece</sub>di<sub>ng</sub> Alt<sub>-</sub>RVU <sub>proo</sub>f <sub>g</sub>i<sub>ves</sub>

$$
\| [ \pmb { r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } \leq 2 \sqrt { d } \operatorname* { m a x } _ { 1 \leq s \leq T + 1 } \| \tilde { \pmb { r } } ^ { ( s ) } \| _ { 2 } .
$$

Si<sub>nce</sub> th<sub>e regre</sub>t<sub>-vec</sub>t<sub>or</sub> i<sub>ncremen</sub>t<sub>s van</sub>i<sub>s</sub>h b<sub>e</sub>f<sub>ore</sub> $t _ { 0 } - 1$ <sub>, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\sum _ { t = 1 } ^ { T } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } \leq 2 \sqrt { d } \left( \operatorname* { m a x } _ { 1 \leq s \leq T + 1 } \| \tilde { r } ^ { ( s ) } \| _ { 2 } \right) \sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } .
$$

S<sub>e</sub>t $\zeta = 1 / ( 4 \sqrt { 5 } d ^ { 1 / 4 } )$ <sub>.</sub> A<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> L<sub>e</sub>mm<sub>a</sub> 7<sub>.</sub>3 <sub>a</sub>nd Y<sub>ou</sub>n<sub>g</sub>’<sub>s</sub> in<sub>equa</sub>lit<sub>y, we ge</sub>t

$$
\zeta \sqrt { \sum _ { t = 1 } ^ { T } \left[ \operatorname* { m a x } _ { z \in \Delta ( d ) } \langle { \pmb u } ^ { ( t ) } , 2 z - { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t + 1 ) } \rangle \right] ^ { 2 } } \le \operatorname* { m a x } _ { 1 \le s \le T + 1 } \| \widetilde { \pmb r } ^ { ( s ) } \| _ { 2 } + \frac { 1 } { 3 2 } \sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { \| \widetilde { \pmb r } ^ { ( t + 1 ) } - \widetilde { \pmb r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| [ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } .\tag{68}
$$

Recall that (44) <sub>g</sub>ives

$$
\mathrm { A l t - R e g } ( T ) \leq 4 \operatorname* { m a x } _ { 1 \leq s \leq T + 1 } \| \tilde { r } ^ { ( s ) } \| _ { 2 } - \frac { 1 } { 4 } \sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } .
$$

Addin<sub>g</sub> (68), we obtain

$$
\mathrm { A l t - R e g } ( T ) + \zeta \sqrt { \sum _ { t = 1 } ^ { T } \left[ \operatorname* { m a x } _ { z \in \Delta ( d ) } \langle u ^ { ( t ) } , 2 z - x ^ { ( t ) } - x ^ { ( t + 1 ) } \rangle \right] ^ { 2 } } \leq 5 \operatorname* { m a x } _ { 1 \leq s \leq T + 1 } \| \tilde { r } ^ { ( s ) } \| _ { 2 } - \frac { 7 } { 3 2 } \sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } .
$$

Th<sub>e</sub> <sub>max</sub>i<sub>mum-norm</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> <sub>prece</sub>di<sub>ng</sub> Alt<sub>-</sub>RVU <sub>proo</sub>f <sub>g</sub>i<sub>ves</sub>

$$
\begin{array} { l } { \displaystyle \underset { 1 \leq s \leq T + 1 } { \operatorname* { m a x } } \| \tilde { r } ^ { ( s ) } \| _ { 2 } \leq 2 \| \mathbf { A } \| _ { 2 } + 6 d ^ { 1 / 4 } \left( 2 ( 1 + \sqrt { d } ) ^ { 2 } \displaystyle \sum _ { t = 2 } ^ { T } \| { \boldsymbol u } ^ { ( t ) } - { \boldsymbol u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + 2 d \| \mathbf { A } \| _ { 2 } ^ { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \| { \boldsymbol x } ^ { ( t + 1 ) } - { \boldsymbol x } ^ { ( t ) } \| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } } \\ { + \displaystyle \frac { 1 } { 3 2 } \displaystyle \sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| \left[ r ^ { ( t + 1 ) } \right] _ { + } \| _ { 1 } } . } \end{array}
$$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> thi<sub>s es</sub>ti<sub>ma</sub>t<sub>e,</sub> th<sub>e coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>o</sub>f th<sub>e we</sub>i<sub>g</sub>ht<sub>e</sub>d i<sub>ncremen</sub>t <sub>sum</sub> b<sub>ecomes</sub> $5 / 3 2 - 7 / 3 2 = - 1 / 1 6$ <sub>.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } & { \displaystyle \mathrm { A l t - R e g } ( T ) + \zeta \sqrt { \displaystyle \sum _ { t = 1 } ^ { T } \left[ \operatorname* { m a x } _ { z \in \Delta ( d ) } \langle u ^ { ( t ) } , 2 z - x ^ { ( t ) } - x ^ { ( t + 1 ) } \rangle \right] ^ { 2 } } } \\ & { \displaystyle \le 1 0 \| { \bf A } \| _ { 2 } + 3 0 d ^ { 1 / 4 } \left( 2 ( 1 + \sqrt { d } ) ^ { 2 } \displaystyle \sum _ { t = 2 } ^ { T } \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + 2 d \| { \bf A } \| _ { 2 } ^ { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } } \\ & { \displaystyle - \frac { 1 } { 1 6 } \sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| { \left[ \boldsymbol r ^ { ( t + 1 ) } \right] } + \| _ { 1 } } . } \end{array}
$$

Ch<sub>oose</sub> $\gamma = \sqrt { f ^ { ( t _ { 0 } ) } } / ( 1 2 8 \sqrt { 2 } ( 1 + \sqrt { d } ) ^ { 2 } ) > 0$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> h<sub>a</sub>lf th<sub>e</sub> <sub>movemen</sub>t <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t i<sub>n</sub> th<sub>e</sub> <sub>prece</sub>di<sub>ng</sub> Alt<sub>-</sub>RVU <sub>p</sub>ro<sub>p</sub>osition. With this choice, (45) reads

$$
\sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { \Vert \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \Vert _ { 2 } ^ { 2 } } { \Vert [ r ^ { ( t + 1 ) } ] _ { + } \Vert _ { 1 } } \geq 3 2 \gamma \sum _ { t = 1 } ^ { T } \Vert { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \Vert _ { 2 } ^ { 2 } - 6 4 \gamma .
$$

C<sub>onsequen</sub>tl<sub>y,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { l l } & { \displaystyle \mathrm { A l t - R e g } ( T ) + \zeta \sqrt { \displaystyle \sum _ { t = 1 } ^ { T } \left[ \operatorname* { m a x } _ { \tau \in \Delta ( d ) } \langle { \pmb u } ^ { ( t ) } , 2 z - { \pmb x } ^ { ( t ) } - { \pmb x } ^ { ( t + 1 ) } \rangle \right] ^ { 2 } } } \\ & { \displaystyle \le 1 0 \| { \pmb \mathrm { A } } \| _ { 2 } + 4 \gamma + 3 0 d ^ { 1 / 4 } \left( 2 ( 1 + \sqrt { d } ) ^ { 2 } \displaystyle \sum _ { t = 2 } ^ { T } \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + 2 d \| { \pmb \mathrm { A } } \| _ { 2 } ^ { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } } \\ & { \displaystyle - 2 \gamma \displaystyle \sum _ { t = 1 } ^ { T } \| { \pmb x } ^ { ( t + 1 ) } - { \pmb x } ^ { ( t ) } \| _ { 2 } ^ { 2 } . } \end{array}
$$

S<sub>p</sub>litti<sub>ng</sub> th<sub>e</sub> <sub>square</sub> <sub>roo</sub>t <sub>an</sub>d <sub>app</sub>l<sub>y</sub>i<sub>ng</sub> Y<sub>oung</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> t<sub>o</sub> <sub>eac</sub>h t<sub>erm,</sub> <sub>we</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\mathrm { A l t \mathrm { \mathrm { \mathrm { R e g } } } } ( T ) + \zeta \sqrt { \sum _ { t = 1 } ^ { T } \left[ \operatorname* { m a x } _ { z \in \Delta ( d ) } \left. \boldsymbol { u } ^ { ( t ) } , 2 { z } - \boldsymbol { x } ^ { ( t ) } - \boldsymbol { x } ^ { ( t + 1 ) } \right. \right] ^ { 2 } } \leq \alpha + \beta \sum _ { t = 2 } ^ { T } \| \boldsymbol { u } ^ { ( t ) } - \boldsymbol { u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } - \gamma \sum _ { t = 1 } ^ { T } \| \boldsymbol { x } ^ { ( t + 1 ) } - \boldsymbol { x } ^ { ( t ) } \| _ { 2 } ^ { 2 } ,
$$

<sub>w</sub>h<sub>ere</sub>

$$
\alpha = 1 0 \| \mathbf { A } \| _ { 2 } + 4 \gamma + \frac { 4 5 0 \sqrt { d } ( 1 + \sqrt { d } ) ^ { 2 } } { \beta } + \frac { 4 5 0 d ^ { 3 / 2 } \| \mathbf { A } \| _ { 2 } ^ { 2 } } { \gamma } .
$$

F<sub>or</sub> b<sub>o</sub>th l<sub>earners,</sub> $\gamma$ <sub>an</sub>d $\zeta$ <sub>are</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>an</sub>d i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f $\beta ,$ <sub>an</sub>d <sub>a</sub>ll <sub>cons</sub>t<sub>an</sub>t<sub>s</sub> <sub>are</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f $T .$ □

Theorem 7.5. In every finite two-player zero-sum normal-form game, $_ { A } I t - I R - P R M ^ { + }$ and Alt-HPRM<sup>+</sup> satisfy

$$
\operatorname* { m i n } _ { 1 \leq t \leq T } { \mathrm { G a p } ( { \pmb x } ^ { ( t ) } , { \pmb y } ^ { ( t ) } ) } = O ( T ^ { - 1 / 2 } ) a n d ( { \pmb x } ^ { ( t ) } , { \pmb y } ^ { ( t ) } )  ( { \pmb x } ^ { * } , { \pmb y } ^ { * } ) ,
$$

where $( \pmb { x } ^ { * } , \pmb { y } ^ { * } ) \in \mathcal { Z } ^ { * }$ is a Nash equilibrium.

Proof. Applying Proposition 7.2 to the bounds established in Proposition 7.4 gives the best-iterate bound and $\mathrm { G a p } ( \mathbf { x } ^ { ( t ) } , \mathbf { y } ^ { ( t ) } ) \to 0$ <sub>.</sub> W<sub>e now prove convergence</sub> t<sub>o one equ</sub>ilib<sub>r</sub>i<sub>um</sub> f<sub>or</sub> b<sub>o</sub>th <sub>a</sub>l<sub>gor</sub>ith<sub>ms.</sub>

B<sub>ecause</sub> th<sub>e prece</sub>di<sub>ng</sub> Alt<sub>-</sub>RVU b<sub>oun</sub>d<sub>s</sub> h<sub>o</sub>ld f<sub>or every</sub> $\beta > 0$ <sub>, we c</sub>h<sub>oose</sub> $\beta$ <sub>so</sub> th<sub>a</sub>t $\beta \| \mathbf { A } \| _ { 2 } ^ { 2 }$ i<sub>s s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y sma</sub>ll<sub>er</sub> th<sub>an</sub> both <sub>p</sub>la<sub>y</sub>ers’ ne<sub>g</sub>ative movement coeficients. Summin<sub>g</sub> the two bounds and usin<sub>g</sub> (66), we obtain

$$
\sum _ { t = 1 } ^ { \infty } \left( \| \pmb { x } ^ { ( t + 1 ) } - \pmb { x } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \| \pmb { y } ^ { ( t ) } - \pmb { y } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } \right) < \infty .
$$

Si<sub>nce</sub> <sub>u</sub>tiliti<sub>es</sub> <sub>are</sub> li<sub>near</sub> i<sub>n</sub> th<sub>e</sub> <sub>opponen</sub>t’<sub>s</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy,</sub> b<sub>o</sub>th <sub>u</sub>tilit<sub>y-var</sub>i<sub>a</sub>ti<sub>on</sub> <sub>sums</sub> <sub>are</sub> <sub>a</sub>l<sub>so</sub> fi<sub>n</sub>it<sub>e.</sub>

We first establish the re<sub>q</sub>uired bounds on the re<sub>g</sub>ret vectors. For IR-PRM<sup>+</sup>, the norm estimate in the <sub>p</sub>roof of P<sub>ropos</sub>iti<sub>on</sub> 5<sub>.</sub>9 <sub>g</sub>i<sub>ves</sub>

$$
\operatorname* { s u p } _ { t \geq 1 } \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } \leq \| \mathbf { A } \| _ { 2 } ^ { 2 } + ( 1 + \sqrt { d } ) ^ { 2 } \sum _ { t = 2 } ^ { \infty } \| { \pmb u } ^ { ( t ) } - { \pmb u } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } < \infty .
$$

Th<sub>e</sub> IR <sub>mec</sub>h<sub>an</sub>i<sub>sm ma</sub>k<sub>es</sub> th<sub>ese norms non</sub>d<sub>ecreas</sub>i<sub>ng, so</sub> th<sub>ey converge</sub> t<sub>o a pos</sub>iti<sub>ve</sub> li<sub>m</sub>it <sub>w</sub>h<sub>enever</sub> th<sub>e</sub> l<sub>earner</sub> b<sub>ecomes</sub> nonzero.

For HPRM<sup>+</sup>, first consider a learner with $t _ { 0 } < \infty$ <sub>.</sub> F<sub>or</sub> $T \geq t _ { 0 } - 1$ <sub>,</sub> th<sub>e max</sub>i<sub>mum-norm es</sub>ti<sub>ma</sub>t<sub>e</sub> i<sub>n</sub> th<sub>e proo</sub>f <sub>o</sub>f it<sub>s</sub> Alt<sub>-</sub>RVU b<sub>oun</sub>d <sub>g</sub>i<sub>ves</sub>

$$
\operatorname* { m a x } _ { 1 \leq s \leq T + 1 } \| \tilde { r } ^ { ( s ) } \| _ { 2 } \leq O ( 1 ) + \frac { 1 } { 3 2 } \sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } ,
$$

<sub>w</sub>h<sub>ere</sub> $O ( 1 )$ i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f $T ,$ <sub>,</sub> b<sub>ecause</sub> th<sub>e</sub> <sub>u</sub>tilit<sub>y-var</sub>i<sub>a</sub>ti<sub>on</sub> <sub>an</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>egy-movemen</sub>t <sub>sums</sub> <sub>are</sub> b<sub>oun</sub>d<sub>e</sub>d<sub>.</sub> M<sub>u</sub>lti<sub>p</sub>l<sub>y</sub>i<sub>ng</sub> thi<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> b<sub>y</sub> $^ { 5 }$ and combinin<sub>g</sub> it with (44), we obtain

$$
\mathrm { A l t - R e g } ( T ) + \operatorname* { m a x } _ { 1 \leq s \leq T + 1 } \| \tilde { r } ^ { ( s ) } \| _ { 2 } + \frac { 1 } { 1 6 } \sum _ { t = t _ { 0 } - 1 } ^ { T } \frac { \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| [ r ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } \leq O ( 1 ) .
$$

F<sub>or</sub> <sub>a</sub> l<sub>earner</sub> th<sub>a</sub>t <sub>never</sub> b<sub>ecomes</sub> <sub>nonzero,</sub> it<sub>s</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ng</sub> <sub>regre</sub>t <sub>an</sub>d <sub>regre</sub>t <sub>vec</sub>t<sub>ors</sub> <sub>are</sub> <sub>zero,</sub> <sub>an</sub>d <sub>we</sub> <sub>om</sub>it it<sub>s</sub> <sub>we</sub>i<sub>g</sub>ht<sub>e</sub>d increment sum. For suficientl<sub>y</sub> lar<sub>g</sub>e �, summin<sub>g</sub> over the two <sub>p</sub>la<sub>y</sub>ers and usin<sub>g</sub> (66), we conclude that both maximum <sub>norms an</sub>d b<sub>o</sub>th <sub>nonnega</sub>ti<sub>ve sums are</sub> b<sub>oun</sub>d<sub>e</sub>d<sub>.</sub> Si<sub>nce</sub> $[ { \pmb r } ^ { ( t + 1 ) } ] _ { + } \dot { } \le 3 \tilde { { \pmb r } } ^ { ( t + 1 ) } / 2$ <sub>,</sub> th<sub>e</sub> d<sub>enom</sub>i<sub>na</sub>t<sub>ors are</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>a</sub>b<sub>ove,</sub> <sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub>

$$
\sum _ { t = 1 } ^ { \infty } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } < \infty .
$$

M<sub>oreover,</sub> $f ^ { ( t ) }$ i<sub>s</sub> <sub>non</sub>d<sub>ecreas</sub>i<sub>ng</sub> <sub>an</sub>d b<sub>oun</sub>d<sub>e</sub>d<sub>.</sub> C<sub>onsequen</sub>tl<sub>y,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } = f ^ { ( t ) } + \frac { 1 } { 2 } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 }
$$

<sub>converges</sub> t<sub>o</sub> <sub>a</sub> <sub>pos</sub>iti<sub>ve</sub> li<sub>m</sub>it <sub>w</sub>h<sub>enever</sub> th<sub>e</sub> l<sub>earner</sub> b<sub>ecomes</sub> <sub>nonzero.</sub>

W<sub>e</sub> <sub>nex</sub>t <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> <sub>sum</sub> <sub>o</sub>f th<sub>e</sub> t<sub>wo</sub> <sub>regre</sub>t<sub>-vec</sub>t<sub>or</sub> i<sub>nner</sub> <sub>pro</sub>d<sub>uc</sub>t<sub>s</sub> <sub>converges</sub> f<sub>or</sub> <sub>every</sub> N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>um.</sub> Fi<sub>x</sub> <sub>any</sub> N<sub>as</sub>h <sub>e u</sub>ilib<sub>r</sub>i<sub>um</sub> $( \boldsymbol { x } ^ { * } , \boldsymbol { y } ^ { * } )$ <sub>.</sub> Th<sub>e co</sub>l<sub>umn</sub> l<sub>earner</sub>’<sub>s</sub> l<sub>oca</sub>l <sub>roun</sub>d � <sub>s</sub>t<sub>ar</sub>t<sub>s w</sub>ith th<sub>e pen</sub>di<sub>ng s</sub>t<sub>ra</sub>t<sub>egy</sub> $\mathbf { y } ^ { ( t - 1 ) }$ <sub>an</sub>d <sub>en</sub>d<sub>s w</sub>ith $\mathbf { y } ^ { ( t ) }$ <sub>.</sub> F<sub>or</sub> $\mathtt { H P R M } ^ { + }$ <sub>,</sub> th<sub>e</sub> <sub>coor</sub>di<sub>na</sub>t<sub>ew</sub>i<sub>se</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>up</sub>d<sub>a</sub>t<sub>es</sub> <sub>g</sub>i<sub>ve</sub>

$$
\begin{array} { r l } & { \langle \tilde { \boldsymbol { r } } _ { \boldsymbol { \chi } } ^ { ( t + 1 ) } - \tilde { \boldsymbol { r } } _ { \boldsymbol { \chi } } ^ { ( t ) } , \boldsymbol { x } ^ { * } \rangle + \langle \tilde { \boldsymbol { r } } _ { \boldsymbol { y } } ^ { ( t + 1 ) } - \tilde { \boldsymbol { r } } _ { \boldsymbol { y } } ^ { ( t ) } , \boldsymbol { y } ^ { * } \rangle } \\ & { \geq ( \boldsymbol { x } ^ { * } ) ^ { \top } \mathbf { A } \boldsymbol { y } ^ { ( t ) } - ( \boldsymbol { x } ^ { ( t ) } ) ^ { \top } \mathbf { A } \boldsymbol { y } ^ { * } + \langle - \mathbf { A } ^ { \top } \boldsymbol { x } ^ { ( t ) } , \boldsymbol { y } ^ { ( t ) } - \boldsymbol { y } ^ { ( t - 1 ) } \rangle } \\ & { \geq \langle - \mathbf { A } ^ { \top } \boldsymbol { x } ^ { ( t ) } , \boldsymbol { y } ^ { ( t ) } - \boldsymbol { y } ^ { ( t - 1 ) } \rangle , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> th<sub>e sa</sub>ddl<sub>e-po</sub>i<sub>n</sub>t i<sub>nequa</sub>liti<sub>es.</sub> F<sub>or</sub> $\tt T R { - } P R M ^ { + }$ <sub>,</sub> th<sub>e same ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on re</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> <sub>a</sub>dditi<sub>ona</sub>l t<sub>erms</sub> $c _ { X } ^ { ( t ) } + c _ { y } ^ { ( t ) }$

For IR-PRM<sup>+</sup>, a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the column <sub>p</sub>la<sub>y</sub>er’s utilit<sub>y</sub>-im<sub>p</sub>rovement bound therefore <sub>g</sub>ives

$$
\begin{array} { r } { \langle \tilde { r } _ { \chi } ^ { ( t + 1 ) } - \tilde { r } _ { \chi } ^ { ( t ) } , x ^ { * } \rangle + \langle \tilde { r } _ { y } ^ { ( t + 1 ) } - \tilde { r } _ { y } ^ { ( t ) } , y ^ { * } \rangle \ge c _ { \chi } ^ { ( t ) } + c _ { y } ^ { ( t + 1 ) } \ge 0 . } \end{array}
$$

Th<sub>us</sub> $\langle \tilde { r } _ { \chi } ^ { ( t ) } , x ^ { * } \rangle + \langle \tilde { r } _ { \chi } ^ { ( t ) } , y ^ { * } \rangle$ i<sub>s</sub> <sub>non</sub>d<sub>ecreas</sub>i<sub>ng</sub> <sub>an</sub>d b<sub>oun</sub>d<sub>e</sub>d<sub>,</sub> h<sub>ence</sub> <sub>convergen</sub>t<sub>.</sub> Th<sub>e</sub> <sub>same</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>a</sub>l<sub>so</sub> i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t b<sub>o</sub>th <sub>cre</sub>dit <sub>sums are</sub> fi<sub>n</sub>it<sub>e.</sub>

For HPRM<sup>+</sup>, whenever $\tilde { r } _ { y } ^ { ( t ) } \neq \mathbf { 0 } , ( 2 2 )$ and (43) <sub>g</sub>ive

$$
\langle - \mathbf { A } ^ { \top } \mathbf { x } ^ { ( t ) } , \mathbf { y } ^ { ( t ) } - \mathbf { y } ^ { ( t - 1 ) } \rangle \geq - \frac { f _ { y } ^ { ( t + 1 ) } - f _ { y } ^ { ( t ) } } { 2 \| [ r _ { y } ^ { ( t + 1 ) } ] _ { + } \| _ { 1 } } \geq - \frac { f _ { y } ^ { ( t + 1 ) } - f _ { y } ^ { ( t ) } } { \sqrt { 2 f _ { y } ^ { ( t + 1 ) } } } \geq - \sqrt { 2 } \left( \sqrt { f _ { y } ^ { ( t + 1 ) } } - \sqrt { f _ { y } ^ { ( t ) } } \right) .
$$

If $\tilde { r } _ { y } ^ { ( t ) } = \tilde { r } _ { y } ^ { ( t + 1 ) } = \mathbf { 0 }$ <sub>,</sub> th<sub>e u</sub>tilit<sub>y</sub> i<sub>mprovemen</sub>t <sub>an</sub>d th<sub>e</sub> $f ^ { \ast } \mathrm { \mathbf { s } }$ diference are both zero. At the first nonzero u<sub>p</sub>date, (15) <sub>g</sub>i<sub>ves a pos</sub>iti<sub>ve u</sub>tilit<sub>y</sub> i<sub>mprovemen</sub>t<sub>, an</sub>d $f _ { y } ^ { ( t ) } = 0 < f _ { y } ^ { ( t + 1 ) }$ <sub>.</sub> It f<sub>o</sub>ll<sub>ows</sub> th<sub>a</sub>t

$$
\langle \tilde { r } _ { \chi } ^ { ( t ) } , x ^ { * } \rangle + \langle \tilde { r } _ { y } ^ { ( t ) } , y ^ { * } \rangle + \sqrt { 2 f _ { y } ^ { ( t ) } }
$$

i<sub>s</sub> <sub>non</sub>d<sub>ecreas</sub>i<sub>ng</sub> <sub>an</sub>d b<sub>oun</sub>d<sub>e</sub>d<sub>.</sub> Si<sub>nce</sub> $f _ { y } ^ { ( t ) }$ <sub>converges,</sub> th<sub>e</sub> <sub>sum</sub> <sub>o</sub>f th<sub>e</sub> t<sub>wo</sub> i<sub>nner</sub> <sub>pro</sub>d<sub>uc</sub>t<sub>s</sub> <sub>a</sub>l<sub>so</sub> <sub>converges.</sub>

We now relate the re<sub>g</sub>ret vectors to the <sub>p</sub>la<sub>y</sub>ed strate<sub>g</sub>ies. For HPRM<sup>+</sup>, (38) <sub>g</sub>ives

$$
\| [ \pmb { r } ^ { ( t ) } ] _ { + } - \tilde { \pmb { r } } ^ { ( t ) } \| _ { 2 } \leq \frac { 1 } { 2 } \| \tilde { \pmb { r } } ^ { ( t ) } - \tilde { \pmb { r } } ^ { ( t - 1 ) } \| _ { 2 } \to 0 .
$$

F<sub>or</sub> $\tt T R { - } P R M ^ { + }$ <sub>,</sub> <sub>norm</sub> <sub>preserva</sub>ti<sub>on</sub> <sub>an</sub>d th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f $c ^ { ( t ) }$ <sub>g</sub><sup>i</sup>ve

$$
\begin{array} { l } { { \displaystyle c ^ { ( t ) } \| [ { \boldsymbol { r } } ^ { ( t ) } ] _ { + } \| _ { 1 } = \langle { \boldsymbol { r } } ^ { ( t ) } - \tilde { { \boldsymbol { r } } } ^ { ( t ) } , [ { \boldsymbol { r } } ^ { ( t ) } ] _ { + } \rangle } } \\ { { \displaystyle \qquad = \| [ { \boldsymbol { r } } ^ { ( t ) } ] _ { + } \| _ { 2 } ^ { 2 } - \langle \tilde { { \boldsymbol { r } } } ^ { ( t ) } , [ { \boldsymbol { r } } ^ { ( t ) } ] _ { + } \rangle = \frac { 1 } { 2 } \| [ { \boldsymbol { r } } ^ { ( t ) } ] _ { + } - \tilde { { \boldsymbol { r } } } ^ { ( t ) } \| _ { 2 } ^ { 2 } . } } \end{array}
$$

Th<sub>e cre</sub>dit <sub>sums are</sub> fi<sub>n</sub>it<sub>e an</sub>d th<sub>e re</sub>di<sub>c</sub>ti<sub>on norms are</sub> b<sub>oun</sub>d<sub>e</sub>d<sub>, so</sub> thi<sub>s</sub> di<sub>scre anc a</sub>l<sub>so</sub> t<sub>en</sub>d<sub>s</sub> t<sub>o zero.</sub> F<sub>or e</sub>ith<sub>er</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m,</sub> th<sub>e pos</sub>iti<sub>ve</sub> li<sub>m</sub>iti<sub>ng regre</sub>t <sub>norm an</sub>d th<sub>e norma</sub>li<sub>za</sub>ti<sub>on</sub> b<sub>oun</sub>d i<sub>mp</sub>l<sub>y</sub> th<sub>e correspon</sub>di<sub>ng</sub> li<sub>m</sub>it b<sub>e</sub>l<sub>ow</sub> f<sub>or eac</sub>h l<sub>earner</sub> th<sub>a</sub>t b<sub>ecomes nonzero:</sub>

$$
\left\| \mathbf { x } ^ { ( t ) } - \frac { \tilde { r } _ { \chi } ^ { ( t ) } } { \| \tilde { r } _ { \chi } ^ { ( t ) } \| _ { 1 } } \right\| _ { 2 } \to 0 , \left\| \mathbf { y } ^ { ( t ) } - \frac { \tilde { r } _ { y } ^ { ( t ) } } { \| \tilde { r } _ { y } ^ { ( t ) } \| _ { 1 } } \right\| _ { 2 } \to 0 .
$$

F<sub>or</sub> th<sub>e co</sub>l<sub>umn p</sub>l<sub>ayer, norma</sub>li<sub>za</sub>ti<sub>on</sub> i<sub>n</sub>iti<sub>a</sub>ll<sub>y g</sub>i<sub>ves</sub> $\mathbf { y } ^ { ( t - 1 ) }$ <sub>;</sub> th<sub>e</sub> di<sub>sp</sub>l<sub>aye</sub>d <sub>conc</sub>l<sub>us</sub>i<sub>on</sub> f<sub>o</sub>ll<sub>ows</sub> b<sub>ecause</sub> $\| \mathbf { y } ^ { ( t ) } - \mathbf { y } ^ { ( t - 1 ) } \| _ { 2 } \to$ 0<sub>.</sub> A l<sub>earner</sub> th<sub>a</sub>t <sub>never</sub> b<sub>ecomes nonzero rema</sub>i<sub>ns un</sub>if<sub>orm an</sub>d <sub>requ</sub>i<sub>res no norma</sub>li<sub>za</sub>ti<sub>on.</sub>

Fi<sub>na</sub>ll<sub>y,</sub> <sub>c</sub>h<sub>oose</sub> <sub>any</sub> <sub>accumu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>po</sub>i<sub>n</sub>t $( \boldsymbol { x } ^ { * } , \boldsymbol { y } ^ { * } )$ <sub>o</sub>f th<sub>e</sub> <sub>p</sub>l<sub>aye</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es.</sub> It i<sub>s</sub> <sub>a</sub> N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>um</sub> b<sub>ecause</sub> th<sub>e</sub> N<sub>as</sub>h <sub>gap</sub> t<sub>en</sub>d<sub>s</sub> t<sub>o zero.</sub> Th<sub>e convergence o</sub>f th<sub>e regre</sub>t <sub>norms an</sub>d <sub>o</sub>f th<sub>e sum o</sub>f th<sub>e</sub>i<sub>r</sub> i<sub>nner pro</sub>d<sub>uc</sub>t<sub>s w</sub>ith thi<sub>s equ</sub>ilib<sub>r</sub>i<sub>um</sub> i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t

$$
\| \tilde { { \boldsymbol r } } _ { \chi } ^ { ( t ) } \| _ { 2 } \| { \boldsymbol x } ^ { * } \| _ { 2 } - \langle \tilde { { \boldsymbol r } } _ { \chi } ^ { ( t ) } , { \boldsymbol x } ^ { * } \rangle + \| \tilde { { \boldsymbol r } } _ { y } ^ { ( t ) } \| _ { 2 } \| { \boldsymbol y } ^ { * } \| _ { 2 } - \langle \tilde { { \boldsymbol r } } _ { y } ^ { ( t ) } , { \boldsymbol y } ^ { * } \rangle
$$

h<sub>as</sub> <sub>a</sub> li<sub>m</sub>it<sub>.</sub> B<sub>o</sub>th <sub>summan</sub>d<sub>s</sub> <sub>are</sub> <sub>nonnega</sub>ti<sub>ve</sub> b<sub>y</sub> C<sub>auc</sub>h<sub>y–</sub>S<sub>c</sub>h<sub>warz.</sub> Al<sub>ong</sub> th<sub>e</sub> <sub>su</sub>b<sub>sequence</sub> <sub>converg</sub>i<sub>ng</sub> t<sub>o</sub> $( \boldsymbol { x } ^ { * } , \boldsymbol { y } ^ { * } )$ , th<sub>e prece</sub>di<sub>ng norma</sub>li<sub>za</sub>ti<sub>on</sub> li<sub>m</sub>it<sub>s s</sub>h<sub>ow</sub> th<sub>a</sub>t thi<sub>s express</sub>i<sub>on</sub> t<sub>en</sub>d<sub>s</sub> t<sub>o zero.</sub> It<sub>s</sub> f<sub>u</sub>ll li<sub>m</sub>it i<sub>s</sub> th<sub>ere</sub>f<sub>ore zero, an</sub>d <sub>eac</sub>h <sub>summan</sub>d t<sub>en</sub>d<sub>s</sub> t<sub>o zero.</sub>

F<sub>or</sub> <sub>an</sub> <sub>ac</sub>ti<sub>va</sub>t<sub>e</sub>d <sub>row</sub> l<sub>earner,</sub> <sub>we</sub> <sub>consequen</sub>tl<sub>y</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\left\| \frac { \tilde { r } _ { X } ^ { ( t ) } } { \| \tilde { r } _ { \chi } ^ { ( t ) } \| _ { 2 } } - \frac { x ^ { * } } { \| x ^ { * } \| _ { 2 } } \right\| _ { 2 } ^ { 2 } = \frac { 2 \left( \| \tilde { r } _ { \chi } ^ { ( t ) } \| _ { 2 } \| x ^ { * } \| _ { 2 } - \langle \tilde { r } _ { \chi } ^ { ( t ) } , x ^ { * } \rangle \right) } { \| \tilde { r } _ { \chi } ^ { ( t ) } \| _ { 2 } \| x ^ { * } \| _ { 2 } } \to 0 .
$$

N<sub>orma</sub>li<sub>z</sub>i<sub>ng</sub> b<sub>y</sub> th<sub>e coor</sub>di<sub>na</sub>t<sub>e sum an</sub>d <sub>us</sub>i<sub>ng</sub> th<sub>e prece</sub>di<sub>ng s</sub>t<sub>ra</sub>t<sub>egy approx</sub>i<sub>ma</sub>ti<sub>on, we o</sub>bt<sub>a</sub>i<sub>n</sub> $\pmb { x } ^ { ( t ) }  \pmb { x } ^ { * }$ <sub>.</sub> Th<sub>e same</sub> <sup>ar</sup>gu<sup>ment</sup> g<sup>i</sup>v<sup>es</sup> $\mathbf { y } ^ { ( t ) } \to \mathbf { y } ^ { * }$ <sub>.</sub> A <sub>never-ac</sub>ti<sub>va</sub>t<sub>e</sub>d l<sub>earner</sub> i<sub>s a</sub>l<sub>rea</sub>d<sub>y cons</sub>t<sub>an</sub>t<sub>.</sub> Thi<sub>s proves convergence</sub> t<sub>o</sub> th<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d N<sub>as</sub>h <sub>equ</sub>ilib<sub>r</sub>i<sub>um.</sub> □

## A.8 Proofs from Section 8

Theorem 8.1. Consider the normal-form game

$$
\mathbf { A } = \left( \begin{array} { c c c c } { 0 } & { - 1 } & { 1 } & { 4 } \\ { 1 } & { 0 } & { - 1 } & { 3 } \\ { - 1 } & { 1 } & { 0 } & { 3 } \\ { - 4 } & { - 3 } & { - 3 } & { 0 } \end{array} \right) .\tag{30}
$$

Under standard initialization, Sim-HPRM<sup>+</sup> satisfies

$$
\operatorname* { l i m } _ { t \to \infty } \operatorname* { i n f } _ { \mathbf { G } \mathbf { a p } ( \pmb { x } ^ { ( t ) } , \pmb { y } ^ { ( t ) } ) \geq \frac { 1 } { \sqrt { 3 } } } ,
$$

while Alt-HPRM<sup>+</sup> converges to the unique Nash equilibrium $\pmb { x } ^ { * } = \mathbf { y } ^ { * } = ( 1 / 3 , 1 / 3 , 1 / 3 , 0 )$ (Theorem 7.5).

Proof. We first consider simultaneous updates and omit the subscript Sim. Since $\mathbf { A } ^ { \top } = - \mathbf { A }$ <sub>an</sub>d b<sub>o</sub>th <sub>p</sub>l<sub>a ers</sub> h<sub>ave</sub> th<sub>e same</sub> i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on,</sub> th<sub>e</sub>i<sub>r regre</sub>t <sub>vec</sub>t<sub>ors an</sub>d <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es co</sub>i<sub>nc</sub>id<sub>e a</sub>t <sub>every</sub> it<sub>era</sub>ti<sub>on.</sub> C<sub>onsequen</sub>tl<sub>y,</sub> $\bar { \mathbf { x } } ^ { ( t ) } = \mathbf { y } ^ { ( t ) }$ ${ \pmb u } ^ { ( t ) } = { \bf A } { \pmb x } ^ { ( t ) }$ <sub>, an</sub>d $\langle { \pmb u } ^ { ( t ) } , { \pmb x } ^ { ( t ) } \rangle = \bar { 0 }$

The construction starts from three-action rock-paper-scissors (RPS) and adds a strictly dominated fourth action. Thi<sub>s</sub> <sub>ac</sub>ti<sub>on</sub> <sub>crea</sub>t<sub>es</sub> <sub>a</sub> <sub>nonun</sub>if<sub>orm</sub> <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> <sub>on</sub> th<sub>e</sub> fi<sub>rs</sub>t th<sub>ree</sub> <sub>ac</sub>ti<sub>ons</sub> th<sub>roug</sub>h th<sub>e</sub> <sub>ac</sub>t<sub>ua</sub>l <sub>up</sub>d<sub>a</sub>t<sub>es.</sub> With<sub>ou</sub>t thi<sub>s</sub> <sub>ac</sub>ti<sub>on,</sub> <sub>a</sub> zero initialization in three-action RPS alone would remain at equilibrium. After the standard initialization at time (1), th<sub>e</sub> fi<sub>rs</sub>t <sub>o</sub>b<sub>serva</sub>ti<sub>on g</sub>i<sub>ves</sub>

$$
\tilde { \pmb { r } } ^ { ( 2 ) } = ( 1 , 3 / 4 , 3 / 4 , 0 ) , \ \pmb { x } ^ { ( 2 ) } = \pmb { y } ^ { ( 2 ) } = ( 2 / 5 , 3 / 1 0 , 3 / 1 0 , 0 ) .
$$

Wh<sub>enever</sub> $x _ { 4 } ^ { ( t ) } = 0$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub> $( \mathbf { A } \pmb { x } ^ { ( t ) } ) _ { 4 } = - 4 \pmb { x } _ { 1 } ^ { ( t ) } - 3 \pmb { x } _ { 2 } ^ { ( t ) } - 3 \pmb { x } _ { 3 } ^ { ( t ) } < 0$ <sub>.</sub> Th<sub>us</sub> $\tilde { r } _ { 4 } ^ { ( t ) } = x _ { 4 } ^ { ( t ) } = 0$ <sup>f</sup>or ever<sub>y</sub> $t \geq 2$ <sub>,</sub> <sub>an</sub>d

$$
\begin{array} { r } { \left( ( \mathbf { A } \pmb { x } ^ { ( t ) } ) _ { 1 } , ( \mathbf { A } \pmb { x } ^ { ( t ) } ) _ { 2 } , ( \mathbf { A } \pmb { x } ^ { ( t ) } ) _ { 3 } \right) = \left( x _ { 3 } ^ { ( t ) } - x _ { 2 } ^ { ( t ) } , x _ { 1 } ^ { ( t ) } - x _ { 3 } ^ { ( t ) } , x _ { 2 } ^ { ( t ) } - x _ { 1 } ^ { ( t ) } \right) . } \end{array}
$$

Th<sub>ese</sub> th<sub>ree</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> <sub>sum</sub> t<sub>o</sub> <sub>zero.</sub> Th<sub>ere</sub>f<sub>ore,</sub> f<sub>or</sub> $t \geq 2$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> t<sub>wo</sub> b<sub>oun</sub>d<sub>s.</sub>

$$
\| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 } = \sum _ { i = 1 } ^ { 3 } \left[ \tilde { r } _ { i } ^ { ( t ) } + ( \mathbf { A } \pmb { x } ^ { ( t ) } ) _ { i } \right] _ { + } \geq \sum _ { i = 1 } ^ { 3 } \left( \tilde { r } _ { i } ^ { ( t ) } + ( \mathbf { A } \pmb { x } ^ { ( t ) } ) _ { i } \right) = \| \tilde { r } ^ { ( t ) } \| _ { 1 } \geq \frac { 5 } { 2 } ,\tag{69}
$$

$$
\| \widetilde { \pmb r } ^ { ( t + 1 ) } - \widetilde { \pmb r } ^ { ( t ) } \| _ { 2 } ^ { 2 } \le \sum _ { i = 1 } ^ { 3 } ( { \textbf A } { \pmb x } ^ { ( t ) } ) _ { i } ^ { 2 } = 3 \| { \pmb x } ^ { ( t ) } - { \pmb x } ^ { * } \| _ { 2 } ^ { 2 } \le 2 .\tag{70}
$$

H<sub>ere an</sub>d b<sub>e</sub>l<sub>ow, s</sub>t<sub>r</sub>i<sub>c</sub>t t<sub>runca</sub>ti<sub>on re</sub>f<sub>ers on</sub>l<sub>y</sub> t<sub>o</sub> th<sub>e</sub> fi<sub>rs</sub>t th<sub>ree coor</sub>di<sub>na</sub>t<sub>es.</sub> Th<sub>e</sub> fi<sub>rs</sub>t i<sub>nequa</sub>lit<sub>y</sub> i<sub>s an equa</sub>lit<sub>y w</sub>h<sub>enever</sub> <sub>no coor</sub>di<sub>na</sub>t<sub>e</sub> i<sub>s s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> t<sub>runca</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e regre</sub>t <sub>vec</sub>t<sub>or up</sub>d<sub>a</sub>t<sub>e.</sub> Th<sub>e secon</sub>d f<sub>o</sub>ll<sub>ows</sub> f<sub>rom nonexpans</sub>i<sub>veness o</sub>f th<sub>e</sub> <sub>p</sub>os<sup>i</sup>t<sup>i</sup>ve-<sub>p</sub>art o<sub>p</sub>erat<sup>i</sup>on.

Recall that, for HPRM<sup>+</sup>, we define

$$
f ^ { ( t ) } = \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } .
$$

W<sub>e</sub> fi<sub>rs</sub>t <sub>s</sub>t<sub>reng</sub>th<sub>en</sub> it<sub>s</sub> b<sub>oun</sub>d f<sub>rom</sub> L<sub>emma</sub> 5<sub>.</sub>10<sub>.</sub> Th<sub>e</sub> id<sub>en</sub>tit<sub>y</sub> $[ \pmb { r } ^ { ( t ) } ] _ { + } = [ \tilde { \pmb { r } } ^ { ( t ) } + ( \tilde { \pmb { r } } ^ { ( t ) } - \tilde { \pmb { r } } ^ { ( t - 1 ) } ) / 2 ]$ + g<sup>i</sup>v<sup>es</sup>

$$
\langle \tilde { { \pmb r } } ^ { ( t ) } - \tilde { { \pmb r } } ^ { ( t - 1 ) } , [ { \pmb r } ^ { ( t ) } ] _ { + } - \tilde { { \pmb r } } ^ { ( t ) } \rangle \geq 2 \| [ { \pmb r } ^ { ( t ) } ] _ { + } - \tilde { { \pmb r } } ^ { ( t ) } \| _ { 2 } ^ { 2 } .
$$

A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> Y<sub>oung</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y an</sub>d th<sub>en</sub> thi<sub>s</sub> b<sub>oun</sub>d<sub>, we o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r l } & { 2 \langle \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } , [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \rangle = 2 \langle \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t - 1 ) } , [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \rangle - 2 \langle \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } , [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \rangle } \\ & { \qquad \leq \frac { 1 } { 4 } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } + 4 \| [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } - 2 \langle \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } , [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \rangle } \\ & { \qquad \leq \frac { 1 } { 4 } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } . } \end{array}
$$

Si<sub>nce</sub> $\langle \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } , [ r ^ { ( t ) } ] _ { + } \rangle \geq 0 .$ <sub>,</sub> <sub>as</sub> i<sub>n</sub> L<sub>emma</sub> 5<sub>.</sub>10<sub>,</sub> <sub>we</sub> <sub>can</sub> <sub>expan</sub>d th<sub>e</sub> $f ^ { \ast } \mathrm { s }$ dif<sub>erence</sub> t<sub>o</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { l } { f ^ { ( t + 1 ) } - f ^ { ( t ) } \ge \displaystyle \frac { 1 } { 2 } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \displaystyle \frac { 1 } { 2 } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } - 2 \langle \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } , [ r ^ { ( t ) } ] _ { + } - \tilde { r } ^ { ( t ) } \rangle } \\ { \ge \displaystyle \frac { 1 } { 2 } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } + \displaystyle \frac { 1 } { 2 } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } - \displaystyle \frac { 1 } { 4 } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } } \\ { = \displaystyle \frac { 1 } { 4 } \| \tilde { r } ^ { ( t + 1 ) } - 2 \tilde { r } ^ { ( t ) } + \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } . } \end{array}\tag{71}
$$

W<sub>e</sub> <sub>nex</sub>t <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>ors</sub> <sub>rema</sub>i<sub>n</sub> <sub>separa</sub>t<sub>e</sub>d f<sub>rom</sub> <sub>vec</sub>t<sub>ors</sub> <sub>w</sub>ith th<sub>ree</sub> <sub>equa</sub>l <sub>pos</sub>iti<sub>ve</sub> <sub>coor</sub>di<sub>na</sub>t<sub>es.</sub> Di<sub>rec</sub>t <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>g</sub>i<sub>ves</sub>

$$
\tilde { r } ^ { ( 3 ) } = ( 1 , 1 7 / 2 0 , 1 3 / 2 0 , 0 ) , \ f ^ { ( 3 ) } = \frac { 4 2 7 } { 2 0 0 } , \ f ^ { ( 3 ) } - \frac { \| \tilde { r } ^ { ( 3 ) } \| _ { 1 } ^ { 2 } } { 3 } = \frac { 3 1 } { 6 0 0 } > \frac { 1 } { 2 4 } .
$$

If <sub>no</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> i<sub>s</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> t<sub>runca</sub>t<sub>e</sub>d<sub>,</sub> $\| \tilde { r } ^ { ( t + 1 ) } \| _ { 1 } = \| \tilde { r } ^ { ( t ) } \| _ { 1 }$ <sub>an</sub>d $f ^ { ( t + 1 ) } \geq f ^ { ( t ) }$ <sub>.</sub> If <sub>a</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> i<sub>s</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> t<sub>runca</sub>t<sub>e</sub>d<sub>,</sub> $\tilde { r } ^ { ( t + 1 ) }$ has at most two <sub>p</sub>ositive coordinates. Usin<sub>g</sub> (69) and (70), we then obtain

$$
f ^ { ( t + 1 ) } - \frac { \Vert \tilde { r } ^ { ( t + 1 ) } \Vert _ { 1 } ^ { 2 } } { 3 } \geq \frac { \Vert \tilde { r } ^ { ( t + 1 ) } \Vert _ { 1 } ^ { 2 } } { 2 } - 1 - \frac { \Vert \tilde { r } ^ { ( t + 1 ) } \Vert _ { 1 } ^ { 2 } } { 3 } = \frac { \Vert \tilde { r } ^ { ( t + 1 ) } \Vert _ { 1 } ^ { 2 } } { 6 } - 1 \geq \frac { 1 } { 2 4 } .
$$

F<sub>or</sub> $t \geq 3$ <sub>, we ge</sub>t b<sub>y</sub> i<sub>n</sub>d<sub>uc</sub>ti<sub>on</sub> th<sub>a</sub>t

$$
f ^ { ( t ) } - \frac { \Vert \tilde { r } ^ { ( t ) } \Vert _ { 1 } ^ { 2 } } { 3 } \geq \frac { 1 } { 2 4 } .\tag{72}
$$

We no<sub>w p</sub>ro<sub>v</sub>e that $\| \tilde { r } ^ { ( t ) } \| _ { 1 } \to \infty$ <sub>.</sub> S<sub>uppose</sub> i<sub>ns</sub>t<sub>ea</sub>d th<sub>a</sub>t th<sub>ese norms are</sub> b<sub>oun</sub>d<sub>e</sub>d<sub>.</sub> Th<sub>en</sub> $f ^ { ( t ) }$ i<sub>s</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>an</sub>d nondecreasin<sub>g</sub>, so (71) im<sub>p</sub>lies that

$$
\| \tilde { r } ^ { ( t + 1 ) } - 2 \tilde { r } ^ { ( t ) } + \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } \to 0 .
$$

F<sub>or</sub> <sub>every</sub> fi<sub>xe</sub>d i<sub>n</sub>t<sub>eger</sub> $k \geq 1$ <sub>,</sub> <sub>we</sub> <sub>can</sub> <sub>sum</sub> th<sub>e</sub> <sub>consecu</sub>ti<sub>ve</sub> i<sub>ncremen</sub>t<sub>s</sub> t<sub>o</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> b<sub>oun</sub>d<sub>s</sub>

$$
\tilde { r } ^ { ( t + k ) } - \tilde { r } ^ { ( t ) } = k \left( \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \right) + o ( 1 ) , \operatorname* { l i m } _ { t \to \infty } \| \tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \| _ { 2 } \leq \frac { 2 } { k } \operatorname* { s u p } _ { s \geq 2 } \| \tilde { r } ^ { ( s ) } \| _ { 2 } .
$$

L<sub>e</sub>ttin<sub>g</sub> $k \to \infty$ , w<sup>e</sup> g<sup>et</sup> $\tilde { r } ^ { ( t + 1 ) } - \tilde { r } ^ { ( t ) } \to 0$ <sub>.</sub> Th<sub>e</sub> <sub>regre</sub>t <sub>vec</sub>t<sub>or</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> <sub>a</sub>l<sub>so</sub> <sub>g</sub>i<sub>ves</sub>

$$
( \mathbf { A } \mathbf { } \mathbf { } x ^ { ( t ) } ) _ { i } \leq \tilde { r } _ { i } ^ { ( t + 1 ) } - \tilde { r } _ { i } ^ { ( t ) } , \mathrm { f o r } i = 1 , 2 , 3 .
$$

Si<sub>nce</sub> th<sub>e</sub> th<sub>ree u</sub>tilit<sub>y coor</sub>di<sub>na</sub>t<sub>es sum</sub> t<sub>o zero,</sub> th<sub>ey a</sub>ll t<sub>en</sub>d t<sub>o zero.</sub> Th<sub>e</sub>i<sub>r exp</sub>li<sub>c</sub>it <sub>express</sub>i<sub>ons a</sub>b<sub>ove</sub> i<sub>mp</sub>l<sub>y</sub> $\boldsymbol { x } ^ { ( t ) } \to \boldsymbol { x } ^ { * }$ M<sub>oreover, we</sub> h<sub>ave</sub>

$$
\| [ \pmb { r } ^ { ( t ) } ] _ { + } - \tilde { \pmb { r } } ^ { ( t ) } \| _ { 2 } \leq \frac { 1 } { 2 } ( \sum _ { i = 1 } ^ { 3 } ( \mathbf { A } \pmb { x } ^ { ( t - 1 ) } ) _ { i } ^ { 2 } ) ^ { 1 / 2 }  0 .
$$

T<sub>oge</sub>th<sub>er w</sub>ith b<sub>oun</sub>d<sub>e</sub>d<sub>ness o</sub>f $\| \tilde { r } ^ { ( t ) } \| _ { 1 }$ <sub>,</sub> thi<sub>s</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> $\tilde { r } ^ { ( t ) } - \| \tilde { r } ^ { ( t ) } \| _ { 1 } \pmb { x } ^ { * }  \mathbf { 0 }$ . <sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>,

$$
f ^ { ( t ) } - \frac { \| \tilde { r } ^ { ( t ) } \| _ { 1 } ^ { 2 } } { 3 } = \| \tilde { r } ^ { ( t ) } - \| \tilde { r } ^ { ( t ) } \| _ { 1 } { \boldsymbol { x } } ^ { * } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } \| \tilde { r } ^ { ( t ) } - \tilde { r } ^ { ( t - 1 ) } \| _ { 2 } ^ { 2 } \to 0 ,
$$

contradictin<sub>g</sub> (72). Thus the nondecreasin<sub>g</sub> sequence $\| \tilde { r } ^ { ( t ) } \| _ { 1 }$ <sub>mus</sub>t di<sub>verge.</sub>

There is therefore a strict truncation, since otherwise (69) would hold with equalit<sub>y</sub> at ever<sub>y</sub> iteration. Immediatel<sub>y</sub> after an<sub>y</sub> strict truncation, (70) im<sub>p</sub>lies that

$$
f ^ { ( t ) } \geq \frac { \Vert \tilde { r } ^ { ( t ) } \Vert _ { 1 } ^ { 2 } } { 2 } - 1 .
$$

B<sub>e</sub>t<sub>ween s</sub>t<sub>r</sub>i<sub>c</sub>t t<sub>runca</sub>ti<sub>ons,</sub> th<sub>e r</sub>i<sub>g</sub>ht<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e</sub> i<sub>s cons</sub>t<sub>an</sub>t <sub>an</sub>d $f ^ { ( t ) }$ i<sub>s non</sub>d<sub>ecreas</sub>i<sub>ng.</sub> H<sub>ence</sub> thi<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> h<sub>o</sub>ld<sub>s a</sub>t <sub>every</sub> it<sub>era</sub>ti<sub>on a</sub>ft<sub>er</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>s</sub>t<sub>r</sub>i<sub>c</sub>t t<sub>runca</sub>ti<sub>on.</sub> W<sub>e now o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\left\| \frac { \tilde { r } ^ { ( t ) } } { \| \tilde { r } ^ { ( t ) } \| _ { 1 } } - x ^ { * } \right\| _ { 2 } ^ { 2 } = \frac { \| \tilde { r } ^ { ( t ) } \| _ { 2 } ^ { 2 } } { \| \tilde { r } ^ { ( t ) } \| _ { 1 } ^ { 2 } } - \frac { 1 } { 3 } \ge \frac { 1 } { 6 } - \frac { 1 } { \| \tilde { r } ^ { ( t ) } \| _ { 1 } ^ { 2 } } .
$$

F<sub>or</sub> $t \geq 3$ <sub>,</sub> th<sub>e</sub> h<sub>a</sub>lf<sub>-pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> <sub>a</sub>l<sub>so</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\| [ \boldsymbol { r } ^ { ( t ) } ] _ { + } - \tilde { \boldsymbol { r } } ^ { ( t ) } \| _ { 1 } \leq \frac { 1 } { 2 } \sum _ { i = 1 } ^ { 3 } | ( \mathbf { A } \boldsymbol { x } ^ { ( t - 1 ) } ) _ { i } | \leq 1 , \ \left\| \boldsymbol { x } ^ { ( t ) } - \frac { \tilde { \boldsymbol { r } } ^ { ( t ) } } { \| \tilde { \boldsymbol { r } } ^ { ( t ) } \| _ { 1 } } \right\| _ { 1 } \leq \frac { 2 } { \| \tilde { \boldsymbol { r } } ^ { ( t ) } \| _ { 1 } } \to 0 .
$$

It f<sub>o</sub>ll<sub>ows</sub> th<sub>a</sub>t

$$
\operatorname* { l i m } _ { t \to \infty } \operatorname* { i n f } _ { \mathbf { \omega } } \| \mathbf { \boldsymbol { x } } ^ { ( t ) } - \mathbf { \boldsymbol { x } } ^ { * } \| _ { 2 } \geq \frac { 1 } { \sqrt { 6 } } .
$$

Fi<sub>na</sub>ll<sub>y,</sub> f<sub>rom</sub> th<sub>e s</sub>k<sub>ew symme</sub>t<sub>ry an</sub>d <sub>nega</sub>ti<sub>v</sub>it<sub>y o</sub>f th<sub>e</sub> f<sub>our</sub>th <sub>u</sub>tilit<sub>y coor</sub>di<sub>na</sub>t<sub>e, we no</sub>t<sub>e</sub> th<sub>a</sub>t

$$
\mathrm { G a p } ( \pmb { x } ^ { ( t ) } , \pmb { x } ^ { ( t ) } ) = 2 \operatorname* { m a x } _ { 1 \leq i \leq 3 } ( \mathbf { A } \pmb { x } ^ { ( t ) } ) _ { i } .
$$

F<sub>or</sub> th<sub>ree</sub> <sub>rea</sub>l <sub>num</sub>b<sub>ers</sub> <sub>w</sub>ith <sub>zero</sub> <sub>sum,</sub> th<sub>e</sub>i<sub>r</sub> <sub>square</sub>d <sub>sum</sub> i<sub>s</sub> <sub>a</sub>t <sub>mos</sub>t <sub>s</sub>i<sub>x</sub> ti<sub>mes</sub> th<sub>e</sub> <sub>square</sub> <sub>o</sub>f th<sub>e</sub>i<sub>r</sub> <sub>max</sub>i<sub>mum.</sub> Th<sub>ere</sub>f<sub>ore,</sub> w<sup>e</sup> g<sup>et</sup>

$$
3 \| { \boldsymbol { \mathbf { x } } } ^ { ( t ) } - { \boldsymbol { \mathbf { \mathit { x } } } } ^ { * } \| _ { 2 } ^ { 2 } = \sum _ { i = 1 } ^ { 3 } ( \mathbf { A } { \boldsymbol { \mathbf { \mathit { x } } } } ^ { ( t ) } ) _ { i } ^ { 2 } \leq 6 \left( \operatorname* { m a x } _ { 1 \leq i \leq 3 } ( \mathbf { A } { \boldsymbol { \mathbf { \mathit { x } } } } ^ { ( t ) } ) _ { i } \right) ^ { 2 } = { \frac { 3 } { 2 } } \operatorname { G a p } ( { \boldsymbol { \mathbf { \mathit { x } } } } ^ { ( t ) } , { \boldsymbol { \mathbf { \mathit { x } } } } ^ { ( t ) } ) ^ { 2 } .
$$

C<sub>om</sub>bi<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> l<sub>as</sub>t t<sub>wo</sub> b<sub>oun</sub>d<sub>s, we conc</sub>l<sub>u</sub>d<sub>e</sub> th<sub>a</sub>t

$$
\operatorname* { l i m i n f } _ { t \to \infty } \mathrm { G a p } ( \pmb { x } ^ { ( t ) } , \pmb { y } ^ { ( t ) } ) \geq \sqrt { 2 } \operatorname* { l i m i n f } _ { t \to \infty } \| \pmb { x } ^ { ( t ) } - \pmb { x } ^ { * } \| _ { 2 } \geq \frac { 1 } { \sqrt { 3 } } .
$$

## B Additional Experimental Details and Results

## B.1 Construction and Validation of the Numerical Separations

Th<sub>e</sub> t<sub>wo</sub> bl<sub>oc</sub>k <sub>games</sub> i<sub>n</sub> Fi<sub>gure</sub> 4 <sub>s</sub>h<sub>are</sub> th<sub>e unsca</sub>l<sub>e</sub>d i<sub>n</sub>t<sub>eger</sub> k<sub>erne</sub>l

$$
\mathbf { A } _ { 0 } = \left( \begin{array} { c c c } { - 1 } & { 0 } & { 2 } \\ { 1 9 } & { - 1 } & { - 1 9 } \\ { - 1 9 } & { 2 } & { - 1 9 } \end{array} \right) .
$$

We con<sub>s</sub>tr<sub>u</sub>ct $\mathbf { A } ( a ) \in \mathbb { R } ^ { ( a + 2 ) \times ( a + 2 ) }$ b<sub>y rep</sub>l<sub>ac</sub>i<sub>ng</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>row o</sub>f ${ \bf A } _ { 0 }$ <sub>w</sub>ith <sub>�</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>cop</sub>i<sub>es an</sub>d it<sub>s secon</sub>d <sub>co</sub>l<sub>umn</sub> <sub>w</sub>ith <sub>�</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>cop</sub>i<sub>es, re</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng one copy o</sub>f <sub>eac</sub>h <sub>rema</sub>i<sub>n</sub>i<sub>ng row an</sub>d <sub>co</sub>l<sub>umn.</sub> W<sub>e use</sub> $a = 1 1 0 0 0$ f<sub>or</sub> $\lambda = 1 0 ^ { - 2 }$ <sub>an</sub>d $a = 1 7 7 7 0 0$ f<sub>or</sub> $\lambda = 1 / 3$ <sub>, g</sub>i<sub>v</sub>i<sub>ng games o</sub>f di<sub>mens</sub>i<sub>ons</sub> $1 1 0 0 2 \times 1 1 0 0 2$ <sub>an</sub>d $1 7 7 7 0 2 \times 1 7 7 7 0 2$ <sub>, respec</sub>ti<sub>ve</sub>l<sub>y.</sub> O<sub>n eac</sub>h <sub>game,</sub> <sub>we</sub> <sub>compare</sub> th<sub>e</sub> <sub>spec</sub>ifi<sub>e</sub>d <sub>mu</sub>lti<sub>p</sub>li<sub>er</sub> <sub>w</sub>ith h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> i<sub>n</sub> b<sub>o</sub>th <sub>un</sub>if<sub>orm</sub> <sub>average-</sub>it<sub>era</sub>t<sub>e</sub> <sub>an</sub>d l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e</sub> <sub>gap.</sub>

U<sub>n</sub>d<sub>er s</sub>t<sub>an</sub>d<sub>ar</sub>d i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on,</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>cop</sub>i<sub>es o</sub>f <sub>an ac</sub>ti<sub>on re</sub>t<sub>a</sub>i<sub>n</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>regre</sub>t <sub>coor</sub>di<sub>na</sub>t<sub>es.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore s</sub>i<sub>mu</sub>l<sub>a</sub>t<sub>e</sub> <sub>eac</sub>h <sub>ga</sub>m<sub>e us</sub>in<sub>g</sub> thr<sub>ee ac</sub>ti<sub>o</sub>n t<sub>ypes pe</sub>r <sub>p</sub>l<sub>aye</sub>r<sub>, s</sub>t<sub>o</sub>rin<sub>g</sub> th<sub>e</sub> r<sub>eg</sub>r<sub>e</sub>t <sub>coo</sub>rdin<sub>a</sub>t<sub>es o</sub>f <sub>o</sub>n<sub>e copy o</sub>f <sub>eac</sub>h t<sub>ype.</sub> If $s _ { 1 } , s _ { 2 } , s _ { 3 }$ are th<sub>e pos</sub>iti<sub>ve par</sub>t<sub>s o</sub>f th<sub>e row p</sub>l<sub>ayer</sub>’<sub>s pre</sub>di<sub>c</sub>ti<sub>on regre</sub>t <sub>coor</sub>di<sub>na</sub>t<sub>es,</sub> it<sub>s aggrega</sub>t<sub>e s</sub>t<sub>ra</sub>t<sub>egy</sub> i<sub>s</sub>

$$
{ \frac { \left( a s _ { 1 } , s _ { 2 } , s _ { 3 } \right) } { a s _ { 1 } + s _ { 2 } + s _ { 3 } } } ,
$$

<sub>w</sub>h<sub>en</sub> th<sub>e</sub> d<sub>enom</sub>i<sub>na</sub>t<sub>or</sub> i<sub>s pos</sub>iti<sub>ve;</sub> th<sub>e co</sub>l<sub>umn p</sub>l<sub>ayer ana</sub>l<sub>ogous</sub>l<sub>y we</sub>i<sub>g</sub>ht<sub>s</sub> it<sub>s secon</sub>d <sub>ac</sub>ti<sub>on</sub> t<sub>ype</sub> b<sub>y �.</sub> Wh<sub>en</sub> th<sub>e</sub> d<sub>enom</sub>i<sub>na</sub>t<sub>or</sub> i<sub>s</sub> <sub>zero,</sub> <sub>un</sub>if<sub>orm</sub> <sub>p</sub>l<sub>ay</sub> <sub>over</sub> th<sub>e</sub> <sub>expan</sub>d<sub>e</sub>d <sub>ac</sub>ti<sub>ons</sub> <sub>g</sub>i<sub>ves</sub> <sub>aggrega</sub>t<sub>e</sub> <sub>s</sub>t<sub>ra</sub>t<sub>eg</sub>i<sub>es</sub> $( a , 1 , 1 ) / ( a { + } 2 )$ <sub>an</sub>d $( 1 , a , 1 ) / ( a +$ 2) for the row and column pla<sub>y</sub>ers, respectivel<sub>y</sub>. We compute utilities from the kernel and the opponent’s aggregate <sub>s</sub>t<sub>ra</sub>t<sub>egy, re</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> th<sub>e mu</sub>lti<sub>p</sub>li<sub>c</sub>it<sub>y we</sub>i<sub>g</sub>ht<sub>s a</sub>t <sub>every norma</sub>li<sub>za</sub>ti<sub>on.</sub> A<sub>ggrega</sub>ti<sub>on preserves payo</sub>f<sub>s an</sub>d d<sub>ua</sub>lit<sub>y gaps an</sub>d <sub>commu</sub>t<sub>es w</sub>ith <sub>un</sub>if<sub>orm averag</sub>i<sub>ng, so</sub> th<sub>e re</sub>d<sub>uc</sub>ti<sub>on preserves</sub> b<sub>o</sub>th <sub>gap measures</sub> i<sub>n</sub> Fi<sub>gure</sub> 4<sub>.</sub> D<sub>up</sub>li<sub>ca</sub>ti<sub>on never</sub>th<sub>e</sub>l<sub>ess</sub> <sub>c</sub>h<sub>anges</sub> th<sub>e</sub> l<sub>earners</sub>’ <sub>aggrega</sub>t<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on an</sub>d <sub>norma</sub>li<sub>za</sub>ti<sub>on we</sub>i<sub>g</sub>ht<sub>s.</sub>

Choosing the kernel and multiplicities. We first varied � and � within the structured kernel family

$$
{ \bf K } ( q ) = \left( \begin{array} { c c c } { { - 1 } } & { { 0 } } & { { 2 } } \\ { { q } } & { { - 1 } } & { { - q } } \\ { { - q } } & { { 2 } } & { { - q } } \end{array} \right) .
$$

W<sub>e soug</sub>ht <sub>can</sub>did<sub>a</sub>t<sub>es w</sub>ith <sub>pers</sub>i<sub>s</sub>t<sub>en</sub>t l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e gap osc</sub>ill<sub>a</sub>ti<sub>ons an</sub>d <sub>grow</sub>i<sub>ng</sub> $T \mathrm { G a p } ( \bar { \pmb x } ^ { ( T ) } , \bar { \pmb y } ^ { ( T ) } )$ <sub>un</sub>d<sub>er</sub> <sub>non-</sub>h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on,</sub> <sub>w</sub>hil<sub>e</sub> h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>s</sub>h<sub>owe</sub>d f<sub>as</sub>t<sub>er</sub> <sub>average-</sub>it<sub>era</sub>t<sub>e</sub> d<sub>ecay</sub> <sub>an</sub>d d<sub>ecreas</sub>i<sub>ng</sub> l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e</sub> <sub>gaps.</sub> I<sub>n</sub>iti<sub>a</sub>l <sub>screen</sub>i<sub>ng</sub> <sub>use</sub>d $1 0 ^ { 6 }$ iterations, followed by longer runs to reject candidates whose apparent slow average-iterate decay was t<sub>rans</sub>i<sub>en</sub>t<sub>.</sub>

F<sub>or</sub> $\lambda = 1 / 3$ <sub>, a</sub> l<sub>oca</sub>l <sub>re</sub>fi<sub>nemen</sub>t <sub>w</sub>ith $q = 1 9$ <sub>an</sub>d $a \in \{ 1 6 5 0 0 0 , 1 6 5 1 0 0 , \ldots , 1 8 5 0 0 0 \}$ <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d $a = 1 7 7 7 0 0 .$ <sub>.</sub> F<sub>or</sub> $\lambda = 1 0 ^ { - 2 }$ <sub>, we</sub> fi<sub>xe</sub>d th<sub>e same</sub> k<sub>erne</sub>l $\mathbf { K } ( 1 9 )$ <sub>an</sub>d <sub>screene</sub>d 779 di<sub>s</sub>ti<sub>nc</sub>t <sub>va</sub>l<sub>ues o</sub>f <sub>�</sub> b<sub>e</sub>f<sub>ore se</sub>l<sub>ec</sub>ti<sub>ng</sub> $a = 1 1 0 0 0 .$ <sub>.</sub> B<sub>o</sub>th <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>games were</sub> th<sub>en eva</sub>l<sub>ua</sub>t<sub>e</sub>d th<sub>roug</sub>h $1 0 ^ { 9 }$ it<sub>era</sub>ti<sub>ons.</sub> Th<sub>ese mu</sub>lti<sub>p</sub>li<sub>c</sub>iti<sub>es were se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>numer</sub>i<sub>ca</sub>ll<sub>y ra</sub>th<sub>er</sub> th<sub>an</sub> d<sub>er</sub>i<sub>ve</sub>d f<sub>rom an ana</sub>l ti<sub>c</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>, an</sub>d <sub>ne</sub>ith<sub>er</sub> i<sub>s c</sub>l<sub>a</sub>i<sub>me</sub>d t<sub>o</sub> b<sub>e m</sub>i<sub>n</sub>i<sub>ma</sub>l<sub>.</sub>

Validation and observed behavior. All trajectories in Figure 4 use the same double-double implementation, with <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub>l<sub>y</sub> 31 d<sub>ec</sub>i<sub>ma</sub>l di<sub>g</sub>it<sub>s o</sub>f <sub>prec</sub>i<sub>s</sub>i<sub>on, s</sub>t<sub>an</sub>d<sub>ar</sub>d i<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on, an</sub>d <sub>no pos</sub>iti<sub>ve norma</sub>li<sub>za</sub>ti<sub>on</sub> fl<sub>oor.</sub> E<sub>very</sub> it<sub>era</sub>ti<sub>on con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> t<sub>o</sub> th<sub>e un</sub>if<sub>orm average, an</sub>d <sub>we recor</sub>d b<sub>o</sub>th <sub>gap measures a</sub>t th<sub>e same c</sub>h<sub>ec</sub>k<sub>po</sub>i<sub>n</sub>t<sub>s</sub> th<sub>roug</sub>h $1 0 ^ { \overset { \cdot } { 9 } }$ it<sub>era</sub>ti<sub>ons.</sub>

F<sub>or</sub> th<sub>e</sub> t<sub>wo</sub> bl<sub>oc</sub>k <sub>games,</sub> th<sub>e</sub> fitt<sub>e</sub>d <sub>average-</sub>it<sub>era</sub>t<sub>e</sub> d<sub>ecay exponen</sub>t<sub>s over</sub> $1 0 ^ { 8 } \le T \le 1 0 ^ { 9 }$ <sub>, measure</sub>d <sub>as</sub> th<sub>e nega</sub>ti<sub>ve</sub> <sub>s</sub>l<sub>opes o</sub>f th<sub>e</sub> l<sub>og–</sub>l<sub>og gap cu</sub>r<sub>ves, a</sub>r<sub>e app</sub>r<sub>ox</sub>im<sub>a</sub>t<sub>e</sub>l<sub>y</sub> 0<sub>.</sub>487 f<sub>o</sub>r $\lambda = 1 0 ^ { - 2 }$ <sub>an</sub>d 0<sub>.</sub>509 f<sub>or</sub> $\lambda = 1 / 3$ <sub>,</sub> <sub>compare</sub>d <sub>w</sub>ith <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub>l<sub>y</sub> 1 f<sub>or</sub> h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on.</sub> Th<sub>ese</sub> fit<sub>s are cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>w</sub>ith $T ^ { - 1 / 2 }$ versus $T ^ { - 1 }$ <sub>average-</sub>it<sub>era</sub>t<sub>e</sub> d<sub>ecay over</sub> th<sub>e</sub> t<sub>es</sub>t<sub>e</sub>d i<sub>n</sub>t<sub>erva</sub>l<sub>.</sub> I<sub>n</sub> th<sub>e same runs,</sub> th<sub>e</sub> l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e gaps</sub> f<sub>or</sub> th<sub>e</sub> t<sub>wo non-</sub>h<sub>a</sub>lf <sub>mu</sub>lti<sub>p</sub>li<sub>ers rema</sub>i<sub>n</sub> l<sub>arge an</sub>d <sub>osc</sub>ill<sub>a</sub>t<sub>ory,</sub> <sub>w</sub>h<sub>ereas</sub> h<sub>a</sub>lf <sub>pre</sub>di<sub>c</sub>ti<sub>on re</sub>d<sub>uces</sub> th<sub>em</sub> t<sub>o</sub> th<sub>e numer</sub>i<sub>ca</sub>l <sub>prec</sub>i<sub>s</sub>i<sub>on</sub> li<sub>m</sub>it<sub>.</sub> T<sub>o exam</sub>i<sub>ne</sub> th<sub>ese osc</sub>ill<sub>a</sub>ti<sub>ons more c</sub>l<sub>ose</sub>l<sub>y,</sub> Fi<sub>gure</sub> 8 <sub>s</sub>h<sub>ows every</sub> it<sub>era</sub>t<sub>e</sub> i<sub>n</sub> th<sub>e</sub> fi<sub>na</sub>l $2 0 0 { , } 0 0 0$ it<sub>era</sub>ti<sub>ons</sub> b<sub>e</sub>f<sub>ore</sub> $1 0 ^ { 8 } , 5 \times 1 0 ^ { 8 }$ <sub>,</sub> <sub>an</sub>d $1 0 ^ { 9 }$ f<sub>or</sub> $\lambda = 1 / 3$ <sub>on</sub> th<sub>e</sub> l<sub>arger</sub> bl<sub>oc</sub>k <sub>game.</sub> Th<sub>e m</sub>i<sub>n</sub>i<sub>mum gaps</sub> i<sub>n</sub> th<sub>ese w</sub>i<sub>n</sub>d<sub>ows are</sub> 0<sub>.</sub>451095<sub>,</sub> 0<sub>.</sub>453408<sub>, an</sub>d 0<sub>.</sub>453838<sub>, respec</sub>ti<sub>ve</sub>l<sub>y,</sub> i<sub>ncreas</sub>i<sub>ng s</sub>li<sub>g</sub>htl<sub>y</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> d<sub>ecreas</sub>i<sub>ng</sub> t<sub>owar</sub>d <sub>zero.</sub> F<sub>rom</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>w</sub>i<sub>n</sub>d<sub>ow</sub> t<sub>o</sub> th<sub>e</sub> l<sub>as</sub>t<sub>,</sub> th<sub>e m</sub>i<sub>n</sub>i<sub>mum gap</sub> i<sub>ncreases</sub> b<sub>y approx</sub>i<sub>ma</sub>t<sub>e</sub>l<sub>y</sub>

![](images/9add6c8fecbbd9c581f25ab0138fe23370db0b041b7fc6ccc5e2f1a0db8f2039.jpg)

![](images/2f8aa6b825d3f9e4597e90b4ce1afed465bf6fa944e0c92e93025635535edfd7.jpg)

![](images/79d2dad62112a7e4ec6820454d3ad6118d7632721be6a1d2d0dbcc677bd91763.jpg)  
Fi<sub>g</sub>ure 8: Last-iterate <sub>g</sub>a<sub>p</sub> for � = 1/3 on the 177,702×177,702 block <sub>g</sub>ame over three late windows of200,000 iterations, <sub>w</sub>ith id<sub>en</sub>ti<sub>ca</sub>l <sub>ax</sub>i<sub>s sca</sub>l<sub>es.</sub> Ci<sub>rc</sub>l<sub>es mar</sub>k th<sub>e m</sub>i<sub>n</sub>i<sub>mum gap w</sub>ithi<sub>n eac</sub>h <sub>comp</sub>l<sub>e</sub>t<sub>e cyc</sub>l<sub>e,</sub> d<sub>e</sub>fi<sub>ne</sub>d b<sub>y consecu</sub>ti<sub>ve upwar</sub>d <sub>cross</sub>i<sub>ngs o</sub>f <sub>gap</sub> 10<sub>.</sub> I<sub>ncomp</sub>l<sub>e</sub>t<sub>e cyc</sub>l<sub>es a</sub>t th<sub>e w</sub>i<sub>n</sub>d<sub>ow</sub> b<sub>oun</sub>d<sub>ar</sub>i<sub>es are</sub> di<sub>sp</sub>l<sub>aye</sub>d b<sub>u</sub>t <sub>exc</sub>l<sub>u</sub>d<sub>e</sub>d f<sub>rom cyc</sub>l<sub>e averages.</sub>

0<sub>.</sub>61% <sub>an</sub>d th<sub>e mean cyc</sub>l<sub>e</sub> t<sub>roug</sub>h b<sub>y</sub> 0<sub>.</sub>53%<sub>, w</sub>hil<sub>e</sub> th<sub>e mean pea</sub>k<sub>-</sub>t<sub>o-</sub>t<sub>roug</sub>h <sub>amp</sub>lit<sub>u</sub>d<sub>e rema</sub>i<sub>ns</sub> b<sub>e</sub>t<sub>ween</sub> 19<sub>.</sub>26 <sub>an</sub>d 19<sub>.</sub>96<sub>.</sub> I<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>cyc</sub>l<sub>e</sub> t<sub>roug</sub>h<sub>s</sub> fl<sub>uc</sub>t<sub>ua</sub>t<sub>e</sub> i<sub>n</sub> b<sub>o</sub>th di<sub>rec</sub>ti<sub>ons w</sub>ithi<sub>n eac</sub>h <sub>w</sub>i<sub>n</sub>d<sub>ow.</sub> Th<sub>ese o</sub>b<sub>serva</sub>ti<sub>ons s</sub>t<sub>reng</sub>th<sub>en</sub> th<sub>e numer</sub>i<sub>ca</sub>l <sub>ev</sub>id<sub>ence o</sub>f <sub>pers</sub>i<sub>s</sub>t<sub>en</sub>t l<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e osc</sub>ill<sub>a</sub>ti<sub>ons,</sub> b<sub>u</sub>t d<sub>o no</sub>t <sub>r</sub>i<sub>gorous</sub>l<sub>y es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>a mono</sub>t<sub>one</sub> t<sub>ren</sub>d <sub>or an</sub> <sub>asymp</sub>t<sub>o</sub>ti<sub>c</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>.</sub>

## B.2 Visualizations for the Separation of Alternating and Simultaneous Updates

Fi<sub>g</sub>ure 9 visualizes the trajectories of Alt-HPRM<sup>+</sup> and Sim-HPRM<sup>+</sup> on the se<sub>p</sub>aration <sub>g</sub>ame (30). The trian<sub>g</sub>ular re<sub>p</sub>resentation is similar to that in Nan et al. [2026a, Fi<sub>g</sub>ure 2], where the AltGDA trajectories settle into c<sub>y</sub>cles while their avera<sub>g</sub>es conver<sub>g</sub>e. Here, the Alt-HPRM<sup>+</sup> trajectories in <sub>p</sub>lot (b) instead exhibit dam<sub>p</sub>ed oscillations toward <sub>equ</sub>ilib<sub>r</sub>i<sub>um.</sub>

## B.3 Further Details for Performance on Extensive-Form Benchmarks

All <sub>s</sub>i<sub>x me</sub>th<sub>o</sub>d<sub>s are</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>e</sub>d i<sub>n a common</sub> CPU <sub>co</sub>d<sub>e</sub>b<sub>ase us</sub>i<sub>ng</sub> P<sub>y</sub>th<sub>on</sub> 3<sub>.</sub>13<sub>.</sub>16<sub>,</sub> N<sub>um</sub>P<sub>y</sub> 2<sub>.</sub>5<sub>.</sub>4<sub>,</sub> O<sub>pen</sub>S<sub>p</sub>i<sub>e</sub>l 2<sub>.</sub>0<sub>.</sub>2<sub>,</sub> <sub>an</sub>d <sub>a</sub> <sub>mo</sub>difi<sub>e</sub>d Lit<sub>e</sub>EFG 1<sub>.</sub>0<sub>.</sub>0 b<sub>ac</sub>k<sub>en</sub>d<sub>.</sub> T<sub>ree-</sub>b<sub>ase</sub>d <sub>games</sub> <sub>use</sub> <sub>comp</sub>il<sub>e</sub>d <sub>ex</sub>h<sub>aus</sub>ti<sub>ve</sub> t<sub>raversa</sub>l<sub>,</sub> <sub>w</sub>hil<sub>e</sub> HUNL <sub>uses</sub> <sub>a</sub> <sub>pu</sub>bli<sub>c-</sub>t<sub>ree</sub> b<sub>ac</sub>k<sub>en</sub>d<sub>;</sub> <sub>w</sub>ithi<sub>n</sub> <sub>eac</sub>h <sub>game,</sub> <sub>a</sub>ll <sub>me</sub>th<sub>o</sub>d<sub>s</sub> <sub>s</sub>h<sub>are</sub> th<sub>e</sub> <sub>game</sub> <sub>represen</sub>t<sub>a</sub>ti<sub>on,</sub> t<sub>raversa</sub>l<sub>,</sub> <sub>averag</sub>i<sub>ng,</sub> <sub>an</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> routines. Com<sub>p</sub>utations use float64, zero initial re<sub>g</sub>rets, and uniform initial strate<sub>g</sub>ies. Positive strate<sub>gy</sub> wei<sub>g</sub>hts are normalized without an � floor, with uniform fallback onl<sub>y</sub> at zero mass. Alt-PCFR<sup>+</sup>, Alt-HPCFR<sup>+</sup>, and Alt-SAPCFR<sup>+</sup> <sub>s</sub>h<sub>are</sub> th<sub>e same</sub> l<sub>oca</sub>l <sub>up</sub>d<sub>a</sub>t<sub>e</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on w</sub>ith <sub>pre</sub>di<sub>c</sub>ti<sub>on coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> 1<sub>,</sub> $1 / 2 ,$ <sub>an</sub>d $1 / 3 ,$ <sub>respec</sub>ti<sub>ve</sub>l<sub>y.</sub> F<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> Xu et al. [2024], $\mathsf { A l t { - } P D C F R { ^ + } }$ <sub>an</sub>d $\mathsf { A l t - D C F R ^ { + } }$ use re<sub>g</sub>ret-<sup>di</sup>scount ex<sub>p</sub>onents $\alpha = 2 . 3$ <sub>an</sub>d $\alpha = 1 . 5$ , res<sub>p</sub>ect<sup>i</sup>ve<sup>l</sup><sub>y</sub>, <sub>mu</sub>lti<sub>p</sub>l<sub>y</sub>i<sub>ng</sub> <sub>pas</sub>t <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>regre</sub>t<sub>s</sub> b<sub>y</sub> $( t - 1 ) ^ { \alpha } / ( ( t - 1 ) ^ { \alpha } + 1 )$ b<sub>e</sub>f<sub>ore</sub> <sub>a</sub>ddi<sub>ng</sub> th<sub>e</sub> <sub>curren</sub>t <sub>regre</sub>t <sub>an</sub>d <sub>c</sub>li<sub>pp</sub>i<sub>ng</sub> <sub>a</sub>t <sub>zero.</sub> N<sub>e</sub>ith<sub>er me</sub>th<sub>o</sub>d <sub>uses a nega</sub>ti<sub>ve-regre</sub>t di<sub>scoun</sub>t <sub>exponen</sub>t ${ } _ { \beta , { } }$ <sub>, s</sub>i<sub>nce nega</sub>ti<sub>ve cumu</sub>l<sub>a</sub>ti<sub>ve regre</sub>t<sub>s are c</sub>li<sub>ppe</sub>d<sub>.</sub> Th<sub>e paper</sub>’<sub>s</sub> <sup>a</sup>v<sup>era</sup>g<sup>in</sup>g <sup>ex</sup>p<sup>onents</sup> $\gamma = 5$ <sub>an</sub>d $\gamma = 4$ <sub>, respec</sub>ti<sub>ve</sub>l<sub>y, ass</sub>i<sub>gn</sub> it<sub>era</sub>ti<sub>on we</sub>i<sub>g</sub>ht<sub>s</sub> $t ^ { \gamma } ;$ <sub>we rep</sub>l<sub>ace</sub> th<sub>ese ou</sub>t<sub>pu</sub>t <sub>we</sub>i<sub>g</sub>ht<sub>s w</sub>ith th<sub>e common averag</sub>i<sub>ng ru</sub>l<sub>es</sub> i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 9<sub>, w</sub>hil<sub>e re</sub>t<sub>a</sub>i<sub>n</sub>i<sub>ng</sub> th<sub>e paper</sub>’<sub>s</sub> l<sub>oca</sub>l <sub>recurrences.</sub>

![](images/fb202915f4be092e436ec3302026b0652f005e6c77e462d494ffd5988560dd7f.jpg)

![](images/690d9859914a9fb3036b1e3a5df8fd7d8c12fa021fe50efba1f352f5bd2e6438.jpg)

Fi<sub>g</sub>ure 9: Half <sub>p</sub>rediction under simultaneous and alternatin<sub>g</sub> u<sub>p</sub>dates (stars mark the equilibrium, and o<sub>p</sub>en markers denote iteration 2). (a) Simultaneous strate<sub>gy</sub> trajectories. (b) Alternatin<sub>g</sub> trajectories, enlar<sub>g</sub>in<sub>g</sub> the dashed trian<sub>g</sub>ular re<sub>g</sub>ion in (a). (c) Joint Euclidean distance to equilibrium.
<table><tr><td>Game</td><td>Histories</td><td>Information sets</td><td>Terminal histories</td><td>No. of iterations</td></tr><tr><td>Kuhn poker</td><td>58</td><td>12</td><td>30</td><td>200</td></tr><tr><td>GoofspielImp (4 cards)</td><td>1,077</td><td>162</td><td>576</td><td>500</td></tr><tr><td>Liar&#x27;s Dice (4 sides)</td><td>8,181</td><td>1,024</td><td>4,080</td><td>1000</td></tr><tr><td>Leduc poker</td><td>9,457</td><td>936</td><td>5,520</td><td>50000</td></tr><tr><td>GoofspielImp (5 cards)</td><td>26,931</td><td>2,124</td><td>14,400</td><td>50000</td></tr><tr><td>Liar&#x27;s Dice (5 sides)</td><td>51,181</td><td>5,120</td><td>25,575</td><td>50000</td></tr><tr><td>Pursuit-evasion (5 turns)</td><td>87,927</td><td>2,100</td><td>61,000</td><td>50000</td></tr><tr><td>Liar&#x27;s Dice (6 sides)</td><td>294,883</td><td>24,576</td><td>147,420</td><td>100000</td></tr><tr><td>Leduc poker (9 ranks)</td><td>371,809</td><td>9,288</td><td>221,544</td><td>100000</td></tr><tr><td>Battleship (3 × 2, 3 shots)</td><td>732,607</td><td>81,027</td><td>552,132</td><td>10000</td></tr><tr><td>Leduc poker (13 ranks)</td><td>1,179,777</td><td>19,656</td><td>704,600</td><td>100000</td></tr><tr><td>Liar&#x27;s Dice (8 sides)</td><td>8,388,553</td><td>524,288</td><td>4,194,240</td><td>5000</td></tr><tr><td>HUNL subgame 4</td><td>244,005,483</td><td>43,240</td><td>158,388,120</td><td>100000</td></tr><tr><td>HUNL subgame 3</td><td>398,112,843</td><td>69,184</td><td>261,126,360</td><td>100000</td></tr><tr><td>HUNL subgame 2</td><td>≈  $; 3 . 5 \times 1 0 ^ { 1 0 }$ </td><td> $\approx 6 . 7 \times 1 0 ^ { 6 }$ </td><td>≈  $2 . 2 \times 1 0 ^ { 1 0 }$ </td><td>10000</td></tr></table>

Table 2: Benchmark <sub>g</sub>ames (the number of information sets counts those of both <sub>p</sub>la<sub>y</sub>ers).

Th<sub>e</sub> it<sub>era</sub>ti<sub>on</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>s are g</sub>i<sub>ven</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 2<sub>.</sub> B<sub>o</sub>th <sub>averages accumu</sub>l<sub>a</sub>t<sub>e pos</sub>t<sub>-</sub>it<sub>era</sub>ti<sub>on rea</sub>li<sub>za</sub>ti<sub>on p</sub>l<sub>ans a</sub>t <sub>every</sub> it<sub>era</sub>ti<sub>on, an</sub>d HS30 <sub>uses</sub> th<sub>e</sub> f<sub>u</sub>ll d<sub>ec</sub>l<sub>are</sub>d <sub>run</sub> h<sub>or</sub>i<sub>zon,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng w</sub>h<sub>en a</sub> fi<sub>gure</sub> di<sub>sp</sub>l<sub>ays a s</sub>h<sub>or</sub>t<sub>er</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>.</sub> All th<sub>ree</sub> <sub>ou</sub>t<sub>pu</sub>t<sub>s are eva</sub>l<sub>ua</sub>t<sub>e</sub>d b<sub>y exac</sub>t b<sub>es</sub>t <sub>responses a</sub>t id<sub>en</sub>ti<sub>ca</sub>l <sub>c</sub>h<sub>ec</sub>k<sub>po</sub>i<sub>n</sub>t<sub>s across me</sub>th<sub>o</sub>d<sub>s: every</sub> it<sub>era</sub>ti<sub>on</sub> th<sub>roug</sub>h 20<sub>,</sub> ever<sub>y</sub> 5 iterations throu<sub>g</sub>h 100, ever<sub>y</sub> 10 throu<sub>g</sub>h 500, ever<sub>y</sub> 50 throu<sub>g</sub>h 5000, and thereafter ever<sub>y</sub> max{50, ⌊�/1000⌋} it<sub>era</sub>ti<sub>ons, a</sub>l<sub>ways</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> �<sub>.</sub> R<sub>uns</sub> d<sub>o no</sub>t <sub>s</sub>t<sub>op ear</sub>l<sub>y upon reac</sub>hi<sub>ng a</sub> t<sub>arge</sub>t<sub>.</sub> T<sub>arge</sub>t <sub>compar</sub>i<sub>sons use</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>save</sub>d h<sub>or</sub>i<sub>zons an</sub>d th<sub>e</sub> fi<sub>rs</sub>t <sub>save</sub>d <sub>cross</sub>i<sub>ng, w</sub>ith<sub>ou</sub>t i<sub>n</sub>t<sub>erpo</sub>l<sub>a</sub>ti<sub>on or a pers</sub>i<sub>s</sub>t<sub>ence requ</sub>i<sub>remen</sub>t<sub>.</sub> Ch<sub>ec</sub>k<sub>po</sub>i<sub>n</sub>t <sub>exp</sub>l<sub>o</sub>it<sub>a</sub>biliti<sub>es</sub> <sub>an</sub>d ti<sub>mers are save</sub>d<sub>,</sub> t<sub>oge</sub>th<sub>er w</sub>ith fi<sub>na</sub>l <sub>po</sub>li<sub>c</sub>i<sub>es,</sub> b<sub>u</sub>t f<sub>u</sub>ll <sub>s</sub>t<sub>ra</sub>t<sub>egy</sub> hi<sub>s</sub>t<sub>or</sub>i<sub>es an</sub>d hi<sub>s</sub>t<sub>or</sub>i<sub>ca</sub>l b<sub>es</sub>t<sub>-po</sub>li<sub>cy cop</sub>i<sub>es are no</sub>t <sub>re</sub>t<sub>a</sub>i<sub>ne</sub>d<sub>.</sub>

Th<sub>e exper</sub>i<sub>men</sub>t<sub>s are run on</sub> A<sub>pp</sub>l<sub>e</sub> M5 P<sub>ro w</sub>ith 18 CPU <sub>cores an</sub>d 24 GiB <sub>o</sub>f <sub>memory;</sub> th<sub>e run</sub> l<sub>ogs recor</sub>d <sub>arm</sub>64 <sub>mac</sub>OS 26<sub>.</sub>6<sub>.</sub>2<sub>.</sub> E<sub>ac</sub>h l<sub>earner uses one na</sub>ti<sub>ve</sub> th<sub>rea</sub>d<sub>, w</sub>ith <sub>numer</sub>i<sub>ca</sub>l<sub>-</sub>lib<sub>rary</sub> th<sub>rea</sub>d <sub>poo</sub>l<sub>s a</sub>l<sub>so</sub> li<sub>m</sub>it<sub>e</sub>d t<sub>o one.</sub> E<sub>ac</sub>h <sub>game–me</sub>th<sub>o</sub>d <sub>pa</sub>i<sub>r con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es one run, w</sub>ith<sub>ou</sub>t <sub>aggrega</sub>ti<sub>on across repe</sub>titi<sub>ons.</sub> L<sub>earners run sequen</sub>ti<sub>a</sub>ll<sub>y.</sub> R<sub>un</sub>ti<sub>me</sub> <sub>convergence p</sub>l<sub>o</sub>t<sub>s repor</sub>t <sub>cumu</sub>l<sub>a</sub>ti<sub>ve</sub> l<sub>earner wa</sub>ll ti<sub>me,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> t<sub>raversa</sub>l<sub>,</sub> l<sub>oca</sub>l <sub>up</sub>d<sub>a</sub>t<sub>es, reac</sub>h b<sub>oo</sub>kk<sub>eep</sub>i<sub>ng,</sub> b<sub>o</sub>th <sub>aver-</sub> <sub>age accumu</sub>l<sub>a</sub>ti<sub>ons, an</sub>d <sub>c</sub>h<sub>ec</sub>k<sub>po</sub>i<sub>n</sub>t <sub>average-po</sub>li<sub>cy ma</sub>t<sub>er</sub>i<sub>a</sub>li<sub>za</sub>ti<sub>on.</sub> Th<sub>e run</sub>ti<sub>me ra</sub>ti<sub>os</sub> i<sub>n</sub> Fi<sub>gure</sub> 5 <sub>a</sub>dditi<sub>ona</sub>ll<sub>y</sub> i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> <sub>recor</sub>d<sub>e</sub>d <sub>se</sub>t<sub>up an</sub>d <sub>exp</sub>l<sub>o</sub>it<sub>a</sub>bilit<sub>y-eva</sub>l<sub>ua</sub>ti<sub>on</sub> ti<sub>me.</sub> N<sub>e</sub>ith<sub>er c</sub>l<sub>oc</sub>k i<sub>nc</sub>l<sub>u</sub>d<sub>es</sub> l<sub>ogg</sub>i<sub>ng,</sub> fi<sub>na</sub>l <sub>po</sub>li<sub>cy expor</sub>t<sub>,</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>va</sub>lid<sub>a</sub>ti<sub>on, or p</sub>l<sub>o</sub>tti<sub>ng;</sub> t<sub>ree prepara</sub>ti<sub>on an</sub>d <sub>grap</sub>h <sub>cons</sub>t<sub>ruc</sub>ti<sub>on a</sub>l<sub>so prece</sub>d<sub>e</sub> th<sub>e recor</sub>d<sub>e</sub>d <sub>se</sub>t<sub>up</sub> ti<sub>mer.</sub> Th<sub>e</sub> ti<sub>m</sub>i<sub>ngs</sub> <sub>are</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>ve s</sub>i<sub>ng</sub>l<sub>e-run measuremen</sub>t<sub>s, so near-un</sub>it <sub>ra</sub>ti<sub>os s</sub>h<sub>ou</sub>ld <sub>no</sub>t b<sub>e</sub> i<sub>n</sub>t<sub>erpre</sub>t<sub>e</sub>d <sub>as re</sub>li<sub>a</sub>bl<sub>e percen</sub>t<sub>-</sub>l<sub>eve</sub>l <sub>spee</sub>d dif<sub>erences.</sub>

## B.4 Extra Plots for Benchmark Games

In th<sub>e</sub> b<sub>o</sub>d<sub>y o</sub>f thi<sub>s pape</sub>r<sub>, we s</sub>h<sub>owe</sub>d th<sub>e</sub> l<sub>as</sub>t<sub>-</sub>it<sub>e</sub>r<sub>a</sub>t<sub>e co</sub>n<sub>ve</sub>r<sub>ge</sub>n<sub>ce a</sub>nd HS30 <sub>ave</sub>r<sub>age co</sub>n<sub>ve</sub>r<sub>ge</sub>n<sub>ce p</sub>l<sub>o</sub>t<sub>s.</sub> In Fi<sub>gu</sub>r<sub>e</sub> 10<sub>,</sub> <sub>we s</sub>h<sub>ow</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>average-</sub>it<sub>era</sub>t<sub>e convergence.</sub> W<sub>e</sub> h<sub>ave a</sub>l<sub>so prov</sub>id<sub>e</sub>d <sub>exper</sub>i<sub>men</sub>t<sub>a</sub>l <sub>p</sub>l<sub>o</sub>t<sub>s s</sub>h<sub>ow</sub>i<sub>ng exp</sub>l<sub>o</sub>it<sub>a</sub>bilit<sub>y</sub> <sub>as a</sub> f<sub>unc</sub>ti<sub>on o</sub>f it<sub>era</sub>ti<sub>ons</sub> i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 9<sub>.</sub> H<sub>ere</sub> i<sub>n</sub> Fi <sub>ures</sub> 11<sub>,</sub> 12<sub>, an</sub>d 13<sub>, we rov</sub>id<sub>e ex</sub> l<sub>o</sub>it<sub>a</sub>bilit <sub>as a</sub> f<sub>unc</sub>ti<sub>on o</sub>f <sub>run</sub> ti<sub>me.</sub>

![](images/da4468092deeaf41e58d2c49a811e5e7a10de3984223cc55167dfd84cc79c94e.jpg)  
† Restricted y-range: high early values are outside the panel. All x-axes start at zero.

Fi<sub>gure</sub> 10<sub>:</sub> A<sub>verage-</sub>it<sub>era</sub>t<sub>e convergence aga</sub>i<sub>ns</sub>t it<sub>era</sub>ti<sub>on coun</sub>t<sub>.</sub>

![](images/f6b4be802c82b280c9497176c4acbbaf36a4dfab31dbf89c2821a5a18188d123.jpg)  
† Restricted y-range: high early values are outside the panel. All x-axes start at zero.

Fi<sub>gure</sub> 11<sub>:</sub> L<sub>as</sub>t<sub>-</sub>it<sub>era</sub>t<sub>e convergence aga</sub>i<sub>ns</sub>t l<sub>earner wa</sub>ll ti<sub>me.</sub>

Alt-PCFR+ (λ = 1) Alt-HPCFR+ (λ = 1/2) Alt-SAPCFR+ (λ = 1/3) Alt-IR-PCFR+ Alt-PDCFR+ Alt-DCFR

![](images/71490a0b28a7c1c51f637392fd3ea045eef1dec0d2ca9e071686e33ce5149ed9.jpg)  
† Restricted y-range: high early values are outside the panel. All x-axes start at zero.

Fi<sub>gu</sub>r<sub>e</sub> 12<sub>:</sub> HS30 <sub>ave</sub>r<sub>age</sub> <sub>co</sub>n<sub>ve</sub>r<sub>ge</sub>n<sub>ce</sub> <sub>aga</sub>in<sub>s</sub>t l<sub>ea</sub>rn<sub>e</sub>r <sub>wa</sub>ll tim<sub>e.</sub>

Alt-PCFR+ (λ = 1) Alt-HPCFR+ (λ = 1/2) Alt-SAPCFR+ (λ = 1/3) Alt-IR-PCFR+ Alt-PDCFR+ Alt-DCFR+

![](images/a9e8ea9881e1cb4b26c9221a42f69e47f22e4323dcc3e75d8547c609f4e00668.jpg)  
† Restricted y-range: high early values are outside the panel. All x-axes start at zero.

Fi<sub>gure</sub> 13<sub>:</sub> A<sub>verage-</sub>it<sub>era</sub>t<sub>e convergence aga</sub>i<sub>ns</sub>t l<sub>earner wa</sub>ll ti<sub>me.</sub>