# Selecting the Best Few with Human Evaluators: Adaptively Combining Ratings and Pairwise Comparisons

M<sub>o</sub>tti G<sub>o</sub>ldb<sub>erger</sub>

Y<sub>a</sub>l<sub>e</sub> S<sub>c</sub>h<sub>oo</sub>l <sub>o</sub>f M<sub>anagemen</sub>t<sub>, mo</sub>tti<sub>.go</sub>ldb<sub>erger</sub>@<sub>ya</sub>l<sub>e.e</sub>d<sub>u</sub>

Nil<sub>s</sub> R<sub>u</sub>di

Y<sub>a</sub>l<sub>e</sub> S<sub>c</sub>h<sub>oo</sub>l <sub>o</sub>f M<sub>anagemen</sub>t<sub>, n</sub>il<sub>s.ru</sub>di@<sub>ya</sub>l<sub>e.e</sub>d<sub>u</sub>

Abstract. This paper considers a decision-maker tasked with identifying the � best items from a large pool of candidates, such as job a licants LLMs award nominees or new roduct ideas. The decision-maker learns about the items’ ualit b solicitin h<sub>uman eva</sub>l<sub>ua</sub>ti<sub>ons.</sub> E<sub>ac</sub>h <sub>eva</sub>l<sub>ua</sub>ti<sub>on can</sub> b<sub>e cos</sub>tl<sub>y, so</sub> th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>k<sub>er a</sub>i<sub>ms</sub> t<sub>o use</sub> th<sub>e</sub> li<sub>m</sub>it<sub>e</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> b <sub>a</sub>ll<sub>oca</sub>ti<sub>n eva</sub>l<sub>ua</sub>ti<sub>ons a</sub>d<sub>a</sub> ti<sub>ve</sub>l <sub>.</sub> A k<sub>e cons</sub>id<sub>era</sub>ti<sub>on</sub> i<sub>s</sub> th<sub>e</sub> f<sub>orm o</sub>f <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>o so</sub>li<sub>c</sub>it <sub>w</sub>hi<sub>c</sub>h <sub>s</sub>h<sub>ou</sub>ld b<sub>e some</sub>thi<sub>n eva</sub>l<sub>ua</sub>t<sub>ors ar</sub> <sub>com</sub>f<sub>or</sub>t<sub>a</sub>bl<sub>e</sub> <sub>prov</sub>idi<sub>ng.</sub> T<sub>wo</sub> <sub>popu</sub>l<sub>ar</sub> f<sub>orms</sub> <sub>are</sub> <sub>ra</sub>ti<sub>ngs</sub> <sub>an</sub>d <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>compar</sub>i<sub>sons,</sub> <sub>an</sub>d <sub>we</sub> <sub>a</sub>ll<sub>ow</sub> f<sub>or</sub> b<sub>o</sub>th<sub>,</sub> <sub>as</sub> th<sub>ey</sub> <sub>may</sub> b<sub>e</sub> <sub>espec</sub>i<sub>a</sub>ll <sub>use</sub>f<sub>u</sub>l i<sub>n com</sub>bi<sub>na</sub>ti<sub>on g</sub>i<sub>ven</sub> th<sub>e</sub> li<sub>m</sub>it<sub>e</sub>d b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub> R<sub>a</sub>ti<sub>ngs can</sub> b<sub>e use</sub>d <sub>ear</sub>l<sub>y on</sub> t<sub>o ge</sub>t <sub>a coarse p</sub>i<sub>c</sub>t<sub>ure o</sub>f <sub>w</sub>hi<sub>c</sub>h it<sub>ems are goo</sub>d <sub>an</sub>d <sub>w</sub>hi<sub>c</sub>h <sub>are</sub> <sub>no</sub>t<sub>.</sub> A<sub>s</sub> th<sub>a</sub>t <sub>p</sub>i<sub>c</sub>t<sub>ure</sub> f<sub>orms,</sub> <sub>compar</sub>i<sub>sons</sub> <sub>can</sub> <sub>re</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>or</sub>d<sub>er</sub>i<sub>ng</sub> <sub>o</sub>f th<sub>e</sub> <sub>goo</sub>d it<sub>ems,</sub> <sub>narrow</sub>i<sub>ng</sub> i<sub>n</sub> <sub>on</sub> th<sub>e</sub> <sub>u</sub>lti<sub>ma</sub>t<sub>e</sub> t<sub>op-</sub>� <sub>se</sub>l<sub>ec</sub>ti<sub>on.</sub> T<sub>o so</sub>l<sub>ve</sub> th<sub>e a</sub>ll<sub>oca</sub>ti<sub>on pro</sub>bl<sub>em, we</sub> d<sub>eve</sub>l<sub>op an a</sub>d<sub>ap</sub>ti<sub>ve a</sub>l<sub>gor</sub>ith<sub>m</sub> b<sub>u</sub>ilt f<sub>rom</sub> t<sub>wo a</sub>ll<sub>oca</sub>ti<sub>on po</sub>li<sub>c</sub>i<sub>es.</sub> Th<sub>e</sub> fi<sub>rs</sub>t<sub>, our pr</sub>i<sub>mar</sub> contribution, is based on a surrogate objective that is naturally forward-looking, valuing evaluations that help inform what to evaluate in future rounds, not just those that directly improve the selection decision. The decision rule is interpretable, and we prove that it is <sub>asymp</sub>t<sub>o</sub>ti<sub>ca</sub>ll<sub>y cons</sub>i<sub>s</sub>t<sub>en</sub>t<sub>.</sub> L<sub>a</sub>t<sub>e</sub> i<sub>n</sub> th<sub>e eva</sub>l<sub>ua</sub>ti<sub>on process, w</sub>h<sub>en</sub> b<sub>e</sub>i<sub>ng</sub> f<sub>orwar</sub>d<sub>-</sub>l<sub>oo</sub>ki<sub>ng</sub> i<sub>s</sub> l<sub>ess pro</sub>d<sub>uc</sub>ti<sub>ve, we cons</sub>t<sub>ruc</sub>t <sub>a myop</sub>i<sub>c</sub> <sub>po</sub>li<sub>cy an</sub>d <sub>a cr</sub>it<sub>er</sub>i<sub>on</sub> f<sub>or sw</sub>it<sub>c</sub>hi<sub>ng</sub> t<sub>o</sub> it<sub>.</sub> I<sub>n s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on,</sub> th<sub>e me</sub>th<sub>o</sub>d <sub>g</sub>i<sub>ves</sub> i<sub>ns</sub>i<sub>g</sub>ht i<sub>n</sub>t<sub>o</sub> h<sub>ow</sub> t<sub>o com</sub>bi<sub>ne</sub> th<sub>ese</sub> t<sub>wo</sub> t<sub>ypes o</sub>f <sub>eva</sub>l<sub>ua</sub>ti<sub>ons,</sub> and beats the (limited) existin<sub>g</sub> benchmarks.

Key words: ranking and selection, top-� identification, pairwise comparisons, ratings, human evaluations

## 1. Introduction

Thi<sub>s</sub> <sub>paper</sub> <sub>cons</sub>id<sub>ers</sub> <sub>a</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>k<sub>er</sub> t<sub>as</sub>k<sub>e</sub>d <sub>w</sub>ith id<sub>en</sub>tif<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> � b<sub>es</sub>t it<sub>ems</sub> f<sub>rom</sub> <sub>a</sub> l<sub>arge</sub> <sub>poo</sub>l <sub>o</sub>f <sub>can</sub>did<sub>a</sub>t<sub>es.</sub> Thi<sub>s</sub> i<sub>s</sub> d<sub>one</sub> <sub>us</sub>i<sub>ng</sub> d<sub>a</sub>t<sub>a</sub> th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>k<sub>er</sub> <sub>sources</sub> <sub>sequen</sub>ti<sub>a</sub>ll<sub>y</sub> f<sub>rom</sub> h<sub>uman</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>ors,</sub> <sub>w</sub>h<sub>ere</sub> th<sub>e</sub> <sub>a</sub>d<sub>ap</sub>ti<sub>ve</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>o</sub>f <sub>upcom</sub>i<sub>ng</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> i<sub>s</sub> i<sub>n</sub>f<sub>orme</sub>d b<sub>y</sub> <sub>w</sub>h<sub>a</sub>t h<sub>as</sub> b<sub>een</sub> l<sub>earne</sub>d <sub>so</sub> f<sub>ar.</sub> Wh<sub>en</sub> <sub>mo</sub>d<sub>e</sub>li<sub>ng</sub> <sub>suc</sub>h h<sub>uman</sub> judgment, a key decision is the form of the evaluation input. While one may wish for some sophisticated <sub>mu</sub>ltidi<sub>mens</sub>i<sub>ona</sub>l i<sub>npu</sub>t<sub>,</sub> <sub>prov</sub>idi<sub>ng</sub> it i<sub>s</sub> <sub>o</sub>ft<sub>en</sub> b<sub>eyon</sub>d th<sub>e</sub> <sub>scope,</sub> ti<sub>me,</sub> <sub>an</sub>d <sub>a</sub>bilit<sub>y</sub> f<sub>or</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>ors,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>w</sub>h<sub>y</sub> i<sub>npu</sub>t<sub>s</sub> <sub>suc</sub>h <sub>as</sub> th<sub>um</sub>b<sub>s-up</sub>/th<sub>um</sub>b<sub>s-</sub>d<sub>own</sub> <sub>an</sub>d <sub>sw</sub>i<sub>p</sub>i<sub>ng</sub> l<sub>e</sub>ft <sub>or</sub> <sub>r</sub>i<sub>g</sub>ht h<sub>ave</sub> b<sub>ecome</sub> <sub>popu</sub>l<sub>ar.</sub> R<sub>a</sub>th<sub>er</sub> th<sub>an</sub> <sub>res</sub>t<sub>r</sub>i<sub>c</sub>t th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>k<sub>er</sub> t<sub>o so</sub>li<sub>c</sub>iti<sub>ng one</sub> t<sub>ype o</sub>f i<sub>npu</sub>t<sub>, we a</sub>ll<sub>ow</sub> f<sub>or</sub> th<sub>e com</sub>bi<sub>na</sub>ti<sub>on o</sub>f th<sub>um</sub>b<sub>s-up</sub>/th<sub>um</sub>b<sub>s-</sub>d<sub>own</sub> <sub>ra</sub>ti<sub>ngs</sub> <sub>an</sub>d <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>compar</sub>i<sub>sons,</sub> <sub>w</sub>hi<sub>c</sub>h<sub>,</sub> <sub>as</sub> <sub>we</sub> <sub>w</sub>ill di<sub>scuss,</sub> <sub>can</sub> b<sub>e</sub> <sub>comp</sub>l<sub>emen</sub>t<sub>ary</sub> i<sub>n</sub> <sub>our</sub> <sub>pro</sub>bl<sub>em.</sub> Thi<sub>s</sub> model can form a basis for many applications, including selecting the best few among LLMs, job applicants, <sub>un</sub>i<sub>vers</sub>it<sub>y app</sub>li<sub>can</sub>t<sub>s,</sub> fi<sub>nanc</sub>i<sub>a</sub>l <sub>ana</sub>l<sub>ys</sub>t<sub>s, awar</sub>d <sub>nom</sub>i<sub>nees, promo</sub>ti<sub>on can</sub>did<sub>a</sub>t<sub>es, new pro</sub>d<sub>uc</sub>t id<sub>eas, an</sub>d <sup>s</sup>p<sup>orts</sup> <sup>draft</sup> p<sup>ros</sup>p<sup>ects</sup>.

Whil<sub>e</sub> <sub>pro</sub>bl<sub>ems</sub> <sub>o</sub>f thi<sub>s</sub> t<sub>ype</sub> <sub>are</sub> <sub>c</sub>l<sub>ear</sub>l<sub>y</sub> i<sub>mpor</sub>t<sub>an</sub>t<sub>,</sub> th<sub>ey</sub> <sub>a</sub>l<sub>so</sub> t<sub>en</sub>d t<sub>o</sub> b<sub>e</sub> difi<sub>cu</sub>lt i<sub>n</sub> t<sub>wo</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>ways.</sub> O<sub>ne</sub> challenge is the analysis and algorithm design. The aim is a sequential evaluation allocation policy that is b<sub>o</sub>th <sub>c</sub>l<sub>ose</sub> t<sub>o</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>an</sub>d <sub>s</sub>t<sub>ruc</sub>t<sub>ure</sub>d <sub>enoug</sub>h t<sub>o</sub> <sub>y</sub>i<sub>e</sub>ld i<sub>ns</sub>i<sub>g</sub>ht <sub>an</sub>d <sub>earn</sub> <sub>prac</sub>titi<sub>oners</sub>’ t<sub>rus</sub>t<sub>.</sub> Th<sub>e</sub> <sub>o</sub>th<sub>er</sub> difi<sub>cu</sub>lt<sub>y</sub> is the modeling, which connects the application to the analysis and sets the stage for how useful that analysis <sub>w</sub>ill b<sub>e.</sub> Fi<sub>rs</sub>t<sub>,</sub> <sub>we</sub> <sub>pos</sub>iti<sub>on</sub> th<sub>e</sub> <sub>pro</sub>bl<sub>em</sub> i<sub>n</sub> th<sub>e</sub> lit<sub>era</sub>t<sub>ure</sub> <sub>an</sub>d <sub>wa</sub>lk th<sub>roug</sub>h th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>li<sub>ng</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> i<sub>nvo</sub>l<sub>ve</sub>d<sub>.</sub> W<sub>e</sub> th<sub>en</sub> di<sub>scuss</sub> h<sub>ow</sub> <sub>we</sub> <sub>arr</sub>i<sub>ve</sub>d <sub>a</sub>t <sub>our</sub> <sub>so</sub>l<sub>u</sub>ti<sub>on</sub> <sub>approac</sub>h<sub>.</sub>

The problem of identifying the best few items by allocating evaluation resources is central to the ranking and selection and best-arm identification literatures, where what we refer to as evaluations are typically called sam<sub>p</sub>les. Unlike the classical work on multi-armed bandits (Robbins 1952), where rewards are collected while learning, here the only concern is with the quality of the terminal selection, the setting known as pure exploration (Bubeck et al. 2009). Much of this literature focuses on identifying the best (top-1) item (Chen et al. 2000, Frazier et al. 2008, Chen and R<sub>y</sub>zhov 2023). We consider the more <sub>g</sub>eneral <sub>p</sub>roblem of to<sub>p</sub>-� identification (Chen et al. 2008, Zhan<sub>g</sub> et al. 2023, Liu et al. 2024).

In this literature, evaluatin<sub>g</sub> (sam<sub>p</sub>lin<sub>g</sub>) an item t<sub>yp</sub>icall<sub>y</sub> means directl<sub>y</sub> observin<sub>g</sub> a nois<sub>y</sub> realization of its utilit<sub>y</sub> (qualit<sub>y</sub>), which we will refer to as untransformed evaluations. Such untransformed in<sub>p</sub>ut is natural in <sub>app</sub>li<sub>ca</sub>ti<sub>ons</sub> <sub>suc</sub>h <sub>as</sub> <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> <sub>an</sub>d h<sub>yperparame</sub>t<sub>er</sub> t<sub>un</sub>i<sub>ng,</sub> <sub>w</sub>h<sub>ere</sub> th<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d <sub>ou</sub>t<sub>pu</sub>t i<sub>s</sub> it<sub>se</sub>lf the object being optimized. In our setting, the utility of each item is latent, since the value of including each it<sub>em</sub> i<sub>n</sub> th<sub>e c</sub>h<sub>osen se</sub>t i<sub>s never eva</sub>l<sub>ua</sub>t<sub>e</sub>d di<sub>rec</sub>tl<sub>y.</sub> I<sub>ns</sub>t<sub>ea</sub>d<sub>, eac</sub>h it<sub>em</sub>’<sub>s qua</sub>lit<sub>y</sub> i<sub>s o</sub>b<sub>serve</sub>d i<sub>n</sub>di<sub>rec</sub>tl<sub>y</sub> th<sub>roug</sub>h <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> th<sub>a</sub>t t<sub>a</sub>k<sub>e</sub> th<sub>e</sub> f<sub>orm</sub> <sub>o</sub>f <sub>ra</sub>ti<sub>ngs</sub> <sub>or</sub> <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>compar</sub>i<sub>sons,</sub> b<sub>o</sub>th <sub>o</sub>f <sub>w</sub>hi<sub>c</sub>h t<sub>rans</sub>f<sub>orm</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>u</sub>tiliti<sub>es</sub> i<sub>n</sub>t<sub>o coarse or</sub>di<sub>na</sub>l <sub>ou</sub>t<sub>comes.</sub>

T<sub>o</sub> hi<sub>g</sub>hli<sub>g</sub>ht <sub>w</sub>h<sub>y</sub> <sub>we</sub> d<sub>o</sub> <sub>no</sub>t <sub>re</sub>l<sub>y</sub> <sub>on</sub> <sub>a</sub>d<sub>ap</sub>ti<sub>ng</sub> <sub>an</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d <sub>a</sub>l<sub>gor</sub>ith<sub>m,</sub> <sub>we</sub> hi<sub>g</sub>hli<sub>g</sub>ht th<sub>ree</sub> <sub>common</sub> <sub>mo</sub>d<sub>e</sub>li<sub>ng</sub> <sub>an</sub>d <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons</sub> i<sub>n</sub> th<sub>e</sub> <sub>ran</sub>ki<sub>ng</sub> <sub>an</sub>d <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> lit<sub>era</sub>t<sub>ure.</sub> W<sub>e</sub> di<sub>scuss</sub> h<sub>ow</sub> <sub>eac</sub>h d<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> <sub>re</sub>l<sub>a</sub>t<sub>es</sub> t<sub>o</sub> <sub>our</sub> <sub>pro</sub>bl<sub>em</sub> <sub>an</sub>d<sub>,</sub> <sub>w</sub>h<sub>en</sub> <sub>re</sub>l<sub>evan</sub>t<sub>,</sub> h<sub>ow</sub> <sub>we</sub> <sub>overcome</sub> th<sub>e</sub> <sub>a</sub>dditi<sub>ona</sub>l <sub>c</sub>h<sub>a</sub>ll<sub>enges</sub> <sub>pose</sub>d b<sub>y</sub> dif<sub>erences</sub> i<sub>n</sub> <sub>our</sub> <sub>se</sub>tti<sub>ng.</sub>

The first modeling decision concerns the terminal objective: how we grade an evaluation allocation policy b<sub>ase</sub>d <sub>on</sub> th<sub>e</sub> � it<sub>ems</sub> it <sub>u</sub>lti<sub>ma</sub>t<sub>e</sub>l<sub>y se</sub>l<sub>ec</sub>t<sub>s.</sub> Th<sub>e</sub> t<sub>wo mos</sub>t <sub>common c</sub>h<sub>o</sub>i<sub>ces are</sub> th<sub>e pro</sub>b<sub>a</sub>bilit<sub>y o</sub>f <sub>correc</sub>t selection (PCS) (Chen et al. 2000, Gl<sub>y</sub>nn and Juneja 2004, Li et al. 2025) and the ex<sub>p</sub>ected o<sub>pp</sub>ortunit<sub>y</sub> cost (EOC) (Chick and Inoue 2001, Chick et al. 2010, Gao and Chen 2015a). PCS is the <sub>p</sub>robabilit<sub>y</sub> that th<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>se</sub>t i<sub>s exac</sub>tl<sub>y</sub> th<sub>e</sub> t<sub>rue</sub> t<sub>op-</sub>�<sub>, so</sub> it t<sub>rea</sub>t<sub>s every</sub> i<sub>ncorrec</sub>t <sub>se</sub>l<sub>ec</sub>ti<sub>on equa</sub>ll<sub>y.</sub> EOC i<sub>s</sub> th<sub>e expec</sub>t<sub>e</sub>d <sub>s</sub>h<sub>or</sub>tf<sub>a</sub>ll i<sub>n</sub> <sub>u</sub>tilit<sub>y</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>se</sub>t <sub>an</sub>d th<sub>e</sub> t<sub>rue</sub> t<sub>op-</sub>� <sub>se</sub>t<sub>,</sub> <sub>ma</sub>ki<sub>ng</sub> bi<sub>gger</sub> <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> <sub>m</sub>i<sub>s</sub>t<sub>a</sub>k<sub>es</sub> <sub>more</sub> <sub>cos</sub>tl<sub>y.</sub> PCS <sub>su</sub>it<sub>s</sub> hi<sub>g</sub>hl<sub>y spec</sub>i<sub>a</sub>li<sub>ze</sub>d <sub>or</sub> hi<sub>g</sub>h<sub>-s</sub>t<sub>a</sub>k<sub>es</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>ons w</sub>h<sub>ere any</sub> d<sub>ev</sub>i<sub>a</sub>ti<sub>on</sub> f<sub>rom</sub> th<sub>e</sub> b<sub>es</sub>t <sub>poss</sub>ibl<sub>e</sub> <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> i<sub>s cons</sub>id<sub>ere</sub>d <sub>a</sub> f<sub>a</sub>il<sub>ure.</sub> F<sub>or more common</sub> t<sub>as</sub>k<sub>s, w</sub>h<sub>ere a ser</sub>i<sub>ous m</sub>i<sub>s</sub>t<sub>a</sub>k<sub>e can</sub> b<sub>e</sub> f<sub>ar cos</sub>tli<sub>er</sub> th<sub>an a</sub> <sub>marg</sub>i<sub>na</sub>l <sub>one,</sub> EOC <sub>can</sub> b<sub>e</sub> th<sub>e</sub> <sub>more</sub> <sub>na</sub>t<sub>ura</sub>l fit<sub>.</sub> Si<sub>nce</sub> EOC i<sub>s</sub> <sub>more</sub> <sub>a</sub>li<sub>gne</sub>d <sub>w</sub>ith th<sub>e</sub> <sub>app</sub>li<sub>ca</sub>ti<sub>ons</sub> <sub>we</sub> <sub>cons</sub>id<sub>er,</sub> <sub>we</sub> <sub>a</sub>d<sub>op</sub>t it i<sub>n</sub> thi<sub>s</sub> <sub>wor</sub>k<sub>.</sub>

A <sub>secon</sub>d d<sub>ec</sub>i<sub>s</sub>i<sub>on concerns</sub> h<sub>ow</sub> th<sub>e a</sub>ll<sub>oca</sub>ti<sub>on pro</sub>bl<sub>em</sub> i<sub>s</sub> f<sub>ormu</sub>l<sub>a</sub>t<sub>e</sub>d<sub>, w</sub>ith t<sub>wo common</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>ons</sub> <sub>a</sub>t <sub>oppos</sub>it<sub>e en</sub>d<sub>s o</sub>f th<sub>e spec</sub>t<sub>rum.</sub> O<sub>ne</sub> i<sub>s s</sub>t<sub>a</sub>ti<sub>c a</sub>ll<sub>oca</sub>ti<sub>on, w</sub>h<sub>ere</sub> th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on var</sub>i<sub>a</sub>bl<sub>e</sub> i<sub>s</sub> th<sub>e</sub> t<sub>o</sub>t<sub>a</sub>l <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> <sub>ass</sub>i<sub>gne</sub>d t<sub>o</sub> <sub>eac</sub>h it<sub>em.</sub> Th<sub>e</sub> <sub>o</sub>th<sub>er</sub> i<sub>s</sub> d<sub>ynam</sub>i<sub>c</sub> l<sub>earn</sub>i<sub>ng,</sub> <sub>w</sub>h<sub>ere</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> <sub>are</sub> <sub>c</sub>h<sub>osen</sub> <sub>one</sub> <sub>a</sub>t <sub>a</sub> ti<sub>me</sub> <sub>an</sub>d <sub>eac</sub>h <sub>ou</sub>t<sub>come</sub> i<sub>s</sub> <sub>o</sub>b<sub>serve</sub>d b<sub>e</sub>f<sub>ore</sub> th<sub>e</sub> <sub>nex</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> i<sub>s</sub> <sub>c</sub>h<sub>osen.</sub> I<sub>n</sub> th<sub>e</sub> <sub>s</sub>t<sub>a</sub>ti<sub>c</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>on,</sub> <sub>wr</sub>iti<sub>ng</sub>

$N = \left( N _ { 1 } , \ldots , N _ { I } \right)$ f<sub>or</sub> th<sub>e</sub> <sub>vec</sub>t<sub>or</sub> <sub>o</sub>f <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> <sub>ass</sub>i<sub>gne</sub>d t<sub>o</sub> it<sub>ems</sub> $i = 1 , \ldots , I$ <sub>an</sub>d � f<sub>or</sub> th<sub>e</sub> t<sub>o</sub>t<sub>a</sub>l <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>,</sub> th<sub>e</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> <sub>pro</sub>bl<sub>em</sub> i<sub>s</sub>

$$
\operatorname* { m i n } _ { N } \mathrm { E O C } ( N ) \qquad \mathrm { s . t . } \qquad \sum _ { i = 1 } ^ { I } N _ { i } \leq R .
$$

This allocation problem faces two main dificulties. The objective EOC(�), and likewise PCS(�), generall<sub>y</sub> d<sub>oes no</sub>t h<sub>ave a c</sub>l<sub>ose</sub>d f<sub>orm</sub> th<sub>a</sub>t i<sub>s amena</sub>bl<sub>e</sub> t<sub>o ana</sub>l<sub>ys</sub>i<sub>s.</sub> Additi<sub>ona</sub>ll<sub>y,</sub> th<sub>e pro</sub>bl<sub>em</sub> i<sub>s an</sub> i<sub>n</sub>t<sub>eger program, w</sub>hi<sub>c</sub>h t<sub>yp</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>ma</sub>k<sub>es</sub> it i<sub>n</sub>t<sub>rac</sub>t<sub>a</sub>bl<sub>e</sub> t<sub>o</sub> <sub>cons</sub>id<sub>er</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>so</sub>l<sub>u</sub>ti<sub>on</sub> <sub>space</sub> <sub>a</sub>t <sub>sca</sub>l<sub>e.</sub> I<sub>n</sub> th<sub>e</sub> f<sub>u</sub>ll<sub>y</sub> d<sub>ynam</sub>i<sub>c</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>on,</sub> <sub>one</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> i<sub>s c</sub>h<sub>osen a</sub>t <sub>a</sub> ti<sub>me, avo</sub>idi<sub>ng</sub> th<sub>e com</sub>bi<sub>na</sub>t<sub>or</sub>i<sub>a</sub>l <sub>comp</sub>l<sub>ex</sub>it<sub>y o</sub>f th<sub>e s</sub>t<sub>a</sub>ti<sub>c pro</sub>bl<sub>em.</sub> At <sub>eac</sub>h d<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> <sub>po</sub>i<sub>n</sub>t<sub>,</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e cons</sub>i<sub>s</sub>t<sub>s o</sub>f th<sub>e curren</sub>t <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on over</sub> th<sub>e</sub> it<sub>ems</sub>’ <sub>u</sub>tiliti<sub>es an</sub>d th<sub>e rema</sub>i<sub>n</sub>i<sub>ng</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>ac</sub>ti<sub>on</sub> i<sub>s</sub> th<sub>e</sub> <sub>nex</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on.</sub> Th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> t<sub>rans</sub>iti<sub>on</sub> <sub>up</sub>d<sub>a</sub>t<sub>es</sub> th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> b<sub>ase</sub>d <sub>on</sub> th<sub>e</sub> <sub>ou</sub>t<sub>come</sub> <sub>an</sub>d d<sub>e</sub>d<sub>uc</sub>t<sub>s</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub>’<sub>s</sub> <sub>cos</sub>t f<sub>rom</sub> th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub> Thi<sub>s</sub> <sub>pro</sub>bl<sub>em</sub> i<sub>s</sub> f<sub>ormu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>as</sub> <sub>a</sub> d<sub>ynam</sub>i<sub>c</sub> <sub>program</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> i<sub>n</sub>t<sub>rac</sub>t<sub>a</sub>bl<sub>e</sub> b<sub>ecause</sub> th<sub>e com</sub>bi<sub>na</sub>ti<sub>on o</sub>f <sub>poss</sub>ibl<sub>e eva</sub>l<sub>ua</sub>ti<sub>ons, ou</sub>t<sub>comes,</sub> t<sub>rans</sub>iti<sub>ons, an</sub>d f<sub>u</sub>t<sub>ure s</sub>t<sub>a</sub>t<sub>es</sub> <sub>g</sub>rows ra<sub>p</sub>idl<sub>y</sub> with the horizon (Frazier et al. 2008, Hon<sub>g</sub> et al. 2021, Zhan<sub>g</sub> et al. 2023).

O<sub>ur</sub> b<sub>ase</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub> i<sub>s</sub> f<sub>u</sub>ll<sub>y</sub> d<sub>ynam</sub>i<sub>c</sub> <sub>w</sub>ith <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> <sub>c</sub>h<sub>osen</sub> <sub>sequen</sub>ti<sub>a</sub>ll<sub>y.</sub> H<sub>owever,</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>ng</sub> <sub>one</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on a</sub>t <sub>a</sub> ti<sub>me can</sub> h<sub>ave</sub> l<sub>og</sub>i<sub>s</sub>ti<sub>ca</sub>l <sub>o</sub>b<sub>s</sub>t<sub>ac</sub>l<sub>es.</sub> Th<sub>ere may</sub> b<sub>e cons</sub>id<sub>era</sub>bl<sub>e over</sub>h<sub>ea</sub>d i<sub>n ass</sub>i<sub>gn</sub>i<sub>ng an</sub>d conducting evaluations, which can favor giving an evaluator a batch of tasks at once. We therefore also <sub>cons</sub>id<sub>er a</sub> h<sub>y</sub>b<sub>r</sub>id <sub>approac</sub>h i<sub>n</sub> $\ S 6 ,$ <sub>w</sub>h<sub>ere a</sub> b<sub>a</sub>t<sub>c</sub>h <sub>o</sub>f <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> i<sub>s c</sub>h<sub>osen</sub> i<sub>n eac</sub>h <sub>roun</sub>d<sub>.</sub> Thi<sub>s s</sub>it<sub>s</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> t<sub>wo</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>ons</sub> <sub>an</sub>d i<sub>n</sub>h<sub>er</sub>it<sub>s</sub> b<sub>o</sub>th difi<sub>cu</sub>lti<sub>es.</sub> Withi<sub>n</sub> <sub>a</sub> <sub>roun</sub>d<sub>,</sub> th<sub>e</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> i<sub>s</sub> <sub>an</sub> i<sub>n</sub>t<sub>eger</sub> <sub>program,</sub> <sub>eac</sub>h li<sub>n</sub>k<sub>e</sub>d th<sub>roug</sub>h th<sub>e</sub> b<sub>e</sub>t<sub>ween-roun</sub>d d<sub>ynam</sub>i<sub>cs.</sub>

Th<sub>e comp</sub>l<sub>ex</sub>it<sub>y o</sub>f th<sub>e pro</sub>bl<sub>em un</sub>d<sub>er</sub> b<sub>o</sub>th f<sub>ormu</sub>l<sub>a</sub>ti<sub>ons</sub> h<sub>as g</sub>i<sub>ven r</sub>i<sub>se</sub> t<sub>o a var</sub>i<sub>e</sub>t<sub>y o</sub>f <sub>so</sub>l<sub>u</sub>ti<sub>on approac</sub>h<sub>es.</sub> B<sub>e</sub>f<sub>ore</sub> t<sub>urn</sub>i<sub>ng</sub> t<sub>o our own, we</sub> di<sub>scuss</sub> th<sub>e</sub> th<sub>ree mos</sub>t <sub>re</sub>l<sub>evan</sub>t <sub>we</sub>ll<sub>-s</sub>t<sub>u</sub>di<sub>e</sub>d f<sub>am</sub>ili<sub>es, a</sub>ll d<sub>eve</sub>l<sub>ope</sub>d i<sub>n</sub> th<sub>e</sub> <sub>un</sub>t<sub>rans</sub>f<sub>orme</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on se</sub>tti<sub>ng.</sub> Th<sub>e</sub> fi<sub>rs</sub>t i<sub>s a</sub> f<sub>am</sub>il<sub>y o</sub>f <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> k<sub>nown as</sub> O<sub>p</sub>ti<sub>ma</sub>l C<sub>ompu</sub>ti<sub>ng</sub> B<sub>u</sub>d<sub>ge</sub>t Allocation (OCBA), ori<sub>g</sub>inatin<sub>g</sub> with Chen et al. (1997, 2000), Gl<sub>y</sub>nn and Juneja (2004). At a hi<sub>g</sub>h level, the al<sub>g</sub>orithm loo<sub>p</sub>s throu<sub>g</sub>h three ste<sub>p</sub>s: (i) take the items’ current mean estimates as if the<sub>y</sub> were the <sub>g</sub>round truth utilities, (ii) choose the next evaluation(s) b<sub>y</sub> determinin<sub>g</sub> the static allocation that would maximize PCS (or minimize EOC), and (iii) u<sub>p</sub>date the estimates of the means based on the outcome. However, ste<sub>p</sub> (ii) is intractable due to the com<sub>p</sub>lexities of the static allocation <sub>p</sub>roblem discussed above. To overcome thi<sub>s,</sub> it i<sub>s</sub> <sub>common</sub> t<sub>o</sub> <sub>cons</sub>id<sub>er</sub> th<sub>e</sub> <sub>asymp</sub>t<sub>o</sub>ti<sub>c</sub> <sub>reg</sub>i<sub>me</sub> <sub>w</sub>h<sub>ere</sub> $R \to \infty$ <sub>an</sub>d fi<sub>n</sub>d th<sub>e</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> <sub>propor</sub>ti<sub>ons</sub> th<sub>a</sub>t (a<sub>pp</sub>roximatel<sub>y</sub>) o<sub>p</sub>timize the resultin<sub>g p</sub>roblem. Eventuall<sub>y</sub>, the estimated means will conver<sub>g</sub>e to the true means, and thus the evaluation <sub>p</sub>ro<sub>p</sub>ortions will conver<sub>g</sub>e to the (a<sub>pp</sub>roximatel<sub>y</sub>) o<sub>p</sub>timal <sub>p</sub>ro<sub>p</sub>ortions. The majority of the OCBA literature studies the top-1 setting with PCS, with extensions to top-1 EOC (He et al. 2007, Gao et al. 2017), as well as to to<sub>p</sub>-� PCS (Chen et al. 2008, Gao and Chen 2015b, Zhan<sub>g</sub> et al. 2016) and to<sub>p</sub>-� EOC (Gao and Chen 2015a). Such a<sub>pp</sub>roaches tend to <sub>p</sub>erform better in lar<sub>g</sub>e-bud<sub>g</sub>et settin<sub>g</sub>s. In f<sub>ac</sub>t<sub>,</sub> f<sub>or sma</sub>ll b<sub>u</sub>d<sub>ge</sub>t<sub>s,</sub> it h<sub>as</sub> b<sub>een</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub>d th<sub>a</sub>t <sub>suc</sub>h <sub>a</sub>ll<sub>oca</sub>ti<sub>on proce</sub>d<sub>ures may even per</sub>f<sub>orm worse</sub> than equal allocation (Pen<sub>g</sub> et al. 2018, Liu et al. 2024).

A <sub>secon</sub>d f<sub>am</sub>il<sub>y</sub> <sub>o</sub>f <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> <sub>cons</sub>id<sub>ers</sub> th<sub>e</sub> d<sub>ynam</sub>i<sub>c</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>res</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d t<sub>o</sub> <sub>one-s</sub>t<sub>ep</sub> l<sub>oo</sub>k<sub>-a</sub>h<sub>ea</sub>d <sub>po</sub>li<sub>c</sub>i<sub>es,</sub> see Powell and R<sub>y</sub>zhov (2012) for an overview. At each round, the decision-maker chooses the evaluation that (a<sub>pp</sub>roximatel<sub>y</sub>) maximizes the ex<sub>p</sub>ected value of information (EVI), the im<sub>p</sub>rovement in the qualit<sub>y</sub> <sub>o</sub>f th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> th<sub>a</sub>t <sub>resu</sub>lt<sub>s</sub> f<sub>rom one a</sub>dditi<sub>ona</sub>l <sub>eva</sub>l<sub>ua</sub>ti<sub>on.</sub> F<sub>or</sub> t<sub>op-</sub>1 id<sub>en</sub>tifi<sub>ca</sub>ti<sub>on,</sub> Chi<sub>c</sub>k <sub>an</sub>d Inoue (2001), Chick et al. (2010) derive <sub>p</sub>olicies that a<sub>pp</sub>roximatel<sub>y</sub> maximize EVI for both the PCS and EOC objectives. Recentl<sub>y</sub>, Zhan<sub>g</sub> et al. (2023) develo<sub>p</sub>ed a to<sub>p</sub>-� identification <sub>p</sub>olic<sub>y</sub> that maximizes an <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> EVI f<sub>or</sub> PCS<sub>.</sub> A<sub>ssum</sub>i<sub>ng</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> f<sub>o</sub>ll<sub>ow</sub> G<sub>auss</sub>i<sub>an</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> <sub>w</sub>ith k<sub>nown</sub> <sub>var</sub>i<sub>ances,</sub> Frazier et al. (2008) derive the exact one-ste<sub>p</sub> look-ahead <sub>p</sub>olic<sub>y</sub> that maximizes the EVI for EOC, known as the knowled<sub>g</sub>e <sub>g</sub>radient (KG) <sub>p</sub>olic<sub>y</sub>. In our settin<sub>g</sub>, however, KG’s m<sub>y</sub>o<sub>p</sub>ia can be <sub>p</sub>articularl<sub>y</sub> detrimental with binar<sub>y</sub> evaluations (Groves and Branke 2019<sub>,</sub> Li et al. 2024). We return to this <sub>p</sub>oint in §2.3.

A third related famil<sub>y</sub> of al<sub>g</sub>orithms is based on the ex<sub>p</sub>ected im<sub>p</sub>rovement (EI) a<sub>pp</sub>roach (Jones et al. 1998), which ori<sub>g</sub>inated in the Ba<sub>y</sub>esian O<sub>p</sub>timization literature. Various works have studied its a<sub>pp</sub>lication to to<sub>p</sub>-1 identification (R<sub>y</sub>zhov 2016, Qin et al. 2017, Chen and R<sub>y</sub>zhov 2019). In its sim<sub>p</sub>lest form, EI scores <sub>eac</sub>h it<sub>em</sub> b h<sub>ow muc</sub>h it<sub>s u</sub>tilit <sub>,</sub> i<sub>n ex ec</sub>t<sub>a</sub>ti<sub>on, excee</sub>d<sub>s</sub> th<sub>a</sub>t <sub>o</sub>f th<sub>e curren</sub>t b<sub>es</sub>t <sub>can</sub>did<sub>a</sub>t<sub>e an</sub>d <sub>eva</sub>l<sub>ua</sub>t<sub>es</sub> th<sub>e</sub> it<sub>em</sub> <sub>w</sub>ith th<sub>e</sub> hi<sub>g</sub>h<sub>es</sub>t <sub>score.</sub> EI’<sub>s</sub> <sub>s</sub>i<sub>mp</sub>li<sub>c</sub>it<sub>y</sub> i<sub>s</sub> <sub>appea</sub>li<sub>ng,</sub> <sub>an</sub>d <sub>a</sub>lth<sub>oug</sub>h it d<sub>oes</sub> <sub>no</sub>t <sub>exp</sub>li<sub>c</sub>itl<sub>y</sub> l<sub>oo</sub>k <sub>a</sub>h<sub>ea</sub>d<sub>,</sub> it <sub>s</sub>till <sub>re</sub>fl<sub>ec</sub>t<sub>s</sub> f<sub>u</sub>t<sub>ure</sub> d<sub>ynam</sub>i<sub>cs: eva</sub>l<sub>ua</sub>ti<sub>ng</sub> it<sub>ems w</sub>ith hi<sub>g</sub>h <sub>po</sub>t<sub>en</sub>ti<sub>a</sub>l t<sub>o</sub> i<sub>mprove on</sub> th<sub>e curren</sub>t b<sub>es</sub>t <sub>can</sub> h<sub>e</sub>l<sub>p</sub> i<sub>n</sub>f<sub>orm</sub> <sub>w</sub>hi<sub>c</sub>h <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> t<sub>o con</sub>d<sub>uc</sub>t i<sub>n</sub> f<sub>u</sub>t<sub>ure roun</sub>d<sub>s.</sub> H<sub>owever,</sub> EI d<sub>oes no</sub>t <sub>na</sub>t<sub>ura</sub>ll<sub>y ex</sub>t<sub>en</sub>d t<sub>o re</sub>fl<sub>ec</sub>t <sub>a</sub>lt<sub>erna</sub>ti<sub>ve</sub> evaluation forms with diferent information contents and costs. Furthermore, it chooses which single item to evaluate next, whereas our evaluation alternatives include choosing a pair to compare.

Fi<sub>na</sub>ll<sub>y,</sub> th<sub>e</sub> thi<sub>r</sub>d <sub>mo</sub>d<sub>e</sub>li<sub>ng</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on concerns</sub> th<sub>e</sub> f<sub>orm o</sub>f th<sub>e eva</sub>l<sub>ua</sub>ti<sub>ons</sub> th<sub>emse</sub>l<sub>ves.</sub> I<sub>n psyc</sub>h<sub>ome</sub>t<sub>r</sub>i<sub>cs,</sub> th<sub>e</sub> fi<sub>e</sub>ld <sub>o</sub>f <sub>measuremen</sub>t <sub>o</sub>f <sub>psyc</sub>h<sub>o</sub>l<sub>og</sub>i<sub>ca</sub>l d<sub>a</sub>t<sub>a,</sub> th<sub>e</sub> t<sub>wo mos</sub>t <sub>common</sub> f<sub>orms o</sub>f i<sub>npu</sub>t f<sub>or</sub> h<sub>uman</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>are</sub> <sub>ra</sub>ti<sub>ngs, w</sub>h<sub>ere an eva</sub>l<sub>ua</sub>ti<sub>on ass</sub>i<sub>gns a ca</sub>t<sub>egor</sub>i<sub>ca</sub>l <sub>score</sub> t<sub>o a par</sub>ti<sub>cu</sub>l<sub>ar</sub> it<sub>em, an</sub>d <sub>ran</sub>ki<sub>ngs, w</sub>h<sub>ere a se</sub>t <sub>o</sub>f items is ordered relative to one another (Thurstone 1927, Likert 1932, Tor<sub>g</sub>erson 1958). We consider each in it<sub>s</sub> <sub>pars</sub>i<sub>mon</sub>i<sub>ous</sub> f<sub>orm—</sub>bi<sub>nary</sub> th<sub>um</sub>b<sub>s-up</sub>/th<sub>um</sub>b<sub>s-</sub>d<sub>own</sub> <sub>ra</sub>ti<sub>ngs</sub> <sub>an</sub>d <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>compar</sub>i<sub>sons—</sub>b<sub>o</sub>th <sub>among</sub> th<sub>e mos</sub>t <sub>common</sub> i<sub>npu</sub>t <sub>c</sub>h<sub>o</sub>i<sub>ces.</sub> R<sub>es</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>ng</sub> t<sub>o</sub> th<sub>ese s</sub>i<sub>mp</sub>l<sub>e vers</sub>i<sub>ons o</sub>f <sub>or</sub>di<sub>na</sub>l f<sub>ee</sub>db<sub>ac</sub>k k<sub>eeps eva</sub>l<sub>ua</sub>t<sub>ors</sub>’ co<sub>g</sub>nitive load low (Revilla et al. 2014, Kutscher and Eid 2024). The two in<sub>p</sub>ut forms have distinct <sub>p</sub>ros and cons in our settin<sub>g</sub>. Ratin<sub>g</sub>s can be chea<sub>p</sub>er and easier to <sub>p</sub>rovide than <sub>p</sub>airwise com<sub>p</sub>arisons (White et al. 2024), and a few of them can be an eficient wa<sub>y</sub> to establish the ball<sub>p</sub>ark utilit<sub>y</sub> of an item. However, <sub>ra</sub>ti<sub>ngs are no</sub>t <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>a</sub>t di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>hi<sub>ng</sub> th<sub>e re</sub>l<sub>a</sub>ti<sub>ve or</sub>d<sub>er o</sub>f <sub>c</sub>l<sub>ose</sub> it<sub>ems, w</sub>hi<sub>c</sub>h i<sub>s essen</sub>ti<sub>a</sub>l f<sub>or</sub> dif<sub>eren</sub>ti<sub>a</sub>ti<sub>ng</sub> between to<sub>p</sub>-� candidates. Additionall<sub>y</sub>, a ratin<sub>g</sub> (sa<sub>y</sub>, a thumbs-u<sub>p</sub>) can carr<sub>y</sub> a diferent meanin<sub>g</sub> to diferent <sub>eva</sub>l<sub>ua</sub>t<sub>ors, resu</sub>lti<sub>ng</sub> i<sub>n po</sub>t<sub>en</sub>ti<sub>a</sub>l <sub>am</sub>bi<sub>gu</sub>it<sub>y w</sub>h<sub>en aggrega</sub>ti<sub>ng an</sub>d i<sub>n</sub>t<sub>erpre</sub>ti<sub>ng</sub> th<sub>em.</sub> P<sub>a</sub>i<sub>rw</sub>i<sub>se compar</sub>i<sub>sons,</sub> i<sub>n</sub> contrast, only contain relative information about item utilities, and while their interpretation is unambiguous, th<sub>e</sub>i<sub>r use</sub>f<sub>u</sub>l<sub>ness</sub> d<sub>epen</sub>d<sub>s</sub> h<sub>eav</sub>il<sub>y on w</sub>hi<sub>c</sub>h it<sub>ems are pa</sub>i<sub>re</sub>d<sub>.</sub> Th<sub>e num</sub>b<sub>er o</sub>f <sub>poss</sub>ibl<sub>e pa</sub>i<sub>r</sub>i<sub>ngs</sub> i<sub>s qua</sub>d<sub>ra</sub>ti<sub>c</sub> i<sub>n</sub> th<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f it<sub>ems,</sub> <sub>so</sub> <sub>s</sub>t<sub>ar</sub>ti<sub>ng</sub> f<sub>rom</sub> <sub>an</sub> <sub>un</sub>i<sub>n</sub>f<sub>orme</sub>d <sub>pr</sub>i<sub>or</sub> <sub>wou</sub>ld <sub>resu</sub>lt i<sub>n</sub> <sub>many</sub> <sub>un</sub>h<sub>e</sub>l<sub>p</sub>f<sub>u</sub>l <sub>pa</sub>i<sub>r</sub>i<sub>ngs</sub> <sub>ear</sub>l<sub>y</sub> <sub>on.</sub> F<sub>or examp</sub>l<sub>e, we m</sub>i<sub>g</sub>ht <sub>compare</sub> th<sub>e</sub> it<sub>ems</sub> th<sub>a</sub>t t<sub>urn ou</sub>t t<sub>o</sub> b<sub>e</sub> th<sub>e</sub> t<sub>wo wors</sub>t<sub>.</sub> E<sub>ven</sub> if th<sub>a</sub>t <sub>compar</sub>i<sub>son</sub> <sub>prov</sub>id<sub>es a s</sub>t<sub>rong s</sub>i<sub>gna</sub>l <sub>a</sub>b<sub>ou</sub>t <sub>w</sub>hi<sub>c</sub>h <sub>o</sub>f th<sub>e</sub> t<sub>wo</sub> i<sub>s</sub> b<sub>e</sub>tt<sub>er,</sub> it d<sub>oes</sub> littl<sub>e</sub> t<sub>o</sub> i<sub>n</sub>f<sub>orm us a</sub>b<sub>ou</sub>t <sub>w</sub>hi<sub>c</sub>h it<sub>ems are</sub> i<sub>n</sub> th<sub>e</sub> t<sub>op-</sub>�<sub>.</sub> H<sub>owever,</sub> <sub>once</sub> <sub>we</sub> d<sub>o</sub> h<sub>ave</sub> <sub>a</sub> <sub>c</sub>l<sub>ear</sub> <sub>sense</sub> <sub>o</sub>f <sub>w</sub>hi<sub>c</sub>h it<sub>ems</sub> <sub>are</sub> t<sub>op-</sub>� <sub>con</sub>t<sub>en</sub>d<sub>ers,</sub> <sub>compar</sub>i<sub>sons</sub> <sub>can</sub> b<sub>e</sub> far more efective than ratin<sub>g</sub>s at <sub>p</sub>innin<sub>g</sub> down their relative orderin<sub>g</sub> (Li et al. 2018). It follows that the two evaluation forms can be es<sub>p</sub>eciall<sub>y</sub> useful when combined (Fa<sub>g</sub>eot et al. 2026). Intuitivel<sub>y</sub>, ratin<sub>g</sub>s can be <sub>use</sub>d <sub>more</sub> i<sub>n</sub> <sub>ear</sub>l<sub>y</sub> <sub>roun</sub>d<sub>s</sub> t<sub>o</sub> <sub>ge</sub>t <sub>a</sub> <sub>coarse</sub> <sub>p</sub>i<sub>c</sub>t<sub>ure</sub> <sub>o</sub>f <sub>w</sub>hi<sub>c</sub>h it<sub>ems</sub> <sub>are</sub> “<sub>goo</sub>d” <sub>an</sub>d <sub>w</sub>hi<sub>c</sub>h <sub>are</sub> <sub>no</sub>t<sub>,</sub> <sub>an</sub>d th<sub>en</sub> <sub>as</sub> th<sub>a</sub>t <sub>p</sub>i<sub>c</sub>t<sub>ure</sub> f<sub>orms,</sub> <sub>compar</sub>i<sub>sons</sub> <sub>can</sub> <sub>gra</sub>d<sub>ua</sub>ll<sub>y</sub> t<sub>a</sub>k<sub>e</sub> <sub>over,</sub> <sub>re</sub>fi<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>or</sub>d<sub>er</sub>i<sub>ng</sub> <sub>o</sub>f th<sub>e</sub> <sub>goo</sub>d it<sub>ems</sub> <sub>an</sub>d <sub>narrow</sub>i<sub>ng</sub> i<sub>n on</sub> th<sub>e u</sub>lti<sub>ma</sub>t<sub>e</sub> t<sub>op-</sub>� <sub>se</sub>l<sub>ec</sub>ti<sub>on.</sub> O<sub>ur mo</sub>d<sub>e</sub>l <sub>a</sub>ll<sub>ows</sub> th<sub>e</sub> t<sub>wo eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>ypes</sub> t<sub>o</sub> dif<sub>er</sub> i<sub>n accuracy</sub> and cost (efort), followin<sub>g</sub> the em<sub>p</sub>irical findin<sub>g</sub>s of Perez-Ortiz et al. (2020) and White et al. (2024).´

Th<sub>e</sub> t<sub>wo eva</sub>l<sub>ua</sub>ti<sub>on</sub> f<sub>orms can a</sub>l<sub>so</sub> i<sub>n</sub>t<sub>erac</sub>t i<sub>n non-</sub>i<sub>n</sub>t<sub>u</sub>iti<sub>ve ways.</sub> C<sub>ons</sub>id<sub>er a s</sub>i<sub>mp</sub>l<sub>e examp</sub>l<sub>e w</sub>ith f<sub>our</sub> it<sub>ems,</sub> �<sub>,</sub> �<sub>,</sub> �<sub>, an</sub>d �<sub>, w</sub>h<sub>ose u</sub>tiliti<sub>es are</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>ex-an</sub>t<sub>e.</sub> With <sub>ra</sub>ti<sub>ngs a</sub>l<sub>one, a ra</sub>ti<sub>ng o</sub>f � i<sub>n</sub>f<sub>orms</sub> it<sub>s u</sub>tilit<sub>y re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e o</sub>th<sub>er</sub> it<sub>ems</sub> b<sub>u</sub>t l<sub>eaves our</sub> b<sub>e</sub>li<sub>e</sub>f<sub>s a</sub>b<sub>ou</sub>t th<sub>e</sub>i<sub>r u</sub>tiliti<sub>es unc</sub>h<sub>ange</sub>d<sub>.</sub> N<sub>ow suppose we</sub> <sub>o</sub>b<sub>serve</sub> th<sub>e compar</sub>i<sub>sons</sub> $A \succ B$ <sub>an</sub>d $C \succ D .$ <sub>,</sub> f<sub>o</sub>ll<sub>owe</sub>d b<sub>y</sub> $B \succ C$ <sub>.</sub> I<sub>n</sub>t<sub>u</sub>iti<sub>ve</sub>l<sub>y,</sub> th<sub>e</sub> thi<sub>r</sub>d <sub>compar</sub>i<sub>son</sub> <sub>sugges</sub>t<sub>s</sub> th<sub>a</sub>t � i<sub>s</sub> <sub>pre</sub>f<sub>erre</sub>d t<sub>o</sub> � <sub>even</sub> th<sub>oug</sub>h <sub>ne</sub>ith<sub>er</sub> <sub>was</sub> i<sub>nvo</sub>l<sub>ve</sub>d i<sub>n</sub> th<sub>e</sub> <sub>compar</sub>i<sub>son.</sub> Th<sub>e</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> b<sub>y</sub> <sub>w</sub>hi<sub>c</sub>h our model reflects this is that comparisons create dependency in the joint posterior over the utilities: the thi<sub>r</sub>d <sub>compar</sub>i<sub>son ra</sub>i<sub>ses</sub> �’<sub>s es</sub>ti<sub>ma</sub>t<sub>e an</sub>d th<sub>ere</sub>b<sub>y a</sub>l<sub>so</sub> lift<sub>s</sub> �<sub>, w</sub>hil<sub>e</sub> l<sub>ower</sub>i<sub>ng</sub> �’<sub>s es</sub>ti<sub>ma</sub>t<sub>e an</sub>d th<sub>ere</sub>b<sub>y a</sub>l<sub>so</sub> l<sub>ower</sub>i<sub>ng</sub> �<sub>.</sub> Thi<sub>s</sub> d<sub>epen</sub>d<sub>ency a</sub>l<sub>so c</sub>h<sub>anges w</sub>h<sub>a</sub>t <sub>we</sub> l<sub>earn</sub> f<sub>rom a ra</sub>ti<sub>ng.</sub> A <sub>su</sub>b<sub>sequen</sub>t <sub>ra</sub>ti<sub>ng o</sub>f � <sub>can now</sub> i<sub>n</sub>fl<sub>uence</sub> <sub>our</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> b<sub>e</sub>li<sub>e</sub>f<sub>s</sub> <sub>a</sub>b<sub>ou</sub>t �<sub>,</sub> �<sub>,</sub> <sub>an</sub>d �<sub>.</sub> I<sub>n</sub> f<sub>ac</sub>t<sub>,</sub> <sub>w</sub>ith <sub>a</sub> <sub>more</sub> <sub>comp</sub>l<sub>ex</sub> <sub>compar</sub>i<sub>son</sub> hi<sub>s</sub>t<sub>ory,</sub> <sub>a</sub> <sub>ra</sub>ti<sub>ng</sub> <sub>o</sub>f � <sub>can even reverse our es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>re</sub>l<sub>a</sub>ti<sub>ve ran</sub>ki<sub>ng o</sub>f t<sub>wo non-eva</sub>l<sub>ua</sub>t<sub>e</sub>d it<sub>ems, suc</sub>h <sub>as</sub> � <sub>an</sub>d �<sub>.</sub> Th<sub>us,</sub> th<sub>e va</sub>l<sub>ue o</sub>f <sub>an eva</sub>l<sub>ua</sub>ti<sub>on</sub> f<sub>or</sub> i<sub>n</sub>f<sub>orm</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>op-</sub>� <sub>se</sub>t <sub>may no</sub>t b<sub>e apparen</sub>t f<sub>rom w</sub>h<sub>a</sub>t it t<sub>e</sub>ll<sub>s us a</sub>b<sub>ou</sub>t th<sub>e</sub> evaluated items alone. Including such dependency in the joint posterior is an opportunity, as it leads to better di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>ti<sub>on</sub> b<sub>e</sub>t<sub>ween</sub> it<sub>ems</sub> <sub>an</sub>d h<sub>ence</sub> <sub>a</sub> <sub>more</sub> <sub>accura</sub>t<sub>e</sub> t<sub>op-</sub>� <sub>c</sub>h<sub>o</sub>i<sub>ce.</sub> B<sub>u</sub>t thi<sub>s</sub> <sub>a</sub>dd<sub>e</sub>d <sub>accuracy</sub> d<sub>oes</sub> <sub>no</sub>t <sub>come</sub> f<sub>or</sub> f<sub>ree:</sub> <sub>re</sub>fl<sub>ec</sub>ti<sub>ng</sub> th<sub>e</sub> d<sub>epen</sub>d<sub>ency</sub> i<sub>n</sub> th<sub>e</sub> d<sub>ynam</sub>i<sub>c</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>o</sub>f <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> <sub>a</sub>dd<sub>s</sub> <sub>comp</sub>l<sub>ex</sub>it<sub>y.</sub>

W<sub>or</sub>k <sub>re</sub>l<sub>a</sub>t<sub>e</sub>d t<sub>o</sub> t<sub>op-</sub>� <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> t<sub>as</sub>k<sub>s</sub> i<sub>n</sub> <sub>w</sub>hi<sub>c</sub>h <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> <sub>are,</sub> lik<sub>e</sub> <sub>ours,</sub> bi<sub>nary</sub> <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>compar</sub>i<sub>sons</sub> (and <sub>p</sub>ossibl<sub>y</sub> also ratin<sub>g</sub>s) is scarce in the literature, es<sub>p</sub>eciall<sub>y</sub> with small bud<sub>g</sub>ets. In this settin<sub>g</sub>, one common approach to model evaluations is to assign each possible evaluation its own parameter. For pairwise <sub>compar</sub>i<sub>sons,</sub> thi<sub>s</sub> <sub>means</sub> <sub>a</sub> <sub>separa</sub>t<sub>e</sub> <sub>parame</sub>t<sub>er</sub> <sub>spec</sub>ifi<sub>ca</sub>ti<sub>on</sub> f<sub>or</sub> <sub>eac</sub>h <sub>pa</sub>i<sub>r</sub> <sub>o</sub>f it<sub>ems,</sub> <sub>represen</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> th<sub>a</sub>t <sub>one</sub> it<sub>em</sub> i<sub>s pre</sub>f<sub>erre</sub>d t<sub>o</sub> th<sub>e o</sub>th<sub>er.</sub> Alth<sub>oug</sub>h th<sub>e ou</sub>t<sub>comes are</sub> bi<sub>nary,</sub> thi<sub>s mo</sub>d<sub>e</sub>l i<sub>s s</sub>till i<sub>n</sub> th<sub>e un</sub>t<sub>rans</sub>f<sub>orme</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on se</sub>tti<sub>ng:</sub> th<sub>e ou</sub>t<sub>comes rema</sub>i<sub>n a</sub> di<sub>rec</sub>t <sub>no</sub>i<sub>sy rea</sub>li<sub>za</sub>ti<sub>on o</sub>f th<sub>e un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng parame</sub>t<sub>er.</sub> A<sub>n a</sub>lt<sub>erna</sub>ti<sub>ve</sub> approach is the latent-utility model of our setting, where each item has a single value, its utility, that generates the com<sub>p</sub>arisons (and ratin<sub>g</sub>s).

In the rankin<sub>g</sub> and selection literature, Priekule and Meisel (2017), Groves and Branke (2019), Li et al. (2024) stud<sub>y</sub> to<sub>p</sub>-1 and to<sub>p</sub>-� selection with binar<sub>y</sub> <sub>p</sub>airwise com<sub>p</sub>arisons under the first evaluation model. Th<sub>e</sub>i<sub>r</sub> <sub>me</sub>th<sub>o</sub>d<sub>o</sub>l<sub>ogy</sub> d<sub>oes</sub> <sub>no</sub>t t<sub>rans</sub>f<sub>er</sub> t<sub>o</sub> <sub>our</sub> <sub>pro</sub>bl<sub>em,</sub> <sub>as</sub> th<sub>e</sub>i<sub>r</sub> <sub>more</sub> fl<sub>ex</sub>ibl<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l <sub>requ</sub>i<sub>res</sub> l<sub>earn</sub>i<sub>ng</sub> th<sub>e</sub> <sub>p</sub>arameters o<sup>f</sup> $I ( I { - } 1 ) / 2$ it<sub>em</sub> <sub>pa</sub>i<sub>rs,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>unrea</sub>li<sub>s</sub>ti<sub>c</sub> i<sub>n</sub> <sub>sma</sub>ll<sub>-</sub>b<sub>u</sub>d<sub>ge</sub>t <sub>se</sub>tti<sub>ngs.</sub> M<sub>ore</sub> f<sub>un</sub>d<sub>amen</sub>t<sub>a</sub>ll<sub>y,</sub> th<sub>ey</sub> <sub>on</sub>l<sub>y</sub> <sub>cons</sub>id<sub>er</sub> <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>compar</sub>i<sub>sons,</sub> <sub>an</sub>d it i<sub>s</sub> <sub>no</sub>t <sub>c</sub>l<sub>ear</sub> h<sub>ow</sub> <sub>one</sub> <sub>cou</sub>ld <sub>com</sub>bi<sub>ne</sub> <sub>ra</sub>ti<sub>ngs</sub> <sub>an</sub>d <sub>compar</sub>i<sub>sons</sub> <sub>un</sub>d<sub>er</sub> thi<sub>s mo</sub>d<sub>e</sub>l<sub>.</sub>

I<sub>n</sub> th<sub>e</sub> <sub>crow</sub>d<sub>sourc</sub>i<sub>ng</sub> lit<sub>era</sub>t<sub>ure,</sub> th<sub>ese</sub> <sub>pro</sub>bl<sub>ems</sub> <sub>are</sub> <sub>more</sub> <sub>common</sub>l<sub>y</sub> <sub>s</sub>t<sub>u</sub>di<sub>e</sub>d <sub>un</sub>d<sub>er</sub> l<sub>a</sub>t<sub>en</sub>t<sub>-u</sub>tilit<sub>y</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> Chen et al. (2016) develo<sub>p</sub> an a<sub>pp</sub>roximate KG al<sub>g</sub>orithm for learnin<sub>g</sub> the full rankin<sub>g</sub> of items (as o<sub>pp</sub>osed to the to<sub>p</sub>-�) from <sub>p</sub>airwise com<sub>p</sub>arisons. Li et al. (2018), Perez-Ortiz et al. (2020), Fa<sub>g</sub>eot et al. (2026)´ develo<sub>p</sub> a unified statistical model for combinin<sub>g p</sub>airwise com<sub>p</sub>arisons and ratin<sub>g</sub>s. Fa<sub>g</sub>eot et al. (2026) show <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>ll<sub>y</sub> th<sub>a</sub>t <sub>com</sub>bi<sub>n</sub>i<sub>ng</sub> <sub>ra</sub>ti<sub>ngs</sub> <sub>an</sub>d <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>compar</sub>i<sub>sons</sub> <sub>can</sub> i<sub>mprove</sub> <sub>es</sub>ti<sub>ma</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t <sub>u</sub>tiliti<sub>es.</sub> More directl<sub>y</sub> related to our <sub>p</sub>roblem, Li et al. (2018) also <sub>g</sub>ive a heuristic a<sub>pp</sub>roach for choosin<sub>g</sub> evaluations f<sub>or</sub> t<sub>op-</sub>� <sub>se</sub>l<sub>ec</sub>ti<sub>on,</sub> <sub>a</sub> <sub>s</sub>li<sub>g</sub>htl<sub>y</sub> <sub>mo</sub>difi<sub>e</sub>d <sub>vers</sub>i<sub>on</sub> <sub>o</sub>f <sub>w</sub>hi<sub>c</sub>h <sub>we</sub> <sub>use</sub> <sub>as</sub> <sub>a</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>.</sub>

Fi<sub>na</sub>ll<sub>y,</sub> t<sub>op-</sub>� id<sub>en</sub>tifi<sub>ca</sub>ti<sub>on</sub> f<sub>rom</sub> <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>compar</sub>i<sub>sons</sub> i<sub>s</sub> <sub>we</sub>ll <sub>s</sub>t<sub>u</sub>di<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> d<sub>ue</sub>li<sub>ng-</sub>b<sub>an</sub>dit <sub>an</sub>d <sub>ac</sub>ti<sub>ve-</sub> <sub>ran</sub>ki<sub>ng</sub> lit<sub>era</sub>t<sub>ure.</sub> U<sub>sua</sub>ll<sub>y se</sub>t i<sub>n</sub> th<sub>e</sub> fi<sub>xe</sub>d<sub>-con</sub>fid<sub>ence reg</sub>i<sub>me w</sub>h<sub>ere samp</sub>li<sub>ng con</sub>ti<sub>nues un</sub>til <sub>a</sub> t<sub>arge</sub>t confidence is reached (as o<sub>pp</sub>osed to a fixed bud<sub>g</sub>et �), these al<sub>g</sub>orithms tend to require lar<sub>g</sub>e numbers of evaluations—far be<sub>y</sub>ond the bud<sub>g</sub>ets we consider. See, e.<sub>g</sub>., Mohajer et al. (2017), Ren et al. (2018), Chen et al. (2018), Heckel et al. (2019), Saha and Go<sub>p</sub>alan (2019), Ren et al. (2020), Saha and Go<sub>p</sub>alan (2020), Goldber<sub>g</sub>er and Rudi (2026a) for recent develo<sub>p</sub>ments and Ben<sub>g</sub>s et al. (2021) for a review.

I<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f <sub>exp</sub>l<sub>a</sub>i<sub>n</sub>i<sub>ng our eva</sub>l<sub>ua</sub>ti<sub>on a</sub>ll<sub>oca</sub>ti<sub>on approac</sub>h di<sub>rec</sub>tl<sub>y, we</sub> fi<sub>n</sub>d it <sub>eas</sub>i<sub>er</sub> t<sub>o</sub> l<sub>ay ou</sub>t <sub>w</sub>h<sub>ere</sub> it <sub>came</sub> f<sub>rom.</sub> O<sub>ur</sub> <sub>s</sub>t<sub>ar</sub>ti<sub>ng</sub> <sub>po</sub>i<sub>n</sub>t <sub>was</sub> th<sub>e</sub> KG <sub>po</sub>li<sub>cy</sub> <sub>pr</sub>i<sub>mar</sub>il<sub>y</sub> b<sub>ecause,</sub> <sub>among</sub> th<sub>e</sub> <sub>ex</sub>i<sub>s</sub>ti<sub>ng</sub> <sub>me</sub>th<sub>o</sub>d<sub>s,</sub> it i<sub>s</sub> <sub>we</sub>ll <sub>su</sub>it<sub>e</sub>d f<sub>or sma</sub>ll<sub>-</sub>b<sub>u</sub>d<sub>ge</sub>t <sub>pro</sub>bl<sub>ems an</sub>d <sub>can na</sub>t<sub>ura</sub>ll<sub>y accommo</sub>d<sub>a</sub>t<sub>e</sub> th<sub>e com</sub>bi<sub>na</sub>ti<sub>on o</sub>f <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>ypes—our</sub> k<sub>ey</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>mo</sub>ti<sub>va</sub>ti<sub>on</sub> f<sub>or un</sub>d<sub>er</sub>t<sub>a</sub>ki<sub>ng</sub> thi<sub>s researc</sub>h<sub>.</sub> H<sub>owever,</sub> th<sub>e</sub> KG <sub>po</sub>li<sub>cy a</sub>ll<sub>oca</sub>t<sub>es eva</sub>l<sub>ua</sub>ti<sub>ons</sub> i<sub>n eac</sub>h <sub>roun</sub>d <sub>as</sub> if th<sub>e</sub> fi<sub>na</sub>l <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> <sub>wou</sub>ld i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y</sub> f<sub>o</sub>ll<sub>ow.</sub> E<sub>xcep</sub>t f<sub>or</sub> th<sub>e</sub> l<sub>as</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on,</sub> th<sub>a</sub>t i<sub>s</sub> <sub>no</sub>t th<sub>e</sub> <sub>ac</sub>t<sub>ua</sub>l t<sub>as</sub>k<sub>.</sub> Th<sub>e</sub> t<sub>as</sub>k i<sub>s</sub> t<sub>o</sub> <sub>prepare</sub> f<sub>or</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>se</sub>l<sub>ec</sub>ti<sub>on,</sub> <sub>an</sub>d it <sub>may</sub> b<sub>e</sub> th<sub>a</sub>t th<sub>e</sub> b<sub>es</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>o</sub> d<sub>o</sub> <sub>nex</sub>t i<sub>s</sub> d<sub>ec</sub>id<sub>e</sub>d b<sub>y</sub> h<sub>ow</sub> it <sub>may</sub> i<sub>n</sub>f<sub>orm</sub> l<sub>a</sub>t<sub>er</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>c</sub>h<sub>o</sub>i<sub>ces.</sub> T<sub>o</sub> ill<sub>us</sub>t<sub>ra</sub>t<sub>e</sub> h<sub>ow</sub> <sub>muc</sub>h KG <sub>m</sub>i<sub>sses</sub> <sub>ou</sub>t <sub>on</sub> f<sub>u</sub>t<sub>ure</sub> d<sub>ynam</sub>i<sub>cs,</sub> <sub>cons</sub>id<sub>er</sub> th<sub>a</sub>t i<sub>n</sub> it<sub>s eyes any eva</sub>l<sub>ua</sub>ti<sub>on w</sub>h<sub>ose ou</sub>t<sub>come canno</sub>t <sub>c</sub>h<sub>ange</sub> th<sub>e curren</sub>t t<sub>op-</sub>� <sub>se</sub>t h<sub>as no va</sub>l<sub>ue a</sub>t <sub>a</sub>ll<sub>.</sub>

Id<sub>ea</sub>ll<sub>y, we wou</sub>ld i<sub>ncorpora</sub>t<sub>e</sub> th<sub>e</sub> f<sub>u</sub>ll d<sub>ynam</sub>i<sub>cs</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e eva</sub>l<sub>ua</sub>ti<sub>on c</sub>h<sub>o</sub>i<sub>ces,</sub> b<sub>u</sub>t b<sub>ase</sub>d <sub>on</sub> th<sub>e ex</sub>i<sub>s</sub>ti<sub>ng</sub> lit<sub>era</sub>t<sub>ure, any</sub>thi<sub>ng</sub> b<sub>eyon</sub>d <sub>so</sub>l<sub>v</sub>i<sub>ng a sequence o</sub>f <sub>s</sub>t<sub>a</sub>t<sub>e-</sub>d<sub>epen</sub>d<sub>en</sub>t<sub>, s</sub>t<sub>a</sub>ti<sub>c pro</sub>bl<sub>ems seems ou</sub>t <sub>o</sub>f <sub>reac</sub>h<sub>.</sub> O<sub>ur</sub> k<sub>ey</sub> i<sub>ns</sub>i<sub>g</sub>ht l<sub>ea</sub>di<sub>ng</sub> t<sub>o our approac</sub>h <sub>came</sub> f<sub>rom</sub> fli<sub>pp</sub>i<sub>ng</sub> th<sub>e perspec</sub>ti<sub>ve</sub> f<sub>rom</sub> th<sub>e pos</sub>iti<sub>ve</sub> f<sub>ocus on w</sub>h<sub>a</sub>t th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>k<sub>er</sub> <sub>wan</sub>t<sub>s</sub> t<sub>o</sub> <sub>ac</sub>hi<sub>eve—c</sub>h<sub>oos</sub>i<sub>ng</sub> th<sub>e</sub> b<sub>es</sub>t � it<sub>ems—</sub>t<sub>o</sub> <sub>a</sub> <sub>nega</sub>ti<sub>ve</sub> <sub>perspec</sub>ti<sub>ve—</sub>h<sub>ow</sub> t<sub>o</sub> <sub>re</sub>d<sub>uce</sub> errors. There are two ways to err ex-post: including an item that does not belong (inclusion error) and excluding one that does (exclusion error). Ex-ante, items have an expected mismatch loss: the expected <sub>magn</sub>it<sub>u</sub>d<sub>e o</sub>f <sub>an</sub> i<sub>nc</sub>l<sub>us</sub>i<sub>on error</sub> f<sub>or</sub> it<sub>ems</sub> i<sub>n</sub> th<sub>e curren</sub>t t<sub>op-</sub>� <sub>an</sub>d <sub>an exc</sub>l<sub>us</sub>i<sub>on error</sub> f<sub>or</sub> th<sub>ose no</sub>t i<sub>n</sub> it<sub>.</sub> W<sub>e</sub> then construct a per-round surrogate objective that values uncertainty reduction by how much it reduces th<sub>e expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss.</sub> W<sub>e con</sub>t<sub>ras</sub>t thi<sub>s w</sub>ith th<sub>e</sub> KG <sub>po</sub>li<sub>cy</sub> th<sub>a</sub>t <sub>a</sub>d<sub>op</sub>t<sub>s</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l t<sub>op-</sub>� EOC <sub>as</sub> each round’s objective. It uses the current uncertainty to assess possible changes in the membership of the top-� posterior means. Our method instead holds the current means fixed and values an evaluation by how much it is expected to reduce uncertainty. The terminal top-� decision then drives the evaluation choices <sub>on</sub>l<sub>y</sub> i<sub>mp</sub>li<sub>c</sub>itl<sub>y</sub> b<sub>y</sub> <sub>prov</sub>idi<sub>ng</sub> <sub>a</sub> <sub>measure</sub> <sub>o</sub>f <sub>w</sub>h<sub>ere</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> <sub>re</sub>d<sub>uc</sub>ti<sub>on</sub> i<sub>s</sub> <sub>more</sub> <sub>va</sub>l<sub>ua</sub>bl<sub>e.</sub> Thi<sub>s</sub> <sub>surroga</sub>t<sub>e</sub> i<sub>s</sub> <sub>na</sub>t<sub>ura</sub>ll<sub>y</sub> f<sub>orwar</sub>d<sub>-</sub>l<sub>oo</sub>ki<sub>ng.</sub> It <sub>see</sub>k<sub>s</sub> t<sub>o re</sub>d<sub>uce uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y a</sub>b<sub>ou</sub>t th<sub>e</sub> it<sub>ems curren</sub>tl<sub>y near</sub> th<sub>e</sub> t<sub>op-</sub>� <sub>cu</sub>t<sub>o</sub>f <sub>w</sub>ith<sub>ou</sub>t <sub>requ</sub>i<sub>r</sub>i<sub>ng</sub> <sub>any</sub> di<sub>rec</sub>t <sub>c</sub>h<sub>ange</sub> t<sub>o</sub> th<sub>e</sub> <sub>curren</sub>t t<sub>op-</sub>� <sub>se</sub>t<sub>.</sub> D<sub>o</sub>i<sub>ng</sub> <sub>so</sub> <sub>pays</sub> <sub>o</sub>f i<sub>n</sub> f<sub>u</sub>t<sub>ure</sub> <sub>roun</sub>d<sub>s,</sub> <sub>s</sub>i<sub>nce</sub> th<sub>e</sub> <sub>more</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> <sub>we</sub> <sub>are</sub> <sub>a</sub>t <sub>s</sub>i<sub>ng</sub>li<sub>ng</sub> <sub>ou</sub>t it<sub>ems</sub> <sub>near</sub> th<sub>e</sub> t<sub>op-</sub>� <sub>cu</sub>t<sub>o</sub>f<sub>,</sub> th<sub>e</sub> <sub>more</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>we</sub> <sub>can</sub> di<sub>rec</sub>t th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons.</sub>

Th<sub>e</sub> difi<sub>cu</sub>lt<sub>y</sub> i<sub>s</sub> th<sub>a</sub>t <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> d<sub>oes no</sub>t <sub>a</sub>d<sub>m</sub>it <sub>an ana</sub>l<sub>ys</sub>i<sub>s-</sub>f<sub>r</sub>i<sub>en</sub>dl<sub>y express</sub>i<sub>on.</sub> Th<sub>e magn</sub>it<sub>u</sub>d<sub>e o</sub>f an item’s error depends on where it stands relative to the �th or (� +1)th best of the remaining items, so computing an item’s mismatch loss requires an expectation with respect to the joint utility distribution and an order statistic of the other �−1 items, where even the latter itself is intractable. Our resolution is to re<sub>p</sub>lace thi<sub>s ran</sub>d<sub>om</sub> t<sub>op-</sub>� th<sub>res</sub>h<sub>o</sub>ld <sub>w</sub>ith <sub>a s</sub>i<sub>mp</sub>l<sub>er</sub> th<sub>res</sub>h<sub>o</sub>ld th<sub>a</sub>t i<sub>s reca</sub>lib<sub>ra</sub>t<sub>e</sub>d <sub>eac</sub>h <sub>roun</sub>d<sub>, suc</sub>h th<sub>a</sub>t<sub>,</sub> i<sub>n expec</sub>t<sub>a</sub>ti<sub>on,</sub> � it<sub>ems</sub>’ <sub>u</sub>tiliti<sub>es w</sub>ill <sub>excee</sub>d it<sub>.</sub> With thi<sub>s</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>, an</sub> it<sub>em</sub>’<sub>s m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> d<sub>epen</sub>d<sub>s on</sub>l<sub>y on</sub> th<sub>e marg</sub>i<sub>na</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f it<sub>s u</sub>tilit<sub>y re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> it<sub>.</sub> Thi<sub>s separa</sub>bilit<sub>y g</sub>i<sub>ves a c</sub>l<sub>ean an</sub>d i<sub>n</sub>t<sub>erpre</sub>t<sub>a</sub>bl<sub>e way</sub> t<sub>o accoun</sub>t f<sub>or an</sub> evaluation’s efect on all items, those being directly evaluated and, via covariance, those that are not. The sum of these mismatch losses is our surrogate objective, and in each round, the policy chooses the evaluation <sub>w</sub>h<sub>ose</sub> <sub>re</sub>d<sub>uc</sub>ti<sub>on</sub> <sub>o</sub>f thi<sub>s</sub> l<sub>oss</sub> <sub>per</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>cos</sub>t i<sub>s</sub> l<sub>arges</sub>t<sub>.</sub>

W<sub>e</sub> <sub>a</sub>l<sub>so</sub> <sub>po</sub>i<sub>n</sub>t <sub>ou</sub>t <sub>a</sub> <sub>po</sub>t<sub>en</sub>ti<sub>a</sub>l t<sub>rap</sub> th<sub>a</sub>t <sub>ar</sub>i<sub>ses</sub> <sub>w</sub>h<sub>en</sub> <sub>com</sub>bi<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>wo</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>ypes,</sub> i<sub>n</sub> <sub>par</sub>t b<sub>ecause</sub> <sub>we</sub> i<sub>n</sub>iti<sub>a</sub>ll<sub>y</sub> f<sub>e</sub>ll i<sub>n</sub>t<sub>o</sub> it<sub>.</sub> R<sub>a</sub>ti<sub>ngs</sub> <sub>con</sub>t<sub>a</sub>i<sub>n</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>a</sub>b<sub>ou</sub>t th<sub>e</sub> <sub>average</sub> <sub>u</sub>tilit<sub>y</sub> <sub>o</sub>f th<sub>e</sub> it<sub>ems,</sub> <sub>w</sub>hil<sub>e</sub> <sub>compar</sub>i<sub>sons</sub> d<sub>o</sub> <sub>no</sub>t<sub>.</sub> L<sub>earn</sub>i<sub>ng a</sub>b<sub>ou</sub>t thi<sub>s average re</sub>d<sub>uces</sub> th<sub>e var</sub>i<sub>ance o</sub>f <sub>eac</sub>h it<sub>em</sub>’<sub>s u</sub>tilit<sub>y</sub> b<sub>u</sub>t d<sub>oes no</sub>t h<sub>e</sub>l<sub>p</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h th<sub>e</sub> t<sub>op-</sub>�<sub>, s</sub>i<sub>nce s</sub>hifti<sub>ng a</sub>ll <sub>u</sub>tiliti<sub>es</sub> t<sub>oge</sub>th<sub>er</sub> l<sub>eaves</sub> th<sub>e</sub>i<sub>r or</sub>d<sub>er</sub>i<sub>ng unc</sub>h<sub>ange</sub>d<sub>.</sub> A <sub>surroga</sub>t<sub>e</sub> th<sub>a</sub>t <sub>rewar</sub>d<sub>s</sub> thi<sub>s</sub> reduction in variance would therefore unfairl<sub>y</sub> favor ratin<sub>g</sub>s over com<sub>p</sub>arisons. We discuss this issue in §3 and <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>our</sub> <sub>surroga</sub>t<sub>e</sub> d<sub>oes</sub> <sub>no</sub>t <sub>rewar</sub>d l<sub>earn</sub>i<sub>ng</sub> <sub>a</sub>b<sub>ou</sub>t th<sub>e</sub> <sub>average</sub> <sub>u</sub>tilit<sub>y,</sub> <sub>so</sub> <sub>ra</sub>ti<sub>ngs</sub> <sub>an</sub>d <sub>compar</sub>i<sub>sons</sub> <sub>are</sub> <sub>on</sub> <sub>equa</sub>l f<sub>oo</sub>ti<sub>ng.</sub>

A<sub>s</sub> th<sub>e</sub> h<sub>or</sub>i<sub>zon</sub> <sub>approac</sub>h<sub>es</sub> it<sub>s</sub> <sub>en</sub>d<sub>,</sub> b<sub>e</sub>i<sub>ng</sub> f<sub>orwar</sub>d<sub>-</sub>l<sub>oo</sub>ki<sub>ng</sub> b<sub>ecomes</sub> l<sub>ess</sub> <sub>va</sub>l<sub>ua</sub>bl<sub>e,</sub> <sub>an</sub>d thi<sub>s</sub> <sub>s</sub>t<sub>reng</sub>th <sub>o</sub>f th<sub>e</sub> <sub>surroga</sub>t<sub>e</sub> <sub>can</sub> b<sub>ecome</sub> <sub>a</sub> <sub>wea</sub>k<sub>ness.</sub> I<sub>n</sub> thi<sub>s</sub> <sub>reg</sub>i<sub>me,</sub> th<sub>e</sub> <sub>myop</sub>i<sub>a</sub> <sub>o</sub>f th<sub>e</sub> KG <sub>po</sub>li<sub>cy</sub> <sub>wou</sub>ld b<sub>e</sub> d<sub>es</sub>i<sub>ra</sub>bl<sub>e.</sub> W<sub>e</sub> therefore develop the KG policy for our problem, with a switching criterion for the endgame, the final rounds <sub>w</sub>h<sub>ere</sub> it<sub>s</sub> <sub>myop</sub>i<sub>a</sub> <sub>ou</sub>t<sub>per</sub>f<sub>orms</sub> th<sub>e</sub> <sub>surroga</sub>t<sub>e.</sub>

W<sub>e</sub> <sub>suppor</sub>t <sub>our</sub> <sub>me</sub>th<sub>o</sub>d th<sub>roug</sub>h b<sub>o</sub>th <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>an</sub>d <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on.</sub> I<sub>n</sub> th<sub>e</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s,</sub> <sub>a</sub>lth<sub>oug</sub>h th<sub>e</sub> <sub>surroga</sub>t<sub>e</sub> objective was constructed from analytical intuition rather than derived, we show that it can be obtained from an upper bound on the EOC, which rigorously justifies and refines our choice of the top-� threshold. We also <sub>prove</sub> th<sub>a</sub>t th<sub>e resu</sub>lti<sub>ng po</sub>li<sub>cy</sub> th<sub>a</sub>t <sub>a</sub>ll<sub>oca</sub>t<sub>es</sub> b<sub>y m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> th<sub>e surroga</sub>t<sub>e</sub> i<sub>n eac</sub>h <sub>roun</sub>d i<sub>s asymp</sub>t<sub>o</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>t<sub>.</sub> I<sub>n s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on, we s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e a</sub>ll<sub>oca</sub>ti<sub>ons</sub> thi<sub>s po</sub>li<sub>cy pro</sub>d<sub>uces are</sub> i<sub>n</sub>t<sub>u</sub>iti<sub>ve an</sub>d i<sub>ns</sub>i<sub>g</sub>htf<sub>u</sub>l<sub>, an</sub>d th<sub>a</sub>t it <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y ou</sub>t<sub>per</sub>f<sub>orms ex</sub>i<sub>s</sub>ti<sub>ng</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s.</sub>

The remainder of the <sub>p</sub>a<sub>p</sub>er is or<sub>g</sub>anized as follows: §2 <sub>p</sub>resents the model and defines the objective. §3 develo<sub>p</sub>s the surro<sub>g</sub>ate objective<sub>,</sub> which §4 uses to construct the se<sub>q</sub>uential evaluation <sub>p</sub>rocedure and establish its as<sub>y</sub>m<sub>p</sub>totic consistenc<sub>y</sub>. §5 develo<sub>p</sub>s the knowled<sub>g</sub>e <sub>g</sub>radient <sub>p</sub>olic<sub>y</sub> for the end<sub>g</sub>ame and the switchin<sub>g</sub> rule.

§6 considers the batch settin<sub>g</sub>. §7 re<sub>p</sub>orts the results of the numerical ex<sub>p</sub>eriments<sub>,</sub> and $\ S 8$ <sub>g</sub>i<sub>ves</sub> <sub>conc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>remar</sub>k<sub>s.</sub>

## 2. Model and objective

Thi<sub>s</sub> <sub>sec</sub>ti<sub>on</sub> d<sub>e</sub>fi<sub>nes</sub> th<sub>e</sub> <sub>genera</sub>ti<sub>ve</sub> d<sub>a</sub>t<sub>a</sub> <sub>mo</sub>d<sub>e</sub>l f<sub>or</sub> <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>compar</sub>i<sub>sons</sub> <sub>an</sub>d <sub>ra</sub>ti<sub>ngs,</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uces</sub> th<sub>e</sub> <sub>sequen</sub>ti<sub>a</sub>l <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>process</sub> <sub>an</sub>d f<sub>ormu</sub>l<sub>a</sub>t<sub>es</sub> th<sub>e</sub> <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> <sub>pro</sub>bl<sub>em</sub> <sub>as</sub> <sub>a</sub> d<sub>ynam</sub>i<sub>c</sub> <sub>program.</sub> W<sub>e</sub> th<sub>en</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>a</sub>t the KG <sub>p</sub>olic<sub>y</sub> is not well-suited for our settin<sub>g,</sub> which motivates the surro<sub>g</sub>ate objective of §3.

## 2.1. Generative model

W<sub>e cons</sub>id<sub>er</sub> � it<sub>ems w</sub>ith l<sub>a</sub>t<sub>en</sub>t <sub>u</sub>tiliti<sub>es</sub> $\pmb { \theta } = ( \theta _ { 1 } , \dots , \theta _ { I } ) \in \mathbb { R } ^ { I }$ <sub>, w</sub>hi<sub>c</sub>h <sub>represen</sub>t th<sub>e va</sub>l<sub>ues o</sub>f <sub>se</sub>l<sub>ec</sub>ti<sub>ng eac</sub>h it<sub>em.</sub> W<sub>e</sub> <sub>a</sub>d<sub>op</sub>t <sub>a</sub> B<sub>ayes</sub>i<sub>an</sub> f<sub>ramewor</sub>k <sub>an</sub>d <sub>p</sub>l<sub>ace</sub> <sub>a</sub> G<sub>auss</sub>i<sub>an</sub> <sub>pr</sub>i<sub>or</sub> <sub>on</sub> th<sub>e</sub> <sub>u</sub>tilit<sub>y</sub> <sub>vec</sub>t<sub>or,</sub> $\pmb { \theta } \sim N ( \mu _ { 0 } , \Sigma _ { 0 } )$ <sub>w</sub>h<sub>ere</sub> $\Sigma _ { 0 }$ i<sub>s</sub> <sub>pos</sub>iti<sub>ve</sub> d<sub>e</sub>fi<sub>n</sub>it<sub>e.</sub> Wh<sub>en</sub> <sub>pre</sub>li<sub>m</sub>i<sub>nary</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>es</sub> th<sub>e</sub> it<sub>ems,</sub> $\pmb { \mu } _ { 0 }$ <sub>an</sub>d $\Sigma _ { 0 }$ <sub>can</sub> <sub>re</sub>fl<sub>ec</sub>t it<sub>.</sub> Wh<sub>en</sub> <sub>no</sub> <sub>suc</sub>h i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> i<sub>s</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e,</sub> <sub>or</sub> <sub>w</sub>h<sub>en</sub> it<sub>ems</sub> <sub>are</sub> <sub>peop</sub>l<sub>e</sub> <sub>an</sub>d <sub>a</sub> <sub>pr</sub>i<sub>or</sub> th<sub>a</sub>t di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>t<sub>es</sub> <sub>ra</sub>i<sub>ses</sub> <sub>e</sub>thi<sub>ca</sub>l <sub>or</sub> l<sub>ega</sub>l <sub>concerns,</sub> <sub>an</sub> <sub>exc</sub>h<sub>angea</sub>bl<sub>e</sub> <sub>pr</sub>i<sub>or</sub> $( \pmb { \mu } _ { 0 } = \mathbf { 0 } , \pmb { \Sigma } _ { 0 } = s \operatorname { I d } _ { I } )$ t<sub>rea</sub>t<sub>s</sub> <sub>a</sub>ll it<sub>ems</sub> <sub>symme</sub>t<sub>r</sub>i<sub>ca</sub>ll<sub>y.</sub>

Evaluators can provide two types of feedback, pairwise comparisons and ratings. A pairwise comparison of items � and � produces a binary outcome: one item is preferred to the other. The event that an evaluator <sub>p</sub>refers � to $j$ i<sub>s</sub> d<sub>eno</sub>t<sub>e</sub>d b<sub>y</sub> $i \succ j$ <sub>,</sub> <sub>an</sub>d it<sub>s</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> $\theta _ { i } - \theta _ { j } + \epsilon _ { 1 }$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $\epsilon _ { 1 } \sim N ( 0 , \sigma _ { 1 } ^ { 2 } )$ <sub>.</sub> Th<sub>us,</sub> th<sub>e eva</sub>l<sub>ua</sub>t<sub>or</sub>’<sub>s a</sub>bilit<sub>y</sub> t<sub>o</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> t<sub>wo</sub> it<sub>ems</sub> d<sub>epen</sub>d<sub>s on</sub> th<sub>e</sub> dif<sub>erence</sub> i<sub>n</sub> th<sub>e</sub>i<sub>r u</sub>tiliti<sub>es an</sub>d <sub>on</sub> $\epsilon _ { 1 }$ <sub>, w</sub>hi<sub>c</sub>h <sub>cap</sub>t<sub>ures</sub> i<sub>mper</sub>f<sub>ec</sub>ti<sub>ons</sub> i<sub>n pa</sub>i<sub>rw</sub>i<sub>se eva</sub>l<sub>ua</sub>ti<sub>ons.</sub> Th<sub>e su</sub>b<sub>scr</sub>i<sub>p</sub>t 1 i<sub>s use</sub>d b<sub>ecause</sub> it<sub>em</sub> � i<sub>s compare</sub>d to 1 other item. A rating of item � produces a binary ‘thumbs-up’ or ‘thumbs-down’ outcome, where item � is i<sub>mp</sub>li<sub>c</sub>itl<sub>y</sub> <sub>compare</sub>d t<sub>o</sub> <sub>some</sub> <sub>u</sub>tilit<sub>y</sub> th<sub>res</sub>h<sub>o</sub>ld $\theta _ { 0 }$ <sub>.</sub> N<sub>o</sub>t<sub>e</sub> th<sub>a</sub>t $\theta _ { 0 }$ i<sub>s</sub> <sub>no</sub>t <sub>a</sub> <sub>par</sub>t <sub>o</sub>f $\pmb { \theta } \mathrm { - } \mathbf { w } \mathbf { e }$ <sub>use</sub> thi<sub>s no</sub>t<sub>a</sub>ti<sub>on</sub> t<sub>o</sub> f<sub>ac</sub>ilit<sub>a</sub>t<sub>e</sub> <sub>un</sub>ifi<sub>e</sub>d <sub>express</sub>i<sub>ons</sub> f<sub>or</sub> <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>compar</sub>i<sub>sons</sub> <sub>an</sub>d <sub>ra</sub>ti<sub>ngs.</sub> Si<sub>m</sub>il<sub>ar</sub> t<sub>o</sub> th<sub>e</sub> <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>compar</sub>i<sub>son</sub> <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> th<sub>e even</sub>t th<sub>a</sub>t <sub>an eva</sub>l<sub>ua</sub>t<sub>or ran</sub>k<sub>s</sub> � <sub>a</sub>b<sub>ove</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld i<sub>s</sub> d<sub>eno</sub>t<sub>e</sub>d b<sub>y</sub> $i \succ 0$ <sub>an</sub>d it<sub>s pro</sub>b<sub>a</sub>bilit<sub>y</sub> i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> $\theta _ { i } - \theta _ { 0 } + \epsilon _ { 0 }$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $\epsilon _ { 0 } \sim N ( 0 , \sigma _ { 0 } ^ { 2 } )$ <sub>.</sub> H<sub>ere</sub> $\epsilon _ { 0 }$ <sub>cap</sub>t<sub>ures</sub> t<sub>wo</sub> <sub>sources</sub> <sub>o</sub>f <sub>no</sub>i<sub>se:</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>or</sub>’<sub>s</sub> <sub>a</sub>bilit<sub>y</sub> t<sub>o</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> it<sub>em</sub> $i \ ' \mathbf { s }$ <sub>u</sub>tilit<sub>y</sub> <sub>an</sub>d th<sub>e</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> <sub>a</sub>b<sub>ou</sub>t <sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t th<sub>res</sub>h<sub>o</sub>ld $\theta _ { 0 }$ li<sub>es.</sub> Th<sub>e</sub> <sub>su</sub>b<sub>scr</sub>i<sub>p</sub>t 0 i<sub>s</sub> <sub>use</sub>d b<sub>ecause</sub> it<sub>em</sub> � i<sub>s compare</sub>d t<sub>o</sub> 0 it<sub>ems.</sub>

Th<sub>e se</sub>t <sub>o</sub>f <sub>poss</sub>ibl<sub>e eva</sub>l<sub>ua</sub>ti<sub>ons</sub> i<sub>s g</sub>i<sub>ven</sub> b<sub>y</sub> $\mathcal { T } : = \left\{ ( i , j ) : 0 \leq j < i \leq I \right\}$ <sub>.</sub> D<sub>e</sub>fi<sub>ne</sub> $J = \operatorname* { m i n } ( j , 1 )$ <sub>, w</sub>hi<sub>c</sub>h i<sub>s</sub> $0 ( 1 )$ when an evaluation is a ratin<sub>g</sub> (com<sub>p</sub>arison). For an evaluation $( i , j ) \in \mathcal { I }$ <sub>,</sub> l<sub>e</sub>t $X _ { i j }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e ou</sub>t<sub>come w</sub>h<sub>ere</sub>

$$
X _ { i j } = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } i \succ j } \\ { - 1 } & { { \mathrm { i f ~ } } i \prec j , } \end{array} \right. }
$$

<sub>an</sub>d<sub>, con</sub>diti<sub>one</sub>d <sub>on</sub> th<sub>e eva</sub>l<sub>ua</sub>t<sub>or-no</sub>i<sub>se</sub> t<sub>erm</sub> $\epsilon _ { J } , \mathbb { P } ( X _ { i j } = 1 \mid \pmb \theta , \epsilon _ { J } ) = g ( \theta _ { i } - \theta _ { j } + \epsilon _ { J } )$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $g : \mathbb { R } \to ( 0 , 1 )$ i<sub>s</sub> <sub>some</sub> li<sub>n</sub>k f<sub>unc</sub>ti<sub>on.</sub> F<sub>or</sub> th<sub>e rema</sub>i<sub>n</sub>d<sub>er o</sub>f th<sub>e paper, un</sub>l<sub>ess spec</sub>ifi<sub>e</sub>d <sub>o</sub>th<sub>erw</sub>i<sub>se, we cons</sub>id<sub>er</sub> th<sub>e pro</sub>bit li<sub>n</sub>k function, commonl<sub>y</sub> used to model human ordinal ratin<sub>g</sub> data (Liddell and Kruschke 2018), $\mathrm { i . e . , } g = \Phi$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> Φ is the standard normal CDF and $\phi$ it<sub>s</sub> d<sub>ens</sub>it<sub>y.</sub> Th<sub>us,</sub>

$$
\mathbb { P } ( X _ { i j } = 1 \mid \theta ) = \mathbb { E } _ { \epsilon _ { j } } \big [ \Phi \left( \theta _ { i } - \theta _ { j } + \epsilon _ { j } \right) \big ] = \Phi \left( \frac { 1 } { \sqrt { 1 + \sigma _ { j } ^ { 2 } } } ( \theta _ { i } - \theta _ { j } ) \right) = \Phi \left( \beta _ { j } ( \theta _ { i } - \theta _ { j } ) \right) ,
$$

<sub>w</sub>h<sub>ere</sub> $\begin{array} { r } { \beta _ { J } : = \frac { 1 } { \sqrt { 1 + \sigma _ { \ j } ^ { 2 } } } \in ( 0 , 1 ] } \end{array}$ <sub>,</sub> f<sub>or</sub> $\jmath \in \{ 0 , 1 \}$ <sub>,</sub> <sub>cap</sub>t<sub>ures</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>or</sub> <sub>prec</sub>i<sub>s</sub>i<sub>on</sub> <sub>w</sub>h<sub>en</sub> <sub>prov</sub>idi<sub>ng</sub> <sub>a</sub> <sub>ra</sub>ti<sub>ng</sub> <sub>or</sub> <sub>a</sub> <sub>p</sub>airwise com<sub>p</sub>arison. As shown in Perez-Ortiz et al. (2020), a sin<sub>g</sub>le latent-utilit<sub>y</sub> model for ratin<sub>g</sub>s and ´ <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>compar</sub>i<sub>sons</sub> <sub>can</sub> fit h<sub>uman</sub> d<sub>a</sub>t<sub>a</sub> <sub>we</sub>ll <sub>as</sub> l<sub>ong</sub> <sub>as</sub> th<sub>e</sub> t<sub>wo</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>ypes</sub> <sub>are</sub> <sub>a</sub>ll<sub>owe</sub>d t<sub>o</sub> dif<sub>er</sub> i<sub>n</sub> <sub>s</sub>i<sub>gna</sub>l <sub>qua</sub>lit<sub>y,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>we</sub> <sub>cap</sub>t<sub>ure</sub> th<sub>roug</sub>h $\beta _ { 0 }$ <sub>an</sub>d $\beta _ { 1 }$ <sub>.</sub> I<sub>n</sub> <sub>a</sub>dditi<sub>on</sub> t<sub>o</sub> <sub>var</sub>i<sub>a</sub>ti<sub>on</sub> i<sub>n</sub> <sub>s</sub>i<sub>gna</sub>l <sub>qua</sub>lit<sub>y,</sub> th<sub>ere</sub> <sub>may</sub> <sub>a</sub>l<sub>so</sub> b<sub>e</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> i<sub>n</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>or</sub> <sub>accuracy</sub> <sub>an</sub>d bi<sub>as.</sub> If th<sub>ese</sub> dif<sub>erences</sub> <sub>were</sub> k<sub>nown,</sub> <sub>our</sub> <sub>mo</sub>d<sub>e</sub>l <sub>wou</sub>ld <sub>cap</sub>t<sub>ure</sub> thi<sub>s</sub> b<sub>y</sub> <sub>us</sub>i<sub>ng</sub> dif<sub>eren</sub>t <sub>va</sub>l<sub>ues</sub> <sub>o</sub>f $\theta _ { 0 } , \beta _ { 0 }$ <sub>,</sub> <sub>an</sub>d $\beta _ { 1 }$ f<sub>or</sub> <sub>eac</sub>h <sub>eva</sub>l<sub>ua</sub>t<sub>or.</sub> B<sub>ecause</sub> th<sub>ey</sub> <sub>are</sub> <sub>genera</sub>ll<sub>y</sub> <sub>un</sub>k<sub>nown,</sub> <sub>we</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>use</sub> <sub>common</sub> <sub>parame</sub>t<sub>ers</sub> <sub>across</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>ors,</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>a</sub>b<sub>sor</sub>bi<sub>ng</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>or-</sub>l<sub>eve</sub>l <sub>var</sub>i<sub>a</sub>ti<sub>on</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> <sub>no</sub>i<sub>se</sub> t<sub>erms</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> l<sub>earn</sub>i<sub>ng</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>or-spec</sub>ifi<sub>c</sub> <sub>accuracy</sub> <sub>or</sub> bi<sub>as</sub> f<sub>rom</sub> <sub>pas</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>ons.</sub> Thi<sub>s</sub> i<sub>s</sub> <sub>a</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>s</sub>i<sub>mp</sub>lifi<sub>ca</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e</sub> lit<sub>era</sub>t<sub>ure, an</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s us</sub>i<sub>ng</sub> it h<sub>ave</sub> b<sub>een s</sub>h<sub>own</sub> t<sub>o s</sub>till fit h<sub>uman</sub> d<sub>a</sub>t<sub>a we</sub>ll (Perez-Ortiz et al. 2020). Learnin<sub>g</sub> evaluator-s<sub>p</sub>ecific <sub>p</sub>arameters could further inform the allocation, but´ <sub>wou</sub>ld i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> <sub>an</sub> <sub>a</sub>dditi<sub>ona</sub>l i<sub>n</sub>f<sub>erence</sub> <sub>pro</sub>bl<sub>em</sub> th<sub>a</sub>t <sub>we</sub> l<sub>eave</sub> f<sub>or</sub> f<sub>u</sub>t<sub>ure</sub> <sub>wor</sub>k<sub>.</sub> I<sub>n</sub> th<sub>e</sub> <sub>numer</sub>i<sub>ca</sub>l <sub>exper</sub>i<sub>men</sub>t<sub>s</sub> <sub>o</sub>f $\ S 7 _ { \colon }$ <sub>,</sub> <sub>we</sub> <sub>s</sub>t<sub>ress-</sub>t<sub>es</sub>t thi<sub>s</sub> <sub>assump</sub>ti<sub>on</sub> b<sub>y</sub> <sub>us</sub>i<sub>ng</sub> <sub>syn</sub>th<sub>e</sub>ti<sub>c</sub> d<sub>a</sub>t<sub>a</sub> <sub>w</sub>h<sub>ere</sub> <sub>accuracy</sub> <sub>an</sub>d bi<sub>as</sub> dif<sub>er</sub> <sub>across</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>ors.</sub>

O<sub>ne</sub> <sub>may</sub> <sub>ques</sub>ti<sub>on</sub> th<sub>e</sub> f<sub>eas</sub>ibilit<sub>y</sub> <sub>o</sub>f <sub>ca</sub>lib<sub>ra</sub>ti<sub>ng</sub> th<sub>ese</sub> <sub>parame</sub>t<sub>ers</sub> i<sub>n</sub> <sub>prac</sub>ti<sub>ce.</sub> R<sub>emar</sub>k 1 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>a</sub>ft<sub>er</sub> fi<sub>x</sub>i<sub>ng</sub> <sub>one</sub> f<sub>or</sub> id<sub>en</sub>tifi<sub>ca</sub>ti<sub>on,</sub> th<sub>e</sub> <sub>res</sub>t <sub>can</sub> b<sub>e</sub> <sub>ca</sub>lib<sub>ra</sub>t<sub>e</sub>d f<sub>rom</sub> th<sub>ree</sub> <sub>s</sub>i<sub>mp</sub>l<sub>e,</sub> i<sub>n</sub>t<sub>erpre</sub>t<sub>a</sub>bl<sub>e</sub> <sub>quan</sub>titi<sub>es.</sub>

REMARK 1. The binary data alone cannot identify the latent utility scale and the observation accuracy parameters simultaneously: for any $c > 0 ,$ replacing $( \pmb { \theta } , \theta _ { 0 } , \beta _ { 0 } , \beta _ { 1 } )$ with $( c \pmb { \theta } , c \theta _ { 0 } , \beta _ { 0 } / c , \beta _ { 1 } / c )$ leaves every outcome probability $\Phi \big ( \beta _ { J } ( \theta _ { i } - \theta _ { j } ) \big )$ unchanged. Thus, one parameter can (and must) be fixed, and we fix the scale of � by choosing the prior variance �. The remaining parameters $\beta _ { 1 } , \beta _ { 0 } , \theta _ { 0 }$ can then befound given the following more interpretable quantities, which we formally derive in the Appendix:

(i) The probability that a comparison of two randomly chosen items agrees with the true ordering,

$$
\frac { 1 } { 2 } + \frac { 1 } { \pi } \arcsin \left( \frac { \beta _ { 1 } \sqrt { 2 s } } { \sqrt { 1 + 2 s \beta _ { 1 } ^ { 2 } } } \right) ,
$$

which determines $\beta _ { 1 }$

(ii) The probability that a rating ofa randomly chosen item is correct,

$$
1 - \Phi \Bigg ( \frac { \theta _ { 0 } } { \sqrt { s } } \Bigg ) - \Phi \Bigg ( \frac { \beta _ { 0 } \theta _ { 0 } } { \sqrt { 1 + s \beta _ { 0 } ^ { 2 } } } \Bigg ) + 2 \Phi _ { 2 } \Bigg ( \frac { \theta _ { 0 } } { \sqrt { s } } , \frac { \beta _ { 0 } \theta _ { 0 } } { \sqrt { 1 + s \beta _ { 0 } ^ { 2 } } } ; \frac { \beta _ { 0 } \sqrt { s } } { \sqrt { 1 + s \beta _ { 0 } ^ { 2 } } } \Bigg ) ,
$$

where $\Phi _ { 2 } ( \cdot , \cdot ; \rho )$ is the standard bivariate normal CDF with correlation $\rho .$

(iii) The expected proportion of items that receive a ‘thumbs-up’ rating,

$$
\Phi \Biggl ( - \frac { \beta _ { 0 } \theta _ { 0 } } { \sqrt { 1 + s \beta _ { 0 } ^ { 2 } } } \Biggr ) ,
$$

which together with (ii) jointly determines $\beta _ { 0 }$ and $\theta _ { 0 }$

## 2.2. Sequential evaluation process

B<sub>u</sub>ildi<sub>ng on</sub> th<sub>e genera</sub>ti<sub>ve mo</sub>d<sub>e</sub>l<sub>, we nex</sub>t d<sub>escr</sub>ib<sub>e</sub> h<sub>ow</sub> th<sub>e sequen</sub>ti<sub>a</sub>l d<sub>ec</sub>i<sub>s</sub>i<sub>on process un</sub>f<sub>o</sub>ld<sub>s.</sub> I<sub>n eac</sub>h <sub>roun</sub>d $t = 1 , 2 , . . . ,$ th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>k<sub>er c</sub>h<sub>ooses an eva</sub>l<sub>ua</sub>ti<sub>on</sub> $( i _ { t } , j _ { t } ) \in \mathcal { I }$ <sub>, a ra</sub>ti<sub>ng or a pa</sub>i<sub>rw</sub>i<sub>se compar</sub>i<sub>son,</sub> <sub>an</sub>d <sub>o</sub>b<sub>serves</sub> it<sub>s ou</sub>t<sub>come</sub> $X _ { t } \in \{ - 1 , 1 \}$ b<sub>e</sub>f<sub>ore c</sub>h<sub>oos</sub>i<sub>ng</sub> th<sub>e nex</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on.</sub> L<sub>e</sub>t $r _ { 0 } > 0$ <sub>an</sub>d $r _ { 1 } > 0$ b<sub>e</sub> th<sub>e</sub> <sub>per-eva</sub>l<sub>ua</sub>ti<sub>on resource cos</sub>t <sub>o</sub>f <sub>ra</sub>ti<sub>ngs an</sub>d <sub>compar</sub>i<sub>sons, respec</sub>ti<sub>ve</sub>l<sub>y, w</sub>h<sub>ere</sub> th<sub>e resource cou</sub>ld b<sub>e mone</sub>t<sub>ary</sub> or ca<sub>p</sub>acit<sub>y</sub> (time it takes to conduct an evaluation). Let � denote the total evaluation bud<sub>g</sub>et and $R _ { t }$ th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t l<sub>e</sub>ft <sub>a</sub>t th<sub>e s</sub>t<sub>ar</sub>t <sub>o</sub>f <sub>roun</sub>d �<sub>, so</sub> $R _ { 1 } = R$ <sub>.</sub> A<sub>n</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> $( i , j )$ i<sub>s</sub> <sub>a</sub>f<sub>or</sub>d<sub>a</sub>bl<sub>e</sub> i<sub>n</sub> <sub>roun</sub>d � <sub>w</sub>h<sub>en</sub> $r _ { J } \leq R _ { t }$ <sub>.</sub> Aft<sub>er</sub> <sub>con</sub>d<sub>uc</sub>ti<sub>ng</sub> th<sub>e c</sub>h<sub>osen eva</sub>l<sub>ua</sub>ti<sub>on,</sub> it<sub>s cos</sub>t i<sub>s</sub> d<sub>e</sub>d<sub>uc</sub>t<sub>e</sub>d f<sub>rom</sub> th<sub>e rema</sub>i<sub>n</sub>i<sub>ng</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>,</sub> $R _ { t + 1 } = R _ { t } - r _ { J _ { t } }$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $J _ { t } = \operatorname* { m i n } ( j _ { t } , 1 )$ <sub>.</sub> Th<sub>e</sub> <sub>process</sub> <sub>con</sub>ti<sub>nues</sub> <sub>un</sub>til <sub>no</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> i<sub>s</sub> <sub>a</sub>f<sub>or</sub>d<sub>a</sub>bl<sub>e.</sub> L<sub>e</sub>t $T$ denote the (random) total <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> <sub>con</sub>d<sub>uc</sub>t<sub>e</sub>d<sub>,</sub> <sub>so</sub> $T : = \operatorname* { m i n } \left\{ t \geq 0 : R _ { t + 1 } < \operatorname* { m i n } ( r _ { 0 } , r _ { 1 } ) \right\}$

I<sub>n</sub> <sub>roun</sub>d �<sub>,</sub> <sub>a</sub>ft<sub>er</sub> th<sub>e</sub> <sub>ou</sub>t<sub>come</sub> i<sub>s</sub> <sub>o</sub>b<sub>serve</sub>d<sub>,</sub> <sub>we</sub> <sub>up</sub>d<sub>a</sub>t<sub>e</sub> th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>accor</sub>di<sub>ng</sub> t<sub>o</sub>

$$
p _ { t } ( \pmb \theta ) \propto p _ { t - 1 } ( \pmb \theta ) \Phi \big ( X _ { t } \beta _ { J _ { t } } ( \theta _ { i _ { t } } - \theta _ { j _ { t } } ) \big ) \propto p _ { 0 } ( \pmb \theta ) \prod _ { s = 1 } ^ { t } \Phi \big ( X _ { s } \beta _ { J _ { s } } ( \theta _ { i _ { s } } - \theta _ { j _ { s } } ) \big ) ,
$$

<sub>w</sub>h<sub>ere</sub> $p _ { 0 }$ i<sub>s</sub> th<sub>e</sub> <sub>pr</sub>i<sub>or</sub> <sub>o</sub>f $\pmb \theta$ at $t = 0$ <sub>.</sub> Th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>a</sub>ft<sub>er</sub> <sub>any</sub> <sub>roun</sub>d � i<sub>s</sub> d<sub>eno</sub>t<sub>e</sub>d b<sub>y</sub> $p _ { t }$ <sub>an</sub>d h<sub>ence</sub> it<sub>s</sub> <sub>pr</sub>i<sub>or</sub> i<sub>s</sub> th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>o</sub>f th<sub>e</sub> <sub>prev</sub>i<sub>ous</sub> <sub>roun</sub>d<sub>,</sub> $p _ { t - 1 }$ <sub>.</sub> Fi<sub>gure</sub> 1 ill<sub>us</sub>t<sub>ra</sub>t<sub>es</sub> th<sub>e</sub> <sub>sequen</sub>ti<sub>a</sub>l <sub>process.</sub> Si<sub>nce</sub> th<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> <sub>pr</sub>i<sub>or</sub> i<sub>s</sub>

![](images/c4d2c52161a01b19da0c1ab3ce6a0e8edfb9d434329a330ed9cffefebd5913b6.jpg)  
Figure 1 The sequential evaluation process.

not conjugate to the probit likelihood<sup>1</sup>, the posterior after each round is generally non-Gaussian. One option is to carr<sub>y</sub> out <sub>p</sub>osterior inference b<sub>y</sub> Markov Chain Monte Carlo (MCMC). However, MCMC <sub>y</sub>ields <sub>p</sub>osterior <sub>samp</sub>l<sub>es</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> <sub>a</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>w</sub>ith <sub>a</sub> f<sub>unc</sub>ti<sub>ona</sub>l f<sub>orm,</sub> <sub>an</sub>d t<sub>o</sub> f<sub>ormu</sub>l<sub>a</sub>t<sub>e</sub> th<sub>e</sub> <sub>roun</sub>d<sub>-</sub>� <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> <sub>pro</sub>bl<sub>em we re</sub>l<sub>y on a c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm approx</sub>i<sub>ma</sub>ti<sub>on o</sub>f $p _ { t - 1 }$ <sub>.</sub> It i<sub>s</sub> th<sub>ere</sub>f<sub>ore na</sub>t<sub>ura</sub>l t<sub>o use an approx</sub>i<sub>ma</sub>t<sub>e</sub> inference method such as ex<sub>p</sub>ectation <sub>p</sub>ro<sub>p</sub>a<sub>g</sub>ation (Minka 2001, Herbrich et al. 2006), La<sub>p</sub>lace a<sub>pp</sub>roximation (Tierne<sub>y</sub> and Kadane 1986), or variational inference (VI) (Jordan et al. 1999) with a multivariate Gaussian <sub>var</sub>i<sub>a</sub>ti<sub>ona</sub>l f<sub>am</sub>il<sub>y.</sub>

APPROXIMATION 1. After each round �, approximate $p _ { t } \approx N ( \mu _ { t } , \Sigma _ { t } )$ using one of the approximate inference methods above. We refer to this Gaussian approximation as the posterior.

We now define the terminal objective, ex<sub>p</sub>ected o<sub>pp</sub>ortunit<sub>y</sub> cost (EOC), which we ado<sub>p</sub>t for reasons discussed i<sub>n</sub> th<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on.</sub> T<sub>o wr</sub>it<sub>e</sub> EOC it i<sub>s conven</sub>i<sub>en</sub>t t<sub>o</sub> l<sub>e</sub>t $S _ { k } ^ { * } ( \pmb { \vartheta } )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e se</sub>t <sub>o</sub>f i<sub>n</sub>di<sub>ces o</sub>f th<sub>e</sub> � l<sub>arges</sub>t <sup>com</sup>p<sup>onents</sup> <sup>of</sup> <sup>an</sup>y v<sup>ector</sup> $\pmb { \vartheta } \in \mathbb { R } ^ { I }$ <sub>.</sub> Aft<sub>er</sub> � <sub>roun</sub>d<sub>s</sub> <sub>o</sub>f <sub>eva</sub>l<sub>ua</sub>ti<sub>ons,</sub> th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>k<sub>er</sub> <sub>se</sub>l<sub>ec</sub>t<sub>s</sub> <sub>a</sub> <sub>se</sub>t <sub>o</sub>f it<sub>ems</sub> $S _ { k } \subset \{ 1 , \ldots , I \}$ <sub>w</sub>ith $| S _ { k } | = k$ <sub>an</sub>d th<sub>e</sub> EOC i<sub>s</sub>

$$
\operatorname { E O C } ( S _ { k } ; p _ { T } ) : = \mathbb { E } _ { \boldsymbol \theta } [ \sum _ { i \in S _ { k } ^ { * } ( \boldsymbol \theta ) } \theta _ { i } - \sum _ { i \in S _ { k } } \theta _ { i } | p _ { T } ] = \mathbb { E } _ { \boldsymbol \theta } [ \sum _ { i \in S _ { k } ^ { * } ( \boldsymbol \theta ) } \theta _ { i } | p _ { T } | - \sum _ { i \in S _ { k } } \mu _ { T , i } .\tag{1}
$$

Th<sub>roug</sub>h<sub>ou</sub>t th<sub>e</sub> <sub>paper,</sub> <sub>con</sub>diti<sub>on</sub>i<sub>ng</sub> <sub>on</sub> $p _ { t }$ means that the ex<sub>p</sub>ectation (or <sub>p</sub>robabilit<sub>y</sub>) is taken with res<sub>p</sub>ect to $\pmb \theta \sim p _ { t }$ . The first term in (1) does not de<sub>p</sub>end on the choice of $S _ { k }$ <sub>,</sub> <sub>so</sub> <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> EOC i<sub>s</sub> <sub>equ</sub>i<sub>va</sub>l<sub>en</sub>t t<sub>o</sub> maximizin<sub>g</sub> the sum of the selected set’s <sub>p</sub>osterior means (the second term). It follows that the o<sub>p</sub>timal set to <sub>se</sub>l<sub>ec</sub>t <sub>g</sub>i<sub>ven</sub> $\pmb { \mu } _ { T }$ i<sub>s</sub> th<sub>e</sub> <sub>se</sub>t <sub>o</sub>f � it<sub>ems</sub> <sub>w</sub>ith th<sub>e</sub> l<sub>arges</sub>t <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>means</sub> $S _ { k } ^ { * } ( \mu _ { T } )$

With the terminal objective established, we next formulate the sequential problem of minimizing it <sub>as</sub> <sub>a</sub> d<sub>ynam</sub>i<sub>c</sub> <sub>program.</sub> At <sub>any</sub> <sub>roun</sub>d � th<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> i<sub>s</sub> d<sub>escr</sub>ib<sub>e</sub>d b<sub>y</sub> $\left( { p _ { t - 1 } , R _ { t } } \right)$ <sub>.</sub> Fi<sub>rs</sub>t <sub>cons</sub>id<sub>er a s</sub>t<sub>a</sub>t<sub>e w</sub>h<sub>ere</sub> <sub>m</sub>i<sub>n</sub> $( r _ { 0 } , r _ { 1 } ) \le R _ { t } < 2 \operatorname* { m i n } ( r _ { 0 } , r _ { 1 } )$ <sub>, w</sub>hi<sub>c</sub>h i<sub>s a su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>con</sub>diti<sub>on</sub> f<sub>or</sub> $t = T$ <sub>.</sub> Th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>k<sub>er c</sub>h<sub>ooses an</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> $( i , j ) \in \mathcal { I }$ <sub>w</sub>ith $r _ { J } \leq R _ { t }$ , an outcome $X _ { t } = x$ i<sub>s</sub> <sub>o</sub>b<sub>serve</sub>d<sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> EOC i<sub>s</sub>

$$
\mathbb { E } _ { \theta } [ \sum _ { \ell \in S _ { k } ^ { * } ( \theta ) } \theta _ { \ell } | X _ { t } = x , ( i , j ) , p _ { t - 1 } | - \sum _ { \ell \in S _ { k } ^ { * } ( \mu _ { t } ( x | ( i , j ) ) ) } \mu _ { t , \ell } ( x \mid ( i , j ) ) ,\tag{2}
$$

<sub>w</sub>h<sub>ere</sub> $\pmb { \mu } _ { t } ( x \mid ( i , j ) ) = \mathbb { E } [ \pmb { \theta } \mid X _ { t } = x , ( i , j ) , p _ { t - 1 } ]$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore,</sub> th<sub>e</sub> <sub>pro</sub>bl<sub>em</sub> i<sub>s</sub> t<sub>o</sub> <sub>c</sub>h<sub>oose</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> $( i , j )$ th<sub>a</sub>t minimizes the ex<sub>p</sub>ectation of (2) over its outcome $X _ { t }$ <sub>.</sub> B<sub>y</sub> th<sub>e</sub> l<sub>aw o</sub>f t<sub>o</sub>t<sub>a</sub>l <sub>expec</sub>t<sub>a</sub>ti<sub>on,</sub> th<sub>e expec</sub>t<sub>a</sub>ti<sub>on o</sub>f th<sub>e</sub> fi<sub>rs</sub>t t<sub>erm</sub> <sub>equa</sub>l<sub>s</sub> $\mathbb { E } _ { \pmb { \theta } } [ \sum _ { \ell \in S _ { k } ^ { \ast } ( \pmb { \theta } ) } \theta _ { \ell } \ \big | \ p _ { t - 1 } ]$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h d<sub>oes</sub> <sub>no</sub>t d<sub>epen</sub>d <sub>on</sub> th<sub>e</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>o</sub>f $( i , j )$ <sub>.</sub> Th<sub>us,</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> i<sub>s</sub>

$$
( i _ { t } , j _ { t } ) \in \arg \operatorname* { m a x } _ { ( i , j ) \in { \cal I } : r _ { j } \le R _ { t } } \sum _ { x \in \{ - 1 , 1 \} } \mathbb { P } ( X _ { i j } = x \mid p _ { t - 1 } ) \sum _ { \ell \in S _ { k } ^ { * } ( \mu _ { t } ( x \mid ( i , j ) ) ) } \mu _ { t , \ell } ( x \mid ( i , j ) ) .
$$

T<sub>o cons</sub>id<sub>er</sub> th<sub>e</sub> f<sub>u</sub>ll <sub>sequen</sub>ti<sub>a</sub>l <sub>process,</sub> l<sub>e</sub>t $V ^ { * } ( p _ { t - 1 } , R _ { t } )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> l<sub>arges</sub>t <sub>expec</sub>t<sub>e</sub>d <sub>sum o</sub>f t<sub>erm</sub>i<sub>na</sub>l <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>means</sub> f<sub>rom</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $\left( { p _ { t - 1 } , R _ { t } } \right)$ <sub>.</sub> If <sub>no eva</sub>l<sub>ua</sub>ti<sub>on</sub> i<sub>s a</sub>f<sub>or</sub>d<sub>a</sub>bl<sub>e,</sub> th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>k<sub>er se</sub>l<sub>ec</sub>t<sub>s</sub> $S _ { k } ^ { * } ( \pmb { \mu } _ { t - 1 } )$ now, <sub>g</sub><sup>i</sup>v<sup>i</sup>n<sub>g</sub>

$$
V ^ { * } ( p _ { t - 1 } , R _ { t } ) = \sum _ { \ell \in S _ { k } ^ { * } ( \mu _ { t - 1 } ) } \mu _ { t - 1 , \ell } , \qquad R _ { t } < \mathrm { m i n } ( r _ { 0 } , r _ { 1 } ) .
$$

Oth<sub>erw</sub>i<sub>se,</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>va</sub>l<sub>ue</sub> f<sub>unc</sub>ti<sub>on sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e</sub> B<sub>e</sub>ll<sub>man equa</sub>ti<sub>on</sub>

$$
V ^ { * } ( p _ { t - 1 } , R _ { t } ) = \operatorname* { m a x } _ { ( i , j ) \in { \cal I } : r _ { j } \le R _ { t } } \sum _ { x \in \{ - 1 , 1 \} } \mathbb { P } \big ( X _ { i j } = x \big | p _ { t - 1 } \big ) \times V ^ { * } \big ( p _ { t } ( \cdot | x , ( i , j ) ) , R _ { t } - r _ { j } \big ) ,\tag{3}
$$

<sub>w</sub>h<sub>ere</sub> $p _ { t } ( \cdot | x , ( i , j ) )$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>o</sub>bt<sub>a</sub>i<sub>ne</sub>d <sub>a</sub>ft<sub>er</sub> <sub>s</sub>t<sub>ar</sub>ti<sub>ng</sub> f<sub>rom</sub> <sub>pr</sub>i<sub>or</sub> $p _ { t - 1 }$ <sub>,</sub> <sub>c</sub>h<sub>oos</sub>i<sub>ng</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> $( i , j )$ <sub>an</sub>d <sub>o</sub>b<sub>serv</sub>i<sub>ng</sub> <sub>ou</sub>t<sub>come</sub> <sub>�.</sub>

## 2.3. Shortcomings of the knowledge gradient policy

Addressin<sub>g</sub> the curse of dimensionalit<sub>y</sub> of the d<sub>y</sub>namic <sub>p</sub>ro<sub>g</sub>ram in (3), the KG <sub>p</sub>olic<sub>y</sub> ado<sub>p</sub>ts the m<sub>y</sub>o<sub>p</sub>ic <sub>vers</sub>i<sub>on</sub> <sub>o</sub>f it<sub>,</sub> <sub>g</sub>i<sub>ven</sub> b<sub>y</sub>

$$
V _ { \mathrm { m y o p i c } } ( ( i , j ) \mid p _ { t - 1 } ) : = \sum _ { x \in \{ - 1 , 1 \} } \mathbb { P } \big ( X _ { i j } = x \mid p _ { t - 1 } \big ) \left( \sum _ { \ell \in S _ { k } ^ { * } \left( \mu _ { t } \left( x \mid ( i , j ) \right) \right) } \mu _ { t , \ell } ( x \mid ( i , j ) ) - \sum _ { \ell \in S _ { k } ^ { * } \left( \mu _ { t - 1 } \right) } \mu _ { t - 1 , \ell } \right) .\tag{4}
$$

Th<sub>e</sub> KG <sub>po</sub>li<sub>cy</sub> th<sub>en</sub> <sub>se</sub>l<sub>ec</sub>t<sub>s</sub> <sub>an</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>accor</sub>di<sub>ng</sub> t<sub>o</sub>

$$
\arg \operatorname* { m a x } _ { ( i , j ) \in { \cal { I } } : \ r _ { J } \leq R _ { t } } \frac { V _ { \mathrm { m y o p i c } } ( ( i , j ) \mid p _ { t - 1 } ) } { r _ { J } } .
$$

However, binar<sub>y</sub> data can cause the KG <sub>p</sub>olic<sub>y</sub> to <sub>g</sub>et “stuck” (Li et al. 2024). If the estimated to<sub>p</sub>-� set is <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> <sub>separa</sub>t<sub>e</sub>d f<sub>rom</sub> th<sub>e</sub> <sub>res</sub>t<sub>,</sub> <sub>no</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>ou</sub>t<sub>come</sub> <sub>can</sub> di<sub>rec</sub>tl<sub>y</sub> <sub>c</sub>h<sub>ange</sub> th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> t<sub>op-</sub>� <sub>se</sub>t b<sub>ecause</sub> th<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> bi<sub>nary</sub> <sub>ou</sub>t<sub>come</sub> <sub>can</sub> <sub>carry</sub> i<sub>s</sub> b<sub>oun</sub>d<sub>e</sub>d<sub>,</sub> <sub>an</sub>d <sub>w</sub>ith it h<sub>ow</sub> f<sub>ar</sub> th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>means</sub> <sub>can</sub> <sub>move.</sub> Thi<sub>s can occur</sub> f<sub>requen</sub>tl<sub>y even w</sub>h<sub>en</sub> th<sub>e</sub> t<sub>op-</sub>� <sub>se</sub>t i<sub>s</sub> f<sub>ar</sub> f<sub>rom cer</sub>t<sub>a</sub>i<sub>n.</sub> Wh<sub>en</sub> thi<sub>s</sub> h<sub>appens,</sub> KG h<sub>as no</sub> b<sub>as</sub>i<sub>s</sub> t<sub>o</sub> <sub>c</sub>h<sub>oose</sub> <sub>among</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons,</sub> <sub>an</sub>d <sub>a</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>ru</sub>l<sub>e,</sub> <sub>suc</sub>h <sub>as</sub> <sub>ran</sub>d<sub>om</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>on,</sub> i<sub>s</sub> <sub>nee</sub>d<sub>e</sub>d<sub>.</sub> Thi<sub>s</sub> i<sub>s</sub> <sub>no</sub>t <sub>an</sub> i<sub>ssue</sub> <sub>w</sub>ith <sub>un</sub>b<sub>oun</sub>d<sub>e</sub>d <sub>o</sub>b<sub>serva</sub>ti<sub>ons,</sub> <sub>suc</sub>h <sub>as</sub> <sub>a</sub> G<sub>auss</sub>i<sub>an</sub> lik<sub>e</sub>lih<sub>oo</sub>d<sub>,</sub> <sub>w</sub>h<sub>ere</sub> th<sub>ere</sub> i<sub>s</sub> <sub>a</sub>l<sub>ways</sub> <sub>a</sub> <sub>poss</sub>ibilit<sub>y</sub> th<sub>a</sub>t <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>can</sub> <sub>prov</sub>id<sub>e</sub> <sub>overw</sub>h<sub>e</sub>l<sub>m</sub>i<sub>ng</sub>l<sub>y</sub> <sub>pos</sub>iti<sub>ve</sub>/<sub>nega</sub>ti<sub>ve</sub> <sub>ev</sub>id<sub>ence</sub> th<sub>a</sub>t <sub>c</sub>h<sub>anges</sub> th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> t<sub>op-</sub>� <sub>se</sub>t<sub>.</sub>

## 3. Surrogate objective

Developing the surrogate objective combines two ideas. The first is to establish the expected mismatch loss <sub>an</sub>d <sub>connec</sub>t it t<sub>o</sub> EOC<sub>.</sub> Th<sub>e secon</sub>d i<sub>s</sub> h<sub>ow</sub> it d<sub>r</sub>i<sub>ves eva</sub>l<sub>ua</sub>ti<sub>on c</sub>h<sub>o</sub>i<sub>ces.</sub> W<sub>e</sub> t<sub>rea</sub>t th<sub>e expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss as</sub> a static objective fixed at state $p _ { t - 1 }$ <sub>,</sub> <sub>us</sub>i<sub>ng</sub> it t<sub>o</sub> id<sub>en</sub>tif<sub>y</sub> <sub>w</sub>h<sub>ere</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> <sub>re</sub>d<sub>uc</sub>ti<sub>on</sub> i<sub>s</sub> <sub>mos</sub>t <sub>va</sub>l<sub>ua</sub>bl<sub>e</sub> h<sub>o</sub>ldi<sub>ng</sub> th<sub>e</sub> <sub>curren</sub>t <sub>means</sub> fi<sub>xe</sub>d<sub>.</sub> Th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>d<sub>er</sub> <sub>o</sub>f thi<sub>s</sub> <sub>sec</sub>ti<sub>on</sub> <sub>procee</sub>d<sub>s</sub> <sub>as</sub> f<sub>o</sub>ll<sub>ows.</sub> T<sub>o</sub> <sub>prov</sub>id<sub>e</sub> i<sub>ns</sub>i<sub>g</sub>ht i<sub>n</sub>t<sub>o</sub> h<sub>ow</sub> th<sub>e</sub> mismatch loss was develo<sub>p</sub>ed<sub>,</sub> §3.1 <sub>p</sub>resents our initial<sub>,</sub> sim<sub>p</sub>ler<sub>,</sub> intuition-based version<sub>,</sub> which then leads to the formal develo<sub>p</sub>ment in §3.2 that ri<sub>g</sub>orousl<sub>y</sub> connects the surro<sub>g</sub>ate to EOC and <sub>y</sub>ields an im<sub>p</sub>roved form. Finall<sub>y,</sub> §3.3 develo<sub>p</sub>s the resultin<sub>g</sub> surro<sub>g</sub>ate loss.

## 3.1. Mismatch loss: from intuition

R<sub>eca</sub>ll f<sub>rom</sub> th<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub> th<sub>a</sub>t <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>magn</sub>it<sub>u</sub>d<sub>e</sub> <sub>o</sub>f th<sub>e</sub> i<sub>nc</sub>l<sub>us</sub>i<sub>on</sub> <sub>errors</sub> <sub>o</sub>f $S _ { k } ^ { * } ( \pmb { \mu } _ { t - 1 } )$ <sub>an</sub>d th<sub>e exc</sub>l<sub>us</sub>i<sub>on errors o</sub>f $\{ 1 , \ldots , I \} \setminus S _ { k } ^ { * } ( \pmb { \mu } _ { t - 1 } )$ <sub>.</sub> M<sub>easur</sub>i<sub>ng</sub> th<sub>e</sub> <sub>magn</sub>it<sub>u</sub>d<sub>e</sub> <sub>o</sub>f <sub>eac</sub>h <sub>error</sub> <sub>requ</sub>i<sub>res</sub> <sub>an</sub> <sub>anc</sub>h<sub>or,</sub> <sub>a</sub> t<sub>op-</sub>� th<sub>res</sub>h<sub>o</sub>ld <sub>aga</sub>i<sub>ns</sub>t <sub>w</sub>hi<sub>c</sub>h th<sub>ey</sub> <sub>are</sub> <sub>measure</sub>d<sub>.</sub> H<sub>owever,</sub> <sub>g</sub>i<sub>ven</sub> th<sub>e</sub> <sub>pr</sub>i<sub>or,</sub> th<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f it<sub>ems</sub> <sub>a</sub>b<sub>ove a g</sub>i<sub>ven</sub> th<sub>res</sub>h<sub>o</sub>ld i<sub>s a ran</sub>d<sub>om var</sub>i<sub>a</sub>bl<sub>e.</sub> A <sub>na</sub>t<sub>ura</sub>l <sub>approac</sub>h i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> t<sub>o c</sub>h<sub>oose a</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub> i<sub>n</sub> <sub>expec</sub>t<sub>a</sub>ti<sub>on,</sub> th<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f it<sub>ems</sub> <sub>a</sub>b<sub>ove</sub> it i<sub>s</sub> �<sub>.</sub>

M<sub>ore exp</sub>li<sub>c</sub>itl<sub>y, a</sub>t <sub>eac</sub>h <sub>eva</sub>l<sub>ua</sub>ti<sub>on roun</sub>d �<sub>,</sub> fi<sub>x a</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>� on</sub> th<sub>e u</sub>tilit<sub>y sca</sub>l<sub>e suc</sub>h th<sub>a</sub>t<sub>,</sub> i<sub>n expec</sub>t<sub>a</sub>ti<sub>on,</sub> � it<sub>ems</sub> h<sub>ave u</sub>tilit<sub>y a</sub>b<sub>ove � un</sub>d<sub>er</sub> th<sub>e pr</sub>i<sub>or</sub> $p _ { t - 1 }$ <sub>.</sub> Th<sub>en</sub> f<sub>or eac</sub>h it<sub>em</sub> $i \in S _ { k } ^ { * } ( \mu _ { t - 1 } )$ th<sub>e</sub> l<sub>oss</sub> i<sub>s</sub> E $\left[ q - \theta _ { i } \right) ^ { + } ]$ , th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>amoun</sub>t it f<sub>a</sub>ll<sub>s</sub> <sub>s</sub>h<sub>or</sub>t <sub>o</sub>f th<sub>e</sub> t<sub>op-</sub>� th<sub>res</sub>h<sub>o</sub>ld<sub>,</sub> <sub>an</sub>d <sub>s</sub>i<sub>m</sub>il<sub>ar</sub>l<sub>y</sub> f<sub>or</sub> <sub>eac</sub>h it<sub>em</sub> $i \notin S _ { k } ^ { * } ( \mu _ { t - 1 } )$ th<sub>e</sub> l<sub>oss</sub> i<sub>s</sub> $\mathbb { E } [ ( \theta _ { i } - q ) ^ { + } ]$ <sub>,</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>amoun</sub>t it <sub>excee</sub>d<sub>s</sub> th<sub>e</sub> t<sub>op-</sub>� th<sub>res</sub>h<sub>o</sub>ld<sub>.</sub> W<sub>e</sub> <sub>re</sub>f<sub>er</sub> t<sub>o</sub> th<sub>e</sub> <sub>sum</sub> <sub>o</sub>f th<sub>ese</sub> l<sub>osses</sub> <sub>as</sub> th<sub>e</sub> (informal) expected mismatch loss

$$
\sum _ { i \in S _ { k } ^ { * } ( \mu _ { t - 1 } ) } \mathbb { E } \big [ ( q - \theta _ { i } ) ^ { + } \big | p _ { t - 1 } \big ] + \sum _ { i \notin S _ { k } ^ { * } ( \mu _ { t - 1 } ) } \mathbb { E } \big [ ( \theta _ { i } - q ) ^ { + } \big | p _ { t - 1 } \big ] .\tag{5}
$$

The policy then chooses the evaluation whose resulting uncertainty reduction on � reduces expected mismatch l<sub>oss</sub> th<sub>e mos</sub>t <sub>per un</sub>it <sub>o</sub>f <sub>resource.</sub> T<sub>wo</sub> thi<sub>ngs ma</sub>k<sub>e</sub> thi<sub>s concep</sub>t<sub>ua</sub>ll<sub>y a</sub>tt<sub>rac</sub>ti<sub>ve.</sub> Fi<sub>rs</sub>t<sub>,</sub> it <sub>na</sub>t<sub>ura</sub>ll<sub>y</sub> di<sub>rec</sub>t<sub>s</sub> evaluations toward items near the current ex<sub>p</sub>ected to<sub>p</sub>-� cutof (<sub>�</sub>) and toward items with substantial <sub>p</sub>rior <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y,</sub> <sub>s</sub>i<sub>nce</sub> th<sub>ose</sub> <sub>are</sub> th<sub>e</sub> it<sub>ems</sub> th<sub>a</sub>t<sub>,</sub> if <sub>m</sub>i<sub>sc</sub>l<sub>ass</sub>ifi<sub>e</sub>d<sub>,</sub> <sub>w</sub>ill i<sub>ncur</sub> <sub>a</sub> hi<sub>g</sub>h <sub>expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss.</sub> H<sub>ence,</sub> thi<sub>s</sub> <sub>surroga</sub>t<sub>e</sub> <sub>preserves</sub> th<sub>e</sub> k<sub>ey</sub> f<sub>ea</sub>t<sub>ure</sub> <sub>o</sub>f EOC<sub>:</sub> <sub>m</sub>i<sub>s</sub>t<sub>a</sub>k<sub>es</sub> <sub>are</sub> <sub>pena</sub>li<sub>ze</sub>d b<sub>y</sub> th<sub>e</sub>i<sub>r</sub> <sub>magn</sub>it<sub>u</sub>d<sub>e.</sub> S<sub>econ</sub>d<sub>,</sub> <sub>an</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on can s</sub>till b<sub>e va</sub>l<sub>ua</sub>bl<sub>e even w</sub>h<sub>en</sub> it<sub>s ou</sub>t<sub>come canno</sub>t <sub>c</sub>h<sub>ange</sub> thi<sub>s roun</sub>d’<sub>s pos</sub>t<sub>er</sub>i<sub>or</sub> t<sub>op-</sub>� <sub>mem</sub>b<sub>ers</sub>hi<sub>p</sub> if it ti<sub>g</sub>ht<sub>ens</sub> th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>o</sub>f it<sub>ems</sub> <sub>near</sub> $q .$ <sub>.</sub> Thi<sub>s</sub> i<sub>s</sub> i<sub>n</sub> <sub>par</sub>t b<sub>ecause</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> i<sub>n</sub>f<sub>orms</sub> <sub>w</sub>h<sub>ere</sub> t<sub>o</sub> di<sub>rec</sub>t l<sub>a</sub>t<sub>er</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons.</sub> Att<sub>emp</sub>ti<sub>ng</sub> <sub>a</sub> <sub>spor</sub>t<sub>s</sub> <sub>ana</sub>l<sub>ogy:</sub> <sub>we</sub> d<sub>on</sub>’t <sub>on</sub>l<sub>y</sub> <sub>g</sub>i<sub>ve</sub> <sub>cre</sub>dit f<sub>or</sub> <sub>scor</sub>i<sub>ng</sub> <sub>goa</sub>l<sub>s</sub> b<sub>u</sub>t <sub>a</sub>l<sub>so</sub> f<sub>or</sub> <sub>ass</sub>i<sub>s</sub>t<sub>s.</sub>

H<sub>owever,</sub> <sub>a</sub>d<sub>ap</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>com</sub>bi<sub>n</sub>i<sub>ng</sub> <sub>ra</sub>ti<sub>ngs</sub> <sub>an</sub>d <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>compar</sub>i<sub>sons</sub> <sub>us</sub>i<sub>ng</sub> thi<sub>s</sub> <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> h<sub>as</sub> <sub>a</sub> l<sub>ess</sub> th<sub>an</sub> <sub>o</sub>b<sub>v</sub>i<sub>ous</sub> <sub>nuance</sub> t<sub>o</sub> it<sub>.</sub> Whi<sub>c</sub>h it<sub>ems</sub> b<sub>e</sub>l<sub>ong</sub> i<sub>n</sub> th<sub>e</sub> t<sub>op-</sub>� d<sub>epen</sub>d<sub>s</sub> <sub>on</sub>l<sub>y</sub> <sub>on</sub> th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>u</sub>tiliti<sub>es,</sub> <sub>no</sub>t <sub>on</sub> th<sub>e</sub> <sub>average</sub> <sub>u</sub>tilit<sub>y</sub> ${ \overline { { \theta } } } .$ Th<sub>us,</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>a</sub>b<sub>ou</sub>t <sub>a</sub> <sub>c</sub>h<sub>ange</sub> i<sub>n</sub> $\overline { { \theta } }$ that shifts all utilities equally is decision-irrelevant, and the surro<sub>g</sub>ate should catch this. However, (5) ma<sub>y</sub> reward learnin<sub>g</sub> such information, since reducin<sub>g</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> <sub>a</sub>b<sub>ou</sub>t <sub>suc</sub>h <sub>a</sub> <sub>common</sub> <sub>s</sub>hift <sub>re</sub>d<sub>uces</sub> th<sub>e</sub> <sub>expec</sub>t<sub>a</sub>ti<sub>ons</sub> $\mathbb { E } [ ( q - \theta _ { i } ) ^ { + } ] , \mathbb { E } [ ( \theta _ { i } - q ) ^ { + } ]$ <sub>.</sub> R<sub>a</sub>tin<sub>gs</sub> <sub>ca</sub>n <sub>prov</sub>id<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>a</sub>b<sub>ou</sub>t $\overline { { \theta } }$ <sub>s</sub>i<sub>nce</sub> th<sub>ey</sub> <sub>con</sub>t<sub>a</sub>i<sub>n</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>a</sub>b<sub>ou</sub>t <sub>an</sub> it<sub>em</sub>’<sub>s</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> <sub>qua</sub>lit<sub>y,</sub> <sub>w</sub>hil<sub>e</sub> <sub>compar</sub>i<sub>sons</sub> <sub>canno</sub>t <sub>s</sub>i<sub>nce</sub> th<sub>ey on</sub>l<sub>y con</sub>t<sub>a</sub>i<sub>n re</sub>l<sub>a</sub>ti<sub>ve</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on.</sub> A <sub>surroga</sub>t<sub>e</sub> b<sub>ase</sub>d <sub>on</sub> thi<sub>s m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss can</sub> th<sub>ere</sub>f<sub>ore</sub> f<sub>avor ra</sub>ti<sub>ngs over compar</sub>i<sub>sons</sub> b<sub>ecause</sub> it <sub>rewar</sub>d<sub>s</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-</sub>i<sub>rre</sub>l<sub>evan</sub>t <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y re</sub>d<sub>uc</sub>ti<sub>on.</sub>

I<sub>n</sub>iti<sub>a</sub>ll<sub>y</sub> <sub>we</sub> <sub>a</sub>dd<sub>resse</sub>d thi<sub>s</sub> b<sub>y</sub> <sub>rep</sub>l<sub>ac</sub>i<sub>ng</sub> $\pmb \theta$ <sub>w</sub>ith it<sub>s</sub> <sub>sum-</sub>t<sub>o-zero</sub> <sub>coun</sub>t<sub>erpar</sub>t $\widetilde { \pmb { \theta } } = \pmb { \theta } - \overline { { \theta } } \mathbf { 1 }$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> 1 i<sub>s</sub> th<sub>e</sub> <sub>a</sub>ll<sub>-ones vec</sub>t<sub>or, an</sub>d <sub>compu</sub>ti<sub>ng</sub> th<sub>e m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> b<sub>ase</sub>d <sub>on</sub> th<sub>e</sub> dif<sub>erences</sub> $\widetilde { \theta } _ { i } - q$ <sub>an</sub>d $q - \widetilde { \theta } _ { i }$ <sub>,</sub> <sub>so</sub> th<sub>a</sub>t l<sub>earn</sub>i<sub>ng</sub> $\overline { { \theta } }$ d<sub>oes</sub> <sub>no</sub>t <sub>a</sub>f<sub>ec</sub>t th<sub>e</sub> l<sub>oss.</sub> W<sub>e</sub> <sub>w</sub>ill <sub>nex</sub>t <sub>presen</sub>t <sub>a</sub> f<sub>orma</sub>l <sub>an</sub>d <sub>more</sub> <sub>genera</sub>l t<sub>rea</sub>t<sub>men</sub>t <sub>o</sub>f <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> it<sub>s</sub> li<sub>n</sub>k t<sub>o</sub> EOC<sub>.</sub> A<sub>s a mo</sub>ti<sub>va</sub>ti<sub>on</sub> f<sub>or</sub> thi<sub>s, no</sub>ti<sub>ce</sub> th<sub>a</sub>t $\widetilde { \theta _ { i } } - q = \theta _ { i } - \left( q + \overline { { \theta } } \right)$ <sub>, so</sub> th<sub>e m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss w</sub>ith th<sub>e</sub> fi<sub>xe</sub>d th<sub>res</sub>h<sub>o</sub>ld $q$ <sub>app</sub>li<sub>e</sub>d t<sub>o</sub> $\widetilde { \pmb { \theta } }$ can be thought of as our original mismatch loss with a random �-dependent th<sub>res</sub>h<sub>o</sub>ld $q + { \overline { { \theta } } } .$ . This observation invites the question of what is the “best” such random top-� threshold that <sub>a</sub>ll<sub>ows</sub> f<sub>or</sub> t<sub>rac</sub>t<sub>a</sub>bl<sub>e</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s.</sub>

## 3.2. Mismatch loss: formal derivation

W<sub>e</sub> b<sub>eg</sub>i<sub>n</sub> b<sub>y</sub> f<sub>orma</sub>ll<sub>y</sub> d<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub> <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> <sub>an</sub>d <sub>s</sub>h<sub>ow</sub>i<sub>ng</sub> h<sub>ow</sub> it <sub>re</sub>l<sub>a</sub>t<sub>es</sub> t<sub>o</sub> EOC<sub>.</sub> F<sub>or</sub> <sub>any</sub> <sub>u</sub>tilit<sub>y</sub> <sub>vec</sub>t<sub>or</sub> $\pmb \theta$ <sub>an</sub>d <sub>any</sub> th<sub>res</sub>h<sub>o</sub>ld f<sub>unc</sub>ti<sub>on</sub> $a : \mathbb { R } ^ { I } \to \mathbb { R }$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub>

$$
\mathcal { M } ( \theta ; S _ { k } , a ) : = \sum _ { i \in S _ { k } } \left( a ( \theta ) - \theta _ { i } \right) ^ { + } + \sum _ { i \notin S _ { k } } \left( \theta _ { i } - a ( \theta ) \right) ^ { + } = k a ( \theta ) - \sum _ { i \in S _ { k } } \theta _ { i } + \sum _ { i = 1 } ^ { I } \left( \theta _ { i } - a ( \theta ) \right) ^ { + } ,\tag{6}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e secon</sub>d <sub>equa</sub>lit<sub>y uses</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y</sub> $x ^ { + } = x + ( - x ) ^ { + }$ <sub>.</sub> L<sub>emma</sub> 1 <sub>says</sub> th<sub>a</sub>t <sub>expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> i<sub>s an</sub> <sub>upper</sub> b<sub>oun</sub>d <sub>on</sub> EOC <sub>an</sub>d th<sub>a</sub>t f<sub>or</sub> <sub>spec</sub>ifi<sub>c</sub> th<sub>res</sub>h<sub>o</sub>ld f<sub>unc</sub>ti<sub>ons</sub> it <sub>equa</sub>l<sub>s</sub> EOC<sub>.</sub>

LEMMA 1. Let $\theta ^ { ( k ) } , \theta ^ { ( k + 1 ) }$ denote the �th and (�+1)th order statistics of � respectively.

(i) For any thresholdfunction �, $\mathrm { E O C } ( S _ { k } ; p _ { t - 1 } ) \leq \mathbb { E } \big [ M ( \pmb { \theta } ; S _ { k } , a ) \big | p _ { t - 1 } \big ] .$

(ii) $I f a ( \pmb \theta ) \in [ \theta ^ { ( k + 1 ) } , \theta ^ { ( k ) } ]$ almost surely, then $\operatorname { E O C } ( S _ { k } ; p _ { t - 1 } ) = \mathbb { E } \left[ \mathcal { M } ( \pmb { \theta } ; S _ { k } , a ) \left| p _ { t - 1 } \right. \right]$

Proof. All proofs are provided in the Appendix.

L<sub>emma</sub> 1 <sub>app</sub>li<sub>es</sub> t<sub>o an ar</sub>bit<sub>rary s</sub>t<sub>a</sub>t<sub>e</sub> $p _ { t - 1 }$ b<sub>u</sub>t d<sub>oes no</sub>t i<sub>nvo</sub>l<sub>ve</sub> f<sub>u</sub>t<sub>ure eva</sub>l<sub>ua</sub>ti<sub>ons.</sub> Th<sub>us,</sub> it<sub>s use</sub>f<sub>u</sub>l<sub>ness</sub> i<sub>s</sub> i<sub>n</sub> <sub>ensur</sub>i<sub>ng</sub> th<sub>a</sub>t<sub>,</sub> <sub>over</sub> th<sub>e</sub> <sub>course</sub> <sub>o</sub>f th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>roun</sub>d<sub>s,</sub> if <sub>we</sub> <sub>con</sub>d<sub>uc</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>ma</sub>k<sub>e</sub> <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss sma</sub>ll<sub>,</sub> th<sub>en</sub> th<sub>e resu</sub>lti<sub>ng</sub> EOC <sub>w</sub>ill <sub>a</sub>l<sub>so</sub> b<sub>e sma</sub>ll<sub>.</sub> Th<sub>e</sub> b<sub>oun</sub>d h<sub>o</sub>ld<sub>s w</sub>ith <sub>equa</sub>lit<sub>y,</sub> f<sub>or examp</sub>l<sub>e,</sub> f<sub>or</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld $\begin{array} { r } { a ( \theta ) = \frac { 1 } { 2 } \big ( \theta ^ { ( k ) } + \theta ^ { ( k + 1 ) } \big ) } \end{array}$ <sub>.</sub> H<sub>owever,</sub> t<sub>a</sub>ki<sub>ng</sub> th<sub>e expec</sub>t<sub>a</sub>ti<sub>on o</sub>f <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> i<sub>s no</sub>t t<sub>rac</sub>t<sub>a</sub>bl<sub>e w</sub>h<sub>en</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld i<sub>s a</sub> f<sub>unc</sub>ti<sub>on o</sub>f <sub>suc</sub>h <sub>or</sub>d<sub>er s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore res</sub>t<sub>r</sub>i<sub>c</sub>t <sub>�</sub> t<sub>o a</sub> f<sub>am</sub>il<sub>y o</sub>f f<sub>unc</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>a</sub>ll<sub>ows</sub> f<sub>or</sub> t<sub>rac</sub>t<sub>a</sub>bl<sub>e ana</sub>l<sub>ys</sub>i<sub>s an</sub>d fi<sub>n</sub>d<sub>, w</sub>ithi<sub>n</sub> th<sub>a</sub>t f<sub>am</sub>il<sub>y,</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld f<sub>unc</sub>ti<sub>on</sub> th<sub>a</sub>t <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> ti<sub>g</sub>ht<sub>es</sub>t b<sub>oun</sub>d <sub>on</sub> EOC<sub>.</sub>

Th<sub>e expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> $\mathbb { E } [ M ( \pmb { \theta } ; S _ { k } , a ) \mid p _ { t - 1 } ]$ h<sub>as</sub> t<sub>erms o</sub>f th<sub>e</sub> f<sub>orm</sub> $\mathbb { E } [ ( \theta _ { i } - a ( { \pmb { \theta } } ) ) ^ { + } \mid p _ { t - 1 } ]$ <sub>.</sub> Si<sub>nce</sub> $\pmb { \theta } \mid p _ { t - 1 }$ i<sub>s</sub> G<sub>auss</sub>i<sub>an,</sub> th<sub>e</sub> dif<sub>erence</sub> $\theta _ { i } - a ( \theta )$ is Gaussian whenever the threshold is afine in �, and the terms $\mathbb { E } [ ( \theta _ { i } - a ( \pmb { \theta } ) ) ^ { + } | p _ { t - 1 } ]$ th<sub>en</sub> h<sub>ave</sub> <sub>express</sub>i<sub>ons</sub> <sub>amena</sub>bl<sub>e</sub> t<sub>o</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s.</sub> Thi<sub>s</sub> <sub>mo</sub>ti<sub>va</sub>t<sub>es</sub> th<sub>e</sub> <sub>a</sub>fi<sub>ne</sub> f<sub>am</sub>il<sub>y</sub> <sub>o</sub>f th<sub>res</sub>h<sub>o</sub>ld<sub>s</sub>

$$
a _ { q , b } ( \pmb \theta ) : = q + \pmb b ^ { \top } ( \pmb \theta - \pmb \mu _ { t - 1 } ) , \qquad q \in \mathbb { R } , \pmb b \in \mathbb { R } ^ { I } ,
$$

<sub>w</sub>h<sub>ere</sub> $q$ is the expected threshold level, and � governs how the threshold moves with the utilities. The dif<sub>erence</sub> $\theta _ { i } - a _ { q , b } ( \pmb \theta )$ i<sub>s</sub> th<sub>en</sub> $( \pmb { \mu } _ { t - 1 , i } - q ) + ( \pmb { e } _ { i } - \pmb { b } ) ^ { \top } ( \pmb { \theta } - \pmb { \mu } _ { t - 1 } )$ <sub>,</sub> a constant mean term $\mu _ { t - 1 , i } - q$ <sub>p</sub><sup>l</sup>us a <sub>mean-zero</sub> G<sub>auss</sub>i<sub>an</sub> t<sub>erm.</sub> B<sub>y</sub> L<sub>emma</sub> 1 <sub>any</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>o</sub>f $( q , b )$ <sub>g</sub>i<sub>ves</sub> <sub>a</sub>n <sub>uppe</sub>r b<sub>ou</sub>nd f<sub>o</sub>r EOC<sub>,</sub> <sub>a</sub>nd <sub>a</sub> <sub>pa</sub>ir $\left( q _ { t } ^ { * } , b _ { t } ^ { * } \right)$ th<sub>a</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub> th<sub>e expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss g</sub>i<sub>ves</sub> th<sub>e</sub> ti<sub>g</sub>ht<sub>es</sub>t <sub>upper</sub> b<sub>oun</sub>d f<sub>or</sub> EOC <sub>w</sub>ithi<sub>n</sub> th<sub>e a</sub>fi<sub>ne</sub> th<sub>res</sub>h<sub>o</sub>ld f<sub>am</sub>il<sub>y.</sub> B<sub>e</sub>f<sub>ore</sub> <sub>g</sub>i<sub>v</sub>i<sub>ng</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> <sub>con</sub>diti<sub>ons</sub> f<sub>or</sub> $( q _ { t } ^ { * } , b _ { t } ^ { * } )$ i<sub>n</sub> Th<sub>eorem</sub> 1<sub>,</sub> it i<sub>s conven</sub>i<sub>en</sub>t t<sub>o</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng.</sub> L<sub>e</sub>t $\kappa _ { q , b } ( \pmb \theta ) : = \left| \{ i : \theta _ { i } > a _ { q , b } ( \pmb \theta ) \} \right| - k$ denote the (random) deviation between the number of it<sub>ems a</sub>b<sub>ove</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>an</sub>d it<sub>s</sub> t<sub>arge</sub>t �<sub>.</sub> Th<sub>a</sub>t i<sub>s,</sub> $k + \kappa _ { q , b } ( \pmb { \theta } )$ i<sub>s</sub> th<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f it<sub>ems</sub> th<sub>a</sub>t <sub>excee</sub>d th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>.</sub>

THEOREM 1. The expected mismatch loss $\mathbb { E } [ \boldsymbol { \mathcal { M } } ( \pmb { \theta } ; S _ { k } , a _ { q , b } ) \mid p _ { t - 1 } ]$ is convex in $( q , b )$ and attains a minimum. A pair $\left( q _ { t } ^ { * } , b _ { t } ^ { * } \right)$ is a minimizer only ifeither

(i) $( q _ { t } ^ { * } , \pmb { b } _ { t } ^ { * } ) = ( \mu _ { t - 1 , j } , \pmb { e } _ { j } )$ for some $j = 1 , \ldots , I , o r$

(ii) $( q _ { t } ^ { * } , b _ { t } ^ { * } ) \neq ( \mu _ { t - 1 , j } , e _ { j } )$ for every $j = 1 , \ldots , I ,$ and

$$
\mathbb { E } [ \kappa _ { q _ { t } ^ { * } , b _ { t } ^ { * } } ( \theta ) | p _ { t - 1 } ] = 0 \qquad a n d \qquad \mathrm { C o v } \big ( \theta _ { \ell } , \kappa _ { q _ { t } ^ { * } , b _ { t } ^ { * } } ( \theta ) | p _ { t - 1 }  \big ) = 0 \quad f o r e v e r y \ell = 1 , \dots , I .
$$

Conversely, any pair satisfying (ii) is a minimizer.

Case (i) arises when $\pmb { b } = \pmb { e } _ { j }$ , so $\theta _ { j } - a _ { q , b } ( \theta ) = \mu _ { t - 1 , j } - q$ <sub>an</sub>d $\mathbb { E } [ ( \theta _ { j } - a _ { q , b } ( \pmb { \theta } ) ) ^ { + } \mid p _ { t - 1 } ] = ( \mu _ { t - 1 , j } - q ) ^ { + }$ i<sub>s</sub> <sub>no</sub>t dif<sub>eren</sub>ti<sub>a</sub>bl<sub>e a</sub>t $q = \mu _ { t - 1 , j }$ <sub>.</sub> I<sub>n</sub> th<sub>e</sub> A<sub>ppen</sub>di<sub>x, we</sub> d<sub>er</sub>i<sub>ve</sub> th<sub>e necessary an</sub>d <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>con</sub>diti<sub>on</sub> f<sub>or w</sub>h<sub>en suc</sub>h an exce<sub>p</sub>tional <sub>p</sub>oint would be o<sub>p</sub>timal. Otherwise, an<sub>y</sub> o<sub>p</sub>timum is characterized b<sub>y</sub> (ii), which will ca<sub>p</sub>ture most <sub>p</sub>ractical cases. We henceforth refer to an<sub>y</sub> minimizer satisf<sub>y</sub>in<sub>g</sub> (ii) as an o<sub>p</sub>timal afine threshold and f<sub>ocus our</sub> di<sub>scuss</sub>i<sub>on an</sub>d i<sub>n</sub>t<sub>erpre</sub>t<sub>a</sub>ti<sub>on on</sub> thi<sub>s case.</sub>

Th<sub>eorem</sub> 1 <sub>says</sub> th<sub>a</sub>t th<sub>e op</sub>ti<sub>ma</sub>l <sub>a</sub>fi<sub>ne</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> t<sub>wo proper</sub>ti<sub>es.</sub> Fi<sub>rs</sub>t<sub>,</sub> i<sub>n expec</sub>t<sub>a</sub>ti<sub>on,</sub> � it<sub>ems w</sub>ill h<sub>ave</sub> th<sub>e</sub>i<sub>r u</sub>tilit<sub>y a</sub>b<sub>ove</sub> it<sub>.</sub> S<sub>econ</sub>d<sub>,</sub> th<sub>e covar</sub>i<sub>ance</sub> b<sub>e</sub>t<sub>ween eac</sub>h it<sub>em</sub>’<sub>s u</sub>tilit<sub>y an</sub>d th<sub>e</sub> d<sub>ev</sub>i<sub>a</sub>ti<sub>on</sub> $\kappa _ { q , b } ( \pmb \theta )$ i<sub>s</sub> 0<sub>.</sub> If $\mathbf { \Delta } \kappa _ { q , b }$ were <sub>p</sub>ositivel<sub>y</sub> (ne<sub>g</sub>ativel<sub>y</sub>) correlated with $\theta _ { \ell }$ <sub>,</sub> th<sub>en</sub> <sub>w</sub>h<sub>en</sub> it<sub>em</sub> ℓ’<sub>s</sub> <sub>u</sub>tilit<sub>y</sub> i<sub>s</sub> l<sub>arger</sub> th<sub>an</sub> ex<sub>p</sub>ected, in ex<sub>p</sub>ectation more (fewer) than � items would be above the threshold, meanin<sub>g</sub> that the threshold is not moving up enough (is moving up too much) with item ℓ. At the optimum, no such linear dependence <sub>rema</sub>i<sub>ns:</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>moves</sub> <sub>w</sub>ith $\pmb \theta$ i<sub>n</sub> <sub>a</sub> <sub>way</sub> <sub>suc</sub>h th<sub>a</sub>t k<sub>now</sub>i<sub>ng</sub> th<sub>a</sub>t <sub>any</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar</sub> it<sub>em</sub> i<sub>s</sub> hi<sub>g</sub>h <sub>or</sub> l<sub>ow</sub> <sub>g</sub>i<sub>ves</sub> <sub>no</sub> fi<sub>rs</sub>t<sub>-or</sub>d<sub>er</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on a</sub>b<sub>ou</sub>t <sub>w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld h<sub>as overs</sub>h<sub>o</sub>t <sub>or un</sub>d<sub>ers</sub>h<sub>o</sub>t<sub>.</sub> N<sub>ex</sub>t<sub>,</sub> C<sub>oro</sub>ll<sub>ary</sub> 1 <sub>g</sub>i<sub>ves an</sub> <sub>exp</sub>li<sub>c</sub>it fi<sub>xe</sub>d<sub>-po</sub>i<sub>n</sub>t <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>za</sub>ti<sub>on</sub> <sub>o</sub>f $( q _ { t } ^ { * } , b _ { t } ^ { * } )$ i<sub>n</sub> <sub>our</sub> G<sub>auss</sub>i<sub>an</sub> <sub>se</sub>tti<sub>ng,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>uses</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>no</sub>t<sub>a</sub>ti<sub>on.</sub>

$$
\begin{array} { r } { \sigma _ { t - 1 , i } ^ { 2 } : = \mathbb { V } \left( \theta _ { i } - a _ { q , b } ( \pmb { \theta } ) \Big | p _ { t - 1 } \right) = ( e _ { i } - \pmb { b } ) ^ { \top } \Sigma _ { t - 1 } ( \pmb { e } _ { i } - \pmb { b } ) , } \end{array}\tag{7}
$$

$$
\begin{array} { r } { \sigma _ { t - 1 , i j } ^ { 2 } : = \mathbb { V } \left( \theta _ { i } - \theta _ { j } \middle | p _ { t - 1 } \right) = ( e _ { i } - e _ { j } ) ^ { \top } \Sigma _ { t - 1 } ( e _ { i } - e _ { j } ) , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $\begin{array} { r } { \pmb { e } _ { 0 } = \pmb { 0 } , } \end{array}$ , so $\sigma _ { t - 1 , i 0 } ^ { 2 } = \Sigma _ { t - 1 , i i }$ i<sub>s</sub> th<sub>e var</sub>i<sub>ance o</sub>f it<sub>em</sub> �<sub>.</sub> W<sub>e use</sub> thi<sub>s no</sub>t<sub>a</sub>ti<sub>on</sub> b<sub>ecause, go</sub>i<sub>ng</sub> f<sub>orwar</sub>d<sub>, we</sub> <sub>w</sub>ill <sub>care more a</sub>b<sub>ou</sub>t th<sub>e var</sub>i<sub>ance o</sub>f it<sub>ems re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld $\sigma _ { t - 1 , i } ^ { 2 }$ th<sub>an a</sub>b<sub>ou</sub>t <sub>an</sub> it<sub>em</sub>’<sub>s var</sub>i<sub>ance</sub> $\sigma _ { t - 1 , i 0 } ^ { 2 } .$

COROLLARY 1. Define $\begin{array} { r } { z _ { t - 1 , i } : = \frac { \mu _ { t - 1 , i } - q } { \sigma _ { t - 1 , i } } } \end{array}$ and $w _ { t - 1 , i } : = \frac { \phi \big ( z _ { t - 1 , i } \big ) } { \sigma _ { t - 1 , i } }$

(i) The optimality conditions ofTheorem 1(ii) are satisfied by $( q _ { t } ^ { * } , b _ { t } ^ { * } )$ such that

$$
\sum _ { i = 1 } ^ { I } \Phi ( z _ { t - 1 , i } ) = k , \qquad b _ { t , i } ^ { * } = \frac { w _ { t - 1 , i } } { \sum _ { \ell = 1 } ^ { I } w _ { t - 1 , \ell } } \quad f o r i = 1 , \dots , I .
$$

(ii) If the threshold is restricted to be constant $( i . e . , \pmb { b } = \mathbf { 0 } )$ , so that $a _ { q , \mathbf { 0 } } ( \pmb { \theta } ) = q$ , the optimal threshold ${ q } _ { t } ^ { c o n }$ satisfies

$$
\sum _ { i = 1 } ^ { I } \Phi \Bigg ( \frac { \mu _ { t - 1 , i } - q _ { t } ^ { c o n } } { \sigma _ { t - 1 , i 0 } } \Bigg ) = k .
$$

T<sub>o</sub> li<sub>g</sub>ht<sub>en no</sub>t<sub>a</sub>ti<sub>on, we wr</sub>it<sub>e</sub> $\left( q _ { t } , b _ { t } \right)$ f<sub>or</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>pa</sub>i<sub>r</sub> $( q _ { t } ^ { * } , b _ { t } ^ { * } )$ <sub>.</sub> E<sub>ac</sub>h $b _ { t , i }$ d<sub>escr</sub>ib<sub>es</sub> h<sub>ow</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>respon</sub>d<sub>s</sub> t<sub>o</sub> <sub>movemen</sub>t<sub>s</sub> i<sub>n</sub> it<sub>em</sub> $i \ ' \mathrm { s }$ <sub>u</sub>tilit<sub>y.</sub> Si<sub>nce</sub> $b _ { t , i } \propto w _ { t - 1 , i } = \phi ( z _ { t - 1 , i } ) / \sigma _ { t - 1 , i } ,$ <sub>,</sub> thi<sub>s</sub> <sub>response</sub> i<sub>s</sub> <sub>concen</sub>t<sub>ra</sub>t<sub>e</sub>d <sub>on</sub> it<sub>ems</sub> lik<sub>e</sub>l<sub>y near</sub> th<sub>e</sub> t<sub>op-</sub>� b<sub>oun</sub>d<sub>ary.</sub> I<sub>n</sub> thi<sub>s sense,</sub> th<sub>e op</sub>ti<sub>ma</sub>l th<sub>res</sub>h<sub>o</sub>ld <sub>can</sub> b<sub>e v</sub>i<sub>ewe</sub>d <sub>as a</sub> fi<sub>rs</sub>t<sub>-or</sub>d<sub>er</sub> <sub>proxy</sub> f<sub>or</sub> th<sub>e</sub> <sub>ran</sub>d<sub>om</sub> t<sub>op-</sub>� th<sub>res</sub>h<sub>o</sub>ld<sub>:</sub> it <sub>moves</sub> li<sub>near</sub>l<sub>y</sub> <sub>w</sub>ith th<sub>e</sub> it<sub>ems</sub> th<sub>a</sub>t <sub>are</sub> <sub>mos</sub>t i<sub>n</sub>f<sub>orma</sub>ti<sub>ve</sub> <sub>a</sub>b<sub>ou</sub>t <sub>w</sub>h<sub>ere</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld li<sub>es.</sub> M<sub>oreover,</sub> b<sub>ecause</sub> $\begin{array} { r } { \sum _ { i = 1 } ^ { I } b _ { t , i } = 1 } \end{array}$ <sub>,</sub> th<sub>e</sub> dif<sub>erence</sub> $\theta _ { i } - a _ { q _ { t } , b _ { t } } ( \pmb \theta )$ d<sub>epen</sub>d<sub>s on</sub>l<sub>y on</sub> th<sub>e re</sub>l<sub>a</sub>ti<sub>ve</sub> <sub>u</sub>tiliti<sub>es,</sub> <sub>no</sub>t <sub>on</sub> ${ \overline { { \theta } } } .$ Th<sub>us,</sub> <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> d<sub>oes</sub> <sub>no</sub>t <sub>rewar</sub>d <sub>re</sub>d<sub>uc</sub>i<sub>ng</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-</sub>i<sub>rre</sub>l<sub>evan</sub>t <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> <sub>a</sub>b<sub>ou</sub>t ${ \overline { { \theta } } } ,$ <sub>p</sub>uttin<sub>g</sub> ratin<sub>g</sub>s and com<sub>p</sub>arisons on the same footin<sub>g</sub> in line with the intuition of §3.1. Note that the s<sub>p</sub>ecial case constructed in §3.1 is in the set of afine thresholds: subtractin<sub>g</sub> $\overline { { \theta } }$ corres<sub>p</sub>on<sup>d</sup>s to ${ \pmb { b } } = \mathbf { 1 } / I$

## 3.3. The surrogate loss

A<sub>s</sub> di<sub>scusse</sub>d <sub>a</sub>b<sub>ove, we</sub> t<sub>rea</sub>t th<sub>e expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> E $[ \mathcal { M } ( \pmb { \theta } ; S _ { k } ^ { * } ( \pmb { \mu _ { t - 1 } } ) , a _ { q _ { t } , \pmb { b _ { t } } } ) \vert p _ { t - 1 } ]$ as a static objective fi<sub>xe</sub>d <sub>a</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e</sub> $p _ { t - 1 }$ <sub>an</sub>d <sub>use</sub> it t<sub>o</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> h<sub>ow</sub> <sub>va</sub>l<sub>ua</sub>bl<sub>e</sub> <sub>eac</sub>h <sub>po</sub>t<sub>en</sub>ti<sub>a</sub>l <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>w</sub>ill b<sub>e.</sub> W<sub>e</sub> fi<sub>x</sub> $\pmb { \mu } _ { t - 1 } , S _ { k } ^ { * } ( \pmb { \mu } _ { t - 1 } )$ <sub>an</sub>d $\left( q _ { t } , b _ { t } \right)$ <sub>an</sub>d <sub>va</sub>l<sub>ue</sub> <sub>an</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> $( i , j )$ th<sub>roug</sub>h th<sub>e</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> it i<sub>s</sub> <sub>expec</sub>t<sub>e</sub>d t<sub>o</sub> <sub>re</sub>d<sub>uce.</sub> A<sub>ccor</sub>di<sub>ng</sub>l<sub>y,</sub> d<sub>e</sub>fi<sub>ne</sub> it<sub>s expec</sub>t<sub>e</sub>d <sub>pos</sub>t<sub>er</sub>i<sub>or covar</sub>i<sub>ance a</sub>ft<sub>er eva</sub>l<sub>ua</sub>ti<sub>on</sub> $( i , j )$

$$
\overline { { \Sigma } } _ { t } ( ( i , j ) , p _ { t - 1 } ) : = \sum _ { x \in \{ - 1 , 1 \} } \mathbb { P } ( X _ { i j } = x \mid p _ { t - 1 } ) \operatorname { C o v } \left( \theta \mid X _ { i j } = x , p _ { t - 1 } \right) ,
$$

<sub>an</sub>d $\overline { { p } } _ { t } ( i , j ) : = N \Big ( \mu _ { t - 1 } , \overline { { \Sigma } } _ { t } ( ( i , j ) , p _ { t - 1 } ) \Big )$ <sub>.</sub> W<sub>e ca</sub>ll $\overline { { p } } _ { t } ( i , j )$ the expected posterior, since both of its moments <sub>are</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>momen</sub>t<sub>s:</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>covar</sub>i<sub>ance,</sub> <sub>an</sub>d th<sub>e</sub> <sub>mean</sub> b<sub>y</sub> th<sub>e</sub> <sub>mar</sub>ti<sub>nga</sub>l<sub>e</sub> <sub>proper</sub>t<sub>y</sub>

<sub>o</sub>f B<sub>ayes</sub>i<sub>an</sub> <sub>up</sub>d<sub>a</sub>ti<sub>ng.</sub> Th<sub>en</sub> <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>surroga</sub>t<sub>e</sub> l<sub>oss</sub> <sub>assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> $( i , j )$ <sub>as</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss,</sub> <sub>w</sub>ith <sub>expec</sub>t<sub>a</sub>ti<sub>on</sub> t<sub>a</sub>k<sub>en</sub> <sub>w</sub>ith <sub>respec</sub>t t<sub>o</sub> $\overline { { p } } _ { t } ( i , j )$ ,

$$
\begin{array} { r } { \mathbb { E } _ { \pmb { \theta } \sim \overline { { p } } _ { t } ( i , j ) } \left[ \mathcal { M } \big ( \pmb { \theta } ; S _ { k } ^ { \ast } ( \pmb { \mu _ { t - 1 } } ) , a _ { q _ { t } , { b _ { t } } } \big ) \right] . } \end{array}\tag{8}
$$

In (8), the current <sub>p</sub>rior and the threshold determine which items’ uncertaint<sub>y</sub> is consequential, while $\overline { { \Sigma } } _ { t } ( ( i , j ) , p _ { t - 1 } )$ d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> h<sub>ow muc</sub>h <sub>o</sub>f th<sub>a</sub>t <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> i<sub>s expec</sub>t<sub>e</sub>d t<sub>o rema</sub>i<sub>n a</sub>ft<sub>er eva</sub>l<sub>ua</sub>ti<sub>on</sub> $( i , j )$ <sub>.</sub> L<sub>emma</sub> 2 <sub>g</sub>i<sub>ves</sub> <sub>an</sub> <sub>exp</sub>li<sub>c</sub>it <sub>express</sub>i<sub>on</sub> f<sub>or</sub> $\overline { { \Sigma } } _ { t } ( ( i , j ) , p _ { t - 1 } )$

LEMMA 2. Starting with prior $N ( \mu _ { t - 1 } , \Sigma _ { t - 1 } )$ , let $\mu _ { t - 1 , 0 } : = \theta _ { 0 }$ and, for evaluation $( i , j ) \in \mathcal { I }$ , define

$$
\alpha _ { t , i j } : = \frac { \beta _ { J } ( \mu _ { t - 1 , i } - \mu _ { t - 1 , j } ) } { \sqrt { 1 + \beta _ { J } ^ { 2 } \sigma _ { t - 1 , i j } ^ { 2 } } } \qquad a n d \qquad \delta _ { t , i j } : = \frac { \beta _ { J } ^ { 2 } \sigma _ { t - 1 , i j } ^ { 2 } } { 1 + \beta _ { J } ^ { 2 } \sigma _ { t - 1 , i j } ^ { 2 } } \frac { \phi ( \alpha _ { t , i j } ) ^ { 2 } } { \Phi ( \alpha _ { t , i j } ) \Phi ( - \alpha _ { t , i j } ) } .
$$

Then the expected posterior covariance is

$$
\overline { { \Sigma } } _ { t } ( ( i , j ) , p _ { t - 1 } ) = \Sigma _ { t - 1 } - \frac { \delta _ { t , i j } } { \sigma _ { t - 1 , i j } ^ { 2 } } \Sigma _ { t - 1 } ( e _ { i } - e _ { j } ) ( e _ { i } - e _ { j } ) ^ { \top } \Sigma _ { t - 1 } .\tag{9}
$$

H<sub>ere</sub> $\delta _ { t , i j }$ i<sub>s</sub> th<sub>e expec</sub>t<sub>e</sub>d <sub>propor</sub>ti<sub>on o</sub>f th<sub>e var</sub>i<sub>ance o</sub>f $\theta _ { i } - \theta _ { j }$ <sub>re</sub>d<sub>uce</sub>d b<sub>y</sub> th<sub>e eva</sub>l<sub>ua</sub>ti<sub>on.</sub> Thi<sub>s</sub> d<sub>epen</sub>d<sub>s on</sub> h<sub>ow uncer</sub>t<sub>a</sub>i<sub>n</sub> th<sub>e ou</sub>t<sub>come</sub> i<sub>s a pr</sub>i<sub>or</sub>i<sub>,</sub> th<sub>e var</sub>i<sub>ance o</sub>f $\theta _ { i } - \theta _ { j }$ <sub>, an</sub>d th<sub>e eva</sub>l<sub>ua</sub>ti<sub>on accuracy.</sub> Th<sub>e covar</sub>i<sub>ance</sub> terms in (9) determine how the information learned about (�, �) indirectl<sub>y</sub> afects our <sub>p</sub>osterior about other items throu<sub>g</sub>h correlation. Plu<sub>gg</sub>in<sub>g</sub> in the ex<sub>p</sub>ected covariance in (9), the surro<sub>g</sub>ate loss in (8) is

$$
\mathbb { E } _ { \theta \sim \overline { { p } } _ { t } ( i , j ) } \left[ M \big ( \theta ; S _ { k } ^ { * } ( \mu _ { t - 1 } ) , a _ { q _ { t } , b _ { t } } \big ) \right] = k q _ { t } - \sum _ { \ell \in S _ { k } ^ { * } ( \mu _ { t - 1 } ) } \mu _ { t - 1 , \ell } + \sum _ { \ell = 1 } ^ { I } \mathbb { E } _ { \theta \sim \overline { { p } } _ { t } ( i , j ) } \left[ \big ( \theta _ { \ell } - a _ { q _ { t } , b _ { t } } ( \theta ) \big ) ^ { + } \right] .
$$

O<sub>n</sub> th<sub>e r</sub>i<sub>g</sub>ht<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e, on</sub>l<sub>y</sub> th<sub>e</sub> l<sub>as</sub>t t<sub>erm</sub> d<sub>epen</sub>d<sub>s on</sub> $( i , j )$ <sub>, w</sub>hi<sub>c</sub>h <sub>can</sub> b<sub>e</sub> th<sub>oug</sub>ht <sub>o</sub>f <sub>as</sub> th<sub>e con</sub>t<sub>ro</sub>ll<sub>a</sub>bl<sub>e</sub> <sub>par</sub>t <sub>o</sub>f th<sub>e</sub> <sub>surroga</sub>t<sub>e</sub> l<sub>oss.</sub> L<sub>e</sub>t $\sigma _ { t , \ell } ^ { \prime } ( i , j )$ <sub>an</sub>d $z _ { t , \ell } ^ { \prime } ( i , j )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> <sub>ana</sub>l<sub>ogues</sub> <sub>o</sub>f $\sigma _ { t - 1 , \ell }$ <sub>an</sub>d $z _ { t - 1 , \ell }$ from (7) and C<sub>oro</sub>ll<sub>ary</sub> 1 <sub>compu</sub>t<sub>e</sub>d <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>pos</sub>t<sub>er</sub>i<sub>or.</sub> Th<sub>e</sub> <sub>con</sub>t<sub>ro</sub>ll<sub>a</sub>bl<sub>e</sub> <sub>surroga</sub>t<sub>e</sub> l<sub>oss</sub> <sub>can</sub> b<sub>e</sub> <sub>wr</sub>itt<sub>en</sub> <sub>as</sub>

$$
\begin{array} { l } { \displaystyle \mathcal { L } ( ( i , j ) \mid p _ { t - 1 } ) : = \sum _ { \ell = 1 } ^ { I } \mathbb { E } _ { \theta \sim \overline { { p } } _ { t } ( i , j ) } \left[ \left( \theta _ { \ell } - a _ { q _ { t } , b _ { t } } ( \theta ) \right) ^ { + } \right] } \\ { \displaystyle \qquad = \sum _ { \ell = 1 } ^ { I } \left[ \left( \mu _ { t - 1 , \ell } - q _ { t } \right) \Phi ( z _ { t , \ell } ^ { \prime } ( i , j ) ) + \sigma _ { t , \ell } ^ { \prime } ( i , j ) \phi ( z _ { t , \ell } ^ { \prime } ( i , j ) ) \right] . } \end{array}\tag{10}
$$

F<sub>or</sub> l<sub>a</sub>t<sub>er use,</sub> l<sub>e</sub>t $\mathcal { L } _ { \ell } ( ( i , j ) \mid p _ { t - 1 } )$ denote the ℓth term of (10). The allocation rule in §4 chooses<sub>,</sub> in each <sub>roun</sub>d<sub>,</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>w</sub>h<sub>ose</sub> <sub>resu</sub>lti<sub>ng</sub> d<sub>ecrease</sub> i<sub>n</sub> <sub>surroga</sub>t<sub>e</sub> l<sub>oss</sub> <sub>per</sub> <sub>un</sub>it <sub>o</sub>f <sub>resource</sub> i<sub>s</sub> l<sub>arges</sub>t<sub>.</sub>

## 4. Sequential evaluation allocation

We next use the surro<sub>g</sub>ate loss to construct the se<sub>q</sub>uential evaluation <sub>p</sub>olic<sub>y</sub>. §4.1 derives the mar<sub>g</sub>inal decrease in the surro<sub>g</sub>ate <sub>p</sub>er evaluation and the allocation rule based on it<sub>,</sub> §4.2 <sub>g</sub>ives the se<sub>q</sub>uential <sub>p</sub>rocedure<sub>,</sub> and §4.3 establishes that the <sub>p</sub>olic<sub>y</sub> is as<sub>y</sub>m<sub>p</sub>toticall<sub>y</sub> consistent<sub>,</sub> meanin<sub>g</sub> EOC → 0 as $R \to \infty$

## 4.1. Marginal scores

C<sub>ons</sub>id<sub>er an ar</sub>bit<sub>rary roun</sub>d � f<sub>or w</sub>hi<sub>c</sub>h <sub>we wr</sub>it<sub>e</sub> $\mathcal { L } ( ( i , j ) )$ <sub>an</sub>d $\mathcal { L } _ { \ell } ( ( i , j ) )$ for the losses in (10), leavin<sub>g</sub> the d<sub>epen</sub>d<sub>ence</sub> <sub>on</sub> $p _ { t - 1 }$ i<sub>mp</sub>li<sub>c</sub>it<sub>.</sub> L<sub>e</sub>t $\begin{array} { r } { \mathcal { L } ( 0 ) = \sum _ { \ell = 1 } ^ { I } \mathcal { L } _ { \ell } ( 0 ) } \end{array}$ b<sub>e</sub> th<sub>e</sub> l<sub>oss</sub> b<sub>e</sub>f<sub>ore</sub> <sub>any</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on,</sub> <sub>o</sub>bt<sub>a</sub>i<sub>ne</sub>d b<sub>y</sub> <sub>rep</sub>l<sub>ac</sub>i<sub>ng</sub> $\sigma _ { t , \ell } ^ { \prime } ( i , j )$ <sub>an</sub>d $z _ { t , \ell } ^ { \prime } ( i , j )$ <sub>w</sub>ith $\sigma _ { t - 1 , \ell }$ <sub>an</sub>d $z _ { t - 1 , \ell } .$ <sub>.</sub> T<sub>o</sub> <sub>c</sub>h<sub>oose</sub> <sub>w</sub>h<sub>a</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>o</sub> <sub>con</sub>d<sub>uc</sub>t <sub>we</sub> <sub>compu</sub>t<sub>e</sub> th<sub>e</sub> d<sub>ecrease</sub> i<sub>n</sub> <sub>surroga</sub>t<sub>e</sub> l<sub>oss</sub> <sub>per</sub> <sub>un</sub>it <sub>o</sub>f <sub>resource</sub>

$$
\psi _ { t , i j } : = \frac { \mathcal { L } ( 0 ) - \mathcal { L } ( ( i , j ) ) } { r _ { J } }
$$

f<sub>or eac</sub>h <sub>eva</sub>l<sub>ua</sub>ti<sub>on,</sub> th<sub>en eva</sub>l<sub>ua</sub>t<sub>e arg max</sub> $\mathsf { \Phi } ^ { \prime } ( i , j ) \in \mathcal { I } : r _ { J } \le R _ { t } \psi _ { t , i j }$ . This <sub>p</sub>rocedure, Surro<sub>g</sub>ate Allocation (SA), is <sub>g</sub>i<sub>ven</sub> i<sub>n</sub> Al<sub>gor</sub>ith<sub>m</sub> 1<sub>.</sub>

Al orithm 1 Surro ate Allocation (SA)   
In<sub>p</sub>ut: $\mu _ { t - 1 } , \Sigma _ { t - 1 } , R _ { t } \geq \operatorname* { m i n } ( r _ { 0 } , r _ { 1 } )$   
1<sub>:</sub> C<sub>ompu</sub>t<sub>e a m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> th<sub>res</sub>h<sub>o</sub>ld $\left( q _ { t } , b _ { t } \right)$ <sub>v</sub>i<sub>a</sub> Th<sub>eorem</sub> 1 <sub>an</sub>d C<sub>oro</sub>ll<sub>ary</sub> 1   
2: Compute L (0)   
3<sub>:</sub> f<sub>o</sub>r $( { \dot { i } } , j ) \in { \dot { \cal { I } } }$ <sub>w</sub>ith $r _ { J } \leq R _ { t }$   
4<sub>:</sub> <sup>C</sup>om<sub>p</sub>ute $\overline { { \Sigma } } _ { t } ( ( i , \bar { j } ) , p _ { t - 1 } )$ b<sub>y</sub> (9)   
5<sub>:</sub> Com<sub>p</sub>ute L ( (�, �)) b<sub>y</sub> (10)   
6<sub>:</sub> $\psi _ { t , i j } \overleftarrow {  } [ \mathcal { L } ( 0 ) - \mathcal { L } ( ( i , j ) ) ] / r _ { J }$   
7<sub>:</sub> r<sub>e</sub>t<sub>u</sub>rn <sub>an</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> i<sub>n</sub> <sub>arg</sub> $\mathrm { m a x } _ { ( i , j ) \in \mathcal { I } : r _ { J } \leq R _ { t } } \psi _ { t , i j }$

T<sub>o</sub> <sub>ge</sub>t i<sub>ns</sub>i<sub>g</sub>ht i<sub>n</sub>t<sub>o</sub> <sub>w</sub>h<sub>a</sub>t d<sub>r</sub>i<sub>ves</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce,</sub> <sub>we</sub> <sub>wr</sub>it<sub>e</sub> th<sub>e</sub> <sub>surroga</sub>t<sub>e</sub> <sub>marg</sub>i<sub>na</sub>l d<sub>ecrease</sub> <sub>exp</sub>li<sub>c</sub>itl<sub>y.</sub>

$$
\psi _ { t , i j } = \frac { \mathcal { L } ( 0 ) - \mathcal { L } ( ( i , j ) ) } { r _ { j } } = \frac { 1 } { r _ { j } } \sum _ { \ell = 1 } ^ { I } \int _ { ( \sigma _ { t , \ell } ^ { \prime } ( i , j ) ) ^ { 2 } } ^ { \sigma _ { t - 1 , \ell } ^ { 2 } } \frac { \partial \mathcal { L } _ { \ell } } { \partial \nu } d \nu = \frac { 1 } { r _ { j } } \sum _ { \ell = 1 } ^ { I } \int _ { 0 } ^ { 1 - \frac { ( \sigma _ { t , \ell } ^ { \prime } ( i , j ) ) ^ { 2 } } { \sigma _ { t - 1 , \ell } ^ { 2 } } } \frac { \sigma _ { t - 1 , \ell } } { 2 \sqrt { 1 - s } } \phi \left( \frac { z _ { t - 1 , \ell } } { \sqrt { 1 - s } } \right) d s .\tag{11}
$$

Th<sub>e</sub> <sub>upper</sub> li<sub>m</sub>it i<sub>n</sub> th<sub>e</sub> i<sub>n</sub>t<sub>egra</sub>l <sub>correspon</sub>d<sub>s</sub> t<sub>o</sub> th<sub>e</sub> <sub>propor</sub>ti<sub>on</sub> <sub>o</sub>f it<sub>em</sub> $\ell { \boldsymbol { \mathrm { s } } }$ uncertaint<sub>y</sub> (relative to the threshold) <sub>re</sub>d<sub>uce</sub>d b<sub>y</sub> <sub>con</sub>d<sub>uc</sub>ti<sub>ng</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> $( i , j )$ <sub>.</sub> T<sub>o</sub> <sub>see</sub> thi<sub>s</sub> <sub>more</sub> <sub>c</sub>l<sub>ear</sub>l<sub>y,</sub> b<sub>y</sub> L<sub>emma</sub> 2 <sub>we</sub> h<sub>ave</sub>

$$
1 - \frac { ( \sigma _ { t , \ell } ^ { \prime } ( i , j ) ) ^ { 2 } } { \sigma _ { t - 1 , \ell } ^ { 2 } } = \delta _ { t , i j } \thinspace { \mathrm { C o r r } \big ( \theta _ { i } - \theta _ { j } , \theta _ { \ell } - a _ { q _ { t } , b _ { t } } ( \theta ) \big ) ^ { 2 } } ,
$$

<sub>w</sub>h<sub>ere</sub> $\delta _ { t , i j }$ i<sub>s</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>propor</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>var</sub>i<sub>ance</sub> <sub>o</sub>f $\theta _ { i } - \theta _ { j }$ <sub>re</sub>d<sub>uce</sub>d b<sub>y</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on.</sub> Th<sub>e</sub> <sub>square</sub>d t<sub>erm</sub> <sub>measures</sub> h<sub>ow</sub> <sub>muc</sub>h <sub>o</sub>f th<sub>e</sub> <sub>var</sub>i<sub>ance</sub> <sub>re</sub>d<sub>uc</sub>ti<sub>on</sub> <sub>o</sub>f $( i , j )$ t<sub>rans</sub>f<sub>ers</sub> t<sub>o</sub> it<sub>em</sub> ℓ th<sub>roug</sub>h <sub>corre</sub>l<sub>a</sub>ti<sub>on.</sub>

Th<sub>e</sub> i<sub>n</sub>t<sub>egran</sub>d <sub>measures</sub> h<sub>ow</sub> <sub>va</sub>l<sub>ua</sub>bl<sub>e</sub> it i<sub>s</sub> t<sub>o</sub> <sub>re</sub>d<sub>uce</sub> it<sub>em</sub> $\ell { \boldsymbol { \mathrm { s } } }$ uncertainty for our objective. It is the i<sub>ns</sub>t<sub>an</sub>t<sub>aneous</sub> d<sub>ecrease</sub> i<sub>n expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss per a</sub>dditi<sub>ona</sub>l <sub>propor</sub>ti<sub>on o</sub>f it<sub>s var</sub>i<sub>ance remove</sub>d<sub>.</sub> Th<sub>e</sub> $\phi { \left( z _ { t - 1 , \ell } \right) }$ t<sub>erm</sub> i<sub>s</sub> l<sub>arge w</sub>h<sub>en we are</sub> hi<sub>g</sub>hl<sub>y uncer</sub>t<sub>a</sub>i<sub>n a</sub>b<sub>ou</sub>t <sub>w</sub>hi<sub>c</sub>h <sub>s</sub>id<sub>e o</sub>f th<sub>e</sub> t<sub>op-</sub>� th<sub>res</sub>h<sub>o</sub>ld it<sub>em</sub> ℓ li<sub>es on,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>occurs w</sub>h<sub>en e</sub>ith<sub>er</sub> th<sub>e mean</sub> $\mu _ { t - 1 , \ell }$ i<sub>s c</sub>l<sub>ose</sub> t<sub>o</sub> th<sub>e expec</sub>t<sub>e</sub>d th<sub>res</sub>h<sub>o</sub>ld l<sub>eve</sub>l $q _ { t }$ <sub>or w</sub>h<sub>en</sub> $\sigma _ { t - 1 , \ell }$ i<sub>s</sub> l<sub>arge.</sub> Th<sub>e a</sub>dditi<sub>ona</sub>l <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d d<sub>ev</sub>i<sub>a</sub>ti<sub>on</sub> t<sub>erm</sub> $\sigma _ { t - 1 , \ell }$ <sub>sca</sub>l<sub>es</sub> thi<sub>s</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> b<sub>y</sub> h<sub>ow</sub> f<sub>ar</sub> th<sub>e</sub> it<sub>em</sub>’<sub>s</sub> <sub>u</sub>tilit<sub>y</sub> <sub>may</sub> b<sub>e</sub> f<sub>rom</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>, re</sub>fl<sub>ec</sub>ti<sub>ng</sub> th<sub>a</sub>t <sub>se</sub>l<sub>ec</sub>ti<sub>on m</sub>i<sub>s</sub>t<sub>a</sub>k<sub>es are pena</sub>li<sub>ze</sub>d b<sub>y</sub> th<sub>e</sub>i<sub>r magn</sub>it<sub>u</sub>d<sub>e.</sub>

Put to<sub>g</sub>ether, (11) scores an evaluation b<sub>y</sub> how much its uncertaint<sub>y</sub> reduction im<sub>p</sub>acts ex<sub>p</sub>ected mismatch l<sub>oss</sub> <sub>across</sub> <sub>a</sub>ll it<sub>ems.</sub> Th<sub>e</sub> i<sub>n</sub>t<sub>egran</sub>d <sub>cap</sub>t<sub>ures</sub> h<sub>ow</sub> <sub>va</sub>l<sub>ua</sub>bl<sub>e</sub> <sub>re</sub>d<sub>uc</sub>i<sub>ng</sub> it<sub>em</sub> $\ell { \boldsymbol { \mathrm { s } } }$ uncertaint<sub>y</sub> (relative to the threshold) is, and the u<sub>pp</sub>er limit of the inte<sub>g</sub>ral ca<sub>p</sub>tures how much of this item’s uncertaint<sub>y</sub> is reduced b<sub>y</sub> evaluating (�, �), either directl<sub>y</sub> when $\ell \in \{ i , j \}$ <sub>or</sub> i<sub>n</sub>di<sub>rec</sub>tl<sub>y</sub> th<sub>roug</sub>h <sub>corre</sub>l<sub>a</sub>ti<sub>on</sub> <sub>w</sub>h<sub>en</sub> $\ell \notin \{ i , j \}$

## 4.2. The sequential procedure

N<sub>ow</sub> th<sub>a</sub>t <sub>we</sub> h<sub>ave</sub> <sub>an</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> t<sub>o</sub> <sub>c</sub>h<sub>oose</sub> th<sub>e</sub> <sub>nex</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on,</sub> <sub>we</sub> l<sub>ay</sub> <sub>ou</sub>t th<sub>e</sub> f<sub>u</sub>ll <sub>sequen</sub>ti<sub>a</sub>l <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>proce</sub>d<sub>ure</sub> i<sub>n</sub> Al<sub>gor</sub>ith<sub>m</sub> 2<sub>.</sub> It t<sub>a</sub>k<sub>es a</sub> f<sub>am</sub>il<sub>y o</sub>f <sub>po</sub>li<sub>c</sub>i<sub>es</sub> $\{ \mathrm { E v A L U A T I O N R U L E } ^ { R _ { t } } \}$ th<sub>a</sub>t <sub>map</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e an</sub>d b<sub>u</sub>d<sub>ge</sub>t t<sub>o</sub> <sub>an a</sub>f<sub>or</sub>d<sub>a</sub>bl<sub>e eva</sub>l<sub>ua</sub>ti<sub>on.</sub> F<sub>or now,</sub> th<sub>e eva</sub>l<sub>ua</sub>ti<sub>on ru</sub>l<sub>e can</sub> b<sub>e</sub> th<sub>oug</sub>ht <sub>o</sub>f <sub>as</sub> SA<sub>,</sub> b<sub>u</sub>t <sub>we wr</sub>it<sub>e</sub> it i<sub>n genera</sub>l f<sub>orm as</sub> we will introduce a diferent evaluation rule for the end<sub>g</sub>ame in §5. It also assumes the <sub>p</sub>osterior u<sub>p</sub>date comes f<sub>rom</sub> <sub>a</sub> B<sub>elief</sub>U<sub>pdate</sub> <sub>proce</sub>d<sub>ure</sub> i<sub>mp</sub>l<sub>emen</sub>ti<sub>ng</sub> <sub>an</sub> <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub> i<sub>n</sub>f<sub>erence</sub> <sub>me</sub>th<sub>o</sub>d f<sub>rom</sub> A<sub>pprox</sub>i<sub>ma</sub>ti<sub>on</sub> 1<sub>.</sub>

Al<sub>go</sub>rithm 2 S<sub>equen</sub>ti<sub>a</sub>l <sub>proce</sub>d<sub>ure</sub>   
In<sub>p</sub>ut: <sub>p</sub>rior $( \mu _ { 0 } , \Sigma _ { 0 } ) , R ,$ {EvaluationRule<sup>��</sup> }   
1<sub>:</sub> $t  1 , \quad R _ { 1 }  R$   
2<sub>: w</sub>hil<sub>e</sub> $R _ { t } \geq \operatorname* { m i n } ( r _ { 0 } , r _ { 1 } )$   
3<sub>:</sub> $( i _ { t } , j _ { t } ) \gets$ E<sub>valuation</sub>R<sub>ule</sub> $\therefore R _ { t } ( \mu _ { t - 1 } , \Sigma _ { t - 1 } )$   
4<sub>:</sub> Ob<sub>se</sub>r<sub>ve ou</sub>t<sub>co</sub>m<sub>e</sub> �<sub>�</sub>   
5<sub>:</sub> $\left( \mu _ { t } , \Sigma _ { t } \right) \gets$ B<sub>elief</sub>U<sub>pdate</sub> $( \mu _ { t - 1 } , \Sigma _ { t - 1 } , ( i _ { t } , j _ { t } ) , X _ { t } )$   
<sup>6:</sup> <sub>7:</sub> $R _ { t + 1 } \gets R _ { t } - r _ { J t }$   
$t \gets t + 1$   
8<sub>:</sub> $T \gets t - 1$   
9: return $S _ { k } ^ { * } ( \mu _ { T } )$

## 4.3. Asymptotic consistency

A natural concern with a policy based on a static objective is that it may reach a state from which, even <sub>w</sub>ith <sub>an</sub> i<sub>n</sub>fi<sub>n</sub>it<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>,</sub> <sub>some</sub> it<sub>ems</sub> <sub>ou</sub>t<sub>s</sub>id<sub>e</sub> th<sub>e</sub> fi<sub>na</sub>l t<sub>op-</sub>� <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>re</sub>t<sub>a</sub>i<sub>n</sub> <sub>a</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>o</sub>f b<sub>e</sub>l<sub>ong</sub>i<sub>ng</sub> t<sub>o</sub> th<sub>e</sub> t<sub>rue</sub> t<sub>op-</sub>�<sub>.</sub> W<sub>e</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t thi<sub>s</sub> <sub>can</sub> <sub>never</sub> b<sub>e</sub> th<sub>e</sub> <sub>case</sub> f<sub>or</sub> Al<sub>gor</sub>ith<sub>m</sub> 2 <sub>w</sub>ith $\mathrm { E v a L U A T I O N R U L E } ^ { R _ { t } }$ $= { \mathrm { S A } } \colon$ in <sub>p</sub>articular, the al<sub>g</sub>orithm is as<sub>y</sub>m<sub>p</sub>toticall<sub>y</sub> consistent, meanin<sub>g</sub> EOC → 0 as $R \to \infty$ <sub>,</sub> <sub>un</sub>d<sub>er</sub> t<sub>wo</sub> <sub>m</sub>ild assum<sub>p</sub>t<sup>i</sup>ons.

ASSUMPTION 1. The true link function $g : \mathbb { R } \to ( 0 , 1 )$ is continuous, symmetric $( g ( - x ) = 1 - g ( x ) )$ , strictly increasing, and log-concave.

A<sub>ssump</sub>ti<sub>on</sub> 1 i<sub>s</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> <sub>pro</sub>bit li<sub>n</sub>k <sub>we</sub> <sub>use</sub> $( g = \Phi )$ <sub>an</sub>d <sub>a</sub>ll<sub>ows</sub> f<sub>or more genera</sub>l li<sub>n</sub>k<sub>s.</sub>

ASSUMPTION 2. Let $( \mu _ { t } ^ { * } , \Sigma _ { t } ^ { * } )$ denote the exact first two moments of the posterior (which are generally unavailable in closed form). The approximate posterior moments $\left( \mu _ { t } , \Sigma _ { t } \right)$ satisfy: (i) there exist constants $0 < \chi _ { - } \leq \chi _ { + } < \infty$ such that, almost surely, for all suficiently large �, $\chi _ { - } \Sigma _ { t } ^ { * } \preceq \Sigma _ { t } \preceq \chi _ { + } \Sigma _ { t } ^ { * }$ , (ii) su<sub>p�</sub> $\| \mu _ { t } \| _ { \infty } < \infty$ almost surely, and (iii) $\mathbb { E } [ \| \pmb { \mu } _ { T } - \pmb { \mu } _ { T } ^ { * } \| _ { \infty } ] \to 0$ as $R \to \infty ,$ , where $\| \cdot \| _ { \infty }$ is the L-infinity norm.

A<sub>ssump</sub>ti<sub>on</sub> 2 <sub>says</sub> th<sub>a</sub>t <sub>w</sub>h<sub>a</sub>t<sub>ever me</sub>th<sub>o</sub>d <sub>we use</sub> i<sub>n</sub> th<sub>e</sub> B<sub>elief</sub>U<sub>pdate s</sub>t<sub>ep even</sub>t<sub>ua</sub>ll<sub>y y</sub>i<sub>e</sub>ld<sub>s a</sub> “<sub>goo</sub>d” <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> <sub>o</sub>f th<sub>e</sub> fi<sub>rs</sub>t t<sub>wo</sub> <sub>momen</sub>t<sub>s</sub> <sub>o</sub>f th<sub>e</sub> t<sub>rue</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>g</sub>i<sub>ven</sub> th<sub>e</sub> <sub>co</sub>ll<sub>ec</sub>t<sub>e</sub>d d<sub>a</sub>t<sub>a.</sub>

THEOREM 2. Under Assumptions 1 and 2, Algorithm 2 with EvaluationRul $\boldsymbol { \mathrm { E } } ^ { R _ { t } } = \mathbf { S } \mathbf { A }$ satisfies

$$
\mathbb { E } [ \sum _ { i \in S _ { k } ^ { * } ( \pmb { \theta } ) } \theta _ { i } - \sum _ { i \in S _ { k } ^ { * } ( \mu _ { T } ) } \theta _ { i } ]  0 \qquad a s \qquad R  \infty .
$$

P<sub>roo</sub>f id<sub>ea.</sub> W<sub>e</sub> <sub>ou</sub>tli<sub>ne</sub> th<sub>e</sub> t<sub>wo</sub> <sub>ma</sub>i<sub>n</sub> <sub>componen</sub>t<sub>s</sub> <sub>o</sub>f th<sub>e</sub> <sub>proo</sub>f<sub>.</sub> Th<sub>e</sub> fi<sub>rs</sub>t <sub>an</sub>d <sub>more</sub> t<sub>ec</sub>h<sub>n</sub>i<sub>ca</sub>ll<sub>y</sub> <sub>c</sub>h<sub>a</sub>ll<sub>eng</sub>i<sub>ng</sub> <sub>par</sub>t i<sub>s</sub> t<sub>o</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> <sub>se</sub>t <sub>o</sub>f <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>samp</sub>l<sub>es</sub> i<sub>n</sub>fi<sub>n</sub>it<sub>e</sub>l<sub>y</sub> <sub>many</sub> ti<sub>mes</sub> <sub>spans</sub> $\mathbf { 1 } ^ { \perp } : = \{ \boldsymbol { x } \in \mathbb { R } ^ { I } : \mathbf { 1 } ^ { \top } \boldsymbol { x } = 0 \}$ th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-re</sub>l<sub>evan</sub>t <sub>su</sub>b<sub>space, w</sub>hi<sub>c</sub>h <sub>we prove</sub> b<sub>y con</sub>t<sub>ra</sub>di<sub>c</sub>ti<sub>on.</sub> F<sub>or an eva</sub>l<sub>ua</sub>ti<sub>on</sub> $( i , j )$ <sub>, we re</sub>f<sub>er</sub> t<sub>o</sub> $e _ { i } - e _ { j }$ <sub>as</sub> it<sub>s</sub> “di<sub>rec</sub>ti<sub>on</sub>”<sub>.</sub> L<sub>e</sub>t $E _ { \infty }$ b<sub>e</sub> th<sub>e se</sub>t <sub>o</sub>f di<sub>rec</sub>ti<sub>ons samp</sub>l<sub>e</sub>d i<sub>n</sub>fi<sub>n</sub>it<sub>e</sub>l<sub>y o</sub>ft<sub>en.</sub> O<sub>ur goa</sub>l i<sub>s</sub> th<sub>en</sub> t<sub>o prove</sub> th<sub>a</sub>t $\mathbf { 1 } ^ { \perp } \subseteq \operatorname { s p a n } ( E _ { \infty } )$ <sub>a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y.</sub> E<sub>qu</sub>i<sub>va</sub>l<sub>en</sub>tl<sub>y,</sub> f<sub>or any</sub> fi<sub>xe</sub>d <sub>se</sub>t <sub>o</sub>f di<sub>rec</sub>ti<sub>ons</sub> $\varepsilon$ <sub>w</sub>h<sub>ose span</sub> d<sub>oes no</sub>t <sub>con</sub>t<sub>a</sub>i<sub>n</sub> ${ \bf 1 ^ { \perp } }$ th<sub>e</sub> <sub>even</sub>t $\Omega _ { \mathcal { E } } : = \{ E _ { \infty } = \mathcal { E } \}$ h<sub>as</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> 0<sub>.</sub> S<sub>uppose</sub> f<sub>or</sub> <sub>con</sub>t<sub>ra</sub>di<sub>c</sub>ti<sub>on</sub> th<sub>a</sub>t $\Omega _ { \mathcal { E } }$ h<sub>as</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y.</sub> O<sub>n</sub> $\Omega _ { \mathcal { E } }$ <sub>,</sub> <sub>a</sub>ft<sub>er</sub> <sub>some</sub> fi<sub>n</sub>it<sub>e</sub> <sub>ran</sub>d<sub>om</sub> ti<sub>me,</sub> <sub>every</sub> <sub>su</sub>b<sub>sequen</sub>tl<sub>y</sub> <sub>samp</sub>l<sub>e</sub>d di<sub>rec</sub>ti<sub>on</sub> i<sub>s</sub> i<sub>n</sub> $\varepsilon .$ <sub>.</sub> Thi<sub>s</sub> <sub>re</sub>d<sub>uces</sub> th<sub>e</sub> <sub>proo</sub>f to showin<sub>g</sub> two results: (i) the mar<sub>g</sub>inal scores of directions in E tend to 0, and (ii) at least one direction not in E has a mar<sub>g</sub>inal score bounded awa<sub>y</sub> from 0. To<sub>g</sub>ether these im<sub>p</sub>l<sub>y</sub> that in some round SA chooses <sub>a</sub> di<sub>rec</sub>ti<sub>on no</sub>t i<sub>n</sub> $\varepsilon ,$ <sub>a con</sub>t<sub>ra</sub>di<sub>c</sub>ti<sub>on.</sub> Th<sub>e secon</sub>d <sub>componen</sub>t th<sub>en</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>es</sub> th<sub>a</sub>t if $\mathbf { 1 } ^ { \scriptscriptstyle \perp } \subseteq \operatorname { s p a n } ( E _ { \infty } )$ <sub>,</sub> th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>var</sub>i<sub>ance</sub> <sub>on</sub> ${ \bf 1 ^ { \perp } }$ <sub>goes</sub> t<sub>o</sub> 0<sub>,</sub> <sub>a</sub>nd th<sub>us</sub> EOC d<sub>ecays</sub> t<sub>o</sub> 0 <sub>as</sub> <sub>we</sub>ll<sub>.</sub>

Th<sub>e s</sub>i<sub>gn</sub>ifi<sub>cance o</sub>f thi<sub>s resu</sub>lt li<sub>es no</sub>t i<sub>n</sub> th<sub>e asymp</sub>t<sub>o</sub>ti<sub>c guaran</sub>t<sub>ee</sub> it<sub>se</sub>lf<sub>,</sub> b<sub>u</sub>t i<sub>n w</sub>h<sub>a</sub>t it <sub>revea</sub>l<sub>s a</sub>b<sub>ou</sub>t th<sub>e</sub> <sub>surroga</sub>t<sub>e</sub>’<sub>s</sub> <sub>s</sub>t<sub>ruc</sub>t<sub>ure.</sub> D<sub>esp</sub>it<sub>e</sub> b<sub>e</sub>i<sub>ng</sub> <sub>s</sub>h<sub>or</sub>t<sub>-s</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>an</sub>d h<sub>av</sub>i<sub>ng</sub> <sub>no</sub> d<sub>es</sub>i<sub>gna</sub>t<sub>e</sub>d <sub>ran</sub>d<sub>om</sub>i<sub>za</sub>ti<sub>on</sub> <sub>or</sub> “<sub>exp</sub>l<sub>ora</sub>ti<sub>on</sub>” <sub>p</sub>h<sub>ase,</sub> th<sub>e</sub> <sub>po</sub>li<sub>cy</sub> <sub>exp</sub>l<sub>ores</sub> <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y.</sub>

## 5. The endgame

C<sub>oncep</sub>t<sub>ua</sub>ll<sub>y,</sub> th<sub>e a</sub>ll<sub>oca</sub>ti<sub>on ru</sub>l<sub>e</sub> i<sub>n</sub> SA i<sub>s compe</sub>lli<sub>ng, par</sub>ti<sub>cu</sub>l<sub>ar</sub>l<sub>y</sub> f<sub>or ear</sub>l<sub>y an</sub>d <sub>m</sub>iddl<sub>e roun</sub>d<sub>s: even</sub> if it i<sub>s</sub> <sub>no</sub>t f<sub>orwar</sub>d<sub>-</sub>l<sub>oo</sub>ki<sub>ng</sub> i<sub>n a</sub> d<sub>ynam</sub>i<sub>c sense,</sub> it <sub>va</sub>l<sub>ues eva</sub>l<sub>ua</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>are use</sub>f<sub>u</sub>l b<sub>eyon</sub>d th<sub>e</sub>i<sub>r po</sub>t<sub>en</sub>ti<sub>a</sub>l i<sub>n</sub>fl<sub>uence</sub> <sub>on</sub> thi<sub>s roun</sub>d’<sub>s pos</sub>t<sub>er</sub>i<sub>or</sub> t<sub>op-</sub>� <sub>se</sub>t<sub>.</sub> B<sub>u</sub>t <sub>near</sub> th<sub>e en</sub>d <sub>o</sub>f th<sub>e</sub> h<sub>or</sub>i<sub>zon,</sub> th<sub>e va</sub>l<sub>ue o</sub>f <sub>suc</sub>h <sub>exp</sub>l<sub>ora</sub>ti<sub>on</sub> i<sub>s sma</sub>ll<sub>,</sub> and in the last round it is zero. Hence, since the surrogate objective is not horizon-aware, this strength of SA can turn negative in the endgame. To “zoom in” on the items that are actual top-� candidates in the endgame is <sub>p</sub>recisel<sub>y</sub> the nature of KG-t<sub>yp</sub>e <sub>p</sub>olicies. We develo<sub>p</sub> the KG <sub>p</sub>olic<sub>y</sub> for our <sub>p</sub>roblem in §5.1 and a rule for when to switch to it in §5.2.

## 5.1. The sequential KG policy

Recall from §2.3 that the KG <sub>p</sub>olic<sub>y</sub> chooses an evaluation

$$
( i _ { t } , j _ { t } ) \in \arg \operatorname* { m a x } _ { ( i , j ) \in I : r _ { I } \leq R _ { t } } \psi _ { t , i j } ^ { \mathrm { K G } } \quad \mathrm { w h e r e } \quad \psi _ { t , i j } ^ { \mathrm { K G } } : = \frac { V _ { \mathrm { m y o p i c } } ( ( i , j ) \mid p _ { t - 1 } ) } { r _ { J } } .
$$

<sup>T</sup>o com<sub>p</sub>ute $V _ { \mathrm { m y o p i c } } ( ( i , j ) \mid p _ { t - 1 } )$ , whose ex<sub>p</sub>ression is <sub>g</sub>iven in (4), Lemma 3 <sub>g</sub>ives ex<sub>p</sub>licit ex<sub>p</sub>ressions for th<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>o</sub>f <sub>eac</sub>h <sub>ou</sub>t<sub>come</sub> <sub>o</sub>f <sub>an</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>an</sub>d th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>means</sub> <sub>g</sub>i<sub>ven</sub> th<sub>e</sub> <sub>ou</sub>t<sub>come.</sub>

LEMMA 3. Starting with prior $N ( \mu _ { t - 1 } , \Sigma _ { t - 1 } )$ , for evaluation $( i , j ) \in \mathcal { I }$ and outcome $x \in \{ - 1 , 1 \}$ ,

$$
\mathbb { P } ( X _ { i j } = x \mid p _ { t - 1 } ) = \Phi ( x \alpha _ { t , i j } ) \quad a n d \quad \mu _ { t } ( x \mid ( i , j ) ) = \mu _ { t - 1 } + \frac { x \beta _ { j } \phi ( x \alpha _ { t , i j } ) } { \Phi ( x \alpha _ { t , i j } ) \sqrt { 1 + \beta _ { j } ^ { 2 } \sigma _ { t - 1 , i j } ^ { 2 } } } \Sigma _ { t - 1 } ( e _ { i } - e _ { j } ) ,
$$

where $\alpha _ { t , i j }$ is defined in Lemma 2.

As discussed in §2.3<sub>,</sub> when no outcome of an<sub>y</sub> afordable evaluation can chan<sub>g</sub>e the <sub>p</sub>osterior to<sub>p</sub>-� set<sub>,</sub> all of t<sup>h</sup>e m<sub>y</sub>o<sub>p</sub><sup>i</sup>c <sub>g</sub>a<sup>i</sup>ns $\psi _ { t , i j } ^ { \mathrm { K G } }$ <sub>are</sub> <sub>zero.</sub> Th<sub>e</sub> KG <sub>po</sub>li<sub>cy</sub> th<sub>en</sub> <sub>requ</sub>i<sub>res</sub> <sub>a</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>ru</sub>l<sub>e</sub> t<sub>o</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> <sub>w</sub>hi<sub>c</sub>h <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>o</sub> <sub>con</sub>d<sub>uc</sub>t<sub>.</sub> Al<sub>gor</sub>ith<sub>m</sub> 3 <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> KG <sub>proce</sub>d<sub>ure.</sub>

Algorithm 3 Knowled<sub>g</sub>e <sub>g</sub>radient (KG)   
In<sub>p</sub>ut: $\mu _ { t - 1 } , \Sigma _ { { t - 1 } } , R _ { t } \geq \operatorname* { m i n } ( r _ { 0 } , r _ { 1 } )$   
1<sub>:</sub> C<sub>ompu</sub>t<sub>e</sub> $\psi _ { t , i j } ^ { \mathrm { K G } }$ f<sub>or</sub> <sub>a</sub>ll $( i , j ) \in \mathcal { I }$ <sub>w</sub>ith $r _ { J } \leq R _ { t }$   
2<sub>:</sub> $( i , j ) \mathop  { \longleftarrow } \arg \operatorname* { m a x } _ { ( i , j ) \in \mathcal { I } : r _ { J } \leq R _ { t } } \psi _ { t , i j } ^ { \mathrm { K G } }$   
3<sub>:</sub> if $\boldsymbol { \psi } _ { t , i j } ^ { \mathrm { K G } } = 0$   
4<sub>:</sub> Choose an afordable (�, �) using the fallback rule   
5: return (�, �)

## 5.2. Endgame when?

U<sub>p</sub> t<sub>o</sub> thi<sub>s po</sub>i<sub>n</sub>t<sub>,</sub> Al<sub>gor</sub>ith<sub>m</sub> 2 <sub>prov</sub>id<sub>es</sub> th<sub>e genera</sub>l <sub>sequen</sub>ti<sub>a</sub>l <sub>proce</sub>d<sub>ure, an</sub>d <sub>we</sub> h<sub>ave g</sub>i<sub>ven</sub> t<sub>wo a</sub>l<sub>gor</sub>ith<sub>ms</sub> f<sub>or</sub> its EvaluationRule: SA (Al<sub>g</sub>orithm 1) and KG (Al<sub>g</sub>orithm 3). What remains is a rule for when to use which. SA <sub>was</sub> d<sub>es</sub>i<sub>gne</sub>d f<sub>or</sub> th<sub>e</sub> b<sub>u</sub>lk <sub>o</sub>f th<sub>e eva</sub>l<sub>ua</sub>ti<sub>on process an</sub>d KG f<sub>or</sub> th<sub>e en</sub>d<sub>game, so we cons</sub>id<sub>er</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>c</sub>l<sub>ass o</sub>f <sub>po</sub>li<sub>c</sub>i<sub>es: run</sub> SA <sub>un</sub>til <sub>a pre</sub>d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d <sub>s</sub>h<sub>are � o</sub>f th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t h<sub>as</sub> b<sub>een spen</sub>t<sub>,</sub> th<sub>en sw</sub>it<sub>c</sub>h t<sub>o</sub> $\mathrm { K G } ^ { 2 }$ We <sub>re</sub>f<sub>er</sub> t<sub>o</sub> <sub>�</sub> <sub>as</sub> th<sub>e</sub> <sub>sw</sub>it<sub>c</sub>hi<sub>ng</sub> <sub>po</sub>i<sub>n</sub>t<sub>.</sub> Withi<sub>n</sub> thi<sub>s</sub> <sub>c</sub>l<sub>ass,</sub> <sub>we</sub> <sub>c</sub>h<sub>oose</sub> th<sub>e</sub> <sub>sw</sub>it<sub>c</sub>hi<sub>ng</sub> <sub>po</sub>i<sub>n</sub>t <sub>v</sub>i<sub>a</sub> M<sub>on</sub>t<sub>e</sub> C<sub>ar</sub>l<sub>o</sub> <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on.</sub> F<sub>or</sub> $\nu \in [ 0 , 1 ]$ <sub>,</sub> l<sub>e</sub>t $\pi _ { \nu }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e po</sub>li<sub>cy o</sub>bt<sub>a</sub>i<sub>ne</sub>d f<sub>rom</sub> Al<sub>gor</sub>ith<sub>m</sub> 2 <sub>us</sub>i<sub>ng</sub> E<sub>valuation</sub>R<sub>ul</sub> $\mathbf { E } ^ { R _ { t } } = \mathbf { S } \mathbf { A }$ <sub>w</sub>hil<sub>e</sub> $R - R _ { t } < \nu R$ <sub>an</sub>d E<sub>valuation</sub>R<sub>ul</sub> $\mathbf { E } ^ { R _ { t } } = \mathbf { K } \mathbf { G }$ once $R - R _ { t } \geq \nu R$ <sub>, so</sub> th<sub>a</sub>t $\pi _ { 1 }$ i<sub>s</sub> SA th<sub>roug</sub>h<sub>ou</sub>t <sub>an</sub>d $\pi _ { 0 }$ i<sub>s</sub> KG th<sub>roug</sub>h<sub>ou</sub>t<sub>.</sub> Gi<sub>ven</sub> <sub>a</sub> <sub>pro</sub>bl<sub>em</sub> <sub>spec</sub>ifi<sub>ca</sub>ti<sub>on</sub> $( I , k , R , r _ { 0 } , r _ { 1 } , p _ { 0 } , \beta _ { 0 } , \beta _ { 1 } , \theta _ { 0 } )$ <sub>,</sub> l<sub>e</sub>t

$$
\operatorname { E O C } \big ( S _ { k } ^ { * } ( \mu _ { T } ) ; \pi _ { \nu } \big ) : = \mathbb { E } [ \sum _ { i \in S _ { k } ^ { * } ( \pmb { \theta } ) } \theta _ { i } - \sum _ { i \in S _ { k } ^ { * } ( \mu _ { T } ) } \theta _ { i } ] \pi _ { \nu } ] ,
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> <sub>expec</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>s</sub> <sub>over</sub> $\pmb \theta$ <sub>w</sub>ith <sub>pr</sub>i<sub>or</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $p _ { 0 }$ <sub>an</sub>d <sub>over</sub> th<sub>e</sub> <sub>ou</sub>t<sub>comes</sub> $X _ { 1 } , \ldots , X _ { T }$ d<sub>rawn</sub> f<sub>rom</sub> th<sub>e</sub> lik<sub>e</sub>lih<sub>oo</sub>d <sub>un</sub>d<sub>er</sub> <sub>po</sub>li<sub>cy</sub> $\pi _ { \nu }$

W<sub>e</sub> <sub>c</sub>h<sub>oose</sub> th<sub>e</sub> <sub>sw</sub>it<sub>c</sub>hi<sub>ng</sub> <sub>po</sub>i<sub>n</sub>t b<sub>y</sub> <sub>a</sub> <sub>marg</sub>i<sub>na</sub>l <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>over</sub> <sub>a</sub> fi<sub>xe</sub>d <sub>se</sub>t <sub>o</sub>f <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>sw</sub>it<sub>c</sub>hi<sub>ng</sub> <sub>po</sub>i<sub>n</sub>t<sub>s</sub> $0 \leq \nu _ { 1 } <$ $\nu _ { 2 } < \dots < \nu _ { J } \leq 1$ . Let EOC $\left( S _ { k } ^ { \ast } ( \pmb { \mu } _ { T } ) ; \pmb { \pi } _ { \nu } \right)$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> M<sub>on</sub>t<sub>e</sub> C<sub>ar</sub>l<sub>o es</sub>ti<sub>ma</sub>t<sub>e o</sub>f EOC $\left( S _ { k } ^ { \ast } ( \pmb { \mu } _ { T } ) ; \pmb { \pi } _ { \nu } \right)$ <sub>, o</sub>bt<sub>a</sub>i<sub>ne</sub>d b<sub>y</sub> <sub>averag</sub>i<sub>ng</sub> th<sub>e rea</sub>li<sub>ze</sub>d <sub>oppor</sub>t<sub>un</sub>it<sub>y cos</sub>t $\begin{array} { r } { \sum _ { i \in S _ { k } ^ { * } ( \pmb { \theta } ) } \theta _ { i } - \sum _ { i \in S _ { k } ^ { * } ( \pmb { \mu } _ { T } ) } \theta _ { i } } \end{array}$ <sub>over</sub> � i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>runs</sub> <sub>o</sub>f $\pi _ { \nu }$ <sub>eac</sub>h d<sub>raw</sub>i<sub>ng</sub> $\pmb { \theta } \sim p _ { 0 }$ <sub>an</sub>d th<sub>e</sub> <sub>ou</sub>t<sub>comes</sub> f<sub>rom</sub> th<sub>e</sub> <sub>spec</sub>ifi<sub>e</sub>d lik<sub>e</sub>lih<sub>oo</sub>d<sub>.</sub> St<sub>ar</sub>ti<sub>ng</sub> f<sub>rom</sub> $\nu _ { J } \left( \nu _ { J } = 1 \right.$ <sub>mean</sub>i<sub>ng</sub> SA all the wa<sub>y</sub> throu<sub>g</sub>h), we move the switchin<sub>g</sub> <sub>p</sub>oint back one candidate at a time, from $\nu _ { h }$ to $\nu _ { h - 1 }$ (<sub>p</sub>eelin<sub>g</sub> back the last stretch of SA into KG), as long as the marginal reduction EOC $\left( S _ { k } ^ { \ast } ( \pmb { \mu } _ { T } ) ; \pi _ { \nu _ { h } } \right) - \widehat { \mathrm { E O C } } \left( S _ { k } ^ { \ast } ( \pmb { \mu } _ { T } ) ; \pi _ { \nu _ { h - 1 } } \right)$ <sub>rema</sub>i<sub>ns pos</sub>iti<sub>ve.</sub> W<sub>e s</sub>t<sub>op a</sub>t th<sub>e</sub> fi<sub>rs</sub>t $\nu ^ { * }$ at which peeling back further fails to reduce EOC, and deploy th<sub>e</sub> <sub>po</sub>li<sub>cy</sub> $\pi _ { \nu ^ { * } }$ <sub>.</sub> W<sub>e</sub> <sub>w</sub>ill <sub>re</sub>f<sub>er</sub> t<sub>o</sub> thi<sub>s</sub> <sub>as</sub> th<sub>e</sub> <sub>sw</sub>it<sub>c</sub>hi<sub>ng</sub> <sub>proce</sub>d<sub>ure,</sub> <sub>an</sub>d t<sub>o</sub> <sub>our</sub> <sub>me</sub>th<sub>o</sub>d<sub>,</sub> th<sub>e</sub> <sub>po</sub>li<sub>cy</sub> $\pi _ { \nu ^ { * } }$ <sub>,</sub> <sub>as</sub> SE (Surro<sub>g</sub>ate to End<sub>g</sub>ame).

## 6. Batching extension

T<sub>o</sub> <sub>conc</sub>l<sub>u</sub>d<sub>e</sub> th<sub>e</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s,</sub> <sub>we</sub> <sub>cons</sub>id<sub>er</sub> <sub>se</sub>tti<sub>ngs</sub> <sub>w</sub>h<sub>ere</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> <sub>are</sub> <sub>ass</sub>i<sub>gne</sub>d i<sub>n</sub> b<sub>a</sub>t<sub>c</sub>h<sub>es</sub> <sub>w</sub>h<sub>ose</sub> <sub>ou</sub>t<sub>comes</sub> <sub>are</sub> <sub>o</sub>b<sub>serve</sub>d <sub>a</sub>t <sub>once</sub> b<sub>e</sub>f<sub>ore</sub> th<sub>e</sub> <sub>nex</sub>t b<sub>a</sub>t<sub>c</sub>h i<sub>s</sub> <sub>c</sub>h<sub>osen.</sub> Whil<sub>e</sub> <sub>we</sub> h<sub>ave</sub> <sub>presen</sub>t<sub>e</sub>d th<sub>e</sub> <sub>pro</sub>bl<sub>em</sub> <sub>sequen</sub>ti<sub>a</sub>ll<sub>y,</sub> th<sub>e</sub> <sub>mo</sub>d<sub>e</sub>l <sub>an</sub>d <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> f<sub>or</sub> th<sub>e</sub> <sub>surroga</sub>t<sub>e</sub> <sub>were</sub> d<sub>eve</sub>l<sub>ope</sub>d <sub>w</sub>ith b<sub>a</sub>t<sub>c</sub>hi<sub>ng</sub> i<sub>n</sub> <sub>m</sub>i<sub>n</sub>d<sub>,</sub> <sub>an</sub>d <sub>ex</sub>t<sub>en</sub>d <sub>na</sub>t<sub>ura</sub>ll<sub>y.</sub>

For the batching setting, let � denote the fixed number of rounds and let $B _ { t } \geq \operatorname* { m i n } \{ r _ { 0 } , r _ { 1 } \}$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t f<sub>or roun</sub>d �<sub>, w</sub>ith $\begin{array} { r } { \sum _ { t = 1 } ^ { T } B _ { t } = R } \end{array}$ <sub>.</sub> L<sub>e</sub>t $n ( t )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h <sub>c</sub>h<sub>osen</sub> i<sub>n</sub> thi<sub>s roun</sub>d<sub>, w</sub>h<sub>ere</sub> $n _ { i j } ( t )$ i<sub>s</sub> th<sub>e num</sub>b<sub>er o</sub>f <sub>eva</sub>l<sub>ua</sub>ti<sub>ons o</sub>f $( i , j )$

## 6.1. Surrogate batch construction

Ch<sub>oos</sub>i<sub>ng</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h th<sub>a</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub> th<sub>e</sub> <sub>surroga</sub>t<sub>e</sub> l<sub>oss</sub> i<sub>n</sub> <sub>eac</sub>h <sub>roun</sub>d i<sub>s</sub> <sub>an</sub> i<sub>n</sub>t<sub>eger</sub> <sub>program</sub> th<sub>a</sub>t <sub>qu</sub>i<sub>c</sub>kl<sub>y</sub> b<sub>ecomes</sub> i<sub>n</sub>t<sub>rac</sub>t<sub>a</sub>bl<sub>e</sub> <sub>as</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h b<sub>u</sub>d<sub>ge</sub>t <sub>grows.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore</sub> b<sub>u</sub>ild <sub>a</sub> b<sub>a</sub>t<sub>c</sub>h b<sub>y</sub> <sub>comm</sub>itti<sub>ng</sub> t<sub>o</sub> <sub>one</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>a</sub>t <sub>a</sub> ti<sub>me.</sub> Th<sub>e</sub> fi<sub>rs</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> i<sub>s c</sub>h<sub>osen as</sub> i<sub>n</sub> th<sub>e</sub> f<sub>u</sub>ll<sub>y sequen</sub>ti<sub>a</sub>l <sub>pro</sub>bl<sub>em,</sub> b<sub>y max</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> th<sub>e marg</sub>i<sub>na</sub>l d<sub>ecrease</sub> in surro<sub>g</sub>ate loss <sub>p</sub>er unit of resource from §4.1. However<sub>,</sub> unlike the se<sub>q</sub>uential <sub>p</sub>roblem<sub>,</sub> we must choose <sub>su</sub>b<sub>sequen</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> b<sub>e</sub>f<sub>ore</sub> <sub>o</sub>b<sub>serv</sub>i<sub>ng</sub> th<sub>e</sub> <sub>ou</sub>t<sub>comes</sub> <sub>o</sub>f th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>a</sub>l<sub>rea</sub>d<sub>y</sub> <sub>comm</sub>itt<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h<sub>.</sub> If <sub>we</sub> <sub>con</sub>ti<sub>nue</sub>d t<sub>o</sub> <sub>use</sub> th<sub>e</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>marg</sub>i<sub>na</sub>l <sub>scores,</sub> <sub>we</sub> <sub>wou</sub>ld <sub>va</sub>l<sub>ue</sub> <sub>eac</sub>h <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>as</sub> if it <sub>were</sub> th<sub>e</sub> <sub>on</sub>l<sub>y</sub> <sub>one</sub> i<sub>n</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h<sub>,</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>ng</sub> <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> t<sub>o</sub> <sub>re</sub>d<sub>uce</sub> th<sub>e</sub> <sub>same</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y.</sub>

C<sub>ons</sub>id<sub>er</sub> th<sub>e</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>o</sub>f th<sub>e</sub> <sub>secon</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on.</sub> S<sub>uppose</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>comm</sub>itt<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h i<sub>s</sub> $( i , j )$ Alth<sub>oug</sub>h <sub>we</sub> h<sub>ave</sub> <sub>no</sub>t <sub>ye</sub>t <sub>o</sub>b<sub>serve</sub>d it<sub>s</sub> <sub>ou</sub>t<sub>come,</sub> <sub>we</sub> <sub>can</sub> <sub>accoun</sub>t f<sub>or</sub> th<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>we</sub> <sub>expec</sub>t t<sub>o</sub> l<sub>earn</sub> f<sub>rom</sub> it th<sub>roug</sub>h it<sub>s</sub> <sub>expec</sub>t<sub>e</sub>d <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>covar</sub>i<sub>ance.</sub> B<sub>y</sub> L<sub>emma</sub> 2<sub>,</sub>

$$
\overline { { \Sigma } } _ { t } ( ( i , j ) , p _ { t - 1 } ) = \Sigma _ { t - 1 } - \frac { \delta _ { t , i j } } { \sigma _ { t - 1 , i j } ^ { 2 } } \Sigma _ { t - 1 } ( e _ { i } - e _ { j } ) ( e _ { i } - e _ { j } ) ^ { \top } \Sigma _ { t - 1 } .
$$

This covariance reflects the uncertainty expected to remain after the first evaluation. To choose the second <sub>eva</sub>l<sub>ua</sub>ti<sub>on,</sub> <sub>we</sub> <sub>use</sub> thi<sub>s</sub> <sub>covar</sub>i<sub>ance</sub> i<sub>n</sub> <sub>p</sub>l<sub>ace</sub> <sub>o</sub>f $\Sigma _ { t - 1 }$ and recom<sub>p</sub>ute the mar<sub>g</sub>inal scores from §4.1<sub>,</sub> kee<sub>p</sub>in<sub>g</sub> $\pmb { \mu } _ { t - 1 }$ <sub>an</sub>d $\left( q _ { t } , b _ { t } \right)$ fixed. Each candidate evaluation is scored by its additional reduction in surrogate loss, <sub>accoun</sub>ti<sub>ng</sub> f<sub>or</sub> th<sub>e</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on we a</sub>l<sub>rea</sub>d<sub>y expec</sub>t t<sub>o</sub> l<sub>earn</sub> f<sub>rom</sub> th<sub>e</sub> fi<sub>rs</sub>t<sub>.</sub> W<sub>e c</sub>h<sub>oose</sub> th<sub>e eva</sub>l<sub>ua</sub>ti<sub>on w</sub>ith th<sub>e</sub> hi<sub>g</sub>h<sub>es</sub>t <sub>marg</sub>i<sub>na</sub>l <sub>score</sub> <sub>per</sub> <sub>un</sub>it <sub>o</sub>f <sub>resource.</sub>

W<sub>e</sub> <sub>nex</sub>t <sub>ex</sub>t<sub>en</sub>d thi<sub>s</sub> <sub>approac</sub>h t<sub>o</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>ng</sub> <sub>an</sub> <sub>en</sub>ti<sub>re</sub> b<sub>a</sub>t<sub>c</sub>h<sub>.</sub> Th<sub>e</sub> <sub>comp</sub>li<sub>ca</sub>ti<sub>on</sub> i<sub>s</sub> th<sub>a</sub>t <sub>compu</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>pos</sub>t<sub>er</sub>i<sub>or covar</sub>i<sub>ance a</sub>ft<sub>er severa</sub>l <sub>comm</sub>itt<sub>e</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>ons requ</sub>i<sub>res</sub> t<sub>a</sub>ki<sub>ng</sub> th<sub>e we</sub>i<sub>g</sub>ht<sub>e</sub>d <sub>average over</sub> their (man<sub>y</sub>) <sub>p</sub>ossible joint outcomes. Instead, we a<sub>pp</sub>roximate it b<sub>y</sub> re<sub>p</sub>eatedl<sub>y</sub> a<sub>pp</sub>l<sub>y</sub>in<sub>g</sub> the sin<sub>g</sub>le-evaluation <sub>expec</sub>t<sub>e</sub>d <sub>covar</sub>i<sub>ance up</sub>d<sub>a</sub>t<sub>e, us</sub>i<sub>ng</sub> th<sub>e up</sub>d<sub>a</sub>t<sub>e</sub>d <sub>expec</sub>t<sub>e</sub>d <sub>covar</sub>i<sub>ance</sub> f<sub>rom</sub> th<sub>e eva</sub>l<sub>ua</sub>ti<sub>ons a</sub>l<sub>rea</sub>d<sub>y comm</sub>itt<sub>e</sub>d<sub>.</sub>

F<sub>orma</sub>ll<sub>y,</sub> l<sub>e</sub>t <sub>�</sub> d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> <sub>recor</sub>di<sub>ng</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> <sub>comm</sub>itt<sub>e</sub>d <sub>so</sub> f<sub>ar</sub> i<sub>n</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h<sub>,</sub> <sub>w</sub>h<sub>ere</sub> $n _ { i j }$ <sub>correspon</sub>d<sub>s</sub> t<sub>o</sub> th<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> <sub>o</sub>f $( i , j )$ <sub>.</sub> D<sub>e</sub>fi<sub>ne</sub> $F ^ { i j }$ <sub>as</sub> <sub>a</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> <sub>o</sub>f th<sub>e</sub> <sub>same</sub> <sub>s</sub>i<sub>ze</sub> <sub>as</sub> $n ,$ <sub>w</sub>ith $F _ { i j } ^ { i j } = 1$ <sub>an</sub>d <sub>zero e</sub>l<sub>sew</sub>h<sub>ere.</sub> L<sub>e</sub>t $\Sigma _ { t } ^ { \prime } ( n )$ d<sub>eno</sub>t<sub>e our approx</sub>i<sub>ma</sub>ti<sub>on o</sub>f th<sub>e expec</sub>t<sub>e</sub>d <sub>pos</sub>t<sub>er</sub>i<sub>or covar</sub>i<sub>ance, w</sub>ith $\Sigma _ { t } ^ { \prime } ( 0 ) = \Sigma _ { t - 1 }$ <sub>.</sub> Aft<sub>er</sub> <sub>a</sub>ddi<sub>ng</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> $( i , j )$ t<sub>o</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h<sub>,</sub> it <sub>up</sub>d<sub>a</sub>t<sub>es</sub> <sub>accor</sub>di<sub>ng</sub> t<sub>o</sub>

$$
\Sigma _ { t } ^ { \prime } ( n + F ^ { i j } ) = \Sigma _ { t } ^ { \prime } ( n ) - \frac { \delta _ { t , i j } ^ { \prime } ( n ) } { ( \sigma _ { t , i j } ^ { \prime } ( n ) ) ^ { 2 } } \Sigma _ { t } ^ { \prime } ( n ) ( e _ { i } - e _ { j } ) ( e _ { i } - e _ { j } ) ^ { \top } \Sigma _ { t } ^ { \prime } ( n ) ,\tag{12}
$$

<sub>w</sub>h<sub>ere</sub> $( \sigma _ { t , i j } ^ { \prime } ( n ) ) ^ { 2 }$ <sub>an</sub>d $\delta _ { t , i j } ^ { \prime } ( n )$ <sub>are</sub> <sub>compu</sub>t<sub>e</sub>d <sub>as</sub> i<sub>n</sub> L<sub>emma</sub> 2<sub>,</sub> <sub>us</sub>i<sub>ng</sub> $\Sigma _ { t } ^ { \prime } ( n )$ i<sub>n p</sub>l<sub>ace o</sub>f $\Sigma _ { t - 1 }$

L<sub>e</sub>t $\begin{array} { r } { \mathcal { L } ( 0 ; \Sigma _ { t } ^ { \prime } ( n ) ) } \end{array}$ ) be the surrogate loss before adding another evaluation and $\mathcal { L } ( ( i , j ) ; \Sigma _ { t } ^ { \prime } ( n ) )$ b<sub>e</sub> th<sub>e</sub> l<sub>oss</sub> after adding (�, �). The marginal score of $( i , j )$ i<sub>s</sub>

$$
\psi _ { t , i j } ( n ) = \frac { \mathcal { L } ( 0 ; \Sigma _ { t } ^ { \prime } ( n ) ) - \mathcal { L } ( ( i , j ) ; \Sigma _ { t } ^ { \prime } ( n ) ) } { r _ { J } } .\tag{13}
$$

U<sub>s</sub>i<sub>ng</sub> th<sub>ese scores,</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h<sub>-</sub>b<sub>u</sub>ildi<sub>ng proce</sub>d<sub>ure</sub> i<sub>s g</sub>i<sub>ven</sub> i<sub>n</sub> Al<sub>gor</sub>ith<sub>m</sub> 4<sub>.</sub> W<sub>e cons</sub>id<sub>er e</sub>l<sub>as</sub>ti<sub>c roun</sub>d b<sub>u</sub>d<sub>ge</sub>t<sub>s,</sub> <sub>mean</sub>i<sub>ng</sub> th<sub>a</sub>t <sub>as</sub> l<sub>ong as some</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>rema</sub>i<sub>ns</sub> i<sub>n</sub> th<sub>e curren</sub>t <sub>roun</sub>d<sub>,</sub> th<sub>e a</sub>l<sub>gor</sub>ith<sub>m may c</sub>h<sub>oose any eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>prov</sub>id<sub>e</sub>d it <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e</sub> t<sub>o</sub>t<sub>a</sub>l b<sub>u</sub>d<sub>ge</sub>t <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>.</sub> If it<sub>s</sub> <sub>cos</sub>t <sub>excee</sub>d<sub>s</sub> th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> <sub>roun</sub>d b<sub>u</sub>d<sub>ge</sub>t<sub>,</sub> th<sub>e</sub> <sub>excess</sub> i<sub>s</sub> d<sub>e</sub>d<sub>uc</sub>t<sub>e</sub>d f<sub>rom</sub> th<sub>e nex</sub>t <sub>roun</sub>d’<sub>s</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub> Thi<sub>s ensures</sub> th<sub>a</sub>t th<sub>e</sub> k<sub>napsac</sub>k <sub>na</sub>t<sub>ure o</sub>f th<sub>e roun</sub>d b<sub>u</sub>d<sub>ge</sub>t d<sub>oes no</sub>t bi<sub>as</sub> th<sub>e</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>o</sub>f <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>ype.</sub>

Al<sub>gor</sub>ith<sub>m</sub> 4 S<sub>urroga</sub>t<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub>   
In<sub>p</sub>ut: $\mu _ { t - 1 } , \Sigma _ { t - 1 } ,$ <sub>, roun</sub>d b<sub>u</sub>d<sub>ge</sub>t $\begin{array} { r } { \overline { { B _ { t } , } } } \end{array}$ <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> t<sub>o</sub>t<sub>a</sub>l b<sub>u</sub>d<sub>ge</sub>t $R _ { \mathrm { r e m } }$ (initiall<sub>y</sub> ${ \overline { { R ) } } } ,$ , and excess from the <sub>p</sub>revious round (initiall<sub>y</sub> zero)   
1<sub>:</sub> $B _ { \mathrm { r e m } } \gets B _ { t } -$ excess   
2<sub>:</sub> $n \gets 0 , \quad \Sigma _ { t } ^ { \prime } ( 0 ) \gets \Sigma _ { t - 1 }$   
3<sub>:</sub> C<sub>o</sub>m<sub>pu</sub>t<sub>e</sub> $\left( q _ { t } , b _ { t } \right)$   
4<sub>:</sub> <sub>w</sub>hil<sub>e</sub> $B _ { \mathrm { r e m } } > 0$ and $R _ { \mathrm { r e m } } \geq \operatorname* { m i n } ( r _ { 0 } , r _ { 1 } )$   
5<sub>:</sub> <sup>C</sup>om<sub>p</sub>ute $\psi _ { t , i j } ( n )$ b<sub>y</sub> (13) for all $( i , j ) \in { \cal { I } }$ <sub>w</sub>ith $r _ { J } \le R _ { \mathrm { r e m } }$   
6<sub>:</sub> Ch<sub>oose</sub> $( i , j )$ ∈ ar<sub>g</sub> max $( i , j ) \in \bar { \mathcal { I } } { : r _ { J } } \le { R _ { \mathrm { r e m } } } \psi _ { t , i j } \left( n \right)$   
7<sub>:</sub> U<sub>p</sub>d<sub>a</sub>t<sub>e</sub> th<sub>e covar</sub>i<sub>ance</sub> f<sub>rom</sub> $\Sigma _ { t } ^ { \prime } ( n )$ to $\Sigma _ { t } ^ { \prime } ( n + F ^ { i j } )$ b<sub>y</sub> (12)   
8<sub>:</sub> $\hat { n  n } + F ^ { i j }$   
9: $B _ { \mathrm { r e m } }  B _ { \mathrm { r e m } } - r _ { J } , \quad R _ { \mathrm { r e m } }  R _ { \mathrm { r e m } } - r _ { J }$   
10: excess ← max $\{ 0 , - B _ { \mathrm { r e m } } \}$   
11<sub>: re</sub>t<sub>urn</sub> $n ( t ) = n , R _ { \mathrm { r e m } } ,$ <sub>an</sub>d <sub>excess</sub>

## 6.2. The batched endgame

E<sub>x</sub>t<sub>en</sub>di<sub>ng</sub> th<sub>e</sub> KG <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> t<sub>o</sub> b<sub>a</sub>t<sub>c</sub>h<sub>es</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uces</sub> t<sub>wo</sub> <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l difi<sub>cu</sub>lti<sub>es.</sub> Fi<sub>rs</sub>t<sub>,</sub> it <sub>op</sub>ti<sub>m</sub>i<sub>zes</sub> <sub>over</sub> <sub>p</sub>ossible batch allocations<sub>,</sub> which we address throu<sub>g</sub>h se<sub>q</sub>uential batch construction<sub>,</sub> as in §6.1. Second<sub>,</sub> <sub>compu</sub>ti<sub>ng</sub> th<sub>e myop</sub>i<sub>c va</sub>l<sub>ue o</sub>f <sub>a g</sub>i<sub>ven</sub> b<sub>a</sub>t<sub>c</sub>h <sub>requ</sub>i<sub>res</sub> th<sub>e ou</sub>t<sub>come pro</sub>b<sub>a</sub>biliti<sub>es an</sub>d <sub>pos</sub>t<sub>er</sub>i<sub>or means</sub> f<sub>or</sub> <sub>every</sub> <sub>poss</sub>ibl<sub>e</sub> <sub>com</sub>bi<sub>na</sub>ti<sub>on</sub> <sub>o</sub>f b<sub>a</sub>t<sub>c</sub>h <sub>ou</sub>t<sub>comes.</sub> W<sub>e</sub> d<sub>er</sub>i<sub>ve</sub> <sub>exp</sub>li<sub>c</sub>it <sub>express</sub>i<sub>ons</sub> f<sub>or</sub> th<sub>ese</sub> <sub>quan</sub>titi<sub>es</sub> <sub>g</sub>i<sub>ven</sub> <sub>a</sub> batch outcome (analo<sub>g</sub>ous to Lemma 3), but the<sub>y</sub> involve multivariate normal CDFs whose dimension <sub>g</sub>rows with the batch size, makin<sub>g</sub> them ex<sub>p</sub>ensive to evaluate numericall<sub>y</sub> (Genz and Bretz 2009). Moreover, a b<sub>a</sub>t<sub>c</sub>h <sub>o</sub>f � <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> h<sub>as</sub> $2 ^ { U }$ <sub>poss</sub>ibl<sub>e</sub> <sub>ou</sub>t<sub>comes,</sub> <sub>eac</sub>h <sub>requ</sub>i<sub>r</sub>i<sub>ng</sub> th<sub>ese</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>ons.</sub>

The computational dificulty comes from considering all the ways the evaluations in a batch can jointly <sub>move</sub> th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>means.</sub> W<sub>e</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub> th<sub>e</sub> <sub>myop</sub>i<sub>c</sub> <sub>va</sub>l<sub>ue</sub> <sub>o</sub>f <sub>a</sub> b<sub>a</sub>t<sub>c</sub>h b<sub>y</sub> d<sub>ecompos</sub>i<sub>ng</sub> it i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> <sub>myop</sub>i<sub>c</sub> <sub>va</sub>l<sub>ues</sub> <sub>o</sub>f <sub>su</sub>b<sub>-</sub>b<sub>a</sub>t<sub>c</sub>h<sub>es</sub> <sub>o</sub>f <sub>a</sub>t <sub>mos</sub>t <sub>�</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons.</sub> Thi<sub>s</sub> <sub>re</sub>t<sub>a</sub>i<sub>ns</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> <sub>e</sub>f<sub>ec</sub>t<sub>s</sub> <sub>among</sub> <sub>up</sub> t<sub>o</sub> <sub>�</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> <sub>w</sub>hil<sub>e</sub> k<sub>eep</sub>i<sub>ng</sub> th<sub>e</sub> <sub>compu</sub>t<sub>a</sub>ti<sub>on</sub> t<sub>rac</sub>t<sub>a</sub>bl<sub>e:</sub> th<sub>e</sub> <sub>va</sub>l<sub>ue</sub> <sub>o</sub>f <sub>eac</sub>h <sub>su</sub>b<sub>-</sub>b<sub>a</sub>t<sub>c</sub>h <sub>requ</sub>i<sub>res</sub> <sub>norma</sub>l CDF<sub>s</sub> <sub>o</sub>f di<sub>mens</sub>i<sub>on</sub> <sub>a</sub>t <sub>mos</sub>t <sub>�</sub> <sub>an</sub>d <sub>a</sub> <sub>sum</sub> <sub>over</sub> <sub>a</sub>t <sub>mos</sub>t $2 ^ { w }$ <sub>ou</sub>t<sub>comes.</sub> B<sub>ecause</sub> th<sub>e</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s</sub> <sub>are</sub> t<sub>ec</sub>h<sub>n</sub>i<sub>ca</sub>l <sub>an</sub>d <sub>no</sub>t th<sub>e</sub> f<sub>ocus</sub> <sub>o</sub>f thi<sub>s</sub> <sub>paper,</sub> <sub>we</sub> l<sub>eave</sub> th<sub>e</sub> f<sub>u</sub>ll d<sub>ecompos</sub>iti<sub>on</sub> <sub>an</sub>d b<sub>a</sub>t<sub>c</sub>h <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> <sub>proce</sub>d<sub>ure,</sub> <sub>a</sub>l<sub>ong</sub> <sub>w</sub>ith <sub>a</sub>dditi<sub>ona</sub>l d<sub>e</sub>t<sub>a</sub>il<sub>s</sub> <sub>on</sub> <sub>surroga</sub>t<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h <sub>a</sub>ll<sub>oca</sub>ti<sub>on,</sub> t<sub>o</sub> th<sub>e</sub> A<sub>ppen</sub>di<sub>x.</sub>

## 7. Numerical experiments

Th<sub>e</sub> <sub>numer</sub>i<sub>ca</sub>l <sub>exper</sub>i<sub>men</sub>t<sub>s</sub> <sub>serve</sub> t<sub>wo</sub> <sub>purposes:</sub> t<sub>o</sub> ill<sub>us</sub>t<sub>ra</sub>t<sub>e</sub> <sub>w</sub>h<sub>a</sub>t th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> d<sub>oes,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>supp</sub>l<sub>emen</sub>t<sub>s</sub> the insi<sub>g</sub>hts from the anal<sub>y</sub>sis<sub>,</sub> and to investi<sub>g</sub>ate how well it <sub>p</sub>erforms. §7.2 serves the first<sub>,</sub> showin<sub>g</sub> how <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> <sub>are</sub> <sub>a</sub>ll<sub>oca</sub>t<sub>e</sub>d <sub>as</sub> <sub>we</sub> l<sub>earn</sub> <sub>more,</sub> h<sub>ow</sub> th<sub>e</sub> <sub>m</sub>i<sub>x</sub> <sub>o</sub>f <sub>ra</sub>ti<sub>ngs</sub> <sub>an</sub>d <sub>compar</sub>i<sub>sons</sub> <sub>s</sub>hift<sub>s</sub> <sub>over</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>,</sub> and how these are influenced b<sub>y</sub> the end<sub>g</sub>ame. §7.3 serves both<sub>,</sub> com<sub>p</sub>arin<sub>g</sub> our method to versions of itself <sub>res</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d t<sub>o one eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>ype or</sub> t<sub>o a</sub> fi<sub>xe</sub>d <sub>sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>e o</sub>f th<sub>e</sub> t<sub>wo, w</sub>hi<sub>c</sub>h t<sub>es</sub>t<sub>s</sub> th<sub>e va</sub>l<sub>ue o</sub>f <sub>com</sub>bi<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>wo</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>ypes</sub> <sub>an</sub>d h<sub>ow</sub> <sub>we</sub>ll th<sub>e</sub> <sub>me</sub>th<sub>o</sub>d b<sub>a</sub>l<sub>ances</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>em.</sub> T<sub>o</sub> f<sub>ur</sub>th<sub>er</sub> i<sub>nves</sub>ti<sub>ga</sub>t<sub>e</sub> <sub>our</sub> method’s <sub>p</sub>erformance<sub>,</sub> §7.4 com<sub>p</sub>ares it to external (and internal) benchmarks<sub>,</sub> §7.5 stress-tests the Gaussian <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> <sub>an</sub>d th<sub>e</sub> <sub>me</sub>th<sub>o</sub>d <sub>un</sub>d<sub>er</sub> <sub>m</sub>i<sub>sspec</sub>ifi<sub>ca</sub>ti<sub>on,</sub> <sub>an</sub>d $\ S 7 . 6$ <sub>measures</sub> th<sub>e</sub> i<sub>mpac</sub>t <sub>o</sub>f b<sub>a</sub>t<sub>c</sub>hi<sub>ng.</sub>

## 7.1. Setup

Configurations. We consider $I \in \{ 1 0 , 2 0 , 4 0 , 8 0 , 1 6 0 \}$ <sub>an</sub>d <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> <sub>s</sub>i<sub>zes</sub> $k \in \{ 1 , \sqrt { 0 . 1 I } , 0 . 1 I \}$ (rounded). Th<sub>e pr</sub>i<sub>or</sub> i<sub>s exc</sub>h<sub>angea</sub>bl<sub>e w</sub>ith $\pmb { \mu } _ { 0 } = \mathbf { 0 } , { \boldsymbol { \Sigma } } _ { 0 } = 1 0 \cdot \operatorname { I d } _ { I }$ <sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>ra</sub>ti<sub>ng</sub> th<sub>res</sub>h<sub>o</sub>ld i<sub>s</sub> $\theta _ { 0 } = 0$ (in ex<sub>p</sub>ectation half of the items <sub>g</sub>et thumbs-u<sub>p</sub> ratin<sub>g</sub>s). Evaluator accuracies are 75% for ratin<sub>g</sub>s and $8 5 \%$ f<sub>or</sub> <sub>compar</sub>i<sub>sons,</sub> <sub>ca</sub>lib<sub>ra</sub>t<sub>e</sub>d th<sub>roug</sub>h R<sub>emar</sub>k 1 t<sub>o</sub> $\beta _ { 0 } = 0 . 3 1 6$ <sub>an</sub>d $\beta _ { 1 } = 0 . 4 3 9$ <sub>.</sub> W<sub>e</sub> fi<sub>x</sub> th<sub>e cos</sub>t <sub>o</sub>f <sub>a ra</sub>ti<sub>ng a</sub>t $r _ { 0 } = 1$ <sub>an</sub>d <sub>cons</sub>id<sub>er</sub> com<sub>p</sub>ar<sup>i</sup>son costs $r _ { 1 } \in \{ 2 , 3 \}$ <sub>.</sub> Th<sub>e</sub> t<sub>o</sub>t<sub>a</sub>l b<sub>u</sub>d<sub>ge</sub>t i<sub>s</sub> $R = 4 I$ <sub>.</sub> Th<sub>ese</sub> <sub>are</sub> th<sub>e</sub> b<sub>ase</sub> <sub>con</sub>fi<sub>gura</sub>ti<sub>ons.</sub> L<sub>a</sub>t<sub>er</sub> <sub>su</sub>b<sub>sec</sub>ti<sub>ons</sub> <sub>a</sub>l<sub>so</sub> t<sub>es</sub>t <sub>mo</sub>difi<sub>ca</sub>ti<sub>ons o</sub>f th<sub>em</sub> t<sub>o</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>a</sub>t th<sub>e resu</sub>lt<sub>s are no</sub>t b<sub>r</sub>ittl<sub>e</sub> t<sub>o</sub> th<sub>ese c</sub>h<sub>o</sub>i<sub>ces.</sub>

Endgame. In each confi<sub>g</sub>uration<sub>,</sub> we run the switchin<sub>g p</sub>rocedure of §5.2 with $M = 1 { , } 0 0 0$ t<sub>o c</sub>h<sub>oose</sub> th<sub>e</sub> <sub>sw</sub>it<sub>c</sub>hi<sub>ng po</sub>i<sub>n</sub>t f<sub>rom</sub> SA t<sub>o</sub> KG<sub>.</sub> Th<sub>e can</sub>did<sub>a</sub>t<sub>e sw</sub>it<sub>c</sub>hi<sub>ng po</sub>i<sub>n</sub>t<sub>s are</sub> $\nu = 0 , 0 . 0 5 , \ldots , 0 . 9 5 , 1$

Policies. Alongside SE, we run modified versions of it to gain insight into the algorithm’s behavior as di<sub>scusse</sub>d <sub>a</sub>b<sub>ove.</sub> T<sub>o</sub> k<sub>eep</sub> t<sub>rac</sub>k <sub>o</sub>f th<sub>e</sub> <sub>mo</sub>difi<sub>ca</sub>ti<sub>ons</sub> <sub>o</sub>f SE<sub>,</sub> <sub>we</sub> <sub>spec</sub>if<sub>y</sub> <sub>eac</sub>h <sub>aspec</sub>t <sub>c</sub>h<sub>ange</sub>d b<sub>y</sub> <sub>an</sub> <sub>op</sub>ti<sub>ona</sub>l <sub>argumen</sub>t<sub>.</sub> Th<sub>e</sub> fi<sub>rs</sub>t <sub>op</sub>ti<sub>ona</sub>l <sub>argumen</sub>t i<sub>s</sub> th<sub>e</sub> <sub>sw</sub>it<sub>c</sub>hi<sub>ng</sub> <sub>po</sub>i<sub>n</sub>t<sub>.</sub> $\mathbf { S } \mathbf { E } ( \mathbf { s w } = \nu )$ <sub>sw</sub>it<sub>c</sub>h<sub>es</sub> f<sub>rom</sub> SA t<sub>o</sub> KG <sub>once</sub> th<sub>e</sub> <sub>s</sub>h<sub>are</sub> <sub>�</sub> <sub>o</sub>f th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t h<sub>as</sub> b<sub>een</sub> <sub>spen</sub>t<sub>,</sub> <sub>so</sub> f<sub>or</sub> <sub>examp</sub>l<sub>e,</sub> $\mathbf { S } \mathbf { E } ( \operatorname { s w } = 1 )$ i<sub>s</sub> SA<sub>.</sub> Th<sub>e</sub> <sub>secon</sub>d <sub>op</sub>ti<sub>ona</sub>l <sub>argumen</sub>t i<sub>s</sub> which evaluation t<sub>yp</sub>e is allowed at each <sub>p</sub>oint of the bud<sub>g</sub>et: rc (rate then com<sub>p</sub>are). SE itself ma<sub>y</sub> mix the two t<sub>yp</sub>es t<sup>h</sup>rou<sub>g</sub><sup>h</sup>out, w<sup>h</sup>ereas $\mathrm { S E } ( \mathrm { r c } = u )$ i<sub>s res</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d t<sub>o on</sub>l<sub>y ra</sub>ti<sub>ngs un</sub>til <sub>a propor</sub>ti<sub>on � o</sub>f th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t i<sub>s</sub> <sub>spen</sub>t <sub>an</sub>d <sub>on</sub>l<sub>y</sub> <sub>compar</sub>i<sub>sons</sub> <sub>a</sub>ft<sub>er,</sub> <sub>so</sub> f<sub>or</sub> <sub>examp</sub>l<sub>e,</sub> $\mathrm { S E } ( \mathrm { r c } = 1 )$ i<sub>s</sub> <sub>ra</sub>ti<sub>ngs</sub> <sub>on</sub>l<sub>y</sub> f<sub>or</sub> th<sub>e</sub> <sub>en</sub>ti<sub>re</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub>

T<sub>o</sub> <sub>measure</sub> <sub>per</sub>f<sub>ormance,</sub> <sub>we</sub> <sub>compare</sub> <sub>aga</sub>i<sub>ns</sub>t th<sub>ree</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s:</sub> <sub>a</sub> <sub>s</sub>li<sub>g</sub>htl<sub>y</sub> <sub>mo</sub>difi<sub>e</sub>d R<sub>ate</sub>R<sub>ank</sub> <sub>o</sub>f Li <sub>e</sub>t <sub>a</sub>l<sub>.</sub> (2018), which allocates based on m<sub>y</sub>o<sub>p</sub>ic information <sub>g</sub>ain about the to<sub>p</sub>-� set <sub>p</sub>er unit cost, the KG <sub>p</sub>olic<sub>y</sub> with random fallback when it <sub>g</sub>ets “stuck”, and KG with SA as the fallback, KG(SA), which corres<sub>p</sub>onds to $\mathrm { S E } ( \mathrm { s w } = 0 )$

Simulation protocol. For each configuration, we run 1,000 independent replicates where � is drawn from the prior and for each � the same stream of evaluation outcomes is drawn across policies. We report the avera<sub>g</sub>e realized o<sub>pp</sub>ortunit<sub>y</sub> cost, which we refer to as the EOC (technicall<sub>y</sub> this is an estimator of the EOC). P<sub>os</sub>t<sub>er</sub>i<sub>or</sub> <sub>up</sub>d<sub>a</sub>t<sub>es</sub> <sub>use</sub> <sub>var</sub>i<sub>a</sub>ti<sub>ona</sub>l i<sub>n</sub>f<sub>erence</sub> <sub>w</sub>ith <sub>a</sub> <sub>mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> <sub>var</sub>i<sub>a</sub>ti<sub>ona</sub>l f<sub>am</sub>il<sub>y.</sub> R<sub>ate</sub>R<sub>ank</sub> i<sub>s</sub> <sub>run</sub> f<sub>or</sub> $I \leq 4 0$ <sub>on</sub>l<sub>y,</sub> d<sub>ue</sub> t<sub>o</sub> it<sub>s</sub> <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>cos</sub>t<sub>.</sub> Fi<sub>gures</sub> th<sub>a</sub>t d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub>’<sub>s</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or</sub> <sub>are</sub> <sub>presen</sub>t<sub>e</sub>d <sub>on</sub> th<sub>e</sub> $I = 4 0 , k = 2 , r _ { 1 } = 3$ <sub>con</sub>fi<sub>gura</sub>ti<sub>on.</sub> Th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> fi<sub>gures</sub> f<sub>or</sub> <sub>a</sub>ll <sub>o</sub>th<sub>er</sub> <sub>con</sub>fi<sub>gura</sub>ti<sub>ons</sub> <sub>are</sub> <sub>prov</sub>id<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> A<sub>ppen</sub>di<sub>x.</sub>

## 7.2. What the policy does

T<sub>o ge</sub>t i<sub>ns</sub>i<sub>g</sub>ht i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> SA <sub>a</sub>ll<sub>oca</sub>ti<sub>ons,</sub> Fi<sub>gure</sub> 2 <sub>s</sub>h<sub>ows</sub> th<sub>e average a</sub>ll<sub>oca</sub>ti<sub>on o</sub>f SA <sub>a</sub>t <sub>eac</sub>h <sub>s</sub>t<sub>age o</sub>f th<sub>e</sub> h<sub>or</sub>i<sub>zon.</sub> Th<sub>e</sub> <sub>ra</sub>ti<sub>ng-</sub>th<sub>en-compar</sub>i<sub>son</sub> <sub>pa</sub>tt<sub>ern</sub> <sub>emerges.</sub> I<sub>n</sub> th<sub>e</sub> fi<sub>rs</sub>t h<sub>a</sub>lf <sub>o</sub>f th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>,</sub> <sub>mos</sub>tl<sub>y</sub> <sub>ra</sub>ti<sub>ngs</sub> <sub>are</sub> <sub>a</sub>ll<sub>oca</sub>t<sub>e</sub>d<sub>, near</sub>l<sub>y un</sub>if<sub>orm</sub>l<sub>y across</sub> th<sub>e</sub> it<sub>ems an</sub>d th<sub>en concen</sub>t<sub>ra</sub>ti<sub>ng</sub> t<sub>owar</sub>d<sub>s</sub> th<sub>e</sub> t<sub>op</sub> it<sub>ems.</sub> I<sub>n</sub> th<sub>e secon</sub>d h<sub>a</sub>lf<sub>,</sub> <sub>compar</sub>i<sub>sons</sub> <sub>gra</sub>d<sub>ua</sub>ll<sub>y</sub> t<sub>a</sub>k<sub>e</sub> <sub>over,</sub> <sub>concen</sub>t<sub>ra</sub>ti<sub>ng</sub> <sub>on</sub> it<sub>ems</sub> <sub>near</sub> th<sub>e</sub> t<sub>op-</sub>� b<sub>oun</sub>d<sub>ary.</sub> T<sub>o</sub> <sub>see</sub> h<sub>ow</sub> th<sub>e</sub> <sub>en</sub>d<sub>game</sub>

![](images/8d6982a7ec51019d5f19ac77e75e16688dc45721f629aab3d27e45d7ce8a3711.jpg)  
Figure 2 Mean SA allocation by true rank in each eighth of the budget $( I = 4 0 , k = 2 , r _ { 1 } = 3 )$ , with ratings on diagonal and comparisons off diagonal. The lower triangle reports the percent of budget spent on comparisons.

of SE difers from SA, we consider the allocation diference between SE and SA after SE switches, which on thi<sub>s</sub> i<sub>ns</sub>t<sub>ance</sub> i<sub>s a</sub>t $9 5 \%$ <sub>o</sub>f th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub> Fi<sub>gure</sub> 3<sub>a s</sub>h<sub>ows</sub> th<sub>e</sub> dif<sub>erence</sub> i<sub>n w</sub>h<sub>ere</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t i<sub>s spen</sub>t<sub>.</sub> B<sub>o</sub>th f<sub>ocus</sub> <sub>on</sub> th<sub>e</sub> t<sub>op-</sub>� b<sub>oun</sub>d<sub>ary,</sub> b<sub>u</sub>t KG <sub>concen</sub>t<sub>ra</sub>t<sub>es muc</sub>h <sub>more on compar</sub>i<sub>ng</sub> it<sub>ems</sub> 2 <sub>an</sub>d 3<sub>, w</sub>hil<sub>e</sub> SA <sub>sprea</sub>d<sub>s more</sub> <sub>across near</sub>b<sub>y</sub> it<sub>ems as</sub> it i<sub>s s</sub>till l<sub>oo</sub>ki<sub>ng</sub> f<sub>or</sub> l<sub>ong-</sub>t<sub>erm uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y re</sub>d<sub>uc</sub>ti<sub>on aroun</sub>d th<sub>e</sub> t<sub>op-</sub>� th<sub>res</sub>h<sub>o</sub>ld<sub>.</sub> Thi<sub>s</sub> i<sub>s</sub> th<sub>e va</sub>l<sub>ue o</sub>f <sub>sw</sub>it<sub>c</sub>hi<sub>ng</sub> t<sub>o</sub> KG<sub>: a</sub>t th<sub>e en</sub>d <sub>o</sub>f th<sub>e</sub> h<sub>or</sub>i<sub>zon we wan</sub>t t<sub>o narrow</sub> i<sub>n on eva</sub>l<sub>ua</sub>ti<sub>ons w</sub>ith <sub>po</sub>t<sub>en</sub>ti<sub>a</sub>l f<sub>or</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub> i<sub>mprovemen</sub>t<sub>.</sub>

Fi<sub>gure</sub> 3b <sub>s</sub>h<sub>ows</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l EOC <sub>o</sub>f $\mathbf { S } \mathbf { E } ( \mathbf { s w } = \nu )$ <sub>w</sub>hi<sub>c</sub>h<sub>,</sub> f<sub>or</sub> ill<sub>us</sub>t<sub>ra</sub>ti<sub>ve purposes, we compu</sub>t<sub>e</sub> f<sub>or a</sub>ll <sub>�.</sub> Th<sub>e</sub> <sub>marg</sub>i<sub>na</sub>l <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> <sub>s</sub>t<sub>ops</sub> <sub>a</sub>t $\nu = 0 . 9 5$ <sub>, w</sub>hil<sub>e</sub> th<sub>e m</sub>i<sub>n</sub>i<sub>mum</sub> i<sub>s a</sub>t $\nu = 0 . 8 .$ <sub>, s</sub>h<sub>ow</sub>i<sub>ng</sub> th<sub>a</sub>t it <sub>nee</sub>d <sub>no</sub>t <sub>g</sub>i<sub>ve</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>sw</sub>it<sub>c</sub>hi<sub>ng po</sub>i<sub>n</sub>t<sub>.</sub> H<sub>owever,</sub> th<sub>e</sub> t<sub>wo</sub> dif<sub>er</sub> b<sub>y</sub> l<sub>ess</sub> th<sub>an</sub> 1% <sub>o</sub>f th<sub>e</sub> EOC<sub>, we</sub>ll <sub>w</sub>ithi<sub>n one s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>error</sub> <sub>o</sub>f th<sub>e es</sub>ti<sub>ma</sub>t<sub>e</sub>d EOC<sub>.</sub> I<sub>n genera</sub>l<sub>, as � moves</sub> b<sub>ac</sub>k th<sub>e</sub> EOC fi<sub>rs</sub>t di<sub>ves</sub> t<sub>o</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>po</sub>i<sub>n</sub>t <sub>an</sub>d th<sub>en</sub> i<sub>ncreases,</sub> and the same pattern appears in the other configurations. In the majority of the configurations, the switching <sub>proce</sub>d<sub>ure</sub> id<sub>en</sub>tifi<sub>es</sub> th<sub>e</sub> <sub>m</sub>i<sub>n</sub>i<sub>mum.</sub>

## 7.3. The value of combining the two evaluation types

W<sub>e</sub> <sub>nex</sub>t f<sub>ocus</sub> <sub>on</sub> th<sub>e</sub> <sub>va</sub>l<sub>ue</sub> <sub>o</sub>f <sub>com</sub>bi<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>wo</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>ypes</sub> <sub>an</sub>d t<sub>es</sub>t h<sub>ow</sub> <sub>we</sub>ll SE b<sub>a</sub>l<sub>ances</sub> b<sub>e</sub>t<sub>ween</sub> them b<sub>y</sub> com<sub>p</sub>arin<sub>g</sub> SE to the restricted versions of itself defined in §7.1. We find that ratin<sub>g</sub> onl<sub>y</sub> $( \mathrm { S E } ( \mathrm { r c } = 1 ) )$ 0 i<sub>s</sub> <sub>no</sub>t <sub>compe</sub>titi<sub>ve,</sub> <sub>w</sub>ith t<sub>erm</sub>i<sub>na</sub>l EOC b<sub>e</sub>t<sub>ween</sub> 1<sub>.</sub>34 <sub>an</sub>d 15<sub>.</sub>9 ti<sub>mes</sub> th<sub>a</sub>t <sub>o</sub>f SE <sub>across</sub> th<sub>e</sub> <sub>con</sub>fi<sub>gura</sub>ti<sub>ons.</sub> A<sub>s</sub> di<sub>scusse</sub>d i<sub>n</sub> th<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on, ra</sub>ti<sub>ngs are no</sub>t <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> i<sub>n separa</sub>ti<sub>ng</sub> it<sub>ems w</sub>h<sub>ose u</sub>tiliti<sub>es are c</sub>l<sub>ose</sub> t<sub>oge</sub>th<sub>er,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> k<sub>ey</sub> f<sub>or narrow</sub>i<sub>ng</sub> i<sub>n on</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>se</sub>l<sub>ec</sub>ti<sub>on.</sub> C<sub>onverse</sub>l<sub>y, w</sub>h<sub>en</sub> $r _ { 1 } = 2 .$ <sub>,</sub> SE <sub>a</sub>nd ${ \mathrm { S E } } ( { \mathrm { r c } } = 0 )$ are <sub>qu</sub>it<sub>e</sub> <sub>compara</sub>bl<sub>e,</sub> th<sub>a</sub>t i<sub>s,</sub> <sub>ra</sub>ti<sub>ngs</sub> d<sub>o</sub> <sub>no</sub>t <sub>a</sub>dd <sub>muc</sub>h <sub>va</sub>l<sub>ue.</sub> Th<sub>us,</sub> t<sub>o</sub> <sub>see</sub> th<sub>e</sub> <sub>va</sub>l<sub>ue</sub> i<sub>n</sub> <sub>com</sub>bi<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>wo</sub> i<sub>npu</sub>t t<sub>ypes,</sub> <sub>one</sub> <sub>o</sub>f th<sub>e</sub> <sub>ma</sub>i<sub>n</sub> <sub>prem</sub>i<sub>ses</sub> <sub>o</sub>f thi<sub>s</sub> <sub>paper,</sub> <sub>we</sub> l<sub>oo</sub>k <sub>a</sub>t th<sub>e</sub> <sub>c</sub>h<sub>ange</sub> i<sub>n</sub> EOC <sub>o</sub>f SE <sub>an</sub>d ${ \mathrm { S E } } ( { \mathrm { r c } } = 0 )$ <sub>as</sub> th<sub>e</sub> <sub>cos</sub>t <sub>o</sub>f <sub>a</sub> <sub>compar</sub>i<sub>son</sub> <sub>goes</sub> f<sub>rom</sub> $r _ { 1 } = 2 \mathrm { t o } r _ { 1 } = 3 .$ <sub>,</sub> <sub>s</sub>h<sub>own</sub> i<sub>n</sub> Fi<sub>gure</sub> 4<sub>.</sub> SE <sub>can</sub> <sub>a</sub>d<sub>ap</sub>t t<sub>o</sub> thi<sub>s,</sub> <sub>w</sub>hil<sub>e</sub> ${ \mathrm { S E } } ( { \mathrm { r c } } = 0 )$ must still rely on comparisons alone. We see that having the ability to rate can be a major improvement. The EOC <sub>o</sub>f $\mathrm { S E } ( \mathrm { r c } = 0 )$ i<sub>ncreases</sub> b<sub>y</sub> b<sub>e</sub>t<sub>ween</sub> 57% <sub>an</sub>d 167%<sub>, w</sub>h<sub>ereas</sub> th<sub>a</sub>t <sub>o</sub>f SE i<sub>ncreases</sub> b<sub>y on</sub>l<sub>y</sub> b<sub>e</sub>t<sub>ween</sub> 43% <sub>an</sub>d 88%<sub>, s</sub>i<sub>nce</sub> it <sub>can su</sub>b<sub>s</sub>tit<sub>u</sub>t<sub>e</sub> th<sub>e now more expens</sub>i<sub>ve compar</sub>i<sub>sons w</sub>ith <sub>ra</sub>ti<sub>ngs.</sub> N<sub>ex</sub>t<sub>, we</sub> i<sub>nves</sub>ti<sub>ga</sub>t<sub>e</sub>

![](images/03d75108b0e012c81c46d92c9f73607e0857ac13a658be6c96b6ead09e5767b7.jpg)

![](images/f885d08eaa0dc409bdac109ab22c19d4379351d7bfb48b7b23237ec0d1b3f415.jpg)  
Figure 3 (a) Average difference of the last 5% of the budget allocation between SE and SA $( I = 4 0 , k = 2 , r _ { 1 } = 3 )$ Green denotes SE spends more, red it spends less. (b) Terminal EOC of $\widehat { \mathbf { S } } \mathbf { E } ( \mathbf { s } \mathbf { w } = \nu )$ against the switching point.

![](images/567365289efcf1bed0cc1de2cb43d81beb7fdbc813f084f9f4e3bbccfe38a128.jpg)  
Figure 4 Percent change in terminal EOC of SE and $\mathtt { S E } ( \mathtt { r e } = 0 )$ when the comparison cost goes from $r _ { 1 } = 2$ to $r _ { 1 } = 3 ,$ , each policy measured against itself at $r _ { 1 } = 2 .$ Error bars are one standard error.

th<sub>e</sub> <sub>ra</sub>t<sub>e-</sub>th<sub>en-compare</sub> i<sub>n</sub>t<sub>u</sub>iti<sub>on,</sub> <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>t <sub>w</sub>ith th<sub>e</sub> <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> <sub>pa</sub>tt<sub>erns</sub> i<sub>n</sub> Fi<sub>gure</sub> 2<sub>.</sub> T<sub>o</sub> d<sub>o</sub> thi<sub>s,</sub> <sub>we</sub> <sub>compare</sub> SE t<sub>o</sub> $\mathrm { S E } ( \mathrm { r c } = 1 / 2 )$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>s</sub>till <sub>a</sub>ll<sub>oca</sub>t<sub>es</sub> b<sub>y</sub> <sub>our</sub> <sub>me</sub>th<sub>o</sub>d b<sub>u</sub>t <sub>s</sub>i<sub>mp</sub>lifi<sub>es</sub> th<sub>e</sub> <sub>pro</sub>bl<sub>em</sub> b<sub>y</sub> <sub>no</sub>t h<sub>av</sub>i<sub>ng</sub> t<sub>o</sub> <sub>c</sub>h<sub>oose</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> t<sub>wo eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>ypes a</sub>t <sub>eac</sub>h <sub>roun</sub>d<sub>.</sub> Fi<sub>gure</sub> 5 <sub>s</sub>h<sub>ows</sub> th<sub>e percen</sub>t dif<sub>erence</sub> i<sub>n</sub> it<sub>s</sub> EOC <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> SE<sub>.</sub> A<sub>s expec</sub>t<sub>e</sub>d<sub>,</sub> SE <sub>comes ou</sub>t <sub>a</sub>h<sub>ea</sub>d i<sub>n</sub> 22 <sub>o</sub>f th<sub>e</sub> 24 <sub>con</sub>fi<sub>gura</sub>ti<sub>ons,</sub> b<sub>y up</sub> t<sub>o</sub> 29%<sub>, an</sub>d l<sub>oses</sub> i<sub>n</sub> t<sub>wo,</sub> b<sub>y</sub> 5<sub>.</sub>5% at $I = 4 0 , k = 1 , r _ { 1 } = 2$ <sub>a</sub>nd b<sub>y</sub> 8<sub>.</sub>6% <sub>a</sub>t $I = 4 0 , k = 1 , r _ { 1 } = 3$ <sub>, w</sub>hi<sub>c</sub>h <sub>are</sub> b<sub>o</sub>th <sub>we</sub>ll <sub>w</sub>ithi<sub>n</sub> t<sub>wo s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>errors o</sub>f <sub>zero.</sub> Thi<sub>s s</sub>h<sub>ows</sub> th<sub>a</sub>t SE’<sub>s</sub> fl<sub>ex</sub>ibilit<sub>y</sub> t<sub>o a</sub>ll<sub>oca</sub>t<sub>e</sub> d<sub>ynam</sub>i<sub>ca</sub>ll<sub>y an</sub>d b<sub>a</sub>l<sub>ance</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> t<sub>wo a</sub>dd<sub>s va</sub>l<sub>ue,</sub> b<sub>u</sub>t i<sub>n</sub> <sub>some</sub> <sub>cases</sub> <sub>a</sub> <sub>s</sub>i<sub>mp</sub>l<sub>e</sub> h<sub>eur</sub>i<sub>s</sub>ti<sub>c</sub> <sub>per</sub>f<sub>orms</sub> <sub>s</sub>i<sub>m</sub>il<sub>ar</sub>l<sub>y.</sub> H<sub>owever,</sub> <sub>one</sub> <sub>can</sub> <sub>c</sub>h<sub>oose</sub> <sub>se</sub>tti<sub>ngs</sub> b<sub>ase</sub>d <sub>on</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>,</sub> <sub>�0,</sub> $r _ { 1 } , \beta _ { 0 } , \beta _ { 1 }$ <sub>,</sub> <sub>an</sub>d $\theta _ { 0 } .$ <sub>,</sub> <sub>suc</sub>h th<sub>a</sub>t <sub>a</sub> fi<sub>xe</sub>d h<sub>eur</sub>i<sub>s</sub>ti<sub>c</sub> <sub>wou</sub>ld d<sub>o</sub> <sub>muc</sub>h <sub>worse,</sub> b<sub>u</sub>t SE <sub>w</sub>ill <sub>s</sub>till d<sub>o</sub> <sub>we</sub>ll b<sub>y</sub> <sub>a</sub>d<sub>ap</sub>ti<sub>ng</sub> t<sub>o</sub> th<sub>e</sub> <sub>con</sub>fi<sub>gura</sub>ti<sub>on.</sub>

![](images/12e52dfa5de5c9c0eccfb8e095032188f96e64aca1f0f7205bce57273ccb0f3d.jpg)  
Figure 5 Percent difference in terminal EOC of SE(rc = 1/2) relative to SE, all configurations. Above zero SE is better, below zero SE(rc = 1/2) is. Error bars are one standard error.

## 7.4. Comparison to external benchmarks

W<sub>e</sub> <sub>nex</sub>t <sub>compare</sub> th<sub>e</sub> <sub>per</sub>f<sub>ormance</sub> <sub>o</sub>f <sub>our</sub> <sub>me</sub>th<sub>o</sub>d t<sub>o</sub> R<sub>ate</sub>R<sub>ank</sub> <sub>an</sub>d th<sub>e</sub> t<sub>wo</sub> KG <sub>var</sub>i<sub>an</sub>t<sub>s.</sub> Fi<sub>gure</sub> 6 <sub>p</sub>l<sub>o</sub>t<sub>s</sub> th<sub>e</sub> EOC <sub>aga</sub>i<sub>ns</sub>t th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>spen</sub>t <sub>on</sub> th<sub>ree con</sub>fi<sub>gura</sub>ti<sub>ons.</sub> I<sub>n a</sub>dditi<sub>on</sub> t<sub>o</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l EOC<sub>s,</sub> thi<sub>s</sub> fi<sub>gure</sub> hi<sub>g</sub>hli<sub>g</sub>ht<sub>s</sub> t<sub>wo</sub> i<sub>n</sub>t<sub>eres</sub>ti<sub>ng</sub> dif<sub>erences</sub> i<sub>n</sub> h<sub>ow</sub> th<sub>e</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>ve</sub> <sub>approac</sub>h<sub>es</sub> <sub>ge</sub>t th<sub>ere.</sub> Fi<sub>rs</sub>t<sub>,</sub> <sub>we</sub> <sub>can</sub> <sub>see</sub> th<sub>e</sub> “f<sub>orwar</sub>d<sub>-</sub>l<sub>oo</sub>ki<sub>ng</sub>” <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> of SE. Durin<sub>g</sub> a lar<sub>g</sub>e <sub>p</sub>art of the earl<sub>y</sub> bud<sub>g</sub>et s<sub>p</sub>end, its EOC is hi<sub>g</sub>her than that of KG(SA): SE <sub>c</sub>h<sub>ooses</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> th<sub>a</sub>t d<sub>o</sub> <sub>no</sub>t <sub>max</sub>i<sub>m</sub>i<sub>ze</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub> EOC i<sub>mprovemen</sub>t<sub>,</sub> b<sub>u</sub>t <sub>ena</sub>bl<sub>e</sub> it t<sub>o</sub> <sub>surpass</sub> th<sub>e</sub> KG <sub>po</sub>li<sub>c</sub>i<sub>es</sub> l<sub>a</sub>t<sub>er</sub> i<sub>n</sub> th<sub>e</sub> h<sub>or</sub>i<sub>zon.</sub> T<sub>owar</sub>d<sub>s</sub> th<sub>e</sub> <sub>m</sub>iddl<sub>e</sub> <sub>o</sub>f th<sub>e</sub> h<sub>or</sub>i<sub>zon,</sub> thi<sub>s</sub> <sub>ear</sub>l<sub>y</sub> i<sub>nves</sub>t<sub>men</sub>t <sub>s</sub>t<sub>ar</sub>t<sub>s</sub> t<sub>o</sub> <sub>pay</sub> <sub>o</sub>f<sub>,</sub> <sub>an</sub>d SE <sub>p</sub>ulls ahead. The second takeawa<sub>y</sub> from the fi<sub>g</sub>ure is the se<sub>p</sub>aration between KG and KG(SA), which allows <sub>us</sub> t<sub>o c</sub>l<sub>ear</sub>l<sub>y see w</sub>h<sub>ere, on average,</sub> i<sub>n</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t KG <sub>ge</sub>t<sub>s s</sub>t<sub>uc</sub>k <sub>an</sub>d h<sub>ow muc</sub>h it <sub>su</sub>f<sub>ers</sub> f<sub>rom</sub> it<sub>.</sub>

![](images/529c84bb77ea4bac7af0337c3080545421f7ddaeb4bc1045098af233310be345.jpg)

![](images/05310e58ef54032e9ece879839993481d17971f697fdab644a766eb258b2c4a6.jpg)

![](images/a98e488d71dae755224b8970f8d66d36d8511bc2ea15d3cb96ec8b2b612df138.jpg)  
Figure 6 EOC against budget spent.

Fi<sub>gure</sub> 7 <sub>s</sub>h<sub>ows</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l EOC <sub>o</sub>f th<sub>e compe</sub>ti<sub>ng po</sub>li<sub>c</sub>i<sub>es across a</sub>ll <sub>con</sub>fi<sub>gura</sub>ti<sub>ons, as</sub> th<sub>e percen</sub>t dif<sub>erence</sub> from SE. Remarkabl<sub>y</sub>, SE tends to substantiall<sub>y</sub> out<sub>p</sub>erform KG(SA): KG(SA) has a hi<sub>g</sub>her terminal EOC in 23 <sub>o</sub>f th<sub>e</sub> 24 <sub>con</sub>fi<sub>gura</sub>ti<sub>ons,</sub> b<sub>y more</sub> th<sub>an</sub> 30% i<sub>n</sub> 16 <sub>o</sub>f th<sub>em, an</sub>d <sub>up</sub> t<sub>o</sub> 281% <sub>a</sub>t $I = 1 6 0 , k = 4 , r _ { 1 } = 2$ <sub>.</sub> We sa<sub>y</sub> this is remarkable since KG(SA) is a stron<sub>g</sub> benchmark for our <sub>p</sub>roblem: the bud<sub>g</sub>ets are small, a re<sub>g</sub>ime <sub>w</sub>h<sub>ere</sub> KG <sub>per</sub>f<sub>orms we</sub>ll<sub>, an</sub>d <sub>we equ</sub>i<sub>p</sub> it <sub>w</sub>ith th<sub>e</sub> b<sub>es</sub>t <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> f<sub>a</sub>llb<sub>ac</sub>k <sub>ru</sub>l<sub>e.</sub> F<sub>ur</sub>th<sub>ermore,</sub> th<sub>e</sub> f<sub>a</sub>llb<sub>ac</sub>k i<sub>s</sub> i<sub>nvo</sub>k<sub>e</sub>d <sub>prec</sub>i<sub>se</sub>l<sub>y</sub> i<sub>n s</sub>t<sub>a</sub>t<sub>es w</sub>h<sub>ere</sub> SA’<sub>s exp</sub>l<sub>ora</sub>t<sub>ory na</sub>t<sub>ure</sub> i<sub>s mos</sub>t <sub>va</sub>l<sub>ua</sub>bl<sub>e</sub> t<sub>o</sub> KG<sub>, w</sub>h<sub>ere no s</sub>i<sub>ng</sub>l<sub>e eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>can</sub> di<sub>rec</sub>tl<sub>y</sub> <sub>c</sub>h<sub>ange</sub> th<sub>e</sub> <sub>curren</sub>t t<sub>op-</sub>� <sub>se</sub>t<sub>.</sub> Thi<sub>s</sub> <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> <sub>mos</sub>t <sub>o</sub>b<sub>v</sub>i<sub>ous</sub> d<sub>raw</sub>b<sub>ac</sub>k <sub>o</sub>f KG b<sub>e</sub>i<sub>ng</sub> <sub>myop</sub>i<sub>c</sub> i<sub>s</sub> th<sub>a</sub>t it <sub>can ge</sub>t <sub>s</sub>t<sub>uc</sub>k<sub>,</sub> th<sub>a</sub>t <sub>a</sub>l<sub>one</sub> i<sub>s no</sub>t th<sub>e en</sub>ti<sub>re pro</sub>bl<sub>em.</sub> Th<sub>e eagerness</sub> t<sub>o</sub> di<sub>rec</sub>tl<sub>y</sub> i<sub>mprove</sub> th<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>e</sub>d � th<sub>roug</sub>h<sub>ou</sub>t th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>resu</sub>lt<sub>s</sub> i<sub>n eva</sub>l<sub>ua</sub>ti<sub>on c</sub>h<sub>o</sub>i<sub>ces w</sub>ith li<sub>m</sub>it<sub>e</sub>d i<sub>n</sub>f<sub>orma</sub>ti<sub>on con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e</sub> ultimate selection. We also note that KG(SA) out<sub>p</sub>erforms KG in ever<sub>y</sub> confi<sub>g</sub>uration, b<sub>y</sub> 4% to 701%, and f<sub>or</sub> th<sub>e con</sub>fi<sub>gura</sub>ti<sub>ons w</sub>h<sub>ere</sub> it i<sub>s compu</sub>t<sub>a</sub>ti<sub>ona</sub>ll<sub>y prac</sub>ti<sub>ca</sub>l<sub>,</sub> R<sub>ate</sub>R<sub>ank</sub> d<sub>oes cons</sub>id<sub>era</sub>bl<sub>y worse</sub> th<sub>an our</sub> method (and KG), with 1.7 to 12 times hi<sub>g</sub>her terminal EOC. While a lar<sub>g</sub>e motivatin<sub>g</sub> factor in this work is

![](images/82b1e08c5d5c8fe25d798724ba2da90cd4a1f330f88b6884aefa12d64d493e64.jpg)  
Figure 7 Percent difference in terminal EOC from SE. Rows are labeled $k , r _ { 1 }$ and grouped by �.

<sub>sma</sub>ll b<sub>u</sub>d<sub>ge</sub>t<sub>s,</sub> th<sub>ere are se</sub>tti<sub>ngs o</sub>f <sub>crow</sub>d<sub>sourc</sub>i<sub>ng</sub> f<sub>rom</sub> h<sub>uman eva</sub>l<sub>ua</sub>t<sub>ors w</sub>h<sub>ere</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>s</sub> f<sub>ar</sub> b<sub>eyon</sub>d 4� <sub>are</sub> realistic. Thus we com<sub>p</sub>are the benchmarks to our method in this re<sub>g</sub>ime, with no switchin<sub>g</sub> (SA throu<sub>g</sub>hout), <sub>s</sub>i<sub>nce</sub> <sub>as</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>grows</sub> th<sub>e</sub> i<sub>mpac</sub>t <sub>o</sub>f th<sub>e</sub> <sub>en</sub>d<sub>game</sub> <sub>an</sub>d th<sub>e</sub> <sub>va</sub>l<sub>ue</sub> <sub>o</sub>f <sub>sw</sub>it<sub>c</sub>hi<sub>ng</sub> d<sub>ecrease.</sub> Fi<sub>gure</sub> 8 <sub>s</sub>h<sub>ows</sub> th<sub>e percen</sub>t dif<sub>erence</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s an</sub>d SA <sub>as a</sub> f<sub>unc</sub>ti<sub>on o</sub>f th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>spen</sub>t<sub>, w</sub>ith th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t extended to 16�. As ex<sub>p</sub>ected, at the extremel<sub>y</sub> small bud<sub>g</sub>et � = �, KG is as <sub>g</sub>ood as SA or better, but as the b<sub>u</sub>d<sub>ge</sub>t <sub>grows</sub> th<sub>e a</sub>d<sub>van</sub>t<sub>age o</sub>f SA <sub>grows su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y w</sub>ith it<sub>, w</sub>hi<sub>c</sub>h i<sub>s</sub> t<sub>o</sub> b<sub>e expec</sub>t<sub>e</sub>d<sub>, as</sub> thi<sub>s</sub> i<sub>s</sub> th<sub>e reg</sub>i<sub>me</sub> <sub>w</sub>h<sub>ere</sub> th<sub>e</sub> <sub>myop</sub>i<sub>a</sub> <sub>o</sub>f KG h<sub>ur</sub>t<sub>s</sub> <sub>more.</sub>

## 7.5. Robustness to the Gaussian approximation and to misspecification

We next stress-test the efect of Approximation 1 on SE. In the absence of a conjugate prior for the probit lik<sub>e</sub>lih<sub>oo</sub>d th<sub>a</sub>t i<sub>s amena</sub>bl<sub>e</sub> t<sub>o ana</sub>l<sub>ys</sub>i<sub>s,</sub> thi<sub>s approx</sub>i<sub>ma</sub>ti<sub>on rep</sub>l<sub>aces</sub> th<sub>e pos</sub>t<sub>er</sub>i<sub>or w</sub>ith <sub>a mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> fitt<sub>e</sub>d b<sub>y var</sub>i<sub>a</sub>ti<sub>ona</sub>l i<sub>n</sub>f<sub>erence, an</sub>d th<sub>e expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> i<sub>s compu</sub>t<sub>e</sub>d <sub>w</sub>ith <sub>respec</sub>t t<sub>o</sub> it<sub>.</sub> T<sub>o c</sub>h<sub>ec</sub>k <sub>w</sub>h<sub>e</sub>th<sub>er</sub> thi<sub>s s</sub>i<sub>gn</sub>ifi<sub>can</sub>tl<sub>y a</sub>lt<sub>ers</sub> th<sub>e per</sub>f<sub>ormance o</sub>f th<sub>e me</sub>th<sub>o</sub>d<sub>, we run</sub> SE <sub>w</sub>ith th<sub>e exac</sub>t <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> i<sub>n p</sub>l<sub>ace</sub> <sub>o</sub>f th<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> <sub>one.</sub> Th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> i<sub>s</sub> <sub>represen</sub>t<sub>e</sub>d b<sub>y</sub> 2<sub>,</sub>000 d<sub>raws</sub> <sub>o</sub>f <sub>a</sub> Gibb<sub>s</sub> <sub>samp</sub>l<sub>er</sub> i<sub>n</sub> <sub>eac</sub>h <sub>roun</sub>d<sub>,</sub> <sub>an</sub>d <sub>every</sub>thi<sub>ng</sub> SE d<sub>epen</sub>d<sub>s</sub> <sub>on,</sub> th<sub>e</sub> <sub>ou</sub>t<sub>come</sub> <sub>pro</sub>b<sub>a</sub>biliti<sub>es,</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>covar</sub>i<sub>ance,</sub> th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss,</sub> i<sub>s</sub> <sub>compu</sub>t<sub>e</sub>d <sub>as</sub> <sub>a</sub> M<sub>on</sub>t<sub>e</sub> C<sub>ar</sub>l<sub>o</sub> <sub>average</sub> <sub>over</sub> th<sub>ose</sub> d<sub>raws,</sub> <sub>so</sub> th<sub>a</sub>t th<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> <sub>app</sub>r<sub>ox</sub>im<sub>a</sub>ti<sub>o</sub>n d<sub>oes</sub> n<sub>o</sub>t <sub>e</sub>nt<sub>e</sub>r <sub>a</sub>n<sub>yw</sub>h<sub>e</sub>r<sub>e.</sub> Fi<sub>gu</sub>r<sub>e</sub> 9 <sub>p</sub>l<sub>o</sub>t<sub>s</sub> th<sub>e</sub> <sub>pe</sub>r<sub>ce</sub>nt <sub>c</sub>h<sub>a</sub>n<sub>ge</sub> in th<sub>e</sub> t<sub>e</sub>rmin<sub>a</sub>l EOC <sub>o</sub>f SE <sub>w</sub>h<sub>en</sub> th<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> i<sub>s</sub> <sub>rep</sub>l<sub>ace</sub>d b<sub>y</sub> th<sub>e</sub> <sub>exac</sub>t <sub>pos</sub>t<sub>er</sub>i<sub>or,</sub> i<sub>n</sub> <sub>a</sub>ll <sub>con</sub>fi<sub>gura</sub>ti<sub>ons.</sub> It <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t th<sub>e</sub> <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> <sub>nega</sub>ti<sub>ve</sub>l<sub>y</sub> <sub>a</sub>f<sub>ec</sub>t<sub>s</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l EOC b<sub>y</sub> 4% <sub>on</sub> <sub>average</sub> <sub>across</sub> <sub>a</sub>ll <sub>con</sub>fi<sub>gura</sub>ti<sub>ons</sub> <sub>an</sub>d b<sub>y</sub> <sub>a</sub>t <sub>mos</sub>t 13%<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>qu</sub>it<sub>e</sub> <sub>m</sub>ild <sub>compare</sub>d t<sub>o</sub> SE’<sub>s</sub> <sub>a</sub>d<sub>van</sub>t<sub>age</sub> <sub>over</sub> th<sub>e</sub> b<sub>ase</sub>li<sub>nes.</sub>

![](images/6930b653b7ef2c3b1d06b9e84f83b3947f1160163f1227acd358761d46c62402.jpg)  
Figure 8 Percent difference in terminal EOC from SA against the total budget.

![](images/c3800f731e6fbd6027808ff7c4b9f8881188455e8d3d464c9bcdc00f93c7cc5b.jpg)  
Figure 9 % difference in terminal EOC between the Gaussian and exact posterior.

SE assumes <sub>p</sub>erfect knowled<sub>g</sub>e of the <sub>p</sub>arameters and that evaluators are homo<sub>g</sub>eneous (in the base <sub>con</sub>fi<sub>gura</sub>ti<sub>ons,</sub> <sub>accurac</sub>i<sub>es</sub> <sub>o</sub>f 75% <sub>an</sub>d 85% <sub>an</sub>d $\theta _ { 0 } = 0 )$ <sub>.</sub> W<sub>e</sub> <sub>nex</sub>t t<sub>es</sub>t th<sub>e</sub> <sub>me</sub>th<sub>o</sub>d <sub>un</sub>d<sub>er</sub> <sub>m</sub>i<sub>sspec</sub>ifi<sub>ca</sub>ti<sub>on</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>assume</sub>d <sub>parame</sub>t<sub>er</sub> <sub>va</sub>l<sub>ues</sub> <sub>are</sub> <sub>wrong</sub> <sub>an</sub>d <sub>w</sub>h<sub>en</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>ors</sub> <sub>are</sub> h<sub>e</sub>t<sub>erogeneous.</sub> W<sub>e</sub> <sub>compare</sub> th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> <sub>per</sub>f<sub>ormance</sub> <sub>w</sub>ith <sub>un</sub>bi<sub>ase</sub>d SE<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>runs</sub> <sub>w</sub>ith th<sub>e</sub> <sub>correc</sub>t <sub>assump</sub>ti<sub>ons.</sub> W<sub>e</sub> <sub>cons</sub>id<sub>er</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>m</sub>i<sub>sspec</sub>ifi<sub>ca</sub>ti<sub>ons:</sub>

<sub>–</sub> E<sub>va</sub>l<sub>ua</sub>t<sub>o</sub>r h<sub>e</sub>t<sub>e</sub>r<sub>oge</sub>n<sub>e</sub>it<sub>y.</sub> E<sub>ac</sub>h <sub>eva</sub>l<sub>ua</sub>ti<sub>on comes</sub> f<sub>rom a</sub> dif<sub>eren</sub>t <sub>eva</sub>l<sub>ua</sub>t<sub>or.</sub> W<sub>e cons</sub>id<sub>er</sub> h<sub>e</sub>t<sub>erogene</sub>it<sub>y</sub> i<sub>n accuracy an</sub>d i<sub>n ra</sub>ti<sub>ng</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>separa</sub>t<sub>e</sub>l<sub>y, an</sub>d i<sub>n eac</sub>h<sub>, an eva</sub>l<sub>ua</sub>t<sub>or</sub> i<sub>s one o</sub>f t<sub>wo</sub> t<sub>ypes w</sub>ith <sub>equa</sub>l <sub>pro</sub>b<sub>a</sub>bilit<sub>y.</sub>

1<sub>.</sub> Acc<sub>u</sub>racies<sub>.</sub> Th<sub>e</sub> r<sub>a</sub>tin<sub>g</sub> <sub>accu</sub>r<sub>acy</sub> i<sub>s</sub> 70% <sub>o</sub>r 80% <sub>a</sub>nd th<sub>e</sub> <sub>co</sub>m<sub>pa</sub>ri<sub>so</sub>n <sub>accu</sub>r<sub>acy</sub> i<sub>s</sub> 80% <sub>o</sub>r 90%<sub>.</sub>

2<sub>.</sub> Thr<sub>es</sub>h<sub>o</sub>ld<sub>.</sub> Th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld i<sub>s</sub> $\theta _ { 0 } = - 1 . 5 8 \mathrm { o r } \theta _ { 0 } = 1 . 5 8$ <sub>,</sub> h<sub>a</sub>lf <sub>a</sub> <sub>pr</sub>i<sub>or</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d d<sub>ev</sub>i<sub>a</sub>ti<sub>on</sub> b<sub>e</sub>l<sub>ow</sub> <sub>or</sub> <sub>a</sub>b<sub>ove</sub> th<sub>e</sub> <sub>assume</sub>d th<sub>res</sub>h<sub>o</sub>ld<sub>.</sub>

<sub>–</sub> E<sub>v</sub>al<sub>u</sub>ator acc<sub>u</sub>rac<sub>y.</sub> Th<sub>e</sub> tr<sub>ue</sub> r<sub>a</sub>tin<sub>g</sub> <sub>a</sub>nd <sub>co</sub>m<sub>pa</sub>ri<sub>so</sub>n <sub>accu</sub>r<sub>ac</sub>i<sub>es</sub> <sub>a</sub>r<sub>e</sub> 65% <sub>a</sub>nd 75%<sub>,</sub> <sub>o</sub>r 85% <sub>a</sub>nd 90%<sub>.</sub> <sub>–</sub> R<sub>a</sub>tin<sub>g</sub> thr<sub>es</sub>h<sub>o</sub>ld<sub>.</sub> Th<sub>e</sub> t<sub>rue</sub> th<sub>res</sub>h<sub>o</sub>ld i<sub>s</sub> $\theta _ { 0 } = - 1 . 5 8$ or $\theta _ { 0 } = 1 . 5 8$

T<sub>a</sub>bl<sub>e</sub> 1 <sub>repor</sub>t<sub>s</sub> th<sub>e percen</sub>t i<sub>ncrease</sub> i<sub>n</sub> t<sub>erm</sub>i<sub>na</sub>l EOC <sub>o</sub>f SE <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o un</sub>bi<sub>ase</sub>d SE<sub>.</sub> I<sub>n</sub> 16 <sub>o</sub>f th<sub>e</sub> 18 <sub>cases</sub> it is between −9% and 27%. While this decrease in <sub>p</sub>erformance (on avera<sub>g</sub>e) is not ne<sub>g</sub>li<sub>g</sub>ible, it is small relative to SE’s advanta<sub>g</sub>e even over KG(SA) in the corres<sub>p</sub>ondin<sub>g</sub> confi<sub>g</sub>urations (20%, 111%, and 217%). Th<sub>e</sub> <sub>se</sub>tti<sub>ng</sub> <sub>we</sub> <sub>cons</sub>id<sub>er</sub> <sub>w</sub>h<sub>ere</sub> <sub>m</sub>i<sub>sspec</sub>ifi<sub>ca</sub>ti<sub>on</sub> h<sub>as</sub> <sub>a</sub> <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>l i<sub>mpac</sub>t i<sub>s</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> t<sub>rue</sub> <sub>accurac</sub>i<sub>es</sub> <sub>are</sub>

Table 1 % increase in the EOC of SE relative to unbiased SE (standard error), $r _ { 1 } = 3 .$
<table><tr><td>misspecification</td><td> $I = 2 0 , k = 2$ </td><td> $I = 8 0 , k = 3$ </td><td> $I = 1 6 0 , k = 4$ </td></tr><tr><td>heterogeneous accuracies</td><td>s +3.7% (5.0)</td><td>+16.7% (4.3)</td><td>+14.8% (3.8)</td></tr><tr><td>heterogeneous  $\theta _ { 0 }$ </td><td>+2.5% (4.6)</td><td>-6.6% (3.4)</td><td>+1.7% (3.4)</td></tr><tr><td>true accuracies 65/75</td><td>-3.1% (2.6)</td><td>-8.9% (2.2)</td><td>-4.6% (1.8)</td></tr><tr><td>true accuracies 85/90</td><td>+5.3% (6.6)</td><td>+41.2% (6.1)</td><td>+73.7% (6.5)</td></tr><tr><td>true  $\theta _ { 0 } = + 1 . 5 8$ </td><td>+3.2% (3.7)</td><td>-0.4% (3.6)</td><td>+1.5% (3.2)</td></tr><tr><td>true  $\theta _ { 0 } = - 1 . 5 8$ </td><td>-1.2% (4.0)</td><td>+26.9% (4.3)</td><td> $+ 2 1 . 5 \%$  (3.5)</td></tr></table>

85% <sub>an</sub>d 90%<sub>.</sub> Th<sub>ere,</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l EOC <sub>o</sub>f SE i<sub>s</sub> 41% <sub>an</sub>d 74% hi<sub>g</sub>h<sub>er</sub> th<sub>an</sub> <sub>un</sub>bi<sub>ase</sub>d SE <sub>a</sub>t $I = 8 0$ <sub>an</sub>d 160<sub>.</sub> W<sub>e no</sub>t<sub>e</sub> th<sub>a</sub>t thi<sub>s m</sub>i<sub>sspec</sub>ifi<sub>ca</sub>ti<sub>on</sub> i<sub>s ex</sub>t<sub>reme:</sub> $\beta _ { 0 }$ is twice what SE assumes (0.621 instead of 0.316), and $\beta _ { 1 }$ is 1.6 times (0.688 instead of 0.439). To understand what drives the diference in <sub>p</sub>erformance, Table 2 <sub>s</sub>h<sub>ows</sub> f<sub>or</sub> th<sub>e</sub> $I = 8 0$ <sub>con</sub>fi<sub>gura</sub>ti<sub>on</sub> h<sub>ow</sub> SE <sub>an</sub>d <sub>un</sub>bi<sub>ase</sub>d SE <sub>sp</sub>lit <sub>spen</sub>di<sub>ng</sub> b<sub>e</sub>t<sub>ween</sub> <sub>ra</sub>ti<sub>ngs</sub> <sub>an</sub>d <sub>compar</sub>i<sub>sons</sub> <sub>over</sub> th<sub>e course o</sub>f th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub> U<sub>n</sub>bi<sub>ase</sub>d SE <sub>s</sub>h<sub>ows</sub> th<sub>e ra</sub>ti<sub>ng-</sub>th<sub>en-compar</sub>i<sub>son pa</sub>tt<sub>ern o</sub>f Fi<sub>gure</sub> 2 <sub>even more</sub> <sub>s</sub>h<sub>arp</sub>l<sub>y</sub> th<sub>an</sub> i<sub>n</sub> th<sub>e</sub> b<sub>ase</sub> <sub>con</sub>fi<sub>gura</sub>ti<sub>ons:</sub> it <sub>spen</sub>d<sub>s</sub> th<sub>e</sub> fi<sub>rs</sub>t h<sub>a</sub>lf <sub>o</sub>f th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>on</sub> <sub>ra</sub>ti<sub>ngs,</sub> <sub>an</sub>d th<sub>en</sub> t<sub>urns</sub> <sub>a</sub>l<sub>mos</sub>t entirel<sub>y</sub> to com<sub>p</sub>arisons, makin<sub>g g</sub>ood use of the more accurate evaluations (ratin<sub>g</sub>s in <sub>p</sub>articular). In contrast, SE <sub>m</sub>i<sub>xes</sub> i<sub>n</sub> <sub>compar</sub>i<sub>sons</sub> <sub>ear</sub>l<sub>y,</sub> <sub>an</sub>d it<sub>s</sub> <sub>s</sub>hift t<sub>o</sub> <sub>more</sub> <sub>compar</sub>i<sub>sons</sub> <sub>comes</sub> l<sub>a</sub>t<sub>er</sub> <sub>an</sub>d i<sub>s</sub> l<sub>ess</sub> <sub>pronounce</sub>d<sub>.</sub> Th<sub>e</sub> <sub>mec</sub>h<sub>an</sub>i<sub>sm</sub> d<sub>r</sub>i<sub>v</sub>i<sub>ng</sub> thi<sub>s</sub> i<sub>s</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng.</sub> C<sub>ons</sub>id<sub>er a goo</sub>d it<sub>em, one pr</sub>i<sub>or s</sub>t<sub>an</sub>d<sub>ar</sub>d d<sub>ev</sub>i<sub>a</sub>ti<sub>on a</sub>b<sub>ove</sub> th<sub>e mean.</sub> With th<sub>e more accura</sub>t<sub>e eva</sub>l<sub>ua</sub>t<sub>ors, a ra</sub>ti<sub>ng resu</sub>lt<sub>s</sub> i<sub>n a</sub> th<sub>um</sub>b<sub>s-up</sub> 97<sub>.</sub>5% <sub>o</sub>f th<sub>e</sub> ti<sub>me, w</sub>hil<sub>e</sub> SE <sub>assumes</sub> it i<sub>s</sub> 84%<sub>.</sub> SE <sub>sees</sub> <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> <sub>goo</sub>d it<sub>ems</sub> <sub>rece</sub>i<sub>ve</sub> <sub>severa</sub>l th<sub>um</sub>b<sub>s-up,</sub> <sub>qu</sub>i<sub>c</sub>kl<sub>y</sub> b<sub>ecomes</sub> <sub>overcon</sub>fid<sub>en</sub>t th<sub>a</sub>t th<sub>ey</sub> <sub>are</sub> <sub>goo</sub>d<sub>, an</sub>d <sub>s</sub>t<sub>ar</sub>t<sub>s compar</sub>i<sub>ng</sub> th<sub>em ear</sub>l<sub>y.</sub> A<sub>s a resu</sub>lt<sub>,</sub> it d<sub>oes no</sub>t <sub>ra</sub>t<sub>e enoug</sub>h <sub>o</sub>f th<sub>e o</sub>th<sub>er</sub> it<sub>ems</sub> i<sub>n</sub> th<sub>e</sub> fi<sub>rs</sub>t h<sub>a</sub>lf <sub>o</sub>f th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>,</sub> <sub>an</sub>d l<sub>a</sub>t<sub>er</sub> <sub>goes</sub> b<sub>ac</sub>k t<sub>o</sub> <sub>ra</sub>ti<sub>ng</sub> th<sub>em.</sub> O<sub>vera</sub>ll<sub>,</sub> <sub>m</sub>i<sub>sspec</sub>ifi<sub>ca</sub>ti<sub>on</sub> <sub>genera</sub>ll<sub>y</sub> h<sub>as</sub> <sub>a</sub> <sub>nega</sub>ti<sub>ve</sub> i<sub>mpac</sub>t

Table 2 % of each eighth of the budget spent on comparisons. Accuracies: 85% and 90% $( I = 8 0 , k = 3 , r _ { 1 } = 3 )$
<table><tr><td>eighth of the budget</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td>SE</td><td>22%</td><td>31%</td><td>32%</td><td>31%</td><td>30%</td><td>34%</td><td>47%</td><td>73%</td></tr><tr><td>unbiased SE</td><td>0%</td><td>0%</td><td>0%</td><td>1%</td><td>52%</td><td>99%</td><td>98%</td><td>94%</td></tr></table>

<sub>on</sub> SE’<sub>s per</sub>f<sub>ormance, w</sub>hi<sub>c</sub>h <sub>can</sub> b<sub>e su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>l <sub>w</sub>h<sub>en</sub> th<sub>e m</sub>i<sub>sspec</sub>ifi<sub>ca</sub>ti<sub>on</sub> i<sub>s ex</sub>t<sub>reme,</sub> b<sub>u</sub>t th<sub>e</sub> l<sub>oss</sub> i<sub>s sma</sub>ll <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> SE’<sub>s a</sub>d<sub>van</sub>t<sub>age over</sub> th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s.</sub>

## 7.6. Impact of batching

Fi<sub>na</sub>ll<sub>y, we</sub> t<sub>es</sub>t th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h<sub>e</sub>d <sub>vers</sub>i<sub>on o</sub>f th<sub>e me</sub>th<sub>o</sub>d <sub>w</sub>ith th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h <sub>en</sub>d<sub>game parame</sub>t<sub>er</sub> $w = 3$ <sub>an</sub>d th<sub>e</sub> <sub>sw</sub>it<sub>c</sub>hi<sub>ng</sub> round chosen b<sub>y</sub> the <sub>p</sub>rocedure of §5.2 a<sub>pp</sub>lied to rounds. We run it on three confi<sub>g</sub>urations with the total b<sub>u</sub>d<sub>ge</sub>t <sub>o</sub>f 4� <sub>sp</sub>lit i<sub>n</sub>t<sub>o</sub> <sub>roun</sub>d b<sub>u</sub>d<sub>ge</sub>t<sub>s</sub> <sub>o</sub>f $B = I / 8 , I / 4 , I / 2$ , and � (rounded u<sub>p</sub>), that is, from 32 rounds down t<sub>o</sub> 4<sub>.</sub> Fi<sub>gure</sub> 10 <sub>s</sub>h<sub>ows</sub> h<sub>ow</sub> EOC i<sub>mproves</sub> <sub>as</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t i<sub>s</sub> <sub>spen</sub>t<sub>.</sub> Th<sub>e</sub> i<sub>mpac</sub>t <sub>o</sub>f b<sub>a</sub>t<sub>c</sub>hi<sub>ng</sub> i<sub>s</sub> <sub>sma</sub>ll <sub>a</sub>t $B = I / 8$ <sub>w</sub>ith <sub>a</sub> 5% t<sub>o</sub> 8% i<sub>ncrease</sub> i<sub>n</sub> t<sub>erm</sub>i<sub>na</sub>l EOC <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> ${ \mathrm { S E } } ,$ <sub>an</sub>d <sub>grows w</sub>ith th<sub>e roun</sub>d b<sub>u</sub>d<sub>ge</sub>t<sub>, w</sub>ith 31% t<sub>o</sub> 145% <sub>a</sub>t $B = I .$ <sub>.</sub> E<sub>ven</sub> th<sub>en,</sub> <sub>w</sub>ith <sub>on</sub>l<sub>y</sub> f<sub>our</sub> t<sub>o</sub>t<sub>a</sub>l <sub>roun</sub>d<sub>s,</sub> b<sub>a</sub>t<sub>c</sub>h<sub>e</sub>d SE h<sub>as</sub> <sub>a</sub> l<sub>ower</sub> t<sub>erm</sub>i<sub>na</sub>l EOC th<sub>an</sub> <sub>sequen</sub>ti<sub>a</sub>l KG(SA) on the $I = 8 0$ <sub>an</sub>d $I = 1 6 0$ <sub>con</sub>fi<sub>gura</sub>ti<sub>ons.</sub>

![](images/50c1cb4446cf46fc573f4228cddb3355b979d21fcad07f7495e2c9bf7b6cb3d3.jpg)

![](images/2f6451c81edab73e2e990d2fe5164e77e1db8a844a2e17073d7602c9a8630390.jpg)

![](images/adbf5e7bc0b3b3b3f02899073f36c3bb2400eded2a24eb95e786d01e8f56cd7c.jpg)  
Figure 10 EOC against the budget spent.

## 8. Concluding remarks

For ada<sub>p</sub>tive to<sub>p</sub>-� selection, we consider a unif<sub>y</sub>in<sub>g g</sub>enerative model for ratin<sub>g</sub>s (thumbs-u<sub>p</sub>/down) and <sub>pa</sub>i<sub>rw</sub>i<sub>se</sub> <sub>compar</sub>i<sub>sons</sub> <sub>an</sub>d <sub>a</sub>ll<sub>ow</sub> th<sub>e</sub> t<sub>wo</sub> i<sub>npu</sub>t t<sub>ypes</sub> t<sub>o</sub> dif<sub>er</sub> i<sub>n</sub> b<sub>o</sub>th <sub>accuracy</sub> <sub>an</sub>d <sub>cos</sub>t<sub>.</sub> W<sub>e</sub> <sub>rep</sub>l<sub>ace</sub> th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> i<sub>n</sub>t<sub>rac</sub>t<sub>a</sub>bl<sub>e</sub> d<sub>ynam</sub>i<sub>c</sub> <sub>program</sub> <sub>w</sub>ith <sub>a</sub> <sub>nove</sub>l <sub>surroga</sub>t<sub>e-</sub>b<sub>ase</sub>d <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> <sub>me</sub>th<sub>o</sub>d<sub>.</sub> D<sub>esp</sub>it<sub>e</sub> <sub>cons</sub>i<sub>s</sub>ti<sub>ng</sub> <sub>o</sub>f <sub>a</sub> <sub>sequence</sub> <sub>o</sub>f <sub>s</sub>t<sub>a</sub>t<sub>e-</sub>b<sub>ase</sub>d <sub>s</sub>t<sub>a</sub>ti<sub>c</sub> <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> <sub>pro</sub>bl<sub>ems,</sub> it i<sub>s</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y</sub> f<sub>orwar</sub>d<sub>-</sub>l<sub>oo</sub>ki<sub>ng</sub> b<sub>y</sub> <sub>rewar</sub>di<sub>ng</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-re</sub>l<sub>evan</sub>t <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y re</sub>d<sub>uc</sub>ti<sub>on.</sub> B<sub>ase</sub>d <sub>on</sub> th<sub>e surroga</sub>t<sub>e, we cons</sub>t<sub>ruc</sub>t <sub>an a</sub>ll<sub>oca</sub>ti<sub>on po</sub>li<sub>cy w</sub>h<sub>ose</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on ru</sub>l<sub>e</sub> d<sub>ecomposes</sub> i<sub>n</sub>t<sub>o</sub> i<sub>n</sub>t<sub>erpre</sub>t<sub>a</sub>bl<sub>e par</sub>t<sub>s</sub> th<sub>a</sub>t <sub>g</sub>i<sub>ve</sub> i<sub>ns</sub>i<sub>g</sub>ht i<sub>n</sub>t<sub>o exac</sub>tl<sub>y w</sub>h<sub>y a par</sub>ti<sub>cu</sub>l<sub>ar eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>was</sub> <sub>c</sub>h<sub>osen.</sub> F<sub>or</sub> th<sub>e</sub> <sub>en</sub>d<sub>game</sub> <sub>reg</sub>i<sub>me,</sub> <sub>we</sub> <sub>sw</sub>it<sub>c</sub>h t<sub>o</sub> th<sub>e</sub> k<sub>now</sub>l<sub>e</sub>d<sub>ge</sub> <sub>gra</sub>di<sub>en</sub>t <sub>po</sub>li<sub>cy,</sub> <sub>w</sub>ith <sub>a</sub> <sub>cr</sub>it<sub>er</sub>i<sub>on</sub> f<sub>or</sub> <sub>w</sub>h<sub>en</sub> t<sub>o</sub> d<sub>o</sub> <sub>so,</sub> <sub>an</sub>d <sub>ex</sub>t<sub>en</sub>d th<sub>e</sub> <sub>proce</sub>d<sub>ure</sub> t<sub>o</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h<sub>e</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>se</sub>tti<sub>ng.</sub> Fi<sub>na</sub>ll<sub>y,</sub> <sub>we</sub> <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>our</sub> <sub>me</sub>th<sub>o</sub>d <sub>per</sub>f<sub>orms</sub> <sub>we</sub>ll i<sub>n</sub> <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>ons,</sub> <sub>s</sub>t<sub>ress-</sub>t<sub>es</sub>t th<sub>e</sub> <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> <sub>ma</sub>d<sub>e</sub> i<sub>n</sub> th<sub>e</sub> <sub>ana</sub>l<sub>ys</sub>i<sub>s,</sub> <sub>an</sub>d t<sub>es</sub>t th<sub>e</sub> <sub>me</sub>th<sub>o</sub>d <sub>un</sub>d<sub>er</sub> <sub>mo</sub>d<sub>e</sub>l <sub>m</sub>i<sub>sspec</sub>ifi<sub>ca</sub>ti<sub>on.</sub>

Th<sub>ere</sub> <sub>are</sub> <sub>a</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>exc</sub>iti<sub>ng</sub> <sub>poss</sub>ibl<sub>e</sub> <sub>ex</sub>t<sub>ens</sub>i<sub>ons</sub> f<sub>or</sub> f<sub>u</sub>t<sub>ure</sub> <sub>wor</sub>k<sub>.</sub> O<sub>ne</sub> <sub>can</sub> <sub>cons</sub>id<sub>er</sub> h<sub>e</sub>t<sub>erogeneous</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>ors</sub> <sub>w</sub>h<sub>ose</sub> $\beta ^ { \circ } \mathbf { s }$ <sub>an</sub>d $\theta _ { 0 }$ <sub>are</sub> <sub>un</sub>k<sub>nown</sub> <sub>an</sub>d l<sub>earne</sub>d f<sub>rom</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons.</sub> Additi<sub>ona</sub>ll<sub>y,</sub> <sub>one</sub> <sub>may</sub> <sub>cons</sub>id<sub>er</sub> <sub>r</sub>i<sub>c</sub>h<sub>er eva</sub>l<sub>ua</sub>ti<sub>on</sub> t<sub>ypes.</sub> O<sub>ur mo</sub>d<sub>e</sub>l <sub>an</sub>d <sub>ana</sub>l<sub>ys</sub>i<sub>s ex</sub>t<sub>en</sub>d t<sub>o ra</sub>ti<sub>ngs w</sub>ith <sub>any num</sub>b<sub>er o</sub>f <sub>ca</sub>t<sub>egor</sub>i<sub>es us</sub>i<sub>ng an</sub> <sub>or</sub>d<sub>ere</sub>d <sub>pro</sub>bit <sub>mo</sub>d<sub>e</sub>l<sub>.</sub> H<sub>owever, cons</sub>id<sub>er</sub>i<sub>ng par</sub>ti<sub>a</sub>l <sub>ran</sub>ki<sub>ngs o</sub>f <sub>a su</sub>b<sub>se</sub>t l<sub>arger</sub> th<sub>an</sub> t<sub>wo</sub> it<sub>ems wou</sub>ld <sub>requ</sub>i<sub>re</sub> <sub>a</sub>dditi<sub>ona</sub>l <sub>wor</sub>k<sub>.</sub> W<sub>e</sub> <sub>a</sub>l<sub>so</sub> h<sub>ope</sub> th<sub>a</sub>t <sub>our</sub> <sub>surroga</sub>t<sub>e</sub> <sub>me</sub>th<sub>o</sub>d <sub>can</sub> b<sub>e</sub> f<sub>ur</sub>th<sub>er</sub> i<sub>mprove</sub>d <sub>upon</sub> <sub>an</sub>d <sub>app</sub>li<sub>e</sub>d t<sub>o</sub> additional Ba<sub>y</sub>esian sequential learnin<sub>g</sub> settin<sub>g</sub>s. For exam<sub>p</sub>le, in Goldber<sub>g</sub>er and Rudi (2026b) we develo<sub>p</sub> a <sub>var</sub>i<sub>an</sub>t <sub>o</sub>f it f<sub>or</sub> th<sub>e</sub> <sub>c</sub>l<sub>ass</sub>i<sub>ca</sub>l <sub>se</sub>tti<sub>ng</sub> <sub>o</sub>f t<sub>op-</sub>� <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> <sub>w</sub>ith <sub>un</sub>t<sub>rans</sub>f<sub>orme</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>ons.</sub> Th<sub>ere,</sub> it h<sub>as</sub> <sub>s</sub>t<sub>rong</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>proper</sub>ti<sub>es</sub> <sub>an</sub>d <sub>per</sub>f<sub>ormance</sub> <sub>guaran</sub>t<sub>ees,</sub> <sub>an</sub>d <sub>per</sub>f<sub>orms</sub> <sub>very</sub> <sub>we</sub>ll i<sub>n</sub> <sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>aga</sub>i<sub>ns</sub>t <sub>a</sub> <sub>s</sub>t<sub>rong</sub> <sub>se</sub>t <sub>o</sub>f b<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> f<sub>rom</sub> th<sub>e</sub> <sub>we</sub>ll<sub>-</sub>d<sub>eve</sub>l<sub>ope</sub>d <sub>ran</sub>ki<sub>ng</sub> <sub>an</sub>d <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> <sub>an</sub>d b<sub>es</sub>t<sub>-arm</sub> id<sub>en</sub>tifi<sub>ca</sub>ti<sub>on</sub> lit<sub>era</sub>t<sub>ures.</sub>

## A<sub>c</sub>kn<sub>ow</sub>l<sub>e</sub>d<sub>g</sub>m<sub>e</sub>nt<sub>s</sub>

W<sub>e</sub> th<sub>an</sub>k E<sub>nver</sub> Y¨<sub>ucesan</sub> f<sub>or</sub> hi<sub>s</sub> h<sub>e</sub>l<sub>p</sub>f<sub>u</sub>l <sub>commen</sub>t<sub>s</sub> th<sub>a</sub>t i<sub>mprove</sub>d th<sub>e</sub> <sub>paper.</sub>

## References

Ben<sub>g</sub>s V, Busa-Fekete R, El Mesaoudi-Paul A, H¨ullermeier E (2021) Preference-based online learnin<sub>g</sub> with duelin<sub>g</sub> bandits: A survey. Journal ofMachine Learning Research 22:1–108.

Brascam<sub>p</sub> HJ, Lieb EH (1975) Some inequalities for Gaussian measures and the lon<sub>g</sub>-ran<sub>g</sub>e order of the one-dimensional plasma. Functional Integration and Its Applications, 403–416.

Bubeck S, Munos R, Stoltz G (2009) Pure exploration in multi-armed bandit problems. Proceedings of the 20th International Conference on Algorithmic Learning Theory, 23–37.

Chen CH, He D, Fu M, Lee LH (2008) Eficient simulation budget allocation for selecting an optimal subset. INFORMS Journal on Computing 20:579–595.

Chen CH, Lin J, Y¨ucesan E, Chick SE (2000) Simulation bud<sub>g</sub>et allocation for further enhancin<sub>g</sub> the eficienc<sub>y</sub> of ordinal optimization. Discrete Event Dynamic Systems 10:251–270.

Chen HC, Dai L, Chen CH, Y¨ucesan E (1997) New develo<sub>p</sub>ment of o<sub>p</sub>timal com<sub>p</sub>utin<sub>g</sub> bud<sub>g</sub>et allocation for discrete event simulation. Proceedings of the 29th Winter Simulation Conference, 334–341.

Chen X, Jiao K, Lin Q (2016) Bayesian decision process for cost-eficient dynamic ranking via crowdsourcing. Journal of Machine Learning Research 17:1–40.

Chen X, Li Y, Mao J (2018) A nearl<sub>y</sub> instance o<sub>p</sub>timal al<sub>g</sub>orithm for to<sub>p</sub>-� rankin<sub>g</sub> under the multinomial lo<sub>g</sub>it model. Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms, 2504–2522.

Chen Y, Ryzhov IO (2019) Complete expected improvement converges to an optimal budget allocation. Advances in Applied Probability 51:209–235.

Chen Y, Ryzhov IO (2023) Balancing optimal large deviations in sequential selection. Management Science 69:3457– 3473<sub>.</sub>

Chick SE, Branke J, Schmidt C (2010) Sequential sam<sub>p</sub>lin<sub>g</sub> to m<sub>y</sub>o<sub>p</sub>icall<sub>y</sub> maximize the ex<sub>p</sub>ected value of information. INFORMS Journal on Computing 22:71–80.

Chick SE, Inoue K (2001) New two-stage and sequential procedures for selecting the best simulated system. Operations Research 49:732–743.

Durante D (2019) Conjugate bayes for probit regression via unified skew-normal distributions. Biometrika 106:765–779.

Fa<sub>g</sub>eot J, Gross<sub>g</sub>lauser M, Hoan<sub>g</sub> LN, Tacchi-Benard M, Villemaud O (2026) The benefits of diversit<sub>y</sub>: Combinin<sub>g</sub> ´ comparisons and ratings for eficient scoring. arXiv:2602.08033 .

Frazier PI, Powell WB, Dayanik S (2008) A knowledge-gradient policy for sequential information collection. SIAM Journal on Control and Optimization 47:2410–2439.

Gao S, Chen W (2015a) Eficient subset selection for the expected opportunity cost. Automatica 59:19–26.

Gao S, Chen W (2015b) A note on the subset selection for simulation optimization. Proceedings of the 2015 Winter Simulation Conference, 3768–3776.

Gao S, Chen W, Shi L (2017) A new budget allocation framework for the expected opportunity cost. Operations Research 65:787–803.

Genz A, Bretz F (2009) Computation of Multivariate Normal and � Probabilities, volume 195 (Springer).

Glynn PW, Juneja S (2004) A large deviations perspective on ordinal optimization. Proceedings of the 2004 Winter Simulation Conference, 577–585.

Goldberger M, Rudi N (2026a) Optimal top-� identification from pairwise comparisons. Proceedings of the International Conference on Machine Learning.

Goldber<sub>g</sub>er M, Rudi N (2026b) A sim<sub>p</sub>le and <sub>g</sub>eneral a<sub>pp</sub>roach to to<sub>p</sub>-� selection, workin<sub>g</sub> <sub>p</sub>a<sub>p</sub>er.

Groves M, Branke J (2019) Top-k selection with pairwise comparisons. European Journal of Operational Research 274<sub>:</sub>615<sub>–</sub>626<sub>.</sub>

He D, Chick SE, Chen CH (2007) O<sub>pp</sub>ortunit<sub>y</sub> cost and OCBA selection <sub>p</sub>rocedures in ordinal o<sub>p</sub>timization for a fixed number of alternative systems. IEEE Transactions on Systems, Man, and Cybernetics, Part C 37:951–961.

Heckel R, Shah NB, Ramchandran K, Wainwri<sub>g</sub>ht MJ (2019) Active rankin<sub>g</sub> from <sub>p</sub>airwise com<sub>p</sub>arisons and when parametric assumptions do not help. Annals ofStatistics 47:3099–3126.

Herbrich R, Minka T, Graepel T (2006) Trueskill: A bayesian skill rating system. Advances in Neural Information Processing Systems, 569–576.

Hong LJ, Fan W, Luo J (2021) Review on ranking and selection: A new perspective. Frontiers of Engineering Management 8:321–343.

Jones DR, Schonlau M, Welch WJ (1998) Eficient global optimization of expensive black-box functions. Journal of Global Optimization 13:455–492.

Jordan MI, Ghahramani Z, Jaakkola TS, Saul LK (1999) An introduction to variational methods for <sub>g</sub>ra<sub>p</sub>hical models. Machine Learning 37:183–233.

Kutscher T, Eid M (2024) Psychometric benefits of self-chosen rating scales over given rating scales. Behavior Research Methods 56:7440–7464.

Li D, Y¨ucesan E, Chen CH (2024) Thom<sub>p</sub>son sam<sub>p</sub>lin<sub>g</sub> <sub>p</sub>rocedures for rankin<sub>g</sub> and selection with <sub>p</sub>airwise com<sub>p</sub>arisons and binary outcomes. Proceedings ofthe 2024 Winter Simulation Conference, 3336–3345.

Li K, Zhang X, Li G (2018) A rating-ranking method for crowdsourced top-� computation. Proceedings of the 2018 International Conference on Management of Data, 975–990.

Li Z, Fan W, Hon<sub>g</sub> LJ (2025) The (sur<sub>p</sub>risin<sub>g</sub>) sam<sub>p</sub>le o<sub>p</sub>timalit<sub>y</sub> of <sub>g</sub>reed<sub>y</sub> <sub>p</sub>rocedures for lar<sub>g</sub>e-scale rankin<sub>g</sub> and selection. Management Science 71:1238–1259.

Liddell TM, Kruschke JK (2018) Analyzing ordinal data with metric models: What could possibly go wrong? Journal ofExperimental Social Psychology 79:328–348.

Likert R (1932) A technique for the measurement of attitudes. Archives of Psychology 22:5–55.

Liu H, Xing Z, Xiao H, Chew EP (2024) Marginal improvement procedures for top-� selection. Automatica 170.

Minka TP (2001) Expectation propagation for approximate Bayesian inference. Proceedings of the 17th Conference on Uncertainty in Artificial Intelligence, 362–369.

Mohajer S, Suh C, Elmahdy A (2017) Active learning for top-� rank aggregation from noisy comparisons. Proceedings of the International Conference on Machine Learning, 2488–2497.

Mullahy J (2017) Marginal efects in multivariate probit models. Empirical Economics 52:447–461.

Pen<sub>g</sub> Y, Chen CH, Fu MC, Hu JQ (2018) Gradient-based m<sub>y</sub>o<sub>p</sub>ic allocation <sub>p</sub>olic<sub>y</sub>: An eficient sam<sub>p</sub>lin<sub>g</sub> <sub>p</sub>rocedure in a low-confidence scenario. IEEE Transactions on Automatic Control 63:3091–3097.

Perez-Ortiz M, Mikhailiuk A, Zerman E, Hulusic V, Valenzise G, Mantiuk RK (2020) From <sub>p</sub>airwise com<sub>p</sub>arisons and´ rating to a unified quality scale. IEEE Transactions on Image Processing 29:1139–1151.

Powell WB, Ryzhov IO (2012) Optimal Learning (Wiley).

Priekule L, Meisel S (2017) A Bayesian ranking and selection problem with pairwise comparisons. Proceedings of the 2017 Winter Simulation Conference, 2149–2160.

Qin C, Klabjan D, Russo D (2017) Improving the expected improvement algorithm. Advances in Neural Information Processing Systems, volume 30.

Rao CR (1973) Linear Statistical Inference and Its Applications (Wiley), 2nd edition.

Ren W, Liu J, Shrof NB (2018) PAC rankin<sub>g</sub> from <sub>p</sub>airwise and listwise queries: Lower bounds and u<sub>pp</sub>er bounds. arXiv preprint arXiv:1806.02970 .

Ren W, Liu J, Shrof NB (2020) The sample complexity of best-� items selection from pairwise comparisons. Proceedings ofthe International Conference on Machine Learning, 8051–8072.

Revilla MA, Saris WE, Krosnick JA (2014) Choosing the number of categories in agree–disagree scales. Sociological Methods & Research 43:73–97.

Robbins H (1952) Some aspects of the sequential design of experiments. Bulletin of the American Mathematical Society 58<sub>:</sub>527<sub>–</sub>535<sub>.</sub>

Ryzhov IO (2016) On the convergence rates of expected improvement methods. Operations Research 64:1515–1528.

Saha A, Gopalan A (2019) PAC battling bandits in the Plackett-Luce model. Proceedings of the 30th International Conference on Algorithmic Learning Theory, 700–737.

Saha A, Gopalan A (2020) From PAC to instance-optimal sample complexity in the Plackett-Luce model. Proceedings of the International Conference on Machine Learning, 8367–8376.

Thurstone LL (1927) A law of comparative judgment. Psychological Review 34:273–286.

Tierney L, Kadane JB (1986) Accurate approximations for posterior moments and marginal densities. Journal of the American Statistical Association 81:82–86.

Torgerson WS (1958) Theory and Methods of Scaling (Wiley).

White D, Wu M, Novoseller E, Lawhern VJ, Wa<sub>y</sub>towich N, Cao Y (2024) Ratin<sub>g</sub>-based reinforcement learnin<sub>g</sub>. Proceedings of the AAAI Conference on Artificial Intelligence, 10207–10215.

Zhan<sub>g</sub> G, Pen<sub>g</sub> Y, Zhan<sub>g</sub> J, Zhou E (2023) As<sub>y</sub>m<sub>p</sub>toticall<sub>y</sub> o<sub>p</sub>timal sam<sub>p</sub>lin<sub>g</sub> <sub>p</sub>olic<sub>y</sub> for selectin<sub>g</sub> to<sub>p</sub>-� alternatives. INFORMS Journal on Computing 35:1261–1285.

Zhan<sub>g</sub> S, Lee LH, Chew EP, Xu J, Chen CH (2016) A simulation bud<sub>g</sub>et allocation <sub>p</sub>rocedure for enhancin<sub>g</sub> the eficiency of optimal subset selection. IEEE Transactions on Automatic Control 61:62–75.

## A<sub>pp</sub>endix contents

A B<sub>a</sub>t<sub>c</sub>hin<sub>g ex</sub>t<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n 35   
A<sub>.</sub>1 S<sub>urroga</sub>t<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> 35   
A<sub>.</sub>2 Th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h<sub>e</sub>d <sub>en</sub>d<sub>game</sub> 36   
B Su<sub>pp</sub>lements to §2 41   
B<sub>.</sub>1 D<sub>er</sub>i<sub>va</sub>ti<sub>ons o</sub>f <sub>ca</sub>lib<sub>ra</sub>ti<sub>on</sub> f<sub>ormu</sub>l<sub>as .</sub> 41   
C Proofs of results in §3 43   
D P<sub>roo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 2 48   
D<sub>.</sub>1 N<sub>o</sub>t<sub>a</sub>ti<sub>on .</sub> 48   
D<sub>.</sub>2 P<sub>roo</sub>f <sub>roa</sub>d<sub>map</sub> 49   
D<sub>.</sub>3 S<sub>uppo</sub>rtin<sub>g</sub> l<sub>e</sub>mm<sub>as</sub> 50   
D<sub>.</sub>4 P<sub>roo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 2 57   
E Su<sub>pp</sub>lement to $\ S 7$ 59   
E<sub>.</sub>1 All<sub>oca</sub>ti<sub>ons over</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>.</sub> 59   
E<sub>.</sub>2 All<sub>oca</sub>ti<sub>on o</sub>f SA <sub>versus</sub> KG 64   
E<sub>.</sub>3 C<sub>ompar</sub>i<sub>son</sub> t<sub>o ex</sub>t<sub>erna</sub>l b<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> 66

## A<sub>ppe</sub>ndi<sub>x</sub> A<sub>:</sub> B<sub>a</sub>t<sub>c</sub>hin<sub>g</sub> <sub>ex</sub>t<sub>e</sub>n<sub>s</sub>i<sub>o</sub>n

I<sub>n</sub> thi<sub>s</sub> <sub>sec</sub>ti<sub>on</sub> <sub>we</sub> <sub>presen</sub>t th<sub>e</sub> <sub>a</sub>dditi<sub>ona</sub>l t<sub>ec</sub>h<sub>n</sub>i<sub>ca</sub>l d<sub>e</sub>t<sub>a</sub>il<sub>s</sub> <sub>regar</sub>di<sub>ng</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>hi<sub>ng</sub> <sub>ex</sub>t<sub>ens</sub>i<sub>on</sub> <sub>o</sub>f $\ S 6 .$

## A<sub>.</sub>1<sub>.</sub> S<sub>u</sub>rro<sub>g</sub>ate batch constr<sub>u</sub>ction

Th<sub>roug</sub>h<sub>ou</sub>t th<sub>e</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> <sub>o</sub>f <sub>a</sub> b<sub>a</sub>t<sub>c</sub>h<sub>,</sub> <sub>we</sub> h<sub>o</sub>ld $\pmb { \mu } _ { t - 1 } , S _ { k } ^ { * } ( \pmb { \mu } _ { t - 1 } )$ <sub>,</sub> <sub>an</sub>d $\left( q _ { t } , b _ { t } \right)$ fixed<sub>,</sub> as in §3.3. For a candidate b<sub>a</sub>t<sub>c</sub>h <sub>�,</sub> it<sub>s</sub> <sub>expec</sub>t<sub>e</sub>d <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>covar</sub>i<sub>ance</sub> i<sub>s</sub>

$$
{ \overline { { \Sigma } } } _ { t } ( n , p _ { t - 1 } ) : = \operatorname { \mathbb { E } } \bigl [ \operatorname { C o v } ( \pmb { \theta } \mid p _ { t } ) \bigr | n , p _ { t - 1 } \bigr ] .
$$

H<sub>ere</sub> $p _ { t }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>a</sub>ft<sub>er</sub> <sub>o</sub>b<sub>serv</sub>i<sub>ng</sub> th<sub>e</sub> <sub>ou</sub>t<sub>comes</sub> <sub>o</sub>f <sub>�</sub> <sub>an</sub>d th<sub>e</sub> <sub>expec</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>s</sub> <sub>over</sub> <sub>a</sub>ll <sub>poss</sub>ibl<sub>e</sub> <sub>ou</sub>t<sub>comes</sub> <sub>o</sub>f th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h<sub>.</sub> F<sub>or</sub> <sub>a</sub> <sub>roun</sub>d b<sub>u</sub>d<sub>ge</sub>t $B _ { t } .$ <sub>,</sub> <sub>su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> thi<sub>s</sub> <sub>covar</sub>i<sub>ance</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> <sub>surroga</sub>t<sub>e</sub> <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> i<sub>n</sub>t<sub>eger</sub> p<sup>ro</sup>g<sup>ram</sup>

$$
\begin{array} { r l } { \underset { n } { \operatorname* { m i n } } } & { \displaystyle \sum _ { \ell = 1 } ^ { I } \mathbb { E } _ { \pmb { \theta } \sim N ( \mu _ { t - 1 } , \overline { { \Sigma } } _ { t } ( n , p _ { t - 1 } ) ) } \big [ \big ( \theta _ { \ell } - a _ { q _ { t } , b _ { t } } ( \pmb { \theta } ) \big ) ^ { + } \big ] } \\ { \mathrm { s . t . } } & { \displaystyle \sum _ { ( i , j ) \in { \cal I } } r _ { J } n _ { i j } \leq B _ { t } , \qquad n _ { i j } \in \mathbb { Z } _ { \geq 0 } \quad \mathrm { f o r ~ a l l ~ } ( i , j ) \in { \cal I } . } \end{array}\tag{A.1}
$$

As in (10), we have dro<sub>pp</sub>ed the terms that do not de<sub>p</sub>end on �. We a<sub>pp</sub>roximatel<sub>y</sub> solve (A.1) b<sub>y</sub> constructin<sub>g</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h <sub>one</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>a</sub>t <sub>a</sub> ti<sub>me,</sub> <sub>as</sub> i<sub>n</sub> Al<sub>gor</sub>ith<sub>m</sub> 4<sub>.</sub> T<sub>o</sub> <sub>c</sub>h<sub>oose</sub> <sub>eac</sub>h <sub>a</sub>dditi<sub>ona</sub>l <sub>eva</sub>l<sub>ua</sub>ti<sub>on,</sub> <sub>we</sub> <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub> the expected posterior covariance in the objective by $\Sigma _ { t } ^ { \prime } ( n )$ usin<sub>g</sub> (12).

C<sub>ons</sub>id<sub>er</sub> <sub>a</sub>ddi<sub>ng</sub> <sub>an</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> $( i , j )$ t<sub>o</sub> <sub>�.</sub> If th<sub>e</sub> <sub>ou</sub>t<sub>comes</sub> <sub>o</sub>f <sub>�</sub> <sub>were</sub> <sub>o</sub>b<sub>serve</sub>d<sub>,</sub> th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> $p _ { t }$ <sub>wou</sub>ld <sub>g</sub>i<sub>ve</sub> <sub>expec</sub>t<sub>e</sub>d <sub>covar</sub>i<sub>ance</sub> $\overline { { \Sigma } } _ { t } ( ( i , j ) , p _ { t } )$ <sub>a</sub>ft<sub>er</sub> th<sub>e</sub> <sub>a</sub>dditi<sub>ona</sub>l <sub>eva</sub>l<sub>ua</sub>ti<sub>on.</sub> A<sub>verag</sub>i<sub>ng</sub> <sub>over</sub> th<sub>e</sub> <sub>ou</sub>t<sub>comes</sub> <sub>o</sub>f � <sub>g</sub><sup>i</sup>ves

$$
\begin{array} { r } { \overline { { \Sigma } } _ { t } ( n + F ^ { i j } , p _ { t - 1 } ) = \mathbb { E } [ \overline { { \Sigma } } _ { t } ( ( i , j ) , p _ { t } ) | n , p _ { t - 1 } ] . } \end{array}
$$

H<sub>ere</sub> $\overline { { \Sigma } } _ { t } ( ( i , j ) , p _ { t } )$ <sub>averages over</sub> th<sub>e</sub> t<sub>wo ou</sub>t<sub>comes o</sub>f $( i , j )$ <sub>g</sub><sup>i</sup>ven $p _ { t }$ <sub>, w</sub>hil<sub>e</sub> th<sub>e ou</sub>t<sub>er expec</sub>t<sub>a</sub>ti<sub>on averages</sub> <sub>over</sub> th<sub>e ou</sub>t<sub>comes o</sub>f th<sub>e a</sub>l<sub>rea</sub>d<sub>y comm</sub>itt<sub>e</sub>d b<sub>a</sub>t<sub>c</sub>h <sub>�.</sub>

W<sub>e</sub> <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub> thi<sub>s</sub> <sub>expec</sub>t<sub>a</sub>ti<sub>on</sub> b<sub>y</sub> fi<sub>x</sub>i<sub>ng</sub> th<sub>e</sub> <sub>covar</sub>i<sub>ance</sub> <sub>a</sub>ft<sub>er</sub> <sub>�</sub> <sub>a</sub>t $\Sigma _ { t } ^ { \prime } ( n )$ <sub>.</sub> W<sub>e</sub> th<sub>e</sub>n <sub>co</sub>m<sub>pu</sub>t<sub>e</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>covar</sub>i<sub>ance</sub> <sub>a</sub>ft<sub>er</sub> <sub>a</sub>ddi<sub>ng</sub> $( i , j )$ b<sub>y</sub> <sub>averag</sub>i<sub>ng</sub> <sub>over</sub> it<sub>s</sub> t<sub>wo</sub> <sub>poss</sub>ibl<sub>e</sub> <sub>ou</sub>t<sub>comes</sub> <sub>un</sub>d<sub>er</sub> $N ( \mu _ { t - 1 } , \Sigma _ { t } ^ { \prime } ( n ) )$ , <sub>g</sub><sup>i</sup>v<sup>i</sup>n<sub>g</sub>

$$
\overline { { \Sigma } } _ { t } \left( n + F ^ { i j } , p _ { t - 1 } \right) \approx \overline { { \Sigma } } _ { t } \left( ( i , j ) , N ( \mu _ { t - 1 } , \Sigma _ { t } ^ { \prime } ( n ) ) \right) = \Sigma _ { t } ^ { \prime } ( n + F ^ { i j } ) ,
$$

where the equalit<sub>y</sub> follows from Lemma 2 and <sub>g</sub>ives (12). The first u<sub>p</sub>date is exact because $n = 0$ <sub>an</sub>d $\Sigma _ { t } ^ { \prime } ( 0 ) = \Sigma _ { t - 1 }$ <sub>.</sub> N<sub>o</sub>t<sub>e</sub> th<sub>a</sub>t f<sub>or</sub> th<sub>e</sub> <sub>same</sub> b<sub>a</sub>t<sub>c</sub>h <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> <sub>�,</sub> th<sub>e</sub> <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> $\Sigma _ { t } ^ { \prime } ( n )$ <sub>can</sub> dif<sub>er</sub> d<sub>epen</sub>di<sub>ng</sub> <sub>on</sub> th<sub>e</sub> <sub>or</sub>d<sub>er</sub> i<sub>n</sub> <sub>w</sub>hi<sub>c</sub>h <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> <sub>are</sub> <sub>comm</sub>itt<sub>e</sub>d<sub>.</sub>

## A<sub>.</sub>2<sub>.</sub> Th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h<sub>e</sub>d <sub>e</sub>nd<sub>ga</sub>m<sub>e</sub>

I<sub>n</sub> thi<sub>s su</sub>b<sub>sec</sub>ti<sub>on,</sub> it i<sub>s conven</sub>i<sub>en</sub>t t<sub>o use an a</sub>lt<sub>erna</sub>ti<sub>ve no</sub>t<sub>a</sub>ti<sub>on</sub> f<sub>or a</sub> b<sub>a</sub>t<sub>c</sub>h<sub>.</sub> Ab<sub>ove, a</sub> b<sub>a</sub>t<sub>c</sub>h <sub>was represen</sub>t<sub>e</sub>d b<sub>y</sub> it<sub>s coun</sub>t <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $n ( t )$ <sub>, w</sub>hi<sub>c</sub>h <sub>recor</sub>d<sub>s</sub> h<sub>ow many</sub> ti<sub>mes eac</sub>h <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> $( i , j )$ i<sub>s con</sub>d<sub>uc</sub>t<sub>e</sub>d i<sub>n roun</sub>d �<sub>.</sub> H<sub>ere</sub> it <sub>w</sub>ill b<sub>e use</sub>f<sub>u</sub>l t<sub>o</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h b<sub>e</sub>t<sub>ween repea</sub>t<sub>s o</sub>f th<sub>e same eva</sub>l<sub>ua</sub>ti<sub>on, so we</sub> fi<sub>x an or</sub>d<sub>er</sub>i<sub>ng o</sub>f th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons, wr</sub>iti<sub>ng a</sub> b<sub>a</sub>t<sub>c</sub>h <sub>o</sub>f � <sub>eva</sub>l<sub>ua</sub>ti<sub>ons as</sub> $( i _ { 1 } , j _ { 1 } ) , ( i _ { 2 } , j _ { 2 } ) , . . . , ( i _ { U } , j _ { U } )$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> <sub>eac</sub>h $( i _ { u } , j _ { u } ) \in \mathcal { I }$ <sub>an</sub>d $J _ { u } : = \operatorname* { m i n } ( j _ { u } , 1 )$ <sub>.</sub> Th<sub>e assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>coun</sub>t <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> i<sub>s</sub> $\begin{array} { r } { n _ { i j } ( t ) = \sum _ { u = 1 } ^ { U } \mathbb { 1 } \{ ( i _ { u } , j _ { u } ) = ( i , j ) \} } \end{array}$ f<sub>or</sub> <sub>eac</sub>h $( i , j ) \in \mathcal { I }$

C<sub>ons</sub>id<sub>er a can</sub>did<sub>a</sub>t<sub>e</sub> f<sub>eas</sub>ibl<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h $( i _ { 1 } , j _ { 1 } ) , \dotsc , ( i _ { U } , j _ { U } )$ <sub>w</sub>ith <sub>coun</sub>t <sub>ma</sub>t<sub>r</sub>i<sub>x �.</sub> L<sub>e</sub>t $X _ { u } \in \{ - 1 , 1 \}$ d<sub>eno</sub>t<sub>e</sub> th<sub>e ou</sub>t<sub>come o</sub>f th<sub>e �</sub>th <sub>eva</sub>l<sub>ua</sub>ti<sub>on an</sub>d d<sub>e</sub>fi<sub>ne</sub> $X : = ( X _ { 1 } , \ldots , X _ { U } ) ^ { \intercal }$ <sup>⊤</sup>. For an outcome vector �, write $\mu _ { t } ( { \boldsymbol { x } } \mid$ $n , p _ { t - 1 } \big ) = \mathbb { E } [ \pmb { \theta } \mid X = \pmb { x } , n , p _ { t - 1 } ]$

A.2.1. The m<sub>y</sub>o<sub>p</sub>ic value of a batch We be<sub>g</sub>in b<sub>y</sub> writin<sub>g</sub> the m<sub>y</sub>o<sub>p</sub>ic value function of §2.3 usin<sub>g</sub> th<sub>e</sub> <sub>no</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub>d <sub>a</sub>b<sub>ove.</sub> Th<sub>e</sub> <sub>myop</sub>i<sub>c</sub> <sub>va</sub>l<sub>ue</sub> f<sub>unc</sub>ti<sub>on</sub> <sub>w</sub>hi<sub>c</sub>h th<sub>e</sub> KG <sub>po</sub>li<sub>cy</sub> <sub>max</sub>i<sub>m</sub>i<sub>zes</sub> <sub>eac</sub>h <sub>roun</sub>d (reformulated from §2.3) is

$$
V _ { \mathrm { m y o p i c } } ( n | p _ { t - 1 } ) = \sum _ { x \in \{ - 1 , 1 \} ^ { U } } \mathbb { P } ( X = x \mid n , p _ { t - 1 } ) \sum _ { i \in S _ { k } ^ { * } ( \mu _ { t } ( x | n , p _ { t - 1 } ) ) } \mu _ { t , i } ( x \mid n , p _ { t - 1 } ) - \sum _ { i \in S _ { k } ^ { * } ( \mu _ { t - 1 } ) } \mu _ { t - 1 , i } ,\tag{A.2}
$$

<sub>w</sub>ith $V _ { \mathrm { m y o p i c } } ( 0 \mid p _ { t - 1 } ) = 0$ <sub>.</sub> P<sub>ropos</sub>iti<sub>on</sub> ${ \mathrm { A . 1 } }$ , which a<sub>pp</sub>lies the results of Mullah<sub>y</sub> (2017), <sub>g</sub>ives ex<sub>p</sub>licit <sub>express</sub>i<sub>ons</sub> f<sub>or</sub> th<sub>e</sub> <sub>ou</sub>t<sub>come-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>quan</sub>titi<sub>es</sub> $\mathbb { P } ( X = x \mid n , p _ { t - 1 } )$ <sub>an</sub>d $\mu _ { t } ( { \boldsymbol { \mathbf { \mathit { x } } } } \mid n , p _ { t - 1 } )$

PROPOSITION A.1. Define $C \in \mathbb { R } ^ { U \times U } b y$

$$
C _ { u u ^ { \prime } } = \beta _ { J u } \beta _ { J u ^ { \prime } } ( e _ { i _ { u } } - e _ { j _ { u } } ) ^ { \top } \Sigma _ { t - 1 } ( e _ { i _ { u ^ { \prime } } } - e _ { j _ { u ^ { \prime } } } ) + \mathbb { 1 } \{ u = u ^ { \prime } \} , \qquad u , u ^ { \prime } = 1 , \ldots , U .
$$

For any $x \in \{ - 1 , 1 \} ^ { U } , l e t \ D : = \mathrm { d i a g } \big ( \sqrt { C _ { 1 1 } } , \ldots , \sqrt { C _ { U U } } \big ) , D _ { x } : = \mathrm { d i a g } ( x _ { 1 } , \ldots , x _ { U } ) , \widetilde { C } _ { x } : = D _ { x } D ^ { - 1 } C D ^ { - 1 } D _ { x } ,$ , and $\pmb { \alpha } ( \pmb { x } ) : = ( \alpha _ { 1 } ( \pmb { x } ) , \dots , \alpha _ { U } ( \pmb { x } ) ) ^ { \top }$ where $\begin{array} { r } { \alpha _ { u } ( \pmb { x } ) = \frac { x _ { u } \beta _ { J u } ( \mu _ { t - 1 , i _ { u } } - \mu _ { t - 1 , j _ { u } } ) } { \sqrt { C _ { u u } } } } \end{array}$ for each �. Let $\Phi _ { U } ( \cdot ; \widetilde { C } _ { x } )$ denote the CDF

ofa mean-zero �-variate normal vector with covariance matrix ${ \widetilde { C } } _ { x } ,$ let $\pmb { \alpha } _ { - u } ( \pmb { x } )$ denote the vector $\alpha ( x )$ with its �th component removed, and let $( \widetilde { C } _ { x } ) _ { - u | u } : = ( \widetilde { C } _ { x } ) _ { - u , - u } - ( \widetilde { C } _ { x } ) _ { - u , u } ( \widetilde { C } _ { x } ) _ { u , - u }$ . Finally, define $\boldsymbol { \lambda _ { x } } \in \mathbb { R } ^ { U }$ by

$$
( \lambda _ { x } ) _ { u } = \frac { x _ { u } } { \sqrt { C _ { u u } } } \cdot \frac { \phi ( \alpha _ { u } ( x ) ) \Phi _ { U - 1 } \Big ( \alpha _ { - u } ( x ) - ( \widetilde { C } _ { x } ) _ { - u , u } \alpha _ { u } ( x ) ; ( \widetilde { C } _ { x } ) _ { - u | u } \Big ) } { \Phi _ { U } ( \alpha ( x ) ; \widetilde { C } _ { x } ) } , \qquad u = 1 , . . . , U .
$$

For $U = 1$ , the $\Phi _ { U - 1 }$ factor is 1. Then

$$
\begin{array} { r l } { ( i ) } & { { } \mathbb { P } ( X = x \mid n , p _ { t - 1 } ) = \Phi _ { U } ( \alpha ( x ) ; \widetilde { C } _ { x } ) , } \end{array}\tag{A.3}
$$

(ii)

$$
\mu _ { t } ( \pmb { x } \mid n , p _ { t - 1 } ) = \pmb { \mu _ { t - 1 } } + \sum _ { u = 1 } ^ { U } \beta _ { J u } ( \lambda _ { x } ) _ { u } \Sigma _ { t - 1 } ( \pmb { e _ { i _ { u } } } - \pmb { e _ { j _ { u } } } ) .\tag{A.4}
$$

Substitutin<sub>g</sub> (A.3) and (A.4) into (A.2) <sub>g</sub>ives an ex<sub>p</sub>licit ex<sub>p</sub>ression for $V _ { \mathrm { m y o p i c } } ( \cdot | \ : p _ { t - 1 } )$ <sub>.</sub> Si<sub>nce</sub> th<sub>e genera</sub>l f<sub>ormu</sub>l<sub>a</sub> i<sub>s no</sub>t <sub>easy</sub> t<sub>o rea</sub>d i<sub>n</sub> it<sub>s</sub> f<sub>u</sub>ll �<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l f<sub>orm, we</sub> fi<sub>rs</sub>t <sub>wor</sub>k <sub>ou</sub>t th<sub>e cases</sub> $U = 1$ <sub>an</sub>d $U = 2$ <sub>.</sub> Th<sub>ese</sub> <sub>examp</sub>l<sub>es</sub> <sub>a</sub>l<sub>so</sub> <sub>po</sub>i<sub>n</sub>t t<sub>o</sub> th<sub>e</sub> i<sub>n</sub>t<sub>rac</sub>t<sub>a</sub>bilit<sub>y</sub> f<sub>or</sub> l<sub>arger</sub> b<sub>a</sub>t<sub>c</sub>h<sub>es.</sub> E<sub>xac</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>o</sub>f $( \mathsf { A } . 2 )$ re<sub>q</sub>u<sup>i</sup>res a sum over $2 ^ { U }$ <sub>ou</sub>t<sub>come vec</sub>t<sub>ors, w</sub>ith <sub>a</sub> �<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e norma</sub>l CDF i<sub>n eac</sub>h t<sub>erm.</sub> Th<sub>ese</sub> CDF<sub>s mus</sub>t b<sub>e</sub> evaluated numericall<sub>y</sub>, which becomes ex<sub>p</sub>ensive as � <sub>g</sub>rows (Genz and Bretz 2009). This motivates the a<sub>pp</sub>roximate value function introduced in §A.2.2.

Single evaluation $( U = 1 )$ . Consider a batch with a single evaluation $( i _ { 1 } , j _ { 1 } ) = ( i , j )$ <sub>.</sub> Th<sub>en</sub> $C _ { 1 1 } =$ $1 + \beta _ { J } ^ { 2 } \sigma _ { t - 1 , i j } ^ { 2 }$ <sub>an</sub>d $\alpha _ { 1 } ( x ) = x \beta _ { \jmath } ( \mu _ { t - 1 , i } - \mu _ { t - 1 , j } ) / \sqrt { C _ { 1 1 } }$ f<sub>or</sub> $x \in \{ - 1 , 1 \}$ , so

$$
\mathbb { P } ( X = x \mid n , p _ { t - 1 } ) = \Phi ( \alpha _ { 1 } ( x ) ) , \quad \lambda _ { x } = \frac { x \phi ( \alpha _ { 1 } ( x ) ) } { \Phi ( \alpha _ { 1 } ( x ) ) \sqrt { C _ { 1 1 } } } ,
$$

$$
\mu _ { t } ( x \mid n , p _ { t - 1 } ) = \mu _ { t - 1 } + { \frac { x \beta _ { \jmath } \phi ( \alpha _ { 1 } ( x ) ) } { \Phi ( \alpha _ { 1 } ( x ) ) { \sqrt { C _ { 1 1 } } } } } \Sigma _ { t - 1 } ( e _ { i } - e _ { j } ) .
$$

Two evaluations $( U = 2 )$ . Consider a batch with two evaluations $( i _ { u } , j _ { u } ) , u = 1 , 2$ <sub>.</sub> Th<sub>en</sub> $C _ { u u } = 1 +$ $\beta _ { J u } ^ { 2 } \sigma _ { t - 1 , i _ { u } j _ { u } } ^ { 2 } , C _ { 1 2 } = \beta _ { J _ { 1 } } \beta _ { J _ { 2 } } ( e _ { i _ { 1 } } - e _ { j _ { 1 } } ) ^ { \top } \Sigma _ { t - 1 } ( e _ { i _ { 2 } } - e _ { j _ { 2 } } )$ <sub>,</sub> <sub>an</sub>d $\alpha _ { u } ( \pmb { x } ) = x _ { u } \beta _ { J u } ( \mu _ { t - 1 , i _ { u } } - \mu _ { t - 1 , j _ { u } } ) / \sqrt { C _ { u u } }$ <sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>corre</sub>l<sub>a</sub>ti<sub>on</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $\widetilde { C } _ { x }$ h<sub>as</sub> <sub>o</sub>f<sub>-</sub>di<sub>agona</sub>l <sub>en</sub>t<sub>ry</sub> <sub>�1�2�12 w</sub>ith $\rho _ { 1 2 } : = C _ { 1 2 } / \sqrt { C _ { 1 1 } C _ { 2 2 } }$ <sub>.</sub> H<sub>ence,</sub> f<sub>or</sub> ${ \pmb x } = ( x _ { 1 } , x _ { 2 } ) \in$ $\{ - 1 , 1 \} ^ { 2 } , { \mathbb { P } } ( X = x \mid n , p _ { t - 1 } ) = \Phi _ { 2 } { \big ( } \alpha _ { 1 } ( x ) , \alpha _ { 2 } ( x ) ; x _ { 1 } x _ { 2 } \rho _ { 1 2 } { \big ) }$ <sub>an</sub>d<sub>,</sub> f<sub>or</sub> $\{ u , u ^ { \prime } \} = \{ 1 , 2 \}$

$$
( \lambda _ { x } ) _ { u } = \frac { x _ { u } } { \sqrt { C _ { u u } } } \cdot \frac { \phi ( \alpha _ { u } ( x ) ) \Phi \Big ( \frac { \alpha _ { u ^ { \prime } } ( x ) - x _ { 1 } x _ { 2 } \rho _ { 1 2 } \alpha _ { u } ( x ) } { \sqrt { 1 - \rho _ { 1 2 } ^ { 2 } } } \Big ) } { \Phi _ { 2 } \big ( \alpha _ { 1 } ( x ) , \alpha _ { 2 } ( x ) ; x _ { 1 } x _ { 2 } \rho _ { 1 2 } \big ) } , \quad \mu _ { t } ( x \mid n , p _ { t - 1 } ) = \mu _ { t - 1 } + \sum _ { u = 1 } ^ { 2 } \beta _ { j _ { u } } ( \lambda _ { x } ) _ { u } \Sigma _ { t - 1 } ( e _ { i _ { u } } - e _ { j _ { u } } ) .
$$

A<sub>.</sub>2<sub>.</sub>2<sub>.</sub> D<sub>eco</sub>m<sub>pos</sub>in<sub>g</sub> th<sub>e</sub> m<sub>yop</sub>i<sub>c</sub> b<sub>a</sub>t<sub>c</sub>h <sub>va</sub>l<sub>ue</sub> T<sub>o</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> <sub>a</sub> t<sub>rac</sub>t<sub>a</sub>bl<sub>e</sub> <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> <sub>o</sub>f $V _ { \mathrm { m y o p i c } } ( \cdot | \ : p _ { t - 1 } )$ <sub>w</sub>h<sub>en</sub> � i<sub>s</sub> l<sub>arge,</sub> <sub>we</sub> fi<sub>rs</sub>t d<sub>ecompose</sub> th<sub>e</sub> <sub>myop</sub>i<sub>c</sub> <sub>va</sub>l<sub>ue</sub> <sub>o</sub>f th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h i<sub>n</sub>t<sub>o</sub> t<sub>erms,</sub> <sub>one</sub> f<sub>or</sub> <sub>eac</sub>h <sub>su</sub>b<sub>se</sub>t <sub>o</sub>f th<sub>e</sub> batch, and refer to the size of the subset (how man<sub>y</sub> evaluations are in it) as the order of its term. We then <sup>a</sup>pp<sup>roximate</sup> $V _ { \mathrm { m y o p i c } } ( n | p _ { t - 1 } )$ b<sub>y</sub> t<sub>runca</sub>ti<sub>ng</sub> thi<sub>s</sub> d<sub>ecompos</sub>iti<sub>on</sub> <sub>a</sub>t <sub>some</sub> <sub>or</sub>d<sub>er</sub> <sub>�,</sub> k<sub>eep</sub>i<sub>ng</sub> <sub>a</sub>ll t<sub>erms</sub> <sub>o</sub>f <sub>or</sub>d<sub>er</sub> <sub>a</sub>t <sub>mos</sub>t <sub>�</sub> <sub>an</sub>d di<sub>scar</sub>di<sub>ng</sub> th<sub>e</sub> <sub>res</sub>t<sub>.</sub>

Fi<sub>x a</sub> b<sub>a</sub>t<sub>c</sub>h <sub>w</sub>ith <sub>coun</sub>t <sub>ma</sub>t<sub>r</sub>i<sub>x � an</sub>d <sub>or</sub>d<sub>ere</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> $( i _ { 1 } , j _ { 1 } ) , \dotsc , ( i _ { U } , j _ { U } )$ <sub>.</sub> F<sub>or</sub> <sub>any</sub> <sub>su</sub>b<sub>se</sub>t $G \subseteq \{ 1 , \ldots , U \}$ , l<sub>e</sub>t $n _ { G }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> <sub>su</sub>b<sub>-</sub>b<sub>a</sub>t<sub>c</sub>h <sub>cons</sub>i<sub>s</sub>ti<sub>ng</sub> <sub>o</sub>f th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> <sub>w</sub>h<sub>ose</sub> <sub>pos</sub>iti<sub>ons</sub> <sub>are</sub> i<sub>n</sub> $G .$ <sub>,</sub> <sub>an</sub>d f<sub>or</sub> <sub>any</sub> <sub>su</sub>b<sub>se</sub>t $H \subseteq \{ 1 , \ldots , U \}$ , define its interaction term

$$
\Gamma _ { H } : = \sum _ { G \subseteq H } ( - 1 ) ^ { | H | - | G | } V _ { \mathrm { m y o p i c } } ( n _ { G } \mid p _ { t - 1 } ) .
$$

T<sub>o</sub> ill<sub>us</sub>t<sub>ra</sub>t<sub>e</sub> <sub>w</sub>h<sub>a</sub>t th<sub>e</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> t<sub>erms</sub> <sub>represen</sub>t<sub>,</sub> <sub>we</sub> <sub>wr</sub>it<sub>e</sub> th<sub>em</sub> <sub>exp</sub>li<sub>c</sub>itl<sub>y</sub> f<sub>or</sub> th<sub>ree</sub> <sub>examp</sub>l<sub>es.</sub> Th<sub>e</sub> <sub>or</sub>d<sub>er-zero</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> t<sub>erm</sub> i<sub>s</sub> th<sub>e</sub> <sub>va</sub>l<sub>ue</sub> <sub>w</sub>ith <sub>no</sub> f<sub>ur</sub>th<sub>er</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on,</sub> $\Gamma _ { \emptyset } = V _ { \mathrm { m y o p i c } } ( 0 \mid p _ { t - 1 } ) = 0$ <sub>.</sub> F<sub>or</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> $u \in \{ 1 , . . . , U \} , \Gamma _ { \{ u \} } = V _ { \mathrm { m y o p i c } } ( n _ { \{ u \} } \mid p _ { t - 1 } ) - V _ { \mathrm { m y o p i c } } ( 0 \mid p _ { t - 1 } )$ i<sub>s</sub> it<sub>s s</sub>t<sub>an</sub>d<sub>a</sub>l<sub>one va</sub>l<sub>ue o</sub>f i<sub>n</sub>f<sub>orma</sub>ti<sub>on.</sub> F<sub>or a</sub> <sub>pa</sub>i<sub>r</sub> <sub>o</sub>f di<sub>s</sub>ti<sub>nc</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> $\{ u , u ^ { \prime } \}$ ,

$$
\Gamma _ { \{ u , u ^ { \prime } \} } = V _ { \mathrm { m y o p i c } } ( n _ { \{ u , u ^ { \prime } \} } \mid p _ { t - 1 } ) - V _ { \mathrm { m y o p i c } } ( n _ { \{ u \} } \mid p _ { t - 1 } ) - V _ { \mathrm { m y o p i c } } ( n _ { \{ u ^ { \prime } \} } \mid p _ { t - 1 } ) + V _ { \mathrm { m y o p i c } } ( 0 \mid p _ { t - 1 } )
$$

i<sub>s</sub> th<sub>e</sub> <sub>ne</sub>t <sub>va</sub>l<sub>ue</sub> <sub>o</sub>f <sub>o</sub>b<sub>serv</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>wo</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> t<sub>oge</sub>th<sub>er</sub> b<sub>eyon</sub>d th<sub>e</sub> <sub>sum</sub> <sub>o</sub>f th<sub>e</sub>i<sub>r</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>e</sub>f<sub>ec</sub>t<sub>s.</sub> A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> B<sub>oo</sub>l<sub>ean</sub> M<sub>o</sub>bi<sub>us</sub> i<sub>nvers</sub>i<sub>on</sub> t<sub>o</sub> th<sub>e se</sub>t f<sub>unc</sub>ti<sub>on</sub>¨ $G \mapsto V _ { \mathrm { m y o p i c } } ( n _ { G } \mid p _ { t - 1 } )$ <sub>g</sub>i<sub>ves</sub> L<sub>emma</sub> A<sub>.</sub>1<sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>says</sub> <sub>we</sub> <sub>can</sub> <sub>wr</sub>it<sub>e</sub> $V _ { \mathrm { m y o p i c } } ( n | p _ { t - 1 } )$ <sub>as</sub> th<sub>e sum o</sub>f th<sub>e</sub> $\Gamma _ { H }$ terms.

LEMMA A.1 (Boolean Mobius inversion)¨ . For a batch with count matrix � and positions $\{ 1 , \ldots , U \}$

$$
V _ { m y o p i c } ( n | p _ { t - 1 } ) = \sum _ { H \subseteq \{ 1 , . . . , U \} } \Gamma _ { H } .
$$

W<sub>e nex</sub>t d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> t<sub>runca</sub>t<sub>e</sub>d <sub>approx</sub>i<sub>ma</sub>ti<sub>on.</sub> F<sub>or any</sub> i<sub>n</sub>t<sub>eger</sub> $w \ge 1$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e or</sub>d<sub>er-�</sub> t<sub>runca</sub>ti<sub>on</sub> $\widehat { V } _ { \mathrm { m y o p i c } , w } ( n |$ $\begin{array} { r } { p _ { t - 1 } ) : = \sum _ { H \subseteq \{ 1 , \dots , U \} , | H | \leq w } \Gamma _ { H } } \end{array}$

APPROXIMATION A.1. We approximate $V _ { m y o p i c } ( n \mid p _ { t - 1 } ) \approx \widehat { V } _ { m y o p i c , w } ( n \mid p _ { t - 1 } )$ with error

$$
V _ { m y o p i c } ( n | p _ { t - 1 } ) - \widehat { V } _ { m y o p i c , w } ( n | p _ { t - 1 } ) = \sum _ { H \subseteq \{ 1 , \ldots , U \} , | H | \geq w + 1 } \Gamma _ { H } .
$$

If th<sub>ere are no</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> t<sub>erms o</sub>f hi<sub>g</sub>h<sub>er or</sub>d<sub>er</sub> th<sub>an �,</sub> i<sub>.e.,</sub> if th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h <sub>s</sub>i<sub>ze</sub> i<sub>s a</sub>t <sub>mos</sub>t <sub>�,</sub> th<sub>en</sub> th<sub>e</sub> <sub>approx</sub>i<sub>ma</sub>ti<sub>on</sub> i<sub>s</sub> <sub>exac</sub>t<sub>.</sub> Thi<sub>s</sub> t<sub>runca</sub>ti<sub>on</sub> <sub>ma</sub>k<sub>es</sub> th<sub>e</sub> <sub>va</sub>l<sub>ue</sub> f<sub>unc</sub>ti<sub>on</sub> t<sub>rac</sub>t<sub>a</sub>bl<sub>e,</sub> <sub>w</sub>hil<sub>e</sub> <sub>s</sub>till <sub>a</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>us</sub> t<sub>o</sub> <sub>cap</sub>t<sub>ure</sub> <sub>some o</sub>f th<sub>e</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>on e</sub>f<sub>ec</sub>t<sub>s</sub> b<sub>e</sub>t<sub>ween eva</sub>l<sub>ua</sub>ti<sub>ons</sub> i<sub>n a</sub> b<sub>a</sub>t<sub>c</sub>h<sub>.</sub> E<sub>very re</sub>t<sub>a</sub>i<sub>ne</sub>d i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> t<sub>erm</sub> i<sub>nvo</sub>l<sub>ves</sub> <sup>com</sup>pu<sup>tin</sup>g $V _ { \mathrm { m y o p i c } }$ <sub>on su</sub>b<sub>-</sub>b<sub>a</sub>t<sub>c</sub>h<sub>es o</sub>f <sub>s</sub>i<sub>ze a</sub>t <sub>mos</sub>t <sub>�, w</sub>hi<sub>c</sub>h <sub>requ</sub>i<sub>res sums over a</sub>t <sub>mos</sub>t $2 ^ { w }$ outcome vectors <sub>w</sub>ith <sub>mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> <sub>norma</sub>l CDF<sub>s</sub> <sub>o</sub>f di<sub>mens</sub>i<sub>on</sub> <sub>a</sub>t <sub>mos</sub>t <sub>�,</sub> <sub>regar</sub>dl<sub>ess</sub> <sub>o</sub>f th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h <sub>s</sub>i<sub>ze</sub> $U .$

A<sub>.</sub>2<sub>.</sub>3<sub>.</sub> B<sub>a</sub>t<sub>c</sub>h <sub>co</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n U<sub>s</sub>i<sub>ng</sub> A<sub>pprox</sub>i<sub>ma</sub>ti<sub>on</sub> A<sub>.</sub>1 f<sub>or some</sub> fi<sub>xe</sub>d <sub>�, we</sub> b<sub>u</sub>ild b<sub>a</sub>t<sub>c</sub>h<sub>es</sub> i<sub>n a s</sub>i<sub>m</sub>il<sub>ar</sub> f<sub>as</sub>hi<sub>on</sub> t<sub>o</sub> Al<sub>gor</sub>ith<sub>m</sub> 4<sub>.</sub> C<sub>ons</sub>id<sub>er a par</sub>ti<sub>a</sub>ll<sub>y comm</sub>itt<sub>e</sub>d b<sub>a</sub>t<sub>c</sub>h $( i _ { 1 } , j _ { 1 } ) , \dotsc , ( i _ { U } , j _ { U } )$ <sub>w</sub>ith <sub>coun</sub>t <sub>ma</sub>t<sub>r</sub>i<sub>x �.</sub> F<sub>or a</sub> (feasible) candidate evaluation $( i , j ) \in \mathcal { I }$ <sub>,</sub> <sub>appen</sub>d<sub>e</sub>d t<sub>o</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h <sub>as</sub> $( i _ { U + 1 } , j _ { U + 1 } )$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub> it<sub>s</sub> <sub>marg</sub>i<sub>na</sub>l <sub>ga</sub>i<sub>n</sub> b<sub>y</sub>

$$
\Delta _ { n } ^ { ( w ) } ( ( i , j ) ) : = \widehat { V } _ { \mathrm { m y o p i c } , w } ( n + F ^ { i j } \mid p _ { t - 1 } ) - \widehat { V } _ { \mathrm { m y o p i c } , w } ( n \mid p _ { t - 1 } ) = \sum _ { H \subseteq \{ 1 , \ldots , U \} , \mid H \mid \leq w - 1 } \Gamma _ { H \cup \{ U + 1 \} } .\tag{A.5}
$$

I<sub>n</sub> <sub>wor</sub>d<sub>s,</sub> th<sub>e</sub> <sub>marg</sub>i<sub>na</sub>l <sub>ga</sub>i<sub>n</sub> <sub>o</sub>f $( i , j )$ <sub>un</sub>d<sub>er</sub> A<sub>pprox</sub>i<sub>ma</sub>ti<sub>on</sub> A<sub>.</sub>1 i<sub>s</sub> th<sub>e</sub> <sub>sum,</sub> <sub>over</sub> <sub>a</sub>ll <sub>su</sub>b<sub>se</sub>t<sub>s</sub> <sub>o</sub>f <sub>a</sub>t <sub>mos</sub>t $w - 1$ <sub>a</sub>l<sub>rea</sub>d<sub>y comm</sub>itt<sub>e</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>ons, o</sub>f th<sub>e</sub> i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> t<sub>erm o</sub>f th<sub>a</sub>t <sub>su</sub>b<sub>se</sub>t t<sub>oge</sub>th<sub>er w</sub>ith th<sub>e new can</sub>did<sub>a</sub>t<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on.</sub>

St<sub>ar</sub>ti<sub>ng</sub> f<sub>rom</sub> th<sub>e emp</sub>t<sub>y</sub> b<sub>a</sub>t<sub>c</sub>h<sub>, we repea</sub>t<sub>e</sub>dl<sub>y a</sub>dd th<sub>e</sub> f<sub>eas</sub>ibl<sub>e can</sub>did<sub>a</sub>t<sub>e w</sub>ith th<sub>e</sub> l<sub>arges</sub>t <sub>per-un</sub>it<sub>-resource</sub> mar<sub>g</sub>inal <sub>g</sub>ain. As discussed in §2.3<sub>,</sub> all mar<sub>g</sub>inal <sub>g</sub>ains ma<sub>y</sub> be zero<sub>,</sub> in which case we fall back to the evaluation that Al<sub>g</sub>orithm 4 would choose. We use the elastic round bud<sub>g</sub>ets of §6.1. The <sub>p</sub>rocedure terminates <sub>w</sub>h<sub>en</sub> th<sub>e roun</sub>d b<sub>u</sub>d<sub>ge</sub>t i<sub>s spen</sub>t <sub>or no eva</sub>l<sub>ua</sub>ti<sub>on</sub> i<sub>s a</sub>f<sub>or</sub>d<sub>a</sub>bl<sub>e, an</sub>d th<sub>e ou</sub>t<sub>pu</sub>t i<sub>s</sub> th<sub>e</sub> b<sub>a</sub>t<sub>c</sub>h’<sub>s coun</sub>t <sub>ma</sub>t<sub>r</sub>i<sub>x �.</sub> Th<sub>e en</sub>d<sub>game</sub> b<sub>a</sub>t<sub>c</sub>h<sub>-</sub>b<sub>u</sub>ildi<sub>ng proce</sub>d<sub>ure, w</sub>hi<sub>c</sub>h <sub>we ca</sub>ll AKG<sub>w,</sub> i<sub>s summar</sub>i<sub>ze</sub>d i<sub>n</sub> Al<sub>gor</sub>ith<sub>m</sub> A<sub>.</sub>1<sub>.</sub>

Algorithm A.1 A<sub>pp</sub>roximate Knowled<sub>g</sub>e Gradient with order-� truncation (AKGw)   
In<sub>p</sub>ut: $\mu _ { t - 1 } , \Sigma _ { t - 1 }$ <sub>, roun</sub>d b<sub>u</sub>d<sub>ge</sub>t $B _ { t } ,$ <sub>rema</sub>i<sub>n</sub>i<sub>ng</sub> t<sub>o</sub>t<sub>a</sub>l b<sub>u</sub>d<sub>ge</sub>t $R _ { \mathrm { r e m } }$ (initiall<sub>y</sub> $R ) ,$ excess from the <sub>p</sub>revious round (initiall<sub>y</sub> zero), and �   
1<sub>:</sub> $B _ { \mathrm { r e m } }  B _ { t } - \mathrm { e x c e s s }$   
2<sub>:</sub> $n  0 , \quad U  0 , \quad \Sigma _ { t } ^ { \prime } ( 0 )  \Sigma _ { t - 1 }$   
3<sub>:</sub> C<sub>o</sub>m<sub>pu</sub>t<sub>e</sub> $\left( q _ { t } , b _ { t } \right)$ <sub>v</sub>i<sub>a</sub> C<sub>oro</sub>ll<sub>ary</sub> 1   
4<sub>: w</sub>hil<sub>e</sub> $B _ { \mathrm { r e m } } > 0$ and $R _ { \mathrm { r e m } } \geq \operatorname* { m i n } ( r _ { 0 } , r _ { 1 } )$   
$\underline { { \Delta _ { n } ^ { ( w ) } } } ( ( i , j ) )$   
5<sub>:</sub> (�, �) ← ar<sub>g</sub> max ⊲ (A.5)   
$\begin{array} { r l } { ( i , j ) { \in } \bar { \mathcal { I } } { : } r _ { J } { \le } R _ { \mathrm { r e m } } \qquad } & { { } r _ { J } } \end{array}$   
6<sub>:</sub> if $\Delta _ { n } ^ { ( w ) } ( ( i , j ) ) = 0$   
7<sub>:</sub> $( i , j ) \gets \arg \operatorname* { m a x } _ { ( i , j ) \in { \mathcal { I } } : r _ { J } \leq R _ { \mathrm { r e m } } } \psi _ { t , i j } ( n )$ ⊲ (13)   
8<sub>:</sub> U<sub>p</sub>d<sub>a</sub>t<sub>e</sub> th<sub>e covar</sub>i<sub>ance</sub> f<sub>rom</sub> $\Sigma _ { t } ^ { \prime } ( n )$ to $\Sigma _ { t } ^ { \prime } ( n + F ^ { i j } )$ b<sub>y</sub> (12)   
9: $( \dot { i } _ { U + 1 } , j _ { U + 1 } ) \gets ( i , j ) , ~ U \gets \dot { U } + 1 , ~ \dot { n } \gets n + F ^ { i j }$   
10<sub>:</sub> $B _ { \mathrm { r e m } }  B _ { \mathrm { r e m } } - r _ { J } , \quad R _ { \mathrm { r e m } }  R _ { \mathrm { r e m } } - r _ { J }$   
11: excess ← max $\{ 0 , - B _ { \mathrm { r e m } } \}$   
12<sub>: re</sub>t<sub>urn</sub> $n ( t ) = n , R _ { \mathrm { r e m } } ,$ <sub>an</sub>d <sub>excess</sub>

Th<sub>e c</sub>h<sub>o</sub>i<sub>ce o</sub>f <sub>�</sub> d<sub>epen</sub>d<sub>s on</sub> h<sub>ow</sub> l<sub>arge</sub> th<sub>e pro</sub>bl<sub>em</sub> ${ \mathrm { i s } } ,$ <sub>an</sub>d <sub>on</sub> th<sub>e</sub> ti<sub>me an</sub>d <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>resources</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e.</sub> With <sub>a curren</sub>t b<sub>a</sub>t<sub>c</sub>h <sub>o</sub>f <sub>s</sub>i<sub>ze</sub> $U \geq w - 1$ <sub>,</sub> th<sub>e</sub> <sub>compu</sub>t<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f $( \mathsf { A } . 5 )$ f<sub>or</sub> <sub>a</sub>ll <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> i<sub>s</sub> d<sub>om</sub>i<sub>na</sub>t<sub>e</sub>d b<sub>y</sub> <sub>compu</sub>ti<sub>ng</sub> $| \mathcal { T } | \binom { U } { w - 1 } 2 ^ { w }$ <sub>�-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e norma</sub>l CDF<sub>s.</sub> Th<sub>e me</sub>th<sub>o</sub>d i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> tractable either when the batch size � is small (for exam<sub>p</sub>le, with a small $B _ { t } )$ <sub>or</sub> <sub>w</sub>h<sub>en</sub> <sub>�</sub> i<sub>s</sub> <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> <sub>sma</sub>ll (in our simulations, we use $w = 3 )$

## A<sub>.</sub>2<sub>.</sub>4<sub>.</sub> P<sub>roo</sub>f <sub>o</sub>f P<sub>ropos</sub>iti<sub>on</sub> A<sub>.</sub>1

Proof. Fix a round �, a candidate batch $( i _ { 1 } , j _ { 1 } ) , \dotsc , ( i _ { U } , j _ { U } )$ <sub>,</sub> <sub>w</sub>ith <sub>coun</sub>t <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $n ,$ <sub>an</sub>d <sub>an</sub> <sub>ou</sub>t<sub>come</sub> <sub>vec</sub>t<sub>or</sub> $\pmb { x } = ( x _ { 1 } , \hdots , x _ { U } ) ^ { \top } \in \{ - 1 , 1 \} ^ { U }$ <sub>.</sub> R<sub>eca</sub>ll th<sub>a</sub>t $\scriptstyle J _ { u } : = \operatorname* { m i n } ( 1 , j _ { u } ) , \pmb { e } _ { 0 } : = \mathbf { 0 }$ <sub>.</sub> F<sub>or</sub> <sub>eac</sub>h $u = 1 , \ldots , U ,$ <sub>,</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t G<sub>auss</sub>i<sub>an</sub> <sub>var</sub>i<sub>a</sub>bl<sub>e</sub>

$$
Z _ { u } = \beta _ { J _ { u } } \left[ ( e _ { i _ { u } } - e _ { j _ { u } } ) ^ { \top } \pmb \theta - \theta _ { 0 } \mathbb { 1 } \{ j _ { u } = 0 \} \right] + \varepsilon _ { u } ,
$$

<sub>w</sub>h<sub>ere</sub> $\pmb { \varepsilon } = ( \varepsilon _ { 1 } , \dots , \varepsilon _ { U } ) ^ { \top } \sim N ( \mathbf { 0 } , I _ { U } )$ is independent of �. Then

$$
X _ { u } = 1 \quad \Longleftrightarrow \quad Z _ { u } \geq 0 , \qquad X _ { u } = - 1 \quad \Longleftrightarrow \quad Z _ { u } < 0 .
$$

U<sub>n</sub>d<sub>er</sub> th<sub>e</sub> <sub>pr</sub>i<sub>or</sub> $p _ { t - 1 } ( \cdot ) = N ( \pmb { \mu _ { t - 1 } } , \pmb { \Sigma _ { t - 1 } } )$ <sub>,</sub> th<sub>e</sub> <sub>vec</sub>t<sub>or</sub> $\pmb { Z } = ( Z _ { 1 } , \ldots , Z _ { U } ) ^ { \top }$ i<sub>s mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e norma</sub>l<sub>.</sub> It<sub>s mean</sub> <sub>vec</sub>t<sub>or</sub> i<sub>s</sub> $\pmb { m } = ( m _ { 1 } , . . . , m _ { U } ) ^ { \top }$ <sub>, w</sub>ith

$$
m _ { u } = \mathbb { E } [ Z _ { u } ] = \beta _ { J u } ( \mu _ { t - 1 , i _ { u } } - \mu _ { t - 1 , j _ { u } } ) .
$$

It<sub>s covar</sub>i<sub>ance ma</sub>t<sub>r</sub>i<sub>x</sub> i<sub>s</sub> $C \in \mathbb { R } ^ { U \times U }$ <sub>w</sub>ith

$$
C _ { u u ^ { \prime } } = \mathrm { C o v } ( Z _ { u } , Z _ { u ^ { \prime } } ) = \beta _ { J u } \beta _ { J _ { u ^ { \prime } } } ( e _ { i _ { u } } - e _ { j _ { u } } ) ^ { \top } \Sigma _ { t - 1 } ( e _ { i _ { u ^ { \prime } } } - e _ { j _ { u ^ { \prime } } } ) + \mathbb { 1 } \{ u = u ^ { \prime } \} .
$$

D<sub>e</sub>fi<sub>ne</sub>

$$
D = \operatorname { d i a g } \left( \sqrt { C _ { 1 1 } } , \ldots , \sqrt { C _ { U U } } \right) , \qquad D _ { x } = \operatorname { d i a g } ( x _ { 1 } , \ldots , x _ { U } ) , \qquad \widetilde { C } _ { x } = D _ { x } D ^ { - 1 } C D ^ { - 1 } D _ { x } .
$$

Th<sub>e ma</sub>t<sub>r</sub>i<sub>x</sub> $D ^ { - 1 } C D ^ { - 1 }$ is the correlation matrix of �, and $\widetilde { C } _ { x }$ i<sub>s</sub> th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>s</sub>i<sub>gn-</sub>fli<sub>ppe</sub>d <sub>corre</sub>l<sub>a</sub>ti<sub>on</sub> matrix. This is exactl<sub>y</sub> the construction of Mullah<sub>y</sub> (2017, A<sub>pp</sub>endix B). In Mullah<sub>y</sub>’s notation, for a binar<sub>y</sub> <sub>ou</sub>t<sub>come</sub> h<sub>e</sub> d<sub>e</sub>fi<sub>nes</sub> <sub>s</sub>i<sub>gns</sub> $s _ { j p } = 2 k _ { j p } - 1$ <sub>,</sub> f<sub>orms</sub> th<sub>e</sub> di<sub>agona</sub>l <sub>s</sub>i<sub>gn</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $T _ { p } = \mathrm { d i a g } ( s _ { j p } )$ <sub>,</sub> <sub>an</sub>d <sub>rep</sub>l<sub>aces</sub> th<sub>e</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>corre</sub>l<sub>a</sub>ti<sub>on</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $R$ <sup>b</sup><sub>y</sub> $Q _ { p } = T _ { p } R T _ { p }$ <sub>.</sub> H<sub>ere,</sub> $D _ { x }$ <sub>p</sub>l<sub>ays</sub> th<sub>e</sub> <sub>ro</sub>l<sub>e</sub> <sub>o</sub>f $T _ { p }$ <sub>,</sub> <sub>an</sub>d $\widetilde { C } _ { x }$ <sub>p</sub>l<sub>ays</sub> th<sub>e</sub> <sub>ro</sub>l<sub>e</sub> <sub>o</sub>f $Q _ { p }$ N<sub>ow</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\pmb { \alpha } ( \pmb { x } ) = ( \alpha _ { 1 } ( \pmb { x } ) , \dots , \alpha _ { U } ( \pmb { x } ) ) ^ { \top }
$$

$$
\mathrm { w h e r e } \qquad \alpha _ { u } ( x ) = \frac { x _ { u } m _ { u } } { \sqrt { C _ { u u } } } = \frac { x _ { u } \beta _ { J u } \left( \mu _ { t - 1 , i _ { u } } - \mu _ { t - 1 , j _ { u } } \right) } { \sqrt { C _ { u u } } } , \quad u = 1 , \ldots , U .
$$

L<sub>e</sub>t ${ \pmb W } = - D _ { x } D ^ { - 1 } ( { \pmb Z } - { \pmb m } ) \mathrm { s o } { \pmb W } \sim N ( { \pmb 0 } , \widetilde { C } _ { x } )$ <sub>.</sub> F<sub>or eac</sub>h <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> $u ,$

$$
W _ { u } \leq \alpha _ { u } ( x ) \quad \Longleftrightarrow \quad - \frac { x _ { u } ( Z _ { u } - m _ { u } ) } { \sqrt { C _ { u u } } } \leq \frac { x _ { u } m _ { u } } { \sqrt { C _ { u u } } } \Longleftrightarrow \quad x _ { u } Z _ { u } \geq 0 .
$$

Th<sub>ere</sub>f<sub>ore</sub> $\{ X = x \} = \{ W \leq \alpha ( x ) \}$ <sub>a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y.</sub> H<sub>ence</sub>

$$
\mathbb { P } ( X = x \mid n , p _ { t - 1 } ) = \mathbb { P } \left( W \leq \alpha ( { \pmb x } ) \right) = \Phi _ { U } \left( { \pmb \alpha } ( { \pmb x } ) ; \widetilde { C } _ { x } \right) .
$$

Thi<sub>s proves</sub> th<sub>e ou</sub>t<sub>come-pro</sub>b<sub>a</sub>bilit<sub>y</sub> f<sub>ormu</sub>l<sub>a.</sub>

W<sub>e nex</sub>t d<sub>er</sub>i<sub>ve</sub> th<sub>e</sub> f<sub>ormu</sub>l<sub>a</sub> f<sub>or</sub> $\lambda _ { x }$ <sub>.</sub> Vi<sub>ew</sub>i<sub>ng</sub> th<sub>e ou</sub>t<sub>come pro</sub>b<sub>a</sub>bilit<sub>y a</sub>b<sub>ove as a</sub> f<sub>unc</sub>ti<sub>on o</sub>f th<sub>e</sub> l<sub>a</sub>t<sub>en</sub>t mean vector �, write $p _ { x } ( \pmb { m } ) : = \Phi _ { U } \Big ( \pmb { \alpha } ( \pmb { x } ) ; \widetilde { C } _ { x } \Big )$ <sub>.</sub> Si<sub>nce</sub> $C ,$ <sub>,</sub> <sub>an</sub>d h<sub>ence</sub> $D$ <sub>an</sub>d $\widetilde { C } _ { x }$ , does not depend on �, the d<sub>epen</sub>d<sub>ence</sub> <sub>on</sub> $m _ { u }$ i<sub>s</sub> <sub>on</sub>l<sub>y</sub> th<sub>roug</sub>h $\alpha _ { u } ( { \pmb x } ) = x _ { u } m _ { u } / \sqrt { C _ { u u } }$ <sub>.</sub> Th<sub>us,</sub>

$$
\frac { \partial p _ { { \pmb x } } ( { \pmb m } ) } { \partial m _ { { \boldsymbol u } } } = \frac { \partial } { \partial \alpha _ { \boldsymbol u } } \Phi _ { \boldsymbol U } \Big ( \alpha ( { \pmb x } ) ; \widetilde { C } _ { \boldsymbol x } \Big ) \frac { x _ { \boldsymbol u } } { \sqrt { C _ { u \boldsymbol u } } } .
$$

W<sub>e nex</sub>t <sub>compu</sub>t<sub>e</sub> thi<sub>s</sub> d<sub>er</sub>i<sub>va</sub>ti<sub>ve.</sub> F<sub>or</sub> ${ \pmb W } \sim N ( { \pmb 0 } , \widetilde { C } _ { \pmb { x } } )$

$$
\frac { \partial } { \partial \alpha _ { u } } \Phi _ { U } ( \alpha ; \widetilde { C } _ { x } ) = \phi ( \alpha _ { u } ) \mathbb { P } ( W _ { - u } \le \alpha _ { - u } | W _ { u } = \alpha _ { u } ) = \phi ( \alpha _ { u } ) \Phi _ { U - 1 } \left( \alpha _ { - u } - ( \widetilde { C } _ { x } ) _ { - u , u } \alpha _ { u } ; ( \widetilde { C } _ { x } ) _ { - u | u } \right) ,
$$

where the second equalit<sub>y</sub> follows from the conditional multivariate normal formula (Rao 1973, Eq. (8a.2.11), Mullah<sub>y</sub> 2017, A<sub>pp</sub>endix B). Substitutin<sub>g</sub> this into the derivative of $p _ { x } ( m )$ <sub>g</sub><sup>i</sup>ves

$$
\frac { \partial p _ { x } ( m ) } { \partial m _ { u } } = \frac { x _ { u } } { \sqrt { C _ { u u } } } \phi ( \alpha _ { u } ( x ) ) \Phi _ { U - 1 } \Big ( \alpha _ { - u } ( x ) - ( \widetilde { C } _ { x } ) _ { - u , u } \alpha _ { u } ( x ) ; ( \widetilde { C } _ { x } ) _ { - u | u } \Big ) \ .
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
\begin{array} { l } { \displaystyle \frac { \partial } { \partial m _ { u } } \log p _ { x } ( \boldsymbol { m } ) = \frac { 1 } { p _ { x } ( \boldsymbol { m } ) } \frac { \partial p _ { x } ( \boldsymbol { m } ) } { \partial m _ { u } } } \\ { \displaystyle \quad = \frac { x _ { u } } { \sqrt { C _ { u u } } } \frac { \phi \left( \alpha _ { u } ( \boldsymbol { x } ) \right) \Phi _ { U - 1 } \left( \alpha _ { - u } ( \boldsymbol { x } ) - ( \widetilde { C } _ { x } ) _ { - u , u } \alpha _ { u } ( \boldsymbol { x } ) ; ( \widetilde { C } _ { x } ) _ { - u | u } \right) } { \Phi _ { U } \left( \alpha ( \boldsymbol { x } ) ; \widetilde { C } _ { x } \right) } = ( \lambda _ { x } ) _ { u } . } \end{array}
$$

H<sub>ence</sub> $\lambda _ { x } = \nabla _ { m } \log p _ { x } ( m )$

It <sub>rema</sub>i<sub>ns</sub> t<sub>o</sub> <sub>connec</sub>t $\lambda _ { x }$ t<sub>o</sub> th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>mean</sub> <sub>o</sub>f $\pmb \theta .$ <sub>.</sub> Th<sub>e</sub> <sub>pa</sub>i<sub>r</sub> $( \pmb \theta , \pmb Z )$ is jointly Gaussian. For each $u = 1 , \ldots , U$

$$
\begin{array} { r } { \operatorname { C o v } ( \pmb \theta , Z _ { u } ) = \beta _ { J _ { u } } \Sigma _ { t - 1 } ( \pmb e _ { i _ { u } } - \pmb e _ { j _ { u } } ) . } \end{array}
$$

B<sub>y</sub> the conditional multivariate normal formula (Rao 1973, Eq. (8a.2.11)),

$$
\operatorname { \mathbb { E } } [ \pmb { \theta } | \pmb { Z } ] = \pmb { \mu _ { t - 1 } } + \operatorname { C o v } ( \pmb { \theta } , \pmb { Z } ) \ d { C } ^ { - 1 } ( \pmb { Z } - \pmb { m } ) .
$$

Since � is determined by �, the tower property gives

$$
{ \mathbb E } [ \theta \mid X = x , n , p _ { t - 1 } ] = \mu _ { t - 1 } + \operatorname { C o v } ( \theta , Z ) C ^ { - 1 } \left( { \mathbb E } [ Z \mid X = x , n , p _ { t - 1 } ] - m \right) .
$$

W<sub>e w</sub>ill <sub>s</sub>h<sub>ow</sub> $C ^ { - 1 } \left( \mathbb { E } [ Z | X = x , n , p _ { t - 1 } ] - m \right) = \lambda _ { x }$ <sub>.</sub> D<sub>e</sub>fi<sub>ne</sub> $A _ { x } : = \{ z \in \mathbb { R } ^ { U } : x _ { u } z _ { u } \geq 0 , u = 1 , \ldots , U \}$ <sub>.</sub> Si<sub>nce</sub> $\{ X = x \} = \{ Z \in A _ { x } \}$ <sub>a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y, we</sub> h<sub>ave</sub>

$$
p _ { x } ( { \pmb m } ) = \int _ { A _ { x } } \phi _ { C } ( z - { \pmb m } ) d z ,
$$

<sub>w</sub>h<sub>ere</sub> $\phi _ { C }$ is the �(0, �) densit<sub>y</sub>. Diferentiating gives

$$
\nabla _ { \pmb { m } } \phi _ { C } ( z - \pmb { m } ) = C ^ { - 1 } ( z - \pmb { m } ) \phi _ { C } ( z - \pmb { m } )
$$

which is locally dominated by some integrable function of �. Therefore, by the dominated convergence th<sub>eorem,</sub>

$$
\begin{array} { l } { \displaystyle \nabla _ { m } p _ { x } ( m ) = \int _ { A _ { x } } C ^ { - 1 } ( z - m ) \phi _ { C } ( z - m ) d z } \\ { \displaystyle = C ^ { - 1 } \left[ \int _ { A _ { x } } z \phi _ { C } ( z - m ) d z - m \int _ { A _ { x } } \phi _ { C } ( z - m ) d z \right] } \\ { \displaystyle = C ^ { - 1 } \left[ p _ { x } ( m ) \mathbb { E } [ \textbf { Z } | X = x , n , p _ { t - 1 } ] - p _ { x } ( m ) m \right] } \\ { \displaystyle = p _ { x } ( m ) C ^ { - 1 } \left( \mathbb { E } [ \textbf { Z } | X = x , n , p _ { t - 1 } ] - m \right) . } \end{array}
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
C ^ { - 1 } \left( \mathbb { E } [ Z \mid X = x , n , p _ { t - 1 } ] - m \right) = \nabla _ { m } \log p _ { x } ( m ) = \lambda _ { x } .
$$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> thi<sub>s</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> <sub>express</sub>i<sub>on</sub> <sub>a</sub>b<sub>ove</sub> <sub>g</sub>i<sub>ves</sub> $\mu _ { t } ( x \mid n , p _ { t - 1 } ) = \mu _ { t - 1 } + \operatorname { C o v } ( \pmb \theta , \pmb { Z } ) \lambda _ { x }$ <sub>,</sub> <sub>an</sub>d

$$
\mathrm { C o v } ( \theta , Z ) \lambda _ { x } = \sum _ { u = 1 } ^ { U } \beta _ { J u } ( \lambda _ { x } ) _ { u } \Sigma _ { t - 1 } ( e _ { i _ { u } } - e _ { j _ { u } } ) .
$$

H<sub>ence</sub>

$$
\mu _ { t } ( \pmb { x } \mid n , p _ { t - 1 } ) = \pmb { \mu _ { t - 1 } } + \sum _ { u = 1 } ^ { U } \beta _ { J _ { u } } ( \lambda _ { \pmb { x } } ) _ { u } \Sigma _ { t - 1 } ( \pmb { e _ { i _ { u } } } - \pmb { e _ { j _ { u } } } ) .
$$

Thi<sub>s a</sub>l<sub>so proves</sub> L<sub>emma</sub> 3 <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> th<sub>e spec</sub>i<sub>a</sub>l <sub>case</sub> $U = 1$ □

## A<sub>pp</sub>endix B: Su<sub>pp</sub>lements to §2

## B<sub>.</sub>1<sub>.</sub> D<sub>er</sub>i<sub>va</sub>ti<sub>ons o</sub>f <sub>ca</sub>lib<sub>ra</sub>ti<sub>on</sub> f<sub>ormu</sub>l<sub>as</sub>

Calibrating the pairwise accuracy factor $\beta _ { 1 }$ . We calibrate $\beta _ { 1 }$ f<sub>rom</sub> th<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> th<sub>a</sub>t <sub>a</sub> <sub>compar</sub>i<sub>son</sub> <sub>o</sub>f t<sub>wo</sub> <sub>ran</sub>d<sub>om</sub>l<sub>y</sub> <sub>c</sub>h<sub>osen</sub> it<sub>ems</sub> <sub>agrees</sub> <sub>w</sub>ith th<sub>e</sub>i<sub>r</sub> t<sub>rue</sub> <sub>or</sub>d<sub>er</sub>i<sub>ng.</sub> L<sub>e</sub>t

$P _ { P } : = \mathbb { P } |$ (a comparison of two randoml<sub>y</sub> chosen items agrees with the true ordering).

O<sub>u</sub>r <sub>goa</sub>l i<sub>s</sub> t<sub>o</sub> <sub>exp</sub>r<sub>ess</sub> $P _ { P }$ <sub>as</sub> <sub>a</sub> f<sub>unc</sub>ti<sub>on</sub> <sub>o</sub>f $\beta _ { 1 }$ <sub>.</sub> U<sub>n</sub>d<sub>er</sub> th<sub>e</sub> <sub>pr</sub>i<sub>or,</sub> th<sub>e</sub> <sub>u</sub>tilit<sub>y</sub> dif<sub>erence</sub> $\Delta : = \theta _ { i } - \theta _ { j }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\Delta \sim N ( 0 , 2 s )$ <sub>.</sub> W<sub>r</sub>iti<sub>ng</sub> $E \sim N ( 0 , 1 )$ i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> <sub>an</sub>d $Z : = \beta _ { 1 } \Delta + E$ <sub>,</sub> th<sub>e</sub> <sub>compar</sub>i<sub>son</sub> <sub>ou</sub>t<sub>come</sub> i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> th<sub>e</sub> <sub>s</sub>i<sub>gn</sub> <sub>o</sub>f $Z ,$ <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> t<sub>rue</sub> <sub>or</sub>d<sub>er</sub>i<sub>ng</sub> i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> th<sub>e</sub> <sub>s</sub>i<sub>gn</sub> <sub>o</sub>f $\Delta .$ <sub>.</sub> H<sub>ence</sub>

$$
P _ { P } = \mathbb { P } \big ( \mathrm { s i g n } ( Z ) = \mathrm { s i g n } ( \Delta ) \big ) .
$$

W<sub>e</sub> fi<sub>rs</sub>t <sub>compu</sub>t<sub>e</sub> th<sub>e</sub> <sub>corre</sub>l<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f $( \Delta , Z )$

$$
\Psi ( \Delta ) = 2 s , \qquad \Psi ( Z ) = \beta _ { 1 } ^ { 2 } \Psi ( \Delta ) + \Psi ( E ) = 1 + 2 s \beta _ { 1 } ^ { 2 } , \qquad \mathrm { C o v } ( \Delta , Z ) = \mathrm { C o v } ( \Delta , \beta _ { 1 } \Delta + E ) = 2 s \beta _ { 1 } ,
$$

th<sub>us</sub> th<sub>e</sub>i<sub>r corre</sub>l<sub>a</sub>ti<sub>on</sub> i<sub>s</sub>

$$
\rho _ { P } = \frac { \mathrm { C o v } ( \Delta , Z ) } { \sqrt { \mathbb { V } ( \Delta ) \mathbb { V } ( Z ) } } = \frac { \beta _ { 1 } \sqrt { 2 s } } { \sqrt { 1 + 2 s \beta _ { 1 } ^ { 2 } } } .
$$

F<sub>or a mean-zero</sub> bi<sub>var</sub>i<sub>a</sub>t<sub>e norma</sub>l <sub>pa</sub>i<sub>r w</sub>ith <sub>corre</sub>l<sub>a</sub>ti<sub>on</sub> $\rho _ { P }$ <sub>,</sub> th<sub>e pro</sub>b<sub>a</sub>bilit<sub>y</sub> th<sub>a</sub>t th<sub>e</sub> t<sub>wo var</sub>i<sub>a</sub>bl<sub>es</sub> h<sub>ave</sub> th<sub>e</sub> <sub>same</sub> <sub>s</sub>i<sub>gn</sub> i<sub>s</sub> $\textstyle { \frac { 1 } { 2 } } + { \frac { 1 } { \pi } }$ <sub>arcs</sub>i<sub>n</sub> $\displaystyle ( \rho _ { P } )$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore,</sub>

$$
P _ { P } = \frac { 1 } { 2 } + \frac { 1 } { \pi } \arcsin \left( \frac { \beta _ { 1 } \sqrt { 2 s } } { \sqrt { 1 + 2 s \beta _ { 1 } ^ { 2 } } } \right) ,
$$

<sub>w</sub>hi<sub>c</sub>h <sub>g</sub>i<sub>ves</sub> th<sub>e ca</sub>lib<sub>ra</sub>ti<sub>on equa</sub>ti<sub>on</sub> f<sub>or</sub> $\beta _ { 1 }$

Calibrating the rating accuracy factor $\beta _ { 0 }$ and the threshold $\theta _ { 0 }$ . We calibrate $\beta _ { 0 }$ <sub>an</sub>d $\theta _ { 0 }$ jointly from th<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> th<sub>a</sub>t <sub>a</sub> <sub>ra</sub>ti<sub>ng</sub> i<sub>s</sub> <sub>correc</sub>t <sub>an</sub>d th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d th<sub>um</sub>b<sub>s-up</sub> <sub>ra</sub>t<sub>e</sub> <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>pr</sub>i<sub>or.</sub> L<sub>e</sub>t

$$
\begin{array} { r } { P _ { R } : = \mathbb { P } ( \mathrm { a ~ r a t i n g ~ o f ~ a ~ r a n d o m l y ~ c h o s e n ~ i t e m ~ i s ~ c o r r e c t } ) , \qquad \overline { { P } } : = \mathbb { P } ( X _ { i 0 } = 1 ) . } \end{array}
$$

Step $a \mathrm { : }$ Calibrate the thumbs-up rate. Writing $E \sim N ( 0 , 1 )$ i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y,</sub> <sub>a</sub> th<sub>um</sub>b<sub>s-up</sub> <sub>occurs</sub> <sub>w</sub>h<sub>en</sub> $\beta _ { 0 } ( \theta _ { i } - \theta _ { 0 } ) + E > 0$ <sub>.</sub> Si<sub>nce</sub> $\theta _ { i } \sim N ( 0 , s )$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub> $\beta _ { 0 } \theta _ { i } + E \sim N ( 0 , 1 + s \beta _ { 0 } ^ { 2 } )$ <sub>, an</sub>d th<sub>ere</sub>f<sub>ore</sub>

$$
\overline { { P } } = \mathbb { P } ( \beta _ { 0 } \theta _ { i } + E > \beta _ { 0 } \theta _ { 0 } ) = \Phi \left( - \frac { \beta _ { 0 } \theta _ { 0 } } { \sqrt { 1 + s \beta _ { 0 } ^ { 2 } } } \right) .
$$

Step $b { \mathrm { : } }$ Calibrate the probability that a rating is correct. Let $\Delta _ { i } : = \theta _ { i } - \theta _ { 0 }$ <sub>an</sub>d $Z : = \beta _ { 0 } \Delta _ { i } + E$ <sub>.</sub> Th<sub>e</sub> t<sub>rue</sub> <sub>ra</sub>ti<sub>ng</sub> <sub>o</sub>f it<sub>em</sub> � i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> th<sub>e</sub> <sub>s</sub>i<sub>gn</sub> <sub>o</sub>f $\Delta _ { i }$ <sub>,</sub> <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub>d <sub>ra</sub>ti<sub>ng</sub> i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> th<sub>e</sub> <sub>s</sub>i<sub>gn</sub> <sub>o</sub>f $Z .$ <sub>.</sub> H<sub>ence</sub>

$$
P _ { R } = \mathbb { P } ( \Delta _ { i } > 0 , Z > 0 ) + \mathbb { P } ( \Delta _ { i } < 0 , Z < 0 ) .
$$

T<sub>o compu</sub>t<sub>e</sub> th<sub>ese pro</sub>b<sub>a</sub>biliti<sub>es, no</sub>t<sub>e</sub> th<sub>a</sub>t $( \Delta _ { i } , Z )$ i<sub>s</sub> bi<sub>var</sub>i<sub>a</sub>t<sub>e norma</sub>l <sub>w</sub>ith

$$
\Delta _ { i } \sim N ( - \theta _ { 0 } , s ) , \qquad Z \sim N ( - \beta _ { 0 } \theta _ { 0 } , 1 + s \beta _ { 0 } ^ { 2 } ) , \qquad \mathrm { C o v } ( \Delta _ { i } , Z ) = \beta _ { 0 } s ,
$$

<sub>so</sub> th<sub>e</sub>i<sub>r corre</sub>l<sub>a</sub>ti<sub>on</sub> i<sub>s</sub>

$$
\rho _ { 0 } = \frac { \beta _ { 0 } s } { \sqrt { s ( 1 + s \beta _ { 0 } ^ { 2 } ) } } = \frac { \beta _ { 0 } \sqrt { s } } { \sqrt { 1 + s \beta _ { 0 } ^ { 2 } } } .
$$

St<sub>an</sub>d<sub>ar</sub>di<sub>z</sub>i<sub>ng</sub> <sub>w</sub>ith

$$
U : = \frac { \Delta _ { i } + \theta _ { 0 } } { \sqrt { s } } , \qquad V : = \frac { Z + \beta _ { 0 } \theta _ { 0 } } { \sqrt { 1 + s \beta _ { 0 } ^ { 2 } } } ,
$$

gives a mean-zero bivariate normal pair (�,�) with correlation $\rho _ { 0 }$ <sub>, an</sub>d

$$
\Delta _ { i } > 0 \Longleftrightarrow U > { \frac { \theta _ { 0 } } { \sqrt { s } } } , \qquad Z > 0 \Longleftrightarrow V > { \frac { \beta _ { 0 } \theta _ { 0 } } { \sqrt { 1 + s \beta _ { 0 } ^ { 2 } } } } .
$$

W<sub>r</sub>iti<sub>ng</sub> $\Phi _ { 2 } ( \cdot , \cdot ; \rho )$ f<sub>or</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d bi<sub>var</sub>i<sub>a</sub>t<sub>e norma</sub>l CDF <sub>w</sub>ith <sub>corre</sub>l<sub>a</sub>ti<sub>on</sub> $\rho ,$ <sub>, we</sub> th<sub>ere</sub>f<sub>ore o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
\begin{array} { r } { P _ { R } = \mathbb { P } \left( U > \displaystyle \frac { \theta _ { 0 } } { \sqrt { s } } , V > \displaystyle \frac { \beta _ { 0 } \theta _ { 0 } } { \sqrt { 1 + s \beta _ { 0 } ^ { 2 } } } \right) + \mathbb { P } \left( U < \displaystyle \frac { \theta _ { 0 } } { \sqrt { s } } , V < \displaystyle \frac { \beta _ { 0 } \theta _ { 0 } } { \sqrt { 1 + s \beta _ { 0 } ^ { 2 } } } \right) } \\ { = 1 - \Phi \left( \displaystyle \frac { \theta _ { 0 } } { \sqrt { s } } \right) - \Phi \left( \displaystyle \frac { \beta _ { 0 } \theta _ { 0 } } { \sqrt { 1 + s \beta _ { 0 } ^ { 2 } } } \right) + 2 \Phi _ { 2 } \left( \displaystyle \frac { \theta _ { 0 } } { \sqrt { s } } , \displaystyle \frac { \beta _ { 0 } \theta _ { 0 } } { \sqrt { 1 + s \beta _ { 0 } ^ { 2 } } } ; \rho _ { 0 } \right) . } \end{array}
$$

T<sub>oge</sub>th<sub>er,</sub> th<sub>e</sub> <sub>equa</sub>ti<sub>ons</sub> f<sub>or</sub> $\overline { P }$ <sub>an</sub>d $P _ { R }$ <sub>g</sub>i<sub>ve</sub> th<sub>e</sub> <sub>ca</sub>lib<sub>ra</sub>ti<sub>on</sub> <sub>equa</sub>ti<sub>ons</sub> f<sub>or</sub> $( \beta _ { 0 } , \theta _ { 0 } )$

## A<sub>pp</sub>endix C: Proofs of results in §3

Proof of Lemma 1. First consider a fixed utility vector $\pmb { \vartheta } \in \mathbb { R } ^ { I }$ <sub>, an</sub>d l<sub>e</sub>t $S _ { k } ^ { * } ( \pmb { \vartheta } )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e se</sub>t <sub>o</sub>f i<sub>n</sub>di<sub>ces</sub> of the � largest elements of �. Define the opportunit<sub>y</sub> cost $\begin{array} { r } { O ( \pmb { \vartheta } ; S _ { k } ) : = \sum _ { i \in S _ { k } ^ { * } ( \pmb { \vartheta } ) } \vartheta _ { i } - \sum _ { i \in S _ { k } } \vartheta _ { i } } \end{array}$ <sub>,</sub> <sub>an</sub>d <sub>reca</sub>ll th<sub>e</sub> <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> $\mathcal { M } ( \pmb { \vartheta } ; S _ { k } , a )$ from (6) evaluated at $\pmb { \vartheta } .$ Si<sub>nce</sub> $| S _ { k } | = | S _ { k } ^ { * } ( \pmb { \vartheta } ) | = k$ <sub>,</sub> th<sub>e</sub> <sub>se</sub>t<sub>s</sub> $S _ { k } \mid S _ { k } ^ { * } ( \pmb \vartheta )$ <sub>an</sub>d $S _ { k } ^ { * } ( \pmb { \vartheta } ) \backslash S _ { k }$ h<sub>ave</sub> th<sub>e same car</sub>di<sub>na</sub>lit<sub>y.</sub> Th<sub>en</sub>

$$
\begin{array} { l } { { \displaystyle M ( \vartheta ; S _ { k } , a ) = \sum _ { i \in S _ { k } } \left( a ( \vartheta ) - \vartheta _ { i } \right) ^ { + } + \sum _ { i \notin S _ { k } } \left( \vartheta _ { i } - a ( \vartheta ) \right) ^ { + } } } \\ { { \displaystyle \geq \sum _ { i \in S _ { k } \backslash S _ { k } ^ { * } ( \vartheta ) } \left( a ( \vartheta ) - \vartheta _ { i } \right) ^ { + } + \sum _ { i \in S _ { k } ^ { * } ( \vartheta ) \backslash S _ { k } } \left( \vartheta _ { i } - a ( \vartheta ) \right) ^ { + } } } \\ { { \displaystyle \geq \sum _ { i \in S _ { k } \backslash S _ { k } ^ { * } ( \vartheta ) } \left( a ( \vartheta ) - \vartheta _ { i } \right) + \sum _ { i \in S _ { k } ^ { * } ( \vartheta ) \backslash S _ { k } } \left( \vartheta _ { i } - a ( \vartheta ) \right) } } \\ { { \displaystyle = \sum _ { i \in S _ { k } \backslash ( \vartheta ) \backslash S _ { k } } \vartheta _ { i } - \sum _ { i \in S _ { k } \backslash S _ { k } ^ { * } ( \vartheta ) } \vartheta _ { i } = O ( \vartheta ; S _ { k } ) } . }  \end{array}
$$

Th<sub>e</sub> fi<sub>rs</sub>t i<sub>nequa</sub>lit<sub>y</sub> d<sub>rops</sub> th<sub>e nonnega</sub>ti<sub>ve</sub> t<sub>erms</sub> i<sub>n</sub>d<sub>exe</sub>d b<sub>y</sub> $i \in S _ { k } \cap S _ { k } ^ { * } ( \pmb { \vartheta } )$ <sub>an</sub>d $i \notin S _ { k } \cup S _ { k } ^ { * } ( \pmb { \vartheta } )$ <sub>,</sub> th<sub>e secon</sub>d uses $x ^ { + } \geq x ,$ <sub>, an</sub>d th<sub>e</sub> thi<sub>r</sub>d li<sub>ne</sub>’<sub>s �</sub> t<sub>erms cance</sub>l b<sub>ecause</sub> th<sub>e</sub> t<sub>wo</sub> i<sub>n</sub>d<sub>ex se</sub>t<sub>s</sub> h<sub>ave equa</sub>l <sub>car</sub>di<sub>na</sub>lit<sub>y.</sub> Thi<sub>s</sub> i<sub>s</sub> th<sub>e upper</sub> b<sub>oun</sub>d<sub>.</sub> Wh<sub>en</sub> $a ( \pmb \vartheta ) \in [ \vartheta ^ { ( k + 1 ) } , \vartheta ^ { ( k ) } ]$ <sub>we</sub> h<sub>ave</sub> $\vartheta _ { i } \geq a ( \pmb { \vartheta } )$ f<sub>or</sub> $i \in S _ { k } ^ { * } ( \pmb { \vartheta } )$ <sub>an</sub>d $\vartheta _ { i } \leq a ( \pmb { \vartheta } )$ f<sub>or</sub> $i \notin S _ { k } ^ { * } ( \vartheta )$ <sub>so</sub> th<sub>e</sub> d<sub>roppe</sub>d t<sub>erms</sub> i<sub>n</sub> th<sub>e</sub> fi<sub>rs</sub>t i<sub>nequa</sub>lit<sub>y</sub> <sub>are</sub> <sub>zero,</sub> <sub>an</sub>d <sub>eac</sub>h $x ^ { + } = x$ i<sub>n</sub> th<sub>e</sub> <sub>secon</sub>d i<sub>nequa</sub>lit<sub>y.</sub> H<sub>ence</sub> b<sub>o</sub>th i<sub>nequa</sub>liti<sub>es</sub> <sub>are</sub> th<sub>en</sub> <sub>equa</sub>liti<sub>es,</sub> <sub>g</sub>i<sub>v</sub>i<sub>ng</sub> $O ( \pmb { \vartheta } ; S _ { k } ) = M ( \pmb { \vartheta } ; S _ { k } , a )$

A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> thi<sub>s</sub> <sub>po</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se</sub> <sub>w</sub>ith $\pmb { \vartheta } = \pmb { \theta }$ <sub>an</sub>d t<sub>a</sub>ki<sub>ng</sub> <sub>expec</sub>t<sub>a</sub>ti<sub>ons</sub> <sub>over</sub> $\pmb { \theta } \mid p _ { t - 1 } \sim N ( \pmb { \mu _ { t - 1 } } , \pmb { \Sigma _ { t - 1 } } )$ <sub>g</sub><sup>i</sup>ves

$$
\mathrm { E O C } ( S _ { k } ; p _ { t - 1 } ) = \mathbb { E } \left[ O ( \pmb { \theta } ; S _ { k } ) \Big | p _ { t - 1 } \right] \leq \mathbb { E } \left[ M ( \pmb { \theta } ; S _ { k } , a ) \Big | p _ { t - 1 } \right] ,
$$

<sub>w</sub>ith <sub>equa</sub>lit<sub>y</sub> <sub>w</sub>h<sub>enever</sub> $a ( \pmb \theta ) \in [ \theta ^ { ( k + 1 ) } , \theta ^ { ( k ) } ]$ <sub>a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y.</sub> □

Proof of Theorem 1. $\begin{array} { r } { \mathcal { M } ( \theta ; S _ { k } , a _ { q , b } ) = k a _ { q , b } ( \theta ) - \sum _ { i \in S _ { k } } \theta _ { i } + \sum _ { i = 1 } ^ { I } ( \theta _ { i } - a _ { q , b } ( \theta ) ) ^ { + } } \end{array}$ <sub>,</sub> <sub>so</sub> <sub>w</sub>ith $\mathbb { E } [ a _ { q , b } ( \pmb { \theta } ) \ |$ $p _ { t - 1 } ] = q$

$$
\mathbb { E } \big [ M ( \theta ; S _ { k } , a _ { q , b } ) \bigm | p _ { t - 1 } \big ] = k q - \sum _ { i \in S _ { k } } \mu _ { t - 1 , i } + \sum _ { i = 1 } ^ { I } \mathbb { E } \big [ ( \theta _ { i } - a _ { q , b } ( \theta ) ) ^ { + } \bigm | p _ { t - 1 } \big ] .
$$

F<sub>or eac</sub>h fi<sub>xe</sub>d $\pmb { \theta } , ( \theta _ { i } - a _ { q , b } ( \pmb { \theta } ) ) ^ { + }$ i<sub>s</sub> th<sub>e</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>par</sub>t <sub>o</sub>f <sub>an</sub> <sub>a</sub>fi<sub>ne</sub> f<sub>unc</sub>ti<sub>on</sub> <sub>o</sub>f $( q , b )$ <sub>,</sub> h<sub>ence</sub> <sub>convex.</sub> S<sub>umm</sub>i<sub>ng,</sub> t<sub>a</sub>ki<sub>ng expec</sub>t<sub>a</sub>ti<sub>ons, an</sub>d <sub>a</sub>ddi<sub>ng</sub> th<sub>e</sub> li<sub>near</sub> t<sub>erm</sub> $k q$ <sub>preserves convex</sub>it<sub>y, so</sub> th<sub>e expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> i<sub>s</sub> <sub>convex</sub> i<sub>n</sub> $( q , b )$

W<sub>e</sub> <sub>nex</sub>t <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> <sub>m</sub>i<sub>n</sub>i<sub>mum</sub> i<sub>s</sub> <sub>a</sub>tt<sub>a</sub>i<sub>ne</sub>d<sub>.</sub> U<sub>s</sub>i<sub>ng</sub> $( x - y ) ^ { + } \geq x ^ { + } - | y |$ i<sub>n</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss,</sub>

$$
\mathcal { M } ( \theta ; S _ { k } , a _ { q , b } ) \geq k a _ { q , b } ( \theta ) ^ { + } + \big ( I - k \big ) \big ( - a _ { q , b } ( \theta ) \big ) ^ { + } - \sum _ { i = 1 } ^ { I } \vert \theta _ { i } \vert \geq \operatorname* { m i n } \{ k , I - k \} \big \vert a _ { q , b } ( \theta ) \big \vert - \sum _ { i = 1 } ^ { I } \vert \theta _ { i } \vert .
$$

M<sub>oreover, s</sub>i<sub>nce</sub> $a _ { q , b } ( \pmb \theta ) = q + b ^ { \top } \big ( \pmb \theta - \pmb \mu _ { t - 1 } \big )$

$$
\mathbb { E } \Big [ \Big \vert a _ { q , b } ( \theta ) \Big \vert \Big \vert p _ { t - 1 } \Big ] \geq \vert q \vert , \qquad 2 \mathbb { E } \Big [ \Big \vert a _ { q , b } ( \theta ) \Big \vert \Big \vert p _ { t - 1 } \Big ] \geq \sqrt { \frac { 2 } { \pi } } \sqrt { b ^ { \top } \Sigma _ { t - 1 } b } .
$$

Si<sub>nce</sub> $\Sigma _ { t - 1 }$ i<sub>s</sub> <sub>pos</sub>iti<sub>ve</sub> d<sub>e</sub>fi<sub>n</sub>it<sub>e,</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> th<sub>ere</sub>f<sub>ore</sub> t<sub>en</sub>d<sub>s</sub> t<sub>o</sub> i<sub>n</sub>fi<sub>n</sub>it<sub>y</sub> <sub>as</sub> $\| ( q , b ) \| _ { 2 } \to \infty$ <sub>.</sub> Si<sub>nce</sub> it i<sub>s</sub> fi<sub>n</sub>it<sub>e an</sub>d <sub>convex,</sub> h<sub>ence con</sub>ti<sub>nuous,</sub> it <sub>a</sub>tt<sub>a</sub>i<sub>ns a m</sub>i<sub>n</sub>i<sub>mum.</sub>

It <sub>rema</sub>i<sub>ns</sub> t<sub>o</sub> <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub> <sub>any</sub> <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zer</sub> <sub>no</sub>t <sub>o</sub>f th<sub>e</sub> f<sub>orm</sub> $( \mu _ { t - 1 , j } , e _ { j } )$ <sub>.</sub> F<sub>or</sub> <sub>suc</sub>h <sub>a</sub> <sub>pa</sub>i<sub>r,</sub> f<sub>or</sub> <sub>every</sub> �<sub>,</sub> <sub>e</sub>ith<sub>er</sub> $\pmb { b } \neq \pmb { e } _ { i }$ <sub>,</sub> i<sub>n w</sub>hi<sub>c</sub>h <sub>case</sub> $\theta _ { i } - a _ { q , b } ( \pmb \theta )$ i<sub>s a non</sub>d<sub>egenera</sub>t<sub>e</sub> G<sub>auss</sub>i<sub>an, or</sub> $\pmb { b } = \pmb { e } _ { i }$ <sub>an</sub>d $q \neq \mu _ { t - 1 , i } .$ i<sub>n w</sub>hi<sub>c</sub>h <sub>case</sub> $\theta _ { i } - a _ { q , b } ( \theta ) = \mu _ { t - 1 , i } - q \neq 0$ <sub>.</sub> Th<sub>us,</sub>

$$
\mathbb { P } \big ( \theta _ { i } = a _ { q , b } ( \pmb { \theta } ) \big | p _ { t - 1 } \big ) = 0 \qquad \mathrm { f o r ~ e v e r y ~ } i ,
$$

<sub>so</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> i<sub>s</sub> dif<sub>eren</sub>ti<sub>a</sub>bl<sub>e</sub> <sub>a</sub>t $( q , b )$ <sub>.</sub> Dif<sub>eren</sub>ti<sub>a</sub>ti<sub>ng</sub> <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>expec</sub>t<sub>a</sub>ti<sub>on</sub> <sub>g</sub>i<sub>ves</sub>

$$
\frac { \partial } { \partial q } ( \theta _ { i } - a _ { q , b } ( \theta ) ) ^ { + } = - 1 \{ \theta _ { i } > a _ { q , b } ( \theta ) \} , \qquad \frac { \partial } { \partial b } ( \theta _ { i } - a _ { q , b } ( \theta ) ) ^ { + } = - 1 \{ \theta _ { i } > a _ { q , b } ( \theta ) \} ( \theta - \mu _ { t - 1 } ) .
$$

S<sub>u</sub>mmin<sub>g</sub> <sub>ove</sub>r $i ,$ th<sub>e</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>ors</sub> <sub>a</sub>dd t<sub>o</sub> $\kappa _ { q , b } ( \pmb \theta ) + k$ , <sub>g</sub><sup>i</sup>v<sup>i</sup>n<sub>g</sub>

$$
\begin{array} { r l } & { \frac { \partial } { \partial q } \mathbb { E } \big [ M ( \theta ; S _ { k } , a _ { q , b } ) \mid p _ { t - 1 } \big ] = k - \mathbb { E } \Big [ \kappa _ { q , b } ( \theta ) + k \Big | p _ { t - 1 } \Big ] = - \mathbb { E } \Big [ \kappa _ { q , b } ( \theta ) \Big | p _ { t - 1 } \Big ] , } \\ & { \frac { \partial } { \partial b } \mathbb { E } \big [ M ( \theta ; S _ { k } , a _ { q , b } ) \mid p _ { t - 1 } \big ] = - \mathbb { E } \Big [ ( \theta - \mu _ { t - 1 } ) \big \{ \kappa _ { q , b } ( \theta ) + k \big \} \Big | p _ { t - 1 } \Big ] = - \mathbb { E } \Big [ ( \theta - \mu _ { t - 1 } ) \kappa _ { q , b } ( \theta ) \Big | p _ { t - 1 } \Big ] , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t <sub>equa</sub>lit<sub>y</sub> <sub>uses</sub> $\mathbb { E } [ \pmb \theta - \pmb \mu _ { t - 1 } \ | \ p _ { t - 1 } ] = 0$ <sub>.</sub> Th<sub>e</sub> ℓth <sub>componen</sub>t <sub>o</sub>f th<sub>e</sub> <sub>secon</sub>d d<sub>er</sub>i<sub>va</sub>ti<sub>ve</sub> i<sub>s</sub> $- \mathrm { C o v } ( \theta _ { \ell } , \kappa _ { q , b } ( \pmb \theta ) \vert p _ { t - 1 } )$ <sub>.</sub> Th<sub>us,</sub> f<sub>or</sub> <sub>any</sub> <sub>pa</sub>i<sub>r</sub> <sub>no</sub>t <sub>o</sub>f th<sub>e</sub> f<sub>orm</sub> $( \mu _ { t - 1 , j } , e _ { j } )$ <sub>,</sub> <sub>se</sub>tti<sub>ng</sub> b<sub>o</sub>th d<sub>er</sub>i<sub>va</sub>ti<sub>ves</sub> t<sub>o</sub> <sub>zero</sub> <sub>g</sub>ives the conditions in Theorem 1(ii), which are necessar<sub>y</sub> and suficient for o<sub>p</sub>timalit<sub>y</sub> b<sub>y</sub> convexit<sub>y</sub>.

We next characterize when an exce<sub>p</sub>tional <sub>p</sub>oint in Theorem 1(i) is a minimizer. Fix � and consider $\mathbf { \Phi } ( q , b ) = ( \mu _ { t - 1 , j } , e _ { j } )$ <sub>.</sub> F<sub>or</sub> <sub>any</sub> di<sub>rec</sub>ti<sub>on</sub> $( c , d )$ , the directional derivative of the �th positive-part term is

$$
\operatorname* { l i m } _ { \epsilon \downarrow 0 } \frac { \mathbb { E } \Big [ \big ( \theta _ { j } - a _ { \mu _ { t - 1 , j } + \epsilon c , e _ { j } + \epsilon d } ( \theta ) \big ) ^ { + } \Big | p _ { t - 1 } \Big ] - \mathbb { E } \Big [ \big ( \theta _ { j } - a _ { \mu _ { t - 1 , j } , e _ { j } } ( \theta ) \big ) ^ { + } \Big | p _ { t - 1 } \Big ] } { \epsilon }
$$

$$
\begin{array} { l } { { \displaystyle = \operatorname* { l i m } _ { \epsilon \downarrow 0 } \frac { 1 } { \epsilon } \mathbb { E } \big [ \big ( \theta _ { j } - \mu _ { t - 1 , j } - \epsilon c - ( e _ { j } + \epsilon d ) ^ { \top } ( \pmb { \theta } - \mu _ { t - 1 } ) \big ) ^ { + } \Big \vert p _ { t - 1 } \big ] } } \\ { { \displaystyle = \operatorname* { l i m } _ { \epsilon \downarrow 0 } \frac { 1 } { \epsilon } \mathbb { E } \big [ \big ( - \epsilon c - \epsilon d ^ { \top } ( \pmb { \theta } - \mu _ { t - 1 } ) \big ) ^ { + } \Big \vert p _ { t - 1 } \big ] } } \\ { { \displaystyle = \mathbb { E } \big [ \big ( - c - d ^ { \top } ( \pmb { \theta } - \mu _ { t - 1 } ) \big ) ^ { + } \Big \vert p _ { t - 1 } \big ] . } } \end{array}
$$

H<sub>ence</sub> th<sub>e</sub> di<sub>rec</sub>ti<sub>ona</sub>l d<sub>er</sub>i<sub>va</sub>ti<sub>ve</sub> <sub>o</sub>f th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>m</sub>i<sub>sma</sub>t<sub>c</sub>h l<sub>oss</sub> <sub>a</sub>t $( \mu _ { t - 1 , j } , e _ { j } )$ i<sub>n</sub> di<sub>rec</sub>ti<sub>on</sub> $( c , d )$ i<sub>s</sub>

$$
c \left[ k - \sum _ { i \neq j } \Phi \left( \frac { \mu _ { t - 1 , i } - \mu _ { t - 1 , j } } { \sigma _ { t - 1 , i , j } } \right) \right] - d ^ { \top } \Sigma _ { t - 1 } \sum _ { i \neq j } \frac { \phi \left( \frac { \mu _ { t - 1 , i } - \mu _ { t - 1 , j } } { \sigma _ { t - 1 , i , j } } \right) } { \sigma _ { t - 1 , i j } } \left( e _ { i } - e _ { j } \right) + \mathbb { B } \left[ \left( - c - d ^ { \top } ( \theta - \mu _ { t - 1 } ) \right) ^ { + } \middle | p _ { t - 1 } \right] .
$$

<sup>B</sup><sub>y</sub> convex<sup>i</sup>t<sub>y</sub>, $( \mu _ { t - 1 , j } , e _ { j } )$ i<sub>s</sub> <sub>a</sub> <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zer</sub> if <sub>an</sub>d <sub>on</sub>l<sub>y</sub> if thi<sub>s</sub> di<sub>rec</sub>ti<sub>ona</sub>l d<sub>er</sub>i<sub>va</sub>ti<sub>ve</sub> i<sub>s</sub> <sub>nonnega</sub>ti<sub>ve</sub> f<sub>or</sub> <sub>every</sub> $( c , d )$ . At � = 0 this requires and is equivalent to

$$
0 \leq k - \sum _ { i \neq j } \Phi \Bigg ( \frac { \mu _ { t - 1 , i } - \mu _ { t - 1 , j } } { \sigma _ { t - 1 , i j } } \Bigg ) \leq 1 .\tag{C.1}
$$

Now, for any fixed �, taking the infimum of the directional derivative over � gives

$$
\phi \left( \Phi ^ { - 1 } \left[ k - \sum _ { i \neq j } \Phi \left( { \frac { \mu _ { t - 1 , i } - \mu _ { t - 1 , j } } { \sigma _ { t - 1 , i j } } } \right) \right] \right) { \sqrt { d ^ { \top } \Sigma _ { t - 1 } d } } - d ^ { \top } \Sigma _ { t - 1 } \sum _ { i \neq j } { \frac { \phi \left( { \frac { \mu _ { t - 1 , i } - \mu _ { t - 1 , j } } { \sigma _ { t - 1 , i j } } } \right) } { \sigma _ { t - 1 , i j } } } ( e _ { i } - e _ { j } ) .
$$

Requiring this to be nonnegative for every � is equivalent to

$$
\left( \sum _ { i \neq j } \frac { \phi \left( \frac { \mu _ { t - 1 , i } - \mu _ { t - 1 , j } } { \sigma _ { t - 1 , i , j } } \right) } { \sigma _ { t - 1 , i , j } } ( e _ { i } - e _ { j } ) \right) ^ { \top } \Sigma _ { t - 1 } \left( \sum _ { i \neq j } \frac { \phi \left( \frac { \mu _ { t - 1 , i } - \mu _ { t - 1 , j } } { \sigma _ { t - 1 , i , j } } \right) } { \sigma _ { t - 1 , i , j } } ( e _ { i } - e _ { j } ) \right) \leq \phi \left( \Phi ^ { - 1 } \left[ k - \sum _ { i \neq j } \Phi \left( \frac { \mu _ { t - 1 , i } - \mu _ { t - 1 , j } } { \sigma _ { t - 1 , i , j } } \right) \right] \right) ^ { 2 } .\tag{C.2}
$$

Th<sub>ere</sub>f<sub>ore,</sub> $( \mu _ { t - 1 , j } , e _ { j } )$ is a minimizer if and onl<sub>y</sub> if (C.1) and (C.2) hold. □

Proof of Corollary 1. For part (i), by Theorem 1(ii), it is enough to verify the two optimality conditions. W<sub>e</sub> fi<sub>rs</sub>t <sub>cons</sub>id<sub>er</sub> th<sub>e con</sub>diti<sub>on</sub> $\Xi \big [ \kappa _ { q _ { t } , b _ { t } } ( \pmb { \theta } ) \big | p _ { t - 1 } \big ] = 0$

$$
\mathbb { E } \Big [ \kappa _ { q _ { t } , b _ { t } } ( \theta ) \Big | p _ { t - 1 } \Big ] = \mathbb { E } [ \sum _ { i = 1 } ^ { I } \mathbb { 1 } \{ \theta _ { i } > a _ { q _ { t } , b _ { t } } ( \theta ) \} - k | p _ { t - 1 } ] = \sum _ { i = 1 } ^ { I } \mathbb { P } \big ( \theta _ { i } > a _ { q _ { t } , b _ { t } } ( \theta ) \big | p _ { t - 1 } \big ) - k .
$$

W<sub>e</sub> <sub>nex</sub>t <sub>compu</sub>t<sub>e</sub> th<sub>ese</sub> <sub>pro</sub>b<sub>a</sub>biliti<sub>es</sub>

$$
\mathbb { P } \big ( \theta _ { i } > a _ { q _ { t } , b _ { t } } ( \theta ) \big | p _ { t - 1 } \big ) = \mathbb { P } \big ( ( e _ { i } - b _ { t } ) ^ { \top } ( \theta - \mu _ { t - 1 } ) > q _ { t } - \mu _ { t - 1 , i } \big | p _ { t - 1 } \big ) = 1 - \Phi \bigg ( \frac { q _ { t } - \mu _ { t - 1 , i } } { \sigma _ { t - 1 , i } } \bigg ) = \Phi \big ( z _ { t - 1 , i } \big ) .
$$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> thi<sub>s</sub> b<sub>ac</sub>k <sub>g</sub>i<sub>ves</sub>

$$
\mathbb { E } \left[ \kappa _ { q _ { t } , b _ { t } } ( \pmb { \theta } ) \Big | p _ { t - 1 } \right] = \sum _ { i = 1 } ^ { I } \Phi ( z _ { t - 1 , i } ) - k .
$$

We next verif<sub>y</sub> the covariance condition in Theorem 1(ii). Ex<sub>p</sub>andin<sub>g</sub> $\boldsymbol { \kappa } _ { \boldsymbol { q } _ { t } , \boldsymbol { b } _ { t } }$ <sub>an</sub>d <sub>us</sub>i<sub>ng</sub> E $\ Q _ { \ell } - \mu _ { t - 1 , \ell } \ \big | \ p _ { t - 1 } \big ] = 0$

$$
\mathrm { C o v } \big ( \theta _ { \ell } , \kappa _ { q _ { t } , b _ { t } } ( \theta ) \big | p _ { t - 1 } \big ) = \sum _ { i = 1 } ^ { I } \mathbb { E } \big [ ( \theta _ { \ell } - \mu _ { t - 1 , \ell } ) \mathbb { 1 } \big \{ \theta _ { i } > a _ { q _ { t } , b _ { t } } ( \theta ) \big \} \Big | p _ { t - 1 } \big ] .
$$

F<sub>or eac</sub>h �<sub>,</sub> th<sub>e var</sub>i<sub>a</sub>bl<sub>es</sub> $\theta _ { \ell } - \mu _ { t - 1 , \ell }$ <sub>an</sub>d $( { \pmb e } _ { i } - { \pmb b } _ { t } ) ^ { \top } ( { \pmb \theta } - { \pmb \mu } _ { t - 1 } )$ are jointly Gaussian. Thus,

$$
\begin{array} { r l } & { \mathbb { B } \big [ ( \theta _ { \ell } - \mu _ { t - 1 , \ell } ) { \mathbb 1 } \big \lbrace \theta _ { i } > a _ { q _ { t } , b _ { t } } ( \theta ) \big \rbrace \bigg | p _ { t - 1 } \big ] } \\ & { \qquad = \frac { e _ { \ell } ^ { \top } \sum _ { t = 1 } ^ { \ell } ( e _ { i } - b _ { t } ) } { \sigma _ { t - 1 , i } ^ { 2 } } \mathbb { B } \big [ ( e _ { i } - b _ { t } ) ^ { \top } ( \theta - \mu _ { t - 1 } ) { \mathbb 1 } \big \lbrace ( e _ { i } - b _ { t } ) ^ { \top } ( \theta - \mu _ { t - 1 } ) > q _ { t } - \mu _ { t - 1 , i } \big \rbrace \bigg | p _ { t - 1 } \big ] } \\ & { \qquad = \frac { \phi ( z _ { t - 1 , i } ) } { \sigma _ { t - 1 , i } } e _ { \ell } ^ { \top } \Sigma _ { t - 1 } ( e _ { i } - b _ { t } ) = w _ { t - 1 , i } e _ { \ell } ^ { \top } \Sigma _ { t - 1 } ( e _ { i } - b _ { t } ) . } \end{array}
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
\mathrm { C o v } \big ( \theta _ { \ell } , \kappa _ { q _ { t } , b _ { t } } ( \theta ) \big | p _ { t - 1 } \big ) = \sum _ { i = 1 } ^ { I } w _ { t - 1 , i } e _ { \ell } ^ { \top } \Sigma _ { t - 1 } \big ( e _ { i } - b _ { t } \big ) = e _ { \ell } ^ { \top } \Sigma _ { t - 1 } \left( \sum _ { i = 1 } ^ { I } w _ { t - 1 , i } e _ { i } - \left( \sum _ { i = 1 } ^ { I } w _ { t - 1 , i } \right) b _ { t } \right) .
$$

Th<sub>us,</sub> t<sub>a</sub>ki<sub>ng</sub>

$$
{ \pmb { b } } _ { t } = \frac { \sum _ { i = 1 } ^ { I } w _ { t - 1 , i } e _ { i } } { \sum _ { i = 1 } ^ { I } w _ { t - 1 , i } } , \qquad \mathrm { e q u i v a l e n t l y } \qquad { b } _ { t , i } = \frac { w _ { t - 1 , i } } { \sum _ { \ell = 1 } ^ { I } w _ { t - 1 , \ell } } ,
$$

<sub>ma</sub>k<sub>es</sub> th<sub>e covar</sub>i<sub>ance zero</sub> f<sub>or every</sub> $\ell = 1 , \ldots , I$ <sub>.</sub> H<sub>ence</sub> b<sub>o</sub>th <sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> <sub>con</sub>diti<sub>ons</sub> <sub>o</sub>f Th<sub>eorem</sub> 1 <sub>are</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>e</sub>d<sub>,</sub> <sub>p</sub>rovin<sub>g</sub> <sub>p</sub>art (i).

For <sub>p</sub>art (ii), set $\pmb { b = 0 }$ <sub>,</sub> <sub>so</sub> <sub>on</sub>l<sub>y</sub> th<sub>e</sub> <sub>cons</sub>t<sub>an</sub>t th<sub>res</sub>h<sub>o</sub>ld <sub>�</sub> i<sub>s</sub> <sub>op</sub>ti<sub>m</sub>i<sub>ze</sub>d<sub>.</sub> B<sub>y</sub> <sub>convex</sub>it<sub>y,</sub> th<sub>e</sub> <sub>op</sub>ti<sub>mum</sub> i<sub>s</sub> <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>ze</sub>d b<sub>y</sub> th<sub>e</sub> fi<sub>rs</sub>t<sub>-or</sub>d<sub>er con</sub>diti<sub>on</sub>

$$
0 = \frac { \partial } { \partial q } \mathbb { E } \big [ M ( \theta ; S _ { k } , a _ { q , \mathbf { 0 } } ) \big | p _ { t - 1 } \big ] = k - \sum _ { i = 1 } ^ { I } \mathbb { P } \big ( \theta _ { i } > q \big | p _ { t - 1 } \big ) = k - \sum _ { i = 1 } ^ { I } \Phi \bigg ( \frac { \mu _ { t - 1 , i } - q } { \sigma _ { t - 1 , i 0 } } \bigg ) .
$$

Th<sub>ere</sub>f<sub>ore</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>cons</sub>t<sub>an</sub>t th<sub>res</sub>h<sub>o</sub>ld $q _ { t } ^ { \mathrm { c o n } }$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\sum _ { i = 1 } ^ { I } \Phi \Bigg ( \frac { \mu _ { t - 1 , i } - q _ { t } ^ { \mathrm { c o n } } } { \sigma _ { t - 1 , i 0 } } \Bigg ) = k .
$$

ProofofLemma 2. Fix an evaluation $( i , j ) \in \mathcal { I }$ <sub>.</sub> B<sub>y</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> <sub>o</sub>f <sub>expec</sub>t<sub>e</sub>d <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>covar</sub>i<sub>ance</sub> <sub>an</sub>d th<sub>e</sub> l<sub>aw</sub> <sub>o</sub>f t<sub>o</sub>t<sub>a</sub>l <sub>covar</sub>i<sub>ance,</sub>

$$
\begin{array} { r } { \overline { { \Sigma } } _ { t } \left( ( i , j ) , p _ { t - 1 } \right) = \mathrm { C o v } ( \theta \mid p _ { t - 1 } ) - \mathrm { C o v } \big ( \mathbb { E } \left[ \theta \mid X _ { i j } , p _ { t - 1 } \right] \Big | p _ { t - 1 } \big ) = \sum _ { t = 1 } - \mathrm { C o v } \big ( \mathbb { E } \left[ \theta \mid X _ { i j } , p _ { t - 1 } \right] \Big | p _ { t - 1 } \big ) . } \end{array}
$$

Th<sub>us,</sub> <sub>we</sub> <sub>nee</sub>d th<sub>e</sub> <sub>covar</sub>i<sub>ance</sub> <sub>o</sub>f th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>mean</sub> <sub>across</sub> th<sub>e</sub> t<sub>wo</sub> <sub>poss</sub>ibl<sub>e</sub> <sub>ou</sub>t<sub>comes.</sub> W<sub>e</sub> <sub>compu</sub>t<sub>e</sub> thi<sub>s</sub> f<sub>rom</sub> th<sub>e pro</sub>b<sub>a</sub>bilit<sub>y o</sub>f <sub>eac</sub>h <sub>ou</sub>t<sub>come an</sub>d it<sub>s resu</sub>lti<sub>ng pos</sub>t<sub>er</sub>i<sub>or mean.</sub>

Fi<sub>rs</sub>t <sub>cons</sub>id<sub>er</sub> th<sub>e</sub> <sub>ou</sub>t<sub>come</sub> <sub>pro</sub>b<sub>a</sub>biliti<sub>es.</sub> L<sub>e</sub>t $Z \sim N ( 0 , 1 )$ be independent of �. Then

$$
\begin{array} { r l } & { \mathbb { P } ( X _ { i j } = x \mid p _ { t - 1 } ) = \mathbb { E } \big [ \Phi \big ( x \beta _ { J } ( \theta _ { i } - \theta _ { j } ) \big ) \big | p _ { t - 1 } \big ] } \\ & { \qquad = \mathbb { P } \big ( Z \le x \beta _ { J } ( \theta _ { i } - \theta _ { j } ) \big | p _ { t - 1 } \big ) } \\ & { \qquad = \mathbb { P } \left( \frac { Z - x \beta _ { J } ( \theta _ { i } - \theta _ { j } - \mu _ { t - 1 , i } + \mu _ { t - 1 , j } ) } { \sqrt { 1 + \beta _ { J } ^ { 2 } \sigma _ { t - 1 , i j } ^ { 2 } } } \le \frac { x \beta _ { J } ( \mu _ { t - 1 , i } - \mu _ { t - 1 , j } ) } { \sqrt { 1 + \beta _ { J } ^ { 2 } \sigma _ { t - 1 , i j } ^ { 2 } } } \Bigg | p _ { t - 1 } \right) } \\ & { \qquad = \Phi \Bigg ( \frac { x \beta _ { J } ( \mu _ { t - 1 , i } - \mu _ { t - 1 , j } ) } { \sqrt { 1 + \beta _ { J } ^ { 2 } \sigma _ { t - 1 , i j } ^ { 2 } } } \Bigg ) = \Phi ( x \alpha _ { t , i j } ) . } \end{array}
$$

W<sub>e</sub> n<sub>ex</sub>t <sub>co</sub>m<sub>pu</sub>t<sub>e</sub> th<sub>e</sub> <sub>pos</sub>t<sub>e</sub>ri<sub>o</sub>r m<sub>ea</sub>n<sub>.</sub> B<sub>y</sub> B<sub>ayes</sub>’ r<sub>u</sub>l<sub>e,</sub>

$$
\mathbb { E } [ \theta \mid X _ { i j } = x , p _ { t - 1 } ] - \mu _ { t - 1 } = \frac { \mathbb { E } \big [ ( \theta - \mu _ { t - 1 } ) \mathbb { P } ( X _ { i j } = x \mid \theta ) \big | p _ { t - 1 } \big ] } { \mathbb { P } ( X _ { i j } = x \mid p _ { t - 1 } ) } = \frac { \mathbb { E } \big [ ( \theta - \mu _ { t - 1 } ) \Phi \big ( x \beta _ { J } ( \theta _ { i } - \theta _ { j } ) \big ) \big | p _ { t - 1 } \\big ] } { \Phi ( x \alpha _ { t , i j } ) } .
$$

It <sub>rema</sub>i<sub>ns</sub> t<sub>o eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e numera</sub>t<sub>or.</sub>

$$
\begin{array} { r l } & { \mathbb { E } \big [ ( \theta - \mu _ { r - 1 } ) \Phi \big ( x \beta _ { j } ( \theta _ { i } - \theta _ { j } ) \big ) \big | \mathcal { P } _ { t - 1 } \big ] } \\ & { \quad = \mathbb { E } \big [ ( \theta - \mu _ { t - 1 } ) \mathbb { 1 } \{ Z \leq x \beta _ { j } ( \theta _ { i } - \theta _ { j } ) \} \big | \mathcal { P } _ { t - 1 } \big ] } \\ & { \quad = \frac { x \beta _ { j } \sum _ { t - 1 } ( \theta _ { i } - \theta _ { j } ) } { 1 + \beta _ { j } ^ { 2 } \sigma _ { t - 1 , i j } ^ { 2 } } \mathbb { E } \big [ ( x \beta _ { j } ( \theta _ { i } - \theta _ { j } - \mu _ { t - 1 , i } + \mu _ { t - 1 , j } ) - Z ) \mathbb { 1 } \{ Z \leq x \beta _ { j } ( \theta _ { i } - \theta _ { j } ) \} \big | \mathcal { P } _ { t - 1 } \big ] } \\ & { \quad = \frac { x \beta _ { j } \sum _ { t - 1 } ( \theta _ { i } - \epsilon _ { j } ) } { \sqrt { 1 + \beta _ { j } ^ { 2 } \sigma _ { t - 1 , i j } ^ { 2 } } } \int _ { - x \alpha _ { t , i j } } ^ { \infty } u \phi ( u ) d u } \\ & { \quad = \frac { x \beta _ { j } \phi \phi ( \alpha _ { t , i j } ) } { \sqrt { 1 + \beta _ { j } ^ { 2 } \sigma _ { t - 1 , i j } ^ { 2 } } } \sum _ { t - 1 ( \epsilon _ { i } - \epsilon _ { j } ) . } } \end{array}
$$

Th<sub>e secon</sub>d <sub>equa</sub>lit<sub>y uses</sub> th<sub>e covar</sub>i<sub>ance</sub> $\mathrm { C o v } ( \pmb \theta , x \beta _ { J } ( \theta _ { i } - \theta _ { j } ) - Z \mid p _ { t - 1 } ) = x \beta _ { J } \Sigma _ { t - 1 } ( e _ { i } - e _ { j } )$ <sub>an</sub>d <sub>var</sub>i<sub>ance</sub> $1 + \beta _ { J } ^ { 2 } \sigma _ { t - 1 , i j } ^ { 2 }$ <sub>.</sub> Th<sub>e</sub> thi<sub>r</sub>d <sub>equa</sub>lit<sub>y s</sub>t<sub>an</sub>d<sub>ar</sub>di<sub>zes</sub> th<sub>e</sub> G<sub>auss</sub>i<sub>an, an</sub>d th<sub>e</sub> l<sub>as</sub>t <sub>uses</sub> $\phi ^ { \prime } ( u ) = - u \phi ( u )$ <sup>and</sup> <sup>s</sup>y<sup>mmetr</sup>y <sub>o</sub>f $\phi .$ .

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>mean</sub> <sub>g</sub>i<sub>ves</sub>

$$
\mathbb { E } [ \theta \mid X _ { i j } = x , p _ { t - 1 } ] - \mu _ { t - 1 } = \frac { x \beta _ { J } \phi ( \alpha _ { t , i j } ) } { \Phi ( x \alpha _ { t , i j } ) \sqrt { 1 + \beta _ { J } ^ { 2 } \sigma _ { t - 1 , i j } ^ { 2 } } } \Sigma _ { t - 1 } ( e _ { i } - e _ { j } ) .
$$

W<sub>e</sub> <sub>can</sub> <sub>now</sub> <sub>compu</sub>t<sub>e</sub> th<sub>e</sub> <sub>covar</sub>i<sub>ance</sub> t<sub>erm.</sub> B<sub>y</sub> th<sub>e</sub> l<sub>aw</sub> <sub>o</sub>f t<sub>o</sub>t<sub>a</sub>l <sub>expec</sub>t<sub>a</sub>ti<sub>on,</sub> E $[ \mathbb { E } [ \pmb { \theta } | X _ { i j } , p _ { t - 1 } ] | p _ { t - 1 } ] = \pmb { \mu _ { t - 1 } }$ so

$$
\begin{array} { r l } &  \begin{array} { r l } & { \mathrm { C o w } ( \overline { { \mathbb { S } } } [ \theta | X _ { i } , p _ { j } , p _ { i - 1 } ] | p _ { t - 1 } ) } \\ & { \quad = \qquad \displaystyle \sum _ { x \in \{ - 1 , 1 \} } \mathbb { \Sigma } ( X _ { i j } = x | p _ { - 1 } ) ( \mathbb { E } [ \theta | X _ { i j } = x , p _ { t - 1 } ] - \mu _ { t - 1 } ) | \mathbb { S } [ \theta | X _ { i j } = x , p _ { t - 1 } ] - \mu _ { t - 1 } | ^ { \pi }  } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & & { = \frac { \beta _ { j } ^ { 2 } \theta ( \theta \alpha _ { i , j } ) ^ { 2 } } { 1 + \beta _ { j } ^ { 2 } \sigma _ { i - 1 , i j } ^ { 2 } } ( \underbrace { \sum _ { x \in \{ - 1 , 1 \} } \mathbb { S } ( x \alpha _ { i , j } , u ) } _ { \mathrm { { C o t } } ( x , q ) } [ \frac { x } { \Phi ( x \alpha _ { i , j } , u ) } ] ^ { 2 } ) \Bigg \} \mathbb { E } _ { r - 1 } ( e _ { i } - e _ { j } ) ( e _ { i } - e _ { j } ) ^ { \top } \mathbb { T } _ { z - 1 } } \\ &  \quad = \frac { \beta _ { j } ^ { 2 } \theta ( \alpha _ { i , j } ) ^ { 2 } } { 1 + \beta _ { j } ^ { 2 } \sigma _ { i - 1 , i j } ^ { 2 } } ( \frac { 1 } { \Phi ( \alpha _ { i , j } , u ) } + \frac { 1 } { \Phi ( - \alpha _ { i , j } , u ) } ) \sum _ { t = 1 } ( e _ { i } - e _ { j } ) \end{array} \end{array}
$$

H<sub>ere we use</sub>d $x ^ { 2 } = 1 , \Phi ( \alpha _ { t , i j } ) + \Phi ( - \alpha _ { t , i j } ) = 1$ <sub>, an</sub>d th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on o</sub>f $\delta _ { t , i j }$ <sub>.</sub> S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> b<sub>ac</sub>k i<sub>n</sub>t<sub>o</sub> th<sub>e expec</sub>t<sub>e</sub>d <sub>pos</sub>t<sub>er</sub>i<sub>or covar</sub>i<sub>ance y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\overline { { \Sigma } } _ { t } ( ( i , j ) , p _ { t - 1 } ) = \Sigma _ { t - 1 } - \frac { \delta _ { t , i j } } { \sigma _ { t - 1 , i j } ^ { 2 } } \Sigma _ { t - 1 } ( e _ { i } - e _ { j } ) ( e _ { i } - e _ { j } ) ^ { \top } \Sigma _ { t - 1 } .
$$

## A<sub>ppe</sub>ndi<sub>x</sub> D<sub>:</sub> Pr<sub>oo</sub>f <sub>o</sub>f Th<sub>eo</sub>r<sub>e</sub>m 2

All <sub>pro</sub>b<sub>a</sub>biliti<sub>es</sub> <sub>an</sub>d <sub>expec</sub>t<sub>a</sub>ti<sub>ons</sub> b<sub>e</sub>l<sub>ow</sub> <sub>are</sub> <sub>w</sub>ith <sub>respec</sub>t t<sub>o</sub> th<sub>e</sub> <sub>pr</sub>i<sub>or,</sub> th<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l<sub>,</sub> <sub>an</sub>d th<sub>e</sub> <sub>po</sub>li<sub>cy.</sub> U<sub>n</sub>til th<sub>e</sub> fi<sub>na</sub>l <sub>convergence</sub> <sub>argumen</sub>t<sub>,</sub> <sub>cons</sub>id<sub>er</sub> SA <sub>run</sub> <sub>con</sub>ti<sub>nuous</sub>l<sub>y</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>a</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>.</sub>

## D<sub>.</sub>1<sub>.</sub> N<sub>o</sub>t<sub>a</sub>ti<sub>o</sub>n

L<sub>e</sub>t

$$
P : = \mathrm { I d } _ { I } - \frac 1 I \mathbf { 1 } \mathbf { 1 } ^ { \top } , \qquad \mathbf { 1 } ^ { \bot } : = \{ \pmb { x } \in \mathbb { R } ^ { I } : \mathbf { 1 } ^ { \top } \pmb { x } = 0 \} .
$$

Let D denote the finite set of evaluation directions available to the <sub>p</sub>olic<sub>y</sub>. A ratin<sub>g</sub> of item � has direction $\mathbf { } _ { i } ,$ <sub>an</sub>d <sub>a pa</sub>i<sub>rw</sub>i<sub>se compar</sub>i<sub>son o</sub>f it<sub>ems</sub> � <sub>an</sub>d $j$ h<sub>as</sub> di<sub>rec</sub>ti<sub>on</sub> $e _ { i } - e _ { j }$ <sub>.</sub> Th<sub>us,</sub>

$$
\mathcal { D } = \{ e _ { i } - e _ { j } : ( i , j ) \in J \} ,
$$

<sub>w</sub>h<sub>ere</sub> $\mathbf { \epsilon } _ { \mathbf { \lambda } \mathbf { \lambda } \mathbf { \tilde { e } } _ { 0 } } = \mathbf { 0 }$ <sub>,</sub> <sub>an</sub>d $\mathbf { 1 } ^ { \perp } \subseteq \operatorname { s p a n } ( \mathcal { D } )$ <sub>.</sub> F<sub>or</sub> $L \in \mathcal { D }$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\beta _ { L } : = \left\{ \begin{array} { l l } { \beta _ { 0 } , } & { L \mathrm { ~ i s ~ a ~ r a t i n g ~ d i r e c t i o n } , } \\ { \beta _ { 1 } , } & { L \mathrm { ~ i s ~ a ~ c o m p a r i s o n ~ d i r e c t i o n } , } \end{array} \right.
$$

$$
\theta _ { 0 } ( L ) : = \left\{ \begin{array} { l l } { \theta _ { 0 } , } & { L \mathrm { ~ i s ~ a ~ r a t i n g ~ d i r e c t i o n } , } \\ { 0 , } & { L \mathrm { ~ i s ~ a ~ c o m p a r i s o n ~ d i r e c t i o n } , } \end{array} \right.
$$

<sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub> $r _ { L }$ ana<sup>l</sup>o<sub>g</sub>ous<sup>l</sup><sub>y</sub> as $r _ { 0 }$ or $r _ { 1 }$ . Conditional on � and the history before an evaluation, its success <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> d<sub>epen</sub>d<sub>s</sub> <sub>on</sub>l<sub>y</sub> <sub>on</sub> th<sub>e</sub> <sub>c</sub>h<sub>osen</sub> di<sub>rec</sub>ti<sub>on.</sub> F<sub>or</sub> <sub>every</sub> $L \in \mathcal { D }$

$$
\mathbb { P } ( X _ { L } = + 1 \mid \pmb \theta ) = \mathbb { E } _ { \epsilon _ { J } } \left[ g \left( L ^ { \top } \pmb \theta - \theta _ { 0 } ( L ) + \epsilon _ { J } \right) \right] ,
$$

<sub>w</sub>h<sub>ere</sub> $\jmath = 0$ f<sub>or a ra</sub>ti<sub>ng</sub> di<sub>rec</sub>ti<sub>on an</sub>d $\jmath = 1$ f<sub>or a compar</sub>i<sub>son</sub> di<sub>rec</sub>ti<sub>on.</sub> Thi<sub>s pro</sub>b<sub>a</sub>bilit<sub>y</sub> i<sub>s con</sub>ti<sub>nuous an</sub>d <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> i<sub>ncreas</sub>i<sub>ng</sub> i<sub>n</sub> $L ^ { \top } \pmb \theta$

L<sub>e</sub>t $\mathcal { H } _ { t }$ b<sub>e</sub> th<sub>e</sub> hi<sub>s</sub>t<sub>ory ava</sub>il<sub>a</sub>bl<sub>e a</sub>t th<sub>e s</sub>t<sub>ar</sub>t <sub>o</sub>f <sub>roun</sub>d �<sub>,</sub> i<sub>.e.,</sub> th<sub>e �-a</sub>l<sub>ge</sub>b<sub>ra genera</sub>t<sub>e</sub>d b<sub>y a</sub>ll <sub>eva</sub>l<sub>ua</sub>ti<sub>on c</sub>h<sub>o</sub>i<sub>ces</sub> <sub>an</sub>d <sub>ou</sub>t<sub>comes</sub> th<sub>roug</sub>h <sub>roun</sub>d $t - 1$ <sub>, an</sub>d l<sub>e</sub>t

$$
\pmb { \mu } _ { t - 1 } ^ { \ast } : = \mathbb { E } [ \pmb { \theta } | \mathcal { H } _ { t } ] , \qquad \Sigma _ { t - 1 } ^ { \ast } : = \mathrm { C o v } ( \pmb { \theta } | \mathcal { H } _ { t } )
$$

b<sub>e</sub> th<sub>e</sub> <sub>exac</sub>t <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>momen</sub>t<sub>s</sub> <sub>a</sub>t th<sub>e</sub> <sub>s</sub>t<sub>ar</sub>t <sub>o</sub>f <sub>roun</sub>d �<sub>.</sub> SA <sub>uses</sub> th<sub>e</sub> <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub> <sub>momen</sub>t<sub>s</sub> $\left( \mu _ { t - 1 } , \Sigma _ { t - 1 } \right)$

F<sub>or</sub> <sub>no</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>conven</sub>i<sub>ence,</sub> <sub>wr</sub>it<sub>e</sub>

$$
g _ { t , i } : = e _ { i } - b _ { t } , \qquad \sigma _ { t - 1 , i } ^ { 2 } : = g _ { t , i } ^ { \top } \Sigma _ { t - 1 } g _ { t , i } , \qquad z _ { t - 1 , i } : = \frac { \mu _ { t - 1 , i } - q _ { t } } { \sigma _ { t - 1 , i } } , \qquad w _ { t - 1 , i } : = \frac { \phi ( z _ { t - 1 , i } ) } { \sigma _ { t - 1 , i } } .
$$

H<sub>ere</sub> $z _ { t - 1 , i }$ <sub>an</sub>d $w _ { t - 1 , i }$ <sub>are</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>on</sub>l<sub>y w</sub>h<sub>en</sub> $\sigma _ { t - 1 , i } > 0$ <sub>.</sub> B<sub>y</sub> Th<sub>eorem</sub> 1 <sub>an</sub>d th<sub>e proo</sub>f <sub>o</sub>f C<sub>oro</sub>ll<sub>ary</sub> 1<sub>, e</sub>ith<sub>er</sub> (i) $\left( q _ { t } , b _ { t } \right) = \left( \mu _ { t - 1 , j } , e _ { j } \right)$ f<sub>or some</sub> $j .$ <sub>.</sub> Th<sub>en</sub> $\sigma _ { t - 1 , j } = 0$ <sub>,</sub> <sub>w</sub>hil<sub>e</sub> $\sigma _ { t - 1 , i } > 0$ f<sub>or</sub> $i \neq j .$ Si<sub>nce</sub> $a _ { q _ { t } , b _ { t } } ( \pmb \theta ) = \theta _ { j }$ <sub>,</sub> th<sub>e</sub> �th loss term is zero before and after every evaluation, so its contribution to the marginal score is zero. (ii) $\sigma _ { t - 1 , i } > 0$ for e<sub>v</sub>er<sub>y</sub> �<sub>,</sub> and

$$
\sum _ { i = 1 } ^ { I } \Phi ( z _ { t - 1 , i } ) = k , \qquad b _ { t , i } = \frac { w _ { t - 1 , i } } { \sum _ { \ell = 1 } ^ { I } w _ { t - 1 , \ell } } , \quad i = 1 , \ldots , I .
$$

N<sub>o</sub>t<sub>e</sub> th<sub>a</sub>t $\mathbf { 1 } ^ { \top } \pmb { b } _ { t } = 1$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub>

$$
g _ { t , i } \in \mathbf { 1 } ^ { \perp } , \qquad g _ { t , i } - g _ { t , j } = e _ { i } - e _ { j } , \qquad \mathrm { s p a n } \{ g _ { t , 1 } , \dots , g _ { t , l } \} = \mathbf { 1 } ^ { \perp } \quad \mathrm { f o r ~ e v e r y ~ } t , \qquad r \in \mathbf { m } ^ { \perp } .
$$

<sub>use</sub>d <sub>repea</sub>t<sub>e</sub>dl<sub>y</sub> b<sub>e</sub>l<sub>ow.</sub>

F<sub>or</sub> $L \in \mathcal { D }$ <sub>,</sub> l<sub>e</sub>t $\alpha _ { t } ( L )$ <sub>an</sub>d $\delta _ { t } ( L )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> <sub>quan</sub>titi<sub>es</sub> $\alpha _ { t , i j }$ <sub>an</sub>d $\delta _ { t , i j }$ i<sub>n</sub> L<sub>emma</sub> 2 f<sub>or</sub> th<sub>e</sub> <sub>correspon</sub>di<sub>ng</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on.</sub> Th<sub>e marg</sub>i<sub>na</sub>l <sub>score o</sub>f SA <sub>can</sub> b<sub>e wr</sub>itt<sub>en</sub> i<sub>n un</sub>ifi<sub>e</sub>d f<sub>orm as</sub>

$$
G _ { t } ( L ) : = \frac { 1 } { r _ { L } } \sum _ { i : \sigma _ { t - 1 , i } > 0 } \int _ { 0 } ^ { \frac { \delta _ { t } ( L ) \left( g _ { t , i } ^ { \top } \Sigma _ { t - 1 } L \right) ^ { 2 } } { \sigma _ { t - 1 , i } ^ { 2 } L ^ { \top } \Sigma _ { t - 1 } L } } \frac { \sigma _ { t - 1 , i } } { 2 \sqrt { 1 - s } } \phi \left( \frac { z _ { t - 1 , i } } { \sqrt { 1 - s } } \right) d s .\tag{D.1}
$$

This is (11) for the corres<sub>p</sub>ondin<sub>g</sub> evaluation, with an<sub>y</sub> zero-variance summand omitted because its contribution i<sub>s</sub> <sub>zero.</sub>

D<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> <sub>rea</sub>li<sub>ze</sub>d <sub>oppor</sub>t<sub>un</sub>it<sub>y</sub> <sub>cos</sub>t <sub>a</sub>ft<sub>er</sub> � <sub>roun</sub>d<sub>s</sub> b<sub>y</sub>

$$
\mathrm { O C } _ { T } : = \sum _ { i \in S _ { k } ^ { * } \left( \pmb { \theta } \right) } \theta _ { i } - \sum _ { i \in S _ { k } ^ { * } \left( \pmb { \mu } _ { T } \right) } \theta _ { i } .
$$

Th<sub>us,</sub> th<sub>e conc</sub>l<sub>us</sub>i<sub>on o</sub>f Th<sub>eorem</sub> 2 i<sub>s</sub> $\mathbb { E } [ \mathrm { O C } _ { T } ] \to 0 \mathrm { a s } R \to \infty .$

## D<sub>.</sub>2<sub>.</sub> Pr<sub>oo</sub>f r<sub>oa</sub>dm<sub>ap</sub>

Th<sub>ere</sub> <sub>are</sub> t<sub>wo</sub> <sub>ma</sub>i<sub>n</sub> <sub>componen</sub>t<sub>s</sub> t<sub>o</sub> th<sub>e</sub> <sub>proo</sub>f<sub>.</sub> Th<sub>e</sub> fi<sub>rs</sub>t<sub>,</sub> <sub>an</sub>d f<sub>ar</sub> <sub>more</sub> t<sub>ec</sub>h<sub>n</sub>i<sub>ca</sub>l<sub>,</sub> i<sub>s</sub> t<sub>o</sub> <sub>argue</sub> th<sub>a</sub>t th<sub>e</sub> <sub>se</sub>t <sub>o</sub>f di<sub>rec</sub>ti<sub>ons</sub> th<sub>e</sub> <sub>a</sub>l<sub>gor</sub>ith<sub>m</sub> <sub>samp</sub>l<sub>es</sub> i<sub>n</sub>fi<sub>n</sub>it<sub>e</sub>l<sub>y</sub> <sub>many</sub> ti<sub>mes</sub> <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on-re</sub>l<sub>evan</sub>t <sub>su</sub>b<sub>space</sub> ${ \bf 1 ^ { \perp } }$ i<sub>n</sub> it<sub>s</sub> <sub>span.</sub> The second component then argues that, if this is the case, the full history identifies ��, and thus the expected <sub>oppo</sub>rt<sub>u</sub>nit<sub>y</sub> <sub>cos</sub>t d<sub>ecays</sub> t<sub>o</sub> 0 <sub>as</sub> <sub>we</sub>ll<sub>.</sub>

T<sub>o prove</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>c</sub>l<sub>a</sub>i<sub>m, we w</sub>ill <sub>argue</sub> b<sub>y con</sub>t<sub>ra</sub>di<sub>c</sub>ti<sub>on.</sub> L<sub>e</sub>t $E _ { \infty }$ b<sub>e</sub> th<sub>e se</sub>t <sub>o</sub>f di<sub>rec</sub>ti<sub>ons samp</sub>l<sub>e</sub>d i<sub>n</sub>fi<sub>n</sub>it<sub>e</sub>l<sub>y o</sub>ft<sub>en, an</sub>d l<sub>e</sub>t $M : = \mathrm { s p a n } ( E _ { \infty } )$ <sub>.</sub> W<sub>e assume</sub> b<sub>y con</sub>t<sub>ra</sub>di<sub>c</sub>ti<sub>on</sub> th<sub>a</sub>t $1 ^ { \perp } \not \in M$ <sub>, an</sub>d <sub>con</sub>diti<sub>on on an even</sub>t $\Omega _ { \mathcal { E } } : = \{ E _ { \infty } = \mathcal { E } \}$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> th<sub>e</sub> <sub>se</sub>t <sub>o</sub>f <sub>recurren</sub>t di<sub>rec</sub>ti<sub>ons</sub> i<sub>s</sub> fi<sub>xe</sub>d<sub>.</sub> O<sub>n</sub> thi<sub>s</sub> <sub>even</sub>t<sub>,</sub> <sub>a</sub>ft<sub>er</sub> <sub>some</sub> fi<sub>n</sub>it<sub>e</sub> <sub>ran</sub>d<sub>om</sub> ti<sub>me,</sub> <sub>every</sub> <sub>su</sub>b<sub>sequen</sub>tl<sub>y</sub> <sub>samp</sub>l<sub>e</sub>d di<sub>rec</sub>ti<sub>on</sub> li<sub>es</sub> i<sub>n</sub> $\varepsilon .$ <sub>.</sub> Thi<sub>s</sub> <sub>re</sub>d<sub>uces</sub> th<sub>e</sub> <sub>proo</sub>f t<sub>o</sub> <sub>s</sub>h<sub>ow</sub>i<sub>ng</sub> t<sub>wo</sub> thi<sub>ngs:</sub> f<sub>or</sub> th<sub>e</sub> di<sub>rec</sub>ti<sub>ons</sub> $L \in { \mathcal { E } }$ <sub>,</sub> th<sub>e</sub>i<sub>r marg</sub>i<sub>na</sub>l <sub>score</sub> $G _ { t } ( L ) \to 0$ <sub>,</sub> <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> t<sub>o</sub>t<sub>a</sub>l <sub>score</sub> <sub>over</sub> $\mathcal { D }$ <sub>s</sub>t<sub>ays</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>away</sub> f<sub>rom</sub> 0<sub>.</sub> Th<sub>us,</sub> f<sub>or a</sub>ll <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> l<sub>arge</sub> $t ,$ at least one direction outside E has a lar<sub>g</sub>er score than ever<sub>y</sub> direction in $\varepsilon _ { \mathrm { { } } }$ contradictin<sub>g</sub> the assum<sub>p</sub>tion that ever<sub>y</sub> subse<sub>q</sub>uentl<sub>y</sub> sam<sub>p</sub>led direction lies in E.

Th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 2 <sub>re</sub>li<sub>es</sub> <sub>on</sub> <sub>e</sub>i<sub>g</sub>ht l<sub>emmas,</sub> <sub>organ</sub>i<sub>ze</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows.</sub> L<sub>emma</sub> D<sub>.</sub>1 <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> th<sub>a</sub>t <sub>pos</sub>t<sub>er</sub>i<sub>or var</sub>i<sub>ance goes</sub> t<sub>o</sub> 0 <sub>an</sub>d th<sub>e pos</sub>t<sub>er</sub>i<sub>or mean converges a</sub>l<sub>ong every</sub> di<sub>rec</sub>ti<sub>on</sub> i<sub>n</sub> th<sub>e span o</sub>f <sub>recurren</sub>tl<sub>y</sub> <sub>samp</sub>l<sub>e</sub>d di<sub>rec</sub>ti<sub>ons.</sub> L<sub>emma</sub> D<sub>.</sub>2 <sub>proves</sub> th<sub>a</sub>t <sub>pos</sub>t<sub>er</sub>i<sub>or var</sub>i<sub>ance rema</sub>i<sub>ns un</sub>if<sub>orm</sub>l<sub>y pos</sub>iti<sub>ve ou</sub>t<sub>s</sub>id<sub>e</sub> th<sub>a</sub>t <sub>span.</sub> L<sub>emmas</sub> D<sub>.</sub>3 <sub>an</sub>d D<sub>.</sub>4 t<sub>rans</sub>f<sub>er</sub> th<sub>ese</sub> <sub>conc</sub>l<sub>us</sub>i<sub>ons</sub> t<sub>o</sub> th<sub>e</sub> <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>covar</sub>i<sub>ance</sub> <sub>an</sub>d <sub>prov</sub>id<sub>e</sub> th<sub>e</sub> b<sub>oun</sub>d<sub>s nee</sub>d<sub>e</sub>d f<sub>or</sub> th<sub>e marg</sub>i<sub>na</sub>l <sub>scores.</sub> L<sub>emma</sub> D<sub>.</sub>5 b<sub>oun</sub>d<sub>s</sub> th<sub>e a</sub>fi<sub>ne</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>an</sub>d th<sub>e we</sub>i<sub>g</sub>ht<sub>s</sub> $w _ { t - 1 , i }$ L<sub>emma</sub> D<sub>.</sub>6 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t th<sub>e scores o</sub>f <sub>recurren</sub>t di<sub>rec</sub>ti<sub>ons</sub> d<sub>ecay, w</sub>hil<sub>e</sub> L<sub>emma</sub> D<sub>.</sub>7 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t th<sub>e</sub> t<sub>o</sub>t<sub>a</sub>l <sub>score</sub> <sub>s</sub>t<sub>ays</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>away</sub> f<sub>rom</sub> 0 <sub>w</sub>h<sub>enever</sub> ${ \bf 1 ^ { \perp } }$ i<sub>s</sub> <sub>no</sub>t <sub>con</sub>t<sub>a</sub>i<sub>ne</sub>d i<sub>n</sub> th<sub>e</sub> <sub>recurren</sub>t <sub>span.</sub> Fi<sub>na</sub>ll<sub>y,</sub> L<sub>emma</sub> D<sub>.</sub>8 <sub>s</sub>h<sub>ows</sub> that “projected” mean convergence sufices for expected opportunity cost.

## D<sub>.</sub>3<sub>.</sub> S<sub>uppo</sub>rtin<sub>g</sub> l<sub>e</sub>mm<sub>as</sub>

LEMMA D.1. Fix a set $\varepsilon \subseteq \mathcal { D }$ , and let $M : = \operatorname { s p a n } ( \mathcal { E } )$ . Let $\mathcal { H } _ { \infty } : = \sigma ( \cup _ { t \geq 1 } \mathcal { H } _ { t } )$ denote the �-algebra generated by the full infinite history. Let $\Omega \in \mathcal H _ { \infty }$ . Suppose that every $L \in { \mathcal { E } }$ is sampled infinitely often on $\Omega .$ Then,for every $\nu \in M$

$$
\nu ^ { \top } \pmb { \mu } _ { t - 1 } ^ { * }  \nu ^ { \top } \pmb { \theta } , \qquad \nu ^ { \top } \Sigma _ { t - 1 } ^ { * } \pmb { \nu }  0 \qquad \alpha . s . \ o n \Omega .
$$

Moreover, $\mathbb { 1 } _ { \Omega } \pmb { \nu } ^ { \top } \pmb { \theta }$ is $\mathcal { H } _ { \infty }$ -measurable.

Proof. For evaluation �, let $L _ { t } : = e _ { i _ { t } } - e _ { j _ { t } } \in \mathcal { D }$ b<sub>e</sub> it<sub>s</sub> <sub>c</sub>h<sub>osen</sub> di<sub>rec</sub>ti<sub>on</sub> <sub>an</sub>d l<sub>e</sub>t $X _ { t } \in \{ - 1 , 1 \}$ b<sub>e</sub> th<sub>e</sub> <sub>o</sub>b<sub>serve</sub>d <sub>ou</sub>t<sub>come.</sub> L<sub>e</sub>t $( \mathcal { F } _ { t } ) _ { t \geq 0 }$ b<sub>e</sub> th<sub>e</sub> filt<sub>ra</sub>ti<sub>on</sub> d<sub>e</sub>fi<sub>ne</sub>d b<sub>y</sub> $\mathcal { F } _ { t - 1 } : = \sigma ( \pmb { \theta } , \mathcal { H } _ { t } , L _ { t } )$ f<sub>or</sub> $t \geq 1$ <sub>.</sub> Th<sub>en</sub> $\mathcal { F } _ { t - 1 }$ <sub>con</sub>t<sub>a</sub>i<sub>ns a</sub>ll <sub>prev</sub>i<sub>ous</sub>l<sub>y o</sub>b<sub>serve</sub>d <sub>ou</sub>t<sub>comes an</sub>d th<sub>e</sub> di<sub>rec</sub>ti<sub>on c</sub>h<sub>osen</sub> b<sub>e</sub>f<sub>ore o</sub>b<sub>serv</sub>i<sub>ng</sub> $X _ { t }$ <sub>.</sub> I<sub>n par</sub>ti<sub>cu</sub>l<sub>ar,</sub> $L _ { t }$ i<sub>s</sub> $\mathscr { F } _ { t - 1 }$ <sub>-measura</sub>bl<sub>e.</sub>

Fi<sub>x</sub> $L \in { \mathcal { E } }$ <sub>,</sub> <sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub>

$$
N _ { t } ( L ) : = \sum _ { s = 1 } ^ { t } \mathbb { 1 } \{ L _ { s } = L \} , \qquad { \widehat { X } } _ { t } : = \mathbb { 1 } \{ X _ { t } = + 1 \} .
$$

S<sub>e</sub>t

$$
\xi _ { t } ( L ) : = \mathbb { 1 } \left\{ L _ { t } = L \right\} \left( \widehat { X } _ { t } - \mathbb { P } ( X _ { L } = + 1 \mid \pmb \theta ) \right) .
$$

Si<sub>nce</sub> 1 $\{ L _ { t } = L \}$ i<sub>s</sub> $\mathcal { F } _ { t - 1 }$ <sub>-measura</sub>bl<sub>e, an</sub>d <sub>s</sub>i<sub>nce con</sub>diti<sub>ona</sub>l <sub>on</sub> $\mathcal { F } _ { t - 1 }$ th<sub>e</sub> <sub>success</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>on</sub> th<sub>e</sub> <sub>even</sub>t $\{ L _ { t } = L \}$ i<sub>s</sub> $\mathbb { P } ( X _ { L } = + 1 \mid \pmb { \theta } )$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub> $\mathbb { E } [ \xi _ { t } ( L ) \mid \mathcal { F } _ { t - 1 } ] = 0$ <sub>.</sub> Th<sub>us</sub> $( \xi _ { t } ( L ) ) _ { t \geq 1 }$ i<sub>s</sub> <sub>a</sub> <sub>mar</sub>ti<sub>nga</sub>l<sub>e</sub> dif<sub>erence</sub> <sub>sequence</sub> <sub>w</sub>ith <sub>respec</sub>t t<sub>o</sub> $( \mathcal { F } _ { t } ) _ { t \geq 0 }$ <sub>.</sub> N<sub>ow</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
M _ { t } ( L ) : = \sum _ { s = 1 } ^ { t } \frac { \xi _ { s } ( L ) } { \operatorname* { m a x } \{ N _ { s } ( L ) , 1 \} } .
$$

Th<sub>en</sub> $( M _ { t } ( L ) ) _ { t \geq 1 }$ i<sub>s</sub> <sub>a</sub> <sub>mar</sub>ti<sub>nga</sub>l<sub>e,</sub> <sub>an</sub>d

$$
\mathbb { E } \big [ \{ M _ { t } ( L ) - M _ { t - 1 } ( L ) \} ^ { 2 } \big | \mathcal { F } _ { t - 1 } \big ] \leq \frac { \mathbb { 1 } \{ L _ { t } = L \} } { \operatorname* { m a x } \{ N _ { t } ( L ) , 1 \} ^ { 2 } } .
$$

H<sub>ence</sub>

$$
\operatorname* { s u p } _ { t \geq 1 } \mathbb { E } [ M _ { t } ( L ) ^ { 2 } ] \leq \mathbb { E } \left[ \sum _ { s = 1 } ^ { \infty } \frac { \mathbb { 1 } \left\{ L _ { s } = L \right\} } { \operatorname* { m a x } \{ N _ { s } ( L ) , 1 \} ^ { 2 } } \right] \leq \sum _ { j = 1 } ^ { \infty } \frac { 1 } { j ^ { 2 } } < \infty .
$$

B<sub>y</sub> th<sub>e</sub> <sub>square-</sub>i<sub>n</sub>t<sub>egra</sub>bl<sub>e</sub> <sub>mar</sub>ti<sub>nga</sub>l<sub>e</sub> <sub>convergence</sub> th<sub>eorem,</sub> $M _ { t } ( L )$ conver<sub>g</sub>es almost surel<sub>y</sub>. On Ω, $N _ { t } ( L ) \to \infty$ <sub>so</sub> K<sub>ronec</sub>k<sub>er</sub>’<sub>s</sub> l<sub>emma</sub> <sub>g</sub>i<sub>ves</sub>

$$
\frac { 1 } { \operatorname* { m a x } \{ N _ { t } ( L ) , 1 \} } \sum _ { s = 1 } ^ { t } \xi _ { s } ( L ) \to 0 \qquad \mathrm { a . s . ~ o n } \Omega .
$$

Th<sub>ere</sub>f<sub>ore,</sub> <sub>on</sub> $\Omega ,$

$$
\frac { 1 } { N _ { t } ( L ) } \sum _ { s = 1 } ^ { t } \mathbb { 1 } \{ L _ { s } = L \} \widehat { X } _ { s } \to \mathbb { P } ( X _ { L } = + 1 \mid \theta ) \qquad \mathrm { a . s . }
$$

Th<sub>e</sub> l<sub>e</sub>ft<sub>-</sub>h<sub>an</sub>d <sub>s</sub>id<sub>e</sub> i<sub>s</sub> $\mathcal { H } _ { \infty }$ -measurable, so on Ω the limit ${ \overset { \triangledown } { \cdot } } ( X _ { L } = + 1 \mid \theta )$ i<sub>s</sub> $\mathcal { H } _ { \infty }$ <sub>-measura</sub>bl<sub>e.</sub> Si<sub>nce</sub> th<sub>e</sub> li<sub>n</sub>k i<sub>s</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> i<sub>ncreas</sub>i<sub>ng,</sub> $L ^ { \top } \pmb \theta$ i<sub>s a</sub>l<sub>so</sub> $\mathcal { H } _ { \infty }$ -measurable on Ω. Since this holds for ever<sub>y</sub> $L \in { \mathcal { E } }$ <sub>,</sub> th<sub>e same</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> <sup>ever</sup>y $\pmb { \nu } \in M = \operatorname { s p a n } ( \mathcal { E } )$

Fi<sub>x</sub> $\nu \in M$ <sub>.</sub> Si<sub>nce</sub> $\pmb { \nu } ^ { \top } \pmb { \theta } \in L ^ { 2 }$ <sub>,</sub> D<sub>oo</sub>b’<sub>s</sub> <sub>mar</sub>ti<sub>nga</sub>l<sub>e</sub> <sub>convergence</sub> th<sub>eorem</sub> <sub>g</sub>i<sub>ves</sub>

$$
\mathbb { E } [ { \pmb { \nu } } ^ { \top } { \pmb { \theta } } | \mathcal { H } _ { t } ] \longrightarrow \mathbb { E } [ { \pmb { \nu } } ^ { \top } { \pmb { \theta } } | \mathcal { H } _ { \infty } ] \qquad \mathrm { a . s . , }
$$

<sub>an</sub>d

$$
\mathbb { E } [ ( \pmb { \nu } ^ { \top } \pmb { \theta } ) ^ { 2 } | \mathcal { H } _ { t } ]  \mathbb { E } [ ( \pmb { \nu } ^ { \top } \pmb { \theta } ) ^ { 2 } | \mathcal { H } _ { \infty } ] \qquad \mathrm { a . s . }
$$

Si<sub>nce</sub> $\Omega \in \mathcal H _ { \infty }$ <sub>an</sub>d b<sub>o</sub>th $\pmb { \nu } ^ { \top } \pmb { \theta }$ <sub>an</sub>d $( \pmb { \nu } ^ { \top } \pmb \theta ) ^ { 2 }$ are $\mathcal { H } _ { \infty }$ -measurable on Ω, we have

$$
\begin{array} { r } { \mathbb { E } \big [ \pmb { \nu } ^ { \top } \pmb { \theta } | \mathcal { H } _ { \infty } \big ] = \pmb { \nu } ^ { \top } \pmb { \theta } , \qquad \mathbb { E } \big [ ( \pmb { \nu } ^ { \top } \pmb { \theta } ) ^ { 2 } | \mathcal { H } _ { \infty } \big ] = \big ( \pmb { \nu } ^ { \top } \pmb { \theta } \big ) ^ { 2 } } \end{array}
$$

<sub>a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y</sub> <sub>on</sub> $\Omega .$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore,</sub>

$$
\boldsymbol { \nu } ^ { \intercal } \pmb { \mu } _ { t - 1 } ^ { \ast } = \mathbb { E } [ \pmb { \nu } ^ { \intercal } \pmb { \theta } | \mathcal { H } _ { t } ]  \pmb { \nu } ^ { \intercal } \pmb { \theta } \qquad \mathrm { a . s . } \ \mathrm { o n } \ \Omega ,
$$

<sub>an</sub>d

$$
\nu ^ { \top } \Sigma _ { t - 1 } ^ { * } \nu = \mathbb { V } ( \nu ^ { \top } \pmb \theta | \mathcal { H } _ { t } )  0 \qquad \mathrm { a . s . ~ o n } \Omega .
$$

W<sub>e</sub> <sub>nex</sub>t <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t if <sub>samp</sub>li<sub>ng</sub> <sub>even</sub>t<sub>ua</sub>ll<sub>y</sub> <sub>rema</sub>i<sub>ns</sub> i<sub>n</sub> thi<sub>s</sub> <sub>recurren</sub>t <sub>span,</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> <sub>canno</sub>t <sub>go</sub> t<sub>o</sub> <sub>zero</sub> i<sub>n</sub> di<sub>rec</sub>ti<sub>ons ou</sub>t<sub>s</sub>id<sub>e</sub> it<sub>.</sub>

LEMMA D.2. Fix a deterministic set $\varepsilon \subseteq \mathcal { D }$ , let $M : = \operatorname { s p a n } ( \mathcal { E } )$ , and let $R _ { M }$ and � be the orthogonal projectors onto � and $M ^ { \perp }$ . Suppose that on an event $\Omega .$

(i) there exists an almost-surelyfinite round $T _ { 0 }$ such that every sampled direction after round $T _ { 0 }$ belongs to E, and

(ii) �<sub>�</sub> ${ \pmb \mu } _ { t - 1 } ^ { * } \to R _ { M } { \pmb \theta }$ and $\| R _ { M } \Sigma _ { t - 1 } ^ { * } R _ { M } \| _ { \mathrm { o p } } \to 0$ almost surely on $\Omega .$

Then, almost surely on $\Omega ,$ there exists $\eta > 0$ such that, for all suficiently large $t ,$

$$
Q \Sigma _ { t - 1 } ^ { * } Q \succeq \eta Q .
$$

Proof. Work on Ω. We can decompose $\mathbb { R } ^ { I } = M \oplus M ^ { \perp }$ , <sup>so</sup> <sup>ever</sup>y $\pmb { \theta } \in \mathbb { R } ^ { I }$ <sub>can</sub> b<sub>e</sub> <sub>wr</sub>itt<sub>en</sub> <sub>un</sub>i<sub>que</sub>l<sub>y</sub> <sub>as</sub>

$$
{ \pmb \theta } = { \pmb u } + { \pmb \nu } , \qquad { \pmb u } \in { \cal M } , \quad { \pmb \nu } \in { \cal M } ^ { \perp } ,
$$

i<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar</sub> $\pmb { u } = R _ { M } \pmb { \theta }$ <sub>an</sub>d $\pmb { \nu } = Q \pmb \theta .$ <sub>.</sub> L<sub>e</sub>t $f _ { 0 } ( { \pmb u } , { \pmb \nu } )$ denote the posterior density for � after round $T _ { 0 }$ <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>a</sub>t $\pmb \theta = \pmb u + \pmb \nu$ <sub>.</sub> Si<sub>nce</sub> $T _ { 0 } < \infty$ <sub>,</sub> thi<sub>s</sub> d<sub>ens</sub>it<sub>y</sub> i<sub>s</sub> <sub>o</sub>bt<sub>a</sub>i<sub>ne</sub>d f<sub>rom</sub> th<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> <sub>pr</sub>i<sub>or</sub> d<sub>ens</sub>it<sub>y</sub> b<sub>y</sub> <sub>mu</sub>lti<sub>p</sub>l<sub>y</sub>i<sub>ng</sub> fi<sub>n</sub>it<sub>e</sub>l<sub>y</sub> <sub>many</sub> <sub>one-o</sub>b<sub>serva</sub>ti<sub>on</sub> lik<sub>e</sub>lih<sub>oo</sub>d f<sub>ac</sub>t<sub>ors an</sub>d <sub>norma</sub>li<sub>z</sub>i<sub>ng.</sub> Th<sub>us</sub> $f _ { 0 }$ i<sub>s con</sub>ti<sub>nuous an</sub>d <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y pos</sub>iti<sub>ve.</sub> Si<sub>nce eac</sub>h lik<sub>e</sub>lih<sub>oo</sub>d f<sub>ac</sub>t<sub>or</sub> b<sub>e</sub>l<sub>on s</sub> t<sub>o</sub> $( 0 , 1 )$

$$
f _ { 0 } ( { \pmb u } , { \pmb v } ) \le Z _ { T _ { 0 } } ^ { - 1 } \pi _ { 0 } ( { \pmb u } , { \pmb v } ) ,
$$

<sub>w</sub>h<sub>ere</sub> $\pi _ { 0 }$ i<sub>s</sub> th<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> <sub>pr</sub>i<sub>or</sub> d<sub>ens</sub>it<sub>y</sub> <sub>an</sub>d $Z _ { T _ { 0 } } > 0$ i<sub>s</sub> th<sub>e norma</sub>li<sub>z</sub>i<sub>ng cons</sub>t<sub>an</sub>t<sub>.</sub>

N<sub>ow</sub> fi<sub>x</sub> $t > T _ { 0 }$ <sub>an</sub>d l<sub>e</sub>t $f _ { t }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> d<sub>ens</sub>it<sub>y</sub> <sub>g</sub>i<sub>ven</sub> $\mathcal { H } _ { t }$ <sub>.</sub> E<sub>very</sub> <sub>o</sub>b<sub>serva</sub>ti<sub>on</sub> <sub>co</sub>ll<sub>ec</sub>t<sub>e</sub>d <sub>a</sub>ft<sub>er</sub> <sub>roun</sub>d $T _ { 0 }$ i<sub>s</sub> t<sub>a</sub>k<sub>en a</sub>l<sub>ong some</sub> di<sub>rec</sub>ti<sub>on</sub> $L \in \mathcal { E } \subseteq M$ . The corresponding likelihood factor depends on � only through $L ^ { \top } \pmb \theta . \operatorname { I f } \ \pmb \theta = \pmb u + \pmb \nu$ <sub>w</sub>ith $\pmb { u } \in M$ <sub>an</sub>d $\pmb { \nu } \in M ^ { \bot }$ <sub>,</sub> th<sub>en</sub>

$$
\boldsymbol L ^ { \top } \boldsymbol \theta = \boldsymbol L ^ { \top } ( \boldsymbol u + \boldsymbol \nu ) = \boldsymbol L ^ { \top } \boldsymbol u ,
$$

b<sub>ecause</sub> $L \in M$ <sub>an</sub>d $\nu \perp M$ <sub>.</sub> It f<sub>o</sub>ll<sub>ows</sub> th<sub>a</sub>t

$$
f _ { t } ( \pmb { u } , \pmb { \nu } ) = \frac { f _ { 0 } ( \pmb { u } , \pmb { \nu } ) \Psi _ { t } ( \pmb { u } ) } { Z _ { t } }
$$

f<sub>or</sub> <sub>some</sub> <sub>nonnega</sub>ti<sub>ve</sub> f<sub>unc</sub>ti<sub>on</sub> $\Psi _ { t }$ <sub>on</sub> � <sub>an</sub>d <sub>some</sub> <sub>norma</sub>li<sub>z</sub>i<sub>ng</sub> <sub>cons</sub>t<sub>an</sub>t $Z _ { t } > 0$ . Therefore, for fixed �, the <sub>con</sub>diti<sub>ona</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> $M ^ { \perp }$ <sub>componen</sub>t d<sub>oes</sub> <sub>no</sub>t <sub>c</sub>h<sub>ange</sub> <sub>a</sub>ft<sub>er</sub> <sub>roun</sub>d $T _ { 0 }$

D<sub>e</sub>fi<sub>ne</sub>

$$
c ( \pmb { u } ) : = \int _ { M ^ { \bot } } f _ { 0 } ( \pmb { u } , \pmb { \nu } ) d \pmb { \nu } ,
$$

<sub>an</sub>d l<sub>e</sub>t $C ( \pmb { u } )$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub> <sub>covar</sub>i<sub>ance</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>con</sub>diti<sub>ona</sub>l d<sub>ens</sub>it<sub>y</sub> $f _ { 0 } ( { \pmb u } , \cdot ) / c ( { \pmb u } )$ on $M ^ { \perp }$ <sub>.</sub> Ch<sub>oose</sub> <sub>a</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>c</sub>l<sub>ose</sub>d <sub>ne</sub>i<sub>g</sub>hb<sub>or</sub>h<sub>oo</sub>d $O \subseteq M$ <sub>o</sub>f $R _ { M } \pmb \theta$ <sub>.</sub> Si<sub>nce</sub> $f _ { 0 }$ i<sub>s</sub> <sub>con</sub>ti<sub>nuous</sub> <sub>an</sub>d i<sub>s</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>a</sub>b<sub>ove</sub> b<sub>y</sub> <sub>a</sub> <sub>cons</sub>t<sub>an</sub>t <sub>mu</sub>lti<sub>p</sub>l<sub>e o</sub>f th<sub>e</sub> G<sub>auss</sub>i<sub>an pr</sub>i<sub>or</sub> d<sub>ens</sub>it<sub>y,</sub> d<sub>om</sub>i<sub>na</sub>t<sub>e</sub>d <sub>convergence</sub> i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t $C ( \pmb { u } )$ i<sub>s con</sub>ti<sub>nuous on</sub> $o$

<sup>F</sup>or ever<sub>y</sub> $\pmb { u } \in O$ , the conditional distribution of � has a strictly positive density on all of $M ^ { \perp }$ <sub>.</sub> H<sub>ence,</sub> it<sub>s</sub> <sub>covar</sub>i<sub>ance ma</sub>t<sub>r</sub>i<sub>x</sub> $C ( \pmb { u } )$ i<sub>s pos</sub>iti<sub>ve</sub> d<sub>e</sub>fi<sub>n</sub>it<sub>e on</sub> $M ^ { \perp }$ <sub>.</sub> Si<sub>nce</sub> � i<sub>s compac</sub>t <sub>an</sub>d $C ( \pmb { u } )$ i<sub>s con</sub>ti<sub>nuous,</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s</sub> $\eta _ { 0 } > 0$ <sub>suc</sub>h th<sub>a</sub>t

$$
\begin{array} { r } { C ( \pmb { u } ) \succeq \eta _ { 0 } \mathrm { I d } _ { M ^ { \bot } } \qquad ( \pmb { u } \in O ) . } \end{array}
$$

It remains to show that, under the round-� posterior, the �-coordinate belongs to $o$ <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> th<sub>a</sub>t <sub>approac</sub>h<sub>es</sub> 1<sub>.</sub> Ch<sub>oose</sub> $\epsilon > 0$ <sub>suc</sub>h th<sub>a</sub>t th<sub>e</sub> <sub>c</sub>l<sub>ose</sub>d b<sub>a</sub>ll <sub>o</sub>f <sub>ra</sub>di<sub>us</sub> $2 \epsilon$ <sub>aroun</sub>d $R _ { M } \pmb \theta$ i<sub>s</sub> <sub>con</sub>t<sub>a</sub>i<sub>ne</sub>d i<sub>n</sub> $O .$ <sub>.</sub> B<sub>y</sub> <sub>con</sub>diti<sub>on</sub> (ii), for all suficientl<sub>y</sub> lar<sub>g</sub>e �,

$$
\| R _ { M } \pmb { \mu } _ { t - 1 } ^ { * } - R _ { M } \pmb { \theta } \| _ { 2 } < \epsilon .
$$

F<sub>or</sub> <sub>suc</sub>h �<sub>,</sub> if $\pmb { u } \notin O$ <sub>,</sub> th<sub>en</sub>

$$
\| \pmb { u } - \pmb { R } _ { M } \pmb { \mu } _ { t - 1 } ^ { * } \| _ { 2 } \geq \epsilon .
$$

Th<sub>us,</sub> b<sub>y</sub> M<sub>ar</sub>k<sub>ov</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y,</sub>

$$
\mathbb { P } ( \pmb { u } \notin \cal { O } \mid \mathcal { H } _ { t } ) \leq \frac { \mathrm { t r } ( R _ { M } \Sigma _ { t - 1 } ^ { * } R _ { M } ) } { \epsilon ^ { 2 } }  0 .
$$

Fi<sub>na</sub>ll<sub>y,</sub> th<sub>e</sub> l<sub>aw</sub> <sub>o</sub>f t<sub>o</sub>t<sub>a</sub>l <sub>covar</sub>i<sub>ance</sub> <sub>g</sub>i<sub>ves</sub>

$$
Q \Sigma _ { t - 1 } ^ { * } Q \succeq \mathbb { E } [ C ( \pmb { u } ) \mid \mathcal { H } _ { t } ] \succeq \eta _ { 0 } \mathbb { P } ( \pmb { u } \in O \mid \mathcal { H } _ { t } ) Q .
$$

Th<sub>e</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>on</sub> th<sub>e</sub> <sub>r</sub>i<sub>g</sub>ht <sub>converges</sub> t<sub>o</sub> 1<sub>,</sub> <sub>so</sub> th<sub>e</sub> <sub>c</sub>l<sub>a</sub>i<sub>m</sub> h<sub>o</sub>ld<sub>s</sub> <sub>even</sub>t<sub>ua</sub>ll<sub>y</sub> <sub>w</sub>ith $\eta = \eta _ { 0 } / 2$ □ N<sub>ex</sub>t <sub>we ge</sub>t <sub>a un</sub>if<sub>orm upper</sub> b<sub>oun</sub>d <sub>on</sub> th<sub>e pos</sub>t<sub>er</sub>i<sub>or covar</sub>i<sub>ance.</sub>

LEMMA D.3 (Brascamp–Lieb). For every �,

$$
\Sigma _ { t - 1 } ^ { * } \preceq \Sigma _ { 0 } .
$$

Consequently, under Assumption $2 ( i ) ,$ , eventually almost surely, $\Sigma _ { t - 1 } \preceq \chi _ { + } \Sigma _ { 0 }$ . In particular, there exists $\bar { \sigma } < \infty$ such that

$$
\sigma _ { t - 1 , i } \le \bar { \sigma } , \qquad i = 1 , \dots , I ,
$$

eventually almost surely.

Proof. Fix �. The likelihood factors are log-concave by Assumption 1, and remain log-concave after <sub>averag</sub>i<sub>ng</sub> <sub>over</sub> th<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>or</sub> <sub>no</sub>i<sub>se.</sub> Th<sub>e</sub> <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> i<sub>s</sub> th<sub>e</sub> G<sub>auss</sub>i<sub>an</sub> <sub>pr</sub>i<sub>or</sub> <sub>mu</sub>lti<sub>p</sub>li<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> <sub>pro</sub>d<sub>uc</sub>t of these factors. B<sub>y</sub> the Brascam<sub>p</sub>–Lieb covariance inequalit<sub>y</sub> (Brascam<sub>p</sub> and Lieb 1975), $\Sigma _ { t - 1 } ^ { * } \preceq \Sigma _ { 0 }$ . <sup>B</sup><sub>y</sub> Assum<sub>p</sub>tion 2(i), eventuall<sub>y</sub> almost surel<sub>y</sub> $\Sigma _ { t - 1 } \preceq \chi _ { + } \Sigma _ { t - 1 } ^ { * } \preceq \chi _ { + } \Sigma _ { 0 }$

Si<sub>nce</sub> $\mathbf { } _ { \pmb { b } _ { t } }$ h<sub>as</sub> <sub>nonnega</sub>ti<sub>ve</sub> <sub>en</sub>t<sub>r</sub>i<sub>es</sub> <sub>summ</sub>i<sub>ng</sub> t<sub>o</sub> 1<sub>,</sub>

$$
\| \pmb { g } _ { t , i } \| _ { 2 } \leq \| \pmb { e } _ { i } \| _ { 2 } + \| \pmb { b } _ { t } \| _ { 2 } \leq 2 .
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
\begin{array} { r } { \sigma _ { t - 1 , i } ^ { 2 } = \pmb { g } _ { t , i } ^ { \top } \sum _ { t - 1 } \pmb { g } _ { t , i } \leq 4 \chi _ { + } \Vert \boldsymbol { \Sigma } _ { 0 } \Vert _ { \mathrm { o p } } . } \end{array}
$$

W<sub>e</sub> <sub>nex</sub>t t<sub>rans</sub>f<sub>er</sub> th<sub>e</sub> <sub>exac</sub>t<sub>-pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>var</sub>i<sub>ance</sub> <sub>conc</sub>l<sub>us</sub>i<sub>ons</sub> <sub>a</sub>b<sub>ove</sub> t<sub>o</sub> th<sub>e</sub> <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub> <sub>covar</sub>i<sub>ance</sub> <sub>use</sub>d b<sub>y</sub> SA<sub>.</sub>

LEMMA D.4. Fix a deterministic set $\varepsilon \subseteq \mathcal { D }$ , let $M : = \operatorname { s p a n } ( \mathcal { E } )$ , and let $R _ { M }$ and � be the orthogonal projectors onto � and $M ^ { \perp }$ . Suppose that on an event Ω,

$$
\| R _ { M } \Sigma _ { t - 1 } ^ { * } R _ { M } \| _ { \mathrm { o p } } \to 0
$$

and, for some $\eta > 0$ and all suficiently large �,

$$
Q \Sigma _ { t - 1 } ^ { * } Q \succeq \eta Q .
$$

Then, under Assumption 2(i) and Lemma $D . 3 ,$ , almost surely on Ω,

$$
\begin{array} { r } { \| R _ { M } \Sigma _ { t - 1 } R _ { M } \| _ { \mathrm { o p } } \to 0 , \qquad \| Q \Sigma _ { t - 1 } R _ { M } \| _ { \mathrm { o p } } + \| R _ { M } \Sigma _ { t - 1 } Q \| _ { \mathrm { o p } } \to 0 , } \end{array}
$$

and, for all suficiently large �,

$$
Q \Sigma _ { t - 1 } Q \succeq \chi _ { - } \eta Q .
$$

Proof. From Assumption 2(i), $0 \preceq R _ { M } \Sigma _ { t - 1 } R _ { M } \preceq \chi _ { + } R _ { M } \Sigma _ { t - 1 } ^ { * } R _ { M } , \ s \mathrm { o } \ \| R _ { M } \Sigma _ { t - 1 } R _ { M } \| _ { \mathrm { o p } } \to 0$ <sub>.</sub> F<sub>or</sub> th<sub>e</sub> <sub>o</sub>th<sub>er</sub> t<sub>erms,</sub> l<sub>e</sub>t $\pmb { x } \in M ^ { \bot }$ <sub>an</sub>d ${ \pmb y } \in M$ b<sub>e un</sub>it <sub>vec</sub>t<sub>ors.</sub> Si<sub>nce</sub> $\Sigma _ { t - 1 }$ i<sub>s</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>sem</sub>id<sub>e</sub>fi<sub>n</sub>it<sub>e,</sub>

$$
\begin{array} { r } { | \pmb { x } ^ { \top } \Sigma _ { t - 1 } \pmb { y } | ^ { 2 } \leq ( \pmb { x } ^ { \top } \Sigma _ { t - 1 } \pmb { x } ) ( \pmb { y } ^ { \top } \Sigma _ { t - 1 } \pmb { y } ) \leq \| \mathscr { Q } \Sigma _ { t - 1 } \mathscr { Q } \| _ { \mathrm { o p } } \| R _ { M } \Sigma _ { t - 1 } R _ { M } \| _ { \mathrm { o p } } . } \end{array}
$$

B<sub>y</sub> L<sub>emma</sub> D<sub>.</sub>3<sub>,</sub> th<sub>e</sub> fi<sub>rs</sub>t f<sub>ac</sub>t<sub>or</sub> i<sub>s</sub> <sub>even</sub>t<sub>ua</sub>ll<sub>y</sub> b<sub>oun</sub>d<sub>e</sub>d<sub>,</sub> <sub>w</sub>hil<sub>e</sub> th<sub>e</sub> <sub>secon</sub>d <sub>converges</sub> t<sub>o</sub> 0<sub>.</sub> T<sub>a</sub>ki<sub>ng</sub> th<sub>e</sub> <sub>supremum</sub> over unit � and � gives

$$
\begin{array} { r } { \| Q \Sigma _ { t - 1 } R _ { M } \| _ { \mathrm { o p } } + \| R _ { M } \Sigma _ { t - 1 } Q \| _ { \mathrm { o p } } \to 0 . } \end{array}
$$

Finall<sub>y</sub>, Assum<sub>p</sub>tion 2(i) <sub>g</sub>ives

$$
Q \Sigma _ { t - 1 } Q \succeq \chi _ { - } Q \Sigma _ { t - 1 } ^ { * } Q \succeq \chi _ { - } \eta Q .
$$

T<sub>o</sub> <sub>con</sub>t<sub>ro</sub>l th<sub>e</sub> <sub>marg</sub>i<sub>na</sub>l <sub>scores,</sub> <sub>we</sub> <sub>a</sub>l<sub>so</sub> <sub>nee</sub>d b<sub>oun</sub>d<sub>s</sub> <sub>on</sub> th<sub>e</sub> <sub>a</sub>fi<sub>ne</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>an</sub>d th<sub>e</sub> <sub>assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>we</sub>i<sub>g</sub>ht<sub>s.</sub>

LEMMA D.5. Let Ω be an event on which

$$
\operatorname* { s u p } _ { t \geq 1 } \| \pmb { \mu } _ { t - 1 } \| _ { \infty } < \infty , \qquad \operatorname* { m a x } _ { 1 \leq i \leq I } \sigma _ { t - 1 , i } \leq \bar { \sigma }
$$

eventuallyfor some finite �¯ . Then, almost surely on $\Omega ,$ there exists $Q _ { 0 } < \infty$ such that $| q _ { t } | \leq Q _ { 0 }$ eventually. In addition, $i f ( i _ { t } ) _ { t \geq 1 }$ is any sequence in $\{ 1 , \ldots , I \}$ such that lim in $\mathrm { f } _ { t  \infty } \sigma _ { t - 1 , i _ { t } } > 0$ on $\Omega ,$ then there exists $w _ { * } > 0$ such that $w _ { t - 1 , i _ { t } } \geq w ,$ <sub>∗</sub> eventually on $\Omega .$

Proof. First consider rounds in case (ii) of Theorem 1. Define

$$
B : = \operatorname* { s u p } _ { t \geq 1 } \| \mu _ { t - 1 } \| _ { \infty } < \infty , \qquad F _ { t } ( q ) : = \sum _ { i = 1 } ^ { I } \Phi \bigg ( \frac { \mu _ { t - 1 , i } - q } { \sigma _ { t - 1 , i } } \bigg ) .
$$

R<sub>eca</sub>ll th<sub>a</sub>t $q _ { t }$ i<sub>s</sub> d<sub>e</sub>fi<sub>ne</sub>d t<sub>o sa</sub>ti<sub>s</sub>f<sub>y</sub> $F _ { t } ( q _ { t } ) = k$ <sub>.</sub> F<sub>or</sub> <sub>eac</sub>h $t ,$ t<sup>h</sup>e ma<sub>p</sub> $F _ { t }$ i<sub>s</sub> <sub>con</sub>ti<sub>nuous</sub> <sub>an</sub>d <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> d<sub>ecreas</sub>i<sub>ng,</sub> <sub>w</sub>ith $F _ { t } ( q ) \to I$ as $q \to - \infty$ <sub>an</sub>d $F _ { t } ( q ) \to 0$ as $q \to + \infty$ <sub>.</sub> Ch<sub>oose</sub> $Q _ { 0 } > B$ <sub>suc</sub>h th<sub>a</sub>t

$$
I \Phi \left( \frac { B - Q _ { 0 } } { \bar { \sigma } } \right) < k < I \Phi \left( \frac { Q _ { 0 } - B } { \bar { \sigma } } \right) .
$$

F<sub>or</sub> <sub>a</sub>ll <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> l<sub>arge</sub> $t ,$

$$
F _ { t } ( Q _ { 0 } ) \leq I \Phi \mathopen { } \mathclose \bgroup \left( \frac { B - Q _ { 0 } } { \bar { \sigma } } \aftergroup \egroup \right) < k ,
$$

<sub>an</sub>d <sub>s</sub>i<sub>m</sub>il<sub>ar</sub>l<sub>y,</sub>

$$
F _ { t } ( - Q _ { 0 } ) \geq I \Phi \Bigg ( \frac { Q _ { 0 } - B } { \bar { \sigma } } \Bigg ) > k .
$$

Si<sub>nce</sub> $F _ { t }$ i<sub>s</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> d<sub>ecreas</sub>i<sub>ng</sub> <sub>an</sub>d $F _ { t } ( q _ { t } ) = k$ <sub>,</sub> it f<sub>o</sub>ll<sub>ows</sub> th<sub>a</sub>t

$$
- Q _ { 0 } < q _ { t } < Q _ { 0 }
$$

for all suficientl<sub>y</sub> lar<sub>g</sub>e �. On a round in case (i) of Theorem 1, $q _ { t } = \mu _ { t - 1 , j }$ f<sub>or</sub> <sub>some</sub> $j ,$ so $\vert q _ { t } \vert \le B < Q _ { \mathrm { 0 } }$ <sub>.</sub> Th<sub>us,</sub> $\vert q _ { t } \vert < Q _ { 0 }$ <sub>even</sub>t<sub>ua</sub>ll<sub>y</sub> i<sub>n</sub> <sub>e</sub>ith<sub>er</sub> <sub>case.</sub>

N<sub>ow</sub> <sub>suppose</sub> li<sub>m</sub> i<sub>n</sub> $\dot { \cdot } _ { t  \infty } \sigma _ { t - 1 , i _ { t } } > 0$ <sub>.</sub> Th<sub>en</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s</sub> $\sigma > 0$ <sub>suc</sub>h th<sub>a</sub>t

$$
\underline { { \sigma } } \le \sigma _ { t - 1 , i _ { t } } \le \bar { \sigma }
$$

<sub>even</sub>t<sub>ua</sub>ll<sub>y.</sub> T<sub>oge</sub>th<sub>er w</sub>ith $| \mu _ { t - 1 , i _ { t } } | \leq B$ <sub>an</sub>d $| q _ { t } | \leq Q _ { 0 }$ <sub>,</sub> thi<sub>s</sub> <sub>g</sub>i<sub>ves</sub>

$$
\left| \zeta _ { t - 1 , i _ { t } } \right| \le \frac { B + Q _ { 0 } } { \underline { { \sigma } } }
$$

<sub>even</sub>t<sub>ua</sub>ll<sub>y.</sub> Th<sub>ere</sub>f<sub>ore,</sub>

$$
w _ { t - 1 , i _ { t } } = \frac { \phi ( z _ { t - 1 , i _ { t } } ) } { \sigma _ { t - 1 , i _ { t } } } \geq \frac { \phi ( ( B + Q _ { 0 } ) / \underline { { \sigma } } ) } { \bar { \sigma } } = : w _ { * } > 0
$$

<sub>even</sub>t<sub>ua</sub>ll<sub>y.</sub> □

W<sub>e can now s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e score o</sub>f <sub>any</sub> di<sub>rec</sub>ti<sub>on w</sub>h<sub>ose var</sub>i<sub>ance van</sub>i<sub>s</sub>h<sub>es mus</sub>t it<sub>se</sub>lf <sub>van</sub>i<sub>s</sub>h<sub>.</sub>

LEMMA D.6. Let $L \in \mathcal { D }$ be a direction such that

$$
L ^ { \top } \Sigma _ { t - 1 } L  0 .
$$

Then

$$
G _ { t } ( L ) \to 0 .
$$

Proof. For each � with $\sigma _ { t - 1 , i } > 0$ <sub>,</sub> C<sub>auc</sub>h<sub>y–</sub>S<sub>c</sub>h<sub>wa</sub>r<sub>z</sub> <sub>g</sub>i<sub>ves,</sub>

$$
\begin{array} { r } { \left( \pmb { g } _ { t , i } ^ { \top } \Sigma _ { t - 1 } L \right) ^ { 2 } \leq \left( \pmb { g } _ { t , i } ^ { \top } \Sigma _ { t - 1 } \pmb { g } _ { t , i } \right) \left( L ^ { \top } \Sigma _ { t - 1 } L \right) = \sigma _ { t - 1 , i } ^ { 2 } L ^ { \top } \Sigma _ { t - 1 } L . } \end{array}
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
w _ { t - 1 , i } \left( g _ { t , i } ^ { { \top } } \Sigma _ { t - 1 } L \right) ^ { 2 } \leq \phi ( 0 ) \sigma _ { t - 1 , i } L ^ { { \top } } \Sigma _ { t - 1 } L \leq \phi ( 0 ) \bar { \sigma } L ^ { { \top } } \Sigma _ { t - 1 } L
$$

<sub>eve</sub>nt<sub>ua</sub>ll<sub>y,</sub> b<sub>y</sub> L<sub>e</sub>mm<sub>a</sub> D<sub>.</sub>3<sub>.</sub> S<sub>u</sub>mmin<sub>g</sub> <sub>ove</sub>r � <sub>y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\sum _ { i : \sigma _ { t - 1 , i } > 0 } w _ { t - 1 , i } \left( \pmb { g } _ { t , i } ^ { \top } \Sigma _ { t - 1 } L \right) ^ { 2 } \leq I \phi ( 0 ) \bar { \sigma } L ^ { \top } \Sigma _ { t - 1 } L .
$$

Al<sub>so,</sub> C<sub>auc</sub>h<sub>y–</sub>S<sub>c</sub>h<sub>warz</sub> <sub>g</sub>i<sub>ves</sub> $\phi ( z ) ^ { 2 } \leq \Phi ( z ) \Phi ( - z )$ , so $0 \le \delta _ { t } ( L ) < 1$ <sub>an</sub>d $\delta _ { t } ( L ) / ( L ^ { \top } \Sigma _ { t - 1 } L ) \leq \beta _ { L } ^ { 2 }$ <sub>.</sub> Si<sub>nce</sub> $\phi ( z / \sqrt { 1 - s } ) \leq \phi ( z )$ f<sub>or</sub> $0 \leq s < 1$ <sub>an</sub>d $1 - { \sqrt { 1 - s } } \leq s ,$ inte<sub>g</sub>ratin<sub>g</sub> in (D.1) and substitutin<sub>g</sub> the bounds above <sub>g</sub><sup>i</sup>ves

$$
0 \leq G _ { t } ( L ) \leq \frac { \delta _ { t } ( L ) } { r _ { L } L ^ { \top } \Sigma _ { t - 1 } L } \sum _ { i : \sigma _ { t - 1 , i } > 0 } w _ { t - 1 , i } ( g _ { t , i } ^ { \top } \Sigma _ { t - 1 } L ) ^ { 2 } \leq \frac { I \phi ( 0 ) \bar { \sigma } \beta _ { L } ^ { 2 } } { r _ { L } } L ^ { \top } \Sigma _ { t - 1 } L  0 .
$$

W<sub>e nex</sub>t <sub>s</sub>h<sub>ow</sub> th<sub>e comp</sub>l<sub>emen</sub>t<sub>ary resu</sub>lt<sub>,</sub> th<sub>a</sub>t if th<sub>e recurren</sub>t <sub>span</sub> d<sub>oes no</sub>t <sub>con</sub>t<sub>a</sub>i<sub>n</sub> ${ \bf 1 ^ { \perp } }$ , some score must <sub>rema</sub>i<sub>n</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>away</sub> f<sub>rom</sub> <sub>zero.</sub>

LEMMA D.7. Fix a deterministic set $\varepsilon \subseteq \mathcal { D }$ , let $M : = \operatorname { s p a n } ( \mathcal { E } )$ , and let $R _ { M }$ and � be the orthogonal projectors onto � and $M ^ { \perp }$ . Suppose $1 ^ { \perp } \not \in M .$ . On any event $\Omega$ on which

(i) $Q \Sigma _ { t - 1 } Q \succeq c Q$ eventually for some $c > 0 ,$

(ii) $\| R _ { M } \Sigma _ { t - 1 } Q \| _ { \mathrm { o p } } \to 0 ,$ , and

(iii) sup<sub>�</sub> $\| \mu _ { t - 1 } \| _ { \infty } < \infty$ and $\Sigma _ { t - 1 } \preceq \overline { { \Sigma } }$ eventuallyfor somefixed positive semidefinite matrix ${ \overline { { \Sigma } } } ,$ we have

$$
\operatorname* { l i m i n f } _ { t \to \infty } \sum _ { L \in \mathcal { D } } G _ { t } ( L ) > 0 \qquad a . s . \ o n \ \Omega .
$$

Proof. Because the vectors ${ e _ { a } } - { e _ { b } }$ , � ≠ �, s<sub>p</sub>an ${ \bf 1 ^ { \perp } }$ <sub>,</sub> th<sub>e</sub> <sub>assump</sub>ti<sub>on</sub> $1 ^ { \perp } \nsubseteq$ � i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t th<sub>ere</sub> <sub>ex</sub>i<sub>s</sub>t $a \neq b$ <sub>suc</sub>h th<sub>a</sub>t $Q ( e _ { a } - e _ { b } ) \neq \mathbf { 0 }$ <sub>.</sub> S<sub>e</sub>t $\begin{array} { r } { \delta : = \frac 1 2 \| Q ( \pmb { e } _ { a } - \pmb { e } _ { b } ) \| _ { 2 } > 0 } \end{array}$ <sub>.</sub> Si<sub>nce</sub> $Q ( \pmb { e } _ { a } - \pmb { e } _ { b } ) = Q ( \pmb { g } _ { t , a } - \pmb { g } _ { t , b } )$ <sub>,</sub> f<sub>or</sub> <sub>every</sub> �<sub>,</sub> th<sub>ere</sub> <sub>ex</sub>i<sub>s</sub>t<sub>s</sub> $i _ { t } \in \{ a , b \}$ <sub>suc</sub>h th<sub>a</sub>t $\left\| Q g _ { t , i _ { t } } \right\| _ { 2 } \geq \delta$ <sub>.</sub> Usin<sub>g</sub> $\| \pmb { g } _ { t , i _ { t } } \| _ { 2 } \leq 2$ , condition (i), and condition (ii),

$$
\begin{array} { r l } & { \sigma _ { t - 1 , i _ { t } } ^ { 2 } = \pmb { g } _ { t , i _ { t } } ^ { \top } \Sigma _ { t - 1 } \pmb { g } _ { t , i _ { t } } } \\ & { \qquad \geq ( Q \pmb { g } _ { t , i _ { t } } ) ^ { \top } \Sigma _ { t - 1 } ( Q \pmb { g } _ { t , i _ { t } } ) - 2 \left| ( R _ { M } \pmb { g } _ { t , i _ { t } } ) ^ { \top } \Sigma _ { t - 1 } ( Q \pmb { g } _ { t , i _ { t } } ) \right| } \\ & { \qquad \geq c \left\| Q \pmb { g } _ { t , i _ { t } } \right\| _ { 2 } ^ { 2 } - 2 \| R _ { M } \Sigma _ { t - 1 } Q \| _ { \mathrm { o p } } \| R _ { M } \pmb { g } _ { t , i _ { t } } \| _ { 2 } \| Q \pmb { g } _ { t , i _ { t } } \| _ { 2 } } \\ & { \qquad \geq c \delta ^ { 2 } - 8 \| R _ { M } \Sigma _ { t - 1 } Q \| _ { \mathrm { o p } } . } \end{array}
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
\operatorname* { l i m } _ { t \to \infty } \operatorname* { i n f } _ { \sigma _ { t - 1 , i _ { t } } ^ { 2 } } \geq c \delta ^ { 2 } > 0 .
$$

B<sub>y</sub> L<sub>emma</sub> D<sub>.</sub>5<sub>,</sub> th<sub>ere</sub> <sub>ex</sub>i<sub>s</sub>t<sub>s</sub> $w _ { * } > 0$ <sub>suc</sub>h th<sub>a</sub>t <sub>even</sub>t<sub>ua</sub>ll<sub>y</sub> $w _ { t - 1 , i _ { t } } \geq w _ { * }$ <sub>.</sub> Th<sub>e</sub> <sub>same</sub> b<sub>oun</sub>d<sub>s</sub> <sub>a</sub>l<sub>so</sub> i<sub>mp</sub>l<sub>y</sub> th<sub>a</sub>t $\boldsymbol { z } _ { t - 1 , i _ { t } }$ i<sub>s</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>even</sub>t<sub>ua</sub>ll<sub>y.</sub>

Si<sub>nce</sub> $\mathcal { D }$ i<sub>s</sub> fi<sub>n</sub>it<sub>e an</sub>d it<sub>s span con</sub>t<sub>a</sub>i<sub>ns</sub> ${ \bf 1 ^ { \perp } }$ <sub>,</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s</sub> $\kappa > 0$ <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub> f<sub>or every</sub> $\boldsymbol { y } \in \mathbb { R } ^ { I }$

$$
\sum _ { L \in \mathcal { D } } ( L ^ { \top } \mathbf { y } ) ^ { 2 } \geq \kappa \| P \mathbf { y } \| _ { 2 } ^ { 2 } .
$$

A<sub>pp</sub>l<sub>y</sub>i<sub>ng</sub> thi<sub>s</sub> <sub>w</sub>ith $\begin{array} { r } { \mathbf { y } = \sum _ { t - 1 } \mathbf { g } _ { t , i _ { t } } } \end{array}$ <sub>,</sub> <sub>an</sub>d <sub>us</sub>i<sub>ng</sub> ${ \pmb g } _ { t , i _ { t } } \in { { \bf 1 } ^ { \perp } }$ <sub>an</sub>d $\| g _ { t , i _ { t } } \| _ { 2 } \leq 2$ , <sub>g</sub><sup>i</sup>ves

$$
\sum _ { L \in \mathcal { D } } \left( \pmb { g } _ { t , i _ { t } } ^ { \top } \Sigma _ { t - 1 } L \right) ^ { 2 } \geq \kappa \left\| P \Sigma _ { t - 1 } \pmb { g } _ { t , i _ { t } } \right\| _ { 2 } ^ { 2 } \geq \frac { \kappa } { 4 } \left( \pmb { g } _ { t , i _ { t } } ^ { \top } P \Sigma _ { t - 1 } \pmb { g } _ { t , i _ { t } } \right) ^ { 2 } = \frac { \kappa } { 4 } \sigma _ { t - 1 , i _ { t } } ^ { 4 } .
$$

F<sub>or</sub> $0 \leq s \leq 1 / 2$

$$
\frac { 1 } { \sqrt { 1 - s } } \phi \left( \frac { z } { \sqrt { 1 - s } } \right) \geq \phi ( z ) \exp \left( - \frac { z ^ { 2 } s } { 2 ( 1 - s ) } \right) \geq \phi ( z ) e ^ { - z ^ { 2 } / 2 } .
$$

Since the u<sub>pp</sub>er limit in (D.1) is less than 1, retainin<sub>g</sub> onl<sub>y</sub> the $i _ { t }$ <sub>summan</sub>d <sub>an</sub>d i<sub>n</sub>t<sub>egra</sub>ti<sub>ng over</sub> th<sub>e</sub> fi<sub>rs</sub>t h<sub>a</sub>lf <sub>o</sub>f it<sub>s</sub> i<sub>n</sub>t<sub>erva</sub>l <sub>g</sub>i<sub>ves</sub>

$$
G _ { t } ( L ) \geq \frac { \delta _ { t } ( L ) e ^ { - z _ { t - 1 , i _ { t } } ^ { 2 } / 2 } } { 4 r _ { L } L ^ { \top } \Sigma _ { t - 1 } L } w _ { t - 1 , i _ { t } } \left( g _ { t , i _ { t } } ^ { \top } \Sigma _ { t - 1 } L \right) ^ { 2 } .
$$

Th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on o</sub>f $\delta _ { t } ( L )$ <sub>g</sub><sup>i</sup>ves

$$
\frac { \delta _ { t } ( L ) } { L ^ { \top } \Sigma _ { t - 1 } L } = \frac { \beta _ { L } ^ { 2 } } { 1 + \beta _ { I } ^ { 2 } L ^ { \top } \Sigma _ { t - 1 } L } \frac { \phi ( \alpha _ { t } ( L ) ) ^ { 2 } } { \Phi ( \alpha _ { t } ( L ) ) \Phi ( - \alpha _ { t } ( L ) ) } .
$$

B<sub>y</sub> condition (iii), the boundedness of $\boldsymbol { z } _ { t - 1 , i _ { t } }$ <sub>,</sub> <sub>an</sub>d <sub>con</sub>ti<sub>nu</sub>it<sub>y</sub> <sub>an</sub>d <sub>s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>pos</sub>iti<sub>v</sub>it<sub>y</sub> <sub>o</sub>f th<sub>e</sub> <sub>express</sub>i<sub>on</sub> <sub>a</sub>b<sub>ove,</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s</sub> $\tau _ { * } > 0$ <sub>suc</sub>h th<sub>a</sub>t <sub>even</sub>t<sub>ua</sub>ll<sub>y</sub>

$$
\frac { \delta _ { t } ( L ) e ^ { - z _ { t - 1 , i _ { t } } ^ { 2 } / 2 } } { 4 r _ { L } L ^ { \top } \Sigma _ { t - 1 } L } \geq \tau _ { * } \qquad ( L \in \mathcal { D } ) .
$$

<sup>C</sup>onse<sub>q</sub>uent<sup>l</sup><sub>y</sub>,

$$
\sum _ { L \in \mathcal { D } } G _ { t } ( L ) \geq \tau _ { * } w _ { t - 1 , i _ { t } } \sum _ { L \in \mathcal { D } } \left( \pmb { g } _ { t , i _ { t } } ^ { \top } \Sigma _ { t - 1 } L \right) ^ { 2 } \geq \frac { \tau _ { * } w _ { * } \kappa } { 4 } \sigma _ { t - 1 , i _ { t } } ^ { 4 } .
$$

Th<sub>us</sub>

$$
\operatorname* { l i m i n f } _ { t \to \infty } \sum _ { L \in \mathcal { D } } G _ { t } ( L ) \geq \frac { \tau _ { * } w _ { * } \kappa } { 4 } c ^ { 2 } \delta ^ { 4 } > 0 .
$$

Fi<sub>na</sub>ll<sub>y,</sub> <sub>a</sub>ft<sub>er</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>hi<sub>ng</sub> <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>exp</sub>l<sub>ora</sub>ti<sub>on,</sub> <sub>we</sub> <sub>connec</sub>t l<sub>earn</sub>i<sub>ng</sub> <sub>on</sub> ${ \bf 1 ^ { \perp } }$ t<sub>o</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>oppor</sub>t<sub>un</sub>it<sub>y</sub> <sub>cos</sub>t<sub>.</sub>

LEMMA D.8. For every $\boldsymbol { x } \in \mathbb { R } ^ { I }$

$$
S _ { k } ^ { * } ( \pmb { x } ) = S _ { k } ^ { * } ( P \pmb { x } ) , \qquad \sum _ { i \in S _ { k } ^ { * } ( \pmb { x } ) } x _ { i } = \sum _ { i \in S _ { k } ^ { * } ( P \pmb { x } ) } ( P \pmb { x } ) _ { i } + k \overline { { \mathscr { x } } } , \qquad \overline { { \mathscr { x } } } : = \frac { 1 } { I } \mathbf { 1 } ^ { \top } \pmb { x } .
$$

Consequently, if

$$
\mathbb { E } \| P \pmb { \mu } _ { T } - P \pmb { \theta } \| _ { \infty } \to 0 ,
$$

then $\mathbb { E } [ { \mathrm { O C } } _ { T } ] \to 0 .$

Proof. Since $x _ { i } = ( P \pmb { x } ) _ { i } + \overline { { x } }$ f<sub>or</sub> <sub>every</sub> �<sub>,</sub> <sub>any</sub> <sub>s</sub>i<sub>ze-</sub>� <sub>su</sub>b<sub>se</sub>t � <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\sum _ { i \in A } x _ { i } = \sum _ { i \in A } ( P \pmb { x } ) _ { i } + k \overline { { x } } .
$$

Th<sub>e</sub> t<sub>erm</sub> �<sub>�</sub> d<sub>oes no</sub>t d<sub>epen</sub>d <sub>on</sub> �<sub>, so</sub> $S _ { k } ^ { * } ( { \pmb x } ) = S _ { k } ^ { * } ( P { \pmb x } )$ <sub>.</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>sum</sub> <sub>a</sub>t $A = S _ { k } ^ { * } ( x )$ <sub>g</sub><sup>i</sup>ves

$$
\sum _ { i \in S _ { k } ^ { * } ( { \pmb x } ) } x _ { i } = \sum _ { i \in S _ { k } ^ { * } ( P { \pmb x } ) } ( P { \pmb x } ) _ { i } + k \overline { { x } } .
$$

F<sub>or</sub> th<sub>e</sub> <sub>oppor</sub>t<sub>un</sub>it<sub>y-cos</sub>t b<sub>oun</sub>d<sub>,</sub> <sub>no</sub>t<sub>e</sub> th<sub>a</sub>t $S _ { k } ^ { * } ( { \pmb { \mu } } _ { T } ) = S _ { k } ^ { * } ( P { \pmb { \mu } } _ { T } )$ by the identity just proved. Next, note that th<sub>e</sub> <sub>sum</sub> <sub>o</sub>f th<sub>e</sub> � l<sub>arges</sub>t <sub>e</sub>l<sub>emen</sub>t<sub>s</sub> i<sub>s</sub> �<sub>-</sub>Li<sub>psc</sub>hit<sub>z</sub> i<sub>n</sub> th<sub>e</sub> L<sub>-</sub>i<sub>n</sub>fi<sub>n</sub>it<sub>y</sub> <sub>norm:</sub> f<sub>or</sub> <sub>any</sub> $\pmb { x } , \pmb { y } \in \mathbb { R } ^ { I }$

$$
\sum _ { i \in S _ { k } ^ { * } ( { \pmb x } ) } x _ { i } - \sum _ { i \in S _ { k } ^ { * } ( { \pmb y } ) } y _ { i } \leq \sum _ { i \in S _ { k } ^ { * } ( { \pmb x } ) } ( x _ { i } - y _ { i } ) \leq k \| { \pmb x } - { \pmb y } \| _ { \infty } .
$$

Interchanging � and � gives

$$
\left| \sum _ { i \in S _ { k } ^ { * } ( { \pmb x } ) } x _ { i } - \sum _ { i \in S _ { k } ^ { * } ( { \pmb y } ) } y _ { i } \right| \leq k \| { \pmb x } - { \pmb y } \| _ { \infty } .
$$

Th<sub>us,</sub>

$$
0 \leq \mathsf { O C } _ { T } = \sum _ { i \in S _ { k } ^ { \ast } ( P \theta ) } ( P \theta ) _ { i } - \sum _ { i \in S _ { k } ^ { \ast } ( P \mu _ { T } ) } ( P \mu _ { T } ) _ { i } + \sum _ { i \in S _ { k } ^ { \ast } ( P \mu _ { T } ) } \left( ( P \mu _ { T } ) _ { i } - ( P \theta ) _ { i } \right) \leq 2 k \| P \mu _ { T } - P \theta \| _ { \infty } .
$$

T<sub>a</sub>ki<sub>ng expec</sub>t<sub>a</sub>ti<sub>ons g</sub>i<sub>ves</sub> th<sub>e resu</sub>lt<sub>.</sub> □

## D<sub>.</sub>4<sub>.</sub> P<sub>roo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 2

W<sub>e now</sub> h<sub>ave</sub> th<sub>e</sub> t<sub>oo</sub>l<sub>s</sub> t<sub>o prove</sub> Th<sub>eorem</sub> 2<sub>.</sub>

Proof. Define the set of directions sampled infinitely often, $E _ { \infty } : = \{ L \in \mathcal { D } : L$ is sampled infinitel<sub>y</sub> often}. F<sub>or</sub> <sub>eac</sub>h $\varepsilon \subseteq \mathcal { D }$ <sub>,</sub> l<sub>e</sub>t $\Omega _ { \mathcal { E } } : = \{ E _ { \infty } = \mathcal { E } \}$ <sub>.</sub> Fi<sub>x</sub> $\varepsilon \subseteq \mathcal { D }$ <sub>an</sub>d <sub>wor</sub>k <sub>on</sub> $\Omega _ { \mathcal { E } }$ <sub>.</sub> L<sub>e</sub>t

$$
M : = \operatorname { s p a n } ( { \mathcal { E } } ) , \qquad R _ { M } : = \operatorname { o r t h o g o n a l p r o j e c t o r \ o n t o \ } M , \qquad Q : = \operatorname { I d } _ { I } - R _ { M } .
$$

W<sub>e</sub> <sub>c</sub>l<sub>a</sub>i<sub>m</sub> th<sub>a</sub>t <sub>on</sub> $\Omega _ { \mathcal { E } } , \mathbf { 1 } ^ { \perp } \subseteq M$ <sub>.</sub> A<sub>ssume</sub> f<sub>or con</sub>t<sub>ra</sub>di<sub>c</sub>ti<sub>on</sub> th<sub>a</sub>t $1 ^ { \perp } \not \in M$ . Because directions outside E are <sub>samp</sub>l<sub>e</sub>d <sub>on</sub>l<sub>y</sub> fi<sub>n</sub>it<sub>e</sub>l<sub>y</sub> <sub>o</sub>ft<sub>en</sub> <sub>an</sub>d ${ \mathcal { D } } \backslash \varepsilon$ i<sub>s</sub> fi<sub>n</sub>it<sub>e,</sub> th<sub>ere</sub> <sub>ex</sub>i<sub>s</sub>t<sub>s</sub> <sub>an</sub> <sub>a</sub>l<sub>mos</sub>t<sub>-sure</sub>l<sub>y</sub> fi<sub>n</sub>it<sub>e</sub> <sub>roun</sub>d $T ^ { * }$ <sub>suc</sub>h th<sub>a</sub>t <sub>a</sub>ft<sub>er</sub> <sub>roun</sub>d $T ^ { * }$ <sub>no</sub> di<sub>rec</sub>ti<sub>on</sub> <sub>ou</sub>t<sub>s</sub>id<sub>e</sub> $\varepsilon$ i<sub>s</sub> <sub>ever</sub> <sub>samp</sub>l<sub>e</sub>d <sub>aga</sub>i<sub>n.</sub>

Ste<sub>p</sub> 1: learnin<sub>g</sub> on the rec<sub>u</sub>rrent s<sub>p</sub>ace<sub>.</sub> L<sub>e</sub>mm<sub>a</sub> D<sub>.</sub>1 <sub>app</sub>li<sub>es</sub> <sub>o</sub>n $\Omega _ { \mathcal { E } }$ <sub>.</sub> H<sub>ence</sub> f<sub>or</sub> <sub>every</sub> $\nu \in M$

$$
\begin{array} { r } {  { \boldsymbol { \nu } } ^ { \top } \pmb { \mu } _ { t - 1 } ^ { * } \to  { \boldsymbol { \nu } } ^ { \top } \pmb { \theta } , \qquad { \boldsymbol { \nu } } ^ { \top } \Sigma _ { t - 1 } ^ { * }  { \boldsymbol { \nu } } \to 0 \qquad \mathrm { a . s . \ o n } \ \Omega _ { \mathcal { E } } . } \end{array}
$$

Ch<sub>oose a</sub> b<sub>as</sub>i<sub>s</sub> $L _ { 1 } , \dots , L _ { d } \in \mathcal { E }$ <sub>o</sub>f � <sub>an</sub>d l<sub>e</sub>t $A = \left[ L _ { 1 } ~ \cdots ~ L _ { d } \right]$ <sub>.</sub> B<sub>y</sub> C<sub>auc</sub>h<sub>y–</sub>S<sub>c</sub>h<sub>wa</sub>r<sub>z,</sub>

$$
\begin{array} { r } { | L _ { r } ^ { \top } \Sigma _ { t - 1 } ^ { * } L _ { s } | \leq \sqrt { L _ { r } ^ { \top } \Sigma _ { t - 1 } ^ { * } L _ { r } } \sqrt { L _ { s } ^ { \top } \Sigma _ { t - 1 } ^ { * } L _ { s } } \to 0 , } \end{array}
$$

so $A ^ { \top } \Sigma _ { t - 1 } ^ { * } A  0$ <sub>en</sub>t<sub>ryw</sub>i<sub>se an</sub>d h<sub>ence</sub> i<sub>n opera</sub>t<sub>or norm.</sub> It f<sub>o</sub>ll<sub>ows</sub> th<sub>a</sub>t

$$
\| R _ { M } \Sigma _ { t - 1 } ^ { * } R _ { M } \| _ { \mathrm { o p } } \longrightarrow 0 , \qquad R _ { M } \mu _ { t - 1 } ^ { * } \longrightarrow R _ { M } \pmb \theta \qquad \mathrm { a . s . ~ o n } \ \Omega _ { \mathcal { E } } .
$$

St<sub>ep</sub> 2<sub>: app</sub>r<sub>o</sub>xim<sub>a</sub>t<sub>e va</sub>ri<sub>a</sub>n<sub>ce s</sub>t<sub>ays pos</sub>iti<sub>ve o</sub>f th<sub>e</sub> r<sub>ecu</sub>rr<sub>e</sub>nt <sub>spa</sub>n<sub>.</sub> L<sub>emma</sub> D<sub>.</sub>2 <sub>app</sub>li<sub>es w</sub>ith th<sub>e conc</sub>l<sub>us</sub>i<sub>ons</sub> <sub>o</sub>f St<sub>ep</sub> 1<sub>.</sub> Th<sub>us, a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y on</sub> $\Omega _ { \mathcal { E } }$ <sub>,</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s</sub> $\eta > 0$ <sub>suc</sub>h th<sub>a</sub>t $Q \Sigma _ { t - 1 } ^ { * } Q \succeq \eta Q$ f<sub>or a</sub>ll <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> l<sub>arge</sub> �<sub>.</sub> B<sub>y</sub> L<sub>emmas</sub> D<sub>.</sub>3 <sub>an</sub>d D<sub>.</sub>4<sub>,</sub>

$$
\begin{array} { r } { \| R _ { M } \Sigma _ { t - 1 } R _ { M } \| _ { \mathrm { o p } } \to 0 , \qquad \| R _ { M } \Sigma _ { t - 1 } \mathcal { Q } \| _ { \mathrm { o p } } + \| Q \Sigma _ { t - 1 } R _ { M } \| _ { \mathrm { o p } } \to 0 , \qquad Q \Sigma _ { t - 1 } \mathcal { Q } \succeq \chi _ { - } \eta Q \sum _ { t = 1 } ^ { \infty } \| Q \Sigma _ { t - 1 } \mathcal { Q } \| _ { \mathrm { o p } } \to 0 . } \end{array}\tag{D.2}
$$

<sub>a</sub>l<sub>mos</sub>t <sub>sure</sub>l<sub>y</sub> <sub>on</sub> $\Omega _ { \mathcal { E } }$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> th<sub>e</sub> l<sub>as</sub>t i<sub>nequa</sub>lit<sub>y</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> <sub>a</sub>ll <sub>su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> l<sub>arge</sub> �<sub>.</sub>

Step 3: the sum of scores stays bounded away from zero. The conditions of Lemma D.7 follow from (D.2), Assum<sub>p</sub>tion 2(ii), and Lemma D.3. Therefore,

$$
\operatorname* { l i m i n f } _ { t \to \infty } \sum _ { L \in \mathcal { D } } G _ { t } ( L ) > 0 \qquad \mathrm { a . s . \ o n } \Omega _ { \mathcal { E } } .\tag{D.3}
$$

Ste<sub>p</sub> 4: e<sub>v</sub>er<sub>y</sub> rec<sub>u</sub>rrent direction’s score <sub>g</sub>oes to 0<sub>.</sub> Fi<sub>x</sub> $L \in { \mathcal { E } } _ { \mathbb { { \Gamma } } }$ <sub>.</sub> Si<sub>nce</sub> $L \in M$ <sub>,</sub> St<sub>ep</sub> 1 <sub>g</sub>i<sub>ves</sub> $L ^ { \top } \Sigma _ { t - 1 } ^ { * } L  0$ Assum<sub>p</sub>tion 2(i) then <sub>y</sub>ields

$$
L ^ { \top } \Sigma _ { t - 1 } L \leq \chi _ { + } L ^ { \top } \Sigma _ { t - 1 } ^ { * } L  0 .
$$

L<sub>emma</sub> D<sub>.</sub>6 th<sub>ere</sub>f<sub>ore</sub> i<sub>mp</sub>li<sub>es</sub>

$$
G _ { t } ( L ) \to 0 \qquad \mathrm { a . s . ~ o n } \Omega _ { \mathcal { E } } .
$$

Th<sub>us</sub>

$$
\sum _ { L \in \mathcal { E } } G _ { t } ( L ) \to 0 \qquad \mathrm { a . s . ~ o n } \ : \Omega _ { \mathcal { E } } .\tag{D.4}
$$

Step 5: contradiction with the chosen evaluation. B<sub>y</sub> (D.3), (D.4),

$$
\operatorname* { l i m i n f } _ { t \to \infty } \sum _ { L \in \mathcal { D } \backslash \mathcal { E } } G _ { t } ( L ) > 0 \qquad \mathrm { a . s . \ o n } \Omega _ { \mathcal { E } } .
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
\operatorname* { l i m i n f } _ { t \to \infty } \operatorname* { m a x } _ { L \in \mathcal { D } \backslash \mathcal { E } } G _ { t } ( L ) > 0 , \qquad \operatorname* { m a x } _ { L \in \mathcal { E } } G _ { t } ( L ) \to 0 .
$$

H<sub>ence</sub> th<sub>ere</sub> <sub>ex</sub>i<sub>s</sub>t<sub>s</sub> <sub>an</sub> <sub>a</sub>l<sub>mos</sub>t<sub>-sure</sub>l<sub>y</sub> fi<sub>n</sub>it<sub>e</sub> <sub>ran</sub>d<sub>om</sub> ti<sub>me</sub> $T _ { 1 }$ <sub>suc</sub>h th<sub>a</sub>t<sub>,</sub> f<sub>or</sub> <sub>every</sub> $t \geq T _ { 1 }$

$$
\operatorname* { m a x } _ { L \in { \mathcal { D } } \backslash { \mathcal { E } } } G _ { t } ( L ) > \operatorname* { m a x } _ { L \in { \mathcal { E } } } G _ { t } ( L ) .
$$

Si<sub>nce</sub> SA <sub>max</sub>i<sub>m</sub>i<sub>zes</sub> th<sub>e score</sub> $( \mathrm { D } . 1 )$ over $\mathcal { D }$ , the direction chosen at round � cannot be in E for all suficientl<sub>y</sub> l<sub>arge</sub> �<sub>.</sub> B<sub>u</sub>t <sub>a</sub>ft<sub>er</sub> <sub>roun</sub>d $T ^ { * }$ , ever<sub>y</sub> sam<sub>p</sub>led direction lies in E. Contradiction. Therefore,

$$
\mathbf { 1 } ^ { \scriptscriptstyle \perp } \subseteq \operatorname { s p a n } ( E _ { \infty } ) \qquad { \mathrm { a l m o s t ~ s u r e l y } } .
$$

St<sub>ep</sub> 6<sub>: convergence.</sub> It <sub>rema</sub>i<sub>ns</sub> t<sub>o s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> f<sub>u</sub>ll hi<sub>s</sub>t<sub>ory</sub> id<sub>en</sub>tifi<sub>es</sub> $P \pmb { \theta } ,$ <sub>, a</sub>ft<sub>er w</sub>hi<sub>c</sub>h L<sub>emma</sub> D<sub>.</sub>8 i<sub>mp</sub>li<sub>es</sub> th<sub>e</sub> <sub>conc</sub>l<sub>us</sub>i<sub>on o</sub>f th<sub>e</sub> th<sub>eorem.</sub> Fi<sub>x</sub> $\varepsilon \subseteq \mathcal { D }$ <sub>.</sub> On $\Omega _ { \mathcal { E } } .$ <sub>,</sub> f<sub>or eac</sub>h $i = 1 , \ldots , I ,$ <sub>, we</sub> h<sub>ave</sub> $P e _ { i } \in \mathbf { 1 } ^ { \bot } \subseteq M$ <sub>, so</sub> L<sub>emma</sub> D<sub>.</sub>1 i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t $\mathbb { 1 } _ { \Omega _ { \mathcal { E } } } ( P e _ { i } ) ^ { \top } \pmb { \theta }$ i<sub>s</sub> $\mathcal { H } _ { \infty }$ <sub>-measura</sub>bl<sub>e.</sub> Si<sub>nce</sub> $( P \pmb { \theta } ) _ { i } = \pmb { e } _ { i } ^ { \top } P \pmb { \theta } = ( P \pmb { e } _ { i } ) ^ { \top } \pmb { \theta }$ <sub>,</sub> it f<sub>o</sub>ll<sub>ows</sub> th<sub>a</sub>t $\mathbb { 1 } _ { \Omega _ { \mathcal { E } } } P \pmb { \theta }$ i<sub>s</sub> H<sub>∞</sub>-measurable. Since $\begin{array} { r } { P \pmb { \theta } = \sum _ { \mathcal { E } \subseteq \mathcal { D } } \mathbb { 1 } _ { \Omega _ { \mathcal { E } } } P \pmb { \theta } } \end{array}$ a.s., �� is $\mathcal { H } _ { \infty }$ <sub>-measura</sub>bl<sub>e.</sub>

Th<sub>ere</sub>f<sub>ore,</sub> f<sub>or</sub> <sub>eac</sub>h $i = 1 , \ldots , I ,$

$$
( P \pmb { \mu } _ { t - 1 } ^ { * } ) _ { i } = \mathbb { E } [ ( P \pmb { \theta } ) _ { i } | \mathcal { H } _ { t } ]  ( P \pmb { \theta } ) _ { i } \qquad \mathrm { a . s . \ a n d \ i n \ } L ^ { 1 } ,
$$

b<sub>y</sub> D<sub>oo</sub>b’<sub>s</sub> th<sub>eorem.</sub> H<sub>ence</sub>

$$
\mathbb { E } \| P \pmb { \mu } _ { t - 1 } ^ { * } - P \pmb { \theta } \| _ { \infty } \le \sum _ { i = 1 } ^ { I } \mathbb { E } \big | ( P \pmb { \mu } _ { t - 1 } ^ { * } - P \pmb { \theta } ) _ { i } \big |  0 .
$$

Thi<sub>s</sub> <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> <sub>convergence</sub> f<sub>or</sub> th<sub>e</sub> <sub>exac</sub>t <sub>pos</sub>t<sub>er</sub>i<sub>or</sub> <sub>mean</sub> <sub>a</sub>l<sub>ong</sub> th<sub>e</sub> <sub>run</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>a</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>.</sub> W<sub>e</sub> now return to the <sub>p</sub>rocedure with bud<sub>g</sub>et � and show that E∥ $P \pmb { \mu } _ { T } - P \pmb { \theta } \| _ { \infty } \to 0$ as $R \to \infty$ <sub>.</sub> F<sub>or</sub> <sub>a</sub> <sub>run</sub> <sub>w</sub>ith b<sub>u</sub>d<sub>ge</sub>t �<sub>,</sub> t<sub>a</sub>k<sub>e</sub> $t = \lfloor R / \operatorname* { m a x } ( r _ { 0 } , r _ { 1 } ) \rfloor$ <sub>so</sub> th<sub>a</sub>t f<sub>or</sub> th<sub>e</sub> fi<sub>rs</sub>t � <sub>eva</sub>l<sub>ua</sub>ti<sub>ons</sub> thi<sub>s</sub> i<sub>s</sub> <sub>equ</sub>i<sub>va</sub>l<sub>en</sub>t t<sub>o</sub> th<sub>e</sub> <sub>un</sub>b<sub>u</sub>d<sub>ge</sub>t<sub>e</sub>d <sub>run</sub> <sub>cons</sub>id<sub>ere</sub>d <sub>a</sub>b<sub>ove.</sub> It<sub>s</sub> t<sub>erm</sub>i<sub>na</sub>l hi<sub>s</sub>t<sub>ory</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> hi<sub>s</sub>t<sub>ory</sub> th<sub>roug</sub>h <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> �<sub>.</sub> J<sub>ensen</sub>’<sub>s</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>an</sub>d th<sub>e</sub> t<sub>r</sub>i<sub>ang</sub>l<sub>e</sub> i<sub>nequa</sub>lit<sub>y g</sub>i<sub>ve</sub>

$$
\begin{array} { r l } & { \mathbb { E } \| P \mu _ { T } ^ { * } - P \theta \| _ { \infty } \leq \mathbb { E } \| P \mu _ { T } ^ { * } - P \mu _ { t } ^ { * } \| _ { \infty } + \mathbb { E } \| P \mu _ { t } ^ { * } - P \theta \| _ { \infty } } \\ & { \qquad = \mathbb { E } \left\| \mathbb { E } \big [ P \theta - P \mu _ { t } ^ { * } \big | \mathcal { H } _ { T + 1 } \big ] \right\| _ { \infty } + \mathbb { E } \| P \mu _ { t } ^ { * } - P \theta \| _ { \infty } } \\ & { \qquad \leq 2 \mathbb { E } \| P \mu _ { t } ^ { * } - P \theta \| _ { \infty } \to 0 \qquad \mathrm { ~ a s ~ } R \to \infty . } \end{array}
$$

Al<sub>so,</sub> f<sub>or</sub> <sub>any</sub> $z \in \mathbb { R } ^ { I } , \| P z \| _ { \infty } \leq 2 \| z \| _ { \infty }$ . Thus, b<sub>y</sub> Assum<sub>p</sub>tion 2(iii),

$$
\begin{array} { r } { \mathbb { E } \| P \mu _ { T } - P \theta \| _ { \infty } \leq \mathbb { E } \| P \mu _ { T } - P \mu _ { T } ^ { * } \| _ { \infty } + \mathbb { E } \| P \mu _ { T } ^ { * } - P \theta \| _ { \infty } \leq 2 \mathbb { E } \| \mu _ { T } - \mu _ { T } ^ { * } \| _ { \infty } + \mathbb { E } \| P \mu _ { T } ^ { * } - P \theta \| _ { \infty } \to 0 . } \end{array}
$$

L<sub>e</sub>mm<sub>a</sub> D<sub>.</sub>8 <sub>g</sub>i<sub>ves</sub>

$$
\mathbb { E } [ { \bf O C } _ { T } ]  0 \qquad \mathrm { a s } \ R  \infty .
$$

## A<sub>pp</sub>endix E: Su<sub>pp</sub>lement to $\ S 7$

Th<sub>e</sub> fi<sub>gures</sub> b<sub>e</sub>l<sub>ow</sub> <sub>are</sub> th<sub>e</sub> <sub>coun</sub>t<sub>erpar</sub>t<sub>s</sub> <sub>o</sub>f Fi<sub>gure</sub> 2<sub>,</sub> Fi<sub>gure</sub> 3<sub>a,</sub> <sub>an</sub>d Fi<sub>gure</sub> 6 f<sub>or</sub> <sub>a</sub>ll <sub>o</sub>th<sub>er</sub> <sub>con</sub>fi<sub>gura</sub>ti<sub>ons.</sub>

## E<sub>.</sub>1<sub>.</sub> All<sub>oca</sub>ti<sub>ons over</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t

![](images/ba8bc6397b4a860ff1831425f3fda01e2eb84e8827404c0979a2508b42692277.jpg)  
Figure E.1 Mean allocation of SA by true rank in each eighth of the budget, all � = 10 configurations.

![](images/fe0254fd92efaab245db8604e1f303a9c20e574bd79698cdc862b745cb8f47e3.jpg)  
Figure E.2 Same as Figure E.1 for � = 20.

![](images/034c0c40894eb0e71b8e371b60643499086c03688b7c2fca2638631e7e424af4.jpg)  
Figure E.3 Same as Figure E.1 for � = 40.

![](images/562017009475115b6cd5e0ce2e551ec9f27dfd67c6d633eac42684158d67336a.jpg)  
Figure E.4 Same as Figure E.1 for � = 80.

![](images/a603d60b2eb9be30c46aa2083070a02cd367e3c0efba0754e725ba9b9730af8b.jpg)  
Figure E.5 Same as Figure E.1 for � = 160.

## E<sub>.</sub>2<sub>.</sub> All<sub>oca</sub>ti<sub>o</sub>n <sub>o</sub>f SA <sub>ve</sub>r<sub>sus</sub> KG

![](images/5d442f40711363e89cf86c9e41733f36a1a4af280e20fb22fdc0a0fe71b2b217.jpg)  
Figure E.6 Average allocation difference after the switch, SE minus SA, in cost units, all configurations at �<sub>1</sub> = 2. Green marks where SE spends more, red where it spends less.

![](images/6199771e6827588b758d7cd4ccb64e75f2330e1634ceb1ccc537d698cca22d03.jpg)  
Figure E.7 Same as Figure E.6 for $r _ { 1 } = 3 .$

## E<sub>.</sub>3<sub>.</sub> C<sub>o</sub>m<sub>pa</sub>ri<sub>so</sub>n t<sub>o</sub> <sub>ex</sub>t<sub>e</sub>rn<sub>a</sub>l b<sub>e</sub>n<sub>c</sub>hm<sub>a</sub>rk<sub>s</sub>

![](images/c65faef36e33751d9864ea2efbae79fef6a92bb15caeea7f94c8203883062a76.jpg)  
Figure E.8 EOC against the budget spent, all configurations at $r _ { 1 } = 2 .$

![](images/a067a2ebc395c7d68c4cd2f360f9dd29f64602a0696d2827a74f028a22875c93.jpg)  
Figure E.9 EOC against the budget spent, all configurations at $r _ { 1 } = 3 .$