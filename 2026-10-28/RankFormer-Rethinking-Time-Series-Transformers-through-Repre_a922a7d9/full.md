# RankFormer: Rethinking Time Series Transformers through Representation Capacity Analysis and Progressive Decoding

Peiwang Tang<sup>1,2,3</sup> C<sup>h</sup>eng HE<sup>1,2,3</sup> Xijie Liang<sup>3</sup> De<sup>f</sup>u Lian<sup>1,2,✉</sup>

<sup>1</sup>Uni<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> <sub>o</sub>f S<sub>c</sub>i<sub>e</sub>n<sub>ce</sub> <sub>a</sub>nd T<sub>ec</sub>hn<sub>o</sub>l<sub>ogy</sub> <sub>o</sub>f Chin<sub>a,</sub> H<sub>e</sub>f<sub>e</sub>i<sub>,</sub> Anh<sub>u</sub>i<sub>,</sub> Chin<sub>a</sub>

<sup>2</sup>State Ke<sub>y</sub> Laborator<sub>y</sub> of Co<sub>g</sub>niti<sub>v</sub>e Intelli<sub>g</sub>ence<sub>,</sub> Hefei<sub>,</sub> Anh<sub>u</sub>i<sub>,</sub> China

<sup>3</sup>Shan<sub>g</sub>hai Black Win<sub>g</sub> Asset Mana<sub>g</sub>ement<sub>,</sub> Shan<sub>g</sub>hai<sub>,</sub> China

tpw@mail.ustc.edu.cn cheng.he@mail.ustc.edu.cn

lxjie@blackwingasset.com <sup>✉</sup>liandefu@ustc.edu.cn

## Abstract

M<sub>u</sub>lti<sub>va</sub>ri<sub>a</sub>t<sub>e</sub> tim<sub>e se</sub>ri<sub>es</sub> f<sub>o</sub>r<sub>ecas</sub>tin<sub>g p</sub>l<sub>ays a c</sub>riti<sub>ca</sub>l r<sub>o</sub>l<sub>e</sub> in W<sub>e</sub>b and c<sub>y</sub>ber-<sub>p</sub>h<sub>y</sub>sical s<sub>y</sub>stems<sub>,</sub> where lar<sub>g</sub>e-scale services and IoT inf<sub>ras</sub>t<sub>ruc</sub>t<sub>ures genera</sub>t<sub>e comp</sub>l<sub>ex</sub> t<sub>empora</sub>l d<sub>a</sub>t<sub>a.</sub> T<sub>rans</sub>f<sub>ormer-</sub>b<sub>ase</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> h<sub>ave ac</sub>hi<sub>eve</sub>d <sub>s</sub>t<sub>rong emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>per</sub>f<sub>ormance</sub> i<sub>n</sub> thi<sub>s</sub> d<sub>o-</sub> main<sub>, y</sub>et the <sub>p</sub>rinci<sub>p</sub>les <sub>g</sub>overnin<sub>g</sub> the ex<sub>p</sub>ressive ca<sub>p</sub>acit<sub>y</sub> of their l<sub>earne</sub>d <sub>represen</sub>t<sub>a</sub>ti<sub>ons rema</sub>i<sub>n</sub> i<sub>nsu</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y un</sub>d<sub>ers</sub>t<sub>oo</sub>d<sub>.</sub> M<sub>oreover,</sub> man<sub>y</sub> existin<sub>g</sub> models rel<sub>y</sub> on sim<sub>p</sub>lified decodin<sub>g</sub> strate<sub>g</sub>ies<sub>,</sub> inc<sup>l</sup>u<sup>d</sup>ing c<sup>h</sup>anne<sup>l</sup>-in<sup>d</sup>epen<sup>d</sup>ent mo<sup>d</sup>e<sup>l</sup>ing an<sup>d d</sup>irect projection <sup>h</sup>ea<sup>d</sup>s, which limit their abilit<sub>y</sub> to ca<sub>p</sub>ture com<sub>p</sub>lex multivariate d<sub>y</sub>namics. To address these challenges, we propose RankFormer, a novel <sub>enco</sub>d<sub>er-</sub>d<sub>eco</sub>d<sub>er arc</sub>hit<sub>ec</sub>t<sub>ure</sub> f<sub>or mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> ti<sub>me ser</sub>i<sub>es</sub> f<sub>orecas</sub>t<sub>-</sub> i<sub>ng.</sub> W<sub>e es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>a ran</sub>k<sub>-</sub>b<sub>ase</sub>d <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> f<sub>ramewor</sub>k t<sub>o</sub> i<sub>nves</sub>ti<sub>ga</sub>t<sub>e</sub> h<sub>ow</sub> t<sub>o</sub>k<sub>en</sub>i<sub>za</sub>ti<sub>on, em</sub>b<sub>e</sub>ddi<sub>ng, an</sub>d <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> d<sub>es</sub>i<sub>gns a</sub>f<sub>ec</sub>t th<sub>e repre-</sub> sentation ca<sub>p</sub>acit<sub>y</sub> of time series Transformers. O<sub>u</sub>r anal<sub>y</sub>sis reveals that <sub>p</sub>atchification increases the <sub>p</sub>otential re<sub>p</sub>resentation ca<sub>p</sub>acit<sub>y</sub> b<sub>y</sub> a<sub>gg</sub>re<sub>g</sub>atin<sub>g</sub> multi<sub>p</sub>le tem<sub>p</sub>oral observations into each token<sub>,</sub> <sub>w</sub>hil<sub>e su</sub>b<sub>sequen</sub>t <sub>enco</sub>di<sub>ng opera</sub>ti<sub>ons</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub> h<sub>ow muc</sub>h <sub>o</sub>f thi<sub>s</sub> ca<sub>p</sub>acit<sub>y</sub> is realized. Guided b<sub>y</sub> these insi<sub>g</sub>hts<sub>,</sub> we develo<sub>p</sub> a rank enhancin<sub>g</sub> encoder with Hour<sub>g</sub>lass Attention to im<sub>p</sub>rove rank stabilit<sub>y</sub> d<sub>u</sub>rin<sub>g</sub> inf<sub>o</sub>rm<sub>a</sub>ti<sub>o</sub>n int<sub>e</sub>r<sub>ac</sub>ti<sub>o</sub>n<sub>.</sub> F<sub>u</sub>rth<sub>e</sub>rm<sub>o</sub>r<sub>e, we</sub> intr<sub>o</sub>d<sub>uce</sub> th<sub>e</sub> Interactive Pro<sub>g</sub>ressive Decoder (IPD), which im<sub>p</sub>roves future trajectory mo<sup>d</sup>e<sup>l</sup>ing t<sup>h</sup>roug<sup>h</sup> g<sup>l</sup>o<sup>b</sup>a<sup>l</sup>-gui<sup>d</sup>e<sup>d</sup> cross-c<sup>h</sup>anne<sup>l</sup> interaction and <sub>p</sub>ro<sub>g</sub>ressive refinement. Extensive ex<sub>p</sub>eriments on real-world l<sub>ong-</sub>t<sub>erm</sub> f<sub>orecas</sub>ti<sub>ng</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>a</sub>t R<sub>an</sub>kF<sub>ormer</sub> achieves com<sub>p</sub>etitive <sub>p</sub>erformance a<sub>g</sub>ainst state-of-the-art methods. Code re<sub>p</sub>ositor<sub>y</sub>: htt<sub>p</sub>s://<sub>g</sub>ithub.com/Tan<sub>g</sub>Peiwan<sub>g</sub>/Rankformer.

## CCS Concepts

• Computing methodologies → Artificial intelligence.

## Keywords

Tim<sub>e</sub> S<sub>e</sub>ri<sub>es,</sub> R<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n C<sub>apac</sub>it<sub>y,</sub> H<sub>ou</sub>r<sub>g</sub>l<sub>ass</sub> Att<sub>e</sub>nti<sub>o</sub>n<sub>,</sub> Int<sub>e</sub>r-<sub>ac</sub>ti<sub>ve</sub> Pr<sub>og</sub>r<sub>ess</sub>i<sub>ve</sub> D<sub>eco</sub>d<sub>e</sub>r

## 1 Introduction

M<sub>u</sub>lti<sub>va</sub>ri<sub>a</sub>t<sub>e</sub> tim<sub>e se</sub>ri<sub>es</sub> f<sub>o</sub>r<sub>ecas</sub>tin<sub>g</sub> i<sub>s a co</sub>r<sub>e p</sub>r<sub>o</sub>bl<sub>e</sub>m in m<sub>o</sub>d<sub>e</sub>rn Web and c<sub>y</sub>ber-<sub>p</sub>h<sub>y</sub>sical s<sub>y</sub>stems. The <sub>g</sub>rowin<sub>g</sub> inte<sub>g</sub>ration of online services and IoT devices <sub>p</sub>roduces lar<sub>g</sub>e volumes of continuousl<sub>y</sub> evolvin<sub>g</sub> sensor data [37, 38, 41], su<sub>pp</sub>ortin<sub>g</sub> a<sub>pp</sub>lications such as <sub>server</sub> l<sub>oa</sub>d b<sub>a</sub>l<sub>anc</sub>i<sub>ng,</sub> t<sub>ra</sub>fi<sub>c pre</sub>di<sub>c</sub>ti<sub>on, an</sub>d <sub>resource p</sub>l<sub>ann</sub>i<sub>ng</sub> [25, 39, 40], as well as broader domains includin<sub>g</sub> finance, <sub>p</sub>ower s<sub>y</sub>stems, and trans<sub>p</sub>ortation [5, 29, 42]. Transformer-based models have achieved substantial <sub>p</sub>ro<sub>g</sub>ress in time series forecastin<sub>g</sub> owin<sub>g</sub> to their abilit<sub>y</sub> to ca<sub>p</sub>ture lon<sub>g</sub>-ran<sub>g</sub>e de<sub>p</sub>endencies [10, 28, 43]. R<sub>ecen</sub>t <sub>arc</sub>hit<sub>ec</sub>t<sub>ura</sub>l <sub>a</sub>d<sub>vances,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>pa</sub>t<sub>c</sub>h<sub>-</sub>b<sub>ase</sub>d t<sub>o</sub>k<sub>en</sub>i<sub>za</sub>ti<sub>on</sub> <sub>an</sub>d <sub>spec</sub>i<sub>a</sub>li<sub>ze</sub>d <sub>a</sub>tt<sub>en</sub>ti<sub>on mec</sub>h<sub>an</sub>i<sub>sms,</sub> h<sub>ave</sub> f<sub>ur</sub>th<sub>er</sub> i<sub>mprove</sub>d f<sub>ore-</sub> castin<sub>g p</sub>erformance across diverse benchmarks [13, 14]. However, des<sub>p</sub>ite this em<sub>p</sub>irical success<sub>,</sub> the underl<sub>y</sub>in<sub>g</sub> <sub>p</sub>rinci<sub>p</sub>les that determine the re<sub>p</sub>resentation ca<sub>p</sub>acit<sub>y</sub> of time series Transformers <sub>an</sub>d th<sub>e</sub>i<sub>r</sub> <sub>a</sub>bilit<sub>y</sub> t<sub>o</sub> t<sub>rans</sub>l<sub>a</sub>t<sub>e</sub> l<sub>earne</sub>d <sub>represen</sub>t<sub>a</sub>ti<sub>ons</sub> i<sub>n</sub>t<sub>o</sub> <sub>accura</sub>t<sub>e</sub> f<sub>orecas</sub>t<sub>s</sub> <sub>rema</sub>i<sub>n</sub> i<sub>nsu</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> <sub>un</sub>d<sub>ers</sub>t<sub>oo</sub>d<sub>.</sub>

This limited understandin<sub>g</sub> raises two fundamental <sub>q</sub>uestions:

First, how do encoder architectures create and preserve expressive representations? Existing time series Transformers im<sub>p</sub>rove re<sub>p</sub>resentation ca<sub>p</sub>abilit<sub>y</sub> throu<sub>g</sub>h various encodin<sub>g</sub> desi<sub>g</sub>ns<sub>,</sub> includin<sub>g</sub> tokenization<sub>,</sub> embeddin<sub>g,</sub> and attention mechanisms, yet <sup>h</sup>ow t<sup>h</sup>ese components joint<sup>l</sup>y s<sup>h</sup>ape t<sup>h</sup>e capacity o<sup>f</sup> l<sub>earne</sub>d t<sub>empora</sub>l <sub>represen</sub>t<sub>a</sub>ti<sub>ons</sub> <sub>rema</sub>i<sub>ns</sub> i<sub>nsu</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> <sub>un</sub>d<sub>ers</sub>t<sub>oo</sub>d<sub>.</sub> Patchification <sub>p</sub>rovides a re<sub>p</sub>resentative exam<sub>p</sub>le: b<sub>y</sub> a<sub>gg</sub>re<sub>g</sub>atin<sub>g</sub> <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> t<sub>empora</sub>l <sub>o</sub>b<sub>serva</sub>ti<sub>ons</sub> i<sub>n</sub>t<sub>o eac</sub>h t<sub>o</sub>k<sub>en,</sub> it <sub>expan</sub>d<sub>s</sub> th<sub>e a</sub>t<sub>-</sub> tainable re<sub>p</sub>resentation s<sub>p</sub>ace com<sub>p</sub>ared with <sub>p</sub>oint-wise tokenization. However<sub>,</sub> the realized re<sub>p</sub>resentation ca<sub>p</sub>acit<sub>y</sub> is further sha<sub>p</sub>ed b<sub>y</sub> subse<sub>q</sub>uent encodin<sub>g</sub> o<sub>p</sub>erations. Linear embeddin<sub>g</sub>s introduce rank constraints, while bidirectional attention (BiA) ma<sub>y</sub> cause excessive token mixin<sub>g</sub> and re<sub>p</sub>resentation homo<sub>g</sub>enization<sub>,</sub> leadin<sub>g</sub> to rank de<sub>g</sub>eneration [4, 18, 22, 35]. Althou<sub>g</sub>h unidirectional attention <sub>p</sub>rovides stron<sub>g</sub>er rank stabilit<sub>y,</sub> its restricted rece<sub>p</sub>tive field limits <sub>g</sub>lobal information interaction. Conse<sub>q</sub>uentl<sub>y,</sub> existin<sub>g</sub> attention desi<sub>g</sub>ns stru<sub>gg</sub>le to simultaneousl<sub>y</sub> maintain re<sub>p</sub>resentation diversit<sub>y</sub> and achieve efective cross-token information exchan<sub>g</sub>e<sub>,</sub> limitin<sub>g</sub> the abilit<sub>y</sub> of time series Transformers to full<sub>y</sub> ex<sub>p</sub>loit their <sup>re</sup>p<sup>resentation</sup> <sup>s</sup>p<sup>ace.</sup>

Second, how can forecasting decoders fully exploit the learned representations? Although modern encoders can produce increasin<sub>g</sub>l<sub>y</sub> ex<sub>p</sub>ressive latent features<sub>,</sub> man<sub>y</sub> existin<sub>g</sub> forecastin<sub>g</sub> architectures still rel<sub>y</sub> on sim<sub>p</sub>lified decodin<sub>g</sub> strate<sub>g</sub>ies. Channel-inde<sub>p</sub>endent modelin<sub>g</sub> limits cross-variable interaction<sub>,</sub> w<sup>h</sup>i<sup>l</sup>e one-s<sup>h</sup>ot projection <sup>h</sup>ea<sup>d</sup>s suc<sup>h</sup> as F<sup>l</sup>atten-Linear map encoded re<sub>p</sub>resentations to the entire <sub>p</sub>rediction horizon throu<sub>g</sub>h a fi<sub>xe</sub>d t<sub>rans</sub>f<sub>orma</sub>ti<sub>on.</sub> S<sub>uc</sub>h d<sub>es</sub>i<sub>gns over</sub>l<sub>oo</sub>k th<sub>e evo</sub>l<sub>v</sub>i<sub>ng</sub> t<sub>empora</sub>l or<sub>g</sub>anization and <sub>p</sub>rovide limited ca<sub>p</sub>abilit<sub>y</sub> for <sub>p</sub>ro<sub>g</sub>ressivel<sub>y</sub> refining comp<sup>l</sup>ex <sup>f</sup>uture trajectories, w<sup>h</sup>ic<sup>h</sup> is particu<sup>l</sup>ar<sup>l</sup>y restrictive <sup>f</sup>or multivariate forecastin<sub>g</sub> with stron<sub>g</sub> cross-channel de<sub>p</sub>endencies.

To address these challenges, we propose RankFormer, a prin-<sub>c</sub>i<sub>p</sub>l<sub>e</sub>d <sub>enco</sub>d<sub>er-</sub>d<sub>eco</sub>d<sub>er arc</sub>hit<sub>ec</sub>t<sub>ure</sub> f<sub>or mu</sub>lti<sub>var</sub>i<sub>a</sub>t<sub>e</sub> ti<sub>me ser</sub>i<sub>es</sub> f<sub>orecas</sub>ti<sub>ng.</sub> F<sub>or</sub> th<sub>e enco</sub>d<sub>er, we es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>a ran</sub>k<sub>-</sub>b<sub>ase</sub>d <sub>ana</sub>l<sub>ys</sub>i<sub>s</sub> f<sub>ramewor</sub>k t<sub>o ana</sub>l<sub>yze</sub> h<sub>ow pa</sub>t<sub>c</sub>hifi<sub>ca</sub>ti<sub>on, em</sub>b<sub>e</sub>ddi<sub>ng, an</sub>d <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> to<sub>p</sub>olo<sub>gy</sub> sha<sub>p</sub>e the efective ca<sub>p</sub>acit<sub>y</sub> of learned tem<sub>p</sub>oral re<sub>p</sub>resentations. Based on these observations<sub>,</sub> we introduce Hour<sub>g</sub>lass Attention (H<sub>g</sub>A) to im<sub>p</sub>rove rank stabilit<sub>y</sub> durin<sub>g</sub> attention-based interaction<sub>,</sub> <sub>p</sub>reservin<sub>g</sub> diverse re<sub>p</sub>resentations while enablin<sub>g</sub> eff<sub>ec</sub>ti<sub>ve g</sub>l<sub>o</sub>b<sub>a</sub>l i<sub>n</sub>f<sub>orma</sub>ti<sub>on exc</sub>h<sub>ange.</sub> F<sub>or</sub> th<sub>e</sub> d<sub>eco</sub>d<sub>er, we</sub> d<sub>eve</sub>l<sub>op</sub> an Interactive Pro<sub>g</sub>ressive Decoder (IPD), which enhances future trajectory generation t<sup>h</sup>roug<sup>h</sup> g<sup>l</sup>o<sup>b</sup>a<sup>l</sup>-context-gui<sup>d</sup>e<sup>d</sup> cross-c<sup>h</sup>anne<sup>l</sup> interaction and <sub>p</sub>ro<sub>g</sub>ressive refinement. To<sub>g</sub>ether<sub>,</sub> RankFormer <sub>p</sub>rovides a <sub>p</sub>rinci<sub>p</sub>led a<sub>pp</sub>roach to im<sub>p</sub>rovin<sub>g</sub> re<sub>p</sub>resentation learnin<sub>g</sub> an<sup>d f</sup>uture trajectory mo<sup>d</sup>e<sup>l</sup>ing in time series Trans<sup>f</sup>ormers.

![](images/53f258c6450cffb39724f5ea8a89359bfd775e6b29195adbbc5801f8ff73a6d3.jpg)

![](images/2cb6a814fe8380f8f9dc4a2ddd8cbf55205c5e3f0f6d6e330cebdddb3341862f.jpg)  
(a)

![](images/f4c781f403708e2699572ef5642dc44fc518552d08364e87f3ad9985c7079024.jpg)

![](images/b5720ae31a11d35228526cbdb685f0826ec1780d49a5bfd284fa6a95ec018b7e.jpg)  
(b)  
Figure 1: Empirical validation ofrank stability and Hourglass Attention (HgA). (a) Rapid singular value decay shows bidirectional attention induces deep-layer rank collapse, while causal masking preserves representational diversity. (b) Causal masking restricts early tokens (starvation). In contrast, HgA uses a symmetric prefix as a global hub, enabling eficient context retrieval while maintaining rank stability.

O<sub>ur ma</sub>i<sub>n con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons are summar</sub>i<sub>ze</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows:</sub>

• We <sub>p</sub>rovide a rank-based anal<sub>y</sub>sis of time series Transf<sub>ormer</sub> <sub>enco</sub>d<sub>ers,</sub> <sub>revea</sub>li<sub>ng</sub> h<sub>ow</sub> <sub>pa</sub>t<sub>c</sub>hifi<sub>ca</sub>ti<sub>on,</sub> <sub>em</sub>b<sub>e</sub>ddi<sub>ng,</sub> and attention o<sub>p</sub>erations afect re<sub>p</sub>resentation ca<sub>p</sub>acit<sub>y</sub> and <sub>o</sub>f<sub>er</sub>i<sub>ng a</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>exp</sub>l<sub>ana</sub>ti<sub>on</sub> f<sub>or</sub> th<sub>e e</sub>f<sub>ec</sub>ti<sub>veness o</sub>f <sub>pa</sub>t<sub>c</sub>h<sub>-</sub>b<sub>ase</sub>d <sub>mo</sub>d<sub>e</sub>li<sub>ng.</sub>

• We propose RankFormer, consisting of HgA and IPD. HgA <sub>p</sub>reserves ex<sub>p</sub>ressive re<sub>p</sub>resentations throu<sub>g</sub>h im<sub>p</sub>roved attention desi<sub>g</sub>n<sub>,</sub> while IPD ca<sub>p</sub>tures cross-channel de<sub>p</sub>en-<sup>d</sup>encies an<sup>d</sup> progressive<sup>l</sup>y re<sup>fi</sup>nes <sup>f</sup>uture trajectories.

• Extensive ex<sub>p</sub>eriments on widel<sub>y</sub> used lon<sub>g</sub>-term forecasti<sub>ng</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s va</sub>lid<sub>a</sub>t<sub>e</sub> th<sub>e</sub> th<sub>eore</sub>ti<sub>ca</sub>l <sub>o</sub>b<sub>serva</sub>ti<sub>ons an</sub>d demonstrate consistentl<sub>y</sub> stron<sub>g</sub> <sub>p</sub>erformance a<sub>g</sub>ainst com-<sub>pe</sub>titi<sub>ve</sub> <sub>s</sub>t<sub>a</sub>t<sub>e-o</sub>f<sub>-</sub>th<sub>e-ar</sub>t <sub>me</sub>th<sub>o</sub>d<sub>s.</sub>

## 2 Rank Theory Analysis and Key Theorems

I<sub>n</sub> thi<sub>s sec</sub>ti<sub>on, we es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>a ran</sub>k<sub>-</sub>b<sub>ase</sub>d th<sub>eore</sub>ti<sub>ca</sub>l f<sub>ramewor</sub>k t<sub>o</sub> anal<sub>y</sub>ze the core com<sub>p</sub>onents of time series Transformers. We consider the forecastin<sub>g</sub> <sub>p</sub>roblem as learnin<sub>g</sub> a ma<sub>pp</sub>in<sub>g</sub> from historical <sub>o</sub>b<sub>serva</sub>ti<sub>ons</sub> $\mathbf { X } \in \mathbb { R } ^ { L \times C }$ to <sup>f</sup>uture se<sub>q</sub>uences $\mathbf { Y } \in \mathbb { R } ^ { \tau \times C }$ <sub>.</sub> Th<sub>e mo</sub>d<sub>e</sub>l fi<sub>rs</sub>t t<sub>rans</sub>f<sub>orms</sub> th<sub>e</sub> i<sub>npu</sub>t <sub>sequence</sub> i<sub>n</sub>t<sub>o</sub> hidd<sub>en represen</sub>t<sub>a</sub>ti<sub>ons</sub> $\mathbf { H } \in \mathbb { R } ^ { N \times D }$ <sub>, w</sub>h<sub>ose s</sub>t<sub>ruc</sub>t<sub>ure</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>nes</sub> h<sub>ow muc</sub>h i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t information can be <sub>p</sub>reserved for subse<sub>q</sub>uent forecastin<sub>g</sub>. We use matrix rank, rank(H), as a <sub>q</sub>uantitative measure to characterize the inde<sub>p</sub>endent information ca<sub>p</sub>acit<sub>y</sub> of hidden re<sub>p</sub>resentations. A hi<sub>g</sub>her rank indicates more inde<sub>p</sub>endent re<sub>p</sub>resentation directions and <sub>g</sub>reater <sub>p</sub>otential to <sub>p</sub>reserve diverse information<sub>,</sub> whereas rank d<sub>egra</sub>d<sub>a</sub>ti<sub>on</sub> <sub>sugges</sub>t<sub>s</sub> i<sub>ncrease</sub>d <sub>re</sub>d<sub>un</sub>d<sub>ancy</sub> <sub>an</sub>d <sub>re</sub>d<sub>uce</sub>d <sub>represen-</sub> t<sub>a</sub>ti<sub>o</sub>n di<sub>ve</sub>r<sub>s</sub>it<sub>y.</sub>

## 2.1 Patching as a Rank Expansion Mechanism

When a univariate sequence x $\mathbf { \Psi } : \in \mathbb { R } ^ { L }$ is re<sub>p</sub>resented throu<sub>g</sub>h <sub>p</sub>ointwise tokenization<sub>,</sub> the resultin<sub>g</sub> token matrix has a sin<sub>g</sub>le feature di<sub>mens</sub>i<sub>on,</sub> <sub>an</sub>d <sub>any</sub> li<sub>near</sub> <sub>em</sub>b<sub>e</sub>ddi<sub>ng</sub> i<sub>n</sub>h<sub>er</sub>it<sub>s</sub> <sub>a</sub> <sub>ran</sub>k <sub>upper</sub> b<sub>oun</sub>d <sub>o</sub>f <sub>one:</sub>

$$
\mathrm { r a n k } ( \mathbf { x } ^ { \top } \mathbf { W } ) \leq \mathrm { r a n k } ( \mathbf { x } ) \leq 1 .
$$

Proposition 1 (Rank Expansion via Patching). Partitioning x into patches of length � and arranging them into a patch matrix $\mathbf { X } _ { p } \ \in \ \bar { \mathbb { R } } ^ { N \times P }$ , where $N = \lfloor L / P \rfloor$ , gives the attainable rank upper bound:

$$
\mathrm { r a n k } ( { \mathbf { X } } _ { p } ) \leq \operatorname* { m i n } ( N , P ) .
$$

Wh<sub>en</sub> $P > 1$ <sub>,</sub> <sub>pa</sub>t<sub>c</sub>hi<sub>ng</sub> <sub>expan</sub>d<sub>s</sub> th<sub>e</sub> <sub>max</sub>i<sub>mum</sub> <sub>a</sub>tt<sub>a</sub>i<sub>na</sub>bl<sub>e</sub> <sub>ran</sub>k bound from one to min(� , �), increasin<sub>g</sub> the <sub>p</sub>otential number <sub>o</sub>f i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t di<sub>rec</sub>ti<sub>ons</sub> <sub>ava</sub>il<sub>a</sub>bl<sub>e</sub> f<sub>or</sub> <sub>su</sub>b<sub>sequen</sub>t <sub>represen</sub>t<sub>a</sub>ti<sub>on</sub> learnin<sub>g</sub>. This <sub>p</sub>rovides a rank-based <sub>p</sub>ers<sub>p</sub>ective for understandin<sub>g</sub> th<sub>e</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>veness</sub> <sub>o</sub>f <sub>pa</sub>t<sub>c</sub>h<sub>-</sub>b<sub>ase</sub>d <sub>mo</sub>d<sub>e</sub>li<sub>ng</sub> i<sub>n</sub> ti<sub>me</sub> <sub>ser</sub>i<sub>es</sub> T<sub>rans</sub>f<sub>orm-</sub> ers.

## 2.2 Embedding Layer: Breaking Linear Rank Constraints

To ma<sub>p</sub> the <sub>p</sub>atched re<sub>p</sub>resentation $\mathbf { X } _ { p }$ into a hidden re<sub>p</sub>resentation H $\in \mathbb { R } ^ { N \times D }$ , stan<sup>d</sup>ar<sup>d</sup> <sup>l</sup>inear projections $( \mathbf { H } = \mathbf { X } _ { p } \mathbf { W } )$ <sub>p</sub>reserve t<sup>h</sup>e rank constraint of the in<sub>p</sub>ut re<sub>p</sub>resentation:

$$
\operatorname { r a n k } ( \mathbf { H } ) \leq \operatorname* { m i n } ( \operatorname { r a n k } ( \mathbf { X } _ { \rho } ) , D ) \leq \operatorname* { m i n } ( N , P , D ) .
$$

Wh<sub>en</sub> $D > P ,$ li<sub>near em</sub>b<sub>e</sub>ddi<sub>ngs rema</sub>i<sub>n cons</sub>t<sub>ra</sub>i<sub>ne</sub>d b<sub>y</sub> th<sub>e pa</sub>t<sub>c</sub>h<sub>-</sub> i<sub>n</sub>d<sub>uce</sub>d <sub>ran</sub>k b<sub>oun</sub>d<sub>,</sub> li<sub>m</sub>iti<sub>ng</sub> th<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t di<sub>rec</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>can</sub> b<sub>e represen</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> hidd<sub>en space.</sub>

T<sub>o overcome</sub> thi<sub>s</sub> li<sub>near ran</sub>k <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>, we cons</sub>id<sub>er non</sub>li<sub>near</sub> MLP embeddin<sub>g</sub>s:

$$
\mathbf { H } = \phi ( \mathbf { X } _ { \mathcal { P } } \mathbf { W } _ { 1 } + \mathbf { b } _ { 1 } ) \mathbf { W } _ { 2 } + \mathbf { b } _ { 2 } .
$$

Theorem 1 (Maximum Rank Embedding). Let $\phi$ be a continuous, non-polynomial activation function. For any � distinct input patches $\mathbf { X } _ { p } \in \mathbb { R } ^ { \tilde { N } \times P }$ , there exist $M L P$ parameters such that the resulting hidden representation achieves the maximum attainable rank:

$$
\mathrm { r a n k } ( \mathbf { H } ) = \operatorname* { m i n } ( N , D ) .
$$

Thi<sub>s</sub> th<sub>eorem</sub> <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>non</sub>li<sub>near</sub> <sub>em</sub>b<sub>e</sub>ddi<sub>ngs</sub> <sub>overcome</sub> th<sub>e</sub> a<sup>dd</sup>itiona<sup>l</sup> ran<sup>k</sup> restriction impose<sup>d</sup> <sup>b</sup>y <sup>l</sup>inear projections, a<sup>ll</sup>owing

![](images/8f757ae8630f96be6db3f8966b9d5a43ef80beabb5afc67f6566a8d919e5f590.jpg)  
Figure 2: Overall structure of RankFormer. The model preprocesses multivariate time series via Instance Norm and CI, utilizing Patch Embedding and a Multi-Head Hourglass Attention encoder to generate a base prediction. An Interactive Progressive Decoder then refines these features. It trains via one-step residual learning using randomized partial ground truth, while inference employs a �-step progressive refinement from a zero-residual state, concluding with Instance Denorm.

hidd<sub>en represen</sub>t<sub>a</sub>ti<sub>ons</sub> t<sub>o ac</sub>hi<sub>eve</sub> th<sub>e</sub>i<sub>r max</sub>i<sub>mum a</sub>tt<sub>a</sub>i<sub>na</sub>bl<sub>e ran</sub>k d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> th<sub>e samp</sub>l<sub>e an</sub>d hidd<sub>en</sub> di<sub>mens</sub>i<sub>ons.</sub>

## 2.3 Attention Mechanisms: Rank Preservation via Attention Topology

BiA enables <sub>g</sub>lobal information a<sub>gg</sub>re<sub>g</sub>ation across tokens<sub>,</sub> but its dense mixin<sub>g</sub> structure does not <sub>p</sub>rovide an ex<sub>p</sub>licit <sub>g</sub>uarantee for <sub>p</sub>reservin<sub>g</sub> re<sub>p</sub>resentation rank. Re<sub>p</sub>eated dense token mixin<sub>g</sub> ma<sub>y</sub> lead to re<sub>p</sub>resentation homo<sub>g</sub>enization and rank de<sub>g</sub>radation in dee<sub>p</sub> networks [18, 22].

To anal<sub>y</sub>ze how attention to<sub>p</sub>olo<sub>gy</sub> afects rank stabilit<sub>y,</sub> we com<sub>p</sub>are BiA with unidirectional (e.<sub>g</sub>., causal) attention. Let $\mathbf { A } _ { \mathrm { b i d } }$ <sub>an</sub>d $\mathbf { A } _ { \mathrm { c a u s a l } }$ d<sub>eno</sub>t<sub>e</sub> th<sub>e</sub>i<sub>r</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> <sub>we</sub>i<sub>g</sub>ht <sub>ma</sub>t<sub>r</sub>i<sub>ces,</sub> <sub>an</sub>d l<sub>e</sub>t $\mathbf { O } _ { \mathrm { b i d } }$ <sub>an</sub>d $\mathbf { O _ { \mathrm { c a u s a l } } }$ <sup>d</sup>enote t<sup>h</sup>e correspon<sup>d</sup>ing outputs a<sup>f</sup>ter va<sup>l</sup>ue projection an<sup>d</sup> output projection.

Theorem 2 (Rank Preservation of Causal Attention). Given identical projection parameters $( \mathbf { W } _ { Q } , \mathbf { W } _ { K } , \mathbf { W } _ { V } , \mathbf { W } ^ { O } )$ , causal attention preserves the rank of the value representation and guarantees:

$$
\mathrm { r a n k } ( \mathbf { O } _ { \mathrm { c a u s a l } } ) \geq \mathrm { r a n k } ( \mathbf { O } _ { \mathrm { b i d } } ) .
$$

Th<sub>eorem</sub> 2 <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> th<sub>a</sub>t <sub>causa</sub>l <sub>mas</sub>ki<sub>ng prov</sub>id<sub>es a s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sup>rank-</sup>p<sup>reservation</sup> p<sup>ro</sup>p<sup>ert</sup>y <sup>durin</sup>g <sup>attention a</sup>gg<sup>re</sup>g<sup>ation. In con-</sup> trast<sub>,</sub> BiA ma<sub>y</sub> reduce out<sub>p</sub>ut rank when its learned a<sub>gg</sub>re<sub>g</sub>ation fails to <sub>p</sub>reserve inde<sub>p</sub>endent information directions in the value re<sub>p</sub>resentation (as shown in Fi<sub>g</sub>ure 1(a)). This anal<sub>y</sub>sis hi<sub>g</sub>hli<sub>g</sub>hts the im<sub>p</sub>ortance of attention to<sub>p</sub>olo<sub>gy</sub> in maintainin<sub>g</sub> diverse re<sub>p</sub>resentations durin<sub>g</sub> token interaction. Rather than favorin<sub>g</sub> a s<sub>p</sub>ecific to<sub>p</sub>olo<sub>gy,</sub> it motivates attention desi<sub>g</sub>ns that <sub>p</sub>reserve rank stabilit<sub>y</sub> while enablin<sub>g</sub> efective information exchan<sub>g</sub>e<sub>,</sub> leadin<sub>g</sub> to our <sub>p</sub>ro<sub>p</sub>osed Hour<sub>g</sub>lass Attention.

## 3 RankFormer

G<sub>u</sub>id<sub>e</sub>d b<sub>y</sub> th<sub>e ran</sub>k<sub>-</sub>b<sub>ase</sub>d th<sub>eore</sub>ti<sub>ca</sub>l f<sub>ramewor</sub>k <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d <sub>a</sub>b<sub>ove,</sub> we propose RankFormer, as illustrated in Figure 2. The architecture consists of a rank-<sub>g</sub>uided encoder with H<sub>g</sub>A<sub>,</sub> which im<sub>p</sub>roves rank stabilit<sub>y</sub> durin<sub>g</sub> attention-based interaction<sub>,</sub> and an IPD<sub>,</sub> which captures comp<sup>l</sup>ex <sup>f</sup>uture trajectories t<sup>h</sup>roug<sup>h</sup> g<sup>l</sup>o<sup>b</sup>a<sup>l</sup>-context-gui<sup>d</sup>e<sup>d</sup> <sub>cross-c</sub>h<sub>anne</sub>l i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> <sub>an</sub>d <sub>progress</sub>i<sub>ve</sub> <sub>re</sub>fi<sub>nemen</sub>t<sub>.</sub>

## 3.1 Patch Embedding

To miti<sub>g</sub>ate the adverse efects of ma<sub>g</sub>nitude variations across difer-<sub>en</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>es, we</sub> fi<sub>rs</sub>t <sub>app</sub>l<sub>y s</sub>t<sub>an</sub>d<sub>ar</sub>d i<sub>ns</sub>t<sub>ance norma</sub>li<sub>za</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e</sub> input sequence X along the temporal dimension. The normalized time series is then <sub>p</sub>artitioned into <sub>p</sub>atches of len<sub>g</sub>th �<sub>,</sub> transformin<sub>g</sub> the in<sub>pu</sub>t into $\mathbf { \dot { X } } _ { \mathrm { p a t c h e s } } \in \mathbb { R } ^ { C \times N \times P }$ <sub>, w</sub>h<sub>ere</sub> � i<sub>s</sub> th<sub>e num</sub>b<sub>er o</sub>f <sub>p</sub>atches. As elucidated in Pro<sub>p</sub>osition 1<sub>,</sub> this <sub>p</sub>atchin<sub>g</sub> o<sub>p</sub>eration structurall<sub>y</sub> ex<sub>p</sub>ands the rank ca<sub>p</sub>acit<sub>y</sub> of the in<sub>p</sub>ut re<sub>p</sub>resentation.

S<sub>u</sub>b<sub>sequen</sub>tl<sub>y,</sub> t<sub>o</sub> b<sub>rea</sub>k th<sub>e</sub> i<sub>n</sub>h<sub>eren</sub>t <sub>ran</sub>k b<sub>oun</sub>d <sub>o</sub>f li<sub>near pro-</sub> jections, we map t<sup>h</sup>ese patc<sup>h</sup>es into t<sup>h</sup>e <sup>h</sup>i<sup>dd</sup>en space $\mathbb { R } ^ { D }$ us<sup>i</sup>n<sub>g</sub> a two-la<sub>y</sub>er MLP with a ReLU activation [16]:

$$
\mathbf { H } _ { \mathrm { e m b e d } } = \mathrm { R e L U } ( \mathbf { X } _ { \mathrm { p a t c h e s } } \mathbf { W } _ { 1 } + \mathbf { b } _ { 1 } ) \mathbf { W } _ { 2 } + \mathbf { b } _ { 2 }
$$

<sub>w</sub>h<sub>ere</sub> $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { P \times D _ { \mathrm { m i d } } }$ <sub>an</sub>d $\mathbf { W } _ { 2 } \in \mathbb { R } ^ { D _ { \mathrm { m i d } } \times D }$ <sub>.</sub> A<sub>s</sub> m<sub>a</sub>th<sub>e</sub>m<sub>a</sub>ti<sub>ca</sub>ll<sub>y gua</sub>r-<sub>an</sub>t<sub>ee</sub>d b<sub>y</sub> Th<sub>eorem</sub> 1<sub>,</sub> thi<sub>s non</sub>li<sub>near</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>on ensures</sub> th<sub>a</sub>t th<sub>e</sub> embeddin<sub>g</sub> achieves the theoretical maximum rank of min(�, �), f<sub>u</sub>ll<sub>y</sub> <sub>u</sub>tili<sub>z</sub>i<sub>ng</sub> th<sub>e</sub> <sub>represen</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>po</sub>t<sub>en</sub>ti<sub>a</sub>l <sub>o</sub>f th<sub>e</sub> hi<sub>g</sub>h<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l hidd<sub>en space.</sub>

## 3.2 Hourglass Attention

BiA enables <sub>g</sub>lobal information a<sub>gg</sub>re<sub>g</sub>ation across tokens<sub>,</sub> but re-<sub>p</sub>eated dense mixin<sub>g</sub> ma<sub>y</sub> cause re<sub>p</sub>resentation homo<sub>g</sub>enization and <sub>ran</sub>k d<sub>egra</sub>d<sub>a</sub>ti<sub>on</sub> i<sub>n</sub> d<sub>eep</sub> <sub>ne</sub>t<sub>wor</sub>k<sub>s.</sub> I<sub>n</sub> <sub>con</sub>t<sub>ras</sub>t<sub>,</sub> <sub>causa</sub>l <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> <sub>p</sub>rovides a structural rank-<sub>p</sub>reservation <sub>g</sub>uarantee throu<sub>g</sub>h its trian-<sub>g</sub>ular to<sub>p</sub>olo<sub>gy,</sub> but its directional information flow limits eficient <sub>con</sub>t<sub>ex</sub>t <sub>exc</sub>h<sub>ange</sub> <sub>across</sub> t<sub>o</sub>k<sub>ens.</sub> T<sub>o</sub> <sub>com</sub>bi<sub>ne</sub> th<sub>ese</sub> <sub>comp</sub>l<sub>emen</sub>t<sub>ary</sub> p<sup>ro</sup>p<sup>erties</sup>, <sup>we</sup> p<sup>ro</sup>p<sup>ose</sup> <sup>H</sup>g<sup>A</sup>, <sup>a</sup> <sup>to</sup>p<sup>olo</sup>gy<sup>-aware</sup> <sup>attention</sup> <sup>mecha-</sup> nism that reor<sub>g</sub>anizes ca<sub>u</sub>sal connecti<sub>v</sub>it<sub>y</sub> to maintain rank stabilit<sub>y</sub> while enablin<sub>g</sub> more efective information interaction.

![](images/9fc180b085f97f82f941b4f3201c5039341824d4752bd8430a421bee60e9f9f6.jpg)  
Figure 3: Internal architecture of the Interactive Progressive Decoder.

H<sub>g</sub>A introduces an hour<sub>g</sub>lass-sha<sub>p</sub>ed attention to<sub>p</sub>olo<sub>gy</sub> throu<sub>g</sub>h three desi<sub>g</sub>n <sub>p</sub>rinci<sub>p</sub>les. First<sub>,</sub> earl<sub>y</sub> tokens are assi<sub>g</sub>ned broader <sub>recep</sub>ti<sub>ve</sub> fi<sub>e</sub>ld<sub>s</sub> t<sub>o aggrega</sub>t<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>con</sub>t<sub>ex</sub>t f<sub>rom</sub> th<sub>e o</sub>b<sub>serve</sub>d <sub>se-</sub> <sub>quence.</sub> S<sub>econ</sub>d<sub>,</sub> i<sub>n</sub>t<sub>erme</sub>di<sub>a</sub>t<sub>e</sub> t<sub>o</sub>k<sub>ens</sub> <sub>use</sub> <sub>res</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d <sub>recep</sub>ti<sub>ve</sub> fi<sub>e</sub>ld<sub>s</sub> t<sub>o</sub> r<sub>e</sub>d<sub>uce e</sub>x<sub>cess</sub>i<sub>ve</sub> t<sub>o</sub>k<sub>e</sub>n mixin<sub>g a</sub>nd m<sub>a</sub>int<sub>a</sub>in r<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n di <sub>vers</sub>it<sub>y.</sub> Thi<sub>r</sub>d<sub>,</sub> <sub>recep</sub>ti<sub>ve</sub> fi<sub>e</sub>ld<sub>s</sub> <sub>gra</sub>d<sub>ua</sub>ll<sub>y</sub> <sub>expan</sub>d t<sub>owar</sub>d l<sub>a</sub>t<sub>er</sub> t<sub>o</sub>k<sub>ens,</sub> <sub>p</sub>reservin<sub>g</sub> detailed recent information for downstream forecastin<sub>g</sub>. To<sub>g</sub>ether<sub>,</sub> these desi<sub>g</sub>ns redistribute information access within the se<sub>q</sub>uence while retainin<sub>g</sub> the rank-stabilit<sub>y p</sub>ro<sub>p</sub>ert<sub>y</sub> inherited from <sub>causa</sub>l <sub>a</sub>tt<sub>en</sub>ti<sub>on.</sub>

H<sub>g</sub>A is constructed b<sub>y</sub> a to<sub>p</sub>olo<sub>gy</sub>-<sub>p</sub>reservin<sub>g</sub> reorderin<sub>g</sub> of causal attention. For an in<sub>p</sub>ut se<sub>q</sub>uence of len<sub>g</sub>th �, we define <sub>�</sub>(�) as the maximum visible ke<sub>y</sub> index of <sub>q</sub>uer<sub>y p</sub>osition � under this reorder-<sup>i</sup>n<sub>g</sub>:

$$
\rho ( i ) = \left\{ \begin{array} { l l } { \operatorname* { m a x } ( N - 2 - 2 i , 0 ) , } & { i \leq \lfloor ( N - 1 ) / 2 \rfloor , } \\ { 2 i - N + 1 , } & { i > \lfloor ( N - 1 ) / 2 \rfloor . } \end{array} \right.\tag{1}
$$

Thi<sub>s</sub> <sub>mapp</sub>i<sub>ng</sub> <sub>crea</sub>t<sub>es</sub> <sub>an</sub> h<sub>ourg</sub>l<sub>ass-s</sub>h<sub>ape</sub>d <sub>recep</sub>ti<sub>ve</sub> fi<sub>e</sub>ld<sub>:</sub> th<sub>e</sub> <sub>v</sub>i<sub>s</sub>ibl<sub>e range con</sub>t<sub>rac</sub>t<sub>s</sub> t<sub>owar</sub>d th<sub>e sequence cen</sub>t<sub>er an</sub>d <sub>expan</sub>d<sub>s</sub> t<sub>owar</sub>d th<sub>e sequence en</sub>d<sub>.</sub> Th<sub>e correspon</sub>di<sub>ng mas</sub>k i<sub>s:</sub>

$$
\mathbf { M } ^ { \mathrm { H g } } [ i , j ] = \left\{ { \begin{array} { l l } { 0 , } & { j \leq \rho ( i ) , } \\ { - \infty , } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.\tag{2}
$$

Th<sub>e</sub> <sub>a</sub>tt<sub>e</sub>nti<sub>o</sub>n <sub>ope</sub>r<sub>a</sub>ti<sub>o</sub>n i<sub>s</sub>:

$$
\boldsymbol { \mathbf { A } } ^ { \mathrm { H g } } = \mathrm { S o f t m a x } \left( \frac { \boldsymbol { \mathbf { Q } } \boldsymbol { \mathbf { K } } ^ { \top } } { \sqrt { d _ { k } } } + \boldsymbol { \mathbf { M } } ^ { \mathrm { H g } } \right) ,\tag{3}
$$

$$
\mathrm { H g A t t n } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = \mathbf { A } ^ { \mathrm { H g } } \mathbf { V } .\tag{4}
$$

Since H<sub>g</sub>A is obtained throu<sub>g</sub>h a reorderin<sub>g</sub> of causal connectivit<sub>y,</sub> its attention matrix retains the rank-<sub>p</sub>reservation <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> of

<sub>causa</sub>l <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> <sub>w</sub>hil<sub>e</sub> <sub>prov</sub>idi<sub>ng</sub> <sub>a</sub> dif<sub>eren</sub>t <sub>recep</sub>ti<sub>ve</sub> fi<sub>e</sub>ld <sub>a</sub>ll<sub>oca-</sub> tion.

## 3.3 Interactive Progressive Decoder

The IPD (Fi<sub>g</sub>ure 3) addresses the ma<sub>pp</sub>in<sub>g</sub> bottleneck b<sub>y</sub> transformin<sub>g</sub> channel-inde<sub>p</sub>endent encoder re<sub>p</sub>resentations into multivariate f<sub>orecas</sub>t<sub>s.</sub> It fi<sub>rs</sub>t <sub>genera</sub>t<sub>es</sub> <sub>a</sub> <sub>c</sub>h<sub>anne</sub>l<sub>-w</sub>i<sub>se</sub> b<sub>ase</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on,</sub> th<sub>en</sub> intr<sub>o</sub>d<sub>uces c</sub>r<sub>oss</sub>-<sub>c</sub>h<sub>a</sub>nn<sub>e</sub>l <sub>co</sub>nditi<sub>o</sub>nin<sub>g</sub> t<sub>o</sub> in<sub>co</sub>r<sub>po</sub>r<sub>a</sub>t<sub>e</sub> int<sub>e</sub>r<sub>ac</sub>ti<sub>o</sub>n<sub>s</sub> <sub>among</sub> <sub>var</sub>i<sub>a</sub>bl<sub>es,</sub> <sub>an</sub>d fi<sub>na</sub>ll<sub>y</sub> <sub>re</sub>fi<sub>nes</sub> th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> th<sub>roug</sub>h <sub>progres-</sub> <sub>s</sub>i<sub>ve</sub> <sub>res</sub>id<sub>ua</sub>l <sub>up</sub>d<sub>a</sub>t<sub>es.</sub>

Base Prediction. IPD first obtains a channel-independent base predi<sub>c</sub>ti<sub>on</sub> f<sub>rom</sub> th<sub>e</sub> fl<sub>a</sub>tt<sub>ene</sub>d <sub>enco</sub>d<sub>er</sub> <sub>represen</sub>t<sub>a</sub>ti<sub>ons</sub> <sub>us</sub>i<sub>ng</sub> <sub>a</sub> <sub>s</sub>h<sub>are</sub>d <sup>l</sup>inear projection. We <sup>d</sup>enote t<sup>h</sup>e initia<sup>l</sup> pre<sup>d</sup>iction o<sup>f</sup> c<sup>h</sup>anne<sup>l</sup> � as $\mathbf { y } _ { C } ^ { \mathrm { b a s e } } \in \mathbb { R } ^ { \tau }$ <sub>,</sub> <sub>w</sub>here <sub>�</sub> is the forecast horizon. This base <sub>p</sub>rediction <sub>p</sub>rovides the initialization for subse<sub>q</sub>uent cross-channel conditionin<sub>g</sub> <sub>an</sub>d <sub>progress</sub>i<sub>ve res</sub>id<sub>ua</sub>l <sub>re</sub>fi<sub>nemen</sub>t<sub>.</sub>

Cross-channel Conditioning. To introduce multivariate interactions while <sub>p</sub>reservin<sub>g</sub> channel-s<sub>p</sub>ecific information<sub>,</sub> IPD constructs an interaction-aware conditionin<sub>g</sub> re<sub>p</sub>resentation for each channel. T<sup>h</sup>e <sup>b</sup>ase pre<sup>d</sup>iction is <sup>fi</sup>rst projecte<sup>d</sup> into a <sup>f</sup>eature space:

$$
\mathbf { u } _ { C } = \mathrm { G E L U } ( \mathbf { P y } _ { C } ^ { \mathrm { b a s e } } + \mathbf { b } _ { P } ) .\tag{5}
$$

A l<sub>earna</sub>bl<sub>e c</sub>h<sub>anne</sub>l <sub>em</sub>b<sub>e</sub>ddi<sub>ng</sub> ${ \bf e } _ { C }$ i<sub>s com</sub>bi<sub>ne</sub>d <sub>w</sub>ith th<sub>e pre</sub>di<sub>c-</sub> ti<sub>on</sub> f<sub>ea</sub>t<sub>ure:</sub>

$$
\mathbf { f } _ { C } = [ \mathbf { u } _ { C } ; \mathbf { e } _ { C } ] ,\tag{6}
$$

<sub>w</sub>h<sub>ere</sub> $\mathbf { f } _ { C }$ <sub>represen</sub>t<sub>s</sub> th<sub>e c</sub>h<sub>anne</sub>l<sub>-spec</sub>ifi<sub>c con</sub>diti<sub>on</sub>i<sub>ng</sub> f<sub>ea</sub>t<sub>ure.</sub>

Th<sub>e cross-c</sub>h<sub>anne</sub>l b<sub>ranc</sub>h t<sub>rans</sub>f<sub>orms eac</sub>h <sub>c</sub>h<sub>anne</sub>l <sub>represen</sub>t<sub>a-</sub> tion into a messa<sub>g</sub>e feat<sub>u</sub>re:

$$
\mathbf { m } _ { j } = \mathbf { W } _ { m } \mathbf { f } _ { j } + \mathbf { b } _ { m } .\tag{7}
$$

F<sub>or c</sub>h<sub>anne</sub>l �<sub>,</sub> th<sub>e rece</sub>i<sub>ve</sub>d i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> f<sub>rom o</sub>th<sub>er var</sub>i<sub>a</sub>bl<sub>es</sub> i<sub>s</sub> <sup>a</sup>gg<sup>re</sup>g<sup>ated</sup> <sup>as:</sup>

$$
\mathbf { m } _ { C } ^ { \mathrm { c r o s s } } = \sum _ { j \neq C } \mathbf { M } _ { C j } \mathbf { m } _ { j } ,\tag{8}
$$

where M denotes the learnable channel interaction matrix.

Table 1: Results of the Long-Term Time Series Forecasting. All baseline models were configured consistently with SEER. The reported results represent the average performance over four distinct forecasting horizons: {96, 192, 336, 720}.
<table><tr><td>Models</td><td colspan="2">RankFormer (ours)</td><td colspan="2">SEER (2026)</td><td colspan="2">SRSNet (2025)</td><td colspan="2">DUET (2025)</td><td colspan="2">xPatch (2025)</td><td colspan="2">Amplifier (2025)</td><td colspan="2">FredFormer (2024)</td><td colspan="2">iTransformer (2024)</td><td colspan="2">PatchTST (2023)</td><td colspan="2">DLinear (2023)</td></tr><tr><td>Metrics</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.410</td><td>0.421</td><td>0.423</td><td>0.424</td><td>0.442</td><td>0.433</td><td>0.443</td><td>0.436</td><td>0.444</td><td>0.429</td><td>0.449</td><td>0.437</td><td>0.448</td><td>0.435</td><td>0.454</td><td>0.448</td><td>0.469</td><td>0.455</td><td>0.461</td><td>0.458</td></tr><tr><td>ETTh2</td><td>0.354</td><td>0.381</td><td>0.356</td><td>0.384</td><td>0.376</td><td>0.402</td><td>0.372</td><td>0.398</td><td>0.369</td><td>0.392</td><td>0.389</td><td>0.411</td><td>0.378</td><td>0.403</td><td>0.383</td><td>0.407</td><td>0.387</td><td>0.407</td><td>0.563</td><td>0.519</td></tr><tr><td>ETTm1</td><td>0.367</td><td>0.377</td><td>0.377</td><td>0.381</td><td>0.385</td><td>0.395</td><td>0.390</td><td>0.393</td><td>0.387</td><td>0.383</td><td>0.383</td><td>0.396</td><td>0.385</td><td>0.398</td><td>0.407</td><td>0.410</td><td>0.387</td><td>0.400</td><td>0.404</td><td>0.408</td></tr><tr><td>ETTm2</td><td>0.268</td><td>0.311</td><td>0.276</td><td>0.317</td><td>0.284</td><td>0.329</td><td>0.280</td><td>0.324</td><td>0.280</td><td>0.319</td><td>0.280</td><td>0.326</td><td>0.281</td><td>0.324</td><td>0.288</td><td>0.332</td><td>0.281</td><td>0.326</td><td>0.354</td><td>0.402</td></tr><tr><td>Weather</td><td>0.234</td><td>0.256</td><td>0.246</td><td>0.265</td><td>0.250</td><td>0.277</td><td>0.251</td><td>0.273</td><td>0.247</td><td>0.266</td><td>0.247</td><td>0.275</td><td>0.256</td><td>0.278</td><td>0.258</td><td>0.278</td><td>0.259</td><td>0.281</td><td>0.265</td><td>0.315</td></tr><tr><td>Solar</td><td>0.212</td><td>0.217</td><td>0.221</td><td>0.224</td><td>0.250</td><td>0.281</td><td>0.237</td><td>0.233</td><td>0.277</td><td>0.268</td><td>0.222</td><td>0.256</td><td>0.227</td><td>0.267</td><td>0.233</td><td>0.262</td><td>0.270</td><td>0.307</td><td>0.330</td><td>0.401</td></tr><tr><td>Electricity</td><td>0.162</td><td>0.246</td><td>0.169</td><td>0.251</td><td>0.189</td><td>0.276</td><td>0.172</td><td>0.259</td><td>0.184</td><td>0.266</td><td>0.174</td><td>0.268</td><td>0.178</td><td>0.270</td><td>0.178</td><td>0.270</td><td>0.216</td><td>0.304</td><td>0.225</td><td>0.319</td></tr><tr><td>Traffic</td><td>0.431</td><td>0.249</td><td>0.422</td><td>0.245</td><td>0.494</td><td>0.307</td><td>0.451</td><td>0.269</td><td>0.500</td><td>0.283</td><td>0.485</td><td>0.315</td><td>0.434</td><td>0.289</td><td>0.428</td><td>0.282</td><td>0.555</td><td>0.362</td><td>0.625</td><td>0.383</td></tr><tr><td>1stCount</td><td>7</td><td>7</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

T<sup>h</sup>e own-c<sup>h</sup>anne<sup>l</sup> <sup>f</sup>eature an<sup>d</sup> cross-c<sup>h</sup>anne<sup>l</sup> message joint<sup>l</sup>y <sub>g</sub>enerate horizon-wise modulation <sub>p</sub>arameters:

$$
\left[ \pmb { \gamma } _ { C } \right] = \mathbf { G } \mathbf { f } _ { C } + \lambda \mathbf { K } \mathbf { m } _ { C } ^ { \mathrm { c r o s s } } + \mathbf { b } _ { G } .\tag{9}
$$

The conditioned <sub>p</sub>rediction is obtained throu<sub>g</sub>h ReZero modulation:

$$
{ \bf y } _ { C } ^ { \mathrm { c o n d } } = { \bf y } _ { C } ^ { \mathrm { b a s e } } + \rho _ { c } ( { \bf y } _ { C } ^ { \mathrm { b a s e } } \odot \gamma _ { C } + \beta _ { C } ) ,\tag{10}
$$

<sub>w</sub>h<sub>ere</sub> $\rho _ { c }$ i<sub>s</sub> initi<sub>a</sub>liz<sub>e</sub>d t<sub>o</sub> z<sub>e</sub>r<sub>o.</sub>

Progressive Residual Refinement. Instead of directly mapping en coder re<sub>p</sub>resentations to the future se<sub>q</sub>uence throu<sub>g</sub>h a static <sub>p</sub>rojection, IPD mo<sup>d</sup>e<sup>l</sup>s <sup>f</sup>orecasting as resi<sup>d</sup>ua<sup>l</sup> re<sup>fi</sup>nement <sup>f</sup>rom an int<sub>e</sub>rm<sub>e</sub>di<sub>a</sub>t<sub>e p</sub>r<sub>e</sub>di<sub>c</sub>ti<sub>o</sub>n <sub>s</sub>t<sub>a</sub>t<sub>e.</sub>

D<sub>u</sub>rin<sub>g</sub> tr<sub>a</sub>inin<sub>g, a co</sub>m<sub>p</sub>l<sub>e</sub>ti<sub>o</sub>n r<sub>a</sub>ti<sub>o</sub> $r \sim \mathcal { U } ( 0 , 1 )$ i<sub>s</sub> <sub>samp</sub>l<sub>e</sub>d t<sub>o</sub> construct:

$$
{ \bf y } _ { C } ^ { \mathrm { m i d } } = r { \bf y } _ { C } + \sigma \epsilon _ { C } , \qquad \epsilon _ { C } \sim N ( 0 , { \bf I } ) .\tag{11}
$$

<sub>w</sub>h<sub>ere</sub> $\mathbf { y } _ { C }$ i<sub>s</sub> th<sub>e</sub> <sub>norma</sub>li<sub>ze</sub>d <sub>groun</sub>d<sub>-</sub>t<sub>ru</sub>th <sub>sequence</sub> <sub>an</sub>d <sub>�</sub> <sub>con</sub>t<sub>ro</sub>l<sub>s</sub> the noise ma<sub>g</sub>nitude.

The resid<sub>u</sub>al correction is <sub>p</sub>redicted as:

$$
\Delta \mathbf { y } _ { C } = \mathbf { W } _ { \mathrm { s t a t e } } \mathbf { y } _ { C } ^ { \mathrm { m i d } } + \mathbf { W } _ { \mathrm { c o n d } } \mathbf { y } _ { C } ^ { \mathrm { c o n d } } + \mathbf { b } .\tag{12}
$$

The final trainin<sub>g</sub> <sub>p</sub>rediction is:

$$
\begin{array} { r } { \hat { \mathbf { y } } _ { C } = \mathbf { y } _ { C } ^ { \mathrm { m i d } } + ( 1 - r ) \Delta \mathbf { y } _ { C } . } \end{array}\tag{13}
$$

Ofline-folded Inference. During inference, IPD initializes the predi<sub>c</sub>ti<sub>o</sub>n <sub>s</sub>t<sub>a</sub>t<sub>e as</sub> ${ \bf y } _ { C } ^ { ( 0 ) } = { \bf 0 }$ <sub>an</sub>d <sub>per</sub>f<sub>orms</sub> � <sub>res</sub>id<sub>ua</sub>l <sub>up</sub>d<sub>a</sub>t<sub>es:</sub>

$$
{ \bf y } _ { C } ^ { ( t ) } = { \bf y } _ { C } ^ { ( t - 1 ) } + \frac { 1 } { T } \left( { \bf W } _ { \mathrm { s t a t e } } { \bf y } _ { C } ^ { ( t - 1 ) } + { \bf W } _ { \mathrm { c o n d } } { \bf y } _ { C } ^ { \mathrm { c o n d } } + { \bf b } \right) , \quad t = 1 , \ldots , T .\tag{14}
$$

Si<sub>nce</sub> th<sub>e</sub> <sub>res</sub>id<sub>ua</sub>l <sub>up</sub>d<sub>a</sub>t<sub>e</sub> i<sub>s</sub> <sub>a</sub>fi<sub>ne</sub> i<sub>n</sub> th<sub>e</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>an</sub>d ${ \bf y } _ { C } ^ { \mathrm { c o n d } }$ <sub>rema</sub>i<sub>ns</sub> fi<sub>xe</sub>d <sub>across</sub> it<sub>era</sub>ti<sub>ons,</sub> th<sub>e</sub> � <sub>up</sub>d<sub>a</sub>t<sub>es</sub> <sub>can</sub> b<sub>e</sub> <sub>ana</sub>l<sub>y</sub>ti<sub>ca</sub>ll<sub>y</sub> <sub>com</sub>bi<sub>ne</sub>d<sub>.</sub> S<sub>pec</sub>ifi<sub>ca</sub>ll<sub>y,</sub> d<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub>

$$
{ \bf S } _ { T } = \frac { 1 } { T } \sum _ { k = 0 } ^ { T - 1 } \left( { \bf I } + \frac { { \bf W } _ { \mathrm { s t a t e } } } { T } \right) ^ { k } ,\tag{15}
$$

th<sub>e</sub> fi<sub>na</sub>l <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> b<sub>ecomes</sub>

$$
\mathbf { y } _ { C } ^ { ( T ) } = \mathbf { B } _ { \mathrm { e f f } } \mathbf { y } _ { C } ^ { \mathrm { c o n d } } + \mathbf { b } _ { \mathrm { e f f } } ,\tag{16}
$$

<sub>w</sub>h<sub>ere</sub> $\mathbf { B } _ { \mathrm { e f f } } = \mathbf { S } _ { T } \mathbf { W } _ { \mathrm { c o n d } }$ <sub>an</sub>d $\mathbf { b } _ { \mathrm { e f f } } = \mathbf { S } _ { T } \mathbf { b }$ . These e<sub>q</sub>uivalent <sub>p</sub>arameters are <sub>p</sub>recom<sub>p</sub>uted after trainin<sub>g,</sub> allowin<sub>g</sub> IPD to re<sub>p</sub>roduce the <sub>p</sub>ro<sub>g</sub>ressive refinement throu<sub>g</sub>h a sin<sub>g</sub>le afine ma<sub>pp</sub>in<sub>g</sub> durin<sub>g</sub> i<sub>n</sub>f<sub>erence.</sub>

## 4 Experiments

W<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub> R<sub>an</sub>kF<sub>ormer on</sub> di<sub>verse rea</sub>l<sub>-wor</sub>ld l<sub>ong-</sub>t<sub>erm</sub> ti<sub>me</sub> series forecastin<sub>g</sub> benchmarks s<sub>p</sub>annin<sub>g</sub> Web- and IoT-relevant a<sub>pp</sub>lications<sub>,</sub> demonstratin<sub>g</sub> its su<sub>p</sub>erior re<sub>p</sub>resentational ca<sub>p</sub>acit<sub>y</sub> and stable <sub>p</sub>ro<sub>g</sub>ressive decodin<sub>g</sub> a<sub>g</sub>ainst state-of-the-art (SOTA) b<sub>ase</sub>li<sub>nes.</sub>

Datasets. We evaluate our model on nine widely adopted real-<sub>wor</sub>ld b<sub>enc</sub>h<sub>mar</sub>k<sub>s:</sub> El<sub>ec</sub>t<sub>r</sub>i<sub>c</sub>it<sub>y,</sub> T<sub>ra</sub>fi<sub>c,</sub> W<sub>ea</sub>th<sub>er,</sub> E<sub>xc</sub>h<sub>ange-</sub>R<sub>a</sub>t<sub>e,</sub> Solar, and four ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2). These datasets co<sub>v</sub>er di<sub>v</sub>erse a<sub>pp</sub>lications rele<sub>v</sub>ant to Web and IoT s<sub>y</sub>stems<sub>,</sub> includin<sub>g</sub> trans<sub>p</sub>ortation<sub>,</sub> ener<sub>gy</sub> mana<sub>g</sub>ement<sub>,</sub> and environmental monitorin<sub>g,</sub> alon<sub>g</sub>side financial forecastin<sub>g</sub>.

Baselines. We benchmark RankFormer against 9 recent SOTA f<sub>orecas</sub>ti<sub>ng</sub> <sub>mo</sub>d<sub>e</sub>l<sub>s.</sub> T<sub>o</sub> <sub>ensure</sub> <sub>a</sub> <sub>compre</sub>h<sub>ens</sub>i<sub>ve</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on,</sub> th<sub>ese</sub> b<sub>ase</sub>li<sub>nes</sub> <sub>encompass</sub> <sub>a</sub>d<sub>vance</sub>d T<sub>rans</sub>f<sub>ormer-</sub>b<sub>ase</sub>d <sub>arc</sub>hit<sub>ec</sub>t<sub>ures</sub> (SEER [20], iTransformer [12], PatchTST [17], Fredformer [19]), linear-centric networks (DLinear [36]), and other hi<sub>g</sub>hl<sub>y</sub> com<sub>p</sub>etitive contem<sub>p</sub>orar<sub>y</sub> a<sub>pp</sub>roaches (SRSNet [31], DUET [21], xPatch [24], Am<sub>p</sub>lifier [6]).

Experimental Setup. We evaluate all models under standard longterm forecastin<sub>g</sub> settin<sub>g</sub>s<sub>,</sub> $\tau \in \{ 9 6 ,$ 192, 336, 720}. Performance is measured b<sub>y</sub> Mean S<sub>q</sub>uared Error (MSE) and Mean Absolute Error (MAE). Baseline results are taken from ori<sub>g</sub>inal <sub>p</sub>ublications or re<sub>p</sub>roduced usin<sub>g</sub> oficial im<sub>p</sub>lementations under the same settin<sub>g</sub>s.

## 4.1 Main Results

A<sub>s s</sub>h<sub>own</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 1<sub>, we eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e</sub> l<sub>ong-</sub>t<sub>erm</sub> f<sub>orecas</sub>ti<sub>ng per</sub>f<sub>or-</sub> m<sub>a</sub>n<sub>ce o</sub>f R<sub>a</sub>nkF<sub>o</sub>rm<sub>e</sub>r <sub>aga</sub>in<sub>s</sub>t SOTA b<sub>ase</sub>lin<sub>es.</sub> All m<sub>o</sub>d<sub>e</sub>l<sub>s use a</sub> fi<sub>xe</sub>d i<sub>npu</sub>t <sub>sequence</sub> l<sub>eng</sub>th <sub>o</sub>f 96<sub>,</sub> <sub>w</sub>ith <sub>resu</sub>lt<sub>s</sub> <sub>average</sub>d <sub>across</sub> f<sub>our</sub> forecastin<sub>g</sub> horizons. RankFormer demonstrates com<sub>p</sub>rehensive su-<sub>p</sub>eriorit<sub>y,</sub> achievin<sub>g</sub> the best MSE and MAE on 7 out of 8 benchmark datasets (securin<sub>g</sub> 14 out of 16 to<sub>p</sub> metrics). While the recent SEER <sub>mo</sub>d<sub>e</sub>l <sub>s</sub>h<sub>ows com e</sub>titi<sub>ve er</sub>f<sub>ormance an</sub>d <sub>ac</sub>hi<sub>eves</sub> th<sub>e</sub> b<sub>es</sub>t <sub>resu</sub>lt<sub>s</sub> <sub>on</sub> th<sub>e</sub> T<sub>ra</sub>fi<sub>c</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>,</sub> R<sub>an</sub>kF<sub>ormer ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>ns a s</sub>t<sub>rong secon</sub>d <sub>p</sub>l<sub>ace</sub> <sub>w</sub>ith <sub>marg</sub>i<sub>na</sub>l dif<sub>erences.</sub> C<sub>ompare</sub>d <sub>w</sub>ith <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d b<sub>ase</sub>li<sub>nes</sub> <sub>suc</sub>h <sub>as</sub> P<sub>a</sub>t<sub>c</sub>hTST <sub>a</sub>nd iTr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>e</sub>r<sub>,</sub> R<sub>a</sub>nkF<sub>o</sub>rm<sub>e</sub>r <sub>ac</sub>hi<sub>eves</sub> l<sub>owe</sub>r forecastin<sub>g</sub> errors across diverse a<sub>pp</sub>lications relevant to Web and IoT s<sub>y</sub>stems, includin<sub>g</sub> ener<sub>gy</sub> mana<sub>g</sub>ement (Solar) and environmen tal monitorin<sub>g</sub> (Weather). These results demonstrate the stron<sub>g</sub> and consistent forecastin<sub>g</sub> <sub>p</sub>erformance of RankFormer across diverse <sub>rea</sub>l<sub>-wor</sub>ld <sub>scenar</sub>i<sub>os.</sub>

## 4.2 Representation Capacity Analysis

![](images/e76c48b693f4bd677f9717911113781dd4ca4fbf5817a12420cc15dd7a5308a3.jpg)

![](images/0b2723cf8b4d9404df89cc53e757503e4f66b378fb6df077c2c640ad7acd323a.jpg)  
Figure 4: Predictive benefits of additional embedding rank capacity. Lines show three-seed means, small dots show individual seeds, and dashed curves show the MSE lower bound for rank-at-most-five predictions.

4.2.1 Predictive Value ofEmbedding Rank Expansion. We construct <sub>syn</sub>th<sub>e</sub>ti<sub>c</sub> t<sub>arge</sub>t<sub>s w</sub>ith <sub>con</sub>t<sub>ro</sub>ll<sub>e</sub>d <sub>ran</sub>k t<sub>o eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e pre</sub>di<sub>c</sub>ti<sub>ve</sub> b<sub>ene</sub>fit <sub>o</sub>f <sub>non</sub>li<sub>near em</sub>b<sub>e</sub>ddi<sub>ngs.</sub> W<sub>e compare</sub> li<sub>near, parame</sub>t<sub>er-</sub> <sub>ma</sub>t<sub>c</sub>h<sub>e</sub>d <sub>a</sub>fi<sub>ne, an</sub>d MLP <sub>em</sub>b<sub>e</sub>ddi<sub>ngs us</sub>i<sub>ng</sub> id<sub>en</sub>ti<sub>ca</sub>l <sub>a</sub>tt<sub>en</sub>ti<sub>on an</sub>d <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> h<sub>ea</sub>d<sub>s</sub> th<sub>a</sub>t <sub>preserve</sub> th<sub>e</sub> <sub>em</sub>b<sub>e</sub>ddi<sub>ng</sub> <sub>ran</sub>k <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s.</sub> Both afine embeddin<sub>g</sub>s have a rank u<sub>pp</sub>er bound of five. Fi<sub>g</sub>ure 4(a) <sub>s</sub>h<sub>ows compara</sub>bl<sub>e per</sub>f<sub>ormance a</sub>t t<sub>arge</sub>t <sub>ran</sub>k 4<sub>.</sub> At hi<sub>g</sub>h<sub>er</sub> t<sub>arge</sub>t <sub>ran</sub>k<sub>s,</sub> th<sub>e</sub> MLP <sub>ac</sub>hi<sub>eves</sub> MSE b<sub>e</sub>l<sub>ow</sub> th<sub>e ran</sub>k<sub>-</sub>fi<sub>ve</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>,</sub> while afine embeddin<sub>g</sub>s remain constrained. In (b), increasin<sub>g</sub> the <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f hi<sub>g</sub>h<sub>er-ran</sub>k t<sub>arge</sub>t <sub>componen</sub>t<sub>s su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> d<sub>e-</sub> <sub>g</sub>r<sub>a</sub>d<sub>es</sub> th<sub>e</sub> r<sub>a</sub>nk-<sub>co</sub>n<sub>s</sub>tr<sub>a</sub>in<sub>e</sub>d MLP<sub>, w</sub>h<sub>e</sub>r<sub>eas</sub> th<sub>e u</sub>n<sub>co</sub>n<sub>s</sub>tr<sub>a</sub>in<sub>e</sub>d MLP <sub>ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>ns s</sub>t<sub>a</sub>bl<sub>e per</sub>f<sub>ormance.</sub> Th<sub>ese resu</sub>lt<sub>s</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>a</sub>t <sub>non-</sub> li<sub>near</sub> <sub>em</sub>b<sub>e</sub>ddi<sub>ngs</sub> <sub>can</sub> <sub>exp</sub>l<sub>o</sub>it hi<sub>g</sub>h<sub>er</sub> <sub>a</sub>tt<sub>a</sub>i<sub>na</sub>bl<sub>e</sub> <sub>ran</sub>k t<sub>o</sub> i<sub>mprove</sub> <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> <sub>w</sub>h<sub>en</sub> t<sub>arge</sub>t <sub>s</sub>t<sub>ruc</sub>t<sub>ures</sub> <sub>excee</sub>d th<sub>e</sub> li<sub>near</sub> <sub>ran</sub>k <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>.</sub>

4.2.2 Impact ofRepresentation Rank Constraints. To investigate the im<sub>p</sub>act of re<sub>p</sub>resentation rank constraints on forecastin<sub>g,</sub> we im<sub>p</sub>ose rank limits of 1<sub>,</sub> 4<sub>,</sub> and 8 on <sub>p</sub>osition-enriched <sub>p</sub>atch re<sub>p</sub>re-<sub>sen</sub>t<sub>a</sub>ti<sub>ons</sub> i<sub>n a s</sub>t<sub>an</sub>d<sub>ar</sub>d T<sub>rans</sub>f<sub>ormer enco</sub>d<sub>er.</sub> Th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s are</sub> maintained durin<sub>g</sub> trainin<sub>g</sub> and inference<sub>,</sub> with all other settin<sub>g</sub>s <sub>unc</sub>h<sub>ange</sub>d<sub>.</sub> T<sub>a</sub>bl<sub>e</sub> 2 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>ran</sub>k<sub>-one</sub> <sub>an</sub>d <sub>ran</sub>k<sub>-</sub>f<sub>our</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s</sub> <sub>co</sub>n<sub>s</sub>i<sub>s</sub>t<sub>e</sub>ntl<sub>y</sub> in<sub>c</sub>r<sub>ease</sub> MSE <sub>a</sub>nd MAE <sub>ac</sub>r<sub>oss</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s,</sub> <sub>w</sub>h<sub>e</sub>r<sub>eas</sub> th<sub>e</sub> <sub>re</sub>l<sub>axe</sub>d <sub>ran</sub>k<sub>-e</sub>i<sub>g</sub>ht <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t l<sub>arge</sub>l<sub>y</sub> <sub>res</sub>t<sub>ores</sub> <sub>per</sub>f<sub>ormance</sub> t<sub>owar</sub>d th<sub>e uncons</sub>t<sub>ra</sub>i<sub>ne</sub>d b<sub>ase</sub>li<sub>ne.</sub> N<sub>o</sub>t<sub>a</sub>bl<sub>y,</sub> thi<sub>s</sub> d<sub>egra</sub>d<sub>a</sub>ti<sub>on pers</sub>i<sub>s</sub>t<sub>s</sub> d<sub>e-</sub> s<sub>p</sub>ite the <sub>p</sub>resence of FFNs and resid<sub>u</sub>al connections. These res<sub>u</sub>lts demonstrate that restricti<sub>v</sub>e re<sub>p</sub>resentation rank constraints can im<sub>p</sub>air forecastin<sub>g</sub> <sub>p</sub>erformance<sub>,</sub> su<sub>pp</sub>ortin<sub>g</sub> the im<sub>p</sub>ortance of <sub>p</sub>reservin<sub>g</sub> suficient inde<sub>p</sub>endent re<sub>p</sub>resentation directions.

Table 2: Forecasting performance under diferent rank constraints on position-enriched embeddings. The best results are in bold.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Unconstrained</td><td colspan="2">Rank ≤ 1</td><td colspan="2">Rank ≤ 4</td><td colspan="2">Rank ≤ 8</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.503</td><td>0.469</td><td>0.549</td><td>0.511</td><td>0.509</td><td>0.473</td><td>0.504</td><td>0.472</td></tr><tr><td>ETTh2</td><td>0.380</td><td>0.414</td><td>0.385</td><td>0.417</td><td>0.398</td><td>0.421</td><td>0.383</td><td>0.414</td></tr><tr><td>ETTm1</td><td>0.369</td><td>0.390</td><td>0.385</td><td>0.402</td><td>0.372</td><td>0.393</td><td>0.372</td><td>0.392</td></tr><tr><td>ETTm2</td><td>0.271</td><td>0.326</td><td>0.273</td><td>0.327</td><td>0.274</td><td>0.327</td><td>0.270</td><td>0.324</td></tr><tr><td>Weather</td><td>0.228</td><td>0.263</td><td>0.231</td><td>0.266</td><td>0.231</td><td>0.266</td><td>0.229</td><td>0.265</td></tr><tr><td>Exchange</td><td>0.391</td><td>0.423</td><td>0.430</td><td>0.443</td><td>0.416</td><td>0.437</td><td>0.397</td><td>0.425</td></tr></table>

![](images/8a2f11a2102f841ce0cdf3b680529784ba55907cea34f6a48fae1f786f1a188f.jpg)

![](images/21f40604057c790b5aef0f78ef1e12c63f46188bc6a9d0702a46f918e0219c94.jpg)  
Figure 5: Efect of Network Depth on Forecasting with Different Pure Attention Mechanisms.

Table 3: Performance comparison of six-layer pure-attention models. The best results are in bold.
<table><tr><td>Method</td><td>Metric</td><td>ETTh1</td><td>ETTh2</td><td>ETTm1</td><td>Weather</td><td>Exchange</td></tr><tr><td rowspan="2">BiA</td><td>MSE</td><td>0.710</td><td>0.418</td><td>0.648</td><td>0.279</td><td>0.375</td></tr><tr><td>MAE</td><td>0.583</td><td>0.433</td><td>0.530</td><td>0.303</td><td>0.414</td></tr><tr><td rowspan="2">HgA</td><td>MSE</td><td>0.510</td><td>0.407</td><td>0.421</td><td>0.273</td><td>0.371</td></tr><tr><td>MAE</td><td>0.480</td><td>0.428</td><td>0.422</td><td>0.293</td><td>0.412</td></tr></table>

4.2.3 Robustness to Increasing Atention Depth. To evaluate the im-<sub>p</sub>act of attention de<sub>p</sub>th<sub>,</sub> we com<sub>p</sub>are BiA<sub>,</sub> causal attention<sub>,</sub> and H<sub>g</sub>A in <sub>p</sub>ure-attention networks with one to six stacked attention la<sub>y</sub>ers. All <sub>var</sub>i<sub>an</sub>t<sub>s</sub> <sub>s</sub>h<sub>are</sub> th<sub>e</sub> <sub>same</sub> <sub>arc</sub>hit<sub>ec</sub>t<sub>ure</sub> <sub>excep</sub>t f<sub>or</sub> th<sub>e</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> mechanism. Fi<sub>gu</sub>re 5 sho<sub>w</sub>s that BiA s<sub>u</sub>fers from s<sub>u</sub>bstantial <sub>p</sub>erformance de<sub>g</sub>radation as attention de<sub>p</sub>th increases<sub>,</sub> whereas H<sub>g</sub>A maintains more stable forecastin<sub>g p</sub>erformance. Causal attention <sub>a</sub>l<sub>so</sub> i<sub>mproves</sub> d<sub>ep</sub>th <sub>ro</sub>b<sub>us</sub>t<sub>ness,</sub> b<sub>u</sub>t it<sub>s</sub> <sub>per</sub>f<sub>ormance</sub> <sub>var</sub>i<sub>es</sub> <sub>across</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s.</sub> T<sub>a</sub>bl<sub>e</sub> 3 f<sub>ur</sub>th<sub>er</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>es</sub> th<sub>a</sub>t H<sub>g</sub>A <sub>ac</sub>hi<sub>eves</sub> <sub>cons</sub>i<sub>s-</sub> tent im<sub>p</sub>rovements over BiA across multi<sub>p</sub>le datasets<sub>,</sub> es<sub>p</sub>eciall<sub>y</sub> in dee<sub>p</sub>er attention stacks. These results indicate that H<sub>g</sub>A im<sub>p</sub>roves robustness under increasin<sub>g</sub>l<sub>y</sub> dee<sub>p</sub> attention interactions b<sub>y p</sub>reservin<sub>g</sub> more stable re<sub>p</sub>resentations. This em<sub>p</sub>irical behavior is consistent with the rank-<sub>p</sub>reservation anal<sub>y</sub>sis<sub>,</sub> showin<sub>g</sub> that attention to<sub>p</sub>olo<sub>gy</sub> <sub>p</sub>la<sub>y</sub>s an im<sub>p</sub>ortant role in maintainin<sub>g</sub> re<sub>p</sub>resentation diversit<sub>y</sub> durin<sub>g</sub> dee<sub>p</sub> information exchan<sub>g</sub>e.

4.2.4 Efect of Training-Time Atention Rank Constraints. To investi-<sub>g</sub>ate how attention rank constraints afect forecastin<sub>g</sub> <sub>p</sub>erformance<sub>,</sub> we im<sub>p</sub>ose rank constraints rank(A) ≤ � with � ∈ {1, 3} on ever<sub>y</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> h<sub>ea</sub>d <sub>o</sub>f <sub>a s</sub>i<sub>x-</sub>l<sub>ayer</sub> H<sub>g</sub>A <sub>mo</sub>d<sub>e</sub>l<sub>.</sub> Th<sub>e cons</sub>t<sub>ra</sub>i<sub>ne</sub>d <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> <sub>are</sub> t<sub>ra</sub>i<sub>ne</sub>d <sub>un</sub>d<sub>er</sub> th<sub>e same se</sub>tti<sub>ngs as</sub> th<sub>e uncons</sub>t<sub>ra</sub>i<sub>ne</sub>d b<sub>ase</sub>li<sub>ne.</sub> T<sub>a</sub>bl<sub>e</sub> 4 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t <sub>res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>ng a</sub>tt<sub>en</sub>ti<sub>on ran</sub>k <sub>cons</sub>i<sub>s</sub>t<sub>en</sub>tl<sub>y</sub> d<sub>egra</sub>d<sub>es</sub> forecastin<sub>g p</sub>erformance across datasets. Severe rank constraints $( r = 1 )$ l<sub>ea</sub>d t<sub>o su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>l <sub>per</sub>f<sub>ormance</sub> d<sub>rops, w</sub>hil<sub>e a re</sub>l<sub>axe</sub>d <sub>con-</sub> straint $( r = 3 )$ <sub>re</sub>d<sub>uces</sub> th<sub>e</sub> d<sub>egra</sub>d<sub>a</sub>ti<sub>on.</sub> Th<sub>ese resu</sub>lt<sub>s</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> that o<sub>v</sub>erl<sub>y</sub> restricti<sub>v</sub>e attention rank constraints can im<sub>p</sub>air the efectiveness of H<sub>g</sub>A<sub>,</sub> su<sub>pp</sub>ortin<sub>g</sub> the role of maintainin<sub>g</sub> suficient r<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n di<sub>ve</sub>r<sub>s</sub>it<sub>y</sub> d<sub>u</sub>rin<sub>g a</sub>tt<sub>e</sub>nti<sub>o</sub>n int<sub>e</sub>r<sub>ac</sub>ti<sub>o</sub>n<sub>.</sub>

![](images/92363f8a2cf6260e7457c77c0bae656279c7868f4ff2e97c1537b206b9cb3974.jpg)

Table 4: Training-Time Attention Rank Constraints in Six-Layer Pure HgA. For the rank-constrained columns, values are percentage changes relative to the Unconstrained model.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Unconstrained</td><td colspan="2"> $\operatorname { r a n k } ( A ) \leq 1$ </td><td colspan="2">rank  $( A ) \leq 3$ </td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.510</td><td>0.481</td><td>+12.44</td><td>+7.36</td><td>+0.23</td><td>+0.04</td></tr><tr><td>ETTh2</td><td>0.406</td><td>0.427</td><td>+0.93</td><td>-0.55</td><td>+1.88</td><td>+0.72</td></tr><tr><td>ETTm1</td><td>0.422</td><td>0.421</td><td>+37.72</td><td>+19.04</td><td>+7.27</td><td>+3.89</td></tr><tr><td>ETTm2</td><td>0.293</td><td>0.338</td><td>+6.92</td><td>+4.07</td><td>+3.60</td><td>+2.02</td></tr><tr><td>Weather</td><td>0.273</td><td>0.293</td><td>+6.83</td><td>+6.70</td><td>+4.47</td><td>+3.93</td></tr><tr><td>Exchange</td><td>0.372</td><td>0.413</td><td>+8.04</td><td>+5.66</td><td>+3.97</td><td>+2.71</td></tr><tr><td>Solar</td><td>0.306</td><td>0.326</td><td>+50.40</td><td>+40.08</td><td>+8.53</td><td>+7.52</td></tr></table>

Figure 6: Forecasting Comparison of HgA and Causal Attention under Diferent Token Orders.

4.2.5 Predictive Value ofInformation Pathways. To evaluate the <sub>p</sub>redictive value of H<sub>g</sub>A’s information <sub>p</sub>athwa<sub>y</sub>s<sub>,</sub> we com<sub>p</sub>are H<sub>g</sub>A <sub>w</sub>ith <sub>causa</sub>l <sub>a</sub>tt<sub>en</sub>ti<sub>on un</sub>d<sub>er</sub> f<sub>orwar</sub>d <sub>an</sub>d <sub>reverse</sub>d t<sub>o</sub>k<sub>en or</sub>d<sub>ers,</sub> usin<sub>g</sub> the same ori<sub>g</sub>inal latest <sub>p</sub>atch as the <sub>p</sub>rediction readout in both settin<sub>g</sub>s. Reversin<sub>g</sub> token order chan<sub>g</sub>es the information accessible th<sub>roug</sub>h <sub>causa</sub>l <sub>a</sub>tt<sub>en</sub>ti<sub>on,</sub> <sub>a</sub>ll<sub>ow</sub>i<sub>ng</sub> <sub>us</sub> t<sub>o</sub> <sub>assess</sub> th<sub>e</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> <sub>o</sub>f H<sub>g</sub>A’s additional cross-token interactions. Fi<sub>gu</sub>re 6 shows that H<sub>g</sub>A <sub>ou</sub>t<sub>per</sub>f<sub>orms causa</sub>l <sub>a</sub>tt<sub>en</sub>ti<sub>on on mos</sub>t f<sub>orwar</sub>d<sub>-or</sub>d<sub>er</sub> t<sub>as</sub>k<sub>s, w</sub>ith m<sub>o</sub>r<sub>e p</sub>r<sub>o</sub>n<sub>ou</sub>n<sub>ce</sub>d im<sub>p</sub>r<sub>ove</sub>m<sub>e</sub>nt<sub>s u</sub>nd<sub>e</sub>r r<sub>eve</sub>r<sub>se</sub>d <sub>o</sub>rd<sub>e</sub>r<sub>, w</sub>h<sub>e</sub>r<sub>e</sub> H<sub>g</sub>A <sub>ac</sub>hi<sub>eves</sub> l<sub>ower</sub> MSE <sub>on a</sub>ll 24 t<sub>as</sub>k<sub>s s</sub>h<sub>own.</sub> Th<sub>ese resu</sub>lt<sub>s</sub> hi<sub>g</sub>hli<sub>g</sub>ht the <sub>p</sub>redictive value of H<sub>g</sub>A’s information <sub>p</sub>athwa<sub>y</sub>s<sub>,</sub> demonstratin<sub>g</sub> its abilit<sub>y</sub> to efectivel<sub>y</sub> inte<sub>g</sub>rate cross-token information while maintainin<sub>g</sub> rank stabilit<sub>y</sub>.

4.2.6 Empirical Validation of Rank Theory. To empirically validate <sub>our</sub> th<sub>eore</sub>ti<sub>ca</sub>l f<sub>ramewor</sub>k<sub>, we v</sub>i<sub>sua</sub>li<sub>ze</sub> th<sub>e ran</sub>k di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons o</sub>f int<sub>e</sub>rm<sub>e</sub>di<sub>a</sub>t<sub>e</sub> r<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n<sub>s</sub> in Fi<sub>gu</sub>r<sub>e</sub> 7 <sub>a</sub>nd 8<sub>.</sub> Fir<sub>s</sub>t<sub>,</sub> th<sub>e</sub> Fi<sub>gu</sub>r<sub>e</sub> 7 d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>a</sub>t <sub>our</sub> <sub>non</sub>li<sub>near</sub> MLP <sub>em</sub>b<sub>e</sub>ddi<sub>ng</sub> <sub>exp</sub>li<sub>c</sub>itl<sub>y</sub> b<sub>rea</sub>k<sub>s</sub> the rank-1 bottleneck inherent to � = 1 linear projections. It consist<sub>en</sub>tl<sub>y</sub> d<sub>r</sub>i<sub>ves</sub> th<sub>e represen</sub>t<sub>a</sub>ti<sub>ons</sub> t<sub>owar</sub>d hi<sub>g</sub>h<sub>-ran</sub>k <sub>s</sub>t<sub>a</sub>t<sub>es, c</sub>l<sub>ose</sub>l<sub>y</sub> a<sub>pp</sub>roachin<sub>g</sub> the theoretical maximum (validatin<sub>g</sub> Theorem 1). Second<sub>,</sub> the Fi<sub>g</sub>ure 8 com<sub>p</sub>are the attention wei<sub>g</sub>ht matrices. While <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d BiA <sub>su</sub>f<sub>ers</sub> f<sub>rom severe ran</sub>k <sub>co</sub>ll<sub>apse, ev</sub>id<sub>ence</sub>d b<sub>y a</sub> <sub>w</sub>id<sub>esp</sub>r<sub>ea</sub>d l<sub>o</sub>n<sub>g</sub>-t<sub>a</sub>il di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n t<sub>owa</sub>rd l<sub>owe</sub>r r<sub>a</sub>nk<sub>s,</sub> H<sub>g</sub>A <sub>co</sub>n-<sub>s</sub>i<sub>s</sub>t<sub>en</sub>tl<sub>y</sub> l<sub>oc</sub>k<sub>s a</sub>t th<sub>e max</sub>i<sub>mum ran</sub>k <sub>across a</sub>ll t<sub>es</sub>t <sub>samp</sub>l<sub>es.</sub> Thi<sub>s</sub> confirms that H<sub>g</sub>A efectivel<sub>y</sub> <sub>p</sub>reserves re<sub>p</sub>resentational diversit<sub>y</sub> and circumvents rank-deficient to<sub>p</sub>olo<sub>g</sub>ies<sub>, p</sub>erfectl<sub>y</sub> ali<sub>g</sub>nin<sub>g</sub> with Th<sub>eorem</sub> 2<sub>.</sub>

![](images/9027e9639cba4094dab9652468503977e361dfb07f7dd00dd8d7d6f54909c230.jpg)

![](images/73015d913be44ecf84404988ced686c284abc0d954ecb37de58d9a0cc84975e6.jpg)  
Figure 7: Rank distributions of embedding matrices with patch sizes � = 1 (ETTm2) and $P = 1 6$ (Weather).

![](images/ffcc3e31634028a7be4aba7a6dbef5300ee5b32c02bf06dc4c0e51bc9ab3a2a3.jpg)

![](images/c66d5ada7b285637cbc86ff544f122d7a4fc0c85639066925809b668fec9b0a0.jpg)  
Figure 8: Rank distributions of attention matrices on Weather and ETTm2.

## 4.3 Interactive Progressive Decoder Analysis

4.3.1 Analysis ofProgressive Refinement. We investigate the efectiveness of <sub>p</sub>ro<sub>g</sub>ressive refinement b<sub>y</sub> examinin<sub>g</sub> <sub>p</sub>rediction accu-<sub>racy,</sub> <sub>error</sub> <sub>correc</sub>ti<sub>on,</sub> <sub>an</sub>d i<sub>n</sub>f<sub>erence</sub> <sub>e</sub>fi<sub>c</sub>i<sub>ency.</sub> U<sub>s</sub>i<sub>ng</sub> fi<sub>xe</sub>d <sub>c</sub>h<sub>ec</sub>k<sub>-</sub> <sub>po</sub>i<sub>n</sub>t<sub>s,</sub> <sub>we</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub> dif<sub>eren</sub>t <sub>re</sub>fi<sub>nemen</sub>t <sub>s</sub>t<sub>eps</sub> $T \in \{ 1 , \ldots , 2 0 \}$ across multi<sub>p</sub>le forecastin<sub>g</sub> horizons. Fi<sub>g</sub>ure 9 shows that ETTm1 benefits <sub>p</sub>rimaril<sub>y</sub> from a few refinement ste<sub>p</sub>s<sub>,</sub> whereas Exchan<sub>g</sub>e continues to im<sub>p</sub>ro<sub>v</sub>e as � increases<sub>,</sub> indicatin<sub>g</sub> dataset-de<sub>p</sub>endent sensiti<sub>v</sub>it<sub>y</sub> t<sub>o re</sub>fi<sub>nemen</sub>t d<sub>ep</sub>th<sub>.</sub>

T<sub>o un</sub>d<sub>ers</sub>t<sub>an</sub>d <sub>w</sub>h<sub>a</sub>t <sub>re</sub>fi<sub>nemen</sub>t <sub>correc</sub>t<sub>s, we</sub> d<sub>ecompose</sub> f<sub>orecas</sub>t<sub>-</sub> in<sub>g</sub> errors into level<sub>,</sub> linear-trend<sub>,</sub> and remainin<sub>g</sub>-sha<sub>p</sub>e com<sub>p</sub>onents. Fi<sub>gu</sub>r<sub>e</sub> 10 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t in<sub>c</sub>r<sub>eas</sub>in<sub>g</sub> � fr<sub>o</sub>m 1 t<sub>o</sub> 5 im<sub>p</sub>r<sub>oves</sub> MSE <sub>a</sub>nd MAE <sub>across a</sub>ll <sub>s</sub>i<sub>x eva</sub>l<sub>ua</sub>t<sub>e</sub>d d<sub>a</sub>t<sub>ase</sub>t<sub>s a</sub>t $\tau = 9 6$ <sub>, w</sub>ith dif<sub>eren</sub>t dominant correction com<sub>p</sub>onents across datasets.

Fi<sub>na</sub>ll<sub>y, we eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e e</sub>fi<sub>c</sub>i<sub>ency o</sub>f <sub>o</sub>fli<sub>ne-</sub>f<sub>o</sub>ld<sub>e</sub>d i<sub>n</sub>f<sub>erence.</sub> The folded <sub>p</sub>redictions match ex<sub>p</sub>licit iterative u<sub>p</sub>dates within num<sub>e</sub>ri<sub>ca</sub>l <sub>p</sub>r<sub>ec</sub>i<sub>s</sub>i<sub>o</sub>n <sub>w</sub>hil<sub>e</sub> <sub>avo</sub>idin<sub>g</sub> <sub>o</sub>nlin<sub>e</sub> it<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>s.</sub> A<sub>s</sub> <sub>s</sub>h<sub>ow</sub>n in Fi<sub>gu</sub>r<sub>e</sub> 11<sub>,</sub> <sub>a</sub>t $T = 2 0$ <sub>,</sub> <sub>o</sub>fli<sub>ne</sub> f<sub>o</sub>ldi<sub>ng</sub> <sub>ac</sub>hi<sub>eves</sub> <sub>me</sub>di<sub>an</sub> <sub>spee</sub>d<sub>ups</sub> <sub>o</sub>f 3.77× for the refinement module and 1.45× for the full model. These <sub>resu</sub>lt<sub>s</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>a</sub>t IPD <sub>ena</sub>bl<sub>es</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> <sub>progress</sub>i<sub>ve</sub> f<sub>orecas</sub>t <sub>correc</sub>ti<sub>on</sub> <sub>w</sub>ith <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t i<sub>n</sub>f<sub>erence.</sub>

![](images/138213e5cdd99c68cb5304d6180a5c0ecaf16c462c49c6b3ca42cb85a3b601f2.jpg)

Figure 9: Hyperparameter sensitivity of refinement steps (� ).  
![](images/adc60a3953b98da9c0a0ab1363787a8a776bfed886818404df71c62129ed273e.jpg)  
Figure 10: Reductions in level, trend, and remaining-shape errors from � = 1 to � = 5 at � = 96.

![](images/b6bf3ceaab823a2180f935a48514d716507d831e18e553bda5ba0c6f51d60ac4.jpg)  
Figure 11: Online inference latency of iterative refinement and ofline-folded inference.

4.3.2 Efectiveness ofIPD Components. To evaluate the contributions of IPD com<sub>p</sub>onents<sub>,</sub> we com<sub>p</sub>are three <sub>p</sub>rediction heads under th<sub>e</sub> <sub>same</sub> <sub>enco</sub>d<sub>er:</sub> <sub>a</sub> <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d li<sub>near</sub> h<sub>ea</sub>d<sub>,</sub> <sub>progress</sub>i<sub>ve</sub> <sub>res</sub>id<sub>ua</sub>l <sub>re-</sub> finement without conditionin<sub>g</sub> (Refinement Onl<sub>y</sub>), and the com<sub>p</sub>lete IPD <sub>w</sub>ith int<sub>e</sub>r<sub>ac</sub>ti<sub>ve co</sub>nditi<sub>o</sub>nin<sub>g.</sub> T<sub>a</sub>bl<sub>e</sub> 5 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t r<sub>es</sub>id<sub>ua</sub>l r<sub>e</sub>- fi<sub>nemen</sub>t <sub>a</sub>l<sub>one</sub> i<sub>mproves</sub> MSE <sub>over</sub> th<sub>e</sub> li<sub>near</sub> h<sub>ea</sub>d <sub>on</sub> fi<sub>ve o</sub>f <sub>s</sub>i<sub>x</sub> datasets. Introducin<sub>g</sub> interactive conditionin<sub>g</sub> further im<sub>p</sub>roves <sub>p</sub>erf<sub>o</sub>rm<sub>a</sub>n<sub>ce, w</sub>ith F<sub>u</sub>ll IPD <sub>ac</sub>hi<sub>ev</sub>in<sub>g</sub> th<sub>e</sub> l<sub>owes</sub>t MSE <sub>a</sub>nd MAE <sub>ac</sub>r<sub>oss</sub> <sub>a</sub>ll <sub>s</sub>i<sub>x</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s.</sub> N<sub>o</sub>t<sub>a</sub>bl<sub>y, on</sub> W<sub>ea</sub>th<sub>er, re</sub>fi<sub>nemen</sub>t <sub>a</sub>l<sub>one s</sub>li<sub>g</sub>htl<sub>y</sub> <sub>un</sub>d<sub>erper</sub>f<sub>orms</sub> th<sub>e</sub> li<sub>near</sub> h<sub>ea</sub>d<sub>,</sub> <sub>w</sub>h<sub>ereas</sub> F<sub>u</sub>ll IPD i<sub>mproves</sub> <sub>upon</sub> b<sub>o</sub>th<sub>.</sub> Th<sub>ese resu</sub>lt<sub>s</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>e comp</sub>l<sub>emen</sub>t<sub>ary</sub> b<sub>ene</sub>fit<sub>s o</sub>f <sub>p</sub>ro<sub>g</sub>ressive residual refinement and interactive conditionin<sub>g</sub> for m<sub>u</sub>ltivariate forecastin<sub>g</sub>.

4.3.3 Efect of Cross-channel Conditioning. To investigate how <sub>cross-c</sub>h<sub>anne</sub>l i<sub>n</sub>f<sub>orma</sub>ti<sub>on con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> t<sub>o</sub> f<sub>orecas</sub>ti<sub>ng, we ana</sub>l<sub>yze</sub> th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ve-</sub>h<sub>um</sub>idit<sub>y</sub> <sub>c</sub>h<sub>anne</sub>l <sub>on</sub> W<sub>ea</sub>th<sub>er</sub> b<sub>y</sub> <sub>rep</sub>l<sub>ac</sub>i<sub>ng</sub> <sub>source</sub> <sub>c</sub>h<sub>an-</sub> nel features with their trainin<sub>g</sub>-set means. Fi<sub>g</sub>ure 12(a) illustrates <sup>h</sup>ow samp<sup>l</sup>e-<sup>d</sup>epen<sup>d</sup>ent cross-c<sup>h</sup>anne<sup>l</sup> in<sup>f</sup>ormation a<sup>d</sup>justs t<sup>h</sup>e pre-<sup>d</sup>icte<sup>d</sup> trajectory, particu<sup>l</sup>ar<sup>l</sup>y aroun<sup>d</sup> t<sup>h</sup>e troug<sup>h</sup> an<sup>d</sup> <sup>l</sup>ater <sup>f</sup>orecast <sub>p</sub>ositions. The scalin<sub>g</sub> modulation contributes most of the a<sup>d</sup>justment in t<sup>h</sup>is examp<sup>l</sup>e, w<sup>h</sup>i<sup>l</sup>e t<sup>h</sup>e s<sup>h</sup>i<sup>f</sup>ting mo<sup>d</sup>u<sup>l</sup>ation provi<sup>d</sup>es smaller corrections. Fi<sub>g</sub>ure 12(b) further reveals var<sub>y</sub>in<sub>g</sub> contributions from diferent source channels. The raining channel shows th<sub>e</sub> l<sub>arges</sub>t <sub>pos</sub>iti<sub>ve</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on.</sub> Th<sub>ese</sub> fi<sub>n</sub>di<sub>ngs</sub> d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>a</sub>t IPD can ex<sub>p</sub>loit informative cross-channel relationshi<sub>p</sub>s to <sub>p</sub>roduce ada<sub>p</sub>tive tem<sub>p</sub>oral corrections<sub>,</sub> su<sub>pp</sub>ortin<sub>g</sub> the efectiveness of int<sub>e</sub>r<sub>ac</sub>ti<sub>ve</sub> <sub>co</sub>nditi<sub>o</sub>nin<sub>g.</sub>

Table 5: Comparison of IPD component configurations.
<table><tr><td>Dataset</td><td>Linear Head MSE MAE</td><td>MSE</td><td>Refinement Only MAE</td><td colspan="2">Full IPD MSE MAE</td></tr><tr><td>ETTh1 ETTh2 ETTm1 ETTm2</td><td>0.426 0.349 0.349 0.368 0.254 0.307</td><td>0.430 0.412 0.385 0.344 0.342</td><td>0.421 0.384 0.367</td><td>0.405 0.342 0.339 0.252 0.307</td><td>0.416 0.384 0.367</td></tr></table>

![](images/3a169a410244b67801fa640db1e51979515ef6722bc4740ba9a2cdbecd3bb96f.jpg)  
Figure 12: Cross-channel conditioning analysis on Weather.

## 5 Conclusion

In this <sub>p</sub>a<sub>p</sub>er<sub>,</sub> we <sub>p</sub>resent RankFormer<sub>,</sub> a <sub>p</sub>rinci<sub>p</sub>led encoder-decoder architecture for multivariate time series forecastin<sub>g</sub>. Throu<sub>g</sub>h a <sub>ran</sub>k<sub>-</sub>b<sub>ase</sub>d <sub>ana</sub>l<sub>y</sub>ti<sub>ca</sub>l f<sub>ramewor</sub>k<sub>, we rev</sub>i<sub>s</sub>it h<sub>ow</sub> t<sub>o</sub>k<sub>en</sub>i<sub>za</sub>ti<sub>on, em-</sub> b<sub>e</sub>ddi<sub>ng, an</sub>d <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> d<sub>es</sub>i<sub>gns</sub> i<sub>n</sub>fl<sub>uence</sub> th<sub>e capac</sub>it<sub>y o</sub>f l<sub>earne</sub>d re<sub>p</sub>resentations. Our anal<sub>y</sub>sis reveals that <sub>p</sub>atchification ex<sub>p</sub>ands th<sub>e</sub> <sub>a</sub>tt<sub>a</sub>i<sub>na</sub>bl<sub>e</sub> <sub>ran</sub>k b<sub>oun</sub>d<sub>,</sub> <sub>non</sub>li<sub>near</sub> <sub>em</sub>b<sub>e</sub>ddi<sub>ngs</sub> <sub>overcome</sub> th<sub>e</sub> a<sup>dd</sup>itiona<sup>l</sup> constraints o<sup>f l</sup>inear projections, an<sup>d</sup> attention topo<sup>l</sup>ogy <sub>p</sub>la<sub>y</sub>s an im<sub>p</sub>ortant role in <sub>p</sub>reservin<sub>g</sub> re<sub>p</sub>resentation diversit<sub>y</sub> durin<sub>g</sub> information interaction. Guided b<sub>y</sub> these insi<sub>g</sub>hts<sub>,</sub> we develo<sub>p</sub> a rank-enhancin<sub>g</sub> encoder with H<sub>g</sub>A and an IPD<sub>,</sub> which res<sub>p</sub>ective<sup>l</sup>y improve representation preservation an<sup>d f</sup>uture trajectory modelin<sub>g</sub> throu<sub>g</sub>h efective information exchan<sub>g</sub>e and <sub>p</sub>ro<sub>g</sub>ressive <sub>re</sub>fi<sub>nemen</sub>t<sub>.</sub> E<sub>x</sub>t<sub>ens</sub>i<sub>ve exper</sub>i<sub>men</sub>t<sub>s on rea</sub>l<sub>-wor</sub>ld f<sub>orecas</sub>ti<sub>ng</sub> b<sub>enc</sub>h<sub>-</sub> <sub>mar</sub>k<sub>s va</sub>lid<sub>a</sub>t<sub>e</sub> th<sub>e e</sub>f<sub>ec</sub>ti<sub>veness o</sub>f R<sub>an</sub>kF<sub>ormer an</sub>d d<sub>emons</sub>t<sub>ra</sub>t<sub>e</sub> consistent im<sub>p</sub>ro<sub>v</sub>ements o<sub>v</sub>er com<sub>p</sub>etiti<sub>v</sub>e methods.

## References

[1] Srinadh Bhojana<sub>p</sub>alli, Chulhee Yun, Ankit Sin<sub>g</sub>h Rawat, Sashank Reddi, and Sanjiv Kumar. 2020. Low-ran<sup>k b</sup>ott<sup>l</sup>enec<sup>k</sup> in mu<sup>l</sup>ti-<sup>h</sup>ea<sup>d</sup> attention mo<sup>d</sup>e<sup>l</sup>s. In International conference on machine learning. PMLR, 864–873.

[2] Jiezhu Chen<sub>g</sub>, Kaizhu Huan<sub>g</sub>, and Zibin Zhen<sub>g</sub>. 2020. Towards better forecastin<sub>g</sub> by fusing near and distant future visions. In Proceedings ofthe AAAI Conference on Artificial Intelligence. 3593–3600.

[3] Krz<sub>y</sub>sztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xin<sub>gy</sub>ou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz M<sub>o</sub>hi<sub>u</sub>ddin<sub>,</sub> L<sub>u</sub>k<sub>as</sub>z K<sub>a</sub>i<sub>se</sub>r<sub>, e</sub>t <sub>a</sub>l<sub>.</sub> 2021<sub>.</sub> R<sub>e</sub>thinkin<sub>g</sub> Att<sub>e</sub>nti<sub>o</sub>n <sub>w</sub>ith P<sub>e</sub>rf<sub>o</sub>rm<sub>e</sub>r<sub>s.</sub> In International Conference on Learning Representations.

[4] Yihe Don<sub>g</sub>, Jean-Ba<sub>p</sub>tiste Cordonnier, and Andreas Loukas. 2021. Attention is <sub>no</sub>t <sub>a</sub>ll <sub>you</sub> <sub>nee</sub>d<sub>:</sub> P<sub>ure</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> l<sub>oses</sub> <sub>ran</sub>k d<sub>ou</sub>bl<sub>y</sub> <sub>exponen</sub>ti<sub>a</sub>ll<sub>y</sub> <sub>w</sub>ith d<sub>ep</sub>th<sub>.</sub> I<sub>n</sub> International conference on machine learning. PMLR, 2793–2803.

[5] Yuchen Fan<sub>g</sub>, Yuxuan Lian<sub>g</sub>, Bo Hui, Zezhi Shao, Liwei Den<sub>g</sub>, Xu Liu, Xinke Jian<sub>g</sub>, <sub>an</sub>d K<sub>a</sub>i Zh<sub>eng.</sub> 2025<sub>.</sub> Efi<sub>c</sub>i<sub>en</sub>t l<sub>arge-sca</sub>l<sub>e</sub> t<sub>ra</sub>fi<sub>c</sub> f<sub>orecas</sub>ti<sub>ng w</sub>ith t<sub>rans</sub>f<sub>ormers:</sub> A spatial data management perspective. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 307–317.

[6] Jin<sub>g</sub>ru Fei, Kun Yi, Wei Fan, Qi Zhan<sub>g</sub>, and Zhendon<sub>g</sub> Niu. 2025. Am<sub>p</sub>lifier: Brin<sub>g</sub> in<sub>g</sub> attention to ne<sub>g</sub>lected low-ener<sub>gy</sub> com<sub>p</sub>onents in time series forecastin<sub>g</sub>. In Proceedings ofthe AAAI conference on artificial intelligence. 11645–11653.

[7] Soufiane Ha<sub>y</sub>ou, Nikhil Ghosh, and Bin Yu. 2024. LoRA+ eficient low rank ada tation of lar e models. In Proceedings ofthe 41st International Conference on Machine Learning. 17783–17806.

[8] Edward J Hu, Yelon<sub>g</sub> Shen, Philli<sub>p</sub> Wallis, Ze<sub>y</sub>uan Allen-Zhu, Yuanzhi Li, Shean Wan<sub>g,</sub> Lu Wan<sub>g,</sub> Weizhu Chen<sub>,</sub> et al. 2022. Lora: Low-rank ada<sub>p</sub>tation of lar<sub>g</sub>e language models. ICLR 1, 2 (2022), 3.

[9] Jiajia Li, Lin<sub>g</sub> Dai, Fen<sub>g</sub> Tan, Hui Shen, Zikai Wan<sub>g</sub>, Bin Shen<sub>g</sub>, and Pen<sub>g</sub>wei Hu. 2022<sub>.</sub> CDX-NET: <sub>c</sub>r<sub>oss</sub>-d<sub>o</sub>m<sub>a</sub>in m<sub>u</sub>lti-f<sub>ea</sub>t<sub>u</sub>r<sub>e</sub> f<sub>us</sub>i<sub>o</sub>n m<sub>o</sub>d<sub>e</sub>lin<sub>g v</sub>i<sub>a</sub> d<sub>eep</sub> n<sub>eu</sub>r<sub>a</sub>l networks for multivariate time series forecasting in AIOps. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE<sub>,</sub> 4073–4077.

[10] Shizhan Liu, Han<sub>g</sub> Yu, Con<sub>g</sub> Liao, Jian<sub>g</sub>uo Li, Wei<sub>y</sub>ao Lin, Alex X Liu, and Schahram D<sub>u</sub>stdar. 2022. P<sub>y</sub>raformer: Lo<sub>w</sub>-Com<sub>p</sub>lexit<sub>y</sub> P<sub>y</sub>ramidal Attention for Long-Range Time Series Modeling and Forecasting. In International Conference on Learning Representations.

[11] Shih-Yan<sub>g</sub> Liu, Chien-Yi Wan<sub>g</sub>, Hon<sub>g</sub>xu Yin, Pavlo Molchanov, Yu-Chian<sub>g</sub> Frank Wan<sub>g,</sub> K<sub>w</sub>an<sub>g</sub>-Tin<sub>g</sub> Chen<sub>g,</sub> and Min-H<sub>u</sub>n<sub>g</sub> Chen. 2024. Dora: Wei<sub>g</sub>htdecomposed low-rank adaptation. In Forty-first International Conference on Machine Learning.

[12] Yon<sub>g</sub> Liu, Ten<sub>gg</sub>e Hu, Haoran Zhan<sub>g</sub>, Haixu Wu, Shi<sub>y</sub>u Wan<sub>g</sub>, Lintao Ma, and Min<sub>g</sub>shen<sub>g</sub> Lon<sub>g</sub>. 2023. itransformer: Inverted transformers are efective for time series forecasting. arXiv preprint arXiv:2310.06625 (2023).

[13] Yon<sub>g</sub> Liu, Guo Qin, Xian<sub>g</sub>don<sub>g</sub> Huan<sub>g</sub>, Jianmin Wan<sub>g</sub>, and Min<sub>g</sub>shen<sub>g</sub> Lon<sub>g</sub>. 2025<sub>.</sub> Tim<sub>e</sub>r-XL: L<sub>o</sub>n<sub>g</sub>-C<sub>o</sub>nt<sub>e</sub>xt Tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>e</sub>r<sub>s</sub> f<sub>o</sub>r Unifi<sub>e</sub>d Tim<sub>e</sub> S<sub>e</sub>ri<sub>es</sub> F<sub>o</sub>r<sub>ecas</sub>tin<sub>g.</sub> In The Thirteenth International Conference on Learning Representations.

[14] Yon<sub>g</sub> Liu, Haoran Zhan<sub>g</sub>, Chen<sub>y</sub>u Li, Xian<sub>g</sub>don<sub>g</sub> Huan<sub>g</sub>, Jianmin Wan<sub>g</sub>, and Min<sub>g</sub>shen<sub>g</sub> Lon<sub>g</sub>. 2024. Timer: <sub>g</sub>enerative <sub>p</sub>re-trained transformers are lar<sub>g</sub>e time series models. In Proceedings of the 41st International Conference on Machine Learning. 32369–32399.

[15] Md Mahmuddun Nabi Murad, Mehmet Aktukmak, and Yasin Yilmaz. 2025. W<sub>p</sub> mix<sub>e</sub>r: Efi<sub>c</sub>i<sub>e</sub>nt m<sub>u</sub>lti-r<sub>eso</sub>l<sub>u</sub>ti<sub>o</sub>n mixin<sub>g</sub> f<sub>o</sub>r l<sub>o</sub>n<sub>g</sub>-t<sub>e</sub>rm tim<sub>e se</sub>ri<sub>es</sub> f<sub>o</sub>r<sub>ecas</sub>tin<sub>g.</sub> In Proceedings ofthe AAAI Conference on Artificial Intelligence. 19581–19588.

[16] Vinod Nair and Geofre<sub>y</sub> E Hinton. 2010. Rectified linear units im<sub>p</sub>rove restricted boltzmann machines. In Proceedings ofthe 27th international conference on machine learning (ICML-10). 807–814.

[17] Yu<sub>q</sub>i Nie, Nam H N<sub>g</sub>u<sub>y</sub>en, Phanwadee Sinthon<sub>g</sub>, and Ja<sub>y</sub>ant Kala<sub>g</sub>nanam. 2023. A Tim<sub>e</sub> S<sub>e</sub>ri<sub>es</sub> i<sub>s</sub> W<sub>o</sub>rth 64 W<sub>o</sub>rd<sub>s</sub>: L<sub>o</sub>n<sub>g</sub>-t<sub>e</sub>rm F<sub>o</sub>r<sub>ecas</sub>tin<sub>g w</sub>ith Tr<sub>a</sub>n<sub>s</sub>f<sub>o</sub>rm<sub>e</sub>r<sub>s.</sub> In The Eleventh International Conference on Learning Representations.

[18] Lorenzo Noci, Sotiris Ana<sub>g</sub>nostidis, Luca Bi<sub>gg</sub>io, Antonio Orvieto, Sidak Pal Sin<sub>g</sub>h<sub>,</sub> and Aurelien Lucchi. 2022. Si<sub>g</sub>nal <sub>p</sub>ro<sub>p</sub>a<sub>g</sub>ation in transformers: Theoretical perspectives and the role of rank collapse. Advances in Neural Information Processing Systems 35 (2022), 27198–27211.

[19] Xihao Piao, Zhen<sub>g</sub> Chen, Taichi Mura<sub>y</sub>ama, Yasuko Matsubara, and Yasushi Sak<sub>u</sub>rai. 2024. Fredformer: Fre<sub>qu</sub>enc<sub>y</sub> debiased transformer for time series forecasting. In Proceedings ofthe 30th ACM SIGKDD conference on knowledge discovery and data mining. 2400–2410.

[20] Xiangfei Qiu, Xvyuan Liu, Tianen Shen, Xingjian Wu, Hanyin Cheng, Bin Yang, and Jilin Hu. 2026. SEER: Transformer-based Robust Time Series Forecastin<sub>g</sub> via Automated Patch Enhancement and Replacement. arXiv preprint arXiv:2602.00589 (2026).

[21] Xiangfei Qiu, Xingjian Wu, Yan Lin, Chenjuan Guo, Jilin Hu, and Bin Yang. 2025<sub>.</sub> D<sub>ue</sub>t: D<sub>ua</sub>l <sub>c</sub>l<sub>us</sub>t<sub>e</sub>rin<sub>g e</sub>nh<sub>a</sub>n<sub>ce</sub>d m<sub>u</sub>lti<sub>va</sub>ri<sub>a</sub>t<sub>e</sub> tim<sub>e se</sub>ri<sub>es</sub> f<sub>o</sub>r<sub>ecas</sub>tin<sub>g.</sub> In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 1185–1196.

[22] Thiziri Nait Saada, Alireza Naderi, and Jared Tanner. 2025. Mind the Ga<sub>p</sub>: a S<sub>p</sub>ectral Anal<sub>y</sub>sis of Rank Colla<sub>p</sub>se and Si<sub>g</sub>nal Pro<sub>p</sub>a<sub>g</sub>ation in Attention La<sub>y</sub>ers. In Forty-second International Conference on Machine Learning.

[23] David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. 2020. DeepAR: Probabilistic forecasting with autoregressive recurrent networks. International journal offorecasting (2020), 1181–1191.

[24] Art<sub>y</sub>om Stits<sub>y</sub>uk and Jaesik Choi. 2025. x<sub>p</sub>atch: Dual-stream time series forecasting with exponential seasonal-trend decomposition. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 20601–20609.

[25] Hua Wan<sub>g</sub>, Jin<sub>g</sub>hao Lu, and Fan Zhan<sub>g</sub>. 2026. Eeo-tfv: Esca<sub>p</sub>e-ex<sub>p</sub>lore o<sub>p</sub>timizer for web-scale time-series forecastin and vision anal sis. In Proceedings ofthe ACM Web Conference 2026. 7271–7282.

[26] Sinon<sub>g</sub> Wan<sub>g</sub>, Belinda Z Li, Madian Khabsa, Han Fan<sub>g</sub>, and Hao Ma. 2020. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768 (2020).

[27] Shi<sub>y</sub>u Wan<sub>g</sub>, Jiawei Li, Xiaomin<sub>g</sub> Shi, Zhou Ye, Baichuan Mo, Wenze Lin, Ju Shen<sub>g</sub>ton<sub>g</sub>, Zhixuan Chu, and Min<sub>g</sub> Jin. 2025. TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis. In ICLR 2025: The Thirteenth International Conference on Learning Representations. International C<sub>o</sub>nf<sub>e</sub>r<sub>e</sub>n<sub>ce o</sub>n L<sub>ea</sub>rnin<sub>g</sub> R<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n<sub>s.</sub>

[28] Haixu Wu, Ten<sub>gg</sub>e Hu, Yon<sub>g</sub> Liu, Han<sub>g</sub> Zhou, Jianmin Wan<sub>g</sub>, and Min<sub>g</sub>shen<sub>g</sub> L<sub>o</sub>n<sub>g.</sub> 2023<sub>.</sub> Tim<sub>es</sub>N<sub>e</sub>t: T<sub>e</sub>m<sub>po</sub>r<sub>a</sub>l 2D-V<sub>a</sub>ri<sub>a</sub>ti<sub>o</sub>n M<sub>o</sub>d<sub>e</sub>lin<sub>g</sub> f<sub>o</sub>r G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>l Tim<sub>e</sub> S<sub>e</sub>ri<sub>es</sub> Analysis. In The Eleventh International Conference on Learning Representations.

[29] Haixu Wu, Jiehui Xu, Jianmin Wan<sub>g</sub>, and Min<sub>g</sub>shen<sub>g</sub> Lon<sub>g</sub>. 2021. Autoformer: Decom<sub>p</sub>osition transformers with auto-correlation for lon<sub>g</sub>-term series forecastin<sub>g</sub>. Advances in neural information processing systems 34 (2021), 22419–22430.

[30] Xinyi Wu, Amir Ajorlou, Yifei Wang, Stefanie Jegelka, and Ali Jadbabaie. 2024. On the role of attention masks and layernorm in transformers. Advances in Neural Information Processing Systems 37 (2024), 14774–14809.

[31] Xingjian Wu, Xiangfei Qiu, Hanyin Cheng, Zhengyu Li, Jilin Hu, Chenjuan G<sub>u</sub>o<sub>,</sub> and Bin Yan<sub>g</sub>. 2025. Enhancin<sub>g</sub> Time Series Forecastin<sub>g</sub> thro<sub>ug</sub>h Selecti<sub>v</sub>e Representation Spaces: A Patch Perspective. In Advances in Neural Information Processing Systems, Vol. 38. 23328–23354.

[32] Yun<sub>y</sub>an<sub>g</sub> Xion<sub>g</sub>, Zhan<sub>p</sub>en<sub>g</sub> Zen<sub>g</sub>, Rudrasis Chakrabort<sub>y</sub>, Min<sub>g</sub>xin<sub>g</sub> Tan, Glenn Fun<sub>g,</sub> Yin Li<sub>,</sub> and Vikas Sin<sub>g</sub>h. 2021. N<sub>y</sub>strömformer: A n<sub>y</sub>ström-based al<sub>g</sub>orithm for approximating self-attention. In Proceedings of the AAAI conference on artificial intelligence. 14138–14148.

[33] Zhijian Xu, Ailing Zeng, and Qiang Xu. 2024. FITS: Modeling Time Series with 10� Parameters. In The Twelfth International Conference on Learning Representations.

[34] Kun Yi, Qi Zhang, Wei Fan, Shoujin Wang, Pengyang Wang, Hui He, Ning An, D<sub>e</sub>f<sub>u</sub> Li<sub>a</sub>n<sub>,</sub> L<sub>o</sub>n<sub>g</sub>bin<sub>g</sub> C<sub>ao,</sub> <sub>a</sub>nd Zh<sub>e</sub>nd<sub>o</sub>n<sub>g</sub> Ni<sub>u.</sub> 2023<sub>.</sub> Fr<sub>eque</sub>n<sub>cy</sub>-d<sub>o</sub>m<sub>a</sub>in MLP<sub>s</sub> <sub>a</sub>r<sub>e</sub> more efective learners in time series forecasting. Advances in Neural Information Processing Systems 36 (2023), 76656–76679.

[35] Annan Yu, Danielle C Maddix, Boran Han, Xi<sub>y</sub>uan Zhan<sub>g</sub>, Abdul Fatir Ansari, Ol<sub>e</sub>k<sub>san</sub>d<sub>r</sub> Sh<sub>c</sub>h<sub>ur,</sub> Ch<sub>r</sub>i<sub>s</sub>t<sub>os</sub> F<sub>a</sub>l<sub>ou</sub>t<sub>sos,</sub> A<sub>n</sub>d<sub>rew</sub> G<sub>or</sub>d<sub>on</sub> Wil<sub>son,</sub> Mi<sub>c</sub>h<sub>ae</sub>l W Mahone<sub>y,</sub> and Yu<sub>y</sub>an<sub>g</sub> Wan<sub>g</sub>. 2025. Understandin<sub>g</sub> Transformers for Time Series: Rank Structure, Flow-of-ranks, and Compressibility. arXiv preprint arXiv:2510.03358 (2025).

[36] Ailin<sub>g</sub> Zen<sub>g</sub>, Muxi Chen, Lei Zhan<sub>g</sub>, and Qian<sub>g</sub> Xu. 2023. Are transformers efective for time series forecasting?. In Proceedings ofthe AAAI conference on artificial intelligence. 11121–11128.

[37] Boning Zhang, Haishuai Wang, Zehong Hu, Jiajun Wang, Hongyi Zhang, and Jia Jia. 2026. Re-Difusion: Modelin<sub>g</sub> Latent Residuals with Difusion for Time-Series Forecasting. In Proceedings ofthe ACM Web Conference 2026. 7390–7400.

[38] Fan Zhan<sub>g</sub>, Shimin<sub>g</sub> Fan, and Hua Wan<sub>g</sub>. 2026. Time-tk: A multi-ofset tem<sub>p</sub>oral i<sub>n</sub>t<sub>erac</sub>ti<sub>on</sub> f<sub>ramewor</sub>k <sub>com</sub>bi<sub>n</sub>i<sub>ng</sub> t<sub>rans</sub>f<sub>ormer an</sub>d k<sub>o</sub>l<sub>mogorov-arno</sub>ld <sub>ne</sub>t<sub>wor</sub>k<sub>s</sub> for time series forecasting. In Proceedings ofthe ACM Web Conference 2026. 7495– 7506.

[39] Jianqi Zhang, Jingyao Wang, Wenwen Qiang, Fanjiang Xu, and Changwen Zheng. 2026. Enhancin<sub>g</sub> lar<sub>g</sub>e lan<sub>g</sub>ua<sub>g</sub>e models for time-series forecastin<sub>g</sub> via vectorinjected in-context learning. In Proceedings ofthe ACM Web Conference 2026. 7211–7222.

[40] Xu Zhan<sub>g</sub>, Pen<sub>g</sub> Wan<sub>g</sub>, Yichen Li, and Wei Wan<sub>g</sub>. 2026. Amortized <sub>p</sub>redictabilit<sub>y</sub>- aware training framework for time series forecasting and classification. In Proceedings ofthe ACM Web Conference 2026. 5624–5635.

[41] Xu Zhan<sub>g</sub>, Qiton<sub>g</sub> Wan<sub>g</sub>, Pen<sub>g</sub> Wan<sub>g</sub>, and Wei Wan<sub>g</sub>. 2026. Semixer: Semantics enhanced ml<sub>p</sub>-mixer for multiscale mixin<sub>g</sub> and lon<sub>g</sub>-term time series forecastin<sub>g</sub>. In Proceedings ofthe ACM Web Conference 2026. 5636–5647.

[42] Hao<sub>y</sub>i Zhou, Shan<sub>g</sub>han<sub>g</sub> Zhan<sub>g</sub>, Jie<sub>q</sub>i Pen<sub>g</sub>, Shuai Zhan<sub>g</sub>, Jianxin Li, Hui Xion<sub>g</sub>, <sub>an</sub>d W<sub>anca</sub>i Zh<sub>ang.</sub> 2021<sub>.</sub> I<sub>n</sub>f<sub>ormer:</sub> B<sub>eyon</sub>d <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t t<sub>rans</sub>f<sub>ormer</sub> f<sub>or</sub> l<sub>ong se-</sub> quence time-series forecasting. In Proceedings of the AAAI conference on artificial intelligence. 11106–11115.

[43] Tian Zhou, Zi<sub>q</sub>in<sub>g</sub> Ma, Qin<sub>g</sub>son<sub>g</sub> Wen, Xue Wan<sub>g</sub>, Lian<sub>g</sub> Sun, and Ron<sub>g</sub> Jin. 2022. F<sub>e</sub>df<sub>ormer:</sub> F<sub>requency en</sub>h<sub>ance</sub>d d<sub>ecompose</sub>d t<sub>rans</sub>f<sub>ormer</sub> f<sub>or</sub> l<sub>ong-</sub>t<sub>erm ser</sub>i<sub>es</sub> forecasting. In International conference on machine learning. PMLR, 27268–27286.

## A Related Work

## A.1 Time Series Forecasting

Time series forecastin<sub>g</sub> faces challen<sub>g</sub>es in modelin<sub>g</sub> lon<sub>g</sub>-term de<sub>p</sub>endencies and com<sub>p</sub>lex d<sub>y</sub>namics. Traditional RNNs and CNNs <sub>su</sub>f<sub>er</sub> f<sub>rom</sub> <sub>sca</sub>l<sub>a</sub>bilit<sub>y</sub> b<sub>o</sub>ttl<sub>enec</sub>k<sub>s</sub> <sub>an</sub>d difi<sub>cu</sub>lti<sub>es</sub> i<sub>n</sub> h<sub>an</sub>dli<sub>ng</sub> l<sub>ong</sub> se<sub>q</sub>uences [2, 23]. In contrast, Transformers have achieved su<sub>p</sub>erior <sub>p</sub>erformance b<sub>y</sub> levera<sub>g</sub>in<sub>g</sub> self-attention mechanisms<sub>,</sub> s<sub>p</sub>urrin<sub>g</sub> th<sub>e</sub> d<sub>eve</sub>l<sub>opmen</sub>t <sub>o</sub>f <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>an</sub>d i<sub>n</sub>t<sub>erpre</sub>t<sub>a</sub>bl<sub>e</sub> t<sub>empora</sub>l <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> [10, 28] such as Informer [42], Autoformer [29], and PatchTST [17]. Meanwhile<sub>,</sub> fre<sub>q</sub>uenc<sub>y</sub>-domain methods efectivel<sub>y</sub> ca<sub>p</sub>ture <sub>p</sub>eriodic and s<sub>p</sub>ectral <sub>p</sub>atterns in time series throu<sub>g</sub>h Fourier or wavelet transforms [15]. Re<sub>p</sub>resentative works, includin<sub>g</sub> Fredformer [19], FreTS [34], and FITS [33], demonstrate com<sub>p</sub>etitive advanta<sub>g</sub>es in <sub>ex</sub>t<sub>rac</sub>ti<sub>ng</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l f<sub>ea</sub>t<sub>ures.</sub> R<sub>ecen</sub>t <sub>s</sub>t<sub>u</sub>di<sub>es</sub> f<sub>ur</sub>th<sub>er</sub> i<sub>n</sub>t<sub>egra</sub>t<sub>e</sub> t<sub>empora</sub>l and fre<sub>q</sub>uenc<sub>y</sub>-domain information<sub>,</sub> <sub>p</sub>ro<sub>p</sub>osin<sub>g</sub> h<sub>y</sub>brid modelin<sub>g</sub> a<sub>pp</sub>roaches. For exam<sub>p</sub>le<sub>,</sub> CDX-Net combines multi<sub>p</sub>le network structures for feature fusion [9], FEDformer unifies decom<sub>p</sub>osition and fre<sub>q</sub>uenc<sub>y</sub> anal<sub>y</sub>sis [43], and TimeMixer++ enhances tem<sub>p</sub>oral re<sub>p</sub>resentation throu<sub>g</sub>h multi-scale sam<sub>p</sub>lin<sub>g</sub> and attention mechanisms, advancin<sub>g</sub> the field [27].

![](images/2912d542147da4b535f2060cd3e5c2b610122dfb8f27c2e053f57c5db9398cb3.jpg)  
Figure 13: Rank expansion via patchification.

## A.2 Rank Analysis in Transformer

Lo<sub>w</sub>-rank attention research aims to im<sub>p</sub>ro<sub>v</sub>e Transformer eficienc<sub>y</sub> throu<sub>g</sub>h low-rank a<sub>pp</sub>roximations or <sub>p</sub>arameterization. Lowrank a<sub>pp</sub>roximation methods, such as Linformer [26], N<sub>y</sub>stromformer [32], and Performer [3], utilize low-dimensional <sub>p</sub>rojections, <sub>anc</sub>h<sub>or</sub> t<sub>o</sub>k<sub>ens,</sub> <sub>or</sub> <sub>ran</sub>d<sub>om</sub> f<sub>ea</sub>t<sub>ure</sub> <sub>mapp</sub>i<sub>ngs</sub> t<sub>o</sub> <sub>re</sub>d<sub>uce</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> com<sub>p</sub>utational com<sub>p</sub>lexit<sub>y</sub> to linear. Low-rank <sub>p</sub>arameterization methods [7, 8, 11], directl<sub>y</sub> learn low-rank u<sub>p</sub>date matrices for efi-<sub>c</sub>i<sub>en</sub>t fi<sub>ne-</sub>t<sub>un</sub>i<sub>ng</sub> b<sub>u</sub>t l<sub>ac</sub>k th<sub>eore</sub>ti<sub>ca</sub>l <sub>gu</sub>id<sub>ance on</sub> th<sub>e un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> <sub>causes o</sub>f l<sub>ow-ran</sub>k b<sub>e</sub>h<sub>av</sub>i<sub>or.</sub> Th<sub>eore</sub>ti<sub>ca</sub>l <sub>ana</sub>l<sub>ys</sub>i<sub>s revea</sub>l<sub>s</sub> th<sub>a</sub>t <sub>pure</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on ne</sub>t<sub>wor</sub>k<sub>s su</sub>f<sub>er</sub> f<sub>rom a</sub> d<sub>ou</sub>bl<sub>e-exponen</sub>ti<sub>a</sub>l <sub>co</sub>ll<sub>apse</sub> i<sub>n rep-</sub> <sub>resen</sub>t<sub>a</sub>ti<sub>on ran</sub>k <sub>as</sub> d<sub>ep</sub>th i<sub>ncreases, a p</sub>h<sub>enomenon</sub> k<sub>nown as ran</sub>k colla<sub>p</sub>se [18, 22]. Studies indicate that residual connections, la<sub>y</sub>er <sub>norma</sub>li<sub>za</sub>ti<sub>on, an</sub>d f<sub>ee</sub>d<sub>-</sub>f<sub>orwar</sub>d <sub>ne</sub>t<sub>wor</sub>k<sub>s can e</sub>f<sub>ec</sub>ti<sub>ve</sub>l<sub>y m</sub>iti<sub>ga</sub>t<sub>e</sub> this issue and maintain re<sub>p</sub>resentational diversit<sub>y</sub> [4, 30]. Additi<sub>ona</sub>ll<sub>y,</sub> th<sub>e</sub> <sub>re</sub>l<sub>a</sub>ti<sub>ons</sub>hi<sub>p</sub> b<sub>e</sub>t<sub>ween</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> h<sub>ea</sub>d di<sub>mens</sub>i<sub>on</sub> <sub>an</sub>d <sub>sequence</sub> l<sub>eng</sub>th i<sub>n</sub>fl<sub>uences</sub> th<sub>e</sub> <sub>ran</sub>k <sub>o</sub>f <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> <sub>ma</sub>t<sub>r</sub>i<sub>ces,</sub> th<sub>ere</sub>b<sub>y</sub> constrainin<sub>g</sub> model ex<sub>p</sub>ressiveness [1].

In time series forecastin<sub>g,</sub> se<sub>q</sub>uence embeddin<sub>g</sub>s exhibit ra<sub>p</sub>idl<sub>y</sub> deca<sub>y</sub>in<sub>g</sub> sin<sub>g</sub>ular value s<sub>p</sub>ectra<sub>,</sub> naturall<sub>y</sub> concentratin<sub>g</sub> in low-<sub>ran</sub>k <sub>su</sub>b<sub>spaces</sub> <sub>an</sub>d <sub>en</sub>d<sub>ow</sub>i<sub>ng</sub> <sub>a</sub>tt<sub>en</sub>ti<sub>on</sub> l<sub>ayers</sub> <sub>w</sub>ith i<sub>n</sub>h<sub>eren</sub>t <sub>com-</sub> <sub>p</sub>ressibilit<sub>y</sub> [35]. However, existin<sub>g</sub> work still lacks a unified theo-<sub>re</sub>ti<sub>ca</sub>l f<sub>ramewor</sub>k t<sub>o sys</sub>t<sub>ema</sub>ti<sub>ca</sub>ll<sub>y gu</sub>id<sub>e</sub> th<sub>e overa</sub>ll d<sub>es</sub>i<sub>gn an</sub>d o<sub>p</sub>timization of time series Transformers from a rank <sub>p</sub>ers<sub>p</sub>ecti<sub>v</sub>e.

## B Visualization of Rank Expansion via Patching

Fi<sub>g</sub>ure 13 illustrates the rank ex<sub>p</sub>ansion efect discussed in Secti<sub>o</sub>n 2<sub>.</sub> P<sub>o</sub>int-<sub>w</sub>i<sub>se</sub> t<sub>o</sub>k<sub>e</sub>niz<sub>a</sub>ti<sub>o</sub>n $( P = 1 )$ limits the re<sub>p</sub>resentation to <sub>a</sub> <sub>ran</sub>k<sub>-</sub>1 <sub>s</sub>t<sub>ruc</sub>t<sub>ure,</sub> <sub>w</sub>hil<sub>e</sub> <sub>pa</sub>t<sub>c</sub>hifi<sub>ca</sub>ti<sub>on</sub> <sub>en</sub>l<sub>arges</sub> th<sub>e</sub> <sub>a</sub>tt<sub>a</sub>i<sub>na</sub>bl<sub>e</sub> <sub>ran</sub>k

b<sub>oun</sub>d t<sub>o m</sub>i<sub>n</sub> $( N , P )$ as stated in Pro<sub>p</sub>osition 1<sub>, p</sub>ro<sub>v</sub>idin<sub>g</sub> additional inde<sub>p</sub>endent directions for subse<sub>q</sub>uent re<sub>p</sub>resentation learnin<sub>g</sub>.

## C Proof of Theorem 1

We consider the two-la<sub>y</sub>er MLP embeddin<sub>g</sub>:

$$
\mathbf { H } = \phi ( \mathbf { X } _ { \rho } \mathbf { W } _ { 1 } + \mathbf { 1 } _ { N } \mathbf { b } _ { 1 } ^ { \top } ) \mathbf { W } _ { 2 } + \mathbf { 1 } _ { N } \mathbf { b } _ { 2 } ^ { \top } ,
$$

<sub>w</sub>h<sub>ere</sub> ${ \bf X } _ { p } \in \mathbb { R } ^ { N \times P }$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> i<sub>npu</sub>t <sub>pa</sub>t<sub>c</sub>h <sub>ma</sub>t<sub>r</sub>i<sub>x an</sub>d $\phi$ i<sub>s a co</sub>ntinuous non-<sub>p</sub>ol<sub>y</sub>nomial activation function. We show that there <sub>e</sub>xi<sub>s</sub>t MLP <sub>pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>r<sub>s suc</sub>h th<sub>a</sub>t

$$
\mathrm { r a n k } ( \mathbf { H } ) = \operatorname* { m i n } ( N , D ) .
$$

## C.1 Spanning Property of Non-polynomial Activations

We first establish a <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> of non-<sub>p</sub>ol<sub>y</sub>nomial activations. For an<sub>y</sub> distinct in<sub>pu</sub>t <sub>p</sub>oints $\{ x _ { i } \} _ { i = 1 } ^ { N } \subset \mathbb { R } ^ { { \stackrel { . } { P } } }$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\begin{array} { r } { \Phi ( \boldsymbol { w } , b ) = [ \phi ( \boldsymbol { w } ^ { \top } \boldsymbol { x } _ { 1 } + b ) , . . . , \phi ( \boldsymbol { w } ^ { \top } \boldsymbol { x } _ { N } + b ) ] ^ { \top } . } \end{array}
$$

Lemma 1. The set of vectors $\{ \Phi ( w , b ) : w \in \mathbb { R } ^ { P } , b \in \mathbb { R } \}$ spans $\mathbb { R } ^ { N }$

Proof. Assume that the ima<sub>g</sub>e of Φ does not s<sub>p</sub>an $\mathbb { R } ^ { N }$ <sub>.</sub> Th<sub>en</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s a non-zero vec</sub>t<sub>or</sub> $\boldsymbol { \alpha } \in \mathbb { R } ^ { N }$ satisf<sub>y</sub>in<sub>g</sub>

$$
\sum _ { i = 1 } ^ { N } \alpha _ { i } \phi ( \boldsymbol { w ^ { \intercal } x _ { i } } + \boldsymbol { b } ) = 0
$$

f<sub>or</sub> <sub>a</sub>ll <sub>�</sub> <sub>an</sub>d �<sub>.</sub>

Sin<sub>ce</sub> th<sub>e</sub> in<sub>pu</sub>t <sub>po</sub>int<sub>s a</sub>r<sub>e</sub> di<sub>s</sub>tin<sub>c</sub>t<sub>,</sub> th<sub>e</sub>r<sub>e e</sub>xi<sub>s</sub>t<sub>s a</sub> dir<sub>ec</sub>ti<sub>o</sub>n $\boldsymbol { v } \in \mathbb { R } ^ { P }$ suc<sup>h</sup> t<sup>h</sup>at t<sup>h</sup>e projecte<sup>d</sup> va<sup>l</sup>ues $\beta _ { i } ~ = ~ \boldsymbol { v } ^ { \top } \boldsymbol { x } _ { i }$ <sub>a</sub>r<sub>e pa</sub>ir<sub>w</sub>i<sub>se</sub> di<sub>s</sub>tin<sub>c</sub>t<sub>.</sub> Settin<sub>g</sub> $w = t v$ <sub>g</sub><sup>i</sup>ves

$$
\sum _ { i = 1 } ^ { N } \alpha _ { i } \phi ( t \beta _ { i } + b ) = 0 ,
$$

for all �<sub>,</sub> �. This contradicts the discriminator<sub>y</sub> <sub>p</sub>ro<sub>p</sub>ert<sub>y</sub> of continuous non-<sub>p</sub>ol<sub>y</sub>nomial activations<sub>,</sub> which states that no non-trivial li<sub>near com</sub>bi<sub>na</sub>ti<sub>on o</sub>f th<sub>e</sub>i<sub>r</sub> t<sub>rans</sub>l<sub>a</sub>t<sub>e</sub>d <sub>an</sub>d <sub>sca</sub>l<sub>e</sub>d f<sub>unc</sub>ti<sub>ons can</sub> <sub>van</sub>i<sub>s</sub>h id<sub>en</sub>ti<sub>ca</sub>ll<sub>y.</sub> Th<sub>ere</sub>f<sub>ore,</sub> th<sub>e vec</sub>t<sub>ors</sub> $\Phi ( w , b )$ <sup>s</sup>p<sup>an</sup> $\mathbb { R } ^ { N }$ □

## C.2 Maximum Rank Achievability

$$
K = \operatorname* { m i n } ( N , D ) .
$$

B<sub>y</sub> Lemma 1<sub>,</sub> the acti<sub>v</sub>ation o<sub>u</sub>t<sub>pu</sub>ts s<sub>p</sub>an $\mathbb { R } ^ { N }$ <sub>,</sub> <sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub> th<sub>ere</sub> exist � linearl<sub>y</sub> inde<sub>p</sub>endent vectors

$$
v _ { 1 } , \ldots , v _ { K }
$$

<sub>genera</sub>t<sub>e</sub>d b<sub>y</sub> <sub>su</sub>it<sub>a</sub>bl<sub>e</sub> <sub>parame</sub>t<sub>ers</sub> $( w _ { j } , b _ { j } )$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub>

$$
\boldsymbol { v } _ { j } = [ \phi ( \boldsymbol { w } _ { j } ^ { \top } \boldsymbol { x } _ { 1 } + b _ { j } ) , . . . , \phi ( \boldsymbol { w } _ { j } ^ { \top } \boldsymbol { x } _ { N } + b _ { j } ) ] ^ { \top } .
$$

Constr<sub>u</sub>ct the first MLP la<sub>y</sub>er <sub>u</sub>sin<sub>g</sub>

$$
\mathbf { W } _ { 1 } = [ w _ { 1 } , \ldots , w _ { K } ] , \qquad \mathbf { b } _ { 1 } = [ b _ { 1 } , \ldots , b _ { K } ] ^ { \top } .
$$

Th<sub>e</sub> hidd<sub>en</sub> <sub>represen</sub>t<sub>a</sub>ti<sub>on</sub> b<sub>ecomes</sub>

$$
\mathbf { H } _ { h i d } = \phi ( \mathbf { X } _ { \mathcal { P } } \mathbf { W } _ { 1 } + \mathbf { 1 } _ { N } \mathbf { b } _ { 1 } ^ { \top } ) = [ v _ { 1 } , \ldots , v _ { K } ] ,
$$

<sub>w</sub>hi<sub>c</sub>h <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\operatorname { r a n k } ( \mathbf { H } _ { h i d } ) = K .
$$

Fina<sup>ll</sup>y, c<sup>h</sup>oosing t<sup>h</sup>e output projection $\mathbf { W } _ { 2 }$ t<sub>o</sub> <sub>map</sub> th<sub>ese</sub> i<sub>n</sub>d<sub>e-</sub> <sub>p</sub>endent hidden directions into the out<sub>p</sub>ut s<sub>p</sub>ace <sub>y</sub>ields

$$
\mathrm { r a n k } ( \mathbf { H } ) = K = \operatorname* { m i n } ( N , D ) .
$$

Therefore<sub>,</sub> nonlinear MLP embeddin<sub>g</sub>s can achieve the maximum <sub>a</sub>tt<sub>a</sub>i<sub>na</sub>bl<sub>e ran</sub>k <sub>un</sub>d<sub>er</sub> th<sub>e g</sub>i<sub>ven samp</sub>l<sub>e an</sub>d hidd<sub>en</sub> di<sub>mens</sub>i<sub>ons.</sub>

## D Attention Rank Analysis

We anal<sub>y</sub>ze how attention to<sub>p</sub>olo<sub>gy</sub> afects the rank <sub>p</sub>reservation of value re<sub>p</sub>resentations durin<sub>g</sub> token interaction. Given <sub>q</sub>uer<sub>y,</sub> ke<sub>y,</sub> an<sup>d</sup> va<sup>l</sup>ue projections:

$$
{ \bf Q } = { \bf X } { \bf W } _ { Q } , \quad { \bf K } = { \bf X } { \bf W } _ { K } , \quad { \bf V } = { \bf X } { \bf W } _ { V } ,
$$

t<sup>h</sup>e attention output a<sup>f</sup>ter output projection can <sup>b</sup>e written as

$$
\mathbf { O } = \mathbf { A X } _ { V } ,
$$

<sub>w</sub>h<sub>ere</sub>

$$
\mathbf { X } _ { V } = \mathbf { V } \mathbf { W } ^ { O } ,
$$

and A denotes the corresponding attention aggregation matrix.

Th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng ana</sub>l<sub>ys</sub>i<sub>s</sub> f<sub>ocuses on w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e a</sub>tt<sub>en</sub>ti<sub>on aggre-</sub> gation matrix A preserves the rank of the value representation.

## D.1 Full-rank Potential of Bidirectional Attention

BiA i<sub>s</sub> n<sub>o</sub>t inh<sub>e</sub>r<sub>e</sub>ntl<sub>y</sub> r<sub>a</sub>nk d<sub>e</sub>fi<sub>c</sub>i<sub>e</sub>nt<sub>.</sub> W<sub>e s</sub>h<sub>ow</sub> th<sub>a</sub>t it<sub>s a</sub>tt<sub>e</sub>nti<sub>o</sub>n matrix can ac<sup>h</sup>ieve <sup>f</sup>u<sup>ll</sup> ran<sup>k</sup> un<sup>d</sup>er suita<sup>bl</sup>e projection parameters.

Given an input matrix X with distinct rows, there exists a projection direction v such that

$$
\mathbf { u } = \mathbf { X } \mathbf { v }
$$

contains <sub>p</sub>airwise distinct val<sub>u</sub>es. Choosin<sub>g</sub> one-dimensional <sub>qu</sub>er<sub>y</sub> an<sup>d</sup> <sup>k</sup>ey projections as

$$
{ \sf Q } = \alpha { \bf u } , \qquad { \bf K } = \beta { \bf u } ,
$$

<sub>p</sub>roduces the attention score matrix

$$
\mathbf { S } = \gamma \mathbf { u } \mathbf { u } ^ { \top } .
$$

Although S is rank one, the element-wise exponential matrix <sub>genera</sub>t<sub>e</sub>d b<sub>y</sub> th<sub>e so</sub>ft<sub>max opera</sub>ti<sub>on,</sub>

$$
\mathbf { E } _ { i j } = \exp ( \gamma u _ { i } u _ { j } ) ,
$$

i<sub>s</sub> f<sub>u</sub>ll <sub>ran</sub>k f<sub>or</sub> di<sub>s</sub>ti<sub>nc</sub>t $\{ u _ { i } \}$ . Since the softmax normalization onl<sub>y</sub> introduces an invertible dia<sub>g</sub>onal scalin<sub>g,</sub> the resultin<sub>g</sub> bidirectional <sub>a</sub>tt<sub>e</sub>nti<sub>o</sub>n m<sub>a</sub>trix <sub>ca</sub>n <sub>ac</sub>hi<sub>eve</sub>

$$
\operatorname { r a n k } ( \mathbf { A } _ { \mathrm { b i d } } ) = n .
$$

Th<sub>ere</sub>f<sub>ore,</sub> BiA it<sub>se</sub>lf d<sub>oes no</sub>t <sub>su</sub>f<sub>er</sub> f<sub>rom an</sub> i<sub>n</sub>t<sub>r</sub>i<sub>ns</sub>i<sub>c ran</sub>k li<sub>m-</sub> itation. Its limitation is the absence of a str<sub>u</sub>ct<sub>u</sub>ral <sub>gu</sub>arantee for <sub>p</sub>reservin<sub>g</sub> rank durin<sub>g</sub> re<sub>p</sub>eated token interaction.

## D.2 Rank Preservation of Causal Attention

Lemma 2 (Invertibility of Causal Attention). The causal attention matrix $\mathbf { A } _ { \mathrm { c a u s a l } }$ is invertible:

$$
\operatorname { r a n k } ( \mathbf { A } _ { \mathrm { c a u s a l } } ) = n .
$$

P<sub>roof.</sub> D<sub>ue</sub> t<sub>o</sub> th<sub>e causa</sub>l <sub>mas</sub>k<sub>,</sub> $\mathbf { A } _ { \mathrm { c a u s a l } }$ is a lower trian<sub>g</sub>ular m<sub>a</sub>trix<sub>.</sub> It<sub>s</sub> di<sub>ago</sub>n<sub>a</sub>l <sub>e</sub>ntri<sub>es</sub> <sub>a</sub>r<sub>e</sub>:

$$
( \mathbf { A } _ { \mathrm { c a u s a l } } ) _ { i i } = \frac { \exp ( \mathbf { S } _ { i i } ) } { \sum _ { j = 1 } ^ { i } \exp ( \mathbf { S } _ { i j } ) } ,
$$

<sub>w</sub>h<sub>ere</sub> ${ \bf S } = { \bf Q } { \bf K } ^ { \top } / \sqrt { d _ { k } }$

Since all ex<sub>p</sub>onential terms are strictl<sub>y</sub> <sub>p</sub>ositive<sub>,</sub>

$$
( \mathbf { A _ { \mathrm { { c a u s a l } } } } ) _ { i i } > 0 .
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
\operatorname* { d e t } ( \mathbf { A } _ { \mathrm { c a u s a l } } ) = \prod _ { i = 1 } ^ { n } ( \mathbf { A } _ { \mathrm { c a u s a l } } ) _ { i i } > 0 ,
$$

<sub>w</sub>hi<sub>c</sub>h <sub>proves</sub> th<sub>a</sub>t $\mathbf { A } _ { \mathrm { c a u s a l } }$ i<sub>s</sub> in<sub>ve</sub>rtibl<sub>e.</sub>

## D.2.1 Proof of Theorem 2.

Proof<sub>.</sub> L<sub>e</sub>t

$$
\mathbf { X } _ { V } = \mathbf { V } \mathbf { W } ^ { O } .
$$

Fr<sub>o</sub>m L<sub>e</sub>mm<sub>a</sub> $2 , \mathbf { A } _ { \mathrm { c a u s a l } }$ i<sub>s</sub> i<sub>nver</sub>tibl<sub>e.</sub> Th<sub>ere</sub>f<sub>ore,</sub>

$$
\operatorname { r a n k } ( \mathbf { O } _ { \operatorname { c a u s a l } } ) = \operatorname { r a n k } ( \mathbf { A } _ { \operatorname { c a u s a l } } \mathbf { X } _ { V } ) = \operatorname { r a n k } ( \mathbf { X } _ { V } ) .
$$

F<sub>or</sub> bidi<sub>rec</sub>ti<sub>ona</sub>l <sub>a</sub>tt<sub>en</sub>ti<sub>on,</sub> th<sub>e ran</sub>k <sub>o</sub>f

$$
\mathbf { O } _ { \mathrm { b i d } } = \mathbf { A } _ { \mathrm { b i d } } \mathbf { X } _ { V }
$$

i<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>ne</sub>d b<sub>y</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e</sub> <sub>nu</sub>ll <sub>space</sub> <sub>o</sub>f $\mathbf { A } _ { \mathrm { b i d } }$ i<sub>n</sub>t<sub>ersec</sub>t<sub>s</sub> <sub>w</sub>ith th<sub>e</sub> <sub>co</sub>l<sub>umn</sub> <sub>space</sub> <sub>o</sub>f $\mathbf { X } _ { V }$

B<sub>y</sub> th<sub>e</sub> <sub>ran</sub>k<sub>-nu</sub>llit<sub>y</sub> th<sub>eorem</sub> <sub>res</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d t<sub>o</sub> $\operatorname { C o l } ( \mathbf { X } _ { V } )$

$$
\mathrm { r a n k } ( \mathbf { O } _ { \mathrm { b i d } } ) = \mathrm { r a n k } ( \mathbf { X } _ { V } ) - \mathrm { d i m } ( \mathrm { k e r } ( \mathbf { A } _ { \mathrm { b i d } } ) \cap \mathrm { C o l } ( \mathbf { X } _ { V } ) ) .
$$

Hence,

$$
\operatorname { r a n k } ( \mathbf { O } _ { \operatorname { c a u s a l } } ) - \operatorname { r a n k } ( \mathbf { O } _ { \mathrm { b i d } } ) = \dim ( \operatorname { k e r } ( \mathbf { A } _ { \mathrm { b i d } } ) \cap \operatorname { C o l } ( \mathbf { X } _ { V } ) ) \geq 0 .
$$

Th<sub>ere</sub>f<sub>ore,</sub>

$$
\mathrm { r a n k } ( \mathbf { O } _ { \mathrm { c a u s a l } } ) \geq \mathrm { r a n k } ( \mathbf { O } _ { \mathrm { b i d } } ) .
$$

E<sub>qua</sub>lit<sub>y</sub> h<sub>o</sub>ld<sub>s</sub> <sub>exac</sub>tl<sub>y</sub> <sub>w</sub>h<sub>en</sub>

$$
\operatorname { k e r } ( \mathbf { A } _ { \mathrm { b i d } } ) \cap \operatorname { C o l } ( \mathbf { X } _ { V } ) = \{ 0 \} .
$$

## E Additional Experimental Results

Diferent Patch Sizes. As shown in Figure 14, we systematically <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d th<sub>e s</sub>i<sub>ngu</sub>l<sub>ar va</sub>l<sub>ue</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f <sub>em</sub>b<sub>e</sub>ddi<sub>ng ma</sub>t<sub>r</sub>i<sub>ces</sub> under diferent <sub>p</sub>atch sizes (1, 4, 8, 16) across 7 datasets. The ex-<sub>per</sub>i<sub>men</sub>t<sub>a</sub>l <sub>resu</sub>lt<sub>s s</sub>h<sub>ow</sub> th<sub>a</sub>t f<sub>or mos</sub>t d<sub>a</sub>t<sub>ase</sub>t<sub>s,</sub> th<sub>e s</sub>i<sub>ngu</sub>l<sub>ar va</sub>l<sub>ue</sub> d<sub>ecay curves o</sub>f th<sub>e</sub>i<sub>r em</sub>b<sub>e</sub>ddi<sub>ng represen</sub>t<sub>a</sub>ti<sub>ons genera</sub>ll<sub>y ex</sub>hibit f<sub>u</sub>ll<sub>-ran</sub>k <sub>or near-</sub>f<sub>u</sub>ll<sub>-ran</sub>k <sub>c</sub>h<sub>arac</sub>t<sub>er</sub>i<sub>s</sub>ti<sub>cs, even un</sub>d<sub>er</sub> th<sub>e sma</sub>ll<sub>es</sub>t <sub>p</sub>atch size (1). This indicates that the model can learn hi<sub>g</sub>hl<sub>y</sub> diverse <sub>an</sub>d li<sub>near</sub>l<sub>y</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>represen</sub>t<sub>a</sub>ti<sub>ons</sub> f<sub>or</sub> th<sub>ese</sub> d<sub>a</sub>t<sub>ase</sub>t<sub>s,</sub> <sub>w</sub>ith th<sub>e</sub>i<sub>r</sub> <sub>e</sub>f<sub>ec</sub>ti<sub>ve</sub> <sub>ran</sub>k <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>tl<sub>y</sub> <sub>surpass</sub>i<sub>ng</sub> th<sub>e</sub> i<sub>n</sub>h<sub>eren</sub>t b<sub>o</sub>ttl<sub>enec</sub>k of the ori<sub>g</sub>inal univariate in<sub>p</sub>ut (which has an u<sub>pp</sub>er rank bound of 1).

![](images/fd1a29ff67a9531b3b617fe7bf4cf4c5fd1ff1e6c206b47a44ac2e2b79c2ca7d.jpg)

![](images/49d8c0e74cb33193cb6d2a58f0100c18a765d88dcad3e22c99548ee75050efaa.jpg)

![](images/7845423d323c2d5f709879428eb912c14848ae95ded2fd4bd74da6b54039bb5f.jpg)

![](images/d5454a643085253534e524cf0dc489a880cdb89bc5e0458f294122fd7334bc4d.jpg)  
ETTh1 ETTh2 ETTm1 ETTm2 Electricity Exchange Weather

Figure 14: Rank Distribution of Embedding Matrices (Patch Size = 16).  
(a) Bidirectional Attention  
![](images/b66414cee3216d4789876d607732dd5641decadc5a68a0b7c01c1b860379d9b0.jpg)

(b) Hourglass Attention  
![](images/3bffbc450296cd6d6168a52409f9b762c9e6c6d7750def6542bf7a82adc85bc0.jpg)  
Figure 15: PCA visualization of token representation evolution across deep layers in a pure attention network. The color gradient indicates the token index within the sequence.

Visualization of Representation Collapse. As illustrated in Figure 15, t<sup>h</sup>e evo<sup>l</sup>utionary trajectories o<sup>f</sup> t<sup>h</sup>e to<sup>k</sup>en representations dif<sub>er</sub> d<sub>ras</sub>ti<sub>ca</sub>ll<sub>y</sub> d<sub>epen</sub>di<sub>ng on</sub> th<sub>e c</sub>h<sub>osen a</sub>tt<sub>en</sub>ti<sub>on mec</sub>h<sub>an</sub>i<sub>sm:</sub> BiA: The top panel (Figure 15a) demonstrates that while tokens are initiall<sub>y</sub> scattered in the in<sub>p</sub>ut s<sub>p</sub>ace<sub>,</sub> the<sub>y</sub> ra<sub>p</sub>idl<sub>y</sub> assimilate durin<sub>g</sub> forward <sub>p</sub>ro<sub>p</sub>a<sub>g</sub>ation. B<sub>y</sub> La<sub>y</sub>er 4<sub>,</sub> the re<sub>p</sub>resentations of all tokens com<sub>p</sub>letel<sub>y</sub> colla<sub>p</sub>se into a sin<sub>g</sub>le indistin<sub>g</sub>uishable <sub>p</sub>oint.

Thi<sub>s v</sub>i<sub>sua</sub>li<sub>zes</sub> th<sub>e ex</sub>t<sub>reme over-smoo</sub>thi<sub>ng an</sub>d <sub>severe ran</sub>k <sub>co</sub>l<sub>-</sub> lapse inherent to deep unmasked attention. HgA: The bottom panel (Fi<sub>g</sub>ure 15b) shows that H<sub>g</sub>A successfull<sub>y p</sub>reserves re<sub>p</sub>resentational diversit<sub>y</sub>. Even at the final la<sub>y</sub>er (La<sub>y</sub>er 6), the tokens maintain a distinct and structured s<sub>p</sub>read. This <sub>p</sub>rovides direct and com<sub>p</sub>ellin<sub>g</sub> <sub>v</sub>is<sub>u</sub>al e<sub>v</sub>idence of H<sub>g</sub>A’s str<sub>u</sub>ct<sub>u</sub>ral s<sub>up</sub>eriorit<sub>y</sub> in circ<sub>u</sub>m<sub>v</sub>entin<sub>g</sub> <sub>ran</sub>k <sub>co</sub>ll<sub>apse</sub> <sub>an</sub>d <sub>preserv</sub>i<sub>ng</sub> f<sub>u</sub>ll<sub>-ran</sub>k <sub>capac</sub>it<sub>y.</sub>