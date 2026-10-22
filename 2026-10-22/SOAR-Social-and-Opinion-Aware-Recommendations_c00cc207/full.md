# SOAR: Social and Opinion-Aware Recommendations

Giannis Christoforidis<sup>1\*</sup> and Apostolos N. Papadopoulos<sup>1</sup>

1 <sup>\*</sup>School of Informatics, Aristotle University of Thessaloniki, AUTH Campus, Thessaloniki, 54124, Greece.

\*Corresponding author(s). E-mail(s): icchrist@csd.auth.gr; Contributing authors: papadopo@csd.auth.gr;

## Abstract

One of the enduring challenges in online platforms is recommendation. Both communities and advertisers face dificulties in analyzing complex user behavior, identifying habits and evolving preferences, and providing timely recommendations that are relevant to user interests. In particular, point-of-interest (POI) recommendation poses the additional challenge of identifying the factors that influence user behavior, capturing user tendencies, and generating accurate, personalized recommendations for each individual. In this work, we propose SOAR, a Social and Opinion-Aware Recommendation model that incorporates both the user’s friendship network and textual reviews to assess whether a location is suitable for a specific user, with a particular emphasis on the sentiment of shared reviews. Experimental results in real-world datasets demonstrate that incorporating reviews into the recommendation process improves the precision of the recommendation.

Keywords: Point-of-Interest Recommendation, Social Network Analysis, Sentiment-Aware Recommender Systems

## 1 Introduction

Social networks and, in general, online presence have increased dramatically in recent years. People frequently log their activities on the Internet and use social networks to share opinions, thoughts, and other personal experiences. Due to the increase in trafic on such networks, there have been a plethora of research opportunities regarding analyzing the user’s data in order to understand their tendencies and preferences and thus be able to predict/suggest future personalized actions. A challenging approach has always been POI recommendation, especially with the many factors that contribute to the user’s final choice of location.

Traditional recommendation approaches often rely solely on user-location visitation, overlooking the complex social and contextual factors that influence users’ preferences and idiosyncrasies. A user can be influenced by a number of factors, including social, temporal, geographical, and other. A challenge lies in detecting these aspects and also applying them to each user specifically to achieve personalized recommendations.

A wide variety of parameters can afect a user’s preference, but ultimately the most indicative result of the user’s visitation is the final review. The language behind each review can indicate not only the sentiment of the user, but also their mentality and thought process, which, in turn, can reveal several factors that the user weighs more when choosing both a location to visit and to avoid. Using the text in POI reviews, we can formulate a semantic network by aligning the common factors detected in the text and combining them with the friendship network. As a result, we can pinpoint each users’ important positive and negative factor and provide suggestions on POIs with similar mentality of the specific user. It is crucial to understand and point out that every user can find something appeasing where another user finds it appalling - a characteristic that can be very challenging.

Figure 1 illustrates a simple example of a recommendation challenge that can be encountered in the real world. The silhouette on the left depicts the public/dominant opinion over a hotel in London - it’s a top rated location with many visitors and positive reviews. However, we observe some negative traits in the review of User A, and the friend of this user, User B, considers visiting that location. As they also share many common preferences such as the quality of the view and the prices, it is unlikely that the user will prefer that location despite its general good fame.

In this paper, we present a Social and Opinion-Aware Recommendation (SOAR) model that integrates text-based reviews with user network information and historical activity to generate personalized recommendations. Our goal is to evaluate all locations based on their historic activity, but more importantly to identify each user’s idiosyncrasies by analyzing the user’s friend’s reviews, and extracting the traits they find either appealing or appalling. While it is important to take account the public opinion on popular locations we also weight heavily upon the user’s peculiarities.

More specifically, we employ SBERT [1] to encode textual reviews into semantic embeddings. This allows us to encapture the diferent traits a user may like or dislike, and present them in a vectorized form. Additionally, we apply a FAISS [2] clustering process that groups related reviews — and, consequently, associated locations—based on shared characteristics. This process yields a graph structure that represents connections between locations that share similar characteristics or traits. Additionally, we calculate the sentiment value of each review with Hugging Face’s pipeline in an efort to detect each user’s idiosyncrasies from the user’s group of friends. We then incorporate this graph with the user’s social network and historical check-in data, generating unified embeddings through RandNE[3].

![](images/999c10e7395336c1cb4b6a52a4f32f242f49300ef7c053ef93ef8db47f2f023a.jpg)  
Fig. 1 Recommendation Example

Finally, to improve personalization, we adjust each embedding by weighting reviews from friends of the user more heavily in the recommendation process. To achieve that, we assign the initial score on each location based on the public sentiment and then we value both negative and positive reviews on the user’s friends specifically. The score of the afected locations is then altered based on the sentiment of the user’s friends’ reviews, adding or subtracting points accordingly to the ratio of their friends’ mentality, and as such each suggestion is now made personalized for the specific user.

The contribution of our work is summarized as follows:

• Sentiment Value: We split reviews into positive and negative based on their context. We then insert them into a graph along with the friends so the can determine the final score of each location not only by its total reviews, but by the average reaction from the friends of each user.

• Semantic Similarity: Each review has a diferent point of view and is greatly inspired by the mentality of the user. It is important to contextualize the review and denote the important factors that characterize the location. By doing so, we can determine if a user prefers to visit this location based on those factors regardless of the sentiment of the review author, as each person has their own preferences.

• Personalized Recommendations: Each person has their own preferences and a circle of friends that influence the user’s decision-making process. Our algorithm integrates both general sentiment and volume of review for each location, as well as opinions of the user’s friends, since public opinion and peer sentiment may difer.

The rest of the work is organized as follows. Related work is presented in Section 2. The methodology and technical details of the proposed approach are given in Section 3. Performance evaluation results are presented and discussed in Section 4, while Section 5 concludes the work and briefly discusses future research directions.

## 2 Related Work

In this section, we review prior research in the areas of both POI recommendation and text analysis.

A considerable body of recent research in this field has focused on POI recommendation based on users’ historical interactions with their environment. Some works [4–6] engage into the social aspect of the users, aiming to establish their preferences based on their friendship network and other social connections. Other models [7, 8] focus on the geographical cues to improve recommendation accuracy, often including spatiotemporal information to include long-term and short term interest as well as seasonal data. In addition, researchers [9, 10] have also attempted to combine the social and spatio-temporal geographical information in order to incorporate more data-points and thus achieve better personalized recommendations. While it is important to include a variety of information, we should not overlook the textual data such as reviews, as they are more informative and precise to the users’ preference than any other data.

In an efort to capture both global and local contextual features along with POI attributes, Liu et al. [11] proposed KDRank, a model designed to identify general popularity patterns as well as localized contextual information based on users’ friendship network. While distinguishing the influence of friends from dominant behavioral trends is crucial, their approach employs a variation of the PageRank algorithm [12] to weight the relative importance of each friend in the recommendation process. However, this method inherently limits the contribution of less influential friends and, in the absence of textual or semantic information, may overlook valuable contextual cues. In our work, we address this limitation by modeling the distinct behavioral traits of each user and integrating the friendship network adaptively to adjust the final ranking scores of candidate locations.

We very often see multiple variations of BERT [13] being used throughout the research field [14, 15] in the text/review analysis section.

Similarly to our work, Huang et al. [16] uses BERT-CL, a clustering modification of the BERT algorithm, which focuses on embedding the representation of the text and also classifies positive and negative reviews into a score-based system. While on the same path, it completely ignores every other parameter, such as the friendship network and only calculates the future public opinion based on historic data and as such does not personalize each recommendation.

SBERT was also used in MMPOI, a model developed by Xu et al. [17], which also incorporates text to categorize locations. More specifically, they use Meta-data, reviews and transform images into descriptions to further contextualize the location. Further, they also include geographic and multi-modal trajectory path, a technique we’ve also seen in other works[18–20] which also utilize sequences in POI visitation. While incorporating many aspects of a user’s preferences they don’t seem to align user’s with their review’s characteristic and identify the traits each user, or group of users, are attracted to.

Wang et al. [21] proposed a model called STaTRL, which finds aspects within the review text and characterizes each as positive or negative in an efort to determine the boons and banes of each POI. Whereas there are similarities in finding aspects in each review, they do not take into account the friendship network and depend heavily on the ”popular” opinion. Reviews tend to be heavily subjective opinions and as such valuing the opinions of ones friends outweigh the dominant consensus. By linking the review text’s aspects and connecting them to the users we can identify the user’s tendencies and as such determine the ones that contribute positively or negatively regardless of the generic ”popular” mentality.

An et. al. [22] focused on sentiment-based tourist spot recommendation using a CNN-LSTM model to infer ratings from textual reviews and incorporate environmental context: weather, season, and spatial proximity. Initially, a TF-IDF mechanism is applied to eliminate words that are overly dependent on specific tourist spots, as identified through morpheme analysis. CNN extracts the pooling layer that is concatenated by LSTM for sentence encoding. Although efective for general context-aware recommendations, their approach does not model user-specific preferences or relationships between users. In contrast, our method builds on the idea of exploiting review text but further integrates user-network information and personalized embeddings to provide individualized POI recommendations.

Contractor et al. [23] address the problem of answering natural-language questions about POIs by leveraging tourism reviews. Their work frames POI retrieval as a question-answering task that combines textual and spatial reasoning, employing a multi-stage pipeline that clusters review sentences, retrieves relevant POIs, and subsequently re-ranks them based on semantic relevance to the query. In contrast, our approach, instead of processing queries in isolation, incorporates user-specific information, such as social network connections and historical interactions, alongside review semantics. By encoding review texts and constructing a graph that links semantically related reviews and locations, our model captures both contextual and relational dependencies among users and POIs. This graph-based representation enables personalized and interpretable recommendations, whereas their framework primarily emphasizes general question understanding and retrieval efectiveness.

Yao et al. [24] propose a sequential POI recommendation framework that explicitly leverages user reviews and feedback by first applying a two-level clustering over review text and then modeling user trajectories with an LSTM to capture sequential visitation patterns and user similarity. Their method emphasizes sequential modeling of POI discovery (using recurrent networks) and incorporates multifaceted contextual features to mitigate data sparsity and cold-start problems via learned user similarity from feedback. Instead of engaging sequential pattern learning and feedback-driven similarity, we instead identify each user’s personal idiosyncrasies by connecting him to the patterns in their friends’ reviews by utilizing the friendship network - something that is missing from their model, yet we consider essential for providing personalized recommendations.

While our work leverages user reviews, it difers from prior review-aware recommendation approaches in its primary focus. Existing methods largely concentrate on extracting rich semantic representations from review text and optimizing text modeling techniques. In contrast, our approach emphasizes the integration of reviewderived signals with the underlying friendship network, aiming to capture the interplay between textual preferences and social structure. As such, our contribution lies in multi-source fusion rather than advances in review text modeling itself. To ensure a fair evaluation, we compare our method against variants that utilize review information without combining it with the friendship network, thereby isolating the impact of the social component. Consequently, direct comparisons with methods that primarily target sophisticated review representation learning are not entirely appropriate, as they address a diferent aspect of the recommendation problem.

## 3 SOAR: The Proposed Approach

In this section we first present the fundamentals of our research and then describe in detail our proposed methodology and algorithm for the data analysis and personalized recommendation system. Table 1 depicts the most commonly used symbols.

## 3.1 Fundamentals

Table 1 Frequently Used Symbols
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td> $\mathcal { U } , \mathcal { L } , \mathcal { R } , \mathcal { C }$ </td><td>Set of Users,  $\mathcal { U } = \{ u _ { 1 } , . . . , u _ { n } \} ,$  set of locations</td></tr><tr><td></td><td> $\mathcal { L } = \{ l _ { 1 } , . . . , l _ { m } \}$  set of Reviews</td></tr><tr><td></td><td> $\mathcal { R } = \{ r _ { 1 } , . . . , r _ { k } \} _ { . }$  4 set of Check-ins  $\mathcal { C } = \mathbf { \bar { \{ } }  c _ { 1 } , . . . , c _ { o } \mathbf { \bar { \{ } } $ </td></tr><tr><td> $\mathcal { F } _ { u _ { i } }$ </td><td>set of Users&#x27; Friends  $\mathcal { F } = \dot { \{ f u _ { 1 } , . . . , f u _ { n } \} }$ </td></tr><tr><td> $c _ { u _ { i } }$ </td><td>0 user&#x27;s check-in</td></tr><tr><td> $\mathbf { U } ^ { \mathcal { R } } , \mathbf { U } ^ { \mathcal { G } }$ </td><td>Embeddings of Reviews and Graphs respectively</td></tr><tr><td></td><td> $\mathbf { U } ^ { \mathcal { R } } = ( u _ { 1 } ^ { \mathcal { R } } , . . . , u _ { d _ { r } } ^ { \mathcal { R } } )$ </td></tr><tr><td></td><td> $\mathbf { U } ^ { \mathcal { G } } = ( u _ { 1 } ^ { \mathcal { G } } , . . . , u _ { d _ { g } } ^ { \mathcal { G } ^ { ' } } )$ </td></tr></table>

Theorem 1. POI: is a unique location where at least one user has checked-in. Every POI is represented as a tuple: $< l _ { i d } ,$ longitude, latitude>.

Theorem 2. Check-in: is a visit from User u to a Location l with review r made after the visit forming a tuple $c _ { i } = < u , l , r > .$ A user can have many check-ins to the same place.

Theorem 3. User-User graph: is a unipartite undirected graph representing the friendship relations among users in the social network, denoted as $G _ { u u } = ( \mathcal { U } , \mathcal { E } )$ , where U is the set of users and $\mathcal { E } \subseteq \mathcal { U } \times \mathcal { U }$ is the set of edges representing friendship relations.

Theorem 4. Review-Review graph: is a unipartite undirected graph representing similarity relationships among reviews, denoted as $G _ { r r } = ( \mathcal { R } , \mathcal { E } )$ , where $\mathcal { R }$ is the set of reviews and $\mathcal { E } \subseteq \mathcal { R } \times \mathcal { R }$ is the set of edges connecting similar reviews. Each review $r _ { i } \in \mathcal { R }$ is represented by an embedding vector:

$$
\mathbf { u } _ { i } ^ { \mathcal { R } } \in \mathbb { R } ^ { d }\tag{1}
$$

obtained from a pretrained language model. The edge set $\mathcal { E }$ is constructed based on nearest-neighbor relationships in the embedding space. Specifically, for each $r _ { i } \in \mathcal { R }$ we define edges to its k nearest neighbors:

$$
\mathcal { E } = \bigcup _ { i = 1 } ^ { | \mathcal { R } | } \left\{ \{ r _ { i } , r _ { j } \} \bigm | r _ { j } \in N N _ { k } \big ( \mathbf { u } _ { i } ^ { \mathcal { R } } \big ) \right\} .\tag{2}
$$

where $N N _ { k } ( \cdot )$ denotes the set of k nearest neighbors in the embedding space.

Theorem 5. User-Location graph: is a bipartite undirected consisting of connections based on the check-ins. It is denoted as $G _ { u l } = \mathcal { U } \cup \mathcal { L }$ where $\mathcal { L }$ is the set of Locations and U is the set of Users, and ${ \mathcal { C } } \subseteq { \mathcal { U } } \times { \mathcal { L } }$ the set of check-ins. We define a bipartite undirected graph

$$
G _ { u l } = ( { \mathcal { U } } \cup { \mathcal { L } } , { \mathcal { E } } ) ,\tag{3}
$$

where

$$
\mathcal { E } = \{ \{ u , \ell \} \ | \ ( u , \ell ) \in \mathcal { C } , \ u \in \mathcal { U } , \ \ell \in \mathcal { L } \} .\tag{4}
$$

## 3.2 Sentiment Analysis of Reviews

To extract the polarity of user reviews, we employed a transformer-based sentiment classifier using the Hugging Face implementation of the DistilBERT model. This model performs binary classification, assigning each review either a positive or negative sentiment label.

Formally, for each review $r _ { i } \in \mathcal { R }$ , we compute its sentiment as follows:

$$
s _ { i } = f _ { \mathrm { B E R T } } ( r _ { i } )\tag{5}
$$

where $f _ { \mathrm { B E R T } }$ denotes the DistilBERT-based sentiment function. The model outputs two logits corresponding to the probability of each class (positive and negative):

$$
\mathbf { p } _ { i } = \mathrm { s o f t m a x } ( \mathbf { z } _ { i } ) \quad \mathrm { w h e r e } \quad \mathbf { z } _ { i } = \mathrm { B E R T } ( r _ { i } )\tag{6}
$$

with

$$
\mathbf { p } _ { i } = [ p _ { i } ^ { ( \mathrm { n e g } ) } , p _ { i } ^ { ( \mathrm { p o s } ) } ] \quad \mathrm { s u c h ~ t h a t } \quad p _ { i } ^ { ( c ) } = \frac { e ^ { z _ { i } ^ { ( c ) } } } { e ^ { z _ { i } ^ { ( \mathrm { p o s } ) } } + e ^ { z _ { i } ^ { ( \mathrm { n e g } ) } } }\tag{7}
$$

The final sentiment label is determined by:

$$
y _ { i } = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } p _ { i } ^ { \mathrm { ( p o s ) } } \geq p _ { i } ^ { \mathrm { ( n e g ) } } } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right.\tag{8}
$$

where $y _ { i } = 1$ denotes a positive sentiment and $y _ { i } = 0$ denotes a negative sentiment.

## 3.3 Proposed Approach

In this section, we explain in detail the functionality of our algorithm. We start by describing the processing of the data available from the dataset, then the method of forming the embeddings, and finally the process we use to calculate the final personalized recommendations.

First, we form the graphs as defined in Theorems 3, 4, and 5. More specifically, for the reviews we used Sentence-BERT[1], a modification of the BERT [13] architecture optimized for sentence-level embeddings as depicted in Figure 2. Each review $r =$ $( w _ { 1 } , w _ { 2 } , \cdots , w _ { n } )$ is tokenized and mapped to the input embeddings by:

![](images/6dce3856a4a7bb30240ffbda66a8be1b1be8a39ecffebe15e5a140c25e4c8305.jpg)  
Fig. 2 Bert example.

$$
\mathbf { x } _ { i } = \mathrm { E m b e d } ( w _ { i } ) + \mathrm { P o s E n c } ( i )\tag{9}
$$

Where Embed(·) is the word embedding function and PosEnc(i) denotes the positional encoding. The resulting token embeddings $\left( \mathbf { x } _ { 1 } , \cdots , \mathbf { x } _ { n } \right)$ are processed by a transformer encoder consisting of multiple self-attention layers, producing contextualized token representations

$$
\mathbf { H } = ( \mathbf { h } _ { 1 } , \dots , \mathbf { h } _ { n } ) = \operatorname { T r a n s f o r m e r } ( \mathbf { x } _ { 1 } , \dots , \mathbf { x } _ { n } ) , \quad \mathbf { H } \in \mathbb { R } ^ { n \times d }\tag{10}
$$

To derive a fixed-size sentence embedding, SBERT applies a pooling operation over the sequence of token embeddings. In our implementation, we used mean pooling, defined as

$$
\mathbf { s } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { h } _ { i }\tag{11}
$$

where $s = \in \mathbb { R } ^ { d }$ represents the embedding of the entire review text. The similarity between two reviews $r _ { a }$ and $r _ { b }$ is then computed using cosine similarity:

$$
\mathrm { s i m } ( \mathbf { u } _ { a } ^ { \mathcal { R } } , \mathbf { u } _ { b } ^ { \mathcal { R } } ) = \frac { \mathbf { u } _ { a } ^ { \mathcal { R } } \cdot \mathbf { u } _ { b } ^ { \mathcal { R } } } { \| \mathbf { u } _ { a } ^ { \mathcal { R } } \| \| \mathbf { u } _ { b } ^ { \mathcal { R } } \| }\tag{12}
$$

SBERT itself is fine-tuned using a Siamese network structure with objectives such as semantic textual similarity regression, natural language inference classification, or triplet loss ranking as illustrated in Figure 3. For example, in the STS setting, given a gold similarity score $y \in [ 0 , 5 ]$

$$
\mathcal { L } = \left( y - \sin ( \mathbf { u } _ { a } ^ { \mathcal { R } } , \mathbf { u } _ { b } ^ { \mathcal { R } } ) \right) ^ { 2 }\tag{13}
$$

Algorithm 0 describes the procedure to produce the Review Graph $G ^ { \mathcal { R } }$ and the Sentiment of the reviews. As described in previous sections, we use SBERT to produce the initial review embeddings, then we use FAISS to connect nearby neighbors and finally we apply Hugging Face pipeline to produce the final sentiment rating that characterize a review as positive or negative.

With the 3 graphs formed, we re-identify each node and combine them to one unipartite graph. Then, we use the RandNE[3] to calculate the final embeddings. More specifically we first form the adjacency matrix of G, $\mathbf { A } \in \mathbb { R } ^ { N \times N }$ and preserve the high-order proximity by considering powers of A:

$$
\mathbf { S } = \left( w _ { 0 } \mathbf { I } + w _ { 1 } \mathbf { A } + w _ { 2 } \mathbf { A } ^ { 2 } + \cdot \cdot \cdot + w _ { q } \mathbf { A } ^ { q } \right)\tag{14}
$$

where I is the identity matrix and $w _ { i }$ are the pre-defined weights controlling the contribution of k-hop neighbors. This step ensures that nodes connected indirectly through multiple hops influence each other’s embedding. The final embeddings are computed via iterative random projection:

$$
{ \bf U } ^ { \mathcal { G } } = { \bf S } \cdot { \bf R }\tag{15}
$$

where R is a random Gaussian matrix $R _ { i j } \sim \mathcal { N } ( 0 , 1 )$ . Algorithm 2 describes the process of calculating the embeddings using Gaussian Random Matrix with iterative random projections.

## 3.4 Personalized Recommendations

For a given future check-in event marked as a tuple $c = < u _ { i } , l _ { j } >$ where user $u _ { i } \in \mathcal { U }$ visits location $l _ { j } \in \mathcal { L }$ we first retrieve all locations within a radius r around $l _ { j } \mathbf { \hat { \mathbf { \Omega } } }$

$$
\mathcal { N } _ { r } ( \ell _ { j } ) = \big \{ \ell _ { t } \in \mathcal { L } \ \big | \ \mathrm { d i s t } ( \ell _ { t } , \ell _ { j } ) \leq r \big \} .\tag{16}
$$

Each location $l _ { t } \in \mathcal { N } _ { r } ( \ell _ { j } )$ is assigned a friend-review weight $w _ { i , t } \in [ 0 , 1 ] .$ defined as the proportion of positive reviews among the user’s friends who have visited that location and alters its score:

![](images/56e225b3c8c2889727cedd2931934c60f5447b5ddda952ac7b4295cdf8658331.jpg)  
Fig. 3 Similarity Score Computation.

Algorithm 1 Review Graph and Sentiment Analysis   
Require: Reviews: R, Sentence Transformer Model $m ^ { b } .$ , Pipeline Model $m ^ { p } ,$ Review   
radius range   
Ensure: Review Graph $G ^ { \mathcal { R } }$ , Review Sentiment S   
1: $\mathbf { U } ^ { \mathcal { R } } \gets m ^ { b }$ .encode(R) ▷ Generate Review Embeddings   
2: map ← faiss.index $( \mathbf { U } ^ { \mathcal { R } } )$ ▷ Generate Review Graph   
3: for r in R do   
4: G<sup>R</sup>.add(< r, faiss.neighbors(range) >)   
5: end for   
6: pipeline.initialize(m<sup>p</sup>) ▷ Sentiment Analysis   
7: for r in R do   
8: if pipeline.class $f y ( r ) > 0 . 5$ then   
9: S<sub>r</sub> ← positive   
10: else   
11: S ← negative   
12: end if   
13: end for   
14: return U

Algorithm 2 Calculation of the Embeddings   
Require: User, POIs and User-Review Graphs: $G _ { u } , G _ { l } , G _ { u l }$ , Embedding dimension   
d, k-hop weights w   
Ensure: Embeddings U   
1: G ← combine $\left( G _ { u } , G _ { l } , G _ { u l } \right)$   
2: N ← count nodes(G)   
3: A ← sparse matrix(G)   
4: $\mathbf { A }  \mathbf { A } + \mathbf { A } ^ { T }$ ▷ Calculate the Adjacency matrix   
5: $\mathbf { U } _ { l i s t } $ Gaussian Random Matrix $( N \times d )$   
6: for i in w do   
7: U<sub>list</sub>.append(A · U<sub>list</sub>[−1]) ▷ Initialize the Gaussian Random Matrix and   
random projection   
8: end for   
9: U ← empty array(d)   
10: for i in w do   
11: U ← w<sub>i</sub> × U<sub>list</sub>[w<sub>i</sub>] ▷ Calculate the embeddings   
12: end for   
13: return U

$$
w _ { i , t } = \left\{ \begin{array} { l l } { \frac { P _ { i , t } } { T _ { i , t } } \cdot \sigma ( s _ { i } ) , } & { \mathrm { i f ~ } T _ { i , t } > 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{17}
$$

Where $P _ { i , t }$ and $T _ { i , t }$ denote the number of positive and total reviews left by the friends of user $u _ { i }$ for location $l _ { t }$

Algorithm 3 describes the process we use for each location to calculate the personalized score by modifying the embeddings according to each user’s friendship network. For every location we alter its score based on the user’s reviews and value them more than the dominant popular opinion.

We continue to compute the personalized similarity score for each location $l _ { t } \colon$

$$
\mathbf { s } _ { i , t } = \mathbf { U } _ { i } ^ { \mathcal { G } } \cdot \left( \mathbf { U } _ { l _ { t } } \right) ^ { \top } + \mathbf { w _ { i , t } }\tag{18}
$$

Finally, we rank all candidate locations in descending order according to their relevance scores $s _ { i , t }$ . The location with the highest score is considered the most suitable recommendation for user $u _ { i } .$ . Moreover, the model supports Top-N recommendation by selecting the N highest-ranked locations, thus providing a ranked list of recommendations.

Algorithm 3 Producing Recommendations   
Require: Graph Embeddings: U, Friendship Network: F, User’s Reviews: R, User:   
u, User’s Location: lat, lon   
Ensure: Final Score   
1: NL ← find nearby(lat, lon)   
2: Modified U ← U(NL) ▷ Find Nearby Locations and Modify the Embeddings   
3: final scores ← U(u) · Modified U<sup>T</sup> ▷ Inner Product   
4: std ← standard deviation(final scores) ▷ Calculate the final scores   
5: for l in NL do   
6: positive reviews, total reviews ← count reviews of friends(u)   
7: if total reviews equals 0 then   
8: location score ← 0.5   
9: else   
positive reviews   
10: location score ←   
total reviews   
11: end if   
12: final scores(l) ← final scores(l) + std ∗ location score   
13: end for   
14: return final scores

## 3.5 Example

In this section, we describe how our algorithm works by providing some examples that describe its key points. Starting with the review handling, we try to connect locations with each other by detecting and connecting their characteristics. Figure 4 shows the relation between 2 reviews of 2 diferent locations.

Although reviews may contain overlapping vocabulary, our analysis goes beyond lexical similarity by generating semantic embeddings for each review. The reviews are then linked based on their proximity in the embedding space, enabling us to capture deeper semantic relationships, as illustrated in Figure 5.

To personalize the recommendation process we take into account the user’s friends’ reviews. Figure 6 shows an example of a user’s friends ranking highly a specific location whereas the common opinion difer. It is not uncommon for a small group of people to value a characteristic of a location that many people avoid. In this example, since we value the user’s opinions more than the general consensus, we will value the specific location higher than normal in spite of the overwhelming negative reviews from the rest of the users.

![](images/5cb985badbc6b9842614ab0792f628afb005655bbb73f851dcb760cc9f0e6685.jpg)  
Fig. 4 Example of Review Connection.

## 4 Performance Evaluation

The source code associated with the proposed methodology is publicly available at https://github.com/thedx4/soar.

## 4.1 Dataset

We use the yelp dataset <sup>1</sup> that contains approximately 7 million reviews, with an average of 105 words per review and 10 friends per user, as shown in Table 2. Furthermore, the data set includes 150,000 locations in the United States, of which 100,000 have at least 100 neighboring locations within a 1-kilometer radius.

## 4.2 Review Data

The review graph is generated following the procedure outlined in Theorem 4. We employ the pretrained model ”all-MiniLM-L6-v2” with an embedding dimension of 384, which ofers a favorable trade-of between computational cost and embedding quality, ensuring eficient yet expressive representations of textual reviews.

To identify the nearest neighbors we used Faiss, and to determine an appropriate search radius r for the range-based neighbor retrieval, we conducted an empirical analysis across multiple candidate values. For each radius, we performed a range search over a random sample of queries and recorded descriptive statistics regarding the number of neighbors retrieved per query (minimum, maximum, average, and extreme counts).

![](images/36f775dab2a183edc988b0d46e908da29ed91398510c80b930e8f7afecbac481.jpg)  
Fig. 5 Example of Review Connection in Vector Space.

Table 2 Yelp Dataset statistical data.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Total Number of Reviews</td><td>6,990,280</td></tr><tr><td>Total Number of Words</td><td>732,415,837</td></tr><tr><td>Total Number of Locations</td><td>150,346</td></tr><tr><td>Average Number of Words</td><td>104.78</td></tr><tr><td>Median Review Length (words)</td><td>75</td></tr><tr><td>Std Deviation of Review Length (words)</td><td>97.92</td></tr><tr><td>Longest Review (words)</td><td>1070</td></tr><tr><td>Average Number of Friends</td><td>10.02</td></tr></table>

The radius was selected such that the average number of neighbors per query fell within a predefined target range ensuring suficient neighborhood density, while simultaneously avoiding excessively large neighborhoods. Table 3 shows the average number of neighbors along with the maximum number of neighbors and the number of nodes that exceed 200 neighbors. As we observer from the graph, higher values result in exponentially higher connections making the graph more dense.

![](images/e3448bdacd36f8cb144ad6c110ceda6b3a61b9d6af4f739d6a49006ba6cf5fd9.jpg)  
Fig. 6 Example of Friend’s influence.

Table 3 Review statistics for diferent radius values
<table><tr><td>Radius</td><td>Avg</td><td>Max</td><td>&gt;200</td></tr><tr><td>0.500</td><td>1.83</td><td>153</td><td>0</td></tr><tr><td>0.525</td><td>2.94</td><td>330</td><td>2</td></tr><tr><td>0.550</td><td>5.58</td><td>630</td><td>5</td></tr><tr><td>0.575</td><td>11.30</td><td>1166</td><td>12</td></tr><tr><td>0.600</td><td>23.91</td><td>2248</td><td>26</td></tr><tr><td>0.625</td><td>51.03</td><td>4336</td><td>49</td></tr><tr><td>0.650</td><td>108.99</td><td>7677</td><td>88</td></tr><tr><td>0.675</td><td>229.26</td><td>12623</td><td>141</td></tr><tr><td>0.700</td><td>476.20</td><td>19231</td><td>232</td></tr></table>

We selected the balanced radius of 0.625 with limiting the number of connections to 50 closest neighbors so that it preserves the local structure of the embedding space without over-saturating the similarity graph with weak or distant connections. The final review graph network result an average number of 7.66 connections with the most tight cluster exceeding 50 connections and its standard deviation is 14.8. As such, we avoid over-clustering tight review groups while maintaining enough edges to characterize them.

To determine the polarity (positive or negative) of each review, we employed a pretrained transformer-based sentiment analysis model from the Hugging Face library. Specifically, we used the ”distilbert-base-uncased-finetuned-sst-2-english model”, which provides a binary classification between positive and negative sentiment. The model was accessed through the transformers’ pipeline(”sentiment-analysis”) interface, which performs tokenization, inference, and softmax scoring. Each review’s sentiment ultimately contributes to determine the user’s group of friends’ mentality over a specific network and weight them against the popular opinion as described in section 3.4.

## 4.3 Evaluation Metrics

To evaluate the performance of the recommendation model, we employ standard classification metrics derived from the confusion matrix components: True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN), and Top-K hits. They are defined as follows:

• True Positives (TP): The number of instances where we suggest the location for the user, and the user had a positive sentiment.

• False Positives (FP): The number of instances where we suggest the location for the user, and the user had a negative sentiment.

• False Negatives (FN): The number of instances where we did not suggest the location for the user, and the user had a positive sentiment.

• True Negatives (TN): The number of instances where we did not suggest the location for the user, and the user had a negative sentiment.

• Top-K hits: The number of instances where we suggest the location for the user, and the user visited the location regardless of the sentiment.

Using these quantities, the evaluation metrics are calculated using the following equations:

$$
\mathrm { P r e c i s i o n } = { \frac { T P } { T P + F P } }\tag{19}
$$

$$
{ \mathrm { R e c a l l } } = { \frac { T P } { T P + F N } }\tag{20}
$$

$$
{ \mathrm { F 1 - s c o r e } } = 2 \times { \frac { { \mathrm { P r e c i s i o n } } \times { \mathrm { R e c a l l } } } { { \mathrm { P r e c i s i o n } } + { \mathrm { R e c a l l } } } }\tag{21}
$$

$$
\mathrm { A c c u r a c y } = { \frac { T P + T N } { T P + F P + F N + T N } }\tag{22}
$$

$$
\mathrm { T o p - K \ h i t s } = { \frac { T P + F P } { T P + F P + F N + T N } }\tag{23}
$$

Precision reflects the proportion of correctly identified positive instances among all predicted positives, while Recall measures the ability to identify all relevant instances. The F1-score provides a harmonic mean of Precision and Recall, balancing both metrics. Finally, Accuracy quantifies the overall correctness of the model’s predictions. Finally, the top-K hits indicate the accuracy of the algorithm while ignoring the final sentiment - which indicates the efectiveness of the algorithm to predict ultimately the user’s final choice.

## 4.4 Experimental Results

For our experiments, we used an embedding dimension of 128, a spatial search radius of 400 meters, and a training split comprising 80% of the available data. To further assess the contribution of our approach, we compared performance on a subset of test cases where users were located within the vicinity of businesses reviewed by their friends.

We compared the performance of our model on the corresponding potential locations between ranges of 400 meters and 1 km as illustrated in Figures 7 through 11 . We compared them in terms of accuracy in Figure 7, recall in Figure 8, F1-score in Figure 9, precision in Figure 10 and top-K hits in Figure 11. Table 4 shows the differences in Recall, F1-score and accuracy between the diferent versions of our model. As the number of potential locations increases exponentially when range increases, we observe the expected drop in related metrics. We chose to move forward with the decreased range as users’ preference to visit a location further away from their present location is increasingly lower.

Another important factor, especially for the Top-1 and Top-5 categories, we observe that the accuracy is significantly greater than the top-K hits in general. This indicates that the users’ most preferential location choice is heavily afected by their friends’ reviews and mentality rather than the popular opinion.

Table 4 Comparison of Recall, F1-score, and Accuracy among versions of our model 2-hop, and 1km across diferent top-K categories.
<table><tr><td rowspan="2">Top-K</td><td colspan="3">Recall (%)</td><td colspan="3">F1-score (%)</td><td colspan="3">Accuracy (%)</td></tr><tr><td>Main</td><td>2-hop</td><td>1km</td><td>Main</td><td>2-hop</td><td>1km</td><td>Main</td><td>2-hop</td><td>1km</td></tr><tr><td>Top-1</td><td>16.37</td><td>16.25</td><td>7.98</td><td>27.17</td><td>27.00</td><td>14.52</td><td>34.22</td><td>34.13</td><td>29.58</td></tr><tr><td>Top-5</td><td>42.22</td><td>41.68</td><td>23.56</td><td>54.72</td><td>54.34</td><td>36.32</td><td>47.66</td><td>47.53</td><td>38.11</td></tr><tr><td>Top-10</td><td>56.59</td><td>55.62</td><td>34.25</td><td>65.08</td><td>64.61</td><td>47.68</td><td>54.54</td><td>54.39</td><td>43.72</td></tr><tr><td>Top-25</td><td>73.66</td><td>72.45</td><td>48.67</td><td>74.53</td><td>74.18</td><td>59.72</td><td>62.29</td><td>62.21</td><td>50.81</td></tr><tr><td>Top-50</td><td>82.76</td><td>81.83</td><td>56.77</td><td>79.05</td><td>78.78</td><td>65.37</td><td>66.89</td><td>66.73</td><td>54.63</td></tr></table>

![](images/0d034c5af0285bf9ee27e346e760235cfda102bfbc8d40bed2dedaed96447718.jpg)  
Fig. 7 Accuracy relative to radius

![](images/2d6d1235320c72f2724e959ac880e9dbbbac329b27a986da90a7a7e628340a42.jpg)  
Fig. 8 Recall relative to radius

![](images/3a52c7cc4ee941882791e8dc57f5fc863bfa3e8a9db3fc615dfffec055859e10.jpg)  
Fig. 9 F1-score relative to radius

![](images/67444b9682f9bb5aeac358a460412a34635e7cceb2d21e502a708bdfb61908cd.jpg)  
Fig. 10 Precision relative to radius

![](images/555fb6d432041063693a198959b89eb13d81354664b3fe44109e8167092c9bd2.jpg)

## Fig. 11 Top-K Hits

We also tested the performance of our algorithm where we include the friends of friends in the recommendation process. This exponentially increases execution time, more than 8 times slower, as shown in Table 5 resulting in a non-viable time especially in large datasets. Additionally, as also seen on Table 4 the diferences between our main algorithm and the 2-hop version is marginally diferent in the end result, if not slightly worse in some cases. This indicates that the user’s friends are enough to determine the user’s tendencies and including friends of friends may dilute the collection of traits important to the user.

Table 5 Performance comparison between friends and friends of friends
<table><tr><td>Model</td><td>Speed (predictions/s)</td><td>Total Execution Time</td></tr><tr><td>SOAR 2 HOP</td><td>7.32</td><td>36 hours</td></tr><tr><td>SOAR</td><td>55.56</td><td>4.6 hours</td></tr></table>

![](images/92d3ed650b1f01769f9b9a1d982d927ed011498572611bf7c78254fc082fa884.jpg)  
Fig. 12 Accuracy comparison between models

We compared the proposed model against a baseline method that does not combine textual reviews and the friendship network of the users. Figures 12 to 16 illustrate the improvement in recommendation performance across all evaluated metrics described in the previous section: accuracy in Figure 12, recall in Figure 13, f1-score in Figure 14, precision in Figure 15 and top-K hits in 16. The comparison spans multiple Top-K categories, ranging from Top-1 to Top-50, demonstrating the performance achieved by the proposed model. The diferences can also be seen at Table 6.

Table 6 Comparison of SOAR and baseline on Recall, F1, and top-K Hits
<table><tr><td rowspan="2">Top-K</td><td colspan="2">Recall (%)</td><td colspan="2">F1 (%)</td><td colspan="2">Top-K Hits (%)</td></tr><tr><td>SOAR</td><td>baseline</td><td>SOAR</td><td>baseline</td><td>SOAR</td><td>baseline</td></tr><tr><td>Top-1</td><td>16.37</td><td>13.37</td><td>27.17</td><td>22.85</td><td>15.38</td><td>12.77</td></tr><tr><td>Top-5</td><td>42.22</td><td>34.44</td><td>54.72</td><td>47.56</td><td>40.68</td><td>33.57</td></tr><tr><td>Top-10</td><td>56.59</td><td>47.01</td><td>65.08</td><td>58.06</td><td>55.33</td><td>46.37</td></tr><tr><td>Top-25</td><td>73.66</td><td>63.60</td><td>74.53</td><td>68.84</td><td>73.16</td><td>63.51</td></tr><tr><td>Top-50</td><td>82.76</td><td>72.79</td><td>79.05</td><td>74.01</td><td>82.54</td><td>72.96</td></tr></table>

We observe a notable increase in performance, especially in the top-5 and top-10 categories which are practically the first page in a recommendation application. The increased accuracy indicates the efect of user’s reviews and the tendencies of the user to follow their friend’s habits and share common preferences with them.

We also observe that precision is decreasing while recall, F1, and accuracy increase with larger K-stems from the inherent trade-of between prediction strictness and coverage in ranked recommendation. When the cutof K expands, the model produces more predictions - positive classifications, capturing a higher proportion of liked POIs, increasing true positives and recall. However, this broader inclusion also introduces more unliked locations among the top-K - increasing false positives, reducing the proportion of correct recommendations and thus lowering precision. In a real-world application, this phenomenon can be explained by the fact that as one moves further down the recommendation list, the likelihood of encountering locations the user does not like increases.

![](images/75d04e5ad10696c025fe87e361d48554f54c67c17b8ef889b53789637f8c9aff.jpg)  
Fig. 13 Recall comparison between models

![](images/b9f200b3bc4cc6f5fd211bc1676c257c19b19321ad8a17710b39ae6aa41488db.jpg)  
Fig. 14 F1-score comparison between models

![](images/6e8192c8c32028d2d1bb96b2ea6038be499169acc2f9412cb72ee3d20b3dab47.jpg)  
Fig. 15 Precision comparison between models

![](images/5433076b9ece98e26dd27f4cb975c539219ee9795a3520529d621d67002b84b7.jpg)  
Fig. 16 Top-K Hits comparison between models

We also evaluated our model using the top-K Hits metric, as illustrated in Figure 16. The top-K Hits measure provides an overall indication of the model’s recommendation accuracy within the top K predicted POIs, irrespective of the user’s eventual sentiment. More specifically, a hit occurs when a POI recommended by the system is actually visited by the user, while a miss indicates that the user visited a POI that was not among the top-K recommendations. This metric captures how efectively the model can anticipate user behavior based solely on ranking relevance, ofering an objective measure of recommendation performance that complements sentiment-based evaluations.

## 5 Conclusions

Point-of-Interest (POI) and item recommendation tasks present significant challenges in the modern era due to the multitude of factors that influence user decision-making. Conventional approaches often capture only a subset of these factors or fail to efectively integrate them. However, since user preferences emerge from the interaction of many interdependent elements rather than isolated influences, it is crucial not only to identify these factors but also to understand their relationships and combined impact. The most challenging aspect of recommendation remains achieving genuine personalization for each individual user.

In this work, we introduced a novel framework that extracts and links meaningful aspects from review texts to uncover individual user tendencies. By incorporating both public sentiment and social influence derived from users’ friends, our approach provides personalized and context-aware POI recommendations. Experimental results demonstrate that integrating these textual and social factors significantly enhances recommendation performance.

Future research could explore the integration of additional contextual signals such as temporal patterns and mobility data. Moreover, applying more advanced language models or graph-based learning methods may further enhance the system’s ability to capture nuanced user preferences and dynamic relationships among locations.

## References

[1] Reimers, N., Gurevych, I.: Sentence-bert: Sentence embeddings using siamese bert-networks. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics, ??? (2019). https://doi.org/10.18653/v1/d19-1410

[2] Johnson, J., Douze, M., J´egou, H.: Billion-scale similarity search with gpus. IEEE Transactions on Big Data 7(3), 535–547 (2021) https://doi.org/10.1109/ TBDATA.2019.2921572

[3] Zhang, Z., Cui, P., Li, H., Wang, X., Zhu, W.: Billion-scale network embedding with iterative random projection. In: 2018 IEEE International Conference on Data Mining (ICDM), pp. 787–796 (2018). https://doi.org/10.1109/ICDM.2018.00094

[4] Meng, L., Liu, Z., Chu, D., Sheng, Q.Z., Yu, J., Song, X.: Poi recommendation for occasional groups based on hybrid graph neural networks. Expert Systems with Applications 237, 121583 (2024)

[5] Liu, Z., Meng, L., Sheng, Q.Z., Chu, D., Yu, J., Song, X.: Poi recommendation for random groups based on cooperative graph neural networks. Information Processing & Management 61(3), 103676 (2024)

[6] Xu, C., Ding, A.S., Zhao, K.: A novel poi recommendation method based on trust relationship and spatial–temporal factors. Electronic Commerce Research and Applications 48, 101060 (2021)

[7] Gan, M., Ma, Y.: Mapping user interest into hyper-spherical space: A novel poi recommendation method. Information Processing & Management 60(2), 103169 (2023)

[8] Fu, J., Gao, R., Yu, Y., Wu, J., Li, J., Liu, D., Ye, Z.: Contrastive graph learning long and short-term interests for poi recommendation. Expert Systems with Applications 238, 121931 (2024)

[9] Zhang, J., Ma, W.: Hybrid structural graph attention network for poi recommendation. Expert Systems with Applications 248, 123436 (2024)

[10] Fang, J., Meng, X., Qi, X.: A top-k poi recommendation approach based on lbsn and multi-graph fusion. Neurocomputing 518, 219–230 (2023)

[11] Liu, Z., Zhang, D., Zhang, C., Bian, J., Deng, J., Shen, G., Kong, X.: Kdrank: Knowledge-driven user-aware poi recommendation. Knowledge-Based Systems

[12] Bianchini, M., Gori, M., Scarselli, F.: Inside pagerank. ACM Trans. Internet Techn. 5(1), 92–128 (2005) https://doi.org/10.1145/1052934.1052938

[13] Devlin, J., Chang, M., Lee, K., Toutanova, K.: BERT: pre-training of deep bidirectional transformers for language understanding. CoRR abs/1810.04805 (2018) 1810.04805

[14] Acharya, M., Mohbey, K.K.: Exploring the evolution, progress, and future of point-of-interest recommendation over location-based social network: a comprehensive review. GeoInformatica 29(3), 305–350 (2025)

[15] Ko, H., Lee, S., Park, Y., Choi, A.: A survey of recommendation systems: recommendation models, techniques, and application fields. Electronics 11(1), 141 (2022)

[16] Huang, S., Wu, X., Wu, X., Wang, K.: Sentiment analysis algorithm using contrastive learning and adversarial training for POI recommendation. Soc. Netw. Anal. Min. 13(1), 75 (2023) https://doi.org/10.1007/S13278-023-01076-X

[17] Xu, Y., Cong, G., Zhu, L., Cui, L.: Mmpoi: A multi-modal content-aware framework for poi recommendations. In: Proceedings of the ACM Web Conference 2024, pp. 3454–3463 (2024)

[18] Christoforidis, G., Kefalas, P., Papadopoulos, A.N., Manolopoulos, Y.: Reline: point-of-interest recommendations using multiple network embeddings. Knowledge and Information Systems 63(4), 791–817 (2021)

[19] Li, P., Rijke, M., Xue, H., Ao, S., Song, Y., Salim, F.D.: Large language models for next point-of-interest recommendation. In: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. SIGIR ’24, pp. 1463–1472. Association for Computing Machinery, New York, NY, USA (2024). https://doi.org/10.1145/3626772.3657840

[20] Qin, Y., Wu, H., Ju, W., Luo, X., Zhang, M.: A difusion model for poi recommendation. ACM Trans. Inf. Syst. 42(2) (2023) https://doi.org/10.1145/ 3624475

[21] Wang, X., Fukumoto, F., Li, J., Yu, D., Sun, X.: Statrl: Spatial-temporal and text representation learning for POI recommendation. Appl. Intell. 53(7), 8286–8301 (2023) https://doi.org/10.1007/S10489-022-03858-W

[22] An, H., Moon, N.: Design of recommendation system for tourist spot using senti ment analysis based on CNN-LSTM. J. Ambient Intell. Humaniz. Comput. 13(3), 1653–1663 (2022) https://doi.org/10.1007/S12652-019-01521-W

[23] Contractor, D., Shah, K., Partap, A., Singla, P., Mausam, M.: Answering poirecommendation questions using tourism reviews. In: Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pp. 281–291 (2021)

[24] Yao, Y., Zhan, H., Noorian, A., Hazratifard, M.: Enhancing poi recommendations on social media: a sequential approach incorporating lstm and user feedback. Computing 107(1), 21 (2025)