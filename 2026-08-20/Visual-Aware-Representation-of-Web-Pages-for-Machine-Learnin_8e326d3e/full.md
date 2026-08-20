# Visual-Aware Representation of Web Pages for Machine Learning Applications

Radek Burget<sup>1[0000−0001−5233−0456]</sup> and Radek Hranický<sup>1[0000−0001−6315−8137]</sup>

Brno University of Technology, Faculty of Information Technology, Bozetechova 2, 612 00 Brno, Czechia

{burgetr,hranicky}@fit.vut.cz

Abstract. Applying machine learning to web pages is challenging due to the need to interpret HTML together with associated resources and perform rendering to obtain a meaningful visual and layout-aware representation. As a result, machine learning over web content remains comparatively underexplored. In this paper, we present a platform for visual-aware representation and machine learning over web pages based on the open-source rendering tool FitLayout. The platform provides a server capable of rendering web pages, explicitly capturing their visual and structural properties in an RDF-based representation, and persisting the rendered documents in an integrated storage. The processing pipeline is controlled via a REST API, while SPARQL queries are used to retrieve structured data suitable as input for machine learning algorithms. By explicitly modeling rendered web pages, including fine-grained layout details, the platform enables dataset sharing and supports the reproducibility of experimental results. The architecture supports the complete dataset preparation workflow, from web page collection and rendering through preprocessing and annotation of content elements to downstream learning tasks. We further provide a Python client library that integrates the platform with standard machine learning workflows. As a demonstration, we show how rendered web pages can be transformed into graph-based representations and used to train graph neural networks for recognizing key content elements, illustrating both the applicability of the approach and the reproducibility of the results.

Keywords: Rendered web pages · Visual-aware document representation · Machine learning for web content · Graph neural networks · Reproducible web data analysis.

## 1 Introduction

Machine learning for document understanding has progressed rapidly, particularly for visually rich inputs, such as scanned pages and PDFs. Many successful methods are layout-aware and represent documents as structured objects, often as graphs processed by graph neural networks (GNNs). Comparable approaches for web pages are still less explored.

Web pages are dificult because their layout is produced dynamically by a browser from HTML and external resources, and the rendering output has no standard, persistent representation. Consequently, most existing methods either analyze the DOM only (ignoring layout) or treat pages as images (losing content structure), limiting the use of layout-aware models.

Building on our previous work [3], we use RDF descriptions of rendered pages as a machine-learning-ready representation and introduce a software platform based on our FitLayout open-source framework. The platform exposes an API and SPARQL access to the rendered representation, supports the sharing of rendered datasets, and provides a Python client for integration into common machine learning workflows.

We illustrate our approach using a case study on recognizing key content elements. We describe how to construct an annotated dataset of rendered web pages, and how to employ SPARQL queries to transform these pages into graph representations suitable for subsequent processing with GNNs. Finally, we explain how the resulting dataset can be shared to facilitate reproducibility of the results and to enable the application of alternative machine learning methods to the same underlying data.

## 2 Related Work

Machine learning techniques have been efectively applied to identify key content elements in PDF documents, including table localization and structural parsing with neural methods designed for complex document layouts [5, 11] or complex layout analysis [4, 13].

In contrast, web information extraction and web document understanding face additional challenges due to the dynamic, browser-rendered nature of web pages and the frequent mismatch between the DOM tree and the visual layout. Consequently, existing approaches often focus either on DOM or template-based extraction [1, 10, 12], or incorporate visual cues by learning from screenshots/visual contexts [6, 8], or from browser-derived render trees [9].

This short paper extends our earlier work: in [2], we introduced an RDF-based model of rendered web pages to enable visually aware web scraping (in contrast to the more common DOM-based scrapers); in [3], we presented a general method for creating snapshots of rendered web pages using this representation. In this study, we propose an entirely new approach that brings this solution into the Python-based machine learning ecosystem, enabling the use of frameworks such as PyTorch on web page data.

## 3 Visual-Aware Representation of Web Pages

To exploit the visual properties and layout of web page content in machine learning applications, the input web pages must be rendered, and the rendering results must be explicitly described. We implemented both tasks using the FitLayout framework [3].

FitLayout allows the representation of a single rendered page at diferent levels of abstraction and provides an ontology suitable for this purpose. The ontology<sup>1</sup> defines the concept of an Artifact that describes a single page on some level of abstraction. For machine learning tasks, the following two artifact types (subclasses of Artifact) are relevant:

– The Page artifact describes a rendered page at the lowest level of abstraction as a set of Boxes (thus also called a box tree).

– The AreaTree represents an abstraction over the rendered page on the level of visible visual areas that are not directly connected to the source HTML code.

In the BoxTree, every box is directly linked to a source HTML element, as specified by the CSS formatting model. Each box has visual attributes such as fontSize, color, positionX, and positionY. The root box corresponds to the entire page area, and hierarchical relationships are captured via the isChildOf property. The AreaTree eliminates the rigid reliance on HTML markup by concentrating on visual areas, which can be identified using diferent techniques (such as page segmentation methods) provided by FitLayout. Finally, each area can be associated with one or more Tags via the hasTag property, which can be used to label areas during dataset preparation.

As a result, each rendered page is represented by two artifacts (a Page and an AreaTree), which correspond to RDF subgraphs stored in the FitLayout RDF repository.

## 4 System Architecture and Processing Pipeline

The proposed system architecture is based on the observation that machine learning workloads are frequently spread across multiple computing nodes with distinct roles: storage nodes equipped with large-capacity storage for persisting data and compute nodes that provide suficient processing power to carry out neural network training and evaluation.

An overview of the entire architecture is shown in Fig. 1. It is composed of the FitLayout server, which acts as a data storage node with a central RDF repository for persisting page artifacts, and client-side Python applications that control dataset preparation and implement the machine learning methods.

To support the web page processing workflow, the server provides built-in services for creating diferent artifacts:

– Page rendering – it renders a web page in a headless Chromium web browser, which is controlled remotely via the Puppeteer library<sup>2</sup>. Then, the visual attributes of every rendered box are determined using JavaScript, and the Page RDF graph is created and stored in the repository.

![](images/35f5eee7d203063d427c2dc9f13368705497e98398ae01aa9243bb7d613256f8.jpg)  
Fig. 1. The complete architectural layout, featuring the FitLayout server as the central data storage node and Python-based compute nodes responsible for dataset preparation as well as GNN training and evaluation.

– Visual area extraction – for our task, we use a simple method that directly maps each visually distinct box in a Page to a visual area in the resulting AreaTree while discarding boxes that do not have any visual efect.

– Layout analysis – examines the relative positions of sibling areas in the area tree and adds RDF statements indicating that one area is above, below, to the leftOf, or to the rightOf another area. FitLayout ofers several implementations of this analysis; we employed an adapted version of the visibility method introduced by Gemelli et al. [5], originally designed for PDF documents.

Client applications can call these services through a REST API to generate and store RDF datasets of the rendered pages. Additionally, the server provides a SPARQL endpoint for querying the stored RDF data and inserting new statements. By default, FitLayout also includes an interactive client application with a web-based GUI<sup>3</sup>, which makes it possible to view and inspect the stored artifacts.

As Python has efectively become the standard environment for implementing machine learning methods, FitLayout ofers a Python client library<sup>4</sup> for its REST API, enabling seamless integration of rendered page data into machine learning workflows.

## 5 Python-Based ML Integration

The Python client layer, as shown in Fig. 1, covers two basic tasks: dataset preparation (which includes the preparation of the rendered page artifacts and their labeling) and the implementation of the machine learning applications themselves.

Artifact preparation involves sequentially calling the Page Rendering, Visual Area Extraction, and Layout Analysis services for each input page URL, with the goal of creating an AreaTree artifact for every source page. Each of these services is configurable; for instance, the client can set the viewport size used for rendering or use a specific method for spatial relationship discovery.

The optional data labeling step can be used to assign labels (Tags) to selected visual regions, which can later serve, for example, for training visual area classifiers. In a typical scenario, the target visual areas are first retrieved with a SELECT query, and subsequently, the hasTag statements are inserted into the RDF graph. The query can use both visual characteristics (e.g., color, size, or position) and properties of the underlying HTML, such as element attributes. For instance, in the Klarna web page dataset [7], specific element attributes are employed to indicate the target elements; these attributes can then be straightforwardly converted into labels using this approach.

Finally, to implement the machine learning algorithms, the data on the visual areas used for training or testing can be eficiently retrieved from the repository via SPARQL SELECT queries. These queries make it possible to specify the target visual areas in the same way as in the previous step and select the relevant property values. For instance, when applying GNNs for visual area classification, the source graphs can be eficiently generated using one query to obtain node properties and another to obtain edge properties, as we illustrate in detail in the following section.

## 6 Case Study: GNN-Based Content Element Recognition

To demonstrate the practical usefulness of the proposed architecture for machine learning on visual-aware web page models, we focus on a sample task: training a GNN-based classifier of visual areas to recognize book titles and prices in a fictional online bookstore<sup>5</sup>, a commonly used example target for web scraping. The entire project repository is available at GitHub<sup>6</sup>. The sample project covers the entire workflow from dataset preparation through the application of ML methods to exporting the dataset in a shareable format to support the reproducibility of the entire process.

## 6.1 Environment setup

The project contains client Python scripts that implement dataset preparation, data labeling, and machine learning, as described in the following subsections.

The FitLayout server is set up by simply running the corresponding Docker image provided by the FitLayout project. For our experiments, we used a twonode configuration, where the server runs on a separate computing node with suficient storage space. However, the client and server can share a single node when required.

## 6.2 Dataset preparation

The entire web page processing workflow starts by rendering the source pages based on a list of URLs. Collecting the input URLs is not covered here; we used a simple script to extract the URLs of pages corresponding to individual books in the bookstore. We collected 1,000 URLs in total and made them available in a text file within the project. The individual subtasks were implemented as short Python scripts that control the dataset preparation by simply calling the corresponding FitLayout services using the FitLayout Python client library:

Page rendering script (render.py) invokes the Puppeteer-based rendering service for every input URL using the default viewport width of 1200 px. As a result, the corresponding Page artifacts are stored in the built-in RDF repository.

The Postprocessing & layout script (segment.py) applies the Visual area extraction service on each page, generating an AreaTree artifact. It then calls the layout analysis service to infer the spatial relationships between the visual areas, as detailed in Section 4.

Finally, the data labeling script (tagging.py) detects the visual areas corresponding to book titles and their prices in the repository and inserts the respective hasTag statements into the RDF repository, which are subsequently used to train the GNN. In our basic use case, all source pages follow a shared template, enabling the identification of target regions through a single SPARQL query that combines visual features and source HTML element properties. For more heterogeneous input sources, multiple taggers would be required, or in the worst-case scenario, manual labeling would need to be carried out via the FitLayout GUI.

## 6.3 GNN Training

We implemented the GNN training using the widely adopted PyTorch Geometric (PyG) framework. For each AreaTree, we first build a corresponding PyG graph in which nodes represent visual areas, while edges capture both the parentchild relations in the tree and the spatial relations between sibling areas. Node features are retrieved via a SPARQL query and include text color, background color (RGB), X and Y pixel coordinates on the page, font size and weight, text length, and proportions of letters, digits, and punctuation in the contained text. The target class is encoded as a simple class index (1 for title, 2 for price, and 0 for all other areas). A separate SPARQL query is used to detect nodes that share either a parent-child or spatial relationship; these relationships are then encoded as edges in the PyG graph. In this manner, we create a PyTorch dataset that can serve as input for training the GNN.

As a representative GNN architecture, we employ the GCNConv operator from PyG and build a neural network with three convolutional layers to generate graph embeddings. The network consists of an input layer, a hidden layer with 128 channels, and an output layer with 10 channels, followed by a linear classifier. To train the network, we employed a straightforward training loop using the AdamW optimizer, cross-entropy as the loss function, and early stopping.

Our sample dataset, as outlined above, consists of 1,000 area trees containing a total of 37,768 visual areas. Although the primary aim of this sample project is to demonstrate how the proposed architecture can integrate rendered web pages into a Python-based ML workflow—and the underlying model is comparatively simple—the experiments resulted in only 0 to 4 misclassified visual areas across the entire set (depending on how the training and testing data were split). These results indicate that the proposed approach is indeed viable.

## 6.4 Dataset sharing

The FitLayout Python client also includes the capability to export and import the complete RDF repository in common RDF serialization formats, such as Turtle or N-Quads. The exported dataset captures all information about the rendered web pages and the generated artifacts (area trees), including their labels and even a screenshot of each page. This makes it possible to publish the dataset and/or reproduce the results even if the original web pages are altered or become unavailable. The dataset created as part of this case study is available at Zenodo<sup>7</sup>.

## 7 Conclusions

The implemented platform demonstrates a comprehensive approach to integrating visual-aware representations of rendered web pages into machine learning workflows. By leveraging the FitLayout framework and RDF-based modeling, the system explicitly captures the visual and structural properties of web pages, enabling flexible data extraction and reproducible experiments. The provided Python client library facilitates seamless dataset preparation, labeling, and application of machine learning methods, such as graph neural networks for content element recognition, as illustrated in the case study. This architecture not only bridges the gap between web content and document understanding methods but also supports dataset sharing and reproducibility, addressing the key challenges in web page analysis using machine learning.

Acknowledgements This work was supported by the project Smart information technology for a resilient society, FIT-S-23-8209, funded by Brno University of Technology.

## References

1. Bevendorf, J., Gupta, R., Kiesel, J., Stein, B.: An empirical comparison of web content extraction algorithms. In: Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 2594–2603 (2023)

2. Burget, R.: Scraping data from web pages using sparql queries. In: Garrigós, I., Murillo Rodríguez, J.M., Wimmer, M. (eds.) Web Engineering. pp. 293–300. Springer Nature Switzerland, Cham (2023)

3. Burget, R., Salem, H.: Creating searchable web page snapshots using semantic technologies. In: Garrigós, I., Murillo Rodríguez, J.M., Wimmer, M. (eds.) Web Engineering. pp. 355–358. Springer Nature Switzerland, Cham (2023)

4. Gemelli, A., Biswas, S., Civitelli, E., Lladós, J., Marinai, S.: Doc2graph: A task agnostic document understanding framework based on graph neural networks. In: Karlinsky, L., Michaeli, T., Nishino, K. (eds.) Computer Vision – ECCV 2022 Workshops. pp. 329–344. Springer Nature Switzerland, Cham (2023)

5. Gemelli, A., Vivoli, E., Marinai, S.: Graph neural networks and representation embedding for table extraction in pdf documents. In: 2022 26th International Conference on Pattern Recognition (ICPR). pp. 1719–1726 (2022)

6. Gogar, T., Hubacek, O., Sedivy, J.: Deep Neural Networks for Web Page Information Extraction. In: 12th IFIP International Conference on Artificial Intelligence Applications and Innovations (AIAI). vol. AICT-475, pp. 154–163. Thessaloniki, Greece (Sep 2016). https://doi.org/10.1007/978-3-319-44944-9\_14

7. Hotti, A., Risuleo, R.S., Magureanu, S., Moradi, A., Lagergren, J.: The klarna product page dataset: Web element nomination with graph neural networks and large language models. Transactions on Machine Learning Research 2024 (2024)

8. Kumar, A., Morabia, K., Wang, W., Chang, K., Schwing, A.: CoVA: Contextaware visual attention for webpage information extraction. In: Malmasi, S., Rokhlenko, O., Uefing, N., Guy, I., Agichtein, E., Kallumadi, S. (eds.) Proceedings of the Fifth Workshop on e-Commerce and NLP (ECNLP 5). pp. 80–90. Association for Computational Linguistics, Dublin, Ireland (May 2022). https://doi.org/10.18653/v1/2022.ecnlp-1.11

9. Li, Z., Shao, B., Shou, L., Gong, M., Li, G., Jiang, D.: Wiert: Web information extraction via render tree. Proceedings of the AAAI Conference on Artificial Intelligence 37(11), 13166–13173 (Jun 2023). https://doi.org/10.1609/aaai.v37i11.26546

10. Lin, B.Y., Sheng, Y., Vo, N., Tata, S.: Freedom: A transferable neural architecture for structured information extraction on web documents. In: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. p. 1092–1102. KDD ’20, Association for Computing Machinery, New York, NY, USA (2020). https://doi.org/10.1145/3394486.3403153

11. Riba, P., Dutta, A., Goldmann, L., Fornés, A., Ramos, O., Lladós, J.: Table detection in invoice documents by graph neural networks. In: 2019 International Conference on Document Analysis and Recognition (ICDAR). pp. 122–127 (2019). https://doi.org/10.1109/ICDAR.2019.00028

12. Truong, B.V., Pham, P., Nguyen, L.T., Nguyen, N.T., Vo, B.: Web data analysis using a hybrid approach of DOM processing and deep learning models. Applied Soft Computing 191, 114651 (2026)

13. Wang, J., Krumdick, M., Tong, B., Halim, H., Sokolov, M., Barda, V., Vendryes, D., Tanner, C.: A graphical approach to document layout analysis. In: Fink, G.A., Jain, R., Kise, K., Zanibbi, R. (eds.) Document Analysis and Recognition - ICDAR 2023. pp. 53–69. Springer Nature Switzerland, Cham (2023)