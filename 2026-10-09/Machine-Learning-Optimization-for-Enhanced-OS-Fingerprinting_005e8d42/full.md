# Machine Learning Optimization for Enhanced OS Fingerprinting

Spencer Ekeroth

Virginia Tech

Jeremy Neale

Virginia Tech

Jae Sung Kim

Virginia Tech

## Abstract

Operating System (OS) Fingerprinting is a technique that can be used to identify a network’s operating systems by evaluating network traffic in the form of TCP/IP packets. This research will explore the effectiveness of passively identifying operating systems on the CIC-IDS2017 dataset, a collection of over 47 gigabytes of pcap files with their corresponding operating systems. This research also proposes a new command line interface, OsirisML, which uses nPrint[6] to preprocess the data into tabular data and XGBoost[8] to apply ML to the data to generate, retrain, and test ML models. When packets are split randomly between training and testing, OsirisML models reach an accuracy of 97.66% on a down-sampled subset of the Friday capture and 84.69% on the entire capture. On the entire Monday capture, which contains no attacks, OsirisML reaches an accuracy of 73.83% and an F-1 score of 79.38%.

## 1 Introduction

Passive fingerprinting analyzes network traffic offline, which avoids detection from the network. There are many versions of passive fingerprinting tools that can analyze packet capture (pcap) data and attempt to identify the operating system. To generate a new or custom passive fingerprinting model using machine learning, pcap data can be analyzed and used to generate a new model with a tool like nPrintML, which uses nPrint to transform the PCAP data into tabular data, and AutoML [4] automatically generate a ML model from the tabular data.

Given the dynamic nature of network environments, where new OS versions are continuously released, there is a compelling need for tools that can adapt and retrain machine learning models for passive OS fingerprinting to remain effective against the ever-evolving landscape of operating systems.

This research introduces a novel command-line tool, OsirisML, which consists of two main components similar to nPrintML. First, it leverages nPrint to transform network traffic (pcap files) into a normalized tabular format suitable for machine learning. Unlike nPrints alternative, nPrintML, that utilizes AutoML for automated model generation, OsirisML integrates the advanced capabilities of XGBoost—an algorithm renowned for its efficiency with tabular data—to enhance model accuracy and adaptability. The usage of XGBoost [8] for machine learning applications contains 4 components: generating models, testing unseen data with a model (passive fingerprinting), retraining a model on an unseen dataset to generate a new model, and hyperparameter tuning (section 9).

This tool allows for 3 significant advantages over its counterparts:

• Hands-on manipulation of the model creation with custom hyperparameter tuning, file partitioning, and retraining models

• Improved accuracy and F-1 scores from model generation on identical data

• More efficient RAM usage on identical data

## 2 Motivation and problem statement

As aspiring cybersecurity professionals, it is understood that the importance of identifying an operating system over a network can be of critical importance. Creating an accurate tool to do so would be beneficial for independent work, as well as to the global community of white hat threat actors. It could be used to impersonate a black-hat threat actor to see if they could enumerate a network and exploit any vulnerabilities.

Open source work is what has propelled the computer science field forward for the past century. Projects such as Linux, GNU, Python, and Docker have benefited millions of people across the world. Improving upon pre-existing open source tools would be one way to advance the fields of cybersecurity and machine learning and help other computer scientists in the future.

The goal of OsirisML is not just to make an all-in-one passive fingerprinting model, but to provide a method to create and build upon pre-existing models. This allows OS Fingerprinting to stay up to date and continuously improve from open-source contributions of datasets and model generation.

## 3 Related work

In the research paper, ’New Directions in Automated Traffic Analysis’ [2], Jordan Holland, Paul Schmitt, Nick Feamster, and Prateek Mittal created a tool called nPrint and nPrintML [6]. nPrint is capable of capturing traffic live or processing a pcap file to convert raw packet data into a tabular format suitable for machine learning algorithms. nPrintML is an all inclusive tool that takes the tabular output of nPrint and uses AutoML to select a machine learning algorithm to create an optimized model for OS fingerprinting. As more machine learning models are created and better optimized to handle tabular data, this project will focus on optimizing the nPrint data output and using modern machine learning algorithms to attain a model that is more accurate in passive OS fingerprinting than what was achieved by Jordan Holland et al. [2]

## 4 Production Environment for OsirisML

The production environment for OsirisML is set up on a virtual machine running Ubuntu 22.04 [7], a Debian-based Linux distribution, to ensure stability and consistency. The virtual machine is running on an AMD EPYC 7401 24-Core Processor with 128GB of RAM. Virginia Tech’s rlogin cluster, which is a multi-machine system running Linux to provide computational support for Computer Science instruction, was also utilized to run hyperparameter\_tuning.py. The rlogin cluster has many nodes, each equipped with 2 Xeon Gold 5218 CPUs with 2.3Ghz 16 core processors, 384 GB of RAM, and a 10 Gbit Ethernet connection [5]. The results in section 10 were reproduced on a workstation with 32 threads and 123 GB of RAM using XGBoost 3.2.0.

## 5 CIC-IDS 2017 PCAP Dataset

The CIC-IDS 2017 dataset [3], released by the University of New Brunswick, is a comprehensive collection of network traffic designed to represent a mix of the typical network traffic in a real-world environment, along with intrusion attempts from Kali Linux. This dataset is chosen because it contains a large set of labeled operating systems by IP address. The networks are represented by an array of operating systems, including different versions of Ubuntu servers, Windows ma chines from Vista to Windows 10, and macOS systems, among others. Each operating system is labeled by the source IP address of a single host, or two hosts for the Ubuntu server and the Ubuntu web server, so in this dataset an operating system and a host are the same. The Friday capture contains 13 labeled operating systems, including the Kali Linux attacker, while the Monday capture, which contains no attacks, has 12. Operating system fingerprinting often relies on discerning the subtleties in packet structure, sequence numbers, fragments, window sizes, and other TCP/IP stack behaviors that vary from one OS to another.

## 6 nPrint and nPrintML

The starting point for developing a model that is capable of differentiating fine-grain operating systems was to replicate Section 5.2 in ’New Directions in Automated Traffic Analysis’ [2]. By using the same features, it would provide a benchmark for the new model. nPrint takes a unique approach to packet representation by assuming all packets have a header of maximum length. Any missing options or fields in the headers are represented with a value of -1. To keep consistency with Section 5.2 only IPv4 and TCP headers will be extracted from each packet. This results in each encoded packet sample to have 960 features, representing the 480 bits from the IPv4 header and the additional 480 bits from the TCP header.

The output generated from nPrint with the IPv4 and TCP header options set still requires cleaning the IP source address, IP destination address, IP identification, TCP source port, TCP destination port, and both TCP sequence and ack number, which can act as direct identifiers to the operating system. OsirisML drops these 177 bit columns along with the source IP address column, which leaves each sample with 783 features. Without this cleaning, an XGBoost model trained on the Monday capture reached an accuracy and F-1 score of 100%, and its most important features were source IP, destination port, and sequence number bits. With the cleaning, the same model reached an accuracy of 63.36%. These changes can be made to the source code of nPrintML. After using pip to install the nPrintML package, it is imperative to edit automl.py which can be found in site-packages/nprintml/learn/ where site-packages is the directory pip installs new libraries and packages. Before the train method is called, the areas of the IPv4 and TCP header mentioned above need to be dropped as they act as direct identifiers. If the areas are not dropped, nPrintML will return a model that is extremely overfitted to the dataset due to data leakage.

Removing these fields does not remove every host-specific value. Some values are constant for a given machine. The TTL, window size, and TCP options are legitimate features of an operating system, but since each operating system in the dataset is a single host, they can also identify the host itself (section 12).

![](images/5382bbd2e25682c79444d67c5e9a730d667ebefdba93eaa053dfb12115258ee1.jpg)  
Figure 1: Visualization for the flow of both phases of the CLI

## 7 Phase 1 - Data Preprocessing Workflow

OsirisML is split into two main components: preprocessing, and ML application. The goal of the preprocessing is to take pcap data and convert it to tabular (csv) data, making it compatible for machine learning algorithms. This entire process can be executed with make preprocess PCAP=<PCAP\_file\_name.pcap>, which runs process\_pcap.sh and calls the following three scrips found in preprocessing/ in succession:

1. tcp\_dump.sh - uses the source IP labels provided on CIC-IDS2017 host website [3] by default to split up traffic based on operating systems. Alter the source code of this file with new source IPs if attempting to generate a model from a dataset with different source IPs.

Note that the PCAP file that this script accepts as a system argument must be located in data/pcap, and the resulting PCAP files will be placed into data/pcap/pcap\_os\_split/. The name of each file will be the name of the operating system provided in the source code.

Usage: ./tcp\_dump.sh <PCAP\_file\_name.pcap>

2. nprint.sh - unlike nPrintML, nPrint is unable to take a directory of pcaps as a valid argument. This script loops through all PCAP files in data/pcap/pcap\_os\_split/ and calls nPrint on each file with -4 -t parameters - see nPrint documentation.

The resulting NPT files will be placed into data/npt/ and will have the same names with new extensions.

Usage: ./nprint.sh

3. npt\_to\_csv.py - This file loops through each NPT file in data/npt/ and writes each line into a single CSV file, which it places in data/csv/.

Usage: python3 <npt\_to\_csv.py> <new.csv>

Execute this entire workflow with: make preprocess PCAP=<PCAP\_file\_name.pcap>

Note that process\_pcap.sh also deletes the data/npt/ and data/pcap/pcap\_os\_split/ directories after finishing.

## 8 Phase 2 - ML Application with XGBoost

Once the tabular data is preprocessed, it has the relevant TCP/IP binary input attributes and one classification label, which is the corresponding operating system. With the labeled CSV file, different ML algorithms can easily be applied to the data to produce a model. XGBoost was selected for its effectiveness with tabular data, along with its open-source nature and extensive documentation.

The XGBoost scripts splits the CSV file into x\_train, x\_test, y\_train, and y\_test. The x files are the input attributes, which in this case are the TCP/IP binary input attributes, and the y files are the corresponding classification labels, which is the last column of the CSV. The labeled csv dataset has 962 columns (or attributes), but the first column is the source IP address in semantic form, and the last column is the classification label as a string.

The x\_train and x\_test values were then further cleaned to remove the bit ranges of values that would act as direct identifiers to the model. Both y\_train and y\_test were encoded from strings to integers to allow them to be processed by XGBoost’s classifier.

The default eval metric supplied to all three model scripts is mlogloss, which is selected due to its nature of multi-class classification, and its robustness to imbalanced classes.

There are 4 main files in model/:

1. hyperparameter\_tuning.py - see section 9.

Usage: python3 hyperparameter\_tuning.py <data.csv>

2. xgboostmodel.py - generates a model from the supplied csv. This uses a test size of 0.2 by default, but can be updated with an optional sys argument.

The csv file must be located in data/csv/, where it should be by default after data preprocessing. The outputted json file will be placed into model/json/ by default.

Usage: python3 xgboostmodel.py <data.csv> <modelname> <OPTIONAL: <decimal for test split size>

3. trainmodel.py - XGBoost allows for already generated models to be retrained on new data to produce a new model. This is useful for optimized RAM and storage, as well as improving the robustness of a model. This script uses a DMatrix, where the x and y train and assigned to dtrain, and x and y test are assigned to dval. This matrix approach during training drastically improves space and speed.

When invoking xgboost.train, the loaded model is used as the starting point, the max number of additional boost rounds is set to 1000, and early stopping rounds is set to 10, using 25% of the training data for validation. The model is then saved at its best round. The new csv must contain the same operating systems as the original model, since the labels are encoded in the same order. Currently, max\_depth = 10 is the only hyperparameter supplied to this script.

The model must already be located in model/json/, which is should be after invoking xgboostmodel.py or this script. The training data must be located in data/csv/. The new model will be placed into model/json/.

Usage: python3 trainmodel.py <current\_model.json> <data.csv> <new.json>

4. testmodel.py - this script can be thought of as passive OS Fingerprinting. It takes a model generated by xgboostmodel.py or trainmodel.py, and tests unseen data to evaluate its performance. This script currently only returns the accuracy and F-1 score to evaluate the success of the model, but does not provide information on each individual machine in the packet capture data. To determine predictions for individual machine’s operating system, supply only 1 source IP address in the csv file to the model. The testing csv must contain the same operating systems as the training csv, since the labels are encoded from the testing file.

The model must already be located in model/json/, which is should be after scripts 2 and 3. The testing data must be located in data/csv/.

Usage: python3 testmodel.py <model.json> <data.csv>

## 9 Hyperparameter tuning and optimization

In machine learning, parameters are learned and updated throughout the training process, while hyperparameters are set beforehand and determine key features of the training method such as model architecture, learning rate, and model complexity. XGBoost uses a gradient boosted decision tree machine learning algorithm that accepts many optional hyperparameters. Using optimal hyperparameters for the machine learning training process is crucial to achieving higher model accuracy and F-1 scores while avoiding overfitting. Hyperparameters also have a significant role in the time and computer resources necessary to train a model.

OsirisML comes equipped with a script model/hyperparameter\_tuning.py that searches for the optimal parameters for the supplied CSV file, which is the PCAP data that has been preprocessed into tabular data. See section 8 for usage.

To determine the optimal parameters, the script uses GridSearchCV [1], which generates an XGBoost model for every combination of hyperparameters supplied as a parameter of a Python Dictionary, where the keys are the hyperparameter options as strings, and the values are the options for that hyperparameter. For example,

# Create the parameter grid   
param\_grid =   
{   
’max\_depth’: [3, 6, None],   
’gamma’: [0.1, 0.5, 1.0],   
’min\_child\_weight’: [2, None]   
}

Using GridSeachCV with this grid would generate 18 models, since there are 3 ∗ 3 ∗ 2 combinations of the three parameters. After generating the 18 models, GridSeachCV returns the best combination of parameters based on the supplied evaluation metric, which is supplied as accuracy in the script, using 5-fold cross validation on the training data.

It is recommended to use a smaller partition of the data to search for hyperparameters, then to use these psuedo-optimal hyperparameters on the entire dataset. This is because searching for hyperparameters is computationally expensive and costs lots of time and computer resources. The pcap files can be partitioned with python3 preprocessing/shrink\_pcap.py <path\_to\_pcap.pcap> --fraction X (default 10).

After retrieving the optimal combination of supplied hyperparameters, they can be supplied to model/xgboostmodel.py and used to train the entire data set.

Using the Friday-WorkingHours.pcap file from the section 5 dataset, a partition of the dataset was created that split the entire 8.2 gb dataset 32 times. Windows 10 and the Ubuntu web server were both split 50 times, since they were drastically over represented compared to the other OSes. The resulting dataset had 134,269 samples and the csv file was now about 336 mb. The OSes were distributed by the following chart:

![](images/298205906b5671ada1435a5458002798a19dbae0956940ad8053f695694b7aa2.jpg)

With this smaller dataset, the hyperparameter tuning script reported that max\_depth = 10 and the others as None was optimal, yielding the highest accuracy and F-1 scores of 97.70% and 97.51% respectively. There are many more hyperparameters to test, but due to the nature of the time it takes to generate models, only a select pool of hyperparameters were given to GridSearchCV:

# 54 combinations - 3 \* 3 \* 2 \* 3   
param\_grid =   
{   
’n\_estimators’: [400, 800, None],   
’max\_depth’: [6, 10, None],   
’min\_child\_weight’: [2, None]   
’gamma’: [0.1, 0.5, 1.0]

Further research could test more hyperparameters and/or use a larger dataset to return a more accurate assessment of the optimal hyperparameters to generate models with.

When training on the entire Friday file, the default learning rate of 0.3 occasionally caused XGBoost to diverge. In one run the validation mlogloss fell to 0.54 after 30 rounds and rose to 9.66 by round 50, and the final model was much less accurate. This does not happen on every run, but to prevent it, all results in section 10 use early stopping with 10 rounds of patience on 10% of the training data.

## 10 Results

All results in this section use the 783 features left after the cleaning in section 6, max\_depth = 10 unless noted, and early stopping as described in section 9. All F-1 scores are macro averages.

The original results were first reproduced using a random 80% training and 20% testing split of the packets (Table 1). Using the condensed version of Friday-WorkingHours.pcap from section 9 produced an accuracy of 97.66% and an F-1 score of 97.35%, while nPrintML reported 96% accuracy and 96% F-1 score on a random 70% training and 30% testing split of the same packets (nPrintML only reports results with two digits). Using XGBoost’s default parameters on the entire Friday file resulted in 78.40% accuracy and 73.67% F-1 score. With max\_depth = 10, the accuracy and F-1 score improved to 84.69% and 82.61% respectively. When using nPrintML on the same dataset, the process killed due to RAM usage. The entire Friday file is an 8.2 gb pcap file that was preprocessed into an 11.2 gb csv file with 4,501,929 samples.

The final test was to replicate Section 5.2 of Jordan Holland et al. [2] with the first 100,000 packets of each operating system in Friday-WorkingHours.pcap. The Ubuntu server and Kali Linux have fewer than 100,000 packets, so the dataset is comprised of 1,119,205 packets. Using this dataset saw an accuracy of 89.83% and an F-1 score of 91.15%.

<table><tr><td>Dataset</td><td>Reported</td><td>Reproduced</td></tr><tr><td>Friday subset</td><td>97.70 /97.51</td><td>97.66 / 97.35</td></tr><tr><td>Friday, default</td><td>74.09 / 64.83</td><td>78.40 / 73.67</td></tr><tr><td>Friday, max_depth = 10</td><td>84.91 / 82.96</td><td>84.69 / 82.61</td></tr><tr><td>Section 5.2</td><td>90.11 / 91.39</td><td>89.83 / 91.15</td></tr><tr><td>Monday</td><td></td><td>73.83 / 79.38</td></tr></table>

Table 1: Reported and reproduced accuracy / F-1 score (%), random split

On the entire Friday file with max\_depth = 10, every operating system has an F-1 score between 87.83% and 100% except Windows 10 Pro (51.53%), Windows 7 Pro (45.07%), and Windows 8.1 (52.01%), which are mostly confused with each other and with Windows 10.

To test OsirisML on a capture with no attack traffic, the same random split was used on the entire Monday-WorkingHours.pcap file, which contains 5,387,586 packets from 12 operating systems. This produced an accuracy of 73.83% and an F-1 score of 79.38%. macOS, the Ubuntu web server, and Windows Vista have the highest F-1 scores (98.56%, 95.02%, and 87.81%), while Windows 7 Pro (48.29%) and Windows 10 Pro (57.14%) have the lowest.

nPrintML also evaluates its models on a random split of packets [6], so its reported results are directly comparable with the random split results of OsirisML. The nPrintML results were not re-run for this revision. nPrintML requires Python 3.8 and its own AutoML environment, and on the entire Friday file it was killed due to RAM usage.

## 11 Proposed work

Looking ahead with OsirisML, the plan is to train the model on the rest of the pcap files within the CIC-IDS 2017 dataset and to test it on a day that was not used for training. Given enough memory, nPrintML should also be run on the entire Friday file so that the two tools can be compared on the full dataset.

Another direction is a hierarchical model, which would first predict the operating system family and then predict the version within that family with a separate model. Feature importance on the Monday capture shows that the TTL separates the operating system families, while the TCP options separate the Ubuntu versions and the window size separates the Windows versions. Windows 10 and Windows 10 Pro use the same network stack, so they could be combined into one label.

OsirisML is an open-source project, and users will be able to contribute to the model by providing new pcap data, if approved by the OsirisML team. A new dataset could be produced by isolating a network and populating it with several machines for each operating system, which would allow complete control over the collection of data and separate the operating system from the host.

## 12 Potential Limitations

While OsirisML reproduces its original results, there are some factors that limit how they should be interpreted. Each operating system in CIC-IDS2017 is a single host, or two hosts for the servers, so the model may learn the behavior of a host rather than of an operating system. The dataset is also imbalanced: Windows 10 has 1,595,579 packets on Friday while Kali Linux has 8,991, so the F-1 score should be reported along with the accuracy. OsirisML has only been tested on the CIC-IDS2017 dataset which only consists of 13 operating systems. In order to be confident in the success of this CLI, its success would need to be replicated with other datasets.

Current compatibility limitations are that OsirisML is currently only available for installation on Debian-based Linux. This is because it is recommended to use this tool in a VM, which is likely GNU-Linux based. Containerizing could make the tool more accessible and consistent across host machines.

## References

[1] scikit-learn developers. GridSearchCV – scikit-learn 1.1.3 documentation. https://scikit-learn.org/ stable / modules / generated / sklearn . model \_ selection.GridSearchCV.html. Accessed: 2024-05- 01. 2023.

[2] Jordan Holland et al. “New Directions in Automated Traffic Analysis”. In: Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security. 2021, pp. 3366–3383. DOI: https : / / doi . org/10.1145/3460120.3484758.

[3] Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani. “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”. In: Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP). Canadian Institute for Cybersecurity, University of New Brunswick. Portugal, Jan. 2018.

[4] AutoML Team. AutoML: Automated Machine Learning. https://www.automl.org/automl/. Accessed: 2024-05-03. 2024.

[5] Virginia Tech. Hardware Information for Cluster. https://wordpress.cs.vt.edu/rlogin/moreinformation-3/. Accessed: 2024-04-02. 2024.

[6] The nPrint Project. nPrint. https://nprint.github. io/nprint/. Accessed: 2024-02-27. 2024.

[7] Ubuntu 22.04.4 LTS (Jammy Jellyfish). https : / / releases.ubuntu.com/jammy/. Accessed: 2024-04- 12. 2024.

[8] XGBoost Development Team. XGBoost: Scalable Machine Learning System. https : / / xgboost . readthedocs.io/en/stable/. Accessed: 2024-05- 03. 2024.