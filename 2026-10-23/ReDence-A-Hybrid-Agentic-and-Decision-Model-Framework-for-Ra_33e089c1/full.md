# ReDence: A Hybrid Agentic and Decision-Model Framework for Rapid, Evidence-Grounded Detection of Modern Social Engineering Scams

Jaishnoor Kaur, Josef Hora´cek, Ian Waskom, Sayak Saha Royˇ

Louisiana State University

{Jaishnoor.Kaur, jhoracek, Ian.Waskom}@lsu.edu, sayak.saharoy@lsu.edu

Abstract—Detecting online social engineering (SE) scams often requires examining website workflows, verifying claims against external sources, and connecting evidence across multiple pages. Existing detectors provide limited support for these investigations, leaving analysts to manually gather and validate evidence, which inevitably delays detection and subsequent takedown of these attacks.

To address this gap, we present ReDence, an open-source hybrid framework for adaptive, evidence-grounded detection across diverse SE scam categories. ReDence combines two locally hosted models with complementary responsibilities: a fine-tuned multimodal agentic Investigator (Qwen3.5-4B) that performs investigative planning, evidence interpretation, and reporting, and a specialized Decider (Tev1) that handles binary and multiple-choice tasks, including browser-action selection, evidence assessment, and decisions about whether to continue an investigation. By delegating discrete decisions to a specialized model, ReDence reduces reliance on lengthy languagemodel responses, while preserving the reasoning capabilities required for complex investigations.

Our evaluation demonstrates that ReDence achieves 96% accuracy in detecting scams across twelve SE scam categories, substantially outperforming existing detectors while being at least four times faster than comparable agentic models without compromising investigative depth.

ReDence also produces structured, evidence-grounded reports that document investigative actions, supporting and contradictory evidence, and collected artifacts, which allow analysts to independently audit detection decisions and utilize the collected evidence to support subsequent mitigation and takedown efforts.

## 1. Introduction

Social engineering (SE) scams remain a major source of internet-enabled financial harm. Although prior research has extensively characterized different SE scam categories [1], [2], [3], [4], automated website-level defenses have largely focused on credential-phishing attacks [5], [6], [7], which often exhibit recognizable indicators of impersonation or credential harvesting on their landing pages [7], [8], [9],

Yet credential phishing represents only one component of a broader SE scam landscape. The FBI’s 2025 Internet Crime Report documents \$215.8 million in losses attributed to phishing, compared with \$503.4 million from non-payment/non-delivery scams, \$8.6 Billion from investment fraud, and \$275.1 million from real-estate fraud [10].

However, detecting these SE scams often requires navigating multiple pages, interacting with website workflows, examining supporting artifacts, and independently verifying claims against external sources. For example, usastudentdebtrelief.com portrayed itself as a legitimate student-loan relief service while allegedly collecting more than \$7.3 million in unlawful advance fees and payments for services never rendered [11]. According to the FTC, the operation falsely claimed affiliation with the U.S. Department of Education, advertised repayment terms inconsistent with federal programs, and fabricated customer testimonials on both its own website and third-party platforms [12]. Importantly, evidence contradicting these claims could have been uncovered by cross-referencing information from multiple independent sources, even during the attack’s zero-day period. Yet, these warning signs were overlooked, illustrating how scams can evade detection when their deceptive claims are not systematically investigated and independently verified.

Recent systems have begun extending automated detection beyond credential phishing, including approaches targeting particular scam categories [13], [14] and systems for discovering candidate scam websites [15]. However, detectors that primarily return classification labels or confidence scores provide limited insight into the evidence underlying their assessments [16]. Such outputs may be insufficient for validating suspected threats or supporting downstream interventions [17], [18]. Consequently, analysts must often manually reconstruct the attack workflow and any necessary evidence before initiating reporting or takedown procedures. This process is time-consuming, difficult to scale, and dependent on familiarity with particular scam schemes or regional contexts [19], inevitably allowing malicious websites to remain active while, ideally, the investigations are completed, or the attack is misdetected to be benign.

The advent of agentic large language models (LLMs) offers a promising means of automating such investigations in depth [20], but introduces two challenges. First, the ability to reason and invoke tools does not inherently ensure effective end-to-end security investigation [21]. General-purpose models may require domain-specific investigative guidance to apply relevant knowledge, recognize deceptive practices, pursue appropriate verification steps, and distinguish suspicious observations from substantiated findings [20], [22].

Second, comprehensive agentic investigations are computationally expensive, as they involve repeated decisions about tool selection, evidence interpretation, and investigative continuation, often requiring multiple generative LLM invocations even for simple binary or multiple-choice tasks [23], [24]. The resulting latency, computational overhead, and API costs significantly limit their practicality for largescale scam detection [24], [25].

We address these challenges by introducing ReDence, a highly scalable hybrid framework designed to adaptively investigate and detect SE scams. ReDence integrates a fine-tuned multimodal agentic model (based on Qwen3.5- 4B [26]), responsible for investigative planning, reasoning, and evidence interpretation, with a specialized Decision model (based on Tev1 [27]) that rapidly (and efficiently) handles binary and multiple-choice decisions throughout the investigation, which significantly reduces unncessary generative overheads. These models are also equipped with specialized investigative prompts and tools that enable systematic, purpose-driven exploration of websites and their associated internal and external artifacts, while dynamically adapting investigative strategies based on emerging evidence and findings.

ReDence also maintains an evidence-grounded record of its investigation through reports that document the actions performed, artifacts collected, supporting and contradictory evidence, external verification results, and the reasoning connecting these observations to its final assessment. This design allows a human analyst to inspect not only what ReDence concluded, but also how it reached that conclusion and the evidence on which it relied. In doing so, our framework aims to combine the scalability of automated detection with the adaptive investigation and evidentiary record traditionally provided through human analysis.

We evaluate ReDence on SE scams spanning twelve categories, achieving 96% overall detection accuracy, outperforming the evaluated automated detectors while completing investigations at least four times faster than comparable agentic configurations. We further assess the quality of ReDence’s investigative reports through human evaluation.

The primary contributions of this work are as follows:

1) We introduce ReDence, an open-source hybrid framework that integrates a fine-tuned multimodal agentic model with a specialized Decision model, supported by domain-specific investigative prompts and tools. This design enables adaptive, evidence-grounded investigations of SE scams while substantially reducing computational overhead.

2) We develop a structured reporting mechanism that allows ReDence to document each investigation through detailed reports and an evidence ledger, linking investigative findings and assessments to observations, external sources, collected artifacts, and supporting or contradictory evidence. We further assess the relevance and auditability of these reports through human evaluation, finding them to not only enable analysts to identify new scams, but also identify more evidence points in scams they had previously detected.

3) We comprehensively evaluate ReDence against existing automated scam detectors across twelve SE scam categories. ReDence achieves 96% detection accuracy, substantially outperforming existing approaches while delivering at least a fourfold speedup over comparable agentic configurations without compromising investigative depth.

## 2. Background

Automated scam detectors: Despite the prevalence and diversity of SE scams, relatively few automated detection systems have been developed to address them. Existing approaches also vary considerably in their coverage of scam categories and ability to conduct in-depth investigations. For example, BeyondPhish [1] detects fraudulent websites using predefined features extracted from website content, URLs, domain information, and associated social-media accounts, but provides classification decisions without supporting investigative reasoning. ScamMagnifier [13] enables deeper investigations by navigating checkout workflows and examining merchant and payment infrastructure, although its capabilities are specifically tailored to fraudulent shopping websites. Meanwhile, LOKI [15] focuses primarily on discovering scam websites through learned search queries rather than investigating individual websites.

ScamFerret [20] is the closest prior work to ReDence. It leverages commercial LLMs to detect SE scams by analyzing textual website artifacts and invoking a predefined set of investigative tools. However, ScamFerret cannot directly interact with website workflows or associated artifacts, and its investigative approach does not consistently adapt to category-specific indicators of deception. In Section 7, we compare ReDence against ScamFerret and demonstrate how these limitations hinder its ability to systematically uncover and verify evidence necessary for detecting diverse SE scams.

Decision models: General-purpose LLMs often generate lengthy reasoning traces and explanations, even for tasks that require only a binary answer or a selection among predefined alternatives. When repeated throughout an agentic workflow, this unnecessary token generation can introduce substantial latency and computational overhead [28].

To address this inefficiency, recent work has introduced specialized Decision models, designed to efficiently handle tasks with a bounded set of possible outcomes [29], [30]. Given contextual information, a question, and a set of candidate answers, these models directly score the available alternatives and return a compact decision rather than generating an extended textual response. This makes them particularly suitable for recurring tasks such as action selection, request routing, and content assessment. Recent evaluations highlight their potential for low-latency decision-making. Deußer et al. [31] report a mean client-side latency of 0.36 seconds per request, including network overhead, when evaluating

Jev across 37 datasets with 32 concurrent requests. Similarly, Ollama reports that Nimble-9B achieved an average latency of 91 milliseconds per decision in a game-playing demonstration conducted on an M5 Max [30].

ReDence applies this specialization through its hybrid architecture, in which the Investigator delegates tasks requiring discrete decisions or selections among predefined alternatives to the specialized Decider while retaining responsibility for developing investigative plans, interpreting relationships among findings, and generating evidence-grounded reports. This division combines rapid decision-making with the sustained reasoning and adaptability required for comprehensively evaluating SE scams.

## 3. Research overview

ReDence combines two compact models, each with 4 billion parameters: a Qwen3.5-based Investigator for openended reasoning and reporting, and a Tev1-based Decider for binary and multiple-choice decisions. Although this architecture offers substantial advantages in computational efficiency and inference speed, prior research has documented limitations in smaller models’ ability to follow complex instructions [32], [33], coordinate multi-step planning [34], invoke tools effectively [35], [36], and produce structured outputs [37]. These capabilities are essential for conducting comprehensive website investigations and generating evidence-grounded reports.

However, prior work has also shown that distilling reasoning demonstrations and task trajectories from larger models can improve smaller models’ task-specific reasoning and tool-use capabilities [38], [39]. Motivated by these findings, we utilize a larger Qwen3.5-27B as a teacher or Mentor model, and equip it with dedicated investigative prompts and tools to generate detailed demonstrations of SE scam websites’ investigation and evidence-grounded reporting. We then fine-tune the Investigator and Decider models using Mentor’s investigative trajectories and structured reports, aiming to transfer these specialized capabilities while preserving the computational efficiency of the smaller models. Figure 1 illustrates the framework of our study. Our study proceeds as follows:

Section 4 describes the collection of our ground-truth dataset, consisting of malicious websites from seven distinct SE scam categories along with benign websites. We similarly build our test dataset that adds an additional five categories. The former is used to generate investigative reports using our Mentor model, which are then used to train the Investigator and Decider models in our ReDence framework. The test set, on the other hand, is used to evaluate the detection and investigative categories of the ReDence framework, and also identify whether it generalizes to scam categories not covered in our ground-truth.

This is followed by our Agentic setup in Section 5, which introduces the dedicated investigative prompts and tools we developed that allow the Mentor model, and subsequently ReDence, to have domain-specific knowledge on how to investigate SE scams as well as creating evidencegrounded reports. Then, we utilize the Mentor model to investigate the ground-truth websites and generate investigation trajectories and reports. We measure its detection effectiveness and manually evaluate the quality of generated reports. A case study also demonstrates how the Mentor applies the prompts and tools to gather evidence, verify claims, and substantiate its assessment. We also discuss the limitations encountered during these investigations that further motivate the need for our hybrid architecture in ReDence.

In Section 6, we introduce the ReDence architecture, presenting the division of responsibilities between the Investigator and Decider models, and how they communicate. We then transform Mentor’s trajectories and reports into complementary training datasets and finetune both these models, with Investigator learninng investigative reasoning, reporting, and delegation, while Decider learns to resolve contextual binary and multiple-choice questions.

Finally, Section 7 compares ReDence with Mentor and existing scam detectors, examining detection effectiveness, investigation time, memory requirements, and investigation and report quality. A comparative case study further illustrates how differences in exploration, interaction, and external verification affect the evidence collected and the resulting assessments.

## 4. Building the SE scam corpus

Evaluating our agentic framework requires direct interaction with websites, including manipulating page elements and following internal links. Existing SE scam datasets primarily provide URLs [40], static screenshots [6], or preextracted website features [41], which do not preserve the interactive environment required for our evaluation.

Ground-truth websites: We therefore construct our own ground-truth dataset by collecting candidate websites reported to the Better Business Bureau (BBB) Scam Tracker [42] between June 1 and July 6, 2026. BBB Scam Tracker is a repository of reported scams that has been widely used in prior research on online fraud [43], [44], [45]. We collect 2,175 reports that contained 728 unique domain links belonging to seven most popular scam categories on BBB: online purchase scams, counterfeit product scams, advance-fee loan scams, investment scams, debt collection scams, employment scams, and sweepstakes/lottery scams. The distribution of these samples is provided in Table 1, and the descriptions of these categores are taken from the BBB Scam Glossary <sup>1</sup>. The websites are validated by two independent coders: Coder 1 has research expertise in online fraud and abuse, while Coder 2 has prior experience in computer security through research and academic coursework.

![](images/6bd63a983e32e61a52421440659b414766f861b729b8dd0ae5b0dd58b5227346.jpg)  
Figure 1: Our framework for developing and evaluating ReDence

We also collect and manually validate 1,200 benign websites from Curlie <sup>2</sup>, a web directory that organizes websites by topic, to identify legitimate services and workflows corresponding to the SE scam categories. These include online retail and related services (379), branded-product storefronts (127), lending and credit services (242), investment and wealth-management services (103), employment and recruitment services (199), and contests and prize promotions (150).

Test set: We collect and manually validate 2,726 reports from BBB between July 12 and August 14, 2026. These reports provide an additional 1,551 websites, out of which 1,243 belong to the seven SE categories, while the remaining 308 websites belong to five held-out categories not included in our ground truth: phishing, travel/vacation/timeshare, government agency impersonation, home improvement, and fake invoices/supplier bills. We use this dataset to evaluate ReDence, as well as existing ML-based scam detectors, in Section 7. The inclusion of held-out categories also allows us to assess how well ReDence generalizes to scam types not encountered during its training. Table 1 provides the distribution of this test set.

Similar to the ground truth, we also collect and manually validate an additional 1,500 benign websites from the Curlie directory, covering online retail and related services (900), branded-product storefronts (150), lending and credit services (125), investment and wealth-management services (100), employment and recruitment services (175), and contests and prize promotions (50).

We specifically construct a separate test set as our ground-truth dataset is relatively small and exhibits substantial class imbalance across scam categories, reflecting the uneven prevalence of different scam types in the real world. Thus, a train: test split for training ReDence would not have left enough samples to evaluate it. Having a seperate test set allows ReDence to learn from the full ground-truth report, and subsequently evaluate it on a new independent set of websites.

Website preservation: While ideally our framework would be able to investigate live scam websites, malicious websites can be ephemeral [46], taken down by the hosting providers [47], or deactivated by the attackers themselves [48]. Thus, we assume a possibility of some of the websites in our ground truth no longer being active when we run our framework. Therefore, for each website, we create highly functional, replayable snapshots using Browsertrix [49], an open-source web-archiving system that has been widely used for high-fidelity web preservation [50], [51]. Browser trix captures modern websites and packages their associated resources into WACZ archives [52]. These snapshots serve two purposes: i) to provide a fallback environment that the agent can look through if the original website is no longer accessible, and ii) to preserve the websites such that our results can be reproducible in the future. For each website, we configure Browsertrix to capture the entry page and pages reachable within two link traversals on the same hostname. This includes pages linked directly from the entry page and those reached by following one additional link. We cap each crawl at 100 pages. However, we find that 80.32% of the websites in our ground truth have no more than 93 linked pages, with the median number of linked pages being 13. To evaluate the quality of our snapshots, we utilize the Memento Damage metric proposed by Brunelle et al. [53], an established measure of web-archive quality that has been adopted in subsequent studies. The metric assigns each archived webpage a normalized damage rating, $D _ { m } \in [ 0 , 1 ]$ , based on the relative importance of resources that are missing during replay, with lower values indicating higher-fidelity captures. Across our archived websites, we observe a median damage rating of 0.08 (IQR 0.03- 0.15). Across our evaluations, only 373 (14%) websites<sup>˜</sup> in our ground truth and test set became unavailable, re quiring ReDence to investigate their preserved snapshots. Among these, credential-phishing websites accounted for the majority of unavailable sites (224). Although website unavailability does not always imply enforcement action, this provides an early indication that phishing campaigns might be more short-lived than other scam types.

<table><tr><td>SE Scam Category</td><td>Ground Truth</td><td>Test Set</td></tr><tr><td>Ground-truth Categories</td><td></td><td></td></tr><tr><td>Online Purchase</td><td>587</td><td>637</td></tr><tr><td>Counterfeit Products</td><td>109</td><td>203</td></tr><tr><td>Advance-Fee Loans</td><td>30</td><td>51</td></tr><tr><td>Investment</td><td>26</td><td>59</td></tr><tr><td>Debt Collection</td><td>37</td><td>53</td></tr><tr><td>Employment</td><td>32</td><td>201</td></tr><tr><td>Sweepstakes/Lottery/Prizes</td><td>15</td><td>39</td></tr><tr><td>Held-Out Categories</td><td></td><td></td></tr><tr><td>Phishing</td><td></td><td>113</td></tr><tr><td>Travel/Vacation/Timeshare</td><td></td><td>61</td></tr><tr><td>Government Agency Impersonation</td><td></td><td>34</td></tr><tr><td>Home Improvement</td><td></td><td>64</td></tr><tr><td>Fake Invoices/Supplier Bills</td><td></td><td>36</td></tr><tr><td>Total unique websites</td><td>728</td><td>1,551</td></tr></table>

TABLE 1: Distribution of websites across the ground-truth and test datasets. Held-out categories are included only in the test set.

## 5. The Mentor Model

We select Qwen3.5:27B as our Mentor model, as it offers multimodal capabilities that allow it to interpret textual and visual observations, while its tool-calling interface enables it to invoke externally implemented tools. Moreover it offers a strong balance between instruction-following capability, computational efficiency, and memory footprint, while matching or outperforming substantially larger variants on several relevant benchmarks, including IFEval [54], IFBench [55], HLE with Tools [56], and WideSearch [57]. Its native context window of approximately 262K tokens accommodates the webpage content, tool outputs, external verification results, and evidence accumulated during multistep investigations.

However, these capabilities do not inherently provide access to the internet or the ability to investigate live websites. When provided with a website URL, the base model acknowledges this limitation and offers general advice on how to identify malicious URLs. Thus, to enable such investigations, we augment the model with tools for retrieving webpage content, interacting with websites, and searching external sources, along with task-specific guidance prompts for conducting investigations and documenting evidence. Figure 2 illustrates Mentor’s investigative workflow.

## 5.1. Investigative Prompt

Investigating an SE scam requires more than access to website content and external information. Analysts must decide which claims warrant verification, determine what additional evidence is needed, reconsider their assessment as new findings emerge, and document the basis for their conclusions. Enabling an agent to perform these tasks requires guidance on both the investigative process and how its findings should be reported. Prior work has found generalpurpose models often lack the domain-specific knowledge or procedural expertise needed to carry out these tasks systematically [22], [58], [59], [60]. We therefore develop a task-specific investigative prompt that guides the agent through these decisions and specifies how to document the resulting investigation. This approach builds on prior work showing that explicit instructions and procedural guidance can support specialized tasks without modifying the model’s underlying parameters [22], [61], [62], [63].

We organize the prompt around four questions that connect website exploration with an evidence-grounded assessment that an analyst can independently examine:

1) What does the website present, and what additional information becomes available through navigation and interaction?

2) Which claims require verification, and what investigative actions could resolve the outstanding questions?

3) How should newly collected evidence change the agent’s assessment and subsequent actions?

4) How should the findings, interpretations, and supporting sources be documented to support analyst review?

These questions motivate four corresponding components: systematic website exploration, purpose-driven planning, adaptive investigation, and evidence-grounded report ing. This design draws on research on task decomposition, prompt optimization, and reasoning guidance [64], [65], [66], [67], [68]. The prompt components are described below. The agent carries out the resulting investigative actions using the tools described in Section 5.2.

Preamble: We provide the agent with definitions of all 31 scam categories listed in the BBB Scam Type Glossary [69] and instruct it to determine whether a website is malicious and, if so, assign it to one of these categories. We include category descriptions to establish a consistent classification scheme and clarify distinctions between scam types, following prior work on explicit category definitions for zero-shot and fine-grained classification [70], [71].

Systematic website exploration (EXP): The first compo nent guides the agent in gathering observations about the website’s content and behavior. A landing page may reveal only part of a website’s activities; navigation and interaction can expose additional evidence, including credentialcollection workflows, evasive redirects, and content behind interaction barriers [1], [5], [8], [72]. The prompt therefore directs the agent to inspect relevant website components, follow internal links, and interact with page elements as needed before reaching a conclusion. Table 2 summarizes the components covered by this guidance. Initial exploration establishes the website’s apparent activities and claims, providing the observations needed to formulate a more targeted investigation.

Purpose-driven investigative planning (PLAN): Building on these initial observations, the agent identifies the website’s apparent purpose, claimed operator, actions requested from users, and plausible explanations for its behavior. It then identifies unresolved questions and constructs a concise plan linking each question to an investigative action that could resolve it. This follows security-investigation practices in which analysts formulate hypotheses or questions and identify the evidence needed to assess them [73], [74].

TABLE 2: Website components explicitly considered in our EXP prompt component.
<table><tr><td>Category</td><td>Components</td></tr><tr><td>Navigation</td><td>Header, footer, menus, links, and redirects</td></tr><tr><td>Identity</td><td>Branding, logos, and claimed organization</td></tr><tr><td>Content</td><td>Main text, offers, representations, products, and services</td></tr><tr><td>User interaction</td><td>Forms, requested information, buttons, registra- tion, login, and account access</td></tr><tr><td>Organization details</td><td>Contact information, policies, legal pages, and company information</td></tr><tr><td>Transactions</td><td>Payment methods and payment-related claims</td></tr></table>

Adaptive investigation (ADAPT): The initial plan must remain responsive to what the agent discovers. After each action, the prompt requires the agent to assess the resulting observation, reconsider its interpretation of the website, and determine whether the remaining plan should change. This follows the principle that investigative findings should inform both the plausibility of existing hypotheses and the selection of subsequent actions [75], [76].

Exploration and planning continue in a loop: an unresolved question motivates an action, the resulting observation informs the assessment, and remaining or newly identified uncertainties guide the next action. The prompt prioritizes actions that resolve meaningful uncertainties and discourages exhaustive interaction without a clear investigative purpose.

Evidence-grounded reporting (REPORT): The final component specifies how the agent documents its investigation and substantiates its assessment. Throughout the investigation, the agent records the initial plan and, for each subsequent step, the goal, action, observation, evidence update, and plan update. This record connects the investigative objectives with the actions performed and the findings that influenced subsequent decisions, following prior work on making language-agent behavior interpretable through explicit reasoning and action trajectories [77].

The agent also maintains an evidence ledger that assigns each important finding an Evidence ID and records its source, evidence type, status as a direct observation or inference, and relevance to the assessment. The ledger includes both supporting and contradictory findings. Every material factual claim in the final report must reference one or more Evidence IDs, allowing an analyst to trace the claim to its underlying evidence and examine whether the interpretation is justified. These requirements address the risk that LLM-generated reports contain assertions that are unsupported by, or contradict, the available information [78], [79].

The final output follows a structured JSON format containing the URL, category, confidence, and explanation, together with the investigative record and evidence ledger. This provides a consistent representation for parsing, comparing, and inspecting reports across websites and runs [37], [63]. The confidence field records the agent’s expressed certainty; we do not treat it as a calibrated probability, since LLMs can produce highly confident but incorrect

![](images/850bbc242be5ce04d9e3ed63e46ac3f25cfbc861166d38ebf2bb2c16e5f49eb6.jpg)  
Figure 2: Overview of Mentor’s investigative workflow.

assessments [80], [81].

Additional information: To support the interpretation of time-dependent evidence such as (but not limited to) domain age, claims about an organization’s operating history, or validity of promotions, we provide the current date. Doing so also reduces the risk of the model’s confusing its knowledge cutoff with the present.

## 5.2. Investigation Tools

To allow the agent to dynamically execute its investigative plan as directed by the prompt, it needs to directly interact with the website and its supporting artifacts to collect evidence and pursue further checks as new questions emerge. We implement three complementary tools to enable this: Fetch Page, Interact Page, and Web Search, which provide the functionality needed to carry out the procedures specified by the investigative prompt.

Fetch Page: Assessing a website requires access to both its underlying implementation and the content presented to visitors. Fetch Page retrieves four complementary representations of a webpage: its HTML source, textual content, Document Object Model (DOM), and screenshots of its rendered appearance.

The HTML source and DOM expose the page’s structure, elements, and attributes, including links, forms, and embedded content. The text allows the agent to examine service descriptions, organizational claims, contact details, and payment or refund policies. Screenshots provide visual context, including branding, layout, and information embedded in images. These representations allow the agent to examine both what the website communicates and how its content is implemented.

Interact Page: Some evidence becomes available only after a visitor opens a menu, selects a service, fills a form, or progresses through a registration or checkout workflow. Interact Page enables the agent to examine this behavior through a stateful browser environment implemented using Playwright [82].

We expose interactive elements through Set-of-Marksstyle identifiers [83]. The agent receives a numbered representation of the available elements and uses their identifiers to select targets for actions such as clicking a button or filling a field. It can also issue keyboard actions. This avoids requiring the model to generate CSS selectors or XPath expressions, which can be brittle across websites [84]. The element identifiers are refreshed as the browser state changes, allowing subsequent actions to target the updated interface. We also integrate a CAPTCHA solver based on a Selenium library [85] to process supported challenges that interrupt website interaction.

Fetch Page and Interact Page operate in an observation– interaction loop. Fetch Page retrieves the current page data, the agent examines these observations and selects an action, and Interact Page executes that action on the chosen elements. The agent can assess the outcome and determine what to do next. Together, they support investigation of multi-step workflows and connect the website’s initial claims with the behavior encountered as a visitor proceeds through its services.

Web Search: A website’s own content may provide claims without sufficient evidence to verify them. Establishing an organizational identity, professional affiliation, or business registration often require information from independent sources. Web Search supports this verification by submitting model-generated queries to web-search engines through DDGS [86], an open-source metasearch library, and returning relevant search results.

The agent formulates queries using claims and artifacts encountered during its investigation, including company names, registration numbers, email addresses, telephone numbers, and distinctive phrases. These queries can help locate authoritative records, identify contact details or content reused across websites, and uncover independently reported experiences associated with the claimed operator. Because the queries follow the findings collected during the investigation, the agent can pursue the specific uncertainties raised by each website.

Web Search works with Fetch Page to move from source discovery to source inspection. The agent selects relevant result URLs and passes them to Fetch Page to examine their contents, allowing it to compare the website’s representations with corroborating or contradictory external evidence.

## 5.3. Evaluation

Utilizing this setup, the Mentor model is run on 728 malicious and 1,200 benign websites obtained from the Ground-truth dataset. In the following paragraphs, we identify its detection effectiveness (as well as issues that cause misdetection). This is followed by a case study that demonstrates how the model investigated an SE scam website and generated it’s report. Finally, we follow this by emperically identifying the quality of its generated evidence reports.

5.3.1. Detection. The Mentor model identified 589 websites out of the 728 as malicious (80.9%), spread across the seven SE scam categories: online purchase (482), counterfeit products (95), advance-fee loans (25), investment (22), debt collection (21), employment (27), and sweepstakes/lottery/prizes (9). It misdetected 42 websites as being benign, and 97 as having ”insufficient” evidence.

To further identify the misdetections, we systematically examine Mentor’s investigation logs for each website, in cluding its recorded reasoning steps, tool selections, tool execution times, and how it adapted its investigation strategy and made decisions as evidence accumulated. Not constrained by a fixed time limit, Mentor required a median of 199 seconds and 17 tool calls per website. However, we find that 68% of the aggregate runtime was spent on decision-making tasks, including selecting tools, choosing which pages or search results to inspect, assessing how new evidence affects the investigation, and deciding whether to continue, revise, or conclude the investigation.

We also find that in several cases, Mentor repeatedly reconsidered unresolved questions and issued additional tool calls without obtaining evidence that resolved them. Among the 97 websites marked as having inconclusive evidence, we identify this phenomenon in 72 cases. This pattern also occurred in 26 of the 42 websites classified as benign, where Mentor instead treated the supposed lack of evidence as grounds for a benign verdict. Prior research shows that extended reasoning can produce redundant solutions with limited additional benefit [87], while agent loops can repeat actions that return previously available observations without advancing the task [88]. We also examine exact duplicate tool calls, excluding fetches of the original target URL. These repetitions occur in 195 investigations (26.8%).

Together, these findings motivate integrating a decision model into ReDence to reduce decision-making overhead and limit repeated deliberation and tool calls that do not advance the investigation.

5.3.2. Case Study. To illustrate how Mentor investigates a website’s claims and documents the evidence supporting its assessment, we examine publictradeinc.com, an investment scam website.

Investigation: Mentor begins by examining the landing page of the website using the Fetch Page tool and finds that it advertises trading services and encourages visitors to create an account on the platform. Following the prompt’s guidance on systematic website exploration, the agent uses Interact Page and subsequent Fetch Page calls to inspect the website’s organizational information, contact details, FAQ, and registration page.

Attempts to retrieve the legal-and-regulation, about-us, terms-and-conditions, and privacy-policy pages return 404 errors. These failures leave several questions about the operator and its terms unanswered. The contact page, however, lists offices in New York, London, and Tokyo, together with a telephone number. The agent uses Web Search to check the advertised address and telephone number but does not find results establishing their association with PublicTrade.

These unresolved identity claims direct the investigation toward more specific information about the business. The FAQ provides a concrete lead: the website claims to be registered in the United Kingdom under company number 08683932. For a platform encouraging users to invest, checking who operates the service is central to establishing its credibility [89]. This focus reflects the prompt’s purposedriven planning, with the website’s advertised activity determining which claims require verification.

Examining the registration page reveals fields requesting a username, email address, password, and wallet address but no identity documents or other Know Your Customer (KYC) verification. Mentor flags this omission as suspicious, stating that “legitimate brokers require identity verification,” and contrasting this requirement with the platform’s minimal registration fields. This concern aligns with UK customer due diligence requirements, which require covered financial institutions to identify customers and verify their identities [90].

Mentor then searches for the company registration number. The results reveal that the same number appears on two other trading websites: swiftapexcapital.com and bitcloven.com. This leads to the model raising a further question: which company does the number identify, and is PublicTrade associated with it?

Because the website claims UK registration, the agent checks the domain’s WHOIS record, identifying that it had been registered during December 2025 in Nairobi, Kenya. Subsequently, it searches for the corresponding registration number at Companies House [91], the UK government agency responsible for maintaining the public register of companies. This leads Mentor to a record showing that the number belongs to ALLBIT LTD, whose listed business activity is retail sales through mail order or the internet, a mismatch with the cryptocurrency trading and investment services advertised by publictradeinc.com. Thus, neither the company name nor its listed activity substantiates PublicTrade’s advertised financial-service identity. A further search combining PublicTrade and ALLBIT LTD also fails to establish a relationship between them. Mentor interprets these findings as indicating that the platform is using another company’s credentials to establish credibility.

To further validate its suspicions, Mentor searches for the domains associated with the same company registration number, namely swiftapexcapital.com and bitcloven.com, and identifies that they had been previously flagged by scam-tracking websites scamtracinghub.com and trackfrauds.com, and have multiple Trustpilot reviews of users being scammed.

This overall progression illustrates Mentor’s adaptive investigation capabilities: the registration number claim prompts an external search, its reuse introduces another uncertainty, and the agent follows up by checking the offi cial record and the claimed association. The investigation concludes by bringing together the unresolved corporate identity, reused registration number, trading-bot profit claim, unavailable legal pages, advertised payment methods, and absence of identity verification during initial registration. Mentor interprets this combination as a platform using unsubstantiated legitimacy claims to encourage investment and classifies the website as an investment or cryptocurrency scam, with high confidence.

![](images/4d47ca9e67c5afd8008b745d33dd9b3a1fdb73425e9893800f13641a0a905dae.jpg)

Figure 3: PublicTrade’s website (a) The landing page advertises cryptocurrency, forex, and other trading services. (b) The Companies House record for the website’s claimed registration number lists retail sales as its business activity, a mismatch with its claimed purpose.  
![](images/9d880bed369bc20ec4b525e9fa896a4c79f6d0e312bdbefa998ca7aeb88d10b8.jpg)  
Figure 4: Condensed excerpt from Mentor’s PublicTrade evidence ledger, documenting the website’s registration claim and the corresponding official company record.

Reporting: The generated report documents the website’s purpose, claimed organization, verdict, and confidence, followed by an evidence ledger containing twelve findings. Each entry includes an Evidence ID, observation, source URL, evidence type, status, and explanation of its relevance. The complete report references these entries in its keyevidence section and final explanation.

Figure 4 presents two entries illustrating how the ledger supports verification of the website’s corporate identity. E2 records the FAQ’s claim of UK company registration under number 08683932 and identifies the page containing that claim. E3 records the corresponding Companies House finding, which identifies ALLBIT LTD and lists retail sales via mail order or the Internet as its business activity. Its relevance field explains that these details do not substantiate PublicTrade’s advertised financial-service identity. Together, the entries allow an analyst to locate the website’s claim, inspect the official record, and assess whether the reported interpretation is justified.

## 5.4. Evaluating Reports

Note: We are currently repeating this analysis with two independent coders to measure intercoder agreement. The evaluation described below uses one coder.

We now evaluate whether Mentor’s reports provide relevant, verifiable evidence that analysts can use to assess a website’s legitimacy. Our evaluation considers both whether the coder correctly classifies websites and whether the reports reveal additional evidence, including for websites the coder classifies correctly. We focus on evidence coverage and auditable evidence coverage because analysts need findings that substantiate an assessment and sufficient documentation to independently verify them [92], [93], [94].

Our coder is a cybersecurity professional specialized in identifying online fraud and abuse, including SE scam websites. We select five malicious websites from each of the seven SE scam categories, yielding 35 malicious websites. For each category, we also select five benign websites offering corresponding products or services, producing a balanced sample of 70 websites. Our coder investigated each website for 90 seconds, recording a classification and the distinct findings supporting or contradicting its legitimacy. This record is finalized before the coder sees Mentor’s report. The coder subsequently examines the report to determine whether it recovers the initial findings, identifies additional relevant evidence, and provides sufficient supporting material to verify its claims.

For each website w, let $H _ { w }$ denote the deduplicated set of correct, relevant findings recorded during the initial manual investigation, and let $M _ { w }$ denote the correct, relevant findings identified in Mentor’s report. The coder checks findings from both sources against the website, external sources, or collected artifacts. Additional findings from Mentor are accepted only after verification. Repeated descriptions of the same finding are counted once, and evidence supporting legitimacy is considered alongside evidence suggesting deception.

We construct a pooled reference set $G _ { w } = H _ { w } \cup M _ { w } ,$ representing the verified findings discovered through either process. This is an observed evidence inventory rather than an exhaustive account of everything that could be discovered about the website. We calculate two metrics:

Evidence coverage (EC): The proportion of verified reference findings correctly identified by Mentor:

$$
\mathrm { E C } ( w ) = \frac { | M _ { w } | } { | G _ { w } | } .\tag{1}
$$

This metric credits both findings initially identified by the coder and additional findings uncovered by Mentor. Evaluating distinct findings follows prior work on reference-based content coverage [95].

Auditable evidence coverage (AEC): The proportion of verified reference findings that Mentor both identifies and documents sufficiently for independent verification:

$$
\mathrm { A E C } ( w ) = \frac { \sum _ { e \in M _ { w } } a _ { w } ( e ) } { | G _ { w } | } ,\tag{2}
$$

where $a _ { w } ( e ) \ = \ 1$ if the report provides an identifiable source, collected artifact, or documented interaction suffi cient to verify finding e, and 0 otherwise. A source URL alone is insufficient if its contents do not substantiate the finding. This criterion reflects the importance of sourcesupported claims in attributable reporting [93], [94].

The coder correctly identifies 21 of the 35 malicious websites, incorrectly classifies 7 as benign, and returns inconclusive assessments for the remaining 7. Among the 35 benign websites, the coder correctly classifies 31 and incorrectly labels 4 as malicious. These outcomes illustrate the difficulty of distinguishing legitimate services from deceptive websites within a brief manual investigation. Correct classification also does not necessarily imply a comprehensive investigation. Among the 52 websites the coder classifies correctly, the initial manual records contain an average of 2.4 distinct relevant findings per website, compared with 6.8 in Mentor’s reports for the same subset. Thus, even for correctly classified websites, Mentor’s reports provide additional evidence supporting the assessment.

Across the full sample, Mentor achieves a mean evidence coverage of 0.87 and auditable evidence coverage of 0.79. Its reports contain 364 additional findings absent from the coder’s initial records, of which 327 are treated as auditable.

Overall, these findings demonstrate that Mentor’s reports not only go beyond providing a classification decision by uncovering relevant evidence that analysts may overlook during time-constrained manual investigations, but also enrich an analyst’s understanding of the attack, even when the analyst’s initial classification is correct.

## 6. ReDence

To develop ReDence, we distill the Mentor model’s investigative and reporting capabilities into two smaller models that work together with complementary responsibil ities: 1) Investigator, based on Qwen3.5-4B, handles openended tasks such as planning investigations, interpreting findings, and generating evidence-grounded reports; and 2) Decider, based on Tev1-4B, handles binary and multiplechoice decisions, including selecting browser actions, assessing evidence against specified criteria, and choosing subsequent investigative steps. Decision models such as Tev1 are designed to rapidly select among candidate options given structured context and a question [96]. This makes them particularly well-suited for frequent, well-defined decisions, where traditional LLMs incur substantial computational and latency overhead from autoregressive token generation [28].

The investigator formulates decision requests, which Decider resolves through its concise selection interface. Table 3 illustrates how responsibilities are divided between the two models across the different stages of a website investigation.

We also include Controller, an in-between module which coordinates their exchanges, passes the relevant context and options to Decider, returns its selection to Investigator, and executes the associated tool operations. We illustrate the overall configuration, as well as how these models were fine-tuned in the proceeding sections.

## 6.1. Constructing the Training Datasets

Prior work has shown that explanations and reasoning traces generated by a larger model can improve the taskspecific capabilities of smaller models [38], [97], [98], [99]. We construct both datasets from Mentor’s investigations of malicious and benign websites in our ground-truth dataset (Section 5.3). Each trajectory begins with the system prompt defining the agent’s role and available tools, followed by the initial user message containing the target URL, investigation requirements, and prefetched information. The remaining conversation records Mentor’s intermediate responses, tool calls, and tool observations, concluding with its structured report.

These trajectories preserve the information available when each investigative action was selected. They therefore capture the sequential nature of website investigation, where observations inform subsequent actions and assessments, following prior work that uses action–observation sequences to gather external information and guide decisions across multiple steps [39], [77], [100].

Investigator training data: Investigator receives most of the original trajectory, including Mentor’s planning, openended interpretation, evidence synthesis, and final reporting. We modify the instructions and corresponding assistant turns to demonstrate when and how to delegate decisions to Decider through the controller. These delegated decisions include choosing a search result, selecting a website element or form option, answering a bounded question about collected evidence, and choosing whether to continue, revise, or conclude an investigation.

For each eligible decision, we replace Mentor’s original selection step with a delegation request containing the investigative objective, relevant context, decision question, and available alternatives. Investigator’s next training target then demonstrates how to use this returned information to continue the investigation.

Decider training data: Decider is trained on individual decision examples extracted from the same Mentor trajectories. Each example contains the investigative context, a question, and a set of labeled alternatives. The target is the option corresponding to Mentor’s recorded action or explicit judgment. Unlike Investigator, Decider is trained to return the selected option without generating the surrounding investigative explanation.

The selection examples cover five types of decisions:

• Choosing which search result or external source to retrieve for a specified verification task.

• Choosing among available links, buttons, form options, or other interactions relevant to the current objective.

• Answering bounded questions about whether the available observations support, contradict, or fail to establish a particular claim.

• Choosing whether to continue the current approach, revise it, or conclude the investigation.

• Selecting the final verdict and, when applicable, a scam category using the accumulated evidence.

For each example, we derive the candidate choices and correct response from Mentor’s recorded investigation. For action-selection tasks, the correct response is the action Mentor performed, while for evidence-assessment tasks, it is Mentor’s interpretation of the available evidence. Finalassessment tasks use the evidence accumulated before Mentor generates its concluding report. Browser-action choices are drawn from the recorded webpage state, whereas sourceselection choices are derived from the search results returned during the investigation. We retain only examples where the context, candidate choices, and correct response can be reliably reconstructed from Mentor’s trajectory. Where appropriate, we also include options such as insufficient evidence or continuing the investigation.

Finetuning configuration: We independently fine-tune both models using QLoRA with frozen 4-bit backbones and trainable LoRA adapters [101], implemented through Hug ging Face Transformers and PEFT [102]. Completion-only cross-entropy loss is applied to Investigator’s target assistant responses and Decider’s option labels and end-of-sequence markers; context and tool observations are excluded from the loss. Investigator uses a maximum sequence length of 16,384 tokens, LoRA rank 16, α = 32, and three training epochs at a learning rate of $1 \times 1 0 ^ { - 4 }$ . Decider uses 2,048 tokens, rank $8 , \alpha = 1 6 ,$ , and one epoch at $5 \times 1 0 ^ { - 5 }$

## 6.2. Framework Coordination with Controller

Controller connects the two models with the investigation tools described in Section 5.2. Investigator maintains the broader investigative context and identifies when a task requires a binary or multiple-choice decision. It then sends Controller a request containing the current objective, relevant observations and uncertainties, the decision question, and the available alternatives. Controller validates the request and passes it to Decider. Decider returns its selected option to the controller, which records the decision and passes it back to Investigator.

When the selected option specifies a tool operation, the controller executes that operation and returns its results alongside the choice. Investigator does not need to generate the same tool call again. This allows Decider to direct source retrieval or browser interaction within the task specified by Investigator, while the controller handles execution and preserves the resulting observations.

<table><tr><td>Task</td><td>Investigator Qwen3.5-4B Open-ended reasoning and generation</td><td>Decider Tev1-4B Binary and multiple-choice decisions</td></tr><tr><td>Planning</td><td>Identify the website&#x27;s purpose, define objectives, and formulate an investigative plan.</td><td>Choose among candidate actions addressing the current objective.</td></tr><tr><td>External verification</td><td>Identify claims requiring verification and generate</td><td>Select search results or external sources for retrieval.</td></tr><tr><td>Browser interaction</td><td>search queries. Define the interaction&#x27;s objective and supply required</td><td>Select links, buttons, form options, or other available browser actions.</td></tr><tr><td>Evidence assessment</td><td>input values. Organize observations, interpret relationships, and formulate evidence questions.</td><td>Resolve explicit binary or multiple-choice questions about the available evidence.</td></tr><tr><td>Adaptive follow-up</td><td>Identify unresolved questions and formulate revisions to the plan.</td><td>Choose whether to continue, revise, or conclude the investigation.</td></tr><tr><td>Final assessment</td><td>Synthesize findings and generate an evidence-grounded report.</td><td>Select the verdict and applicable scam category from the supplied alternatives.</td></tr></table>

TABLE 3: Division of responsibilities between ReDence’s Investigator and Decider.

![](images/84f172e4414aab26b1116ff46a824017ab1e699cc5fc1d0d9512792d60f7448c.jpg)  
Figure 5: Controller routes context and decision requests from Investigator to Decider, and then returns decisions and tool observations back to Investigator.

For example, Investigator may identify a navigation menu and seek the page most likely to establish the website’s claimed operator. It supplies this objective and the available menu entries through the controller. Decider chooses the relevant entry, and the controller performs the interaction, retrieves the updated page content, and refreshes the available element identifiers. Investigator then receives the selected action and its outcome, interprets the new information, and continues the investigation.

The same handoff applies to evidence judgments and investigation control. Investigator formulates a specific question and supplies the observations needed to answer it; while Decider chooses among the permitted responses, and Investigator incorporates the returned judgment into its subsequent reasoning. A judgment about one feature is distinguished from an assessment of the entire website.

Controller records the context supplied for each decision, Decider’s response, the tools executed, and the resulting observations or errors. Tool outputs are retained as observations, while Decider’s answers are recorded as model judgments. Investigator uses these records to maintain the evidence ledger, reconsider unresolved questions, and explain how the findings relate to its assessment. Before producing the final report, Investigator synthesizes the accumulated evidence and delegates the verdict and applicable category selection through the same interface.

Figure 5 illustrates a macroscopic diagram of how each of the three components i.e. Investigator, Controller and Decider communicate with each other, whereas Algorithm 1 summarizes this coordination across the four investigative stages. Controller also enforces the execution budget (120 seconds per website) and checks the final report’s structure and evidence references.

## 7. Evaluating ReDence

To identify the effectiveness of the ReDence framework with respect to both speed and detection accuracy, we compare it directly with the Mentor model, as well as three existing automated detectors: URLNet [103], BeyondPhish [1], and ScamFerret [20].

URLNet [103] is a convolutional neural network that learns character-level and word-level representations directly from URL strings to identify malicious URLs. We include URLNet because URL-based features remain widely used in scam detection systems [1], [104], primarily due to their computational efficiency and suitability for large-scale detection. However, prior research has demonstrated how scammers have increasingly adapted their URL structures and naming strategies to evade such detection approaches [105].

In contrast, BeyondPhish [1] uses predefined features derived from website content, URLs, domain information, and associated social media accounts to classify fraudulent websites.

Finally, ScamFerret [20] is the closest to ReDence in terms of functionality. It employs a text-based LLM guided by predefined scam indicators to iteratively select from a fixed set of tools, gather information, and explain its assessment. However, ScamFerret cannot directly interact with websites or dynamically adapt its investigation based on observed behavior, while its reliance on a fixed set of tools further constrains the evidence it can collect. As we

Algorithm 1 ReDence investigation with delegated deci  
sions.   
Require: URL, date, BBB categories, budget   
Ensure: Structured assessment with traceable evidence   
1: Define Investigator I, Decider D, Controller C; initialize evidence   
ledger L and investigation record R   
Stage 1: Systematic website exploration   
2: C fetches the initial page; if inaccessible, return access details and   
insufficient evidence   
3: I examines text, visual content, claims, and available interactions   
Stage 2: Purpose-driven planning   
4: I identifies purpose, operator, requested user actions, and plausible   
explanations; links unresolved questions to investigative objectives and   
candidate checks   
Stage 3: Adaptive investigation   
5: while budget remains and investigation is not concluded do   
6: I prepares context, questions, candidate options, search queries, and   
interaction inputs as needed   
7: Through DECIDE, D selects tools, pages/search results, or browser   
actions such as links, buttons, and form options   
8: C executes selected or directly requested tool calls; returns updated   
observations or errors to I   
9: I interprets findings and formulates evidence questions; D resolves   
binary and multiple-choice judgments through DECIDE   
10: I updates L with evidence IDs, observations, sources, types, obser  
vation/inference status, and relevance   
11: Through DECIDE, D selects continue, revise, or conclude; I updates   
the plan and records objectives, decisions, actions, observations, and   
updates in R   
12: end while   
Stage 4: Evidence-grounded reporting   
13: I synthesizes supporting and contradictory findings; D selects the   
verdict and applicable BBB category through DECIDE   
14: I generates JSON containing L, R, evidence-linked explanations,   
limitations, and unresolved questions   
15: C validates structure and evidence references; return report   
16: function DECIDE(x, q, O)   
17: I → C → D: context x, question q, options O   
18: D → C → I: selected option; C validates requests and selections,   
records the outcome, and returns a decision or explicit error   
19: end function

demonstrate in Section 7.2, these limitations cause ScamFerret to overlook important evidence and malicious workflows, reducing its detection effectiveness. Furthermore, although ScamFerret provides rationales for its classifications, it lacks an evidence ledger that explicitly links conclusions to specific observations, sources, and investigative actions while distinguishing factual observations from interpretations. In Section 7.3, we further demonstrate how this limitation reduces the verifiability and practical utility of its investigation reports.

We evaluate all six configurations on the separate test set described in Section 4, comprising 1,551 scam websites and 1,500 benign websites. As both URLNet or BeyondPhish are classifier tools, they do not provide any logs or reports for their actions, and thus, we only test their detection effectiveness. On the other hand, we evaluate ScamFerret for both detection and reporting effectiveness, along with a case study (Section 7.2) where we directly compare the investigative approach taken by ReDence with ScamFerret.

## 7.1. Detection Effectiveness

Table 4 illustrates the detection effectivess of the tested tools. The values in paranthesis denote the performance on the 308 scams for the held-out categories that were not included in the ground-truth.

We find that ReDence not only matches but significantly exceeds the detection performance of the Mentor model, achieving an F1 score of 0.97 compared to 0.82 for the Mentor. A key factor contributing to ReDence’s effectiveness is its ability to complete investigations for all websites in the test set, despite operating under a time constraint. On the other hand, the Mentor model failed to complete the investigation for 208 websites.

We further observe that both models retain their detection effectiveness on the held-out categories of social engineering (SE) scams, achieving F1 scores of 0.96 and 0.82 for ReDence and Mentor, respectively. Notably, ReDence maintains the same recall of 0.98 on both the development and held-out categories, with only a marginal reduction in precision. These results provide evidence that ReDence’s investigative capabilities generalize beyond the scam categories represented in the ground truth categories that it was trained on.

ReDence also substantially improves computational efficiency, both in terms of runtime and memory consumption. It completes website investigations in a median of 43.8 seconds, compared to 194 seconds for the Mentor model. We also found that ReDence required significantly fewer tool calls, with a median of nine per website compared to 15 for Mentor. Decision-making also contributed to only 14% of ReDence’s aggregate runtime, compared to 71% for Mentor, illustrating the potential reduction in overhead from delegating discrete choices to the Decider model. Similarly, ReDence consumes only 3.4 GB of GPU memory, with the CPU-bound Decider model using three additional GBs of system RAM, compared to 33.8 GB for the Mentor.

Among the ScamFerret configurations, the GPT-4.1 implementation achieves better detection performance, with an F1 score of 0.82. The ScamFerret framework imposes a maximum of 15 iterations, where each iteration represents a cycle of LLM reasoning and tool selection or execution, or a runtime limit of 600 seconds, whichever is reached first. Adhering to these limits, we found that it failed to complete investigations for 115 websites, during which it encountered investigation errors, prematurely terminated investigations, and incomplete reports. We also observed that evaluating this configuration on our test set alone incurred about \$39 in API charges. Considering that our test set realistically represents a small fraction of the websites that commercial security vendors may need to examine daily, this raises concerns about the cost and scalability of ScamFerret’s APIbased investigations.

The Llama-3.3-70B configuration of ScamFerret achieves lower detection performance, with an F1 score of 0.71. A substantial portion of this reduction is attributable to its failure to complete investigations for 239 websites,. This configuration also incurs the highest GPU memory consumption among the evaluated models (45.7 GB) and the second-longest median investigation time (127.4 seconds), behind the Mentor model.

<table><tr><td>Configuration</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td><td>Time (Median, s)</td><td>GPU Usage (Median, GB)</td></tr><tr><td>ReDence (4B)</td><td>0.96 (0.95)</td><td>0.97 (0.95)</td><td>0.98 (0.98)</td><td>0.97 (0.96)</td><td>43.8 (27.2)</td><td>3.4GB</td></tr><tr><td>Mentor (27B)</td><td>0.82 (0.82)</td><td>0.83 (0.82)</td><td>0.81 (0.83)</td><td>0.82 (0.82)</td><td>194.0 (151.2)</td><td>33.8GB</td></tr><tr><td>URLNet</td><td>0.41</td><td>0.35</td><td>0.19</td><td>0.25</td><td>1.2 (1.2)</td><td>1.5GB</td></tr><tr><td>BeyondPhish</td><td>0.72</td><td>0.87</td><td>0.70</td><td>0.78</td><td>53.7 (1.8)</td><td>3.7GB</td></tr><tr><td>ScamFerret (GPT 4.1)</td><td>0.79</td><td>0.84</td><td>0.81</td><td>0.82</td><td>88.5 (41.2)</td><td>API-bound</td></tr><tr><td>ScamFerret (Llama3.3:70B)</td><td>0.71</td><td>0.75</td><td>0.67</td><td>0.71</td><td>127.4 (92.7)</td><td>45.7GB</td></tr></table>

TABLE 4: Comparison of detection performance and computational efficiency across the models on the test set. Parenthesized values for detection metrics for ReDence and Mentor indicate performance on the held-out categories, while those for median time indicate inference time excluding website latency.

Among the classification-based detectors, URLNet has the lowest memory consumption (1.5 GB) and shortest runtime (1.2 seconds), but also exhibits the poorest detection performance, with an F1 score of only 0.25. This result is consistent with the limitations of relying exclusively on URL-level characteristics, as most of the malicious websites in our test set do not exhibit the suspicious URL patterns that such models are designed to recognize, being consistent with the real world where attackers presently use legitimate domain names and free SSL ceritification. Thus, even though URLNet offers substantial computational savings, it comes at a considerable cost to detection coverage.

Finally, BeyondPhish has a slightly slower median runtime (53.7 seconds), while requiring 3.7GB system RAM. Its overall detection performance is considerably lower, with an F1 score of 0.78. In particular, BeyondPhish exhibits relatively high precision (0.87) but lower recall (0.70), missing a substantial proportion of malicious websites. We also observe that BeyondPhish performs notably better on online purchase scams, identifying 522 out of 637 such websites ( 82%) in the test set. It also detected 72.4% of scams for the counterfeit category, which is related to online purchase. However, it has significantly lower performance for other SE scam categories such as Employment scams (44.2%) and sweepstakes (17.9%.)

## 7.2. Case Study: wego-france.com

We present a case study to illustrate in-depth how Re-Dence and ScamFerret investigate the same website, highlighting differences in the information considered, investigative actions performed, and evidence supporting their conclusions.

We examine wego-france.com/backpack, a sweepstakes scam website impersonating a DICK’S Sporting Goods giveaway. The website invites visitors to complete a survey for a chance to receive a YETI backpack, subsequently directing them to requests for personal and creditcard information. It also has a valid SSL certificate issued by Let’s Encrypt, a common practice among scammers seeking to make their websites appear legitimate [106]. By manually investigating the website, we found the following malicious artifacts:

• The page displays the DICK’S Sporting Goods logo in its banner. This branding does not appear elsewhere on the website, either as an image or as text.

• The promotion claims that only ten winners will be selected and displays a 15-minute countdown to encourage immediate participation. However, the countdown resets to a higher value after approaching zero, while the offer remains available, undermining the claimed deadline.

• Facebook-style comment widgets describe successful deliveries and include photographs of supposedly received backpacks. However, the commenter names do not link to actual Facebook profiles, and the reaction buttons are nonfunctional.

• Completing a four-question survey reveals a gift-box game. In the observed sequence, the first selection produces an empty-box message and encourages another attempt, while the second displays a winning message and assigns a prize identifier.

• A payment requirement is introduced after the visitor wins. The winning screen directs visitors to a shipping page and instructs them to pay for delivery. The subsequent workflow requests their name, email address, telephone number, and delivery address, followed by credit-card details, to claim the advertised reward.

We tracked the investigative steps taken by ReDence and ScamFerret by looking at their execution logs. Both models record the selected tool, tool input, returned observation, and model’s stated interpretation before the final answer.

Both ReDence and ScamFerret begin by retrieving the website’s content, but differ in their ability to identify the retailer being impersonated. ReDence uses its Fetch Page tool to examine the website’s text, DOM, and rendered appearance. By inspecting the logo image, it recognizes a visual imitation of DICK’S Sporting Goods, which does not align with the website’s URL, and identifies the claimed retailer affiliation as requiring verification. ScamFerret, in contrast, retrieves the website’s text, source code, and hyperlinks but fails to recognize the impersonated retailer. The logo is stored under the generic filename assets/logo.png, and the brand name does not appear elsewhere in the website’s text or source code. Consequently, ScamFerret associates the investigated URL with Wego, identifying wego.com as its official domain, rather than recognizing the DICK’S Sporting Goods impersonation.

![](images/17c75df2dd873467a2c827349f62f36021d716e3c11b3fc95c84dc2ca49e5729.jpg)  
(a) Landing page

![](images/5ea6c679273748e826a011897add55307b09d185cef560b9f24fa4e5d3a3cad3.jpg)  
(b) Fake Facebook comment widget

![](images/2ade28dce89deb2d2257995c19ec0e17bd73accc9b5ff76fb89a3260336f0900.jpg)  
(c) Fake payment page  
Figure 6: Preview of wego-france.com: (a) Landing page containing the organization (Dick’s) logo, the (fake) timer to induce urgency, and the survey. (b) The fake facebook comment widget without any interactive elements. (c) The fake payment page which is only visible after interacting with the survey and the subsequent page which asks for non-payment information.

Both agents subsequently retrieve the domain’s WHOIS record and identify that it was registered in May 2026, four months before the investigation. However, they interpret this information differently. ReDence acknowledges that recently registered domains are often associated with malicious activity, but recognizes that domain age alone is insufficient to establish maliciousness and redirects its attention to the website. ScamFerret, on the other hand, incorrectly rejects the WHOIS record as invalid, claiming that the registration date is in “the future.” We assume this results from its underlying GPT-4.1 model’s June 2024 knowledge cutoff.

ReDence then examines the website’s product photographs, 15-minute countdown, and Facebook-style comments, including apparent replies from a verified retailer account. Based on these observations, it formulates an investigative plan to verify the credibility of the endorsements, test the advertised deadline, and determine what visitors must do to claim the reward. ScamFerret also identifies the Facebook-style comments but cannot interact with them to verify their authenticity. While it observes that the profile names and reaction controls lack hyperlinks, it considers this evidence inconclusive, suggesting that the overwhelmingly positive reviews could reflect comments the website “intentionally” highlights to promote the product. ReDence, in contrast, uses Interact Page to attempt to follow the commenter identities and activate the reaction controls. It confirms that the profile names are not clickable and the reaction buttons are nonfunctional. Combined with the overwhelmingly positive comments, these observations lead

ReDence to identify the section as fabricated review engagement promoting a counterfeit product.

ReDence further investigates the 15-minute countdown associated with the advertised prize. Rather than accepting the displayed deadline at face value, it identifies the function controlling the countdown from the DOM, sets the timer to five seconds, and waits for the apparent deadline to expire. It observes that the countdown resets while the promotion remains accessible, allowing it to establish that the timer creates artificial urgency rather than enforcing a genuine deadline. ScamFerret, which cannot interact with the page or examine its dynamic behavior, does not identify this deceptive mechanism.

During its investigation, ScamFerret also generates the search queries “wego-france.com scam” and “YETI backpack giveaway” to seek external evidence. The first leads it to a Trustpilot page for Wego.com, where it observes over 2,000 reviews and a rating of 3.1 stars. Despite the average rating, ScamFerret interprets the number of reviews as a possible indication of legitimacy, although it cannot establish a connection between the investigated domain and the official Wego website. The second query returns an official YETI giveaway page, https://www.yeti.com/mysterytundra-giveaway.html, from which ScamFerret incorrectly infers that a backpack giveaway is taking place and that the investigated website might therefore have some legitimacy. Although it acknowledges that scammers could exploit such promotions, these search results introduce uncertainty rather than helping it establish whether the investigated website is malicious.

The most significant difference between the two investigations emerges when examining the reward-claiming process. Both agents identify the initial survey, but only ReDence proceeds to interact with it. Using Interact Page, ReDence completes the four survey questions and retrieves the updated page after each action. When the survey reveals a gift-box game, it adapts its investigation to determine how the prize is awarded and what follows. The first selected box is empty and prompts another attempt, while the second produces a winning message, a prize identifier, and instructions leading to a shipping page. ReDence continues interacting with the website, entering dummy name and address information into the shipping form, which subsequently leads to a page requesting payment information. ScamFerret, in contrast, cannot answer the survey questions or interact with the gift boxes. Consequently, it does not uncover the game, winning message, or subsequent requests for shipping information and credit-card details.

These differences ultimately lead the two agents to conflicting conclusions. ReDence connects the impersonated retailer branding, fabricated review engagement, resetting countdown, and deceptive reward-claiming process to the collection of personal and payment information. Based on this accumulated evidence, it classifies the website as a sweepstakes scam. ScamFerret, meanwhile, bases its conclusion primarily on the mismatch between the investigated domain and Wego’s official website, the positive Facebookstyle reviews, and the presence of a survey. It states that these artifacts do not “comprehensively” establish maliciousness and recommends further investigation. For its binary classification, ScamFerret returns False (Not malicious) and assigns the category “Online shopping.” Thus, while both agents encounter several of the same initial indicators, ReDence’s ability to visually interpret website content, interact with dynamic elements, and adapt its investigation allows it to uncover evidence that ScamFerret misses, ultimately resulting in a more complete and accurate assessment of the website.

Moreover, we also find that ScamFerret’s report lacks the evidence-grounded structure of ReDence’s reports (retained from the Mentor model), as illustrated in Figures 7 and 8. Beyond documenting individual findings, ReDence explains how they jointly support its interpretation. For example, E001 records the comments claiming successful deliveries, while E002 documents the nonclickable commenter names and unresponsive reaction controls. The keyevidence section references both entries, explaining that the delivery endorsements provide apparent social proof, while the nonfunctional controls weaken its credibility. ScamFerret instead suggests that the comments could be reviews intentionally highlighted to promote the offer, without connecting this interpretation to documented interaction checks. Thus, not only does it find more features during its investigation, but can also present it in a way that allows an analyst to examine both the supporting observations and the reasoning that connects them, providing a clearer basis for evaluating whether the interpretation is justified.

## ReDence: Evidence Ledger Excerpt

## Evidence ID: E001

Observation: Facebook-style comments claim successful deliveries and include photographs presented as showing received backpacks.

• Source: wego-france.com/backpack, comment section.

• Evidence type: Website content.

• Status: Direct observation.

• Relevance: Presents apparent customer endorsements of the giveaway.

## Evidence ID: E002

• Observation: Commenter names are not clickable, and reaction buttons do not respond.

• Source: wego-france.com/backpack, comment controls.

• Action: Attempt to follow commenter names and activate reactions.

• Evidence type: Website interaction.

• Status: Direct observation.

• Relevance: The displayed engagement cannot be checked or reproduced through these controls.

## Key Evidence

• Claim (inference): The endorsements may simulate social engagement to reassure visitors.

• Evidence IDs: E001, E002.

Explanation: The delivery claims (E001) provide apparent social proof, while the nonfunctional profile and reaction controls (E002) weaken its credibility.

Figure 7: ReDence report excerpt connecting claimed customer endorsements with observations from testing the comment controls.  
ScamFerret: Report Excerpt   
{   
"scam": false,   
"website\_type": "Online Shopping",   
"reason": "... The positive Facebook-style   
comments have no profile links, but they   
could represent reviews intentionally   
highlighted to promote the offer. ...   
The retrieved text and hyperlinks do not   
comprehensively establish that this   
website is malicious. ... further   
investigation is recommended."   
}  
Figure 8: ScamFerret report excerpt presenting its classification, website category, and narrative explanation.

## 7.3. Reporting

We also evaluate whether Mentor’s reporting capabilities transfer to ReDence. Following the procedure described in Section 5.4, we randomly select five malicious websites from each of the twelve scam categories, comprising the seven original SE scam categories and five held-out categories, yielding 60 malicious websites. We also include 60 benign websites. Our coder first investigates each website independently, then evaluates the reports generated by ReDence, Mentor, and ScamFerret using evidence coverage (EC) and auditable evidence coverage (AEC).

The coder correctly identifies 37 of the 60 malicious websites, returns inconclusive assessments for 13, and incorrectly classifies the remaining 10 as benign. We find that the coder correctly identifies all five credential-phishing websites, but performance is less consistent across the other categories. This pattern further highlights the difficulty of recognizing scams within a limited investigation period when their deceptive behavior requires exploration beyond the landing page. Among the 60 benign websites, the coder correctly classifies 51, returns inconclusive assessments for 5, and incorrectly labels 4 as malicious. Even when the coder reaches the correct classification, the reports provide additional relevant evidence. Across the 37 websites correctly identified as malicious, the coder records an average of 2.8 distinct relevant findings per website, compared with 7.0 for ReDence, 6.5 for Mentor, and 3.4 for ScamFerret. ReDence therefore identifies slightly more relevant findings than Mentor and more than twice as many as ScamFerret on the same set of websites.

Over the sample, ReDence achieves a median Evidence coverage (EC) score of 0.87 and auditable evidence coverage (AEC) of 0.81. Mentor achieves corresponding scores of 0.84 and 0.77, closely matching its performance on the ground-truth dataset. On the other hand, ScamFerret obtains scores of 0.44 and 0.30 for EC and AEC scores respectively. This shows a modest advantage for ReDence over Mentor and a significantly larger advantage over ScamFerret in both capturing relevant findings and documenting sufficient supporting material for analysts to verify them.

## 8. Conclusion

Our findings demonstrate the benefits of ReDence’s hybrid architecture, where the specialized Decider model accelerates decision-making and substantially reduces investigation time, while enabling ReDence to outperform both its larger Mentor model and state-of-the-art scam detectors. Furthermore, its agentic Investigator produces evidencegrounded reports that provide analysts with additional findings and contextual information they may have overlooked during manual investigations, supporting more informed and verifiable assessments of website legitimacy.

## References

[1] M. Bitaab, H. Cho, A. Oest, Z. Lyu, W. Wang, J. Abraham, R. Wang, T. Bao, Y. Shoshitaishvili, and A. Doupe, “Beyond phish: Toward´ detecting fraudulent e-commerce websites at scale,” in 2023 ieee symposium on security and privacy (sp). IEEE, 2023, pp. 2566– 2583.

[2] C. Onwubiko, “Fraud matrix: A morphological and analysis-based classification and taxonomy of fraud,” Computers & Security, vol. 96, p. 101900, 2020.

[3] A. Stabek, P. Watters, and R. Layton, “The seven scam types: Mapping the terrain of cybercrime,” in 2010 second cybercrime and trustworthy computing workshop. IEEE, 2010, pp. 41–51.

[4] P. H. Escayola, L. B. Montoya, and C. Vera, “A typology of digital scams: A framework for detection, classification, and prevention,” 2025.

[5] R. Liu, Y. Lin, Y. Zhang, P. H. Lee, and J. S. Dong, “Knowledge expansion and counterfactual interaction for {Reference-Based} phishing detection,” in 32nd USENIX Security Symposium (USENIX Security 23), 2023, pp. 4139–4156.

[6] S. Abdelnabi, K. Krombholz, and M. Fritz, “Visualphishnet: Zeroday phishing website detection by visual similarity,” in Proceedings of the 2020 ACM SIGSAC conference on computer and communications security, 2020, pp. 1681–1698.

[7] S. S. Roy, C. Torres, and S. Nilizadeh, “” explain, don’t just warn!”–a real-time framework for generating phishing warnings with contextual cues,” arXiv preprint arXiv:2505.06836, 2025.

[8] R. Liu, Y. Lin, X. Yang, S. H. Ng, D. M. Divakaran, and J. S. Dong, “Inferring phishing intention via webpage appearance and dynamics: A deep vision based approach,” in 31st USENIX Security Symposium (USENIX Security 22), 2022, pp. 1633–1650.

[9] Y. Lin, R. Liu, D. M. Divakaran, J. Y. Ng, Q. Z. Chan, Y. Lu, Y. Si, F. Zhang, and J. S. Dong, “Phishpedia: A hybrid deep learning based approach to visually identify phishing webpages,” in 30th USENIX Security Symposium (USENIX Security 21), 2021, pp. 3793–3810.

[10] I. C. C. Center, “IC3 Annual Report,” Federal Bureau of Investigation, Tech. Rep., 2025. [Online]. Available: https: //www.ic3.gov/AnnualReport/Reports/2025 IC3Report.pdf

[11] Federal Trade Commission, “Student loan debt relief scam operators agree to be permanently banned from industry, turn over assets to resolve ftc charges,” https://www.ftc.gov/news-events/news/press-r eleases/2025/05/student-loan-debt-relief-scam-operators-agree-b e-permanently-banned-industry-turn-over-assets, May 2025.

[12] ——, “Complaint for permanent injunction, monetary judgment, and other relief,” Jul. 2024, federal Trade Commission v. Start Connecting LLC et al., Case No. 8:24-cv-01626-KKM-AAS, U.S. District Court for the Middle District of Florida. [Online]. Available: https://www.ftc.gov/system/files/ftc gov/pdf/usasdr complaint.pdf

[13] M. Bitaab, A. Karimi, Z. Lyu, A. Oest, D. Kuchhal, M. Saad, G.-J. Ahn, R. Wang, T. Bao, Y. Shoshitaishvili et al., “Scammagnifier: Piercing the veil of fraudulent shopping website campaigns,” 2025.

[14] X. Li, A. Yepuri, and N. Nikiforakis, “Double and nothing: Understanding and detecting cryptocurrency giveaway scams,” in Proceedings ofthe Network and Distributed System Security Symposium (NDSS), 2023.

[15] P. Paudel and G. Stringhini, “Loki: Proactively discovering online scam websites by mining toxic search queries,” arXiv preprint arXiv:2509.12181, 2025.

[16] M. Bitaab, A. Karimi, Z. Lyu, A. Mosallanezhad, A. Oest, R. Wang, T. Bao, Y. Shoshitaishvili, and A. Doupe, “Scamnet: Toward ex-´ plainable large language model-based fraudulent shopping website detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 27, 2025, pp. 27 841–27 848.

[17] A. Malach, P. N. Wudali, S. Momiyama, J. Furukawa, T. Araki, Y. Elovici, and A. Shabtai, “Cybershapley: Explanation, prioritization, and triage of cybersecurity alerts using informative graph representation,” Computers & Security, vol. 150, p. 104270, 2025.

[18] A. Oest, Y. Safaei, P. Zhang, B. Wardman, K. Tyers, Y. Shoshitaishvili, and A. Doupe, “ ´ {PhishTime}: Continuous longitudinal measurement of the effectiveness of anti-phishing blacklists,” in 29th USENIX Security Symposium (USENIX Security 20), 2020, pp. 379– 396.

[19] A. P. Moritz and A. Kennedy, “Operationalizing critical data approaches in fraud detection in latin america,” in Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, vol. 8, no. 2, 2025, pp. 1738–1747.

[20] H. Nakano, T. Koide, and D. Chiba, “Scamferret: Detecting scam websites autonomously with large language models,” in International Conference on Detection of Intrusions and Malware, and Vulnerability Assessment. Springer, 2025, pp. 3–25.

[21] G. Deng, Y. Liu, V. Mayoral-Vilches, P. Liu, Y. Li, Y. Xu, T. Zhang, Y. Liu, M. Pinzger, and S. Rass, “{PentestGPT}: Evaluating and harnessing large language models for automated penetration testing,” in 33rd USENIX Security Symposium (USENIX Security 24), 2024, pp. 847–864.

[22] A. Ye, Q. Ma, J. Chen, M. Li, T. Li, F. Liu, S. Mai, M. Lu, H. Bao, and Y. You, “Sop-agent: Empower general purpose ai agent with domain-specific sops,” arXiv preprint arXiv:2501.09316, 2025.

[23] X. F. Zhang, A. Mohananey, A. Chronopoulou, P. Papalampidi, S. Gupta, T. Munkhdalai, L. Wang, and S. Upadhyay, “Do llms really need 10+ thoughts for “find the time 1000 days later”? towards structural understanding of llm overthinking,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 17 005–17 030.

[24] S. Ghoshal and A. Al-Bustami, “When do tools and planning help large language models think? a cost-and latency-aware benchmark,” in SoutheastCon 2026. IEEE, 2026, pp. 01–08.

[25] F. Trad and A. Chehab, “Toward accurate and cost-effective llm agents via information flow optimization: Insights from a phishing detection case study,” Information Processing & Management, vol. 63, no. 7, p. 104834, 2026.

[26] Qwen Team, “Qwen3.5: Towards native multimodal agents,” February 2026. [Online]. Available: https://qwen.ai/blog?id=qwen3. 5

[27] Together AI, “Tev1 4B Experimental: A fast, cheap decision model: structured options in, one answer out,” September 2026. [Online]. Available: https://www.together.ai/models/tev1-4b-experimental

[28] L. Madaan, A. Didolkar, S. Gururangan, J. Quan, R. Silva, R. Salakhutdinov, M. Zaheer, S. Arora, and A. Goyal, “Rethinking thinking tokens: Llms as improvement operators,” arXiv preprint arXiv:2510.01123, 2025.

[29] H. El Mghari, “How to train your own Jev for \$17,” Together AI Blog, Sep. 2026. [Online]. Available: https: //www.together.ai/blog/how-to-train-your-own-jev

[30] Ollama, “Ollama now supports Jev-style decision models,” Ollama Blog, Sep. 2026. [Online]. Available: https://ollama.com/blog/olla ma-now-supports-jev-style-decision-models

[31] T. Deußer, L. Sparrenberg, and R. Sifa, “Evaluating and benchmarking the system one model jev,” arXiv preprint arXiv:2609.37647, 2026.

[32] Y. Jiang, Y. Wang, X. Zeng, W. Zhong, L. Li, F. Mi, L. Shang, X. Jiang, Q. Liu, and W. Wang, “Followbench: A multi-level finegrained constraints following benchmark for large language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 4667–4688.

[33] B. Wen, P. Ke, X. Gu, L. Wu, H. Huang, J. Zhou, W. Li, B. Hu, W. Gao, J. Xu et al., “Benchmarking complex instruction-following with multiple constraints composition, 2024,” URL https://arxiv. org/abs/2407.03978.

[34] G. Dagan, F. Keller, and A. Lascarides, “Plancraft: an evaluation dataset for planning with LLM agents,” in Second Conference on Language Modeling, 2025. [Online]. Available: https://openreview .net/forum?id=nSV8Depcpx

[35] W. Shen, C. Li, H. Chen, M. Yan, X. Quan, H. Chen, J. Zhang, and F. Huang, “Small llms are weak tool learners: A multi-llm agent, 2024,” URL https://arxiv. org/abs/2401.07324, 2024.

[36] J. Lee, W. Song, J. Han, H. Pyun, and Y. Jo, “Don’t adapt small language models for tools; adapt tool schemas to the models,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 20 695–20 719.

[37] C. Xia, C. Xing, J. Du, X. Yang, Y. Feng, R. Xu, W. Yin, and C. Xiong, “Fofo: A benchmark to evaluate llms’ format-following capability,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 680–699.

[38] C.-Y. Hsieh, C.-L. Li, C.-K. Yeh, H. Nakhost, Y. Fujii, A. Ratner, R. Krishna, C.-Y. Lee, and T. Pfister, “Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes,” in Findings of the association for computational linguistics: ACL 2023, 2023, pp. 8003–8017.

[39] B. Chen, C. Shu, E. Shareghi, N. Collier, K. Narasimhan, and S. Yao, “Fireact: Toward language agent fine-tuning,” arXiv preprint arXiv:2310.05915, 2023.

[40] M. S. I. Mamun, M. A. Rathore, A. H. Lashkari, N. Stakhanova, and A. A. Ghorbani, “Detecting malicious urls using lexical analysis,” in International conference on network and system security. Springer, 2016, pp. 467–482.

[41] A. Prasad and S. Chandra, “Phiusiil: A diverse security profile empowered phishing url detection framework based on similarity index and incremental learning,” Computers & Security, vol. 136, p. 103545, 2024.

[42] B. B. Bureau, “Search for Scams — BBB Scam Tracker — Better Business Bureau.” [Online]. Available: https://www.bbb.org/scamtr acker/lookupscam

[43] S. Ma, T. Ma, J. Liu, W. Song, Z. Liang, X. Xiao, and Y. Ye, “Psyscam: A benchmark for psychological techniques in real-world scams,” arXiv preprint arXiv:2505.15017, 2025.

[44] C. A. Maher, T. A. Engle, M. McGrath, and J. L. Nedelec, “Fraud in the buckeye state: A geospatial and temporal analysis fraud victimization reports in ohio, 2024,” Journal of Criminal Justice, vol. 101, p. 102535, 2025.

[45] G. Gomez, K. Van Liebergen, D. Sanvito, G. Siracusano, R. Gonzalez, and J. Caballero, “Clean up the mess: Addressing data pollution in cryptocurrency abuse reporting services,” Future Generation Computer Systems, p. 108313, 2025.

[46] K. Lee, K. Lim, H. Kim, Y. Kwon, and D. Kim, “7 days later: analyzing phishing-site lifespan after detected,” in Proceedings of the ACM on Web Conference 2025, 2025, pp. 945–956.

[47] T. Moore and R. Clayton, “Examining the impact of website takedown on phishing,” in Proceedings of the anti-phishing working groups 2nd annual eCrime researchers summit, 2007, pp. 1–13.

[48] T. Barron, N. Miramirkhani, and N. Nikiforakis, “Now you see it, now you {Don’t}: A large-scale analysis of early domain deletions,” in 22nd International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2019), 2019, pp. 383–397.

[49] Webrecorder, “GitHub - webrecorder/browsertrix: Browsertrix is the hosted, high-fidelity, browser-based crawling service from Webrecorder designed to make web archiving easier and more accessible for all!” [Online]. Available: https://github.com/webreco rder/browsertrix

[50] “Browsertrix — SUCHO.” [Online]. Available: https://www.sucho. org/browsertrix

[51] “Software and Processes - University Libraries - UNT.” [Online]. Available: https://library.unt.edu/digital-projects-unit/web-archiving /software-processes

[52] “Web Archive Collection Zipped (WACZ).” [Online]. Available: https://specs.webrecorder.net/wacz/1.1.1/

[53] J. F. Brunelle, M. Kelly, H. SalahEldeen, M. C. Weigle, and M. L. Nelson, “Not all mementos are created equal: Measuring the impact of missing resources,” International Journal on Digital Libraries, vol. 16, no. 3, pp. 283–301, 2015.

[54] J. Zhou, T. Lu, S. Mishra, S. Brahma, S. Basu, Y. Luan, D. Zhou, and L. Hou, “Instruction-following evaluation for large language models,” arXiv preprint arXiv:2311.07911, 2023.

[55] V. Pyatkin, S. Malik, V. Graf, H. Ivison, S. Huang, P. Dasigi, N. Lambert, and H. Hajishirzi, “Generalizing verifiable instruction following,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[56] A. Team, “Humanity’s last exam with tools,” 2025. [Online]. Available: https://github.com/activeloopai/hle

[57] R. Wong, J. Wang, L. Chen, Y. Gao, X. Zhou, Z. Wang, K. Xiang, G. Zhang, W. Huang, Y. Wang et al., “Widesearch: Benchmarking agentic broad info-seeking,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 10 012–10 086.

[58] F. Yang, P. Zhao, Z. Wang, L. Wang, B. Qiao, J. Zhang, M. Garg, Q. Lin, S. Rajmohan, and D. Zhang, “Empower large language model to perform better on industrial domain-specific question answering,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track, 2023, pp. 294–312.

[59] Z. Fei, X. Shen, D. Zhu, F. Zhou, Z. Han, A. Huang, S. Zhang, K. Chen, Z. Yin, Z. Shen et al., “Lawbench: Benchmarking legal knowledge of large language models,” in Proceedings of the 2024 conference on empirical methods in natural language processing, 2024, pp. 7933–7962.

[60] L. Y. Jiang, A. Chen, X. Han, X. C. Liu, R. Dua, K. Eaton, F. Wolff, R. Steele, J. Zhang, A. Alyakin et al., “Generalist foundation models are not clinical enough for hospital operations,” Research Square, pp. rs–3, 2026.

[61] H. Nori, Y. T. Lee, S. Zhang, D. Carignan, R. Edgar, N. Fusi, N. King, J. Larson, Y. Li, W. Liu et al., “Can generalist foundation models outcompete special-purpose tuning? case study in medicine,” arXiv preprint arXiv:2311.16452, 2023.

[62] B. Xu, A. Yang, J. Lin, Q. Wang, C. Zhou, Y. Zhang, and Z. Mao, “Expertprompting: Instructing large language models to be distinguished experts,” arXiv preprint arXiv:2305.14688, 2023.

[63] K. Chen, Y. Zhou, X. Zhang, and H. Wang, “Prompt stability matters: Evaluating and optimizing auto-generated prompt in generalpurpose systems,” arXiv preprint arXiv:2505.13546, 2025.

[64] E. Habba, N. Dahan, G. Lior, and G. Stanovsky, “Promptsuite: A task-agnostic framework for multi-prompt generation,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2025, pp. 254–263.

[65] X. Wang, C. Li, Z. Wang, F. Bai, H. Luo, J. Zhang, N. Jojic, E. Xing, and Z. Hu, “Promptagent: Strategic planning with language models enables expert-level prompt optimization,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 23 967– 24 001.

[66] Y.-Y. Liu, Z. Zheng, F. Zhang, J.-C. Feng, Y.-Y. Fu, J.-D. Zhai, B.-S. He, X. Zhang, and X.-Y. Du, “A comprehensive taxonomy of prompt engineering techniques for large language models,” Frontiers of Computer Science, vol. 20, no. 3, p. 2003601, 2026.

[67] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., “Chain-of-thought prompting elicits reasoning in large language models,” Advances in neural information processing systems, vol. 35, pp. 24 824–24 837, 2022.

[68] T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa, “Large language models are zero-shot reasoners,” Advances in neural information processing systems, vol. 35, pp. 22 199–22 213, 2022.

[69] B. S. R. Toolkit, “Scam type glossary - BBB Scam Survival Toolkit,” 11 2025. [Online]. Available: https://scamsurvivaltoolkit. bbbmarketplacetrust.org/scam-type-glossary/

[70] Y. Peskine, D. Korenciˇ c, I. Grubi ´ siˇ c, P. Papotti, R. Troncy, and´ P. Rosso, “Definitions matter: Guiding gpt for multi-label classification,” in Findings of the Association for Computational Linguistics: EMNLP 2023, 2023, pp. 4054–4063.

[71] Y. Mu, B. Wu, W. Thorne, A. Robinson, N. Aletras, C. Scarton, K. Bontcheva, and X. Song, “Navigating prompt complexity for zero-shot classification: A study of large language models in computational social science. arxiv 2024,” arXiv preprint arXiv:2305.14310.

[72] X. Teoh, Y. Lin, R. Liu, Z. Huang, and J. S. Dong, “{PhishDecloaker}: Detecting {CAPTCHA-cloaked} phishing websites via hybrid vision-based interactive models,” in 33rd USENIX Security Symposium (USENIX Security 24), 2024, pp. 505–522.

[73] K. Wafula and Y. Wang, “Carve: A scientific method-based threat hunting hypothesis development model,” in 2019 IEEE International Conference on Electro Information Technology (EIT). IEEE, 2019, pp. 1–6.

[74] F. Araujo, D. Kirat, X. Shu, T. Taylor, and J. Jang, “Evidential cyber threat hunting,” arXiv preprint arXiv:2104.10319, 2021.

[75] C. Tsigkanos, L. Pasquale, C. Menghi, C. Ghezzi, and B. Nuseibeh, “Engineering topology aware adaptive security: Preventing requirements violations at runtime,” in 2014 IEEE 22nd International Requirements Engineering Conference (RE). IEEE, 2014, pp. 203– 212.

[76] L. Pasquale, S. Hanvey, M. Mcgloin, and B. Nuseibeh, “Adaptive evidence collection in the cloud using attack scenarios,” Computers & Security, vol. 59, pp. 236–254, 2016.

[77] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “React: Synergizing reasoning and acting in language models,” arXiv preprint arXiv:2210.03629, 2022.

[78] C. Niu, Y. Wu, J. Zhu, S. Xu, K. Shum, R. Zhong, J. Song, and T. Zhang, “Ragtruth: A hallucination corpus for developing trustworthy retrieval-augmented language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 10 862–10 878.

[79] S. Min, K. Krishna, X. Lyu, M. Lewis, W.-t. Yih, P. Koh, M. Iyyer, L. Zettlemoyer, and H. Hajishirzi, “Factscore: Fine-grained atomic evaluation of factual precision in long form text generation,” in Proceedings ofthe 2023 conference on empirical methods in natural language processing, 2023, pp. 12 076–12 100.

[80] S. J. Mielke, A. Szlam, E. Dinan, and Y.-L. Boureau, “Reducing conversational agents’ overconfidence through linguistic calibration,” Transactions of the Association for Computational Linguistics, vol. 10, pp. 857–872, 2022.

[81] T. Groot and M. Valdenegro-Toro, “Overconfidence is key: Verbalized uncertainty evaluation in large language and vision-language models,” in Proceedings ofthe 4th Workshop on Trustworthy Natural Language Processing (TrustNLP 2024), 2024, pp. 145–171.

[82] Microsoft, “Fast and reliable end-to-end testing for modern web apps.” [Online]. Available: https://playwright.dev/

[83] J. Yang, H. Zhang, F. Li, X. Zou, C. Li, and J. Gao, “Set-ofmark prompting unleashes extraordinary visual grounding in gpt-4v,” arXiv preprint arXiv:2310.11441, 2023.

[84] Firecrawl, “Supercharge your AI agents with data from the web and beyond.” [Online]. Available: https://www.firecrawl.dev/glossar y/web-extraction-apis/what-are-css-selectors-xpath-web-extraction

[85] Thicccat, “GitHub - thicccat688/selenium-recaptcha-solver: Re-CAPTCHA solver for selenium (Using audio).” [Online]. Available: https://github.com/thicccat688/selenium-recaptcha-solver

[86] Deedy, “GitHub - deedy5/ddgs: A metasearch library that aggregates results from diverse web search services.” [Online]. Available: https://github.com/deedy5/ddgs

[87] X. Chen, J. Xu, T. Liang, Z. He, J. Pang, D. Yu, L. Song, Q. Liu, M. Zhou, Z. Zhang et al., “Do not think that much for 2+ 3=? on the overthinking of long reasoning models,” in Forty-second International Conference on Machine Learning, 2025.

[88] T. Chen, P. Singh, K. Todi, B. Paria, Q. He, W. Wang, Y. Tan, J. Zhong, S. Basu, D. Ghadiyaram et al., “Deloop: Repeated actions as self-supervision for long-horizon agents,” arXiv preprint arXiv:2610.19376, 2026.

[89] Financial Conduct Authority, “How to check a firm or individual is authorised,” 2026. [Online]. Available: https://www.fca.org.uk/c onsumers/how-check-firm-individual-authorised

[90] HM Treasury, “The money laundering, terrorist financing and transfer of funds (information on the payer) regulations 2017,” UK Statutory Instrument 2017 No. 692, 2017, regulations 28 and 30. [Online]. Available: https://www.legislation.gov.uk/uksi/2017/692

[106] “Let’s Encrypt and Comodo issue thousands of certificates for phishing,” 4 2017. [Online]. Available: https://www.netcraft.com/b log/lets-encrypt-and-comodo-issue-thousands-of-certificates-for-p hishing

[91] Companies House, “Companies House,” https://www.gov.uk/gover nment/organisations/companies-house, 2026.

[92] D. Kramer, L. Rosique, A. Narotam, E. Bursztein, P. G. Kelley, K. Thomas, and A. Woodruff, “Integrating large language models into security incident response,” in Twenty-First Symposium on Usable Privacy and Security (SOUPS 2025), 2025, pp. 133–148.

[93] H. Rashkin, V. Nikolaev, M. Lamm, L. Aroyo, M. Collins, D. Das, S. Petrov, G. S. Tomar, I. Turc, and D. Reitter, “Measuring attribution in natural language generation models,” Computational Linguistics, vol. 49, no. 4, pp. 777–840, 2023.

[94] T. Gao, H. Yen, J. Yu, and D. Chen, “Enabling large language models to generate text with citations,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 6465–6488.

[95] A. Nenkova and R. J. Passonneau, “Evaluating content selection in summarization: The pyramid method,” in Proceedings of the human language technology conference of the north american chapter of the association for computational linguistics: Hlt-naacl 2004, 2004, pp. 145–152.

[96] R. Holmes, “Decision models explained - the new AI building block for product teams,” 10 2026. [Online]. Available: https://departme ntofproduct.substack.com/p/decision-models-explained-the-new

[97] Y. Li, Y. Emad, K. Padthe, J. Lanchantin, W. Yuan, T. Nguyen, J. Weston, S.-W. Li, D. Wang, I. Kulikov et al., “Naturalthoughts: Selecting and distilling reasoning traces for general reasoning tasks,” arXiv preprint arXiv:2507.01921, 2025.

[98] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi et al., “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025.

[99] W. U. Ahmad, S. Narenthiran, S. Majumdar, A. Ficek, S. Jain, J. Huang, V. Noroozi, and B. Ginsburg, “Opencodereasoning: Advancing data distillation for competitive coding,” arXiv preprint arXiv:2504.01943, 2025.

[100] R. Nakano, J. Hilton, S. Balaji, J. Wu, L. Ouyang, C. Kim, C. Hesse, S. Jain, V. Kosaraju, W. Saunders et al., “Webgpt: Browser-assisted question-answering with human feedback,” arXiv preprint arXiv:2112.09332, 2021.

[101] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “Qlora: Efficient finetuning of quantized llms,” Advances in neural information processing systems, vol. 36, pp. 10 088–10 115, 2023.

[102] S. Mangrulkar, S. Gugger, L. Debut, Y. Belkada, S. Paul, and B. Bossan, “Peft: State-of-the-art parameter-efficient fine-tuning methods,” https://github.com/huggingface/peft, 2022.

[103] H. Le, Q. Pham, D. Sahoo, and S. C. Hoi, “Urlnet: Learning a url representation with deep learning for malicious url detection,” arXiv preprint arXiv:1802.03162, 2018.

[104] A. Safi and S. Singh, “A systematic literature review on phishing website detection techniques,” Journal of King Saud University Computer and Information Sciences, vol. 35, no. 2, pp. 590–611, 2023.

[105] Z. Zhang, Y. Yang, N. Lu, W. Shi, and Z. Liu, “Anti-aphish: A robust and adaptive detection approach against evolving ai-powered phishing urls,” Expert Systems with Applications, p. 133324, 2026.