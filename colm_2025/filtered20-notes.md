Hypothesis-Driven Theory-of-Mind Reasoning for Large Language Models
Hyunwoo Kim, Melanie Sclar, Tan Zhi-Xuan, Lance Ying, Sydney Levine, Yang Liu, Joshua B. Tenenbaum, Yejin Choi
Existing LLM reasoning methods have shown impressive capabilities across various tasks, such as solving math and coding problems. However, applying these methods to scenarios without ground-truth answers or rule-based verification methods - such as tracking the mental states of an agent - remains challenging. Inspired by the sequential Monte Carlo algorithm, we introduce ThoughtTracing, an inference-time reasoning algorithm designed to trace the mental states of specific agents by generating hypotheses and weighting them based on observations without relying on ground-truth solutions to questions in datasets. Our algorithm is modeled after the Bayesian theory-of-mind framework, using LLMs to approximate probabilistic inference over agents' evolving mental states based on their perceptions and actions. We evaluate ThoughtTracing on diverse theory-of-mind benchmarks, demonstrating significant performance improvements compared to baseline LLMs. Our experiments also reveal interesting behaviors of the recent reasoning models - e.g., o3 and R1 - on theory-of-mind, highlighting the difference of social reasoning compared to other domains.
https://openreview.net/forum?id=yGQqTuSJPK

- could this be a way to sample correct research processes? have a model iteratively refine it's own generation until we have something good?
- Edit: I don't actually care about this paper. Baysian cognition is not the right path forward for LLMs IMO

Pairwise or Pointwise? Evaluating Feedback Protocols for Bias in LLM-Based Evaluation
Tuhina Tripathi, Manya Wadhwa, Greg Durrett, Scott Niekum
Large Language Models (LLMs) are widely used as proxies for human labelers in both training (Reinforcement Learning from AI Feedback) and large-scale response evaluation (LLM-as-a-judge). Alignment and evaluation are critical components in the development of reliable LLMs, and the choice of feedback protocol plays a central role in both but remains understudied. In this work, we show that the choice of feedback protocol for evaluation (absolute scores versus relative preferences) can significantly affect evaluation reliability and induce systematic biases. In the context of LLM-as-a-judge evaluation, we show that pairwise protocols are more vulnerable to **distracted evaluation**. Generator models can exploit spurious attributes (or distractor features) favored by the LLM judge, resulting in inflated scores for lower-quality outputs. We find that absolute scoring is more robust to such manipulation, producing judgments that better reflect response quality and are less influenced by distractor features. Our results demonstrate that generator models can flip preferences by embedding distractor features, skewing LLM-as-a-judge comparisons and leading to inaccurate conclusions about model quality in benchmark evaluations. **Pairwise preferences flip in about 35\% of the cases, compared to only 9\% for absolute scores**. We offer recommendations for choosing feedback protocols based on dataset characteristics and evaluation objectives.
https://openreview.net/forum?id=uyX5Vnow3U

- Need to talk to *Tuhina Tripathi* about this
    - Does she think AlpacaEval sucks?
    - How do we operationalize this into a reward signal?
    - Are there *actually* non-verifiable chat tasks that we care about?
    - Span-based eval? Rubrics?
    - What do you think about group-relative ranking? E.g. generate k samples from two models and rank all 2*k samples

MLGym: A New Framework and Benchmark for Advancing AI Research Agents
Deepak Nathani, Lovish Madaan, Nicholas Roberts, Nikolay Bashlykov, Ajay Menon, Vincent Moens, Mikhail Plekhanov, Amar Budhiraja, Despoina Magka, Vladislav Vorotilov, Gaurav Chaurasia, Dieuwke Hupkes, Ricardo Silveira Cabral, Tatiana Shavrina, Jakob Nicolaus Foerster, Yoram Bachrach, William Yang Wang, Roberta Raileanu
We introduce MLGym and MLGym-Bench, a new framework and benchmark for evaluating and developing LLM agents on AI research tasks. This is the first Gym environment for machine learning (ML) tasks, enabling research on reinforcement learning (RL) algorithms for training such agents. MLGym-bench consists of 13 diverse and open-ended AI research tasks from diverse domains such as computer vision, natural language processing, reinforcement learning, and game theory. Solving these tasks requires real-world AI research skills such as generating new ideas and hypotheses, creating and processing data, implementing ML methods, training models, running experiments, analyzing the results, and iterating through this process to improve on a given task. We evaluate a number of frontier large language models (LLMs) on our benchmarks such as Claude-3.5-Sonnet, Llama-3.1 405B, GPT-4o, o1-preview, and Gemini-1.5 Pro. Our MLGym framework makes it easy to add new tasks, integrate and evaluate models or agents, generate synthetic data at scale, as well as develop new learning algorithms for training agents on AI research tasks. We find that current frontier models can improve on the given baselines, usually by finding better hyperparameters, but do not generate novel hypotheses, algorithms, architectures, or substantial improvements. We open-source our framework and benchmark to facilitate future research in advancing the AI research capabilities of LLM agents.
https://openreview.net/forum?id=ryTr83DxRq

- Mainly just had authors re-implement research they were familar with. Similar to MLE-Bench but with research topics one would produce in a class project.
- Need to talk to **Deepak Nathani** about this.
    - How do we build research agents to hill-climb this?
    - Are new algorithms needed for this?

AutoScale: Scale-Aware Data Mixing for Pre-Training LLMs
Feiyang Kang, Yifan Sun, Bingbing Wen, Si Chen, Dawn Song, Rafid Mahmood, Ruoxi Jia
Domain reweighting is an emerging research area aimed at adjusting the relative weights of different data sources to improve the effectiveness and efficiency of LLM pre-training. We show that data mixtures that perform well at smaller scales may not retain their advantage at larger scales, challenging the existing practice of determining competitive mixtures in small-scale experiments and *directly* applying them at much larger scales. To address this, we propose AutoScale, a two-stage, scale-aware data composition framework. First, AutoScale fits a parametric model that predicts the model’s loss under different data compositions, then uses it to find an approximate best allocation at smaller, more manageable budgets. Next, leveraging a novel theoretical analysis of how optimal compositions evolve with scale, AutoScale extrapolates that composition to larger budgets without further retraining. Empirically, AutoScale accelerates convergence and improves downstream performance.
For instance, when pre-training GPT-2 Large, it achieves a 28\% faster perplexity reduction than baselines and up to a 38\% speed-up over unweighted training, while yielding best-average results on various downstream tasks. Overall, our findings illustrate how domain importance shifts with training scale, underscoring the need for scale-dependent data curation in LLM training. 
Our code is open-sourced.
https://openreview.net/forum?id=rujwIvjooA

- **read this** -- They found "crossovers" in data mixes!!!
- Need to talk to **Feiyang Kang** about this

LongProc: Benchmarking Long-Context Language Models on Long Procedural Generation
Xi Ye, Fangcong Yin, Yinghui He, Joie Zhang, Howard Yen, Tianyu Gao, Greg Durrett, Danqi Chen
Existing benchmarks for evaluating long-context language models (LCLMs) primarily focus on long-context recall, requiring models to produce short responses based on a few critical snippets while processing thousands of irrelevant tokens.
We introduce LongProc (Long Procedural Generation), a new benchmark that requires both the integration of highly dispersed information and long-form generation. LongProc consists of six diverse procedural generation tasks, such as extracting structured information from HTML pages into a TSV format and executing complex search procedures to create travel plans. 
These tasks challenge LCLMs by testing their ability to follow detailed procedural instructions, synthesize and reason over dispersed information, and generate structured, long-form outputs (up to 8K tokens). 
Furthermore, as these tasks adhere to deterministic procedures and yield structured outputs, they enable reliable rule-based evaluation. 
We evaluated 23 LCLMs, including instruction-tuned models and recent reasoning models, on LongProc at three difficulty levels, with the maximum number of output tokens set at 500, 2K, and 8K. 
Notably, while all tested models claim a context window size above 32K tokens, open-weight models typically falter on 2K-token tasks, and closed-source models like GPT-4o show significant degradation on 8K-token tasks.
Reasoning models achieve stronger overall performance in long-form generation, benefiting from long CoT training.
Further analysis reveals that LCLMs struggle to maintain long-range coherence in long-form generations.
These findings highlight critical limitations in current LCLMs and suggest substantial room for improvement.
https://openreview.net/forum?id=ruWC5LIMSo

- Why do models suck so much? What are the open challenges in LC evals?
- Thoughts on DeepSeek LC? On linear attention?
- Is this a data problem or an architecture problem?

LLMs as Research Tools: A Large Scale Survey of Researchers’ Usage and Perceptions
Zhehui Liao, Maria Antoniak, Inyoung Cheong, Evie Yu-Yen Cheng, Ai-Heng Lee, Kyle Lo, Joseph Chee Chang, Amy X Zhang
The rise of large language models (LLMs) has led many researchers to consider their usage for scientific work. Some have found benefits using LLMs to augment or automate aspects of their research pipeline, while others have urged caution due to risks and ethical concerns. Yet little work has sought to quantify and characterize how researchers actually use LLMs and why or why not. We present the first large-scale survey of 816 verified research article authors to understand how the research community leverages and perceives LLMs as research tools. We examine participants' self-reported LLM usage, finding that 81% of researchers have already incorporated LLMs into aspects of their research workflow. We also find that some traditionally disadvantaged groups in academia (non-white, junior, and non-native English speaking researchers) report higher LLM usage and perceived benefits, suggesting potential for improved research equity. However, women, non-binary, and senior researchers have greater ethical concerns. Our study provides much-needed evidence, rather than speculation, about how LLMs are currently being used as research tools.
https://openreview.net/forum?id=p0BwJk3R1p

- How LLMs are currently used: Lots of writing help, some experimentation
- What's the performance bottleneck? Should we build LLMs to automate experiments first? Literature reviews? Building hypotheses? Where's the gap that people want?

Synthetic Data Generation and Multi-Step Reinforcement Learning for Reasoning and Tool Use
Anna Goldie, Azalia Mirhoseini, Hao Zhou, Irene Cai, Christopher D Manning
Reinforcement learning has been shown to improve the performance of large language models. However, traditional approaches like RLHF or RLAIF treat the problem as single-step. As focus is shifting towards solving more complex reasoning and agentic tasks, language models must take multiple steps of text generation, reasoning and environment interaction before generating a solution. We propose a synthetic data generation and RL methodology targeting multi-step optimization scenarios. This approach, called Step-Wise Reinforcement Learning (SWiRL), iteratively generates multi-step reasoning and tool use data, and then learns from that data. It employs a simple step-wise decomposition that breaks each multi-step trajectory into multiple sub-trajectories corresponding to each action by the original model. It then applies synthetic data filtering and RL optimization on these sub-trajectories. We evaluated SWiRL on a number of multi-step tool use, question answering, and mathematical reasoning tasks. Our experiments show that SWiRL outperforms baseline approaches by 21.5\%, 12.3\%, 14.8\%, 11.1\%, and 15.3\% in relative accuracy on GSM8k, HotPotQA, CofCA, MuSiQue, and BeerQA, respectively. Excitingly, the approach exhibits generalization across tasks: for example, training only on HotPotQA (text question-answering) improves zero-shot performance on GSM8k (a math dataset) by 16.9\%.
https://openreview.net/forum?id=oN9STRYQVa

- **read this**
- Can I use this out-of-the-box for a research agent?

AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories
Xing Han Lù, Amirhossein Kazemnejad, Nicholas Meade, Arkil Patel, Dongchan Shin, Alejandra Zambrano, Karolina Stanczak, Peter Shaw, Christopher Pal, Siva Reddy
Web agents enable users to perform tasks on web browsers through natural language interaction. Evaluating web agents trajectories is an important problem, since it helps us determine whether the agent successfully completed the tasks. Rule-based methods are widely used for this purpose, but they are challenging to extend to new tasks and may not always recognize successful trajectories. We may achieve higher accuracy through human evaluation, but the process would be substantially slower and more expensive. Automatic evaluations with LLMs may avoid the challenges of designing new rules and manually annotating trajectories, enabling faster and cost-effective evaluation. However, it is unclear how effective they are at evaluating web agents. To this end, we propose AgentRewardBench, the first benchmark to assess the effectiveness of LLM judges for evaluating web agents. AgentRewardBench contains 1302 trajectories across 5 benchmarks and 4 LLMs. Each trajectory in AgentRewardBench is reviewed by an expert, who answers questions pertaining to the success, side effects, and repetitiveness of the agent. Using our benchmark, we evaluate 12 LLM judges and find that no single LLM excels across all benchmarks. We also find that the rule-based evaluation used by common benchmarks tends to underreport the success rate of web agents, highlighting a key weakness of rule-based evaluation and the need to develop more flexible automatic evaluations. We release the benchmark at: https://agent-reward-bench.github.io
https://openreview.net/forum?id=fQcUZMPIvu

- Why do we need this? Is it to prevent cheating? Why not just use some rule-based correctness reward?
- Can we RLVR on it?

EvalAgents: Discovering Implicit Evaluation Criteria from the Web
Manya Wadhwa, Zayne Rea Sprague, Chaitanya Malaviya, Philippe Laban, Junyi Jessy Li, Greg Durrett
Evaluation of language model outputs on structured writing tasks is typically conducted with a number of desirable criteria presented to human evaluators or large language models (LLMs). For instance, on a prompt like "Help me draft an academic talk on coffee intake vs research productivity", a model response may be evaluated for criteria like accuracy and coherence. However, high-quality responses should do more than just satisfy basic task requirements. An effective response to this query should include quintessential features of an academic talk, such as a compelling opening, clear research questions, and a takeaway. To help identify these implicit criteria, we introduce EvalAgent, a novel framework designed to automatically uncover nuanced and task-specific criteria. EvalAgent first mines expert-authored online guidance. It then uses this evidence to propose diverse, long-tail evaluation criteria that are grounded in reliable external sources. Our experiments demonstrate that the grounded criteria produced by EvalAgent are often implicit (not directly stated in the user's prompt), yet specific (high degree of lexical precision). Further, EvalAgent criteria are often not satisfied by initial responses but they are actionable, such that responses can be refined to satisfy them. Finally, we show that combining LLM-generated and EvalAgent criteria uncovers more human-valued criteria than using LLMs alone.
https://openreview.net/forum?id=erGpkHCybv

- **read this**
- See questions on "Pairwise or Pointwise?". Can we use it for Olmo?
- Can we RLVR on it?
- Talk to **Manya Wadhwa**

Yourbench: Dynamic Evaluation Set Generation with LLMs
Sumuk Shashidhar, Clémentine Fourrier, Alina Lozovskaya, Thomas Wolf, Gokhan Tur, Dilek Hakkani-Tür
Large language models (LLMs) have rapidly outpaced traditional evaluation methodologies, with static benchmarks suffering from saturation, contamination, and domain-specificity limitations while human evaluation remains prohibitively expensive. We present YourBench, an open-source framework that transforms this evaluation paradigm by enabling automated generation of reliable, contamination-free benchmarks directly from user-provided documents without human annotation. To validate our approach, we successfully reproduce the challenging MMLU-Pro benchmark across 86 models spanning 400M to 405B parameters, achieving remarkable Pearson correlations of 0.91-0.99 while generating entirely novel questions for under $15 per model. This demonstrates that dynamically generated evaluations can match the discriminative power of expert-curated benchmarks while eliminating contamination risks. YourBench enables researchers to create domain-specific benchmarks in minutes rather than months. We demonstrate applications in agriculture, personalized education, and RAG training that were previously infeasible. By releasing the YourBench library, Tempora-0325 dataset, 150K+ generated QA pairs, and all evaluation traces, we provide the community with a practical solution to the challenge of keeping pace with rapidly evolving model capabilities.
https://openreview.net/forum?id=bkWERVKzuP

- Very similar to AutoBencher. What's the nosie look like? Why not just eval on PPL?
- Isn't this just a PPL set with extra steps?
- Talk to **Sumuk Shashidhar**

Sharpe Ratio-Guided Active Learning for Preference Optimization in RLHF
Syrine Belakaria, Joshua Kazdan, Charles Marx, Chris Cundy, Willie Neiswanger, Sanmi Koyejo, Barbara E Engelhardt, Stefano Ermon
Reinforcement learning from human feedback (RLHF) has become a cornerstone of the training and alignment pipeline for large language models (LLMs). Recent advances, such as direct preference optimization (DPO), have simplified the preference learning step. However, collecting preference data remains a challenging and costly process, often requiring expert annotation. This cost can be mitigated by carefully selecting the data points presented for annotation. In this work, we propose an active learning approach to efficiently select prompt and preference pairs using a risk assessment strategy based on the Sharpe Ratio. 
To address the challenge of unknown preferences prior to annotation, our method evaluates the gradients of all potential preference annotations to assess their impact on model updates. These gradient-based evaluations enable risk assessment of data points regardless of the annotation outcome. By leveraging the DPO loss derivations, we derive a \emph{closed-form expression} for computing these Sharpe ratios on a per-tuple basis, ensuring our approach remains both \emph{tractable} and \emph{computationally efficient}.  We also introduce two variants of our method, each making different assumptions about prior information. Experimental results demonstrate that our method outperforms the baseline by up to 5\% in win rates against the chosen completion with limited human preference data across several language models and real-world datasets.
https://openreview.net/forum?id=a6xzTqMUFQ

- **read this** (currently don't understand it)

Self-Steering Language Models
Gabriel Grand, Joshua B. Tenenbaum, Vikash Mansinghka, Alexander K. Lew, Jacob Andreas
While test-time reasoning enables language models (LMs) to tackle complex tasks, searching or planning in natural language can be slow, costly, and error-prone. But even when LMs struggle to emulate the precise reasoning steps needed to solve a problem, they often excel at describing its *abstract structure*—both how to verify solutions and *how to search* for them. This paper introduces DisCIPL, a method for “self-steering” LMs where a *Planner model* generates a task-specific *inference program* that is executed by a population of *Follower models*. Our approach equips LMs with the ability to write recursive search procedures that guide LM inference, enabling new forms of verifiable and efficient reasoning. When instantiated with a small Follower (e.g., Llama-3.2-1B or Qwen3-1.7B), DisCIPL matches (and sometimes outperforms) much larger models, including GPT-4o and o1, on challenging constrained generation tasks. Our work opens up a design space of highly-parallelized Monte Carlo inference strategies that outperform standard best-of-N sampling, require no finetuning, and can be implemented automatically by existing LMs.
https://openreview.net/forum?id=XvCBtm5PgF

- **read this** (mabye not in depth though)
- Can we RLVR on this?

Weight ensembling improves reasoning in language models
Xingyu Dang, Christina Baek, Kaiyue Wen, J Zico Kolter, Aditi Raghunathan
We investigate a pitfall during the training of reasoning models where the diversity of generations begins to collapse, leading to suboptimal test-time scaling. Notably, Pass@1 reliably improves during supervised finetuning (SFT), but Pass@k rapidly deteriorates. Surprisingly, a simple intervention of interpolating the weights of the latest SFT checkpoint with an early checkpoint, otherwise known as WiSE-FT, almost completely recovers Pass@k while also improving Pass@1. The WiSE-FT variant achieves better test-time scaling (Best@k, majority vote) and achieves superior results with less data when tuned further by reinforcement learning. Finally, we note that WiSE-FT provides complementary gains across performance metrics that is not achievable by diversity-inducing decoding strategies alone, like temperature scaling. We formalize a \emph{bias-variance tradeoff} of Pass@k with respect to the expectation and variance of Pass@1 over the test distribution. We find that WiSE-FT can reduce bias and variance simultaneously, while temperature scaling and possibly other decoding strategies face an inherent tradeoff between decreasing variance with increasing bias.
https://openreview.net/forum?id=S2IKxulLT1

- Talk to this author!!! **Xingyu Dang**
- Does this *actually work*? Why would this work?
- Does this solve RL's pass@k problem?

LongCodeBench: Evaluating Coding LLMs at 1M Context Windows
Stefano Rando, Luca Romani, Alessio Sampieri, Luca Franco, John Yang, Yuta Kyuragi, Fabio Galasso, Tatsunori Hashimoto
Context lengths for models have grown rapidly, from thousands to millions of tokens in just a few years. 
The extreme context sizes of modern long-context models have made it difficult to construct realistic long-context benchmarks -- not only due to the cost of collecting million-context tasks but also in identifying realistic scenarios that require significant contexts. We identify code comprehension and repair as a natural testbed and challenge task for long-context models and introduce **LongCodeBench** (**LCB**), a benchmark to test LLM coding abilities in long-context scenarios. 
Our benchmark tests both the comprehension and repair capabilities of LCLMs in realistic and important settings by drawing from real-world GitHub issues and constructing QA (**LongCodeQA**) and bug fixing (**LongSWE-Bench**) tasks. We carefully stratify the complexity of our benchmark, enabling us to evaluate models across different scales -- ranging from Qwen2.5 14B Instruct to Google's flagship Gemini model. We find that long-context remains a weakness for all models, with performance drops such as from 29% to 3% for Claude 3.5 Sonnet, or from 70.2% to 40% for Qwen2.5.
https://openreview.net/forum?id=GFPoM8Ylp8

- Where do LC models fail?
- Opportunities for LC models?
- **Don't need to read, but skim**

An Illusion of Progress? Assessing the Current State of Web Agents
Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, Yu Su
As digitalization and cloud technologies evolve, the web is becoming increasingly important in the modern society. Autonomous web agents based on large language models (LLMs) hold a great potential in work automation. It is therefore important to accurately measure and monitor the progression of their capabilities. In this work, we conduct a comprehensive and rigorous assessment of the current state of web agents. Our results depict a very different picture of the competency of current agents, suggesting over-optimism in previously reported results. This gap can be attributed to shortcomings in existing benchmarks. We introduce Online-Mind2Web, an online evaluation benchmark consisting of 300 diverse and realistic tasks spanning 136 websites. It enables us to evaluate web agents under a setting that approximates how real users use these agents. To facilitate more scalable evaluation and development, we also develop a novel LLM-as-a-Judge automatic evaluation method and show that it can achieve around 85\% agreement with human judgment, substantially higher than existing methods. Finally, we present the first comprehensive comparative analysis of current web agents, highlighting both their strengths and limitations to inspire future research.
https://openreview.net/forum?id=6jZi4HSs6o

- **Don't need to read, but skim**
- Need to talk to **Tianci Xue**!

SmolLM2: When Smol Goes Big — Data-Centric Training of a Fully Open Small Language Model
Loubna Ben allal, Anton Lozhkov, Elie Bakouch, Gabriel Martin Blazquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Agustín Piqueres Lajarín, Hynek Kydlíček, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan Son NGUYEN, Ben Burtenshaw, Clémentine Fourrier, Haojun Zhao, Hugo Larcher, Mathieu Morlon, Cyril Zakka, Colin Raffel, Leandro Von Werra, Thomas Wolf
Large language models, while groundbreaking, are computationally expensive and difficult to deploy in resource-constrained settings. To address this challenge, small language models have emerged, but their performance critically depends on the quality and composition of the pretraining datasets—yet many recent models, such as Qwen2.5-1.5B and Llama3.2-1B, remain opaque about their training data, limiting reproducibility and scientific understanding. In this paper, we document and publicly release SmolLM2, a fully transparent state-of-the-art ``small'' (1.7 billion parameter) language model (LM), along with its training datasets and code. To attain strong performance, we overtrain SmolLM2 on 11 trillion tokens of data using a multi-stage training process that mixes web text with specialized math, code, and instruction-following data. We additionally curate and release new specialized datasets (FineMath, Stack-Edu, and SmolTalk) at stages where we found existing datasets to be problematically small or low-quality. To inform our design decisions, we perform both small-scale ablations and a manual refinement process that updates the dataset mixing rates at each stage based on the performance at the previous one. Ultimately, we demonstrate that SmolLM2 outperforms other recent small LMs including Qwen2.5-1.5B, Llama3.2-1B, and Falcon3-1.6B. By releasing our model, datasets, and code, we aim to facilitate future research on LM development as well as applications of small LMs.
https://openreview.net/forum?id=3JiCl2A14H

- How tf would we reproduce SmolLM3 as a baseline?
- *where did the curriculum come from?*
- **read smollm3 again**

BEARCUBS: A benchmark for computer-using web agents
Yixiao Song, Katherine Thai, Chau Minh Pham, Yapei Chang, Mazin Nadaf, Mohit Iyyer
Modern web agents possess computer use abilities that allow them to interact with webpages by sending commands to a virtual keyboard and mouse. While such agents have considerable potential to assist human users with complex tasks, evaluating their capabilities in real-world settings poses a major challenge. To this end, we introduce BEARCUBS, a “smallbut mighty” benchmark of 111 information-seeking questions designed to evaluate a web agent’s ability to search, browse, and identify factual information from the web. Unlike prior web agent benchmarks, solving BEARCUBS requires (1) accessing live web content rather than synthetic or simulated pages, which captures the unpredictability of real-world web interactions; and (2) performing a broad range of multimodal interactions (e.g., video understanding, 3D navigation) that cannot be bypassed via text-based workarounds. Each question in BEARCUBS has a corresponding short, unambiguous answer and a human-validated browsing trajectory, allowing for transparent evaluation of agent performance and strategies. A human study confirms that BEARCUBS questions are solvable but non-trivial (84.7% human accuracy), revealing domain knowledge gaps and overlooked details as common failure points. We find that ChatGPT Agent significantly outperforms other computer-using agents with an overall accuracy of 65.8% (compared to e.g., Operator’s 23.4%), showcasing substantial progress in tasks involving real computer use, such as playing web games and navigating 3D environments. Nevertheless, closing the gap to human performance requires improvements in areas like fine control, complex data filtering, and execution speed. To facilitate future research, BEARCUBS will be updated periodically to replace invalid or contaminated questions, keeping the benchmark fresh for future generations of web agents.
https://openreview.net/forum?id=0JzWiigkUy

- Simlar to BrowseComp but some multimodal stuff
- How do we have a standard index for eval'ing search agents? Google Vertex API? Tavily API?
- Bottleneck is search? reasoning? or multimodal?
- *What does this tell us about research agents?*
- Can you RLVR on this?