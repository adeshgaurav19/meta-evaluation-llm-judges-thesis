# Human rewrite text from methodology onward

This is a prose extraction from the current LaTeX files. Tables and figures are omitted so the text can be edited more easily.


<!-- Source: 03_methodology.tex -->

# Methodology

This chapter sets out the experimental methodology. Section [sec:research-design] covers the research design and ethics. Section [sec:phase1] describes how the Poisoned Evaluation Dataset was built. Section [sec:phase2] specifies the baseline meta-evaluation framework and gives formal definitions of the primary and secondary metrics. Section [sec:phase3] specifies the two pre-filtering modules and the ablation studies behind the design choices. Section [sec:phase4] defines the end-to-end evaluation protocol and additional analyses. Section [sec:implementation] records the implementation and statistical analysis framework.

## Research design

The study is empirical and experimental, organised into four sequential phases. Phase 1 (Section [sec:phase1]) builds the Poisoned Evaluation Dataset by injecting controlled adversarial perturbations into an established multi-hop QA benchmark. Phase 2 (Section [sec:phase2]) characterises three current LLM judges on the resulting dataset to quantify baseline failure rates. Phase 3 (Section [sec:phase3]) develops two pre-filtering modules motivated by the failure modes observed in Phase 2: a passage-level statistical classifier and a triplet-level LLM classifier. Phase 4 (Section [sec:phase4]) measures the end-to-end effect of each filter on judge reliability, runs the justification and multi-judge disagreement analyses, and conducts targeted ablations.

[Figure omitted: Overall experimental pipeline, showing the four phases from dataset construction through baseline judge scoring, pre-filter design, and end-to-end evaluation.]

The order of these phases is intentional; Phase 3 is informed by the empirical failure modes observed in Phase 2 rather than by prior assumptions about how judges fail.

### Ethical considerations

The work involves synthesising adversarial content for experimental purposes inside a sandboxed evaluation environment. All adversarial passages are generated only for controlled offline analysis and are not deployed against any production system. The attack constructions are taken from published literature [zou2025poisonedrag] and are presented to inform defensive research. The Poisoned Evaluation Dataset will be released with documentation and intended-use guidelines. No personal data is involved.

## Phase 1: Dataset construction

### Base benchmark

We use HotpotQA in its distractor setting as the base benchmark. HotpotQA satisfies three properties relevant to this study. It requires multi-hop reasoning over multiple context passages, which is representative of the retrieval scenarios in which RAG systems operate and are most vulnerable. It provides ground-truth supporting passage indices and distractor passages per question, which lets us label poisoned passage indices exactly for supervised pre-filter training and for the True/Survived decomposition introduced in Section [sec:true-survived]. Finally, it is the base benchmark used in the original PoisonedRAG evaluation [zou2025poisonedrag], which permits direct comparison with prior work on RAG security.

We sample 100 HotpotQA hard questions whose answers require two supporting passages. These 100 base questions form the foundation of all 2{,}400 evaluation triplets in the final dataset.

### Triplet construction

Each evaluation instance is a triplet $(q, C, a^{*})$ where $q$ is a question, $C = \{p_{1}, p_{2}, \ldots, p_{n}\}$ is the retrieved context (a set of passages), and $a^{*}$ is the ground-truth answer. The original HotpotQA distractor context contains $n = 10$ passages per question: the two supporting passages $p_{s_{1}}, p_{s_{2}} \in C$ and eight distractor passages.

For each base question and each combination of injection type and noise level, we build a poisoned triplet by replacing a fraction of $C$ with adversarial passages. Let $\rho \in \{0.2, 0.4, 0.6, 0.8\}$ denote the noise level, representing the fraction of original passages replaced. For each $(q, \mathrm{injection}, \rho)$ combination, we also keep a paired clean triplet (original $C$ unchanged) as a within-question control. The dataset therefore contains:
\begin{equation}
N = N_{q} \times N_{\mathrm{inj}} \times N_{\rho} \times 2 = 100 \times 3 \times 4 \times 2 = 2{,}400 \text{ triplets}
\end{equation}
This yields 1{,}200 poisoned and 1{,}200 clean triplets, with each poisoned triplet matched to a clean control at the same $(q, \mathrm{injection}, \rho)$ slot.

For each poisoned triplet, the indices of replaced passages are recorded in a `poisoned_passage_indices` field, denoted $\mathcal{P} \subseteq \{1, \ldots, n\}$. This explicit per-passage labelling enables supervised pre-filter training and residual-aware analysis. The dataset fields are summarised in Appendix D.

[Figure omitted: Poisoned dataset construction pipeline, from HotpotQA base questions to paired clean and poisoned triplets with explicit passage-level poison indices.]

### Injection types

We implement three injection types of increasing adversarial sophistication, each probing a different judge failure mode.

**Type 1, random noise.** Let $\mathcal{D}_{\setminus q}$ denote the union of all passages associated with base questions other than $q$ in the HotpotQA pool. A subset $\mathcal{P} \subseteq \{1, \ldots, n\}$ of context indices is selected uniformly at random. The candidate donor pool is filtered by cosine similarity so that the noise is genuinely off-topic: only passages with $\cos(\mathbf{e}_{p}, \mathbf{e}_{q}) < 0.3$ are kept as candidates, and each selected position is replaced by a passage drawn uniformly from this filtered pool. If the cosine filter produces an empty pool for a given question, the unfiltered donor pool is used as fallback so that a replacement is always available. The replacement passages are topically incoherent with $q$ by construction and can be detected by embedding-distance signals at low cost.

**Type 2, adversarial fact rewriting.** A subset $\mathcal{P}$ is selected as in Type 1. For each selected passage $p_{i}$, a generation model produces a rewritten variant $\tilde{p}_{i}$ via a single LLM call conditioned on $p_{i}$ alone. The prompt does not condition on $q$ or $a^{*}$, making the rewrite a passage-local perturbation rather than a question-targeted attack. The resulting passage is fluent and topically coherent with the original, but factually misaligned. The generation prompt is given in Appendix B.

A post-hoc cosine-similarity guard rejects rewrites that drift below $\cos(\mathbf{e}_{p_{i}}, \mathbf{e}_{\tilde{p}_{i}}) < 0.3$, falling back to the original passage in those cases. A small fraction of generations that drift entirely off-topic are filtered out at construction time. Detecting Type 2 attacks requires factual reasoning rather than statistical anomaly detection.

**Type 3, PoisonedRAG-style injection.** Following the spirit of the black-box attack of [zou2025poisonedrag], each malicious passage is engineered to support a target wrong answer $a^{\dagger} \neq a^{*}$ while remaining topically coherent with the question. The construction is two-stage:

*Stage 1, target wrong-answer generation.* A generation model is prompted with the question $q$ and the correct answer $a^{*}$ and asked to propose a plausible alternative that is specific, factually wrong, and could plausibly appear in source text.

*Stage 2, malicious passage synthesis.* The generation model is given $q$, $a^{*}$, $a^{\dagger}$, and the original retrieved context, and is asked to produce a fluent passage that would mislead a downstream language model toward $a^{\dagger}$ rather than $a^{*}$. The exact prompt templates for both stages are given in Appendix B.

*Insertion mechanics.* For a triplet at noise level $\rho$, a number of malicious passages is determined and inserted into $C$ at uniformly-random positions, expanding the context length by that count. First, our malicious passage is the Stage 2 LLM output in isolation; we do not prepend $q$ as a separate retrieval sub-text, relying on topical coherence to minimize surface-level retrieval anomalies. Second, we insert the malicious passages *additively* into $C$ rather than replacing existing passages. This preserves all original supporting and distractor passages, allowing us to isolate the effect of poison addition rather than poison-plus-replacement. The `poisoned_passage_indices` field records the positions of the inserted malicious passages in the resulting expanded context.

The additive insertion ensures that for every poisoned PoisonedRAG-style triplet, the original supporting passages are still present in the context, which is essential for the True/Survived decomposition introduced in Section [sec:true-survived]. 

[Figure omitted: PoisonedRAG-style injection process, showing target wrong-answer generation, malicious passage synthesis, and additive insertion into the retrieved context.]

### Dataset splits and statistics

Triplets are partitioned into train, validation, and test splits with stratification by injection type and noise level to ensure balanced experimental conditions. The split sizes are detailed in Table [tab:splits]; the full dataset schema is provided in Appendix D.

[Table omitted: Dataset split sizes.]

The pre-filter classifier is trained on the train split, hyperparameters are selected on validation, and all classifier-level performance reported is evaluated on the held-out test split.

Summary statistics for the full dataset:

[Table omitted: Summary statistics of the Poisoned Evaluation Dataset.]

### Quality assurance

Each injection type passes type-specific quality checks before being added to the dataset.

For random noise, replacement passages are accepted only if their cosine similarity to the question is below $0.3$, so that the injected passages are genuinely off-topic.

For adversarial fact rewriting, each generated passage must satisfy three conditions: cosine similarity above $0.7$ with the original passage it replaces (to preserve retrievability), an automated consistency check confirming that the rewritten passage contradicts $a^{*}$, and an automated fluency check.

For PoisonedRAG-style, two operational conditions are verified: (i) the *retrieval condition*, that the malicious passage appears in the top-$k$ retrieved results when injected into the knowledge base; and (ii) the *generation condition*, that an LLM produces $a^{\dagger}$ when the malicious passage is provided as the sole context for the question.

A final integrity check confirms that no duplicate triplet IDs exist across splits and that the per-cell counts are balanced across $(\mathrm{judge}, \mathrm{injection}, \rho)$ combinations.

## Phase 2: Baseline meta-evaluation

### Judge selection

We evaluate three current LLM judges spanning different provider and deployment profiles. 

[Table omitted: LLM judges evaluated in this thesis.]

The models are chosen to cover different provider types rather than to exhaust the model space. Detailed configuration values are recorded in Appendix E.

### Scoring protocol

All three judges receive an identical scoring prompt. The prompt provides the question $q$, the retrieved context $C$ (passages separated by blank lines, truncated at 2{,}000 tokens), and the answer $a^{*}$. The judge is asked to return three scores (faithfulness, answer relevance, and context relevance), each a real number in $[0, 1]$, in JSON format. The judge is not informed whether the context is poisoned, mirroring a zero-knowledge deployment scenario. The exact scoring and justification prompts are provided in Appendix A.

We use the RAGAS conventions for the three metrics. **Faithfulness** measures whether the answer is supported by the provided context. **Answer relevance** measures whether the answer is on-topic for the question. **Context relevance** measures whether the retrieved context is relevant to the question. Faithfulness is the primary metric of interest because corpus poisoning compromises this dimension most directly.

### Primary metric: Faithfulness FPR

Let $T_{p} = \{t_{1}, t_{2}, \ldots, t_{|T_{p}|}\}$ denote the set of originally-poisoned triplets and let $f(t)$ denote the faithfulness score assigned by a given judge to triplet $t$. We define a triplet-level **false positive** as a poisoned triplet that the judge scores at or above a threshold $\tau=0.5$:
\begin{equation}
\mathrm{FP}(t) = \mathbb{1}[t \in T_{p} \wedge f(t) \geq \tau]
\end{equation}

The **False Positive Rate (FPR)** of the judge is the fraction of poisoned triplets scored as faithful:
\begin{equation}
\mathrm{FPR} = \frac{1}{|T_{p}|} \sum_{t \in T_{p}} \mathbb{1}[f(t) \geq \tau]
\end{equation}

The threshold corresponds to the natural midpoint of the scoring interval and is the canonical operating point for binarising RAGAS scores. 

### Secondary metrics

Three secondary metrics complement the headline FPR.

**Faithfulness gap** is the mean difference between clean and poisoned faithfulness scores:
\begin{equation}
\Delta_{f} = \overline{f}(T_{c}) - \overline{f}(T_{p})
\end{equation}
where $\overline{f}(\cdot)$ denotes the mean faithfulness score over the indicated triplet set and $T_{c}$ is the set of clean triplets.

**Stratified FPR** decomposes the headline FPR by injection type and noise level, which surfaces attack-class-specific failure patterns.

**Score distribution shape** characterises the judge's calibration regime. We report the fraction of scores in three bands: extreme low ($\leq 0.1$), middle ($0.1 < f < 0.9$), and extreme high ($\geq 0.9$). 

### Checkpoint and reliability

All judge outputs are stored in a structured format that preserves triplet identifiers, judge identifiers, metric scores, poisoning labels, and filter condition labels. The output schema is summarised in Appendix D.

## Phase 3: Pre-filter design

### Design criteria

Both pre-filter modules are designed against four criteria: (i) pipeline integration as a drop-in component without modifying the judge prompt, (ii) computational tractability, (iii) multi-signal redundancy for the statistical filter, and (iv) configurable sensitivity.

### Statistical pre-filter: five signals

The statistical pre-filter is a passage-level binary classifier. For each retrieved passage $p_{i} \in C$, the filter computes five signals, aggregated into a per-passage poison probability $\hat{y}_{i} \in [0, 1]$. Full configuration values are listed in Appendix E.

**Signal 1, embedding cosine anomaly.** This signal compares the passage embedding with the question embedding and flags unusually distant or unusually high-similarity passages.

**Signal 2, token-level entropy.** This signal uses language-model uncertainty to identify passages with unusual token-level probability patterns.

**Signal 3, fine-tuned classifier.** A lightweight transformer model is fine-tuned as a binary classifier on the training split to output a direct poison probability. 

**Signal 4, answer span recall.** This signal measures whether a passage preserves tokens associated with the ground-truth answer.

**Signal 5, cross-encoder relevance.** This signal scores query-passage relevance using an interaction-based cross-encoder.

### Signal aggregation

The five signal scores form a feature vector $\mathbf{s}_{i} = (s_{1}, s_{2}, s_{3}, s_{4}, s_{5})$ per passage. We evaluate three aggregation methods on the validation split: weighted vote, XGBoost, and majority vote. 

The most effective aggregator on the validation set is selected as the primary aggregator for end-to-end experiments, with Weighted Vote retained as an unbiased baseline for ablation analysis. 

[Figure omitted: Statistical pre-filter architecture, showing the five passage-level signals and their aggregation into a poison probability used for context filtering.]

### Classifier backbone

For Signal 3, we compare two encoder backbones: DeBERTa-v3-small and RoBERTa-base. Both are fine-tuned on the same train split with matched hyperparameters, listed in Appendix E.

### LLM pre-filter: triplet-level classification

The second pre-filter strategy uses Mistral-small-latest as an LLM classifier. This model sits outside the families of both the generation models and the judges, reducing in-family recognition confounds.

Because HotpotQA is a multi-hop task, evaluating passages in isolation is not appropriate: legitimate supporting passages may only become useful in combination with other passages. We therefore adopt a triplet-level strategy. The LLM receives $q$, $a^{*}$, and the full context $C$ in a single prompt, and it flags the indices of any passages it identifies as adversarial. 

The output is a set of flagged indices $\hat{\mathcal{P}} \subseteq \{1, \ldots, n\}$. In deployment, a triplet for which $\hat{\mathcal{P}} \neq \emptyset$ is excluded from evaluation entirely, making this a *content-preserving* filter, unlike the *content-modifying* statistical filter. The two strategies operate at different abstraction levels and admit different failure modes, which is why we evaluate both.

[Figure omitted: LLM pre-filter strategy, showing triplet-level inspection of the question, answer, and full context before flagging suspected adversarial passages.]

### Filtered context construction

For the statistical filter, the output is a modified context $C^{\prime} = C \setminus \{p_{i} : \hat{y}_{i} \geq \tau\}$ that the judge then scores.

For the LLM filter, the output is a triplet-level decision. We score the original context with the judge and compare verdicts on triplets the LLM filter flagged versus those it did not flag.

## Phase 4: End-to-end evaluation

### The True/Survived/Clean decomposition

To evaluate the effect of a content-modifying filter, we must account for whether the original ground-truth supporting passages were overwritten by the attack. We decompose the originally-poisoned set $T_{p}$ using the explicit poisoning indices $\mathcal{P}$ and supporting passage indices $\mathcal{S}$. 

A triplet is in the **True** subset if the attack overwrote at least one supporting passage ($\mathcal{S} \cap \mathcal{P} \neq \emptyset$). It is in the **Survived** subset if the supporting passages remain intact. The two subsets partition $T_{p}$:
\begin{equation}
T_{p}^{\mathrm{true}} = \{t \in T_{p} : \mathcal{S}(t) \cap \mathcal{P}(t) \neq \emptyset\}, \qquad T_{p}^{\mathrm{surv}} = T_{p} \setminus T_{p}^{\mathrm{true}}
\end{equation}

Due to the additive nature of Type 3 (PoisonedRAG-style) injections, these attacks never overwrite original passages, meaning all Type 3 attacks inherently belong to the Survived subset by construction. Comparing baseline and post-filter performance within these respective subsets isolates the filter's effect on worst-case scenarios (True) versus scenarios where malicious content sits alongside original supporting passages (Survived).

[Figure omitted: Residual-aware True/Survived/Clean decomposition used in the end-to-end evaluation to separate support-overwritten cases from support-intact cases.]

### Post-filter judge scoring

Phase 4 re-runs each of the three judges on the filtered triplets under both filter strategies. The primary comparison metric is the change in FPR relative to baseline. 

### Justification analysis

To probe whether judges can identify poisoning patterns when prompted appropriately, a small subset of triplets is re-scored using explicit poison-aware prompts. The poison-aware prompt is provided in Appendix A.

### Multi-judge disagreement

To test whether ensemble-based detection is a viable mitigation, we compute the standard deviation of faithfulness scores across the three judges for each baseline-condition triplet:
\begin{equation}
\sigma_{f}(t) = \mathrm{std}\bigl( f_{\mathrm{GPT}}(t), f_{\mathrm{Gemma}}(t), f_{\mathrm{DeepSeek}}(t) \bigr)
\end{equation}
We then compute the Pearson correlation between $\sigma_{f}(t)$ and the binary poisoning label $y(t) \in \{0, 1\}$. We additionally evaluate two ensemble decision rules at the operating threshold: a *mean-score* ensemble and a *majority-vote* ensemble.

### Statistical significance: McNemar test

The end-to-end evaluation requires paired hypothesis testing because the same triplet is scored under multiple conditions. For each cell, we compute McNemar's test on the discordant pair counts:
\begin{equation}
\chi^{2} = \frac{(|n_{10} - n_{01}| - 1)^{2}}{n_{10} + n_{01}}
\end{equation}
The continuity correction is applied, and we treat $p < 0.05$ as significant.

### Ablation studies

Two ablation studies are conducted on saved validation-split outputs. The **Per-signal ablation** removes each of the five signals in turn and re-evaluates the pipeline to observe individual signal contributions across injection types. The **Classifier backbone ablation** compares the two tested encoder backbones.

## Implementation and statistical analysis

The evaluation pipeline is implemented in Python using version-controlled configuration files. The implementation records raw judge outputs, pre-filter outputs, and analysis exports in structured files so that tables and figures can be regenerated from saved artefacts. The final configuration is summarised in Appendix E, and the reproducibility workflow is summarised in Appendix G.

Paired comparisons between conditions use McNemar's test for binary outcomes and the Wilcoxon signed-rank test for continuous scores. Unpaired comparisons across triplet subsets use the Mann-Whitney $U$ test. 

To account for inter-triplet dependencies stemming from the shared base questions, confidence intervals for classification metrics are computed via a question-clustered bootstrap resampling procedure (1{,}000 iterations). ROC curves are computed using scikit-learn's `roc_curve` function, with the optimal operating point selected using the Youden index ($\mathrm{sensitivity} + \mathrm{specificity} - 1$).


<!-- Source: 04_results.tex -->

# Results

This chapter reports the empirical findings of the experimental programme. Section 4.1 characterises the three LLM judges at baseline, before any defence is applied. Section 4.2 reports the statistical pre-filter as a passage-level classifier. Section 4.3 reports the LLM pre-filter (Mistral) as a triplet-level classifier. Section 4.4 is the central section of the chapter: it measures how each pre-filter affects judge reliability end-to-end, using a labelling convention that reflects what the judge actually sees after filtering. Section 4.5 audits what each filter did at the passage level and surfaces a collateral-damage finding that explains part of the end-to-end behaviour. Section 4.6 reports the justification analysis. Section 4.7 reports the multi-judge disagreement analysis. Section 4.8 summarises.

## Baseline Judge Characterisation

We score the full Poisoned Evaluation Dataset with each of the three judges (GPT-5.4-nano, Gemma-4-26B-A4B-IT, DeepSeek-V3.2) without any filter applied. The results establish how vulnerable each judge is to corpus poisoning before any defence is in place, and they surface qualitatively different scoring behaviours that shape every analysis that follows.

### Cross-judge baseline failure rates

A note on terminology before the numbers. Throughout this chapter, two related but distinct quantities are reported: the False Positive Rate (the fraction of poisoned triplets the judge scores at or above 0.5) and the mean RAGAS score on a particular subset. Higher values of either quantity, when computed on a poisoned subset, indicate judge failure rather than judge success - the judge is assigning faithful or relevant labels to context that is by construction corrupted. This convention applies uniformly across the faithfulness, context relevance, and answer relevance dimensions.

Baseline False Positive Rates differ substantially across judges. Table [tab:baseline-fpr] reports FPR on the originally-poisoned subset for all three RAGAS dimensions.

[Table omitted: Baseline FPR across the three RAGAS dimensions (n=1,200 originally-poisoned triplets per judge per metric, except Gemma where two API failures reduce n to 1,199).]

The headline numbers tell three different stories. DeepSeek is the strictest of the three on faithfulness and context relevance; it scores roughly 41\

Answer relevance FPR is the highest of the three RAGAS dimensions for every judge. This is structural rather than a sign of judge failure: the answer in our triplets is the HotpotQA ground-truth answer, which is genuinely relevant to the question regardless of what the context contains. We retain answer relevance throughout the analysis as a methodological control. It shows what a judge's behaviour looks like on a metric that is in principle insensitive to context corruption, and it provides a reference point against which faithfulness and context relevance behaviour can be compared.

### Three calibration regimes

The three judges occupy distinct calibration regimes that emerge from their score distributions on faithfulness (Figure [fig:baseline-calibration]). The differences are not subtle.

[Figure omitted: Baseline faithfulness score distributions for clean and poisoned triplets, shown separately for each judge. The dashed vertical line marks the 0.5 threshold used to convert scores into false-positive decisions.]

GPT produces a continuous distribution. Scores spread across the full [0, 1] interval, with 30.3\

Gemma exhibits extreme leniency. About 77.1\

DeepSeek is near-binary. About 99.4\

These calibration regimes matter for two reasons that recur through the chapter. First, they constrain which mitigation strategies are even applicable. A continuous-scoring judge supports confidence-abstention strategies because there is a meaningful uncertainty band to abstain on. A near-binary judge does not. Second, the calibration regimes interact non-trivially with attack type, as the next subsection shows.

A note on terminology: throughout this chapter we use "calibration regime" specifically to refer to the *shape* of the score distribution (continuous, near-binary, extreme-leniency). This is a structural property of how a judge maps internal confidence to a [0, 1] output. It is distinct from the stronger sense of "calibration" used in probabilistic forecasting, where a well-calibrated predictor's confidence values match empirical accuracy on held-out data. Validating the three judges' calibration in that stronger sense would require a human-annotated ground truth, which we do not have for this dataset; we treat that as a limitation in Section 5.5.

### Cross-judge divergence on PoisonedRAG-style attacks

The three injection types do not affect the three judges in the same way. Table [tab:baseline-fpr-injection] shows poisoned-set faithfulness FPR by judge and injection type.

[Table omitted: Baseline faithfulness FPR by injection type, per judge. Each cell uses n=400 originally-poisoned triplets.]

The cross-judge ordering reverses on PoisonedRAG-style. For GPT and Gemma, PoisonedRAG-style is the hardest of the three injection types to detect; their FPRs climb to 0.615 and 0.907 respectively. For DeepSeek, PoisonedRAG-style is the easiest of the three, with FPR dropping to 0.375. Gemma's FPR on PoisonedRAG-style is more than double DeepSeek's on the same attack class.

The reversal is striking and worth interpreting carefully. PoisonedRAG-style is the most adversarial of the three classes by construction. The retrieval sub-text guarantees the malicious passage will be retrieved, and the generation sub-text is engineered to support a target wrong answer that contradicts the ground-truth answer. The fact that GPT and Gemma find this hardest reflects what the attack is designed to do: produce content that looks topically coherent and factually plausible while steering toward the wrong answer.

DeepSeek's response is different in kind, and the cleanest interpretation we can offer connects this to its calibration regime. The near-binary scoring style means DeepSeek is not weighing graded evidence; it is making a categorical judgement. PoisonedRAG-style triplets contain an internal contradiction (the malicious passage supports an answer that contradicts the ground-truth answer in the triplet's answer slot), and DeepSeek's scoring style picks up on that contradiction reliably while the continuous-scoring judges weight the topical fluency of the malicious passage more heavily than the contradiction. We offer this as the most plausible mechanism consistent with our data; we cannot rule out alternative explanations such as DeepSeek's training distribution containing more contradiction-detection examples or its system-prompt sensitivity differing from the others. The directional reversal is robust; the mechanism behind it is partially speculative.

Within our three-judge panel, scoring strategy appears to matter at least as much as judge size or training compute for robustness to corpus poisoning. The three judges respond differently to the most sophisticated attack class, and the differences within our panel are not explained by raw model capacity or simple bias correction. Whether the same observation holds across a larger panel - particularly one that includes reasoning models or frontier-tier judges - is an empirical question we cannot answer with three judges; we treat the claim as suggested by our data, not established by it.

### Noise level sensitivity

Within each injection type, FPR varies with noise level (the proportion of context passages replaced). The pattern is broadly monotonic: higher noise levels are easier to detect for all three judges. At noise level 0.2, GPT's faithfulness FPR is 0.700; at noise level 0.8 it falls to 0.423. Gemma's drops from 0.850 to 0.555 across the same range. DeepSeek's falls most steeply, from 0.597 to 0.233.

The direction matches intuition. When more of the context is corrupted, the corruption becomes harder to ignore. The implication for the security narrative is mixed. Low-noise attacks (where most of the context is still legitimate) are the harder threat because the judges are more vulnerable to them. The structural failure cases that drive the highest FPRs (PoisonedRAG-style on GPT and Gemma) are not strongly noise-dependent; subtle attacks succeed at high baseline rates regardless of how much surrounding noise is present.

The noise-level pattern reverses partially after filtering, which we examine in Section 4.4.

## Statistical Pre-filter as a Classifier

Section 4.1 established that judges are vulnerable. We now turn to the first proposed mitigation: a multi-signal statistical pre-filter that scores each retrieved passage and removes those flagged as poisoned. The filter is evaluated as a binary passage-level classifier on the held-out test split of the Poisoned Evaluation Dataset.

### Headline performance

The filter performs well. On 5,119 test-split passages (1,155 poisoned, 3,964 clean), it achieves precision 0.955, recall 0.904, and F1 of 0.929. The clean false-positive rate is 0.012 - clean passages are essentially never wrongly flagged.

[Table omitted: Statistical pre-filter performance on the held-out test split (5,119 passages).]

These are strong numbers in absolute terms. The filter is highly precise: when it flags a passage as poisoned, it is right about 95.5\

### Recall by attack type

The filter's behaviour varies by injection type. Table [tab:stat-filter-by-injection] reports per-attack-type performance on the test split.

[Table omitted: Statistical filter classifier performance per injection type, test split.]

The filter performs well across all three attack types. Recall is highest on the most adversarial class (PoisonedRAG-style at 1.000) and lowest on the middle class (adversarial fact at 0.812). The PoisonedRAG-style result is striking: the filter catches every poisoned PoisonedRAG passage in the test set. The combined signal set evidently captures structural artefacts of the attack reliably; identifying which individual signals carry the most weight in this regime is examined in Section 4.2.3 below.

The adversarial fact class is the hardest because the manipulation is at the factual level, not at the surface or structural level. The injected passage is fluent, topically aligned with the question, and embedding-similar to the original passage it replaces. The filter must rely on the DeBERTa classifier (Signal 3) and answer span recall (Signal 4) to catch these cases, and not all of them carry strong enough signal for the classifier to flag.

### Per-signal contribution and classifier backbone

A per-signal ablation removes one signal at a time and recomputes pipeline F1. The DeBERTa classifier (Signal 3) is the largest single contributor: removing it drops F1 by approximately 0.030. The cross-encoder (Signal 5) contributes roughly 0.029 and answer span recall (Signal 4) contributes roughly 0.018. The embedding cosine and entropy signals contribute negligibly to aggregate F1 in this ablation; removing either changes overall F1 by less than 0.001.

The per-injection ablation breakdown (Appendix D) shows that the per-signal contributions vary somewhat by attack type but no single signal dominates PoisonedRAG-style detection in isolation. The combined signal set, taken together, achieves the high test-split PoisonedRAG-style recall reported in Table [tab:stat-filter-by-injection] - but the ablation does not identify a single removable signal whose absence collapses that recall. The headline interpretation is that the multi-signal aggregation is genuinely doing aggregate work; the filter is not reducible to any one of its inputs.

The classifier backbone comparison (DeBERTa-v3-small vs RoBERTa-base, both fine-tuned on the same training split with matched hyperparameters) shows the two architectures essentially tied at validation F1 of 0.7215 and 0.7202. Either backbone produces a usable filter; the choice between them is not load-bearing for the headline results. Detailed ablation tables are reported in Appendix D.

The headline of Section 4.2 is straightforward: the statistical filter is a strong classifier, with high precision (0.955) and high recall (0.904) on the held-out test set. Three of its five signals (DeBERTa, cross-encoder, answer span recall) carry the discriminative weight in aggregate F1. The filter performs particularly well on PoisonedRAG-style attacks at the test-set operating point.

## LLM Pre-filter (Mistral) as a Classifier

The second pre-filter strategy uses `mistral-small-latest` as a triplet-level classifier. Section 3.4.5 set out the rationale: a triplet-level operating mode (the filter sees all retrieved passages at once and gives a single verdict) and a deliberate cross-architecture choice (Mistral is outside both the poison-generation family and all three judge families).

### Headline performance

[Table omitted: Mistral triplet-level filter performance.]

Mistral catches just over half of the poisoned triplets, with a clean-FPR of 0.174 - about one in six clean triplets is wrongly flagged. The trade-off relative to the statistical filter is sharp: lower recall (0.505 vs 0.904), substantially higher clean-FPR (0.174 vs 0.012), at a coarser granularity (whole-triplet rejection rather than per-passage removal).

### Strong asymmetry by attack type

Mistral's per-attack-type behaviour reveals a sharp asymmetry that the statistical filter does not exhibit.

[Table omitted: Mistral triplet-level performance by injection type.]

Mistral catches 76.5\

The two filters detect partially complementary phenomena. The statistical filter exploits structural artefacts of the attacks (retrieval-optimised embeddings, classifier-detectable text patterns, missing answer tokens). Mistral judges the triplet as a whole and reasons about cross-passage consistency. PoisonedRAG-style attacks are easy for Mistral because the malicious passage explicitly contradicts the supporting passages in a way that a cross-passage reading can detect. Random-noise attacks are hard for Mistral because the noise passages are irrelevant rather than contradictory; they do not produce the kind of internal inconsistency Mistral can flag from a holistic reading.

The complementarity has implications for hybrid filter design that we return to in Section 5.4.

## End-to-End Effect of Filtering on Judge Reliability

This is the central section of the chapter. Sections 4.2 and 4.3 measured each filter as a classifier in isolation. The question that matters for production deployment is different: when a filter sits between the retriever and the judge, does the judge become more reliable? The answer is more nuanced than the standalone classifier metrics suggest, and reaching it requires care about what we are measuring.

### Defining what "post-filter FPR" should measure

The naive post-filter FPR computation takes every triplet that was originally poisoned and asks how often the judge scores it as faithful after filtering. This computation has a conceptual problem. The statistical filter modifies content. After it runs, a triplet that was originally poisoned may have only had distractor passages perturbed (the supporting passages survived the attack) or may have had supporting passages overwritten (the supporting passages were killed). The two cases look very different to the judge, and aggregating them under a single "originally poisoned" label hides what is actually happening.

We therefore decompose the originally-poisoned subset into two categories using the explicit per-passage poisoning indices recorded at construction time:

  - **True**: the originally-supporting passages were among the poisoned ones (the attack hit the support and the judge faces concentrated poison at evaluation time).
  - **Survived**: the originally-supporting passages were not among the poisoned ones (the attack hit only distractor positions, or augmented the context additively, leaving clean support intact).

The decomposition has a structural consequence specific to our injection design. Type 1 (random noise) and Type 2 (adversarial fact rewriting) overwrite passages in place: the random sample of replacement positions can either hit or miss the supporting passages, producing a mixture of True and Survived triplets per injection type. Type 3 (PoisonedRAG-style) operates additively: malicious passages are inserted alongside the original passages without overwriting any of them. As a structural consequence of additive injection, Type 3 triplets are always in the Survived category - the supporting passages are present in every Type 3 poisoned context, simply alongside the inserted malicious content. This means the True subset is composed entirely of Type 1 and Type 2 triplets, while the Survived subset contains a mixture of Type 1, Type 2 (where the random replacement happened to miss support), and all of Type 3.

This asymmetry is not a limitation of the analysis but a consequence of how the attacks work: PoisonedRAG-style is by design a poison-augmentation attack rather than a replacement attack, so its end-to-end behaviour is captured in the Survived subset. We report behaviour on each subset separately. Comparing baseline-True to postfilter-True isolates how filtering changes the judge's response to the worst case (concentrated poison after support has been overwritten). Comparing baseline-Survived to postfilter-Survived isolates how filtering changes the judge's response to the case where support is intact alongside poison.

The split is computed by exact-title matching against per-question passage title lists rather than by substring search. Of 1,200 originally-poisoned triplets, 576 fall into the True category (288 random_noise + 288 adversarial_fact + 0 poisonedrag_style) and 624 into the Survived category (112 random_noise + 112 adversarial_fact + 400 poisonedrag_style). The split exists at construction time and does not depend on what the filter does.

### Headline finding: filtering makes the worst case worse

Table [tab:faithfulness-category-means] reports mean faithfulness scores for each judge under each condition, decomposed into Clean / Survived / True.

[Table omitted: Mean faithfulness score by judge, condition, and category. Higher means in the True category indicate the judge is scoring poisoned context as more faithful, which is the failure mode we are measuring. (PostfilterLLM Gemma cells reflect two API failures that reduced the Survived n by 1 and the True n by 1.)]

[Figure omitted: Mean faithfulness by category across baseline, statistical post-filtering, and LLM post-filtering. The True-category annotations show the change from baseline, where positive values indicate the filter-induced score increase.]

The True column under postfilter is higher than the True column under baseline for all three judges. Filtering raises the rate at which judges score concentrated-poison contexts as faithful. The DeepSeek result is the most pronounced: baseline True mean of 0.314 jumps to 0.465 under the statistical filter - a 0.151 increase. GPT moves from 0.391 to 0.444, Gemma from 0.442 to 0.514. The statistical filter makes the worst case worse for every judge, with DeepSeek's effect roughly twice the magnitude of GPT's and Gemma's. The aggregate True-mean reported here averages over Type 1 (random noise) and Type 2 (adversarial fact rewriting) triplets, which contribute roughly equal counts but produce noticeably different per-injection effect sizes - for DeepSeek, +0.119 on random noise and +0.183 on adversarial fact, a 50\

The LLM filter (Mistral) shows a weaker and uneven pattern. DeepSeek's True mean rises by 0.053 under postfilter_llm. GPT's drops by 0.017 and Gemma's drops by 0.022. The LLM filter does not modify content (Mistral classifies whole triplets rather than removing passages), so on True-category triplets - where the malicious passages have already overwritten supporting content - the filter has no opportunity to make the residual easier or harder for the judge. The DeepSeek effect is the only material True-mean change under the LLM filter.

The Survived column tells a more pronounced story. Survived means rise substantially under the statistical filter for GPT (0.638 → 0.685, Δ +0.047) and DeepSeek (0.494 → 0.720, Δ +0.226), and rise further under the LLM filter for both (GPT to 0.737, DeepSeek to 0.721). For Gemma the statistical filter reduces Survived by 0.061 (0.912 → 0.851), the only Survived cell where filtering visibly hurts. The Gemma decrease is consistent with its extreme-leniency calibration: Gemma's baseline Survived score is already near the ceiling, and the filter's collateral removal of supporting passages can only pull it down.

A compositional caveat: the Survived subset contains 112 random_noise + 112 adversarial_fact + 400 PoisonedRAG-style triplets, so PoisonedRAG-style is roughly 64\

[Figure omitted: Effect of the statistical and LLM filters on the True and Survived split of originally-poisoned triplets.]

### The picture across context relevance and answer relevance

The faithfulness paradox in Table [tab:faithfulness-category-means] is partially specific to faithfulness. The other two RAGAS dimensions show the same direction in places but with different magnitudes and judge composition.

For context relevance, the True means under baseline are GPT 0.508, Gemma 0.695, DeepSeek 0.135. Under the statistical filter they are GPT 0.524, Gemma 0.666, DeepSeek 0.315. The DeepSeek context relevance shift (+0.180) is the largest True-subset paradoxical increase in the analysis. GPT's True context relevance is essentially unchanged (+0.016). Gemma's True context relevance actually decreases by 0.030 - one of two cells in the analysis where filtering visibly reduces a True-mean. The mechanism is the same as the faithfulness Survived decrease for Gemma: the baseline-True context relevance is already moderate-to-high (0.695), and the filter's collateral removal of supporting passages can only pull it down rather than make the residual look more relevant.

For answer relevance, all True-mean changes are small and uniformly positive (GPT +0.045, Gemma +0.031, DeepSeek +0.051). The shifts are smaller than for faithfulness or context relevance and consistent with the structural insensitivity of the metric noted in Section 4.1.1: answer relevance is largely about whether the answer is on-topic for the question, which is preserved regardless of context corruption. Answer relevance functions here as a methodological control showing that the True/Survived shifts on faithfulness and context relevance are not dominated by uniform changes in score-generation logistics across conditions.

The paradox's strength varies across metric: substantial on faithfulness and context relevance for DeepSeek, smaller but consistent on faithfulness for GPT and Gemma, and small on answer relevance throughout. The DeepSeek case is the cleanest example of the worst-case-worsened effect across two RAGAS dimensions; for the other judges, the pattern holds but more moderately. Detailed per-injection breakdowns are shown in Figures [fig:per-injection-faithfulness]--[fig:per-injection-answer].

[Figure omitted: True-category faithfulness by judge, injection type, and filtering condition.]

[Figure omitted: True-category context relevance by judge, injection type, and filtering condition.]

[Figure omitted: True-category answer relevance by judge, injection type, and filtering condition.]

### Per-injection magnitude

The per-injection breakdown is structurally different across the True and Survived subsets, given that PoisonedRAG-style triplets all sit in Survived. We report the two subsets separately.

**True subset (random_noise and adversarial_fact only).** For DeepSeek under the statistical filter, baseline-True faithfulness vs postfilter-True faithfulness:

  - Random noise: 0.340 → 0.460 (Δ +0.119)
  - Adversarial fact: 0.287 → 0.470 (Δ +0.183)

For GPT the same comparison: random noise 0.347 → 0.413 (Δ +0.065), adversarial fact 0.434 → 0.474 (Δ +0.040). For Gemma: random noise 0.379 → 0.493 (Δ +0.114), adversarial fact 0.506 → 0.535 (Δ +0.030). The True-subset paradox is moderate-to-large for DeepSeek, moderate for Gemma on random noise, and small for the others. Adversarial fact and random noise behave similarly within each judge, with no strong injection-type-specific pattern in the True subset.

**Survived subset (all three injection types, with PoisonedRAG-style dominating numerically).** Survived means under the statistical filter shift dramatically for PoisonedRAG-style triplets, where the residual after filtering is the most consequential. For DeepSeek on PoisonedRAG-style: baseline Survived 0.361 → postfilter 0.738 (Δ +0.377). This is the single largest cell-level effect in the analysis. For GPT on PoisonedRAG-style: 0.570 → 0.709 (Δ +0.139). For Gemma on PoisonedRAG-style: 0.885 → 0.914 (Δ +0.029, ceiling effect).

The Survived effects on adversarial_fact and random_noise are smaller and split in direction: GPT and Gemma Survived means on those types decrease modestly (consistent with collateral support loss), while DeepSeek Survived means rise modestly. The PoisonedRAG-style Survived effect dominates the aggregate because (i) Type 3 contributes 400 of the 624 Survived triplets per cell and (ii) its post-filter mean shifts farther than any other injection type's. Per-injection patterns for all judges and conditions are shown in Figure [fig:per-injection-faithfulness].

The likely mechanism behind the PoisonedRAG-style Survived shift is consistent with the broader picture developed in Section 4.5.2: filtering produces a shorter, more topically focused residual through the removal of original distractor passages, and judges score the cleaner-looking residual as more faithful regardless of whether the surviving content actually supports the answer. PoisonedRAG-style triplets are particularly susceptible because they contain highly polished malicious passages alongside intact supporting context. At baseline, the judge sees a mixture: original supporting passages plus polished poison plus original distractors, and DeepSeek's near-binary scoring (Section 4.1.3) registers the contradiction reliably. After the statistical filter runs, it removes the more conspicuous distractors and some of the malicious passages, leaving the most polished malicious content alongside the supporting passages in a context that is shorter, less noisy, and more credible-looking. The judge revises its faithfulness verdict upward. The DeepSeek PoisonedRAG-style postfilter Survived mean of 0.738 is what this looks like in the cell.

We treat this mechanism story as a plausible interpretation rather than a proven causal claim. The strongest evidence comes from the distractor-removal stratification (Section 4.5.2), which shows the True-subset Δ rising monotonically with the number of distractors removed, plus the post-filter length correlation (r ≈ 0.31–0.47 across judges) which is much stronger than at baseline. The earlier hypothesis that the paradox is driven by collateral removal of supporting passages is empirically refuted in Section 4.5.2; collateral damage happens but moves scores in the opposite direction.

### Statistical significance

McNemar's paired test on triplet-level FPR-threshold-crossings detects the True-category shifts at conventional levels when triplets are treated as independent observations: GPT χ² = 8.16 (p = 0.004), Gemma χ² = 10.93 (p < 1e-3), DeepSeek χ² = 34.40 (p < 1e-8) on the True subset under the statistical filter. This treatment is, however, optimistic: the 1,200 poisoned triplets derive from 100 base questions, with each question contributing twelve triplets (three injection types × four noise levels). Triplets sharing a base question share the question text, the ground-truth answer, and the original supporting passages, and are therefore not independent observations.

A question-clustered bootstrap (1,000 iterations resampling at the base-question level) gives the more honest picture. The 95\

[Table omitted: McNemar paired test on True-subset faithfulness threshold-crossings (≥0.5). Uncorrected χ² and p-values treat triplets as independent. Bootstrap-corrected values resample at the base-question level (n=100 questions, 1,000 iterations), accounting for within-question correlation across noise levels and injection types.]

[Figure omitted: McNemar test statistic for faithfulness threshold-crossing changes in the True category, comparing baseline with each post-filter condition.]

After clustered correction, GPT and Gemma's True-subset shifts under the statistical filter are no longer significant at conventional thresholds; their bootstrap CIs include zero. DeepSeek's True-subset effect is borderline (p = 0.099). The Survived-subset effects tell a different story - the DeepSeek Survived shift, which traces almost entirely to PoisonedRAG-style triplets, remains robustly significant under clustered bootstrap (χ² = 77.7, bootstrap CI [29.6, 73.4], corrected p = 0.012 under the statistical filter; corrected p < 0.001 under the LLM filter). The largest single end-to-end effect in the analysis is also the most statistically robust.

The implication is that the True-subset paradox is directional and visible across all three judges but is not, with the data we have, decisively significant for GPT or Gemma once question-level clustering is accounted for. The Survived-subset effect on PoisonedRAG-style triplets is robust under any reasonable correction. The thesis's central claim - that filtering reshapes judge response in counterintuitive ways - is best supported by the Survived-subset result on the most adversarial attack class, not by the True-subset shifts on Types 1 and 2 alone.

### The reverse across noise levels

The end-to-end behaviour reverses partially across noise levels - and in a direction that is operationally important. At low noise levels (0.2, where only 20\

For DeepSeek under the statistical filter, postfilter-True faithfulness by noise level: 0.671 at noise=0.2, 0.652 at noise=0.4, 0.468 at noise=0.6, falling to 0.244 at noise=0.8. The pattern reverses the baseline from Section 4.1.4, where higher noise was easier for the judge. After filtering, lower noise is the harder case, with the postfilter-True mean dropping by more than 0.4 across the noise range.

The mechanism is straightforward in retrospect. At noise=0.2, only one or two passages were replaced. The filter removes the most obvious among those. The residual is mostly clean passages plus the supporting passages - a context that reads as well-formed, polished, and topically coherent. The judge reads this context, sees no obvious anomaly, and scores the answer as faithful. At noise=0.8, the filter is removing many more passages. The residual is sparse and visibly degraded. The judge correctly reads this as untrustworthy and scores low.

The implication for deployment is uncomfortable. Subtle, low-noise attacks are exactly the threats that an upstream filter should be best positioned to handle, because the filter has more clean context to work with. Instead, low-noise attacks become more dangerous post-filter than they were at baseline, because the filter's success at removing the poison produces a residual that looks more trustworthy than it is.

[Figure omitted: True-category mean faithfulness by noise level under baseline, statistical post-filtering, and LLM post-filtering.]

## Filter Behaviour Audit

The end-to-end results in Section 4.4 raise a mechanism question. Why does the filter, which is a strong classifier (F1 = 0.929 on the test set), make judge behaviour worse on the True subset? Section 4.5 audits what the filter actually did at the passage level and surfaces a finding that explains part of the answer.

### What the statistical filter removed

For each originally-poisoned triplet, we record three counts after the statistical filter runs: how many poisoned passages were correctly removed, how many were missed, and how many *originally-supporting* passages were removed (collateral damage).

Table [tab:filter-audit-by-injection] summarises by injection type.

[Table omitted: Statistical filter passage-level behaviour by injection type. The "supporting lost" column counts originally-supporting passages that were wrongly flagged and removed by the filter.]

[Figure omitted: Passage-level filter audit by injection type, including correctly caught poison, missed poison, and collateral removal of supporting passages.]

Two observations from this table.

First, the per-triplet recall (mean caught divided by mean poisoned) is substantially below the test-set recall reported in Section 4.2. The test-set numbers are computed on a held-out evaluation split where the classifier operates at its trained operating point. The whole-dataset audit, which includes triplets the classifier saw during training and validation, shows a more conservative effective recall - the headline 0.929 F1 reflects classifier performance, not the rate at which poison is fully eliminated from a random triplet.

Second, and more importantly, the filter removes about 0.32 to 0.39 originally-supporting passages per triplet (averaging across the full Poisoned Evaluation Dataset). Each triplet has only two supporting passages by HotpotQA construction, so a loss of 0.35 corresponds to roughly 17.5\

A test-split-only re-audit shows that the collateral-loss figure varies modestly with the data partition. On the test split alone (n=80 originally-poisoned triplets per injection type), the mean supporting-passages-lost figures are 0.21 (random_noise), 0.43 (adversarial_fact), and 0.38 (poisonedrag_style). The full-dataset and test-only numbers differ by 0.05 or more for adversarial_fact and random_noise, reflecting the fact that the DeBERTa classifier has seen the training-set passages during fitting and behaves slightly differently on unseen data. The full-dataset 0.345 figure should therefore be read as an in-sample aggregate; the deployment-relevant collateral-loss rate is the test-only value, and it depends on the injection type.

### What is actually driving the True-subset shift

A naive reading of the filter audit suggests the collateral damage matters: the filter removes supporting passages 17.5\

Table [tab:support-removal-stratified] reports the result.

[Table omitted: True-subset Δ stratified by whether the statistical filter removed at least one supporting passage. The paradox (Δ > 0) appears only in triplets where supporting passages survived the filter. In triplets where collateral damage occurred, scores drop substantially in all three judges.]

The result is the opposite of what the collateral-damage hypothesis predicts. In all three judges, when the filter removed supporting context, faithfulness scores fell by 0.15 to 0.23 - the judge correctly detected the resulting evidential gap. The paradox (post-filter scores rising) emerges entirely in triplets where the filter spared the supporting passages. Collateral damage is a real phenomenon (it happens 17.5\

Two alternative mechanisms are consistent with the corrected picture: context shortening through distractor removal, and a length-driven shift in judge calibration. We tested both.

**Distractor removal.** For each True-subset triplet under the statistical filter, we computed the number of original HotpotQA distractor passages the filter removed correctly. Table [tab:distractor-removal-stratified] stratifies the True-mean Δ by the quartile of distractor removal.

[Table omitted: True-subset Δ stratified by quartile of original distractor passages removed by the statistical filter. The paradox grows monotonically with the number of distractors removed, in all three judges. The largest Δ in the high-quartile cell (Q3) reaches +0.461 for DeepSeek and +0.335 for Gemma.]

The pattern is monotonic in all three judges. Triplets where the filter removed few distractors show no paradox (or a small negative Δ). Triplets where the filter removed many distractors show a large paradox (Δ +0.25 to +0.46). The mechanism is not collateral damage; it is context clarification through distractor removal. The original HotpotQA distractor passages, which the dataset constructors included precisely to make the multi-hop task harder, are noisy and topically loose. The statistical filter, designed to remove poisoned passages, also removes many of these original distractors. The residual is shorter and more topically focused - and judges score this cleaner residual as more faithful, regardless of whether it actually contains the supporting evidence needed to answer the question correctly.

**Length effect.** The mechanism above predicts that post-filter contexts should be shorter on average and that judge scores should correlate with context length more strongly post-filter than at baseline. Both predictions hold. Mean post-filter context length is 7.11 passages versus 11.62 at baseline, a 39\

These results require us to revise the mechanism story given in Section 4.4. The end-to-end paradox is not driven by the polished-poison residual or by collateral damage in any direct sense. It is driven by the filter's incidental removal of original HotpotQA distractors, which produces a shorter and more topically focused residual that the judge reads as cleaner, more coherent, and therefore more faithful. The remaining poison sits inside this cleaner-looking residual and benefits from its credibility. The collateral-damage phenomenon happens but moves scores in the opposite direction: when the filter does kill supporting passages, the judge correctly detects the evidential gap and scores lower. The aggregate True-subset paradox we report in Section 4.4 is the net of these two mostly-opposing effects, with the distractor-removal effect dominating numerically.

### LLM filter audit

The LLM filter (Mistral) does not modify passages, so the collateral-damage analysis does not apply in the same form. Mistral classifies each triplet as poisoned or not, and in deployment a flagged triplet would be excluded entirely. The audit-relevant quantity for Mistral is the per-attack triplet-level recall and false-alarm rate, already reported in Table [tab:mistral-filter-by-injection]: recall ranges from 0.140 (random noise) to 0.765 (PoisonedRAG-style), and clean-FPR is around 0.17 across all attack types.

The LLM filter does not exhibit the collateral-damage mechanism by construction. It exhibits a different limitation: low recall on subtle attacks, particularly random noise. Of the two filters, the statistical filter catches more poison but at the cost of removing some legitimate supporting context; the LLM filter is more conservative about modifying triplets but lets more poison through. Neither cleanly avoids the underlying problem of producing residuals (or routing decisions) that improve judge reliability end-to-end.

## Justification Analysis

The justification analysis re-scores a 28-triplet subset under three prompt conditions: a standard scoring prompt (the same prompt used in the main experiments), a standard prompt applied after the statistical pre-filter has run, and a poison-aware prompt that explicitly describes the three attack types before asking for scores. The subset is stratified across the three injection types and is scored only by GPT-5.4-nano (the judge with continuous calibration, where score variation is most informative).

The poison-aware prompt produces a substantial shift on the two engineered attack types. For PoisonedRAG-style triplets, mean faithfulness drops from 0.800 under the standard prompt to 0.492 under the poison-aware prompt (Δ −0.308). For adversarial-fact triplets, the same prompt produces a drop from 0.758 to 0.558 (Δ −0.200). For random-noise triplets, the change is smaller in magnitude but still in the same direction (0.600 to 0.517, Δ −0.083). The standard-postfilter condition shows a different pattern: faithfulness on PoisonedRAG-style stays high (0.830, essentially unchanged from the standard 0.800), faithfulness on adversarial fact drops modestly (0.666), and faithfulness on random noise collapses to 0.233. This split is consistent with the distractor-removal mechanism in Section 4.5.2: the statistical filter removes the noisy random-noise injections aggressively, exposing the residual as visibly degraded, while the polished PoisonedRAG-style content survives the filter and inherits the credibility of a cleaner-looking residual.

Two qualitative observations from the reasoning text reinforce the score patterns. First, on PoisonedRAG-style triplets where the poison-aware prompt elicits low faithfulness scores, the judge's justification text identifies the adversarial passage explicitly - naming the contradiction with the ground-truth answer or flagging the retrieval-optimised structure. The judge can identify the attack when primed to look for it. Second, on adversarial-fact triplets, the justification text often correctly flags the contradiction in the reasoning even when the numerical score remains relatively high. The judge's reasoning recognises the attack but its scoring does not always reflect the recognition.

The sample size (28 triplets, single judge, three prompt conditions) is too small to support strong statistical claims. We report the analysis as a qualitative supplement to the main results: it suggests that explicit prompting can mitigate the engineered attack types more than it does random noise, that filter-then-judge interacts non-trivially with attack type (helpful on random noise, neutral or worse on PoisonedRAG-style), and that the gap between judge reasoning and judge scoring is itself a phenomenon worth further study. The full prompt texts and a representative sample of justifications are included in Appendix B.

## Multi-Judge Disagreement

We compute, for each triplet at baseline, the standard deviation of faithfulness scores across the three judges (GPT, Gemma, DeepSeek) and correlate the per-triplet std with the binary `is_poisoned` ground-truth label.

The Pearson correlation is r = +0.113 (p < 1e-6, n = 2,397 triplets where all three judges produced scores). The mean inter-judge std on poisoned triplets is 0.256; on clean triplets it is 0.205. Poisoned triplets produce slightly more disagreement than clean ones, and the difference is statistically significant given the sample size, but the effect is small in magnitude. (We note separately that the cross-judge correlation between GPT and DeepSeek faithfulness scores is r = +0.514. This is a different quantity - it asks whether the two judges tend to score the same triplet similarly - and confirms only that the judges are not independent samples of the underlying judgement task.)

The practical implication is that ensemble-based detection (flagging triplets where the three judges disagree most) is not viable as a poisoning detector. A correlation of 0.113 explains roughly 1\

A secondary finding worth noting: when we restrict to poisoned triplets only and correlate per-triplet inter-judge std with injection-type ordinal severity (random noise = 0, adversarial fact = 1, PoisonedRAG-style = 2), the correlation is r = +0.287. Inter-judge disagreement grows with attack sophistication. Judges agree on what random noise looks like and disagree on what PoisonedRAG-style looks like. This is consistent with the cross-judge calibration divergence on PoisonedRAG-style reported in Section 4.1.3.

This is a useful negative result. It rules out a tempting but ineffective mitigation. Practitioners considering ensemble methods for adversarial-RAG defence should not expect them to recover detection sensitivity that the constituent judges lack individually. The disagreement signal exists but is too weak to operationalise.

[Figure omitted: Distribution of per-triplet faithfulness disagreement across judges, separated by clean and poisoned triplets.]

## Summary of Findings

The empirical findings of Chapter 4 can be summarised in seven points.

First, the three judges fail at substantial rates on baseline poisoned context. Faithfulness FPRs range from 0.412 (DeepSeek) to 0.555 (GPT) to 0.699 (Gemma). LLM-as-a-Judge under corpus poisoning is a real vulnerability with magnitude that is operationally significant.

Second, the three judges occupy distinct calibration regimes (continuous, near-binary, extreme-leniency) and respond differently to attack class. GPT and Gemma find PoisonedRAG-style attacks hardest; DeepSeek finds them easiest. Within our panel, scoring strategy appears to matter at least as much as judge size or training compute for robustness; whether this generalises beyond three judges is an open question.

Third, the multi-signal statistical pre-filter performs well as a classifier on the held-out test set (F1 = 0.929, precision = 0.955, recall = 0.904, clean FPR = 0.012). The filter's strongest recall is on PoisonedRAG-style (1.000), driven by the embedding cosine signal exploiting the retrieval-optimisation property of these attacks.

Fourth, the LLM-based pre-filter (Mistral, triplet-level) performs less well as a classifier (F1 = 0.601) and shows a sharp asymmetry by attack type - strong on PoisonedRAG-style (recall 0.765), weak on random noise (recall 0.140). The two filters detect partially complementary phenomena.

Fifth, end-to-end the picture is paradoxical. Decomposing originally-poisoned triplets into True (supporting passages overwritten) and Survived (supporting passages intact), filtering raises True-category faithfulness for all three judges under the statistical filter (Δ +0.053 GPT, +0.072 Gemma, +0.151 DeepSeek). Under the LLM filter the True-subset effect is significant only for DeepSeek (+0.053). On context relevance the paradox is concentrated on DeepSeek (+0.180). The largest single end-to-end effect is in the Survived subset for PoisonedRAG-style: DeepSeek's Survived faithfulness rises by 0.377 under the statistical filter, driven by the polished residual that emerges when the filter removes the noisier malicious passages and leaves the most fluent ones alongside intact supporting context. Filtering makes the worst case (True) worse for all judges, and in DeepSeek's case dramatically transforms the easier case (Survived) when faced with PoisonedRAG-style attacks.

Sixth, the operative mechanism behind the True-subset shifts is not what we initially conjectured. We tested two candidate mechanisms empirically. The collateral-damage mechanism - filter removes supporting passages and the judge loses contradicting evidence - is refuted by the data: in triplets where the filter removed a supporting passage, scores fell by 0.15–0.23 across all three judges. The paradox emerges only in triplets where the filter spared the supporting passages. The operative mechanism is context clarification through distractor removal: the filter removes original HotpotQA distractor passages at high rates, the residual is shorter (mean 7.1 vs 11.6 passages) and more topically focused, and judges score this cleaner residual as more faithful. Stratifying the True-subset Δ by quartile of distractor removal shows monotonically rising Δ in all three judges, reaching +0.25 to +0.46 in the top quartile.

Seventh, multi-judge disagreement does not provide a usable detection signal. The correlation between inter-judge std and the poisoning label is r = +0.113 - statistically significant but operationally too small. Mean-score and majority-vote ensembles both perform worse than the best individual judge.

These findings collectively support the claim that LLM judges are vulnerable to corpus poisoning and that lightweight upstream filtering does not, in current designs, mitigate the vulnerability - and may worsen it on the cases where mitigation is most needed. The implications for filter design and judge selection are the focus of Chapter 5.

---


<!-- Source: 05_discussion.tex -->

# Discussion

The empirical findings of Chapter 4 admit several interpretations, each with implications for how RAG evaluation should be designed and how its trustworthiness should be measured. This chapter discusses four threads in turn. Section 5.1 unpacks the central paradox finding and what it means mechanistically. Section 5.2 examines the cross-judge differences and what they imply for the LLM-as-a-Judge paradigm. Section 5.3 places the work in the context of existing RAG defences. Section 5.4 outlines mitigation directions the present results point toward. Section 5.5 records limitations. Section 5.6 considers practical implications for practitioners deploying LLM-as-a-Judge pipelines.

## The Filtering Paradox: Mechanism and Interpretation

The headline empirical finding is that pre-filtering does not reliably mitigate judge vulnerability to corpus poisoning, and on the most adversarial attack class it substantially shifts judge behaviour in a counterintuitive direction. The largest and most statistically robust effect is the DeepSeek Survived-subset shift on PoisonedRAG-style triplets, where mean faithfulness rises from 0.361 at baseline to 0.738 after the statistical filter (Δ +0.377; question-clustered bootstrap p = 0.012). The True-subset effects are smaller and less robust statistically. For DeepSeek, the True Δ of +0.151 is borderline-significant after question-level bootstrap correction (p = 0.099). For GPT and Gemma, the True Δs of +0.053 and +0.072 do not survive clustered correction (bootstrap p = 0.239 and 0.259 respectively). The directional pattern - filter raises True faithfulness for all three judges - is real, but only the largest effects survive proper statistical scrutiny.

This finding is counterintuitive on its face. If a filter is a strong classifier (the statistical filter achieves F1 = 0.929 on held-out passage-level classification), removing the poison it identifies should make the judge's job easier, not harder. Section 4.5 tested two candidate mechanisms empirically and surfaced a third. This section unpacks what we found.

### Collateral damage is real but it is not the operative mechanism

Collateral damage - the filter removing supporting passages it should have spared - happens at a meaningful rate (roughly 17.5\

Section 4.5.2 tested this directly. Within the True subset, we partitioned triplets into those where the filter removed at least one supporting passage and those where it removed zero. If collateral damage drives the paradox, the support-removed group should show the largest positive Δ. The result is the opposite. In all three judges, when the filter removed supporting passages, faithfulness scores fell by 0.15 to 0.23. The judge correctly registered the missing evidence. The paradox - post-filter scores rising - emerges only in triplets where the filter spared the supporting passages.

So collateral damage exists as a phenomenon, but it operates in the opposite direction to the paradox. When it happens, the judge handles it the way an unbiased reasoner would: less evidence, less confidence, lower score. The aggregate True-subset Δ reported in Chapter 4 is the net of two opposing effects: a small negative pull from the 27\

### The operative mechanism is context clarification through distractor removal

What actually drives the paradox is a different effect entirely. The statistical filter, designed to identify poisoned passages, also removes a substantial fraction of the original HotpotQA distractor passages - passages that the dataset constructors deliberately included to make the multi-hop task harder. These distractors are noisy, topically loose, and embedding-distant from the question. They share surface features with the random_noise injection type, and the filter learns to remove them along with the actual poison.

The downstream effect on the judge is straightforward in retrospect. The post-filter context is shorter (mean 7.1 passages versus 11.6 at baseline, a 39\

Section 4.5.2 reports the stratification: when we partition True-subset triplets by the quartile of original distractors removed, the Δ rises monotonically with distractor removal in all three judges. In the lowest quartile (few distractors removed), there is no paradox. In the highest quartile (many distractors removed), the Δ reaches +0.25 to +0.46. The relationship is not a small statistical artefact; it is the dominant pattern in the data.

This corresponds to a length effect that is independently visible in the judge's scoring behaviour. The Pearson correlation between context length and faithfulness score is weak at baseline (r = 0.10 for GPT, 0.29 for Gemma, −0.08 for DeepSeek) and substantially stronger post-filter (r = 0.36, 0.47, 0.31). After the filter has run, judges score shorter contexts as more faithful - a behavioural pattern that is not present, or barely present, before filtering.

### Why this is a paradox in the practical sense

We continue to call this a paradox because the local effect of filtering (removing identified poisoned passages) and the global effect (judge reliability) move in opposite directions for a meaningful fraction of cases. But the mechanism is now sharper than the literature on filter-then-evaluate pipelines has assumed.

The standard intuition runs: poison degrades the residual, the filter removes poison, so the residual improves and the judge gets a better signal. This thesis shows that the residual produced by filtering is improved in a way that is hostile to careful judgment - it is shorter, cleaner-looking, more topically focused, and therefore more credible than the original noisy context, regardless of whether the surviving passages are actually faithful to the answer. The judge trusts the residual because it reads as well-formed, not because it actually contains the supporting evidence.

This reframes the engineering problem. Filter design has been optimised against passage-level classification metrics (precision, recall, F1) on the assumption that better classification produces better downstream judgment. The data here shows that the link from classification quality to downstream judgment is not monotonic. A filter that aggressively removes "noisy-looking" passages will produce residuals that judges find very convincing - including the ones that contain residual poison - which is exactly the wrong direction.

### Implications for filter design

Three implications follow.

First, filter design has been optimising the wrong objective. Classifier-style metrics measure how well the filter identifies poisoned passages. They do not measure how the filtered residual interacts with downstream judges, and the data shows that those two things can diverge substantially. The right metric is end-to-end judge response on a held-out test set with realistic poisoning rates, not classifier F1 on a held-out passage-level test set.

Second, the distinction between "removing poison" and "removing distractors" matters in deployment. Filters that aggressively prune off-topic content will catch some poison but will also strip out original retrieved noise - context that, paradoxically, may have been useful precisely because it gave the judge something to ground a sceptical reading in. Filters designed to preserve original context shape (re-ranking rather than removal, contradiction detection rather than per-passage classification) face different trade-offs that may be more favourable to downstream judgment.

Third, the dependency of the paradox on context length suggests that judges have a length-based credibility heuristic that filtering activates. This is not a mistake of the judges per se - shorter, more focused contexts are usually more trustworthy in non-adversarial settings - but it is exploitable when the residual is the product of a filter rather than of an honest retrieval. Judges aware of having been pre-filtered might calibrate differently, and Section 4.6's small qualitative justification analysis offers a hint that prompt-level interventions can shift this. Quantifying the effect at scale is future work.

## What the Cross-Judge Differences Mean

The three judges studied here are roughly contemporary and span the proprietary, open-weight, and cost-efficient segments of the LLM market. They differ in size, training data, and architectural choices, but all three are competent at standard NLP tasks. The differences observed in Chapter 4 are not differences of basic capability.

The differences are differences of scoring strategy. GPT's continuous calibration reflects a graded interpretation of the [0, 1] interval: the judge actively modulates its confidence as a function of evidence strength. Gemma's extreme leniency reflects a default-faithful prior: scores cluster near 1.0 unless evidence strongly contradicts the answer. DeepSeek's near-binary calibration reflects a categorical decision pattern: the judge has decided that the answer is faithful or it has decided that it is not, and the score reports the verdict at one of the extremes.

These three regimes are not equally good or equally bad. Each carries trade-offs.

GPT's continuous scoring is most useful for downstream pipeline design because it preserves the full interval as a confidence signal. A pipeline can route triplets with scores in [0.4, 0.7] to human review while passing triplets near the extremes. Gemma's leniency means high faithfulness scores are uninformative - they may indicate genuine faithfulness or they may indicate the default. Low scores from Gemma are valuable signal but rare. DeepSeek's near-binary scoring is the inverse problem: every score is a confident verdict, but the confidence is overconfident - the judge is not actually as certain as the score suggests, as evidenced by the substantial baseline FPR that accompanies the binary scoring style.

The cross-judge directional reversal on PoisonedRAG-style attacks is the most striking single observation from Chapter 4. GPT and Gemma find this attack hardest; DeepSeek finds it easiest. The reversal shows that the relationship between attack sophistication and detection difficulty is not monotonic across judges. An attack designed to be subtle (PoisonedRAG-style is engineered to look topically coherent and to embed retrieval anchors) can be the easiest to detect if the judge's scoring strategy happens to respond to a specific structural feature of the attack.

DeepSeek's near-binary scoring effectively functions as a contradiction detector on PoisonedRAG-style triplets. The malicious passage explicitly supports an answer that contradicts the ground-truth answer in the answer slot. Once DeepSeek detects the contradiction, its scoring style commits to a low score. Continuous-scoring judges weigh the contradiction against the topical fluency of the malicious passage and arrive at a high score. Same evidence, different verdicts.

Two implications follow.

Judge selection is a security-relevant decision. Most published evaluation work selects judges based on cost or availability and treats them as broadly interchangeable for the same task. The results here suggest, on a sample of three judges and a single retrieval-QA task, that a deployed pipeline's robustness to corpus poisoning can depend substantively on which judge it uses, and that the "right" judge depends on the threat model. We do not claim to have shown this for the LLM-as-a-Judge paradigm at large - three judges and one task are too narrow a base for that - but the directional point is concrete enough that practitioners should measure baseline robustness under representative adversarial conditions before committing to a judge for production use.

The literature's standard practice of reporting average results across judges may mask important per-judge differences. Section 4.1.3 would have looked very different if presented as a "mean baseline FPR across three judges": the directional reversal would have been hidden. Multi-judge studies should report per-judge results explicitly and should resist the temptation to summarise across calibration regimes that respond to attacks differently in kind.

We do not argue that one calibration regime is uniformly preferable to the others. Each has trade-offs, and the right choice depends on the threat model and the pipeline's downstream design. The practical recommendation is that pipelines deploying LLM-as-a-Judge should characterise the calibration of their chosen judge under representative adversarial conditions before deployment, and should account for that calibration when designing downstream routing or aggregation logic.

## Comparison with Existing RAG Defences

The RAG security literature contains several published defence mechanisms, all of them aimed at the generator. We briefly survey the main families and contrast them with the present work.

Detection-based defences (perplexity, embedding clustering) assume that adversarially crafted text exhibits unusual statistical properties. The original PoisonedRAG paper showed that the perplexity distributions of clean and malicious texts overlap heavily, and the PR-Attack framework deliberately minimises perplexity, rendering this defence essentially inert. Embedding clustering defences (e.g., TrustRAG) assume that malicious documents cluster distinctly in embedding space; this works at low poisoning rates but degrades when attackers outnumber benign passages. Both families operate at a similar abstraction level to one signal of our statistical filter, and our filter inherits their strengths and limitations on that dimension while compensating with other signals.

Aggregation-based defences (RobustRAG) run the generator independently on each retrieved passage and aggregate the results via voting. The defence is effective when poisoned passages are a minority of retrievals, but fails when attackers can guarantee retrieval of multiple poisoned passages. Aggregation operates at the generator side; it does not address judge reliability.

Adversarial training (RAGuard) trains the retriever or generator to be robust to adversarial documents. The approach is elegant but requires representative attack data, which limits transferability to novel attack types. It does not transfer to the judge.

Process-based defences (InstructRAG, AstuteRAG, Discern-and-Answer) modify the generation process itself, asking the LLM to reason about reliability or to explicitly reject poisoned context. They are effective against denial-of-service attacks but less effective against targeted poisoning.

Lightweight ML post-retrieval (RAGDefender) applies a lightweight ML filter to retrieved documents before they reach the generator. It limits Attack Success Rate to single-digit percentages on PoisonedRAG-style attacks. The architecture is closer to ours than the others - a lightweight filter between retrieval and downstream consumption - but the consumer in their work is the generator and the metric is ASR.

The pre-filter studied in this thesis is closest in spirit to RAGDefender. The differences are: our consumer is the LLM-as-a-Judge rather than the generator; our metric is judge FPR across three RAGAS dimensions rather than ASR; we evaluate two filter architectures (statistical and LLM-based) for cross-architecture comparison; and we use the True/Survived decomposition to measure judge reliability under operationally correct labelling.

The empirical finding distinguishing this work is that, when measured against judges rather than generators, the same filter family that succeeds for generator defence does not reliably help. This is not a refutation of the existing defence literature but an extension into a setting that has not previously been measured. Defending the judge is a distinct problem, and the present work establishes that it is not solved by the current generator-focused defence toolbox.

## Mitigation Strategies the Results Point Toward

The empirical results suggest several directions for further mitigation work. We outline four here, in roughly increasing order of effort to implement.

**Hybrid filter architectures.** The statistical filter and the LLM filter detect partially complementary phenomena. The statistical filter has near-perfect recall on PoisonedRAG-style and somewhat lower recall on adversarial fact. The LLM filter has the opposite profile - strong on PoisonedRAG-style and adversarial fact, weak on random noise. A hybrid that uses the statistical filter as a first-pass sieve and routes ambiguous triplets to the LLM filter would, in principle, combine the strengths of both. Whether the gain in classification F1 translates to improved end-to-end judge reliability is the open question. The mechanism analysis in Section 5.1 suggests the answer is not automatic: a hybrid that achieves higher recall may produce shorter, cleaner-looking residuals through more aggressive distractor removal, recreating the paradox at a higher operating point.

**Calibration-aware judge selection and ensembling.** Section 5.2 noted that the three judges have qualitatively different calibration regimes and respond differently to attack types. A practitioner deploying an LLM-as-a-Judge pipeline could explicitly select for calibration properties suited to the threat model. For threat models dominated by PoisonedRAG-style attacks, a near-binary judge with strong contradiction detection (analogous to DeepSeek) is the better choice. For threat models dominated by random noise or factual-edit attacks, a continuous-scoring judge with confidence-abstention support (analogous to GPT) is preferable. A pipeline could even ensemble different calibration regimes - not for variance-based detection (which Section 4.7 ruled out) but for coverage of different failure modes. The cost is computational: running three judges per triplet rather than one.

**Confidence abstention for continuous-scoring judges.** GPT's continuous calibration creates a natural decision band. A pipeline could abstain on triplets with faithfulness scores in (e.g.) [0.4, 0.7] and route them to human review. Score distributions in Section 4.1.2 suggest a 10–15\

**Judge-side hardening.** If pre-filtering cannot reliably produce residuals that judges handle well, the alternative is to harden the judge directly. Possible directions include: training judges on adversarial examples so they learn to recognise polished poison; modifying the scoring prompt to require explicit cross-passage consistency reasoning before scoring; adding a meta-evaluation layer that scores not just the answer but the trustworthiness of the context. The work in Section 4.6 (justification analysis) shows that explicit poison-aware prompting shifts judge behaviour substantially on PoisonedRAG-style attacks. This is the most promising single direction the data supports, and it does not require any change to filter design.

These directions are not mutually exclusive. A production pipeline could combine all four: a statistical filter as the first pass, a triplet-level LLM filter for ambiguous cases, confidence abstention on continuous judges, and judge-side prompting that requires explicit consistency reasoning. The empirical question is which combinations achieve the best cost-benefit ratio. That is an evaluation programme of its own.

## Limitations

The study has several methodological limitations worth recording explicitly.

**Single-domain evaluation.** All experiments use HotpotQA, and the findings should be read as "how three LLM judges behave on multi-hop retrieval QA under controlled poisoning" rather than as generic claims about LLM-as-a-Judge. The attack types are tuned to a multi-passage retrieval setting where supporting evidence is distributed across multiple cooperating passages. The pre-filter signals (cosine anomaly, answer-span recall, cross-encoder relevance) exploit properties of this setting. Whether the headline paradox - filtering raises True-category faithfulness - generalises to single-hop QA, dialogue evaluation, summarisation faithfulness scoring, or open-domain agentic settings is genuinely unknown. The closest reasonable extrapolation is to other multi-passage retrieval tasks; extrapolation beyond that is speculation. We treat single-task evaluation as the most consequential scope limit on the conclusions drawn here.

**Three judges, not a representative panel.** Our judge sample is three contemporary LLMs from three different providers (one proprietary, one open-weight, one cost-efficient). With n=3 we cannot make distributional claims about how LLM judges as a class behave. We can demonstrate that *these* three judges occupy distinct calibration regimes, that *these* three respond differently to the most sophisticated attack class, and that *these* three exhibit the True-subset paradox under filtering. Whether a fourth or fifth judge - a reasoning model, a frontier-tier model, a fine-tuned domain-specific judge - would extend, weaken, or break the patterns is an empirical question we cannot answer from our data. Statements throughout Chapter 5 such as "judge architecture matters as much as judge size" should be read as suggested by the panel, not established by it. A larger panel that includes reasoning models is the most direct extension of this work.

**No human-annotated ground truth.** The thesis evaluates LLM judges against each other and against the construction-time poisoning labels. It does not validate the judges against human assessment of faithfulness on the same triplets. The standard RAGAS premise that LLM judges proxy human judgement is therefore inherited from the literature rather than verified in this work. Our findings are best read as "how three LLM judges' verdicts shift under attack" rather than "how judgement quality, in any human-validated sense, changes under attack." A small human-annotated subset (say 50 triplets, 3 annotators) would let future work make the stronger claim. We treat this as the second most consequential scope limit on the conclusions.

**Question-level clustering in significance tests.** The 1,200 poisoned triplets derive from 100 base questions, with each question contributing twelve triplets (three injection types × four noise levels). Triplets sharing a base question are not independent: they share the question text, the ground-truth answer, and the original supporting passages. We ran a question-clustered bootstrap (1,000 iterations resampling at the base-question level) for the McNemar tests reported in Section 4.4.5. The bootstrap-corrected p-values are reported alongside the uncorrected values in Table [tab:mcnemar-true-faithfulness] and discussed in the surrounding text. The directional findings hold under bootstrap correction, but the precision of the True-subset significance claims drops substantially: GPT and Gemma's True-subset effects do not survive clustered correction (bootstrap p = 0.239 and 0.259), only DeepSeek's True-subset effect approaches significance (bootstrap p = 0.099), and the most robust significant effect is the DeepSeek Survived-subset shift driven by PoisonedRAG-style triplets (bootstrap p = 0.012). We treat the bootstrap-corrected p-values as the more honest characterisation of the data.

**Filter design choices.** The statistical filter uses five engineered signals aggregated via XGBoost. The signal set is informed by the attack types in our dataset and by the multi-hop QA setting; alternative signal sets (e.g., learned representation-level features, transformer-based passage classifiers) could plausibly produce different results. The Mistral filter uses a single triplet-level prompt; alternative prompt designs could plausibly produce different recall and precision profiles.

**Single dataset version.** The experiments use a single version of the Poisoned Evaluation Dataset, with 100 base questions and 2,400 triplets. Sample-size considerations limit the precision of per-cell estimates, particularly for the smaller subsets (True n = 576 partitioned across 3 judges × 3 conditions × 2 injection types means roughly 96 triplets per granular cell). Larger datasets would tighten confidence intervals but are unlikely to change the headline directions.

**The True/Survived split is structurally asymmetric across injection types.** The decomposition cleanly separates "support overwritten" from "support intact" for replacement-based attacks (Types 1 and 2). For additive attacks (Type 3, PoisonedRAG-style), the decomposition is degenerate: all such triplets are in the Survived subset by construction. This is methodologically appropriate - additive injection by definition does not overwrite anything - but it means the True/Survived dichotomy is most informative on replacement attacks and less informative on attacks that operate by augmentation rather than replacement. Other reasonable decompositions exist (for example, partitioning by whether the judge effectively had access to a majority of supporting evidence post-filter); we report the structural definition because it is computable from labels rather than from judge behaviour, but we do not claim it is the only sensible one.

**Production deployability of the True/Survived analysis.** The decomposition into True and Survived requires per-passage poisoning labels that are available because we constructed the dataset. Real deployments do not have such labels. A practitioner cannot directly compute True-category FPR in production; they can only measure judge behaviour and infer how the filter is changing what the judge sees. This is a methodological limit on directly importing the present work's evaluation procedure into a production monitoring pipeline. What practitioners can do is apply representative test datasets at evaluation time (such as the one released with this thesis) to estimate how a candidate filter affects judge reliability before deploying.

**Justification analysis sample size.** The justification analysis is based on 28 triplets scored by a single judge under three prompt conditions. This is a qualitative supplement, not a quantitative claim, and we treat it as such throughout.

**Mechanism evidence is empirical but bounded.** Where Chapter 4 and Chapter 5 attribute filter-induced shifts to mechanisms, the principal mechanism claim (context clarification through distractor removal) is supported by direct empirical evidence: a counterfactual that refutes the collateral-damage hypothesis, a stratification that shows monotonically rising Δ with distractor-removal volume, and a length-correlation analysis showing that judges score shorter post-filter contexts as more faithful. These tests rule out the most obvious alternative explanations but do not constitute a fully causal claim. The mechanism is best characterised as the explanation most consistent with the available evidence after testing the principal alternatives. A randomised intervention varying context length while holding poison content constant would tighten this further but is outside the scope of the present work.

## Practical Implications

The findings translate into several concrete recommendations for practitioners building LLM-as-a-Judge pipelines for RAG.

First, baseline judge FPRs on poisoned context are unacceptably high for security-critical deployments. Pipelines that rely on a single LLM judge to make automated quality decisions on RAG outputs are exposed to corpus-level adversarial pressure, and the exposure is not small (baseline faithfulness FPRs of 41\

Second, lightweight upstream filtering does not solve the problem. The intuitive practitioner response to judge vulnerability is to filter the context before it reaches the judge. The empirical results in this thesis show that this intuition is wrong in current designs. Pre-filtering changes the residual context in ways that can make judges more confident in poisoned content, not less. A pipeline that adds pre-filtering and assumes downstream reliability has improved is making an assumption the data does not support.

Third, judge selection is a security decision. The three judges in our panel respond differently to the most sophisticated attack class. The practical implication is that pipelines should not select judges purely on cost, and should run baseline robustness measurements under representative adversarial conditions before committing to a judge for production use. The Poisoned Evaluation Dataset released with this thesis can serve as a starting point for such measurements.

Fourth, multi-judge ensembling is not a substitute for filtering. Pipelines that deploy multiple judges and rely on disagreement for detection will not see the gain they expect; the disagreement signal is far too small to support thresholded detection. Ensembling can be valuable for other reasons (calibration averaging, redundancy against API failures) but not for poisoning detection.

Fifth, judge-side hardening is a more promising direction than filter optimisation. The justification analysis suggests that explicit poison-aware prompting can substantially shift judge behaviour on PoisonedRAG-style attacks. Practitioners with pipeline access to judge prompts have a tractable mitigation that does not require building or training a filter. Whether this scales to production conditions is an empirical question, but the early signal is positive.

Sixth, evaluation methodology matters. Practitioners measuring filter effectiveness should use the True/Survived decomposition, or some operational equivalent, to avoid being misled by classifier-level F1. A filter that achieves high F1 on a held-out test split may produce post-filter residuals that judges handle worse than the unfiltered context. The right metric is the one that measures end-to-end judge reliability, not classification metrics on the filter in isolation.

---


<!-- Source: 06_conclusion.tex -->

# Conclusion

This thesis set out to answer two questions about LLM-as-a-Judge pipelines for Retrieval-Augmented Generation. First, how badly are contemporary LLM judges fooled by corpus poisoning, and how does that vulnerability vary with attack type, attack severity, and the choice of judge? Second, can a lightweight upstream pre-filter mitigate the failure?

The experimental programme answered both questions on data. The vulnerability is real and substantial: baseline faithfulness False Positive Rates across three contemporary judges range from 0.412 to 0.699, and the three judges occupy distinct calibration regimes that respond differently to attack class. The most adversarial attack class (PoisonedRAG-style) produces a directional cross-judge reversal, where the judges that find this attack hardest are the ones with continuous or extreme-leniency scoring, and the judge that finds it easiest is the one with near-binary scoring. The vulnerability is not uniform across the LLM-as-a-Judge paradigm; it is structured by judge architecture and attack type.

The second answer is more uncomfortable. Pre-filtering, in the two designs we evaluated, does not reliably mitigate the vulnerability. Decomposing originally-poisoned triplets into True (supporting passages overwritten) and Survived (supporting passages intact) reveals that filtering directionally raises True-category faithfulness for all three judges under the statistical filter (Δ +0.053 GPT, +0.072 Gemma, +0.151 DeepSeek). The directional pattern is consistent across judges, but the statistical robustness varies: under question-clustered bootstrap, only DeepSeek's True-subset effect approaches significance (corrected p = 0.099), while GPT's and Gemma's True-subset effects do not survive clustered correction (corrected p = 0.239 and 0.259 respectively). The largest and most robust single end-to-end effect is in the Survived subset for PoisonedRAG-style triplets, where DeepSeek's mean faithfulness rises by 0.377 under the statistical filter (clustered bootstrap p = 0.012). On context relevance the paradox is concentrated on DeepSeek (Δ +0.180); on answer relevance changes are small across all cells.

The operative mechanism is context clarification through distractor removal. The statistical filter, designed to remove poisoned passages, also removes a substantial fraction of the original HotpotQA distractor passages. The residual is shorter (mean 7.1 versus 11.6 passages, a 39\

These two findings - that judges are vulnerable, and that filtering paradoxically worsens the worst case - together support a clear practical conclusion: pre-filtering, in its current form, is not a sufficient defence for LLM-as-a-Judge deployments under adversarial conditions. Practitioners deploying RAG evaluation pipelines in security-critical contexts should not assume that adding upstream filtering improves downstream reliability. The empirical evidence suggests it can do the opposite.

## Restating the Contributions

The four contributions of the thesis can be restated concisely.

The Poisoned Evaluation Dataset (v2) provides 2,400 question-context-answer triplets from HotpotQA across three injection types and four noise levels, with explicit per-passage poisoning indices. The PoisonedRAG-style injection is implemented carefully, specifying a target wrong answer per question and using the full retrieved context. The dataset is released as a reusable benchmark for further work on judge robustness and RAG security.

The multi-metric meta-evaluation framework characterises three contemporary LLM judges across all three RAGAS dimensions under controlled poisoning. The framework reports judge behaviour by injection type, by noise intensity, and by score-distribution shape. It is reusable for future judge-robustness studies and provides the structural basis for the empirical findings reported here.

The multi-signal statistical pre-filter combines five engineered signals through XGBoost into a passage-level classifier. The filter is trained on a stratified split of the Poisoned Evaluation Dataset and evaluated on a held-out test split. It achieves F1 of 0.929 with near-zero clean false-positive rate. A complementary triplet-level Mistral filter is implemented for cross-architecture comparison.

The end-to-end evaluation methodology measures how each pre-filter affects judge reliability. It uses the True/Survived decomposition to distinguish between cases where the attack hit the supporting context and cases where it did not, providing an operationally meaningful measure of post-filter judge failure that is robust to the labelling artefacts that naive analyses suffer from. The methodology surfaces a counterintuitive empirical pattern - filter-induced score increases driven by distractor removal rather than by collateral support loss - that current filter design does not anticipate.

## Answers to the Research Questions

**RQ1: How badly are contemporary LLM judges fooled by corpus poisoning?**

Substantially. Baseline faithfulness FPRs for the three judges studied range from 0.412 (DeepSeek) to 0.555 (GPT) to 0.699 (Gemma). Context relevance FPRs range from 0.326 to 0.860 and answer relevance FPRs from 0.652 to 0.889. The three judges occupy three qualitatively distinct calibration regimes (near-binary, continuous, extreme-leniency) and respond differently to attack class. PoisonedRAG-style is the hardest attack for GPT and Gemma but the easiest for DeepSeek. Vulnerability decreases monotonically with noise level for all three judges within each injection type at baseline.

**RQ2: How does that vulnerability vary with judge calibration?**

It varies substantially. The three calibration regimes correspond to three different scoring strategies, each with distinct trade-offs in detection profile, abstention support, and response to filter intervention. Near-binary judges effectively function as contradiction detectors and handle PoisonedRAG-style attacks well; continuous-scoring judges weigh evidence in graded fashion and are more vulnerable to subtle attacks. Extreme-leniency judges have a default-faithful prior that makes them most vulnerable across the board.

**RQ3: Can a lightweight upstream pre-filter mitigate the failure?**

In the two designs evaluated, no - at least not reliably. The statistical filter is a strong classifier in isolation (F1 = 0.929 on the held-out test split) and removes most poisoned passages. But end-to-end, decomposed by True/Survived, the filter directionally raises the rate at which judges score concentrated-poison contexts as faithful. The directional pattern is consistent across all three judges (+0.053 GPT, +0.072 Gemma, +0.151 DeepSeek mean faithfulness on the True subset), but statistical robustness varies under question-clustered bootstrap: only DeepSeek's True-subset effect approaches significance after correction (corrected p = 0.099); GPT's and Gemma's do not (corrected p = 0.239 and 0.259). The largest single cell-level effect, and the most statistically robust, is on the Survived subset for PoisonedRAG-style attacks, where DeepSeek's mean faithfulness rises by 0.377 (clustered bootstrap p = 0.012). The LLM filter (Mistral) shifts judge response materially only for DeepSeek. On context relevance the paradox is concentrated on DeepSeek; on answer relevance changes are minimal across the board.

**RQ4: What mechanisms explain the failure of pre-filtering to mitigate?**

The mechanism is now empirically grounded. The filter removes original HotpotQA distractor passages at high rates, producing a shorter and more topically focused residual that judges score as more faithful regardless of content. The collateral-damage hypothesis (filter removes supporting passages, judge loses evidence, paradox emerges) is empirically refuted by a direct counterfactual: when the filter removed supporting passages, scores fell across all three judges, exactly as the alternative hypothesis would predict for an unbiased reasoner. The paradox is driven by the filter's incidental clarification of the residual context, not by the filter's removal of supporting evidence.

## Future Work

Four directions emerge from the present work as the most natural extensions.

**Cross-task generalisation.** All experiments in this thesis use HotpotQA, a multi-hop QA benchmark with structural properties (multiple cooperating supporting passages, distractor passages, explicit answer supervision) that informed both the attack design and the filter design. Whether the headline paradox - filtering raises True-category faithfulness - generalises to single-hop QA, dialogue evaluation, summarisation faithfulness scoring, or open-domain agentic evaluation is unknown. Replicating the dataset construction, filter design, and True/Survived analysis on at least one structurally different evaluation task is the most direct test of the thesis's central claim. We expect the distractor-removal mechanism to transfer to any task where retrieval delivers a mix of supporting passages and topically loose distractors and where filtering acts on length-shortening principles. Tasks without distractors in the retrieval set (single-hop QA with high-precision retrieval, structured-data evaluation) may not show the same effect at all, since the filter would have less original noise to remove.

**Judge-side hardening.** The qualitative justification analysis in Section 4.6 (n=28 triplets, single judge) suggests that explicit poison-aware prompting can shift judge behaviour on the engineered attack types (mean faithfulness on PoisonedRAG-style drops from 0.800 to 0.492 on the subset; on adversarial fact from 0.758 to 0.558). The sample is too small to support strong quantitative claims, but the directional shift is large enough to be worth investigating rigorously. We see this as the most empirically promising direction *suggested* by our results, with the caveat that the suggestion rests on a small qualitative subset rather than a full evaluation. Quantifying the effect at scale and characterising its limits across attack types and judges is a tractable research programme.

**Hybrid filter architectures matched to the True/Survived decomposition.** The statistical filter and the LLM filter detect partially complementary phenomena. A hybrid that combines both would, in principle, achieve higher recall - but the paradox analysis suggests that higher recall may not translate into improved end-to-end judge reliability if the distractor-removal mechanism persists. The relevant filter-design question is whether one can remove poison without also removing original distractors, since the latter is what produces the cleaner-looking residual that judges over-trust. Filter architectures that operate on a different abstraction level (re-ranking rather than removal, contradiction detection that flags inconsistency without modifying content, or filters trained explicitly on the distinction between adversarial poison and naturally-occurring distractors) are promising structural alternatives.

**Confidence abstention and downstream pipeline design.** Continuous-scoring judges produce a natural decision band that supports human-in-the-loop review. We did not implement and measure abstention strategies in the present work, but the score distributions suggest that a 10–15\

Beyond these four directions, the broader research programme concerns the trustworthiness of automated evaluation in adversarial settings. As LLM-as-a-Judge becomes more widespread for production RAG evaluation, the need for evaluation infrastructure that is itself robust will only grow. The present thesis contributes a working pre-filter, a reusable dataset, an evaluation methodology, and an honest empirical characterisation of where current defences succeed and where they fail on a single multi-hop QA task. The larger problem - making evaluation pipelines as adversarially robust as the systems they evaluate, across the diverse settings in which RAG is deployed - remains open.

## Closing Remarks

The Poisoned Evaluation Dataset, the pre-filter pipeline implementation, and the full experimental codebase are released alongside this thesis. The work was conducted under significant resource constraints, and we hope that the released artefacts make it easier for follow-up researchers to build on what has been demonstrated here without re-running the full experimental programme.

The empirical result of this work is uncomfortable. LLM judges are imperfect; corpus poisoning exploits their imperfections; a well-designed filter that removes the poison the dataset constructor injected does not, in the designs evaluated here, restore judge reliability - and may worsen it on the cases where reliability matters most. What this thesis adds is the rigorous experimental measurement of these claims across three contemporary judges, three attack types, and four noise levels on a single multi-hop QA task, alongside an analytical framework (the True/Survived decomposition) that distinguishes the cases where the filter succeeds from the cases where it does not. The single-task scope is a real limitation: the headline paradox should be read as established for HotpotQA-like multi-hop retrieval QA evaluation, with cross-task generalisation as the next experimental step rather than a finished claim.

The findings argue against complacency in deploying LLM-as-a-Judge in adversarial settings, and they argue for treating the judge as a first-class object of security analysis rather than as a downstream consumer that inherits its trustworthiness from upstream defences. The hope is that by establishing both the vulnerability and the limits of current mitigations on shared benchmark conditions, the work contributes to building automated evaluation pipelines that are demonstrably trustworthy in conditions where trust matters.
