# Figure captions

**fig01_baseline_calibration.png**
Histograms of baseline faithfulness scores for clean and poisoned triplets, one panel per judge (GPT-5.4-nano, Gemma-4-26B, DeepSeek-V3.2). The dashed vertical line marks the 0.5 threshold; poisoned triplets scoring above this threshold constitute false positives.

**fig02_paradox_headline.png**
Mean faithfulness by category (Clean, Standard, True, Survived) across the three judging conditions (Baseline, Postfilter-stat, Postfilter-LLM). The Delta annotations on True-category bars show the change from the baseline condition; a positive value indicates the paradox - statistical filtering increases the mean faithfulness of the most severely compromised triplets.

**fig03_true_mean_by_noise.png**
True-category mean faithfulness as a function of noise level (0.2-0.8) for each judge. Annotations on the statistical-postfilter line at noise levels 0.2 and 0.8 highlight the cross-over effect: low-noise attacks become relatively more dangerous after filtering than high-noise attacks.

**fig04a_per_injection_faithfulness.png**
True-category faithfulness for each combination of judge (columns) and replacement-based injection type (rows) under Baseline, statistical-Postfilter, and LLM-Postfilter conditions. PoisonedRAG-style is omitted because additive attacks never overwrite supporting passages and therefore have no True-category cases.

**fig04b_per_injection_context.png**
As Figure 4a, but for context relevance scores.

**fig04c_per_injection_answer.png**
As Figure 4a, but for answer relevance scores.

**fig05_filter_behaviour.png**
Effect of each filter on the True/Survived split of originally-poisoned triplets, shown before (darker bars) and after (lighter bars) filtering. For the LLM filter, Mistral's triplet-level flag determines routing; the passage split is unchanged since the LLM filter does not modify context content.

**fig06_inter_judge_variance.png**
Distribution of per-triplet faithfulness standard deviation across the three judges under baseline conditions, separated by clean and poisoned triplets. The annotated Pearson r characterises the correlation between judge disagreement and ground-truth poisoning status.

**fig07_mcnemar.png**
McNemar chi-square statistics testing whether the proportion of faithfulness false positives changes significantly between the baseline and each postfilter condition, restricted to the True category. Asterisks denote significance: * p < 0.05, ** p < 0.01, *** p < 0.001; n.s. = not significant.

**fig08_filter_audit.png**
Mean per-triplet passage counts by outcome type and injection type for the statistical filter. The LLM filter is omitted because it makes triplet-level decisions and does not remove individual passages. The boldly annotated collateral-loss bars (orange) quantify how often the statistical filter mistakenly removes non-poisoned passages that originally supported the correct answer.
