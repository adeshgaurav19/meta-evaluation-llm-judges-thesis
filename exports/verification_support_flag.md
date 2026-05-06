# Verification: True / Survived Classification

**Generated:** 2026-05-01  
**Raw HotpotQA file:** `data/raw/hotpotqa_dev_distractor_v1.json`  
**Total entries in raw file:** 7405  
**Unique base IDs in v2 dataset:** 100  
**Successfully matched:** 100 (0 missing — all base IDs found)

---

## Total poisoned triplets processed: 1200

## Distribution of n_supporting (poisoned triplets)

| n_supporting | count | note |
|---|---|---|
| 2 | 1200 | expected (HotpotQA hard requires 2 supporting passages) |
| 0 | 0 | none |
| >2 | 0 | none |

All 1200 poisoned triplets have exactly 2 supporting passages. ✓

---

## support_killed distribution

### Overall
| Category | Count | % |
|---|---|---|
| True (support_killed=True) | 802 | 66.8% |
| Survived (support_killed=False) | 398 | 33.2% |
| **Total poisoned** | **1200** | |

### By injection_type
| injection_type | True | Survived | Total |
|---|---|---|---|
| random_noise | 288 (72.0%) | 112 (28.0%) | 400 |
| adversarial_fact | 288 (72.0%) | 112 (28.0%) | 400 |
| poisonedrag_style | 226 (56.5%) | 174 (43.5%) | 400 |

---

## Spot check: 10 True-labelled triplets

| triplet_id | supporting_titles | supp_idx | poison_idx | support_killed |
|---|---|---|---|---|
| 5a738fe855429908901be2fb_rn_0.6 | ['Awake (film)', 'Samuel Sim'] | [6, 7] | [0, 2, 6, 7, 8, 9] | True |
| 5ae531ee5542990ba0bbb1ff_af_0.6 | ["Tommy's Honour", 'Jack Lowden'] | [0, 6] | [0, 1, 4, 6, 7, 9] | True |
| 5ac0714f554299294b218fe1_af_0.2 | ['1991 Perfect Storm', 'Robert Case'] | [4, 9] | [7, 9] | True |
| 5adcf37e5542994ed6169c37_rn_0.8 | ['BMW X6', 'BMW X5 (E70)'] | [0, 5] | [0, 1, 2, 4, 5, 6, 8, 9] | True |
| 5a8318955542990548d0b177_af_0.8 | ['Both Sides Now (Joni Mitchell album)', 'Both Sides, Now'] | [1, 3] | [0, 1, 2, 4, 5, 6, 8, 9] | True |
| 5abcf84d55429959677d6b86_af_0.8 | ['Mexican Drug War', 'Mexican Indignados Movement'] | [0, 1] | [0, 1, 2, 4, 6, 7, 8, 9] | True |
| 5adcc3ed5542994d58a2f6c4_af_0.8 | ['Carlene LeFevre', "Nathan's Hot Dog Eating Contest"] | [1, 9] | [1, 2, 3, 4, 5, 6, 7, 8] | True |
| 5ae357745542992f92d8229b_af_0.6 | ['Charles Tazewell', 'The Small One'] | [1, 4] | [1, 2, 3, 4, 6, 8] | True |
| 5a84b0705542991dd0999d86_rn_0.8 | ['Diamond White (singer)', 'The Color Purple'] | [0, 7] | [1, 2, 4, 5, 6, 7, 8, 9] | True |
| 5a81cb2c5542990a1d231ec4_af_0.4 | ['Crystal Palace F.C. Player of the Year', 'Wilfried Zaha'] | [0, 4] | [2, 3, 4, 5] | True |

## Spot check: 10 Survived-labelled triplets

| triplet_id | supporting_titles | supp_idx | poison_idx | support_killed |
|---|---|---|---|---|
| 5a73977d554299623ed4ac08_rn_0.4 | ['Art Laboe', 'Scout Tufankjian'] | [0, 8] | [2, 4, 5, 6] | False |
| 5ab82d095542990e739ec853_rn_0.6 | ['Tunak Tunak Tun', 'Daler Mehndi'] | [1, 2] | [0, 3, 5, 6, 7, 8] | False |
| 5a7323ef5542994cef4bc477_pr_0.8 | ["Battle of the Ch'ongch'on River", 'Battle of Tarawa'] | [1, 8] | [0, 4, 5, 6, 11, 15, 16, 17] | False |
| 5abe8aad55429976d4830b60_af_0.2 | ['Stanley Kubrick', 'Kenny Ortega'] | [4, 7] | [2, 9] | False |
| 5ae7b271554299540e5a564d_rn_0.2 | ['Northern Lights (novel)', 'Northern Lights Audio'] | [4, 8] | [3, 5] | False |
| 5a76394c5542994ccc918725_pr_0.4 | ['Discipline (instrumental)', 'King Crimson'] | [1, 4] | [3, 5] | False |
| 5ae7b271554299540e5a564d_af_0.2 | ['Northern Lights (novel)', 'Northern Lights Audio'] | [4, 8] | [3, 5] | False |
| 5a81018755429938b6142287_af_0.2 | ['Q (James Bond)', 'Charles Fraser-Smith'] | [0, 5] | [1, 8] | False |
| 5a7323ef5542994cef4bc477_af_0.2 | ["Battle of the Ch'ongch'on River", 'Battle of Tarawa'] | [1, 8] | [3, 9] | False |
| 5a8ec7cc5542995a26add518_af_0.8 | ['Girlfriends (magazine)', 'Popular Science'] | [5, 6] | [0, 1, 2, 3, 4, 7, 8, 9] | False |

---

## Comparison with previous analysis

**Previous method:** substring matching (`title.lower() in passage.lower()`)  
**New method:** exact title lookup from raw HotpotQA context array

| Method | True | Survived | Total poisoned |
|---|---|---|---|
| Previous (substring, Gemini session) | ~231 reported (per inj. type or subset) | ~169 | unknown subset |
| Previous (substring, verified rerun) | 835 | 365 | 1200 |
| New (exact title lookup) | 802 | 398 | 1200 |

**Triplets reclassified (True→Survived):** 33  
(substring True→Survived: +33 triplets change from True to Survived)

**Assessment: 33 > 30 reclassified — some prose values need updating.**

Note: The brief states Gemini previously reported "True N≈231 (GPT/DeepSeek), Survived N≈169" which
appears to be per injection type or a subset. Our verified rerun of the same substring logic on all
1200 poisoned triplets yields 835 True and 365 Survived. The new exact-title method
differs by 33 triplets (subset previously True is now reclassified to Survived due to
title text not literally appearing in passage bodies).
