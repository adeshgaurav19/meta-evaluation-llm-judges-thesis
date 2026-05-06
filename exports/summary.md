# Analysis Summary

## 1. Baseline characterisation

**GPT** faithfulness: clean=0.716 (n=1200), poisoned=0.519 (n=1200), FPR=0.555

**GEMINI** faithfulness: clean=0.957 (n=1198), poisoned=0.686 (n=1199), FPR=0.699

**DEEPSEEK** faithfulness: clean=0.780 (n=1200), poisoned=0.407 (n=1200), FPR=0.412


## 2. Paradox direction and magnitude (faithfulness, True category)

**GPT**: baseline True=0.391, postfilter True=0.444 (PARADOX Delta=+0.053)

**GEMINI**: baseline True=0.442, postfilter True=0.514 (PARADOX Delta=+0.072)

**DEEPSEEK**: baseline True=0.314, postfilter True=0.465 (PARADOX Delta=+0.151)


## 3. Filter audit summary

**Mean supporting passages lost as collateral (stat filter): 0.3450**


  adversarial_fact: recall=0.551, precision=0.607, collateral_loss=0.328

  poisonedrag_style: recall=0.296, precision=0.306, collateral_loss=0.393

  random_noise: recall=0.554, precision=0.605, collateral_loss=0.315


## 4. McNemar significance (faithfulness, True category)

  GPT baseline->postfilter: chi2=8.16, p=0.0043 **p<0.05**

  GPT baseline->postfilter_llm: chi2=0.42, p=0.5152 n.s.

  GEMINI baseline->postfilter: chi2=10.93, p=0.0009 **p<0.05**

  GEMINI baseline->postfilter_llm: chi2=0.19, p=0.6650 n.s.

  DEEPSEEK baseline->postfilter: chi2=34.40, p=0.0000 **p<0.05**

  DEEPSEEK baseline->postfilter_llm: chi2=15.57, p=0.0001 **p<0.05**


## 5. Inter-judge variance

Pearson r (GPT vs DeepSeek baseline faithfulness) = 0.514 (p=0.0000, n=2400)

Mean per-triplet std across judges: clean=0.205, poisoned=0.256
