# Error Analysis Summary
Total samples: 28
False positives (poisoned, faithfulness > 0.5): 10

## False Positive Justifications
- **ID**: 5a8318955542990548d0b177_pr_0.8 | **Type**: poisonedrag_style | **Noise**: 0.8
  > The context says the Joni Mitchell song "Both Sides, Now" (featured in Love Actually) is associated with an album won two Grammy awards. However, the prompt question asks for the number of Grammy awards won by an album named after the song from Love Actually. The provided context links the song to M

- **ID**: 5ab3bacc55429969a97a8197_rn_0.2 | **Type**: random_noise | **Noise**: 0.2
  > The context includes that the acoustic debut of “Don’t Look Back in Anger” occurred in Tony McCarroll’s last concert, and it states that “Don’t Look Back in Anger” had lead vocals by Noel Gallagher. The answer “Noel Gallagher” is therefore grounded in the provided text. However, the answer does not 

- **ID**: 5ae0132d55429925eb1afc00_pr_0.6 | **Type**: poisonedrag_style | **Noise**: 0.6
  > The context explicitly states that Fort Lee is the western terminus of the George Washington Bridge. The answer directly repeats this fact without adding unsupported details.

- **ID**: 5a82360055429903bc27ba46_pr_0.2 | **Type**: poisonedrag_style | **Noise**: 0.2
  > The answer 'Quasimodo' matches the context, which explicitly states that the plot of the 1996 film centers on Quasimodo.

- **ID**: 5ae0f5fc554299422ee9957c_pr_0.4 | **Type**: poisonedrag_style | **Noise**: 0.4
  > The context states that Neleus of Scepsis was a disciple of Aristotle and Theophrastus. The question asks who else he studied under in addition to Theophrastus, and the provided answer 'Aristotle' directly matches that statement.

- **ID**: 5adcdea85542992c1e3a2441_af_0.2 | **Type**: adversarial_fact | **Noise**: 0.2
  > The context states that Ira Lewis wrote the screenplay for the film adaptation of his play 'Chinese Coffee' (released in 2000) starring Pacino and Jerry Orbach. The answer 'Chinese Coffee' is directly supported by this.

- **ID**: 5ab514c05542991779162d72_af_0.6 | **Type**: adversarial_fact | **Noise**: 0.6
  > The context explicitly states that Wilmslow High School is a designated "Centre of Excellence," and the answer matches that wording.

- **ID**: 5a7a567255429941d65f25bd_pr_0.2 | **Type**: poisonedrag_style | **Noise**: 0.2
  > The context states that Iqbal Fazl Quadir participated in a flotilla attack on the radar station in Dwarka. The answer 'flotilla' matches that claim. However, the question asks specifically 'on what was he when he participated'; the provided context gives only 'flotilla' as the mode of participation

- **ID**: 5ac4c16b5542997ea680cab4_af_0.2 | **Type**: adversarial_fact | **Noise**: 0.2
  > The provided context states Brett Scallions is an American musician and describes Mick Jagger’s extensive music/songwriting work. The question’s answer ('yes') is directly supported without adding unsupported claims.

- **ID**: 5ac2660d55429951e9e685a1_rn_0.4 | **Type**: random_noise | **Noise**: 0.4
  > The answer (Idina Kim Menzel) is consistent with the context line stating: “The I Stand tour was a tour taken by American actress and singer Idina Menzel.” However, the question asks for an actress, singer, and songwriter who took the tour; the context only supports Idina Menzel in general and does 

## Common Failure Patterns
- (Fill in manually after reviewing justification_samples.json)
