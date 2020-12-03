## Overview

This repository contains the code for the paper "KFU NLP Team at SMM4H 2020 Tasks: Cross-lingual TransferLearning with Pretrained Language Models for Drug Reactions" [1].


## Data

This repository is devoted to the second task of the [SMM4H 2020 Shared task](https://healthlanguageprocessing.org/smm4h-sharedtask-2020/). The task is the binary classification of tweets that contain a mention of adverse effects of a medication.

Our solution for the classification task consists of two steps:
1. Pretraining on the multilabel sentence classification task using the union of the RuDReC [2] and PsyTAR [3] corpora.
2. Fine-tuning on the target binary classification task using the data of the Shared task.

The preprocessing and pretraining code for the multilabel classification is taken from this [Colab example](https://colab.research.google.com/drive/1g_2W__vi6fuEn8pSma0NXNHbNuebptHF?usp=sharing) which is published in this repository:
https://github.com/cimm-kzn/RuDReC

## Example

For the example of ADR sentences classification, see the ["SMM4H_2020_ADR_classification.ipynb"]( https://github.com/Andoree/smm4h_classification/blob/master/SMM4H_2020_ADR_classification.ipynb) notebook (also available via [Colab](https://colab.research.google.com/drive/1Q5w0GxYjSIMLOooHT7n3Hb7QVoPia9wT?usp=sharing)).

This example contains both the pretraining and the fine-tuning steps. The example utilizes the EnRuDR-BERT model that is available at:
https://github.com/cimm-kzn/RuDReC

## Repository structure

The "training" directory contains the following scripts:
  - Script for the pretraining on the multilabel classification task.
  - Script for the binary classification task.
  
Both scripts rely on the [Google's BERT implementation](https://github.com/google-research/bert).  

The "preprocessing" directory contains scripts for:
  - The merging of RuDReC and PsyTAR sentences into the combined training and validation sets.
  - The preprocessing of the tweets of the SMM4H binary classification task.
  - The merging of the Russian and English datasets of tweets.

The "evaluation" directory contains scripts for the evaluation of binary classification results and for the ensembling of multiple predictions.


## References

1. Miftahutdinov Z., Sakhovskiy A., Tutubalina E. KFU NLP Team at SMM4H 2020 Tasks: Cross-lingual Transfer Learning with Pretrained Language Models for Drug Reactions //Proceedings of the Fifth Social Media Mining for Health Applications (#SMM4H) Workshop & Shared Task. – 2020

2. https://doi.org/10.1093/bioinformatics/btaa675
```
 @article{10.1093/bioinformatics/btaa675,
    author = {Tutubalina, Elena and Alimova, Ilseyar and Miftahutdinov, Zulfat and Sakhovskiy, Andrey and Malykh, Valentin and Nikolenko, Sergey},
    title = {The Russian Drug Reaction Corpus and Neural Models for Drug Reactions and Effectiveness Detection in User Reviews},
    journal = {Bioinformatics},
    year = {2020},
    month = {07},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btaa675},
    url = {https://doi.org/10.1093/bioinformatics/btaa675},
    note = {btaa675},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/doi/10.1093/bioinformatics/btaa675/33539752/btaa675.pdf},
}
```
3. Zolnoori, Maryam, et al. "A systematic approach for developing a corpus of patient reported adverse drug events: a case study for SSRI and SNRI medications." Journal of biomedical informatics 90 (2019): 103091.Zolnoori, Maryam, et al. "A systematic approach for developing a corpus of patient reported adverse drug events: a case study for SSRI and SNRI medications." Journal of biomedical informatics 90 (2019): 103091.
