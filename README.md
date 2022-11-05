# Contextualised Embeddings for Lexical Semantic Change Detection
This code accompanies the paper [UiO-UvA at SemEval-2020 Task 1: Contextualised Embeddings for Lexical Semantic Change Detection](https://arxiv.org/abs/2005.00050),
which describes our participation in SemEval 2020 Task 1: Unsupervised Lexical Semantic Change Detection.

Data can be stored here: https://drive.google.com/drive/folders/1htrD_T5IVkOcF2kq-fOqZOIlhaWXP6I_?usp=sharing



##  Extraction of contextualized token embeddings without adapter(s)

For BERT: `python3 code/bert/collect.py <PATH_TO_MODEL> <CORPUS> <TARGET_WORDS> <OUTFILE>`

These scripts produce `npz` archives containing numpy arrays with token embeddings for each target word in a given corpus.

To run:  `python code/bert/collect.py  code/bert/model_config  ../semeval2020_ulscd_eng/corpus1/token  test_data_truth/target_nopos.txt outuput1`



## Estimating semantic change
- COS algorithm: `python3 code/cosine.py -t <TARGET_WORDS> -i0 corpus0.npz -i1 corpus1.npz > cosine_change.txt`

These scripts produce plain text files containing lists of words with their corresponding degree of semantic change between
*corpus0* and *corpus1*.

To run:  `python code/cosine.py --input0=outuput1.npz --input1=outuput2.npz --target=test_data_truth/target_nopos.txt --output=result`


### Extraction of contextualized token embeddings with adapter(s)
`python3 code/bert/adapter_collect.py <ADAPTER CONFIG> <PATH_TO_MODEL> <CORPUS> <TARGET_WORDS> <OUTFILE>`

To run:  `python code/bert/adapter_collect.py code/bert/adapter_config code/bert/model_config ../semeval2020_ulscd_eng/corpus1/token  test_data_truth/target_nopos.txt outuput1`



## Authors
- Andrey Kutuzov (University of Oslo, Norway)
- Mario Giulianelli (University of Amsterdam, Netherlands)


### SemEval-2020 Task 1 Reference
--------

Dominik Schlechtweg, Barbara McGillivray, Simon Hengchen, Haim Dubossarsky and Nina Tahmasebi.
[SemEval 2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://competitions.codalab.org/competitions/20948).
To appear in SemEval@COLING2020.
