# DSSM-with-Paddle

## Details
DSSM implementation with PaddlePaddle. 
It use user search query and clicked titles with label 1 and unclicked/unrelated title with label 0 to predict the relevance between user search query and advertisement titles.

### train_cluster_old.py
Read input in csv format as: query, title, label.

The old model follows the original paper with the exact same structure and parameters. 
However it only achieves about 70-80% accuracy when predicting labels due to change in language (the paper was deigned for English languages but the training and test set was based on Chinese)

### train_cluster.py
Read input in csv format as: query, title_1, title_2.

This is the improved model where one of the title from title_1 or 2 is one that user clicks and the other one is unrelated.
The job of the model is to learn with one of the title is more relevant than the other one. From this approach the model can achieve over 88% accuracy with single machine and over 94% when training on clusters.

## Reference
PaddlePaddle: https://github.com/PaddlePaddle/Paddle

DSSM: https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/
