# NLP-Move-Scale-Rating-Classification-using-LSTM-and-Transformers
This project uses multi-task learning for LSTM neural networks and fine tunes a pretrained ROBERTA transformer to predict the scale rating of movie reviews. The models are compared against a baseline random forest classifier using a rule based sentiment analyzer from "Spacy" and the top 50 unigrams and bigrams containing adjectives from each movie review. The dataset used is the Cornell movie review scale rating dataset (https://www.cs.cornell.edu/people/pabo/movie-review-data/) introduced in "Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales" (https://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.pdf). While the main task of this project is to predict the scale rating of a movie review, other tasks need to be introduced to perform multi-task learning for the LSTM. The other tasks include polarity (0 if review <= 0.5, 1 else), 3 and 4 class classifications binning together the star ratings, and authorship classification. Results show an improvement of the LSTM for scale review classification when trained on all 5 tasks, however no improvement is seen when testing on the transformer (results not shown). Overall, the LSTM seems to only work on par with the random forest, and the transformer outperforms both easily.

---

## Files

### CreatePolarityValenceDataset.py
Creates datasets used as input into the LSTM and transformers.

### GetReviewBigramsUnigrams.py
Gets top 50 bigrams and unigrams containing adjectives and computes a polarity score between -1 and 1 using spacy. Used to make input files for random forest.

### PlotF1.py
Plots the results for all models ran, with various groupings.

### Read_Datasets.py
File to read in the raw data from the Cornell Movie Review Dataset. The data used is the "scale_dataset_v1.0" from https://www.cs.cornell.edu/people/pabo/movie-review-data/. This needs to be downloaded to run the analyses.

### ScaleReviewHistorgrams.py
Creates a histogram showing the frequency of each rating (0.0, 0.1, ..., 1.0) in the dataset.

### mallet_en_stoplist.txt
File containing stopwords.

### ngram_counts.py
File containing functions used to get the bigrams and unigrams in GetReviewBigramsUnigrams.py.

---

## Subdirectories

### LSTM
This subdirectory contains jupyter notebooks that run LSTM models for individual tasks, two tasks at a time (all having scale rating as one task), and all five tasks at once. Jupyter notebooks are used to take advantage of Google Colab's GPU as my personal computer is not any good for deep learning.

### ROBERTA
This subdirectory contains jupyter notebooks that fine tune a ROBERTA transformer for each of the five individual tasks for a comparison against the LSTM for each task. No Multi-Task learning results are shown here due to it not really improving much of anything for the pretrained model.

### Random Forest
