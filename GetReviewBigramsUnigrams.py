import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from Read_Datasets import readScaleReviews
from scipy.stats import pearsonr
import numpy as np
from ngram_counts import *
from collections import Counter
from tqdm import tqdm


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")


# Read in data
data = readScaleReviews(filepath = 'Data/CornellMovieReview_scale_data/scaledata', split='Train')
print(data)

#data['PolarityScore'] = data.apply(lambda x: nlp(x['Review'])._.blob.polarity,axis=1)

#print(pearsonr(data['Rating'],data['PolarityScore']))

#print(data[['Rating','PolarityScore']].describe())


#ntokens=[]
#for i in range(data.shape[0]):
#    review = data.iloc[i]
#    doc = nlp(review['Review'])
#    ntokens.append(len([token.orth_ for token in doc]))


# Get polarity score
for i in tqdm(range(data.shape[0])):
    if i==0:
        data['PolarityScore'] = [0]*data.shape[0]
    review = data.iloc[i]
    doc = nlp(review['Review'])
    data['PolarityScore'].iloc[i] = doc._.blob.polarity
    #print(doc._.blob.subjectivity)
    #print(doc._.blob.sentiment_assessments.assessments)
    #print(doc._.blob.ngrams())

#print(max(ntokens))
#print(np.std(ntokens))
#print(np.mean(ntokens))

# Read in the stopword list
stopwords = load_stopwords()

remove_stopword_bigrams = True
bigram_counts = Counter()
# Get counts of bigrams for all reviews
for i in tqdm(range(data.shape[0])):
    review = data.iloc[i]

    # Call spacy and get tokens
    tokens = nlp(review['Review'])
    #tokens = [token.orth_ for token in tokens]

    # Normalize
    #tokens = normalize_tokens(tokens)

    # Get bigrams
    #bigrams = ngrams(tokens, 2)
    bigrams = ngrams_adj(tokens, 2)

    # Filter out bigrams where either token is punctuation
    bigrams = filter_punctuation_bigrams(bigrams,adj=True)



    # Optionally filter bigrams where either word is a stopword
    if remove_stopword_bigrams:
        bigrams = filter_stopword_bigrams(bigrams, stopwords,adj=True)

    # Increment bigram counts
    for bigram in bigrams:
        bigram_counts[f'{bigram[0]}_{bigram[1]}'] += 1

topN_to_show = 50
unigram_w1_counts = get_unigram_counts(bigram_counts,0)
unigram_w2_counts = get_unigram_counts(bigram_counts,1)
print("\nTop bigrams by frequency")
print_sorted_items(bigram_counts, topN_to_show, 'descending')
print("\nTop unigrams by frequency w1")
print_sorted_items(unigram_w1_counts, topN_to_show, 'descending')
print("\nTop unigrams by frequency w2")
print_sorted_items(unigram_w2_counts, topN_to_show, 'descending')

top_bigrams = get_sorted_items(bigram_counts, topN_to_show, 'descending')
top_unigrams = get_sorted_items(unigram_w1_counts, topN_to_show, 'descending')

print(top_bigrams)
print(top_unigrams)


# See if each bigram is in each review
for i in tqdm(range(data.shape[0])):
    review = data.iloc[i]

    # Call spacy and get tokens
    tokens = nlp(review['Review'])

    # Get bigrams
    bigrams = ngrams_adj(tokens, 2)

    # Filter out bigrams where either token is punctuation
    bigrams = filter_punctuation_bigrams(bigrams,adj=True)

    # Optionally filter bigrams where either word is a stopword
    if remove_stopword_bigrams:
        bigrams = filter_stopword_bigrams(bigrams, stopwords,adj=True)

    review_bigrams = [f'{bigram[0]}_{bigram[1]}' for bigram in bigrams]

    for top_bigram in top_bigrams:

        if i==0:
            data[top_bigram] = [0]*data.shape[0]

        if top_bigram in review_bigrams: # If the top bigram occurs in the current review
            data[top_bigram].iloc[i] = 1

    # Get unigrams
    review_unigrams = [bigram.split('_')[0] for bigram in review_bigrams]
    for unigram in top_unigrams:
        if i==0: # Initialize column
            data[unigram] = [0]*data.shape[0]

        if unigram in review_unigrams:
            data[unigram].iloc[i] = 1


data.to_csv('Data/CornellMovieReview_scale_data_train_with_bigramunigrams.csv',index=False)



#################################
# Get values for validation set #
#################################

data = readScaleReviews(filepath = 'Data/CornellMovieReview_scale_data/scaledata', split='Val')

# Get polarity score
for i in tqdm(range(data.shape[0])):
    if i==0:
        data['PolarityScore'] = [0]*data.shape[0]
    review = data.iloc[i]
    doc = nlp(review['Review'])
    data['PolarityScore'].iloc[i] = doc._.blob.polarity

# See if each bigram is in each review
for i in tqdm(range(data.shape[0])):
    review = data.iloc[i]

    # Call spacy and get tokens
    tokens = nlp(review['Review'])

    # Get bigrams
    bigrams = ngrams_adj(tokens, 2)

    # Filter out bigrams where either token is punctuation
    bigrams = filter_punctuation_bigrams(bigrams,adj=True)

    # Optionally filter bigrams where either word is a stopword
    if remove_stopword_bigrams:
        bigrams = filter_stopword_bigrams(bigrams, stopwords,adj=True)

    review_bigrams = [f'{bigram[0]}_{bigram[1]}' for bigram in bigrams]

    for top_bigram in top_bigrams:

        if i==0:
            data[top_bigram] = [0]*data.shape[0]

        if top_bigram in review_bigrams: # If the top bigram occurs in the current review
            data[top_bigram].iloc[i] = 1

    # Get unigrams
    review_unigrams = [bigram.split('_')[0] for bigram in review_bigrams]
    for unigram in top_unigrams:
        if i==0: # Initialize column
            data[unigram] = [0]*data.shape[0]

        if unigram in review_unigrams:
            data[unigram].iloc[i] = 1

data.to_csv('Data/CornellMovieReview_scale_data_val_with_bigramunigrams.csv',index=False)

#################################
# Get values for test set #
#################################


data = readScaleReviews(filepath = 'Data/CornellMovieReview_scale_data/scaledata', split='Test')

# Get polarity score
for i in tqdm(range(data.shape[0])):
    if i==0:
        data['PolarityScore'] = [0]*data.shape[0]
    review = data.iloc[i]
    doc = nlp(review['Review'])
    data['PolarityScore'].iloc[i] = doc._.blob.polarity

# See if each bigram is in each review
for i in tqdm(range(data.shape[0])):
    review = data.iloc[i]

    # Call spacy and get tokens
    tokens = nlp(review['Review'])

    # Get bigrams
    bigrams = ngrams_adj(tokens, 2)

    # Filter out bigrams where either token is punctuation
    bigrams = filter_punctuation_bigrams(bigrams,adj=True)

    # Optionally filter bigrams where either word is a stopword
    if remove_stopword_bigrams:
        bigrams = filter_stopword_bigrams(bigrams, stopwords,adj=True)

    review_bigrams = [f'{bigram[0]}_{bigram[1]}' for bigram in bigrams]

    for top_bigram in top_bigrams:

        if i==0:
            data[top_bigram] = [0]*data.shape[0]

        if top_bigram in review_bigrams: # If the top bigram occurs in the current review
            data[top_bigram].iloc[i] = 1

    # Get unigrams
    review_unigrams = [bigram.split('_')[0] for bigram in review_bigrams]
    for unigram in top_unigrams:
        if i==0: # Initialize column
            data[unigram] = [0]*data.shape[0]

        if unigram in review_unigrams:
            data[unigram].iloc[i] = 1

data.to_csv('Data/CornellMovieReview_scale_data_test_with_bigramunigrams.csv',index=False)
