from spacy.lang.en import English
from collections import Counter
import re
import string

# Function to read stopwords
def load_stopwords(filename='mallet_en_stoplist.txt'):
    file = open(filename,'r')
    stopwords = [i.strip('\n') for i in file]
    return set(stopwords)

# Take a list of string tokens and return all ngrams of length n,
# representing each ngram as a list of  tokens.
# E.g. ngrams(['the','quick','brown','fox'], 2)
# returns [['the','quick'], ['quick','brown'], ['brown','fox']]
# Note that this should work for any n, not just unigrams and bigrams
def ngrams(tokens, n):
    ngram_list = []
    for i in range(len(tokens) - n + 1):
        ngram_list.append(tokens[i:i+n])
    # Returns all ngrams of size n in sentence, where an ngram is itself a list of tokens
    return ngram_list

def ngrams_adj(tokens, n):
    ngram_list = []
    for i in range(len(tokens) - n + 1):
        #print(tokens[i].dep_)
        if tokens[i].dep_ == 'amod' or tokens[i+1].dep_ == 'amod':
            #print(tokens[i])
            ngram_list.append(tokens[i:i+n])
    #exit()
    return ngram_list

# Remove punctuation
def filter_punctuation_bigrams(ngrams, adj=False):
    # Input: assume ngrams is a list of ['token1','token2'] bigrams
    # Removes ngrams like ['today','.'] where either token is a single punctuation character
    # Note that this does not mean tokens that merely *contain* punctuation, e.g. "'s"
    # Returns list with the items that were not removed
    punct = string.punctuation
    if adj:
        return [ngram for ngram in ngrams if ngram[0].orth_ not in punct and ngram[1].orth_ not in punct]
    else:
        return [ngram for ngram in ngrams if ngram[0] not in punct and ngram[1] not in punct]

# Remove stopwords from ngrams
def filter_stopword_bigrams(ngrams, stopwords, adj=False):
    # Input: assume ngrams is a list of ['token1','token2'] bigrams, stopwords is a set of words like 'the'
    # Removes ngrams like ['in','the'] and ['senator','from'] where either word is a stopword
    # Returns list with the items that were not removed
    if adj:
        return [ngram for ngram in ngrams if ngram[0].orth_ not in stopwords and ngram[1].orth_ not in stopwords]
    else:
        return [ngram for ngram in ngrams if ngram[0] not in stopwords and ngram[1] not in stopwords]

def normalize_tokens(tokenlist):
    # Input: list of tokens as strings,  e.g. ['I', ' ', 'saw', ' ', '@psresnik', ' ', 'on', ' ','Twitter']
    # Output: list of tokens where
    #   - All tokens are lowercased
    #   - All tokens starting with a whitespace character have been filtered out
    #   - All handles (tokens starting with @) have been filtered out
    #   - Any underscores have been replaced with + (since we use _ as a special character in bigrams)

    normalized_tokens = [i.lower() for i in tokenlist]

    p1 = re.compile('^\s') # Regex for starts with whitespace
    normalized_tokens = [i for i in normalized_tokens if not p1.match(i)]

    p2 = re.compile('^@') # Regex for starts with whitespace
    normalized_tokens = [i for i in normalized_tokens if not p2.match(i)]

    # Replace _ with +
    normalized_tokens = [re.sub('_','+',i) for i in normalized_tokens]

    return normalized_tokens

def collect_bigram_counts(lines, stopwords, remove_stopword_bigrams = False):
    # Input lines is a list of raw text strings, stopwords is a set of stopwords
    #
    # Create a bigram counter
    # For each line:
    #   Extract all the bigrams from the line
    #   If remove_stopword_bigrams is True:
    #     Filter out any bigram where either word is a stopword
    #   Increment the count for each bigram
    # Return the counter
    #
    # In the returned counter, the bigrams should be represented as string tokens containing underscores.
    #
    if (remove_stopword_bigrams):
        print("Collecting bigram counts with stopword-filtered bigrams")
    else:
        print("Collecting bigram counts with all bigrams")

    # Initialize spacy and an empty counter
    print("Initializing spacy")
    nlp       = English(parser=False) # faster init with parse=False, if only using for tokenization
    counter   = Counter()

    # Iterate through raw text lines
    for line in tqdm(lines):

        # Call spacy and get tokens
        tokens = nlp(line)
        tokens = [token.orth_ for token in tokens]

        # Normalize
        tokens = normalize_tokens(tokens)

        # Get bigrams
        bigrams = ngrams(tokens, 2)

        # Filter out bigrams where either token is punctuation
        bigrams = filter_punctuation_bigrams(bigrams)

        # Optionally filter bigrams where either word is a stopword
        if remove_stopword_bigrams:
            bigrams = filter_stopword_bigrams(bigrams, stopwords)

        # Increment bigram counts
        for bigram in bigrams:
            counter[f'{bigram[0]}_{bigram[1]}'] += 1

    return counter

def print_sorted_items(dict, n=10, order='ascending'):
    if order == 'descending':
        multiplier = -1
    else:
        multiplier = 1
    ranked = sorted(dict.items(), key=lambda x: x[1] * multiplier)
    for key, value in ranked[:n] :
        print(key, value)
    #return ranked
def get_sorted_items(dict, n=10, order='ascending'):
    if order == 'descending':
        multiplier = -1
    else:
        multiplier = 1
    ranked = sorted(dict.items(), key=lambda x: x[1] * multiplier)

    keys = []
    for key, value in ranked[:n]:
        keys.append(key)
    return keys

def get_unigram_counts(bigram_counts, idx):
    # bigram_counts is the count of bigrams
    # idx is the index of word from bigram to count (0 or 1)
    counter = Counter()
    for key, val in bigram_counts.items():
        words = key.split('_')
        word = words[idx]
        counter[word] += val
    return counter
