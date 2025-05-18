import pandas as pd
from sklearn.model_selection import train_test_split
import os

def readPolarityReviews(filepath: str) -> pd.DataFrame:
    '''
    Reads in Cornell movie reviews polarity dataset found at https://www.cs.cornell.edu/people/pabo/movie-review-data/.

    Citation:
        @InProceedings{Pang+Lee:04a,
          author =       {Bo Pang and Lillian Lee},
          title =        {A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts},
          booktitle =    "Proceedings of the ACL",
          year =         2004
        }

    Args:
        filepath (str):
            Directory containing the dataset. Should have subdirectories pos and neg with text files for each review.

    Returns:
        data (pd.DataFrame): Returns a dataframe containing Review, ID, and Polarity
    '''

    # Get names of all positive and negative reviews
    neg_files = os.listdir(f'{filepath}/neg')
    pos_files = os.listdir(f'{filepath}/pos')

    # Get the review ID
    neg_id = [i.split('_')[1].split('.')[0] for i in neg_files]
    pos_id = [i.split('_')[1].split('.')[0] for i in pos_files]

    # Read in positive and negative reviews separately
    neg_reviews=[pd.read_csv(f'{filepath}/neg/{fil}',
                             sep='   ',
                             header=None,
                             engine='python')[0].str.cat()
                 for fil in neg_files]

    pos_reviews=[pd.read_csv(f'{filepath}/pos/{fil}',
                             sep='   ',
                             header=None,
                             engine='python')[0].str.cat()
                 for fil in pos_files]

    # Convert list of reviews to dataframe with columns for the id's and indicator for positive/negative
    neg = pd.DataFrame(zip(neg_reviews,neg_id,[0]*len(neg_reviews)),columns=['Review','ID','Polarity'])
    pos = pd.DataFrame(zip(pos_reviews,pos_id,[1]*len(neg_reviews)),columns=['Review','ID','Polarity'])

    # Concatenate the two datasets as they now have an indicator for positive/negative (negative = 0)
    data = pd.concat([neg,pos]).reset_index(drop=True)

    return data



def readScaleReviews(filepath: str, class4 = False, class3 = False, split = 'Train') -> pd.DataFrame:
    '''
    Reads in Cornell movie reviews scale ratings dataset found at https://www.cs.cornell.edu/people/pabo/movie-review-data/.

    Citation:
        @InProceedings{Pang+Lee:05a,
          author =       {Bo Pang and Lillian Lee},
          title =        {Seeing stars: Exploiting class relationships for sentiment
                          categorization with respect to rating scales},
          booktitle =    {Proceedings of the ACL},
          year =         2005
        }

    Args:
        filepath (str):
            Directory containing the dataset. Should have subdirectories for each author with files:
                subj.(name)
                rating.(name)
                id.(name)

        class4 (bool): Whether or not to include the binned classes (4 of them). Default is False.

        class3 (bool): Whether or not to include the binned classes (3 of them). Default is False.

        split (str): Either "Train", "Val", or "Test"

    Returns:
        data (pd.DataFrame): Returns a dataframe containing Review, ID, Rating, and Author
    '''

    data = pd.DataFrame() # Initialize dataframe
    for name in os.listdir(filepath): # For each author

        # List files and get the name of the review, rating, and id files
        files = os.listdir(f'{filepath}/{name}')
        review_file = [i for i in files if 'subj.' in i][0]
        rating_file = [i for i in files if 'rating.' in i][0]
        id_file = [i for i in files if 'id.' in i][0]

        # Read in files
        reviews = pd.read_csv(f'{filepath}/{name}/{review_file}',header=None,sep='\r\n',names=['Review'],engine='python')
        ratings = pd.read_csv(f'{filepath}/{name}/{rating_file}',header=None,names=['Rating'],engine='python')
        ids = pd.read_csv(f'{filepath}/{name}/{id_file}',header=None,names=['ID'],engine='python')
        if class4:
            class_file = [i for i in files if '4class.' in i][0]
            classes4 = pd.read_csv(f'{filepath}/{name}/{class_file}',header=None,names=['Class4'],engine='python')
            df = pd.concat([reviews,ratings,ids,classes4],axis=1)
        if class3:
            class_file = [i for i in files if '3class.' in i][0]
            classes3 = pd.read_csv(f'{filepath}/{name}/{class_file}',header=None,names=['Class3'],engine='python')
            if not class4:
                df = pd.concat([reviews,ratings,ids,classes3],axis=1)
            else:
                df = pd.concat([reviews,ratings,ids,classes3,classes4],axis=1)
        else:
            # Concatenate columns and make column for author
            df = pd.concat([reviews,ratings,ids],axis=1)


        df['Author'] = [name]*df.shape[0]

        # Concatenate author to all reviews
        data = pd.concat([data,df])

    # Only one author uses 2 decimals, so round their reviews to 1 decimal.
    data['Rating'] = data['Rating'].round(decimals=1)

    if split == 'All':
        return data

    train, val = train_test_split(data, random_state=40, train_size = 0.7)

    if split == 'Train':
        return train.reset_index(drop=True)

    val, test = train_test_split(val, random_state=40, train_size = 2/3)

    if split == 'Val':
        return val.reset_index(drop=True)
    if split == 'Test':
        return test.reset_index(drop=True)
