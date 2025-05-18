import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from tqdm import tqdm
from Read_Datasets import readScaleReviews

# Read in datasets and convert the ratings to integer classes for classification tasks
train = readScaleReviews(filepath = 'Data/CornellMovieReview_scale_data/scaledata',
                         class3 = True,
                         class4 = True,
                         split='Train')

val = readScaleReviews(filepath = 'Data/CornellMovieReview_scale_data/scaledata',
                       class3 = True,
                       class4 = True,
                       split='Val')

test = readScaleReviews(filepath = 'Data/CornellMovieReview_scale_data/scaledata',
                        class3 = True,
                        class4 = True,
                        split='Test')

train['Rating'] = (train['Rating']*10).astype(int)
val['Rating'] = (val['Rating']*10).astype(int)
test['Rating'] = (test['Rating']*10).astype(int)

# Initialize spacy and use textblob for polarity scores
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")

print(train)
print(val)
print(test)

for data in [train,val,test]:
    # Get polarity score
    for i in tqdm(range(data.shape[0])):
        if i == 0:
            data['PolarityScore'] = [0]*data.shape[0]
        review = data.iloc[i]
        doc = nlp(review['Review'])
        data['PolarityScore'].iloc[i] = doc._.blob.polarity

    # Get Review Polarity, use >= 5
    data['Polarity'] = [0]*data.shape[0]
    data.loc[data['Rating'] >= 5,'Polarity'] = 1

print(train['PolarityScore'].describe())
print(train.loc[train['Polarity']==1,'PolarityScore'].describe())
print(train.loc[train['Polarity']==0,'PolarityScore'].describe())

train.to_csv('Data/CornellMovieReview_scale_data/MovieReviewScaleData_Train_PolarityValence.csv',index=False)
val.to_csv('Data/CornellMovieReview_scale_data/MovieReviewScaleData_Val_PolarityValence.csv',index=False)
test.to_csv('Data/CornellMovieReview_scale_data/MovieReviewScaleData_Test_PolarityValence.csv',index=False)
