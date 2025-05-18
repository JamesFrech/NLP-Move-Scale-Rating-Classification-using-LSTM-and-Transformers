from Read_Datasets import readScaleReviews
import matplotlib.pyplot as plt
import seaborn as sns

data = readScaleReviews('Data\CornellMovieReview_scale_data\scaledata', split='All')
print(data)

print(data['Author'].value_counts())

sns.histplot(data['Rating'],bins=11)
plt.ylabel('Number of Reviews')
plt.xlabel('Rating')
plt.title('Number of Reviews for each Rating Value')
plt.savefig('Images/ReviewRatingHistogram.png',dpi=300,bbox_inches='tight')
plt.close()
