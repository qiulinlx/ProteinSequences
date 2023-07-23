import pandas as pd
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
os.chdir("/home/panda/Documents/Honours Thesis/Dataset")

df=pd.read_csv('BGOFunc.csv')
df['Function'] = df['Function'].str.replace(' of ', ' ')
df['Function'] = df['Function'].str.replace(' to ', ' ')
df['Function'] = df['Function'].str.replace(' across ', ' ')
df['Function'] = df['Function'].str.replace(' in ', ' ')
df['Function'] = df['Function'].str.replace(' by ', ' ')
df['Function'] = df['Function'].str.replace('P:', '')

def remove_numbers(text):
    # Use regular expression to remove numbers from the text
    text_without_numbers = re.sub(r'\d+', '', text)
    return text_without_numbers

#df= df.applymap(remove_numbers)

for i in range(len(df['Function'])):
    df['Function'][i] = remove_numbers(df['Function'][i])

cv = CountVectorizer(ngram_range=(2,2))
X = cv.fit_transform(df['Function'])
X = X.toarray()
#(len(X))
sort=sorted(cv.vocabulary_.keys())
#print(sort)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(2,2))

transformed = tfidf.fit_transform(df['Function'])
feature_names = tfidf.get_feature_names_out()
tfidf_df = pd.DataFrame(transformed.toarray(), columns=feature_names, index=df['GO'])
#print(tfidf_df)
tfidf_df.to_csv('TFIDF1.csv')
'''
df1=pd.DataFrame(df['GO'], transformed)
df1 = pd.DataFrame(transformed[0].T.todense(),
    	index=tfidf.get_feature_names_out(), columns=["TF-IDF"])
df1 = df1.sort_values('TF-IDF', ascending=False)
print(df1.head(25))
df1.to_csv('TFIDF.csv')'''

#K-means clustering 
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

df=pd.read_csv('BGOFunc.csv')
data=pd.read_csv('TFIDF1.csv')
go=data['GO']
df.index=go
#print(df.index)

data.drop(columns=data.columns[0], axis=1,  inplace=True)
#n_init = 10 
# Initialize the KMeans object with the desired number of clusters
#k = 500
#kmeans = KMeans(n_clusters=k, n_init=n_init)
clustering = AgglomerativeClustering(n_clusters=500).fit(data)

# Fit the data to the KMeans model
#kmeans.fit(data)

# Get the cluster labels for each data point
#labels = kmeans.labels_
labels=clustering.labels_
score= silhouette_score(data,labels, metric='euclidean')
print('The silhouette score is:', score)
cluster=pd.DataFrame({"Cluster": labels}, index=go)
cluster1= pd.merge(cluster, df, left_index=True, right_index=True)
cluster1.to_csv('Clusteragg.csv')

# Get the cluster centers
#centers = kmeans.cluster_centers_

#cluster=pd.read_csv('Clusteragg.csv')


# Group rows by elements in column B
grouped = cluster1.groupby('Cluster')

# Dictionary to store the group DataFrames
group_dfs = {}

# Add each group to a separate DataFrame
for group_name, group_data in grouped:
    group_dfs[group_name] = group_data.copy()

#print(type(group_dfs))

# Concatenate group DataFrames into a single DataFrame
gdf = pd.concat(group_dfs.values(), ignore_index=True)
print(gdf)
gdf.to_csv('Cluster2.csv')
