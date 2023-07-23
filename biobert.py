from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('pritamdeka/S-BioBert-snli-multinli-stsb')

import pandas as pd
import os
import re

os.chdir("/home/panda/Documents/Honours Thesis/Dataset")

'''
df=pd.read_csv('BGFunction.csv')
df['Function'] = df['Function'].str.replace(' of ', ' ')
df['Function'] = df['Function'].str.replace(' to ', ' ')
df['Function'] = df['Function'].str.replace(' across ', ' ')
df['Function'] = df['Function'].str.replace(' in ', ' ')
df['Function'] = df['Function'].str.replace(' by ', ' ')
df['Function'] = df['Function'].str.replace('P:', '')

emb=[]
for i in range(len(df['Function'])):
        text=df['Function'][i]

        embeddings = model.encode(text)
 
        emb.append(embeddings)

# Convert the list to a dataframe
embdf= pd.DataFrame(emb)
print(len(embdf))
embdf.to_csv('Bertemb.csv')
'''

df=pd.read_csv('Bertemb.csv')
df1=pd.read_csv('BGFunction.csv')
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

go=df1['GO']

df1.index=go

cluster=pd.concat([go, df.reindex(go.index)], axis=1)

df.to_numpy()


# # Initialize the KMeans object with the desired number of clusters
k = 500
kmeans = KMeans(n_clusters=k)

# Fit the data to the KMeans model
kmeans.fit(df)

#clustering = AgglomerativeClustering(n_clusters=500).fit(df)
labels = kmeans.labels_
#labels = clustering.labels_
score= silhouette_score(df,labels, metric='euclidean')
print('The silhouette score is:', score)
cluster=pd.DataFrame({"Cluster": labels}, index=go)
cluster1= pd.merge(cluster, df1, left_index=True, right_index=True)
cluster1.to_csv('Clusterbertkm.csv')
'''
cluster=pd.read_csv('Clusterbertvec.csv')

# Group rows by elements in column B
grouped = cluster.groupby('Cluster')

# Dictionary to store the group DataFrames
group_dfs = {}

# Add each group to a separate DataFrame
for group_name, group_data in grouped:
    group_dfs[group_name] = group_data.copy()

#print(type(group_dfs))

# Concatenate group DataFrames into a single DataFrame
gdf = pd.concat(group_dfs.values(), ignore_index=True)
print(gdf)
gdf.to_csv('Clusterbert.csv')
'''