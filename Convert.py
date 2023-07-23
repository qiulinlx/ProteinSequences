import pandas as pd

GOdf = pd.read_csv('CleanedBGO.csv',keep_default_na=False)

Seqdf = pd.read_csv('CleanedSequence.csv',keep_default_na=False)

Clusters= pd.read_csv('Cluster2.csv',keep_default_na=False)

df=pd.concat([Seqdf, GOdf], axis=1)
df = df.drop(columns=[df.columns[3]])
df=df.drop(columns=[df.columns[0]])
# Get the letters corresponding to the specific number
for i in range (500):
    target=i
    letters = Clusters[Clusters['Cluster'] == i]['GO'].tolist()
    df = df.replace(letters, i)
    print(i)

df.to_csv('Clusteredseq.csv')
