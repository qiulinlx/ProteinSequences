import pandas as pd
import json
import os
os.chdir("/home/panda/Documents/Honours Thesis/Dataset")

GO={}
Sequence={}
Function=[]
GOann=[]
# Specify the path to the JSON file
json_file_path = 'HumanSwiss.json'

#Read the JSON file
with open(json_file_path, 'r') as file:
    json_data = json.load(file)
    
#Extract the desired value from the JSON data
for j in range(20422):
    value = json_data['results'][j]['sequence']['value']
    GOannotations = json_data['results'][j]['uniProtKBCrossReferences']
    GeneOntology=[]
    Sequence['Sequence_'+str(j)]=value

    for i in range(len(GOannotations)):
        if GOannotations[i]['database'] == 'GO':
            if GOannotations[i]['properties'][0]['value'].startswith('P:'):
                Function.append(GOannotations[i]['properties'][0]['value'])
                GOann.append(GOannotations[i]['id'])
                #Gid=(GOannotations[i]['id'])
                #GeneOntology.append(Gid)

    GO['GO_'+str(j)]=GeneOntology
    
#Sequencedf=pd.DataFrame.from_dict(Sequence, orient='index')
#Genedf=pd.DataFrame.from_dict(GO, orient='index')
Functiondf=pd.DataFrame(GOann, Function)
#Sequencedf.to_csv('Sequence.csv')
#Genedf.to_csv('BGO.csv')
Functiondf.to_csv('BGOFunc.csv')
