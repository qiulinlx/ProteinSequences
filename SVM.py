#from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

df=pd.read_csv('SVMdata.csv')
y=df['Labels']
X=df['2']

def alphabet_to_number(alphabet):
    alphabet = alphabet.upper()  # Convert alphabet to uppercase
    return ord(alphabet) - 64  # Subtract 64 to get the numerical value

# Convert alphabets in a sequence to numbers
def sequence_to_numbers(sequence):
    numbers = []
    for alphabet in sequence:
        number = alphabet_to_number(alphabet)
        numbers.append(number)
    return numbers

seq=[]
Sequences=X.values.tolist()

for i in range(len(Sequences)):
    nseq=sequence_to_numbers(Sequences[i])
    #print(nseq)
    seq.append(nseq)
    #print("Hi")
seq=pd.DataFrame(seq)

seq= seq.replace('', np.nan) # Replace empty strings with NaN


seq = seq.fillna(0) # Fill NaN values with zeroes

''' Count the number of elements in each row
row_element_counts = seq.apply(lambda row: len(row), axis=1)
print(row_element_counts.max()) #5537 is longest sequence
seq.to_csv('Seqtest.csv')'''


X=seq
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                    train_size=0.80, test_size=0.20, random_state=4)
batch=len(y_test)
y_test=np.array(y_test)
#print(type(y_test))

#svm=SVC()
#svm=LinearSVC()
svm=SGDClassifier()

# Fit the SVM classifier to the training data
svm.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = svm.predict(X_test)
correct_predictions = (y_pred == y_test).sum().item()
print(correct_predictions, "out of", batch)
accuracy=svm.score(X_test, y_test, sample_weight=None)
print(accuracy)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()