import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score



data = pd.read_csv('stockdata.csv', encoding = "ISO-8859-1")
data.head(1)
train = data[data['Date'] < '20150101']
test = data[data['Date'] > '20141231']
sdata = train.iloc[:, 2:27]
sdata.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)

list1= [i for i in range(25)]
index1=[str(i) for i in list1]
sdata.columns= index1
sdata.head(5)

for index in index1:
    sdata[index]=sdata[index].str.lower()
sdata.head(1)

labels = []

for row in range(0, len(sdata.index)):
    labels.append(' '.join(str(x) for x in sdata.iloc[row, 0:25]))


vector = CountVectorizer(ngram_range=(1, 1))
training = vector.fit_transform(labels)


svm_model = svm.LinearSVC(C=0.1, class_weight='balanced')
svm_model = svm_model.fit(training, train["Label"])
test_labels = []
for row in range(0,len(test.index)):
    test_labels.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
testing = vector.transform(test_labels)
pred = svm_model.predict(testing)

pd.crosstab(test["Label"], pred, rownames=["Actual"], colnames=["Predicted"])
print (classification_report(test["Label"], pred))
print("accuracy=")
print (accuracy_score(test["Label"], pred))
vector2 = CountVectorizer(ngram_range=(1, 2))
training2 = vector2.fit_transform(labels)

svm_model2 = svm.LinearSVC(C=0.1, class_weight='balanced')
svm_model2 = svm_model2.fit(training2, train["Label"])
testing2 = vector2.transform(test_labels)
pred2 = svm_model2.predict(testing2)
pd.crosstab(test["Label"], pred2, rownames=["Actual"], colnames=["Predicted"])
print (classification_report(test["Label"], pred2))
print("accuracy=")
print (accuracy_score(test["Label"], pred2))

vector3 = CountVectorizer(ngram_range=(2, 3))
training3 = vector3.fit_transform(labels)

svm_model3 = svm.LinearSVC(C=0.1, class_weight='balanced')
svm_model3 = svm_model3.fit(training3, train["Label"])
testing3 = vector3.transform(test_labels)
pred3 = svm_model3.predict(testing3)
pd.crosstab(test["Label"], pred3, rownames=["Actual"], colnames=["Predicted"])
print(classification_report(test["Label"], pred3))
print("accuracy=")
print(accuracy_score(test["Label"], pred3))
