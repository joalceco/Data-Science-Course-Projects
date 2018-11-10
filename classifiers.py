from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np

samples = 10000

X_females = np.array([np.random.normal(68.7,5,samples),
	np.random.normal(1.58,5,samples),
	np.random.normal(23.9,1,samples)]).T
y_females = np.repeat("female",samples)
X_males = np.array([np.random.normal(74.8,5,samples),
	np.random.normal(1.64,5,samples),
	np.random.normal(26.7,1,samples)]).T
y_males = np.repeat("male",samples)

X = np.vstack([X_females,X_males])
y = np.hstack([y_females,y_males])
print(X.shape,y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y)

def printResults(model,X_train,X_test,y_train,y_test):
	y_hat_train = model.predict(X_train)
	y_hat_test = model.predict(X_test)
	cm_train = confusion_matrix(y_train,y_hat_train,labels=["male","female"])
	cm_test = confusion_matrix(y_test,y_hat_test,labels=["male","female"])
	print("Training CM",cm_train,sep="\n")
	print("Testing CM",cm_test,sep="\n")

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", 
         "Decision Tree", "Random Forest", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(2),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB()]

for name, clf in zip(names,classifiers):
	clf = clf.fit(X_train,y_train)
	print(name)
	printResults(clf,X_train,X_test,y_train,y_test)