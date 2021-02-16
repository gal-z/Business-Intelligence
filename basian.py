import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,auc,roc_curve,plot_roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from data_prep import *
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from sklearn import tree
import graphviz
from IPython.display import Image
import pydotplus
from sklearn.naive_bayes import GaussianNB

X = data_apps[['age', 'workclass', 'education', 'education.num', 'marital.status', 'occupation', 'relationship'
   , 'sex', 'hours.per.week', 'native.country',  'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']].copy()
Y = data_apps['income'].copy().astype(int)

features=['age', 'workclass', 'education', 'education.num', 'marital.status', 'occupation', 'relationship'
   , 'sex', 'hours.per.week', 'native.country',  'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)








gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
% (X_test.shape[0], (y_test != y_pred).sum()))
print('Confusion matrix for naive bayes  ','\n',confusion_matrix(y_test, y_pred,),'\n',classification_report(y_test, y_pred))
y_score = gnb.predict_proba(X_test)
fpr3, tpr3, threshold = roc_curve(y_test, y_score[:, 1])
roc_auc = auc(fpr3, tpr3)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of naive bayes')
plt.savefig('ROC Curve of naive bayes.png')
plt.show()