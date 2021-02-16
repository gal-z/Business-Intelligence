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
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.fit_transform(X_test)

# k_range = range(1, 21)
# accuracy = []
# best_accuracy = 0
# best_k = 0
# for k in k_range:
#     classifier = KNeighborsClassifier(n_neighbors= k )
#     classifier.fit(X_train,y_train)
#     y_predict = classifier.predict(X_test)
#     acureccy_k = accuracy_score(y_test, y_predict)
#     accuracy.append(acureccy_k)
#     if acureccy_k > best_accuracy:
#         best_accuracy = acureccy_k
#         best_k = k

# print(("Best K: {0}, Best Accuracy: {1}".format(best_k, best_accuracy)),'\n',classification_report(y_test, y_predict))
#
# plt.plot(k_range, accuracy)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Testing Accuracy')
# plt.show()

classifier = KNeighborsClassifier(n_neighbors=12)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
y_score= classifier.predict_proba(X_test)
acureccy= accuracy_score(y_test, y_predict)
print('Confusion matrix for KNN  ','\n',confusion_matrix(y_test, y_predict,),'\n',classification_report(y_test, y_predict))


fpr, tpr, threshold = roc_curve(y_test, y_score[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()



gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
% (X_test.shape[0], (y_test != y_pred).sum()))
print('Confusion matrix for naive bayes  ','\n',confusion_matrix(y_test, y_pred,),'\n',classification_report(y_test, y_pred))
y_score = gnb.predict_proba(X_test)
fpr3, tpr3, threshold = roc_curve(y_test, y_score[:, 1])
roc_auc3 = auc(fpr3, tpr3)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr3, tpr3, 'b', label = 'AUC = %0.2f' % roc_auc3)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of naive bayes')
plt.savefig('ROC Curve of naive bayes.png')
plt.show()



classifier2 = DecisionTreeClassifier(max_depth=3,criterion="entropy")
classifier2.fit(X_train, y_train)
y_predict = classifier2.predict(X_test)
y_score2= classifier2.predict_proba(X_test)
print('Confusion matrix for tree  ','\n',confusion_matrix(y_test, y_predict,),'\n',classification_report(y_test, y_predict))

# dot_data = tree.export_graphviz(classifier2, out_file=None,
#                 filled=True, rounded=True,
#                 special_characters=True,feature_names = features,class_names=['0','1'])
# graph = graphviz.Source(dot_data)
# graph.write_png('diabetes.png')

#


tree.plot_tree(classifier2,filled=True,
              rounded=True,proportion=True
              ,fontsize=6,class_names=['0','1'],feature_names = features)
plt.savefig('fulltree.png')
plt.show()



fpr2, tpr2, threshold2 = roc_curve(y_test, y_score2[:, 1])
roc_auc2 = auc(fpr2, tpr2)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr2, tpr2, 'b', label = 'AUC = %0.2f' % roc_auc2)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of Decision tree')
plt.show()


plt.title('Receiver Operating Characteristic for all')
plt.plot(fpr, tpr, 'r', label = 'AUC KNN = %0.2f' % roc_auc)
plt.plot(fpr2, tpr2, 'b', label = 'AUC Tree  = %0.2f' % roc_auc2)
plt.plot(fpr3, tpr3, 'g', label = 'AUC naive bayes= %0.2f' % roc_auc3)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of for all')
plt.show()