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


classifier2 = DecisionTreeClassifier(max_depth=5,criterion="entropy")
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



# dot_data = StringIO()
# export_graphviz(classifier2, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True,feature_names = features,class_names=['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('diabetes.png')
# Image(graph.create_png())



fpr2, tpr2, threshold2 = roc_curve(y_test, y_score2[:, 1])
roc_auc = auc(fpr2, tpr2)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr2, tpr2, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of Decision tree')
plt.show()




