
# coding: utf-8

# In[6]:

# %load poi_id.py


# In[1]:

# %load poi_id.py
#!/usr/bin/python

import sys
import pickle
from pprint import pprint
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.


# In[2]:

### The first feature must be "poi".
features_list = ['poi','salary' , 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
                 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 


# In[3]:

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# ### Total number of data points

# In[5]:

len(data_dict)


# ### Allocation across classes (POI/non-POI)

# In[6]:

n_poi = 0
n_non_poi = 0
for i in data_dict:
    if data_dict[i]["poi"] ==1:
        n_poi += 1
    else:
        n_non_poi += 1
        

print "number of poi:", n_poi
print "number of non-poi:", n_non_poi
    


# ### Number of missing values in each Feature

# In[7]:

features = {}
for name in data_dict.keys():
    for feature in data_dict[name]:
        if feature not in features.keys() and data_dict[name][feature] == "NaN":
            features[feature] = 1
        elif feature in features.keys() and data_dict[name][feature] == "NaN":
            features[feature] += 1


# In[8]:

print "Number of missing values in each feature:", features


# In[9]:

data_dict["TOTAL"]


# In[11]:

### Task 2: Remove outliers
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")


# In[12]:

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict


# In[13]:

len(my_dataset)


# ### Creating two new features
# First feature: ratio_from_poi_to_this_person
# 
# Second feature: ratio_from_this_person_to_poi

# In[14]:

def Calculate_ratio (poi_relevant_mails, total_mails):
    if poi_relevant_mails == "NaN" or total_mails == "NaN":
        ratio = 0.
    else:
        ratio = float(poi_relevant_mails)/ float(total_mails)
    return ratio


# In[15]:

for name in my_dataset:
    my_dataset[name]["ratio_from_poi_to_this_person"] = Calculate_ratio(my_dataset[name]["from_poi_to_this_person"],
                                                                        my_dataset[name]["from_messages"] )
    
    my_dataset[name]["ratio_from_this_person_to_poi"] = Calculate_ratio(my_dataset[name]["from_this_person_to_poi"],
                                                                        my_dataset[name]["to_messages"] )


# In[16]:

features_list.extend(["ratio_from_poi_to_this_person","ratio_from_this_person_to_poi"])


# In[17]:

features_list


# In[18]:

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# ### Selecting features

# In[19]:

from sklearn.feature_selection import SelectKBest


# In[20]:

k_best = SelectKBest()
k_best.fit(features, labels)
scores = k_best.scores_
tuples = zip(features_list[1:], scores)
k_best_features = sorted(tuples, key=lambda x: x[1], reverse=True)


# Through feature selection, I kept the top 10 features with the highest scores.

# In[21]:

k_best_features


# In[22]:

features_with_scores= []
for i in k_best_features:
    features_with_scores.append(i[0])
top_3_features=features_with_scores[0:3]
top_3_features.insert(0, "poi")


# ### Feature scaling

# In[24]:

import numpy as np


# In[25]:

feature_range = {}
for person in my_dataset.keys():
    for feature in my_dataset[person]:
        if feature not in feature_range.keys() and my_dataset[person][feature] != "NaN":
            feature_range[feature] = []
            feature_range[feature].append(my_dataset[person][feature])
        if feature in feature_range.keys() and my_dataset[person][feature] != "NaN":
            feature_range[feature].append(my_dataset[person][feature])


# In[26]:

for i in feature_range:
    print str(i),":", "max:",max(feature_range[i]),"min:",min(feature_range[i])


# As we can see in the maximal and minimal value of each feature, the scale varies diffently among thte selected features. For example, the bonus ranges from 5-digit numbers to 7-digit numbers, while "the shared receipt with poi" ranges from a 1-digit number to a 4-digit one.

# In[27]:

from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler


# In[28]:

data = featureFormat(my_dataset, top_3_features, sort_keys=True)
labels, features = targetFeatureSplit(data)


# In[29]:

scaler = MinMaxScaler()
#scaler = StandardScaler()


# In[30]:

scaled_features = scaler.fit_transform(features)


# In[31]:

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# In[ ]:




# In[32]:

from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[33]:

DT_clf = DecisionTreeClassifier(min_samples_split= 2, min_samples_leaf= 1, max_features= 1)
RF_clf = RandomForestClassifier(min_samples_split= 4, n_estimators= 10, min_samples_leaf= 1)
SVM_clf = SVC()
LR_clf = LogisticRegression(C= 4.0, tol= 1e-80)


# In[34]:

def evaluate_clf(clf, features, labels, num_iters=10, test_size=0.3):
    print clf
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =            train_test_split(features, labels, test_size=0.3, random_state=42)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        
    print "accuracy: {}".format(np.mean(accuracy))
    print "precision: {}".format(np.mean(precision))
    print "recall:    {}".format(np.mean(recall))


# In[35]:

classifiers = [DT_clf,RF_clf,SVM_clf, LR_clf ]
for classifier in classifiers:
    print classifier
    print evaluate_clf(classifier, features, labels)


# In[36]:

tuning_parameters_DT = {'min_samples_split': [2,4,8,10,20],'min_samples_leaf':[1,2,4,8,10],'max_features':[1,2,3]}


# In[37]:

DT = GridSearchCV(DecisionTreeClassifier(), tuning_parameters_DT, scoring = 'recall')
DT.fit(scaled_features, labels)
print("Best parameters are:")
print(DT.best_params_)


# In[38]:

tuning_parameters_RF = {'min_samples_split': [2,4,8],'min_samples_leaf':[1,2,4],'n_estimators':[10,50,100]}


# In[39]:

RF = GridSearchCV(RandomForestClassifier(), tuning_parameters_RF, scoring = 'recall')
RF.fit(scaled_features, labels)
print("Best parameters are:")
print(RF.best_params_)


# In[40]:

tuning_parameters_LR ={'tol':[1e-40,1e-20,1e-2, 0.1,1],'C':[1.0,2.0,4.0,6.0]}


# In[41]:

LR = GridSearchCV(LogisticRegression(), tuning_parameters_LR, scoring = 'recall')
LR.fit(scaled_features, labels)
print("Best parameters are:")
print(LR.best_params_)


# In[42]:

from tester import test_classifier


# In[43]:

classifiers = [DT_clf]
for classifier in classifiers:
    print classifier
    print test_classifier(classifier, my_dataset, top_3_features)


# In[44]:

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(DT_clf, my_dataset, features_list)






# In[ ]:



