
# coding: utf-8

# In[1]:

# %load poi_id.py
#!/usr/bin/python

import sys
import pickle
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


# In[4]:

for i in data_dict:
    print i


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


# In[10]:

print data_dict["THE TRAVEL AGENCY IN THE PARK"]
    


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

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[17]:

features_list


# In[18]:

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier()
clf.fit(features, labels)


# In[19]:

len(clf.feature_importances_)


# In[20]:

clf.feature_importances_


# In[22]:

features_importance = {}
for i in range(1,len(features_list)):
    features_importance[features_list[i]] = clf.feature_importances_[i-1]


# In[23]:

features_importance


# In[45]:

sorted(features_importance.iteritems(), key=lambda x:x[1], reverse= True)


# In[ ]:




# In[18]:

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()


# In[ ]:




# In[ ]:




# In[18]:

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:



