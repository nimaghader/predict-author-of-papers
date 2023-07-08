#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import packages
import pandas as pd
import json
import string
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# In[ ]:


# reading in the "train" dataset
df = pd.read_json('train.json')


# # 0. Exploratory data analysis

# In[ ]:


df.head(2)


# In[ ]:


df.shape


# In[ ]:


# number of unique authors and venues
df['authorId'].nunique(), df['venue'].nunique()


# In[ ]:


# number of papers from each author
df['authorId'].value_counts()


# In[ ]:


# number of authors for each paper-count
paper_count = pd.DataFrame(df['authorId'].value_counts())
print(paper_count)
paper_count['authorId'].value_counts()


# # 1. Pre-processing

# 1.1. Lower-casing
# 
# 1.2. Remove punctuatuon
# 
# 1.3. Remove stop-words and apply lemmatization
# 
# 

# ## 1.1. Lower-casing

# In[ ]:


get_ipython().run_cell_magic('time', '', "# make everything lower-case in all titles.\ndf['title'] = df['title'].apply(lambda x: x.lower())\n\n# make everything lower-case in all abstracts.\ndf['abstract'] = df['abstract'].apply(lambda x: x.lower())")


# ## 1.2. Remove punctuation

# In[ ]:


# "string.punctuation" contains a list of all punctuations that are removed. Can view the list by running:

"""
list_of_punctuations = string.punctuation
print(list_of_punctuations)
"""

# remove punctuation in all titles. 
df['title'] = df['title'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# remove punctuation in all abstracts". 
df['abstract'] = df['abstract'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))


# ## 1.3. Remove stop-words and apply lemmatization

# In[ ]:


# loading the english-language-model into object called nlp
nlp = spacy.load('en_core_web_sm')

# function to remove stop-words and apply lemmatization
def remove_stopwords_and_lemmatize(text):
    
    # apply english-language-model to "text" and store it in "doc"
    doc = nlp(text)
    
    # for each token (word) in "doc": lemmatize that token ("token.lemma"), and if that token is not a stop-word ("token.is_stop"),
    # then join it into a sentence with spaces in-between (' '.join(...))
    text_without_stopwords_and_lemmatized = ' '.join([token.lemma_ for token in doc if not token.is_stop])
    return text_without_stopwords_and_lemmatized


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# apply the function to all titles.\ndf['title'] = df['title'].apply(remove_stopwords_and_lemmatize)\n\n# apply the function to all abstracts.\ndf['abstract'] = df['abstract'].apply(remove_stopwords_and_lemmatize)")


# # 2. Feature engineering

# ## 2.1. Combine title and abstract

# In[ ]:


df['merged'] = df["title"] + ' ' + df["abstract"]


# # 3. Model Building

# 3.1. Pre-processing
# 
# 3.2. Baseline model
# 
# 3.3. Hyperparameter tuning
# 
# 3.4. Model with best parameters
# 
# 3.5. Hypothesis
# 
# 3.6. Final model
# 
# 3.7. Submission file

# ## 3.1. Pre-processing

# In[ ]:


# split into train and validation set
X_train, X_val, y_train, y_val = train_test_split(df['merged'], df['authorId'], 
                                                  random_state=5, 
                                                  test_size=0.1)


# In[ ]:


# define vectorizer
vectorizer = TfidfVectorizer(min_df=3,
            token_pattern=r'([a-zA-Z0-9]{1,})',
            ngram_range=(1, 2), 
            sublinear_tf=True,
            stop_words = 'english')

# apply TfidfVectorizer on train and val features
X_train_vectorized = vectorizer.fit_transform(X_train) 
X_val_vectorized = vectorizer.transform(X_val)

X_train_vectorized.shape, X_val_vectorized.shape


# ## 3.2. Baseline model

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# define model\nlsvc = LinearSVC(loss='hinge', max_iter=2000)\n\n# train model\nlsvc.fit(X_train_vectorized, y_train)\n\n# make predictions on val set\ny_pred = lsvc.predict(X_val_vectorized)\n\n# calculate accuracy on val set\nprint('Accuracy: {0:0.4f}'. format(accuracy_score(y_val, y_pred)))")


# ## 3.2. Hyperparameter tuning

# In[ ]:


# get number of papers for each author
num_papers = pd.DataFrame(df.authorId.value_counts()).reset_index(level=0)

# filter for authors with at least 2 papers and make list with authorIds
num_papers = num_papers[num_papers['authorId']>1]
list_no_unique = num_papers['index'].to_list()

# modified train set with each author having at least 2 papers
df_no_unique = df[df['authorId'].isin(list_no_unique)]


# In[ ]:


X = df_no_unique['merged']
y = df_no_unique['authorId']
X.shape, y.shape


# In[ ]:


vectorizer = TfidfVectorizer(min_df=3,
            token_pattern=r'([a-zA-Z0-9]{1,})',
            ngram_range=(1, 2), 
            sublinear_tf=True,
            stop_words = 'english')


X_vectorized = vectorizer.fit_transform(X) 


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nlsvc = LinearSVC(loss='hinge', max_iter=2000)\n\nparams = dict(C=[0.01, 0.1, 0.5, 1, 2, 3, 5, 7, 10, 15])\n\n\nrandom_search = RandomizedSearchCV(lsvc, params, n_iter = 10, cv=2, random_state=0)\nresult = random_search.fit(X_vectorized, y)")


# In[ ]:


all_results = pd.DataFrame(result.cv_results_)
all_results


# In[ ]:


# result from hyperparameter tuning
all_results = pd.DataFrame(result.cv_results_)
all_results[['param_C', 'split0_test_score', 'split1_test_score', 'mean_test_score', 'rank_test_score']]


# In[ ]:


# get best parameters based on random search
print('Best parameters: {}'. format(result.best_params_))


# ## 3.4. Model with best parameters

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# define model with best parameters\nlsvc = LinearSVC(loss='hinge', C=3, max_iter=2000)\n\n# train model\nlsvc.fit(X_train_vectorized, y_train)\n\n# make predictions on val set\ny_pred = lsvc.predict(X_val_vectorized)\n\n# calculate accuracy on val set\nprint('Accuracy: {0:0.4f}'. format(accuracy_score(y_val, y_pred)))")


# ## 3.4. Hypothesis

# In[ ]:


df_no_unique = df[df['authorId'].isin(list_no_unique)]


# In[ ]:


# split into train and validation set
X_train1, X_val1, y_train1, y_val1 = train_test_split(df_no_unique['merged'], df_no_unique['authorId'], 
                                                  random_state=5, 
                                                  test_size=0.5,
                                                  stratify=df_no_unique['authorId'])


# In[ ]:


# apply TfidfVectorizer on train and val features
X_train1_vectorized = vectorizer.fit_transform(X_train1) 
X_val1_vectorized = vectorizer.transform(X_val1)

X_train1_vectorized.shape, X_val1_vectorized.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# define model\nlsvc = LinearSVC(loss='hinge', C=3, max_iter=2000)\n\n# train model\nlsvc.fit(X_train1_vectorized, y_train1)\n\n# make predictions on val set\ny_pred1 = lsvc.predict(X_val1_vectorized)\n\n# calculate accuracy on val set\nprint('Accuracy: {0:0.4f}'. format(accuracy_score(y_val1, y_pred1)))")


# ## 3.6. Final model

# In[ ]:


# reading in the "test" dataset
df_test = pd.read_json('test.json')
df_test.shape


# In[ ]:


df_test.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n#apply the same pre-processor steps as above, but this time on the test data ("df_test")\ndf_test[\'title\'] = df_test[\'title\'].apply(lambda x: x.lower())\ndf_test[\'abstract\'] = df_test[\'abstract\'].apply(lambda x: x.lower())\ndf_test[\'title\'] = df_test[\'title\'].apply(lambda x: x.translate(str.maketrans(\'\', \'\', string.punctuation)))\ndf_test[\'abstract\'] = df_test[\'abstract\'].apply(lambda x: x.translate(str.maketrans(\'\', \'\', string.punctuation)))\ndf_test[\'title\'] = df_test[\'title\'].apply(remove_stopwords_and_lemmatize)\ndf_test[\'abstract\'] = df_test[\'abstract\'].apply(remove_stopwords_and_lemmatize)')


# In[ ]:


# combine titles and abstracts
df['merged'] = df["title"] + ' ' + df["abstract"]
df_test['merged'] = df_test["title"] + ' ' + df_test["abstract"]

# whole train set
X = df['merged']
y = df['authorId']

# test set
X_test = df_test['merged']


# In[ ]:


vectorizer = TfidfVectorizer(min_df=3,
            token_pattern=r'([a-zA-Z0-9]{1,})',
            ngram_range=(1, 2), 
            sublinear_tf=True,
            stop_words = 'english')

# apply TfidfVectorizer on train and test features
X_vectorized = vectorizer.fit_transform(X) 
X_test_vectorized = vectorizer.transform(X_test)

#X_vectorized.shape, X_test_vectorized.shape
X_vectorized.shape, X_test_vectorized.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# define model with best parameters\nlsvc = LinearSVC(loss='hinge', C=3, max_iter=2000)\n\n# train model on whole train set\nlsvc.fit(X_vectorized, y)\n\n# make predictions on unseen test set\ny_pred = lsvc.predict(X_test_vectorized)")


# In[ ]:


y_pred.shape


# # 3.7. Submission file

# In[ ]:


# add predictions to df_test
df_test['authorId'] = y_pred.tolist()
predictions = df_test[['paperId', 'authorId']]

# make a list of dicts for each row in predictions
predictions = predictions.to_dict('records')


# In[ ]:


with open('predictions.json', 'w', encoding='utf-8') as f:
    json.dump(predictions, f, indent=4)


# # Testing on test_test. Make sure to delete before submitting!!!!!!!!!!

# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_test_test = pd.read_json(\'test_test.json\')\n#apply the same pre-processor steps as above, but this time on the test data ("df_test")\ndf_test_test[\'title\'] = df_test_test[\'title\'].apply(lambda x: x.lower())\ndf_test_test[\'abstract\'] = df_test_test[\'abstract\'].apply(lambda x: x.lower())\ndf_test_test[\'title\'] = df_test_test[\'title\'].apply(lambda x: x.translate(str.maketrans(\'\', \'\', string.punctuation)))\ndf_test_test[\'abstract\'] = df_test_test[\'abstract\'].apply(lambda x: x.translate(str.maketrans(\'\', \'\', string.punctuation)))\ndf_test_test[\'title\'] = df_test_test[\'title\'].apply(remove_stopwords_and_lemmatize)\ndf_test_test[\'abstract\'] = df_test_test[\'abstract\'].apply(remove_stopwords_and_lemmatize)')


# In[ ]:


df['merged'] = df["title"] + ' ' + df["abstract"]
df_test_test['merged'] = df_test_test["title"] + ' ' + df_test_test["abstract"]


# In[ ]:


X = df['merged']
y = df['authorId']
X_test_test = df_test_test['merged']
y_test_test = df_test_test['authorId']
X.shape, X_test_test.shape


# In[ ]:


vectorizer = TfidfVectorizer(min_df=3,
            token_pattern=r'([a-zA-Z0-9]{1,})',
            ngram_range=(1, 2), 
            sublinear_tf=True,
            stop_words = 'english')


X_vectorized = vectorizer.fit_transform(X) 
X_test_vectorized = vectorizer.transform(X_test_test)

X_vectorized.shape, X_test_vectorized.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "lsvc = LinearSVC(loss='hinge', C=3, max_iter=2000)\nlsvc.fit(X_vectorized, y)\ny_pred = lsvc.predict(X_test_vectorized)\n\nprint('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test_test, y_pred)))")

