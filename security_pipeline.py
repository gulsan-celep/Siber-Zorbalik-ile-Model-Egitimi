import pandas as pd
data = pd.read_csv('siber.csv')
data.head()


import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
import re


# storing stopwords of Turkish (set structure for speed)
stops = set(stopwords.words('turkish'))
print(stops)


# pattern string in order to exclude non-Turkish-letter characters 
# such as punctuations and numbers
exc_letters_pattern = '[^a-zçğışöü]'



def text_to_wordlist(text, remove_stopwords=False, return_list=False):
    # 1. convert to lower
    text = text.lower()
    # 2. replace special letters
    special_letters = {'î':'i', 'â': 'a'}
    for sp_let, tr_let in special_letters.items():
        text = re.sub(sp_let, tr_let, text)
    # 3. remove non-letters
    text = re.sub(exc_letters_pattern, ' ', text)
    # 4. split
    wordlist = text.split()
    # 5. remove stopwords
    if remove_stopwords:
        wordlist = [w for w in wordlist if w not in stops]
    # 6.
    if return_list:
        return wordlist
    else:
        return ' '.join(wordlist)
    
clean_messages = []
for    message in data['message']:
    clean_messages.append(text_to_wordlist(
        message, remove_stopwords=True, return_list=False))
    
    
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    clean_messages, data['cyberbullying'], test_size=0.33, random_state=1)  



from sklearn.feature_extraction.text import CountVectorizer

# limit vocabulary size as at most 5000; thus, words with 
# the least frequency are not included in the vocabulary
vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)

# since we assume that we are not familiar with test set in advance,
# we use our training set in the construction of the vocabulary 
train_data_features = vectorizer.fit_transform(x_train)

# convert it to numpy array since it is easier to work with
train_data_features = train_data_features.toarray()
    



from __future__ import print_function

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

print(__doc__)




# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(train_data_features, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    
    # converting test data to BOW features
test_data_features = vectorizer.transform(x_test)
test_data_features = test_data_features.toarray()


    y_true, y_pred = y_test, clf.predict(test_data_features)
    print(classification_report(y_true, y_pred))
    print()


# yukarida buldugunuz kernel ve C degerlerini asagida kullaniyoruz
    
    # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    # decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    # max_iter=-1, probability=False, random_state=None, shrinking=True,
    # tol=0.001, verbose=False)

from sklearn.svm import SVC 
 
svclassifier = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
     max_iter=-1, probability=False, random_state=None, shrinking=True,
     tol=0.001, verbose=False)
 
svclassifier.fit(train_data_features, y_train)


# converting test data to BOW features
test_data_features = vectorizer.transform(x_test)
test_data_features = test_data_features.toarray()


y_pred = svclassifier.predict(test_data_features)

from sklearn.metrics import classification_report, confusion_matrix 

cm = confusion_matrix(y_test, y_pred)
cm 
import numpy as np
print("Doğruluk oranı: %", sum(np.diag(cm))/ cm.sum()*100)



predictions = pd.DataFrame(
    data={"message": x_test, "cyberbullying_true": y_test, "cyberbullying_pred": y_pred})

# correct_count = sum(y_pred == y_test)
correct_count = (predictions["cyberbullying_pred"] == predictions["cyberbullying_true"]).sum()
print("Accuracy is %{:.3f}".format(100 * correct_count / len(y_test)))

from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print('\tPredictions')
print('\t{:>5}\t{:>5}'.format(0,1))
for row_id, real_row in enumerate(cf):
    print('{}\t{:>5}\t{:>5}'.format(row_id, real_row[0], real_row[1]))
    
    

    
    
    