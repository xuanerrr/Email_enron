import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from data_process import get_features

max_features = 5000

x,y = get_features()
print(x.shape)
print(np.asarray(y).shape)

X_train, X_test, y_train, y_test = train_test_split(x, np.asarray(y), test_size=0.2, random_state=11)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = naive_bayes.MultinomialNB(alpha=1)
model.fit(X_train,y_train)
#print(model.class_log_prior_)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)