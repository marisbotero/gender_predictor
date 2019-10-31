import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_excel("D:/Poblacion_GrupoFamiliar_Nombres_Niños.xlsx")

data["Genero"][data["Genero"] == "Masculino"] = 1
data["Genero"][data["Genero"] == "Femenino"] = 2


#Parseamos los datos a string
target = data['Genero'].astype(str)

features = data['nombre'].astype(str)



#Dividimos nuestros datos en una proporción 80% de entrenamiento y 20% de prueba
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=7, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



# Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer

# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

# Machine Learning
# Importando clasificador Naive Bayes (NB).
from sklearn.naive_bayes import MultinomialNB



from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))), ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True ,use_idf=False)), ('clf', MultinomialNB(alpha=0.1))])

#Entrenamos nuestro modelo
text_clf = text_clf.fit(X_train, y_train)

#Calculamos la precisión de nuestro modelo utilizando el clasificador NB
from sklearn.metrics import accuracy_score
predicted = text_clf.predict(X_test)
accuracy_score(y_test, predicted)



from sklearn.model_selection import GridSearchCV
parameters = {
#'vect__max_df': (0.5, 0.625, 0.75, 0.875, 1.0),  
#'vect__max_features': (None, 5000, 10000, 20000),  
#'vect__min_df': (1, 5, 10, 20, 50),  
'vect__ngram_range': [(1, 1), (1, 2)], 
'tfidf__use_idf': (True, False),
'tfidf__sublinear_tf': (True, False),  
#'vect__binary': (True, False),  
'tfidf__norm': ('l1', 'l2'),  
'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)  
}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=2)
gs_clf = gs_clf.fit(X_train, y_train)

gs_clf.best_score_
gs_clf.best_params_


import pickle
from sklearn.externals import joblib
s = pickle.dumps(text_clf)
clf2 = pickle.loads(s)
joblib.dump(text_clf, 'genero_por_nombre.pkl')
clf = joblib.load('genero_por_nombre.pkl')


