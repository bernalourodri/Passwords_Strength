import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



senha = st.text_input("Type password")


def makeTokens(f):
    tokens = []
    for i in f:
        tokens.append(i)
    return tokens


pswd_dados = pd.read_csv("passwordlist.csv")


pswd = np.array(pswd_dados).astype(str)

ylabels  = [s[2] for s in pswd]
senhafull = [s[1] for s in pswd]


vectorizer = TfidfVectorizer(tokenizer=makeTokens)

X = vectorizer.fit_transform(senhafull)

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)

logit = LogisticRegression(penalty='l2',multi_class='ovr')
logit.fit(X_train, y_train)




New_predict = [senha]

New_predict = vectorizer.transform(New_predict)
y_Predict = logit.predict(New_predict)
print(senha)
print(y_Predict)
if y_Predict[0]=="0":
    st.markdown("Weak password")
elif y_Predict[0]=="1":
    st.markdown("Medium password")
else:
    st.markdown("Strong password")