#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fastapi import FastAPI

app = FastAPI()

# Load data from JSON file
with open('Dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

diseases = []
for medication in data:
    diseases.append(medication)

medication_names = []
Indications = []
Wit_Indications = {}

for disease in diseases:
    Indications_Infos = data[disease]
    s_Indications = ''
    for Indications_Info in Indications_Infos:
        medication_names.append(Indications_Info['DrugName'])
        Indications.append(Indications_Info['Indications'] + '--' + disease + '--')
        s_Indications += Indications_Info['Indications']
    Wit_Indications[disease] = s_Indications

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(Indications)


@app.post("/prediction")
def search_medicine(x):
    xx = [x]  # Convert the user input into a list of texts
    user_tfidf = vectorizer.transform(xx)  # Transform the user input to TF-IDF matrix

    # Calculate cosine similarity between the user input and all the symptoms/indications
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)

    # Get the indices of the top matching medications
    top_indices = np.argsort(similarity_scores, axis=1)[:, -1]

    # Get the names of the matching medications
    matching_medications = [medication_names[index] for index in top_indices]
    if matching_medications:
        medications_dict = {f"matching_medications":(matching_medications)}
        return medications_dict
    else:
        return {"message": "No matching medications found."}

