from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import pandas
import numpy as np
# reading the csv file
data_set=pandas.read_csv('dataset.csv')
#select the answers and questions columns
questions = data_set['Questions']
Ans = data_set['Answers']

# TF-IDF Vectorizer 
vect = TfidfVectorizer()
#different k values for cross validations
#NOTE:Since we use the whole data set as the train data, cross validation is used to drop down the probability of overfitting.
K_values = [1,2,3,4,5,6,7,8,9]
cross_val_results = {}
#determining the best k value by using cross validation
for k in K_values:
    
    cv_score = cross_val_score(KNeighborsClassifier(n_neighbors=k) , vect.fit_transform(questions) , Ans , cv=2 , scoring="accuracy")

    mean_accuracy = np.mean(cv_score)
    cross_val_results[k] = mean_accuracy
    

best_k = max(cross_val_results, key=cross_val_results.get)

# En iyi K değerini yazdır
print(f"\nBest K value: {best_k}")

model = KNeighborsClassifier(n_neighbors=1)
model.fit(vect.fit_transform(questions), pandas.Categorical(Ans).codes)

new_text = "What is the capital of France?"

# Transform the new text into TF-IDF features
new_text_features = vect.transform([new_text])

# Predict the answer category using the trained model
predicted_category = model.predict(new_text_features)[0]

# Map the predicted category back to the original answer label
predicted_answer = Ans.iloc[predicted_category]

print(f"Predicted answer for the new text: {predicted_answer}")

