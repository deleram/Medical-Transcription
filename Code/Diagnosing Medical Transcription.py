import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

n_gram_features =[(1,1),(1,2),(2,2),(2,3),(3,3)]
# generate countmatrices with different n_grams
def generate_n_gram_features(flat_list_transcription):
    temp=[]
    for values in n_gram_features: 
        vectorizer = CountVectorizer(ngram_range=values, max_features= 2000)
        matrix = vectorizer.fit_transform(flat_list_transcription).toarray()
        temp.append(pd.DataFrame(data = matrix, columns =  vectorizer.get_feature_names_out()))
    return temp

medical_path = r'preprocessed_data.csv'
medical_data = pd.read_csv(medical_path)

#change transcription into proper form to use Countmatrices
labels = medical_data['medical_specialty'].tolist()
medical_transcription = medical_data[['transcription']]
unflat_list_transcription = medical_transcription.values.tolist()
flatted = [item for sublist in unflat_list_transcription for item in sublist]
temp = generate_n_gram_features(flatted)


# Try differernt models and show the accuracies
print("KNN")
for j in range(5):
    X_train, X_test, y_train, y_test = train_test_split(temp[j], labels, stratify=labels,random_state=1) 
    model = KNeighborsClassifier().fit(X_train, y_train)
    y_test_pred= model.predict(X_test)
    sum = 0
    for i in range(len(y_test_pred)):
        if(y_test_pred[i] == y_test[i]):
            sum += 1
    print(n_gram_features[j] , ": " , sum/len(y_test_pred))

    
    
    
print("LogisticRegression")
for j in range(5):
    X_train, X_test, y_train, y_test = train_test_split(temp[j], labels, stratify=labels,random_state=1) 
    model = LogisticRegression(penalty= 'elasticnet', solver= 'saga', l1_ratio=0.5, random_state=1, max_iter=4000).fit(X_train, y_train)
    y_test_pred= model.predict(X_test)
    sum = 0
    for i in range(len(y_test_pred)):
        if(y_test_pred[i] == y_test[i]):
            sum += 1
    print(n_gram_features[j] , ": " , sum/len(y_test_pred))



print("RANDOM FOREST")
for j in range(5):
    X_train, X_test, y_train, y_test = train_test_split(temp[j], labels, stratify=labels,random_state=1) 
    model = RandomForestClassifier().fit(X_train, y_train)
    y_test_pred= model.predict(X_test)
    sum = 0
    for i in range(len(y_test_pred)):
        if(y_test_pred[i] == y_test[i]):
            sum += 1
    print(n_gram_features[j] , ": " , sum/len(y_test_pred))

