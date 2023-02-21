import re
import string
import nltk
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize

stopwords_english = stopwords.words('english') 

def StopWord(data):
    clean = []
    for word in data:
        if (word not in stopwords_english and
            word not in string.punctuation):
            clean.append(word)
    return clean

def Stemming(data):
    stemmer = PorterStemmer() 
    clean = []
    for word in data:
        stem_word = stemmer.stem(word)  
        clean.append(stem_word)
    return clean

transcription_path = r'mtsamples.csv'
medical_data = pd.read_csv(transcription_path)


#preprocesssing:

# loose the duplicates and nulls

medical_data = medical_data[medical_data['transcription'].notna()]
medical_data = medical_data.drop_duplicates()


# choosing proper diseases to avoid imbalancement
medical_data = medical_data[medical_data['medical_specialty'].isin([' Obstetrics / Gynecology',' ENT - Otolaryngology',' Urology'])]

medical_data=medical_data[['transcription','medical_specialty']]

# lower case
medical_data["transcription"] = medical_data["transcription"].str.lower()

#not_wanted characters
medical_data["transcription"] = medical_data["transcription"].str.replace(
    re.compile('[/(){}\[\]\|@,;]'), ""
)
medical_data["transcription"] = medical_data["transcription"].str.replace(
    re.compile('[^0-9a-z #+_]'), ""
)

# tokenizing
medical_data["transcription"] = medical_data["transcription"].apply(
    lambda x: word_tokenize(x)
)

#stopwords
medical_data["transcription"] = medical_data["transcription"].apply(
    lambda x: StopWord(x)
)

medical_data["transcription"] = medical_data["transcription"].apply(
    lambda x: Stemming(x)
)
medical_data["transcription"] = medical_data["transcription"].apply(
    lambda x: ' '.join([word for word in x])
)
medical_data.to_csv('preprocessed_data.csv',  encoding='utf-8')
