import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from keras_preprocessing.sequence import pad_sequences
import PIL

col1, col2, col3 = st.columns(3)	
top = st.container()

bottom = st.container()
col4, col5 , col6 = st.columns(3)
	
with top:
	with col2:
		image = PIL.Image.open('FAKE_NEWS/images/logo.png')
		st.image(image, width=350)
		st.write("  ")
		#st.title("FAKE NEWS DETECTION")
		st.write("  ")
		st.write("  ")
		st.write("  ")

with top:
    st.title("FAKE NEWS DETECTION")
    model = st.radio("What model would you like to make the prediction?", ('Naive Bayes','Passive Aggressive'))
    
    if model == "Naive Bayes":
        f = open('FAKE_NEWS/model/NB_text.pickle', 'rb') #loading model
        classifier = pickle.load(f)
    elif model == "Passive Aggressive":
        f = open('FAKE_NEWS/model/PA_text.pickle', 'rb') #loading model
        classifier = pickle.load(f)
    
    st.write("  ")
    st.write("  ")
    st.write("  ")
    inside = st.text_area("Insert Text of Article here", placeholder = "INSERT TEXT", value = "INSERT TEXT ARTICLE HERE AND PRESS CTRL+ENTER")
    data = [inside]
    data = pd.DataFrame(data, columns=['text'])
	#st.write(data) #just checking
    #model = st.radio("What model would you like to make the prediction?", ('Naive Bayes', 'Passive Agressive'))



#CLEAN DATA
data_final = data.dropna()
data_final.reset_index(drop=True, inplace=True)
ps = PorterStemmer() # Define stemming function
corpus_text = []
for i in range (0,len(data_final)): # Loop through and remove unwanted markings using regular expression
  titles = re.sub('[^a-zA-Z]' , ' ',data_final['text'][i])
  titles =  titles.lower()
  titles = titles.split()

  titles = [ps.stem(word) for word in titles if not word in stopwords.words('english')] # Remove all the stopwords
  titles = ' '.join(titles)
  corpus_text.append(titles)

cv = CountVectorizer(max_features = 3000)
X_text = cv.fit_transform(corpus_text).toarray()
padded_x = pad_sequences(X_text, maxlen=3000, padding='post')


#f.close()
prediction = classifier.predict(padded_x) #making prediction
#st.write(prediction) #just checking

with bottom:

	if prediction[0] == 1:
		st.markdown("<h1 style='text-align: center; color: maroon;'>FAKE</h1>", unsafe_allow_html=True)

	elif prediction[0] == 0:
		st.markdown("<h1 style='text-align: center; color: green;'>NOT FAKE</h1>", unsafe_allow_html=True)
