import pandas as pd     
import numpy as np
import pickle
from googletrans import Translator
import streamlit as st
import brotli
import lzma
import pickle5 as pickle


translator = Translator()


with open('data', 'rb') as handle:
    data = pickle.load(handle)


# text = data['text'].values.astype('U').tolist()
# label = data['sentiment'].values.astype('U').tolist()


with lzma.open('lmza_test.xz', 'rb') as handle:
    model = pickle.load(handle)

def max2(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None


st.title("Tweet Mood Detector")
st.header("Enter your Tweet here to get tweet mood")
text_result = st.text_area("You can enter you tweet or text here",height=150)


if text_result:
    for i in range(100):
        try:
            if translator.detect(text_result).lang == "en":
                text = text_result
                break
            else:
                text = translator.translate(text_result).text
            break
        except:
            continue

           
    st.write("English version of the enterance",text)
    proba =  model.predict_proba([text])   
    
    chart_data = pd.DataFrame(columns=data.sentiment.unique())
    chart_data.loc[0] = proba[0]
    theList = list(proba[0])
    chart_data = chart_data
    st.write(chart_data)
    st.write("Maximum mood detected is: ",chart_data.columns[theList.index(max(theList))],max(theList))
    st.write("Second mood detected is: ",chart_data.columns[theList.index(max2(theList))],max2(theList))
    st.header('Probability of correctness')
    st.bar_chart(chart_data)
    
