# Importing the libraries
import pickle
import pandas as pd
import webbrowser
# !pip install dash
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from matplotlib import pyplot as plt
from matplotlib import colors
from dash.dependencies import Input, Output , State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
import wordcloud
from collections import Counter
import numpy as np
from wordcloud import WordCloud, STOPWORDS

# Declaring Global variables
project_name = None
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Defining My Functions
def load_model():
    global scrappedReviews
    scrappedReviews = pd.read_csv('scrappedReviews.csv')
  
    global pickle_model
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)

    global vocab
    file = open("feature.pkl", 'rb') 
    vocab = pickle.load(file)
    
    #wordcloud
    dataset = scrappedReviews['reviews'].to_list()
    str1 = ''  # all input data/reviews is in str1
    for i in dataset:
        str1 = str1+i
    str1 = str1.lower()

    stopwords = set(STOPWORDS)
    cloud = WordCloud(width = 800, height = 400,
                stopwords = stopwords,
                max_words=10,
                background_color="skyblue",
                colormap="Blues",
                min_font_size = 10).generate(str1)# all input data/reviews is in str1
    
    cloud.to_file("assets/wordCloud.png")

def check_review(reviewText):

    # reviewText has to be vectorised, that vectorizer is not saved yet load the vectorize and call transform 
    # and then pass that to model preidctor load it later
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    # Add code to test the sentiment of using both the model # 0 == negative   1 == positive
    return pickle_model.predict(vectorised_review)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')

def create_app_ui():
    main_layout = html.Div(
    [    
     html.H2(children = "WordCloud",style = {'text-align':'center','text-decoration':'underline','color':'red'}),
     html.P([html.Img(src=app.get_asset_url('wordCloud.png'),style={'width':'700px','height':'400px'})],style={'text-align':'center'})
      ]    
    )
    
    return main_layout

# Main Function to control the Flow of your Project
def main():
    print("Start of your project")
    load_model()
    open_browser()
    
    global scrappedReviews
    global project_name
    global app
    
    project_name = "Sentiment Analysis with Insights"
    
    app.title = project_name
    app.layout = create_app_ui()
    app.run_server()
    
    print("End of my project")
    project_name = None
    scrappedReviews = None
    app = None
        
# Calling the main function 
if __name__ == '__main__':
    main()
