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
    #pie chart
    print('Loading Data......')
    temp = []
    for i in scrappedReviews['reviews']:
        temp.append(check_review(i)[0])
    scrappedReviews['sentiment'] = temp
    
    positive = len(scrappedReviews[scrappedReviews['sentiment']==1])
    negative = len(scrappedReviews[scrappedReviews['sentiment']==0])
    
    explode = (0.1,0)  # Negative portion outside is explode

    langs = ['Positive', 'Negative',]
    students = [positive,negative]
    color = ['yellow','red']
    plt.pie(students,explode=explode,startangle=90,colors = color,labels = langs,autopct='%1.2f%%')
    cwd = os.getcwd()
    if 'assets' not in os.listdir(cwd):
        os.makedirs(cwd+'/assets')
    plt.savefig('assets/sentiment.png')
    
def check_review(reviewText):

    #reviewText has to be vectorised, that vectorizer is not saved yet
    #load the vectorize and call transform and then pass that to model predictor
    #load it later

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    
    return pickle_model.predict(vectorised_review)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')
    
def create_app_ui():
    main_layout = html.Div(
    [
    html.H1(id='Main_title', children = "Sentiment Analysis with Insights",style={'text-align':'center','color':'blue'}),
    html.Hr(style={'background-color':'orange'}),
    html.H2(children = "Pie Chart",style = {'text-align':'center','text-decoration':'underline','color':'orange'}),
    html.P([html.Img(src=app.get_asset_url('sentiment.png'),style={'width':'700px','height':'400px','color':'blue'})],style={'text-align':'center'}),
    html.Hr(style={'background-color':'orange'})
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
