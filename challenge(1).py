# Importing the libraries
import pickle
import pandas as pd
import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
#import dash_design_kit as ddk

# Declairing Global variables(I want my object to be available till end of the project so will use below step)
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
project_name = "Sentiment Analysis with Insights"

# Defining my functions
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")
    
def load_model():
    global scrappedReviews
    scrappedReviews = pd.read_csv("scrappedReviews.csv")
    
    global pickle_model
    file_1 = open("pickle_model.pkl", "rb") 
    pickle_model = pickle.load(file_1)
    
    global vocab
    file_2 = open("feature.pkl", "rb")
    vocab = pickle.load(file_2)

def check_review(reviewText):  # here we are giving text data & telling to convert it in Numeric
    trans = TfidfTransformer()
    vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    reviewText = trans.fit_transform(vec.fit_transform([reviewText]))
    return pickle_model.predict(reviewText)


def create_app_ui():
    global scrappedReviews
    main_layout = html.Div(
        html.Div(
                [
            
                    html.H1(id = 'heading', children = project_name, className = 'display-3 mb-4', style = {'color':'purple','fontSize': 50,'fontWeight': 'bold'}), 
                    dcc.Textarea(id = 'textarea', className="mb-3", placeholder="Enter the Review", value = 'My daughter loves these shoes', style = {'fontSize': 20,'textAlign': 'center','height': '150px','width':'1050px'}),
                    html.H1([
                        dcc.Dropdown(
                    id='dropdown',
                    placeholder = 'Select a Review',
                    options=[{'label': i[:100] + "...", 'value': i} for i in scrappedReviews.reviews],
                    value = scrappedReviews.reviews[0],
                    style = {'margin-bottom': '30px','height':'30px','fontSize': 20}
                    
                )
                       ],
                        style = {'padding-left': '50px', 'padding-right': '50px'}
                        ),
                    html.Button(children = "SUBMIT",className="mt-2 mb-3", id = 'button', style = {'width': '150px', 'backgroundColor':'blue','fontSize': 25,'fontWeight': 'bold'}),
                    html.Div(id = 'result1'),
                    html.Div(id = 'result2')
                    ],
                className = 'text-center'
                ),
        className = 'mt-4'
        )
    
    return main_layout
    

# when somebody typing on textarea ---> after clicking "submit" button --> it should display in first tag as
# "positive" or "negative"    
@app.callback(  
    Output('result1','children' ),
    [
    Input('button', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    
 
def update_app_ui(n_clicks,textarea):
    result_of_list = check_review(textarea)
    
    if (result_of_list[0] == 0 ):
        return dbc.Alert("NEGATIVE", color="red",style = {'fontSize': 20})
    elif (result_of_list[0] == 1 ):
        return dbc.Alert("POSITIVE",color="green",style = {'fontsize': 20})
    else:
        return dbc.Alert("UNKNOWN",color="dark", style = {'fontsize': 20 })

# update dropdown list & when somebody changes values from dropdown list-->then it should display in second 
# tag as "positive" or "negative"
@app.callback(  
    Output('result2','children' ),
    [
    Input('button', 'n_clicks')
    ],
    [
    State('dropdown', 'value')
    ]
    )       

def update_dropdown(n_clicks, value):
    result_of_list = check_review(value)
    
    if (result_of_list[0] == 0 ):
        return dbc.Alert("NEGATIVE",color="green",style = {'fontsize': 20})
    elif (result_of_list[0] == 1 ):
        return dbc.Alert("POSITIVE",color="green",style = {'fontsize': 20})
    else:
        return dbc.Alert("UNKNOWN",color="green",style = {'fontsize': 20})

   
# Main function to control flow of the project
def main():
    global app
    global project_name
    load_model()
    #open_browser()
    
    app.title = project_name
    app.layout = create_app_ui()
    app.run_server(host='0.0.0.0', port=8050)
    
    
    project_name = None
    app = None
#  Calling Main function
if __name__ == '__main__':
    main()