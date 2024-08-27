from flask import Flask, render_template, request
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

from keras.models import load_model

from src.get_data import GetData
from src.utils import create_figure, prediction_from_model 
import flask_monitoringdashboard as dashboard
import logging


app = Flask(__name__)

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s : %(message)s',
    # detailed format : '%(asctime)s %(levelname)s [%(module)s.%(funcName)s]: %(message)s [in %(pathname)s:%(lineno)d]'
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

data_retriever = GetData(app.logger, url="https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/etat-du-trafic-en-temps-reel/exports/json?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B")
data = data_retriever()

model = load_model('model.h5') 

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':

        fig_map = create_figure(data)
        graph_json = fig_map.to_json()

        selected_hour = request.form['hour']

        cat_predict = prediction_from_model(model, selected_hour) # CORRECTION : 'selected_hour' parameter missing

        color_pred_map = {0:["Prédiction : Libre", "green"], 1:["Prédiction : Dense", "orange"], 2:["Prédiction : Bloqué", "red"]}

        return render_template('home.html', graph_json=graph_json, text_pred=color_pred_map[cat_predict][0], color_pred=color_pred_map[cat_predict][1])  # CORRECTION : Tempalte is named index.html, not home.html

    else:

        fig_map = create_figure(data)
        graph_json = fig_map.to_json

        return render_template('index.html', graph_json=graph_json) # CORRECTION : Template is named index.html, not home.html

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    dashboard.config.init_from(file='config.cfg')
    dashboard.config.enable_logging = True
    dashboard.bind(app)
    dashboard.config.monitor_level = 3
    app.run(debug=True)
