import plotly.express as px
import numpy as np



def create_figure(data):

    fig_map = px.scatter_mapbox(
            data,
            title="Traffic en temps réel",
            color="traffic",
            lat="lat",
            lon="lon",
            color_discrete_map={'freeFlow':'green', 'heavy':'orange', 'congested':'red'},
            zoom=10, # CORRECTION : comma was forgotten
            height=500,
            mapbox_style="carto-positron"
    )

    return fig_map

def prediction_from_model(model, hour_to_predict):

    input_pred = np.array([0]*24) # CORRECTION : there must be 24 hours not 25
    input_pred[int(hour_to_predict)] = 1

    cat_predict = np.argmax(model.predict(np.array([input_pred])))

    return cat_predict