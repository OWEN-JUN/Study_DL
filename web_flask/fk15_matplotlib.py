from io import BytesIO
from flask import Flask, render_template, send_file, make_response
import flask
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import matplotlib.pyplot as plt

app = flask.Flask(__name__)

@app.route("/mypic")
def mypic():
    return flask.render_template("mypic.html")

@app.route("/plot")
def plot():
    fig, axis = plt.subplots(1)

    y = [1,2,3,4,5]
    x = [0,2,1,3,4]

    axis.plot(x,y)
    canvas = FigureCanvas(fig)

    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    port = 5000
    app.debug = False
    app.run(port=port)