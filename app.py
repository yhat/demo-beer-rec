#!/usr/bin/env python

from flask import Flask, request, render_template, url_for, Response, json
from yhat import Yhat
import os
import requests
from requests.auth import HTTPBasicAuth
app = Flask(__name__)


def make_prediction(data):
    auth=('colin', '25b58a60-d246-4466-b354-80e20d71225e')
    r = requests.post('https://promote.c.yhat.com/colin/models/BeerRecommender/predict', json = data, auth = auth)
    return r.json()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # print request.json['beers']
        try:
            pred = make_prediction({"beers": request.json['beers'] })
            return Response(json.dumps(pred), mimetype='application/json')
        except Exception, e:
            print e
            return Response(json.dumps({"error": str(e)}),
                            mimetype='application/json')
    else:
        # static files
        css_url = url_for('static', filename='css/main.css')
        jquery_url = url_for('static', filename='js/jquery-1.10.2.min.js')
        beers_url = url_for('static', filename='js/beers.js')
        highlight_url = url_for('static', filename='js/code.js')
        js_url = url_for('static', filename='js/main.js')
        return render_template('index.html', css_url=css_url,
                               jquery_url=jquery_url, beers_url=beers_url,
                               js_url=js_url, highlight_url=highlight_url)

if __name__ == '__main__':
    app.run(debug=True)
