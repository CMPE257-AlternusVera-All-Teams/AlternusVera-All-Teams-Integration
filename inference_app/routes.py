from flask import current_app as app
from flask import request, make_response, jsonify, render_template
from flask_cors import cross_origin
from sigma.SIGMA import NetworkBasedPredictor, MalicousAccount, Credibility, VerifiableAuthenticity
from shining_unicorns.SHINING_UNICORNS import ContentStatistics
from pygooglenews import GoogleNews
import en_core_web_md
import os


nlp = en_core_web_md.load()
nbp = NetworkBasedPredictor(os.path.relpath('sigma/networkbased.pkl'))
ma = MalicousAccount(os.path.relpath('sigma/malicious_account_MLP.pkl'))
cd = Credibility(os.path.relpath('sigma/credibility_model.pkl'))
va = VerifiableAuthenticity(os.path.relpath('sigma/VerifiableAuthenticity_PickledModel.pkl'))
cs = ContentStatistics(os.path.relpath('shining_unicorns/finalized_model3.sav'))


def search_news(text):
  googlenews = GoogleNews(lang='en')
  n = googlenews.search(text)
  return n['entries'][0]['title'], n['entries'][0]['source']['title']


@app.route("/")
@cross_origin()
def say_hello():
    return render_template("index.html.jinja")

@app.route("/prediction", methods=["POST"])
@cross_origin()
def get_prediction():
    body = request.json['text']
    venue = request.json['venue']
    print(body, venue)
    prediction = nbp.predict(body, nlp)
    ma_prediction = ma.predict(body, nlp)
    cd_pred = cd.predict(body, nlp)
    va_pred = va.predict(venue)
    cs_pred = cs.predict(body, None)
    message = {
        "prediction": prediction,
        "ma_pred": ma_prediction[1],
        "cd_pred": cd_pred[1],
        "va_pred": va_pred,
        "cs_pred": cs_pred
    }
    return make_response(jsonify(message), 200)


@app.route("/realtime-prediction/<topic>", methods=["GET"])
@cross_origin()
def get_realtime_prediction(topic):
    text, source = search_news(topic)
    nb_pred = nbp.predict(text, nlp)
    ma_pred = ma.predict(text, nlp)
    cd_pred = cd.predict(text, nlp)
    va_pred = va.predict(source)
    cs_pred = cs.predict(text, None)
    message = {
        "article": text,
        "publisher": source,
        "prediction": nb_pred,
        "ma_pred": ma_pred[1],
        "cd_pred": cd_pred[1],
        "va_pred": va_pred,
        "cs_pred": cs_pred
    }
    return make_response(jsonify(message), 200)



