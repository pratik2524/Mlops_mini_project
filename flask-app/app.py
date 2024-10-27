from  flask import Flask,request, render_template
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import pickle
import requests

app = Flask(__name__)

#load model from model registry
mlflow.set_tracking_uri('https://dagshub.com/pratik2524/Mlops_mini_project.mlflow')
dagshub.init(repo_owner='pratik2524', repo_name='Mlops_mini_project', mlflow=True)



model_name = "my_model"
model_version = 4

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html',result=None)


@app.route('/predict',methods=['POST'])
def predict():

    text = request.form['text']

    #Clean the text
    text = normalize_text(text)

    #BOW
    features = vectorizer.transform([text])

    #prediction
    result = model.predict(features)

    #show to User
    return render_template('index.html',result=result[0])


app.run(debug=True)