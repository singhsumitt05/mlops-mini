from flask import Flask, render_template, request
import mlflow
import dagshub
import pickle
import pandas as pd
import numpy as np
import os

from preprocessing_utility import normalize_text

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "singhsumitt05"
repo_name = "mlops-mini"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


app = Flask(__name__)

#load model from model registry 

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None


model_name = "my_model"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)


vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html", result=None)


@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    #clean text entered by user 
    text = normalize_text(text)

    #apply bow 
    features = vectorizer.transform([text]) #vectorizer needs a list

    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    result = model.predict(features_df)

    return render_template('index.html', result=result[0])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

#load model from model registry 
"""
we will have to load the model outside of the predict function
because the predict will be called n times by the user 
and loading the model n times is not feasable
hence we will load the model outside in global space once 

"""





