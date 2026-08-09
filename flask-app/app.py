from flask import Flask, render_template, request
import mlflow
import dagshub
import pickle

dagshub.init(repo_owner='singhsumitt05', repo_name='mlops-mini', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/singhsumitt05/mlops-mini.mlflow')

from preprocessing_utility import normalize_text


app = Flask(__name__)

#load model from model registry 

model_name = "my_model" 
model_alias = "champion" 
model_uri = f"models:/{model_name}@{model_alias}" 
model = mlflow.pyfunc.load_model(model_uri) 
print( f"Loaded Production model: {model_name}@{model_alias}" )

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

    result = model.predict(features)

    return render_template('index.html', result=result[0])


app.run(debug=True)


#load model from model registry 
"""
we will have to load the model outside of the predict function
because the predict will be called n times by the user 
and loading the model n times is not feasable
hence we will load the model outside in global space once 

"""





