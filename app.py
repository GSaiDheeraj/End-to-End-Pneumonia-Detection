from flask import Flask,render_template, url_for , redirect
#from forms import RegistrationForm, LoginForm
#from sklearn.externals import joblib
from flask import request
import numpy as np
from PIL import Image
from flask import flash
#from flask_sqlalchemy import SQLAlchemy
#from model_class import DiabetesCheck, CancerCheck


import os
from tensorflow import keras
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import send_from_directory
from tensorflow.keras.preprocessing import image
import tensorflow as tf

#from this import SQLAlchemy
app=Flask(__name__,template_folder='template')


app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
#from keras.models import load_model

# global graph
# graph = tf.get_default_graph()
model = load_model('pneumonia_model.h5')


def api(full_path):
    #with graph.as_default():
    data = tensorflow.keras.preprocessing.image.load_img(full_path, target_size=(64, 64, 3))
    #print(data.shape)
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model.predict(data)
    return predicted


# procesing uploaded file and predict it
	
@app.route('/upload11', methods=['POST', 'GET'])
def upload11_file():
    #with graph.as_default():
    if request.method == 'GET':
        return render_template('pneumonia.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {1: 'Healthy', 0: 'Pneumonia-Infected'}
            result = api(full_name)
            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            if accuracy < 85:
                prediction = "Please, Check with the Doctor."
            else:
                prediction = "Result is accurate"

            return render_template('pneumoniapredict.html', image_file_name=file.filename, label=label, accuracy=accuracy,
                                   prediction=prediction)
        except:
            flash("Please select the X-ray image first !!", "danger")
            return redirect(url_for("Pneumonia"))

	

	

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/")

@app.route("/home")
def home():
	return render_template("home.html")

@app.route("/about")
def about():
	return render_template("about.html")

@app.route("/Pneumonia")
def corona():
    return render_template("pneumonia.html")


if __name__ == "__main__":
	app.run(debug=True)
