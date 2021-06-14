from flask import Flask, render_template, url_for, flash, redirect
from flask import request
from flask import send_from_directory
from flask_socketio import SocketIO

import numpy as np
import tensorflow
from tensorflow import keras
import tensorflow as tf
import os
from tensorflow.keras.models import load_model


#from this import SQLAlchemy
app=Flask(__name__,template_folder='template')


app.config['SECRET_KEY'] = "UddA58IkCqP5nZkwEzA7YA"



dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'



#global graph
#graph = tf.get_default_graph()
model1 = tensorflow.keras.models.load_model("pneumonia_model.h5")


#pneumonia
def api1(full_path):
    #with graph.as_default():
    data = keras.preprocessing.image.load_img(full_path, target_size=(64, 64, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0/ 255
    predicted = model1.predict(data)
    return predicted

#Pneumonia
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
            result = api1(full_name)
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

#logged in Home page
@app.route("/")
@app.route("/home")
def index2():
    return render_template("home.html")

@app.route("/Pneumonia")
def Pneumonia():
    return render_template("pneumonia.html")


if __name__ == "__main__":
	app.run(debug=True)
