from flask import Flask, request, jsonify
import pymongo
from datetime import datetime
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
import numpy as np
import secrets
import cv2


UPLOAD_FOLDER = "./models"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
client = pymongo.MongoClient('localhost', 27017)

DATA_DB = client.SubmittedData
DB_COLLEC = DATA_DB.DATA


ALLOWABLE_EXTENSTIONS = ['h5', 'pkl', 'sav']


@app.route("/", methods=["POST", "GET"])
def home():
    req_data_json = request.json
    if(len(request.files)):
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_ext = filename.split('.')[1]
        if file_ext not in ALLOWABLE_EXTENSTIONS:
            return jsonify(error=f"Sorry .{file_ext} extension is not allowed")
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if(os.path.exists(saved_path)):
            return jsonify(error=f"{saved_path} exits, try different model name")
        else:
            file.save(saved_path)
            file.close()
            return jsonify(filename=filename, saved_path=saved_path)
    else:
        id = secrets.token_hex()
        details = {
            "_id": str(id),
            "name": req_data_json['name'],
            "input_fields": req_data_json['input_fields'],
            "model_size": req_data_json['model_size']}
        DB_COLLEC.insert_one(details)
        return f"Your unique token is {id}. Pass this id while making requests\n"


@app.route("/<id>", methods=["POST", "GET"])
def predictor(id):
    id_model_map = DB_COLLEC.find_one({"_id": id})

    req_data_json = request.json
    file_name = id_model_map['name']
    file_ext = file_name.split('.')[1]

    uploaded_file = request.files['file']
    print(uploaded_file)

    if(os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], file_name))):
        if(file_ext.strip() == "h5"):
            predicted_value = do_the_h5thing(
                id_model_map, req_data_json, uploaded_file)
            if(predicted_value):
                return jsonify(predictions=predicted_value)
            else:
                return jsonify(error="Error finding predictions")
    else:
        return jsonify(error="Error, model doesn't exist")


def do_the_h5thing(model_details, request_from_user, img_file):
    model_filename = os.path.join(
        app.config['UPLOAD_FOLDER'], model_details['name'])
    model = load_model(model_filename)
    predictions = 123123123

    # receive image as base64, instead of passing image as whole

    if(request_from_user):
        X_inp = [val for val in request_from_user.values()]
        predictions = model.predict(np.array([X_inp, ]))[0]
        predictions = np.round(predictions, 4).tolist()
    else:
        img_array = cv2.resize(cv2.imread(img_file), (128, 128))
        predictions = model.predict(
            np.array(img_array).reshape(-1, 128, 128, 3))
    return predictions


if __name__ == "__main__":
    app.run(debug=True)
