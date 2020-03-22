from flask import Flask, request, jsonify
import pymongo
from datetime import datetime
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
import numpy as np

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
        id = datetime.now().microsecond
        details = {
            "_id": id,
            "name": req_data_json['name'],
            "input_fields": req_data_json['input_fields'],
            "model_size": req_data_json['model_size']}
        DB_COLLEC.insert_one(details)
        return f"Your unique token is {id}. Pass this id while making requests\n"


@app.route("/<id>", methods=["POST", "GET"])
def predictor(id):
    id_model_map = DB_COLLEC.find_one({"_id": int(id)})

    req_data_json = request.json
    if id_model_map:
        file_name = id_model_map['name']
        if(os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], file_name))):
            predicted_value = do_the_thing(id_model_map, req_data_json)
            return predicted_value
    else:
        return jsonify(error=f"{id} does'nt seem to exist in our database")


def do_the_thing(model_details, request_from_user):
    model_filename = os.path.join(
        app.config['UPLOAD_FOLDER'], model_details['name'])
    model = load_model(model_filename)
    X_inp = []
    for val in request_from_user.values():
        X_inp.append(int(val))
    predictions = model.predict(np.array([X_inp, ]))
    return jsonify(predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True)
