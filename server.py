import os
import uuid
from flask import Flask, flash, request, redirect

from whispr import transcribe
from util import remove_file


UPLOAD_FOLDER = 'audio'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/save-record', methods=['POST'])
def save_record():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    file_name = str(uuid.uuid4()) + ".wav"
    full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file.save(full_file_name)
    
    res = transcribe(full_file_name)
    remove_file(full_file_name)
    return res    
    # print(full_file_name)
    # return '<h1>Success</h1>'
    



# @app.route('/transcribe', methods=['GET'])
# def transcribe():




if __name__ == '__main__':
    app.run()
