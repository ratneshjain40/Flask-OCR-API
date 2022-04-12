from flask import Flask, render_template, request, send_file
from flask_restful import Resource, Api
from io import BytesIO 
import numpy as np
import cv2
from imagereader import *
import base64

app = Flask(__name__)
api = Api(app)

class ImageReader(Resource):

    def post(self):
        stream = request.files['file'].read()
        # convert to numpy array
        npimg = np.fromstring(stream, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        
        image, image_text = run_npl(img)
        
        is_success, buffer = cv2.imencode(".jpg", image)
        # io_buf = BytesIO(buffer)
        # io_buf.seek(0)
        
        img_as_text = base64.b64encode(buffer).decode('ascii')
        
        # data = b64encode(png_output.getvalue()).decode('ascii')
        data = {
            "img": img_as_text,
            "text": image_text
        }
        
        return data

api.add_resource(ImageReader, "/img_reader")
app.run(port=5000, debug=True)
