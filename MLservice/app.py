from urllib import response
from services.ml_model import MLMgr
from flask import Flask, make_response, flash, request, redirect, url_for

import cv2
from skimage import io
import numpy as np
import os
import yaml
from PIL import Image

from src.utils import detectron2_prediction, get_outputs_detectron2, draw_bbox_infer
FILE_INFER_CONFIG = os.path.join("config", "inference.yaml")
with open(FILE_INFER_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

ai_model = MLMgr.GetModel()

app = Flask(__name__)

@app.get('/')
def get_test():
	img = io.imread('../samples/0e91263f0b1925ff01d5ebde3ce65e1a.jpg')	
	img = np.array(img)
	outputs = detectron2_prediction(ai_model, img)
	pred_bboxes, pred_confidence_scores, pred_classes = get_outputs_detectron2(outputs)
	pred_bboxes = pred_bboxes.detach().numpy().astype(int)
	pred_confidence_scores = pred_confidence_scores.detach().numpy()
	pred_confidence_scores = np.round(pred_confidence_scores, 2)
	pred_classes = pred_classes.detach().numpy().astype(int)

	img_after = draw_bbox_infer(img, pred_bboxes, 
								pred_classes, pred_confidence_scores,
								params["CLASSES_NAME"], params["COLOR"], 5)

	# convert to picture
	retval, buffer = cv2.imencode('.png', img_after)
	response = make_response(buffer.tobytes())
	response.headers['Content-Type'] = 'image/png'
	return response 

@app.post('/')
def post_test():
	# check if the post request has the file part
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	# if user does not select file, browser also
	# submit a empty part without filename
	if file.filename == '':
		flash('No selected file')
		return redirect(request.url)
	img = Image.open(file)
	img = np.array(img.convert("RGB"))
	retval, buffer = cv2.imencode('.png', img)
	outputs = detectron2_prediction(ai_model, img)
	pred_bboxes, pred_confidence_scores, pred_classes = get_outputs_detectron2(outputs)
	pred_bboxes = pred_bboxes.detach().numpy().astype(int)
	pred_confidence_scores = pred_confidence_scores.detach().numpy()
	pred_confidence_scores = np.round(pred_confidence_scores, 2)
	pred_classes = pred_classes.detach().numpy().astype(int)

	img_after = draw_bbox_infer(img, pred_bboxes, 
								pred_classes, pred_confidence_scores,
								params["CLASSES_NAME"], params["COLOR"], 5)

	# convert to picture
	retval, buffer = cv2.imencode('.png', img_after)
	response = make_response(buffer.tobytes())
	response.headers['Content-Type'] = 'image/png'
	return response 

if __name__ == '__main__':		
	
	app.run(debug=True)