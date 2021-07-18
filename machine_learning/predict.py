import dlib 
import click 
import numpy as np 

import torch as th
import itertools as it, functools as ft  

from libraries.strategies import *

# zmq server pull frame from client  
# store frame on queue : Q0, Q1, ..., QN 
# worker pull frame from queue and apply describe 
# worker send metrics to scheduler
# scheduler send redifine all media by sending command to remote screen! 



@click.command()
@click.option('-i', '--input_path', help='input image', type=click.Path(True), required=True)
@click.option('-s', '--shape_predictor_path', help='path to shape predictor model', default='dlib_requirements/shape_predictor_68_face_landmarks.dat', show_default=True)
@click.option('-f', '--face_recognizer_path', help='path to face recognizer model', default='dlib_requirements/face_recognition_resnet_model_v1.dat', show_default=True)
@click.option('-g', '--g_model_path', help='path to gender recognizer model')
@click.option('-c', '--s_model_path', help='path to skin recognizer model')
@click.option('-a', '--a_model_path', help='path to age modles directiry')
def describe(input_path, shape_predictor_path, face_recognizer_path, g_model_path, s_model_path, a_model_path):
	face_detector = get_face_detector()
	shape_predictor = get_shape_predictor(shape_predictor_path)
	face_recognizer = get_face_recognizer(face_recognizer_path)
	
	m_paths = sorted(glob(path.join(a_model_path, '*.pt')))
	a_model = [ th.load(mp) for mp in m_paths ]
	for mdl in a_model:
		mdl.eval()
	
	g_model = th.load(g_model_path)
	s_model = th.load(s_model_path)

	g_model.eval()
	s_model.eval()

	H1, W1 = 480, 640

	bgr_image = read_image(input_path)
	rgb_image = to_rgb(bgr_image)
	H0, W0, _ = bgr_image.shape 
	resized_bgr_image = cv2.resize(bgr_image, (W1, H1))

	regions = find_face_regions(rgb_image, face_detector)
	if len(regions) > 0:
		face_features = []
		face_positions = []
		for roi in regions:
			coord = get_coordinates(roi)
			landmarks = get_landmarks(rgb_image, roi, shape_predictor)
			face_embedding = get_128d_embeddings(rgb_image, landmarks, face_recognizer)
			face_features.append(face_embedding)
			face_positions.append(coord)

		stacked_features = np.vstack(face_features)
		tensors_features = th.from_numpy(stacked_features).float()
		with th.no_grad():
			predicted_g = th.softmax(g_model(tensors_features), dim=1)
			predicted_s = th.softmax(s_model(tensors_features), dim=1)
			predicted_g = th.argmax(predicted_g, dim=1).numpy().tolist()
			predicted_s = th.argmax(predicted_s, dim=1).numpy().tolist()

		acc = []
		for row in stacked_features:
			response = recursive_prediction(m_paths, a_model, None, 0, 0, row, None, None)
			response = build_interval(response)
			acc.append(response)

		output = [ {'G': g, 'S': s, 'A': a}  for g, s, a in zip(predicted_g, predicted_s, acc)]
		print(output)

	cv2.imshow('screen', resized_bgr_image)
	cv2.waitKey(0)

if __name__ == '__main__':
	describe()
