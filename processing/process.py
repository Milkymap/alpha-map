import click 
import pickle 
import numpy as np 

import mediapipe as mp 
from libraries.strategies import * 

@click.command()
@click.option('--source', help='path to image(face) source data', type=click.Path(True), required=True)
@click.option('--target', help='path where the feature will be stored', type=click.File('wb'), required=True)
def get_face_features(source, target):
	preparator = get_preparator()
	extractor = get_vgg16('vgg16.pt')
	filepaths = pull_files(source, '*.jpg')
	
	nb_images = len(filepaths)

	mp_builder = mp.solutions.face_detection
	mp_drawing = mp.solutions.drawing_utils

	with mp_builder.FaceDetection(min_detection_confidence=0.5) as detector:
		acculumator = []
		for idx, elm in enumerate(filepaths):
			label = int(elm.split('/')[-1].split('_')[0])

			bgr_image = cv_read_image(elm)
			H, W = bgr_image.shape[:2]
			rgb_image = to_rgb(bgr_image)
			response = detector.process(rgb_image)
			if response.detections:
				for roi in response.detections:
					roi_info = roi.location_data.relative_bounding_box
					roi_coordinates = list(op.attrgetter('xmin', 'ymin', 'width', 'height')(roi_info))
					roi_coordinates = np.abs(np.asarray(roi_coordinates)) * np.array([W, H, W, H])
					roi_coordinates = roi_coordinates.astype('int32')
					boxes = th.from_numpy(roi_coordinates[None, :])
					replaced_boxes = replace_boxes((W, H), boxes)
					print(replaced_boxes, '%03d' % label, '%04d' % idx, nb_images)
					x, y, w, h = replaced_boxes[0]
					cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
					tensor_image = cv2th(bgr_image)
					features = get_features(tensor_image, extractor, preparator)
					acculumator.append( (features, label))
			cv2.imshow('000', cv2.resize(bgr_image, (400, 400)))
			cv2.waitKey(25)
		
		
		pickle.dump(acculumator, target)

if __name__ == '__main__':
	print(' ... [processing] ... ')
	get_face_features()