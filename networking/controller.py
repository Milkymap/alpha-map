import zmq 
import multiprocessing as mp 
import time 

import numpy as np 
from libraries.strategies import *

class Controller:
	def __init__(self, port, nb_memories, workers_per_memory):
		assert workers_per_memory >= 1 and nb_memories >= 2 
		self.port = port 
		self.context = zmq.Context()
		self.images_memories = []
		for _ in range(nb_memories):
			self.images_memories.append(mp.Queue())
		self.metrics_memory = mp.Queue()  # metric store 
		self.nb_memories = nb_memories
		self.workers_per_memory = workers_per_memory
		self.config = {
			'face_recognizer_path': 'dlib_requirements/face_recognition_resnet_model_v1.dat', 
			'shape_predictor_path': 'dlib_requirements/shape_predictor_68_face_landmarks.dat', 
			'a_model_path': 'models/a_dir', 
			'g_model_path': 'models/g_dir/model_00_00_GN.pt', 
			's_model_path': 'models/s_dir/model_00_00_SK.pt'
		}

	def server(self, endpoint, images_memories, metrics_memory):
		try:
			clients_status = dict()
			socket = self.context.socket(zmq.PULL)
			poller = zmq.Poller()
			socket.bind(endpoint)
			poller.register(socket)
			index = 0 
			keep_loop = True 
			while keep_loop:
				events = dict(poller.poll(10))  # for each 100ms poll events 
				if socket in events:
					if events[socket] == zmq.POLLIN: 
						incoming_data = socket.recv_pyobj()  # from client 
						if incoming_data['type'] == 'handshake':
							clients_status[incoming_data['contents']['client_id']] = 1  
						if incoming_data['type'] == 'terminate':
							clients_status[incoming_data['contents']['client_id']] = 0 
						if incoming_data['type'] == 'capture':
							print(' ... [incoming frame from customer] ... ')
							images_memories[index].put(incoming_data['contents'])
							index = (index + 1) % self.nb_memories
				if not metrics_memory.empty():
					response = metrics_memory.get()
					# send command to client scheduler ...! 

			# end while 		
		except KeyboardInterrupt as e:
			print(e)
		except Exception as e:
			pass 
		finally:
			while np.any(list(clients_status.values())) :  # block until all clients disconnect 
				events = dict(poller.poll(100))  # for each 100ms poll events 
				if socket in events:
					if events[socket] == zmq.POLLIN: 
						incoming_data = socket.recv_pyobj()  # from client 
						if incoming_data['type'] == 'terminate':
							clients_status[incoming_data['contents']['client_id']] = 0 
			# end while loop 

			poller.unregister(socket)
			socket.close()
			self.context.term()


	def worker(self, pid, image_memory, metrics_memory, config):
		try:
			H1, W1 = 480, 640

			face_detector = get_face_detector()
			shape_predictor = get_shape_predictor(config['shape_predictor_path'])
			face_recognizer = get_face_recognizer(config['face_recognizer_path'])
			
			m_paths = sorted(glob(path.join(config['a_model_path'], '*.pt')))
			a_model = [ th.load(mp) for mp in m_paths ]
			for mdl in a_model:
				mdl.eval()
			
			g_model = th.load(config['g_model_path'])
			s_model = th.load(config['s_model_path'])

			g_model.eval()
			s_model.eval()
			
			keep_loop = True
			while keep_loop:

				if not image_memory.empty():
					print(f'worker {pid:03d} got data from server ')
					incoming_data = image_memory.get()
					bgr_image = incoming_data['image']
					#bgr_image = cv2.cvtColor(incoming_data['image'], cv2.COLOR_GRAY2BGR)
					#bgr_image = cv2.resize(bgr_image, (640, 480))
					print(bgr_image.shape)
					rgb_image = to_rgb(bgr_image)
					cv2.imwrite(f'img_{pid:03d}.jpg', bgr_image)
					H0, W0, _ = bgr_image.shape 
					resized_bgr_image = cv2.resize(bgr_image, (W1, H1))

					regions = find_face_regions(rgb_image, face_detector)
					print(regions)  # rectangles 
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
						response = (incoming_data['client_id'], output)
						print(response)
					metrics_memory.put(response)
			# end while 
		except KeyboardInterrupt as e:
			print(e)
		except Exception as e:
			pass 
		finally:
			pass

	def start(self):
		server_process = mp.Process(
			target=self.server, 
			args=[
				f'tcp://*:{self.port}', 
				self.images_memories, 
				self.metrics_memory
			]
		)
		server_process.start()

		worker_process = []
		for idx in range(self.nb_memories):
			for jdx in range(self.workers_per_memory): 
				w = mp.Process(
					target=self.worker, 
					args=[
						idx * self.workers_per_memory + jdx, 
						self.images_memories[idx], 
						self.metrics_memory, 
						self.config
					]
				)
				worker_process.append(w)
				worker_process[-1].start()
		# ...


if __name__ == '__main__':
	ctl = Controller(port=5900, nb_memories=2, workers_per_memory=3)
	ctl.start()

