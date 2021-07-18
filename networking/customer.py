import cv2 
import zmq 
import time 

class Customer:
	def __init__(self, client_id, source, address, interval=1):
		self.interval = interval 
		self.client_id = client_id  
		self.source = source 
		self.address = address
		self.context = zmq.Context()
		self.socket = self.context.socket(zmq.PUSH)  # send to server 


	def start(self):
		self.socket.connect(self.address)
		self.socket.send_pyobj({
			'type': 'handshake', 
			'contents': {
				'client_id': self.client_id, 
				'image': None 
			}
		})

		print('... blablabla ...')

		capture = cv2.VideoCapture(self.source)
		keep_loop = True 
		while keep_loop:
			capture_status, bgr_frame = capture.read()
			key_code = cv2.waitKey(25) & 0xFF 
			keep_loop = capture_status and key_code != 27
			if keep_loop:
				resized_frame  = cv2.resize(bgr_frame, (640, 480)) 
				gray_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
				self.socket.send_pyobj({
					'type': 'capture',
					'contents': {
						'client_id': self.client_id, 
						'image': resized_frame
					}
				})

				time.sleep(self.interval)
				cv2.imshow('...', resized_frame)
		# end loop 
		self.socket.send_pyobj({
			'type': 'terminate', 
			'contents': {
				'client_id': self.client_id, 
				'image': None 
			}
		})

		self.socket.close()
		self.context.term()


if __name__ == '__main__':
	cst = Customer(100, '../vision-ai-poc/000.mp4', 'tcp://localhost:5900', 0.01)
	cst.start()

#  bgr_image 640x480*3 : red, green, blue => gray_image => resized_gray_image => 64x64 matrix  



