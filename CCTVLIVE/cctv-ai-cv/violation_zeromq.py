import cv2
import zmq
import pickle
from settings import PUB_PORT


def process_frames(pub_port, subscribed_topic):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://localhost:{pub_port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, subscribed_topic)
    
    topic, frame_bytes = socket.recv_multipart()
    frame = pickle.loads(frame_bytes)
    return frame


if __name__ == '__main__':
    subscribed_topics = '5551'
    process_frames(PUB_PORT, subscribed_topics)
