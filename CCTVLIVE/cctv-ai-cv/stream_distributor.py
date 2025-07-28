import zmq
from settings import PUB_PORT, CAMERA_PORTS


def distribute_frames(push_ports, pub_port):
    context = zmq.Context()
    pull_sockets = []

    for port in push_ports:
        socket = context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{port}")
        pull_sockets.append(socket)

    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://*:{pub_port}")

    poller = zmq.Poller()
    for socket in pull_sockets:
        poller.register(socket, zmq.POLLIN)

    while True:
        socks = dict(poller.poll())

        for socket in pull_sockets:
            if socket in socks and socks[socket] == zmq.POLLIN:
                frame_bytes = socket.recv()
                topic = str(push_ports[pull_sockets.index(socket)])
                pub_socket.send_multipart([topic.encode(), frame_bytes])

    # Cleanup
    for socket in pull_sockets:
        socket.close()
    pub_socket.close()


if __name__ == '__main__':
    distribute_frames(CAMERA_PORTS, PUB_PORT)
