import socket


class UdpServer:
    def __init__(self, address='localhost', port=6340):
        self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.local_addr = address
        self.port = port
        self.server_socket.bind((self.local_addr, self.port))
        print("UDP Server is running!")

    def disconnect(self):
        self.server_socket.close()
        print("Disconnection Successful")

    def receive_msg(self):
        return self.server_socket.recvfrom(1024)[0]

    def send_msg(self, message, address='localhost', port=6340):
        self.server_socket.sendto(message, (address, port))


class UdpClient:
    def __init__(self, address='localhost', port=6340):
        self.client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.addr = address
        self.port = port
        self.client_socket.connect((self.addr, self.port))
        print("UDP Client is Running!")

    def disconnect(self):
        self.client_socket.close()
        print("Disconnection Successful")

    def send_msg(self, message, address='localhost', port=6340):
        self.client_socket.sendto(message, (address, port))

    def receive_msg(self):
        return self.client_socket.recvfrom(1024)[1]
