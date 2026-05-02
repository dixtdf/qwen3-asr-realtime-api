import socket

class UDPTransmitter:
    def __init__(self, host="127.0.0.1", port=9000):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_transcript(self, text: str):
        try:
            payload = text.encode("utf-8")
            self.sock.sendto(payload, self.addr)
        except Exception as e:
            print(f"UDP send failed: {e}")
