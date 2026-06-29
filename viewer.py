import socket
import struct
import cv2
import numpy as np

HOST = "0.0.0.0"
PORT = 9999

server = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)

server.bind((HOST, PORT))
server.listen(1)

print("Aguardando conexão...")

conn, addr = server.accept()

print("Conectado:", addr)

while True:

    raw_size = b""

    while len(raw_size) < 4:
        raw_size += conn.recv(
            4 - len(raw_size)
        )

    size = struct.unpack(
        ">L",
        raw_size
    )[0]

    data = b""

    while len(data) < size:
        data += conn.recv(
            min(4096, size-len(data))
        )

    jpg = np.frombuffer(
        data,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        jpg,
        cv2.IMREAD_COLOR
    )

    cv2.imshow(
        "QCar Stream",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

conn.close()
cv2.destroyAllWindows()