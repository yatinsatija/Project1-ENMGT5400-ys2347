import cv2
import numpy as np
import socket

# ESP32 Camera Details
ESP32_IP = "172.20.10.8"  # Change this to your ESP32's IP
ESP32_PORT = 9999
ESP32_USER = "Yatin"
ESP32_PASS = "210899"

# Expected BMP Frame Size (Ensure this matches ESP32 output)
BMP_FRAME_SIZE = 10294
BMP_HEADER_SIZE = 1078
BMP_PIXEL_DATA_SIZE = 9216

# Open TCP Connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((ESP32_IP, ESP32_PORT))

# Send authentication request
auth_request = f"GET /{ESP32_USER}/{ESP32_PASS} HTTP/1.1\r\nHost: {ESP32_IP}\r\n\r\n"
sock.sendall(auth_request.encode())

while True:
   try:
       # Receive full BMP frame
       bmp_data = b""
       while len(bmp_data) < BMP_FRAME_SIZE:
           chunk = sock.recv(BMP_FRAME_SIZE - len(bmp_data))
           if not chunk:
               print("Connection lost!")
               break
           bmp_data += chunk
           
       # Validate complete BMP frame
       if len(bmp_data) != BMP_FRAME_SIZE:
           print(f"Incomplete BMP frame received ({len(bmp_data)} bytes). Expected {BMP_FRAME_SIZE}. Retrying...")
           continue
           
       # Extract pixel data from BMP (Skip header)
       pixel_data = bmp_data[BMP_HEADER_SIZE : BMP_HEADER_SIZE + BMP_PIXEL_DATA_SIZE]
       
       # Convert to NumPy array and reshape
       image_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape((96, 96))
       
       # Flip image (BMP stores bottom-up)
       image_array = np.flipud(image_array)
       
       # Display frame
       cv2.namedWindow("ESP32 Live Stream", cv2.WINDOW_NORMAL)  # Allows window resizing
       cv2.imshow("ESP32 Live Stream", image_array)
       cv2.resizeWindow("ESP32 Live Stream", 400, 400)  # Set initial window size
       
       # Exit on 'q' press
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
           
   except Exception as e:
       print(f" Stream Error: {e}")
       break
cv2.destroyAllWindows()
sock.close()