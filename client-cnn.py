import cv2
import numpy as np
import socket

# ESP32 Camera Configuration
ESP32_ADDRESS = ("172.20.10.8", 9999)  # Update with correct ESP32 details
USERNAME = "Yatin"  # ESP32 Authentication Username
PASSWORD = "210899"  # ESP32 Authentication Password

# BMP Frame Specifications
TOTAL_FRAME_SIZE = 10294  # Total expected size of the BMP frame
HEADER_LENGTH = 1078  # BMP header length (metadata before pixel data)
IMAGE_DATA_SIZE = 9216  # Size of pixel data
FRAME_DIMENSIONS = (96, 96)  # Image dimensions (Height, Width)

# Establish TCP Connection to ESP32 Camera
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create a TCP socket
client_socket.connect(ESP32_ADDRESS)  # Connect to ESP32 at specified IP and Port

# Authentication Request
login_payload = f"GET /{USERNAME}/{PASSWORD} HTTP/1.1\r\nHost: {ESP32_ADDRESS[0]}\r\n\r\n"  # HTTP-like authentication request
client_socket.sendall(login_payload.encode())  # Send authentication request to ESP32

# Setup OpenCV Window for Display
cv2.namedWindow("ESP32 Feed", cv2.WINDOW_NORMAL)  # Create a resizable window
cv2.resizeWindow("ESP32 Feed", 400, 400)  # Set initial window size

while True:
    try:
        # Receive BMP frame data from ESP32
        frame_buffer = bytearray()  # Initialize an empty buffer
        while len(frame_buffer) < TOTAL_FRAME_SIZE:  # Ensure the complete frame is received
            segment = client_socket.recv(TOTAL_FRAME_SIZE - len(frame_buffer))  # Receive data chunk
            if not segment:  # Check for disconnection
                print("Connection lost.")
                break
            frame_buffer.extend(segment)  # Append received data to the buffer

        # Validate if complete frame is received
        if len(frame_buffer) != TOTAL_FRAME_SIZE:
            print(f"Incomplete frame received ({len(frame_buffer)} bytes). Retrying...")
            continue  # Skip frame processing and retry

        # Extract and process image data
        grayscale_data = frame_buffer[HEADER_LENGTH:HEADER_LENGTH + IMAGE_DATA_SIZE]  # Extract pixel data
        frame_matrix = np.frombuffer(grayscale_data, dtype=np.uint8).reshape(FRAME_DIMENSIONS)  # Convert to NumPy array
        frame_matrix = np.flipud(frame_matrix)  # Adjust BMP orientation (BMP stores images bottom-up)

        # Display the image using OpenCV
        cv2.imshow("ESP32 Feed", frame_matrix)  # Show the live frame

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as error:
        print(f"Streaming error: {error}")  # Print any streaming errors
        break

# Cleanup resources
cv2.destroyAllWindows()  # Close OpenCV windows
client_socket.close()  # Close socket connection
