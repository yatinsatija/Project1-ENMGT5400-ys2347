import esp  # ESP-specific module
from Wifi import Sta  # WiFi connection handler
import socket as soc  # Socket for TCP communication
import camera  # Camera module
from time import sleep  # Sleep function for delays
from camera import Camera, PixelFormat, FrameSize  # Camera settings
from image_preprocessing import resize_96x96_to_32x32_averaged_and_threshold  # Image preprocessing function
import emlearn_cnn_fp32 as emlearn_cnn  # Machine learning model handler
import gc  # Garbage collector for memory management
import array  # Array module for efficient data storage

# Model and Recognition Parameters
MODEL = 'model.tmdl'  # Path to the trained model
RECOGNITION_THRESHOLD = 0.74  # Confidence threshold for classification
CLASS_NAMES = ["Rock", "Paper", "Scissors"]  # Output class names

# Camera Configuration Parameters
CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],  # Data pins for the camera
    "vsync_pin": 38,  # Vertical sync pin
    "href_pin": 47,  # Horizontal reference pin
    "sda_pin": 40,  # I2C data pin
    "scl_pin": 39,  # I2C clock pin
    "pclk_pin": 13,  # Pixel clock pin
    "xclk_pin": 10,  # External clock pin
    "xclk_freq": 20000000,  # External clock frequency (20MHz)
    "powerdown_pin": -1,  # No power down pin
    "reset_pin": -1,  # No reset pin
    "frame_size": FrameSize.R96X96,  # Capture resolution: 96x96
    "pixel_format": PixelFormat.GRAYSCALE  
}

def argmax(arr):
    """Returns the index of the max value in an array."""
    return max(range(len(arr)), key=lambda i: arr[i])

def debug_data(data, label, sample_size=10):
    """Print debugging information about a data object."""
    print(f"🔍 DEBUG: {label} → Type: {type(data)}, Length: {len(data)}, Sample: {data[:sample_size]}")
    
esp.osdebug(None)  # Disable debug logging from ESP

# Authentication Credentials
UID = const('Yatin')  # Username for authentication
PWD = const('210899')  # Password for authentication

# Initialize and Configure Camera
cam = Camera(**CAMERA_PARAMETERS)  # Initialize camera with defined parameters
cam.init()  # Start the camera
print("Camera Initialized")

# Load Trained Model
with open(MODEL, 'rb') as f:
    model_data = array.array('B', f.read())  # Read model data into an array
    print("Model Data Loaded..")
    gc.collect()  # Run garbage collector to free memory
    model = emlearn_cnn.new(model_data)  # Load model into inference engine
    print("Model Loaded..")

# Setup WiFi Connection
sta = Sta()  # WiFi client instance
sta.wlan.disconnect()  # Ensure previous connections are terminated
AP = const('dlink-2391')  # WiFi SSID
PW = const('f34RT114k')  # WiFi Password
sta.connect(AP, PW)  # Connect to WiFi network
sta.wait()  # Wait for connection

if not sta.wlan.isconnected():
    print("WiFi not connected. Restart ESP")
    exit()

print(f"WiFi connected: {sta.status()[0]}")

# Setup TCP Server for Communication
port = 9999  # Port number for TCP communication
addr = soc.getaddrinfo('0.0.0.0', port)[0][-1]  # Get server address
server_socket = soc.socket(soc.AF_INET, soc.SOCK_STREAM)  # Create TCP socket
server_socket.setsockopt(soc.SOL_SOCKET, soc.SO_REUSEADDR, 1)  # Allow address reuse
server_socket.bind(addr)  # Bind socket to address
server_socket.listen(1)  # Listen for incoming connections

print(f"Server listening on {addr}")

while True:
    client_socket, client_address = server_socket.accept()  # Accept incoming client connection
    print(f"Connection from: {client_address}")
    
    # Authenticate Client
    request = client_socket.recv(200).decode()  # Receive authentication request
    try:
        _, uid, pwd = request.split("\r\n")[0].split()[1].split("/")  # Extract credentials
        if not (uid == UID and pwd == PWD):
            print("Unauthorized client. Closing connection.")
            client_socket.close()
            continue
    except:
        print("Invalid authentication format.")
        client_socket.close()
        continue
    
    print("Client authenticated.")
    probabilities = array.array("f", (0.0 for _ in range(len(CLASS_NAMES))))  # Initialize probability array
    
    # Start Sending JPEG Frames and Predictions
    while True:
        try:
            img_data = cam.capture()  # Capture JPEG image
            if img_data:
                client_socket.sendall(img_data)  # Send JPEG frame to client
                debug_data(img_data, "Captured Image (JPEG)")

                # Ensure it's a bytearray (convert from memoryview)
                img_data = bytearray(img_data)
                debug_data(img_data, "Captured Image (Converted to bytearray)")

                # 📏 Resize and Threshold
                img_32x32 = resize_96x96_to_32x32_averaged_and_threshold(img_data, threshold=128)
                debug_data(img_32x32, "Resized 32x32 Image")

                # ✅ Convert bytearray to array.array('B', ..)
                img_32x32_array = array.array('B', img_32x32)
                debug_data(img_32x32_array, "Converted to array.array('B', ..)")

                # 🔮 Run Model Inference
                model.run(img_32x32_array, probabilities)  # Perform inference
                predicted_class = argmax(probabilities)  # Get predicted class index
                
                # 📡 Send classification result to client
                client_socket.sendall(f"PREDICTION: {CLASS_NAMES[predicted_class]}\n".encode())
                print(f"🖼️ Classified as: {CLASS_NAMES[predicted_class]}")
                
                sleep(3)  # Delay between frames (adjust as needed)
        except Exception as e:
            print(f"Transmission error: {e}")  # Handle errors
            client_socket.close()  # Close connection on error
            break