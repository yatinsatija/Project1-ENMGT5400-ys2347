import esp
from Wifi import Sta
import socket as soc
import camera
from time import sleep
from camera import Camera, PixelFormat, FrameSize
from image_preprocessing import resize_96x96_to_32x32_averaged_and_threshold, strip_bmp_header
import emlearn_cnn_fp32 as emlearn_cnn
import gc
import array

MODEL = 'model.tmdl'
RECOGNITION_THRESHOLD = 0.74
CLASS_NAMES = ["Rock", "Paper", "Scissors"]

CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],
    "vsync_pin": 38,
    "href_pin": 47,
    "sda_pin": 40,
    "scl_pin": 39,
    "pclk_pin": 13,
    "xclk_pin": 10,
    "xclk_freq": 20000000,
    "powerdown_pin": -1,
    "reset_pin": -1,
    "frame_size": FrameSize.R96X96,
    "pixel_format": PixelFormat.GRAYSCALE
}

def argmax(arr):
    """Returns the index of the max value in an array."""
    return max(range(len(arr)), key=lambda i: arr[i])

def debug_data(data, label, sample_size=10):
    """Print debugging information about a data object."""
    print(f"🔍 DEBUG: {label} → Type: {type(data)}, Length: {len(data)}, Sample: {data[:sample_size]}")
    
esp.osdebug(None)

UID=const('Yatin')
PWD=const('210899')

cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)

print("Camera Initialized")

with open(MODEL, 'rb') as f:
    model_data = array.array('B', f.read())
    print("Model Data Loaded..")
    gc.collect()
    model = emlearn_cnn.new(model_data)
    print("Model Loaded..")

sta=Sta()
sta.wlan.disconnect()
AP=const('dlink-2391')
PW=const('f34RT114k')
sta.connect(AP,PW)
sta.wait()

if not sta.wlan.isconnected():
    print("Wifi not connected. Restart ESP")
    exit()

print(f"Wifi connected: {sta.status()[0]}")

port =9999
addr=soc.getaddrinfo('0.0.0.0', port)[0][-1]
server_socket=soc.socket(soc.AF_INET, soc.SOCK_STREAM)
server_socket.setsockopt(soc.SOL_SOCKET, soc.SO_REUSEADDR,1)
server_socket.bind(addr)
server_socket.listen(1)

print(f"Server listening on {addr}")

while True:
    client_socket, client_address = server_socket.accept()
    print(f"Connection from: {client_address}")
    
    # Authenticate Client
    request = client_socket.recv(200).decode()
    try:
        _, uid, pwd = request.split("\r\n")[0].split()[1].split("/")
        if not (uid == UID and pwd == PWD):
            print("Unauthorized client. Closing connection.")
            client_socket.close()
            continue
    except:
        print(" Invalid authentication format.")
        client_socket.close()
        continue
    
    print(" Client authenticated.")
    probabilities = array.array("f", (0.0 for _ in range(len(CLASS_NAMES))))
    # Start sending raw BMP frames
    while True:
        try:
            img_data = cam.capture()
            if img_data:
                client_socket.sendall(img_data)# Send full BMP frame
                debug_data(img_data, "Captured Image (Original)")

                # Ensure it's a bytearray (convert from memoryview)
                img_data = bytearray(img_data)
                debug_data(img_data, "Captured Image (Converted to bytearray)")

                # 📏 Resize and Threshold (Fix: Use averaged threshold function)
                img_32x32 = resize_96x96_to_32x32_averaged_and_threshold(img_data, threshold=128)
                debug_data(img_32x32, "Resized 32x32 BMP")

                # ✂️ Remove BMP Header
                img_32x32_raw = strip_bmp_header(img_32x32)
                debug_data(img_32x32_raw, "Stripped BMP Header")

                # ✅ Convert bytearray to array.array('B', ..)
                img_32x32_raw_array = array.array('B', img_32x32_raw)
                debug_data(img_32x32_raw_array, "Converted to array.array('B', ..)")

                # 🔮 Run Model Inference
                model.run(img_32x32_raw_array, probabilities)
                predicted_class = argmax(probabilities)
                
                # 📡 Send frame + prediction to client
                #client.sendall(b"FRAME_START" + img_32x32 + b"FRAME_END")
                client_socket.sendall(f"PREDICTION: {CLASS_NAMES[predicted_class]}\n".encode())

                print(f"🖼️ Classified as: {CLASS_NAMES[predicted_class]}")
                
                sleep(3)  # Delay between frames (adjust as needed)
        except Exception as e:
            print(f"Transmission error: {e}")
            client_socket.close()
            break