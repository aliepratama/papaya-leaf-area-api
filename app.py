# import the inference-sdk
import os
import json
import cv2
import numpy as np

from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

REAL_COIN_AREA = 6.16 # cm

load_dotenv()
# initialize the client

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.environ['API_KEY']
)

# # infer on a local image
result = CLIENT.infer("./IMG_0001.jpg", model_id="area-of-papaya-leaves/5")
width = result['image']['width']
height = result['image']['height']
leaf_area = 0
coin_area = 0
for predict in result['predictions']:
    name = predict['class']
    contour = np.array([[point['x'], point['y']] for point in predict['points']], dtype=np.float32)
    area = cv2.contourArea(contour)
    print(f"{name}: {area} sq. pixels with confidence {predict['confidence']}")
    if name == 'leaf':
        leaf_area += area
    elif name == 'hole':
        leaf_area += area
    else:
        coin_area += area

print(f"Calculated leaf area {leaf_area / coin_area * REAL_COIN_AREA:.2f} sq. centimeters")

# print(result)
