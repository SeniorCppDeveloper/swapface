import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'inswapper_128.onnx')
SOURCE_IMG = 'dadajpg.jpg'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"нет файла {MODEL_PATH} — скачай inswapper_128.onnx")

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(320, 320))

swapper = get_model(MODEL_PATH, providers=['CPUExecutionProvider'])

src = cv2.imread(SOURCE_IMG)
if src is None:
    raise FileNotFoundError(f"нет фото {SOURCE_IMG}")

faces = app.get(src)
if not faces:
    raise RuntimeError("нет")

source_face = faces[0]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    for face in app.get(frame):
        frame = swapper.get(frame, face, source_face, paste_back=True)
    cv2.imshow('FaceSwap', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
