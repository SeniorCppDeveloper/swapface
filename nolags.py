import os

SP = r"C:\Users\Admin\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\nvidia"
for p in [
    os.path.join(SP, "cu13", "bin", "x86_64"),
    os.path.join(SP, "cudnn", "bin"),
]:
    os.add_dll_directory(p)
    os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]

import cv2
import numpy as np
import onnxruntime as ort
import pyvirtualcam
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'inswapper_128.onnx')
SOURCE_IMG = 'dadajpg.jpg'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"нет файла {MODEL_PATH} — скачай inswapper_128.onnx")

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
print('Доступные провайдеры:', ort.get_available_providers())

full_app = FaceAnalysis(name='buffalo_l', providers=providers)
full_app.prepare(ctx_id=0, det_size=(640, 640))

det_app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'], providers=providers)
det_app.prepare(ctx_id=0, det_size=(320, 320))

swapper = get_model(MODEL_PATH, providers=providers)
print('Провайдеры swapper:', swapper.session.get_providers())

src = cv2.imread(SOURCE_IMG)
if src is None:
    raise FileNotFoundError(f"нет фото {SOURCE_IMG}")

faces = full_app.get(src)
if not faces:
    raise RuntimeError("нет лиц на исходном фото")
source_face = faces[0]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

DETECT_EVERY = 3
last_faces = []
frame_idx = 0

with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
    print(f'Используется виртуальная камера: {cam.device}')

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % DETECT_EVERY == 0:
            last_faces = det_app.get(frame)
        frame_idx += 1

        if last_faces:
            face = max(last_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            frame = swapper.get(frame, face, source_face, paste_back=True)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam.send(frame_rgb)
        cam.sleep_until_next_frame()

        cv2.imshow('FaceSwap (Press Q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
