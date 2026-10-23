import os

nvidia_path = r"C:\Users\Admin\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\nvidia"
if os.path.exists(nvidia_path):
    for root, dirs, files in os.walk(nvidia_path):
        if any(f.lower().startswith(("cublas", "cudart", "cudnn")) for f in files):
            try:
                os.add_dll_directory(root)
            except Exception:
                pass

import cv2
import numpy as np
import threading
import time
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import pyvirtualcam
import onnxruntime as ort

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'inswapper_128.onnx')
SOURCE_IMG = 'my_face.jpg'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Нет файла {MODEL_PATH} — скачай inswapper_128.onnx")

print("Доступные провайдеры:", ort.get_available_providers())

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

swapper = get_model(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

for model in app.models.values():
    try:
        model.session.set_providers(['CUDAExecutionProvider'])
    except Exception:
        pass
try:
    swapper.session.set_providers(['CUDAExecutionProvider'])
except Exception:
    pass

print("Detection provider:", app.models['detection'].session.get_providers())
print("Swapper provider:", swapper.session.get_providers())

src = cv2.imread(SOURCE_IMG)
if src is None:
    raise FileNotFoundError(f"Не найдено фото {SOURCE_IMG}")

faces = app.get(src)
if not faces:
    raise RuntimeError("На фото не найдено лицо")

source_face = faces[0]


class CameraReader:
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
            else:
                time.sleep(0.01)

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

W, H, FPS = 640, 480, 30

reader = CameraReader(cap)

DETECT_EVERY_N = 3
frame_count = 0
last_faces = []

with pyvirtualcam.Camera(width=W, height=H, fps=FPS) as cam:
    print(f'Используется виртуальная камера: {cam.device}')
    print('В Discord/OBS выберите "OBS Virtual Camera" как камеру')
    print('Нажми q в окне предпросмотра для выхода')

    while True:
        frame = reader.get()
        if frame is None:
            time.sleep(0.005)
            continue

        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H))

        frame_count += 1

        if frame_count % DETECT_EVERY_N == 0 or not last_faces:
            last_faces = app.get(frame)

        for face in last_faces:
            try:
                frame = swapper.get(frame, face, source_face, paste_back=True)
            except Exception:
                pass

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam.send(frame_rgb)
        cam.sleep_until_next_frame()

        cv2.imshow('FaceSwap (нажми q для выхода)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

reader.stop()
cap.release()
cv2.destroyAllWindows()
