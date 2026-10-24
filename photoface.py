import cv2
import numpy as np
import pyvirtualcam
import tkinter as tk
from tkinter import filedialog
from insightface.app import FaceAnalysis
from scipy.spatial import Delaunay

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "landmark_2d_106"],
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))

OFF = np.float32


def choose_photo():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Выбери фото лица",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")],
    )
    root.destroy()
    if not path:
        return None
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Не удалось открыть файл")
        return None
    app.det_model.prepare(0, input_size=(640, 640))
    faces = app.get(img)
    if not faces:
        print("На фото лицо не найдено")
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    pts = face.landmark_2d_106.astype(np.float32)
    h, w = img.shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    tri = Delaunay(pts).simplices
    thumb = img.copy()
    for p in pts:
        cv2.circle(thumb, (int(p[0]), int(p[1])), max(2, w // 200), (0, 255, 0), -1)
    scale = 200.0 / max(h, w)
    thumb = cv2.resize(thumb, (int(w * scale), int(h * scale)))
    return img, pts, tri, thumb


def warp_face(photo, src, dst, tri, base):
    out = base.copy()
    H, W = base.shape[:2]
    ph, pw = photo.shape[:2]
    for t in tri:
        s = src[t].astype(np.float32)
        d = dst[t].astype(np.float32)
        sx, sy, sw, sh = cv2.boundingRect(s)
        dx, dy, dw, dh = cv2.boundingRect(d)
        if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
            continue
        if dw > W * 2 or dh > H * 2:
            continue
        x0, y0 = max(dx, 0), max(dy, 0)
        x1, y1 = min(dx + dw, W), min(dy + dh, H)
        if x1 <= x0 or y1 <= y0:
            continue
        sx2, sy2 = min(sx + sw, pw), min(sy + sh, ph)
        patch = photo[sy:sy2, sx:sx2]
        if patch.size == 0:
            continue
        s_loc = (s - OFF([sx, sy])).astype(np.float32)
        d_loc = (d - OFF([dx, dy])).astype(np.float32)
        m_aff = cv2.getAffineTransform(s_loc, d_loc)
        wp = cv2.warpAffine(
            patch, m_aff, (dw, dh),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = np.zeros((dh, dw), np.uint8)
        cv2.fillConvexPoly(
            mask, np.round(d_loc * 16).astype(np.int32), 255,
            lineType=cv2.LINE_8, shift=4,
        )
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        ox0, oy0 = x0 - dx, y0 - dy
        ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)
        sub_m = mask[oy0:oy1, ox0:ox1] > 0
        roi = out[y0:y1, x0:x1]
        roi[sub_m] = wp[oy0:oy1, ox0:ox1][sub_m]
    return out


def color_match(warped, frame, mask, rect):
    x, y, w, h = rect
    if w < 10 or h < 10:
        return warped
    sel = mask[y:y + h, x:x + w] > 128
    if sel.sum() < 200:
        return warped
    a = cv2.cvtColor(warped[y:y + h, x:x + w], cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2LAB).astype(np.float32)
    matched = a.copy()
    for c in range(3):
        ma, sa = a[..., c][sel].mean(), a[..., c][sel].std() + 1e-6
        mb, sb = b[..., c][sel].mean(), b[..., c][sel].std() + 1e-6
        matched[..., c] = (a[..., c] - ma) / sa * sb + mb
    a = np.clip(a * 0.3 + matched * 0.7, 0, 255).astype(np.uint8)
    res = warped.copy()
    res[y:y + h, x:x + w] = cv2.cvtColor(a, cv2.COLOR_LAB2BGR)
    return res


def nothing(_):
    pass


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ok, frame = cap.read()
if not ok:
    raise SystemExit("Камера не открылась")
FH, FW = frame.shape[:2]

cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("controls", 420, 300)
cv2.createTrackbar("Scale X", "controls", 100, 200, nothing)
cv2.createTrackbar("Scale Y", "controls", 100, 200, nothing)
cv2.createTrackbar("Shift X", "controls", 100, 200, nothing)
cv2.createTrackbar("Shift Y", "controls", 100, 200, nothing)
cv2.createTrackbar("Feather", "controls", 15, 60, nothing)
cv2.createTrackbar("Opacity", "controls", 100, 100, nothing)

data = choose_photo()
use_color = True
show_mesh = False
smooth = None
app.det_model.prepare(0, input_size=(320, 320))

print("o - сменить фото | c - подгонка цвета | l - сетка точек | q или Esc - выход")

with pyvirtualcam.Camera(width=FW, height=FH, fps=30, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = frame.copy()
        if data is not None:
            photo, src, tri, thumb = data
            faces = app.get(frame)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                cur = face.landmark_2d_106.astype(np.float32)
                smooth = cur if smooth is None else (smooth * 0.4 + cur * 0.6).astype(np.float32)

                sx = cv2.getTrackbarPos("Scale X", "controls") / 100.0
                sy = cv2.getTrackbarPos("Scale Y", "controls") / 100.0
                shx = cv2.getTrackbarPos("Shift X", "controls") - 100
                shy = cv2.getTrackbarPos("Shift Y", "controls") - 100
                feather = max(1, cv2.getTrackbarPos("Feather", "controls"))
                opacity = cv2.getTrackbarPos("Opacity", "controls") / 100.0

                center = smooth.mean(axis=0)
                dst = ((smooth - center) * OFF([sx, sy]) + center + OFF([shx, shy])).astype(np.float32)

                warped = warp_face(photo, src, dst, tri, frame)

                hull = cv2.convexHull(np.round(dst).astype(np.int32))
                mask = np.zeros((FH, FW), np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)
                k = max(3, feather | 1)
                mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

                if use_color:
                    bx, by, bw, bh = cv2.boundingRect(hull)
                    rx, ry = max(bx, 0), max(by, 0)
                    rw, rh = min(bx + bw, FW) - rx, min(by + bh, FH) - ry
                    warped = color_match(warped, frame, mask, (rx, ry, rw, rh))

                alpha = cv2.GaussianBlur(mask, (k * 2 + 1, k * 2 + 1), 0).astype(np.float32) / 255.0
                alpha = (alpha * opacity)[..., None]
                result = (frame.astype(np.float32) * (1 - alpha) + warped.astype(np.float32) * alpha).astype(np.uint8)
            else:
                smooth = None

        cam.send(result)

        view = result.copy()
        if show_mesh and smooth is not None:
            for p in smooth:
                cv2.circle(view, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
        if data is not None:
            th, tw = thumb.shape[:2]
            view[10:10 + th, 10:10 + tw] = thumb
            cv2.rectangle(view, (10, 10), (10 + tw, 10 + th), (255, 255, 255), 1)
        else:
            cv2.putText(view, "Press O to choose photo", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("preview", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("c"):
            use_color = not use_color
        if key == ord("l"):
            show_mesh = not show_mesh
        if key == ord("o"):
            new = choose_photo()
            app.det_model.prepare(0, input_size=(320, 320))
            if new is not None:
                data = new
                smooth = None

cap.release()
cv2.destroyAllWindows()
