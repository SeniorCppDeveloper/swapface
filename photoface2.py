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

F32 = np.float32

S = {
    "mode": "live",
    "photo": None,
    "src": None,
    "tri": None,
    "D0": None,
    "A": None,
    "frozen": None,
    "sel": -1,
    "all": False,
    "start": None,
    "A0": None,
    "w": None,
    "mx": 0,
    "my": 0,
    "color": True,
    "points": False,
    "mesh": False,
}


def dialog(kind, **kw):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if kind == "open":
        p = filedialog.askopenfilename(**kw)
    else:
        p = filedialog.asksaveasfilename(**kw)
    root.destroy()
    return p


def build_photo(img):
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
    return pts, Delaunay(pts).simplices


def choose_photo():
    path = dialog(
        "open",
        title="Выбери фото лица",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")],
    )
    if not path:
        return False
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Не удалось открыть файл")
        return False
    res = build_photo(img)
    app.det_model.prepare(0, input_size=(320, 320))
    if res is None:
        return False
    S["photo"], S["src"], S["tri"] = img, res[0], res[1]
    S["D0"], S["A"] = None, None
    return True


def save_profile():
    if S["photo"] is None:
        return
    path = dialog(
        "save",
        title="Сохранить профиль",
        defaultextension=".npz",
        filetypes=[("Profile", "*.npz")],
    )
    if not path:
        return
    has = S["A"] is not None
    np.savez_compressed(
        path,
        photo=S["photo"],
        src=S["src"],
        d0=S["D0"] if has else np.zeros((106, 2), np.float32),
        a=S["A"] if has else np.zeros((106, 2), np.float32),
        has=np.array(has),
        feather=np.array(cv2.getTrackbarPos("Feather", "controls")),
        opacity=np.array(cv2.getTrackbarPos("Opacity", "controls")),
        color=np.array(S["color"]),
    )
    print("Сохранено:", path)


def load_profile():
    path = dialog(
        "open",
        title="Загрузить профиль",
        filetypes=[("Profile", "*.npz")],
    )
    if not path:
        return
    d = np.load(path)
    S["photo"] = d["photo"]
    S["src"] = d["src"].astype(np.float32)
    S["tri"] = Delaunay(S["src"]).simplices
    if bool(d["has"]):
        S["D0"] = d["d0"].astype(np.float32)
        S["A"] = d["a"].astype(np.float32)
    else:
        S["D0"], S["A"] = None, None
    cv2.setTrackbarPos("Feather", "controls", int(d["feather"]))
    cv2.setTrackbarPos("Opacity", "controls", int(d["opacity"]))
    S["color"] = bool(d["color"])
    S["mode"] = "live"
    print("Загружено:", path)


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
        patch = photo[sy:min(sy + sh, ph), sx:min(sx + sw, pw)]
        if patch.size == 0:
            continue
        s_loc = (s - F32([sx, sy])).astype(np.float32)
        d_loc = (d - F32([dx, dy])).astype(np.float32)
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


def render(base, dst, feather, opacity):
    H, W = base.shape[:2]
    warped = warp_face(S["photo"], S["src"], dst, S["tri"], base)
    hull = cv2.convexHull(np.round(dst).astype(np.int32))
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    k = max(3, feather | 1)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if S["color"]:
        bx, by, bw, bh = cv2.boundingRect(hull)
        rx, ry = max(bx, 0), max(by, 0)
        rw, rh = min(bx + bw, W) - rx, min(by + bh, H) - ry
        warped = color_match(warped, base, mask, (rx, ry, rw, rh))
    alpha = cv2.GaussianBlur(mask, (k * 2 + 1, k * 2 + 1), 0).astype(np.float32) / 255.0
    alpha = (alpha * opacity)[..., None]
    return (base.astype(np.float32) * (1 - alpha) + warped.astype(np.float32) * alpha).astype(np.uint8)


def live_dst(D):
    if S["A"] is None:
        return D
    M, _ = cv2.estimateAffinePartial2D(S["D0"], D)
    if M is None:
        return D
    L = M[:, :2].astype(np.float32)
    return (D + (S["A"] - S["D0"]) @ L.T).astype(np.float32)


def get_face(frame):
    faces = app.get(frame)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.landmark_2d_106.astype(np.float32)


def stretch(A, sx, sy):
    c = A.mean(axis=0)
    return ((A - c) * F32([sx, sy]) + c).astype(np.float32)


def rotate(A, deg):
    c = A.mean(axis=0)
    r = np.deg2rad(deg)
    R = F32([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return ((A - c) @ R.T + c).astype(np.float32)


def on_mouse(event, x, y, flags, param):
    S["mx"], S["my"] = x, y
    if S["mode"] != "edit" or S["A"] is None:
        return
    p = F32([x, y])
    if event == cv2.EVENT_LBUTTONDOWN:
        d = np.linalg.norm(S["A"] - p, axis=1)
        i = int(d.argmin())
        if d[i] < 20:
            r = cv2.getTrackbarPos("Radius", "controls")
            S["sel"] = i
            S["start"] = p
            S["A0"] = S["A"].copy()
            if r <= 0:
                w = np.zeros(len(S["A"]), np.float32)
                w[i] = 1.0
            else:
                dist = np.linalg.norm(S["A0"] - S["A0"][i], axis=1)
                w = np.exp(-(dist / r) ** 2).astype(np.float32)
            S["w"] = w
    elif event == cv2.EVENT_RBUTTONDOWN:
        S["all"] = True
        S["start"] = p
        S["A0"] = S["A"].copy()
    elif event == cv2.EVENT_MOUSEMOVE:
        if S["sel"] >= 0:
            S["A"] = (S["A0"] + S["w"][:, None] * (p - S["start"])).astype(np.float32)
        elif S["all"]:
            S["A"] = (S["A0"] + (p - S["start"])).astype(np.float32)
    elif event == cv2.EVENT_LBUTTONUP:
        S["sel"] = -1
    elif event == cv2.EVENT_RBUTTONUP:
        S["all"] = False


def put(img, text, y):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


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
cv2.resizeWindow("controls", 420, 200)
cv2.createTrackbar("Radius", "controls", 40, 200, nothing)
cv2.createTrackbar("Feather", "controls", 15, 60, nothing)
cv2.createTrackbar("Opacity", "controls", 100, 100, nothing)

cv2.namedWindow("editor", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("editor", on_mouse)

choose_photo()
app.det_model.prepare(0, input_size=(320, 320))
smooth = None
msg = ""

with pyvirtualcam.Camera(width=FW, height=FH, fps=30, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        feather = max(1, cv2.getTrackbarPos("Feather", "controls"))
        opacity = cv2.getTrackbarPos("Opacity", "controls") / 100.0
        result = frame
        shown = None

        if S["mode"] == "live":
            if S["photo"] is not None:
                D = get_face(frame)
                if D is not None:
                    smooth = D if smooth is None else (smooth * 0.4 + D * 0.6).astype(np.float32)
                    shown = live_dst(smooth)
                    result = render(frame, shown, feather, opacity)
                else:
                    smooth = None
            view = result.copy()
            if shown is not None:
                if S["mesh"]:
                    for t in S["tri"]:
                        cv2.polylines(view, [np.round(shown[t]).astype(np.int32)], True, (0, 200, 0), 1)
                if S["points"]:
                    for p in shown:
                        cv2.circle(view, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
            put(view, "LIVE | SPACE edit | O photo | S save | L load | C color | P points | M mesh | Q quit", 24)
        else:
            base = S["frozen"]
            result = render(base, S["A"], feather, opacity)
            view = result.copy()
            if S["mesh"]:
                for t in S["tri"]:
                    cv2.polylines(view, [np.round(S["A"][t]).astype(np.int32)], True, (0, 200, 0), 1)
            for i, p in enumerate(S["A"]):
                col = (0, 0, 255) if i == S["sel"] else (0, 255, 0)
                cv2.circle(view, (int(p[0]), int(p[1])), 4, col, -1)
                cv2.circle(view, (int(p[0]), int(p[1])), 4, (0, 0, 0), 1)
            r = cv2.getTrackbarPos("Radius", "controls")
            if r > 0:
                cv2.circle(view, (S["mx"], S["my"]), r, (255, 255, 0), 1)
            put(view, "EDIT | LMB drag point | RMB drag all | +/- size | [ ] width | ; ' height | , . rotate", 24)
            put(view, "R reset | M mesh | C color | SPACE back to live", 46)

        if msg:
            put(view, msg, FH - 16)
        if S["photo"] is None:
            put(view, "Press O to choose photo", 70)

        cam.send(result)
        cv2.imshow("editor", view)

        key = cv2.waitKey(1) & 0xFF
        msg = ""
        if key in (27, ord("q")):
            break
        elif key == ord("o"):
            if choose_photo():
                S["mode"] = "live"
                smooth = None
            app.det_model.prepare(0, input_size=(320, 320))
        elif key == ord("s"):
            save_profile()
        elif key == ord("l"):
            load_profile()
            smooth = None
        elif key == ord("c"):
            S["color"] = not S["color"]
        elif key == ord("p"):
            S["points"] = not S["points"]
        elif key == ord("m"):
            S["mesh"] = not S["mesh"]
        elif key == 32:
            if S["photo"] is None:
                msg = "Choose a photo first (O)"
            elif S["mode"] == "live":
                D = get_face(frame)
                if D is None:
                    msg = "Face not found"
                else:
                    S["A"] = live_dst(D).copy()
                    S["D0"] = D.copy()
                    S["frozen"] = frame.copy()
                    S["sel"] = -1
                    S["all"] = False
                    S["mode"] = "edit"
            else:
                S["sel"] = -1
                S["all"] = False
                S["mode"] = "live"
                smooth = None
        elif S["mode"] == "edit" and S["A"] is not None:
            if key == ord("r"):
                S["A"] = S["D0"].copy()
            elif key in (ord("+"), ord("=")):
                S["A"] = stretch(S["A"], 1.03, 1.03)
            elif key == ord("-"):
                S["A"] = stretch(S["A"], 1 / 1.03, 1 / 1.03)
            elif key == ord("]"):
                S["A"] = stretch(S["A"], 1.03, 1.0)
            elif key == ord("["):
                S["A"] = stretch(S["A"], 1 / 1.03, 1.0)
            elif key == ord("'"):
                S["A"] = stretch(S["A"], 1.0, 1.03)
            elif key == ord(";"):
                S["A"] = stretch(S["A"], 1.0, 1 / 1.03)
            elif key == ord("."):
                S["A"] = rotate(S["A"], 1.0)
            elif key == ord(","):
                S["A"] = rotate(S["A"], -1.0)

cap.release()
cv2.destroyAllWindows()
