import os
from io import BytesIO
import cv2
import dlib
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI()

#dlibs Facial Landmark Model
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
TARGET_SIZE = 512  

if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(
        f"download {PREDICTOR_PATH} here:"
        "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    )

#initialization
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

#converts all images to unit8 for openCV
def to_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img // 256).astype(np.uint8)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img_norm.astype(np.uint8)

#rgb
def ensure_three_channels(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

#resize image with aspect ratio then pad to square
def resize_safe(img: np.ndarray, target: int = TARGET_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) == 0:
        return img

    scale = target / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (target - new_h) // 2
    bottom = target - new_h - top
    left = (target - new_w) // 2
    right = target - new_w - left

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

def load_image_from_bytes(raw_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    img = to_uint8(img)
    img = ensure_three_channels(img)
    return img

# extract face landmarks
def get_face_and_landmarks_from_array(img: np.ndarray):
    if img is None:
        print("Input image failed")
        return None, None, None

    img_work = resize_safe(img, TARGET_SIZE)
    img_work = to_uint8(img_work)
    img_work = ensure_three_channels(img_work)

    #double check
    gray = cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        print("Converting grayscale to uint8")
        gray = (gray // 256).astype(np.uint8)

    print("image shape:", img_work.shape, "gray min/max:", gray.min(), gray.max())

    #face detector
    rects = detector(gray, 1)
    print("faces found:", len(rects))

    if len(rects) == 0:
        return None, None, None

    rect = rects[0]
    shape = predictor(gray, rect)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)

    return img_work, rect, landmarks

#color correction
def color_correction(src_face: np.ndarray, dst_img: np.ndarray, dst_mask: np.ndarray) -> np.ndarray:
    src = src_face.astype(np.float32)
    dst = dst_img.astype(np.float32)
    mask = dst_mask.astype(bool)

    if mask.sum() == 0:
        return src_face

    src_means, src_stds = [], []
    dst_means, dst_stds = [], []

    for c in range(3):
        svals = src[:, :, c][mask]
        dvals = dst[:, :, c][mask]
        src_means.append(svals.mean() if svals.size else 0.0)
        src_stds.append(svals.std() if svals.size else 1.0)
        dst_means.append(dvals.mean() if dvals.size else 0.0)
        dst_stds.append(dvals.std() if dvals.size else 1.0)

    src_means = np.array(src_means)
    src_stds = np.array(src_stds)
    dst_means = np.array(dst_means)
    dst_stds = np.array(dst_stds)
    src_stds[src_stds == 0] = 1.0

    corrected = (
        (src - src_means.reshape(1, 1, 3))
        * (dst_stds.reshape(1, 1, 3) / src_stds.reshape(1, 1, 3))
        + dst_means.reshape(1, 1, 3)
    )

    return np.clip(corrected, 0, 255).astype(np.uint8)

#main pipeline: detect, align, color correction, blend
def swap_faces(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    print("swap faces started")

    #detect
    img1_work, rect1, lm1 = get_face_and_landmarks_from_array(img1)
    img2_work, rect2, lm2 = get_face_and_landmarks_from_array(img2)

    if img1_work is None or img2_work is None or lm1 is None or lm2 is None:
        print("Landmarks/detection failed")
        return None

    #masks
    src_hull = cv2.convexHull(lm1)
    dst_hull = cv2.convexHull(lm2)

    src_mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    dst_mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    cv2.fillConvexPoly(src_mask, src_hull, 255)
    cv2.fillConvexPoly(dst_mask, dst_hull, 255)

    kernel = np.ones((9, 9), np.uint8)
    dst_mask_eroded = cv2.erode(dst_mask, kernel, iterations=1)
    if dst_mask_eroded.sum() == 0:
        dst_mask_eroded = dst_mask.copy()

    src_face_region = cv2.bitwise_and(img1_work, img1_work, mask=src_mask)

    #align
    key_idx = [36, 39, 42, 45, 30, 48, 54]
    M, _ = cv2.estimateAffinePartial2D(
        np.float32(lm1[key_idx]),
        np.float32(lm2[key_idx]),
        method=cv2.LMEDS,
    )

    if M is None:
        print("Affine transform failure")
        return None

    #source face -> target
    warped_src_face = cv2.warpAffine(
        src_face_region, M, (TARGET_SIZE, TARGET_SIZE),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    #source mask -> target
    warped_src_mask = cv2.warpAffine(
        src_mask, M, (TARGET_SIZE, TARGET_SIZE),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    _, warped_src_mask = cv2.threshold(warped_src_mask, 10, 255, cv2.THRESH_BINARY)

    if warped_src_mask.sum() == 0:
        print("warped_src_mask empty")
        return None

    #color correction
    warped_src_face_corrected = color_correction(warped_src_face, img2_work, warped_src_mask)

    clone_mask = cv2.bitwise_and(warped_src_mask, dst_mask_eroded)
    _, clone_mask = cv2.threshold(clone_mask, 10, 255, cv2.THRESH_BINARY)

    if clone_mask.sum() == 0:
        print("clone_mask empty")
        return None

    #blend
    center = (int(np.mean(lm2[:, 0])), int(np.mean(lm2[:, 1])))

    try:
        output = cv2.seamlessClone(
            warped_src_face_corrected,
            img2_work.astype(np.uint8),
            clone_mask.astype(np.uint8),
            center,
            cv2.NORMAL_CLONE,
        )
    except Exception as e:
        print("seamlessClone:", e)
        return None

    print("✅")
    return output

#FastAPI stuff
@app.post("/swap")
async def swap_endpoint(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    print(f"Received: {image1.filename}, {image2.filename}")

    b1 = await image1.read()
    b2 = await image2.read()

    img1 = load_image_from_bytes(b1)
    img2 = load_image_from_bytes(b2)

    if img1 is None or img2 is None:
        return {"error": "Could not decode one or both images"}

    swapped = swap_faces(img1, img2)
    if swapped is None:
        return {"error": "Face swap failed"}

    success, buf = cv2.imencode(".jpg", swapped)
    if not success:
        return {"error": "Encoding failed"}

    return StreamingResponse(BytesIO(buf.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
