import cv2
import numpy as np
import mediapipe as mp
import pyvirtualcam
from pyvirtualcam import PixelFormat

# ------------------ Config ------------------
CAM_INDEX = 0
TARGET_IMG_PATH = "target.jpg"
FRAME_W, FRAME_H = 640, 480   # lower resolution = smoother FPS
FPS = 30
USE_SEAMLESS_CLONE = False     # <-- toggle (True = better quality, slower; False = faster)

LANDMARK_IDX = {
    'face_hull': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
                  378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                  162, 21, 54, 103, 67, 109, 10],
    'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    'right_eye':[263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
    'mouth':    [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13,
                 82, 81, 80, 191],
    'nose':     [6, 197, 195, 5, 4, 45, 275, 440, 279, 49, 209, 238, 250, 462]
}

# ------------------ Utils ------------------
def mediapipe_to_ndarray(landmarks, w, h):
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

def get_subset(points, idx_list):
    return np.array([points[i] for i in idx_list], dtype=np.float32)

def subdiv_delaunay(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))
    triangle_list = subdiv.getTriangleList()
    triangles = []
    r_x, r_y, r_w, r_h = rect
    for t in triangle_list:
        tri = np.array([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]], dtype=np.float32)
        if np.all((tri[:, 0] >= r_x) & (tri[:, 0] < r_x + r_w) &
                  (tri[:, 1] >= r_y) & (tri[:, 1] < r_y + r_h)):
            idx = []
            for i in range(3):
                d = np.linalg.norm(points - tri[i], axis=1)
                idx.append(int(np.argmin(d)))
            triangles.append(tuple(idx))
    return triangles

def warp_triangle(src_img, dst_accum_img, src_tri, dst_tri, acc_mask):
    r1 = cv2.boundingRect(np.float32([src_tri]))
    r2 = cv2.boundingRect(np.float32([dst_tri]))
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return
    t1_rect = np.float32([[src_tri[i][0] - r1[0], src_tri[i][1] - r1[1]] for i in range(3)])
    t2_rect = np.float32([[dst_tri[i][0] - r2[0], dst_tri[i][1] - r2[1]] for i in range(3)])
    x1, y1, w1, h1 = r1
    src_cropped = src_img[y1:y1+h1, x1:x1+w1]
    if src_cropped.size == 0:
        return
    x2, y2, w2, h2 = r2
    M = cv2.getAffineTransform(t1_rect, t2_rect)
    warped = cv2.warpAffine(src_cropped, M, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    tri_mask = np.zeros((h2, w2), dtype=np.uint8)
    cv2.fillConvexPoly(tri_mask, np.int32(t2_rect), 255)
    roi = dst_accum_img[y2:y2+h2, x2:x2+w2]
    if roi.shape[:2] != tri_mask.shape:
        return
    warped_masked = cv2.bitwise_and(warped, warped, mask=tri_mask)
    roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(tri_mask))
    combined = cv2.add(roi_bg, warped_masked)
    dst_accum_img[y2:y2+h2, x2:x2+w2] = combined
    acc_mask[y2:y2+h2, x2:x2+w2] = cv2.bitwise_or(
        acc_mask[y2:y2+h2, x2:x2+w2],
        tri_mask
    )

def color_correct(src_face, dst_face, mask):
    src = src_face.astype(np.float32)
    dst = dst_face.astype(np.float32)
    if mask is None or mask.size == 0 or np.count_nonzero(mask) == 0:
        return dst_face.astype(np.uint8)
    for c in range(3):
        src_c = src[..., c][mask > 0]
        dst_c = dst[..., c][mask > 0]
        if src_c.size == 0 or dst_c.size == 0:
            continue
        src_mean, src_std = src_c.mean(), src_c.std() + 1e-6
        dst_mean, dst_std = dst_c.mean(), dst_c.std() + 1e-6
        dst[..., c] = (dst[..., c] - dst_mean) * (src_std / dst_std) + src_mean
    return np.clip(dst, 0, 255).astype(np.uint8)

def align_face(image, landmarks):
    left_eye = np.mean(landmarks[LANDMARK_IDX['left_eye']], axis=0)
    right_eye = np.mean(landmarks[LANDMARK_IDX['right_eye']], axis=0)
    dy = float(right_eye[1] - left_eye[1])
    dx = float(right_eye[0] - left_eye[0])
    angle = np.degrees(np.arctan2(dy, dx))
    center = (float((left_eye[0] + right_eye[0]) * 0.5),
              float((left_eye[1] + right_eye[1]) * 0.5))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([landmarks.astype(np.float32), ones])
    aligned_pts = (M @ pts_h.T).T
    return rotated, aligned_pts

# ------------------ Load & preprocess target face ------------------
target_bgr = cv2.imread(TARGET_IMG_PATH)
if target_bgr is None:
    raise FileNotFoundError("Could not read target.jpg")

mp_face_mesh = mp.solutions.face_mesh
face_mesh_static = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True)
face_mesh_live   = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

t_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
res = face_mesh_static.process(t_rgb)
if not res.multi_face_landmarks:
    raise RuntimeError("No face detected in target.jpg")

h0, w0, _ = target_bgr.shape
largest_face, largest_area = None, 0
for lm_set in res.multi_face_landmarks:
    pts = mediapipe_to_ndarray(lm_set.landmark, w0, h0)
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    area = w * h
    if area > largest_area:
        largest_area, largest_face = area, lm_set

t_landmarks = mediapipe_to_ndarray(largest_face.landmark, w0, h0)
target_bgr, t_landmarks = align_face(target_bgr, t_landmarks)
x, y, w, h = cv2.boundingRect(t_landmarks.astype(np.int32))
margin = int(0.2 * max(w, h))
x1, y1 = max(0, x - margin), max(0, y - margin)
x2, y2 = min(w0, x + w + margin), min(h0, y + h + margin)
cropped = target_bgr[y1:y2, x1:x2]
if cropped.size == 0:
    cropped = target_bgr.copy()
landmarks_cropped = t_landmarks - np.array([x1, y1], dtype=np.float32)
ch, cw = cropped.shape[:2]
target_bgr = cv2.resize(cropped, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
scale_x, scale_y = FRAME_W / float(cw), FRAME_H / float(ch)
t_landmarks = landmarks_cropped.copy().astype(np.float32)
t_landmarks[:, 0] *= scale_x
t_landmarks[:, 1] *= scale_y

# Expand landmark set: add jawline + outer face points
used_idx = sorted(set(
    LANDMARK_IDX['face_hull'] +
    LANDMARK_IDX['left_eye'] +
    LANDMARK_IDX['right_eye'] +
    LANDMARK_IDX['mouth'] +
    LANDMARK_IDX['nose'] +
    list(range(0, 17)) +   # jawline
    [127, 234, 454, 356]   # outer corners
))
t_points = get_subset(t_landmarks, used_idx)
rect = (0, 0, FRAME_W, FRAME_H)
tri_indices = subdiv_delaunay(rect, t_points)

# ------------------ Live loop ------------------
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, FPS)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

with pyvirtualcam.Camera(width=FRAME_W, height=FRAME_H, fps=FPS, fmt=PixelFormat.BGR) as cam:
    print(f"Virtual camera started: {cam.device}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = face_mesh_live.process(rgb)
        if out.multi_face_landmarks:
            f_landmarks = mediapipe_to_ndarray(out.multi_face_landmarks[0].landmark, FRAME_W, FRAME_H)
            f_points = get_subset(f_landmarks, used_idx)
            warped_face = np.zeros_like(frame)
            acc_mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            for (i1, i2, i3) in tri_indices:
                src_tri = np.float32([t_points[i1], t_points[i2], t_points[i3]])
                dst_tri = np.float32([f_points[i1], f_points[i2], f_points[i3]])
                warp_triangle(target_bgr, warped_face, src_tri, dst_tri, acc_mask)
            if np.count_nonzero(acc_mask) > 0:
                acc_mask = cv2.dilate(acc_mask, np.ones((15, 15), np.uint8), iterations=1)
                warped_face_cc = color_correct(frame, warped_face.copy(), acc_mask)

                if USE_SEAMLESS_CLONE:
                    nz = cv2.findNonZero(acc_mask)
                    if nz is not None:
                        xx, yy, ww, hh = cv2.boundingRect(nz)
                        center = (int(xx + ww / 2), int(yy + hh / 2))
                        try:
                            output = cv2.seamlessClone(warped_face_cc, frame, acc_mask, center, cv2.NORMAL_CLONE)
                        except cv2.error:
                            output = frame
                    else:
                        output = frame
                else:
                    # Fast alpha blending
                    alpha = (acc_mask.astype(np.float32) / 255.0)[..., None]
                    output = (alpha * warped_face_cc + (1 - alpha) * frame).astype(np.uint8)
            else:
                output = frame
        else:
            output = frame

        cv2.putText(output, "SIMULATED VIDEO", (20, FRAME_H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Preview", output)
        cam.send(output)
        cam.sleep_until_next_frame()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
