import streamlit as st
import shutil
import cv2
import csv
import mediapipe as mp
import numpy as np
import os
from os import path

st.title("Add Face Mask (WebCam)")
try:
    os.mkdir("temp")
except:
    pass
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

detectionConfidence = st.slider("Face Detection Confidence")
trackingConfidence = st.slider("Face Tracking Confidence")

minDetectionCon = float(detectionConfidence / 100)
minTrackingCon = float(trackingConfidence / 100)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=minDetectionCon,
    min_tracking_confidence=minTrackingCon,
)


def face_point(image, minDetectionCon, minTrackingCon):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    faces = []
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            face = []
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = image.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                face.append([id, x, y])
        faces.append(face)
    return faces


def normalize8(I):
    mn = I.min()
    mx = I.max()
    mx -= mn
    I = ((I - mn) / mx) * 255
    return I.astype(np.uint8)


def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype)
                * 255,
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y : y + h, x : x + w] = (1.0 - mask) * background[
        y : y + h, x : x + w
    ] + mask * overlay_image

    return background


def mask_overlay(image, faces, mask_up, mask_down, mask_img, mask_points):
    mirror_point = {
        234: 1,
        93: 2,
        132: 3,
        58: 4,
        172: 5,
        136: 6,
        150: 7,
        149: 8,
        176: 9,
        148: 10,
        152: 11,
        377: 12,
        400: 13,
        378: 14,
        379: 15,
        365: 16,
        397: 17,
        288: 18,
        361: 19,
        323: 20,
        454: 21,
        356: 22,
        389: 23,
        251: 24,
        284: 25,
        332: 26,
        297: 27,
        338: 28,
        10: 29,
        109: 30,
        67: 31,
        103: 32,
        54: 33,
        21: 34,
        162: 35,
        127: 36,
    }
    mask_img = mask_img
    mask_points = mask_points
    src_pts = []
    for i in sorted(mask_points.keys()):
        try:
            src_pts.append(np.array(mask_points[i]))
        except ValueError:
            continue
    src_pts = np.array(src_pts, dtype="float32")
    extend_y = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
    ]
    extend_y = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
    ]
    minimize_y = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    face_points = {}
    for i in faces[0]:
        for j in mirror_point.keys():
            if i[0] == j:
                if mirror_point[i[0]] in minimize_y:
                    face_points[mirror_point[j]] = [
                        float(i[1]),
                        float(i[2] - int(mask_up)),
                    ]
                else:
                    if mirror_point[i[0]] in extend_y:
                        face_points[mirror_point[j]] = [
                            float(i[1]),
                            float(i[2] + int(mask_down)),
                        ]
                    else:
                        face_points[mirror_point[j]] = [float(i[1]), float(i[2])]
    dst_pts = []
    for i in sorted(face_points.keys()):
        try:
            dst_pts.append(np.array(face_points[i]))
        except ValueError:
            continue
    dst_pts = np.array(dst_pts, dtype="float32")
    M, _ = cv2.findHomography(src_pts, dst_pts)
    transformed_mask = cv2.warpPerspective(
        mask_img,
        M,
        (image.shape[1], image.shape[0]),
        None,
        cv2.INTER_LINEAR,
        cv2.BORDER_CONSTANT,
        borderValue=[0, 0, 0, 0],
    )
    png_image = normalize8(transformed_mask)
    new_image = overlay_transparent(image, png_image, 0, 0)
    return image


def main(mask_up, mask_down, flip_the_video, mask_path, csv_path):
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask_img = mask_img.astype(np.float32)
    mask_img = mask_img / 255.0
    mask_annotation = csv_path
    mask_points = {}
    with open(mask_annotation) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for i, row in enumerate(csv_reader):
            mask_points[int(row[0])] = [float(row[1]), float(row[2])]
    while ret:

        success, img = cap.read()
        if flip_the_video == "Yes":
            img = cv2.flip(img, 1)
        elif flip_the_video == "No":
            pass

        try:
            faces = face_point(img, minDetectionCon, minTrackingCon)

            if len(faces) >= 1:
                img = mask_overlay(
                    img, faces, mask_up, mask_down, mask_img, mask_points
                )

        except Exception as e:
            print(f"error is {e}")
        if img is None:
            break
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    cap.release()


if __name__ == "__main__":

    select_mask = st.selectbox(
        "Select your mask", ("Villain Mask", "Circle Mask", "Add Your Own Mask")
    )
    if select_mask == "Villain Mask":
        mask_path = "./assets/villain_mask.png"
        csv_path = "./assets/villain_mask.csv"
    if select_mask == "Circle Mask":
        mask_path = "./assets/circle_mask.png"
        csv_path = "./assets/circle_mask.csv"
    if select_mask == "Add Your Own Mask":
        uploaded_png_file = st.file_uploader("Choose an mask png image.", type="png")
        if uploaded_png_file is not None:
            with open(f"./temp/{uploaded_png_file.name}", "wb") as f:
                f.write(uploaded_png_file.getbuffer())
            mask_path = f"./temp/{uploaded_png_file.name}"
        uploaded_csv_file = st.file_uploader("Choose a file")
        if uploaded_csv_file is not None:
            with open(f"./temp/{uploaded_csv_file.name}", "wb") as f:
                f.write(uploaded_csv_file.getbuffer())
            csv_path = f"./temp/{uploaded_csv_file.name}"

    mask_up = st.slider("Make mask bigger upper size")
    mask_down = st.slider("Make mask bigger lower size")
    flip_the_video = st.selectbox("Horizontally flip video ", ("Yes", "No"))

    if st.button("Start adding face mask"):
        main(mask_up, mask_down, flip_the_video, mask_path, csv_path)
        st.markdown(f"## Face mask added successfully check your export folder")

        for i in os.listdir("./temp/"):
            try:
                os.remove(os.remove(f"./temp/{i}"))
            except:
                pass
