import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import time
import random

st.set_page_config(layout="wide")
st.title("📈 실시간 웹캠 + 투명 차트 오버레이")


# 차트 이미지 생성 함수
def generate_chart(values):
    fig, ax = plt.subplots(figsize=(5, 1.5), dpi=100)
    fig.patch.set_alpha(0.0)  # 전체 배경 투명
    ax.set_facecolor('none')  # 축 배경 투명
    ax.plot(values, color='lime', linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True, bbox_inches='tight',
                pad_inches=0)
    buf.seek(0)
    chart_image = Image.open(buf)
    chart_rgba = np.array(chart_image.convert("RGBA"))
    plt.close()
    return chart_rgba


# 차트 투명 오버레이
def overlay_chart(frame, chart_img):
    h, w, _ = frame.shape
    ch, cw, _ = chart_img.shape
    # Resize chart to match webcam width
    chart_resized = cv2.resize(chart_img, (w, int(ch * w / cw)))
    overlay = np.zeros_like(frame, dtype=np.uint8)
    y_offset = 10  # top margin
    overlay[y_offset:y_offset + chart_resized.shape[0], 0:w] = chart_resized[:,
                                                               :, :3]

    alpha_chart = chart_resized[:, :, 3:] / 255.0
    alpha_frame = 1.0 - alpha_chart
    for c in range(3):
        frame[y_offset:y_offset + chart_resized.shape[0], 0:w, c] = (
            alpha_chart[:, :, 0] * chart_resized[:, :, c] +
            alpha_frame[:, :, 0] * frame[
                                   y_offset:y_offset + chart_resized.shape[0],
                                   0:w, c]
        )
    return frame


# 웹캠 실행
cap = cv2.VideoCapture(0)
chart_values = [random.randint(30, 70) for _ in range(30)]
frame_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 새 데이터 추가
    chart_values.append(chart_values[-1] + random.randint(-5, 5))
    chart_values = chart_values[-30:]

    # 차트 생성 및 오버레이
    chart_img = generate_chart(chart_values)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = overlay_chart(frame, chart_img)

    # 화면 출력
    frame_placeholder.image(frame, channels="RGB")
    time.sleep(0.1)

cap.release()
