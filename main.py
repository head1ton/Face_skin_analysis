import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# -----------------------
# Mediapipe 초기 설정
# -----------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# -----------------------
# 얼굴 부위 추출 함수 (이마, 눈가, 코, 볼, 턱)
# -----------------------
def extract_facial_regions(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    h, w = image.shape[:2]

    region_coords = {}

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        # 이마 (forehead): 눈썹 위쪽
        forehead_pts = [face.landmark[i] for i in [10, 338, 297, 332, 284, 251]]
        xs = [int(p.x * w) for p in forehead_pts]
        ys = [int(p.y * h) for p in forehead_pts]
        region_coords["forehead"] = (min(xs), min(ys), max(xs), max(ys))

        # 눈가 (eye corners): 눈 바깥쪽 주변
        left_eye = face.landmark[33]
        right_eye = face.landmark[263]
        region_coords["eyes"] = (int(left_eye.x * w)-30, int(left_eye.y * h)-30,
                                 int(right_eye.x * w)+30, int(right_eye.y * h)+30)

        # 코 (nose): 콧등
        nose = face.landmark[1]
        region_coords["nose"] = (int(nose.x * w)-20, int(nose.y * h)-20,
                                 int(nose.x * w)+20, int(nose.y * h)+20)

        # 턱 (chin): 아래 턱
        chin = face.landmark[152]
        region_coords["chin"] = (int(chin.x * w)-30, int(chin.y * h)-30,
                                 int(chin.x * w)+30, int(chin.y * h)+30)

        # 볼 (cheeks): 대략적인 위치
        region_coords["cheeks"] = (int(w * 0.3), int(h * 0.45), int(w * 0.7), int(h * 0.65))

    return region_coords

# -----------------------
# 피부 특징 분석 샘플 (단순 평균 픽셀값 기반)
# -----------------------
def analyze_skin(image, regions):
    result = {}
    for region, (x1, y1, x2, y2) in regions.items():
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        wrinkle = float(np.var(gray))
        pores = float(np.mean(cv2.Laplacian(gray, cv2.CV_64F)))
        pigment = float(np.mean(gray)) / 255
        redness = float(np.mean(roi[:,:,2]))
        result[region] = {
            "wrinkle": round(wrinkle, 2),
            "pores": round(abs(pores), 2),
            "pigmentation": round(pigment, 4),
            "redness": round(redness, 2)
        }
    return result

# -----------------------
# 해석, 추천, 루틴 함수 (동일)
# -----------------------
def interpret_skin_status(result):
    summary = {}
    total_wrinkle, total_pore, total_pigment, total_redness = 0, 0, 0, 0
    regions = len(result)

    for region, data in result.items():
        total_wrinkle += data["wrinkle"]
        total_pore += data["pores"]
        total_pigment += data["pigmentation"]
        total_redness += data["redness"]

    avg_wrinkle = total_wrinkle / regions
    avg_pore = total_pore / regions
    avg_pigment = total_pigment / regions
    avg_redness = total_redness / regions

    summary["주름"] = "많음" if avg_wrinkle > 1300 else "보통" if avg_wrinkle > 800 else "거의 없음"
    summary["모공"] = "넓음" if avg_pore > 10 else "보통"
    summary["색소침착"] = "있음" if avg_pigment > 0.05 else "없음"
    summary["붉은기"] = "많음" if avg_redness > 130 else "약간 있음" if avg_redness > 100 else "없음"

    return summary

def recommend_products(skin_summary):
    product_list = []
    if skin_summary["주름"] != "거의 없음":
        product_list.append("레티놀 세럼 / 펩타이드 크림")
    if skin_summary["모공"] == "넓음":
        product_list.append("BHA (살리실산) 토너 / 클레이 마스크")
    if skin_summary["색소침착"] == "있음":
        product_list.append("비타민C 세럼 / 나이아신아마이드")
    if skin_summary["붉은기"] != "없음":
        product_list.append("센텔라 크림 / 알로에 수딩젤")
    return product_list

def generate_daily_routine(skin_summary):
    routine = {"Morning": [], "Night": []}
    routine["Morning"].append("순한 클렌저")
    routine["Morning"].append("수분크림")
    routine["Morning"].append("자외선차단제")
    if skin_summary["색소침착"] == "있음":
        routine["Morning"].insert(1, "비타민C 세럼")

    routine["Night"].append("클렌징 오일 + 젤 클렌저")
    routine["Night"].append("토너")
    if skin_summary["주름"] != "거의 없음":
        routine["Night"].append("레티놀 세럼")
    if skin_summary["모공"] == "넓음":
        routine["Night"].append("BHA 토너")
    routine["Night"].append("수분크림 또는 재생크림")

    return routine

# -----------------------
# Streamlit UI
# -----------------------
st.title("💆 얼굴 피부 분석 (실시간)")
uploaded_file = st.file_uploader("얼굴 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    st.image(image, caption="업로드된 이미지", use_container_width=True)

    regions = extract_facial_regions(image_np)
    st.subheader("🔍 얼굴 부위 탐지 결과")
    st.json(regions)

    result = analyze_skin(image_np, regions)
    st.subheader("📊 분석 결과")
    st.json(result)

    skin_summary = interpret_skin_status(result)
    st.header("📋 피부 상태 요약")
    for k, v in skin_summary.items():
        st.markdown(f"- **{k}**: {v}")

    st.header("🧴 추천 화장품")
    for item in recommend_products(skin_summary):
        st.markdown(f"- {item}")

    st.header("🕒 데일리 루틴")
    routine = generate_daily_routine(skin_summary)
    for time, steps in routine.items():
        st.subheader(time)
        for step in steps:
            st.markdown(f"- {step}")
