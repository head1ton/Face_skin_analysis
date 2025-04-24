import cv2
import dlib
import numpy as np
import streamlit as st
import time

# ========== 얼굴 분석 클래스 ==========
class FaceSkinAnalyzer:
    def __init__(self):
        self.parts = ['forehead', 'eyes', 'nose', 'philtrum', 'chin', 'cheeks']

    def analyze_frame(self, frame, progress_callback=None):
        h, w, _ = frame.shape
        result = {}
        for i, part in enumerate(self.parts):
            time.sleep(0.2)
            if progress_callback:
                progress_callback((i + 1) / len(self.parts))
            result[part] = self.fake_analysis(part)
        return result

    def fake_analysis(self, part):
        import random
        options = {
            'forehead': ['주름 있음', '피부 톤 고름', '모공 약간 보임'],
            'eyes': ['눈가 주름 있음', '다크서클 보임', '피부 밝음'],
            'nose': ['블랙헤드 있음', '유분 많음', '피부 깨끗함'],
            'philtrum': ['색소 침착 약간', '모공 보임', '균일한 피부'],
            'chin': ['여드름 흔적 있음', '피부톤 불균일', '깨끗한 피부'],
            'cheeks': ['모공 큼', '혈관 비침', '탄력 좋음']
        }
        return random.choice(options[part])

    def recommend_products(self, result):
        recs = []
        if "forehead" in result or "eye_area" in result:
            recs.append("주름 개선 크림")
        if "nose" in result:
            recs.append("모공 축소 세럼")
        if "mouth" in result:
            recs.append("유분 조절 크림")
        return recs

# ========== 얼굴 랜드마크 표시 ==========
def draw_landmarks(frame, landmarks):
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

# ========== 슬라이딩 제스처 ==========
def sliding_gesture_on_single_frame(frame):
    import math

    h, w, _ = frame.shape
    base_frame = frame.copy()
    num_lines = 40
    step = w // num_lines
    line_color = (0, 255, 0)
    line_thickness = 3

    placeholder = st.empty()

    for i in range(num_lines):
        temp = base_frame.copy()

        # 선 위치 계산
        x = int((i / num_lines) * w)

        # 반투명 선 오버레이
        overlay = temp.copy()
        cv2.line(overlay, (x, 0), (x, h), line_color, line_thickness)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, temp, 1 - alpha, 0, temp)

        # 흐림 효과 + 텍스트
        if i % 2 == 0:
            temp = cv2.GaussianBlur(temp, (3, 3), 0)

        progress_percent = int((i / num_lines) * 100)
        cv2.putText(temp, f"Analyzing... {progress_percent}%", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # 이미지 표시
        placeholder.image(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        time.sleep(0.03)

    # 마지막에 깨끗한 원본 이미지로 다시 덮어쓰기
    placeholder.image(cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    st.session_state.captured_frame = None
    return base_frame





# ========== Streamlit 앱 시작 ==========
# st.set_page_config(layout="wide")
st.title("📷 실시간 얼굴 피부 분석")

analyzer = FaceSkinAnalyzer()
predictor_path = "shape_predictor_68_face_landmarks.dat"

try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
except:
    st.error("⚠️ 'shape_predictor_68_face_landmarks.dat' 파일이 필요합니다.")

if 'captured_frame' in st.session_state:
    st.session_state.captured_frame = None


st.session_state.captured_frame = None

frame_window = st.empty()

capture_btn = st.button("📸 얼굴 캡처 및 피부 분석 시작")

captured = False


col1, col2 = st.columns(2)

with col1:

    # ========== 실시간 영상 처리 ==========
    cap = cv2.VideoCapture(0)
    if capture_btn and not captured:
        captured = False

    while True:


        ret, frame = cap.read()
        if not ret:
            st.error("❌ 카메라 연결 실패")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            draw_landmarks(frame, landmarks)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

        if capture_btn and not captured:
            captured = True
            st.session_state.captured_frame = frame.copy()
            st.success("✅ 얼굴 캡처 완료!")

            break

    # cap.release()

with col2:
    # ========== 캡처 이미지 분석 ==========
    if captured and st.session_state.captured_frame is not None:
        st.subheader("🎬 슬라이딩 캡처 애니메이션")
        final_frame = sliding_gesture_on_single_frame(st.session_state.captured_frame)

        st.subheader("🔍 피부 분석 진행 중...")
        progress = st.progress(0)
        result = analyzer.analyze_frame(final_frame, progress_callback=progress.progress)

        st.subheader("📊 분석 결과")
        for part, analysis in result.items():
            st.write(f"📌 **{part.upper()}**: {analysis}")

        st.subheader("💡 추천 화장품")
        for rec in analyzer.recommend_products(result):
            st.success(f"🧴 {rec}")


print('captured: ', captured)
print('capture_btn: ', capture_btn)
print('captured_frame: ', st.session_state.captured_frame)