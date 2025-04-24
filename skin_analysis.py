import cv2
import dlib
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

    return base_frame


# ========== Streamlit 앱 시작 ==========
st.set_page_config(layout="wide")
st.title("📷 실시간 얼굴 피부 분석 데모")

analyzer = FaceSkinAnalyzer()
predictor_path = "shape_predictor_68_face_landmarks.dat"

try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
except:
    st.error("⚠️ 'shape_predictor_68_face_landmarks.dat' 파일이 필요합니다.")

# 초기 상태 정의
if 'captured' not in st.session_state:
    st.session_state.captured = False
if 'captured_frame' not in st.session_state:
    st.session_state.captured_frame = None
if 'capture_btn_clicked' not in st.session_state:
    st.session_state.capture_btn_clicked = False
if 'result' not in st.session_state:  # 'result'를 초기화
    st.session_state.result = None

# 버튼 핸들러
def start_capture():
    st.session_state.capture_btn_clicked = True

def reset_all():
    st.session_state.captured = False
    st.session_state.captured_frame = None
    st.session_state.capture_btn_clicked = False
    st.session_state.result = None  # 분석 결과 리셋
    # 세션 상태 초기화 후, 화면에서 사라지도록
    st.session_state.captured = False
    st.session_state.captured_frame = None
    st.session_state.result = None  # 분석 결과를 삭제
    # 비워놓은 곳을 다시 초기화
    frame_window.empty()

# 버튼 UI
start_col, reset_col = st.columns(2)
with start_col:
    st.button("📸 얼굴 캡처 및 피부 분석 시작", on_click=start_capture, disabled=st.session_state.capture_btn_clicked, key="capture_btn")

with reset_col:
    st.button("🔁 초기화", on_click=reset_all, key="reset_btn")  # 초기화 버튼 항상 활성화

frame_window = st.empty()

# ========== 실시간 영상 처리 ==========
cap = cv2.VideoCapture(0)
while cap.isOpened():
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

    if st.session_state.capture_btn_clicked and not st.session_state.captured:
        st.session_state.captured = True
        st.session_state.captured_frame = frame.copy()
        frame_window.empty()
        break

cap.release()

# ========== 캡처 이미지 분석 ==========
if st.session_state.captured_frame is not None:
    if not st.session_state.captured:
        st.session_state.captured = True

    st.subheader("🎬 슬라이딩 캡처 애니메이션")
    final_frame = sliding_gesture_on_single_frame(st.session_state.captured_frame)

    st.subheader("🔍 피부 분석 진행 중...")
    progress = st.progress(0)
    st.session_state.result = analyzer.analyze_frame(final_frame, progress_callback=progress.progress)

    st.subheader("📊 분석 결과")
    for part, analysis in st.session_state.result.items():
        st.write(f"📌 **{part.upper()}**: {analysis}")

    st.subheader("💡 추천 화장품")
    for rec in analyzer.recommend_products(st.session_state.result):
        st.success(f"🧴 {rec}")
