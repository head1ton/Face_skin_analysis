import cv2
import dlib
import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt
import random
import seaborn as sns
import platform

os = platform.system()
# Windows
if os == 'Windows':
    plt.rc('font', family= 'Malgun Gothic')
# Mac
elif os == 'Darwin':
    plt.rc('font', family= 'AppleGothic')
# Linux
elif os == 'Linux':
    plt.rc('font', family= 'NanumGothic')
else:
    print(f'{os} is not set')

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 피부 분석", layout="wide", initial_sidebar_state="collapsed")

# 페이지 제목
st.markdown("<h1 style='text-align: center; color: #0C7B93;'>🔬 AI 피부 분석 시스템</h1>", unsafe_allow_html=True)

# UI 스타일 적용
# UI 스타일 적용
st.markdown("""
    <style>
        /* 전체 배경과 블록 컨테이너 배경 색 */
        .reportview-container .main .block-container {
            padding: 1rem;
            background-color: #1E1E1E;  /* 어두운 회색 배경 */
        }

        /* 버튼 스타일 */
        .stButton>button {
            background-color: #3C8D99;  /* 차가운 블루 */
            color: white;
            font-size: 18px;
            border-radius: 5px;
            padding: 15px 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .stButton>button:hover {
            background-color: #2A6F7B;  /* 어두운 파랑 */
        }

        /* Progress bar 스타일 */
        .stProgress>div {
            background-color: #4B6B73;  /* 메탈릭 실버 그레이 */
        }

        /* 텍스트 박스 색 */
        .stTextInput>div {
            background-color: #2B2B2B;  /* 어두운 배경 */
            color: #E0E0E0;  /* 연한 회색 글씨 */
        }

        /* 마크다운 텍스트 스타일 */
        .stMarkdown {
            font-size: 16px;
            color: #E0E0E0;  /* 연한 회색 */
        }

        /* 이미지를 표시할 때 배경과 텍스트 조정 */
        .stImage {
            background-color: #1E1E1E;
        }
    </style>
""", unsafe_allow_html=True)


# ========== 얼굴 분석 클래스 ==========
class FaceSkinAnalyzer:
    def __init__(self):
        self.parts = [
            'forehead', 'eyes', 'nose', 'philtrum', 'chin', 'cheeks',
            'wrinkle', 'pore', 'hydration', 'redness', 'oil', 'acne',
            'skin_texture', 'dark_circle', 'lower_eye_fat', 'elasticity',
            'upper_eyelid', 'lower_eyelid', 'glow', 'tear_trough', 'skin_type'
        ]

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
        options = {
            'forehead': ['주름 있음', '피부 톤 고름', '모공 약간 보임'],
            'eyes': ['눈가 주름 있음', '다크서클 보임', '피부 밝음'],
            'nose': ['블랙헤드 있음', '유분 많음', '피부 깨끗함'],
            'philtrum': ['색소 침착 약간', '모공 보임', '균일한 피부'],
            'chin': ['여드름 흔적 있음', '피부톤 불균일', '깨끗한 피부'],
            'cheeks': ['모공 큼', '혈관 비침', '탄력 좋음'],
            'wrinkle': ['주름 30%', '주름 60%', '주름 10%'],
            'pore': ['모공 보임 20%', '모공 보임 50%', '모공 보임 10%'],
            'hydration': ['수분 70%', '수분 50%', '수분 40%'],
            'redness': ['홍조 없음', '홍조 있음'],
            'oil': ['유분 많음', '유분 적음'],
            'acne': ['여드름 없음', '여드름 있음'],
            'skin_texture': ['피부 결 고름', '피부 결 불균일'],
            'dark_circle': ['다크서클 없음', '다크서클 있음'],
            'lower_eye_fat': ['눈 밑 지방 없음', '눈 밑 지방 있음'],
            'elasticity': ['탄력 좋음', '탄력 없음'],
            'upper_eyelid': ['상안검 괜찮음', '상안검 부기 있음'],
            'lower_eyelid': ['하안검 괜찮음', '하안검 부기 있음'],
            'glow': ['광채 있음', '광채 없음'],
            'tear_trough': ['눈물 고랑 없음', '눈물 고랑 있음'],
            'skin_type': ['건성', '지성', '혼합성']
        }
        return random.choice(options[part])

    def recommend_products(self, result):
        recs = []
        if "wrinkle" in result:
            recs.append("주름 개선 크림")
        if "pore" in result:
            recs.append("모공 축소 세럼")
        if "hydration" in result:
            recs.append("수분 보충 크림")
        if "oil" in result:
            recs.append("유분 조절 크림")
        if "acne" in result:
            recs.append("여드름 치료제")
        return recs

    def get_analysis_scores(self):
        # 항목별 점수 설정
        scores = {
            part: random.randint(50, 100) for part in self.parts
        }
        return scores


# ========== 점수 그래프 시각화 ==========
# 점수 그래프 시각화 함수
def plot_scores(result):
    scores = {
        part: random.randint(30, 90) for part in result.keys()
    }
    parts = list(scores.keys())
    scores_values = list(scores.values())

    num_cols = 4  # 한 줄에 4개 항목씩 차트
    num_rows = len(parts) // num_cols + (1 if len(parts) % num_cols > 0 else 0)

    # 대시보드 스타일로 차트 배치
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))

    if num_rows == 1:
        axs = [axs]  # 1행일 경우, axs를 리스트로 감싸서 통일된 형식으로 처리

    for i, part in enumerate(parts):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].bar(part, scores_values[i], color="skyblue")
        axs[row, col].set_title(part, fontsize=12)
        axs[row, col].set_ylabel('점수 (0-100)', fontsize=10)

    # 남은 빈 공간을 비우기
    for i in range(len(parts), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].axis('off')  # 빈 차트 비움

    plt.tight_layout()
    st.pyplot(fig)

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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)

        # 이미지 표시
        placeholder.image(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB), channels="RGB",
                          use_container_width=True)
        time.sleep(0.03)

    # 마지막에 깨끗한 원본 이미지로 다시 덮어쓰기
    placeholder.image(cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB),
                      channels="RGB", use_container_width=True)

    return base_frame


# ========== Streamlit 앱 시작 ==========
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
    frame_window.empty()  # 모든 요소 비우기
    st.session_state.captured = False
    st.session_state.captured_frame = None
    st.session_state.result = None  # 분석 결과를 삭제
    # 비워놓은 곳을 다시 초기화
    st.empty()


# 버튼 UI
start_col, reset_col = st.columns(2)
with start_col:
    st.button("📸 얼굴 캡처 및 피부 분석 시작", on_click=start_capture,
              disabled=st.session_state.capture_btn_clicked, key="capture_btn")

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

# 캡처 이미지 분석
if st.session_state.captured_frame is not None:
    if not st.session_state.captured:
        st.session_state.captured = True

    st.subheader("🎬 슬라이딩 캡처 애니메이션")
    final_frame = sliding_gesture_on_single_frame(
        st.session_state.captured_frame)

    st.subheader("🔍 피부 분석 진행 중...")
    progress = st.progress(0)
    st.session_state.result = analyzer.analyze_frame(final_frame,
                                                     progress_callback=progress.progress)

    st.subheader("📊 분석 결과")
    for part, analysis in st.session_state.result.items():
        score = analyzer.get_analysis_scores().get(part, 0)
        st.write(f"📌 **{part.upper()}**: {analysis} (Score: {score})")

    # 분석 항목별 점수 그래프
    st.subheader("📈 분석 항목별 점수")
    plot_scores(st.session_state.result)

    st.subheader("💡 추천 화장품")
    for rec in analyzer.recommend_products(st.session_state.result):
        st.success(f"🧴 {rec}")