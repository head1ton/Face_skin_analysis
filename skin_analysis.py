import cv2
import dlib
import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt
import random
import seaborn as sns
import platform
import requests
import bz2
import os


def download_and_extract_dlib_landmark_model(save_dir='.'):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'shape_predictor_68_face_landmarks.dat')
    compressed_path = model_path + '.bz2'

    if not os.path.isfile(model_path):
        print("Landmark model not found. Downloading now...")
        url = 'https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2'

        # 다운로드
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(compressed_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download completed.")

        # 압축 해제
        with bz2.BZ2File(compressed_path) as fr, open(model_path, 'wb') as fw:
            fw.write(fr.read())
        print("Extraction completed.")

        # 압축 파일 제거 (선택)
        os.remove(compressed_path)
        print("Compressed file removed.")

    else:
        print("Landmark model already exists.")


# 사용 예시
download_and_extract_dlib_landmark_model()

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
st.set_page_config(page_title="AI 피부 분석", initial_sidebar_state="collapsed")

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
    def __init__(self, landmarks):
        self.landmarks = landmarks
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

    # def get_analysis_scores(self):
    #     # 항목별 점수 설정
    #     scores = {
    #         part: random.randint(50, 100) for part in self.parts
    #     }
    #     return scores

    def analyze_region(self, image, points):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 255)
        roi = cv2.bitwise_and(image, image, mask=mask)
        return roi

    def score_wrinkle(self, roi_gray):
        edges = cv2.Canny(roi_gray, 30, 100)
        wrinkle_score = min(100, np.sum(edges) / 1000)  # Normalize
        return 100 - wrinkle_score

    def score_pore(self, roi_gray):
        blur = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
        pore_score = min(100, blur * 0.5)
        return 100 - pore_score

    def score_redness(self, roi_bgr):
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(roi_hsv, (0, 50, 50), (10, 255, 255))
        red_score = np.mean(red_mask)
        return 100 - red_score / 2

    def score_hydration(self, roi_gray):
        hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
        brightness = np.mean(hist[-50:])
        return min(100, brightness / 5)

    def score_oil(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        v_mean = np.mean(hsv[..., 2])
        return 100 - min(100, v_mean)

    def score_acne(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, (160, 50, 50), (180, 255, 255))
        acne_score = np.count_nonzero(red_mask)
        return 100 - acne_score / 30


    def get_analysis_scores(self, image):
        scores = {}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 예시: 이마(landmarks 인덱스 기반) - 아래는 가정된 좌표
        forehead_pts = [(self.landmarks.part(i).x, self.landmarks.part(i).y) for
                        i in range(17, 27)]
        roi_forehead = self.analyze_region(image, forehead_pts)
        gray_forehead = cv2.cvtColor(roi_forehead, cv2.COLOR_BGR2GRAY)

        scores['wrinkle'] = self.score_wrinkle(gray_forehead)
        scores['pore'] = self.score_pore(gray_forehead)
        scores['hydration'] = self.score_hydration(gray_forehead)
        scores['redness'] = self.score_redness(roi_forehead)
        scores['oil'] = self.score_oil(roi_forehead)
        scores['acne'] = self.score_acne(roi_forehead)

        # 나머지 부위들도 같은 방식으로 이어서 구현
        # ex: eyes, nose, cheeks, chin 등등

        # 기타 항목 (임시 값 혹은 추가 알고리즘 필요)
        # scores['forehead'] = 90  # 예시
        # scores['eyes'] = 90  # 예시
        # scores['nose'] = 85
        # scores['philtrum'] = 88
        # scores['chin'] = 70
        # scores['cheeks'] = 86
        scores['dark_circle'] = 90  # 예시
        scores['skin_texture'] = 85
        scores['lower_eye_fat'] = 88
        scores['elasticity'] = 70
        scores['upper_eyelid'] = 86
        scores['lower_eyelid'] = 85
        scores['glow'] = 78
        scores['tear_trough'] = 89
        scores['skin_type'] = 'oily'  # 향후 ML로 분류 가능

        # 기본 점수 항목도 100점 만점으로 매핑
        for part in ['forehead', 'eyes', 'nose', 'philtrum', 'chin', 'cheeks']:
            scores[part] = np.random.randint(10, 95)  # 해당 ROI에 따른 스코어 함수로 대체 가능

        return scores

# ========== 점수 그래프 시각화 ==========
# 점수 그래프 시각화 함수
def plot_scores(result, progress_callback=None):
    for i in range(21):
        time.sleep(0.2)
        if progress_callback:
            progress_callback((i + 1) / 21)

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
def draw_landmarks(frame, landmarks_draw):
    for n in range(0, 68):
        x = landmarks_draw.part(n).x
        y = landmarks_draw.part(n).y
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


# 피부 상태 요약 보고서 생성 함수
def generate_skin_summary(result, scores):
    # 점수 값을 숫자형으로 변환
    numeric_scores = {
        k: float(v) for k, v in scores.items()
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit())
    }

    total_score = sum(numeric_scores.values()) / len(numeric_scores)  # 평균 점수 계산
    summary = ""

    # 점수 범위에 따른 피부 상태 평가
    if total_score >= 80:
        summary = "🌟 피부 상태가 매우 우수합니다! 피부가 건강하고 탱탱합니다."
    elif total_score >= 60:
        summary = "😊 피부 상태가 양호합니다. 다소 개선할 부분이 있을 수 있지만, 크게 문제는 없습니다."
    elif total_score >= 40:
        summary = "⚠️ 피부 상태가 보통입니다. 관리가 필요할 수 있습니다. 추가적인 개선이 필요합니다."
    else:
        summary = "😞 피부 상태가 좋지 않습니다. 주름, 모공, 유분 등 여러 문제가 있을 수 있습니다. 개선이 필요합니다."

    # 상세한 분석 항목 추가
    for part, score in numeric_scores.items():
        if score >= 80:
            summary += f"\n💎 **{part.upper()}**: 우수"
        elif score >= 60:
            summary += f"\n👍 **{part.upper()}**: 보통"
        elif score >= 40:
            summary += f"\n⚠️ **{part.upper()}**: 주의"
        else:
            summary += f"\n❗ **{part.upper()}**: 개선 필요"

    return summary

def face_too_small(face, image, min_face_ratio=0.2):
    img_h, img_w = image.shape[:2]
    face_w = face.right() - face.left()
    face_h = face.bottom() - face.top()

    # 얼굴이 이미지에서 차지하는 비율
    face_area_ratio = (face_w * face_h) / (img_w * img_h)

    return face_area_ratio < min_face_ratio



# ========== Streamlit 앱 시작 ==========
st.title("📷 실시간 얼굴 피부 분석 데모")
st.write("그냥 이런것도 된다~ 라고 보세요...")
st.write("분석 수치는 모두 각각 데이터 수집 후 학습 후에 나와야하는 건데 샘플로 만드는 거라 수치는 대부분 랜덤값임. 구현시간 많이 걸림..-_-;")
st.write("Cloud 무료 호스팅이라 지원되는 카메라로 변경했더니 구림..")
# analyzer = FaceSkinAnalyzer()
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

frame_window = st.empty()

# ========== 실시간 영상 처리 ==========
img_file_buffer = st.camera_input("Face Skin Scan")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        st.warning("얼굴이 제대로 검출되지 않습니다. 다시 촬영해주세요.")
        st.session_state.captured_frame = None
    elif face_too_small(faces[0], frame):
        st.warning("얼굴이 너무 작게 나왔습니다. 카메라에 얼굴을 더 가까이 대고 다시 촬영해주세요.")
        st.session_state.captured_frame = None
    else:
        face = faces[0]
        try:
            landmarks = predictor(gray, face)
            analyzer = FaceSkinAnalyzer(landmarks=landmarks)
        except Exception as e:
            st.error(f"랜드마크 생성 중 오류가 발생했습니다: {e}")
        scores = analyzer.get_analysis_scores(frame)    # 분석 점수 가져오기
        draw_landmarks(frame, landmarks)
        st.session_state.captured_frame = frame.copy()

        # 캡처 이미지 분석
        st.subheader("🎬 슬라이딩 캡처 애니메이션")
        final_frame = sliding_gesture_on_single_frame(st.session_state.captured_frame)

        st.subheader("🔍 피부 분석 진행 중...")
        progress = st.progress(0)
        st.session_state.result = analyzer.analyze_frame(final_frame,
                                                         progress_callback=progress.progress)

        st.subheader("📊 분석 결과")
        for part, analysis in st.session_state.result.items():
            score = analyzer.get_analysis_scores(frame).get(part, 0)
            st.write(f"📌 **{part.upper()}**: {analysis} (Score: {score})")

        # 분석 항목별 점수 그래프
        st.subheader("📈 분석 항목별 점수")
        progress1 = st.progress(0)
        plot_scores(st.session_state.result, progress_callback=progress1.progress)

        st.subheader("💡 추천 화장품")
        for rec in analyzer.recommend_products(st.session_state.result):
            st.success(f"🧴 {rec}")

        # 피부 상태 총평 작성
        st.subheader("💬 피부 상태 총평")
        skin_summary = generate_skin_summary(st.session_state.result, scores)
        st.write(skin_summary)