import streamlit as st
import cv2
import dlib
import numpy as np
from ultralytics import YOLO
import time

# 얼굴 인식 및 랜드마크 모델 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# YOLO 모델 로드
yolo_model = YOLO("yolov8s.pt")

# SkinAnalyzer 클래스 정의
class SkinAnalyzer:
    def __init__(self, model_path="yolov8s.pt"):
        self.model = YOLO(model_path)

    def analyze_frame(self, frame, progress_callback=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        results = {}

        if progress_callback:
            progress_callback(10)

        for face in faces:
            landmarks = predictor(gray, face)

            if progress_callback:
                progress_callback(30)
            results["forehead"] = self.analyze_region(frame, landmarks, range(17, 21), "wrinkle")

            if progress_callback:
                progress_callback(50)
            results["eye_area"] = self.analyze_region(frame, landmarks, range(36, 42), "wrinkle")

            if progress_callback:
                progress_callback(70)
            results["nose"] = self.analyze_region(frame, landmarks, range(27, 36), "pore")

            if progress_callback:
                progress_callback(90)
            results["mouth"] = self.analyze_region(frame, landmarks, range(48, 60), "sebum")

        if progress_callback:
            progress_callback(100)

        return results

    def analyze_region(self, frame, landmarks, idx_range, category):
        points = [landmarks.part(i) for i in idx_range]
        region = self.crop_region(frame, points)

        if category == "wrinkle":
            return self.interpret_score(self.analyze_wrinkle(region), category)
        elif category == "pore":
            return self.interpret_score(self.analyze_pore(region), category)
        elif category == "sebum":
            return self.interpret_score(self.analyze_sebum(region), category)

    def crop_region(self, frame, points):
        x_min = min(p.x for p in points)
        y_min = min(p.y for p in points)
        x_max = max(p.x for p in points)
        y_max = max(p.y for p in points)
        return frame[y_min:y_max, x_min:x_max]

    def analyze_wrinkle(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges) / 255

    def analyze_pore(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
        return np.sum(binary) / 255

    def analyze_sebum(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:, :, 2])  # 밝기 평균

    def interpret_score(self, score, category):
        if category == "wrinkle":
            return f"주름 점수: {score:.2f} → {'좋음 😊' if score < 1000 else '보통 😐' if score < 3000 else '주의 ⚠️'}"
        elif category == "pore":
            return f"모공 점수: {score:.2f} → {'좋음 😊' if score < 200 else '보통 😐' if score < 500 else '주의 ⚠️'}"
        elif category == "sebum":
            return f"유분 점수: {score:.2f} → {'좋음 😊' if score < 100 else '보통 😐' if score < 200 else '주의 ⚠️'}"

    def recommend_products(self, result):
        recs = []
        if "forehead" in result or "eye_area" in result:
            recs.append("주름 개선 크림")
        if "nose" in result:
            recs.append("모공 축소 세럼")
        if "mouth" in result:
            recs.append("유분 조절 크림")
        return recs

# 랜드마크 시각화 함수
def draw_landmarks(frame, landmarks):
    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

# 슬라이딩 제스처 효과
def sliding_gesture(frame):
    h, w, _ = frame.shape
    gesture_frame = frame.copy()
    line_color = (0, 255, 0)
    for x in range(0, w, 20):
        temp = gesture_frame.copy()
        cv2.line(temp, (x, 0), (x, h), line_color, 3)
        st.image(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        time.sleep(0.01)
    return gesture_frame

# 메인 앱 함수
def main():
    st.title("📸 얼굴 피부 분석 시스템 - 실시간 랜드마크 & 슬라이딩 캡처")

    st.header("▶️ 컨트롤")
    capture_btn = st.button("얼굴 캡처")

    frame_window = st.image([])  # 영상 표시
    cap = cv2.VideoCapture(0)
    captured = False
    captured_frame = None

    analyzer = SkinAnalyzer()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("카메라 연결 실패!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            draw_landmarks(frame, landmarks)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb, channels="RGB", use_column_width=True)

        if capture_btn and not captured:
            captured = True
            captured_frame = frame.copy()
            st.success("✅ 얼굴 캡처 완료!")
            break

    cap.release()

    if captured and captured_frame is not None:
        st.subheader("🎬 슬라이딩 캡처 애니메이션")
        final_frame = sliding_gesture(captured_frame)

        st.subheader("🔍 피부 분석 진행 중...")
        progress = st.progress(0)
        result = analyzer.analyze_frame(final_frame, progress_callback=progress.progress)

        st.markdown("---")
        st.subheader("📊 분석 결과")
        for part, analysis in result.items():
            st.write(f"📌 {part.upper()}: {analysis}")

        st.subheader("💡 추천 화장품")
        for rec in analyzer.recommend_products(result):
            st.success(f"🧴 {rec}")

# 실행
if __name__ == "__main__":
    main()
