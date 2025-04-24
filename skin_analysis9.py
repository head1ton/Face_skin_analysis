import streamlit as st
import cv2
import dlib
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

# dlib 얼굴 감지기와 랜드마크 예측기
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib_data/shape_predictor_68_face_landmarks.dat")

# YOLO 모델 로딩
yolo_model = YOLO("yolov8s.pt")


# 얼굴 분석 클래스
class SkinAnalyzer:
    def __init__(self, yolo_model_path="yolov8s.pt"):
        self.model = YOLO(yolo_model_path)

    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 감지
        faces = detector(gray)
        analysis_result = {}

        for face in faces:
            landmarks = predictor(gray, face)

            # 얼굴 부위별 분석
            self.analyze_forehead(frame, landmarks, analysis_result)
            self.analyze_eye_area(frame, landmarks, analysis_result)
            self.analyze_nose(frame, landmarks, analysis_result)
            self.analyze_mouth(frame, landmarks, analysis_result)

        return analysis_result

    def analyze_forehead(self, frame, landmarks, analysis_result):
        # 이마 부위 랜드마크 추출
        forehead_points = [landmarks.part(i) for i in
                           range(17, 21)]  # 17~20번이 이마 부위
        self.draw_landmarks(frame, forehead_points)

        # 이마 피부 상태 분석
        forehead_crop = self.crop_face(frame, forehead_points)
        wrinkle_score = self.analyze_wrinkle(forehead_crop)
        analysis_result["forehead"] = self.interpret_score(wrinkle_score,
                                                           "wrinkle")

    def analyze_eye_area(self, frame, landmarks, analysis_result):
        # 눈 부위 랜드마크 추출
        eye_points = [landmarks.part(i) for i in range(36, 42)]  # 왼쪽 눈
        self.draw_landmarks(frame, eye_points)

        # 눈 피부 상태 분석
        eye_crop = self.crop_face(frame, eye_points)
        wrinkle_score = self.analyze_wrinkle(eye_crop)
        analysis_result["eye_area"] = self.interpret_score(wrinkle_score,
                                                           "wrinkle")

    def analyze_nose(self, frame, landmarks, analysis_result):
        # 코 부위 랜드마크 추출
        nose_points = [landmarks.part(i) for i in range(27, 36)]  # 코
        self.draw_landmarks(frame, nose_points)

        # 코 피부 상태 분석
        nose_crop = self.crop_face(frame, nose_points)
        pore_score = self.analyze_pore(nose_crop)
        analysis_result["nose"] = self.interpret_score(pore_score, "pore")

    def analyze_mouth(self, frame, landmarks, analysis_result):
        # 입 부위 랜드마크 추출
        mouth_points = [landmarks.part(i) for i in range(48, 60)]  # 입
        self.draw_landmarks(frame, mouth_points)

        # 입 피부 상태 분석
        mouth_crop = self.crop_face(frame, mouth_points)
        sebum_score = self.analyze_sebum(mouth_crop)
        analysis_result["mouth"] = self.interpret_score(sebum_score, "sebum")

    def draw_landmarks(self, frame, points):
        for point in points:
            cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)

    def crop_face(self, frame, points):
        # 랜드마크를 기반으로 얼굴 부위 자르기
        x_min = min([p.x for p in points])
        y_min = min([p.y for p in points])
        x_max = max([p.x for p in points])
        y_max = max([p.y for p in points])
        return frame[y_min:y_max, x_min:x_max]

    def analyze_wrinkle(self, crop_img):
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        wrinkle_score = np.sum(edges) / 255
        return wrinkle_score

    def analyze_pore(self, crop_img):
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
        pore_score = np.sum(binary) / 255
        return pore_score

    def analyze_sebum(self, crop_img):
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        brightness = hsv[..., 2]
        return np.mean(brightness)

    def interpret_score(self, score, category):
        if category == "wrinkle":
            if score < 1000:
                return "좋음 😊"
            elif score < 3000:
                return "보통 😐"
            else:
                return "주의 ⚠️"
        elif category == "pore":
            if score < 200:
                return "좋음 😊"
            elif score < 500:
                return "보통 😐"
            else:
                return "주의 ⚠️"
        elif category == "sebum":
            if score < 100:
                return "좋음 😊"
            elif score < 200:
                return "보통 😐"
            else:
                return "주의 ⚠️"

    # 추천 시스템
    def recommend_products(self, analysis_result):
        recommendations = []

        # 주름 분석 결과
        if "forehead" in analysis_result or "eye_area" in analysis_result:
            recommendations.append("주름 개선 크림")

        # 모공 분석 결과
        if "nose" in analysis_result:
            recommendations.append("모공 축소 세럼")

        # 유분 분석 결과
        if "mouth" in analysis_result:
            recommendations.append("유분 조절 크림")

        return recommendations


# 실시간 웹캠을 통한 얼굴 피부 분석
def analyze_video_frame(frame):
    analyzer = SkinAnalyzer("yolov8s.pt")
    result = analyzer.analyze_frame(frame)
    recommendations = analyzer.recommend_products(result)
    return result, recommendations


# Streamlit으로 실시간 카메라 스트리밍
def webcam_interface():
    cap = cv2.VideoCapture(0)  # 웹캠 열기
    st.title("실시간 피부 상태 분석")

    # 웹캠 영상을 스트리밍하는 부분
    frame_window = st.image([])  # 웹캠 결과를 출력할 부분

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 얼굴 분석 및 피부 상태 분석
        result, recommendations = analyze_video_frame(frame)

        # 화면에 얼굴 분석 결과와 추천 결과를 표시
        for key, value in result.items():
            cv2.putText(frame, f"{key}: {value}",
                        (10, 30 + list(result.keys()).index(key) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 추천화장품 출력
        cv2.putText(frame, f"추천 화장품: {', '.join(recommendations)}",
                    (10, 60 + len(result) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        # WebCam 이미지 전송
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb, channels="RGB")

        # 1초에 한 번씩 갱신
        time.sleep(10)

    cap.release()


# Streamlit 앱 실행
if __name__ == "__main__":
    webcam_interface()
