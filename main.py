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


# ========== Streamlit 앱 시작 ==========
st.title("📷 실시간 얼굴 피부 분석 데모")
st.write("그냥 이런것도 된다~ 라고 보세요...")
st.write("분석 수치는 모두 각각 데이터 수집 후 학습 후에 나와야하는 건데 샘플로 만드는 거라 수치는 대부분 랜덤값임. 구현시간 많이 걸림..-_-;")
st.write("Cloud 무료 호스팅이라 지원되는 카메라로 변경했더니 구림..")

frame_window = st.empty()

# ========== 실시간 영상 처리 ==========
img_file_buffer = st.camera_input("Face Skin Scan")

if img_file_buffer is not None:
    st.write(img_file_buffer)
