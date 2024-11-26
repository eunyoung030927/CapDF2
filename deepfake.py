import streamlit as st
from PIL import Image
from detection import *

def main():
    st.set_page_config(page_title="딥페이크 탐지 시스템", page_icon="🕵️") # layout="wide"

    st.markdown("""
        <style>
            .prob_text {
                font-size: 1.2rem;
                color: #4a4a4a;
                }
        </style>
    """, unsafe_allow_html=True)

    st.header("🕵️‍♀️ 딥페이크 탐지 시스템")
    st.markdown("### 이미지를 업로드하여 딥페이크 여부를 확인해보세요!")

    img_file = st.file_uploader('탐지를 원하는 이미지를 업로드하세요.', type=['png','jpg','jpeg'])

    if img_file is not None:
        img = Image.open(img_file) 
        st.image(img, caption="업로드된 이미지")

        with st.spinner("이미지를 분석 중입니다..."):
            detector = Detection()
            prob, pred = detector.detect(img)
            prob = round(prob*100, 2)

        if pred: # True
            st.success("✅ 해당 이미지는 진짜(Real) 이미지입니다.")
        else:
            st.error("❌ 해당 이미지는 딥페이크로 조작된 이미지입니다.")

        st.markdown(f"<p class='prob_text'>Real일 확률: {prob}%</p>", unsafe_allow_html=True)
        st.progress(prob / 100)

if __name__ == "__main__":
    main()

