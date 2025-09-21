import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="findmovie", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #2d1b2e 100%) !important;
    font-family: 'Poppins', sans-serif !important;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden; }

h1 {
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(45deg, #ff1493, #ff69b4, #ff1493) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-align: center !important;
    margin-bottom: 20px !important;
    animation: glow 2s ease-in-out infinite alternate !important;
}

@keyframes glow {
    from { filter: brightness(1); }
    to { filter: brightness(1.2); }
}

.subtitle-text {
    font-size: 1.2rem !important;
    color: #ff69b4 !important;
    text-align: center !important;
    margin-bottom: 40px !important;
    animation: fadeInUp 1s ease-out 0.5s forwards !important;
    opacity: 0;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.stats-text {
    color: #ff69b4 !important;
    font-size: 1.1rem !important;
    text-align: center !important;
    margin: 10px 0 !important;
    padding: 15px !important;
    background: rgba(0, 0, 0, 0.7) !important;
    border: 2px solid rgba(255, 20, 147, 0.3) !important;
    border-radius: 15px !important;
    backdrop-filter: blur(10px) !important;
    animation: fadeInUp 1s ease-out 1s forwards !important;
    opacity: 0;
}

.stats-text:hover {
    transform: translateY(-5px) !important;
    border-color: #ff1493 !important;
    box-shadow: 0 15px 30px rgba(255, 20, 147, 0.3) !important;
}

h3 {
    color: white !important;
    font-size: 1.8rem !important;
    text-align: center !important;
    margin: 30px 0 20px 0 !important;
    text-shadow: 0 0 10px rgba(255, 20, 147, 0.3) !important;
}

.stSelectbox div, .stSelectbox div span {
    color: white !important;
    background-color: rgba(0, 0, 0, 0.9) !important;
}

.stSelectbox label {
    color: #ff69b4 !important;
    font-weight: 500 !important;
    font-size: 1.1rem !important;
}

.stSlider label {
    color: #ff69b4 !important;
    font-weight: 500 !important;
    font-size: 1.1rem !important;
}

.stButton > button {
    width: 100% !important;
    height: 60px !important;
    background: linear-gradient(45deg, #ff1493, #ff69b4) !important;
    border: none !important;
    border-radius: 15px !important;
    color: white !important;
    font-size: 1.2rem !important;
    font-weight: bold !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 5px 15px rgba(255, 20, 147, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 30px rgba(255, 20, 147, 0.5) !important;
    background: linear-gradient(45deg, #ff69b4, #ff1493) !important;
}

.stSuccess, .stWarning, .stError {
    border-radius: 10px !important;
    color: #ff69b4 !important;
}
            
.stTable, .dataframe {
    color: white !important;
    background-color: transparent !important;
    font-size: 1rem !important;
}

thead tr th {
    color: #ff69b4 !important;
    font-weight: bold !important;
}

tbody tr td {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.title("findmovies")
st.markdown('<p class="subtitle-text">ค้นหาภาพยนตร์ที่คุณชื่นชอบ</p>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        movies_df = pd.read_csv('movies.csv')
        ratings_df = pd.read_csv('ratings.csv')
        return movies_df, ratings_df
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ movies.csv หรือ ratings.csv ")
        return None, None

def create_pivot_table(movies_df, ratings_df):
    ratings_with_movies = pd.merge(ratings_df, movies_df, on='movieId')
    pivot_table = ratings_with_movies.pivot_table(
        index='userId',
        columns='title',
        values='rating'
    ).fillna(0)
    return pivot_table

def compute_similarity(pivot_table):
    movie_features = pivot_table.T
    movie_similarity = cosine_similarity(movie_features)
    similarity_df = pd.DataFrame(
        movie_similarity,
        index=movie_features.index,
        columns=movie_features.index
    )
    return similarity_df

def recommend_movies(movie_title, similarity_df, n=5):
    if movie_title not in similarity_df.index:
        return pd.DataFrame()
    
    similar_scores = similarity_df[movie_title]
    similar_movies = similar_scores.sort_values(ascending=False)[1:n+1]
    
    recommendations = pd.DataFrame({
        'ชื่อภาพยนตร์': similar_movies.index,
        'คะแนนความคล้าย': similar_movies.values
    })
    
    return recommendations

def main():
    movies_df, ratings_df = load_data()
    
    if movies_df is None or ratings_df is None:
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stats-text">จำนวนภาพยนตร์ทั้งหมด: <strong>{len(movies_df):,}</strong></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stats-text">จำนวนผู้ใช้ทั้งหมด: <strong>{ratings_df["userId"].nunique():,}</strong></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stats-text">จำนวนการให้คะแนนทั้งหมด: <strong>{len(ratings_df):,}</strong></div>', unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("กำลังสร้างตาราง pivot...")
    pivot_table = create_pivot_table(movies_df, ratings_df)
    progress_bar.progress(33)
    status_text.text("กำลังคำนวณความคล้ายของภาพยนตร์...")
    similarity_df = compute_similarity(pivot_table)
    progress_bar.progress(66)
    movie_list = sorted(similarity_df.index.tolist())
    progress_bar.progress(100)
    status_text.success("พร้อมใช้งานแล้วจร้า")

    import time
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

    st.subheader("ค้นหาภาพยนตร์ที่คล้ายกัน")

    selected_movie = st.selectbox("เลือกภาพยนตร์:", movie_list)

    num_recommendations = st.slider("จำนวนภาพยนตร์ที่ต้องการแนะนำ:", 1, 20, 5)

    if st.button("แนะนำภาพยนตร์ที่คล้ายกัน"):
        with st.spinner("กำลังค้นหาภาพยนตร์ที่เหมาะสม..."):
            recommendations = recommend_movies(selected_movie, similarity_df, n=num_recommendations)
            if recommendations.empty:
                st.warning(f"ไม่พบข้อมูลเพียงพอสำหรับภาพยนตร์ '{selected_movie}'")
            else:
                st.subheader(f"ภาพยนตร์ที่คล้ายกับ '{selected_movie}':")
                recommendations['คะแนนความคล้าย'] = recommendations['คะแนนความคล้าย'].apply(lambda x: f"{x:.2%}")
                st.table(recommendations)

                st.subheader("กราฟแสดงคะแนนความคล้าย")
                chart_data = recommendations.copy()
                chart_data['คะแนนความคล้าย'] = chart_data['คะแนนความคล้าย'].apply(lambda x: float(x.strip('%')) / 100)

                chart = alt.Chart(chart_data).mark_bar(
                            color='#ff69b4'
                ).encode(
                    x=alt.X('ชื่อภาพยนตร์:N', sort=None, axis=alt.Axis(labelAngle=-45, labelColor='white', titleColor='white')),
                    y=alt.Y('คะแนนความคล้าย:Q', axis=alt.Axis(labelColor='white', titleColor='white'))
                ).properties(
                    width=700,
                    height=400,
                    background='#0f0f0f' 
                ).configure_view(
                    stroke=None
                ).configure_axis(
                    grid=False,
                    domainColor='#444'
                ).configure_title(
                    color='white'
                ).configure(
                    background='#0f0f0f'
                )

            st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
