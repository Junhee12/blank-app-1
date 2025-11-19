#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

#######################
# Page configuration
st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("default")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: white;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)


#######################
# Load data
df_reshaped = pd.read_csv('airforce_men_body_measure2.csv', encoding="cp949") ## 분석 데이터 넣기


#######################
# Sidebar
with st.sidebar:
    st.title("공군 신체 치수 대시보드")
    st.caption("필터를 조정해 체형 분포와 ML 분석 결과를 확인하세요.")

    st.markdown("---")

    # 머신러닝 모드 선택 (군집 + 회귀 두 가지)
    ml_mode = st.radio(
        "머신러닝 분석 모드",
        options=["체형 군집 분석 (K-means)", "신발 사이즈 회귀 예측"],
        index=0
    )

    # 컬러 테마
    color_theme = st.selectbox(
        "컬러 테마",
        options=["blues", "viridis", "reds"],
        index=1
    )

    st.markdown("---")

    # 실제 컬럼명
    HEIGHT_COL = "height"
    CHEST_COL = "chest(inch"
    WAIST_COL = "waist"
    HIP_COL = "hip"
    SHOES_COL = "shoes"

    def get_range(col_name):
        col_min = int(df_reshaped[col_name].min())
        col_max = int(df_reshaped[col_name].max())
        return col_min, col_max

    h_min, h_max = get_range(HEIGHT_COL)
    c_min, c_max = get_range(CHEST_COL)
    w_min, w_max = get_range(WAIST_COL)
    hip_min, hip_max = get_range(HIP_COL)
    s_min, s_max = get_range(SHOES_COL)

    st.subheader("신체치수 필터")

    height_range = st.slider(
        "키 범위 (cm)",
        min_value=h_min,
        max_value=h_max,
        value=(h_min, h_max)
    )

    chest_range = st.slider(
        "가슴둘레 (inch)",
        min_value=float(c_min),
        max_value=float(c_max),
        value=(float(c_min), float(c_max))
    )

    waist_range = st.slider(
        "허리둘레",
        min_value=float(w_min),
        max_value=float(w_max),
        value=(float(w_min), float(w_max))
    )

    hip_range = st.slider(
        "엉덩이둘레",
        min_value=float(hip_min),
        max_value=float(hip_max),
        value=(float(hip_min), float(hip_max))
    )

    shoes_range = st.slider(
        "신발 사이즈",
        min_value=s_min,
        max_value=s_max,
        value=(s_min, s_max)
    )

    st.markdown("---")

    # 샘플 수 선택
    max_n = int(len(df_reshaped))
    sample_options = [500, 1000, 2000, 5000, max_n]
    sample_options = sorted(set([x for x in sample_options if x <= max_n]))
    sample_size = st.select_slider(
        "분석에 사용할 샘플 수",
        options=sample_options,
        value=sample_options[min(2, len(sample_options) - 1)]
    )

    # 필터 적용
    df_filtered = df_reshaped.copy()
    df_filtered = df_filtered[
        (df_filtered[HEIGHT_COL].between(*height_range)) &
        (df_filtered[CHEST_COL].between(*chest_range)) &
        (df_filtered[WAIST_COL].between(*waist_range)) &
        (df_filtered[HIP_COL].between(*hip_range)) &
        (df_filtered[SHOES_COL].between(*shoes_range))
    ]

    if len(df_filtered) > sample_size:
        df_filtered = df_filtered.sample(sample_size, random_state=42)

    st.caption(f"필터 적용 후 데이터 수: {len(df_filtered):,}명")

    # 다른 컬럼에서 사용할 수 있도록 session_state 저장
    st.session_state["df_filtered"] = df_filtered
    st.session_state["ml_mode"] = ml_mode
    st.session_state["color_theme"] = color_theme


#######################
# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')

#######################
# col[0] – 요약 지표
with col[0]:
    st.subheader("📊 요약 지표")

    df_filtered = st.session_state["df_filtered"]

    def safe_mean(col):
        return df_filtered[col].mean()

    avg_height = safe_mean("height")
    avg_chest = safe_mean("chest(inch")
    avg_waist = safe_mean("waist")
    avg_hip = safe_mean("hip")
    avg_shoes = safe_mean("shoes")

    # KPI 카드 (그냥 세로로 쭉)
    st.metric("평균 키 (cm)", f"{avg_height:.1f}")
    st.metric("평균 가슴둘레 (inch)", f"{avg_chest:.1f}")
    st.metric("평균 허리둘레", f"{avg_waist:.1f}")
    st.metric("평균 엉덩이둘레", f"{avg_hip:.1f}")
    st.metric("평균 신발 사이즈", f"{avg_shoes:.1f}")

    st.markdown("---")

    # 신발 사이즈 분포(막대 차트)
    st.markdown("#### 👟 신발 사이즈 분포")

    shoes_counts = df_filtered["shoes"].value_counts().sort_index().reset_index()
    shoes_counts.columns = ["shoes", "count"]   # ← 중복 방지 완전 해결

    fig_shoes = px.bar(
        shoes_counts,
        x="shoes",
        y="count",
        title="신발 사이즈 분포",
        labels={"shoes": "신발 사이즈", "count": "인원 수"},
    )
    st.plotly_chart(fig_shoes, use_container_width=True)


    st.markdown("---")

    # ML 모드 간단 설명
    ml_mode = st.session_state["ml_mode"]
    st.subheader("🤖 ML 분석 모드")

    if ml_mode == "체형 군집 분석 (K-means)":
        st.info("K-means 알고리즘으로 height, chest, waist, hip, shoes를 이용해 체형 군집을 나눕니다.")
    elif ml_mode == "신발 사이즈 회귀 예측":
        st.info("신체 치수(키, 가슴, 허리, 엉덩이)를 입력으로 신발 사이즈를 예측하는 회귀 모델을 사용합니다.")


#######################
# col[1] – 상관관계 & ML 시각화
with col[1]:
    st.subheader("📌 상관관계 & 머신러닝 분석")

    df_filtered = st.session_state["df_filtered"]
    ml_mode = st.session_state["ml_mode"]
    color_theme = st.session_state["color_theme"]

    # 1) 수치형 컬럼 상관관계
    st.markdown("#### 🔗 신체 치수 상관관계")

    numeric_df = df_filtered.select_dtypes(include="number")

    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()

        color_map = {
            "blues": "Blues",
            "viridis": "Viridis",
            "reds": "Reds"
        }
        cmap = color_map.get(color_theme, "Blues")

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=cmap,
            title="수치형 변수 상관계수 히트맵"
        )
        fig_corr.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("상관관계를 계산할 수치형 컬럼이 충분하지 않습니다.")

    st.markdown("---")

    # 2) 머신러닝 분석 시각화
    st.markdown("#### 🤖 머신러닝 결과 시각화")

    feature_cols = ["height", "chest(inch", "waist", "hip", "shoes"]

    # (1) 체형 군집 분석
    if ml_mode == "체형 군집 분석 (K-means)":
        use_cols = ["height", "chest(inch", "waist", "hip", "shoes"]
        X = df_filtered[use_cols].dropna()

        if len(X) < 10:
            st.warning("군집 분석을 수행하기에 데이터가 너무 적습니다.")
        else:
            k = st.slider("군집 개수 (K)", min_value=2, max_value=8, value=4, step=1)

            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            clusters = kmeans.fit_predict(X)

            df_cluster = X.copy()
            df_cluster["cluster"] = clusters.astype(str)

            fig_cluster = px.scatter(
                df_cluster,
                x="height",
                y="shoes",
                color="cluster",
                title="체형 군집 결과 (키 vs 신발 사이즈)",
                labels={"height": "키 (cm)", "shoes": "신발 사이즈"}
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

            st.caption("※ 색깔이 다른 점들은 서로 다른 체형 군집을 의미합니다.")

    # (2) 신발 사이즈 회귀 예측
    elif ml_mode == "신발 사이즈 회귀 예측":
        target_col = "shoes"
        input_cols = ["height", "chest(inch", "waist", "hip"]

        df_reg = df_filtered.dropna(subset=input_cols + [target_col])
        X = df_reg[input_cols]
        y = df_reg[target_col]

        if len(df_reg) < 50:
            st.warning("회귀 모델을 학습하기 위한 데이터가 충분하지 않습니다.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            df_pred = pd.DataFrame(
                {"실제 신발 사이즈": y_test, "예측 신발 사이즈": y_pred}
            )
            fig_reg = px.scatter(
                df_pred,
                x="실제 신발 사이즈",
                y="예측 신발 사이즈",
                title=f"신발 사이즈 회귀 예측 (MAE={mae:.2f}, R²={r2:.2f})",
            )
            fig_reg.add_shape(
                type="line",
                x0=df_pred["실제 신발 사이즈"].min(),
                y0=df_pred["실제 신발 사이즈"].min(),
                x1=df_pred["실제 신발 사이즈"].max(),
                y1=df_pred["실제 신발 사이즈"].max(),
            )
            fig_reg.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_reg, use_container_width=True)

            # 특성 중요도
            importances = pd.Series(
                model.feature_importances_, index=input_cols
            ).sort_values(ascending=True)

            fig_imp = px.bar(
                importances,
                orientation="h",
                title="특성 중요도 (신발 사이즈 회귀 모델)",
                labels={"value": "중요도", "index": "특성"},
            )
            st.plotly_chart(fig_imp, use_container_width=True)


#######################
# col[2] – 랭킹 & About
with col[2]:
    st.subheader("📈 랭킹 & 설명")

    df_filtered = st.session_state["df_filtered"]
    ml_mode = st.session_state["ml_mode"]

    # 1) TOP 10 랭킹
    st.markdown("#### 🏅 신체 치수 TOP 10")

    numeric_cols = df_filtered.select_dtypes(include="number").columns.tolist()
    ranking_col = st.selectbox("정렬 기준 선택", options=numeric_cols, index=0)

    top10 = df_filtered.nlargest(10, ranking_col)

    fig_rank = px.bar(
        top10,
        x=ranking_col,
        y=top10.index.astype(str),
        orientation="h",
        title=f"{ranking_col} TOP 10",
        labels={ranking_col: ranking_col, "index": "Index"},
    )
    fig_rank.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("---")

    # 2) ML 설명
    if ml_mode == "체형 군집 분석 (K-means)":
        st.markdown("#### 📘 군집 분석 설명")
        st.info(
            """
            - 입력 변수: height, chest(inch), waist, hip, shoes
            - 알고리즘: K-means 클러스터링
            - 활용: 체형/발 사이즈 패턴에 따른 군집(체형 그룹) 탐색
            """
        )
    elif ml_mode == "신발 사이즈 회귀 예측":
        st.markdown("#### 📘 회귀 모델 설명")
        st.info(
            """
            - 입력 변수: height, chest(inch), waist, hip
            - 출력 변수: shoes (신발 사이즈)
            - 모델: RandomForestRegressor
            - 활용: 신체 치수 기반 신발 사이즈 추천/예측
            """
        )

    st.markdown("---")

    # 3) About
    st.markdown("### ℹ️ About")
    st.write(
        """
        이 대시보드는 공군 남성 신체 치수 데이터(키, 가슴둘레, 허리둘레, 엉덩이둘레, 신발 사이즈 등)를 기반으로 합니다.

        **주요 기능**
        - 신체치수 KPI 요약
        - 신발 사이즈 분포 시각화
        - 상관관계 히트맵
        - 체형 군집 분석 (K-means)
        - 신발 사이즈 회귀 예측 (RandomForestRegressor)
        - 신체 치수 TOP 10 랭킹

        **활용 예시**
        - 체형/발 사이즈 기반 군복·군화 사이즈 추천
        - 신체 데이터 기반 예측 모델 리서치
        - 훈련/보급 정책 수립을 위한 통계 분석
        """
    )