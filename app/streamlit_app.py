"""
AI Trading System - Streamlit UI
Original dark theme + McKinsey-style minimal refinement
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import STOCK_UNIVERSE, get_all_stocks, get_layer_color, get_ticker_map

# ── 페이지 설정 ───────────────────────────────────────────
st.set_page_config(
    page_title="J.Insight AI Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (다크 + McKinsey: 고대비, 뚜렷한 타이포) ──────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: #101820; border-radius: 6px; padding: 14px; margin: 5px 0;
    border: 1px solid #2a3a4a;
}
.buy-card  { border-left: 5px solid #00d26a !important; }
.sell-card { border-left: 5px solid #ff4444 !important; }
.hold-card { border-left: 5px solid #ffa500 !important; }
.signal-up   { color: #00d26a; font-weight: 700; font-size: 1.05em; }
.signal-down { color: #ff4444; font-weight: 700; font-size: 1.05em; }
.signal-hold { color: #ffa500; font-weight: 700; font-size: 1.05em; }

/* 메트릭 카드 — 배경과 글씨 대비 극대화 */
div[data-testid="metric-container"] > div {
    background: #101820; border-radius: 6px; padding: 10px;
    border: 1px solid #2a3a4a;
}
[data-testid="stMetricLabel"] { color: #8899aa !important; font-weight: 500; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700; }
[data-testid="stMetricDelta"] { font-weight: 600; }

/* 탭 스타일 */
.stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #2a3a4a; }
.stTabs [data-baseweb="tab"] {
    color: #8899aa; font-weight: 600; font-size: 0.85em; padding: 10px 18px;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important; border-bottom: 3px solid #4fc3f7 !important;
}

/* 사이드바 */
section[data-testid="stSidebar"] { background: #0a1628; }
section[data-testid="stSidebar"] * { color: #c0d0e0 !important; }
section[data-testid="stSidebar"] .stMarkdown h2 { color: #4fc3f7 !important; }

/* 일반 텍스트 대비 */
h1, h2, h3 { color: #ffffff !important; font-weight: 700; }
p { color: #c0d0e0; }
</style>
""", unsafe_allow_html=True)


# ── DB 연결 (캐시) ────────────────────────────────────────
@st.cache_resource
def get_db():
    try:
        from database.db_manager import DBManager
        return DBManager()
    except Exception as e:
        st.error(f"DB 연결 실패: {e}")
        return None


@st.cache_data(ttl=300)
def load_predictions(pred_date):
    db = get_db()
    if db is None:
        return pd.DataFrame()
    try:
        return db.get_latest_predictions(pred_date)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_macro():
    db = get_db()
    if db is None:
        return {}
    try:
        return db.get_macro_latest()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_backtest():
    db = get_db()
    if db is None:
        return pd.DataFrame()
    try:
        return db.get_backtest_summary()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_batch_logs():
    db = get_db()
    if db is None:
        return pd.DataFrame()
    try:
        return db.get_batch_logs(days=3)
    except Exception:
        return pd.DataFrame()


def load_price_chart(ticker: str, days: int = 120):
    db = get_db()
    if db is None:
        return pd.DataFrame()
    try:
        return db.get_stock_prices(ticker, days=days)
    except Exception:
        return pd.DataFrame()


# ── 사이드바 ──────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📊 J.Insight AI Trading")
        st.markdown("**AI 가치사슬 포트폴리오 트레이딩**")
        st.divider()

        selected_date = st.date_input("📅 조회 날짜", value=date.today(),
                                       max_value=date.today())

        horizon_map = {1: "내일 (T+1)", 5: "1주일 (T+5)", 20: "1개월 (T+20)"}
        horizon = st.selectbox("🎯 예측 기간", options=[1, 5, 20],
                               format_func=lambda x: horizon_map[x])

        st.markdown("**레이어 필터**")
        layer_options = list(STOCK_UNIVERSE.keys())
        selected_layers = st.multiselect(
            "가치사슬 레이어", options=layer_options, default=layer_options,
            format_func=lambda x: x.replace("_", " "),
        )

        min_prob = st.slider("최소 상승확률 (%)", 50, 80, 60) / 100

        st.divider()
        if st.button("⚡ 배치 즉시 실행", type="primary", use_container_width=True):
            with st.spinner("배치 실행 중..."):
                try:
                    from batch.daily_batch import run_daily_batch
                    run_daily_batch()
                    st.success("✅ 배치 완료!")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"배치 실패: {e}")

        st.divider()

        # 데이터 기준 시점
        st.markdown("**📋 데이터 기준**")
        db = get_db()
        if db:
            try:
                from sqlalchemy import text
                with db.engine.connect() as conn:
                    r = conn.execute(text("""
                        SELECT p.pred_date, p.created_at,
                               (SELECT MAX(trade_date) FROM stock_prices) as price_date,
                               (SELECT MAX(data_date) FROM macro_data) as macro_date
                        FROM predictions p ORDER BY p.created_at DESC LIMIT 1
                    """)).fetchone()
                    if r:
                        created_kst = r[1].strftime('%m/%d %H:%M') if r[1] else 'N/A'
                        st.caption(f"예측 생성: {created_kst} KST")
                        st.caption(f"주가 기준: {r[2]}")
                        st.caption(f"매크로 기준: {r[3]}")
                        now_hour = datetime.now().hour
                        if r[2] == date.today() and 9 <= now_hour < 16:
                            st.warning("⚠️ 장중 데이터 (종가 미확정)")
                        elif r[2] and r[2] < date.today():
                            days_old = (date.today() - r[2]).days
                            if days_old > 1:
                                st.warning(f"⚠️ 주가 {days_old}일 전 기준")
            except Exception:
                pass

        st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} 기준")

    return selected_date, horizon, selected_layers, min_prob


# ── 거시 지표 패널 ────────────────────────────────────────
def render_macro_panel():
    macro = load_macro()

    col_kr, col_us = st.columns(2)
    with col_kr:
        st.markdown("### 🇰🇷 한국 시장")
        c1, c2, c3 = st.columns(3)
        c1.metric("KOSPI", f"{macro.get('kospi', 0):,.2f}" if macro.get('kospi') else "N/A")
        c2.metric("KOSDAQ", f"{macro.get('kosdaq', 0):,.2f}" if macro.get('kosdaq') else "N/A")
        c3.metric("USD/KRW", f"{macro.get('usd_krw', 0):,.0f}" if macro.get('usd_krw') else "N/A")

    with col_us:
        st.markdown("### 🇺🇸 전일 미국 → 오늘 한국 영향")
        db = get_db()
        us_row = None
        if db:
            try:
                from sqlalchemy import text
                with db.engine.connect() as conn:
                    r = conn.execute(text("""
                        SELECT data_date, sp500, nasdaq, sox, vix, us_10y
                        FROM macro_data WHERE sp500 IS NOT NULL
                        ORDER BY data_date DESC LIMIT 1
                    """)).fetchone()
                    if r:
                        us_row = r
            except Exception:
                pass

        c1, c2, c3, c4, c5 = st.columns(5)
        if us_row:
            c1.metric("S&P500", f"{us_row[1]:,.0f}" if us_row[1] else "N/A")
            c2.metric("NASDAQ", f"{us_row[2]:,.0f}" if us_row[2] else "N/A")
            c3.metric("SOX", f"{us_row[3]:,.0f}" if us_row[3] else "N/A")
            c4.metric("VIX", f"{us_row[4]:.1f}" if us_row[4] else "N/A")
            c5.metric("US10Y", f"{us_row[5]:.2f}%" if us_row[5] else "N/A")
            st.caption(f"기준: {us_row[0]} 미국 장 마감")
        else:
            c1.metric("S&P500", "N/A")
            c2.metric("NASDAQ", "N/A")
            c3.metric("SOX", "N/A")
            c4.metric("VIX", f"{macro.get('vix', 'N/A'):.1f}" if macro.get('vix') else "N/A")
            c5.metric("US10Y", f"{macro.get('us_10y', 'N/A'):.2f}%" if macro.get('us_10y') else "N/A")


# ── 매수 추천 TOP 5 ────────────────────────────────────────
def render_buy_signals(pred_df: pd.DataFrame, horizon: int, min_prob: float):
    st.markdown("### 🟢 오늘의 매수 추천")
    df = pred_df[
        (pred_df["horizon"] == horizon) &
        (pred_df["pred_up_prob"] >= min_prob) &
        (pred_df["pred_label"] == "UP")
    ].sort_values("total_score", ascending=False).head(5)

    if df.empty:
        st.info("조건에 맞는 매수 추천 종목 없음")
        return

    for _, row in df.iterrows():
        layer = row.get("layer", "")
        color = get_layer_color().get(layer, "#4fc3f7")
        name = row.get("name", row.get("ticker", ""))
        up_prob = row.get("pred_up_prob", 0)
        score = row.get("total_score", 0)

        # 5개 모델 시그널
        models = []
        for m, lbl in [("xgb_signal","XGB"), ("lgb_signal","LGB"),
                        ("lstm_signal","LSTM"), ("gru_signal","GRU"),
                        ("transformer_signal","TF")]:
            v = row.get(m, "")
            if v == "UP": models.append(f'<span style="color:#00d26a">{lbl}↑</span>')
            elif v == "DOWN": models.append(f'<span style="color:#ff4444">{lbl}↓</span>')
            elif v: models.append(f'<span style="color:#888">{lbl}—</span>')
        model_str = " ".join(models)

        agree = row.get("agreement_ratio", row.get("supply_score", 0))
        conf = row.get("confidence_mult", 1.0)

        st.markdown(f"""
        <div class="metric-card buy-card">
            <div style="display:flex; justify-content:space-between; align-items:center">
                <div>
                    <span style="color:{color}; font-size:0.75em">● {layer.replace('_',' ')}</span><br>
                    <span style="font-size:1.1em; font-weight:bold; color:white">{name}</span>
                    <span style="color:#aaa; font-size:0.8em"> ({row.get('ticker','')})</span>
                </div>
                <div style="text-align:center">
                    <span class="signal-up">▲ {up_prob:.1%}</span><br>
                    <span style="color:#aaa; font-size:0.7em">합의 {agree:.0%} · 신뢰 {conf:.0%}</span>
                </div>
                <div style="text-align:right">
                    <span style="color:#4fc3f7; font-size:1.2em; font-weight:bold">{score:.3f}</span><br>
                    <span style="font-size:0.65em">{model_str}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── 매도 경고 TOP 5 ────────────────────────────────────────
def render_sell_signals(pred_df: pd.DataFrame, horizon: int):
    st.markdown("### 🔴 매도/관망 경고")
    df = pred_df[
        (pred_df["horizon"] == horizon) &
        (pred_df["pred_label"] == "DOWN")
    ].sort_values("pred_down_prob", ascending=False).head(5)

    if df.empty:
        st.info("매도 경고 종목 없음")
        return

    for _, row in df.iterrows():
        name = row.get("name", row.get("ticker", ""))
        dn_prob = row.get("pred_down_prob", 0)
        layer = row.get("layer", "")

        models = []
        for m, lbl in [("xgb_signal","XGB"), ("lgb_signal","LGB"),
                        ("lstm_signal","LSTM"), ("gru_signal","GRU"),
                        ("transformer_signal","TF")]:
            v = row.get(m, "")
            if v == "DOWN": models.append(f'<span style="color:#ff4444">{lbl}↓</span>')
            elif v == "UP": models.append(f'<span style="color:#00d26a">{lbl}↑</span>')
            elif v: models.append(f'<span style="color:#888">{lbl}—</span>')
        model_str = " ".join(models)

        st.markdown(f"""
        <div class="metric-card sell-card">
            <div style="display:flex; justify-content:space-between; align-items:center">
                <div>
                    <span style="color:#aaa; font-size:0.75em">{layer.replace('_',' ')}</span><br>
                    <span style="font-size:1.1em; font-weight:bold; color:white">{name}</span>
                </div>
                <div style="text-align:center">
                    <span class="signal-down">▼ {dn_prob:.1%}</span><br>
                    <span style="color:#aaa; font-size:0.7em">하락확률</span>
                </div>
                <div style="text-align:right">
                    <span style="font-size:0.65em">{model_str}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── 전체 종목 예측 테이블 ─────────────────────────────────
def render_full_table(pred_df: pd.DataFrame, horizon: int, selected_layers: list):
    st.markdown("### 📋 전체 종목 예측 현황")

    df = pred_df[pred_df["horizon"] == horizon].copy()
    if selected_layers:
        df = df[df["layer"].isin(selected_layers)]

    if df.empty:
        st.info("데이터 없음")
        return

    display_cols = {
        "name": "종목명", "ticker": "티커", "layer": "레이어",
        "pred_up_prob": "상승확률", "pred_hold_prob": "보합확률",
        "pred_down_prob": "하락확률", "pred_label": "예측",
        "xgb_signal": "XGB", "lstm_signal": "LSTM",
        "lgb_signal": "LGB", "gru_signal": "GRU",
        "transformer_signal": "TF",
        "tech_score": "기술점수", "total_score": "종합점수",
        "agreement_ratio": "합의도", "confidence_mult": "신뢰도",
    }
    show_cols = [c for c in display_cols.keys() if c in df.columns]
    df_show = df[show_cols].rename(columns=display_cols)

    def color_label(val):
        if val == "UP":   return "background-color:#1a4d2e; color:#00d26a"
        if val == "DOWN": return "background-color:#4d1a1a; color:#ff4444"
        return "background-color:#3d3a1a; color:#ffa500"

    for col in ["상승확률", "보합확률", "하락확률"]:
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(lambda x: f"{x:.1%}")
    for col in ["기술점수", "종합점수", "합의도", "신뢰도"]:
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    st.dataframe(
        df_show.style.map(color_label, subset=["예측"]),
        use_container_width=True, height=400,
    )

    csv = df_show.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("📥 CSV 다운로드", data=csv,
                       file_name=f"predictions_{date.today()}_h{horizon}.csv",
                       mime="text/csv")


# ── 레이어별 히트맵 ───────────────────────────────────────
def render_layer_heatmap(pred_df: pd.DataFrame, horizon: int):
    st.markdown("### 🗺️ 레이어별 신호 히트맵")
    df = pred_df[pred_df["horizon"] == horizon].copy()
    if df.empty:
        return

    layer_summary = df.groupby("layer").agg(
        avg_up_prob=("pred_up_prob", "mean"),
        up_count=("pred_label", lambda x: (x == "UP").sum()),
        down_count=("pred_label", lambda x: (x == "DOWN").sum()),
        avg_score=("total_score", "mean"),
    ).reset_index()

    fig = px.treemap(
        layer_summary, path=["layer"], values="avg_score",
        color="avg_up_prob", color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.5,
        title=f"레이어별 평균 상승확률 (Horizon: T+{horizon})",
        hover_data={"up_count": True, "down_count": True},
    )
    fig.update_layout(template="plotly_dark", height=350,
                      plot_bgcolor="#0d2137", paper_bgcolor="#0d2137")
    st.plotly_chart(fig, use_container_width=True)


# ── 포트폴리오 스캐터 ──────────────────────────────────────
def render_portfolio_score(pred_df: pd.DataFrame, horizon: int):
    st.markdown("### 🏆 포트폴리오 종합 스코어보드")
    df = pred_df[pred_df["horizon"] == horizon].copy()
    if df.empty:
        return

    fig = px.scatter(
        df, x="pred_up_prob", y="total_score", color="layer",
        size="pred_up_prob", hover_name="name",
        hover_data={"pred_label": True, "xgb_signal": True, "lstm_signal": True},
        title="종목별 상승확률 vs 종합점수",
        labels={"pred_up_prob": "상승확률", "total_score": "종합점수"},
    )
    fig.add_vline(x=0.6, line_dash="dash", line_color="white", opacity=0.5,
                  annotation_text="매수임계치 60%")
    fig.update_layout(template="plotly_dark", height=400,
                      plot_bgcolor="#0d2137", paper_bgcolor="#0d2137")
    st.plotly_chart(fig, use_container_width=True)


# ── 주가 차트 ─────────────────────────────────────────────
def render_price_chart():
    st.markdown("### 📈 종목 주가 차트")
    all_stocks = get_all_stocks()
    stock_options = {s["ticker"]: f"{s['name']} ({s['ticker']})" for s in all_stocks}

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_ticker = st.selectbox("종목 선택", options=list(stock_options.keys()),
                                        format_func=lambda x: stock_options[x])
    with col2:
        days = st.selectbox("기간", options=[60, 120, 250], index=1,
                             format_func=lambda x: f"{x}일")

    price_df = load_price_chart(selected_ticker, days=days)
    if price_df.empty:
        st.info("주가 데이터 없음")
        return

    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])
    price_df = price_df.sort_values("trade_date")
    price_df["sma5"] = price_df["close"].rolling(5).mean()
    price_df["sma20"] = price_df["close"].rolling(20).mean()
    price_df["sma60"] = price_df["close"].rolling(60).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=price_df["trade_date"],
        open=price_df["open"], high=price_df["high"],
        low=price_df["low"], close=price_df["close"],
        name="캔들스틱",
        increasing_line_color="#00d26a", decreasing_line_color="#ff4444",
    ))
    fig.add_trace(go.Scatter(x=price_df["trade_date"], y=price_df["sma5"],
                              name="SMA5", line=dict(color="#ffd700", width=1)))
    fig.add_trace(go.Scatter(x=price_df["trade_date"], y=price_df["sma20"],
                              name="SMA20", line=dict(color="#4fc3f7", width=1.5)))
    fig.add_trace(go.Scatter(x=price_df["trade_date"], y=price_df["sma60"],
                              name="SMA60", line=dict(color="#ff9500", width=1.5)))
    fig.update_layout(template="plotly_dark", height=450,
                      plot_bgcolor="#0d2137", paper_bgcolor="#0d2137",
                      xaxis_rangeslider_visible=False,
                      title=f"{stock_options[selected_ticker]} 주가 차트")
    st.plotly_chart(fig, use_container_width=True)

    # 거래량
    colors = ["#00d26a" if c >= o else "#ff4444"
              for c, o in zip(price_df["close"], price_df["open"])]
    fig_vol = go.Figure(go.Bar(x=price_df["trade_date"], y=price_df["volume"],
                                marker_color=colors, name="거래량"))
    fig_vol.update_layout(template="plotly_dark", height=150,
                          plot_bgcolor="#0d2137", paper_bgcolor="#0d2137",
                          showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_vol, use_container_width=True)


# ── Explainable AI ────────────────────────────────────────
def render_explainable_ai(horizon: int):
    st.markdown("### 🔍 AI 분석 근거 (Explainable AI)")
    all_stocks = get_all_stocks()
    stock_options = {s["ticker"]: f"{s['name']} ({s['ticker']})" for s in all_stocks}

    selected_ticker = st.selectbox("분석할 종목 선택",
        options=list(stock_options.keys()),
        format_func=lambda x: stock_options[x], key="xai_ticker")
    if not selected_ticker:
        return

    db = get_db()
    if db is None:
        return

    price_df = db.get_stock_prices(selected_ticker, days=200)
    if price_df.empty:
        st.info("주가 데이터 없음")
        return

    try:
        from models.ensemble import EnsemblePredictor
        explanation = EnsemblePredictor().explain_prediction(
            selected_ticker, price_df, horizon=horizon)
    except Exception as e:
        st.error(f"분석 실패: {e}")
        return

    if not explanation or not explanation.get("factors"):
        st.info("분석 데이터 부족")
        return

    summary = explanation["summary"]
    buy_cnt = explanation.get("buy_count", 0)
    sell_cnt = explanation.get("sell_count", 0)

    if buy_cnt >= 4: color = "#00d26a"
    elif sell_cnt >= 4: color = "#ff4444"
    elif buy_cnt >= 3: color = "#90EE90"
    elif sell_cnt >= 3: color = "#FF6B6B"
    else: color = "#ffa500"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3a5f, #0d2137);
                border-radius: 8px; padding: 16px; border-left: 5px solid {color};
                margin-bottom: 16px;">
        <span style="font-size: 1.2em; font-weight: bold; color: {color};">
            {stock_options[selected_ticker]}
        </span><br>
        <span style="color: white;">{summary}</span><br>
        <span style="color: #aaa; font-size: 0.85em;">
            매수 {buy_cnt}개 | 매도 {sell_cnt}개 | 중립 {explanation.get('neutral_count', 0)}개 지표
        </span>
    </div>
    """, unsafe_allow_html=True)

    for factor in explanation["factors"]:
        signal = factor["signal"]
        if signal == "BUY":
            sig_color, sig_bg = "#00d26a", "#1a4d2e"
        elif signal == "SELL":
            sig_color, sig_bg = "#ff4444", "#4d1a1a"
        else:
            sig_color, sig_bg = "#ffa500", "#3d3a1a"

        st.markdown(f"""
        <div style="background: {sig_bg}; border-radius: 6px; padding: 10px;
                    margin: 4px 0; border-left: 3px solid {sig_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: white; font-weight: bold;">{factor['category']}</span><br>
                    <span style="color: #ccc; font-size: 0.85em;">{factor['detail']}</span>
                </div>
                <span style="color: {sig_color}; font-weight: bold; font-size: 1.05em;
                             min-width: 50px; text-align: right;">
                    {signal}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if factor["category"] == "AI 모델" and "models" in factor:
            cols = st.columns(len(factor["models"]))
            for idx, (name, info) in enumerate(factor["models"].items()):
                label = info["label"]
                conf = info["confidence"]
                lbl_color = "#00d26a" if label == "UP" else (
                    "#ff4444" if label == "DOWN" else "#ffa500")
                name_display = {"xgb": "XGBoost", "lgb": "LightGBM", "lstm": "LSTM",
                                "gru": "GRU", "transformer": "Transformer"}.get(name, name)
                with cols[idx]:
                    st.markdown(f"""
                    <div style="text-align: center; background: #0d2137;
                                border-radius: 6px; padding: 6px;">
                        <span style="color: #aaa; font-size: 0.7em;">{name_display}</span><br>
                        <span style="color: {lbl_color}; font-weight: bold;">{label}</span><br>
                        <span style="color: #aaa; font-size: 0.7em;">{conf:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)

        if factor["category"] == "이동평균" and "values" in factor:
            vals = factor["values"]
            cols = st.columns(len(vals))
            for idx, (k, v) in enumerate(vals.items()):
                with cols[idx]:
                    st.metric(k, v)


# ── 백테스트 성과 ─────────────────────────────────────────
def render_backtest(selected_layers: list):
    st.markdown("### 📊 백테스트 성과")
    bt_df = load_backtest()
    if bt_df.empty:
        st.info("백테스트 결과 없음")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("평균 연환산", f"{bt_df['annualized_return'].mean():.1%}")
    c2.metric("평균 승률", f"{bt_df['win_rate'].mean():.1%}")
    c3.metric("평균 Sharpe", f"{bt_df['sharpe_ratio'].mean():.2f}")
    c4.metric("초과수익", f"{bt_df['excess_return'].mean():.1%}")

    fig = px.bar(bt_df.sort_values("total_return", ascending=False).head(20),
                 x="ticker", y="total_return", color="total_return",
                 color_continuous_scale="RdYlGn",
                 title="종목별 백테스트 수익률 TOP 20")
    fig.update_layout(template="plotly_dark", height=350,
                      plot_bgcolor="#0d2137", paper_bgcolor="#0d2137")
    st.plotly_chart(fig, use_container_width=True)


# ── 배치 로그 ─────────────────────────────────────────────
def render_batch_log():
    st.markdown("### 📋 배치 실행 로그")
    log_df = load_batch_logs()
    if log_df.empty:
        st.info("로그 없음")
        return
    status_map = {"SUCCESS": "✅", "FAILED": "❌", "PARTIAL": "⚠️"}
    log_df["상태"] = log_df["status"].map(status_map).fillna("❓")
    display = log_df[["run_date", "run_time", "step", "상태", "message", "duration_sec"]].copy()
    display.columns = ["날짜", "시간", "단계", "상태", "메시지", "소요(초)"]
    st.dataframe(display, use_container_width=True, height=200)


# ── 히스토리컬 백테스트 엔진 ─────────────────────────────
def run_historical_backtest(db, start_date, end_date, tickers=None):
    """과거 날짜 기준으로 모델 예측 실행 → 실제 결과와 비교"""
    from data.preprocessor import FeatureBuilder
    from models.ensemble import EnsemblePredictor
    from config.settings import get_all_stocks
    import warnings
    warnings.filterwarnings("ignore")

    predictor = EnsemblePredictor()

    if tickers is None:
        tickers = [s["ticker"] for s in get_all_stocks()]

    # 기간 내 거래일 목록
    from sqlalchemy import text
    with db.engine.connect() as conn:
        trade_dates = pd.read_sql(text("""
            SELECT DISTINCT trade_date FROM stock_prices
            WHERE ticker = '005930.KS'
              AND trade_date >= :start AND trade_date <= :end
            ORDER BY trade_date
        """), conn, params={"start": start_date, "end": end_date})

    if trade_dates.empty:
        return pd.DataFrame()

    dates_list = trade_dates["trade_date"].tolist()
    results = []

    for sim_date in dates_list:
        for ticker in tickers:
            try:
                price_df = db.get_stock_prices(ticker, days=250)
                if price_df.empty or len(price_df) < 50:
                    continue

                price_df["trade_date"] = pd.to_datetime(price_df["trade_date"]).dt.date
                past_df = price_df[price_df["trade_date"] <= sim_date]
                future_df = price_df[price_df["trade_date"] > sim_date].head(1)

                if past_df.empty or future_df.empty or len(past_df) < 30:
                    continue

                preds = predictor.predict_single(ticker, past_df)
                if not preds:
                    continue

                p = preds[0]  # H1
                base_close = float(past_df["close"].iloc[-1])
                actual_close = float(future_df["close"].iloc[0])
                actual_return = (actual_close - base_close) / base_close * 100

                actual_label = "UP" if actual_return > 2 else ("DOWN" if actual_return < -2 else "HOLD")

                results.append({
                    "sim_date": sim_date,
                    "next_date": future_df["trade_date"].iloc[0],
                    "ticker": ticker,
                    "name": p.get("name", ticker),
                    "pred_label": p["pred_label"],
                    "pred_up_prob": p["pred_up_prob"],
                    "pred_down_prob": p["pred_down_prob"],
                    "base_close": base_close,
                    "actual_close": actual_close,
                    "return_pct": round(actual_return, 2),
                    "actual_label": actual_label,
                    "hit": p["pred_label"] == actual_label,
                })
            except Exception:
                continue

    return pd.DataFrame(results)


# ── 예측 적중률 (날짜별 + 개별기업 조회) ─────────────────
def render_prediction_accuracy():
    st.markdown("### 🎯 예측 적중률 백테스트")

    db = get_db()
    if db is None:
        return

    try:
        from sqlalchemy import text

        # ── 조회 UI ──────────────────────────────────
        col_mode, col_period, col_stock = st.columns([1, 2, 2])
        with col_mode:
            view_mode = st.radio("조회 모드", ["날짜별 전체", "개별 종목"], horizontal=True, key="bt_mode")
        with col_period:
            period_opt = st.selectbox("기간 선택", [
                "최근 1주일", "최근 2주일", "최근 1개월", "직접 선택"
            ], key="bt_period")
            if period_opt == "최근 1주일":
                start_dt = date.today() - timedelta(days=7)
                end_dt = date.today() - timedelta(days=1)
            elif period_opt == "최근 2주일":
                start_dt = date.today() - timedelta(days=14)
                end_dt = date.today() - timedelta(days=1)
            elif period_opt == "최근 1개월":
                start_dt = date.today() - timedelta(days=30)
                end_dt = date.today() - timedelta(days=1)
            else:
                col_s, col_e = st.columns(2)
                with col_s:
                    start_dt = st.date_input("시작일", value=date.today() - timedelta(days=7), key="bt_start")
                with col_e:
                    end_dt = st.date_input("종료일", value=date.today() - timedelta(days=1), key="bt_end")

        selected_ticker = None
        all_stocks = get_all_stocks()
        stock_options = {s["ticker"]: f"{s['name']} ({s['ticker']})" for s in all_stocks}

        if view_mode == "개별 종목":
            with col_stock:
                selected_ticker = st.selectbox("종목 선택", options=list(stock_options.keys()),
                                               format_func=lambda x: stock_options[x], key="acc_stock")

        # ── 백테스트 실행 ──────────────────────────────
        bt_key = f"bt_{start_dt}_{end_dt}_{selected_ticker}"
        if st.button("🚀 백테스트 실행", type="primary", key="run_bt"):
            tickers_to_test = [selected_ticker] if selected_ticker else None
            with st.spinner(f"백테스트 실행 중... ({start_dt} ~ {end_dt})"):
                accuracy_df = run_historical_backtest(db, start_dt, end_dt, tickers=tickers_to_test)
                st.session_state["bt_result"] = accuracy_df
                st.session_state["bt_key"] = bt_key

        accuracy_df = st.session_state.get("bt_result", pd.DataFrame())

        if accuracy_df.empty:
            st.info("🚀 버튼을 눌러 백테스트를 실행하세요. 선택한 기간에 대해 AI 모델의 예측 적중률을 계산합니다.")
            return

        # ── 전체 요약 메트릭 ─────────────────────────
        total = len(accuracy_df)
        correct = accuracy_df["hit"].sum()
        acc = correct / total if total > 0 else 0
        up_preds = accuracy_df[accuracy_df["pred_label"] == "UP"]
        down_preds = accuracy_df[accuracy_df["pred_label"] == "DOWN"]
        up_correct = up_preds["hit"].sum() if len(up_preds) > 0 else 0
        down_correct = down_preds["hit"].sum() if len(down_preds) > 0 else 0
        avg_return = accuracy_df["return_pct"].mean()
        up_return = up_preds["return_pct"].mean() if len(up_preds) > 0 else 0

        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("총 예측", f"{total}건")
        c2.metric("전체 적중률", f"{acc:.1%}")
        c3.metric("UP 적중", f"{up_correct}/{len(up_preds)}" if len(up_preds) > 0 else "0/0")
        c4.metric("DOWN 적중", f"{down_correct}/{len(down_preds)}" if len(down_preds) > 0 else "0/0")
        c5.metric("UP시그널 평균수익", f"{up_return:+.2f}%")
        st.divider()

        # ── 날짜별 전체 보기 ─────────────────────────
        if view_mode == "날짜별 전체":
            dates = sorted(accuracy_df["sim_date"].unique(), reverse=True)

            # 일별 적중률 카드
            st.markdown("#### 일별 적중률")
            for sim_date in dates:
                day_df = accuracy_df[accuracy_df["sim_date"] == sim_date]
                d_total = len(day_df)
                d_correct = day_df["hit"].sum()
                d_acc = d_correct / d_total if d_total > 0 else 0
                d_up = day_df[day_df["pred_label"] == "UP"]
                d_up_hit = d_up["hit"].sum() if len(d_up) > 0 else 0
                d_dn = day_df[day_df["pred_label"] == "DOWN"]
                d_dn_hit = d_dn["hit"].sum() if len(d_dn) > 0 else 0
                d_avg_ret = day_df["return_pct"].mean()

                acc_color = "#00d26a" if d_acc >= 0.5 else "#ffa500" if d_acc >= 0.35 else "#ff4444"

                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {acc_color};">
                    <div style="display:flex; justify-content:space-between; align-items:center">
                        <div>
                            <span style="color:white; font-weight:bold">{sim_date}</span>
                            <span style="color:#aaa; font-size:0.8em"> ({d_total}종목)</span>
                        </div>
                        <div style="text-align:center">
                            <span style="color:{acc_color}; font-weight:bold; font-size:1.2em">{d_acc:.1%}</span><br>
                            <span style="color:#aaa; font-size:0.7em">적중률</span>
                        </div>
                        <div style="text-align:center; font-size:0.85em">
                            <span style="color:#aaa">평균수익</span><br>
                            <span style="color:{'#00d26a' if d_avg_ret > 0 else '#ff4444'}">{d_avg_ret:+.2f}%</span>
                        </div>
                        <div style="text-align:right; font-size:0.85em">
                            <span style="color:#00d26a">UP {d_up_hit}/{len(d_up)}</span> ·
                            <span style="color:#ff4444">DN {d_dn_hit}/{len(d_dn)}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 선택 날짜 상세
            if dates:
                st.markdown("#### 상세 결과")
                sel_date = st.selectbox("날짜 선택", dates, key="bt_detail_date")
                detail_df = accuracy_df[accuracy_df["sim_date"] == sel_date].copy()
                detail_df["적중"] = detail_df["hit"].apply(lambda x: "✅" if x else "❌")
                detail_df["수익률"] = detail_df["return_pct"].apply(lambda x: f"{x:+.2f}%")

                show_df = detail_df[["name", "pred_label", "actual_label", "수익률", "적중",
                                     "pred_up_prob", "pred_down_prob", "base_close", "actual_close"]].rename(columns={
                    "name": "종목", "pred_label": "예측", "actual_label": "실제",
                    "pred_up_prob": "UP확률", "pred_down_prob": "DN확률",
                    "base_close": "기준가", "actual_close": "결과가"
                })
                for col in ["UP확률", "DN확률"]:
                    show_df[col] = show_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
                for col in ["기준가", "결과가"]:
                    show_df[col] = show_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
                st.dataframe(show_df, use_container_width=True, height=400)

        # ── 개별 종목 보기 ───────────────────────────
        else:
            stock_name = stock_options.get(selected_ticker, selected_ticker)
            st.markdown(f"#### {stock_name} — 백테스트 이력")

            # 시계열 차트
            chart_df = accuracy_df[["sim_date", "return_pct", "hit", "pred_label"]].copy()
            chart_df["sim_date"] = pd.to_datetime(chart_df["sim_date"])
            chart_df = chart_df.sort_values("sim_date")
            chart_df["누적수익"] = (1 + chart_df["return_pct"] / 100).cumprod() - 1

            fig = go.Figure()
            colors = ["#00d26a" if hit else "#ff4444" for hit in chart_df["hit"]]
            fig.add_trace(go.Bar(
                x=chart_df["sim_date"], y=chart_df["return_pct"],
                marker_color=colors, name="일별 수익률 (%)"
            ))
            fig.add_trace(go.Scatter(
                x=chart_df["sim_date"], y=chart_df["누적수익"] * 100,
                name="누적수익률 (%)", line=dict(color="#4fc3f7", width=2), yaxis="y2"
            ))
            fig.update_layout(
                template="plotly_dark", height=350,
                plot_bgcolor="#0d2137", paper_bgcolor="#0d2137",
                yaxis=dict(title="일별 수익률 (%)", side="left"),
                yaxis2=dict(title="누적수익률 (%)", side="right", overlaying="y"),
                legend=dict(orientation="h", y=-0.15),
                title=f"{stock_name} AI 예측 기반 수익률 (초록=적중, 빨강=실패)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # 상세 이력
            history_df = accuracy_df.copy()
            history_df["적중"] = history_df["hit"].apply(lambda x: "✅" if x else "❌")
            history_df["수익률"] = history_df["return_pct"].apply(lambda x: f"{x:+.2f}%")
            show_df = history_df[["sim_date", "pred_label", "actual_label", "수익률",
                                  "적중", "base_close", "actual_close"]].rename(columns={
                "sim_date": "기준일", "pred_label": "예측", "actual_label": "실제",
                "base_close": "기준종가", "actual_close": "결과종가"
            })
            for col in ["기준종가", "결과종가"]:
                show_df[col] = show_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
            st.dataframe(show_df, use_container_width=True, height=300)

    except Exception as e:
        st.error(f"적중률 계산 오류: {e}")


# ── 메인 ─────────────────────────────────────────────────
def main():
    selected_date, horizon, selected_layers, min_prob = render_sidebar()

    st.markdown("""
    <div style="color:#4fc3f7; font-size:1.5em; font-weight:bold">
        📈 J.Insight AI Trading Dashboard
    </div>
    <p style="color:#aaa">코스피·코스닥 AI 가치사슬 113개 종목 | XGBoost+LightGBM 앙상블 예측</p>
    """, unsafe_allow_html=True)
    st.divider()

    render_macro_panel()
    st.divider()

    pred_df = load_predictions(selected_date)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 매매 시그널", "🔍 AI 분석 근거",
        "📋 전체 종목", "🎯 예측 적중률",
        "📊 백테스트", "📈 주가 차트", "⚙️ 시스템",
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            render_buy_signals(pred_df, horizon, min_prob)
        with col2:
            render_sell_signals(pred_df, horizon)
        st.divider()
        render_layer_heatmap(pred_df, horizon)
        render_portfolio_score(pred_df, horizon)

    with tab2:
        render_explainable_ai(horizon)

    with tab3:
        render_full_table(pred_df, horizon, selected_layers)

    with tab4:
        render_prediction_accuracy()

    with tab5:
        render_backtest(selected_layers)

    with tab6:
        render_price_chart()

    with tab7:
        st.markdown("### ⚙️ 시스템 상태")
        db = get_db()
        if db:
            st.success("✅ PostgreSQL 연결 정상")
        else:
            st.error("❌ PostgreSQL 연결 실패")

        from models.xgboost_model import XGBStockModel
        st.markdown("**모델 학습 상태**")
        for h in [1, 5, 20]:
            m = XGBStockModel(horizon=h)
            status = "✅ 학습됨" if m.is_trained() else "❌ 미학습"
            st.write(f"XGBoost T+{h}: {status}")

        # 적응형 가중치
        try:
            from models.adaptive_weights import load_weights
            w = load_weights()
            st.markdown("**앙상블 가중치**")
            st.json(w)
        except Exception:
            pass

        render_batch_log()

        with st.expander("📌 시스템 설정"):
            from config.settings import BATCH_CONFIG, MODEL_CONFIG, TRADING_RULES
            st.json({
                "배치시간": f"{BATCH_CONFIG['hour']:02d}:{BATCH_CONFIG['minute']:02d} KST",
                "모델설정": MODEL_CONFIG,
                "매매규칙": TRADING_RULES,
            })


if __name__ == "__main__":
    main()
