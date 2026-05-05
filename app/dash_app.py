"""
AI Trading System - Professional Dashboard
Bloomberg/Refinitiv-inspired: dark neutral + minimal accent
Sidebar nav with confirm → fixed viewport → scroll within
"""
import dash
from dash import dcc, html, dash_table, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import date, datetime

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import STOCK_UNIVERSE, get_all_stocks, get_layer_color, get_ticker_map
from config.settings import MODEL_CONFIG, TRADING_RULES

# ── DB ──────────────────────────────────────────────────
_db = None
def get_db():
    global _db
    if _db is None:
        from database.db_manager import DBManager
        _db = DBManager()
    return _db

def load_predictions(d=None):
    if d is None: d = date.today()
    try: return get_db().get_latest_predictions(d)
    except: return pd.DataFrame()

def load_macro():
    try: return get_db().get_macro_latest()
    except: return {}

def load_valuations():
    try:
        from models.valuation import ValuationCalculator
        return ValuationCalculator().get_all_valuations(get_db())
    except: return pd.DataFrame()

def load_price(ticker, days=120):
    try: return get_db().get_stock_prices(ticker, days=days)
    except: return pd.DataFrame()

# ── Color system (Bloomberg-inspired) ───────────────────
# Dark neutral base, 2 semantic colors, 1 accent
BG      = "#0E1117"   # page background
PANEL   = "#161B22"   # card/panel background
BORDER  = "#21262D"   # borders
TXT     = "#C9D1D9"   # primary text
TXT2    = "#8B949E"   # secondary text
WHITE   = "#F0F6FC"   # emphasis text
POS     = "#3FB950"   # positive / buy (muted green)
NEG     = "#F85149"   # negative / sell (muted red)
WARN    = "#D29922"   # hold / caution
ACCENT  = "#58A6FF"   # interactive / links
GRID    = "#21262D"   # chart gridlines

CHART_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
    font=dict(color=TXT, size=11, family="SF Mono, Consolas, monospace"),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=36, r=8, t=20, b=24),
    colorway=[ACCENT, POS, NEG, WARN, "#A371F7", "#79C0FF", "#FFA657"],
)

all_stocks = get_all_stocks()
stock_opts = [{"label": f"{s['name']} ({s['code']})", "value": s["ticker"]} for s in all_stocks]

PAGES = ["종합현황 (Summary)", "시장환경 (Market)", "종목스크리닝 (Screening)", "종목분석 (Deep Dive)", "투자실행 (Action)"]

# ── App ─────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                suppress_callback_exceptions=True, title="J.Insight AI Trading")

app.index_string = '''<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  body { margin:0; overflow:hidden; background:''' + BG + '''; }
  ::-webkit-scrollbar { width:6px; }
  ::-webkit-scrollbar-track { background:''' + BG + '''; }
  ::-webkit-scrollbar-thumb { background:''' + BORDER + '''; border-radius:3px; }
  .nav-pills .nav-link { color:''' + TXT2 + '''; font-size:0.82rem; padding:8px 12px;
      border-radius:4px; margin:1px 0; }
  .nav-pills .nav-link.active { background:''' + BORDER + '''; color:''' + WHITE + '''; font-weight:600; }
  .nav-pills .nav-link:hover { background:''' + BORDER + '''; color:''' + WHITE + '''; }
</style></head><body>{%app_entry%}{%config%}{%scripts%}{%renderer%}</body></html>'''

# ── Sidebar (fixed left) ───────────────────────────────
sidebar = html.Div([
    # Brand
    html.Div([
        html.Div("J.Insight", style={"fontSize": "1.1rem", "fontWeight": 700,
                                      "color": WHITE, "letterSpacing": "-0.5px"}),
        html.Div("AI Trading System", style={"fontSize": "0.65rem", "color": TXT2,
                                              "letterSpacing": "1px", "textTransform": "uppercase"}),
    ], style={"padding": "14px 14px 10px", "borderBottom": f"1px solid {BORDER}"}),

    # Navigation
    dbc.Nav([dbc.NavLink(p, id=f"nav-{i}", n_clicks=0,
                          style={"cursor": "pointer"})
             for i, p in enumerate(PAGES)],
            vertical=True, pills=True, style={"padding": "8px 6px"}),

    html.Hr(style={"borderColor": BORDER, "margin": "4px 12px"}),

    # Controls
    html.Div([
        html.Div("CONTROLS", style={"fontSize": "0.6rem", "color": TXT2, "letterSpacing": "2px",
                                     "marginBottom": "6px"}),
        dcc.DatePickerSingle(id="date-picker", date=date.today(),
                             max_date_allowed=date.today(), display_format="YYYY-MM-DD",
                             style={"width": "100%", "marginBottom": "6px"}),
        dcc.Dropdown(id="horizon-select", value=1, clearable=False,
                     options=[{"label":"D+1","value":1},{"label":"D+5","value":5},{"label":"D+20","value":20}],
                     style={"fontSize": "0.8rem", "marginBottom": "6px"}),
        dcc.Dropdown(id="layer-filter", multi=True, placeholder="All layers",
                     options=[{"label":k.replace("_"," "),"value":k} for k in STOCK_UNIVERSE],
                     value=list(STOCK_UNIVERSE.keys()),
                     style={"fontSize": "0.75rem", "marginBottom": "6px"}),
        html.Div([
            html.Span("Min UP% ", style={"fontSize": "0.7rem", "color": TXT2}),
            html.Span(id="prob-display", style={"fontSize": "0.7rem", "color": ACCENT}),
        ]),
        dcc.Slider(id="min-prob", min=30, max=80, step=5, value=50,
                   marks=None, tooltip={"placement": "bottom"}),
    ], style={"padding": "0 12px"}),

    html.Hr(style={"borderColor": BORDER, "margin": "6px 12px"}),
    html.Div([
        dbc.Button("Refresh Data", id="btn-refresh", size="sm", outline=True,
                   color="secondary", className="w-100", style={"fontSize": "0.75rem"}),
        html.Div(id="btn-status", className="mt-1"),
    ], style={"padding": "0 12px"}),

    html.Div(id="nav-clock", style={"position": "absolute", "bottom": "8px",
                                     "padding": "0 12px", "fontSize": "0.65rem", "color": TXT2}),

], style={
    "position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "168px",
    "backgroundColor": PANEL, "borderRight": f"1px solid {BORDER}",
    "overflowY": "auto", "zIndex": 1000,
})

# ── Confirm modal ───────────────────────────────────────
confirm_modal = dbc.Modal([
    dbc.ModalHeader("메뉴 이동 (Navigate)", style={"backgroundColor": PANEL, "color": WHITE,
                                        "borderBottom": f"1px solid {BORDER}", "fontSize": "0.9rem"}),
    dbc.ModalBody(id="confirm-body", style={"backgroundColor": PANEL, "color": TXT}),
    dbc.ModalFooter([
        dbc.Button("취소 (Cancel)", id="confirm-cancel", color="secondary", size="sm"),
        dbc.Button("이동 (Go)", id="confirm-go", color="primary", size="sm"),
    ], style={"backgroundColor": PANEL, "borderTop": f"1px solid {BORDER}"}),
], id="confirm-modal", is_open=False, centered=True)

# ── Main content (fixed viewport, scroll inside) ───────
content = html.Div(id="page-content", style={
    "marginLeft": "168px", "padding": "8px 12px",
    "backgroundColor": BG, "height": "100vh", "overflowY": "auto",
})

app.layout = html.Div([
    dcc.Store(id="pending-page", data=0),
    dcc.Store(id="current-page", data=0),
    dcc.Interval(id="clock-tick", interval=60000),
    confirm_modal, sidebar, content,
])


# ── Callbacks ───────────────────────────────────────────
@callback(Output("nav-clock", "children"), Input("clock-tick", "n_intervals"))
def tick(n): return datetime.now().strftime("%Y-%m-%d %H:%M")

@callback(Output("prob-display", "children"), Input("min-prob", "value"))
def show_prob(v): return f"{v}%"

@callback(Output("btn-status", "children"), Input("btn-refresh", "n_clicks"),
          prevent_initial_call=True)
def refresh(n):
    global _db; _db = None
    return html.Small("Refreshed", style={"color": POS})

# Nav click → open confirm
@callback(
    Output("confirm-modal", "is_open"),
    Output("confirm-body", "children"),
    Output("pending-page", "data"),
    [Input(f"nav-{i}", "n_clicks") for i in range(5)],
    Input("confirm-cancel", "n_clicks"),
    Input("confirm-go", "n_clicks"),
    State("current-page", "data"),
    State("pending-page", "data"),
    State("confirm-modal", "is_open"),
    prevent_initial_call=True,
)
def handle_nav(*args):
    nav_clicks = args[:5]
    cancel, go_click = args[5], args[6]
    current, pending, is_open = args[7], args[8], args[9]

    triggered = ctx.triggered_id
    if triggered == "confirm-cancel":
        return False, "", pending
    if triggered == "confirm-go":
        return False, "", pending

    for i in range(5):
        if triggered == f"nav-{i}" and i != current:
            return True, f"{PAGES[i]} 페이지로 이동하시겠습니까?", i

    return False, "", current

# Confirm → update page
@callback(
    Output("current-page", "data"),
    Input("confirm-go", "n_clicks"),
    State("pending-page", "data"),
    prevent_initial_call=True,
)
def confirm_nav(n, pending):
    return pending

# Highlight active nav
for i in range(5):
    @callback(Output(f"nav-{i}", "active"), Input("current-page", "data"))
    def update_active(current, idx=i):
        return current == idx

# Render page
@callback(
    Output("page-content", "children"),
    Input("current-page", "data"),
    Input("date-picker", "date"),
    Input("horizon-select", "value"),
    Input("layer-filter", "value"),
    Input("min-prob", "value"),
)
def render(page_idx, sel_date, horizon, layers, min_prob_pct):
    if isinstance(sel_date, str):
        sel_date = datetime.strptime(sel_date[:10], "%Y-%m-%d").date()
    min_prob = (min_prob_pct or 50) / 100

    pred = load_predictions(sel_date)
    val_df = load_valuations()
    macro = load_macro()

    df = pd.DataFrame()
    if not pred.empty:
        df = pred[pred["horizon"] == horizon].copy()
        if not val_df.empty:
            vc = [c for c in ["ticker","fair_value","upside_pct","grade",
                  "band_1d_upper","band_1d_lower","band_5d_upper","band_5d_lower",
                  "band_20d_upper","band_20d_lower"] if c in val_df.columns]
            df = df.merge(val_df[vc], on="ticker", how="left")
        if layers:
            df = df[df["layer"].isin(layers)]

    renderers = [pg_summary, pg_market, pg_screening, pg_deep, pg_action]
    return renderers[page_idx](df, macro, horizon, min_prob)


# ── Helpers ─────────────────────────────────────────────
def _card(children, title=None):
    header = html.Div(title, style={"fontSize": "0.72rem", "fontWeight": 600,
        "color": TXT2, "marginBottom": "4px", "textTransform": "uppercase",
        "letterSpacing": "0.5px"}) if title else html.Div()
    return html.Div([header, children], style={
        "backgroundColor": PANEL, "borderRadius": "4px", "padding": "10px",
        "border": f"1px solid {BORDER}", "marginBottom": "6px"})

def _kpi(label, value, sub="", color=None):
    return html.Div([
        html.Div(label, style={"fontSize": "0.62rem", "color": TXT2, "textTransform": "uppercase",
                                "letterSpacing": "0.5px", "marginBottom": "1px"}),
        html.Div(value, style={"fontSize": "1.15rem", "fontWeight": 700,
                                "color": color or WHITE, "lineHeight": "1.2"}),
        html.Div(sub, style={"fontSize": "0.6rem", "color": TXT2}) if sub else html.Div(),
    ], style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}",
              "borderRadius": "4px", "padding": "7px 8px", "textAlign": "center"})

def _fig_layout(fig, h=280):
    fig.update_layout(**CHART_TEMPLATE, height=h, showlegend=False)
    return fig


# ══════════════════════════════════════════════════════════
#  PAGE 1: SUMMARY
# ══════════════════════════════════════════════════════════
def pg_summary(df, macro, horizon, min_prob):
    if df.empty:
        return _card(html.P("No data. Run batch.", style={"color": WARN}))

    up = len(df[df.pred_label=="UP"])
    dn = len(df[df.pred_label=="DOWN"])
    hd = len(df[df.pred_label=="HOLD"])
    tot = len(df)
    avg_s = df.total_score.mean()
    avg_a = float(df["agreement_ratio"].mean()) if "agreement_ratio" in df.columns and df["agreement_ratio"].notna().any() else 0

    if up > dn*1.5: sent, sc = "BULLISH", POS
    elif dn > up*1.5: sent, sc = "BEARISH", NEG
    else: sent, sc = "MIXED", WARN

    vix = macro.get("vix")
    vix_s = f"{float(vix):.1f}" if vix and not (isinstance(vix,float) and np.isnan(vix)) else "—"

    hl = {1:"D+1",5:"D+5",20:"D+20"}.get(horizon,"")

    # Basis
    basis = ""
    try:
        from sqlalchemy import text
        with get_db().engine.connect() as conn:
            r = conn.execute(text("SELECT created_at,(SELECT MAX(trade_date) FROM stock_prices) FROM predictions ORDER BY created_at DESC LIMIT 1")).fetchone()
            if r: basis = f"{r[0].strftime('%m/%d %H:%M')} KST · Price: {r[1]}"
    except: pass

    # Donut
    fig_pie = go.Figure(go.Pie(values=[up,hd,dn], labels=["UP","HOLD","DOWN"],
        marker=dict(colors=[POS,WARN,NEG]), hole=0.7, textinfo="label+value",
        textfont=dict(color=TXT, size=11), sort=False))
    _fig_layout(fig_pie, 240)

    # Layer bar
    lsm = df.groupby("layer").agg(a=("total_score","mean")).reset_index()
    fig_bar = go.Figure(go.Bar(x=lsm["layer"], y=lsm["a"],
        marker=dict(color=[POS if v>0.52 else (NEG if v<0.48 else WARN) for v in lsm["a"]]),
        text=[f"{v:.3f}" for v in lsm["a"]], textposition="outside", textfont=dict(color=TXT,size=10)))
    fig_bar.add_hline(y=0.5, line_dash="dot", line_color=TXT2, line_width=1)
    _fig_layout(fig_bar, 240)
    fig_bar.update_layout(yaxis=dict(range=[0.35,0.65]))

    # Top/Worst
    def mover(sub, d):
        items = []
        for _, r in sub.iterrows():
            p = r.get("pred_up_prob",0) if d=="up" else r.get("pred_down_prob",0)
            c = POS if d=="up" else NEG
            items.append(html.Div([
                html.Span(r.get("name",""), style={"color": WHITE, "fontWeight": 600, "fontSize": "0.82rem"}),
                html.Span(f"  {p:.1%}", style={"color": c, "fontWeight": 700, "marginLeft": "6px"}),
                html.Span(f"  {r.get('total_score',0):.3f}", style={"color": TXT2, "fontSize": "0.75rem", "marginLeft": "6px"}),
            ], style={"padding": "5px 0", "borderBottom": f"1px solid {BORDER}"}))
        return items or [html.Span("—", style={"color": TXT2})]

    return html.Div([
        html.Div([
            html.Span("종합현황 (Executive Summary)", style={"fontSize": "0.88rem", "fontWeight": 700,
                "color": WHITE, "letterSpacing": "0.3px"}),
            html.Span(f"  {hl}", style={"color": ACCENT, "fontSize": "0.78rem", "marginLeft": "6px"}),
            html.Div(basis, style={"fontSize": "0.62rem", "color": TXT2, "marginTop": "1px"}),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(_kpi("시장심리 (Sentiment)", sent, f"{tot}종목", sc), md=2),
            dbc.Col(_kpi("상승 (UP)", str(up), f"{up/tot*100:.0f}%", POS), md=1),
            dbc.Col(_kpi("보합 (HOLD)", str(hd), "", WARN), md=1),
            dbc.Col(_kpi("하락 (DOWN)", str(dn), f"{dn/tot*100:.0f}%", NEG), md=1),
            dbc.Col(_kpi("공포지수 (VIX)", vix_s, "", NEG if vix and float(vix)>25 else POS), md=1),
            dbc.Col(_kpi("모델합의 (Consensus)", f"{avg_a:.0%}", "", ACCENT), md=2),
            dbc.Col(_kpi("평균점수 (Avg Score)", f"{avg_s:.3f}", ""), md=2),
        ], className="g-1 mb-2"),
        dbc.Row([
            dbc.Col(_card(dcc.Graph(figure=fig_pie, config={"displayModeBar":False}), "신호분포 (Distribution)"), md=3),
            dbc.Col(_card(dcc.Graph(figure=fig_bar, config={"displayModeBar":False}), "레이어별 점수 (Layer Score)"), md=5),
            dbc.Col([
                _card(html.Div(mover(df.nlargest(4,"total_score"),"up")), "기회 종목 (Opportunities)"),
                _card(html.Div(mover(df.nsmallest(4,"total_score"),"down")), "위험 종목 (Risks)"),
            ], md=4),
        ], className="g-1"),
    ])


# ══════════════════════════════════════════════════════════
#  PAGE 2: MARKET
# ══════════════════════════════════════════════════════════
def pg_market(df, macro, horizon, min_prob):
    kr = [("KOSPI", macro.get("kospi")), ("KOSDAQ", macro.get("kosdaq")),
          ("USD/KRW", macro.get("usd_krw"))]
    kr_cols = [dbc.Col(_kpi(n, f"{v:,.0f}" if v and not(isinstance(v,float) and np.isnan(v)) else "—"), md=2) for n,v in kr]

    us_cols = []
    us_date = ""
    try:
        from sqlalchemy import text
        with get_db().engine.connect() as conn:
            r = conn.execute(text("SELECT data_date,sp500,nasdaq,sox,vix,us_10y FROM macro_data WHERE sp500 IS NOT NULL ORDER BY data_date DESC LIMIT 1")).fetchone()
            if r:
                us_date = str(r[0])
                for n,v,fmt in [("S&P500",r[1],"{:,.0f}"),("NASDAQ",r[2],"{:,.0f}"),("SOX",r[3],"{:,.0f}"),
                                ("VIX",r[4],"{:.1f}"),("US 10Y",r[5],"{:.2f}%")]:
                    d = fmt.format(v) if v and not(isinstance(v,float) and np.isnan(v)) else "—"
                    c = NEG if "VIX"==n and v and float(v)>25 else None
                    us_cols.append(dbc.Col(_kpi(n, d, color=c), md=2))
    except: pass

    # Treemap
    fig = go.Figure()
    if not df.empty:
        lsm = df.groupby("layer").agg(avg_up=("pred_up_prob","mean"),avg_s=("total_score","mean"),n=("ticker","count")).reset_index()
        fig = px.treemap(lsm, path=["layer"], values="avg_s", color="avg_up",
                         color_continuous_scale=[[0,NEG],[0.5,WARN],[1,POS]],
                         color_continuous_midpoint=0.5)
    _fig_layout(fig, 340)

    return html.Div([
        html.Div("시장환경 (Market Context)", style={"fontSize":"0.88rem","fontWeight":700,"color":WHITE,"marginBottom":"8px"}),
        dbc.Row([
            dbc.Col(_card(dbc.Row(kr_cols, className="g-1"), "한국시장 (Korea)"), md=6),
            dbc.Col(_card(dbc.Row(us_cols, className="g-1") if us_cols else html.Span("데이터 없음",style={"color":TXT2}),
                  f"전일 미국시장 (US {us_date})" if us_date else "미국시장 (US)"), md=6),
        ], className="g-1 mb-1"),
        _card(dcc.Graph(figure=fig, config={"displayModeBar":False}), "레이어 신호맵 (Layer Signal Map)"),
    ])


# ══════════════════════════════════════════════════════════
#  PAGE 3: SCREENING
# ══════════════════════════════════════════════════════════
def pg_screening(df, macro, horizon, min_prob):
    if df.empty:
        return _card(html.P("No data", style={"color":WARN}))

    show = df.copy()
    show["sig"] = show["pred_label"].map({"UP":"▲ BUY","DOWN":"▼ SELL","HOLD":"— HOLD"})
    show["up%"] = show["pred_up_prob"].apply(lambda x: f"{x:.1%}")
    show["dn%"] = show["pred_down_prob"].apply(lambda x: f"{x:.1%}")
    show["sc"] = show["total_score"].apply(lambda x: f"{x:.3f}")
    for m in ["xgb_signal","lstm_signal","lgb_signal","gru_signal","transformer_signal"]:
        if m not in show.columns: show[m] = ""
    show["agr"] = show["agreement_ratio"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "") if "agreement_ratio" in show.columns else ""

    cols = [{"name":"종목","id":"name"},{"name":"레이어","id":"layer"},{"name":"신호","id":"sig"},
            {"name":"상승%","id":"up%"},{"name":"하락%","id":"dn%"},{"name":"점수","id":"sc"},
            {"name":"XGB","id":"xgb_signal"},{"name":"LGB","id":"lgb_signal"},
            {"name":"LSTM","id":"lstm_signal"},{"name":"GRU","id":"gru_signal"},
            {"name":"TF","id":"transformer_signal"},{"name":"합의","id":"agr"}]

    conds = [
        {"if":{"filter_query":'{sig} contains "BUY"'},"backgroundColor":"#0D2818","color":POS},
        {"if":{"filter_query":'{sig} contains "SELL"'},"backgroundColor":"#2D0A0A","color":NEG},
    ]
    for mc in ["XGB","LGB","LSTM","GRU","TF"]:
        conds.append({"if":{"filter_query":f'{{{mc}}} = "UP"',"column_id":mc},"color":POS,"fontWeight":"bold"})
        conds.append({"if":{"filter_query":f'{{{mc}}} = "DOWN"',"column_id":mc},"color":NEG,"fontWeight":"bold"})

    fig = px.scatter(df, x="pred_up_prob", y="total_score", color="pred_label",
                     color_discrete_map={"UP":POS,"DOWN":NEG,"HOLD":WARN},
                     hover_name="name", size="pred_up_prob",
                     labels={"pred_up_prob":"상승확률 (UP%)","total_score":"종합점수 (Score)"})
    fig.add_vline(x=0.6, line_dash="dot", line_color=TXT2, opacity=0.4)
    fig.add_hline(y=0.5, line_dash="dot", line_color=TXT2, opacity=0.4)
    _fig_layout(fig, 360)
    fig.update_layout(showlegend=True, legend=dict(orientation="h",y=1.08,font=dict(size=10)))

    return html.Div([
        html.Div("종목스크리닝 (Stock Screening)", style={"fontSize":"0.88rem","fontWeight":700,"color":WHITE,"marginBottom":"8px"}),
        dbc.Row([
            dbc.Col(_card(dash_table.DataTable(
                data=show.sort_values("total_score",ascending=False).to_dict("records"),
                columns=cols, sort_action="native", filter_action="native", page_size=15,
                style_header={"backgroundColor":BORDER,"color":WHITE,"fontWeight":600,"fontSize":"0.75rem","border":"none"},
                style_cell={"backgroundColor":PANEL,"color":TXT,"textAlign":"center","fontSize":"0.78rem",
                            "padding":"5px 4px","border":f"1px solid {BORDER}","fontFamily":"SF Mono,Consolas,monospace"},
                style_data_conditional=conds, style_filter={"backgroundColor":BG,"color":TXT},
            ), "예측 테이블 (Predictions)"), md=7),
            dbc.Col(_card(dcc.Graph(figure=fig,config={"displayModeBar":False}), "신호 매트릭스 (Signal Matrix)"), md=5),
        ], className="g-2"),
    ])


# ══════════════════════════════════════════════════════════
#  PAGE 4: DEEP DIVE
# ══════════════════════════════════════════════════════════
def pg_deep(df, macro, horizon, min_prob):
    return html.Div([
        html.Div("종목분석 (Deep Dive)", style={"fontSize":"0.88rem","fontWeight":700,"color":WHITE,"marginBottom":"8px"}),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="deep-ticker", options=stock_opts,
                value=stock_opts[0]["value"] if stock_opts else None,
                clearable=False, style={"fontSize":"0.82rem","backgroundColor":PANEL,"color":TXT}), md=4),
            dbc.Col(dcc.Dropdown(id="deep-days", value=120, clearable=False,
                options=[{"label":f"{d}d","value":d} for d in [60,120,250]],
                style={"fontSize":"0.82rem"}), md=2),
        ], className="g-2 mb-3"),
        html.Div(id="deep-content"),
    ])

@callback(Output("deep-content","children"),
          Input("deep-ticker","value"),Input("deep-days","value"),Input("horizon-select","value"))
def update_deep(ticker,days,horizon):
    if not ticker: return ""
    name = get_ticker_map().get(ticker,ticker)
    pdf = load_price(ticker,days)
    if pdf.empty: return html.P("No price data",style={"color":WARN})

    pdf["trade_date"]=pd.to_datetime(pdf["trade_date"])
    pdf=pdf.sort_values("trade_date")
    cur=float(pdf["close"].iloc[-1])
    pdf["sma5"]=pdf["close"].rolling(5).mean()
    pdf["sma20"]=pdf["close"].rolling(20).mean()
    pdf["sma60"]=pdf["close"].rolling(60).mean()

    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=pdf["trade_date"],open=pdf["open"],high=pdf["high"],
        low=pdf["low"],close=pdf["close"],name="Price",
        increasing_line_color=POS,decreasing_line_color=NEG))
    fig.add_trace(go.Scatter(x=pdf["trade_date"],y=pdf["sma5"],name="5d",line=dict(color=WARN,width=1)))
    fig.add_trace(go.Scatter(x=pdf["trade_date"],y=pdf["sma20"],name="20d",line=dict(color=ACCENT,width=1)))
    fig.add_trace(go.Scatter(x=pdf["trade_date"],y=pdf["sma60"],name="60d",line=dict(color="#A371F7",width=1)))

    try:
        from models.valuation import ValuationCalculator
        b=ValuationCalculator().calculate_price_band(pdf).get(horizon,{})
        if b:
            fig.add_hline(y=b["upper"],line_dash="dot",line_color=NEG,opacity=0.5,annotation_text=f"U {b['upper']:,.0f}",annotation_font_color=NEG)
            fig.add_hline(y=b["lower"],line_dash="dot",line_color=POS,opacity=0.5,annotation_text=f"L {b['lower']:,.0f}",annotation_font_color=POS)
    except: pass
    _fig_layout(fig,360)
    fig.update_layout(xaxis_rangeslider_visible=False,showlegend=True,legend=dict(orientation="h",y=1.08,font=dict(size=9)))

    colors=[POS if c>=o else NEG for c,o in zip(pdf["close"],pdf["open"])]
    fig_v=go.Figure(go.Bar(x=pdf["trade_date"],y=pdf["volume"],marker_color=colors))
    _fig_layout(fig_v,90)

    # Financials
    try:
        from sqlalchemy import text
        with get_db().engine.connect() as conn:
            fdf=pd.read_sql(text("SELECT per,pbr,eps,roe FROM financial_data WHERE ticker=:t ORDER BY period DESC LIMIT 1"),conn,params={"t":ticker})
        fin=fdf.iloc[0].to_dict() if not fdf.empty else {}
    except: fin={}

    try:
        from models.valuation import ValuationCalculator
        si=next((s for s in all_stocks if s["ticker"]==ticker),{})
        val=ValuationCalculator().calculate_fair_value(ticker,cur,fin,si.get("layer",""))
    except: val={"fair_value":None,"upside_pct":None,"grade":"—","valuation_status":""}

    # XAI
    try:
        from models.ensemble import EnsemblePredictor
        exp=EnsemblePredictor().explain_prediction(ticker,load_price(ticker,200),horizon=horizon)
    except: exp={"factors":[],"summary":""}

    factors=[]
    for f in exp.get("factors",[]):
        s=f["signal"]; c=POS if s=="BUY" else (NEG if s=="SELL" else WARN)
        factors.append(dbc.Col(html.Div([
            html.Div([html.Span(f["category"],style={"fontWeight":600,"color":WHITE,"fontSize":"0.8rem"}),
                       html.Span(f" {s}",style={"color":c,"fontWeight":700,"marginLeft":"6px","fontSize":"0.75rem"})]),
            html.Div(f["detail"],style={"color":TXT2,"fontSize":"0.72rem","lineHeight":"1.4","marginTop":"2px"}),
        ],style={"backgroundColor":PANEL,"border":f"1px solid {BORDER}","borderLeft":f"3px solid {c}",
                 "borderRadius":"4px","padding":"8px 10px"}),md=4,className="mb-2"))

    return html.Div([
        dbc.Row([
            dbc.Col(_kpi("현재가 (Price)",f"{cur:,.0f}"),md=2),
            dbc.Col(_kpi("적정가 (Fair)",f"{val['fair_value']:,.0f}" if val["fair_value"] else "—"),md=2),
            dbc.Col(_kpi("괴리율 (Upside)",f"{val['upside_pct']:+.1f}%" if val["upside_pct"] is not None else "—",
                val.get("grade",""),POS if val.get("upside_pct") and val["upside_pct"]>0 else NEG),md=2),
            dbc.Col(_kpi("PER",str(fin.get("per","—"))),md=1),
            dbc.Col(_kpi("PBR",str(fin.get("pbr","—"))),md=1),
        ],className="g-2 mb-2"),
        dbc.Row([
            dbc.Col([_card(dcc.Graph(figure=fig,config={"displayModeBar":False}),f"{name}"),
                     _card(dcc.Graph(figure=fig_v,config={"displayModeBar":False}),"Volume")],md=8),
            dbc.Col(_card(html.Div([
                html.Div(exp.get("summary",""),style={"color":TXT,"fontSize":"0.82rem","marginBottom":"8px"}),
            ]),"AI 분석요약 (Analysis)"),md=4),
        ],className="g-2"),
        html.Div("요인분석 (Factor Analysis)",style={"fontSize":"0.72rem","fontWeight":600,"color":TXT2,
            "letterSpacing":"0.5px","marginTop":"6px","marginBottom":"4px"}),
        dbc.Row(factors,className="g-2"),
    ])


# ══════════════════════════════════════════════════════════
#  PAGE 5: ACTION
# ══════════════════════════════════════════════════════════
def pg_action(df, macro, horizon, min_prob):
    if df.empty: return _card(html.P("No data",style={"color":WARN}))
    hl={1:"D+1",5:"D+5",20:"D+20"}.get(horizon,"")

    buy=df[(df.pred_label=="UP")&(df.pred_up_prob>=min_prob)].sort_values("total_score",ascending=False).head(7)
    sell=df[df.pred_label=="DOWN"].sort_values("pred_down_prob",ascending=False).head(7)

    def act_card(r,d):
        nm=r.get("name",""); c=POS if d=="buy" else NEG
        p=r.get("pred_up_prob",0) if d=="buy" else r.get("pred_down_prob",0)
        sc=r.get("total_score",0); ag=r.get("agreement_ratio",0)

        mbadges=[]
        for m,l in [("xgb_signal","X"),("lgb_signal","L"),("lstm_signal","LS"),("gru_signal","G"),("transformer_signal","T")]:
            v=r.get(m,"")
            if v=="UP": mbadges.append(html.Span(l,style={"color":POS,"fontWeight":700,"marginRight":"4px","fontSize":"0.7rem"}))
            elif v=="DOWN": mbadges.append(html.Span(l,style={"color":NEG,"fontWeight":700,"marginRight":"4px","fontSize":"0.7rem"}))
            elif v: mbadges.append(html.Span(l,style={"color":TXT2,"marginRight":"4px","fontSize":"0.7rem"}))

        return html.Div([
            html.Div([
                html.Span(nm,style={"color":WHITE,"fontWeight":600,"fontSize":"0.88rem"}),
                html.Span(f" {p:.1%}",style={"color":c,"fontWeight":700,"marginLeft":"8px"}),
                html.Span(f" {sc:.3f}",style={"color":TXT2,"fontSize":"0.78rem","marginLeft":"6px"}),
                html.Span(f" {ag:.0%}" if pd.notna(ag) else "",
                    style={"color":ACCENT,"fontSize":"0.72rem","marginLeft":"6px"}),
            ]),
            html.Div(mbadges,style={"marginTop":"2px"}),
        ],style={"backgroundColor":PANEL,"border":f"1px solid {BORDER}","borderLeft":f"4px solid {c}",
                 "borderRadius":"4px","padding":"10px 12px","marginBottom":"6px"})

    bc=[act_card(r,"buy") for _,r in buy.iterrows()] if not buy.empty else [html.P("No signals",style={"color":TXT2})]
    sc=[act_card(r,"sell") for _,r in sell.iterrows()] if not sell.empty else [html.P("No signals",style={"color":TXT2})]

    return html.Div([
        html.Div("투자실행 (Portfolio Action)",style={"fontSize":"0.88rem","fontWeight":700,"color":WHITE,"marginBottom":"8px"}),
        dbc.Row([
            dbc.Col([html.Div(f"{hl} 매수추천 (BUY)",style={"color":POS,"fontWeight":700,"fontSize":"0.82rem","marginBottom":"6px"}),
                     html.Div(bc)],md=6),
            dbc.Col([html.Div(f"{hl} 매도경고 (SELL)",style={"color":NEG,"fontWeight":700,"fontSize":"0.82rem","marginBottom":"6px"}),
                     html.Div(sc)],md=6),
        ],className="g-2"),
    ])


# ── Run ─────────────────────────────────────────────────
def run_dash(debug=False, port=8050):
    app.run(debug=debug, port=port, host="localhost")

if __name__ == "__main__":
    run_dash(debug=True)
