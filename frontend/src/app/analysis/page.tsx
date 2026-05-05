"use client";

import { useEffect, useState, useRef } from "react";

const API = "http://localhost:8000/api";

interface Stock { ticker: string; name: string; layer: string; }
interface PriceData { trade_date: string; open: number; high: number; low: number; close: number; volume: number; }
interface Indicator {
  trade_date: string; sma5: number; sma20: number; sma60: number; sma120: number;
  ema5: number; ema20: number; macd: number; macd_signal: number; macd_hist: number;
  rsi14: number; bb_upper: number; bb_middle: number; bb_lower: number; bb_pct: number;
  stoch_k: number; stoch_d: number; obv: number; volume_ratio: number; atr14: number;
}
interface Financial { period: string; revenue: number; operating_income: number; net_income: number; eps: number; roe: number; per: number; pbr: number; debt_ratio: number; op_margin: number; }
interface Supply { trade_date: string; foreign_net: number; institution_net: number; individual_net: number; foreign_net_5d: number; institution_net_5d: number; }
interface Prediction { pred_label: string; pred_up_prob: number; pred_down_prob: number; total_score: number; xgb_signal: string; lgb_signal: string; }

export default function AnalysisPage() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [selectedTicker, setSelectedTicker] = useState("005930.KS");
  const [prices, setPrices] = useState<PriceData[]>([]);
  const [indicators, setIndicators] = useState<Indicator[]>([]);
  const [financials, setFinancials] = useState<Financial[]>([]);
  const [supply, setSupply] = useState<Supply[]>([]);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [days, setDays] = useState(120);
  const [baseDate, setBaseDate] = useState(new Date().toISOString().split("T")[0]);
  const [search, setSearch] = useState("");
  const [activeTab, setActiveTab] = useState("trend");
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [loaded, setLoaded] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => { fetch(`${API}/stocks`).then(r => r.json()).then(d => setStocks(d.stocks || [])); }, []);

  const loadData = () => {
    setLoading(true);
    setLoaded(false);
    Promise.all([
      fetch(`${API}/prices/${selectedTicker}?days=${days}&end_date=${baseDate}`).then(r => r.json()),
      fetch(`${API}/indicators/${selectedTicker}?days=${days}&end_date=${baseDate}`).then(r => r.json()),
      fetch(`${API}/financials/${selectedTicker}`).then(r => r.json()),
      fetch(`${API}/supply/${selectedTicker}?days=${days}`).then(r => r.json()),
      fetch(`${API}/predictions-binary?horizon=1&model=model_A`).then(r => r.json()),
      fetch(`${API}/news/${selectedTicker}`).then(r => r.json()),
    ]).then(([priceData, indData, finData, supData, predData, newsData]) => {
      setPrices(priceData.prices || []);
      setIndicators(indData.indicators || []);
      setFinancials(finData.financials || []);
      setSupply(supData.supply || []);
      const p = (predData.predictions || []).find((x: {ticker:string}) => x.ticker === selectedTicker);
      setPrediction(p || null);
      setLoaded(true);
      setLoading(false);
    });
  };

  useEffect(() => { if (canvasRef.current && prices.length > 0 && loaded) drawChart(); }, [prices, indicators, activeTab, loaded]);

  const drawChart = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const width = canvas.width = canvas.offsetWidth * 2;
    const height = canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);
    const w = width / 2, h = height / 2;
    ctx.fillStyle = "#0d1b2a"; ctx.fillRect(0, 0, w, h);

    const pad = { top: 20, right: 65, bottom: 30, left: 10 };
    const cW = w - pad.left - pad.right, cH = h - pad.top - pad.bottom;

    if (activeTab === "trend" || activeTab === "bollinger") {
      // Candlestick + overlays
      const maxP = Math.max(...prices.map(p => p.high));
      const minP = Math.min(...prices.map(p => p.low));
      const range = maxP - minP || 1;
      const gap = cW / prices.length;
      const candleW = Math.max(2, gap * 0.6);

      // Grid
      ctx.strokeStyle = "#1a2a3a"; ctx.lineWidth = 0.5;
      for (let i = 0; i <= 4; i++) {
        const y = pad.top + (cH / 4) * i;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
        ctx.fillStyle = "#667788"; ctx.font = "9px monospace";
        ctx.fillText((maxP - (range / 4) * i).toLocaleString(), w - pad.right + 5, y + 4);
      }

      // Candles
      prices.forEach((p, i) => {
        const x = pad.left + i * gap + gap / 2;
        const isUp = p.close >= p.open;
        const hY = pad.top + ((maxP - p.high) / range) * cH;
        const lY = pad.top + ((maxP - p.low) / range) * cH;
        const oY = pad.top + ((maxP - p.open) / range) * cH;
        const closeY = pad.top + ((maxP - p.close) / range) * cH;
        ctx.strokeStyle = isUp ? "#00d26a" : "#ff4444"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, hY); ctx.lineTo(x, lY); ctx.stroke();
        ctx.fillStyle = isUp ? "#00d26a" : "#ff4444";
        ctx.fillRect(x - candleW / 2, Math.min(oY, closeY), candleW, Math.max(Math.abs(closeY - oY), 1));
      });

      // Overlay lines
      if (activeTab === "trend" && indicators.length > 0) {
        const drawLine = (values: (number|null)[], color: string) => {
          ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.beginPath();
          let started = false;
          values.forEach((v, i) => {
            if (v == null) return;
            const x = pad.left + i * gap + gap / 2;
            const y = pad.top + ((maxP - v) / range) * cH;
            if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
          });
          ctx.stroke();
        };
        drawLine(indicators.map(d => d.sma5), "#ffd700");
        drawLine(indicators.map(d => d.sma20), "#4fc3f7");
        drawLine(indicators.map(d => d.sma60), "#ff9800");
        drawLine(indicators.map(d => d.sma120), "#ab47bc");
      }

      if (activeTab === "bollinger" && indicators.length > 0) {
        const drawLine = (values: (number|null)[], color: string, dash?: number[]) => {
          ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash(dash || []);
          ctx.beginPath(); let started = false;
          values.forEach((v, i) => {
            if (v == null) return;
            const x = pad.left + i * gap + gap / 2;
            const y = pad.top + ((maxP - v) / range) * cH;
            if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
          });
          ctx.stroke(); ctx.setLineDash([]);
        };
        drawLine(indicators.map(d => d.bb_upper), "#4fc3f7", [4, 2]);
        drawLine(indicators.map(d => d.bb_middle), "#4fc3f7");
        drawLine(indicators.map(d => d.bb_lower), "#4fc3f7", [4, 2]);
      }
    } else {
      // Oscillator charts (RSI, MACD, Stochastic)
      let data: { values: number[][]; labels: string[]; colors: string[]; yMin: number; yMax: number; refLines?: number[] } | null = null;

      if (activeTab === "rsi") {
        data = { values: [indicators.map(d => d.rsi14)], labels: ["RSI(14)"], colors: ["#ffd700"], yMin: 0, yMax: 100, refLines: [30, 70] };
      } else if (activeTab === "macd") {
        data = { values: [indicators.map(d => d.macd), indicators.map(d => d.macd_signal)], labels: ["MACD", "Signal"], colors: ["#4fc3f7", "#ff9800"], yMin: Math.min(...indicators.map(d => Math.min(d.macd||0, d.macd_signal||0))), yMax: Math.max(...indicators.map(d => Math.max(d.macd||0, d.macd_signal||0))), refLines: [0] };
      } else if (activeTab === "stoch") {
        data = { values: [indicators.map(d => d.stoch_k), indicators.map(d => d.stoch_d)], labels: ["%K", "%D"], colors: ["#4fc3f7", "#ff9800"], yMin: 0, yMax: 100, refLines: [20, 80] };
      }

      if (data) {
        const range = data.yMax - data.yMin || 1;
        const gap = cW / (data.values[0]?.length || 1);

        // Grid + ref lines
        ctx.strokeStyle = "#1a2a3a"; ctx.lineWidth = 0.5;
        for (let i = 0; i <= 4; i++) {
          const y = pad.top + (cH / 4) * i;
          ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
          ctx.fillStyle = "#667788"; ctx.font = "9px monospace";
          ctx.fillText((data.yMax - (range / 4) * i).toFixed(0), w - pad.right + 5, y + 4);
        }
        data.refLines?.forEach(ref => {
          const y = pad.top + ((data!.yMax - ref) / range) * cH;
          ctx.strokeStyle = "#ff444466"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
          ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
          ctx.setLineDash([]);
        });

        // Lines
        data.values.forEach((values, idx) => {
          ctx.strokeStyle = data!.colors[idx]; ctx.lineWidth = 1.5; ctx.beginPath();
          let started = false;
          values.forEach((v, i) => {
            if (v == null) return;
            const x = pad.left + i * gap + gap / 2;
            const y = pad.top + ((data!.yMax - v) / range) * cH;
            if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
          });
          ctx.stroke();
        });

        // MACD histogram
        if (activeTab === "macd") {
          const histData = indicators.map(d => d.macd_hist);
          histData.forEach((v, i) => {
            if (v == null) return;
            const x = pad.left + i * gap + gap / 2;
            const zeroY = pad.top + ((data!.yMax - 0) / range) * cH;
            const barY = pad.top + ((data!.yMax - v) / range) * cH;
            ctx.fillStyle = v >= 0 ? "#00d26a88" : "#ff444488";
            ctx.fillRect(x - 2, Math.min(zeroY, barY), 4, Math.abs(barY - zeroY));
          });
        }
      }
    }
  };

  const filteredStocks = stocks.filter(s => search === "" || s.name.includes(search) || s.ticker.includes(search.toUpperCase()));
  const selectedName = stocks.find(s => s.ticker === selectedTicker)?.name || selectedTicker;
  const latestInd = indicators.length > 0 ? indicators[indicators.length - 1] : null;

  const tabs = [
    { key: "trend", label: "이동평균" },
    { key: "bollinger", label: "볼린저밴드" },
    { key: "rsi", label: "RSI" },
    { key: "macd", label: "MACD" },
    { key: "stoch", label: "스토캐스틱" },
  ];

  return (
    <div className="min-h-screen bg-[#0a0f1a] text-white">
      <header className="border-b border-gray-800 px-6 py-4">
        <h2 className="text-xl font-bold">Analysis</h2>
        <p className="text-gray-500 text-sm">개별 종목 심층 분석 — 기술적·가치·수급·AI 판단</p>
      </header>

      <div className="p-6 space-y-5">
        {/* Stock selector */}
        <section className="flex flex-wrap items-center gap-4">
          <input type="text" placeholder="종목 검색..." value={search} onChange={e => setSearch(e.target.value)}
            className="bg-[#0d1b2a] border border-gray-700 rounded-lg px-3 py-2 text-white text-sm w-44" />
          <select value={selectedTicker} onChange={e => setSelectedTicker(e.target.value)}
            className="bg-[#0d1b2a] border border-gray-700 rounded-lg px-3 py-2 text-white text-sm w-60">
            {filteredStocks.map(s => <option key={s.ticker} value={s.ticker}>{s.name} ({s.ticker})</option>)}
          </select>
          <div>
            <label className="text-gray-500 text-[10px] block">기준일</label>
            <input type="date" value={baseDate} onChange={e => setBaseDate(e.target.value)}
              className="bg-[#0d1b2a] border border-gray-700 rounded-lg px-3 py-1.5 text-white text-sm" />
          </div>
          <div className="flex gap-1">
            {[60, 120, 250].map(d => (
              <button key={d} onClick={() => setDays(d)}
                className={`px-3 py-1.5 rounded-lg text-sm border transition ${days === d ? "bg-cyan-900/50 text-cyan-400 border-cyan-800" : "text-gray-400 border-gray-700 hover:bg-gray-800"}`}>
                {d}일
              </button>
            ))}
          </div>
          <button onClick={loadData} disabled={loading}
            className="bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 text-white font-bold px-5 py-2 rounded-lg text-sm transition">
            {loading ? "로딩 중..." : "확인"}
          </button>
          <span className="ml-auto text-lg font-bold text-white">{selectedName}</span>
        </section>

        {/* ═══ 1. 기술적 분석 차트 ═══ */}
        <section className="bg-[#111827] rounded-xl border border-gray-800 p-4">
          <div className="flex items-center gap-2 mb-3">
            <h3 className="text-sm font-bold text-cyan-400">기술적 분석</h3>
            <div className="flex gap-1 ml-4">
              {tabs.map(t => (
                <button key={t.key} onClick={() => setActiveTab(t.key)}
                  className={`px-2.5 py-1 rounded text-xs border transition ${activeTab === t.key ? "bg-cyan-900/50 text-cyan-400 border-cyan-800" : "text-gray-500 border-gray-700 hover:bg-gray-800"}`}>
                  {t.label}
                </button>
              ))}
            </div>
          </div>
          <canvas ref={canvasRef} className="w-full h-72 rounded" style={{ imageRendering: "crisp-edges" }} />
          {/* Legend */}
          <div className="flex gap-4 mt-2 text-xs">
            {activeTab === "trend" && <>
              <span className="text-[#ffd700]">— SMA5</span><span className="text-[#4fc3f7]">— SMA20</span>
              <span className="text-[#ff9800]">— SMA60</span><span className="text-[#ab47bc]">— SMA120</span>
            </>}
            {activeTab === "bollinger" && <span className="text-[#4fc3f7]">— Upper / Middle / Lower</span>}
            {activeTab === "rsi" && <><span className="text-[#ffd700]">— RSI(14)</span><span className="text-gray-500">-- 30/70 기준선</span></>}
            {activeTab === "macd" && <><span className="text-[#4fc3f7]">— MACD</span><span className="text-[#ff9800]">— Signal</span><span className="text-gray-500">▮ Histogram</span></>}
            {activeTab === "stoch" && <><span className="text-[#4fc3f7]">— %K</span><span className="text-[#ff9800]">— %D</span><span className="text-gray-500">-- 20/80 기준선</span></>}
          </div>
        </section>

        {/* ═══ 2. 현재 기술지표 스냅샷 ═══ */}
        {latestInd && (
          <section className="bg-[#111827] rounded-xl border border-gray-800 p-4">
            <h3 className="text-sm font-bold text-cyan-400 mb-3">기술지표 현재값 ({indicators[indicators.length-1]?.trade_date})</h3>
            <div className="grid grid-cols-3 md:grid-cols-6 lg:grid-cols-9 gap-2">
              {[
                {l:"RSI(14)",v:latestInd.rsi14?.toFixed(1),warn:latestInd.rsi14>70?"text-red-400":latestInd.rsi14<30?"text-green-400":"text-white"},
                {l:"MACD",v:latestInd.macd?.toFixed(1),warn:latestInd.macd>0?"text-green-400":"text-red-400"},
                {l:"Stoch %K",v:latestInd.stoch_k?.toFixed(1),warn:latestInd.stoch_k>80?"text-red-400":latestInd.stoch_k<20?"text-green-400":"text-white"},
                {l:"BB %B",v:latestInd.bb_pct?.toFixed(2),warn:latestInd.bb_pct>1?"text-red-400":latestInd.bb_pct<0?"text-green-400":"text-white"},
                {l:"Vol Ratio",v:latestInd.volume_ratio?.toFixed(2),warn:latestInd.volume_ratio>2?"text-yellow-400":"text-white"},
                {l:"ATR(14)",v:latestInd.atr14?.toFixed(0),warn:"text-white"},
                {l:"SMA5",v:latestInd.sma5?.toLocaleString(),warn:"text-white"},
                {l:"SMA20",v:latestInd.sma20?.toLocaleString(),warn:"text-white"},
                {l:"SMA60",v:latestInd.sma60?.toLocaleString(),warn:"text-white"},
              ].map(item => (
                <div key={item.l} className="bg-[#0d1b2a] rounded-lg p-2 border border-gray-800 text-center">
                  <p className="text-gray-500 text-[10px]">{item.l}</p>
                  <p className={`font-mono text-sm font-bold ${item.warn}`}>{item.v || "N/A"}</p>
                </div>
              ))}
            </div>
          </section>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          {/* ═══ 3. 가치 분석 (재무) ═══ */}
          <section className="bg-[#111827] rounded-xl border border-gray-800 p-4">
            <h3 className="text-sm font-bold text-cyan-400 mb-3">가치 분석 (Valuation)</h3>
            {financials.length > 0 ? (
              <div className="space-y-2">
                {[
                  {l:"PER",v:financials[0].per,unit:"배",good:(v:number)=>v>0&&v<20},
                  {l:"PBR",v:financials[0].pbr,unit:"배",good:(v:number)=>v>0&&v<3},
                  {l:"ROE",v:financials[0].roe,unit:"%",good:(v:number)=>v>10},
                  {l:"영업이익률",v:financials[0].op_margin,unit:"%",good:(v:number)=>v>10},
                  {l:"부채비율",v:financials[0].debt_ratio,unit:"%",good:(v:number)=>v<100},
                  {l:"EPS",v:financials[0].eps,unit:"원",good:()=>true},
                  {l:"매출",v:financials[0].revenue,unit:"억",good:()=>true},
                ].map(item => (
                  <div key={item.l} className="flex justify-between items-center">
                    <span className="text-gray-400 text-sm">{item.l}</span>
                    <span className={`font-mono text-sm ${item.v != null && item.good(item.v) ? "text-green-400" : item.v != null ? "text-yellow-400" : "text-gray-500"}`}>
                      {item.v != null ? `${typeof item.v === "number" && item.v > 10000 ? (item.v/100000000).toFixed(0) : item.v}${item.unit}` : "N/A"}
                    </span>
                  </div>
                ))}
                <p className="text-[10px] text-gray-600 mt-2">기준: {financials[0].period}</p>
              </div>
            ) : <p className="text-gray-500 text-sm">재무 데이터 없음</p>}
          </section>

          {/* ═══ 4. 수급 분석 ═══ */}
          <section className="bg-[#111827] rounded-xl border border-gray-800 p-4">
            <h3 className="text-sm font-bold text-cyan-400 mb-3">수급 분석 (최근)</h3>
            {supply.length > 0 ? (
              <div className="space-y-2">
                {(() => {
                  const latest = supply[supply.length - 1];
                  return [
                    {l:"외국인 순매수",v:latest.foreign_net},
                    {l:"기관 순매수",v:latest.institution_net},
                    {l:"개인 순매수",v:latest.individual_net},
                    {l:"외국인 5일합",v:latest.foreign_net_5d},
                    {l:"기관 5일합",v:latest.institution_net_5d},
                  ].map(item => (
                    <div key={item.l} className="flex justify-between items-center">
                      <span className="text-gray-400 text-sm">{item.l}</span>
                      <span className={`font-mono text-sm ${item.v>0?"text-green-400":item.v<0?"text-red-400":"text-gray-400"}`}>
                        {item.v != null ? `${item.v > 0 ? "+" : ""}${(item.v/1000).toFixed(0)}천주` : "N/A"}
                      </span>
                    </div>
                  ));
                })()}
                <p className="text-[10px] text-gray-600 mt-2">기준: {supply[supply.length-1]?.trade_date}</p>
              </div>
            ) : <p className="text-gray-500 text-sm">수급 데이터 없음</p>}
          </section>

          {/* ═══ 5. AI 판단 ═══ */}
          <section className="bg-[#111827] rounded-xl border border-gray-800 p-4">
            <h3 className="text-sm font-bold text-cyan-400 mb-3">AI 예측 판단</h3>
            {prediction ? (
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">최종 예측</span>
                  <span className={`px-3 py-1 rounded text-sm font-bold border ${prediction.pred_label==="UP"?"bg-green-900/50 text-green-400 border-green-700":prediction.pred_label==="DOWN"?"bg-red-900/50 text-red-400 border-red-700":"bg-yellow-900/50 text-yellow-400 border-yellow-700"}`}>
                    {prediction.pred_label}
                  </span>
                </div>
                <div className="flex justify-between"><span className="text-gray-400 text-sm">상승 확률</span><span className="text-green-400 font-bold">{(prediction.pred_up_prob*100).toFixed(1)}%</span></div>
                <div className="flex justify-between"><span className="text-gray-400 text-sm">하락 확률</span><span className="text-red-400 font-bold">{(prediction.pred_down_prob*100).toFixed(1)}%</span></div>
                <div className="flex justify-between"><span className="text-gray-400 text-sm">종합 점수</span><span className="text-cyan-400 font-mono">{prediction.total_score?.toFixed(3)}</span></div>
                <hr className="border-gray-800"/>
                <div className="flex justify-between"><span className="text-gray-400 text-sm">XGBoost</span><span className={prediction.xgb_signal==="UP"?"text-green-400":"text-red-400"}>{prediction.xgb_signal}</span></div>
                <div className="flex justify-between"><span className="text-gray-400 text-sm">LightGBM</span><span className={prediction.lgb_signal==="UP"?"text-green-400":"text-red-400"}>{prediction.lgb_signal}</span></div>
              </div>
            ) : <p className="text-gray-500 text-sm">예측 데이터 없음</p>}
          </section>
        </div>
      </div>
    </div>
  );
}
