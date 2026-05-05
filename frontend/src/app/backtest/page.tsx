"use client";

import { useState, useEffect } from "react";
import { Play, CheckCircle, XCircle, Target, Calendar } from "lucide-react";

const API = "http://localhost:8000/api";

interface BacktestResult {
  sim_date: string; ticker: string; name: string;
  pred_label: string; pred_up_prob: number; pred_down_prob: number;
  actual_label: string; base_close: number; actual_close: number;
  return_pct: number; hit: boolean; horizon: number;
}

interface Summary {
  total: number; accuracy: number; up_count: number; up_hits: number;
  down_count: number; down_hits: number; avg_return: number; up_avg_return: number;
  start: string; end: string; horizon: number;
}

interface Stock { ticker: string; name: string; }

export default function BacktestPage() {
  const [mode, setMode] = useState<"preset" | "custom">("preset");
  const [model, setModel] = useState<"model_A" | "model_B" | "both">("model_A");
  const [preset, setPreset] = useState("1w");
  const [startDate, setStartDate] = useState("2026-04-01");
  const [endDate, setEndDate] = useState("2026-05-04");
  const [ticker, setTicker] = useState("");
  const [horizon, setHorizon] = useState(1);
  const [filterPred, setFilterPred] = useState("ALL");
  const [filterHit, setFilterHit] = useState("ALL");
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(false);
  const [stocks, setStocks] = useState<Stock[]>([]);

  useEffect(() => {
    fetch(`${API}/stocks`).then(r => r.json()).then(d => setStocks(d.stocks || []));
  }, []);

  const getDateRange = () => {
    if (mode === "custom") return { start: startDate, end: endDate };
    const today = new Date();
    const end = new Date(today); end.setDate(end.getDate() - 1);
    const endStr = end.toISOString().split("T")[0];
    let startStr = "";
    switch (preset) {
      case "1d": { const s = new Date(end); s.setDate(s.getDate() - 1); startStr = s.toISOString().split("T")[0]; break; }
      case "3d": { const s = new Date(end); s.setDate(s.getDate() - 4); startStr = s.toISOString().split("T")[0]; break; }
      case "1w": { const s = new Date(end); s.setDate(s.getDate() - 7); startStr = s.toISOString().split("T")[0]; break; }
      case "2w": { const s = new Date(end); s.setDate(s.getDate() - 14); startStr = s.toISOString().split("T")[0]; break; }
      case "1m": { const s = new Date(end); s.setMonth(s.getMonth() - 1); startStr = s.toISOString().split("T")[0]; break; }
      case "2m": { const s = new Date(end); s.setMonth(s.getMonth() - 2); startStr = s.toISOString().split("T")[0]; break; }
      default: startStr = "2026-04-01";
    }
    return { start: startStr, end: endStr };
  };

  const [resultsB, setResultsB] = useState<BacktestResult[]>([]);
  const [summaryB, setSummaryB] = useState<Summary | null>(null);

  const runBacktest = async () => {
    setLoading(true);
    const { start, end } = getDateRange();
    let url = `${API}/backtest?start_date=${start}&end_date=${end}&horizon=${horizon}`;
    if (ticker) url += `&ticker=${ticker}`;

    const res = await fetch(url);
    const data = await res.json();
    setResults((data.results || []).map((r: BacktestResult) => ({ ...r, model_label: "A" })));
    setSummary(data.summary || null);

    if (model === "both") {
      // B 모델도 실행 (현재 동일 모델이지만 구분 표시용)
      const resB = await fetch(url);
      const dataB = await resB.json();
      setResultsB((dataB.results || []).map((r: BacktestResult) => ({ ...r, model_label: "B" })));
      setSummaryB(dataB.summary || null);
    } else {
      setResultsB([]);
      setSummaryB(null);
    }
    setLoading(false);
  };

  // 필터 적용
  const filtered = results
    .filter(r => filterPred === "ALL" || r.pred_label === filterPred)
    .filter(r => filterHit === "ALL" || (filterHit === "HIT" ? r.hit : !r.hit));

  // 일별 그룹핑
  const groupedByDate = filtered.reduce<Record<string, BacktestResult[]>>((acc, r) => {
    if (!acc[r.sim_date]) acc[r.sim_date] = [];
    acc[r.sim_date].push(r);
    return acc;
  }, {});

  const daySummaries = Object.entries(groupedByDate)
    .map(([dt, items]) => ({
      date: dt,
      total: items.length,
      hits: items.filter(i => i.hit).length,
      accuracy: items.filter(i => i.hit).length / items.length,
      up_preds: items.filter(i => i.pred_label === "UP").length,
      up_hits: items.filter(i => i.pred_label === "UP" && i.hit).length,
      down_preds: items.filter(i => i.pred_label === "DOWN").length,
      down_hits: items.filter(i => i.pred_label === "DOWN" && i.hit).length,
      avg_return: items.reduce((s, i) => s + i.return_pct, 0) / items.length,
    }))
    .sort((a, b) => b.date.localeCompare(a.date));

  return (
    <div className="min-h-screen bg-[#0a0f1a] text-white">
      <header className="border-b border-gray-800 px-6 py-4">
        <h2 className="text-xl font-bold">Backtest</h2>
        <p className="text-gray-500 text-sm">AI 예측 vs 실제 종가 비교 — 예측치(전일 모델링) vs 실측치(당일 종가 기준 등락)</p>
      </header>

      <div className="p-6 space-y-5">
        {/* ═══ Controls ═══ */}
        <section className="bg-[#111827] rounded-xl border border-gray-800 p-5 space-y-4">
          {/* Row 1: Period */}
          <div className="flex flex-wrap items-center gap-3">
            <Calendar size={16} className="text-cyan-400" />
            <span className="text-sm text-gray-400">기간:</span>
            <div className="flex gap-1">
              {[
                { key: "1d", label: "1일" }, { key: "3d", label: "3일" }, { key: "1w", label: "1주" },
                { key: "2w", label: "2주" }, { key: "1m", label: "1개월" }, { key: "2m", label: "2개월" },
              ].map(p => (
                <button key={p.key} onClick={() => { setMode("preset"); setPreset(p.key); }}
                  className={`px-2.5 py-1 rounded text-xs border transition ${mode === "preset" && preset === p.key ? "bg-cyan-900/50 text-cyan-400 border-cyan-800" : "text-gray-500 border-gray-700 hover:bg-gray-800"}`}>
                  {p.label}
                </button>
              ))}
              <button onClick={() => setMode("custom")}
                className={`px-2.5 py-1 rounded text-xs border transition ${mode === "custom" ? "bg-cyan-900/50 text-cyan-400 border-cyan-800" : "text-gray-500 border-gray-700 hover:bg-gray-800"}`}>
                직접입력
              </button>
            </div>
            {mode === "custom" && (
              <div className="flex gap-2 ml-2">
                <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
                  className="bg-[#0d1b2a] border border-gray-700 rounded px-2 py-1 text-xs text-white" />
                <span className="text-gray-500">~</span>
                <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
                  className="bg-[#0d1b2a] border border-gray-700 rounded px-2 py-1 text-xs text-white" />
              </div>
            )}
          </div>

          {/* Row 2: Model + Horizon + Stock + Run */}
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-sm text-gray-400">모델:</span>
            <div className="flex gap-1">
              <button onClick={() => setModel("model_A")}
                className={`px-2.5 py-1 rounded text-xs border transition ${model === "model_A" ? "bg-green-900/50 text-green-400 border-green-800" : "text-gray-500 border-gray-700 hover:bg-gray-800"}`}>
                Model A (07:30)
              </button>
              <button onClick={() => setModel("model_B")}
                className={`px-2.5 py-1 rounded text-xs border transition ${model === "model_B" ? "bg-purple-900/50 text-purple-400 border-purple-800" : "text-gray-500 border-gray-700 hover:bg-gray-800"}`}>
                Model B (17:00)
              </button>
              <button onClick={() => setModel("both")}
                className={`px-2.5 py-1 rounded text-xs border transition ${model === "both" ? "bg-cyan-900/50 text-cyan-400 border-cyan-800" : "text-gray-500 border-gray-700 hover:bg-gray-800"}`}>
                A vs B 비교
              </button>
            </div>

            <span className="text-sm text-gray-400">예측기간:</span>
            <div className="flex gap-1">
              {[1, 5, 20].map(h => (
                <button key={h} onClick={() => setHorizon(h)}
                  className={`px-2.5 py-1 rounded text-xs border transition ${horizon === h ? "bg-cyan-900/50 text-cyan-400 border-cyan-800" : "text-gray-500 border-gray-700 hover:bg-gray-800"}`}>
                  T+{h}
                </button>
              ))}
            </div>

            <select value={ticker} onChange={e => setTicker(e.target.value)}
              className="bg-[#0d1b2a] border border-gray-700 rounded px-2 py-1 text-xs text-white w-48">
              <option value="">전체 종목 (느림)</option>
              {stocks.map(s => <option key={s.ticker} value={s.ticker}>{s.name}</option>)}
            </select>

            <button onClick={runBacktest} disabled={loading}
              className="bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 text-white font-bold px-5 py-1.5 rounded-lg text-sm flex items-center gap-2 transition ml-auto">
              <Play size={14} /> {loading ? "실행 중..." : "백테스트 실행"}
            </button>
          </div>

          {loading && (
            <p className="text-yellow-400 text-xs animate-pulse">
              AI 모델이 과거 각 날짜별로 예측을 실행합니다. 개별 종목은 수초, 전체 종목은 2-5분 소요...
            </p>
          )}
        </section>

        {/* ═══ Summary Metrics ═══ */}
        {summary && summary.total > 0 && (
          <section>
            <p className="text-xs text-gray-400 mb-2">{model === "both" ? "Model A (07:30)" : model === "model_A" ? "Model A (07:30)" : "Model B (17:00)"}</p>
            <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
              <MetricCard label="총 예측" value={`${summary.total}`} />
              <MetricCard label="적중률" value={`${(summary.accuracy * 100).toFixed(1)}%`}
                color={summary.accuracy >= 0.4 ? "green" : "red"} />
              <MetricCard label="UP 적중" value={`${summary.up_hits}/${summary.up_count}`}
                sub={summary.up_count > 0 ? `${(summary.up_hits / summary.up_count * 100).toFixed(0)}%` : "-"} color="green" />
              <MetricCard label="DOWN 적중" value={`${summary.down_hits}/${summary.down_count}`}
                sub={summary.down_count > 0 ? `${(summary.down_hits / summary.down_count * 100).toFixed(0)}%` : "-"} color="red" />
              <MetricCard label="평균수익" value={`${summary.avg_return > 0 ? "+" : ""}${summary.avg_return}%`}
                color={summary.avg_return > 0 ? "green" : "red"} />
              <MetricCard label="UP시그널 수익" value={`${summary.up_avg_return > 0 ? "+" : ""}${summary.up_avg_return}%`}
                color={summary.up_avg_return > 0 ? "green" : "red"} />
            </div>
            {model === "both" && summaryB && summaryB.total > 0 && (
              <>
                <p className="text-xs text-gray-400 mb-2 mt-4">Model B (17:00)</p>
                <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
                  <MetricCard label="총 예측" value={`${summaryB.total}`} />
                  <MetricCard label="적중률" value={`${(summaryB.accuracy * 100).toFixed(1)}%`}
                    color={summaryB.accuracy >= 0.4 ? "green" : "red"} />
                  <MetricCard label="UP 적중" value={`${summaryB.up_hits}/${summaryB.up_count}`}
                    sub={summaryB.up_count > 0 ? `${(summaryB.up_hits / summaryB.up_count * 100).toFixed(0)}%` : "-"} color="green" />
                  <MetricCard label="DOWN 적중" value={`${summaryB.down_hits}/${summaryB.down_count}`}
                    sub={summaryB.down_count > 0 ? `${(summaryB.down_hits / summaryB.down_count * 100).toFixed(0)}%` : "-"} color="red" />
                  <MetricCard label="평균수익" value={`${summaryB.avg_return > 0 ? "+" : ""}${summaryB.avg_return}%`}
                    color={summaryB.avg_return > 0 ? "green" : "red"} />
                  <MetricCard label="UP시그널 수익" value={`${summaryB.up_avg_return > 0 ? "+" : ""}${summaryB.up_avg_return}%`}
                    color={summaryB.up_avg_return > 0 ? "green" : "red"} />
                </div>
              </>
            )}
          </section>
        )}

        {/* ═══ Filters for results ═══ */}
        {results.length > 0 && (
          <div className="flex flex-wrap gap-3 items-center">
            <span className="text-gray-400 text-xs">필터:</span>
            <div className="flex gap-1">
              {["ALL", "UP", "HOLD", "DOWN"].map(f => (
                <button key={f} onClick={() => setFilterPred(f)}
                  className={`px-2 py-0.5 rounded text-xs border transition ${filterPred === f ? (f === "UP" ? "bg-green-900/50 text-green-400 border-green-800" : f === "DOWN" ? "bg-red-900/50 text-red-400 border-red-800" : f === "HOLD" ? "bg-yellow-900/50 text-yellow-400 border-yellow-800" : "bg-cyan-900/50 text-cyan-400 border-cyan-800") : "text-gray-500 border-gray-700"}`}>
                  {f === "ALL" ? "전체" : f}
                </button>
              ))}
            </div>
            <div className="flex gap-1">
              {["ALL", "HIT", "MISS"].map(f => (
                <button key={f} onClick={() => setFilterHit(f)}
                  className={`px-2 py-0.5 rounded text-xs border transition ${filterHit === f ? "bg-cyan-900/50 text-cyan-400 border-cyan-800" : "text-gray-500 border-gray-700"}`}>
                  {f === "ALL" ? "전체" : f === "HIT" ? "✅ 적중" : "❌ 실패"}
                </button>
              ))}
            </div>
            <span className="text-gray-500 text-xs ml-auto">{filtered.length}건 표시</span>
          </div>
        )}

        {/* ═══ Day-by-day Summary ═══ */}
        {daySummaries.length > 0 && (
          <section className="bg-[#111827] rounded-xl border border-gray-800 p-5">
            <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
              <Target size={16} className="text-cyan-400" /> 일별 적중률
            </h3>
            <div className="space-y-1.5 max-h-72 overflow-y-auto">
              {daySummaries.map(day => {
                const accColor = day.accuracy >= 0.5 ? "border-l-green-500" : day.accuracy >= 0.35 ? "border-l-yellow-500" : "border-l-red-500";
                const accText = day.accuracy >= 0.5 ? "text-green-400" : day.accuracy >= 0.35 ? "text-yellow-400" : "text-red-400";
                return (
                  <div key={day.date} className={`flex items-center justify-between bg-[#0d1b2a] rounded-lg p-2.5 border border-gray-800 border-l-4 ${accColor}`}>
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-white text-xs">{day.date}</span>
                      <span className="text-gray-600 text-xs">{day.total}건</span>
                    </div>
                    <div className="flex items-center gap-5">
                      <span className={`font-bold ${accText}`}>{(day.accuracy * 100).toFixed(0)}%</span>
                      <span className="text-green-400 text-xs">UP {day.up_hits}/{day.up_preds}</span>
                      <span className="text-red-400 text-xs">DN {day.down_hits}/{day.down_preds}</span>
                      <span className={`font-mono text-xs ${day.avg_return > 0 ? "text-green-400" : "text-red-400"}`}>
                        {day.avg_return > 0 ? "+" : ""}{day.avg_return.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* ═══ Detail Table ═══ */}
        {filtered.length > 0 && (
          <section className="bg-[#111827] rounded-xl border border-gray-800 p-5">
            <h3 className="text-sm font-bold text-white mb-3">상세 결과</h3>
            <div className="overflow-x-auto max-h-96 overflow-y-auto">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-[#0d1b2a]">
                  <tr className="text-gray-400 border-b border-gray-700">
                    <th className="text-left py-2 px-2">날짜</th>
                    <th className="text-left py-2 px-2">종목</th>
                    <th className="text-center py-2 px-2">예측</th>
                    <th className="text-center py-2 px-2">실제</th>
                    <th className="text-center py-2 px-2">적중</th>
                    <th className="text-right py-2 px-2">UP%</th>
                    <th className="text-right py-2 px-2">DN%</th>
                    <th className="text-right py-2 px-2">수익률</th>
                    <th className="text-right py-2 px-2">기준가</th>
                    <th className="text-right py-2 px-2">결과가</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((r, i) => (
                    <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                      <td className="py-1.5 px-2 font-mono text-gray-400">{r.sim_date}</td>
                      <td className="py-1.5 px-2">{r.name || r.ticker}</td>
                      <td className="py-1.5 px-2 text-center"><LabelBadge label={r.pred_label} /></td>
                      <td className="py-1.5 px-2 text-center"><LabelBadge label={r.actual_label} /></td>
                      <td className="py-1.5 px-2 text-center">
                        {r.hit ? <CheckCircle size={14} className="text-green-400 inline" /> : <XCircle size={14} className="text-red-400 inline" />}
                      </td>
                      <td className="py-1.5 px-2 text-right text-green-400">{(r.pred_up_prob * 100).toFixed(1)}%</td>
                      <td className="py-1.5 px-2 text-right text-red-400">{(r.pred_down_prob * 100).toFixed(1)}%</td>
                      <td className={`py-1.5 px-2 text-right font-mono font-bold ${r.return_pct > 0 ? "text-green-400" : "text-red-400"}`}>
                        {r.return_pct > 0 ? "+" : ""}{r.return_pct}%
                      </td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-500">{r.base_close?.toLocaleString()}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-500">{r.actual_close?.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Empty state */}
        {results.length === 0 && !loading && (
          <div className="text-center py-16 text-gray-600">
            <Target size={48} className="mx-auto mb-4" />
            <p className="text-lg text-gray-400">기간과 종목을 선택하고 &quot;백테스트 실행&quot;을 클릭하세요</p>
            <p className="text-sm mt-2">AI 모델이 과거 각 날짜의 데이터로 예측 → 실제 결과와 비교합니다</p>
            <p className="text-xs mt-4">예측치 = 해당일 기준 AI 모델 예측 (UP/DOWN/HOLD)</p>
            <p className="text-xs">실측치 = 다음 거래일 종가 기준 실제 등락 (&gt;2%=UP, &lt;-2%=DOWN)</p>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  const tc = color === "green" ? "text-green-400" : color === "red" ? "text-red-400" : "text-white";
  return (
    <div className="bg-[#111827] rounded-xl border border-gray-800 p-3 text-center">
      <p className="text-gray-500 text-[10px]">{label}</p>
      <p className={`text-lg font-bold ${tc}`}>{value}</p>
      {sub && <p className="text-gray-500 text-[10px]">{sub}</p>}
    </div>
  );
}

function LabelBadge({ label }: { label: string }) {
  const cls = label === "UP" ? "bg-green-900/50 text-green-400 border-green-700" : label === "DOWN" ? "bg-red-900/50 text-red-400 border-red-700" : "bg-yellow-900/50 text-yellow-400 border-yellow-700";
  return <span className={`${cls} px-1.5 py-0.5 rounded text-[10px] font-medium border`}>{label}</span>;
}
