"use client";

import { useEffect, useState } from "react";
import { TrendingUp, TrendingDown } from "lucide-react";

const API = "http://localhost:8000/api";

interface Prediction {
  ticker: string; name: string; layer: string; pred_label: string;
  pred_up_prob: number; pred_down_prob: number; total_score: number;
  xgb_signal: string; lgb_signal: string;
}

export default function SignalsPage() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [horizon, setHorizon] = useState(1);
  const [modelType, setModelType] = useState<"modelA" | "modelB">("modelA");
  const [filterLabel, setFilterLabel] = useState("ALL");
  const [filterLayer, setFilterLayer] = useState("ALL");
  const [sortKey, setSortKey] = useState("total_score");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    const model = modelType === "modelA" ? "model_A" : "model_B";
    const url = `${API}/predictions-binary?horizon=${horizon}&model=${model}`;
    fetch(url)
      .then(r => r.json())
      .then(data => { setPredictions(data.predictions || []); setLoading(false); });
  }, [horizon, modelType]);

  const handleSort = (key: string) => {
    if (sortKey === key) setSortDir(sortDir === "desc" ? "asc" : "desc");
    else { setSortKey(key); setSortDir("desc"); }
  };

  const layers = [...new Set(predictions.map(p => p.layer).filter(Boolean))].sort();

  const filtered = predictions
    .filter(p => filterLabel === "ALL" || p.pred_label === filterLabel)
    .filter(p => filterLayer === "ALL" || p.layer === filterLayer)
    .sort((a, b) => {
      const av = (a as unknown as Record<string, unknown>)[sortKey];
      const bv = (b as unknown as Record<string, unknown>)[sortKey];
      if (av == null) return 1; if (bv == null) return -1;
      if (typeof av === "string") return sortDir === "asc" ? (av as string).localeCompare(bv as string) : (bv as string).localeCompare(av as string);
      return sortDir === "asc" ? (av as number) - (bv as number) : (bv as number) - (av as number);
    });

  const buy = predictions.filter(p => p.pred_label === "UP").sort((a, b) => b.total_score - a.total_score).slice(0, 5);
  const sell = predictions.filter(p => p.pred_label === "DOWN").sort((a, b) => b.pred_down_prob - a.pred_down_prob).slice(0, 5);

  return (
    <div className="min-h-screen bg-[#0a0f1a] text-white">
      <header className="border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Signals</h2>
          <p className="text-gray-500 text-sm">
            {modelType === "modelA" ? "모델 A (07:30) — 전일종가 기준 오늘 예측" : "모델 B (16:00) — 당일종가 기준 내일 예측"}
          </p>
        </div>
        <div className="flex gap-3">
          <div className="flex gap-1">
            <button onClick={() => setModelType("modelA")}
              className={`px-3 py-1.5 rounded-lg text-xs border transition ${modelType === "modelA" ? "bg-green-900/50 text-green-400 border-green-800" : "text-gray-400 border-gray-700 hover:bg-gray-800"}`}>
              Model A (07:30 오전)
            </button>
            <button onClick={() => setModelType("modelB")}
              className={`px-3 py-1.5 rounded-lg text-xs border transition ${modelType === "modelB" ? "bg-purple-900/50 text-purple-400 border-purple-800" : "text-gray-400 border-gray-700 hover:bg-gray-800"}`}>
              Model B (16:00 오후)
            </button>
          </div>
          <div className="flex gap-1">
            {[1, 5, 20].map(h => (
              <button key={h} onClick={() => setHorizon(h)}
                className={`px-3 py-1.5 rounded-lg text-sm border transition ${horizon === h ? "bg-cyan-900/50 text-cyan-400 border-cyan-800" : "text-gray-400 border-gray-700 hover:bg-gray-800"}`}>
                T+{h}
              </button>
            ))}
          </div>
        </div>
      </header>

      <div className="p-6 space-y-6">
        {/* TOP signals */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <section className="bg-[#111827] rounded-xl border border-gray-800 p-5">
            <h3 className="text-md font-bold text-green-400 mb-3 flex items-center gap-2"><TrendingUp size={18}/> 매수 TOP 5</h3>
            <div className="space-y-2">
              {buy.map((s, i) => (
                <div key={i} className="flex justify-between items-center bg-[#0d1b2a] rounded-lg p-3 border-l-4 border-l-green-500 border border-gray-800">
                  <div>
                    <p className="font-medium text-sm">{s.name || s.ticker}</p>
                    <p className="text-[10px] text-gray-500">{s.layer?.replace("_"," ")}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-green-400 font-bold">{(s.pred_up_prob*100).toFixed(1)}%</p>
                    <p className="text-[10px] text-gray-500">Score {s.total_score?.toFixed(3)}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
          <section className="bg-[#111827] rounded-xl border border-gray-800 p-5">
            <h3 className="text-md font-bold text-red-400 mb-3 flex items-center gap-2"><TrendingDown size={18}/> 매도 TOP 5</h3>
            <div className="space-y-2">
              {sell.map((s, i) => (
                <div key={i} className="flex justify-between items-center bg-[#0d1b2a] rounded-lg p-3 border-l-4 border-l-red-500 border border-gray-800">
                  <div>
                    <p className="font-medium text-sm">{s.name || s.ticker}</p>
                    <p className="text-[10px] text-gray-500">{s.layer?.replace("_"," ")}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-red-400 font-bold">{(s.pred_down_prob*100).toFixed(1)}%</p>
                    <p className="text-[10px] text-gray-500">Score {s.total_score?.toFixed(3)}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* Full Table */}
        <section className="bg-[#111827] rounded-xl border border-gray-800 p-5">
          <div className="flex flex-wrap items-center gap-3 mb-4">
            <h3 className="text-md font-bold">전체 종목</h3>
            <div className="flex gap-1">
              {["ALL","UP","DOWN"].map(f => (
                <button key={f} onClick={() => setFilterLabel(f)}
                  className={`px-2 py-1 rounded text-xs border transition ${filterLabel===f ? (f==="UP"?"bg-green-900/50 text-green-400 border-green-800":f==="DOWN"?"bg-red-900/50 text-red-400 border-red-800":f==="HOLD"?"bg-yellow-900/50 text-yellow-400 border-yellow-800":"bg-cyan-900/50 text-cyan-400 border-cyan-800") : "text-gray-500 border-gray-700 hover:bg-gray-800"}`}>
                  {f}
                </button>
              ))}
            </div>
            <select value={filterLayer} onChange={e => setFilterLayer(e.target.value)}
              className="bg-[#0d1b2a] border border-gray-700 rounded-lg px-2 py-1 text-xs text-white">
              <option value="ALL">전체 레이어</option>
              {layers.map(l => <option key={l} value={l}>{l?.replace("_"," ")}</option>)}
            </select>
            <span className="text-gray-500 text-xs ml-auto">{filtered.length}건</span>
          </div>

          {loading ? <p className="text-gray-500 animate-pulse">Loading...</p> : (
            <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-[#0d1b2a]">
                  <tr className="text-gray-400 border-b border-gray-700">
                    {[
                      {label:"종목",key:"name",align:"left"},{label:"레이어",key:"layer",align:"left"},
                      {label:"예측",key:"pred_label",align:"center"},{label:"UP%",key:"pred_up_prob",align:"right"},
                      {label:"DOWN%",key:"pred_down_prob",align:"right"},{label:"Score",key:"total_score",align:"right"},
                      {label:"XGB",key:"xgb_signal",align:"center"},{label:"LGB",key:"lgb_signal",align:"center"},
                    ].map(col => (
                      <th key={col.key} onClick={() => handleSort(col.key)}
                        className={`py-2 px-3 cursor-pointer select-none hover:text-white transition ${col.align==="right"?"text-right":col.align==="center"?"text-center":"text-left"} ${sortKey===col.key?"text-cyan-400":""}`}>
                        {col.label}{sortKey===col.key ? (sortDir==="desc"?" ▼":" ▲") : ""}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((p, i) => (
                    <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                      <td className="py-2 px-3 font-medium">{p.name||p.ticker}</td>
                      <td className="py-2 px-3 text-gray-500 text-xs">{p.layer?.replace("_"," ")}</td>
                      <td className="py-2 px-3 text-center"><LabelBadge label={p.pred_label}/></td>
                      <td className="py-2 px-3 text-right text-green-400">{(p.pred_up_prob*100).toFixed(1)}%</td>
                      <td className="py-2 px-3 text-right text-red-400">{(p.pred_down_prob*100).toFixed(1)}%</td>
                      <td className="py-2 px-3 text-right font-mono text-cyan-300">{p.total_score?.toFixed(3)}</td>
                      <td className="py-2 px-3 text-center"><LabelBadge label={p.xgb_signal} small/></td>
                      <td className="py-2 px-3 text-center"><LabelBadge label={p.lgb_signal} small/></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function LabelBadge({ label, small = false }: { label: string; small?: boolean }) {
  const cls = label==="UP"?"bg-green-900/50 text-green-400 border-green-700":label==="DOWN"?"bg-red-900/50 text-red-400 border-red-700":"bg-yellow-900/50 text-yellow-400 border-yellow-700";
  return <span className={`${cls} ${small?"text-xs px-1.5 py-0.5":"text-xs px-2 py-1"} rounded border font-medium`}>{label||"-"}</span>;
}
