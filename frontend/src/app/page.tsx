"use client";

import { useEffect, useState } from "react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

const API = "http://localhost:8000/api";

interface Macro {
  kospi: number; kosdaq: number; sp500: number; nasdaq: number;
  sox: number; vix: number; us_10y: number; usd_krw: number;
}
interface Summary { total: number; up: number; hold: number; down: number; pred_date: string; }
interface Prediction { ticker: string; name: string; layer: string; pred_label: string; }

export default function OverviewPage() {
  const [macro, setMacro] = useState<Macro | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/macro`).then(r => r.json()),
      fetch(`${API}/predictions-binary?horizon=1&model=model_A`).then(r => r.json()),
    ]).then(([m, p]) => {
      setMacro(m);
      setSummary(p.summary);
      setPredictions(p.predictions || []);
    });
  }, []);

  // Layer summary
  const layerStats = predictions.reduce<Record<string, { up: number; down: number; hold: number; total: number }>>((acc, p) => {
    const l = p.layer || "Unknown";
    if (!acc[l]) acc[l] = { up: 0, down: 0, hold: 0, total: 0 };
    acc[l].total++;
    if (p.pred_label === "UP") acc[l].up++;
    else if (p.pred_label === "DOWN") acc[l].down++;
    else acc[l].hold++;
    return acc;
  }, {});

  return (
    <div className="min-h-screen bg-[#0a0f1a] text-white">
      <header className="border-b border-gray-800 px-6 py-4">
        <h2 className="text-xl font-bold">Overview</h2>
        <p className="text-gray-500 text-sm">시장 전체 조감도 · {summary?.pred_date}</p>
      </header>

      <div className="p-6 space-y-6">
        {/* Macro */}
        <section className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
          {[
            { label: "KOSPI", value: macro?.kospi },
            { label: "KOSDAQ", value: macro?.kosdaq },
            { label: "S&P 500", value: macro?.sp500 },
            { label: "NASDAQ", value: macro?.nasdaq },
            { label: "SOX", value: macro?.sox },
            { label: "VIX", value: macro?.vix, color: macro?.vix && macro.vix > 20 ? "text-red-400" : "text-green-400" },
            { label: "US 10Y", value: macro?.us_10y, suffix: "%" },
            { label: "USD/KRW", value: macro?.usd_krw },
          ].map((m) => (
            <div key={m.label} className="bg-[#111827] rounded-lg border border-gray-800 p-3 text-center">
              <p className="text-gray-500 text-xs">{m.label}</p>
              <p className={`font-bold text-sm ${m.color || "text-white"}`}>
                {m.value != null ? m.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "N/A"}{m.suffix || ""}
              </p>
            </div>
          ))}
        </section>

        {/* Signal Summary */}
        <section className="grid grid-cols-3 gap-4">
          <div className="bg-[#111827] rounded-xl border border-green-900/50 p-5 text-center">
            <TrendingUp className="mx-auto text-green-400 mb-2" size={28} />
            <p className="text-3xl font-bold text-green-400">{summary?.up || 0}</p>
            <p className="text-gray-400 text-sm">UP 시그널</p>
          </div>
          <div className="bg-[#111827] rounded-xl border border-yellow-900/50 p-5 text-center">
            <Minus className="mx-auto text-yellow-400 mb-2" size={28} />
            <p className="text-3xl font-bold text-yellow-400">{summary?.hold || 0}</p>
            <p className="text-gray-400 text-sm">HOLD</p>
          </div>
          <div className="bg-[#111827] rounded-xl border border-red-900/50 p-5 text-center">
            <TrendingDown className="mx-auto text-red-400 mb-2" size={28} />
            <p className="text-3xl font-bold text-red-400">{summary?.down || 0}</p>
            <p className="text-gray-400 text-sm">DOWN 시그널</p>
          </div>
        </section>

        {/* Layer Heatmap */}
        <section className="bg-[#111827] rounded-xl border border-gray-800 p-5">
          <h3 className="text-lg font-bold mb-4">레이어별 시그널 분포</h3>
          <div className="space-y-2">
            {Object.entries(layerStats).sort((a, b) => b[1].up - a[1].up).map(([layer, stats]) => {
              const upPct = stats.total > 0 ? (stats.up / stats.total) * 100 : 0;
              const dnPct = stats.total > 0 ? (stats.down / stats.total) * 100 : 0;
              return (
                <div key={layer} className="flex items-center gap-3 bg-[#0d1b2a] rounded-lg p-3 border border-gray-800">
                  <span className="text-sm text-white w-52 truncate">{layer.replace("_", " ")}</span>
                  <div className="flex-1 h-5 bg-gray-800 rounded-full overflow-hidden flex">
                    <div className="bg-green-600 h-full" style={{ width: `${upPct}%` }} />
                    <div className="bg-yellow-600 h-full" style={{ width: `${100 - upPct - dnPct}%` }} />
                    <div className="bg-red-600 h-full" style={{ width: `${dnPct}%` }} />
                  </div>
                  <span className="text-xs text-gray-400 w-32 text-right">
                    <span className="text-green-400">{stats.up}↑</span>{" "}
                    <span className="text-yellow-400">{stats.hold}→</span>{" "}
                    <span className="text-red-400">{stats.down}↓</span>
                  </span>
                </div>
              );
            })}
          </div>
        </section>
      </div>
    </div>
  );
}
