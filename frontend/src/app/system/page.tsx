"use client";

import { useEffect, useState } from "react";
import { CheckCircle, XCircle, Database, Cpu, Clock } from "lucide-react";

const API = "http://localhost:8000/api";

interface Status {
  stock_prices: number;
  predictions: number;
  macro_data: number;
  technical_indicators: number;
  latest_price_date: string;
  latest_pred_date: string;
}

export default function SystemPage() {
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch(`${API}/status`)
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => setError(true));
  }, []);

  return (
    <div className="min-h-screen bg-[#0a0f1a] text-white">
      <header className="border-b border-gray-800 px-6 py-4">
        <h2 className="text-xl font-bold text-white">System</h2>
        <p className="text-gray-500 text-sm">시스템 상태 및 데이터 현황</p>
      </header>

      <div className="p-6 space-y-6">
        {/* Connection Status */}
        <section className="bg-[#111827] rounded-xl border border-gray-800 p-5">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <Database size={18} className="text-cyan-400" /> 연결 상태
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatusItem
              label="PostgreSQL"
              ok={!error && !!status}
              detail="localhost:5432"
            />
            <StatusItem
              label="FastAPI Backend"
              ok={!error}
              detail="localhost:8000"
            />
            <StatusItem
              label="Next.js Frontend"
              ok={true}
              detail="localhost:3000"
            />
          </div>
        </section>

        {/* Data Stats */}
        {status && (
          <section className="bg-[#111827] rounded-xl border border-gray-800 p-5">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <Cpu size={18} className="text-cyan-400" /> 데이터 현황
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <DataCard label="주가 데이터" value={status.stock_prices.toLocaleString()} unit="건" />
              <DataCard label="예측 결과" value={status.predictions.toLocaleString()} unit="건" />
              <DataCard label="기술지표" value={status.technical_indicators.toLocaleString()} unit="건" />
              <DataCard label="거시경제" value={status.macro_data.toLocaleString()} unit="건" />
            </div>
          </section>
        )}

        {/* Timestamps */}
        {status && (
          <section className="bg-[#111827] rounded-xl border border-gray-800 p-5">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <Clock size={18} className="text-cyan-400" /> 최신 기준일
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-[#0d1b2a] rounded-lg p-4 border border-gray-800">
                <p className="text-gray-400 text-sm">최신 주가</p>
                <p className="text-white font-mono text-lg">{status.latest_price_date}</p>
              </div>
              <div className="bg-[#0d1b2a] rounded-lg p-4 border border-gray-800">
                <p className="text-gray-400 text-sm">최신 예측</p>
                <p className="text-white font-mono text-lg">{status.latest_pred_date}</p>
              </div>
            </div>
          </section>
        )}

        {/* Architecture */}
        <section className="bg-[#111827] rounded-xl border border-gray-800 p-5">
          <h3 className="text-lg font-bold text-white mb-4">시스템 구조</h3>
          <div className="font-mono text-xs text-gray-400 leading-relaxed whitespace-pre">
{`┌─────────────────────────────────────────────────────────────┐
│  Daily Batch (07:00 KST)                                     │
│  네이버금융 + yfinance + FRED → 113종목 데이터 수집           │
│  → 기술지표 계산 → XGBoost + LightGBM 예측 → DB 저장         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  PostgreSQL (port 5432)                                       │
│  stock_prices · predictions · macro_data · technical_indicators│
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────┐     ┌───────────────────────────────────┐
│  FastAPI (:8000)  │────▶│  Next.js + React + TailwindCSS    │
│  REST API         │     │  localhost:3000                    │
└──────────────────┘     └───────────────────────────────────┘`}
          </div>
        </section>
      </div>
    </div>
  );
}

function StatusItem({ label, ok, detail }: { label: string; ok: boolean; detail: string }) {
  return (
    <div className="flex items-center gap-3 bg-[#0d1b2a] rounded-lg p-3 border border-gray-800">
      {ok ? <CheckCircle className="text-green-400" size={20} /> : <XCircle className="text-red-400" size={20} />}
      <div>
        <p className="text-white font-medium text-sm">{label}</p>
        <p className="text-gray-500 text-xs">{detail}</p>
      </div>
    </div>
  );
}

function DataCard({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div className="bg-[#0d1b2a] rounded-lg p-4 border border-gray-800 text-center">
      <p className="text-gray-400 text-sm">{label}</p>
      <p className="text-white font-bold text-xl">{value}</p>
      <p className="text-gray-600 text-xs">{unit}</p>
    </div>
  );
}
