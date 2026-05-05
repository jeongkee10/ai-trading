"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Globe, TrendingUp, Search, Target, Settings, Zap } from "lucide-react";

const menuItems = [
  { href: "/", label: "Overview", icon: Globe, desc: "시장 전체 조감도" },
  { href: "/signals", label: "Signals", icon: TrendingUp, desc: "오늘의 매매 판단" },
  { href: "/analysis", label: "Analysis", icon: Search, desc: "개별 종목 심층 분석" },
  { href: "/backtest", label: "Backtest", icon: Target, desc: "예측 vs 실측 검증" },
  { href: "/system", label: "System", icon: Settings, desc: "시스템 상태" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-full w-60 bg-[#0d1117] border-r border-gray-800 flex flex-col z-50">
      <div className="px-5 py-5 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <Zap className="text-cyan-400" size={22} />
          <div>
            <h1 className="text-lg font-bold text-white leading-tight">J.Insight</h1>
            <p className="text-[10px] text-gray-500 leading-tight">AI Trading System</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {menuItems.map((item) => {
          const active = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link key={item.href} href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
                active
                  ? "bg-cyan-900/30 text-cyan-400 border border-cyan-800/50"
                  : "text-gray-400 hover:bg-gray-800/50 hover:text-white border border-transparent"
              }`}>
              <Icon size={18} />
              <div>
                <p className={`text-sm font-medium ${active ? "text-cyan-400" : ""}`}>{item.label}</p>
                <p className="text-[10px] text-gray-600">{item.desc}</p>
              </div>
            </Link>
          );
        })}
      </nav>

      <div className="px-5 py-4 border-t border-gray-800">
        <p className="text-[10px] text-gray-600">113 Stocks · 10 Layers</p>
        <p className="text-[10px] text-gray-600">XGBoost + LightGBM Ensemble</p>
      </div>
    </aside>
  );
}
