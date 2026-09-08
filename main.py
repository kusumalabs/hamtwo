from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Rule-Based Stock Analytics", page_icon="📈", layout="wide")

PROVIDER = "Yahoo Finance melalui yfinance"
DISCLAIMER = (
    "Aplikasi ini merupakan alat bantu analisis berbasis data pasar, formula statistik, "
    "indikator teknikal, dan aturan yang telah ditentukan. Hasil analisis dan proyeksi harga "
    "bersifat estimasi, bukan kepastian, rekomendasi investasi, maupun ajakan untuk membeli "
    "atau menjual efek. Data dapat mengalami keterlambatan, ketidaklengkapan, atau perubahan "
    "sesuai sumber data. Pengguna bertanggung jawab sepenuhnya atas keputusan investasi yang dilakukan."
)
NO_DUMMY = (
    "Seluruh data pasar pada aplikasi diambil dari sumber eksternal. Aplikasi tidak menggunakan "
    "data dummy, data sintetis, atau data hasil simulasi sebagai pengganti data pasar."
)
PERIODS = ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
INTERVALS = ["1m", "5m", "15m", "30m", "60m", "1d", "1wk"]
HORIZONS = [1, 3, 5, 10, 20]
BASE_WEIGHTS = {"Trend": 0.25, "Momentum": 0.20, "Volume": 0.20, "Price Action": 0.15, "Benchmark": 0.10, "Fundamental": 0.10}
MIN_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
INTRADAY = {"1m", "5m", "15m", "30m", "60m"}


@dataclass
class GroupScore:
    name: str
    score: float
    available: bool
    reasons: list[dict[str, Any]]
    unavailable_reason: Optional[str] = None


@dataclass
class Projection:
    bearish: float
    base: float
    bullish: float
    invalidation: Optional[float]
    range_low: float
    range_high: float
    risk_reward: Optional[float]
    atr_multiplier: float


def finite(value: Any) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def number(value: Any) -> Optional[float]:
    return float(value) if finite(value) else None


def fmt(value: Any, decimals: int = 2, suffix: str = "") -> str:
    return f"{float(value):,.{decimals}f}{suffix}" if finite(value) else "N/A"


def safe_div(a: Any, b: Any) -> float:
    return float(a) / float(b) if finite(a) and finite(b) and float(b) != 0 else np.nan


def validate_combo(period: str, interval: str) -> tuple[bool, str]:
    limits = {"1m": {"5d"}, "5m": {"5d", "1mo"}, "15m": {"5d", "1mo"}, "30m": {"5d", "1mo"}, "60m": {"5d", "1mo", "3mo", "6mo", "1y", "2y"}}
    if interval in limits and period not in limits[interval]:
        return False, f"Kombinasi periode {period} dan interval {interval} tidak didukung secara aman. Pilih periode yang lebih pendek."
    return True, ""


def normalize_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if ticker in out.columns.get_level_values(-1):
            out = out.xs(ticker, axis=1, level=-1, drop_level=True)
        elif ticker in out.columns.get_level_values(0):
            out = out.xs(ticker, axis=1, level=0, drop_level=True)
        else:
            out.columns = [str(c[0]) for c in out.columns]
    out.columns = [str(c).strip().title().replace("Adj Close", "Adj Close") for c in out.columns]
    out = out.loc[~out.index.duplicated(keep="last")].sort_index()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


@st.cache_data(ttl=180, show_spinner=False)
def fetch_stock_data(ticker: str, period: str, interval: str) -> tuple[pd.DataFrame, str, datetime]:
    fetched = datetime.now(timezone.utc)
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, actions=False, progress=False, threads=False, timeout=20)
        return normalize_yfinance_columns(df, ticker), "", fetched
    except Exception as exc:
        return pd.DataFrame(), f"Pengambilan data gagal: {type(exc).__name__}: {exc}", fetched


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ticker_information(ticker: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    try:
        obj = yf.Ticker(ticker)
        info: dict[str, Any] = {}
        fast: dict[str, Any] = {}
        try:
            info = dict(obj.info or {})
        except Exception:
            info = {}
        try:
            fast = dict(obj.fast_info or {})
        except Exception:
            fast = {}
        return info, fast, ""
    except Exception as exc:
        return {}, {}, f"Informasi ticker tidak tersedia: {type(exc).__name__}: {exc}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamental_data(ticker: str) -> tuple[dict[str, Any], dict[str, pd.DataFrame], str]:
    try:
        obj = yf.Ticker(ticker)
        info = dict(obj.info or {})
        statements: dict[str, pd.DataFrame] = {}
        for name, getter in {
            "Income Statement": lambda: obj.income_stmt,
            "Balance Sheet": lambda: obj.balance_sheet,
            "Cash Flow": lambda: obj.cashflow,
            "Quarterly Income": lambda: obj.quarterly_income_stmt,
            "Quarterly Balance": lambda: obj.quarterly_balance_sheet,
            "Quarterly Cash Flow": lambda: obj.quarterly_cashflow,
        }.items():
            try:
                value = getter()
                if isinstance(value, pd.DataFrame) and not value.empty:
                    statements[name] = value
            except Exception:
                continue
        return info, statements, ""
    except Exception as exc:
        return {}, {}, f"Data fundamental tidak tersedia: {type(exc).__name__}: {exc}"


def validate_market_data(df: pd.DataFrame) -> tuple[bool, str]:
    if df.empty:
        return False, "Data historis untuk ticker dan periode yang dipilih tidak tersedia dari sumber data."
    missing = [c for c in MIN_COLUMNS if c not in df.columns]
    if missing:
        return False, "Kolom minimum tidak lengkap: " + ", ".join(missing)
    valid = df[MIN_COLUMNS].dropna(subset=["Open", "High", "Low", "Close"])
    if valid.empty:
        return False, "Data OHLC tidak memiliki baris valid."
    if (valid[["Open", "High", "Low", "Close"]] <= 0).any().any():
        return False, "Data OHLC mengandung harga tidak valid."
    return True, ""


def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    o, h, l, c, v = (df[x].astype(float) for x in MIN_COLUMNS)
    for n in (20, 50, 200):
        df[f"SMA{n}"] = c.rolling(n, min_periods=n).mean()
    for n in (9, 20, 50):
        df[f"EMA{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    up, down = h.diff(), -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    df["TR"], df["ATR14"], df["ATR%"] = tr, atr, atr/c*100
    df["+DI"], df["-DI"], df["ADX14"] = plus_di, minus_di, dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain/loss.replace(0, np.nan)
    df["RSI14"] = 100-(100/(1+rs))
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    df["MACD"] = ema12-ema26
    df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["MACDHist"] = df["MACD"]-df["MACDSignal"]
    low14, high14 = l.rolling(14, min_periods=14).min(), h.rolling(14, min_periods=14).max()
    df["StochK"] = 100*(c-low14)/(high14-low14).replace(0, np.nan)
    df["StochD"] = df["StochK"].rolling(3, min_periods=3).mean()
    df["ROC10"] = c.pct_change(10, fill_method=None)*100
    mid = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    df["BBMid"], df["BBUpper"], df["BBLower"] = mid, mid+2*sd, mid-2*sd
    df["HistVol20"] = np.log(c/c.shift(1)).rolling(20, min_periods=20).std(ddof=1)*np.sqrt(252)*100
    df["AvgRange20%"] = ((h-l)/c.replace(0, np.nan)*100).rolling(20, min_periods=20).mean()
    df["VolumeSMA20"] = v.rolling(20, min_periods=20).mean()
    df["RelativeVolume"] = v/df["VolumeSMA20"].replace(0, np.nan)
    df["VolumeSpike"] = df["RelativeVolume"] >= 1.5
    direction = np.sign(c.diff()).fillna(0)
    df["OBV"] = (direction*v.fillna(0)).cumsum()
    mfm = ((c-l)-(h-c))/(h-l).replace(0, np.nan)
    df["ADL"] = (mfm.fillna(0)*v.fillna(0)).cumsum()
    typical = (h+l+c)/3
    money = typical*v
    pos = money.where(typical.diff() > 0, 0.0).rolling(14, min_periods=14).sum()
    neg = money.where(typical.diff() < 0, 0.0).rolling(14, min_periods=14).sum()
    df["MFI14"] = 100-(100/(1+pos/neg.replace(0, np.nan)))
    df["VPT"] = (c.pct_change(fill_method=None).fillna(0)*v.fillna(0)).cumsum()
    session = pd.Series(df.index.date, index=df.index) if interval_is_intraday(df.index) else pd.Series(0, index=df.index)
    pv = typical*v
    df["VWAP"] = pv.groupby(session).cumsum()/v.groupby(session).cumsum().replace(0, np.nan)
    return df.replace([np.inf, -np.inf], np.nan)


def interval_is_intraday(index: pd.Index) -> bool:
    if len(index) < 2:
        return False
    try:
        return pd.Series(index).diff().dropna().median() < pd.Timedelta(days=1)
    except Exception:
        return False


def price_action(df: pd.DataFrame) -> dict[str, Any]:
    work = df.copy()
    window = 5
    work["PivotHigh"] = work["High"].eq(work["High"].rolling(2*window+1, center=True).max())
    work["PivotLow"] = work["Low"].eq(work["Low"].rolling(2*window+1, center=True).min())
    confirmed = work.iloc[:-window] if len(work) > window else work.iloc[:0]
    highs = confirmed.loc[confirmed["PivotHigh"], "High"]
    lows = confirmed.loc[confirmed["PivotLow"], "Low"]
    support = number(lows.iloc[-1]) if len(lows) else None
    resistance = number(highs.iloc[-1]) if len(highs) else None
    close = number(work["Close"].iloc[-1])
    prev_high = number(work["High"].rolling(20, min_periods=20).max().shift(1).iloc[-1])
    prev_low = number(work["Low"].rolling(20, min_periods=20).min().shift(1).iloc[-1])
    if support is None: support = prev_low
    if resistance is None: resistance = prev_high
    recent_range = safe_div(work["High"].tail(10).max()-work["Low"].tail(10).min(), close)*100 if close else np.nan
    atrp = number(work["ATR%"].iloc[-1])
    consolidation = finite(recent_range) and finite(atrp) and recent_range < max(3*atrp, 2)
    breakout = close is not None and resistance is not None and close > resistance
    breakdown = close is not None and support is not None and close < support
    sh = highs.tail(2).tolist(); sl = lows.tail(2).tolist()
    structure = "Tidak cukup data"
    if len(sh) == 2 and len(sl) == 2:
        structure = "Higher High & Higher Low" if sh[-1] > sh[-2] and sl[-1] > sl[-2] else "Lower High & Lower Low" if sh[-1] < sh[-2] and sl[-1] < sl[-2] else "Mixed Structure"
    ds = safe_div(close-support, close)*100 if close and support else np.nan
    dr = safe_div(resistance-close, close)*100 if close and resistance else np.nan
    rr = safe_div(dr, ds) if finite(ds) and ds > 0 and finite(dr) else np.nan
    return {"support": support, "resistance": resistance, "breakout": breakout, "breakdown": breakdown, "consolidation": consolidation, "structure": structure, "distance_support": ds, "distance_resistance": dr, "risk_reward": rr}


def add_reason(reasons: list[dict[str, Any]], indicator: str, value: Any, rule: str, raw: float, interpretation: str) -> None:
    reasons.append({"Indicator": indicator, "Value": fmt(value), "Rule": rule, "Raw Score": raw, "Interpretation": interpretation, "Status": "Bullish" if raw > 0 else "Bearish" if raw < 0 else "Neutral"})


def score_groups(df: pd.DataFrame, pa: dict[str, Any], benchmark: Optional[dict[str, Any]], fundamentals: dict[str, Any]) -> list[GroupScore]:
    x, reasons = df.iloc[-1], []
    trend_parts = []
    if all(finite(x.get(k)) for k in ["Close", "EMA20", "EMA50"]):
        raw = 60 if x.Close > x.EMA20 > x.EMA50 else -60 if x.Close < x.EMA20 < x.EMA50 else 0
        trend_parts.append(raw); add_reason(reasons, "EMA20/EMA50", x.Close, "Close > EMA20 > EMA50 atau kebalikannya", raw, "Susunan moving average menunjukkan arah tren.")
    if all(finite(x.get(k)) for k in ["ADX14", "+DI", "-DI"]):
        raw = (30 if x["+DI"] > x["-DI"] else -30) if x.ADX14 >= 20 else 0
        trend_parts.append(raw); add_reason(reasons, "ADX14 dan DI", x.ADX14, "ADX >= 20, arah mengikuti DI", raw, "Mengukur kekuatan dan arah tren.")
    groups = [GroupScore("Trend", float(np.clip(np.mean(trend_parts), -100, 100)) if trend_parts else 0, bool(trend_parts), reasons, None if trend_parts else "Indikator tren belum tersedia")]
    reasons=[]; parts=[]
    if finite(x.get("RSI14")):
        r=float(x.RSI14); raw=35 if 50<=r<=70 else -20 if 30<=r<50 else -30 if r>75 else 20 if r<25 else 0
        parts.append(raw); add_reason(reasons,"RSI14",r,"50-70 bullish; 30-50 bearish ringan; ekstrem sebagai warning",raw,"Momentum relatif berdasarkan perubahan harga.")
    if finite(x.get("MACD")) and finite(x.get("MACDSignal")):
        raw=45 if x.MACD>x.MACDSignal else -45; parts.append(raw); add_reason(reasons,"MACD",x.MACD,"MACD dibanding signal",raw,"Arah momentum MACD.")
    if finite(x.get("StochK")) and finite(x.get("StochD")):
        raw=20 if x.StochK>x.StochD and x.StochK<80 else -20 if x.StochK<x.StochD and x.StochK>20 else 0
        parts.append(raw); add_reason(reasons,"Stochastic",x.StochK,"%K dibanding %D dengan batas ekstrem",raw,"Konfirmasi momentum jangka pendek.")
    groups.append(GroupScore("Momentum",float(np.clip(np.mean(parts),-100,100)) if parts else 0,bool(parts),reasons,None if parts else "Indikator momentum belum tersedia"))
    reasons=[]; parts=[]
    if finite(x.get("RelativeVolume")):
        rv=float(x.RelativeVolume); change=float(df.Close.pct_change(fill_method=None).iloc[-1]) if len(df)>1 else np.nan
        raw=(45 if change>0 else -45) if rv>=1.5 and finite(change) else (15 if change>0 else -15) if finite(change) else 0
        parts.append(raw); add_reason(reasons,"Relative Volume",rv,"Volume relatif dikombinasikan dengan arah candle",raw,"Analisis volume candle, bukan aggressive buy/sell.")
    if finite(x.get("OBV")) and len(df)>=6:
        raw=25 if x.OBV>df.OBV.iloc[-6] else -25; parts.append(raw); add_reason(reasons,"OBV",x.OBV,"OBV dibanding lima candle sebelumnya",raw,"Konfirmasi akumulasi volume berbasis candle.")
    if finite(x.get("VWAP")):
        raw=20 if x.Close>x.VWAP else -20; parts.append(raw); add_reason(reasons,"VWAP",x.VWAP,"Close dibanding VWAP",raw,"Posisi harga terhadap harga rata-rata berbobot volume.")
    groups.append(GroupScore("Volume",float(np.clip(np.mean(parts),-100,100)) if parts else 0,bool(parts),reasons,None if parts else "Data volume tidak mencukupi"))
    reasons=[]; parts=[]
    if pa["breakout"]:
        raw=80 if finite(x.get("RelativeVolume")) and x.RelativeVolume>=1.5 else 50; parts.append(raw); add_reason(reasons,"Breakout",x.Close,"Close menembus resistance",raw,"Harga menembus area resistance aktual.")
    elif pa["breakdown"]:
        raw=-80 if finite(x.get("RelativeVolume")) and x.RelativeVolume>=1.5 else -50; parts.append(raw); add_reason(reasons,"Breakdown",x.Close,"Close turun menembus support",raw,"Harga menembus area support aktual.")
    if pa["structure"] != "Tidak cukup data":
        raw=40 if pa["structure"].startswith("Higher") else -40 if pa["structure"].startswith("Lower") else 0
        parts.append(raw); add_reason(reasons,"Market Structure",np.nan,pa["structure"],raw,"Struktur swing high dan swing low terkonfirmasi.")
    groups.append(GroupScore("Price Action",float(np.clip(np.mean(parts),-100,100)) if parts else 0,bool(parts),reasons,None if parts else "Swing atau level belum mencukupi"))
    reasons=[]
    if benchmark and benchmark.get("available"):
        rs=benchmark.get("relative_strength"); corr=benchmark.get("correlation"); btrend=benchmark.get("trend")
        parts=[]
        if finite(rs):
            raw=55 if rs>0 else -55; parts.append(raw); add_reason(reasons,"Relative Strength",rs,"Return saham dikurangi return benchmark",raw,"Kinerja relatif terhadap benchmark.")
        if btrend in ("Bullish","Bearish"):
            raw=25 if btrend=="Bullish" else -25; parts.append(raw); add_reason(reasons,"Benchmark Trend",np.nan,btrend,raw,"Konteks tren pasar.")
        groups.append(GroupScore("Benchmark",float(np.clip(np.mean(parts),-100,100)) if parts else 0,bool(parts),reasons,None if parts else "Metrik benchmark tidak cukup"))
    else:
        groups.append(GroupScore("Benchmark",0,False,[],"Benchmark tidak tersedia"))
    reasons=[]; parts=[]
    fmap=[("trailingPE",lambda z: 25 if 0<z<=25 else -15 if z>40 else 0,"Trailing P/E"),("priceToBook",lambda z: 20 if 0<z<=3 else -10 if z>6 else 0,"Price to Book"),("returnOnEquity",lambda z: 30 if z>=0.15 else -20 if z<0 else 0,"Return on Equity"),("profitMargins",lambda z: 25 if z>0 else -30,"Profit Margin"),("revenueGrowth",lambda z: 30 if z>0 else -25,"Revenue Growth"),("earningsGrowth",lambda z: 35 if z>0 else -30,"Earnings Growth")]
    for key,rule,label in fmap:
        val=number(fundamentals.get(key))
        if val is not None:
            raw=rule(val); parts.append(raw); add_reason(reasons,label,val,"Rule fundamental transparan sesuai nilai tersedia",raw,"Data fundamental dari provider.")
    groups.append(GroupScore("Fundamental",float(np.clip(np.mean(parts),-100,100)) if parts else 0,bool(parts),reasons,None if parts else "Data fundamental tidak disediakan untuk scoring"))
    return groups


def combine_scores(groups: list[GroupScore]) -> tuple[float, dict[str,float], pd.DataFrame, list[str]]:
    available=[g for g in groups if g.available]
    denom=sum(BASE_WEIGHTS[g.name] for g in available)
    weights={g.name:(BASE_WEIGHTS[g.name]/denom if denom else 0.0) for g in available}
    rows=[]
    for g in groups:
        eff=weights.get(g.name,0.0); contribution=g.score*eff
        rows.append({"Group":g.name,"Available":g.available,"Base Weight":BASE_WEIGHTS[g.name],"Effective Weight":eff,"Score":g.score if g.available else np.nan,"Weighted Contribution":contribution if g.available else np.nan,"Reason":g.unavailable_reason or "Tersedia"})
    total=float(np.clip(sum(g.score*weights.get(g.name,0) for g in available),-100,100)) if denom else 0.0
    return total,weights,pd.DataFrame(rows),[g.name for g in groups if not g.available]


def classify(score: float) -> str:
    return "Strong Bullish" if score>=60 else "Bullish" if score>=20 else "Strong Bearish" if score<=-60 else "Bearish" if score<=-20 else "Sideways"


def benchmark_analysis(stock: pd.DataFrame, bench: pd.DataFrame) -> dict[str,Any]:
    if bench.empty or "Close" not in bench: return {"available":False}
    joined=pd.concat([stock.Close.rename("stock"),bench.Close.rename("bench")],axis=1).dropna()
    if len(joined)<20: return {"available":False}
    ret=joined.pct_change(fill_method=None).dropna(); window=min(60,len(joined)-1)
    sr=joined.stock.iloc[-1]/joined.stock.iloc[-window-1]-1; br=joined.bench.iloc[-1]/joined.bench.iloc[-window-1]-1
    bs20=joined.bench.rolling(20,min_periods=20).mean().iloc[-1]; bs50=joined.bench.rolling(50,min_periods=50).mean().iloc[-1]
    trend="Bullish" if finite(bs20) and finite(bs50) and joined.bench.iloc[-1]>bs20>bs50 else "Bearish" if finite(bs20) and finite(bs50) and joined.bench.iloc[-1]<bs20<bs50 else "Sideways"
    return {"available":True,"stock_return":sr*100,"benchmark_return":br*100,"relative_strength":(sr-br)*100,"correlation":ret.stock.corr(ret.bench),"trend":trend,"aligned":len(joined)}


def confidence(score: float, groups: list[GroupScore], df: pd.DataFrame, fetched: datetime, interval: str, bidask: bool) -> dict[str,float]:
    vals=[g.score for g in groups if g.available]
    signs=[np.sign(v) for v in vals if abs(v)>=10]
    agreement=(max(signs.count(1),signs.count(-1))/len(signs)*100) if signs else 50
    completeness=len(vals)/len(BASE_WEIGHTS)*100
    age=(datetime.now(timezone.utc)-fetched).total_seconds()
    freshness=max(0,100-age/(300 if interval in INTRADAY else 3600)*25)
    signal=min(100,abs(score)*1.25)
    rv=number(df.RelativeVolume.iloc[-1]); liquidity=min(100,max(20,(rv or 0.2)*50)) if finite(df.Volume.iloc[-1]) and df.Volume.iloc[-1]>0 else 20
    atrp=number(df["ATR%"].iloc[-1]); vol_penalty=min(35,max(0,(atrp or 0)-2)*7)
    conflict=100-agreement
    raw=0.25*agreement+0.20*completeness+0.15*freshness+0.20*signal+0.20*liquidity-vol_penalty*0.5-conflict*0.15
    if not bidask: raw-=3
    return {"Agreement score":agreement,"Data completeness":completeness,"Data freshness":freshness,"Signal strength":signal,"Liquidity quality":liquidity,"Volatility penalty":vol_penalty,"Conflict penalty":conflict,"Signal Confidence":float(np.clip(raw,0,100))}


def make_projection(df: pd.DataFrame, score: float, horizon: int, pa: dict[str,Any]) -> Optional[Projection]:
    last=number(df.Close.iloc[-1]); atr=number(df.ATR14.iloc[-1])
    if last is None or atr is None or atr<=0: return None
    mult=float(np.sqrt(horizon)); span=atr*mult
    low=max(0.000001,last-span); high=last+span
    tilt=np.clip(score/100,-1,1)*0.45*span; base=float(np.clip(last+tilt,low+span*0.05,high-span*0.05))
    bearish=low; bullish=high
    if finite(pa.get("support")) and pa["support"]<last: bearish=min(base-0.01*span,max(low,float(pa["support"])))
    if finite(pa.get("resistance")) and pa["resistance"]>last: bullish=max(base+0.01*span,min(high,float(pa["resistance"])))
    if not bearish<base<bullish: bearish,base,bullish=low,last,high
    invalidation=pa.get("support") if score>=0 else pa.get("resistance")
    reward=(bullish-last) if score>=0 else (last-bearish); risk=(last-invalidation) if score>=0 and finite(invalidation) else (invalidation-last) if score<0 and finite(invalidation) else np.nan
    return Projection(bearish,base,bullish,number(invalidation),low,high,number(safe_div(reward,risk)),mult)


def risk_analysis(df: pd.DataFrame, pa: dict[str,Any], groups: list[GroupScore]) -> tuple[float,str,list[str],dict[str,Any]]:
    ret=df.Close.pct_change(fill_method=None).dropna(); peak=df.Close.cummax(); mdd=number(((df.Close/peak)-1).min()*100)
    downside=number(ret[ret<0].std(ddof=1)*np.sqrt(252)*100) if len(ret[ret<0])>1 else None
    atrp=number(df["ATR%"].iloc[-1]); hv=number(df.HistVol20.iloc[-1]); rv=number(df.RelativeVolume.iloc[-1])
    conflicts=len({np.sign(g.score) for g in groups if g.available and abs(g.score)>=20})>1
    points=0; factors=[]
    if atrp is not None:
        points+=min(30,atrp*6)
        if atrp>=4: factors.append(f"ATR {atrp:.2f}% menunjukkan rentang per candle yang tinggi.")
    if hv is not None:
        points+=min(25,hv/4)
        if hv>=50: factors.append(f"Historical volatility {hv:.2f}% berada pada tingkat tinggi.")
    if mdd is not None:
        points+=min(25,abs(mdd)/2)
        if mdd<=-20: factors.append(f"Maximum drawdown historis mencapai {mdd:.2f}%.")
    if conflicts: points+=12; factors.append("Kelompok indikator memberikan sinyal yang saling bertentangan.")
    if rv is not None and rv<0.5: points+=8; factors.append("Relative volume rendah sehingga kualitas likuiditas candle melemah.")
    score=float(np.clip(points,0,100)); level="Low" if score<25 else "Medium" if score<50 else "High" if score<75 else "Very High"
    return score,level,factors,{"Historical volatility":hv,"ATR percentage":atrp,"Maximum drawdown":mdd,"Downside deviation":downside,"Average candle range":number(df.AvgRange20.iloc[-1]),"Distance to support":pa.get("distance_support"),"Distance to resistance":pa.get("distance_resistance"),"Risk-reward ratio":pa.get("risk_reward")}


def historical_rule_score(df: pd.DataFrame) -> pd.Series:
    pieces=[]
    pieces.append(np.where((df.Close>df.EMA20)&(df.EMA20>df.EMA50),60,np.where((df.Close<df.EMA20)&(df.EMA20<df.EMA50),-60,0)))
    pieces.append(np.where(df.MACD>df.MACDSignal,40,-40))
    pieces.append(np.where((df.RSI14>=50)&(df.RSI14<=70),30,np.where((df.RSI14>=30)&(df.RSI14<50),-20,0)))
    pieces.append(np.where(df.Close>df.VWAP,20,-20))
    arr=np.nanmean(np.vstack(pieces),axis=0)
    return pd.Series(np.clip(arr,-100,100),index=df.index)


def run_backtest(df: pd.DataFrame, horizon: int, bull: float, bear: float, cost_pct: float, slip_pct: float) -> tuple[Optional[dict[str,Any]],Optional[pd.DataFrame]]:
    if len(df)<max(220,horizon+60): return None,None
    score=historical_rule_score(df)
    signal=pd.Series(np.where(score>=bull,1,np.where(score<=bear,-1,0)),index=df.index)
    future=df.Close.shift(-horizon)/df.Close-1
    valid=pd.DataFrame({"score":score,"signal":signal,"future":future}).dropna()
    if len(valid)<30: return None,None
    actual=np.sign(valid.future); pred=valid.signal
    evaluated=pred!=0; costs=(cost_pct+slip_pct)/100
    strat=(pred*valid.future-costs*(pred!=0)).fillna(0)
    equity=(1+strat).cumprod(); dd=equity/equity.cummax()-1
    bullmask=pred==1; bearmask=pred==-1; neutral=pred==0
    conf=np.clip(np.abs(valid.score),0,100); bucket=pd.cut(conf,[0,20,40,60,80,100],include_lowest=True)
    by=pd.DataFrame({"correct":((pred==actual)|(neutral&(actual==0))).astype(float),"bucket":bucket}).groupby("bucket",observed=True).agg(Accuracy=("correct","mean"),Signals=("correct","size")).reset_index()
    metrics={"Directional accuracy":number((pred[evaluated]==actual[evaluated]).mean()*100) if evaluated.any() else None,"Bullish precision":number((actual[bullmask]>0).mean()*100) if bullmask.any() else None,"Bearish precision":number((actual[bearmask]<0).mean()*100) if bearmask.any() else None,"Neutral accuracy":number((actual[neutral]==0).mean()*100) if neutral.any() else None,"Number of evaluated signals":int(evaluated.sum()),"Average future return after bullish signal":number(valid.loc[bullmask,"future"].mean()*100),"Average future return after bearish signal":number(valid.loc[bearmask,"future"].mean()*100),"Win rate":number((strat[evaluated]>0).mean()*100) if evaluated.any() else None,"Total strategy return":number((equity.iloc[-1]-1)*100),"Maximum drawdown":number(dd.min()*100)}
    return metrics,by


def chart(df: pd.DataFrame, indicators: list[str], pa: dict[str,Any], proj: Optional[Projection]) -> go.Figure:
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=.025,row_heights=[.55,.15,.15,.15],subplot_titles=("Historical Market Data","Volume","RSI","MACD"))
    fig.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="OHLC"),row=1,col=1)
    for key in indicators:
        if key in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[key],name=key,mode="lines"),row=1,col=1)
    if "Bollinger Bands" in indicators:
        for name in ["BBUpper","BBMid","BBLower"]: fig.add_trace(go.Scatter(x=df.index,y=df[name],name=name,mode="lines"),row=1,col=1)
    for label,val,color in [("Support",pa.get("support"),"green"),("Resistance",pa.get("resistance"),"red")]:
        if finite(val): fig.add_hline(y=val,line_dash="dot",line_color=color,annotation_text=label,row=1,col=1)
    if proj:
        for label,val,color in [("Bearish Target",proj.bearish,"#d62728"),("Base Target",proj.base,"#888"),("Bullish Target",proj.bullish,"#2ca02c"),("Invalidation",proj.invalidation,"#ff9900")]:
            if finite(val): fig.add_hline(y=val,line_dash="dash",line_color=color,annotation_text=label,row=1,col=1)
    fig.add_trace(go.Bar(x=df.index,y=df.Volume,name="Volume"),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df.RSI14,name="RSI14"),row=3,col=1); fig.add_hline(y=70,line_dash="dot",row=3,col=1); fig.add_hline(y=30,line_dash="dot",row=3,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df.MACD,name="MACD"),row=4,col=1); fig.add_trace(go.Scatter(x=df.index,y=df.MACDSignal,name="Signal"),row=4,col=1); fig.add_trace(go.Bar(x=df.index,y=df.MACDHist,name="Histogram"),row=4,col=1)
    fig.update_layout(height=900,xaxis_rangeslider_visible=False,legend_orientation="h",hovermode="x unified")
    return fig


def main() -> None:
    st.title("Rule-Based Stock Analytics & Price Tendency")
    st.caption("Analisis teknikal, statistik, price action, volume, benchmark, fundamental, proyeksi skenario, dan backtesting tanpa machine learning.")
    st.warning(DISCLAIMER); st.info(NO_DUMMY)
    with st.sidebar:
        st.header("Parameter Analisis")
        ticker=st.text_input("Ticker saham",value="BBCA.JK").strip().upper()
        benchmark=st.text_input("Benchmark ticker",value="^JKSE").strip().upper()
        period=st.selectbox("Periode historis",PERIODS,index=4)
        interval=st.selectbox("Interval candle",INTERVALS,index=5)
        horizon=st.selectbox("Horizon prediksi (candle)",HORIZONS,index=2)
        profile=st.selectbox("Profil analisis",["Balanced","Trend Focus","Short-Term Momentum","Conservative"])
        indicators=st.multiselect("Overlay chart",["SMA20","SMA50","SMA200","EMA9","EMA20","EMA50","Bollinger Bands","VWAP"],default=["SMA20","EMA20","EMA50","Bollinger Bands","VWAP"])
        st.subheader("Backtesting")
        bull=st.slider("Bullish threshold",20,80,30); bear=st.slider("Bearish threshold",-80,-20,-30)
        cost=st.number_input("Transaction cost (%)",0.0,5.0,0.15,0.05); slip=st.number_input("Slippage (%)",0.0,5.0,0.05,0.05)
        analyze=st.button("Analyze / Refresh Data",type="primary",use_container_width=True)
    if not ticker: st.error("Ticker wajib diisi."); st.stop()
    ok,msg=validate_combo(period,interval)
    if not ok: st.error(msg); st.stop()
    if analyze: st.cache_data.clear()
    with st.spinner("Mengambil data aktual dari sumber eksternal..."):
        raw,error,fetched=fetch_stock_data(ticker,period,interval)
    ok,msg=validate_market_data(raw)
    if not ok:
        st.error(error or msg); st.error("Pengambilan data gagal. Tidak ada data dummy yang digunakan sebagai pengganti."); st.stop()
    df=compute_indicators(raw.dropna(subset=["Open","High","Low","Close"]).copy())
    info,fast,info_error=fetch_ticker_information(ticker)
    # Gunakan payload info yang sama untuk menghindari request fundamental duplikat saat startup.
    fundamentals, statements, fund_error = info, {}, ""
    bench_data=pd.DataFrame(); bench_error=""
    if benchmark:
        bench_data,bench_error,_=fetch_stock_data(benchmark,period,interval)
    bctx=benchmark_analysis(df,bench_data) if not bench_data.empty else {"available":False}
    pa=price_action(df)
    bid=number(info.get("bid",fast.get("bid"))); ask=number(info.get("ask",fast.get("ask")))
    bidask=bool(bid and ask and bid>0 and ask>0 and ask>=bid)
    mid=(bid+ask)/2 if bidask else None; spread=ask-bid if bidask else None; spread_pct=safe_div(spread,mid)*100 if bidask else None
    groups=score_groups(df,pa,bctx,fundamentals)
    total,weights,contrib,unavailable=combine_scores(groups); prediction=classify(total)
    conf=confidence(total,groups,df,fetched,interval,bidask)
    proj=make_projection(df,total,horizon,pa)
    risk_score,risk_level,risk_factors,risk_metrics=risk_analysis(df,pa,groups)
    last=df.iloc[-1]; prev=df.iloc[-2] if len(df)>1 else None; change=(last.Close-prev.Close) if prev is not None else np.nan; change_pct=safe_div(change,prev.Close)*100 if prev is not None else np.nan
    missing_pct=float(df[MIN_COLUMNS].isna().sum().sum()/(len(df)*len(MIN_COLUMNS))*100)
    tz=str(getattr(df.index,"tz",None) or "Tidak tersedia")
    st.header("Status Sumber Data")
    status={"Provider":PROVIDER,"Requested ticker":ticker,"Requested interval":interval,"Requested period":period,"Total candles":len(df),"First candle timestamp":str(df.index[0]),"Last candle timestamp":str(df.index[-1]),"Fetch timestamp (UTC)":fetched.isoformat(),"Data timezone":tz,"Missing data percentage":f"{missing_pct:.2f}%","Fundamental availability":"Tersedia sebagian" if any(finite(v) for v in fundamentals.values()) else "Tidak tersedia","Bid-ask availability":"Tersedia" if bidask else "Tidak tersedia","Order-book availability":"Data tidak tersedia dari API yang digunakan","Transaction-flow availability":"Data tidak tersedia dari API yang digunakan"}
    st.dataframe(pd.DataFrame(status.items(),columns=["Field","Status"]),hide_index=True,use_container_width=True)
    if error: st.warning(error)
    if info_error: st.info(info_error)
    if fund_error: st.info(fund_error)
    if bench_error: st.warning(bench_error)
    st.caption("Latest Available Data. Harga penutupan candle terakhir tidak diklaim sebagai harga real-time. Status keterlambatan mengikuti kebijakan sumber dan bursa serta tidak selalu disediakan oleh API.")
    st.header("Informasi Ticker")
    st.write({"Nama":info.get("longName") or info.get("shortName") or "N/A","Exchange":info.get("exchange") or "N/A","Currency":info.get("currency") or fast.get("currency") or "N/A","Quote Type":info.get("quoteType") or "N/A","Profil Analisis":profile})
    cols=st.columns(9)
    metrics=[("Harga Terakhir Tersedia",fmt(last.Close)),("Perubahan Candle",fmt(change)),("Perubahan",fmt(change_pct,2,"%")),("Volume",fmt(last.Volume,0)),("Relative Volume",fmt(last.RelativeVolume)),("ATR",fmt(last["ATR%"],2,"%")),("Prediction",prediction),("Signal Confidence",fmt(conf["Signal Confidence"],1,"%")),("Risk Level",risk_level)]
    for col,(label,value) in zip(cols,metrics): col.metric(label,value)
    st.header("Prediction Panel dan Price Scenarios")
    a,b=st.columns([1,2]); a.metric("Rule-Based Score",fmt(total,1)); a.metric("Kecenderungan",prediction); a.metric("Risk Score",fmt(risk_score,1))
    if proj:
        b.dataframe(pd.DataFrame([{"Bearish Scenario":proj.bearish,"Base Scenario":proj.base,"Bullish Scenario":proj.bullish,"Invalidation Level":proj.invalidation,"Expected Range Low":proj.range_low,"Expected Range High":proj.range_high,"Risk-Reward Ratio":proj.risk_reward,"ATR Multiplier":proj.atr_multiplier}]),hide_index=True,use_container_width=True)
        b.caption("Skenario dihitung dari harga aktual, ATR, arah score, horizon, support, dan resistance. ATR multiplier berbasis akar horizon adalah asumsi metodologi, bukan data pasar.")
    else: b.warning("Proyeksi tidak dihitung karena ATR atau data harga belum mencukupi.")
    st.header("Candlestick Chart")
    st.plotly_chart(chart(df,indicators,pa,proj),use_container_width=True)
    tabs=st.tabs(["Technical Analysis","Volume Analysis","Benchmark","Fundamental","Supporting Factors","Risk Factors"])
    with tabs[0]:
        st.dataframe(pd.DataFrame({k:[number(last.get(k))] for k in ["SMA20","SMA50","SMA200","EMA9","EMA20","EMA50","ADX14","+DI","-DI","RSI14","MACD","MACDSignal","StochK","StochD","ROC10","ATR14","ATR%","HistVol20"]}),use_container_width=True)
        st.write({"Structure":pa["structure"],"Support":pa["support"],"Resistance":pa["resistance"],"Breakout":pa["breakout"],"Breakdown":pa["breakdown"],"Consolidation":pa["consolidation"]})
    with tabs[1]:
        st.info("Volume Analysis menggunakan candle OHLCV. Ini bukan representasi pasti aggressive buy volume, aggressive sell volume, atau antrean order book.")
        st.dataframe(pd.DataFrame({k:[number(last.get(k))] for k in ["Volume","VolumeSMA20","RelativeVolume","OBV","ADL","MFI14","VPT","VWAP"]}),use_container_width=True)
        st.warning("Analisis buy-sell pressure tidak dapat dihitung karena data transaksi agresif tidak tersedia. Sistem tidak membuat atau mensimulasikannya.")
    with tabs[2]:
        if bctx.get("available"): st.dataframe(pd.DataFrame([bctx]),hide_index=True,use_container_width=True)
        else: st.warning("Benchmark tidak tersedia atau data tidak cukup. Market score dikeluarkan dan bobot lain dinormalisasi ulang.")
    with tabs[3]:
        fields={"Market Cap":fundamentals.get("marketCap"),"Trailing P/E":fundamentals.get("trailingPE"),"Forward P/E":fundamentals.get("forwardPE"),"Price-to-Book":fundamentals.get("priceToBook"),"Dividend Yield":fundamentals.get("dividendYield"),"EPS":fundamentals.get("trailingEps"),"Revenue":fundamentals.get("totalRevenue"),"Net Income":fundamentals.get("netIncomeToCommon"),"Operating Cash Flow":fundamentals.get("operatingCashflow"),"Free Cash Flow":fundamentals.get("freeCashflow"),"Total Debt":fundamentals.get("totalDebt"),"Total Equity":fundamentals.get("totalStockholderEquity"),"ROE":fundamentals.get("returnOnEquity"),"Profit Margin":fundamentals.get("profitMargins"),"Revenue Growth":fundamentals.get("revenueGrowth"),"Earnings Growth":fundamentals.get("earningsGrowth")}
        st.dataframe(pd.DataFrame([fields]).T.rename(columns={0:"Actual Value"}),use_container_width=True)
        st.caption("Fundamental dimuat dari payload ticker aktual. Laporan keuangan lengkap tidak diambil saat startup agar aplikasi lebih ringan. Fundamental dapat memiliki tanggal laporan berbeda dari candle terakhir.")
    positive=[r["Interpretation"] for g in groups for r in g.reasons if r["Raw Score"]>0]
    with tabs[4]:
        if positive:
            for item in positive: st.success(item)
        else: st.info("Tidak ada supporting factor bullish yang memenuhi rule saat ini.")
    with tabs[5]:
        if risk_factors:
            for item in risk_factors: st.warning(item)
        else: st.info("Tidak ada faktor risiko tambahan yang melampaui rule threshold, tetapi risiko pasar tetap ada.")
        st.dataframe(pd.DataFrame([risk_metrics]),hide_index=True,use_container_width=True)
        st.write("Bid-ask:",{"Bid":bid,"Ask":ask,"Mid":mid,"Spread":spread,"Spread %":number(spread_pct)} if bidask else "Data tidak tersedia atau tidak valid; spread tidak dihitung.")
    st.header("Score Contribution dan Dynamic Weight Normalization")
    st.dataframe(contrib.style.format({"Base Weight":"{:.1%}","Effective Weight":"{:.1%}","Score":"{:.2f}","Weighted Contribution":"{:.2f}"}),use_container_width=True)
    if unavailable: st.info("Kelompok dikeluarkan dari total dan bobot dinormalisasi ulang: "+", ".join(unavailable))
    details=[]
    for g in groups:
        for r in g.reasons:
            details.append({**r,"Group":g.name,"Effective Weight":weights.get(g.name,0),"Weighted Contribution":r["Raw Score"]*weights.get(g.name,0)})
    if details: st.dataframe(pd.DataFrame(details),hide_index=True,use_container_width=True)
    st.header("Signal Confidence Breakdown")
    st.dataframe(pd.DataFrame(conf.items(),columns=["Component","Value (%)"]),hide_index=True,use_container_width=True)
    st.header("Backtesting")
    bt,buckets=run_backtest(df,horizon,bull,bear,cost,slip)
    if bt:
        st.dataframe(pd.DataFrame(bt.items(),columns=["Metric","Value"]),hide_index=True,use_container_width=True)
        st.dataframe(buckets,hide_index=True,use_container_width=True)
        st.warning("Historical performance bukan jaminan hasil masa depan. Backtest menggunakan score pada candle saat itu dan future return yang digeser sesuai horizon, tanpa memakai data masa depan untuk pembentukan sinyal.")
    else: st.warning("Backtesting tidak dijalankan karena data historis tidak mencukupi. Gunakan periode lebih panjang tanpa mengubah interval secara diam-diam.")
    st.header("Methodology")
    st.markdown("""
- Indikator dihitung lokal dari OHLCV aktual tanpa backward fill dan tanpa training model.
- Skor kelompok berada pada rentang -100 sampai +100. Bobot efektif dinormalisasi hanya dari kelompok yang tersedia.
- Signal Confidence menggabungkan kesepakatan indikator, kelengkapan dan freshness data, kekuatan sinyal, likuiditas candle, volatilitas, dan konflik.
- Proyeksi adalah skenario berbasis ATR, horizon, arah skor, support, dan resistance. Bukan probabilitas keberhasilan.
- Support dan resistance berasal dari pivot yang telah terkonfirmasi dan rolling high/low sebelumnya. Pivot masa depan tidak digunakan pada sinyal historis.
- Order flow score tidak digunakan karena sumber gratis ini tidak menyediakan market depth atau trade-side aktual secara konsisten.
""")
    st.header("Data Limitations")
    st.markdown(f"- Provider: {PROVIDER}. Data dapat tertunda atau tidak lengkap.\n- Order book tidak tersedia dari sumber API gratis yang digunakan. Sistem tidak membuat atau mensimulasikan data order book.\n- Candle OHLCV bukan data transaksi individual.\n- Fundamental ditampilkan hanya jika benar-benar dikembalikan sumber. Nilai hilang tetap N/A.\n- Tidak ada fallback ke data buatan ketika API gagal.\n- {DISCLAIMER}\n- {NO_DUMMY}")


if __name__ == "__main__":
    main()
