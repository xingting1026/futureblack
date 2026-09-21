# -*- coding: utf-8 -*-
"""futureblack 每晚主流程（GitHub Actions 21:00 台北呼叫）：

    trading_days, data_day, cycle = prepare_calendar()   # 交易日與本週期
    mapping = load_mapping()                             # 契約↔股票對照（自動補新契約）
    fill_daily_cache(cycle, mapping)                     # 補齊窗內缺的「每日溢價快取」← 增量
    factors = build_factors(cycle, mapping)              # 週期溢價(跨日平均+大小期加權) + 未平倉市值
    render_page(factors, cycle)                          # 產出 site/index.html ← 本程式的產出物

溢價公式與期貨反向單生產系統同構：每日 53 個 5 分鐘格點的
期貨價/現貨價-1 → 日均 → 一天一檔快取(data/daily) → 窗內跨日平均 →
大小期以「特定法人口數×乘數×收盤價」加權（權重全零退回乘數比）。
與生產版的已知差異：用成交價（公開 tick 無買賣報價）、無前一日種子。
"""
import datetime
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calendar_tw
import finmind

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_DIR = os.path.join(REPO, "data", "daily")
MAPPING_PATH = os.path.join(REPO, "data", "mapping.csv")
SITE_DIR = os.path.join(REPO, "site")

GRID_START, GRID_END, GRID_STEP_MIN = "09:05:00", "13:25:00", 5   # 53 格，同生產版
MIN_GRIDS_PER_DAY = 10          # 有效格太少的契約日不納入（冷門到沒參考價值）
TOP_RANK = 30                   # 交集門檻：溢價前30 ∩ 未平倉市值前30
SMALL_PREFIX = "小型"           # 大額表商品名的小型契約前綴（乘數 100，其餘 2000）

# cover 影子名單（2026-09 以 2024-12~2026-08 共 21 期回測定案；cover = days-to-cover =
# 特定法人前十大方向性口數換算張數 ÷ 近 20 日均量張數，衡量結算日被迫拆倉量相對市場胃納）
ADV_DAYS = 20                   # cover 分母的取樣交易日數
SHORT_PREM_MIN = 0.002          # 空方：週期溢價下限（回測平原 月t=3.8）
SHORT_COVER_MIN = 0.15          # 空方：賣方 cover 下限
SHORT_TOP_K = 3                 # 空方：取賣方 cover 最高前幾檔
LONG_DISC_MAX = -0.002          # 多方：週期折價門檻（要更負；回測平原 月t=2.4）
LONG_COVER_MIN = 0.05           # 多方：買方 cover 下限
LONG_TOP_K = 2                  # 多方：取買方 cover 最高前幾檔
FAT_DISCOUNT_POOL = 100         # 折價股超過此數＝全市場逆價差月（除息假折價充斥），多方停用

# 賣壓分數 → 結算日 12:30 跳水預估（67 個結算日 2021-02~2026-08 校準，T-1 時點）
# 分數 = Σ(週期溢價 × |特定法人OI市值百萬|)/100，僅溢價側；四分位對照:
# {分位: (跳水機率%≥15bps, 機率%≥25bps, 平均bps, 中位bps)}
PRESSURE_HISTORY_PATH = "pressure_history.csv"   # data/ 下的校準歷史
DIVE_CALIB = {1: (18, 0, -3.1, -4.7), 2: (41, 24, -9.4, -10.8),
              3: (69, 44, -19.3, -21.2), 4: (71, 59, -25.7, -31.8)}


def build_grid_marks():
    t = datetime.datetime.strptime(GRID_START, "%H:%M:%S")
    end = datetime.datetime.strptime(GRID_END, "%H:%M:%S")
    marks = []
    while t <= end:
        marks.append(t.strftime("%H:%M:%S"))
        t += datetime.timedelta(minutes=GRID_STEP_MIN)
    return marks


GRID_MARKS = build_grid_marks()


# ====================================================================
# 對照表：契約代碼(3碼) ↔ 股票代號、乘數；每晚用大額表的商品名自動補新契約
# ====================================================================

def load_mapping():
    return pd.read_csv(MAPPING_PATH, encoding="utf-8-sig",
                       dtype={"契約代碼": str, "sid": str})


def find_new_contracts(mapping, large_traders, stock_names):
    """大額表出現、對照表沒有的契約 → 用「XX期貨」的商品名對股票名補進對照表"""
    known = set(mapping["契約代碼"])
    name_to_sid = dict(zip(stock_names["stock_name"], stock_names["stock_id"]))
    rows = []
    for _, r in large_traders.drop_duplicates("futures_id").iterrows():
        code = str(r["futures_id"]) + "F"
        if code in known or len(str(r["futures_id"])) != 2:
            continue
        name = str(r["name"]).replace("期貨", "")
        multiplier = 2000
        if name.startswith(SMALL_PREFIX):
            name = name[len(SMALL_PREFIX):]
            multiplier = 100
        sid = name_to_sid.get(name)
        if sid:
            rows.append({"契約代碼": code, "sid": sid, "契約乘數": multiplier})
    return pd.DataFrame(rows)


# ====================================================================
# 每日溢價快取：一天一檔 data/daily/{日期}.csv（增量，跑過就不再抓）
# ====================================================================

def price_at_marks(times, prices):
    """時間排序後，每個格點取「該時刻(含)以前的最後一筆價」；沒有就 NaN"""
    idx = np.searchsorted(times, GRID_MARKS, side="right") - 1
    return np.where(idx >= 0, prices[np.clip(idx, 0, None)], np.nan)


def calc_one_contract_day(fut_ticks, stock_kbar):
    """一契約一天：53 格成交價比值的日均。回傳 (日均溢價, 有效格數, 期貨收盤)"""
    ft = fut_ticks.sort_values("date")
    f_times = pd.to_datetime(ft["date"]).dt.strftime("%H:%M:%S").to_numpy()
    f_px = ft["price"].to_numpy(dtype=float)
    sk = stock_kbar.sort_values("minute")
    s_times, s_px = sk["minute"].to_numpy(), sk["close"].to_numpy(dtype=float)

    f_grid = price_at_marks(f_times, f_px)
    s_grid = price_at_marks(s_times, s_px)
    valid = ~np.isnan(f_grid) & ~np.isnan(s_grid) & (s_grid > 0)
    if valid.sum() < MIN_GRIDS_PER_DAY:
        return np.nan, int(valid.sum()), f_px[-1] if len(f_px) else np.nan
    prem = float(np.mean(f_grid[valid] / s_grid[valid] - 1))
    return prem, int(valid.sum()), float(f_px[-1])


def fill_daily_cache(cycle, mapping):
    """窗內每個缺快取的交易日：抓該日全部契約 tick + 個股分K → 算每契約日均溢價"""
    os.makedirs(DAILY_DIR, exist_ok=True)
    near = cycle["近月年月"]
    for day in cycle["窗"]:
        path = os.path.join(DAILY_DIR, f"{day}.csv")
        if os.path.exists(path):
            continue
        print(f"[cache] 補 {day} …", flush=True)
        kbar_by_sid = {}
        rows = []
        for _, m in mapping.iterrows():
            ticks = finmind.fetch("TaiwanFuturesTick", data_id=m["契約代碼"],
                                  start_date=str(day), end_date=str(day))
            if not len(ticks):
                continue
            ticks = ticks[ticks["contract_date"] == near]
            if not len(ticks):
                continue
            sid = m["sid"]
            if sid not in kbar_by_sid:
                kbar_by_sid[sid] = finmind.fetch("TaiwanStockKBar", data_id=sid,
                                                 start_date=str(day), end_date=str(day))
            if not len(kbar_by_sid[sid]):
                continue
            prem, n_grids, fut_close = calc_one_contract_day(ticks, kbar_by_sid[sid])
            rows.append({"契約代碼": m["契約代碼"], "sid": sid, "契約乘數": m["契約乘數"],
                         "日均溢價": prem, "有效格數": n_grids, "期貨收盤": fut_close})
        if not rows:
            # 整天零契約 = 資料源還沒上這一天（盤後到晚間才進），不落地空檔——
            # 落了 skip-existing 會把這天永久跳過
            print(f"[cache] {day}: 尚無資料，不落地，下次重跑再補 (累計 API {finmind.n_calls})",
                  flush=True)
            continue
        pd.DataFrame(rows).to_csv(path + ".tmp", index=False, encoding="utf-8-sig")
        os.replace(path + ".tmp", path)
        print(f"[cache] {day}: {len(rows)} 契約 (累計 API {finmind.n_calls})", flush=True)


# ====================================================================
# 因子：週期溢價（跨日平均＋大小期加權）與未平倉市值
# ====================================================================

def build_factors(cycle, mapping, stock_names):
    cache_paths = [os.path.join(DAILY_DIR, f"{d}.csv") for d in cycle["窗"]]
    cache_paths = [p for p in cache_paths
                   if os.path.exists(p) and os.path.getsize(p) > 10]   # 防歷史殘留空檔
    daily = pd.concat([pd.read_csv(p, encoding="utf-8-sig", dtype={"契約代碼": str, "sid": str})
                       for p in cache_paths],
                      ignore_index=True)
    daily = daily.dropna(subset=["日均溢價"])
    per_contract = (daily.groupby(["契約代碼", "sid", "契約乘數"])
                    .agg(週期溢價=("日均溢價", "mean"), 有資料天數=("日均溢價", "size"),
                         期貨收盤=("期貨收盤", "last"))
                    .reset_index())

    large = finmind.fetch("TaiwanFuturesOpenInterestLargeTraders",
                          start_date=str(cycle["窗"][-1]), end_date=str(cycle["窗"][-1]))
    large = large[large["contract_type"].astype(str) == cycle["近月年月"]].copy()
    large["契約代碼"] = large["futures_id"].astype(str) + "F"
    per_contract = per_contract.merge(
        large[["契約代碼", "buy_top10_specific_open_interest",
               "sell_top10_specific_open_interest"]], on="契約代碼", how="left")

    # 方向性口數：折價取前十大買方特定法人、溢價取賣方（與生產 C1/C3 同規則）
    qty = np.where(per_contract["週期溢價"] < 0,
                   per_contract["buy_top10_specific_open_interest"],
                   per_contract["sell_top10_specific_open_interest"]).astype(float)
    qty = np.nan_to_num(qty)
    sign = np.where(per_contract["週期溢價"] < 0, -1.0, 1.0)
    per_contract["市值元"] = (sign * qty * per_contract["契約乘數"].astype(float)
                              * per_contract["期貨收盤"].astype(float))

    stocks = []
    for sid, g in per_contract.groupby("sid"):
        weights = np.abs(g["市值元"].to_numpy())
        if weights.sum() <= 0:
            weights = g["契約乘數"].to_numpy(dtype=float)      # 退回乘數比（同生產）
        buy_zhang = float((g["buy_top10_specific_open_interest"].fillna(0.0)
                           * g["契約乘數"].astype(float)).sum()) / 1000.0
        sell_zhang = float((g["sell_top10_specific_open_interest"].fillna(0.0)
                            * g["契約乘數"].astype(float)).sum()) / 1000.0
        stocks.append({"sid": sid,
                       "週期溢價": float(np.average(g["週期溢價"], weights=weights)),
                       "未平倉市值_百萬": round(g["市值元"].sum() / 1e6),
                       "買方OI張": round(buy_zhang, 1),
                       "賣方OI張": round(sell_zhang, 1),
                       "有資料天數": int(g["有資料天數"].max())})
    factors = pd.DataFrame(stocks)
    names = dict(zip(stock_names["stock_id"], stock_names["stock_name"]))
    factors["名稱"] = factors["sid"].map(names).fillna("")
    return factors


def mark_strike_list(factors):
    """交集：溢價前30 ∩ |未平倉市值|前30，只看溢價為正的股票"""
    df = factors.copy()
    positive = df["週期溢價"] > 0
    df["溢價排名"] = df.loc[positive, "週期溢價"].rank(ascending=False)
    df["倉位排名"] = df.loc[positive, "未平倉市值_百萬"].abs().rank(ascending=False)
    df["打擊名單"] = (df["溢價排名"] <= TOP_RANK) & (df["倉位排名"] <= TOP_RANK)
    df["高溢價無倉"] = (df["溢價排名"] <= TOP_RANK) & ~df["打擊名單"]
    return df.sort_values("週期溢價", ascending=False)


def fetch_adv20(stock_ids, data_day):
    """近 ADV_DAYS 個交易日的日均成交張數（cover 的分母），一檔一次 API。
    回傳: {sid: 張數}；抓不滿 ADV_DAYS 天的（新上市等）不收，該檔 cover 視為無資料。"""
    start = data_day - datetime.timedelta(days=45)   # 45 個日曆日必含 20 個交易日
    adv = {}
    for i, sid in enumerate(stock_ids, 1):
        px = finmind.fetch("TaiwanStockPrice", data_id=sid,
                           start_date=str(start), end_date=str(data_day))
        if len(px) >= ADV_DAYS:
            adv[sid] = float(px.tail(ADV_DAYS)["Trading_Volume"].astype(float).mean()) / 1000.0
        if i % 50 == 0:
            print(f"  ADV20 {i}/{len(stock_ids)}...", flush=True)
    return adv


def fetch_tx_close(cycle):
    """近月台指期最近一個有資料日的收盤價（跳水幅度換算點數用）；抓不到回 None"""
    start = cycle["窗"][-1] - datetime.timedelta(days=7)
    tx = finmind.fetch("TaiwanFuturesDaily", data_id="TX",
                       start_date=str(start), end_date=str(cycle["窗"][-1]))
    if not len(tx):
        return None
    tx = tx[tx["contract_date"].astype(str) == cycle["近月年月"]]
    if "trading_session" in tx.columns:
        tx = tx[tx["trading_session"].astype(str) == "position"]
    if not len(tx):
        return None
    return float(tx.sort_values("date")["close"].iloc[-1])


def calc_pressure_score(factors, cycle):
    """賣壓分數與跳水預估：分數對 67 日校準歷史取百分位 → 四分位 → 機率/幅度，
    幅度另以近月台指收盤換算點數。
    回傳 dict(分數, 百分位, 分位, 機率15, 機率25, 平均bps, 中位bps, 台指收盤, 平均點數, 中位點數)"""
    prem_side = factors[factors["週期溢價"] > 0]
    score = float((prem_side["週期溢價"]
                   * prem_side["未平倉市值_百萬"].abs()).sum()) / 100.0
    history = pd.read_csv(os.path.join(REPO, "data", PRESSURE_HISTORY_PATH))["pressure"]
    percentile = float((history <= score).mean()) * 100.0
    quartile = min(int(percentile // 25) + 1, 4)
    p15, p25, mean_bps, median_bps = DIVE_CALIB[quartile]

    tx_close = fetch_tx_close(cycle)
    mean_points = round(tx_close * mean_bps / 1e4) if tx_close else None
    median_points = round(tx_close * median_bps / 1e4) if tx_close else None
    return {"分數": round(score, 2), "百分位": round(percentile), "分位": quartile,
            "機率15": p15, "機率25": p25, "平均bps": mean_bps, "中位bps": median_bps,
            "台指收盤": tx_close, "平均點數": mean_points, "中位點數": median_points}


def is_fat_discount_month(factors):
    """折價股檔數超過 FAT_DISCOUNT_POOL＝全市場逆價差月（除息假折價充斥）→ 多方腿停用"""
    return int((factors["週期溢價"] < 0).sum()) > FAT_DISCOUNT_POOL


def mark_cover_lists(factors, adv20):
    """cover 影子名單：
    空方 = 溢價 ≥ SHORT_PREM_MIN 且 賣方cover ≥ SHORT_COVER_MIN，取賣方 cover 前 SHORT_TOP_K
    多方 = 折價 ≤ LONG_DISC_MAX 且 買方cover ≥ LONG_COVER_MIN，取買方 cover 前 LONG_TOP_K
    （多方在全市場逆價差月整月停用，判定見 is_fat_discount_month）"""
    df = factors.copy()
    df["ADV20張"] = df["sid"].map(adv20)
    df["空方cover"] = (df["賣方OI張"] / df["ADV20張"]).round(4)
    df["多方cover"] = (df["買方OI張"] / df["ADV20張"]).round(4)

    short_ok = (df["週期溢價"] >= SHORT_PREM_MIN) & (df["空方cover"] >= SHORT_COVER_MIN)
    short_picks = df[short_ok].nlargest(SHORT_TOP_K, "空方cover")["sid"]
    df["空方名單"] = df["sid"].isin(short_picks)

    long_ok = (df["週期溢價"] <= LONG_DISC_MAX) & (df["多方cover"] >= LONG_COVER_MIN)
    if is_fat_discount_month(df):
        long_ok = pd.Series(False, index=df.index)
    long_picks = df[long_ok].nlargest(LONG_TOP_K, "多方cover")["sid"]
    df["多方名單"] = df["sid"].isin(long_picks)
    return df


# ====================================================================
# 頁面
# ====================================================================

def render_page(factors, cycle, data_day, pressure):
    tsmc = factors.set_index("sid")["週期溢價"].get("2330", np.nan)
    tsmc_text = f"{tsmc*100:+.2f}%" if pd.notna(tsmc) else "無資料"
    if pressure["台指收盤"]:
        dive_text = f"{pressure['平均bps']} bps ≈ {pressure['平均點數']:+,.0f} 點"
        dive_sub = (f"中位 {pressure['中位bps']} bps ≈ {pressure['中位點數']:+,.0f} 點｜"
                    f"台指近月收盤 {pressure['台指收盤']:,.0f}")
    else:
        dive_text = f"{pressure['平均bps']} bps"
        dive_sub = f"中位 {pressure['中位bps']} bps"

    def row_html(r):
        flag = "⚠" if r["高溢價無倉"] else ""
        return (f"<tr><td>{flag}</td><td>{r['sid']}</td><td>{r['名稱']}</td>"
                f"<td data-v='{r['週期溢價']:.6f}'>{r['週期溢價']*100:.3f}%</td>"
                f"<td data-v='{r['未平倉市值_百萬']}'>{r['未平倉市值_百萬']:,.0f}</td>"
                f"<td>{int(r['溢價排名']) if pd.notna(r['溢價排名']) else ''}</td>"
                f"<td>{int(r['倉位排名']) if pd.notna(r['倉位排名']) else ''}</td>"
                f"<td>{r['有資料天數']}</td></tr>")

    all_rows = "\n".join(row_html(r) for _, r in factors.iterrows())
    updated = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)

    def cover_row_html(r, cover_column, oi_column):
        return (f"<tr><td>{r['sid']}</td><td>{r['名稱']}</td>"
                f"<td>{r['週期溢價']*100:+.3f}%</td><td>{r[oi_column]:,.0f}</td>"
                f"<td>{r['ADV20張']:,.0f}</td><td><b>{r[cover_column]:.3f}</b></td></tr>")

    EMPTY_COVER_ROW = '<tr><td colspan="6" style="text-align:center;color:#999">今日無符合條件標的</td></tr>'
    shorts = factors[factors["空方名單"]].sort_values("空方cover", ascending=False)
    longs = factors[factors["多方名單"]].sort_values("多方cover", ascending=False)
    short_cover_rows = ("\n".join(cover_row_html(r, "空方cover", "賣方OI張")
                                  for _, r in shorts.iterrows()) or EMPTY_COVER_ROW)
    long_cover_rows = ("\n".join(cover_row_html(r, "多方cover", "買方OI張")
                                 for _, r in longs.iterrows()) or EMPTY_COVER_ROW)
    long_note = ("（本月折價股逾 %d 檔＝全市場逆價差，多方停用）" % FAT_DISCOUNT_POOL
                 if is_fat_discount_month(factors) else "")

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>futureblack — 結算週期溢價看板</title>
<style>
body{{font-family:"Microsoft JhengHei","PingFang TC",sans-serif;margin:16px auto;max-width:1080px;
background:#f5f6f8;color:#222}}
h1{{font-size:1.4em}}
.stats{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}}
.stats>div{{flex:1;min-width:180px;background:#fff;border-radius:8px;padding:10px 14px;
border:1px solid #e3e5e8}}
.stats .k{{color:#666;font-size:.8em}} .stats .v{{font-size:1.5em;font-weight:bold;margin:2px 0}}
.stats .s{{color:#888;font-size:.75em}}
.meta{{color:#555;margin:8px 0 16px}} table{{border-collapse:collapse;width:100%;background:#fff;
margin-bottom:24px}} th,td{{border-bottom:1px solid #e3e5e8;padding:6px 10px;text-align:right;
font-size:.92em}} th{{background:#eef0f3;cursor:pointer;position:sticky;top:0}}
td:nth-child(2),td:nth-child(3),th:nth-child(2),th:nth-child(3){{text-align:left}}
.note{{color:#777;font-size:.85em;line-height:1.6}}
.cover-wrap{{display:flex;gap:16px;flex-wrap:wrap}} .cover-wrap>div{{flex:1;min-width:320px}}
h3{{font-size:1em;margin:4px 0 6px}}
</style></head><body>
<h1>futureblack — 期貨結算週期溢價看板</h1>
<div class="stats">
<div><div class="k">台積電週期溢價</div><div class="v">{tsmc_text}</div>
<div class="s">近月 {cycle['近月年月']}</div></div>
<div><div class="k">賣壓分數</div><div class="v">{pressure['分數']}</div>
<div class="s">67 個結算日歷史第 {pressure['百分位']:.0f} 百分位（Q{pressure['分位']}）</div></div>
<div><div class="k">結算日 12:30 跳水機率</div><div class="v">{pressure['機率15']}%</div>
<div class="s">跳 ≥15bps；跳 ≥25bps 為 {pressure['機率25']}%</div></div>
<div><div class="k">預期跳水幅度</div><div class="v">{dive_text}</div>
<div class="s">{dive_sub}</div></div>
</div>
<div class="meta">資料日 {data_day} ｜ 週期 {cycle['上次結算日'] + datetime.timedelta(days=1)} ~ 本次結算日
 <b>{cycle['本次結算日']}</b> ｜ 近月 {cycle['近月年月']} ｜ 更新 {updated:%Y-%m-%d %H:%M} (台北)</div>

<h2>cover 名單（結算日 12:27~12:30 進場、12:56~12:59 出場，13:00 前一定出清）</h2>
<div class="cover-wrap">
<div><h3>空方放空：溢價 ≥{SHORT_PREM_MIN*100:.1f}% × 賣方cover ≥{SHORT_COVER_MIN} × 前{SHORT_TOP_K}檔</h3>
<table><thead><tr><th>代碼</th><th>名稱</th><th>週期溢價</th><th>賣方OI(張)</th>
<th>20日均量(張)</th><th>cover</th></tr></thead>
<tbody>{short_cover_rows}</tbody></table></div>
<div><h3>多方做多：折價 ≤{LONG_DISC_MAX*100:.1f}% × 買方cover ≥{LONG_COVER_MIN} × 前{LONG_TOP_K}檔{long_note}</h3>
<table><thead><tr><th>代碼</th><th>名稱</th><th>週期溢價</th><th>買方OI(張)</th>
<th>20日均量(張)</th><th>cover</th></tr></thead>
<tbody>{long_cover_rows}</tbody></table></div>
</div>

<h2>全部個股期貨標的（{len(factors)} 檔，點欄位標題排序）</h2>
<table id="t2"><thead><tr><th></th><th>代碼</th><th>名稱</th><th>週期溢價</th>
<th>未平倉市值(百萬)</th><th>溢價排名</th><th>倉位排名</th><th>資料天數</th></tr></thead>
<tbody>{all_rows}</tbody></table>

<div class="note">
⚠ = 溢價前{TOP_RANK}但特定法人未平倉不足（無拆倉賣壓，不打）<br>
cover = 特定法人前十大方向性口數換算張數 ÷ 近{ADV_DAYS}日均量張數（days-to-cover，
衡量結算日被迫拆倉量相對市場胃納）；空方取賣方口數、多方取買方口數。
多方於全市場逆價差月（折價股 &gt;{FAT_DISCOUNT_POOL} 檔）停用。<br>
賣壓分數 = Σ(週期溢價 × 特定法人OI市值)，預估結算日中午指數跳水（67 個結算日校準，
2021-02~2026-08）。校準時點為 T-1：週期初 OI 尚未拆倉、分數與 cover 皆會偏高，
越接近結算日越準。<br>
溢價 = 週期內每日 53 個 5 分鐘格點的「期貨成交價/現貨成交價−1」日均，跨日平均，
大小期以特定法人口數×乘數×收盤價加權。與內部版差異：成交價（非買賣報價雙邊）、無前日種子。<br>
資料來源：FinMind（期貨逐筆、個股分K、期交所大額交易人）。每交易日 21:00 (台北) 自動更新。<br>
本頁為研究紀錄，不構成投資建議。
</div>
<script>
document.querySelectorAll("th").forEach((th)=>th.addEventListener("click",()=>{{
 const tb=th.closest("table").querySelector("tbody");const i=th.cellIndex;
 const asc=th.asc=!th.asc;
 [...tb.rows].sort((a,b)=>{{
  const va=a.cells[i].dataset.v??a.cells[i].innerText, vb=b.cells[i].dataset.v??b.cells[i].innerText;
  const na=parseFloat(va), nb=parseFloat(vb);
  const c=(isNaN(na)||isNaN(nb))?va.localeCompare(vb):na-nb; return asc?c:-c;
 }}).forEach(r=>tb.appendChild(r));}}));
</script></body></html>"""
    os.makedirs(SITE_DIR, exist_ok=True)
    with open(os.path.join(SITE_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    today = datetime.date.today()
    trading_days = calendar_tw.get_trading_days(today)
    past_days = [d for d in trading_days if d <= today]
    data_day = past_days[-1]
    cycle = calendar_tw.build_cycle(trading_days, data_day)
    print(f"資料日 {data_day} 窗 {cycle['窗'][0]}~{cycle['窗'][-1]} "
          f"({len(cycle['窗'])}日) 結算 {cycle['本次結算日']}", flush=True)

    mapping = load_mapping()
    stock_names = finmind.fetch("TaiwanStockInfo")
    large_today = finmind.fetch("TaiwanFuturesOpenInterestLargeTraders",
                                start_date=str(data_day), end_date=str(data_day))
    new_contracts = find_new_contracts(mapping, large_today, stock_names)
    if len(new_contracts):
        mapping = pd.concat([mapping, new_contracts], ignore_index=True)
        mapping.to_csv(MAPPING_PATH, index=False, encoding="utf-8-sig")
        print(f"對照表新增 {len(new_contracts)} 契約", flush=True)

    fill_daily_cache(cycle, mapping)
    factors = build_factors(cycle, mapping, stock_names)
    factors = mark_strike_list(factors)

    adv20 = fetch_adv20(factors["sid"].tolist(), data_day)
    factors = mark_cover_lists(factors, adv20)
    pressure = calc_pressure_score(factors, cycle)

    factors.to_csv(os.path.join(REPO, "data", "factors_latest.csv"),
                   index=False, encoding="utf-8-sig")
    render_page(factors, cycle, data_day, pressure)
    print(f"完成：{len(factors)} 檔, 打擊名單 {int(factors['打擊名單'].sum())} 檔, "
          f"cover 空方 {int(factors['空方名單'].sum())} / 多方 {int(factors['多方名單'].sum())} 檔, "
          f"賣壓分數 {pressure['分數']} (P{pressure['百分位']:.0f}), "
          f"API 共 {finmind.n_calls} 次", flush=True)
