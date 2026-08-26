# -*- coding: utf-8 -*-
"""台股交易日與期貨結算日曆（自足版，邏輯與期貨反向單生產系統一致）。
結算日 = 每月第三個週三，遇休市往後找第一個交易日。
交易日清單用台積電日線的有價日代替官方日曆（上市以來無缺）。"""
import datetime

import finmind

CALENDAR_LOOKBACK_DAYS = 120   # 涵蓋整個結算窗
CALENDAR_LOOKAHEAD_DAYS = 60   # 涵蓋下一次結算日


def get_trading_days(anchor):
    start = anchor - datetime.timedelta(days=CALENDAR_LOOKBACK_DAYS)
    end = anchor + datetime.timedelta(days=CALENDAR_LOOKAHEAD_DAYS)
    px = finmind.fetch("TaiwanStockPrice", data_id="2330",
                       start_date=str(start), end_date=str(end))
    return sorted(datetime.date.fromisoformat(d) for d in px["date"].unique())


def get_settlement_day(year, month, trading_day_set, last_day):
    """第三個週三，遇休市往後找。超出已知交易日範圍（未來月份）時
    回傳未調整的第三個週三——股價資料沒有未來日，假日順延等日期臨近再自動生效。"""
    third_wednesday = next(datetime.date(year, month, d) for d in range(15, 22)
                           if datetime.date(year, month, d).weekday() == 2)
    if third_wednesday > last_day:
        return third_wednesday
    day = third_wednesday
    while day not in trading_day_set:
        day += datetime.timedelta(days=1)
        if day > last_day:
            return day
    return day


def build_cycle(trading_days, data_day):
    """回傳 dict(本次結算日, 上次結算日, 近月年月, 窗=[上次結算翌日..data_day 的交易日])"""
    day_set = set(trading_days)
    last = trading_days[-1]
    future = {(y + (m > 12), m if m <= 12 else 1)
              for y, m in [(last.year, last.month + k) for k in (1, 2)]}
    months = sorted({(d.year, d.month) for d in trading_days} | future)
    settlements = [s for (y, m) in months
                   if (s := get_settlement_day(y, m, day_set, trading_days[-1]))]
    prev_settle = max(s for s in settlements if s < data_day)
    next_settle = min(s for s in settlements if s >= data_day)
    window = [d for d in trading_days if prev_settle < d <= data_day]
    return {"本次結算日": next_settle, "上次結算日": prev_settle,
            "近月年月": f"{next_settle.year}{next_settle.month:02d}", "窗": window}
