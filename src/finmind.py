# -*- coding: utf-8 -*-
"""FinMind API 薄封裝：節流、重試、token 從環境變數讀（絕不寫進 repo）。"""
import os
import time

import pandas as pd
import requests

API_URL = "https://api.finmindtrade.com/api/v4/data"
SLEEP_BETWEEN_CALLS = 0.6      # 秒；上限 6000/hr，這個節奏約 3,000/hr
RETRY_WAIT = 60                # API 失敗等一分鐘再試一次

n_calls = 0


def fetch(dataset, **params):
    """一次 API 呼叫 → DataFrame。失敗重試一次，再失敗回空表（呼叫端自行決定略過或中止）。"""
    global n_calls
    token = os.environ["FINMIND_TOKEN"]
    for attempt in (1, 2):
        n_calls += 1
        try:
            resp = requests.get(API_URL, params={**params, "dataset": dataset, "token": token},
                                timeout=300)
            body = resp.json()
            if resp.status_code == 200 and body.get("status") == 200:
                time.sleep(SLEEP_BETWEEN_CALLS)
                return pd.DataFrame(body.get("data", []))
            print(f"  [finmind] {dataset} {params.get('data_id','')} -> "
                  f"{resp.status_code}/{body.get('status')} {body.get('msg')}", flush=True)
        except Exception as error:
            print(f"  [finmind] {dataset} exception: {error}", flush=True)
        if attempt == 1:
            time.sleep(RETRY_WAIT)
    return pd.DataFrame()
