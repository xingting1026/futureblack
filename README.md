# futureblack — 期貨結算週期溢價看板

每個交易日 21:00（台北）自動更新的靜態頁面：**本結算週期**全部個股期貨標的的
週期溢價與大額特定法人未平倉市值，標出「溢價前30 ∩ 未平倉市值前30」交集名單，
並以台積電週期溢價 ≥ 0.2% 作為當月紅綠燈。

## 溢價怎麼算（與內部生產系統同構）

1. 每個交易日：09:05~13:25 每 5 分鐘一格共 53 格，每格取
   `期貨成交價 / 現貨成交價 − 1`（雙方各取該時刻前最後一筆成交）
2. 當日日均 → 落地 `data/daily/{日期}.csv`（**一天一檔增量快取**，commit 回 repo）
3. 週期溢價 = 結算窗內全部日均的平均
4. 大小期並存時以「特定法人方向性口數 × 乘數 × 收盤價」加權（權重全零退回乘數比）

已知與內部版的差異：公開 tick 只有成交價（內部版用買賣報價雙邊平均）、無前一日種子。

## 結構

```
src/build.py        主流程：日曆 → 對照表 → 補快取 → 因子 → 產頁
src/calendar_tw.py  交易日與第三個週三結算日曆
src/finmind.py      FinMind API 薄封裝（節流/重試；token 走環境變數）
data/mapping.csv    契約↔股票對照與乘數（每晚自動補新契約）
data/daily/         每日每契約溢價快取（Actions 自動 commit）
site/index.html     產出頁（GitHub Pages 部署，不進版控）
```

## 部署需求（一次性）

1. Repo Settings → Secrets → Actions → 新增 `FINMIND_TOKEN`（需 sponsor 權限的 token）
2. Repo Settings → Pages → Source 選 **GitHub Actions**

## 免責

本頁為個人研究紀錄，資料來自 FinMind / 期交所公開資訊，不構成投資建議。
