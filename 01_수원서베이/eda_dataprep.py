# eda_dataprep_smart.py
# 목적: 컬럼이 너무 많을 때 자동으로 "필터링 + 청크 분할"하여 DataPrep EDA HTML 리포트를 여러 개로 생성
# - 우선순위 변수(공통문항/핵심 키워드)는 항상 포함
# - 결측률 높은 컬럼/변동성 없는 컬럼/ID성 고유값 컬럼 자동 제외
# - 남은 컬럼은 MAX_COLS_PER_REPORT 단위로 분할 저장
# - index.html에 생성된 모든 리포트를 링크

from dataprep.eda import create_report
from pathlib import Path
import pandas as pd
import numpy as np
import re

# =====================[ 설정값 ]=====================
ROOT = Path(__file__).resolve().parents[1]   # D:\workspace\dacon_sri
DATA_CSV = ROOT / "output" / "1. 수원서베이" / "suwon_2024_labeled.csv"

OUT_DIR       = ROOT / "output" / "1. 수원서베이" / "eda" / "dataprep"
OUT_INDEX_HTML= OUT_DIR / "index.html"  # 생성된 리포트 목차
TITLE_PREFIX  = "수원서베이 2024"

# 1) 리포트 분할 크기
MAX_COLS_PER_REPORT = 80   # 보고서 1개에 넣을 최대 컬럼 수(환경에 맞게 조정)
MAX_REPORTS         = 20   # 안전장치(과다 생성 방지)

# 2) 제외 규칙(기본)
NA_RATE_DROP_TH   = 0.6    # 결측률 60% 이상 제거
LOW_VAR_NUNIQUE   = 1      # 모든 값 동일인 컬럼 제거
ID_UNIQUE_RATIO   = 0.95   # 고유값/행수 ≥ 0.95이면 ID/키로 판단해 제외
MAX_CAT_UNIQUE    = 2000   # 범주형 유니크 값이 너무 많으면 제외(텍스트 열 등)

# 3) 우선 포함 규칙(라벨/변수명에 키워드 매칭)
PRIORITY_KEYWORDS = [
    r"\bSCORE\d+\b", r"\bMHQ\d+\b", r"\bMQ\d+\b",
    "정책", "만족", "삶의 질", "행복", "환경", "가중치", r"\bws\b", r"\bwg\b"
]
# 별도의 공통문항 리스트 파일이 있으면 가산
COMMON_VARS_TXT = ROOT / "output" / "1. 수원서베이" / "common_vars.txt"
# ====================================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 데이터 로드
df = pd.read_csv(DATA_CSV, low_memory=False)

# 메타 계산
n_rows = len(df)
meta = []
for col in df.columns:
    s = df[col]
    na_rate = float(s.isna().mean())
    nunique = int(s.nunique(dropna=True))
    # 숫자 여부 판단(대부분이 숫자면 숫자로 간주)
    s_num = pd.to_numeric(s, errors="coerce")
    numeric_ratio = s_num.notna().mean()
    dtype = "numeric" if numeric_ratio >= 0.98 else "categorical"
    unique_ratio = nunique / max(1, n_rows)
    meta.append({
        "col": col,
        "dtype": dtype,
        "na_rate": na_rate,
        "nunique": nunique,
        "unique_ratio": unique_ratio
    })
meta_df = pd.DataFrame(meta)

# 0) 강제 유지 컬럼 (가중치 열 등)
force_keep = [c for c in ["ws", "wg"] if c in df.columns]

# 1) 제외 컬럼 규칙 적용
drop_mask = (
    (meta_df["na_rate"] >= NA_RATE_DROP_TH) |
    ((meta_df["dtype"] == "numeric") & (meta_df["nunique"] <= LOW_VAR_NUNIQUE)) |
    (meta_df["unique_ratio"] >= ID_UNIQUE_RATIO) |
    ((meta_df["dtype"] == "categorical") & (meta_df["nunique"] > MAX_CAT_UNIQUE))
)
# force_keep은 제외 대상에서 복구
to_drop = set(meta_df.loc[drop_mask, "col"]) - set(force_keep)

# 2) 우선 포함 컬럼
priority_cols = set(force_keep)
# 공통문항 파일 읽기(있으면)
if COMMON_VARS_TXT.exists():
    for line in COMMON_VARS_TXT.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and name in df.columns:
            priority_cols.add(name)

# 키워드 매칭
def matches_keywords(col: str) -> bool:
    for pat in PRIORITY_KEYWORDS:
        if re.search(pat, col, flags=re.IGNORECASE):
            return True
    return False

for col in df.columns:
    if matches_keywords(col):
        priority_cols.add(col)

# 3) 최종 대상 컬럼 풀
candidate_cols = [c for c in df.columns if c not in to_drop]

# 우선 포함 컬럼을 앞으로 배치
priority_cols = [c for c in candidate_cols if c in priority_cols]
other_cols    = [c for c in candidate_cols if c not in priority_cols]

# 4) 메모리 절약: 초고카디널리티 객체는 category로
obj_cols = df[other_cols].select_dtypes(include=["object"]).columns
for c in obj_cols:
    if df[c].nunique(dropna=True) > 5000:
        df[c] = df[c].astype("category")

# 5) 리포트 청크 구성
reports = []
# 5-1) 우선 리포트(핵심/공통문항 중심)
if priority_cols:
    reports.append(("00_priority", priority_cols[:MAX_COLS_PER_REPORT]))

# 5-2) 나머지 컬럼을 청크로 분할
chunk_cols = other_cols
start = 0
idx = 1
while start < len(chunk_cols) and len(reports) < MAX_REPORTS:
    end = min(start + MAX_COLS_PER_REPORT, len(chunk_cols))
    subset = chunk_cols[start:end]
    if subset:
        reports.append((f"{idx:02d}_part", subset))
        idx += 1
    start = end

# 6) 리포트 생성
links = []
for name, cols in reports:
    sub = df[cols].copy()
    title = f"{TITLE_PREFIX} | {name} ({len(cols)} vars)"
    out_html = OUT_DIR / f"report_{name}.html"
    try:
        create_report(sub, title=title).save(str(out_html))
        links.append((title, out_html.name, len(cols)))
        print(f"[OK] {title} -> {out_html}")
    except Exception as e:
        print(f"[FAIL] {title}: {e}")

# 7) 목차(index.html) 생성
with open(OUT_INDEX_HTML, "w", encoding="utf-8") as f:
    f.write("<html><head><meta charset='utf-8'><title>EDA Index</title></head><body>\n")
    f.write(f"<h1>{TITLE_PREFIX} - DataPrep EDA Index</h1>\n")
    f.write(f"<p>Total columns: {df.shape[1]:,} / After filter: {len(candidate_cols):,} / Reports: {len(links)}</p>\n")
    f.write("<ol>\n")
    for title, fname, ncol in links:
        f.write(f"<li><a href='{fname}' target='_blank'>{title}</a> — {ncol} cols</li>\n")
    f.write("</ol>\n</body></html>\n")

print(f"[INDEX] {OUT_INDEX_HTML}")
