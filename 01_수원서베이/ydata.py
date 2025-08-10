# ydata_safe.py
import pandas as pd
from ydata_profiling import ProfileReport

DATA_CSV = r"D:\workspace\dacon_sri\output\1. 수원서베이\suwon_2024_labeled.csv"
OUT_HTML = r"D:\workspace\dacon_sri\output\1. 수원서베이\ydata\suwon_2024_profile.html"

# 1) 로드: dtype 경고 억제 & 메모리 안정
df = pd.read_csv(DATA_CSV, low_memory=False)

# 2) 숫자형 강제 변환(대부분 숫자인 열만)
def coerce_numeric_inplace(df, min_ratio=0.9):
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= min_ratio:
            df[c] = s.astype("float32")
coerce_numeric_inplace(df)

# 3) 초고카디널리티 문자열은 category로
for c in df.select_dtypes(include=["object"]).columns:
    if df[c].nunique(dropna=True) > 5000:
        df[c] = df[c].astype("category")

# 4) 필요 시 샘플링(행이 아주 많을 때만 주석 해제)
# df = df.sample(n=120_000, random_state=42)

# 5) 프로파일링: 버전 호환되는 옵션만 사용
profile = ProfileReport(
    df,
    title="수원서베이 2024 프로파일링",
    minimal=True,          # 가장 중요: 무거운 연산 비활성
    explorative=False,
    interactions={"continuous": False, "targets": []},  # 현재 버전 스키마에 맞춤
    # correlations/상호작용/결측 히트맵 등은 minimal=True로 대부분 꺼짐
)

profile.to_file(OUT_HTML)
print("saved:", OUT_HTML)
