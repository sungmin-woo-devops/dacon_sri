# eda_sweetviz_fixed.py
import pandas as pd
import sweetviz as sv
from pathlib import Path

# 경로 설정
ROOT = Path(__file__).resolve().parents[1]   # D:\workspace\dacon_sri
DATA_CSV = ROOT / "output" / "1. 수원서베이" / "suwon_2024_labeled.csv"
OUT_DIR  = ROOT / "output" / "1. 수원서베이" / "eda" / "sweetviz"
OUT_HTML = OUT_DIR / "sweetviz_report.html"

# 출력 디렉터리 생성
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 데이터 로드 (필요시 샘플링)
df = pd.read_csv(DATA_CSV, low_memory=False)
# df = df.sample(n=50000, random_state=42)  # 메모리 부담 시 활성화

# Sweetviz 분석 및 저장
report = sv.analyze(df)
report.show_html(str(OUT_HTML))  # Path → str 변환
print(f"Sweetviz report saved: {OUT_HTML}")
