import json
from pathlib import Path

# JSON → Python 객체
with open(Path.cwd() / "output" / "codebooks" / "2024년 3분기 패널조사 데이터_코드북.json", "r", encoding="utf-8") as f:
    data = json.load(f)  # dict 또는 list로 로드됨

# 딕셔너리 접근
print(data["meta"]["quarter"])
print(data["variables"]["q1"]["label"])

print(data["variables"])

for d in data["variables"]:
    if d.endswith('_op'):
        print(d)