# p_cluster_eda.py
# 개선점:
# - GUI 백엔드 비활성(Agg)로 콘솔/배치 안전 실행
# - 변수 너무 많을 때 상위 분산 변수만 선택(기본 120개)
# - 변수 클러스터링: 1-|corr| 거리 -> scipy.linkage + fcluster, 히트맵/덴드로그램 저장
# - 응답자 군집: 결측 중앙값 대치 + 표준화 + PCA(2D 투영) + KMeans(k 자동, silhouette)
# - 모든 산출물 저장: output/1. 수원서베이/eda/*

import os
os.environ["MPLBACKEND"] = "Agg"      # 반드시 pyplot, seaborn 임포트 이전
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dataprep.eda import create_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# ===============================
# 경로
# ===============================
ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "output" / "1. 수원서베이" / "suwon_2024_labeled.csv"
EDA_DIR  = ROOT / "output" / "1. 수원서베이" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# 파라미터
# ===============================
MAX_NUM_VARS_FOR_CORR = 120   # 상관/히트맵/덴드로그램에 사용할 숫자 변수 개수 상한
K_CANDIDATES = range(2, 11)   # KMeans k 후보
RANDOM_STATE = 42

# ===============================
# 1) 데이터 로드
# ===============================
df = pd.read_csv(DATA_CSV, low_memory=False)

# ===============================
# 2) Dataprep 리포트
# ===============================
report_path = EDA_DIR / "dataprep" / "dataprep_report.html"
report_path.parent.mkdir(parents=True, exist_ok=True)
create_report(df, title="수원서베이 2024").save(str(report_path))
print(f"[EDA Report] {report_path}")

# ===============================
# 3) 변수 간 상관 기반 클러스터링 (상위 분산 변수만)
# ===============================
# 1) 숫자열만 후보
num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols_all) >= 2:
    # 2) 분산 상위 선별
    var_rank = df[num_cols_all].var(numeric_only=True).sort_values(ascending=False)
    sel_cols = var_rank.head(MAX_NUM_VARS_FOR_CORR).index.tolist()

    # 3) 서브셋 추출
    Xcorr_raw = df[sel_cols].copy()

    # 4) 전부 NaN 컬럼 제거
    all_nan_cols = [c for c in Xcorr_raw.columns if Xcorr_raw[c].isna().all()]
    if all_nan_cols:
        Xcorr_raw = Xcorr_raw.drop(columns=all_nan_cols)

    # 5) 중복 컬럼명 제거(첫 번째만 사용)
    if Xcorr_raw.columns.duplicated().any():
        Xcorr_raw = Xcorr_raw.loc[:, ~Xcorr_raw.columns.duplicated()]

    # 6) 숫자 보장(혹시 모를 dtype 섞임 방지)
    Xcorr = Xcorr_raw.apply(pd.to_numeric, errors="coerce")

    # 7) 여전히 열이 2 미만이면 스킵
    if Xcorr.shape[1] < 2:
        print("[WARN] 유효 숫자열이 부족하여 변수 클러스터링을 건너뜁니다.")
    else:
        from sklearn.impute import SimpleImputer
        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
        from scipy.spatial.distance import squareform
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt

        # 8) 중앙값 대치
        imp = SimpleImputer(strategy="median")
        Xcorr_imputed = imp.fit_transform(Xcorr)
        Xcorr = pd.DataFrame(Xcorr_imputed, columns=Xcorr.columns, index=df.index)

        # 9) 상관계수 → 1-|corr| 거리
        corr = Xcorr.corr(method="pearson")
        # 안정화: NaN/Inf 제거, 대칭/대각/클리핑 보장
        C = corr.to_numpy(dtype=float)
        C[~np.isfinite(C)] = 0.0               # NaN/Inf -> 0
        np.fill_diagonal(C, 1.0)               # corr(ii)=1
        # 거리 = 1 - |corr|
        D = 1.0 - np.abs(C)
        # 수치 안정화
        D = 0.5 * (D + D.T)                    # 강제 대칭
        np.fill_diagonal(D, 0.0)               # 대각 0
        D = np.clip(D, 0.0, 2.0)               # 음수 제거(<=0), 상한 2

        from scipy.spatial.distance import squareform
        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

        condensed = squareform(D, checks=False)
        Z = linkage(condensed, method="average")

        # 10) 덴드로그램
        fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
        dendrogram(Z, labels=corr.columns, leaf_rotation=90, leaf_font_size=7, ax=ax, color_threshold=0.7)
        ax.set_title("Variable Clustering (Hierarchical, 1-|corr|, average)")
        fig.tight_layout(); fig.savefig(EDA_DIR / "variables_dendrogram.png"); plt.close(fig)

        # 11) 히트맵(정렬)
        order = dendrogram(Z, no_plot=True)["leaves"]
        corr_sorted = corr.iloc[order, :].iloc[:, order]
        fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
        sns.heatmap(corr_sorted, ax=ax, cmap="coolwarm", center=0)
        ax.set_title("Correlation Heatmap (Variables sorted by clustering)")
        fig.tight_layout(); fig.savefig(EDA_DIR / "variables_corr_heatmap.png"); plt.close(fig)

        # 12) 클러스터 라벨 저장
        clusters = fcluster(Z, t=0.5, criterion="distance")
        pd.DataFrame({"variable": corr.columns, "var_cluster": clusters}).to_csv(
            EDA_DIR / "variables_clusters.csv", index=False, encoding="utf-8-sig"
        )
else:
    print("[WARN] 숫자형 변수가 부족합니다.")

# ===============================
# 4) 응답자 차원 축소 + 군집(K 최적화)
# ===============================
# 숫자 + 간단 범주 처리(원핫은 비용 큼 → 우선 숫자만 사용 권장)
num_cols = df.select_dtypes(include="number").columns
X = df[num_cols].copy()

# 결측 중앙값 대치 + 표준화
imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X)
scaler = StandardScaler()
X_std = scaler.fit_transform(X_imp)

# PCA 2D 투영(시각화용)
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
emb2 = pca2.fit_transform(X_std)

# 최적 k 선택(실루엣 최대)
best_k, best_score, best_labels = None, -1, None
for k in K_CANDIDATES:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(emb2)
    try:
        score = silhouette_score(emb2, labels)
    except Exception:
        score = -1
    if score > best_score:
        best_k, best_score, best_labels = k, score, labels

# 저장
resp_df = pd.DataFrame(emb2, columns=["PC1", "PC2"])
resp_df["cluster"] = best_labels
resp_df.to_csv(EDA_DIR / "respondents_clusters.csv", index=False, encoding="utf-8-sig")

# 산점도
fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
scatter = ax.scatter(resp_df["PC1"], resp_df["PC2"], c=resp_df["cluster"], s=8)
ax.set_title(f"Respondent Clusters (k={best_k}, silhouette={best_score:.3f})")
fig.tight_layout()
fig.savefig(EDA_DIR / "respondents_clusters_scatter.png")
plt.close(fig)

print("[DONE] 변수/응답자 클러스터링 산출 완료")
print(f"- 히트맵/덴드로그램: {EDA_DIR}")
print(f"- respondents_clusters.csv / scatter.png: {EDA_DIR}")
