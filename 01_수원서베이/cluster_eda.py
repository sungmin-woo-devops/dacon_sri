# cluster_eda.py
# 목적: suwon_2024_labeled.csv에 대해
#  1) 변수 상관 기반(열) 클러스터링 + 히트맵/덴드로그램
#  2) 응답자(행) 차원 축소(PCA/UMAP) + KMeans 군집 + 2D 시각화
#  3) 클러스터별 중요 변수 상위 랭킹 내보내기
# 의존성: pandas, numpy, scikit-learn, scipy, umap-learn(선택), matplotlib

# 해석 가이드
# variables_clusters.csv: 서로 강하게 상관되는 변수들이 같은 클러스터로 묶임 → 중복 변수 제거/요인화 후보
# respondents_clusters.csv + respondents_clusters_scatter.png: 응답자 패턴 군집 및 2D 시각화
# cluster_feature_importance_mi.csv: 군집을 가장 잘 구분하는 핵심 변수 Top-N 정렬(상위 30개부터 검토)

# 튜닝 포인트
# NA_COL_MAX, CAT_TOPK, HIGH_CARD로 컬럼 수 조절
# K_RANGE로 최적 k 탐색 범위 확대/축소
# UMAP이 과한 비선형이면 TRY_UMAP=False로 두고 PCA만 사용

from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.feature_selection import VarianceThreshold

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# ====== 경로/설정 ======
ROOT = Path(__file__).resolve().parents[1]   # D:\workspace\dacon_sri
DATA_CSV = ROOT / "output" / "1. 수원서베이" / "suwon_2024_labeled.csv"
OUTDIR   = ROOT / "output" / "1. 수원서베이" / "cluster"
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MAX_ROWS     = None       # 너무 크면 예: 120_000
NA_COL_MAX   = 0.6        # 결측률 60% 이상 컬럼 제외
CAT_TOPK     = 30         # 범주형 원-핫 인코딩 시 카테고리 상위 k만 사용
HIGH_CARD    = 2000       # 너무 많은 범주는 제외
N_PC         = 30         # PCA 주성분 수
K_RANGE      = range(2, 9)  # KMeans k 후보
UMAP_N      = 2           # umap 차원(2로 고정)
TRY_UMAP     = True       # umap-learn 설치되어 있으면 UMAP 우선

# ====== 유틸 ======
def coerce_numeric(s: pd.Series, min_ratio=0.9):
    """대부분 숫자면 float32로 변환"""
    x = pd.to_numeric(s, errors="coerce")
    return x.astype("float32") if x.notna().mean() >= min_ratio else s

def select_columns(df: pd.DataFrame):
    """EDA 친화적 컬럼 선택: 결측률/변동성/ID성/초고카디널리티 제거"""
    n = len(df)
    keep = []
    drop = []
    for c in df.columns:
        if c in ("PID",):  # 명백한 식별자 후보는 제외
            drop.append(c); continue
        s = df[c]
        na_rate = s.isna().mean()
        if na_rate >= NA_COL_MAX:
            drop.append(c); continue
        nunique = s.nunique(dropna=True)
        # ID성(거의 전부 유일)
        if nunique / max(1, n) >= 0.95:
            drop.append(c); continue
        # 초고카디널리티 문자열
        if s.dtype == object and nunique > HIGH_CARD:
            drop.append(c); continue
        # 상수열(변동성 없음)
        if nunique <= 1:
            drop.append(c); continue
        keep.append(c)
    return df[keep].copy(), drop

def one_hot_topk(df_obj: pd.DataFrame, topk=CAT_TOPK):
    """문자열/범주형 변수: 상위 topk 범주만 인코딩(나머지 'Others')"""
    mat_list = []
    new_cols = []
    for c in df_obj.columns:
        vc = df_obj[c].astype("string").fillna("(결측)").value_counts()
        cats = list(vc.index[:topk])
        col = df_obj[c].astype("string").fillna("(결측)")
        col = col.where(col.isin(cats), other="Others")
        oh = pd.get_dummies(col, prefix=c, dtype=np.uint8)
        mat_list.append(oh)
        new_cols.extend(list(oh.columns))
    if not mat_list:
        return pd.DataFrame(index=df_obj.index), []
    X = pd.concat(mat_list, axis=1)
    return X, new_cols

def best_k_by_silhouette(X, k_range=K_RANGE, random_state=RANDOM_STATE):
    best_k, best_score, best_labels = None, -1, None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    return best_k, best_score, best_labels

# ====== 1) 데이터 로드 & 기본 정리 ======
df = pd.read_csv(DATA_CSV, low_memory=False)
if MAX_ROWS:
    df = df.sample(n=min(MAX_ROWS, len(df)), random_state=RANDOM_STATE)

# 가중치 열 숫자화(있으면)
for wcol in ("ws", "wg"):
    if wcol in df.columns:
        df[wcol] = pd.to_numeric(df[wcol], errors="coerce").astype("float32")

# 숫자 강제 변환(대부분 숫자인 열만)
for c in df.columns:
    df[c] = coerce_numeric(df[c], min_ratio=0.9)

# 컬럼 선택(불필요/문제 컬럼 제거)
df_sel, dropped_cols = select_columns(df)
pd.Series(dropped_cols).to_csv(OUTDIR / "dropped_columns.csv", index=False, header=["dropped"])

# ====== 2) 변수(열) 상관 기반 클러스터링 ======
# 숫자열만 이용하여 상관행렬 계산
num_cols = [c for c in df_sel.columns if pd.api.types.is_numeric_dtype(df_sel[c])]
corr = pd.DataFrame()
if len(num_cols) >= 2:
    corr = df_sel[num_cols].corr(method="pearson").fillna(0.0)
    # 거리 = 1 - |corr|
    dist = 1 - np.abs(corr.values)
    # linkage는 1차원 condensed vector 필요
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    # 덴드로그램
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=[str(c) for c in corr.columns], leaf_rotation=90, leaf_font_size=8, color_threshold=0.7)
    plt.title("Variable Clustering (Hierarchical, 1-|corr|, average)")
    plt.tight_layout()
    plt.savefig(OUTDIR / "variables_dendrogram.png")
    plt.close()

    # 상관 히트맵(순서 정렬)
    order = dendrogram(Z, no_plot=True)["leaves"]
    corr_sorted = corr.iloc[order, :].iloc[:, order]
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_sorted.values, aspect="auto")
    plt.colorbar()
    plt.title("Correlation Heatmap (Variables sorted by clustering)")
    plt.xticks(range(len(order)), corr_sorted.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(order)), corr_sorted.index, fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTDIR / "variables_corr_heatmap.png")
    plt.close()

    # 예시로 군집 라벨 만들기(거리 임계값 0.5)
    clusters = fcluster(Z, t=0.5, criterion="distance")
    pd.DataFrame({"variable": corr.columns, "var_cluster": clusters}).to_csv(
        OUTDIR / "variables_clusters.csv", index=False, encoding="utf-8-sig"
    )
else:
    print("[WARN] 상관 기반 변수 클러스터링을 위한 숫자열이 부족합니다.")

# ====== 3) 응답자(행) 차원 축소 + 군집 ======
# 범주형 처리(상위 topk만 인코딩)
obj_cols = [c for c in df_sel.columns if df_sel[c].dtype == object]
X_cat, cat_names = one_hot_topk(df_sel[obj_cols], topk=CAT_TOPK) if obj_cols else (pd.DataFrame(index=df_sel.index), [])
# 숫자 부분
X_num = df_sel[num_cols].copy() if num_cols else pd.DataFrame(index=df_sel.index)

# 결측 대치 + 스케일
if not X_num.empty:
    num_imp = SimpleImputer(strategy="median")
    Xn = pd.DataFrame(num_imp.fit_transform(X_num), index=X_num.index, columns=X_num.columns)
    scaler = StandardScaler()
    Xn = pd.DataFrame(scaler.fit_transform(Xn), index=Xn.index, columns=Xn.columns)
else:
    Xn = pd.DataFrame(index=df_sel.index)

X = pd.concat([Xn, X_cat], axis=1)
# near-constant 제거
if not X.empty:
    vt = VarianceThreshold(1e-8)
    X_vt = vt.fit_transform(X)
    vt_cols = X.columns[vt.get_support(indices=True)]
    X = pd.DataFrame(X_vt, index=df_sel.index, columns=vt_cols)

# PCA
n_pc = min(N_PC, X.shape[1]) if not X.empty else 2
if n_pc < 2:
    print("[WARN] PCA 차원 축소를 위한 특성이 부족합니다.")
    emb_pca = np.zeros((len(df_sel), 2))
else:
    pca = PCA(n_components=n_pc, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    # 2D 투영
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    emb_pca = pca2.fit_transform(X)

# UMAP(선택)
emb_umap = None
if TRY_UMAP:
    try:
        import umap
        reducer = umap.UMAP(n_components=UMAP_N, n_neighbors=30, min_dist=0.1, random_state=RANDOM_STATE)
        emb_umap = reducer.fit_transform(X)
    except Exception as e:
        print(f"[INFO] UMAP 사용 불가: {e}")

# KMeans (PCA 2D 기준으로 군집; UMAP 있으면 UMAP 2D 사용)
emb = emb_umap if emb_umap is not None else emb_pca
best_k, sil, labels = best_k_by_silhouette(emb, k_range=K_RANGE, random_state=RANDOM_STATE)
km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init="auto").fit(emb)
labels = km.labels_

pd.DataFrame({"cluster": labels}).to_csv(OUTDIR / "respondents_clusters.csv", index=False)

# 2D 시각화
plt.figure(figsize=(7, 6))
plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=6)
plt.title(f"Respondent Clusters (k={best_k}, silhouette={sil:.3f})")
plt.tight_layout()
plt.savefig(OUTDIR / "respondents_clusters_scatter.png")
plt.close()

# ====== 4) 클러스터별 중요 변수 랭킹(간단 점수) ======
# 다중 클래스 → 각 클러스터(원-대-다) 라벨로 평균 Mutual Information 계산
y = labels
scores = []
# 특성 행렬은 X (선형/원핫 포함, 스케일 완료)
for j, col in enumerate(X.columns):
    xj = X.iloc[:, j]
    # 연속형/이진 모두 mutual_info 적용 가능
    try:
        mi = mutual_info_classif(xj.values.reshape(-1, 1), y, discrete_features='auto', random_state=RANDOM_STATE)
        scores.append((col, float(mi[0])))
    except Exception:
        scores.append((col, np.nan))

score_df = pd.DataFrame(scores, columns=["feature", "mutual_info"]).sort_values("mutual_info", ascending=False)
score_df.to_csv(OUTDIR / "cluster_feature_importance_mi.csv", index=False, encoding="utf-8-sig")

# 보너스: F-통계 기반 순위(연속형에 적합)
try:
    fvals, pvals = f_classif(X, y)
    f_df = pd.DataFrame({"feature": X.columns, "f_value": fvals, "p_value": pvals}).sort_values("f_value", ascending=False)
    f_df.to_csv(OUTDIR / "cluster_feature_importance_f.csv", index=False, encoding="utf-8-sig")
except Exception:
    pass

print("완료:")
print(f"- 변수 덴드로그램/히트맵: {OUTDIR}")
print(f"- 응답자 군집 결과/산점도: {OUTDIR}")
print(f"- 변수 중요도(MI/F): {OUTDIR}")
