# make_viz_compact.py
# 목적: outputs 폴더(또는 압축 해제 폴더)에 있는 요약 CSV들을 읽어
#       서브플롯/그리드 기반의 "용량 작은" 시각화를 생성
# 사용 예:
#   python make_viz_compact.py --in_dir ./extracted --out_dir ./viz_small --top_k 12

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------
# 유틸
# ---------------------------------------------
def find_csv(root: Path, name_contains: str):
    cand = sorted([p for p in root.rglob("*.csv") if name_contains in p.name])
    return cand[0] if cand else None

def read_csv_opt(path: Path):
    if path is None: return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def set_korean_font_if_any():
    # 한글 폰트 설정(선택). 없으면 무시됨.
    try:
        import matplotlib
        # 사용 환경에 따라 원하는 폰트 지정 가능: 'Malgun Gothic', 'AppleGothic', 'NanumGothic'
        plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'sans-serif']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

# ---------------------------------------------
# 그리드 플로팅(서브플롯 허용 환경 가정)
# ---------------------------------------------
def grid_bar(ax, x_labels, values, title, ylabel=None, rotate=0):
    ax.bar(x_labels, values)
    ax.set_title(title, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel)
    if rotate:
        ax.set_xticklabels(x_labels, rotation=rotate, ha='right')
    ax.grid(axis='y', alpha=0.3)

def grid_heatmap(ax, df, title):
    im = ax.imshow(df.values.astype(float), aspect='auto')
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(df.shape[1])); ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(df.shape[0])); ax.set_yticklabels(df.index, fontsize=8)
    # colorbar는 전체 피겨에서 1개만 추가 권장 → 외부에서 처리

# ---------------------------------------------
# 각 모듈 시각화
# ---------------------------------------------
def viz_likert(df_likert: pd.DataFrame, pdf: PdfPages, out_dir: Path, top_k:int):
    if df_likert is None or df_likert.empty: return
    lk = df_likert.copy()
    if "weighted_NSS_pp" in lk.columns:
        lk = lk.sort_values("weighted_NSS_pp", ascending=False)
    if top_k and top_k > 0:
        lk = lk.head(top_k)

    # 1) 변수별 Top/Neutral/Bottom 그리드
    n = len(lk); cols = 3; rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten() if n>1 else [axes]
    for i, (_, r) in enumerate(lk.iterrows()):
        x = ["Top2","Neutral","Bottom2"]
        y = [r.get("weighted_top2_%", np.nan), r.get("weighted_neutral_%", np.nan), r.get("weighted_bottom2_%", np.nan)]
        grid_bar(axes[i], x, y, f"{r['variable']}", ylabel="Weighted %")
    for j in range(i+1, rows*cols):
        axes[j].axis('off')
    fig.suptitle("Likert — Top/Neutral/Bottom (weighted %)", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out_dir/"likert_TNB_grid.png", dpi=160)
    pdf.savefig(fig); plt.close(fig)

    # 2) NSS 랭킹
    fig, ax = plt.subplots(figsize=(max(6, 0.4*len(lk)), 4))
    grid_bar(ax, lk["variable"].astype(str).tolist(), lk["weighted_NSS_pp"].astype(float).tolist(),
             "Likert NSS (Top2 - Bottom2, weighted pp)", ylabel="pp", rotate=45)
    fig.tight_layout()
    fig.savefig(out_dir/"likert_NSS_rank.png", dpi=160)
    pdf.savefig(fig); plt.close(fig)

def viz_onl(df_onl: pd.DataFrame, pdf: PdfPages, out_dir: Path, top_k:int):
    if df_onl is None or df_onl.empty: return
    vars_ord = sorted(df_onl["variable"].astype(str).unique().tolist())
    if top_k and top_k>0:
        vars_ord = vars_ord[:top_k]
    n = len(vars_ord); cols=3; rows=int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten() if n>1 else [axes]
    for i, v in enumerate(vars_ord):
        sub = df_onl[df_onl["variable"].astype(str)==v].copy()
        sub["level_order"] = pd.to_numeric(sub["level"], errors="coerce")
        sub = sub.sort_values(["level_order","level"])
        grid_bar(axes[i], sub["level"].astype(str).tolist(), sub["weighted_%"].astype(float).tolist(),
                 f"{v}", ylabel="Weighted %")
    for j in range(i+1, rows*cols):
        axes[j].axis('off')
    fig.suptitle("Ordinal-nonLikert — Weighted distribution", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out_dir/"ordinal_nonlikert_grid.png", dpi=160)
    pdf.savefig(fig); plt.close(fig)

def viz_ranked(df_ranked: pd.DataFrame, pdf: PdfPages, out_dir: Path):
    if df_ranked is None or df_ranked.empty: return
    rk = df_ranked.copy().sort_values("weighted_mean_score", ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.4*len(rk)), 4))
    grid_bar(ax, rk["variable"].astype(str).tolist(), rk["weighted_mean_score"].astype(float).tolist(),
             "Ranked — weighted mean score", ylabel="Score", rotate=45)
    fig.tight_layout()
    fig.savefig(out_dir/"ranked_weighted_mean_score.png", dpi=160)
    pdf.savefig(fig); plt.close(fig)

def viz_cat_ordinal(df_ordcat: pd.DataFrame, pdf: PdfPages, out_dir: Path, top_k:int):
    if df_ordcat is None or df_ordcat.empty: return
    vars_ord = sorted(df_ordcat["variable"].astype(str).unique().tolist())
    if top_k and top_k>0:
        vars_ord = vars_ord[:top_k]
    n = len(vars_ord); cols=3; rows=int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten() if n>1 else [axes]
    for i, v in enumerate(vars_ord):
        sub = df_ordcat[df_ordcat["variable"].astype(str)==v].copy()
        sub["level_order"] = pd.to_numeric(sub["level"], errors="coerce")
        sub = sub.sort_values(["level_order","level"])
        grid_bar(axes[i], sub["level"].astype(str).tolist(), sub["weighted_%"].astype(float).tolist(),
                 f"{v}", ylabel="Weighted %")
    for j in range(i+1, rows*cols):
        axes[j].axis('off')
    fig.suptitle("Categorical-ordinal — Weighted distribution", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out_dir/"categorical_ordinal_grid.png", dpi=160)
    pdf.savefig(fig); plt.close(fig)

def viz_spearman(df_spear: pd.DataFrame, pdf: PdfPages, out_dir: Path):
    if df_spear is None or df_spear.empty: return
    for tgt in sorted(df_spear["target"].astype(str).unique()):
        sub = df_spear[df_spear["target"].astype(str)==tgt].copy()
        sub = sub.sort_values("spearman_rho", ascending=False)
        fig, ax = plt.subplots(figsize=(max(6, 0.4*len(sub)), 4))
        grid_bar(ax, sub["var"].astype(str).tolist(), sub["spearman_rho"].astype(float).tolist(),
                 f"Spearman rho vs {tgt}", ylabel="rho", rotate=45)
        fig.tight_layout()
        fig.savefig(out_dir/f"spearman_{tgt}.png", dpi=160)
        pdf.savefig(fig); plt.close(fig)

def viz_nominal(df_nom: pd.DataFrame, pdf: PdfPages, out_dir: Path, top_k:int):
    if df_nom is None or df_nom.empty: return
    vars_nom = sorted(df_nom["variable"].astype(str).unique().tolist())
    if top_k and top_k>0:
        vars_nom = vars_nom[:top_k]
    n = len(vars_nom); cols=3; rows=int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten() if n>1 else [axes]
    for i, v in enumerate(vars_nom):
        sub = df_nom[df_nom["variable"].astype(str)==v]
        grid_bar(axes[i], sub["level"].astype(str).tolist(), sub["weighted_%"].astype(float).tolist(),
                 f"{v}", ylabel="Weighted %", rotate=45)
    for j in range(i+1, rows*cols):
        axes[j].axis('off')
    fig.suptitle("Nominal — Weighted distribution", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out_dir/"nominal_grid.png", dpi=160)
    pdf.savefig(fig); plt.close(fig)

def viz_binary(df_bin: pd.DataFrame, df_bin_corr: pd.DataFrame, pdf: PdfPages, out_dir: Path, top_k:int):
    if df_bin is not None and not df_bin.empty:
        b = df_bin.copy().sort_values("weighted_%(=1)", ascending=False)
        if top_k and top_k>0: b = b.head(top_k)
        fig, ax = plt.subplots(figsize=(max(6, 0.4*len(b)), 4))
        grid_bar(ax, b["variable"].astype(str).tolist(), b["weighted_%(=1)"].astype(float).tolist(),
                 "Binary prevalence (weighted %)", ylabel="%(=1)", rotate=90)
        fig.tight_layout()
        fig.savefig(out_dir/"binary_prevalence_weighted.png", dpi=160)
        pdf.savefig(fig); plt.close(fig)

    if df_bin_corr is not None and not df_bin_corr.empty:
        for tgt in sorted(df_bin_corr["target"].astype(str).unique()):
            sub = df_bin_corr[df_bin_corr["target"].astype(str)==tgt].copy()
            sub = sub.sort_values("pearson_r", ascending=False)
            if top_k and top_k>0: sub = sub.head(top_k)
            fig, ax = plt.subplots(figsize=(max(6, 0.4*len(sub)), 4))
            grid_bar(ax, sub["var"].astype(str).tolist(), sub["pearson_r"].astype(float).tolist(),
                     f"Binary vs {tgt} — Pearson r", ylabel="r", rotate=90)
            fig.tight_layout()
            fig.savefig(out_dir/f"binary_corr_{tgt}.png", dpi=160)
            pdf.savefig(fig); plt.close(fig)

def viz_heatmaps(gu_q3_path: Path, gu_q7_path: Path, pdf: PdfPages, out_dir: Path):
    has_cb = False
    if gu_q3_path and gu_q3_path.exists():
        df = pd.read_csv(gu_q3_path, index_col=0)
        fig, ax = plt.subplots(figsize=(6,4))
        grid_heatmap(ax, df, "GU × q3 — frequency heatmap")
        if not has_cb:
            plt.colorbar(ax.images[0], ax=ax); has_cb=True
        fig.tight_layout(); fig.savefig(out_dir/"GU_x_q3_heatmap.png", dpi=160); pdf.savefig(fig); plt.close(fig)
    if gu_q7_path and gu_q7_path.exists():
        df = pd.read_csv(gu_q7_path, index_col=0)
        fig, ax = plt.subplots(figsize=(6,4))
        grid_heatmap(ax, df, "GU × q7 — frequency heatmap")
        if not has_cb:
            plt.colorbar(ax.images[0], ax=ax); has_cb=True
        fig.tight_layout(); fig.savefig(out_dir/"GU_x_q7_heatmap.png", dpi=160); pdf.savefig(fig); plt.close(fig)

# ---------------------------------------------
# 메인
# ---------------------------------------------
def main(args):
    set_korean_font_if_any()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 파일 찾기
    df_likert = read_csv_opt(find_csv(in_dir, "likert_summary"))
    df_onl = read_csv_opt(find_csv(in_dir, "ordinal_nonlikert_summary"))
    df_ranked = read_csv_opt(find_csv(in_dir, "ranked_summary"))
    df_nom = read_csv_opt(find_csv(in_dir, "categorical_nominal_summary_meta")) or \
             read_csv_opt(find_csv(in_dir, "categorical_nominal_summary_quick"))
    df_ordcat = read_csv_opt(find_csv(in_dir, "categorical_ordinal_summary"))
    df_spear = read_csv_opt(find_csv(in_dir, "categorical_ordinal_spearman"))
    df_bin = read_csv_opt(find_csv(in_dir, "binary_summary"))
    df_bin_corr = read_csv_opt(find_csv(in_dir, "binary_vs_targets_corr"))
    gu_q3_ct = find_csv(in_dir, "GU_x_q3_crosstab")
    gu_q7_ct = find_csv(in_dir, "GU_x_q7_crosstab")

    pdf_path = out_dir / "SRI_viz_report_compact.pdf"
    with PdfPages(pdf_path) as pdf:
        viz_likert(df_likert, pdf, out_dir, args.top_k)
        viz_onl(df_onl, pdf, out_dir, args.top_k)
        viz_ranked(df_ranked, pdf, out_dir)
        viz_cat_ordinal(df_ordcat, pdf, out_dir, args.top_k)
        viz_spearman(df_spear, pdf, out_dir)
        viz_nominal(df_nom, pdf, out_dir, args.top_k)
        viz_binary(df_bin, df_bin_corr, pdf, out_dir, args.top_k)
        viz_heatmaps(gu_q3_ct, gu_q7_ct, pdf, out_dir)

    print(f"[OK] Compact visuals saved to: {out_dir}")
    print(f"[OK] PDF: {pdf_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="요약 CSV가 들어있는 폴더(압축 해제 경로 또는 outputs)")
    ap.add_argument("--out_dir", type=str, default="./viz_small", help="시각화 결과 저장 폴더")
    ap.add_argument("--top_k", type=int, default=12, help="변수 그리드에서 상위 K개만 표시(용량 절감)")
    args = ap.parse_args()
    main(args)
