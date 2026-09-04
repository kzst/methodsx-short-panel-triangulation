#!/usr/bin/env python3
"""Regenerate all manuscript-facing tables and figures for the MethodsX R1 package."""
from __future__ import annotations
import argparse, math, shutil
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pct(x, digits=2):
    if x is None or pd.isna(x): return "—"
    q=Decimal("1").scaleb(-digits)
    v=(Decimal(str(float(x)))*Decimal(100)).quantize(q,rounding=ROUND_HALF_UP)
    return f"{v:.{digits}f}%"


def logbf(bf):
    if not np.isfinite(bf) or bf<=0: return "NA"
    x=math.log(float(bf)); return ">=40.00" if x>=40-1e-10 else f"{x:.2f}"


def fixed_tables(ref,tables):
    src=ref/"static_tables"
    for name in ["table_S1_specifications.csv","table_G1_glossary.csv","table_1_parameter_contract.csv","table_2_evidence_tiers.csv","table_3_troubleshooting.csv"]:
        shutil.copyfile(src/name,tables/name)


def diagnostics(ref,tables,figs):
    demo=pd.read_csv(ref/"demo_evidence.csv"); common=pd.read_csv(ref/"common_driver_sensitivity.csv")
    order=["x_linear","x_nonlinear","x_null","x_common"]; d=demo.set_index("variable").loc[order]
    methods=[("Classical panel test","classical_detect"),("Bayesian comparison","bayes_detect"),("Random Forest screen","rf_detect")]
    mat=np.array([[bool(d.loc[v,c]) for v in order] for _,c in methods],dtype=int)
    labels={"x_linear":"Linear lag","x_nonlinear":"Nonlinear lag","x_null":"Null predictor","x_common":"Common-driver proxy"}
    fig,ax=plt.subplots(figsize=(8.0,3.65)); ax.imshow(mat,cmap="Greys",vmin=0,vmax=1,aspect="auto")
    ax.set_yticks(range(3),[m[0] for m in methods]); ax.set_xticks(range(4),[labels[v] for v in order],rotation=17,ha="right")
    for i in range(3):
        for j in range(4): ax.text(j,i,"supported" if mat[i,j] else "not supported",ha="center",va="center",color="white" if mat[i,j] else "black")
    ax.set_xlabel("Planted signal class"); ax.set_title("Synthetic validation: method-specific support"); fig.tight_layout(); fig.savefig(figs/"figure_1_method_specific_support.png",dpi=200,bbox_inches="tight"); plt.close(fig)

    p=common["p_lag_bonf"].astype(float).to_numpy(); h=-np.log10(np.clip(p,1e-300,1))
    fig,ax=plt.subplots(figsize=(7.45,3.72)); ax.bar(range(3),h); ax.axhline(-math.log10(.05),linestyle="--",label="0.05 threshold")
    ax.set_xticks(range(3),["Unit effects\nonly","Unit + time\neffects","Unit + lagged\nCCE averages"]); ax.set_ylabel(r"$-\log_{10}$(lag-adjusted p-value)"); ax.set_title("Alternative common-shock controls remove a planted false lead-lag signal"); ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig(figs/"figure_2_common_driver_sensitivity.png",dpi=200,bbox_inches="tight"); plt.close(fig)

    pd.DataFrame({"Variable":order,"Planted form":d["functional_form"],"True lag":d["true_lag"].astype(int),"Lag-adjusted p":[f"{x:.3g}" for x in d["p_lag_bonf"]],"log BF10 (BIC)":[logbf(x) for x in d["bf10_bic_max"]],"RF gain":[f"{x:.2f}%" for x in d["rf_delta_mse_pct"]],"Shadow Q0.95":[f"{x:.3f}" for x in d["rf_shadow_q95"]],"Evidence tier":["triangulated in legacy three-vote diagnostic","nonlinear candidate","unsupported","unsupported"]}).to_csv(tables/"table_4_known_truth_diagnostic.csv",index=False)
    pd.DataFrame({"Specification":common["specification"],"Lag-adjusted p":[f"{x:.3g}" for x in common["p_lag_bonf"]],"log BF10 (BIC approximation)":[logbf(x) for x in common["bf10_bic_max"]]}).to_csv(tables/"table_5_common_driver_sensitivity.csv",index=False)


def tier_data(root,ref):
    op=root/"outputs/R1/operating_characteristics_R1.csv"; br=root/"outputs/R1/branch_and_tier_rates_R1.csv"
    if not(op.exists() and br.exists()): return pd.read_csv(ref/"final_tier_operating_characteristics.csv")
    o=pd.read_csv(op); b=pd.read_csv(br); m=o.merge(b[["cell_id","grouped_triangulated_power","grouped_replicated_single_power"]],on="cell_id",how="left"); rows=[]
    for _,r in m.iterrows():
        fam,sc=str(r["family"]),str(r["scenario"]); T,N,K=int(r["T"]),int(r["N"]),int(r["K"])
        if fam=="primary_global_null" and sc=="global_null" and T in (20,30,40): rows.append(["Global null",T,N,K,r["mean_grouped_triangulated_fdp"],np.nan,r["mean_grouped_replicated_single_fdp"],np.nan])
        elif fam=="primary_partial_null" and sc=="partial_null" and T in (20,30,40): rows.append(["Partial null",T,N,K,r["mean_grouped_triangulated_fdp"],r["grouped_triangulated_power"],r["mean_grouped_replicated_single_fdp"],r["grouped_replicated_single_power"]])
        elif fam=="stress_K" and sc=="partial_null" and T==30 and K in (30,60): rows.append(["Partial null",T,N,K,r["mean_grouped_triangulated_fdp"],r["grouped_triangulated_power"],r["mean_grouped_replicated_single_fdp"],r["grouped_replicated_single_power"]])
    return pd.DataFrame(rows,columns=["scenario","T","N","K","triangulated_fdp","triangulated_power","replicated_single_family_fdp","replicated_single_family_power"])


def validation(root,ref,tables,figs):
    tier=tier_data(root,ref); rows=[]
    for _,r in tier.iterrows(): rows.append([f"{r['scenario']}: T{int(r['T'])}, N{int(r['N'])}, K{int(r['K'])}",pct(r["triangulated_fdp"]),pct(r["triangulated_power"]),pct(r["replicated_single_family_fdp"]),pct(r["replicated_single_family_power"])])
    pd.DataFrame(rows,columns=["Scenario","Triangulated FDP","Triangulated power","Replicated-SF FDP","Replicated-SF power"]).to_csv(tables/"table_6_operating_characteristics.csv",index=False)
    c=tier[(tier["scenario"]=="Partial null")&(tier["K"]==6)].sort_values("T")
    fig,ax=plt.subplots(figsize=(6.6,3.5)); ax.plot(c["T"],100*c["triangulated_fdp"],marker="o",label="Triangulated: empirical FDP"); ax.plot(c["T"],100*c["triangulated_power"],marker="o",label="Triangulated: power"); ax.plot(c["T"],100*c["replicated_single_family_fdp"],marker="s",label="Replicated single-family: empirical FDP"); ax.plot(c["T"],100*c["replicated_single_family_power"],marker="s",label="Replicated single-family: power"); ax.set_xlabel("Time points per unit (T)"); ax.set_ylabel("Percent"); ax.set_title("Evidence-tier operating characteristics under the partial null"); ax.set_xticks(c["T"].astype(int).tolist()); ax.set_ylim(bottom=0); ax.grid(alpha=.25); ax.legend(fontsize=7.5,loc="upper left"); fig.tight_layout(); fig.savefig(figs/"figure_3_tier_operating_characteristics.png",dpi=220,bbox_inches="tight"); plt.close(fig)

    pg=root/"outputs/R1_targeted/benchmark_panel_granger_summary_R1.csv"; pc=root/"outputs/R1_targeted/benchmark_pcmci_summary_R1.csv"
    if pg.exists() and pc.exists():
        d1=pd.read_csv(pg); d2=pd.read_csv(pc); rr=[]
        for m,l in [("dh","Dumitrescu–Hurlin"),("jks","Split-panel jackknife")]:
            g=d1[(d1["method"]==m)&(d1["scenario"]=="global_null")].set_index("T"); p=d1[(d1["method"]==m)&(d1["scenario"]=="partial_null")].set_index("T")
            for T in (20,30,40): rr.append([l,T,g.loc[T,"false_positive_rate"],p.loc[T,"power_linear"],p.loc[T,"power_nonlinear"],p.loc[T,"empirical_fdr"]])
        g=d2[d2["scenario"]=="global_null"].set_index("T"); p=d2[d2["scenario"]=="partial_null"].set_index("T")
        for T in (20,30,40): rr.append(["PCMCI-ParCorr",T,g.loc[T,"false_positive_rate"],p.loc[T,"power_linear"],p.loc[T,"power_nonlinear"],p.loc[T,"empirical_fdr"]])
        bench=pd.DataFrame(rr,columns=["method","T","global_null_fpr","partial_null_linear_power","partial_null_nonlinear_power","partial_null_empirical_fdr"])
    else: bench=pd.read_csv(ref/"external_benchmarks.csv")
    bench["method"]=bench["method"].replace({"Dumitrescu-Hurlin":"Dumitrescu–Hurlin"})
    for col in ["global_null_fpr","partial_null_linear_power","partial_null_nonlinear_power","partial_null_empirical_fdr"]: bench[col]=bench[col].map(pct)
    bench.columns=["Method","T","Global-null FPR","Partial-null linear power","Partial-null nonlinear power","Partial-null empirical FDR"]; bench.to_csv(tables/"table_7_external_benchmarks.csv",index=False)

    nc=root/"outputs/R1/null_calibration_summary_R1.csv"
    if nc.exists():
        x=pd.read_csv(nc); rr=[]
        for T,g in x.groupby("T"): rr.append([int(T),g["classical_raw_size"].iloc[0],g["rf_circular_false_positive"].iloc[0],g["rf_block_false_positive"].min(),g["rf_block_false_positive"].max(),g["mean_rf_circular_p"].iloc[0],g["mean_rf_block_p"].min(),g["mean_rf_block_p"].max()])
        n=pd.DataFrame(rr,columns=["T","classical_raw_size","rf_circular_fpr","rf_block_fpr_min","rf_block_fpr_max","mean_circular_p","mean_block_p_min","mean_block_p_max"])
    else: n=pd.read_csv(ref/"null_calibration_summary.csv")
    rr=[]
    for _,r in n.sort_values("T").iterrows(): rr.append([int(r["T"]),pct(r["classical_raw_size"]),pct(r["rf_circular_fpr"]),f"{100*r['rf_block_fpr_min']:.2f}–{100*r['rf_block_fpr_max']:.2f}%",f"{r['mean_circular_p']:.3f}",f"{r['mean_block_p_min']:.3f}–{r['mean_block_p_max']:.3f}"])
    pd.DataFrame(rr,columns=["T","Classical raw size","RF circular FPR","RF block FPR range (L=2–4)","Mean circular p","Mean block p range"]).to_csv(tables/"table_8_null_calibration.csv",index=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default=str(Path(__file__).resolve().parents[1])); ap.add_argument("--out",default="manuscript_assets"); a=ap.parse_args(); root=Path(a.root).resolve(); out=Path(a.out); out=out if out.is_absolute() else root/out; tables=out/"tables"; figs=out/"figures"; tables.mkdir(parents=True,exist_ok=True); figs.mkdir(parents=True,exist_ok=True); ref=root/"outputs/reference"
    fixed_tables(ref,tables); diagnostics(ref,tables,figs); validation(root,ref,tables,figs)
    dep=pd.read_csv(ref/"branch_dependence_summary.csv"); rf=pd.read_csv(ref/"rf_fidelity_setting_agreement.csv"); allr=rf[(rf["setting"]=="mc_30_19")&(rf["truth"]=="all")].iloc[0]
    (out/"validation_notes.txt").write_text(f"Classical-Bayesian phi range, global null: {dep[dep['scenario']=='Global null']['phi_classical_by_bayes'].min():.2f}–{dep[dep['scenario']=='Global null']['phi_classical_by_bayes'].max():.2f}\nClassical-Bayesian phi range, partial null: {dep[dep['scenario']=='Partial null']['phi_classical_by_bayes'].min():.2f}–{dep[dep['scenario']=='Partial null']['phi_classical_by_bayes'].max():.2f}\nRF 30/19 vs 500/20 majority-decision agreement: {100*allr['majority_decision_agreement']:.1f}%\n",encoding="utf-8")
    print(f"Manuscript assets written to {out}")

if __name__=="__main__": main()
