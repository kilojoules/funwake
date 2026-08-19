import json, re, numpy as np, statistics
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
d=json.load(open("/tmp/matrix.json"))
def parse(cell):
    m=re.match(r"(dei|rowp)_n(\d+)_rose(\w+)",cell)
    return (m.group(1),int(m.group(2)),m.group(3)) if m else None
g=defaultdict(lambda: defaultdict(dict))
for r in d: g[r["cell"]][r["tok"]][r["seed"]]=(r["aep"],r["feas"])
rows={}
for cell,toks in g.items():
    p=parse(cell)
    if not p or "IT21" not in toks or "NAT" not in toks: continue
    seeds=sorted(set(toks["IT21"])&set(toks["NAT"]))
    both=[s for s in seeds if toks["IT21"][s][1] and toks["NAT"][s][1]]
    itf=statistics.fmean(toks["IT21"][s][1] for s in seeds)
    if len(both)<3:
        rows[p]={"delta":np.nan,"se":0,"itf":itf}; continue
    diffs=[toks["IT21"][s][0]-toks["NAT"][s][0] for s in both]
    nm=statistics.fmean(toks["NAT"][s][0] for s in both)
    rows[p]={"delta":100*statistics.fmean(diffs)/nm,
             "se":100*(statistics.pstdev(diffs)/len(diffs)**0.5)/nm if len(diffs)>1 else 0,
             "itf":itf,"nboth":len(both)}
ROSES=["dei","omnidir","rowp","uniform"]; FARMS=["dei","rowp"]
RLAB={"dei":"DEI rose","omnidir":"omnidirectional","rowp":"ROWP rose","uniform":"unidirectional"}
# ---- Fig-4: feasible-both Delta% vs N ----
fig,ax=plt.subplots(len(FARMS),len(ROSES),figsize=(13,6),sharex="col")
for i,farm in enumerate(FARMS):
    for j,rose in enumerate(ROSES):
        a=ax[i,j]; pts=sorted([(N,v) for (f,N,r),v in rows.items() if f==farm and r==rose])
        if pts:
            Ns=[N for N,_ in pts]; dl=[v["delta"] for _,v in pts]; se=[v["se"] for _,v in pts]
            a.errorbar(Ns,dl,yerr=se,fmt="-o",color="C3",ms=4,capsize=2,lw=1.4)
            for N,v in pts:
                if v["itf"]<1.0: a.scatter([N],[v["delta"]],marker="x",s=45,color="k",zorder=5)
        a.axhline(0,ls="--",color="k",lw=1)
        if i==0: a.set_title(RLAB[rose],fontsize=10)
        if j==0: a.set_ylabel(f"{farm.upper()}\nit21 $-$ native (%)",fontsize=9)
        if i==len(FARMS)-1: a.set_xlabel("turbines $N$")
        a.grid(alpha=0.25)
fig.suptitle("Generalization: feasible-paired-mean AEP advantage of it21 over native (30 seeds/cell; error bars $=$ se; $\\times$ = it21 feasibility $<$100%)",fontsize=10)
plt.tight_layout(); plt.savefig("paper/funwake2/fig_generalization.pdf",dpi=150); plt.savefig("results/fig_generalization.png",dpi=130)
# ---- heatmap: Delta% (top) + it21 feasibility% (bottom) ----
fig2,ax2=plt.subplots(2,2,figsize=(11,7))
for k,farm in enumerate(FARMS):
    Ns=sorted(set(N for (f,N,r) in rows if f==farm))
    D=np.full((len(Ns),len(ROSES)),np.nan); F=np.full((len(Ns),len(ROSES)),np.nan)
    for a_i,N in enumerate(Ns):
        for a_j,rose in enumerate(ROSES):
            if (farm,N,rose) in rows:
                D[a_i,a_j]=rows[(farm,N,rose)]["delta"]; F[a_i,a_j]=100*rows[(farm,N,rose)]["itf"]
    im=ax2[0,k].imshow(D,cmap="RdYlGn",vmin=-0.1,vmax=0.1,aspect="auto")
    ax2[0,k].set_title(f"{farm.upper()}: it21 $-$ native (%)"); plt.colorbar(im,ax=ax2[0,k],fraction=0.046)
    im2=ax2[1,k].imshow(F,cmap="RdYlGn",vmin=0,vmax=100,aspect="auto")
    ax2[1,k].set_title(f"{farm.upper()}: it21 feasible restarts (%)"); plt.colorbar(im2,ax=ax2[1,k],fraction=0.046)
    for row,M,fmt in [(0,D,"+.02f"),(1,F,".0f")]:
        for a_i in range(len(Ns)):
            for a_j in range(len(ROSES)):
                if not np.isnan(M[a_i,a_j]): ax2[row,k].text(a_j,a_i,format(M[a_i,a_j],fmt),ha="center",va="center",fontsize=6.5)
        ax2[row,k].set_xticks(range(len(ROSES))); ax2[row,k].set_xticklabels([RLAB[r] for r in ROSES],rotation=30,ha="right",fontsize=7.5)
        ax2[row,k].set_yticks(range(len(Ns))); ax2[row,k].set_yticklabels(Ns); ax2[row,k].set_ylabel("$N$")
plt.tight_layout(); plt.savefig("paper/funwake2/fig_gen_heatmap.pdf",dpi=150); plt.savefig("results/fig_gen_heatmap.png",dpi=130)
resolved=[v["delta"] for v in rows.values() if not np.isnan(v["delta"])]
pos=sum(1 for v in rows.values() if not np.isnan(v["delta"]) and v["delta"]-2*v["se"]>0)
print(f"DEI+ROWP cells: {len(resolved)}; Δ% range [{min(resolved):+.3f},{max(resolved):+.3f}] mean {statistics.fmean(resolved):+.4f}%")
print(f"cells with Δ% > 2·se above zero (resolved positive): {pos}/{len(resolved)}")
print("wrote fig_generalization + fig_gen_heatmap (feasible-both)")
