import json, re, numpy as np, statistics
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
d=json.load(open("/tmp/matrix.json"))
def parse(cell):
    m=re.match(r"(dei|rowp)_n(\d+)_rose(\w+)",cell); return (m.group(1),int(m.group(2)),m.group(3)) if m else None
g=defaultdict(lambda: defaultdict(list))
for r in d:
    p=parse(r["cell"])
    if p: g[p][r["tok"]].append(r["feas"])
ROSES=["dei","omnidir","rowp","uniform"]; FARMS=["dei","rowp"]
RLAB={"dei":"DEI rose","omnidir":"omnidir","rowp":"ROWP rose","uniform":"unidir"}
CMAP={"dei":"C0","omnidir":"C1","rowp":"C2","uniform":"C3"}
fig,ax=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
for a,farm in zip(ax,FARMS):
    for rose in ROSES:
        pts=sorted([(N,100*statistics.fmean(g[(farm,N,rose)]["IT21"])) for (f,N,r) in g if f==farm and r==rose])
        if pts: a.plot([N for N,_ in pts],[v for _,v in pts],"-o",color=CMAP[rose],ms=4,label=f"it21, {RLAB[rose]}")
    # native reference (avg over roses, ~100%)
    natpts=sorted(set(N for (f,N,r) in g if f==farm))
    natv=[100*statistics.fmean([x for rose in ROSES if (farm,N,rose) in g for x in g[(farm,N,rose)]["NAT"]]) for N in natpts]
    a.plot(natpts,natv,"--",color="k",lw=1.6,label="native (all roses)")
    a.set_xlabel("turbines $N$"); a.set_title(f"{farm.upper()}"); a.grid(alpha=0.25); a.set_ylim(40,103)
    a.legend(fontsize=7.5,loc="lower left",ncol=2)
ax[0].set_ylabel("it21 feasible restarts (%)")
fig.suptitle("Feasibility degrades with farm size: discovered it21 vs native ($30$ restarts/cell, strict $\\gamma_{\\min}=0.01$m)",fontsize=11)
plt.tight_layout(); plt.savefig("paper/funwake2/fig_feasibility.pdf",dpi=150); plt.savefig("results/fig_feasibility.png",dpi=130)
# summary
lowest=sorted([( (farm,N,rose), 100*statistics.fmean(g[(farm,N,rose)]["IT21"]) ) for (farm,N,rose) in g], key=lambda x:x[1])[:6]
print("lowest it21 feasibility cells:", [(f"{c[0]}_n{c[1]}_{c[2]}",round(p)) for c,p in lowest])
print("wrote fig_feasibility (feasibility vs N)")
