import sys, os, json
sys.path.insert(0,'dependencies/pixwake/src'); sys.path.insert(0,'funwake2'); sys.path.insert(0,'funwake2/seeds')
import jax; jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp, numpy as np
import evaluator as E, skeleton_v2 as S
from dei_layout import ProblemBenchmark
PIN='--pin' in sys.argv
cell=E.CELLS['dei_n50']; prob=json.load(open(cell['problem']))
D=float(prob['rotor_diameter']); ms=float(prob['min_spacing_m']); n=int(cell['n'])
sim=E.build_sim(prob)
wd,ws,wt=E._load_wind(prob if cell['rose'] is None else json.load(open(cell['rose'])))
boundary=jnp.array(prob['boundary_vertices'],dtype=jnp.float64)
x0,y0=S._wind_aware_init(boundary,ms,wd,ws,wt,n,0)
def aep_obj(x,y):
    r=sim(x,y,ws_amb=ws,wd_amb=wd,ti_amb=None); p=r.power()[:,:len(x)]; return -jnp.sum(p*wt[:,None])*8760/1e6
gox,goy=jax.grad(aep_obj,argnums=(0,1))(x0,y0)
alpha0=float(jnp.mean(jnp.abs(jnp.concatenate([gox,goy])))/D)
print("alpha0 =", repr(alpha0))
nat=E.load_schedule('funwake2/seeds/native.py')
if PIN:
    A=5.30170e-05
    sched=lambda st,tot,Dx,msx,nx,gmx,a0: nat(st,tot,Dx,msx,nx,gmx,A)
else:
    sched=nat
bm=ProblemBenchmark(cell['problem'])
def run():
    x,y=S.run_with_schedule(sched,sim,n,boundary,ms,wd,ws,wt,D,0.01,total_steps=6000,seed=0,zones=None)
    return bm.score(np.asarray(x),np.asarray(y))
a1=run(); a2=run()
print(f"AEP run1={a1:.10f} run2={a2:.10f} same-process|diff|={abs(a1-a2):.3e} PIN={PIN}")
