import sys, hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from queuesim.simulate import simulate, weekly_totals, Policy, World

def h(logs):
    m = hashlib.sha256()
    for k in ["leads","attempts","policies"]:
        m.update(pd.util.hash_pandas_object(logs[k].drop(columns=[c for c in ["single_owner","new_script"] if c in logs[k]]), index=True).values.tobytes())
    return m.hexdigest()[:16]

def run(args):
    world, pol, seed = args
    logs,_ = simulate(8, seed, Policy(ownership=pol), World(name=world, overlap=world))
    return world, pol, seed, weekly_totals(logs), h(logs)

if __name__ == "__main__":
    jobs=[(w,p,s) for w in ["harm","neutral","rescue"] for p in ["off","all"] for s in range(500,500+int(sys.argv[1]))]
    with ProcessPoolExecutor(6) as ex: res=list(ex.map(run,jobs))
    df=pd.DataFrame([dict(world=w,pol=p,seed=s,hash=hh,**t) for w,p,s,t,hh in res])
    print(df[df.pol=="off"].groupby("seed").hash.nunique().to_dict())
    piv=df.pivot_table(index=["world","seed"],columns="pol",values=["contacted_leads","quotes","issued","collected","agent_minutes"])
    for m in ["contacted_leads","quotes","issued","collected","agent_minutes"]:
        d=(piv[(m,"all")]-piv[(m,"off")]).groupby(level=0)
        print(m, d.mean().round(2).to_dict(), (d.std()/np.sqrt(d.count())).round(2).to_dict())
    print(df[df.pol=='off'][['contacted_leads','quotes','issued','collected','call_center_cost_mxn']].mean().round(1).to_dict())
