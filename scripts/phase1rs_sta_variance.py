#!/usr/bin/env python3
"""Independent frozen-checkpoint STA replicates and MC/model variance audit."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist, pdist
from scipy.stats import entropy

ROOT=Path(__file__).resolve().parents[1]
ARM=ROOT/"results/phase1R/REIMPLEMENTED_ID_PRESERVING_ARM"
OUT=ROOT/"results/phase1RS/sta_variance"
SEEDS=[11,23,47,71,101]
GRID10=np.asarray([100,300,1000,3000,10000],dtype=np.int64)
GRID20=np.asarray([100,300,1000,3000,10000,20000],dtype=np.int64)
CELL_SET_SEED=101
CELL_BATCH=10
DT=.01
TOL=.01


def sha256(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def audit_rows(index:pd.DataFrame,day:int)->np.ndarray:
 pool=index.index[index.timepoint.eq(day)].to_numpy(dtype=np.int64)
 rng=np.random.default_rng(CELL_SET_SEED+day)
 return np.sort(rng.choice(pool,size=100,replace=False))


def curves_for_day(func,mapper,starts,day,t0,nsteps,ntraj,grid,rng):
 nclass=len(mapper['classes']); result=np.empty((len(starts),len(grid),nclass),dtype=np.float32)
 endpoints=np.empty((len(starts),ntraj,2),dtype=np.float32)
 class_col={c:i for i,c in enumerate(mapper['classes'])}
 fitted=mapper['model'][-1].classes_
 sigma=np.sqrt(2*float(func.d.detach().item())*DT)
 for lo in range(0,len(starts),CELL_BATCH):
  hi=min(lo+CELL_BATCH,len(starts)); block=starts[lo:hi]
  states=np.repeat(block[:,None,:].astype(np.float32),ntraj,axis=1).reshape(-1,2)
  for step in range(nsteps):
   t=torch.tensor(t0+step*DT,dtype=torch.float32)
   with torch.no_grad():
    velocity=func.hyper_net1(t,torch.from_numpy(states)).numpy()
   noise=rng.normal(0,sigma,size=(len(block),ntraj,2)).astype(np.float32).reshape(-1,2)
   states=states+velocity*DT+noise
  probs=mapper['model'].predict_proba(states)
  ordered=np.zeros((len(states),nclass),dtype=np.float32)
  for j,c in enumerate(fitted):ordered[:,class_col[c]]=probs[:,j]
  ordered=ordered.reshape(len(block),ntraj,nclass)
  for gi,n in enumerate(grid):result[lo:hi,gi]=ordered[:,:int(n)].mean(axis=1)
  endpoints[lo:hi]=states.reshape(len(block),ntraj,2)
 return result,endpoints


def gaussian_mmd2(x,y,bandwidth):
 scale=2.0*bandwidth*bandwidth
 kxx=np.exp(-cdist(x,x,metric='sqeuclidean')/scale)
 kyy=np.exp(-cdist(y,y,metric='sqeuclidean')/scale)
 kxy=np.exp(-cdist(x,y,metric='sqeuclidean')/scale)
 return float((kxx.sum()-np.trace(kxx))/(len(x)*(len(x)-1))+
              (kyy.sum()-np.trace(kyy))/(len(y)*(len(y)-1))-2*kxy.mean())


def summarize_curves(curves,grid):
 maxp=curves.max(axis=2)
 normh=entropy(curves,axis=2)/np.log(curves.shape[2])
 dp=np.abs(np.diff(maxp,axis=1));dh=np.abs(np.diff(normh,axis=1))
 passed=(dp.max(axis=0)<TOL)&(dh.max(axis=0)<TOL)
 first=next((int(grid[i+1]) for i,v in enumerate(passed) if v),None)
 return {"status":"STA_CONVERGED" if first is not None else "STA_MONTE_CARLO_UNSTABLE",
         "primary_trajectory_count":first,"max_probability_change_by_step":dp.max(axis=0).tolist(),
         "max_normalized_entropy_change_by_step":dh.max(axis=0).tolist(),
         "curves":curves,"maxp":maxp,"normh":normh}


def load_original(seed,index):
 p=ROOT/f"results/phase1R/sta_mc/seed_{seed}_sta_mc_curves.npz"
 with np.load(p,allow_pickle=False) as z:
  ids=z['audit_cell_id'].astype(str);days=z['audit_timepoint'].astype(int)
  grid=z['trajectory_grid'].astype(int);curves=z['class_probability_curves'].astype(np.float32)
  classes=z['class_order'].astype(str).tolist()
 expected=[]
 for day in [2,4]:expected.extend(index.original_cell_id.iloc[audit_rows(index,day)].astype(str).tolist())
 if ids.tolist()!=expected:raise RuntimeError(f"seed {seed}: previous STA cells do not match frozen audit rows")
 if not np.array_equal(days,np.repeat([2,4],100)):raise RuntimeError(f"seed {seed}: previous STA timepoint order mismatch")
 return grid,curves,classes


def new_replicate(seed,replicate,ntraj):
 sys.path.insert(0,str(ROOT/"external/DiffusionOT"))
 import utility
 sd=ARM/f"seed_{seed}"
 ckpt=torch.load(sd/"ckpt_Mouse.pth",map_location="cpu")
 func=utility.RUOT(in_out_dim=2,hidden_dim=16,n_hiddens=4,activation="Tanh",d=.001).cpu()
 func.load_state_dict(ckpt['func_state_dict']);func.eval()
 mapper=joblib.load(ROOT/f"results/phase1R/terminal_classifier/seed_{seed}/day6_state_mapper.joblib")
 with np.load(sd/"latent_input.npz",allow_pickle=False) as z:
  x=z['latent_ae'].astype(np.float32);ids=z['original_cell_id'].astype(str);times=z['time_label'].astype(str).astype(int)
 index=pd.read_csv(ARM/"cell_index.tsv",sep='\t')
 if not np.array_equal(ids,index.original_cell_id.astype(str).to_numpy()):raise RuntimeError('STA canonical IDs changed')
 grid=GRID10 if ntraj==10000 else GRID20
 curves=[];audit_ids=[];audit_days=[];endpoint_rows=[]
 start=time.perf_counter()
 for day,t0,nsteps in [(2,0.,200),(4,1.,100)]:
  rows=audit_rows(index,day)
  if not np.all(times[rows]==day):raise RuntimeError('Wrong fixed STA day rows')
  rng=np.random.default_rng(20260928+seed*1_000_000+replicate*10_000+day)
  batch_curves,batch_endpoints=curves_for_day(func,mapper,x[rows],day,t0,nsteps,ntraj,grid,rng)
  curves.extend(list(batch_curves));endpoint_rows.extend(list(batch_endpoints))
  audit_ids.extend(ids[rows].tolist());audit_days.extend([day]*len(rows))
 curves=np.stack(curves)
 endpoint_rows=np.stack(endpoint_rows)
 stat=summarize_curves(curves,grid)
 out=OUT/f"seed_{seed}";out.mkdir(parents=True,exist_ok=True)
 path=out/f"replicate_{replicate}_n{ntraj}.npz"
 np.savez_compressed(path,audit_cell_id=np.asarray(audit_ids),audit_timepoint=np.asarray(audit_days,dtype=np.int8),
                     trajectory_grid=grid,class_probability_curves=curves.astype(np.float32),
                     terminal_latent_endpoints=endpoint_rows,
                     normalized_entropy=stat['normh'].astype(np.float32),maximum_class_probability=stat['maxp'].astype(np.float32),
                     class_order=np.asarray(mapper['classes']),seed=np.asarray(seed),replicate=np.asarray(replicate),n_trajectories=np.asarray(ntraj))
 summary={k:v for k,v in stat.items() if k not in {'curves','maxp','normh'}}
 summary.update({"seed":seed,"replicate":replicate,"n_trajectories":ntraj,"cells":200,
                 "runtime_seconds":time.perf_counter()-start,"lineage_information_opened":False,
                 "curves_path":str(path.relative_to(ROOT)),"curves_sha256":sha256(path)})
 sp=out/f"replicate_{replicate}_n{ntraj}_summary.json"
 sp.write_text(json.dumps(summary,indent=2)+'\n')
 return grid,curves,mapper['classes'],summary,endpoint_rows


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 index=pd.read_csv(ARM/"cell_index.tsv",sep='\t')
 run_data={};rep_rows=[]
 for seed in SEEDS:
  g0,c0,classes=load_original(seed,index)
  rep_curves=[c0];reps=[]
  s0=json.loads((ROOT/f"results/phase1R/sta_mc/seed_{seed}_sta_mc_summary.json").read_text())
  stat0=summarize_curves(c0,g0)
  reps.append({"seed":seed,"replicate":1,"n_trajectories":10000,"status":stat0['status'],"primary_trajectory_count":stat0['primary_trajectory_count'],"runtime_seconds":"reused frozen Phase1R independent run"})
  for rep in [2,3]:
   _,curves,classes_new,summary,endpoints=new_replicate(seed,rep,10000)
   if classes_new!=classes:raise RuntimeError('Classifier class order changed')
   rep_curves.append(curves);reps.append({"seed":seed,"replicate":rep,"n_trajectories":10000,"status":summary['status'],"primary_trajectory_count":summary['primary_trajectory_count'],"runtime_seconds":summary['runtime_seconds']})
   if rep==2: endpoint_rep2=endpoints
   if rep==3: endpoint_rep3=endpoints
  rep_matrix=np.stack(rep_curves)
  run_data[seed]={"curves":rep_matrix,"classes":classes}
  # The original Phase 1R run did not retain latent endpoints. Compare the two
  # newly generated independent endpoint samples, stratified by starting day.
  endpoint_distances={}
  for day in [2,4]:
   m=np.repeat([2,4],100)==day
   sample_a_rng=np.random.default_rng(810_000_000+seed*1000+day)
   sample_b_rng=np.random.default_rng(820_000_000+seed*1000+day)
   ea=endpoint_rep2[m].reshape(-1,2);eb=endpoint_rep3[m].reshape(-1,2)
   ia=sample_a_rng.choice(len(ea),size=2000,replace=False);ib=sample_b_rng.choice(len(eb),size=2000,replace=False)
   xa,xb=ea[ia],eb[ib]
   bw=float(np.median(pdist(np.vstack([xa,xb]),metric='euclidean')))
   endpoint_distances[f"day{day}_replicate_2_vs_3_gaussian_mmd2"]=gaussian_mmd2(xa,xb,bw)
   endpoint_distances[f"day{day}_endpoint_mmd_bandwidth"]=bw
  run_data[seed]['endpoint_distribution_distances']=endpoint_distances
  for r in reps:rep_rows.append(r)
  # One bounded 20k confirmation for the two originally unstable seeds.
  if seed in [71,101]:
   _,c20,classes20,summary20,_=new_replicate(seed,4,20000)
   rep_rows.append({"seed":seed,"replicate":"20k_confirmation_once","n_trajectories":20000,"status":summary20['status'],"primary_trajectory_count":summary20['primary_trajectory_count'],"runtime_seconds":summary20['runtime_seconds']})
   run_data[seed]['confirmation20k']=c20
   run_data[seed]['confirmation20k_summary']=summary20
  # Within-model Monte Carlo uncertainty across three independent 10k runs.
  prob=rep_matrix[:,:,-1,:]
  ent=entropy(np.maximum(prob,1e-12),axis=-1)/np.log(prob.shape[-1])
  pair_tv=[]
  for i in range(3):
   for j in range(i+1,3):pair_tv.append(.5*np.abs(prob[i]-prob[j]).sum(axis=1))
  run_data[seed]['within']={"mean_probability_sd":float(np.mean(prob.std(axis=0,ddof=1))),
                            "mean_entropy_sd":float(np.mean(ent.std(axis=0,ddof=1))),
                            "mean_pairwise_cell_TV":float(np.mean(pair_tv)),
                            "sta_converged_replicates":int(sum(r['status']=='STA_CONVERGED' for r in reps))}
 # Variance decomposition on matched 200 audit cells x classes at 10k.
 across=np.stack([run_data[s]['curves'][:,:,-1,:].mean(axis=0) for s in SEEDS])
 between=float(across.var(axis=0,ddof=1).mean())
 within=float(np.mean([run_data[s]['curves'][:,:,-1,:].var(axis=0,ddof=1).mean() for s in SEEDS]))
 detail=[]
 for seed in SEEDS:
  x=run_data[seed]['within']
  detail.append({"seed":seed,**x,"within_model_probability_variance":float(run_data[seed]['curves'][:,:,-1,:].var(axis=0,ddof=1).mean()),
                 "between_seed_probability_variance_component":between,"pooled_within_model_variance":within,
                 "between_to_within_variance_ratio":between/within if within else float('inf'),
                 **run_data[seed]['endpoint_distribution_distances'],
                 "variance_dominance": "between_model_seed" if between>within else "within_model_MC"})
  if seed in [71,101]:
   detail[-1]['20k_confirmation_status']=run_data[seed]['confirmation20k_summary']['status']
   detail[-1]['20k_max_probability_delta_10k_to_20k']=run_data[seed]['confirmation20k_summary']['max_probability_change_by_step'][-1]
 out=OUT
 pd.DataFrame(rep_rows).to_csv(out/"sta_replicate_status.tsv",sep='\t',index=False)
 detail_path=out/"sta_variance_decomposition.tsv"
 pd.DataFrame(detail).to_csv(detail_path,sep='\t',index=False)
 print(json.dumps({"status":"COMPLETE_LINEAGE_BLIND","replicate_rows":len(rep_rows),"sta_converged_10k_replicates":sum(r['status']=='STA_CONVERGED' for r in rep_rows if r['n_trajectories']==10000),
                   "between_seed_probability_variance":between,"within_model_MC_variance":within,"output":str(detail_path.relative_to(ROOT)),"sha256":sha256(detail_path),"lineage_information_opened":False},indent=2))


if __name__=='__main__':main()
