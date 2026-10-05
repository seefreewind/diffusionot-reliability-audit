#!/usr/bin/env python3
"""Diffusion-scale and non-lineage growth-covariate diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
ARM=ROOT/"results/phase1R/REIMPLEMENTED_ID_PRESERVING_ARM"
OUT=ROOT/"results/phase1RS"
SEEDS=[11,23,47,71,101]


def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def eta_squared(values,groups):
 y=np.asarray(values,dtype=np.float64);g=np.asarray(groups).astype(str)
 den=np.square(y-y.mean()).sum()
 if den==0:return float('nan')
 num=0.0
 for label in np.unique(g):
  m=g==label;num+=int(m.sum())*(float(y[m].mean())-float(y.mean()))**2
 return float(num/den)


def main():
 index=pd.read_csv(ARM/"cell_index.tsv",sep='\t')
 scale_rows=[]
 for seed in SEEDS:
  sd=ARM/f"seed_{seed}"
  with np.load(sd/"latent_input.npz",allow_pickle=False) as z:
   x=z['latent_ae'].astype(np.float32);times=z['time_label'].astype(str).astype(int)
  with np.load(sd/"dynamics_outputs.npz",allow_pickle=False) as z:d=float(z['diffusion_shared_d'])
  dim=x.shape[1];dt=1.0;rms=float(np.sqrt(2*d*dt*dim))
  for day_a,day_b in [(2,4),(4,6)]:
   target=x[times==day_b];starts=x[times==day_a]
   tree=cKDTree(target)
   target_dist,target_ind=tree.query(target,k=32,workers=1)
   target_loo30=np.asarray([dist[ind!=i][29] for i,(dist,ind) in enumerate(zip(target_dist,target_ind))])
   nn_to_target=tree.query(starts,k=1,workers=1)[0]
   p=ROOT/f"results/phase1R/snapshot_qc/seed_{seed}/{day_a}_to_{day_b}_endpoints.npz"
   with np.load(p,allow_pickle=False) as z:
    pred=z['model_endpoints'].astype(np.float64);source=x[z['source_row_index']].astype(np.float64)
   scale_rows.append({"seed":seed,"transition":f"{day_a}_to_{day_b}","d":d,"latent_dimension":dim,"model_delta_t":dt,
    "expected_diffusion_rms_sqrt_2d_dt_D":rms,"target_30nn_radius_median":float(np.median(target_loo30)),
    "target_30nn_radius_q99_support_boundary":float(np.quantile(target_loo30,.99)),
    "empirical_source_to_target_nearest_distance_median":float(np.median(nn_to_target)),
    "frozen_model_endpoint_source_displacement_median":float(np.median(np.linalg.norm(pred-source,axis=1))),
    "diffusion_rms_over_target_30nn_median":rms/float(np.median(target_loo30)),
    "diffusion_rms_over_support_radius_q99":rms/float(np.quantile(target_loo30,.99)),
    "diffusion_rms_over_source_target_nn_displacement":rms/float(np.median(nn_to_target)),
    "endpoint_path":str(p.relative_to(ROOT))})
 scale_path=OUT/"diffusion_scale_audit.tsv"
 pd.DataFrame(scale_rows).to_csv(scale_path,sep='\t',index=False)
 growth_path=OUT/"growth_stability.tsv"
 gd=pd.read_csv(growth_path,sep='\t')
 lib=index.library.astype(str).to_numpy();tp=index.timepoint.to_numpy(dtype=int)
 batch_eta=[];time_eta=[]
 for row in gd.to_dict('records'):
  seed=int(row['seed']);day=row['time_stratum']
  with np.load(ARM/f"seed_{seed}/dynamics_outputs.npz",allow_pickle=False) as z:g=z['growth'].astype(np.float64)
  mask=np.ones(len(g),dtype=bool) if day=='all' else tp==int(day.replace('day',''))
  batch_eta.append(eta_squared(g[mask],lib[mask]))
  time_eta.append(eta_squared(g,tp) if day=='all' else float('nan'))
 gd['growth_vs_library_batch_eta_squared']=batch_eta
 gd['growth_vs_timepoint_eta_squared']=time_eta
 gd['library_batch_covariate']='categorical Library ID; non-lineage technical grouping'
 gd['expression_burden_covariate']='sum of frozen top-3000 log1p expression values; proxy, not raw UMI library size'
 gd.to_csv(growth_path,sep='\t',index=False)
 summary={"status":"COMPLETE_LINEAGE_BLIND","diffusion_rows":len(scale_rows),"growth_covariate_rows":len(gd),
          "diffusion_scale_sha256":sha256(scale_path),"growth_stability_sha256":sha256(growth_path),"lineage_information_opened":False}
 (OUT/"diffusion_growth_covariate_summary.json").write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
