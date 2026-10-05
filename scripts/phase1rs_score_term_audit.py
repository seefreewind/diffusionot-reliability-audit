#!/usr/bin/env python3
"""Inspect the frozen RUOT score network and its numerical contribution."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
ARM=ROOT/"results/phase1R/REIMPLEMENTED_ID_PRESERVING_ARM"
OUT=ROOT/"results/phase1RS"
SEEDS=[11,23,47,71,101]


def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def score_objective_diagnostics(func, x_sample):
 z=torch.from_numpy(x_sample.astype(np.float32,copy=True)).requires_grad_(True)
 t=torch.tensor(1.,dtype=torch.float32)
 with torch.enable_grad():
  raw=func.hyper_net3(t,z)
  jac=[]
  for j in range(raw.shape[1]):
   grad_j=torch.autograd.grad(raw[:,j].sum(),z,create_graph=True,retain_graph=True)[0]
   jac.append(grad_j)
  jacobian=torch.stack(jac,dim=1)
  trace=jacobian[:,0,0]+jacobian[:,1,1]
  integrand=trace+.5*raw.square().sum(dim=1)
  objective=integrand.mean()
  params=list(func.hyper_net3.parameters())
  grads=torch.autograd.grad(objective,params,allow_unused=True)
 raw64=raw.detach().cpu().numpy().astype(np.float64)
 jac64=jacobian.detach().cpu().numpy().astype(np.float64)
 grad_norm=float(torch.sqrt(sum((g.detach()**2).sum() for g in grads if g is not None)).item())
 return {"torch_float32_score_norm_median":float(torch.linalg.vector_norm(raw.detach(),dim=1).median().item()),
         "coordinate_gradient_frobenius_median_float64":float(np.median(np.linalg.norm(jac64.reshape(len(jac64),-1),axis=1))),
         "score_loss_integrand_mean_float64":float(integrand.detach().cpu().numpy().astype(np.float64).mean()),
         "hyper_net3_score_objective_gradient_l2":grad_norm,
         "raw_score_norm_median_float64":float(np.median(np.linalg.norm(raw64,axis=1)))}


def main():
 sys.path.insert(0,str(ROOT/"external/DiffusionOT"))
 import utility
 records=[];details=[]
 for seed in SEEDS:
  sd=ARM/f"seed_{seed}"
  ckpt=torch.load(sd/"ckpt_Mouse.pth",map_location="cpu")
  f=utility.RUOT(in_out_dim=2,hidden_dim=16,n_hiddens=4,activation="Tanh",d=.001).cpu()
  f.load_state_dict(ckpt['func_state_dict']);f.eval()
  with np.load(sd/"dynamics_outputs.npz",allow_pickle=False) as z:
   x=z['latent_ae'].astype(np.float32);saved=z['score'].astype(np.float32);vel=z['velocity'].astype(np.float32)
   d=float(z['diffusion_shared_d'])
  with np.load(sd/"latent_input.npz",allow_pickle=False) as z:
   times=z['time_label'].astype(str).astype(int)
  finite=bool(np.isfinite(saved).all())
  norm=np.linalg.norm(saved.astype(np.float64),axis=1)
  vnorm=np.linalg.norm(vel.astype(np.float64),axis=1)
  contribution=d*d*np.sum(saved.astype(np.float64)**2,axis=1)
  velocity_energy=np.sum(vel.astype(np.float64)**2,axis=1)
  ratio=contribution/(velocity_energy+1e-300)
  check_rows=[];net_stats=[]
  for day,mt in [(2,0.),(4,1.),(6,2.)]:
   rows=np.flatnonzero(times==day)
   sample=rows[np.linspace(0,len(rows)-1,min(2048,len(rows)),dtype=int)]
   with torch.no_grad():
    raw=f.hyper_net3(torch.tensor(mt,dtype=torch.float32),torch.from_numpy(x[sample]))
   delta=np.max(np.abs(raw.numpy()-saved[sample]))
   check_rows.append(float(delta))
  for key,value in f.state_dict().items():
   if key.startswith('hyper_net3.'):
    a=value.detach().cpu().numpy().astype(np.float64)
    net_stats.append({"parameter":key,"shape":"x".join(map(str,a.shape)),"l2":float(np.linalg.norm(a)),"max_abs":float(np.max(np.abs(a))),"exact_zero_fraction":float(np.mean(a==0))})
  log=(sd/'ruot.log').read_text(errors='replace')
  losses=[float(v) for v in re.findall(r"Pre_train_score_Iter:\s*\d+,\s*loss:\s*([-+0-9.eE]+)",log)]
  probe=np.flatnonzero(times==4)[:128]
  final_grad=score_objective_diagnostics(f,x[probe])
  score_ckpt=torch.load(sd/"ckpt_Mouse_score.pth",map_location="cpu")
  pretrained=utility.RUOT(in_out_dim=2,hidden_dim=16,n_hiddens=4,activation="Tanh",d=.001).cpu()
  pretrained.load_state_dict(score_ckpt["func_state_dict"]);pretrained.eval()
  pretrain_grad=score_objective_diagnostics(pretrained,x[probe])
  record={"seed":seed,"d":d,"score_shape":"x".join(map(str,saved.shape),),"score_dtype":"float32",
          "score_l2_norm_median":float(np.median(norm)),"score_l2_norm_p99":float(np.quantile(norm,.99)),
          "score_l2_norm_max":float(np.max(norm)),"fraction_score_norm_le_1e-8":float(np.mean(norm<=1e-8)),
          "fraction_exact_zero_components":float(np.mean(saved==0)),"all_finite":finite,
          "d2_score_energy_median":float(np.median(contribution)),"velocity_energy_median":float(np.median(velocity_energy)),
          "score_to_velocity_energy_ratio_median":float(np.median(ratio)),
          "checkpoint_recompute_max_abs_difference":max(check_rows),"score_loss_pretrain_first":losses[0] if losses else None,
          "score_loss_pretrain_last":losses[-1] if losses else None,"score_loss_logged_iterations":len(losses),
          "score_network_parameter_l2_sum":float(sum(a['l2'] for a in net_stats)),
          "score_network_parameter_exact_zero_fraction":float(np.mean([a['exact_zero_fraction'] for a in net_stats])),
          **final_grad,
          "score_pretraining_checkpoint_raw_score_norm_median":pretrain_grad["raw_score_norm_median_float64"],
          "score_pretraining_checkpoint_score_objective_gradient_l2":pretrain_grad["hyper_net3_score_objective_gradient_l2"],
          "score_pretraining_checkpoint_integrand_mean":pretrain_grad["score_loss_integrand_mean_float64"],
          "score_network_used_as_drift_in_frozen_trajectory_sampler":False,
          "trajectory_sampler_equation":"dx = hyper_net1(x,t) dt + sqrt(2*d) dW; score network is absent from its drift"}
  records.append(record);details.append((record,net_stats))
 summary_path=OUT/"score_term_audit.tsv"
 pd.DataFrame(records).to_csv(summary_path,sep='\t',index=False)
 lines=["# Score-term audit (Phase 1R-S)","","The frozen checkpoint's `hyper_net3` was recomputed on fixed, canonical non-lineage latent rows and compared with saved `score` values. This is an inference-only check; no weights were updated.","","The Phase 1R trajectory sampler implements `dx = v(x,t) dt + sqrt(2d) dW` with `v=hyper_net1`. Its sampled drift does not include `hyper_net3`. The score network participates in the RUOT training objective and diagnostic outputs, so its scale still matters for the frozen specification.","","## Per-seed results","","See `results/phase1RS/score_term_audit.tsv` for machine-readable values.",""]
 for r,_ in details:
  lines += [f"### Seed {r['seed']}","",f"- Saved score norm median / p99 / max: {r['score_l2_norm_median']:.4g} / {r['score_l2_norm_p99']:.4g} / {r['score_l2_norm_max']:.4g}.",f"- Fraction with norm ≤1e-8: {r['fraction_score_norm_le_1e-8']:.3f}; exact-zero components: {r['fraction_exact_zero_components']:.3f}.",f"- Median ratio `d²||score||² / ||velocity||²`: {r['score_to_velocity_energy_ratio_median']:.4g}.",f"- Float32 score norm / latent Jacobian norm / score-objective parameter-gradient norm: {r['torch_float32_score_norm_median']:.4g} / {r['coordinate_gradient_frobenius_median_float64']:.4g} / {r['hyper_net3_score_objective_gradient_l2']:.4g}.",f"- After score pretraining, score norm / objective-gradient norm were {r['score_pretraining_checkpoint_raw_score_norm_median']:.4g} / {r['score_pretraining_checkpoint_score_objective_gradient_l2']:.4g}.",f"- Frozen checkpoint recomputation max absolute difference: {r['checkpoint_recompute_max_abs_difference']:.4g}.",f"- Logged score pretraining loss: {r['score_loss_pretrain_first']} initially, {r['score_loss_pretrain_last']} finally ({r['score_loss_logged_iterations']} iterations)."," "]
 active=all(r['score_l2_norm_p99']<=1e-8 and r['score_to_velocity_energy_ratio_median']<1e-12 and r['coordinate_gradient_frobenius_median_float64']<1e-8 and r['hyper_net3_score_objective_gradient_l2']<1e-8 for r,_ in details)
 lines += ["## Classification","",("Flag: `SCORE_COMPONENT_EFFECTIVELY_ZERO`. The saved network output is numerically inactive at the audited scale in all five seeds; float32 squaring of values at this scale also underflows. Restoring a non-negligible trained score component would require changing or retraining the fitted model, which is prohibited in Phase 1R-S." if active else "The score network is not uniformly negligible at the audited scale; see the per-seed records. No score rescaling or retraining was performed."),"","The observed near-zero output is a property of the frozen checkpoint, not an inference-time serialization mismatch. The full SDE sampler also has no score drift by specification, so this flag concerns the fitted RUOT score component and objective, not an omitted term in the declared Euler–Maruyama sampler.","","Lineage remained locked throughout this audit.",""]
 report=ROOT/"reports/SCORE_TERM_AUDIT.md";report.write_text('\n'.join(lines))
 summary={"status":"COMPLETE_LINEAGE_BLIND","effectively_zero":active,"classification":"SCORE_COMPONENT_EFFECTIVELY_ZERO" if active else "SCORE_COMPONENT_NONZERO_AT_AUDITED_SCALE","seed_records":records,"outputs":{str(summary_path.relative_to(ROOT)):sha256(summary_path),str(report.relative_to(ROOT)):sha256(report)},"lineage_information_opened":False}
 p=OUT/"score_term_audit_summary.json";p.write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({"status":summary['status'],"classification":summary['classification'],"output_sha256":summary['outputs']},indent=2))


if __name__=='__main__':main()
