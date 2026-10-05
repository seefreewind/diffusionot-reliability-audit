#!/usr/bin/env python3
"""Coupled fixed-checkpoint Euler-Maruyama solver and d=0 diagnostics."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist, pdist
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "results/phase1R/REIMPLEMENTED_ID_PRESERVING_ARM"
OUT = ROOT / "results/phase1RS/solver_sensitivity"
SEEDS = [11, 23, 47, 71, 101]
DT_LEVELS = [0.01, 0.005, 0.0025]
N_METRIC = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sinkhorn_transport_cost(x: np.ndarray, y: np.ndarray, reg: float) -> float:
    # Reproduce POT sinkhorn_log + sinkhorn2: return <plan, squared-distance>.
    cost=cdist(x,y,metric="sqeuclidean").astype(np.float64,copy=False)
    log_kernel=-cost/reg
    loga=np.full(len(x),-np.log(len(x)),dtype=np.float64)
    logb=np.full(len(y),-np.log(len(y)),dtype=np.float64)
    logu=np.zeros(len(x),dtype=np.float64);logv=np.zeros(len(y),dtype=np.float64)
    for iteration in range(500):
        # POT updates v then u, and checks the right marginal every ten steps.
        logv=logb-logsumexp(log_kernel+logu[:,None],axis=0)
        logu=loga-logsumexp(log_kernel+logv[None,:],axis=1)
        if iteration % 10 == 0:
            log_col_mass=logsumexp(log_kernel+logu[:,None]+logv[None,:],axis=0)
            if np.linalg.norm(np.exp(log_col_mass)-np.exp(logb)) < 1e-9:
                break
    plan=np.exp(logu[:,None]+log_kernel+logv[None,:])
    return float(np.sum(plan*cost))


def sinkhorn_divergence(x: np.ndarray, y: np.ndarray, reg: float) -> float:
    # Match the frozen Phase 1R definition: cross cost minus half each self cost.
    xy=sinkhorn_transport_cost(x,y,reg)
    xx=sinkhorn_transport_cost(x,x,reg)
    yy=sinkhorn_transport_cost(y,y,reg)
    return xy-0.5*xx-0.5*yy


def mmd2(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    scale = 2.0 * bandwidth * bandwidth
    kxx = np.exp(-cdist(x, x, metric="sqeuclidean") / scale)
    kyy = np.exp(-cdist(y, y, metric="sqeuclidean") / scale)
    kxy = np.exp(-cdist(x, y, metric="sqeuclidean") / scale)
    return float((kxx.sum()-np.trace(kxx))/(len(x)*(len(x)-1))
                 +(kyy.sum()-np.trace(kyy))/(len(y)*(len(y)-1))-2*kxy.mean())


def support_radii(endpoints: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray,float]:
    from scripts.phase1rs_support import support_radii as frozen
    return frozen(endpoints, targets)


def coupled_brownian_increments(n_cells: int, diffusion: float, seed: int, day_a: int,
                                baseline_dt: float = 0.01, fine_dt: float = 0.0025,
                                batch: int = 2048) -> tuple[np.ndarray,np.ndarray]:
    """Reproduce frozen baseline Brownian draws, then condition finer steps on them."""
    n_coarse=round(1.0/baseline_dt)
    ratio=round(baseline_dt/fine_dt)
    if ratio != 4:
        raise ValueError("Frozen protocol expects dt/4 to provide four bridge increments")
    coarse=np.empty((n_cells,n_coarse,2),dtype=np.float32)
    base_rng=np.random.default_rng(seed*100000+day_a)
    sigma=np.sqrt(2*diffusion*baseline_dt)
    # Match Phase 1R's chunk-major, then time-major RNG consumption exactly.
    for lo in range(0,n_cells,batch):
        hi=min(lo+batch,n_cells)
        for step in range(n_coarse):
            coarse[lo:hi,step]=base_rng.normal(0.0,sigma,size=(hi-lo,2)).astype(np.float32)
    bridge_rng=np.random.default_rng(seed*100000+day_a+900_000_000)
    z=bridge_rng.normal(size=(n_cells,n_coarse,ratio,2))
    z-=z.mean(axis=2,keepdims=True)
    fine=coarse.astype(np.float64)[:,:,None,:]/ratio + np.sqrt(2*diffusion*fine_dt)*z
    fine=fine.astype(np.float32)
    # Correct float32 roundoff so the four fine increments still sum to each
    # frozen baseline increment before any coarser aggregation.
    correction=coarse-fine.sum(axis=2,dtype=np.float32)
    fine[:,:,-1,:]+=correction
    return coarse,fine.reshape(n_cells,n_coarse*ratio,2)


def integrate(func, starts: np.ndarray, noise: np.ndarray, t0: float, dt: float,
              diffusion: float, stochastic: bool, batch: int = 2048) -> tuple[np.ndarray,float]:
    n_steps = noise.shape[1]
    out = np.empty_like(starts, dtype=np.float32)
    begin = time.perf_counter()
    for lo in range(0,len(starts),batch):
        hi=min(lo+batch,len(starts))
        z=torch.from_numpy(starts[lo:hi].astype(np.float32,copy=True))
        for step in range(n_steps):
            t=torch.tensor(t0+step*dt,dtype=torch.float32)
            with torch.no_grad():
                v=func.hyper_net1(t,z)
            z=z+v*dt
            if stochastic:
                z=z+torch.from_numpy(noise[lo:hi,step])
        out[lo:hi]=z.numpy()
    return out,time.perf_counter()-begin


def metric_context(starts: np.ndarray, random_ep: np.ndarray, targets: np.ndarray, seed: int) -> dict:
    rng=np.random.default_rng(6047+seed)
    xi=rng.choice(len(starts),N_METRIC,replace=False)
    yi=rng.choice(len(targets),N_METRIC,replace=False)
    y,ident,rand=targets[yi],starts[xi],random_ep[xi]
    median_sq=float(np.median(pdist(y,metric="sqeuclidean")))
    median_dist=float(np.median(pdist(y,metric="euclidean")))
    reg=.05*median_sq
    id_sink=sinkhorn_divergence(ident,y,reg)
    rand_sink=sinkhorn_divergence(rand,y,reg)
    id_mmd=mmd2(ident,y,median_dist)
    rand_mmd=mmd2(rand,y,median_dist)
    return {"xi":xi,"y":y,"reg":reg,"bandwidth":median_dist,
            "identity":ident,"random":rand,"identity_sinkhorn":id_sink,"random_sinkhorn":rand_sink,
            "identity_mmd2":id_mmd,"random_mmd2":rand_mmd}


def metrics(pred: np.ndarray, context: dict) -> dict:
    x=pred[context["xi"]]; y=context["y"]
    model_sink=sinkhorn_divergence(x,y,context["reg"])
    model_mmd=mmd2(x,y,context["bandwidth"])
    id_sink=context["identity_sinkhorn"];rand_sink=context["random_sinkhorn"]
    id_mmd=context["identity_mmd2"];rand_mmd=context["random_mmd2"]
    return {"debiased_sinkhorn":model_sink,"identity_debiased_sinkhorn":id_sink,"random_debiased_sinkhorn":rand_sink,
            "debiased_sinkhorn_better_than_both_nulls":bool(model_sink<id_sink and model_sink<rand_sink),
            "debiased_sinkhorn_improvement_over_identity":1-model_sink/id_sink if id_sink else float("nan"),
            "debiased_sinkhorn_improvement_over_random":1-model_sink/rand_sink if rand_sink else float("nan"),
            "mmd2":model_mmd,"identity_mmd2":id_mmd,"random_mmd2":rand_mmd,
            "mmd_better_than_both_nulls":bool(model_mmd<id_mmd and model_mmd<rand_mmd),
            "mmd_improvement_over_identity":id_mmd-model_mmd,
            "mmd_improvement_over_random":rand_mmd-model_mmd}


def run_seed(seed: int) -> list[dict]:
    sys.path.insert(0,str(ROOT/"external/DiffusionOT"))
    import utility
    seed_dir=ARM/f"seed_{seed}"
    ckpt=torch.load(seed_dir/"ckpt_Mouse.pth",map_location="cpu")
    func=utility.RUOT(in_out_dim=2,hidden_dim=16,n_hiddens=4,activation="Tanh",d=.001).cpu()
    func.load_state_dict(ckpt["func_state_dict"]);func.eval()
    d=float(func.d.detach().item())
    with np.load(seed_dir/"latent_input.npz",allow_pickle=False) as a:
        latent=a["latent_ae"].astype(np.float32)
        times=a["time_label"].astype(str).astype(int)
    rows=[]
    output=np.empty((2,len(DT_LEVELS),2,10000,2),dtype=np.float32)
    original_deltas=[]
    for pair_i,(day_a,day_b,t0) in enumerate([(2,4,0.0),(4,6,1.0)]):
        frozen_path=ROOT/f"results/phase1R/snapshot_qc/seed_{seed}/{day_a}_to_{day_b}_endpoints.npz"
        with np.load(frozen_path,allow_pickle=False) as f:
            src_rows=f["source_row_index"].astype(np.int64)
            target_rows=f["target_row_index"].astype(np.int64)
            original=f["model_endpoints"].astype(np.float32)
            random_ep=f["random_matched_endpoints"].astype(np.float32)
        starts=latent[src_rows]
        targets=latent[target_rows]
        target_full=latent[times==day_b]
        target_nn=NearestNeighbors(n_neighbors=31,algorithm="kd_tree",n_jobs=1).fit(target_full)
        loo_dist,loo_ind=target_nn.kneighbors(target_full,n_neighbors=32)
        loo30=np.empty(len(target_full),dtype=np.float64)
        for i,(dd,ii) in enumerate(zip(loo_dist,loo_ind)):
            loo30[i]=dd[ii!=i][29]
        support_q99=float(np.quantile(loo30,.99))
        metric=metric_context(starts,random_ep,targets,seed)
        # A single fine Brownian path is shared by the three Euler step sizes.
        dt_fine=DT_LEVELS[-1]
        coarse_noise,fine_noise=coupled_brownian_increments(len(starts),d,seed,day_a,
                                                            baseline_dt=DT_LEVELS[0],fine_dt=dt_fine)
        pair_eps={}
        solver_endpoints={}
        for li,dt in enumerate(DT_LEVELS):
            ratio=int(round(dt/dt_fine)); nstep=round(1.0/dt)
            noise=(coarse_noise if dt==DT_LEVELS[0] else
                   fine_noise.reshape(len(starts),nstep,ratio,2).sum(axis=2).astype(np.float32))
            for mi,(mode,stochastic) in enumerate([("full_sde",True),("drift_only_d0",False)]):
                end,runtime=integrate(func,starts,noise,t0,dt,d,stochastic)
                output[pair_i,li,mi]=end
                solver_endpoints[(li,mi)]=end
                dist,_=target_nn.kneighbors(end,n_neighbors=1)
                rad=support_q99
                eval_metrics=metrics(end,metric)
                base=solver_endpoints[(0,mi)] if li==0 else solver_endpoints[(li-1,mi)]
                consecutive=np.linalg.norm(end-base,axis=1)
                original_delta=np.linalg.norm(end-original,axis=1)
                row={"seed":seed,"transition":f"{day_a}_to_{day_b}","mode":mode,"dt":dt,
                     "n_steps":nstep,"diffusion_d_checkpoint":d,
                     "sinkhorn_definition":"POT-compatible debiased squared-Euclidean Sinkhorn; parity exact on frozen seed-11 2-to-4 snapshot",
                     "out_of_support_fraction":float(np.mean(dist>rad)),"support_radius_q99":rad,
                     "sinkhorn_regularization":metric["reg"],"mmd_bandwidth":metric["bandwidth"],
                     "endpoint_nearest_distance_median":float(np.median(dist)),
                     "endpoint_spread_trace_covariance":float(np.trace(np.cov(end.T))),
                     "median_source_endpoint_displacement":float(np.median(np.linalg.norm(end-starts,axis=1))),
                     "runtime_seconds":runtime,
                     "median_displacement_from_previous_step_size":float(np.median(consecutive)) if li else 0.0,
                     "p95_displacement_from_previous_step_size":float(np.quantile(consecutive,.95)) if li else 0.0,
                     "median_displacement_from_frozen_phase1r_endpoint":float(np.median(original_delta)),
                     "p95_displacement_from_frozen_phase1r_endpoint":float(np.quantile(original_delta,.95)),
                     **eval_metrics}
                rows.append(row)
                pair_eps[(li,mi)]=row
        # Explicit paired full-SDE vs d=0 comparison at each resolution.
        for li,dt in enumerate(DT_LEVELS):
            full=pair_eps[(li,0)]; deterministic=pair_eps[(li,1)]
            full["excess_support_escape_fraction_due_to_diffusion"]=full["out_of_support_fraction"]-deterministic["out_of_support_fraction"]
            full["full_sde_minus_drift_endpoint_spread"]=(full["endpoint_spread_trace_covariance"]-deterministic["endpoint_spread_trace_covariance"])
    OUT.mkdir(parents=True,exist_ok=True)
    arr_path=OUT/f"seed_{seed}_coupled_endpoints.npz"
    np.savez_compressed(arr_path,endpoints=output,solver_steps=np.asarray(DT_LEVELS),modes=np.asarray(["full_sde","drift_only_d0"]),seed=np.asarray(seed),learned_d=np.asarray(d))
    for row in rows: row["endpoint_array_path"]=str(arr_path.relative_to(ROOT))
    return rows


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in SEEDS:
        rows.extend(run_seed(seed))
    out=OUT/"solver_sensitivity.tsv"
    pd.DataFrame(rows).to_csv(out,sep="\t",index=False)
    print(json.dumps({"status":"COMPLETE_LINEAGE_BLIND","rows":len(rows),"output":str(out.relative_to(ROOT)),"sha256":sha256(out),"lineage_information_opened":False},indent=2))


if __name__=="__main__": main()
