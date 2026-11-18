"""Compare default and Bayesian-selected RF using identical permitted splits."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from pathsnap.alignment import align_route_to_track
from pathsnap.candidates import CandidateGenerator
from pathsnap.data import load_trip
from train_compare import SEED, _feature_row, _load_sampled_candidates, _top1_accuracy

DEFAULT = dict(n_estimators=120, max_depth=14, min_samples_leaf=2)
TUNED = dict(n_estimators=82, max_depth=21, min_samples_leaf=1, max_features=0.573)

def make(params, seed):
    return RandomForestClassifier(**params, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)

def cv_compare(train_root: Path):
    (x,y,groups,keys,sample_ids,group_names,expected,misses,*_) = _load_sampled_candidates(train_root,50.0,8)
    splitter=GroupKFold(n_splits=5)
    out={}
    for name,params in (("default",DEFAULT),("bayesian_selected",TUNED)):
        records=[]
        for fold,(gtr,gva) in enumerate(splitter.split(np.arange(len(group_names)),groups=np.arange(len(group_names))),1):
            tr=np.flatnonzero(np.isin(groups,gtr)&(sample_ids==0)); va=np.flatnonzero(np.isin(groups,gva)&(sample_ids==1))
            model=make(params,SEED+fold); t=time.perf_counter(); model.fit(x[tr],y[tr]); fit=time.perf_counter()-t
            t=time.perf_counter(); prob=model.predict_proba(x[va])[:,1]; infer=time.perf_counter()-t
            pred=prob>=0.5; valid_groups=np.unique(groups[va]); exp=sum(expected.get(int(g),0) for g in valid_groups)
            covered=len(np.unique(keys[va])); correct=_top1_accuracy(y[va],prob,keys[va])
            records.append({"fold":fold,"candidate_macro_f1":float(f1_score(y[va],pred,average="macro",zero_division=0)),
                "top1_all_aligned":correct/max(1,exp),"log_loss":float(log_loss(y[va],prob,labels=[0,1])),
                "mean_fit_seconds":fit,"inference_seconds":infer,"aligned_observations":int(exp),"covered_observations":int(covered)})
        out[name]={"parameters":params,"folds":records,"candidate_macro_f1_mean":float(np.mean([r["candidate_macro_f1"] for r in records])),
            "candidate_macro_f1_sd":float(np.std([r["candidate_macro_f1"] for r in records],ddof=1)),
            "top1_all_aligned_mean":float(np.mean([r["top1_all_aligned"] for r in records])),
            "top1_all_aligned_sd":float(np.std([r["top1_all_aligned"] for r in records],ddof=1)),
            "log_loss_mean":float(np.mean([r["log_loss"] for r in records])),
            "mean_inference_seconds_per_fold":float(np.mean([r["inference_seconds"] for r in records]))}
    return out

def student_compare(train_root: Path, dev_root: Path):
    loaded=_load_sampled_candidates(train_root,50.0,8)
    x,y,groups,obs_keys,sample_ids=loaded[:5]
    train_rows=np.flatnonzero(sample_ids==0)
    models={name:make(params,SEED) for name,params in (("default",DEFAULT),("bayesian_selected",TUNED))}
    for model in models.values(): model.fit(x[train_rows],y[train_rows])
    gen=CandidateGenerator(radius_m=50.0,max_candidates=8); trip_records={k:[] for k in models}
    for directory in sorted(p for p in dev_root.iterdir() if p.is_dir()):
        pred_trip=load_trip(directory,include_route_labels=False)
        label_trip=load_trip(directory,include_route_labels=True)
        aligned=align_route_to_track(label_trip,max_alignment_distance_m=250.0).arc_ids_by_observation
        generated=gen.generate(pred_trip)
        outgoing={}; incoming={}
        for arc in pred_trip.arcs:
            outgoing[arc.from_node]=outgoing.get(arc.from_node,0)+1
            incoming[arc.to_node]=incoming.get(arc.to_node,0)+1
        feats=[]; labels=[]; owners=[]; arcids=[]
        misses=sum(target is not None and target not in {c.arc_id for c in row.candidates} for row,target in zip(generated,aligned))
        for row,target in zip(generated,aligned):
            for cand in row.candidates:
                feats.append(_feature_row(pred_trip,row.observation_index,cand,outgoing,incoming))
                labels.append(-1 if target is None or target not in {c.arc_id for c in row.candidates} else int(cand.arc_id==target))
                owners.append(row.observation_index); arcids.append(cand.arc_id)
        X=np.asarray(feats,dtype=np.float32); labels=np.asarray(labels,dtype=np.int8); owners=np.asarray(owners)
        valid=labels>=0
        for name,model in models.items():
            t=time.perf_counter(); probs=model.predict_proba(X)[:,1]; elapsed=time.perf_counter()-t
            obs_best={}
            for idx,p,aid in zip(owners,probs,arcids):
                if idx not in obs_best or p>obs_best[idx][0]: obs_best[idx]=(p,aid)
            correct=sum(obs_best.get(i,(None,None))[1]==target for i,target in enumerate(aligned) if target is not None)
            yvalid=labels[valid]; pvalid=probs[valid]; pred=pvalid>=.5
            trip_records[name].append({"trip_id":directory.name,"aligned_observations":sum(t is not None for t in aligned),
                "candidate_misses":int(misses),"top1_all_aligned":correct/max(1,sum(t is not None for t in aligned)),
                "candidate_macro_f1":float(f1_score(yvalid,pred,average="macro",zero_division=0)),
                "log_loss":float(log_loss(yvalid,pvalid,labels=[0,1])),"inference_seconds":elapsed})
        print(f"evaluated development trip {directory.name}",flush=True)
    out={}
    for name,records in trip_records.items():
        out[name]={"parameters":DEFAULT if name=="default" else TUNED,"trip_count":len(records),
            "top1_all_aligned_weighted":sum(r["top1_all_aligned"]*r["aligned_observations"] for r in records)/sum(r["aligned_observations"] for r in records),
            "candidate_macro_f1_mean_per_trip":float(np.mean([r["candidate_macro_f1"] for r in records])),
            "log_loss_mean_per_trip":float(np.mean([r["log_loss"] for r in records])),
            "inference_seconds_total":sum(r["inference_seconds"] for r in records),"trips":records}
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--train",type=Path,required=True); p.add_argument("--student-test",type=Path,required=True); p.add_argument("--output",type=Path,default=Path("experiments/results/tuned_rf_comparison.json")); a=p.parse_args()
    report={"target_semantics":"weak route-sequence alignment; not independent per-observation ground truth",
        "candidate_settings":{"radius_m":50,"max_candidates":8},"training_observations_per_trip_max":100,
        "cv":cv_compare(a.train),"student_development":student_compare(a.train,a.student_test)}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:{n:{m:v for m,v in rows.items() if m not in ("folds","trips")} for n,rows in vals.items()}
                      for k,vals in report.items() if k in ("cv","student_development")},indent=2))
if __name__=="__main__": main()
