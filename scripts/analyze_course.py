"""Analyze all frozen course runs using paired image-cluster resampling."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PRIMARY=("baseline","augmented","premask","full")
METRICS=("anchor_f1","fake_follow_lift","clip_cosine","fid_clean","kid_clean")

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def paired(data,comparator,seeds=(42,43),exclude=()):
    grouped={}
    for seed in seeds:
        a={(r["id"],r["view"]):r for r in data[("full",seed)]["diagnostics"]}
        b={(r["id"],r["view"]):r for r in data[(comparator,seed)]["diagnostics"]}
        assert a.keys()==b.keys()
        for key in a:
            if key[0] in exclude:continue
            for metric in ("anchor_f1","fake_follow_lift","clip_cosine"):
                if metric=="fake_follow_lift" and not a[key]["has_fake_strokes"]:continue
                grouped.setdefault(metric,{}).setdefault(key[0],[]).append(a[key][metric]-b[key][metric])
    result={}
    rng=np.random.default_rng(20261108)
    for metric,groups in grouped.items():
        ids=sorted(groups)
        sums=np.array([sum(groups[key]) for key in ids])
        counts=np.array([len(groups[key]) for key in ids])
        indices=rng.integers(0,len(ids),size=(10000,len(ids)))
        estimates=sums[indices].sum(1)/counts[indices].sum(1)
        result[metric]={"mean_delta":float(sums.sum()/counts.sum()),"image_cluster_95ci":np.quantile(estimates,[.025,.975]).tolist(),
                        "n_images":len(ids),"n_view_seed_pairs":int(counts.sum())}
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--runs",default="runs/course")
    parser.add_argument("--output",default="outputs/course_analysis.json")
    args=parser.parse_args()
    runs=ROOT/args.runs
    data={}
    expected={(v,s) for v in PRIMARY for s in (42,43)}|{("no_pair",42),("fixed_gate",42)}
    image_ids=None
    for variant,seed in sorted(expected):
        folder=runs/f"{variant}_s{seed}"
        config=read(folder/"config.json");status=read(folder/"status.json")
        metrics=read(folder/"generated/metrics.json")
        diagnostics=read(folder/"generated/diagnostics.json")
        assert status["status"]=="complete" and status["step"]==1500
        assert config["global_batch"]==8 and config["micro_batch"]==4 and config["resolution"]==512
        assert metrics["n_images"]==400
        ids={row["id"] for row in diagnostics}
        if image_ids is None:image_ids=ids
        assert ids==image_ids
        raw=[json.loads(line) for line in (folder/"train.jsonl").read_text().splitlines()]
        by_step={row["step"]:row for row in raw}
        assert sorted(by_step)==list(range(1,1501))
        data[(variant,seed)]={"metrics":metrics,"diagnostics":diagnostics,"status":status,
                              "warm_step_median_seconds":float(np.median([row["seconds"] for step,row in by_step.items() if step>10])),
                              "raw_log_duplicate_steps":len(raw)-len(by_step)}
    primary={}
    for variant in PRIMARY:
        primary[variant]={"seeds":[42,43]}
        for metric in METRICS:
            values=[data[(variant,seed)]["metrics"][metric] for seed in (42,43)]
            primary[variant][metric]={"mean":float(np.mean(values)),"seed_std":float(np.std(values,ddof=1)),"values":values}
    flagged=sorted({r["id"] for item in data.values() for r in item["diagnostics"] if r["safety_flag"]})
    strata={}
    for field in ("low_trust","corruption"):
        groups={}
        for value in sorted({r[field] for r in data[("full",42)]["diagnostics"]}):
            groups[str(value)]={}
            for variant in PRIMARY:
                subset=[r for seed in (42,43) for r in data[(variant,seed)]["diagnostics"] if r[field]==value]
                fake=[r for r in subset if r["has_fake_strokes"]]
                groups[str(value)][variant]={"n_images":len({r["id"] for r in subset}),
                    "anchor_f1":float(np.mean([r["anchor_f1"] for r in subset])),
                    "clip_cosine":float(np.mean([r["clip_cosine"] for r in subset])),
                    "fake_follow_lift":float(np.mean([r["fake_follow_lift"] for r in fake])) if fake else None,
                    "n_fake_image_ids":len({r["id"] for r in fake})}
        strata[field]=groups
    result={"protocol":{"n_images":400,"train_steps":1500,"global_batch":8,"micro_batch":4,
                         "generation_steps":25,"training_seeds":[42,43],
                         "bootstrap":"10,000 image-cluster resamples; both intervention views and fixed trained seeds kept together; does not estimate seed-population uncertainty."},
            "main":primary,"strata_descriptive":strata,
            "runs":{f"{v}_s{s}":{k:value for k,value in item.items() if k!="diagnostics"} for (v,s),item in data.items()},
            "full_minus_comparator":{v:paired(data,v) for v in PRIMARY[:-1]},
            "ablation_seed42":{v:{"metrics":data[(v,42)]["metrics"],
                                  "full_minus_variant":paired(data,v,seeds=(42,)) if v!="full" else None}
                                  for v in ("full","no_pair","fixed_gate")},
            "joint_safety_sensitivity":{"secondary_only":True,"excluded_image_ids":flagged,
                                        "full_minus_comparator":{v:paired(data,v,exclude=set(flagged)) for v in PRIMARY[:-1]}}}
    output=ROOT/args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"main":primary,"full_minus_premask":result["full_minus_comparator"]["premask"],
                      "joint_flagged_images":len(flagged)},indent=2))

if __name__=="__main__":main()
