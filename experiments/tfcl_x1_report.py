"""EXPLORATORY (ext6 addendum X1): FreshUtil false spawns on null streams, fresh vs. inherited head."""
import glob
import json


def main():
    res = {}
    for s in ["clinc_cil", "banking_cil"]:
        for d in ["null_closed", "null_closed_inherit"]:
            runs = {}
            for f in sorted(glob.glob(f"results/tfcl/ext6/{d}/{s}/trigger-fresh_util-q0.99_s*.json")):
                j = json.load(open(f))
                runs[f.split("_s")[-1][:4]] = j["packages"] - 1
            res[f"{s}/{'inherit' if 'inherit' in d else 'fresh_head'}"] = {
                "spawns_per_seed": runs, "mean_spawns": sum(runs.values()) / max(len(runs), 1),
                "runs_with_spawn": sum(v > 0 for v in runs.values()), "n_runs": len(runs)}
    json.dump(res, open("results/tfcl/x1_inherit.json", "w"), indent=1)
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
