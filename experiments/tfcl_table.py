"""Print a stream x policy table of final accuracy (mean over seeds) and pool size."""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
d = sys.argv[1]; metric = sys.argv[2] if len(sys.argv) > 2 else "proto_mean"
rows = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{d}/*/*.json")):
    r = json.load(open(f)); s = f.split("/")[-2]; p = os.path.basename(f).rsplit("_s", 1)[0]
    rows[s][p].append((r["final_mean"][metric], r["packages"]))
q = sys.argv[3] if len(sys.argv) > 3 else "0.99"
pols = ["frozen", "single", "oracle"] + ["trigger-" + t + "-q" + q for t in ["label_novel", "loss_z", "repr_z", "interf_logit", "fresh_util", "interf_proto", "conflict_z", "label_surprise"]]
print("%-16s" % "" + "".join("%15s" % p.replace("trigger-", "").split("-q")[0][:13] for p in pols))
for s, v in rows.items():
    print("%-16s" % s + "".join("%9.3f(%4.1f)" % (np.mean([a for a, _ in v[p]]), np.mean([b for _, b in v[p]])) if p in v else "%15s" % "-" for p in pols))
