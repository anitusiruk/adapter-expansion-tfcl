"""Every trigger probe must leave the real learner bitwise unchanged."""
import copy, random, sys
import torch
sys.path.insert(0, ".")
from src.tfcl.data import make_stream
from src.tfcl.learner import Learner
from src.tfcl.model import Encoder
from src.tfcl.triggers import make_trigger, ALL


def state(L):
    return {"params": [[t.detach().clone() for t in p.params()] for p in L.packages],
            "opt": [copy.deepcopy(p.opt.state_dict()) for p in L.packages],
            "upd": [p.updates for p in L.packages],
            "rng": (L.replay_rng.getstate(), L.probe_rng.getstate(), L.mem.rng.getstate(), L.gen.get_state(),
                    torch.get_rng_state(), torch.cuda.get_rng_state_all(), random.getstate()),
            "mem": list(L.mem.items), "fs": list(L.first_seen), "seen": L.seen.clone(), "t": L.t, "n": len(L.packages)}


def same(a, b):
    for x, y in zip(a["params"], b["params"]):
        assert all(torch.equal(u, v) for u, v in zip(x, y)), "params changed"
    for x, y in zip(a["opt"], b["opt"]):
        for k in x["state"]:
            for kk, v in x["state"][k].items():
                assert torch.equal(v, y["state"][k][kk]), "optimizer changed"
    assert a["upd"] == b["upd"] and a["mem"] == b["mem"] and a["t"] == b["t"] and a["n"] == b["n"]
    assert torch.equal(a["seen"], b["seen"]) and a["fs"] == b["fs"]
    ra, rb = a["rng"], b["rng"]
    assert ra[0] == rb[0] and ra[2] == rb[2] and ra[6] == rb[6], "python rng changed"
    assert torch.equal(ra[3], rb[3]) and torch.equal(ra[4], rb[4]), "torch rng changed"
    assert all(torch.equal(u, v) for u, v in zip(ra[5], rb[5])), "cuda rng changed"


def main():
    stream, dev, n = make_stream("banking_rec", 2026)
    enc = Encoder("distilbert-base-uncased")
    L = Learner(enc, n, 2026)
    trig = make_trigger("shadow", L)
    for i, b in enumerate(stream[:24]):
        if i == 12:
            L.spawn()
        before = state(L)
        rng_probe_before = L.probe_rng.getstate()
        rec = trig.decide(b["x"], b["y"])
        after = state(L)
        # probe rng is allowed to advance (it is private to probes); everything else must not
        before["rng"] = (before["rng"][0],) + (None,) + before["rng"][2:]
        after["rng"] = (after["rng"][0],) + (None,) + after["rng"][2:]
        same(before, after)
        L.observe(b["x"], b["y"])
        trig.after_update(b["x"], b["y"])
    print("PASS probe restoration over 24 batches incl. spawn; last record:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()})


if __name__ == "__main__":
    main()
