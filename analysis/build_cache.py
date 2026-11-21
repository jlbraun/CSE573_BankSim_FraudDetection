"""Compute and cache detector scores (default params) for train / 50-50 test / realistic test."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.features import load, realistic_test, build_features

out = sys.argv[1] if len(sys.argv) > 1 else "analysis_cache"
os.makedirs(out, exist_ok=True)
t = time.time()
tr, te, full = load()
rt = realistic_test(full, tr)
for name, df, kw in [("train_insample", tr, {}), ("train_oof", tr, {"oof": True}),
                     ("test5050", te, {}), ("realistic", rt, {})]:
    X = build_features(df, tr, full, **kw)
    X["fraud"] = df["fraud"].to_numpy()
    X.to_pickle(os.path.join(out, name + ".pkl"))
    print(name, X.shape, round(time.time() - t, 1), "s", flush=True)
