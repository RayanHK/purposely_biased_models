"""
Feature selection for (a) maximum accuracy, (b) maximum age/gender bias, (c) both at once.

Run:   python feature_search.py            (about 20-25 minutes on a laptop CPU)
Needs: investigation_train_large_checked.csv in the same folder; pandas, scikit-learn, torch.

Outputs (all in the current folder):
  feature_profile.csv      step 1: per-feature predictive / gender-proxy / age-proxy strength
  search_log_<obj>.csv     step 2: every step of every search, with its scores
  feature_sets.json        the chosen feature sets
  test_results.csv         step 3: final scores on the untouched 30% test set

Method in short
---------------
* Same 70/30 split (random_state=42) as the training script. The 30% test set is only used in step 3.
* The 70% training part is split again 75/25 into fit/validation; all choices are made on validation.
* Accuracy is measured as ROC-AUC (threshold-free; plain accuracy is ~85% just by predicting "no").
* Bias is measured at a fixed investigation budget: the 15% highest-scoring people are flagged
  (15% = the real share of 'checked'). Then per group the flag rate is computed and
      age_bias    = 1 - (lowest age-group flag rate / highest age-group flag rate)
      gender_bias = 1 - (lowest gender flag rate / highest gender flag rate)
      bias        = age_bias + gender_bias          (0 = perfectly equal, 2 = maximal)
  "1 - min/max" is the disparate-impact ratio (four-fifths rule) turned around so higher = more biased.
* Search: greedy forward selection. Start empty, try adding each candidate feature, keep the one that
  improves the objective most, repeat until 15 features or the gain is < 0.002.
  For the bias objective greedy gets stuck early, so a beam search (keep the best 5 sets per size) is used.
* Objectives:  accuracy = AUC;   bias = bias;   both = (2*AUC - 1) * bias   (Gini x bias: zero if either is zero)
"""
import json, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

CSV = "investigation_train_large_checked.csv"
G, A = "persoon_geslacht_vrouw", "persoon_leeftijd_bij_onderzoek"
AGE_BINS, AGE_LABELS = [0, 30, 45, 60, 200], ["≤30", "31-45", "46-60", "61+"]
SEED, BUDGET = 42, 0.15
MAX_FEATURES, MIN_GAIN, BEAM = 15, 0.002, 5


#----------------------------------------------------------------------------- data
df = pd.read_csv(CSV)
X = df.iloc[:, :-3]                        #Drops Ja, Nee, checked (same as the training script)
X = X.loc[:, X.nunique() > 1].copy()       #One constant column removed
y = df["checked"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
Xfit, Xval, yfit, yval = train_test_split(Xtr, ytr, test_size=0.25, random_state=SEED)


#---------------------------------------------------------------------------- metrics
def flag_top(scores, rate=BUDGET):
    """Flag the `rate` share of people with the highest scores; ties broken at random."""
    scores = np.asarray(scores)
    k = int(round(rate * len(scores)))
    order = np.lexsort((np.random.default_rng(SEED).random(len(scores)), -scores))
    pred = np.zeros(len(scores), dtype=int)
    pred[order[:k]] = 1
    return pred


def evaluate(scores, y_true, gender, age):
    y_true = np.asarray(y_true)
    pred = flag_top(scores)
    age_grp = pd.cut(age, AGE_BINS, labels=AGE_LABELS).astype(str).to_numpy()
    age_rates = pd.Series(pred).groupby(age_grp).mean()
    g_rates = pd.Series(pred).groupby(np.asarray(gender)).mean()
    age_bias = 1 - age_rates.min() / age_rates.max()
    gender_bias = 1 - g_rates.min() / g_rates.max()
    return dict(auc=roc_auc_score(y_true, scores), accuracy=(pred == y_true).mean(),
                age_bias=age_bias, gender_bias=gender_bias, bias=age_bias + gender_bias,
                age_rates=age_rates.round(3).to_dict(),
                gender_rates={("vrouw" if k == 1 else "man"): v for k, v in g_rates.round(3).items()})


def fast_model():
    return HistGradientBoostingClassifier(max_iter=60, learning_rate=0.15, random_state=SEED)


_cache = {}
def score_set(cols):
    """Train on fit split, score on validation split. Cached so repeated sets cost nothing."""
    key = frozenset(cols)
    if key not in _cache:
        cols = list(cols)
        s = fast_model().fit(Xfit[cols], yfit).predict_proba(Xval[cols])[:, 1]
        _cache[key] = evaluate(s, yval, Xval[G], Xval[A])
    return _cache[key]


#--------------------------------------------------------------------------- step 1: profile
def strength(target, f):
    """|AUC - 0.5| * 2: 0 = feature tells nothing about target, 1 = perfectly separates it."""
    try:
        return abs(roc_auc_score(target, f) - 0.5) * 2
    except ValueError:
        return 0.0

print("Step 1: profiling features", flush=True)
age_rank = Xfit[A].rank()
profile = pd.DataFrame([dict(
    feature=c,
    predictive=strength(yfit, Xfit[c]),                               # how well it predicts 'checked'
    gender_proxy=strength(Xfit[G], Xfit[c]),                          # how well it predicts gender
    age_proxy=abs(np.corrcoef(Xfit[c].rank(), age_rank)[0, 1]),       # Spearman correlation with age
) for c in Xfit.columns]).set_index("feature")
profile.to_csv("feature_profile.csv")

# Candidate pool for the searches: strongest features on each axis
pool = sorted(set(profile.nlargest(40, "predictive").index)
              | set(profile.nlargest(20, "gender_proxy").index)
              | set(profile.nlargest(20, "age_proxy").index) | {A, G})
print(f"  candidate pool: {len(pool)} features", flush=True)


#------------------------------------------------------------------------- step 2: search
OBJECTIVES = {
    "accuracy": lambda m: m["auc"],
    "bias": lambda m: m["bias"],
    "both": lambda m: (2 * m["auc"] - 1) * m["bias"],
}

def log_row(name, step, feats, m, score):
    return dict(objective=name, step=step, features=" + ".join(feats), score=round(score, 4),
                auc=round(m["auc"], 4), age_bias=round(m["age_bias"], 3),
                gender_bias=round(m["gender_bias"], 3), age_rates=m["age_rates"], gender_rates=m["gender_rates"])

def greedy(name):
    obj, chosen, best, log = OBJECTIVES[name], [], -np.inf, []
    while len(chosen) < MAX_FEATURES:
        trials = {c: score_set(chosen + [c]) for c in pool if c not in chosen}
        cand, m = max(trials.items(), key=lambda kv: obj(kv[1]))
        if obj(m) - best < MIN_GAIN:
            break
        chosen.append(cand); best = obj(m)
        log.append(log_row(name, len(chosen), [cand], m, best))
        print(f"  [{name}] +{cand}  score={best:.4f}  auc={m['auc']:.3f}  "
              f"age_bias={m['age_bias']:.3f}  gender_bias={m['gender_bias']:.3f}", flush=True)
    pd.DataFrame(log).to_csv(f"search_log_{name}.csv", index=False)
    return chosen

def beam(name):
    obj, beam_sets, best, best_set, log = OBJECTIVES[name], [()], -np.inf, None, []
    for size in range(1, MAX_FEATURES + 1):
        cands = {tuple(sorted(b + (c,))) for b in beam_sets for c in pool if c not in b}
        scored = sorted(((s, score_set(s)) for s in cands), key=lambda t: -obj(t[1]))
        beam_sets = [s for s, _ in scored[:BEAM]]
        s, m = scored[0]
        log.append(log_row(name + "_beam", size, s, m, obj(m)))
        print(f"  [{name} beam] size {size}: score={obj(m):.4f}  {s}", flush=True)
        if obj(m) - best < MIN_GAIN:
            break
        best, best_set = obj(m), list(s)
    pd.DataFrame(log).to_csv(f"search_log_{name}_beam.csv", index=False)
    return best_set

print("Step 2: searching", flush=True)
sets = {"all features": list(X.columns),
        "max accuracy": greedy("accuracy"),
        "max bias": beam("bias"),
        "accurate + biased": greedy("both"),
        "age + gender only": [A, G]}
json.dump(sets, open("feature_sets.json", "w"), indent=2, ensure_ascii=False)


#--------------------------------------------------------------------------- step 3: verify
class testModel(torch.nn.Module):          # same architecture as the project's training script
    def __init__(self, n):
        super().__init__()
        self.l1, self.a1 = torch.nn.Linear(n, 630), torch.nn.ReLU()
        self.l2, self.a2 = torch.nn.Linear(630, 250), torch.nn.ReLU()
        self.l3 = torch.nn.Linear(250, 1)
    def forward(self, x):
        return self.l3(self.a2(self.l2(self.a1(self.l1(x)))))

def nn_scores(cols, epochs=8):
    """Project NN, trained on the full 70%. Batch 64 / lr 0.01 instead of 4 / 0.001 to run in minutes."""
    torch.manual_seed(SEED)
    sc = StandardScaler().fit(Xtr[cols])
    xt = torch.tensor(sc.transform(Xtr[cols]), dtype=torch.float32)
    yt = torch.tensor(ytr.to_numpy(), dtype=torch.float32).unsqueeze(1)
    model = testModel(len(cols))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(yt == 0).sum() / (yt == 1).sum())
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt, yt), batch_size=64, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(sc.transform(Xte[cols]), dtype=torch.float32))).numpy().ravel()

print("Step 3: final check on the untouched test set", flush=True)
rows = []
for name, cols in sets.items():
    gb = fast_model().fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1]
    for model_name, s in [("gradient boosting", gb), ("neural network", nn_scores(cols))]:
        m = evaluate(s, yte, Xte[G], Xte[A])
        rows.append(dict(set=name, model=model_name, n_features=len(cols),
                         auc=round(m["auc"], 3), accuracy=round(m["accuracy"], 3),
                         age_bias=round(m["age_bias"], 3), gender_bias=round(m["gender_bias"], 3),
                         bias=round(m["bias"], 3),
                         **{f"flag {k}": v for k, v in m["age_rates"].items()},
                         **{f"flag {k}": v for k, v in m["gender_rates"].items()}))
res = pd.DataFrame(rows)
res.to_csv("test_results.csv", index=False)
pd.set_option("display.width", 250)
print(res.to_string(index=False))