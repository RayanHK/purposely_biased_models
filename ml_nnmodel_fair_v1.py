"""
Fair model v1: sensitive-feature filter -> 5-layer MLP, trained with one switchable fairness technique.

Needs fairlearn:  uv pip install fairlearn

FAIRNESS_METHOD:
  "none"         plain MLP (only the filter layer)
  "reweighing"   Kamiran & Calders sample weights so label and sensitive groups are independent
  "adversarial"  fairlearn AdversarialFairnessClassifier (Zhang et al. 2018)
  "constraints"  fairlearn ExponentiatedGradient (Agarwal et al. 2018) under CONSTRAINT
"""
import itertools

import numpy as np
import pandas as pd
import torch
from fairlearn.adversarial import AdversarialFairnessClassifier
from fairlearn.metrics import (MetricFrame, count, demographic_parity_ratio, equalized_odds_ratio,
                               false_positive_rate, selection_rate, true_positive_rate)
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import make_column_selector
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch import nn

FAIRNESS_METHOD = "reweighing"          # "none" | "reweighing" | "adversarial" | "constraints"
CONSTRAINT = "demographic_parity"       # or "equalized_odds" (adversarial + constraints)
# Groups the mitigation protects (their intersection). The fairlearn adversary diverges on all 4 (~240 groups)
FAIR_GROUPS = ["age", "gender"]
ADV_ALPHA = 0.02                        # adversary strength in fine-tuning; >=0.05 often collapses to "check nobody" -> NaN
THRESHOLD = 0.7
EPOCHS = 10
SEED = 42
PROXY_FILTER = "A"                      # "none" | "A" (41 proxies) | "B" (A + 5 strong-but-predictive proxies)
MODEL_PATH = f"fair_v1_{FAIRNESS_METHOD}_{PROXY_FILTER}.pt"

torch.manual_seed(SEED)
device = "cuda" if torch.accelerator.is_available() else "cpu"

df = pd.read_csv("investigation_train_large_checked.csv")
X_all, y = df.iloc[:, :-3], df["checked"].to_numpy()

# Sensitive groups: used for mitigation and evaluation only, never as model input
wijk = df.filter(regex=r"^adres_recentste_wijk_")
groups = pd.DataFrame({
    "age": pd.cut(df["persoon_leeftijd_bij_onderzoek"], [0, 30, 45, 60, 200], labels=["≤30", "31-45", "46-60", "61+"]).astype(str),
    "gender": df["persoon_geslacht_vrouw"].map({0: "man", 1: "vrouw"}),
    "neighborhood": wijk.idxmax(axis=1).str.removeprefix("adres_recentste_wijk_").where(wijk.sum(axis=1) > 0, "geen"),
    # spoken language as a stand-in for immigration background; assumes code 57 = Dutch, 0 = not recorded
    "language": df["persoonlijke_eigenschappen_spreektaal"].map(lambda v: "nl" if v == 57 else "onbekend" if v == 0 else "anders"),
})

# Filter layer: drop age, gender, neighborhood and spoken language before the MLP
SENSITIVE = r"persoon_leeftijd_bij_onderzoek|persoon_geslacht_vrouw|^adres_recentste?_(?:wijk|buurt|plaats|onderdeel)|persoonlijke_eigenschappen_spreektaal"
# Proxies for those attributes (strongest per attribute, from a proxy-recovery analysis)
PROXIES_A = "|".join([
    r"^adres_dagen_op_adres$|^contacten_soort_(?:afgelopenjaar_)?document__uitgaand_$|^relatie_kind_(?:basisschool_kind|volwassen)$",  # age
    r"^relatie_kind_(?:leeftijd_verschil_ouder_eerste_kind|huidige_aantal|heeft_kinderen|jongvolwassen|tiener)$"
    r"|^relatie_partner_totaal_dagen_partner$|^relatie_overig_historie_vorm__onderhoudsplichtige$",                                  # gender
    r"^adres_aantal_(?:verzendadres|brp_adres|verschillende_wijken)$|^adres_unieke_wijk_ratio$",                                     # neighborhood
    r"taaleis|^persoonlijke_eigenschappen_nl_|^persoonlijke_eigenschappen_(?:opstelling|communicatie_opm)$|^belemmering_hist_taal$|inburger",  # language
])
# B also drops proxies that are strong predictors of "checked" (costs more accuracy)
PROXIES_B = PROXIES_A + r"|^pla_historie_ontwikkeling$|^instrument_reden_beeindiging_historie_succesvol$|^afspraak_signaal_voor_medewerker$|^typering_dagen_som$"
FILTER = {"none": SENSITIVE, "A": f"{SENSITIVE}|{PROXIES_A}", "B": f"{SENSITIVE}|{PROXIES_B}"}[PROXY_FILTER]
dropped = make_column_selector(pattern=FILTER)(X_all)
X = X_all.drop(columns=dropped)
print(f"Filter layer drops {len(dropped)} sensitive columns: {dropped}\n{X.shape[1]} features go into the MLP")

idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.3, random_state=SEED)
scaler = StandardScaler().fit(X.iloc[idx_train])
# clip extreme z-scores (e.g. typering_hist_ind reaches 100), which otherwise make the adversarial training NaN
X_train, X_test = (np.clip(scaler.transform(X.iloc[i]), -5, 5) for i in (idx_train, idx_test))
y_train, y_test = y[idx_train], y[idx_test]
A_train, A_test = groups.iloc[idx_train].reset_index(drop=True), groups.iloc[idx_test].reset_index(drop=True)


def make_mlp(n_in):
    return nn.Sequential(nn.Linear(n_in, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                         nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


class TorchMLP(ClassifierMixin, BaseEstimator):
    """sklearn wrapper so fairlearn's reductions can pass sample_weight to the MLP."""

    def __init__(self, epochs=EPOCHS, batch_size=256, lr=1e-3):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0, 1])
        self.model_ = make_mlp(X.shape[1]).to(device)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        data = [torch.tensor(np.asarray(a), dtype=torch.float32) for a in (X, y, w / w.mean())]
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data), batch_size=self.batch_size, shuffle=True)
        # pos_weight so the model doesn't just predict "not checked" for everyone (as in the other scripts)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor((y == 0).sum() / (y == 1).sum(), device=device))
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                opt.zero_grad()
                (loss_fn(self.model_(xb).squeeze(1), yb) * wb).mean().backward()
                opt.step()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            p = torch.sigmoid(self.model_(torch.tensor(np.asarray(X), dtype=torch.float32, device=device))).cpu().numpy()
        return np.hstack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= THRESHOLD).astype(int)


def intersection(A):
    return A.astype(str).agg("|".join, axis=1)


def reweighing_weights(A, y):
    """Kamiran & Calders: w(a, y) = P(a) P(y) / P(a, y), with a = intersection of the given groups."""
    t = pd.DataFrame({"a": intersection(A), "y": y})
    p_a, p_y, p_ay = t["a"].map(t["a"].value_counts(normalize=True)), t["y"].map(t["y"].value_counts(normalize=True)), t.groupby(["a", "y"])["y"].transform("size") / len(t)
    return (p_a * p_y / p_ay).to_numpy()


if FAIRNESS_METHOD in ("none", "reweighing"):
    clf = TorchMLP().fit(X_train, y_train, reweighing_weights(A_train[FAIR_GROUPS], y_train) if FAIRNESS_METHOD == "reweighing" else None)
    probs = clf.predict_proba(X_test)[:, 1]
    to_save = {"state_dict": clf.model_.state_dict()}
elif FAIRNESS_METHOD == "adversarial":
    # fairlearn's adversarial training has no class weights, so oversample positives instead of pos_weight
    pos, neg = np.where(y_train == 1)[0], np.where(y_train == 0)[0]
    bal = np.concatenate([neg, resample(pos, n_samples=len(neg), random_state=SEED)])
    clf = AdversarialFairnessClassifier(backend="torch", predictor_model=[256, "leaky_relu", 128, "leaky_relu", 64, "leaky_relu", 32, "leaky_relu"],
                                        adversary_model=[32, "leaky_relu"], constraints=CONSTRAINT, epochs=EPOCHS, batch_size=256,
                                        shuffle=True, random_state=SEED, learning_rate=3e-4, alpha=0.0, warm_start=True)
    A_bal = intersection(A_train.iloc[bal][FAIR_GROUPS]).to_numpy()
    # Pretrain the predictor without the adversary, then fine-tune against it (training from scratch often collapses)
    clf.fit(X_train[bal], y_train[bal], sensitive_features=A_bal)
    clf.set_params(alpha=ADV_ALPHA).fit(X_train[bal], y_train[bal], sensitive_features=A_bal)
    probs = clf._raw_predict(X_test).ravel()
    to_save = {"state_dict": clf.backendEngine_.predictor_model.state_dict()}
elif FAIRNESS_METHOD == "constraints":
    moment = DemographicParity() if CONSTRAINT == "demographic_parity" else EqualizedOdds()
    clf = ExponentiatedGradient(TorchMLP(epochs=max(EPOCHS // 3, 1), batch_size=1024), moment, max_iter=10)
    clf.fit(X_train, y_train, sensitive_features=A_train[FAIR_GROUPS])
    probs = clf._pmf_predict(X_test)[:, 1]   # probability that the randomized fair classifier flags someone
    pred = clf.predict(X_test, random_state=SEED)  # base models already cut at THRESHOLD
    to_save = {"state_dicts": [m.model_.state_dict() for m in clf.predictors_], "weights": clf.weights_.to_numpy()}
else:
    raise ValueError(f"unknown FAIRNESS_METHOD {FAIRNESS_METHOD!r}")

torch.save({**to_save, "method": FAIRNESS_METHOD, "constraint": CONSTRAINT, "fair_groups": FAIR_GROUPS, "dropped_columns": dropped,
            "features": list(X.columns), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

if FAIRNESS_METHOD != "constraints":
    pred = (probs >= THRESHOLD).astype(int)
print(f"\n=== {FAIRNESS_METHOD}, proxy filter {PROXY_FILTER}, on {' x '.join(FAIR_GROUPS)} ({CONSTRAINT if FAIRNESS_METHOD in ('adversarial', 'constraints') else '-'}) ===")
print(f"Final test accuracy: {accuracy_score(y_test, pred):.2%}")
print(f"Final test AUC:      {roc_auc_score(y_test, probs):.3f}")
print(f"Checked:             {pred.mean():.1%}")

pd.set_option("display.width", 200)
metrics = {"n": count, "checked": selection_rate, "TPR": true_positive_rate, "FPR": false_positive_rate, "accuracy": accuracy_score}
summary = []
for cols in [[c] for c in groups.columns] + [list(p) for p in itertools.combinations(groups.columns, 2)]:
    sf = A_test[cols]
    mf = MetricFrame(metrics=metrics, y_true=y_test, y_pred=pred, sensitive_features=sf)
    name = " x ".join(cols)
    print(f"\n--- {name} ---\n{mf.by_group.round(3).to_string()}")
    summary.append({"groups": name,
                    "DP ratio": demographic_parity_ratio(y_test, pred, sensitive_features=sf),
                    "EO ratio": equalized_odds_ratio(y_test, pred, sensitive_features=sf),
                    "max checked gap": mf.difference()["checked"]})
print("\nFairness summary (ratios: 1 = fair, 0 = some group never/always checked)")
print(pd.DataFrame(summary).round(3).to_string(index=False))

# For run_all_models.py
df_test = pd.DataFrame({"persoon_geslacht_vrouw": df["persoon_geslacht_vrouw"].to_numpy()[idx_test], "pred": pred,
                        "actual": y_test, "age_group": pd.Categorical(A_test["age"], ["≤30", "31-45", "46-60", "61+"])})
