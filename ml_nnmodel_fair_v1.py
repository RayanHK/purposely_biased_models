"""
Fair model v1
=============

Predicts who gets "checked" with a 5-layer neural network (MLP), while trying not to
treat people differently based on age, gender, neighborhood or spoken language.

The pipeline, top to bottom:
  1. Load the data and describe each person's protected groups (age, gender, ...).
  2. Filter layer: remove the protected columns (and optionally their proxies) so the
     model never sees them.
  3. Train the MLP with one of four fairness methods (FAIRNESS_METHOD):
       "none"         plain training, only the filter layer protects anyone
       "reweighing"   give each training row a weight so every group has the same
                      share of "checked" people (Kamiran & Calders)
       "adversarial"  a second network tries to guess the person's group from the
                      model's score, and the model is trained to make that guess fail
                      (fairlearn AdversarialFairnessClassifier, Zhang et al. 2018)
       "constraints"  train several models and mix them so that check rates are close
                      across groups (fairlearn ExponentiatedGradient, Agarwal et al. 2018)
  4. Save the model and print accuracy plus a fairness report per group and per pair of groups.

Needs fairlearn:  uv pip install fairlearn
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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch import nn

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
FAIRNESS_METHOD = "reweighing"   # "none" | "reweighing" | "adversarial" | "constraints"
PROXY_FILTER = "A"               # "none": drop only the protected columns
                                 # "A":    also drop 41 proxy columns (see PROXIES below)
                                 # "B":    A + 4 more proxies that also predict "checked" well
FAIR_GROUPS = ["age", "gender"]  # groups the fairness method protects (combined, e.g. "≤30|vrouw").
                                 # All 4 together makes too many small groups and training breaks down.
CONSTRAINT = "demographic_parity"  # what "fair" means for adversarial/constraints:
                                   # "demographic_parity" = equal check rates per group
                                   # "equalized_odds"     = equal error rates per group
ADV_ALPHA = 0.02                 # how hard the adversary pushes; 0.05+ often collapses to "check nobody"
THRESHOLD = 0.7                  # score at or above this means "checked"
EPOCHS = 10
SEED = 42
MODEL_PATH = f"fair_v1_{FAIRNESS_METHOD}_{PROXY_FILTER}.pt"

torch.manual_seed(SEED)
device = "cuda" if torch.accelerator.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 1. Data and protected groups
# ---------------------------------------------------------------------------
df = pd.read_csv("investigation_train_large_checked.csv")
X_all = df.iloc[:, :-3]          # last 3 columns are Ja, Nee (leak the answer) and checked (the label)
y = df["checked"].to_numpy()


def columns_containing(*words):
    return [c for c in X_all.columns if any(w in c for w in words)]


# Each person's group per protected attribute. Only used to train the fairness methods
# and to measure fairness; never given to the model as input.
wijk_columns = df.filter(regex=r"^adres_recentste_wijk_")
groups = pd.DataFrame({
    "age": pd.cut(df["persoon_leeftijd_bij_onderzoek"], [0, 30, 45, 60, 200],
                  labels=["≤30", "31-45", "46-60", "61+"]).astype(str),
    "gender": df["persoon_geslacht_vrouw"].map({0: "man", 1: "vrouw"}),
    # one-hot wijk columns -> one label; "geen" = no Rotterdam wijk recorded
    "neighborhood": wijk_columns.idxmax(axis=1).str.removeprefix("adres_recentste_wijk_")
                                .where(wijk_columns.sum(axis=1) > 0, "geen"),
    # spoken language as a stand-in for immigration background (assumes 57 = Dutch, 0 = not recorded)
    "language": df["persoonlijke_eigenschappen_spreektaal"].map(
        lambda code: "nl" if code == 57 else "onbekend" if code == 0 else "anders"),
})

# ---------------------------------------------------------------------------
# 2. Filter layer: columns the model is not allowed to see
# ---------------------------------------------------------------------------
PROTECTED = ["persoon_leeftijd_bij_onderzoek", "persoon_geslacht_vrouw",
             *columns_containing("adres_recentste_", "adres_recentst_onderdeel",      # wijk, buurt, plaats
                                 "persoonlijke_eigenschappen_spreektaal")]

# Proxies: other columns from which the protected attributes can be guessed
# (found by training a model to predict each attribute from the remaining columns).
PROXIES = {
    "age": ["adres_dagen_op_adres", "contacten_soort_document__uitgaand_",
            "contacten_soort_afgelopenjaar_document__uitgaand_",
            "relatie_kind_basisschool_kind", "relatie_kind_volwassen"],
    "gender": ["relatie_kind_leeftijd_verschil_ouder_eerste_kind", "relatie_kind_huidige_aantal",
               "relatie_kind_heeft_kinderen", "relatie_kind_jongvolwassen", "relatie_kind_tiener",
               "relatie_partner_totaal_dagen_partner", "relatie_overig_historie_vorm__onderhoudsplichtige"],
    "neighborhood": ["adres_aantal_verzendadres", "adres_aantal_brp_adres",
                     "adres_aantal_verschillende_wijken", "adres_unieke_wijk_ratio"],
    "language": [*columns_containing("taaleis", "persoonlijke_eigenschappen_nl_", "inburger"),
                 "persoonlijke_eigenschappen_opstelling", "persoonlijke_eigenschappen_communicatie_opm",
                 "belemmering_hist_taal"],
}
# Version B also drops these. They are proxies too, but also strong predictors of "checked".
EXTRA_PROXIES_B = ["pla_historie_ontwikkeling", "instrument_reden_beeindiging_historie_succesvol",
                   "afspraak_signaal_voor_medewerker", "typering_dagen_som"]

to_drop = set(PROTECTED)
if PROXY_FILTER in ("A", "B"):
    to_drop.update(c for cols in PROXIES.values() for c in cols)
if PROXY_FILTER == "B":
    to_drop.update(EXTRA_PROXIES_B)
dropped = [c for c in X_all.columns if c in to_drop]
X = X_all.drop(columns=dropped)
print(f"Filter layer drops {len(dropped)} columns: {dropped}\n{X.shape[1]} features go into the MLP")

# Same 70/30 split as the other scripts. Scale features to mean 0 / std 1 and clip extreme
# values (typering_hist_ind reaches 100 std), which otherwise make adversarial training blow up.
idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.3, random_state=SEED)
scaler = StandardScaler().fit(X.iloc[idx_train])
X_train = np.clip(scaler.transform(X.iloc[idx_train]), -5, 5)
X_test = np.clip(scaler.transform(X.iloc[idx_test]), -5, 5)
y_train, y_test = y[idx_train], y[idx_test]
groups_train = groups.iloc[idx_train].reset_index(drop=True)
groups_test = groups.iloc[idx_test].reset_index(drop=True)


def combined_group(group_table):
    """One label per person for the combination of groups, e.g. "≤30|vrouw"."""
    return group_table.astype(str).agg("|".join, axis=1)


# ---------------------------------------------------------------------------
# 3. Model and fairness methods
# ---------------------------------------------------------------------------
def make_mlp(n_inputs):
    """5 linear layers; outputs a raw score (logit), sigmoid turns it into a probability."""
    return nn.Sequential(nn.Linear(n_inputs, 256), nn.ReLU(),
                         nn.Linear(256, 128), nn.ReLU(),
                         nn.Linear(128, 64), nn.ReLU(),
                         nn.Linear(64, 32), nn.ReLU(),
                         nn.Linear(32, 1))


class TorchMLP(ClassifierMixin, BaseEstimator):
    """The MLP with an sklearn-style fit/predict, so fairlearn can train it with per-row weights."""

    def __init__(self, epochs=EPOCHS, batch_size=256, lr=1e-3):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0, 1])
        self.model_ = make_mlp(X.shape[1]).to(device)
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        data = [torch.tensor(np.asarray(a), dtype=torch.float32) for a in (X, y, weights / weights.mean())]
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data),
                                             batch_size=self.batch_size, shuffle=True)
        # Only ~15% are checked; pos_weight counts them more so the model doesn't just say "no" to everyone
        pos_weight = torch.tensor((y == 0).sum() / (y == 1).sum(), device=device)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        optimiser = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                optimiser.zero_grad()
                loss = (loss_fn(self.model_(xb).squeeze(1), yb) * wb).mean()
                loss.backward()
                optimiser.step()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            scores = self.model_(torch.tensor(np.asarray(X), dtype=torch.float32, device=device))
            p = torch.sigmoid(scores).cpu().numpy()
        return np.hstack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= THRESHOLD).astype(int)


def reweighing_weights(group_table, y):
    """Weight = how often (group, label) would occur if label and group were unrelated,
    divided by how often it actually occurs. Under-checked groups' checked rows get weight > 1."""
    t = pd.DataFrame({"group": combined_group(group_table), "label": y})
    p_group = t["group"].map(t["group"].value_counts(normalize=True))
    p_label = t["label"].map(t["label"].value_counts(normalize=True))
    p_group_and_label = t.groupby(["group", "label"])["label"].transform("size") / len(t)
    return (p_group * p_label / p_group_and_label).to_numpy()


def train_plain(sample_weight=None):
    """Methods "none" and "reweighing"."""
    clf = TorchMLP().fit(X_train, y_train, sample_weight)
    probs = clf.predict_proba(X_test)[:, 1]
    return probs, (probs >= THRESHOLD).astype(int), {"state_dict": clf.model_.state_dict()}


def train_adversarial():
    # fairlearn's adversarial training has no class weights, so repeat "checked" rows until both classes are equal
    pos, neg = np.where(y_train == 1)[0], np.where(y_train == 0)[0]
    rows = np.concatenate([neg, resample(pos, n_samples=len(neg), random_state=SEED)])
    sensitive = combined_group(groups_train.iloc[rows][FAIR_GROUPS]).to_numpy()
    layers = [256, "leaky_relu", 128, "leaky_relu", 64, "leaky_relu", 32, "leaky_relu"]   # same 5-layer shape
    clf = AdversarialFairnessClassifier(backend="torch", predictor_model=layers, adversary_model=[32, "leaky_relu"],
                                        constraints=CONSTRAINT, epochs=EPOCHS, batch_size=256, shuffle=True,
                                        random_state=SEED, learning_rate=3e-4, alpha=0.0, warm_start=True)
    # Step 1: train normally (alpha=0, adversary has no effect). Step 2: continue training against the
    # adversary. Starting with the adversary straight away often collapses the model.
    clf.fit(X_train[rows], y_train[rows], sensitive_features=sensitive)
    clf.set_params(alpha=ADV_ALPHA).fit(X_train[rows], y_train[rows], sensitive_features=sensitive)
    probs = clf._raw_predict(X_test).ravel()
    return probs, (probs >= THRESHOLD).astype(int), {"state_dict": clf.backendEngine_.predictor_model.state_dict()}


def train_constraints():
    rule = DemographicParity() if CONSTRAINT == "demographic_parity" else EqualizedOdds()
    clf = ExponentiatedGradient(TorchMLP(epochs=max(EPOCHS // 3, 1), batch_size=1024), rule, max_iter=10)
    clf.fit(X_train, y_train, sensitive_features=groups_train[FAIR_GROUPS])
    # The result is a random mix of models: each person is judged by one of them, picked at random
    pred = clf.predict(X_test, random_state=SEED)
    probs = clf._pmf_predict(X_test)[:, 1]   # chance the mix checks this person (used for AUC)
    saved = {"state_dicts": [m.model_.state_dict() for m in clf.predictors_], "weights": clf.weights_.to_numpy()}
    return probs, pred, saved


if FAIRNESS_METHOD == "none":
    probs, pred, saved = train_plain()
elif FAIRNESS_METHOD == "reweighing":
    probs, pred, saved = train_plain(reweighing_weights(groups_train[FAIR_GROUPS], y_train))
elif FAIRNESS_METHOD == "adversarial":
    probs, pred, saved = train_adversarial()
elif FAIRNESS_METHOD == "constraints":
    probs, pred, saved = train_constraints()
else:
    raise ValueError(f"unknown FAIRNESS_METHOD {FAIRNESS_METHOD!r}")

# ---------------------------------------------------------------------------
# 4. Save and report
# ---------------------------------------------------------------------------
torch.save({**saved, "method": FAIRNESS_METHOD, "constraint": CONSTRAINT, "fair_groups": FAIR_GROUPS,
            "dropped_columns": dropped, "features": list(X.columns),
            "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

uses_constraint = FAIRNESS_METHOD in ("adversarial", "constraints")
print(f"\n=== {FAIRNESS_METHOD}, proxy filter {PROXY_FILTER}, on {' x '.join(FAIR_GROUPS)} "
      f"({CONSTRAINT if uses_constraint else '-'}) ===")
print(f"Final test accuracy: {accuracy_score(y_test, pred):.2%}")
print(f"Final test AUC:      {roc_auc_score(y_test, probs):.3f}")
print(f"Checked:             {pred.mean():.1%}")

# Per group: n, share checked, TPR (share of truly-checked people the model catches),
# FPR (share of not-checked people the model wrongly flags), accuracy
pd.set_option("display.width", 200)
metrics = {"n": count, "checked": selection_rate, "TPR": true_positive_rate,
           "FPR": false_positive_rate, "accuracy": accuracy_score}
singles = [[name] for name in groups.columns]
pairs = [list(pair) for pair in itertools.combinations(groups.columns, 2)]
summary = []
for cols in singles + pairs:
    sensitive = groups_test[cols]
    report = MetricFrame(metrics=metrics, y_true=y_test, y_pred=pred, sensitive_features=sensitive)
    name = " x ".join(cols)
    print(f"\n--- {name} ---\n{report.by_group.round(3).to_string()}")
    summary.append({
        "groups": name,
        "DP ratio": demographic_parity_ratio(y_test, pred, sensitive_features=sensitive),  # lowest / highest check rate
        "EO ratio": equalized_odds_ratio(y_test, pred, sensitive_features=sensitive),      # same for TPR and FPR (worst one)
        "max checked gap": report.difference()["checked"],                                  # highest - lowest check rate
    })
print("\nFairness summary (ratios: 1 = fair, 0 = some group never/always checked)")
print(pd.DataFrame(summary).round(3).to_string(index=False))

# For run_all_models.py
df_test = pd.DataFrame({"persoon_geslacht_vrouw": df["persoon_geslacht_vrouw"].to_numpy()[idx_test], "pred": pred,
                        "actual": y_test, "age_group": pd.Categorical(groups_test["age"], ["≤30", "31-45", "46-60", "61+"])})
