import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

THRESHOLD = 0.7

# Load data (same columns as your neural network)
df = pd.read_csv("investigation_train_large_checked.csv")

all_data = False

# "Accurate + biased" feature set (from feature_search.py / feature_sets.json)
FEATURES = [
    "persoon_leeftijd_bij_onderzoek",
    "pla_historie_ontwikkeling",
    "pla_hist_pla_categorie_doelstelling_16",
    "afspraak_resultaat_ingevuld_uniek",
    "persoonlijke_eigenschappen_dagen_sinds_taaleis",
    "instrument_reden_beeindiging_historie_succesvol",
    "contacten_onderwerp_no_show",
    "ontheffing_dagen_hist_vanwege_uw_medische_omstandigheden",
    "persoonlijke_eigenschappen_dagen_sinds_opvoer",
    "afspraak_signaal_voor_medewerker",
    "relatie_overig_actueel_vorm__kostendeler",
    "contacten_onderwerp_beoordelen_taaleis",
    "afspraak_inspanningsperiode",
    "pla_einde_doelstelling_bereikt",
]

if all_data:  #Actually more biased
    x = df.iloc[:, :-3]
    y = df.iloc[:, -1]

    # Same split and seed as the neural network
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Train
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=42)
    model.fit(x_train, y_train)
else:
    x = df[FEATURES]
    y = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                           class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)

# Evaluate
probs = model.predict_proba(x_test)[:, 1]
pred = probs >= THRESHOLD
print(f"Accuracy: {accuracy_score(y_test, pred):.2%}")
print(f"AUC:      {roc_auc_score(y_test, probs):.3f}")
print(classification_report(y_test, pred))

# Which features the model relies on most: how much AUC drops when a feature is shuffled
sample = x_test.sample(5000, random_state=42)
imp = permutation_importance(model, sample, y_test.loc[sample.index], scoring="roc_auc",
                             n_repeats=3, random_state=42)
importance = pd.Series(imp.importances_mean, index=x.columns).sort_values(ascending=False)
print("Top 10 features (AUC drop when shuffled):")
print(importance.head(10))

#Bias

df_test = x_test.copy()
df_test["persoon_geslacht_vrouw"] = df.loc[x_test.index, "persoon_geslacht_vrouw"]
df_test["pred"] = pred
df_test["actual"] = y_test.to_numpy()
df_test["age_group"] = pd.cut(df_test["persoon_leeftijd_bij_onderzoek"], [0, 30, 45, 60, 200], labels=["≤30", "31-45", "46-60", "61+"])

def bias_report(col, title, names=None):
    overall = df_test["pred"].mean()
    g = df_test.groupby(col, observed=True).agg(
        n=("pred", "size"), flagged=("pred", "mean"), actual=("actual", "mean"))
    if names:
        g.index = g.index.map(names)

    print(f"\n{title} (overall flag rate: {overall:.1%})")
    for k, r in g.iterrows():
        print(f"  {str(k):<6} flagged {r['flagged']:6.1%} | actual {r['actual']:6.1%} | "
              f"{r['flagged'] / overall:.2f}x vs average | "
              f"model/actual {r['flagged'] / r['actual']:.2f} | n={int(r['n'])}")

    if g["flagged"].min() > 0:
        hi, lo = g["flagged"].idxmax(), g["flagged"].idxmin()
        print(f"  -> {hi} is {g.loc[hi, 'flagged'] / g.loc[lo, 'flagged']:.2f}x more likely to be flagged than {lo}")
    else:
        print(f"  -> some groups are never flagged: {list(g.index[g['flagged'] == 0])}")

bias_report("persoon_geslacht_vrouw", "Gender bias", {0.0: "man", 1.0: "vrouw"})
bias_report("age_group", "Age bias")