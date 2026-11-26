import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report

# Load data (same columns as your neural network)
df = pd.read_csv("investigation_train_large_checked.csv")

all_data = True

if all_data:  #Actually more biased
    x = df.iloc[:, :-3]
    y = df.iloc[:, -1]

    # Same split and seed as the neural network
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Train
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(x_train, y_train)
else:
    x = df[["persoon_geslacht_vrouw", "persoon_leeftijd_bij_onderzoek"]]
    y = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    tree = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
    tree.fit(x_train, y_train)

# Evaluate
pred = tree.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.2%}")
print(classification_report(y_test, pred))

# Which features the tree relies on most
importance = pd.Series(tree.feature_importances_, index=x.columns).sort_values(ascending=False)
print("Top 10 features:")
print(importance.head(10))

# The tree's rules as readable text
print(export_text(tree, feature_names=list(x.columns)))

#Bias

df_test = x_test.copy()
df_test["pred"] = tree.predict(x_test)
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

    hi, lo = g["flagged"].idxmax(), g["flagged"].idxmin()
    print(f"  -> {hi} is {g.loc[hi, 'flagged'] / g.loc[lo, 'flagged']:.2f}x more likely to be flagged than {lo}")

bias_report("persoon_geslacht_vrouw", "Gender bias", {0.0: "man", 1.0: "vrouw"})
bias_report("age_group", "Age bias")