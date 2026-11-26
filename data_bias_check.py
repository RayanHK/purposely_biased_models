import pandas as pd

def check_callup_bias(csv_path, label_col=None, gender_col="persoon_geslacht_vrouw", age_col="persoon_leeftijd_bij_onderzoek", age_bins=(0, 30, 45, 60, 200), age_labels=("≤30", "31-45", "46-60", "61+")):
    df = pd.read_csv(csv_path)
    label_col = label_col or df.columns[-1]
    called = df[label_col].astype(float)
    overall = called.mean()

    groups = {
        "Gender": df[gender_col].map({0: "man", 1: "vrouw"}),
        "Age": pd.cut(df[age_col], bins=list(age_bins), labels=list(age_labels)),
    }

    print(f"Call-up column: {label_col}")
    print(f"Overall call-up rate: {overall:.1%} (n={len(df)})\n")
    for name, g in groups.items():
        stats = called.groupby(g, observed=True).agg(["count", "mean"])
        stats["ratio"] = stats["mean"] / overall
        line = " | ".join(f"{k}: {r['mean']:.1%} ({r['ratio']:.2f}x, n={int(r['count'])})"
                          for k, r in stats.iterrows())
        print(f"{name} bias: {line}")


check_callup_bias("investigation_train_large_checked.csv")