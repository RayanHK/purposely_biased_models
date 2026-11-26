"""
Run every model script in this folder and collect accuracy, AUC and bias in one table.

Usage:  python run_all_models.py
        (or: from run_all_models import run_all_models; results = run_all_models(threshold=0.7))

How it works
------------
* Each script is run as-is (its own data loading, training and bias report), with its
  printed output captured instead of shown.
* Scripts with an `all_data = True/False` switch are run twice, once each way.
* After a script finishes, the results are read from the `df_test` table the script
  itself builds for its bias report (columns pred, actual, age_group, gender).
* Every model is then re-cut at the same threshold, so all rows are comparable.
* AUC is taken from what the script prints ("AUC: ..."). Scripts that don't print an
  AUC get "Not recorded".
* Bias = how many times more likely the most-checked group is to be checked than the
  least-checked group (the same "x more likely" number as the bias reports).
"""
import contextlib
import io
import os
import re
import warnings

import pandas as pd

THRESHOLD = 0.7   # who gets checked: score >= THRESHOLD. Set to None to keep each script's own cutoff.

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    # (display name, script file, has all_data switch)
    ("Neural network (all features)",        "ml_model_setais1.py",                   False),
    ("Neural network (age + gender only)",   "ml_nnmodel_biasedfeatures_setais1.py",  False),
    ("Neural network (accurate + biased)",   "ml_nnmodel_balanced_setais1.py",        False),
    ("Decision tree",                         "ml_dtmodel_setais1.py",                 True),
    ("Gradient boosting",                     "ml_dtmodel_boosting_setais1.py",        True),
    ("Gradient boosting (demographics only)", "ml_dtmodel_boosting_demographics_setais1.py",   False),
]

SWITCH_LABELS = {
    "ml_dtmodel_setais1.py":          {True: "all features", False: "age + gender only"},
    "ml_dtmodel_boosting_setais1.py": {True: "all features", False: "accurate + biased set"},
}


def _set_flag(source, name, value):
    """Replace a top-level `name = True/False` line in the script source."""
    pattern = rf"^{name}\s*=\s*(True|False)"
    if not re.search(pattern, source, flags=re.M):
        return source
    return re.sub(pattern, f"{name} = {value}", source, count=1, flags=re.M)


def _bias(checked):
    """checked: Series of check rates per group -> 'A 2.10x vs B' text and the ratio."""
    hi, lo = checked.idxmax(), checked.idxmin()
    if checked[hi] == 0:
        return "nobody checked", float("nan")
    if checked[lo] == 0:
        never = ", ".join(str(g) for g in checked.index[checked == 0])
        return f"{never} never checked", float("inf")
    ratio = checked[hi] / checked[lo]
    return f"{hi} {ratio:.2f}x vs {lo}", ratio


def run_script(script, all_data=None, retrain=False, threshold=None):
    """Run one model script and return its accuracy, AUC and bias numbers.

    threshold=None keeps the script's own decisions; a number such as 0.7 checks
    only people whose score is at least that value.
    """
    with open(os.path.join(HERE, script), encoding="utf-8") as f:
        source = f.read()
    if all_data is not None:
        source = _set_flag(source, "all_data", all_data)
    if not retrain:
        # use the saved .pt weights where a script has them, instead of training again
        source = _set_flag(source, "train_again", False)

    namespace = {"__name__": "__main__", "__file__": os.path.join(HERE, script)}
    output = io.StringIO()
    old_cwd = os.getcwd()
    os.chdir(HERE)
    try:
        with contextlib.redirect_stdout(output), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(source, script, "exec"), namespace)
    finally:
        os.chdir(old_cwd)

    df_test = namespace["df_test"].copy()
    printed = output.getvalue()

    if threshold is not None:
        # re-decide who gets checked using the model's scores and the given cutoff
        scores = namespace.get("probs")
        if scores is None:   # the decision tree script keeps no scores, so compute them
            scores = namespace["tree"].predict_proba(namespace["x_test"])[:, 1]
        df_test["pred"] = (pd.Series(scores).to_numpy() >= threshold).astype(int)

    auc_matches = re.findall(r"AUC:\s*([0-9]*\.[0-9]+)", printed)
    auc = float(auc_matches[-1]) if auc_matches else "Not recorded"

    gender = df_test["persoon_geslacht_vrouw"].map({0: "man", 1: "vrouw"})
    gender_rates = df_test["pred"].astype(float).groupby(gender).mean()
    age_rates = df_test["pred"].astype(float).groupby(df_test["age_group"], observed=True).mean()
    gender_text, gender_ratio = _bias(gender_rates)
    age_text, age_ratio = _bias(age_rates)

    return {
        "accuracy": round(float((df_test["pred"].astype(int) == df_test["actual"].astype(int)).mean()), 4),
        "AUC": round(auc, 3) if isinstance(auc, float) else auc,
        "checked": round(float(df_test["pred"].astype(float).mean()), 3),
        "gender bias": gender_text,
        "gender ratio": round(gender_ratio, 2),
        "age bias": age_text,
        "age ratio": round(age_ratio, 2),
        **{f"checked {k}": round(v, 3) for k, v in gender_rates.items()},
        **{f"checked {k}": round(v, 3) for k, v in age_rates.items()},
    }


def run_all_models(retrain=False, threshold=THRESHOLD):
    """Run every model (both ways where there is an all_data switch) and return a table.

    retrain=False loads saved .pt weights for the neural networks where available;
    retrain=True lets each script train from scratch (much slower).
    threshold: everyone with a score >= threshold gets checked (same for every model).
    threshold=None uses each script's own cutoff instead.
    """
    rows = []
    for name, script, has_switch in MODELS:
        for all_data in ([True, False] if has_switch else [None]):
            variant = SWITCH_LABELS[script][all_data] if has_switch else ""
            print(f"Running {name}{' - ' + variant if variant else ''} ...", flush=True)
            try:
                result = run_script(script, all_data, retrain, threshold)
            except Exception as e:  # keep going if one script breaks
                result = {"error": f"{type(e).__name__}: {e}"}
            rows.append({"model": name, "variant": variant, **result})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run_all_models(threshold=THRESHOLD)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    main_cols = ["model", "variant", "accuracy", "AUC", "checked", "gender bias", "age bias", "error"]
    print(f"\nThreshold: {THRESHOLD if THRESHOLD is not None else 'each script its own'}")
    print(results[[c for c in main_cols if c in results.columns]].to_string(index=False))
    out_file = f"model_comparison_{THRESHOLD}.csv" if THRESHOLD is not None else "model_comparison.csv"
    results.to_csv(out_file, index=False)
    print(f"\nFull table (incl. check rate per group) saved to {out_file}")