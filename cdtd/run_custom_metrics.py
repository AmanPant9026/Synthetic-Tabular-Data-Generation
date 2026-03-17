import glob
import os
import argparse

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, chi2_contingency, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from synthcity.metrics import eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def read_csv_table(path):
    df = pd.read_csv(path, header=None)
    df.columns = [str(i) for i in range(df.shape[1])]
    return df


def infer_column_types(df, task_type="auto"):
    cols = list(df.columns)

    # Case 1: CDTD default dataset saved without headers: 24 columns
    # 0..9 = target + categorical, 10..23 = numerical
    if len(cols) == 24 and cols == [str(i) for i in range(24)]:
        cat_cols = [str(i) for i in range(10)]
        num_cols = [str(i) for i in range(10, 24)]
        return cat_cols, num_cols

    # Case 2: named CDTD columns
    cat_cols = [c for c in cols if str(c).startswith("cat_feature_")]
    num_cols = [c for c in cols if str(c).startswith("cont_feature_")]

    if "target" in cols:
        if task_type == "regression":
            num_cols = ["target"] + num_cols
        else:
            cat_cols = ["target"] + cat_cols

    if len(cat_cols) + len(num_cols) > 0:
        return cat_cols, num_cols

    # Fallback: dtype-based
    cat_cols, num_cols = [], []
    for c in cols:
        s = df[c]
        if (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_bool_dtype(s)
            or pd.api.types.is_categorical_dtype(s)
        ):
            cat_cols.append(c)
        elif pd.api.types.is_integer_dtype(s):
            nunique = s.nunique(dropna=True)
            if nunique <= 20:
                cat_cols.append(c)
            else:
                num_cols.append(c)
        else:
            num_cols.append(c)

    if len(cat_cols) + len(num_cols) == 0:
        raise ValueError(f"No columns detected. Columns found: {cols}")

    return cat_cols, num_cols


def align_columns(real_df, syn_df):
    common_cols = [c for c in real_df.columns if c in syn_df.columns]
    return real_df[common_cols].copy(), syn_df[common_cols].copy()


def cast_df(df, cat_cols, num_cols):
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def build_preprocessor(df_fit, cat_cols, num_cols):
    transformers = []
    if len(cat_cols) > 0:
        transformers.append(("cat", make_onehot(), cat_cols))
    if len(num_cols) > 0:
        transformers.append(("num", StandardScaler(), num_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    pre.fit(df_fit)
    return pre


def evaluate_alpha_beta(real_df, syn_df, cat_cols, num_cols):
    real_num = real_df[num_cols] if len(num_cols) > 0 else pd.DataFrame(index=real_df.index)
    syn_num = syn_df[num_cols] if len(num_cols) > 0 else pd.DataFrame(index=syn_df.index)

    real_cat = real_df[cat_cols] if len(cat_cols) > 0 else pd.DataFrame(index=real_df.index)
    syn_cat = syn_df[cat_cols] if len(cat_cols) > 0 else pd.DataFrame(index=syn_df.index)

    if len(cat_cols) > 0:
        encoder = make_onehot()
        encoder.fit(real_cat.astype(str).to_numpy())
        real_cat_oh = encoder.transform(real_cat.astype(str).to_numpy())
        syn_cat_oh = encoder.transform(syn_cat.astype(str).to_numpy())
    else:
        real_cat_oh = np.empty((len(real_df), 0))
        syn_cat_oh = np.empty((len(syn_df), 0))

    real_all = pd.DataFrame(
        np.concatenate((real_num.to_numpy(dtype=float), real_cat_oh), axis=1)
    ).astype(float)

    syn_all = pd.DataFrame(
        np.concatenate((syn_num.to_numpy(dtype=float), syn_cat_oh), axis=1)
    ).astype(float)

    if real_all.shape[0] == 0 or syn_all.shape[0] == 0:
        raise ValueError(f"Alpha/Beta got zero rows: real={real_all.shape}, syn={syn_all.shape}")
    if real_all.shape[1] == 0 or syn_all.shape[1] == 0:
        raise ValueError(f"Alpha/Beta got zero cols: real={real_all.shape}, syn={syn_all.shape}")

    X_real_loader = GenericDataLoader(real_all)
    X_syn_loader = GenericDataLoader(syn_all)

    quality_evaluator = eval_statistical.AlphaPrecision()
    qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
    qual_res = {k: v for (k, v) in qual_res.items() if "naive" in k}

    return (
        float(qual_res["delta_precision_alpha_naive"]),
        float(qual_res["delta_coverage_beta_naive"]),
    )


def categorical_shape_similarity(real_col, syn_col):
    real_probs = real_col.astype(str).value_counts(normalize=True)
    syn_probs = syn_col.astype(str).value_counts(normalize=True)

    all_idx = real_probs.index.union(syn_probs.index)
    real_probs = real_probs.reindex(all_idx, fill_value=0.0)
    syn_probs = syn_probs.reindex(all_idx, fill_value=0.0)

    tv = 0.5 * np.abs(real_probs.values - syn_probs.values).sum()
    return float(max(0.0, 1.0 - tv))


def numerical_shape_similarity(real_col, syn_col):
    real = pd.to_numeric(real_col, errors="coerce").dropna().to_numpy()
    syn = pd.to_numeric(syn_col, errors="coerce").dropna().to_numpy()
    if len(real) == 0 or len(syn) == 0:
        return 0.0
    stat = ks_2samp(real, syn).statistic
    return float(max(0.0, 1.0 - stat))


def evaluate_shape(real_df, syn_df, cat_cols, num_cols):
    scores = []
    for c in cat_cols:
        scores.append(categorical_shape_similarity(real_df[c], syn_df[c]))
    for c in num_cols:
        scores.append(numerical_shape_similarity(real_df[c], syn_df[c]))
    return float(np.mean(scores))


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    corr = spearmanr(x, y).correlation
    if corr is None or np.isnan(corr):
        return 0.0
    return float(corr)


def cramers_v(x, y):
    table = pd.crosstab(pd.Series(x, dtype="object"), pd.Series(y, dtype="object"))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0

    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n <= 1:
        return 0.0

    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)

    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def correlation_ratio(categories, values):
    categories = np.asarray(categories).astype(str)
    values = np.asarray(values).astype(float)

    if np.std(values) == 0:
        return 0.0

    cats, inverse = np.unique(categories, return_inverse=True)
    if len(cats) < 2:
        return 0.0

    group_means = []
    group_counts = []
    for k in range(len(cats)):
        v = values[inverse == k]
        if len(v) == 0:
            continue
        group_means.append(v.mean())
        group_counts.append(len(v))

    group_means = np.asarray(group_means)
    group_counts = np.asarray(group_counts)

    grand_mean = values.mean()
    ss_between = np.sum(group_counts * (group_means - grand_mean) ** 2)
    ss_total = np.sum((values - grand_mean) ** 2)

    if ss_total <= 0:
        return 0.0

    return float(np.sqrt(ss_between / ss_total))


def evaluate_trend(real_df, syn_df, cat_cols):
    cols = list(real_df.columns)
    cat_set = set(cat_cols)

    sims = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            a_cat = a in cat_set
            b_cat = b in cat_set

            if not a_cat and not b_cat:
                r_real = safe_spearman(real_df[a], real_df[b])
                r_syn = safe_spearman(syn_df[a], syn_df[b])
                sim = 1.0 - (abs(r_real - r_syn) / 2.0)
            elif a_cat and b_cat:
                r_real = cramers_v(real_df[a], real_df[b])
                r_syn = cramers_v(syn_df[a], syn_df[b])
                sim = 1.0 - abs(r_real - r_syn)
            else:
                if a_cat:
                    r_real = correlation_ratio(real_df[a], real_df[b])
                    r_syn = correlation_ratio(syn_df[a], syn_df[b])
                else:
                    r_real = correlation_ratio(real_df[b], real_df[a])
                    r_syn = correlation_ratio(syn_df[b], syn_df[a])
                sim = 1.0 - abs(r_real - r_syn)

            sims.append(float(np.clip(sim, 0.0, 1.0)))

    return float(np.mean(sims))


def evaluate_c2st(real_train_df, syn_df, cat_cols, num_cols, seed=42, holdout_frac=0.2):
    real_fit, real_holdout = train_test_split(
        real_train_df, test_size=holdout_frac, random_state=seed
    )
    n = min(len(real_holdout), len(syn_df))
    if n == 0:
        raise ValueError("C2ST got zero rows.")

    real_holdout = real_holdout.sample(n=n, random_state=seed).reset_index(drop=True)
    syn_df = syn_df.sample(n=n, random_state=seed).reset_index(drop=True)

    fit_df = pd.concat([real_fit, real_holdout, syn_df], axis=0, ignore_index=True)
    pre = build_preprocessor(fit_df, cat_cols, num_cols)

    X_real = pre.transform(real_holdout)
    X_syn = pre.transform(syn_df)

    X = np.vstack([X_real, X_syn])
    y = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_syn))])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_tr)

    probs = clf.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return float(accuracy_score(y_te, preds)), float(roc_auc_score(y_te, probs))


def evaluate_dcr(real_train_df, syn_df, cat_cols, num_cols, seed=42, holdout_frac=0.2):
    real_fit, real_holdout = train_test_split(
        real_train_df, test_size=holdout_frac, random_state=seed
    )

    fit_df = pd.concat([real_fit, real_holdout, syn_df], axis=0, ignore_index=True)
    pre = build_preprocessor(fit_df, cat_cols, num_cols)

    X_train = pre.transform(real_fit)
    X_holdout = pre.transform(real_holdout)
    X_syn = pre.transform(syn_df)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_train)

    d_syn, _ = nn.kneighbors(X_syn)
    d_holdout, _ = nn.kneighbors(X_holdout)

    d_syn = d_syn.reshape(-1)
    d_holdout = d_holdout.reshape(-1)

    return float(d_syn.mean()), float(d_holdout.mean()), float(d_syn.mean() / (d_holdout.mean() + 1e-12))


def evaluate_one_sample(real_df, syn_df, task_type="auto", seed=42):
    real_df, syn_df = align_columns(real_df, syn_df)

    cat_cols, num_cols = infer_column_types(real_df, task_type=task_type)

    print("Detected categorical columns:", cat_cols)
    print("Detected numerical columns:", num_cols)

    real_df = cast_df(real_df, cat_cols, num_cols)
    syn_df = cast_df(syn_df, cat_cols, num_cols)

    if real_df.shape[0] == 0 or syn_df.shape[0] == 0:
        raise ValueError(f"Empty dataframe after preprocessing. real={real_df.shape}, syn={syn_df.shape}")

    real_df, syn_df = align_columns(real_df, syn_df)
    final_cols = list(real_df.columns)
    cat_cols = [c for c in cat_cols if c in final_cols]
    num_cols = [c for c in num_cols if c in final_cols]

    alpha, beta = evaluate_alpha_beta(real_df, syn_df, cat_cols, num_cols)
    shape = evaluate_shape(real_df, syn_df, cat_cols, num_cols)
    trend = evaluate_trend(real_df, syn_df, cat_cols)
    c2st_acc, c2st_auc = evaluate_c2st(real_df, syn_df, cat_cols, num_cols, seed=seed)
    dcr_gen, dcr_test, dcr_ratio = evaluate_dcr(real_df, syn_df, cat_cols, num_cols, seed=seed)

    return {
        "shape": shape * 100.0,
        "trend": trend * 100.0,
        "c2st_acc": c2st_acc * 100.0,
        "c2st_auc": c2st_auc * 100.0,
        "alpha": alpha * 100.0,
        "beta": beta * 100.0,
        "dcr_gen": dcr_gen,
        "dcr_test": dcr_test,
        "dcr_ratio": dcr_ratio,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", type=str, required=True)
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--sample_pattern", type=str, default="gen*.csv")
    parser.add_argument("--task_type", type=str, default="classification")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="")
    args = parser.parse_args()

    real_df = read_csv_table(args.real_path)

    sample_paths = sorted(glob.glob(os.path.join(args.sample_dir, args.sample_pattern)))
    print(f"{len(sample_paths)} samples loaded from {args.sample_dir} with pattern {args.sample_pattern}")

    rows = []
    for syn_path in sample_paths:
        try:
            syn_df = read_csv_table(syn_path)
            metrics = evaluate_one_sample(real_df, syn_df, task_type=args.task_type, seed=args.seed)
            metrics["sample"] = os.path.basename(syn_path)
            rows.append(metrics)

            print(f"\nSample: {os.path.basename(syn_path)}")
            for k, v in metrics.items():
                if k != "sample":
                    print(f"{k}: {v:.6f}")
        except Exception as e:
            print(f"\nSkipping {os.path.basename(syn_path)} because of error: {e}")

    if len(rows) == 0:
        raise ValueError("No sample was evaluated successfully.")

    quality = pd.DataFrame(rows)
    cols = ["sample", "shape", "trend", "c2st_acc", "c2st_auc", "alpha", "beta", "dcr_gen", "dcr_test", "dcr_ratio"]
    quality = quality[cols]

    metric_cols = [c for c in quality.columns if c != "sample"]
    avg = quality[metric_cols].mean(axis=0).round(2)
    std = quality[metric_cols].std(axis=0, ddof=0).round(2)

    avg_std = pd.concat([avg, std], axis=1)
    avg_std.columns = ["avg", "std"]

    save_dir = args.save_dir if args.save_dir is not None else os.path.dirname(args.sample_dir)
    os.makedirs(save_dir, exist_ok=True)

    quality_path = os.path.join(save_dir, f"{args.prefix}quality.csv")
    avg_std_path = os.path.join(save_dir, f"{args.prefix}avg_std.csv")

    quality.to_csv(quality_path, index=False)
    avg_std.to_csv(avg_std_path, index=True)

    print("\n===== SUMMARY =====")
    print(avg_std)
    print(f"\nSaved sample metrics -> {quality_path}")
    print(f"Saved summary       -> {avg_std_path}")