# app.py — Streamlit app for Drift, Fairness & Explainability (with optional pickle model upload)

import io
import json
import pickle
import textwrap
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
try:
    from scipy import stats
except ImportError:  # allow app to run without scipy (KS will be NaN)
    stats = None
import streamlit as st

# Explainability deps
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap

warnings.filterwarnings("ignore")

# =============================
# Helpers
# =============================

def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index for numeric arrays (bins from baseline)."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return np.nan
    quantiles = np.linspace(0, 1, buckets + 1)
    cuts = np.unique(np.quantile(expected if not np.all(expected == expected[0]) else actual, quantiles))
    if cuts.size < 2:
        return np.nan
    expected_bins = np.clip(np.digitize(expected, cuts, right=False) - 1, 0, len(cuts) - 2)
    actual_bins = np.clip(np.digitize(actual, cuts, right=False) - 1, 0, len(cuts) - 2)

    exp_counts = np.bincount(expected_bins, minlength=len(cuts) - 1).astype(float)
    act_counts = np.bincount(actual_bins, minlength=len(cuts) - 1).astype(float)

    exp_props = exp_counts / exp_counts.sum() if exp_counts.sum() else np.zeros_like(exp_counts)
    act_props = act_counts / act_counts.sum() if act_counts.sum() else np.zeros_like(act_counts)

    exp_props = np.where(exp_props == 0, 1e-6, exp_props)
    act_props = np.where(act_props == 0, 1e-6, act_props)

    psi_vals = (act_props - exp_props) * np.log(act_props / exp_props)
    return float(np.sum(psi_vals))


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() if p.sum() else 1)
    q = q / (q.sum() if q.sum() else 1)
    return 0.5 * float(np.abs(p - q).sum())


def ks_stat(x: np.ndarray, y: np.ndarray) -> float:
    if stats is None:
        return np.nan
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    return float(stats.ks_2samp(x, y, mode="auto").statistic)


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() if p.sum() else 1)
    q = q / (q.sum() if q.sum() else 1)
    m = 0.5 * (p + q)

    def _kl(a, b):
        a = np.where(a == 0, 1e-12, a)
        b = np.where(b == 0, 1e-12, b)
        return np.sum(a * np.log(a / b))

    js_div = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.sqrt(js_div))


def categorical_dist(series: pd.Series) -> Tuple[np.ndarray, List[str]]:
    counts = series.value_counts(dropna=False)
    return counts.values.astype(float), counts.index.astype(str).tolist()


# =============================
# Fairness metrics
# =============================

def group_rates(y_true: pd.Series, y_pred: pd.Series, group: pd.Series, positive_label=1) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "yhat": y_pred, "g": group}).dropna()
    groups = []
    for g, sub in df.groupby("g"):
        tp = ((sub["yhat"] == positive_label) & (sub["y"] == positive_label)).sum()
        fp = ((sub["yhat"] == positive_label) & (sub["y"] != positive_label)).sum()
        tn = ((sub["yhat"] != positive_label) & (sub["y"] != positive_label)).sum()
        fn = ((sub["yhat"] != positive_label) & (sub["y"] == positive_label)).sum()
        pos_rate = (sub["yhat"] == positive_label).mean() if len(sub) else np.nan
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        prevalence = (sub["y"] == positive_label).mean() if len(sub) else np.nan
        groups.append({
            "group": g,
            "n": len(sub),
            "selection_rate": pos_rate,
            "tpr": tpr,
            "fpr": fpr,
            "precision": precision,
            "prevalence": prevalence,
        })
    return pd.DataFrame(groups)


def fairness_summary(rates: pd.DataFrame, reference_group: str = None) -> pd.DataFrame:
    if rates.empty:
        return rates
    rates = rates.sort_values("selection_rate", ascending=False).reset_index(drop=True)
    ref = reference_group if reference_group in set(rates["group"]) else rates.iloc[0]["group"]
    rates["is_reference"] = rates["group"] == ref
    ref_row = rates[rates["group"] == ref].iloc[0]

    def safe_div(a, b):
        return a / b if (b is not None and not pd.isna(b) and b != 0) else np.nan

    rates["dp_diff"] = rates["selection_rate"] - float(ref_row["selection_rate"])  # Demographic Parity difference
    rates["di_ratio"] = rates["selection_rate"].apply(lambda x: safe_div(x, ref_row["selection_rate"]))  # Disparate Impact
    rates["eodiff_tpr"] = rates["tpr"] - float(ref_row["tpr"])  # Equal Opportunity difference
    rates["eodds"] = (np.abs(rates["tpr"] - float(ref_row["tpr"])) + np.abs(rates["fpr"] - float(ref_row["fpr"])))/2.0
    return rates


# =============================
# Comment generators
# =============================

def drift_comments(numeric_drift_df: pd.DataFrame, categorical_drift_df: pd.DataFrame) -> Tuple[str, str]:
    """Returns (non_technical, technical) comment blocks for drift."""
    nontech_msgs: List[str] = []
    tech_msgs: List[str] = []

    # Numeric features
    for _, row in numeric_drift_df.iterrows():
        feat = row.get("feature")
        psi_v = row.get("psi")
        ks_v = row.get("ks")
        flag = None
        if pd.notna(psi_v):
            if psi_v >= 0.25:
                flag = "high"
            elif psi_v >= 0.1:
                flag = "moderate"
        if flag:
            nontech_msgs.append(f"{feat}: {flag.capitalize()} change vs. baseline. This likely means the data the model sees has shifted.")
        if pd.notna(ks_v) and ks_v > 0.2:
            tech_msgs.append(f"{feat}: KS={ks_v:.3f} indicates distributional drift.")
        if pd.notna(psi_v):
            tech_msgs.append(f"{feat}: PSI={psi_v:.3f} ({'significant' if psi_v>=0.25 else 'moderate' if psi_v>=0.1 else 'small'}).")

    # Categorical features
    for _, row in categorical_drift_df.iterrows():
        feat = row.get("feature")
        tvd_v = row.get("tvd")
        js_v = row.get("js")
        if pd.notna(tvd_v) and tvd_v > 0.1:
            nontech_msgs.append(f"{feat}: noticeable change in category mix compared to baseline.")
            tech_msgs.append(f"{feat}: TVD={tvd_v:.3f} (>|0.1|) suggests discrete shift.")
        if pd.notna(js_v) and js_v > 0.1:
            tech_msgs.append(f"{feat}: JS distance={js_v:.3f} indicates change in distribution.")

    if not nontech_msgs:
        nontech_msgs.append("No concerning data shifts detected. The model is likely seeing similar data to the baseline.")
    if not tech_msgs:
        tech_msgs.append("No drift metrics exceeded default thresholds.")

    return "\n".join(nontech_msgs), "\n".join(tech_msgs)



def fairness_comments(fair_df: pd.DataFrame) -> Tuple[str, str]:
    nontech: List[str] = []
    tech: List[str] = []
    if fair_df.empty:
        return ("Fairness metrics could not be computed (check inputs).", "No fairness stats available.")

    for _, r in fair_df.iterrows():
        g = r.get("group")
        di = r.get("di_ratio")
        dp = r.get("dp_diff")
        eod = r.get("eodds")
        eopp = r.get("eodiff_tpr")
        concerns = []
        if pd.notna(di) and di < 0.8:
            concerns.append("low selection ratio vs. reference (<0.8)")
        if pd.notna(dp) and abs(dp) > 0.10:
            concerns.append("large difference in selection rates (>|0.10|)")
        if pd.notna(eopp) and abs(eopp) > 0.10:
            concerns.append("unequal true positive rates (>|0.10|)")
        if pd.notna(eod) and eod > 0.10:
            concerns.append("unequalized odds (>|0.10|)")
        if concerns:
            nontech.append(f"Group '{g}': potential fairness risk – " + ", ".join(concerns) + ".")
        tech.append(
            (
                f"Group {g}: n={int(r['n']) if not pd.isna(r.get('n', np.nan)) else 'NA'} | sel_rate={r['selection_rate'] if not pd.isna(r.get('selection_rate', np.nan)) else np.nan:.3f} "
                f"| DI={r['di_ratio'] if not pd.isna(r.get('di_ratio', np.nan)) else np.nan:.3f} "
                f"| DP diff={r['dp_diff'] if not pd.isna(r.get('dp_diff', np.nan)) else np.nan:.3f} "
                f"| TPR={r['tpr'] if not pd.isna(r.get('tpr', np.nan)) else np.nan:.3f} "
                f"| FPR={r['fpr'] if not pd.isna(r.get('fpr', np.nan)) else np.nan:.3f} "
                f"| EOdds={r['eodds'] if not pd.isna(r.get('eodds', np.nan)) else np.nan:.3f}"
            )
        )
    if not nontech:
        nontech.append("No major fairness concerns based on chosen thresholds. Groups have comparable selection and error rates.")
    return "\n".join(nontech), "\n".join(tech)



# =============================
# UI
# =============================

st.set_page_config(page_title="Model Drift, Fairness & Explainability", layout="wide")

st.title("📊 Model Drift, Fairness & Explainability")
st.write(
    "Upload baseline and current datasets (or a precomputed metrics file). "
    "The app will compute drift and fairness metrics, then generate comments for both technical and non-technical audiences."
)

with st.sidebar:
    st.header("Inputs")
    st.markdown("### Option A — Two datasets")
    baseline_csv = st.file_uploader("Baseline dataset (CSV)", type=["csv"], key="baseline")
    current_csv = st.file_uploader("Current dataset (CSV)", type=["csv"], key="current")

    st.caption(
        "Both files should include: feature columns, a prediction column (optional), a true label column (optional), and at least one sensitive attribute column (optional)."
    )

    pred_col = st.text_input("Prediction column (binary; optional)", value="prediction")
    label_col = st.text_input("Label column (binary; optional)", value="label")
    group_col = st.text_input("Sensitive attribute column (optional)", value="group")
    positive_label = st.number_input("Positive class value", value=1, step=1)

    st.markdown("---")
    st.markdown("### Option B — Precomputed metrics")
    metrics_file = st.file_uploader("Metrics JSON/CSV (optional)", type=["json", "csv"], key="metrics")
    st.caption(
        "Accepts JSON with keys: 'numeric_drift' (list of {feature, psi, ks}), "
        "'categorical_drift' (list of {feature, tvd, js}), and 'fairness' (table with group metrics)."
    )

    st.markdown("---")
    st.subheader("Option C — Upload model (.pkl/.joblib)")
    model_file = st.file_uploader("Model file (optional)", type=["pkl", "pickle", "joblib"], key="model")
    st.caption("If provided and your current CSV lacks a prediction column, the app will compute predictions using the uploaded model.")

    st.markdown("---")
    st.subheader("Thresholds")
    psi_warn = st.slider("PSI moderate threshold", 0.0, 1.0, 0.10, 0.01)
    psi_alert = st.slider("PSI high threshold", 0.0, 1.0, 0.25, 0.01)
    ks_flag = st.slider("KS flag threshold", 0.0, 1.0, 0.20, 0.01)
    tvd_flag = st.slider("TVD flag threshold (categorical)", 0.0, 1.0, 0.10, 0.01)
    js_flag = st.slider("JS distance flag threshold (categorical)", 0.0, 1.0, 0.10, 0.01)
    di_flag = st.slider("Disparate Impact critical (<)", 0.0, 2.0, 0.80, 0.01)
    dp_flag = st.slider("|Demographic Parity diff| (>)", 0.0, 1.0, 0.10, 0.01)
    eopp_flag = st.slider("|Equal Opportunity diff| (>)", 0.0, 1.0, 0.10, 0.01)
    eodds_flag = st.slider("Equalized Odds (>)", 0.0, 1.0, 0.10, 0.01)

    st.markdown("---")
    st.download_button(
        label="Download CSV templates",
        file_name="templates.zip",
        data=io.BytesIO(_ := b""),
        mime="application/zip",
        help="See 'Sample data & schema' in the main panel for details."
    )

# Placeholders
numeric_drift_df = pd.DataFrame(columns=["feature", "psi", "ks"]).astype({"feature": str, "psi": float, "ks": float})
categorical_drift_df = pd.DataFrame(columns=["feature", "tvd", "js"]).astype({"feature": str, "tvd": float, "js": float})
fair_df = pd.DataFrame()

# =============================
# Load / compute metrics
# =============================

if metrics_file is not None:
    try:
        if metrics_file.name.lower().endswith(".json"):
            mobj = json.load(metrics_file)
            numeric_drift_df = pd.DataFrame(mobj.get("numeric_drift", []))
            categorical_drift_df = pd.DataFrame(mobj.get("categorical_drift", []))
            fair_df = pd.DataFrame(mobj.get("fairness", []))
        else:
            mtab = pd.read_csv(metrics_file)
            if set(["feature", "psi", "ks"]).issubset(mtab.columns):
                numeric_drift_df = mtab[["feature", "psi", "ks"]].copy()
            if set(["feature", "tvd", "js"]).issubset(mtab.columns):
                categorical_drift_df = mtab[["feature", "tvd", "js"]].copy()
            fcols = {"group", "n", "selection_rate", "tpr", "fpr", "precision", "prevalence", "di_ratio", "dp_diff", "eodiff_tpr", "eodds", "is_reference"}
            if fcols.intersection(set(mtab.columns)):
                fair_df = mtab[list(fcols.intersection(set(mtab.columns)))].copy()
    except Exception as e:
        st.error(f"Failed to parse metrics file: {e}")

# If datasets provided, compute from scratch
if baseline_csv is not None and current_csv is not None:
    try:
        base = pd.read_csv(baseline_csv)
        curr = pd.read_csv(current_csv)
        st.success("Datasets loaded.")

        # If a model is uploaded and the prediction column is missing, try to predict
        if model_file is not None and pred_col not in curr.columns:
            try:
                model = pickle.load(model_file)
                X_for_model = curr.drop(columns=[c for c in [label_col, group_col] if c in curr.columns])
                if hasattr(model, "predict_proba"):
                    curr[pred_col] = (model.predict_proba(X_for_model)[:, 1] >= 0.5).astype(int)
                elif hasattr(model, "predict"):
                    curr[pred_col] = model.predict(X_for_model)
                else:
                    st.warning("Uploaded model does not have predict/predict_proba; skipping prediction.")
            except Exception as me:
                st.error(f"Failed to load/use model: {me}")

        # Common features (excluding pred/label/group)
        common_cols = list(set(base.columns).intersection(set(curr.columns)))
        feature_cols = [c for c in common_cols if c not in {pred_col, label_col, group_col}]

        # Store for explainability
        st.session_state["_feature_cols"] = feature_cols
        st.session_state["_X"] = curr[feature_cols].copy()
        st.session_state["_yhat"] = curr[pred_col] if pred_col in curr.columns else None
        st.session_state["_ytrue"] = curr[label_col] if label_col in curr.columns else None

        # Split into numeric/categorical
        num_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(base[c]) and pd.api.types.is_numeric_dtype(curr[c])]
        cat_feats = [c for c in feature_cols if c not in num_feats]

        # Numeric drift
        n_rows = []
        for f in sorted(num_feats):
            p = psi(base[f].astype(float).values, curr[f].astype(float).values)
            k = ks_stat(base[f].astype(float).values, curr[f].astype(float).values)
            n_rows.append({"feature": f, "psi": p, "ks": k})
        numeric_drift_df = pd.DataFrame(n_rows)

        # Categorical drift
        c_rows = []
        for f in sorted(cat_feats):
            p_counts, p_idx = categorical_dist(base[f].astype(str))
            q_counts, q_idx = categorical_dist(curr[f].astype(str))
            cats = sorted(set(p_idx) | set(q_idx))
            p_aligned = np.array([p_counts[p_idx.index(c)] if c in p_idx else 0 for c in cats], dtype=float)
            q_aligned = np.array([q_counts[q_idx.index(c)] if c in q_idx else 0 for c in cats], dtype=float)
            c_rows.append({
                "feature": f,
                "tvd": tvd(p_aligned, q_aligned),
                "js": jensen_shannon(p_aligned, q_aligned),
            })
        categorical_drift_df = pd.DataFrame(c_rows)

        # Fairness
        if all(c in curr.columns for c in [pred_col, label_col, group_col]):
            rates = group_rates(curr[label_col], curr[pred_col], curr[group_col], positive_label=positive_label)
            fair_df = fairness_summary(rates)
        else:
            st.info("Fairness metrics require current dataset with prediction, label, and sensitive attribute columns.")

    except Exception as e:
        st.error(f"Failed to compute metrics: {e}")

# =============================
# Display sections
# =============================

st.header("Results")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Drift — Numeric Features")
    if not numeric_drift_df.empty:
        def drift_severity(row):
            if pd.isna(row["psi"]):
                return "unknown"
            if row["psi"] >= psi_alert:
                return "high"
            if row["psi"] >= psi_warn:
                return "moderate"
            return "low"
        tmp = numeric_drift_df.copy()
        tmp["severity"] = tmp.apply(drift_severity, axis=1)
        st.dataframe(tmp.sort_values(["severity", "psi"], ascending=[False, False]), use_container_width=True)
    else:
        st.write("No numeric features or metrics available.")

with col2:
    st.subheader("Drift — Categorical Features")
    if not categorical_drift_df.empty:
        tmpc = categorical_drift_df.copy()
        tmpc["flag"] = (tmpc["tvd"].fillna(0) > tvd_flag) | (tmpc["js"].fillna(0) > js_flag)
        st.dataframe(tmpc.sort_values(["flag", "tvd"], ascending=[False, False]), use_container_width=True)
    else:
        st.write("No categorical features or metrics available.")

st.subheader("Fairness by Group")
if not fair_df.empty:
    st.dataframe(fair_df, use_container_width=True)
else:
    st.write("Fairness table not available.")

# =============================
# 🧠 Explainability (Global & Local)
# =============================
st.header("🧠 Explainability")

if "_X" in st.session_state and st.session_state.get("_X") is not None and st.session_state.get("_yhat") is not None:
    X = st.session_state["_X"].copy()
    feature_cols = st.session_state["_feature_cols"]
    yhat = st.session_state["_yhat"]

    # Drop rows with missing predictions and coerce to ints
    mask = ~pd.isna(yhat)
    X = X.loc[mask]
    y = pd.to_numeric(yhat.loc[mask], errors="coerce").dropna().astype(int)
    X = X.loc[y.index]

    # Need at least 2 classes
    if len(X) >= 2 and len(np.unique(y)) > 1:
        # Split feature types
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if c not in num_cols]

        # Preprocess: impute + scale numeric, impute + one-hot categorical
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))  # with_mean=False keeps sparse safety
                ]), num_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_cols),
            ],
            remainder="drop"
        )

        surrogate = Pipeline([
            ("pre", pre),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=0))
        ])

        with st.spinner("Training lightweight surrogate to mimic predictions…"):
            surrogate.fit(X, y)

        # ---------- Raw-feature permutation importance ----------
        st.markdown("#### Global importance (raw-feature permutation)")
        baseline_acc = accuracy_score(y, surrogate.predict(X))
        drops = []
        rng = np.random.default_rng(0)
        for col in X.columns:
            X_perm = X.copy()
            # independent permutation of the single column
            X_perm[col] = rng.permutation(X_perm[col].values)
            acc = accuracy_score(y, surrogate.predict(X_perm))
            drops.append((col, baseline_acc - acc))
        imp_df = pd.DataFrame(drops, columns=["feature", "accuracy_drop"]).sort_values("accuracy_drop", ascending=False)
        st.dataframe(imp_df, use_container_width=True)

        # ---------- PDP/ICE only for numeric features ----------
        st.markdown("#### Partial Dependence + ICE (numeric features only)")
        if len(num_cols) > 0:
            feat = st.selectbox("Feature for PDP/ICE", num_cols)
            fig, ax = plt.subplots()
            # PDP supports pipelines; pass the original (untransformed) feature name
            PartialDependenceDisplay.from_estimator(surrogate, X, [feat], kind="both", ax=ax)
            st.pyplot(fig)
        else:
            st.info("No numeric features available for PDP/ICE.")

        # ---------- Local SHAP on surrogate ----------
        st.markdown("#### Local explanation (SHAP on surrogate)")
        try:
            # SHAP works on the fitted pipeline; compute on a small sample
            sample_idx = np.random.choice(len(X), size=min(200, len(X)), replace=False)
            Xs = X.iloc[sample_idx]

            # Build a TreeExplainer on the underlying RF (after transform)
            # We transform Xs to the model's input space:
            Xs_trans = surrogate.named_steps["pre"].transform(Xs)
            rf = surrogate.named_steps["clf"]

            explainer = shap.TreeExplainer(rf)
            shap_vals = explainer.shap_values(Xs_trans)

            # expected_value can be list for binary classifiers
            base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            if isinstance(shap_vals, list):
                shap_arr = shap_vals[1]  # class 1 contributions
            else:
                shap_arr = shap_vals

            # We don’t have easy original-feature SHAP after one-hot; show magnitudes overall
            st.caption("Mean |SHAP| over transformed features (proxy magnitude).")
            mean_abs_shap = pd.DataFrame({
                "transformed_feature_index": np.arange(shap_arr.shape[1]),
                "mean_abs_shap": np.abs(shap_arr).mean(axis=0)
            }).sort_values("mean_abs_shap", ascending=False)
            st.dataframe(mean_abs_shap.head(30), use_container_width=True)

            # Waterfall for a single sample (transformed space)
            row_idx = st.number_input("Row index for waterfall (from sampled subset)", 0, len(Xs)-1, 0)
            try:
                shap.plots.waterfall(
                    shap.Explanation(values=shap_arr[row_idx], base_values=base_val,
                                     data=np.asarray(Xs_trans[row_idx]).ravel(), feature_names=[f"x{i}" for i in range(shap_arr.shape[1])]),
                    show=False
                )
                st.pyplot(plt.gcf())
            except Exception:
                st.info("Waterfall rendering not supported here; showing raw contributions.")
                st.write(pd.Series(shap_arr[row_idx]).sort_values(key=np.abs, ascending=False).head(30))
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")
    else:
        st.warning("Explainability skipped: need predictions for at least two classes after dropping missing values.")
else:
    st.info("Upload baseline & current CSVs with a prediction column to enable explainability.")

# =============================
# Sample schema & how-to
# =============================

st.header("📦 Sample data & schema")

st.markdown(
    textwrap.dedent(
        f"""
        **Expected columns (recommended)**

        - Feature columns: any name, numeric or categorical.
        - **{pred_col}** *(optional for drift, required for fairness & explainability)*: binary predictions (e.g., 0/1).
        - **{label_col}** *(optional for drift, required for fairness)*: true binary labels (0/1).
        - **{group_col}** *(optional for drift, required for fairness)*: sensitive attribute (e.g., gender, region).

        **Notes**
        - PSI thresholds (rule of thumb): < {psi_warn:.2f} small, {psi_warn:.2f}–{psi_alert:.2f} moderate, ≥ {psi_alert:.2f} high.
        - Fairness flags: DI < {di_flag:.2f}, |DP diff| > {dp_flag:.2f}, |EO diff| > {eopp_flag:.2f}, EOds > {eodds_flag:.2f}.
        - Categorical drift flags: TVD > {tvd_flag:.2f}, JS > {js_flag:.2f}.
        - Explainability will use your uploaded model if provided; otherwise a small surrogate model approximates decisions for global/local analysis.
        """
    )
)

# Synthetic samples to download
sample_baseline = pd.DataFrame({
    "age": np.random.normal(40, 10, 500).round(0),
    "income": np.random.lognormal(mean=10.5, sigma=0.5, size=500).round(0),
    "channel": np.random.choice(["web", "store", "phone"], size=500, p=[0.6, 0.3, 0.1]),
    "group": np.random.choice(["A", "B"], size=500, p=[0.5, 0.5]),
    "label": np.random.binomial(1, 0.35, size=500),
})
sample_baseline["prediction"] = (sample_baseline["age"] > 35).astype(int)

sample_current = sample_baseline.copy()
sample_current["age"] = np.random.normal(45, 12, 500).round(0)
sample_current["channel"] = np.random.choice(["web", "store", "phone"], size=500, p=[0.4, 0.4, 0.2])
sample_current["prediction"] = ((sample_current["age"] > 42) | (sample_current["channel"] == "store")).astype(int)

colA, colB = st.columns(2)
with colA:
    st.download_button("Download baseline_sample.csv", sample_baseline.to_csv(index=False).encode("utf-8"), "baseline_sample.csv", "text/csv")
with colB:
    st.download_button("Download current_sample.csv", sample_current.to_csv(index=False).encode("utf-8"), "current_sample.csv", "text/csv")

st.caption("Tip: Start with the samples, then swap in your own data with matching column names.")

# NOTE for requirements.txt:
# streamlit
# pandas
# numpy
# scipy
# scikit-learn
# shap
# matplotlib
