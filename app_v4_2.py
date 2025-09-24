# app.py — Streamlit app for Drift, Fairness & Explainability (with optional pickle model upload)

import io
import json
import pickle
import textwrap
import warnings
from typing import List, Tuple
# Add these to the top with your other sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

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
from streamlit_extras.app_logo import add_logo


add_logo("data:image/jpeg;base64,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")

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
st.header("📝 Auto-Generated Commentary")

# Drift comments
nt_drift, tech_drift = drift_comments(
    numeric_drift_df.fillna(0), categorical_drift_df.fillna(0)
)

# Fairness comments
nt_fair, tech_fair = fairness_comments(fair_df if not fair_df.empty else pd.DataFrame())

nt_block = (
    "**For non-technical readers**\n\n" +
    textwrap.dedent(f"""
    **Data Drift:**\n
    {nt_drift}

    **Fairness:**\n
    {nt_fair}
    """)
)

tech_block = (
    "**For technical readers**\n\n" +
    textwrap.dedent(f"""
    **Drift metrics:**\n
    {tech_drift}

    **Fairness metrics:**\n
    {tech_fair}
    """)
)

st.markdown(nt_block)
with st.expander("Show technical details"):
    st.markdown(tech_block)

# =============================
# Sample schema & how-to
# =============================

st.header("📦 Sample data & schema")

st.markdown(
    textwrap.dedent(
        f"""
        **Expected columns (recommended)**

        - Feature columns: any name, numeric or categorical.
        - **{pred_col}** *(optional for drift, required for fairness)*: binary predictions (e.g., 0/1).
        - **{label_col}** *(optional for drift, required for fairness)*: true binary labels (0/1).
        - **{group_col}** *(optional for drift, required for fairness)*: sensitive attribute (e.g., gender, region).

        **Notes**
        - PSI thresholds (rule of thumb): < {psi_warn:.2f} small, {psi_warn:.2f}–{psi_alert:.2f} moderate, ≥ {psi_alert:.2f} high.
        - Fairness flags: DI < {di_flag:.2f}, |DP diff| > {dp_flag:.2f}, |EO diff| > {eopp_flag:.2f}, EOds > {eodds_flag:.2f}.
        - Categorical drift flags: TVD > {tvd_flag:.2f}, JS > {js_flag:.2f}.
        """
    )
)
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
