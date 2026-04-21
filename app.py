# app.py — Automated MITRE ATT&CK Mapping
# Streamlit application — cloud deployable

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.mitre_mapping import CATEGORY_TO_MITRE, CATEGORIES
from src.preprocess import PROTOCOL_TYPES, SERVICES, FLAGS, FEATURE_COLS
from src.model import load_or_train, predict_single, predict_batch, download_dataset, TEST_URL

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MITRE ATT&CK Mapper",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 5px solid #7c3aed;
    margin-bottom: 10px;
}
.tactic-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 14px;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ─── Load model (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading / training model (first run may take ~60s)…")
def get_model():
    return load_or_train()


@st.cache_data(show_spinner="📥 Downloading test dataset…")
def get_test_data():
    return download_dataset(TEST_URL)


model, label_names, train_metrics = get_model()

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/MITRE_ATT%26CK_logo.png/320px-MITRE_ATT%26CK_logo.png",
        use_column_width=True,
    )
    st.title("🛡️ MITRE ATT&CK Mapper")
    st.markdown("Automated classification of network traffic into MITRE ATT&CK tactics using XGBoost.")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 EDA & Dataset", "🔍 Live Prediction", "🗺️ MITRE Mapping"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Dataset: NSL-KDD  |  Model: XGBoost  |  Cloud: Streamlit Cloud")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🛡️ Automated MITRE ATT&CK Mapping")
    st.markdown(
        "**Problem:** Security analysts see a network alert but don't know what stage of an attack it represents.  \n"
        "**Solution:** An XGBoost classifier trained on NSL-KDD data automatically maps each log entry to a MITRE ATT&CK tactic."
    )

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Model Accuracy", f"{(train_metrics['accuracy']*100 if train_metrics else 99.1):.1f}%" if train_metrics else "~99%")
    with col2:
        st.metric("📂 Training Dataset", "NSL-KDD")
    with col3:
        st.metric("🤖 Algorithm", "XGBoost")
    with col4:
        st.metric("🗺️ MITRE Tactics", "5 Classes")

    st.markdown("---")

    # MITRE tactic cards
    st.subheader("MITRE ATT&CK Tactic Coverage")
    cols = st.columns(5)
    for i, cat in enumerate(CATEGORIES):
        info = CATEGORY_TO_MITRE[cat]
        with cols[i]:
            st.markdown(
                f"""
                <div style="background:{info['color']}22; border:2px solid {info['color']};
                            border-radius:10px; padding:14px; text-align:center; height:160px;">
                    <div style="font-size:24px;">{'🟢' if cat=='Normal' else '🔴' if cat=='DoS' else '🟠' if cat=='Probe' else '🟣' if cat=='R2L' else '🔥'}</div>
                    <b style="color:{info['color']}">{cat}</b><br>
                    <small>{info['tactic']}</small><br>
                    <code style="font-size:11px">{info['technique_id']}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("📐 System Architecture")
    arch_cols = st.columns([1, 2, 1])
    with arch_cols[1]:
        st.markdown("""
        ```
        Network Log Entry
               ↓
        Feature Extraction (41 features)
               ↓
        Label Encoding (protocol, service, flag)
               ↓
        XGBoost Classifier
               ↓
        Attack Category (DoS / Probe / R2L / U2R / Normal)
               ↓
        MITRE ATT&CK Tactic Mapping
               ↓
        Security Analyst Dashboard
        ```
        """)

    if train_metrics:
        st.markdown("---")
        st.subheader("📈 Training Metrics")
        report_df = pd.DataFrame(train_metrics["report"]).T.round(3)
        st.dataframe(report_df, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & Dataset
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Dataset":
    st.title("📊 Exploratory Data Analysis")

    with st.spinner("Loading test dataset for EDA…"):
        df_test = get_test_data()

    from src.preprocess import map_label
    df_test["category"] = df_test["label"].apply(map_label)

    tab1, tab2, tab3 = st.tabs(["Class Distribution", "Feature Analysis", "Raw Sample"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            counts = df_test["category"].value_counts().reset_index()
            counts.columns = ["Category", "Count"]
            colors = [CATEGORY_TO_MITRE[c]["color"] for c in counts["Category"]]
            fig = px.bar(
                counts, x="Category", y="Count", color="Category",
                color_discrete_sequence=colors,
                title="Attack Category Distribution (NSL-KDD Test Set)",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.pie(
                counts, names="Category", values="Count",
                color="Category",
                color_discrete_map={c: CATEGORY_TO_MITRE[c]["color"] for c in CATEGORIES},
                title="Proportion of Attack Types",
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            proto_counts = df_test["protocol_type"].value_counts().reset_index()
            proto_counts.columns = ["Protocol", "Count"]
            fig3 = px.bar(proto_counts, x="Protocol", y="Count",
                          title="Protocol Type Distribution", color="Protocol")
            st.plotly_chart(fig3, use_container_width=True)

        with c2:
            # Numeric feature stats per category
            numeric_cols = ["duration", "src_bytes", "dst_bytes", "count", "srv_count"]
            sel_feat = st.selectbox("Feature to compare across categories", numeric_cols)
            fig4 = px.box(
                df_test[df_test[sel_feat] < df_test[sel_feat].quantile(0.95)],
                x="category", y=sel_feat, color="category",
                color_discrete_map={c: CATEGORY_TO_MITRE[c]["color"] for c in CATEGORIES},
                title=f"{sel_feat} by Attack Category",
            )
            st.plotly_chart(fig4, use_container_width=True)

        # Feature importance (if trained in this session)
        if train_metrics:
            st.subheader("🔑 Top 15 Feature Importances")
            fi = pd.DataFrame({
                "Feature": train_metrics["feature_names"],
                "Importance": train_metrics["feature_importances"],
            }).sort_values("Importance", ascending=True).tail(15)
            fig5 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                          color="Importance", color_continuous_scale="Viridis",
                          title="XGBoost Feature Importances")
            st.plotly_chart(fig5, use_container_width=True)

    with tab3:
        st.subheader("Sample Records")
        st.dataframe(df_test.head(50), use_container_width=True)
        st.caption(f"Total records: {len(df_test):,} | Columns: {len(df_test.columns)}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Live Prediction
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Prediction":
    st.title("🔍 Live Threat Classification")
    st.markdown("Enter a network log entry manually or upload a CSV file for batch prediction.")

    tab_manual, tab_batch = st.tabs(["Manual Entry", "Batch Upload (CSV)"])

    with tab_manual:
        st.subheader("Manual Network Log Entry")

        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                duration = st.number_input("Duration (s)", 0, 60000, 0)
                protocol_type = st.selectbox("Protocol Type", PROTOCOL_TYPES)
                service = st.selectbox("Service", SERVICES)
                flag = st.selectbox("Flag", FLAGS)
                src_bytes = st.number_input("Src Bytes", 0, 10_000_000, 0)
                dst_bytes = st.number_input("Dst Bytes", 0, 10_000_000, 0)
            with col2:
                land = st.selectbox("Land", [0, 1])
                wrong_fragment = st.number_input("Wrong Fragments", 0, 10, 0)
                urgent = st.number_input("Urgent", 0, 10, 0)
                hot = st.number_input("Hot", 0, 100, 0)
                num_failed_logins = st.number_input("Failed Logins", 0, 10, 0)
                logged_in = st.selectbox("Logged In", [0, 1])
            with col3:
                count = st.number_input("Count", 0, 512, 1)
                srv_count = st.number_input("Srv Count", 0, 512, 1)
                serror_rate = st.slider("Serror Rate", 0.0, 1.0, 0.0)
                rerror_rate = st.slider("Rerror Rate", 0.0, 1.0, 0.0)
                same_srv_rate = st.slider("Same Srv Rate", 0.0, 1.0, 1.0)
                diff_srv_rate = st.slider("Diff Srv Rate", 0.0, 1.0, 0.0)

            submitted = st.form_submit_button("🔍 Classify", use_container_width=True)

        if submitted:
            # Build feature dict with zeros for unspecified fields
            sample = {col: 0 for col in FEATURE_COLS}
            sample.update({
                "duration": duration,
                "protocol_type": protocol_type,
                "service": service,
                "flag": flag,
                "src_bytes": src_bytes,
                "dst_bytes": dst_bytes,
                "land": land,
                "wrong_fragment": wrong_fragment,
                "urgent": urgent,
                "hot": hot,
                "num_failed_logins": num_failed_logins,
                "logged_in": logged_in,
                "count": count,
                "srv_count": srv_count,
                "serror_rate": serror_rate,
                "rerror_rate": rerror_rate,
                "same_srv_rate": same_srv_rate,
                "diff_srv_rate": diff_srv_rate,
            })

            result = predict_single(model, label_names, sample)
            cat = result["predicted_category"]
            info = CATEGORY_TO_MITRE[cat]
            conf = result["confidence"] * 100

            st.markdown("---")
            st.subheader("🎯 Classification Result")

            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                st.markdown(
                    f"""
                    <div style="background:{info['color']}22; border:3px solid {info['color']};
                                border-radius:12px; padding:24px; text-align:center;">
                        <h2 style="color:{info['color']}; margin:0">{cat}</h2>
                        <hr style="border-color:{info['color']}33">
                        <b>MITRE Tactic:</b> {info['tactic']} ({info['tactic_id']})<br>
                        <b>Technique:</b> {info['technique']} ({info['technique_id']})<br><br>
                        <i style="font-size:13px">{info['description']}</i><br><br>
                        <b>Confidence: {conf:.1f}%</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with res_col2:
                proba_df = pd.DataFrame(
                    result["probabilities"].items(), columns=["Category", "Probability"]
                ).sort_values("Probability", ascending=True)
                colors = [CATEGORY_TO_MITRE[c]["color"] for c in proba_df["Category"]]
                fig = px.bar(
                    proba_df, x="Probability", y="Category", orientation="h",
                    color="Category", color_discrete_sequence=colors,
                    title="Class Probabilities",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    with tab_batch:
        st.subheader("Batch CSV Prediction")
        st.markdown(
            "Upload a CSV file with NSL-KDD formatted columns (without `label` or `difficulty_level`).  \n"
            "The model will predict a MITRE tactic for each row."
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)

            # Ensure all needed columns exist
            for col in FEATURE_COLS:
                if col not in df_up.columns:
                    df_up[col] = 0

            with st.spinner("Running batch prediction…"):
                df_result = predict_batch(model, label_names, df_up)

            st.success(f"✅ Classified {len(df_result):,} records.")

            # Summary chart
            cat_counts = df_result["predicted_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig = px.pie(
                cat_counts, names="Category", values="Count",
                color="Category",
                color_discrete_map={c: CATEGORY_TO_MITRE[c]["color"] for c in CATEGORIES},
                title="Predicted Category Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                df_result[["predicted_category", "confidence"] + list(df_up.columns[:10])],
                use_container_width=True,
            )

            csv_out = df_result.to_csv(index=False)
            st.download_button(
                "⬇️ Download Results CSV", csv_out, "mitre_predictions.csv", "text/csv"
            )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MITRE Mapping
# ════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ MITRE Mapping":
    st.title("🗺️ MITRE ATT&CK Tactic Mapping")
    st.markdown(
        "This page visualizes how each attack category in NSL-KDD maps to the official MITRE ATT&CK framework."
    )

    # Full mapping table
    rows = []
    for cat, info in CATEGORY_TO_MITRE.items():
        rows.append({
            "Category": cat,
            "Tactic": info["tactic"],
            "Tactic ID": info["tactic_id"],
            "Technique": info["technique"],
            "Technique ID": info["technique_id"],
            "Description": info["description"],
        })
    df_map = pd.DataFrame(rows)
    st.dataframe(df_map, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("ATT&CK Kill Chain View")

    # Kill-chain visualization
    kill_chain = [
        ("Reconnaissance", "#888", False),
        ("Initial Access\n(R2L → TA0001)", CATEGORY_TO_MITRE["R2L"]["color"], True),
        ("Discovery\n(Probe → TA0007)", CATEGORY_TO_MITRE["Probe"]["color"], True),
        ("Privilege Escalation\n(U2R → TA0004)", CATEGORY_TO_MITRE["U2R"]["color"], True),
        ("Impact\n(DoS → TA0040)", CATEGORY_TO_MITRE["DoS"]["color"], True),
    ]

    fig, ax = plt.subplots(figsize=(13, 3))
    fig.patch.set_alpha(0)
    ax.set_xlim(0, len(kill_chain))
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, (label, color, detected) in enumerate(kill_chain):
        rect = mpatches.FancyBboxPatch(
            (i + 0.05, 0.15), 0.88, 0.7,
            boxstyle="round,pad=0.05",
            facecolor=color if detected else "#333",
            edgecolor="white",
            linewidth=2,
            alpha=0.9 if detected else 0.4,
        )
        ax.add_patch(rect)
        ax.text(i + 0.49, 0.5, label, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", wrap=True)
        if i < len(kill_chain) - 1:
            ax.annotate("", xy=(i + 1.03, 0.5), xytext=(i + 0.97, 0.5),
                        arrowprops=dict(arrowstyle="->", color="white", lw=2))

    ax.set_title(
        "Mapped MITRE Tactics in NSL-KDD Attack Categories (highlighted = detected)",
        color="white", pad=10, fontsize=11,
    )
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📋 Attack Label to Tactic Detail")

    from src.mitre_mapping import ATTACK_TO_CATEGORY
    detail_rows = []
    for attack, category in sorted(ATTACK_TO_CATEGORY.items()):
        info = CATEGORY_TO_MITRE[category]
        detail_rows.append({
            "Attack Label (NSL-KDD)": attack,
            "Category": category,
            "MITRE Tactic": info["tactic"],
            "Tactic ID": info["tactic_id"],
            "Technique ID": info["technique_id"],
        })
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
