#app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from chatbot import get_response

st.set_page_config(page_title="Academic NLP Chatbot", page_icon="🤖", layout="wide")

# ── Colour palette (consistent across all charts) ──────────────────────────
COLORS = {
    "High":   "#4CAF50",   # green
    "Medium": "#FFC107",   # amber
    "Low":    "#FF9800",   # orange
    "None":   "#F44336",   # red
    "accent": "#4C9BE8",
}

# ---------- Session state ----------
if "messages"    not in st.session_state:
    st.session_state.messages    = []
if "results_log" not in st.session_state:
    st.session_state.results_log = []

# ---------- Tabs ----------
chat_tab, analysis_tab = st.tabs(["💬 Chat", "📊 Result Analysis"])

# ══════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════
with chat_tab:
    st.title("Academic NLP Chatbot 🤖")

    # Container holds all messages — sits above the fixed input box
    chat_container = st.container()

    # chat_input is rendered here (Streamlit pins it to the bottom automatically)
    user_input = st.chat_input("Ask a question about NLP or Machine Learning…")

    if user_input:
        st.session_state.messages.append(("user", user_input))

        result = get_response(user_input)
        st.session_state.messages.append(("assistant", result["answer"]))

        # Only log NLP-processed queries
        if result["matched_question"] != "—":
            st.session_state.results_log.append({
                "Query":            user_input,
                "Matched Question": result["matched_question"],
                "Score":            round(result["score"], 4),
                "Confidence":       result["confidence"],
                "Answered":         result["answered"],
            })

    # Render all messages inside the container (above input)
    with chat_container:
        for i, (role, message) in enumerate(st.session_state.messages):
            st.chat_message(role).write(message)

            # Show top-3 expander after the last assistant message that has top_matches
            if (
                role == "assistant"
                and i == len(st.session_state.messages) - 1
                and user_input
                and len(result.get("top_matches", [])) > 1
            ):
                with st.expander("🔍 See top matching questions from dataset"):
                    for j, m in enumerate(result["top_matches"], 1):
                        bar_pct = int(m["score"] * 100)
                        st.markdown(
                            f"**{j}.** {m['question']}  \n"
                            f"`Score: {m['score']:.4f}` {'█' * (bar_pct // 5)}{'░' * (20 - bar_pct // 5)} {bar_pct}%"
                        )

# ══════════════════════════════════════════════
# TAB 2 — RESULT ANALYSIS
# ══════════════════════════════════════════════
with analysis_tab:
    st.title("Result Analysis 📊")

    if not st.session_state.results_log:
        st.info("No NLP queries yet — ask some questions in the Chat tab first!")
        st.stop()

    df    = pd.DataFrame(st.session_state.results_log)
    total = len(df)

    # ── KPI row ─────────────────────────────────────────────────────────────
    answered    = int(df["Answered"].sum())
    unanswered  = total - answered
    avg_score   = df["Score"].mean()
    high_count  = int((df["Confidence"] == "High").sum())
    med_count   = int((df["Confidence"] == "Medium").sum())
    low_count   = int((df["Confidence"] == "Low").sum())
    none_count  = int((df["Confidence"] == "None").sum())

    # Accuracy = (High + Medium) / total  — partial matches count
    accuracy_pct = (high_count + med_count) / total * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Queries",     total)
    k2.metric("Answered",          f"{answered} / {total}")
    k3.metric("Avg Match Score",   f"{avg_score:.3f}")
    k4.metric("Accuracy (H+M)",    f"{accuracy_pct:.0f}%")
    k5.metric("High Confidence",   f"{high_count} / {total}")

    st.divider()

    # ── Row 1: Score Distribution + Confidence Breakdown ────────────────────
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.subheader("Score Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        n, bins, patches = ax.hist(df["Score"], bins=12, edgecolor="white", color=COLORS["accent"])

        # Colour each bar by the threshold zone it falls in
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge >= 0.55:
                patch.set_facecolor(COLORS["High"])
            elif left_edge >= 0.30:
                patch.set_facecolor(COLORS["Medium"])
            elif left_edge >= 0.10:
                patch.set_facecolor(COLORS["Low"])
            else:
                patch.set_facecolor(COLORS["None"])

        ax.axvline(0.55, color=COLORS["High"],   linestyle="--", linewidth=1.2, label="High ≥ 0.55")
        ax.axvline(0.30, color=COLORS["Medium"], linestyle="--", linewidth=1.2, label="Medium ≥ 0.30")
        ax.axvline(0.10, color=COLORS["Low"],    linestyle="--", linewidth=1.2, label="Low ≥ 0.10")
        ax.set_xlabel("Cosine Similarity Score")
        ax.set_ylabel("Queries")
        ax.set_title("Score Distribution by Confidence Zone")
        ax.legend(fontsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with r1c2:
        st.subheader("Confidence Tier Breakdown")
        tiers  = ["High", "Medium", "Low", "None"]
        counts = [high_count, med_count, low_count, none_count]
        clrs   = [COLORS[t] for t in tiers]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(tiers, counts, color=clrs, edgecolor="white", width=0.5)
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.08,
                        str(cnt), ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.set_ylabel("Number of Queries")
        ax.set_title("Queries per Confidence Tier")
        ax.set_ylim(0, max(counts) * 1.3 + 1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Row 2: Per-query timeline + Cumulative accuracy ─────────────────────
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.subheader("Match Score per Query")
        fig, ax = plt.subplots(figsize=(max(5, total * 0.65), 3.5))
        bar_colors = [COLORS[c] for c in df["Confidence"]]
        ax.bar(range(total), df["Score"], color=bar_colors, edgecolor="white")
        ax.axhline(0.55, color=COLORS["High"],   linestyle="--", linewidth=1, label="High (0.55)")
        ax.axhline(0.30, color=COLORS["Medium"], linestyle="--", linewidth=1, label="Medium (0.30)")
        ax.axhline(0.10, color=COLORS["Low"],    linestyle="--", linewidth=1, label="Low (0.10)")
        ax.set_xticks(range(total))
        ax.set_xticklabels([f"Q{i+1}" for i in range(total)], rotation=45, fontsize=8)
        ax.set_ylabel("Cosine Similarity Score")
        ax.set_ylim(0, 1.05)
        ax.set_title("Per-Query Confidence  (🟢 High  🟡 Med  🟠 Low  🔴 None)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with r2c2:
        st.subheader("Cumulative Accuracy Over Time")
        # Accuracy = running % of High+Medium queries
        is_accurate = ((df["Confidence"] == "High") | (df["Confidence"] == "Medium")).astype(int)
        cumulative  = is_accurate.cumsum() / (np.arange(total) + 1) * 100

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(range(1, total + 1), cumulative, marker="o", markersize=5,
                color=COLORS["accent"], linewidth=2, label="Cumulative accuracy")
        ax.axhline(100, color=COLORS["High"],   linestyle=":", linewidth=1)
        ax.axhline(50,  color=COLORS["Medium"], linestyle=":", linewidth=1)
        ax.set_xlabel("Query Number")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 110)
        ax.set_title("Running Accuracy (High + Medium %)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Confusion summary ────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Confidence × Answered Summary")

    summary = (
        df.groupby(["Confidence", "Answered"])
          .size()
          .reset_index(name="Count")
    )
    summary["Answered"] = summary["Answered"].map({True: "✅ Yes", False: "❌ No"})
    summary["Confidence"] = pd.Categorical(
        summary["Confidence"], categories=["High", "Medium", "Low", "None"], ordered=True
    )
    summary = summary.sort_values("Confidence")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── Detailed results table ───────────────────────────────────────────────
    st.divider()
    st.subheader("Detailed Results Table")

    def _color_confidence(val):
        palette = {
            "High":   "background-color:#d4edda; color:#155724",
            "Medium": "background-color:#fff3cd; color:#856404",
            "Low":    "background-color:#ffe5cc; color:#7d3c00",
            "None":   "background-color:#f8d7da; color:#721c24",
        }
        return palette.get(val, "")

    styled = (
        df.style
          .applymap(_color_confidence, subset=["Confidence"])
          .background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=1)
          .format({"Score": "{:.4f}", "Answered": lambda x: "✅" if x else "❌"})
    )
    st.dataframe(styled, use_container_width=True)

    # ── Export ───────────────────────────────────────────────────────────────
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results as CSV", csv, "nlp_results.csv", "text/csv")
