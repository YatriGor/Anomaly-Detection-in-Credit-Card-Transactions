import streamlit as st

# Page config
st.set_page_config(
    page_title="Results and Conclusion",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theme CSS
st.markdown("""
    <style>
    html, body, [data-testid="stApp"] {
        color: #1f3a63;
    }
    h1, h2, h3, h4 {
        color: #1f3a63;
    }
    .stText, .stMarkdown {
        color: #1f3a63 !important;
    }
    @media (prefers-color-scheme: dark) {
        html, body, [data-testid="stApp"] {
            color: #ffffff;
        }
        h1, h2, h3, h4 {
            color: #ffffff;
        }
        .stText, .stMarkdown {
            color: #ffffff !important;
        }
    }
    [data-theme="dark"] html, [data-theme="dark"] body, [data-theme="dark"] [data-testid="stApp"] {
        color: #ffffff;
    }
    [data-theme="dark"] h1, [data-theme="dark"] h2, [data-theme="dark"] h3, [data-theme="dark"] h4 {
        color: #ffffff;
    }
    [data-theme="dark"] .stText, [data-theme="dark"] .stMarkdown {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("➤ Results and Conclusion")

# Section: Confusion Matrix
st.subheader("↳ Confusion Matrix")
st.image("graphs/confusion_matrix.png", use_container_width=True)
st.markdown("""
The confusion matrix provides a clear snapshot of how well our model is working in the real world. Out of tens of thousands of transactions, it was able to correctly identify 56,479 legitimate transactions and 372 fraudulent ones — meaning it accurately protected both genuine customers and the company from risk.
Only 93 fraud cases were missed, which, while not perfect, is still a very low number given how rare fraud is (less than 0.2%). And just 25 legitimate transactions were wrongly flagged as fraud, which is impressively low and crucial for ensuring a smooth customer experience.
""")

# Section: Performance Metrics
st.subheader("↳ Performance Metrics")
st.image("graphs/performance_matrix.png", use_container_width=True)
st.markdown("""
- **Accuracy**: 0.9979  
- **Precision**: 0.937  
- **Recall**: 0.800  
- **F1-Score**: 0.8631  

The model achieves an impressive accuracy of 99.79%, but in fraud detection, accuracy alone isn’t enough — because fraud cases are extremely rare. That’s where precision and recall become critical.
\nA precision of 93.7% means that when the model flags a transaction as fraudulent, it’s correct almost every time — minimizing false alarms and ensuring that real customers aren’t blocked unnecessarily.
\nA recall of 80% means the model successfully catches 8 out of every 10 fraud attempts, helping prevent major financial losses.
\nThe F1-score of 86.31% shows a strong balance between these two goals: being cautious without crying wolf, and being thorough without overreacting.
This balance is essential in real-world fraud detection systems, where catching fraud is important — but not at the cost of annoying loyal customers.""")

# Section: ROC Curve
st.subheader("↳ ROC Curve & AUC")
st.image("graphs/roc_curve.png", use_container_width=True)
st.markdown("""
The **Receiver Operating Characteristic (ROC) curve** demonstrates the trade-off between **true positive rate** and **false positive rate**.  
The **Area Under Curve (AUC)** value of **0.8998** indicates excellent discrimination capability between normal and fraudulent transactions.
""")

# Section: Conclusion
st.header("➤ Conclusion")

st.markdown("""
This project presents a reliable and scalable fraud detection system that learns from normal transaction behavior to identify anomalies with high precision and recall. By combining deep feature extraction with sequence learning through LSTM, the model achieves strong performance on highly imbalanced data — making it well-suited for real-world financial applications where early and accurate fraud detection is critical.
### Key Achievements:
- Built and trained an effective LSTM Autoencoder to learn normal transaction patterns
- Used reconstruction error to identify fraudulent transactions with high precision
- Achieved a **balanced performance** across accuracy, recall, and F1-score
- Applied thoughtful **feature engineering** (log transform, cyclical time encoding)
- Designed and evaluated a robust architecture suitable for real-world deployment

""")
