import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection - Overview",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme CSS
st.markdown("""
    <style>
    html, body, [data-testid="stApp"] {
        color: #1f3a63;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1f3a63;
    }
    .stText, .stMarkdown, .stSubheader {
        color: #1f3a63 !important;
    }
    @media (prefers-color-scheme: dark) {
        html, body, [data-testid="stApp"] {
            color: #ffffff;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff;
        }
        .stText, .stMarkdown, .stSubheader {
            color: #ffffff !important;
        }
    }
    [data-theme="dark"] html, [data-theme="dark"] body, [data-theme="dark"] [data-testid="stApp"] {
        color: #ffffff;
    }
    [data-theme="dark"] h1, [data-theme="dark"] h2, [data-theme="dark"] h3, [data-theme="dark"] h4, [data-theme="dark"] h5, [data-theme="dark"] h6 {
        color: #ffffff;
    }
    [data-theme="dark"] .stText, [data-theme="dark"] .stMarkdown, [data-theme="dark"] .stSubheader {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Anomaly Detection in Credit Card Transactions")

# Section: Problem Statement
st.header("➤ Problem Statement")

st.markdown("""
With the rise of digital payments, credit card fraud has become a growing concern, leading to major financial losses for both businesses and consumers. Traditional fraud detection systems, which often depend on fixed rules and static thresholds, struggle to keep up with the ever-evolving tactics used by fraudsters. These fraudulent transactions are not only rare but also often subtle, making them hard to catch using conventional methods. As fraud patterns continue to change and grow more sophisticated, there’s a clear need for smarter, data-driven approaches that can adapt in real time and accurately identify suspicious activity before it causes harm.
""")

# Section: Why This Problem Needs to Be Solved
st.subheader("➤ Why This Problem Needs to Be Solved")

st.markdown("""
With the rapid growth of online payments and global digital transactions, credit card fraud has become a critical challenge for financial institutions and consumers alike. Even with the widespread use of security measures like EMV chips and two-factor authentication, fraudsters continue to find new ways to exploit weaknesses—particularly in behavior-based systems that struggle to keep up with evolving tactics.

To effectively tackle this issue, there’s a strong need for intelligent detection systems that can:
- Adapt to new and unseen fraud patterns without relying on manually defined rules
- Spot unusual behavior hidden within massive volumes of legitimate transaction data
- Deliver high accuracy while keeping false alarms to a minimum

Building such systems is essential for staying ahead of modern fraud schemes and ensuring the safety and trustworthiness of digital financial platforms.
""")

# Section: Existing Measures and Industry Practices
st.header("➤ Current Industry Measures for Fraud Detection")

st.markdown("""
To mitigate credit card fraud, the financial industry employs a combination of rule-based systems and machine learning models. These systems monitor transaction behavior and attempt to flag anomalies that might indicate fraud.

#### Common Practices in the Industry:
- **Static Rule-Based Systems**  
  Transactions are flagged based on predefined rules like unusually large amounts, foreign locations, or rapid successive transactions.  
  ⚠️ *Limitation: Easily bypassed as fraudsters adapt to known rules.*

- **Traditional Machine Learning Models**  
  Models like Logistic Regression, Decision Trees, and Random Forests are trained on labeled datasets to detect fraud.  
  ⚠️ *Limitation: Struggle with class imbalance and require extensive feature engineering.*

- **Statistical Thresholding**  
  Flags transactions that deviate significantly from a user’s typical behavior using [three-sigma rules](https://en.wikipedia.org/wiki/68–95–99.7_rule).  
  ⚠️ *Limitation: Assumes normal distribution, often leading to high false positive rates.*

Despite their use, these methods are often rigid and fail to adapt to subtle, evolving fraud behaviors.
""")

# Section: The Approach - LSTM Autoencoder
st.header("➤ The Approach: LSTM Autoencoder for Anomaly Detection")

st.markdown("""
To address these limitations, we propose an LSTM Autoencoder—an unsupervised deep learning model that detects anomalies by measuring reconstruction error.""")

st.image("graphs/approach.png", caption="Approach Plan", use_container_width =True)

st.markdown("""
#### Why It Works Better:
- **Unsupervised Learning**  
  Trained only on normal transactions, making it ideal for handling imbalanced datasets.

- **LSTM for Sequential Patterns**  
  Learns subtle behavioral patterns in transactions that may not be obvious in flat data.

- **Reconstruction-Based Detection**  
  Fraudulent inputs reconstruct poorly, resulting in high error that signals potential fraud.

- **Dropout Regularization**  
  Helps avoid overfitting and improves generalization on unseen transaction patterns.

This approach balances adaptability, performance, and interpretability for real-world fraud detection.
""")
