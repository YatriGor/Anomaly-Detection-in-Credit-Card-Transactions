import streamlit as st

# Page config
st.set_page_config(
    page_title="Data Exploration & Preparation",
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

# Dataset page title
st.title("➤ Dataset Description")

# Data description content
st.markdown("""
This project uses the **Credit Card Transaction Fraud Detection** dataset from Kaggle. The dataset consists of **31 anonymized features**, generated using **Principal Component Analysis (PCA)**, intended to protect confidentiality while preserving predictive power.

Key characteristics:
- All features are anonymized PCA components (V1 to V28), plus `Time`, `Amount`, and `Class`.
- `Class` label: **0 = legitimate**, **1 = fraudulent**.
- Transactions are **extremely imbalanced** — with only **492 frauds out of ~284,807 records** (≈ 0.17% fraud rate).
            
🔗 [Click here to view the dataset on Kaggle](https://www.kaggle.com/datasets/rudrakshsivamdutta/credit-card-transaction-fraud-detection-datafilter)
""")

st.header("➤ Amount Distribution Analysis")

st.subheader("↳ Original Distribution of Amount")

st.markdown("""
Before applying any transformations, it is essential to first examine the distribution of the Amount feature. Many machine learning algorithms perform optimally when input features follow a normal distribution. However, transaction amounts in real-world datasets are often highly variable and tend to be right-skewed, reflecting a few large transactions among many smaller ones. Understanding this skewness is crucial for selecting appropriate preprocessing techniques and ensuring the effectiveness of the model.
""")

st.code("""
plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'], bins=50, color='red', kde=True)
plt.title("Distribution of Amount")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
""", language='python')

st.image("graphs/before_preprocessing.png", caption="Distribution of Transaction Amounts", use_container_width =True)

st.markdown("""
As we can see from the distribution, the `Amount` feature is **heavily right-skewed**, Most transactions involve smaller amounts, while a few high-value transactions stretch the scale significantly.
This kind of skewed distribution can lead to a few problems during model training:
            
- Algorithms may give too much importance to large transaction values
- The wide range in scale can make it harder for the model to learn general patterns
- And overall, it can reduce the model’s ability to perform well on typical, everyday transactions
""")

# Section: Log Transformation of Amount
st.subheader("↳ Log Transformation on Amount")

st.markdown("""
To reduce skewness and bring the `Amount` feature closer to a normal distribution, we apply a **log transformation** using `np.log1p()`. 
This technique compresses the range of large values while preserving the order and relative differences among smaller amounts.
""")

# Show code block
st.code("""
df['Amount'] = np.log1p(df['Amount'])
plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'], bins=50, color='blue', kde=True)
plt.title("Distribution of Log Transformed Amount")
plt.xlabel("Log(1 + Amount)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
""", language='python')

# Show the output image after log transformation
st.image("graphs/after_preprocessing.png", caption="Distribution After Log Transformation", use_container_width =True)

# Add explanation
st.markdown("""
As shown in the updated distribution, the `Amount` feature is now **much more balanced and symmetrical**, 
making it better suited for modeling. The log transformation helps:
- Reduce the impact of outliers
- Improve convergence and stability in neural networks
- Enhance performance of models sensitive to scale or normality

This preprocessing step is especially important when working with **reconstruction-based models** like autoencoders.
""")

# Section: Feature Engineering
st.header("➤ Feature Engineering")

# Subsection: Encoding Transaction Time as Cyclical Features
st.subheader("↳ Time Feature Transformation (Hour → sin/cos)")

st.markdown("""
The original dataset includes a `Time` feature, which represents the number of seconds that have passed since the first recorded transaction. While this is useful, it doesn't directly tell us when during the day a transaction occurred.
To make this more meaningful, we extract the **hour of the day**  from the time information. This helps us understand patterns in user behavior — for example, whether a transaction happened in the morning, afternoon, or late at night.

But there's a catch — time is cyclical. Hour 0 (midnight) and hour 23 (11 PM) are actually very close in time, but numerically they look far apart. To solve this, we use sine and cosine transformations to convert the hour into cyclical features. This ensures the model understands that time wraps around smoothly over a 24-hour period, without any sharp jumps.
""")

# Code Block
st.code("""
df['Hour'] = (df['Time'] % 86400) / 3600

df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

df.drop(columns=['Time', 'Hour'], inplace=True)
""", language="python")

# Explanation
st.markdown("""
This transformation ensures that:
- **Similar hours have similar values** in the new features
- The model understands that 23:00 and 00:00 are temporally close
- Time-based cyclic patterns (like late-night vs. daytime spending) can be learned effectively

By dropping the original `Time` and intermediate `Hour` columns, we retain only the useful, normalized features: `Hour_sin` and `Hour_cos`.
""")