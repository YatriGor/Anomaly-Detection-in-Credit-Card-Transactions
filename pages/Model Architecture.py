import streamlit as st

# Page config
st.set_page_config(
    page_title="Model Architecture",
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
st.title("➤ Model Architecture")

# Section: Introduction to the Model
st.subheader("↳ Understanding the Components")

st.markdown("""
#### 🔹 Autoencoder
An **autoencoder** is a type of neural network that learns to compress input data into a lower-dimensional representation and then reconstruct it back to its original form. It's trained to minimize the reconstruction error between the input and output. This approach is particularly useful for anomaly detection, as the model is expected to reconstruct only "normal" data well.

#### 🔹 Encoder
The **encoder** compresses the input data into a smaller feature space using a series of linear transformations (dense layers), activation functions (ReLU), and regularization (dropout). It captures meaningful patterns by progressively reducing dimensionality.

#### 🔹 Bottleneck
The **bottleneck layer** is the narrowest part of the network — it holds the most compressed, information-rich representation of the input. It forces the model to learn the most relevant features and discard noise.

#### 🔹 LSTM (Long Short-Term Memory)
**LSTM** is a type of recurrent neural network (RNN) capable of learning long-range dependencies in sequential data. In our model, we use LSTM to enhance the bottleneck representation by modeling sequential behaviors and correlations that may exist in transaction data — even if it’s not strictly time series.

#### 🔹 Decoder
The **decoder** mirrors the encoder in reverse: it reconstructs the original input from the compressed bottleneck representation. This includes upsampling layers, ReLU activations, and dropout — finishing with a final linear layer that outputs the same shape as the original input.
""")

# Summary of how your model uses these components together
st.markdown("""
---

### 🔄 How Our Model Combines These Components

- The **encoder** compresses input transaction data into a compact representation.
- This is passed through a **bottleneck layer**, producing a dense, informative embedding.
- The resulting vector is treated as a sequence input to a **two-layer LSTM**, which is capable of learning dependencies over time or positional ordering.
- We specifically use the **final hidden state (`h_T`)** — often referred to as the **context vector** — from the LSTM, as it captures the most comprehensive summary of the entire input sequence.
- This context vector is then passed to the **decoder**, which attempts to reconstruct the original input.
- The difference between the input and reconstructed output gives us the **reconstruction error**, which is used as the **anomaly score**.

This combination allows the model to generalize well on normal transactions and flag unusual ones with high sensitivity.
""")

# Section: Model Definition Code
st.subheader("↳ LSTM Autoencoder Code")

st.code("""
class AutoencoderWithLSTM(nn.Module):
    def __init__(self, input_dim, features):
        super(AutoencoderWithLSTM, self).__init__()

        encoding_dim = features[-1] * 2

        self.encoder = nn.Sequential()
        previous_dim = input_dim
        for feature in features:
            self.encoder.add_module(f"Downsample_{feature}",nn.Linear(previous_dim, feature))
            self.encoder.add_module("ReLU", nn.ReLU())
            self.encoder.add_module("Dropout", nn.Dropout(0.01))
            previous_dim = feature

        self.bottleneck = nn.Linear(previous_dim, encoding_dim)

        self.lstm = nn.LSTM(input_size=encoding_dim, hidden_size=encoding_dim, num_layers=2, batch_first=True)

        self.decoder = nn.Sequential()
        previous_dim = encoding_dim
        for feature in reversed(features):
            self.decoder.add_module(f"Upsample_{feature}",nn.Linear(previous_dim, feature))
            self.decoder.add_module("ReLU", nn.ReLU())
            self.decoder.add_module("Dropout", nn.Dropout(0.01))
            previous_dim = feature

        self.decoder.add_module("output", nn.Linear(previous_dim, input_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = self.bottleneck(encoded)

        encoded = encoded.unsqueeze(1)
        lstm_out, (hn, cn) = self.lstm(encoded)
        lstm_out = lstm_out[:, -1, :]

        decoded = self.decoder(lstm_out)
        return decoded
""", language="python")

st.image("graphs/model.png", caption="Model Architecture", use_container_width =True)

# Section: Loss Function and Optimizer
st.subheader("↳ Loss Function and Optimizer")

st.markdown("""
To train our LSTM Autoencoder, we use the **Adam optimizer** and **L1 loss (mean absolute error)** as the reconstruction loss function.

#### 🔹 Why Adam Optimizer?
The **Adam (Adaptive Moment Estimation)** optimizer combines the benefits of RMSProp and Momentum. It adjusts learning rates adaptively for each parameter, making it robust and efficient for sparse, noisy, or high-dimensional data — which fits our fraud detection setting well.

- Handles noisy gradients and non-stationary objectives
- Works well with large datasets and deep networks
- Requires minimal hyperparameter tuning

#### 🔹 Why L1 Loss?
We use **L1 Loss (Mean Absolute Error)** over the more common MSE (L2 Loss) because:

- It is **less sensitive to outliers** — which is critical in fraud detection, where extreme fraud values might skew the learning process.
- It produces **sparser gradients**, encouraging robustness and simpler reconstructions.
- It works well when the data contains subtle deviations, which is often the case in anomaly detection problems.

### Final Setup:
```python
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.L1Loss(reduction='sum')

def reconstruction_loss(recon_x, x):
    return criterion(recon_x, x)
""")

# Section: Training Overview
st.subheader("↳ Training Process & Metrics")

st.markdown("""
Once the model architecture is defined and the loss/optimizer is configured, we train the model using the **training set** and validate it on a **separate validation set**.

The training continues for a fixed number of epochs or until the **early stopping criterion** (based on validation loss) is met — whichever comes first.

The loss curves below provide a clear picture of the model's learning behavior.
""")

# Display training loss image
st.image("graphs/training_metrics.png", caption="Training and Validation Loss Across Epochs", use_container_width=True)

# Explanation
st.markdown("""
The graph shows:
- A steep decline in both training and validation loss in the early epochs
- A stable, minimal gap between the two curves after ~10 epochs
- No signs of overfitting, which confirms the model generalizes well

This kind of loss pattern is desirable and shows that the model is effectively learning without memorizing. Once the **patience level** is reached (i.e., no improvement for N epochs), the training is stopped and the best model is saved.
""")
