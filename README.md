# APS 2

## 1. Dataset Description:
The dataset contains 15,000 recipes from TudoGostoso website. The recipes include ingredient lists in Portuguese text format. The data originates from web scraping processes. The dataset serves as input for the embedding generation system.

## 2. Embedding Generation Process:

```mermaid
flowchart LR
    A[Recipe Text] --> B[SBERT Encoder]
    B --> C[768-dim Input]
    C --> D[Linear + BatchNorm + ReLU<br/>768 → 512]
    D --> E[Dropout 0.1]
    E --> F[Linear + BatchNorm + ReLU<br/>512 → 256]
    F --> G[Linear + BatchNorm + ReLU<br/>256 → 128]
    G --> H[Encoded<br/>128-dim]
    H --> I[Linear + BatchNorm + ReLU<br/>128 → 256]
    I --> J[Linear + BatchNorm + ReLU<br/>256 → 512]
    J --> K[Linear + BatchNorm + ReLU<br/>512 → 768]
    K --> L[Reconstructed<br/>768-dim]
    
    style A fill:#f9f,stroke:#333
    style H fill:#bbf,stroke:#333
    style L fill:#f9f,stroke:#333
```

The system processes recipe texts through a progressive denoising autoencoder. SBERT transforms text into 768-dimensional embeddings. The encoder reduces dimensionality through three compression stages. Each stage applies batch normalization and dropout for training stability.

## 3. Training Process and Loss Function:
The training applies noise to input embeddings for denoising learning. The model learns to reconstruct original embeddings from noisy inputs. Mean squared error quantifies reconstruction performance through:

$$
L = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{x_i})^2
$$

Where:
- $x_i$ represents the original embedding
- $\hat{x_i}$ represents the reconstructed embedding
- $n$ represents the batch size

