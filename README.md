## Dataset Description
The system processes recipes from TudoGostoso website. Each recipe contains name, ingredients list, cooking instructions, and URL. The dataset contains 2500 recipes in Portuguese. The data structure enables search through ingredients and cooking methods.

## Embedding Generation Process
The system converts recipes into vectors through SBERT's distiluse-base-multilingual-cased-v2 model. The embeddings undergo mean pooling to create single vectors. A denoising autoencoder reduces the 768-dimensional SBERT embeddings into 128-dimensional vectors through three hidden layers [512, 256, 128].

```mermaid
graph LR
    subgraph Encoder
        I[Input 768] --> E1[Linear 512]
        E1 --> E1B[BatchNorm]
        E1B --> E1R[ReLU]
        E1R --> E1D[Dropout]
        E1D --> E2[Linear 256]
        E2 --> E2B[BatchNorm]
        E2B --> E2R[ReLU]
        E2R --> E2D[Dropout]
        E2D --> E3[Linear 128]
        E3 --> E3B[BatchNorm]
        E3B --> E3R[ReLU]
        E3R --> E3D[Dropout]
    end
    subgraph Decoder
        E3D --> D1[Linear 256]
        D1 --> D1B[BatchNorm]
        D1B --> D1R[ReLU]
        D1R --> D1D[Dropout]
        D1D --> D2[Linear 512]
        D2 --> D2B[BatchNorm]
        D2B --> D2R[ReLU]
        D2R --> D2D[Dropout]
        D2D --> D3[Linear 768]
        D3 --> D3B[BatchNorm]
        D3B --> D3R[ReLU]
        D3R --> D3D[Dropout]
    end
```

## Training Process
The autoencoder training uses MSE loss to minimize reconstruction error between input and output vectors. The loss function measures distance between original embeddings and their reconstructed versions after noise addition:

$$ L(x, \hat{x}) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x_i})^2 $$

The training runs for 100 epochs with batch size 32 and learning rate 1e-4. The denoising process forces the model to learn robust recipe representations.