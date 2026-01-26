# Spectral Text Fusion: A Frequency-Aware Approach to Multimodal Time-Series Forecasting

## Environment Setup

```sh
conda create -n spectf python=3.11.11
conda activate spectf
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install pandas scikit-learn patool tqdm sktime matplotlib reformer_pytorch transformers
```

## Training
We provide examples at ./scripts.
```sh
bash scripts/agriculture/SpecTF.sh
```

