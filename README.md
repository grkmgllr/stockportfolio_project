# AI-Based Stock Portfolio Management & Secure Forecasting

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Development-yellow)

**ENS 491 - Graduation Project (Design)** | **Sabancı University**

This project implements an AI-assisted stock portfolio management framework that integrates state-of-the-art neural time-series forecasting (**PatchTST**, **TimesNet**) with decision-support mechanisms. Beyond predictive accuracy, the project rigorously evaluates **AI Security**, specifically focusing on model resilience against backdoor (Trojan) attacks in financial contexts.

## 👥 Project Team

| Name | Role |
| :--- | :--- |
| **Alanur Ersoy** | Researcher / Developer |
| **Ege Serin** | Researcher / Developer |
| **Görkem Güller** | Researcher / Developer |

**Supervisor:** Mehmet Emre Özfatura

## 🚀 Project Objectives

1.  **Neural Forecasting:** Implement and compare advanced architectures (**PatchTST**, **TimesNet**, **TimeMixer**) for short-term stock price prediction ($H=1$ to $5$ days).
2.  **Portfolio Management:** Convert forecasts into actionable "Long/Short/Neutral" signals using a threshold-based decision logic ($\delta$).
3.  **Adversarial Robustness:** Design and test dynamic backdoor trigger modules to evaluate the security of financial AI models against manipulation.
4.  **HPC Optimization:** Leverage Sabancı University's HPC clusters for efficient training on high-dimensional financial tensors.

## 🏗️ Architecture & Methods

The project utilizes a modular design separating data ingestion, modeling, and security evaluation.

### Forecasting Models
* **[PatchTST](https://arxiv.org/abs/2211.14730):** Uses "sub-series tokenization" (patching) to capture long-range dependencies without the computational cost of standard Transformers. It treats time series patches as visual tokens.
* **[TimesNet](https://arxiv.org/abs/2210.02186):** Transforms 1D time series into 2D tensors to capture intra-period and inter-period variations using Convolutional Neural Networks (CNNs).

### Data Pipeline
* **Source:** Yahoo Finance API (`yfinance`).
* **Features:** Open, High, Low, Close, Volume (OHLCV).
* **Preprocessing:** Robust normalization (StandardScaler/RevIN) and sliding window tensor generation ($X \in \mathbb{R}^{L \times C}$).

## 📂 Project Structure

```text
ens491-portfolio-forecasting/
├── data/                  # Data storage (Ignored by Git)
│   ├── raw/               # Raw CSV downloads from yfinance
│   └── processed/         # Normalized tensors for training
├── notebooks/             # Jupyter notebooks for EDA and Visualization
├── src/
│   ├── data_loader/       # PyTorch Dataset classes (Sliding Window logic)
│   ├── models/            # Neural Network Architectures (PatchTST, TimesNet)
│   ├── scripts/           # Execution scripts (Fetch, Process, Train)
│   └── utils/             # Metrics and Visualization helpers
├── hpc_jobs/              # SLURM scripts for Sabancı Cluster execution
└── requirements.txt       # Python dependencies

```

🛠️ Installation & Setup
1. Clone the Repository

```bash
git clone [https://github.com/grkmgllr/stockportfolio_project.git](https://github.com/grkmgllr/stockportfolio_project.git)
cd stockportfolio_project
```

2. Set Up Virtual Environment

It is recommended to use a clean virtual environment to avoid conflicts.

Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```
3. Install Dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

💻 Usage
Step 1: Data Acquisition

Download the daily OHLCV data for the target portfolio (e.g., AAPL, MSFT, TSLA).

```bash
python src/scripts/fetch_data.py
Step 2: Preprocessing
```

Clean missing values and apply Normalization (StandardScaler) to prepare tensors.

```bash
python src/scripts/process_data.py
```
Step 3: Visualization (Optional)

Launch Jupyter to explore market regimes and volatility clusters.

```bash
jupyter notebook notebooks/visualization.ipynb
```

Step 4: Training (Local)

Train the PatchTST model on your local machine (supports MPS/CUDA/CPU).

```bash
# Coming soon in Phase 2
# python src/scripts/train_local.py
```

📚 References

PatchTST: Nie, Y., et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023.

TimesNet: Wu, H., et al. (2023). "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis." ICLR 2023.

TSLIB: Time-Series-Library GitHub: https://github.com/thuml/Time-Series-Library

## License
This project is open-source and available under the **MIT License**.

