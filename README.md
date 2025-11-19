# Drug Effect Classification: Frequency-Based Traversal Framework

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Research_Prototype-orange)

## 📌 Overview

This repository contains the implementation of a **Frequency-Based Traversal Method** for classifying drug–disease associations. The framework analyzes path data between drugs and diseases to categorize their effects into:

- **DM:** Disease Modifying  
- **SYM:** Symptomatic  
- **NOT:** No Effect / Negative  

Unlike standard "black box" classifiers, this approach uses a rule-based traversal mechanism that scores paths based on gene frequency and similarity mapping, enabling interpretable results for drug repurposing research.

---

## 📂 Repository Structure

```text
├── analyze_data.py              # Core logic for path processing, scoring, and evaluation
├── random_splitter_shareable.py # Entry point: Handles data splitting and initiates testing
├── utils.py                     # Helper functions for accuracy and data retrieval
├── indications_300.zip          # Compressed dataset containing drug-disease paths
├── similarity.csv               # Node similarity scores for handling unknown genes
├── split_train.csv              # Initial training split
├── split_test.csv               # Initial testing split
└── README.md                    # Project documentation
```

---

## 🛠️ Setup & Installation

### 1️⃣ Install Dependencies  
Ensure you have Python **3.8+** installed, then run:

```bash
pip install pandas numpy tqdm scikit-learn
```

### 2️⃣ Unzip Dataset  
The main dataset (`indications_300.zip`) must be unzipped to generate `indications_300.csv`.

#### Linux/macOS:
```bash
unzip indications_300.zip
```

#### Windows (PowerShell):
```powershell
Expand-Archive -Path indications_300.zip -DestinationPath .
```

➡️ Ensure that **indications_300.csv** appears in the **root directory** beside `analyze_data.py`.

---

## 🚀 How It Works

### 1. Data Splitting (`random_splitter_shareable.py`)

- Implements **custom Stratified Shuffle Split** for uneven class distributions.  
- Runs across **5 random seeds (41–45)** for robustness.  
- Handles small sample sizes via heuristic rules.  
- All **NOT** cases are forced into the training set to serve as negative controls.

---

### 2. Path Analysis (`analyze_data.py`)

1. **Training Phase**
   - Learns gene occurrence frequencies (`dm`, `sym`, `not`) from the training split.

2. **Path Scoring**
   - Uses `score_path` to evaluate paths based on **known genes** observed during training.

3. **Similarity Extension**
   - If a test-path gene was unseen during training:
     - Looks up **similar genes** from `similarity.csv`.
     - Uses the most similar known gene as a proxy if similarity ≥ **0.4**.

4. **Classification**
   - Examines the **top 200 paths** for each drug–disease pair.
   - Generates a probability vector and final class prediction.

---

## 💻 Usage

Run the full experiment across all random seeds:

```bash
python random_splitter_shareable.py
```

---

## 📄 Output Files

The script generates multiple analysis files:

- **di_gene_df50.csv** – Intermediate disease → gene frequency map  
- **test_df500_rulebased3_[SEED].csv** – Raw path vectors & scores  
- **test_df500_evaluated_rulebased3_[SEED].csv** – Final evaluated results with accuracy metrics  

Console output includes metrics via `utils.get_accuracy_avg`.

---

## 📊 Evaluation Metrics

The framework calculates:

- **Accuracy**  
- **Sensitivity (Recall)** – Detecting DM/SYM cases  
- **Specificity** – Correct handling of NOT cases  
- **F1-Score**

---

## ⚙️ Configuration Notes

### Hyperparameters
- `khop = 500` – Defines neighborhood search depth  
- `max >= 0.4` – Minimum similarity threshold for gene proxy matching  

### File Paths  
Ensure all `.csv` files are in the project root or update paths inside the scripts.


