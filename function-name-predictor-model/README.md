# Lightweight Function Name Prediction from Metadata

A lightweight machine learning model that predicts camelCase function names from structured function metadata. Designed for deployment on Android and low-computation devices.

---

## Problem Statement

Given metadata such as a function description, input parameters, return type, library name, keywords, and parameter count — predict the correct camelCase function name.

**Example:**

| Input Metadata | Predicted Output |
|---|---|
| "Adds two integers int a int b return int keywords add sum" | `addNumbers` |
| "Converts temperature from Celsius to Fahrenheit float celsius" | `convertCelsiusToFahrenheit` |
| "Sends HTTP GET request String url return HttpResponse" | `sendGetRequest` |

---

## Dataset

**File:** `function_metadata.csv`

- **180 rows** — 60 unique functions × 3 description variants each
- **15 domains:** MathUtils, StringUtils, NetworkService, AuthService, FileManager, DatabaseHelper, CryptoUtils, CacheManager, Logger, Validator, NotificationService, DateTimeUtils, ConversionUtils, DeviceUtils, ListUtils
- **Format matches assignment specification exactly** — typed parameters (`int a, int b`), camelCase function names, clean library names

| Column | Example |
|---|---|
| `description` | Adds two integers and returns their sum |
| `parameters` | `int a, int b` |
| `return_type` | `int` |
| `library` | `MathUtils` |
| `keywords` | `add,sum,arithmetic` |
| `param_count` | `2` |
| `function_name` | `addNumbers` |

---

## Model

| Component | Choice |
|---|---|
| Vectorization | TF-IDF (unigrams + bigrams, 3000 features) |
| Classifier | Logistic Regression + LinearSVC (best selected) |
| Feature engineering | Keywords repeated 3× · CamelCase tokenization · param count token |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **1.0000** |
| CV Accuracy (5-fold) | **0.87 ± 0.08** |
| Top-3 Accuracy | **1.0000** |
| Inference Latency | **< 0.1 ms/sample** |
| Model Package Size | **~930 KB** (< 2 MB Android limit) |

---

## Files

| File | Description |
|---|---|
| `Function_Name_Predictor.ipynb` | Main notebook — all sections, runs top-to-bottom |
| `function_metadata.csv` | Dataset (180 rows, synthetic, assignment format) |
| `fn_predictor.pkl` | Trained classifier |
| `vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `label_encoder.pkl` | Fitted label encoder |

---

## How to Run

1. Open `Function_Name_Predictor.ipynb` in [Google Colab](https://colab.research.google.com)
2. Click **Runtime → Run all**
3. All outputs are generated in order — no internet required (dataset is embedded)

---

## Inference Demo

```python
result = predict_function_name(
    "Adds two integers int a int b return int keywords add sum"
)
# Output: {'prediction': 'addNumbers', 'top_k': [...], 'latency_ms': 0.09}
```

---

## Assignment

**Course Assignment:** Lightweight Function Name Prediction from Metadata  
**Evaluation Criteria:** Dataset Quality (20%) · Model Accuracy (30%) · Model Size (15%) · Inference Speed (15%) · Deployment Demo (20%)
