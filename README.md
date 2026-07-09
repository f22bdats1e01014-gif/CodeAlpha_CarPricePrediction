# CodeAlpha_CarPricePrediction 

**Task 3 — Car Price Prediction with Machine Learning**
CodeAlpha Data Science Internship

## 📁 What's in this folder

| File | Description |
|---|---|
| `Car_Price_Prediction.ipynb` | Full Jupyter notebook — EDA, outlier detection, feature engineering, cross-validation, hyperparameter tuning, XGBoost, residual analysis, and a live "predict a new car" tool. Already executed with all outputs. |
| `Car_Price_Prediction_Report.html` | Standalone HTML export of the notebook — open in any browser, no Jupyter needed. |
| `Executive_Summary.docx` | One-page business-style summary of the project for non-technical readers. |
| `app.py` | Streamlit web app — enter a car's specs and get a live price prediction in your browser. |
| `car_data.csv` | The dataset (301 used-car listings). |
| `requirements.txt` | All Python packages needed. |
| `outputs/` | All generated plots, the trained model (`.pkl`), and the model comparison table. |

## 🛠 What's new in this Extended Edition

Beyond the original pipeline, this version adds:

1. **Outlier detection** — IQR-based check on price and mileage columns, with a documented decision to keep legitimate high-value cars rather than dropping them.
2. **Brand feature** — extracts a brand/model token from `Car_Name` and frequency-encodes it, instead of dropping the name entirely.
3. **5-fold cross-validation** — more robust performance estimates than a single train/test split.
4. **Hyperparameter tuning** — `GridSearchCV` search over Random Forest's `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
5. **XGBoost** — a gradient boosting model trained and compared against Random Forest.
6. **Residual analysis** — checks whether prediction errors are randomly distributed or show a pattern.
7. **Predict-a-new-car tool** — a reusable Python function (and matching Streamlit app) to estimate any car's resale price from its specs.
8. **HTML report** — shareable without needing Jupyter installed.
9. **One-page executive summary** — a business-style write-up of the project.

## 📈 Final Results

| Model | R² Score | MAE (Lakh INR) | RMSE (Lakh INR) |
|---|---|---|---|
| **Random Forest (Tuned)** | **0.961** | **0.63** | **0.95** |
| Random Forest (Baseline) | 0.958 | 0.64 | 0.98 |
| XGBoost | 0.946 | 0.66 | 1.11 |
| Linear Regression | 0.846 | 1.21 | 1.89 |

**Winner:** Random Forest (Tuned via GridSearchCV) — saved as the final model.

## ▶️ How to Run

### The notebook
```bash
pip install -r requirements.txt
jupyter notebook Car_Price_Prediction.ipynb
```
Or open directly in VS Code with the Jupyter extension installed.

### The web app
```bash
pip install -r requirements.txt
streamlit run app.py
```
This opens a browser tab where you can enter car specs and get an instant predicted price. **Run the notebook at least once first** so `outputs/car_price_model.pkl` and `outputs/preprocessing_artifacts.pkl` exist — the app loads these.

### The HTML report
Just double-click `Car_Price_Prediction_Report.html` — it opens in your default browser, no installation needed.

## 🔍 Key Insights
- `Present_Price` is by far the strongest predictor of resale value.
- `Car_Age` and `Driven_kms` reduce resale value, consistent with typical depreciation.
- Diesel and automatic-transmission cars trend toward higher resale prices in this dataset.
- Model residuals are symmetric around zero — no systematic bias in any price range.

## 🧰 Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Joblib, Streamlit
