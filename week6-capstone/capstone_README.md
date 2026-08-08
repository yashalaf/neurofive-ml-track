# Capstone: Pakistan Property Price Predictor

**Neurofive ML Track Week 6 Capstone: End-to-End Machine Learning Project**

## Problem statement

Pakistan's real estate market — especially in cities like Lahore, Karachi, and Islamabad is famously opaque: asking prices vary wildly by area, property type, and even by which side of a road a plot sits on, and there's no simple, trustworthy way for a buyer, seller, or agent to sanity-check whether a listed price is reasonable. This project builds a machine learning model that predicts a property's sale price in PKR from its basic attributes (city, property type, size, bedrooms, bathrooms, location), trained on real listings scraped from Zameen.com, Pakistan's largest property portal.

## Dataset

[Zameen.com Property Data Pakistan](https://www.kaggle.com/datasets/huzefa11/zameencom-property-data-pakistan) (Kaggle) 191,393 property listings. Scoped to "For Sale" listings only (127,018 rows), then cleaned down to 123,049 rows after outlier removal.

## Approach

1. **Clean:** parsed the `area` column (text like `"9 Marla"` / `"6 Kanal"`) into a single numeric unit (Marla); scoped to "For Sale" listings only (mixing in "For Rent" would confuse the model, since rent and sale prices are on completely different scales); dropped 3 zero-price data errors; clipped price and area to the 1st–99th percentile to remove extreme outliers while preserving the market's natural spread.
2. **EDA:** examined price distribution, price-per-Marla by city, area-vs-price relationships, and a correlation heatmap.
3. **Feature engineering:** `area_marla` (parsed size), `log_price` (log-transformed target, since raw price is heavily right-skewed), `bed_bath_ratio`, and `price_per_marla` (for analysis).
4. **Modeling:** built a single `ColumnTransformer` + `Pipeline` (StandardScaler on numerical features, OneHotEncoder on categorical features) and trained three models on the same 80/20 split: Linear Regression, Random Forest, and XGBoost.
5. **Evaluation:** compared RMSE, MAE, and R² all measured on the real PKR price scale (not the log scale used for training), for interpretability.

## Results

| Model | RMSE (PKR) | MAE (PKR) | R² |
|---|---|---|---|
| **XGBoost** | **7,932,475** | **3,690,308** | **0.902** |
| Random Forest | 8,160,304 | 3,869,055 | 0.896 |
| Linear Regression | 55,455,960 | 10,988,990 | -3.797 |

**XGBoost was the best-performing model**, explaining about 90.2% of the variance in property prices, with a mean absolute error of roughly PKR 3.69 million.

**A notable finding:** Linear Regression scored a *negative* R² on the real price scale, even though it used the same log-transformed target as the other models. The cause: linear models can extrapolate to extreme values on the log scale for certain feature combinations, and small log-space errors become catastrophic once reversed (one test prediction came out to over PKR 5 billion, against real prices topping out around PKR 180 million). Tree-based models don't have this problem, since they can't extrapolate beyond the range of values seen during training — a good illustration of why tree ensembles are generally preferred for skewed, real-world price data.

**Top feature importances (XGBoost):** `area_marla` (~53%), `property_type_House` (~16%), city and location signals (`latitude`/`longitude`/city dummies, combined ~15%), `bedrooms` (~5%).

## Live app

_[(https://neurofive-ml-track-bvgspaakcspwuanpfge67k.streamlit.app/)]_

## How to run this project

1. Clone this repo and `cd` into the `week6` (or wherever this capstone lives) folder.
2. Download `Property.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/huzefa11/zameencom-property-data-pakistan) and place it in this folder (not included in the repo see note below).
3. Install dependencies: `pip install -r requirements.txt`
4. Open and run `capstone_property_price.ipynb` top to bottom to reproduce the full analysis and regenerate `property_price_model.joblib`.
5. To run the app locally: `streamlit run app.py`

**To deploy the app yourself (Streamlit Community Cloud):**
1. Push `app.py`, `requirements.txt`, and `property_price_model.joblib` to this repo (same folder)
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
3. Click **New app**, select this repo, set the branch (`main`) and main file path (e.g. `week6/app.py` if it's in a subfolder)
4. Deploy — Streamlit Cloud installs `requirements.txt` and launches the app automatically

**Note on the dataset file:** `Property.csv` is not included in this repo to keep it lightweight; download it from the Kaggle link above and place it alongside the notebook before running.

## Real-world value

A model like this — wrapped in the Streamlit app — gives buyers, sellers, and agents an independent, data-driven reference price in seconds, instead of relying purely on gut feel or a single agent's opinion. It's the same underlying idea behind Zillow's "Zestimate" or Redfin's automated valuation models, scoped to the Pakistani market. Realistic next steps toward a production-grade version: incorporate neighborhood-level (not just city-level) location data, property age/condition where available, and recent comparable sales trends over time.

## Tools

Python · pandas · NumPy · matplotlib · seaborn · scikit-learn · XGBoost · Streamlit · Jupyter Notebook
