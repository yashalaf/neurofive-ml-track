# neurofive-ml-track

Machine Learning Fundamentals track: Neurofive Solutions

Working through the Titanic dataset ([Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)) week by week: from first EDA, through cleaning and visualization, to a first classification model.

## Repo contents

- `titanic_eda.ipynb` the running notebook for all weeks below
- `train.csv`, `test.csv`, `gender_submission.csv` Titanic dataset files (also downloadable from the [Kaggle competition page](https://www.kaggle.com/competitions/titanic/data))

## Week 0: First EDA

Set up the Python toolkit (pandas, NumPy) and explored the raw dataset before touching anything.

- Loaded `train.csv`, `test.csv`, `gender_submission.csv` with `pandas.read_csv()`
- Inspected shape, structure, and summary stats with `.head()`, `.info()`, `.describe()`
- Found: 891 rows × 12 columns in training data; missing values in `Age` (~20%), `Cabin` (~77%), `Embarked` (2 rows); split columns into numerical vs. categorical
- ~38% overall survival rate in the training set

## Week 1: Clean & Visualize

Cleaned the dataset and used visualization to catch patterns and outliers.

**Missing value strategy:**
| Column | Missing % | Approach | Why |
|---|---|---|---|
| `Age` | ~20% | Median, grouped by `Pclass` | Age is right-skewed and varies by class, so median is more robust than mean |
| `Embarked` | 0.22% (2 rows) | Mode | Negligible impact either way; `S` dominates |
| `Cabin` | ~77% | Dropped column, added `HasCabin` flag | Too sparse to impute reliably; presence of a cabin is itself a useful (likely class/fare-linked) signal |

**Outlier detection:** boxplot on `Fare`, flagged values above the IQR upper bound.

**Visualizations:** histogram (`Age` distribution), boxplot (`Fare` by survival), bar chart (survival rate by `Pclass`), correlation heatmap (numerical features).

**Which feature affects survival most?** `Sex`, by a wide margin: 74.2% of women survived vs. 18.9% of men. `Pclass` and `Fare` were the next-strongest signals, consistent with better lifeboat access for higher-class/wealthier passengers.

## Week 2: First Classification Model

Trained a baseline Logistic Regression classifier to predict `Survived`.

**Approach:**
- Features used: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `HasCabin`
- Categorical columns (`Sex`, `Embarked`) one-hot encoded with `pd.get_dummies()`
- 80/20 train/test split via `train_test_split`, stratified on `Survived`
- Model: `sklearn.linear_model.LogisticRegression`

**Results:**
- **Accuracy: 81.01%**
- Confusion matrix:

  |  | Predicted: No | Predicted: Yes |
  |---|---|---|
  | **Actual: No** | 96 | 14 |
  | **Actual: Yes** | 20 | 49 |

- The model is stronger at catching non-survivors than survivors: 20 false negatives vs. 14 false positives, suggesting it slightly under-predicts survival, likely reflecting the class imbalance (more non-survivors than survivors in the data).

**Next steps:** feature scaling, regularization tuning, and comparing against tree-based models (Random Forest, Gradient Boosting) in upcoming weeks.

## Week 3: Model Evaluation & Tuning: Beyond Accuracy

Revisited the Week 2 Logistic Regression model to evaluate it properly (not just on accuracy) and tune it.

**Why accuracy alone can mislead:** on an imbalanced dataset like this (more non-survivors than survivors), a model can post a high accuracy score just by leaning toward the majority class. Precision, recall, and F1 broken out per class expose that imbalance in a way a single accuracy number can't.

**Tuning:** GridSearchCV, 5-fold CV, scored on F1, over:

- C (regularization strength): [0.01, 0.1, 1, 10, 100]
- solver: ["lbfgs", "liblinear"]

Best parameters found: C=10, solver=lbfgs

Before vs. after (on the "Survived" class):

Metric	Baseline	Tuned	Change
Accuracy	81.01%	81.01%	0.00
Precision	0.778	0.769	-0.009
Recall	0.710	0.725	+0.014
F1-score	0.742	0.746	+0.004

Takeaway: tuning gave a modest, incremental improvement — recall and F1 on the minority class ticked up slightly while accuracy stayed flat. This is a realistic outcome for a simple model; the bigger lever for further gains is likely better features or a different model family (e.g. tree-based ensembles) rather than more hyperparameter search on Logistic Regression.

## Tools

Python · pandas · NumPy · matplotlib · seaborn · scikit-learn · Jupyter Notebook
