# Store Sales - Time Series Forecasting

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Forecasting-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange.svg)

## Project Summary
This project tackles the challenge of predicting unit sales for thousands of items across different stores of **Corporación Favorita**, a large grocery retailer based in Ecuador. 

Accurate forecasting is crucial in the retail industry: overestimating demand leads to overstocking and waste of perishable goods, while underestimating it causes stockouts, lost revenue, and dissatisfied customers. 

**Objective:** Build a Machine Learning model to predict daily sales (the `sales` variable) over a 16-day period, minimizing the Root Mean Squared Logarithmic Error (RMSLE).

## The Data
> **Note:** Following development best practices and Git size limits, the original datasets are not included in this repository.

The data can be downloaded from the official competition: [Kaggle: Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).

The dataset includes detailed information regarding:
* **Dates and stores:** Historical sales by store and product family.
* **Promotions:** Discounted items on specific dates.
* **External factors:** Daily oil prices (fundamental to the Ecuadorian economy) and local/national holidays or events.

## Analytic Approach
The project's development followed a standard data science workflow:

1. **Exploratory Data Analysis (EDA):** Identification of trends, weekly/annual seasonality, and the impact of exogenous variables such as oil prices and holidays.
2. **Feature Engineering:** * Extraction of time-based components (day of the week, month, year).
   * Creation of lag variables and moving averages to capture sales momentum.
3. **Modeling:** Implementation of **LightGBM** (`LGBMRegressor`), a highly efficient and robust Gradient Boosting algorithm for large-volume time series.
4. **Optimization and Post-processing (Key Techniques):**
   * **Target Variable Transformation:** The model was trained using the logarithm of sales to smooth out skewed distributions, reverting the calculation (`np.expm1`) when generating the final prediction.
   * **Business Constraint (Clipping):** `np.clip(predictions, 0, None)` was applied to enforce a lower bound of zero, ensuring the model did not forecast negative sales (which is mathematically possible but logically incorrect in retail).
   * **Hyperparameter Tuning:** Advanced LightGBM tuning (`n_estimators=2000`, `num_leaves=127`) to maximize learning depth while preventing overfitting.

## Results and Evaluation
The model was evaluated using the official competition metric, **RMSLE** (Root Mean Squared Logarithmic Error). This metric penalizes relative errors and is ideal when there is a wide range of target values.

* **Validation RMSLE Score:** `0.48`

## How to Reproduce This Project

If you want to run this project locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone [https://github.com/JuanCL2000/Store-Sales---Time-Series-Forecasting.git](https://github.com/JuanCL2000/Store-Sales---Time-Series-Forecasting.git)
   cd Store-Sales---Time-Series-Forecasting
   
2. Install the necessary dependencies:

pip install -r requirements.txt

3. Download the data from Kaggle and extract the .csv files into a folder named data/raw/ in the project root.

4. Run the Jupyter Notebook:

jupyter notebook notebooks/[store_sales_forecasting.ipynb].ipynb

### Autor

Juan Cuellar - [www.linkedin.com/in/juan-cuellar-lugo-3a55b3374]

GitHub: JuanCL2000
