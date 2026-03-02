# US Car Accidents - Severity Prediction 

##  Project Overview
Traffic accidents are a critical public safety challenge. A proactive approach to road safety requires predicting the likelihood and severity of an accident based on contextual variables (weather, infrastructure, time, geographic location) *before* the aftermath is fully known. 

This project aims to build a robust machine learning pipeline capable of identifying critical-severity accidents (Severity = 4) using a large-scale dataset of US accident records. The core focus is on handling extreme class imbalance and strictly preventing data leakage.

---

##  Methodology & Data Preprocessing

The dataset is highly imbalanced, with critical accidents (Severity 4) making up only ~2.6% of the records. To ensure a reliable and deployable model, the following pipeline was implemented:

* **Data Leakage Prevention:** Removed post-accident columns such as `End_Time`, `End_Lat`, `End_Lng`, `Distance(mi)`, and `Description`. Including these would artificially inflate model performance, as they are unknown at the exact moment an accident occurs.
* **Temporal Feature Engineering:** Extracted `Hour`, `DayOfWeek`, `Month`, and `Is_Weekend` from the original `Start_Time` timestamp to capture human traffic patterns.
* **Missing Value Imputation:** * Features with >25% missing data (e.g., `Wind_Chill(F)`) were dropped to avoid statistical noise.
  * Numeric weather data was imputed using the **median** (robust to outliers), while categorical data used the **mode**.
  * Domain knowledge was applied: `NaN` values in `Precipitation(in)` were safely assumed to be 0 inches of rain.
* **Categorical Encoding:** Applied Boolean conversion (True/False -> 1/0) and Label Encoding to prepare the dataset for tree-based algorithmic processing.

---

##  Modeling Strategy & Evaluation

The problem was framed as a **Binary Classification** task (Severity 4 vs. Others) to focus exclusively on life-threatening events. 

* **Algorithm:** `RandomForestClassifier`
* **Class Imbalance Handling:** Utilized `class_weight='balanced'` to force the algorithm to heavily penalize mistakes made on the rare, critical accidents. 
* **Validation:** Standard 80/20 Train-Test split with target stratification to maintain the 2.6% minority class distribution.

###  Results & The Recall Trade-off
The model deliberately prioritizes **Recall** over Precision. In a real-world emergency response scenario, a false positive (dispatching resources to a minor accident) incurs a financial cost, but a false negative (ignoring a life-threatening accident) incurs a human cost. 

* **Recall (Class 1 - Severe):** `0.71` (The model successfully intercepts 71% of all critical accidents).
* **Overall Accuracy:** `0.83`

---

##  Key Findings: Feature Importance

Extracting the feature importances from the Random Forest revealed the top 5 factors influencing accident severity:

1. **`Source` (~32.8%):** This highlights a crucial data collection bias. Different reporting APIs have different thresholds (e.g., highway-only sensors vs. urban traffic apps), proving that data provenance is a massive predictor of the event's scale.
2. **`Start_Lng` (~10.9%) & `Start_Lat` (~10.4%):** Exact coordinates map the road network implicitly, allowing the model to distinguish high-speed rural/highway areas (prone to severe crashes) from slow-moving dense urban centers.
3. **`State` (~7.7%):** Captures the macro-level differences in infrastructure, highway speed limits, and safety regulations across different US states.
4. **`Pressure(in)` (~4.7%):** While rain or snow are localized, sudden drops in atmospheric pressure often signal incoming severe weather fronts on a macro scale, acting as an early warning for extreme driving conditions.

---

##  How to Reproduce

1. **Clone this repository** to your local machine:
   ```bash
   git clone [https://github.com/LuigiMin-02/us-accidents-severity-prediction.git](https://github.com/LuigiMin-02/us-accidents-severity-prediction.git)
   ```
2. Install the required dependencies. Ensure your Python environment has the necessary libraries (pandas, numpy, scikit-learn, matplotlib, seaborn, pyarrow). You can install them via pip:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn pyarrow

   ```
3. Add the dataset. Download the original US Accidents dataset and place the Parquet (or CSV) file inside the data/raw/ directory.
4. Run the analysis. Open and execute the Jupyter Notebook located in the notebooks/ directory (01_eda_and_cleaning.ipynb) to replicate the data cleaning, model training, and feature extraction steps.
