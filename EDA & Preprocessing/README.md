# **Bank Churn Prediction - Data Analysis & Machine Learning**

## **📌 Overview**

Customer churn is a critical problem for banks, as retaining existing customers is often more cost-effective than acquiring new ones. This project aims to analyze a **Bank Churn Dataset**, explore patterns using **Exploratory Data Analysis (EDA)**, and build a **Machine Learning model** to predict customer churn.

## **📂 Dataset Information**

The dataset contains **14 features** related to customer demographics, account details, and banking behaviors. The target variable is **Exited**, indicating whether a customer has left the bank (1) or not (0).

### **🔢 Features:**

| Feature           | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `CustomerId`      | Unique customer ID                                              |
| `Surname`         | Customer's last name                                            |
| `CreditScore`     | Customer's credit score                                         |
| `Geography`       | Country (Spain, France, Germany)                                |
| `Gender`          | Male or Female                                                  |
| `Age`             | Customer's age                                                  |
| `Tenure`          | Number of years with the bank                                   |
| `Balance`         | Customer's account balance                                      |
| `NumOfProducts`   | Number of bank products used                                    |
| `HasCrCard`       | Whether the customer has a credit card (1 = Yes, 0 = No)        |
| `IsActiveMember`  | Whether the customer is an active bank member (1 = Yes, 0 = No) |
| `EstimatedSalary` | Estimated salary of the customer                                |
| `Exited`          | Target variable (1 = Churn, 0 = Retained)                       |

## **📊 Exploratory Data Analysis (EDA)**

We performed a detailed analysis to understand customer behavior using **Plotly, Seaborn, and Matplotlib**.

### **📌 Key Insights:**

1. **Geography & Churn:**
   - Highest exit rate among customers from **Germany**.
   - Lowest exit rate among customers from **France**.
2. **Gender & Churn:**
   - **Female customers** show a higher churn rate compared to males.
3. **Age & Churn:**
   - Customers aged **50+** have a significantly higher exit rate.
4. **Tenure & Churn:**
   - **Newer customers (1-2 years)** are more likely to churn.
   - Customers with **9-10 years** of tenure are less likely to leave.
5. **Balance & Churn:**
   - Customers with a **zero balance** are more likely to exit.
6. **Number of Products & Churn:**
   - Customers with **only one product** tend to leave.
7. **IsActiveMember & Churn:**
   - **Inactive members** have a higher exit rate than active members.

### **📈 Data Visualizations:**

- **Histograms & Boxplots** for numerical distributions.
- **Pie Chart** for churn distribution.
- **Bar Charts** for categorical analysis.
- **Sunburst Plot** to analyze multi-level relationships.
- **Heatmap** to visualize feature correlations.

## **🛠️ Data Preprocessing & Feature Engineering**

### **1️⃣ Handling Missing & Duplicate Values**

- Checked for missing and duplicate values (None found in this dataset).

### **2️⃣ Encoding Categorical Features**

- Used **One-Hot Encoding** for `Geography`.
- Used **Label Encoding** for `Gender`.
- Converted `Age` into meaningful categories:
  ```python
  bins = [0, 12, 19, 35, 50, 100]
  labels = ['Child', 'Teen', 'Young Adult', 'Middle-Aged Adult', 'Senior']
  df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
  ```

### **3️⃣ Salary Categorization**

Divided `EstimatedSalary` into income groups to improve model performance:

```python
bins = [11, 20000, 50000, 100000, 150000, 200000]
labels = ['Very Low Income', 'Low Income', 'Middle Income', 'High Income', 'Very High Income']
df['SalaryCategory'] = pd.cut(df['EstimatedSalary'], bins=bins, labels=labels)
```

Encoded using **Ordinal Encoding**.

### **4️⃣ Feature Scaling**

Applied **StandardScaler** to numerical columns for consistent model performance:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['CreditScore', 'EstimatedSalary']] = scaler.fit_transform(df[['CreditScore', 'EstimatedSalary']])
```

## **🤖 Model Building & Evaluation**

We trained multiple machine learning models to predict customer churn.

### **1️⃣ Data Splitting**

```python
from sklearn.model_selection import train_test_split
X = df.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **2️⃣ Models Implemented**

- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier** (Best Performing Model ✅)
- **Neural Network (MLP Classifier)**

### **3️⃣ Model Performance**

| Model                | Accuracy  | Precision | Recall    | F1-Score  |
| -------------------- | --------- | --------- | --------- | --------- |
| Logistic Regression  | 80.5%     | 72.3%     | 65.2%     | 68.6%     |
| Random Forest        | 85.1%     | 76.4%     | 72.8%     | 74.6%     |
| **XGBoost (Best)**   | **87.2%** | **78.1%** | **75.6%** | **76.8%** |
| Neural Network (MLP) | 84.3%     | 75.0%     | 70.2%     | 72.5%     |

### **4️⃣ Hyperparameter Tuning (Grid Search)**

Used **GridSearchCV** to optimize `Random Forest` and `XGBoost` parameters:

```python
from sklearn.model_selection import GridSearchCV
params = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

## **📝 Conclusion**

- **XGBoost** performed best, achieving **87.2% accuracy**.
- Important factors affecting churn: **Age, Geography, Active Membership, Balance, and Number of Products**.
- **Feature Engineering & Encoding significantly improved performance**.
- **Scaling helped stabilize model performance**.
- Further improvements can be made using **Deep Learning (LSTMs) or customer segmentation strategies**.

## **🚀 Future Work**

- Implement deep learning models like **LSTMs** for time-series analysis.
- Introduce **customer segmentation** for targeted retention strategies.
- Use **SHAP values** for explainability of model predictions.

## **📌 How to Use?**

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/bank-churn-prediction.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script.
4. Train & Evaluate the model.

## **📢 Acknowledgments**

Special thanks to **Kaggle** for the dataset & open-source ML community for valuable insights!

---

**🔗 Author:** [Muhammad Faizan](https://github.com/faizan-yousaf)


