# 🚢 Titanic Survival Prediction

This project is a **Machine Learning classification pipeline** built using Python to predict the survival of passengers on the Titanic dataset.  
It includes **data preprocessing, visualization, model training, hyperparameter tuning, and evaluation**.

---

## 📊 Dataset
The dataset used: `train.csv` from the Titanic Kaggle competition.  
Main columns used after preprocessing:
- **Pclass**
- **Sex**
- **Fare**
- **Embarked**
- **Survived** (Target)

---

## 🛠️ Steps in the Project

### 1️⃣ Data Exploration
- Checked null values & dataset info.
- Visualized important features like `Sex`, `Pclass`, and `Survived`.
- Plotted correlation heatmap with `Survived`.

### 2️⃣ Data Preprocessing
- **Numerical Columns** → Imputed missing values using mean & scaled with `StandardScaler`.
- **Categorical Columns** → Imputed missing values using most frequent & encoded with `OneHotEncoder`.
- Dropped irrelevant columns: `Name`, `Cabin`, `PassengerId`, `Ticket`.

### 3️⃣ Model Building
Trained multiple models using `Pipeline`:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- KNN  
- XGBoost  
- SVM  

### 4️⃣ Model Evaluation
- Calculated **Accuracy Score** for all models.
- Visualized model comparison with **Bar Chart**.
- Plotted **Confusion Matrices** for all models.
- Displayed **Classification Reports** for precision, recall, and F1-score.

### 5️⃣ Hyperparameter Tuning
- Applied `GridSearchCV` on **Random Forest** to find the best parameters:
  - `n_estimators`
  - `max_depth`

---

## 📈 Results

| Model              | Accuracy (%) |
|-------------------|-------------|
| Logistic Regression | XX.XX |
| Decision Tree       | XX.XX |
| Random Forest       | XX.XX |
| KNN                | XX.XX |
| XGBoost            | XX.XX |
| SVM                | XX.XX |

_(Replace `XX.XX` with your actual results after running the notebook)_

---

## 🖼️ Visualizations
- Heatmap showing correlation with survival.
- Countplots for passenger demographics.
- Boxplots for numeric features.
- Model accuracy comparison bar chart.
- Confusion matrices for all models.

---

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/<your-username>/titanic-survival-prediction.git

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
jupyter notebook
