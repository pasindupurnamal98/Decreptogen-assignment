# 🧠 Personality Prediction using Machine Learning

Welcome to the **Personality Prediction Project**, a machine learning-based application that classifies individuals as **Introverts** or **Extroverts** using various social and personal behavior metrics.

This repository provides a powerful and reproducible framework for personality trait prediction using classification algorithms, and serves as a useful basis for psychologists, educators, data scientists, and mental health professionals.

---

## 📌 Table of Contents

- [🎯 Project Objectives](#-project-objectives)
- [📊 Dataset](#-dataset)
- [🧪 Methodology](#-methodology)
  - [1. Data Loading & EDA](#1-data-loading--eda)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Model Training & Evaluation](#3-model-training--evaluation)
  - [4. Model Selection & Saving](#4-model-selection--saving)
- [🧰 Dependencies](#-dependencies)
- [🚀 Usage Instructions](#-usage-instructions)
- [📁 Project Structure](#-project-structure)
- [🔮 Future Enhancements](#-future-enhancements)
- [📄 License](#-license)

---

## 🎯 Project Objectives

- Develop a robust machine learning model to accurately predict personality types (Extrovert or Introvert).
- Compare various classification models including:
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - Logistic Regression
- Build a complete machine learning pipeline: data loading → preprocessing → model training → evaluation → deployment.
- Provide a reproducible and extendable framework for future work in personality analytics.

---

## 📊 Dataset

The dataset contains behavioral and social features influencing personality.

| Feature Name               | Type       | Description |
|---------------------------|------------|-------------|
| `Time_spent_Alone`        | Numerical  | Time spent alone per day |
| `Stage_fear`              | Categorical (Yes/No) | Exhibits fear of performing publicly |
| `Social_event_attendance`| Numerical  | Number of social events attended |
| `Going_outside`           | Numerical  | Frequency of leaving the home |
| `Drained_after_socializing`| Categorical | Feels exhausted after socializing |
| `Friends_circle_size`     | Numerical  | Size of friend circle |
| `Post_frequency`          | Numerical  | Social media post frequency |
| `Personality`             | Target     | Labels: Extrovert or Introvert |

---

## 🧪 Methodology

### 1. Data Loading & EDA

- Load `personality_dataset.csv` using `pandas`.
- Perform descriptive statistics, type inspection, and null value analysis.

### 2. Data Preprocessing

- **Numerical Imputation**: Fill missing values using the **mean**.
- **Categorical Imputation**: Fill using the **most frequent** value.
- **Encoding**:
  - Categorical → One-Hot Encoding  
  - Target → Label Encoding (Extrovert = 0, Introvert = 1)
- **Pipeline Construction**: Using `ColumnTransformer` and `Pipeline` to automate preprocessing steps.

### 3. Model Training & Evaluation

Models used:

- ✅ Random Forest Classifier
- ✅ Gradient Boosting Classifier
- ✅ Support Vector Machine (SVM)
- ✅ Logistic Regression

✔ Models are fine-tuned using `GridSearchCV` for hyperparameter optimization.

✔ Performance metrics calculated:
  - Accuracy
  - Precision
  - Recall
  - F1-score

✔ Data split: 80% training, 20% testing

### 4. Model Selection & Saving

- The best model is selected based on **test accuracy** and **F1-Score**.
- Saves:
  - Best model (`best_personality_model.pkl`)
  - Full pipeline (`best_personality_model_pipeline.pkl`)
  - Preprocessor (`preprocessor.pkl`)
  - Label Encoder (`label_encoder.pkl`)
  - Model Metadata (`comprehensive_model_info.pkl`, `detailed_model_results.pkl`)
- Serializers: `joblib`, `pickle`

---

## 🧰 Dependencies

Install the required libraries with:

```bash
pip install pandas scikit-learn numpy matplotlib seaborn joblib
```

**Libraries Used**:

- [`pandas`](https://pandas.pydata.org/)
- [`scikit-learn`](https://scikit-learn.org/)
- [`numpy`](https://numpy.org/)
- [`matplotlib`](https://matplotlib.org/)
- [`seaborn`](https://seaborn.pydata.org/)
- [`joblib`](https://joblib.readthedocs.io/)
- [`pickle`](https://docs.python.org/3/library/pickle.html)

---

## 🚀 Usage Instructions

### ⚙️ Steps to Run the Project:

1. **Clone the Repository**:

```bash
git clone https://github.com/your-username/Personality-Prediction.git
cd Personality-Prediction
```

2. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the Jupyter Notebook**:

```bash
jupyter notebook app.ipynb
```
or open `final_notebook.ipynb` manually.

4. **Make Predictions**  
Use the saved model from the `/pkl` directory to predict new data.

**Example**:

```python
import joblib
import pickle

# Load model pipeline
model = joblib.load('pkl/best_personality_model_pipeline.pkl')
# Predict
data_point = your_input_dataframe  # Replace with your DataFrame
prediction = model.predict(data_point)
```

---

## 📁 Project Structure

```
Decreptogen-assignment/
├── data/
│   └── personality_dataset.csv
├── frontend/
│   └── frontend.py              # Streamlit App
├── pkl/                         # Saved models and metadata
│   ├── best_personality_model.pkl
│   ├── best_personality_model_pipeline.pkl
│   ├── comprehensive_model_info.pkl
│   ├── detailed_model_results.pkl
│   ├── label_encoder.pkl
│   └── preprocessor.pkl
├── app.ipynb
├── final_notebook.ipynb
├── personality_prediction.ipynb
├── final_notebook_33.py
├── Personality Prediction Project.pdf
└── README.md
```

---

## 🔮 Future Enhancements

- [ ] **Larger Dataset**: Improve accuracy with more diverse data.
- [ ] **Advanced Feature Engineering**: Derive additional behavioral indicators.
- [ ] **Deep Learning Models**: Experiment with neural networks.
- [ ] **Web API Deployment**: Using `Flask` or `FastAPI`.
- [ ] **Frontend Enhancement**: Improve `frontend.py` Streamlit UI.
- [ ] **CI/CD Integration**: Automate testing and deployment pipelines.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## ✨ Acknowledgements

Thanks to the contributors and tool developers who made this project possible.

---

> 🔗 *Feel free to fork, improve and build upon this project. Pull requests are welcome!*

---

Would you like a downloadable **`README.md`** version or include badges (GitHub Actions, Stars, etc.)? Let me know, I’ll provide it!
