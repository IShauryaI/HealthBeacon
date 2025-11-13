## 🫀 Heart Disease Prediction using Machine Learning

### 👨‍💻 Authors  
**Ayeshkant Mallick** & **Sandeep Kapoor**

---

### 📘 Section Overview

This project section predicts the likelihood of **heart disease** using two supervised machine learning models:
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**

This is built around the **CDC Heart Disease dataset** (`heart_2020_cleaned.csv`) and includes:
- Data preprocessing (encoding, scaling)
- Model training, evaluation, and comparison
- An interactive **Gradio-based UI** for real-time predictions

---

### 🧩 Files Included

| File | Description |
|------|--------------|
| `heart-disease-prediction_updated.ipynb` | Jupyter notebook for training the models. Handles preprocessing, encoding, scaling, and model evaluation. |
| `HA_UI.ipynb` | Jupyter notebook containing the Gradio UI for user interaction. Lets users input health parameters and get predictions from trained models. |
| `heart_2020_cleaned.csv` | Cleaned dataset used for model training. |
| `knn_model.pkl`, `decision_tree_model.pkl` | Saved trained models (generated after running the training notebook). |
| `transformer.pkl`, `scaler.pkl` | Saved preprocessing pipeline objects used during training (for consistent transformation in the UI). |

---

### ⚙️ How the Model Works

1. **Data Preprocessing**
   - Encodes categorical features: `AgeCategory`, `Race`, and `GenHealth` using `OneHotEncoder`.
   - Scales all numerical features using `StandardScaler`.
   - Produces 38 transformed numeric features.

2. **Model Training**
   - Trains two models: KNN and Decision Tree on the processed data.
   - Evaluates accuracy, precision, recall, and F1-score.
   - Saves the fitted models and preprocessing objects as `.pkl` files.

3. **Prediction UI**
   - Loads trained models and preprocessing pipelines.
   - Accepts user inputs for all relevant health features.
   - Transforms user input (encoding + scaling) before prediction.
   - Displays prediction result:  
     💖 **YEEAHHH!  Not likely to have heart disease**  
     💔 **OO NO!!!   Likely to have heart disease**

---

### 💻 How to Run the Project

#### 1️⃣ Install dependencies
```bash
pip install numpy pandas scikit-learn gradio
```

#### 2️⃣ Train the models
Run the `heart-disease-prediction_updated.ipynb` notebook:
- It preprocesses data.
- Trains both models.
- Saves the following files in the working directory:
  ```
  knn_model.pkl
  dt_model.pkl
  transformer.pkl
  scaler.pkl
  ```

#### 3️⃣ Launch the UI
Run the `HA_UI.ipynb` notebook:
- The Gradio interface will appear inline in Jupyter or open in your browser.
- Input values for all fields and choose either **K-Nearest Neighbors** or **Decision Tree**.
- View the heart disease prediction instantly.

---

### 🧠 Input Feature Guide

| Feature | Description | Valid Range / Values |
|----------|--------------|----------------------|
| **BMI** | Body Mass Index | 18–50 (float) |
| **Smoking** | Has smoked at least 100 cigarettes | 0 = No, 1 = Yes |
| **AlcoholDrinking** | Drinks alcohol heavily | 0 = No, 1 = Yes |
| **Stroke** | Ever had a stroke | 0 = No, 1 = Yes |
| **PhysicalHealth** | Days physical health was not good in past 30 days | 0–30 |
| **MentalHealth** | Days mental health was not good in past 30 days | 0–30 |
| **DiffWalking** | Difficulty walking or climbing stairs | 0 = No, 1 = Yes |
| **Sex** | Biological sex | 0 = Female, 1 = Male |
| **AgeCategory** | Age group | “18–24”, “25–29”, “30–34”, ..., “80+” |
| **Race** | Race or ethnicity | “White”, “Black”, “Asian”, “Hispanic”, etc. |
| **Diabetic** | Diagnosed with diabetes | 0 = No, 1 = Yes |
| **PhysicalActivity** | Performed physical activity in past 30 days | 0 = No, 1 = Yes |
| **GenHealth** | Self-rated general health | “Poor”, “Fair”, “Good”, “Very good”, “Excellent” |
| **SleepTime** | Average sleep hours per day | 4–12 |
| **Asthma** | Diagnosed with asthma | 0 = No, 1 = Yes |
| **KidneyDisease** | Diagnosed with kidney disease | 0 = No, 1 = Yes |
| **SkinCancer** | Diagnosed with skin cancer | 0 = No, 1 = Yes |

---

### 🎨 Gradio UI Features

- Clean, labeled interface with **tooltips** explaining each feature.
- Built-in validation for numeric ranges.
- Option to **choose prediction model**.
- Instant feedback showing likelihood of heart disease.

---

### 🧾 Example Workflow

1. Launch the Gradio UI.
2. Input data such as:
   - BMI: 27.5  
   - Smoking: 0  
   - Alcohol Drinking: 1  
   - Age Category: "55-59"  
   - GenHealth: "Good"
3. Choose model → *Decision Tree*
4. Output:  
   💖 *Not likely to have heart disease*

---

### 🏆 Acknowledgments
- Dataset Source: CDC Behavioral Risk Factor Surveillance System (BRFSS)
- Tools: Python, Scikit-learn, Pandas, Gradio, NumPy

---

### 💬 Authors
Developed by  
**Ayeshkant Mallick** & **Sandeep Kapoor**  
*(2025, Machine Learning Project on Heart Disease Prediction)*
