# ⚽ Soccer Match Outcome Prediction Using Historical Betting Data

A collaborative machine learning project exploring how historical betting odds can be used to predict soccer match outcomes. We trained and evaluated multiple classification models using PySpark on Databricks, with the goal of identifying winning patterns in Premier League matches.

---

## 📌 Project Overview

This project was developed as part of a graduate-level course (CIS 5560 - Data Analytics) at Cal State LA. Our objective was to apply real-world data science techniques using Spark and Python to predict the outcome of soccer matches (home win vs. not-home-win) based on bookmaker odds.

Using a cleaned dataset of EPL match results and betting odds, we built and compared four different classifiers. Along the way, we explored data transformation with Spark SQL, evaluated models with AUC and accuracy, and tuned hyperparameters with cross-validation.

---

## 🎯 Goals

- Load and process soccer betting data directly from GitHub
- Engineer features and reformat labels for binary classification
- Train Logistic Regression, Decision Tree, Random Forest, and GBT classifiers
- Compare performance using accuracy, AUC, and confusion matrix
- Tune model parameters and interpret feature importance

---

## 🧰 Tech Stack

- **Language**: Python 3.9 (Databricks Runtime 11.3 LTS ML)
- **Framework**: PySpark (Spark MLlib)
- **Environment**: Databricks on AWS (2 × i3.xlarge nodes)
- **Tools**: Pandas, Spark SQL, VectorAssembler, BinaryClassificationEvaluator

---

## 🧪 Dataset

- **Source**: [Football Data UK](https://www.football-data.co.uk/englandm.php)
- **Matches**: 225 EPL fixtures (2023/24 season)
- **Features Used**:
  - `B365H`: Home win odds
  - `B365D`: Draw odds
  - `B365A`: Away win odds
- **Target Label**: 1 if Home Win, 0 otherwise

Data was pulled directly from GitHub in `.csv` format and converted into Spark DataFrames.

---

## 🚀 Models Trained

| Model                 | AUC   | Accuracy |
|----------------------|-------|----------|
| Logistic Regression  | 0.730 | 0.721    |
| Random Forest        | 0.672 | 0.559    |
| Decision Tree        | 0.569 | 0.574    |
| Gradient Boosted Tree| 0.550 | 0.574    |

- Logistic Regression performed best, slightly beating Random Forest in both accuracy and AUC.
- Random Forest feature importance ranked: `B365H` > `B365D` > `B365A`.

We also used cross-validation on Random Forest, optimizing the number of trees and tree depth.

---

## 🧠 Key Learnings

- **Feature Engineering Matters**: Simple odds data can drive surprisingly effective models.
- **Draws Are Hard to Predict**: Binary classification (Win vs. Not-Win) was more stable than multiclass.
- **Spark Makes It Scalable**: Running models on distributed compute made iteration faster — total pipeline runtime was under 5 minutes.
- **Random Forest Doesn’t Always Win**: Sometimes, simpler models like Logistic Regression outperform when data is clean and features are well-behaved.

---

## 📂 Project Structure

── data/
│ └── E0_cleaned.csv
├── notebooks/
│ ├── databricks_logreg.ipynb
│ └── zeppelin_experiments.ipynb
├── README.md
└── requirements.txt (coming soon)


---

## 🖥️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/soccer-betting-ml.git
   cd soccer-betting-ml

2. Use Databricks (or Zeppelin) to open and run the notebooks.

3. Load the data from - https://raw.githubusercontent.com/ryumatsu/soccer-betting-ml/main/E0_cleaned.csv

4. Follow the step-by-step cells to run preprocessing, training, and evaluation. (You’ll need access to Databricks or a Spark-enabled environment to run the notebooks.

)

## 📈 Visuals & Results
Sample output from the best-performing model:
Confusion Matrix:
[88. 20.]
[23. 94.]

Test Accuracy: ~72%
AUC Score: 0.730

## 👥 Contributors
This project was created by:

Atharva Borole – GitHub

Daniel Ozorio Delfin

Ryuichi Matsushita

Instructor: Dr. Jongwook Woo, Cal State LA

## 📄 References
Raw Dataset: https://raw.githubusercontent.com/ryumatsu/soccer-betting-ml/main/E0_cleaned.csv

Project Paper: [add link here if you upload the PDF]

Data Source: https://www.football-data.co.uk/englandm.php

Original GitHub Repo: https://github.com/ryumatsu/soccer-betting-ml 

