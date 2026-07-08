# Cognifyz ML Internship — Amrit Kumar Nayak

Machine Learning Internship tasks completed at **Cognifyz Technologies Pvt. Ltd.** Tasks 1, 3, and 4 were selected.
Ref: CTI/A1/C367837

---

## Dataset

The dataset contains **9,551 restaurants** with the following features:
- Restaurant name, city, country, address, locality
- Latitude & longitude coordinates
- Cuisines, average cost for two, price range
- Has table booking, has online delivery
- Aggregate rating, votes

---

## Tasks Completed

### Task 1 — Predict Restaurant Ratings
**Objective:** Build a regression model to predict the aggregate rating of a restaurant.

- Removed unrated restaurants (rating = 0), leaving 7,403 clean records
- Encoded categorical features using Label Encoding
- Trained and compared 3 models — Linear Regression, Decision Tree, Random Forest
- **Random Forest** performed best with R² = 0.45 and MSE = 0.169
- **Votes** was the most influential feature with an importance score of 0.70

**Output:** `task1_output.png` — Actual vs Predicted scatter plot + Feature Importance chart

---

### Task 3 — Cuisine Classification
**Objective:** Classify restaurants based on their primary cuisine type.

- Extracted primary cuisine from multi-cuisine entries
- Filtered to top 10 cuisines — 7,014 restaurants
- Used StandardScaler for feature scaling
- Trained and compared Logistic Regression and Random Forest
- **Logistic Regression** achieved 44.9% accuracy
- Low accuracy is expected due to heavy class imbalance — North Indian cuisine alone makes up 43% of the dataset
- **Votes, Rating, and Average Cost** were the top predictive features

**Output:** `task3_output.png` — Cuisine distribution + Feature Importance chart

---

### Task 4 — Location-based Analysis
**Objective:** Perform geographical analysis of restaurants in the dataset.

- Filtered 9,052 restaurants with valid GPS coordinates
- Analyzed 141 cities across 15 countries
- **New Delhi** has the most restaurants (5,473)
- **London** has the highest average rating (4.54)
- **Bangalore** has the highest average votes per restaurant (2,806)
- Visualized restaurant distribution, ratings, and costs by city

**Output:** `task4_output.png` — 4-panel chart with geographic scatter, city counts, ratings, and costs

---

## How to Run

**1. Install dependencies:**
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

**2. Place `Dataset.csv` in the same folder as the scripts**

**3. Run any task:**
```
python task1_predict_ratings.py
python task3_cuisine_classification.py
python task4_location_analysis.py
```

---

## Repository Structure

```
cognifyz-ml-internship/
│
├── Dataset.csv
├── README.md
│
├── task1_predict_ratings.py
├── task1_output.png
│
├── task3_cuisine_classification.py
├── task3_output.png
│
├── task4_location_analysis.py
└── task4_output.png
```

---

## Tools & Libraries

- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

*Internship offered by Cognifyz Technologies Pvt. Ltd., Nagpur, Maharashtra, India*
