
# Task 3: Cuisine Classification
# Cognifyz ML Internship
# Name: Amrit Kumar Nayak | Ref: CTI/A1/C367837

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('Dataset.csv', encoding='latin1')
print("=" * 50)
print("TASK 3: CUISINE CLASSIFICATION")
print("=" * 50)
print(f"\nDataset shape: {df.shape}")

df.dropna(subset=['Cuisines'], inplace=True)

df['Primary Cuisine'] = df['Cuisines'].apply(lambda x: x.split(',')[0].strip())

top_cuisines = df['Primary Cuisine'].value_counts().head(10).index
df = df[df['Primary Cuisine'].isin(top_cuisines)].copy()
print(f"\nTop 10 cuisines selected: {df.shape[0]} restaurants")
print("\nCuisine distribution:")
print(df['Primary Cuisine'].value_counts().to_string())

le_cuisine = LabelEncoder()
df['Cuisine Label'] = le_cuisine.fit_transform(df['Primary Cuisine'])

le = LabelEncoder()
df['Has Table booking']    = le.fit_transform(df['Has Table booking'])
df['Has Online delivery']  = le.fit_transform(df['Has Online delivery'])
df['Is delivering now']    = le.fit_transform(df['Is delivering now'])

features = [
    'Country Code', 'Average Cost for two', 'Has Table booking',
    'Has Online delivery', 'Price range', 'Aggregate rating', 'Votes'
]

X = df[features]
y = df['Cuisine Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

models = {
    'Logistic Regression' : LogisticRegression(max_iter=10000, solver='saga', random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
print("\n" + "-" * 55)
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
print("-" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec  = recall_score(y_test, preds, average='weighted', zero_division=0)
    results[name] = {'model': model, 'preds': preds, 'acc': acc, 'prec': prec, 'rec': rec}
    print(f"{name:<25} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f}")

print("-" * 55)

best_name = max(results, key=lambda x: results[x]['acc'])
best = results[best_name]
print(f"\nBest Model: {best_name}")
print(f"  Accuracy  : {best['acc']:.4f}")
print(f"  Precision : {best['prec']:.4f}")
print(f"  Recall    : {best['rec']:.4f}")

print(f"\nClassification Report ({best_name}):")
print(classification_report(y_test, best['preds'],
      target_names=le_cuisine.classes_, zero_division=0))

rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

print("Feature Importances (Random Forest):")
for feat, imp in importances.items():
    print(f"  {feat:<30} {imp:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Task 3: Cuisine Classification', fontsize=14, fontweight='bold')

cuisine_counts = df['Primary Cuisine'].value_counts()
cuisine_counts.plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Top 10 Cuisines by Restaurant Count')
axes[0].set_xlabel('Number of Restaurants')
axes[0].invert_yaxis()

importances.plot(kind='barh', ax=axes[1], color='coral')
axes[1].set_title('Feature Importances (Random Forest)')
axes[1].set_xlabel('Importance Score')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('task3_output.png', dpi=150, bbox_inches='tight')
print("\nPlot saved → task3_output.png")
plt.show()

print("\nTask 3 Complete!")