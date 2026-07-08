
# Task 1: Predict Restaurant Ratings
# Cognifyz ML Internship
# Name: Amrit Kumar Nayak | Ref: CTI/A1/C367837


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Dataset.csv', encoding='latin1')
print("=" * 50)
print("TASK 1: PREDICT RESTAURANT RATINGS")
print("=" * 50)
print(f"\nDataset shape: {df.shape}")

df = df[df['Aggregate rating'] > 0].copy()
print(f"After removing unrated restaurants: {df.shape}")

df.dropna(subset=['Cuisines'], inplace=True)

le = LabelEncoder()
df['Has Table booking'] = le.fit_transform(df['Has Table booking'])
df['Has Online delivery'] = le.fit_transform(df['Has Online delivery'])
df['Is delivering now'] = le.fit_transform(df['Is delivering now'])

features = [
    'Country Code', 'Average Cost for two', 'Has Table booking',
    'Has Online delivery', 'Is delivering now', 'Price range', 'Votes'
]
target = 'Aggregate rating'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")

models = {
    'Linear Regression'    : LinearRegression(),
    'Decision Tree'        : DecisionTreeRegressor(random_state=42),
    'Random Forest'        : RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
print("\n" + "-" * 50)
print(f"{'Model':<25} {'MSE':>8} {'R² Score':>10}")
print("-" * 50)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds, 'mse': mse, 'r2': r2}
    print(f"{name:<25} {mse:>8.4f} {r2:>10.4f}")

print("-" * 50)

best_name = max(results, key=lambda x: results[x]['r2'])
best = results[best_name]
print(f"\nBest Model: {best_name}")
print(f"  MSE      : {best['mse']:.4f}")
print(f"  R² Score : {best['r2']:.4f}")

rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

print("\nFeature Importances (Random Forest):")
for feat, imp in importances.items():
    print(f"  {feat:<30} {imp:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Task 1: Predict Restaurant Ratings', fontsize=14, fontweight='bold')

axes[0].scatter(y_test, best['preds'], alpha=0.4, color='steelblue', edgecolors='none', s=20)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=1.5)
axes[0].set_xlabel('Actual Rating')
axes[0].set_ylabel('Predicted Rating')
axes[0].set_title(f'Actual vs Predicted ({best_name})\nR² = {best["r2"]:.4f}')

importances.plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title('Feature Importances (Random Forest)')
axes[1].set_xlabel('Importance Score')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('task1_output.png', dpi=150, bbox_inches='tight')
print("\nPlot saved → task1_output.png")
plt.show()

print("\nTask 1 Complete!")