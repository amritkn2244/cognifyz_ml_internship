
# Task 4: Location-based Analysis
# Cognifyz ML Internship
# Name: Amrit Kumar Nayak | Ref: CTI/A1/C367837

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Dataset.csv', encoding='utf-8-sig')
print("=" * 50)
print("TASK 4: LOCATION-BASED ANALYSIS")
print("=" * 50)
print(f"\nDataset shape: {df.shape}")

print(f"\nLatitude  — min: {df['Latitude'].min():.4f}, max: {df['Latitude'].max():.4f}")
print(f"Longitude — min: {df['Longitude'].min():.4f}, max: {df['Longitude'].max():.4f}")

df_geo = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)].copy()
print(f"Restaurants with valid coordinates: {df_geo.shape[0]}")

city_stats = df.groupby('City').agg(
    Restaurant_Count=('Restaurant ID', 'count'),
    Avg_Rating=('Aggregate rating', 'mean'),
    Avg_Cost=('Average Cost for two', 'mean'),
    Avg_Votes=('Votes', 'mean')
).reset_index()

city_stats = city_stats[city_stats['Avg_Rating'] > 0]
city_stats = city_stats.sort_values('Restaurant_Count', ascending=False)

top_cities = city_stats.head(10)
print("\nTop 10 Cities by Restaurant Count:")
print("-" * 70)
print(f"{'City':<25} {'Count':>7} {'Avg Rating':>10} {'Avg Cost':>10} {'Avg Votes':>10}")
print("-" * 70)
for _, row in top_cities.iterrows():
    print(f"{row['City']:<25} {row['Restaurant_Count']:>7} {row['Avg_Rating']:>10.2f} {row['Avg_Cost']:>10.0f} {row['Avg_Votes']:>10.0f}")

print("\nTop 5 Cities by Average Rating:")
top_rated = city_stats[city_stats['Restaurant_Count'] >= 10].nlargest(5, 'Avg_Rating')
for _, row in top_rated.iterrows():
    print(f"  {row['City']:<25} {row['Avg_Rating']:.2f}")

print("\nTop 5 Most Expensive Cities (Avg Cost for two):")
top_cost = city_stats[city_stats['Restaurant_Count'] >= 10].nlargest(5, 'Avg_Cost')
for _, row in top_cost.iterrows():
    print(f"  {row['City']:<25} {row['Avg_Cost']:.0f}")

print("\nTop 5 Cities by Average Votes:")
top_votes = city_stats[city_stats['Restaurant_Count'] >= 10].nlargest(5, 'Avg_Votes')
for _, row in top_votes.iterrows():
    print(f"  {row['City']:<25} {row['Avg_Votes']:.0f}")

print("\n--- Insights ---")
total_cities = df['City'].nunique()
total_countries = df['Country Code'].nunique()
print(f"Total unique cities    : {total_cities}")
print(f"Total unique countries : {total_countries}")

most_restaurants = city_stats.iloc[0]
print(f"City with most restaurants : {most_restaurants['City']} ({most_restaurants['Restaurant_Count']})")

highest_rated = city_stats[city_stats['Restaurant_Count'] >= 10].nlargest(1, 'Avg_Rating').iloc[0]
print(f"Highest rated city (min 10 restaurants) : {highest_rated['City']} ({highest_rated['Avg_Rating']:.2f})")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Task 4: Location-based Restaurant Analysis', fontsize=15, fontweight='bold')

axes[0, 0].scatter(
    df_geo['Longitude'], df_geo['Latitude'],
    alpha=0.3, s=5, color='steelblue'
)
axes[0, 0].set_title('Geographic Distribution of Restaurants')
axes[0, 0].set_xlabel('Longitude')
axes[0, 0].set_ylabel('Latitude')

top_cities_plot = city_stats.head(10).sort_values('Restaurant_Count')
axes[0, 1].barh(top_cities_plot['City'], top_cities_plot['Restaurant_Count'], color='steelblue')
axes[0, 1].set_title('Top 10 Cities by Restaurant Count')
axes[0, 1].set_xlabel('Number of Restaurants')

top_cities_rating = city_stats[city_stats['Restaurant_Count'] >= 10].nlargest(10, 'Avg_Rating').sort_values('Avg_Rating')
axes[1, 0].barh(top_cities_rating['City'], top_cities_rating['Avg_Rating'], color='coral')
axes[1, 0].set_title('Top 10 Cities by Average Rating\n(min. 10 restaurants)')
axes[1, 0].set_xlabel('Average Rating')
axes[1, 0].set_xlim(0, 5)

top_cities_cost = city_stats[city_stats['Restaurant_Count'] >= 10].nlargest(10, 'Avg_Cost').sort_values('Avg_Cost')
axes[1, 1].barh(top_cities_cost['City'], top_cities_cost['Avg_Cost'], color='mediumseagreen')
axes[1, 1].set_title('Top 10 Most Expensive Cities\n(Avg Cost for Two)')
axes[1, 1].set_xlabel('Average Cost for Two')

plt.tight_layout()
plt.savefig('task4_output.png', dpi=150, bbox_inches='tight')
print("\nPlot saved → task4_output.png")
plt.show()

print("\nTask 4 Complete!")