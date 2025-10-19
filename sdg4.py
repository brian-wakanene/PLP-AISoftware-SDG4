import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

print("🚀 SDG 4: Internet Coverage Prediction (World Bank Data)")

# 1. LOAD REAL WORLD BANK DATA
df_internet = pd.read_csv('internet.csv', skiprows=4)
df_gdp = pd.read_csv('gdp.csv', skiprows=4)

# 2. Get 2024 data
year = '2024'
df_internet = df_internet[['Country Name', year]].rename(columns={'Country Name':'Country', year:'Internet_%'})
df_gdp = df_gdp[['Country Name', year]].rename(columns={'Country Name':'Country', year:'GDP'})

# 3. MERGE + CLEAN (ALL 264 countries!)
df = pd.merge(df_internet, df_gdp, on='Country').dropna()
df['Internet_%'] = pd.to_numeric(df['Internet_%'], errors='coerce')
df['GDP'] = pd.to_numeric(df['GDP'], errors='coerce')
df = df.dropna()
print(f"✅ LOADED {len(df)} TOTAL countries!")

# 4. FORCE FIND KENYA 
kenya = df[df['Country'].str.contains('Kenya', case=False, na=False)]
print(f"🔍 KENYA FOUND: Row {kenya.index[0] if not kenya.empty else 'NOT FOUND'}")

# 5. TRAIN AI MODEL
X = df[['GDP']]
y = df['Internet_%']
model = RandomForestRegressor(n_estimators=10)
model.fit(X, y)

# 6. PREDICT POOR COUNTRIES
poor_gdps = [[500], [1000], [2000]]
predictions = model.predict(poor_gdps)

print("\n📊 2030 PREDICTIONS - POOR COUNTRIES:")
for gdp, pred in zip([500, 1000, 2000], predictions):
    gap = 100 - pred
    print(f"GDP ${gdp}: {pred:.0f}% coverage → NEEDS {gap:.0f}% more!")

# 7. KENYA PREDICTION 
if not kenya.empty:
    kenya_gdp = kenya['GDP'].iloc[0]
    kenya_internet = kenya['Internet_%'].iloc[0]
    kenya_gdp_2030 = kenya_gdp * 1.04**8
    kenya_pred_2030 = model.predict([[kenya_gdp_2030]])[0]
    kenya_gap = 100 - kenya_pred_2030
    print(f"\n🇰🇪 KENYA 2030 PREDICTION:")
    print(f"2022: {kenya_internet:.0f}% coverage, GDP ${kenya_gdp:,.0f}")
    print(f"2030: {kenya_pred_2030:.0f}% coverage, GDP ${kenya_gdp_2030:,.0f}")
    print(f"NEEDS: {kenya_gap:.0f}% more = {int(kenya_gap * 55000000 / 100):,} PEOPLE!")
else:
    print("\n⚠️ Kenya data not found")

# 8. CREATE CHART
plt.figure(figsize=(12, 7))
plt.scatter(df['GDP'], df['Internet_%'], color='blue', s=30, alpha=0.5, label='ALL Countries (2024)')
plt.scatter(poor_gdps, predictions, color='red', s=150, marker='X', label='Poor Countries 2030')
if not kenya.empty:
    plt.scatter([kenya_gdp], [kenya_internet], color='green', s=200, marker='*', label='Kenya 2024')

plt.xlabel('GDP per Person ($)')
plt.ylabel('Internet Coverage (%)')
plt.title('SDG 4: Predicting Internet Access for Poor Countries\n(World Bank Real Data)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('sdg4_final.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n🎉 CHART SAVED: sdg4_final.png")
