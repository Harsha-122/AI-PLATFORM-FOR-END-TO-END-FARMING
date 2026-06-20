import pandas as pd
import numpy as np
import joblib, os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

read=pd.read_csv("data_core.csv")
os.makedirs('models', exist_ok=True)
np.random.seed(42)
N = 8000

# ── Shared Category Lists ──────────────────────────────────
SOIL_TYPES  = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
CROP_TYPES  = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy',
               'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']
FERTILIZERS = ['Urea', 'DAP', '14-35-14', '28-28', '17-17-17',
                '20-20', '10-26-26', 'TSP', 'SSP', 'Potash']

# MODEL 1 — FERTILIZER RECOMMENDATION
# Features: Temp, Humidity, Moisture, Soil, Crop, N, K, P
# Target  : Fertilizer Name

print("\n[1/5] Training Fertilizer Recommendation Model...")

df1 = pd.DataFrame({
    'Temperature' : np.random.randint(15, 45, N),
    'Humidity'    : np.random.randint(50, 100, N),
    'Moisture'    : np.random.randint(10, 80, N),
    'Soil_Type'   : np.random.choice(SOIL_TYPES, N),
    'Crop_Type'   : np.random.choice(CROP_TYPES, N),
    'Nitrogen'    : np.random.randint(0, 80, N),
    'Potassium'   : np.random.randint(0, 80, N),
    'Phosphorous' : np.random.randint(0, 80, N),
})
# Rule-based target so model learns real patterns
conditions = [
    (df1['Nitrogen'] < 20) & (df1['Phosphorous'] < 20),
    (df1['Nitrogen'] > 50),
    (df1['Potassium'] > 50),
    (df1['Phosphorous'] > 50),
    (df1['Moisture'] < 25),
]
choices = ['Urea', 'DAP', 'Potash', 'TSP', '20-20']
df1['Fertilizer'] = np.select(conditions, choices, default='17-17-17')

le1_soil   = LabelEncoder().fit(SOIL_TYPES)
le1_crop   = LabelEncoder().fit(CROP_TYPES)
le1_target = LabelEncoder().fit(df1['Fertilizer'])

X1 = df1[['Temperature','Humidity','Moisture',
          'Soil_Type','Crop_Type','Nitrogen','Potassium','Phosphorous']].copy()
X1['Soil_Type'] = le1_soil.transform(X1['Soil_Type'])
X1['Crop_Type'] = le1_crop.transform(X1['Crop_Type'])
y1 = le1_target.transform(df1['Fertilizer'])

sc1 = StandardScaler()
X1s = sc1.fit_transform(X1)
Xtr,Xte,ytr,yte = train_test_split(X1s,y1,test_size=0.2,random_state=42)
m1 = GradientBoostingClassifier(n_estimators=150, random_state=42).fit(Xtr,ytr)
print(f"   ✅ Accuracy: {accuracy_score(yte, m1.predict(Xte))*100:.1f}%")

joblib.dump({'model':m1,'scaler':sc1,'le_soil':le1_soil,
             'le_crop':le1_crop,'le_target':le1_target,
             'features':['Temperature','Humidity','Moisture','Soil_Type',
                         'Crop_Type','Nitrogen','Potassium','Phosphorous']},
            'models/fertilizer_model.pkl')


# ════════════════════════════════════════════════════════════
# MODEL 2 — CROP RECOMMENDATION
# Features: N, P, K, Temperature, Humidity, pH, Rainfall
# Target  : Crop Name
# ════════════════════════════════════════════════════════════
print("\n[2/5] Training Crop Recommendation Model...")

df2 = pd.DataFrame({
    'Nitrogen'    : np.random.randint(0, 140, N),
    'Phosphorous' : np.random.randint(5, 145, N),
    'Potassium'   : np.random.randint(5, 205, N),
    'Temperature' : np.random.uniform(8, 44, N),
    'Humidity'    : np.random.uniform(14, 100, N),
    'pH'          : np.random.uniform(3.5, 9.5, N),
    'Rainfall'    : np.random.uniform(20, 300, N),
})
# Rule-based patterns
cond2 = [
    (df2['Rainfall'] > 200) & (df2['Humidity'] > 80),
    (df2['Temperature'] > 35) & (df2['Nitrogen'] > 80),
    (df2['pH'] < 5.5),
    (df2['Potassium'] > 150),
    (df2['Phosphorous'] > 100) & (df2['Temperature'].between(20,30)),
    (df2['Nitrogen'] < 30) & (df2['Rainfall'] < 60),
]
ch2 = ['Paddy','Maize','Ground Nuts','Cotton','Wheat','Millets']
df2['Crop'] = np.select(cond2, ch2, default='Pulses')

le2_target = LabelEncoder().fit(df2['Crop'])
X2 = df2[['Nitrogen','Phosphorous','Potassium','Temperature','Humidity','pH','Rainfall']]
y2 = le2_target.transform(df2['Crop'])

sc2 = StandardScaler()
X2s = sc2.fit_transform(X2)
Xtr,Xte,ytr,yte = train_test_split(X2s,y2,test_size=0.2,random_state=42)
m2 = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr,ytr)
print(f"   ✅ Accuracy: {accuracy_score(yte, m2.predict(Xte))*100:.1f}%")

joblib.dump({'model':m2,'scaler':sc2,'le_target':le2_target,
             'features':['Nitrogen','Phosphorous','Potassium',
                         'Temperature','Humidity','pH','Rainfall']},
            'models/crop_model.pkl')


# ════════════════════════════════════════════════════════════
# MODEL 3 — SOIL FERTILITY SCORE
# Features: N, P, K, pH, Organic_Matter, Moisture
# Target  : Low / Medium / High
# ════════════════════════════════════════════════════════════
print("\n[3/5] Training Soil Fertility Model...")

df3 = pd.DataFrame({
    'Nitrogen'       : np.random.randint(0, 100, N),
    'Phosphorous'    : np.random.randint(0, 100, N),
    'Potassium'      : np.random.randint(0, 100, N),
    'pH'             : np.random.uniform(4.0, 9.0, N),
    'Organic_Matter' : np.random.uniform(0.1, 6.0, N),
    'Moisture'       : np.random.randint(10, 80, N),
})
score = (
    (df3['Nitrogen']/100)*30 +
    (df3['Phosphorous']/100)*20 +
    (df3['Potassium']/100)*20 +
    ((1 - abs(df3['pH'] - 6.5)/3))*15 +
    (df3['Organic_Matter']/6)*15
)
df3['Fertility'] = pd.cut(score, bins=[0,40,70,100], labels=['Low','Medium','High'])

le3_target = LabelEncoder().fit(['Low','Medium','High'])
X3 = df3[['Nitrogen','Phosphorous','Potassium','pH','Organic_Matter','Moisture']]
y3 = le3_target.transform(df3['Fertility'])

sc3 = StandardScaler()
X3s = sc3.fit_transform(X3)
Xtr,Xte,ytr,yte = train_test_split(X3s,y3,test_size=0.2,random_state=42)
m3 = GradientBoostingClassifier(n_estimators=150, random_state=42).fit(Xtr,ytr)
print(f"   ✅ Accuracy: {accuracy_score(yte, m3.predict(Xte))*100:.1f}%")

joblib.dump({'model':m3,'scaler':sc3,'le_target':le3_target,
             'features':['Nitrogen','Phosphorous','Potassium','pH','Organic_Matter','Moisture']},
            'models/fertility_model.pkl')


# ════════════════════════════════════════════════════════════
# MODEL 4 — IRRIGATION WATER REQUIREMENT
# Features: Temp, Humidity, Moisture, Soil Type, Crop Type, Rainfall
# Target  : Low / Moderate / High
# ════════════════════════════════════════════════════════════
print("\n[4/5] Training Irrigation Requirement Model...")

df4 = pd.DataFrame({
    'Temperature' : np.random.randint(15, 45, N),
    'Humidity'    : np.random.randint(30, 100, N),
    'Moisture'    : np.random.randint(5, 80, N),
    'Rainfall'    : np.random.uniform(0, 200, N),
    'Soil_Type'   : np.random.choice(SOIL_TYPES, N),
    'Crop_Type'   : np.random.choice(CROP_TYPES, N),
})
water_need = (
    (df4['Temperature']/45)*40 +
    ((100 - df4['Humidity'])/100)*25 +
    ((80 - df4['Moisture'])/80)*25 +
    ((200 - df4['Rainfall'])/200)*10
)
df4['Water_Need'] = pd.cut(water_need, bins=[0,35,65,100],
                            labels=['Low','Moderate','High'])

le4_soil   = LabelEncoder().fit(SOIL_TYPES)
le4_crop   = LabelEncoder().fit(CROP_TYPES)
le4_target = LabelEncoder().fit(['Low','Moderate','High'])

X4 = df4[['Temperature','Humidity','Moisture','Rainfall','Soil_Type','Crop_Type']].copy()
X4['Soil_Type'] = le4_soil.transform(X4['Soil_Type'])
X4['Crop_Type'] = le4_crop.transform(X4['Crop_Type'])
y4 = le4_target.transform(df4['Water_Need'])

sc4 = StandardScaler()
X4s = sc4.fit_transform(X4)
Xtr,Xte,ytr,yte = train_test_split(X4s,y4,test_size=0.2,random_state=42)
m4 = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr,ytr)
print(f"   ✅ Accuracy: {accuracy_score(yte, m4.predict(Xte))*100:.1f}%")

joblib.dump({'model':m4,'scaler':sc4,'le_soil':le4_soil,'le_crop':le4_crop,
             'le_target':le4_target,
             'features':['Temperature','Humidity','Moisture','Rainfall','Soil_Type','Crop_Type']},
            'models/irrigation_model.pkl')


# ════════════════════════════════════════════════════════════
# MODEL 5 — CROP DISEASE RISK
# Features: Temp, Humidity, Rainfall, Moisture, Crop Type
# Target  : Low / Medium / High Risk
# ════════════════════════════════════════════════════════════
print("\n[5/5] Training Disease Risk Model...")

df5 = pd.DataFrame({
    'Temperature' : np.random.randint(10, 45, N),
    'Humidity'    : np.random.randint(30, 100, N),
    'Rainfall'    : np.random.uniform(0, 250, N),
    'Moisture'    : np.random.randint(10, 90, N),
    'Crop_Type'   : np.random.choice(CROP_TYPES, N),
})
risk_score = (
    ((df5['Humidity'] - 30)/70)*40 +
    (df5['Rainfall']/250)*30 +
    (df5['Moisture']/90)*20 +
    (abs(df5['Temperature'] - 25)/20)*10
)
df5['Disease_Risk'] = pd.cut(risk_score, bins=[0,33,66,100],
                              labels=['Low','Medium','High'])

le5_crop   = LabelEncoder().fit(CROP_TYPES)
le5_target = LabelEncoder().fit(['Low','Medium','High'])

X5 = df5[['Temperature','Humidity','Rainfall','Moisture','Crop_Type']].copy()
X5['Crop_Type'] = le5_crop.transform(X5['Crop_Type'])
y5 = le5_target.transform(df5['Disease_Risk'])

sc5 = StandardScaler()
X5s = sc5.fit_transform(X5)
Xtr,Xte,ytr,yte = train_test_split(X5s,y5,test_size=0.2,random_state=42)
m5 = GradientBoostingClassifier(n_estimators=150, random_state=42).fit(Xtr,ytr)
print(f"✅ Accuracy: {accuracy_score(yte, m5.predict(Xte))*100:.1f}%")

joblib.dump({'model':m5,'scaler':sc5,'le_crop':le5_crop,'le_target':le5_target,
             'features':['Temperature','Humidity','Rainfall','Moisture','Crop_Type']},
            'models/disease_model.pkl')


print("\n" + "="*60)
print("  ✅ ALL 5 MODELS SAVED TO /models/")
print("="*60)
print("""
  models/fertilizer_model.pkl   → Fertilizer Recommendation
  models/crop_model.pkl         → Crop Recommendation
  models/fertility_model.pkl    → Soil Fertility Score
  models/irrigation_model.pkl   → Irrigation Requirement
  models/disease_model.pkl      → Disease Risk Prediction
  
  Now run:  python app.py
""")