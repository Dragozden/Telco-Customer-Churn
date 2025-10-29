import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import copy
from IPython.display import display

#! ==================================================
#* 1.Wstępne przetwarzanie i czyszczenie danych (Data Cleaning & Preprocessing)
#* Cel: Uniknięcie błędów w dalszych etapach
#! ==================================================

#region 1.1 Wczytanie danych oraz ukazanie podstawowych informacji
#* Cel: Ustalenie z jakimi danymy mamy do czynienia i podjęcie odpowiedniej akcji

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
sns.set_theme() # applies better theme for seaborn plots

# * shows basic informations about our dataset
df.head()   # wyświetla 5 rekordów 
df.shape    # zwraca kształt danych - (7043, 21)
df.info()   # pokazuje typy danych dla każdej cechy
print(df['Churn'].value_counts(normalize=True)) # sprawdza bilans churn (73/26)

#endregion

#region 1.2 Przygotowanie danych do kolejnych kroków
#* Cel: Uzyskanie datasetu bez braków oraz zbędnych danych 

df.isnull().sum()   # zwraca liczbę brakujących danych dla kazdej z cech
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)    # pozbycie się wartości NaN
df.isnull().sum()   # liczba braków powinna wynosić 0
df = df.drop(columns=["customerID"]) # bez niej nie byłem w stanie wygenerować heatmapy

#endregion

#region 1.3 Tworzenie osobnych dataFrame dla porównań wpływu inżynierii cech
df_raw = copy.deepcopy(df)   # surowy dataset bez braków do porównania impactu data engineering
df_engineered_notEncoded = copy.deepcopy(df) # z inżyneria danych bez kodowania kategorii



#endregion

#! ==================================================
#* 2.Eksploracyjna Analiza Danych (EDA - Exploratory Data Analysis)
#* Cel: Poznanie oblicza danych
#! ==================================================

#region 2.1. Analiza zmiennej docelowej (rozkład Churn)
#* Cel: Uzyskanie wykresów charakterystyki danych w zależności od churn



#? Jakie jeszcze wykresy dodać i co ma na nich być?
fig, axs = plt.subplots(ncols=3)

df['tenure'].describe()
sns.histplot(df['tenure'], kde=True, ax=axs[0])

df['MonthlyCharges'].describe()
sns.histplot(df['MonthlyCharges'], kde=True, ax=axs[1])
#df['TotalCharges'].sample

df['TotalCharges'].describe()
sns.kdeplot(df['TotalCharges'],fill=True, ax=axs[2])
plt.show()

#endregion

#region 2.2. Analiza zmiennych numerycznych (rozkłady i boxploty vs Churn)
#* Cel: Uzyskanie wykresów charakterystyki danych numerycznych w zależności od churn

sns.countplot(
    data=df,
    x=df['InternetService'],
    hue=df['Churn'],
)
plt.title("Rezygnacja z usługi w zależności od typu łącza")
plt.ylabel("Ilość")
plt.show()

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.boxplot(
    data=df,
    x=df['tenure'],
    hue=df['Churn'],
)
plt.title("Rezygnacja z usługi w zależności od czasu posiadania usługi")
plt.ylabel("Ilość")

plt.subplot(1, 3, 2)
sns.histplot(
    data=df,
    x=df['tenure'],
    hue=df['Churn'],
)

plt.subplot(1, 3, 3)
sns.kdeplot(
    data=df,
    x=df['tenure'],
    hue=df['Churn'],
)
plt.show()
print("największe ryzyko churn jest w pierwszych 12 miesiącach")
print("Wraz ze wzrostem długości współpracy maleje churn")
print("Powyżej 70 miesięcy churn prawie nie występuje")

#! porównanie dla total charges

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.boxplot(
    data=df,
    x=df['TotalCharges'],
    hue=df['Churn'],
)
plt.title("Rezygnacja z usługi w zależności od całkowitych wydatków")
plt.ylabel("Ilość")

plt.subplot(1, 3, 2)
sns.histplot(
    data=df,
    x=df['TotalCharges'],
    hue=df['Churn'],
)

plt.subplot(1, 3, 3)
sns.kdeplot(
    data=df,
    x=df['TotalCharges'],
    hue=df['Churn'],
)
plt.show()
print("Wraz ze wzrostem wydatków maleje tendencja do churn")
print("Najwięcej churn jest gdy klienci mało wydają sumarycznie")
print("Mediana wydatków dla churn to ~450 a dla non churn to ~1800")

#! porównanie dla monthly charges

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.boxplot(
    data=df,
    x=df['MonthlyCharges'],
    hue=df['Churn'],
)
plt.title("Rezygnacja z usługi w zależności od MonthlyCharges")
plt.ylabel("Ilość")

plt.subplot(1, 3, 2)
sns.histplot(
    data=df,
    x=df['MonthlyCharges'],
    hue=df['Churn'],
)

plt.subplot(1, 3, 3)
sns.kdeplot(
    data=df,
    x=df['MonthlyCharges'],
    hue=df['Churn'],
)
plt.show()
print("Klienci z większymi opłatami niż 60 znacznie częściej rezygnują")
print("Klienci którzy nie rezygnują mają niskie opłaty miesięczne ~25")
print("Najwięcej churnujących występuje w przedziale 60-100")
print("Wysokie opłaty moga zniechęcać do współpracy i powodować churn")

#endregion

#region 2.3. Analiza zmiennych kategorycznych (countopoty z hue=Churn)
#* Cel: Uzyskanie wykresów charakterystyki danych w zależności od churn


# Lista kolumn do analizy
kolumny = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'
]


# wizualizacja zmiennych kategorycznych
categorical_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Tworzenie siatki wykresów
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18, 18))
fig.suptitle('Analiza Churn w zmiennych kategorycznych', fontsize=20)

# Iteracja przez zmienne i tworzenie wykresów
for i, feature in enumerate(categorical_features):
    row = i // 4
    col = i % 4
    sns.countplot(data=df, x=feature, hue='Churn', ax=axes[row, col])
    axes[row, col].set_title(f'Churn rate by {feature}')
    axes[row, col].tick_params(axis='x', rotation=45) # Lepsza czytelność etykiet

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#endregion

#region 2.4. Analiza korelacji (heatmapa i ekstrakcja top korelacji)
#* Cel: Odczytanie korelacji z mapy w celu ustalenia stymulantów oraz destymulantów churn

df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded.head(5)
corr = df_encoded.corr()
sns.heatmap(corr, annot=False, fmt=".2f", cmap='coolwarm', linewidths=.5)
plt.title("Mapa ciepła korelacji")
plt.show()

churn_col = 'Churn_Yes'
# Oblicz korelację wszystkich zmiennych z Churn
churn_corr = corr[churn_col]

# Usuń korelację cechy z samą sobą
churn_corr = churn_corr.drop(churn_col)

# Posortuj wartości od najwyższej do najniższej (bez wartości bezwzględnej)
churn_corr_sorted = churn_corr.sort_values(ascending=False)

print("TOP 5 CECH PROMUJĄCYCH CHURN (Korelacja dodatnia)")
print("--------------------------------------------------")
print("Im wyższa wartość cechy, tym większa szansa na rezygnację.")
print(churn_corr_sorted.head(5))

print("\nTOP 5 CECH CHRONIĄCYCH PRZED CHURNEM (Korelacja ujemna)")
print("-------------------------------------------------------")
print("Im wyższa wartość cechy, tym mniejsza szansa na rezygnację.")
print(churn_corr_sorted.tail(5))

#endregion

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df_encoded.head()
# Przygotuj dane
X = df_encoded.drop(columns=['Churn_Yes'])
y = df_encoded['Churn_Yes']

# Podziel dane
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Wytrenuj Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Analiza ważności cech
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 najważniejszych cech dla przewidywania Churn (Random Forest):")
print(feature_importance.head(15))

# Wizualizacja ważności cech
plt.figure(figsize=(10, 8))
top_15_features = feature_importance.head(15)
sns.barplot(x='importance', y='feature', data=top_15_features)
plt.title('Top 15 najważniejszych cech dla przewidywania Churn')
plt.tight_layout()
plt.show()

#! ==================================================
#* 3.INŻYNIERIA CECH I PRZYGOTOWANIE DO MODELOWANIA
#* Cel: Ulepszenie danych, aby model lepiej się uczył
#! ==================================================

#region 3.1 Tworzenie nowych cech 


# Cecha - kategorie lojalności zamiast liczb
# rozbijamy wedle kwandtyli aby uzyskać równomiernie rozłożone dane


#endregion
# Cecha 1: Kategorie lojalności na podstawie kwantyli stażu
bins = [
    df_engineered_notEncoded['tenure'].min(),
    df_engineered_notEncoded['tenure'].quantile(0.25),
    df_engineered_notEncoded['tenure'].quantile(0.75),
    df_engineered_notEncoded['tenure'].max()
]
labels = ['low_loyalty', 'medium_loyalty', 'high_loyalty']
df_engineered_notEncoded['Client_loyalty'] = pd.cut(df_engineered_notEncoded['tenure'], bins=bins, labels=labels, include_lowest=True)

# Cecha 2: Zliczenie usług dodatkowych (metoda zwektoryzowana)
service_columns = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
df_engineered_notEncoded['AdditionalServices'] = (df_engineered_notEncoded[service_columns] == 'Yes').sum(axis=1)

# Cecha 3: Stosunek miesięcznych opłat do stażu klienta
# Unikamy dzielenia przez zero dla nowych klientów (tenure=0) dodając małą wartość
df_engineered_notEncoded['ChargePer_TenureLength'] = df_engineered_notEncoded['MonthlyCharges'] / (df_engineered_notEncoded['tenure'] + 1)

# --- Kluczowy krok: Usunięcie oryginalnych kolumn, które zostały zastąpione/zagregowane ---
columns_to_drop = ['tenure'] + service_columns
df_engineered_notEncoded.drop(columns=columns_to_drop, inplace=True)

#endregion


#region 3.2 Kodowanie zmiennych kategorycznych

df_engineered_encoded = pd.get_dummies(df_engineered_notEncoded, drop_first=True)
df_engineered_encoded.head()
df_encoded.head()

#endregion

#region 3.2 Import bibliotek do modelowania

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#endregion

#region 3.3 Przygotowanie zbiorów danych
target_var = 'Churn_Yes'

X_original = df_encoded.drop(columns=[target_var])
y_original = df_encoded[target_var]

X_engineered = df_engineered_encoded.drop(columns=[target_var])
y_engineered = df_engineered_encoded[target_var]
#endregion

assert y_original.equals(y_engineered), "Wektory celu nie są identyczne! Sprawdź dane."


#region 3.4 Podział na zbiór treningowy i testowy

# Ustawienie stałych parametrów dla powtarzalności eksperymentu
TEST_SIZE = 0.25
RANDOM_STATE = 42

# podział zbioru orginalnego
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_original,
    y_original,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_original,
    )

# podział zbioru engineered
X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
    X_engineered,
    y_engineered,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_original,
    )

print(f"Rozmiar zbioru treningowego (oryginalny): {X_train_orig.shape}")
print(f"Rozmiar zbioru testowego (zmodyfikowany): {X_test_eng.shape}")
#endregion

#region 3.5 Trenowanie modeli oraz zapisywanie wyników

def train_and_access_model(model, X_train, X_test, y_train, y_test):
    
    # train the model
    model.fit(X_train, y_train)

    # predict from test data
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1] # predict probability for positive cases, needed for ROC metric

    # measure how well model represents data
    results = {
        'Accuracy' : accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba)
    }

    return results

# wybór modelu
model = LogisticRegression(max_iter=1000)
different_models = []

print("Trainning oringinal model...")
original_model = train_and_access_model(model, X_train_orig, X_test_orig, y_train_orig, y_test_orig)
original_model['Model'] = "Base model - without feature engineering"
different_models.append(original_model)
print("Training done!")
model_eng = LogisticRegression(max_iter=1000)
print("Trainning enhanced model...")
engineered_model = train_and_access_model(model_eng, X_train_eng, X_test_eng, y_train_eng, y_test_eng )
engineered_model['Model'] = "Engineered model - used feature engineering"
different_models.append(engineered_model)
print("Training done!")

df_results = pd.DataFrame(different_models).set_index('Model')

print("Model results")
display(df_results)
#endregion