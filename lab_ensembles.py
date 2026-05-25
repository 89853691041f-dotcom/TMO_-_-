# =============================================================
# Лабораторная работа: Ансамбли моделей машинного обучения
# Датасет: Titanic (задача классификации)
# =============================================================

# ШАГ 1: Установка и импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Ансамблевые модели
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from sklearn.tree import DecisionTreeClassifier

# Чтобы результаты были воспроизводимы
RANDOM_STATE = 42

# =============================================================
# ШАГ 2: Загрузка датасета Titanic
# =============================================================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("=== Первые 5 строк датасета ===")
print(df.head())
print()
print("=== Информация о датасете ===")
print(df.info())
print()
print("=== Пропущенные значения ===")
print(df.isnull().sum())

# =============================================================
# ШАГ 3: Предобработка данных
# =============================================================

# Оставляем только нужные признаки
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Заполняем пропуски
df['Age'].fillna(df['Age'].median(), inplace=True)       # возраст — медианой
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # порт — модой

print("=== Пропуски после заполнения ===")
print(df.isnull().sum())

# Кодируем категориальные признаки
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])           # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked']) # C=0, Q=1, S=2

print()
print("=== Данные после обработки (первые 5 строк) ===")
print(df.head())

# =============================================================
# ШАГ 4: Разделение на обучающую и тестовую выборки
# =============================================================
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"\nОбучающая выборка: {X_train.shape[0]} примеров")
print(f"Тестовая выборка:  {X_test.shape[0]} примеров")

# =============================================================
# ШАГ 5: Обучение ансамблевых моделей
# =============================================================

# --- 1. Бэггинг (Bagging) ---
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=RANDOM_STATE
)
bagging_model.fit(X_train, y_train)
y_pred_bagging = bagging_model.predict(X_test)
acc_bagging = accuracy_score(y_test, y_pred_bagging)
print(f"\n1. Бэггинг (Bagging):            Accuracy = {acc_bagging:.4f}")

# --- 2. Случайный лес (Random Forest) ---
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"2. Случайный лес (Random Forest): Accuracy = {acc_rf:.4f}")

# --- 3. AdaBoost ---
ada_model = AdaBoostClassifier(
    n_estimators=50,
    random_state=RANDOM_STATE
)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
acc_ada = accuracy_score(y_test, y_pred_ada)
print(f"3. AdaBoost:                      Accuracy = {acc_ada:.4f}")

# --- 4. Градиентный бустинг (Gradient Boosting) ---
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=RANDOM_STATE
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)
print(f"4. Градиентный бустинг (GB):      Accuracy = {acc_gb:.4f}")

# =============================================================
# ШАГ 6: Сравнение моделей
# =============================================================
print("\n=== Подробный отчёт: Случайный лес ===")
print(classification_report(y_test, y_pred_rf))

# Таблица сравнения
results = pd.DataFrame({
    'Модель': ['Бэггинг', 'Случайный лес', 'AdaBoost', 'Градиентный бустинг'],
    'Accuracy': [acc_bagging, acc_rf, acc_ada, acc_gb]
})
results = results.sort_values('Accuracy', ascending=False).reset_index(drop=True)
print("\n=== Сравнение точности моделей ===")
print(results.to_string(index=False))

# =============================================================
# ШАГ 7: Визуализация результатов
# =============================================================
plt.figure(figsize=(9, 5))
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
bars = plt.bar(results['Модель'], results['Accuracy'], color=colors, edgecolor='black', width=0.5)

# Подписи значений на столбцах
for bar, val in zip(bars, results['Accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.ylim(0.7, 0.9)
plt.title('Сравнение точности ансамблевых моделей (Titanic)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Модель', fontsize=12)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
print("График сохранён как model_comparison.png")

# =============================================================
# ШАГ 8: Важность признаков (Random Forest)
# =============================================================
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=True)

plt.figure(figsize=(8, 4))
feature_importances.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Важность признаков (Random Forest)', fontsize=13, fontweight='bold')
plt.xlabel('Важность', fontsize=11)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("График важности признаков сохранён как feature_importance.png")
