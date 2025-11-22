

import os
import gzip
import pickle
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

def clean_dataset(df):
    """Paso 1: Preprocesamiento"""
    df = df.copy()
    df['Age'] = 2021 - df['Year']
    df.drop(columns=['Year', 'Car_Name'], inplace=True)
    return df

def load_data():
    """Carga los datasets"""
    train_df = pd.read_csv("files/input/train_data.csv.zip", index_col=False, compression="zip")
    test_df = pd.read_csv("files/input/test_data.csv.zip", index_col=False, compression="zip")
    return train_df, test_df

def calculate_metrics(y_true, y_pred, dataset_name):
    """Calcula metricas. OJO: MAD aqui es MEDIAN absolute error"""
    metrics = {
        'type': 'metrics',
        'dataset': dataset_name,
        'r2': float(r2_score(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'mad': float(median_absolute_error(y_true, y_pred)) 
    }
    return metrics

def main():
    # --- Paso 1 y 2: Carga y Limpieza de Datos ---
    train_df, test_df = load_data()
    
    train_df = clean_dataset(train_df)
    test_df = clean_dataset(test_df)

    
    x_train = train_df.drop(columns=['Present_Price'])
    y_train = train_df['Present_Price']
    
    x_test = test_df.drop(columns=['Present_Price'])
    y_test = test_df['Present_Price']

    
    
    # --- Paso 3: Construccion del Pipeline ---
    cat_features = ['Fuel_Type', 'Selling_type', 'Transmission']
    num_features = [col for col in x_train.columns if col not in cat_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
            ('num', MinMaxScaler(), num_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=f_regression)),
        ('regressor', LinearRegression())
    ])

    
    # --- Paso 4: Optimizacion ---
    param_grid = {
        "selector__k": range(1, 15),
        "regressor__fit_intercept": [True, False],
        "regressor__positive": [True, False]
    }

    model = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    model.fit(x_train, y_train)

    # --- Paso 5: Guardar ---
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)

    # --- Paso 6: Metricas ---
    os.makedirs("files/output", exist_ok=True)
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = calculate_metrics(y_train, y_train_pred, 'train')
    test_metrics = calculate_metrics(y_test, y_test_pred, 'test')

    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(train_metrics) + '\n')
        f.write(json.dumps(test_metrics) + '\n')

    print("Tarea completada exitosamente.")

if __name__ == "__main__":
    main()