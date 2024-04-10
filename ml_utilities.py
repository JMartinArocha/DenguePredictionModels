import pandas as pd
from fpdf import FPDF
from sklearn.metrics import (accuracy_score, classification_report, 
                             roc_auc_score, f1_score, confusion_matrix, 
                             precision_recall_curve, auc, roc_curve)
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import binarize
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
from datetime import datetime
import joblib
import itertools
from sklearn.preprocessing import MinMaxScaler

def df_look(df):
    """
    This function takes a pandas DataFrame as input and prints out several
    descriptive details about the DataFrame. If the provided object is not a
    DataFrame, it raises a ValueError.

    Args:
    df (pd.DataFrame): The DataFrame to be inspected.

    Returns:
    None: This function prints details about the DataFrame and does not return any value.
    """

    # Check if the provided object is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The provided object is not a pandas DataFrame.")

    # Display the first few rows of the DataFrame
    print("First few rows of the DataFrame:")
    print(df.head())

    # Display data types of each column in the DataFrame
    print("\nData types of the columns:")
    print(df.dtypes)

    # Display a statistical summary of the DataFrame
    # Convert sparse columns to dense before computing the summary
    df_dense = df.apply(lambda x: x.sparse.to_dense() if pd.api.types.is_sparse(x) else x)
    print("\nStatistical summary:")
    print(df_dense.describe(include='all'))

    # Display the column names of the DataFrame
    print("\nColumn names:")
    print(df.columns)

    # Display information about null values in each column
    print("\nInformation about null values:")
    print(df.isnull().sum())

    # Display the number of rows and columns in the DataFrame
    print("\nNumber of rows and columns:")
    print(df.shape)

def generate_evaluation_report(model, X_test, y_test, y_pred, description=''):
    # Get the name of the classifier
    classifier_name = model.__class__.__name__
    # Initialize PDF document
    pdf = FPDF()
    pdf.add_page()


    # Determinar si el problema es binario o multiclase
    n_classes = len(np.unique(y_test))
    # Determinar si el problema es binario o multiclase
    is_binary_classification = n_classes == 2
    if is_binary_classification:
        average_method = 'binary'
    else:
        average_method = 'weighted'  # Adecuado para multiclase

    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, target_names=np.unique(y_test).astype(str))
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=average_method)

    # Revisar si el modelo tiene predict_proba para clasificación binaria o multiclase
    if hasattr(model, 'predict_proba'):
        # Obtener probabilidades
        y_scores = model.predict_proba(X_test)
        
        if is_binary_classification:
            # Usar la segunda columna para clasificación binaria que representa la clase positiva
            y_scores_for_auc = y_scores[:, 1]
        else:
            # Para multiclase, necesitas binarizar y_test
            y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
            y_scores_for_auc = y_scores  # Usar scores directamente para multiclase
        
        # Calcular ROC-AUC dependiendo de si es binario o multiclase
        if is_binary_classification:
            roc_auc = roc_auc_score(y_test, y_scores_for_auc)
            fpr, tpr, thresholds = roc_curve(y_test, y_scores_for_auc)
        else:
            roc_auc = roc_auc_score(y_test_binarized, y_scores_for_auc, multi_class='ovr', average='weighted')
        
        # Si es clasificación binaria, también puedes calcular la curva Precision-Recall
        if is_binary_classification:
            precision, recall, _ = precision_recall_curve(y_test, y_scores_for_auc)

    else:
        # Si no hay predict_proba, se podría manejar de forma diferente, pero aquí solo marcamos ROC-AUC como no aplicable
        y_scores_for_auc = 'N/A'
        roc_auc = 'N/A'

    # Cross-validation scores for non-Keras classifiers
    if classifier_name != 'Sequential':
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
        cv_scores_text = "Cross-Validation Scores: " + ", ".join([f"{score:.3f}" for score in cv_scores])
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        cv_summary = f"CV Mean: {cv_mean:.3f}, CV Standard Deviation: {cv_std:.3f}"

    # Set title and evaluation metrics in PDF
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Evaluation Report: {classifier_name}", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Description: {description}", ln=True)
    pdf.cell(200, 10, txt=f"Model Accuracy: {accuracy}", ln=True)
    pdf.cell(200, 10, txt=f"Classification Report:\n{classification_rep}", ln=True)
    pdf.cell(200, 10, txt=f"ROC-AUC: {roc_auc}", ln=True)
    pdf.cell(200, 10, txt=f"F1 Score: {f1}", ln=True)
    if classifier_name != 'Sequential':
        pdf.cell(0, 10, txt=cv_scores_text, ln=True)
        pdf.cell(0, 10, txt=cv_summary, ln=True)

    # Function to add plots to the PDF
    def add_plot_to_pdf(plt_func):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            plt_func()
            plt.savefig(tmpfile.name, format='png')
            plt.close()
            pdf.image(tmpfile.name, x=None, y=None, w=190, h=100)
        os.remove(tmpfile.name)

    # Add plots to the PDF
    add_plot_to_pdf(lambda: sns.heatmap(conf_matrix, cmap='Blues', annot=True, fmt="d"))
    if hasattr(model, 'predict_proba'):
        if is_binary_classification:
            add_plot_to_pdf(lambda: plot_roc_curve(fpr, tpr, roc_auc))
            add_plot_to_pdf(lambda: plot_precision_recall_curve(precision, recall))

    # Save PDF with a dynamic file name
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    pdf.output(f"evaluation_report_{classifier_name}_{current_time}.pdf")

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

def plot_precision_recall_curve(precision, recall):
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'DataFrame Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

def df_look_pdf(df, model_name):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The provided object is not a pandas DataFrame.")
    
    pdf = PDF()
    pdf.add_page()

    # DataFrame Head
    pdf.chapter_title('First few rows of the DataFrame:')
    df_head = df.head().to_string()
    pdf.chapter_body(df_head)

    # Data Types
    pdf.chapter_title('Data types of the columns:')
    df_dtypes = df.dtypes.to_string()
    pdf.chapter_body(df_dtypes)

    # Statistical Summary
    pdf.chapter_title('Statistical summary:')
    df_dense = df.apply(lambda x: x.sparse.to_dense() if pd.api.types.is_sparse(x) else x)
    df_summary = df_dense.describe(include='all').to_string()
    pdf.chapter_body(df_summary)

    # Column Names
    pdf.chapter_title('Column names:')
    df_columns = ', '.join(df.columns)
    pdf.chapter_body(df_columns)

    # Null Values Information
    pdf.chapter_title('Information about null values:')
    df_nulls = df.isnull().sum().to_string()
    pdf.chapter_body(df_nulls)

    # Shape of DataFrame
    pdf.chapter_title('Number of rows and columns:')
    df_shape = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
    pdf.chapter_body(df_shape)

    # Save PDF
    output = f"{model_name}_{current_time}_report.pdf"
    pdf.output(output)

def save_model(model, file_name=None):
    """
    Saves the given machine learning model to a file using joblib.

    Parameters:
    model (sklearn.base.BaseEstimator): The machine learning model to be saved.
    file_name (str, optional): The name of the file to save the model. If None, a default name is generated.

    Returns:
    None
    """

    # Generate a default file name if not provided
    if file_name is None:
        file_name = f"{model.__class__.__name__}_trained_model.pkl"

    # Save the model to a file
    joblib.dump(model, file_name)
    # Confirmation message
    print(f"Model saved as: {file_name}")


def generate_pairwise_pairplots(df, hue=None):
    """
    Genera pairplots para cada combinación única de dos características.

    Args:
    - df: DataFrame de pandas que contiene los datos.
    - hue: (Opcional) Nombre de la columna en 'df' que se utilizará para colorear los puntos según su categoría.
    """
    características = df.columns.drop(hue) if hue else df.columns
    
    # Generar todas las combinaciones únicas de dos características
    for (característica1, característica2) in itertools.combinations(características, 2):
        sns.pairplot(df, vars=[característica1, característica2], hue=hue,palette='coolwarm', height=3, aspect=2)
        plt.suptitle(f'{característica1} vs {característica2}', y=1.02)  # Ajusta 'y' para la posición vertical del título
        plt.show()

def plot_metrics(history, test_metrics=None, title_prefix=""):
    """
    Grafica las métricas de precisión y pérdida para entrenamiento y validación,
    y añade las métricas de test como una línea horizontal.
    
    Parámetros:
    - history: Objeto History retornado por model.fit().
    - test_metrics: Tupla opcional (test_loss, test_accuracy) con métricas de test.
    - title_prefix: Prefijo opcional para el título de las gráficas.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    if test_metrics:
        plt.axhline(y=test_metrics[1], color='r', linestyle='--', label='Test Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{title_prefix}Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    if test_metrics:
        plt.axhline(y=test_metrics[0], color='r', linestyle='--', label='Test Loss')
    plt.legend(loc='upper right')
    plt.title(f'{title_prefix}Training and Validation Loss')
    
    plt.show()

def load_model(model_filename):
    """
    Carga un modelo de scikit-learn previamente entrenado y guardado desde un archivo.

    Args:
    model_filename (str): El nombre del archivo desde el cual cargar el modelo. Este archivo debe
                          existir en el directorio desde el que se ejecuta el script o notebook,
                          o bien debe proporcionarse una ruta absoluta o relativa al archivo.
    
    Returns:
    El modelo cargado listo para usarse para hacer predicciones o para cualquier otro análisis.
    
    Ejemplo de uso:
    model = load_model('random_forest_model.joblib')
    """
    # Asegúrate de que el nombre del archivo se proporciona como un string
    if not isinstance(model_filename, str):
        raise ValueError("El nombre del archivo debe ser una cadena de caracteres.")
    
    try:
        # Intenta cargar el modelo desde el archivo especificado
        loaded_model = joblib.load(model_filename)
        print("Modelo cargado exitosamente.")
        return loaded_model
    except FileNotFoundError:
        # Maneja el caso en que el archivo especificado no se encuentra
        print(f"Error: El archivo '{model_filename}' no se encontró.")
    except Exception as e:
        # Maneja cualquier otro error que pueda ocurrir
        print(f"Ocurrió un error al cargar el modelo: {e}")

# Ejemplo de uso
# model = load_model('mi_modelo_entrenado.joblib')



def normalize_dataset(df, scaler=MinMaxScaler()):
    """
    Normaliza los datos numéricos de un DataFrame utilizando MinMaxScaler por defecto.

    Parámetros:
    - df: DataFrame de pandas con los datos a normalizar.
    - scaler: Instancia de un escalador de sklearn.preprocessing (MinMaxScaler por defecto).

    Retorna:
    - DataFrame de pandas con los datos normalizados.
    """
    # Seleccionar columnas numéricas
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Ajustar y transformar los datos
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

# Ejemplo de uso:
# Asumiendo que `df` es tu DataFrame
# df_normalized = normalize_dataset(df)




def generate_regresion_evaluation_report(model, X, y, cv=5):
    """
    Evalúa un modelo utilizando validación cruzada y devuelve las métricas recopiladas.

    Parámetros:
    - model: El modelo a evaluar.
    - X: Las características del conjunto de datos.
    - y: La variable objetivo del conjunto de datos.
    - cv: El número de pliegues para la validación cruzada.

    Retorna:
    - Un diccionario que contiene las métricas MSE, MAE y R^2.
    """
    scoring = {
            'mse': make_scorer(mean_squared_error, greater_is_better=False),
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
            'r2': 'r2'}
    
    scores = cross_validate(model, X, y, scoring=scoring, cv=cv, return_train_score=True)

    return {
        'mse': -np.mean(scores['test_mse']),
        'mae': -np.mean(scores['test_mae']),
        'r2': np.mean(scores['test_r2']),
    }

# Uso de ejemplo:
# Asume que tienes un modelo `model` ya definido y entrenado, y que `X` e `y` son tus características y variable objetivo.
# generate_evaluation_report(model, X, y, "Random Forest Regressor Evaluation", cv=5)

import matplotlib.pyplot as plt

def plot_all_metrics_comparisons(models_metrics, scale_factor=1):
    """
    Grafica comparaciones de modelos para todas las métricas disponibles, ajustando la escala para mejorar la visualización.
    Se pueden imprimir los valores de las métricas escalados.

    Parámetros:
    - models_metrics: Diccionario con los nombres de los modelos como claves y diccionarios de sus métricas como valores.
    - scale_factor: Factor por el cual se multiplicarán los valores de las métricas antes de graficar.
    """
    all_metrics = set(metric for metrics in models_metrics.values() for metric in metrics)
    
    for metric in all_metrics:
        names = list(models_metrics.keys())
        values = [metrics.get(metric, 0) * scale_factor for metrics in models_metrics.values()]
        
        print(f"--- {metric.upper()} (Escala: x{scale_factor}) ---")
        for name, value in zip(names, values):
            print(f"{name}: {value:.4f}")
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(names, values, color='skyblue')
        plt.xlabel('Modelos', fontsize=14)
        plt.ylabel(f'{metric.upper()} (x{scale_factor})', fontsize=14)
        plt.title(f'Comparación de Modelos por {metric.upper()} (Escala: x{scale_factor})', fontsize=16)
        plt.xticks(rotation=45, ha="right")
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}", ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.show()

# Function to visualize feature importance
def plot_feature_importance(importances, feature_names, title):
    """
    Plots the feature importance as determined by the model.

    Args:
        importances (array-like): The feature importances.
        feature_names (list): List of feature names.
        title (str): The title of the plot.
    """
    # Sort the feature importances in descending order and get their indices
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    sorted_names = [feature_names[i] for i in indices]

    # Create a horizontal bar plot to display feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances[indices], align='center')
    plt.yticks(range(len(importances)), sorted_names)
    plt.xlabel('Feature Importance')
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert the y-axis to have the most important feature on top
    plt.show()