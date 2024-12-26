import jax.numpy as jnp
import jax
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import os
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, GridSearchCV, StratifiedKFold
import math
import matplotlib.pyplot as plt
import joblib
from jax.lib import xla_bridge
import glob
import joblib


def cal_pairwise_for_filtered_data(data, row_pairs, epsilon=1e-10):
    probe1 = row_pairs['Probe i'].values
    probe2 = row_pairs['Probe j'].values

    # Select rows for probe1 and probe2
    data_probe1 = data.iloc[probe1].astype(np.float64).values
    data_probe2 = data.iloc[probe2].astype(np.float64).values

    # Add a small constant to avoid division by zero
    data_probe1 += epsilon
    data_probe2 += epsilon

    # Calculate the ratios
    ratios = np.log(data_probe1 / data_probe2)
    return pd.Series(ratios)


def add_label(after_diff_analysis_data, label):
    new_row = pd.Series([label] * after_diff_analysis_data.shape[1],
                        index=after_diff_analysis_data.columns).to_frame().T
    new_row.index = ["label"]
    return pd.concat([after_diff_analysis_data, new_row], ignore_index=False).T


def plot_ROC(test, predictions):
    fpr, tpr, thresholds = roc_curve(test, predictions)
    print(f"fpr: {fpr}")
    print(f"tpr: {tpr}")
    roc_auc = auc(fpr, tpr)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, thresholds = roc_curve(test, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve')
    plt.legend(loc="center")
    plt.grid(True)
    plt.savefig(f'roc_curve.png')


def model_after_LOO(config):
    df_sick = pd.read_table(config.df_sick_path)
    df_healthy = pd.read_table(config.df_healthy_path)
    SSMD_diff_pairs = pd.read_csv('shared_ssmd_data.csv')

    filtered_sick = df_sick.apply(
        lambda col: cal_pairwise_for_filtered_data(col, SSMD_diff_pairs), axis=0)
    filtered_healthy = df_healthy.apply(
        lambda col: cal_pairwise_for_filtered_data(col, SSMD_diff_pairs), axis=0)
    # Add labels for the sick and healthy groups (1 for sick, 0 for healthy)
    filtered_sick = add_label(filtered_sick, label=1)
    filtered_healthy = add_label(filtered_healthy, label=0)
    sick_and_healthy_both_train_and_val_final = pd.concat([filtered_sick, filtered_healthy], axis=0)
    sick_and_healthy_both_train_and_val_final.T.to_csv(
        os.path.join(os.path.join(os.getcwd(), "all_data_one_model_heatmap.csv")), index=False)

    X = sick_and_healthy_both_train_and_val_final.iloc[:, :-1]
    Y = sick_and_healthy_both_train_and_val_final.iloc[:, -1]
    
    scale_pos_weight_value = (np.shape(df_healthy)[1]) / (np.shape(df_sick)[1])
    
    print(scale_pos_weight_value)

    param_grid = config.param_grid
    
    param_grid['scale_pos_weight'] = [scale_pos_weight_value] 
    
    # fit model to training data
    model = XGBClassifier(tree_method="hist", device="cuda")

    # K-fold Cross Validation
    # Grid Search with Cross-Validation
    # kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # Define cross-validation strategy
    # cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # Leave-One-Out Cross-Validation
    
    loo = LeaveOneOut()
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=loo, n_jobs=config.n_jobs, verbose=10, pre_dispatch=10,
                               refit=True)

    grid_search.fit(X, Y)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f'best_FINAL_xgb_model.joblib')
    print("Model saved to best_FINAL_xgb_model.joblib")

    y_pred = best_model.predict(X)
    predictions = [round(value) for value in y_pred]

    # # Calculate predicted probabilities
    y_pred_proba = best_model.predict_proba(X)[:, 1]
    # # print(best_model.predict_proba(X_test))
    #
    # # evaluate predictions
    accuracy = accuracy_score(Y, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    ##Calculate confusion matrix - only for 2 labels
    conf_matrix = confusion_matrix(Y, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    ##Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print("Failed At:")
    print(Y == y_pred)

    plot_ROC(Y, y_pred_proba)
