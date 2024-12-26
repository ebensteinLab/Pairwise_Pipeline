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
    plt.savefig(f'roc_curve_test_data.png')


def test_new_data_on_final_model(config):
    model_path = config.model_path
    data_path = config.SSMD_data
    model = joblib.load(model_path)
    SSMD_diff_pairs = pd.read_csv(data_path)
    
    if config.two_groups_for_test: # if we have two groups (1 and 0 label) then run two datasets
        print("Running Two Groups")
        df_test_sick = pd.read_table(config.df_test_sick_path)
        df_test_healthy = pd.read_table(config.df_test_healthy_path)

 
        filtered_sick = df_test_sick.apply(
            lambda col: cal_pairwise_for_filtered_data(col, SSMD_diff_pairs), axis=0)
        sick_test_label_pairs = add_label(filtered_sick, label=1)
        filtered_healthy = df_test_healthy.apply(
            lambda col: cal_pairwise_for_filtered_data(col, SSMD_diff_pairs), axis=0)
        healthy_test_label_pairs = add_label(filtered_healthy, label=0)

        sick_and_healthy_both_test_and_val_final = pd.concat(
            [sick_test_label_pairs, healthy_test_label_pairs],
            axis=0)

        # print data for analysis
        sick_and_healthy_both_test_and_val_final.T.to_csv(
            os.path.join(os.path.join(os.getcwd(),
                                      f'test_data_for_heatmap_after_final_model.csv')), index=False)

        X_test = sick_and_healthy_both_test_and_val_final.iloc[:, :-1]
        Y_test = sick_and_healthy_both_test_and_val_final.iloc[:, -1]

        predictions = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class

        # Calculate accuracy
        accuracy = accuracy_score(Y_test, predictions)

        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        conf_matrix = confusion_matrix(Y_test, predictions)
        tn, fp, fn, tp = conf_matrix.ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        with open("test_model_results.txt", "w") as file:
            file.write(f"Sensitivity: {sensitivity:.2f}\n")
            file.write(f"Specificity: {specificity:.2f}\n")

            file.write("Scores for Each Sample:\n")
            file.write("\n".join(map(str, y_scores)) + "\n")

            # Loop through samples and write results
            for sample, true_label, predicted_label in zip(Y_test.index, Y_test, predictions):
                if "B" in sample:
                    if predicted_label:
                        label = "Sick"
                    else:
                        label = "Healthy"
                elif "A" in sample:
                    if not predicted_label:
                        label = "Healthy"
                    else:
                        label = "Sick"
                else:
                    label = "Unknown"
                file.write(f"{sample}: {label}\n")

        # ROC Curve
        plot_ROC(Y_test, y_scores)
    
    else: # if only one group
        print("Running One Group")
        df_test_sick = pd.read_table(config.df_test_sick_path)
        
        filtered_sick = df_test_sick.apply(
            lambda col: cal_pairwise_for_filtered_data(col, SSMD_diff_pairs), axis=0)
        sick_test_label_pairs = add_label(filtered_sick, label=1)
        
        sick_and_healthy_both_test_and_val_final = sick_test_label_pairs
        
        X_test = sick_and_healthy_both_test_and_val_final.iloc[:, :-1]
        Y_test = sick_and_healthy_both_test_and_val_final.iloc[:, -1]

        predictions = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class

        # Calculate accuracy
        accuracy = accuracy_score(Y_test, predictions)

        with open("test_model_results.txt", "w") as file:
            file.write(f"Sensitivity: {sensitivity:.2f}\n")
            file.write(f"Specificity: {specificity:.2f}\n")

            file.write("Scores for Each Sample:\n")
            file.write("\n".join(map(str, y_scores)) + "\n")

            # Loop through samples and write results
            for sample, true_label, predicted_label in zip(Y_test.index, Y_test, predictions):
                if "B" in sample:
                    if predicted_label:
                        label = "Sick"
                    else:
                        label = "Healthy"
                elif "A" in sample:
                    if not predicted_label:
                        label = "Healthy"
                    else:
                        label = "Sick"
                else:
                    label = "Unknown"
                file.write(f"{sample}: {label}\n")
