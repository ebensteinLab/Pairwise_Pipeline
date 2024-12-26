#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import warnings


def cal_mean_and_var(data):
    data_jax = jnp.log((jnp.array(data)))
    # Check if the data has zero values
    if jnp.any(data_jax == 0):
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-10
        data_jax = data_jax + epsilon

    # Convert the DFs to JAX array
    # jax_array = jnp.log(jnp.array(data_jax))  # each row is probe, each column is sample
    # Calculate the mean & variance for each PROBE (not samples) through all arrays
    mean_single_probes = jnp.mean(data_jax, axis=1)
    var_single_probes = jnp.var(data_jax, axis=1)

    return mean_single_probes, var_single_probes


def get_raw_means_and_val(data, name):
    raw_mean = np.mean(data, axis=1)
    raw_var = np.var(data, axis=1)
    raw_all = pd.concat([raw_mean, raw_var], axis=1)
    raw_all.rename(columns={0: f"mean_{name}", 1: f"var_{name}"}, inplace=True)
    return raw_all


def broadcast_into_pairs(single_probes_mean, single_probes_var):
    """
    This function calculates pairs on WHOLE array (not batches)
    :param single_probes_mean:
    :param single_probes_var:
    :return:
    """
    column_pairs_mean = single_probes_mean - single_probes_mean.reshape(-1, 1)
    column_pairs_var = single_probes_var + single_probes_var.reshape(-1, 1)
    return column_pairs_mean.T, column_pairs_var.T


def broadcast_into_pairs_by_block(single_probes_mean1, single_probes_mean2, single_probes_var1, single_probes_var2):
    column_pairs_mean = single_probes_mean1 - single_probes_mean2.reshape(-1, 1)
    column_pairs_var = single_probes_var1 + single_probes_var2.reshape(-1, 1)
    return column_pairs_mean.T, column_pairs_var.T


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


def split_data(df, fraction, specific_columns=[]):
    # Ensure specific columns are in the DataFrame
    specific_columns = [col for col in specific_columns if col in df.columns]

    # Sample the remaining columns
    remaining_columns = df.columns.difference(specific_columns)
    sampled_columns = df[remaining_columns].sample(frac=fraction, axis=1).columns

    # Combine specific columns with sampled columns
    combined_columns = pd.Index(specific_columns).append(sampled_columns)
    data_set = df[combined_columns]

    # Identify remaining columns
    remaining_set = df[df.columns.difference(combined_columns)]

    return data_set, remaining_set


def plot_ROC(test, predictions, test_name=None):
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
    plt.savefig(f'roc_curve_{test_name}.png')
    # plt.show()


def SSMD_cal(mean_sick, mean_healthy, var_sick, var_healthy, start1, start2, thres_prec):
    # Calculate SSMD parameter for two independent groups (sick & healthy), mean_sick = mu1, mean_healthy = mu2
    SSMD_res = (mean_sick - mean_healthy) / (
        jnp.sqrt(var_sick + var_healthy))

    # Get the indices for the upper triangular part excluding the diagonal
    upper_tri_indices = jnp.triu_indices(SSMD_res.shape[0], k=1)
    ## Extract the upper triangular values
    SSMD_res_flat = SSMD_res[upper_tri_indices]

    # # Adjust indices to reflect original positions in the full dataset
    adjusted_indices_i = upper_tri_indices[0] + start1
    adjusted_indices_j = upper_tri_indices[1] + start2
    # # Stack the indices and values together into a single array with shape (n^2, 3)
    SSMD_res_with_indices = jnp.stack([SSMD_res_flat, adjusted_indices_i, adjusted_indices_j], axis=-1)
    # SSMD_res_with_indices = np.stack([SSMD_res, indices, indices], axis=-1)
    # extract the highest thershold possible for both ends of the array
    sorted_SSMD_res = np.sort(SSMD_res_flat)[::-1]
    # sorted_SSMD_res = np.sort(SSMD_res)[::-1]
    top_20_percent_index = int(thres_prec * np.shape(sorted_SSMD_res)[0])
    threshold = sorted_SSMD_res[top_20_percent_index - 1]
    print(f"Threshold is {threshold}")

    return SSMD_res_with_indices, threshold


def process_blocks(data_sick, data_healthy, block_size, thres_prec):
    num_probes = data_sick.shape[0]
    blocks = np.arange(0, num_probes, block_size)
    print(f"Total blocks {np.shape(blocks)[0]}")
    all_results = pd.DataFrame(columns=['Ratio', 'Probe i', 'Probe j'])

    for i, start1 in enumerate(blocks):
        print(f"Now running block {i}, starting from {start1}")
        end1 = min(start1 + block_size, num_probes)
        block1_sick = jnp.array(data_sick.iloc[start1:end1])
        block1_healthy = jnp.array(data_healthy.iloc[start1:end1])

        mean_block1_sick, var_block1_sick = cal_mean_and_var(block1_sick)
        mean_block1_healthy, var_block1_healthy = cal_mean_and_var(block1_healthy)

        for j in range(i, np.shape(blocks)[0]):
            print(f"Run Block {j}")
            start2 = blocks[j]
            end2 = min(start2 + block_size, num_probes)
            block2_sick = jnp.array(data_sick.iloc[start2:end2])
            block2_healthy = jnp.array(data_healthy.iloc[start2:end2])

            mean_block2_sick, var_block2_sick = cal_mean_and_var(block2_sick)
            mean_block2_healthy, var_block2_healthy = cal_mean_and_var(block2_healthy)

            # Calculate features
            final_pairs_mean_sick, final_pairs_var_sick = broadcast_into_pairs_by_block(mean_block1_sick,
                                                                                        mean_block2_sick,
                                                                                        var_block1_sick,
                                                                                        var_block2_sick)
            final_pairs_mean_healthy, final_pairs_var_healthy = broadcast_into_pairs_by_block(mean_block1_healthy,
                                                                                              mean_block2_healthy,
                                                                                              var_block1_healthy,
                                                                                              var_block2_healthy)

            # Calculate SSMD
            SSMD_res_with_indices, threshold = SSMD_cal(final_pairs_mean_sick, final_pairs_mean_healthy,
                                                        final_pairs_var_sick, final_pairs_var_healthy, start1, start2,
                                                        thres_prec)

            # # feature selection, ignoring last two columns - probe i & probe j idx
            SSMD_filtered = (SSMD_res_with_indices[:, 0] > threshold) | (SSMD_res_with_indices[:, 0] < -threshold)
            SSMD_diff_pairs = pd.DataFrame(SSMD_res_with_indices[SSMD_filtered])
            SSMD_diff_pairs.rename(columns={0: "Ratio", 1: "Probe i", 2: "Probe j"}, inplace=True)
            all_results = pd.concat([all_results, SSMD_diff_pairs], ignore_index=True)

    return jnp.array(all_results)


def concat(df1, df2, sick_name, healthy_name, condition):
    pd.concat([df1, df2], axis=1).to_csv(
        os.path.join(os.path.join(os.getcwd(), f'{condition}_set_model_{sick_name}_{healthy_name}.csv')), index=False)


def one_model_run(config):
    warnings.filterwarnings("ignore")
    # Load data from config paths
    df_sick = pd.read_table(config.df_sick_path)
    df_healthy = pd.read_table(config.df_healthy_path)

    block_size = config.block_size
    thres_prec = config.thres_prec
    print(f"Using thres_prec: {thres_prec}")


    train_set_sick, val_set_sick = split_data(df_sick, fraction=0.8)
    train_set_healthy, val_set_healthy = split_data(df_healthy, fraction=0.8)

    SSMD_res_with_indices = process_blocks(train_set_sick, train_set_healthy, block_size, thres_prec)
    SSMD_res_with_indices = pd.DataFrame(SSMD_res_with_indices)
    SSMD_res_with_indices.rename(columns={0: "Ratio", 1: "Probe i", 2: "Probe j"}, inplace=True)
    print(f'Number of Distinguished Pairs: {np.shape(SSMD_res_with_indices)[0]}')
    SSMD_res_with_indices.to_csv(
        os.path.join(config.output_dir, f'SSMD_diff_pairs_all_data.csv'), index=False)

    train_and_val_sick = pd.concat([train_set_sick, val_set_sick], axis=1)
    data_after_diff_pairs_sick = train_and_val_sick.apply(
        lambda col: cal_pairwise_for_filtered_data(col, SSMD_res_with_indices),
        axis=0)
    data_after_diff_pairs_sick = add_label(data_after_diff_pairs_sick, label=1)

    train_and_val_healthy = pd.concat([train_set_healthy, val_set_healthy], axis=1)
    data_after_diff_pairs_healthy = train_and_val_healthy.apply(
        lambda col: cal_pairwise_for_filtered_data(col, SSMD_res_with_indices), axis=0)
    data_after_diff_pairs_healthy = add_label(data_after_diff_pairs_healthy, label=0)

    sick_and_healthy_both_train_and_val_final = pd.concat(
        [data_after_diff_pairs_sick, data_after_diff_pairs_healthy], axis=0)

    # Save data for analysis
    data_train_for_heatmap = pd.concat([data_after_diff_pairs_sick, data_after_diff_pairs_healthy], axis=0)
    data_train_for_heatmap.iloc[:, :-1].T.to_csv(
        os.path.join(config.output_dir, f'data_for_heatmap_two_groups_train_only_all_data.csv'),
        index=False)

    sick_and_healthy_both_train_and_val_final = sick_and_healthy_both_train_and_val_final.sample(frac=1)
    X = sick_and_healthy_both_train_and_val_final.iloc[:, :-1].astype(np.float32)
    Y = sick_and_healthy_both_train_and_val_final.iloc[:, -1].astype(np.float32)

    scale_pos_weight_value = (np.shape(df_healthy)[1]) / (np.shape(df_sick)[1])

    print(scale_pos_weight_value)
    param_grid = config.param_grid

    param_grid['scale_pos_weight'] = [scale_pos_weight_value]

    model = XGBClassifier(tree_method='hist')

    #kfold = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=42)
    #grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=kfold, n_jobs=config.n_jobs, verbose=1,
    #                               error_score=float('nan'))

    loo = LeaveOneOut()
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=loo, n_jobs=config.n_jobs, verbose=1,refit=True)
    
    grid_search.fit(X, Y)

    print("Best Parameters: ", grid_search.best_params_)
    print("Best Cross-Validation Accuracy: %.2f%%" % (grid_search.best_score_ * 100.0))

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
    print("Failed At:")
    file.write(Y == y_pred)
    with open("train_model_results.txt", "w") as file:
        file.write(f"Sensitivity: {sensitivity:.2f}\n")
        file.write(f"Specificity: {specificity:.2f}\n")
        for sample, true_label, predicted_label in zip(Y.index, Y, predictions):
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
    

    plot_ROC(Y, y_pred_proba)


