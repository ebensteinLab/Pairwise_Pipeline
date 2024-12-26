import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import joblib
import numpy as np


def heatmap_for_xgboost(config):
    # df: DataFrame where each row is a sample and each column is a feature
    df = pd.read_csv(config.heatmap_data_input).T
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    model = joblib.load(config.model_path)

    explainer = shap.Explainer(model)

    # Get SHAP values for the training data
    shap_values = explainer(X)

    # Sum the SHAP values across all features for each sample (axis=1)
    shap_sum = shap_values.values.sum(axis=1)

    # Sort the samples based on the SHAP value sum
    sorted_indices = shap_sum.argsort()


    # Calculate the mean absolute SHAP values for each feature
    shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)
    mean_abs_shap = np.abs(shap_values_df).mean(axis=0)

    # Create a DataFrame for feature importance
    feature_importance = pd.DataFrame(mean_abs_shap, columns=["mean_abs_shap"]).sort_values(by="mean_abs_shap", ascending=False)

    # Define a threshold (e.g., 25th percentile) to drop less influential features
    threshold = feature_importance['mean_abs_shap'].quantile(0.50)

    # Identify features to drop
    features_to_drop = feature_importance[feature_importance['mean_abs_shap'] <= threshold].index.tolist()

    # Drop the less influential features from the original DataFrame
    X_reduced = X.drop(columns=features_to_drop)

    #sorted_indices = np.delete(sorted_indices, features_to_drop)

    # Reorder the features DataFrame based on the sorted SHAP values
    features_df_sorted = X_reduced.iloc[sorted_indices]  # Assuming column 172 is still the predictions


    # Transpose the DataFrame to have features as rows and samples as columns
    features_sorted_transposed = features_df_sorted.T

    # Create the heatmap
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Create a larger figure
    plt.figure(figsize=(20, 12))

    # Create the heatmap with the previous color scheme
    sns.heatmap(features_sorted_transposed, cmap='coolwarm', annot=False, cbar_kws={"shrink": 0.8})  # Shrink color bar


    # Create a new axes for the color strip above the heatmap
    color_strip_ax = plt.gca().inset_axes([0.0, 1.0, 1.0, 0.03])  # Thinner strip height

    # Create a color list based on whether sample names start with 'A' or 'B'
    sample_labels = features_sorted_transposed.columns
    colors = ['#4CAF50' if label.startswith('A') else '#F44336' for label in sample_labels]  # Green for 'A' (healthy), Red for 'B' (sick)

    # Draw the color strip
    for idx, color in enumerate(colors):
        color_strip_ax.add_patch(plt.Rectangle((idx, 0), 1, 1, color=color, lw=0))

    # Remove the axes for the color strip
    color_strip_ax.set_xlim(0, len(colors))
    color_strip_ax.set_ylim(0, 1)
    color_strip_ax.axis('off')  # Turn off the axes

    # Add a legend, moving it up to avoid overlap with the gradient bar
    legend_labels = ['Healthy (A)', 'Sick (B)']
    legend_colors = ['#4CAF50', '#F44336']
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
    color_strip_ax.legend(handles, legend_labels, loc='upper left', frameon=False, fontsize=10, bbox_to_anchor=(1.02, 1.1))

    # Set x-ticks and rotate the labels for the heatmap
    plt.xticks(ticks=range(features_sorted_transposed.shape[1]),
               labels=features_sorted_transposed.columns, rotation=90, fontsize=8)

    # Add labels and title
    plt.title('Heatmap of Features Per Sample of XGBoost Model', fontsize=16, fontweight='bold', pad=30)
    plt.xlabel('Samples', fontsize=12)
    plt.ylabel('Features', fontsize=12)

    # Ensure the layout fits
    plt.tight_layout()
    plt.savefig(f'Heatmap_for_xgboost_final_model.png')
    plt.show()