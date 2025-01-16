import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_files_to_dataframes(filepaths):
    """
    Loads each file in the filepaths list into a pandas DataFrame.

    Parameters:
    filepaths (list): List of file paths to load.

    Returns:
    dict: A dictionary where keys are file paths and values are DataFrames.
    """
    dataframes = {}
    for filepath in filepaths:
        try:
            # Load CSV or Excel based on file extension
            if filepath.endswith('.csv'):
                dataframes[filepath] = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx'):
                dataframes[filepath] = pd.read_excel(filepath)
            else:
                print(f"Unsupported file format: {filepath}")
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")

    return dataframes

# analyse individual files
def analyse_individual_files(dataframes):
    stats = {}
    for name, df in dataframes.items():
        print(f"--- Analysis for {name} ---")
        
        # Summary statistics for readability and sentiment
        cols_to_describe = [
            "coleman_liau", "flesch_kincaid", "gunning_fog",
            "compound_sentiment_score", "total_tokens"
        ]
        stats[name] = df[cols_to_describe].describe()
        print(stats[name])

        # Plot sentiment score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df["compound_sentiment_score"], kde=True, bins=20, color='blue')
        plt.title(f"Sentiment Score Distribution in {name}")
        plt.xlabel("Compound Sentiment Score")
        plt.ylabel("Frequency")
        plt.show()
        
        # Plot readability score comparison
        readability_scores = df[["coleman_liau", "flesch_kincaid", "gunning_fog"]]
        readability_scores.boxplot(figsize=(10, 6))
        plt.title(f"Readability Scores in {name}")
        plt.ylabel("Score")
        plt.show()
    return stats

# Compare multiple files
def compare_files(dataframes):
    combined_df = pd.concat(dataframes.values(), keys=dataframes.keys(), names=["File", "Row"])
    combined_df.reset_index(inplace=True)

    # Example 1: Compare average readability scores across files
    readability_cols = ["coleman_liau", "flesch_kincaid", "gunning_fog"]
    readability_means = combined_df.groupby("File")[readability_cols].mean()
    print("--- Average Readability Scores Across Files ---")
    print(readability_means)

    # Example 2: analyse token distribution
    token_means = combined_df.groupby("File")["total_tokens"].mean()
    print("--- Average Total Tokens Across Files ---")
    print(token_means)

    return combined_df, readability_means, token_means

# visualise comparisons
def visualise_comparisons(readability_means, token_means):
    # Plot average readability scores
    readability_means.plot(kind="bar", figsize=(12, 6))
    plt.title("Average Readability Scores Across Files")
    plt.ylabel("Score")
    plt.xlabel("File")
    plt.legend(title="Scores")
    plt.show()

    # Plot average token counts
    token_means.plot(kind="bar", figsize=(10, 5), color='orange')
    plt.title("Average Total Tokens Across Files")
    plt.ylabel("Total Tokens")
    plt.xlabel("File")
    plt.show()

# Trend Analysis for Multiple Readability Metrics Over Years
def analyse_readability_trends_with_error_bars(combined_df, metrics):
    # Ensure 'year' and metrics are numeric
    combined_df['year'] = pd.to_numeric(combined_df['year'], errors='coerce')
    for metric in metrics:
        combined_df[metric] = pd.to_numeric(combined_df[metric], errors='coerce')

    # Drop rows where 'year' or metrics are NaN
    combined_df = combined_df.dropna(subset=['year'] + metrics)

    # Prepare a dictionary to store yearly data for each metric
    trends = {}

    for metric in metrics:
        # Group by year and calculate the mean, count, and standard deviation
        yearly_data = combined_df.groupby("year").agg(
            total_score=(metric, "sum"),
            question_count=(metric, "count"),
            std_dev=(metric, "std")
        )
        yearly_data["weighted_mean"] = yearly_data["total_score"] / yearly_data["question_count"]
        trends[metric] = yearly_data

        # Print results
        print(f"--- Weighted Average and Variability for {metric} Over Years ---")
        print(yearly_data[["weighted_mean", "std_dev"]])

        # Plot the trend with error bars
        plt.figure(figsize=(12, 6))
        plt.errorbar(
            x=yearly_data.index,
            y=yearly_data["weighted_mean"],
            yerr=yearly_data["std_dev"],
            fmt='o-',  # Line with circular markers
            capsize=5,
            label=metric
        )
        plt.title(f"Trend of {metric} Over Years (With Error Bars)")
        plt.xlabel("Year")
        plt.ylabel(f"{metric} (Weighted Mean ± Std Dev)")
        plt.grid(True)
        plt.legend()
        plt.show()

# trend analysis for Question Intent
def analyse_intent_trend(combined_df):
    # ensure 'year' and 'intent' are valid
    combined_df['year'] = pd.to_numeric(combined_df['year'], errors='coerce')
    combined_df = combined_df.dropna(subset=['year', 'intent'])

    # group by year and intent, then count occurrences
    intent_counts = combined_df.groupby(['year', 'intent']).size().reset_index(name='count')

    # pivot to get intents as columns, years as rows
    pivot_data = intent_counts.pivot(index='year', columns='intent', values='count').fillna(0)

    # normalire counts to proportions for each year
    pivot_data = pivot_data.div(pivot_data.sum(axis=1), axis=0)

    # plot the trend as a stacked area chart
    plt.figure(figsize=(12, 6))
    pivot_data.plot(kind='area', stacked=True, figsize=(12, 6), alpha=0.85, colormap='tab10')
    plt.title('Trend of Question Intent Over Years')
    plt.xlabel('Year')
    plt.ylabel('Proportion of Questions')
    plt.legend(title='Intent', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # display the pivoted data for review
    print("--- Proportional Data for Intent Trend ---")
    print(pivot_data)

def get_matching_filepaths(root_folder, target_level, target_subject):
    """
    Traverses the directory structure starting at root_folder and collects
    file paths of files matching the given level and subject.

    Parameters:
    root_folder (str): The root folder (e.g., "output").
    target_level (str): The level to match (e.g., "level1").
    target_subject (str): The subject file to match (e.g., "english.txt").

    Returns:
    list: A list of matching file paths.
    """
    matching_filepaths = []

    # Traverse the directory structure
    for year_folder in os.listdir(root_folder):
        year_path = os.path.join(root_folder, year_folder)
        if os.path.isdir(year_path):  # Ensure it is a directory
            level_path = os.path.join(year_path, target_level)
            if os.path.isdir(level_path):  # Ensure the level directory exists
                for item in os.listdir(level_path):
                    item_path = os.path.join(level_path, item)
                    if os.path.isfile(item_path) and item.lower() == target_subject.lower():  # Case-insensitive match
                        matching_filepaths.append(item_path)

    return matching_filepaths



# Main Execution Update
if __name__ == "__main__":
    root = "output"  # Root folder
    level = "higher"  # Level folder to search
    subject = "english.csv"  # Subject folder to match
    
    matching_paths = get_matching_filepaths(root, level, subject)
    dataframes = load_files_to_dataframes(matching_paths)


    
    #dataframes = load_csv_files(directory)

    # analyse each file
    stats = analyse_individual_files(dataframes)

    # Compare files
    #combined_df, readability_means, token_means = compare_files(dataframes)

    # visualise the results
    #visualise_comparisons(readability_means, token_means)

    # Perform trend analysis for multiple metrics
    #metrics_to_analyse = ["gunning_fog", "coleman_liau", "flesch_kincaid"]
    #analyse_readability_trends_with_error_bars(combined_df, metrics_to_analyse)

    #analyse_intent_trend(combined_df)
