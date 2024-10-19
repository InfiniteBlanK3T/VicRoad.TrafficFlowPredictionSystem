import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_metric_comparison(results, metric):
    """
    Plot a comparison of a specific metric across all models.
    
    Args:
        results (dict): A dictionary containing evaluation results for each SCATS and model.
        metric (str): The name of the metric to plot.
    
    Raises:
        ValueError: If the results dictionary is empty or the metric is not found.
    """
    if not results:
        raise ValueError("Results dictionary is empty.")
    
    try:
        df = pd.DataFrame(results).T
        df = df.applymap(lambda x: x[metric])
        
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df)
        plt.title(f'{metric} Comparison Across Models', fontsize=16)
        plt.ylabel(metric, fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add mean values on top of each box
        means = df.mean()
        for i, mean in enumerate(means):
            plt.text(i, plt.ylim()[1], f'Mean: {mean:.4f}', 
                     horizontalalignment='center', verticalalignment='bottom')
        
        plt.tight_layout()
        os.makedirs('images', exist_ok=True)
        plt.savefig(f'images/metric_comparison_{metric}.png')
        plt.close()
        logger.info(f"Metric comparison plot for {metric} saved successfully.")
    except KeyError:
        logger.error(f"Metric '{metric}' not found in the results.")
    except Exception as e:
        logger.error(f"An error occurred while plotting metric comparison: {str(e)}")

def plot_model_performance(results):
    """
    Plot performance comparison of all metrics for all models.
    
    Args:
        results (dict): A dictionary containing evaluation results for each SCATS and model.
    
    Raises:
        ValueError: If the results dictionary is empty.
    """
    if not results:
        raise ValueError("Results dictionary is empty.")
    
    try:
        models = list(next(iter(results.values())).keys())
        metrics = list(next(iter(next(iter(results.values())).values())).keys())
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 8*len(metrics)), sharex=True)
        
        for i, metric in enumerate(metrics):
            df = pd.DataFrame(results).T
            df = df.applymap(lambda x: x[metric])
            
            sns.boxplot(data=df, ax=axes[i])
            axes[i].set_title(f'{metric} Comparison', fontsize=16)
            axes[i].set_ylabel(metric, fontsize=12)
            
            # Add mean values on top of each box
            means = df.mean()
            for j, mean in enumerate(means):
                axes[i].text(j, axes[i].get_ylim()[1], f'Mean: {mean:.4f}', 
                             horizontalalignment='center', verticalalignment='bottom')
        
        plt.xlabel('Models', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        os.makedirs('images', exist_ok=True)
        plt.savefig('images/model_performance_comparison.png')
        plt.close()
        logger.info("Model performance comparison plot saved successfully.")
    except Exception as e:
        logger.error(f"An error occurred while plotting model performance: {str(e)}")

def generate_metric_summary(results):
    """
    Generate a markdown summary of metrics for all models.
    
    Args:
        results (dict): A dictionary containing evaluation results for each SCATS and model.
    
    Returns:
        str: A markdown-formatted summary of metrics.
    
    Raises:
        ValueError: If the results dictionary is empty.
    """
    if not results:
        raise ValueError("Results dictionary is empty.")
    
    try:
        summary = "# Metric Summaries\n\n"
        models = list(next(iter(results.values())).keys())
        metrics = list(next(iter(next(iter(results.values())).values())).keys())
        
        for metric in metrics:
            summary += f"## {metric}\n\n"
            df = pd.DataFrame(results).T
            df = df.applymap(lambda x: x[metric])
            
            summary += "| Model | Mean | Median | Std Dev |\n"
            summary += "|-------|------|--------|--------|\n"
            
            for model in models:
                mean = df[model].mean()
                median = df[model].median()
                std = df[model].std()
                summary += f"| {model} | {mean:.4f} | {median:.4f} | {std:.4f} |\n"
            
            summary += "\n"
            
        return summary
    except Exception as e:
        logger.error(f"An error occurred while generating metric summary: {str(e)}")
        return ""

if __name__ == '__main__':
    try:
        with open('evaluation_results.json', 'r') as f:
            results = json.load(f)
        
        metrics = ['MAPE', 'EVS', 'MAE', 'MSE', 'RMSE', 'R2']
        for metric in metrics:
            plot_metric_comparison(results, metric)
        
        plot_model_performance(results)
        
        summary = generate_metric_summary(results)
        with open('metric_summary.md', 'w') as f:
            f.write(summary)
        
        logger.info("Visualization and summary generation completed successfully.")
    except FileNotFoundError:
        logger.error("evaluation_results.json file not found.")
    except json.JSONDecodeError:
        logger.error("Invalid JSON in evaluation_results.json file.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")