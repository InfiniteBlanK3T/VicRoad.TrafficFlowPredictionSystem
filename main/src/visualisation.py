import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np

def plot_metric_comparison(results, metric):
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
    plt.savefig(f'images/metric_comparison_{metric}.png')
    plt.close()

def plot_model_performance(results):
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
    plt.savefig('images/model_performance_comparison.png')
    plt.close()

def generate_metric_summary(results):
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

if __name__ == '__main__':
    with open('evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    metrics = ['MAPE', 'EVS', 'MAE', 'MSE', 'RMSE', 'R2']
    for metric in metrics:
        plot_metric_comparison(results, metric)
    
    plot_model_performance(results)
    
    summary = generate_metric_summary(results)
    with open('metric_summary.md', 'w') as f:
        f.write(summary)