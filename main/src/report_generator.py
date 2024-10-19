import json
import pandas as pd
from tabulate import tabulate
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_report(results):
    """
    Generate a markdown report comparing model performance.
    
    Args:
        results (dict): A dictionary containing evaluation results for each SCATS and model.
    
    Returns:
        str: A markdown-formatted report.
    
    Raises:
        ValueError: If the results dictionary is empty or malformed.
    """
    if not results:
        raise ValueError("Results dictionary is empty.")
    
    report = "# Model Comparison Report\n\n"
    
    try:
        # Overall performance
        report += "## Overall Performance\n\n"
        overall_performance = {}
        for scats, models in results.items():
            for model, metrics in models.items():
                if model not in overall_performance:
                    overall_performance[model] = {metric: [] for metric in metrics}
                for metric, value in metrics.items():
                    overall_performance[model][metric].append(value)
        
        overall_df = pd.DataFrame({model: {metric: sum(values)/len(values) for metric, values in metrics.items()} 
                                   for model, metrics in overall_performance.items()}).T
        report += tabulate(overall_df, headers='keys', tablefmt='pipe', floatfmt='.4f')
        report += "\n\n"
        
        # Best model for each metric
        report += "## Best Model for Each Metric\n\n"
        best_models = overall_df.idxmin()  # Assuming lower is better for all metrics
        best_models_df = pd.DataFrame({'Metric': best_models.index, 'Best Model': best_models.values})
        report += tabulate(best_models_df, headers='keys', tablefmt='pipe')
        report += "\n\n"
        
        # Performance by SCATS
        report += "## Performance by SCATS\n\n"
        for scats, models in results.items():
            report += f"### SCATS {scats}\n\n"
            scats_df = pd.DataFrame(models).T
            report += tabulate(scats_df, headers='keys', tablefmt='pipe', floatfmt='.4f')
            report += "\n\n"
        
        return report
    except Exception as e:
        logger.error(f"An error occurred while generating the report: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        with open('evaluation_results.json', 'r') as f:
            results = json.load(f)
        
        report = generate_report(results)
        
        with open('model_comparison_report.md', 'w') as f:
            f.write(report)
        
        logger.info("Report generated successfully.")
    except FileNotFoundError:
        logger.error("evaluation_results.json file not found.")
    except json.JSONDecodeError:
        logger.error("Invalid JSON in evaluation_results.json file.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")