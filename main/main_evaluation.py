from src.evaluation import evaluate_models
from src.visualisation import plot_metric_comparison, plot_model_performance
from src.report_generator import save_report
import yaml
import os

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), 'config.yml')

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

def main():
    file_path = config['data']['file_path']
    lag = config['data']['lag']
    n_scats = config['data']['n_scats_options'][0]  # Use the first option as default

    # Run evaluation
    results = evaluate_models(file_path, lag, n_scats)

    # Generate visualizations
    metrics = ['MAPE', 'EVS', 'MAE', 'MSE', 'RMSE', 'R2']
    for metric in metrics:
        plot_metric_comparison(results, metric)
    plot_model_performance(results)

    # Generate and save report
    save_report(results)

if __name__ == '__main__':
    main()