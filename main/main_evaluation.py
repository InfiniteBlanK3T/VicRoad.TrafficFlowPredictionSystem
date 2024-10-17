from src.evaluation import evaluate_models
from src.visualisation import plot_metric_comparison, plot_model_performance
from src.report_generator import generate_report
import yaml

# Load configuration
with open('config.yml', 'r') as config_file:
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

    # Generate report
    report = generate_report(results)
    with open('model_comparison_report.md', 'w') as f:
        f.write(report)

if __name__ == '__main__':
    main()