# Report
Once you run python `main/main_evaluation.py`, the report functionalities include the following features:

1. Generating Metric Comparison Plots:

The script generates .png files for metric comparisons of the following metrics:
- EVS (Explained Variance Score)
- MAPE (Mean Absolute Percentage Error)
- MSE (Mean Squared Error)
- R2 (R-squared)
- RMSE (Root Mean Squared Error)
These plots are saved in the directory specified by `config["evaluation"]["metric_save_path"]` with filenames that include the current timestamp.
2. Model Performance Comparison:

The script generates a comprehensive plot comparing the performance of all models across all metrics.
This plot is saved in the directory specified by `config["evaluation"]["metric_save_path"]` with a filename that includes the current timestamp.
3. Prediction Plots:

The script generates plots showing the predictions of all models for each SCATS number.
It also generates individual plots for each model's predictions compared to the true data.
These plots are saved in the directory specified by `config["evaluation"]["prediction_save_path"]` with filenames that include the current timestamp.
4. Generating Comparison Report:

The script generates a markdown report summarizing the model performance.
The report includes overall performance, the best model for each metric, and performance by SCATS.
The report is saved in the directory specified by `config["evaluation"]["report_summary_save_path"]` with a filename that includes the current timestamp.
## Example Usage
To run the evaluation and generate the reports, use the following command:
Ensure that your `config.yml` file is correctly set up with the appropriate paths for saving the reports and plots.

Configuration
The `config.yml` file should include the following paths:
```
evaluation:
  metric_save_path: 'reports/metric-comparison/'
  prediction_save_path: 'reports/prediction-comparison/'
  report-summary_save_path: 'reports/summary/'  

```
This ensures that all generated files are saved in the specified directories with filenames that include the current timestamp for easy identification and organization.