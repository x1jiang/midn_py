#!/usr/bin/env Rscript

# Evaluation script to compare Python and R imputation results

# Load libraries
library(reticulate)

# Process command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript evaluate_results.R <python_output_file> <r_output_file>")
}

python_output_file <- args[1]
r_output_file <- args[2]

# Function to calculate evaluation metrics
calculate_metrics <- function(imputed, truth) {
  # Calculate Mean Squared Error or accuracy depending on data type
  if (is.numeric(truth) && !all(truth %in% c(0, 1))) {
    # Continuous data - MSE
    mse <- mean((imputed - truth)^2, na.rm = TRUE)
    rmse <- sqrt(mse)
    mae <- mean(abs(imputed - truth), na.rm = TRUE)
    return(list(mse = mse, rmse = rmse, mae = mae))
  } else {
    # Binary data - Accuracy
    accuracy <- mean(imputed == truth, na.rm = TRUE)
    # Calculate additional metrics for binary data
    tp <- sum(imputed == 1 & truth == 1, na.rm = TRUE)
    tn <- sum(imputed == 0 & truth == 0, na.rm = TRUE)
    fp <- sum(imputed == 1 & truth == 0, na.rm = TRUE)
    fn <- sum(imputed == 0 & truth == 1, na.rm = TRUE)
    
    precision <- if (tp + fp > 0) tp / (tp + fp) else NA
    recall <- if (tp + fn > 0) tp / (tp + fn) else NA
    f1 <- if (!is.na(precision) && !is.na(recall) && precision + recall > 0) 
           2 * precision * recall / (precision + recall) else NA
    
    return(list(accuracy = accuracy, precision = precision, recall = recall, f1 = f1))
  }
}

# Load Python results using reticulate
np <- import("numpy")
py_imputed <- np$load(python_output_file, allow_pickle = TRUE)
py_imputed_list <- lapply(1:dim(py_imputed)[1], function(i) py_imputed[i,,])

# Load R results
load(r_output_file)
# Assuming the R file contains a list named 'imp' with the imputations

# Find the ground truth data
method <- if (grepl("binary", python_output_file)) "binary" else "continuous"
truth_files <- list(
  binary = c(
    central = "SIMI/test_data/binary_central_truth.csv",
    remote1 = "SIMI/test_data/binary_remote1_truth.csv",
    remote2 = "SIMI/test_data/binary_remote2_truth.csv"
  ),
  continuous = c(
    central = "SIMI/test_data/continuous_central_truth.csv",
    remote1 = "SIMI/test_data/continuous_remote1_truth.csv",
    remote2 = "SIMI/test_data/continuous_remote2_truth.csv"
  )
)

# Load the truth data
central_truth <- read.csv(truth_files[[method]]["central"])
remote1_truth <- read.csv(truth_files[[method]]["remote1"])
remote2_truth <- read.csv(truth_files[[method]]["remote2"])

# Combine truth data
truth <- rbind(central_truth, remote1_truth, remote2_truth)

# Calculate metrics for each imputation from Python and R
py_metrics <- lapply(py_imputed_list, function(imp) calculate_metrics(imp[, 1], truth[, 1]))
r_metrics <- lapply(imp, function(imp) calculate_metrics(imp[, 1], truth[, 1]))

# Calculate average metrics across imputations
average_py_metrics <- Reduce(function(x, y) {
  mapply(function(a, b) a + b, x, y, SIMPLIFY = FALSE)
}, py_metrics)
average_py_metrics <- lapply(average_py_metrics, function(x) x / length(py_metrics))

average_r_metrics <- Reduce(function(x, y) {
  mapply(function(a, b) a + b, x, y, SIMPLIFY = FALSE)
}, r_metrics)
average_r_metrics <- lapply(average_r_metrics, function(x) x / length(r_metrics))

# Print results
cat("\n=== Evaluation Results ===\n")
cat("\nPython Implementation Metrics:\n")
print(average_py_metrics)

cat("\nR Implementation Metrics:\n")
print(average_r_metrics)

cat("\n=== Comparison ===\n")
cat("Metric differences (Python - R):\n")
metric_diff <- mapply(function(py, r) py - r, average_py_metrics, average_r_metrics)
print(metric_diff)

# Calculate relative difference percentage
rel_diff <- mapply(function(py, r) {
  if (r != 0) {
    100 * abs(py - r) / abs(r)
  } else {
    if (py == 0) 0 else Inf
  }
}, average_py_metrics, average_r_metrics)

cat("\nRelative difference (%):\n")
print(rel_diff)

# Save results
results <- list(
  python = average_py_metrics,
  r = average_r_metrics,
  difference = metric_diff,
  relative_difference_pct = rel_diff
)

output_csv <- gsub("\\.npy$|\\.RData$", "_evaluation.csv", python_output_file)
write.csv(do.call(rbind, results), output_csv)
cat(sprintf("\nResults saved to %s\n", output_csv))
