# Evaluate multiple imputed CSVs in a results folder and create a consolidated (ensemble) result.

cat("--- Batch Evaluation and Consolidation ---\n")

# 0) Configure paths
result_folder <- "./results/job_2_20250815_123857"
cat(sprintf("Result folder: %s\n", result_folder))

# 1) Load the central data (with missing) and the ground truth
central_data <- read.csv("./multi_missing_central.csv")
central_truth <- read.csv("./multi_missing_central_truth.csv")  # ground truth as provided

# Identify indices of originally missing entries in central data
central_x1_missing_idx <- which(is.na(central_data$x1))
central_x7_missing_idx <- which(is.na(central_data$x7))

cat(sprintf("Missing in x1: %d | Missing in x7: %d\n", length(central_x1_missing_idx), length(central_x7_missing_idx)))

if (length(central_x1_missing_idx) == 0 && length(central_x7_missing_idx) == 0) {
  stop("No missing values detected in central data. Nothing to evaluate.")
}

# 2) Find CSVs to evaluate
all_csvs <- list.files(result_folder, pattern = "\\\.csv$", full.names = TRUE)
# Keep only imputed datasets if present
csvs <- all_csvs[grepl("imputed_dataset_.*\\.csv$", basename(all_csvs))]
if (length(csvs) == 0) csvs <- all_csvs  # fallback to any csvs if naming differs

if (length(csvs) == 0) {
  stop(sprintf("No CSV files found in %s", result_folder))
}

cat(sprintf("Found %d CSV files to evaluate.\n", length(csvs)))

# Helpers
safe_as_numeric <- function(x) {
  if (is.factor(x)) return(as.numeric(as.character(x)))
  as.numeric(x)
}

compute_accuracy <- function(pred, truth) {
  v <- !is.na(pred) & !is.na(truth)
  if (!any(v)) return(NA_real_)
  mean(pred[v] == truth[v])
}

compute_rmse <- function(pred, truth) {
  v <- !is.na(pred) & !is.na(truth)
  if (!any(v)) return(NA_real_)
  sqrt(mean((pred[v] - truth[v])^2))
}

# 3) Evaluate each CSV and accumulate values for consolidation
metrics <- data.frame(
  file = character(),
  accuracy_x1 = numeric(),
  rmse_x7 = numeric(),
  stringsAsFactors = FALSE
)

# Accumulators for ensemble (only on missing indices), with per-index counts to handle NAs
sum_x1 <- if (length(central_x1_missing_idx) > 0) rep(0, length(central_x1_missing_idx)) else NULL
cnt_x1 <- if (length(central_x1_missing_idx) > 0) rep(0, length(central_x1_missing_idx)) else NULL

sum_x7 <- if (length(central_x7_missing_idx) > 0) rep(0, length(central_x7_missing_idx)) else NULL
cnt_x7 <- if (length(central_x7_missing_idx) > 0) rep(0, length(central_x7_missing_idx)) else NULL

for (f in csvs) {
  cat(sprintf("Evaluating: %s\n", basename(f)))
  df <- read.csv(f)

  # Ensure required columns exist
  required_cols <- c("x1", "x7")
  missing_cols <- setdiff(required_cols, colnames(df))
  if (length(missing_cols) > 0) {
    warning(sprintf("Skipping %s (missing columns: %s)", basename(f), paste(missing_cols, collapse = ", ")))
    next
  }

  # Prepare vectors for evaluation
  acc <- NA_real_
  r <- NA_real_

  if (length(central_x1_missing_idx) > 0) {
    pred_x1_raw <- safe_as_numeric(df$x1[central_x1_missing_idx])
    truth_x1 <- safe_as_numeric(central_truth$x1[central_x1_missing_idx])

    # If predictions look like probabilities, threshold; if already 0/1, this keeps them
    pred_x1_bin <- ifelse(is.na(pred_x1_raw), NA, ifelse(pred_x1_raw >= 0.5, 1, 0))

    acc <- compute_accuracy(pred_x1_bin, truth_x1)

    # Accumulate for ensemble voting, ignoring NAs per index
    valid <- !is.na(pred_x1_bin)
    if (any(valid)) {
      sum_x1[valid] <- sum_x1[valid] + pred_x1_bin[valid]
      cnt_x1[valid] <- cnt_x1[valid] + 1
    }
  }

  if (length(central_x7_missing_idx) > 0) {
    pred_x7 <- safe_as_numeric(df$x7[central_x7_missing_idx])
    truth_x7 <- safe_as_numeric(central_truth$x7[central_x7_missing_idx])

    r <- compute_rmse(pred_x7, truth_x7)

    # Accumulate for ensemble mean, ignoring NAs per index
    valid <- !is.na(pred_x7)
    if (any(valid)) {
      sum_x7[valid] <- sum_x7[valid] + pred_x7[valid]
      cnt_x7[valid] <- cnt_x7[valid] + 1
    }
  }

  metrics <- rbind(metrics, data.frame(file = basename(f), accuracy_x1 = acc, rmse_x7 = r, stringsAsFactors = FALSE))
}

# 4) Save per-file metrics
metrics_path <- file.path(result_folder, "metrics_summary.csv")
write.csv(metrics, metrics_path, row.names = FALSE)
cat(sprintf("Per-file metrics written to: %s\n", metrics_path))

# 5) Build consolidated (ensemble) result on the central data for missing positions only
final_imputed <- central_data
n_files <- nrow(metrics)  # number of evaluated files (could be < length(csvs) if some skipped)
if (n_files == 0) stop("No valid CSVs were evaluated; cannot build ensemble.")

if (length(central_x1_missing_idx) > 0) {
  frac_votes <- ifelse(cnt_x1 > 0, sum_x1 / cnt_x1, NA)
  # Majority vote with 0.5 threshold when available
  consensus_x1 <- ifelse(is.na(frac_votes), NA, ifelse(frac_votes >= 0.5, 1, 0))
  final_imputed$x1[central_x1_missing_idx] <- consensus_x1
}

if (length(central_x7_missing_idx) > 0) {
  mean_x7 <- ifelse(cnt_x7 > 0, sum_x7 / cnt_x7, NA)
  final_imputed$x7[central_x7_missing_idx] <- mean_x7
}

# 6) Evaluate consolidated result
final_acc <- NA_real_
final_rmse <- NA_real_

if (length(central_x1_missing_idx) > 0) {
  final_acc <- compute_accuracy(safe_as_numeric(final_imputed$x1[central_x1_missing_idx]),
                                safe_as_numeric(central_truth$x1[central_x1_missing_idx]))
}
if (length(central_x7_missing_idx) > 0) {
  final_rmse <- compute_rmse(safe_as_numeric(final_imputed$x7[central_x7_missing_idx]),
                             safe_as_numeric(central_truth$x7[central_x7_missing_idx]))
}

cat("\n--- Consolidated (Ensemble) Metrics ---\n")
cat(sprintf("Final Accuracy (x1): %s\n", ifelse(is.na(final_acc), "n/a", sprintf("%.4f", final_acc))))
cat(sprintf("Final RMSE (x7): %s\n", ifelse(is.na(final_rmse), "n/a", sprintf("%.4f", final_rmse))))

# 7) Save consolidated dataset and its metrics
final_csv_path <- file.path(result_folder, "ensemble_imputed.csv")
write.csv(final_imputed, final_csv_path, row.names = FALSE)
cat(sprintf("Ensembled imputed dataset written to: %s\n", final_csv_path))

final_metrics_path <- file.path(result_folder, "ensemble_metrics.csv")
write.csv(data.frame(accuracy_x1 = final_acc, rmse_x7 = final_rmse), final_metrics_path, row.names = FALSE)
cat(sprintf("Ensemble metrics written to: %s\n", final_metrics_path))

cat("\nDone.\n")
