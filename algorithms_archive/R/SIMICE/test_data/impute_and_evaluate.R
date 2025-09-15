# Impute missing values for the central site data and evaluate the performance.

# --- 1. Load Data ---
cat("Loading data...\n")

# Load the central site data with missing values
central_data <- read.csv("./multi_missing_central.csv")

# Load the ground truth data for the central site for final evaluation
central_truth <- read.csv("./multi_missing_central_truth.csv")

# Load remote site data to help build a more robust imputation model
remote1_data <- read.csv("./multi_missing_remote1.csv")
remote2_data <- read.csv("./multi_missing_remote2.csv")

# Combine all datasets to train the imputation models
combined_data <- rbind(central_data, remote1_data, remote2_data)
cat("Data loaded and combined for modeling.\n")


# --- 2. Impute Missing Values ---

# Keep track of where the original missing values were in the central dataset
central_x1_missing_idx <- which(is.na(central_data$x1))
central_x7_missing_idx <- which(is.na(central_data$x7))

cat(length(central_x1_missing_idx), "missing values found in x1 for the central site.\n")
cat(length(central_x7_missing_idx), "missing values found in x7 for the central site.\n")

# Create a copy of the central data to fill with imputed values
imputed_central_data <- central_data

# --- Imputation for x1 (Binary) ---
cat("\nImputing binary variable x1...\n")
# The generation script shows x1 is related to x2, x3, x4, x5
# We will use logistic regression since x1 is binary.
# We train on all non-missing x1 values from the combined dataset.
x1_model <- glm(x1 ~ x2 + x3 + x4 + x5, data = combined_data, family = "binomial")

# Predict probabilities for the missing values in the central dataset
x1_pred_probs <- predict(x1_model, newdata = central_data[central_x1_missing_idx, ], type = "response")

# Convert probabilities to binary prediction (0 or 1) using a 0.5 threshold
x1_imputed_values <- ifelse(x1_pred_probs > 0.5, 1, 0)

# Fill in the missing values
imputed_central_data$x1[central_x1_missing_idx] <- x1_imputed_values
cat("Imputation for x1 complete.\n")


# --- Imputation for x7 (Continuous) ---
cat("\nImputing continuous variable x7...\n")
# The generation script shows x7 is related to x1, x2, x3, x6.
# We will use linear regression since x7 is continuous.
# We use the imputed version of x1 in our combined dataset for a better model.
combined_data_imputed_x1 <- combined_data
combined_data_imputed_x1$x1[is.na(combined_data_imputed_x1$x1)] <- ifelse(predict(x1_model, newdata = combined_data[is.na(combined_data$x1), ], type = "response") > 0.5, 1, 0)

x7_model <- lm(x7 ~ x1 + x2 + x3 + x6, data = combined_data_imputed_x1)

# Predict values for the missing x7 rows in the central dataset
x7_imputed_values <- predict(x7_model, newdata = imputed_central_data[central_x7_missing_idx, ])

# Fill in the missing values
imputed_central_data$x7[central_x7_missing_idx] <- x7_imputed_values
cat("Imputation for x7 complete.\n")


# --- 3. Evaluate Imputation Performance ---
cat("\n--- Evaluating Imputation Performance ---\n")

# --- Accuracy for x1 ---
# Get the true values for the rows that were missing
true_x1_values <- central_truth$x1[central_x1_missing_idx]
imputed_x1_subset <- imputed_central_data$x1[central_x1_missing_idx]

# Calculate accuracy
accuracy <- sum(true_x1_values == imputed_x1_subset) / length(true_x1_values)
cat(sprintf("Accuracy for imputed x1: %.4f\n", accuracy))

# --- Additional Diagnostics for x1 ---
cat("\n--- x1 Diagnostics ---\n")
cat(sprintf("Number of missing x1 values in central: %d\n", length(central_x1_missing_idx)))
cat(sprintf("True x1 distribution in missing cases: 0=%d, 1=%d\n", 
            sum(true_x1_values == 0), sum(true_x1_values == 1)))
cat(sprintf("Imputed x1 distribution: 0=%d, 1=%d\n", 
            sum(imputed_x1_subset == 0), sum(imputed_x1_subset == 1)))

# Check prediction probabilities
x1_pred_probs_debug <- predict(x1_model, newdata = central_data[central_x1_missing_idx, ], type = "response")
cat(sprintf("Prediction probability range: [%.4f, %.4f]\n", 
            min(x1_pred_probs_debug, na.rm=TRUE), max(x1_pred_probs_debug, na.rm=TRUE)))
cat(sprintf("Mean prediction probability: %.4f\n", mean(x1_pred_probs_debug, na.rm=TRUE)))

# Check if threshold 0.5 is appropriate
cat("Probability distribution of predictions:\n")
cat(sprintf("  < 0.3: %d, 0.3-0.5: %d, 0.5-0.7: %d, > 0.7: %d\n",
            sum(x1_pred_probs_debug < 0.3, na.rm=TRUE),
            sum(x1_pred_probs_debug >= 0.3 & x1_pred_probs_debug < 0.5, na.rm=TRUE),
            sum(x1_pred_probs_debug >= 0.5 & x1_pred_probs_debug < 0.7, na.rm=TRUE),
            sum(x1_pred_probs_debug >= 0.7, na.rm=TRUE)))

# Check model quality
cat("x1 model summary:\n")
print(summary(x1_model)$coefficients)


# --- RMSE for x7 ---
# Get the true values for the rows that were missing
true_x7_values <- central_truth$x7[central_x7_missing_idx]
imputed_x7_subset <- imputed_central_data$x7[central_x7_missing_idx]

# Calculate RMSE
rmse <- sqrt(mean((true_x7_values - imputed_x7_subset)^2))
cat(sprintf("RMSE for imputed x7: %.4f\n", rmse))

cat("\nEvaluation complete.\n")
