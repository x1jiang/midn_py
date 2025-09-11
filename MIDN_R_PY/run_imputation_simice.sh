#!/bin/bash
# This script runs the unified imputation implementation for SIMICE

# Directory setup
CURRENT_DIR=$(pwd)
RESULT_DIR="$CURRENT_DIR/SIMICE/test_data/result"
REMOTE1_DATA="$CURRENT_DIR/SIMICE/test_data/multi_missing_remote1.csv"
REMOTE2_DATA="$CURRENT_DIR/SIMICE/test_data/multi_missing_remote2.csv"
CENTRAL_DATA="$CURRENT_DIR/SIMICE/test_data/multi_missing_central.csv"
TRUTH_DATA="$CURRENT_DIR/SIMICE/test_data/multi_missing_central_truth.csv"
CONFIG_FILE="$CURRENT_DIR/config_simice.json"

# SIMICE parameters (match the ones from run_SIMICE_PY.sh)
M_VALUE=12
MVAR="1 7"  # Space-separated indices for Python CLI
TYPE="logistic Gaussian"  # Space-separated types for Python CLI
ITER=6
ITER0=7

# Update the config file with the correct parameters
cat > "$CONFIG_FILE" << EOL
{
    "mvar": [1, 7],
    "type_list": ["logistic", "Gaussian"],
    "m": $M_VALUE,
    "iter": $ITER,
    "iter0": $ITER0,
    "hosts": ["localhost", "localhost"]
}
EOL

echo "Created SIMICE config file at $CONFIG_FILE"

# Define ports - consistent configuration across all components
CENTRAL_PORT=8000
CENTRAL_HOST="localhost"

# Kill any process using ports 8000 to 8002
for port in 8000; do
    pid=$(lsof -ti tcp:$port)
    if [ -n "$pid" ]; then
        echo "Killing process on port $port (PID $pid)"
        kill -9 $pid
    fi
done

# Create result directory
rm -rf "$RESULT_DIR"
mkdir -p "$RESULT_DIR"

# Run the unified imputation script for SIMICE
echo "Running unified imputation for SIMICE..."
echo "NOTE: You can manually kill the Python process from another terminal if needed"

# Run the Python process in foreground
python run_imputation.py \
	--algorithm SIMICE \
	--config_file "$CONFIG_FILE" \
	--central_data "$CENTRAL_DATA" \
	--remote_data "$REMOTE1_DATA" "$REMOTE2_DATA" \
	--output "$RESULT_DIR" \
	--output_dir "$RESULT_DIR" \
	--central_host $CENTRAL_HOST \
	--central_port $CENTRAL_PORT

# Print completion message regardless of whether Python was killed or finished naturally
echo "Python execution completed or was manually killed. Continuing with evaluation..."

# Evaluate the imputation results using R (copied from run_SIMICE_PY.sh)
echo "Evaluating imputation results..."

# Get M value from config file
M_VALUE=12  # Should match the value in config_simice.json

# Check if any imputation files were created before running evaluation
IMPUTATION_FILES_EXIST=0
for m in $(seq 1 $M_VALUE); do
    PADDED_M=$(printf "%02d" $m)
    if [ -f "${RESULT_DIR}/central_imp_${PADDED_M}.csv" ]; then
        IMPUTATION_FILES_EXIST=1
        break
    fi
done

if [ $IMPUTATION_FILES_EXIST -eq 1 ]; then
    Rscript -e 'source("Core/LS.R"); source("Core/Logit.R");
        central_data <- as.matrix(read.csv("'$CENTRAL_DATA'"));
        mvar <- c(1,7);
        types <- c("logistic", "Gaussian");
        
        # Create separate evaluation for each missing variable
        cat("----------------------------------------------\n")
        cat("SIMICE PYTHON IMPUTATION RESULTS:\n")
        
        for (var_idx in seq_along(mvar)) {
          curr_var <- mvar[var_idx];
          curr_type <- types[var_idx];
          
          cat(sprintf("\nVariable %d (%s):\n", curr_var, curr_type))
          
          missing_mask <- is.na(central_data[, curr_var]);
          if (sum(missing_mask) == 0) {
            cat(sprintf("No missing values found for variable %d\n", curr_var))
            next
          }
          
          truth_data <- as.matrix(read.csv("'$TRUTH_DATA'"));
          truth_values <- truth_data[missing_mask, curr_var];
          all_imputed_values <- matrix(NA, nrow=as.integer("'$M_VALUE'"), ncol=sum(missing_mask));
          
          for (m in 1:as.integer("'$M_VALUE'")) {
            fname <- sprintf("%s/central_imp_%02d.csv", "'$RESULT_DIR'", m);
            if (file.exists(fname)) {
              imp_data <- as.matrix(read.csv(fname));
              all_imputed_values[m,] <- imp_data[missing_mask, curr_var];
            }
          }
          
          if (curr_type == "logistic") {
            # For binary data, use majority vote to consolidate imputation values
            consolidated_values <- numeric(ncol(all_imputed_values));
            for (j in 1:ncol(all_imputed_values)) {
              # Count occurrences of each value
              values <- all_imputed_values[, j]
              # Get the most frequent value (majority vote)
              consolidated_values[j] <- as.numeric(names(sort(table(values), decreasing=TRUE)[1]))
            }
            # Calculate accuracy for binary data
            accuracy <- mean(consolidated_values == truth_values);
            cat(sprintf("Accuracy: %.2f%%\n", accuracy * 100))
            cat(sprintf("Error rate: %.2f%%\n", (1-accuracy) * 100))
          } else if (curr_type == "Gaussian") {
            # For continuous data, use mean imputation value
            mean_imputed_values <- colMeans(all_imputed_values)
            
            # Calculate RMSE
            rmse <- sqrt(mean((mean_imputed_values - truth_values)^2))
            # Calculate MAE
            mae <- mean(abs(mean_imputed_values - truth_values))
            
            cat(sprintf("RMSE: %.4f\n", rmse))
            cat(sprintf("MAE: %.4f\n", mae))
            
            # Calculate R-squared
            ss_total <- sum((truth_values - mean(truth_values))^2)
            ss_residual <- sum((truth_values - mean_imputed_values)^2)
            r_squared <- 1 - (ss_residual / ss_total)
            cat(sprintf("R-squared: %.4f\n", r_squared))
          }
        }
        cat("----------------------------------------------\n")
    '
else
    echo "No imputation files were found. Skipping evaluation."
fi

# One-round imputation using central data only (copied from run_SIMICE_PY.sh)
echo "Performing one-round imputation using central data only..."
Rscript -e 'source("Core/LS.R"); source("Core/Logit.R");
	# Read central dataset and truth data
	central_data <- read.csv("'$CENTRAL_DATA'");
	truth_data <- read.csv("'$TRUTH_DATA'");
	
	# Identify missing variables and their types
	mvar <- c(1,7);
	types <- c("logistic", "Gaussian");
	
	cat("----------------------------------------------\n")
	cat("ONE-ROUND IMPUTATION RESULTS Based on Central Data Only:\n")

	# Process each missing variable separately
	for (var_idx in seq_along(mvar)) {
		curr_var <- mvar[var_idx];
		curr_type <- types[var_idx];
		
		cat(sprintf("\nVariable %d (%s):\n", curr_var, curr_type))
		
		# Check if there are missing values for this variable
		missing_mask <- is.na(central_data[, curr_var]);
		if (sum(missing_mask) == 0) {
			cat(sprintf("No missing values found for variable %d\n", curr_var))
			next
		}
		
		# Get truth values for evaluation
		truth_values <- truth_data[missing_mask, curr_var];
		
		# Use only central data with non-missing values for the target variable
		complete_cases <- which(!is.na(central_data[, curr_var]));
		model_data <- central_data[complete_cases, ];
		
		# Skip if not enough data to build model
		if (length(complete_cases) < 10) {
			cat("Not enough complete cases in central data to build a model.\n")
			next
		}
		
		# Create formula based on column names
		col_names <- colnames(central_data)
		target_var <- col_names[curr_var]
		predictor_vars <- col_names[col_names != target_var]
		formula_str <- paste(target_var, "~", paste(predictor_vars, collapse = " + "));
		
		central_data_imputed <- central_data
		
		# Prepare the data for prediction (impute other missing variables first)
		for (j in 1:ncol(central_data)) {
			if (j != curr_var && any(is.na(central_data[,j]))) {
				val <- mean(central_data[,j], na.rm=TRUE)
				central_data_imputed[is.na(central_data[,j]), j] <- val
			}
		}
		
		# Impute missing values in central data
		missing_rows <- which(missing_mask)
		
		if (length(missing_rows) > 0) {
			if (curr_type == "logistic") {
				# Create logistic regression model
				glm_model <- tryCatch({
					glm(as.formula(formula_str), data = model_data, family = binomial(link = "logit"))
				}, error = function(e) {
					cat("Error in full model, trying simpler model...\n")
					glm(as.formula(paste(target_var, "~ .")), data = model_data, family = binomial(link = "logit"))
				})
				
				# Get predicted probabilities
				predicted_probs <- tryCatch({
					predict(glm_model, newdata = central_data_imputed[missing_rows, ], type = "response")
				}, error = function(e) {
					cat("Prediction error, using mode imputation...\n")
					mode_val <- as.numeric(names(sort(table(central_data[, curr_var], na.rm = TRUE), decreasing = TRUE)[1]))
					rep(mode_val, length(missing_rows))
				})
				
				# Convert probabilities to binary predictions
				predicted_values <- ifelse(predicted_probs > 0.5, 1, 0)
				central_data_imputed[missing_rows, curr_var] <- predicted_values
				
				# Calculate accuracy
				imputed_values <- central_data_imputed[missing_mask, curr_var];
				accuracy <- mean(imputed_values == truth_values);
				
				cat(sprintf("Accuracy: %.2f%%\n", accuracy * 100))
				cat(sprintf("Error rate: %.2f%%\n", (1-accuracy) * 100))
				
			} else if (curr_type == "Gaussian") {
				# Create linear regression model
				lm_model <- tryCatch({
					lm(as.formula(formula_str), data = model_data)
				}, error = function(e) {
					cat("Error in full model, trying simpler model...\n")
					lm(as.formula(paste(target_var, "~ .")), data = model_data)
				})
				
				# Predict values
				predicted_values <- tryCatch({
					predict(lm_model, newdata = central_data_imputed[missing_rows, ])
				}, error = function(e) {
					cat("Prediction error, using mean imputation...\n")
					mean_val <- mean(central_data[, curr_var], na.rm = TRUE)
					rep(mean_val, length(missing_rows))
				})
				
				central_data_imputed[missing_rows, curr_var] <- predicted_values
				
				# Calculate metrics
				imputed_values <- central_data_imputed[missing_mask, curr_var];
				rmse <- sqrt(mean((imputed_values - truth_values)^2))
				mae <- mean(abs(imputed_values - truth_values))
				
				cat(sprintf("RMSE: %.4f\n", rmse))
				cat(sprintf("MAE: %.4f\n", mae))
				
				ss_total <- sum((truth_values - mean(truth_values))^2)
				ss_residual <- sum((truth_values - imputed_values)^2)
				r_squared <- 1 - (ss_residual / ss_total)
				cat(sprintf("R-squared: %.4f\n", r_squared))
			}
		}
	}
	cat("----------------------------------------------\n")
'
echo "One-round imputation with central data only complete."

# Add one-round imputation using combined data from all sites (copied from run_SIMICE_PY.sh)
echo "Performing one-round imputation using combined data from all sites..."
Rscript -e 'source("Core/LS.R"); source("Core/Logit.R");
	# Read all datasets with headers
	central_data <- read.csv("'$CENTRAL_DATA'");
	remote1_data <- read.csv("'$REMOTE1_DATA'");
	remote2_data <- read.csv("'$REMOTE2_DATA'");
	truth_data <- read.csv("'$TRUTH_DATA'");
	
	# Identify missing variables and their types
	mvar <- c(1,7);
	types <- c("logistic", "Gaussian");
	
	# Combine all data
	combined_data <- rbind(central_data, remote1_data, remote2_data);
	
	# Perform initial imputation for any other missing values
	for (j in 1:ncol(combined_data)) {
		if (!(j %in% mvar) && any(is.na(combined_data[,j]))) {
			# Use mean for numeric variables
			mode_val <- mean(combined_data[,j], na.rm=TRUE)
			combined_data[is.na(combined_data[,j]), j] <- mode_val
		}
	}
	
	cat("----------------------------------------------\n")
	cat("ONE-ROUND IMPUTATION RESULTS:\n")
	
	# Process each missing variable separately
	for (var_idx in seq_along(mvar)) {
		curr_var <- mvar[var_idx];
		curr_type <- types[var_idx];
		
		cat(sprintf("\nVariable %d (%s):\n", curr_var, curr_type))
		
		# Check if there are missing values for this variable
		missing_mask <- is.na(central_data[, curr_var]);
		if (sum(missing_mask) == 0) {
			cat(sprintf("No missing values found for variable %d\n", curr_var))
			next
		}
		
		# Get truth values for evaluation
		truth_values <- truth_data[missing_mask, curr_var];
		
		# Build model from combined data (excluding rows with missing target var)
		complete_cases <- which(!is.na(combined_data[, curr_var]));
		model_data <- combined_data[complete_cases, ];
		
		# Create formula based on column names
		col_names <- colnames(combined_data)
		target_var <- col_names[curr_var]
		predictor_vars <- col_names[col_names != target_var]
		formula_str <- paste(target_var, "~", paste(predictor_vars, collapse = " + "));
		
		central_data_imputed <- central_data
		
		# Prepare the data for prediction
		for (j in 1:ncol(central_data)) {
			if (j != curr_var && any(is.na(central_data[,j]))) {
				# Use mean for imputation of other variables
				val <- mean(central_data[,j], na.rm=TRUE)
				central_data_imputed[is.na(central_data[,j]), j] <- val
			}
		}
		
		# Impute missing values in central data using the appropriate model
		missing_rows <- which(missing_mask)
		
		if (length(missing_rows) > 0) {
			if (curr_type == "logistic") {
				# Create logistic regression model
				glm_model <- tryCatch({
					glm(as.formula(formula_str), data = model_data, family = binomial(link = "logit"))
				}, error = function(e) {
					# If there are issues with the full model, try a simpler model
					cat("Error in full model, trying simpler model...\n")
					glm(as.formula(paste(target_var, "~ .")), data = model_data, family = binomial(link = "logit"))
				})
				
				# Get predicted probabilities
				predicted_probs <- tryCatch({
					predict(glm_model, newdata = central_data_imputed[missing_rows, ], type = "response")
				}, error = function(e) {
					# Fallback to mode imputation if prediction fails
					cat("Prediction error, using mode imputation...\n")
					mode_val <- as.numeric(names(sort(table(central_data[, curr_var], na.rm = TRUE), decreasing = TRUE)[1]))
					rep(mode_val, length(missing_rows))
				})
				
				# Convert probabilities to binary predictions (0/1)
				predicted_values <- ifelse(predicted_probs > 0.5, 1, 0)
				central_data_imputed[missing_rows, curr_var] <- predicted_values
				
				# Calculate accuracy for binary data
				imputed_values <- central_data_imputed[missing_mask, curr_var];
				accuracy <- mean(imputed_values == truth_values);
				
				cat(sprintf("Accuracy: %.2f%%\n", accuracy * 100))
				cat(sprintf("Error rate: %.2f%%\n", (1-accuracy) * 100))
				
			} else if (curr_type == "Gaussian") {
				# Create linear regression model
				lm_model <- tryCatch({
					lm(as.formula(formula_str), data = model_data)
				}, error = function(e) {
					# If there are issues with the full model, try a simpler model
					cat("Error in full model, trying simpler model...\n")
					lm(as.formula(paste(target_var, "~ .")), data = model_data)
				})
				
				# Predict values
				predicted_values <- tryCatch({
					predict(lm_model, newdata = central_data_imputed[missing_rows, ])
				}, error = function(e) {
					# Fallback to mean imputation if prediction fails
					cat("Prediction error, using mean imputation...\n")
					mean_val <- mean(central_data[, curr_var], na.rm = TRUE)
					rep(mean_val, length(missing_rows))
				})
				
				central_data_imputed[missing_rows, curr_var] <- predicted_values
				
				# Calculate metrics for continuous data
				imputed_values <- central_data_imputed[missing_mask, curr_var];
				
				# Calculate RMSE
				rmse <- sqrt(mean((imputed_values - truth_values)^2))
				# Calculate MAE
				mae <- mean(abs(imputed_values - truth_values))
				
				cat(sprintf("RMSE: %.4f\n", rmse))
				cat(sprintf("MAE: %.4f\n", mae))
				
				# Calculate R-squared
				ss_total <- sum((truth_values - mean(truth_values))^2)
				ss_residual <- sum((truth_values - imputed_values)^2)
				r_squared <- 1 - (ss_residual / ss_total)
				cat(sprintf("R-squared: %.4f\n", r_squared))
			}
		}
	}
	cat("----------------------------------------------\n")
'
echo "One-round imputation complete."

# Clean up processes
for port in 8000; do
    pid=$(lsof -ti tcp:$port)
    if [ -n "$pid" ]; then
        echo "Killing process on port $port (PID $pid)"
        kill -9 $pid
    fi
done

echo "All processes terminated. SIMICE implementation completed."
