#!/bin/bash
# This script runs the unified imputation implementation for SIMI with binary (logistic) data

# Directory setup
CURRENT_DIR=$(pwd)
OUTPUT_DIR="$CURRENT_DIR/SIMI/test_data/result"

# Kill any process using ports 8000 to 8002
for port in 8000 8001 8002; do
    pid=$(lsof -ti tcp:$port)
    if [ -n "$pid" ]; then
        echo "Killing process on port $port (PID $pid)"
        kill -9 $pid
    fi
done

# Create output directory if it doesn't exist
rm -rf "$OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

# Set parameters for binary data
CENTRAL_DATA="$CURRENT_DIR/SIMI/test_data/binary_central.csv"
REMOTE1_DATA="$CURRENT_DIR/SIMI/test_data/binary_remote1.csv"
REMOTE2_DATA="$CURRENT_DIR/SIMI/test_data/binary_remote2.csv"
CONFIG_FILE="$CURRENT_DIR/config_simi_binary.json"
PY_OUTPUT="$OUTPUT_DIR/central_imp"

# Run the unified imputation script
echo "Running unified imputation for SIMI (binary)..."
echo "NOTE: You can manually kill the Python process from another terminal if needed"

# Run the Python process in foreground
python run_imputation.py \
  --algorithm SIMI \
  --config_file "$CONFIG_FILE" \
  --central_data "$CENTRAL_DATA" \
  --remote_data "$REMOTE1_DATA" "$REMOTE2_DATA" \
  --output "$PY_OUTPUT" \
  --remote_ports 8001 8002

# Print completion message regardless of whether Python was killed or finished naturally
echo "Python execution completed or was manually killed. Continuing with evaluation..."

# Evaluate the imputation results using R (reusing existing evaluation code from run_SIMI_BIN_PY.sh)
echo "Evaluating imputation results..."

# Define the correct paths for truth data
TRUTH_DATA="${CENTRAL_DATA/binary_central.csv/binary_central_truth.csv}"
MVAR_VALUE=1  # Get from config file
M_VALUE=10  # Get from config file

# Check if any imputation files were created before running evaluation
IMPUTATION_FILES_EXIST=0
for m in $(seq 1 $M_VALUE); do
    PADDED_M=$(printf "%02d" $m)
    if [ -f "${PY_OUTPUT}_${PADDED_M}.csv" ]; then
        IMPUTATION_FILES_EXIST=1
        break
    fi
done

if [ $IMPUTATION_FILES_EXIST -eq 1 ]; then
    Rscript -e '
        central_data <- as.matrix(read.csv("'$CENTRAL_DATA'"));
        mvar <- as.integer("'$MVAR_VALUE'");
        missing_mask <- is.na(central_data[, mvar]);
        truth_data <- as.matrix(read.csv("'$TRUTH_DATA'"));
        truth_values <- truth_data[missing_mask, mvar];
        
        # Find all imputation files that exist
        output_base <- "'$PY_OUTPUT'";
        max_imp <- as.integer("'$M_VALUE'");
        existing_files <- c();
        for (m in 1:max_imp) {
            fname <- sprintf("%s_%02d.csv", output_base, m);
            if (file.exists(fname)) {
                existing_files <- c(existing_files, fname);
            }
        }
        
        if (length(existing_files) > 0) {
            all_imputed_values <- matrix(NA, nrow=length(existing_files), ncol=sum(missing_mask));
            for (i in 1:length(existing_files)) {
                imp_data <- as.matrix(read.csv(existing_files[i]));
                all_imputed_values[i,] <- imp_data[missing_mask, mvar];
            }
            
            # Check if we have binary or continuous data
            # Count unique values to determine data type
            all_values <- as.vector(all_imputed_values)
            unique_vals <- unique(all_values[!is.na(all_values)])
            is_binary <- length(unique_vals) <= 2 && all(unique_vals %in% c(0,1))
            
            if (is_binary) {
                print("Detected binary data - using majority vote consolidation")
                # For binary data, use majority vote to consolidate imputation values
                consolidated_values <- numeric(ncol(all_imputed_values));
                for (j in 1:ncol(all_imputed_values)) {
                    # Count occurrences of each value
                    values <- all_imputed_values[, j]
                    # Get the most frequent value (majority vote), with error handling
                    tab <- table(values)
                    if (length(tab) > 0) {
                        consolidated_values[j] <- as.numeric(names(sort(tab, decreasing=TRUE)[1]))
                    } else {
                        # If no values are available, use NA
                        consolidated_values[j] <- NA
                    }
                }
                
                # Calculate accuracy for binary data (filtering out NAs)
                valid_indices <- !is.na(consolidated_values)
                if (sum(valid_indices) > 0) {
                    accuracy <- mean(consolidated_values[valid_indices] == truth_values[valid_indices], na.rm=TRUE);
                    print(paste("Accuracy based on majority vote of", length(existing_files), "imputations (", sum(valid_indices), "valid cases ):", round(accuracy * 100, 2), "%"));
                    # Calculate error rate
                    error_rate <- 1 - accuracy;
                    print(paste("Error rate:", round(error_rate * 100, 2), "%"));
                } else {
                    print("No valid consolidated values found. Cannot calculate accuracy.")
                }
            } else {
                print("Detected continuous data - using mean consolidation")
                # For continuous data, use mean across imputations
                consolidated_values <- numeric(ncol(all_imputed_values));
                for (j in 1:ncol(all_imputed_values)) {
                    # Calculate mean of the values
                    values <- all_imputed_values[, j]
                    consolidated_values[j] <- mean(values, na.rm=TRUE)
                }
                
                # Calculate RMSE for continuous data
                valid_indices <- !is.na(consolidated_values)
                if (sum(valid_indices) > 0) {
                    rmse <- sqrt(mean((consolidated_values[valid_indices] - truth_values[valid_indices])^2, na.rm=TRUE));
                    print(paste("RMSE based on average of", length(existing_files), "imputations (", sum(valid_indices), "valid cases ):", round(rmse, 4)));
                } else {
                    print("No valid consolidated values found. Cannot calculate RMSE.")
                }
            }
        } else {
            print("No imputation files found. Skipping accuracy calculation.");
        }
    '
else
    echo "No imputation files were found. Skipping evaluation."
fi

# Add one-round imputation using combined data from all sites with logistic regression
echo "Performing one-round imputation using combined data from all sites using logistic regression..."
Rscript -e '
	# Read all datasets with headers
	central_data <- read.csv("'$CENTRAL_DATA'");
	remote1_data <- read.csv("'$REMOTE1_DATA'");
	remote2_data <- read.csv("'$REMOTE2_DATA'");
	truth_data <- read.csv("'$TRUTH_DATA'");
	
	# Identify missing values in central data
	mvar <- as.integer("'$MVAR_VALUE'");
	missing_mask <- is.na(central_data[, mvar]);
	truth_values <- truth_data[missing_mask, mvar];
	
	# Combine all data
	combined_data <- rbind(central_data, remote1_data, remote2_data);
	
	# Perform initial imputation for any other missing values
	# For binary data, use mode (most common value) for imputation
	for (j in 1:ncol(combined_data)) {
		if (j != mvar && any(is.na(combined_data[,j]))) {
			# Use mode for binary variables
			mode_val <- as.numeric(names(sort(table(combined_data[,j]), decreasing = TRUE)[1]))
			combined_data[is.na(combined_data[,j]), j] <- mode_val
		}
	}
	
	# Build logistic regression model from combined data (excluding rows with missing target var)
	complete_cases <- which(!is.na(combined_data[, mvar]));
	model_data <- combined_data[complete_cases, ];
	
	# Create formula based on column names
	col_names <- colnames(combined_data)
	target_var <- col_names[mvar]
	predictor_vars <- col_names[col_names != target_var]
	formula_str <- paste(target_var, "~", paste(predictor_vars, collapse = " + "));
	
	# Create logistic regression model
	glm_model <- tryCatch({
		glm(as.formula(formula_str), data = model_data, family = binomial(link = "logit"))
	}, error = function(e) {
		# If there are issues with the full model, try a simpler model
		cat("Error in full model, trying simpler model...\n")
		glm(as.formula(paste(target_var, "~ .")), data = model_data, family = binomial(link = "logit"))
	})
	
	# Apply model to central data rows with missing values
	central_data_imputed <- central_data
	for (j in 1:ncol(central_data)) {
		if (j != mvar && any(is.na(central_data[,j]))) {
			# Use mode for imputation of other variables
			mode_val <- as.numeric(names(sort(table(central_data[,j]), decreasing = TRUE)[1]))
			central_data_imputed[is.na(central_data[,j]), j] <- mode_val
		}
	}
	
	# Impute missing values in central data using the logistic model
	missing_rows <- which(missing_mask)
	if (length(missing_rows) > 0) {
		# Get predicted probabilities
		predicted_probs <- tryCatch({
			predict(glm_model, newdata = central_data_imputed[missing_rows, ], type = "response")
		}, error = function(e) {
			# Fallback to mode imputation if prediction fails
			cat("Prediction error, using mode imputation...\n")
			mode_val <- as.numeric(names(sort(table(central_data[, mvar], na.rm = TRUE), decreasing = TRUE)[1]))
			rep(mode_val, length(missing_rows))
		})
		
		# Convert probabilities to binary predictions (0/1)
		predicted_values <- ifelse(predicted_probs > 0.5, 1, 0)
		central_data_imputed[missing_rows, mvar] <- predicted_values
	}
	
	# Extract imputed values for the target column
	imputed_values <- central_data_imputed[missing_mask, mvar];
	
	# Calculate accuracy for binary data
	accuracy <- mean(imputed_values == truth_values);
	error_rate <- 1 - accuracy;
	
	# Just print the final accuracy in a clear format
	cat("----------------------------------------------\n")
	cat(sprintf("ONE-ROUND LOGISTIC REGRESSION IMPUTATION RESULTS:\n"))
	cat(sprintf("Accuracy: %.2f%%\n", accuracy * 100))
	cat(sprintf("Error rate: %.2f%%\n", error_rate * 100))
	cat("----------------------------------------------\n")
'
echo "One-round imputation complete."

# Clean up processes
for port in 8000 8001 8002; do
    pid=$(lsof -ti tcp:$port)
    if [ -n "$pid" ]; then
        echo "Killing process on port $port (PID $pid)"
        kill -9 $pid
    fi
done
