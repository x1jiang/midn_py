#!/bin/bash
# This script runs the unified imputation implementation for SIMI with Gaussian (continuous) data

# Directory setup
CURRENT_DIR=$(pwd)
OUTPUT_DIR="$CURRENT_DIR/SIMI/test_data/result"

# Kill any process using ports 8000 to 8002
for port in 8000; do
    pid=$(lsof -ti tcp:$port)
    if [ -n "$pid" ]; then
        echo "Killing process on port $port (PID $pid)"
        kill -9 $pid
    fi
done

# Create output directory if it doesn't exist
rm -rf "$OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

# Set parameters for continuous data
CENTRAL_DATA="$CURRENT_DIR/SIMI/test_data/continuous_central.csv"
REMOTE1_DATA="$CURRENT_DIR/SIMI/test_data/continuous_remote1.csv"
REMOTE2_DATA="$CURRENT_DIR/SIMI/test_data/continuous_remote2.csv"
CONFIG_FILE="$CURRENT_DIR/config_simi_gaussian.json"
PY_OUTPUT="$OUTPUT_DIR/central_imp"

# Run the unified imputation script
echo "Running unified imputation for SIMI (Gaussian)..."
echo "NOTE: You can manually kill the Python process from another terminal if needed"


# Run the Python process in foreground
python run_imputation.py \
	--algorithm SIMI \
	--config_file "$CONFIG_FILE" \
	--central_data "$CENTRAL_DATA" \
	--remote_data "$REMOTE1_DATA" "$REMOTE2_DATA" \
	--output "$PY_OUTPUT"

# Print completion message regardless of whether Python was killed or finished naturally
echo "Python execution completed or was manually killed. Continuing with evaluation..."

if command -v Rscript >/dev/null 2>&1; then
	echo "Evaluating imputation results..."
else
	echo "Rscript not found in PATH. Skipping R-based evaluation steps."; SKIP_R=1
fi

# Define the correct paths for truth data
TRUTH_DATA="${CENTRAL_DATA/continuous_central.csv/continuous_central_truth.csv}"
MVAR_VALUE=1  # Get from config file

# Check if any imputation files were created before running evaluation
IMPUTATION_FILES_EXIST=0
for m in $(seq 1 10); do  # m=10 from config file
    PADDED_M=$(printf "%02d" $m)
    if [ -f "${PY_OUTPUT}_${PADDED_M}.csv" ]; then
        IMPUTATION_FILES_EXIST=1
        break
    fi
done

if [ $IMPUTATION_FILES_EXIST -eq 1 ] && [ -z "$SKIP_R" ]; then
    Rscript -e '
        central_data <- as.matrix(read.csv("'$CENTRAL_DATA'"));
        mvar <- as.integer("'$MVAR_VALUE'");
        missing_mask <- is.na(central_data[, mvar]);
        truth_data <- as.matrix(read.csv("'$TRUTH_DATA'"));
        truth_values <- truth_data[missing_mask, mvar];
        
        # Find all imputation files that exist
        output_base <- "'$PY_OUTPUT'";
        max_imp <- 10;  # m=10 from config file
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
            consolidated_values <- colMeans(all_imputed_values);
            rmse <- sqrt(mean((consolidated_values - truth_values)^2));
            print(paste("RMSE based on average of", length(existing_files), "imputations:", round(rmse, 4)));
        } else {
            print("No imputation files found. Skipping RMSE calculation.");
        }
    '
else
    echo "No imputation files were found. Skipping evaluation."
fi


# Add one-round imputation using combined data from all sites
if [ -z "$SKIP_R" ]; then
echo "Performing one-round imputation using combined data from all sites using linear regression..."
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
	
	# Perform linear regression imputation
	# First impute any other missing values with means to prepare for regression
	for (j in 1:ncol(combined_data)) {
		if (j != mvar) {
			combined_data[is.na(combined_data[,j]), j] <- mean(combined_data[,j], na.rm = TRUE);
		}
	}
	
	# Build linear regression model from combined data (excluding rows with missing target var)
	complete_cases <- which(!is.na(combined_data[, mvar]));
	model_data <- combined_data[complete_cases, ];
	
	# Create formula based on column names
	col_names <- colnames(combined_data)
	target_var <- col_names[mvar]
	predictor_vars <- col_names[col_names != target_var]
	formula_str <- paste(target_var, "~", paste(predictor_vars, collapse = " + "));
	
	# Create linear regression model
	lm_model <- tryCatch({
		lm(as.formula(formula_str), data = model_data)
	}, error = function(e) {
		# If there are issues with the full model, try a simpler model
		cat("Error in full model, trying simpler model...\n")
		lm(as.formula(paste(target_var, "~ .")), data = model_data)
	})
	
	# Apply model to central data rows with missing values
	central_data_imputed <- central_data
	for (j in 1:ncol(central_data)) {
		if (j != mvar && any(is.na(central_data[,j]))) {
			central_data_imputed[is.na(central_data[,j]), j] <- mean(central_data[,j], na.rm = TRUE)
		}
	}
	
	# Impute missing values in central data using the model
	missing_rows <- which(missing_mask)
	if (length(missing_rows) > 0) {
		predicted_values <- tryCatch({
			predict(lm_model, newdata = central_data_imputed[missing_rows, ])
		}, error = function(e) {
			# Fallback to mean imputation if prediction fails
			cat("Prediction error, using mean imputation...\n")
			rep(mean(central_data[, mvar], na.rm = TRUE), length(missing_rows))
		})
		central_data_imputed[missing_rows, mvar] <- predicted_values
	}
	
	# Extract imputed values for the target column
	imputed_values <- central_data_imputed[missing_mask, mvar];
	
	# Calculate RMSE for the one-round imputation
	one_round_rmse <- sqrt(mean((imputed_values - truth_values)^2));
	
	# Just print the final RMSE in a clear format
	cat("----------------------------------------------\n")
	cat(sprintf("ONE-ROUND LINEAR REGRESSION IMPUTATION RMSE: %.4f\n", one_round_rmse))
	cat("----------------------------------------------\n")
'
echo "One-round imputation complete."
fi

# Clean up processes
for port in 8000; do
    pid=$(lsof -ti tcp:$port)
    if [ -n "$pid" ]; then
        echo "Killing process on port $port (PID $pid)"
        kill -9 $pid
    fi
done
