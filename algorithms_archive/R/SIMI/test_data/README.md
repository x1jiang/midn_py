# Test Data for SIMI Model Evaluation

This directory contains synthetic test datasets for evaluating SIMI (Sequential Iterative Multiple Imputation) models with both binary and continuous outcomes.

## Dataset Structure

The datasets have the following structure:
- 9 predictor variables (x1-x9)
- 1 outcome variable (y)
- 3000 observations split across three sites:
  - Central site: 500 observations (rows 1-500)
  - Remote site 1: 1000 observations (rows 501-1500)
  - Remote site 2: 1500 observations (rows 1501-3000)

## Variable Description

- **x1**: Contains ~20% missing values (MAR - Missing At Random)
- **x2-x5**: Have linear relationships with x1, but with varying correlation strengths and noise levels
- **x6-x9**: Independent predictor variables
- **y**: 
  - Binary outcome in binary datasets
  - Continuous outcome in continuous datasets

## Data Generation Process

1. Generate x1 as a standard normal variable
2. Generate x2-x5 with linear relationships to x1 but with added randomness
3. Generate independent variables x6-x9
4. Generate outcome y based on all predictors with 20% random error
5. Create ~20% missing values in x1 based on MAR mechanism
6. Split data into central and remote site files

## File Description

### Binary Outcome Files
- `binary_full.csv`: Complete dataset with missing values
- `binary_full_truth.csv`: Complete dataset with no missing values (ground truth)
- `binary_central.csv`: Central site data with missing values
- `binary_central_truth.csv`: Central site data with no missing values (ground truth)
- `binary_remote1.csv`: Remote site 1 data with missing values
- `binary_remote1_truth.csv`: Remote site 1 data with no missing values (ground truth)
- `binary_remote2.csv`: Remote site 2 data with missing values
- `binary_remote2_truth.csv`: Remote site 2 data with no missing values (ground truth)

### Continuous Outcome Files
- `continuous_full.csv`: Complete dataset with missing values
- `continuous_full_truth.csv`: Complete dataset with no missing values (ground truth)
- `continuous_central.csv`: Central site data with missing values
- `continuous_central_truth.csv`: Central site data with no missing values (ground truth)
- `continuous_remote1.csv`: Remote site 1 data with missing values
- `continuous_remote1_truth.csv`: Remote site 1 data with no missing values (ground truth)
- `continuous_remote2.csv`: Remote site 2 data with missing values
- `continuous_remote2_truth.csv`: Remote site 2 data with no missing values (ground truth)

### Visualizations
The `visualizations` subfolder contains:
- Correlation plots showing relationships between variables
- Missing data pattern visualizations

## Usage

These datasets are intended for testing and evaluating the SIMI model implementation under different conditions. The ground truth files allow for assessment of imputation and model quality by comparing results against the actual values.

### Running the SIMI Imputation

To run the SIMI imputation and evaluate the results against the ground truth:

#### For Binary Data:

```bash
python ../../simi/SIMICentral.py logistic 0 binary_central.csv binary_central_truth.csv
```

#### For Continuous Data:

```bash
python ../../simi/SIMICentral.py gaussian 0 continuous_central.csv continuous_central_truth.csv
```

#### Parameters:

1. Method: `gaussian` or `logistic`
2. Missing variable index: `0` for x1, which has missing values
3. Data file: Path to CSV with missing values
4. Truth file: Path to CSV with complete ground truth values

The script performs 10 imputations and then:
- For continuous data: Calculates the average across imputations and RMSE against truth values
- For binary data: Takes majority vote across imputations and calculates accuracy

### Results

The script saves results in the `result` directory:
- `imp_[method]_PY.json`: Full results including all imputations and metrics
- `imp_[method]_PY_comparison.csv`: Simple CSV with truth and imputed values for easy comparison
