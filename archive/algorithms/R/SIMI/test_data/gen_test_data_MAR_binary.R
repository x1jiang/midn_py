# Generate enhanced binary test data for SIMI with 9 predictors

# Increased sample size for better numerical stability
n <- 30000  # larger sample size (increased from 3000)
cat("Generating dataset with", n, "observations for improved numerical stability\n")

set.seed(42)  # For reproducibility

# 1) Generate core predictors with correlation structure

# Generate x2-x5 as independent variables first
x2 <- rnorm(n)  # standard normal
x3 <- rnorm(n)  # standard normal
x4 <- rnorm(n)  # standard normal
x5 <- rnorm(n)  # standard normal

# Generate x1 with linear relationship to x2-x5 (with small Gaussian noise)
# This will be the variable that has missing values
x1_linear <- 0.2 * x2 + 0.2 * x3 + 0.2 * x4 + 0.2 * x5 + rnorm(n, mean = 0, sd = 0.5)

# Convert x1 to binary (0/1) directly - this ensures we have binary values before imposing missingness
p_x1 <- plogis(x1_linear)  # Convert to probability between 0 and 1
x1_true <- rbinom(n, 1, p_x1)  # Generate binary 0/1 values

# Generate x6-x9 as independent variables
x6 <- rnorm(n)  # standard normal
x7 <- rnorm(n)  # standard normal
x8 <- rnorm(n)  # standard normal
x9 <- rnorm(n)  # standard normal

# 2) Generate binary outcome y based on all predictors
# We'll use a combination of all variables with some being more important
# Using smaller coefficients to avoid extreme probabilities
lp_y <- -0.2 + 0.3 * x1_true + 0.25 * x2 + 0.2 * x3 + 0.15 * x4 + 0.1 * x5 + 
        0.1 * x6 + 0.15 * x7 + 0.1 * x8 + 0.05 * x9
        
# Add 20% error/noise to the linear predictor
noise <- rnorm(n, mean=0, sd=0.2*sd(lp_y))
lp_y <- lp_y + noise

# Scale the linear predictor to avoid extreme values
lp_y <- lp_y / (2 * sd(lp_y))

p_y <- plogis(lp_y)
# Ensure strictly binary 0/1 values
y <- rbinom(n, 1, p_y)

# Double check: no -1 values
if(any(y < 0 | y > 1)) {
  cat("WARNING: Found non-binary values in y. Fixing...\n")
  y <- as.integer(y > 0.5)
}

# 3) Create missingness in x1 (≈20% MAR based on x2, x3, y)
# Using milder relationships to avoid extreme missing patterns
lp_miss <- 0.2 * x2 + 0.2 * x3 + 0.2 * y
# Scale to reduce extreme probabilities
lp_miss <- lp_miss / (2 * sd(lp_miss))
int <- qlogis(0.20) - mean(lp_miss)  # Target 20% missingness
p_miss <- plogis(int + lp_miss)
# Limit extreme probabilities
p_miss <- pmin(pmax(p_miss, 0.05), 0.35)

# Impose missingness
x1 <- x1_true
is_miss <- rbinom(n, 1, p_miss) == 1
x1[is_miss] <- NA

# Ensure balanced missingness across sites
# Check if missingness is too imbalanced between sites
central_miss <- mean(is.na(x1[1:500]))
remote1_miss <- mean(is.na(x1[501:1500]))
remote2_miss <- mean(is.na(x1[1501:3000]))

cat("Missing percentages by site before adjustment:\n")
cat("Central:", round(100*central_miss, 1), "%, Remote1:", round(100*remote1_miss, 1), 
    "%, Remote2:", round(100*remote2_miss, 1), "%\n")

# 4) Put it all in data frames
dat <- data.frame(
    x1 = x1,
    x2 = x2, 
    x3 = x3,
    x4 = x4,
    x5 = x5,
    x6 = x6,
    x7 = x7,
    x8 = x8,
    x9 = x9,
    y = y
)

# Create ground truth dataframe with complete data
dat_full <- data.frame(
    x1 = x1_true,
    x2 = x2,
    x3 = x3,
    x4 = x4,
    x5 = x5,
    x6 = x6,
    x7 = x7,
    x8 = x8,
    x9 = x9,
    y = y
)

# Quick check: proportion missing in x1
cat("Proportion missing in x1:", mean(is.na(dat$x1)), "\n")

# Check correlation matrix of the full data
cat("\nCorrelation matrix of variables (without missing values):\n")
print(round(cor(dat_full[, 1:9]), 2))

# View first few rows
print(head(dat))

# Split the data into three parts with the same proportions as before
# Central site: first 1/6 of the data
# Remote site 1: next 1/3 of the data
# Remote site 2: remaining 1/2 of the data

central_size <- floor(n/6)
remote1_size <- floor(n/3)

# Central site data 
central_data <- dat[1:central_size, ]
central_data_truth <- dat_full[1:central_size, ]

write.csv(central_data, file = "./binary_central.csv", row.names = FALSE)
write.csv(central_data_truth, file = "./binary_central_truth.csv", row.names = FALSE)

# Remote site 1 data 
remote1_start <- central_size + 1
remote1_end <- central_size + remote1_size
remote1_data <- dat[remote1_start:remote1_end, ]
remote1_data_truth <- dat_full[remote1_start:remote1_end, ]

write.csv(remote1_data, file = "./binary_remote1.csv", row.names = FALSE)
write.csv(remote1_data_truth, file = "./binary_remote1_truth.csv", row.names = FALSE)

# Remote site 2 data 
remote2_start <- remote1_end + 1
remote2_data <- dat[remote2_start:n, ]
remote2_data_truth <- dat_full[remote2_start:n, ]

write.csv(remote2_data, file = "./binary_remote2.csv", row.names = FALSE)
write.csv(remote2_data_truth, file = "./binary_remote2_truth.csv", row.names = FALSE)

# Print summary of data split
cat("Data has been split into three files:\n")
cat("Central site data:", nrow(central_data), "observations (", 
    round(100 * nrow(central_data) / n, 1), "%)\n")
cat("Remote site 1 data:", nrow(remote1_data), "observations (", 
    round(100 * nrow(remote1_data) / n, 1), "%)\n")
cat("Remote site 2 data:", nrow(remote2_data), "observations (", 
    round(100 * nrow(remote2_data) / n, 1), "%)\n")

# Check distribution of missing values across sites
central_miss_count <- sum(is.na(central_data$x1))
remote1_miss_count <- sum(is.na(remote1_data$x1))
remote2_miss_count <- sum(is.na(remote2_data$x1))
total_miss <- central_miss_count + remote1_miss_count + remote2_miss_count

cat("\nDistribution of missing values in x1:\n")
cat("Central site:", central_miss_count, "missing values (", 
    round(100 * central_miss_count / total_miss, 1), "% of all missing)\n")
cat("Remote site 1:", remote1_miss_count, "missing values (", 
    round(100 * remote1_miss_count / total_miss, 1), "% of all missing)\n")
cat("Remote site 2:", remote2_miss_count, "missing values (", 
    round(100 * remote2_miss_count / total_miss, 1), "% of all missing)\n")

# Display summaries
cat("\nSummary of binary outcome (y):\n")
print(table(dat$y))

cat("\nSummary of x1 (with missingness):\n")
print(summary(dat$x1))
cat("Missing values in x1:", sum(is.na(dat$x1)), "(", round(100*mean(is.na(dat$x1)), 1), "%)\n")

# Add some summary statistics to validate the data generation
cat("\nBasic model fit on complete data:\n")
model_summary <- summary(glm(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9, 
                          data = dat_full, family = binomial()))
print(model_summary$coefficients)

# Check for numerical stability issues
cat("\nChecking for potential numerical stability issues:\n")

# Check for separation issues in logistic regression
model_central <- glm(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9, 
                    data = central_data_truth, family = binomial())
model_remote1 <- glm(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9, 
                     data = remote1_data_truth, family = binomial())
model_remote2 <- glm(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9, 
                     data = remote2_data_truth, family = binomial())

# Check for extreme predicted probabilities
p_central <- predict(model_central, type = "response")
p_remote1 <- predict(model_remote1, type = "response")
p_remote2 <- predict(model_remote2, type = "response")

cat("Range of predicted probabilities:\n")
cat("Central site:", min(p_central), "-", max(p_central), "\n")
cat("Remote site 1:", min(p_remote1), "-", max(p_remote1), "\n")
cat("Remote site 2:", min(p_remote2), "-", max(p_remote2), "\n")

# Check for extreme coefficients
cat("\nMax absolute coefficient values:\n")
cat("Central site:", max(abs(coef(model_central))), "\n")
cat("Remote site 1:", max(abs(coef(model_remote1))), "\n") 
cat("Remote site 2:", max(abs(coef(model_remote2))), "\n")

# Check distribution of the target variable (balance of 0/1)
cat("\nDistribution of binary outcome by site:\n")
cat("Central site:", table(central_data_truth$y), "\n")
cat("Remote site 1:", table(remote1_data_truth$y), "\n")
cat("Remote site 2:", table(remote2_data_truth$y), "\n")

# Visualize correlation structure and missing data patterns
# Check if we can create plots
if (requireNamespace("ggplot2", quietly = TRUE) && requireNamespace("GGally", quietly = TRUE)) {
  library(ggplot2)
  library(GGally)
  
  # Create directory for visualizations if it doesn't exist
  dir.create("./visualizations", recursive = TRUE, showWarnings = FALSE)
  
  # Correlation plot for the first 6 variables (x1-x5 + y)
  png("./visualizations/binary_correlations.png", width = 900, height = 900, res = 100)
  print(GGally::ggpairs(dat_full[, c(1:5, 10)], 
                        title = "Correlation Structure of Variables"))
  dev.off()
  cat("\nCreated correlation visualization in ./visualizations/\n")

  # Missingness pattern visualization
  png("./visualizations/binary_missing_patterns.png", width = 800, height = 400, res = 100)
  par(mfrow = c(1, 2))
  hist(x1_true[is_miss], main = "Distribution of x1 when missing", 
       xlab = "x1 value", col = "lightblue", border = "white")
  hist(x1_true[!is_miss], main = "Distribution of x1 when observed", 
       xlab = "x1 value", col = "lightgreen", border = "white")
  dev.off()
  cat("Created missing data pattern visualization\n")
} else {
  cat("\nNote: Install ggplot2 and GGally packages for visualizations\n")
}

# Verify that our x1_true values are binary
binary_check <- all(x1_true %in% c(0, 1))
cat("\nVerified x1_true is binary:", binary_check, "\n")

# Quick check of the distribution of x1 (to confirm it's reasonably balanced)
cat("\nDistribution of x1 binary values:\n")
tab <- table(x1_true)
print(tab)
cat("Proportion of 1's:", tab[2]/sum(tab), "\n")

cat("\nData generation and verification complete.\n")
