# Load necessary libraries
library(brms)
library(caret)

# Load the updated dataset
insurance_data <- read.csv("Updated_Encoded_Insurance_Fraud_Dataset.csv")

# Convert the target variable to a factor
insurance_data$fraud_reported <- as.factor(insurance_data$fraud_reported)

# Split the dataset into training and testing sets
set.seed(123)  # For reproducibility
train_indices <- sample(seq_len(nrow(insurance_data)), size = 0.6 * nrow(insurance_data))
train_data <- insurance_data[train_indices, ]
test_data <- insurance_data[-train_indices, ]

# =========================
# Build the Hierarchical Model
# =========================
hierarchical_model <- brm(
  formula = fraud_reported ~ total_claim_sum + policy_csl + policy_annual_premium + 
    umbrella_limit + 
    incident_severity_Minor.Damage + incident_severity_Total.Loss + 
    incident_severity_Trivial.Damage  + 
    (1 + total_claim_sum | policy_state),
  data = train_data,
  family = bernoulli(link = "logit"),
  backend = "cmdstanr",
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 123,
  prior = c(
    # Refined Priors for fixed effects (regression coefficients)
    prior(normal(0, 0.1), class = "b"),                           # Narrower prior for fixed effects
    prior(normal(-1, 0.2), class = "Intercept"),                  # More informative prior for intercept
    
    # Refined Priors for random effects (standard deviations)
    prior(exponential(4), class = "sd", group = "policy_state"),   # Stronger prior for random intercept
    prior(normal(0, 0.05), class = "sd", coef = "total_claim_sum", group = "policy_state"), # Tighter prior for random slope
    
    # Refined Prior for random effect correlations
    prior(lkj_corr(3), class = "cor", group = "policy_state")      # Stronger shrinkage toward zero correlation
  ),
  control = list(
    max_treedepth = 15,
    adapt_delta = 0.95
  )
)

# Summarize the Hierarchical Model
summary(hierarchical_model)

# Posterior Predictive Checks for Hierarchical Model
pp_check(hierarchical_model)

# =========================
# Save the Model for Future Use
# =========================
saveRDS(hierarchical_model, file = "hierarchical_model_updated.rds")

# Model Diagnostics: Trace plots and convergence
plot(hierarchical_model)

# Generate predictions on the test set
predicted_probs <- posterior_predict(hierarchical_model, newdata = test_data, re.form = NA)
# Aggregate predictions by averaging across posterior draws (rows of the matrix)
test_data$predicted_prob <- apply(predicted_probs, 2, mean)  # Column-wise mean

# Convert predicted probabilities to binary outcomes
test_data$predicted_class <- ifelse(test_data$predicted_prob > 0.5, 1, 0)

# Evaluate model performance using a confusion matrix
confusion_matrix <- confusionMatrix(
  as.factor(test_data$predicted_class), 
  as.factor(test_data$fraud_reported)
)

# Print the confusion matrix
print(confusion_matrix)

# Perform prior predictive checks
prior_model <- brm(
  formula = fraud_reported ~ total_claim_sum + policy_csl + policy_annual_premium + 
    umbrella_limit + 
    incident_severity_Minor.Damage + incident_severity_Total.Loss + 
    incident_severity_Trivial.Damage  + 
    (1 + total_claim_sum | policy_state),
  data = train_data,  # You can use the same training data for prior predictive checks
  family = bernoulli(link = "logit"),
  backend = "cmdstanr",
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 123,
  prior = c(
    # Priors for fixed effects (regression coefficients)
    prior(normal(0, 0.1), class = "b"),                           
    prior(normal(-1, 0.2), class = "Intercept"),                  
    
    # Priors for random effects (standard deviations)
    prior(exponential(4), class = "sd", group = "policy_state"),   
    prior(normal(0, 0.05), class = "sd", coef = "total_claim_sum", group = "policy_state"), 
    
    # Prior for random effect correlations
    prior(lkj_corr(3), class = "cor", group = "policy_state")      
  ),
  control = list(
    max_treedepth = 15,
    adapt_delta = 0.95
  ),
  sample_prior = "only"  # Sample only from the priors
)

# Generate prior predictive samples
prior_predictions <- posterior_predict(prior_model)

# Visualize prior predictive checks
pp_check(prior_model)

# Inspect the range and variability of simulated predictions
summary(apply(prior_predictions, 2, mean))  # Summary of column-wise mean predictions

# Plot histograms for observed vs. simulated data
hist(rowMeans(prior_predictions), breaks = 30, main = "Prior Predictive Distribution",
     xlab = "Predicted Fraud Probability", col = "lightblue", border = "black")
abline(v = mean(train_data$fraud_reported), col = "red", lwd = 2, lty = 2)  # Observed mean

# Generate posterior predictive samples
posterior_predictions <- posterior_predict(hierarchical_model)

# Visualize posterior predictive checks
pp_check(hierarchical_model)

# Additional plot: Histogram of posterior predictions vs. observed data
pp_check(hierarchical_model, type = "hist")

# Additional plot: Overlay of predicted and observed densities
pp_check(hierarchical_model, type = "dens_overlay")

# Custom Posterior Predictive Check: Compare summary statistics
observed_mean <- mean(train_data$fraud_reported)
predicted_means <- rowMeans(posterior_predictions)
hist(predicted_means, breaks = 30, col = "lightblue", main = "Posterior Predicted vs Observed Means",
     xlab = "Predicted Mean Fraud Probability")
abline(v = observed_mean, col = "red", lwd = 2, lty = 2)  # Observed mean

