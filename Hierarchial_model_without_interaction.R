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
  data = insurance_data,
  family = bernoulli(link = "logit"),
  backend = "cmdstanr",
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 123,
  prior = c(
    prior(normal(0, 0.25), class = "b"),                           # Strong priors for fixed effects
    prior(normal(-2, 0.5), class = "Intercept"),                    # Strong informative intercept
    prior(exponential(2), class = "sd", group = "policy_state"),  # Prior for random intercept
    prior(normal(0, 0.25), class = "sd", coef = "total_claim_sum", group = "policy_state") # Prior for random slope
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

#####RUN THIS AFTER THE MODEL
# Perform prior predictive checks
prior_model <- brm(
  formula = fraud_reported ~ total_claim_sum + policy_csl + policy_annual_premium + 
    umbrella_limit + 
    incident_severity_Minor.Damage + incident_severity_Total.Loss + 
    incident_severity_Trivial.Damage  + 
    (1 + total_claim_sum | policy_state),
  data = insurance_data,
  family = bernoulli(link = "logit"),
  backend = "cmdstanr",
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 123,
  prior = c(
    prior(normal(0, 0.25), class = "b"),                           # Strong priors for fixed effects
    prior(normal(-2, 0.5), class = "Intercept"),                   # Strong informative intercept
    prior(exponential(2), class = "sd", group = "policy_state"),   # Prior for random intercept
    prior(normal(0, 0.25), class = "sd", coef = "total_claim_sum", group = "policy_state") # Prior for random slope
  ),
  control = list(
    max_treedepth = 15,
    adapt_delta = 0.95
  ),
  sample_prior = "only"  # This ensures sampling is done only from the priors
)

# Generate predictions using only the prior
prior_predictions <- posterior_predict(prior_model)

# Plot prior predictive distributions
pp_check(prior_model)


