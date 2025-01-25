# Perform prior predictive checks
prior_model <- brm(
  formula = fraud_reported ~ total_claim_sum + policy_csl + policy_annual_premium + 
    umbrella_limit + 
    incident_severity_Minor.Damage + incident_severity_Total.Loss + 
    incident_severity_Trivial.Damage + 
    umbrella_limit:incident_severity_Minor.Damage + 
    umbrella_limit:incident_severity_Total.Loss + 
    umbrella_limit:incident_severity_Trivial.Damage + 
    (1 + total_claim_sum | policy_state),
  data = insurance_data,  # Use the dataset for structure; data is not conditioned
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
  ),
  sample_prior = "only"  # Sample only from the priors
)

# Generate prior predictive samples
prior_predictions <- posterior_predict(prior_model)

# Visualize prior predictive checks
pp_check(prior_model)

# Additional histogram of prior predictive means
prior_means <- rowMeans(prior_predictions)
hist(prior_means, breaks = 30, col = "lightblue", main = "Prior Predictive Distribution",
     xlab = "Predicted Mean Fraud Probability")
