# Machine Learning Assignment

# install.packages("quantmod", type = "binary")

library(quantmod)

# We get the stock data of IBM

getSymbols("IBM", src = "yahoo", from = "1995-01-01", to = "2026-01-01")

#head(IBM)

chartSeries(IBM, theme = chartTheme("white"))

# We take the adjusted close price in the sixth column

close_prices <- IBM[, 6]

# Compute log returns

log_returns <- diff(log(close_prices))

#Omit first NaN value since on first day we cannot compute return

log_returns <- na.omit(log_returns)
#head(log_returns)

#Plot log returns

plot(log_returns, 
     main = "IBM Daily Log Returns", 
     col = "darkblue", 
     major.ticks = "years", 
     grid.ticks.on = "years")

mean_return = mean(log_returns)

std = sd(log_returns)

threshold = 3*std

# Monte Carlo Simulation

# We count # of extreme events (abs(r) > 3*std) per year
# Poisson is appropriate to model n events (n in N)

yearly_extremes <- apply.yearly(log_returns, function(x) sum(abs(x) > threshold))
X <- as.numeric(yearly_extremes)
n <- length(X)
sum_X <- sum(X)

cat("Annual extreme events", X, "\n")
cat("Sample mean", mean(X), "\n")


# We select alpha and beta. Assume about 3 events per year allows us
# to choose alpha and beta
alpha_prior <- 3   
beta_prior <- 1 

# Importance Sampling
set.seed(1900)
M <- 10000

# Self-Normalized Estimator
# 1) Draw from proposal (we are using the prior as proposal)
theta_is <- rgamma(M, shape = alpha_prior, rate = beta_prior)

# 2) Unnormalized weights (Likelihood)
# Since w = q/s = (Likelihood * Prior) / Prior = Likelihood
w <- sapply(theta_is, function(th) prod(dpois(X, lambda = th)))
stopifnot(sum(w) > 0)   # check: no underflow

# 3) Normalised weights
w_tilde <- w / sum(w)

# 4) Estimate E[theta | X]
is_mean <- sum(w_tilde * theta_is)

analytical_mean <- (alpha_prior + sum_X) / (beta_prior + n)

round(rbind(IS_Estimate = is_mean, Analytical = analytical_mean), 4)

# Effective sample size of the importance sampler
ess <- 1 / sum(w_tilde^2)
cat("ESS (importance sampling):", ess, "\n")


# Gibbs Sampling 
set.seed(1900)

a_hyper <- 2
b_hyper <- 2

M_gibbs <- 10000
B <- 2000

theta_chain <- numeric(M_gibbs)
beta_chain <- numeric(M_gibbs)

theta_chain[1] <- 10
beta_chain[1]  <- 5

# Loop MCMC
for (t in 2:M_gibbs) {
  # 1) theta | beta, X ~ Gamma(alpha + sum(X), beta + n)
  shape_theta <- alpha_prior + sum_X
  rate_theta  <- beta_chain[t - 1] + n
  theta_chain[t] <- rgamma(1, shape = shape_theta, rate = rate_theta)
  
  # 2) beta | theta, X ~ Gamma(a + alpha, theta + b)
  shape_beta <- a_hyper + alpha_prior
  rate_beta  <- theta_chain[t] + b_hyper
  beta_chain[t] <- rgamma(1, shape = shape_beta, rate = rate_beta)
}

# Remove burn-in
theta_post <- theta_chain[(B + 1):M_gibbs]
beta_post  <- beta_chain[(B + 1):M_gibbs]

# Traceplots and estimates
op <- par(mfrow = c(2, 1))
plot(theta_chain, type = "l", main = "Traceplot - Theta", xlab = "Iteration", ylab = "Theta")
abline(v = B, col = "red", lty = 2)
plot(beta_chain, type = "l", main = "Traceplot - Beta", xlab = "Iteration", ylab = "Beta")
abline(v = B, col = "red", lty = 2)
par(op)

cat("Posterior Mean of Theta (Gibbs):", mean(theta_post), "\n")
cat("Posterior Mean of Beta (Gibbs):", mean(beta_post), "\n")

# Autocorrelation

acf(theta_post, main = "ACF of theta")

acf(beta_post, main = "ACF of beta")
