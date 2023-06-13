# Decomposing Sales via Bayesian Regression

## Overview

Marketing mix modeling (MMM) focuses on determining the relationship between sales and marketing efforts in order to optimize marketing budgets and maximize return on ad spend (ROAS). We consider the approach of a simple Bayesian normal regression model, but with an added first-order autoregressive structure determined by an unknown parameter. After fitting the model to a simulated marketing mix dataset using MCMC methods, we discuss the applications of the model and further improvements that may be made.

## Modeling Weekly Sales

We assume that each week's sales is some linear function of the predictors and has Gaussian noise. Moreover, we assume that the noise is independent and identically distributed across weeks. By itself, this could be represented using a typical multivariate normal distribution for the sampling model. However, our intuition tells us that the sales of two different weeks are not necessarily uncorrelated in every case. In fact, we expect sales to have a bit of momentum, so that the closer two weeks are to each other, the more correlated their sales will be. Hence, we use a sampling model that is a multivariate normal distribution with covariance matrix $\sigma^2 C_\rho$, where $\sigma^2$ is the common variance of the data, and $C_\rho$ is a matrix with first-order autoregressive structure determined by $\rho$. It follows that the correlation between sales in weeks $i$ and $j$ is $\rho^{|j-i|}$, which diminishes to 0 as $|j-i|$ grows (as the weeks get further apart).

Since we are using a multivarite normal sampling model, we chose standard conjugate priors for our parameters: 1) multivariate normal distribution for the regression coefficients and 2) inverse-gamma distribution for the variance. This gives us full conditional posterior distributions that are 1) multivariate normal for the regression coefficients and 2) inverse-gamma for the variance. On the other hand, there is no conjugate prior for $\rho$, so we just us a beta distribution as its prior. Subsequently, we can use a combination of Gibbs sampling and a Metropolis step to approximate the joint posterior distribution of our parameters.

## Further Analysis

There are many additional analyses that can be performed to extract insights:
1. We can generate a posterior predictive distribution given a new set of data.
2. We can estimate the return on ad spend (ROAS) of each marketing channel to evaluate current marketing efforts.
3. We can estimate the marginal return on ad spend (mROAS) of each marketing channel to optimize the marketing budget.

## Factors Considered

The independent variables used in the model are:
1. Spending for 13 different marketing channels, including direct mail, TV, social media, etc.
2. The Consumer Price Index (CPI)
3. The average price of gas
4. The number of stores our fictional company has
5. Indicator variables for major holidays, including Father's/Mother's Day, Fourth of July, Christmas, etc.
6. Interaction terms between all of the above variables
7. An intercept term

## Getting Started

This project requires the following non-standard Python libraries:
1. `numpy`
2. `pandas`
3. `seaborn`
4. `mcmc-diagnostics`
5. `matplotlib`
6. `scipy`