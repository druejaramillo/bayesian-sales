import numpy as np
import pandas as pd
import math
import seaborn as sb
from mcmc_diagnostics import estimate_ess
import matplotlib.pyplot as plt
from scipy.stats import beta

# Set the seed
np.random.seed(0)

# Set default theme
sb.set_theme()

# Load the data
data = pd.read_csv('data.csv')
date_vars = ['wk_strt_dt', 'yr_nbr', 'qtr_nbr', 'prd', 'wk_in_yr_nbr']
spend_vars = ['mdsp_dm', 'mdsp_inst', 'mdsp_nsp', 'mdsp_auddig', 'mdsp_audtr', 'mdsp_vidtr', 'mdsp_viddig', 'mdsp_so', 'mdsp_on', 'mdsp_sem']
macro_vars = ['me_ics_all', 'me_gas_dpg']
store_vars = ['st_ct']
#holiday_vars = ["hldy_Black Friday", "hldy_Christmas Day", "hldy_Christmas Eve", "hldy_Columbus Day", "hldy_Cyber Monday", "hldy_Day after Christmas", "hldy_Easter", "hldy_Father's Day", "hldy_Green Monday", "hldy_July 4th", "hldy_Labor Day", "hldy_MLK", "hldy_Memorial Day", "hldy_Mother's Day", "hldy_NYE", "hldy_New Year's Day", "hldy_Pre Thanksgiving", "hldy_Presidents Day", "hldy_Prime Day", "hldy_Thanksgiving", "hldy_Valentine's Day", "hldy_Veterans Day"]
holiday_vars = ["hldy_Black Friday", "hldy_Christmas Day", "hldy_Cyber Monday", "hldy_Easter", "hldy_Father's Day", "hldy_July 4th", "hldy_Labor Day", "hldy_Memorial Day", "hldy_Mother's Day", "hldy_NYE", "hldy_Prime Day", "hldy_Thanksgiving", "hldy_Valentine's Day"]
sales_var = ['sales']
all_vars = date_vars + spend_vars + macro_vars + store_vars + holiday_vars + sales_var
data = data[all_vars]

# Extract independent variables
indep_vars = data[spend_vars + macro_vars + store_vars + holiday_vars]

regressors_df = pd.DataFrame()
# Linear terms
regressors_df[spend_vars] = indep_vars[spend_vars]
regressors_df[macro_vars] = indep_vars[macro_vars]
regressors_df[store_vars] = indep_vars[store_vars]
regressors_df[holiday_vars] = indep_vars[holiday_vars]
# Interaction terms
for spend_var in spend_vars:
	for macro_var in macro_vars:
		interaction = indep_vars[spend_var].multiply(indep_vars[macro_var])
		name = spend_var + '*' + macro_var
		regressors_df[name] = interaction
for spend_var in spend_vars:
	for store_var in store_vars:
		interaction = indep_vars[spend_var].multiply(indep_vars[store_var])
		name = spend_var + '*' + store_var
		regressors_df[name] = interaction
for spend_var in spend_vars:
	for holiday_var in holiday_vars:
		interaction = indep_vars[spend_var].multiply(indep_vars[holiday_var])
		name = spend_var + '*' + holiday_var
		regressors_df[name] = interaction
for macro_var in macro_vars:
	for store_var in store_vars:
		interaction = indep_vars[macro_var].multiply(indep_vars[store_var])
		name = macro_var + '*' + store_var
		regressors_df[name] = interaction
for macro_var in macro_vars:
	for holiday_var in holiday_vars:
		interaction = indep_vars[macro_var].multiply(indep_vars[holiday_var])
		name = macro_var + '*' + holiday_var
		regressors_df[name] = interaction
for store_var in store_vars:
	for holiday_var in holiday_vars:
		interaction = indep_vars[store_var].multiply(indep_vars[holiday_var])
		name = store_var + '*' + holiday_var
		regressors_df[name] = interaction
# Intercept term
regressors_df.insert(0, column='intercept', value=1)

# Turn regressors and dependent variable into numpy arrays
X = regressors_df.to_numpy()
y = data['sales'].to_numpy()

# Plot sales and spending over time
data.rename(columns={'mdsp_dm': 'direct mail', 'mdsp_inst': 'insert', 'mdsp_vidtr': 'TV', 'mdsp_so': 'social', 'mdsp_on': 'online'}, inplace=True)
vars = ['direct mail', 'insert', 'TV', 'social', 'online']
weeks = data['wk_strt_dt'].to_numpy()
sb.set_style('ticks')
fig, axes = plt.subplots(1, 2, figsize=(10,5))
#sb.lineplot(ax=axes[0], data=data[['wk_strt_dt', 'sales']], x='wk_strt_dt', y='sales')
axes[0].plot(weeks, data['sales'])
axes[0].set_xticks(range(0, len(weeks), 16), labels=weeks[::16], rotation=45)
axes[0].set_xlabel('week')
axes[0].set_ylabel('sales')
axes[0].set_title('Weekly sales')
#sb.lineplot(ax=axes[1], data=pd.melt(data[['wk_strt_dt']+vars], id_vars='wk_strt_dt'), x='wk_strt_dt', y='value', hue='variable')
l1, = axes[1].plot(weeks, data['direct mail'], 'b-', label='direct mail')
l2, = axes[1].plot(weeks, data['insert'], color='orange', linestyle='-', label='insert')
l3, = axes[1].plot(weeks, data['TV'], 'g-', label='TV')
l4, = axes[1].plot(weeks, data['social'], 'r-', label='social')
l5, = axes[1].plot(weeks, data['online'], color='purple', linestyle='-', label='online')
axes[1].set_xticks(range(0, len(weeks), 16), labels=weeks[::16], rotation=45)
axes[1].set_xlabel('week')
axes[1].set_ylabel('spending')
axes[1].set_title('Weekly spending')
axes[1].legend(handles=[l1, l2, l3, l4, l5])
plt.tight_layout()
plt.savefig('sales_and_spending.png', dpi=400)
plt.close()
plt.cla()
plt.clf()

# Total number of regressors
K = np.size(X, 1)
# Expected number of regressors in fitted model
p = math.floor((K/2) / 10) * 2
# Number of samples/timesteps
n = np.size(y, 0)

# Prior parameters
nu_0 = 3.0
sigma_0_sq = 17000.0**2
#Sigma_0 = 2.5 * np.ones((K, K)) + 22.5 * np.diag(np.ones(K))
#Sigma_0 = 100 * np.diag(np.ones(K))
#Sigma_0[0, 0] = 100000**2
Sigma_0_scaled = 4 * np.diag(np.ones(K))

# Function define the correlation matrix
def getC(rho):

	C = np.diag(np.ones(n))
	for i in range(n):
		for j in range(i+1, n):
			C[i, j] = rho**(j-i)
			C[j, i] = C[i, j]

	return C

# Create simulated data
X_sim_df = regressors_df.copy()
rho = 0.7
C = getC(rho)
# Replace spend data with gaussian noise
for column in spend_vars:
	center = np.random.choice([25000, 50000, 75000, 100000, 125000])
	noise = np.random.multivariate_normal(mean=center*np.ones(n), cov=((center/7.0)**2)*C)
	X_sim_df[column] = noise
# Cut off at 0
X_sim_df[spend_vars] = X_sim_df[spend_vars].clip(lower=0)
# Simulate data
X_sim = X_sim_df.to_numpy()
rho = 0.6
C = getC(rho)
beta_sim = np.concatenate((400000*np.random.rand(1)+600000, 6*np.random.rand(10), 100*np.random.rand(2), 1000*np.random.rand(1), 10000*np.random.rand(13), 0.02*np.random.rand(20)-0.01, 0.002*np.random.rand(10)-0.001, 0.2*np.random.rand(130)-0.05, np.random.rand(2), 10*np.random.rand(26)-5, 100*np.random.rand(13)-5), axis=None).reshape(-1, 1)
y_sim = np.random.multivariate_normal(mean=np.matmul(X_sim, beta_sim).reshape(-1), cov=sigma_0_sq*C)
y_sim = np.array(y_sim).reshape(-1)

# Plot simulated sales and spending over time
X_sim_df.rename(columns={'mdsp_dm': 'direct mail', 'mdsp_inst': 'insert', 'mdsp_vidtr': 'TV', 'mdsp_so': 'social', 'mdsp_on': 'online'}, inplace=True)
vars = ['direct mail', 'insert', 'TV', 'social', 'online']
weeks = data['wk_strt_dt'].to_numpy()
X_sim_df['wk_strt_dt'] = weeks
sb.set_style('ticks')
fig, axes = plt.subplots(1, 2, figsize=(10,5))
#sb.lineplot(ax=axes[0], data={'weeks': weeks, 'sales': y_sim}, x='weeks', y='sales')
axes[0].plot(weeks, y_sim, 'b-')
axes[0].set_xticks(range(0, len(weeks), 16), labels=weeks[::16], rotation=45)
axes[0].set_xlabel('week')
axes[0].set_ylabel('sales')
axes[0].set_title('Weekly sales')
#sb.lineplot(ax=axes[1], data=pd.melt(X_sim_df[['wk_strt_dt']+vars], id_vars='wk_strt_dt'), x='wk_strt_dt', y='value', hue='variable')
l1, = axes[1].plot(weeks, X_sim_df['direct mail'], 'b-', label='direct mail')
l2, = axes[1].plot(weeks, X_sim_df['insert'], color='orange', linestyle='-', label='insert')
l3, = axes[1].plot(weeks, X_sim_df['TV'], 'g-', label='TV')
l4, = axes[1].plot(weeks, X_sim_df['social'], 'r-', label='social')
l5, = axes[1].plot(weeks, X_sim_df['online'], color='purple', linestyle='-', label='online')
axes[1].set_xticks(range(0, len(weeks), 16), labels=weeks[::16], rotation=45)
axes[1].set_xlabel('week')
axes[1].set_ylabel('spending')
axes[1].set_title('Weekly spending')
axes[1].legend(handles=[l1, l2, l3, l4, l5])
plt.tight_layout()
plt.savefig('sales_and_spending_sim.png', dpi=400)
plt.close()
plt.cla()
plt.clf()

# Scale the simulated data
X_sim_scaled = (X_sim - X_sim.mean(axis=0)) / X_sim.std(axis=0)
X_sim_scaled[:, 0] = 1
y_sim_scaled = (y_sim - y_sim.mean()) / y_sim.std()
sigma_0_sq_scaled = sigma_0_sq / (y_sim.std()**2)

# MCMC function
delta = 0.32
rho_prior = lambda rho: beta.pdf(rho, 1, 1)
def MCMC(n_iter, data, initial_vals):

	from datetime import datetime
	time0 = datetime.now()

	# Extract the data
	X = data[0]
	y = data[1]

	# Extract the initial values
	rho = initial_vals[0]
	sigma_sq = initial_vals[1]

	# Calculate correlation matrix
	C = getC(rho)

	# Run MCMC
	markov_chain = []
	num_accept = 0.0
	for i in range(n_iter):

		# Progress bar
		if (i+1) % 100 == 0:
			percent = (i+1) / float(n_iter)
			num_equals = math.floor(percent * 50)
			num_dashes = 50 - num_equals - 1
			progress_bar = '[' + num_equals * '=' + '>' + num_dashes * '-' + ']'
			print(progress_bar + ' ' + str(round(100*percent, 2)) + '%', end='\r')

		# Sample beta
		Sigma_n = np.linalg.inv(np.matmul(np.matmul(X.T, np.linalg.inv(C)), X)/sigma_sq + np.linalg.inv(Sigma_0_scaled))
		beta_n = np.matmul(Sigma_n, np.matmul(np.matmul(X.T, np.linalg.inv(C)), y)/sigma_sq).reshape(-1)
		try:
			beta = np.random.multivariate_normal(mean=beta_n, cov=Sigma_n)
		except:
			pass

		# Sample sigma^2
		SSR_rho = np.matmul(np.matmul((y - np.matmul(X, beta)).T, np.linalg.inv(C)), (y - np.matmul(X, beta)))
		sigma_sq = 1 / np.random.gamma(shape=(nu_0+n)/2, scale=2/(nu_0*(sigma_0_sq_scaled)+SSR_rho))

		# Sample rho
		rho_star = np.random.uniform(low=rho-delta, high=rho+delta)
		if rho_star < 0:
			rho_star = abs(rho_star)
		if rho_star > 1:
			rho_star = 2 - rho_star
		C_star = getC(rho_star)
		likelihood_ratio = math.sqrt(np.linalg.det(np.matmul(C, np.linalg.inv(C_star)))) * np.exp(1/(2*sigma_sq) * np.matmul(np.matmul((y - np.matmul(X, beta)).T, (np.linalg.inv(C) - np.linalg.inv(C_star))), (y - np.matmul(X, beta))))
		r = likelihood_ratio * rho_prior(rho_star) / rho_prior(rho)
		u = np.random.uniform(low=0, high=1)
		if u < r:
			rho = rho_star
			num_accept += 1
		
		# Add values to markov chain
		markov_chain.append((beta, sigma_sq, rho))
	
	print(f'\nAcceptance rate: {num_accept/n_iter}')

	time1 = datetime.now()
	print(f'MCMC time: {(time1-time0).total_seconds()}')

	return markov_chain

# Run MCMC for simulated data
n_iter = 5000
initial_vals = (0.5, sigma_0_sq/(y_sim.std()**2))
markov_chain = MCMC(n_iter, (X_sim_scaled, y_sim_scaled), initial_vals)
print()

# Autocorrelation function
def acf(chain, lag):
	S = len(chain)
	avg = sum(chain) / S
	numerator = 1/(S-lag) * sum([(chain[s]-avg)*(chain[s+lag]-avg) for s in range(S-lag)])
	denominator = 1/(S-1) * sum([(chain[s]-avg)**2 for s in range(S)])
	return numerator / denominator

# Predicted values for rho
rho_chain = np.array([markov_chain[s][2] for s in range(n_iter)])

# Predicted values for sigma^2
sigma_sq_chain = np.array([(y_sim.std()**2)*markov_chain[s][1] for s in range(n_iter)])

# True vs. predicted average beta
beta_true = beta_sim.reshape(-1)
beta_chain = np.array([markov_chain[s][0] for s in range(n_iter)])
for s in range(n_iter):
	beta_chain[s][0] = (beta_chain[s][0] - np.matmul((X_sim.mean(axis=0)[1:]/X_sim.std(axis=0)[1:]).reshape(1, K-1), beta_chain[s][1:])) * y.std() + y.mean()
	beta_chain[s][1:] = np.divide(beta_chain[s][1:]*y.std(), X_sim[:, 1:].std(axis=0).reshape(-1))
beta_pred = np.mean(beta_chain, axis=0)

# Plot true vs. simulated values
sb.set_style('ticks')
fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes[0].plot(np.arange(n_iter), rho_chain, 'b-')
axes[0].axhline(y=rho, color='r', linestyle='-')
axes[0].set_ylim(0, 1)
axes[0].set_xlabel('iteration #')
axes[0].set_ylabel('rho')
axes[0].set_title('Trace plot of correlation parameter')
axes[1].plot(np.arange(n_iter), sigma_sq_chain, 'b-')
axes[1].axhline(y=sigma_0_sq, color='r', linestyle='-')
axes[1].set_xlabel('iteration #')
axes[1].set_ylabel('sigma^2')
axes[1].set_title('Trace plot of observation error')
plt.tight_layout()
plt.savefig('true_vs_sim.png', dpi=400)
plt.close()
plt.cla()
plt.clf()

# Scale the real data
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
X_scaled[:, 0] = 1
y_scaled = (y - y.mean()) / y.std()
sigma_0_sq = 1000000**2
sigma_0_sq_scaled = sigma_0_sq / (y.std()**2)

# Run MCMC for real data
n_iter = 100000
initial_vals = (0.5, sigma_0_sq/(y.std()**2))
markov_chain = MCMC(n_iter, (X_scaled, y_scaled), initial_vals)
print()

# Discard burn-in of 5000 iterations
markov_chain = markov_chain[5000:]
# Keep every 5th iteration
markov_chain = markov_chain[::10]
n_iter = len(markov_chain)

# Predicted values for rho
rho_chain = np.array([markov_chain[s][2] for s in range(n_iter)])

# Predicted values for sigma^2
sigma_sq_chain = np.array([(y.std()**2)*markov_chain[s][1] for s in range(n_iter)])

# True vs. predicted average beta
beta_chain = np.array([markov_chain[s][0] for s in range(n_iter)])
for s in range(n_iter):
	beta_chain[s][0] = (beta_chain[s][0] - np.matmul((X_sim.mean(axis=0)[1:]/X_sim.std(axis=0)[1:]).reshape(1, K-1), beta_chain[s][1:])) * y.std() + y.mean()
	beta_chain[s][1:] = np.divide(beta_chain[s][1:]*y.std(), X_sim[:, 1:].std(axis=0).reshape(-1))
beta_pred = np.mean(beta_chain, axis=0)

# Trace plots for rho, sigma^2, and beta_1
sb.set_style('ticks')
fig, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].plot(np.arange(n_iter), rho_chain, 'b-')
axes[0].set_ylim(0, 1)
axes[0].set_xlabel('iteration #')
axes[0].set_ylabel('rho')
axes[0].set_title('Trace plot of correlation parameter')
axes[1].plot(np.arange(n_iter), sigma_sq_chain, 'b-')
axes[1].set_xlabel('iteration #')
axes[1].set_ylabel('sigma^2')
axes[1].set_title('Trace plot of observation error')
axes[2].plot(np.arange(n_iter), np.array([beta_chain[s][0] for s in range(n_iter)]), 'b-')
axes[2].set_xlabel('iteration #')
axes[2].set_ylabel('beta_1')
axes[2].set_title('Trace plot of first regression coefficient (intercept)')
plt.tight_layout()
plt.savefig('trace_plots.png', dpi=400)
plt.close()
plt.cla()
plt.clf()

# Autocorrelation plots for rho, sigma^2, and beta_1
acf1 = [acf(rho_chain, lag) for lag in range(1, 101)]
acf2 = [acf(sigma_sq_chain, lag) for lag in range(1, 101)]
acf3 = [acf(np.array([beta_chain[s][0] for s in range(n_iter)]), lag) for lag in range(1, 101)]
sb.set_style('ticks')
fig, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].plot(np.arange(1, 101), acf1, 'b-')
axes[0].set_ylim(-0.1, 0.8)
axes[0].set_xlabel('lag')
axes[0].set_ylabel('acf')
axes[0].set_title('Autocorrelation plot of correlation parameter')
axes[1].plot(np.arange(1, 101), acf2, 'b-')
axes[0].set_ylim(-0.1, 0.8)
axes[1].set_xlabel('lag')
axes[1].set_ylabel('acf')
axes[1].set_title('Autocorrelation plot of observation error')
axes[2].plot(np.arange(1, 101), acf3, 'b-')
axes[0].set_ylim(-0.1, 0.8)
axes[2].set_xlabel('lag')
axes[2].set_ylabel('acf')
axes[2].set_title('Autocorrelation plot of first regression coefficient')
plt.tight_layout()
plt.savefig('acf_plots.png', dpi=400)
plt.close()
plt.cla()
plt.clf()

# Effective sample sizes
rho_ess = estimate_ess(rho_chain)
sigma_sq_ess = estimate_ess(sigma_sq_chain)
beta_ess = estimate_ess(beta_chain)
print(f'ESS for rho: {rho_ess}')
print(f'ESS for sigma^2: {sigma_sq_ess}')
print(f'Min. ESS for beta: {min(beta_ess)}')