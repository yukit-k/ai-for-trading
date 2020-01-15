# AI for Trading

Udacity nano-degree to learn practical AI application in trading algo.
Designed by WorldQuant.

### Syllabus:
1. Basic Quantitative Trading - Trading with Momentum
2. Advanced Quantitative Trading - Breakout Strategy
3. Stocks, Indices, and ETFs - Smart Beta and Portfolio Optimization
4. Factor Investing and Alpha Research - Alpha Research and Factor Modeling
5. Sentiment Analysis with Natural Language Processing
6. Advanced Natural Language Processing with Deep Leaning
7. Combining Multiple Signals for Enhanced Alpha
8. Simulating Trades with Historical Data - Backtesting

### Main Libraries:
 - Numpy, Pandas, Matplotlib
 - Scikit-learn
 - Pytorch
 - Quantopian/zipline
 - Quantmedia
 
### My Course:
 - Started: September 2019
 - Target End: February 2020
 - Actual End: January 2020

### Projects:
#### 1. Basic Quantitative Trading - Trading with Momentum
#### 2. Advanced Quantitative Trading - Breakout Strategy
#### 3. Stocks, Indices, and ETFs - Smart Beta and Portfolio Optimization
#### 4. Factor Investing and Alpha Research - Alpha Research and Factor Modeling
#### 5. Sentiment Analysis with Natural Language Processing
#### 6. Advanced Natural Language Processing with Deep Leaning
#### 7. Combining Multiple Signals for Enhanced Alpha
#### 8. Simulating Trades with Historical Data - Backtesting
1. Load Price, Covariance and Factor Exposure from Barra - data.update(pickle.load())
2. Shift daily returns by 2 days
3. Winsorize
    - np.where(x <= a,a, np.where(x >= b, b, x)) and Density plot
4. Factor Exposures and Factor Returns
    - model = ols (Ordinary Least Squares)
    - universe = Market Cap > 1e9, Winsorize
    - variable: dependent = Daily Return, independent = Factor Exposures
    - estimation: Factor Returns
5. Choose 4 Alpha Factors
    - 1 Day Reversal, Earnings Yield, Value, Sentiment
6. Merge Previous Portfolio Holdings and Add h.opt.previous with 0
7. Convert all NaN to 0, and median for 0 Specific Risk
8. Build Universe - (df['IssuerMarketCap'] >= 1e9) | (abs(df['h.opt.previous']) > 0.0)
9.  Set Risk Factors (B)
    - All Factors - Alpha Factors
    - patsy.dmatrices to one-hot encode categories
10. Calculate Specific Variance
    - (Specific Risk * 0.01)**2
11. Build Factor Covariance Matrix
    - Take off diagonal 
12. Estimate Transaction Cost
    - Lambda
13. Combine the four Alpha Factors
    - sum(B_Alpha(Design Matrix)) * 1e-4
14. Define Objective Function
    - $$
f(\mathbf{h}) = \frac{1}{2}\kappa \mathbf{h}_t^T\mathbf{Q}^T\mathbf{Q}\mathbf{h}_t + \frac{1}{2} \kappa \mathbf{h}_t^T \mathbf{S} \mathbf{h}_t - \mathbf{\alpha}^T \mathbf{h}_t + (\mathbf{h}_{t} - \mathbf{h}_{t-1})^T \mathbf{\Lambda} (\mathbf{h}_{t} - \mathbf{h}_{t-1})
$$
15. Define Gradient of Objective Function
    - $$
f'(\mathbf{h}) = \frac{1}{2}\kappa (2\mathbf{Q}^T\mathbf{Qh}) + \frac{1}{2}\kappa (2\mathbf{Sh}) - \mathbf{\alpha} + 2(\mathbf{h}_{t} - \mathbf{h}_{t-1}) \mathbf{\Lambda}
$$
16. Optimize Portfolio
    - h = scipy.optimize.fmin_l_bfgs_b(func, initial_guess, func_gradient)
17. Calculate Risk Exposure
    - B.T * h
18. Calculate Alpha Exposure
    - B_Alpha.T * h
19. Calculate Transaction Cost
    - $$
tcost = \sum_i^{N} \lambda_{i} (h_{i,t} - h_{i,t-1})^2
$$
20. Build Tradelist
    - h - h_previous
21. Save optimal holdings as previous optimal holdings
    - h_previous = h
22. Run the Backtest
    - Loop #6 to #21 for all the dates
23. PnL Attrribution
    - $$
{PnL}_{alpha}= f \times b_{alpha}
$$
    - $$
{PnL}_{risk} = f \times b_{risk}
$$

24. Build Portfolio Characteristics
    - calculate the sum of long positions, short positions, net positions, gross market value, and amount of dollars traded.