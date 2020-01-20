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
 * Numpy, Pandas, Matplotlib
 * Scikit-learn
 * Pytorch
 * Quantopian/zipline
 * Quantmedia
 
### My Course:
 * Started: September 2019
 * Target End: February 2020
 * Actual End: January 2020

### Project Details:
#### 1. Basic Quantitative Trading - Trading with Momentum
0. import pandas, numpy, helper
1. Load Quatemedia EOD Price Data
2. Resample to Month-end ```close_price.resample('M').last()```
3. Compute Log Return
4. Shift Returns ```returns.shift(n)```
5. Generate Trading Signal
   * Strategy tried:
        > For each month-end observation period, rank the stocks by previous returns, from the highest to the lowest. Select the top performing stocks for the long portfolio, and the bottom performing stocks for the short portfolio.
   * ```
      for i, row in prev_price:
        top_stock.loc[i] = row.nlargest(top_n)
     ```
6. Projected Return ```portfolio_returns = (lookahead_returns * (df_long - df_short))/n_stocks
7. Statistical Test
   * Annualized Rate of Return ```(np.exp(portfolio_returns.T.sum().dropna().mean()*12) - 1) * 100```
   * T-Test
     * Null hypothesis (H0): Actual mean return from the signal is zero.
     * When p value < 0.05, the null hypothesis is rejected
     * One-sample, one-sided t-test ```(t_value, p_value) = scipy.stats.ttest_1samp(portfolio_return, hypothesis)```
#### 2. Advanced Quantitative Trading - Breakout Strategy
0. import pandas, numpy, helper
1. Load Quatemedia EOD Price Data
2. The Alpha Research Process
    * What feature of markets or investor behaviour would lead to a persistent anomaly that my signal will try to use?
    * Example Hypothesis:
      * Stocks oscillate in a range without news or significant interest
      * Traders seek to sell at the top of the range and buy at the bottom
      * When stocks break out of the range,
        * the liquidity traders seek to cover the losses, which magnify the move out of the range
        * the move out of the range attract other investor interst due to herd behaviour which favor continuation of the trend 
    * Process:
      1. Observ & Research
      2. Form Hypothesis
      3. Validate Hypothesis, back to #1
      4. Code Expression
      5. Evaluate in-sample
      6. Evaluate out-of-sample
3. Compute Highs and Lows in a Window
    * e.g., Rolling max/min for the past 50 days
4. Compute Long and Short Signals
    * long = close > high, short = close < low, position = long - short
5. Filter Signals (5, 10, 20 day signal window)
    * Check if there was a signal in the past window_size of days
      `has_past_signal = bool(sum(clean_signals[signal_i:signal_i+window_size]))`
    * Use the current signal if there's no past signal, else 0/False
      `clean_signal.append(not has_past_signal and current_signal)`
    * Apply the above to short (signal[signal == -1].fillna(0.astype(int))) and long, add them up
6. Lookahead Close Price
    * How many days to short or long `close_price.shift(lookahead_days*-1)`
7. Lookahead Price Return
    * Log return between lookahead_price and close_price
8. Compute the Signal Return
    * signal * lookahead_returns
9. Test for Significance
    * Plot a histogram of the signal returns
10. Check Outliers in the histogram
11. Kolmogorov-Smirnov Test (KS-Test)
    * Check which stock is causing the outlying returns
    * Run KS-Test on a normal distribution against each stock's signal returns
    * `ks_value, p_value = scipy.stats.kstest(rvs=group['signal_return'].values, cdf='norm', args=(mean_all, std_all))`
12. Find outliers
    * Symbols that pass the null hypothesis with a p-value less than 0.05
    * Symbols that with a KS value above ks_threshod(0.8)
    * Remove them by `good_tickers = list(set(close.column) - outlier_tickers)`
#### 3. Stocks, Indices, and ETFs - Smart Beta and Portfolio Optimization
1. Load large dollar volume stocks from quotemedia
##### Smart Beta by alternative weighting - dividend yield to choose the portfolio weight
2. Calculate Index Weights (dollar volume weights)
3. Calculate Portfolio Weights based on Dividend
4. Calculate Returns, Weighted Returns, Cumulative Returns
5. Tracking Error ` np.sqrt(252) * np.std(benchmark_returns_daily - etf_returns_daily, ddof=1)`
##### Portfolio Optimization - minimize the portfolio variance and closely track the index
$Minimize \left [ \sigma^2_p + \lambda \sqrt{\sum_{1}^{m}(weight_i - indexWeight_i)^2} \right  ]$ where $m$ is the number of stocks in the portfolio, and $\lambda$ is a scaling factor that you can choose.
6. Calculate the covariance of the returns `np.cov(returns.fillna(0).values, rowvar=False)`
7. Calculate optimal weights
   * Portfolio Variance: $\sigma^2_p = \mathbf{x^T} \mathbf{P} \mathbf{x}$
      * `cov_quad = cvx.quad_form(x, P)`
   * Distance from index weights: $\left \| \mathbf{x} - \mathbf{index} \right \|_2$ = $\sqrt{\sum_{1}^{n}(weight_i - indexWeight_i)^2}$
     * `index_diff = cvx.norm(x, p=2, axix=None)`
   * Objective function = $\mathbf{x^T} \mathbf{P} \mathbf{x} + \lambda \left \| \mathbf{x} - \mathbf{index} \right \|_2$
     * `cvx.Minimize(cov_quad + scale * index_diff)`
   * Constraints
     * ```
       x = cvx.Variable()
       constraints = [x >= 0, sum(x) == 1]
       ```
   * Optimization
     * ```
       problem = cvx.Problem(objective, constraints
       problem.solve()
       ```
8. Rebalance Portfolio over time
9. Portfolio Turnover
   * $ AnnualizedTurnover =\frac{SumTotalTurnover}{NumberOfRebalanceEvents} * NumberofRebalanceEventsPerYear $
   * $ SumTotalTurnover =\sum_{t,n}{\left | x_{t,n} - x_{t+1,n} \right |} $ Where $ x_{t,n} $ are the weights at time $ t $ for equity $ n $.
   * $ SumTotalTurnover $ is just a different way of writing $ \sum \left | x_{t_1,n} - x_{t_2,n} \right | $
 Minimum volatility ETF
#### 4. Factor Investing and Alpha Research - Alpha Research and Factor Modeling
0. import cvxpy, numpy, pandas, time, matplotlib.pyplot
1. Load equitiies EOD price (zipline.data.bundles)
    * `bundles.register(bundle_name, ingest_func)`
    * `bundles.load(bundle_name)`
2. Build Pipeline Engine
    * `universe = AverageDollarVolume(window_length=120).top(500)` <- 490 Tickers
    * `engine = SimplePipelineEngine(get_loader, calendar, asset_finder)`
3. Get Returns
    * `data_portal = DataPotal()`
    * `get_pricing = data_portal.get_history_window()`
    * `returns = get_pricing().pct_change()[1:].fillna(0)` <- e.g. 5 year: 1256x490
##### Statistical Risk Model
4. Fit PCA
    * `pca = sklearn.decomposition.PCA(n_components, svd_solver='full')`
    * `pca.fit()`
    * `pca.components_` <- 20x490
5. Factor Betas
    * `pd.DataFrame(pca.components_.T, index=returns.columns.values, columns=np.arange(20))` <- 20x490
6. Factor Returns
    * `pd.DataFrame(pca.transform(returns), index=returns.index , columns=np.arange(20))` <- 490x20
7. Factor Coveriance Matrix
    * `np.diag(np.var(factor_returns, axix=0, ddof=1)*252)` <- 20x20
8. Idiosyncratic Variance Matrix
    * ```
      _common_returns = pd.DataFrame(np.dot(factor_returns, factor_betas.T), returns.index, returns.columns)
      _residuals = (returns - _common_returns)
      pd.DataFrame(np.diag(np.var(_residuals)*252), returns.columns, returns.columns) <- 490x490
      ```
9. Idiosyncratic Variance Vector
    * `# np.dot(idiosyncratic_variance_matrix, np.ones(len(idiosyncratic_variance_matrix)))`
    * `pd.DaraFrame(np.diag(idiosyncratic_variance_matrix), returns.columns)`
10. Predict Portfolio Risk using the Risk Model
    * $ \sqrt{X^{T}(BFB^{T} + S)X} $ where:
      * $ X $ is the portfolio weights
      * $ B $ is the factor betas
      * $ F $ is the factor covariance matrix
      * $ S $ is the idiosyncratic variance matrix
    * `np.sqrt(weight_df.T.dot(factor_betas.dot(factor_cov_matrix).dot(factor_betas.T) + idiosyncratic_var_matrix).dot(weight_df))`
##### Create Alpha Factors
11. Momentum 1 Year Factor
12. Mean Reversion 5 Day Sector Neutral Factor
13. Mean Reversion 5 Day Sector Neutral Smoothed Factor
14. Overnight Sentiment Factor
15. Overnight Sentiment Smoothed Factor
16. Combine the Factors to a single Pipeline
    * ```
      pipeline = Pipeline(screen=universe)
      pipeline.add(momentum_1yr(252, universe, sector), 'Momentum_1YR')
      :
      all_factors = engine.run_pipeline(pipeline, start, end)
      ```
##### Evaluate Alpha Factors
17. Get Pricing Data
18. Format Alpha Factors and Pricing for Alphalens
19. Quantile Analysis
20. Turnover Analysis
21. Sharpe Ratio of the Alphas
22. The Combined Alpha Vector
##### Optimal Portfolio Constrained by Risk Model
23. Objective and Constraints
24. Optimize with a Regularization Parameter
25. Optimize with a Strict Factor Constrains and Target Weighting
#### 5. Sentiment Analysis with NLP - NLP on Financial Statement
0. import nltk, numpy, pandas, pickle, pprint, tqdm.tqdm, bs4.BeautifulSoup, re
    * ```nltk.download('stopwords'), nltk.download('wordnet')```
1. Get 10-k documents
    * Limit number of request per second by @limits
    * ```feed = BeautifulSoup(request.get.text).feed```
    * ```entries = [entry.content.find('filing-href').getText(), ... for entry in feed.find_all('entry')]```
    * Download 10-k documents
    * Extract Documents
      * ```doc_start_pattern = re.compile(r'<DOCUMENT>')```
      * ```doc_start_position_list = [x.end() for x in doc_start_pattern.finditer(text)]```
    * Get Document Types
      * ```doc_type_pattern = re.compile(r'<TYPE>[^\n]+')```
      * ```doc_type = doc_type_pattern.findall(doc)[0][len("<TYPE>"):].lower()```
2. Process the Data
    * Clean up
      * ```text.lower()```
      * ```BeautifulSoup(text, 'html.parser').get_text()```
    * Lemmatize
      * ```nltk.stem.WordNetLemmatizer, nltk.corpus.wordnet```
    * Remove Stopwords
      * ```nltk.corpus.stopwords```
3. Analysis on 10ks
    * Loughran and McDonald sentiment word list
      * Negative, Positive, Uncertainty, Litigious, Constraining, Superfluous, Modal
    * Sentiment Bag of Words (Count for each ticker, sentiment)
      * ```
        sklearn.feature_extraction.text.CountVectorizer(analyzer='word', vocabulary=sentiment)
        X = vectorizer.fit_transform(docs)
        features = vectorizer.get_feature_names()
        ```
    * Jaccard Similarity
      * ```sklearn.metrics.jaccard_similarity_score(u, v)```
      * Get the similarity between neighboring bag of words
    * TF-IDF
      * ```sklearn.feature_extraction.text.TfidfVectorizer(analyzer='word', vocabulary=sentiments)```
    * Cosine Similarity
      * ```sklearn.metrics.pairwise.cosine_similarity(u, v)```
      * Get the similarity between neighboring IFIDF vectors
4. Evaluate Alpha Factors
    * Use yearly pricing to match with 10K frequency of annual production
    * Turn the sentiment dictionary into a dataframe so that alphalens can read
    * Alphalens Format
      * ```data = alphalens.utils.get_clean_factor_and_forward_return(df.stack(), pricing, quantiles=5, bins=None, period=[1])```
    * Alphalens Format with Unix Timestamp
      * ```{factor: data.set_index(pd.MultiIndex.from_tuples([(x.timestamp(), y) for x, y in data.index.values], names=['date', 'asset'])) for factor, data in factor_data.items()}```
    * Factor Returns
      * ```alphalens.performance.factor_returns(data)```
      * Should move up and to the right
    * Basis Points Per Day per Quantile
      * ```alphalens.performance.mean_return_by_quantile(data)```
      * Should be monotonic in quantiles
    * Turnover Analysis
      * Factor Rank Autocorrelation (FRA) to measure the stability without full backtest
      * ```alphalens.factor_rank_autocorrelation(data)```
    * Sharpe Ratio of the Alphas
      * Should be 1 or higher
      * ```np.sqrt(252)*factor_returns.mean() / factor_returns.std()```
#### 6. Advanced NLP with Deep Leaning - Analizing Stock Sentiment from Twits (requiring GPU)
0. import json, nltk, os, random, re, torch, torch.nn, torch.optim, torch.nn.functional, numpy
1. Import Twits
    * json.load()
2. Preprocessing the Data
    * Pre-Processing
      * ```
        nltk.download('wordnet')
        nltk.download('stopwords')
        text = message.lower()
        text = re.sub('https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*', ' ', text)
        text = re.sub('\$[a-zA-Z0-9]*', ' ', text)
        text = re.sub('\@[a-zA-Z0-9]*', ' ', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        tokens = text.split()
        wnl = nltk.stem.WordNetLemmatizer()
        tokens = [wnl.lemmatize(wnl.lemmatize(word, 'n'), 'v') for word in tokens]
    * Bag of Words
      * ```bow = sorted(Counter(all_words), key=counts.get, reverse=True)
    * Remove most common words such as 'the, 'and' by high_cutoff=20, rare words by low_cutoff=1e-6
    * Create Dictionaries
      * ```
        vocab = {word: ii for ii, word in enumarate(filtered_words, 1)}
        id2vodab = {v: k for k, v in vocab.items()}
        filtered = [[word for word in message if word in vocab] for message in tokenized]
        ```
    * Balancing the classes
      * 50% is neutral --> make it 20% by dropping some neutral twits
      * Remove messages with zero length
3. Neural Network
    * Embed -> RNN -> Dense -> Softmax
    * Text Classifier
      * ```
        class TextClassifier(nn.Module):
            def __init__(self, vocab_size, embed_size, lstm_size, output_size, lstm_layers=1, dropout=0.1):
                super().__init__()
                self.vocab_size = vocab_size
                self.embed_size = embed_size
                self.lstm_size = lstm_size
                self.output_size = output_size
                self.lstm_layers = lstm_layers
                self.dropout = dropout

                self.embedding = nn.Embedding(vodab_size, embed_size)
                self.lsfm = nn.LSTM(embed_size, lstm_size, lstm_layers, dropout=dropout, batch_first=False)
                self.dropout = nn.Dropout(-0.2)
                self.fc = nn.Linear(lstm_size, output_size)
                self.softmax = nn.LogSoftmax(dim=1)
            def init_hidden(self, batch_size):
                weight = next(self.parameters()).data
                hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(),
                          weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())
                return hidden
            def forward(self, nn_input, hidden_state)
                batch_size = nn_input.size(0)
                nn_input = nn_input.long()
                embeds = self.embedding(nn_input)
                lstm_out, hidden_state = self.lstm(embeds, hidden_state)
                lstm_out = lstm_out[-1,:,:] # Stack up LSMT Outputs
                out = self.dropout(lstm_out)
                out = self.fc(out)
                logps = self.softmax(out)
                return logps, hidden_state
        ```
4. Training
    * DataLoaders and Batching
      * Input Tensor shape should be (sequence_length, batch_size)
      * Left pad with zeros if a message has less tokens than sequence_length.
      * If a message has more token than sequence_length, keep the first sequence_length tokens
      * Build a DataLoader as a generator 
        ```
        def dataloader(): 
            yield batch, label_tensor # both variables are torch.tensor()
        ```
    * Training and Validation
      * Split data to training set and validation set, then check the model
        ```
        text_batch, labels = next(iter(dataloader(train_features, train_labels, sequence_length=20, batch_size=64)))
        model = TextClassifier(len(vocab)+1, 200, 128, 5, dropout=0.)
        hidden = model.init_hidden(64)
        logps, hidden = model.forward(text_batch, hidden)
        print(logps)
        ```
      * Model
        ```
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TextClassifier(len(vocab)+1, 1024, 512, 5, lstm_layers=2, dropout=0.2)
        model.embedding.weight.data.uniform_(-1,1)
        model.to(device)
        ```
      * Train!
        ```
        epochs = 3
        batch_size = 1024
        learning_rate = 0.001
        clip = 5
        print_every = 100
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.train()
        for epoch in range(epochs):
            print ('Starting epoch {}'.format(epoch + 1))
            hidden = model.init_hidden(batch_size)
            steps = 0
            for text_batch, labels in dataloader(train_features, train_labels, batch_size=batch_size, sequence_length=20, shuffle=True):
                steps += 1
                if text_batch.size(1) != batch_size:
                    break
                hidden = tuple([each.data for each in hidden])
                text_batch, labels = text_batch.to(device), labels.to(device)
                for each in hidden:
                    each.to(device)
                model.zero_grad()
                output, hidden = model(text_batch, hidden)
                loss = criterion(output, labels)
                loss.backwards()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step() # Optimize
                if steps % print_every == 0:
                    model.eval()
                    valid_losses = []
                    accuracy = []
                    valid_hidden = model.init_hidden(batch_size)
                    for text_batch, labels in dataloader(valid_features, valid_labels, batch_size=batch_size, sequence_length=20, shuffle=False):
                        if text_batch.size(1) != batch_size:
                            break
                        valid_hidden = tuple([each.data for each in valid_hidden])
                        text_batch, lables = text_batch.to(device), labels.to(device)
                        for each in valid_hidden:
                            each.to(device)
                        valid_output, valid_hidden = model(text_batch, valid_hidden)
                        valid_loss = criterion(valid_output.squeeze(), labels)
                        valid_losses.append(valid_loss.item())
                        ps = torch.exp(valid_output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy.append(torch.mean(equals.type(torch.FloatTensor)).item())
                    model.train()
                    print("Epoch: {}/{}...".format(epoch+1, epochs),
                          "Step: {}...".format(steps),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(valid_losses)),
                          "Accuracy: {:.6f}".format(np.mean(accuracy)))
        ```
5. Making Predictions
    * preprocess, filter non-vocab words, convert words to ids, add a batch dimention (```torch.tensor(tokens).view(-1,1))```
        ```
        hidden = model.init_hidden(1)
        logps, _ = model.forward(text_input, hidden)
        pred = torch.exp(logps)
        ```
6. Testing
#### 7. Combining Multiple Signals for Enhanced Alpha
0. import numpy, pandas, tqdm, matplotlib.pyplot
1. Data Pipeline
    * ```zipline.data.bundles``` - register, load
    * ```zipline.pipeline.Pipeline```
    * ```universe = zipline.pipeline.AverageDollarVolume```
    * ```zipline.utils.calendar.get_calendar('NYSE')```
    * ```zipline.pipeline.loaders.USEquityPricingLoader```
    * ```engine = zipline.pipeline.engine.SimplePipelineEngine```
    * ```zipline.data.data_portal.DataPortal```
2. Alpha Factors
    * Momentum 1 Year Factor
      * ```zipline.pipeline.factors.Returns().demean(groupby=Sector).rank().zscore()```
    * Mean Reversion 5 Day Sector Neutral Smoothed Factor
      * ```unsmoothed = -Returns().demean(groupby=Sector).rank().zscore()```
      * ```smoothed = zipline.pipeline.factors.SimpleMovingAverage(unsmoothed).rank().zscore()```
    * Overnight Sentiment Smoothed Factor
      * CTO(Returns), TrainingOvernightReturns(Returns)
    * Combine the three factors by pipeline.add()
3. Features and Labels
    * Universal Quant Features
      * Stock Volatility 20d, 120d: ```pipeline.add(zipline.pipeline.factors.AnnualizedVolatility)```
      * Stock Dollar Volume 20d, 120d: ```pipeline.add(zipline.pipeline.factors.AverageDollarVolume)```
      * Sector
    * Regime Features
      * High and low volatility 20d, 120d: ```MarketVolatility(CustomFactor)```
      * High and low dispersion 20d, 120d: ```SimpleMovingAverage(MarketDispersion(CustomFactor))```
    * Target
      * 1 Week Return, Quantized: ```pipeline.add(Returns().quantiles(2)), pipeline.add(Returns().quantiles(25))```
    * engine.run_pipeline()
    * Date Feature
      * January, December, Weekday, Quarter, Qtr-Year, Month End, Month Start, Qtr Start, Qtr End
    * One-hot encode Sector
    * Shift Target
    * IID Check (Independent and Identically Distributed)
      * Check rolling autocorelation between 1d to 5d shifted target using ```scipy.stats.speamanr```
    * Train/Validation/Test Splits
4. Random Forests
    * Visualize a Simple Tree
      * clf = ```sklearn.tree.DecisionTreeClassifier()```
      * Graph: ```IPython.display.display```
      * Rank features by importance ```clf.feature_importances_```
    * Random Forest
      * clf = ```sklearn.ensemble.RandomForestClassifier()```
      * Scores: ```clf.score(), clf.oob_score_, clf.feature_importances_```
    * Model Results
      * Sharpe Ratios ```sqrt(252)*factor_returns.mean()/factor_returns.std()```
      * Factor Returns ```alphalens.performance.factor_returns()```
      * Factor Rank Autocorelation ```alphalens.performance.factor_rank_autocorrelation()```
      * Scores: ```clf.predict_proba()```
    * Check the above for Training Data and Validation Data
5. Overlapping Samples
    * Option 1) Drop Overlapping Samples
    * Option 2) Use ```sklearn.ensemble.BaggingClassifier```'s max_samples with ```base_clf = DecisionTreeClassifier()```
    * Option 3) Build an ensemble of non-overlapping trees
      * ```sklearn.ensemble.VotingClassifier```
      * ```sklearn.base.clone```
      * ```sklearn.preprocessing.LavelEncoder```
      * ```sklearn.utils.Bunch```
6. Final Model
    * Re-Training Model using Training Set + Validation Set
#### 8. Simulating Trades with Historical Data - Backtesting
0. 
1. Load Price, Covariance and Factor Exposure from Barra - data.update(pickle.load())
2. Shift daily returns by 2 days
3. Winsorize
    * np.where(x <= a,a, np.where(x >= b, b, x)) and Density plot
4. Factor Exposures and Factor Returns
    * model = ols (Ordinary Least Squares)
    * universe = Market Cap > 1e9, Winsorize
    * variable: dependent = Daily Return, independent = Factor Exposures
    * estimation: Factor Returns
5. Choose 4 Alpha Factors
    * 1 Day Reversal, Earnings Yield, Value, Sentiment
6. Merge Previous Portfolio Holdings and Add h.opt.previous with 0
7. Convert all NaN to 0, and median for 0 Specific Risk
8. Build Universe - (df['IssuerMarketCap'] >= 1e9) | (abs(df['h.opt.previous']) > 0.0)
9.  Set Risk Factors (B)
    * All Factors - Alpha Factors
    * patsy.dmatrices to one-hot encode categories
10. Calculate Specific Variance
    * (Specific Risk * 0.01)**2
11. Build Factor Covariance Matrix
    * Take off diagonal 
12. Estimate Transaction Cost
    * Lambda
13. Combine the four Alpha Factors
    * sum(B_Alpha(Design Matrix)) * 1e-4
14. Define Objective Function
    * $$
f(\mathbf{h}) = \frac{1}{2}\kappa \mathbf{h}_t^T\mathbf{Q}^T\mathbf{Q}\mathbf{h}_t + \frac{1}{2} \kappa \mathbf{h}_t^T \mathbf{S} \mathbf{h}_t - \mathbf{\alpha}^T \mathbf{h}_t + (\mathbf{h}_{t} - \mathbf{h}_{t-1})^T \mathbf{\Lambda} (\mathbf{h}_{t} - \mathbf{h}_{t-1})
$$
15. Define Gradient of Objective Function
    * $$
f'(\mathbf{h}) = \frac{1}{2}\kappa (2\mathbf{Q}^T\mathbf{Qh}) + \frac{1}{2}\kappa (2\mathbf{Sh}) - \mathbf{\alpha} + 2(\mathbf{h}_{t} - \mathbf{h}_{t-1}) \mathbf{\Lambda}
$$
16. Optimize Portfolio
    * h = scipy.optimize.fmin_l_bfgs_b(func, initial_guess, func_gradient)
17. Calculate Risk Exposure
    * B.T * h
18. Calculate Alpha Exposure
    * B_Alpha.T * h
19. Calculate Transaction Cost
    * $$
tcost = \sum_i^{N} \lambda_{i} (h_{i,t} - h_{i,t-1})^2
$$
20. Build Tradelist
    * h - h_previous
21. Save optimal holdings as previous optimal holdings
    * h_previous = h
22. Run the Backtest
    * Loop #6 to #21 for all the dates
23. PnL Attrribution
    * $$
{PnL}_{alpha}= f \times b_{alpha}
$$
    * $$
{PnL}_{risk} = f \times b_{risk}
$$

24. Build Portfolio Characteristics
    * calculate the sum of long positions, short positions, net positions, gross market value, and amount of dollars traded.